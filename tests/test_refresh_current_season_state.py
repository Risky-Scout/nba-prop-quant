"""Stage 2 safe current-season refresh guards.

Every test is offline: `subprocess.run` and socket creation are blocked for the
whole module, and the preflight/ingestion boundaries are substituted with
synthetic parquet fixtures.
"""

from __future__ import annotations

import hashlib
import importlib.util
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "ops/refresh_current_season_state.py"

SEASON = 2026
SLATE_DATE = "2026-10-24"


def load_refresher():
    spec = importlib.util.spec_from_file_location(
        "refresh_current_season_state_under_test",
        MODULE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the refresh wrapper")

    module = importlib.util.module_from_spec(spec)

    # dataclasses resolve annotations through sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


refresher = load_refresher()


# ---------------------------------------------------------------------------
# offline guards
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail loudly if any test tries to spawn a process or open a socket."""

    calls = SimpleNamespace(subprocess=0, socket=0)

    def blocked_subprocess(*args, **kwargs):
        calls.subprocess += 1
        raise AssertionError(
            "a test attempted to spawn a subprocess (live API access)"
        )

    def blocked_socket(*args, **kwargs):
        calls.socket += 1
        raise AssertionError("a test attempted to open a network socket")

    monkeypatch.setattr(refresher.subprocess, "run", blocked_subprocess)
    monkeypatch.setattr(socket, "create_connection", blocked_socket)
    monkeypatch.setattr(socket.socket, "connect", blocked_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked_socket)

    yield calls

    assert calls.subprocess == 0
    assert calls.socket == 0


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------


def stats_frame(entries, drop=()):
    rows = []

    for index, (game_id, player_id, date) in enumerate(entries, start=1):
        row = {column: float(index) for column in refresher.REQUIRED_STATS_COLUMNS}
        row.update(
            {
                "player_id": player_id,
                "game_id": game_id,
                "date": pd.Timestamp(date),
                "season": SEASON,
                "team_id": 10,
                "home_team_id": 10,
                "visitor_team_id": 20,
            }
        )
        rows.append(row)

    frame = pd.DataFrame(rows, columns=list(refresher.REQUIRED_STATS_COLUMNS))

    if drop:
        frame = frame.drop(columns=list(drop))

    return frame


def advanced_frame(entries, drop=()):
    rows = [
        {
            "player_id": player_id,
            "game_id": game_id,
            "date": pd.Timestamp(date),
            "pace": 100.0,
        }
        for game_id, player_id, date in entries
    ]

    frame = pd.DataFrame(rows, columns=["player_id", "game_id", "date", "pace"])

    if drop:
        frame = frame.drop(columns=list(drop))

    return frame


def games_frame(entries, drop=()):
    rows = [
        {
            "id": game_id,
            "date": pd.Timestamp(date),
            "season": SEASON,
            "home_team_id": 10,
            "visitor_team_id": 20,
            "status": status,
        }
        for game_id, date, status in entries
    ]

    frame = pd.DataFrame(
        rows,
        columns=["id", "date", "season", "home_team_id", "visitor_team_id", "status"],
    )

    if drop:
        frame = frame.drop(columns=list(drop))

    return frame


PREVIOUS_STATS_ENTRIES = [
    (1, 100, "2026-10-21"),
    (1, 101, "2026-10-21"),
    (2, 100, "2026-10-22"),
    (2, 102, "2026-10-22"),
]

STAGED_STATS_ENTRIES = PREVIOUS_STATS_ENTRIES + [
    (4, 100, "2026-10-23"),
    (4, 103, "2026-10-23"),
]

PREVIOUS_GAMES_ENTRIES = [
    (1, "2026-10-21", "Final"),
    (2, "2026-10-22", "Final"),
    (3, "2026-10-25", "1:00 PM ET"),
]

STAGED_GAMES_ENTRIES = [
    (1, "2026-10-21", "Final"),
    (2, "2026-10-22", "Final"),
    (3, "2026-10-25", "1:00 PM ET"),
    (4, "2026-10-23", "Final"),
]

PRIOR_SEASON = 2025


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture
def live(tmp_path):
    """A populated live data root with 2026 state plus a frozen 2025 season."""

    data_dir = tmp_path / "data"

    write_parquet(
        stats_frame(PREVIOUS_STATS_ENTRIES),
        refresher.stats_path_for(data_dir, SEASON),
    )
    write_parquet(
        games_frame(PREVIOUS_GAMES_ENTRIES),
        refresher.games_path_for(data_dir, SEASON),
    )
    write_parquet(
        advanced_frame(PREVIOUS_STATS_ENTRIES),
        refresher.advanced_path_for(data_dir, SEASON),
    )

    write_parquet(
        stats_frame([(900, 1, "2025-11-01")]),
        refresher.stats_path_for(data_dir, PRIOR_SEASON),
    )
    write_parquet(
        games_frame([(900, "2025-11-01", "Final")]),
        refresher.games_path_for(data_dir, PRIOR_SEASON),
    )
    write_parquet(
        advanced_frame([(900, 1, "2025-11-01")]),
        refresher.advanced_path_for(data_dir, PRIOR_SEASON),
    )

    return SimpleNamespace(
        root=tmp_path,
        data_dir=data_dir,
        stats=refresher.stats_path_for(data_dir, SEASON),
        games=refresher.games_path_for(data_dir, SEASON),
        advanced=refresher.advanced_path_for(data_dir, SEASON),
        prior_stats=refresher.stats_path_for(data_dir, PRIOR_SEASON),
        prior_games=refresher.games_path_for(data_dir, PRIOR_SEASON),
        prior_advanced=refresher.advanced_path_for(data_dir, PRIOR_SEASON),
    )


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_tree_hashes(data_dir: Path) -> dict[str, str]:
    root = data_dir / "raw"

    return {
        str(path.relative_to(root)): sha256_bytes(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def shadow_directories(root: Path) -> list[Path]:
    return sorted(root.glob(".nba_prop_refresh_shadow_*"))


def preflight_stub(
    *,
    returncode: int = 0,
    action: str = "READY_TO_REFRESH",
    payload: dict | None = None,
):
    resolved = (
        payload
        if payload is not None
        else {
            "season": SEASON,
            "target_date": SLATE_DATE,
            "require_advanced": True,
            "latest_final_date": "2026-10-23",
            "latest_final_game_ids": [4],
        }
    )

    result = refresher.PreflightResult(
        returncode=returncode,
        action=action,
        payload=resolved,
        stdout="",
        stderr="",
    )

    def run_preflight(**kwargs):
        return result

    return run_preflight


def ingest_stub(
    *,
    stats=None,
    games=None,
    advanced=None,
    extra_advanced_path: str | None = None,
    corrupt: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
):
    """Substitute for the real ingester: writes synthetic shadow parquet."""

    frames = {
        "stats": stats if stats is not None else stats_frame(STAGED_STATS_ENTRIES),
        "games": games if games is not None else games_frame(STAGED_GAMES_ENTRIES),
        "advanced": (
            advanced
            if advanced is not None
            else advanced_frame(STAGED_STATS_ENTRIES)
        ),
    }

    def run_shadow_ingest(*, shadow_data_dir, season, python_executable=None):
        paths = {
            "stats": refresher.stats_path_for(shadow_data_dir, season),
            "games": refresher.games_path_for(shadow_data_dir, season),
            "advanced": refresher.advanced_path_for(shadow_data_dir, season),
        }

        for name, path in paths.items():
            if name in missing:
                continue

            if name in corrupt:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"this is not a parquet file")
                continue

            write_parquet(frames[name], path)

        if extra_advanced_path is not None:
            extra = refresher.advanced_root_for(shadow_data_dir) / extra_advanced_path
            write_parquet(frames["advanced"], extra)

    return run_shadow_ingest


def install(monkeypatch, *, preflight=None, ingest=None):
    monkeypatch.setattr(
        refresher,
        "run_preflight",
        preflight if preflight is not None else preflight_stub(),
    )
    monkeypatch.setattr(
        refresher,
        "run_shadow_ingest",
        ingest if ingest is not None else ingest_stub(),
    )


def run(live, *, slate_date: str = SLATE_DATE) -> int:
    return refresher.main(
        [
            "--slate-date",
            slate_date,
            "--season",
            str(SEASON),
            "--data-dir",
            str(live.data_dir),
        ]
    )


def assert_live_unchanged(live, before: dict[str, str]) -> None:
    assert raw_tree_hashes(live.data_dir) == before


def run_expect_refusal(
    live,
    capsys,
    expected: str,
    *,
    code: int | None = None,
    slate_date: str = SLATE_DATE,
) -> None:
    """Assert the refresh was refused for the specific expected reason."""

    before = raw_tree_hashes(live.data_dir)
    result = run(live, slate_date=slate_date)
    captured = capsys.readouterr()

    assert expected in captured.err, captured.err
    assert result == (
        refresher.EXIT_VALIDATION_FAILED if code is None else code
    )
    assert_live_unchanged(live, before)
    assert shadow_directories(live.root) == []


# ---------------------------------------------------------------------------
# interface reuse: real preflight output, real ingester command line
# ---------------------------------------------------------------------------


def load_preflight_module():
    spec = importlib.util.spec_from_file_location(
        "preflight_current_season_refresh_under_test",
        PROJECT_ROOT / "ops/preflight_current_season_refresh.py",
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the preflight script")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


@pytest.mark.parametrize(
    ("action", "code"),
    [
        ("READY_TO_REFRESH", 0),
        ("SKIP_REFRESH", 10),
        ("HOLD_ADVANCED_NOT_READY", 31),
    ],
)
def test_parses_the_real_preflight_output_contract(capsys, action, code):
    """The wrapper reads the preflight's own emitted status and JSON payload."""

    preflight = load_preflight_module()

    payload = {
        "season": SEASON,
        "target_date": SLATE_DATE,
        "require_advanced": True,
        "latest_final_game_ids": [4, 5],
    }

    with pytest.raises(SystemExit) as excinfo:
        preflight.finish(
            action=action,
            code=code,
            message="Human readable explanation with no braces.",
            payload=payload,
        )

    assert excinfo.value.code == code

    parsed_action, parsed_payload = refresher.parse_preflight_output(
        capsys.readouterr().out
    )

    assert parsed_action == action
    assert parsed_payload == payload

    result = refresher.PreflightResult(
        returncode=code,
        action=parsed_action,
        payload=parsed_payload,
    )

    assert result.ready is (action == refresher.READY_ACTION)
    assert result.final_game_ids == {4, 5}
    assert result.advanced_required is True


def test_ingester_is_redirected_by_environment_only(tmp_path, monkeypatch):
    """No live path reaches the ingester; the shadow root travels in the env."""

    recorded = {}

    def record(command, **kwargs):
        recorded["command"] = list(command)
        recorded["env"] = dict(kwargs.get("env") or {})
        recorded["cwd"] = kwargs.get("cwd")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(refresher.subprocess, "run", record)

    shadow = tmp_path / "shadow" / "data"

    refresher.run_shadow_ingest(shadow_data_dir=shadow, season=SEASON)

    assert recorded["command"] == [
        sys.executable,
        str(PROJECT_ROOT / "scripts/01_ingest_history_resume.py"),
        "--start-season",
        str(SEASON),
        "--end-season",
        str(SEASON),
        "--include-advanced",
        "--skip-players",
        "--force",
    ]
    assert recorded["env"]["NBA_PROP_DATA_DIR"] == str(shadow)
    assert recorded["cwd"] == str(PROJECT_ROOT)


def test_ingester_flags_are_supported_by_the_existing_script():
    """Guard against inventing flags scripts/01_ingest_history_resume.py lacks."""

    source = (PROJECT_ROOT / "scripts/01_ingest_history_resume.py").read_text(
        encoding="utf-8"
    )

    for flag in (
        "--start-season",
        "--end-season",
        "--include-advanced",
        "--skip-players",
        "--force",
    ):
        assert f'"{flag}"' in source

    preflight_source = (
        PROJECT_ROOT / "ops/preflight_current_season_refresh.py"
    ).read_text(encoding="utf-8")

    for flag in ("--season", "--target-date", "--opening-day", "--require-advanced"):
        assert f'"{flag}"' in preflight_source


# ---------------------------------------------------------------------------
# 1-3: preflight gating
# ---------------------------------------------------------------------------


def test_preflight_ready_allows_staging_path_to_proceed(live, monkeypatch):
    staged: dict[str, Path] = {}

    def recording_ingest(*, shadow_data_dir, season, python_executable=None):
        staged["shadow"] = Path(shadow_data_dir)
        ingest_stub()(
            shadow_data_dir=shadow_data_dir,
            season=season,
            python_executable=python_executable,
        )

    install(monkeypatch, ingest=recording_ingest)

    assert run(live) == refresher.EXIT_OK
    assert staged["shadow"].is_absolute()
    assert live.data_dir not in staged["shadow"].parents


def test_preflight_skip_performs_no_live_changes(live, monkeypatch):
    before = raw_tree_hashes(live.data_dir)

    def forbidden_ingest(**kwargs):
        raise AssertionError("ingestion must not run when preflight is not ready")

    install(
        monkeypatch,
        preflight=preflight_stub(
            returncode=10,
            action="SKIP_REFRESH",
            payload={"season": SEASON},
        ),
        ingest=forbidden_ingest,
    )

    assert run(live) == 10
    assert_live_unchanged(live, before)
    assert shadow_directories(live.root) == []


@pytest.mark.parametrize(
    ("action", "code"),
    [
        ("HOLD_STANDARD_NOT_READY", 30),
        ("HOLD_ADVANCED_NOT_READY", 31),
        ("PRESEASON_BLOCK", 20),
    ],
)
def test_preflight_hold_performs_no_live_changes(live, monkeypatch, action, code):
    before = raw_tree_hashes(live.data_dir)

    def forbidden_ingest(**kwargs):
        raise AssertionError("ingestion must not run when preflight holds")

    install(
        monkeypatch,
        preflight=preflight_stub(
            returncode=code,
            action=action,
            payload={"season": SEASON},
        ),
        ingest=forbidden_ingest,
    )

    assert run(live) == code
    assert_live_unchanged(live, before)
    assert shadow_directories(live.root) == []


# ---------------------------------------------------------------------------
# 4: single-writer lock
# ---------------------------------------------------------------------------


def test_concurrent_lock_prevents_second_writer(live, monkeypatch):
    before = raw_tree_hashes(live.data_dir)

    def forbidden_preflight(**kwargs):
        raise AssertionError("preflight must not run without the lock")

    install(
        monkeypatch,
        preflight=forbidden_preflight,
        ingest=ingest_stub(),
    )

    lock_path = refresher.lock_path_for(live.data_dir.resolve())

    with refresher.refresh_lock(lock_path):
        assert run(live) == refresher.EXIT_ALREADY_LOCKED

    assert_live_unchanged(live, before)

    install(monkeypatch)
    assert run(live) == refresher.EXIT_OK


# ---------------------------------------------------------------------------
# 5-9: shape, column and readability validation
# ---------------------------------------------------------------------------


def test_empty_staged_stats_fails_and_preserves_live_files(
    live,
    monkeypatch,
    capsys,
):
    install(monkeypatch, ingest=ingest_stub(stats=stats_frame([])))

    run_expect_refusal(live, capsys, "staged stats.parquet: empty")


def test_empty_staged_advanced_fails_and_preserves_live_files(
    live,
    monkeypatch,
    capsys,
):
    install(monkeypatch, ingest=ingest_stub(advanced=advanced_frame([])))

    run_expect_refusal(live, capsys, "staged advanced.parquet: empty")


@pytest.mark.parametrize("column", sorted(refresher.REQUIRED_STATS_COLUMNS))
def test_missing_required_standard_column_fails(live, monkeypatch, capsys, column):
    install(
        monkeypatch,
        ingest=ingest_stub(
            stats=stats_frame(STAGED_STATS_ENTRIES, drop=(column,)),
        ),
    )

    run_expect_refusal(
        live,
        capsys,
        f"staged stats.parquet: missing inference-required columns ['{column}']",
    )


def test_missing_advanced_date_fails(live, monkeypatch, capsys):
    install(
        monkeypatch,
        ingest=ingest_stub(
            advanced=advanced_frame(STAGED_STATS_ENTRIES, drop=("date",)),
        ),
    )

    run_expect_refusal(
        live,
        capsys,
        "staged advanced.parquet: missing inference-required columns ['date']",
    )


@pytest.mark.parametrize("name", ["stats", "advanced", "games"])
def test_corrupt_staged_parquet_fails(live, monkeypatch, capsys, name):
    install(monkeypatch, ingest=ingest_stub(corrupt=(name,)))

    run_expect_refusal(
        live,
        capsys,
        f"staged {name}.parquet: unreadable parquet",
    )


@pytest.mark.parametrize("name", ["stats", "advanced", "games"])
def test_missing_staged_file_fails(live, monkeypatch, capsys, name):
    install(monkeypatch, ingest=ingest_stub(missing=(name,)))

    run_expect_refusal(
        live,
        capsys,
        f"staged {name}.parquet: missing file",
    )


def test_missing_games_operational_column_fails(live, monkeypatch, capsys):
    install(
        monkeypatch,
        ingest=ingest_stub(games=games_frame(STAGED_GAMES_ENTRIES, drop=("id",))),
    )

    run_expect_refusal(
        live,
        capsys,
        "staged games.parquet: missing operational columns ['id']",
    )


def test_games_without_final_status_column_fails(live, monkeypatch, capsys):
    install(
        monkeypatch,
        ingest=ingest_stub(
            games=games_frame(STAGED_GAMES_ENTRIES, drop=("status",)),
        ),
    )

    run_expect_refusal(
        live,
        capsys,
        "grading cannot confirm finality",
    )


# ---------------------------------------------------------------------------
# 10-13: strict leakage guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("date", ["2026-10-24", "2026-10-25"])
def test_staged_stats_date_not_before_slate_fails(live, monkeypatch, capsys, date):
    entries = STAGED_STATS_ENTRIES + [(5, 104, date)]

    install(
        monkeypatch,
        ingest=ingest_stub(
            stats=stats_frame(entries),
            games=games_frame(STAGED_GAMES_ENTRIES + [(5, date, "Final")]),
        ),
    )

    run_expect_refusal(
        live,
        capsys,
        f"staged stats.parquet: leakage guard failed; max date {date}",
    )


@pytest.mark.parametrize("date", ["2026-10-24", "2026-10-25"])
def test_staged_advanced_date_not_before_slate_fails(live, monkeypatch, capsys, date):
    entries = STAGED_STATS_ENTRIES + [(5, 104, date)]

    install(
        monkeypatch,
        ingest=ingest_stub(
            advanced=advanced_frame(entries),
            games=games_frame(STAGED_GAMES_ENTRIES + [(5, date, "Final")]),
        ),
    )

    run_expect_refusal(
        live,
        capsys,
        f"staged advanced.parquet: leakage guard failed; max date {date}",
    )


def test_staged_dates_strictly_before_slate_pass(live, monkeypatch):
    install(monkeypatch)

    assert run(live) == refresher.EXIT_OK


# ---------------------------------------------------------------------------
# 14-18: regression guards
# ---------------------------------------------------------------------------


def test_stats_row_count_regression_fails(live, monkeypatch, capsys):
    install(
        monkeypatch,
        ingest=ingest_stub(stats=stats_frame(PREVIOUS_STATS_ENTRIES[:2])),
    )

    run_expect_refusal(live, capsys, "stats: row-count regression 4 -> 2")


def test_stats_distinct_game_regression_fails(live, monkeypatch, capsys):
    # More rows than before, but only one distinct game.
    entries = [(1, player, "2026-10-21") for player in range(100, 110)]

    install(monkeypatch, ingest=ingest_stub(stats=stats_frame(entries)))

    run_expect_refusal(live, capsys, "stats: distinct-game regression 2 -> 1")


def test_advanced_row_count_regression_fails(live, monkeypatch, capsys):
    install(
        monkeypatch,
        ingest=ingest_stub(advanced=advanced_frame(PREVIOUS_STATS_ENTRIES[:2])),
    )

    run_expect_refusal(live, capsys, "advanced: row-count regression 4 -> 2")


def test_advanced_distinct_game_regression_fails(live, monkeypatch, capsys):
    entries = [(1, player, "2026-10-21") for player in range(100, 110)]

    install(monkeypatch, ingest=ingest_stub(advanced=advanced_frame(entries)))

    run_expect_refusal(live, capsys, "advanced: distinct-game regression 2 -> 1")


def test_games_distinct_game_regression_fails(live, monkeypatch, capsys):
    install(
        monkeypatch,
        ingest=ingest_stub(
            stats=stats_frame(PREVIOUS_STATS_ENTRIES),
            advanced=advanced_frame(PREVIOUS_STATS_ENTRIES),
            games=games_frame(PREVIOUS_GAMES_ENTRIES[:2]),
        ),
    )

    run_expect_refusal(live, capsys, "games: distinct-game regression 3 -> 2")


# ---------------------------------------------------------------------------
# 19-20: advanced layout and referential integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    ["season=2026/extra.parquet", "stray.parquet", "season=2026/nested/x.parquet"],
)
def test_unexpected_staged_advanced_parquet_fails_without_deleting_it(
    live,
    monkeypatch,
    capsys,
    extra,
):
    observed: dict[str, Path] = {}

    def recording_ingest(*, shadow_data_dir, season, python_executable=None):
        observed["extra"] = refresher.advanced_root_for(Path(shadow_data_dir)) / extra
        ingest_stub(extra_advanced_path=extra)(
            shadow_data_dir=shadow_data_dir,
            season=season,
        )
        assert observed["extra"].exists()

    install(monkeypatch, ingest=recording_ingest)

    run_expect_refusal(
        live,
        capsys,
        "staged advanced tree: unexpected parquet file(s)",
    )


def test_unexpected_live_advanced_parquet_fails_without_deleting_it(
    live,
    monkeypatch,
    capsys,
):
    stray = refresher.advanced_root_for(live.data_dir) / "stray.parquet"
    write_parquet(advanced_frame(PREVIOUS_STATS_ENTRIES), stray)

    def forbidden_ingest(**kwargs):
        raise AssertionError("ingestion must not run with a poisoned live tree")

    install(monkeypatch, ingest=forbidden_ingest)

    run_expect_refusal(
        live,
        capsys,
        "live advanced tree: unexpected parquet file(s)",
    )

    assert stray.exists()


@pytest.mark.parametrize("store", ["stats", "advanced"])
def test_game_ids_absent_from_games_fail(live, monkeypatch, capsys, store):
    orphan = STAGED_STATS_ENTRIES + [(77, 105, "2026-10-23")]

    kwargs = (
        {"stats": stats_frame(orphan)}
        if store == "stats"
        else {"advanced": advanced_frame(orphan)}
    )

    install(monkeypatch, ingest=ingest_stub(**kwargs))

    run_expect_refusal(
        live,
        capsys,
        f"staged {store}.parquet references game ids absent from staged "
        "games.parquet: [77]",
    )


def test_preflight_final_games_missing_from_staged_state_fails(
    live,
    monkeypatch,
    capsys,
):
    install(
        monkeypatch,
        preflight=preflight_stub(
            payload={
                "season": SEASON,
                "require_advanced": True,
                "latest_final_game_ids": [4, 6],
            }
        ),
    )

    run_expect_refusal(
        live,
        capsys,
        "staged stats.parquet is missing games the preflight declared ready: [6]",
    )


# ---------------------------------------------------------------------------
# 21-25: successful refresh semantics
# ---------------------------------------------------------------------------


@pytest.fixture
def successful_refresh(live, monkeypatch):
    staged_hashes: dict[str, str] = {}

    def recording_ingest(*, shadow_data_dir, season, python_executable=None):
        ingest_stub()(shadow_data_dir=shadow_data_dir, season=season)

        staged_hashes["stats"] = sha256_bytes(
            refresher.stats_path_for(shadow_data_dir, season)
        )
        staged_hashes["games"] = sha256_bytes(
            refresher.games_path_for(shadow_data_dir, season)
        )
        staged_hashes["advanced"] = sha256_bytes(
            refresher.advanced_path_for(shadow_data_dir, season)
        )

    install(monkeypatch, ingest=recording_ingest)

    before = raw_tree_hashes(live.data_dir)
    code = run(live)

    return SimpleNamespace(
        live=live,
        code=code,
        before=before,
        after=raw_tree_hashes(live.data_dir),
        staged_hashes=staged_hashes,
    )


def test_successful_refresh_changes_exactly_three_live_files(successful_refresh):
    result = successful_refresh

    assert result.code == refresher.EXIT_OK
    assert set(result.after) == set(result.before)

    changed = {
        name
        for name in result.after
        if result.after[name] != result.before[name]
    }

    assert changed == {
        f"seasons/season={SEASON}/stats.parquet",
        f"seasons/season={SEASON}/games.parquet",
        f"advanced/season={SEASON}/advanced.parquet",
    }


def test_successful_refresh_leaves_prior_seasons_byte_identical(successful_refresh):
    result = successful_refresh

    for name in (
        f"seasons/season={PRIOR_SEASON}/stats.parquet",
        f"seasons/season={PRIOR_SEASON}/games.parquet",
        f"advanced/season={PRIOR_SEASON}/advanced.parquet",
    ):
        assert result.after[name] == result.before[name]


def test_successful_refresh_preserves_exact_staged_stats_bytes(successful_refresh):
    assert (
        sha256_bytes(successful_refresh.live.stats)
        == successful_refresh.staged_hashes["stats"]
    )


def test_successful_refresh_preserves_exact_staged_advanced_bytes(successful_refresh):
    assert (
        sha256_bytes(successful_refresh.live.advanced)
        == successful_refresh.staged_hashes["advanced"]
    )


def test_successful_refresh_preserves_exact_staged_games_bytes(successful_refresh):
    assert (
        sha256_bytes(successful_refresh.live.games)
        == successful_refresh.staged_hashes["games"]
    )


def test_successful_refresh_preserves_staged_row_order(successful_refresh):
    committed = pd.read_parquet(successful_refresh.live.stats)
    expected = stats_frame(STAGED_STATS_ENTRIES)

    assert list(committed.columns) == list(expected.columns)
    assert committed["game_id"].tolist() == expected["game_id"].tolist()
    assert committed["player_id"].tolist() == expected["player_id"].tolist()


def test_successful_refresh_creates_missing_current_season_files(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    install(monkeypatch)

    assert (
        refresher.main(
            [
                "--slate-date",
                SLATE_DATE,
                "--season",
                str(SEASON),
                "--data-dir",
                str(data_dir),
            ]
        )
        == refresher.EXIT_OK
    )

    assert refresher.stats_path_for(data_dir, SEASON).exists()
    assert refresher.games_path_for(data_dir, SEASON).exists()
    assert refresher.advanced_path_for(data_dir, SEASON).exists()


# ---------------------------------------------------------------------------
# 26-27: commit failure, rollback and hash verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failing_call", [1, 2, 3])
def test_commit_failure_rolls_back(live, monkeypatch, failing_call):
    before = raw_tree_hashes(live.data_dir)

    real_replace = refresher.replace_file
    calls = {"count": 0}

    def flaky_replace(source, destination):
        calls["count"] += 1

        if calls["count"] == failing_call:
            raise OSError("simulated commit failure")

        real_replace(source, destination)

    install(monkeypatch)
    monkeypatch.setattr(refresher, "replace_file", flaky_replace)

    assert run(live) == refresher.EXIT_COMMIT_FAILED
    assert_live_unchanged(live, before)
    assert shadow_directories(live.root) == []


def test_commit_failure_removes_destination_without_predecessor(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    real_replace = refresher.replace_file
    calls = {"count": 0}

    def flaky_replace(source, destination):
        calls["count"] += 1

        if calls["count"] == 3:
            raise OSError("simulated commit failure")

        real_replace(source, destination)

    install(monkeypatch)
    monkeypatch.setattr(refresher, "replace_file", flaky_replace)

    code = refresher.main(
        [
            "--slate-date",
            SLATE_DATE,
            "--season",
            str(SEASON),
            "--data-dir",
            str(data_dir),
        ]
    )

    assert code == refresher.EXIT_COMMIT_FAILED
    assert not refresher.stats_path_for(data_dir, SEASON).exists()
    assert not refresher.games_path_for(data_dir, SEASON).exists()
    assert not refresher.advanced_path_for(data_dir, SEASON).exists()


def test_post_commit_hash_mismatch_fails_and_rolls_back(live, tmp_path):
    staged_root = tmp_path / "staged"
    staged_root.mkdir()

    staged_stats = staged_root / "stats.parquet"
    write_parquet(stats_frame(STAGED_STATS_ENTRIES), staged_stats)

    before = raw_tree_hashes(live.data_dir)

    candidate = refresher.CommitCandidate(
        name="stats",
        staged=staged_stats,
        live=live.stats,
        staged_sha256="0" * 64,
    )

    with pytest.raises(refresher.CommitError) as excinfo:
        refresher.commit_candidates([candidate], tmp_path / "rollback")

    assert "post-commit hash mismatch" in str(excinfo.value)
    assert_live_unchanged(live, before)


def test_commit_verifies_matching_hashes(live, tmp_path):
    staged_root = tmp_path / "staged"
    staged_root.mkdir()

    staged_stats = staged_root / "stats.parquet"
    write_parquet(stats_frame(STAGED_STATS_ENTRIES), staged_stats)

    candidate = refresher.CommitCandidate(
        name="stats",
        staged=staged_stats,
        live=live.stats,
        staged_sha256=sha256_bytes(staged_stats),
    )

    messages = refresher.commit_candidates([candidate], tmp_path / "rollback")

    assert len(messages) == 1
    assert sha256_bytes(live.stats) == sha256_bytes(staged_stats)


# ---------------------------------------------------------------------------
# 28-30: shadow cleanup and offline guarantee
# ---------------------------------------------------------------------------


def test_shadow_data_removed_after_success(live, monkeypatch):
    install(monkeypatch)

    assert run(live) == refresher.EXIT_OK
    assert shadow_directories(live.root) == []


def test_shadow_data_removed_after_failure(live, monkeypatch, capsys):
    install(monkeypatch, ingest=ingest_stub(stats=stats_frame([])))

    run_expect_refusal(live, capsys, "staged stats.parquet: empty")


def test_shadow_root_is_outside_the_live_raw_tree(live, monkeypatch):
    observed: dict[str, Path] = {}

    def recording_ingest(*, shadow_data_dir, season, python_executable=None):
        observed["shadow"] = Path(shadow_data_dir)
        observed["live_raw_snapshot"] = raw_tree_hashes(live.data_dir)
        ingest_stub()(shadow_data_dir=shadow_data_dir, season=season)

    install(monkeypatch, ingest=recording_ingest)

    before = raw_tree_hashes(live.data_dir)

    assert run(live) == refresher.EXIT_OK

    # The live tree was still untouched at the moment of the fetch.
    assert observed["live_raw_snapshot"] == before
    assert live.data_dir / "raw" not in observed["shadow"].parents


def test_no_test_contacts_the_network(live, monkeypatch, block_network):
    install(monkeypatch)

    assert run(live) == refresher.EXIT_OK
    assert block_network.subprocess == 0
    assert block_network.socket == 0

    # Prove the guard is armed rather than merely unexercised.
    with pytest.raises(AssertionError):
        refresher.subprocess.run(["true"])

    block_network.subprocess = 0
