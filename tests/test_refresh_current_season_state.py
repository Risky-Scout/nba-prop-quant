"""Tests for the fail-closed current-season rolling-state refresh wrapper.

Nothing here touches the network. The two subprocess boundaries the wrapper
owns — the readiness preflight and the whole-season ingester — are substituted,
and an autouse fixture makes any socket use raise instead of connecting.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import io
import os
import socket
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]

MODULE_PATH = PROJECT / "ops" / "refresh_current_season_state.py"

SEASON = 2026
PRIOR_SEASON = 2025

SLATE_DATE = "2026-11-15"

LIVE_COMPLETED = [
    (1, "2026-11-10"),
    (2, "2026-11-11"),
    (3, "2026-11-12"),
]

STAGED_COMPLETED = LIVE_COMPLETED + [(4, "2026-11-14")]

SCHEDULED = [
    (5, "2026-11-20"),
    (6, "2026-11-21"),
]


def load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "refresh_current_season_state_under_test",
        MODULE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the refresh wrapper")

    module = importlib.util.module_from_spec(spec)

    # dataclasses resolves annotations through sys.modules, so the module has
    # to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


refresh = load_wrapper()

REAL_RUN_PREFLIGHT = refresh.run_preflight


class NetworkAccessAttempted(RuntimeError):
    """Raised instead of opening a socket while these tests run."""


@pytest.fixture(autouse=True)
def block_all_network(monkeypatch):
    def deny(*args, **kwargs):
        raise NetworkAccessAttempted(
            f"network access is forbidden in this test module: {args!r}"
        )

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)

    yield


# ----------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------


def make_stats(games, players_per_game=2, season=SEASON):
    records = []

    for game_id, date in games:
        for index in range(players_per_game):
            records.append(
                {
                    "player_id": game_id * 100 + index,
                    "game_id": game_id,
                    "date": pd.Timestamp(date),
                    "season": season,
                    "team_id": 1 + (index % 2),
                    "home_team_id": 1,
                    "visitor_team_id": 2,
                    "minutes": 30.5,
                    "pts": 20 + index,
                    "reb": 5,
                    "ast": 4,
                    "stl": 1,
                    "blk": 1,
                    "fg3m": 2,
                    "fga": 15,
                    "fta": 4,
                    "oreb": 1,
                    "turnover": 2,
                    "fg3a": 6,
                    "pf": 3,
                }
            )

    return pd.DataFrame(records)


def make_advanced(games, players_per_game=2, season=SEASON):
    records = []

    for game_id, date in games:
        for index in range(players_per_game):
            records.append(
                {
                    "player_id": game_id * 100 + index,
                    "game_id": game_id,
                    "date": pd.Timestamp(date),
                    "season": season,
                    "team_id": 1 + (index % 2),
                    "pie": 0.12,
                    "usage_percentage": 0.24,
                }
            )

    return pd.DataFrame(records)


def make_games(games, season=SEASON):
    records = []

    for game_id, date in games:
        records.append(
            {
                "id": game_id,
                "date": pd.Timestamp(date),
                "season": season,
                "home_team_id": 1,
                "visitor_team_id": 2,
                "status": "Final",
                "status_state": "post",
                "postponed": False,
            }
        )

    return pd.DataFrame(records)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}

    if not root.exists():
        return digests

    for path in sorted(root.rglob("*")):
        if path.is_file():
            key = str(path.relative_to(root))
            digests[key] = hash_bytes(path.read_bytes())

    return digests


@dataclass
class Env:
    data_dir: Path
    live: object
    prior_paths: list[Path] = field(default_factory=list)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def raw_hashes(self) -> dict[str, str]:
        return hash_tree(self.raw_dir)


@pytest.fixture
def env(tmp_path) -> Env:
    data_dir = tmp_path / "data"

    prior = refresh.season_paths(data_dir, PRIOR_SEASON)
    write_parquet(
        make_stats(
            [(90, "2025-12-01"), (91, "2025-12-02")],
            season=PRIOR_SEASON,
        ),
        prior.stats,
    )
    write_parquet(
        make_games(
            [(90, "2025-12-01"), (91, "2025-12-02")],
            season=PRIOR_SEASON,
        ),
        prior.games,
    )
    write_parquet(
        make_advanced(
            [(90, "2025-12-01"), (91, "2025-12-02")],
            season=PRIOR_SEASON,
        ),
        prior.advanced,
    )

    live = refresh.season_paths(data_dir, SEASON)
    write_parquet(make_stats(LIVE_COMPLETED), live.stats)
    write_parquet(make_advanced(LIVE_COMPLETED), live.advanced)
    write_parquet(make_games(LIVE_COMPLETED + SCHEDULED), live.games)

    return Env(
        data_dir=data_dir,
        live=live,
        prior_paths=[prior.stats, prior.games, prior.advanced],
    )


def stage_valid(shadow_root: Path) -> None:
    staged = refresh.season_paths(shadow_root, SEASON)

    write_parquet(make_stats(STAGED_COMPLETED), staged.stats)
    write_parquet(make_advanced(STAGED_COMPLETED), staged.advanced)
    write_parquet(make_games(STAGED_COMPLETED + SCHEDULED), staged.games)


def staging(
    *,
    stats: pd.DataFrame | bytes | None = None,
    advanced: pd.DataFrame | bytes | None = None,
    games: pd.DataFrame | bytes | None = None,
    extra_advanced: str | None = None,
) -> Callable[[Path], None]:
    """Build a stage function that overrides parts of the valid staged tree."""

    def stage(shadow_root: Path) -> None:
        stage_valid(shadow_root)
        staged = refresh.season_paths(shadow_root, SEASON)

        for override, path in (
            (stats, staged.stats),
            (advanced, staged.advanced),
            (games, staged.games),
        ):
            if override is None:
                continue

            if isinstance(override, bytes):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(override)
            else:
                write_parquet(override, path)

        if extra_advanced is not None:
            stray = staged.advanced_root / extra_advanced
            write_parquet(make_advanced([(1, "2026-11-10")]), stray)

    return stage


@dataclass
class RunResult:
    code: int
    output: str
    preflight_calls: list
    ingest_calls: list
    live_raw_during_fetch: dict
    staged_hashes: dict


def run_refresh(
    monkeypatch,
    env: Env,
    *,
    preflight_code: int = 0,
    preflight_script: Path | None = None,
    ingest_code: int = 0,
    stage: Callable[[Path], None] = stage_valid,
    slate_date: str = SLATE_DATE,
    season: int = SEASON,
) -> RunResult:
    preflight_calls: list = []
    ingest_calls: list = []
    live_raw_during_fetch: dict = {}
    staged_hashes: dict = {}

    if preflight_script is None:

        def fake_preflight(**kwargs):
            preflight_calls.append(kwargs)
            return preflight_code

        monkeypatch.setattr(refresh, "run_preflight", fake_preflight)

    else:
        monkeypatch.setattr(refresh, "PREFLIGHT_SCRIPT", preflight_script)

        def recording_preflight(**kwargs):
            preflight_calls.append(kwargs)
            return REAL_RUN_PREFLIGHT(**kwargs)

        monkeypatch.setattr(refresh, "run_preflight", recording_preflight)

    def fake_ingest(*, season, shadow_root):
        shadow_root = Path(shadow_root)

        ingest_calls.append(
            {"season": season, "shadow_root": shadow_root}
        )
        live_raw_during_fetch.update(hash_tree(env.raw_dir))

        stage(shadow_root)

        staged = refresh.season_paths(shadow_root, season)

        for label, path in staged.as_items():
            if path.exists():
                staged_hashes[label] = hash_bytes(path.read_bytes())

        return ingest_code

    monkeypatch.setattr(refresh, "run_shadow_ingest", fake_ingest)

    buffer = io.StringIO()

    with redirect_stdout(buffer), redirect_stderr(buffer):
        code = refresh.main(
            [
                "--slate-date",
                slate_date,
                "--season",
                str(season),
                "--data-dir",
                str(env.data_dir),
            ]
        )

    return RunResult(
        code=code,
        output=buffer.getvalue(),
        preflight_calls=preflight_calls,
        ingest_calls=ingest_calls,
        live_raw_during_fetch=live_raw_during_fetch,
        staged_hashes=staged_hashes,
    )


def write_fake_preflight(
    directory: Path,
    name: str,
    *,
    exit_code: int,
    prose: list[str],
) -> Path:
    script = directory / name
    lines = ["import sys"]

    for line in prose:
        lines.append(f"print({line!r})")

    lines.append(f"sys.exit({exit_code})")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return script


def leftover_workspaces(data_dir: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in data_dir.iterdir()
        if entry.name.startswith(refresh.SHADOW_PREFIX)
        or entry.name.startswith(refresh.ROLLBACK_PREFIX)
    )


# ----------------------------------------------------------------------
# 1-6: preflight readiness is the exit code
# ----------------------------------------------------------------------


def test_preflight_exit_zero_permits_shadow_ingestion(monkeypatch, env):
    result = run_refresh(monkeypatch, env, preflight_code=0)

    assert result.code == refresh.EXIT_OK
    assert len(result.preflight_calls) == 1
    assert len(result.ingest_calls) == 1


@pytest.mark.parametrize("stop_code", [10, 20, 30, 31])
def test_preflight_stop_code_performs_no_fetch_and_no_change(
    monkeypatch,
    env,
    stop_code,
):
    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env, preflight_code=stop_code)

    assert result.code == stop_code
    assert result.ingest_calls == []
    assert "withholds readiness" in result.output
    assert env.raw_hashes() == before


@pytest.mark.parametrize("unexpected", [1, 2, 5, 42, 99, 127])
def test_unexpected_preflight_exit_fails_closed(
    monkeypatch,
    env,
    unexpected,
):
    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env, preflight_code=unexpected)

    assert result.code == refresh.EXIT_PREFLIGHT_UNEXPECTED
    assert result.code not in refresh.PREFLIGHT_STOP_CODES
    assert result.ingest_calls == []
    assert f"unexpected code {unexpected}" in result.output
    assert env.raw_hashes() == before


# ----------------------------------------------------------------------
# 7-9, 46: banner prose is never a machine contract
# ----------------------------------------------------------------------


def test_readiness_ignores_banner_prose(monkeypatch, env, tmp_path):
    """Identical exit codes must behave identically whatever the banner says."""
    ready_with_prose = write_fake_preflight(
        tmp_path,
        "ready_with_prose.py",
        exit_code=0,
        prose=[
            "=" * 40,
            "CURRENT-SEASON HISTORY REFRESH PREFLIGHT — READ ONLY",
            "Action: READY_TO_REFRESH",
            "Writes performed: NONE",
        ],
    )

    ready_without_prose = write_fake_preflight(
        tmp_path,
        "ready_without_prose.py",
        exit_code=0,
        prose=[],
    )

    outcomes = []

    for script in (ready_with_prose, ready_without_prose):
        result = run_refresh(
            monkeypatch,
            env,
            preflight_script=script,
        )
        outcomes.append((result.code, len(result.ingest_calls)))

    assert outcomes == [
        (refresh.EXIT_OK, 1),
        (refresh.EXIT_OK, 1),
    ]


def test_removing_banner_prose_has_no_behavioral_effect(
    monkeypatch,
    env,
    tmp_path,
):
    hold_with_prose = write_fake_preflight(
        tmp_path,
        "hold_with_prose.py",
        exit_code=30,
        prose=[
            "Action: HOLD_STANDARD_NOT_READY",
            "Wait and rerun this read-only preflight later.",
        ],
    )

    hold_without_prose = write_fake_preflight(
        tmp_path,
        "hold_without_prose.py",
        exit_code=30,
        prose=[],
    )

    hold_with_unrelated_prose = write_fake_preflight(
        tmp_path,
        "hold_with_unrelated_prose.py",
        exit_code=30,
        prose=["totally unrelated operator note", "banner redesigned"],
    )

    before = env.raw_hashes()
    outcomes = []

    for script in (
        hold_with_prose,
        hold_without_prose,
        hold_with_unrelated_prose,
    ):
        result = run_refresh(monkeypatch, env, preflight_script=script)
        outcomes.append((result.code, len(result.ingest_calls)))

    assert outcomes == [(30, 0), (30, 0), (30, 0)]
    assert env.raw_hashes() == before


def test_banner_prose_is_not_a_machine_contract(monkeypatch, env, tmp_path):
    """Prose that contradicts the exit code must be ignored entirely."""
    says_ready_but_holds = write_fake_preflight(
        tmp_path,
        "says_ready_but_holds.py",
        exit_code=30,
        prose=["Action: READY_TO_REFRESH", "It is appropriate to refresh."],
    )

    says_hold_but_ready = write_fake_preflight(
        tmp_path,
        "says_hold_but_ready.py",
        exit_code=0,
        prose=["Action: HOLD_ADVANCED_NOT_READY", "Do NOT run the refresh."],
    )

    before = env.raw_hashes()

    holding = run_refresh(
        monkeypatch,
        env,
        preflight_script=says_ready_but_holds,
    )

    assert holding.code == 30
    assert holding.ingest_calls == []
    assert env.raw_hashes() == before

    proceeding = run_refresh(
        monkeypatch,
        env,
        preflight_script=says_hold_but_ready,
    )

    assert proceeding.code == refresh.EXIT_OK
    assert len(proceeding.ingest_calls) == 1


def test_source_contains_no_banner_prose_parser():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "Action" + ": " not in source
    assert "Action" + ":" not in source

    for status in (
        "READY_TO_REFRESH",
        "SKIP_REFRESH",
        "PRESEASON_BLOCK",
        "HOLD_STANDARD_NOT_READY",
        "HOLD_ADVANCED_NOT_READY",
    ):
        assert status not in source

    for stdout_capture in ("capture_output", "stdout", "communicate("):
        assert stdout_capture not in source


def test_wrapper_accepts_no_api_key_argument():
    source = MODULE_PATH.read_text(encoding="utf-8")

    for forbidden in ("--api-key", "--bdl-api-key", "api_key", "BDL_API_KEY"):
        assert forbidden not in source


# ----------------------------------------------------------------------
# 10: single writer
# ----------------------------------------------------------------------


def test_concurrent_writer_rejected(monkeypatch, env):
    lock_path = env.data_dir / refresh.LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    before = env.raw_hashes()

    holder = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        result = run_refresh(monkeypatch, env)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)

    assert result.code == refresh.EXIT_LOCK_BUSY
    assert result.preflight_calls == []
    assert result.ingest_calls == []
    assert "ANOTHER REFRESH IS ALREADY RUNNING" in result.output
    assert env.raw_hashes() == before


def test_lock_is_released_so_a_later_refresh_can_run(monkeypatch, env):
    first = run_refresh(monkeypatch, env)
    second = run_refresh(monkeypatch, env)

    assert first.code == refresh.EXIT_OK
    assert second.code == refresh.EXIT_OK


# ----------------------------------------------------------------------
# 11: the ingester never sees the live data root
# ----------------------------------------------------------------------


def test_live_state_unchanged_during_shadow_fetch(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK
    assert result.live_raw_during_fetch == before

    shadow_root = result.ingest_calls[0]["shadow_root"]

    assert shadow_root != env.data_dir
    assert env.raw_dir not in shadow_root.parents
    assert shadow_root.name.startswith(refresh.SHADOW_PREFIX)


# ----------------------------------------------------------------------
# 12-18: schema and readability
# ----------------------------------------------------------------------


def test_empty_staged_stats_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(stats=make_stats([])),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged stats is empty" in result.output
    assert env.raw_hashes() == before


def test_empty_staged_advanced_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(advanced=make_advanced([])),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged advanced is empty" in result.output
    assert env.raw_hashes() == before


@pytest.mark.parametrize("column", refresh.STATS_REQUIRED_COLUMNS)
def test_missing_required_standard_column_fails(monkeypatch, env, column):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            stats=make_stats(STAGED_COMPLETED).drop(columns=[column]),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged stats is missing required columns" in result.output
    assert column in result.output
    assert env.raw_hashes() == before


@pytest.mark.parametrize("column", refresh.ADVANCED_REQUIRED_COLUMNS)
def test_missing_required_advanced_column_fails(monkeypatch, env, column):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            advanced=make_advanced(STAGED_COMPLETED).drop(columns=[column]),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged advanced is missing required columns" in result.output
    assert column in result.output
    assert env.raw_hashes() == before


@pytest.mark.parametrize("column", refresh.GAMES_REQUIRED_COLUMNS)
def test_missing_required_games_column_fails(monkeypatch, env, column):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            games=make_games(STAGED_COMPLETED + SCHEDULED).drop(
                columns=[column]
            ),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged games is missing required columns" in result.output
    assert column in result.output
    assert env.raw_hashes() == before


def test_corrupt_staged_stats_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(stats=b"this is not a parquet file"),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged stats is not readable parquet" in result.output
    assert env.raw_hashes() == before


def test_corrupt_staged_advanced_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(advanced=b"this is not a parquet file"),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged advanced is not readable parquet" in result.output
    assert env.raw_hashes() == before


def test_corrupt_staged_games_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(games=b"this is not a parquet file"),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged games is not readable parquet" in result.output
    assert env.raw_hashes() == before


def test_missing_staged_file_fails(monkeypatch, env):
    before = env.raw_hashes()

    def stage(shadow_root: Path) -> None:
        stage_valid(shadow_root)
        refresh.season_paths(shadow_root, SEASON).advanced.unlink()

    result = run_refresh(monkeypatch, env, stage=stage)

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged advanced is missing" in result.output
    assert env.raw_hashes() == before


def test_failed_shadow_ingestion_fails_closed(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env, ingest_code=1)

    assert result.code == refresh.EXIT_INGEST_FAILED
    assert "shadow ingestion exited 1" in result.output
    assert env.raw_hashes() == before


# ----------------------------------------------------------------------
# 19-22: strict leakage rule
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_date", [SLATE_DATE, "2026-11-16"])
def test_stats_date_not_strictly_before_slate_date_fails(
    monkeypatch,
    env,
    bad_date,
):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            stats=make_stats(
                LIVE_COMPLETED + [(4, bad_date)],
            ),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged stats would leak same-or-later-day results" in result.output
    assert env.raw_hashes() == before


@pytest.mark.parametrize("bad_date", [SLATE_DATE, "2026-11-16"])
def test_advanced_date_not_strictly_before_slate_date_fails(
    monkeypatch,
    env,
    bad_date,
):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            advanced=make_advanced(
                LIVE_COMPLETED + [(4, bad_date)],
            ),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert (
        "staged advanced would leak same-or-later-day results"
        in result.output
    )
    assert env.raw_hashes() == before


def test_day_before_slate_date_is_accepted(monkeypatch, env):
    result = run_refresh(monkeypatch, env, slate_date="2026-11-15")

    assert result.code == refresh.EXIT_OK


def test_future_scheduled_games_are_legitimate(monkeypatch, env):
    """games.parquet legitimately carries dates at or after the slate date."""
    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK

    committed = pd.read_parquet(env.live.games)
    latest = pd.to_datetime(committed["date"]).max()

    assert latest > pd.Timestamp(SLATE_DATE)


# ----------------------------------------------------------------------
# 23-27: regression floors
# ----------------------------------------------------------------------


def test_stats_row_regression_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            stats=make_stats(STAGED_COMPLETED, players_per_game=1),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "stats row count fell from 6 to 4" in result.output
    assert "distinct game count fell" not in result.output
    assert env.raw_hashes() == before


def test_stats_distinct_game_regression_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            stats=make_stats(LIVE_COMPLETED[:2], players_per_game=4),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "stats distinct game count fell from 3 to 2" in result.output
    assert "row count fell" not in result.output
    assert env.raw_hashes() == before


def test_advanced_row_regression_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            advanced=make_advanced(STAGED_COMPLETED, players_per_game=1),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "advanced row count fell from 6 to 4" in result.output
    assert "distinct game count fell" not in result.output
    assert env.raw_hashes() == before


def test_advanced_distinct_game_regression_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            advanced=make_advanced(LIVE_COMPLETED[:2], players_per_game=4),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "advanced distinct game count fell from 3 to 2" in result.output
    assert "row count fell" not in result.output
    assert env.raw_hashes() == before


def test_games_distinct_game_regression_fails(monkeypatch, env):
    """Dropping the two scheduled games shrinks the distinct game-id set."""
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(games=make_games(STAGED_COMPLETED)),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "games distinct game count fell from 5 to 4" in result.output
    assert "stats " not in result.output.split("regresses against")[-1]
    assert env.raw_hashes() == before


def test_regression_floors_do_not_apply_without_live_state(
    monkeypatch,
    env,
):
    for path in (env.live.stats, env.live.advanced, env.live.games):
        path.unlink()

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK
    assert env.live.stats.exists()
    assert env.live.advanced.exists()
    assert env.live.games.exists()


def test_unreadable_live_state_blocks_the_refresh(monkeypatch, env):
    env.live.stats.write_bytes(b"corrupted live state")
    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "live stats is not readable parquet" in result.output
    assert env.raw_hashes() == before


# ----------------------------------------------------------------------
# 28-29: advanced tree layout
# ----------------------------------------------------------------------


def test_unexpected_live_advanced_parquet_fails(monkeypatch, env):
    stray = env.raw_dir / "advanced" / "stray.parquet"
    write_parquet(make_advanced([(1, "2026-11-10")]), stray)

    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert result.ingest_calls == []
    assert "live advanced tree contains unexpected parquet" in result.output
    assert "stray.parquet" in result.output
    assert stray.exists(), "an unexpected file must never be deleted"
    assert env.raw_hashes() == before


def test_unexpected_shadow_advanced_parquet_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(extra_advanced="season=2026/extra.parquet"),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged advanced tree contains unexpected parquet" in result.output
    assert "extra.parquet" in result.output
    assert env.raw_hashes() == before


def test_nested_unexpected_live_advanced_parquet_fails(monkeypatch, env):
    stray = env.raw_dir / "advanced" / "season=2026" / "backup" / "old.parquet"
    write_parquet(make_advanced([(1, "2026-11-10")]), stray)

    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "live advanced tree contains unexpected parquet" in result.output
    assert stray.exists()
    assert env.raw_hashes() == before


# ----------------------------------------------------------------------
# 30-31: every referenced game must resolve to a staged game row
# ----------------------------------------------------------------------


def test_stats_game_id_absent_from_staged_games_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            stats=make_stats(STAGED_COMPLETED + [(999, "2026-11-13")]),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged stats references 1 game id(s)" in result.output
    assert "[999]" in result.output
    assert env.raw_hashes() == before


def test_advanced_game_id_absent_from_staged_games_fails(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(
            advanced=make_advanced(STAGED_COMPLETED + [(999, "2026-11-13")]),
        ),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "staged advanced references 1 game id(s)" in result.output
    assert "[999]" in result.output
    assert env.raw_hashes() == before


# ----------------------------------------------------------------------
# 32-36: what a successful refresh changes, byte for byte
# ----------------------------------------------------------------------


def test_successful_refresh_changes_exactly_three_current_season_files(
    monkeypatch,
    env,
):
    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK

    after = env.raw_hashes()
    changed = {
        key
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }

    assert changed == {
        str(env.live.stats.relative_to(env.raw_dir)),
        str(env.live.games.relative_to(env.raw_dir)),
        str(env.live.advanced.relative_to(env.raw_dir)),
    }

    assert leftover_workspaces(env.data_dir) == []


def test_prior_season_files_remain_byte_identical(monkeypatch, env):
    before = {
        path: hash_bytes(path.read_bytes()) for path in env.prior_paths
    }

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK

    after = {
        path: hash_bytes(path.read_bytes()) for path in env.prior_paths
    }

    assert after == before


@pytest.mark.parametrize("label", ["stats", "advanced", "games"])
def test_committed_file_hash_equals_staged_hash(monkeypatch, env, label):
    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK

    destination = dict(env.live.as_items())[label]

    assert hash_bytes(destination.read_bytes()) == result.staged_hashes[label]


def test_committed_bytes_are_not_reserialized(monkeypatch, env):
    """Row order and column order survive because bytes are copied, not rewritten."""
    staged_stats = make_stats(STAGED_COMPLETED).iloc[::-1].reset_index(drop=True)
    expected_order = staged_stats["player_id"].tolist()
    expected_columns = list(staged_stats.columns)

    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(stats=staged_stats),
    )

    assert result.code == refresh.EXIT_OK

    committed = pd.read_parquet(env.live.stats)

    assert committed["player_id"].tolist() == expected_order
    assert list(committed.columns) == expected_columns


# ----------------------------------------------------------------------
# 37-41: commit failure and rollback
# ----------------------------------------------------------------------


def failing_replace(monkeypatch, fail_on_call: int):
    calls = {"count": 0}
    real_replace = os.replace

    def replace(source: Path, destination: Path) -> None:
        calls["count"] += 1

        if calls["count"] == fail_on_call:
            raise OSError(f"simulated replacement failure #{fail_on_call}")

        real_replace(source, destination)

    monkeypatch.setattr(refresh, "_atomic_replace", replace)

    return calls


@pytest.mark.parametrize("fail_on_call", [1, 2, 3])
def test_replacement_failure_rolls_back(monkeypatch, env, fail_on_call):
    before = env.raw_hashes()

    failing_replace(monkeypatch, fail_on_call)

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_COMMIT_ROLLED_BACK
    assert "the previous live bytes were restored and verified" in result.output
    assert env.raw_hashes() == before
    assert leftover_workspaces(env.data_dir) == []


def test_replacement_failure_removes_a_destination_with_no_predecessor(
    monkeypatch,
    env,
):
    env.live.stats.unlink()
    before = env.raw_hashes()

    # stats is created where nothing existed, then the games replacement fails.
    failing_replace(monkeypatch, 2)

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_COMMIT_ROLLED_BACK
    assert not env.live.stats.exists()
    assert env.raw_hashes() == before


def test_post_commit_hash_mismatch_rolls_back(monkeypatch, env):
    before = env.raw_hashes()

    calls = {"count": 0}
    real_replace = os.replace

    def replace(source: Path, destination: Path) -> None:
        calls["count"] += 1

        if calls["count"] == 3:
            destination.write_bytes(b"bytes that are not the staged bytes")
            return

        real_replace(source, destination)

    monkeypatch.setattr(refresh, "_atomic_replace", replace)

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_POST_COMMIT_ROLLED_BACK
    assert "post-commit hash verification failed" in result.output
    assert env.raw_hashes() == before
    assert leftover_workspaces(env.data_dir) == []


def test_rollback_verification_failure_hard_fails(monkeypatch, env):
    state = {"phase": "commit"}

    def replace(source: Path, destination: Path) -> None:
        if state["phase"] == "commit":
            state["phase"] = "rollback"
            raise OSError("simulated replacement failure")

        destination.write_bytes(b"rollback wrote the wrong bytes")

    monkeypatch.setattr(refresh, "_atomic_replace", replace)

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_ROLLBACK_FAILED
    assert result.code != refresh.EXIT_COMMIT_ROLLED_BACK
    assert "rollback could not be verified" in result.output
    assert str(env.live.stats) in result.output
    assert str(env.live.advanced) in result.output
    assert str(env.live.games) in result.output


def test_post_commit_mismatch_with_failed_rollback_hard_fails(
    monkeypatch,
    env,
):
    calls = {"count": 0}
    real_replace = os.replace

    def replace(source: Path, destination: Path) -> None:
        calls["count"] += 1

        if calls["count"] <= 3:
            if calls["count"] == 3:
                destination.write_bytes(b"not the staged bytes")
                return

            real_replace(source, destination)
            return

        destination.write_bytes(b"rollback wrote the wrong bytes")

    monkeypatch.setattr(refresh, "_atomic_replace", replace)

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_ROLLBACK_FAILED
    assert "post-commit hash verification failed" in result.output
    assert "rollback could not be verified" in result.output
    assert str(env.live.advanced) in result.output


# ----------------------------------------------------------------------
# 42-43: the shadow tree never survives
# ----------------------------------------------------------------------


def test_shadow_directory_removed_after_success(monkeypatch, env):
    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK
    assert not result.ingest_calls[0]["shadow_root"].exists()
    assert leftover_workspaces(env.data_dir) == []


def test_shadow_directory_removed_after_failure(monkeypatch, env):
    result = run_refresh(
        monkeypatch,
        env,
        stage=staging(stats=make_stats([])),
    )

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert not result.ingest_calls[0]["shadow_root"].exists()
    assert leftover_workspaces(env.data_dir) == []


# ----------------------------------------------------------------------
# 44-45: no network
# ----------------------------------------------------------------------


def test_no_test_contacts_balldontlie(monkeypatch, env):
    with pytest.raises(NetworkAccessAttempted):
        socket.create_connection(("api.balldontlie.io", 443))

    with pytest.raises(NetworkAccessAttempted):
        socket.getaddrinfo("api.balldontlie.io", 443)

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK


def test_no_test_contacts_any_network_endpoint(monkeypatch, env):
    for host, port in (
        ("example.invalid", 80),
        ("127.0.0.1", 9),
        ("localhost", 8080),
    ):
        with pytest.raises(NetworkAccessAttempted):
            socket.create_connection((host, port))

    with pytest.raises(NetworkAccessAttempted):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    result = run_refresh(monkeypatch, env)

    assert result.code == refresh.EXIT_OK


# ----------------------------------------------------------------------
# Wiring of the two subprocess boundaries
# ----------------------------------------------------------------------


def test_preflight_receives_the_slate_date_and_season(monkeypatch, env):
    result = run_refresh(monkeypatch, env)

    call = result.preflight_calls[0]

    assert call["season"] == SEASON
    assert call["slate_date"] == SLATE_DATE
    assert call["data_dir"] == env.data_dir


def test_ingest_receives_the_shadow_root_and_season(monkeypatch, env):
    result = run_refresh(monkeypatch, env)

    call = result.ingest_calls[0]

    assert call["season"] == SEASON
    assert call["shadow_root"].is_absolute()
    assert call["shadow_root"] != env.data_dir


def test_wrapper_points_at_the_unmodified_production_scripts():
    assert refresh.PREFLIGHT_SCRIPT == (
        PROJECT / "ops" / "preflight_current_season_refresh.py"
    )
    assert refresh.INGEST_SCRIPT == (
        PROJECT / "scripts" / "01_ingest_history_resume.py"
    )
    assert refresh.PREFLIGHT_SCRIPT.exists()
    assert refresh.INGEST_SCRIPT.exists()


def test_lock_path_is_canonical():
    assert refresh.LOCK_RELATIVE_PATH == Path(".locks") / (
        "current_season_refresh.lock"
    )


def test_slate_date_must_be_exact_iso(monkeypatch, env):
    before = env.raw_hashes()

    result = run_refresh(monkeypatch, env, slate_date="2026-11-5")

    assert result.code == refresh.EXIT_VALIDATION_FAILED
    assert "must be exact YYYY-MM-DD" in result.output
    assert result.preflight_calls == []
    assert env.raw_hashes() == before
