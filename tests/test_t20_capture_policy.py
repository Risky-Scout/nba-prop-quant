"""What the T-20 capture path may and may not accept.

Two separate contracts are asserted here, both of which the production
controller leans on but neither of which it owns.

The first is the collector's new ``--offset-minutes`` narrowing. Production
consumes one window per game, T-20, while the capture schedule defines seven;
the narrowing is what lets a scheduled production run capture that one window
without deleting the research capability or changing what any existing caller
gets. "Backwards compatible" is the whole point of the argument, so the
default is asserted to be literally unchanged.

The second is that an ineligible capture is refused rather than quietly
priced. ``select_capture_bundle`` is the gate the pricing path already goes
through, and the controller deliberately calls the same function so that a
capture it accepts is a capture pricing will accept. These tests pin the
refusals that matter for a scheduled capture: a post-tip capture, a capture
whose component digests do not match the bytes on disk, a manual capture, a
capture that recorded collection errors, and two captures claiming the same
window.

No test here reaches balldontlie, writes outside ``tmp_path``, or runs any
model, pricing or publication code.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from nba_prop_quant.prospective_snapshot import (
    PRIMARY_OFFSET_MINUTES,
    select_capture_bundle,
)
from nba_prop_quant.snapshot_schedule import CAPTURE_OFFSETS_MINUTES

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from t20_capture_fixture import (  # noqa: E402
    SLATE,
    at,
    game,
    prop,
    write_capture,
)

COLLECTOR = PROJECT / "scripts" / "11_collect_snapshots.py"


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "collector_under_test", COLLECTOR
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


collector = load_collector()


def run_collector(argv: list[str], *, tmp_path: Path):
    """The collector as production invokes it: a real subprocess.

    The argument validation under test happens before the balldontlie client
    is constructed, so this reaches no network. It is still run with a key
    that cannot authenticate and a data root inside ``tmp_path``, so a
    regression that moved the validation later would fail the test rather
    than silently start calling the API.
    """
    environment = dict(os.environ)

    environment["BDL_API_KEY"] = "not-a-real-key"

    environment["NBA_PROP_DATA_DIR"] = str(tmp_path / "data")

    environment["NBA_PROP_MODEL_DIR"] = str(tmp_path / "models")

    environment["PYTHONPATH"] = str(PROJECT / "src")

    return subprocess.run(
        [sys.executable, str(COLLECTOR), *argv],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
        timeout=120,
        check=False,
    )


# ----------------------------------------------------------------------
# the collector's offset narrowing
# ----------------------------------------------------------------------


def test_the_research_capture_schedule_is_unchanged():
    """The seven windows research depends on are all still defined."""
    assert CAPTURE_OFFSETS_MINUTES == (1440, 480, 180, 90, 45, 20, 5)

    assert PRIMARY_OFFSET_MINUTES == 20


def test_omitting_the_offset_filter_selects_every_offset():
    """The behaviour every existing research and manual caller has.

    ``None`` rather than ``CAPTURE_OFFSETS_MINUTES`` is the contract: the
    collector only narrows the due list when a caller asked it to, so an
    unnarrowed run goes through exactly the code path it went through before
    this argument existed.
    """
    assert collector.selected_offsets(None) is None


def test_naming_the_production_offset_selects_only_it():
    assert collector.selected_offsets([20]) == (20,)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ([20, 5], (5, 20)),
        ([5, 20], (5, 20)),
        ([20, 20], (20,)),
        ([1440, 480, 180, 90, 45, 20, 5], (5, 20, 45, 90, 180, 480, 1440)),
    ],
)
def test_repeated_offsets_are_deduplicated_and_ordered(raw, expected):
    assert collector.selected_offsets(raw) == expected


@pytest.mark.parametrize("unknown", [0, 7, 19, 21, 60, -20])
def test_an_offset_the_schedule_does_not_define_is_refused(unknown):
    """A window that cannot exist must not be reported as nothing-to-do.

    Narrowing to an undefined offset would match no due window, so the run
    would capture nothing and exit green -- which is the exact silent failure
    the T-20 controller exists to end.
    """
    with pytest.raises(ValueError) as error:
        collector.selected_offsets([unknown])

    assert "--offset-minutes" in str(error.value)

    assert str(unknown) in str(error.value)


def test_one_unknown_offset_refuses_the_whole_request():
    with pytest.raises(ValueError):
        collector.selected_offsets([20, 7])


def test_the_offset_filter_is_rejected_without_scheduled(tmp_path):
    """It narrows *due* windows, so it is meaningless unscheduled.

    Accepting it on a manual run would read as "capture T-20" while actually
    capturing whatever the manual path captures.
    """
    result = run_collector(
        ["--date", SLATE, "--offset-minutes", "20"], tmp_path=tmp_path
    )

    assert result.returncode != 0, result.stdout

    assert "--scheduled" in result.stderr, result.stderr

    assert not (tmp_path / "data" / "snapshots" / "capture_runs").exists()


def test_an_undefined_offset_is_rejected_by_the_command_line(tmp_path):
    result = run_collector(
        ["--date", SLATE, "--scheduled", "--offset-minutes", "7"],
        tmp_path=tmp_path,
    )

    assert result.returncode != 0, result.stdout

    assert "--offset-minutes" in result.stderr, result.stderr

    assert not (tmp_path / "data" / "snapshots" / "capture_runs").exists()


def test_the_collector_still_accepts_its_pre_existing_arguments():
    """No existing caller's command line may have become invalid."""
    parser_source = COLLECTOR.read_text(encoding="utf-8")

    for argument in (
        '"--date"',
        '"--scheduled"',
        '"--now-utc"',
        '"--grace-minutes"',
    ):
        assert argument in parser_source, argument


def test_the_capture_payload_format_is_unchanged():
    """The append-only snapshot format is not renegotiated by this change.

    ``select_capture_bundle`` recomputes ``capture_id`` from the payload, so
    an added or renamed key would invalidate every capture ever written.
    """
    tree = ast.parse(COLLECTOR.read_text(encoding="utf-8"))

    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(
            isinstance(target, ast.Name) and target.id == "capture_run"
            for target in node.targets
        )
    ]

    assert len(literals) == 1, literals

    keys = {key.value for key in literals[0].keys}

    assert keys == {
        "active_players_count",
        "capture_id",
        "capture_reason",
        "component_sha256",
        "date",
        "due_game_ids",
        "error_count",
        "errors",
        "game_ids",
        "games_count",
        "grace_minutes",
        "injuries_count",
        "lineups_count",
        "player_props_count",
        "team_ids",
        "window_ids",
    }


# ----------------------------------------------------------------------
# which captures pricing is allowed to read
# ----------------------------------------------------------------------


def eligible_capture(snapshot_dir: Path, **overrides) -> str:
    """One capture that `select_capture_bundle` accepts, before corruption."""
    arguments = {
        "date": SLATE,
        "captured_at": at(22, 40).isoformat(),
        "window_ids": ["101:T-20m"],
        "due_game_ids": [101],
        "games": [game(101, at(23, 0))],
        "props": [prop(101, 900)],
    }

    arguments.update(overrides)

    return write_capture(snapshot_dir, **arguments)


def test_the_baseline_capture_is_eligible(tmp_path):
    """Without this the refusal tests below could pass for any reason."""
    snapshot_dir = tmp_path / "snapshots"

    capture_id = eligible_capture(snapshot_dir)

    bundle = select_capture_bundle(
        snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
    )

    assert bundle.capture_id == capture_id

    assert bundle.window_id == "101:T-20m"

    assert bundle.offset_minutes == 20

    assert bundle.captured_at == at(22, 40).isoformat()


def test_a_post_tip_capture_is_refused(tmp_path):
    """Case 13. A capture taken after tip-off is not a T-20 snapshot.

    This is the relabelling the controller must never be able to benefit
    from: the window id says T-20 and the payload says scheduled, and it is
    still refused, because the recorded capture time is after the recorded
    tip time.
    """
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(snapshot_dir, captured_at=at(23, 10).isoformat())

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    assert "pre-tip" in str(error.value)


def test_a_capture_taken_exactly_at_tip_is_refused(tmp_path):
    """The pre-tip test is strict, and the boundary is the tip itself."""
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(snapshot_dir, captured_at=at(23, 0).isoformat())

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    assert "pre-tip" in str(error.value)


@pytest.mark.parametrize(
    "component",
    ["games", "active_players", "injuries", "lineups", "player_props"],
)
def test_a_wrong_component_hash_is_refused(tmp_path, component):
    """Case 14. Every component digest is recomputed, not just the games.

    A capture whose recorded digest does not describe the bytes on disk is
    not a point-in-time snapshot of anything, and the market data pricing
    reads is one of those components.
    """
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(
        snapshot_dir,
        component_overrides={component: "0" * 64},
    )

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    message = str(error.value)

    assert component in message

    assert "hash mismatch" in message


def test_a_missing_component_hash_is_refused(tmp_path):
    """All six components are required, so an absent digest is not a pass."""
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(snapshot_dir)

    runs = snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"

    runs.write_text(
        runs.read_text(encoding="utf-8").replace('"player_props":', '"props":'),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    assert "player_props" in str(error.value)


def test_a_manual_capture_cannot_satisfy_a_scheduled_window(tmp_path):
    """``capture_reason`` is provenance, and production requires scheduled.

    An operator capturing by hand at a convenient moment is not the T-20
    protocol, however the window is labelled.
    """
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(snapshot_dir, reason="manual")

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    assert "exactly one scheduled capture, found 0" in " ".join(
        str(error.value).split()
    )


def test_a_capture_that_recorded_errors_is_refused(tmp_path):
    """A partial capture is refused rather than priced as complete."""
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(snapshot_dir, error_count=1)

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    assert "contains errors" in str(error.value)


def test_two_captures_claiming_one_window_are_refused(tmp_path):
    """Ambiguity is a failure, not a choice between two candidates.

    This is why the controller treats ``completed_window_ids`` as the
    authority on what has already been captured: capturing a window twice
    would make that window unpriceable rather than merely wasteful.
    """
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(snapshot_dir)

    eligible_capture(snapshot_dir, captured_at=at(22, 41).isoformat())

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    assert "found 2" in str(error.value)


def test_a_capture_without_the_target_game_is_refused(tmp_path):
    """The window may say the game is there; the payload has to agree."""
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(
        snapshot_dir,
        games=[game(102, at(23, 0), home=3, away=4)],
    )

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    assert "exactly one target game" in str(error.value)


def test_a_research_window_does_not_satisfy_the_production_window(tmp_path):
    """A T-45 capture is a research artifact, not a substitute for T-20."""
    snapshot_dir = tmp_path / "snapshots"

    eligible_capture(
        snapshot_dir,
        captured_at=at(22, 15).isoformat(),
        window_ids=["101:T-45m"],
    )

    with pytest.raises(RuntimeError) as error:
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=20
        )

    assert "101:T-20m" in str(error.value)

    # And it is a perfectly good T-45 capture, which is the point: the
    # refusal above is about the window, not about the capture being broken.
    assert (
        select_capture_bundle(
            snapshot_dir, date=SLATE, game_id=101, offset_minutes=45
        ).window_id
        == "101:T-45m"
    )


# ----------------------------------------------------------------------
# one window, one scheduled capture, however many sweeps are in flight
# ----------------------------------------------------------------------


def sweep_settings(tmp_path: Path):
    """The settings object the collector reads its snapshot root from."""
    return SimpleNamespace(
        bdl_api_key="test-key",
        bdl_base_url="https://example.invalid",
        bdl_requests_per_minute=600,
        snapshot_dir=tmp_path / "snapshots",
    )


def sweep_client(games: list[dict]):
    """A balldontlie client that answers from a fixed slate and no network."""

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def games(self, dates):
            return iter([dict(row) for row in games])

        def active_players(self, team_ids=None):
            return iter([])

        def injuries(self, team_ids=None, player_ids=None):
            return iter([])

        def lineups(self, game_ids):
            return iter([])

        def live_player_props(self, game_id):
            return [prop(int(game_id), 900 + int(game_id))]

    return FakeClient


#: How long a sweep holds at the gate below waiting for the others.
#:
#: The gate is deliberately breakable. Two sweeps that are free to overlap
#: both reach it and it releases them together, which is the interleaving
#: that produces the defect; two sweeps that have been serialised cannot
#: both reach it, and the one that does has to be let through rather than
#: deadlocking the test against the fix it is checking.
GATE_SECONDS = 2.0


def concurrent_scheduled_sweeps(
    tmp_path: Path,
    *,
    games: list[dict],
    nows: list[str],
) -> list[BaseException]:
    """Run one scheduled sweep per entry in ``nows``, as concurrently as the
    collector allows.

    Each sweep is the real ``main`` of the real collector, loaded as its own
    module object so that nothing is shared between them but the snapshot
    tree on disk -- which is exactly what two producers on the production
    host share. The interleaving is pinned rather than hoped for: every
    sweep holds at a barrier immediately after it reads the recorded windows
    for the first time, so sweeps that *can* overlap have all decided which
    windows are unclaimed before any of them records one. That is the
    ordering two sweeps inside one capture grace window reach whenever their
    collection overlaps, and guessing at it with sleeps would make this test
    flake either way.
    """
    settings = sweep_settings(tmp_path)

    read_claims = threading.Barrier(
        len(nows), timeout=GATE_SECONDS
    )

    failures: list[BaseException] = []

    def sweep(now: str) -> None:
        try:
            module = load_collector()

            module.get_settings = lambda: settings

            module.BDLClient = sweep_client(games)

            module.parse_args = lambda: SimpleNamespace(
                date=SLATE,
                scheduled=True,
                now_utc=now,
                grace_minutes=5,
                offset_minutes=[PRIMARY_OFFSET_MINUTES],
            )

            recorded_windows = module.completed_window_ids

            waited: list[bool] = []

            def read_recorded_windows(path):
                value = recorded_windows(path)

                if not waited:
                    waited.append(True)

                    try:
                        read_claims.wait()
                    except threading.BrokenBarrierError:
                        pass

                return value

            module.completed_window_ids = read_recorded_windows

            module.main()
        except BaseException as error:  # noqa: BLE001
            failures.append(error)

            read_claims.abort()

    threads = [
        threading.Thread(
            target=sweep, args=(now,), name=f"sweep-{index}"
        )
        for index, now in enumerate(nows)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=60)

        assert not thread.is_alive(), f"{thread.name} did not finish"

    return failures


def capture_runs_of(snapshot_dir: Path) -> list[dict]:
    path = snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def scheduled_captures_of(snapshot_dir: Path, window_id: str) -> list[dict]:
    path = snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"

    envelopes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return [
        envelope
        for envelope in envelopes
        if (envelope.get("payload") or {}).get("capture_reason") == "scheduled"
        and window_id in ((envelope.get("payload") or {}).get("window_ids") or [])
    ]


@pytest.mark.parametrize(
    "nows",
    [
        # The two sweeps were invoked with the same instant, which is what
        # a catch-up run and a scheduled run reach when both compute the
        # window's own T-20 time.
        [at(22, 40).isoformat(), at(22, 40).isoformat()],
        # And with different instants inside one grace window, which is what
        # two runs that merely overlap reach. Both are due: due_windows
        # accepts any now in [target, target + grace).
        [at(22, 40).isoformat(), at(22, 42).isoformat()],
    ],
    ids=["same-instant", "within-one-grace-window"],
)
def test_two_concurrent_sweeps_record_one_scheduled_capture(tmp_path, nows):
    """The production defect of 2026-10-23, as the collector reaches it.

    Three games tipped at 02:00 UTC and their shared T-20 window was swept by
    two producers inside the same grace window. Deciding which windows are
    unclaimed and recording the claim were two separate steps with nothing
    between them, so both sweeps read an empty claim set and both appended a
    scheduled capture naming ``101:T-20m``. Pricing then refused the window
    for being ambiguous -- correctly, and permanently, because the capture
    log is append-only and both captures are genuine.

    What is asserted is the invariant pricing already requires: a window has
    exactly one scheduled capture, and the capture it has is one pricing will
    accept.
    """
    games = [
        game(101, at(23, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    failures = concurrent_scheduled_sweeps(
        tmp_path, games=games, nows=nows
    )

    assert failures == [], failures

    snapshot_dir = tmp_path / "snapshots"

    # Stated first because this is the production symptom verbatim: on the
    # code that lost three games it raises "21717943:T-20m: expected exactly
    # one scheduled capture, found 2" with 101 in place of the game id.
    for game_id in (101, 102):
        bundle = select_capture_bundle(
            snapshot_dir,
            date=SLATE,
            game_id=game_id,
            offset_minutes=PRIMARY_OFFSET_MINUTES,
        )

        assert bundle.window_id == f"{game_id}:T-20m"

        assert bundle.captured_at in nows

    for window_id in ("101:T-20m", "102:T-20m"):
        assert len(scheduled_captures_of(snapshot_dir, window_id)) == 1, (
            f"{window_id} was recorded by more than one scheduled capture"
        )


def test_the_sweep_that_arrives_second_captures_nothing(tmp_path):
    """Serialising the sweeps is what keeps the component digests honest.

    A capture is content-addressed over the rows carrying its own
    ``captured_at``, so two sweeps invoked with the same instant write into
    one another's digests whatever either of them claims, and
    ``select_capture_bundle`` would refuse both for a hash mismatch rather
    than for an ambiguous window. That is why the claim is held across the
    collection and not merely across the append: the sweep that arrives
    second finds the window already recorded and captures nothing at all,
    which is the same thing any later scheduled run already does.
    """
    games = [game(101, at(23, 0))]

    failures = concurrent_scheduled_sweeps(
        tmp_path,
        games=games,
        nows=[at(22, 40).isoformat(), at(22, 40).isoformat()],
    )

    assert failures == [], failures

    snapshot_dir = tmp_path / "snapshots"

    runs = capture_runs_of(snapshot_dir)

    assert len(runs) == 1, runs

    assert runs[0]["payload"]["window_ids"] == ["101:T-20m"]

    assert runs[0]["payload"]["due_game_ids"] == [101]

    # One sweep's worth of component rows, which is what that capture's
    # recorded digests describe.
    for component in ("games", "player_props"):
        path = snapshot_dir / component / f"{SLATE}.jsonl"

        rows = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(rows) == 1, (component, rows)


def test_a_sweep_refuses_rather_than_waiting_out_the_capture_window(tmp_path):
    """The bounded wait fails closed, and says what it was waiting for.

    Waiting longer than the grace window cannot produce a capture pricing
    would accept, because the window stops being due. So a sweep that cannot
    get the claim inside it exits non-zero -- the controller reads that as a
    failed capture and refuses to serve -- rather than appending a second
    claim to a window whose holder is still mid-collection.
    """
    settings = sweep_settings(tmp_path)

    capture_run_path = (
        settings.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"
    )

    with collector.capture_window_claim(
        capture_run_path, timeout_seconds=0
    ) as held:
        with pytest.raises(RuntimeError) as error:
            with collector.capture_window_claim(
                capture_run_path, timeout_seconds=0
            ):
                pass

    assert str(held) in str(error.value)

    assert "refusing" in str(error.value)

    # And the claim is released on the way out, so the next sweep gets it.
    with collector.capture_window_claim(
        capture_run_path, timeout_seconds=0
    ):
        pass
