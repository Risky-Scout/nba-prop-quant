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
import os
import subprocess
import sys
from pathlib import Path

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
