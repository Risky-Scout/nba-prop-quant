"""Tests for the production T-20 capture, serving and publication controller.

The properties under test are about time, ownership and refusal, which is
why every collaborator that touches the world is injected. A test that had to
wait for a real T-20 window could assert none of this, and a test that called
balldontlie or published anything would be a production action rather than a
test: the clock, sleeping, the schedule and subprocess execution are all
fakes here, and the only real artifacts written are the snapshot JSONL files
a capture produces, inside a pytest temporary directory.

Nothing here trains, fits, prices, serves or publishes. The repository is
only read from.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from nba_prop_quant.snapshot_schedule import (
    CAPTURE_OFFSETS_MINUTES,
    due_windows,
)

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from t20_capture_fixture import (  # noqa: E402
    SLATE,
    at,
    eligible_game_ids,
    game,
    prop,
    write_capture,
)

WORKFLOW_DIR = PROJECT / ".github" / "workflows"

T20_WORKFLOW = WORKFLOW_DIR / "nba_t20_production.yml"

LIFECYCLE_WORKFLOW = WORKFLOW_DIR / "nba_production_lifecycle.yml"

ENTRY_POINT = PROJECT / "ops" / "run_t20_production_controller.py"


PRODUCTION_SHA = "a728b342a2f728c13a705abda8a5bbb79caf53ea"


def load_ops(name: str):
    spec = importlib.util.spec_from_file_location(
        f"ops_{name}", PROJECT / "ops" / f"{name}.py"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


controller = load_ops("run_t20_production_controller")

validator = load_ops("validate_workflows")


@pytest.fixture(autouse=True)
def block_all_network(monkeypatch):
    """No test in this module may reach the network."""

    def deny(*args, **kwargs):
        raise RuntimeError(
            f"network access is forbidden in this test module: {args!r}"
        )

    import socket

    monkeypatch.setattr(socket.socket, "connect", deny)

    monkeypatch.setattr(socket, "create_connection", deny)


# ----------------------------------------------------------------------
# fixtures: a slate, a clock, and a world the controller can act on
# ----------------------------------------------------------------------


class Clock:
    """A clock that only advances when the controller sleeps."""

    def __init__(self, now: datetime) -> None:
        self.now = now

        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

        self.now = self.now + timedelta(seconds=seconds)


class World:
    """A fake runner that performs the side effects the real tools would.

    The collector fake is a genuine reimplementation of the collector's own
    selection -- ``due_windows``, the offset narrowing the command asks for,
    and ``completed_window_ids`` -- so the capture it writes is the capture
    production would have, and the test is about the controller rather than
    about a stub that always agrees with it.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        schedules: list[list[dict]],
        clock: Clock,
        capture: bool = True,
        serving_outcome: str = "SERVED",
        feed_outcome: str = "PUBLISHED",
        collector_returncode: int = 0,
    ) -> None:
        self.data_root = tmp_path / "data"

        self.snapshot_dir = self.data_root / "snapshots"

        self.model_dir = tmp_path / "bundle" / "models"

        self.bundle_root = tmp_path / "bundle"

        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.schedules = list(schedules)

        self.clock = clock

        self.capture = capture

        self.serving_outcome = serving_outcome

        self.feed_outcome = feed_outcome

        self.collector_returncode = collector_returncode

        self.commands: list[list[str]] = []

        self.schedule_reads: list[str] = []

        self.served: list[list[int]] = []

        self.published: list[list[int]] = []

    # -- injected collaborators ---------------------------------------

    def schedule_source(self, slate_date: str) -> list[dict]:
        self.schedule_reads.append(slate_date)

        index = min(len(self.schedule_reads) - 1, len(self.schedules) - 1)

        return [dict(row) for row in self.schedules[index]]

    def __call__(self, command, *, environment, cwd):
        self.commands.append(list(command))

        script = str(command[1])

        if script.endswith("11_collect_snapshots.py"):
            return self._collect(command)

        if script.endswith("run_incumbent_production_serving.py"):
            return self._serve(command)

        if script.endswith("nba_distribution_publication.py"):
            return self._completed(command)

        if script.endswith("run_wizardofodds_publication.py"):
            return self._publish(command)

        raise AssertionError(f"unexpected command: {command}")

    # -- fakes ---------------------------------------------------------

    @staticmethod
    def _value(command, flag):
        return command[command.index(flag) + 1]

    @staticmethod
    def _values(command, flag):
        return [
            command[position + 1]
            for position, token in enumerate(command)
            if token == flag
        ]

    def _completed(self, command, *, returncode: int = 0, stdout: str = ""):
        import subprocess

        return subprocess.CompletedProcess(
            args=list(command), returncode=returncode, stdout=stdout, stderr=""
        )

    def _collect(self, command):
        if self.collector_returncode != 0:
            return self._completed(
                command,
                returncode=self.collector_returncode,
                stdout="collector refused",
            )

        if not self.capture:
            return self._completed(command)

        now = datetime.fromisoformat(
            self._value(command, "--now-utc")
        ).astimezone(UTC)

        offsets = {int(value) for value in self._values(command, "--offset-minutes")}

        games = self.schedules[-1]

        due = due_windows(
            [dict(row) for row in games],
            now,
            grace_minutes=int(self._value(command, "--grace-minutes")),
        )

        if offsets:
            due = [row for row in due if int(row["offset_minutes"]) in offsets]

        completed = controller.completed_window_ids(
            self.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"
        )

        unseen = [
            row for row in due if str(row["window_id"]) not in completed
        ]

        if not unseen:
            return self._completed(command, stdout="no unseen windows due")

        write_capture(
            self.snapshot_dir,
            date=SLATE,
            captured_at=now.isoformat(),
            window_ids=sorted(str(row["window_id"]) for row in unseen),
            due_game_ids=sorted(int(row["game_id"]) for row in unseen),
            games=[dict(row) for row in games],
            props=[
                prop(int(row["id"]), 600 + int(row["id"]))
                for row in games
            ],
        )

        return self._completed(command, stdout="captured")

    def _serve(self, command):
        receipt_path = Path(self._value(command, "--receipt-path"))

        served_ids = eligible_game_ids(self.snapshot_dir, SLATE)

        outcome = self.serving_outcome

        if outcome == "SERVED" and not served_ids:
            outcome = "NO_GAMES_ON_SLATE"

        self.served.append(served_ids)

        digest = hashlib.sha256(
            json.dumps(served_ids).encode("utf-8")
        ).hexdigest()

        receipt = {
            "outcome": outcome,
            "slate_date": self._value(command, "--slate-date"),
            "served_game_ids": served_ids,
            "prediction_artifact": (
                {"sha256": digest} if outcome == "SERVED" else None
            ),
            "priced_market_artifact": (
                {"sha256": digest} if outcome == "SERVED" else None
            ),
        }

        receipt_path.parent.mkdir(parents=True, exist_ok=True)

        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        return self._completed(command)

    def _publish(self, command):
        stage = self._value(command, "--stage")

        receipt_path = Path(self._value(command, "--receipt-path"))

        receipt_path.parent.mkdir(parents=True, exist_ok=True)

        if stage == "feed":
            served = json.loads(
                Path(
                    self._value(command, "--serving-receipt")
                ).read_text(encoding="utf-8")
            )

            published = list(served.get("served_game_ids") or [])

            self.published.append(published)

            receipt_path.write_text(
                json.dumps(
                    {
                        "outcome": self.feed_outcome,
                        "mode": self._value(command, "--mode"),
                        "slate_date": self._value(command, "--slate-date"),
                        "published_game_ids": published,
                        "release": {
                            "release_id": f"release-{len(self.published)}"
                        },
                    }
                ),
                encoding="utf-8",
            )

            return self._completed(command)

        receipt_path.write_text(
            json.dumps(
                {
                    "outcome": "PUBLISHED",
                    "mode": self._value(command, "--mode"),
                    "slate_date": self._value(command, "--slate-date"),
                    "page_sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )

        return self._completed(command)


def write_morning_receipt(
    world: World,
    *,
    slate_date: str = SLATE,
    production_sha: str = PRODUCTION_SHA,
    outcome: str = "NO_GAMES_ON_SLATE",
    refresh_outcome: str = "REFRESHED",
    ready: bool = True,
) -> Path:
    """The durable receipt the morning lifecycle's grading step already copies."""
    path = (
        world.data_root
        / "processed"
        / "incumbent_serving"
        / f"{slate_date}.json"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-10-20T09:45:00+00:00",
                "incumbent_authority": "frozen_deployment_bundle",
                "incumbent_fit_id": None,
                "incumbent_version": "nba_prop_quant_20260818T205213Z",
                "model_authority": "incumbent",
                "outcome": outcome,
                "production_code_sha": production_sha,
                "serving_bundle_root": str(world.bundle_root),
                "serving_model_dir": str(world.model_dir),
                "slate_date": slate_date,
                "slate_readiness": {
                    "ready": ready,
                    "reason": "the current-season rolling state advanced",
                    "refresh_outcome": refresh_outcome,
                },
            }
        ),
        encoding="utf-8",
    )

    return path


def build(
    world: World,
    *,
    clock: Clock,
    segment_start: datetime,
    segment_end: datetime,
    grace_minutes: int = 5,
    run_dir: Path | None = None,
):
    return controller.T20Controller(
        slate_date=SLATE,
        segment_start=segment_start,
        segment_end=segment_end,
        data_root=world.data_root,
        production_sha=PRODUCTION_SHA,
        run_dir=run_dir or (world.data_root.parent / "run"),
        grace_minutes=grace_minutes,
        clock=clock,
        sleep=clock.sleep,
        schedule_source=world.schedule_source,
        run_command=world,
    )


#: The second production segment: 19:25 UTC to 00:25 UTC the next day.
SEGMENT_2 = (at(19, 25), at(0, 25, day=21))

#: The first production segment: 14:25 UTC to 19:25 UTC.
SEGMENT_1 = (at(14, 25), at(19, 25))


def collector_commands(world: World) -> list[list[str]]:
    return [
        command
        for command in world.commands
        if str(command[1]).endswith("11_collect_snapshots.py")
    ]


# ----------------------------------------------------------------------
# the cheap exit
# ----------------------------------------------------------------------


def test_a_slate_with_no_games_is_a_cheap_no_op(tmp_path):
    """Case 1: nothing scheduled, nothing captured, nothing installed."""
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[]], clock=clock)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_NO_WINDOWS

    assert receipt["games_discovered"] == 0

    assert receipt["captures_completed"] == 0

    assert world.commands == []

    # One schedule read and no morning receipt: an off day must not be
    # reported as a lifecycle failure, and must not pay for anything.
    assert world.schedule_reads == [SLATE]


def test_games_outside_this_segment_are_a_cheap_no_op(tmp_path):
    """Case 2: a real slate whose T-20 windows belong to another segment."""
    clock = Clock(at(14, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_1[0], segment_end=SEGMENT_1[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_NO_WINDOWS

    assert receipt["games_discovered"] == 1

    assert receipt["t20_windows_in_segment"] == []

    assert world.commands == []

    assert clock.slept == []


def test_the_probe_reports_work_without_touching_anything(tmp_path):
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    plan = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).plan()

    assert plan["pending_windows"] == ["101:T-20m"]

    assert world.commands == []


def test_a_t20_time_no_segment_covers_fails_closed(tmp_path):
    """A tip so early that its T-20 falls before the first segment."""
    clock = Clock(at(14, 25))

    # 14:00 UTC tip puts T-20 at 13:40 UTC, before 14:25 UTC.
    world = World(
        tmp_path, schedules=[[game(101, at(14, 0))]], clock=clock
    )

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_1[0],
            segment_end=SEGMENT_1[1],
        ).run()

    assert error.value.outcome == controller.OUTCOME_UNSUPPORTED_TIME

    assert "101:T-20m" in str(error.value)

    assert world.commands == []


# ----------------------------------------------------------------------
# capture grouping
# ----------------------------------------------------------------------


def test_one_due_game_is_one_capture(tmp_path):
    """Case 3."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    assert receipt["captures_completed"] == 1

    assert len(collector_commands(world)) == 1

    assert receipt["captures"][0]["window_ids"] == ["101:T-20m"]

    assert len(receipt["capture_ids"]) == 1

    # Slept to the window rather than polling toward it.
    assert clock.slept == [(at(22, 40) - at(19, 25)).total_seconds()]

    # Served once, derived once, published once.
    assert len(receipt["serving_invocations"]) == 1

    assert len(receipt["distribution_outcomes"]) == 1

    assert [record["outcome"] for record in receipt["feed_publication_outcomes"]] == [
        "PUBLISHED"
    ]

    assert [record["outcome"] for record in receipt["page_publication_outcomes"]] == [
        "PUBLISHED"
    ]


def test_games_sharing_a_tip_are_one_capture(tmp_path):
    """Case 4: three games at 19:00 ET are one capture event, not three."""
    clock = Clock(at(19, 25))

    games = [
        game(101, at(23, 0)),
        game(102, at(23, 0), home=3, away=4),
        game(103, at(23, 0), home=5, away=6),
    ]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["captures_completed"] == 1

    assert len(collector_commands(world)) == 1

    assert receipt["captures"][0]["window_ids"] == [
        "101:T-20m",
        "102:T-20m",
        "103:T-20m",
    ]

    assert len(receipt["serving_invocations"]) == 1

    assert len(receipt["feed_publication_outcomes"]) == 1


def test_each_unique_t20_time_is_its_own_capture(tmp_path):
    """Case 5: three tip groups, three captures, and no more."""
    clock = Clock(at(19, 25))

    games = [
        game(101, at(23, 0)),
        game(102, at(23, 0), home=3, away=4),
        game(103, at(23, 30), home=5, away=6),
        game(104, at(2, 0, day=21), home=7, away=8),
    ]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=at(5, 25, day=21)
    ).run()

    assert receipt["captures_completed"] == 3

    assert [capture["window_ids"] for capture in receipt["captures"]] == [
        ["101:T-20m", "102:T-20m"],
        ["103:T-20m"],
        ["104:T-20m"],
    ]

    # One capture per unique T-20 time, not per game, per player or per prop.
    assert len(collector_commands(world)) == 3

    assert len(receipt["serving_invocations"]) == 3

    assert len(receipt["feed_publication_outcomes"]) == 3


def test_production_schedules_only_the_t20_offset(tmp_path):
    """Case 6: the six research windows are never scheduled by production."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=at(5, 25, day=21)
    ).run()

    assert receipt["captures_completed"] == 1

    for command in collector_commands(world):
        assert "--offset-minutes" in command

        assert command[command.index("--offset-minutes") + 1] == "20"

    captured = controller.completed_window_ids(
        world.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"
    )

    assert captured == {"101:T-20m"}

    research = {
        f"101:T-{offset}m"
        for offset in CAPTURE_OFFSETS_MINUTES
        if offset != 20
    }

    assert not (captured & research)

    # And the capability itself is untouched.
    assert CAPTURE_OFFSETS_MINUTES == (1440, 480, 180, 90, 45, 20, 5)


def test_an_already_completed_window_is_not_recaptured(tmp_path):
    """Case 7: the capture log is the authority on restart and retry."""
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    write_capture(
        world.snapshot_dir,
        date=SLATE,
        captured_at=at(22, 40).isoformat(),
        window_ids=["101:T-20m"],
        due_game_ids=[101],
        games=games,
        props=[prop(101, 601)],
    )

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_NO_WINDOWS

    assert receipt["already_completed_windows"] == ["101:T-20m"]

    assert world.commands == []


# ----------------------------------------------------------------------
# schedule changes
# ----------------------------------------------------------------------


def test_a_tip_that_moved_later_is_rescheduled(tmp_path):
    """Case 8: revalidated at the window, recomputed, slept to again."""
    clock = Clock(at(19, 25))

    original = [game(101, at(23, 0))]

    moved = [game(101, at(23, 30))]

    world = World(tmp_path, schedules=[original, moved], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["captures_completed"] == 1

    assert receipt["rescheduled_windows"][0]["window_id"] == "101:T-20m"

    assert receipt["rescheduled_windows"][0]["moved_to_utc"] == (
        at(23, 10).isoformat()
    )

    assert receipt["captures"][0]["due_at_utc"] == at(23, 10).isoformat()

    # Slept to the original window, found the move, slept to the new one.
    assert len(clock.slept) == 2


def test_a_tip_that_moved_into_a_later_segment_is_deferred(tmp_path):
    """Case 9: the later segment owns it, so this one leaves it alone."""
    clock = Clock(at(19, 25))

    original = [game(101, at(23, 0))]

    moved = [game(101, at(1, 0, day=21))]

    world = World(tmp_path, schedules=[original, moved], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    assert receipt["captures_completed"] == 0

    assert receipt["deferred_to_later_segment"][0]["window_id"] == "101:T-20m"

    assert receipt["deferred_to_later_segment"][0]["moved_to_utc"] == (
        at(0, 40, day=21).isoformat()
    )

    assert collector_commands(world) == []


def test_a_tip_that_moved_earlier_past_grace_is_a_missed_window(tmp_path):
    """Case 10: no later capture may be relabelled T-20."""
    clock = Clock(at(19, 25))

    original = [game(101, at(23, 0))]

    moved = [game(101, at(22, 0))]

    world = World(tmp_path, schedules=[original, moved], clock=clock)

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert error.value.outcome == controller.OUTCOME_MISSED_WINDOW

    assert "101:T-20m" in str(error.value)

    assert collector_commands(world) == []


def test_a_runner_picked_up_past_grace_is_a_missed_window(tmp_path):
    """Case 11: pickup latency on this runner has been measured in hours."""
    clock = Clock(at(23, 30))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert error.value.outcome == controller.OUTCOME_MISSED_WINDOW

    assert "capture grace" in str(error.value)

    assert collector_commands(world) == []

    assert clock.slept == []


def test_a_game_withdrawn_from_the_slate_is_recorded_not_captured(tmp_path):
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))], []], clock=clock
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["captures_completed"] == 0

    assert receipt["withdrawn_windows"][0]["window_id"] == "101:T-20m"

    assert collector_commands(world) == []


# ----------------------------------------------------------------------
# missing snapshot semantics
# ----------------------------------------------------------------------


def test_a_due_game_with_no_capture_is_a_required_snapshot_failure(tmp_path):
    """Case 12: the silence this controller exists to break."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0))]],
        clock=clock,
        capture=False,
    )

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert error.value.outcome == (
        controller.OUTCOME_REQUIRED_SNAPSHOT_MISSING
    )

    assert "game 101" in str(error.value)

    # Nothing was served and nothing was published.
    assert [
        command
        for command in world.commands
        if not str(command[1]).endswith("11_collect_snapshots.py")
    ] == []


def test_a_failed_collector_is_a_required_snapshot_failure(tmp_path):
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0))]],
        clock=clock,
        collector_returncode=3,
    )

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert error.value.outcome == (
        controller.OUTCOME_REQUIRED_SNAPSHOT_MISSING
    )


def test_serving_reporting_no_games_after_a_capture_is_a_failure(tmp_path):
    """A real slate must never be recorded as an empty one here."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0))]],
        clock=clock,
        serving_outcome="NO_GAMES_ON_SLATE",
    )

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert error.value.outcome == (
        controller.OUTCOME_REQUIRED_SNAPSHOT_MISSING
    )

    assert "about to be recorded as an empty one" in str(error.value)

    # And nothing was published behind it.
    assert [
        command
        for command in world.commands
        if str(command[1]).endswith("run_wizardofodds_publication.py")
    ] == []


def test_a_refused_feed_publication_fails_the_segment(tmp_path):
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0))]],
        clock=clock,
        feed_outcome="NOT_SERVED_NOTHING_TO_PUBLISH",
    )

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert error.value.outcome == controller.OUTCOME_FAILED

    assert [
        command
        for command in world.commands
        if "--stage" in command and command[command.index("--stage") + 1] == "page"
    ] == []


# ----------------------------------------------------------------------
# the morning lifecycle's handoff
# ----------------------------------------------------------------------


def test_an_absent_morning_receipt_is_morning_production_not_ready(tmp_path):
    """Case 18."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert error.value.outcome == controller.OUTCOME_MORNING_NOT_READY

    assert world.commands == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"slate_date": "2026-10-19"},
        {"production_sha": "0" * 40},
        {"outcome": "NO_PRODUCTION_SLATE"},
        {"outcome": "SERVING_FAILED"},
        {"refresh_outcome": "PRESEASON_BLOCK"},
        {"ready": False},
    ],
)
def test_a_morning_receipt_that_proves_nothing_is_refused(tmp_path, overrides):
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    write_morning_receipt(world, **overrides)

    with pytest.raises(controller.ControllerRefused) as error:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert error.value.outcome == controller.OUTCOME_MORNING_NOT_READY

    assert world.commands == []


def test_the_morning_receipt_supplies_the_verified_serving_roots(tmp_path):
    """The controller never installs the frozen package; it reuses the proof."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    serving = next(
        command
        for command in world.commands
        if str(command[1]).endswith("run_incumbent_production_serving.py")
    )

    assert serving[serving.index("--model-dir") + 1] == str(world.model_dir)

    assert serving[serving.index("--frozen-bundle-root") + 1] == str(
        world.bundle_root
    )

    assert (
        receipt["morning_readiness"]["incumbent_authority"]
        == "frozen_deployment_bundle"
    )

    # And no frozen-package install was invoked by this controller.
    assert not any(
        "install_frozen_model_artifacts.py" in " ".join(command)
        for command in world.commands
    )


# ----------------------------------------------------------------------
# cumulative serving and publication
# ----------------------------------------------------------------------


def test_later_groups_serve_and_publish_the_cumulative_slate(tmp_path):
    """Cases 15, 16 and 17 together, which is the only way they mean anything.

    Three tip groups on one slate. After the second capture the slate served
    must still carry the first group's games, and after the third it must
    carry all four; and the feed must publish the cumulative set rather than
    replacing it with the newest group. The controller is not what makes that
    true -- the prediction script enumerates every game with an eligible
    bundle and pricing rewrites the whole parquet -- so this is the proof that
    the existing behaviour is cumulative and that nothing new is needed.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(23, 0)),
        game(102, at(23, 0), home=3, away=4),
        game(103, at(23, 30), home=5, away=6),
        game(104, at(2, 0, day=21), home=7, away=8),
    ]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world,
        clock=clock,
        segment_start=SEGMENT_2[0],
        segment_end=at(5, 25, day=21),
    ).run()

    assert receipt["captures_completed"] == 3

    assert world.served == [
        [101, 102],
        [101, 102, 103],
        [101, 102, 103, 104],
    ]

    assert world.published == [
        [101, 102],
        [101, 102, 103],
        [101, 102, 103, 104],
    ]

    # Each game carries its own capture, not the newest one.
    from nba_prop_quant.prospective_snapshot import select_capture_bundle

    bundles = {
        game_id: select_capture_bundle(
            world.snapshot_dir, date=SLATE, game_id=game_id
        )
        for game_id in (101, 102, 103, 104)
    }

    assert bundles[101].captured_at == at(22, 40).isoformat()

    assert bundles[102].captured_at == at(22, 40).isoformat()

    assert bundles[103].captured_at == at(23, 10).isoformat()

    assert bundles[104].captured_at == at(1, 40, day=21).isoformat()

    assert bundles[101].capture_id == bundles[102].capture_id

    assert len({bundle.capture_id for bundle in bundles.values()}) == 3


def test_an_unchanged_priced_artifact_is_not_published_twice(tmp_path):
    """Publication idempotency, proven from this invocation's own receipt."""
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0)), game(102, at(23, 30), home=3, away=4)]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    # A collector that captures the first group and then stops writing would
    # make the second group's served slate byte-identical to the first's. The
    # controller refuses that for a different reason, so instead force the
    # digest to repeat by serving the same game set twice.
    world.serving_outcome = "SERVED"

    monkeypatched = []

    original = world._serve

    def serve_with_frozen_digest(command):
        result = original(command)

        path = Path(World._value(command, "--receipt-path"))

        receipt = json.loads(path.read_text(encoding="utf-8"))

        receipt["priced_market_artifact"] = {"sha256": "f" * 64}

        path.write_text(json.dumps(receipt), encoding="utf-8")

        monkeypatched.append(command)

        return result

    world._serve = serve_with_frozen_digest

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["captures_completed"] == 2

    assert len(monkeypatched) == 2

    outcomes = [
        record["outcome"] for record in receipt["feed_publication_outcomes"]
    ]

    assert outcomes == [
        "PUBLISHED",
        controller.PUBLICATION_OUTCOME_SKIPPED,
    ]

    assert len(world.published) == 1


# ----------------------------------------------------------------------
# segment resolution
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "cron,now,expected_start,expected_end,slate",
    [
        (
            "25 14 * * *",
            at(14, 30),
            at(14, 25),
            at(19, 25),
            "2026-10-20",
        ),
        (
            "25 19 * * *",
            at(19, 30),
            at(19, 25),
            at(0, 25, day=21),
            "2026-10-20",
        ),
        (
            "25 0 * * *",
            at(1, 0, day=21),
            at(0, 25, day=21),
            at(5, 25, day=21),
            "2026-10-20",
        ),
    ],
)
def test_the_three_segments_are_contiguous_and_half_open(
    cron, now, expected_start, expected_end, slate
):
    start, end = controller.resolve_segment(now_utc=now, cron=cron)

    assert (start, end) == (expected_start, expected_end)

    # The slate date is Eastern, not the UTC calendar date. The 00:25 UTC
    # segment is the evening before in Eastern time.
    assert controller.slate_date_for(start) == slate


def test_the_segments_cover_the_supported_window_without_overlap():
    segments = [
        controller.resolve_segment(
            now_utc=datetime(2026, 10, 20, hour, minute, tzinfo=UTC)
            + timedelta(minutes=1),
            cron=f"{minute} {hour} * * *",
        )
        for hour, minute in sorted(controller.SEGMENT_STARTS_UTC)
    ]

    ordered = sorted(segments)

    for (_, end), (start, _) in pairwise(ordered):
        assert end <= start, "segments must not overlap"

    start, end = controller.supported_window("2026-10-20")

    assert start == at(14, 25)

    assert end == at(5, 25, day=21)


def test_a_delayed_pickup_resolves_the_scheduled_segment_not_the_clock():
    """Pickup latency must not move the invocation into another segment."""
    start, end = controller.resolve_segment(
        now_utc=at(4, 0, day=21), cron="25 0 * * *"
    )

    assert (start, end) == (at(0, 25, day=21), at(5, 25, day=21))


def test_an_unknown_cron_is_refused():
    with pytest.raises(controller.ControllerRefused) as error:
        controller.resolve_segment(now_utc=at(12, 0), cron="7 3 * * *")

    assert error.value.outcome == controller.OUTCOME_FAILED


# ----------------------------------------------------------------------
# the workflow
# ----------------------------------------------------------------------


def workflow(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    return payload


def triggers(payload: dict) -> dict:
    return payload.get("on") or payload.get(True) or {}


def t20_job() -> dict:
    return workflow(T20_WORKFLOW)["jobs"]["t20"]


def t20_commands() -> str:
    return "\n".join(
        str(step.get("run", "")) for step in t20_job().get("steps") or []
    )


def test_the_workflow_schedules_exactly_the_three_segments():
    crons = sorted(
        entry["cron"] for entry in triggers(workflow(T20_WORKFLOW))["schedule"]
    )

    assert crons == sorted(validator.T20_SCHEDULE_CRONS)

    # And the controller's own table is the same three starts, so the code and
    # the schedule cannot drift apart.
    assert sorted(controller.SEGMENT_STARTS_UTC) == sorted(
        (int(cron.split()[1]), int(cron.split()[0])) for cron in crons
    )


def test_the_workflow_is_not_a_five_minute_poller():
    for entry in triggers(workflow(T20_WORKFLOW))["schedule"]:
        minute = str(entry["cron"]).split()[0]

        assert "/" not in minute, entry

        assert "," not in minute, entry

        assert minute not in ("0", "00", "*"), entry

    assert len(triggers(workflow(T20_WORKFLOW))["schedule"]) == 3


def test_the_workflow_runs_on_the_production_runner_and_environment():
    job = t20_job()

    assert job["runs-on"] == ["self-hosted", "nba-production"]

    assert job["environment"] == "wizardofodds-production"

    assert "pull_request" not in triggers(workflow(T20_WORKFLOW))


def test_the_workflow_checks_out_the_authoritative_production_ref():
    checkout = [
        step
        for step in t20_job()["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout")
    ]

    assert len(checkout) == 1

    assert "production/wizardofodds-integration" in str(
        checkout[0]["with"]["ref"]
    )

    # And the SHA it landed on is recorded and checked at run time.
    assert "git rev-parse HEAD" in t20_commands()

    assert "$PRODUCTION_REF" in t20_commands()


def test_the_workflow_runs_the_controller_and_nothing_heavier():
    commands = t20_commands()

    assert validator.T20_ENTRY_POINT in commands

    for entry_point in validator.T20_FORBIDDEN_ENTRY_POINTS:
        assert entry_point not in commands, entry_point


def test_the_workflow_installs_production_dependencies_only():
    commands = t20_commands()

    assert "pip install -e ." in commands

    for token in validator.T20_FORBIDDEN_INSTALL_TOKENS:
        assert token not in commands, token


def test_the_expensive_install_is_gated_behind_the_cheap_probe():
    steps = t20_job()["steps"]

    names = [step.get("name") for step in steps]

    probe = next(
        step for step in steps if "--probe-only" in str(step.get("run", ""))
    )

    assert probe.get("id") == "probe"

    gated = [
        step
        for step in steps
        if "pip install -e ." in str(step.get("run", ""))
        or (
            validator.T20_ENTRY_POINT in str(step.get("run", ""))
            and "--probe-only" not in str(step.get("run", ""))
        )
    ]

    assert [step["name"] for step in gated] == [
        "Install the production package",
        "Capture T-20, serve the slate and publish the canary",
    ], names

    for step in gated:
        assert step.get("if") == "steps.probe.outputs.t20_work == 'true'", step

    # And the probe comes before every one of them.
    assert names.index(probe["name"]) < min(
        names.index(step["name"]) for step in gated
    )

    # The probe itself installs two small wheels, not the scientific stack.
    probe_install = next(
        step
        for step in steps
        if "python -m pip install 'httpx" in str(step.get("run", ""))
    )

    assert probe_install.get("if") is None

    assert "pandas" not in str(probe_install["run"])


def test_the_workflow_binds_secrets_only_to_environment_variables():
    for line in T20_WORKFLOW.read_text(encoding="utf-8").splitlines():
        if "secrets." in line and not line.strip().startswith("#"):
            assert line.strip().split(":")[0].isupper(), line

    assert validator.secret_leak_problems(T20_WORKFLOW.name) == []


def test_the_credential_is_the_existing_bdl_key_and_not_an_odds_api_key():
    text = T20_WORKFLOW.read_text(encoding="utf-8")

    assert "secrets.BDL_API_KEY" in text

    assert "ODDS_API_KEY" not in text


def test_the_workflow_reuses_the_existing_deployment_key_by_name():
    text = T20_WORKFLOW.read_text(encoding="utf-8")

    assert "NBA_WOO_SSH_PRIVATE_KEY: ${{ secrets.NBA_WOO_SSH_PRIVATE_KEY }}" in (
        text
    )


def test_the_lifecycle_workflow_still_owns_the_once_daily_work():
    """This change adds a scheduler; it does not move the heavy path."""
    lifecycle = LIFECYCLE_WORKFLOW.read_text(encoding="utf-8")

    for entry_point in validator.T20_FORBIDDEN_ENTRY_POINTS:
        assert entry_point in lifecycle, entry_point

    assert "cron: '37 9 * * *'" in lifecycle


def test_the_static_validator_covers_the_new_workflow():
    assert validator.validate() == []

    assert validator.T20_WORKFLOW == T20_WORKFLOW.name


def test_the_validator_catches_a_fourth_segment(tmp_path, monkeypatch):
    """The rule must actually fail on the mistake it exists to catch."""
    broken = tmp_path / "workflows"

    broken.mkdir()

    for path in (
        T20_WORKFLOW,
        LIFECYCLE_WORKFLOW,
        WORKFLOW_DIR / "ci.yml",
    ):
        (broken / path.name).write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    payload = (broken / T20_WORKFLOW.name).read_text(encoding="utf-8")

    (broken / T20_WORKFLOW.name).write_text(
        payload.replace(
            "    - cron: '25 0 * * *'",
            "    - cron: '25 0 * * *'\n    - cron: '25 9 * * *'",
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(validator, "WORKFLOW_DIR", broken)

    assert any(
        "contiguous segments" in problem for problem in validator.validate()
    )


# ----------------------------------------------------------------------
# the freeze
# ----------------------------------------------------------------------


def test_the_frozen_canonical_identities_are_unchanged():
    """Case 19. This is a scheduling change; the model must be where it was."""
    model_spec = PROJECT / "research" / "final_model" / "final_model_spec.json"

    policy = (
        PROJECT
        / "research"
        / "final_model"
        / "live_shadow_promotion_policy.json"
    )

    assert (
        hashlib.sha256(model_spec.read_bytes()).hexdigest()
        == "154109658127920ebeac381bd21e8bb29390505384124419899236cce37fdc64"
    )

    assert (
        hashlib.sha256(policy.read_bytes()).hexdigest()
        == "a0784aff602d9ee14ba1ef91b30fe972fb7eeb02ec80e0b8afbb7d9ff764aee8"
    )

    assert (
        json.loads(model_spec.read_text(encoding="utf-8"))["factor_spec_hash"]
        == "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
    )


def test_live_publication_remains_disabled():
    """Case 20."""
    switch = json.loads(
        (PROJECT / "ops" / "wizardofodds_publishing_switch.json").read_text(
            encoding="utf-8"
        )
    )

    assert switch["modes"]["live"]["state"] == "DISABLED"

    assert switch["modes"]["canary"]["state"] == "ENABLED"

    assert controller.PUBLICATION_MODE == "canary"

    source = ENTRY_POINT.read_text(encoding="utf-8")

    # No mode argument, so there is no code path from a schedule to live.
    assert '"--mode"' in source

    assert "PUBLICATION_MODE" in source

    for token in ("--mode live", "--stage live", "NBA_PROP_WOO_PUBLISH"):
        assert token not in source, token

    workflow = (
        PROJECT / ".github" / "workflows" / "nba_t20_production.yml"
    ).read_text(encoding="utf-8")

    for token in validator.PUBLICATION_ACTIVATION_TOKENS:
        assert token not in workflow, token


def test_the_controller_has_no_mode_option():
    with pytest.raises(SystemExit):
        controller.parse_args(
            [
                "--data-root",
                "/tmp",
                "--production-sha",
                "x",
                "--mode",
                "live",
            ]
        )


# ----------------------------------------------------------------------
# constants that must not drift
# ----------------------------------------------------------------------


def test_the_outcome_vocabulary_is_pinned_to_the_serving_module():
    """Stated locally so the probe stays light; pinned here so it stays true."""
    serving = load_ops("run_incumbent_production_serving")

    assert controller.REFRESH_OUTCOMES_THAT_SERVE == (
        serving.REFRESH_OUTCOMES_THAT_SERVE
    )

    assert controller.MORNING_OUTCOMES_THAT_PROVE_READINESS == frozenset(
        {serving.OUTCOME_SERVED, serving.OUTCOME_NO_GAMES_ON_SLATE}
    )

    assert controller.SERVING_OUTCOME_SERVED == serving.OUTCOME_SERVED

    assert controller.SERVING_OUTCOME_NO_GAMES == (
        serving.OUTCOME_NO_GAMES_ON_SLATE
    )

    assert serving.OUTCOME_NO_PRODUCTION_SLATE not in (
        controller.MORNING_OUTCOMES_THAT_PROVE_READINESS
    )

    # The durable receipt this controller reads is the one the lifecycle
    # already copies, at the path the lifecycle already writes it to.
    assert (
        f"$NBA_PROP_DATA_DIR/{controller.MORNING_RECEIPT_RELATIVE.as_posix()}"
        in LIFECYCLE_WORKFLOW.read_text(encoding="utf-8")
    )


def test_the_publication_outcome_vocabulary_is_pinned():
    publication = load_ops("run_wizardofodds_publication")

    assert controller.PUBLICATION_OUTCOME_PUBLISHED == (
        publication.OUTCOME_PUBLISHED
    )

    assert controller.PUBLICATION_OUTCOME_NOT_SERVED == (
        publication.OUTCOME_NOT_SERVED
    )

    assert controller.PUBLICATION_MODE in publication.FEED_DESTINATIONS


def test_completed_window_ids_matches_the_collectors_own_function(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "collector_under_test",
        PROJECT / "scripts" / "11_collect_snapshots.py",
    )

    collector = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(collector)

    snapshot_dir = tmp_path / "snapshots"

    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 40).isoformat(),
        window_ids=["101:T-20m", "102:T-20m"],
        due_game_ids=[101, 102],
        games=[game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)],
    )

    path = snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"

    assert controller.completed_window_ids(path) == (
        collector.completed_window_ids(path)
    )

    assert controller.completed_window_ids(path) == {
        "101:T-20m",
        "102:T-20m",
    }

    assert controller.completed_window_ids(
        snapshot_dir / "capture_runs" / "2026-10-21.jsonl"
    ) == set()


def test_a_diverted_consumer_snapshot_directory_is_refused(
    tmp_path, monkeypatch
):
    """Producer and consumer must be looking at the same tree."""
    monkeypatch.setenv(
        "NBA_PROP_GATE3_SNAPSHOT_DIR", str(tmp_path / "elsewhere")
    )

    with pytest.raises(controller.ControllerRefused) as error:
        controller.resolve_snapshot_dir(tmp_path / "data")

    assert error.value.outcome == controller.OUTCOME_FAILED

    monkeypatch.setenv(
        "NBA_PROP_GATE3_SNAPSHOT_DIR", str(tmp_path / "data" / "snapshots")
    )

    assert controller.resolve_snapshot_dir(tmp_path / "data") == (
        (tmp_path / "data" / "snapshots").resolve()
    )


def test_the_receipt_carries_everything_a_segment_must_record(tmp_path):
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    for field in (
        "slate_date",
        "production_sha",
        "segment_start_utc",
        "segment_end_utc",
        "games_discovered",
        "t20_windows_in_segment",
        "already_completed_windows",
        "captures_attempted",
        "captures_completed",
        "capture_ids",
        "serving_invocations",
        "distribution_outcomes",
        "feed_publication_outcomes",
        "page_publication_outcomes",
        "start_time_utc",
        "finish_time_utc",
        "wall_seconds",
        "final_outcome",
    ):
        assert field in receipt, field

    # Small: no model arrays, no parquet, no API payload archives.
    assert len(json.dumps(receipt)) < 8000

    assert controller.render(receipt).startswith(
        "## NBA T-20 production controller"
    )
