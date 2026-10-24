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

        #: The moments the controller woke up at, in order. Asserted on in
        #: preference to :attr:`slept` wherever a test is about which windows
        #: the controller came awake for, because the number of sleeps now
        #: depends on how many advisory windows share the segment and the
        #: moments do not.
        self.woke: list[datetime] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

        self.now = self.now + timedelta(seconds=seconds)

        self.woke.append(self.now)


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
        advisory_serving_outcome: str = "SERVED",
        advisory_serving_returncode: int = 0,
        advisory_feed_outcome: str = "PUBLISHED",
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

        self.advisory_serving_outcome = advisory_serving_outcome

        self.advisory_serving_returncode = advisory_serving_returncode

        self.advisory_feed_outcome = advisory_feed_outcome

        self.commands: list[list[str]] = []

        #: When each command in :attr:`commands` was issued, by the injected
        #: clock. The reservation before a T-20 capture is a statement about
        #: time, so proving it needs the time and not only the order.
        self.command_times: list[datetime] = []

        self.schedule_reads: list[str] = []

        self.served: list[list[int]] = []

        self.published: list[list[int]] = []

        # The advisory windows are recorded apart from the T-20 ones all the
        # way through, so every assertion in this module about what production
        # served and published stays an assertion about T-20. Mixing them into
        # one list would make a test that says "the slate was served twice"
        # quietly mean "something was served twice".
        self.advisory_served: list[tuple[int, list[int]]] = []

        self.advisory_published: list[tuple[str, str]] = []

    # -- injected collaborators ---------------------------------------

    def schedule_source(self, slate_date: str) -> list[dict]:
        self.schedule_reads.append(slate_date)

        index = min(len(self.schedule_reads) - 1, len(self.schedules) - 1)

        return [dict(row) for row in self.schedules[index]]

    def __call__(self, command, *, environment, cwd):
        self.commands.append(list(command))

        self.command_times.append(self.clock())

        script = str(command[1])

        if script.endswith("11_collect_snapshots.py"):
            return self._collect(command)

        if script.endswith("run_incumbent_production_serving.py"):
            return self._serve(command)

        if script.endswith("run_advisory_window_serving.py"):
            return self._serve_advisory(command)

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

    @staticmethod
    def _optional(command, flag):
        return command[command.index(flag) + 1] if flag in command else None

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

    def _serve_advisory(self, command):
        """What ops/run_advisory_window_serving.py writes, in receipt shape.

        A real receipt, built the way the real one is: the production outcome
        and provenance ``build_receipt`` emits, plus the three advisory fields.
        The served games are read out of the capture log at this window's own
        offset, so the fake agrees with what was actually captured rather than
        with the controller that asked for it.
        """
        offset = int(self._value(command, "--snapshot-offset-minutes"))

        receipt_path = Path(self._value(command, "--receipt-path"))

        served_ids = eligible_game_ids(
            self.snapshot_dir, SLATE, offset_minutes=offset
        )

        self.advisory_served.append((offset, served_ids))

        outcome = self.advisory_serving_outcome

        if outcome == "SERVED" and not served_ids:
            outcome = "NO_GAMES_ON_SLATE"

        digest = hashlib.sha256(
            json.dumps([offset, served_ids]).encode("utf-8")
        ).hexdigest()

        receipt_path.parent.mkdir(parents=True, exist_ok=True)

        receipt_path.write_text(
            json.dumps(
                {
                    "advisory": True,
                    "advisory_capture_offset_minutes": offset,
                    "advisory_capture_window_ids": self._values(
                        command, "--capture-window-id"
                    ),
                    "outcome": outcome,
                    "prediction_artifact": (
                        {"sha256": digest} if outcome == "SERVED" else None
                    ),
                    "priced_market_artifact": (
                        {"sha256": digest} if outcome == "SERVED" else None
                    ),
                    "served_game_ids": served_ids,
                    "slate_date": self._value(command, "--slate-date"),
                }
            ),
            encoding="utf-8",
        )

        return self._completed(
            command, returncode=self.advisory_serving_returncode
        )

    def _publish(self, command):
        stage = self._value(command, "--stage")

        receipt_path = Path(self._value(command, "--receipt-path"))

        receipt_path.parent.mkdir(parents=True, exist_ok=True)

        # A named destination is how an advisory release is addressed and the
        # T-20 release is published by not naming one, so this is the same
        # distinction the controller makes.
        feed_destination = self._optional(command, "--feed-destination-name")

        page_destination = self._optional(command, "--page-destination-name")

        if feed_destination or page_destination:
            return self._publish_advisory(
                command,
                stage=stage,
                receipt_path=receipt_path,
                feed_destination=feed_destination,
                page_destination=page_destination,
            )

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

    def _publish_advisory(
        self, command, *, stage, receipt_path, feed_destination, page_destination
    ):
        served = (
            json.loads(
                Path(
                    self._value(command, "--serving-receipt")
                ).read_text(encoding="utf-8")
            )
            if stage == "feed"
            else {}
        )

        offset = int(served.get("advisory_capture_offset_minutes") or 0)

        self.advisory_published.append(
            (stage, str(feed_destination or page_destination))
        )

        if stage == "feed":
            digest = str(
                (served.get("priced_market_artifact") or {}).get("sha256") or ""
            )

            receipt_path.write_text(
                json.dumps(
                    {
                        "mode": self._value(command, "--mode"),
                        "outcome": self.advisory_feed_outcome,
                        "published_game_ids": list(
                            served.get("served_game_ids") or []
                        ),
                        "release": {
                            "release_id": (
                                f"nba_prop_quant_advisory_t{offset}m__{SLATE}__"
                                + digest[:16]
                            )
                        },
                        "slate_date": self._value(command, "--slate-date"),
                    }
                ),
                encoding="utf-8",
            )

            return self._completed(command)

        receipt_path.write_text(
            json.dumps(
                {
                    "mode": self._value(command, "--mode"),
                    "outcome": "PUBLISHED",
                    "page_prefix": self._optional(command, "--page-prefix"),
                    "page_sha256": "1" * 64,
                    "slate_date": self._value(command, "--slate-date"),
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


def write_served_record(
    world: World,
    *,
    window_ids: list[str],
    slate_date: str = SLATE,
    release_id: str = "release-0",
) -> Path:
    """The durable statement that a published release carried these windows."""
    path = (
        world.data_root
        / "processed"
        / "t20_served_windows"
        / f"{slate_date}.jsonl"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "offset_minutes": 20,
                    "page_sha256": "0" * 64,
                    "priced_market_artifact_sha256": "1" * 64,
                    "production_sha": PRODUCTION_SHA,
                    "publication_mode": "canary",
                    "publication_outcome": "PUBLISHED",
                    "record_type": controller.SERVED_RECORD_TYPE,
                    "recorded_at_utc": at(23, 0).isoformat(),
                    "release_id": release_id,
                    "slate_date": slate_date,
                    "window_ids": list(window_ids),
                },
                sort_keys=True,
            )
            + "\n"
        )

    return path


def served_records(world: World, *, slate_date: str = SLATE) -> list[dict]:
    path = (
        world.data_root
        / "processed"
        / "t20_served_windows"
        / f"{slate_date}.jsonl"
    )

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def capture_commands(world: World) -> list[list[str]]:
    """Every collector invocation, at whatever offset it asked for."""
    return [
        command
        for command in world.commands
        if str(command[1]).endswith("11_collect_snapshots.py")
    ]


def collector_commands(
    world: World, *, offset_minutes: int = 20
) -> list[list[str]]:
    """The collector invocations at one offset, the production one by default.

    Narrowed to an offset because the controller now captures the advisory
    windows through the same collector. Every assertion in this module about
    the production capture is an assertion about T-20 specifically, and
    counting advisory invocations into it would make "one capture per due
    group" mean something else.
    """
    wanted = str(int(offset_minutes))

    return [
        command
        for command in capture_commands(world)
        if "--offset-minutes" in command
        and command[command.index("--offset-minutes") + 1] == wanted
    ]


def publication_commands(world: World, *, stage: str) -> list[list[str]]:
    """The T-20 publication invocations at one stage.

    An advisory release is addressed by naming its destination; the T-20
    release is published by naming none, because its destination is the
    mode's default. Filtering on that is what keeps every assertion in this
    module about what production published an assertion about T-20.
    """
    return [
        command
        for command in world.commands
        if str(command[1]).endswith("run_wizardofodds_publication.py")
        and "--stage" in command
        and command[command.index("--stage") + 1] == stage
        and "--feed-destination-name" not in command
        and "--page-destination-name" not in command
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

    # Slept to the window rather than polling toward it. Asserted as the
    # total rather than as one sleep, because the advisory windows between
    # 19:25 and 22:40 are what the controller wakes up for on the way, and
    # what this test is about is that it was awake at 22:40 and not before.
    assert sum(clock.slept) == (at(22, 40) - at(19, 25)).total_seconds()

    assert receipt["captures"][0]["attempted_at_utc"] == at(22, 40).isoformat()

    assert (
        collector_commands(world)[0][
            collector_commands(world)[0].index("--now-utc") + 1
        ]
        == at(22, 40).isoformat()
    )

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


def test_production_schedules_only_the_t20_and_advisory_offsets(tmp_path):
    """Case 6: the research windows are never scheduled by production.

    Narrower than it was, and stronger where it counts. It used to assert
    that T-20 was the only window production ever captured, which stopped
    being true when the early advisory windows were added. What it asserts
    now is the part that was always the point: T-1440, T-480 and T-5 are
    research windows and production schedules none of them, every capture is
    taken at a named offset rather than at all of them, and T-20 is still the
    only window that is served to the production canary.
    """
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=at(5, 25, day=21)
    ).run()

    assert receipt["captures_completed"] == 1

    # Every capture names exactly one offset. Omitting the flag is what
    # selects every offset the schedule defines, which is what research and
    # manual callers get and what production must never ask for.
    requested = set()

    for command in capture_commands(world):
        assert command.count("--offset-minutes") == 1

        requested.add(int(command[command.index("--offset-minutes") + 1]))

    assert requested == {20, 180, 90, 45}

    captured = controller.completed_window_ids(
        world.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"
    )

    assert captured == {
        "101:T-20m",
        "101:T-180m",
        "101:T-90m",
        "101:T-45m",
    }

    research = {
        f"101:T-{offset}m"
        for offset in CAPTURE_OFFSETS_MINUTES
        if offset != controller.PRODUCTION_OFFSET_MINUTES
        and offset not in controller.ADVISORY_OFFSETS_MINUTES
    }

    assert research == {"101:T-1440m", "101:T-480m", "101:T-5m"}

    assert not (captured & research)

    # The production canary is served from the T-20 window and from nothing
    # else: one serving invocation and one feed publication for the one T-20
    # group, whatever the advisory windows did beside it.
    assert len(receipt["serving_invocations"]) == 1

    assert [record["outcome"] for record in receipt["feed_publication_outcomes"]] == [
        "PUBLISHED"
    ]

    assert [offset for offset, _ in world.advisory_served] == [180, 90, 45]

    # And the capability itself is untouched.
    assert CAPTURE_OFFSETS_MINUTES == (1440, 480, 180, 90, 45, 20, 5)


def advisory_commands(world: World) -> list[list[str]]:
    """Every command the advisory path issued, serving and publication both."""
    return [
        command
        for command in world.commands
        if str(command[1]).endswith("run_advisory_window_serving.py")
        or "--feed-destination-name" in command
        or "--page-destination-name" in command
    ]


def test_an_advisory_window_publishes_to_its_own_destinations(tmp_path):
    """Addressed by name, and never by the T-20 destination's name."""
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    expected = []

    for offset in controller.ADVISORY_OFFSETS_MINUTES:
        expected += [
            ("feed", controller.ADVISORY_FEED_DESTINATIONS[offset]),
            ("page", controller.ADVISORY_PAGE_DESTINATIONS[offset]),
        ]

    assert world.advisory_published == expected

    named = {name for _, name in world.advisory_published}

    assert controller.T20_FEED_DESTINATION not in named

    assert controller.T20_PAGE_DESTINATION not in named

    # Each release id carries the offset it was captured at.
    for offset in controller.ADVISORY_OFFSETS_MINUTES:
        assert any(
            f"advisory_t{offset}m" in release
            for release in receipt["advisory_releases"]
        ), receipt["advisory_releases"]

    assert len(receipt["advisory_releases"]) == len(
        controller.ADVISORY_OFFSETS_MINUTES
    )

    # And no advisory release id could be read as a T-20 one.
    for release in receipt["advisory_releases"]:
        assert release.startswith("nba_prop_quant_advisory_t")


def test_no_advisory_work_interleaves_a_t20_serve_and_its_publication(tmp_path):
    """The T-20 sequence is one uninterrupted run of commands.

    The priced-markets parquet is one file per slate and both the T-20 path
    and the advisory path rewrite it, so the T-20 serve, the derivation behind
    its prices and its two publications have to be contiguous. The publisher
    and the grader both re-hash that file against the receipt, so advisory
    bytes could never be published or graded as T-20 -- but a serve
    interrupted halfway would still fail the segment, which is the thing this
    rules out.
    """
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    serve = next(
        index
        for index, command in enumerate(world.commands)
        if str(command[1]).endswith("run_incumbent_production_serving.py")
    )

    page = max(
        index
        for index, command in enumerate(world.commands)
        if command in publication_commands(world, stage="page")
    )

    assert serve < page

    interleaved = [
        command
        for command in world.commands[serve : page + 1]
        if command in advisory_commands(world)
    ]

    assert interleaved == []


def test_an_advisory_serve_does_not_start_inside_the_t20_reservation(tmp_path):
    """The one rule that keeps the advisory path from costing a T-20 window.

    The T-45 window of the 23:00 game falls at 22:15, five minutes before the
    22:20 T-20 window of the 22:40 game. Five minutes is inside the ten-minute
    reservation, so that capture is still taken on the clock -- it is cheap and
    it is the point-in-time market -- and its serving waits until the T-20
    group is done.
    """
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0)), game(102, at(22, 40))]],
        clock=clock,
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    reservation = timedelta(minutes=controller.T20_RESERVATION_MINUTES)

    t20_due = [at(22, 20), at(22, 40)]

    issued_at = dict(
        zip(
            [tuple(command) for command in world.commands],
            world.command_times,
            strict=True,
        )
    )

    # The T-45 capture at 22:15 happened, on the clock, inside the reservation.
    assert at(22, 15) in {
        issued_at[tuple(command)]
        for command in capture_commands(world)
        if command[command.index("--offset-minutes") + 1] == "45"
    }

    # And no advisory serving or publication command was issued inside the
    # reservation before either T-20 capture.
    for command in advisory_commands(world):
        issued = issued_at[tuple(command)]

        for due in t20_due:
            assert not (due - reservation <= issued < due), (
                f"{command[1]} ran at {issued.isoformat()}, inside the "
                f"reservation before the T-20 capture due at {due.isoformat()}"
            )

    # The held window was served once the T-20 group it was waiting on had
    # been captured, which is what the reservation is for: it delays an
    # advisory release and never drops one.
    t20_capture = max(
        index
        for index, command in enumerate(world.commands)
        if command in collector_commands(world)
        and command[command.index("--now-utc") + 1] == at(22, 20).isoformat()
    )

    held = [
        index
        for index, command in enumerate(world.commands)
        if str(command[1]).endswith("run_advisory_window_serving.py")
        and command[command.index("--snapshot-offset-minutes") + 1] == "45"
        and issued_at[tuple(command)] >= at(22, 15)
    ]

    assert held and min(held) > t20_capture

    # Both T-20 windows were captured on time regardless.
    assert receipt["captures_completed"] == 2

    assert [capture["attempted_at_utc"] for capture in receipt["captures"]] == [
        at(22, 20).isoformat(),
        at(22, 40).isoformat(),
    ]

    # And the held window was served in the end rather than lost.
    assert 45 in {offset for offset, _ in world.advisory_served}


def test_an_already_completed_window_is_not_recaptured(tmp_path):
    """Case 7: the capture log is the authority on restart and retry.

    Every window this segment owns is already in it, advisory included, so
    the restart recaptures nothing: the capture log is one log and it is the
    authority for all four windows.

    It is no longer a no-op, which is the one claim here that the served-
    window ledger took away. The T-20 window is captured and no release
    carries it, so this run serves and publishes it from the capture that
    already exists. What it must not do, and still does not do, is run the
    collector again -- the window's T-20 time has passed and a snapshot taken
    now would be a snapshot of now. The no-op case is the test below, where
    publication has been recorded as well.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    for captured_at, window_id in (
        (at(20, 0), "101:T-180m"),
        (at(21, 30), "101:T-90m"),
        (at(22, 15), "101:T-45m"),
        (at(22, 40), "101:T-20m"),
    ):
        write_capture(
            world.snapshot_dir,
            date=SLATE,
            captured_at=captured_at.isoformat(),
            window_ids=[window_id],
            due_game_ids=[101],
            games=games,
            props=[prop(101, 601)],
        )

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["already_completed_windows"] == ["101:T-20m"]

    assert receipt["advisory_pending_windows"] == []

    assert collector_commands(world) == []

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE


def test_a_window_a_release_already_carries_is_the_cheap_no_op(tmp_path):
    """Case 7 once publication has been recorded: there is nothing to do.

    The same four captures as above, with the durable statement that a
    release carried the T-20 one. Nothing runs at all -- no collector, no
    serving, no publication -- which is what every later segment of a
    finished slate has to cost.

    A finished slate means all four windows now, not just the T-20 one. The
    advisory captures are written for the same reason the test above writes
    them: a segment that owns three uncaptured advisory windows has work to
    do whatever the T-20 window's state is, and the statement being made
    here is about the segment that has none.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    for captured_at, window_id in (
        (at(20, 0), "101:T-180m"),
        (at(21, 30), "101:T-90m"),
        (at(22, 15), "101:T-45m"),
        (at(22, 40), "101:T-20m"),
    ):
        write_capture(
            world.snapshot_dir,
            date=SLATE,
            captured_at=captured_at.isoformat(),
            window_ids=[window_id],
            due_game_ids=[101],
            games=games,
            props=[prop(101, 601)],
        )

    write_served_record(world, window_ids=["101:T-20m"])

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_NO_WINDOWS

    assert receipt["already_completed_windows"] == ["101:T-20m"]

    assert receipt["captured_unserved_windows"] == []

    assert receipt["advisory_pending_windows"] == []

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

    # Woke at the original window, found the move, slept to the new one and
    # woke there last of all.
    assert at(22, 40) in clock.woke

    assert clock.woke[-1] == at(23, 10)


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
    assert publication_commands(world, stage="feed") == []

    assert publication_commands(world, stage="page") == []


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

    assert publication_commands(world, stage="page") == []


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

    for entry_point in validator.LIFECYCLE_ENTRY_POINTS:
        assert entry_point in lifecycle, entry_point

    assert "cron: '37 9 * * *'" in lifecycle

    # The entry points the controller owns belong to neither workflow. They
    # are forbidden in the T-20 workflow for a reason of their own, which is
    # why they are a separate list rather than more of the lifecycle's work.
    for entry_point in validator.CONTROLLER_OWNED_ENTRY_POINTS:
        assert entry_point not in lifecycle, entry_point

        assert entry_point not in T20_WORKFLOW.read_text(encoding="utf-8")


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


def test_the_reservation_leaves_the_capture_claim_clear_of_a_t20_sweep():
    """The arithmetic behind the reservation, where the two changes meet.

    The capture claim is an exclusive lock held across a whole collector
    sweep, so a sweep that overruns delays the next one rather than racing
    it. Before the advisory windows there was only ever one sweep per group;
    now a segment can take an advisory capture in the gap before a T-20
    capture it also owns, and the reservation is what keeps the two apart.

    Three things have to hold at once, and each is a different bound:

    * the reservation has to be a large multiple of what an advisory window
      actually costs, or a window that ran a little long would eat into it;
    * it has to stay under the twenty-five minutes between T-45 and T-20, or
      every T-45 window whose T-20 the same segment owns would be deferred
      and the T-45 release would reliably arrive after the T-20 one;
    * and the measured cost has to fit inside the capture grace window, which
      is the real bound on the claim. The reservation gates advisory
      *serving*; an advisory *capture* may be taken at any time strictly
      before the T-20 capture, because another game's T-45 can fall on the
      same minute as this game's T-20. The claim is therefore held for one
      collector invocation at worst, and the T-20 sweep behind it still has
      to find the claim free inside its grace or refuse.

    Pinned as arithmetic rather than left as prose because the third bound is
    the one neither branch could state on its own.
    """
    reservation_seconds = controller.T20_RESERVATION_MINUTES * 60
    measured = controller.ADVISORY_SWEEP_MEASURED_SECONDS

    assert reservation_seconds >= 4 * measured, (
        f"a {controller.T20_RESERVATION_MINUTES}-minute reservation is only "
        f"{reservation_seconds / measured:.1f}x the measured {measured}s "
        "advisory cycle, which is not enough room for a stage that runs long"
    )

    assert controller.T20_RESERVATION_MINUTES < 25, (
        "a reservation of twenty-five minutes or more defers every T-45 "
        "window whose T-20 the same segment owns, because that is the gap "
        "between the two by definition"
    )

    # Read off the parser rather than restated, because this is the value the
    # controller actually hands the collector as --grace-minutes.
    grace_seconds = (
        controller.parse_args(
            ["--data-root", "/tmp", "--production-sha", "0" * 40]
        ).grace_minutes
        * 60
    )

    assert measured < grace_seconds, (
        f"an advisory sweep measured at {measured}s does not clear the "
        f"{grace_seconds}s capture grace, so a T-20 sweep waiting on the "
        "claim behind it could refuse with its window uncaptured"
    )


# ----------------------------------------------------------------------
# captured, and never published
# ----------------------------------------------------------------------
#
# The 2026-10-23 defect, in the controller's own terms. A run captured the
# 01:40Z window for three games and died in serving. The capture-run log
# said the window was done, because that log is the only place completion
# was ever recorded, so the next run's probe saw it under
# already_completed_windows, had nothing pending, and exited a green no-op.
# The three games were captured, marked complete, never priced, and absent
# from the published feed without anything saying so.


def captured_but_unpublished(tmp_path, *, clock, games):
    """The state a run that captured and then died in serving leaves.

    Produced by actually running the controller and actually failing the
    serving step, rather than by writing the capture by hand: the whole
    claim is about what the real sequence leaves on disk.
    """
    world = World(
        tmp_path, schedules=[games], clock=clock, serving_outcome="FAILED"
    )

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused):
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    return world


def as_the_entry_point_sees_it(monkeypatch, world: World) -> None:
    """Point the module-level collaborators at this world.

    ``run`` and ``main`` build the controller themselves, so a test that
    goes in through the entry point has to replace the schedule source and
    the subprocess runner where the controller looks them up.
    """
    monkeypatch.setattr(controller, "_bdl_games", world.schedule_source)

    monkeypatch.setattr(controller, "_run_subprocess", world)


def test_a_capture_that_died_in_serving_is_visible_to_the_next_run(
    tmp_path, monkeypatch
):
    """The state the probe could not see, and now reports by name.

    ``already_completed_windows`` said the window was done and it was the
    only thing said about it. The probe now distinguishes a window a
    release carries from one that was merely captured, reports the second
    under its own key, and asks for the full job rather than exiting a
    no-op -- which is the only way the recovery below can ever be started
    by the schedule.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]

    world = captured_but_unpublished(tmp_path, clock=clock, games=games)

    as_the_entry_point_sees_it(monkeypatch, world)

    probe = controller.run(
        controller.parse_args(
            [
                "--data-root",
                str(world.data_root),
                "--production-sha",
                PRODUCTION_SHA,
                "--segment-start-utc",
                SEGMENT_2[0].isoformat(),
                "--segment-end-utc",
                SEGMENT_2[1].isoformat(),
                "--slate-date",
                SLATE,
                "--probe-only",
            ]
        )
    )

    assert probe["already_completed_windows"] == ["101:T-20m", "102:T-20m"]

    assert probe["pending_windows"] == []

    # The distinction that did not exist: captured is not served.
    assert probe["captured_unserved_windows"] == [
        "101:T-20m",
        "102:T-20m",
    ]

    assert probe["final_outcome"] == "T20_WORK_IN_SEGMENT"


def test_the_probe_asks_for_the_job_when_only_recovery_is_outstanding(
    tmp_path, monkeypatch
):
    """``t20_work`` is what the schedule reads, and recovery is work.

    The workflow the production branch runs is unchanged: the probe already
    decides whether the heavy job starts, and this teaches it that an
    unpublished capture is a reason to start it.
    """
    clock = Clock(at(19, 25))

    world = captured_but_unpublished(
        tmp_path, clock=clock, games=[game(101, at(23, 0))]
    )

    as_the_entry_point_sees_it(monkeypatch, world)

    output = tmp_path / "github_output"

    exit_code = controller.main(
        [
            "--data-root",
            str(world.data_root),
            "--production-sha",
            PRODUCTION_SHA,
            "--segment-start-utc",
            SEGMENT_2[0].isoformat(),
            "--segment-end-utc",
            SEGMENT_2[1].isoformat(),
            "--slate-date",
            SLATE,
            "--probe-only",
            "--github-output",
            str(output),
        ]
    )

    assert exit_code == 0

    written = output.read_text(encoding="utf-8")

    assert "t20_work=true" in written

    assert "pending_windows=0" in written

    assert "captured_unserved_windows=1" in written


def test_a_later_run_recovers_the_capture_without_recapturing_it(tmp_path):
    """Recovery is serving and publication, and nothing else.

    No collector invocation, so no snapshot is taken, re-timed or
    relabelled; the capture-run log is byte-identical to what the failed
    run left. What runs is the selection pricing performs, then the same
    serving and publication entry points, over the capture that already
    exists.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]

    world = captured_but_unpublished(tmp_path, clock=clock, games=games)

    capture_log = (
        world.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"
    )

    before = capture_log.read_bytes()

    recovery_clock = Clock(at(23, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    # Nothing captured, and nothing slept for: the window is in the past.
    assert collector_commands(recovery) == []

    assert recovery_clock.slept == []

    assert capture_log.read_bytes() == before

    assert [
        record["window_ids"] for record in receipt["recovered_windows"]
    ] == [["101:T-20m", "102:T-20m"]]

    # Both games reached a release, from the capture the failed run made.
    assert recovery.published == [[101, 102]]

    assert receipt["captured_unserved_windows"] == []


def test_the_recovered_capture_is_the_one_the_failed_run_took(tmp_path):
    """Recovery may not fabricate, re-time or relabel a snapshot.

    The bundle the recovery serves is asserted to be the same capture id,
    at the same captured_at, under the same window id as the one the failed
    run wrote -- which is the invariant that separates recovering a window
    from quietly replacing it with a later snapshot.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = captured_but_unpublished(tmp_path, clock=clock, games=games)

    # The row naming the T-20 window, not the first row in the log. The
    # segment takes its advisory captures before it reaches the T-20 one, so
    # the log opens with T-180 and the first line stopped being the capture
    # this test is about.
    rows = [
        json.loads(line)
        for line in (world.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    captured = next(
        row
        for row in rows
        if "101:T-20m" in (row.get("payload") or {}).get("window_ids", [])
    )

    recovery_clock = Clock(at(23, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    bundle = receipt["recovered_windows"][0]["verified_bundles"][0]

    assert bundle["window_id"] == "101:T-20m"

    assert bundle["captured_at_utc"] == captured["captured_at"]

    assert bundle["capture_id"] == captured["payload"]["capture_id"]

    # And the capture is still the T-20 capture it always was: taken at the
    # window, long before the run that published it.
    assert bundle["captured_at_utc"] == at(22, 40).isoformat()


def test_a_refusal_after_a_capture_names_what_it_is_abandoning(
    tmp_path, monkeypatch
):
    """The receipt the losing run of 2026-10-23 should have written.

    Serving fails with the capture already on disk. The refusal stands and
    keeps its own outcome -- nothing is downgraded -- but the receipt and
    the job summary now say which windows exist and are carried by no
    release, where before the operator had to read the capture log against
    the published feed to find out.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]

    world = captured_but_unpublished(tmp_path, clock=clock, games=games)

    retry_clock = Clock(at(23, 50))

    retry = World(
        tmp_path,
        schedules=[games],
        clock=retry_clock,
        serving_outcome="FAILED",
    )

    write_morning_receipt(retry)

    as_the_entry_point_sees_it(monkeypatch, retry)

    receipt_path = tmp_path / "receipt.json"

    summary_path = tmp_path / "summary.md"

    exit_code = controller.main(
        [
            "--data-root",
            str(retry.data_root),
            "--production-sha",
            PRODUCTION_SHA,
            "--segment-start-utc",
            SEGMENT_2[0].isoformat(),
            "--segment-end-utc",
            SEGMENT_2[1].isoformat(),
            "--slate-date",
            SLATE,
            "--receipt-path",
            str(receipt_path),
            "--summary-path",
            str(summary_path),
        ]
    )

    assert exit_code == 1

    failure = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert failure["final_outcome"] == controller.OUTCOME_FAILED

    assert failure["captured_unserved_windows"] == [
        "101:T-20m",
        "102:T-20m",
    ]

    assert "captured but unserved: 101:T-20m, 102:T-20m" in (
        summary_path.read_text(encoding="utf-8")
    )

    # The capture is still there to be recovered from, which is the whole
    # reason for naming it.
    assert world.snapshot_dir.joinpath(
        "capture_runs", f"{SLATE}.jsonl"
    ).exists()


def test_a_completed_segment_with_an_unserved_capture_is_not_a_success(
    tmp_path, monkeypatch
):
    """The post-condition behind the two paths above.

    Every path that publishes also records, so this ending should be
    unreachable -- which is exactly why it is asserted rather than assumed.
    A publication that forgot to record would otherwise reach the end of a
    segment, exit 0, and leave the next run reading the window as
    unpublished for the rest of the slate's life.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    monkeypatch.setattr(
        controller.T20Controller,
        "_record_served_windows",
        lambda self, **kwargs: None,
    )

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED

    assert receipt["captured_unserved_windows"] == ["101:T-20m"]

    assert (
        controller.OUTCOME_CAPTURED_NOT_SERVED
        not in controller.OUTCOMES_THAT_SUCCEED
    )


def test_an_outcome_that_is_not_a_success_exits_non_zero(
    tmp_path, monkeypatch
):
    """``OUTCOMES_THAT_SUCCEED`` has always said which endings succeed.

    Nothing consulted it, so a segment that captured three games and
    published none of them exited 0 all the same.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    recovery_clock = Clock(at(23, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    as_the_entry_point_sees_it(monkeypatch, recovery)

    monkeypatch.setattr(
        controller.T20Controller,
        "_record_served_windows",
        lambda self, **kwargs: None,
    )

    receipt_path = tmp_path / "receipt.json"

    exit_code = controller.main(
        [
            "--data-root",
            str(recovery.data_root),
            "--production-sha",
            PRODUCTION_SHA,
            "--segment-start-utc",
            SEGMENT_2[0].isoformat(),
            "--segment-end-utc",
            SEGMENT_2[1].isoformat(),
            "--slate-date",
            SLATE,
            "--receipt-path",
            str(receipt_path),
        ]
    )

    assert exit_code == 1

    failure = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert failure["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED


def test_a_segment_that_cannot_publish_a_capture_it_made_fails(tmp_path):
    """The same ending reached without a refusal, which is the harder case.

    Here the capture succeeds, serving succeeds and the feed reports that
    it published nothing. The controller refuses that today; what is new is
    that the window it captured is recorded as unserved rather than as
    completed, so the next run can still recover it.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = World(
        tmp_path, schedules=[games], clock=clock, feed_outcome="NOTHING"
    )

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused):
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert served_records(world) == []

    assert controller.served_window_ids(
        world.data_root
        / "processed"
        / "t20_served_windows"
        / f"{SLATE}.jsonl"
    ) == set()


def test_recovery_is_safe_to_run_twice(tmp_path):
    """A re-serve is idempotent: the second run finds nothing to do."""
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    first_clock = Clock(at(23, 50))

    first = World(tmp_path, schedules=[games], clock=first_clock)

    write_morning_receipt(first)

    build(
        first,
        clock=first_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    second_clock = Clock(at(23, 55))

    second = World(tmp_path, schedules=[games], clock=second_clock)

    write_morning_receipt(second)

    receipt = build(
        second,
        clock=second_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_NO_WINDOWS

    assert second.commands == []

    assert second.published == []

    # One release, from the first recovery, and one row recording it.
    assert len(served_records(first)) == 1


def test_publication_is_recorded_on_publication_and_names_the_release(
    tmp_path,
):
    """What the durable row has to carry for a later run to trust it."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    records = served_records(world)

    assert len(records) == 1

    row = records[0]

    assert row["record_type"] == controller.SERVED_RECORD_TYPE

    assert row["window_ids"] == ["101:T-20m"]

    assert row["publication_outcome"] == "PUBLISHED"

    assert row["publication_mode"] == "canary"

    assert row["release_id"] == "release-1"

    assert row["slate_date"] == SLATE

    assert row["production_sha"] == PRODUCTION_SHA

    assert row["offset_minutes"] == 20


def test_each_group_records_what_the_release_it_published_carries(tmp_path):
    """Two groups, two releases, and the second row names both windows.

    The slate is re-served whole after every capture, so the release the
    second group publishes carries the first group's game too. The row has
    to say so, otherwise a later run would read the first game as
    unpublished and recover something that was never lost.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(20, 0)), game(102, at(23, 0), home=3, away=4)]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    records = served_records(world)

    assert [row["window_ids"] for row in records] == [
        ["101:T-20m"],
        ["101:T-20m", "102:T-20m"],
    ]

    assert receipt["captured_unserved_windows"] == []


def test_the_summary_table_states_both_halves_of_done(tmp_path):
    """A reader of the job summary can see which windows are unpublished."""
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    recovery_clock = Clock(at(23, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    summary = controller.render(receipt)

    assert "| recovered without recapture | 101:T-20m |" in summary

    assert "| captured but unserved | none |" in summary


def test_a_torn_ledger_line_does_not_hide_the_rows_before_it(tmp_path):
    """The log is append-only and read the way the capture log is read."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock
    )

    path = write_served_record(world, window_ids=["101:T-20m"])

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"record_type": "t20_windows_serv')

    assert controller.served_window_ids(path) == {"101:T-20m"}
