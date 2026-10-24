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

#: The three prop routes that read Gate 3 role state, and the only ones a
#: role-state failure may cost. Imported rather than typed out in the test
#: that matters; repeated here only so the fake world can build a receipt.
GATE3_ROUTES = ("assists", "points_assists", "points_rebounds")


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
        live_refusal: str | None = None,
        live_page_returncode: int = 0,
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

        #: When set, every live refresh refuses with this message and exits
        #: nonzero, which is the state the repository is actually in until
        #: both approval variables are set in the production environment.
        self.live_refusal = live_refusal

        self.live_page_returncode = live_page_returncode

        #: What the single live destination is serving, as the fake chain
        #: leaves it. One pointer, one window, one digest -- which is the
        #: design, so a test can assert the pointer never went backwards.
        self.live_release: str | None = None

        self.live_offset: int | None = None

        self.live_rows: str | None = None

        #: Every live refresh asked for, in order, as (offset, rows digest).
        self.live_refreshes: list[tuple[int, str]] = []

        #: Every live page publication asked for, in order.
        self.live_pages: list[str] = []

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

        #: Every slate date the serving-window gates were asked about, in
        #: order, so a test can say the gates were consulted before the
        #: slate was served rather than only that nothing blew up.
        self.window_gate_calls: list[str] = []

        #: Make the gates refuse, which is what a capture that does not
        #: satisfy the live serving protocol looks like to the controller.
        self.window_gates_refuse: bool = False

        #: Game ids whose role state is not ready, which is what missing
        #: lineup records look like to the controller: the three Gate 3
        #: routes are rejected for those games and the window still serves.
        self.window_gates_degrade: tuple[int, ...] = ()

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

        if script.endswith("run_progressive_live_refresh.py"):
            return self._refresh_live(command)

        if script.endswith("assert_serving_window_gates.py"):
            return self._window_gates(command)

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

    def _window_gates(self, command):
        slate = self._value(command, "--slate-date")

        self.window_gate_calls.append(slate)

        status = Path(self._value(command, "--status-path"))

        status.parent.mkdir(parents=True, exist_ok=True)

        if self.window_gates_refuse:
            outcome = "SERVING_WINDOW_GATES_REFUSED"
        elif self.window_gates_degrade:
            outcome = "SERVING_WINDOW_ROUTES_DEGRADED"
        else:
            outcome = "SERVING_WINDOW_GATES_PASS"

        degraded = {
            str(game): list(GATE3_ROUTES)
            for game in (self.window_gates_degrade or ())
        }

        status.write_text(
            json.dumps(
                {
                    "degraded_reasons": {
                        game: ["the capture carries no lineup records"]
                        for game in degraded
                    },
                    "degraded_routes": degraded,
                    "outcome": outcome,
                    "route_filter_reason": "gate3_role_state_unavailable",
                    "slate_date": slate,
                }
            ),
            encoding="utf-8",
        )

        if self.window_gates_refuse:
            return self._completed(
                command,
                returncode=1,
                stdout="t20_protocol_compatible failed for " + slate,
            )

        return self._completed(command)

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
                            "release_id": f"release-{len(self.published)}",
                            # The digest the live half is required to carry.
                            # In the real receipt it is the digest of the rows
                            # payload; here it is a function of the served
                            # artifact, which is the property that matters:
                            # the same priced bytes digest the same and
                            # different ones do not.
                            "rows_sha256": str(
                                (
                                    served.get("priced_market_artifact") or {}
                                ).get("sha256")
                                or ""
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

            return self._completed(command)

        mode = self._value(command, "--mode")

        if mode == "live":
            self.live_pages.append(
                self._value(command, "--feed-receipt")
            )

            if self.live_page_returncode != 0:
                receipt_path.write_text(
                    json.dumps(
                        {
                            "message": "the live page stage was made to fail",
                            "mode": mode,
                            "outcome": "PUBLICATION_STAGE_FAILED",
                            "slate_date": self._value(
                                command, "--slate-date"
                            ),
                        }
                    ),
                    encoding="utf-8",
                )

                return self._completed(
                    command, returncode=self.live_page_returncode
                )

        receipt_path.write_text(
            json.dumps(
                {
                    "outcome": "PUBLISHED",
                    "mode": mode,
                    "slate_date": self._value(command, "--slate-date"),
                    "page_sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )

        return self._completed(command)

    def _refresh_live(self, command):
        """What ops/run_progressive_live_refresh.py writes, in receipt shape.

        Enough of that module's decision for a controller test to be about
        the controller: the ordering rule, the idempotent case, the
        establishment case and the refusal, all driven off the canary feed
        receipt the controller actually handed over rather than off the
        controller's own intention.

        Its real rules are not reimplemented here and are not tested here.
        The attestation read back off the canary destination, the three-way
        digest match, the publisher's own gates, the post-tip guard and the
        durable ledger are exercised against the real publisher over real
        local destinations in tests/test_nba_progressive_live_refresh.py.
        """
        offset = int(self._value(command, "--snapshot-offset-minutes"))

        slate_date = self._value(command, "--slate-date")

        receipt_path = Path(self._value(command, "--receipt-path"))

        receipt_path.parent.mkdir(parents=True, exist_ok=True)

        canary = json.loads(
            Path(
                self._value(command, "--canary-feed-receipt")
            ).read_text(encoding="utf-8")
        )

        rows_sha256 = str((canary.get("release") or {}).get("rows_sha256") or "")

        self.live_refreshes.append((offset, rows_sha256))

        def refuse(outcome: str, message: str, *, returncode: int = 1):
            receipt_path.write_text(
                json.dumps(
                    {
                        "live_outcome": outcome,
                        "message": message,
                        "mode": "live",
                        "observed_live_release_id": self.live_release or "",
                        "offset_minutes": offset,
                        "outcome": (
                            "LIVE_REFRESH_REFUSED"
                            if returncode
                            else "LIVE_REFRESH_DECLINED"
                        ),
                        "slate_date": slate_date,
                        "stage": "live",
                    }
                ),
                encoding="utf-8",
            )

            return self._completed(command, returncode=returncode)

        if self.live_refusal is not None:
            return refuse(
                controller.LIVE_OUTCOME_REFUSED, self.live_refusal
            )

        if self.live_offset is not None and offset > self.live_offset:
            # Exits zero. The ordering rule declining a window is the rule
            # working, and on a staggered slate it is the ordinary case.
            return refuse(
                controller.LIVE_OUTCOME_ALREADY_CLOSER_TO_TIP,
                f"the T-{self.live_offset} release is already live and this "
                f"is the T-{offset} refresh, which is an earlier capture "
                "window",
                returncode=0,
            )

        previous = self.live_release or ""

        if rows_sha256 and rows_sha256 == self.live_rows:
            outcome = controller.LIVE_OUTCOME_ALREADY_CURRENT

            release_id = previous

        else:
            outcome = (
                controller.LIVE_OUTCOME_ESTABLISHED
                if self.live_release is None
                else controller.LIVE_OUTCOME_REPLACED
            )

            release_id = f"nba_prop_quant__{slate_date}__{rows_sha256[:16]}"

            self.live_release = release_id

            self.live_rows = rows_sha256

        self.live_offset = offset

        receipt_path.write_text(
            json.dumps(
                {
                    "live_outcome": outcome,
                    "mode": "live",
                    "offset_minutes": offset,
                    "outcome": "PUBLISHED",
                    "previous_release_id": previous,
                    "reason": (
                        f"the T-{offset} release {release_id} is live"
                    ),
                    "release": {
                        "release_id": release_id,
                        "rows_sha256": rows_sha256,
                    },
                    "slate_date": slate_date,
                    "stage": "live",
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
                            ),
                            "rows_sha256": digest,
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


def served_window_ids_for(
    world: World, *, slate_date: str = SLATE
) -> set[str]:
    """Every T-20 window the durable ledger says a release carries."""
    return controller.served_window_ids(
        world.data_root
        / "processed"
        / "t20_served_windows"
        / f"{slate_date}.jsonl"
    )


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


def publication_commands(
    world: World, *, stage: str, mode: str = "canary"
) -> list[list[str]]:
    """The T-20 publication invocations at one stage, in one mode.

    An advisory release is addressed by naming its destination; the T-20
    release is published by naming none, because its destination is the
    mode's default. Filtering on that is what keeps every assertion in this
    module about what production published an assertion about T-20.

    The mode is filtered for the same reason, one layer along. A live page is
    also published without a destination name -- live mode's default page
    destination is the served root -- so without this every assertion about
    what the canary published would quietly have become an assertion about
    the canary and the live page together.
    """
    return [
        command
        for command in world.commands
        if str(command[1]).endswith("run_wizardofodds_publication.py")
        and "--stage" in command
        and command[command.index("--stage") + 1] == stage
        and command[command.index("--mode") + 1] == mode
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


def test_a_segment_owning_no_t20_window_still_serves_its_advisory_windows(
    tmp_path,
):
    """The early exit is about all three kinds of work, not just the T-20 one.

    A segment that owns no T-20 window used to be the cheap no-op the probe
    exists to make cheap, and two changes now have a claim on that branch:
    one adds a captured-but-unserved window as work, the other adds advisory
    windows. The afternoon segment of a normal slate is the case where only
    the second applies -- it owns the first T-180 window of the evening and
    nothing else -- and dropping that term from the condition would make it
    exit green with the window uncaptured and nothing said about it.

    A game tipping at 22:00 puts T-180 at 19:00, inside the 14:25 segment,
    and leaves T-90, T-45 and T-20 to the 19:25 one.
    """
    clock = Clock(at(14, 25))

    games = [game(101, at(22, 0))]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_1[0], segment_end=SEGMENT_1[1]
    ).run()

    assert receipt["t20_windows_in_segment"] == []

    assert receipt["advisory_windows_in_segment"] == ["101:T-180m"]

    assert receipt["final_outcome"] != controller.OUTCOME_NO_WINDOWS

    # Captured at its own offset, and no T-20 capture was taken.
    assert [
        command[command.index("--offset-minutes") + 1]
        for command in capture_commands(world)
    ] == ["180"]

    assert collector_commands(world) == []

    # And served and published, through the advisory entry point.
    assert [offset for offset, _ in world.advisory_served] == [180]

    assert receipt["captures_completed"] == 0


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


def test_an_advisory_only_segment_degrades_when_the_morning_proves_nothing():
    """A segment with nothing but advisory work must not turn the evening red.

    The first half of the readiness guard. Before the advisory windows a
    segment owning no T-20 window was a green no-op, and an advisory window
    is a reading aid: a missing morning receipt is recorded as the advisory
    failure it is and the segment still ends successfully, so the exit code
    keeps saying what it said about production.
    """
    assert (
        controller.OUTCOME_SEGMENT_COMPLETE_ADVISORY_DEGRADED
        in controller.OUTCOMES_THAT_SUCCEED
    )


def test_a_readiness_refusal_is_not_degraded_while_a_capture_is_unserved(
    tmp_path,
):
    """The other half, and the one the two changes had to agree on.

    An unserved window is a T-20 window production is still missing, and
    recovering it publishes a production release off a capture that already
    exists. That needs the same verified incumbent a fresh capture does. So
    the refusal stands here rather than being filed as an advisory failure:
    degrading it would end the segment at an outcome that exits zero, with
    the window still uncarried by any release and nothing saying so.

    The segment has no pending T-20 window and no pending advisory window --
    every one of the four was captured by the run that then died in serving
    -- so the unserved window is the only thing keeping this out of the
    degraded branch.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    recovery_clock = Clock(at(23, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    # The failing run wrote one; removing it is how this segment gets the
    # refusal under test, rather than by omitting a call whose effect the
    # shared data root would keep anyway.
    (
        recovery.data_root / "processed" / "incumbent_serving" / f"{SLATE}.json"
    ).unlink()

    segment = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    )

    plan = segment.plan()

    assert plan["pending_rows"] == []

    assert plan["advisory_pending_rows"] == []

    assert plan["captured_unserved_windows"] == ["101:T-20m"]

    with pytest.raises(controller.ControllerRefused) as error:
        segment.run()

    assert error.value.outcome == controller.OUTCOME_MORNING_NOT_READY

    # Nothing was served or published off an unproven incumbent on the way.
    assert recovery.commands == []


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


def test_live_publication_is_decided_in_the_commit_and_gated_outside_it():
    """Case 20, restated for what is now true.

    This used to assert that live publishing was DISABLED in the committed
    switch, and that was the right assertion while the decision had not been
    taken. The decision has now been taken -- the owner asked for the early
    windows to publish live on the reviewed progression -- so the state is
    ENABLED and asserting otherwise would be asserting against the decision.

    What replaces it is stronger rather than weaker, because it pins the part
    the old assertion was really protecting: that a commit cannot publish
    anything. The switch is one of three conditions; the other two are
    variables in the wizardofodds-production GitHub environment, they cannot
    be set from this repository at all, and with the committed state as it is
    and the environment silent the publisher still refuses. That is asserted
    here against the real switch and the real gate rather than described.

    The canary stays the only mode a *schedule* can reach: the controller has
    no mode option, its own publication mode is canary, and the one live
    invocation it makes is to the refresh entry point, which has no mode
    argument either and is reachable only after a canary of the same bytes.
    """
    switch = json.loads(
        (PROJECT / "ops" / "wizardofodds_publishing_switch.json").read_text(
            encoding="utf-8"
        )
    )

    assert switch["modes"]["live"]["state"] == "ENABLED"

    assert switch["modes"]["canary"]["state"] == "ENABLED"

    # The decision is recorded, and it is not the placeholder it replaced.
    token = switch["modes"]["live"]["approval_token"]

    assert token and token != "no-approval-has-been-granted"

    # And it opens nothing. The gate is the committed state *and* both
    # variables, read here through the publisher's own function with the
    # environment as a runner without them would have it.
    publisher = load_ops("publish_incumbent_to_wizardofodds")

    read = publisher.read_publishing_switch(
        publisher.MODE_LIVE, project_root=PROJECT, environment={}
    )

    assert read.state == publisher.STATE_ENABLED

    assert read.enabled is False

    assert sorted(read.blockers) == [
        f"{publisher.PUBLISH_ENV_VAR} does not match the switch state",
        (
            f"{publisher.PUBLISH_APPROVAL_ENV_VAR} does not match the "
            "approval token"
        ),
    ]

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publisher.assert_mode_enabled(read)

    for variable in (
        publisher.PUBLISH_ENV_VAR,
        publisher.PUBLISH_APPROVAL_ENV_VAR,
    ):
        assert variable in str(refusal.value), variable

    # Half the approval is still no approval.
    for carried in (
        {publisher.PUBLISH_ENV_VAR: publisher.STATE_ENABLED},
        {publisher.PUBLISH_APPROVAL_ENV_VAR: token},
        {
            publisher.PUBLISH_ENV_VAR: publisher.STATE_ENABLED,
            publisher.PUBLISH_APPROVAL_ENV_VAR: "something-else",
        },
    ):
        assert (
            publisher.read_publishing_switch(
                publisher.MODE_LIVE,
                project_root=PROJECT,
                environment=carried,
            ).enabled
            is False
        ), carried

    assert controller.PUBLICATION_MODE == "canary"

    source = ENTRY_POINT.read_text(encoding="utf-8")

    assert '"--mode"' in source

    assert "PUBLICATION_MODE" in source

    # The controller never spells the forbidden flags as one string, and
    # never names either approval variable: it does not read them, test them
    # or pass them on, and the publisher takes them out of the environment.
    for token in (*validator.PUBLICATION_ACTIVATION_FLAGS, "NBA_PROP_WOO"):
        assert token not in source, token

    # The workflow carries the two variables and decides nothing. Checked
    # through the validator's own rule rather than by a substring scan, which
    # is what that rule exists for.
    assert validator.APPROVAL_CARRYING_WORKFLOW == T20_WORKFLOW.name

    workflow = yaml.safe_load(T20_WORKFLOW.read_text(encoding="utf-8"))

    assert (
        validator.activation_problems(
            T20_WORKFLOW.name, workflow.get("jobs") or {}
        )
        == []
    )

    text = T20_WORKFLOW.read_text(encoding="utf-8")

    for flag in validator.PUBLICATION_ACTIVATION_FLAGS:
        assert flag not in text, flag

    # Both halves, bound from the environment's own vars and from nowhere
    # else. Both from vars rather than one from each, because neither value
    # is secret -- one is the literal ENABLED and the other is committed in
    # plaintext in the switch -- so there is one kind of thing to set, under
    # the names the switch's own what_enabling_live_requires uses.
    for variable in validator.PUBLICATION_APPROVAL_VARIABLES:
        assert f"{variable}: ${{{{ vars.{variable} }}}}" in text, variable


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
# the progressive live refresh, as the controller drives it
# ----------------------------------------------------------------------


def live_commands(
    world: World, *, offset: int | None = None
) -> list[list[str]]:
    """Every live refresh invocation, optionally narrowed to one window."""
    return [
        command
        for command in world.commands
        if str(command[1]).endswith("run_progressive_live_refresh.py")
        and (
            offset is None
            or int(command[command.index("--snapshot-offset-minutes") + 1])
            == offset
        )
    ]


def live_records(receipt: dict, *, offset: int | None = None) -> list[dict]:
    """The live refresh records, optionally narrowed to one window."""
    return [
        record
        for record in receipt["live_refreshes"]
        if offset is None or int(record["offset_minutes"]) == offset
    ]


def test_each_window_refreshes_live_after_its_own_canary_published(tmp_path):
    """The chain, in order, over one segment that owns all four windows.

    T-180 establishes the live release, T-90 replaces it, T-45 replaces that
    and T-20 replaces that as the final pregame version. The ordering claim
    is asserted on the live pointer the fake destination ends up holding and
    on the sequence of offsets that moved it, not on the controller's
    intention: a refresh is only recorded as having moved the pointer when
    the refresh itself says it did.
    """
    clock = Clock(at(14, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world,
        clock=clock,
        segment_start=SEGMENT_1[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    # Four windows, each refreshed once, closest to tip last.
    assert [int(command[command.index("--snapshot-offset-minutes") + 1])
            for command in live_commands(world)] == [180, 90, 45, 20]

    assert [record["offset_minutes"] for record in receipt["live_refreshes"]] == [
        180,
        90,
        45,
        20,
    ]

    assert [record["outcome"] for record in receipt["live_refreshes"]] == [
        controller.LIVE_OUTCOME_ESTABLISHED,
        controller.LIVE_OUTCOME_REPLACED,
        controller.LIVE_OUTCOME_REPLACED,
        controller.LIVE_OUTCOME_REPLACED,
    ]

    # Each one replaced the one before it, and the last is final.
    releases = [record["release_id"] for record in receipt["live_refreshes"]]

    assert len(set(releases)) == 4

    assert [
        record["previous_release_id"] for record in receipt["live_refreshes"]
    ] == ["", *releases[:-1]]

    assert receipt["live_refreshes"][-1]["final_pregame_window"] is True

    assert [
        record["final_pregame_window"]
        for record in receipt["live_refreshes"][:-1]
    ] == [False, False, False]

    # One live destination, and it ends the evening naming the T-20 release.
    assert world.live_release == releases[-1]

    assert world.live_offset == 20

    assert receipt["live_release_id"] == releases[-1]

    assert receipt["live_refreshes_with_faults"] == []

    assert receipt["progressive_live_offsets"] == [180, 90, 45, 20]


def test_a_window_refreshes_live_from_its_own_canary_and_nothing_else(tmp_path):
    """Each live refresh is handed the receipt of its own canary.

    The equality the whole chain rests on, checked at the point where it
    could be got wrong: the destination name and the feed receipt the
    controller passes to the refresh are the ones the canary stage it just
    ran actually used, so no window can be authorised by another window's
    canary.
    """
    clock = Clock(at(14, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    build(
        world,
        clock=clock,
        segment_start=SEGMENT_1[0],
        segment_end=SEGMENT_2[1],
    ).run()

    expected = {
        **controller.ADVISORY_FEED_DESTINATIONS,
        20: controller.T20_FEED_DESTINATION,
    }

    for command in live_commands(world):
        offset = int(command[command.index("--snapshot-offset-minutes") + 1])

        assert (
            command[command.index("--canary-destination-name") + 1]
            == expected[offset]
        )

        # The live destinations are named explicitly and are the single pair.
        assert (
            command[command.index("--live-feed-destination-name") + 1]
            == controller.LIVE_FEED_DESTINATION
        )

        assert (
            command[command.index("--live-page-destination-name") + 1]
            == controller.LIVE_PAGE_DESTINATION
        )

        # The canary receipt handed over is the one that window's canary
        # stage wrote, which is the file in that window's own stage
        # directory, and the digest in it is the digest the live release has
        # to carry.
        receipt = json.loads(
            Path(
                command[command.index("--canary-feed-receipt") + 1]
            ).read_text(encoding="utf-8")
        )

        assert receipt["outcome"] == "PUBLISHED"

        assert receipt["mode"] == "canary"

        assert receipt["release"]["rows_sha256"]

        assert (offset, receipt["release"]["rows_sha256"]) in (
            world.live_refreshes
        )

        # And the tip of the window it is for, so the post-tip guard has a
        # subject. Never defaulted and never absent.
        assert "--window-tip-utc" in command

        for binding in world._values(command, "--window-tip-utc"):
            window_id, _, tip = binding.partition("=")

            assert window_id.endswith(f":T-{offset}m"), binding

            assert datetime.fromisoformat(tip) == at(23, 0)

        # And the capture each window's prices were selected from, so the
        # composite manifest can name the snapshot behind that game's card.
        # Carried out of the bundle selection the pricing path itself
        # performed rather than selected again here, and never defaulted:
        # the refresh refuses an advancing window it cannot name a capture
        # for, because a card on the served root whose snapshot nobody can
        # find again is worse than no card.
        assert "--window-capture" in command

        captures = world._values(command, "--window-capture")

        assert len(captures) == len(
            world._values(command, "--window-tip-utc")
        )

        for binding in captures:
            window_id, _, identity = binding.partition("=")

            assert window_id.endswith(f":T-{offset}m"), binding

            capture_id, separator, captured_at = identity.partition("@")

            assert separator, binding

            assert capture_id, binding

            assert datetime.fromisoformat(captured_at) < at(23, 0), binding


def test_every_window_names_the_capture_the_pricing_path_itself_selected(
    tmp_path,
):
    """The capture identity is carried, never invented and never re-derived.

    The composite manifest records the snapshot behind every game's card, and
    this controller is the only thing in the chain that knows it: the live
    refresh is handed already-published bytes and has no capture log to read.
    So what it passes has to be exactly what the pricing path selected, and
    that is checked here against ``select_capture_bundle`` run independently
    over the same log -- not against a value this controller recorded.
    """
    clock = Clock(at(14, 25))

    games = [
        game(101, at(23, 0)),
        game(102, at(23, 30), home=3, away=4),
    ]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    build(
        world,
        clock=clock,
        segment_start=SEGMENT_1[0],
        segment_end=SEGMENT_2[1],
    ).run()

    from nba_prop_quant.prospective_snapshot import select_capture_bundle

    commands = live_commands(world)

    assert commands

    seen = 0

    for command in commands:
        offset = int(command[command.index("--snapshot-offset-minutes") + 1])

        for binding in world._values(command, "--window-capture"):
            window_id, _, identity = binding.partition("=")

            game_id = int(window_id.split(":")[0])

            bundle = select_capture_bundle(
                world.snapshot_dir,
                date=SLATE,
                game_id=game_id,
                offset_minutes=offset,
            )

            assert identity == f"{bundle.capture_id}@{bundle.captured_at}", (
                binding
            )

            assert bundle.window_id == window_id

            seen += 1

    assert seen == len(games) * len(
        controller.PROGRESSIVE_LIVE_OFFSETS_MINUTES
    )


def test_the_live_page_is_built_from_the_live_feed_receipt(tmp_path):
    """The page a reader opens is rebuilt from the release that just went live.

    Pinned to the live refresh's own receipt and not to the canary's, which
    is what makes it the page for the release live is serving rather than a
    page for a preview with the same numbers in it.
    """
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    pages = publication_commands(world, stage="page", mode="live")

    assert len(pages) == len(live_commands(world))

    for command in pages:
        pinned = Path(command[command.index("--feed-receipt") + 1])

        assert pinned.name == "live_refresh.json"

        receipt = json.loads(pinned.read_text(encoding="utf-8"))

        assert receipt["mode"] == "live"

        assert receipt["outcome"] == "PUBLISHED"

        assert receipt["release"]["release_id"]

    # And the page follows its own feed refresh rather than preceding it.
    order = [
        index
        for index, command in enumerate(world.commands)
        if str(command[1]).endswith("run_progressive_live_refresh.py")
        or (
            str(command[1]).endswith("run_wizardofodds_publication.py")
            and "--mode" in command
            and command[command.index("--mode") + 1] == "live"
        )
    ]

    for first, second in pairwise(order):
        earlier = world.commands[first]

        later = world.commands[second]

        if str(earlier[1]).endswith("run_progressive_live_refresh.py"):
            assert str(later[1]).endswith("run_wizardofodds_publication.py")

        else:
            assert str(later[1]).endswith("run_progressive_live_refresh.py")


def test_a_refused_live_refresh_degrades_the_segment_without_failing_it(
    tmp_path,
):
    """The state this repository is actually in, and what it costs.

    Until both approval variables are set in the wizardofodds-production
    environment every live refresh refuses. A slate that was captured,
    served and published to the canary must not be a red job because of that,
    and it must not be a silently green one either: the outcome is its own
    word, the count leads the report, and the refusal each refresh wrote is
    carried verbatim so the message naming both variables reaches whoever
    reads the summary.
    """
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0))]],
        clock=clock,
        live_refusal=(
            "ops/wizardofodds_publishing_switch.json records live publishing "
            "as ENABLED and this process does not carry its approval. Both "
            "variables must be set in the wizardofodds-production GitHub "
            "environment: NBA_PROP_WOO_PUBLISH and "
            "NBA_PROP_WOO_PUBLISH_APPROVAL"
        ),
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == (
        controller.OUTCOME_SEGMENT_COMPLETE_LIVE_DEGRADED
    )

    assert receipt["final_outcome"] in controller.OUTCOMES_THAT_SUCCEED

    # The T-20 deliverable is untouched: served, published, page built.
    assert [
        record["outcome"] for record in receipt["feed_publication_outcomes"]
    ] == ["PUBLISHED"]

    assert [
        record["outcome"] for record in receipt["page_publication_outcomes"]
    ] == ["PUBLISHED"]

    # Nothing is live, and nothing claims to be.
    assert world.live_release is None

    assert receipt["live_release_id"] == ""

    assert receipt["live_refreshes_with_faults"] == receipt["live_refreshes"]

    for record in receipt["live_refreshes"]:
        assert record["outcome"] == controller.LIVE_OUTCOME_REFUSED

        for variable in (
            "NBA_PROP_WOO_PUBLISH",
            "NBA_PROP_WOO_PUBLISH_APPROVAL",
        ):
            assert variable in record["reason"], variable

    # No live page was built behind a refusal.
    assert publication_commands(world, stage="page", mode="live") == []

    # And the summary says so rather than burying it.
    summary = controller.render(receipt)

    assert "live failures" in summary

    assert controller.LIVE_OUTCOME_REFUSED in summary

    assert "NBA_PROP_WOO_PUBLISH_APPROVAL" in summary


def test_a_live_page_that_fails_is_its_own_outcome(tmp_path):
    """The feed moved and the page did not, which is worth its own word."""
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0))]],
        clock=clock,
        live_page_returncode=1,
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == (
        controller.OUTCOME_SEGMENT_COMPLETE_LIVE_DEGRADED
    )

    # Every window's page failed, and each is recorded as the page failure
    # rather than as a refusal: the feed did move.
    assert {record["outcome"] for record in receipt["live_refreshes"]} == {
        controller.LIVE_OUTCOME_PAGE_FAILED
    }

    # The release is live and the record says which one, so the gap between
    # the feed and the page is nameable rather than a mystery.
    assert receipt["live_refreshes"][-1]["release_id"] == world.live_release

    assert receipt["live_refreshes"][0]["live_outcome"] == (
        controller.LIVE_OUTCOME_ESTABLISHED
    )


def test_an_unchanged_priced_artifact_does_not_refresh_live_twice(tmp_path):
    """The idempotent case, and the one place it must not be assumed.

    A canary skip says nothing about the live pointer, because the canary can
    publish while the live refresh refuses. So a repeat of identical priced
    bytes asks the refresh again, and the refresh is what decides: it skips
    itself once it has succeeded on those bytes and is retried while it has
    not.
    """
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]],
        clock=clock,
    )

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    # One capture of two games sharing a tip is one serve, one canary and one
    # live refresh of the T-20 window.
    assert len(live_commands(world, offset=20)) == 1

    assert [
        record["outcome"] for record in live_records(receipt, offset=20)
    ] == [controller.LIVE_OUTCOME_REPLACED]

    # And an advisory window re-served from the same bytes skips itself
    # rather than minting a second release of them.
    assert controller.LIVE_OUTCOME_SKIPPED in {
        record["outcome"] for record in receipt["live_refreshes"]
    }


def test_a_live_refusal_is_retried_when_the_canary_is_skipped(tmp_path):
    """A canary skip must not be read as live being current.

    Two T-20 groups whose priced artifact is identical: the second canary is
    skipped, and because the first live refresh refused, the live refresh is
    asked again rather than inheriting the canary's answer.
    """
    clock = Clock(at(19, 25))

    world = World(
        tmp_path,
        schedules=[[game(101, at(23, 0)), game(102, at(23, 40), home=3, away=4)]],
        clock=clock,
        live_refusal="both approval variables are absent",
    )

    # The second group prices the cumulative slate, so the artifact differs.
    # Forcing them identical is what makes the skip reachable at all.
    world._serve_original = world._serve

    def serve(command):
        result = world._serve_original(command)

        path = Path(world._value(command, "--receipt-path"))

        receipt = json.loads(path.read_text(encoding="utf-8"))

        receipt["priced_market_artifact"] = {"sha256": "f" * 64}

        path.write_text(json.dumps(receipt), encoding="utf-8")

        return result

    world._serve = serve

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["captures_completed"] == 2

    # The canary was published once and skipped once.
    assert [
        record["outcome"] for record in receipt["feed_publication_outcomes"]
    ] == ["PUBLISHED", controller.PUBLICATION_OUTCOME_SKIPPED]

    # And live was asked both times, because it had not succeeded.
    assert len(live_commands(world, offset=20)) == 2

    assert [
        record["outcome"] for record in live_records(receipt, offset=20)
    ] == [
        controller.LIVE_OUTCOME_REFUSED,
        controller.LIVE_OUTCOME_REFUSED,
    ]


# ----------------------------------------------------------------------
# constants that must not drift
# ----------------------------------------------------------------------


def test_the_live_vocabulary_is_pinned_to_the_refresh_module():
    """Stated locally so the probe stays light; pinned here so it stays true.

    The refresh module imports the publisher, which pulls pandas and the
    transports behind it, so the controller restates the offsets, the
    destinations and the outcome words rather than importing them. Restating
    is only safe while something compares the two copies.
    """
    refresh = load_ops("run_progressive_live_refresh")

    assert controller.PROGRESSIVE_LIVE_OFFSETS_MINUTES == (
        refresh.PROGRESSIVE_OFFSETS_MINUTES
    )

    assert controller.LIVE_FEED_DESTINATION == refresh.LIVE_FEED_DESTINATION

    assert controller.LIVE_PAGE_DESTINATION == refresh.LIVE_PAGE_DESTINATION

    assert controller.LIVE_PUBLICATION_MODE == refresh.publisher.MODE_LIVE

    for mine, theirs in (
        (controller.LIVE_OUTCOME_ESTABLISHED, refresh.OUTCOME_ESTABLISHED),
        (controller.LIVE_OUTCOME_REPLACED, refresh.OUTCOME_REPLACED),
        (
            controller.LIVE_OUTCOME_ALREADY_CURRENT,
            refresh.OUTCOME_ALREADY_CURRENT,
        ),
        (
            controller.LIVE_OUTCOME_PROMOTED_THEN_INCOMPLETE,
            refresh.OUTCOME_PROMOTED_THEN_INCOMPLETE,
        ),
        (controller.LIVE_OUTCOME_REFUSED, refresh.OUTCOME_REFUSED),
        (
            controller.LIVE_OUTCOME_ALREADY_CLOSER_TO_TIP,
            refresh.OUTCOME_ALREADY_CLOSER_TO_TIP,
        ),
    ):
        assert mine == theirs

    # Every outcome the refresh exits zero on is one the controller treats as
    # faultless, and the controller's two extra words are its own.
    assert refresh.OUTCOMES_THAT_SUCCEED <= (
        controller.LIVE_OUTCOMES_WITHOUT_FAULT
    )

    # And the page is only built after an outcome that actually published,
    # which is a strictly narrower set than the faultless one.
    assert controller.LIVE_OUTCOMES_THAT_PUBLISH < (
        controller.LIVE_OUTCOMES_WITHOUT_FAULT
    )

    assert controller.LIVE_OUTCOMES_THAT_PUBLISH == (
        refresh.OUTCOMES_THAT_SUCCEED - {refresh.OUTCOME_ALREADY_CLOSER_TO_TIP}
    )

    assert controller.LIVE_OUTCOMES_WITHOUT_FAULT - (
        refresh.OUTCOMES_THAT_SUCCEED
    ) == {controller.LIVE_OUTCOME_SKIPPED}

    # The canary each window is authorised by, agreed between the two.
    assert refresh.CANARY_FEED_DESTINATIONS == {
        **controller.ADVISORY_FEED_DESTINATIONS,
        20: controller.T20_FEED_DESTINATION,
    }


def test_the_live_entry_point_belongs_to_the_controller():
    """No workflow step may invoke the refresh directly."""
    assert "ops/run_progressive_live_refresh.py" in (
        validator.CONTROLLER_OWNED_ENTRY_POINTS
    )

    assert str(controller.LIVE_REFRESH_ENTRY_POINT) in (
        validator.CONTROLLER_OWNED_ENTRY_POINTS
    )

    for workflow in sorted(WORKFLOW_DIR.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")

        assert "run_progressive_live_refresh" not in text, workflow.name


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
        "live_feed_destination",
        "live_page_destination",
        "live_refreshes",
        "live_refreshes_with_faults",
        "live_release_id",
        "progressive_live_offsets",
        "start_time_utc",
        "finish_time_utc",
        "wall_seconds",
        "final_outcome",
    ):
        assert field in receipt, field

    # Small: no model arrays, no parquet, no API payload archives. The bound
    # is a bound on what *kind* of thing may be in here, and it moved because
    # a segment now records four live refreshes as well as three advisory
    # windows -- one short object per window per stream, which is the record
    # the whole chain is read from. Still nothing per row, per player or per
    # quote, which is what the bound is for.
    assert len(json.dumps(receipt)) < 16000

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


def as_the_entry_point_sees_it(
    monkeypatch, world: World, *, clock: Clock | None = None
) -> None:
    """Point the module-level collaborators at this world.

    ``run`` and ``main`` build the controller themselves, so a test that
    goes in through the entry point has to replace the schedule source and
    the subprocess runner where the controller looks them up.

    ``clock`` replaces the wall clock the entry point would otherwise build
    the controller with. Required by any test whose subject depends on where
    now sits relative to a tip: the fixture slate is a fixed day in the past,
    so under the real clock every one of its games has tipped and the
    post-tip publication guard decides the whole run.
    """
    monkeypatch.setattr(controller, "_bdl_games", world.schedule_source)

    monkeypatch.setattr(controller, "_run_subprocess", world)

    if clock is None:
        return

    # A subclass rather than a factory function, so a test that goes on to
    # patch a method of ``controller.T20Controller`` still patches something
    # the instances the entry point builds inherit from.
    class WithInjectedClock(controller.T20Controller):
        def __init__(self, **kwargs):
            super().__init__(
                **{**kwargs, "clock": clock, "sleep": clock.sleep}
            )

    monkeypatch.setattr(controller, "T20Controller", WithInjectedClock)


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

    recovery_clock = Clock(at(22, 50))

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

    recovery_clock = Clock(at(22, 50))

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
    # window, by the run before the one that published it.
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

    retry_clock = Clock(at(22, 50))

    retry = World(
        tmp_path,
        schedules=[games],
        clock=retry_clock,
        serving_outcome="FAILED",
    )

    write_morning_receipt(retry)

    as_the_entry_point_sees_it(monkeypatch, retry, clock=retry_clock)

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

    recovery_clock = Clock(at(22, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    as_the_entry_point_sees_it(monkeypatch, recovery, clock=recovery_clock)

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


def replay_the_capture_log(world: World) -> None:
    """Append every capture-run row of this slate again, byte for byte.

    The duplicate 2026-10-23 is still in, reproduced by the route a claim
    cannot cover: the envelope is repeated and the component rows are left
    alone, so the two records carry one instant, one window set, one due-game
    set and one set of component digests because they are one capture.
    """
    path = world.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"

    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def test_a_window_recorded_twice_is_recovered_rather_than_stuck(tmp_path):
    """The state the three lost games are still in, and the way out of it.

    A captured-but-unpublished window is recoverable only if the capture can
    be proved, and ``_verify_captured`` proves it with the pricing path's own
    ``select_capture_bundle``. A second record of that capture used to make
    that proof impossible for the rest of the slate's life, so the recovery
    this controller gained had nothing to recover with: the probe asked for
    the job, the job verified the capture, and the verification refused.

    Both records are records of one capture, so the capture is proved and the
    window reaches the feed. Nothing recaptures: the collector is not invoked
    and the log still holds both rows afterwards.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = captured_but_unpublished(tmp_path, clock=clock, games=games)

    replay_the_capture_log(world)

    capture_log = world.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"

    before = capture_log.read_bytes()

    recovery_clock = Clock(at(22, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    assert [
        record["window_ids"] for record in receipt["recovered_windows"]
    ] == [["101:T-20m"]]

    assert receipt["captured_unserved_windows"] == []

    assert [row["window_ids"] for row in served_records(recovery)] == [
        ["101:T-20m"]
    ]

    # The capture the failed run took, not a later one.
    bundle = receipt["recovered_windows"][0]["verified_bundles"][0]

    assert bundle["captured_at_utc"] == at(22, 40).isoformat()

    # No collector ran, and the append-only log is the log it was.
    assert collector_commands(recovery) == []

    assert capture_log.read_bytes() == before


def test_recovering_a_duplicated_window_twice_yields_the_same_release(
    tmp_path,
):
    """Idempotency over a log that has two records of one capture.

    The second run has to read the duplicate the same way the first did,
    otherwise a window could be published once per reading. It finds the
    window recorded as served, has nothing pending, and leaves the slate with
    the one release the first run published.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = captured_but_unpublished(tmp_path, clock=clock, games=games)

    replay_the_capture_log(world)

    first_clock = Clock(at(22, 50))

    first = World(tmp_path, schedules=[games], clock=first_clock)

    write_morning_receipt(first)

    build(
        first,
        clock=first_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    released = served_records(first)

    second_clock = Clock(at(22, 55))

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

    assert served_records(second) == released

    assert [row["release_id"] for row in released] == ["release-1"]


def test_two_captures_of_one_window_still_stop_the_recovery(tmp_path):
    """And the refusal says which recorded facts the two differ in.

    A second *capture* is not a second record of one capture. Both are
    genuine point-in-time data, nothing in the data model says which one T-20
    pricing is entitled to read, and a controller that published either would
    be publishing a snapshot nobody chose. So recovery refuses, records the
    window with the reason the selection gave, carries it out under
    ``captured_unserved_windows``, and ends the segment on the outcome that
    is not a success -- which is how a refusal that costs only its own window
    still says what it abandoned.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = captured_but_unpublished(tmp_path, clock=clock, games=games)

    # A second sweep of the same window a minute later: its own instant, its
    # own component rows, its own digests.
    write_capture(
        world.snapshot_dir,
        date=SLATE,
        captured_at=at(22, 41).isoformat(),
        window_ids=["101:T-20m"],
        due_game_ids=[101],
        games=[game(101, at(23, 0))],
        props=[prop(101, 701)],
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

    assert [
        record["window_id"] for record in receipt["unrecoverable_windows"]
    ] == ["101:T-20m"]

    reason = " ".join(receipt["unrecoverable_windows"][0]["reason"].split())

    assert "2 different point-in-time captures" in reason

    assert "captured_at" in reason

    assert receipt["captured_unserved_windows"] == ["101:T-20m"]

    assert receipt["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED

    assert (
        controller.OUTCOME_CAPTURED_NOT_SERVED
        not in controller.OUTCOMES_THAT_SUCCEED
    )

    assert served_records(recovery) == []


def test_recovery_is_safe_to_run_twice(tmp_path):
    """A re-serve is idempotent: the second run finds nothing to do."""
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    first_clock = Clock(at(22, 50))

    first = World(tmp_path, schedules=[games], clock=first_clock)

    write_morning_receipt(first)

    build(
        first,
        clock=first_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    second_clock = Clock(at(22, 55))

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


# ----------------------------------------------------------------------
# a window that can never be recovered costs only itself
# ----------------------------------------------------------------------


def conflict(tmp_path, *, games, window_id: str, game_id: int, captured_at):
    """A second scheduled capture of one window, which is a real conflict.

    Written at its own ``captured_at`` so no other window's component
    digests move: ``_component_records`` groups by that timestamp, and a
    duplicate sharing one would corrupt the capture it was meant to leave
    alone. Both rows are genuine scheduled captures and the log is
    append-only, so nothing can undo this -- which is the point.
    """
    return write_capture(
        tmp_path / "data" / "snapshots",
        date=SLATE,
        captured_at=captured_at.isoformat(),
        window_ids=[window_id],
        due_game_ids=[game_id],
        games=[dict(row) for row in games],
        props=[prop(game_id, 600 + game_id)],
    )


def test_an_unrecoverable_window_does_not_cost_the_segment_its_own_captures(
    tmp_path,
):
    """The blast radius of one capture production can never serve.

    ``_recover`` verified every unserved window of the slate in one call and
    raised on the first refusal, so one conflicted capture took the whole
    segment with it: the pending T-20 windows of games that had not tipped
    were never captured, and the advisory windows after them never ran. On
    the 2026-10-24 slate the 19:25Z segment owns 27 of the 32 windows, so
    one unservable capture existing by 15:25Z cost all 27.

    The window itself is lost either way. What must not be lost with it is
    everything the segment still could have done.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(23, 0)),
        game(102, at(23, 0), home=3, away=4),
        game(103, at(0, 0, day=21), home=5, away=6),
    ]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    conflict(
        tmp_path,
        games=games,
        window_id="101:T-20m",
        game_id=101,
        captured_at=at(22, 45),
    )

    recovery_clock = Clock(at(22, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    # The one window nothing can serve, named with the reason selection
    # refused it rather than as an anonymous count.
    assert [
        record["window_id"] for record in receipt["unrecoverable_windows"]
    ] == ["101:T-20m"]

    assert (
        "expected exactly one scheduled capture"
        in receipt["unrecoverable_windows"][0]["reason"]
    )

    # The segment's own T-20 window: captured, served and published.
    assert "103:T-20m" in served_window_ids_for(recovery)

    # The advisory window after it ran too.
    assert "103:T-45m" in {
        str(record["window_id"]) for record in receipt["advisory_windows"]
    }

    # Still fails closed, and still on the outcome that reaches the exit
    # code: production is missing a window and that is not a success.
    assert receipt["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED

    assert (
        controller.OUTCOME_CAPTURED_NOT_SERVED
        not in controller.OUTCOMES_THAT_SUCCEED
    )


def test_one_unrecoverable_window_does_not_discard_the_one_beside_it(
    tmp_path,
):
    """Verification was a single call over every unserved window at once.

    The first refusal ended it, so a window that would have verified was
    discarded for being in the same batch as one that would not.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    conflict(
        tmp_path,
        games=games,
        window_id="101:T-20m",
        game_id=101,
        captured_at=at(22, 45),
    )

    recovery_clock = Clock(at(22, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    # The recoverable one reached a release.
    assert [
        record["window_ids"] for record in receipt["recovered_windows"]
    ] == [["102:T-20m"]]

    assert "102:T-20m" in served_window_ids_for(recovery)

    # The refusal names the other one, and only the other one.
    assert [
        record["window_id"] for record in receipt["unrecoverable_windows"]
    ] == ["101:T-20m"]

    assert receipt["captured_unserved_windows"] == ["101:T-20m"]

    assert receipt["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED


def test_a_capture_serving_could_not_select_is_not_recorded_as_served(
    tmp_path,
):
    """The ledger row says what the release it names actually carries.

    ``_record_served_windows`` writes every T-20 window in the capture log,
    on the stated grounds that a window serving could not have used would
    have refused the run before reaching that line. Isolating the refusal
    makes that false: the publication now happens *beside* a window
    selection rejects. Recording it would claim a release carries a window
    no release can carry, and -- worse -- would retire it from the unserved
    set, turning the permanently missing window into a green segment.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    conflict(
        tmp_path,
        games=games,
        window_id="101:T-20m",
        game_id=101,
        captured_at=at(22, 45),
    )

    recovery_clock = Clock(at(22, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    rows = [
        row
        for row in served_records(recovery)
        if row["offset_minutes"] == 20
    ]

    assert rows, "the recoverable window should have produced a ledger row"

    for row in rows:
        assert "101:T-20m" not in row["window_ids"], (
            "the ledger claims a release carries a window selection refuses"
        )

    assert controller.served_window_ids(
        recovery.data_root
        / "processed"
        / "t20_served_windows"
        / f"{SLATE}.jsonl"
    ) == {"102:T-20m"}


def test_a_post_tip_capture_is_isolated_the_way_a_conflict_is(tmp_path):
    """A second permanent cause, to show the isolation is not conflict-shaped.

    A capture taken after its game tipped is refused by a different check
    than a duplicate is, and is just as permanent: the window is in the
    past, so no later run can take a pre-tip capture of it.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    # 104 tips at 22:20 and nothing captured its T-20 window at 22:00,
    # because the run that owned it never saw it on the schedule. This is a
    # capture of it taken after the game started.
    late = game(104, at(22, 20), home=7, away=8)

    with_late = [*games, late]

    write_capture(
        tmp_path / "data" / "snapshots",
        date=SLATE,
        captured_at=at(22, 25).isoformat(),
        window_ids=["104:T-20m"],
        due_game_ids=[104],
        games=[dict(row) for row in with_late],
        props=[prop(104, 704)],
    )

    # Before the two games beside it tip, so this is a test about the
    # capture selection refuses and not about the post-tip publication
    # guard, which would otherwise refuse all three windows at once.
    recovery_clock = Clock(at(22, 50))

    recovery = World(tmp_path, schedules=[with_late], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    unrecoverable = {
        record["window_id"]: record["reason"]
        for record in receipt["unrecoverable_windows"]
    }

    assert list(unrecoverable) == ["104:T-20m"]

    assert "not pre-tip" in unrecoverable["104:T-20m"]

    # The two windows beside it were served from the captures that existed.
    assert served_window_ids_for(recovery) == {"101:T-20m", "102:T-20m"}

    assert receipt["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED


def test_the_unrecoverable_window_stops_blocking_the_segments_after_it(
    tmp_path,
):
    """The condition was sticky, and that is what made it expensive.

    Every cause is permanent, so each later segment re-derived the same
    unserved set, re-attempted the same doomed recovery and skipped its own
    work in the same place. One conflicted capture at 15:25Z therefore cost
    every segment for the rest of the slate, not only the one it appeared
    in. Here the third segment of the day runs after the second has already
    met the window and failed on it.
    """
    clock = Clock(at(19, 25))

    # 104 tips late enough that its T-20 window belongs to the third
    # segment, so the two runs below own genuinely different work.
    games = [
        game(101, at(23, 0)),
        game(102, at(23, 0), home=3, away=4),
        game(104, at(2, 0, day=21), home=5, away=6),
    ]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    conflict(
        tmp_path,
        games=games,
        window_id="101:T-20m",
        game_id=101,
        captured_at=at(22, 45),
    )

    second_clock = Clock(at(22, 50))

    second_segment = World(tmp_path, schedules=[games], clock=second_clock)

    write_morning_receipt(second_segment)

    second_receipt = build(
        second_segment,
        clock=second_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert (
        second_receipt["final_outcome"]
        == controller.OUTCOME_CAPTURED_NOT_SERVED
    )

    # The third segment, over the same state. The window is still
    # unrecoverable and still named, and this segment still does its own
    # work rather than stopping where the last one stopped.
    third_clock = Clock(at(0, 25, day=21))

    third_segment = World(tmp_path, schedules=[games], clock=third_clock)

    write_morning_receipt(third_segment)

    receipt = build(
        third_segment,
        clock=third_clock,
        segment_start=at(0, 25, day=21),
        segment_end=at(5, 25, day=21),
    ).run()

    assert [
        record["window_id"] for record in receipt["unrecoverable_windows"]
    ] == ["101:T-20m"]

    assert "104:T-20m" in served_window_ids_for(third_segment)

    assert "104:T-45m" in {
        str(record["window_id"]) for record in receipt["advisory_windows"]
    }

    assert receipt["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED


def test_a_recovery_that_cannot_publish_still_takes_the_capture_it_owns(
    tmp_path,
):
    """The capture is the half of this that cannot be retried.

    Recovery's second failure mode: the windows verify and the serving
    stage refuses anyway. That refusal used to land before any collector
    had run, so the segment lost the T-20 captures of games that had not
    tipped. The market at T-20 exists once -- a publication can wait for
    the next segment and a snapshot cannot -- so the refusal is deferred
    until after the work rather than raised instead of it.

    This is the shape the live slate is in until the serving half of the
    defect is fixed: one capture selection cannot accept makes
    ``ProspectiveSnapshotClient`` refuse the whole slate, so recovery's
    serve fails for a reason the controller cannot do anything about.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(23, 0)),
        game(103, at(0, 0, day=21), home=5, away=6),
    ]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    retry_clock = Clock(at(23, 0))

    retry = World(
        tmp_path,
        schedules=[games],
        clock=retry_clock,
        serving_outcome="FAILED",
    )

    write_morning_receipt(retry)

    with pytest.raises(controller.ControllerRefused) as refused:
        build(
            retry,
            clock=retry_clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    # Deferred, never downgraded: the refusal keeps its own outcome.
    assert refused.value.outcome == controller.OUTCOME_FAILED

    assert refused.value.captured_unserved_windows == [
        "101:T-20m",
        "103:T-20m",
    ]

    # The window this segment owned was captured, which is what the early
    # refusal used to cost. Serving still cannot publish it, so it is
    # named above as unserved and a later run can still recover it.
    assert "103:T-20m" in controller.completed_window_ids(
        retry.snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"
    )


class ServingFailsThenWorks(World):
    """A world whose first production serve fails and whose later ones do not.

    Recovery and the segment's own capture work serve through separate
    invocations of the same entry point, and the case below is the one
    where they differ: the serving failure that deferred the refusal is
    gone by the time the capture is served, and the release that capture
    publishes carries the recovered windows too.
    """

    def _serve(self, command):
        self.serving_outcome = "FAILED" if not self.served else "SERVED"

        return super()._serve(command)


def test_a_deferred_refusal_the_later_release_resolved_is_not_raised(
    tmp_path,
):
    """A deferred refusal is re-raised only while it is still true.

    The slate is re-served whole after every capture, so a release
    published later in the same segment carries the windows the earlier
    recovery could not. Raising the deferred refusal anyway would fail a
    segment that had in fact published everything, which is its own kind of
    false report -- and the opposite error to the one being fixed here.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(23, 0)),
        game(103, at(0, 0, day=21), home=5, away=6),
    ]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    retry_clock = Clock(at(22, 50))

    retry = ServingFailsThenWorks(
        tmp_path, schedules=[games], clock=retry_clock
    )

    write_morning_receipt(retry)

    receipt = build(
        retry,
        clock=retry_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    # The recovery failure is reported rather than hidden ...
    assert [
        record["outcome"] for record in receipt["recovery_faults"]
    ] == [controller.OUTCOME_FAILED]

    # ... and is not raised, because the capture work's own release carries
    # the window the failed recovery was about.
    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    assert receipt["captured_unserved_windows"] == []

    assert served_window_ids_for(retry) == {"101:T-20m", "103:T-20m"}


def test_a_slate_with_nothing_recoverable_is_not_served_again(tmp_path):
    """No window verified, so there is nothing to publish from.

    Serving anyway would re-price and re-publish the slate's existing
    release for no reason, minting a release that carries exactly what the
    last one carried. The windows that caused it are recorded either way.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    conflict(
        tmp_path,
        games=games,
        window_id="101:T-20m",
        game_id=101,
        captured_at=at(22, 45),
    )

    retry_clock = Clock(at(23, 50))

    retry = World(tmp_path, schedules=[games], clock=retry_clock)

    write_morning_receipt(retry)

    receipt = build(
        retry,
        clock=retry_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert retry.served == []

    assert retry.published == []

    assert receipt["recovered_windows"] == []

    assert [
        record["window_id"] for record in receipt["unrecoverable_windows"]
    ] == ["101:T-20m"]

    assert receipt["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED


def test_the_receipt_says_which_unserved_windows_no_run_will_ever_serve(
    tmp_path, monkeypatch
):
    """The reason string was actively misleading for a permanent cause.

    It said the captures "stay exactly as they were taken; a later run will
    serve and publish from them without recapturing", which is true of a
    window waiting for a later run and false of a conflicted or post-tip
    one. An operator reading it would wait for a recovery that can never
    happen.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    conflict(
        tmp_path,
        games=games,
        window_id="101:T-20m",
        game_id=101,
        captured_at=at(22, 45),
    )

    recovery_clock = Clock(at(22, 50))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    as_the_entry_point_sees_it(monkeypatch, recovery, clock=recovery_clock)

    receipt_path = tmp_path / "receipt.json"

    summary_path = tmp_path / "summary.md"

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
            "--summary-path",
            str(summary_path),
        ]
    )

    assert exit_code == 1

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert "101:T-20m" in receipt["reason"]

    assert "no later run" in receipt["reason"]

    # The misleading promise must not be made about this window.
    assert "a later run will serve and publish from them" not in (
        receipt["reason"]
    )

    summary = controller.render(receipt)

    assert "101:T-20m" in summary

    assert "expected exactly one scheduled capture" in summary


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


def serving_provenance(world: World, *, slate_date: str = SLATE) -> list[dict]:
    path = (
        world.data_root
        / "processed"
        / "incumbent_serving_provenance"
        / f"{slate_date}.jsonl"
    )

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_every_serve_leaves_a_durable_receipt_for_the_bytes_it_wrote(tmp_path):
    """Why: serving rewrites the slate's priced markets in place.

    The morning receipt in ``processed/incumbent_serving`` provenances the
    morning's bytes, and the receipt for the bytes a T-20 serve leaves behind
    used to exist only in ``RUNNER_TEMP``. So from the next morning nothing
    provenanced the priced markets on disk and the grader refused every slate
    this controller published -- which is how run 40082999575 ended.
    """
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    production = [
        row
        for row in serving_provenance(world)
        if row["offset_minutes"] == 20
    ]

    assert len(production) == 1

    row = production[0]

    assert row["record_type"] == controller.SERVING_PROVENANCE_RECORD_TYPE

    # The grader must be able to tell this row from the advisory ones in the
    # same file, because only this one published to the production feed.
    assert controller.ADVISORY_RECEIPT_MARKER_FIELD not in row

    assert row["outcome"] == "SERVED"

    assert row["slate_date"] == SLATE

    assert row["offset_minutes"] == 20

    assert row["served_window_ids"] == ["101:T-20m"]

    # The whole receipt, unedited: the grader re-verifies it against the
    # parquet, so a summarised row would be a row it could not use.
    assert row["priced_market_artifact"]["sha256"] == (
        served_records(world)[0]["priced_market_artifact_sha256"]
    )


def test_the_receipt_is_durable_before_the_release_that_describes_it(tmp_path):
    """Order matters: the bytes exist the moment serving returns.

    A publication that went out first and then failed to record its own
    provenance would be a live release nothing could grade.
    """
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    path = (
        world.data_root
        / "processed"
        / "incumbent_serving_provenance"
        / f"{SLATE}.jsonl"
    )

    seen: list[bool] = []

    publish = world._publish

    def watched(command):
        seen.append(path.exists())

        return publish(command)

    world._publish = watched

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert seen and all(seen), (
        "a release was published before its serving receipt was durable"
    )


def test_a_serve_whose_receipt_cannot_be_recorded_publishes_nothing(tmp_path):
    """Fail closed. An ungradable release is worse than a missing one."""
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    # A file where the ledger's directory has to go, so the append cannot
    # happen for a reason the controller did not invent.
    blocked = world.data_root / "processed" / "incumbent_serving_provenance"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("not a directory", encoding="utf-8")

    with pytest.raises(controller.ControllerRefused) as refusal:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert "could not be recorded" in str(refusal.value)

    assert not served_records(world), (
        "a release was recorded despite its provenance never being written"
    )


def test_an_advisory_window_records_the_receipt_for_its_own_bytes(tmp_path):
    """The advisory serves rewrite the same file, so they record too."""
    clock = Clock(at(14, 25))

    world = World(tmp_path, schedules=[[game(101, at(20, 0))]], clock=clock)

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_1[0], segment_end=SEGMENT_1[1]
    ).run()

    rows = serving_provenance(world)

    assert rows, "the advisory serves left no provenance at all"

    offsets = sorted(int(row["offset_minutes"]) for row in rows)

    assert offsets == sorted(controller.ADVISORY_OFFSETS_MINUTES), (
        "a serve that rewrote the slate's priced markets left no provenance"
    )

    for row in rows:
        assert row["record_type"] == controller.SERVING_PROVENANCE_RECORD_TYPE

        assert row["priced_market_artifact"]["sha256"]


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

    recovery_clock = Clock(at(22, 50))

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


# ----------------------------------------------------------------------
# nothing is published after tip
# ----------------------------------------------------------------------


class PublishesToADirectory(World):
    """A world whose publication stage leaves a release and a pointer behind.

    The other fakes in this module write receipts, which is enough to assert
    what the controller did. It is not enough to assert what a *refusal* left
    alone: "the release already serving is untouched" is a claim about bytes
    on the far side of the publisher, so this one writes them. The shape is
    the publisher's own -- an immutable release directory and a
    ``current.json`` naming it, read back by every downstream consumer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.published_root = self.data_root.parent / "published"

    def _publish(self, command):
        result = super()._publish(command)

        # The T-20 release only. An advisory release is addressed by naming
        # its destination and lands in a confined root of its own, which is
        # the distinction the controller itself makes.
        if self._optional(command, "--feed-destination-name") or self._optional(
            command, "--page-destination-name"
        ):
            return result

        if self._value(command, "--stage") != "feed":
            return result

        receipt = json.loads(
            Path(self._value(command, "--receipt-path")).read_text(
                encoding="utf-8"
            )
        )

        release_id = str((receipt.get("release") or {}).get("release_id") or "")

        release_dir = self.published_root / "releases" / release_id

        release_dir.mkdir(parents=True, exist_ok=True)

        release_dir.joinpath("rows.json").write_text(
            json.dumps(receipt.get("published_game_ids") or [], sort_keys=True),
            encoding="utf-8",
        )

        self.published_root.joinpath(controller_pointer()).write_text(
            json.dumps({"release_id": release_id}, sort_keys=True),
            encoding="utf-8",
        )

        return result


def controller_pointer() -> str:
    """The pointer name the real publisher writes, not a name invented here."""
    return publisher_module().CURRENT_POINTER


def publisher_module():
    return load_ops("publish_incumbent_to_wizardofodds")


def tree_digest(root: Path) -> list[tuple[str, str]]:
    """Every file under ``root`` with its digest, so equality means bytes."""
    if not root.exists():
        return []

    return sorted(
        (
            str(path.relative_to(root)),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    )


def test_a_window_whose_game_has_tipped_is_not_published(tmp_path):
    """The guard the 2026-10-23 slate did not have.

    A sound pre-tip capture that an earlier run never published, reached by
    a later run after the game has started. Everything about the capture is
    still correct -- it was taken at the window, it is the only scheduled
    capture of it and every component digest still matches -- so selection
    accepts it and the old controller served and published it. The release
    that produced was a pregame page for a game already under way.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    # Ten minutes after the game started, which is the smallest version of
    # the real incident's two hours and forty minutes.
    recovery_clock = Clock(at(23, 10))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    refused = receipt["post_tip_refused_windows"]

    assert [record["window_id"] for record in refused] == ["101:T-20m"]

    assert (
        refused[0]["outcome"] == controller.OUTCOME_POST_TIP_PUBLICATION
    )

    # The refusal says the game, when it started and how long ago that was.
    assert refused[0]["reason"] == (
        "game 101 tipped at 2026-10-20T23:00:00+00:00, 10 minutes before "
        "2026-10-20T23:10:00+00:00, so the window 101:T-20m is not a "
        "pregame price and is not published. The release already serving "
        "is left exactly as it is: nothing is retracted, overwritten or "
        "repointed"
    )

    # Refused before the serving entry point, so nothing was priced, nothing
    # was published and no ledger row was appended.
    assert recovery.served == []

    assert recovery.published == []

    assert publication_commands(recovery, stage="feed") == []

    assert served_records(recovery) == []

    # And the window stays outstanding, on the outcome that reaches the exit
    # code rather than on one that reads as a success.
    assert receipt["captured_unserved_windows"] == ["101:T-20m"]

    assert receipt["final_outcome"] == controller.OUTCOME_CAPTURED_NOT_SERVED

    assert (
        controller.OUTCOME_POST_TIP_PUBLICATION
        not in controller.OUTCOMES_THAT_SUCCEED
    )


def test_a_post_tip_refusal_exits_non_zero(tmp_path, monkeypatch):
    """A loud refusal, not a silent skip.

    The whole defect was that every stage of the late run reported success,
    so this is asserted through ``main`` -- the thing the workflow's exit
    status actually comes from -- rather than off the receipt.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    recovery_clock = Clock(at(23, 10))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    as_the_entry_point_sees_it(monkeypatch, recovery, clock=recovery_clock)

    receipt_path = tmp_path / "receipt.json"

    summary_path = tmp_path / "summary.md"

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
            "--summary-path",
            str(summary_path),
        ]
    )

    assert exit_code == 1

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert [
        record["outcome"] for record in receipt["post_tip_refused_windows"]
    ] == [controller.OUTCOME_POST_TIP_PUBLICATION]

    summary = summary_path.read_text(encoding="utf-8")

    assert "| refused for being post-tip | 101:T-20m |" in summary

    assert "tipped at 2026-10-20T23:00:00+00:00" in summary


def test_a_game_tipping_exactly_now_is_refused(tmp_path):
    """The boundary, pinned in the closed direction.

    A game whose tip is this instant has started. ``select_capture_bundle``
    draws the capture-side boundary the same way -- it requires the capture
    to be strictly before the tip -- so a release at exactly the tip is
    refused for the same reason a capture at exactly the tip is.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    recovery_clock = Clock(at(23, 0))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert [
        record["window_id"] for record in receipt["post_tip_refused_windows"]
    ] == ["101:T-20m"]

    assert "0 minutes before" in receipt["post_tip_refused_windows"][0][
        "reason"
    ]

    assert recovery.published == []

    # One second earlier is still a pregame price, and is published.
    before = Clock(at(23, 0) - timedelta(seconds=1))

    ahead = World(tmp_path, schedules=[games], clock=before)

    write_morning_receipt(ahead)

    cleared = build(
        ahead,
        clock=before,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert cleared["post_tip_refused_windows"] == []

    assert ahead.published == [[101]]

    assert cleared["pre_tip_margins"] == [
        {
            "game_id": 101,
            "minutes_before_tip": 0.017,
            "tip_utc": "2026-10-20T23:00:00+00:00",
            "window_id": "101:T-20m",
        }
    ]


def test_a_game_that_tipped_in_an_earlier_group_does_not_block_a_later_one(
    tmp_path,
):
    """The scoping the guard would be useless -- or harmful -- without.

    A release is cumulative: one serve at T-20 prices every game with an
    eligible capture at that offset, so the last group of a staggered slate
    is published long after the first game of it tipped. Here the 20:00 game
    tipped two hours and forty minutes before the 23:00 game's window is
    published, which is exactly the interval the real incident ran late by.
    Its rows belong in that release, because they were priced and published
    before its own tip, and the guard must say nothing about it: the windows
    a publication is refused for are the windows it is *for*.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(20, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    # Two groups, both published, and the second published at 22:40 -- two
    # hours and forty minutes after the first game started.
    assert world.published == [[101], [101, 102]]

    assert [moment.isoformat() for moment in world.clock.woke][-1:] == [
        "2026-10-20T22:40:00+00:00"
    ]

    assert receipt["post_tip_refused_windows"] == []

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    # The guard cleared only the second group's own window at the second
    # publication, and the margin it records is that window's own.
    assert [
        (record["window_id"], record["minutes_before_tip"])
        for record in receipt["pre_tip_margins"]
    ] == [("101:T-20m", 20.0), ("102:T-20m", 20.0)]

    assert served_window_ids_for(world) == {"101:T-20m", "102:T-20m"}


def test_a_refusal_leaves_the_release_and_its_pointer_exactly_as_they_were(
    tmp_path,
):
    """The guard declines to add. It never retracts.

    A slate with a release already published and a second window that no
    release carries, reached after that second game has tipped. The refusal
    has to cost exactly the new publication: the release directory, its rows
    and the ``current.json`` that names it are the bytes the earlier run
    left, down to the digest, and the pointer still names the same release.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(20, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    world = PublishesToADirectory(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    # The first group only: the controller is stopped before the second
    # window by a segment that ends between them, which is how a staggered
    # slate is genuinely split across invocations.
    build(
        world,
        clock=clock,
        segment_start=SEGMENT_2[0],
        segment_end=at(21, 0),
    ).run()

    pointer = world.published_root / controller_pointer()

    assert json.loads(pointer.read_text(encoding="utf-8")) == {
        "release_id": "release-1"
    }

    before = tree_digest(world.published_root)

    assert before, "the first group should have published a release"

    # Now the second window, captured by a run that died before publishing
    # it, reached after its game has tipped.
    late_clock = Clock(at(22, 40))

    late = PublishesToADirectory(
        tmp_path,
        schedules=[games],
        clock=late_clock,
        serving_outcome="FAILED",
    )

    write_morning_receipt(late)

    with pytest.raises(controller.ControllerRefused):
        build(
            late,
            clock=late_clock,
            segment_start=at(21, 0),
            segment_end=SEGMENT_2[1],
        ).run()

    recovery_clock = Clock(at(23, 20))

    recovery = PublishesToADirectory(
        tmp_path, schedules=[games], clock=recovery_clock
    )

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=at(21, 0),
        segment_end=SEGMENT_2[1],
    ).run()

    assert [
        record["window_id"] for record in receipt["post_tip_refused_windows"]
    ] == ["102:T-20m"]

    # Byte-identical, and the pointer still names the release it named.
    assert tree_digest(world.published_root) == before

    assert json.loads(pointer.read_text(encoding="utf-8")) == {
        "release_id": "release-1"
    }

    # Nothing was written towards a new release either: no publication
    # command ran, and the durable ledger still says what it said.
    assert publication_commands(recovery, stage="feed") == []

    assert [row["window_ids"] for row in served_records(recovery)] == [
        ["101:T-20m"]
    ]


def test_an_advisory_window_reached_after_its_tip_is_refused(tmp_path):
    """The advisory half, which is reached by a different route.

    An advisory window is captured hours before tip and is normally served
    long before it, but a serve that would run inside the reservation before
    a T-20 capture is held back and retried in a later gap -- and that gap
    can fall after the game. The release it would make carries the same wrong
    claim the T-20 one would, published to the advisory destination instead
    of to the canary.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    held = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    )

    # The T-45 window's capture, taken at 22:15 and still unserved when the
    # game starts: the same state a held window is left in by a segment that
    # ran out of gaps.
    held._advisory_pending = {}

    held._advisory_held = {
        45: [
            {
                "game_id": 101,
                "offset_minutes": 45,
                "capture_at_utc": at(22, 15),
                "tip_utc": at(23, 0),
                "window_id": "101:T-45m",
            }
        ]
    }

    world.clock.now = at(23, 5)

    held._drain_advisory(readiness=morning_readiness_for(world), next_t20_due=None)

    assert [
        (record["window_id"], record["outcome"])
        for record in held.advisory
    ] == [("101:T-45m", controller.ADVISORY_OUTCOME_POST_TIP)]

    assert (
        controller.ADVISORY_OUTCOME_POST_TIP
        not in controller.ADVISORY_OUTCOMES_WITHOUT_FAULT
    )

    # Nothing was served and nothing was published for it.
    assert world.advisory_served == []

    assert world.advisory_published == []


def morning_readiness_for(world: World) -> dict:
    return controller.morning_readiness(
        data_root=world.data_root,
        slate_date=SLATE,
        production_sha=PRODUCTION_SHA,
        project_root=PROJECT,
    )


def test_an_advisory_post_tip_refusal_fails_the_segment_after_the_captures(
    tmp_path,
):
    """Loud, and still unable to cost the slate a capture.

    Every other advisory failure degrades the segment and lets it succeed,
    because a defect in a reading aid must not cost the production release.
    A release for a game already under way is not that kind of defect, so it
    reaches the exit code -- but it reaches it from ``run`` after all the
    capture work, never by raising out of the advisory path with a later
    T-20 group still ahead of it.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(20, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    segment = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    )

    # Game 101's T-90 window, captured and still unserved when the segment
    # runs out of T-20 work: the state a window held back for a reservation
    # is left in. Seeded at the start of the last drain, which is where a
    # real one would still be sitting, and which is after both captures.
    original = segment._finish_advisory

    def hold_the_first_window(readiness):
        segment._advisory_held.setdefault(90, []).append(
            {
                "game_id": 101,
                "offset_minutes": 90,
                "capture_at_utc": at(18, 30),
                "tip_utc": at(20, 0),
                "window_id": "101:T-90m",
            }
        )

        return original(readiness)

    segment._finish_advisory = hold_the_first_window

    receipt = segment.run()

    # The refusal is there, named, with the tip and the interval.
    post_tip = [
        record
        for record in receipt["advisory_windows"]
        if record["outcome"] == controller.ADVISORY_OUTCOME_POST_TIP
    ]

    assert [record["window_id"] for record in post_tip] == ["101:T-90m"]

    assert "tipped at 2026-10-20T20:00:00+00:00" in post_tip[0]["reason"]

    # The segment's own T-20 captures both happened and both were published:
    # the refusal cost the advisory release and nothing else.
    assert world.published == [[101], [101, 102]]

    assert served_window_ids_for(world) == {"101:T-20m", "102:T-20m"}

    # And it is not a success.
    assert (
        receipt["final_outcome"] == controller.OUTCOME_POST_TIP_PUBLICATION
    )

    assert (
        receipt["final_outcome"] not in controller.OUTCOMES_THAT_SUCCEED
    )


def test_every_production_release_declares_the_windows_it_is_for(tmp_path):
    """The guard is a property of the chokepoint, not of its callers.

    ``_serve_and_publish`` is the only way to a production release -- the
    capture path and the recovery path both go through it -- and it takes
    the windows it is publishing for as a required argument. A caller that
    forgot them would not reach a publication; it would fail to call.
    """
    import inspect

    signature = inspect.signature(
        controller.T20Controller._serve_and_publish
    )

    windows = signature.parameters["windows"]

    assert windows.kind is inspect.Parameter.KEYWORD_ONLY

    assert windows.default is inspect.Parameter.empty

    source = inspect.getsource(controller.T20Controller)

    assert source.count("self._serve_and_publish(") == 2

    assert "assert_no_window_has_tipped(" in source


def test_the_2026_10_23_incident_shape_is_refused():
    """The real slate's numbers, through the guard that was missing.

    Game 21717941 -- CHI at MIL -- tipped at 20:00 UTC on 2026-10-23 and its
    T-20 window was priced and published at 22:40, two hours and forty
    minutes later. Asserted on the guard directly, with that game's own id
    and those two instants, so the refusal that evening needed is pinned to
    the evening it is about rather than to a fixture slate.
    """
    tip = datetime(2026, 10, 23, 20, 0, tzinfo=UTC)

    published_at = datetime(2026, 10, 23, 22, 40, tzinfo=UTC)

    row = {
        "game_id": 21717941,
        "tip_utc": tip,
        "offset_minutes": 20,
        "capture_at_utc": tip - timedelta(minutes=20),
        "window_id": "21717941:T-20m",
    }

    with pytest.raises(controller.ControllerRefused) as refused:
        controller.assert_no_window_has_tipped([row], now=published_at)

    assert (
        refused.value.outcome == controller.OUTCOME_POST_TIP_PUBLICATION
    )

    assert str(refused.value) == (
        "game 21717941 tipped at 2026-10-23T20:00:00+00:00, 160 minutes "
        "before 2026-10-23T22:40:00+00:00, so the window 21717941:T-20m is "
        "not a pregame price and is not published. The release already "
        "serving is left exactly as it is: nothing is retracted, "
        "overwritten or repointed"
    )

    # The same window, at the time it was actually for, is published.
    assert controller.assert_no_window_has_tipped(
        [row], now=row["capture_at_utc"]
    ) == [
        {
            "game_id": 21717941,
            "minutes_before_tip": 20.0,
            "tip_utc": "2026-10-23T20:00:00+00:00",
            "window_id": "21717941:T-20m",
        }
    ]


# --------------------------------------------------------------------------
# the serving-window gates
# --------------------------------------------------------------------------


def test_the_window_gates_are_asked_before_the_slate_is_served(tmp_path):
    """The two checks that moved off promotion are asked at every window.

    Asked here, and asked first: the gates have to decide before pricing
    produces an artifact, or a window that fails them leaves behind a priced
    slate nothing is allowed to publish.
    """
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert world.window_gate_calls == [SLATE]

    scripts = [Path(command[1]).name for command in world.commands]

    assert scripts.index("assert_serving_window_gates.py") < scripts.index(
        "run_incumbent_production_serving.py"
    )


def test_a_window_whose_gates_refuse_serves_and_publishes_nothing(tmp_path):
    """Fail-closed where the inputs exist, which is the whole point of moving."""
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)
    world.window_gates_refuse = True

    write_morning_receipt(world)

    with pytest.raises(controller.ControllerRefused) as refusal:
        build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    assert (
        refusal.value.outcome
        == controller.OUTCOME_SERVING_WINDOW_GATES_REFUSED
    )

    scripts = [Path(command[1]).name for command in world.commands]

    # The production serving entry point is never reached, so there is no
    # priced artifact for this window and nothing to publish from it. The
    # advisory windows earlier in the evening are a separate path and are
    # deliberately not disturbed by a T-20 gate refusal.
    assert "run_incumbent_production_serving.py" not in scripts

    assert world.served == []
    assert world.published == []


def test_a_refused_window_gate_exits_non_zero(tmp_path, monkeypatch):
    """Through ``main``, which is where the workflow's exit status comes from."""
    clock = Clock(at(19, 25))

    world = World(tmp_path, schedules=[[game(101, at(23, 0))]], clock=clock)
    world.window_gates_refuse = True

    write_morning_receipt(world)

    as_the_entry_point_sees_it(monkeypatch, world, clock=clock)

    receipt_path = tmp_path / "gate-receipt.json"

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
            "--receipt-path",
            str(receipt_path),
        ]
    )

    assert exit_code == 1

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert (
        receipt["final_outcome"]
        == controller.OUTCOME_SERVING_WINDOW_GATES_REFUSED
    )


class LosesTheRunnerAfterTheFirstWindow(Clock):
    """A clock whose next sleep never returns, the way 2026-10-24 ended.

    Run 40113297251's self-hosted segment published its first window and
    then lost communication with the server while asleep, waiting for a
    window hours away. The job was marked complete underneath it and its
    ``RUNNER_TEMP`` diagnostics went with it, so the only thing that can
    carry the first window forward is what was written to the durable data
    root before the sleep began.
    """

    def __init__(self, now: datetime, *, world_published: list) -> None:
        super().__init__(now)

        self.world_published = world_published

    def sleep(self, seconds: float) -> None:
        if self.world_published:
            raise RuntimeError(
                "The self-hosted runner lost communication with the server."
            )

        super().sleep(seconds)


def test_a_published_window_survives_the_segment_dying_before_the_next(
    tmp_path,
):
    """Evidence is durable before the controller waits for a future window.

    Two games, two T-20 windows hours apart. The first is captured, served
    and published; the runner then disappears while asleep for the second.
    The published window has to be on disk already, outside ``RUNNER_TEMP``,
    because nothing after the sleep ever runs.
    """
    games = [
        game(101, at(20, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    published: list[list[int]] = []

    clock = LosesTheRunnerAfterTheFirstWindow(
        at(19, 25), world_published=published
    )

    world = World(tmp_path, schedules=[games], clock=clock)

    world.published = published

    write_morning_receipt(world)

    segment = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    )

    with pytest.raises(RuntimeError, match="lost communication"):
        segment.run()

    # The first window was published ...
    assert world.published == [[101]]

    # ... and its evidence is in the durable data root, not in the run
    # directory the job deletes.
    assert served_window_ids_for(world) == {"101:T-20m"}

    # The ledger is in the data root the next day still has, and the run
    # directory -- which is ``RUNNER_TEMP`` in production -- holds no copy
    # of it to be the authority instead.
    assert (
        world.data_root / controller.SERVED_WINDOWS_RELATIVE / f"{SLATE}.jsonl"
    ).exists()

    assert not list(
        (world.data_root.parent / "run").rglob("*t20_served_windows*")
    )


def test_the_next_invocation_serves_only_what_the_dead_segment_did_not(
    tmp_path,
):
    """A dead segment leaves a true account of what is left to do.

    The run after the one above sees the first window as served and the
    second as still owed. Neither error is acceptable here: re-serving the
    first would mint a release that carries what the last one carried, and
    reading the second as done would retire a window nothing ever published.
    """
    games = [
        game(101, at(20, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    published: list[list[int]] = []

    dying = LosesTheRunnerAfterTheFirstWindow(
        at(19, 25), world_published=published
    )

    world = World(tmp_path, schedules=[games], clock=dying)

    world.published = published

    write_morning_receipt(world)

    with pytest.raises(RuntimeError, match="lost communication"):
        build(
            world,
            clock=dying,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

    # The run after it, on the same durable data root.
    retry_clock = Clock(at(22, 30))

    retry = World(tmp_path, schedules=[games], clock=retry_clock)

    write_morning_receipt(retry)

    segment = build(
        retry,
        clock=retry_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    )

    plan = segment.plan()

    # The window the dead segment published is not owed again ...
    assert "101:T-20m" not in plan["captured_unserved_windows"]

    assert "101:T-20m" in plan["already_completed_windows"]

    # ... and the one it never reached still is.
    assert plan["pending_windows"] == ["102:T-20m"]


# ----------------------------------------------------------------------
# a serving-window input failure is not a model failure
# ----------------------------------------------------------------------
#
# The two gates this controller asks before serving fail in different
# domains and must therefore fail differently.
#
# ``t20_protocol_compatible`` is about the capture every route reads, and
# its failure refuses the window -- the case the two tests above cover.
#
# ``gate3_role_readiness`` is about role state, which exactly three prop
# routes consume. A game whose capture carries no lineup records costs that
# game those three routes, through the rejection ``15_price_markets``
# already applies per row under ``gate3_role_state_unavailable``, and costs
# nothing else. It is not evidence about the fitted model: changing the
# serving fit repairs no lineup, and a mid-evening authority change is its
# own inconsistency. The tests below pin that whole policy.


def window_gate_decisions(world: World) -> list[dict]:
    """Every serving-window gate decision the controller read back."""
    return [
        json.loads(
            Path(command[command.index("--status-path") + 1]).read_text(
                encoding="utf-8"
            )
        )
        for command in world.commands
        if str(command[1]).endswith("assert_serving_window_gates.py")
    ]


def test_the_routes_a_role_state_failure_can_cost_are_exactly_gate_threes():
    """Which routes consume the gate is read from the module that owns them.

    The policy is "reject the routes that read role state", so what those
    routes are may not be a list somebody retyped. ``gate3_v2`` owns the
    routing and the window-gate entry point imports its set; the pricing
    step and the publisher screen on the same set.
    """
    gates = load_ops("assert_serving_window_gates")

    from nba_prop_quant.gate3_v2 import GATE3_CHANGED_PROPS

    assert gates.GATE3_DEPENDENT_ROUTES == tuple(sorted(GATE3_CHANGED_PROPS))

    assert set(gates.GATE3_DEPENDENT_ROUTES) == set(GATE3_ROUTES)

    # And the gate is route-scoped rather than window-refusing, which is the
    # whole distinction.
    assert gates.ROUTE_SCOPED_GATE == "gate3_role_readiness"

    assert gates.WINDOW_REFUSING_GATES == ("t20_protocol_compatible",)

    assert gates.ROUTE_SCOPED_GATE not in gates.WINDOW_REFUSING_GATES


def test_missing_role_data_does_not_change_which_fit_is_served(tmp_path):
    """The serving authority is identical with and without role state.

    Two runs of the same evening, differing only in whether one game's
    capture carries lineup records. The fit the serving entry point is
    invoked under has to be the same in both, because a lineup is not
    evidence about a model.
    """
    games = [
        game(101, at(20, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    def serve_the_evening(root, *, degrade):
        clock = Clock(at(19, 25))

        world = World(root, schedules=[games], clock=clock)

        world.window_gates_degrade = degrade

        write_morning_receipt(world)

        receipt = build(
            world,
            clock=clock,
            segment_start=SEGMENT_2[0],
            segment_end=SEGMENT_2[1],
        ).run()

        return world, receipt

    healthy, healthy_receipt = serve_the_evening(
        tmp_path / "healthy", degrade=()
    )

    degraded, degraded_receipt = serve_the_evening(
        tmp_path / "degraded", degrade=(101,)
    )

    def served_fits(world):
        return [
            command[command.index("--registry-root") + 1]
            for command in world.commands
            if str(command[1]).endswith(
                "run_incumbent_production_serving.py"
            )
            and "--registry-root" in command
        ]

    assert served_fits(degraded) == served_fits(healthy)

    # Both evenings served and published the same two groups.
    assert degraded.published == healthy.published == [[101], [101, 102]]

    assert (
        degraded_receipt["final_outcome"]
        == healthy_receipt["final_outcome"]
        == controller.OUTCOME_SEGMENT_COMPLETE
    )


def test_a_game_with_no_role_state_costs_only_its_own_three_routes(tmp_path):
    """Unaffected routes serve, and the affected ones are named.

    The slate is served and published exactly as it would have been. What
    changes is that the segment says, per game, which routes it could not
    carry -- so the omission is a recorded fact rather than a silence.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(20, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    world = World(tmp_path, schedules=[games], clock=clock)

    world.window_gates_degrade = (101,)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    # Every route on every game still served and published ...
    assert world.served == [[101], [101, 102]]

    assert world.published == [[101], [101, 102]]

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    # ... except the three that read role state, on the one game that could
    # not supply it, recorded under the reason the pricing step uses.
    degraded = receipt["route_degraded_games"]

    assert {record["game_id"] for record in degraded} == {"101"}

    assert degraded[0]["routes"] == list(GATE3_ROUTES)

    assert [
        record["route_filter_reason"]
        for record in receipt["serving_window_gate_outcomes"]
    ] == ["gate3_role_state_unavailable"] * 2

    # 102 is never named, because 102's role state was fine.
    assert "102" not in {record["game_id"] for record in degraded}


def test_the_live_refresh_is_told_which_games_lost_their_routes(tmp_path):
    """The handover, so the live half can keep what this window cannot beat.

    The controller is the only thing that holds both facts -- the gate stage's
    measurement and the live refresh invocation -- so it passes the first to
    the second. It decides nothing: the games and the routes are read back out
    of the gate receipt, and a window that measured no degradation passes none.
    """
    clock = Clock(at(19, 25))

    games = [
        game(101, at(20, 0)),
        game(102, at(23, 0), home=3, away=4),
    ]

    world = World(tmp_path, schedules=[games], clock=clock)

    world.window_gates_degrade = (101,)

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    def handed(offset: int | None) -> list[list[str]]:
        return [
            [
                command[index + 1]
                for index, item in enumerate(command)
                if item == "--role-state-degraded-game"
            ]
            for command in live_commands(world, offset=offset)
        ]

    gated = handed(controller.PRODUCTION_OFFSET_MINUTES)

    assert gated, "the segment has to have reached the T-20 live refresh"

    expected = "101=" + ",".join(GATE3_ROUTES)

    assert all(carried == [expected] for carried in gated), gated

    # And the game whose role state was fine is never named, so the live half
    # is never told to hold back a game this window can serve in full.
    assert not any(
        item.startswith("102=") for carried in gated for item in carried
    )

    # The measurement is this window's. The advisory windows this segment
    # replays were gated by nothing here, so they are handed nothing: a game
    # held back at a window whose role state nobody measured would be a game
    # held back on no evidence.
    assert all(
        carried == []
        for offset in controller.PROGRESSIVE_LIVE_OFFSETS_MINUTES
        if offset != controller.PRODUCTION_OFFSET_MINUTES
        for carried in handed(offset)
    )


def test_a_window_that_lost_no_routes_hands_over_nothing(tmp_path):
    """The handover is a measurement, not a default.

    The complement of the test above, and the one that makes it mean
    something: an ordinary window must not pass a degradation the gate stage
    did not measure, because every game named is a game held back.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(20, 0)), game(102, at(23, 0), home=3, away=4)]

    world = World(tmp_path, schedules=[games], clock=clock)

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    commands = live_commands(world)

    assert commands, "the segment has to have reached the live refresh"

    for command in commands:
        assert "--role-state-degraded-game" not in command


def test_a_degraded_window_is_not_a_failed_segment(tmp_path):
    """One game's missing lineups must not cost the slate its forecast.

    The directive this implements is explicit about it, and so is the
    repository's own principle: a slate served by slightly less than every
    route is strictly better than no slate.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(20, 0)), game(102, at(23, 0), home=3, away=4)]

    world = World(tmp_path, schedules=[games], clock=clock)

    # Every game, not just one.
    world.window_gates_degrade = (101, 102)

    write_morning_receipt(world)

    receipt = build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    assert receipt["final_outcome"] == controller.OUTCOME_SEGMENT_COMPLETE

    assert receipt["final_outcome"] in controller.OUTCOMES_THAT_SUCCEED

    assert world.published == [[101], [101, 102]]

    assert served_window_ids_for(world) == {"101:T-20m", "102:T-20m"}


def test_no_role_state_failure_reaches_a_promotion_or_a_rollback(tmp_path):
    """Nothing in this path can change the serving authority.

    Asserted two ways, because one of them alone would be weak. The
    controller invokes no promotion or rollback entry point on a degraded
    window, and the window-gate module -- the only thing that learns about
    the failure -- holds a registry handle and never calls either method on
    it.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(20, 0))]

    world = World(tmp_path, schedules=[games], clock=clock)

    world.window_gates_degrade = (101,)

    write_morning_receipt(world)

    build(
        world, clock=clock, segment_start=SEGMENT_2[0], segment_end=SEGMENT_2[1]
    ).run()

    invoked = {Path(str(command[1])).name for command in world.commands}

    assert "run_daily_fit_promotion.py" not in invoked

    assert "seal_promoted_fit_bundle.py" not in invoked

    source = (
        PROJECT / "ops" / "assert_serving_window_gates.py"
    ).read_text(encoding="utf-8")

    for forbidden in (".rollback(", ".promote(", ".demote("):
        assert forbidden not in source, (
            f"the window-gate module names {forbidden}, which would make a "
            "serving-window input failure a model-level event"
        )


def test_a_degraded_window_still_may_not_publish_after_a_tip(tmp_path):
    """Degrading routes does not buy a window any extra time.

    The post-tip guard is taken before the gates are even asked, so a
    window reached after its own game started is refused whether or not its
    role state was ready -- and in particular "most of this game's routes
    were fine" is never a reason to publish a pregame card for a game
    already under way.
    """
    clock = Clock(at(19, 25))

    games = [game(101, at(23, 0))]

    captured_but_unpublished(tmp_path, clock=clock, games=games)

    recovery_clock = Clock(at(23, 10))

    recovery = World(tmp_path, schedules=[games], clock=recovery_clock)

    recovery.window_gates_degrade = (101,)

    write_morning_receipt(recovery)

    receipt = build(
        recovery,
        clock=recovery_clock,
        segment_start=SEGMENT_2[0],
        segment_end=SEGMENT_2[1],
    ).run()

    assert [
        record["window_id"] for record in receipt["post_tip_refused_windows"]
    ] == ["101:T-20m"]

    # Nothing was served and nothing was published, and the gates were
    # never even consulted: the tip settles it first.
    assert recovery.served == []

    assert recovery.published == []

    assert recovery.window_gate_calls == []

    # The window stays unserved and the segment is red for it. Which of the
    # two refusals names the segment is the pre-existing ranking -- a window
    # production is missing outranks why it is missing -- and the point here
    # is only that degrading routes changed neither.
    assert (
        receipt["final_outcome"]
        == controller.OUTCOME_CAPTURED_NOT_SERVED
    )

    assert receipt["final_outcome"] not in controller.OUTCOMES_THAT_SUCCEED

    assert receipt["captured_unserved_windows"] == ["101:T-20m"]
