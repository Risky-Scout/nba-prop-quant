#!/usr/bin/env python3
"""Capture the one market snapshot production consumes, then serve it.

WHAT WAS MISSING
----------------

Production prices from a point-in-time market snapshot. ``15_price_markets.py``
defaults to ``--input-source snapshot`` and ``10_predict_slate.py`` reads the
same bundle, both through ``ProspectiveSnapshotClient``, which accepts exactly
one scheduled capture per game at the primary T-20 window. The producer of
those captures -- ``scripts/11_collect_snapshots.py`` -- existed and worked,
and nothing ever ran it: no workflow, no cron, no launchd unit. The morning
lifecycle fires at 09:37 UTC, roughly thirteen hours before that evening's
first T-20 window, so it could never have consumed a capture of its own slate.

With no capture present, ``ProspectiveSnapshotClient`` discovers no bundles,
``games()`` returns nothing, ``10_predict_slate.py`` prints "No games found"
and exits 0 without writing, and serving reports ``NO_GAMES_ON_SLATE``. A real
slate with no snapshot was therefore indistinguishable from a day with no
games, and the job went green either way. This controller is the missing
producer, and it refuses that silence: inside a due T-20 group, a scheduled
game whose capture is absent is :data:`OUTCOME_REQUIRED_SNAPSHOT_MISSING`, not
an empty slate.

WHAT IT IS NOT
--------------

It is not a second lifecycle. It does not refresh history, train, recalibrate,
register, promote, grade, shadow, monitor, backfill, or install the frozen
model package. The morning lifecycle remains the once-daily heavy and
state-maintenance path and remains the authority that establishes the verified
incumbent runtime; this controller consumes the durable receipt that work left
behind and fails closed when it cannot find it
(:data:`OUTCOME_MORNING_NOT_READY`).

It is not a poller either. The capture schedule's due test is a five-minute
grace around each target, so a naive producer would have to wake every five
minutes all evening. Instead the controller is handed one bounded scheduling
segment, computes the T-20 times inside it from the real schedule, and sleeps
to each of them. Three scheduled invocations a day cover the whole NBA
evening; sleeping is what replaces polling.

And it captures only T-20. The capture schedule also defines T-1440, T-480,
T-180, T-90, T-45 and T-5, which remain available to research and manual
callers; production consumes none of them, so production schedules none of
them. That narrowing is expressed as ``--offset-minutes`` on the existing
collector rather than as a second collector.

ONE CAPTURE PER DUE GROUP
-------------------------

Games that tip together share a T-20 time and are one capture event. The
collector already decides that: ``due_windows`` selects every window whose
target is within grace of now, and ``completed_window_ids`` refuses to capture
a window twice. So the controller's job is to be awake at the right moment and
to invoke the collector once; it never runs a capture per game, per player or
per prop. After a group is captured the slate is served once, the
distributions are derived once, the feed is published once and the page is
published once.

Later groups re-serve the same slate, and that is cumulative rather than
replacing: ``10_predict_slate.py`` enumerates every game with an eligible
bundle, ``15_price_markets.py`` rewrites the whole priced parquet, and the
feed publishes every row in it. A game captured at 18:40 is still priced and
still published after the 21:40 capture.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Deliberately the only package import at module scope. ``snapshot_schedule``
# is standard library only, which is what lets the segment probe run before
# the scientific environment is installed -- and the probe exists so that an
# off-day segment costs almost nothing. Everything heavier
# (:mod:`nba_prop_quant.prospective_snapshot`, the serving entry point, the
# publication entry point) is imported lazily or invoked as a subprocess.
from nba_prop_quant.snapshot_schedule import (  # noqa: E402
    CAPTURE_OFFSETS_MINUTES,
    capture_schedule,
)

NBA_TZ = ZoneInfo("America/New_York")

#: The window production consumes, and the only one it schedules.
PRODUCTION_OFFSET_MINUTES = 20

#: The three scheduled segment starts, as UTC (hour, minute). Each segment
#: owns the half-open five hours that follow it, so together they cover
#: 14:25 UTC on the slate date through 05:25 UTC the next day without
#: overlapping. Stated here and asserted against the workflow by a test.
SEGMENT_STARTS_UTC: tuple[tuple[int, int], ...] = ((0, 25), (14, 25), (19, 25))

SEGMENT_HOURS = 5

#: The complete supported production window for one slate date, as UTC
#: (hour, minute) on the slate date and on the day after it. A T-20 time
#: outside it belongs to no segment, so it would be silently uncaptured;
#: :func:`unsupported_t20_times` refuses the run instead.
SUPPORTED_WINDOW_START_UTC = (14, 25)
SUPPORTED_WINDOW_END_UTC = (5, 25)

#: The collector, and the four entry points the lifecycle already uses.
#: Invoked as subprocesses rather than imported, so the code that captures,
#: serves, derives and publishes is byte-for-byte the code the morning
#: lifecycle runs. Nothing here reimplements any of it.
COLLECTOR_SCRIPT = Path("scripts") / "11_collect_snapshots.py"
SERVING_ENTRY_POINT = Path("ops") / "run_incumbent_production_serving.py"
DISTRIBUTION_ENTRY_POINT = Path("ops") / "nba_distribution_publication.py"
PUBLICATION_ENTRY_POINT = Path("ops") / "run_wizardofodds_publication.py"

#: Canary, always. There is no mode argument and no code path to live
#: publication here: a schedule that could activate live publishing would
#: make every scheduled run an activation decision.
PUBLICATION_MODE = "canary"

#: Where the morning lifecycle's own serving receipt is durably kept. Written
#: by the lifecycle's grading step, which copies the run-temporary receipt
#: into the durable data root so a slate can be graded on a later day. That
#: copy is the handoff this controller reuses; no new receipt is introduced.
MORNING_RECEIPT_RELATIVE = Path("processed") / "incumbent_serving"

#: The serving outcomes that prove the morning lifecycle got all the way
#: through its own serving step for this slate.
#:
#: ``NO_GAMES_ON_SLATE`` is included because on a real slate it is the
#: *expected* morning outcome: at 09:37 UTC no T-20 capture for that evening
#: exists yet, so the prediction script correctly reports no games. What it
#: proves is the part this controller needs -- the incumbent resolved and the
#: frozen bundle verified, both of which happen before the readiness branch.
#: ``NO_PRODUCTION_SLATE`` is excluded: that is the preseason no-op, and there
#: is no slate to capture. A failed serving run writes no durable copy at all.
#:
#: Stated rather than imported from the serving module, which pulls the fit
#: registry and the frozen-bundle installer behind it and would put that whole
#: stack in front of the lightweight probe. A test pins these against
#: ``run_incumbent_production_serving``'s own constants.
MORNING_OUTCOMES_THAT_PROVE_READINESS: frozenset[str] = frozenset(
    {"SERVED", "NO_GAMES_ON_SLATE"}
)

#: The refresh outcomes a slate may be served under, pinned by the same test
#: against ``run_incumbent_production_serving.REFRESH_OUTCOMES_THAT_SERVE``.
REFRESH_OUTCOMES_THAT_SERVE: frozenset[str] = frozenset(
    {"REFRESHED", "OPENING_DAY_NO_REFRESH"}
)

#: What the serving step records when it priced the slate.
SERVING_OUTCOME_SERVED = "SERVED"
SERVING_OUTCOME_NO_GAMES = "NO_GAMES_ON_SLATE"

OUTCOME_NO_WINDOWS = "NO_T20_WINDOWS_IN_SEGMENT"
OUTCOME_SEGMENT_COMPLETE = "T20_SEGMENT_COMPLETE"
OUTCOME_MORNING_NOT_READY = "MORNING_PRODUCTION_NOT_READY"

#: Distinguished from the outcome above because the operator action differs.
#: ``MORNING_PRODUCTION_NOT_READY`` means no proof of a verified incumbent
#: exists for this slate and the morning lifecycle has to run. This one means
#: the proof exists and is sound, and the only thing wrong with it is that
#: accepted production code was merged forward after it was written -- so the
#: supported action is the readiness refresh, not a lifecycle run. Both still
#: refuse to serve: the receipt is required either way and neither is bypassed.
OUTCOME_MORNING_STALE_AFTER_MERGE = "MORNING_PRODUCTION_STALE_AFTER_MERGE"

#: The workflow that re-issues the receipt, named in the refusal so the
#: message says what to do rather than only what is wrong.
READINESS_REFRESH_WORKFLOW = "nba_production_readiness_refresh.yml"
OUTCOME_REQUIRED_SNAPSHOT_MISSING = "REQUIRED_T20_SNAPSHOT_MISSING"
OUTCOME_MISSED_WINDOW = "MISSED_T20_WINDOW"
OUTCOME_UNSUPPORTED_TIME = "UNSUPPORTED_T20_TIME"
OUTCOME_FAILED = "T20_CONTROLLER_FAILED"

OUTCOMES_THAT_SUCCEED: frozenset[str] = frozenset(
    {OUTCOME_NO_WINDOWS, OUTCOME_SEGMENT_COMPLETE}
)

PUBLICATION_OUTCOME_PUBLISHED = "PUBLISHED"
PUBLICATION_OUTCOME_NOT_SERVED = "NOT_SERVED_NOTHING_TO_PUBLISH"
PUBLICATION_OUTCOME_SKIPPED = "SKIPPED_ALREADY_PUBLISHED_THIS_STATE"

EXIT_OK = 0
EXIT_FAILED = 1


class ControllerRefused(RuntimeError):
    """A fail-closed refusal carrying the outcome it must be reported as."""

    def __init__(self, outcome: str, message: str) -> None:
        super().__init__(message)

        self.outcome = outcome


# ----------------------------------------------------------------------
# time
# ----------------------------------------------------------------------


def parse_utc(raw: str, *, what: str) -> datetime:
    text = str(raw)

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        value = datetime.fromisoformat(text)
    except ValueError as error:
        raise ControllerRefused(
            OUTCOME_FAILED, f"{what} is not an ISO timestamp: {raw!r}"
        ) from error

    if value.tzinfo is None:
        raise ControllerRefused(
            OUTCOME_FAILED, f"{what} must carry a timezone: {raw!r}"
        )

    return value.astimezone(UTC)


def slate_date_for(moment: datetime) -> str:
    """The NBA slate a moment belongs to, in US/Eastern.

    Not the UTC calendar date. The third segment starts at 00:25 UTC, which is
    the evening before in Eastern time, and the slate it is capturing for is
    that evening's -- so a UTC date here would send the whole late segment at
    tomorrow's schedule and capture nothing.
    """
    return moment.astimezone(NBA_TZ).date().isoformat()


def resolve_segment(
    *, now_utc: datetime, cron: str | None = None
) -> tuple[datetime, datetime]:
    """The segment this invocation owns, as a half-open UTC interval.

    Resolved from the cron expression that fired when there is one, because
    runner pickup on this repository's self-hosted runner has been delayed by
    hours and the clock at startup is therefore not evidence of which segment
    was scheduled. The scheduled time is: it is the most recent occurrence of
    the cron's time of day at or before now. A badly delayed pickup then lands
    on the right segment and reports its windows as missed, which is the
    truth, rather than quietly adopting the next segment's.
    """
    if cron:
        fields = str(cron).split()

        if len(fields) != 5:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"--cron must be a five-field cron expression, got {cron!r}",
            )

        try:
            minute, hour = int(fields[0]), int(fields[1])
        except ValueError as error:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"--cron must name a fixed minute and hour, got {cron!r}",
            ) from error

        if (hour, minute) not in SEGMENT_STARTS_UTC:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"{cron!r} is not one of this controller's segment starts "
                + ", ".join(
                    f"{h:02d}:{m:02d}" for h, m in sorted(SEGMENT_STARTS_UTC)
                ),
            )

        start = now_utc.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )

        if start > now_utc:
            start -= timedelta(days=1)

        return start, start + timedelta(hours=SEGMENT_HOURS)

    for hour, minute in sorted(SEGMENT_STARTS_UTC, reverse=True):
        start = now_utc.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )

        if start > now_utc:
            start -= timedelta(days=1)

        end = start + timedelta(hours=SEGMENT_HOURS)

        if start <= now_utc < end:
            return start, end

    raise ControllerRefused(
        OUTCOME_FAILED,
        f"{now_utc.isoformat()} falls in no scheduled T-20 segment, so the "
        "segment this invocation owns cannot be established; pass --cron or "
        "--segment-start-utc explicitly",
    )


def supported_window(slate_date: str) -> tuple[datetime, datetime]:
    """The complete production window for one slate, as a UTC interval."""
    day = datetime.fromisoformat(slate_date).date()

    start_hour, start_minute = SUPPORTED_WINDOW_START_UTC
    end_hour, end_minute = SUPPORTED_WINDOW_END_UTC

    return (
        datetime(
            day.year, day.month, day.day, start_hour, start_minute, tzinfo=UTC
        ),
        datetime(
            day.year, day.month, day.day, end_hour, end_minute, tzinfo=UTC
        )
        + timedelta(days=1),
    )


# ----------------------------------------------------------------------
# the T-20 schedule
# ----------------------------------------------------------------------


def t20_rows(
    games: Sequence[dict[str, Any]], *, offset_minutes: int
) -> list[dict[str, Any]]:
    """One row per game at the production offset, and nothing else.

    The research offsets are computed by the shared schedule and then dropped
    here, which is the whole of production's capture policy: the capability
    stays, the scheduling does not.
    """
    try:
        rows = capture_schedule(list(games))
    except ValueError as error:
        raise ControllerRefused(
            OUTCOME_FAILED,
            "the slate schedule cannot be turned into capture windows, so "
            f"no T-20 time can be computed: {error}",
        ) from error

    return [
        row for row in rows if int(row["offset_minutes"]) == int(offset_minutes)
    ]


def unsupported_t20_times(
    rows: Sequence[dict[str, Any]], *, slate_date: str
) -> list[dict[str, Any]]:
    """T-20 times no segment covers.

    Returned rather than skipped. A game whose capture window falls outside
    14:25 UTC through 05:25 UTC the next day would be captured by nobody, and
    omitting it silently is how a slate ends up published with a game missing.
    """
    start, end = supported_window(slate_date)

    return [
        {
            "game_id": int(row["game_id"]),
            "window_id": str(row["window_id"]),
            "capture_at_utc": row["capture_at_utc"].isoformat(),
            "tip_utc": row["tip_utc"].isoformat(),
        }
        for row in rows
        if not (start <= row["capture_at_utc"] < end)
    ]


def owned_rows(
    rows: Sequence[dict[str, Any]],
    *,
    segment_start: datetime,
    segment_end: datetime,
) -> list[dict[str, Any]]:
    """The rows whose T-20 time this segment owns: ``[start, end)``."""
    return [
        row
        for row in rows
        if segment_start <= row["capture_at_utc"] < segment_end
    ]


def completed_window_ids(capture_run_path: Path) -> set[str]:
    """The windows already captured for this slate.

    Read from the collector's own append-only capture-run log, which is the
    capture authority on restart and on retry. Reimplemented here rather than
    imported because the collector is a numbered script that pulls the API
    client and the settings model behind it, and the probe must stay light; a
    test pins this against the collector's own function.
    """
    if not capture_run_path.exists():
        return set()

    completed: set[str] = set()

    for line in capture_run_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row.get("snapshot_type") != "capture_run":
            continue

        for window_id in (row.get("payload") or {}).get("window_ids", []):
            completed.add(str(window_id))

    return completed


def resolve_snapshot_dir(data_root: Path) -> Path:
    """Where the collector writes and the serving scripts read.

    One directory, checked rather than assumed. The collector writes to
    ``Settings.snapshot_dir``, which is ``$NBA_PROP_DATA_DIR/snapshots``; the
    serving scripts read through ``resolve_gate3_snapshot_dir``, which honours
    ``NBA_PROP_GATE3_SNAPSHOT_DIR`` first. Production sets no override and the
    two therefore agree. If one is ever set to somewhere else, the producer
    and the consumer would be looking at different trees and the capture would
    verify here and be invisible to pricing, so that is refused.
    """
    snapshot_dir = (Path(data_root) / "snapshots").expanduser().resolve()

    override = os.environ.get("NBA_PROP_GATE3_SNAPSHOT_DIR")

    if override:
        consumed = Path(override).expanduser().resolve()

        if consumed != snapshot_dir:
            raise ControllerRefused(
                OUTCOME_FAILED,
                "NBA_PROP_GATE3_SNAPSHOT_DIR points the serving scripts at "
                f"{consumed} while the collector writes to {snapshot_dir}, so "
                "a capture made here would be invisible to pricing",
            )

    return snapshot_dir


# ----------------------------------------------------------------------
# the morning lifecycle's handoff
# ----------------------------------------------------------------------


def _recorded_sha_is_ancestor(
    project_root: Path, *, recorded_sha: str, production_sha: str
) -> bool:
    """Whether the receipt's code is strictly behind the code about to serve.

    Answered from git rather than inferred, and answered quietly: this only
    decides which refusal the operator is handed, so a repository git cannot
    read is not an error here -- it just means the staleness cannot be
    attributed to a merge, and the general refusal is the right one.
    """
    if not recorded_sha or recorded_sha == production_sha:
        return False

    for candidate in (recorded_sha, production_sha):
        if (
            subprocess.run(
                ("git", "cat-file", "-e", f"{candidate}^{{commit}}"),
                cwd=project_root,
                capture_output=True,
                check=False,
            ).returncode
            != 0
        ):
            return False

    return (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", recorded_sha, production_sha),
            cwd=project_root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def morning_readiness(
    *,
    data_root: Path,
    slate_date: str,
    production_sha: str,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Prove, from durable state, that this morning established the incumbent.

    The morning lifecycle installs and verifies the frozen runtime bundle and
    then serves; its serving step resolves the incumbent and re-hashes the
    bundle before it looks at readiness at all, so a receipt that exists is a
    receipt whose bundle verified. The lifecycle already copies that receipt
    into the durable data root, so this is a read of existing state rather
    than a new artifact, and it is why this controller never runs the frozen
    package installer: the authority it would establish is already established
    and recorded.

    Every check is about *this* slate and *this* code. A receipt for another
    date, or produced by another production SHA, does not prove the runtime
    serving is about to load was verified, and a controller that accepted it
    would be reusing yesterday's proof.

    One mismatch is singled out and still refused. When the recorded SHA is a
    strict ancestor of the running one, the receipt is sound and the only
    thing wrong with it is that accepted production code was merged after it
    was written. That is a routine event with a supported remedy -- the
    readiness refresh re-issues the receipt under the new revision -- and
    telling the two cases apart is the difference between an operator running
    the right thing and an operator losing an evening's slate. The refusal
    still happens: nothing here serves on a receipt that describes other code.
    """
    path = Path(data_root) / MORNING_RECEIPT_RELATIVE / f"{slate_date}.json"

    def refuse(reason: str) -> ControllerRefused:
        return ControllerRefused(
            OUTCOME_MORNING_NOT_READY,
            f"the morning production lifecycle has not proven a verified "
            f"incumbent runtime for {slate_date}: {reason}",
        )

    if not path.exists():
        raise refuse(
            f"{path} does not exist, so no lifecycle run has recorded a "
            "verified incumbent for this slate"
        )

    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise refuse(f"{path} is not readable JSON ({error})") from error

    if not isinstance(receipt, dict):
        raise refuse(f"{path} does not hold a receipt object")

    recorded_slate = str(receipt.get("slate_date") or "")

    if recorded_slate != slate_date:
        raise refuse(
            f"{path} is the receipt for {recorded_slate or 'no slate'}"
        )

    recorded_sha = str(receipt.get("production_code_sha") or "")

    if recorded_sha != str(production_sha):
        if _recorded_sha_is_ancestor(
            Path(project_root),
            recorded_sha=recorded_sha,
            production_sha=str(production_sha),
        ):
            raise ControllerRefused(
                OUTCOME_MORNING_STALE_AFTER_MERGE,
                f"the readiness receipt for {slate_date} is sound but stale: "
                f"{path} was produced by production SHA {recorded_sha}, "
                f"accepted production code has since moved forward to "
                f"{production_sha}, and the runtime that receipt verified is "
                "therefore not the code about to serve. This is the one "
                "staleness with a supported remedy: run "
                f"{READINESS_REFRESH_WORKFLOW} to re-issue the receipt under "
                "the current revision, which re-verifies the frozen runtime "
                "and fails closed if any fitted artifact moved. Serving is "
                "refused until it has",
            )

        raise refuse(
            f"{path} was produced by production SHA "
            f"{recorded_sha or 'nothing'} and this run is "
            f"{production_sha}, so the runtime it verified is not the code "
            "about to serve"
        )

    outcome = str(receipt.get("outcome") or "")

    if outcome not in MORNING_OUTCOMES_THAT_PROVE_READINESS:
        raise refuse(
            f"{path} records the serving outcome {outcome or 'nothing'}, "
            "which is not one of "
            + ", ".join(sorted(MORNING_OUTCOMES_THAT_PROVE_READINESS))
        )

    readiness = receipt.get("slate_readiness") or {}

    if readiness.get("ready") is not True:
        raise refuse(f"{path} records that the slate was not ready to serve")

    refresh_outcome = str(readiness.get("refresh_outcome") or "")

    if refresh_outcome not in REFRESH_OUTCOMES_THAT_SERVE:
        raise refuse(
            f"{path} records the refresh outcome "
            f"{refresh_outcome or 'nothing'}, which is not one a slate may "
            "be served under"
        )

    model_dir = Path(str(receipt.get("serving_model_dir") or ""))
    bundle_root = Path(str(receipt.get("serving_bundle_root") or ""))

    for label, candidate in (
        ("serving_model_dir", model_dir),
        ("serving_bundle_root", bundle_root),
    ):
        if not str(candidate) or not candidate.is_dir():
            raise refuse(
                f"{path} names {label} {str(candidate) or 'nothing'}, which "
                "is not a directory on this machine"
            )

    authority = str(receipt.get("incumbent_authority") or "")
    version = str(receipt.get("incumbent_version") or "")

    if not authority or not version:
        raise refuse(
            f"{path} does not name the incumbent authority it verified"
        )

    return {
        "incumbent_authority": authority,
        "incumbent_fit_id": receipt.get("incumbent_fit_id"),
        "incumbent_version": version,
        "morning_outcome": outcome,
        "production_code_sha": recorded_sha,
        "receipt_path": str(path),
        "recorded_at": receipt.get("generated_at"),
        "refresh_outcome": refresh_outcome,
        "serving_bundle_root": str(bundle_root),
        "serving_model_dir": str(model_dir),
        "slate_date": recorded_slate,
    }


# ----------------------------------------------------------------------
# the controller
# ----------------------------------------------------------------------


def _bdl_games(slate_date: str) -> list[dict[str, Any]]:
    """The authoritative slate schedule, from the source the collector uses.

    Imported here rather than at module scope so that the probe -- which does
    call this -- needs only ``httpx`` and ``tenacity``, and so that a test can
    inject a schedule without reaching the network.
    """
    from nba_prop_quant.api import BDLClient

    api_key = os.environ.get("BDL_API_KEY")

    if not api_key:
        raise ControllerRefused(
            OUTCOME_FAILED,
            "BDL_API_KEY is not set, so the slate schedule cannot be read "
            "and no T-20 time can be computed",
        )

    with BDLClient(
        api_key=api_key,
        base_url=os.environ.get(
            "BDL_BASE_URL", "https://api.balldontlie.io"
        ),
    ) as client:
        return list(client.games(dates=[slate_date]))


def _run_subprocess(
    command: Sequence[str], *, environment: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class T20Controller:
    """One bounded segment of T-20 capture, serving and canary publication.

    Every collaborator that touches the world -- the clock, sleeping, the
    schedule source, subprocess execution -- is injected, because the
    behaviour worth testing is entirely about time and ordering and a test
    that had to wait for a real T-20 window could not assert any of it.
    """

    def __init__(
        self,
        *,
        slate_date: str,
        segment_start: datetime,
        segment_end: datetime,
        data_root: Path,
        production_sha: str,
        project_root: Path = PROJECT_ROOT,
        registry_root: Path | None = None,
        run_dir: Path | None = None,
        offset_minutes: int = PRODUCTION_OFFSET_MINUTES,
        grace_minutes: int = 5,
        summary_path: Path | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        schedule_source: Callable[[str], list[dict[str, Any]]] | None = None,
        run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        if offset_minutes not in CAPTURE_OFFSETS_MINUTES:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"--offset-minutes {offset_minutes} is not a window the "
                f"capture schedule defines {list(CAPTURE_OFFSETS_MINUTES)}",
            )

        if grace_minutes <= 0:
            raise ControllerRefused(
                OUTCOME_FAILED, "--grace-minutes must be positive"
            )

        if segment_end <= segment_start:
            raise ControllerRefused(
                OUTCOME_FAILED,
                "the segment must end after it starts, got "
                f"{segment_start.isoformat()} to {segment_end.isoformat()}",
            )

        self.slate_date = str(slate_date)
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.data_root = Path(data_root)
        self.production_sha = str(production_sha)
        self.project_root = Path(project_root)
        self.registry_root = registry_root
        self.offset_minutes = int(offset_minutes)
        self.grace = timedelta(minutes=int(grace_minutes))
        self.grace_minutes = int(grace_minutes)
        self.summary_path = summary_path

        self.run_dir = Path(
            run_dir
            if run_dir is not None
            else Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "t20"
        )

        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleep = sleep or time.sleep
        self.schedule_source = schedule_source or _bdl_games
        self.run_command = run_command or _run_subprocess

        self.snapshot_dir = resolve_snapshot_dir(self.data_root)

        self.capture_run_path = (
            self.snapshot_dir / "capture_runs" / f"{self.slate_date}.jsonl"
        )

        self.captures: list[dict[str, Any]] = []
        self.serving: list[dict[str, Any]] = []
        self.distributions: list[dict[str, Any]] = []
        self.feed: list[dict[str, Any]] = []
        self.page: list[dict[str, Any]] = []
        self.deferred: list[dict[str, Any]] = []
        self.rescheduled: list[dict[str, Any]] = []
        self.withdrawn: list[dict[str, Any]] = []

        self._published_priced_digest: str | None = None

    # -- planning ------------------------------------------------------

    def plan(self) -> dict[str, Any]:
        """What this segment has to do, before anything expensive happens.

        Deliberately cheap: one schedule read and one read of the capture-run
        log. An off day, or a segment with no T-20 window in it, is answered
        here and the caller never installs the scientific environment, never
        verifies the frozen tree, never serves and never publishes.
        """
        games = list(self.schedule_source(self.slate_date))

        rows = t20_rows(games, offset_minutes=self.offset_minutes)

        unsupported = unsupported_t20_times(rows, slate_date=self.slate_date)

        if unsupported:
            start, end = supported_window(self.slate_date)

            raise ControllerRefused(
                OUTCOME_UNSUPPORTED_TIME,
                f"{len(unsupported)} T-20 window(s) for {self.slate_date} "
                f"fall outside the supported production window "
                f"{start.isoformat()} to {end.isoformat()}, so no segment "
                "would capture them: "
                + ", ".join(
                    f"{row['window_id']} at {row['capture_at_utc']}"
                    for row in unsupported
                ),
            )

        owned = owned_rows(
            rows,
            segment_start=self.segment_start,
            segment_end=self.segment_end,
        )

        completed = completed_window_ids(self.capture_run_path)

        pending = [
            row for row in owned if str(row["window_id"]) not in completed
        ]

        return {
            "games_discovered": len(games),
            "t20_windows_for_slate": [str(row["window_id"]) for row in rows],
            "t20_windows_in_segment": [
                str(row["window_id"]) for row in owned
            ],
            "already_completed_windows": sorted(
                str(row["window_id"])
                for row in owned
                if str(row["window_id"]) in completed
            ),
            "pending_windows": [str(row["window_id"]) for row in pending],
            "rows": rows,
            "pending_rows": pending,
        }

    # -- the segment ---------------------------------------------------

    def run(self) -> dict[str, Any]:
        started_at = self.clock()

        plan = self.plan()

        if not plan["pending_rows"]:
            return self._receipt(
                plan=plan,
                started_at=started_at,
                outcome=OUTCOME_NO_WINDOWS,
                reason=(
                    f"the segment {self.segment_start.isoformat()} to "
                    f"{self.segment_end.isoformat()} contains no uncaptured "
                    f"T-{self.offset_minutes} window for {self.slate_date}, "
                    "so no capture, serving or publication was performed"
                ),
            )

        # Only now, with work to do, is the morning handoff required. A day
        # with nothing to capture must not be reported as a lifecycle failure.
        readiness = morning_readiness(
            data_root=self.data_root,
            slate_date=self.slate_date,
            production_sha=self.production_sha,
            project_root=self.project_root,
        )

        self._work(plan, readiness)

        return self._receipt(
            plan=plan,
            started_at=started_at,
            outcome=OUTCOME_SEGMENT_COMPLETE,
            reason=(
                f"{len(self.captures)} T-{self.offset_minutes} capture(s) "
                f"completed for {self.slate_date}, each served and published "
                "to the canary destination"
            ),
            readiness=readiness,
        )

    def _work(self, plan: dict[str, Any], readiness: dict[str, Any]) -> None:
        rows = {str(row["window_id"]): row for row in plan["pending_rows"]}

        handled: set[str] = set()

        # Re-planning happens when a tip moved later while the controller was
        # asleep. Bounded so a schedule that kept moving cannot loop forever;
        # in practice one re-plan is the most that ever happens.
        replans_remaining = len(rows) + 4

        while True:
            outstanding = {
                window_id: row
                for window_id, row in rows.items()
                if window_id not in handled
            }

            if not outstanding:
                return

            due = min(row["capture_at_utc"] for row in outstanding.values())

            group = [
                row
                for row in outstanding.values()
                if row["capture_at_utc"] == due
            ]

            self._sleep_until(due)

            # One schedule read per capture attempt, immediately before the
            # capture. Not a poll: between groups the controller is asleep.
            fresh = {
                str(row["window_id"]): row
                for row in t20_rows(
                    self.schedule_source(self.slate_date),
                    offset_minutes=self.offset_minutes,
                )
            }

            now = self.clock()

            ready: list[dict[str, Any]] = []
            moved_later = False

            for row in sorted(group, key=lambda item: int(item["game_id"])):
                window_id = str(row["window_id"])

                current = fresh.get(window_id)

                if current is None:
                    self.withdrawn.append(
                        {
                            "window_id": window_id,
                            "observed_at_utc": now.isoformat(),
                            "reason": (
                                "the game is no longer on the slate schedule, "
                                "so it has no T-20 window to capture"
                            ),
                        }
                    )

                    handled.add(window_id)

                    continue

                target = current["capture_at_utc"]

                if now >= target + self.grace:
                    raise ControllerRefused(
                        OUTCOME_MISSED_WINDOW,
                        f"{window_id} was due at {target.isoformat()} and it "
                        f"is now {now.isoformat()}, which is past the "
                        f"{self.grace_minutes}-minute capture grace, so the "
                        "required T-20 capture can no longer be made and no "
                        "later capture may be labelled T-20",
                    )

                if now < target:
                    if target >= self.segment_end:
                        self.deferred.append(
                            {
                                "window_id": window_id,
                                "moved_to_utc": target.isoformat(),
                                "observed_at_utc": now.isoformat(),
                                "reason": (
                                    "the tip moved and this window now falls "
                                    "in a later scheduling segment, which "
                                    "owns it"
                                ),
                            }
                        )

                        handled.add(window_id)

                    else:
                        self.rescheduled.append(
                            {
                                "window_id": window_id,
                                "was_due_utc": due.isoformat(),
                                "moved_to_utc": target.isoformat(),
                                "observed_at_utc": now.isoformat(),
                            }
                        )

                        rows[window_id] = current

                        moved_later = True

                    continue

                ready.append(current)

            if not ready:
                if moved_later:
                    replans_remaining -= 1

                    if replans_remaining < 0:
                        raise ControllerRefused(
                            OUTCOME_FAILED,
                            "the slate schedule kept moving and the segment "
                            "plan did not settle, so no capture was attempted",
                        )

                continue

            self._capture_group(ready, now=now, readiness=readiness)

            for row in ready:
                handled.add(str(row["window_id"]))

    def _sleep_until(self, target: datetime) -> None:
        remaining = (target - self.clock()).total_seconds()

        if remaining > 0:
            self.sleep(remaining)

    # -- one due group -------------------------------------------------

    def _capture_group(
        self,
        group: Sequence[dict[str, Any]],
        *,
        now: datetime,
        readiness: dict[str, Any],
    ) -> None:
        """Capture one due group, then serve and publish the slate once."""
        index = len(self.captures) + 1

        stage_dir = self.run_dir / f"group-{index:02d}"

        stage_dir.mkdir(parents=True, exist_ok=True)

        window_ids = sorted(str(row["window_id"]) for row in group)

        game_ids = sorted(int(row["game_id"]) for row in group)

        capture = self.run_command(
            [
                sys.executable,
                str(COLLECTOR_SCRIPT),
                "--date",
                self.slate_date,
                "--scheduled",
                "--offset-minutes",
                str(self.offset_minutes),
                "--now-utc",
                now.isoformat(),
                "--grace-minutes",
                str(self.grace_minutes),
            ],
            environment=self._environment(),
            cwd=self.project_root,
        )

        if capture.returncode != 0:
            raise ControllerRefused(
                OUTCOME_REQUIRED_SNAPSHOT_MISSING,
                f"{COLLECTOR_SCRIPT} exited {capture.returncode} for "
                f"{', '.join(window_ids)}: "
                + ((capture.stderr or capture.stdout) or "")[-2000:],
            )

        bundles = self._verify_captured(game_ids, window_ids=window_ids)

        self.captures.append(
            {
                "attempted_at_utc": now.isoformat(),
                "capture_ids": sorted(
                    {bundle["capture_id"] for bundle in bundles}
                ),
                "due_at_utc": min(
                    row["capture_at_utc"] for row in group
                ).isoformat(),
                "game_ids": game_ids,
                "verified_bundles": bundles,
                "window_ids": window_ids,
            }
        )

        self._serve_and_publish(stage_dir, readiness=readiness)

    def _verify_captured(
        self, game_ids: Sequence[int], *, window_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Prove the capture the pricing path requires is actually there.

        The same selection pricing performs, performed here: exactly one
        scheduled capture for the window, no recorded errors, every component
        hash recomputed, the capture id recomputed, the target game present
        and the capture pre-tip. A game that is on the schedule and reaches
        this point without an eligible bundle is the failure this controller
        exists to make loud -- left alone it would reach serving as an empty
        slate and publish nothing with a green job.

        Imported here because ``prospective_snapshot`` pulls pandas, which the
        probe must not need.
        """
        from nba_prop_quant.prospective_snapshot import select_capture_bundle

        verified: list[dict[str, Any]] = []

        for game_id in game_ids:
            try:
                bundle = select_capture_bundle(
                    self.snapshot_dir,
                    date=self.slate_date,
                    game_id=int(game_id),
                    offset_minutes=self.offset_minutes,
                )
            except RuntimeError as error:
                raise ControllerRefused(
                    OUTCOME_REQUIRED_SNAPSHOT_MISSING,
                    f"game {game_id} is on the {self.slate_date} schedule and "
                    f"its T-{self.offset_minutes} window "
                    f"({', '.join(window_ids)}) is due, but no eligible "
                    f"scheduled capture can be selected for it: {error}",
                ) from error

            verified.append(
                {
                    "capture_id": bundle.capture_id,
                    "captured_at_utc": bundle.captured_at,
                    "game_id": int(game_id),
                    "window_id": bundle.window_id,
                }
            )

        return verified

    # -- serve, derive, publish ----------------------------------------

    def _environment(self) -> dict[str, str]:
        """The data root the capture and serving subprocesses resolve from.

        Passed explicitly for the same reason the serving step passes the
        model directory explicitly: the collector writes to
        ``Settings.snapshot_dir``, which is derived from this variable, and a
        capture written somewhere other than where pricing reads would verify
        here and be invisible there.
        """
        environment = dict(os.environ)

        environment["NBA_PROP_DATA_DIR"] = str(self.data_root)

        return environment

    def _stage(
        self,
        label: str,
        command: Sequence[str],
        *,
        outcome: str = OUTCOME_FAILED,
    ) -> subprocess.CompletedProcess[str]:
        result = self.run_command(
            list(command),
            environment=self._environment(),
            cwd=self.project_root,
        )

        if result.returncode != 0:
            raise ControllerRefused(
                outcome,
                f"{label} exited {result.returncode} for {self.slate_date}: "
                + ((result.stderr or result.stdout) or "")[-2000:],
            )

        return result

    def _serve_and_publish(
        self, stage_dir: Path, *, readiness: dict[str, Any]
    ) -> None:
        serving_receipt = stage_dir / "incumbent_serving.json"

        feed_receipt = stage_dir / "wizardofodds_feed.json"

        page_receipt = stage_dir / "wizardofodds_page.json"

        serving_command = [
            sys.executable,
            str(SERVING_ENTRY_POINT),
            "--slate-date",
            self.slate_date,
            "--data-root",
            str(self.data_root),
            "--production-sha",
            self.production_sha,
            "--model-dir",
            readiness["serving_model_dir"],
            "--frozen-bundle-root",
            readiness["serving_bundle_root"],
            "--receipt-path",
            str(serving_receipt),
        ]

        if self.registry_root is not None:
            serving_command += ["--registry-root", str(self.registry_root)]

        if self.summary_path is not None:
            serving_command += ["--summary-path", str(self.summary_path)]

        self._stage(str(SERVING_ENTRY_POINT), serving_command)

        served = self._read_receipt(serving_receipt, what="serving receipt")

        outcome = str(served.get("outcome") or "")

        if outcome == SERVING_OUTCOME_NO_GAMES:
            raise ControllerRefused(
                OUTCOME_REQUIRED_SNAPSHOT_MISSING,
                f"{SERVING_ENTRY_POINT} reported {outcome} for "
                f"{self.slate_date} immediately after a verified "
                f"T-{self.offset_minutes} capture of that slate's due games, "
                "so a real slate was about to be recorded as an empty one",
            )

        if outcome != SERVING_OUTCOME_SERVED:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"{SERVING_ENTRY_POINT} recorded the outcome "
                f"{outcome or 'nothing'} for {self.slate_date}, which is not "
                "a served slate",
            )

        self.serving.append(
            {
                "outcome": outcome,
                "priced_market_artifact_sha256": str(
                    (served.get("priced_market_artifact") or {}).get("sha256")
                    or ""
                ),
                "receipt_path": str(serving_receipt),
                "slate_date": str(served.get("slate_date") or ""),
            }
        )

        self._stage(
            str(DISTRIBUTION_ENTRY_POINT),
            [
                sys.executable,
                str(DISTRIBUTION_ENTRY_POINT),
                "--date",
                self.slate_date,
                "--data-root",
                str(self.data_root),
                "--model-dir",
                readiness["serving_model_dir"],
            ],
        )

        self.distributions.append(
            {
                "outcome": "DERIVED",
                "slate_date": self.slate_date,
            }
        )

        priced_digest = self.serving[-1]["priced_market_artifact_sha256"]

        # Publication idempotency, kept as small as it can be: if the priced
        # artifact is byte-identical to the one this invocation already
        # published, republishing would mint a second release of the same
        # rows. Proven from this invocation's own receipt rather than by
        # inventing a durable publication ledger.
        if priced_digest and priced_digest == self._published_priced_digest:
            for record in (self.feed, self.page):
                record.append(
                    {
                        "outcome": PUBLICATION_OUTCOME_SKIPPED,
                        "priced_market_artifact_sha256": priced_digest,
                        "reason": (
                            "this invocation already published the identical "
                            "priced artifact, so the release it would make is "
                            "the release already serving"
                        ),
                        "slate_date": self.slate_date,
                    }
                )

            return

        feed_command = [
            sys.executable,
            str(PUBLICATION_ENTRY_POINT),
            "--stage",
            "feed",
            "--mode",
            PUBLICATION_MODE,
            "--slate-date",
            self.slate_date,
            "--data-root",
            str(self.data_root),
            "--serving-receipt",
            str(serving_receipt),
            "--receipt-path",
            str(feed_receipt),
        ]

        if self.summary_path is not None:
            feed_command += ["--summary-path", str(self.summary_path)]

        self._stage(
            f"{PUBLICATION_ENTRY_POINT} --stage feed",
            feed_command,
        )

        published = self._read_receipt(
            feed_receipt, what="feed publication receipt"
        )

        self.feed.append(
            {
                "outcome": str(published.get("outcome") or ""),
                "release_id": str(
                    (published.get("release") or {}).get("release_id") or ""
                ),
                "receipt_path": str(feed_receipt),
                "slate_date": str(published.get("slate_date") or ""),
            }
        )

        if str(published.get("outcome") or "") != PUBLICATION_OUTCOME_PUBLISHED:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"{PUBLICATION_ENTRY_POINT} --stage feed recorded "
                f"{published.get('outcome') or 'nothing'} for a slate this "
                "controller served, so the page would be built from a "
                "release this run did not publish",
            )

        page_command = [
            sys.executable,
            str(PUBLICATION_ENTRY_POINT),
            "--stage",
            "page",
            "--mode",
            PUBLICATION_MODE,
            "--slate-date",
            self.slate_date,
            "--feed-receipt",
            str(feed_receipt),
            "--receipt-path",
            str(page_receipt),
        ]

        if self.summary_path is not None:
            page_command += ["--summary-path", str(self.summary_path)]

        self._stage(
            f"{PUBLICATION_ENTRY_POINT} --stage page",
            page_command,
        )

        page = self._read_receipt(
            page_receipt, what="page publication receipt"
        )

        self.page.append(
            {
                "outcome": str(page.get("outcome") or ""),
                "page_sha256": str(page.get("page_sha256") or ""),
                "receipt_path": str(page_receipt),
                "slate_date": str(page.get("slate_date") or ""),
            }
        )

        self._published_priced_digest = priced_digest or None

    def _read_receipt(self, path: Path, *, what: str) -> dict[str, Any]:
        if not path.exists():
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"{path} does not exist, so there is no {what} to read",
            )

        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ControllerRefused(
                OUTCOME_FAILED, f"{path} is not readable JSON ({error})"
            ) from error

        if not isinstance(document, dict):
            raise ControllerRefused(
                OUTCOME_FAILED, f"{path} does not hold a {what}"
            )

        return document

    # -- receipt -------------------------------------------------------

    def _receipt(
        self,
        *,
        plan: dict[str, Any],
        started_at: datetime,
        outcome: str,
        reason: str,
        readiness: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        finished_at = self.clock()

        return {
            "already_completed_windows": plan["already_completed_windows"],
            "capture_ids": sorted(
                {
                    capture_id
                    for capture in self.captures
                    for capture_id in capture["capture_ids"]
                }
            ),
            "captures_attempted": len(self.captures),
            "captures_completed": len(self.captures),
            "captures": self.captures,
            "deferred_to_later_segment": self.deferred,
            "distribution_outcomes": self.distributions,
            "feed_publication_outcomes": self.feed,
            "final_outcome": outcome,
            "finish_time_utc": finished_at.isoformat(),
            "games_discovered": plan["games_discovered"],
            "morning_readiness": readiness,
            "offset_minutes": self.offset_minutes,
            "page_publication_outcomes": self.page,
            "production_sha": self.production_sha,
            "publication_mode": PUBLICATION_MODE,
            "reason": reason,
            "rescheduled_windows": self.rescheduled,
            "segment_end_utc": self.segment_end.isoformat(),
            "segment_start_utc": self.segment_start.isoformat(),
            "serving_invocations": self.serving,
            "slate_date": self.slate_date,
            "snapshot_dir": str(self.snapshot_dir),
            "start_time_utc": started_at.isoformat(),
            "t20_windows_in_segment": plan["t20_windows_in_segment"],
            "wall_seconds": round(
                (finished_at - started_at).total_seconds(), 3
            ),
            "withdrawn_windows": self.withdrawn,
        }


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def render(receipt: dict[str, Any]) -> str:
    rows: list[tuple[str, Any]] = [
        ("outcome", receipt.get("final_outcome")),
        ("slate date", receipt.get("slate_date")),
        (
            "segment",
            (
                f"{receipt.get('segment_start_utc')} .. "
                f"{receipt.get('segment_end_utc')}"
            ),
        ),
        ("games discovered", receipt.get("games_discovered")),
        (
            "T-20 windows in segment",
            ", ".join(receipt.get("t20_windows_in_segment") or []) or "none",
        ),
        (
            "already captured",
            ", ".join(receipt.get("already_completed_windows") or [])
            or "none",
        ),
        ("captures completed", receipt.get("captures_completed")),
        ("serving invocations", len(receipt.get("serving_invocations") or [])),
        (
            "feed publications",
            ", ".join(
                str(record.get("outcome"))
                for record in receipt.get("feed_publication_outcomes") or []
            )
            or "none",
        ),
        (
            "page publications",
            ", ".join(
                str(record.get("outcome"))
                for record in receipt.get("page_publication_outcomes") or []
            )
            or "none",
        ),
        ("publication mode", receipt.get("publication_mode")),
        ("wall seconds", receipt.get("wall_seconds")),
    ]

    lines = [
        "## NBA T-20 production controller",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    lines += [f"| {label} | {value} |" for label, value in rows]

    lines += ["", str(receipt.get("reason") or "")]

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_t20_production_controller",
        description=(
            "Capture the production T-20 market snapshot for one bounded "
            "scheduling segment, then serve, derive and publish what it "
            "makes available. Refreshes nothing, fits nothing, promotes "
            "nothing and publishes to the canary destination only."
        ),
    )

    parser.add_argument("--data-root", type=Path, required=True)

    parser.add_argument(
        "--production-sha",
        required=True,
        help="the production checkout this run is executing, which must be "
        "the one the morning lifecycle verified the frozen runtime under",
    )

    parser.add_argument(
        "--segment-start-utc",
        default=None,
        help="inclusive start of the segment this invocation owns",
    )

    parser.add_argument(
        "--segment-end-utc",
        default=None,
        help="exclusive end of the segment this invocation owns",
    )

    parser.add_argument(
        "--cron",
        default=None,
        help="the cron expression that fired, used to resolve the segment "
        "when it is not given explicitly. Resolved from the scheduled time "
        "rather than from the clock, so a delayed runner pickup lands on the "
        "segment that was scheduled.",
    )

    parser.add_argument(
        "--slate-date",
        default=None,
        help="defaults to the segment start's date in America/New_York, "
        "which is the slate a late-evening segment is capturing for",
    )

    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)

    parser.add_argument(
        "--offset-minutes", type=int, default=PRODUCTION_OFFSET_MINUTES
    )

    parser.add_argument("--grace-minutes", type=int, default=5)

    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="resolve the segment and report whether it holds any uncaptured "
        "T-20 window, without capturing, serving or publishing. Reads the "
        "schedule and the capture log and nothing else, so an off-day "
        "segment can answer before the production environment is installed.",
    )

    parser.add_argument("--now-utc", default=None)
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--github-output", type=Path, default=None)

    return parser.parse_args(argv)


def _write(path: Path | None, text: str, *, append: bool) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def _segment_bounds(
    args: argparse.Namespace, *, now_utc: datetime
) -> tuple[datetime, datetime]:
    if bool(args.segment_start_utc) != bool(args.segment_end_utc):
        raise ControllerRefused(
            OUTCOME_FAILED,
            "--segment-start-utc and --segment-end-utc are given together or "
            "not at all",
        )

    if args.segment_start_utc:
        return (
            parse_utc(args.segment_start_utc, what="--segment-start-utc"),
            parse_utc(args.segment_end_utc, what="--segment-end-utc"),
        )

    return resolve_segment(now_utc=now_utc, cron=args.cron)


def run(args: argparse.Namespace) -> dict[str, Any]:
    now_utc = (
        parse_utc(args.now_utc, what="--now-utc")
        if args.now_utc
        else datetime.now(UTC)
    )

    segment_start, segment_end = _segment_bounds(args, now_utc=now_utc)

    slate_date = args.slate_date or slate_date_for(segment_start)

    controller = T20Controller(
        slate_date=slate_date,
        segment_start=segment_start,
        segment_end=segment_end,
        data_root=args.data_root,
        production_sha=args.production_sha,
        project_root=args.project_root,
        registry_root=args.registry_root,
        run_dir=args.run_dir,
        offset_minutes=args.offset_minutes,
        grace_minutes=args.grace_minutes,
        summary_path=args.summary_path,
    )

    if not args.probe_only:
        return controller.run()

    plan = controller.plan()

    return {
        "already_completed_windows": plan["already_completed_windows"],
        "final_outcome": (
            OUTCOME_NO_WINDOWS
            if not plan["pending_rows"]
            else "T20_WORK_IN_SEGMENT"
        ),
        "games_discovered": plan["games_discovered"],
        "offset_minutes": controller.offset_minutes,
        "pending_windows": plan["pending_windows"],
        "probe_only": True,
        "production_sha": controller.production_sha,
        "reason": (
            f"{len(plan['pending_windows'])} uncaptured "
            f"T-{controller.offset_minutes} window(s) fall in "
            f"{segment_start.isoformat()} to {segment_end.isoformat()}"
        ),
        "segment_end_utc": segment_end.isoformat(),
        "segment_start_utc": segment_start.isoformat(),
        "slate_date": slate_date,
        "t20_windows_in_segment": plan["t20_windows_in_segment"],
    }


def _github_output(args: argparse.Namespace, receipt: dict[str, Any]) -> None:
    if args.github_output is None:
        return

    pending = receipt.get("pending_windows") or []

    _write(
        args.github_output,
        "\n".join(
            [
                f"t20_work={'true' if pending else 'false'}",
                f"segment_start_utc={receipt['segment_start_utc']}",
                f"segment_end_utc={receipt['segment_end_utc']}",
                f"slate_date={receipt['slate_date']}",
                f"pending_windows={len(pending)}",
            ]
        )
        + "\n",
        append=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except (ControllerRefused, OSError) as error:
        outcome = getattr(error, "outcome", OUTCOME_FAILED)

        failure = {
            "error": type(error).__name__,
            "final_outcome": outcome,
            "message": str(error),
            "publication_mode": PUBLICATION_MODE,
            "slate_date": args.slate_date,
        }

        _write(
            args.receipt_path,
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            append=False,
        )

        _write(
            args.summary_path,
            f"## NBA T-20 production controller\n\n{outcome}: {error}\n",
            append=True,
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_FAILED

    _write(
        args.receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        append=False,
    )

    if args.probe_only:
        _github_output(args, receipt)

    else:
        _write(args.summary_path, render(receipt), append=True)

    print(json.dumps(receipt, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
