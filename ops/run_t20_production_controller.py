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

THE EARLY ADVISORY WINDOWS
--------------------------

T-20 is twenty minutes before tip, which is when the production release is
available and is too late to be much use to a reader deciding what to bet. So
this controller also captures, serves and publishes T-180, T-90 and T-45 --
:data:`ADVISORY_OFFSETS_MINUTES` -- as *advisory* releases, three hours, an
hour and a half and three quarters of an hour before tip.

They are handled here, inside the invocation that already owns the segment,
rather than by a workflow of their own, and that is a decision about the
runner rather than about the code. The self-hosted runner executes one job at
a time; the 19:25 UTC segment holds it from 19:25 until its last window is
done, which on a normal slate is past 01:00; and almost every advisory window
falls inside that span. A second workflow would therefore not run earlier --
it would queue behind this one on the runner and start after every window it
was for had passed. Sharing this invocation is the only arrangement in which
an advisory window is served at the time it is for. It also shortens the
longest sleep this job ever takes, which matters on a runner that has twice
dropped a long-idling job: the 19:25 segment's first T-20 window is over three
hours away, and the advisory windows in between break that one sleep into
several short ones.

Three properties keep the advisory path from costing anything the T-20 path
has, and all three are enforced rather than intended:

* **T-20 comes first.** No advisory work is started inside
  :data:`T20_RESERVATION_MINUTES` of a T-20 capture this segment still owns.
  The reservation is re-checked at every stage boundary, so an advisory serve
  that takes longer than expected is abandoned rather than allowed to run
  into the window.
* **An advisory failure is not a T-20 failure.** Everything the advisory path
  does is recorded and nothing it does raises: a capture that fails, a serve
  that fails or a publication that fails is reported as
  :data:`OUTCOME_SEGMENT_COMPLETE_ADVISORY_DEGRADED`, which still succeeds,
  because a defect in an advisory window must never cost the slate its
  production release.
* **An advisory release cannot be mistaken for a T-20 one.** It is captured
  at its own window id, served through ``ops/run_advisory_window_serving.py``
  -- which refuses the T-20 offset and refuses to write a T-20 window id --
  and published to :data:`ADVISORY_FEED_DESTINATIONS`, which are confined to
  ``canary-advisory/t<offset>m`` and carry the offset in their release prefix.
  No advisory release touches the T-20 canary's ``current.json``, because no
  advisory run resolves a path inside the directory that holds it.
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

#: The window production consumes, and the only one it is required to capture.
PRODUCTION_OFFSET_MINUTES = 20

#: The early windows served for advisory reading, in the order they occur.
#:
#: A subset of :data:`CAPTURE_OFFSETS_MINUTES` and no addition to it: the
#: capture schedule has always defined these three and nothing scheduled them.
#: T-1440 and T-480 are a day and eight hours out, which is before the books
#: have posted most of a slate's props; T-5 is five minutes before tip, which
#: is later than the window a reader already gets. So these three are the ones
#: with an advisory destination configured, and the serving entry point
#: refuses any other.
#:
#: On by default, which is deliberate. GitHub fires scheduled events from the
#: default branch, so a scheduled run executes ``main``'s copy of the workflow
#: and this branch's copy of the Python. An advisory path switched on by a
#: workflow input would therefore do nothing until the scheduler copy was
#: separately synced, and the point of this change is that a reader gets the
#: early windows tonight.
ADVISORY_OFFSETS_MINUTES: tuple[int, ...] = (180, 90, 45)

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

#: The advisory windows' serving entry point. A separate script rather than an
#: argument to the one above, because the production orchestrator is held
#: byte-identical to production outside two named functions and the offset
#: would have to be added to its argument parser. The advisory one imports
#: that orchestrator and calls its functions by name, so the model, the
#: incumbent resolution, the frozen-bundle verification and the receipt shape
#: are the production ones and not a second copy of them.
ADVISORY_SERVING_ENTRY_POINT = Path("ops") / "run_advisory_window_serving.py"

#: Canary, always. There is no mode argument and no code path to live
#: publication here: a schedule that could activate live publishing would
#: make every scheduled run an activation decision.
PUBLICATION_MODE = "canary"

#: The feed destination each advisory window publishes to, by name.
#:
#: One destination per window, and the reason is the pointer rather than the
#: tidiness. A publication writes ``current.json`` at the root of the
#: destination it resolved, so the only way an advisory release can be
#: structurally incapable of repointing the T-20 canary is for it to resolve a
#: root the T-20 canary's pointer is not inside. Each of these is confined to
#: ``canary-advisory/t<offset>m`` where the T-20 canary is confined to
#: ``canary``: neither path is a prefix of the other, so neither destination
#: can address the other's pointer, releases or audit log.
#:
#: Their release prefixes carry the offset too, so ``current.json``, the
#: release directory, the release id in the manifest and the rows path the
#: manifest names all state the window the prices came from.
ADVISORY_FEED_DESTINATIONS: dict[int, str] = {
    180: "wizardofodds_nba_advisory_t180m_canary",
    90: "wizardofodds_nba_advisory_t90m_canary",
    45: "wizardofodds_nba_advisory_t45m_canary",
}

#: The page destination each advisory window's page is written to, by name.
#: Confined the same way and disjoint from the T-20 page canary for the same
#: reason.
ADVISORY_PAGE_DESTINATIONS: dict[int, str] = {
    180: "wizardofodds_nba_player_props_page_advisory_t180m_canary",
    90: "wizardofodds_nba_player_props_page_advisory_t90m_canary",
    45: "wizardofodds_nba_player_props_page_advisory_t45m_canary",
}

#: The subdirectory each advisory page is written to inside its destination.
#: The destination confinement already separates the windows from each other
#: and from T-20; this carries the offset in the page's own file path as well,
#: so a reader who has only the URL can tell which window they are reading.
ADVISORY_PAGE_PREFIXES: dict[int, str] = {
    offset: f"page-preview-advisory-t{offset}m"
    for offset in ADVISORY_OFFSETS_MINUTES
}

#: The feed destination the T-20 window publishes to, by the name
#: ``run_wizardofodds_publication.py`` resolves for canary mode. Stated here
#: so the cross-routing refusal can name it: the T-20 release is published by
#: *not* passing a destination name, and this is the name that would be
#: resolved if one were.
T20_FEED_DESTINATION = "wizardofodds_nba_canary"

T20_PAGE_DESTINATION = "wizardofodds_nba_player_props_page_canary"

#: How much room an advisory window needs before the next T-20 capture this
#: segment owns.
#:
#: Measured rather than guessed. On the production runner a complete capture,
#: serve, derive, publish-feed and publish-page cycle for a seven-game,
#: 2,075-row slate took two minutes and nine seconds (run 40033448558, the
#: 23:40 UTC group of 2026-10-23). Ten minutes is nearly five times that, and
#: it is re-checked at every stage boundary rather than only before the first,
#: so the exposure if a stage runs long is one stage rather than four.
#:
#: Ten and not more: T-45 and T-20 are twenty-five minutes apart by
#: definition, so a reservation above twenty-five would defer every T-45
#: window whose T-20 this segment also owns -- which is most of them -- and
#: the T-45 release would reliably arrive after the T-20 one it is meant to
#: precede.
T20_RESERVATION_MINUTES = 10

#: Where the morning lifecycle's own serving receipt is durably kept. Written
#: by the lifecycle's grading step, which copies the run-temporary receipt
#: into the durable data root so a slate can be graded on a later day. That
#: copy is the handoff this controller reuses; no new receipt is introduced.
MORNING_RECEIPT_RELATIVE = Path("processed") / "incumbent_serving"

#: The field only ``ops/run_advisory_window_serving.py`` writes. Stated here
#: rather than imported for the reason the readiness outcomes are stated here:
#: that module imports the serving orchestrator, which pulls the fit registry
#: and the frozen-bundle installer behind it, and the probe must stay light. A
#: test pins the two constants equal.
ADVISORY_RECEIPT_MARKER_FIELD = "advisory"

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

#: The segment finished its T-20 work and something in the advisory path did
#: not. A distinct outcome and still a success: the T-20 release is the
#: production deliverable and an advisory defect must not turn a slate that
#: was captured, served and published into a red job. Distinct so the run is
#: not silently green either -- the receipt names every advisory failure and
#: the step summary leads with the count.
OUTCOME_SEGMENT_COMPLETE_ADVISORY_DEGRADED = "T20_SEGMENT_COMPLETE_ADVISORY_DEGRADED"

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
    {
        OUTCOME_NO_WINDOWS,
        OUTCOME_SEGMENT_COMPLETE,
        OUTCOME_SEGMENT_COMPLETE_ADVISORY_DEGRADED,
    }
)

PUBLICATION_OUTCOME_PUBLISHED = "PUBLISHED"
PUBLICATION_OUTCOME_NOT_SERVED = "NOT_SERVED_NOTHING_TO_PUBLISH"
PUBLICATION_OUTCOME_SKIPPED = "SKIPPED_ALREADY_PUBLISHED_THIS_STATE"

#: What one advisory window's attempt is recorded as.
ADVISORY_OUTCOME_PUBLISHED = "ADVISORY_PUBLISHED"

#: Captured on time and not served, because serving it would have run inside
#: the reservation before a T-20 capture. The capture is durable and the
#: window is retried at the next gap, so this is a delay and not a loss.
ADVISORY_OUTCOME_HELD_FOR_T20 = "ADVISORY_HELD_FOR_T20_WINDOW"

#: Its capture time fell outside every scheduling segment, so no invocation
#: would have been awake for it. Recorded rather than refused: the equivalent
#: T-20 condition is :data:`OUTCOME_UNSUPPORTED_TIME` and fails the run,
#: which is right for the window production is required to have and wrong for
#: one it is not.
ADVISORY_OUTCOME_UNSUPPORTED_TIME = "ADVISORY_UNSUPPORTED_TIME"

#: Its capture time passed while the controller was doing T-20 work, so the
#: point-in-time snapshot it is for no longer exists to be taken.
ADVISORY_OUTCOME_MISSED = "ADVISORY_WINDOW_MISSED"

ADVISORY_OUTCOME_WITHDRAWN = "ADVISORY_GAME_WITHDRAWN"

#: The tip moved and the window moved with it, out past the end of this
#: segment. The invocation that owns the segment it moved into owns it now,
#: exactly as :attr:`T20Controller.deferred` records for the T-20 window.
#: Recorded rather than slept to, because sleeping past the segment end would
#: hold the single self-hosted runner into the next invocation's segment and
#: queue the capture that segment is required to have behind an advisory one.
ADVISORY_OUTCOME_DEFERRED = "ADVISORY_DEFERRED_TO_LATER_SEGMENT"

ADVISORY_OUTCOME_FAILED = "ADVISORY_FAILED"

#: Advisory outcomes that are not a defect: nothing went wrong, the window
#: was simply not this segment's to serve. Only an outcome outside this set
#: degrades the segment.
ADVISORY_OUTCOMES_WITHOUT_FAULT: frozenset[str] = frozenset(
    {
        ADVISORY_OUTCOME_PUBLISHED,
        ADVISORY_OUTCOME_HELD_FOR_T20,
        ADVISORY_OUTCOME_UNSUPPORTED_TIME,
        ADVISORY_OUTCOME_WITHDRAWN,
        ADVISORY_OUTCOME_DEFERRED,
    }
)

#: How many times one advisory window may be re-planned onto a later capture
#: time before it is given up on. Bounded for the same reason the T-20 plan's
#: re-planning is: a schedule that kept moving must not loop forever. In
#: practice one re-plan is the most that ever happens.
ADVISORY_REPLAN_LIMIT = 4

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


def advisory_rows(
    games: Sequence[dict[str, Any]],
    *,
    offsets: Sequence[int] = ADVISORY_OFFSETS_MINUTES,
) -> list[dict[str, Any]]:
    """One row per game per advisory offset, earliest capture time first.

    The same shared schedule :func:`t20_rows` reads, narrowed to the advisory
    offsets instead of to the production one. Sorted by capture time because
    the controller walks these in time order and a stable order makes the
    receipt readable; the secondary keys make it total.
    """
    requested = [int(offset) for offset in offsets]

    unknown = sorted(
        offset for offset in requested if offset not in ADVISORY_OFFSETS_MINUTES
    )

    if unknown:
        raise ControllerRefused(
            OUTCOME_FAILED,
            f"advisory offset(s) {unknown} are not among the advisory windows "
            f"{list(ADVISORY_OFFSETS_MINUTES)}, so no destination is "
            "configured to publish them and nothing would carry the release",
        )

    if PRODUCTION_OFFSET_MINUTES in requested:
        raise ControllerRefused(
            OUTCOME_FAILED,
            f"T-{PRODUCTION_OFFSET_MINUTES} was asked for as an advisory "
            "window. It is the production window: it has its own capture, its "
            "own serving entry point, its own receipt and its own release "
            "stream, and serving it as advisory would put the same snapshot "
            "into two release streams under two names",
        )

    try:
        rows = capture_schedule(list(games))
    except ValueError as error:
        raise ControllerRefused(
            OUTCOME_FAILED,
            "the slate schedule cannot be turned into capture windows, so "
            f"no advisory time can be computed: {error}",
        ) from error

    wanted = set(requested)

    return sorted(
        (row for row in rows if int(row["offset_minutes"]) in wanted),
        key=lambda row: (
            row["capture_at_utc"],
            int(row["offset_minutes"]),
            int(row["game_id"]),
        ),
    )


def assert_window_is_not_production(
    window_ids: Sequence[str], *, offset_minutes: int
) -> list[str]:
    """Every window an advisory step is about is a window at its own offset.

    The one assertion that makes "an advisory run cannot write a T-20 window
    id" a property rather than a consequence. It is checked before the
    collector is invoked, so a mis-wired advisory group refuses instead of
    appending a ``:T-20m`` window to the capture log, and again on the window
    ids handed to the advisory serving entry point, which refuses them in the
    same terms from its own side.

    Both halves matter. A ``:T-20m`` id would name the production record. An
    id at some other offset would mean this step is about a window it is not
    serving, and the receipt would describe a capture that did not produce
    these prices.
    """
    suffix = f":T-{int(offset_minutes)}m"

    production_suffix = f":T-{PRODUCTION_OFFSET_MINUTES}m"

    claimed = sorted(
        str(window_id)
        for window_id in window_ids
        if str(window_id).endswith(production_suffix)
    )

    if claimed:
        raise ControllerRefused(
            OUTCOME_FAILED,
            "an advisory step would write the production window(s) "
            + ", ".join(claimed)
            + f", and a T-{PRODUCTION_OFFSET_MINUTES} window id is the name "
            "of the production record. No advisory capture, receipt or "
            "release may carry one",
        )

    wrong = sorted(
        str(window_id)
        for window_id in window_ids
        if not str(window_id).endswith(suffix)
    )

    if wrong:
        raise ControllerRefused(
            OUTCOME_FAILED,
            f"a T-{offset_minutes} advisory step was handed the window(s) "
            + ", ".join(wrong)
            + f", which are not {suffix} windows, so it is not the step those "
            "captures belong to",
        )

    return sorted(str(window_id) for window_id in window_ids)


def assert_destinations_are_advisory(
    *, offset_minutes: int, feed: str, page: str
) -> None:
    """The destination pair an advisory window publishes to is its own.

    Checked because the two mistakes it prevents are both one typo away and
    neither would announce itself. A T-20 destination name here would publish
    advisory prices over the production canary's ``current.json``. The wrong
    advisory window's name would publish one window's prices under another
    window's release prefix, which is the mislabelling this whole design
    exists to make impossible.
    """
    expected_feed = ADVISORY_FEED_DESTINATIONS.get(int(offset_minutes))

    expected_page = ADVISORY_PAGE_DESTINATIONS.get(int(offset_minutes))

    for label, actual, expected in (
        ("feed", feed, expected_feed),
        ("page", page, expected_page),
    ):
        if expected is None:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"no advisory {label} destination is configured for T-"
                f"{offset_minutes}, so nothing would carry its release",
            )

        if actual != expected:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"a T-{offset_minutes} advisory release would be published to "
                f"the {label} destination {actual!r} and its own is "
                f"{expected!r}. The T-20 canary is "
                f"{T20_FEED_DESTINATION!r} and its page canary is "
                f"{T20_PAGE_DESTINATION!r}; neither is ever written by an "
                "advisory run",
            )

    for label, actual, production in (
        ("feed", feed, T20_FEED_DESTINATION),
        ("page", page, T20_PAGE_DESTINATION),
    ):
        if actual == production:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"an advisory release would be published to the production "
                f"T-{PRODUCTION_OFFSET_MINUTES} {label} destination "
                f"{production!r}, which would repoint the release a reader "
                "fetches as the production one",
            )


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

    # An advisory receipt is the same shape as this one, by design: it is
    # built by the same function, from the same incumbent, under the same
    # production SHA, and it would satisfy every check below. The only thing
    # that distinguishes it is the marker, so the marker is refused here.
    # ``ops/run_advisory_window_serving.py`` separately refuses to write into
    # this directory at all, and this is the second of the two.
    if receipt.get(ADVISORY_RECEIPT_MARKER_FIELD) is True:
        raise refuse(
            f"{path} is an advisory receipt: it carries "
            f"{ADVISORY_RECEIPT_MARKER_FIELD}, which only "
            f"{ADVISORY_SERVING_ENTRY_POINT} writes, and it records the "
            f"T-{receipt.get('advisory_capture_offset_minutes')} capture "
            "window rather than a production serving run. An advisory run "
            "proves nothing about the T-20 slate and is not a readiness "
            "receipt for one"
        )

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
        advisory_offsets: Sequence[int] = ADVISORY_OFFSETS_MINUTES,
        t20_reservation_minutes: int = T20_RESERVATION_MINUTES,
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

        # Checked here as well as in :func:`advisory_rows`, because a
        # controller built with an unserveable advisory offset must refuse
        # before it reads a schedule rather than after.
        unknown = sorted(
            int(offset)
            for offset in advisory_offsets
            if int(offset) not in ADVISORY_OFFSETS_MINUTES
        )

        if unknown:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"--advisory-offset-minutes {unknown} are not among the "
                f"advisory windows {list(ADVISORY_OFFSETS_MINUTES)}, so no "
                "destination is configured to publish them",
            )

        if offset_minutes in {int(offset) for offset in advisory_offsets}:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"T-{offset_minutes} is this run's production window and was "
                "also asked for as an advisory one, which would publish one "
                "capture into two release streams under two names",
            )

        if t20_reservation_minutes < 0:
            raise ControllerRefused(
                OUTCOME_FAILED,
                "--t20-reservation-minutes cannot be negative: it is the room "
                "an advisory window must leave before a T-20 capture, and a "
                "negative reservation would be permission to run into one",
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
        self.advisory_offsets = tuple(
            offset
            for offset in ADVISORY_OFFSETS_MINUTES
            if offset in {int(value) for value in advisory_offsets}
        )
        self.t20_reservation = timedelta(minutes=int(t20_reservation_minutes))
        self.t20_reservation_minutes = int(t20_reservation_minutes)
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

        #: One record per advisory window this segment did anything about,
        #: whether it was published, held, missed or failed.
        self.advisory: list[dict[str, Any]] = []

        self._published_priced_digest: str | None = None

        #: Publication idempotency, per destination. The T-20 digest and each
        #: advisory window's digest are tracked apart, because the same priced
        #: bytes published to two different destinations are two different
        #: releases and skipping the second would leave a destination empty.
        self._published_advisory_digests: dict[int, str] = {}

        #: Advisory windows captured and not yet served, newest capture last.
        #: Held rather than served immediately when the room before the next
        #: T-20 capture is short; drained at the next gap and, failing that,
        #: after the segment's last T-20 window.
        self._advisory_held: dict[int, list[dict[str, Any]]] = {}

        #: Advisory window ids this segment has finished with, in any sense.
        self._advisory_handled: set[str] = set()

        #: The advisory windows this segment owns, by window id. Seeded from
        #: the plan when the segment starts working and refreshed from the
        #: live schedule at every advisory capture attempt, so a tip that
        #: moved moves its advisory windows with it.
        self._advisory_pending: dict[str, dict[str, Any]] = {}

        #: One record per advisory window re-planned onto a later capture
        #: time. Kept apart from :attr:`rescheduled`, which is the T-20
        #: record, and not a terminal outcome: a re-planned window is still
        #: pending and is still this segment's to take.
        self._advisory_rescheduled: list[dict[str, Any]] = []

        #: How many times each advisory window has been re-planned.
        self._advisory_replans: dict[str, int] = {}

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

        advisory = self._plan_advisory(games, completed=completed)

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
            **advisory,
        }

    def _plan_advisory(
        self, games: Sequence[dict[str, Any]], *, completed: set[str]
    ) -> dict[str, Any]:
        """The advisory windows this segment owns, and the ones nobody does.

        Deliberately not symmetric with the T-20 plan above. A T-20 time no
        segment covers fails the run, because production is required to have
        that window; an advisory time no segment covers is recorded and the
        segment carries on, because refusing would mean an unreachable
        advisory window cost the slate its production release. That asymmetry
        is the whole of the advisory path's claim on reliability.
        """
        if not self.advisory_offsets:
            return {
                "advisory_offsets": [],
                "advisory_windows_in_segment": [],
                "advisory_pending_windows": [],
                "advisory_pending_rows": [],
                "advisory_unsupported": [],
            }

        rows = advisory_rows(games, offsets=self.advisory_offsets)

        start, end = supported_window(self.slate_date)

        unsupported = [
            {
                "capture_at_utc": row["capture_at_utc"].isoformat(),
                "game_id": int(row["game_id"]),
                "offset_minutes": int(row["offset_minutes"]),
                "outcome": ADVISORY_OUTCOME_UNSUPPORTED_TIME,
                "reason": (
                    "its capture time falls outside the supported production "
                    f"window {start.isoformat()} to {end.isoformat()}, so no "
                    "scheduling segment would have been awake for it"
                ),
                "tip_utc": row["tip_utc"].isoformat(),
                "window_id": str(row["window_id"]),
            }
            for row in rows
            if not (start <= row["capture_at_utc"] < end)
        ]

        owned = [
            row
            for row in rows
            if self.segment_start <= row["capture_at_utc"] < self.segment_end
        ]

        pending = [
            row for row in owned if str(row["window_id"]) not in completed
        ]

        return {
            "advisory_offsets": list(self.advisory_offsets),
            "advisory_windows_in_segment": [
                str(row["window_id"]) for row in owned
            ],
            "advisory_pending_windows": [
                str(row["window_id"]) for row in pending
            ],
            "advisory_pending_rows": pending,
            "advisory_unsupported": unsupported,
        }

    # -- the segment ---------------------------------------------------

    def run(self) -> dict[str, Any]:
        started_at = self.clock()

        plan = self.plan()

        self.advisory += list(plan["advisory_unsupported"])

        if not plan["pending_rows"] and not plan["advisory_pending_rows"]:
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
        #
        # Guarded only when the segment's work is advisory. With a T-20 window
        # pending, a receipt that does not prove a verified incumbent is the
        # refusal it has always been. With none, this segment would have been
        # a green no-op before the advisory windows existed, and an advisory
        # window must not be the reason an evening turns red -- so the missing
        # proof is recorded as the advisory failure it is.
        try:
            readiness = morning_readiness(
                data_root=self.data_root,
                slate_date=self.slate_date,
                production_sha=self.production_sha,
                project_root=self.project_root,
            )
        except ControllerRefused as refusal:
            if plan["pending_rows"]:
                raise

            for row in plan["advisory_pending_rows"]:
                self._record_advisory(
                    row,
                    outcome=ADVISORY_OUTCOME_FAILED,
                    reason=(
                        "the advisory windows need the same verified "
                        f"incumbent the T-{self.offset_minutes} window does, "
                        f"and {refusal.outcome}: {refusal}"
                    ),
                )

            return self._receipt(
                plan=plan,
                started_at=started_at,
                outcome=OUTCOME_SEGMENT_COMPLETE_ADVISORY_DEGRADED,
                reason=(
                    f"the segment held no T-{self.offset_minutes} window and "
                    f"{len(plan['advisory_pending_rows'])} advisory window(s) "
                    "it could not serve, because no verified incumbent is "
                    f"proven for {self.slate_date}: {refusal}"
                ),
            )

        # Seeded once, here, rather than in each of the two places that take
        # advisory windows: between them the rows are refreshed off later
        # schedule reads, and re-seeding from the plan would put the stale
        # capture times back.
        self._advisory_pending = {
            str(row["window_id"]): row for row in plan["advisory_pending_rows"]
        }

        if plan["pending_rows"]:
            self._work(plan, readiness)

        # Whatever advisory windows are left: the ones after this segment's
        # last T-20 window, the ones held back for a T-20 capture, and all of
        # them on a segment that owns no T-20 window at all.
        self._finish_advisory(readiness)

        faults = [
            record
            for record in self.advisory
            if str(record.get("outcome")) not in ADVISORY_OUTCOMES_WITHOUT_FAULT
        ]

        published = [
            record
            for record in self.advisory
            if str(record.get("outcome")) == ADVISORY_OUTCOME_PUBLISHED
        ]

        reason = (
            f"{len(self.captures)} T-{self.offset_minutes} capture(s) "
            f"completed for {self.slate_date}, each served and published "
            "to the canary destination"
        )

        if self.advisory_offsets:
            reason += (
                f"; {len(published)} advisory window(s) published to their "
                f"own confined canary destinations and {len(faults)} failed"
            )

        return self._receipt(
            plan=plan,
            started_at=started_at,
            outcome=(
                OUTCOME_SEGMENT_COMPLETE_ADVISORY_DEGRADED
                if faults
                else OUTCOME_SEGMENT_COMPLETE
            ),
            reason=reason,
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

            # The gap before the next T-20 capture is where the advisory
            # windows live. Taken before the sleep rather than instead of it:
            # what replaces polling is still sleeping, and this only decides
            # what the controller wakes up for on the way.
            self._advisory_up_to(due, readiness=readiness)

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

    # -- the advisory windows ------------------------------------------

    def _next_advisory(
        self, *, before: datetime | None
    ) -> dict[str, Any] | None:
        """The earliest advisory window left, optionally before a deadline.

        ``before`` is the next T-20 capture time. A window at exactly that
        moment is left for later rather than taken now: the T-20 window is
        the one this segment is required to have, and the tie goes to it.
        """
        outstanding = [
            row
            for row in self._advisory_pending.values()
            if str(row["window_id"]) not in self._advisory_handled
            and (before is None or row["capture_at_utc"] < before)
        ]

        if not outstanding:
            return None

        return min(
            outstanding,
            key=lambda row: (
                row["capture_at_utc"],
                int(row["offset_minutes"]),
                int(row["game_id"]),
            ),
        )

    def _advisory_up_to(
        self, deadline: datetime, *, readiness: dict[str, Any]
    ) -> None:
        """Every advisory window due before the next T-20 capture."""
        while True:
            row = self._next_advisory(before=deadline)

            if row is None:
                self._drain_advisory(readiness=readiness, next_t20_due=deadline)

                return

            self._sleep_until(row["capture_at_utc"])

            self._capture_advisory_window(row)

            self._drain_advisory(readiness=readiness, next_t20_due=deadline)

    def _finish_advisory(self, readiness: dict[str, Any]) -> None:
        """The advisory windows no T-20 window is in front of any more.

        Everything after this segment's last T-20 capture, everything that was
        held back for one, and -- on a segment that owns no T-20 window at all
        -- all of them. No deadline, because there is no T-20 capture left in
        this segment to reserve room for.

        Deliberately does not re-seed the pending windows from the plan: by
        the time this runs they may have been refreshed off a schedule read
        the plan predates, and the planned times are the stale ones.
        """
        while True:
            row = self._next_advisory(before=None)

            if row is None:
                break

            self._sleep_until(row["capture_at_utc"])

            self._capture_advisory_window(row)

            self._drain_advisory(readiness=readiness, next_t20_due=None)

        self._drain_advisory(readiness=readiness, next_t20_due=None)

        # Anything still held after the last drain was never served, and the
        # segment has no gap left to serve it in. Recorded once, here, rather
        # than on every drain that passed it over.
        for offset in self.advisory_offsets:
            self._hold_advisory(
                self._advisory_held.get(offset) or [],
                offset=offset,
                reason=(
                    f"the segment ended with the T-{offset} capture taken and "
                    "unserved, so the window's release was not made. The "
                    "capture is durable and the window is served by the next "
                    "invocation that owns a gap"
                ),
            )

    def _room_for_advisory(self, next_t20_due: datetime | None) -> bool:
        """Whether there is room to serve an advisory window right now.

        The one rule that makes the advisory path unable to cost the T-20
        window: no advisory serving starts inside the reservation before a
        T-20 capture this segment still owns. Asked again at every stage
        boundary, not only before the first, so a stage that runs long costs
        one stage of exposure rather than the whole sequence.
        """
        if next_t20_due is None or not self.t20_reservation:
            return True

        return next_t20_due - self.clock() >= self.t20_reservation

    def _record_advisory(
        self,
        row: dict[str, Any],
        *,
        outcome: str,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._advisory_handled.add(str(row["window_id"]))

        self.advisory.append(
            {
                "capture_at_utc": (
                    row["capture_at_utc"].isoformat()
                    if hasattr(row["capture_at_utc"], "isoformat")
                    else str(row["capture_at_utc"])
                ),
                "game_id": int(row["game_id"]),
                "observed_at_utc": self.clock().isoformat(),
                "offset_minutes": int(row["offset_minutes"]),
                "outcome": outcome,
                "reason": reason,
                "window_id": str(row["window_id"]),
                **(extra or {}),
            }
        )

    def _revalidate_advisory(
        self, row: dict[str, Any]
    ) -> dict[str, Any] | None:
        """One schedule read per advisory capture attempt, just before it.

        The same discipline the T-20 loop keeps, for the same reason: the
        controller has been asleep, and the tip it is about to capture three
        hours before may not be at the hour it was planned against. The read
        refreshes every advisory window still pending, not only this one,
        because they all come off the same schedule -- otherwise the next
        window would be woken for at its own stale time and the one that
        actually moved would be slept past.

        Returns the window to capture now, or ``None`` when it was recorded
        here instead: withdrawn, deferred into a later segment, re-planned
        onto a later time in this one, or missed outright. Never raises; a
        moved tip must not be able to cost the slate its T-20 release.
        """
        window_id = str(row["window_id"])

        fresh = {
            str(candidate["window_id"]): candidate
            for candidate in advisory_rows(
                self.schedule_source(self.slate_date),
                offsets=self.advisory_offsets,
            )
        }

        for pending_id, pending in list(self._advisory_pending.items()):
            if pending_id in self._advisory_handled:
                continue

            refreshed = fresh.get(pending_id)

            if refreshed is not None:
                self._advisory_pending[pending_id] = refreshed

        now = self.clock()

        current = fresh.get(window_id)

        if current is None:
            self._record_advisory(
                row,
                outcome=ADVISORY_OUTCOME_WITHDRAWN,
                reason=(
                    "the game is no longer on the slate schedule, so it has "
                    "no advisory window to capture"
                ),
            )

            return None

        target = current["capture_at_utc"]

        if target >= self.segment_end:
            self._record_advisory(
                current,
                outcome=ADVISORY_OUTCOME_DEFERRED,
                reason=(
                    f"the tip moved and this window is now due at "
                    f"{target.isoformat()}, in a later scheduling segment, "
                    "which owns it"
                ),
            )

            return None

        if now >= target + self.grace:
            self._record_advisory(
                current,
                outcome=ADVISORY_OUTCOME_MISSED,
                reason=(
                    f"it was due at {target.isoformat()} and it is now "
                    f"{now.isoformat()}, past the {self.grace_minutes}-minute "
                    "capture grace, so the point-in-time snapshot it is for "
                    "can no longer be taken"
                ),
            )

            return None

        if now < target:
            replans = self._advisory_replans.get(window_id, 0) + 1

            self._advisory_replans[window_id] = replans

            if replans > ADVISORY_REPLAN_LIMIT:
                self._record_advisory(
                    current,
                    outcome=ADVISORY_OUTCOME_FAILED,
                    reason=(
                        "the slate schedule kept moving and this window was "
                        f"re-planned {replans - 1} times without settling, "
                        "so no capture was attempted for it"
                    ),
                )

                return None

            self._advisory_rescheduled.append(
                {
                    "moved_to_utc": target.isoformat(),
                    "observed_at_utc": now.isoformat(),
                    "was_due_utc": row["capture_at_utc"].isoformat(),
                    "window_id": window_id,
                }
            )

            return None

        return current

    def _capture_advisory_window(self, row: dict[str, Any]) -> None:
        """Take one advisory capture, and never raise doing it.

        The capture is the time-critical half of an advisory window and the
        cheap half: one collector invocation, narrowed to this window's offset.
        Serving and publishing it is not time-critical at all, because the
        capture *is* the point-in-time market -- a T-180 capture served an hour
        later still prices the market as it stood three hours before tip -- so
        the two are separated and only this half is held to the clock.
        """
        current = self._revalidate_advisory(row)

        if current is None:
            return

        window_id = str(current["window_id"])

        offset = int(current["offset_minutes"])

        now = self.clock()

        try:
            assert_window_is_not_production([window_id], offset_minutes=offset)
        except ControllerRefused as refusal:
            self._record_advisory(
                current,
                outcome=ADVISORY_OUTCOME_FAILED,
                reason=str(refusal),
            )

            return

        capture = self.run_command(
            [
                sys.executable,
                str(COLLECTOR_SCRIPT),
                "--date",
                self.slate_date,
                "--scheduled",
                "--offset-minutes",
                str(offset),
                "--now-utc",
                now.isoformat(),
                "--grace-minutes",
                str(self.grace_minutes),
            ],
            environment=self._environment(),
            cwd=self.project_root,
        )

        if capture.returncode != 0:
            self._record_advisory(
                current,
                outcome=ADVISORY_OUTCOME_FAILED,
                reason=(
                    f"{COLLECTOR_SCRIPT} exited {capture.returncode}: "
                    + ((capture.stderr or capture.stdout) or "")[-1000:]
                ),
            )

            return

        bundle = self._advisory_bundle(current)

        if bundle is None:
            return

        self._advisory_held.setdefault(offset, []).append(
            {**current, "capture": bundle}
        )

    def _advisory_bundle(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Prove the advisory capture the pricing path will select is there.

        The same selection pricing performs, performed at this window's own
        offset. A window that reaches this point without an eligible bundle
        is recorded as missed rather than refused, because the alternative --
        raising -- would make a game whose tip moved while the controller was
        asleep cost the slate its T-20 release.
        """
        from nba_prop_quant.prospective_snapshot import select_capture_bundle

        try:
            bundle = select_capture_bundle(
                self.snapshot_dir,
                date=self.slate_date,
                game_id=int(row["game_id"]),
                offset_minutes=int(row["offset_minutes"]),
            )
        except RuntimeError as error:
            self._record_advisory(
                row,
                outcome=ADVISORY_OUTCOME_MISSED,
                reason=(
                    "no eligible scheduled capture can be selected for it, "
                    "so there is nothing at this window to price: "
                    f"{error}"
                ),
            )

            return None

        return {
            "capture_id": bundle.capture_id,
            "captured_at_utc": bundle.captured_at,
            "game_id": int(row["game_id"]),
            "window_id": bundle.window_id,
        }

    def _drain_advisory(
        self, *, readiness: dict[str, Any], next_t20_due: datetime | None
    ) -> None:
        """Serve and publish every advisory window with room to be served.

        Coalesced per offset, for the reason the T-20 path re-serves
        cumulatively: ``10_predict_slate.py`` enumerates every game with an
        eligible bundle at the offset it was given, so one invocation at T-180
        prices every game whose T-180 capture exists. Four captures taken over
        an hour are therefore one serve and one release rather than four, and
        the release published is the whole of what is available.

        Largest offset first, because that is the order the windows occur in
        and the earliest window is the one a reader has been waiting longest
        for.
        """
        for offset in self.advisory_offsets:
            held = self._advisory_held.get(offset) or []

            if not held:
                continue

            # Left held rather than recorded, so the window is tried again at
            # the next gap. Anything still held when the segment runs out of
            # gaps is recorded then, once, by :meth:`_finish_advisory`.
            if not self._room_for_advisory(next_t20_due):
                return

            self._serve_and_publish_advisory(
                offset, held, readiness=readiness, next_t20_due=next_t20_due
            )

    def _hold_advisory(
        self, rows: Sequence[dict[str, Any]], *, offset: int, reason: str
    ) -> None:
        for row in rows:
            self._record_advisory(
                row, outcome=ADVISORY_OUTCOME_HELD_FOR_T20, reason=reason
            )

        self._advisory_held[offset] = []

    def _serve_and_publish_advisory(
        self,
        offset: int,
        held: Sequence[dict[str, Any]],
        *,
        readiness: dict[str, Any],
        next_t20_due: datetime | None,
    ) -> None:
        """One advisory release: serve the window, derive, publish, record.

        Every stage boundary re-asks whether there is still room before the
        next T-20 capture, so the T-20 window's protection is not a single
        decision taken minutes earlier. Nothing here raises: an advisory
        release that cannot be made is a recorded advisory failure, never a
        failure of the segment.
        """
        rows = list(held)

        windows = sorted(
            window_id
            for window_id in completed_window_ids(self.capture_run_path)
            if window_id.endswith(f":T-{offset}m")
        )

        index = (
            sum(
                1
                for record in self.advisory
                if record.get("offset_minutes") == offset
                and record.get("outcome") == ADVISORY_OUTCOME_PUBLISHED
            )
            + 1
        )

        stage_dir = self.run_dir / f"advisory-t{offset}m-{index:02d}"

        feed_destination = ADVISORY_FEED_DESTINATIONS.get(offset, "")

        page_destination = ADVISORY_PAGE_DESTINATIONS.get(offset, "")

        try:
            assert_window_is_not_production(windows, offset_minutes=offset)

            assert_destinations_are_advisory(
                offset_minutes=offset,
                feed=feed_destination,
                page=page_destination,
            )

            stage_dir.mkdir(parents=True, exist_ok=True)
        except (ControllerRefused, OSError) as error:
            self._fail_advisory(rows, offset=offset, reason=str(error))

            return

        serving_receipt = stage_dir / "advisory_serving.json"

        served = self._advisory_serve(
            offset,
            rows,
            windows=windows,
            readiness=readiness,
            receipt_path=serving_receipt,
        )

        if served is None:
            return

        if not self._room_for_advisory(next_t20_due):
            self._hold_advisory(
                rows,
                offset=offset,
                reason=(
                    f"the T-{offset} window was served and the reservation "
                    f"before the T-{self.offset_minutes} capture due at "
                    f"{next_t20_due.isoformat()} closed before it could be "
                    "published, so publication was held back"
                ),
            )

            return

        digest = str(
            (served.get("priced_market_artifact") or {}).get("sha256") or ""
        )

        if digest and digest == self._published_advisory_digests.get(offset):
            self._publish_advisory_record(
                rows,
                offset=offset,
                outcome=ADVISORY_OUTCOME_PUBLISHED,
                reason=(
                    "this invocation already published the identical priced "
                    f"artifact to the T-{offset} advisory destination, so the "
                    "release it would make is the release already serving"
                ),
                extra={
                    "feed_destination": feed_destination,
                    "page_destination": page_destination,
                    "priced_market_artifact_sha256": digest,
                    "publication_outcome": PUBLICATION_OUTCOME_SKIPPED,
                    "release_id": "",
                },
            )

            self._advisory_held[offset] = []

            return

        feed_receipt = stage_dir / "advisory_wizardofodds_feed.json"

        feed = self._advisory_stage(
            f"{PUBLICATION_ENTRY_POINT} --stage feed (T-{offset} advisory)",
            [
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
                "--feed-destination-name",
                feed_destination,
                "--receipt-path",
                str(feed_receipt),
            ],
            rows,
            offset=offset,
        )

        if feed is None:
            return

        release_id = str((feed.get("release") or {}).get("release_id") or "")

        if str(feed.get("outcome") or "") != PUBLICATION_OUTCOME_PUBLISHED:
            self._fail_advisory(
                rows,
                offset=offset,
                reason=(
                    f"{PUBLICATION_ENTRY_POINT} --stage feed recorded "
                    f"{feed.get('outcome') or 'nothing'} for the T-{offset} "
                    "advisory window, so no page is built from it"
                ),
            )

            return

        if not self._room_for_advisory(next_t20_due):
            self._hold_advisory(
                rows,
                offset=offset,
                reason=(
                    f"the T-{offset} advisory feed was published as "
                    f"{release_id} and the reservation before the T-"
                    f"{self.offset_minutes} capture due at "
                    f"{next_t20_due.isoformat()} closed before the page could "
                    "be built, so the page was held back"
                ),
            )

            return

        page_receipt = stage_dir / "advisory_wizardofodds_page.json"

        page = self._advisory_stage(
            f"{PUBLICATION_ENTRY_POINT} --stage page (T-{offset} advisory)",
            [
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
                "--page-source-destination-name",
                feed_destination,
                "--page-destination-name",
                page_destination,
                "--page-prefix",
                ADVISORY_PAGE_PREFIXES[offset],
                "--receipt-path",
                str(page_receipt),
            ],
            rows,
            offset=offset,
        )

        if page is None:
            return

        self._published_advisory_digests[offset] = digest or ""

        self._publish_advisory_record(
            rows,
            offset=offset,
            outcome=ADVISORY_OUTCOME_PUBLISHED,
            reason=(
                f"the T-{offset} capture of {len(windows)} window(s) was "
                f"served and published as {release_id} to {feed_destination}, "
                f"and its page to {page_destination}"
            ),
            extra={
                "feed_destination": feed_destination,
                "page_destination": page_destination,
                "page_sha256": str(page.get("page_sha256") or ""),
                "priced_market_artifact_sha256": digest,
                "publication_outcome": PUBLICATION_OUTCOME_PUBLISHED,
                "release_id": release_id,
                "serving_receipt_path": str(serving_receipt),
                "windows_priced": windows,
            },
        )

        self._advisory_held[offset] = []

    def _advisory_serve(
        self,
        offset: int,
        rows: Sequence[dict[str, Any]],
        *,
        windows: Sequence[str],
        readiness: dict[str, Any],
        receipt_path: Path,
    ) -> dict[str, Any] | None:
        command = [
            sys.executable,
            str(ADVISORY_SERVING_ENTRY_POINT),
            "--slate-date",
            self.slate_date,
            "--data-root",
            str(self.data_root),
            "--snapshot-offset-minutes",
            str(offset),
            "--production-sha",
            self.production_sha,
            "--model-dir",
            readiness["serving_model_dir"],
            "--frozen-bundle-root",
            readiness["serving_bundle_root"],
            "--receipt-path",
            str(receipt_path),
        ]

        for window_id in windows:
            command += ["--capture-window-id", window_id]

        if self.registry_root is not None:
            command += ["--registry-root", str(self.registry_root)]

        served = self._advisory_stage(
            f"{ADVISORY_SERVING_ENTRY_POINT} (T-{offset})",
            command,
            rows,
            offset=offset,
        )

        if served is None:
            return None

        # The receipt has to agree that it is the advisory receipt for this
        # window before anything is published from it. Checked here as well as
        # inside the serving entry point, because this is the process that
        # chose the destination and the two facts have to match.
        problems = [
            problem
            for problem in (
                None
                if served.get(ADVISORY_RECEIPT_MARKER_FIELD) is True
                else (
                    f"it does not carry {ADVISORY_RECEIPT_MARKER_FIELD}, so "
                    "it is not an advisory receipt"
                ),
                None
                if int(served.get("advisory_capture_offset_minutes") or 0)
                == offset
                else (
                    "it records the capture offset "
                    f"{served.get('advisory_capture_offset_minutes')} and "
                    f"this is the T-{offset} window"
                ),
                None
                if str(served.get("outcome") or "") == SERVING_OUTCOME_SERVED
                else (
                    f"it records the outcome {served.get('outcome')!r} rather "
                    f"than {SERVING_OUTCOME_SERVED}"
                ),
            )
            if problem is not None
        ]

        if problems:
            self._fail_advisory(
                rows,
                offset=offset,
                reason=(
                    f"the T-{offset} advisory serving receipt at "
                    f"{receipt_path} cannot be published from: "
                    + "; ".join(problems)
                ),
            )

            return None

        return served

    def _advisory_stage(
        self,
        label: str,
        command: Sequence[str],
        rows: Sequence[dict[str, Any]],
        *,
        offset: int,
    ) -> dict[str, Any] | None:
        """One advisory subprocess and its receipt, or a recorded failure.

        The advisory twin of :meth:`_stage`, and the difference is the whole
        reason it exists separately: that one raises and this one records.
        """
        result = self.run_command(
            list(command),
            environment=self._environment(),
            cwd=self.project_root,
        )

        if result.returncode != 0:
            self._fail_advisory(
                rows,
                offset=offset,
                reason=(
                    f"{label} exited {result.returncode}: "
                    + ((result.stderr or result.stdout) or "")[-1500:]
                ),
            )

            return None

        path = Path(str(command[command.index("--receipt-path") + 1]))

        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self._fail_advisory(
                rows,
                offset=offset,
                reason=f"{label} wrote no readable receipt at {path} ({error})",
            )

            return None

        if not isinstance(receipt, dict):
            self._fail_advisory(
                rows,
                offset=offset,
                reason=f"{label} wrote no receipt object at {path}",
            )

            return None

        return receipt

    def _fail_advisory(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        offset: int,
        reason: Any,
    ) -> None:
        for row in rows:
            self._record_advisory(
                row, outcome=ADVISORY_OUTCOME_FAILED, reason=str(reason)
            )

        self._advisory_held[offset] = []

    def _publish_advisory_record(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        offset: int,
        outcome: str,
        reason: str,
        extra: dict[str, Any],
    ) -> None:
        for row in rows:
            self._record_advisory(
                row, outcome=outcome, reason=reason, extra=extra
            )

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

        advisory_published = [
            record
            for record in self.advisory
            if str(record.get("outcome")) == ADVISORY_OUTCOME_PUBLISHED
        ]

        return {
            "advisory_offsets": list(self.advisory_offsets),
            "advisory_pending_windows": plan["advisory_pending_windows"],
            "advisory_publications_completed": len(
                {
                    (
                        record.get("offset_minutes"),
                        record.get("release_id"),
                    )
                    for record in advisory_published
                }
            ),
            "advisory_releases": sorted(
                {
                    str(record.get("release_id") or "")
                    for record in advisory_published
                    if record.get("release_id")
                }
            ),
            "advisory_rescheduled_windows": self._advisory_rescheduled,
            "advisory_windows": self.advisory,
            "advisory_windows_in_segment": plan["advisory_windows_in_segment"],
            "advisory_windows_with_faults": [
                record
                for record in self.advisory
                if str(record.get("outcome"))
                not in ADVISORY_OUTCOMES_WITHOUT_FAULT
            ],
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
            "t20_feed_destination": T20_FEED_DESTINATION,
            "t20_reservation_minutes": self.t20_reservation_minutes,
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
        (
            "advisory windows served",
            ", ".join(
                f"T-{offset}" for offset in receipt.get("advisory_offsets") or []
            )
            or "none",
        ),
        (
            "advisory releases",
            ", ".join(receipt.get("advisory_releases") or []) or "none",
        ),
        (
            "advisory failures",
            len(receipt.get("advisory_windows_with_faults") or []),
        ),
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

    advisory = receipt.get("advisory_windows") or []

    if advisory:
        lines += [
            "",
            "### Early advisory windows",
            "",
            (
                "Advisory releases are published to their own confined "
                "canary destinations and never to the T-20 canary. An "
                "advisory failure degrades this report and does not fail "
                "the segment."
            ),
            "",
            "| window | outcome | release | why |",
            "| --- | --- | --- | --- |",
        ]

        lines += [
            "| {window} | {outcome} | {release} | {reason} |".format(
                window=record.get("window_id"),
                outcome=record.get("outcome"),
                release=record.get("release_id") or "-",
                reason=str(record.get("reason") or "").replace("|", "/")[:300],
            )
            for record in advisory
        ]

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

    parser.add_argument(
        "--advisory-offset-minutes",
        type=int,
        action="append",
        default=None,
        dest="advisory_offsets",
        help="an early capture window to serve for advisory reading, "
        "repeatable. Defaults to "
        + ", ".join(f"T-{offset}" for offset in ADVISORY_OFFSETS_MINUTES)
        + ". Advisory releases go to their own confined canary destinations "
        "and never to the T-20 canary, and an advisory failure degrades the "
        "report rather than failing the segment.",
    )

    parser.add_argument(
        "--no-advisory-windows",
        action="store_true",
        help="capture and serve the T-20 window only, as this controller did "
        "before the advisory windows existed. The escape hatch for an "
        "evening where the runner is degraded and the production release is "
        "the only thing that matters.",
    )

    parser.add_argument(
        "--t20-reservation-minutes",
        type=int,
        default=T20_RESERVATION_MINUTES,
        help="how much room an advisory window must leave before the next "
        "T-20 capture this segment owns. Re-checked at every advisory stage "
        "boundary; an advisory window with less room is captured on time and "
        "served at the next gap.",
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
        advisory_offsets=(
            ()
            if args.no_advisory_windows
            else (args.advisory_offsets or ADVISORY_OFFSETS_MINUTES)
        ),
        t20_reservation_minutes=args.t20_reservation_minutes,
        grace_minutes=args.grace_minutes,
        summary_path=args.summary_path,
    )

    if not args.probe_only:
        return controller.run()

    plan = controller.plan()

    # The probe gates the expensive install, so "is there work" has to include
    # the advisory windows: a segment whose only work is a T-180 capture must
    # still install the package. Reported under the field name the workflow
    # already reads, because a scheduled run executes the default branch's
    # copy of the workflow and renaming the output would silently stop gating.
    pending = list(plan["pending_windows"]) + list(
        plan["advisory_pending_windows"]
    )

    return {
        "advisory_offsets": list(controller.advisory_offsets),
        "advisory_pending_windows": plan["advisory_pending_windows"],
        "advisory_windows_in_segment": plan["advisory_windows_in_segment"],
        "already_completed_windows": plan["already_completed_windows"],
        "final_outcome": (
            OUTCOME_NO_WINDOWS if not pending else "T20_WORK_IN_SEGMENT"
        ),
        "games_discovered": plan["games_discovered"],
        "offset_minutes": controller.offset_minutes,
        "pending_windows": plan["pending_windows"],
        "probe_only": True,
        "production_sha": controller.production_sha,
        "reason": (
            f"{len(plan['pending_windows'])} uncaptured "
            f"T-{controller.offset_minutes} window(s) and "
            f"{len(plan['advisory_pending_windows'])} uncaptured advisory "
            f"window(s) fall in {segment_start.isoformat()} to "
            f"{segment_end.isoformat()}"
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

    advisory = receipt.get("advisory_pending_windows") or []

    _write(
        args.github_output,
        "\n".join(
            [
                # The gate the workflow already reads. True for advisory work
                # as well as T-20 work, because the step it gates is the
                # package install and both need it.
                f"t20_work={'true' if pending or advisory else 'false'}",
                f"segment_start_utc={receipt['segment_start_utc']}",
                f"segment_end_utc={receipt['segment_end_utc']}",
                f"slate_date={receipt['slate_date']}",
                f"pending_windows={len(pending)}",
                f"advisory_pending_windows={len(advisory)}",
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
