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
  production release. The one advisory outcome that does reach the exit code
  is :data:`ADVISORY_OUTCOME_POST_TIP`, and even that is still not raised from
  the advisory path: it is recorded there and escalated by :meth:`run` once
  every capture is behind it.
* **An advisory release cannot be mistaken for a T-20 one.** It is captured
  at its own window id, served through ``ops/run_advisory_window_serving.py``
  -- which refuses the T-20 offset and refuses to write a T-20 window id --
  and published to :data:`ADVISORY_FEED_DESTINATIONS`, which are confined to
  ``canary-advisory/t<offset>m`` and carry the offset in their release prefix.
  No advisory release touches the T-20 canary's ``current.json``, because no
  advisory run resolves a path inside the directory that holds it.

NOTHING IS PUBLISHED AFTER TIP
------------------------------

``select_capture_bundle`` refuses a capture *taken* after its game tipped.
Nothing refused a release *published* after its game tipped, and the two are
not the same question: the first is about when a snapshot may exist, the
second about when a price may be shown. Between them sits every way a sound
pre-tip capture can be reached late -- a runner picked up hours after the
window, a recovery of a capture an earlier run never published, an advisory
window held back for a T-20 reservation and retried in a gap that has moved
past the game.

On 2026-10-23 the T-20 group holding game 21717941 was priced and published
two hours and forty minutes after that game tipped, and every stage reported
success. :func:`assert_no_window_has_tipped` is the refusal that was missing.
It sits at :meth:`T20Controller._serve_and_publish`, which every production
release goes through, and the windows it is about are the ones the caller
declares it is publishing for -- the group just captured or the set recovery
verified -- never every window in the release. That scoping is what makes it
safe on a staggered slate: a release is cumulative, so the last T-20 group of
a twelve-game evening is published two hours after the first game tipped, and
that first game's rows belong in it because they were priced and published
before its own tip.

A refusal is taken before the serving entry point runs. Nothing is priced,
no release is minted, no pointer is repointed and no ledger row is appended,
so the release a reader already has is the release a reader keeps. The
refusal never retracts anything; it only declines to add.

THE LIVE RELEASE, A COMPOSITE REFRESHED PER GAME
------------------------------------------------

Everything above publishes a canary: a real release, read back and attested,
in a directory a reader does not fetch. What a reader fetches is the single
live destination, and this controller now refreshes it once per window.

Each game of the slate stands at its own latest validated window. A game gets
no card until one of its windows has passed canary, serves T-180 once T-180
has, and is replaced by T-90, then T-45, then T-20 as each of its own windows
validates. The live release is the composite of all of them at once, so an
evening with several tip times can hold one game at T-20 while another is
still at T-180. One pointer and one URL, as before; what changed is that the
window a reader is shown is now a fact about each game rather than about the
whole slate.

The refresh happens immediately after a window's canary feed *and* page have
both succeeded, and that order is the authorisation rather than a preference.
``ops/run_progressive_live_refresh.py`` publishes the games of that canary at
their own window and copies every other game's rows byte for byte out of the
attested release it was already serving from, so a window with no canary has
nothing that could authorise it and no row reaches the live root that no
canary validated. Everything else about a live publication -- the switch, the
destination confinement, the receipt verification, the row rules, the
rollback target, the readback, the atomic pointer -- belongs to the frozen
publisher, which that module calls by name.

Three properties mirror the advisory ones, and the third is new:

* **A live failure is not a T-20 failure.** Nothing on this path raises. A
  refresh that cannot be made is reported as
  :data:`OUTCOME_SEGMENT_COMPLETE_LIVE_DEGRADED`, which still succeeds, and
  that is load-bearing rather than lenient: until both approval variables are
  set in the ``wizardofodds-production`` environment *every* refresh refuses,
  and a slate that was captured, served and published must not be a red job
  because of a decision nobody has carried to the runner yet. The refusal is
  recorded verbatim, so the receipt and the step summary name both variables.
* **The prior valid release survives.** The refresh promotes last, after every
  gate, so a missing capture, a failed serve, a canary that did not validate,
  a digest that does not match, a row that traces to no attested source, a
  game that has already tipped or a transport that will not talk all leave the
  live pointer naming the most recent composite that did validate, unaltered.
* **No game moves away from its tip.** The offsets descend, so a game may be
  replaced by a window closer to its own tip and never by one further from it.
  That is reachable on this runner: pickup has been delayed by hours, so a
  T-180 invocation can finish after a T-45 one, and without the rule it would
  walk that game back to the market as it stood three hours before tip -- with
  every digest agreeing and every readback passing, because what it published
  is a real validated release. It is simply the wrong one. The rule is applied
  per game, which is what makes it safe to apply at all: measured against one
  slate-wide pointer, the first game to reach T-20 would forbid every later
  game's T-180.

This controller still has no mode option and still publishes canary on its
own account. What it gained is one invocation of one entry point that has no
mode argument either, after a canary of the identical bytes, and which refuses
unless the committed switch and both environment variables agree.
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

#: The entry point that replaces the live release with the window that has
#: just passed canary. Invoked as a subprocess like every other stage, so the
#: publisher, the attestation match, the ordering rule and the durable record
#: are the ones that module owns and not a second copy here.
LIVE_REFRESH_ENTRY_POINT = Path("ops") / "run_progressive_live_refresh.py"

#: The entry point that recomputes the two serving-window validation checks
#: against this slate's own capture. Invoked as a subprocess like every other
#: stage, so the checks are the ones adaptive_validation owns rather than a
#: second copy here that could answer differently.
WINDOW_GATE_ENTRY_POINT = Path("ops") / "assert_serving_window_gates.py"

#: The windows that refresh the live release, in the order they occur: the
#: three advisory offsets and then the production one. Composed from the two
#: tuples above rather than listed again, so a window this controller serves
#: and a window that can replace the live release cannot come apart.
#:
#: Restated here rather than imported from the refresh module, for the reason
#: :data:`ADVISORY_RECEIPT_MARKER_FIELD` is restated -- that module imports
#: the publisher, which pulls pandas and the transports behind it, and the
#: probe must stay light. A test pins the two copies equal.
PROGRESSIVE_LIVE_OFFSETS_MINUTES: tuple[int, ...] = (
    *ADVISORY_OFFSETS_MINUTES,
    PRODUCTION_OFFSET_MINUTES,
)

#: live mode's own word, held as a constant so the only spelling of it here is
#: ``"--mode", LIVE_PUBLICATION_MODE`` -- two arguments and never the one
#: string a workflow is forbidden to contain. The distinction is not cosmetic:
#: what is forbidden is a *schedule* choosing live, and this controller
#: chooses it only after a canary of the identical bytes has validated.
LIVE_PUBLICATION_MODE = "live"

#: The single live feed and page destinations, replaced once per window. One
#: of each, which is the whole design: a reader opens one URL and what changes
#: through the evening is which release is current there.
LIVE_FEED_DESTINATION = "wizardofodds_nba_live"

LIVE_PAGE_DESTINATION = "wizardofodds_nba_player_props_page_live"

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
#:
#: Ten is also what keeps the capture claim out of T-20's way. The claim an
#: advisory sweep takes beside ``capture_runs/<slate>.jsonl`` is exclusive and
#: held across the whole sweep, so an advisory sweep that overran would block
#: the T-20 sweep behind it until ``--grace-minutes`` expired. The reservation
#: is what bounds that: an advisory sweep cannot start inside the last ten
#: minutes before a T-20 capture, so a sweep costing 2m09s clears the claim
#: with roughly eight minutes to spare. The margin is pinned by
#: :data:`ADVISORY_SWEEP_MEASURED_SECONDS` below rather than left as prose.
T20_RESERVATION_MINUTES = 10

#: The measured end-to-end cost of one advisory window, in seconds, from the
#: run named above. Recorded as a number so the claim-headroom argument is a
#: test rather than a comment: the reservation has to stay a large multiple of
#: this, because it is the only thing standing between a long advisory sweep
#: and a T-20 sweep waiting on the capture claim.
ADVISORY_SWEEP_MEASURED_SECONDS = 129

#: Where this controller durably records which T-20 windows a published
#: release actually carried.
#:
#: Completion was recorded on capture and nowhere else. ``completed_window_ids``
#: reads the collector's append-only capture-run log, and both the collector
#: and :meth:`T20Controller.plan` treat a window in it as done -- which is
#: right for *capture* and wrong for the question a run has to answer, because
#: the capture is the input to serving rather than the result of it. A run that
#: captured a window and then died before publishing left that window looking
#: identical to one that had been served, so the next run saw it in
#: ``already_completed_windows``, had nothing pending, and exited a green no-op.
#:
#: That is how 2026-10-23 ended. The 01:40Z window for the three games tipping
#: at 02:00 UTC was captured, serving raised on the duplicate capture above,
#: and the scheduled run twenty seconds later reported success with nine of
#: twelve games published and no statement anywhere about the other three.
#:
#: So publication is recorded too, separately and durably: one row per
#: successful publication, beside the morning receipt in the same durable data
#: root, append-only for the same reason the capture log is. The capture log is
#: untouched and remains the capture authority -- nothing here may retract,
#: re-time or relabel a snapshot, and a window with no capture is not made
#: recoverable by anything in this file.
SERVED_WINDOWS_RELATIVE = Path("processed") / "t20_served_windows"

#: The row type in that ledger, so a reader can tell its rows from anything a
#: later writer adds to the same tree, exactly as ``snapshot_type`` does for
#: the capture log.
SERVED_RECORD_TYPE = "t20_windows_served"

#: Where every serving receipt this controller obtains is durably kept, one
#: append-only file per slate.
#:
#: Serving rewrites ``processed/priced_markets/<slate>.parquet`` in place, so
#: the last serve of a slate owns the bytes that survive the night. The
#: morning lifecycle's receipt is copied into ``processed/incumbent_serving``
#: and provenances the morning's bytes; every serve after it -- each advisory
#: window, each T-20 segment -- replaced those bytes and kept its receipt only
#: in ``RUNNER_TEMP``, which the job deletes. The next morning the grader
#: compared the surviving parquet against the morning receipt, found the
#: digests disagreed and refused, which is how run 40082999575 ended: the
#: 2026-10-23 slate was served, published and then ungradable.
#:
#: The morning receipt is not touched, because the readiness proof the
#: controller itself reads is that exact file and a later write there would
#: re-date it. The provenance for the later bytes goes here instead: whole
#: receipts, appended in the order they were produced, fsynced before the
#: release they describe is published, never rewritten.
SERVING_PROVENANCE_RELATIVE = Path("processed") / "incumbent_serving_provenance"

#: The row type in that ledger, for the reason :data:`SERVED_RECORD_TYPE`
#: exists. A reader that does not recognise it must skip the row rather than
#: guess at it.
SERVING_PROVENANCE_RECORD_TYPE = "incumbent_serving_provenance"

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

#: The segment finished its T-20 work, published every canary it was meant to,
#: and something in the live refresh did not happen.
#:
#: Its own outcome, for the same reason the advisory one has its own, and with
#: one addition: this is the outcome a segment reports when the two approval
#: variables are not set, which is the state this repository is in until
#: somebody sets them in the wizardofodds-production environment. So it has to
#: be a success -- the canary is published, the slate is served, the T-20
#: deliverable is complete -- and it has to be impossible to mistake for a
#: clean run. The receipt names every live refusal with the message the
#: refresh wrote, which names both variables.
OUTCOME_SEGMENT_COMPLETE_LIVE_DEGRADED = "T20_SEGMENT_COMPLETE_LIVE_DEGRADED"

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

#: A window whose serving-window validation checks did not pass. Its own
#: outcome rather than the generic failure, because the operator response is
#: different: the capture is there and the fit is sound, and what failed is
#: the protocol compatibility of that capture with what serving requires.
OUTCOME_SERVING_WINDOW_GATES_REFUSED = "T20_SERVING_WINDOW_GATES_REFUSED"

#: A release this run would have made for a game that has already started.
#:
#: The guard nothing anywhere performed. ``select_capture_bundle`` refuses a
#: capture *taken* after its game tipped, which is a statement about when a
#: snapshot may exist; nothing was a statement about when a price may be
#: shown. On 2026-10-23 the T-20 group holding game 21717941 was priced and
#: published two hours and forty minutes after that game tipped, because
#: runner pickup was late, and every stage of the pipeline reported success:
#: the capture was pre-tip and eligible, the serving receipt said SERVED and
#: the feed said PUBLISHED. The release that landed was a pregame page for a
#: game in its third quarter.
#:
#: Refused rather than recorded, because a post-tip release is the one claim
#: this integration must never make. It is also permanent in exactly the way
#: a duplicate conflict is -- the window's T-20 moment is behind the slate
#: and no later run moves a tip backwards -- so a window refused here is
#: refused identically by every segment after it.
OUTCOME_POST_TIP_PUBLICATION = "POST_TIP_PUBLICATION_REFUSED"

#: A segment that finishes with a capture of this slate that no release
#: carries. Distinguished from every refusal above because nothing refused:
#: the capture is sound, the recovery was attempted, and it did not get all
#: the way to a publication. It is not a success either, which is the whole
#: point -- the state it names is the one that was previously invisible.
OUTCOME_CAPTURED_NOT_SERVED = "T20_CAPTURED_WINDOWS_UNSERVED"

#: What one captured-but-unserved window that selection refuses is recorded
#: as. Not a segment outcome: a segment that has one of these still ends on
#: OUTCOME_CAPTURED_NOT_SERVED above, because what the exit code has to say
#: is that production is missing a window. This names the window, inside
#: that outcome, and separates the ones a later run can still publish from
#: the ones no run will.
RECOVERY_OUTCOME_UNRECOVERABLE = "UNRECOVERABLE_NO_ELIGIBLE_CAPTURE"

OUTCOMES_THAT_SUCCEED: frozenset[str] = frozenset(
    {
        OUTCOME_NO_WINDOWS,
        OUTCOME_SEGMENT_COMPLETE,
        OUTCOME_SEGMENT_COMPLETE_ADVISORY_DEGRADED,
        OUTCOME_SEGMENT_COMPLETE_LIVE_DEGRADED,
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

#: Its game started before the release could be made. The advisory twin of
#: :data:`OUTCOME_POST_TIP_PUBLICATION`, and the only advisory outcome that
#: reaches the exit code.
#:
#: Recorded here rather than raised, because raising from inside the advisory
#: path is the one thing that path may never do: a refusal thrown while a
#: later T-20 group is still ahead would cost the slate the capture of a game
#: that has not tipped, and a capture is the one step in this pipeline that
#: cannot be retried. So the refusal is taken where it is, and :meth:`run`
#: escalates it to the segment outcome once all the capture work is behind
#: it -- the same arrangement production already uses for a recovery refusal
#: that has to wait for the work it would otherwise have prevented.
ADVISORY_OUTCOME_POST_TIP = "ADVISORY_POST_TIP_PUBLICATION_REFUSED"

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

#: What one window's live refresh is recorded as. Restated from
#: ``run_progressive_live_refresh`` for the reason the advisory marker is
#: restated, and pinned against that module's own constants by a test.
LIVE_OUTCOME_ESTABLISHED = "LIVE_ESTABLISHED"
LIVE_OUTCOME_REPLACED = "LIVE_REPLACED"
LIVE_OUTCOME_ALREADY_CURRENT = "LIVE_ALREADY_CURRENT"
LIVE_OUTCOME_PROMOTED_THEN_INCOMPLETE = "LIVE_PROMOTED_THEN_INCOMPLETE"
LIVE_OUTCOME_REFUSED = "LIVE_REFUSED"

#: The ordering rule declined this window because every game in it is already
#: standing at one closer to its own tip. Not a defect: a re-run of a window
#: that already published, or a delayed invocation whose games have since been
#: advanced by a later window, asks to move those games away from tip and is
#: told no, which is the ordering rule working.
#:
#: Rare now, and that is the point of the per-game rule. A window belongs to a
#: game and its offset is measured from that game's own tip, so a slate with
#: several tip times interleaves them -- the first game's T-20 capture is
#: taken before the last game's T-180 capture, which is a later moment in the
#: evening and an earlier point in its own game's window. Decided against one
#: slate-wide pointer that was the ordinary evening case and it cost every
#: later game its early windows; decided per game it only fires when the
#: particular games asked for are genuinely already closer in.
LIVE_OUTCOME_ALREADY_CLOSER_TO_TIP = "LIVE_ALREADY_CLOSER_TO_TIP"

#: The live feed was refreshed and the page a reader actually opens was not.
#: Its own outcome because it is the worst of the live failures: the release
#: is current, the digests agree, and the page is the window before. Nothing
#: a reader sees is wrong -- the page is a real validated release -- but it is
#: not the one the feed beside it names.
LIVE_OUTCOME_PAGE_FAILED = "LIVE_PAGE_FAILED"

#: The live refresh was not attempted because this invocation had already
#: refreshed live from the identical priced artifact for this window. The
#: live twin of :data:`PUBLICATION_OUTCOME_SKIPPED`, and recorded only after
#: a refresh that succeeded on those bytes: a refresh that failed on them is
#: retried, because the pointer is then one window behind for a reason this
#: invocation could still fix.
LIVE_OUTCOME_SKIPPED = "LIVE_SKIPPED_ALREADY_CURRENT_THIS_STATE"

#: Live outcomes that are not a defect. Only an outcome outside this set
#: degrades the segment. The first three are the ones after which the live
#: pointer names this window's release, the fourth is this invocation having
#: already put it there, and the fifth is the ordering rule leaving a release
#: closer to tip alone.
LIVE_OUTCOMES_WITHOUT_FAULT: frozenset[str] = frozenset(
    {
        LIVE_OUTCOME_ESTABLISHED,
        LIVE_OUTCOME_REPLACED,
        LIVE_OUTCOME_ALREADY_CURRENT,
        LIVE_OUTCOME_SKIPPED,
        LIVE_OUTCOME_ALREADY_CLOSER_TO_TIP,
    }
)

#: The live outcomes after which there is a release to build the page from.
#: Narrower than the faultless set on purpose: the ordering rule declining a
#: window is not a fault and is also not a publication, so the page stage is
#: not reached and the page is not reported as rebuilt.
LIVE_OUTCOMES_THAT_PUBLISH: frozenset[str] = frozenset(
    {
        LIVE_OUTCOME_ESTABLISHED,
        LIVE_OUTCOME_REPLACED,
        LIVE_OUTCOME_ALREADY_CURRENT,
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
    """A fail-closed refusal carrying the outcome it must be reported as.

    It also carries what the run was leaving behind. A refusal that happens
    after a capture is the case that cost 2026-10-23 three games, and the
    receipt it wrote said only that serving had failed -- so the windows the
    run had captured and would never publish were named nowhere, by the run
    that lost them or by the one that followed it.
    """

    def __init__(self, outcome: str, message: str) -> None:
        super().__init__(message)

        self.outcome = outcome

        self.captured_unserved_windows: list[str] = []


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

    # The production window first, because it is the specific mistake and
    # "not among the advisory windows" would be a true but useless way to
    # describe it.
    if PRODUCTION_OFFSET_MINUTES in requested:
        raise ControllerRefused(
            OUTCOME_FAILED,
            f"T-{PRODUCTION_OFFSET_MINUTES} was asked for as an advisory "
            "window. It is the production window: it has its own capture, its "
            "own serving entry point, its own receipt and its own release "
            "stream, and serving it as advisory would put the same snapshot "
            "into two release streams under two names",
        )

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


def windows_that_have_tipped(
    rows: Sequence[dict[str, Any]], *, now: datetime
) -> list[dict[str, Any]]:
    """The windows among ``rows`` whose game has already started.

    The subject is the windows a publication is *for*, never every window in
    the release, and that distinction is the whole of why this is scoped to
    rows rather than read off the capture log. A release is cumulative -- one
    serve at an offset prices every game with an eligible capture at that
    offset -- so by the time the last T-20 group of a staggered twelve-game
    slate is published, the first game tipped two hours earlier and its rows
    are correctly in the release, because they were priced and published
    before their own tip. What has to still be ahead is the tip of the game
    whose window this publication is the publication for.

    Earliest tip first, so a refusal naming several reads in the order the
    games started.
    """
    tipped = [
        {
            "game_id": int(row["game_id"]),
            "minutes_since_tip": round(
                (now - row["tip_utc"]).total_seconds() / 60.0, 3
            ),
            "observed_at_utc": now.isoformat(),
            "tip_utc": row["tip_utc"].isoformat(),
            "window_id": str(row["window_id"]),
        }
        for row in rows
        if now >= row["tip_utc"]
    ]

    return sorted(
        tipped, key=lambda entry: (entry["tip_utc"], entry["window_id"])
    )


def post_tip_reason(entry: dict[str, Any]) -> str:
    """Why one window is refused, in the terms an operator can act on.

    The game, the moment it started and how long ago that was, because
    "post-tip" on its own does not tell an operator whether a runner was
    thirty seconds late or three hours late, and those are different
    incidents.
    """
    return (
        f"game {entry['game_id']} tipped at {entry['tip_utc']}, "
        f"{entry['minutes_since_tip']:g} minutes before "
        f"{entry['observed_at_utc']}, so the window {entry['window_id']} is "
        "not a pregame price and is not published. The release already "
        "serving is left exactly as it is: nothing is retracted, "
        "overwritten or repointed"
    )


def assert_no_window_has_tipped(
    rows: Sequence[dict[str, Any]], *, now: datetime
) -> list[dict[str, Any]]:
    """Every window this publication is for is still ahead of its own tip.

    Checked before the serving entry point is invoked, so a refusal costs
    nothing and writes nothing: no priced artifact is rebuilt, no release is
    minted, no ``current.json`` is repointed and no ledger row is appended.
    The release a reader already has is the release a reader keeps.

    Returns the windows it cleared, with how long each has left, so a receipt
    can show the margin rather than only the verdict.
    """
    tipped = windows_that_have_tipped(rows, now=now)

    if tipped:
        raise ControllerRefused(
            OUTCOME_POST_TIP_PUBLICATION,
            "; ".join(post_tip_reason(entry) for entry in tipped),
        )

    return [
        {
            "game_id": int(row["game_id"]),
            "minutes_before_tip": round(
                (row["tip_utc"] - now).total_seconds() / 60.0, 3
            ),
            "tip_utc": row["tip_utc"].isoformat(),
            "window_id": str(row["window_id"]),
        }
        for row in sorted(rows, key=lambda row: str(row["window_id"]))
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


def served_window_ids(served_path: Path) -> set[str]:
    """The windows some published release of this slate actually carried.

    The second half of "is this window done", and the half nothing recorded.
    :func:`completed_window_ids` answers whether a capture exists, which is a
    fact about the input; this answers whether that input ever reached a
    reader, which is the fact a run is actually responsible for.

    Read the same way and for the same reason: an append-only log of what
    happened, tolerant of a line a killed process never finished writing,
    because a torn last line must not hide the rows before it.
    """
    if not served_path.exists():
        return set()

    served: set[str] = set()

    for line in served_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except ValueError:
            continue

        if not isinstance(row, dict):
            continue

        if row.get("record_type") != SERVED_RECORD_TYPE:
            continue

        for window_id in row.get("window_ids") or []:
            served.add(str(window_id))

    return served


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

        self.served_path = (
            self.data_root
            / SERVED_WINDOWS_RELATIVE
            / f"{self.slate_date}.jsonl"
        )

        self.serving_provenance_path = (
            self.data_root
            / SERVING_PROVENANCE_RELATIVE
            / f"{self.slate_date}.jsonl"
        )

        self.captures: list[dict[str, Any]] = []
        self.recovered: list[dict[str, Any]] = []

        #: One record per captured-but-unserved window this segment proved
        #: selection will not accept, carrying the selection's own reason.
        #: Permanent: see :meth:`_record_unrecoverable`.
        self.unrecoverable: list[dict[str, Any]] = []

        #: One record per recovery that verified and still did not reach a
        #: release, because a serving or publication stage refused. Separate
        #: from :attr:`unrecoverable` because these captures are intact and
        #: a later segment can still publish them.
        self.recovery_faults: list[dict[str, Any]] = []

        #: How long each published window had left before its own tip, in the
        #: order the publications were made. The cleared half of the post-tip
        #: guard, kept because a receipt that only ever records refusals
        #: cannot show how close a release that went out actually was.
        self.pre_tip_margins: list[dict[str, Any]] = []

        self.publications: list[dict[str, Any]] = []
        self.window_gates: list[dict[str, Any]] = []
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

        #: One record per live refresh this segment attempted, in order, so
        #: "which window is live and what happened on the way there" is
        #: answerable from the receipt without opening the destination.
        self.live: list[dict[str, Any]] = []

        self._published_priced_digest: str | None = None

        #: Publication idempotency, per destination. The T-20 digest and each
        #: advisory window's digest are tracked apart, because the same priced
        #: bytes published to two different destinations are two different
        #: releases and skipping the second would leave a destination empty.
        self._published_advisory_digests: dict[int, str] = {}

        #: The priced digest the live refresh last *succeeded* on, per window.
        #: Separate from the publication digests above because succeeding at a
        #: canary and succeeding at live are different facts: the canary can
        #: publish and the live refresh refuse, which is exactly the state
        #: this repository is in until both approval variables are set, and in
        #: that state a canary skip must not be read as live being current.
        self._live_refreshed_digests: dict[int, str] = {}

        #: The feed publication receipt each window's live refresh is the live
        #: half of, retained so a repeat of identical priced bytes can still
        #: retry a live refresh that failed. The canary stage writes one per
        #: stage directory and the skip path has none of its own.
        self._feed_receipts: dict[int, Path] = {}

        #: The serving receipt behind each of those, for the same reason: the
        #: publisher re-hashes the priced artifact against it.
        self._serving_receipts: dict[int, Path] = {}

        #: The capture every window's prices were taken from, by window id.
        #: Filled in at the one place that already knows it -- the bundle
        #: selection the pricing path itself performs -- and read only when a
        #: live refresh is handed a window, so that the composite manifest can
        #: name the snapshot behind each game's card. Carried rather than
        #: re-selected because selecting again would be a second answer to a
        #: question already answered, and a composite may not name a capture
        #: nobody can find again.
        self._capture_identity: dict[str, dict[str, str]] = {}

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

        # Slate-wide rather than segment-wide, and deliberately. A window
        # captured and never published was lost by whichever run owned it,
        # and the run that can do something about it is any later one --
        # including one in a different segment. Scoping this to the segment
        # would mean the only run allowed to recover a window is the one that
        # already failed to publish it.
        unserved = self._unserved_rows(rows, completed=completed)

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
            "captured_unserved_windows": [
                str(row["window_id"]) for row in unserved
            ],
            "pending_windows": [str(row["window_id"]) for row in pending],
            "rows": rows,
            "pending_rows": pending,
            "unserved_rows": unserved,
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

    def _unserved_rows(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        completed: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """This slate's windows that are captured and that nothing published.

        Both facts are read from durable state rather than from this run's
        own memory, because the run that left a window in this state is by
        construction a run that did not get to write a receipt.
        """
        if completed is None:
            completed = completed_window_ids(self.capture_run_path)

        served = served_window_ids(self.served_path)

        return [
            row
            for row in rows
            if str(row["window_id"]) in completed
            and str(row["window_id"]) not in served
        ]

    # -- the segment ---------------------------------------------------

    def run(self) -> dict[str, Any]:
        started_at = self.clock()

        plan = self.plan()

        self.advisory += list(plan["advisory_unsupported"])

        # Three kinds of work now, and the segment exits early only if it has
        # none of them. ``pending_rows`` is a T-20 window still to capture,
        # ``unserved_rows`` is one already captured that no release carries,
        # and ``advisory_pending_rows`` is an early window this segment owns.
        # They are independent: a segment can own no T-20 window at all and
        # still have three advisory windows to serve, and a segment with
        # nothing left to capture can still be the run that recovers a window
        # an earlier one captured and died before publishing.
        if (
            not plan["pending_rows"]
            and not plan["unserved_rows"]
            and not plan["advisory_pending_rows"]
        ):
            return self._receipt(
                plan=plan,
                started_at=started_at,
                outcome=OUTCOME_NO_WINDOWS,
                reason=(
                    f"the segment {self.segment_start.isoformat()} to "
                    f"{self.segment_end.isoformat()} contains no uncaptured "
                    f"T-{self.offset_minutes} window for {self.slate_date}, "
                    "no advisory window it owns, and no capture of that "
                    "slate waiting to be published, so no capture, serving "
                    "or publication was performed"
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
        #
        # A captured-but-unserved window counts as T-20 work for this guard.
        # Recovery publishes a production release off an existing capture and
        # needs the same verified incumbent a fresh capture does; degrading
        # here with one outstanding would turn a window production is still
        # missing into an advisory footnote.
        try:
            readiness = morning_readiness(
                data_root=self.data_root,
                slate_date=self.slate_date,
                production_sha=self.production_sha,
                project_root=self.project_root,
            )
        except ControllerRefused as refusal:
            if plan["pending_rows"] or plan["unserved_rows"]:
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

        # Recovery first. These windows are already past their T-20 time, so
        # nothing is gained by making them wait behind a group that is still
        # in the future, and the games they belong to may be about to tip.
        # Ahead of the advisory windows for the same reason, and that is the
        # stronger statement: an advisory release is a reading aid, and a
        # recovered T-20 release is a window production still owes.
        deferred: ControllerRefused | None = None

        try:
            if plan["unserved_rows"]:
                # Recovery keeps its place at the front and loses its veto
                # over everything behind it. Its two failure modes are
                # confined differently, because they mean different things.
                #
                # A window selection refuses is recorded by :meth:`_recover`
                # and skipped: it is already lost and nothing this segment
                # does can change that.
                #
                # A serving or publication stage that refuses is caught here
                # and re-raised below, after the work rather than instead of
                # it. The refusal is not downgraded -- it leaves this method
                # with its own outcome and its own message, exactly as it
                # did when it was raised from this line. What changes is
                # when. Raised here it lands before any collector has run,
                # so it costs the segment the captures of games that have
                # not tipped yet, and a capture is the one thing in this
                # pipeline that cannot be retried: the market at T-20 exists
                # once. Publication can wait for the next segment. The
                # snapshot cannot.
                try:
                    self._recover(plan["unserved_rows"], readiness=readiness)
                except ControllerRefused as refusal:
                    deferred = refusal

                    self._record_recovery_fault(refusal)

            if plan["pending_rows"]:
                self._work(plan, readiness)

            # Whatever advisory windows are left: the ones after this
            # segment's last T-20 window, the ones held back for a T-20
            # capture, and all of them on a segment that owns no T-20 window
            # at all. Inside the same guard as the T-20 work so that a
            # refusal raised anywhere in the segment still leaves naming the
            # windows it is abandoning.
            self._finish_advisory(readiness)

            # The deferred refusal, raised now that the work it would have
            # prevented has happened. Conditional on the windows still being
            # unserved because the slate is re-served whole: if the capture
            # work above published a release, that release carries the
            # windows this refusal was about and the refusal has stopped
            # being true. Raising a resolved refusal would be its own kind
            # of false report.
            if deferred is not None and self._unserved_rows(plan["rows"]):
                raise deferred

        except ControllerRefused as error:
            # A refusal after a capture is the shape the incident had. The
            # refusal itself stands -- nothing is downgraded and nothing is
            # retried here -- but it leaves with the windows it is
            # abandoning attached, so the receipt and the job summary name
            # them instead of the operator having to read the capture log
            # against the published feed.
            error.captured_unserved_windows = [
                str(row["window_id"])
                for row in self._unserved_rows(plan["rows"])
            ]

            raise

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

        live_faults = [
            record
            for record in self.live
            if str(record.get("outcome")) not in LIVE_OUTCOMES_WITHOUT_FAULT
        ]

        outstanding = [
            str(row["window_id"])
            for row in self._unserved_rows(plan["rows"])
        ]

        # Ahead of the advisory verdict, and not merged into it. An advisory
        # window that failed degrades the segment; a T-20 capture nothing
        # published is a window production is missing, and that outcome is
        # the one that has to reach the exit code. The advisory records are
        # on the receipt either way -- the outcome says which fact is worse.
        if outstanding:
            return self._receipt(
                plan=plan,
                started_at=started_at,
                outcome=OUTCOME_CAPTURED_NOT_SERVED,
                reason=self._unserved_reason(outstanding),
                readiness=readiness,
            )

        # The advisory path's post-tip refusals, escalated to the segment
        # outcome now that every capture this segment owns is behind it.
        #
        # Behind the outstanding check above because a T-20 window production
        # is missing is the worse fact and keeps the outcome it has always
        # had; a post-tip T-20 refusal lands there anyway, because the
        # window it refused stays unserved.
        #
        # Decided here rather than where the refusal is taken for the reason
        # the deferred recovery refusal is: raised from inside the advisory
        # path it would abandon the segment with a later T-20 group still
        # ahead of it, and the capture of a game that has not tipped is the
        # one thing here that cannot be retried. Reported rather than
        # degraded because a release for a game already under way is not an
        # advisory footnote -- it is the claim this integration exists to
        # refuse, and an operator who sees a green job does not go looking
        # for it.
        post_tip = [
            record
            for record in self.advisory
            if str(record.get("outcome")) == ADVISORY_OUTCOME_POST_TIP
        ]

        if post_tip:
            return self._receipt(
                plan=plan,
                started_at=started_at,
                outcome=OUTCOME_POST_TIP_PUBLICATION,
                reason=(
                    f"{len(post_tip)} advisory window(s) of {self.slate_date} "
                    "were reached after their own game had started, so the "
                    "release each one would have made was refused: "
                    + "; ".join(
                        str(record.get("reason") or "") for record in post_tip
                    )
                    + ". Every release already published is untouched. This "
                    "segment is not reported as a success"
                ),
                readiness=readiness,
            )

        reason = (
            f"{len(self.captures)} T-{self.offset_minutes} capture(s) "
            f"completed for {self.slate_date}"
            + (
                f" and {len(plan['captured_unserved_windows'])} earlier "
                "capture(s) recovered"
                if self.recovered
                else ""
            )
            + ", each served and published to the canary destination"
        )

        if self.advisory_offsets:
            reason += (
                f"; {len(published)} advisory window(s) published to their "
                f"own confined canary destinations and {len(faults)} failed"
            )

        if self.live:
            reason += (
                f"; {len(self.live) - len(live_faults)} of {len(self.live)} "
                "live refresh(es) left the live release naming the window "
                "that published"
            )

            if live_faults:
                reason += (
                    ". The live refusals are in live_refreshes with the "
                    "message each one wrote; the release already live is "
                    "still serving"
                )

        # Advisory first when both degraded, because it is the broader
        # failure: an advisory window that did not publish has no canary and
        # so could not have refreshed live either, and the counts of both are
        # in the receipt whichever word the outcome carries.
        return self._receipt(
            plan=plan,
            started_at=started_at,
            outcome=(
                OUTCOME_SEGMENT_COMPLETE_ADVISORY_DEGRADED
                if faults
                else OUTCOME_SEGMENT_COMPLETE_LIVE_DEGRADED
                if live_faults
                else OUTCOME_SEGMENT_COMPLETE
            ),
            reason=reason,
            readiness=readiness,
        )

    def _recover(
        self, rows: Sequence[dict[str, Any]], *, readiness: dict[str, Any]
    ) -> None:
        """Publish captures an earlier run made and never got to publish.

        No collector is invoked and nothing sleeps. The windows are past
        their T-20 time by definition -- that is what makes them
        unrecoverable by capture -- and a snapshot taken now would be a
        snapshot of now, so there is nothing here that could produce one,
        relabel one or re-time one. What this does is exactly what a second
        due group in the same segment already does: prove the existing
        captures are the ones pricing accepts, then run the same serving and
        publication entry points over the slate, which enumerate every
        eligible capture and publish the whole priced artifact.

        The proof is ``select_capture_bundle`` and therefore still requires
        exactly one scheduled capture per window, no recorded errors, every
        component digest recomputed and the capture pre-tip. A capture that
        has stopped being eligible is never published.

        Proven one window at a time, which is the difference between this
        and the capture path. A window that no longer verifies is recorded
        and skipped rather than raised for: it is already lost, and a single
        verification over the whole set made it take the verifiable windows
        beside it, the segment's own pending captures and the advisory
        windows after them. The unit of isolation is the verification and
        not the publication because publication is not divisible -- the
        serving entry point prices the whole slate and the feed publishes
        every row of it, so there is no such thing as publishing one window.
        What that costs is nothing, because the selection performed here is
        the selection serving performs: a window refused here is a window
        serving would not have put in the release either.
        """
        verified: list[dict[str, Any]] = []

        recovering: list[dict[str, Any]] = []

        now = self.clock()

        for row in sorted(rows, key=lambda row: str(row["window_id"])):
            bundle, error = self._select_capture(int(row["game_id"]))

            if error is not None:
                self._record_unrecoverable(row, error)

                continue

            # The post-tip refusal, recorded the same way and for the same
            # reason: it is permanent, and a permanent refusal must cost the
            # window it is about and nothing else. A recovery set spans the
            # whole slate, so on a staggered evening it routinely holds games
            # that have tipped beside games whose tip is still an hour away,
            # and raising for the first would discard the second.
            #
            # Behind the selection rather than in front of it, because the two
            # ask different questions and the selection's answer is the more
            # specific of the two when both apply. Selection asks whether the
            # *capture* is eligible, which includes it being pre-tip and is a
            # fact about the snapshot, settled for good when it was written.
            # This asks whether a *release* may be made now, which depends on
            # the clock and is the question nothing asked. A capture taken
            # after tip fails both -- its window is necessarily in the past by
            # the time anything reads it -- and the operator is better served
            # by being told the capture is unusable than by being told the
            # game has started.
            tipped = windows_that_have_tipped([row], now=now)

            if tipped:
                self._record_unrecoverable(
                    row,
                    ControllerRefused(
                        OUTCOME_POST_TIP_PUBLICATION,
                        post_tip_reason(tipped[0]),
                    ),
                    outcome=OUTCOME_POST_TIP_PUBLICATION,
                )

                continue

            recovering.append(row)

            verified.append(bundle)

        # Nothing verified, so there is nothing to publish from. Returning
        # rather than serving anyway matters: a serve here would re-publish
        # the slate's existing release for no reason, and the windows that
        # caused it are recorded above either way.
        if not verified:
            return

        stage_dir = self.run_dir / "recovery"

        stage_dir.mkdir(parents=True, exist_ok=True)

        self.recovered.append(
            {
                "capture_ids": sorted(
                    {bundle["capture_id"] for bundle in verified}
                ),
                "game_ids": sorted(
                    {int(bundle["game_id"]) for bundle in verified}
                ),
                "recovered_at_utc": self.clock().isoformat(),
                "reason": (
                    "these windows were captured by an earlier run that did "
                    "not publish them, so they are served from the captures "
                    "that already exist"
                ),
                "verified_bundles": verified,
                "window_ids": sorted(
                    str(bundle["window_id"]) for bundle in verified
                ),
            }
        )

        # The post-tip guard is about the windows the release is *for*, and
        # those are the ones being recovered -- not every row handed in. A row
        # this pass already refused is not part of the publication, so letting
        # it reach the guard would make one excluded window refuse the whole
        # recovery, which is the blast radius the isolation above removes.
        self._serve_and_publish(
            stage_dir, windows=recovering, readiness=readiness
        )

    def _record_unrecoverable(
        self,
        row: dict[str, Any],
        error: RuntimeError,
        *,
        outcome: str = RECOVERY_OUTCOME_UNRECOVERABLE,
    ) -> None:
        """Record one window that no later run of this controller will serve.

        Permanent by construction rather than by judgement. Recovery only
        ever looks at windows already past their T-20 time, so no run can
        take a different capture of one; the capture log is append-only, so
        the bytes selection reads do not change; and the selection is a
        deterministic function of those bytes. A window refused here is
        refused identically by every segment that follows, which is why it
        is reported as a window production has lost rather than as one still
        waiting for a later run.

        The reason recorded is the selection's own, not a summary of it, so
        the receipt distinguishes a duplicate conflict from a post-tip
        capture from a digest that no longer matches the bytes on disk.

        ``outcome`` separates the refusals selection makes from the one the
        clock makes. Both are permanent, both leave the window unserved and
        both keep the segment red, so they share this record and the unserved
        set it feeds; they are told apart because the operator response
        differs. A conflicted or corrupt capture is a defect in the capture
        log. A window refused for :data:`OUTCOME_POST_TIP_PUBLICATION` has a
        perfectly sound capture and was simply reached too late, which is a
        question about scheduling and runner pickup.
        """
        window_id = str(row["window_id"])

        if any(
            record["window_id"] == window_id for record in self.unrecoverable
        ):
            return

        self.unrecoverable.append(
            {
                "game_id": int(row["game_id"]),
                "observed_at_utc": self.clock().isoformat(),
                "outcome": str(outcome),
                "reason": str(error),
                "window_id": window_id,
            }
        )

    def _unserved_reason(self, outstanding: Sequence[str]) -> str:
        """Why each window is unserved, split by whether anything can fix it.

        The single sentence this replaces said the captures "stay exactly as
        they were taken; a later run will serve and publish from them
        without recapturing". That is true of a window whose segment ran out
        of time and false of a conflicted or post-tip one, and the operator
        it misleads is the one who waits for a recovery that cannot happen.
        """
        refused = {
            str(record["window_id"]): str(record["reason"])
            for record in self.unrecoverable
        }

        dead = [
            window_id for window_id in outstanding if window_id in refused
        ]

        waiting = [
            window_id for window_id in outstanding if window_id not in refused
        ]

        reason = (
            f"{len(outstanding)} T-{self.offset_minutes} capture(s) of "
            f"{self.slate_date} exist and no release carries them: "
            f"{', '.join(outstanding)}"
        )

        if dead:
            reason += (
                f". {len(dead)} of them cannot be served from the capture "
                "that exists, for a reason no later run can repair, so "
                "production has lost the window: "
                + "; ".join(
                    f"{window_id} ({refused[window_id]})"
                    for window_id in dead
                )
            )

        if waiting:
            reason += (
                f". {len(waiting)} capture(s) are sound and stay exactly as "
                "they were taken; a later run will serve and publish from "
                f"them without recapturing: {', '.join(waiting)}"
            )

        return reason + ". This segment is not reported as a success"

    def _record_recovery_fault(self, refusal: ControllerRefused) -> None:
        """Record a recovery that got past verification and still did not land.

        Kept apart from :attr:`unrecoverable` because the two mean opposite
        things to whoever reads the receipt. A window in that list is gone.
        A fault here is the serving or publication stage refusing, which
        says nothing about the captures: they still verify, and the next
        segment can still publish them.
        """
        self.recovery_faults.append(
            {
                "observed_at_utc": self.clock().isoformat(),
                "outcome": str(refusal.outcome),
                "reason": str(refusal),
            }
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

        # Finished with as a *capture*, which is what the scheduling loop
        # asks about. Its serving is tracked by the held list instead, because
        # a window whose capture was taken must not be woken for again -- the
        # capture is the point-in-time market and taking it twice is at best
        # wasted work and at worst a second scheduled capture for a window
        # that accepts exactly one.
        self._advisory_handled.add(window_id)

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

        return self._remember_capture(
            {
                "capture_id": bundle.capture_id,
                "captured_at_utc": bundle.captured_at,
                "game_id": int(row["game_id"]),
                "window_id": bundle.window_id,
            }
        )

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

    def _drop_windows_that_have_tipped(
        self,
        held: Sequence[dict[str, Any]],
        *,
        offset: int,
    ) -> list[dict[str, Any]]:
        """The held windows still ahead of their tip, refusing the rest.

        An advisory window is captured three hours, an hour and a half or
        three quarters of an hour before tip, so on an ordinary evening it is
        served and published long before its game starts. It can still be
        reached late: a window served inside the reservation before a T-20
        capture is held back, and the gap it is retried in can fall after the
        game it is for has tipped. That release would be the same wrong claim
        the T-20 one would be -- a pregame page for a game already under way
        -- published to the advisory destination instead of the canary.

        Refused per window and not per offset, which is the same isolation
        recovery keeps. One offset's held set spans the slate, so on a
        staggered evening it holds games that have tipped beside games whose
        tip is still an hour away, and refusing the whole offset would take
        the second with the first.
        """
        tipped = {
            entry["window_id"]: entry
            for entry in windows_that_have_tipped(held, now=self.clock())
        }

        for row in held:
            entry = tipped.get(str(row["window_id"]))

            if entry is None:
                continue

            self._record_advisory(
                row,
                outcome=ADVISORY_OUTCOME_POST_TIP,
                reason=post_tip_reason(entry),
            )

        remaining = [
            row for row in held if str(row["window_id"]) not in tipped
        ]

        self._advisory_held[offset] = list(remaining)

        return remaining

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
        release that cannot be made is a recorded advisory failure, and it
        never costs the segment a capture. One of those failures --
        :data:`ADVISORY_OUTCOME_POST_TIP` -- does reach the exit code, and it
        does so from :meth:`run` after all the capture work rather than from
        here, for exactly that reason.
        """
        rows = self._drop_windows_that_have_tipped(held, offset=offset)

        # Every held window was for a game that has started, so there is no
        # release left to make. Returning before the serve rather than serving
        # anyway is the point of the guard: nothing is priced, nothing is
        # published and the advisory release already serving stays exactly
        # where it is.
        if not rows:
            self._advisory_held[offset] = []

            return

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

            retained = self._feed_receipts.get(offset)

            if retained is not None:
                self._refresh_live(
                    stage_dir,
                    rows,
                    offset=offset,
                    canary_destination=feed_destination,
                    feed_receipt=retained,
                    serving_receipt=self._serving_receipts[offset],
                    priced_digest=digest,
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

        self._refresh_live(
            stage_dir,
            rows,
            offset=offset,
            canary_destination=feed_destination,
            feed_receipt=feed_receipt,
            serving_receipt=serving_receipt,
            priced_digest=digest,
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

        # This serve has already rewritten the slate's priced markets, so its
        # provenance is recorded before the advisory release is published,
        # exactly as the T-20 path does it. A failure here fails the advisory
        # window rather than the segment: the T-20 serve later in the night
        # rewrites those bytes and records its own receipt, so the slate stays
        # gradable, and an advisory window has never been allowed to take the
        # production window down with it.
        try:
            self._persist_serving_provenance(
                served, offset_minutes=offset, window_ids=windows
            )
        except ControllerRefused as error:
            self._fail_advisory(rows, offset=offset, reason=str(error))

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

    # -- the live release ----------------------------------------------

    def _record_live(
        self,
        *,
        offset: int,
        outcome: str,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.live.append(
            {
                "feed_destination": LIVE_FEED_DESTINATION,
                "final_pregame_window": (
                    int(offset) == min(PROGRESSIVE_LIVE_OFFSETS_MINUTES)
                ),
                "observed_at_utc": self.clock().isoformat(),
                "offset_minutes": int(offset),
                "outcome": outcome,
                "page_destination": LIVE_PAGE_DESTINATION,
                "reason": reason,
                **(extra or {}),
            }
        )

    def _refresh_live(
        self,
        stage_dir: Path,
        rows: Sequence[dict[str, Any]],
        *,
        offset: int,
        canary_destination: str,
        feed_receipt: Path,
        serving_receipt: Path,
        priced_digest: str,
    ) -> None:
        """Advance this group's games to the window that has just published.

        Called immediately after a window's canary feed and page have both
        succeeded, and in that order on purpose: what authorises a live
        publication is a canary of the identical bytes, so the canary has to
        have happened and have validated before this is reachable at all.

        Nothing here raises. A live refresh that cannot be made degrades the
        segment and never fails it, which is the same asymmetry the advisory
        path rests on and is load-bearing for a different reason here: until
        both approval variables are set in the wizardofodds-production
        environment, *every* live refresh refuses, and a slate that was
        captured, served and published must not be a red job because of a
        decision nobody has carried to the runner yet. The refusal is recorded
        verbatim, so the receipt and the step summary name both variables.

        The tips are this group's own, and so are the games this refresh
        advances. A release is cumulative -- one serve at an offset prices
        every game with an eligible capture at that offset -- so by the time
        the last T-20 group of a twelve-game slate publishes, the first game
        tipped two hours earlier and its rows are correctly in the release
        because they were priced before their own tip. The live slate the
        refresh composes holds every game at its own latest validated window,
        so handing it this group advances this group and leaves every other
        game standing exactly where its own last successful refresh put it.
        Each window is handed the capture its prices were selected from as
        well as its tip, because the composite manifest names the snapshot
        behind every game's card and this controller is the only thing that
        knows it.
        """
        self._feed_receipts[offset] = Path(feed_receipt)

        self._serving_receipts[offset] = Path(serving_receipt)

        if priced_digest and priced_digest == self._live_refreshed_digests.get(
            offset
        ):
            self._record_live(
                offset=offset,
                outcome=LIVE_OUTCOME_SKIPPED,
                reason=(
                    "this invocation already refreshed the live release from "
                    f"the identical T-{offset} priced artifact, so the "
                    "release it would publish is the release already serving"
                ),
                extra={"priced_market_artifact_sha256": priced_digest},
            )

            return

        live_receipt = stage_dir / "live_refresh.json"

        label = f"{LIVE_REFRESH_ENTRY_POINT} (T-{offset})"

        command = [
            sys.executable,
            str(LIVE_REFRESH_ENTRY_POINT),
            "--slate-date",
            self.slate_date,
            "--data-root",
            str(self.data_root),
            "--snapshot-offset-minutes",
            str(offset),
            "--canary-feed-receipt",
            str(feed_receipt),
            "--serving-receipt",
            str(serving_receipt),
            "--canary-destination-name",
            canary_destination,
            "--live-feed-destination-name",
            LIVE_FEED_DESTINATION,
            "--live-page-destination-name",
            LIVE_PAGE_DESTINATION,
            "--receipt-path",
            str(live_receipt),
        ]

        for row in rows:
            tip = row["tip_utc"]

            window_id = str(row["window_id"])

            command += [
                "--window-tip-utc",
                f"{window_id}="
                + (tip.isoformat() if hasattr(tip, "isoformat") else str(tip)),
            ]

            # Omitted rather than guessed when selection never recorded one.
            # The refresh refuses an advancing window it cannot name a
            # capture for, which is the right outcome: a card on the served
            # root whose snapshot nobody can find again is worse than no
            # card, and the refusal degrades the segment without touching
            # what is already live.
            capture = self._capture_identity.get(window_id)

            if capture is not None:
                command += [
                    "--window-capture",
                    "{}={}@{}".format(
                        window_id,
                        capture["capture_id"],
                        capture["captured_at_utc"],
                    ),
                ]

        if self.summary_path is not None:
            command += ["--summary-path", str(self.summary_path)]

        result = self.run_command(
            command, environment=self._environment(), cwd=self.project_root
        )

        refreshed = self._live_receipt(live_receipt)

        if refreshed is None:
            self._record_live(
                offset=offset,
                outcome=LIVE_OUTCOME_REFUSED,
                reason=(
                    f"{label} exited {result.returncode} and wrote no "
                    f"readable receipt at {live_receipt}: "
                    + ((result.stderr or result.stdout) or "")[-1500:]
                ),
            )

            return

        outcome = str(refreshed.get("live_outcome") or LIVE_OUTCOME_REFUSED)

        released = refreshed.get("release") or {}

        shared = {
            "canary_destination": canary_destination,
            "priced_market_artifact_sha256": priced_digest,
            "release_id": str(released.get("release_id") or ""),
            "rows_sha256": str(released.get("rows_sha256") or ""),
        }

        # Two things end here, and only one of them is a fault. A refusal is
        # recorded and degrades the segment; the ordering rule declining a
        # window is recorded and does not. Both stop before the page for the
        # same reason: there is no release to build one from, and rebuilding
        # the page from a release that did not change would say the page had
        # been refreshed when nothing was.
        if result.returncode != 0 or outcome not in LIVE_OUTCOMES_THAT_PUBLISH:
            self._record_live(
                offset=offset,
                outcome=outcome,
                reason=str(
                    refreshed.get("message") or refreshed.get("reason") or ""
                )
                or f"{label} exited {result.returncode}",
                extra={
                    **shared,
                    "observed_live_release_id": str(
                        refreshed.get("observed_live_release_id") or ""
                    ),
                },
            )

            return

        page_receipt = stage_dir / "live_wizardofodds_page.json"

        page = self.run_command(
            [
                sys.executable,
                str(PUBLICATION_ENTRY_POINT),
                "--stage",
                "page",
                "--mode",
                LIVE_PUBLICATION_MODE,
                "--slate-date",
                self.slate_date,
                "--feed-receipt",
                str(live_receipt),
                "--receipt-path",
                str(page_receipt),
                *(
                    ("--summary-path", str(self.summary_path))
                    if self.summary_path is not None
                    else ()
                ),
            ],
            environment=self._environment(),
            cwd=self.project_root,
        )

        built = self._live_receipt(page_receipt)

        if page.returncode != 0 or (built or {}).get(
            "outcome"
        ) != PUBLICATION_OUTCOME_PUBLISHED:
            self._record_live(
                offset=offset,
                outcome=LIVE_OUTCOME_PAGE_FAILED,
                reason=(
                    f"the T-{offset} release {shared['release_id']} is live "
                    "and the page a reader opens was not rebuilt from it "
                    f"({PUBLICATION_ENTRY_POINT} --stage page exited "
                    f"{page.returncode}): "
                    + (
                        str((built or {}).get("message") or "")
                        or ((page.stderr or page.stdout) or "")[-1500:]
                    )
                ),
                extra={**shared, "live_outcome": outcome},
            )

            return

        self._live_refreshed_digests[offset] = priced_digest or ""

        self._record_live(
            offset=offset,
            outcome=outcome,
            reason=str(refreshed.get("reason") or ""),
            extra={
                **shared,
                "page_sha256": str((built or {}).get("page_sha256") or ""),
                "previous_release_id": str(
                    refreshed.get("previous_release_id") or ""
                ),
            },
        )

    def _live_receipt(self, path: Path) -> dict[str, Any] | None:
        """One receipt, read without refusing. ``None`` when there is none.

        The live twin of :meth:`_read_receipt`, and it does not raise for the
        same reason :meth:`_advisory_stage` does not: a live refresh writes a
        receipt on the way out of a refusal as well as a success, so reading
        it is how the refusal gets into the segment's own record. A receipt
        that is missing or unreadable is itself recorded, by the caller.
        """
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

        return document if isinstance(document, dict) else None

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

        self._serve_and_publish(stage_dir, windows=group, readiness=readiness)

    def _select_capture(
        self, game_id: int
    ) -> tuple[dict[str, Any] | None, RuntimeError | None]:
        """One game's eligible T-20 bundle, or the error selection refused it with.

        The selection is ``select_capture_bundle`` and is not reimplemented
        here: exactly one scheduled capture for the window, no recorded
        errors, every component hash recomputed, the capture id recomputed,
        the target game present and the capture pre-tip.

        Returned rather than raised because the two callers need opposite
        things from a refusal, and that difference is the whole of this
        split. A capture this run has just taken and cannot select is the
        failure this controller exists to make loud. A capture an earlier
        run took and this one cannot select is one window of the slate, and
        raising for it costs the segment every other window it owns.

        Imported here because ``prospective_snapshot`` pulls pandas, which the
        probe must not need.
        """
        from nba_prop_quant.prospective_snapshot import select_capture_bundle

        try:
            bundle = select_capture_bundle(
                self.snapshot_dir,
                date=self.slate_date,
                game_id=int(game_id),
                offset_minutes=self.offset_minutes,
            )
        except RuntimeError as error:
            return None, error

        return self._remember_capture(
            {
                "capture_id": bundle.capture_id,
                "captured_at_utc": bundle.captured_at,
                "game_id": int(game_id),
                "window_id": bundle.window_id,
            }
        ), None

    def _remember_capture(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """Keep a selected bundle's capture identity under its window id.

        Returns the bundle so the two selection sites can keep reading as one
        expression. Last write wins and cannot differ: ``select_capture_bundle``
        accepts exactly one scheduled capture per window and recomputes its
        id, so a second selection of the same window is the same answer.
        """
        self._capture_identity[str(bundle["window_id"])] = {
            "capture_id": str(bundle["capture_id"]),
            "captured_at_utc": str(bundle["captured_at_utc"]),
        }

        return bundle

    def _verify_captured(
        self, game_ids: Sequence[int], *, window_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Prove the capture the pricing path requires is actually there.

        The same selection pricing performs, performed here. A game that is
        on the schedule and reaches this point without an eligible bundle is
        the failure this controller exists to make loud -- left alone it
        would reach serving as an empty slate and publish nothing with a
        green job.

        This is the caller for a capture the run has just taken, and it is
        deliberately all-or-nothing: the collector was invoked moments ago
        for this group and every window in it must be there.
        """
        verified: list[dict[str, Any]] = []

        for game_id in game_ids:
            bundle, error = self._select_capture(int(game_id))

            if error is not None:
                raise ControllerRefused(
                    OUTCOME_REQUIRED_SNAPSHOT_MISSING,
                    f"game {game_id} is on the {self.slate_date} schedule and "
                    f"its T-{self.offset_minutes} window "
                    f"({', '.join(window_ids)}) is due, but no eligible "
                    f"scheduled capture can be selected for it: {error}",
                ) from error

            verified.append(bundle)

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

    def _assert_serving_window_gates(self, stage_dir: Path) -> None:
        """Refuse this window when a serving-window validation check fails.

        Invoked rather than reimplemented, for the reason every other stage
        here is: a second answer to "is this capture what the serving
        protocol requires" could disagree with the first, and the one that
        decides is the module that owns the checks.

        The refusal is a hard one and is placed before the slate is served,
        so a window that cannot satisfy them produces no priced artifact at
        all rather than one nothing is allowed to publish.
        """
        status = stage_dir / "serving_window_gates.json"

        command = [
            sys.executable,
            str(WINDOW_GATE_ENTRY_POINT),
            "--slate-date",
            self.slate_date,
            "--data-root",
            str(self.data_root),
            "--status-path",
            str(status),
        ]

        if self.registry_root is not None:
            command += ["--registry-root", str(self.registry_root)]

        if self.summary_path is not None:
            command += ["--summary-path", str(self.summary_path)]

        self._stage(
            str(WINDOW_GATE_ENTRY_POINT),
            command,
            outcome=OUTCOME_SERVING_WINDOW_GATES_REFUSED,
        )

        self.window_gates.append(
            {
                "slate_date": self.slate_date,
                "status_path": str(status),
                "window_offset_minutes": self.offset_minutes,
            }
        )

    def _serve_and_publish(
        self,
        stage_dir: Path,
        *,
        windows: Sequence[dict[str, Any]],
        readiness: dict[str, Any],
    ) -> None:
        """Serve the slate once and publish it once, for ``windows``.

        ``windows`` are the schedule rows this publication is being made for
        -- the group just captured, or the set recovery verified -- and they
        are required rather than defaulted. Every path to a production
        release runs through this method, so making the caller say which
        windows it is publishing for is what makes the post-tip guard below a
        property of the controller rather than of whoever remembers to call
        it: a new caller cannot reach a publication without answering the
        question.
        """
        self.pre_tip_margins += assert_no_window_has_tipped(
            windows, now=self.clock()
        )

        # The two validation checks a morning fit cannot answer, asked here
        # where the capture they are about exists. They were promotion
        # prerequisites, which made the daily fit permanently unpromotable
        # because they ask about an evening capture of a morning fit; the
        # invariant they carry is enforced at this window instead, recomputed
        # against this window's own capture rather than read off a record
        # written for an earlier slate. Before the slate is served, so a
        # window that fails them produces no priced artifact to publish.
        self._assert_serving_window_gates(stage_dir)

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

        self._persist_serving_provenance(
            served,
            offset_minutes=self.offset_minutes,
            window_ids=[
                str(window.get("window_id") or "") for window in windows
            ],
        )

        self.serving.append(
            {
                "outcome": outcome,
                "priced_market_artifact_sha256": str(
                    (served.get("priced_market_artifact") or {}).get("sha256")
                    or ""
                ),
                "receipt_path": str(serving_receipt),
                "serving_provenance_path": str(self.serving_provenance_path),
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

            # Still recorded as served. Byte-identical priced output is the
            # evidence that the release already published carries exactly
            # these windows; skipping the ledger row here would leave them
            # looking unpublished and send the next run round again.
            self._record_served_windows(
                publication_outcome=PUBLICATION_OUTCOME_SKIPPED,
                priced_digest=priced_digest,
            )

            # The canary is current and live may not be. A canary skip says
            # nothing about the live pointer: the canary can publish while the
            # live refresh refuses, which is the state this repository is in
            # until both approval variables are set, so the refresh is asked
            # again from the receipt the earlier publication left behind. It
            # skips itself when it has already succeeded on these bytes.
            retained = self._feed_receipts.get(self.offset_minutes)

            if retained is not None:
                self._refresh_live(
                    stage_dir,
                    windows,
                    offset=self.offset_minutes,
                    canary_destination=T20_FEED_DESTINATION,
                    feed_receipt=retained,
                    serving_receipt=self._serving_receipts[
                        self.offset_minutes
                    ],
                    priced_digest=priced_digest,
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

        self._record_served_windows(
            publication_outcome=PUBLICATION_OUTCOME_PUBLISHED,
            priced_digest=priced_digest,
            release_id=self.feed[-1]["release_id"],
            page_sha256=self.page[-1]["page_sha256"],
        )

        self._published_priced_digest = priced_digest or None

        self._refresh_live(
            stage_dir,
            windows,
            offset=self.offset_minutes,
            canary_destination=T20_FEED_DESTINATION,
            feed_receipt=feed_receipt,
            serving_receipt=serving_receipt,
            priced_digest=priced_digest,
        )

    def _record_served_windows(
        self,
        *,
        publication_outcome: str,
        priced_digest: str,
        release_id: str = "",
        page_sha256: str = "",
    ) -> None:
        """Write down which T-20 windows the release just published carries.

        Every T-20 window the capture log holds for this slate, because that
        is exactly the set serving consumed: ``ProspectiveSnapshotClient``
        enumerates the scheduled windows at this offset in that log and
        builds a bundle for every one of them with no branch that skips one,
        ``15_price_markets.py`` rewrites the whole priced artifact, and the
        feed publishes every row in it.

        Minus the windows this segment has proved serving cannot select.
        That subtraction used to be unnecessary and is now load-bearing: a
        window the capture log held that serving could not have used would
        once have refused the run before reaching this line, and now the
        run continues past it on purpose. Claiming it here would assert
        that a release carries a window no release can carry, and -- worse
        -- would retire it from the unserved set, so the permanently
        missing window would read as published and the segment would exit
        green. Omitting it is the conservative direction: a window left out
        of this row stays unserved and keeps failing the segment.

        Appended and fsynced, one row, never rewritten. The claim it makes
        is about what a release carried, so a row may not be withdrawn by a
        later run any more than a capture may.
        """
        refused = {
            str(record["window_id"]) for record in self.unrecoverable
        }

        windows = sorted(
            window_id
            for window_id in completed_window_ids(self.capture_run_path)
            if window_id.endswith(f":T-{self.offset_minutes}m")
            and window_id not in refused
        )

        row = {
            "offset_minutes": self.offset_minutes,
            "page_sha256": str(page_sha256),
            "priced_market_artifact_sha256": str(priced_digest or ""),
            "production_sha": self.production_sha,
            "publication_mode": PUBLICATION_MODE,
            "publication_outcome": str(publication_outcome),
            "record_type": SERVED_RECORD_TYPE,
            "recorded_at_utc": self.clock().isoformat(),
            "release_id": str(release_id),
            "slate_date": self.slate_date,
            "window_ids": windows,
        }

        self.served_path.parent.mkdir(parents=True, exist_ok=True)

        with self.served_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

            handle.flush()

            os.fsync(handle.fileno())

        self.publications.append(row)

    def _persist_serving_provenance(
        self,
        served: dict[str, Any],
        *,
        offset_minutes: int,
        window_ids: Sequence[str],
    ) -> None:
        """Keep the receipt for the bytes this serve just wrote.

        Called the moment a serve is known to have succeeded and before the
        release it produced is published, because by then
        ``processed/priced_markets/<slate>.parquet`` already holds the new
        bytes and the only copy of their provenance is in a directory the job
        will delete. See :data:`SERVING_PROVENANCE_RELATIVE`.

        The receipt is stored whole and unedited. The grader re-verifies it
        against the parquet it is about to read, so a row that this ledger
        paraphrased or summarised would be a row the grader could not use;
        the added fields are only the ones that say which serve wrote it, and
        none of them collide with a name ``build_receipt`` emits.

        A failure here is a refusal, not a warning. Publishing a release
        whose provenance was not written down is exactly the state that made
        2026-10-23 ungradable.
        """
        row = {
            **served,
            "offset_minutes": int(offset_minutes),
            "record_type": SERVING_PROVENANCE_RECORD_TYPE,
            "recorded_at_utc": self.clock().isoformat(),
            "served_window_ids": sorted(str(value) for value in window_ids),
        }

        try:
            self.serving_provenance_path.parent.mkdir(parents=True, exist_ok=True)

            with self.serving_provenance_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

                handle.flush()

                os.fsync(handle.fileno())
        except OSError as error:
            raise ControllerRefused(
                OUTCOME_FAILED,
                f"the serving receipt for {self.slate_date} could not be "
                f"recorded at {self.serving_provenance_path} ({error}), so "
                "the release about to be published would describe priced "
                "markets nothing provenances and the slate could not be "
                "graded",
            ) from error

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
            # What a run that captured and did not publish leaves behind,
            # stated after the work rather than copied from the plan: the
            # whole point is whether anything is *still* in that state.
            "captured_unserved_windows": [
                str(row["window_id"])
                for row in self._unserved_rows(plan["rows"])
            ],
            "recovered_windows": self.recovered,
            # Named apart from captured_unserved_windows above, which is the
            # whole outstanding set. These are the subset of it that no
            # later run will clear, with the reason selection gave for each.
            "unrecoverable_windows": self.unrecoverable,
            # Both halves of the post-tip guard, on the receipt whatever the
            # outcome: the windows it refused, from either path, and the
            # margin every window it cleared actually had.
            "post_tip_refused_windows": [
                record
                for record in [*self.unrecoverable, *self.advisory]
                if str(record.get("outcome"))
                in (OUTCOME_POST_TIP_PUBLICATION, ADVISORY_OUTCOME_POST_TIP)
            ],
            "pre_tip_margins": self.pre_tip_margins,
            "recovery_faults": self.recovery_faults,
            "served_window_records": self.publications,
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
            "serving_window_gate_outcomes": self.window_gates,
            "final_outcome": outcome,
            "finish_time_utc": finished_at.isoformat(),
            "games_discovered": plan["games_discovered"],
            "live_feed_destination": LIVE_FEED_DESTINATION,
            "live_page_destination": LIVE_PAGE_DESTINATION,
            "live_refreshes": self.live,
            "live_refreshes_with_faults": [
                record
                for record in self.live
                if str(record.get("outcome"))
                not in LIVE_OUTCOMES_WITHOUT_FAULT
            ],
            "live_release_id": next(
                (
                    str(record.get("release_id") or "")
                    for record in reversed(self.live)
                    if str(record.get("outcome")) in LIVE_OUTCOMES_WITHOUT_FAULT
                    and record.get("release_id")
                ),
                "",
            ),
            "morning_readiness": readiness,
            "offset_minutes": self.offset_minutes,
            "page_publication_outcomes": self.page,
            "production_sha": self.production_sha,
            "progressive_live_offsets": list(PROGRESSIVE_LIVE_OFFSETS_MINUTES),
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
        (
            "recovered without recapture",
            ", ".join(
                window_id
                for record in receipt.get("recovered_windows") or []
                for window_id in record.get("window_ids") or []
            )
            or "none",
        ),
        (
            "captured but unserved",
            ", ".join(receipt.get("captured_unserved_windows") or [])
            or "none",
        ),
        (
            "unrecoverable",
            ", ".join(
                str(record.get("window_id"))
                for record in receipt.get("unrecoverable_windows") or []
            )
            or "none",
        ),
        (
            "refused for being post-tip",
            ", ".join(
                str(record.get("window_id"))
                for record in receipt.get("post_tip_refused_windows") or []
            )
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
        ("live release", receipt.get("live_release_id") or "none"),
        (
            "live refreshes",
            ", ".join(
                f"T-{record.get('offset_minutes')} "
                f"{record.get('outcome')}"
                for record in receipt.get("live_refreshes") or []
            )
            or "none",
        ),
        (
            "live failures",
            len(receipt.get("live_refreshes_with_faults") or []),
        ),
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

    unrecoverable = receipt.get("unrecoverable_windows") or []

    if unrecoverable:
        lines += [
            "",
            "### Windows no later run will serve",
            "",
            (
                "These captures exist and selection refuses them. The T-20 "
                "window each one belongs to is in the past, so no run can "
                "take another capture of it and no run can publish this "
                "one. They are listed so the operator stops waiting for a "
                "recovery that cannot happen, and the segment stays red "
                "while they are outstanding."
            ),
            "",
            "| window | why |",
            "| --- | --- |",
        ]

        lines += [
            "| {window} | {reason} |".format(
                window=record.get("window_id"),
                reason=str(record.get("reason") or "").replace("|", "/")[:300],
            )
            for record in unrecoverable
        ]

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
                "the segment, with one exception: a window refused for "
                "being past its own tip fails it, because a release for a "
                "game already under way is not a pregame price wherever it "
                "is published."
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

    refreshes = receipt.get("live_refreshes") or []

    if refreshes:
        lines += [
            "",
            "### The live release",
            "",
            (
                "One live destination, replaced by each window that passes "
                "canary: T-180 establishes it, T-90, T-45 and T-20 replace "
                "it, and T-20 is the final pregame version. Every refresh "
                "publishes the bytes its own canary validated, matched on "
                "the rows digest. A refresh that cannot be made leaves the "
                "most recent release that did validate serving, and degrades "
                "this report without failing the segment."
            ),
            "",
            "| window | outcome | release | replaced | why |",
            "| --- | --- | --- | --- | --- |",
        ]

        lines += [
            "| T-{offset} | {outcome} | {release} | {previous} | {reason} |".format(
                offset=record.get("offset_minutes"),
                outcome=record.get("outcome"),
                release=record.get("release_id") or "-",
                previous=record.get("previous_release_id") or "-",
                reason=str(record.get("reason") or "").replace("|", "/")[:300],
            )
            for record in refreshes
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
    # still install the package. A captured window no release carries counts
    # for the same reason, and it is the case the probe most has to ask for:
    # the full job is the only thing with the environment to serve and publish
    # from a capture that already exists. Reported under the field name the
    # workflow already reads, because a scheduled run executes the default
    # branch's copy of the workflow and renaming the output would silently
    # stop gating.
    work = (
        list(plan["pending_windows"])
        + list(plan["advisory_pending_windows"])
        + list(plan["captured_unserved_windows"])
    )

    return {
        "advisory_offsets": list(controller.advisory_offsets),
        "advisory_pending_windows": plan["advisory_pending_windows"],
        "advisory_windows_in_segment": plan["advisory_windows_in_segment"],
        "already_completed_windows": plan["already_completed_windows"],
        "captured_unserved_windows": plan["captured_unserved_windows"],
        "final_outcome": (
            OUTCOME_NO_WINDOWS if not work else "T20_WORK_IN_SEGMENT"
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
            + (
                f", and {len(plan['captured_unserved_windows'])} captured "
                f"window(s) of {slate_date} are not carried by any "
                "release and can be served from the captures that already "
                "exist"
                if plan["captured_unserved_windows"]
                else ""
            )
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

    # A captured window no release carries is work in this segment too, and
    # it is work the probe has to ask for: the full job is what has the
    # environment to serve and publish from an existing capture, and the
    # probe is the only thing that can start it. Expressed here rather than
    # in the workflow so the schedule the production branch runs is unchanged.
    unserved = receipt.get("captured_unserved_windows") or []

    advisory = receipt.get("advisory_pending_windows") or []

    _write(
        args.github_output,
        "\n".join(
            [
                # The gate the workflow already reads. True for advisory work
                # and for a captured-but-unserved window as well as for T-20
                # work, because the step it gates is the package install and
                # all three need it.
                "t20_work="
                + ("true" if pending or unserved or advisory else "false"),
                f"segment_start_utc={receipt['segment_start_utc']}",
                f"segment_end_utc={receipt['segment_end_utc']}",
                f"slate_date={receipt['slate_date']}",
                f"pending_windows={len(pending)}",
                f"captured_unserved_windows={len(unserved)}",
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

        unserved = list(getattr(error, "captured_unserved_windows", []) or [])

        failure = {
            "captured_unserved_windows": unserved,
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
            f"## NBA T-20 production controller\n\n{outcome}: {error}\n"
            + (
                "\ncaptured but unserved: "
                + ", ".join(unserved)
                + ". These captures exist and no release carries them; a "
                "later run of this controller will serve and publish from "
                "them without recapturing.\n"
                if unserved
                else ""
            ),
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

    # The probe reports work rather than performing it, so "there is work"
    # is its success. For a segment run, OUTCOMES_THAT_SUCCEED is the list
    # of endings that are actually an ending -- it has always said so and
    # nothing consulted it, which is how a segment that captured three games
    # and published none of them exited 0.
    if (
        not args.probe_only
        and receipt.get("final_outcome") not in OUTCOMES_THAT_SUCCEED
    ):
        return EXIT_FAILED

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
