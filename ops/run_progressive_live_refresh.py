#!/usr/bin/env python3
"""Hold every game of the live slate at its own latest validated window.

WHAT WAS MISSING
----------------

Every piece of live publication existed and nothing joined them up. The two
live destinations are fully addressed in ``ops/wizardofodds_destinations.json``
and have been for months. The publisher takes ``--mode live``, refuses it on
the committed switch, requires a prior canary attestation of the same bytes
and requires a rollback target before it will move anything. The page builder
has a live entry in ``MODE_DESTINATIONS`` naming the served root. What was
absent was the thing none of them can supply on its own: a statement of
*which* capture window each game is live at, *which* window may replace it,
and what happens to the pointer when a later window fails.

That is this module, and it is deliberately thin. Every rule a live
publication rests on is the frozen publisher's own, called by name: the switch
gate, the destination confinement, the release construction, the
canary-attestation match, the rollback-target requirement, the readback, the
atomic pointer rotation, the retention sweep and the audit append. It then
calls ``run_wizardofodds_publication.verify_promoted_release`` -- again by
name -- so the post-promotion readback a live refresh is held to is the one
the canary feed stage is held to.

What it adds is the decisions the publisher has no opinion about.

WHY THE ONE-POINTER CHAIN WAS WRONG
-----------------------------------

The first version of this file published the slate through a single
forward-only pointer: one "closest window offset reached so far" for the whole
evening, and any later refresh at a larger offset declined as going backwards.

That is correct when every game tips at once and degenerate otherwise,
because a window belongs to a *game* and its offset is measured from that
game's own tip. On the eight-game 2026-10-24 slate, tips from 6:00 PM to
10:00 PM ET, the first game reached T-20 at 5:40 PM -- and from that moment
the rule declined the T-180, T-90 and T-45 windows of all seven games behind
it. Fourteen of twenty-four window groups declined; most of the slate with no
live card at all. Every decline was the rule working as written, every
receipt was correct and the segment was green, which is what made it
invisible.

PER-GAME COMPOSITION
--------------------

The live slate is a composite that holds, simultaneously, each game at its own
latest validated window. Game A at T-20, game B at T-90 and game C at T-180
is one live release carrying exactly those three cards at exactly those three
windows. For each game independently: no card before its first capture, then
its T-180, then its T-90 in place of that, then its T-45, then its T-20 as the
final pregame version.

Two invariants, and both are properties of how the composition is computed
rather than things checked afterwards.

*A game is never downgraded.* The offset a game is composed at is the smaller
of the offset recorded for it and the offset being refreshed, and nothing
anywhere takes the larger. A T-180 refresh that arrives after a game's T-20 --
which is the ordinary case on a staggered slate and not a fault -- leaves that
game at T-20 and goes on to advance the games the refresh is actually for.

*Advancing one game leaves the others byte-identical.* Every other game's
component is selected out of its own stored source release and is a function
of that release's bytes alone, so games B..H come out of the composition
unchanged whatever A does.

``ops/live_slate_composite.py`` is where the composition lives. This file is
the entry point: it decides which games may advance and then publishes what
that module assembled.

SELECTION, NEVER COMPUTATION
----------------------------

Nothing is re-derived, re-priced, re-routed or recomputed. Each game's rows
are copied byte for byte out of a release its own canary already validated
and read back, and the composite is those rows concatenated in the
publisher's own logical-key order with the distribution records the same
bytes carry.

So the digest identity the chain has always rested on survives composition.
A composite whose games all come from one source release produces a rows
payload byte-identical to that release's own -- same contract version, same
rows, same distributions, same slate -- and therefore the identical
``rows_sha256``. In that case the three statements of one digest still agree
exactly as before:

* the canary feed publication receipt this run's canary stage wrote;
* ``canary_attestation.json`` read back from the canary destination itself;
* the release built for the live destination.

A composite spanning more than one window has a digest of its own, and what
replaces the single-release comparison is strictly stronger rather than
weaker: ``publish_incumbent_to_wizardofodds.assert_canary_attested`` is
applied once per source release, every composite row is looked up in the
recomputed bytes of the release its game names, and
``COMPOSITE_UNATTESTED_ROWS``, ``LIVE_SOURCE_MISMATCH_COUNT`` and
``COMPOSITE_UNATTESTED_DISTRIBUTIONS`` all have to be zero. One row that
cannot be traced to canary-attested bytes refuses the whole publication.

THE PRIOR VALID COMPOSITE SURVIVES EVERY FAILURE
------------------------------------------------

Structurally, not by compensation. The order of operations is switch,
destinations, canary receipt, canary attestation, the per-game advancement
read, the attested sources, the composition, the attestation audit, the
release, the transport, the rollback target, write, readback, *promote*.
Every refusal above happens before ``promote``, and ``promote`` is the only
thing that touches ``current.json``. So a refresh that fails for any reason --
a game's source missing, a digest that does not match, an unattested row, a
game that has already tipped, a transport that will not talk, an HTTPS
readback that disagrees -- leaves the live pointer naming the last composite
that did validate, and the composite before that is still at the destination
because ``previous.json`` names it and ``prune_releases`` is pointer-driven
rather than date-driven.

This module adds no second retention scheme. ``previous.json`` and
``rollback`` are the mechanism; what is new is only that the thing being
retained now has a window per game attached to it.

POST-TIP IS PER GAME
--------------------

No game is newly published after its own tip. A game whose card was validly
published before its tip keeps that card: the guard refuses new publication
for that game and does not retract the slate, because retracting it would
take down seven games to punish one. So a refresh is for the games whose tips
are still ahead, the tipped ones are recorded as refused by name, and a
refresh in which no game is left to advance publishes nothing at all.

WHAT IT CANNOT DO
-----------------

It cannot fit, refit, recalibrate, predict, price, grade, promote a candidate
or publish a canary. It computes no probability, price, edge, expected value,
quantile, line, distribution or route: every number it puts live was produced
by the serving step and has already been published, read back and attested one
directory below. ``candidate_authority`` is read off the switch by the
publisher and has to be ``shadow_only`` or the publication refuses.

It also cannot authorise itself. Live publication needs ``modes.live.state``
set to ``ENABLED`` in a reviewed commit, the ``NBA_PROP_WOO_PUBLISH``
environment variable set to that same state and the
``NBA_PROP_WOO_PUBLISH_APPROVAL`` environment variable set to the switch's own
approval token. The commit is the first of those and the only one this
repository can hold; the other two live in the ``wizardofodds-production``
GitHub environment. When they are absent or disagree this module refuses by
name, naming both of them and the values they must carry, and it never falls
back to canary: there is no mode argument here, no second destination and no
retry at a lower privilege, because a run that published a preview while
reporting a live success would be the one failure no later check could see.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ops"))

# Imported as modules and used by name. Every rule a live publication rests on
# -- which switch states permit it, which destination is the served root,
# which rows are publishable, how a release is named, what a canary has to
# have proved, what a rollback target is, how a pointer is rotated, what
# retention keeps -- is read out of the frozen publisher rather than restated
# here, and the readback a promoted release is held to is the publication
# orchestrator's own.
import live_slate_composite as composite  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import run_wizardofodds_publication as publication  # noqa: E402

#: The windows of the progressive chain, in the order they occur. The offsets
#: descend, so the chain reads left to right and the ordering rule reads as
#: "the offset may only get smaller".
#:
#: T-20 is in this tuple and is not an advisory window: it is the production
#: window, the final pregame version, and the one the whole capture schedule
#: already existed for. The three before it are the advisory windows
#: ``ops/run_advisory_window_serving.py`` serves. What the four have in common
#: is the only property this module needs -- each has a canary that validates
#: before live is touched -- so they are one tuple here rather than two.
PROGRESSIVE_OFFSETS_MINUTES: tuple[int, ...] = (180, 90, 45, 20)

#: The last window before tip, and so the last release a reader is shown.
#: Named rather than left implicit as the smallest element above, so the
#: receipt and the durable record can say in words which release is final.
FINAL_PREGAME_OFFSET_MINUTES = 20

#: The single live feed destination, replaced once per window.
#:
#: One destination and not one per window, which is the whole design. A reader
#: opens one URL; what changes through the evening is which release
#: ``current.json`` there names. Per-offset live destinations would publish
#: four pages nobody asked for and leave the page a reader actually opens
#: stale from T-180 onwards.
LIVE_FEED_DESTINATION = "wizardofodds_nba_live"

#: The single live page destination, built from the feed release above and
#: written at the served root. Named here so this module can refuse a page
#: destination that is not it; the page itself is published by
#: ``ops/run_wizardofodds_publication.py --stage page --mode live``, which
#: already resolves this name for live mode and which this module does not
#: need to change to reach.
LIVE_PAGE_DESTINATION = "wizardofodds_nba_player_props_page_live"

#: The canary destination each window must have validated at before its
#: release may go live, by name.
#:
#: Stated here rather than imported from the T-20 controller, for the reason
#: that controller states its own advisory offsets rather than importing them:
#: this entry point is independently invocable, and a table that lived only in
#: the caller would be a table a direct invocation could skip. A test pins
#: these against the controller's own.
CANARY_FEED_DESTINATIONS: dict[int, str] = {
    180: "wizardofodds_nba_advisory_t180m_canary",
    90: "wizardofodds_nba_advisory_t90m_canary",
    45: "wizardofodds_nba_advisory_t45m_canary",
    20: "wizardofodds_nba_canary",
}

#: The GitHub environment the two approval variables live in, named in the
#: refusal below so the message says where to set them rather than only that
#: they are missing. It is the environment the T-20 workflow's job already
#: declares, so they resolve for that job and for nothing else here.
PRODUCTION_ENVIRONMENT = "wizardofodds-production"

#: Where the durable, append-only record of live publication is kept.
#:
#: Its own tree, and the choice is load-bearing rather than tidy. The witness
#: ledgers beside it record what happened and decide nothing; this file
#: decides something. It is the only record of *which capture window* is live,
#: because the published rows cannot say so -- ``PUBLISHED_FIELDS`` is pinned
#: and carries no capture offset -- and ``current.json`` names a release id
#: whose prefix is the live stream's rather than the window's. Mixing a
#: decision input into a witness ledger would turn a file nothing reads into a
#: file the whole chain depends on.
#:
#: One line per attempt, including every refusal, so "what did live do
#: tonight" is answerable from this file alone.
LIVE_EVIDENCE_RELATIVE = Path("processed") / "live_progressive_refresh"

#: The live pointer served nothing for this slate and now names this window's
#: release.
OUTCOME_ESTABLISHED = "LIVE_ESTABLISHED"

#: The live pointer named an earlier window's release and now names this one.
OUTCOME_REPLACED = "LIVE_REPLACED"

#: The live pointer already named this release, so nothing was written at all.
#: The idempotent case, and deliberately not a republication: rewriting
#: byte-identical release files would change nothing a reader sees and would
#: put a second audit entry against one release.
OUTCOME_ALREADY_CURRENT = "LIVE_ALREADY_CURRENT"

#: The pointer moved and the run failed afterwards -- in the retention sweep,
#: in the audit append, or in the post-promotion readback.
#:
#: Its own outcome, reported as a failure because the live destination is in a
#: state somebody has to look at, and recorded as live-bearing so the next
#: refresh can still establish which window is serving and go forward from it
#: rather than refusing for the rest of the evening. It is distinguishable
#: from a refusal with certainty rather than by guesswork: the pointer is
#: re-read after the failure and this outcome is only recorded when it names
#: the rows digest this refresh was publishing, which nothing else could have
#: put there.
OUTCOME_PROMOTED_THEN_INCOMPLETE = "LIVE_PROMOTED_THEN_INCOMPLETE"

#: Every game this refresh is for is already at a window closer to its own
#: tip, so nothing was written and the pointer is unchanged.
#:
#: Its own outcome, and a success, because it is the no-downgrade invariant
#: working rather than anything failing.
#:
#: Narrowed from the slate to the games, which is the whole of what the
#: composite changed. It used to fire whenever *any* game of the slate had
#: reached a smaller offset, which on a staggered slate meant almost always:
#: the 6:00 game's T-20 capture is taken at 5:40 and the 10:00 game's T-180
#: capture at 7:00, an hour and twenty minutes *later* in the evening and
#: three hours *earlier* in its own game's window, so the advisory refresh
#: that followed was declined and the later game never got a card. Now a
#: refresh is declined only when every game it is for has already validly
#: reached a window closer to its own tip, which is a real repeat and not a
#: collision between two games' schedules.
OUTCOME_ALREADY_CLOSER_TO_TIP = "LIVE_ALREADY_CLOSER_TO_TIP"

#: Nothing was written and the live pointer is unchanged.
OUTCOME_REFUSED = "LIVE_REFUSED"

#: The outcomes after which a live release is serving. Read by the ordering
#: decision: the most recent of these in the durable record says which window
#: live is at, and the pointer at the destination has to agree with it.
OUTCOMES_THAT_ARE_LIVE: frozenset[str] = frozenset(
    {
        OUTCOME_ESTABLISHED,
        OUTCOME_REPLACED,
        OUTCOME_ALREADY_CURRENT,
        OUTCOME_PROMOTED_THEN_INCOMPLETE,
    }
)

#: The outcomes this entry point exits zero on.
OUTCOMES_THAT_SUCCEED: frozenset[str] = frozenset(
    {
        OUTCOME_ESTABLISHED,
        OUTCOME_REPLACED,
        OUTCOME_ALREADY_CURRENT,
        OUTCOME_ALREADY_CLOSER_TO_TIP,
    }
)

EXIT_OK = 0
EXIT_REFUSED = 1


class LiveRefreshRefused(RuntimeError):
    """A fail-closed refusal: the live pointer was not touched."""


class LiveAlreadyCloserToTip(LiveRefreshRefused):
    """The ordering rule declined this window. Nothing failed.

    A subclass of the refusal because everything a refusal guarantees is
    guaranteed here too -- nothing written, no canary published in live's
    place, the pointer left naming the release it already named -- and the
    one thing that differs is the verdict.
    """


class LiveRefreshIncomplete(RuntimeError):
    """The pointer moved and the run did not finish. Not a clean refusal."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def parse_utc(raw: str, *, what: str) -> datetime:
    text = str(raw)

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        value = datetime.fromisoformat(text)
    except ValueError as error:
        raise LiveRefreshRefused(
            f"{what} is not an ISO timestamp: {raw!r}"
        ) from error

    if value.tzinfo is None:
        raise LiveRefreshRefused(f"{what} must carry a timezone: {raw!r}")

    return value.astimezone(UTC)


# ----------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------


def assert_offset_is_progressive(offset_minutes: int) -> int:
    """The window this refresh is for, or the refusal naming why not.

    Only the four windows of the chain. A research offset the capture schedule
    defines and nothing publishes -- T-1440, T-480, T-5 -- has no canary
    destination, so it has no attestation and nothing could prove the bytes it
    would put live had ever been validated.
    """
    offset = int(offset_minutes)

    if offset not in PROGRESSIVE_OFFSETS_MINUTES:
        raise LiveRefreshRefused(
            f"--snapshot-offset-minutes {offset} is not one of the "
            f"progressive live windows {list(PROGRESSIVE_OFFSETS_MINUTES)}, "
            "so no canary destination validates it and nothing could prove "
            "the bytes it would publish were ever canaried"
        )

    return offset


def assert_destinations_are_the_live_ones(
    *, offset_minutes: int, canary: str, feed: str, page: str
) -> None:
    """The three destinations this refresh names are the right three.

    Checked rather than trusted because each of the three mistakes it prevents
    is one typo away and none of them would announce itself. A canary
    destination belonging to another window would authorise these bytes with
    an attestation of a different window's prices. A feed destination that is
    not the live one would put a live-intended release somewhere no reader
    fetches while reporting success. A page destination that is not the live
    one would leave the page a reader opens showing the window before.
    """
    expected_canary = CANARY_FEED_DESTINATIONS.get(int(offset_minutes))

    if expected_canary is None:
        raise LiveRefreshRefused(
            f"no canary destination is configured for T-{offset_minutes}, so "
            "there is nothing that could have validated its release"
        )

    for label, actual, expected in (
        ("canary", canary, expected_canary),
        ("live feed", feed, LIVE_FEED_DESTINATION),
        ("live page", page, LIVE_PAGE_DESTINATION),
    ):
        if actual != expected:
            raise LiveRefreshRefused(
                f"the T-{offset_minutes} live refresh names the {label} "
                f"destination {actual!r} and its own is {expected!r}, so it "
                "would read or write a stream this window does not own"
            )


def game_of_window(window_id: str) -> str:
    """The game a window id belongs to.

    A window id is ``GAME_ID:T-OFFSETm`` and this is the only place that shape
    is taken apart. It is the join between the caller's schedule read and the
    per-game composition: the tips arrive keyed by window and everything after
    this point is keyed by game.
    """
    game_id, separator, _ = str(window_id).partition(":")

    if not separator or not game_id:
        raise LiveRefreshRefused(
            f"the window id {window_id!r} does not name a game before its "
            "offset, so the game whose tip it is about cannot be told from "
            "the window it is about"
        )

    return game_id


def partition_windows_by_tip(
    tips: dict[str, datetime], *, now: datetime, offset_minutes: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split this refresh's windows into the pregame ones and the tipped ones.

    The guard the canary path does not have and does not need. A preview
    priced after tip is a preview of a stale market in a directory nobody
    reads; the same bytes at the served root are a pregame price on a game
    that has already started, which is the one claim this integration must
    never make.

    It is not hypothetical. On 2026-10-23 the T-20 group holding game
    21717941 -- CHI at MIL -- was captured, served and published two hours and
    forty minutes after that game tipped, because runner pickup was late and
    the capture grace is about when a *snapshot* may be taken rather than
    about when a *price* may be published. Nothing anywhere refused it. The
    canary record of that evening is correct and unremarkable; the same
    release at the served root would have been a pregame page for a game in
    its third quarter.

    Scoped per game, which is the whole of what changed. The old guard refused
    the refresh when *any* window it was for had tipped, and on a composite
    that is a retraction: one late game would have taken seven games' valid
    cards off the page to prevent its own. So the tipped games are partitioned
    out and recorded, the games still ahead of their tips advance, and a game
    whose card was published before its own tip keeps it -- the composite
    carries it because it was validly published, not because this refresh is
    claiming anything new about it.

    The subject is still the windows this refresh is *for*, not every window
    in the release. A release is cumulative -- one serve at an offset prices
    every game with an eligible capture at that offset -- so by the time the
    last T-20 group of a twelve-game slate is published, the first game tipped
    two hours earlier, and its rows are correctly in the release because they
    were priced before their own tip.
    """
    if not tips:
        raise LiveRefreshRefused(
            "no --window-tip-utc was supplied, so there is no window this "
            "refresh could be shown to be ahead of. The tip of every window a "
            "live release is published for is required and not defaulted: a "
            "default would make the post-tip guard a property of this file "
            "rather than of the caller's own schedule read"
        )

    suffix = f":T-{int(offset_minutes)}m"

    wrong = sorted(
        window_id for window_id in tips if not window_id.endswith(suffix)
    )

    if wrong:
        raise LiveRefreshRefused(
            f"this refresh is for T-{offset_minutes} and was given the tip of "
            + ", ".join(wrong)
            + f", which are not {suffix} windows, so the guard would be "
            "checking a window this release is not being refreshed for"
        )

    pregame: list[dict[str, Any]] = []

    tipped: list[dict[str, Any]] = []

    for window_id, tip in sorted(tips.items()):
        entry = {
            "game_id": game_of_window(window_id),
            "minutes_before_tip": round(
                (tip - now).total_seconds() / 60.0, 3
            ),
            "tip_utc": tip.isoformat(),
            "window_id": window_id,
        }

        (tipped if now >= tip else pregame).append(entry)

    return pregame, tipped


def post_tip_refusal(
    tipped: list[dict[str, Any]], *, now: datetime
) -> LiveRefreshRefused:
    """The refusal for a refresh in which every game has already started.

    Nothing is left to publish, so this is a refusal rather than a partial
    composite: the games named here keep whatever they already have and this
    refresh claims nothing new about any of them.
    """
    return LiveRefreshRefused(
        "the game(s) "
        + ", ".join(
            f"{entry['window_id']} (tipped {entry['tip_utc']}, "
            f"{abs(round(entry['minutes_before_tip']))} minutes ago)"
            for entry in tipped
        )
        + f" have already started and it is now {now.isoformat()}. A "
        "window whose game has tipped is not a pregame price and does not "
        "go live; the release already live is left serving"
    )


# ----------------------------------------------------------------------
# the approval
# ----------------------------------------------------------------------


def assert_live_approval_is_carried(
    switch: publisher.PublishingSwitch,
) -> dict[str, Any]:
    """This process carries the approval the committed switch requires.

    The publisher refuses the same thing from inside
    :func:`publish_incumbent_to_wizardofodds.assert_mode_enabled`, and this
    asks it first and separately for two reasons.

    The message. The publisher's blockers say which condition failed; this
    says what to do about it, by naming both variables, the exact values they
    have to carry and the GitHub environment they belong to. An operator
    reading a red segment should not have to open the publisher to find out
    what is missing.

    And the shape of the failure. A live refresh that cannot be authorised
    must be a refusal and never a quiet downgrade, so the two states are
    distinguished in the message and neither of them reaches a destination.
    The first -- live DISABLED in the commit -- means the reviewed decision has
    not been taken. The second -- ENABLED in the commit with the environment
    silent or disagreeing -- means it has been taken and has not reached the
    runner, which is a misconfiguration of a decision rather than the absence
    of one, and is the louder of the two.
    """
    if switch.state != publisher.STATE_ENABLED:
        raise LiveRefreshRefused(
            f"live publishing is {switch.state} in {switch.source}, so the "
            "reviewed decision to publish has not been taken and no live "
            f"refresh is attempted: {switch.reason or 'no reason recorded'}"
        )

    if switch.blockers:
        raise LiveRefreshRefused(
            f"{switch.source} records live publishing as "
            f"{publisher.STATE_ENABLED} and this process does not carry its "
            "approval ("
            + "; ".join(switch.blockers)
            + f"). Both variables must be set in the {PRODUCTION_ENVIRONMENT} "
            "GitHub environment and read by the workflow step that runs this "
            f"chain: {publisher.PUBLISH_ENV_VAR} to the literal "
            f"{publisher.STATE_ENABLED!r}, character for character, and "
            f"{publisher.PUBLISH_APPROVAL_ENV_VAR} to this file's own "
            "modes.live.approval_token. Nothing is published to a canary in "
            "their place and nothing is published live"
        )

    return switch.payload()


# ----------------------------------------------------------------------
# what the canary validated
# ----------------------------------------------------------------------


def read_canary_feed_receipt(
    path: Path, *, slate_date: str, offset_minutes: int, destination: str
) -> dict[str, Any]:
    """The canary publication this refresh is the live half of.

    Read for one thing and checked for five. The one thing is the rows digest
    the canary published, which is the digest the live release has to carry.
    The five checks are that the receipt records a publication rather than a
    refusal or a no-op, that it is this slate's, that it is a canary's, that
    it was published to this window's own canary destination, and that it
    declares the publication contract a reader's page is built against.

    A canary stage that did not publish is not a licence to publish live from
    whatever release happened to be current. That is the ordering mistake the
    publication orchestrator already prevents at the page, read one stage
    further along.
    """
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LiveRefreshRefused(
            f"the canary feed publication receipt at {path} could not be read "
            f"({error}), so there is no evidence this window's release was "
            "canaried and nothing goes live"
        ) from error

    if not isinstance(receipt, dict):
        raise LiveRefreshRefused(
            f"{path} does not hold a canary feed publication receipt"
        )

    outcome = str(receipt.get("outcome") or "")

    if outcome != publication.OUTCOME_PUBLISHED:
        raise LiveRefreshRefused(
            f"{path} records the canary feed outcome {outcome or 'nothing'} "
            f"rather than {publication.OUTCOME_PUBLISHED}, so the "
            f"T-{offset_minutes} window has no validated canary release and "
            "nothing is replaced live"
        )

    recorded = str(receipt.get("slate_date") or "")

    if recorded != slate_date:
        raise LiveRefreshRefused(
            f"{path} is the canary receipt for {recorded or 'no slate'} and "
            f"this refresh would publish {slate_date}, so it does not "
            "provenance the rows that would go live"
        )

    mode = str(receipt.get("mode") or "")

    if mode != publisher.MODE_CANARY:
        raise LiveRefreshRefused(
            f"{path} records the mode {mode or 'nothing'} rather than "
            f"{publisher.MODE_CANARY}. What authorises a live refresh is a "
            "canary of these bytes, and a receipt of anything else is not one"
        )

    published_to = str((receipt.get("destination") or {}).get("name") or "")

    if published_to and published_to != destination:
        raise LiveRefreshRefused(
            f"{path} records a canary publication to {published_to!r} and "
            f"this is the T-{offset_minutes} refresh, whose canary is "
            f"{destination!r}. A canary of another window's prices does not "
            "validate this window's release"
        )

    contract = receipt.get("contract_version")

    if contract != publisher.PUBLICATION_CONTRACT_VERSION:
        raise LiveRefreshRefused(
            f"{path} records publication contract {contract!r} and live is "
            f"published under {publisher.PUBLICATION_CONTRACT_VERSION}, so "
            "the release it describes is not one a reader's page could render"
        )

    release = receipt.get("release") or {}

    rows_sha256 = str(release.get("rows_sha256") or "")

    release_id = str(release.get("release_id") or "")

    if not rows_sha256 or not release_id:
        raise LiveRefreshRefused(
            f"{path} names no release id and rows digest, so there is no "
            "canary release the live one could be required to be identical to"
        )

    return {
        "canary_destination": destination,
        "canary_release_id": release_id,
        "path": str(path),
        "priced_market_artifact_sha256": str(
            (receipt.get("provenance") or {}).get(
                "priced_market_artifact_sha256"
            )
            or ""
        ),
        "rows_sha256": rows_sha256,
        "slate_date": recorded,
    }


def read_attestation_from_the_canary(
    transport: Any, *, destination: str, offset_minutes: int
) -> dict[str, Any]:
    """``canary_attestation.json`` as the canary destination holds it now.

    Not as a caller remembers it. The attestation is the destination's own
    statement about what it is serving, ``ensure_canary_attestation`` rewrites
    it whenever that stops being true, and a copy handed in on a command line
    could have been written by a canary of a slate from before. Reading it
    back is also what makes "the canary validated these bytes" a fact about
    the canary host rather than about this run's temporary directory.
    """
    attestation = publisher.read_pointer(transport, publisher.ATTESTATION_NAME)

    if not attestation:
        raise LiveRefreshRefused(
            f"the canary destination {destination!r} holds no readable "
            f"{publisher.ATTESTATION_NAME}, so there is no evidence the "
            f"T-{offset_minutes} release was ever canaried and nothing goes "
            "live"
        )

    return attestation


def assert_canary_validated_this_release(
    attestation: dict[str, Any], *, canary: dict[str, Any], offset_minutes: int
) -> dict[str, Any]:
    """The canary destination is serving the release this refresh would copy.

    Three equalities after three state checks, each refused separately
    because each describes a different mistake.

    The rows digest says the canary is serving the bytes this run's canary
    stage published, so the attestation has not been overtaken by another
    window's canary between the two stages. The source artifact digest says
    that canary was run on the priced file this run produced, which is the
    equality :func:`publish_incumbent_to_wizardofodds.assert_canary_attested`
    checks again against the parquet on disk. And the release id says the
    attestation is about the canary release this run published rather than
    about a republication of the same rows under a different provenance.
    """
    if str(attestation.get("mode") or "") != publisher.MODE_CANARY:
        raise LiveRefreshRefused(
            "the attestation at the canary destination records mode "
            f"{attestation.get('mode')!r} rather than "
            f"{publisher.MODE_CANARY!r}, so it is not a canary attestation"
        )

    if str(attestation.get("outcome") or "") != publisher.OUTCOME_PUBLISHED:
        raise LiveRefreshRefused(
            "the canary attestation records the outcome "
            f"{attestation.get('outcome')!r}, so no canary succeeded for the "
            f"T-{offset_minutes} release and nothing goes live"
        )

    if attestation.get("verified") is not True:
        raise LiveRefreshRefused(
            "the canary attestation records no verified readback, so the "
            f"T-{offset_minutes} canary did not prove its own destination "
            "held what was sent and its release is not copied live"
        )

    for label, attested, published in (
        (
            "rows digest",
            str(attestation.get("rows_sha256") or ""),
            canary["rows_sha256"],
        ),
        (
            "release id",
            str(attestation.get("release_id") or ""),
            canary["canary_release_id"],
        ),
        (
            "source artifact digest",
            str(attestation.get("source_artifact_sha256") or ""),
            canary["priced_market_artifact_sha256"],
        ),
    ):
        if published and attested != published:
            raise LiveRefreshRefused(
                f"the canary destination attests the {label} "
                f"{attested or 'nothing'} and this run's canary stage "
                f"published {published}, so what the canary is serving is not "
                f"what the T-{offset_minutes} live refresh would publish"
            )

    return {
        "attested_at": attestation.get("attested_at"),
        "release_id": str(attestation.get("release_id") or ""),
        "rows_sha256": str(attestation.get("rows_sha256") or ""),
        "source_artifact_sha256": str(
            attestation.get("source_artifact_sha256") or ""
        ),
    }


def assert_a_single_window_composite_is_the_canary_release(
    release: Any,
    *,
    assembled: composite.Composite,
    canary: dict[str, Any],
    attestation: dict[str, Any],
    canary_release_id: str,
) -> str:
    """The three statements of one digest, where there is still one digest.

    A composite whose games all come out of this window's own canary release,
    and all of that release's games, *is* that release: the rows payload holds
    the contract version, the rows, the distributions and the slate date and
    nothing about the destination, so it serialises to the identical bytes and
    digests to the identical ``rows_sha256``. In that case the equality the
    chain has always asserted still applies exactly as written, and it is
    asserted rather than quietly dropped.

    A composite spanning more than one window has a digest of its own and
    cannot be compared to any single attestation. What stands in its place is
    not weaker: every source release behind it goes through the publisher's
    own ``assert_canary_attested``, every row is looked up in the recomputed
    bytes of the release its game names, and all three untraceable-row counts
    have to be zero. Returned rather than silent, so the receipt says which of
    the two claims was available.
    """
    sources = {
        component.source_release_id
        for component in assembled.components.values()
    }

    if sources != {canary_release_id}:
        return (
            f"this composite spans {len(sources)} canary release(s), so its "
            "rows digest is its own and each component's attestation is "
            "checked against its own source release instead"
        )

    priced = assembled.games_of_source(canary_release_id)

    if priced != set(assembled.components):
        return (
            f"this composite holds {len(assembled.components)} of the "
            f"{len(priced)} game(s) {canary_release_id} prices, so its rows "
            "digest is its own and each component's attestation is checked "
            "against its own source release instead"
        )

    for label, built, validated in (
        ("canary publication", release.rows_sha256, canary["rows_sha256"]),
        (
            "canary attestation",
            release.rows_sha256,
            attestation["rows_sha256"],
        ),
    ):
        if validated and built != validated:
            raise LiveRefreshRefused(
                f"the release built for the live destination digests to "
                f"{built} and the {label} covers {validated}, so what was "
                "canaried is not what would go live. Nothing is promoted and "
                "the composite already live is left serving"
            )

    return (
        f"this composite is the whole of {canary_release_id}, so its rows "
        "digest is that canary's own and is equal to both the publication "
        "receipt's and the destination's attestation"
    )


# ----------------------------------------------------------------------
# which window is live
# ----------------------------------------------------------------------


def evidence_path(data_root: Path, slate_date: str) -> Path:
    return Path(data_root) / LIVE_EVIDENCE_RELATIVE / f"{slate_date}.jsonl"


def read_evidence(path: Path) -> list[dict[str, Any]]:
    """Every live attempt recorded for this slate, oldest first.

    A malformed line is a refusal rather than something to skip. This file is
    the only record of which window is live, so a line that cannot be read is
    a hole in exactly the thing the ordering rule depends on, and carrying on
    past it would let a truncated record read as an empty one -- which is the
    state in which any window may go live.
    """
    if not Path(path).exists():
        return []

    entries: list[dict[str, Any]] = []

    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue

        try:
            entry = json.loads(line)
        except ValueError as error:
            raise LiveRefreshRefused(
                f"{path}:{number} is not readable JSON ({error}), so the "
                "record of which window is live has a hole in it and no "
                "refresh can be shown to be going forward"
            ) from error

        if not isinstance(entry, dict):
            raise LiveRefreshRefused(
                f"{path}:{number} is not a JSON object, so it does not record "
                "a live publication"
            )

        entries.append(entry)

    return entries


def last_live_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most recent entry after which a live release was serving."""
    for entry in reversed(entries):
        if str(
            entry.get("outcome") or ""
        ) in OUTCOMES_THAT_ARE_LIVE and entry.get("live_release_id"):
            return entry

    return None


def _slate_of_the_current_release(
    transport: Any, *, current: dict[str, Any]
) -> str:
    """The slate the live release is of, read off its own manifest.

    From the manifest rather than parsed out of the release id. The id is
    ``prefix__slate__digest`` and the slate is in it, but reading a structured
    field is a fact where splitting a string is an inference -- and the
    manifest is the document ``current.json`` names, so if it is absent then
    the pointer does not name a complete release anyway.
    """
    relative = str(current.get("manifest_relative") or "")

    if not relative or not transport.exists(relative):
        raise LiveRefreshRefused(
            f"{publisher.CURRENT_POINTER} at the live destination names the "
            f"manifest {relative or 'nowhere'}, which is not there, so the "
            "release being served is not complete and is not replaced"
        )

    try:
        manifest = json.loads(transport.read_bytes(relative))
    except ValueError as error:
        raise LiveRefreshRefused(
            f"{relative} at the live destination is not readable JSON "
            f"({error}), so the release being served cannot be identified"
        ) from error

    return str(manifest.get("slate_date") or "")


def resolve_live_state(
    transport: Any,
    *,
    entries: list[dict[str, Any]],
    slate_date: str,
    evidence: Path,
) -> dict[str, Any]:
    """Which release is live, which window it came from, or a refusal.

    The pointer at the destination is the authority on *what* is live and the
    durable record is the authority on *which window* it came from, and the
    two have to agree before anything is replaced. They are separate because
    neither can answer both questions: ``current.json`` names a release id
    whose prefix is the live stream's rather than the window's, and the
    published rows cannot carry the offset because ``PUBLISHED_FIELDS`` is
    pinned. So the window is recorded beside the destination, and the
    recording is bound to it by the release id and the rows digest.

    Every disagreement is a refusal. A record naming a release the destination
    is not serving, or a destination serving this slate with no record behind
    it, means the answer to "which window is live" is unknown -- and a refresh
    that cannot tell which window is live cannot tell whether it is going
    forwards. The destination's own ``audit_log.jsonl`` holds every action
    ever taken there, so the refusal names it: reconciling the two is a read
    rather than a guess.
    """
    current = publisher.read_pointer(transport, publisher.CURRENT_POINTER)

    recorded = last_live_entry(entries)

    if not current:
        if recorded is not None:
            raise LiveRefreshRefused(
                f"{evidence} records {recorded.get('live_release_id')!r} as "
                "the live release for this slate and the destination has no "
                f"readable {publisher.CURRENT_POINTER}, so what a reader is "
                "being served is unknown. Reconcile against "
                f"{publisher.AUDIT_LOG_NAME} at the destination before "
                "another release is published live"
            )

        return {
            "established": False,
            "live_components": [],
            "live_offset_minutes": None,
            "live_release_id": "",
            "live_rows_sha256": "",
            "live_slate_date": "",
            "reason": (
                "the live destination serves no release at all, so this "
                "window establishes the live pointer rather than replacing one"
            ),
        }

    release_id = str(current.get("release_id") or "")

    rows_sha256 = str(current.get("rows_sha256") or "")

    served_slate = _slate_of_the_current_release(transport, current=current)

    if served_slate != slate_date:
        if recorded is not None:
            raise LiveRefreshRefused(
                f"{evidence} records {recorded.get('live_release_id')!r} as "
                f"the live release for {slate_date} and the destination is "
                f"serving {release_id!r} for {served_slate or 'no slate'}, so "
                "the two disagree about what is live. Reconcile against "
                f"{publisher.AUDIT_LOG_NAME} at the destination before "
                "another release is published live"
            )

        return {
            "established": True,
            "live_components": [],
            "live_offset_minutes": None,
            "live_release_id": release_id,
            "live_rows_sha256": rows_sha256,
            "live_slate_date": served_slate,
            "reason": (
                f"the live destination serves {release_id} for "
                f"{served_slate or 'no slate'}, so this window establishes "
                f"the live release for {slate_date}"
            ),
        }

    if recorded is None:
        raise LiveRefreshRefused(
            f"the live destination is serving {release_id!r} for {slate_date} "
            f"and {evidence} records no live publication for that slate, so "
            "which capture window a reader is being shown is unknown and no "
            "refresh can be shown to be going forward. Reconcile against "
            f"{publisher.AUDIT_LOG_NAME} at the destination first"
        )

    for label, served, logged in (
        ("release id", release_id, str(recorded.get("live_release_id") or "")),
        ("rows digest", rows_sha256, str(recorded.get("rows_sha256") or "")),
    ):
        if served != logged:
            raise LiveRefreshRefused(
                f"the live destination serves the {label} "
                f"{served or 'nothing'} and {evidence} records "
                f"{logged or 'nothing'}, so something other than this chain "
                "changed what is live. Reconcile against "
                f"{publisher.AUDIT_LOG_NAME} at the destination before "
                "another release is published live"
            )

    offset = recorded.get("offset_minutes")

    if offset is None:
        raise LiveRefreshRefused(
            f"{evidence} records {release_id!r} as live without the capture "
            "window it came from, so whether this refresh would be going "
            "forward or backward cannot be established"
        )

    components = recorded.get("games")

    if not components:
        raise LiveRefreshRefused(
            f"{evidence} records {release_id!r} as live without the per-game "
            "composition it was assembled from, so which window each game is "
            "serving is unknown and no game can be shown to be going forward. "
            f"Reconcile against {publisher.AUDIT_LOG_NAME} at the destination "
            "before another composite is published live"
        )

    return {
        "established": True,
        "live_components": components,
        "live_offset_minutes": int(offset),
        "live_release_id": release_id,
        "live_rows_sha256": rows_sha256,
        "live_slate_date": served_slate,
        "reason": (
            f"the live destination serves the T-{int(offset)} release "
            f"{release_id} for {slate_date}, which holds "
            f"{len(components)} game(s) at their own latest validated windows"
        ),
    }


def resolve_advancement(
    *,
    offset_minutes: int,
    pregame: list[dict[str, Any]],
    tipped: list[dict[str, Any]],
    live_components: dict[str, composite.GameComponent],
) -> dict[str, Any]:
    """Which games this refresh advances, which it leaves, and why.

    The per-game replacement of the one-pointer ordering rule, and the place
    the no-downgrade invariant is enforced. Per game:

    *No record.* It advances: this window is the first one that validated for
    it, so this is where its card comes from.

    *Recorded at a larger offset.* It advances, because this window is closer
    to its tip than the one it is serving.

    *Recorded at this offset.* It advances, which is how a cumulative release
    grows within one window: a segment that captures three tip groups serves
    and publishes three times at T-20, each release holding every game priced
    so far, and the later ones have to be able to replace the earlier ones for
    the games they are about.

    *Recorded at a smaller offset.* It is left exactly as it is. This is the
    one-pointer rule's decline, narrowed from the slate to the game it is
    actually about: a T-180 refresh arriving after that game's T-20 cannot
    walk it back, and -- unlike before -- it does not stop the refresh from
    advancing the games it is for.

    A game whose tip has passed is left as it is too, and for a different
    reason: not because this window is behind it, but because no game is
    newly published after it has started.

    When nothing is left to advance the refresh writes nothing. That is the
    ordering rule declining a window rather than anything failing, and it is
    the ordinary evening state on a staggered slate.
    """
    advancing: list[str] = []

    declined: list[dict[str, Any]] = []

    for entry in pregame:
        game_id = str(entry["game_id"])

        held = live_components.get(game_id)

        if held is None or int(held.offset_minutes) >= int(offset_minutes):
            advancing.append(game_id)
            continue

        declined.append(
            {
                "game_id": game_id,
                "held_window": composite.window_label(held.offset_minutes),
                "reason": (
                    f"game {game_id} is already serving its "
                    f"{composite.window_label(held.offset_minutes)} card, "
                    f"which is closer to its tip than T-{offset_minutes}, so "
                    "it is left exactly as it is and is never walked back to "
                    "an earlier window"
                ),
                "window_id": entry["window_id"],
            }
        )

    refused = [
        {
            "game_id": str(entry["game_id"]),
            "reason": (
                f"game {entry['game_id']} tipped at {entry['tip_utc']} and is "
                "not newly published after its own tip. Whatever card it "
                "already had was published before it started and is carried "
                "unchanged"
            ),
            "window_id": entry["window_id"],
        }
        for entry in tipped
    ]

    return {
        "advancing_game_ids": sorted(advancing),
        "declined": declined,
        "post_tip_refused": refused,
    }


def assert_something_advances(
    advancement: dict[str, Any],
    *,
    offset_minutes: int,
    tipped: list[dict[str, Any]],
    now: datetime,
) -> str:
    """At least one game of this refresh has a window to advance to.

    Two ways there is nothing, and they are different verdicts. Every game
    already closer to its own tip is the ordering rule working, and a success.
    Every game already started is the post-tip guard working, and a refusal:
    the games are named, nothing is written, and what they were serving stays
    served.
    """
    if advancement["advancing_game_ids"]:
        declined = advancement["declined"]

        return (
            f"{len(advancement['advancing_game_ids'])} game(s) advance to "
            f"T-{offset_minutes}"
            + (
                f", and {len(declined)} are already closer to their own tips "
                "and are carried unchanged"
                if declined
                else ""
            )
            + (
                f", and {len(advancement['post_tip_refused'])} have already "
                "tipped and are not newly published"
                if advancement["post_tip_refused"]
                else ""
            )
        )

    if advancement["declined"]:
        raise LiveAlreadyCloserToTip(
            f"this is the T-{offset_minutes} refresh and every game it is for "
            "is already serving a window closer to its own tip ("
            + "; ".join(
                f"{entry['game_id']} at {entry['held_window']}"
                for entry in advancement["declined"]
            )
            + "). A game is never replaced by an earlier one of its own "
            "windows, so nothing is written and every game's latest valid "
            "prediction is left serving"
        )

    raise post_tip_refusal(tipped, now=now)


# ----------------------------------------------------------------------
# establishing the first live release
# ----------------------------------------------------------------------


def assert_the_slate_was_served(
    *, slate_date: str, data_root: Path, serving_receipt: Path
) -> dict[str, Any]:
    """The incumbent served this slate, and the rows carry its provenance.

    Asked of the serving receipt and the priced parquet before anything is
    read from either destination. Nothing from here goes into the composite --
    every published number comes out of an attested canary release and not out
    of this file -- but a refresh of a slate the incumbent never served, or of
    a parquet whose receipt does not provenance it, is a refresh with no
    serving run behind it and is refused on the publisher's own terms.

    ``verify_receipt`` is the publisher's own, called by name, so the identity
    fields the composite's components are compared against are the ones the
    publisher would have embedded itself.
    """
    priced = publisher.priced_markets_path(Path(data_root), slate_date)

    if not priced.exists():
        raise LiveRefreshRefused(
            f"{priced} does not exist, so the incumbent has not served "
            f"{slate_date} and there is nothing to publish live"
        )

    try:
        receipt = json.loads(Path(serving_receipt).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LiveRefreshRefused(
            f"{serving_receipt} could not be read ({error}), so the rows at "
            f"{priced} carry no provenance and are not published live"
        ) from error

    return publisher.verify_receipt(
        receipt, slate_date=slate_date, priced_path=priced
    )


def build_composite_release(
    assembled: composite.Composite,
    *,
    switch: publisher.PublishingSwitch,
    destination: publisher.Destination,
    slate_date: str,
) -> Any:
    """The composite expressed as a release, built the publisher's way.

    ``publish_incumbent_to_wizardofodds.build_release`` is called by name, so
    there is no second definition of what a release payload holds, how it is
    digested, how it is named or what its manifest carries. What is handed to
    it is the already-attested rows and distributions the composite selected
    and a provenance assembled out of the components' own identities.

    Then the one equality that makes this a copy rather than a construction:
    the release's rows digest has to be the digest of the composite assembled
    out of attested bytes. It cannot normally differ -- both are
    ``serialise`` over the same payload -- which is exactly why it is asserted
    rather than assumed. The only ways it could differ are the ways that
    matter, and all of them end with a reader being shown numbers no canary
    read back.
    """
    release = publisher.build_release(
        rows=assembled.rows,
        distributions=assembled.distributions,
        provenance=composite.composite_provenance(
            assembled.components, slate_date=slate_date
        ),
        switch=switch,
        destination=destination,
        slate_date=slate_date,
    )

    if release.rows_sha256 != assembled.rows_sha256:
        raise LiveRefreshRefused(
            f"the release built for the live destination digests to "
            f"{release.rows_sha256} and the composite of canary-attested rows "
            f"digests to {assembled.rows_sha256}, so what was canaried is not "
            "what would go live. Nothing is promoted and the composite "
            "already live is left serving"
        )

    if release.rows_bytes != assembled.rows_bytes:
        raise LiveRefreshRefused(
            "the release built for the live destination does not hold the "
            "bytes of the composite assembled out of canary-attested rows, so "
            "it is not a copy of them. Nothing is promoted and the composite "
            "already live is left serving"
        )

    return release


def establish_first_live_release(
    transport: Any,
    *,
    release: Any,
    slate_date: str,
    provenance: dict[str, Any],
    composite_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Seed a live destination that has never served anything.

    :func:`publish_incumbent_to_wizardofodds.assert_rollback_target_exists`
    requires a ``current.json`` before it will let an activation move one, and
    it is right to: an activation must never be the step that leaves live with
    nowhere to go back to. Its own refusal says what the exception is --
    "seed the destination with the release it is already serving before
    activating" -- and on a destination that has never served anything there
    is nothing to seed it with.

    So the first release of the chain seeds itself. Both pointers are written
    to the same release, which is the truthful state of a destination holding
    one: what is current is what there is, and what it would roll back to is
    itself. Every refresh after it finds a rollback target and goes through
    the publisher's own gate with that gate untouched -- this path is
    reachable exactly once per live destination, and only when
    ``current.json`` is provably absent rather than merely unreadable.

    It adds no mechanism. The release bytes go through ``write_release``, the
    readback through ``verify_release``, the pointers through the transport's
    own atomic replace, the retention through ``prune_releases`` and the
    record through ``append_audit_entry``. What it does not do is run before
    the canary gate: the caller has already applied
    :func:`publish_incumbent_to_wizardofodds.assert_canary_attested` to every
    source release behind these exact bytes, so a live destination cannot be
    established from something no canary read back.
    """
    if transport.exists(publisher.CURRENT_POINTER):
        raise LiveRefreshRefused(
            f"{publisher.CURRENT_POINTER} is at the live destination, so it is "
            "already serving something and is not established over. This "
            "release must go through the publisher's own rollback gate"
        )

    publisher.write_release(transport, release)

    recorded = (
        composite.write_composite_manifest(
            transport,
            release_id=release.release_id,
            manifest=composite_manifest,
        )
        if composite_manifest is not None
        else {}
    )

    verification = {
        **publisher.verify_release(transport, release),
        **recorded,
    }

    pointer = {
        "manifest_relative": release.manifest_relative,
        "mode": publisher.MODE_LIVE,
        "promoted_at": _utc_now().isoformat(),
        "release_id": release.release_id,
        "rows_relative": release.rows_relative,
        "rows_sha256": release.rows_sha256,
    }

    # The rollback target first. A run that dies between the two writes leaves
    # a previous pointer and no current one, which reads as "nothing is being
    # served" and is recoverable; the other order would leave a served
    # release with no way back, which is the state the gate this path stands
    # in for exists to prevent.
    transport.write_bytes(
        publisher.PREVIOUS_POINTER, publisher.serialise(pointer)
    )

    transport.write_bytes(
        publisher.CURRENT_POINTER, publisher.serialise(pointer)
    )

    retention = publisher.prune_releases(transport, retained=None)

    audit_relative = publisher.append_audit_entry(
        transport,
        {
            "action": publisher.ACTION_PUBLISHED,
            "mode": publisher.MODE_LIVE,
            "model_authority": publisher.MODEL_AUTHORITY,
            "production_code_sha": provenance.get("production_code_sha"),
            "recorded_at": _utc_now().isoformat(),
            "release_id": release.release_id,
            "row_count": release.row_count,
            "rows_sha256": release.rows_sha256,
            "slate_date": slate_date,
            "source_artifact_sha256": provenance.get(
                "priced_market_artifact_sha256"
            ),
        },
    )

    return {
        "audit_relative": audit_relative,
        "promotion": {
            "promoted": True,
            "reason": (
                "the live destination served nothing, so this release is both "
                "what is current and what it would roll back to"
            ),
            "release_id": release.release_id,
            "rollback_target": release.release_id,
        },
        "provenance": provenance,
        "release": {
            "manifest_relative": release.manifest_relative,
            "manifest_sha256": release.manifest_sha256,
            "release_id": release.release_id,
            "row_count": release.row_count,
            "rows_relative": release.rows_relative,
            "rows_sha256": release.rows_sha256,
        },
        "retention": retention,
        "verification": verification,
    }


# ----------------------------------------------------------------------
# the durable record
# ----------------------------------------------------------------------


def append_evidence(path: Path, entry: dict[str, Any]) -> str:
    """One line, appended, never rewritten.

    Appended rather than read-and-rewritten, which is the opposite of the
    choice ``append_audit_entry`` makes at the destination and for the
    opposite reason: that transport offers one atomic write and no append,
    and this is an ordinary local file where the append is the atomic
    operation. A crash during one can leave a short final line, which
    :func:`read_evidence` refuses rather than skips.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    return str(path)


def promote_composite_release(
    transport: Any,
    *,
    release: Any,
    slate_date: str,
    provenance: dict[str, Any],
    composite_manifest: dict[str, Any],
    retained: int | None,
) -> dict[str, Any]:
    """Replace the live composite, in the publisher's own order and no other.

    This is the step ``publish_incumbent_to_wizardofodds.publish`` performs
    for a release built out of one priced parquet. A composite cannot go
    through that function, because the release it builds is the whole slate
    re-derived at one window and the whole point here is that the slate is not
    at one window -- but every operation is that function's own, called by
    name, in that function's own order:

    ``assert_rollback_target_exists``, ``write_release``, ``verify_release``,
    ``promote``, ``prune_releases``, ``append_audit_entry``.

    The one addition is the composite's own manifest, written before the
    readback and read back with the release, so a destination that accepted
    the rows and lost the statement of which window each game is at does not
    get promoted either.

    The gate ``publish`` applies before all of this --
    ``assert_canary_attested`` -- has already been applied by the caller once
    per source release behind these bytes, which is why it is not repeated
    here: the composite's digest is the digest of those attested rows and
    nothing between the audit and this call can touch them.
    """
    rollback_target = publisher.assert_rollback_target_exists(transport)

    publisher.write_release(transport, release)

    recorded = composite.write_composite_manifest(
        transport, release_id=release.release_id, manifest=composite_manifest
    )

    verification = {
        **publisher.verify_release(transport, release),
        **recorded,
    }

    promotion = publisher.promote(
        transport, release, mode=publisher.MODE_LIVE
    )

    # After the pointers, never before: the releases kept are the ones the
    # pointers name, so pruning first could remove the rollback target this
    # publication has just created.
    retention = publisher.prune_releases(transport, retained=retained)

    audit_relative = publisher.append_audit_entry(
        transport,
        {
            "action": (
                publisher.ACTION_PUBLISHED
                if promotion["promoted"]
                else publisher.ACTION_REPUBLISHED
            ),
            "mode": publisher.MODE_LIVE,
            "model_authority": publisher.MODEL_AUTHORITY,
            "production_code_sha": provenance.get("production_code_sha"),
            "recorded_at": _utc_now().isoformat(),
            "release_id": release.release_id,
            "row_count": release.row_count,
            "rows_sha256": release.rows_sha256,
            "slate_date": slate_date,
            "source_artifact_sha256": provenance.get(
                "priced_market_artifact_sha256"
            ),
        },
    )

    return {
        "activation": {"rollback_target": rollback_target},
        "audit_relative": audit_relative,
        "promotion": promotion,
        "provenance": provenance,
        "release": {
            "manifest_relative": release.manifest_relative,
            "manifest_sha256": release.manifest_sha256,
            "release_id": release.release_id,
            "row_count": release.row_count,
            "rows_relative": release.rows_relative,
            "rows_sha256": release.rows_sha256,
        },
        "retention": retention,
        "verification": verification,
    }


def evidence_entry(
    *,
    attempted_at: datetime,
    canary: dict[str, Any] | None,
    attestation: dict[str, Any] | None,
    composite_audit: dict[str, Any] | None = None,
    games: list[dict[str, Any]] | None = None,
    live_release_id: str,
    manifest_sha256: str,
    offset_minutes: int,
    outcome: str,
    previous_release_id: str,
    promoted_at: str,
    reason: str,
    row_count: int | None,
    rows_sha256: str,
    slate_date: str,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    """What one live attempt leaves behind, whatever happened to it.

    Every field is a fact about this attempt and none is derived from another
    entry, so a reader of one line knows which windows it was for, which
    offset, which release went live, what that release's bytes digest to,
    which canary validated them, what the live pointer named before and what
    the outcome was. ``live_release_id`` is empty on a refusal, which is what
    lets the ordering read pass over refusals without having to interpret
    them.

    ``games`` is the per-game composition the attempt left live, one entry per
    game in the composite manifest's own shape. It is the durable half of the
    no-downgrade invariant: the next refresh reads it to find out which window
    each game has already validly reached, and composes each game at the
    smaller of that offset and its own. Written on every live-bearing outcome
    and empty on a refusal, so a refusal can never be mistaken for a
    composition in which every game lost its card.
    """
    return {
        "attempted_at_utc": attempted_at.isoformat(),
        "canary": canary,
        "canary_attestation": attestation,
        "composite_audit": composite_audit,
        "feed_destination": LIVE_FEED_DESTINATION,
        "final_pregame_window": (
            int(offset_minutes) == FINAL_PREGAME_OFFSET_MINUTES
        ),
        "games": games or [],
        "live_release_id": live_release_id,
        "manifest_sha256": manifest_sha256,
        "mode": publisher.MODE_LIVE,
        "model_authority": publisher.MODEL_AUTHORITY,
        "offset_minutes": int(offset_minutes),
        "outcome": outcome,
        "previous_release_id": previous_release_id,
        "promoted_at_utc": promoted_at,
        "reason": reason,
        "row_count": row_count,
        "rows_sha256": rows_sha256,
        "slate_date": slate_date,
        "windows": windows,
    }


# ----------------------------------------------------------------------
# one refresh
# ----------------------------------------------------------------------


def _window_tips(raw: list[str] | None) -> dict[str, str]:
    tips: dict[str, str] = {}

    for item in raw or []:
        window_id, separator, tip = str(item).partition("=")

        if not separator or not window_id.strip() or not tip.strip():
            raise LiveRefreshRefused(
                f"--window-tip-utc {item!r} is not WINDOW_ID=TIP_UTC, so the "
                "window it is about and the tip it claims cannot be told apart"
            )

        tips[window_id.strip()] = tip.strip()

    return tips


def _window_captures(raw: list[str] | None) -> dict[str, dict[str, str]]:
    """``WINDOW_ID=CAPTURE_ID@CAPTURE_TIMESTAMP``, one per window.

    Parsed strictly and never partially. A capture identity with a missing
    half is refused rather than recorded empty, because a composite manifest
    that named a capture id and no timestamp, or a timestamp and no id, would
    be a provenance record nobody could follow back to a snapshot.
    """
    captures: dict[str, dict[str, str]] = {}

    for item in raw or []:
        window_id, separator, identity = str(item).partition("=")

        capture_id, at, captured_at = identity.partition("@")

        if (
            not separator
            or not at
            or not window_id.strip()
            or not capture_id.strip()
            or not captured_at.strip()
        ):
            raise LiveRefreshRefused(
                f"--window-capture {item!r} is not "
                "WINDOW_ID=CAPTURE_ID@CAPTURE_TIMESTAMP, so the capture a "
                "game's card came out of cannot be recorded"
            )

        captures[window_id.strip()] = {
            "capture_id": capture_id.strip(),
            "captured_at": captured_at.strip(),
        }

    return captures


def refresh(
    args: argparse.Namespace, *, publishing: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One window's live refresh, or the first refusal on the way to it.

    Ordered so that nothing is read from the live destination until every
    cheap local fact has held, and nothing is written to it until the canary,
    the per-game advancement read, the attestation audit and the clock have
    all been satisfied. Every refusal before the write leaves the live pointer
    exactly as it was, and so does every refusal after it up to ``promote``,
    which is the last thing either path does before the bookkeeping.

    ``publishing`` is the caller's window onto the one fact a failure handler
    cannot recover for itself: the digest this refresh was about to promote.
    It is written immediately before the first byte reaches the destination
    and is how :func:`main` tells a clean refusal apart from a promotion that
    did not finish. A composite's digest is its own -- it is not the canary's
    except in the single-window case -- so it cannot be read back off the
    canary receipt after the fact.
    """
    now = (
        parse_utc(args.now_utc, what="--now-utc")
        if args.now_utc
        else _utc_now()
    )

    offset = assert_offset_is_progressive(args.snapshot_offset_minutes)

    canary_name = (
        args.canary_destination_name
        if args.canary_destination_name is not None
        else CANARY_FEED_DESTINATIONS[offset]
    )

    assert_destinations_are_the_live_ones(
        offset_minutes=offset,
        canary=canary_name,
        feed=args.live_feed_destination_name,
        page=args.live_page_destination_name,
    )

    windows, tipped = partition_windows_by_tip(
        {
            window_id: parse_utc(tip, what=f"--window-tip-utc {window_id}")
            for window_id, tip in _window_tips(args.window_tips).items()
        },
        now=now,
        offset_minutes=offset,
    )

    # Raised here and not left to the advancement read, so a refresh in which
    # every game has already started refuses before the switch, the canary or
    # either destination is consulted. Nothing it could learn further on could
    # make a started game publishable.
    if not windows:
        raise post_tip_refusal(tipped, now=now)

    captures = _window_captures(args.window_captures)

    project_root = Path(args.project_root).resolve()

    data_root = Path(args.data_root).resolve()

    switch = publisher.read_publishing_switch(
        publisher.MODE_LIVE,
        project_root=project_root,
        switch_path=args.switch_path,
        environment=None,
    )

    approval = assert_live_approval_is_carried(switch)

    canary = read_canary_feed_receipt(
        args.canary_feed_receipt,
        slate_date=args.slate_date,
        offset_minutes=offset,
        destination=canary_name,
    )

    document = publisher.read_destinations(
        project_root=project_root, destinations_path=args.destinations_path
    )

    attested, attestation_path, stored = _attestation_from_the_canary(
        document,
        offset_minutes=offset,
        canary=canary,
        canary_name=canary_name,
        canary_root=args.canary_destination_root,
        beside=Path(args.receipt_path),
        data_root=data_root,
        slate_date=args.slate_date,
    )

    provenance = assert_the_slate_was_served(
        slate_date=args.slate_date,
        data_root=data_root,
        serving_receipt=args.serving_receipt,
    )

    live = publisher.resolve_destination(
        publisher.MODE_LIVE,
        document,
        root=args.live_destination_root,
        name=(
            None
            if args.live_destination_root is not None
            else args.live_feed_destination_name
        ),
    )

    publisher.assert_live_destination_is_the_served_root(live)

    evidence = evidence_path(data_root, args.slate_date)

    entries = read_evidence(evidence)

    transport = publisher.build_transport(live)

    try:
        identity = (
            transport.verify_identity()
            if hasattr(transport, "verify_identity")
            else None
        )

        state = resolve_live_state(
            transport,
            entries=entries,
            slate_date=args.slate_date,
            evidence=evidence,
        )

        held = composite.recorded_components(state["live_components"])

        advancement = resolve_advancement(
            offset_minutes=offset,
            pregame=windows,
            tipped=tipped,
            live_components=held,
        )

        ordering = assert_something_advances(
            advancement, offset_minutes=offset, tipped=tipped, now=now
        )

        assembled = assemble_composite(
            data_root=data_root,
            slate_date=args.slate_date,
            offset_minutes=offset,
            canary_name=canary_name,
            canary_release_id=canary["canary_release_id"],
            advancing=advancement["advancing_game_ids"],
            captures=captures,
            held=held,
        )

        manifest_document = composite.composite_manifest(
            assembled,
            release_id="",
            rows_sha256=assembled.rows_sha256,
            slate_date=args.slate_date,
            feed_destination=args.live_feed_destination_name,
        )

        if (
            state["live_slate_date"] == args.slate_date
            and state["live_rows_sha256"] == assembled.rows_sha256
        ):
            return _receipt(
                args,
                approval=approval,
                attestation=attested,
                attestation_path=attestation_path,
                canary=canary,
                composite_manifest={
                    **manifest_document,
                    "release_id": state["live_release_id"],
                },
                destination=publisher.destination_payload(
                    live, identity=identity
                ),
                live_outcome=OUTCOME_ALREADY_CURRENT,
                manifest_sha256="",
                now=now,
                offset=offset,
                ordering=ordering,
                advancement=advancement,
                previous_release_id=state["live_release_id"],
                promoted=None,
                promoted_at="",
                reason=(
                    f"the live destination already serves "
                    f"{state['live_release_id']}, whose rows digest is the "
                    "one this composite of "
                    f"{len(assembled.components)} game(s) assembles to, so "
                    "nothing was written and the pointer was not touched"
                ),
                release={
                    "release_id": state["live_release_id"],
                    "rows_sha256": state["live_rows_sha256"],
                },
                row_count=None,
                state=state,
                stored_sources=stored,
                verification=None,
                windows=windows,
            )

        publisher.assert_mode_enabled(switch)

        release = build_composite_release(
            assembled,
            switch=switch,
            destination=live,
            slate_date=args.slate_date,
        )

        identity_claim = assert_a_single_window_composite_is_the_canary_release(
            release,
            assembled=assembled,
            canary=canary,
            attestation=attested,
            canary_release_id=canary["canary_release_id"],
        )

        manifest_document = {
            **composite.composite_manifest(
                assembled,
                release_id=release.release_id,
                rows_sha256=release.rows_sha256,
                slate_date=args.slate_date,
                feed_destination=args.live_feed_destination_name,
            ),
            "digest_identity": identity_claim,
        }

        if publishing is not None:
            publishing["release_id"] = release.release_id

            publishing["rows_sha256"] = release.rows_sha256

        if state["established"]:
            published = promote_composite_release(
                transport,
                release=release,
                slate_date=args.slate_date,
                provenance=provenance,
                composite_manifest=manifest_document,
                retained=live.retained_releases,
            )

            live_outcome = OUTCOME_REPLACED

        else:
            published = establish_first_live_release(
                transport,
                release=release,
                slate_date=args.slate_date,
                provenance=provenance,
                composite_manifest=manifest_document,
            )

            live_outcome = OUTCOME_ESTABLISHED

        released = published["release"]

        promoted = publication.verify_promoted_release(
            transport,
            release_id=str(released["release_id"]),
            rows_sha256=str(released["rows_sha256"]),
        )
    except publication.PublicationStageFailed as error:
        # The post-promotion readback. By the time it runs the pointer has
        # already moved, so this cannot be reported as a clean refusal.
        raise LiveRefreshIncomplete(str(error)) from error
    finally:
        transport.close()

    return _receipt(
        args,
        approval=approval,
        attestation=attested,
        attestation_path=attestation_path,
        canary=canary,
        composite_manifest=manifest_document,
        destination=publisher.destination_payload(live, identity=identity),
        live_outcome=live_outcome,
        manifest_sha256=str(released.get("manifest_sha256") or ""),
        now=now,
        offset=offset,
        ordering=ordering,
        advancement=advancement,
        previous_release_id=str(state["live_release_id"]),
        promoted=promoted,
        promoted_at=str(promoted.get("promoted_at") or ""),
        reason=(
            f"{promoted['row_count']} rows of a composite holding "
            f"{len(assembled.components)} game(s) at their own latest "
            f"validated windows, {len(advancement['advancing_game_ids'])} of "
            f"them advanced to T-{offset} by this refresh, were published "
            f"live as {released['release_id']} under publication contract "
            f"{promoted['contract_version']}"
            + (
                ", which is the final pregame version for the games at it"
                if offset == FINAL_PREGAME_OFFSET_MINUTES
                else ""
            )
        ),
        release=released,
        row_count=promoted.get("row_count"),
        state=state,
        stored_sources=stored,
        verification=published.get("verification"),
        windows=windows,
    )


def assemble_composite(
    *,
    data_root: Path,
    slate_date: str,
    offset_minutes: int,
    canary_name: str,
    canary_release_id: str,
    advancing: list[str],
    captures: dict[str, dict[str, str]],
    held: dict[str, composite.GameComponent],
) -> composite.Composite:
    """The whole live slate: this window's advances over everything held.

    The advancing games take their component out of this window's own canary
    release; every other game keeps the component recorded for it, read back
    out of its own stored source. Which is the second invariant in its
    strongest available form -- a game nobody advanced is reassembled from
    bytes this refresh cannot reach and did not look at, so it comes out
    byte-identical by construction rather than by care.

    Every source behind the result is then re-read, re-digested and put
    through the publisher's own canary gate by
    :func:`live_slate_composite.audit_composite`, and all three untraceable-row
    counts have to be zero before anything is published.
    """
    components = dict(held)

    advanced = read_advancing_components(
        data_root=data_root,
        slate_date=slate_date,
        offset_minutes=offset_minutes,
        canary_name=canary_name,
        canary_release_id=canary_release_id,
        advancing=advancing,
        captures=captures,
    )

    components.update(advanced)

    bundles = {
        release_id: composite.read_source_bundle(
            data_root, slate_date, release_id
        )
        for release_id in sorted(
            {
                component.source_release_id
                for component in components.values()
            }
        )
    }

    assembled = composite.compose(
        components, bundles, slate_date=slate_date
    )

    composite.assert_every_row_is_attested(assembled.audit)

    return assembled


def read_advancing_components(
    *,
    data_root: Path,
    slate_date: str,
    offset_minutes: int,
    canary_name: str,
    canary_release_id: str,
    advancing: list[str],
    captures: dict[str, dict[str, str]],
) -> dict[str, composite.GameComponent]:
    """One component per advancing game, out of this window's canary release.

    The capture identity is required and never defaulted. A component that
    could not name the capture its prices were taken from would be a row on
    the served root with no snapshot behind it that anybody could find again,
    and inventing one here is exactly the thing this chain may never do -- so
    the refusal names the flag and the window instead.
    """
    bundle = composite.read_source_bundle(
        data_root, slate_date, canary_release_id
    )

    components: dict[str, composite.GameComponent] = {}

    for game_id in advancing:
        window_id = f"{game_id}:T-{int(offset_minutes)}m"

        capture = captures.get(window_id)

        if capture is None:
            raise LiveRefreshRefused(
                f"no --window-capture was supplied for {window_id}, so the "
                "capture this game's prices were taken from cannot be named "
                "in the composite manifest and the game is not published. "
                "Every advancing window needs "
                "WINDOW_ID=CAPTURE_ID@CAPTURE_TIMESTAMP"
            )

        components[game_id] = composite.component_for_game(
            bundle,
            game_id=game_id,
            offset_minutes=offset_minutes,
            canary_destination=canary_name,
            capture_id=capture["capture_id"],
            capture_timestamp=capture["captured_at"],
        )

    return components


def _attestation_from_the_canary(
    document: dict[str, Any],
    *,
    offset_minutes: int,
    canary: dict[str, Any],
    canary_name: str,
    canary_root: Path | None,
    beside: Path,
    data_root: Path,
    slate_date: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Read the canary's attestation, check it, and keep what it covers.

    The attestation is read from the destination rather than from the caller,
    so "the canary validated these bytes" is a fact about the canary host and
    not about this run's temporary directory, and it is left on disk because
    the publisher's own gate takes a path.

    The release it covers is copied into the durable source store in the same
    connection, which is what lets a composite reassemble a game hours later.
    The canary destinations retain two releases and no more -- ``current.json``
    and ``previous.json`` -- so a game that advanced at T-180 out of the first
    of four cumulative T-180 releases has a source the canary host has since
    pruned. Copying it while the attestation of those exact bytes is in hand
    is the only moment at which it can be kept honestly.
    """
    transport = publisher.build_transport(
        publisher.resolve_destination(
            publisher.MODE_CANARY,
            document,
            root=canary_root,
            name=None if canary_root is not None else canary_name,
        )
    )

    try:
        attestation = read_attestation_from_the_canary(
            transport, destination=canary_name, offset_minutes=offset_minutes
        )

        attested = assert_canary_validated_this_release(
            attestation, canary=canary, offset_minutes=offset_minutes
        )

        release_id = canary["canary_release_id"]

        stored = composite.retain_source_release(
            transport,
            data_root=data_root,
            slate_date=slate_date,
            release_id=release_id,
            rows_relative=(
                f"{publisher.RELEASES_DIRECTORY}/{release_id}/"
                f"{publisher.ROWS_NAME}"
            ),
            manifest_relative=(
                f"{publisher.RELEASES_DIRECTORY}/{release_id}/"
                f"{publisher.MANIFEST_NAME}"
            ),
            attestation=attestation,
        )
    finally:
        transport.close()

    path = Path(beside).parent / "live_canary_attestation.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return attested, path, stored


def _receipt(
    args: argparse.Namespace,
    *,
    advancement: dict[str, Any],
    approval: dict[str, Any],
    attestation: dict[str, Any],
    attestation_path: Path,
    canary: dict[str, Any],
    composite_manifest: dict[str, Any],
    destination: dict[str, Any],
    live_outcome: str,
    manifest_sha256: str,
    now: datetime,
    offset: int,
    ordering: str,
    previous_release_id: str,
    promoted: dict[str, Any] | None,
    promoted_at: str,
    reason: str,
    release: dict[str, Any],
    row_count: int | None,
    state: dict[str, Any],
    stored_sources: dict[str, Any],
    verification: dict[str, Any] | None,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    """This refresh's receipt, in the publication receipt's own shape.

    ``outcome`` is the publication vocabulary's ``PUBLISHED`` rather than this
    module's own verdict, and that is deliberate.
    ``run_wizardofodds_publication.pin_from_feed_receipt`` reads that field to
    decide whether there is a verified release to build a page from, and that
    file is pinned syntax-tree-identical to production and may not learn a new
    word -- so the live page stage reads this receipt unchanged, and the
    progressive verdict is carried beside it in ``live_outcome``.

    ``composite`` is the whole per-game manifest, carried in the receipt as
    well as written at the destination so the controller's segment report can
    say which window every game is at without opening a connection.
    """
    return {
        "advancement": advancement,
        "canary": canary,
        "canary_attestation": attestation,
        "canary_attestation_path": str(attestation_path),
        "composite": composite_manifest,
        "composite_source_retained": stored_sources,
        "contract_version": publisher.PUBLICATION_CONTRACT_VERSION,
        "destination": destination,
        "final_pregame_window": offset == FINAL_PREGAME_OFFSET_MINUTES,
        "generated_at": _utc_now().isoformat(),
        "live_approval": approval,
        "live_outcome": live_outcome,
        "live_state_before": state,
        "mode": publisher.MODE_LIVE,
        "model_authority": publisher.MODEL_AUTHORITY,
        "offset_minutes": offset,
        "ordering": ordering,
        "outcome": publication.OUTCOME_PUBLISHED,
        "previous_release_id": previous_release_id,
        "promoted": promoted,
        "promoted_at": promoted_at,
        "reason": reason,
        "release": release,
        "row_count": row_count,
        "slate_date": args.slate_date,
        "stage": "live",
        "verification": verification,
        "windows": windows,
        "_evidence": {
            "attempted_at": now,
            "manifest_sha256": manifest_sha256,
        },
    }


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def render(receipt: dict[str, Any]) -> str:
    release = receipt.get("release") or {}

    manifest = receipt.get("composite") or {}

    advancement = receipt.get("advancement") or {}

    rows: list[tuple[str, Any]] = [
        ("live outcome", receipt.get("live_outcome")),
        ("capture window", f"T-{receipt.get('offset_minutes')}"),
        ("slate date", receipt.get("slate_date")),
        ("release", release.get("release_id")),
        ("rows digest", str(release.get("rows_sha256") or "")[:16]),
        (
            "canary that validated it",
            (receipt.get("canary") or {}).get("canary_release_id"),
        ),
        ("replaced", receipt.get("previous_release_id") or "nothing"),
        ("rows live", receipt.get("row_count")),
        ("games live", len(manifest.get("games") or [])),
        (
            "advanced by this refresh",
            ", ".join(advancement.get("advancing_game_ids") or []) or "none",
        ),
        (
            composite.UNATTESTED_ROWS_FIELD,
            manifest.get(composite.UNATTESTED_ROWS_FIELD),
        ),
        (
            composite.SOURCE_MISMATCH_FIELD,
            manifest.get(composite.SOURCE_MISMATCH_FIELD),
        ),
        (
            "final pregame version",
            "yes" if receipt.get("final_pregame_window") else "no",
        ),
    ]

    lines = [
        f"## NBA per-game live composite (T-{receipt.get('offset_minutes')})",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    lines += [f"| {label} | {value} |" for label, value in rows]

    games = manifest.get("games") or []

    if games:
        lines += [
            "",
            "| game | window | source release | capture |",
            "| --- | --- | --- | --- |",
        ]

        lines += [
            "| {} | {} | {} | {} |".format(
                game.get("GAME_ID"),
                game.get("SOURCE_WINDOW"),
                game.get("SOURCE_RELEASE_ID"),
                str(game.get("CAPTURE_ID") or "")[:16],
            )
            for game in games
        ]

    lines += ["", str(receipt.get("reason") or "")]

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_progressive_live_refresh",
        description=(
            "Publish the already-validated release of one capture window to "
            "the single live destination, replacing the window before it. "
            "Computes nothing, re-derives nothing, and refuses unless the "
            "committed switch and both approval variables agree."
        ),
    )

    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", type=Path, required=True)

    parser.add_argument(
        "--snapshot-offset-minutes",
        type=int,
        required=True,
        help="the capture window whose validated release goes live. Required "
        "and not defaulted: the window is the whole subject of the ordering "
        "rule, and a default would make it a property of this file rather "
        "than of the caller's decision.",
    )

    parser.add_argument(
        "--window-tip-utc",
        action="append",
        default=None,
        dest="window_tips",
        metavar="WINDOW_ID=TIP_UTC",
        help="when the game behind one of the windows this refresh is for "
        "tips, repeatable. Required: a release whose window has already "
        "tipped is not a pregame price and does not go live.",
    )

    parser.add_argument(
        "--window-capture",
        action="append",
        default=None,
        dest="window_captures",
        metavar="WINDOW_ID=CAPTURE_ID@CAPTURE_TIMESTAMP",
        help="the capture one of this refresh's windows was priced from, "
        "repeatable. Required for every window that advances: the composite "
        "manifest names the capture behind each game's card, and this module "
        "may not read the capture log to find out -- a live publisher that "
        "could would be a live publisher with an opinion about captures.",
    )

    parser.add_argument(
        "--canary-feed-receipt",
        type=Path,
        required=True,
        help="the receipt the canary feed stage wrote for this window. The "
        "rows digest it names is the digest the live release has to carry.",
    )

    parser.add_argument(
        "--serving-receipt",
        type=Path,
        required=True,
        help="the serving receipt the priced artifact was produced under, as "
        "the publisher requires it and for the same reason: the rows "
        "published carry its identities and are re-hashed against it.",
    )

    parser.add_argument("--canary-destination-name", default=None)
    parser.add_argument("--canary-destination-root", type=Path, default=None)

    parser.add_argument(
        "--live-feed-destination-name", default=LIVE_FEED_DESTINATION
    )

    parser.add_argument(
        "--live-page-destination-name", default=LIVE_PAGE_DESTINATION
    )

    parser.add_argument("--live-destination-root", type=Path, default=None)

    parser.add_argument("--switch-path", type=Path, default=None)
    parser.add_argument("--destinations-path", type=Path, default=None)

    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    parser.add_argument(
        "--now-utc",
        default=None,
        help="the moment the post-tip guard is evaluated at. Defaults to the "
        "clock.",
    )

    parser.add_argument(
        "--receipt-path",
        type=Path,
        required=True,
        help="where this refresh's receipt is written. Required: the live page "
        "stage is pinned to the release it names, so it is not optional "
        "output.",
    )

    parser.add_argument("--summary-path", type=Path, default=None)

    return parser.parse_args(argv)


def _write(path: Path | None, text: str, *, append: bool) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def _observed_live_release(args: argparse.Namespace) -> dict[str, Any]:
    """What ``current.json`` names after a failure, best effort.

    Asked so that a run which promoted and then broke is recorded as having
    promoted. Best effort because the failure may be the transport itself, and
    a second failure here must not replace the first one in the report.
    """
    try:
        document = publisher.read_destinations(
            project_root=Path(args.project_root),
            destinations_path=args.destinations_path,
        )

        transport = publisher.build_transport(
            publisher.resolve_destination(
                publisher.MODE_LIVE,
                document,
                root=args.live_destination_root,
                name=(
                    None
                    if args.live_destination_root is not None
                    else args.live_feed_destination_name
                ),
            )
        )

        try:
            return publisher.read_pointer(
                transport, publisher.CURRENT_POINTER
            ) or {}
        finally:
            transport.close()
    except Exception:  # noqa: BLE001 - observation only, never a verdict
        # Deliberately total. This reads the destination to describe it, not
        # to decide anything: a destination that cannot even be read has
        # certainly not been written by this run, which is what the caller
        # needs to know.
        return {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    attempted_at = _utc_now()

    # What the live destination was serving before this refresh touched it.
    #
    # Read up front and not in the failure handler, because it is the only
    # thing that can tell a clean refusal apart from a run that promoted and
    # then failed. The digest alone cannot: a refusal of the same bytes that
    # are *already* live finds the canary's rows digest at the destination
    # and nothing has moved. That is not a hypothetical -- the T-20 group of
    # a slate is re-served whenever a later tip group captures, so a window
    # the post-tip guard refuses can carry exactly the digest the pointer
    # already names, and calling that an incomplete promotion would raise a
    # loud incident about a destination nothing wrote to.
    #
    # One extra pointer read per refresh is what it costs. The alternative is
    # a classification that is wrong in the one case somebody is reading it.
    observed_before = _observed_live_release(args)

    # The digest this refresh was about to promote, filled in by the refresh
    # itself immediately before it writes anything. Empty means nothing was
    # ever on its way to the destination.
    publishing: dict[str, Any] = {}

    try:
        receipt = refresh(args, publishing=publishing)

    except (
        LiveRefreshRefused,
        LiveRefreshIncomplete,
        composite.CompositeRefused,
        publisher.PublishingRefused,
        publication.PublicationStageFailed,
        OSError,
        ValueError,
    ) as error:
        observed = _observed_live_release(args)

        offset = int(args.snapshot_offset_minutes)

        # A refusal left the pointer alone; an incomplete run moved it. The
        # two are told apart by the pointer itself rather than by which
        # exception was raised, and by both halves of the claim: the pointer
        # has to have changed, and it has to now name the digest this refresh
        # was publishing. Either half on its own is satisfiable by something
        # else -- a pointer that already carried these bytes, or a pointer
        # somebody moved for another reason -- and only this refresh could
        # have made both true at once.
        expected = str(publishing.get("rows_sha256") or "")

        promoted = bool(
            expected
            and str(observed.get("rows_sha256") or "") == expected
            and observed != observed_before
        )

        # The ordering rule declining a window is the one of these that is
        # not a failure. It is still handled here, by the same code and with
        # the same guarantees, because it reaches this point by raising and
        # because every claim the refusal path makes about the destination is
        # a claim this case needs too.
        declined = isinstance(error, LiveAlreadyCloserToTip)

        outcome = (
            OUTCOME_PROMOTED_THEN_INCOMPLETE
            if promoted
            else OUTCOME_ALREADY_CLOSER_TO_TIP
            if declined
            else OUTCOME_REFUSED
        )

        failure = {
            "error": type(error).__name__,
            "live_outcome": outcome,
            "message": str(error),
            "mode": publisher.MODE_LIVE,
            "observed_live_release_id": str(observed.get("release_id") or ""),
            "offset_minutes": offset,
            "outcome": (
                "LIVE_REFRESH_DECLINED"
                if declined
                else "LIVE_REFRESH_REFUSED"
            ),
            "slate_date": args.slate_date,
            "stage": "live",
        }

        try:
            append_evidence(
                evidence_path(Path(args.data_root), args.slate_date),
                evidence_entry(
                    attempted_at=attempted_at,
                    canary=None,
                    attestation=None,
                    live_release_id=(
                        str(observed.get("release_id") or "")
                        if promoted
                        else ""
                    ),
                    manifest_sha256="",
                    offset_minutes=offset,
                    outcome=outcome,
                    previous_release_id="",
                    promoted_at=str(observed.get("promoted_at") or ""),
                    reason=str(error),
                    row_count=None,
                    rows_sha256=str(observed.get("rows_sha256") or "")
                    if promoted
                    else "",
                    slate_date=args.slate_date,
                    windows=[],
                ),
            )
        except OSError:  # pragma: no cover - the data root is unwritable
            pass

        _write(
            args.receipt_path,
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            append=False,
        )

        _write(
            args.summary_path,
            f"## NBA per-game live composite (T-{offset})\n\n"
            f"{outcome}: {error}\n",
            append=True,
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_OK if outcome in OUTCOMES_THAT_SUCCEED else EXIT_REFUSED

    recorded = receipt.pop("_evidence")

    release = receipt["release"]

    manifest = receipt["composite"]

    receipt["evidence_path"] = append_evidence(
        evidence_path(Path(args.data_root), args.slate_date),
        evidence_entry(
            attempted_at=recorded["attempted_at"],
            canary=receipt["canary"],
            attestation=receipt["canary_attestation"],
            composite_audit={
                name: manifest.get(name)
                for name in (
                    composite.UNATTESTED_ROWS_FIELD,
                    composite.SOURCE_MISMATCH_FIELD,
                    composite.UNATTESTED_DISTRIBUTIONS_FIELD,
                    "attested_rows",
                    "digest_identity",
                    "source_releases",
                )
            },
            games=list(manifest.get("games") or []),
            live_release_id=str(release.get("release_id") or ""),
            manifest_sha256=recorded["manifest_sha256"],
            offset_minutes=int(receipt["offset_minutes"]),
            outcome=str(receipt["live_outcome"]),
            previous_release_id=str(receipt["previous_release_id"] or ""),
            promoted_at=str(receipt["promoted_at"] or ""),
            reason=str(receipt["reason"]),
            row_count=receipt.get("row_count"),
            rows_sha256=str(release.get("rows_sha256") or ""),
            slate_date=args.slate_date,
            windows=receipt["windows"],
        ),
    )

    _write(
        args.receipt_path,
        json.dumps(receipt, default=str, indent=2, sort_keys=True) + "\n",
        append=False,
    )

    _write(args.summary_path, render(receipt), append=True)

    print(json.dumps(receipt, default=str, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
