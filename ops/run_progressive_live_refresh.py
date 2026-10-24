#!/usr/bin/env python3
"""Replace the live release with the window that has just passed canary.

WHAT WAS MISSING
----------------

Every piece of live publication existed and nothing joined them up. The two
live destinations are fully addressed in ``ops/wizardofodds_destinations.json``
and have been for months. The publisher takes ``--mode live``, refuses it on
the committed switch, requires a prior canary attestation of the same bytes
and requires a rollback target before it will move anything. The page builder
has a live entry in ``MODE_DESTINATIONS`` naming the served root. What was
absent was the thing none of them can supply on its own: a statement of
*which* capture window is live, *which* window may replace it, and what
happens to the pointer when a later window fails.

That is this module, and it is deliberately thin. It does not publish. It
calls ``publish_incumbent_to_wizardofodds.publish`` with ``mode="live"``, by
name, so the switch gate, the destination confinement, the receipt
verification, the row rules, the release construction, the canary-attestation
match, the rollback-target requirement, the readback and the atomic pointer
rotation are the frozen publisher's own and not a second copy of any of them.
It then calls ``run_wizardofodds_publication.verify_promoted_release`` --
again by name -- so the post-promotion readback a live refresh is held to is
the one the canary feed stage is held to.

What it adds is four decisions the publisher has no opinion about.

THE PROGRESSIVE CHAIN
---------------------

T-180 establishes the live release, T-90 replaces it, T-45 replaces that and
T-20 replaces that as the final pregame version. Four refreshes of one live
pointer, not four live pages: ``wizardofodds_nba_live`` and
``wizardofodds_nba_player_props_page_live`` are single destinations with a
single ``current.json`` each, and a reader who opens the page at any point in
the evening is shown the most recent window that validated.

The offsets descend, so "a later window may replace an earlier one" is "a
smaller offset may replace a larger one", and the reverse is refused. That
matters because the self-hosted runner makes it reachable: pickup on this
repository's runner has been delayed by hours, so a T-180 invocation could
finish after a T-45 one, and without the ordering rule it would walk the live
page back to the market as it stood three hours before tip -- with every
digest agreeing and every readback passing, because what it published is a
real validated release. It is simply the wrong one.

IDENTICAL BYTES, PROVED BY DIGEST
---------------------------------

The live release is not re-derived. It is the release the canary validated,
and three independent statements of one digest have to agree before anything
is written:

* the canary feed publication receipt this run's canary stage wrote, which
  names the ``rows_sha256`` the canary published;
* ``canary_attestation.json`` read back from the canary destination itself,
  which names the ``rows_sha256`` and the ``source_artifact_sha256`` that
  destination is actually serving and proved by readback;
* the release built for the live destination, whose ``rows_sha256`` is
  compared against both before it is promoted.

The rows digest is the digest of the rows payload, which holds the contract
version, the rows, the distributions and the slate date and nothing about the
destination -- so it is identical at the canary and at live by construction,
and any difference at all means the bytes changed between the two stages. The
release *id* does differ, because it carries the destination's own prefix, and
that is the point: one set of rows, published under the live stream's name.

``publish_incumbent_to_wizardofodds.assert_canary_attested`` performs the
first two of those comparisons from inside the publisher, against the release
it has just built, and both are kept. That one is the gate. This module's runs
before a connection to the live host is opened at all, so a canary that does
not match is refused without the live destination being touched.

THE PRIOR VALID RELEASE SURVIVES EVERY FAILURE
----------------------------------------------

Structurally, not by compensation. The publisher's order of operations is
switch, destination, priced artifact, serving receipt, rows, release,
transport, canary attestation, rollback target, write, readback, *promote*.
Every refusal above happens before ``promote``, and ``promote`` is the only
thing that touches ``current.json``. So a refresh that fails for any reason --
no capture, a failed serve, a canary that did not validate, a digest that does
not match, a game that has already tipped, a transport that will not talk --
leaves the live pointer naming the last release that did validate, and the
release before that is still at the destination because ``previous.json``
names it and ``prune_releases`` is pointer-driven rather than date-driven.

This module adds no second retention scheme. ``previous.json`` and
``rollback`` are the mechanism; what is new is only that the thing being
retained now has a capture window attached to it.

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

#: The live release is already at a window closer to tip than this one, so
#: nothing was written and the pointer is unchanged.
#:
#: Its own outcome, and a success, because it is the ordering rule working
#: rather than anything failing. It is also the ordinary evening state on a
#: staggered slate and not an edge case: the windows are per game, the
#: offsets are measured from each game's own tip, and a twelve-game slate
#: with five tip times interleaves them -- the 7:00 game's T-20 capture is at
#: 6:40 and the 10:30 game's T-180 capture is at 7:30, an hour *later* in the
#: evening and three hours *earlier* in its own game's window. So an advisory
#: window firing after some game has already reached T-20 is routine, and
#: what the chain does about it is exactly what the rule says: the release
#: closer to tip stays serving. Reporting that as a fault every evening would
#: make the segment's own verdict unreadable, which is the only thing a
#: verdict is for.
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


def assert_no_window_has_tipped(
    tips: dict[str, datetime], *, now: datetime, offset_minutes: int
) -> list[dict[str, Any]]:
    """Every window this refresh is for is still ahead of its tip.

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

    The subject is the windows this refresh is *for*, not every window in the
    release. A release is cumulative -- one serve at an offset prices every
    game with an eligible capture at that offset -- so by the time the last
    T-20 group of a twelve-game slate is published, the first game tipped two
    hours earlier, and its rows are correctly in the release because they were
    priced before their own tip. What has to still be ahead is the tip of the
    game whose window this refresh is the refresh for.
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

    tipped = sorted(
        (window_id, tip) for window_id, tip in tips.items() if now >= tip
    )

    if tipped:
        raise LiveRefreshRefused(
            "the game(s) "
            + ", ".join(
                f"{window_id} (tipped {tip.isoformat()}, "
                f"{round((now - tip).total_seconds() / 60.0)} minutes ago)"
                for window_id, tip in tipped
            )
            + f" have already started and it is now {now.isoformat()}. A "
            "window whose game has tipped is not a pregame price and does not "
            "go live; the release already live is left serving"
        )

    return [
        {
            "minutes_before_tip": round((tip - now).total_seconds() / 60.0, 3),
            "tip_utc": tip.isoformat(),
            "window_id": window_id,
        }
        for window_id, tip in sorted(tips.items())
    ]


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


def assert_live_release_is_the_canary_release(
    release: Any, *, canary: dict[str, Any], attestation: dict[str, Any]
) -> None:
    """The bytes about to be promoted are the bytes that were validated.

    The last of the three statements of one digest, taken on the release the
    publisher actually built for the live destination. It cannot normally
    differ -- the rows payload is a function of the priced parquet, the
    distributions and the slate date, and the publisher re-hashes the parquet
    against the serving receipt before it builds anything -- which is exactly
    why it is asserted rather than assumed: the only ways it could differ are
    the ways that matter, and all of them end with a reader being shown
    numbers no canary ever read back.
    """
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
                f"{built} and the {label} covers {validated}, so live would "
                "be given bytes no canary validated. Nothing is promoted and "
                "the release already live is left serving"
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

    return {
        "established": True,
        "live_offset_minutes": int(offset),
        "live_release_id": release_id,
        "live_rows_sha256": rows_sha256,
        "live_slate_date": served_slate,
        "reason": (
            f"the live destination serves the T-{int(offset)} release "
            f"{release_id} for {slate_date}"
        ),
    }


def assert_the_refresh_goes_forward(
    *, offset_minutes: int, live_offset_minutes: int | None
) -> str:
    """A later window may replace an earlier one, and never the reverse.

    The offsets descend through the evening, so going forward is the offset
    getting smaller. Equality is allowed and is how a cumulative release grows
    within one window: a segment that captures three tip groups serves and
    publishes three times at T-20, each release holding every game priced so
    far, and the later ones have to be able to replace the earlier ones.

    Going backwards is declined rather than treated as a failure, because on
    a staggered slate it is the normal case. A window belongs to a game and
    its offset is measured from that game's own tip, so an evening's windows
    interleave: on the twelve-game slate of 2026-10-23 the 7:00 game's T-20
    capture is taken at 6:40 and the 10:30 game's T-180 capture an hour later
    at 7:30, which is three hours earlier in its own game's window. The
    advisory refresh that follows it is asking to put a T-180 release live
    over a T-20 one, and the answer is the one the rule gives: the release
    closer to tip stays serving, nothing is written, and the chain carries on.
    """
    if live_offset_minutes is None:
        return (
            f"no release is live for this slate, so the T-{offset_minutes} "
            "window establishes it"
        )

    if int(offset_minutes) > int(live_offset_minutes):
        raise LiveAlreadyCloserToTip(
            f"the T-{live_offset_minutes} release is already live and this is "
            f"the T-{offset_minutes} refresh, which is an earlier capture "
            "window. The chain only ever moves closer to tip, so a later "
            "window is never replaced by an earlier one and the release "
            "already live is left serving"
        )

    if int(offset_minutes) == int(live_offset_minutes):
        return (
            f"the T-{offset_minutes} release is already live and this refresh "
            "is the same window, which is how a cumulative release grows "
            "within one window"
        )

    return (
        f"the T-{live_offset_minutes} release is live and the "
        f"T-{offset_minutes} window is closer to tip, so it replaces it"
    )


# ----------------------------------------------------------------------
# establishing the first live release
# ----------------------------------------------------------------------


def build_live_release(
    *,
    switch: publisher.PublishingSwitch,
    destination: publisher.Destination,
    slate_date: str,
    data_root: Path,
    serving_receipt: Path,
) -> tuple[Any, dict[str, Any]]:
    """The release the live destination would hold, built the publisher's way.

    Needed only by the establishment path below, which cannot go through
    ``publish`` and therefore has to have the release in hand before it writes
    anything. Every step is the publisher's own function called by name and in
    the publisher's own order -- the receipt verification that re-hashes the
    parquet, the row rules, the distribution projection, the release
    construction -- so there is no second definition of what a live release
    contains or of what it is named.
    """
    import pandas as pd

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

    provenance = publisher.verify_receipt(
        receipt, slate_date=slate_date, priced_path=priced
    )

    rows = publisher.published_rows(
        pd.read_parquet(priced), switch=switch, slate_date=slate_date
    )

    release = publisher.build_release(
        rows=rows,
        distributions=publisher.published_distributions(
            publisher.read_distributions_document(
                publisher.distributions_path(Path(data_root), slate_date)
            ),
            rows=rows,
            priced_sha256=str(
                provenance.get("priced_market_artifact_sha256") or ""
            ),
        ),
        provenance=provenance,
        switch=switch,
        destination=destination,
        slate_date=slate_date,
    )

    return release, provenance


def establish_first_live_release(
    transport: Any,
    *,
    release: Any,
    slate_date: str,
    provenance: dict[str, Any],
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
    :func:`publish_incumbent_to_wizardofodds.assert_canary_attested` to these
    exact bytes, so a live destination cannot be established from something no
    canary read back.
    """
    if transport.exists(publisher.CURRENT_POINTER):
        raise LiveRefreshRefused(
            f"{publisher.CURRENT_POINTER} is at the live destination, so it is "
            "already serving something and is not established over. This "
            "release must go through the publisher's own rollback gate"
        )

    publisher.write_release(transport, release)

    verification = publisher.verify_release(transport, release)

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


def evidence_entry(
    *,
    attempted_at: datetime,
    canary: dict[str, Any] | None,
    attestation: dict[str, Any] | None,
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
    """
    return {
        "attempted_at_utc": attempted_at.isoformat(),
        "canary": canary,
        "canary_attestation": attestation,
        "feed_destination": LIVE_FEED_DESTINATION,
        "final_pregame_window": (
            int(offset_minutes) == FINAL_PREGAME_OFFSET_MINUTES
        ),
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


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    """One window's live refresh, or the first refusal on the way to it.

    Ordered so that nothing is read from the live destination until every
    cheap local fact has held, and nothing is written to it until the canary,
    the ordering and the clock have all been satisfied. Every refusal before
    the write leaves the live pointer exactly as it was, and so does every
    refusal the publisher makes after it: ``promote`` is the last thing either
    path does before the bookkeeping.
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

    windows = assert_no_window_has_tipped(
        {
            window_id: parse_utc(tip, what=f"--window-tip-utc {window_id}")
            for window_id, tip in _window_tips(args.window_tips).items()
        },
        now=now,
        offset_minutes=offset,
    )

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

    attested, attestation_path = _attestation_from_the_canary(
        document,
        offset_minutes=offset,
        canary=canary,
        canary_name=canary_name,
        canary_root=args.canary_destination_root,
        beside=Path(args.receipt_path),
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
        state = resolve_live_state(
            transport,
            entries=entries,
            slate_date=args.slate_date,
            evidence=evidence,
        )

        ordering = assert_the_refresh_goes_forward(
            offset_minutes=offset,
            live_offset_minutes=state["live_offset_minutes"],
        )

        if (
            state["live_slate_date"] == args.slate_date
            and state["live_rows_sha256"] == canary["rows_sha256"]
        ):
            return _receipt(
                args,
                approval=approval,
                attestation=attested,
                attestation_path=attestation_path,
                canary=canary,
                destination=publisher.destination_payload(live),
                live_outcome=OUTCOME_ALREADY_CURRENT,
                manifest_sha256="",
                now=now,
                offset=offset,
                ordering=ordering,
                previous_release_id=state["live_release_id"],
                promoted=None,
                promoted_at="",
                reason=(
                    f"the live destination already serves "
                    f"{state['live_release_id']}, whose rows digest is the "
                    f"one the T-{offset} canary validated, so nothing was "
                    "written and the pointer was not touched"
                ),
                release={
                    "release_id": state["live_release_id"],
                    "rows_sha256": state["live_rows_sha256"],
                },
                row_count=None,
                state=state,
                verification=None,
                windows=windows,
            )

        if state["established"]:
            published = _replace_the_live_release(
                args,
                attestation_path=attestation_path,
                canary=canary,
                attested=attested,
                offset=offset,
                live_name=(
                    None
                    if args.live_destination_root is not None
                    else args.live_feed_destination_name
                ),
            )

            live_outcome = OUTCOME_REPLACED

        else:
            publisher.assert_mode_enabled(switch)

            release, provenance = build_live_release(
                switch=switch,
                destination=live,
                slate_date=args.slate_date,
                data_root=data_root,
                serving_receipt=args.serving_receipt,
            )

            assert_live_release_is_the_canary_release(
                release, canary=canary, attestation=attested
            )

            publisher.assert_canary_attested(
                publisher.read_attestation(attestation_path),
                release=release,
                provenance=provenance,
            )

            published = establish_first_live_release(
                transport,
                release=release,
                slate_date=args.slate_date,
                provenance=provenance,
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
        destination=published.get("destination")
        or publisher.destination_payload(live),
        live_outcome=live_outcome,
        manifest_sha256=str(released.get("manifest_sha256") or ""),
        now=now,
        offset=offset,
        ordering=ordering,
        previous_release_id=str(state["live_release_id"]),
        promoted=promoted,
        promoted_at=str(promoted.get("promoted_at") or ""),
        reason=(
            f"{promoted['row_count']} rows of the T-{offset} capture window "
            f"were published live as {released['release_id']} under "
            f"publication contract {promoted['contract_version']}"
            + (
                ", which is the final pregame version"
                if offset == FINAL_PREGAME_OFFSET_MINUTES
                else ""
            )
        ),
        release=released,
        row_count=promoted.get("row_count"),
        state=state,
        verification=published.get("verification"),
        windows=windows,
    )


def _attestation_from_the_canary(
    document: dict[str, Any],
    *,
    offset_minutes: int,
    canary: dict[str, Any],
    canary_name: str,
    canary_root: Path | None,
    beside: Path,
) -> tuple[dict[str, Any], Path]:
    """Read the canary's attestation, check it, and leave it on disk.

    On disk because the publisher's own gate takes a path, and from the
    destination rather than from the caller so the file that gate reads is the
    file the canary host holds.
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
    finally:
        transport.close()

    attested = assert_canary_validated_this_release(
        attestation, canary=canary, offset_minutes=offset_minutes
    )

    path = Path(beside).parent / "live_canary_attestation.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return attested, path


def _replace_the_live_release(
    args: argparse.Namespace,
    *,
    attestation_path: Path,
    canary: dict[str, Any],
    attested: dict[str, Any],
    offset: int,
    live_name: str | None,
) -> dict[str, Any]:
    """Hand the whole publication to the publisher, then check the digest.

    ``publish`` opens its own transport, which is why this is a separate call
    rather than something done through the one already held: the publisher
    owns the connection it writes over, and borrowing a transport into it
    would be a second way of doing the one thing it is the authority on.
    """
    receipt = publisher.publish(
        mode=publisher.MODE_LIVE,
        slate_date=args.slate_date,
        data_root=Path(args.data_root),
        project_root=Path(args.project_root),
        destination_root=args.live_destination_root,
        switch_path=args.switch_path,
        destinations_path=args.destinations_path,
        receipt_path=args.serving_receipt,
        canary_attestation=attestation_path,
        destination_name=live_name,
    )

    released = receipt["release"]

    for label, published, validated in (
        ("canary publication", released["rows_sha256"], canary["rows_sha256"]),
        (
            "canary attestation",
            released["rows_sha256"],
            attested["rows_sha256"],
        ),
    ):
        if validated and published != validated:
            raise LiveRefreshIncomplete(
                f"the release promoted live digests to {published} and the "
                f"{label} covers {validated}. The pointer has moved to bytes "
                f"no canary validated for T-{offset} and has to be rolled back"
            )

    return receipt


def _receipt(
    args: argparse.Namespace,
    *,
    approval: dict[str, Any],
    attestation: dict[str, Any],
    attestation_path: Path,
    canary: dict[str, Any],
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
    """
    return {
        "canary": canary,
        "canary_attestation": attestation,
        "canary_attestation_path": str(attestation_path),
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
        (
            "final pregame version",
            "yes" if receipt.get("final_pregame_window") else "no",
        ),
    ]

    lines = [
        f"## NBA progressive live refresh (T-{receipt.get('offset_minutes')})",
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
    except Exception:  # pragma: no cover - the transport is already broken
        return {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    attempted_at = _utc_now()

    try:
        receipt = refresh(args)

    except (
        LiveRefreshRefused,
        LiveRefreshIncomplete,
        publisher.PublishingRefused,
        publication.PublicationStageFailed,
        OSError,
        ValueError,
    ) as error:
        observed = _observed_live_release(args)

        offset = int(args.snapshot_offset_minutes)

        # A refusal left the pointer alone; an incomplete run moved it. The
        # two are told apart by the pointer itself rather than by which
        # exception was raised: only this refresh could have put the canary's
        # rows digest at the live destination, so finding it there is proof
        # the promotion happened.
        expected = _canary_rows_digest(args)

        promoted = bool(
            expected and str(observed.get("rows_sha256") or "") == expected
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
            f"## NBA progressive live refresh (T-{offset})\n\n"
            f"{outcome}: {error}\n",
            append=True,
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_OK if outcome in OUTCOMES_THAT_SUCCEED else EXIT_REFUSED

    recorded = receipt.pop("_evidence")

    release = receipt["release"]

    receipt["evidence_path"] = append_evidence(
        evidence_path(Path(args.data_root), args.slate_date),
        evidence_entry(
            attempted_at=recorded["attempted_at"],
            canary=receipt["canary"],
            attestation=receipt["canary_attestation"],
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


def _canary_rows_digest(args: argparse.Namespace) -> str:
    """The rows digest the canary published, read without any checking.

    Used only to tell a refusal apart from a promotion that did not finish,
    which is a question about what is at the destination rather than about
    whether the receipt was acceptable -- so the receipt is read for one field
    and nothing is concluded from it.
    """
    try:
        receipt = json.loads(
            Path(args.canary_feed_receipt).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return ""

    if not isinstance(receipt, dict):
        return ""

    return str((receipt.get("release") or {}).get("rows_sha256") or "")


if __name__ == "__main__":
    raise SystemExit(main())
