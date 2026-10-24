#!/usr/bin/env python3
"""Hold every game of a live slate at its own latest validated window.

WHAT WAS WRONG WITH ONE POINTER PER SLATE
-----------------------------------------

The progressive chain published the live slate through a single forward-only
pointer. It recorded one "closest window offset reached so far" for the whole
evening and declined any later refresh at a larger offset.

On a slate where every game tips at once that is correct and complete. On a
staggered slate it is degenerate, because a window belongs to a *game* and its
offset is measured from that game's own tip. The 6:00 PM game reaches T-20 at
5:40 PM; the 10:00 PM game reaches T-180 at 7:00 PM, an hour and twenty
minutes *later* in the evening and three hours *earlier* in its own game's
window. With one pointer the second refresh is declined, and so is every
advisory window of every later game for the rest of the night.

Measured on the eight-game 2026-10-24 slate, with tips from 6:00 PM to 10:00
PM ET, the first game's T-20 at 5:40 PM suppressed the T-180, T-90 and T-45
windows of all seven games behind it: fourteen of twenty-four window groups
declined, and most of the slate with no live card at all. Every one of those
declines was the ordering rule working exactly as written, which is what made
it invisible -- the segment was green, the receipts were correct, and the page
a reader opened held one game.

WHAT REPLACES IT
----------------

A composite. The live slate holds, simultaneously, each game at its own
latest validated window: game A at T-20, game B at T-90 and game C at T-180
is a single live release carrying exactly those three cards at exactly those
three windows. Per game and independently:

* before any capture, that game has no card;
* once its T-180 validates, the slate serves that game's T-180;
* once its T-90 validates, that replaces the game's T-180;
* once its T-45 validates, that replaces the game's T-90;
* once its T-20 validates, that replaces the game's T-45.

Two invariants hold over every refresh. A game is never downgraded to a
window earlier than the latest it has already validly reached, because
composition takes the smaller of the recorded offset and the offset being
refreshed and never the other way round. And advancing one game never
removes or alters another's latest valid prediction, because every other
game's component is copied out of its own stored source release and is a
function of that release's bytes alone -- so games B..H come out byte-
identical whatever A does.

COMPOSITION IS SELECTION, NEVER COMPUTATION
-------------------------------------------

Nothing here synthesises, re-prices, re-routes or recomputes anything. Every
per-game component is copied byte for byte out of a release a canary already
validated and read back, and the composite is the concatenation of those rows
in the publisher's own :data:`publish_incumbent_to_wizardofodds.LOGICAL_KEY`
order together with the distribution records those same bytes carry. There is
no arithmetic in this module and no import of anything that could do any: the
only thing it knows how to do with a row is move it.

That is what makes the attestation chain survive composition. A single-window
composite -- every game of one source release advancing together -- produces a
rows payload byte-identical to that release's own, and therefore the identical
``rows_sha256``, which is why the three-way digest identity the chain has
always asserted still holds in that case rather than being special-cased away.

HOW A ROW IS TRACED BACK TO A CANARY
------------------------------------

Three documents are copied out of the canary destination when a game advances,
and kept under :data:`SOURCE_STORE_RELATIVE` for the rest of the evening:

* ``rows.json`` -- the release payload, the bytes the canary published;
* ``manifest.json`` -- which carries that release's own provenance;
* ``canary_attestation.json`` -- the canary destination's own statement, proved
  by readback, of what it is serving.

They are kept because the canary destinations retain two releases and no more:
``prune_releases`` keeps what ``current.json`` and ``previous.json`` name, so
a game that advanced at T-180 from the first of four cumulative T-180 releases
has a source the canary host no longer holds. Reading it back off the canary
every time would make the composite fail for a game that did nothing wrong.

The store is content-addressed in effect -- a release id is a prefix, a slate
and the first sixteen characters of its own rows digest -- and
:func:`retain_source_release` refuses to overwrite a stored document with
different bytes, so a release id cannot come to mean two things.

Verification recomputes. :func:`read_source_bundle` digests the stored rows
bytes with the publisher's own :func:`publish_incumbent_to_wizardofodds.digest`
and :func:`assert_source_is_attested` hands the result to the publisher's own
:func:`publish_incumbent_to_wizardofodds.assert_canary_attested` against the
stored attestation -- the same gate, unweakened, applied once per source
release instead of once per publication. ``COMPOSITE_UNATTESTED_ROWS`` is then
counted by looking each composite row up in those recomputed bytes, not by
reading a digest field off a manifest and believing it.

WHAT THE MANIFEST RECORDS
-------------------------

One entry per game, each field read out of the attested bytes rather than
derived:

``GAME_ID``
    The game, as the published rows spell it.
``SOURCE_WINDOW``
    The capture window the component came from, ``T-180m`` through ``T-20m``.
``CAPTURE_ID`` and ``CAPTURE_TIMESTAMP``
    The capture the window's prices were taken from, as the controller's own
    ``select_capture_bundle`` recomputed them. Supplied by the caller rather
    than read here, because a live publisher that could read the capture log
    would be a live publisher that could have an opinion about captures.
``SOURCE_RELEASE_ID`` and ``SOURCE_ROWS_SHA256``
    The canary release the rows were copied from, and its recomputed digest.
``PRODUCTION_SHA``, ``FIT_ID``, ``MODEL_ID``
    ``production_code_sha``, ``incumbent_fit_id`` and ``incumbent_version``
    from the source release's own manifest provenance.
``FACTOR_ID``
    The frozen bundle manifest digest the rows carry in ``manifest_sha256``,
    which is what identifies the fitted marginals, copula and factor
    artifacts the prices came out of.
``POLICY_ID``
    The pricing and projection schema versions the release was approved
    under, which is what identifies the policy the prices were quoted by.

Three counts are reported beside them and all three are required to be zero:
``COMPOSITE_UNATTESTED_ROWS``, ``LIVE_SOURCE_MISMATCH_COUNT`` and
``COMPOSITE_UNATTESTED_DISTRIBUTIONS``. A composite with even one row that
cannot be traced to a canary-attested source refuses to publish.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ops"))

# Imported as a module and used by name, for the reason the refresh entry
# point imports it that way: what a release is, what a row is, how bytes are
# serialised and digested, what a canary attestation has to have proved and
# what order a publication happens in are the frozen publisher's own and are
# not restated here.
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402

#: Where the attested source releases a composite is assembled from are kept.
#:
#: Beside the live record rather than inside it, and durable rather than in a
#: stage directory, because a game that advanced three hours ago still has its
#: component in tonight's composite and the canary host no longer holds the
#: release it came from.
SOURCE_STORE_RELATIVE = Path("processed") / "live_composite_sources"

#: The three documents one source release is remembered by. All three are
#: copied byte for byte: the rows are what is published, the manifest is where
#: the component's identities are read from, and the attestation is the canary
#: destination's own proof that it served those exact rows.
SOURCE_ROWS_NAME = "rows.json"
SOURCE_MANIFEST_NAME = "manifest.json"
SOURCE_ATTESTATION_NAME = "canary_attestation.json"

#: The composite's own manifest, written beside the release it describes at
#: the live destination and carried in the refresh receipt.
COMPOSITE_MANIFEST_NAME = "composite_manifest.json"

#: The composite manifest's schema version. Its own, not the publication
#: contract's: this document is read by operators and by this repository's own
#: tests and is not part of what a reader's page is rendered against.
COMPOSITE_MANIFEST_CONTRACT_VERSION = 1

#: Rows in the composite that cannot be traced back to the canary-attested
#: bytes of the source release their game names. Required to be zero.
UNATTESTED_ROWS_FIELD = "COMPOSITE_UNATTESTED_ROWS"

#: Rows in the composite that are present in their source release and are not
#: byte-identical to it. Required to be zero.
SOURCE_MISMATCH_FIELD = "LIVE_SOURCE_MISMATCH_COUNT"

#: Distribution records in the composite that are absent from, or differ from,
#: the source release of the game whose wager they forecast. Required to be
#: zero: a published price with no forecast behind it is a hole in the page,
#: and one with somebody else's forecast behind it is worse.
UNATTESTED_DISTRIBUTIONS_FIELD = "COMPOSITE_UNATTESTED_DISTRIBUTIONS"


class CompositeRefused(RuntimeError):
    """A fail-closed refusal: no composite was assembled or published."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def window_label(offset_minutes: int) -> str:
    """``T-180m``, the way a window id spells its own offset."""
    return f"T-{int(offset_minutes)}m"


# ----------------------------------------------------------------------
# one game's component
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GameComponent:
    """One game, at one window, out of one attested release.

    Frozen, and every field a fact about the source rather than about the
    composite it is going into, so the same component read out of the durable
    record a refresh later composes to the same bytes.
    """

    game_id: str
    offset_minutes: int
    source_release_id: str
    source_rows_sha256: str
    capture_id: str
    capture_timestamp: str
    canary_destination: str
    production_sha: str
    fit_id: Any
    model_id: str
    factor_id: str
    policy_id: str
    row_count: int
    served_at: str
    source_artifact_sha256: str
    incumbent_authority: str = ""

    def manifest_entry(self) -> dict[str, Any]:
        """This component as the composite manifest records it."""
        return {
            "CAPTURE_ID": self.capture_id,
            "CAPTURE_TIMESTAMP": self.capture_timestamp,
            "FACTOR_ID": self.factor_id,
            "FIT_ID": self.fit_id,
            "GAME_ID": self.game_id,
            "MODEL_ID": self.model_id,
            "POLICY_ID": self.policy_id,
            "PRODUCTION_SHA": self.production_sha,
            "ROW_COUNT": int(self.row_count),
            "SOURCE_CANARY_DESTINATION": self.canary_destination,
            "SOURCE_RELEASE_ID": self.source_release_id,
            "SOURCE_ROWS_SHA256": self.source_rows_sha256,
            "SOURCE_SERVED_AT": self.served_at,
            "SOURCE_WINDOW": window_label(self.offset_minutes),
            "SOURCE_ARTIFACT_SHA256": self.source_artifact_sha256,
            "SOURCE_INCUMBENT_AUTHORITY": self.incumbent_authority,
        }

def window_offset(label: Any) -> int:
    """The offset a ``T-180m`` window label states, or a refusal.

    Strict on purpose. This is read back out of the durable record to decide
    whether a game would be going forward, so a label this cannot parse has
    to stop the composite rather than fall back to a number -- every possible
    fallback is either a downgrade or a false claim about which window a card
    came from.
    """
    text = str(label or "")

    if not re.fullmatch(r"T-\d+m", text):
        raise CompositeRefused(
            f"{text!r} is not a window label of the form T-180m, so the "
            "offset the game it belongs to is serving cannot be established "
            "and no composite is assembled"
        )

    return int(text[2:-1])


def component_from_record(record: dict[str, Any]) -> GameComponent:
    """One component read back out of the durable record, or a refusal.

    Every field is required. A record missing one is a hole in exactly the
    thing composition reads to decide what a game is already serving, and
    defaulting it would turn a truncated record into a licence to downgrade.
    """
    if not isinstance(record, dict):
        raise CompositeRefused(
            "a per-game live component was recorded as "
            f"{type(record).__name__} rather than an object, so what that "
            "game is serving cannot be established"
        )

    required = (
        "CAPTURE_ID",
        "CAPTURE_TIMESTAMP",
        "FACTOR_ID",
        "GAME_ID",
        "MODEL_ID",
        "POLICY_ID",
        "PRODUCTION_SHA",
        "SOURCE_CANARY_DESTINATION",
        "SOURCE_RELEASE_ID",
        "SOURCE_ROWS_SHA256",
        "SOURCE_WINDOW",
    )

    missing = sorted(
        name for name in required if record.get(name) in (None, "")
    )

    if missing:
        raise CompositeRefused(
            "a per-game live component records no "
            + ", ".join(missing)
            + ", so the window and source release that game is serving "
            "cannot be established and no composite is assembled"
        )

    return GameComponent(
        game_id=str(record["GAME_ID"]),
        offset_minutes=window_offset(record["SOURCE_WINDOW"]),
        source_release_id=str(record["SOURCE_RELEASE_ID"]),
        source_rows_sha256=str(record["SOURCE_ROWS_SHA256"]),
        capture_id=str(record["CAPTURE_ID"]),
        capture_timestamp=str(record["CAPTURE_TIMESTAMP"]),
        canary_destination=str(record["SOURCE_CANARY_DESTINATION"]),
        production_sha=str(record["PRODUCTION_SHA"]),
        fit_id=record.get("FIT_ID"),
        model_id=str(record["MODEL_ID"]),
        factor_id=str(record["FACTOR_ID"]),
        policy_id=str(record["POLICY_ID"]),
        row_count=int(record.get("ROW_COUNT") or 0),
        served_at=str(record.get("SOURCE_SERVED_AT") or ""),
        source_artifact_sha256=str(record.get("SOURCE_ARTIFACT_SHA256") or ""),
        incumbent_authority=str(
            record.get("SOURCE_INCUMBENT_AUTHORITY") or ""
        ),
    )


def recorded_components(records: Any) -> dict[str, GameComponent]:
    """The per-game composition one durable entry describes, keyed by game."""
    if records in (None, []):
        return {}

    if not isinstance(records, list):
        raise CompositeRefused(
            "the per-game live composition was recorded as "
            f"{type(records).__name__} rather than a list of components"
        )

    components: dict[str, GameComponent] = {}

    for record in records:
        component = component_from_record(record)

        if component.game_id in components:
            raise CompositeRefused(
                f"game {component.game_id} is recorded twice in one live "
                "composition, so which window it is serving is ambiguous"
            )

        components[component.game_id] = component

    return components


# ----------------------------------------------------------------------
# the attested source store
# ----------------------------------------------------------------------


@dataclass
class SourceBundle:
    """One canary release, as the store holds it and this module re-reads it.

    ``rows_sha256`` is recomputed from ``rows_bytes`` on every read and is
    never taken from the manifest or the attestation, which are the documents
    it is checked *against*.
    """

    release_id: str
    rows_bytes: bytes = field(repr=False)
    rows_document: dict[str, Any] = field(repr=False)
    manifest: dict[str, Any] = field(repr=False)
    attestation: dict[str, Any] = field(repr=False)
    rows_sha256: str = ""

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self.manifest.get("provenance") or {})

    def rows(self) -> list[dict[str, Any]]:
        rows = self.rows_document.get("rows")

        if not isinstance(rows, list):
            raise CompositeRefused(
                f"the stored rows of {self.release_id} carry no rows list, so "
                "no component can be taken out of it"
            )

        return rows

    def distributions(self) -> dict[str, Any]:
        records = self.rows_document.get(publisher.DISTRIBUTIONS_KEY)

        return records if isinstance(records, dict) else {}


def source_directory(
    data_root: Path, slate_date: str, release_id: str
) -> Path:
    return (
        Path(data_root)
        / SOURCE_STORE_RELATIVE
        / str(slate_date)
        / str(release_id)
    )


def _write_once(path: Path, payload: bytes, *, what: str) -> None:
    """Write a stored document, or require the stored one to be these bytes.

    Never an overwrite with different content. A release id is derived from
    its own rows digest, so two different documents under one id means either
    a collision nobody will see or a store somebody has edited, and both of
    them end with a composite claiming a canary validated bytes it never saw.
    """
    if path.exists():
        existing = path.read_bytes()

        if existing != payload:
            raise CompositeRefused(
                f"{path} already holds a different {what} for this release "
                f"({publisher.digest(existing)} rather than "
                f"{publisher.digest(payload)}), so the stored source cannot "
                "be trusted and no composite is assembled from it"
            )

        return

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_bytes(payload)


def retain_source_release(
    transport: Any,
    *,
    data_root: Path,
    slate_date: str,
    release_id: str,
    rows_relative: str,
    manifest_relative: str,
    attestation: dict[str, Any],
) -> dict[str, Any]:
    """Copy one canary release's three documents into the durable store.

    Copied rather than referenced, and copied at the moment the canary's own
    attestation of these exact bytes has just been read back off that
    destination. The digest is recomputed here too: a transport that handed
    back something other than what the attestation names is refused before
    anything is stored, so the store cannot come to hold a document no canary
    attested.
    """
    for label, relative in (
        ("rows", rows_relative),
        ("manifest", manifest_relative),
    ):
        if not relative or not transport.exists(relative):
            raise CompositeRefused(
                f"the canary destination holds no {label} at "
                f"{relative or 'any path'} for {release_id}, so that release "
                "cannot be kept as the source of a live component"
            )

    rows_bytes = transport.read_bytes(rows_relative)

    recomputed = publisher.digest(rows_bytes)

    attested = str(attestation.get("rows_sha256") or "")

    if recomputed != attested:
        raise CompositeRefused(
            f"the rows of {release_id} read back off the canary destination "
            f"digest to {recomputed} and its attestation covers "
            f"{attested or 'nothing'}, so the bytes offered as a live "
            "component are not the bytes that canary proved it was serving"
        )

    directory = source_directory(data_root, slate_date, release_id)

    _write_once(
        directory / SOURCE_ROWS_NAME, rows_bytes, what="rows document"
    )

    _write_once(
        directory / SOURCE_MANIFEST_NAME,
        transport.read_bytes(manifest_relative),
        what="release manifest",
    )

    _write_once(
        directory / SOURCE_ATTESTATION_NAME,
        publisher.serialise(attestation),
        what="canary attestation",
    )

    return {
        "release_id": release_id,
        "rows_sha256": recomputed,
        "stored_at": str(directory),
    }


def read_source_bundle(
    data_root: Path, slate_date: str, release_id: str
) -> SourceBundle:
    """One stored source release, with its rows digest recomputed.

    A bundle that is absent, unreadable or short a document is a refusal and
    never a game quietly dropped out of the composite: the previous live
    composite stays serving, which is the only outcome that cannot show a
    reader a slate with a hole in it.
    """
    directory = source_directory(data_root, slate_date, release_id)

    documents: dict[str, Any] = {}

    for name in (
        SOURCE_ROWS_NAME,
        SOURCE_MANIFEST_NAME,
        SOURCE_ATTESTATION_NAME,
    ):
        path = directory / name

        try:
            documents[name] = path.read_bytes()
        except OSError as error:
            raise CompositeRefused(
                f"the attested source of {release_id} is missing its {name} "
                f"at {path} ({error}), so the component it would supply "
                "cannot be shown to have been canaried and the live composite "
                "already serving is left alone"
            ) from error

    parsed: dict[str, Any] = {}

    for name in (
        SOURCE_ROWS_NAME,
        SOURCE_MANIFEST_NAME,
        SOURCE_ATTESTATION_NAME,
    ):
        try:
            document = json.loads(documents[name])
        except ValueError as error:
            raise CompositeRefused(
                f"{directory / name} is not readable JSON ({error}), so the "
                f"stored source of {release_id} cannot be verified"
            ) from error

        if not isinstance(document, dict):
            raise CompositeRefused(
                f"{directory / name} does not hold a JSON object, so the "
                f"stored source of {release_id} cannot be verified"
            )

        parsed[name] = document

    rows_bytes = documents[SOURCE_ROWS_NAME]

    return SourceBundle(
        release_id=str(release_id),
        rows_bytes=rows_bytes,
        rows_document=parsed[SOURCE_ROWS_NAME],
        manifest=parsed[SOURCE_MANIFEST_NAME],
        attestation=parsed[SOURCE_ATTESTATION_NAME],
        rows_sha256=publisher.digest(rows_bytes),
    )


def _release_from_bundle(bundle: SourceBundle) -> publisher.Release:
    """The stored source expressed as the publisher's own release object.

    Built from the recomputed digest so the publisher's canary gate is
    comparing the attestation against the bytes on disk rather than against
    anything either document claims about them.
    """
    return publisher.Release(
        release_id=bundle.release_id,
        rows_relative=str(bundle.manifest.get("rows_relative") or ""),
        manifest_relative="",
        rows_bytes=bundle.rows_bytes,
        manifest_bytes=b"",
        rows_sha256=bundle.rows_sha256,
        manifest_sha256="",
        row_count=len(bundle.rows()),
    )


def assert_source_is_attested(
    bundle: SourceBundle, *, slate_date: str
) -> dict[str, Any]:
    """A canary validated these exact stored bytes, by the publisher's gate.

    :func:`publish_incumbent_to_wizardofodds.assert_canary_attested` is the
    gate, called by name and unweakened, against a release reconstructed from
    the stored bytes and the provenance the stored manifest carries. It is
    applied once per source release rather than once per publication, so a
    composite of four windows passes it four times where a single-window
    publication passed it once.

    What is added around it is the bookkeeping that gate takes for granted
    when there is only one release in play: that the stored manifest is this
    release's own, that it was published under the contract a reader's page is
    rendered against, that the attestation is about this release id, and that
    all of it is this slate's.
    """
    for label, recorded, expected in (
        (
            "release id",
            str(bundle.manifest.get("release_id") or ""),
            bundle.release_id,
        ),
        (
            "rows digest",
            str(bundle.manifest.get("rows_sha256") or ""),
            bundle.rows_sha256,
        ),
        (
            "slate date",
            str(bundle.manifest.get("slate_date") or ""),
            str(slate_date),
        ),
    ):
        if recorded != expected:
            raise CompositeRefused(
                f"the stored manifest of {bundle.release_id} records the "
                f"{label} {recorded or 'nothing'} and the stored bytes are "
                f"{expected}, so the source of a live component is not the "
                "release it claims to be"
            )

    if bundle.manifest.get("contract_version") != (
        publisher.PUBLICATION_CONTRACT_VERSION
    ):
        raise CompositeRefused(
            f"the stored manifest of {bundle.release_id} declares publication "
            f"contract {bundle.manifest.get('contract_version')!r} and live is "
            f"published under {publisher.PUBLICATION_CONTRACT_VERSION}, so "
            "its rows are not ones a reader's page could render"
        )

    if str(bundle.attestation.get("release_id") or "") != bundle.release_id:
        raise CompositeRefused(
            "the stored canary attestation is about "
            f"{bundle.attestation.get('release_id')!r} and the stored bytes "
            f"are {bundle.release_id}, so no canary of this component's "
            "release has been kept"
        )

    try:
        attested = publisher.assert_canary_attested(
            bundle.attestation,
            release=_release_from_bundle(bundle),
            provenance=bundle.provenance,
        )
    except publisher.PublishingRefused as error:
        raise CompositeRefused(
            f"the source release {bundle.release_id} of a live component is "
            f"not canary-attested: {error}"
        ) from error

    return {
        "attested_at": attested.get("attested_at"),
        "release_id": bundle.release_id,
        "rows_sha256": bundle.rows_sha256,
        "source_artifact_sha256": attested.get("source_artifact_sha256"),
    }


# ----------------------------------------------------------------------
# selecting one game out of a release
# ----------------------------------------------------------------------


def rows_of_game(bundle: SourceBundle, game_id: str) -> list[dict[str, Any]]:
    """Every row of one game in one release, in the publisher's own order.

    The rows are returned as they are, which is the whole of what selection
    means here: the objects handed back are the ones parsed out of the stored
    bytes and no field of any of them is read, rewritten or recomputed.
    """
    selected = [
        row
        for row in bundle.rows()
        if isinstance(row, dict) and str(row.get("game_id")) == str(game_id)
    ]

    return sorted(
        selected,
        key=lambda row: tuple(
            str(row.get(name)) for name in publisher.LOGICAL_KEY
        ),
    )


def identities_of_game(
    bundle: SourceBundle, game_id: str, rows: list[dict[str, Any]]
) -> dict[str, str]:
    """``FACTOR_ID`` and ``POLICY_ID`` for one game, out of its own rows.

    Both are read off the rows and the release manifest rather than named
    anywhere in this repository, and both are required to be single-valued
    across the game: a game whose rows disagree about which frozen bundle
    priced them, or under which pricing schema, is not one component and is
    refused rather than described by whichever row happened to sort first.
    """
    factors = {str(row.get("manifest_sha256") or "") for row in rows}

    schemas = {
        str(row.get("market_pricing_schema_version") or "") for row in rows
    }

    for label, values in (
        ("frozen bundle manifest digest", factors),
        ("market pricing schema version", schemas),
    ):
        if len(values) != 1 or "" in values:
            raise CompositeRefused(
                f"the rows of game {game_id} in {bundle.release_id} carry "
                f"{len(values)} distinct {label}(s) "
                f"({', '.join(sorted(value or 'nothing' for value in values))}"
                "), so the component cannot state the artifacts its prices "
                "came out of"
            )

    projection = bundle.manifest.get("approved_projection_schema_version")

    return {
        "factor_id": factors.pop(),
        "policy_id": (
            f"market_pricing_schema_v{schemas.pop()}"
            f"+projection_schema_v{projection}"
        ),
    }


def component_for_game(
    bundle: SourceBundle,
    *,
    game_id: str,
    offset_minutes: int,
    canary_destination: str,
    capture_id: str,
    capture_timestamp: str,
) -> GameComponent:
    """One game's component, assembled out of one attested release.

    Refuses a game the release does not price. A window whose game has no rows
    in the release its own canary published is a game that was not served at
    that window, and advancing it to a window it has no prices at would put an
    empty card on the page.
    """
    rows = rows_of_game(bundle, game_id)

    if not rows:
        raise CompositeRefused(
            f"the canary release {bundle.release_id} prices no rows for game "
            f"{game_id}, so there is nothing of that game's "
            f"{window_label(offset_minutes)} window to put live"
        )

    provenance = bundle.provenance

    identities = identities_of_game(bundle, game_id, rows)

    return GameComponent(
        game_id=str(game_id),
        offset_minutes=int(offset_minutes),
        source_release_id=bundle.release_id,
        source_rows_sha256=bundle.rows_sha256,
        capture_id=str(capture_id),
        capture_timestamp=str(capture_timestamp),
        canary_destination=str(canary_destination),
        production_sha=str(provenance.get("production_code_sha") or ""),
        fit_id=provenance.get("incumbent_fit_id"),
        model_id=str(provenance.get("incumbent_version") or ""),
        factor_id=identities["factor_id"],
        policy_id=identities["policy_id"],
        row_count=len(rows),
        served_at=str(provenance.get("served_at") or ""),
        source_artifact_sha256=str(
            provenance.get("priced_market_artifact_sha256") or ""
        ),
        incumbent_authority=str(provenance.get("incumbent_authority") or ""),
    )


# ----------------------------------------------------------------------
# the composite
# ----------------------------------------------------------------------


@dataclass
class Composite:
    """The rows, the distributions and the audit of one live composition."""

    rows: list[dict[str, Any]] = field(repr=False)
    distributions: dict[str, Any] = field(repr=False)
    components: dict[str, GameComponent]
    audit: dict[str, Any]
    bundles: dict[str, SourceBundle] = field(repr=False, default_factory=dict)
    rows_bytes: bytes = field(repr=False, default=b"")
    rows_sha256: str = ""

    def games_of_source(self, release_id: str) -> set[str]:
        """Every game one source release prices, whether composed or not.

        Read off the stored bytes rather than off the components, because the
        question it answers is whether the composite is the *whole* of that
        release -- which is what decides whether the single-digest identity
        the chain has always asserted is available.
        """
        bundle = self.bundles.get(str(release_id))

        if bundle is None:
            return set()

        return {
            str(row.get("game_id"))
            for row in bundle.rows()
            if isinstance(row, dict)
        }


def compose(
    components: dict[str, GameComponent],
    bundles: dict[str, SourceBundle],
    *,
    slate_date: str,
) -> Composite:
    """Select each game's rows out of its own source and concatenate them.

    The output order is the publisher's own
    :data:`publish_incumbent_to_wizardofodds.LOGICAL_KEY` sort, which is the
    order ``published_rows`` leaves a release in, so a composite of one
    release's whole game set is byte-identical to that release.

    Nothing is computed. Every row object and every distribution record handed
    out here is the object parsed out of a stored, attested document.
    """
    if not components:
        raise CompositeRefused(
            "no game has a validated window, so there is no composite to "
            "publish and an empty live slate is not a publication"
        )

    rows: list[dict[str, Any]] = []

    distributions: dict[str, Any] = {}

    for game_id in sorted(components):
        component = components[game_id]

        bundle = bundles[component.source_release_id]

        selected = rows_of_game(bundle, game_id)

        if not selected:
            raise CompositeRefused(
                f"the attested source {component.source_release_id} of game "
                f"{game_id} holds no rows for it, so that game's "
                f"{window_label(component.offset_minutes)} card cannot be "
                "reassembled and the composite already serving is left alone"
            )

        rows += selected

        source_distributions = bundle.distributions()

        for row in selected:
            key = publisher.distribution_key(row)

            if key not in source_distributions:
                raise CompositeRefused(
                    f"the attested source {component.source_release_id} "
                    f"carries no distribution for {key}, so game {game_id} "
                    "would be published with a price and no forecast behind it"
                )

            distributions[key] = source_distributions[key]

    ordered = sorted(
        rows,
        key=lambda row: tuple(
            str(row.get(name)) for name in publisher.LOGICAL_KEY
        ),
    )

    duplicates = publisher._duplicate_keys(ordered)

    if duplicates:
        raise CompositeRefused(
            "the composite holds more than one row per published quote, so "
            "two games' components overlap: " + "; ".join(duplicates)
        )

    audit = audit_composite(
        ordered,
        distributions,
        components=components,
        bundles=bundles,
        slate_date=slate_date,
    )

    payload = {
        "contract_version": publisher.PUBLICATION_CONTRACT_VERSION,
        publisher.DISTRIBUTIONS_KEY: distributions,
        "rows": ordered,
        "slate_date": str(slate_date),
    }

    rows_bytes = publisher.serialise(payload)

    return Composite(
        rows=ordered,
        distributions=distributions,
        components=components,
        audit=audit,
        bundles=dict(bundles),
        rows_bytes=rows_bytes,
        rows_sha256=publisher.digest(rows_bytes),
    )


def audit_composite(
    rows: list[dict[str, Any]],
    distributions: dict[str, Any],
    *,
    components: dict[str, GameComponent],
    bundles: dict[str, SourceBundle],
    slate_date: str,
) -> dict[str, Any]:
    """Count every composite row that cannot be traced to attested bytes.

    Computed from the source bytes and not from any recorded digest. Each
    source release is read out of the store, its rows digest recomputed by
    :func:`read_source_bundle` and its attestation checked by
    :func:`assert_source_is_attested` before a single row of it is looked at;
    a release that fails either is one whose every row is unattested, and the
    refusal names it.

    Then each row is looked up by the publisher's own logical key in the bytes
    of the release its game names. A row no source holds is unattested. A row
    a source holds and does not match byte for byte is a source mismatch.
    They are counted separately because they are different failures: the first
    is a row from nowhere, and the second is a row that was edited on the way.
    """
    attested_sources: dict[str, dict[str, Any]] = {}

    for release_id in sorted(bundles):
        attested_sources[release_id] = assert_source_is_attested(
            bundles[release_id], slate_date=slate_date
        )

    indexed: dict[str, dict[tuple[str, ...], bytes]] = {}

    for game_id, component in components.items():
        bundle = bundles[component.source_release_id]

        indexed[game_id] = {
            tuple(
                str(row.get(name)) for name in publisher.LOGICAL_KEY
            ): publisher.serialise(row)
            for row in rows_of_game(bundle, game_id)
        }

    unattested: list[str] = []

    mismatched: list[str] = []

    for row in rows:
        game_id = str(row.get("game_id"))

        key = tuple(str(row.get(name)) for name in publisher.LOGICAL_KEY)

        named = " / ".join(key)

        source = indexed.get(game_id)

        if source is None or key not in source:
            unattested.append(named)
            continue

        if publisher.serialise(row) != source[key]:
            mismatched.append(named)

    unattested_distributions: list[str] = []

    for key, record in distributions.items():
        game_id = str(key).split("|")[0]

        component = components.get(game_id)

        if component is None:
            unattested_distributions.append(key)
            continue

        held = bundles[component.source_release_id].distributions()

        if key not in held or publisher.serialise(
            record
        ) != publisher.serialise(held[key]):
            unattested_distributions.append(key)

    return {
        UNATTESTED_DISTRIBUTIONS_FIELD: len(unattested_distributions),
        UNATTESTED_ROWS_FIELD: len(unattested),
        SOURCE_MISMATCH_FIELD: len(mismatched),
        "attested_rows": len(rows) - len(unattested) - len(mismatched),
        "attested_sources": attested_sources,
        "mismatched_rows": sorted(mismatched)[:20],
        "row_count": len(rows),
        "unattested_distributions": sorted(unattested_distributions)[:20],
        "unattested_rows": sorted(unattested)[:20],
    }


def assert_every_row_is_attested(audit: dict[str, Any]) -> None:
    """All three counts are zero, or nothing is published.

    One refusal naming all three, because an operator reading it needs to
    know which kind of untraceable row the composite had rather than only
    that it had one.
    """
    broken = [
        f"{field_name}={audit.get(field_name)}"
        for field_name in (
            UNATTESTED_ROWS_FIELD,
            SOURCE_MISMATCH_FIELD,
            UNATTESTED_DISTRIBUTIONS_FIELD,
        )
        if int(audit.get(field_name) or 0) != 0
    ]

    if not broken:
        return

    raise CompositeRefused(
        "the live composite carries content no canary attested ("
        + ", ".join(broken)
        + "), and a composite with even one row that cannot be traced to a "
        "canary-attested source does not publish. The first offenders are "
        + ", ".join(
            audit.get("unattested_rows", [])[:3]
            + audit.get("mismatched_rows", [])[:3]
            + audit.get("unattested_distributions", [])[:3]
        )
        + ". The composite already serving is left alone"
    )


# ----------------------------------------------------------------------
# what the composite says about itself
# ----------------------------------------------------------------------


def composite_provenance(
    components: dict[str, GameComponent], *, slate_date: str
) -> dict[str, Any]:
    """The provenance the composite release's manifest carries.

    The identity fields have to agree across every component and are refused
    when they do not: the production SHA, the model and the fit are what
    priced the slate, nothing retrains inside an evening, and a composite
    whose games were priced by two different models is not one slate's
    predictions. ``published_authority`` is the publisher's own constant and
    is not reachable from here.

    The two fields that legitimately differ per component are stated as what
    they are rather than as one component's value pretending to be the
    slate's. ``served_at`` is the most recent component's serve, which is when
    what a reader is being shown was last refreshed. ``input_state_fingerprint``
    is a fingerprint of the composite's own inputs -- the digest of the
    per-game source releases -- so two composites with the same fingerprint
    are the same slate assembled out of the same attested bytes.
    """
    identities: dict[str, set[str]] = {}

    for name in (
        "production_sha",
        "model_id",
        "fit_id",
        "incumbent_authority",
    ):
        identities[name] = {
            str(getattr(component, name) or "")
            for component in components.values()
        }

    disagreeing = sorted(
        name for name, values in identities.items() if len(values) > 1
    )

    if disagreeing:
        raise CompositeRefused(
            "the games of this composite were priced under more than one "
            + ", ".join(disagreeing)
            + " ("
            + "; ".join(
                f"{name}: " + ", ".join(sorted(identities[name]))
                for name in disagreeing
            )
            + "), so they are not one slate's predictions and are not "
            "published as one"
        )

    fingerprint = publisher.digest(
        publisher.serialise(
            [
                [
                    component.game_id,
                    window_label(component.offset_minutes),
                    component.source_release_id,
                    component.source_rows_sha256,
                ]
                for component in sorted(
                    components.values(), key=lambda item: item.game_id
                )
            ]
        )
    )

    served = sorted(
        component.served_at
        for component in components.values()
        if component.served_at
    )

    one = next(iter(components.values()))

    return {
        "composite_component_count": len(components),
        "composite_input_state_fingerprint": fingerprint,
        "composite_windows": {
            component.game_id: window_label(component.offset_minutes)
            for component in sorted(
                components.values(), key=lambda item: item.game_id
            )
        },
        "incumbent_authority": one.incumbent_authority,
        "incumbent_fit_id": one.fit_id,
        "incumbent_version": one.model_id,
        "input_state_fingerprint": fingerprint,
        "model_authority": publisher.MODEL_AUTHORITY,
        "production_code_sha": one.production_sha,
        "served_at": served[-1] if served else "",
        "slate_date": str(slate_date),
    }


def composite_manifest(
    composite: Composite,
    *,
    release_id: str,
    rows_sha256: str,
    slate_date: str,
    feed_destination: str,
) -> dict[str, Any]:
    """The composite's own manifest: one entry per game, plus the three counts.

    Written beside the release at the live destination and carried in the
    refresh receipt, so "which window is each game at, out of which canary,
    from which capture" is answerable from the destination alone.
    """
    audit = composite.audit

    return {
        UNATTESTED_DISTRIBUTIONS_FIELD: int(
            audit[UNATTESTED_DISTRIBUTIONS_FIELD]
        ),
        UNATTESTED_ROWS_FIELD: int(audit[UNATTESTED_ROWS_FIELD]),
        SOURCE_MISMATCH_FIELD: int(audit[SOURCE_MISMATCH_FIELD]),
        "attested_rows": int(audit["attested_rows"]),
        "contract_version": COMPOSITE_MANIFEST_CONTRACT_VERSION,
        "feed_destination": str(feed_destination),
        "games": [
            composite.components[game_id].manifest_entry()
            for game_id in sorted(composite.components)
        ],
        "generated_at": _utc_now(),
        "mode": publisher.MODE_LIVE,
        "model_authority": publisher.MODEL_AUTHORITY,
        "release_id": str(release_id),
        "row_count": int(audit["row_count"]),
        "rows_sha256": str(rows_sha256),
        "slate_date": str(slate_date),
        "source_releases": sorted(
            {
                component.source_release_id
                for component in composite.components.values()
            }
        ),
    }


def manifest_relative(release_id: str) -> str:
    return (
        f"{publisher.RELEASES_DIRECTORY}/{release_id}/"
        f"{COMPOSITE_MANIFEST_NAME}"
    )


def write_composite_manifest(
    transport: Any, *, release_id: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Write the composite manifest and read it back off the destination.

    Read back for the reason :func:`publish_incumbent_to_wizardofodds.
    verify_release` reads the release back: a destination that accepted the
    write and kept something else is only visible to a reader that fetches it.
    """
    relative = manifest_relative(release_id)

    payload = publisher.serialise(manifest)

    transport.write_bytes(relative, payload)

    if not transport.exists(relative):
        raise CompositeRefused(
            f"{relative} is not at the live destination after being written, "
            "so the composite's own manifest is missing and the release is "
            "not promoted"
        )

    expected = publisher.digest(payload)

    actual = publisher.digest(transport.read_bytes(relative))

    if actual != expected:
        raise CompositeRefused(
            f"{relative} reads back as {actual} but was written as {expected}, "
            "so the live destination does not hold the composite manifest "
            "that was sent and the release is not promoted"
        )

    return {
        "composite_manifest_relative": relative,
        "composite_manifest_sha256": expected,
    }
