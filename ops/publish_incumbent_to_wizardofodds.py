#!/usr/bin/env python3
"""Publish the incumbent's already-priced slate to a WizardOfOdds destination.

Serving produces two parquet files and a provenance receipt and then stops.
``docs/wizardofodds/PRODUCTION_AUTOMATION.md`` says so in as many words --
"nothing here deploys to, publishes on, or contacts wizardofodds.com. That
belongs to the publishing step that follows Step 3D" -- and until now that
step did not exist. This is it.

WHAT IT PUBLISHES
-----------------

Rows read out of ``processed/priced_markets/<slate>.parquet``, the artifact
``ops/run_incumbent_production_serving.py`` named in its receipt. Nothing is
computed here. Every probability, price, edge and EV is copied across
verbatim, and ``docs/WIZARDOFODDS_INTEGRATION.md`` is explicit that the
consumer "must not recalibrate probabilities" -- a publisher that recomputed
them would be the first thing to break that. There is no model, no copula, no
calibration and no pricing import anywhere in this module, and a test pins
that.

The published row is a projection of the runtime's own columns, under the
runtime's own names, chosen to be the smallest record the destination needs:
who, which game, which prop, which side, which line, the model's probability,
the market's, the posted odds, the fair odds, the edge and EV, the book, and
the deployment the numbers came from. The rest of the priced file is model
internals -- the integration contract says they "are not part of the website
contract" -- and is left where it is.

WHOSE NUMBERS
-------------

The incumbent's, established from the receipt rather than assumed. A receipt
whose ``model_authority`` is not ``incumbent`` is refused, and so is an input
frame carrying any shadow-candidate column, because the way a candidate
probability reaches a feed is not somebody choosing to publish one: it is a
file that looks close enough to the right file.

WHY IT IS DISABLED
------------------

Two modes, switched independently in ``ops/wizardofodds_publishing_switch.json``.

    live
        DISABLED, and not enableable from this repository. Every fact needed
        to describe the destination -- host, directory, protocol, credentials
        -- is external, so ``ops/wizardofodds_destinations.json`` carries
        ``live: null`` and this module refuses live mode by naming the facts
        it does not have. Turning it on additionally needs the committed state
        flipped, the environment variable to agree and a matching approval
        token: three independent things, so no single mistake publishes.

    canary
        ENABLED, because a canary nobody can run proves nothing. It is safe by
        construction rather than by permission: there is no live destination to
        reach, a canary whose destination resolves to the live one is refused,
        and a canary against the real host can only address paths beneath its
        own non-live subdirectory of that host's NBA target.

WHICH DESTINATIONS THERE ARE
----------------------------

A mode decides what a run may do. A name decides which destination of that
mode it does it to, and the two are checked against each other, so a
destination built as a canary is never reinterpreted as a live one.

Unnamed, the canary is a ``local_directory`` under a scratch root the operator
passes per run. That is what proves the publisher's own rules, and it reaches
nothing.

Named with ``--destination-name``, it is a remote destination over
``ssh_rsync``: the one the WizardOfOdds NBA canary is configured for, carrying
its host, port, user, private staging path, served target, public URL and the
*name* of the environment variable its key is read from. No credential is in
this repository and none belongs here. Reaching a real host is therefore
something an operator asks for by name, never something a default does.

The remote path is SSH and rsync and nothing else. No FTP, no FTPS, no SFTP
library, no HTTP API and no object store: each would be a second contract to
write, test and review, and the destination speaks this one.

HOW A REMOTE RELEASE IS WRITTEN
-------------------------------

Bytes arrive where nobody can read them and are proven before they are
served. rsync puts the release in a private staging directory that is mode 700
and outside every document root; the destination is then asked for its digest,
so a transfer that ended early, arrived altered or went somewhere other than
where it was addressed is caught while it is still unreachable. Only a
verified staging file is copied into the served tree, by a rename within the
directory, and its digest is checked again there -- because the digest that
matters is of what is served, not of what was uploaded.

Then the release is read back twice: through the transport, which is the
publisher's existing check, and over its public HTTPS URL, which is the only
check that speaks for the destination as a destination. A file can be correct
on disk and still be unreachable, misaddressed, or shadowed by a web server
rule, and nothing but fetching the public URL can see that. Promotion happens
after both.

WHAT A DESTINATION KEEPS
------------------------

The release it serves, the one before it, and the receipts. Nothing else:
every other release payload is pruned once the pointers have been rotated
past it. Dated snapshots accumulating per day is how the legacy NBA output
became something nobody could reason about, and the audit log and manifests
are kept regardless, so what was published stays readable after the bytes it
describes are gone.

RELEASES, AND GETTING BACK
--------------------------

A release is written whole, verified by reading it back through the transport,
and only then promoted by replacing a pointer. The previous pointer is kept,
so rollback is restoring a pointer to bytes that are still sitting there --
not a rebuild, and nothing to do with the model. Promotion is the last step
because any earlier ordering has a window where the destination serves a
release that was never checked.

Every action that changed a destination is appended to an audit log there, so
what a destination is serving and how it got there are both readable from the
destination itself.

The two records a destination keeps about itself answer different questions
and are maintained differently. ``audit_log.jsonl`` is history: one appended
line per completed action, never rewritten, because rewriting it would
falsify what was published when. ``canary_attestation.json`` is current state:
it describes the release the destination is serving now, so it is rewritten
whenever that stops being what it says -- including by a rerun that promoted
nothing, since the release id is the rows digest and a release whose
provenance was corrected republishes under the same id.

WHAT ACTIVATION WOULD TAKE
--------------------------

Activation is ``--mode live``, and it is gated on seven independent facts, all
checked before a byte is written: the mode is named explicitly, the committed
switch enables it, the receipt's authority is the incumbent, the environment
carries the matching approval token, a successful canary of this same source
artifact is attested, the release verifies at the destination, and a rollback
target is already there to return to.

The canary gate is matched on two digests -- the source artifact's and the
published rows' -- so an activation cannot inherit a canary that proved
something else. The rollback gate means no activation is the step that leaves
a destination with nowhere to go back to, which includes the first one: before
a destination is activated against for the first time, its rollback target has
to be seeded from whatever it is already serving.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_AUTHORITY = "incumbent"
CANDIDATE_AUTHORITY = "shadow_only"

MODE_CANARY = "canary"
MODE_LIVE = "live"
MODES: tuple[str, ...] = (MODE_CANARY, MODE_LIVE)

STATE_DISABLED = "DISABLED"
STATE_ENABLED = "ENABLED"

#: Enabling live publication needs all three: the committed switch set to
#: ENABLED, this variable set to the same state, and a separately supplied
#: token matching the switch's own. The shape is taken from the shadow
#: publishing switch, which already works this way.
PUBLISH_ENV_VAR = "NBA_PROP_WOO_PUBLISH"
PUBLISH_APPROVAL_ENV_VAR = "NBA_PROP_WOO_PUBLISH_APPROVAL"

SWITCH_RELATIVE = Path("ops") / "wizardofodds_publishing_switch.json"
DESTINATIONS_RELATIVE = Path("ops") / "wizardofodds_destinations.json"

#: Where serving leaves the artifact and the receipt, relative to the data
#: root. The same two locations ``ops/grade_incumbent_production_slate.py``
#: reads, so the publisher and the grader provenance the same bytes.
PRICED_MARKETS_RELATIVE = Path("processed") / "priced_markets"
SERVING_RECEIPTS_RELATIVE = Path("processed") / "incumbent_serving"

TRANSPORT_LOCAL_DIRECTORY = "local_directory"
TRANSPORT_SSH_RSYNC = "ssh_rsync"

#: Destinations addressed by name rather than by mode. A mode still decides
#: what a run is allowed to do; a name decides which of several destinations
#: of that mode it does it to, so reaching a real host is something an
#: operator asks for explicitly instead of inheriting from a default.
NAMED_DESTINATIONS_KEY = "named_destinations"

#: Where an ``ssh_rsync`` destination lands bytes before they are published:
#: a subdirectory of the private staging root, which is mode 700 and outside
#: any document root. Uploads are verified here, and only a verified upload is
#: copied into the served tree.
STAGING_SUBDIRECTORY = "releases_staging"

SSH_CONNECT_TIMEOUT_SECONDS = 20
SSH_COMMAND_TIMEOUT_SECONDS = 180
PUBLIC_READBACK_TIMEOUT_SECONDS = 60

#: Sentinel ``sha256sum`` is replaced by when the path is not there, so a
#: missing file and an empty one are distinguishable. An absent upload is a
#: partial release; a file that hashes to the digest of nothing is a different
#: defect, and a transport that reported both as "" could not tell them apart.
REMOTE_MISSING = "MISSING"

#: The published record's own version. Independent of the two feed versions:
#: this names the shape of what is written to the destination, not the shape
#: of what was read from the runtime.
PUBLICATION_CONTRACT_VERSION = 1

RELEASES_DIRECTORY = "releases"
ROWS_NAME = "rows.json"
MANIFEST_NAME = "manifest.json"
CURRENT_POINTER = "current.json"
PREVIOUS_POINTER = "previous.json"

#: What a successful canary leaves at its own destination, and what a later
#: activation is required to be handed. It ties a canary run to the exact
#: source artifact it published, so "this was canaried first" is a fact read
#: off a file rather than an operator's recollection.
ATTESTATION_NAME = "canary_attestation.json"

#: Append-only record of every action that changed a destination. Written
#: after the action completes, so a refusal leaves no entry and no bytes.
AUDIT_LOG_NAME = "audit_log.jsonl"

OUTCOME_PUBLISHED = "PUBLISHED"
OUTCOME_ROLLED_BACK = "ROLLED_BACK"

ACTION_PUBLISHED = "PUBLISHED"
ACTION_REPUBLISHED = "REPUBLISHED_NO_OP"
ACTION_ROLLED_BACK = "ROLLED_BACK"

EXIT_OK = 0
EXIT_REFUSED = 1

#: The receipt fields a publication is provenanced by. ``incumbent_fit_id`` is
#: required to be present rather than non-empty for the same reason serving
#: requires it that way: null is the true value before the first human
#: promotion, and demanding an id there would only invite inventing one.
REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "generated_at",
    "incumbent_authority",
    "incumbent_version",
    "input_state_fingerprint",
    "model_authority",
    "priced_market_artifact",
    "production_code_sha",
    "slate_date",
)

REQUIRED_PROVENANCE_KEYS: tuple[str, ...] = ("incumbent_fit_id",)

#: Shadow-candidate columns, by the exact names
#: ``shadow_runtime.ShadowDecision.payload`` emits. An input frame carrying
#: any of them is a shadow log rather than a priced slate, and is refused on
#: that basis rather than on the strength of a label elsewhere saying so.
CANDIDATE_SOURCE_COLUMNS: tuple[str, ...] = (
    "candidate_independent_product",
    "candidate_probability",
    "candidate_standard_error",
    "incumbent_probability",
    "independence_probability",
    "served_model",
    "served_probability",
)

#: The three props whose Gate 3 routing changed, and which the integration
#: contract therefore holds to ``gate3_role_ready``.
GATE3_PROPS: frozenset[str] = frozenset(
    {"assists", "points_assists", "points_rebounds"}
)

#: What identifies one published row. Two rows sharing this are the same
#: quote published twice, whatever else differs about them.
LOGICAL_KEY: tuple[str, ...] = (
    "vendor",
    "game_id",
    "player_id",
    "prop_type",
    "line_value",
)

#: Source column -> published field. The published name is the runtime's own
#: in every case: renaming on the way out would make the destination's
#: vocabulary diverge from the contract that documents it.
PUBLISHED_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "quote_id"),
    ("vendor", "vendor"),
    ("game_id", "game_id"),
    ("date", "date"),
    ("player_id", "player_id"),
    ("player_name", "player_name"),
    ("team_id", "team_id"),
    ("opponent_id", "opponent_id"),
    ("is_home", "is_home"),
    ("prop_type", "prop_type"),
    ("line_value", "line_value"),
    ("market_type", "market_type"),
    ("over_odds", "over_odds"),
    ("under_odds", "under_odds"),
    ("calibrated_q_over_nonpush", "calibrated_q_over_nonpush"),
    ("calibrated_q_under_nonpush", "calibrated_q_under_nonpush"),
    ("calibrated_p_push", "calibrated_p_push"),
    ("calibration_method", "calibration_method"),
    ("market_devig_q_over", "market_devig_q_over"),
    ("market_devig_q_under", "market_devig_q_under"),
    ("market_hold_pct", "market_hold_pct"),
    ("calibrated_fair_over_american", "calibrated_fair_over_american"),
    ("calibrated_fair_under_american", "calibrated_fair_under_american"),
    ("calibrated_edge_over", "calibrated_edge_over"),
    ("calibrated_edge_under", "calibrated_edge_under"),
    ("calibrated_ev_over", "calibrated_ev_over"),
    ("calibrated_ev_under", "calibrated_ev_under"),
    ("model_preferred_side", "model_preferred_side"),
    ("model_preferred_edge", "model_preferred_edge"),
    ("model_preferred_ev", "model_preferred_ev"),
    ("freeze_id", "freeze_id"),
    ("freeze_stage", "freeze_stage"),
    ("manifest_sha256", "manifest_sha256"),
    ("market_pricing_schema_version", "market_pricing_schema_version"),
)

#: Columns read to decide whether a row may be published at all, which are not
#: themselves published. Named so the schema check covers them: a priced file
#: missing one of these cannot be screened, and publishing it unscreened would
#: be the fail-closed rules quietly not applying.
SCREENED_COLUMNS: tuple[str, ...] = (
    "advanced_latest_date",
    "availability_out",
    "gate3_role_ready",
    "history_latest_date",
    "projection_schema_version",
    "quote_eligible",
    "quote_filter_reason",
)

#: The probability fields the destination is given. Pinned as a list so a test
#: can assert each is copied from the source byte for byte, and so the set
#: cannot grow to include a ``raw_*`` field: the integration contract says
#: those "should not be published as the model's price".
PUBLISHED_PROBABILITY_FIELDS: tuple[str, ...] = (
    "calibrated_p_push",
    "calibrated_q_over_nonpush",
    "calibrated_q_under_nonpush",
    "market_devig_q_over",
    "market_devig_q_under",
)


class PublishingRefused(RuntimeError):
    """The publisher declined to write anything."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def serialise(payload: Any) -> bytes:
    """One object, one byte string, always.

    Sorted keys and a fixed separator, so two runs over the same rows produce
    the same bytes and the destination's digest is a statement about the rows
    rather than about dictionary ordering.
    """
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)

    return sha.hexdigest()


# ----------------------------------------------------------------------
# the switch
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PublishingSwitch:
    """One mode's committed state, and whether this process may use it."""

    mode: str
    state: str
    reason: str
    blockers: tuple[str, ...] = ()
    approved_freeze_id: str = ""
    approved_freeze_stage: str = ""
    approved_pricing_schema_version: int | None = None
    approved_projection_schema_version: int | None = None
    published_authority: str = ""
    candidate_authority: str = ""
    source: str = ""

    @property
    def enabled(self) -> bool:
        return self.state == STATE_ENABLED and not self.blockers

    def payload(self) -> dict[str, Any]:
        return {
            "approved_freeze_id": self.approved_freeze_id,
            "approved_freeze_stage": self.approved_freeze_stage,
            "approved_market_pricing_schema_version": (
                self.approved_pricing_schema_version
            ),
            "approved_projection_schema_version": (
                self.approved_projection_schema_version
            ),
            "blockers": list(self.blockers),
            "candidate_authority": self.candidate_authority,
            "enabled": self.enabled,
            "mode": self.mode,
            "published_authority": self.published_authority,
            "reason": self.reason,
            "source": self.source,
            "state": self.state,
        }


def read_publishing_switch(
    mode: str,
    *,
    project_root: Path | None = None,
    switch_path: Path | None = None,
    environment: dict[str, str] | None = None,
) -> PublishingSwitch:
    """The committed state for one mode, plus what this process is missing.

    A missing or unreadable file reads as disabled. The absence of a
    prohibition is not an approval, and a publisher that treated an unparseable
    switch as permission would be exactly the single mistake the three
    conditions exist to rule out.

    The environment conditions are asked of live only. For the canary they
    would be ceremony: the canary's safety is that there is no live
    destination for it to reach and the transport only writes local
    directories, not that somebody exported the right variable.
    """
    if mode not in MODES:
        raise PublishingRefused(
            f"{mode!r} is not a publishing mode; expected one of {list(MODES)}"
        )

    root = Path(project_root or PROJECT_ROOT)

    path = Path(switch_path) if switch_path is not None else root / SWITCH_RELATIVE

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return PublishingSwitch(
            mode=mode,
            state=STATE_DISABLED,
            reason=(
                f"the publishing switch at {path} could not be read ({error}), "
                "and an unreadable switch is not an approval"
            ),
            source=str(path),
        )

    declared = ((document.get("modes") or {}).get(mode)) or {}

    state = str(declared.get("state") or STATE_DISABLED)

    token = str(declared.get("approval_token") or "")

    blockers: list[str] = []

    if mode == MODE_LIVE:
        env = dict(os.environ if environment is None else environment)

        if env.get(PUBLISH_ENV_VAR) != state:
            blockers.append(
                f"{PUBLISH_ENV_VAR} does not match the switch state"
            )

        if not token or env.get(PUBLISH_APPROVAL_ENV_VAR) != token:
            blockers.append(
                f"{PUBLISH_APPROVAL_ENV_VAR} does not match the approval token"
            )

    version = document.get("approved_market_pricing_schema_version")
    projection_version = document.get("approved_projection_schema_version")

    return PublishingSwitch(
        mode=mode,
        state=state,
        reason=str(declared.get("reason") or ""),
        blockers=tuple(blockers),
        approved_freeze_id=str(document.get("approved_freeze_id") or ""),
        approved_freeze_stage=str(document.get("approved_freeze_stage") or ""),
        approved_pricing_schema_version=(
            int(version) if version is not None else None
        ),
        approved_projection_schema_version=(
            int(projection_version) if projection_version is not None else None
        ),
        published_authority=str(document.get("published_authority") or ""),
        candidate_authority=str(document.get("candidate_authority") or ""),
        source=str(path),
    )


def assert_mode_enabled(switch: PublishingSwitch) -> None:
    if switch.state != STATE_ENABLED:
        raise PublishingRefused(
            f"{switch.mode} publishing is {switch.state} in {switch.source}: "
            f"{switch.reason or 'no reason recorded'}"
        )

    if switch.blockers:
        raise PublishingRefused(
            f"{switch.mode} publishing is enabled in {switch.source} but this "
            "process does not carry its approval: " + "; ".join(switch.blockers)
        )

    if switch.published_authority != MODEL_AUTHORITY:
        raise PublishingRefused(
            f"{switch.source} records the published authority as "
            f"{switch.published_authority!r}, not {MODEL_AUTHORITY!r}, so this "
            "publisher has nothing it is allowed to publish"
        )

    if switch.candidate_authority != CANDIDATE_AUTHORITY:
        raise PublishingRefused(
            f"{switch.source} records the candidate authority as "
            f"{switch.candidate_authority!r}, not {CANDIDATE_AUTHORITY!r}"
        )

    if not switch.approved_freeze_id:
        raise PublishingRefused(
            f"{switch.source} names no approved freeze id, so there is no "
            "deployment this publication is approved for"
        )

    if switch.approved_pricing_schema_version is None:
        raise PublishingRefused(
            f"{switch.source} names no approved market pricing schema version"
        )


# ----------------------------------------------------------------------
# the destination
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteDestination:
    """Where an ``ssh_rsync`` destination is, and what it may touch.

    Two paths rather than one, because publishing to a served directory and
    uploading to it are different operations with different exposure. Bytes
    arrive in ``private_staging`` -- mode 700, outside every document root --
    and are verified there. ``public_target`` is served, so nothing reaches it
    that has not already been checked where nobody can read it.

    ``canary_subpath`` is what keeps a canary a canary. Every path this
    destination addresses is resolved beneath ``public_target/canary_subpath``,
    so the destination cannot name the served root, cannot name anything
    beside it, and has no way to express the legacy NBA pages and feeds that
    sit in the parent directory. A canary that is non-live because of where it
    can write is a stronger statement than one that is non-live because of
    what it was asked to write.

    No credential is held here. ``credential_env`` is the name of the
    variable the private key is read from, which is a fact about the
    deployment; the key is a fact about the environment and never about this
    repository.
    """

    host: str
    port: int
    user: str
    credential_env: str
    private_staging: str
    public_target: str
    public_base_url: str
    canary_subpath: str
    retained_releases: int | None

    @property
    def published_root(self) -> str:
        """The one directory this destination writes, served."""
        return f"{self.public_target.rstrip('/')}/{self.canary_subpath.strip('/')}"

    @property
    def staging_root(self) -> str:
        """The one directory this destination uploads to, unserved."""
        return (
            f"{self.private_staging.rstrip('/')}/{STAGING_SUBDIRECTORY}/"
            f"{self.canary_subpath.strip('/')}"
        )

    @property
    def published_base_url(self) -> str:
        return (
            self.public_base_url.rstrip("/")
            + "/"
            + self.canary_subpath.strip("/")
            + "/"
        )


@dataclass(frozen=True)
class Destination:
    mode: str
    transport: str
    root: Path
    release_prefix: str
    source: str
    name: str = ""
    remote: RemoteDestination | None = None
    retained_releases: int | None = None


def read_destinations(
    *,
    project_root: Path | None = None,
    destinations_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or PROJECT_ROOT)

    path = (
        Path(destinations_path)
        if destinations_path is not None
        else root / DESTINATIONS_RELATIVE
    )

    try:
        return {
            **json.loads(path.read_text(encoding="utf-8")),
            "source": str(path),
        }
    except (OSError, ValueError) as error:
        raise PublishingRefused(
            f"the destination configuration at {path} could not be read "
            f"({error}), so there is nowhere this publisher is configured to "
            "write"
        ) from error


def resolve_destination(
    mode: str,
    document: dict[str, Any],
    *,
    root: Path | None = None,
    name: str | None = None,
) -> Destination:
    """The destination for one mode, or a refusal naming what is missing.

    Without a name this is the mode's own destination, which is how it has
    always worked. ``live`` is null in the committed configuration, and the
    refusal quotes the facts that are still missing for it rather than saying
    "not configured", because the difference between a missing line of config
    and a missing agreement is the whole difference between finishing and
    being blocked.

    With a name it is one of the named destinations, and the name has to agree
    with the mode. A named destination declares which mode it belongs to, so
    pointing a live run at a destination built as a canary is a refusal rather
    than a reinterpretation.
    """
    source = str(document.get("source") or DESTINATIONS_RELATIVE)

    if name:
        return _resolve_named_destination(mode, document, name=name, source=source)

    declared = (document.get("destinations") or {}).get(mode)

    if not declared:
        missing = document.get("missing_external_facts") or []

        detail = (
            " The facts it would need are external to this repository: "
            + "; ".join(str(fact) for fact in missing)
            if missing
            else ""
        )

        raise PublishingRefused(
            f"{source} configures no {mode} destination, so there is nowhere "
            f"to publish.{detail}"
        )

    transport = _supported_transport(declared, document, mode=mode, source=source)

    if transport != TRANSPORT_LOCAL_DIRECTORY:
        raise PublishingRefused(
            f"{source} asks for the {transport!r} transport for the unnamed "
            f"{mode} destination. A remote destination is addressed by name, "
            "with --destination-name, so reaching a real host is never the "
            "default"
        )

    resolved = root if root is not None else declared.get("root")

    if not resolved:
        raise PublishingRefused(
            f"no {mode} destination root was supplied and {source} declares "
            "none, so there is nowhere to write"
        )

    return Destination(
        mode=mode,
        transport=transport,
        root=Path(resolved),
        release_prefix=str(declared.get("release_prefix") or "nba_prop_quant"),
        source=source,
    )


def _supported_transport(
    declared: dict[str, Any],
    document: dict[str, Any],
    *,
    mode: str,
    source: str,
) -> str:
    transport = str(declared.get("transport") or "")

    supported = tuple(document.get("supported_transports") or ())

    if transport not in supported:
        raise PublishingRefused(
            f"{source} asks for the {transport!r} transport for {mode}, and "
            f"only {list(supported)} is implemented"
        )

    return transport


def _resolve_named_destination(
    mode: str,
    document: dict[str, Any],
    *,
    name: str,
    source: str,
) -> Destination:
    named = document.get(NAMED_DESTINATIONS_KEY) or {}

    declared = named.get(name)

    if not declared:
        raise PublishingRefused(
            f"{source} configures no destination named {name!r}; it names "
            f"{sorted(named)}"
        )

    declared_mode = str(declared.get("mode") or "")

    if declared_mode != mode:
        raise PublishingRefused(
            f"{source} declares {name!r} a {declared_mode or 'modeless'} "
            f"destination, and this run is {mode}. A destination built for one "
            "mode is not reinterpreted as another"
        )

    transport = _supported_transport(declared, document, mode=mode, source=source)

    if transport != TRANSPORT_SSH_RSYNC:
        raise PublishingRefused(
            f"{source} asks for the {transport!r} transport for {name!r}, and "
            f"a named destination is remote: only {TRANSPORT_SSH_RSYNC!r} is "
            "implemented for one"
        )

    remote = _read_remote(declared, name=name, source=source)

    return Destination(
        mode=mode,
        transport=transport,
        root=Path(remote.published_root),
        release_prefix=str(declared.get("release_prefix") or "nba_prop_quant"),
        source=source,
        name=name,
        remote=remote,
        retained_releases=remote.retained_releases,
    )


#: Every field an ``ssh_rsync`` destination is described by. Required rather
#: than defaulted: a remote destination with a guessed user, a guessed path or
#: a guessed credential name is a destination that looks configured and writes
#: somewhere nobody chose.
REQUIRED_REMOTE_FIELDS: tuple[str, ...] = (
    "canary_subpath",
    "credential_env",
    "host",
    "port",
    "private_staging",
    "public_base_url",
    "public_target",
    "user",
)


def _read_remote(
    declared: dict[str, Any], *, name: str, source: str
) -> RemoteDestination:
    missing = [
        field_name
        for field_name in REQUIRED_REMOTE_FIELDS
        if declared.get(field_name) in (None, "", [], {})
    ]

    if missing:
        raise PublishingRefused(
            f"{source} describes {name!r} without " + ", ".join(sorted(missing))
        )

    try:
        port = int(declared["port"])
    except (TypeError, ValueError) as error:
        raise PublishingRefused(
            f"{source} gives {name!r} the port {declared.get('port')!r}, "
            "which is not a port number"
        ) from error

    for field_name in ("private_staging", "public_target"):
        value = str(declared[field_name])

        if not value.startswith("/"):
            raise PublishingRefused(
                f"{source} gives {name!r} the relative {field_name} "
                f"{value!r}; a remote path is resolved against whatever "
                "directory the login happens to start in, so it has to be "
                "absolute"
            )

    base_url = str(declared["public_base_url"])

    if not base_url.startswith("https://"):
        raise PublishingRefused(
            f"{source} gives {name!r} the public base URL {base_url!r}. The "
            "readback is the step that proves what the destination serves, "
            "and reading it back over plaintext would prove it against "
            "whatever answered"
        )

    declared_subpath = str(declared["canary_subpath"])

    subpath = declared_subpath.rstrip("/")

    # A leading slash is refused rather than stripped. It reads as an absolute
    # path, and somebody who wrote one meant a location rather than a
    # subdirectory: normalising it would silently publish somewhere other than
    # where the configuration appears to say.
    if (
        declared_subpath.startswith("/")
        or not subpath
        or not _is_safe_remote_segment_path(subpath)
    ):
        raise PublishingRefused(
            f"{source} gives {name!r} the canary subpath "
            f"{declared_subpath!r}. It is the only thing confining this "
            "destination to a non-live directory, so it has to be a plain "
            "relative path below the served root and cannot be empty"
        )

    retained = declared.get("retained_releases")

    if retained is not None:
        try:
            retained = int(retained)
        except (TypeError, ValueError) as error:
            raise PublishingRefused(
                f"{source} gives {name!r} the retention "
                f"{declared.get('retained_releases')!r}, which is not a count"
            ) from error

        if retained < 2:
            raise PublishingRefused(
                f"{source} asks {name!r} to retain {retained} releases. The "
                "release currently served and the one before it are both "
                "required: retaining fewer would prune the rollback target"
            )

    return RemoteDestination(
        host=str(declared["host"]),
        port=port,
        user=str(declared["user"]),
        credential_env=str(declared["credential_env"]),
        private_staging=str(declared["private_staging"]).rstrip("/"),
        public_target=str(declared["public_target"]).rstrip("/"),
        public_base_url=base_url,
        canary_subpath=subpath,
        retained_releases=retained,
    )


def _is_safe_remote_segment_path(relative: str) -> bool:
    """Whether a relative path addresses something strictly below its root.

    No absolute paths, no ``..``, no ``.``, no empty segments and no
    backslashes. The published tree is the one place this publisher writes on
    a host that also serves the legacy NBA pages, and a path that could climb
    out of it would make the confinement a convention rather than a property.
    """
    if not relative or relative.startswith("/") or "\\" in relative:
        return False

    segments = relative.split("/")

    return all(
        segment not in ("", ".", "..") and not segment.startswith("-")
        for segment in segments
    )


def assert_remote_canary_is_confined(destination: Destination) -> None:
    """A remote canary writes below the served root, never at it.

    The served root holds the legacy NBA feed the website already publishes.
    This publisher never addresses it, and the check is here rather than left
    implicit because "the subpath is not empty" is the entire reason that is
    true.
    """
    remote = destination.remote

    if remote is None:
        return

    published = remote.published_root.rstrip("/")

    target = remote.public_target.rstrip("/")

    if published == target or not published.startswith(target + "/"):
        raise PublishingRefused(
            f"the remote canary would publish to {published}, which is not "
            f"strictly below the served root {target}, so it is not confined "
            "to a non-live directory"
        )


def assert_canary_is_not_live(canary: Destination, document: dict[str, Any]) -> None:
    """A canary pointed at the live destination is not a canary."""
    live = (document.get("destinations") or {}).get(MODE_LIVE) or {}

    live_root = live.get("root")

    if not live_root:
        return

    if Path(live_root).resolve() == canary.root.resolve():
        raise PublishingRefused(
            f"the canary destination {canary.root} is the live destination, so "
            "running it would alter live output"
        )


class LocalDirectoryTransport:
    """Bytes in a directory tree, with an atomic pointer swap.

    The only transport implemented, and deliberately so: which protocol the
    real destination speaks is a fact this repository does not have, and
    guessing wrong would mean writing, testing and reviewing a transport
    against an invented contract. What this one is for is proving the
    publisher, the verification and the rollback, all of which are
    transport-independent.

    ``os.replace`` is the atomic rename the release contract asks for, so
    promotion is one syscall and a reader either sees the old pointer or the
    new one.
    """

    name = TRANSPORT_LOCAL_DIRECTORY

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _resolve(self, relative: str) -> Path:
        return self.root / relative

    def write_bytes(self, relative: str, payload: bytes) -> None:
        target = self._resolve(relative)

        target.parent.mkdir(parents=True, exist_ok=True)

        staging = target.with_name(target.name + ".partial")

        staging.write_bytes(payload)

        os.replace(staging, target)

    def read_bytes(self, relative: str) -> bytes:
        return self._resolve(relative).read_bytes()

    def exists(self, relative: str) -> bool:
        return self._resolve(relative).exists()

    def listing(self) -> list[str]:
        if not self.root.exists():
            return []

        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        )

    def remove_tree(self, relative: str) -> None:
        shutil.rmtree(self._resolve(relative), ignore_errors=True)

    def close(self) -> None:
        return None


class RemoteCommandFailed(PublishingRefused):
    """A command at the destination did not succeed."""


class SshRsyncTransport:
    """Bytes to a served directory on a real host, over SSH and rsync.

    Three hops, because the destination is a live web server and the only
    orderings that are safe on one are the ones where nothing unverified is
    ever served.

    First the release is written to a local spool, which is what gives rsync
    something whole to send. Then rsync puts it in ``private_staging`` -- mode
    700, outside every document root -- and the upload is verified there by
    asking the destination itself for the digest. A transfer that ended early,
    went somewhere other than where it was addressed, or arrived altered is
    caught at this point, with the bytes still unreachable by anybody.

    Only then is the file copied into the served tree, and the copy is a
    ``mv`` from a temporary name in the destination directory, so a reader
    sees the old file or the new one and never a partial one. The digest is
    checked again after the move, because the check that matters is of what is
    served rather than of what was uploaded.

    ``read_bytes`` reads the served file back over SSH, which is what makes
    the publisher's existing verification a statement about this transport
    too. ``public_readback`` reads it over HTTPS, which is the only check that
    speaks for what a consumer would actually get: a file can be correct on
    disk and unreachable, misaddressed or shadowed by a rule in the web
    server's configuration, and nothing but a fetch of the public URL can see
    that.

    The private key never reaches the filesystem the repository is checked out
    on. It is read from the environment variable the destination names, into a
    file in a 700 temporary directory that ``close`` removes.
    """

    name = TRANSPORT_SSH_RSYNC

    def __init__(
        self,
        destination: Destination,
        *,
        environment: dict[str, str] | None = None,
    ) -> None:
        remote = destination.remote

        if remote is None:
            raise PublishingRefused(
                "an ssh_rsync destination was resolved without the facts "
                "describing where it is"
            )

        self.destination = destination

        self.remote = remote

        self.root = Path(remote.published_root)

        for executable in ("ssh", "rsync"):
            if shutil.which(executable) is None:
                raise PublishingRefused(
                    f"{executable} is not on PATH, and this transport is SSH "
                    "and rsync only"
                )

        env = dict(os.environ if environment is None else environment)

        material = env.get(remote.credential_env) or ""

        if not material.strip():
            raise PublishingRefused(
                f"{remote.credential_env} carries no private key, so there is "
                f"no credential to authenticate to {remote.user}@{remote.host} "
                "with. The key is supplied by the environment and is never "
                "committed"
            )

        self._workspace = Path(tempfile.mkdtemp(prefix="nba_woo_ssh_rsync_"))

        os.chmod(self._workspace, 0o700)

        self._key_path = self._workspace / "credential"

        self._key_path.touch(mode=0o600)

        self._key_path.write_text(
            material if material.endswith("\n") else material + "\n",
            encoding="utf-8",
        )

        os.chmod(self._key_path, 0o600)

        self._known_hosts = self._workspace / "known_hosts"

        self._known_hosts.touch(mode=0o600)

        self._spool = self._workspace / "spool"

        self._spool.mkdir(mode=0o700)

        self._closed = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Remove the key, the known hosts file and the spool."""
        if not self._closed:
            shutil.rmtree(self._workspace, ignore_errors=True)

            self._closed = True

    def __enter__(self) -> SshRsyncTransport:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # addressing
    # ------------------------------------------------------------------

    def _relative(self, relative: str) -> str:
        if not _is_safe_remote_segment_path(relative):
            raise PublishingRefused(
                f"{relative!r} does not address a path below "
                f"{self.remote.published_root}, so this transport will not "
                "write it"
            )

        return relative

    def _published(self, relative: str) -> str:
        return f"{self.remote.published_root}/{self._relative(relative)}"

    def _staged(self, relative: str) -> str:
        return f"{self.remote.staging_root}/{self._relative(relative)}"

    # ------------------------------------------------------------------
    # the remote
    # ------------------------------------------------------------------

    def _ssh_argv(self) -> list[str]:
        return [
            "ssh",
            "-i",
            str(self._key_path),
            "-p",
            str(self.remote.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self._known_hosts}",
        ]

    def _assert_open(self) -> None:
        """Refuse to act through a transport whose credential has been removed.

        ``close`` deletes the key and the spool, which makes every later
        operation fail somewhere further in -- at an ssh that cannot read its
        identity file, or at a spool directory that gets silently recreated
        without one. Both are confusing in a way the real cause is not, and a
        second operation on a destination wants a second transport.
        """
        if self._closed:
            raise PublishingRefused(
                "this transport has been closed and its credential removed; "
                "a further operation on the destination needs a transport of "
                "its own"
            )

    def _run_remote(self, script: str) -> str:
        """One shell script at the destination, or a refusal carrying its error.

        ``set -eu`` so a script's first failing command is the script's
        failure. The destination's own stderr is quoted in the refusal: an
        authentication failure, a permission denial and a missing directory
        are different problems with different fixes, and a publisher that
        reported them all as "the remote command failed" would make the
        operator guess which one it hit.
        """
        self._assert_open()

        completed = subprocess.run(
            [
                *self._ssh_argv(),
                f"{self.remote.user}@{self.remote.host}",
                f"set -eu; {script}",
            ],
            capture_output=True,
            text=True,
            timeout=SSH_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )

        if completed.returncode != 0:
            raise RemoteCommandFailed(
                f"{self.remote.user}@{self.remote.host} refused a command "
                f"(exit {completed.returncode}): "
                f"{(completed.stderr or completed.stdout).strip()}"
            )

        return completed.stdout

    def verify_identity(self) -> dict[str, Any]:
        """Who the destination thinks this is, and what it holds.

        Asked before anything is written. The user is the one fact about a
        deployment that a wrong answer to is silent: a key that authenticates
        as somebody else succeeds, writes files owned by that somebody, and
        looks exactly like a working publication until the ownership matters.
        """
        reported = self._run_remote("id -un; pwd").splitlines()

        user = reported[0].strip() if reported else ""

        if user != self.remote.user:
            raise PublishingRefused(
                f"{self.remote.host} authenticated this key as {user!r}, not "
                f"the {self.remote.user!r} the destination is configured for. "
                "Publishing as another account would write files that account "
                "owns"
            )

        return {
            "host": self.remote.host,
            "port": self.remote.port,
            "private_staging": self.remote.private_staging,
            "public_target": self.remote.public_target,
            "published_root": self.remote.published_root,
            "user": user,
        }

    def _remote_digest(self, path: str) -> str:
        quoted = shlex.quote(path)

        output = self._run_remote(
            f"if [ -f {quoted} ]; then sha256sum -- {quoted} | cut -d' ' -f1; "
            f"else printf '%s\\n' {REMOTE_MISSING}; fi"
        )

        return output.strip()

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def write_bytes(self, relative: str, payload: bytes) -> None:
        self._assert_open()

        safe = self._relative(relative)

        expected = digest(payload)

        local = self._spool / safe

        local.parent.mkdir(parents=True, exist_ok=True)

        local.write_bytes(payload)

        self._upload(safe, local)

        staged = self._remote_digest(self._staged(safe))

        if staged != expected:
            raise PublishingRefused(
                f"{safe} uploaded to {self.remote.staging_root} hashes to "
                f"{staged}, not the {expected} that was sent, so the upload is "
                "partial or altered and nothing is published from it"
            )

        self._place(safe)

        served = self._remote_digest(self._published(safe))

        if served != expected:
            raise PublishingRefused(
                f"{safe} at {self.remote.published_root} hashes to {served}, "
                f"not the {expected} that was staged, so the destination does "
                "not serve what was verified"
            )

    def _upload(self, safe: str, local: Path) -> None:
        """rsync one file into the private staging tree.

        No ``--partial``: rsync writes a temporary and renames it when the
        transfer completes, so an interrupted transfer leaves the staging path
        absent rather than short. ``--checksum`` so a staging file that
        already exists is compared by content rather than by size and
        timestamp, which is the comparison a rerun needs.
        """
        target = self._staged(safe)

        parent = target.rsplit("/", 1)[0]

        self._run_remote(f"mkdir -p -- {shlex.quote(parent)}")

        completed = subprocess.run(
            [
                "rsync",
                "--checksum",
                "--times",
                "--chmod=F600",
                "--no-perms",
                "-e",
                shlex.join(self._ssh_argv()),
                "--",
                str(local),
                f"{self.remote.user}@{self.remote.host}:{target}",
            ],
            capture_output=True,
            text=True,
            timeout=SSH_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )

        if completed.returncode != 0:
            raise RemoteCommandFailed(
                f"rsync of {safe} to {self.remote.user}@{self.remote.host} "
                f"failed (exit {completed.returncode}): "
                f"{(completed.stderr or completed.stdout).strip()}"
            )

    def _place(self, safe: str) -> None:
        """Copy a verified staging file into the served tree, atomically.

        ``cp`` to a temporary name in the destination directory and then
        ``mv``, which is a rename within one filesystem: a reader of the
        served path sees the file that was there before this call or the one
        after it. Copying straight onto the served path would leave a window
        in which the file served is neither.
        """
        published = self._published(safe)

        parent = published.rsplit("/", 1)[0]

        self._run_remote(
            f"mkdir -p -- {shlex.quote(parent)}; "
            f"tmp={shlex.quote(parent)}/.nba_woo_place.$$; "
            f"cp -- {shlex.quote(self._staged(safe))} \"$tmp\"; "
            'chmod 644 "$tmp"; '
            f"mv -f -- \"$tmp\" {shlex.quote(published)}"
        )

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def read_bytes(self, relative: str) -> bytes:
        quoted = shlex.quote(self._published(relative))

        return base64.b64decode(self._run_remote(f"base64 -- {quoted}"))

    def exists(self, relative: str) -> bool:
        return self._remote_digest(self._published(relative)) != REMOTE_MISSING

    def listing(self) -> list[str]:
        root = shlex.quote(self.remote.published_root)

        output = self._run_remote(
            f"if [ -d {root} ]; then find {root} -type f -printf '%P\\n'; fi"
        )

        return sorted(line for line in output.splitlines() if line.strip())

    def remove_tree(self, relative: str) -> None:
        self._run_remote(f"rm -rf -- {shlex.quote(self._published(relative))}")

    # ------------------------------------------------------------------
    # the public readback
    # ------------------------------------------------------------------

    def public_url(self, relative: str) -> str:
        return self.remote.published_base_url + self._relative(relative)

    def public_readback(self, relative: str) -> bytes:
        """What a consumer fetching the public URL would receive.

        The only check that speaks for the destination as a destination rather
        than as a filesystem. Cache directives are refused on the request
        because a readback served from an intermediary is a statement about
        the intermediary.
        """
        url = self.public_url(relative)

        request = urllib.request.Request(
            url,
            headers={
                "Accept": "*/*",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "nba-prop-quant-publisher/1",
            },
        )

        try:
            with urllib.request.urlopen(
                request, timeout=PUBLIC_READBACK_TIMEOUT_SECONDS
            ) as response:
                if response.status != 200:
                    raise PublishingRefused(
                        f"{url} answered {response.status}, so the release is "
                        "not publicly readable and is not promoted"
                    )

                return response.read()
        except urllib.error.HTTPError as error:
            raise PublishingRefused(
                f"{url} answered {error.code} {error.reason}, so the release "
                "is not publicly readable and is not promoted"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise PublishingRefused(
                f"{url} could not be read back ({error}), so what the "
                "destination serves is unproven and the release is not "
                "promoted"
            ) from error


def build_transport(
    destination: Destination,
    *,
    environment: dict[str, str] | None = None,
) -> LocalDirectoryTransport | SshRsyncTransport:
    if destination.transport == TRANSPORT_LOCAL_DIRECTORY:
        return LocalDirectoryTransport(destination.root)

    if destination.transport == TRANSPORT_SSH_RSYNC:
        return SshRsyncTransport(destination, environment=environment)

    raise PublishingRefused(
        f"the {destination.transport!r} transport is not implemented"
    )


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------


def serving_receipt_path(data_root: Path, slate_date: str) -> Path:
    return Path(data_root) / SERVING_RECEIPTS_RELATIVE / f"{slate_date}.json"


def priced_markets_path(data_root: Path, slate_date: str) -> Path:
    return Path(data_root) / PRICED_MARKETS_RELATIVE / f"{slate_date}.parquet"


def verify_receipt(
    receipt: dict[str, Any], *, slate_date: str, priced_path: Path
) -> dict[str, Any]:
    """Check the receipt actually provenances the bytes about to be published.

    The same question ``ops/grade_incumbent_production_slate.py`` asks before
    it writes a grade, for the same reason: a receipt naming a different slate
    or a different set of bytes is not provenance for this publication, and
    publishing the identities anyway would put somebody else's fit id and
    production SHA on the destination's manifest.
    """
    recorded = str(receipt.get("slate_date") or "")

    if recorded != slate_date:
        raise PublishingRefused(
            f"the serving receipt is for slate {recorded or 'unknown'}, not "
            f"{slate_date}, so it does not provenance these rows"
        )

    authority = str(receipt.get("model_authority") or "")

    if authority != MODEL_AUTHORITY:
        raise PublishingRefused(
            f"the serving receipt records the model authority as "
            f"{authority or 'nothing'}, not {MODEL_AUTHORITY!r}. Only the "
            "incumbent is published"
        )

    missing = sorted(
        [
            name
            for name in REQUIRED_PROVENANCE_FIELDS
            if receipt.get(name) in (None, "", {}, [])
        ]
        + [name for name in REQUIRED_PROVENANCE_KEYS if name not in receipt]
    )

    if missing:
        raise PublishingRefused(
            "the serving receipt is missing required provenance: "
            + ", ".join(missing)
        )

    artifact = receipt.get("priced_market_artifact") or {}

    expected = str(artifact.get("sha256") or "")

    if not expected:
        raise PublishingRefused(
            "the serving receipt records no priced-market artifact digest, so "
            "the rows being published cannot be tied to the run that produced "
            "them"
        )

    actual = sha256_file(priced_path)

    if actual != expected:
        raise PublishingRefused(
            f"{priced_path} holds {actual} but the serving receipt provenances "
            f"{expected}. The priced markets changed after they were served, "
            "so the receipt's identities do not describe these rows"
        )

    return {
        "incumbent_authority": receipt.get("incumbent_authority"),
        "incumbent_fit_id": receipt.get("incumbent_fit_id"),
        "incumbent_version": receipt.get("incumbent_version"),
        "input_state_fingerprint": receipt.get("input_state_fingerprint"),
        "model_authority": authority,
        "priced_market_artifact_sha256": expected,
        "priced_market_artifact_path": str(artifact.get("path") or priced_path),
        "production_code_sha": receipt.get("production_code_sha"),
        "served_at": receipt.get("generated_at"),
        "slate_date": slate_date,
    }


# ----------------------------------------------------------------------
# the rows
# ----------------------------------------------------------------------


def _scalar(value: Any) -> Any:
    """One parquet cell as something JSON can hold, losing no precision.

    Floats are passed through as floats rather than formatted, so the
    published number is the source number and a test can assert equality
    rather than closeness. ``NaN`` becomes ``None``: JSON has no NaN, and a
    published ``NaN`` would be a malformed probability by any reading.
    """
    if value is None or value is pd.NaT:
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).date().isoformat()

    if isinstance(value, (bool,)):
        return bool(value)

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float):
        return None if math.isnan(value) else value

    return value


def _finite_probability(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number) and 0.0 <= number <= 1.0


def assert_input_is_not_a_candidate_frame(frame: pd.DataFrame) -> None:
    present = [name for name in CANDIDATE_SOURCE_COLUMNS if name in frame.columns]

    if present:
        raise PublishingRefused(
            "the input frame carries shadow-candidate columns "
            + ", ".join(present)
            + ", so it is a shadow log rather than the incumbent's priced "
            "slate. Candidate probabilities are never published"
        )


def assert_schema_is_publishable(frame: pd.DataFrame) -> None:
    required = {name for name, _ in PUBLISHED_FIELDS} | set(SCREENED_COLUMNS)

    missing = sorted(required - set(frame.columns))

    if missing:
        raise PublishingRefused(
            "the priced markets do not carry "
            + ", ".join(missing)
            + ", so they are not the feed this publisher is a contract for"
        )


def row_violations(
    row: pd.Series, *, switch: PublishingSwitch, slate_date: str
) -> list[str]:
    """Which of the contract's fail-closed rules this row breaks.

    Section 9 of ``docs/WIZARDOFODDS_INTEGRATION.md``, rule for rule. The
    priced file holds only quotes that already passed the quote filter, so a
    row breaking any of these is the runtime disagreeing with its own
    contract. That is reported as a refusal rather than quietly dropped: a
    publisher that filtered it would publish a thinner feed and say nothing,
    which is the failure the fail-closed rules exist to make loud.
    """
    broken: list[str] = []

    if str(row.get("freeze_id") or "") != switch.approved_freeze_id:
        broken.append(
            f"freeze_id {row.get('freeze_id')!r} is not the approved "
            f"deployment {switch.approved_freeze_id!r}"
        )

    if switch.approved_freeze_stage and (
        str(row.get("freeze_stage") or "") != switch.approved_freeze_stage
    ):
        broken.append(
            f"freeze_stage {row.get('freeze_stage')!r} is not the approved "
            f"stage {switch.approved_freeze_stage!r}"
        )

    for column, approved in (
        ("market_pricing_schema_version", switch.approved_pricing_schema_version),
        ("projection_schema_version", switch.approved_projection_schema_version),
    ):
        if approved is None:
            continue

        try:
            actual = int(row.get(column))
        except (TypeError, ValueError):
            broken.append(f"{column} is not a version number")
            continue

        if actual != approved:
            broken.append(f"{column} is {actual}, not the approved {approved}")

    if not bool(row.get("quote_eligible")):
        broken.append("quote_eligible is false")

    if str(row.get("quote_filter_reason") or "").strip():
        broken.append(
            f"quote_filter_reason is {row.get('quote_filter_reason')!r}"
        )

    try:
        if int(row.get("availability_out")) == 1:
            broken.append("availability_out is 1")
    except (TypeError, ValueError):
        broken.append("availability_out is not readable")

    for column in PUBLISHED_PROBABILITY_FIELDS:
        if not _finite_probability(row.get(column)):
            broken.append(f"{column} is {row.get(column)!r}, not a probability")

    for column in (
        "calibrated_fair_over_american",
        "calibrated_fair_under_american",
    ):
        value = _scalar(row.get(column))

        if value is None:
            broken.append(f"{column} is missing")

    try:
        line = float(row.get("line_value"))

        if not math.isfinite(line):
            raise ValueError(line)
    except (TypeError, ValueError):
        broken.append(f"line_value {row.get('line_value')!r} is invalid")

    if str(row.get("prop_type") or "") in GATE3_PROPS:
        try:
            ready = int(row.get("gate3_role_ready"))
        except (TypeError, ValueError):
            ready = 0

        if ready != 1:
            broken.append(
                f"{row.get('prop_type')} is a Gate 3 prop and "
                "gate3_role_ready is not 1"
            )

    slate = pd.Timestamp(slate_date)

    for column in ("history_latest_date", "advanced_latest_date"):
        moment = pd.to_datetime(row.get(column), errors="coerce")

        if pd.isna(moment) or not moment < slate:
            broken.append(
                f"{column} {row.get(column)!r} is not strictly before the "
                f"slate date {slate_date}"
            )

    return broken


def published_rows(
    frame: pd.DataFrame, *, switch: PublishingSwitch, slate_date: str
) -> list[dict[str, Any]]:
    """The rows to publish, in a fixed order, or a refusal.

    Sorted on the logical key so the payload is a function of the rows and not
    of the order the join happened to leave them in.
    """
    assert_input_is_not_a_candidate_frame(frame)

    assert_schema_is_publishable(frame)

    if frame.empty:
        raise PublishingRefused(
            f"the priced markets for {slate_date} hold no rows, and an empty "
            "feed is not a publication"
        )

    rows: list[dict[str, Any]] = []

    refusals: list[str] = []

    for position, (_, source) in enumerate(frame.iterrows()):
        broken = row_violations(source, switch=switch, slate_date=slate_date)

        if broken:
            refusals.append(f"row {position}: " + "; ".join(broken))
            continue

        rows.append(
            {
                published: _scalar(source.get(column))
                for column, published in PUBLISHED_FIELDS
            }
        )

    if refusals:
        raise PublishingRefused(
            f"{len(refusals)} of {len(frame)} priced rows break the "
            "integration contract's fail-closed rules, so none is published: "
            + " | ".join(refusals[:5])
        )

    duplicates = _duplicate_keys(rows)

    if duplicates:
        raise PublishingRefused(
            "the priced markets hold more than one row per published quote: "
            + "; ".join(duplicates)
        )

    return sorted(rows, key=lambda row: tuple(str(row[name]) for name in LOGICAL_KEY))


def _duplicate_keys(rows: list[dict[str, Any]]) -> list[str]:
    seen: dict[tuple[str, ...], int] = {}

    for row in rows:
        key = tuple(str(row[name]) for name in LOGICAL_KEY)

        seen[key] = seen.get(key, 0) + 1

    return [
        " / ".join(key) + f" appears {count} times"
        for key, count in sorted(seen.items())
        if count > 1
    ]


# ----------------------------------------------------------------------
# the release
# ----------------------------------------------------------------------


@dataclass
class Release:
    release_id: str
    rows_relative: str
    manifest_relative: str
    rows_bytes: bytes = field(repr=False)
    manifest_bytes: bytes = field(repr=False)
    rows_sha256: str = ""
    manifest_sha256: str = ""
    row_count: int = 0


def build_release(
    *,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    switch: PublishingSwitch,
    destination: Destination,
    slate_date: str,
) -> Release:
    """A whole release, named by what is in it.

    The release id is derived from the payload digest rather than from a clock,
    which is what makes a rerun idempotent: the same rows produce the same id,
    so the second run recognises its own release instead of stacking a new one
    beside it.
    """
    payload = {
        "contract_version": PUBLICATION_CONTRACT_VERSION,
        "rows": rows,
        "slate_date": slate_date,
    }

    rows_bytes = serialise(payload)

    rows_sha256 = digest(rows_bytes)

    release_id = f"{destination.release_prefix}__{slate_date}__{rows_sha256[:16]}"

    manifest = {
        "contract_version": PUBLICATION_CONTRACT_VERSION,
        "mode": destination.mode,
        "model_authority": MODEL_AUTHORITY,
        "candidate_authority": CANDIDATE_AUTHORITY,
        "approved_freeze_id": switch.approved_freeze_id,
        "approved_freeze_stage": switch.approved_freeze_stage,
        "approved_market_pricing_schema_version": (
            switch.approved_pricing_schema_version
        ),
        "approved_projection_schema_version": (
            switch.approved_projection_schema_version
        ),
        "provenance": provenance,
        "release_id": release_id,
        "row_count": len(rows),
        "rows_relative": f"{RELEASES_DIRECTORY}/{release_id}/{ROWS_NAME}",
        "rows_sha256": rows_sha256,
        "slate_date": slate_date,
        "transport": destination.transport,
    }

    manifest_bytes = serialise(manifest)

    return Release(
        release_id=release_id,
        rows_relative=f"{RELEASES_DIRECTORY}/{release_id}/{ROWS_NAME}",
        manifest_relative=f"{RELEASES_DIRECTORY}/{release_id}/{MANIFEST_NAME}",
        rows_bytes=rows_bytes,
        manifest_bytes=manifest_bytes,
        rows_sha256=rows_sha256,
        manifest_sha256=digest(manifest_bytes),
        row_count=len(rows),
    )


def write_release(transport: LocalDirectoryTransport, release: Release) -> None:
    transport.write_bytes(release.rows_relative, release.rows_bytes)

    transport.write_bytes(release.manifest_relative, release.manifest_bytes)


def verify_release(transport: LocalDirectoryTransport, release: Release) -> dict[str, Any]:
    """Read the release back through the transport and check it is what was sent.

    Read back rather than remembered. The point is to catch a destination that
    accepted the write and kept something else -- a truncated upload, a
    partially written object, a path that silently went somewhere other than
    where it was addressed -- and only a readback can see that. Promotion
    happens after this and never before.
    """
    for relative, expected in (
        (release.rows_relative, release.rows_sha256),
        (release.manifest_relative, release.manifest_sha256),
    ):
        if not transport.exists(relative):
            raise PublishingRefused(
                f"{relative} is not at the destination after being written, so "
                "the release is partial and is not promoted"
            )

        actual = digest(transport.read_bytes(relative))

        if actual != expected:
            raise PublishingRefused(
                f"{relative} reads back as {actual} but was written as "
                f"{expected}, so the destination does not hold the release "
                "that was sent and it is not promoted"
            )

    readback = json.loads(transport.read_bytes(release.rows_relative))

    if len(readback.get("rows") or []) != release.row_count:
        raise PublishingRefused(
            f"{release.rows_relative} reads back "
            f"{len(readback.get('rows') or [])} rows, not the "
            f"{release.row_count} that were written"
        )

    return {
        "manifest_sha256": release.manifest_sha256,
        "public": verify_public_readback(transport, release),
        "readback_rows": len(readback.get("rows") or []),
        "rows_sha256": release.rows_sha256,
        "verified": True,
    }


def verify_public_readback(
    transport: Any, release: Release
) -> dict[str, Any]:
    """Fetch the release from its public URL and check it is the release.

    A transport that serves no public URL reports that it was not checked
    rather than reporting a pass, because "there was nothing to fetch" and
    "what was fetched was right" are different facts and a verification
    summary that conflated them would let a local canary claim a public
    readback it never did.

    For one that does, this is the check the destination exists for. The
    digests say the bytes a consumer receives are the bytes that were
    published, and the row count says the document it parses holds the rows
    that were priced -- which is the whole of what the website is promised.
    """
    reader = getattr(transport, "public_readback", None)

    if reader is None:
        return {
            "checked": False,
            "reason": (
                "this transport serves no public URL, so there is nothing to "
                "read a release back from"
            ),
        }

    bodies: dict[str, bytes] = {}

    for relative, expected in (
        (release.rows_relative, release.rows_sha256),
        (release.manifest_relative, release.manifest_sha256),
    ):
        body = reader(relative)

        actual = digest(body)

        if actual != expected:
            raise PublishingRefused(
                f"{transport.public_url(relative)} serves {actual} but the "
                f"release was published as {expected}, so what the "
                "destination serves publicly is not the release and it is not "
                "promoted"
            )

        bodies[relative] = body

    document = json.loads(bodies[release.rows_relative])

    rows = document.get("rows") or []

    if len(rows) != release.row_count:
        raise PublishingRefused(
            f"{transport.public_url(release.rows_relative)} serves "
            f"{len(rows)} rows, not the {release.row_count} that were "
            "published"
        )

    return {
        "checked": True,
        "manifest_sha256": digest(bodies[release.manifest_relative]),
        "manifest_url": transport.public_url(release.manifest_relative),
        "readback_rows": len(rows),
        "rows_sha256": digest(bodies[release.rows_relative]),
        "rows_url": transport.public_url(release.rows_relative),
    }


def prune_releases(
    transport: Any, *, retained: int | None
) -> dict[str, Any]:
    """Remove every release the destination is not serving or holding back to.

    Retention rather than accumulation, and an explicit two rather than a
    date window: what a destination owes its consumers is the release it
    serves, and what it owes an operator is the one before it. A dated
    snapshot per day is how the legacy NBA output grew to something nobody
    could reason about, and the receipts and manifests that record what was
    published are kept regardless -- it is the payloads that are pruned, so
    the audit trail outlives the bytes it describes.

    Pointer-driven, so the two releases that are kept are the two the pointers
    name rather than the two that sort last. A release is only ever removed
    after the pointers have been rotated, so nothing reachable is pruned.
    """
    if retained is None:
        return {"pruned": [], "retained": None}

    kept = {
        str((read_pointer(transport, pointer) or {}).get("release_id") or "")
        for pointer in (CURRENT_POINTER, PREVIOUS_POINTER)
    } - {""}

    present = {
        name.split("/")[1]
        for name in transport.listing()
        if name.startswith(f"{RELEASES_DIRECTORY}/") and name.count("/") >= 2
    }

    removed: list[str] = []

    for release_id in sorted(present - kept):
        transport.remove_tree(f"{RELEASES_DIRECTORY}/{release_id}")

        removed.append(release_id)

    return {"pruned": removed, "retained": sorted(kept)}


def append_audit_entry(
    transport: LocalDirectoryTransport, entry: dict[str, Any]
) -> str:
    """Record one completed action at the destination, keeping the earlier ones.

    Read-and-rewrite rather than an append, because the transport offers one
    write and it is atomic. That is the right trade here: an audit log a
    crashed run could truncate would be worse than one written whole every
    time, and the file holds one line per release rather than per row.
    """
    existing = (
        transport.read_bytes(AUDIT_LOG_NAME)
        if transport.exists(AUDIT_LOG_NAME)
        else b""
    )

    line = json.dumps(entry, sort_keys=True).encode("utf-8") + b"\n"

    transport.write_bytes(AUDIT_LOG_NAME, existing + line)

    return AUDIT_LOG_NAME


def read_audit_log(transport: LocalDirectoryTransport) -> list[dict[str, Any]]:
    if not transport.exists(AUDIT_LOG_NAME):
        return []

    text = transport.read_bytes(AUDIT_LOG_NAME).decode("utf-8")

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def attestation_for(
    *,
    mode: str,
    release: Release,
    provenance: dict[str, Any],
    verification: dict[str, Any],
    slate_date: str,
) -> dict[str, Any]:
    return {
        "attested_at": _utc_now(),
        "manifest_sha256": release.manifest_sha256,
        "mode": mode,
        "model_authority": MODEL_AUTHORITY,
        "outcome": OUTCOME_PUBLISHED,
        "production_code_sha": provenance.get("production_code_sha"),
        "public_readback": verification.get("public") or {},
        "release_id": release.release_id,
        "row_count": release.row_count,
        "rows_sha256": release.rows_sha256,
        "slate_date": slate_date,
        "source_artifact_sha256": provenance.get(
            "priced_market_artifact_sha256"
        ),
        "verified": bool(verification.get("verified")),
    }


#: The one attestation field that is a clock rather than a fact about the
#: release. Two attestations differing only here describe the same publication,
#: so rewriting one as the other would change the destination's bytes without
#: changing anything it claims.
ATTESTATION_CLOCK_FIELD = "attested_at"

#: What a current release's manifest must carry for that release to be
#: attestable. These are the field names of the provenance projection
#: :func:`verify_receipt` returns and :func:`build_release` embeds, which are
#: not the serving receipt's own field names in
#: :data:`REQUIRED_PROVENANCE_FIELDS`: the receipt has ``generated_at`` and a
#: ``priced_market_artifact`` object where the manifest has ``served_at`` and
#: that object's digest.
ATTESTABLE_PROVENANCE_FIELDS: tuple[str, ...] = (
    "incumbent_authority",
    "incumbent_version",
    "input_state_fingerprint",
    "model_authority",
    "priced_market_artifact_sha256",
    "production_code_sha",
    "served_at",
    "slate_date",
)

#: Present rather than non-empty, for the reason
#: :data:`REQUIRED_PROVENANCE_KEYS` gives: null is the true value before the
#: first human promotion.
ATTESTABLE_PROVENANCE_KEYS: tuple[str, ...] = ("incumbent_fit_id",)


def _attestation_identity(attestation: dict[str, Any]) -> dict[str, Any]:
    """Everything an attestation claims, which is all of it but the clock."""
    return {
        name: value
        for name, value in attestation.items()
        if name != ATTESTATION_CLOCK_FIELD
    }


def attestable_current_release(
    transport: Any, *, release: Release
) -> dict[str, Any]:
    """The release ``current.json`` names, read back and cross-checked.

    The attestation is a statement about what the destination is serving, so it
    may only describe the release the destination says is current -- read back
    from there rather than remembered from this run. Three documents have to
    agree before anything is attested: the pointer, the manifest it names, and
    the rows that manifest describes.

    Checked from the specific to the general, so that each way this can be
    wrong is refused by the check that names it and the manifest digest is left
    as the catch-all for the rest. Refused rather than repaired throughout: a
    destination whose pointer, manifest and rows disagree is in a state no
    attestation should describe, and picking one of the three to believe is how
    a wrong attestation would become a confident one.
    """
    current = read_pointer(transport, CURRENT_POINTER)

    if not current:
        raise PublishingRefused(
            f"{CURRENT_POINTER} names no release at the destination, so there "
            f"is no current release for {ATTESTATION_NAME} to describe"
        )

    if str(current.get("release_id") or "") != release.release_id:
        raise PublishingRefused(
            f"{CURRENT_POINTER} names release "
            f"{current.get('release_id')!r} and this publication wrote "
            f"{release.release_id!r}, so attesting it would describe something "
            "the destination is not serving"
        )

    manifest_relative = str(current.get("manifest_relative") or "")

    if not manifest_relative or not transport.exists(manifest_relative):
        raise PublishingRefused(
            f"{CURRENT_POINTER} names a manifest at "
            f"{manifest_relative or 'no path'}, which is not at the "
            "destination, so the current release cannot be attested"
        )

    manifest_bytes = transport.read_bytes(manifest_relative)

    try:
        manifest = json.loads(manifest_bytes)
    except ValueError as error:
        raise PublishingRefused(
            f"{manifest_relative} is not readable JSON ({error}), so the "
            "current release cannot be attested"
        ) from error

    authority = str(manifest.get("model_authority") or "")

    if authority != MODEL_AUTHORITY:
        raise PublishingRefused(
            "the current release's manifest records the model authority as "
            f"{authority or 'nothing'}, not {MODEL_AUTHORITY!r}, so it is not "
            "an incumbent publication and is not attested"
        )

    provenance = manifest.get("provenance") or {}

    missing = sorted(
        [
            name
            for name in ATTESTABLE_PROVENANCE_FIELDS
            if provenance.get(name) in (None, "", {}, [])
        ]
        + [
            name
            for name in ATTESTABLE_PROVENANCE_KEYS
            if name not in provenance
        ]
    )

    if missing:
        raise PublishingRefused(
            "the current release's manifest is missing required provenance: "
            + ", ".join(missing)
        )

    rows_relative = str(manifest.get("rows_relative") or "")

    rows_sha256 = str(manifest.get("rows_sha256") or "")

    for label, pointed, recorded in (
        (
            "release id",
            str(current.get("release_id") or ""),
            str(manifest.get("release_id") or ""),
        ),
        ("rows path", str(current.get("rows_relative") or ""), rows_relative),
        ("rows digest", str(current.get("rows_sha256") or ""), rows_sha256),
    ):
        if pointed != recorded:
            raise PublishingRefused(
                f"{CURRENT_POINTER} and {manifest_relative} disagree on the "
                f"{label} of the current release ({pointed or 'nothing'} "
                f"against {recorded or 'nothing'}), so what the destination "
                "serves is not settled and is not attested"
            )

    if not rows_relative or not transport.exists(rows_relative):
        raise PublishingRefused(
            "the current release's manifest names rows at "
            f"{rows_relative or 'no path'}, which are not at the destination, "
            "so there is nothing to attest"
        )

    served = digest(transport.read_bytes(rows_relative))

    if served != rows_sha256:
        raise PublishingRefused(
            f"{rows_relative} reads back as {served} but the current "
            f"release's manifest records {rows_sha256}, so the destination is "
            "not serving the release it says it is"
        )

    manifest_sha256 = digest(manifest_bytes)

    # Last, so that anything the checks above name is refused by the check
    # that names it. This one catches whatever they do not: a manifest reading
    # back as the one this publication wrote is that manifest field for field,
    # which is what makes attesting the release this run built the same thing
    # as attesting the release the destination holds.
    if manifest_sha256 != release.manifest_sha256:
        raise PublishingRefused(
            f"{manifest_relative} reads back as {manifest_sha256} but this "
            f"publication wrote {release.manifest_sha256}, so the destination "
            "does not hold the manifest that would be attested"
        )

    return {
        "manifest_relative": manifest_relative,
        "manifest_sha256": manifest_sha256,
        "provenance": provenance,
        "release_id": str(manifest.get("release_id") or ""),
        "rows_sha256": rows_sha256,
        "slate_date": str(manifest.get("slate_date") or ""),
    }


def ensure_canary_attestation(
    transport: Any, *, release: Release, verification: dict[str, Any]
) -> dict[str, Any]:
    """Make ``canary_attestation.json`` describe the release now current.

    The attestation and the audit log beside it answer different questions and
    are maintained differently. ``audit_log.jsonl`` records what each run did,
    and is appended to and never rewritten, because rewriting it would falsify
    history. The attestation records what the destination is serving *now*, so
    it has to be rewritten whenever that stops being what it says.

    It used to be written only by a publication that promoted something, on the
    reasoning that a rerun of the release already current changes nothing a
    reader would see. A rerun can change the manifest, though: the release id
    is the rows digest, so a release whose *provenance* was corrected
    republishes under the same id. That is how a destination came to serve a
    corrected manifest while its attestation still named the production SHA the
    correction had replaced -- and the attestation, not the manifest, is what a
    later activation reads.

    Rewritten only when stale. An attestation that already describes the
    current release differs from a freshly built one by its timestamp alone,
    and restamping it would make a no-op rerun change the destination's bytes.
    """
    current = attestable_current_release(transport, release=release)

    expected = attestation_for(
        mode=MODE_CANARY,
        release=release,
        provenance=current["provenance"],
        verification=verification,
        slate_date=current["slate_date"],
    )

    existing = read_pointer(transport, ATTESTATION_NAME)

    if existing is not None and _attestation_identity(
        existing
    ) == _attestation_identity(expected):
        return {
            "attested_at": existing.get(ATTESTATION_CLOCK_FIELD),
            "reason": "the attestation already describes the current release",
            "refreshed": False,
            "relative": ATTESTATION_NAME,
            "release_id": current["release_id"],
        }

    transport.write_bytes(ATTESTATION_NAME, serialise(expected))

    return {
        "attested_at": expected[ATTESTATION_CLOCK_FIELD],
        "reason": (
            "the attestation did not describe the current release"
            if existing is not None
            else "the destination had no attestation"
        ),
        "refreshed": True,
        "relative": ATTESTATION_NAME,
        "release_id": current["release_id"],
    }


def read_attestation(path: Path | None) -> dict[str, Any]:
    """The canary attestation an activation was handed, or the refusal."""
    if path is None:
        raise PublishingRefused(
            "live publication requires --canary-attestation naming the "
            f"{ATTESTATION_NAME} a successful canary of this same source "
            "artifact wrote. Activating something that was never canaried is "
            "the one sequence this contract exists to prevent"
        )

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PublishingRefused(
            f"the canary attestation at {path} could not be read ({error}), "
            "so there is no evidence this artifact was canaried"
        ) from error


def assert_canary_attested(
    attestation: dict[str, Any],
    *,
    release: Release,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Require a prior successful canary of *these* bytes, not of something.

    Matched on two digests. The source artifact digest says the canary ran on
    the same priced file; the rows digest says that file still renders to the
    same published rows. Either one alone would let an activation inherit a
    canary that proved a different thing.
    """
    if str(attestation.get("mode") or "") != MODE_CANARY:
        raise PublishingRefused(
            "the attestation records mode "
            f"{attestation.get('mode')!r}, not {MODE_CANARY!r}, so it is not "
            "a canary attestation"
        )

    if str(attestation.get("outcome") or "") != OUTCOME_PUBLISHED:
        raise PublishingRefused(
            f"the canary attestation records outcome "
            f"{attestation.get('outcome')!r}, so no canary succeeded for this "
            "artifact"
        )

    if attestation.get("verified") is not True:
        raise PublishingRefused(
            "the canary attestation does not record a verified readback, so "
            "the canary did not prove the destination held what was sent"
        )

    if str(attestation.get("model_authority") or "") != MODEL_AUTHORITY:
        raise PublishingRefused(
            "the canary attestation records the model authority as "
            f"{attestation.get('model_authority')!r}, not {MODEL_AUTHORITY!r}"
        )

    attested_source = str(attestation.get("source_artifact_sha256") or "")

    source = str(provenance.get("priced_market_artifact_sha256") or "")

    if attested_source != source:
        raise PublishingRefused(
            f"the canary attestation covers source artifact {attested_source} "
            f"but this activation publishes {source}. The canary that would "
            "authorise this release was run on different bytes"
        )

    if str(attestation.get("rows_sha256") or "") != release.rows_sha256:
        raise PublishingRefused(
            "the canary attestation covers rows "
            f"{attestation.get('rows_sha256')} but this activation would "
            f"publish {release.rows_sha256}, so what was canaried is not what "
            "would go live"
        )

    return {
        "attested_at": attestation.get("attested_at"),
        "release_id": attestation.get("release_id"),
        "rows_sha256": attestation.get("rows_sha256"),
        "source_artifact_sha256": attested_source,
    }


def assert_rollback_target_exists(
    transport: LocalDirectoryTransport,
) -> dict[str, Any]:
    """Require somewhere to go back to before changing what is live.

    The release currently pointed at is the rollback target of the activation
    about to happen: promotion rotates it into ``previous``. Checking it is
    both named and still present means an activation cannot be the step that
    leaves the destination with no way back -- including the first one, which
    therefore needs its rollback target seeded from whatever the destination
    is already serving.
    """
    current = read_pointer(transport, CURRENT_POINTER)

    if not current:
        raise PublishingRefused(
            f"{CURRENT_POINTER} names no release at the destination, so this "
            "activation would have no rollback target. Seed the destination "
            "with the release it is already serving before activating"
        )

    rows_relative = str(current.get("rows_relative") or "")

    if not rows_relative or not transport.exists(rows_relative):
        raise PublishingRefused(
            "the release currently at the destination names rows at "
            f"{rows_relative or 'no path'}, which are not there, so it could "
            "not be rolled back to"
        )

    actual = digest(transport.read_bytes(rows_relative))

    expected = str(current.get("rows_sha256") or "")

    if expected and actual != expected:
        raise PublishingRefused(
            f"the release currently at the destination reads back as {actual}, "
            f"not the {expected} recorded when it was promoted, so it is not a "
            "rollback target that can be trusted"
        )

    return {
        "release_id": current.get("release_id"),
        "rows_relative": rows_relative,
        "rows_sha256": actual,
    }


def read_pointer(
    transport: LocalDirectoryTransport, relative: str
) -> dict[str, Any] | None:
    if not transport.exists(relative):
        return None

    try:
        return json.loads(transport.read_bytes(relative))
    except ValueError:
        return None


def promote(
    transport: LocalDirectoryTransport, release: Release, *, mode: str
) -> dict[str, Any]:
    """Make a verified release the current one, keeping the last as rollback.

    The pointer is written through the transport's atomic replace, so a reader
    sees either the release that was current before this call or the one after
    it, never a half-written pointer naming neither.

    A rerun of the release already current is a no-op on both pointers. It has
    to be: rotating ``current`` into ``previous`` when the two are the same
    release would destroy the rollback target, which is precisely the thing a
    rerun must not be able to do.
    """
    current = read_pointer(transport, CURRENT_POINTER)

    if current and str(current.get("release_id")) == release.release_id:
        return {
            "promoted": False,
            "reason": "the release already current is this one",
            "release_id": release.release_id,
            "rollback_target": (
                read_pointer(transport, PREVIOUS_POINTER) or {}
            ).get("release_id"),
        }

    if current:
        transport.write_bytes(PREVIOUS_POINTER, serialise(current))

    transport.write_bytes(
        CURRENT_POINTER,
        serialise(
            {
                "manifest_relative": release.manifest_relative,
                "mode": mode,
                "promoted_at": _utc_now(),
                "release_id": release.release_id,
                "rows_relative": release.rows_relative,
                "rows_sha256": release.rows_sha256,
            }
        ),
    )

    return {
        "promoted": True,
        "reason": "the release was verified and is now current",
        "release_id": release.release_id,
        "rollback_target": (current or {}).get("release_id"),
    }


def rollback(transport: LocalDirectoryTransport) -> dict[str, Any]:
    """Point the destination back at the release that was current before.

    Restoration, not reconstruction. The previous release's bytes are still at
    the destination, so there is nothing to rebuild, nothing to refit and no
    model involved: the pointer moves back and the destination serves what it
    served yesterday. Refused rather than guessed at when the previous release
    is absent or its bytes are gone, because a rollback that silently landed
    somewhere else would be worse than staying put.
    """
    previous = read_pointer(transport, PREVIOUS_POINTER)

    if not previous:
        raise PublishingRefused(
            f"{PREVIOUS_POINTER} names no previous release, so there is "
            "nothing to roll back to"
        )

    rows_relative = str(previous.get("rows_relative") or "")

    if not rows_relative or not transport.exists(rows_relative):
        raise PublishingRefused(
            "the previous release's rows at "
            f"{rows_relative or 'an unnamed path'} are not at the "
            "destination, so rolling back would point at nothing"
        )

    actual = digest(transport.read_bytes(rows_relative))

    expected = str(previous.get("rows_sha256") or "")

    if expected and actual != expected:
        raise PublishingRefused(
            f"the previous release's rows read back as {actual}, not the "
            f"{expected} recorded when it was promoted, so it is not the "
            "release it claims to be"
        )

    public = _public_rows_digest(transport, rows_relative)

    if public is not None and public != actual:
        raise PublishingRefused(
            f"the previous release's rows are served publicly as {public} but "
            f"hold {actual} at the destination, so rolling back would point at "
            "a release the destination does not actually serve"
        )

    superseded = read_pointer(transport, CURRENT_POINTER)

    transport.write_bytes(
        CURRENT_POINTER,
        serialise({**previous, "promoted_at": _utc_now()}),
    )

    audit_relative = append_audit_entry(
        transport,
        {
            "action": ACTION_ROLLED_BACK,
            "mode": str(previous.get("mode") or ""),
            "model_authority": MODEL_AUTHORITY,
            "public_rows_sha256": public,
            "recorded_at": _utc_now(),
            "release_id": previous.get("release_id"),
            "rows_sha256": actual,
            "superseded_release_id": (superseded or {}).get("release_id"),
        },
    )

    return {
        "audit_relative": audit_relative,
        "outcome": OUTCOME_ROLLED_BACK,
        "public_rows_sha256": public,
        "restored_release_id": previous.get("release_id"),
        "rows_sha256": actual,
        "superseded_release_id": (superseded or {}).get("release_id"),
    }


def _public_rows_digest(transport: Any, rows_relative: str) -> str | None:
    """What the public URL serves for one release's rows, when there is one.

    ``None`` for a transport with no public URL, which is not a failure: a
    local canary has nothing public to disagree with. For one that has, the
    digest is the fact a rollback needs -- restoring a pointer to bytes the
    destination holds but does not serve would be a rollback to nothing a
    consumer can see.
    """
    reader = getattr(transport, "public_readback", None)

    return None if reader is None else digest(reader(rows_relative))


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------


def publish(
    *,
    mode: str,
    slate_date: str,
    data_root: Path,
    project_root: Path | None = None,
    destination_root: Path | None = None,
    switch_path: Path | None = None,
    destinations_path: Path | None = None,
    receipt_path: Path | None = None,
    priced_path: Path | None = None,
    canary_attestation: Path | None = None,
    destination_name: str | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One publication, start to finish, or the first refusal on the way.

    Ordered so the cheapest refusals come first and nothing is read from the
    destination before the switch has allowed this mode at all. The two
    activation-only gates -- a prior canary of these bytes, and a rollback
    target to return to -- are checked after the release is built, because
    both are statements about this exact release, and before anything is
    written, because an activation that fails them must leave live untouched.
    """
    root = Path(project_root or PROJECT_ROOT)

    switch = read_publishing_switch(
        mode,
        project_root=root,
        switch_path=switch_path,
        environment=environment,
    )

    assert_mode_enabled(switch)

    document = read_destinations(
        project_root=root, destinations_path=destinations_path
    )

    destination = resolve_destination(
        mode, document, root=destination_root, name=destination_name
    )

    if mode == MODE_CANARY:
        assert_canary_is_not_live(destination, document)

        assert_remote_canary_is_confined(destination)

    priced = (
        Path(priced_path)
        if priced_path is not None
        else priced_markets_path(data_root, slate_date)
    )

    if not priced.exists():
        raise PublishingRefused(
            f"{priced} does not exist, so the incumbent has not served "
            f"{slate_date} and there is nothing to publish"
        )

    receipt_file = (
        Path(receipt_path)
        if receipt_path is not None
        else serving_receipt_path(data_root, slate_date)
    )

    if not receipt_file.exists():
        raise PublishingRefused(
            f"{receipt_file} does not exist, so the rows at {priced} carry no "
            "provenance and are not published"
        )

    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except ValueError as error:
        raise PublishingRefused(
            f"{receipt_file} is not readable JSON ({error}), so the rows carry "
            "no provenance"
        ) from error

    provenance = verify_receipt(
        receipt, slate_date=slate_date, priced_path=priced
    )

    rows = published_rows(
        pd.read_parquet(priced), switch=switch, slate_date=slate_date
    )

    release = build_release(
        rows=rows,
        provenance=provenance,
        switch=switch,
        destination=destination,
        slate_date=slate_date,
    )

    transport = build_transport(destination, environment=environment)

    try:
        identity = (
            transport.verify_identity()
            if hasattr(transport, "verify_identity")
            else None
        )

        activation: dict[str, Any] | None = None

        if mode == MODE_LIVE:
            activation = {
                "canary": assert_canary_attested(
                    read_attestation(canary_attestation),
                    release=release,
                    provenance=provenance,
                ),
                "rollback_target": assert_rollback_target_exists(transport),
            }

        write_release(transport, release)

        verification = verify_release(transport, release)

        promotion = promote(transport, release, mode=mode)

        # After the pointers, because the attestation describes the release
        # they name. A rerun that promoted nothing still refreshes it when it
        # is stale, and leaves it byte-identical when it is not.
        attestation = (
            ensure_canary_attestation(
                transport, release=release, verification=verification
            )
            if mode == MODE_CANARY
            else None
        )

        # After the pointers, never before: the releases kept are the ones the
        # pointers name, so pruning first could remove the rollback target
        # this publication is about to create.
        retention = prune_releases(
            transport, retained=destination.retained_releases
        )

        audit_relative = append_audit_entry(
            transport,
            {
                "action": (
                    ACTION_PUBLISHED
                    if promotion["promoted"]
                    else ACTION_REPUBLISHED
                ),
                "mode": mode,
                "model_authority": MODEL_AUTHORITY,
                "production_code_sha": provenance.get("production_code_sha"),
                "recorded_at": _utc_now(),
                "release_id": release.release_id,
                "row_count": release.row_count,
                "rows_sha256": release.rows_sha256,
                "slate_date": slate_date,
                "source_artifact_sha256": provenance.get(
                    "priced_market_artifact_sha256"
                ),
            },
        )
    finally:
        transport.close()

    return {
        "activation": activation,
        "attestation": attestation,
        "audit_relative": audit_relative,
        "destination": destination_payload(destination, identity=identity),
        "generated_at": _utc_now(),
        "live_publishing": read_publishing_switch(
            MODE_LIVE,
            project_root=root,
            switch_path=switch_path,
            environment=environment,
        ).state,
        "mode": mode,
        "model_authority": MODEL_AUTHORITY,
        "outcome": OUTCOME_PUBLISHED,
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
        "slate_date": slate_date,
        "switch": switch.payload(),
        "verification": verification,
    }


def destination_payload(
    destination: Destination, *, identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    """What the receipt records about where this went.

    The remote facts are addresses and are recorded; the credential is a name
    and is recorded as one. No key material passes through here, which is the
    same reason the configuration holds the variable's name rather than its
    value.
    """
    payload: dict[str, Any] = {
        "mode": destination.mode,
        "release_prefix": destination.release_prefix,
        "root": str(destination.root),
        "source": destination.source,
        "transport": destination.transport,
    }

    if destination.name:
        payload["name"] = destination.name

    if destination.retained_releases is not None:
        payload["retained_releases"] = destination.retained_releases

    remote = destination.remote

    if remote is not None:
        payload["remote"] = {
            "credential_env": remote.credential_env,
            "host": remote.host,
            "port": remote.port,
            "private_staging": remote.private_staging,
            "public_base_url": remote.published_base_url,
            "public_target": remote.public_target,
            "published_root": remote.published_root,
            "staging_root": remote.staging_root,
            "user": remote.user,
        }

    if identity is not None:
        payload["authenticated_as"] = identity

    return payload


def render(receipt: dict[str, Any]) -> str:
    if receipt.get("outcome") == OUTCOME_ROLLED_BACK:
        rows = [
            ("outcome", receipt["outcome"]),
            ("restored release", receipt.get("restored_release_id")),
            ("superseded release", receipt.get("superseded_release_id")),
            ("public rows digest", receipt.get("public_rows_sha256")),
            ("audit entry", receipt.get("audit_relative")),
        ]
    else:
        destination = receipt.get("destination") or {}

        public = (receipt.get("verification") or {}).get("public") or {}

        rows = [
            ("outcome", receipt.get("outcome")),
            ("mode", receipt.get("mode")),
            ("slate date", receipt.get("slate_date")),
            ("model authority", receipt.get("model_authority")),
            ("live publishing", receipt.get("live_publishing")),
            ("release", (receipt.get("release") or {}).get("release_id")),
            ("published rows", (receipt.get("release") or {}).get("row_count")),
            ("readback verified", (receipt.get("verification") or {}).get("verified")),
            ("public readback", public.get("checked")),
            ("public rows", public.get("readback_rows")),
            ("transport", destination.get("transport")),
            ("destination", destination.get("name") or destination.get("root")),
            ("destination root", destination.get("root")),
            ("public url", public.get("rows_url")),
            ("retained releases", (receipt.get("retention") or {}).get("retained")),
            ("pruned releases", (receipt.get("retention") or {}).get("pruned")),
            ("audit entry", receipt.get("audit_relative")),
        ]

    lines = [
        "## WizardOfOdds incumbent publication",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    lines += [f"| {label} | {value} |" for label, value in rows]

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="publish_incumbent_to_wizardofodds",
        description=(
            "Publish the incumbent's already-priced slate to a WizardOfOdds "
            "destination. Computes nothing, promotes nothing, and refuses "
            "every mode its switch does not enable."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=MODE_CANARY,
        help="canary writes a non-live destination. live is the activation "
        "verb, is DISABLED, and cannot be enabled from this repository: the "
        "host, directory, protocol and credentials are all external facts.",
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=None,
        help="where the release tree is written. Required for the unnamed "
        "canary, whose root is a scratch location this repository does not "
        "assert. Ignored by a named destination, which carries its own.",
    )
    parser.add_argument(
        "--destination-name",
        default=None,
        help="publish to one of the configuration's named destinations "
        "instead of the mode's own. A named destination is remote and carries "
        "its own host, paths and credential variable, so reaching a real host "
        "is something asked for by name rather than the default.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--switch-path", type=Path, default=None)
    parser.add_argument("--destinations-path", type=Path, default=None)
    parser.add_argument(
        "--serving-receipt",
        type=Path,
        default=None,
        help="the receipt ops/run_incumbent_production_serving.py wrote. "
        f"Defaults to <data-root>/{SERVING_RECEIPTS_RELATIVE}/<slate-date>.json .",
    )
    parser.add_argument(
        "--canary-attestation",
        type=Path,
        default=None,
        help=f"the {ATTESTATION_NAME} a successful canary of this same source "
        "artifact wrote. Required for live publication and refused without.",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="restore the previous release at the destination and exit. Moves "
        "a pointer back to bytes that are already there; refits nothing.",
    )
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()

    if args.rollback:
        document = read_destinations(
            project_root=project_root, destinations_path=args.destinations_path
        )

        destination = resolve_destination(
            args.mode,
            document,
            root=args.destination_root,
            name=args.destination_name,
        )

        transport = build_transport(destination)

        try:
            restored = rollback(transport)
        finally:
            transport.close()

        return {
            **restored,
            "destination": destination_payload(destination),
            "generated_at": _utc_now(),
            "mode": args.mode,
            "model_authority": MODEL_AUTHORITY,
            "slate_date": args.slate_date,
        }

    return publish(
        mode=args.mode,
        slate_date=args.slate_date,
        data_root=Path(args.data_root).resolve(),
        project_root=project_root,
        destination_root=args.destination_root,
        switch_path=args.switch_path,
        destinations_path=args.destinations_path,
        receipt_path=args.serving_receipt,
        canary_attestation=args.canary_attestation,
        destination_name=args.destination_name,
    )


def _write(path: Path | None, text: str, *, append: bool) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except (PublishingRefused, OSError, ValueError) as error:
        failure = {
            "error": type(error).__name__,
            "message": str(error),
            "mode": args.mode,
            "model_authority": MODEL_AUTHORITY,
            "outcome": "PUBLISHING_REFUSED",
            "slate_date": args.slate_date,
        }

        _write(
            args.receipt_path,
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            append=False,
        )

        _write(
            args.summary_path,
            "## WizardOfOdds incumbent publication\n\n" f"REFUSED: {error}\n",
            append=True,
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_REFUSED

    _write(
        args.receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        append=False,
    )

    _write(args.summary_path, render(receipt), append=True)

    print(json.dumps(receipt, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
