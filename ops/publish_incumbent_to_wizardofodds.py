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
        construction rather than by permission: ``local_directory`` is the only
        transport implemented, there is no live destination to reach, and a
        canary whose destination resolves to the live one is refused.

RELEASES, AND GETTING BACK
--------------------------

A release is written whole, verified by reading it back through the transport,
and only then promoted by replacing a pointer. The previous pointer is kept,
so rollback is restoring a pointer to bytes that are still sitting there --
not a rebuild, and nothing to do with the model. Promotion is the last step
because any earlier ordering has a window where the destination serves a
release that was never checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
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

#: The published record's own version. Independent of the two feed versions:
#: this names the shape of what is written to the destination, not the shape
#: of what was read from the runtime.
PUBLICATION_CONTRACT_VERSION = 1

RELEASES_DIRECTORY = "releases"
ROWS_NAME = "rows.json"
MANIFEST_NAME = "manifest.json"
CURRENT_POINTER = "current.json"
PREVIOUS_POINTER = "previous.json"

OUTCOME_PUBLISHED = "PUBLISHED"
OUTCOME_ROLLED_BACK = "ROLLED_BACK"

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
class Destination:
    mode: str
    transport: str
    root: Path
    release_prefix: str
    source: str


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
) -> Destination:
    """The destination for one mode, or a refusal naming what is missing.

    ``live`` is null in the committed configuration because every fact needed
    to describe it is external to this repository. The refusal quotes those
    facts rather than saying "not configured", because the difference between
    a missing line of config and a missing agreement with the destination is
    the whole difference between finishing this and being blocked on it.
    """
    source = str(document.get("source") or DESTINATIONS_RELATIVE)

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

    transport = str(declared.get("transport") or "")

    supported = tuple(document.get("supported_transports") or ())

    if transport not in supported:
        raise PublishingRefused(
            f"{source} asks for the {transport!r} transport for {mode}, and "
            f"only {list(supported)} is implemented"
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


def build_transport(destination: Destination) -> LocalDirectoryTransport:
    if destination.transport != TRANSPORT_LOCAL_DIRECTORY:
        raise PublishingRefused(
            f"the {destination.transport!r} transport is not implemented"
        )

    return LocalDirectoryTransport(destination.root)


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
        "readback_rows": len(readback.get("rows") or []),
        "rows_sha256": release.rows_sha256,
        "verified": True,
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

    superseded = read_pointer(transport, CURRENT_POINTER)

    transport.write_bytes(
        CURRENT_POINTER,
        serialise({**previous, "promoted_at": _utc_now()}),
    )

    return {
        "outcome": OUTCOME_ROLLED_BACK,
        "restored_release_id": previous.get("release_id"),
        "rows_sha256": actual,
        "superseded_release_id": (superseded or {}).get("release_id"),
    }


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
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One publication, start to finish, or the first refusal on the way.

    Ordered so the cheapest refusals come first and nothing is read from the
    destination before the switch has allowed this mode at all.
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

    destination = resolve_destination(mode, document, root=destination_root)

    if mode == MODE_CANARY:
        assert_canary_is_not_live(destination, document)

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

    transport = build_transport(destination)

    write_release(transport, release)

    verification = verify_release(transport, release)

    promotion = promote(transport, release, mode=mode)

    return {
        "destination": {
            "mode": destination.mode,
            "release_prefix": destination.release_prefix,
            "root": str(destination.root),
            "source": destination.source,
            "transport": destination.transport,
        },
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
        "slate_date": slate_date,
        "switch": switch.payload(),
        "verification": verification,
    }


def render(receipt: dict[str, Any]) -> str:
    if receipt.get("outcome") == OUTCOME_ROLLED_BACK:
        rows = [
            ("outcome", receipt["outcome"]),
            ("restored release", receipt.get("restored_release_id")),
            ("superseded release", receipt.get("superseded_release_id")),
        ]
    else:
        rows = [
            ("outcome", receipt.get("outcome")),
            ("mode", receipt.get("mode")),
            ("slate date", receipt.get("slate_date")),
            ("model authority", receipt.get("model_authority")),
            ("live publishing", receipt.get("live_publishing")),
            ("release", (receipt.get("release") or {}).get("release_id")),
            ("published rows", (receipt.get("release") or {}).get("row_count")),
            ("readback verified", (receipt.get("verification") or {}).get("verified")),
            ("destination", (receipt.get("destination") or {}).get("root")),
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
        help="canary writes a non-live destination. live is DISABLED and "
        "cannot be enabled from this repository: the host, directory, "
        "protocol and credentials are all external facts.",
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=None,
        help="where the release tree is written. Required for the canary, "
        "whose root is a scratch location this repository does not assert.",
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
            args.mode, document, root=args.destination_root
        )

        return {
            **rollback(build_transport(destination)),
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
