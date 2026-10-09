#!/usr/bin/env python3
"""Produce the incumbent's slate projections and priced markets, every day.

The lifecycle refreshed state, fitted a candidate and shadowed it, and then
stopped. Nothing automated ever ran the two scripts that produce the artifact
production actually serves -- ``scripts/10_predict_slate.py`` and
``scripts/15_price_markets.py`` -- so the daily priced slate was a manual act.
This is the step that runs them.

WHOSE MODEL
-----------

The incumbent's, resolved from the registry's promotion state and nothing
else. Specifically not: the candidate the adaptive fit just registered, the
shadow candidate, an unpromoted fit, or whichever fit directory has the newest
timestamp. This module never reads the adaptive fit's status file and never
lists the registry's fit directories, because both are ways to end up serving
something nobody approved.

The repository's own authority contract has two layers and this module
honours both. ``state/promotion_state.json`` records which adaptive fit is
approved; the frozen deployment bundle in the model directory is the sealed
set of artifacts the serving scripts read. ADAPTIVE_FIT_REGISTRY.md §12 is
explicit that "a promoted fit is an input to a future bundle, not a
replacement for one". So:

    no promoted fit
        The frozen bundle is the incumbent. This is the state production is
        in before the first human promotion, and serving from it is correct.

    a promoted fit whose artifacts are what the bundle holds
        The incumbent is that fit, named by ``fit_id``. Established by
        comparing the fit manifest's recorded artifact digests against the
        bytes in the serving tree, not by trusting a label.

    a promoted fit whose artifacts are not what the bundle holds
        Refusal. The approved fit has not been sealed into a serving bundle,
        so serving the old bundle would report an authority that is not the
        one in force. The refusal names the step a human has to perform.

FAILING CLOSED
--------------

A valid production slate that produces no priced markets is a production
failure and exits nonzero, so the lifecycle goes red. There is no
continue-on-error on this step and this module has no fallback model: an
unresolvable incumbent is a refusal, never a substitution.

Two outcomes are *not* failures, and the difference is the whole reason this
module decides readiness instead of the YAML:

    NO_PRODUCTION_SLATE
        The refresh reported that nothing was written -- before opening day,
        that is the preseason no-op. There is no slate to serve.

    NO_GAMES_ON_SLATE
        A regular-season date with no games on it. The prediction script's own
        documented behaviour is to report that and write nothing, and pricing
        a slate of nothing is not a thing to attempt.

Readiness is read from the refresh classifier's status file rather than from
``RUN_ADAPTIVE``. They happen to agree today, but they are different
questions: whether the incumbent should price tonight's games has nothing to
do with whether a candidate was refitted this morning, and binding them would
mean any future off-day fit policy silently stopped production serving.

WHAT IT DOES NOT DO
-------------------

It does not publish. It does not promote. It does not fit, refit or
recalibrate. It writes the two artifacts the serving scripts write, plus its
own provenance receipt, and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.adaptive_fit_registry import (  # noqa: E402
    FitRegistry,
    RegistryError,
    resolve_registry_root,
    sha256_file,
)
from nba_prop_quant.production import (  # noqa: E402
    load_verified_manifest_metadata,
)

#: The two scripts that serve a slate. Invoked rather than reimplemented: a
#: second prediction path here would be a second model.
PREDICT_SCRIPT = Path("scripts") / "10_predict_slate.py"
PRICE_SCRIPT = Path("scripts") / "15_price_markets.py"

#: Where those scripts write, relative to the data root.
PROJECTIONS_RELATIVE = Path("processed") / "projections"
PRICED_MARKETS_RELATIVE = Path("processed") / "priced_markets"

#: The refresh's own semantic fingerprint of the live rolling tree. Used as
#: the input state fingerprint so the receipt names the data generation the
#: projections were produced from.
SEASON_STATE_RELATIVE = Path(".state") / "current_season_state.json"

#: The refresh outcome that means a real regular-season slate exists.
REFRESH_OUTCOME_REFRESHED = "REFRESHED"

MODEL_AUTHORITY = "incumbent"

AUTHORITY_PROMOTED_FIT = "promoted_fit"
AUTHORITY_FROZEN_BUNDLE = "frozen_deployment_bundle"

OUTCOME_SERVED = "SERVED"
OUTCOME_NO_PRODUCTION_SLATE = "NO_PRODUCTION_SLATE"
OUTCOME_NO_GAMES_ON_SLATE = "NO_GAMES_ON_SLATE"
OUTCOME_FAILED = "SERVING_FAILED"

EXIT_OK = 0
EXIT_FAILED = 1

#: Every field the receipt must carry a value for before a served run counts
#: as provenanced. Checked before the receipt is written, so an incomplete
#: receipt is a failure rather than a quietly thinner record.
#:
#: ``incumbent_version`` carries the identity the brief asks for: the promoted
#: ``fit_id`` when one is in force and the bundle's ``freeze_id`` when none is.
#: ``incumbent_fit_id`` is required to be *present* rather than non-empty,
#: because null is the true and meaningful value before the first human
#: promotion and demanding a fit id there would only invite inventing one.
REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "generated_at",
    "incumbent_authority",
    "incumbent_version",
    "input_state_fingerprint",
    "model_authority",
    "prediction_artifact",
    "priced_market_artifact",
    "production_code_sha",
    "slate_date",
)

#: Fields that must appear, whose value may legitimately be null.
REQUIRED_PROVENANCE_KEYS: tuple[str, ...] = ("incumbent_fit_id",)


class ServingRefused(RuntimeError):
    """The incumbent could not be resolved, so nothing was served."""


class ServingFailed(RuntimeError):
    """A valid production slate failed to produce its required artifacts."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# incumbent authority
# ----------------------------------------------------------------------


def _bundle_identity(model_dir: Path, project_root: Path) -> dict[str, str]:
    """The frozen bundle's own verified identity.

    ``load_verified_manifest_metadata`` re-hashes every file the manifest
    names, so this is a verification and not a read. A corrupt or incomplete
    serving tree raises here, which is the fail-closed behaviour: production
    must not price from artifacts that do not match their manifest.
    """
    metadata = load_verified_manifest_metadata(
        model_dir=model_dir,
        project_root=project_root,
    )

    return {
        "freeze_id": str(metadata["freeze_id"]),
        "freeze_stage": str(metadata["freeze_stage"]),
        "manifest_sha256": str(metadata["manifest_sha256"]),
    }


def _promoted_fit_is_what_is_served(
    registry: FitRegistry,
    fit_id: str,
    model_dir: Path,
) -> tuple[bool, list[str]]:
    """Whether the serving tree holds exactly the promoted fit's artifacts.

    Compared by digest. A fit manifest records its artifacts under the
    candidate tree's own ``models/`` prefix, which is the serving model
    directory, so the comparison is between the bytes the registry sealed and
    the bytes the serving scripts will load.
    """
    manifest = registry.load_manifest(fit_id)

    hashes: dict[str, str] = dict(manifest.get("artifact_hashes") or {})

    differing: list[str] = []

    for relative, expected in sorted(hashes.items()):
        candidate = Path(relative)

        if candidate.parts[:1] != ("models",):
            continue

        served = model_dir / Path(*candidate.parts[1:])

        if not served.exists() or sha256_file(served) != expected:
            differing.append(relative)

    return (not differing), differing


def resolve_incumbent(
    *,
    registry: FitRegistry,
    model_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Resolve the authority this run is allowed to serve from.

    Reads the promotion state and the frozen bundle. It does not list fit
    directories, inspect timestamps or look at the adaptive fit's output,
    because every one of those can name a fit nobody promoted.
    """
    try:
        state = registry.current()

    except RegistryError as error:
        raise ServingRefused(
            f"the registry's promotion state could not be read, so the "
            f"incumbent is unknown and nothing may be served: {error}"
        ) from error

    fit_id = state.get("current_good_fit_id")

    bundle = _bundle_identity(model_dir, project_root)

    if fit_id is None:
        return {
            "authority": AUTHORITY_FROZEN_BUNDLE,
            "fit_id": None,
            "frozen_bundle": bundle,
            "model_dir": str(model_dir),
            "promoted_at": None,
            "reason": (
                "no adaptive fit has been promoted, so the frozen deployment "
                "bundle is the incumbent serving authority"
            ),
            "version": bundle["freeze_id"],
        }

    # A promoted fit that cannot be verified must never serve, and must never
    # be silently replaced by the bundle either.
    try:
        registry.verify(fit_id)
        registry.assert_validation_pass(fit_id)

    except RegistryError as error:
        raise ServingRefused(
            f"the promoted fit {fit_id} did not verify, so it may not serve "
            f"and nothing else may serve in its place: {error}"
        ) from error

    sealed, differing = _promoted_fit_is_what_is_served(
        registry, fit_id, model_dir
    )

    if not sealed:
        raise ServingRefused(
            f"fit {fit_id} is the promoted incumbent but the serving tree "
            f"at {model_dir} does not hold its artifacts "
            f"({len(differing)} differ, first: {differing[0]}). The approved "
            "fit has not been sealed into a serving bundle, so serving would "
            "report an authority that is not the one in force. A human must "
            "seal a runtime bundle from the promoted fit before this step can "
            "serve it."
        )

    return {
        "authority": AUTHORITY_PROMOTED_FIT,
        "fit_id": fit_id,
        "frozen_bundle": bundle,
        "model_dir": str(model_dir),
        "promoted_at": state.get("promoted_at"),
        "reason": (
            f"fit {fit_id} is the current good fit and the serving tree holds "
            "exactly its artifacts"
        ),
        "version": fit_id,
    }


# ----------------------------------------------------------------------
# readiness
# ----------------------------------------------------------------------


def slate_readiness(refresh_status: Path | None) -> dict[str, Any]:
    """Whether a real production slate exists to serve.

    Decided from the refresh classifier's own status file. ``writes_performed``
    is the fact that matters: before opening day the refresh writes nothing
    and there is no slate, and the classifier has already verified that claim
    against the locked opening day.
    """
    if refresh_status is None:
        return {
            "ready": True,
            "reason": (
                "no refresh status was supplied, so readiness was not "
                "narrowed and the slate is attempted"
            ),
            "refresh_outcome": None,
        }

    path = Path(refresh_status)

    if not path.exists():
        raise ServingRefused(
            f"the refresh status file {path} does not exist, so whether a "
            "production slate exists cannot be established"
        )

    status = _load_json(path)

    outcome = status.get("outcome")

    if outcome == REFRESH_OUTCOME_REFRESHED:
        return {
            "ready": True,
            "reason": "the current-season rolling state advanced",
            "refresh_outcome": outcome,
        }

    return {
        "ready": False,
        "reason": (
            f"the refresh reported {outcome}, so no regular-season slate "
            "exists to serve"
        ),
        "refresh_outcome": outcome,
    }


def input_state_fingerprint(data_root: Path) -> str | None:
    """The refresh's own fingerprint of the live rolling tree."""
    path = Path(data_root) / SEASON_STATE_RELATIVE

    if not path.exists():
        return None

    record = _load_json(path)

    value = record.get("datasets_fingerprint")

    return str(value) if value else None


# ----------------------------------------------------------------------
# running the serving scripts
# ----------------------------------------------------------------------


def _serving_environment(model_dir: Path, data_root: Path) -> dict[str, str]:
    """The environment the serving scripts resolve their inputs from.

    The model directory is passed explicitly so the incumbent resolution above
    is what decides which artifacts get loaded, rather than whatever the
    ambient settings happen to point at.
    """
    environment = dict(os.environ)

    environment["NBA_PROP_MODEL_DIR"] = str(model_dir)
    environment["NBA_PROP_DATA_DIR"] = str(data_root)

    return environment


def _run(
    script: Path,
    arguments: list[str],
    *,
    project_root: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=str(project_root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def serve(
    *,
    slate_date: str,
    incumbent: dict[str, Any],
    project_root: Path,
    data_root: Path,
    predict_arguments: list[str] | None = None,
    price_arguments: list[str] | None = None,
) -> dict[str, Any]:
    """Run prediction then pricing, and report what they produced.

    The empty-slate distinction rests on the prediction script's own
    behaviour: it fails closed on every error it can detect and returns 0
    without writing only on the documented no-games branch. So exit 0 with no
    artifact is read as an empty slate, and any nonzero exit is a failure.
    """
    model_dir = Path(incumbent["model_dir"])

    environment = _serving_environment(model_dir, data_root)

    projection_path = (
        Path(data_root) / PROJECTIONS_RELATIVE / f"{slate_date}.parquet"
    )

    priced_path = (
        Path(data_root) / PRICED_MARKETS_RELATIVE / f"{slate_date}.parquet"
    )

    prediction = _run(
        PREDICT_SCRIPT,
        ["--date", slate_date, *(predict_arguments or [])],
        project_root=project_root,
        environment=environment,
    )

    if prediction.returncode != 0:
        raise ServingFailed(
            f"{PREDICT_SCRIPT} exited {prediction.returncode} for slate "
            f"{slate_date}: {(prediction.stderr or prediction.stdout)[-2000:]}"
        )

    if not projection_path.exists():
        return {
            "outcome": OUTCOME_NO_GAMES_ON_SLATE,
            "prediction_artifact": None,
            "priced_market_artifact": None,
            "reason": (
                f"{PREDICT_SCRIPT} reported no games for {slate_date} and "
                "wrote no projections, so there was nothing to price"
            ),
        }

    pricing = _run(
        PRICE_SCRIPT,
        ["--date", slate_date, *(price_arguments or [])],
        project_root=project_root,
        environment=environment,
    )

    if pricing.returncode != 0:
        raise ServingFailed(
            f"{PRICE_SCRIPT} exited {pricing.returncode} for slate "
            f"{slate_date}: {(pricing.stderr or pricing.stdout)[-2000:]}"
        )

    if not priced_path.exists():
        raise ServingFailed(
            f"{PRICE_SCRIPT} exited 0 for slate {slate_date} but wrote no "
            f"priced markets at {priced_path}"
        )

    return {
        "outcome": OUTCOME_SERVED,
        "prediction_artifact": _artifact(projection_path),
        "priced_market_artifact": _artifact(priced_path),
        "reason": (
            f"the incumbent produced projections and priced markets for "
            f"{slate_date}"
        ),
    }


# ----------------------------------------------------------------------
# receipt
# ----------------------------------------------------------------------


def build_receipt(
    *,
    slate_date: str,
    production_sha: str | None,
    incumbent: dict[str, Any],
    readiness: dict[str, Any],
    fingerprint: str | None,
    served: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "incumbent_authority": incumbent["authority"],
        "incumbent_fit_id": incumbent["fit_id"],
        "incumbent_reason": incumbent["reason"],
        "incumbent_version": incumbent["version"],
        "input_state_fingerprint": fingerprint,
        "model_authority": MODEL_AUTHORITY,
        "outcome": served["outcome"],
        "prediction_artifact": served["prediction_artifact"],
        "priced_market_artifact": served["priced_market_artifact"],
        "production_code_sha": production_sha,
        "promoted_at": incumbent["promoted_at"],
        "reason": served["reason"],
        "serving_model_dir": incumbent["model_dir"],
        "slate_date": slate_date,
        "slate_readiness": readiness,
    }


def missing_provenance(receipt: dict[str, Any]) -> list[str]:
    """Required provenance fields the receipt does not actually carry.

    Only meaningful for a run that served something: a no-slate day has no
    artifact to provenance, and demanding one would turn the preseason no-op
    into a failure.
    """
    if receipt.get("outcome") != OUTCOME_SERVED:
        return []

    return sorted(
        [
            name
            for name in REQUIRED_PROVENANCE_FIELDS
            if receipt.get(name) in (None, "", {}, [])
        ]
        + [name for name in REQUIRED_PROVENANCE_KEYS if name not in receipt]
    )


def render(receipt: dict[str, Any]) -> str:
    rows = [
        ("outcome", receipt["outcome"]),
        ("slate date", receipt["slate_date"]),
        ("model authority", receipt["model_authority"]),
        ("incumbent authority", receipt["incumbent_authority"]),
        ("incumbent version", receipt["incumbent_version"]),
        ("incumbent fit id", receipt["incumbent_fit_id"] or "none promoted"),
        ("production code SHA", receipt["production_code_sha"] or "unknown"),
        ("input state fingerprint", receipt["input_state_fingerprint"] or "n/a"),
    ]

    for label, key in (
        ("projections", "prediction_artifact"),
        ("priced markets", "priced_market_artifact"),
    ):
        artifact = receipt.get(key)

        rows.append(
            (label, artifact["sha256"][:16] if artifact else "not produced")
        )

    lines = [
        "## Incumbent production serving",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    lines += [f"| {label} | {value} |" for label, value in rows]

    lines += ["", receipt["reason"]]

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_incumbent_production_serving",
        description=(
            "Produce the incumbent's slate projections and priced markets. "
            "Serves the current good fit only; publishes and promotes nothing."
        ),
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help="the fit registry holding the promotion state. The incumbent is "
        "resolved from it; fit directories are never listed.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="the serving artifact tree holding frozen_manifests/LATEST.json. "
        "Defaults to the checkout's models/ directory.",
    )
    parser.add_argument(
        "--refresh-status",
        type=Path,
        default=None,
        help="the JSON ops/classify_refresh_outcome.py emitted. Readiness is "
        "read from it rather than from RUN_ADAPTIVE.",
    )
    parser.add_argument("--production-sha", default=None)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)

    return parser.parse_args(argv)


def _write(path: Path | None, text: str, *, append: bool) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()

    data_root = Path(args.data_root).resolve()

    model_dir = (
        Path(args.model_dir).resolve()
        if args.model_dir is not None
        else project_root / "models"
    )

    readiness = slate_readiness(args.refresh_status)

    registry = FitRegistry(
        root=resolve_registry_root(
            args.registry_root, project_root=project_root
        ),
        project_root=project_root,
    )

    incumbent = resolve_incumbent(
        registry=registry,
        model_dir=model_dir,
        project_root=project_root,
    )

    if not readiness["ready"]:
        served = {
            "outcome": OUTCOME_NO_PRODUCTION_SLATE,
            "prediction_artifact": None,
            "priced_market_artifact": None,
            "reason": readiness["reason"],
        }

    else:
        served = serve(
            slate_date=args.slate_date,
            incumbent=incumbent,
            project_root=project_root,
            data_root=data_root,
        )

    receipt = build_receipt(
        slate_date=args.slate_date,
        production_sha=args.production_sha,
        incumbent=incumbent,
        readiness=readiness,
        fingerprint=input_state_fingerprint(data_root),
        served=served,
    )

    incomplete = missing_provenance(receipt)

    if incomplete:
        raise ServingFailed(
            "the serving receipt is missing required provenance: "
            + ", ".join(incomplete)
        )

    return receipt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except (ServingRefused, ServingFailed, RegistryError, OSError) as error:
        failure = {
            "error": type(error).__name__,
            "message": str(error),
            "model_authority": MODEL_AUTHORITY,
            "outcome": OUTCOME_FAILED,
            "slate_date": args.slate_date,
        }

        _write(
            args.receipt_path,
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            append=False,
        )

        _write(
            args.summary_path,
            "## Incumbent production serving\n\n"
            f"FAILED: {error}\n",
            append=True,
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_FAILED

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
