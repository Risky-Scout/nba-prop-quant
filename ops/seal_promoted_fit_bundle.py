#!/usr/bin/env python3
"""Seal the promoted daily fit into a verified runtime bundle.

THE GAP THIS CLOSES
-------------------

``resolve_incumbent`` refuses to serve a promoted fit whose artifacts are not
the bytes in the serving tree, and names the step a human has to perform:
"seal a runtime bundle from the promoted fit". That refusal is right and is
not weakened here. What was missing is the step itself. Nothing in production
ever performed it, so a promoted fit could only ever produce a refusal, and
the only non-refusing path was to leave the fit unpromoted and serve the
August freeze -- which is what production did every day.

FREEZE IS NOT FIT
-----------------

The sealed bundle keeps the base bundle's ``freeze_id`` and ``freeze_stage``.
That is not a convenience: it is what the two identities mean. The freeze is
the certified architecture -- the routes, the families, the schema, the Gate 3
policy, the serving sources -- and a daily fit re-estimates numerical
parameters *under* that architecture and may not move any of it; the registry
refuses a candidate whose architecture contract does not match. So the
architecture the sealed bundle serves is still ``nba_prop_quant_20260818T205213Z``
and saying otherwise would make ``approved_freeze_id`` in the publishing
switch false about a bundle that in fact honours it.

What the sealed manifest gains is a ``daily_fit`` block naming the fit, its
training cutoff and the base manifest it was sealed over, so the bundle can
always say which day's parameters it carries.

HOW IT FAILS
------------

Into a staging directory, verified there, and moved into place in one
``os.replace`` only once every verification has passed. A failed seal leaves
the previous bundle exactly where it was and changes no pointer, so the
controlled fallback is the thing that was already serving. Nothing is written
to the registry, no promotion state is touched and no model is fitted: this
copies bytes the registry already sealed and re-hashes them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

if str(PROJECT_ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ops"))

from nba_prop_quant.adaptive_fit_registry import (  # noqa: E402
    FitRegistry,
    RegistryError,
    resolve_registry_root,
    sha256_file,
)

from install_frozen_model_artifacts import (  # noqa: E402
    MANIFEST_RELATIVE,
    MODEL_ARTIFACT_GROUP,
    MODEL_PREFIX,
    REQUIRED_MODEL_BINARIES,
    runtime_artifact_records,
    unrecorded_model_binaries,
    verify_frozen_bundle,
    verify_runtime_model_artifacts,
)

#: Where sealed bundles are kept, under the runtime root the installer uses.
SEALED_RELATIVE = Path("sealed_fit_bundles")

#: Written last inside the staging tree, so its presence at the final path
#: means the move completed over a fully verified bundle.
READY_RELATIVE = Path("SEALED_FIT_BUNDLE.json")

#: Where the sealed bundle's location is recorded durably, so a later job on
#: this machine can find the bundle in force without being handed it.
POINTER_RELATIVE = Path("processed") / "serving_bundle_pointer.json"

MODEL_DIR_VARIABLE = "FROZEN_RUNTIME_MODEL_DIR"
BUNDLE_ROOT_VARIABLE = "FROZEN_RUNTIME_BUNDLE_ROOT"

OUTCOME_SEALED = "SEALED"
OUTCOME_REUSED = "REUSED"
OUTCOME_NO_PROMOTED_FIT = "NO_PROMOTED_FIT"
OUTCOME_REFUSED = "SEAL_REFUSED"

EXIT_OK = 0
EXIT_FAILED = 1


class SealRefused(RuntimeError):
    """The promoted fit could not be sealed, so serving authority is unmoved."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fit_model_artifacts(manifest: dict[str, Any]) -> dict[str, str]:
    """The fit's artifacts that live under ``models/``, by relative path.

    These are exactly the records ``_promoted_fit_is_what_is_served``
    compares against the serving tree, so sealing them is what makes that
    comparison able to succeed.
    """
    return {
        relative: str(digest)
        for relative, digest in sorted(
            (manifest.get("artifact_hashes") or {}).items()
        )
        if Path(relative).parts[:1] == (MODEL_PREFIX,)
    }


def reseal_manifest(
    manifest: dict[str, Any],
    overlaid: dict[str, str],
    *,
    fit_id: str,
    training_cutoff: Any,
    base_manifest_sha256: str,
) -> dict[str, Any]:
    """The base manifest with the overlaid artifacts' digests corrected.

    Every record the base manifest already has is kept, in its own group, in
    its own position. A record whose path the fit overlaid gets the fit's
    digest; every other record keeps the digest the freeze gave it, because
    its bytes did not move. A ``models/`` artifact the fit carries that the
    base manifest never recorded is *added* to the runtime group rather than
    left out, so it is hashed and verified like everything else -- leaving it
    unrecorded would put a file serving can load outside the manifest, which
    is the one outcome a seal must not produce.
    """
    sealed = json.loads(json.dumps(manifest))

    files = sealed.setdefault("files", {})

    recorded: set[str] = set()
    corrected: list[str] = []

    for group, records in files.items():
        for record in records:
            relative = str(record["path"])

            recorded.add(relative)

            if relative in overlaid and record["sha256"] != overlaid[relative]:
                record["sha256"] = overlaid[relative]
                corrected.append(relative)

            elif relative in overlaid:
                corrected.append(relative)

    added = sorted(set(overlaid) - recorded)

    if added:
        group = files.setdefault(MODEL_ARTIFACT_GROUP, [])

        for relative in added:
            group.append(
                {
                    "path": relative,
                    "sha256": overlaid[relative],
                }
            )

        group.sort(key=lambda record: str(record["path"]))

    sealed["daily_fit"] = {
        "added_artifacts": added,
        "base_manifest_sha256": base_manifest_sha256,
        "corrected_artifacts": sorted(corrected),
        "fit_id": fit_id,
        "sealed_at": _utc_now(),
        "training_cutoff": (
            None if training_cutoff is None else str(training_cutoff)
        ),
    }

    return sealed


def _copy_tree(source: Path, destination: Path) -> int:
    """Copy the base bundle into staging, preserving every byte it has."""
    shutil.copytree(source, destination, symlinks=False)

    return sum(1 for path in destination.rglob("*") if path.is_file())


def seal(
    *,
    registry: FitRegistry,
    fit_id: str,
    base_bundle_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    """Build, verify and publish the sealed bundle for one promoted fit."""
    # The fit first, by the registry's own rules. A fit that does not verify
    # or whose fit-time validation does not pass is not a thing to seal, and
    # finding that out here rather than after copying a bundle is the point.
    registry.verify(fit_id)
    registry.assert_validation_pass(fit_id)

    manifest = registry.load_manifest(fit_id)

    overlaid = fit_model_artifacts(manifest)

    if not overlaid:
        raise SealRefused(
            f"fit {fit_id} records no artifact under {MODEL_PREFIX}/, so "
            "there is nothing a serving bundle could be sealed from"
        )

    base_root = Path(base_bundle_root)

    base_manifest_path = base_root / MODEL_PREFIX / MANIFEST_RELATIVE

    if not base_manifest_path.exists():
        raise SealRefused(
            f"the base bundle at {base_root} carries no frozen manifest at "
            f"{base_manifest_path}, so there is nothing to seal over"
        )

    # The base has to be a bundle serving would already have accepted.
    # Sealing over a tree that does not verify would produce a sealed bundle
    # whose non-fit records were never true.
    base_identity = verify_frozen_bundle(base_root)
    verify_runtime_model_artifacts(base_root / MODEL_PREFIX)

    base_manifest = _load_json(base_manifest_path)
    base_manifest_sha256 = sha256_file(base_manifest_path)

    sealed_root = (
        Path(runtime_root) / SEALED_RELATIVE / f"{fit_id}"
    )

    staging = (
        Path(runtime_root) / SEALED_RELATIVE / f".staging-{fit_id}"
    )

    if sealed_root.is_dir() and (sealed_root / READY_RELATIVE).exists():
        # Already sealed, and re-verified rather than trusted: the marker
        # says a seal completed, the verification says the tree still holds
        # what it completed over.
        identity = _verify_sealed(sealed_root, overlaid, fit_id)

        return {
            **identity,
            "base_freeze_id": base_identity["freeze_id"],
            "outcome": OUTCOME_REUSED,
            "reason": (
                f"the sealed bundle for {fit_id} already exists at "
                f"{sealed_root} and re-verified"
            ),
        }

    if staging.exists():
        shutil.rmtree(staging)

    staging.parent.mkdir(parents=True, exist_ok=True)

    copied = _copy_tree(base_root, staging)

    try:
        for relative, digest in overlaid.items():
            source = registry.fit_dir(fit_id) / "artifacts" / relative

            if not source.exists():
                raise SealRefused(
                    f"fit {fit_id} records {relative} but the registry has "
                    f"no such file at {source}, so the seal would be "
                    "incomplete"
                )

            if sha256_file(source) != digest:
                raise SealRefused(
                    f"fit {fit_id} records {relative} as {digest} but the "
                    "registry's own copy does not hash to it, so the fit is "
                    "not intact and may not be sealed"
                )

            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        sealed_manifest = reseal_manifest(
            base_manifest,
            overlaid,
            fit_id=fit_id,
            training_cutoff=manifest.get("training_cutoff"),
            base_manifest_sha256=base_manifest_sha256,
        )

        manifest_path = staging / MODEL_PREFIX / MANIFEST_RELATIVE
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(sealed_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        identity = _verify_sealed(staging, overlaid, fit_id)

        (staging / READY_RELATIVE).write_text(
            json.dumps(
                {
                    "base_bundle_root": str(base_root),
                    "base_freeze_id": base_identity["freeze_id"],
                    "base_manifest_sha256": base_manifest_sha256,
                    "fit_id": fit_id,
                    "sealed_at": _utc_now(),
                    "training_cutoff": str(
                        manifest.get("training_cutoff") or ""
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if sealed_root.exists():
        shutil.rmtree(sealed_root)

    os.replace(staging, sealed_root)

    identity = _verify_sealed(sealed_root, overlaid, fit_id)

    return {
        **identity,
        "base_freeze_id": base_identity["freeze_id"],
        "copied_files": copied,
        "outcome": OUTCOME_SEALED,
        "reason": (
            f"fit {fit_id} was sealed into {sealed_root} and the sealed "
            "bundle verified whole"
        ),
    }


def _verify_sealed(
    root: Path, overlaid: dict[str, str], fit_id: str
) -> dict[str, Any]:
    """Every check serving will make, made here before serving is pointed at it.

    Three, and they are three different statements:

        the whole manifest verifies against the tree
            the check ``10_predict_slate.py`` and ``15_price_markets.py``
            make before they load anything.

        every runtime model artifact the manifest names re-hashes inside the
        model directory
            the check ``resolve_incumbent`` makes through
            ``_bundle_identity``.

        every artifact the fit recorded is byte-for-byte what the serving
        tree holds
            the check ``_promoted_fit_is_what_is_served`` makes, and the one
            that decides whether the authority serving reports is the fit.

    A bundle that passes the first two and fails the third would serve
    happily while reporting an authority that is not in force, which is the
    exact failure this whole module exists to make impossible.
    """
    whole = verify_frozen_bundle(root)

    runtime = verify_runtime_model_artifacts(root / MODEL_PREFIX)

    manifest = _load_json(root / MODEL_PREFIX / MANIFEST_RELATIVE)

    unrecorded = unrecorded_model_binaries(
        runtime_artifact_records(manifest)
    )

    if unrecorded:
        raise SealRefused(
            f"the sealed bundle for {fit_id} does not record "
            + ", ".join(unrecorded)
            + ", which serving loads, so it would verify a set that is not "
            "the set that runs"
        )

    differing: list[str] = []

    for relative, digest in sorted(overlaid.items()):
        served = root / relative

        if not served.exists() or sha256_file(served) != digest:
            differing.append(relative)

    if differing:
        raise SealRefused(
            f"the sealed bundle for {fit_id} does not hold the fit's own "
            f"artifacts ({len(differing)} differ, first: {differing[0]}), so "
            "serving would report an authority that is not the one in force"
        )

    return {
        "bundle_root": str(root),
        "fit_id": fit_id,
        "freeze_id": str(whole["freeze_id"]),
        "freeze_stage": str(runtime["freeze_stage"]),
        "manifest_sha256": str(runtime["manifest_sha256"]),
        "model_dir": str(root / MODEL_PREFIX),
        "required_model_binaries": len(REQUIRED_MODEL_BINARIES),
        "sealed_artifacts": len(overlaid),
        "verified_manifest_entries": int(whole["verified_manifest_entries"]),
        "verified_runtime_artifacts": int(runtime["verified_artifacts"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    registry = FitRegistry(
        root=resolve_registry_root(
            args.registry_root, project_root=PROJECT_ROOT
        ),
        project_root=PROJECT_ROOT,
    )

    fit_id = args.fit_id or registry.current().get("current_good_fit_id")

    if fit_id is None:
        return {
            "evaluated_at": _utc_now(),
            "fit_id": None,
            "outcome": OUTCOME_NO_PROMOTED_FIT,
            "reason": (
                "no adaptive fit is promoted, so there is nothing to seal "
                "and the frozen deployment bundle remains the incumbent"
            ),
        }

    receipt = seal(
        registry=registry,
        fit_id=fit_id,
        base_bundle_root=Path(args.base_bundle_root),
        runtime_root=Path(args.runtime_root),
    )

    receipt["evaluated_at"] = _utc_now()

    if args.data_root is not None:
        pointer = Path(args.data_root) / POINTER_RELATIVE
        pointer.parent.mkdir(parents=True, exist_ok=True)
        temporary = pointer.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "bundle_root": receipt["bundle_root"],
                    "fit_id": receipt["fit_id"],
                    "freeze_id": receipt["freeze_id"],
                    "model_dir": receipt["model_dir"],
                    "published_at": _utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, pointer)
        receipt["pointer_path"] = str(pointer)

    return receipt


def render(receipt: dict[str, Any]) -> str:
    lines = [
        "## Promoted-fit runtime bundle",
        "",
        f"- Outcome: `{receipt['outcome']}`",
        f"- Fit: `{receipt.get('fit_id') or 'none'}`",
    ]

    if receipt.get("bundle_root"):
        lines += [
            f"- Bundle root: `{receipt['bundle_root']}`",
            f"- Model dir: `{receipt['model_dir']}`",
            f"- Architecture freeze: `{receipt['freeze_id']}` "
            f"(`{receipt['freeze_stage']}`)",
            f"- Sealed fit artifacts: {receipt['sealed_artifacts']}",
            f"- Verified manifest entries: "
            f"{receipt['verified_manifest_entries']}",
            f"- Verified runtime artifacts: "
            f"{receipt['verified_runtime_artifacts']}",
        ]

    lines += ["", receipt["reason"], ""]

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seal the promoted daily fit into a verified runtime bundle and "
            "publish it as the serving bundle."
        )
    )
    parser.add_argument("--registry-root", required=True)
    parser.add_argument("--base-bundle-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--fit-id", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--receipt-path", default=None)
    parser.add_argument("--github-env", default=None)
    parser.add_argument("--summary-path", default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except (RegistryError, SealRefused, OSError, RuntimeError) as error:
        receipt = {
            "evaluated_at": _utc_now(),
            "fit_id": args.fit_id,
            "outcome": OUTCOME_REFUSED,
            "reason": f"{type(error).__name__}: {error}",
        }

    if args.receipt_path is not None:
        path = Path(args.receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # The pointer is published only on a verified seal. A refusal leaves
    # whatever the installer exported standing, which is the base bundle --
    # the controlled fallback rather than a half-moved authority.
    if (
        args.github_env is not None
        and receipt["outcome"] in (OUTCOME_SEALED, OUTCOME_REUSED)
    ):
        with Path(args.github_env).open("a", encoding="utf-8") as handle:
            handle.write(f"{MODEL_DIR_VARIABLE}={receipt['model_dir']}\n")
            handle.write(
                f"{BUNDLE_ROOT_VARIABLE}={receipt['bundle_root']}\n"
            )

    text = render(receipt)

    print(text)

    if args.summary_path is not None:
        with Path(args.summary_path).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    return (
        EXIT_FAILED
        if receipt["outcome"] == OUTCOME_REFUSED
        else EXIT_OK
    )


if __name__ == "__main__":
    raise SystemExit(main())
