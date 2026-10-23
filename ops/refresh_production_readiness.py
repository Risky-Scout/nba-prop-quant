#!/usr/bin/env python3
"""Authorise a same-day readiness refresh for a newer production SHA.

WHY THIS EXISTS
---------------

``run_t20_production_controller.py`` will not serve a slate unless the durable
serving receipt for that slate was produced by the exact production SHA the
controller is itself running. That gate is right: a receipt produced by other
code does not prove the runtime about to serve was verified.

It has one operational consequence nobody chose. The morning lifecycle writes
the receipt at 09:37 UTC. If accepted production code is merged after that --
any ordinary merge to ``production/wizardofodds-integration`` -- then every
T-20 segment for the rest of that day refuses with
``MORNING_PRODUCTION_NOT_READY`` and the whole evening's slate silently
produces no forecast. That is what happened on 2026-10-21, 2026-10-22 and to
the late game of 2026-10-20.

The fix is not to weaken the gate. The receipt genuinely must describe the code
about to serve. The fix is to re-issue the receipt under the new SHA, which is
what this module authorises.

WHAT IT DOES AND DOES NOT DO
----------------------------

It does not serve anything and it does not write a receipt. It decides whether
a refresh is *permitted*, and if so it materialises the refresh-status document
that the supported serving entry point reads. The receipt is then written by
``run_incumbent_production_serving.py`` exactly as the morning lifecycle writes
it -- same entry point, same checks, same durable location.

It does not re-run the current-season data refresh. That refresh already ran
this morning and advanced the rolling state; running it a second time on the
same day would classify as no-new-data, which is not an outcome a slate may be
served under, so a naive rerun would fail for a reason that has nothing to do
with readiness. Instead the morning run's *own recorded* refresh
classification is carried forward, from the durable receipt, for the same
slate. Nothing is invented: if the morning run did not record a servable
refresh outcome for this slate, there is nothing to carry and this refuses.

It does not retrain, refit, recalibrate or promote anything. A source SHA
moving forward is not evidence about the model, and this module imports no
model, no fit registry and no calibration code.

WHAT MAKES IT SAFE
------------------

Four refusals, all fail-closed, and the narrowest possible authorisation:

``FORWARD ONLY``
    The recorded SHA must be a strict ancestor of the new SHA. That is the
    only situation this module exists for: accepted code moved forward on the
    production branch. A sideways SHA, an unrelated SHA, or a rollback is
    refused, because then the receipt is not merely stale -- it describes
    different code, and re-issuing it would be laundering.

``FITTED ARTIFACTS UNCHANGED``
    Every fitted artifact the freeze manifest pins must have the identical
    digest at both SHAs. If a model, copula, marginal, calibration or policy
    artifact changed, the morning run's proof says nothing about what would
    now be served and this refuses rather than refreshing.

``GOVERNANCE INTACT``
    At the new SHA the serving source contract must still describe the files
    it locks, and the adaptive update protocol must still point at that
    contract. A refresh is not a route around the byte-pinning.

``MORNING PROOF EXISTS``
    The receipt being replaced must itself record a serving outcome that
    proved readiness, a ready slate, a servable refresh outcome, and this
    slate date. A refresh can only ever carry forward a proof that was
    actually obtained; it cannot manufacture one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

#: Where the morning lifecycle's serving receipt durably lives. Stated the way
#: the controller states it so the two cannot drift apart silently; a test pins
#: them against each other.
MORNING_RECEIPT_RELATIVE = pathlib.Path("processed") / "incumbent_serving"

#: The serving outcomes that prove the morning run got through its own serving
#: step. Same set the controller gates on.
OUTCOMES_THAT_PROVE_READINESS: frozenset[str] = frozenset(
    {"SERVED", "NO_GAMES_ON_SLATE"}
)

#: The refresh outcomes a slate may be served under, and therefore the only
#: ones that can be carried forward.
REFRESH_OUTCOMES_THAT_SERVE: frozenset[str] = frozenset(
    {"REFRESHED", "OPENING_DAY_NO_REFRESH"}
)

#: The freeze manifest group holding fitted output. The other four groups are
#: source code, configuration and audit data: they are *expected* to move when
#: production code is merged, and the serving source contract is what governs
#: them. Only this group answers "did the model change".
FITTED_GROUP = "model_artifacts"

MANIFEST_RELATIVE = "models/frozen_manifests/LATEST.json"
CONTRACT_RELATIVE = (
    "models/frozen_manifests/"
    "nba_prop_quant_v2_adaptive_serving_source_contract.json"
)
PROTOCOL_RELATIVE = (
    "models/frozen_manifests/nba_prop_quant_v2_adaptive_update_protocol.json"
)

OUTCOME_AUTHORISED = "READINESS_REFRESH_AUTHORISED"
OUTCOME_NOT_REQUIRED = "READINESS_REFRESH_NOT_REQUIRED"
OUTCOME_REFUSED = "READINESS_REFRESH_REFUSED"

EXIT_OK = 0
EXIT_REFUSED = 1


class RefreshRefused(RuntimeError):
    """A readiness refresh is not permitted, and no receipt may be re-issued."""


def _git(project_root: pathlib.Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RefreshRefused(
            f"git {' '.join(arguments)} failed in {project_root}: "
            f"{result.stderr.strip() or 'no error output'}"
        )

    return result.stdout


def _blob_at(project_root: pathlib.Path, sha: str, relative: str) -> Any:
    """One tracked JSON file as it stands at one commit."""
    try:
        text = _git(project_root, "show", f"{sha}:{relative}")

    except RefreshRefused as error:
        raise RefreshRefused(
            f"{relative} cannot be read at {sha}, so the two revisions "
            f"cannot be compared: {error}"
        ) from error

    try:
        return json.loads(text)

    except ValueError as error:
        raise RefreshRefused(
            f"{relative} at {sha} is not readable JSON ({error})"
        ) from error


def _fitted_digests(manifest: Any, *, sha: str) -> dict[str, str]:
    files = (manifest or {}).get("files")

    if not isinstance(files, dict) or FITTED_GROUP not in files:
        raise RefreshRefused(
            f"the freeze manifest at {sha} records no {FITTED_GROUP} group, "
            "so whether the fitted artifacts changed cannot be established"
        )

    records = files[FITTED_GROUP]

    if not isinstance(records, list) or not records:
        raise RefreshRefused(
            f"the freeze manifest at {sha} records an empty {FITTED_GROUP} "
            "group, so there is nothing to compare"
        )

    digests: dict[str, str] = {}

    for record in records:
        path = str((record or {}).get("path") or "")
        digest = str((record or {}).get("sha256") or "")

        if not path or not digest:
            raise RefreshRefused(
                f"the freeze manifest at {sha} has a {FITTED_GROUP} record "
                "without a path and digest"
            )

        digests[path] = digest

    return digests


def require_forward_only(
    project_root: pathlib.Path, *, recorded_sha: str, production_sha: str
) -> None:
    """The recorded SHA must be a strict ancestor of the one about to serve."""
    for label, candidate in (
        ("the receipt's", recorded_sha),
        ("this run's", production_sha),
    ):
        result = subprocess.run(
            ("git", "cat-file", "-e", f"{candidate}^{{commit}}"),
            cwd=project_root,
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            raise RefreshRefused(
                f"{label} production SHA {candidate or 'nothing'} is not a "
                f"commit in {project_root}, so the two revisions cannot be "
                "related and the refresh is refused"
            )

    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", recorded_sha, production_sha),
        cwd=project_root,
        capture_output=True,
        check=False,
    )

    if ancestry.returncode != 0:
        raise RefreshRefused(
            f"the receipt was produced by {recorded_sha}, which is not an "
            f"ancestor of {production_sha}. A readiness refresh exists for "
            "one situation only -- accepted production code moved forward -- "
            "and this is not that situation, so the receipt is not merely "
            "stale and may not be re-issued"
        )


def require_fitted_artifacts_unchanged(
    project_root: pathlib.Path, *, recorded_sha: str, production_sha: str
) -> dict[str, Any]:
    """Fail closed if any fitted artifact differs between the two revisions."""
    before = _fitted_digests(
        _blob_at(project_root, recorded_sha, MANIFEST_RELATIVE),
        sha=recorded_sha,
    )

    after = _fitted_digests(
        _blob_at(project_root, production_sha, MANIFEST_RELATIVE),
        sha=production_sha,
    )

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(
        path for path in set(before) & set(after) if before[path] != after[path]
    )

    if added or removed or changed:
        raise RefreshRefused(
            "the fitted artifacts are not the ones the morning run verified, "
            "so its proof says nothing about what would be served and the "
            "refresh fails closed. "
            f"changed={changed or 'none'} added={added or 'none'} "
            f"removed={removed or 'none'}"
        )

    return {
        "fitted_artifact_count": len(after),
        "fitted_artifacts_identical": True,
        "freeze_manifest": MANIFEST_RELATIVE,
    }


def require_governance_intact(
    project_root: pathlib.Path, *, production_sha: str
) -> dict[str, Any]:
    """The serving source contract and its protocol pointer must still hold."""
    contract_text = _git(
        project_root, "show", f"{production_sha}:{CONTRACT_RELATIVE}"
    )

    contract = json.loads(contract_text)
    locked = contract.get("locked_serving_source_files")

    if not isinstance(locked, dict) or not locked:
        raise RefreshRefused(
            f"{CONTRACT_RELATIVE} at {production_sha} locks no serving source "
            "files, so the byte-pinning cannot be checked"
        )

    divergent: list[str] = []

    for relative, entry in sorted(locked.items()):
        recorded = str((entry or {}).get("current_sha256") or "")

        actual = hashlib.sha256(
            _git(
                project_root, "show", f"{production_sha}:{relative}"
            ).encode()
        ).hexdigest()

        if recorded != actual:
            divergent.append(relative)

    if divergent:
        raise RefreshRefused(
            "the serving source contract no longer describes the files it "
            f"locks at {production_sha}, so a refresh would be a route "
            f"around the byte-pinning: {', '.join(divergent)}"
        )

    protocol = _blob_at(project_root, production_sha, PROTOCOL_RELATIVE)

    pointer = str(
        protocol.get("adaptive_serving_source_contract_sha256") or ""
    )

    contract_digest = hashlib.sha256(contract_text.encode()).hexdigest()

    if pointer != contract_digest:
        raise RefreshRefused(
            "the adaptive update protocol does not point at the serving "
            f"source contract at {production_sha}: it names {pointer or 'nothing'} "
            f"and the contract digests to {contract_digest}"
        )

    return {
        "locked_serving_source_files": len(locked),
        "serving_source_contract": CONTRACT_RELATIVE,
        "serving_source_contract_sha256": contract_digest,
        "update_protocol": PROTOCOL_RELATIVE,
    }


def read_morning_receipt(
    *, data_root: pathlib.Path, slate_date: str
) -> dict[str, Any]:
    """The proof being carried forward, or a refusal naming what is missing."""
    path = pathlib.Path(data_root) / MORNING_RECEIPT_RELATIVE / f"{slate_date}.json"

    if not path.exists():
        raise RefreshRefused(
            f"{path} does not exist, so no run has recorded a verified "
            f"incumbent for {slate_date} and there is no proof to carry "
            "forward. The morning lifecycle must run before a refresh can"
        )

    try:
        receipt = json.loads(path.read_text())

    except ValueError as error:
        raise RefreshRefused(
            f"{path} is not readable JSON ({error})"
        ) from error

    if not isinstance(receipt, dict):
        raise RefreshRefused(f"{path} does not hold a receipt object")

    recorded_slate = str(receipt.get("slate_date") or "")

    if recorded_slate != slate_date:
        raise RefreshRefused(
            f"{path} is the receipt for {recorded_slate or 'no slate'}, not "
            f"{slate_date}"
        )

    outcome = str(receipt.get("outcome") or "")

    if outcome not in OUTCOMES_THAT_PROVE_READINESS:
        raise RefreshRefused(
            f"{path} records the serving outcome {outcome or 'nothing'}, "
            "which did not prove readiness, so there is nothing to carry "
            "forward. Expected one of "
            + ", ".join(sorted(OUTCOMES_THAT_PROVE_READINESS))
        )

    readiness = receipt.get("slate_readiness") or {}

    if readiness.get("ready") is not True:
        raise RefreshRefused(
            f"{path} records that the slate was not ready to serve, so a "
            "refresh cannot make it ready"
        )

    refresh_outcome = str(readiness.get("refresh_outcome") or "")

    if refresh_outcome not in REFRESH_OUTCOMES_THAT_SERVE:
        raise RefreshRefused(
            f"{path} records the refresh outcome "
            f"{refresh_outcome or 'nothing'}, which is not one a slate may "
            "be served under, so it may not be carried forward"
        )

    return receipt


def authorise(
    *,
    data_root: pathlib.Path,
    slate_date: str,
    production_sha: str,
    project_root: pathlib.Path,
) -> dict[str, Any]:
    """Decide whether this production SHA may re-issue today's receipt."""
    receipt = read_morning_receipt(data_root=data_root, slate_date=slate_date)

    recorded_sha = str(receipt.get("production_code_sha") or "")

    if not recorded_sha:
        raise RefreshRefused(
            "the receipt does not name the production SHA that produced it, "
            "so whether code moved forward cannot be established"
        )

    readiness = receipt["slate_readiness"]

    if recorded_sha == production_sha:
        return {
            "carried_refresh_outcome": readiness["refresh_outcome"],
            "generated_at": datetime.now(UTC).isoformat(),
            "outcome": OUTCOME_NOT_REQUIRED,
            "production_code_sha": production_sha,
            "reason": (
                f"the receipt for {slate_date} was already produced by "
                f"{production_sha}, so it describes the code about to serve "
                "and nothing needs re-issuing"
            ),
            "recorded_production_code_sha": recorded_sha,
            "slate_date": slate_date,
        }

    require_forward_only(
        project_root,
        recorded_sha=recorded_sha,
        production_sha=production_sha,
    )

    fitted = require_fitted_artifacts_unchanged(
        project_root,
        recorded_sha=recorded_sha,
        production_sha=production_sha,
    )

    governance = require_governance_intact(
        project_root, production_sha=production_sha
    )

    return {
        "carried_refresh_outcome": readiness["refresh_outcome"],
        "fitted_artifacts": fitted,
        "generated_at": datetime.now(UTC).isoformat(),
        "governance": governance,
        "morning_outcome": str(receipt["outcome"]),
        "outcome": OUTCOME_AUTHORISED,
        "production_code_sha": production_sha,
        "reason": (
            f"accepted production code moved forward from {recorded_sha} to "
            f"{production_sha} with every fitted artifact identical and the "
            "serving source governance intact, so the readiness proof the "
            f"morning run obtained for {slate_date} may be re-issued under "
            "the new revision"
        ),
        "recorded_at": receipt.get("generated_at"),
        "recorded_production_code_sha": recorded_sha,
        "slate_date": slate_date,
    }


def carried_refresh_status(decision: dict[str, Any]) -> dict[str, Any]:
    """The refresh-status document the serving entry point will read.

    Deliberately minimal and deliberately not a fabrication: it carries the
    outcome the morning run itself recorded, and the slate date it recorded it
    for, which are the two fields the serving entry point reads and compares.
    Everything else in it is provenance, so a human reading the file can see
    where the classification came from.
    """
    return {
        "carried_forward_from": {
            "production_code_sha": decision["recorded_production_code_sha"],
            "recorded_at": decision.get("recorded_at"),
            "receipt": str(
                MORNING_RECEIPT_RELATIVE / f"{decision['slate_date']}.json"
            ),
        },
        "note": (
            "This is not a new refresh. The current-season rolling state was "
            "advanced once today by the morning lifecycle and this is that "
            "run's own recorded classification, carried forward so the "
            "serving receipt can be re-issued under a newer production SHA "
            "without re-running the refresh or refitting anything."
        ),
        "outcome": decision["carried_refresh_outcome"],
        "slate_date": decision["slate_date"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="refresh_production_readiness",
        description=(
            "Authorise re-issuing today's production readiness receipt under "
            "a newer production SHA. Refuses unless accepted code moved "
            "strictly forward, every fitted artifact is identical, the "
            "serving source governance is intact, and the run being carried "
            "forward actually proved readiness."
        ),
    )

    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--production-sha",
        required=True,
        help="the production checkout about to serve",
    )
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--refresh-status-path",
        type=pathlib.Path,
        default=None,
        help="where the carried-forward refresh status is written, to be "
        "passed to run_incumbent_production_serving.py --refresh-status",
    )
    parser.add_argument("--decision-path", type=pathlib.Path, default=None)
    parser.add_argument("--summary-path", type=pathlib.Path, default=None)
    parser.add_argument("--github-output", type=pathlib.Path, default=None)

    return parser.parse_args(argv)


def _write(path: pathlib.Path | None, text: str, *, append: bool = False) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        decision = authorise(
            data_root=args.data_root,
            slate_date=args.slate_date,
            production_sha=args.production_sha,
            project_root=args.project_root,
        )

    except (RefreshRefused, OSError) as error:
        decision = {
            "error": type(error).__name__,
            "generated_at": datetime.now(UTC).isoformat(),
            "outcome": OUTCOME_REFUSED,
            "production_code_sha": args.production_sha,
            "reason": str(error),
            "slate_date": args.slate_date,
        }

        _write(
            args.decision_path,
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
        )

        _write(
            args.summary_path,
            "## Production readiness refresh\n\n"
            f"{OUTCOME_REFUSED}: {error}\n",
            append=True,
        )

        print(json.dumps(decision, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_REFUSED

    _write(
        args.decision_path,
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
    )

    if decision["outcome"] == OUTCOME_AUTHORISED:
        _write(
            args.refresh_status_path,
            json.dumps(carried_refresh_status(decision), indent=2, sort_keys=True)
            + "\n",
        )

    _write(
        args.summary_path,
        "## Production readiness refresh\n\n"
        f"{decision['outcome']}: {decision['reason']}\n",
        append=True,
    )

    _write(
        args.github_output,
        f"outcome={decision['outcome']}\n"
        f"refresh_required="
        f"{'true' if decision['outcome'] == OUTCOME_AUTHORISED else 'false'}\n",
        append=True,
    )

    print(json.dumps(decision, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
