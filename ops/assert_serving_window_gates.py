#!/usr/bin/env python3
"""Hold the serving window to the two checks a morning fit cannot answer.

WHY THIS EXISTS
---------------

``REQUIRED_VALIDATION_CHECKS`` has fifteen names. Thirteen of them judge the
fitted model and are answerable from the candidate tree the moment it is
built. Two do not judge the model at all:

    gate3_role_readiness
        needs a live captured lineup snapshot for the slate.

    t20_protocol_compatible
        needs a real capture at the frozen serving offset for the slate.

Both were promotion prerequisites, and both ask about a capture that is taken
in the evening about a fit produced that morning. So promotion required
information that does not exist when promotion has to happen, and the daily
fit was never promotable -- which is why production served an August freeze
while a validated fit sat in the registry every day.

Moving them off the promotion prerequisite would weaken the system if that
were all that happened. It is not. They are enforced here instead, and here
is strictly more checking than there:

    there
        answered once, from whatever capture happened to exist when the
        validation record was written, and then never asked again. A fit
        promoted on Monday carried Monday's answer into every slate after it.

    here
        re-evaluated at the window that is about to publish, against that
        window's own capture, every time. Yesterday's answer cannot satisfy
        today's window, and a recorded ``true`` is not read as evidence --
        the check is recomputed from the capture on disk.

WHAT IT REFUSES
---------------

``t20_protocol_compatible`` is required of every production window, whatever
the serving authority. The caller reaches this script only after it has
captured and verified the window's group, so a capture that is absent or will
not resolve to a verified bundle at the frozen offset is an anomaly in the
one place the whole pricing path depends on, and is refused.

``gate3_role_readiness`` is required when the serving authority is a promoted
daily fit. The check reads the fit's own recorded role-state hash, so it is a
question about that fit against that capture. With no promoted fit the
frozen deployment bundle is the authority, the question has no subject, and
the check is reported rather than enforced -- which leaves the pre-existing
frozen-bundle path behaving exactly as it already did.

Nothing here fits, prices, promotes, publishes or writes a model artifact. It
reads captures and a validation record, recomputes two checks, and either
refuses or records what it measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.adaptive_fit_registry import (  # noqa: E402
    SERVING_WINDOW_VALIDATION_CHECKS,
    FitRegistry,
    RegistryError,
    resolve_registry_root,
)
from nba_prop_quant.adaptive_training import (  # noqa: E402
    snapshot_directory,
)
from nba_prop_quant.adaptive_validation import (  # noqa: E402
    check_gate3_role_readiness,
    check_t20_protocol_compatible,
)

AUTHORITY_PROMOTED_FIT = "promoted_fit"
AUTHORITY_FROZEN_BUNDLE = "frozen_deployment_bundle"

OUTCOME_PASS = "SERVING_WINDOW_GATES_PASS"
OUTCOME_REFUSED = "SERVING_WINDOW_GATES_REFUSED"

EXIT_OK = 0
EXIT_FAILED = 1


class WindowGateRefused(RuntimeError):
    """A serving-window gate did not pass, so this window may not publish."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def evaluate_window_gates(
    *,
    project_root: Path,
    slate_date: str,
    data_root: Path,
    registry: FitRegistry | None,
) -> dict[str, Any]:
    """Recompute both window checks against this slate's capture on disk.

    Returns the measurement. Deciding what to do about it is
    :func:`assert_window_gates`, so the measurement can be reported by a
    caller that is not entitled to refuse.
    """
    snapshot_dir = snapshot_directory(Path(data_root))

    fit_id: str | None = None
    role_state_hash: Any = None
    candidate: Path | None = None

    if registry is not None:
        try:
            fit_id = registry.current().get("current_good_fit_id")
        except RegistryError as error:
            raise WindowGateRefused(
                "the registry's promotion state could not be read, so the "
                "serving authority this window would publish under is "
                f"unknown: {error}"
            ) from error

    if fit_id is not None:
        assert registry is not None
        manifest = registry.load_manifest(fit_id)
        role_state_hash = manifest.get("role_state_hash")
        candidate = registry.fit_dir(fit_id) / "artifacts"

    authority = (
        AUTHORITY_PROMOTED_FIT
        if fit_id is not None
        else AUTHORITY_FROZEN_BUNDLE
    )

    protocol = check_t20_protocol_compatible(
        slate_date=slate_date,
        snapshot_dir=snapshot_dir,
    )

    readiness = (
        check_gate3_role_readiness(
            project_root=Path(project_root),
            candidate=candidate,
            role_state_hash=role_state_hash,
            slate_date=slate_date,
            snapshot_dir=snapshot_dir,
        )
        if candidate is not None
        else None
    )

    results = {"t20_protocol_compatible": protocol}

    if readiness is not None:
        results["gate3_role_readiness"] = readiness

    return {
        "authority": authority,
        "checks": {
            name: result.payload() for name, result in results.items()
        },
        "enforced": sorted(
            name
            for name in results
            if name == "t20_protocol_compatible"
            or authority == AUTHORITY_PROMOTED_FIT
        ),
        "evaluated_at": _utc_now(),
        "fit_id": fit_id,
        "slate_date": str(slate_date),
        "snapshot_dir": str(snapshot_dir),
    }


def assert_window_gates(measurement: dict[str, Any]) -> None:
    """Refuse the window when an enforced check is false or unmeasurable.

    ``NOT_EVALUABLE`` is a refusal here and not a pass. The caller has
    already captured and verified this window, so "there is no capture to
    measure against" is a contradiction of the thing that got us here, and
    treating it as a pass would be the exact substitution this module exists
    to prevent.
    """
    refusals: list[str] = []

    for name in measurement["enforced"]:
        payload = measurement["checks"][name]

        if payload["passed"] is None:
            refusals.append(
                f"{name} could not be measured for "
                f"{measurement['slate_date']} ({payload['evidence']}), and a "
                "window that has just captured cannot also have no capture"
            )
            continue

        if not payload["passed"]:
            refusals.append(
                f"{name} failed for {measurement['slate_date']}: "
                f"{payload['error'] or payload['evidence']}"
            )

    if refusals:
        raise WindowGateRefused(
            "this serving window may not publish: " + "; ".join(refusals)
        )


def record_window_gates(
    registry: FitRegistry,
    fit_id: str,
    measurement: dict[str, Any],
) -> dict[str, Any] | None:
    """Fold the measured window answers into the fit's validation record.

    The record is the registry's own accounting of a fit and it was
    permanently incomplete while nothing could answer these two names. The
    live evaluation above is what gates; this only stops the record claiming
    a fit was never asked. Existing answers are carried through unchanged, so
    recording can add a name and can never remove or flip a fit-time one.
    """
    existing = registry.load_validation(fit_id)

    if existing is None:
        return None

    checks = dict(existing.get("checks") or {})

    for name in SERVING_WINDOW_VALIDATION_CHECKS:
        payload = measurement["checks"].get(name)

        if payload is None or payload["passed"] is None:
            continue

        checks[name] = bool(payload["passed"])

    if checks == dict(existing.get("checks") or {}):
        return existing

    return registry.record_validation(
        fit_id,
        checks,
        actor=existing.get("actor"),
        notes=(
            f"serving-window checks measured at the {measurement['slate_date']}"
            " window against that window's own capture"
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    registry: FitRegistry | None = None

    if args.registry_root is not None:
        registry = FitRegistry(
            root=resolve_registry_root(
                args.registry_root, project_root=PROJECT_ROOT
            ),
            project_root=PROJECT_ROOT,
        )

    measurement = evaluate_window_gates(
        project_root=PROJECT_ROOT,
        slate_date=args.slate_date,
        data_root=Path(args.data_root),
        registry=registry,
    )

    try:
        assert_window_gates(measurement)

    except WindowGateRefused as error:
        measurement["outcome"] = OUTCOME_REFUSED
        measurement["reason"] = str(error)
        return measurement

    if registry is not None and measurement["fit_id"] is not None:
        record_window_gates(
            registry, measurement["fit_id"], measurement
        )

        # The record is now complete for this fit, so the fail-closed
        # whole-set assertion is available and is taken. It cannot pass on a
        # record the live evaluation above did not just satisfy.
        registry.assert_serving_window_validation_pass(
            measurement["fit_id"]
        )

    measurement["outcome"] = OUTCOME_PASS
    measurement["reason"] = (
        "every enforced serving-window check was recomputed against the "
        f"{measurement['slate_date']} capture and passed"
    )

    return measurement


def render(receipt: dict[str, Any]) -> str:
    lines = [
        "## Serving-window gates",
        "",
        f"- Outcome: `{receipt['outcome']}`",
        f"- Slate date: `{receipt['slate_date']}`",
        f"- Serving authority: `{receipt['authority']}`",
        f"- Fit: `{receipt['fit_id'] or 'none'}`",
        f"- Enforced: `{', '.join(receipt['enforced']) or 'none'}`",
        "",
        "| Check | Enforced | Result | Evidence |",
        "| --- | --- | --- | --- |",
    ]

    for name in sorted(receipt["checks"]):
        payload = receipt["checks"][name]

        result = (
            "NOT_EVALUABLE"
            if payload["passed"] is None
            else ("PASS" if payload["passed"] else "FAIL")
        )

        lines.append(
            f"| `{name}` "
            f"| {'yes' if name in receipt['enforced'] else 'no'} "
            f"| {result} "
            f"| {payload['error'] or payload['evidence']} |"
        )

    lines += ["", receipt["reason"], ""]

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the two serving-window validation checks against "
            "this slate's capture and refuse the window if they do not pass."
        )
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--registry-root", default=None)
    parser.add_argument("--status-path", default=None)
    parser.add_argument("--summary-path", default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except (RegistryError, WindowGateRefused) as error:
        receipt = {
            "authority": None,
            "checks": {},
            "enforced": [],
            "evaluated_at": _utc_now(),
            "fit_id": None,
            "outcome": OUTCOME_REFUSED,
            "reason": str(error),
            "slate_date": args.slate_date,
            "snapshot_dir": None,
        }

    if args.status_path is not None:
        path = Path(args.status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    text = render(receipt)

    print(text)

    if args.summary_path is not None:
        with Path(args.summary_path).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    return EXIT_OK if receipt["outcome"] == OUTCOME_PASS else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
