#!/usr/bin/env python3
"""Step 3D: promote the morning's validated daily fit, or keep the last good one.

WHAT STEP 3D IS
---------------

The frozen update protocol is explicit about the division:

    "promotion": {"automatic_promotion": false,
                  "note": "Step 3C registers an immutable candidate and
                           stops. Live validation and promotion are Step 3D."}

So the thing the protocol prohibits is Step 3C promoting what it just fitted:
the stage that produces a candidate may not also approve it. Step 3D is named
there as the stage that does approve it, and it is a separate stage reading
separate evidence. This is that stage. It fits nothing, reads no training
data, and cannot reach the fitting code: it re-verifies a candidate somebody
else produced and either promotes it or leaves the incumbent alone.

WHAT IT REQUIRES
----------------

Every fit-time validation check, through the registry's own
``assert_validation_pass``, under the registry's own promotion lock, with the
registry's own atomic single state replacement. Nothing about those is
reimplemented here, because a second implementation of "may this fit be
promoted" could disagree with the first and the one that decides is the
registry's.

What it does not require is the two serving-window checks. They ask about a
capture taken in the evening, about a fit produced that morning, so requiring
them here made promotion unreachable -- see
``adaptive_fit_registry.SERVING_WINDOW_VALIDATION_CHECKS`` for why that is not
a weakening and ``ops/assert_serving_window_gates.py`` for where they are now
enforced, per window, against the capture they are about.

THE FALLBACK IS A PREVIOUS GOOD FIT, NOT A DEFAULT
--------------------------------------------------

A fit that fails validation is not promoted and nothing is displaced: the
registry's single atomic state replacement is the last thing that happens, so
every refusal before it leaves ``current_good_fit_id`` exactly as it was.
This step reports that as ``FALLBACK_ACTIVE`` with the reason, and it reports
honestly which kind of fallback it is -- yesterday's promoted daily fit is a
different situation from never having promoted anything, and an operator's
response to the two differs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.adaptive_fit_registry import (  # noqa: E402
    FIT_TIME_VALIDATION_CHECKS,
    SERVING_WINDOW_VALIDATION_CHECKS,
    FitRegistry,
    PromotionRefused,
    RegistryError,
    ValidationRefused,
    resolve_registry_root,
    sha256_canonical,
)

OUTCOME_PROMOTED = "PROMOTED"
OUTCOME_ALREADY_CURRENT = "ALREADY_CURRENT"
OUTCOME_NO_FIT = "NO_DAILY_FIT"
OUTCOME_REFUSED = "PROMOTION_REFUSED"

FALLBACK_NONE = "NONE"
FALLBACK_VALIDATION_FAILED = "DAILY_FIT_VALIDATION_FAILED"
FALLBACK_NO_FIT = "NO_DAILY_FIT_REGISTERED"

ACTOR = "nba-production-lifecycle-step3d"

#: ``nba_prop_quant_fit_<YYYYMMDD>_<16 hex>``. Used only to pick the newest
#: fit for a date when no status document names one; never to construct an id.
FIT_ID_PATTERN = re.compile(
    r"^nba_prop_quant_fit_(?P<date>\d{8})_(?P<digest>[0-9a-f]{16})$"
)

EXIT_OK = 0
EXIT_FAILED = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_fit(
    registry: FitRegistry,
    *,
    adaptive_status: Path | None,
    slate_date: str,
    explicit: str | None,
) -> tuple[str | None, str]:
    """Which fit this run is asked about, and how it was identified.

    The adaptive protocol's own status document is preferred, because it is
    the record of what this morning's run actually registered. The registry
    is the fallback for a promotion run outside the lifecycle, and it is
    narrowed to the slate date so that a day whose fit failed cannot promote
    a fit from an earlier day by looking like the newest one.
    """
    if explicit is not None:
        return explicit, "explicit --fit-id"

    if adaptive_status is not None and Path(adaptive_status).is_file():
        status = _load_json(Path(adaptive_status))

        if isinstance(status, dict) and status.get("fit_id"):
            return str(status["fit_id"]), "adaptive protocol status document"

    stamp = str(slate_date).replace("-", "")

    same_day = sorted(
        fit_id
        for fit_id in registry.list_fits()
        if (match := FIT_ID_PATTERN.match(fit_id))
        and match.group("date") == stamp
    )

    if same_day:
        return same_day[-1], f"newest fit registered on {slate_date}"

    return None, f"no fit is registered for {slate_date}"


def gate_status(
    registry: FitRegistry, fit_id: str
) -> dict[str, Any]:
    """Every canonical gate, partitioned, with the answer on record."""
    record = registry.load_validation(fit_id) or {}

    checks = record.get("checks") or {}

    def view(names: tuple[str, ...]) -> dict[str, Any]:
        return {
            "failed": sorted(
                name
                for name in names
                if name in checks and not checks[name]
            ),
            "missing": sorted(
                name for name in names if name not in checks
            ),
            "names": list(names),
            "status": {
                name: (
                    "NOT_RECORDED"
                    if name not in checks
                    else ("PASS" if checks[name] else "FAIL")
                )
                for name in names
            },
        }

    fit_time = view(FIT_TIME_VALIDATION_CHECKS)
    window = view(SERVING_WINDOW_VALIDATION_CHECKS)

    return {
        "canonical_gate_count": (
            len(FIT_TIME_VALIDATION_CHECKS)
            + len(SERVING_WINDOW_VALIDATION_CHECKS)
        ),
        "fit_time_gates": fit_time,
        "fit_time_gates_pass": (
            not fit_time["failed"] and not fit_time["missing"]
        ),
        "serving_window_gates": window,
    }


def fallback_view(
    registry: FitRegistry,
    *,
    candidate: str | None,
    reason: str,
) -> dict[str, Any]:
    """What is serving, and whether calling it today's fit would be false."""
    state = registry.current()

    current = state.get("current_good_fit_id")

    active = current != candidate or candidate is None

    return {
        "fallback_active": bool(active),
        "fallback_fit_id": current if active else None,
        "fallback_reason": reason if active else FALLBACK_NONE,
        "promotion_state": state,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    registry = FitRegistry(
        root=resolve_registry_root(
            args.registry_root, project_root=PROJECT_ROOT
        ),
        project_root=PROJECT_ROOT,
    )

    fit_id, how = candidate_fit(
        registry,
        adaptive_status=(
            Path(args.adaptive_status) if args.adaptive_status else None
        ),
        slate_date=args.slate_date,
        explicit=args.fit_id,
    )

    base = {
        "actor": ACTOR,
        "candidate_fit_id": fit_id,
        "evaluated_at": _utc_now(),
        "identified_by": how,
        "slate_date": args.slate_date,
    }

    if fit_id is None:
        return {
            **base,
            **fallback_view(
                registry, candidate=None, reason=FALLBACK_NO_FIT
            ),
            "outcome": OUTCOME_NO_FIT,
            "reason": how,
            "training_cutoff": None,
        }

    manifest = registry.load_manifest(fit_id)

    base["training_cutoff"] = (
        None
        if manifest.get("training_cutoff") is None
        else str(manifest["training_cutoff"])
    )

    # One digest over the whole recorded artifact set, by the registry's own
    # canonical serialisation, so two fits with the same parameters have the
    # same digest and a single changed byte anywhere changes it.
    base["fit_artifact_digest"] = sha256_canonical(
        manifest.get("artifact_hashes") or {}
    )

    gates = gate_status(registry, fit_id)

    base["gates"] = gates

    if registry.current().get("current_good_fit_id") == fit_id:
        return {
            **base,
            **fallback_view(registry, candidate=fit_id, reason=FALLBACK_NONE),
            "outcome": OUTCOME_ALREADY_CURRENT,
            "reason": (
                f"fit {fit_id} is already the current good fit, so promotion "
                "has nothing to do and the incumbent is unmoved"
            ),
        }

    try:
        state = registry.promote(
            fit_id,
            reason=(
                f"daily adaptive fit for the {args.slate_date} slate, "
                f"trained through {base['training_cutoff']}"
            ),
            actor=ACTOR,
        )

    except (PromotionRefused, ValidationRefused, RegistryError) as error:
        return {
            **base,
            **fallback_view(
                registry,
                candidate=fit_id,
                reason=FALLBACK_VALIDATION_FAILED,
            ),
            "outcome": OUTCOME_REFUSED,
            "reason": f"{type(error).__name__}: {error}",
        }

    return {
        **base,
        "fallback_active": False,
        "fallback_fit_id": None,
        "fallback_reason": FALLBACK_NONE,
        "outcome": OUTCOME_PROMOTED,
        "previous_good_fit_id": state.get("previous_good_fit_id"),
        "promoted_at": state.get("promoted_at"),
        "promotion_state": registry.current(),
        "reason": (
            f"every one of the {len(FIT_TIME_VALIDATION_CHECKS)} fit-time "
            f"gates passed for {fit_id}, so it is the current good fit"
        ),
    }


def render(receipt: dict[str, Any]) -> str:
    lines = [
        "## Daily fit promotion (Step 3D)",
        "",
        f"- Outcome: `{receipt['outcome']}`",
        f"- Slate date: `{receipt['slate_date']}`",
        f"- Candidate fit: `{receipt.get('candidate_fit_id') or 'none'}`"
        f" ({receipt['identified_by']})",
        f"- Training cutoff: `{receipt.get('training_cutoff') or 'none'}`",
        f"- Fallback active: "
        f"`{'YES' if receipt.get('fallback_active') else 'NO'}`",
        f"- Fallback reason: `{receipt.get('fallback_reason')}`",
    ]

    gates = receipt.get("gates")

    if gates:
        lines += [
            "",
            f"Canonical promotion gates: {gates['canonical_gate_count']} "
            f"({len(gates['fit_time_gates']['names'])} fit-time, "
            f"{len(gates['serving_window_gates']['names'])} serving-window).",
            "",
            "| Gate | Stage | Status |",
            "| --- | --- | --- |",
        ]

        for stage, key in (
            ("fit-time", "fit_time_gates"),
            ("serving-window", "serving_window_gates"),
        ):
            for name in gates[key]["names"]:
                lines.append(
                    f"| `{name}` | {stage} | "
                    f"{gates[key]['status'][name]} |"
                )

    lines += ["", receipt["reason"], ""]

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the morning's validated daily fit when every fit-time "
            "gate passes, and otherwise leave the incumbent in place."
        )
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--registry-root", required=True)
    parser.add_argument("--adaptive-status", default=None)
    parser.add_argument("--fit-id", default=None)
    parser.add_argument("--receipt-path", default=None)
    parser.add_argument("--github-output", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit nonzero when the daily fit was not promoted. Off by "
            "default: a slate served by a slightly older validated "
            "authority is better than no slate, so an unpromotable fit "
            "degrades rather than stopping the lifecycle."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except RegistryError as error:
        receipt = {
            "actor": ACTOR,
            "candidate_fit_id": args.fit_id,
            "evaluated_at": _utc_now(),
            "fallback_active": True,
            "fallback_reason": FALLBACK_VALIDATION_FAILED,
            "identified_by": "registry unreadable",
            "outcome": OUTCOME_REFUSED,
            "reason": f"{type(error).__name__}: {error}",
            "slate_date": args.slate_date,
            "training_cutoff": None,
        }

    if args.receipt_path is not None:
        path = Path(args.receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.github_output is not None:
        promoted = receipt["outcome"] in (
            OUTCOME_PROMOTED,
            OUTCOME_ALREADY_CURRENT,
        )

        with Path(args.github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"outcome={receipt['outcome']}\n")
            handle.write(
                f"promoted={'true' if promoted else 'false'}\n"
            )
            handle.write(
                f"fit_id={receipt.get('candidate_fit_id') or ''}\n"
            )
            handle.write(
                "fallback_active="
                f"{'true' if receipt.get('fallback_active') else 'false'}\n"
            )

    text = render(receipt)

    print(text)

    if args.summary_path is not None:
        with Path(args.summary_path).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    if receipt["outcome"] in (OUTCOME_PROMOTED, OUTCOME_ALREADY_CURRENT):
        return EXIT_OK

    return EXIT_FAILED if args.strict else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
