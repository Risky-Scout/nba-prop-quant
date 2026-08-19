from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path.cwd()


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print()
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    if not (
        ROOT / "src/nba_prop_quant"
    ).exists():
        raise SystemExit(
            "ERROR: run from nba_prop_quant_blueprint project root."
        )

    print("=" * 118)
    print("SENIOR-QUANT REVIEW PREFLIGHT")
    print("=" * 118)
    print(
        "Sequence: freeze verification -> production contract -> "
        "synthetic grading integration -> real 2025 replay -> "
        "pricing/grading audit -> dossier -> freeze verification."
    )

    run("scripts/verify_frozen_manifest.py")
    run("scripts/10a_validate_production_contract.py")
    run("ops/qa_grading_synthetic_integration.py")
    run("ops/qa_replay_2025_grading.py")
    run("ops/audit_pricing_grading_contract.py")
    run("ops/build_senior_quant_review_dossier.py")
    run("scripts/verify_frozen_manifest.py")

    print()
    print("=" * 118)
    print("PREFLIGHT COMPLETE")
    print("=" * 118)
    print(
        "PASS: synthetic grader QA, real 2025 development replay, "
        "contract audit, and senior-quant dossier completed."
    )
    print(
        "The real 2026-27 external-test performance ledger was not used "
        "for either QA replay."
    )
    print(
        "Review dossier: review/SENIOR_QUANT_REVIEW_DOSSIER.md"
    )


if __name__ == "__main__":
    main()
