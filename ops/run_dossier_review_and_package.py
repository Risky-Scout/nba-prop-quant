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
    print("DOSSIER REVIEW → EXACT REVIEWER HANDOFF")
    print("=" * 118)

    run(
        "ops/audit_senior_quant_dossier_clarity.py"
    )

    run(
        "ops/build_senior_quant_reviewer_handoff.py"
    )

    print()
    print(
        "PASS: dossier clarity review completed before package construction."
    )


if __name__ == "__main__":
    main()
