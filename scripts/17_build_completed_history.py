from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    # This command performs the completed-history model build for a 2026-27 preseason.
    # It intentionally does not start live collectors, because those are long-running
    # operational processes and should be scheduled separately.
    run("scripts/00_check_api.py")
    run(
        "scripts/01_ingest_history.py",
        "--start-season", "2001",
        "--end-season", "2025",
        "--include-advanced",
    )
    run("scripts/02_build_base.py")
    run("scripts/03_tune_dynamic_priors.py")
    run("scripts/04_build_features.py")
    run("scripts/05_train_minutes.py")
    run("scripts/06_train_targets.py")
    run("scripts/07_fit_marginals.py")
    run("scripts/08_fit_copula.py")
    run("scripts/09_backtest.py")


if __name__ == "__main__":
    main()
