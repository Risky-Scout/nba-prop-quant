#!/usr/bin/env python3
"""Run one day's adaptive fit under the frozen architecture.

Modes:

    --dry-run             verify environment, contracts and data state, build
                          the plan, fit nothing
    --benchmark-only      run the real production fit path in isolated
                          staging and record a benchmark; register nothing
    --register-candidate  run the same real fit path and register the
                          candidate immutably

No mode promotes. Registration leaves the candidate
REGISTERED / NOT PROMOTED; live validation and promotion are Step 3D.

This tool never fetches data. It consumes a rolling state the Step 3B refresh
has already committed, and verifies that state before spending time fitting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nba_prop_quant.adaptive_fit_registry import (  # noqa: E402
    FitRegistry,
    RegistryError,
    resolve_registry_root,
)
from nba_prop_quant.adaptive_training import (  # noqa: E402
    MODE_BENCHMARK_ONLY,
    MODE_DRY_RUN,
    MODE_REGISTER_CANDIDATE,
    OUTCOME_ALREADY_RUNNING,
    OUTCOME_NO_NEW_TRAINING_DATA,
    AdaptiveTrainingError,
    CandidateIncomplete,
    DataStateError,
    FrozenPolicyViolation,
    ProductionFitEngine,
    SelectorInvocationRefused,
    TrainingLocked,
    run_daily_fit,
)
from nba_prop_quant.slate import SlateDateError  # noqa: E402


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_FROZEN_POLICY = 3
EXIT_DATA_STATE = 4
EXIT_SELECTOR_REFUSED = 5
EXIT_CANDIDATE_INCOMPLETE = 6
EXIT_ALREADY_RUNNING = 7
EXIT_REGISTRY = 8

_EXIT_BY_ERROR = (
    (FrozenPolicyViolation, EXIT_FROZEN_POLICY),
    (SelectorInvocationRefused, EXIT_SELECTOR_REFUSED),
    (DataStateError, EXIT_DATA_STATE),
    (CandidateIncomplete, EXIT_CANDIDATE_INCOMPLETE),
    (TrainingLocked, EXIT_ALREADY_RUNNING),
    (RegistryError, EXIT_REGISTRY),
    (SlateDateError, EXIT_DATA_STATE),
    (AdaptiveTrainingError, EXIT_ERROR),
)


def exit_code_for(error: Exception) -> int:
    for kind, code in _EXIT_BY_ERROR:
        if isinstance(error, kind):
            return code

    return EXIT_ERROR


def emit(payload: dict, stream=None) -> None:
    print(
        json.dumps(payload, sort_keys=True, indent=2, default=str),
        file=stream if stream is not None else sys.stdout,
    )


def head_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.stdout.strip() if result.returncode == 0 else ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_adaptive_daily_fit",
        description=(
            "Daily adaptive fit under the frozen architecture. Re-estimates "
            "numerical parameters only; never selects a model, never promotes."
        ),
    )

    parser.add_argument(
        "--slate-date",
        required=True,
        help="Slate date YYYY-MM-DD. All training data must precede it.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Rolling data root a Step 3B refresh has already committed.",
    )

    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help=(
            "Isolated operational root for staging, locks and reports. Must "
            "be outside the repository."
        ),
    )

    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help=(
            "Step 3A fit registry root. Required for --register-candidate; "
            "optional for --benchmark-only, where a disposable temporary "
            "registry is appropriate."
        ),
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Repository root holding the frozen contracts.",
    )

    parser.add_argument(
        "--training-cutoff",
        default=None,
        help=(
            "Optional assertion about the expected cutoff. It can only "
            "narrow: a value beyond the verified data is refused."
        ),
    )

    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        default=True,
        help="Keep the staging workspace for inspection (default).",
    )

    parser.add_argument(
        "--discard-workspace",
        dest="keep_workspace",
        action="store_false",
        help="Remove the staging workspace when the run finishes.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--dry-run",
        dest="mode",
        action="store_const",
        const=MODE_DRY_RUN,
        help="Verify and plan only. Fits nothing.",
    )

    mode.add_argument(
        "--benchmark-only",
        dest="mode",
        action="store_const",
        const=MODE_BENCHMARK_ONLY,
        help="Run the real fit path and record a benchmark. Registers nothing.",
    )

    mode.add_argument(
        "--register-candidate",
        dest="mode",
        action="store_const",
        const=MODE_REGISTER_CANDIDATE,
        help="Run the real fit path and register the candidate. Never promotes.",
    )

    return parser.parse_args(argv)


def build_registry(args: argparse.Namespace) -> FitRegistry | None:
    if args.registry_root is None:
        if args.mode == MODE_REGISTER_CANDIDATE:
            raise AdaptiveTrainingError(
                "--register-candidate requires --registry-root"
            )

        return None

    project_root = Path(args.project_root).resolve()

    root = resolve_registry_root(
        args.registry_root, project_root=project_root
    )

    return FitRegistry(root=root, project_root=project_root)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    project_root = Path(args.project_root).resolve()

    try:
        registry = build_registry(args)

        engine = (
            None
            if args.mode == MODE_DRY_RUN
            else ProductionFitEngine(project_root)
        )

        result = run_daily_fit(
            project_root=project_root,
            data_root=Path(args.data_root).resolve(),
            work_root=Path(args.work_root).resolve(),
            slate_date=args.slate_date,
            registry=registry,
            engine=engine,
            mode=args.mode,
            source_commit_sha=head_commit(project_root),
            requested_cutoff=args.training_cutoff,
            keep_workspace=args.keep_workspace,
        )

    except (AdaptiveTrainingError, RegistryError, SlateDateError) as error:
        emit(
            {
                "error": type(error).__name__,
                "message": str(error),
                "mode": args.mode,
                "status": "failed",
            },
            stream=sys.stderr,
        )

        return exit_code_for(error)

    except (OSError, ValueError, KeyError) as error:
        emit(
            {
                "error": type(error).__name__,
                "message": str(error),
                "mode": args.mode,
                "status": "failed",
            },
            stream=sys.stderr,
        )

        return EXIT_ERROR

    emit(result)

    if result["outcome"] == OUTCOME_ALREADY_RUNNING:
        return EXIT_ALREADY_RUNNING

    # A day with no newly completed NBA data is a correct, successful outcome.
    if result["outcome"] == OUTCOME_NO_NEW_TRAINING_DATA:
        return EXIT_OK

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
