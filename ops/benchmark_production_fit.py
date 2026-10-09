#!/usr/bin/env python3
"""Benchmark the real production fit over the full corpus, mutating nothing.

The adaptive fit's runtime over the whole 2001-to-present history was never
measured. Every timing on record comes from a stub engine in the test suite,
which says nothing about whether a real fit completes inside the lifecycle's
own timeout on the machine that has to run it. This measures it.

It is not authorization to change production model state, and the design is
what makes that true rather than a promise. It wraps
``ops/run_adaptive_daily_fit.py --benchmark-only``, which runs the identical
``ProductionFitEngine`` the registering path runs -- the same DAG, the same
frozen policy guards, the same computed validation -- and stops before
registration. No registry root is passed, so there is no registry for it to
write to. The staging workspace lives under an explicit scratch root.

And it is checked rather than assumed. The things that must not move are
fingerprinted before the fit and re-fingerprinted after it: the serving model
tree, the fit registry if one is configured, and the three frozen
identifiers. A difference is a failure of the benchmark, reported as such,
even when the fit itself succeeded.

WHAT THE RECEIPT RECORDS
------------------------

Per stage: elapsed seconds, from the recorder the real fit path already
carries. Per run: the input state fingerprint, the fitting-information
digest, start and end, total runtime, peak RSS, artifact completeness, the
full computed validation report, the calibration stage's own result including
any fallback origins, and the prediction smoke test's result.

The receipt is small and is the only thing worth committing. The staging
workspace holds fitted model binaries and is not: it stays under the scratch
root and is named in the receipt by path and size, not carried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FIT_ENTRY_POINT = Path("ops") / "run_adaptive_daily_fit.py"

#: The identifiers the deployment is pinned to. Fingerprinted around the run
#: because a benchmark that moved one of them would not be a benchmark.
FROZEN_IDENTIFIER_PATHS: tuple[str, ...] = (
    "research/final_model/final_model_spec.json",
    "research/final_model/live_shadow_promotion_policy.json",
    "research/final_upstream_remediation/factor_spec.json",
)

#: The stages a benchmark run must have completed. ``registration`` is absent
#: on purpose: benchmark-only stops before it, and a benchmark that reached
#: registration would mean the isolation failed.
REQUIRED_STAGES: tuple[str, ...] = (
    "input_snapshot",
    "feature_build",
    "minutes_fit",
    "target_fit",
    "ensemble_weight_fit",
    "marginal_fit",
    "dependence_fit",
    "calibration_fit",
    "gate3_fit",
    "candidate_assembly",
    "validation",
)

FORBIDDEN_STAGE = "registration"

#: The computed check that answers "could this candidate actually price".
SMOKE_TEST_CHECK = "prediction_smoke_test"

CALIBRATION_STAGE = "calibration_fit"

DEFAULT_RECEIPT = Path("ops") / "evidence" / "production_fit_benchmark.json"

EXIT_OK = 0
EXIT_FAILED = 1


class BenchmarkRefused(RuntimeError):
    """The benchmark could not be run in a state that proves anything."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _digest_tree(root: Path) -> str | None:
    """A single digest over every regular file under ``root``.

    ``None`` when the tree is absent, which is a meaningful answer: a
    benchmark host need not have a fit registry at all.
    """
    path = Path(root)

    if not path.exists():
        return None

    accumulator = hashlib.sha256()

    for entry in sorted(path.rglob("*")):
        if not entry.is_file() or entry.is_symlink():
            continue

        accumulator.update(entry.relative_to(path).as_posix().encode("utf-8"))

        accumulator.update(hashlib.sha256(entry.read_bytes()).digest())

    return accumulator.hexdigest()


def _digest_file(path: Path) -> str | None:
    target = Path(path)

    if not target.is_file():
        return None

    return hashlib.sha256(target.read_bytes()).hexdigest()


def production_state_fingerprint(
    *,
    project_root: Path,
    model_dir: Path,
    registry_root: Path | None,
) -> dict[str, str | None]:
    """Everything this run is forbidden to change."""
    fingerprint: dict[str, str | None] = {
        "model_tree": _digest_tree(model_dir),
        "registry_tree": (
            _digest_tree(registry_root) if registry_root is not None else None
        ),
    }

    for relative in FROZEN_IDENTIFIER_PATHS:
        fingerprint[relative] = _digest_file(Path(project_root) / relative)

    return fingerprint


def mutations(
    before: dict[str, str | None], after: dict[str, str | None]
) -> list[str]:
    return sorted(
        name for name in before if before[name] != after.get(name)
    )


# ----------------------------------------------------------------------
# running the real fit path
# ----------------------------------------------------------------------


def fit_command(
    *,
    project_root: Path,
    data_root: Path,
    work_root: Path,
    slate_date: str,
    training_cutoff: str | None,
) -> list[str]:
    """The exact invocation. No registry root, so nothing can be registered."""
    command = [
        sys.executable,
        str(Path(project_root) / FIT_ENTRY_POINT),
        "--benchmark-only",
        "--slate-date",
        slate_date,
        "--data-root",
        str(data_root),
        "--work-root",
        str(work_root),
        "--project-root",
        str(project_root),
        "--keep-workspace",
    ]

    if training_cutoff is not None:
        command += ["--training-cutoff", training_cutoff]

    return command


def run_fit(command: list[str], *, project_root: Path) -> dict[str, Any]:
    """Run the fit and return its JSON result.

    The entry point emits JSON on stdout and diagnostics on stderr, so a
    nonzero exit still carries a usable reason.
    """
    completed = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise BenchmarkRefused(
            f"the fit exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout).strip()[-3000:]}"
        )

    try:
        return json.loads(completed.stdout)

    except json.JSONDecodeError as error:
        raise BenchmarkRefused(
            f"the fit exited 0 but did not emit a JSON result: {error}"
        ) from error


# ----------------------------------------------------------------------
# the receipt
# ----------------------------------------------------------------------


def stage_report(benchmark: dict[str, Any]) -> dict[str, Any]:
    seconds = dict(benchmark.get("stage_seconds") or {})

    absent = [stage for stage in REQUIRED_STAGES if stage not in seconds]

    return {
        "completed": sorted(seconds),
        "elapsed_seconds": dict(sorted(seconds.items())),
        "every_required_stage_completed": not absent,
        "missing_stages": absent,
        "registration_was_not_reached": FORBIDDEN_STAGE not in seconds,
    }


def validation_report(result: dict[str, Any]) -> dict[str, Any]:
    report = dict(result.get("validation_report") or {})

    checks = dict(result.get("validation_checks") or {})

    failed = sorted(name for name, passed in checks.items() if not passed)

    smoke = next(
        (
            entry
            for entry in report.get("results", [])
            if entry.get("name") == SMOKE_TEST_CHECK
        ),
        None,
    )

    return {
        "checks": dict(sorted(checks.items())),
        "deferred": sorted(result.get("deferred_validation_checks") or []),
        "every_computed_check_passed": not failed,
        "failed_checks": failed,
        "prediction_smoke_test": smoke,
        "report": report,
    }


def calibration_report(
    result: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    fallbacks = dict(result.get("calibration_fallbacks") or {})

    return {
        "elapsed_seconds": (benchmark.get("stage_seconds") or {}).get(
            CALIBRATION_STAGE
        ),
        "fallback_origins": dict(sorted(fallbacks.items())),
        "every_route_fitted": not fallbacks,
        "stage_completed": CALIBRATION_STAGE
        in (benchmark.get("stage_seconds") or {}),
    }


def workspace_report(result: dict[str, Any]) -> dict[str, Any]:
    workspace = result.get("workspace")

    benchmark = dict(result.get("benchmark") or {})

    return {
        "candidate_artifact_bytes": benchmark.get("candidate_artifact_bytes"),
        "candidate_artifact_count": (benchmark.get("details") or {}).get(
            "candidate_artifact_count"
        ),
        "path": workspace,
    }


def build_receipt(
    *,
    result: dict[str, Any],
    command: list[str],
    before: dict[str, str | None],
    after: dict[str, str | None],
    slate_date: str,
    production_sha: str | None,
    input_state_fingerprint: str | None,
) -> dict[str, Any]:
    benchmark = dict(result.get("benchmark") or {})

    changed = mutations(before, after)

    stages = stage_report(benchmark)

    validation = validation_report(result)

    calibration = calibration_report(result, benchmark)

    passed = (
        result.get("outcome") == "COMPLETED"
        and stages["every_required_stage_completed"]
        and stages["registration_was_not_reached"]
        and validation["every_computed_check_passed"]
        and not changed
    )

    return {
        "calibration": calibration,
        "command": command,
        "fitting_information_digest": result.get(
            "training_data_manifest_sha256"
        ),
        "generated_at": _utc_now(),
        "input_state_fingerprint": input_state_fingerprint,
        "mode": "benchmark-only",
        "outcome": result.get("outcome"),
        "passed": passed,
        "peak_rss_bytes": benchmark.get("peak_rss_bytes"),
        "production_code_sha": production_sha,
        "production_state": {
            "fingerprint_after": after,
            "fingerprint_before": before,
            "mutated": changed,
            "mutation_occurred": bool(changed),
        },
        "promotion_performed": False,
        "publishing_performed": False,
        "runtime": {
            "ended_at_utc": benchmark.get("ended_at_utc"),
            "started_at_utc": benchmark.get("started_at_utc"),
            "total_seconds": benchmark.get("total_seconds"),
        },
        "schema_version": 1,
        "slate_date": slate_date,
        "stages": stages,
        "validation": validation,
        "workspace": workspace_report(result),
    }


def render(receipt: dict[str, Any]) -> str:
    runtime = receipt["runtime"]

    rows = [
        ("outcome", str(receipt["outcome"])),
        ("benchmark", "PASS" if receipt["passed"] else "FAIL"),
        ("total seconds", str(runtime["total_seconds"])),
        ("peak RSS bytes", str(receipt["peak_rss_bytes"])),
        (
            "every required stage",
            "yes" if receipt["stages"]["every_required_stage_completed"] else "no",
        ),
        (
            "registration reached",
            "no" if receipt["stages"]["registration_was_not_reached"] else "YES",
        ),
        (
            "computed validation",
            "all passed"
            if receipt["validation"]["every_computed_check_passed"]
            else ", ".join(receipt["validation"]["failed_checks"]),
        ),
        (
            "deferred checks",
            ", ".join(receipt["validation"]["deferred"]) or "none",
        ),
        (
            "calibration fallbacks",
            ", ".join(receipt["calibration"]["fallback_origins"]) or "none",
        ),
        (
            "production state mutated",
            "YES" if receipt["production_state"]["mutation_occurred"] else "no",
        ),
    ]

    lines = [
        "## Full-data production fit benchmark",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    lines += [f"| {label} | {value} |" for label, value in rows]

    lines += ["", "### Stage elapsed seconds", "", "| stage | seconds |", "| --- | --- |"]

    lines += [
        f"| {stage} | {seconds} |"
        for stage, seconds in receipt["stages"]["elapsed_seconds"].items()
    ]

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark_production_fit",
        description=(
            "Benchmark the real production fit over the full corpus. "
            "Registers nothing, promotes nothing, publishes nothing."
        ),
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="the full historical corpus the production refresh maintains",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="isolated scratch root for staging. Must be outside the "
        "repository and must not be the production work root.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="the serving artifact tree, fingerprinted before and after",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help="the production fit registry. Fingerprinted before and after; "
        "never passed to the fit, which therefore has nothing to register to.",
    )
    parser.add_argument("--training-cutoff", default=None)
    parser.add_argument("--production-sha", default=None)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--receipt-path", type=Path, default=ROOT / DEFAULT_RECEIPT
    )
    parser.add_argument("--summary-path", type=Path, default=None)

    return parser.parse_args(argv)


def _input_state_fingerprint(data_root: Path) -> str | None:
    path = Path(data_root) / ".state" / "current_season_state.json"

    if not path.is_file():
        return None

    record = json.loads(path.read_text(encoding="utf-8"))

    value = record.get("datasets_fingerprint")

    return str(value) if value else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()

    work_root = Path(args.work_root).resolve()

    if work_root == project_root or project_root in work_root.parents:
        raise BenchmarkRefused(
            f"the scratch root {work_root} is inside the repository at "
            f"{project_root}; a benchmark must not stage inside version "
            "control"
        )

    model_dir = (
        Path(args.model_dir).resolve()
        if args.model_dir is not None
        else project_root / "models"
    )

    registry_root = (
        Path(args.registry_root).resolve()
        if args.registry_root is not None
        else None
    )

    before = production_state_fingerprint(
        project_root=project_root,
        model_dir=model_dir,
        registry_root=registry_root,
    )

    command = fit_command(
        project_root=project_root,
        data_root=Path(args.data_root).resolve(),
        work_root=work_root,
        slate_date=args.slate_date,
        training_cutoff=args.training_cutoff,
    )

    result = run_fit(command, project_root=project_root)

    after = production_state_fingerprint(
        project_root=project_root,
        model_dir=model_dir,
        registry_root=registry_root,
    )

    return build_receipt(
        result=result,
        command=command,
        before=before,
        after=after,
        slate_date=args.slate_date,
        production_sha=args.production_sha,
        input_state_fingerprint=_input_state_fingerprint(args.data_root),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except BenchmarkRefused as error:
        failure = {
            "error": type(error).__name__,
            "generated_at": _utc_now(),
            "message": str(error),
            "mode": "benchmark-only",
            "passed": False,
            "slate_date": args.slate_date,
        }

        args.receipt_path.parent.mkdir(parents=True, exist_ok=True)

        args.receipt_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_FAILED

    args.receipt_path.parent.mkdir(parents=True, exist_ok=True)

    args.receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(render(receipt))

    print(render(receipt), end="")

    return EXIT_OK if receipt["passed"] else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
