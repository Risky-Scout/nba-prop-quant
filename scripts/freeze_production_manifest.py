from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = MODEL_DIR / "frozen_manifests"

CRITICAL_MODEL_ARTIFACTS = [
    "dynamic_params.json",
    "experience_curves.joblib",
    "ensemble_weights.json",
    "mean_model_selection.json",
    "marginal_selection.json",
    "marginals_pre2025.joblib",
    "marginals.joblib",
    "copula_pre2025.joblib",
    "copula.joblib",
    "combo_dependence_policy.json",
    "market_probability_calibration.json",
    "market_probability_calibration_policy.json",
]

TARGET_MODEL_PATTERNS = [
    "pts.joblib",
    "reb.joblib",
    "ast.joblib",
    "stl.joblib",
    "blk.joblib",
    "fg3m.joblib",
]

CRITICAL_SOURCE_FILES = [
    "src/nba_prop_quant/settings.py",
    "src/nba_prop_quant/game_context.py",
    "src/nba_prop_quant/decay.py",
    "src/nba_prop_quant/kalman.py",
    "src/nba_prop_quant/experience.py",
    "src/nba_prop_quant/features.py",
    "src/nba_prop_quant/model.py",
    "src/nba_prop_quant/distributions.py",
    "src/nba_prop_quant/copula.py",
    "src/nba_prop_quant/pricing.py",
    "src/nba_prop_quant/market.py",
    "src/nba_prop_quant/availability.py",
    "src/nba_prop_quant/interpret.py",
    "src/nba_prop_quant/slate.py",
    "src/nba_prop_quant/pipeline.py",
    "src/nba_prop_quant/live.py",
    "src/nba_prop_quant/production.py",
    "tests/test_production_live.py",
]

CRITICAL_SCRIPT_FILES = [
    "scripts/05_train_minutes.py",
    "scripts/06_train_targets.py",
    "scripts/06b_fit_mean_ensemble.py",
    "scripts/06c_select_mean_models.py",
    "scripts/07_fit_marginals.py",
    "scripts/08_fit_copula.py",
    "scripts/08b_bootstrap_copula_crps.py",
    "scripts/08c_tune_copula_shrinkage.py",
    "scripts/08d_refine_copula_shrinkage_cv.py",
    "scripts/09_backtest.py",
    "scripts/09c_fit_probability_calibration.py",
    "scripts/09d_select_probability_calibration.py",
    "scripts/09e_evaluate_oof_calibrated_betting.py",
    "scripts/10_predict_slate.py",
    "scripts/10a_validate_production_contract.py",
    "scripts/15_price_markets.py",
    "scripts/freeze_production_manifest.py",
    "scripts/verify_frozen_manifest.py",
]

CRITICAL_CONFIG_FILES = [
    "configs/model.yaml",
]

KEY_DATA_AUDIT_FILES = [
    "data/processed/oof_selected_means.parquet",
    "data/processed/selected_means_distribution_split.parquet",
    "data/processed/market_backtest/calibration_walkforward/pooled_event_metrics.csv",
    "data/processed/market_backtest/calibration_walkforward/calibration_selection_audit.csv",
    "data/processed/market_backtest/calibrated_oof/event_scoring_summary.csv",
    "data/processed/market_backtest/calibrated_oof/event_game_cluster_bootstrap.csv",
]

BETTING_MONITORING_GRID = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="pre_live_integration_architecture_baseline",
        help=(
            "Manifest stage label. After live pricing is wired and tested, "
            "freeze again with --stage external_test_deployment."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()

    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def load_json_if_exists(path: Path):
    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def git_info() -> dict:
    try:
        inside = subprocess.run(
            [
                "git",
                "rev-parse",
                "--is-inside-work-tree",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        if inside != "true":
            return {
                "available": False,
            }

        commit = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        branch = subprocess.run(
            [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()

        return {
            "available": True,
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "status_porcelain": status,
        }

    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
        }


def pip_freeze_record(
    freeze_path: Path,
) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    freeze_text = (
        result.stdout.strip()
        + "\n"
    )

    freeze_path.write_text(
        freeze_text,
        encoding="utf-8",
    )

    return file_record(
        freeze_path
    )


def run_pytest() -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    output = (
        result.stdout
        + result.stderr
    ).strip()

    return {
        "returncode": int(
            result.returncode
        ),
        "passed": bool(
            result.returncode
            == 0
        ),
        "output": output,
    }


def find_minutes_model() -> list[Path]:
    candidates = []

    for pattern in [
        "*minute*.joblib",
        "*minutes*.joblib",
    ]:
        candidates.extend(
            MODEL_DIR.glob(
                pattern
            )
        )

    return sorted(
        set(
            candidates
        )
    )


def records_for_paths(
    relative_paths: list[str],
    required: bool = False,
) -> tuple[list[dict], list[str]]:
    records = []
    missing = []

    for relative in relative_paths:
        path = ROOT / relative

        if path.exists():
            records.append(
                file_record(
                    path
                )
            )
        else:
            missing.append(
                relative
            )

    if required and missing:
        raise RuntimeError(
            "Missing required files: "
            + ", ".join(
                missing
            )
        )

    return records, missing


def summarize_policy() -> dict:
    mean_selection = load_json_if_exists(
        MODEL_DIR
        / "mean_model_selection.json"
    )

    marginal_selection = load_json_if_exists(
        MODEL_DIR
        / "marginal_selection.json"
    )

    dependence_policy = load_json_if_exists(
        MODEL_DIR
        / "combo_dependence_policy.json"
    )

    calibration_policy = load_json_if_exists(
        MODEL_DIR
        / "market_probability_calibration_policy.json"
    )

    return {
        "mean_model_selection": (
            mean_selection
        ),
        "marginal_selection": (
            marginal_selection
        ),
        "combo_dependence_policy": (
            dependence_policy
        ),
        "market_probability_calibration_policy": (
            calibration_policy
        ),
    }


def human_markdown(
    manifest: dict,
) -> str:
    lines = []

    lines.append(
        "# Frozen NBA Prop Quant Production Manifest"
    )

    lines.append("")
    lines.append(
        f"- Freeze ID: `{manifest['freeze_id']}`"
    )
    lines.append(
        f"- Created UTC: `{manifest['created_utc']}`"
    )
    lines.append(
        f"- Freeze stage: `{manifest['freeze_stage']}`"
    )
    lines.append(
        f"- Python: `{manifest['environment']['python_version']}`"
    )
    lines.append(
        f"- Platform: `{manifest['environment']['platform']}`"
    )
    lines.append(
        f"- Pytest passed: `{manifest['validation']['pytest']['passed']}`"
    )
    lines.append("")

    lines.append(
        "## External-test rule"
    )
    lines.append("")
    lines.append(
        "The 2025 sportsbook sample is development/model-selection data. "
        "The first external market test is 2026-27."
    )
    lines.append("")
    lines.append(
        "No betting threshold is frozen from 2025 ROI. "
        "The monitoring grid is recorded prospectively."
    )
    lines.append("")

    lines.append(
        "## Monitoring edge grid"
    )
    lines.append("")
    lines.append(
        ", ".join(
            f"{100*x:g}%"
            for x in manifest[
                "betting_policy"
            ][
                "monitoring_edge_grid"
            ]
        )
    )
    lines.append("")

    lines.append(
        "## Frozen files"
    )
    lines.append("")

    for section in [
        "model_artifacts",
        "source_files",
        "scripts",
        "configs",
        "data_audit_files",
    ]:
        lines.append(
            f"### {section.replace('_', ' ').title()}"
        )
        lines.append("")

        records = manifest[
            "files"
        ][
            section
        ]

        for record in records:
            lines.append(
                f"- `{record['path']}` — "
                f"`{record['sha256']}`"
            )

        lines.append("")

    return "\n".join(
        lines
    )


def main() -> None:
    args = parse_args()

    if not (
        ROOT
        / "src/nba_prop_quant"
    ).exists():
        raise SystemExit(
            "ERROR: run from the nba_prop_quant_blueprint project root."
        )

    stale_marker = (
        MODEL_DIR
        / "DEPENDENCE_ARTIFACTS_STALE_AFTER_ZINB_FIX.txt"
    )

    if stale_marker.exists():
        raise SystemExit(
            "ERROR: stale dependence marker still exists: "
            f"{stale_marker}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    created = datetime.now(
        timezone.utc
    )

    freeze_id = (
        "nba_prop_quant_"
        + created.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    pytest_result = run_pytest()

    if not pytest_result[
        "passed"
    ]:
        raise SystemExit(
            "ERROR: pytest failed. Manifest not frozen.\n"
            + pytest_result[
                "output"
            ]
        )

    model_paths = [
        MODEL_DIR
        / name
        for name in (
            CRITICAL_MODEL_ARTIFACTS
            + TARGET_MODEL_PATTERNS
        )
    ]

    model_paths.extend(
        find_minutes_model()
    )

    seen = set()
    model_records = []
    missing_models = []

    for path in model_paths:
        if path in seen:
            continue

        seen.add(
            path
        )

        if path.exists():
            model_records.append(
                file_record(
                    path
                )
            )
        else:
            missing_models.append(
                str(
                    path.relative_to(
                        ROOT
                    )
                )
            )

    source_records, missing_sources = records_for_paths(
        CRITICAL_SOURCE_FILES
    )

    script_records, missing_scripts = records_for_paths(
        CRITICAL_SCRIPT_FILES
    )

    config_records, missing_configs = records_for_paths(
        CRITICAL_CONFIG_FILES
    )

    data_records, missing_data = records_for_paths(
        KEY_DATA_AUDIT_FILES
    )

    freeze_path = (
        OUTPUT_DIR
        / f"{freeze_id}_pip_freeze.txt"
    )

    pip_record = pip_freeze_record(
        freeze_path
    )

    manifest = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "freeze_stage": args.stage,
        "created_utc": created.isoformat(),
        "project_root": str(
            ROOT
        ),
        "external_test_status": {
            "development_market_period": "2025 season / Jan-Jun 2026 opening-prop coverage",
            "first_external_market_test": "2026-27",
            "2025_is_external_test": False,
        },
        "betting_policy": {
            "threshold_frozen": False,
            "reason": (
                "No edge threshold is selected from 2025 ROI; "
                "thresholds remain a fixed prospective monitoring grid."
            ),
            "monitoring_edge_grid": BETTING_MONITORING_GRID,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "pip_freeze": pip_record,
        },
        "git": git_info(),
        "validation": {
            "pytest": pytest_result,
        },
        "policies": summarize_policy(),
        "files": {
            "model_artifacts": model_records,
            "source_files": source_records,
            "scripts": script_records,
            "configs": config_records,
            "data_audit_files": data_records,
        },
        "missing_optional_files": {
            "model_artifacts": missing_models,
            "source_files": missing_sources,
            "scripts": missing_scripts,
            "configs": missing_configs,
            "data_audit_files": missing_data,
        },
    }

    json_path = (
        OUTPUT_DIR
        / f"{freeze_id}.json"
    )

    markdown_path = (
        OUTPUT_DIR
        / f"{freeze_id}.md"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

    markdown_path.write_text(
        human_markdown(
            manifest
        ),
        encoding="utf-8",
    )

    latest_json = (
        OUTPUT_DIR
        / "LATEST.json"
    )

    latest_md = (
        OUTPUT_DIR
        / "LATEST.md"
    )

    shutil.copy2(
        json_path,
        latest_json,
    )

    shutil.copy2(
        markdown_path,
        latest_md,
    )

    print(
        "=" * 118
    )

    print(
        "NBA PROP QUANT PRODUCTION FREEZE COMPLETE"
    )

    print(
        "=" * 118
    )

    print(
        f"Freeze ID: {freeze_id}"
    )

    print(
        f"Freeze stage: {args.stage}"
    )

    print(
        f"Pytest: PASS"
    )

    print(
        f"Model artifacts hashed: "
        f"{len(model_records)}"
    )

    print(
        f"Source files hashed: "
        f"{len(source_records)}"
    )

    print(
        f"Scripts hashed: "
        f"{len(script_records)}"
    )

    print(
        f"Data/audit artifacts hashed: "
        f"{len(data_records)}"
    )

    print()

    if missing_models:
        print(
            "Missing optional model paths:"
        )

        for item in missing_models:
            print(
                "  ",
                item,
            )

        print()

    print(
        f"Manifest JSON: {json_path}"
    )

    print(
        f"Manifest Markdown: {markdown_path}"
    )

    print(
        f"Latest pointer: {latest_json}"
    )

    print()

    print(
        "BETTING THRESHOLD STATUS: NOT FROZEN."
    )

    print(
        "2026-27 remains the first external market test."
    )


if __name__ == "__main__":
    main()
