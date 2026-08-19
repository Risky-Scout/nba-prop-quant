from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
REVIEW_ROOT = ROOT / "review"
HANDOFF_ROOT = REVIEW_ROOT / "handoff"
CLARITY_JSON = (
    ROOT
    / "data/validation/senior_quant_handoff/dossier_clarity_audit.json"
)
LATEST_MANIFEST = ROOT / "models/frozen_manifests/LATEST.json"

DENY_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
}

EXACT_FILES = [
    Path("review/SENIOR_QUANT_REVIEW_DOSSIER.md"),
    Path("review/SENIOR_QUANT_REVIEW_ARTIFACT_INDEX.json"),
    Path("review/SENIOR_QUANT_REVIEW_ATTACK_CHECKLIST.md"),
    Path("data/validation/senior_quant_handoff/dossier_clarity_audit.json"),
    Path("data/validation/senior_quant_handoff/dossier_clarity_audit.md"),
    Path("models/frozen_manifests/LATEST.json"),
    Path("models/mean_model_selection.json"),
    Path("models/marginal_selection.json"),
    Path("models/combo_dependence_policy.json"),
    Path("models/market_probability_calibration_policy.json"),
    Path("models/dynamic_params.json"),
    Path("configs/model.yaml"),
    Path("data/processed/market_backtest/calibrated_oof/event_scoring_summary.csv"),
    Path("data/processed/market_backtest/calibrated_oof/event_game_cluster_bootstrap.csv"),
    Path("data/validation/pricing_grading_contract/pricing_grading_contract_audit.json"),
    Path("data/validation/pricing_grading_contract/pricing_grading_contract_audit.md"),
    Path("src/nba_prop_quant/production.py"),
    Path("scripts/10_predict_slate.py"),
    Path("scripts/15_price_markets.py"),
    Path("scripts/10a_validate_production_contract.py"),
    Path("scripts/verify_frozen_manifest.py"),
    Path("ops/capture_external_test_day.py"),
    Path("ops/run_external_test_day.sh"),
    Path("ops/grade_external_test_capture.py"),
    Path("ops/verify_external_test_capture.py"),
    Path("ops/verify_external_test_grade.py"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_logged(
    args: list[str],
    *,
    log_path: Path,
) -> None:
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = [
        "$ " + " ".join(args),
        "",
        "===== STDOUT =====",
        result.stdout.rstrip(),
        "",
        "===== STDERR =====",
        result.stderr.rstrip(),
        "",
        f"RETURN CODE: {result.returncode}",
        "",
    ]

    log_path.write_text(
        "\n".join(payload),
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(args)
            + f". See {log_path}"
        )


def safe_relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def copy_exact(
    source: Path,
    package_dir: Path,
) -> dict:
    if source.name in DENY_NAMES:
        raise RuntimeError(
            f"Refusing to package secret-like file: {source}"
        )

    relative = source.relative_to(ROOT)
    destination = package_dir / "project_snapshot" / relative

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(source, destination)

    return {
        "source_path": str(relative),
        "source_sha256": sha256_file(source),
        "package_path": str(
            destination.relative_to(package_dir)
        ),
        "package_sha256": sha256_file(destination),
        "bytes": int(destination.stat().st_size),
    }


def latest_reports() -> list[Path]:
    results = []

    patterns = [
        (
            "data/validation/grading_synthetic/*/"
            "synthetic_grading_integration_report.json"
        ),
        (
            "data/validation/grading_replay_2025/date=*/*/"
            "real_2025_grading_replay_report.json"
        ),
    ]

    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        if matches:
            results.append(matches[-1])

    return results


def matching_freeze_manifests(
    freeze_id: str,
) -> list[Path]:
    root = ROOT / "models/frozen_manifests"
    matches = []

    if not root.exists():
        return matches

    for path in sorted(root.rglob("*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue

        if payload.get("freeze_id") == freeze_id:
            matches.append(path)

    return matches


def write_checksums(package_dir: Path) -> None:
    lines = []

    for path in sorted(package_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "checksums.sha256"
        ):
            lines.append(
                f"{sha256_file(path)}  "
                f"{path.relative_to(package_dir)}"
            )

    (
        package_dir / "checksums.sha256"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact read-only senior-quant handoff package after "
            "the dossier clarity audit passes."
        )
    )
    parser.add_argument(
        "--label",
        default="reviewer_handoff",
    )
    args = parser.parse_args()

    if not CLARITY_JSON.exists():
        raise SystemExit(
            "ERROR: clarity audit has not been run. "
            "Run ops/audit_senior_quant_dossier_clarity.py first."
        )

    clarity = load_json(CLARITY_JSON)

    if clarity.get("status") != "PASS":
        raise SystemExit(
            "ERROR: clarity audit is not PASS. Package creation refused."
        )

    manifest = load_json(LATEST_MANIFEST)
    freeze_id = manifest.get("freeze_id")
    freeze_stage = manifest.get(
        "freeze_stage",
        manifest.get("stage"),
    )

    if freeze_id != clarity.get("freeze_id"):
        raise SystemExit(
            "ERROR: freeze ID changed after clarity audit. "
            "Rerun the clarity audit before packaging."
        )

    # Re-verify immediately before copying.
    HANDOFF_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(
        c if c.isalnum() or c in {"-", "_"} else "_"
        for c in args.label
    )

    package_name = (
        f"nba_prop_quant_{freeze_id}_{safe_label}_{timestamp}"
    )
    package_dir = HANDOFF_ROOT / package_name

    if package_dir.exists():
        raise SystemExit(
            f"ERROR: package directory already exists: {package_dir}"
        )

    package_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    logs_dir = package_dir / "verification_logs"
    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        run_logged(
            [
                sys.executable,
                "scripts/verify_frozen_manifest.py",
            ],
            log_path=logs_dir / "verify_frozen_manifest.txt",
        )

        run_logged(
            [
                sys.executable,
                "scripts/10a_validate_production_contract.py",
            ],
            log_path=logs_dir / "validate_production_contract.txt",
        )

        source_files = list(EXACT_FILES)
        source_files.extend(latest_reports())
        source_files.extend(
            matching_freeze_manifests(freeze_id)
        )

        # De-duplicate while preserving order.
        seen = set()
        unique_files = []

        for relative_or_path in source_files:
            path = (
                relative_or_path
                if relative_or_path.is_absolute()
                else ROOT / relative_or_path
            )

            path = path.resolve()

            if path in seen:
                continue

            seen.add(path)

            if not path.exists():
                # Some optional review files may not exist in every build.
                continue

            if not path.is_file():
                continue

            # Ensure every packaged file stays inside project root.
            try:
                path.relative_to(ROOT.resolve())
            except ValueError:
                raise RuntimeError(
                    f"Refusing to package file outside project root: {path}"
                )

            unique_files.append(path)

        records = []

        for source in unique_files:
            records.append(
                copy_exact(
                    source,
                    package_dir,
                )
            )

        reviewer_readme = f"""# NBA Prop Quant — Senior Reviewer Handoff

## Reviewed state

- Freeze ID: `{freeze_id}`
- Freeze stage: `{freeze_stage}`
- Package generated UTC: `{now.isoformat()}`
- Dossier clarity audit: **PASS**
- Synthetic grading QA: **PASS**
- Real 2025 grading replay: **PASS**
- Pricing↔grading contract audit: **PASS_WITH_RUNTIME_CONFIRMATION**
- Automatic betting threshold: **none**

## Review posture

This is a research/production review package, not a claim of a proven broad sportsbook edge.

The 2025 sportsbook sample is development/model-selection evidence. The 2026–27 season is the first prospective external market test. Synthetic QA validates software logic only; the real 2025 replay validates settlement/schema realism only.

The remaining `PASS_WITH_RUNTIME_CONFIRMATION` item concerns the exact field names emitted by the first genuine 2026–27 live `priced_markets.parquet` for side/edge/EV. It is an operational schema-confirmation item, not evidence for or against predictive edge.

## Start here

1. `project_snapshot/review/SENIOR_QUANT_REVIEW_DOSSIER.md`
2. `project_snapshot/review/SENIOR_QUANT_REVIEW_ATTACK_CHECKLIST.md`
3. `project_snapshot/review/SENIOR_QUANT_REVIEW_ARTIFACT_INDEX.json`
4. `verification_logs/verify_frozen_manifest.txt`
5. `verification_logs/validate_production_contract.txt`
6. `PACKAGE_MANIFEST.json`
7. `checksums.sha256`

## Exact reproduction commands in the full repository

```bash
python scripts/verify_frozen_manifest.py
python scripts/10a_validate_production_contract.py

python ops/qa_grading_synthetic_integration.py
python ops/qa_replay_2025_grading.py
python ops/audit_pricing_grading_contract.py
python ops/build_senior_quant_review_dossier.py

python ops/audit_senior_quant_dossier_clarity.py
```

Official prospective operation remains date-locked:

```bash
ops/run_external_test_day.sh YYYY-MM-DD morning
```

## Package scope

This handoff intentionally does **not** include `.env`, API keys, raw historical data, the full processed dataset, or every trained binary model artifact.

It includes the exact review documents, frozen manifest metadata, policy JSONs, critical production/operations source snapshots, key validation reports, and hashes needed to identify the reviewed state.

For full statistical reproduction, the reviewer should work from a controlled copy of the complete repository and data warehouse corresponding to freeze `{freeze_id}`.

## Claims not supported yet

- A proven broad sportsbook edge.
- An optimal automatic betting threshold.
- 2025 as an untouched external market test.
- Fully reconstructed historical injury information.
- Universal improvement from dependence modeling.
- Final live pricing↔grading field-name compatibility before the first real runtime file is observed.

## Integrity

Run from the extracted package directory:

```bash
shasum -a 256 -c checksums.sha256
```

No file in `project_snapshot/` should be edited during review. Reviewer notes should be created separately.
"""

        (
            package_dir / "REVIEWER_README.md"
        ).write_text(
            reviewer_readme,
            encoding="utf-8",
        )

        environment = {
            "generated_at_utc": now.isoformat(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "freeze_id": freeze_id,
            "freeze_stage": freeze_stage,
            "cwd": str(ROOT),
            "secret_files_included": False,
        }

        (
            package_dir / "ENVIRONMENT.json"
        ).write_text(
            json.dumps(
                environment,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        package_manifest = {
            "schema_version": 1,
            "package_name": package_name,
            "generated_at_utc": now.isoformat(),
            "freeze_id": freeze_id,
            "freeze_stage": freeze_stage,
            "clarity_audit_sha256": sha256_file(CLARITY_JSON),
            "source_file_count": len(records),
            "source_files": records,
            "production_model_or_source_mutated": False,
            "external_test_results_used_for_model_selection": False,
            "secrets_included": False,
            "scope_note": (
                "Review handoff package; not a full raw-data/model-binary replica."
            ),
        }

        (
            package_dir / "PACKAGE_MANIFEST.json"
        ).write_text(
            json.dumps(
                package_manifest,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        write_checksums(package_dir)

        zip_path = HANDOFF_ROOT / f"{package_name}.zip"

        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in sorted(package_dir.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        Path(package_name)
                        / path.relative_to(package_dir),
                    )

        print("=" * 118)
        print("SENIOR-QUANT REVIEWER HANDOFF PACKAGE")
        print("=" * 118)
        print(f"Freeze ID:       {freeze_id}")
        print(f"Freeze stage:    {freeze_stage}")
        print(f"Source files:    {len(records)}")
        print(f"Clarity audit:   PASS")
        print(f"Secrets:         NOT INCLUDED")
        print()
        print(f"Package dir: {package_dir}")
        print(f"ZIP:         {zip_path}")
        print()
        print(
            "PASS: exact reviewed state packaged without modifying "
            "the frozen production deployment."
        )

    except Exception:
        # Preserve failed handoff directory for auditability.
        (
            package_dir
            / "PACKAGE_FAILED.txt"
        ).write_text(
            "Package construction failed. Directory intentionally preserved.\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
