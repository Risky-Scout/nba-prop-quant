from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
LATEST = ROOT / "models/frozen_manifests/LATEST.json"
WORKPAPER = (
    ROOT
    / "review/workpapers/NBA_Player_Prop_Model_Workpaper_2026_08_19.docx"
)
INVENTORY_SCRIPT = (
    ROOT
    / "ops/inventory_model_constants.py"
)

OUTPUT_ROOT = ROOT / "dist/model_packages"

SECRET_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
}

EXTRA_PROJECT_FILES = [
    Path("README.md"),
    Path("pyproject.toml"),
    Path("configs/model.yaml"),
]

OPERATIONAL_SUPPORT_PATTERNS = [
    "ops/*.py",
    "ops/*.sh",
    "ops/*.md",
]

FULL_SOURCE_PATTERNS = [
    "src/nba_prop_quant/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def run_command(
    args: list[str],
    log_path: Path,
) -> None:
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path.write_text(
        "\n".join(
            [
                "$ "
                + " ".join(
                    args
                ),
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
        ),
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(
                args
            )
            + f". See {log_path}"
        )


def check_secret_name(
    path: Path,
) -> None:
    if path.name in SECRET_NAMES:
        raise RuntimeError(
            f"Secret-like path is not packageable: {path}"
        )


def copy_with_record(
    source: Path,
    destination_root: Path,
    *,
    package_relative: Path,
    expected_sha256: str | None = None,
    provenance: str,
) -> dict:
    check_secret_name(
        source
    )

    if expected_sha256 is not None:
        actual = sha256_file(
            source
        )

        if actual != expected_sha256:
            raise RuntimeError(
                "Frozen source hash mismatch before package copy: "
                f"{source}\nexpected={expected_sha256}\nactual={actual}"
            )

    destination = (
        destination_root
        / package_relative
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    source_hash = sha256_file(
        source
    )

    packaged_hash = sha256_file(
        destination
    )

    if source_hash != packaged_hash:
        raise RuntimeError(
            f"Copy hash mismatch: {source}"
        )

    return {
        "source_path": str(
            source.relative_to(
                ROOT
            )
        ),
        "package_path": str(
            package_relative
        ),
        "sha256": packaged_hash,
        "bytes": int(
            destination.stat().st_size
        ),
        "provenance": provenance,
    }


def manifest_records(
    manifest: dict,
) -> list[
    tuple[str, dict]
]:
    rows = []

    files = manifest.get(
        "files",
        {},
    )

    for section in (
        "model_artifacts",
        "source_files",
        "scripts",
        "configs",
        "data_audit_files",
    ):
        for record in files.get(
            section,
            []
        ):
            rows.append(
                (
                    section,
                    record,
                )
            )

    return rows


def matching_freeze_files(
    freeze_id: str,
) -> list[Path]:
    root = ROOT / "models/frozen_manifests"
    result = []

    if not root.exists():
        return result

    for path in sorted(
        root.iterdir()
    ):
        if not path.is_file():
            continue

        if freeze_id in path.name:
            result.append(
                path
            )

    for name in (
        "LATEST.json",
        "LATEST.md",
    ):
        path = root / name

        if path.exists():
            result.append(
                path
            )

    return result


def checksum_manifest(
    package_dir: Path,
) -> None:
    lines = []

    for path in sorted(
        package_dir.rglob(
            "*"
        )
    ):
        if (
            path.is_file()
            and path.name
            != "checksums.sha256"
        ):
            lines.append(
                f"{sha256_file(path)}  "
                f"{path.relative_to(package_dir)}"
            )

    (
        package_dir
        / "checksums.sha256"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hash-verified NBA prop model package with work paper "
            "and exhaustive constant inventory."
        )
    )

    parser.add_argument(
        "--label",
        default="production_model_package",
    )

    args = parser.parse_args()

    if not (
        ROOT
        / "src/nba_prop_quant"
    ).exists():
        raise SystemExit(
            "ERROR: run from nba_prop_quant_blueprint project root."
        )

    for required in (
        LATEST,
        WORKPAPER,
        INVENTORY_SCRIPT,
    ):
        if not required.exists():
            raise SystemExit(
                f"ERROR: required package input missing: {required}"
            )

    manifest = load_json(
        LATEST
    )

    freeze_id = manifest[
        "freeze_id"
    ]

    freeze_stage = manifest.get(
        "freeze_stage"
    )

    now = datetime.now(
        timezone.utc
    )

    timestamp = now.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    safe_label = "".join(
        char
        if char.isalnum()
        or char in {
            "-",
            "_",
        }
        else "_"
        for char in args.label
    )

    package_name = (
        f"{freeze_id}_{safe_label}_{timestamp}"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    package_dir = (
        OUTPUT_ROOT
        / package_name
    )

    if package_dir.exists():
        raise SystemExit(
            f"ERROR: package already exists: {package_dir}"
        )

    package_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    logs = (
        package_dir
        / "verification_logs"
    )

    try:
        run_command(
            [
                sys.executable,
                "scripts/verify_frozen_manifest.py",
            ],
            logs
            / "verify_frozen_manifest.txt",
        )

        run_command(
            [
                sys.executable,
                "scripts/10a_validate_production_contract.py",
            ],
            logs
            / "validate_production_contract.txt",
        )

        inventory_out = (
            package_dir
            / "constant_inventory"
        )

        run_command(
            [
                sys.executable,
                str(
                    INVENTORY_SCRIPT.relative_to(
                        ROOT
                    )
                ),
                "--output-dir",
                str(
                    inventory_out
                ),
            ],
            logs
            / "inventory_model_constants.txt",
        )

        records = []
        seen_sources = set()

        # 1. Exact frozen files from the authoritative manifest.
        for section, record in manifest_records(
            manifest
        ):
            relative = Path(
                record[
                    "path"
                ]
            )

            source = (
                ROOT
                / relative
            )

            if not source.exists():
                raise RuntimeError(
                    f"Frozen manifest file missing: {source}"
                )

            records.append(
                copy_with_record(
                    source,
                    package_dir,
                    package_relative=(
                        Path(
                            "frozen_project"
                        )
                        / relative
                    ),
                    expected_sha256=record[
                        "sha256"
                    ],
                    provenance=(
                        "frozen_manifest:"
                        + section
                    ),
                )
            )

            seen_sources.add(
                source.resolve()
            )

        # 2. Freeze manifest metadata and pip-freeze files.
        for source in matching_freeze_files(
            freeze_id
        ):
            if source.resolve() in seen_sources:
                continue

            relative = source.relative_to(
                ROOT
            )

            records.append(
                copy_with_record(
                    source,
                    package_dir,
                    package_relative=(
                        Path(
                            "frozen_project"
                        )
                        / relative
                    ),
                    expected_sha256=None,
                    provenance="freeze_metadata",
                )
            )

            seen_sources.add(
                source.resolve()
            )

        # 3. Work paper.
        records.append(
            copy_with_record(
                WORKPAPER,
                package_dir,
                package_relative=Path(
                    "documentation/NBA_Player_Prop_Model_Workpaper_2026_08_19.docx"
                ),
                expected_sha256=None,
                provenance="workpaper",
            )
        )

        # 4. Complete current Python source/test snapshot. Frozen-manifest
        # files remain authoritative for the frozen production state; this
        # extra snapshot makes the package reviewable as a complete codebase.
        for pattern in FULL_SOURCE_PATTERNS:
            for source in sorted(
                ROOT.glob(
                    pattern
                )
            ):
                if (
                    not source.is_file()
                    or "__pycache__"
                    in source.parts
                    or ".before_"
                    in source.name
                ):
                    continue

                if source.resolve() in seen_sources:
                    continue

                relative = source.relative_to(
                    ROOT
                )

                records.append(
                    copy_with_record(
                        source,
                        package_dir,
                        package_relative=(
                            Path(
                                "full_current_source_snapshot"
                            )
                            / relative
                        ),
                        expected_sha256=None,
                        provenance="current_source_snapshot",
                    )
                )

                seen_sources.add(
                    source.resolve()
                )

        # 5. Current operational-support layer. These are explicitly not
        # represented as part of the frozen statistical model unless they
        # also appear in the frozen manifest.
        for pattern in OPERATIONAL_SUPPORT_PATTERNS:
            for source in sorted(
                ROOT.glob(
                    pattern
                )
            ):
                if (
                    not source.is_file()
                    or ".before_"
                    in source.name
                ):
                    continue

                if source.resolve() in seen_sources:
                    continue

                relative = source.relative_to(
                    ROOT
                )

                records.append(
                    copy_with_record(
                        source,
                        package_dir,
                        package_relative=(
                            Path(
                                "operational_support"
                            )
                            / relative
                        ),
                        expected_sha256=None,
                        provenance="current_operational_support",
                    )
                )

                seen_sources.add(
                    source.resolve()
                )

        # 6. Small package/project metadata.
        for relative in EXTRA_PROJECT_FILES:
            source = (
                ROOT
                / relative
            )

            if (
                not source.exists()
                or not source.is_file()
                or source.resolve()
                in seen_sources
            ):
                continue

            records.append(
                copy_with_record(
                    source,
                    package_dir,
                    package_relative=(
                        Path(
                            "project_metadata"
                        )
                        / relative
                    ),
                    expected_sha256=None,
                    provenance="project_metadata",
                )
            )

            seen_sources.add(
                source.resolve()
            )

        # Package README.
        readme = f"""# NBA Player Prop Quant Model Package

## Frozen statistical deployment

- Freeze ID: `{freeze_id}`
- Freeze stage: `{freeze_stage}`
- Package generated UTC: `{now.isoformat()}`

The `frozen_project/` tree is copied from the production freeze manifest and
verified against the manifest SHA-256 values before packaging.

The `operational_support/` tree contains current operational capture, grading,
preflight, and review utilities. These files are included for operability and
auditability; they are not silently represented as frozen model artifacts unless
they also appear in the frozen manifest.

## Work paper

`documentation/NBA_Player_Prop_Model_Workpaper_2026_08_19.docx`

## Constant register

`constant_inventory/FULL_CONSTANT_INVENTORY.csv`
- every non-docstring Python `ast.Constant` occurrence in the configured model source scope.

`constant_inventory/NAMED_CONSTANTS.csv`
- literal module/class assignments, function defaults, keyword-only defaults,
  and argparse defaults/choices.

`constant_inventory/SHELL_CONSTANTS.csv`
- shell assignments plus numeric and quoted literals from operational/model shell scripts.

`constant_inventory/CONFIG_CONSTANTS.csv`
- scalar leaves from model/config JSON, YAML, and TOML.

`constant_inventory/FULL_CONSTANT_INVENTORY.json`
- machine-readable full register and parse coverage.

The package builder fails if any scanned Python/config file produces a parse error.

## Backtesting posture

2025 sportsbook data are development/model-selection evidence.
The first prospective external market test is 2026-27.
No automatic betting threshold is frozen from the 2025 ROI diagnostics.

## Secrets / raw data

This package intentionally excludes `.env`, API keys, raw historical warehouses,
and unlisted credentials/secrets.

## Integrity

From the package root:

```bash
shasum -a 256 -c checksums.sha256
```
"""

        (
            package_dir
            / "README_MODEL_PACKAGE.md"
        ).write_text(
            readme,
            encoding="utf-8",
        )

        package_manifest = {
            "schema_version": 1,
            "package_name": package_name,
            "generated_at_utc": now.isoformat(),
            "freeze_id": freeze_id,
            "freeze_stage": freeze_stage,
            "frozen_manifest_verified": True,
            "production_contract_verified": True,
            "workpaper_included": True,
            "constant_inventory_included": True,
            "raw_data_included": False,
            "secrets_included": False,
            "source_file_records": records,
            "constant_inventory_files": [
                str(
                    path.relative_to(
                        package_dir
                    )
                )
                for path in sorted(
                    inventory_out.iterdir()
                )
                if path.is_file()
            ],
        }

        (
            package_dir
            / "MODEL_PACKAGE_MANIFEST.json"
        ).write_text(
            json.dumps(
                package_manifest,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        checksum_manifest(
            package_dir
        )

        zip_path = (
            OUTPUT_ROOT
            / (
                package_name
                + ".zip"
            )
        )

        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in sorted(
                package_dir.rglob(
                    "*"
                )
            ):
                if path.is_file():
                    archive.write(
                        path,
                        Path(
                            package_name
                        )
                        / path.relative_to(
                            package_dir
                        ),
                    )

        print("=" * 118)
        print("NBA PROP QUANT MODEL PACKAGE")
        print("=" * 118)
        print(
            f"Freeze ID:            {freeze_id}"
        )
        print(
            f"Freeze stage:         {freeze_stage}"
        )
        print(
            f"Packaged source files:{len(records):,}"
        )
        print(
            "Work paper:           INCLUDED"
        )
        print(
            "Constant inventory:   INCLUDED"
        )
        print(
            "Secrets/raw data:     EXCLUDED"
        )
        print()
        print(
            f"Package directory: {package_dir}"
        )
        print(
            f"ZIP:               {zip_path}"
        )
        print()
        print(
            "PASS: model package created from the verified frozen deployment."
        )

    except Exception:
        (
            package_dir
            / "PACKAGE_FAILED.txt"
        ).write_text(
            "Package construction failed. Directory intentionally preserved for audit.\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
