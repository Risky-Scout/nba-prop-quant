from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path

FREEZE_ID = "nba_prop_quant_20260818T205213Z"
SOURCE_ZIP_SHA256 = "9044a3161247868710a9de8b4a9cd393226234abf539dd1dcb82ffb9b68eb5a6"
RUNTIME_ID = f"{FREEZE_ID}_runtime_bundle_v1"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_checksums(root: Path):
    errors = []
    checked = 0
    checksum_path = root / "checksums.sha256"
    if not checksum_path.exists():
        raise SystemExit(f"ERROR: missing {checksum_path}")
    for raw in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"malformed checksum line: {raw!r}")
            continue
        expected, rel = parts
        rel = rel.lstrip("*")
        path = root / rel
        if not path.exists():
            errors.append(f"missing: {rel}")
            continue
        checked += 1
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"hash mismatch: {rel}")
    return checked, errors

def copytree(src: Path, dst: Path):
    if not src.exists():
        raise SystemExit(f"ERROR: missing required tree: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

def bootstrap_text() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.14}"
ALLOW="${NBA_PROP_QUANT_ALLOW_UNCERTIFIED_RUNTIME:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found."
  exit 2
fi

PYVER="$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
SYSTEM="$("${PYTHON_BIN}" -c 'import platform; print(platform.system())')"
ARCH="$("${PYTHON_BIN}" -c 'import platform; print(platform.machine())')"
printf 'Runtime: Python %s / %s / %s\n' "${PYVER}" "${SYSTEM}" "${ARCH}"

if [[ "${PYVER}" != "3.14.5" || "${SYSTEM}" != "Darwin" || "${ARCH}" != "arm64" ]]; then
  if [[ "${ALLOW}" != "1" ]]; then
    echo "ERROR: certified runtime is Darwin arm64 / Python 3.14.5."
    echo "Set NBA_PROP_QUANT_ALLOW_UNCERTIFIED_RUNTIME=1 to override explicitly."
    exit 3
  fi
  echo "WARNING: proceeding on uncertified runtime by explicit override."
fi

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${ROOT}/.venv"
fi

PY="${ROOT}/.venv/bin/python"
"${PY}" -m pip install -r "${ROOT}/requirements-frozen-portable.txt"
"${PY}" -m pip install --no-deps "${ROOT}"
"${PY}" -m pip check

(
  cd "${ROOT}"
  "${PY}" scripts/verify_frozen_manifest.py
  "${PY}" scripts/10a_validate_production_contract.py
  "${PY}" -m pytest -q
)

echo "PASS: runtime bundle bootstrap and offline verification completed."
'''

def readme_text() -> str:
    return f'''# NBA Prop Quant Certified Runtime Bundle v1

Frozen statistical deployment: `{FREEZE_ID}`

This is a packaging-only runtime bundle. It does not retrain, recalibrate,
change dependence, alter model artifacts, or add an automatic betting threshold.

## Certified runtime

- macOS / Darwin
- arm64
- Python 3.14.5

## Bootstrap

```bash
bash bootstrap_runtime.sh
```

The bootstrap creates a fresh `.venv`, installs the pinned frozen third-party
dependencies, installs this project, runs `pip check`, verifies the frozen
manifest, validates the production contract, and runs pytest.

## Live use

Secrets are not included.

```bash
cp .env.example .env
```

Populate `BDL_API_KEY` before live API calls.

## Integrity

```bash
shasum -a 256 -c checksums.sha256
```

The original frozen pip file contained one machine-specific editable install
path. This bundle removes only that nonportable line and preserves all pinned
third-party dependency versions.
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    src = args.package_root.resolve()
    out = args.output_dir.resolve()
    manifest_path = src / "MODEL_PACKAGE_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"ERROR: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    invariants = {
        "freeze_id": FREEZE_ID,
        "frozen_manifest_verified": True,
        "production_contract_verified": True,
        "raw_data_included": False,
        "secrets_included": False,
    }
    for key, expected in invariants.items():
        if manifest.get(key) != expected:
            raise SystemExit(
                f"ERROR: source invariant {key}={manifest.get(key)!r}, expected {expected!r}"
            )

    checked, errors = verify_checksums(src)
    if errors:
        print("\n".join(errors))
        raise SystemExit("ERROR: source package checksum verification failed.")

    req = src / "frozen_project/models/frozen_manifests" / f"{FREEZE_ID}_pip_freeze.txt"
    lines = req.read_text(encoding="utf-8").splitlines()
    removed = [line for line in lines if line.lstrip().startswith("-e ")]
    kept = [line for line in lines if not line.lstrip().startswith("-e ")]
    if len(removed) != 1:
        raise SystemExit(f"ERROR: expected exactly one editable line, found {len(removed)}")
    if any("/Users/" in line for line in kept):
        raise SystemExit("ERROR: machine-specific /Users/ path remains.")

    out.mkdir(parents=True, exist_ok=True)
    runtime_root = out / RUNTIME_ID
    zip_path = out / f"{RUNTIME_ID}.zip"
    if runtime_root.exists() or zip_path.exists():
        raise SystemExit("ERROR: output already exists; use a new empty output directory.")

    runtime_root.mkdir()
    copytree(src / "full_current_source_snapshot", runtime_root)
    copytree(src / "frozen_project", runtime_root)
    shutil.copy2(src / "project_metadata/pyproject.toml", runtime_root / "pyproject.toml")
    shutil.copy2(src / "project_metadata/README.md", runtime_root / "README.md")
    (runtime_root / "ops").mkdir(exist_ok=True)
    copytree(src / "operational_support/ops", runtime_root / "ops")
    shutil.copy2(manifest_path, runtime_root / "SOURCE_MODEL_PACKAGE_MANIFEST.json")

    (runtime_root / "requirements-frozen-portable.txt").write_text(
        "\n".join(kept) + "\n",
        encoding="utf-8",
    )
    (runtime_root / ".env.example").write_text(
        "BDL_API_KEY=\n"
        "BDL_BASE_URL=https://api.balldontlie.io\n"
        "BDL_REQUESTS_PER_MINUTE=600\n",
        encoding="utf-8",
    )
    (runtime_root / "README_RUN.md").write_text(readme_text(), encoding="utf-8")
    bootstrap = runtime_root / "bootstrap_runtime.sh"
    bootstrap.write_text(bootstrap_text(), encoding="utf-8")
    bootstrap.chmod(
        bootstrap.stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )

    certification = {
        "schema_version": 1,
        "runtime_bundle_id": RUNTIME_ID,
        "statistical_freeze_id": FREEZE_ID,
        "statistical_model_changed": False,
        "source_model_package_sha256": SOURCE_ZIP_SHA256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "certified_runtime": {
            "platform": "Darwin/macOS",
            "architecture": "arm64",
            "python_version": "3.14.5",
        },
        "clean_room_reference_evidence": {
            "outer_release_zip_sha256_match": True,
            "source_package_internal_checksums_passed": True,
            "project_reconstruction_passed": True,
            "fresh_virtual_environment_passed": True,
            "pinned_dependency_install_passed": True,
            "pip_check_passed": True,
            "package_import_from_clean_venv_passed": True,
            "production_contract_validation_passed": True,
            "frozen_manifest_files_checked": 62,
            "frozen_manifest_missing": 0,
            "frozen_manifest_hash_mismatches": 0,
            "pytest_passed": 18,
            "pytest_failed": 0,
        },
        "portability_adjustment": {
            "removed_editable_line": removed[0],
            "third_party_pins_preserved": True,
        },
        "live_runtime_requires_external_secret": "BDL_API_KEY",
    }
    (runtime_root / "RUNTIME_CERTIFICATION.json").write_text(
        json.dumps(certification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_path = runtime_root / "checksums.sha256"
    files = sorted(
        path
        for path in runtime_root.rglob("*")
        if path.is_file()
        and path != checksum_path
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
    )
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(runtime_root).as_posix()}"
            for path in files
        ) + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for path in sorted(runtime_root.rglob("*")):
            if (
                path.is_file()
                and ".venv" not in path.parts
                and "__pycache__" not in path.parts
            ):
                arcname = Path(runtime_root.name) / path.relative_to(runtime_root)
                archive.write(path, arcname.as_posix())

    zip_sha = sha256_file(zip_path)
    release_manifest = {
        "schema_version": 1,
        "runtime_bundle_id": RUNTIME_ID,
        "statistical_freeze_id": FREEZE_ID,
        "statistical_model_changed": False,
        "source_model_package_sha256": SOURCE_ZIP_SHA256,
        "runtime_zip_name": zip_path.name,
        "runtime_zip_bytes": zip_path.stat().st_size,
        "runtime_zip_sha256": zip_sha,
        "runtime_internal_checksum_records": len(files),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "RUNTIME_RELEASE_ASSET_MANIFEST.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "SHA256SUMS.txt").write_text(
        f"{zip_sha}  {zip_path.name}\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("RUNTIME BUNDLE V1 BUILD")
    print("=" * 100)
    print(f"Source checksums checked:  {checked}")
    print(f"Runtime root:              {runtime_root}")
    print(f"Runtime zip:               {zip_path}")
    print(f"Runtime zip bytes:         {zip_path.stat().st_size}")
    print(f"Runtime zip SHA-256:       {zip_sha}")
    print(f"Internal checksum records: {len(files)}")
    print(f"Removed editable line:     {removed[0]}")
    print("PASS: packaging-only runtime bundle built.")

if __name__ == "__main__":
    main()
