"""Build the canonical NBA Prop Quant v2 / Gate 3 production runtime bundle.

This is a deployment-integrity tool. It packages the already-frozen v2 model so
it can be installed, verified and executed reproducibly on a clean hosted
runner. It performs no training, fitting, selection or recalibration, and it
never contacts an external API.

The bundle it produces is shaped like a project root, so the unchanged
``nba_prop_quant.production.load_verified_manifest_metadata`` verifier accepts
it with ``project_root=<bundle>`` and ``model_dir=<bundle>/models``.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import gzip
import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any


DEFAULT_CONTRACT = (
    "models/frozen_manifests/"
    "nba_prop_quant_v2_gate3_runtime_contract.json"
)

RUNTIME_MANIFEST_NAME = "RUNTIME_MANIFEST.json"
RUNTIME_SHA256SUMS_NAME = "RUNTIME_SHA256SUMS.txt"

CONTENT_SCAN_MAX_BYTES = 2 * 1024 * 1024

TEXT_SCAN_SUFFIXES = {
    ".cfg",
    ".conf",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class BuildError(RuntimeError):
    """Raised when the runtime bundle cannot be built safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and verify the frozen NBA Prop Quant v2 / Gate 3 "
            "production runtime bundle."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository checkout containing src/, scripts/ and research/.",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the frozen model artifacts. "
            "Defaults to <project-root>/models."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding historical runtime state "
            "(raw/seasons, raw/advanced). Defaults to <project-root>/data."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the staged bundle and archive.",
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help=(
            "Runtime contract JSON. "
            f"Defaults to <project-root>/{DEFAULT_CONTRACT}."
        ),
    )

    parser.add_argument(
        "--gate3-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Gate 3 deployment artifact directory. Defaults to "
            "<project-root>/research/v2_gate3_deployment_artifacts."
        ),
    )

    parser.add_argument(
        "--pip-freeze",
        type=Path,
        default=None,
        help=(
            "Optional dependency lock file to hash into the manifest "
            "(for example the output of `pip freeze`)."
        ),
    )

    parser.add_argument(
        "--runtime-version",
        type=int,
        default=None,
        help="Override the runtime version recorded in the manifest.",
    )

    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Permit building from a worktree with uncommitted tracked "
            "changes. The manifest records the dirty state either way."
        ),
    )

    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the staged bundle directory after the archive is built.",
    )

    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise BuildError(
            f"git {' '.join(args)} failed in {root}: "
            f"{result.stderr.strip()}"
        )

    return result.stdout.strip()


def resolve_source_provenance(
    root: Path,
    model_source_commit: str,
    allow_dirty: bool,
) -> dict[str, Any]:
    """Confirm the checkout descends from the frozen model commit."""

    inside = git(root, "rev-parse", "--is-inside-work-tree")

    if inside != "true":
        raise BuildError(f"{root} is not a git worktree.")

    head = git(root, "rev-parse", "HEAD")

    try:
        branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    except BuildError:
        branch = "DETACHED"

    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            model_source_commit,
            head,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    if ancestry.returncode != 0:
        raise BuildError(
            "Frozen model commit "
            f"{model_source_commit} is not an ancestor of HEAD ({head}). "
            "Refusing to package a tree that is not anchored to the "
            "frozen mathematical model."
        )

    dirty_paths = [
        line
        for line in git(root, "status", "--porcelain").splitlines()
        if line and not line.startswith("??")
    ]

    if dirty_paths and not allow_dirty:
        raise BuildError(
            "Worktree has uncommitted tracked changes; the recorded "
            "production_source_commit would not describe the packaged "
            "files. Commit first or pass --allow-dirty.\n  "
            + "\n  ".join(dirty_paths[:20])
        )

    return {
        "model_source_commit": model_source_commit,
        "production_source_commit": head,
        "production_source_branch": branch,
        "production_source_dirty": bool(dirty_paths),
    }


def verify_frozen_model_sources(
    root: Path,
    model_source_commit: str,
    relative_paths: Iterable[str],
) -> dict[str, str]:
    """Fail unless every mathematical source file matches the frozen commit."""

    verified: dict[str, str] = {}
    problems: list[str] = []

    for relative in sorted(relative_paths):
        path = root / relative

        if not path.exists():
            problems.append(f"{relative}: missing from the worktree")
            continue

        blob = subprocess.run(
            ["git", "cat-file", "blob", f"{model_source_commit}:{relative}"],
            cwd=root,
            capture_output=True,
            check=False,
        )

        if blob.returncode != 0:
            problems.append(
                f"{relative}: absent from frozen commit "
                f"{model_source_commit[:12]}"
            )
            continue

        frozen_sha = sha256_bytes(blob.stdout)
        actual_sha = sha256_file(path)

        if frozen_sha != actual_sha:
            problems.append(
                f"{relative}: differs from frozen commit "
                f"{model_source_commit[:12]}"
            )
            continue

        verified[relative] = actual_sha

    if problems:
        raise BuildError(
            "Frozen mathematical model source verification failed:\n  "
            + "\n  ".join(problems)
        )

    return verified


def verify_sha256sums_group(root: Path, relative_sums: str) -> int:
    sums_path = root / relative_sums

    if not sums_path.exists():
        raise BuildError(f"Missing checksum group: {relative_sums}")

    directory = sums_path.parent
    checked = 0

    for raw in sums_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue

        expected, filename = raw.split(None, 1)
        target = directory / filename.strip()

        if not target.exists():
            raise BuildError(
                f"{relative_sums}: listed file missing: {filename.strip()}"
            )

        if sha256_file(target) != expected:
            raise BuildError(
                f"{relative_sums}: hash mismatch for {filename.strip()}"
            )

        checked += 1

    return checked


def source_gate3_policy(project_root: Path) -> dict[str, str]:
    """Read the 10-prop policy that gate3_v2.load_gate3_runtime() enforces."""

    source = project_root / "src/nba_prop_quant/gate3_v2.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        targets = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]

        if "required_policy" not in targets:
            continue

        try:
            return dict(ast.literal_eval(node.value))
        except ValueError as exc:
            raise BuildError(
                f"Could not read required_policy from {source}: {exc}"
            ) from exc

    raise BuildError(f"No required_policy assignment found in {source}")


def verify_gate3_policy(
    artifact_dir: Path,
    expected_policy: dict[str, str],
    expected_candidate_id: str,
    expected_lock_commit: str,
    project_root: Path,
) -> dict[str, Any]:
    from_source = source_gate3_policy(project_root)

    if from_source != expected_policy:
        raise BuildError(
            "Runtime contract Gate 3 policy does not match the policy "
            "enforced by src/nba_prop_quant/gate3_v2.py."
        )

    manifest_path = artifact_dir / "deployment_manifest.json"

    if not manifest_path.exists():
        raise BuildError(
            f"Gate 3 deployment manifest missing: {manifest_path}"
        )

    manifest = load_json(manifest_path)
    observed = manifest.get("gate3_policy")

    if observed != expected_policy:
        raise BuildError(
            "Gate 3 10-prop policy mismatch between the deployment "
            "manifest and the runtime contract."
        )

    lock_commit = str(manifest["gate3_lock_commit"])

    if lock_commit != expected_lock_commit:
        raise BuildError(
            "Gate 3 policy lock commit mismatch: manifest="
            f"{lock_commit}, contract={expected_lock_commit}"
        )

    candidate_id = "nba_prop_quant_v2_gate3_" + lock_commit[:12]

    if candidate_id != expected_candidate_id:
        raise BuildError(
            "Gate 3 candidate policy ID mismatch: derived="
            f"{candidate_id}, contract={expected_candidate_id}"
        )

    if manifest.get("prospective_claim_allowed") is not False:
        raise BuildError(
            "Gate 3 deployment manifest must keep "
            "prospective_claim_allowed false."
        )

    return {
        "candidate_id": candidate_id,
        "gate3_lock_commit": lock_commit,
        "deployment_manifest_sha256": sha256_file(manifest_path),
    }


def verify_capture_window(
    root: Path,
    expected_window: str,
    expected_offset_minutes: int,
) -> None:
    policy_path = (
        root / "research/v2_gate3_capture_lock/GATE3_CAPTURE_POLICY.json"
    )

    if not policy_path.exists():
        raise BuildError(f"Gate 3 capture policy missing: {policy_path}")

    policy = load_json(policy_path)
    window = policy.get("primary_certification_window", {})

    if str(window.get("window_label")) != expected_window:
        raise BuildError(
            "Gate 3 capture window label mismatch: "
            f"{window.get('window_label')!r} != {expected_window!r}"
        )

    if int(window.get("offset_minutes_before_tip", -1)) != int(
        expected_offset_minutes
    ):
        raise BuildError(
            "Gate 3 capture offset mismatch: "
            f"{window.get('offset_minutes_before_tip')} != "
            f"{expected_offset_minutes}"
        )

    if window.get("fallback_allowed") is not False:
        raise BuildError(
            "Gate 3 primary certification window must not allow fallback."
        )

    if policy.get("retuning_from_prospective_results") is not False:
        raise BuildError(
            "Gate 3 capture policy must keep "
            "retuning_from_prospective_results false."
        )

    source = root / "src/nba_prop_quant/prospective_snapshot.py"
    text = source.read_text(encoding="utf-8")
    match = re.search(r"^PRIMARY_OFFSET_MINUTES\s*=\s*(\d+)", text, re.MULTILINE)

    if match is None:
        raise BuildError(
            "Could not read PRIMARY_OFFSET_MINUTES from "
            "prospective_snapshot.py"
        )

    if int(match.group(1)) != int(expected_offset_minutes):
        raise BuildError(
            "Runtime source capture offset "
            f"({match.group(1)}) does not match the locked "
            f"{expected_window} policy."
        )


def path_is_forbidden(relative: str, globs: Iterable[str]) -> str | None:
    parts = Path(relative).parts

    for pattern in globs:
        if fnmatch.fnmatch(Path(relative).name, pattern):
            return pattern

        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return pattern

    return None


def scan_forbidden_content(
    path: Path,
    compiled_patterns: list[tuple[str, re.Pattern[str]]],
) -> str | None:
    if path.suffix.lower() not in TEXT_SCAN_SUFFIXES:
        return None

    if path.stat().st_size > CONTENT_SCAN_MAX_BYTES:
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None

    for source, pattern in compiled_patterns:
        if pattern.search(text):
            return source

    return None


def collect_bundle_files(
    contract: dict[str, Any],
    project_root: Path,
    model_dir: Path,
    data_dir: Path,
) -> tuple[dict[str, list[tuple[str, Path]]], list[str]]:
    """Resolve every contract group to (stage_relative_path, source_path)."""

    roots = {
        "project_root": project_root,
        "model_dir": model_dir,
        "data_dir": data_dir,
    }

    resolved: dict[str, list[tuple[str, Path]]] = {}
    absent_optional: list[str] = []

    for group_name in sorted(contract["groups"]):
        group = contract["groups"][group_name]
        base = roots[group["root"]]
        prefix = group.get("stage_prefix", "")
        required = bool(group.get("required", True))
        entries: list[tuple[str, Path]] = []

        for relative in group.get("files", []):
            source = base / relative
            stage = f"{prefix}/{relative}" if prefix else relative

            if not source.exists():
                if required:
                    raise BuildError(
                        f"Required runtime resource missing "
                        f"[{group_name}]: {source}"
                    )

                absent_optional.append(f"{group_name}:{relative}")
                continue

            entries.append((stage, source))

        for spec in group.get("patterns", []):
            matches = sorted(
                path
                for path in base.glob(spec["glob"])
                if path.is_file()
            )

            if len(matches) < int(spec.get("min_files", 0)):
                raise BuildError(
                    f"Required historical runtime data missing "
                    f"[{group_name}]: {base}/{spec['glob']} matched "
                    f"{len(matches)} file(s), needs at least "
                    f"{spec.get('min_files', 0)}. "
                    f"Read by {spec.get('read_by', 'the runtime')}."
                )

            for path in matches:
                relative = path.relative_to(base).as_posix()
                stage = f"{prefix}/{relative}" if prefix else relative
                entries.append((stage, path))

        resolved[group_name] = sorted(set(entries))

    return resolved, absent_optional


def enforce_secret_exclusion(
    resolved: dict[str, list[tuple[str, Path]]],
    contract: dict[str, Any],
) -> None:
    globs = list(contract.get("forbidden_path_globs", []))

    compiled = [
        (raw, re.compile(raw))
        for raw in contract.get("forbidden_content_patterns", [])
    ]

    violations: list[str] = []

    for group_name in sorted(resolved):
        for stage, source in resolved[group_name]:
            hit = path_is_forbidden(stage, globs)

            if hit is None:
                hit = path_is_forbidden(source.as_posix(), globs)

            if hit is not None:
                violations.append(
                    f"{stage}: excluded filename pattern {hit!r}"
                )
                continue

            content_hit = scan_forbidden_content(source, compiled)

            if content_hit is not None:
                violations.append(
                    f"{stage}: content matched secret pattern "
                    f"{content_hit!r}"
                )

    if violations:
        raise BuildError(
            "Refusing to package credential-bearing files:\n  "
            + "\n  ".join(violations)
        )


def stage_bundle(
    resolved: dict[str, list[tuple[str, Path]]],
    staging: Path,
) -> None:
    if staging.exists():
        shutil.rmtree(staging)

    staging.mkdir(parents=True)

    for group_name in sorted(resolved):
        for stage, source in resolved[group_name]:
            destination = staging / stage
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def file_record(staging: Path, stage: str) -> dict[str, Any]:
    path = staging / stage

    return {
        "path": stage,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def build_manifest(
    contract: dict[str, Any],
    resolved: dict[str, list[tuple[str, Path]]],
    staging: Path,
    provenance: dict[str, Any],
    gate3: dict[str, Any],
    frozen_sources: dict[str, str],
    absent_optional: list[str],
    dependency_lock: dict[str, Any],
    runtime_version: int,
    created: datetime,
) -> dict[str, Any]:
    files: dict[str, list[dict[str, Any]]] = {}

    for group_name in sorted(resolved):
        files[group_name] = [
            file_record(staging, stage)
            for stage, _ in resolved[group_name]
        ]

    def hashes_for(*group_names: str) -> dict[str, str]:
        out: dict[str, str] = {}

        for group_name in group_names:
            for record in files.get(group_name, []):
                out[record["path"]] = record["sha256"]

        return dict(sorted(out.items()))

    runtime_id = (
        f"{contract['runtime_name']}_"
        + created.strftime("%Y%m%dT%H%M%SZ")
    )

    total_bytes = sum(
        record["bytes"]
        for records in files.values()
        for record in records
    )

    file_count = sum(len(records) for records in files.values())

    return {
        "schema_version": int(contract["schema_version"]),
        "freeze_id": runtime_id,
        "freeze_stage": str(contract["manifest_freeze_stage"]),
        "runtime_name": str(contract["runtime_name"]),
        "runtime_version": int(runtime_version),
        "runtime_id": runtime_id,
        "contract_name": str(contract["contract_name"]),
        "contract_version": int(contract["contract_version"]),
        "created_utc": created.isoformat(),
        "build_timestamp_utc": created.isoformat(),
        "model_source_commit": provenance["model_source_commit"],
        "production_source_commit": provenance["production_source_commit"],
        "production_source_branch": provenance["production_source_branch"],
        "production_source_dirty": provenance["production_source_dirty"],
        "gate3_candidate_policy_id": gate3["candidate_id"],
        "gate3_policy_lock_commit": gate3["gate3_lock_commit"],
        "gate3_deployment_manifest_sha256": (
            gate3["deployment_manifest_sha256"]
        ),
        "gate3_policy": dict(contract["gate3_policy"]),
        "primary_certification_window": str(
            contract["primary_certification_window"]
        ),
        "primary_certification_offset_minutes": int(
            contract["primary_certification_offset_minutes"]
        ),
        "prospective_claim_allowed": False,
        "auto_bet": False,
        "betting_threshold_frozen": False,
        "frozen_model_source_verification": {
            "commit": provenance["model_source_commit"],
            "verified": True,
            "files": dict(sorted(frozen_sources.items())),
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "dependency_lock": dependency_lock,
        "model_artifact_hashes": hashes_for(
            "model_artifacts",
            "model_provenance_artifacts",
            "gate3_deployment_artifacts",
        ),
        "source_file_hashes": hashes_for(
            "source_files",
            "scripts",
            "packaging",
        ),
        "runtime_data_hashes": hashes_for("runtime_data"),
        "files": files,
        "absent_optional_files": sorted(absent_optional),
        "excluded_development_resources": list(
            contract.get("excluded_development_resources", [])
        ),
        "bundle": {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "manifest_file": RUNTIME_MANIFEST_NAME,
            "sha256sums_file": RUNTIME_SHA256SUMS_NAME,
            "sha256sums_self_excluded": True,
            "runtime_project_root": ".",
            "runtime_model_dir": "models",
            "runtime_data_dir": "data",
        },
    }


def write_manifest_files(
    staging: Path,
    contract: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    targets = [
        RUNTIME_MANIFEST_NAME,
        f"models/frozen_manifests/{contract['runtime_manifest_basename']}",
        "models/frozen_manifests/LATEST.json",
    ]

    for relative in targets:
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    return targets


def write_sha256sums(staging: Path) -> Path:
    lines = []

    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue

        relative = path.relative_to(staging).as_posix()

        if relative == RUNTIME_SHA256SUMS_NAME:
            continue

        lines.append(f"{sha256_file(path)}  {relative}")

    sums_path = staging / RUNTIME_SHA256SUMS_NAME
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return sums_path


def verify_sha256sums_tree(root: Path) -> int:
    sums_path = root / RUNTIME_SHA256SUMS_NAME

    if not sums_path.exists():
        raise BuildError(f"Missing {RUNTIME_SHA256SUMS_NAME} in {root}")

    listed: set[str] = set()

    for raw in sums_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue

        expected, relative = raw.split(None, 1)
        relative = relative.strip()
        target = root / relative

        if not target.exists():
            raise BuildError(f"Bundled file missing after extraction: {relative}")

        if sha256_file(target) != expected:
            raise BuildError(f"Bundled file hash mismatch: {relative}")

        listed.add(relative)

    on_disk = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }

    unlisted = on_disk - listed - {RUNTIME_SHA256SUMS_NAME}

    if unlisted:
        raise BuildError(
            "Bundled files absent from "
            f"{RUNTIME_SHA256SUMS_NAME}: {sorted(unlisted)[:10]}"
        )

    return len(listed)


def create_archive(staging: Path, archive_path: Path) -> Path:
    """Write a byte-deterministic gzip tarball of the staged bundle."""

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    members = sorted(
        path
        for path in staging.rglob("*")
        if path.is_file()
    )

    raw = io.BytesIO()

    with tarfile.open(fileobj=raw, mode="w") as tar:
        for path in members:
            relative = path.relative_to(staging).as_posix()
            info = tarfile.TarInfo(name=f"{staging.name}/{relative}")
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE

            with path.open("rb") as handle:
                tar.addfile(info, handle)

    with archive_path.open("wb") as handle, gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=handle,
        mtime=0,
    ) as gz:
        gz.write(raw.getvalue())

    return archive_path


def import_verifier(project_root: Path):
    """Return the unchanged production manifest verifier.

    The installed package is preferred so the round-trip test exercises the
    same verifier the runtime will use; a bare runner falls back to the
    checkout's source tree.
    """

    try:
        from nba_prop_quant.production import (
            load_verified_manifest_metadata,
        )
    except ImportError:
        source_root = str(project_root / "src")

        if source_root not in sys.path:
            sys.path.insert(0, source_root)

        from nba_prop_quant.production import (
            load_verified_manifest_metadata,
        )

    return load_verified_manifest_metadata


def roundtrip_verify(
    archive_path: Path,
    staging_name: str,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Extract the archive, verify every hash, then run the real verifier."""

    with tempfile.TemporaryDirectory(prefix="nba_runtime_roundtrip_") as tmp:
        temp_root = Path(tmp)

        with tarfile.open(archive_path, mode="r:gz") as tar:
            for member in tar.getmembers():
                target = (temp_root / member.name).resolve()

                if not str(target).startswith(str(temp_root.resolve())):
                    raise BuildError(
                        f"Unsafe archive member path: {member.name}"
                    )

            tar.extractall(temp_root, filter="data")

        extracted = temp_root / staging_name

        if not extracted.is_dir():
            raise BuildError(
                f"Archive did not contain expected root {staging_name}"
            )

        verified_files = verify_sha256sums_tree(extracted)

        reloaded = load_json(extracted / RUNTIME_MANIFEST_NAME)

        if reloaded != manifest:
            raise BuildError(
                "Round-trip manifest does not match the built manifest."
            )

        checks = {
            "model_source_commit": (
                reloaded["model_source_commit"],
                contract["model_source_commit"],
            ),
            "gate3_candidate_policy_id": (
                reloaded["gate3_candidate_policy_id"],
                contract["gate3_candidate_policy_id"],
            ),
            "primary_certification_window": (
                reloaded["primary_certification_window"],
                contract["primary_certification_window"],
            ),
        }

        for name, (observed, expected) in checks.items():
            if observed != expected:
                raise BuildError(
                    f"Round-trip {name} mismatch: "
                    f"{observed!r} != {expected!r}"
                )

        if reloaded["auto_bet"] is not False:
            raise BuildError("Round-trip manifest must keep auto_bet false.")

        if reloaded["prospective_claim_allowed"] is not False:
            raise BuildError(
                "Round-trip manifest must keep "
                "prospective_claim_allowed false."
            )

        if reloaded["gate3_policy"] != contract["gate3_policy"]:
            raise BuildError("Round-trip Gate 3 10-prop policy mismatch.")

        load_verified_manifest_metadata = import_verifier(project_root)

        metadata = load_verified_manifest_metadata(
            model_dir=extracted / "models",
            project_root=extracted,
        )

        if metadata["freeze_id"] != manifest["freeze_id"]:
            raise BuildError(
                "Verifier returned an unexpected freeze_id: "
                f"{metadata['freeze_id']}"
            )

        return {
            "verified_files": verified_files,
            "freeze_id": metadata["freeze_id"],
            "freeze_stage": metadata["freeze_stage"],
            "manifest_sha256": metadata["manifest_sha256"],
        }


def build_runtime_bundle(
    project_root: Path,
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    contract_path: Path,
    gate3_artifact_dir: Path,
    pip_freeze: Path | None = None,
    runtime_version: int | None = None,
    allow_dirty: bool = False,
    keep_staging: bool = False,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    gate3_artifact_dir = gate3_artifact_dir.expanduser().resolve()

    if not contract_path.exists():
        raise BuildError(f"Runtime contract missing: {contract_path}")

    contract = load_json(contract_path)

    provenance = resolve_source_provenance(
        project_root,
        str(contract["model_source_commit"]),
        allow_dirty=allow_dirty,
    )

    frozen_sources = verify_frozen_model_sources(
        project_root,
        str(contract["model_source_commit"]),
        contract["frozen_model_source_files"],
    )

    checksum_groups = {
        relative: verify_sha256sums_group(project_root, relative)
        for relative in contract.get("sha256sums_groups", [])
    }

    gate3 = verify_gate3_policy(
        gate3_artifact_dir,
        dict(contract["gate3_policy"]),
        str(contract["gate3_candidate_policy_id"]),
        str(contract["gate3_policy_lock_commit"]),
        project_root,
    )

    verify_capture_window(
        project_root,
        str(contract["primary_certification_window"]),
        int(contract["primary_certification_offset_minutes"]),
    )

    resolved, absent_optional = collect_bundle_files(
        contract,
        project_root,
        model_dir,
        data_dir,
    )

    enforce_secret_exclusion(resolved, contract)

    created = datetime.now(timezone.utc)

    staging_name = (
        f"{contract['runtime_name']}_"
        + created.strftime("%Y%m%dT%H%M%SZ")
    )

    staging = output_dir / staging_name

    stage_bundle(resolved, staging)

    dependency_lock: dict[str, Any] = {
        "declared_dependencies_path": "pyproject.toml",
        "declared_dependencies_sha256": sha256_file(
            project_root / "pyproject.toml"
        ),
    }

    if pip_freeze is not None:
        lock_path = pip_freeze.expanduser().resolve()

        if not lock_path.exists():
            raise BuildError(f"Dependency lock file missing: {lock_path}")

        dependency_lock["pip_freeze_sha256"] = sha256_file(lock_path)
        dependency_lock["pip_freeze_filename"] = lock_path.name

    manifest = build_manifest(
        contract=contract,
        resolved=resolved,
        staging=staging,
        provenance=provenance,
        gate3=gate3,
        frozen_sources=frozen_sources,
        absent_optional=absent_optional,
        dependency_lock=dependency_lock,
        runtime_version=(
            int(runtime_version)
            if runtime_version is not None
            else int(contract["runtime_version"])
        ),
        created=created,
    )

    manifest_targets = write_manifest_files(staging, contract, manifest)

    sums_path = write_sha256sums(staging)

    archive_path = create_archive(
        staging,
        output_dir / f"{staging_name}.tar.gz",
    )

    roundtrip = roundtrip_verify(
        archive_path,
        staging_name,
        contract,
        manifest,
        project_root,
    )

    if not keep_staging:
        shutil.rmtree(staging)

    return {
        "archive_path": archive_path,
        "archive_sha256": sha256_file(archive_path),
        "checksum_groups": checksum_groups,
        "contract_path": contract_path,
        "manifest": manifest,
        "manifest_targets": manifest_targets,
        "resolved": resolved,
        "roundtrip": roundtrip,
        "staging_name": staging_name,
        "staging_path": staging if keep_staging else None,
        "sha256sums_name": sums_path.name,
    }


def print_summary(result: dict[str, Any]) -> None:
    manifest = result["manifest"]
    rule = "=" * 78

    print(rule)
    print("NBA PROP QUANT v2 / GATE 3 PRODUCTION RUNTIME BUNDLE")
    print(rule)
    print(f"Runtime ID              : {manifest['runtime_id']}")
    print(f"Freeze stage            : {manifest['freeze_stage']}")
    print(f"Model source commit     : {manifest['model_source_commit']}")
    print(
        "Production source commit: "
        f"{manifest['production_source_commit']}"
    )
    print(
        "Gate 3 candidate        : "
        f"{manifest['gate3_candidate_policy_id']}"
    )
    print(
        "Certification window    : "
        f"{manifest['primary_certification_window']}"
    )
    print(f"Prospective claim       : {manifest['prospective_claim_allowed']}")
    print(f"auto_bet                : {manifest['auto_bet']}")
    print()

    print("Bundled groups:")

    for group_name in sorted(manifest["files"]):
        records = manifest["files"][group_name]
        size = sum(record["bytes"] for record in records)
        print(f"  {group_name:30s} {len(records):5d} files  {size:>14,d} B")

    print()
    print("Frozen mathematical source verified against "
          f"{manifest['model_source_commit'][:12]}:")

    for relative in manifest["frozen_model_source_verification"]["files"]:
        print(f"  OK  {relative}")

    print()
    print("Checksum groups verified:")

    for relative, count in sorted(result["checksum_groups"].items()):
        print(f"  OK  {relative} ({count} files)")

    if manifest["absent_optional_files"]:
        print()
        print("Optional resources absent from this build:")

        for entry in manifest["absent_optional_files"]:
            print(f"  --  {entry}")

    print()
    print(
        f"Bundle files            : {manifest['bundle']['file_count']:,} "
        f"({manifest['bundle']['total_bytes']:,} bytes)"
    )
    print(f"Archive                 : {result['archive_path']}")
    print(f"Archive SHA-256         : {result['archive_sha256']}")
    print(
        "Round-trip verified     : "
        f"{result['roundtrip']['verified_files']:,} files"
    )
    print(
        "Verifier freeze_id      : "
        f"{result['roundtrip']['freeze_id']}"
    )
    print()
    print("PASS: runtime bundle built, hashed and round-trip verified.")
    print("No betting threshold is active. auto_bet remains false.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    project_root = args.project_root.expanduser().resolve()

    model_dir = (
        args.model_dir
        if args.model_dir is not None
        else project_root / "models"
    )

    data_dir = (
        args.data_dir
        if args.data_dir is not None
        else project_root / "data"
    )

    contract_path = (
        args.contract
        if args.contract is not None
        else project_root / DEFAULT_CONTRACT
    )

    gate3_artifact_dir = (
        args.gate3_artifact_dir
        if args.gate3_artifact_dir is not None
        else project_root / "research/v2_gate3_deployment_artifacts"
    )

    try:
        result = build_runtime_bundle(
            project_root=project_root,
            model_dir=model_dir,
            data_dir=data_dir,
            output_dir=args.output_dir,
            contract_path=contract_path,
            gate3_artifact_dir=gate3_artifact_dir,
            pip_freeze=args.pip_freeze,
            runtime_version=args.runtime_version,
            allow_dirty=args.allow_dirty,
            keep_staging=args.keep_staging,
        )
    except BuildError as exc:
        print("FAIL: runtime bundle not built.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    print_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
