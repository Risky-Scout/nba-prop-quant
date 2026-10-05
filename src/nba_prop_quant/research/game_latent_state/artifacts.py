"""Versioned, hashed research artifacts for the shadow latent-state layer.

SHADOW / RESEARCH ONLY.

Every artifact records the source production SHA, the training cutoff, the
seasons used, the data fingerprints, the seed, the code SHA and the hash of
every file written, so a reviewer can tell exactly which bytes produced which
number. Nothing here writes to ``models/``, ``promotion_state.json`` or
``current_good_fit_id``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import DEPENDENCE_MODEL_VERSION

ARTIFACT_SCHEMA_VERSION = 1

# Recorded in every manifest. The shadow branch is not a promotion path, and
# the manifest says so explicitly so no downstream tool can mistake a shadow
# artifact for a registered production fit.
PROMOTION_ELIGIBILITY = "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def git_sha(project_root: Path, ref: str = "HEAD") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", ref],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def git_branch(project_root: Path) -> str | None:
    """Current branch name, or ``None`` on a detached head or outside a repo."""
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def directory_fingerprint(root: Path, pattern: str = "**/*.parquet") -> dict[str, str]:
    """Per-file digests for an input tree, so data drift is detectable."""
    root = Path(root)
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.glob(pattern))
        if path.is_file()
    }


@dataclass
class ArtifactManifest:
    artifact_name: str
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    dependence_model_version: str = DEPENDENCE_MODEL_VERSION
    promotion_eligibility: str = PROMOTION_ELIGIBILITY
    source_production_sha: str | None = None
    source_production_ref: str | None = None
    code_sha: str | None = None
    branch: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    seed: int | None = None
    training_cutoff: str | None = None
    seasons_used: list[int] = field(default_factory=list)
    training_seasons: list[int] = field(default_factory=list)
    validation_seasons: list[int] = field(default_factory=list)
    input_fingerprints: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def write_json(payload: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def write_checksums(paths: Sequence[Path], destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in sorted(Path(entry) for entry in paths):
        if not path.exists() or path.resolve() == destination.resolve():
            continue
        lines.append(f"{sha256_file(path)}  {path.name}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def finalize_manifest(
    manifest: ArtifactManifest,
    artifact_dir: Path,
    outputs: Mapping[str, Path],
    manifest_name: str = "manifest.json",
    checksum_name: str = "SHA256SUMS.txt",
) -> dict[str, Any]:
    """Hash every output, write the manifest, then checksum the whole set."""
    artifact_dir = Path(artifact_dir)
    manifest.outputs = {
        name: sha256_file(path) for name, path in sorted(outputs.items()) if Path(path).exists()
    }
    payload = manifest.to_payload()
    manifest_path = write_json(payload, artifact_dir / manifest_name)
    write_checksums(
        [*[Path(path) for path in outputs.values()], manifest_path],
        artifact_dir / checksum_name,
    )
    return payload
