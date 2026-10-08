"""Containment contract for the shadow latent-state branch.

SHADOW / RESEARCH ONLY.

This module is the single declaration of which repository paths decide
production behaviour. It only ever *reads* git: nothing here writes, and the
path lists exist so that both the validation run and the branch-safety tests
check containment against the same definition rather than two drifting copies.

Gate H is this file plus :func:`modified_production_paths` returning an empty
list.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

PRODUCTION_REF = "production/wizardofodds-integration"

#: Directory prefixes whose bytes decide production behaviour: the scheduled
#: automation, the frozen configs and model artifacts, the operational
#: runbooks, and the production scripts the pipeline executes.
PROTECTED_PRODUCTION_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "configs/",
    "models/",
    "ops/",
    "scripts/",
    "docs/",
    "release/",
    "review/",
)

#: Production modules the shadow layer imports and reads but must never edit.
#: The dependence and marginal modules are the load-bearing ones: the whole
#: no-double-count argument rests on ``copula.py`` being untouched.
PROTECTED_PRODUCTION_SOURCES: frozenset[str] = frozenset(
    {
        "src/nba_prop_quant/adaptive_fit_registry.py",
        "src/nba_prop_quant/adaptive_training.py",
        "src/nba_prop_quant/copula.py",
        "src/nba_prop_quant/distributions.py",
        "src/nba_prop_quant/features.py",
        "src/nba_prop_quant/gate3_v2.py",
        "src/nba_prop_quant/model.py",
        "src/nba_prop_quant/pipeline.py",
        "src/nba_prop_quant/pricing.py",
        "src/nba_prop_quant/production.py",
        "src/nba_prop_quant/slate.py",
    }
)


#: Protected paths this integration is declared to change, each with its
#: reason. Gate H's absolute form — no protected path changes at all — held for
#: as long as the dependence work was pure research, and it is the right shape
#: for a research branch. Integrating the work into production is a production
#: change by construction, so the enforced invariant becomes "nothing
#: *undeclared* changes" and the declaration lives here, next to the path lists
#: it qualifies, rather than being implicit in whatever a diff happens to show.
#:
#: Nothing here may be a model artifact, a pricing path or a publishing path.
#: The branch-safety tests pin that, so widening this mapping cannot quietly
#: become permission to edit the served model.
#:
#: A declaration is permission for a *pending* change, so it is removed once
#: that change has landed in production; otherwise it sits here as standing
#: permission to edit a protected path unreviewed. What the lineage owns in
#: production once a change has landed is recorded in
#: :data:`SHADOW_OWNED_PRODUCTION_PATHS`, which grants no permission at all.
DECLARED_INTEGRATION_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "ops/evidence/production_shadow_closure.json": (
            "the production-ops closure receipt: the reconciled test counts, "
            "the production SHA both hashes were verified at, and why one "
            "commit reported four different pass/skip splits. Evidence only: "
            "nothing reads it and it decides no production behaviour"
        ),
    }
)

#: Protected paths this lineage introduced into production and now owns.
#:
#: This is a registry, not permission. It exists because two different
#: questions were previously answered by one mapping: "may this branch change
#: this protected path" and "is this protected path part of the shadow's
#: surface". Those coincided only on the branch that first deployed the
#: shadow. Once that branch merged, every entry became permanently stale under
#: the first question while remaining permanently true under the second, which
#: made the production branch impossible to extend without either leaving loose
#: permission behind or dropping the surface from the record.
#:
#: Separating them keeps both guarantees and is strictly tighter than the
#: single mapping was: changing anything here still requires a fresh entry in
#: :data:`DECLARED_INTEGRATION_PATHS` and the review that comes with it, so
#: being owned buys nothing. The same restrictions apply — no model artifact,
#: no config, no production script, no release or review path, no protected
#: production source module — and the branch-safety tests pin that for this
#: mapping exactly as they do for the declaration.
SHADOW_OWNED_PRODUCTION_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "ops/run_production_shadow.py": (
            "the shadow's production entry point. Reads the marginals and the "
            "copula the live pricing path reads, records what the candidate "
            "would have said, and publishes nothing. It is additive: no "
            "existing production script calls it and it writes no production "
            "state"
        ),
        ".github/workflows/nba_production_lifecycle.yml": (
            "one added non-blocking step that invokes the shadow entry point "
            "after the daily fit. Additive only: every pre-existing step is "
            "unchanged, and the step cannot fail the lifecycle"
        ),
    }
)

#: Workflows whose pre-existing steps must survive a change intact. Owning or
#: declaring a workflow buys the right to *add* a step, not to rewrite the
#: lifecycle: the modelling, refresh and registration steps are the production
#: path, and the branch-safety tests check them line by line against the
#: production ref rather than trusting either mapping.
ADDITIVE_ONLY_WORKFLOWS: tuple[str, ...] = (
    ".github/workflows/nba_production_lifecycle.yml",
)


def _git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def production_merge_base(project_root: Path) -> str | None:
    """The commit this branch diverged from production at, or ``None``."""
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return _git(project_root, "merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue
    return None


def changed_paths(project_root: Path) -> list[str]:
    base = production_merge_base(project_root)
    if base is None:
        return []
    return [
        line
        for line in _git(project_root, "diff", "--name-only", base, "HEAD").splitlines()
        if line
    ]


def modified_production_paths(project_root: Path) -> list[str]:
    """Protected paths this branch has modified, declared or not.

    Reported as-is. Gate H reads this, so a branch that changes a protected
    path is still visibly a branch that changed production, even when the
    change is a declared one.
    """
    return sorted(
        path
        for path in changed_paths(project_root)
        if path.startswith(PROTECTED_PRODUCTION_PREFIXES)
        or path in PROTECTED_PRODUCTION_SOURCES
    )


def undeclared_production_paths(project_root: Path) -> list[str]:
    """Protected paths this branch changed without declaring them.

    This is the invariant the branch-safety tests enforce. Empty means every
    production path this branch touches is named in
    :data:`DECLARED_INTEGRATION_PATHS` with a reason.
    """
    return sorted(
        set(modified_production_paths(project_root)) - set(DECLARED_INTEGRATION_PATHS)
    )


def stale_integration_declarations(project_root: Path) -> list[str]:
    """Declared paths this branch does not actually change.

    A declaration that has outlived its change is permission left lying
    around, so it is reported rather than ignored.
    """
    touched = set(modified_production_paths(project_root))
    return sorted(set(DECLARED_INTEGRATION_PATHS) - touched)


# ---------------------------------------------------------------------
# What a merge would add to production history, as opposed to its tree
# ---------------------------------------------------------------------

#: No new object in the merge path may exceed this.
MAX_MERGE_PATH_BLOB_BYTES = 10 * 1024 * 1024

#: Reported, not failed. Worth a reviewer's attention before it becomes a
#: habit; the repository's largest legitimate blob is a 3.3 MB inventory.
NOTABLE_MERGE_PATH_BLOB_BYTES = 5 * 1024 * 1024

#: Generated datasets and caches. Never acceptable at any size: a 1 KB pickle
#: of fitted marginals is still a cache, and a small parquet is still a
#: regenerated export that nothing reads.
GENERATED_ARTIFACT_NAMES: frozenset[str] = frozenset(
    {
        "oof_gaussian_residuals.parquet",
        "factor_loadings.parquet",
        "joint_event_grades.parquet",
    }
)
GENERATED_ARTIFACT_SUFFIXES: tuple[str, ...] = (
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".joblib",
    ".h5",
    ".onnx",
    ".pt",
    ".pth",
)


def tree_blobs(project_root: Path, ref: str = "HEAD") -> dict[str, int]:
    """``{path: bytes}`` for every blob in one commit's tree."""
    out: dict[str, int] = {}
    for line in _git(project_root, "ls-tree", "-r", "-l", ref).splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) < 4 or fields[1] != "blob" or fields[3] == "-":
            continue
        out[path] = int(fields[3])
    return out


def merge_path_blobs(project_root: Path, base: str | None = None) -> list[dict]:
    """Every blob a merge into production would add, with size and path.

    This is the question a tree listing cannot answer. ``git ls-tree`` reports
    what a branch *has*; a merge transfers what the branch's history
    *reaches*. A blob that was committed and later deleted is absent from
    every subsequent tree and still travels with an ordinary merge commit,
    because the commits that contain it come along too. The latent-state
    research branches are a live example: a 65 MB residual parquet was
    committed once and removed two days later, so it is invisible to
    ``git ls-tree HEAD`` and reachable from all three of their heads.

    Entries are sorted largest first. A blob may appear under more than one
    path across history; each (object, path) pair is reported once.
    """
    base = base or production_merge_base(project_root)
    if base is None:
        return []
    listing = _git(project_root, "rev-list", f"{base}..HEAD", "--objects")
    candidates: dict[tuple[str, str], None] = {}
    for line in listing.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            candidates[(sha, path)] = None
    if not candidates:
        return []

    probe = "\n".join(sha for sha, _ in candidates) + "\n"
    completed = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        cwd=str(project_root),
        input=probe,
        capture_output=True,
        text=True,
        check=True,
    )
    sizes: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] == "blob":
            sizes[fields[0]] = int(fields[2])

    out = [
        {"sha": sha, "path": path, "bytes": sizes[sha]}
        for sha, path in candidates
        if sha in sizes
    ]
    out.sort(key=lambda entry: (-entry["bytes"], entry["path"]))
    return out


def is_generated_artifact(path: str) -> bool:
    name = Path(path).name
    return name in GENERATED_ARTIFACT_NAMES or Path(
        path
    ).suffix in GENERATED_ARTIFACT_SUFFIXES


#: CI checks out a pull request, not a branch, so the base it should audit
#: against is the PR's target. GitHub puts that in ``GITHUB_BASE_REF``.
BASE_REF_ENV_VARS: tuple[str, ...] = ("PRODUCTION_BASE_REF", "GITHUB_BASE_REF")


def resolve_audit_base(project_root: Path) -> str | None:
    """The commit to audit against, preferring an explicitly configured base.

    Falls back to :func:`production_merge_base`. Returns ``None`` only when
    nothing resolves, which callers must treat as a failure rather than as an
    empty merge path: a guard that cannot see the base has not checked
    anything, and reporting that as a pass is how a 65 MB blob gets in.
    """
    for variable in BASE_REF_ENV_VARS:
        ref = os.environ.get(variable, "").strip()
        if not ref:
            continue
        for candidate in (ref, f"origin/{ref}"):
            try:
                return _git(project_root, "merge-base", "HEAD", candidate)
            except subprocess.CalledProcessError:
                continue
    return production_merge_base(project_root)


def audit_merge_path(project_root: Path, base: str | None = None) -> dict:
    """Classify the merge path's new objects against the blob contract.

    ``passed`` is false when a new object breaches the size ceiling, is a
    generated dataset, cache or model binary, or when the base could not be
    resolved at all. ``notable`` is informational.
    """
    resolved_base = base or resolve_audit_base(project_root)
    blobs = merge_path_blobs(project_root, resolved_base)
    oversized = [entry for entry in blobs if entry["bytes"] > MAX_MERGE_PATH_BLOB_BYTES]
    notable = [
        entry
        for entry in blobs
        if entry["bytes"] > NOTABLE_MERGE_PATH_BLOB_BYTES
        and entry["bytes"] <= MAX_MERGE_PATH_BLOB_BYTES
    ]
    generated = [entry for entry in blobs if is_generated_artifact(entry["path"])]
    return {
        "base": resolved_base,
        "base_resolved": resolved_base is not None,
        "new_blob_count": len(blobs),
        "largest": blobs[0] if blobs else None,
        "over_ceiling": oversized,
        "notable": notable,
        "generated_artifacts": generated,
        "ceiling_bytes": MAX_MERGE_PATH_BLOB_BYTES,
        "notable_bytes": NOTABLE_MERGE_PATH_BLOB_BYTES,
        "passed": bool(resolved_base) and not oversized and not generated,
    }
