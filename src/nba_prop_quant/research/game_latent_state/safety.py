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

import subprocess
from pathlib import Path

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
    """Protected paths this branch has modified. Empty means gate H passes."""
    return sorted(
        path
        for path in changed_paths(project_root)
        if path.startswith(PROTECTED_PRODUCTION_PREFIXES)
        or path in PROTECTED_PRODUCTION_SOURCES
    )
