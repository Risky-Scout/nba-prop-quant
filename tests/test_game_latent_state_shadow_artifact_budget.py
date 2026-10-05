"""The research branches may not carry a large artifact into production history.

The latent-state research reads a 66 MB out-of-fold residual dataset and
writes hundreds of megabytes of fitted-marginal caches. None of it belongs in
the repository: production history would inherit every byte forever, and a
consolidation branch that quietly picked one up would be very hard to undo.

These tests are the guard, and they run in CI because CI runs the test suite.
They read the git index rather than the working tree, so a file that exists
locally and is correctly untracked passes, while a file that has been staged
or committed fails no matter how it got there.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

#: No tracked blob may exceed this. The largest legitimate artifact in the
#: repository is a 3.3 MB constant inventory, so this leaves real headroom
#: while still being an order of magnitude below the research residual
#: dataset the shadow work reads.
MAX_TRACKED_BLOB_BYTES = 10 * 1024 * 1024

#: Research artifacts are committed deliberately -- the factor loadings, the
#: graded joint events -- but they are small summaries, not datasets. A
#: research parquet larger than this is a dataset that escaped.
MAX_RESEARCH_PARQUET_BYTES = 2 * 1024 * 1024

#: Datasets and caches the research drivers produce. Never tracked, at any
#: size: a 1 KB pickle of fitted marginals is still a cache.
NEVER_TRACKED_NAMES = ("oof_gaussian_residuals.parquet",)
NEVER_TRACKED_SUFFIXES = (".pkl", ".pickle", ".npy", ".npz")

RESEARCH_PREFIX = "research/"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def tracked_blobs() -> list[tuple[str, int]]:
    """``(path, bytes)`` for every blob in the current commit.

    ``git ls-tree -l`` reports the size of the blob as committed, which is the
    number that matters: a file's working-tree size can differ from what the
    repository is carrying, and it is the latter that production history would
    inherit. Submodule and symlink entries have no blob size and are skipped.
    """
    out: list[tuple[str, int]] = []
    for line in git("ls-tree", "-r", "-l", "HEAD").splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) < 4 or fields[1] != "blob" or fields[3] == "-":
            continue
        out.append((path, int(fields[3])))
    return out


def test_no_tracked_blob_exceeds_the_repository_budget():
    oversized = [
        (path, size)
        for path, size in tracked_blobs()
        if size > MAX_TRACKED_BLOB_BYTES
    ]
    assert oversized == [], (
        "tracked blobs over the "
        f"{MAX_TRACKED_BLOB_BYTES // (1024 * 1024)} MB budget: {oversized}"
    )


def test_no_research_dataset_or_cache_is_tracked():
    offenders = [
        path
        for path, _ in tracked_blobs()
        if Path(path).name in NEVER_TRACKED_NAMES
        or Path(path).suffix in NEVER_TRACKED_SUFFIXES
    ]
    assert offenders == [], f"research datasets or caches are tracked: {offenders}"


def test_committed_research_parquets_are_summaries_not_datasets():
    oversized = [
        (path, size)
        for path, size in tracked_blobs()
        if path.startswith(RESEARCH_PREFIX)
        and path.endswith(".parquet")
        and size > MAX_RESEARCH_PARQUET_BYTES
    ]
    assert oversized == [], f"research parquets over budget: {oversized}"


def test_the_research_residual_dataset_is_ignored_rather_than_merely_absent():
    """Being untracked by luck is not the same as being untracked by rule.

    ``git check-ignore`` answers whether the path would be refused if someone
    ran ``git add`` on it, which is the property that actually protects the
    merge path.
    """
    for relative in (
        "research/game_latent_state/oof_gaussian_residuals.parquet",
        "research/game_latent_state_v2/oof_gaussian_residuals.parquet",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=str(PROJECT),
            capture_output=True,
        )
        assert result.returncode == 0, f"{relative} is not ignored"
