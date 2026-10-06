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

import pytest

from nba_prop_quant.research.game_latent_state.safety import (
    MAX_MERGE_PATH_BLOB_BYTES,
    audit_merge_path,
    merge_path_blobs,
    resolve_audit_base,
    tree_blobs,
)

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
        "research/game_latent_state_bucket_repair/oof_gaussian_residuals.parquet",
        "research/game_latent_state_v2/oof_gaussian_residuals.parquet",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=str(PROJECT),
            capture_output=True,
        )
        assert result.returncode == 0, f"{relative} is not ignored"


def test_the_regenerated_parquet_exports_are_ignored_too():
    """Nothing reads them, so nothing should be able to commit them.

    ``factor_spec.json`` carries the full loadings payload and the validation
    report carries every summary computed from the grades, so both parquet
    exports are derived files with no reader. They are identified by hash in
    the ``SHA256SUMS`` files instead.
    """
    for directory in (
        "research/game_latent_state",
        "research/game_latent_state_bucket_repair",
    ):
        for name in ("factor_loadings.parquet", "joint_event_grades.parquet"):
            result = subprocess.run(
                ["git", "check-ignore", "-q", f"{directory}/{name}"],
                cwd=str(PROJECT),
                capture_output=True,
            )
            assert result.returncode == 0, f"{directory}/{name} is not ignored"


# ---------------------------------------------------------------------
# The merge path, which is not the same thing as the tree
# ---------------------------------------------------------------------


def test_the_merge_path_audit_has_a_base_to_audit_against():
    """A guard that cannot see its base has not checked anything.

    Reported as a failure rather than skipped, because silently passing is the
    failure mode this whole file exists to prevent.
    """
    assert resolve_audit_base(PROJECT) is not None, (
        "no production base resolved: fetch the production branch or set "
        "PRODUCTION_BASE_REF"
    )


def test_no_new_object_in_the_merge_path_breaches_the_blob_contract():
    report = audit_merge_path(PROJECT)
    assert report["over_ceiling"] == [], (
        "new objects over the "
        f"{MAX_MERGE_PATH_BLOB_BYTES // (1024 * 1024)} MB ceiling would enter "
        f"production history: {report['over_ceiling']}"
    )
    assert report["generated_artifacts"] == [], (
        "generated datasets, caches or model binaries would enter production "
        f"history: {report['generated_artifacts']}"
    )
    assert report["passed"]


def commit_in(root: Path, *paths: str) -> None:
    for args in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "guard@test"],
        ["git", "config", "user.name", "guard"],
    ):
        if args[1] == "init" and (root / ".git").exists():
            continue
        subprocess.run(args, cwd=str(root), check=True, capture_output=True)
    subprocess.run(["git", "add", *paths], cwd=str(root), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "step"], cwd=str(root), check=True, capture_output=True
    )


def test_the_audit_sees_a_blob_that_was_added_and_later_deleted(tmp_path):
    """The property the tree guard cannot have, stated as a test.

    A file committed and then removed is absent from the final tree and still
    reachable from it, so an ordinary merge carries the object. This is not
    hypothetical: it is what happened to the 65 MB residual parquet on the
    research branches, and it is why those branches must not be merged
    directly.
    """
    (tmp_path / "keep.txt").write_text("baseline\n", encoding="utf-8")
    commit_in(tmp_path, "keep.txt")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    big = tmp_path / "research" / "oof_gaussian_residuals.parquet"
    big.parent.mkdir(parents=True, exist_ok=True)
    big.write_bytes(b"\0" * (MAX_MERGE_PATH_BLOB_BYTES + 1))
    commit_in(tmp_path, "research/oof_gaussian_residuals.parquet")

    big.unlink()
    subprocess.run(
        ["git", "rm", "-q", "--cached", "research/oof_gaussian_residuals.parquet"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "remove it again"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    # The tree no longer mentions it, which is exactly why a tree-only guard
    # would report this branch as clean.
    assert not any("oof_gaussian_residuals" in path for path in tree_blobs(tmp_path))

    reachable = merge_path_blobs(tmp_path, base=base)
    assert any("oof_gaussian_residuals" in entry["path"] for entry in reachable)

    report = audit_merge_path(tmp_path, base=base)
    assert report["passed"] is False
    assert report["over_ceiling"], "the oversized deleted blob was not reported"
    assert report["generated_artifacts"], "the generated dataset was not reported"


def test_an_unresolvable_base_is_a_failure_not_an_empty_merge_path(tmp_path):
    (tmp_path / "only.txt").write_text("no production ref here\n", encoding="utf-8")
    commit_in(tmp_path, "only.txt")
    report = audit_merge_path(tmp_path)
    assert report["base_resolved"] is False
    assert report["passed"] is False


@pytest.mark.parametrize(
    "name",
    [
        "fitted_marginals.pkl",
        "moments.pickle",
        "loadings.npy",
        "model.joblib",
        "weights.pt",
        "net.onnx",
    ],
)
def test_a_model_binary_or_cache_is_refused_whatever_its_size(tmp_path, name):
    (tmp_path / "keep.txt").write_text("baseline\n", encoding="utf-8")
    commit_in(tmp_path, "keep.txt")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    (tmp_path / name).write_bytes(b"tiny")
    commit_in(tmp_path, name)

    report = audit_merge_path(tmp_path, base=base)
    assert report["over_ceiling"] == [], "a 4-byte file is not an oversize failure"
    assert [entry["path"] for entry in report["generated_artifacts"]] == [name]
    assert report["passed"] is False
