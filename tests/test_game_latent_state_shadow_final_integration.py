"""The final clean integration contract, checked live rather than trusted.

``research/final_integration/integration_verification.json`` is a snapshot. It
was written by a run against the commit *before* the commit that carries it,
because nothing can hash the commit it is part of. A snapshot is useful as a
record and useless as a guarantee: it cannot notice a later commit that adds a
200 MB parquet or deletes a carried file.

So the contract is re-derived here, against the real head, every time the
suite runs. CI runs the suite on every production pull request, which makes
this the enforcement point and the JSON the receipt.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

INTEGRATION_DIR = PROJECT / "research" / "final_integration"

MANIFEST_PATH = INTEGRATION_DIR / "carry_manifest.json"
SNAPSHOT_PATH = INTEGRATION_DIR / "integration_verification.json"

#: Studies that were closed without adopting a change. Their reports are
#: carried; their drivers are not.
CLOSED_STUDY_REPORTS = (
    "research/count_space_forensic/count_space_forensic.json",
    "research/marginal_convention_audit/marginal_convention_audit.json",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verifier():
    """Import the verifier by path; it is a numbered script, not a module."""
    path = INTEGRATION_DIR / "00_verify_integration.py"
    spec = importlib.util.spec_from_file_location("final_integration_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load(MANIFEST_PATH)


# ----------------------------------------------------------------------
# the four checks, live
# ----------------------------------------------------------------------


def test_the_tree_still_matches_the_carry_manifest(verifier, manifest):
    result = verifier.check_manifest_contents(manifest)
    assert result["carried_paths_not_tracked"] == []
    assert result["excluded_paths_tracked"] == []
    assert result["excluded_paths_on_disk_but_not_ignored"] == []
    assert result["passed"] is True


def test_the_merge_path_still_satisfies_the_blob_contract(verifier, manifest):
    result = verifier.check_merge_path(manifest["production_base"])
    assert result["base_resolved"] is True
    assert result["new_blob_count"] > 0, (
        "an empty merge path against the recorded production base means the "
        "base is wrong, not that the branch is clean"
    )
    assert result["over_ceiling"] == []
    assert result["generated_artifacts_by_name"] == []
    assert result["generated_artifacts_by_suffix"] == []
    assert result["passed"] is True


def test_the_blob_sweep_covers_parquet_by_suffix_not_only_by_name(verifier):
    """The named list cannot catch a parquet nobody thought to name."""
    assert ".parquet" in verifier.GENERATED_SUFFIXES
    for suffix in (".pkl", ".joblib", ".npy", ".pt"):
        assert suffix in verifier.GENERATED_SUFFIXES


def test_every_carried_checksum_file_still_verifies(verifier):
    result = verifier.check_checksum_files()
    assert result["files_checked"] > 0
    for entry in result["files"]:
        assert entry["mismatched"] == [], entry["path"]
        assert entry["absent"] == [], entry["path"]
    assert result["lines_verified"] > 0
    assert result["passed"] is True


def test_a_checksum_line_for_a_regenerated_export_is_classified_not_ignored(
    verifier,
):
    """Those lines exist to let a rebuild be checked; they are not failures.

    They are also not skipped silently. If every line in a checksum file were
    one of these, ``entries`` would be positive and ``checked`` zero, which the
    report shows rather than hides.
    """
    result = verifier.check_checksum_files()
    recorded = [
        name
        for entry in result["files"]
        for name in entry["recorded_only_generated_artifacts"]
    ]
    assert recorded, "the carried checksum files should record regenerated exports"
    for name in recorded:
        assert Path(name).suffix in verifier.GENERATED_SUFFIXES
    assert result["lines_recording_a_regenerated_artifact"] == len(recorded)


def test_every_provenance_hash_in_the_final_spec_still_resolves(verifier):
    result = verifier.check_provenance_hashes()
    assert result["artifacts_unresolved"] == []
    assert result["artifacts_resolved"] > 0
    assert result["factor_spec_hash_is_the_authoritative_one"] is True
    assert result["passed"] is True


# ----------------------------------------------------------------------
# no research history was merged
# ----------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def resolves(rev: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "-q", "--verify", rev],
            cwd=str(PROJECT),
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=str(PROJECT),
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def superseded_research_heads(manifest) -> dict[str, str]:
    """The superseded research branch tips that this checkout can resolve."""
    found = {}
    for ref in manifest["superseded_research_branches"]:
        for candidate in (f"origin/{ref}", ref):
            if resolves(candidate):
                found[ref] = candidate
                break
    return found


def test_no_superseded_research_history_is_reachable(manifest):
    """None of the superseded research tips is an ancestor of this head.

    This is the structural statement behind "do not merge the research PRs",
    and it is the one worth making. Counting merge commits was a proxy, and a
    wrong one: GitHub builds a synthetic merge of the head into the base to
    test a pull request, and merging this branch into production is itself a
    merge commit. Both are joins between this lineage and production, which is
    the whole point of the branch. What must never happen is a *third* lineage
    becoming reachable, and that is checkable directly.
    """
    heads = superseded_research_heads(manifest)
    if not heads:
        pytest.skip("no superseded research ref is available in this checkout")
    reachable = sorted(ref for ref, rev in heads.items() if is_ancestor(rev, "HEAD"))
    assert reachable == [], f"superseded research history is reachable: {reachable}"


def test_every_merge_in_this_lineage_only_joins_production(manifest):
    """A merge commit here may join production, and nothing else.

    Complements the reachability check above by constraining shape as well as
    content: every merge after the declared base must have a parent that is
    production history, so the only joins possible are production-into-lineage
    and lineage-into-production.
    """
    base = manifest["production_base"]
    merges = [line for line in git("rev-list", "--merges", f"{base}..HEAD").splitlines()]
    for merge in merges:
        parents = git("rev-list", "--parents", "-n", "1", merge).split()[1:]
        joins_production = [
            parent for parent in parents if is_ancestor(parent, base) or parent == base
        ]
        assert joins_production, (
            f"merge {merge} has no production parent, so it joins a lineage "
            f"other than production: parents {parents}"
        )


def test_the_production_base_is_an_ancestor_of_this_branch(manifest):
    base = manifest["production_base"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"],
        cwd=str(PROJECT),
        check=True,
    )


def test_the_declared_base_actually_has_a_merge_path_to_audit(manifest, verifier):
    """A manifest naming the wrong base would audit an empty merge path.

    Stated as the property that matters rather than as an equality against the
    live merge base, because once this branch is merged the live merge base is
    HEAD and the live merge path is empty by definition. The declared base is
    what keeps the blob contract enforceable on the production branch too.
    """
    base = manifest["production_base"]
    assert is_ancestor(base, "HEAD")
    assert git("rev-parse", base) != git("rev-parse", "HEAD")
    assert verifier.check_merge_path(base)["new_blob_count"] > 0


def test_the_declared_base_is_where_this_lineage_left_production(manifest):
    """While this lineage is still unmerged, the live merge base must agree.

    After the merge the live merge base is HEAD, which agrees with nothing and
    means only that there is no longer a merge pending; the check above is the
    one that still has teeth then.
    """
    from nba_prop_quant.research.game_latent_state.safety import (
        production_merge_base,
    )

    resolved = production_merge_base(PROJECT)
    if resolved is None:
        pytest.skip("the production ref is not available in this checkout")
    if resolved == git("rev-parse", "HEAD"):
        pytest.skip("this head is production, so there is no pending merge")
    assert resolved == manifest["production_base"]


# ----------------------------------------------------------------------
# the record itself
# ----------------------------------------------------------------------


def test_the_snapshot_records_a_commit_in_this_lineage(manifest):
    """The receipt must be from this branch, not from somewhere else."""
    snapshot = load(SNAPSHOT_PATH)
    assert snapshot["passed"] is True
    assert snapshot["production_base"] == manifest["production_base"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", snapshot["head_sha"], "HEAD"],
        cwd=str(PROJECT),
        check=True,
    )


def test_the_manifest_names_the_superseded_research_pull_requests(manifest):
    assert manifest["superseded_research_pull_requests"] == [16, 17, 18, 20, 21, 22]
    assert "histories not merged" in manifest["superseded_handling"]


def test_the_closed_studies_are_carried_as_evidence_without_their_drivers(
    manifest,
):
    """Both halves of the disposition, checked together.

    The reports have to be present -- otherwise the conclusions the final spec
    carries forward are unattributable -- and the drivers have to be absent,
    because a production branch should not carry the machinery of a study that
    changed nothing.
    """
    tracked = set(git("ls-files").splitlines())
    for report in CLOSED_STUDY_REPORTS:
        assert report in tracked, report

    excluded = manifest["exclude"]["forensic_experimentation_drivers"]
    for path in excluded["paths"]:
        assert path not in tracked, path
    for path in excluded["evidence_preserved_as"]:
        assert path in tracked, path


def test_the_closed_study_reports_are_hashed_by_the_final_model_spec():
    """So the evidence cannot be swapped after the drivers are gone."""
    import hashlib

    spec = load(PROJECT / "research/final_model/final_model_spec.json")
    recorded = {
        entry["path"]: entry["sha256"]
        for entry in spec["source_artifacts"].values()
        if entry.get("path")
    }
    for report in CLOSED_STUDY_REPORTS:
        assert report in recorded, report
        actual = hashlib.sha256((PROJECT / report).read_bytes()).hexdigest()
        assert actual == recorded[report], report


def test_the_excluded_runtime_module_is_really_gone():
    """``censored.py`` existed only for the closed forensic study."""
    assert not (
        PROJECT / "src/nba_prop_quant/research/game_latent_state/censored.py"
    ).exists()
    with pytest.raises(ImportError):
        importlib.import_module(
            "nba_prop_quant.research.game_latent_state.censored"
        )


def test_nothing_carried_still_imports_the_excluded_module():
    """Scanned as imports, not as text.

    A substring scan would flag this file, which names the module in order to
    assert its absence. The import statements are what matters.
    """
    import ast

    offenders: list[str] = []
    for path in [
        *(PROJECT / "src/nba_prop_quant/research").rglob("*.py"),
        *(PROJECT / "research").rglob("*.py"),
        *(PROJECT / "tests").glob("test_game_latent_state_shadow*.py"),
    ]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            if any(name.split(".")[-1] == "censored" for name in names):
                offenders.append(path.relative_to(PROJECT).as_posix())
    assert sorted(set(offenders)) == []
