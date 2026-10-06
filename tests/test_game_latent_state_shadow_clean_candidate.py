"""The clean candidate ships the accepted repair and nothing the research rejected.

This branch exists because three research rounds produced one accepted model
and five rejected components, and the only safe way to get the first without
the second was to rebuild from the production base rather than merge. These
tests are what stops the rejected components from reappearing: as code that
could be imported, as a dial that could be switched on, or as a parameter
count that quietly says zero when it is not.

The equivalence proof is regression-tested rather than recomputed. Recomputing
it needs the V2 search modules, which this branch deliberately does not carry;
asserting against the recorded proof is what makes the reuse of the repair's
holdout grades auditable here.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from nba_prop_quant.research.game_latent_state.covariance import SharedFactorLoadings
from nba_prop_quant.research.game_latent_state.safety import (
    MAX_MERGE_PATH_BLOB_BYTES,
    audit_merge_path,
)

PROJECT = Path(__file__).resolve().parents[1]
V1_DIR = PROJECT / "research" / "game_latent_state"
REPAIR_DIR = PROJECT / "research" / "game_latent_state_bucket_repair"
V2_DIR = PROJECT / "research" / "game_latent_state_v2"
RESOLUTION_DIR = PROJECT / "research" / "game_latent_state_resolution"

#: The search-space modules the rejected components live in. Absent from this
#: branch, so the dials cannot be set even by mistake.
REJECTED_MODULES = (
    "nba_prop_quant.research.game_latent_state.v2",
    "nba_prop_quant.research.game_latent_state.temporal",
    "nba_prop_quant.research.game_latent_state.bridge",
    "nba_prop_quant.research.game_latent_state.countspace",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def resolution() -> dict:
    return load(RESOLUTION_DIR / "final_resolution.json")


@pytest.fixture(scope="module")
def repair_spec() -> dict:
    return load(REPAIR_DIR / "factor_spec.json")


# ---------------------------------------------------------------------
# The shipped model is the accepted one
# ---------------------------------------------------------------------


def test_the_shipped_spec_is_the_accepted_repair(resolution, repair_spec):
    final = resolution["winning_dependence_model"]
    assert final["model"] == "accepted bucket repair"
    assert final["k_game"] == 6
    assert final["r_contrast"] == 6
    assert final["factor_spec_hash"] == repair_spec["spec_hash"]


def test_the_shipped_spec_hash_matches_the_run_that_graded_it(repair_spec):
    """A spec whose hash does not match its validation report was not graded."""
    report = load(REPAIR_DIR / "validation_report.json")
    assert report["factor_spec_hash"] == repair_spec["spec_hash"]


def test_the_rejected_dials_are_all_off(resolution):
    final = resolution["winning_dependence_model"]
    assert final["r_symmetric"] == 0
    assert final["role_deviation"] is False
    assert final["bridge_weight"] == 0.0
    assert final["temporal_treatment"] == "T0_pooled_empirical_bayes"
    assert final["predictive_sd_inflation"] is None


def test_the_shipped_loadings_carry_no_rejected_structure(repair_spec):
    """The rejected layers are absent from the payload, not merely zeroed."""
    loadings = SharedFactorLoadings.from_payload(repair_spec["loadings"])
    assert loadings.symmetric is None
    assert loadings.role_deviation is None
    assert dict(loadings.role_offset) == {}


# ---------------------------------------------------------------------
# The rejected components cannot come back
# ---------------------------------------------------------------------


@pytest.mark.parametrize("module", REJECTED_MODULES)
def test_a_rejected_search_module_is_not_on_this_branch(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_no_carried_module_imports_a_rejected_one():
    package = PROJECT / "src" / "nba_prop_quant" / "research" / "game_latent_state"
    rejected_names = tuple(module.rsplit(".", 1)[1] for module in REJECTED_MODULES)
    offenders: list[str] = []
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in rejected_names:
            if f"from .{name} import" in text or f"from {name} import" in text:
                offenders.append(f"{path.name} -> {name}")
    assert offenders == [], f"carried modules import rejected ones: {offenders}"


def test_the_predictive_sd_inflation_has_no_runtime_surface():
    """The rejected uncertainty layer must not be callable from this branch."""
    package = PROJECT / "src" / "nba_prop_quant" / "research" / "game_latent_state"
    offenders = [
        path.name
        for path in sorted(package.glob("*.py"))
        if "pooled_uncertainty_inflation" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"the inflation layer is still reachable in {offenders}"


# ---------------------------------------------------------------------
# The equivalence proof, regression-tested against its record
# ---------------------------------------------------------------------


def test_the_equivalence_proof_is_exact(resolution):
    proof = resolution["equivalence_proof"]
    assert proof["loadings_payload_bitwise_identical"] is True
    assert proof["model_enters_only_through_loadings"] is True
    assert proof["max_abs_correlation_difference"] == 0.0
    assert proof["max_abs_cholesky_difference"] == 0.0
    assert proof["bitwise_identical_everywhere"] is True
    assert proof["games_checked"] == 600


def test_the_equivalence_proof_covers_both_refit_windows(resolution):
    windows = resolution["equivalence_proof"]["by_refit_window"]
    assert sorted(windows) == ["2024", "2025"]
    for window, entry in windows.items():
        assert entry["bitwise_identical_everywhere"] is True, window
        assert entry["games_checked"] == 300, window


def test_reusing_the_repair_grades_is_what_the_proof_licenses(resolution):
    assert resolution["further_holdout_simulation_required"] is False
    assert resolution["reused_holdout_metrics"]["games_simulated"] == 600


def test_the_recorded_equivalence_evidence_still_matches_its_source(resolution):
    """A hash recorded in the resolution must still describe the file."""
    from nba_prop_quant.research.game_latent_state.artifacts import sha256_file

    proof = resolution["equivalence_proof"]
    assert proof["source_sha256"] == sha256_file(
        V2_DIR / "equivalence_and_calibration.json"
    )
    reused = resolution["reused_holdout_metrics"]
    assert reused["source_sha256"] == sha256_file(REPAIR_DIR / "validation_report.json")


@pytest.mark.parametrize(
    "checksum_file",
    sorted(
        path.relative_to(PROJECT).as_posix()
        for directory in (V1_DIR, REPAIR_DIR, V2_DIR, RESOLUTION_DIR)
        for path in directory.glob("SHA256SUMS.*.txt")
    ),
)
def test_every_carried_checksum_file_still_describes_what_it_covers(checksum_file):
    """A checksum file nobody verifies is a comment that looks like a guarantee.

    Entries for the generated parquet exports are expected to name files this
    branch does not carry -- being identified by hash instead of committed is
    the whole point of them -- so those are checked for *being* generated
    artifacts rather than for matching. Everything else must match its bytes.
    """
    from nba_prop_quant.research.game_latent_state.artifacts import sha256_file
    from nba_prop_quant.research.game_latent_state.safety import is_generated_artifact

    path = PROJECT / checksum_file
    entries = [
        line.split(None, 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries, f"{checksum_file} covers nothing"

    for digest, name in entries:
        covered = path.parent / name.strip()
        if not covered.exists():
            assert is_generated_artifact(covered.name), (
                f"{checksum_file} covers {name.strip()}, which is absent and is "
                "not a generated artifact this branch deliberately excludes"
            )
            continue
        assert sha256_file(covered) == digest, (
            f"{checksum_file} is stale for {name.strip()}: regenerate it "
            "alongside the artifact it covers"
        )


# ---------------------------------------------------------------------
# The predictive-SD rejection
# ---------------------------------------------------------------------


def test_the_sd_inflation_is_recorded_as_rejected(resolution):
    verdict = resolution["predictive_sd_verdict"]
    assert verdict["verdict"] == "REJECTED"
    assert verdict["runtime_behaviour_carried"] is False
    assert verdict["replacement_tuned_against_holdout"] is False
    assert verdict["raw_mean_squared_z"] == pytest.approx(0.6652, abs=5e-4)
    assert verdict["calibrated_mean_squared_z"] == pytest.approx(0.3908, abs=5e-4)


def test_the_inflated_uncertainty_was_further_from_calibrated_than_the_raw_one(
    resolution,
):
    """The reason for the rejection, not just the fact of it."""
    verdict = resolution["predictive_sd_verdict"]
    raw_gap = abs(verdict["raw_mean_squared_z"] - verdict["target"])
    inflated_gap = abs(verdict["calibrated_mean_squared_z"] - verdict["target"])
    assert inflated_gap > raw_gap


# ---------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------


def test_no_pairwise_or_player_indexed_parameters(resolution):
    counts = resolution["parameter_accounting"]
    assert counts["pairwise_parameter_count"] == 0
    assert counts["player_indexed"] == 0


def test_the_rejected_layers_contribute_no_parameters(resolution):
    counts = resolution["parameter_accounting"]
    assert counts["active_role_deviation_parameters"] == 0
    assert counts["active_symmetric_subspace_parameters"] == 0


def test_the_accepted_role_scale_is_reported_as_active_rather_than_as_zero(resolution):
    """Accounting honesty, as a test.

    The accepted repair's multiplicative role layer leaves the pooled block
    untouched, so every pooled-level check passes whether or not it is there.
    That is precisely why it has to be reported explicitly: a manifest that
    folded it into a single "role parameters: 0" line would be wrong in a way
    no other test on this branch would catch.
    """
    role_scale = resolution["parameter_accounting"]["role_scale"]
    assert role_scale["active"] is True
    assert role_scale["value_count"] == 3
    assert role_scale["free_parameters"] == 2
    assert role_scale["pooled_same_team_block_unchanged"] is True
    assert resolution["parameter_accounting"]["accounting_discrepancy"]


def test_the_role_scale_is_not_inert_on_individual_pairs(repair_spec):
    """Backs the accounting claim against the shipped loadings themselves."""
    loadings = SharedFactorLoadings.from_payload(repair_spec["loadings"])
    scales = dict(loadings.role_scale)
    assert scales, "the accepted repair fitted role scales"
    multipliers = [
        scales[left] * scales[right] for left in scales for right in scales
    ]
    assert max(multipliers) > 1.0
    assert min(multipliers) < 1.0


# ---------------------------------------------------------------------
# Reused metrics are the accepted ones
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legs", "repair_log_loss", "production_log_loss"),
    [
        ("2", 0.504275, 0.504199),
        ("3", 0.333822, 0.333807),
        ("4", 0.213939, 0.213865),
    ],
)
def test_the_reused_log_losses_are_the_accepted_ones(
    resolution, legs, repair_log_loss, production_log_loss
):
    payload = resolution["reused_holdout_metrics"]["by_legs"][legs]
    assert payload["log_loss"]["candidate"] == pytest.approx(repair_log_loss, abs=5e-7)
    assert payload["log_loss"]["baseline_production"] == pytest.approx(
        production_log_loss, abs=5e-7
    )


def test_the_reused_metrics_carry_the_full_acceptance_evidence(resolution):
    reused = resolution["reused_holdout_metrics"]
    assert reused["same_player_contract"]["max_block_deviation"] < 1e-9
    assert reused["stability_and_psd"]["numerical_failures"] == 0
    assert reused["stability_and_psd"]["min_covariance_eigenvalue"] > 0.0
    assert reused["repair_gates"]["all_passed"] is True
    assert set(reused["marginal_preservation"]) == {
        "candidate",
        "baseline_independence",
        "baseline_production",
    }
    assert reused["latent_bucket_rmse"]["candidate"] < (
        reused["latent_bucket_rmse"]["baseline_independence"]
    )


# ---------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------


def test_the_resolution_records_the_blob_contract_without_claiming_to_have_met_it(
    resolution,
):
    """The artifact cannot audit the commit that carries it.

    It is written first, so at that moment the merge path is empty and any
    ``passed`` it recorded would be vacuously true. What it should carry is the
    contract and a pointer to the check that runs against the real head.
    """
    audit = resolution["merge_path_audit"]
    assert audit["ceiling_bytes"] == MAX_MERGE_PATH_BLOB_BYTES
    assert audit["snapshot_is_not_the_verdict"]
    assert "passed" not in audit


def test_the_live_merge_path_audit_is_what_passes(resolution):
    """And the check that is not vacuous runs here, against the real head."""
    audit = audit_merge_path(PROJECT, base=resolution["production_base"])
    assert audit["base_resolved"] is True
    assert audit["new_blob_count"] > 0, (
        "an empty merge path against the recorded production base means the "
        "base is wrong, not that the branch is clean"
    )
    assert audit["over_ceiling"] == []
    assert audit["generated_artifacts"] == []
    assert audit["passed"] is True


def test_the_resolution_touches_no_production_path(resolution):
    assert resolution["protected_production_paths_modified"] == []


def test_the_research_branches_are_recorded_as_unmergeable_directly(resolution):
    assert resolution["research_branches_must_not_be_merged_directly"]
    assert len(resolution["research_branches_left_open"]) == 3


def test_every_carried_manifest_declares_itself_non_promotable():
    for directory in (V1_DIR, REPAIR_DIR, V2_DIR, RESOLUTION_DIR):
        for path in sorted(directory.glob("*.json")):
            payload = load(path)
            if not isinstance(payload, dict):
                continue
            eligibility = payload.get("promotion_eligibility")
            if eligibility is None:
                continue
            assert "NOT_ELIGIBLE_FOR_PROMOTION" in eligibility, path
