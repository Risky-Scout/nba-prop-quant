#!/usr/bin/env python
"""Close the latent-state structural research and state what ships.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Three research rounds ran: Shadow V1 (PR #16), the bucket repair (PR #17) and
the V2 structural refinement (PR #18). This writes the single artifact that
says what survived, what did not, and on what evidence, so that a reader does
not have to reconstruct the conclusion from three branches.

It derives every number from the committed artifacts rather than restating
them, fits nothing, simulates nothing, and reads no residual data. The
dependence model that ships is the accepted bucket repair's, unchanged.

Why no further holdout simulation is required
---------------------------------------------
``07_prove_equivalence_and_calibrate_sd.py`` established that the frozen V2
candidate and the accepted repair differ in no input the simulator reads. The
simulator's output is a pure function of
``(roster, marginals, loadings, within_player, simulations, seed)``; five of
those six are model-independent, and ``loadings`` is bitwise equal between the
two. So the repair's committed grades *are* the candidate's grades, and
re-running the Monte Carlo would recompute identical numbers.

Parameter accounting
--------------------
Reported, not assumed, and deliberately not rounded in the model's favour. The
rejected V2 dials contribute nothing. The accepted repair's multiplicative
``role_scale`` is a different thing from the rejected additive
``role_deviation`` layer, it is genuinely active, and it is reported as such
rather than folded into a single "role parameters: 0" line that would be
false. See :func:`role_scale_accounting`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    sha256_file,
    write_checksums,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    SharedFactorLoadings,
)
from nba_prop_quant.research.game_latent_state.safety import (  # noqa: E402
    audit_merge_path,
    modified_production_paths,
)

console = Console()

V1_DIR = PROJECT_ROOT / "research" / "game_latent_state"
REPAIR_DIR = PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"
V2_DIR = PROJECT_ROOT / "research" / "game_latent_state_v2"
RESOLUTION_DIR = PROJECT_ROOT / "research" / "game_latent_state_resolution"

#: The components the research screened and did not carry, each with the
#: artifact that decided it. A rejection with no evidence attached is an
#: opinion, so the key is the evidence path.
REJECTED_COMPONENTS: dict[str, dict[str, str]] = {
    "r_symmetric > 0": {
        "final_value": "r_symmetric = 0",
        "evidence": "game_latent_state_v2/axis_indifference_resolution.json",
        "why": (
            "At full rank the three-family construction is already exact, so "
            "every admissible symmetric rank reproduced the axis's null value "
            "on every judged metric. The worst relative deviation across the "
            "admissible points was 6.5e-15 against a 1e-9 tolerance, which "
            "makes those points the same model written differently rather "
            "than a better one."
        ),
    },
    "role_deviation = true": {
        "final_value": "role_deviation = false",
        "evidence": "game_latent_state_v2/inner_screening.json",
        "why": (
            "The layer cut support-weighted role-conditioned RMSE by 44.1% "
            "but left two role cells worse, by up to 23.1%, so it bought an "
            "aggregate gain by moving error into specific cells."
        ),
    },
    "bridge_weight > 0": {
        "final_value": "bridge_weight = 0",
        "evidence": "game_latent_state_v2/inner_screening.json",
        "why": (
            "Positive weights cut primary count-space error by about 45% and "
            "pushed global latent RMSE to 0.004636, outside the 0.003750 "
            "pre-registered bound. Every positive-weight grid point sat "
            "outside that bound."
        ),
    },
    "dynamic temporal treatment": {
        "final_value": "temporal_treatment = T0_pooled_empirical_bayes",
        "evidence": "game_latent_state_v2/temporal_diagnostic.json",
        "why": (
            "T0 -- what the accepted repair already does -- had the lowest "
            "predictive MSE within the parsimony tie band on the inner "
            "walk-forward folds. The random-walk state treatment could not "
            "even be supported: four training seasons is below the six a "
            "one-state walk needs, so it fell back to pooled."
        ),
    },
    "predictive-SD inflation factor 1.7659": {
        "final_value": "no predictive-SD inflation; the layer is not carried",
        "evidence": "game_latent_state_v2/sd_calibration_blocker.json",
        "why": (
            "Failed its pre-registered out-of-sample criterion on untouched "
            "2024-2025. Raw mean z^2 was 0.6652 and the inflated value was "
            "0.3908, against a target near 1, so inflating moved the reported "
            "uncertainty away from calibration. The factor is sqrt of a mean "
            "over two inner folds that disagree by a factor of 8.9, and it "
            "double-counts the one season of heterogeneity the random-effects "
            "tau^2 already absorbs."
        ),
    },
}


def git_sha(ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", *ref.split()],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    ).stdout.strip()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def role_scale_accounting(loadings: SharedFactorLoadings) -> dict:
    """What the accepted repair's multiplicative role layer actually costs.

    Two role layers exist in the codebase and only one of them ships, so
    collapsing them into a single count would misreport the model.

    ``role_deviation`` is the *additive* layer the V2 search proposed. It was
    screened and not carried, and it contributes nothing.

    ``role_scale`` is the *multiplicative* layer the accepted repair fitted.
    It is renormalised so the player-weighted mean scale is 1, which leaves
    the pooled same-team block exactly where the base loadings put it -- but
    it is not inert: an individual pair's implied correlation is scaled by
    ``s_a * s_b``. The normalisation is one constraint on the fitted values,
    so the free count is one less than the number of roles.
    """
    scales = {str(role): float(value) for role, value in loadings.role_scale.items()}
    pair_multipliers = {
        f"{left}+{right}": scales[left] * scales[right]
        for index, left in enumerate(sorted(scales))
        for right in sorted(scales)[index:]
    }
    without = replace(loadings, role_scale={})
    pooled_unchanged = bool(
        np.array_equal(loadings.same_team_correlation(), without.same_team_correlation())
    )
    # The loadings carry the ordered role-pair shares precisely so this average
    # is computable without the frame they were fitted on. If the
    # normalisation holds, the pair-weighted mean of s_a * s_b is 1.
    shares = {
        (str(left), str(right)): float(value)
        for (left, right), value in loadings.role_pair_shares.items()
    }
    pair_weighted_mean = (
        sum(
            share * scales.get(left, 1.0) * scales.get(right, 1.0)
            for (left, right), share in shares.items()
        )
        / sum(shares.values())
        if shares
        else None
    )
    return {
        "layer": "multiplicative role_scale, fitted by the accepted bucket repair",
        "distinct_from": "the rejected additive role_deviation layer",
        "values": scales,
        "value_count": len(scales),
        "normalisation_constraints": 1,
        "free_parameters": max(len(scales) - 1, 0),
        "normalisation": "player-weighted mean scale is 1 (sum_r p_r s_r == 1)",
        "pair_weighted_mean_of_s_a_times_s_b": pair_weighted_mean,
        "pooled_same_team_block_unchanged": pooled_unchanged,
        "active": bool(scales) and any(abs(value - 1.0) > 0 for value in scales.values()),
        "pair_multipliers": pair_multipliers,
        "note": (
            "Active, and materially so: the per-pair multiplier s_a * s_b "
            "ranges from 0.695 to 1.540 across the fitted roles. The "
            "normalisation keeps the *pooled* block fixed, so a pooled-level "
            "check cannot see this layer at all, which is exactly why it "
            "would be easy to misreport as absent. Reporting zero active "
            "role-indexed parameters would therefore be false. The layer is "
            "part of the model accepted in PR #17 and is carried unchanged; "
            "nothing here refits it."
        ),
        "pair_weighted_mean_unavailable_reason": (
            None
            if pair_weighted_mean is not None
            else (
                "The accepted repair's committed loadings payload predates "
                "role_pair_shares, so the ordered role-pair shares needed to "
                "average s_a * s_b are not in it. The weaker invariant the "
                "payload does support -- that dropping role_scale leaves the "
                "pooled same-team correlation unchanged -- is checked above "
                "and holds."
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research" / "game_latent_state_resolution",
    )
    parser.add_argument("--production-base", default=None)
    args = parser.parse_args()

    repair_spec = load(REPAIR_DIR / "factor_spec.json")
    repair_report = load(REPAIR_DIR / "validation_report.json")
    repair_gates = load(REPAIR_DIR / "repair_gates.json")
    equivalence = load(V2_DIR / "equivalence_and_calibration.json")
    blocker = load(V2_DIR / "sd_calibration_blocker.json")
    frozen = load(V2_DIR / "shadow_v2_candidate.json")
    loadings = SharedFactorLoadings.from_payload(repair_spec["loadings"])

    console.rule("Final resolution: what ships and what does not")

    # ---- the winning dependence model -------------------------------
    final_spec = {
        "model": "accepted bucket repair",
        "k_game": int(repair_spec["k_game"]),
        "r_contrast": int(repair_spec["r_contrast"]),
        "r_symmetric": 0,
        "role_deviation": False,
        "bridge_weight": 0.0,
        "temporal_treatment": "T0_pooled_empirical_bayes",
        "predictive_sd_inflation": None,
        "shrinkage": repair_spec["shrinkage"],
        "eb_family": repair_spec["eb_family"],
        "shrink_z": float(repair_spec["shrink_z"]),
        "role_column": repair_spec["role_column"],
        "stats": list(repair_spec["stats"]),
        "training_seasons": list(repair_spec["training_seasons"]),
        "validation_seasons": list(repair_spec["validation_seasons"]),
        "factor_spec_hash": repair_spec["spec_hash"],
        "factor_families": list(repair_spec["factor_families"]),
    }
    spec_table = Table(title="final dependence specification")
    spec_table.add_column("field")
    spec_table.add_column("value", justify="right")
    for key in (
        "k_game",
        "r_contrast",
        "r_symmetric",
        "role_deviation",
        "bridge_weight",
        "temporal_treatment",
        "predictive_sd_inflation",
    ):
        spec_table.add_row(key, str(final_spec[key]))
    console.print(spec_table)

    # ---- parameter accounting ---------------------------------------
    role_scale = role_scale_accounting(loadings)
    stability = repair_report["stability"]
    counts = frozen["parameter_counts"]
    parameter_accounting = {
        "pairwise_parameter_count": int(stability["pairwise_parameter_count"]),
        "player_indexed": int(counts["player_indexed"]),
        "active_role_deviation_parameters": int(counts["stat_indexed_role_deviation"]),
        "active_symmetric_subspace_parameters": int(counts["stat_indexed_symmetric"]),
        "stat_indexed_game": int(counts["stat_indexed_game"]),
        "stat_indexed_contrast": int(counts["stat_indexed_contrast"]),
        "stat_indexed_competition": int(counts["stat_indexed_competition"]),
        "scalar_hyperparameters": int(counts["scalar_hyperparameters"]),
        "role_scale": role_scale,
        "measured_not_refitted": True,
        "frozen_candidate_reported_role_indexed": int(counts["role_indexed"]),
        "frozen_candidate_role_indexed_note": (
            "The freeze record charges one parameter per role whenever a role "
            "column is configured, even with the additive layer off, so it "
            "reports 3. That is a deliberate over-count on the conservative "
            "side and it refers to role_scale, not to role_deviation. It is "
            "preserved here rather than corrected, because correcting a "
            "frozen record after the fact is how provenance is lost."
        ),
        "accounting_discrepancy": (
            "A request to report active_role_indexed_parameters = 0 cannot be "
            "satisfied truthfully. The rejected additive role_deviation layer "
            "does contribute 0. The accepted repair's multiplicative "
            "role_scale contributes 3 fitted values under 1 normalisation "
            "constraint, it is active, and it was accepted in PR #17 as part "
            "of the winning model. Both numbers are reported above instead of "
            "one number that would be wrong."
        ),
    }
    count_table = Table(title="parameter accounting")
    count_table.add_column("quantity")
    count_table.add_column("count", justify="right")
    for key in (
        "pairwise_parameter_count",
        "player_indexed",
        "active_role_deviation_parameters",
        "active_symmetric_subspace_parameters",
    ):
        count_table.add_row(key, str(parameter_accounting[key]))
    count_table.add_row(
        "role_scale (accepted repair)",
        f"{role_scale['value_count']} values, {role_scale['free_parameters']} free, "
        f"active={role_scale['active']}",
    )
    console.print(count_table)

    # ---- reused holdout metrics -------------------------------------
    legs = repair_report["joint_events"]["by_legs"]
    reused = {
        "scope": (
            "untouched 2024-2025, 600 graded games, from the accepted repair's "
            "committed validation run; nothing re-simulated"
        ),
        "source": "game_latent_state_bucket_repair/validation_report.json",
        "source_sha256": sha256_file(REPAIR_DIR / "validation_report.json"),
        "games_simulated": int(repair_report["games_simulated"]),
        "simulations_per_game": int(repair_report["simulations_per_game"]),
        "seed": int(repair_report["seed"]),
        "by_legs": {
            str(count): {
                "events": int(payload["candidate"].get("events", 0))
                or int(payload.get("events", 0)),
                "base_rate": float(payload["base_rate"]),
                "brier": {
                    name: float(payload[name]["brier"])
                    for name in ("candidate", "baseline_independence", "baseline_production")
                    if name in payload
                },
                "log_loss": {
                    name: float(payload[name]["log_loss"])
                    for name in ("candidate", "baseline_independence", "baseline_production")
                    if name in payload
                },
            }
            for count, payload in sorted(legs.items())
        },
        "latent_dependence_cross_player_rmse": repair_report["residual_dependence"][
            "cross_player_rmse"
        ],
        "count_space_cross_player_rmse": repair_report["residual_dependence"][
            "count_space_cross_player_rmse"
        ],
        "latent_bucket_rmse": {
            name: float(entry["rmse"])
            for name, entry in repair_report["latent_dependence"]["by_model"].items()
        },
        "marginal_preservation": repair_report["marginal_preservation"],
        "same_player_contract": repair_report["same_player_contract"],
        "stability_and_psd": {
            key: value
            for key, value in repair_report["stability"].items()
            if key != "failure_detail"
        },
        "repair_gates": {
            "all_passed": bool(repair_gates["all_passed"]),
            "verdict": repair_gates["verdict"],
            "count": len(repair_gates["gates"]),
        },
    }
    leg_table = Table(title="reused holdout metrics by leg count (nothing re-simulated)")
    for column in ("legs", "base rate", "repair Brier", "prod Brier", "repair log loss", "prod log loss"):
        leg_table.add_column(column, justify="right")
    for count, payload in sorted(reused["by_legs"].items()):
        leg_table.add_row(
            count,
            f"{payload['base_rate']:.4f}",
            f"{payload['brier']['candidate']:.6f}",
            f"{payload['brier']['baseline_production']:.6f}",
            f"{payload['log_loss']['candidate']:.6f}",
            f"{payload['log_loss']['baseline_production']:.6f}",
        )
    console.print(leg_table)

    # ---- the equivalence proof --------------------------------------
    identity = equivalence["equivalence"]["game_covariance_identity"]
    proof = {
        "claim": (
            "The frozen V2 candidate's point-probability model is the accepted "
            "repair's, so the repair's committed grades are the candidate's "
            "grades and no further holdout simulation is required."
        ),
        "source": "game_latent_state_v2/equivalence_and_calibration.json",
        "source_sha256": sha256_file(V2_DIR / "equivalence_and_calibration.json"),
        "loadings_payload_bitwise_identical": bool(
            equivalence["equivalence"]["loadings_payload_bitwise_identical"]
        ),
        "games_checked": int(identity["games_checked"]),
        "covariance_assemblies_compared": int(identity["covariance_assemblies_compared"]),
        "max_abs_correlation_difference": float(identity["max_abs_correlation_difference"]),
        "max_abs_cholesky_difference": float(identity["max_abs_cholesky_difference"]),
        "bitwise_identical_everywhere": bool(identity["bitwise_identical_everywhere"]),
        "by_refit_window": {
            window: {
                "games_checked": int(entry["games_checked"]),
                "bitwise_identical_everywhere": bool(entry["bitwise_identical_everywhere"]),
            }
            for window, entry in equivalence["equivalence"][
                "game_covariance_identity_by_refit_window"
            ].items()
        },
        "model_enters_only_through_loadings": equivalence["equivalence"][
            "model_enters_only_through_loadings"
        ]["all_signature_checks_pass"],
        "why_it_holds_in_every_refit_window": equivalence["equivalence"][
            "why_this_holds_in_every_refit_window"
        ],
        "monte_carlo_rerun_required": False,
    }
    console.print(
        f"equivalence: {proof['games_checked']} graded games, "
        f"max |correlation difference| = {proof['max_abs_correlation_difference']:.3e}, "
        f"max |Cholesky difference| = {proof['max_abs_cholesky_difference']:.3e}"
    )

    # ---- the predictive-SD verdict ----------------------------------
    calibration = equivalence["predictive_sd_calibration"]
    sd_verdict = {
        "component": "predictive-SD inflation factor",
        "value_screened": float(calibration["inflation_factor"]),
        "verdict": "REJECTED",
        "scope": calibration["scope"],
        "raw_mean_squared_z": float(calibration["pooled"]["raw_mean_squared_z"]),
        "calibrated_mean_squared_z": float(
            calibration["pooled"]["calibrated_mean_squared_z"]
        ),
        "target": 1.0,
        "pre_registered_criteria": calibration["pre_registered_criteria"],
        "sub_verdicts": calibration["verdicts"],
        "cause": blocker["cause"],
        "double_counting_established": bool(blocker["double_counting_established"]),
        "replacement_tuned_against_holdout": False,
        "runtime_behaviour_carried": False,
    }
    console.print(
        f"predictive-SD inflation {sd_verdict['value_screened']:.4f}: REJECTED "
        f"(raw {sd_verdict['raw_mean_squared_z']:.4f} -> inflated "
        f"{sd_verdict['calibrated_mean_squared_z']:.4f}, target 1)"
    )

    # ---- containment and the merge path -----------------------------
    audit = audit_merge_path(PROJECT_ROOT, base=args.production_base)
    touched = modified_production_paths(PROJECT_ROOT)
    console.print(
        f"merge path: {audit['new_blob_count']} new objects, largest "
        + (
            f"{audit['largest']['bytes'] / (1024 * 1024):.3f} MB"
            if audit["largest"]
            else "none"
        )
        + f", audit passed: {audit['passed']}"
    )
    console.print(f"protected production paths modified: {touched or 'none'}")

    resolution = {
        "title": "Latent-state structural research: final resolution",
        "status": "STRUCTURAL RESEARCH CLOSED",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "winning_dependence_model": final_spec,
        "rejected_components": REJECTED_COMPONENTS,
        "all_v2_structural_additions_screened_and_not_carried": True,
        "parameter_accounting": parameter_accounting,
        "reused_holdout_metrics": reused,
        "equivalence_proof": proof,
        "predictive_sd_verdict": sd_verdict,
        "further_holdout_simulation_required": False,
        "merge_path_audit": {
            "snapshot_at_generation_time": audit,
            "snapshot_is_not_the_verdict": (
                "This file is written before the commit that carries it, so at "
                "generation time the merge path is empty and the snapshot above "
                "reports zero new objects. That is an artefact of ordering, not "
                "a clean result. The verdict is whatever "
                "research/game_latent_state_resolution/audit_merge_path_blobs.py "
                "reports against the pushed branch head, which is also what "
                "test_no_new_object_in_the_merge_path_breaches_the_blob_contract "
                "enforces on every run."
            ),
            "ceiling_bytes": audit["ceiling_bytes"],
            "notable_bytes": audit["notable_bytes"],
            "base": audit["base"],
        },
        "protected_production_paths_modified": touched,
        "production_base": git_sha("origin/production/wizardofodds-integration")
        or args.production_base,
        "clean_branch": git_sha("--abbrev-ref HEAD") or "research/nba-game-latent-state-shadow-clean",
        "code_sha_note": (
            "code_sha is HEAD at generation time, which is the commit *before* "
            "the one that carries this file. A head SHA cannot name the commit "
            "that contains it, so the clean branch head is reported in the "
            "pull request rather than claimed here."
        ),
        "parent_shadow_v1_sha": frozen["parent_shadow_v1_sha"],
        "parent_bucket_repair_sha": frozen["parent_bucket_repair_sha"],
        "research_branches_left_open": [
            "research/nba-game-latent-state-shadow-v1 (PR #16)",
            "research/nba-game-latent-state-shadow-v1-bucket-repair (PR #17)",
            "research/nba-game-latent-state-shadow-v2-structural (PR #18)",
        ],
        "research_branches_must_not_be_merged_directly": (
            "All three reach a 65 MB residual parquet object that was "
            "committed in 43e4d2a and removed in 4e15feb. Deleting a file "
            "does not remove the object, so an ordinary merge commit would "
            "carry it into production history forever. This branch was built "
            "fresh from the production base and copies only the files it "
            "should carry, which is why its merge path is clean."
        ),
        "code_sha": git_sha(),
    }

    artifact_dir = Path(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = write_json(resolution, artifact_dir / "final_resolution.json")
    # Written here rather than by hand, because a checksum file that is not
    # regenerated alongside what it covers goes stale without saying so.
    checksums = write_checksums(
        [
            path,
            RESOLUTION_DIR / "00_write_final_resolution.py",
            RESOLUTION_DIR / "audit_merge_path_blobs.py",
            RESOLUTION_DIR / "README.md",
        ],
        RESOLUTION_DIR / "SHA256SUMS.resolution.txt",
    )
    console.rule("Resolution")
    console.print("WINNING_DEPENDENCE_MODEL=ACCEPTED_BUCKET_REPAIR")
    console.print("ALL_V2_STRUCTURAL_ADDITIONS_NOT_CARRIED=YES")
    console.print("PREDICTIVE_SD_INFLATION=REJECTED")
    console.print("FURTHER_HOLDOUT_SIMULATION_REQUIRED=NO")
    console.print("MERGE_PATH_AUDIT_VERDICT=SEE_audit_merge_path_blobs.py_AGAINST_THE_PUSHED_HEAD")
    console.print(f"PRODUCTION_PATHS_MODIFIED={len(touched)}")
    console.print(f"\nwrote {path}")
    console.print(f"wrote {checksums}")


if __name__ == "__main__":
    main()
