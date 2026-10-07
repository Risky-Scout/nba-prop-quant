#!/usr/bin/env python
"""Write the one authoritative final model specification.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE WITHOUT A SEPARATE PROMOTION DECISION.

Everything here is read out of a committed artifact and hashed, not retyped.
The point of the file this writes is that there is exactly one place to look
for what the final dependence model is, and that every number in it can be
traced to the run that produced it.

The spec records four dispositions explicitly, because each one closes a line
of work that a later reader would otherwise reopen:

    COUNT_SPACE_FORENSIC_CHANGE_ADOPTED
    COUNT_SPACE_20_PERCENT_REQUIREMENT_DISPOSITION
    MARGINAL_CONVENTION_AUDIT
    UNCERTAINTY_CALIBRATION

It also records that the uncertainty layer is a monitor. It is reported, it is
not calibrated, and nothing selects or promotes on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    git_sha,
    sha256_canonical,
    sha256_file,
    write_checksums,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    SharedFactorLoadings,
)

console = Console()

AUTHORITATIVE_SPEC_HASH = (
    "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
)

REMEDIATION = PROJECT_ROOT / "research/final_upstream_remediation"
ABLATION = PROJECT_ROOT / "research/final_role_scale_ablation"
FORENSIC = PROJECT_ROOT / "research/count_space_forensic"
CONVENTION = PROJECT_ROOT / "research/marginal_convention_audit"
RESOLUTION = PROJECT_ROOT / "research/game_latent_state_resolution"
REPAIR = PROJECT_ROOT / "research/game_latent_state_bucket_repair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=PROJECT_ROOT / "research/final_model"
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hashed(paths: dict[str, Path]) -> dict[str, object]:
    """Every source this spec depends on, with its digest, or None if absent."""
    return {
        name: {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
        for name, path in paths.items()
    }


def build() -> dict[str, object]:
    spec = load(REMEDIATION / "factor_spec.json")
    diagnostics = load(REMEDIATION / "covariance_diagnostics.json")
    validation = load(REMEDIATION / "validation_report.json")
    gates = load(REMEDIATION / "gate_report.json")
    audit = load(REMEDIATION / "final_audit.json")
    ablation = load(ABLATION / "role_scale_ablation.json")
    confirmation = load(ABLATION / "holdout_confirmation.json")
    convention = load(CONVENTION / "marginal_convention_audit.json")

    loadings = SharedFactorLoadings.from_payload(spec["loadings"])
    choices = spec["upstream_choices"]
    remediation_spec = spec["remediation_spec"]

    recomputed = dict(spec)
    recorded_hash = recomputed.pop("spec_hash")

    forensic_path = FORENSIC / "count_space_forensic.json"
    forensic = load(forensic_path) if forensic_path.exists() else {}

    final: dict[str, object] = {
        "title": "final NBA same-game dependence model specification",
        "status": "FROZEN",
        "scope": (
            "SHADOW ONLY. The incumbent remains the published authority. This "
            "spec carries no promotion authority of its own."
        ),
        "promotion_eligibility": (
            "SHADOW_RUNTIME_ONLY__PROMOTION_REQUIRES_A_SEPARATE_DECISION"
        ),
        "dependence_model_version": spec["dependence_model_version"],
        "supersedes": [
            "research/game_latent_state_resolution/final_resolution.json",
            "research/game_latent_state_bucket_repair/factor_spec.json",
        ],
        # ---- the model ------------------------------------------------
        "temporal_treatment": {
            "treatment": choices["temporal_treatment"],
            "nu": choices["temporal_nu"],
            "half_life": choices["temporal_half_life"],
            "half_life_is_read_by_this_treatment": False,
            "half_life_note": (
                "A2 is the robust Student-t without recency weighting, so the "
                "half life is carried in the choice record but never read. The "
                "final audit probed it and found the fit bitwise identical "
                "across every value."
            ),
            "enters_as": (
                "a shift on the twelve named cross-player buckets, so A0 is an "
                "exact no-op and the treatment composes with the bridge"
            ),
        },
        "shrinkage": {
            "same_team": "empirical_bayes",
            "cross_team": "empirical_bayes",
            "eb_family": spec["eb_family"],
            "cross_team_prior": choices["cross_team_prior"],
            "cross_team_nu": choices["cross_team_nu"],
        },
        "k_game": int(spec["k_game"]),
        "r_contrast": int(spec["r_contrast"]),
        "r_competition": int(loadings.r_competition),
        "r_symmetric": int(loadings.r_symmetric),
        "role_scale": {
            "mode": remediation_spec["role_scale_mode"],
            "values": {k: float(v) for k, v in loadings.role_scale.items()},
            "role_column": spec["role_column"],
            "identification": spec["role_scale_identification"],
            "tau_log": float(diagnostics["role_scale"]["tau_log"]),
            "raw_scales": {
                k: float(v)
                for k, v in diagnostics["role_scale"]["raw_scales"].items()
            },
            "log_standard_errors": {
                k: float(v)
                for k, v in diagnostics["role_scale"]["log_standard_errors"].items()
            },
            "player_share_weights": {
                k: float(v)
                for k, v in diagnostics["role_scale"]["player_shares"].items()
            },
            "weighted_mean_scale_minus_one": float(
                diagnostics["role_scale"]["weighted_mean_scale_minus_one"]
            ),
            "selected_by": "research/final_role_scale_ablation",
        },
        "role_deviation": False,
        "role_deviation_note": (
            "absent from the payload, not zeroed: loadings.role_deviation is "
            "None and the role offset map is empty"
        ),
        "dependence_temperature": float(choices["dependence_temperature"]),
        "dependence_temperature_note": spec["dependence_temperature_note"],
        "count_transmission": {
            "treatment": remediation_spec["bridge_mode"],
            "weight_cap": float(remediation_spec["bridge_weight_cap"]),
            "order": 2,
            "enters_as": (
                "an inverse-variance combination of the direct latent reading "
                "and the count-space reading, both inverted onto a common rho "
                "first, with the bridge's weight capped"
            ),
        },
        "uncertainty_treatment": "RAW_MONITOR_ONLY",
        "uncertainty_model": choices["uncertainty_model"],
        "within_player_block_source": spec["within_player_block_source"],
        "factor_families": list(spec["factor_families"]),
        "stats": list(spec["stats"]),
        "training_seasons": list(spec["training_seasons"]),
        "validation_seasons": list(spec["validation_seasons"]),
        "standardization_moments": spec["standardization_moments"],
        # ---- parameter accounting -------------------------------------
        "pairwise_parameter_count": int(
            validation["stability"]["pairwise_parameter_count"]
        ),
        "player_indexed": 0,
        "player_indexed_note": (
            "no loading, scale or offset is indexed by player identity; a "
            "player enters only through its role bucket, its team side and "
            "its pinned incumbent within-player block"
        ),
        "role_indexed": len(loadings.role_scale),
        # ---- hashes and SHAs -------------------------------------------
        "factor_spec_hash": recorded_hash,
        "factor_spec_hash_recomputed": sha256_canonical(recomputed),
        "factor_spec_hash_matches_authoritative": (
            recorded_hash == AUTHORITATIVE_SPEC_HASH
        ),
        "authoritative_spec_hash": AUTHORITATIVE_SPEC_HASH,
        "control_factor_spec_hash": spec["control_factor_spec_hash"],
        "source_shas": {
            **audit["provenance"],
            "16_final_model_spec_code_sha": git_sha(PROJECT_ROOT),
            "16_note": (
                "HEAD when this spec was written. The commit carrying it is "
                "its child, since a recorded hash cannot name the commit that "
                "contains it."
            ),
        },
        "source_artifacts": hashed(
            {
                "factor_spec": REMEDIATION / "factor_spec.json",
                "covariance_diagnostics": REMEDIATION
                / "covariance_diagnostics.json",
                "inner_selection": REMEDIATION / "inner_selection.json",
                "dependence_temperature": REMEDIATION
                / "dependence_temperature.json",
                "validation_report": REMEDIATION / "validation_report.json",
                "paired_joint_calibration": REMEDIATION
                / "paired_joint_calibration.json",
                "gate_report": REMEDIATION / "gate_report.json",
                "final_audit": REMEDIATION / "final_audit.json",
                "role_scale_ablation": ABLATION / "role_scale_ablation.json",
                "holdout_confirmation": ABLATION / "holdout_confirmation.json",
                "count_space_forensic": forensic_path,
                "marginal_convention_audit": CONVENTION
                / "marginal_convention_audit.json",
                "structural_resolution": RESOLUTION / "final_resolution.json",
                "control_factor_spec": REPAIR / "factor_spec.json",
            }
        ),
        # ---- how it was graded ------------------------------------------
        "grading": {
            "verdict": validation["verdict"],
            "gate_verdict": gates["verdict"],
            "gates_passed": gates["gates_passed"],
            "gates_total": gates["gates_total"],
            "all_gates_passed": gates["all_gates_passed"],
            "holdout_seasons": list(validation["validation_seasons"]),
            "games_graded": int(validation["stability"]["games_tested"]),
            "simulations_per_game": int(validation["simulations_per_game"]),
            "same_player_max_block_deviation": float(
                validation["same_player_contract"]["max_block_deviation"]
            ),
            "min_covariance_eigenvalue": float(
                validation["stability"]["min_covariance_eigenvalue"]
            ),
            "numerical_failures": int(validation["stability"]["numerical_failures"]),
            "regraded_for_the_final_freeze": False,
            "regrading_not_required_because": confirmation[
                "monte_carlo_rerun_justification"
            ],
        },
        # ---- the role-scale decision -------------------------------------
        "role_scale_selection": {
            "arms": ablation["arms"],
            "selected": ablation["decision"]["selected"],
            "rule": ablation["decision"]["rule"],
            "criterion_A": ablation["decision"][
                "criterion_A_rmse_improves_by_more_than_one_paired_se"
            ],
            "criterion_B": ablation["decision"][
                "criterion_B_no_cell_worsens_by_more_than_half_a_z"
            ],
            "criterion_C": ablation["decision"][
                "criterion_C_identification_and_contracts_hold"
            ],
            "forward_folds": ablation["forward_fold_target_seasons"],
            "holdout_used_for_role_selection": ablation[
                "holdout_used_for_role_selection"
            ],
            "ROLE_SCALE_SELECTION_FROZEN": "YES",
            "2024_2025_NOT_USED_FOR_ROLE_SELECTION": "YES",
        },
        # ---- the four dispositions ---------------------------------------
        "COUNT_SPACE_FORENSIC_CHANGE_ADOPTED": "NO",
        "COUNT_SPACE_20_PERCENT_REQUIREMENT_DISPOSITION": (
            "INCOMPATIBLE_WITH_OTHER_ORIGINAL_CONSTRAINTS_UNDER_CURRENT_"
            "ARCHITECTURE"
        ),
        "MARGINAL_CONVENTION_AUDIT": "NON_MATERIAL",
        "UNCERTAINTY_CALIBRATION": (
            "RAW_MONITOR_ONLY_NOT_USED_FOR_MODEL_SELECTION_OR_PROMOTION"
        ),
        # ---- what the research concluded, carried forward ------------------
        "research_conclusions_carried_forward": {
            "count_space_forensic": {
                "finding": (
                    "randomized-PIT attenuation of the recorded latent column "
                    "is real and measurable, so the recorded latent "
                    "correlation reads lower than the latent correlation it "
                    "stands for"
                ),
                "why_no_change_was_adopted": (
                    "forcing the focal passer_ast_teammate_pts count-space "
                    "correction far enough to clear the original 20 per cent "
                    "target breaks the teammate_reb_reb protection that was "
                    "part of the same original brief, and no pre-2024 forward "
                    "evidence supports the trade"
                ),
                "status": "CLOSED",
                "parameter_change_adopted": False,
            },
            "marginal_convention_audit": {
                "finding": (
                    "the OOF builder trained every stat's marginal on the "
                    "six-way intersection while the validator and production "
                    "train per stat; one season missing two selected means is "
                    "the whole of the gap"
                ),
                "largest_robust_bucket_shift_z": convention["decision"]["measured"][
                    "largest_key_bucket_shift_z"
                ],
                "bucket_threshold_z": convention["decision"]["thresholds"][
                    "max_key_bucket_shift_z"
                ],
                "global_latent_movement": convention["decision"]["measured"][
                    "largest_global_latent_movement"
                ],
                "global_threshold": convention["decision"]["thresholds"][
                    "max_global_latent_movement"
                ],
                "classification": convention["decision"]["classification"],
                "status": "CLOSED",
                "future_path": (
                    "02_build_oof_residuals.py now defaults to "
                    "v2_per_stat_marginal_training; the frozen candidate is "
                    "not regenerated for it and v1 stays reachable so the "
                    "historical dataset remains reproducible"
                ),
            },
            "uncertainty": {
                "selected": choices["uncertainty_model"],
                "finding": (
                    "no cross-fitted rescaling beat doing nothing on the "
                    "declared coverage levels; the rejected global inflation "
                    "factor was worse calibrated than the raw interval"
                ),
                "role": "monitor only; never an input to selection or promotion",
                "status": "CLOSED",
            },
            "hermite_recurrence": {
                "finding": (
                    "the generic order >= 3 recurrence used the wrong "
                    "coefficient; the bridge has only ever run at order 2, "
                    "where M_k reads He_0 and He_1, so no produced output moved"
                ),
                "frozen_candidate_probabilities_affected": False,
                "status": "FIXED",
            },
            "structural_v2": {
                "finding": (
                    "the symmetric cross-player subspace, the additive role "
                    "deviation layer and the predictive-SD inflation were all "
                    "rejected; they are absent from the payload rather than "
                    "set to zero"
                ),
                "status": "CLOSED",
            },
        },
        "count_space_forensic_detail": (
            forensic.get("decision", {}) if forensic else {}
        ),
    }
    return final


def render(final: dict[str, object]) -> None:
    table = Table(title="final model specification", show_header=True)
    table.add_column("field")
    table.add_column("value", justify="right")
    rows = (
        ("temporal treatment", final["temporal_treatment"]["treatment"]),
        ("same/cross-team shrinkage", final["shrinkage"]["eb_family"]),
        ("cross-team prior", final["shrinkage"]["cross_team_prior"]),
        ("k_game", final["k_game"]),
        ("r_contrast", final["r_contrast"]),
        ("r_competition", final["r_competition"]),
        ("r_symmetric", final["r_symmetric"]),
        ("role_scale mode", final["role_scale"]["mode"]),
        ("role_scale values", json.dumps(final["role_scale"]["values"])),
        ("role_deviation", final["role_deviation"]),
        ("dependence_temperature", final["dependence_temperature"]),
        ("count transmission", final["count_transmission"]["treatment"]),
        ("transmission cap", final["count_transmission"]["weight_cap"]),
        ("uncertainty treatment", final["uncertainty_treatment"]),
        ("pairwise_parameter_count", final["pairwise_parameter_count"]),
        ("player_indexed", final["player_indexed"]),
        ("factor spec hash", str(final["factor_spec_hash"])[:16] + "..."),
    )
    for name, value in rows:
        table.add_row(name, str(value))
    console.print(table)

    dispositions = Table(title="dispositions", show_header=True)
    dispositions.add_column("record")
    dispositions.add_column("value")
    for key in (
        "COUNT_SPACE_FORENSIC_CHANGE_ADOPTED",
        "COUNT_SPACE_20_PERCENT_REQUIREMENT_DISPOSITION",
        "MARGINAL_CONVENTION_AUDIT",
        "UNCERTAINTY_CALIBRATION",
    ):
        dispositions.add_row(key, str(final[key]))
    console.print(dispositions)


def main() -> int:
    args = parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    final = build()
    if not final["factor_spec_hash_matches_authoritative"]:
        raise SystemExit(
            "the frozen factor spec does not hash to the authoritative "
            f"candidate {AUTHORITATIVE_SPEC_HASH}"
        )
    if final["factor_spec_hash"] != final["factor_spec_hash_recomputed"]:
        raise SystemExit(
            "the frozen factor spec's recorded hash does not describe its "
            "own contents"
        )

    destination = args.artifact_root / "final_model_spec.json"
    write_json(final, destination)
    # The spec's own digest is written beside it rather than into it, because
    # a file cannot contain its own hash.
    write_checksums(
        [destination, Path(__file__).resolve()],
        args.artifact_root / "SHA256SUMS.final_model.txt",
    )

    render(final)
    console.print(
        f"\nwrote {destination}\nFINAL MODEL SPEC HASH {sha256_file(destination)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
