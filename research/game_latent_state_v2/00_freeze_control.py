#!/usr/bin/env python
"""Freeze the accepted bucket repair as the immutable Shadow V2 control.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Every number in ``repair_control_baseline.json`` is *copied* out of the
accepted repair's own committed validation artifacts and is never recomputed
for candidate selection. That matters for two reasons.

First, the repair's held-out numbers come from a 2024-2025 simulation. Any
V2 selection step that recomputed them would be reading the final holdout,
which is precisely what the frozen-candidate discipline forbids. Copying them
means the control is a constant of the V2 research rather than an output of
it.

Second, a copied control cannot drift. If V2 re-derived the control through
its own code path, a refactor could move the baseline and a V2 gate could pass
or fail for a reason unrelated to V2. The gates compare against literals.

The script also cross-checks the copied values against the control figures
stated in the research brief, to the precision the brief quotes them at. A
mismatch means the artifacts and the brief disagree about what was accepted,
and there is no safe way to continue from that, so it raises.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

console = Console()

#: Provenance fixed by the brief. The repair is the control; V1 is its parent.
ACCEPTED_SHADOW_V1_SHA = "1c5b8c93569ee25afd4eb4222158300702bd9471"
ACCEPTED_REPAIR_SHA = "c965451de2662b953f45c791157f3826207f753f"
SOURCE_PRODUCTION_SHA = "a30ed672001f8774b6cfec9ad7a3ccb382506c4b"
PR_SHADOW_V1 = 16
PR_BUCKET_REPAIR = 17

#: The three opponent buckets the repair moved as collateral, and the four
#: same-team buckets V2 gate 1 and gate 4 are stated on.
OPPONENT_BUCKETS = ("opponent_ast_ast", "opponent_pts_reb", "opponent_fg3m_reb")
SAME_TEAM_BUCKETS = (
    "teammate_reb_reb",
    "teammate_ast_ast",
    "passer_ast_teammate_pts",
    "teammate_pts_reb",
)

#: Control figures as the brief states them, with the tolerance implied by the
#: number of digits quoted. Checked against the artifacts, never used in place
#: of them: the artifacts carry full float64 precision and the gates use that.
BRIEF_FIGURES: dict[str, tuple[float, float]] = {
    "latent_observed_teammate_reb_reb": (0.009417, 5e-7),
    "latent_repair_teammate_reb_reb": (0.005190, 5e-7),
    "latent_z_teammate_reb_reb": (-1.94, 5e-3),
    "latent_observed_teammate_ast_ast": (0.008971, 5e-7),
    "latent_repair_teammate_ast_ast": (0.004741, 5e-7),
    "latent_z_teammate_ast_ast": (-1.89, 5e-3),
    "latent_cross_player_rmse": (0.002765903, 5e-10),
    "count_cross_player_rmse": (0.007861206, 5e-10),
    "count_observed_passer_ast_teammate_pts": (0.058413, 5e-7),
    "count_repair_passer_ast_teammate_pts": (0.037154, 5e-7),
    "count_absolute_error_passer_ast_teammate_pts": (0.021259, 5e-7),
    "brier_2_leg": (0.16544182, 5e-9),
    "brier_3_leg": (0.09740873, 5e-9),
    "brier_4_leg": (0.05545531, 5e-9),
    "max_same_player_block_deviation": (2.220446049250313e-16, 1e-30),
    "min_covariance_eigenvalue": (0.10253, 5e-6),
    "pairwise_parameter_count": (0.0, 0.0),
}


def git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_branch(root: Path) -> str:
    return subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"),
    )
    parser.add_argument(
        "--v1-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state"),
    )
    parser.add_argument(
        "--artifact-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state_v2"),
    )
    args = parser.parse_args()

    repair_root = Path(args.repair_root)
    v1_root = Path(args.v1_root)
    report = json.loads((repair_root / "validation_report.json").read_text())
    v1_report = json.loads((v1_root / "validation_report.json").read_text())
    gates = json.loads((repair_root / "repair_gates.json").read_text())
    candidate = json.loads((repair_root / "bucket_repair_candidate.json").read_text())
    paired = json.loads((repair_root / "paired_joint_calibration.json").read_text())

    console.rule("Freezing the accepted bucket repair as the V2 control")
    console.print(f"repair verdict: {report['verdict']}")
    console.print(f"repair gates all passed: {gates['all_passed']}")
    if not gates["all_passed"]:
        raise AssertionError("the control repair's own gates did not all pass")

    latent = report["latent_dependence"]
    residual = report["residual_dependence"]
    v1_latent = v1_report["latent_dependence"]
    legs = report["joint_events"]["by_legs"]

    observed_latent = latent["observed_buckets"]
    observed_latent_se = latent["observed_bucket_se"]
    repair_latent = latent["by_model"]["candidate"]["implied_buckets"]
    v1_repair_latent = v1_latent["by_model"]["candidate"]["implied_buckets"]
    observed_count = residual["observed_buckets"]
    repair_count = residual["by_model"]["candidate"]["buckets"]
    repair_count_error = residual["by_model"]["candidate"]["bucket_errors"]

    def latent_z(bucket: str, fitted: dict[str, float]) -> float:
        return (fitted[bucket] - observed_latent[bucket]) / observed_latent_se[bucket]

    console.rule("Latent buckets: V1 and repair against the untouched holdout")
    table = Table(title="held-out latent dependence, z in game-clustered SE")
    for column in (
        "bucket",
        "observed",
        "SE",
        "V1 fitted",
        "V1 z",
        "repair fitted",
        "repair z",
    ):
        table.add_column(column, justify="right")
    latent_control: dict[str, dict[str, float]] = {}
    for bucket in sorted(observed_latent):
        record = {
            "observed": float(observed_latent[bucket]),
            "observed_se": float(observed_latent_se[bucket]),
            "v1_fitted": float(v1_repair_latent[bucket]),
            "v1_z": float(latent_z(bucket, v1_repair_latent)),
            "repair_fitted": float(repair_latent[bucket]),
            "repair_z": float(latent_z(bucket, repair_latent)),
        }
        latent_control[bucket] = record
        table.add_row(
            bucket,
            f"{record['observed']:+.6f}",
            f"{record['observed_se']:.6f}",
            f"{record['v1_fitted']:+.6f}",
            f"{record['v1_z']:+.2f}",
            f"{record['repair_fitted']:+.6f}",
            f"{record['repair_z']:+.2f}",
        )
    console.print(table)
    console.print(
        "The three opponent rows are the collateral the repair's global "
        "shrinkage change produced: "
        + ", ".join(
            f"{bucket} {latent_control[bucket]['v1_z']:+.2f} -> "
            f"{latent_control[bucket]['repair_z']:+.2f}"
            for bucket in OPPONENT_BUCKETS
        )
    )

    count_control = {
        bucket: {
            "observed": float(observed_count[bucket]),
            "repair_fitted": float(repair_count[bucket]),
            "repair_absolute_error": abs(float(repair_count_error[bucket])),
        }
        for bucket in sorted(observed_count)
    }

    checks = {
        "latent_observed_teammate_reb_reb": latent_control["teammate_reb_reb"][
            "observed"
        ],
        "latent_repair_teammate_reb_reb": latent_control["teammate_reb_reb"][
            "repair_fitted"
        ],
        "latent_z_teammate_reb_reb": latent_control["teammate_reb_reb"]["repair_z"],
        "latent_observed_teammate_ast_ast": latent_control["teammate_ast_ast"][
            "observed"
        ],
        "latent_repair_teammate_ast_ast": latent_control["teammate_ast_ast"][
            "repair_fitted"
        ],
        "latent_z_teammate_ast_ast": latent_control["teammate_ast_ast"]["repair_z"],
        "latent_cross_player_rmse": float(residual["cross_player_rmse"]["candidate"]),
        "count_cross_player_rmse": float(
            residual["count_space_cross_player_rmse"]["candidate"]
        ),
        "count_observed_passer_ast_teammate_pts": count_control[
            "passer_ast_teammate_pts"
        ]["observed"],
        "count_repair_passer_ast_teammate_pts": count_control[
            "passer_ast_teammate_pts"
        ]["repair_fitted"],
        "count_absolute_error_passer_ast_teammate_pts": count_control[
            "passer_ast_teammate_pts"
        ]["repair_absolute_error"],
        "brier_2_leg": float(legs["2"]["candidate"]["brier"]),
        "brier_3_leg": float(legs["3"]["candidate"]["brier"]),
        "brier_4_leg": float(legs["4"]["candidate"]["brier"]),
        "max_same_player_block_deviation": float(
            report["same_player_contract"]["max_block_deviation"]
        ),
        "min_covariance_eigenvalue": float(
            report["stability"]["min_covariance_eigenvalue"]
        ),
        "pairwise_parameter_count": float(
            report["stability"]["pairwise_parameter_count"]
        ),
    }

    console.rule("Cross-check against the control figures stated in the brief")
    mismatches = []
    check_table = Table(title="artifact value vs brief value")
    for column in ("figure", "artifact", "brief", "tolerance", "agrees"):
        check_table.add_column(column, justify="right")
    for name, (expected, tolerance) in BRIEF_FIGURES.items():
        actual = checks[name]
        agrees = abs(actual - expected) <= tolerance
        if not agrees:
            mismatches.append((name, actual, expected, tolerance))
        check_table.add_row(
            name,
            f"{actual:.10g}",
            f"{expected:.10g}",
            f"{tolerance:.1g}",
            "yes" if agrees else "NO",
        )
    console.print(check_table)
    if mismatches:
        raise AssertionError(
            "the committed repair artifacts disagree with the control figures "
            f"in the brief: {mismatches}"
        )

    payload = {
        "note": (
            "Immutable control. Every number here is copied from the accepted "
            "bucket repair's committed validation artifacts and is never "
            "recomputed for V2 candidate selection."
        ),
        "control_is": "accepted bucket repair",
        "accepted_shadow_v1_sha": ACCEPTED_SHADOW_V1_SHA,
        "accepted_bucket_repair_sha": ACCEPTED_REPAIR_SHA,
        "source_production_sha": SOURCE_PRODUCTION_SHA,
        "pr_shadow_v1": PR_SHADOW_V1,
        "pr_bucket_repair": PR_BUCKET_REPAIR,
        "pull_requests_must_not_be_merged": True,
        "frozen_from_branch": git_branch(PROJECT_ROOT),
        "frozen_at_code_sha": git_sha(PROJECT_ROOT),
        "repair_candidate_name": candidate["candidate_name"],
        "repair_hyperparameters": candidate["hyperparameters"],
        "repair_factor_spec_hash": report["factor_spec_hash"],
        "repair_verdict": report["verdict"],
        "repair_gates_all_passed": bool(gates["all_passed"]),
        "training_seasons": report["training_seasons"],
        "validation_seasons": report["validation_seasons"],
        "simulations_per_game": report["simulations_per_game"],
        "games_simulated": report["games_simulated"],
        "seed": report["seed"],
        "latent_buckets": latent_control,
        "count_buckets": count_control,
        "latent_cross_player_rmse": residual["cross_player_rmse"],
        "count_cross_player_rmse": residual["count_space_cross_player_rmse"],
        "latent_rms_z_error": {
            model: float(value["rms_z_error"])
            for model, value in latent["by_model"].items()
        },
        "joint_events_by_legs": {
            legs_count: {
                "events": record["events"],
                "base_rate": record["base_rate"],
                **{
                    model: {
                        "brier": record[model]["brier"],
                        "log_loss": record[model]["log_loss"],
                    }
                    for model in (
                        "baseline_independence",
                        "baseline_production",
                        "candidate",
                    )
                },
            }
            for legs_count, record in legs.items()
        },
        "paired_repair_vs_v1": {
            "bootstrap_draws": paired["bootstrap_draws"],
            "by_legs": paired["by_legs"],
        },
        "same_player_contract": report["same_player_contract"],
        "stability": report["stability"],
        "marginal_preservation_candidate": report["marginal_preservation"][
            "candidate"
        ],
        "production_surface": report["production_surface"],
        "opponent_collateral_buckets": list(OPPONENT_BUCKETS),
        "same_team_target_buckets": list(SAME_TEAM_BUCKETS),
        "brief_figure_cross_check": {
            name: {
                "artifact": checks[name],
                "brief": expected,
                "tolerance": tolerance,
                "agrees": True,
            }
            for name, (expected, tolerance) in BRIEF_FIGURES.items()
        },
    }

    artifact_dir = Path(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "repair_control_baseline.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    console.print(f"\nwrote {out_path.relative_to(PROJECT_ROOT)}")
    console.print(
        "CONTROL_FROZEN=YES  "
        f"control_latent_rmse={checks['latent_cross_player_rmse']:.9f}  "
        f"control_count_rmse={checks['count_cross_player_rmse']:.9f}"
    )


if __name__ == "__main__":
    main()
