#!/usr/bin/env python
"""Evaluate the thirteen Shadow V2 acceptance gates.

SHADOW / RESEARCH ONLY. Reads committed artifacts and writes one JSON report.
Cannot promote anything.

Every control number is read from ``repair_control_baseline.json``, which
copies the accepted bucket repair's committed report verbatim and is never
recomputed here. Every candidate number is read from the single untouched
2024-2025 validation run. The thresholds below are declared in code before
that run produced the numbers they are applied to, and no gate is relaxed to
obtain a pass.

Two structural gates are not statistical at all. The cross-team isolation
gate reads an algebraic identity -- the symmetric subspace cancels from
``A - B`` exactly -- so its tolerance is float64 noise rather than a
confidence interval. The parameter-count gate reads the frozen spec, not the
run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_DIR = PROJECT_ROOT / "research" / "game_latent_state_v2"

#: The two buckets the accepted repair exists to repair. V2 may not give back
#: what the repair won on them.
REPAIR_TARGET_BUCKETS = ("teammate_reb_reb", "teammate_ast_ast")

#: The bucket the count-space work targets.
PRIMARY_COUNT_BUCKET = "passer_ast_teammate_pts"

#: The opponent buckets the repair moved as collateral, and which the
#: symmetric subspace exists to leave alone.
PROTECTED_CROSS_BUCKETS = (
    "opponent_ast_ast",
    "opponent_pts_reb",
    "opponent_fg3m_reb",
)

# ---------------------------------------------------------------------
# pre-registered thresholds
# ---------------------------------------------------------------------

#: GATE 1. The temporal work does not change the point estimate -- the
#: walk-forward screen selected the pooled estimator -- so what it must not do
#: is make the primary unstable bucket worse, and what it must add is a
#: calibrated predictive SD where the static fit reported none.
MAX_TEMPORAL_Z_DEGRADATION = 0.25

#: GATE 2. The cross-team identity. Not a tolerance on an estimate.
CROSS_TEAM_IDENTITY_TOLERANCE = 1e-12

#: GATE 3. The repair's target buckets may not give back more than this
#: fraction of the absolute error the repair achieved.
MAX_TARGET_ERROR_GIVEBACK = 0.10

#: GATE 4. The count-space error on the primary count bucket must improve by
#: at least this fraction against the control.
MIN_COUNT_IMPROVEMENT = 0.20

#: GATE 5. The role layer must cut support-weighted role-conditioned RMSE by
#: at least this fraction against the candidate's own role-blind twin, break
#: no cell the twin kept inside the z limit, and make no supported cell worse
#: by more than the declared relative limit.
MIN_ROLE_IMPROVEMENT = 0.20
ROLE_CELL_REGRESSION_LIMIT = 0.05

#: GATES 6 and 7. Global latent and count-space bucket RMSE may not worsen
#: the control's by more than 3%.
MAX_GLOBAL_RATIO = 1.03

#: GATE 8.
MAX_SAME_PLAYER_BLOCK_DEVIATION = 1e-9

#: GATE 11. Paired, game-clustered log-loss degradation per leg count. One
#: order of magnitude tighter than the framework's Brier tolerance because a
#: paired comparison on shared games is that much sharper than two
#: independent runs.
MAX_LOG_LOSS_DEGRADATION = 0.0005

#: GATE 13. Nothing in the merge path may carry a blob this large. The
#: research residual dataset is 66 MB and is deliberately never committed.
MAX_TRACKED_BLOB_BYTES = 10 * 1024 * 1024

ALLOWED_RESEARCH_PREFIXES = (
    "research/",
    "src/nba_prop_quant/research/",
    "tests/test_game_latent_state_shadow",
    ".gitignore",
)


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def tracked_blob_sizes() -> list[tuple[str, int]]:
    """Every tracked blob and its size in bytes, largest first."""
    listing = git("ls-tree", "-r", "-l", "HEAD")
    out: list[tuple[str, int]] = []
    for line in listing.splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) < 4 or fields[1] != "blob" or fields[3] == "-":
            continue
        out.append((path, int(fields[3])))
    return sorted(out, key=lambda entry: entry[1], reverse=True)


def main() -> None:
    control = load(V2_DIR / "repair_control_baseline.json")
    report = load(V2_DIR / "validation_report.json")
    freeze = load(V2_DIR / "shadow_v2_candidate.json")
    screening = load(V2_DIR / "inner_screening.json")
    temporal = load(V2_DIR / "temporal_diagnostic.json")
    bridge = load(V2_DIR / "count_bridge.json")

    latent = report["latent_dependence"]
    observed = latent["observed_buckets"]
    se = latent["observed_bucket_se"]
    candidate_implied = latent["by_model"]["candidate"]["implied_buckets"]
    repair_implied = latent["by_model"]["repair"]["implied_buckets"]
    v1_implied = latent["by_model"]["v1"]["implied_buckets"]

    control_buckets = control["latent_buckets"]

    def z(implied: dict, bucket: str) -> float:
        return (implied[bucket] - observed[bucket]) / se[bucket]

    def abs_error(implied: dict, bucket: str) -> float:
        return abs(implied[bucket] - observed[bucket])

    gates: list[dict] = []

    def record(number: int, name: str, passed: bool, evidence: dict) -> None:
        gates.append(
            {
                "gate": number,
                "name": name,
                "passed": bool(passed),
                "evidence": evidence,
            }
        )

    # ---- GATE 0 is not a gate: the run has to be paired at all ------
    pairing = report["control_pairing"]
    if not pairing["paired"]:
        raise SystemExit(
            "the validation run's observed held-out buckets differ from the "
            f"control's by {pairing['max_observed_bucket_drift']}; the two "
            "runs do not share the same held-out games, so no paired gate "
            "below is meaningful"
        )

    # ---- GATE 1: the unstable bucket -------------------------------
    calibration = temporal["uncertainty_calibration"]
    bucket = "teammate_ast_ast"
    degradation = abs(z(candidate_implied, bucket)) - abs(
        control_buckets[bucket]["repair_z"]
    )
    inflation = float(calibration["pooled"]["inflation"])
    record(
        1,
        "the temporally unstable bucket is not degraded and now carries a "
        "calibrated predictive SD",
        degradation <= MAX_TEMPORAL_Z_DEGRADATION and inflation > 1.0,
        {
            "bucket": bucket,
            "selected_treatment": temporal["selected_treatment_for_primary_bucket"],
            "observed": observed[bucket],
            "observed_se": se[bucket],
            "control_z": control_buckets[bucket]["repair_z"],
            "candidate_z": z(candidate_implied, bucket),
            "abs_z_degradation": degradation,
            "limit_abs_z_degradation": MAX_TEMPORAL_Z_DEGRADATION,
            "predictive_sd_inflation": inflation,
            "note": (
                "the walk-forward screen in 01_temporal_diagnostic.py selected "
                "the pooled estimator on point accuracy, so the improvement "
                "this gate reads is the calibrated predictive SD the static "
                "fit does not report at all"
            ),
        },
    )

    # ---- GATE 2: cross-team isolation ------------------------------
    identity = float(report["structural_identities"]["cross_team_blocks_identical_to_v1_path"])
    protected: dict[str, dict] = {}
    cross_unchanged = True
    for name in PROTECTED_CROSS_BUCKETS:
        gap = abs(candidate_implied[name] - v1_implied[name])
        cross_unchanged = cross_unchanged and gap <= CROSS_TEAM_IDENTITY_TOLERANCE
        protected[name] = {
            "observed": observed[name],
            "v1_implied": v1_implied[name],
            "repair_implied": repair_implied[name],
            "candidate_implied": candidate_implied[name],
            "v1_z": z(v1_implied, name),
            "repair_z": z(repair_implied, name),
            "candidate_z": z(candidate_implied, name),
            "candidate_minus_v1": gap,
        }
    record(
        2,
        "the same-team repair is algebraically unable to move the cross-team "
        "block, and does not",
        identity <= CROSS_TEAM_IDENTITY_TOLERANCE and cross_unchanged,
        {
            "max_abs_cross_team_block_difference_from_v1_path": identity,
            "tolerance": CROSS_TEAM_IDENTITY_TOLERANCE,
            "protected_buckets": protected,
            "symmetric_subspace_rank": report["structural_identities"][
                "symmetric_subspace_rank"
            ],
            "note": (
                "the repair moved these three buckets as collateral because it "
                "coupled one estimator and one raised rank across both blocks; "
                "V2 reaches the same-team block through a subspace that "
                "cancels from A - B, so the opponent buckets are V1's exactly"
            ),
        },
    )

    # ---- GATE 3: the repair's targets are not given back -----------
    targets: dict[str, dict] = {}
    targets_held = True
    for name in REPAIR_TARGET_BUCKETS:
        control_error = abs(control_buckets[name]["repair_fitted"] - observed[name])
        candidate_error = abs_error(candidate_implied, name)
        giveback = (
            (candidate_error - control_error) / control_error
            if control_error > 0
            else 0.0
        )
        held = giveback <= MAX_TARGET_ERROR_GIVEBACK
        targets_held = targets_held and held
        targets[name] = {
            "observed": observed[name],
            "control_implied": control_buckets[name]["repair_fitted"],
            "candidate_implied": candidate_implied[name],
            "v1_implied": v1_implied[name],
            "control_abs_error": control_error,
            "candidate_abs_error": candidate_error,
            "error_giveback_fraction": giveback,
            "control_z": control_buckets[name]["repair_z"],
            "candidate_z": z(candidate_implied, name),
            "held": bool(held),
        }
    record(
        3,
        "both repaired same-team buckets are held at the control's accuracy",
        targets_held,
        {"limit_error_giveback": MAX_TARGET_ERROR_GIVEBACK, "buckets": targets},
    )

    # ---- GATE 4: the count-space attenuation --------------------------
    count = report["residual_dependence"]
    count_observed = count["observed_buckets"][PRIMARY_COUNT_BUCKET]
    candidate_count = count["by_model"]["candidate"]["buckets"][PRIMARY_COUNT_BUCKET]
    control_count = control["count_buckets"][PRIMARY_COUNT_BUCKET]
    # The control's absolute error as the control measured it, against its own
    # observed moment. The two observed moments are the same quantity on the
    # same games, so the drift between them is reported alongside and belongs
    # in the evidence rather than in the arithmetic.
    control_count_error = abs(control_count["repair_absolute_error"])
    candidate_count_error = abs(candidate_count - count_observed)
    improvement = (
        1.0 - candidate_count_error / control_count_error
        if control_count_error > 0
        else 0.0
    )
    record(
        4,
        f"{PRIMARY_COUNT_BUCKET} count-space error improves by at least "
        f"{MIN_COUNT_IMPROVEMENT:.0%}",
        improvement >= MIN_COUNT_IMPROVEMENT,
        {
            "bucket": PRIMARY_COUNT_BUCKET,
            "observed_count_space": count_observed,
            "control_observed_count_space": control_count["observed"],
            "observed_count_space_drift": abs(
                count_observed - control_count["observed"]
            ),
            "control_simulated": control_count["repair_fitted"],
            "candidate_simulated": candidate_count,
            "control_abs_error": control_count_error,
            "candidate_abs_error": candidate_count_error,
            "improvement_fraction": improvement,
            "threshold": MIN_COUNT_IMPROVEMENT,
            "bridge_weight": freeze["hyperparameters"]["bridge_weight"],
            "bridge_method": bridge["bridge_method"],
            "note": (
                "the latent target for this bucket is the correlation the "
                "training count-space moment implies through the inverted "
                "discrete Gaussian-copula bridge, so the quantity measured "
                "here is the one the fit targeted"
            ),
        },
    )

    # ---- GATE 5: the role layer ---------------------------------------
    role = latent["role_conditioned"]["candidate"]
    improvement_fraction = float(role.get("role_rmse_improvement_fraction", 0.0))
    newly = list(role.get("newly_exceeding_cells", []))
    regressed = list(role.get("regressed_cells", []))
    record(
        5,
        "the role-conditioned deviation improves supported role cells without "
        "breaking any",
        (
            float(role["measurable"]) == 1.0
            and improvement_fraction >= MIN_ROLE_IMPROVEMENT
            and not newly
            and not regressed
        ),
        {
            "role_cells_measured": role["role_cells"],
            "role_conditioned_rmse": role["role_conditioned_rmse"],
            "role_blind_twin_rmse": role.get("pooled_only_role_conditioned_rmse"),
            "improvement_fraction": improvement_fraction,
            "threshold": MIN_ROLE_IMPROVEMENT,
            "newly_exceeding_cells": newly,
            "regressed_cells": regressed,
            "worst_cell_rmse_regression": role["worst_cell_rmse_regression"],
            "cell_regression_limit": ROLE_CELL_REGRESSION_LIMIT,
            "worst_role_abs_z": role["worst_role_abs_z"],
            "worst_role_blind_abs_z": role["worst_pooled_abs_z"],
            "v1_role_conditioned_rmse": latent["role_conditioned"]["v1"][
                "role_conditioned_rmse"
            ],
            "repair_role_conditioned_rmse": latent["role_conditioned"]["repair"][
                "role_conditioned_rmse"
            ],
            "role_quadratic_share": report["structural_identities"][
                "role_quadratic_share"
            ],
        },
    )

    # ---- GATE 6: global latent RMSE ------------------------------------
    control_latent_rmse = control["latent_cross_player_rmse"]["candidate"]
    candidate_latent_rmse = latent["by_model"]["candidate"]["rmse"]
    record(
        6,
        "global latent-space bucket RMSE does not worsen by more than 3%",
        candidate_latent_rmse <= control_latent_rmse * MAX_GLOBAL_RATIO,
        {
            "control_rmse": control_latent_rmse,
            "candidate_rmse": candidate_latent_rmse,
            "v1_rmse": latent["by_model"]["v1"]["rmse"],
            "independence_rmse": latent["by_model"]["baseline_independence"]["rmse"],
            "ratio": candidate_latent_rmse / control_latent_rmse,
            "limit_ratio": MAX_GLOBAL_RATIO,
        },
    )

    # ---- GATE 7: count-space RMSE --------------------------------------
    control_count_rmse = control["count_cross_player_rmse"]["candidate"]
    candidate_count_rmse = count["count_space_cross_player_rmse"]["candidate"]
    record(
        7,
        "count-space bucket RMSE does not worsen by more than 3%",
        candidate_count_rmse <= control_count_rmse * MAX_GLOBAL_RATIO,
        {
            "control_rmse": control_count_rmse,
            "candidate_rmse": candidate_count_rmse,
            "by_model": count["count_space_cross_player_rmse"],
            "ratio": candidate_count_rmse / control_count_rmse,
            "limit_ratio": MAX_GLOBAL_RATIO,
        },
    )

    # ---- GATE 8: the same-player contract ------------------------------
    deviation = report["same_player_contract"]["max_block_deviation"]
    record(
        8,
        "the incumbent same-player copula block is preserved to 1e-9",
        deviation <= MAX_SAME_PLAYER_BLOCK_DEVIATION,
        {
            "candidate_max_block_deviation": deviation,
            "control_max_block_deviation": control["same_player_contract"][
                "max_block_deviation"
            ],
            "limit": MAX_SAME_PLAYER_BLOCK_DEVIATION,
            "games_checked": report["same_player_contract"]["games_checked"],
        },
    )

    # ---- GATE 9: PSD and numerical stability ---------------------------
    stability = report["stability"]
    record(
        9,
        "every held-out covariance is PSD and no game failed numerically",
        stability["min_covariance_eigenvalue"] > 0.0
        and stability["numerical_failures"] == 0,
        {
            "candidate_min_covariance_eigenvalue": stability[
                "min_covariance_eigenvalue"
            ],
            "control_min_covariance_eigenvalue": control["stability"][
                "min_covariance_eigenvalue"
            ],
            "numerical_failures": stability["numerical_failures"],
            "failure_detail": stability["failure_detail"],
            "games_tested": stability["games_tested"],
            "max_dimensions": stability["max_dimensions"],
        },
    )

    # ---- GATE 10: marginal preservation --------------------------------
    gate_a = next(
        entry for entry in report["acceptance_gates"] if entry["gate"] == "A"
    )
    record(
        10,
        "marginal preservation Gate A still passes unchanged",
        gate_a["passed"],
        {
            "gate_a": gate_a,
            "control_gate_a": control["marginal_preservation_candidate"],
        },
    )

    # ---- GATE 11: paired log loss per leg count ------------------------
    legs = report["joint_events"]["by_legs"]
    log_loss_detail: dict[str, dict] = {}
    log_loss_ok = True
    for count_key in sorted(legs):
        paired = legs[count_key]["paired"]
        entry: dict[str, object] = {}
        for comparison in ("candidate_minus_v1", "candidate_minus_repair"):
            delta = paired[comparison]["log_loss"]
            within = (
                delta["delta"] <= MAX_LOG_LOSS_DEGRADATION
                or delta["indistinguishable_from_zero"]
            )
            log_loss_ok = log_loss_ok and within
            entry[comparison] = {**delta, "within_tolerance": bool(within)}
        entry["candidate_minus_independence"] = paired[
            "candidate_minus_baseline_independence"
        ]["log_loss"]
        entry["log_loss_by_model"] = {
            name: legs[count_key][name]["log_loss"]
            for name in report["models"]
            if name in legs[count_key]
        }
        entry["log_loss_numerically_stable"] = all(
            legs[count_key][name]["log_loss_numerically_stable"]
            for name in report["models"]
            if name in legs[count_key]
        )
        log_loss_ok = log_loss_ok and bool(entry["log_loss_numerically_stable"])
        log_loss_detail[count_key] = entry
    record(
        11,
        "paired game-clustered log loss does not degrade at any leg count",
        log_loss_ok,
        {
            "limit_degradation": MAX_LOG_LOSS_DEGRADATION,
            "by_legs": log_loss_detail,
            "note": (
                "a delta whose game-clustered CI95 covers zero passes on that "
                "ground: the paired difference is then not resolvable, which "
                "is the honest reading of a tolerance narrower than the noise"
            ),
        },
    )

    # ---- GATE 12: no pairwise or player-indexed parameters -------------
    counts = freeze["parameter_counts"]
    record(
        12,
        "no pairwise and no player-indexed parameters, so unseen players and "
        "unseen roles need no refit",
        counts["pairwise"] == 0
        and counts["player_indexed"] == 0
        and stability["pairwise_parameter_count"] == 0
        and stability["unseen_player_simulation_ok"] is True,
        {
            "frozen_parameter_counts": counts,
            "run_pairwise_parameter_count": stability["pairwise_parameter_count"],
            "run_player_indexed_parameter_count": stability.get(
                "player_indexed_parameter_count"
            ),
            "unseen_player_simulation_ok": stability["unseen_player_simulation_ok"],
            "unseen_role_fallback": (
                "a role the fit never saw gets h = 0, which is the pooled "
                "loading; see covariance.SharedFactorLoadings.offset_for_role"
            ),
        },
    )

    # ---- GATE 13: the research boundary --------------------------------
    # Four independent checks. The driver's own surface audit catches a
    # modified protected source; the diff against the accepted repair head
    # catches a file written outside the research namespace; the blob audit
    # catches a large artifact that would follow the branch into production
    # history; and the freeze record catches a candidate retuned after the
    # holdout was opened.
    surface = report["production_surface"]
    changed = git(
        "diff", "--name-only", f"{freeze['parent_bucket_repair_sha']}..HEAD"
    ).split()
    offenders = sorted(
        set(surface["modified_paths"])
        | {path for path in changed if not path.startswith(ALLOWED_RESEARCH_PREFIXES)}
    )
    blobs = tracked_blob_sizes()
    oversized = [
        {"path": path, "bytes": size}
        for path, size in blobs
        if size > MAX_TRACKED_BLOB_BYTES
    ]
    record(
        13,
        "production untouched, nothing retuned after the freeze, and no large "
        "artifact in the merge path",
        not offenders
        and not oversized
        and freeze["holdout_used_for_selection"] is False
        and freeze["production_integration_started"] is False
        and screening["holdout_used_for_selection"] is False
        and freeze["factor_spec_hash"] == report["factor_spec_hash"],
        {
            "parent_bucket_repair_sha": freeze["parent_bucket_repair_sha"],
            "parent_shadow_v1_sha": freeze["parent_shadow_v1_sha"],
            "driver_surface_modified_paths": surface["modified_paths"],
            "changed_paths_since_parent": changed,
            "allowed_research_prefixes": list(ALLOWED_RESEARCH_PREFIXES),
            "offenders": offenders,
            "largest_tracked_blobs": [
                {"path": path, "bytes": size} for path, size in blobs[:5]
            ],
            "max_tracked_blob_bytes": MAX_TRACKED_BLOB_BYTES,
            "oversized_tracked_blobs": oversized,
            "holdout_used_for_selection": freeze["holdout_used_for_selection"],
            "production_integration_started": freeze[
                "production_integration_started"
            ],
            "frozen_spec_hash_matches_validated_spec": (
                freeze["factor_spec_hash"] == report["factor_spec_hash"]
            ),
        },
    )

    failed = [entry for entry in gates if not entry["passed"]]
    verdict = (
        "SHADOW V2 ACCEPTED FOR PRODUCTION-INTEGRATION DESIGN."
        if not failed
        else "SHADOW V2 REJECTED: "
        + "; ".join(
            f"GATE {entry['gate']} ({entry['name']}) evidence="
            f"{json.dumps(entry['evidence'], sort_keys=True)}"
            for entry in failed
        )
    )

    out = {
        "scope": "Shadow V2 acceptance gates, untouched 2024-2025 holdout",
        "control_is": control["control_is"],
        "parent_shadow_v1_sha": freeze["parent_shadow_v1_sha"],
        "parent_bucket_repair_sha": freeze["parent_bucket_repair_sha"],
        "candidate_name": freeze["candidate_name"],
        "candidate_factor_spec_hash": report["factor_spec_hash"],
        "framework_verdict": report["framework_verdict"],
        "gates": gates,
        "all_passed": not failed,
        "verdict": verdict,
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
    }
    path = V2_DIR / "v2_gates.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    for entry in gates:
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"V2 GATE {entry['gate']:>2} ({entry['name']}): {status}")
    print()
    print(verdict if not failed else verdict[:600])
    print()
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
