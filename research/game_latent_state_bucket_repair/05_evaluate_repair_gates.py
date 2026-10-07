#!/usr/bin/env python3
"""Evaluate the twelve repair acceptance gates against the immutable V1 control.

Every control number is read from ``v1_control_baseline.json``, which copies the
accepted V1 report verbatim and is never recomputed here. Every repair number
is read from the single untouched 2024-2025 validation run. The gate thresholds
are the ones stated in the brief and are not adjusted by anything this script
sees.

Two of the gates are measured by the paired comparison rather than by
differencing two reports, because at 3 and 4 legs the independent Monte Carlo
error on each run's Brier score is an order of magnitude wider than the 0.0005
tolerance: see 04_compare_joint_calibration.py.

SHADOW / RESEARCH ONLY. Reads committed artifacts and writes one JSON report.
Cannot promote anything.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPAIR_DIR = PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"

TARGET_BUCKETS = ("teammate_reb_reb", "teammate_ast_ast")
PROTECTED_BUCKETS = ("passer_ast_teammate_pts", "teammate_pts_reb")

MIN_TARGET_ERROR_REDUCTION = 0.30
MIN_TARGET_Z_IMPROVEMENT = 1.0
MAX_TARGET_OVERSHOOT_Z = 1.96
MAX_RMSE_DEGRADATION = 0.05
MAX_PROTECTED_Z_DEGRADATION = 1.0
MAX_SAME_PLAYER_BLOCK_DEVIATION = 1e-9
MAX_MULTI_LEG_BRIER_DEGRADATION = 0.0005

ALLOWED_REPAIR_PREFIXES = (
    "research/",
    "src/nba_prop_quant/research/",
    "tests/test_game_latent_state_shadow",
)


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def z_errors(implied: dict, observed: dict, se: dict) -> dict[str, float]:
    return {key: (implied[key] - observed[key]) / se[key] for key in observed}


def main() -> None:
    control = load(REPAIR_DIR / "v1_control_baseline.json")
    report = load(REPAIR_DIR / "validation_report.json")
    paired = load(REPAIR_DIR / "paired_joint_calibration.json")

    v1_latent = control["v1_latent_dependence"]
    observed = v1_latent["observed_buckets"]
    se = v1_latent["observed_bucket_se"]
    v1_implied = v1_latent["candidate_implied_buckets"]

    latent = report["latent_dependence"]
    repair_implied = latent["by_model"]["candidate"]["implied_buckets"]

    if latent["observed_buckets"] != observed:
        raise SystemExit(
            "the repair run's observed buckets differ from the control's; the "
            "two runs do not share the same held-out games"
        )

    v1_z = z_errors(v1_implied, observed, se)
    repair_z = z_errors(repair_implied, observed, se)

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

    # GATE 1: both targets improve materially on the untouched holdout.
    target_evidence: dict[str, dict] = {}
    gate1 = True
    for bucket in TARGET_BUCKETS:
        v1_error = abs(v1_implied[bucket] - observed[bucket])
        repair_error = abs(repair_implied[bucket] - observed[bucket])
        reduction = 1.0 - repair_error / v1_error
        z_improvement = abs(v1_z[bucket]) - abs(repair_z[bucket])
        improved = (
            reduction >= MIN_TARGET_ERROR_REDUCTION
            or z_improvement >= MIN_TARGET_Z_IMPROVEMENT
        )
        gate1 = gate1 and improved
        target_evidence[bucket] = {
            "observed": observed[bucket],
            "se": se[bucket],
            "v1_implied": v1_implied[bucket],
            "repair_implied": repair_implied[bucket],
            "v1_abs_error": v1_error,
            "repair_abs_error": repair_error,
            "abs_error_reduction_fraction": reduction,
            "v1_z": v1_z[bucket],
            "repair_z": repair_z[bucket],
            "abs_z_improvement": z_improvement,
            "improved": bool(improved),
        }
    record(
        1,
        "both target buckets improve materially on untouched 2024-2025",
        gate1,
        {
            "min_abs_error_reduction": MIN_TARGET_ERROR_REDUCTION,
            "min_abs_z_improvement": MIN_TARGET_Z_IMPROVEMENT,
            "buckets": target_evidence,
        },
    )

    # GATE 2: neither target overshoots materially in the opposite direction.
    overshoot = {
        bucket: max(0.0, repair_z[bucket]) for bucket in TARGET_BUCKETS
    }
    record(
        2,
        "neither target bucket materially overshoots in the opposite direction",
        all(value <= MAX_TARGET_OVERSHOOT_Z for value in overshoot.values()),
        {
            "limit_overshoot_z": MAX_TARGET_OVERSHOOT_Z,
            "overshoot_z": overshoot,
            "note": (
                "overshoot is the signed z-error in the positive direction; V1 "
                "undershot both targets, so any positive z is the direction "
                "this gate guards"
            ),
        },
    )

    # GATE 3: global latent-space bucket RMSE may not worsen by >5%.
    v1_rmse = v1_latent["candidate_rmse"]
    repair_rmse = latent["by_model"]["candidate"]["rmse"]
    record(
        3,
        "global latent-space bucket RMSE does not worsen by more than 5%",
        repair_rmse <= v1_rmse * (1.0 + MAX_RMSE_DEGRADATION),
        {
            "v1_rmse": v1_rmse,
            "repair_rmse": repair_rmse,
            "ratio": repair_rmse / v1_rmse,
            "limit_ratio": 1.0 + MAX_RMSE_DEGRADATION,
        },
    )

    # GATE 4: count-space bucket RMSE may not worsen by >5%.
    v1_count_rmse = control["v1_count_space_cross_player_rmse"]["candidate"]
    repair_count_rmse = report["residual_dependence"]["count_space_cross_player_rmse"][
        "candidate"
    ]
    record(
        4,
        "count-space bucket RMSE does not worsen by more than 5%",
        repair_count_rmse <= v1_count_rmse * (1.0 + MAX_RMSE_DEGRADATION),
        {
            "v1_rmse": v1_count_rmse,
            "repair_rmse": repair_count_rmse,
            "ratio": repair_count_rmse / v1_count_rmse,
            "limit_ratio": 1.0 + MAX_RMSE_DEGRADATION,
        },
    )

    # GATE 5: the two protected buckets may not each worsen by >1.0 z.
    protected: dict[str, dict] = {}
    gate5 = True
    for bucket in PROTECTED_BUCKETS:
        degradation = abs(repair_z[bucket]) - abs(v1_z[bucket])
        gate5 = gate5 and degradation <= MAX_PROTECTED_Z_DEGRADATION
        protected[bucket] = {
            "observed": observed[bucket],
            "v1_implied": v1_implied[bucket],
            "repair_implied": repair_implied[bucket],
            "v1_z": v1_z[bucket],
            "repair_z": repair_z[bucket],
            "abs_z_degradation": degradation,
        }
    record(
        5,
        "protected neighbour buckets do not each worsen by more than 1.0 z",
        gate5,
        {"limit_abs_z_degradation": MAX_PROTECTED_Z_DEGRADATION, "buckets": protected},
    )

    # GATE 6: same-player block deviation stays at machine precision.
    deviation = report["same_player_contract"]["max_block_deviation"]
    record(
        6,
        "same-player copula block preserved to 1e-9",
        deviation <= MAX_SAME_PLAYER_BLOCK_DEVIATION,
        {
            "repair_max_block_deviation": deviation,
            "v1_max_block_deviation": control["v1_same_player_contract"][
                "max_block_deviation"
            ],
            "limit": MAX_SAME_PLAYER_BLOCK_DEVIATION,
            "games_checked": report["same_player_contract"]["games_checked"],
        },
    )

    # GATE 7: every held-out covariance PSD and no numerical failures.
    stability = report["stability"]
    min_eig = stability["min_covariance_eigenvalue"]
    failures = stability["numerical_failures"]
    record(
        7,
        "every held-out covariance PSD within tolerance and no numerical failures",
        min_eig > 0.0 and failures == 0,
        {
            "repair_min_covariance_eigenvalue": min_eig,
            "v1_min_covariance_eigenvalue": control["v1_stability"][
                "min_covariance_eigenvalue"
            ],
            "repair_numerical_failures": failures,
            "v1_numerical_failures": control["v1_stability"]["numerical_failures"],
            "games_tested": stability["games_tested"],
            "failure_detail": stability["failure_detail"],
        },
    )

    # GATE 8: marginal Gate A still passes.
    gate_a = next(
        entry for entry in report["acceptance_gates"] if entry["gate"] == "A"
    )
    record(
        8,
        "marginal preservation Gate A still passes unchanged",
        gate_a["passed"],
        {
            "gate_a": gate_a,
            "v1_gate_a": next(
                entry for entry in control["v1_acceptance_gates"]
                if entry["gate"] == "A"
            ),
        },
    )

    # GATES 9 and 10: paired multi-leg Brier.
    legs = paired["by_legs"]
    two = legs["2"]["repair_minus_v1"]
    record(
        9,
        "2-leg Brier does not materially worsen against accepted V1",
        two["ci95"][0] <= 0.0 <= two["ci95"][1]
        or two["delta"] <= MAX_MULTI_LEG_BRIER_DEGRADATION,
        {
            "brier_v1": legs["2"]["brier_v1"],
            "brier_repair": legs["2"]["brier_repair"],
            "paired_delta": two["delta"],
            "paired_delta_ci95": two["ci95"],
            "indistinguishable_from_zero": bool(
                two["ci95"][0] <= 0.0 <= two["ci95"][1]
            ),
        },
    )

    multi: dict[str, dict] = {}
    gate10 = True
    for count in ("3", "4"):
        delta = legs[count]["repair_minus_v1"]
        within = delta["delta"] <= MAX_MULTI_LEG_BRIER_DEGRADATION
        gate10 = gate10 and within
        multi[count] = {
            "brier_v1": legs[count]["brier_v1"],
            "brier_repair": legs[count]["brier_repair"],
            "paired_delta": delta["delta"],
            "paired_delta_ci95": delta["ci95"],
            "within_tolerance": bool(within),
            "v1_minus_independence": legs[count]["v1_minus_independence"],
            "repair_minus_independence": legs[count]["repair_minus_independence"],
        }
    # The brief's extra 4-leg constraint: V1 already runs a small regression
    # against independence there, and the repair may not deepen it.
    four = multi["4"]
    four_leg_regression_not_worsened = (
        four["repair_minus_independence"]["delta"]
        <= four["v1_minus_independence"]["delta"]
        + MAX_MULTI_LEG_BRIER_DEGRADATION
    )
    gate10 = gate10 and four_leg_regression_not_worsened
    record(
        10,
        "3-leg and 4-leg Brier within 0.0005 of V1, and V1's 4-leg regression not deepened",
        gate10,
        {
            "limit_degradation": MAX_MULTI_LEG_BRIER_DEGRADATION,
            "by_legs": multi,
            "four_leg_regression_not_worsened": bool(
                four_leg_regression_not_worsened
            ),
        },
    )

    # GATE 11: no pairwise parameters.
    record(
        11,
        "pairwise_parameter_count remains 0",
        stability["pairwise_parameter_count"] == 0,
        {
            "repair_pairwise_parameter_count": stability["pairwise_parameter_count"],
            "v1_pairwise_parameter_count": control["v1_stability"][
                "pairwise_parameter_count"
            ],
            "frozen_candidate_parameter_counts": load(
                REPAIR_DIR / "bucket_repair_candidate.json"
            )["parameter_counts"],
        },
    )

    # GATE 12: production untouched. Two independent checks, because the
    # driver's own surface audit and a git diff against the parent V1 head can
    # fail in different ways: the first catches a modified protected source,
    # the second catches a repair file written outside the research namespace.
    surface = report["production_surface"]
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{control['parent_shadow_v1_sha']}..HEAD"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    offenders = sorted(
        set(surface["modified_paths"])
        | {
            path
            for path in changed
            if not path.startswith(ALLOWED_REPAIR_PREFIXES)
        }
    )
    record(
        12,
        "production remains untouched",
        not offenders,
        {
            "parent_shadow_v1_sha": control["parent_shadow_v1_sha"],
            "driver_surface_modified_paths": surface["modified_paths"],
            "changed_paths_since_parent": changed,
            "allowed_repair_prefixes": list(ALLOWED_REPAIR_PREFIXES),
            "offenders": offenders,
        },
    )

    failed = [entry for entry in gates if not entry["passed"]]
    verdict = (
        "BUCKET REPAIR ACCEPTED FOR SHADOW V2 DESIGN."
        if not failed
        else "BUCKET REPAIR REJECTED: "
        + "; ".join(
            f"GATE {entry['gate']} ({entry['name']}) evidence="
            f"{json.dumps(entry['evidence'], sort_keys=True)}"
            for entry in failed
        )
    )

    out = {
        "scope": "repair acceptance gates, untouched 2024-2025 holdout",
        "parent_shadow_v1_sha": control["parent_shadow_v1_sha"],
        "repair_factor_spec_hash": report["factor_spec_hash"],
        "gates": gates,
        "all_passed": not failed,
        "verdict": verdict,
    }
    path = REPAIR_DIR / "repair_gates.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    for entry in gates:
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"REPAIR GATE {entry['gate']:>2} ({entry['name']}): {status}")
    print()
    print(verdict if not failed else verdict[:400])
    print()
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
