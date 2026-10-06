#!/usr/bin/env python
"""Evaluate the fourteen final-upstream gates against the accepted repair.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The control is the accepted bucket repair, re-run through the *same*
unmodified V1 validation driver at the same seed, simulation count and
residual dataset as the candidate, so every number on both sides comes from
one code path and the two runs are paired event for event. Gates 9 and 10 read
the paired comparison rather than the difference of two reports, because at
three and four legs the independent Monte Carlo error on each run's Brier is an
order of magnitude wider than the 0.0005 tolerance.

No threshold here is adjusted by anything this script sees, and nothing here
can change the model: it reads committed artifacts and writes one report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.artifacts import git_sha

ARTIFACT_DIR = PROJECT_ROOT / "research" / "final_upstream_remediation"

TARGET_BUCKETS = ("teammate_ast_ast", "teammate_reb_reb")

#: The three buckets the brief protects for item 3, plus the two the accepted
#: repair already protected. A cross-team prior change and a transmission
#: bridge can only reach these through the shared blocks, so they are where a
#: regression would show first.
PROTECTED_BUCKETS = (
    "opponent_ast_ast",
    "opponent_pts_reb",
    "opponent_fg3m_reb",
    "passer_ast_teammate_pts",
    "teammate_pts_reb",
)

MAX_TARGET_Z_DEGRADATION = 0.25
MAX_TARGET_OVERSHOOT_Z = 1.96
MAX_RMSE_DEGRADATION = 0.05
MAX_PROTECTED_Z_DEGRADATION = 1.0
MAX_SAME_PLAYER_BLOCK_DEVIATION = 1e-9
MAX_BRIER_DEGRADATION = 0.0005
MAX_ROLE_NEUTRALITY_DEVIATION = 1e-10

#: The global inflation factor the V2 structural round rejected. Reviving it
#: anywhere in the carried spec would reintroduce a dial that was fitted to a
#: single pooled target, so the gate looks for the number itself.
FORBIDDEN_INFLATION_FACTOR = 1.7659

ALLOWED_RESEARCH_PREFIXES = (
    "research/",
    "src/nba_prop_quant/research/",
    "tests/test_game_latent_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--control-report",
        type=Path,
        default=ARTIFACT_DIR / "control_validation_report.json",
        help="validation_report.json from the paired control re-run",
    )
    parser.add_argument(
        "--production-ref", default="origin/production/wizardofodds-integration"
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def z_errors(implied: dict, observed: dict, se: dict) -> dict[str, float]:
    return {key: (implied[key] - observed[key]) / se[key] for key in observed}


def changed_paths(production_ref: str) -> list[str]:
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", production_ref],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{merge_base}..HEAD"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return sorted(diff)


def inflation_factor_present(paths: list[str]) -> list[str]:
    """Any carried file that mentions the rejected global inflation factor."""
    needle = f"{FORBIDDEN_INFLATION_FACTOR}"
    offenders: list[str] = []
    for relative in paths:
        path = PROJECT_ROOT / relative
        if not path.exists() or path.suffix in {".parquet", ".pkl", ".pickle"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if needle in text:
            offenders.append(relative)
    return offenders


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_root)

    frozen = load(artifact_dir / "frozen_spec.json")
    spec = load(artifact_dir / "factor_spec.json")
    diagnostics = load(artifact_dir / "covariance_diagnostics.json")
    report = load(artifact_dir / "validation_report.json")
    control = load(args.control_report)
    paired = load(artifact_dir / "paired_joint_calibration.json")
    inner = load(artifact_dir / "inner_selection.json")
    temperature = load(artifact_dir / "dependence_temperature.json")

    latent = report["latent_dependence"]
    control_latent = control["latent_dependence"]
    observed = latent["observed_buckets"]
    se = latent["observed_bucket_se"]
    if control_latent["observed_buckets"] != observed:
        raise SystemExit(
            "the two runs report different observed buckets, so they do not "
            "share the same held-out games"
        )

    candidate_implied = latent["by_model"]["candidate"]["implied_buckets"]
    control_implied = control_latent["by_model"]["candidate"]["implied_buckets"]
    candidate_z = z_errors(candidate_implied, observed, se)
    control_z = z_errors(control_implied, observed, se)

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

    # GATE 1: the two target buckets the repair fixed stay fixed. The brief's
    # item 1 asks for a robust season treatment, and the pre-2024 folds chose
    # the pooled one, so the gate is that nothing was given back.
    target_evidence: dict[str, dict] = {}
    gate1 = True
    for bucket in TARGET_BUCKETS:
        degradation = abs(candidate_z[bucket]) - abs(control_z[bucket])
        held = degradation <= MAX_TARGET_Z_DEGRADATION
        gate1 = gate1 and held
        target_evidence[bucket] = {
            "observed": observed[bucket],
            "se": se[bucket],
            "control_implied": control_implied[bucket],
            "candidate_implied": candidate_implied[bucket],
            "control_abs_z": abs(control_z[bucket]),
            "candidate_abs_z": abs(candidate_z[bucket]),
            "abs_z_degradation": degradation,
            "within_tolerance": bool(held),
        }
    record(
        1,
        "the repaired target buckets are not given back",
        gate1,
        {
            "max_abs_z_degradation_allowed": MAX_TARGET_Z_DEGRADATION,
            "by_bucket": target_evidence,
        },
    )

    # GATE 2: no target bucket overshoots the observed value materially in the
    # opposite direction.
    overshoot = {
        bucket: candidate_z[bucket]
        for bucket in TARGET_BUCKETS
    }
    record(
        2,
        "no target bucket overshoots the observed value",
        all(abs(value) <= MAX_TARGET_OVERSHOOT_Z for value in overshoot.values()),
        {"max_abs_z_allowed": MAX_TARGET_OVERSHOOT_Z, "signed_z": overshoot},
    )

    # GATE 3: global latent-space bucket RMSE.
    latent_ratio = (
        latent["by_model"]["candidate"]["rmse"]
        / control_latent["by_model"]["candidate"]["rmse"]
    )
    record(
        3,
        "global latent-space bucket RMSE does not worsen materially",
        latent_ratio <= 1.0 + MAX_RMSE_DEGRADATION,
        {
            "control_rmse": control_latent["by_model"]["candidate"]["rmse"],
            "candidate_rmse": latent["by_model"]["candidate"]["rmse"],
            "ratio": latent_ratio,
            "max_ratio_allowed": 1.0 + MAX_RMSE_DEGRADATION,
        },
    )

    # GATE 4: count-space cross-player RMSE, which is the space the bridge in
    # item 4 reads its second source from.
    count_candidate = report["residual_dependence"][
        "count_space_cross_player_rmse"
    ]["candidate"]
    count_control = control["residual_dependence"][
        "count_space_cross_player_rmse"
    ]["candidate"]
    count_ratio = count_candidate / count_control
    record(
        4,
        "count-space cross-player RMSE does not worsen materially",
        count_ratio <= 1.0 + MAX_RMSE_DEGRADATION,
        {
            "control_rmse": count_control,
            "candidate_rmse": count_candidate,
            "ratio": count_ratio,
            "max_ratio_allowed": 1.0 + MAX_RMSE_DEGRADATION,
        },
    )

    # GATE 5: the protected buckets.
    protected: dict[str, dict] = {}
    gate5 = True
    for bucket in PROTECTED_BUCKETS:
        degradation = abs(candidate_z[bucket]) - abs(control_z[bucket])
        held = degradation <= MAX_PROTECTED_Z_DEGRADATION
        gate5 = gate5 and held
        protected[bucket] = {
            "observed": observed[bucket],
            "control_abs_z": abs(control_z[bucket]),
            "candidate_abs_z": abs(candidate_z[bucket]),
            "abs_z_degradation": degradation,
            "within_tolerance": bool(held),
        }
    record(
        5,
        "no protected cross-team bucket worsens beyond one z",
        gate5,
        {
            "max_abs_z_degradation_allowed": MAX_PROTECTED_Z_DEGRADATION,
            "by_bucket": protected,
        },
    )

    # GATE 6: same-player blocks are the incumbent's, at machine precision.
    deviation = report["same_player_contract"]["max_block_deviation"]
    record(
        6,
        "every same-player block stays pinned at the incumbent's",
        deviation <= MAX_SAME_PLAYER_BLOCK_DEVIATION,
        {
            "max_block_deviation": deviation,
            "tolerance": MAX_SAME_PLAYER_BLOCK_DEVIATION,
            "games_checked": report["same_player_contract"]["games_checked"],
        },
    )

    # GATE 7: PSD and no numerical failures on every held-out game simulated.
    stability = report["stability"]
    record(
        7,
        "every held-out game covariance is PSD with no numerical failure",
        stability["min_covariance_eigenvalue"] > 0.0
        and stability["numerical_failures"] == 0,
        {
            "min_covariance_eigenvalue": stability["min_covariance_eigenvalue"],
            "numerical_failures": stability["numerical_failures"],
            "games_tested": stability["games_tested"],
            "max_dimensions": stability["max_dimensions"],
        },
    )

    # GATE 8: the marginal-preservation gate the accepted lineage uses, still
    # passing on the candidate's own run.
    gate_a = next(
        entry for entry in report["acceptance_gates"] if entry["gate"] == "A"
    )
    control_gate_a = next(
        entry for entry in control["acceptance_gates"] if entry["gate"] == "A"
    )
    record(
        8,
        "marginal calibration is preserved (lineage gate A)",
        bool(gate_a["passed"]),
        {"candidate_gate_a": gate_a, "control_gate_a": control_gate_a},
    )

    # GATES 9 and 10: paired multi-leg Brier.
    two_leg = paired["by_legs"]["2"]
    record(
        9,
        "2-leg joint calibration does not worsen beyond tolerance (paired)",
        bool(two_leg["within_tolerance"]),
        {
            "candidate_minus_control": two_leg["candidate_minus_control"],
            "tolerance": MAX_BRIER_DEGRADATION,
            "brier_control": two_leg["brier_control"],
            "brier_candidate": two_leg["brier_candidate"],
        },
    )
    multi = {
        legs: paired["by_legs"][legs]
        for legs in ("3", "4")
        if legs in paired["by_legs"]
    }
    record(
        10,
        "3-leg and 4-leg joint calibration does not worsen beyond tolerance (paired)",
        all(bool(entry["within_tolerance"]) for entry in multi.values()),
        {
            "tolerance": MAX_BRIER_DEGRADATION,
            "by_legs": {
                legs: {
                    "candidate_minus_control": entry["candidate_minus_control"],
                    "brier_control": entry["brier_control"],
                    "brier_candidate": entry["brier_candidate"],
                    "candidate_minus_independence": entry[
                        "candidate_minus_independence"
                    ],
                }
                for legs, entry in multi.items()
            },
        },
    )

    # GATE 11: the parameter classes the brief forbids are empty. Read off the
    # frozen spec's own accounting and cross-checked against the loadings that
    # were actually written.
    counts = frozen["parameter_counts"]
    payload = spec["loadings"]
    structural_absent = {
        "role_deviation_in_payload": payload.get("role_deviation") is None,
        "symmetric_in_payload": payload.get("symmetric") is None,
    }
    record(
        11,
        "no pairwise, player-indexed, role-deviation or symmetric parameter",
        counts["pairwise"] == 0
        and counts["player_indexed"] == 0
        and counts["active_role_deviation_parameters"] == 0
        and counts["active_symmetric_subspace_parameters"] == 0
        and all(structural_absent.values()),
        {"parameter_counts": counts, "loading_payload": structural_absent},
    )

    # GATE 12: the role layer is identified. A share-weighted mean of exactly
    # one is what makes the role scales a redistribution rather than a second
    # chance to raise the pooled level.
    role_fit = diagnostics.get("role_scale", {})
    neutrality = abs(float(role_fit.get("weighted_mean_scale_minus_one", 1.0)))
    scales = role_fit.get("scales", {})
    shares = role_fit.get("player_shares", {})
    reconstructed = (
        abs(sum(shares[role] * scales[role] for role in scales) - 1.0)
        if scales and shares
        else float("inf")
    )
    record(
        12,
        "the multiplicative role layer is pooled-neutral and identified",
        neutrality <= MAX_ROLE_NEUTRALITY_DEVIATION
        and reconstructed <= MAX_ROLE_NEUTRALITY_DEVIATION,
        {
            "weighted_mean_scale_minus_one": role_fit.get(
                "weighted_mean_scale_minus_one"
            ),
            "recomputed_from_published_shares_and_scales": reconstructed,
            "tolerance": MAX_ROLE_NEUTRALITY_DEVIATION,
            "scales": scales,
            "player_shares": shares,
            "tau_log": role_fit.get("tau_log"),
        },
    )

    # GATE 13: nothing was selected on the holdout, and the rejected global
    # inflation factor was not revived.
    carried = changed_paths(args.production_ref)
    offenders = inflation_factor_present(carried)
    selection_clean = (
        inner["holdout_used_for_selection"] is False
        and temperature["holdout_used_for_selection"] is False
        and frozen["holdout_used_for_selection"] is False
        and sorted(int(s) for s in frozen["holdout_seasons"]) == [2024, 2025]
        and sorted(int(s) for s in spec["training_seasons"]) == [2020, 2021, 2022, 2023]
    )
    record(
        13,
        "every selection used pre-2024 folds only and no global inflation "
        "factor was revived",
        selection_clean and not offenders,
        {
            "inner_selection_holdout_used": inner["holdout_used_for_selection"],
            "temperature_holdout_used": temperature["holdout_used_for_selection"],
            "frozen_holdout_used": frozen["holdout_used_for_selection"],
            "training_seasons": spec["training_seasons"],
            "holdout_seasons": frozen["holdout_seasons"],
            "forbidden_inflation_factor": FORBIDDEN_INFLATION_FACTOR,
            "files_mentioning_it": offenders,
        },
    )

    # GATE 14: the production surface is untouched, checked twice: by the
    # validation driver's own scan and by the diff against production.
    driver_scan = report["production_surface"]["modified_paths"]
    outside = [
        path
        for path in carried
        if not path.startswith(ALLOWED_RESEARCH_PREFIXES)
    ]
    record(
        14,
        "the production surface is untouched",
        not driver_scan and not outside,
        {
            "validation_driver_scan": driver_scan,
            "paths_changed_against_production": carried,
            "paths_outside_the_research_surface": outside,
            "allowed_prefixes": list(ALLOWED_RESEARCH_PREFIXES),
        },
    )

    passed = all(entry["passed"] for entry in gates)
    out = {
        "title": "Final upstream remediation: gate report",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "control": "accepted bucket repair, re-run paired through the same driver",
        "control_report": str(args.control_report),
        "candidate_factor_spec_hash": spec["spec_hash"],
        "control_factor_spec_hash": frozen["control"]["factor_spec_hash"],
        "upstream_choices": frozen["upstream_choices"],
        "gates": gates,
        "gates_passed": sum(1 for entry in gates if entry["passed"]),
        "gates_total": len(gates),
        "all_gates_passed": passed,
        "verdict": (
            "ALL FOURTEEN GATES PASS"
            if passed
            else "AT LEAST ONE GATE FAILS: "
            + ", ".join(
                str(entry["gate"]) for entry in gates if not entry["passed"]
            )
        ),
        "code_sha": git_sha(PROJECT_ROOT),
    }
    path = artifact_dir / "gate_report.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    for entry in gates:
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"GATE {entry['gate']:>2} ({entry['name']}): {status}")
    print()
    print(out["verdict"])
    print(f"wrote {path}")

    # Numbers the report quotes directly, so they are not retyped by hand.
    print()
    print("latent bucket z, control vs candidate:")
    for bucket in sorted(observed):
        print(
            f"  {bucket:<26s} observed {observed[bucket]:+.6f}  "
            f"control z {control_z[bucket]:+7.3f}  candidate z {candidate_z[bucket]:+7.3f}"
        )
    print(
        f"\nglobal latent RMSE: control {control_latent['by_model']['candidate']['rmse']:.8f}"
        f"  candidate {latent['by_model']['candidate']['rmse']:.8f}"
    )
    print(
        f"rms z: control {control_latent['by_model']['candidate']['rms_z_error']:.4f}"
        f"  candidate {latent['by_model']['candidate']['rms_z_error']:.4f}"
    )
    print(
        "independence baseline RMSE: "
        f"{latent['by_model']['baseline_independence']['rmse']:.8f}"
    )
    print(
        "\nrole scales: "
        + ", ".join(f"{role}={value:.6f}" for role, value in sorted(scales.items()))
    )
    if np.isfinite(reconstructed):
        print(f"share-weighted mean minus one: {reconstructed:.3e}")


if __name__ == "__main__":
    main()
