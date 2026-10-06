#!/usr/bin/env python
"""Render the final upstream remediation report from the committed artifacts.

SHADOW / RESEARCH ONLY. Reads JSON and prints text. Every number in the report
comes from a file some other driver wrote, so none of them is retyped by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "research" / "final_upstream_remediation"
CONTROL_DIR = PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"

ITEM_TITLES = {
    "1": "temporal AST-AST season random effects",
    "2": "multiplicative role_scale",
    "3": "cross-team shrinkage prior",
    "4": "latent-to-count transmission bridge",
    "5": "cross-player dependence temperature",
    "6": "predictive uncertainty calibration",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_DIR)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rule(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root)

    inner = load(root / "inner_selection.json")
    temperature = load(root / "dependence_temperature.json")
    frozen = load(root / "frozen_spec.json")
    spec = load(root / "factor_spec.json")
    diagnostics = load(root / "covariance_diagnostics.json")
    report = load(root / "validation_report.json")
    control = load(root / "control_validation_report.json")
    paired = load(root / "paired_joint_calibration.json")
    gates = load(root / "gate_report.json")
    published = load(CONTROL_DIR / "validation_report.json")

    print("FINAL UPSTREAM REMEDIATION REPORT")
    print(f"branch              : {frozen['branch']}")
    print(f"code sha            : {frozen['code_sha']}")
    print(f"control             : {frozen['control']['name']}")
    print(f"control spec hash   : {frozen['control']['factor_spec_hash']}")
    print(f"candidate spec hash : {spec['spec_hash']}")
    print(f"training seasons    : {spec['training_seasons']}")
    print(f"holdout seasons     : {spec['validation_seasons']} (never fitted)")

    rule("THE SIX DECISIONS")
    for entry in frozen["items"]:
        print(
            f"  item {entry['item']}  {ITEM_TITLES[entry['item']]:<42s} "
            f"{entry['selected']}"
        )

    rule("ITEM 1: TEMPORAL TREATMENT (pre-2024 forward predictive log score)")
    item1 = inner["item_1_temporal"]
    for label in item1["order"]:
        margin = item1["paired_difference_against_control"][label]
        print(
            f"  {label:<28s} score {item1['pooled_scores'][label]:+.6f}  "
            f"vs A0 {margin['mean']:+.6f} +/- {margin['standard_error']:.6f}"
        )
    print(f"  selected: {item1['selected']}")
    print(f"  reason  : {item1['reason']}")

    rule("ITEM 2: ROLE SCALE (pre-2024 supported-cell RMSE)")
    item2 = inner["item_2_role_scale"]
    for label, value in sorted(item2["pooled_supported_cell_rmse"].items()):
        print(f"  {label:<28s} {value:.6f}")
    difference = item2["paired_difference_log_shrunk_minus_ratio"]
    print(
        f"  paired difference (log_shrunk - ratio): {difference['mean']:+.3e} "
        f"+/- {difference['standard_error']:.3e}"
    )
    print(f"  selected: {item2['selected']}")
    role = diagnostics["role_scale"]
    print(
        "  frozen scales: "
        + ", ".join(f"{k}={v:.6f}" for k, v in sorted(role["scales"].items()))
    )
    print(
        "  raw scales   : "
        + ", ".join(f"{k}={v:.6f}" for k, v in sorted(role["raw_scales"].items()))
    )
    print(
        "  log standard errors: "
        + ", ".join(
            f"{k}={v:.6f}" for k, v in sorted(role["log_standard_errors"].items())
        )
    )
    print(f"  tau_log = {role['tau_log']:.6f}")
    print(
        f"  share-weighted mean minus one = "
        f"{role['weighted_mean_scale_minus_one']:.3e}"
    )

    rule("ITEM 3: CROSS-TEAM PRIOR (pre-2024 cross-team latent RMSE)")
    item3 = inner["item_3_cross_team"]
    for label, value in sorted(
        item3["pooled_rmse"].items(), key=lambda pair: pair[1]
    ):
        print(f"  {label:<28s} {value:.6f}")
    print(f"  selected: {item3['selected']}")

    rule("ITEM 4: TRANSMISSION BRIDGE (pre-2024 forward folds)")
    item4 = inner["item_4_transmission"]
    print(f"  homogeneity of the two sources: {item4['homogeneity_verdict']}")
    worst = max(item4["homogeneity"], key=lambda e: e["max_disagreement_in_sigma"])
    print(
        f"  worst disagreement: {worst['max_disagreement_in_sigma']:.2f} sigma "
        f"({worst['block']}, {worst['target_season']}), p = {worst['p_value']:.3e}"
    )
    print(
        f"  {'cap':<10s} {'latent x':>9s} {'count x':>9s} {'target gain':>12s} "
        f"{'worst prot dz':>14s}  admissible"
    )
    for label in item4["cap_grid_labels"]:
        entry = item4["admissibility"][label]
        print(
            f"  {label:<10s} {entry['forward_latent_rmse_ratio']:9.4f} "
            f"{entry['forward_count_rmse_ratio']:9.4f} "
            f"{entry['forward_target_bucket_improvement']:+12.6f} "
            f"{entry['worst_protected_z_increase']:+14.4f}  "
            f"{entry['admissible']}"
        )
    print(f"  selected cap: {item4['selected_cap']}")

    rule("ITEM 5: DEPENDENCE TEMPERATURE (exact orthant integral, pre-2024)")
    print(f"  events priced: {temperature['scores']['events']:,}")
    print(f"  folds        : {temperature['fold_target_seasons']}")
    print(
        f"  {'lambda':<8s} {'log loss':>10s} {'Brier':>10s} "
        f"{'vs lambda=1':>14s} {'se':>12s}"
    )
    for value in temperature["temperature_grid"]:
        label = f"{value:.2f}"
        entry = temperature["scores"]["by_temperature"][label]
        margin = temperature["selection"][
            "paired_difference_against_full_model"
        ][label]
        print(
            f"  {label:<8s} {entry['log_loss']:10.6f} {entry['brier']:10.6f} "
            f"{margin['mean']:+14.3e} {margin['standard_error']:12.3e}"
        )
    print(f"  selected: lambda = {temperature['selected_temperature']}")
    print(f"  reason  : {temperature['selection']['reason']}")

    rule("ITEM 6: PREDICTIVE UNCERTAINTY (forward cross-fitted coverage)")
    item6 = inner["item_6_uncertainty"]
    print(
        f"  {'candidate':<28s} {'68%':>7s} {'80%':>7s} {'90%':>7s} {'95%':>7s} "
        f"{'mean z^2':>9s} {'loss':>8s}"
    )
    for label in item6["order"]:
        table = item6["coverage"][label]
        print(
            f"  {label:<28s} {table['coverage_68']:7.3f} "
            f"{table['coverage_80']:7.3f} {table['coverage_90']:7.3f} "
            f"{table['coverage_95']:7.3f} {table['mean_squared_z']:9.4f} "
            f"{item6['losses'][label]:8.4f}"
        )
    print(f"  selected: {item6['selected']}")

    rule("PARAMETER ACCOUNTING")
    for key, value in sorted(frozen["parameter_counts"].items()):
        print(f"  {key:<42s} {value}")
    print(
        f"  {'role_scale values / free parameters':<42s} "
        f"{len(frozen['role_scale'])} / "
        f"{frozen['parameter_counts']['role_scale_free_parameters']}"
    )

    rule("HELD-OUT LATENT DEPENDENCE (all 2024-2025 games)")
    latent = report["latent_dependence"]
    control_latent = control["latent_dependence"]
    observed = latent["observed_buckets"]
    se = latent["observed_bucket_se"]
    candidate_implied = latent["by_model"]["candidate"]["implied_buckets"]
    control_implied = control_latent["by_model"]["candidate"]["implied_buckets"]
    print(f"  games {latent['games']:,}, cross-team pairs {latent['cross_team_pairs']:,.0f}")
    print(
        f"  {'bucket':<26s} {'observed':>10s} {'control':>10s} "
        f"{'candidate':>10s} {'ctrl z':>8s} {'cand z':>8s} {'d|z|':>7s}"
    )
    for bucket in sorted(observed):
        control_z = (control_implied[bucket] - observed[bucket]) / se[bucket]
        candidate_z = (candidate_implied[bucket] - observed[bucket]) / se[bucket]
        print(
            f"  {bucket:<26s} {observed[bucket]:+10.6f} "
            f"{control_implied[bucket]:+10.6f} {candidate_implied[bucket]:+10.6f} "
            f"{control_z:+8.3f} {candidate_z:+8.3f} "
            f"{abs(candidate_z) - abs(control_z):+7.3f}"
        )
    print(
        f"  global RMSE : control {control_latent['by_model']['candidate']['rmse']:.8f}"
        f"  candidate {latent['by_model']['candidate']['rmse']:.8f}"
        f"  independence {latent['by_model']['baseline_independence']['rmse']:.8f}"
    )
    print(
        f"  global rms z: control "
        f"{control_latent['by_model']['candidate']['rms_z_error']:.4f}"
        f"  candidate {latent['by_model']['candidate']['rms_z_error']:.4f}"
        f"  independence "
        f"{latent['by_model']['baseline_independence']['rms_z_error']:.4f}"
    )
    print(
        "  published control report agrees on the observed buckets: "
        f"{published['latent_dependence']['observed_buckets'] == observed}"
    )

    rule("PAIRED JOINT CALIBRATION (same games, same seed, same draws)")
    print(f"  events {paired['events']:,} over {paired['games']:,} games")
    print(
        f"  {'legs':>4s} {'brier ctrl':>12s} {'brier cand':>12s} "
        f"{'cand-ctrl':>13s} {'ci95 low':>13s} {'ci95 high':>13s}  ok"
    )
    for legs in sorted(paired["by_legs"]):
        entry = paired["by_legs"][legs]
        delta = entry["candidate_minus_control"]
        print(
            f"  {legs:>4s} {entry['brier_control']:12.8f} "
            f"{entry['brier_candidate']:12.8f} {delta['delta']:+13.3e} "
            f"{delta['ci95'][0]:+13.3e} {delta['ci95'][1]:+13.3e}  "
            f"{entry['within_tolerance']}"
        )

    rule("MARGINAL, SAME-PLAYER AND PSD")
    marginal = report["marginal_preservation"]["candidate"]
    stability = report["stability"]
    print(
        f"  marginal dimensions probed        {marginal['dimensions_probed']:,}"
    )
    print(
        f"  max |over-probability error|      "
        f"{marginal['max_abs_over_probability_error']:.6f}"
    )
    print(f"  max |mean z|                      {marginal['max_abs_mean_z']:.4f}")
    print(
        f"  same-player block deviation       "
        f"{report['same_player_contract']['max_block_deviation']:.3e} over "
        f"{report['same_player_contract']['games_checked']} games"
    )
    print(
        f"  min covariance eigenvalue         "
        f"{stability['min_covariance_eigenvalue']:.6f}"
    )
    print(
        f"  numerical failures                {stability['numerical_failures']}"
    )
    print(
        f"  count-space cross-player RMSE     candidate "
        f"{report['residual_dependence']['count_space_cross_player_rmse']['candidate']:.6f}"
        f"  control "
        f"{control['residual_dependence']['count_space_cross_player_rmse']['candidate']:.6f}"
        f"  production "
        f"{report['residual_dependence']['count_space_cross_player_rmse']['baseline_production']:.6f}"
    )

    rule("GATES")
    for entry in gates["gates"]:
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"  GATE {entry['gate']:>2} {status}  {entry['name']}")
    print(
        f"  {gates['gates_passed']}/{gates['gates_total']} passed -- "
        f"{gates['verdict']}"
    )

    rule("LINEAGE GATES ON THE CANDIDATE'S OWN RUN")
    for entry in report["acceptance_gates"]:
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"  GATE {entry['gate']} {status}  {entry['name']}")
    print(f"  {report['verdict']}")

    rule("HOLDOUT DISCIPLINE")
    print(f"  inner selection used holdout      {inner['holdout_used_for_selection']}")
    print(
        f"  temperature used holdout          "
        f"{temperature['holdout_used_for_selection']}"
    )
    print(f"  freeze certifies holdout unused   {not frozen['holdout_used_for_selection']}")
    print(f"  seasons used for selection        {inner['seasons_used']}")
    print(f"  promotion eligibility             {frozen['promotion_eligibility']}")


if __name__ == "__main__":
    main()
