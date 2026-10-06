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

#: The buckets the brief names as protected. Reported separately from the rest
#: because a gate reads them.
PROTECTED_BUCKETS = (
    "opponent_ast_ast",
    "opponent_pts_reb",
    "opponent_fg3m_reb",
    "passer_ast_teammate_pts",
    "teammate_pts_reb",
)


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
    print(f"  folds scored: {item1['forward_fold_target_seasons']}")
    print(
        f"  {'candidate':<30s} {'log score':>11s} {'se':>9s} "
        f"{'vs A0':>12s} {'se':>10s}"
    )
    for label in item1["candidates_scored"]:
        margin = item1["paired_difference_against_control"][label]
        print(
            f"  {label:<30s} {item1['pooled_forward_log_score'][label]:+11.6f} "
            f"{item1['pooled_forward_log_score_se'][label]:9.6f} "
            f"{margin['mean']:+12.6f} {margin['standard_error']:10.6f}"
        )
    print(f"  selected: {item1['selected']}")
    print(f"  reason  : {item1['selection']['reason']}")
    print(
        f"  robustness grid: nu = {item1['nu_selected']} from {item1['nu_grid']}, "
        f"half life = {item1['half_life_selected']} from {item1['half_life_grid']}"
    )
    print(
        "  nu forward log scores: "
        + ", ".join(
            f"{key}={value:+.6f}"
            for key, value in sorted(
                item1["nu_forward_log_scores"].items(), key=lambda p: float(p[0])
            )
        )
    )

    rule("ITEM 1 EFFECT: WHAT THE TEMPORAL LAYER ACTUALLY MOVED")
    temporal = diagnostics["temporal"]
    print(f"  treatment applied in the frozen fit: {temporal['treatment']}")
    print(f"  max |shift| across buckets         : {temporal['max_abs_shift']:.3e}")
    primary = item1["primary_bucket"]
    series = item1["primary_bucket_series"]
    print(f"  primary bucket: {primary}")
    print(
        "    per-season: "
        + ", ".join(
            f"{season}={value:+.6f}+/-{error:.6f}"
            for season, value, error in zip(
                series["seasons"],
                series["estimates"],
                series["standard_errors"],
                strict=True,
            )
        )
    )
    by_bucket = temporal["by_bucket"]
    print(
        f"  {'bucket':<26s} {'fixed':>11s} {'robust':>11s} {'shift':>11s} "
        f"{'I^2':>7s} {'Q p':>9s}"
    )
    for bucket in sorted(by_bucket, key=lambda name: -abs(by_bucket[name]["shift"])):
        entry = by_bucket[bucket]
        print(
            f"  {bucket:<26s} {entry['fixed_effect_posterior_mean']:+11.6f} "
            f"{entry['selected_posterior_mean']:+11.6f} {entry['shift']:+11.3e} "
            f"{entry['i_squared']:7.3f} {entry['q_p_value']:9.4f}"
        )

    rule("ITEM 2: ROLE SCALE (pre-2024 supported-cell RMSE)")
    item2 = inner["item_2_role_scale"]
    for label, value in sorted(item2["pooled_supported_cell_rmse"].items()):
        print(f"  {label:<28s} {value:.6f}")
    difference = item2["paired_difference_log_shrunk_minus_ratio"]
    print(
        f"  paired difference (log_shrunk - ratio): {difference['mean']:+.3e} "
        f"+/- {difference['standard_error']:.3e} "
        f"(tie band {item2['parsimony_tie_band_se']:.1f} se)"
    )
    print(f"  selected: {item2['selected']}")
    print(f"  reason  : {item2['reason']}")
    role = diagnostics["role_scale"]
    for name in sorted(role["scales"]):
        print(
            f"  {name:<12s} raw {role['raw_scales'][name]:.6f}  "
            f"shrunk {role['scales'][name]:.6f}  "
            f"log se {role['log_standard_errors'][name]:.6f}  "
            f"share {role['player_shares'][name]:.6f}"
        )
    print(f"  tau_log = {role['tau_log']:.6f}")
    print(
        "  share-weighted mean minus one = "
        f"{role['weighted_mean_scale_minus_one']:.3e}"
    )

    rule("ITEM 3: CROSS-TEAM PRIOR (pre-2024 cross-team latent RMSE)")
    item3 = inner["item_3_cross_team"]
    for label in item3["simplicity_order"]:
        print(
            f"  {label:<28s} cross-team {item3['pooled_cross_team_latent_rmse'][label]:.8f}"
        )
    print(f"  selected: {item3['selected']}")
    print(f"  reason  : {item3['selection']['reason']}")
    print(
        "  same-team RMSE is identical across the candidates by construction: "
        "the prior only touches the cross-team block"
    )

    rule("ITEM 4: TRANSMISSION BRIDGE (pre-2024 forward folds)")
    item4 = inner["item_4_transmission"]
    print(f"  homogeneity of the two sources: {item4['homogeneity_verdict']}")
    worst = max(item4["homogeneity"], key=lambda e: e["max_disagreement_in_sigma"])
    print(
        f"  worst disagreement: {worst['max_disagreement_in_sigma']:.2f} sigma "
        f"({worst['block']}, {worst['target_season']}), p = {worst['p_value']:.3e}"
    )
    print(
        f"  rejected in {sum(1 for e in item4['homogeneity'] if not e['homogeneous'])}"
        f" of {len(item4['homogeneity'])} fold-block tests"
    )
    print(
        f"  {'cap':<10s} {'latent x':>9s} {'count x':>9s} {'target gain':>12s} "
        f"{'worst prot dz':>14s}  admissible"
    )
    for value in item4["cap_grid"]:
        label = f"cap_{value:.2f}"
        entry = item4["admissibility"][label]
        print(
            f"  {label:<10s} {entry['forward_latent_rmse_ratio']:9.4f} "
            f"{entry['forward_count_rmse_ratio']:9.4f} "
            f"{entry['forward_target_bucket_improvement']:+12.6f} "
            f"{entry['worst_protected_z_increase']:+14.4f}  "
            f"{entry['admissible']}"
        )
    print(f"  selected cap: {item4['selected_cap']}")
    print(f"  rule        : {item4['selection_rule']}")

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
        margin = temperature["selection"]["paired_difference_against_full_model"][label]
        print(
            f"  {label:<8s} {entry['log_loss']:10.6f} {entry['brier']:10.6f} "
            f"{margin['mean']:+14.3e} {margin['standard_error']:12.3e}"
        )
    print(f"  selected: lambda = {temperature['selected_temperature']}")
    print(f"  reason  : {temperature['selection']['reason']}")
    print(f"  pricing : {temperature['pricing']}")

    rule("ITEM 6: PREDICTIVE UNCERTAINTY (forward cross-fitted coverage)")
    item6 = inner["item_6_uncertainty"]
    order = sorted(item6["coverage_loss"], key=lambda key: item6["coverage_loss"][key])
    print(
        f"  {'candidate':<34s} {'68%':>7s} {'80%':>7s} {'90%':>7s} {'95%':>7s} "
        f"{'mean z^2':>9s} {'loss':>8s}"
    )
    for label in order:
        table = item6["candidates"][label]["coverage"]
        print(
            f"  {label:<34s} {table['coverage_68']:7.3f} "
            f"{table['coverage_80']:7.3f} {table['coverage_90']:7.3f} "
            f"{table['coverage_95']:7.3f} {table['mean_squared_z']:9.4f} "
            f"{item6['coverage_loss'][label]:8.4f}"
        )
    print(f"  selected: {item6['selected']}")
    print(f"  reason  : {item6['selection']['reason']}")
    print(f"  note    : {item6['note']}")

    rule("PARAMETER ACCOUNTING")
    for key, value in sorted(frozen["parameter_counts"].items()):
        print(f"  {key:<42s} {value}")
    print(f"  {'role_scale values':<42s} {len(frozen['role_scale'])}")
    print(
        f"  {'role_scale weighted mean minus one':<42s} "
        f"{frozen['role_scale_weighted_mean_minus_one']:.3e}"
    )

    rule("HELD-OUT LATENT DEPENDENCE (all 2024-2025 games)")
    latent = report["latent_dependence"]
    control_latent = control["latent_dependence"]
    observed = latent["observed_buckets"]
    se = latent["observed_bucket_se"]
    candidate_implied = latent["by_model"]["candidate"]["implied_buckets"]
    control_implied = control_latent["by_model"]["candidate"]["implied_buckets"]
    print(
        f"  games {latent['games']:,}, same-team pairs "
        f"{latent['same_team_pairs']:,.0f}, cross-team pairs "
        f"{latent['cross_team_pairs']:,.0f}"
    )
    print(
        f"  {'bucket':<26s} {'observed':>10s} {'control':>10s} "
        f"{'candidate':>10s} {'ctrl z':>8s} {'cand z':>8s} {'d|z|':>7s} prot"
    )
    for bucket in sorted(observed):
        control_z = (control_implied[bucket] - observed[bucket]) / se[bucket]
        candidate_z = (candidate_implied[bucket] - observed[bucket]) / se[bucket]
        print(
            f"  {bucket:<26s} {observed[bucket]:+10.6f} "
            f"{control_implied[bucket]:+10.6f} {candidate_implied[bucket]:+10.6f} "
            f"{control_z:+8.3f} {candidate_z:+8.3f} "
            f"{abs(candidate_z) - abs(control_z):+7.3f} "
            f"{'yes' if bucket in PROTECTED_BUCKETS else '-'}"
        )
    print(
        f"  global RMSE : control {control_latent['by_model']['candidate']['rmse']:.8f}"
        f"  candidate {latent['by_model']['candidate']['rmse']:.8f}"
        f"  independence {latent['by_model']['baseline_independence']['rmse']:.8f}"
    )
    print(
        "  global rms z: control "
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
    print(f"  tolerance: {paired['max_brier_degradation']:.4f} Brier")
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
    for legs in sorted(paired["by_legs"]):
        entry = paired["by_legs"][legs]
        delta = entry["candidate_minus_independence"]
        print(
            f"  {legs}-leg candidate minus independence: {delta['delta']:+.3e} "
            f"[{delta['ci95'][0]:+.3e}, {delta['ci95'][1]:+.3e}]"
        )

    rule("MARGINAL, SAME-PLAYER AND PSD")
    marginal = report["marginal_preservation"]["candidate"]
    stability = report["stability"]
    print(f"  marginal dimensions probed        {marginal['dimensions_probed']:,}")
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
    print(f"  numerical failures                {stability['numerical_failures']}")
    print(f"  pairwise parameters at simulation {stability['pairwise_parameter_count']}")
    residual = report["residual_dependence"]
    control_residual = control["residual_dependence"]
    print(
        f"  count-space cross-player RMSE     candidate "
        f"{residual['count_space_cross_player_rmse']['candidate']:.6f}"
        f"  control "
        f"{control_residual['count_space_cross_player_rmse']['candidate']:.6f}"
        f"  production "
        f"{residual['count_space_cross_player_rmse']['baseline_production']:.6f}"
    )

    rule("GATES")
    for entry in gates["gates"]:
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"  GATE {entry['gate']:>2} {status}  {entry['name']}")
    print(f"  {gates['gates_passed']}/{gates['gates_total']} passed -- {gates['verdict']}")

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
    print(
        f"  freeze certifies holdout unused   "
        f"{not frozen['holdout_used_for_selection']}"
    )
    print(f"  seasons used for selection        {inner['seasons_used']}")
    print(f"  promotion eligibility             {frozen['promotion_eligibility']}")


if __name__ == "__main__":
    main()
