#!/usr/bin/env python
"""Audit-only reconciliation of the committed remediation artifacts.

SHADOW / RESEARCH ONLY. This driver fits nothing, selects nothing and
simulates nothing. It reads the already-written factor specs, validation
reports and joint-event grades, recomputes only the quantities the audit asks
for that no committed artifact happens to carry, and writes one report.

The three recomputed quantities are:

* log loss on the existing grades, which the reports summarise as Brier only;
* held-out role-pair cell errors, which only exist for the pre-2024 selection
  folds, computed from the two frozen loading sets and the held-out residuals;
* the held-out game subsets the two simulation universes actually used, from
  the validator's own deterministic selection rule.

None of these touches a model parameter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nba_prop_quant.research.game_latent_state.artifacts import git_sha
from nba_prop_quant.research.game_latent_state.covariance import SharedFactorLoadings
from nba_prop_quant.research.game_latent_state.factors import standardize_residuals
from nba_prop_quant.research.game_latent_state.remediation import (
    MIN_CELL_GAMES,
    MIN_CELL_PAIRS,
    ROLE_PAIR_CELLS,
    SeasonSeries,
    fit_student_t_random_effects,
)

ARTIFACT_DIR = PROJECT_ROOT / "research" / "final_upstream_remediation"
REPAIR_DIR = PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"
SHADOW_DIR = PROJECT_ROOT / "research" / "game_latent_state"

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS = (2024, 2025)
ROLE_COLUMN = "role_bucket"

#: The count-space bucket the brief's original item-4 gate names.
COUNT_TARGET_BUCKET = "passer_ast_teammate_pts"

#: The brief's original requirement on that bucket: the absolute count-space
#: error has to fall by at least this fraction. Recorded here as the original
#: number, not as whatever the gate script ended up measuring.
ORIGINAL_COUNT_ERROR_REDUCTION = 0.20

#: Probabilities are clipped before taking a logarithm. A graded event that
#: came back at exactly 0 or 1 would otherwise make the log loss infinite and
#: destroy the comparison for every other event in the fold.
LOG_LOSS_CLIP = 1e-6

BOOTSTRAP_DRAWS = 4000
CELL_BOOTSTRAP_DRAWS = 400
SEED = 73


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--control-grades", type=Path, default=Path("/tmp/control_validation")
        / "joint_event_grades.parquet"
    )
    parser.add_argument(
        "--residuals",
        type=Path,
        default=SHADOW_DIR / "oof_gaussian_residuals.parquet",
    )
    parser.add_argument("--games-per-season-small", type=int, default=200)
    parser.add_argument("--games-per-season-large", type=int, default=300)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rule(title: str) -> None:
    print()
    print(f"=== {title} ===")


# ----------------------------------------------------------------------
# 1. the original count-space target gate
# ----------------------------------------------------------------------


def count_target(candidate: dict, control: dict) -> dict[str, object]:
    cand = candidate["residual_dependence"]
    ctrl = control["residual_dependence"]
    observed = float(cand["observed_buckets"][COUNT_TARGET_BUCKET])
    if ctrl["observed_buckets"][COUNT_TARGET_BUCKET] != observed:
        raise SystemExit("the two runs disagree on the observed count-space target")

    control_predicted = float(ctrl["by_model"]["candidate"]["buckets"][COUNT_TARGET_BUCKET])
    candidate_predicted = float(cand["by_model"]["candidate"]["buckets"][COUNT_TARGET_BUCKET])
    control_error = float(
        ctrl["by_model"]["candidate"]["bucket_errors"][COUNT_TARGET_BUCKET]
    )
    candidate_error = float(
        cand["by_model"]["candidate"]["bucket_errors"][COUNT_TARGET_BUCKET]
    )
    reduction = (abs(control_error) - abs(candidate_error)) / abs(control_error)

    # The same reduction measured against the two baselines, so the reading
    # of the gate cannot be confused with a different denominator. The gate
    # is against the accepted repair control; these are reported only so the
    # difference between the three references is visible.
    alternatives: dict[str, dict[str, float]] = {}
    for name in ("baseline_production", "baseline_independence"):
        reference = float(cand["by_model"][name]["bucket_errors"][COUNT_TARGET_BUCKET])
        alternatives[name] = {
            "reference_error": reference,
            "reference_abs_error": abs(reference),
            "fractional_error_reduction": (abs(reference) - abs(candidate_error))
            / abs(reference),
            "percentage_error_reduction": 100.0
            * (abs(reference) - abs(candidate_error))
            / abs(reference),
        }

    return {
        "gate_reference": "accepted bucket repair control",
        "reduction_against_other_references": alternatives,
        "bucket": COUNT_TARGET_BUCKET,
        "space": "count",
        "observed": observed,
        "control_predicted": control_predicted,
        "candidate_predicted": candidate_predicted,
        "control_error": control_error,
        "candidate_error": candidate_error,
        "control_abs_error": abs(control_error),
        "candidate_abs_error": abs(candidate_error),
        "absolute_error_reduction": abs(control_error) - abs(candidate_error),
        "fractional_error_reduction": reduction,
        "percentage_error_reduction": 100.0 * reduction,
        "original_requirement": ORIGINAL_COUNT_ERROR_REDUCTION,
        "original_requirement_met": bool(reduction >= ORIGINAL_COUNT_ERROR_REDUCTION),
    }


# ----------------------------------------------------------------------
# 3. log loss on the existing grades
# ----------------------------------------------------------------------


def log_loss(probability: np.ndarray, realized: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    return -(realized * np.log(clipped) + (1.0 - realized) * np.log(1.0 - clipped))


def clustered_bootstrap_ci(
    per_event: np.ndarray,
    games: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    """Game-clustered bootstrap of the mean of a per-event quantity."""
    unique = np.unique(games)
    index = {game: position for position, game in enumerate(unique)}
    buckets: list[list[float]] = [[] for _ in unique]
    for value, game in zip(per_event, games, strict=True):
        buckets[index[game]].append(float(value))
    sums = np.array([sum(values) for values in buckets])
    counts = np.array([len(values) for values in buckets], dtype=float)

    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        picks = rng.integers(0, len(unique), size=len(unique))
        means[draw] = sums[picks].sum() / counts[picks].sum()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(np.std(means, ddof=1)), float(low), float(high)


def paired_log_loss(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
) -> dict[str, object]:
    for column in ("game_id", "family", "n_legs", "realized"):
        if not candidate[column].equals(control[column]):
            raise SystemExit(f"the two grade sets disagree on {column}")
    for column in ("p_baseline_independence", "p_baseline_production"):
        if not np.allclose(candidate[column], control[column]):
            raise SystemExit(f"the two grade sets disagree on {column}")

    realized = candidate["realized"].to_numpy(dtype=float)
    games = candidate["game_id"].to_numpy()
    losses = {
        "candidate": log_loss(candidate["p_candidate"].to_numpy(dtype=float), realized),
        "control": log_loss(control["p_candidate"].to_numpy(dtype=float), realized),
        "production": log_loss(
            candidate["p_baseline_production"].to_numpy(dtype=float), realized
        ),
        "independence": log_loss(
            candidate["p_baseline_independence"].to_numpy(dtype=float), realized
        ),
    }

    out: dict[str, object] = {
        "events": int(len(candidate)),
        "games": int(len(np.unique(games))),
        "clip": LOG_LOSS_CLIP,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "by_legs": {},
    }
    for legs in sorted(candidate["n_legs"].unique()):
        mask = (candidate["n_legs"] == legs).to_numpy()
        entry: dict[str, object] = {
            "events": int(mask.sum()),
            "games": int(len(np.unique(games[mask]))),
            "base_rate": float(realized[mask].mean()),
        }
        for name, value in losses.items():
            entry[f"log_loss_{name}"] = float(value[mask].mean())
        for label, (left, right) in (
            ("candidate_minus_control", ("candidate", "control")),
            ("candidate_minus_production", ("candidate", "production")),
            ("control_minus_production", ("control", "production")),
        ):
            difference = losses[left][mask] - losses[right][mask]
            se, low, high = clustered_bootstrap_ci(
                difference, games[mask], BOOTSTRAP_DRAWS, SEED
            )
            entry[label] = {
                "delta": float(difference.mean()),
                "standard_error": se,
                "ci95": [low, high],
                "favours_candidate": bool(high < 0.0),
                "significant": bool(high < 0.0 or low > 0.0),
            }
        out["by_legs"][str(int(legs))] = entry  # type: ignore[index]
    return out


# ----------------------------------------------------------------------
# 4. held-out role-pair cells
# ----------------------------------------------------------------------


def per_game_role_cells(
    frame: pd.DataFrame,
    stats: tuple[str, ...],
    role_column: str = ROLE_COLUMN,
    value_prefix: str = "zs_",
) -> dict[tuple[str, str], dict[str, object]]:
    """Per-game cell contributions, so a game-clustered SE is available.

    Identical accumulation to ``role_pair_cell_moments``: the team-sum identity
    applied within each ordered role pair, with the diagonal self-products
    removed. This version keeps the per-game terms instead of only their total,
    because the audit needs the clustered standard error of each cell and the
    library version returns the pooled matrix alone.
    """
    columns = [f"{value_prefix}{stat}" for stat in stats]
    usable = frame.dropna(subset=["game_id", "team_id", role_column, *columns])
    roles = sorted(str(value) for value in usable[role_column].unique())
    n = len(stats)

    blocks: dict[tuple[str, str], list[np.ndarray]] = {
        (a, b): [] for a in roles for b in roles
    }
    counts: dict[tuple[str, str], list[float]] = {key: [] for key in blocks}

    for _, game in usable.groupby("game_id", sort=True):
        game_block: dict[tuple[str, str], np.ndarray] = {}
        game_count: dict[tuple[str, str], float] = {}
        for _, team in game.groupby("team_id", sort=True):
            totals: dict[str, np.ndarray] = {}
            own: dict[str, np.ndarray] = {}
            size: dict[str, int] = {}
            for role, members in team.groupby(role_column, sort=True):
                values = members[columns].to_numpy(dtype=float)
                label = str(role)
                totals[label] = values.sum(axis=0)
                own[label] = values.T @ values
                size[label] = values.shape[0]
            for first in totals:
                for second in totals:
                    block = np.outer(totals[first], totals[second])
                    count = size[first] * size[second]
                    if first == second:
                        block = block - own[first]
                        count -= size[first]
                    if count <= 0:
                        continue
                    key = (first, second)
                    game_block[key] = game_block.get(key, np.zeros((n, n))) + block
                    game_count[key] = game_count.get(key, 0.0) + count
        for key, block in game_block.items():
            blocks[key].append(block)
            counts[key].append(game_count[key])

    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, terms in blocks.items():
        if not terms:
            continue
        stacked = np.stack(terms)
        count = np.asarray(counts[key], dtype=float)
        total = count.sum()
        if total <= 0:
            continue
        matrix = stacked.sum(axis=0) / total
        out[key] = {
            "correlation": 0.5 * (matrix + matrix.T),
            "per_game_blocks": stacked,
            "per_game_counts": count,
            "pairs": float(total),
            "games": int(len(count)),
            "supported": bool(
                len(count) >= MIN_CELL_GAMES and total >= MIN_CELL_PAIRS
            ),
        }
    return out


def cell_standard_errors(cell: dict[str, object], draws: int, seed: int) -> np.ndarray:
    """Game-clustered bootstrap SE of every entry of one cell's matrix."""
    blocks = np.asarray(cell["per_game_blocks"])
    counts = np.asarray(cell["per_game_counts"], dtype=float)
    rng = np.random.default_rng(seed)
    games = len(counts)
    drawn = np.empty((draws, blocks.shape[1], blocks.shape[2]))
    for draw in range(draws):
        picks = rng.integers(0, games, size=games)
        total = counts[picks].sum()
        matrix = blocks[picks].sum(axis=0) / total
        drawn[draw] = 0.5 * (matrix + matrix.T)
    return drawn.std(axis=0, ddof=1)


def role_cells(
    residuals: pd.DataFrame,
    candidate_spec: dict,
    control_spec: dict,
) -> dict[str, object]:
    moments = {
        stat: (float(value["mean"]), float(value["sd"]))
        for stat, value in candidate_spec["standardization_moments"].items()
    }
    if control_spec["standardization_moments"] != candidate_spec[
        "standardization_moments"
    ]:
        raise SystemExit(
            "the two specs standardize differently, so their cells are not "
            "measured on the same scale"
        )

    held_out = residuals.loc[residuals["season"].isin(list(HOLDOUT_SEASONS))]
    standardized, _ = standardize_residuals(held_out, STATS, moments=moments)
    observed = per_game_role_cells(standardized, STATS)

    candidate = SharedFactorLoadings.from_payload(candidate_spec["loadings"])
    control = SharedFactorLoadings.from_payload(control_spec["loadings"])

    by_cell: dict[str, object] = {}
    worst_label = None
    worst_delta = -float("inf")
    for position, (first, second) in enumerate(ROLE_PAIR_CELLS):
        cell = observed.get((first, second))
        label = f"{first}+{second}"
        if cell is None:
            by_cell[label] = {"present": False}
            continue
        truth = np.asarray(cell["correlation"], dtype=float)
        se = cell_standard_errors(cell, CELL_BOOTSTRAP_DRAWS, SEED + position)
        safe = np.where(se > 0.0, se, np.nan)

        entry: dict[str, object] = {
            "present": True,
            "supported": bool(cell["supported"]),
            "games": cell["games"],
            "pairs": cell["pairs"],
        }
        errors: dict[str, float] = {}
        z_errors: dict[str, float] = {}
        for name, loadings in (("control", control), ("candidate", candidate)):
            implied = loadings.same_team_correlation_for_roles(first, second)
            residual = implied - truth
            errors[name] = float(np.sqrt(np.mean(np.square(residual))))
            z_errors[name] = float(
                np.sqrt(np.nanmean(np.square(residual / safe)))
            )
            entry[f"{name}_rmse"] = errors[name]
            entry[f"{name}_rms_z"] = z_errors[name]
            entry[f"{name}_max_abs_error"] = float(np.max(np.abs(residual)))
        entry["delta_rmse"] = errors["candidate"] - errors["control"]
        entry["delta_rms_z"] = z_errors["candidate"] - z_errors["control"]
        by_cell[label] = entry

        if cell["supported"] and entry["delta_rmse"] > worst_delta:
            worst_delta = float(entry["delta_rmse"])
            worst_label = label

    supported = [
        value
        for value in by_cell.values()
        if isinstance(value, dict) and value.get("supported")
    ]
    return {
        "definition": (
            "per cell, the root mean square over the 6x6 stat block of "
            "(model-implied same-team correlation for the ordered role pair "
            "minus the held-out observed one); rms z divides each entry by its "
            "game-clustered bootstrap standard error"
        ),
        "min_cell_games": MIN_CELL_GAMES,
        "min_cell_pairs": MIN_CELL_PAIRS,
        "cell_bootstrap_draws": CELL_BOOTSTRAP_DRAWS,
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "by_cell": by_cell,
        "supported_cells": len(supported),
        "pooled_supported_rmse": {
            "control": float(
                np.sqrt(np.mean([value["control_rmse"] ** 2 for value in supported]))
            ),
            "candidate": float(
                np.sqrt(np.mean([value["candidate_rmse"] ** 2 for value in supported]))
            ),
        },
        "worst_supported_cell_deterioration": {
            "cell": worst_label,
            "delta_rmse": None if worst_label is None else worst_delta,
        },
    }


# ----------------------------------------------------------------------
# 5. the two simulation universes
# ----------------------------------------------------------------------


def evenly_spaced(game_ids: list[int], count: int) -> list[int]:
    """The validator's own subsample rule, reproduced exactly."""
    if count and len(game_ids) > count:
        picks = np.linspace(0, len(game_ids) - 1, count)
        return [game_ids[round(index)] for index in picks]
    return list(game_ids)


def universes(
    residuals: pd.DataFrame,
    small: int,
    large: int,
) -> dict[str, object]:
    per_season: dict[str, object] = {}
    small_all: set[int] = set()
    large_all: set[int] = set()
    for season in HOLDOUT_SEASONS:
        available = sorted(
            int(value)
            for value in residuals.loc[residuals["season"] == season, "game_id"].unique()
        )
        picked_small = set(evenly_spaced(available, small))
        picked_large = set(evenly_spaced(available, large))
        per_season[str(season)] = {
            "games_available": len(available),
            f"picked_{small}": len(picked_small),
            f"picked_{large}": len(picked_large),
            "overlap": len(picked_small & picked_large),
        }
        small_all |= picked_small
        large_all |= picked_large
    return {
        "rule": (
            "numpy.linspace over the sorted held-out game ids of each season, "
            "rounded to an index: an evenly spaced subsample that spans the "
            "whole calendar"
        ),
        "games_per_season_small": small,
        "games_per_season_large": large,
        "total_small": len(small_all),
        "total_large": len(large_all),
        "overlap": len(small_all & large_all),
        "small_is_subset_of_large": bool(small_all <= large_all),
        "in_small_only": len(small_all - large_all),
        "overlap_fraction_of_small": len(small_all & large_all) / len(small_all),
        "by_season": per_season,
    }


# ----------------------------------------------------------------------
# 6. half life
# ----------------------------------------------------------------------


def half_life_status(inner: dict, diagnostics: dict) -> dict[str, object]:
    """Whether the recorded half life reaches the selected treatment at all."""
    treatment = diagnostics["temporal"]["treatment"]
    carried = diagnostics["temporal"].get("half_life")
    primary = inner["item_1_temporal"]["primary_bucket"]
    # The per-bucket fit records the half life it was actually given. Under A2
    # that is null for every bucket, which is the proof the dial never
    # reached the estimator.
    inside = {
        bucket: entry.get("half_life")
        for bucket, entry in diagnostics["temporal"]["by_bucket"].items()
    }

    recorded_series = inner["item_1_temporal"]["primary_bucket_series"]
    probe = SeasonSeries(
        name=inner["item_1_temporal"]["primary_bucket"],
        seasons=tuple(int(value) for value in recorded_series["seasons"]),
        estimates=np.asarray(recorded_series["estimates"], dtype=float),
        standard_errors=np.asarray(
            recorded_series["standard_errors"], dtype=float
        ),
    )

    nu = float(inner["item_1_temporal"]["nu_selected"])
    recorded_value = float(inner["item_1_temporal"]["half_life_selected"])
    a2 = fit_student_t_random_effects(probe, nu=nu)
    a3 = fit_student_t_random_effects(probe, nu=nu, half_life=recorded_value)

    return {
        "selected_treatment": treatment,
        "recorded_half_life_in_choices": inner["item_1_temporal"][
            "half_life_selected"
        ],
        "half_life_carried_in_the_diagnostics_header": carried,
        "half_life_recorded_inside_each_bucket_fit": inside,
        "half_life_is_null_in_every_bucket_fit": all(
            value is None for value in inside.values()
        ),
        "half_life_reaches_the_selected_treatment": False,
        "probe_bucket": primary,
        "a2_model_label": a2.model,
        "a2_half_life": a2.half_life,
        "a2_posterior_mean": float(a2.posterior_mean),
        "a3_model_label": a3.model,
        "a3_posterior_mean": float(a3.posterior_mean),
        "a2_differs_from_a3": bool(
            not np.isclose(a2.posterior_mean, a3.posterior_mean)
        ),
        "a2_reproduces_the_frozen_posterior_mean": bool(
            np.isclose(
                a2.posterior_mean,
                diagnostics["temporal"]["by_bucket"][primary][
                    "selected_posterior_mean"
                ],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "frozen_posterior_mean": diagnostics["temporal"]["by_bucket"][primary][
            "selected_posterior_mean"
        ],
        "status": (
            "INACTIVE / NOT APPLICABLE"
            if treatment == "A2_robust_student_t"
            else "ACTIVE"
        ),
        "reason": (
            "temporal_fitter passes half_life only on the A3 branch; the A2 "
            "branch calls fit_student_t_random_effects(series, nu=nu), leaving "
            "half_life at its None default, which is the uniform-weight case. "
            "Every fit record in the artifacts carries half_life = null. The "
            "selected value was chosen on the A3 grid and carried into the "
            "frozen choices, where it has no computational effect."
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root)

    frozen = load(root / "frozen_spec.json")
    candidate_spec = load(root / "factor_spec.json")
    control_spec = load(REPAIR_DIR / "factor_spec.json")
    diagnostics = load(root / "covariance_diagnostics.json")
    inner = load(root / "inner_selection.json")
    temperature = load(root / "dependence_temperature.json")
    candidate_report = load(root / "validation_report.json")
    control_report = load(root / "control_validation_report.json")
    published_repair = load(REPAIR_DIR / "validation_report.json")
    gates = load(root / "gate_report.json")

    print("FINAL REMEDIATION AUDIT (reconciliation of committed artifacts)")

    rule("1. ORIGINAL COUNT-SPACE TARGET GATE")
    count = count_target(candidate_report, control_report)
    print(f"  bucket            {count['bucket']} (count space)")
    print(f"  observed          {count['observed']:+.8f}")
    print(f"  control predicted {count['control_predicted']:+.8f}")
    print(f"  control error     {count['control_error']:+.8f}")
    print(f"  cand.  predicted  {count['candidate_predicted']:+.8f}")
    print(f"  cand.  error      {count['candidate_error']:+.8f}")
    print(
        f"  absolute error    {count['control_abs_error']:.8f} -> "
        f"{count['candidate_abs_error']:.8f}"
    )
    print(f"  error reduction   {count['percentage_error_reduction']:.2f}%")
    print(
        f"  original gate     >= {100 * count['original_requirement']:.0f}% "
        f"against the {count['gate_reference']} -> "
        f"{'PASS' if count['original_requirement_met'] else 'FAIL'}"
    )
    for name, entry in count["reduction_against_other_references"].items():  # type: ignore[union-attr]
        print(
            f"  for reference: against {name:<24s} the same candidate error is "
            f"{entry['percentage_error_reduction']:.2f}% lower"
        )

    rule("2. UNCERTAINTY CALIBRATION")
    item6 = inner["item_6_uncertainty"]
    losses = item6["coverage_loss"]
    baseline = losses["U0_raw"]
    better = {
        name: value for name, value in losses.items() if value < baseline
    }
    uncertainty = {
        "selected": item6["selected"],
        "coverage_loss": losses,
        "raw_loss": baseline,
        "candidates_better_than_raw": better,
        "improved_over_raw": bool(better),
        "statement": (
            "NO CALIBRATION IMPROVEMENT - RAW RETAINED"
            if not better
            else "calibration improved over the raw interval"
        ),
    }
    for name in sorted(losses, key=lambda key: losses[key]):
        print(f"  {name:<34s} coverage loss {losses[name]:.6f}")
    print(f"  {uncertainty['statement']}")

    rule("3. PAIRED LOG LOSS ON THE EXISTING GRADES")
    candidate_grades = pd.read_parquet(root / "joint_event_grades.parquet")
    control_grades = pd.read_parquet(args.control_grades)
    logs = paired_log_loss(candidate_grades, control_grades)
    print(f"  events {logs['events']:,} over {logs['games']:,} games")
    print(
        f"  {'legs':>4s} {'control':>10s} {'candidate':>10s} {'production':>11s} "
        f"{'cand-ctrl':>12s} {'ci95':>26s}"
    )
    for legs, entry in logs["by_legs"].items():  # type: ignore[union-attr]
        margin = entry["candidate_minus_control"]
        print(
            f"  {legs:>4s} {entry['log_loss_control']:10.6f} "
            f"{entry['log_loss_candidate']:10.6f} "
            f"{entry['log_loss_production']:11.6f} {margin['delta']:+12.3e} "
            f"[{margin['ci95'][0]:+.3e}, {margin['ci95'][1]:+.3e}]"
        )
    for legs, entry in logs["by_legs"].items():  # type: ignore[union-attr]
        margin = entry["candidate_minus_production"]
        print(
            f"  {legs}-leg candidate minus production: {margin['delta']:+.6f} "
            f"[{margin['ci95'][0]:+.6f}, {margin['ci95'][1]:+.6f}] "
            f"significant={margin['significant']}"
        )

    rule("4. HELD-OUT ROLE-PAIR CELLS")
    residuals = pd.read_parquet(args.residuals)
    cells = role_cells(residuals, candidate_spec, control_spec)
    print(
        f"  {'cell':<22s} {'games':>6s} {'pairs':>10s} sup "
        f"{'ctrl rmse':>10s} {'cand rmse':>10s} {'d rmse':>11s} "
        f"{'ctrl z':>8s} {'cand z':>8s} {'d z':>8s}"
    )
    for label, entry in cells["by_cell"].items():  # type: ignore[union-attr]
        if not entry.get("present"):
            print(f"  {label:<22s} absent from the held-out seasons")
            continue
        print(
            f"  {label:<22s} {entry['games']:6d} {entry['pairs']:10.0f} "
            f"{'y' if entry['supported'] else 'n':>3s} "
            f"{entry['control_rmse']:10.6f} {entry['candidate_rmse']:10.6f} "
            f"{entry['delta_rmse']:+11.3e} {entry['control_rms_z']:8.3f} "
            f"{entry['candidate_rms_z']:8.3f} {entry['delta_rms_z']:+8.3f}"
        )
    worst = cells["worst_supported_cell_deterioration"]
    print(
        f"  pooled supported RMSE: control "
        f"{cells['pooled_supported_rmse']['control']:.6f}  candidate "
        f"{cells['pooled_supported_rmse']['candidate']:.6f}"
    )
    print(
        f"  worst supported-cell change: {worst['cell']} "
        f"{worst['delta_rmse']:+.3e}"
    )

    rule("5. VALIDATION UNIVERSE RECONCILIATION")
    universe = universes(
        residuals, args.games_per_season_small, args.games_per_season_large
    )
    print(f"  selection rule: {universe['rule']}")
    print(
        f"  published repair run : {universe['games_per_season_large']} games/season "
        f"-> {universe['total_large']} games"
    )
    print(
        f"  paired runs          : {universe['games_per_season_small']} games/season "
        f"-> {universe['total_small']} games"
    )
    print(
        f"  overlap {universe['overlap']} games "
        f"({100 * universe['overlap_fraction_of_small']:.1f}% of the smaller set); "
        f"nested = {universe['small_is_subset_of_large']}"
    )
    count_rmse = {
        "published_repair_600": published_repair["residual_dependence"][
            "count_space_cross_player_rmse"
        ],
        "paired_control_400": control_report["residual_dependence"][
            "count_space_cross_player_rmse"
        ],
        "paired_candidate_400": candidate_report["residual_dependence"][
            "count_space_cross_player_rmse"
        ],
    }
    latent_rmse = {
        "published_repair_600": published_repair["residual_dependence"][
            "cross_player_rmse"
        ],
        "paired_control_400": control_report["residual_dependence"][
            "cross_player_rmse"
        ],
        "paired_candidate_400": candidate_report["residual_dependence"][
            "cross_player_rmse"
        ],
    }
    print(f"  {'run':<24s} {'independence':>14s} {'production':>12s} {'model':>12s}")
    for name, table in count_rmse.items():
        print(
            f"  {name:<24s} {table['baseline_independence']:14.8f} "
            f"{table['baseline_production']:12.8f} {table['candidate']:12.8f}"
        )
    print("  latent-space cross-player RMSE, same three runs:")
    for name, table in latent_rmse.items():
        print(f"  {name:<24s} model {table['candidate']:.18f}")
    identical_latent = (
        latent_rmse["published_repair_600"]["candidate"]
        == latent_rmse["paired_control_400"]["candidate"]
    )
    print(
        "  the published 600-game run and the paired 400-game run report the "
        f"same control model latent RMSE to the last digit: {identical_latent}"
    )

    rule("6. PROVENANCE")
    lineage = {
        "clean_control_spec_hash": control_spec["spec_hash"],
        "clean_control_artifact_dir": str(REPAIR_DIR.relative_to(PROJECT_ROOT)),
        "inner_selection_code_sha": inner["code_sha"],
        "dependence_temperature_code_sha": temperature["code_sha"],
        "frozen_spec_code_sha": frozen["code_sha"],
        "frozen_spec_candidate_hash": frozen["factor_spec_hash"],
        "candidate_factor_spec_hash": candidate_spec["spec_hash"],
        "confirmatory_validation_code_sha": load(root / "manifest.validation.json")[
            "code_sha"
        ],
        "gate_report_code_sha": gates["code_sha"],
        "branch_head": git_sha(PROJECT_ROOT),
    }
    for key, value in lineage.items():
        print(f"  {key:<36s} {value}")

    rule("HALF LIFE")
    half = half_life_status(inner, diagnostics)
    print(f"  selected treatment            {half['selected_treatment']}")
    print(f"  half life carried in choices  {half['recorded_half_life_in_choices']}")
    print(
        f"  half life inside every fit    null in all "
        f"{len(half['half_life_recorded_inside_each_bucket_fit'])} bucket fits: "
        f"{half['half_life_is_null_in_every_bucket_fit']}"
    )
    print(f"  probe bucket                  {half['probe_bucket']}")
    print(
        f"  A2 (no half life)             {half['a2_posterior_mean']:+.9f}  "
        f"label {half['a2_model_label']}"
    )
    print(
        f"  A3 (half life carried)        {half['a3_posterior_mean']:+.9f}  "
        f"label {half['a3_model_label']}"
    )
    print(
        f"  A2 reproduces the frozen mean {half['a2_reproduces_the_frozen_posterior_mean']} "
        f"({half['frozen_posterior_mean']:+.12f})"
    )
    print(f"  status                        {half['status']}")

    out = {
        "title": "Final remediation audit: reconciliation of committed artifacts",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "audit_only": True,
        "nothing_was_refitted_or_reselected": True,
        "count_target_gate": count,
        "uncertainty": uncertainty,
        "paired_log_loss": logs,
        "role_cells": {
            key: value for key, value in cells.items() if key != "by_cell"
        }
        | {
            "by_cell": {
                label: {
                    key: value
                    for key, value in entry.items()
                    if key not in {"per_game_blocks", "per_game_counts"}
                }
                for label, entry in cells["by_cell"].items()  # type: ignore[union-attr]
            }
        },
        "validation_universe": universe,
        "count_space_rmse_by_run": count_rmse,
        "latent_space_rmse_by_run": latent_rmse,
        "provenance": lineage,
        "half_life": half,
        "code_sha": git_sha(PROJECT_ROOT),
    }
    path = root / "audit_reconciliation.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
