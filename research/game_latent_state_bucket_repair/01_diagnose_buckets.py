#!/usr/bin/env python
"""Diagnose the two under-captured teammate buckets on PRE-2024 data only.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Quantifies, for ``teammate_reb_reb`` and ``teammate_ast_ast`` and for the five
neighbouring buckets accepted V1 already reproduces well:

* the raw pooled correlation and its game-clustered standard error,
* the signal-to-standard-error ratio,
* the value surviving V1's 1.96-SE soft threshold,
* pair and game counts,
* season-by-season and role-bucket stability, including sign stability,
* bootstrap uncertainty, and
* the decomposition of the total attenuation into a *shrinkage* loss and a
  *rank-truncation* loss.

That last decomposition is the point of this script: it separates hypothesis A
(over-aggressive thresholding) from hypothesis C (the representation cannot
express the stat pair) instead of assuming one of them.

Seasons 2024 and 2025 are filtered out before anything is computed, so nothing
here can inform a choice using the final holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    project_psd_rank,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    DEFAULT_SHRINK_Z,
    dominating_additive_gram,
    pair_moments,
    soft_threshold,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.repair import (  # noqa: E402
    V1_CONTROL_SPEC,
    fit_repaired_factors,
)

console = Console()

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")

#: Final-evaluation seasons. Excluded from every computation in this file.
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

#: The two repair targets, then the five neighbours the brief asks to compare.
TARGET_BUCKETS: dict[str, tuple[str, str, str]] = {
    "teammate_reb_reb": ("same_team", "reb", "reb"),
    "teammate_ast_ast": ("same_team", "ast", "ast"),
}
NEIGHBOUR_BUCKETS: dict[str, tuple[str, str, str]] = {
    "passer_ast_teammate_pts": ("same_team", "ast", "pts"),
    "teammate_pts_reb": ("same_team", "pts", "reb"),
    "teammate_pts_pts": ("same_team", "pts", "pts"),
    "opponent_reb_reb": ("cross_team", "reb", "reb"),
    "opponent_ast_ast": ("cross_team", "ast", "ast"),
}
ALL_BUCKETS = {**TARGET_BUCKETS, **NEIGHBOUR_BUCKETS}


def load_pre_holdout(residual_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(residual_path)
    pre = frame[~frame["season"].isin(HOLDOUT_SEASONS)].copy()
    if pre.empty:
        raise ValueError("no pre-holdout residual rows available")
    leaked = sorted(set(pre["season"].unique()) & set(HOLDOUT_SEASONS))
    if leaked:
        raise AssertionError(f"holdout seasons leaked into diagnostics: {leaked}")
    return pre


def bucket_entry(
    name: str,
    moments,
    matrix_same: np.ndarray,
    matrix_cross: np.ndarray,
) -> float:
    kind, first, second = ALL_BUCKETS[name]
    index = {stat: position for position, stat in enumerate(moments.stats)}
    i, j = index[first], index[second]
    source = matrix_same if kind == "same_team" else matrix_cross
    return float(source[i, j])


def bucket_se(name: str, moments) -> float:
    kind, first, second = ALL_BUCKETS[name]
    index = {stat: position for position, stat in enumerate(moments.stats)}
    i, j = index[first], index[second]
    source = moments.same_team_se if kind == "same_team" else moments.cross_team_se
    return float(source[i, j])


def rank_sweep(
    same: np.ndarray,
    cross: np.ndarray,
    stats: tuple[str, ...],
) -> list[dict[str, object]]:
    """Fitted same-team block over a grid of ``(k_game, r_contrast)``.

    Shows which rank channel each target bucket loses through, and that at full
    rank the construction reproduces the shrunk block exactly.
    """
    index = {stat: position for position, stat in enumerate(stats)}
    additive = dominating_additive_gram(same, cross)
    competition = additive - same
    game_gram = 0.5 * (additive + cross)
    contrast_gram = 0.5 * (additive - cross)
    _, competition_loadings = project_psd_rank(competition, rank=len(stats))

    out: list[dict[str, object]] = []
    for k_game in (2, 3, 4, 6):
        for r_contrast in (1, 2, 3, 6):
            _, game_loadings = project_psd_rank(game_gram, rank=k_game)
            _, contrast_loadings = project_psd_rank(contrast_gram, rank=r_contrast)
            fitted = (
                game_loadings @ game_loadings.T
                + contrast_loadings @ contrast_loadings.T
                - competition_loadings @ competition_loadings.T
            )
            out.append(
                {
                    "k_game": k_game,
                    "r_contrast": r_contrast,
                    "teammate_reb_reb": float(fitted[index["reb"], index["reb"]]),
                    "teammate_ast_ast": float(fitted[index["ast"], index["ast"]]),
                    "max_abs_deviation_from_shrunk": float(
                        np.max(np.abs(fitted - same))
                    ),
                    "rmse_vs_shrunk": float(np.sqrt(np.mean((fitted - same) ** 2))),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        default=str(
            PROJECT_ROOT
            / "research"
            / "game_latent_state"
            / "oof_gaussian_residuals.parquet"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"),
    )
    parser.add_argument("--bootstrap", type=int, default=800)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    console.rule("Bucket-repair diagnostics (pre-2024 only)")
    pre = load_pre_holdout(Path(args.residuals))
    seasons = [int(season) for season in sorted(pre["season"].unique())]
    console.print(f"seasons used: {seasons}  (holdout {list(HOLDOUT_SEASONS)} excluded)")
    console.print(
        f"rows {len(pre):,}   games {pre['game_id'].nunique():,}   "
        f"players {pre['player_id'].nunique():,}"
    )

    standardized, moment_constants = standardize_residuals(pre, STATS)

    console.rule("Pooled estimates and the 1.96-SE soft threshold")
    pooled = pair_moments(
        standardized, STATS, bootstrap=args.bootstrap, seed=args.seed
    )
    same_thresholded = soft_threshold(
        pooled.same_team, pooled.same_team_se, DEFAULT_SHRINK_Z
    )
    cross_thresholded = soft_threshold(
        pooled.cross_team, pooled.cross_team_se, DEFAULT_SHRINK_Z
    )

    control = fit_repaired_factors(
        standardized,
        STATS,
        V1_CONTROL_SPEC,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    fitted_same = control.loadings.same_team_correlation()
    fitted_cross = control.loadings.cross_team_correlation()

    table = Table(title="observed -> thresholded -> V1-fitted")
    for column in (
        "bucket",
        "observed",
        "se",
        "signal/se",
        "after 1.96SE",
        "V1 fitted",
        "thr loss",
        "rank loss",
        "total loss",
    ):
        table.add_column(column, justify="right")

    diagnostics: dict[str, dict[str, object]] = {}
    for name in ALL_BUCKETS:
        observed = bucket_entry(name, pooled, pooled.same_team, pooled.cross_team)
        se = bucket_se(name, pooled)
        after = bucket_entry(name, pooled, same_thresholded, cross_thresholded)
        final = bucket_entry(name, pooled, fitted_same, fitted_cross)
        scale = abs(observed) if abs(observed) > 1e-12 else 1.0
        record = {
            "observed": observed,
            "standard_error": se,
            "signal_to_se": observed / se if se > 0 else float("nan"),
            "after_soft_threshold_1p96": after,
            "v1_fitted": final,
            "shrinkage_loss_fraction": (observed - after) / scale,
            "rank_loss_fraction": (after - final) / scale,
            "total_loss_fraction": (observed - final) / scale,
            "is_repair_target": name in TARGET_BUCKETS,
        }
        diagnostics[name] = record
        table.add_row(
            name,
            f"{observed:+.6f}",
            f"{se:.6f}",
            f"{record['signal_to_se']:+.2f}",
            f"{after:+.6f}",
            f"{final:+.6f}",
            f"{record['shrinkage_loss_fraction']:.1%}",
            f"{record['rank_loss_fraction']:.1%}",
            f"{record['total_loss_fraction']:.1%}",
        )
    console.print(table)
    console.print(
        "The soft threshold removes a fixed 1.96 standard errors, so it costs a "
        "fraction 1.96 / |signal-to-se| of the signal. That is why the two "
        "weakest significant buckets are taxed hardest."
    )

    console.rule("Pair and game counts")
    counts = {
        "games": int(pooled.games),
        "same_team_ordered_pairs": float(pooled.same_team_pairs),
        "cross_team_ordered_pairs": float(pooled.cross_team_pairs),
        "player_games": int(len(standardized)),
    }
    console.print(counts)

    console.rule("Season-by-season stability")
    per_season: dict[str, dict[str, dict[str, float]]] = {}
    season_table = Table(title="per-season pooled estimate (+/- game-clustered SE)")
    season_table.add_column("bucket")
    for season in seasons:
        season_table.add_column(str(season), justify="right")
    season_moments = {}
    for season in seasons:
        subset = standardized[standardized["season"] == season]
        season_moments[season] = pair_moments(
            subset, STATS, bootstrap=args.bootstrap, seed=args.seed
        )
    for name in ALL_BUCKETS:
        cells = []
        per_season[name] = {}
        for season in seasons:
            moments = season_moments[season]
            value = bucket_entry(name, moments, moments.same_team, moments.cross_team)
            se = bucket_se(name, moments)
            per_season[name][str(season)] = {
                "estimate": value,
                "standard_error": se,
                "signal_to_se": value / se if se > 0 else float("nan"),
            }
            cells.append(f"{value:+.5f}\n±{se:.5f}")
        season_table.add_row(name, *cells)
    console.print(season_table)

    sign_stability = {}
    for name in ALL_BUCKETS:
        values = [per_season[name][str(s)]["estimate"] for s in seasons]
        positive = sum(1 for value in values if value > 0)
        significant = [
            per_season[name][str(s)]["signal_to_se"]
            for s in seasons
            if abs(per_season[name][str(s)]["signal_to_se"]) >= 1.96
        ]
        sign_stability[name] = {
            "seasons": len(values),
            "seasons_positive": positive,
            "seasons_negative": len(values) - positive,
            "sign_consistent": positive in (0, len(values)),
            "seasons_individually_significant": len(significant),
            "min_estimate": float(min(values)),
            "max_estimate": float(max(values)),
            "spread": float(max(values) - min(values)),
        }
    console.print("sign / significance stability:")
    for name, record in sign_stability.items():
        console.print(
            f"  {name:26s} positive in {record['seasons_positive']}/"
            f"{record['seasons']} seasons, individually significant in "
            f"{record['seasons_individually_significant']}, "
            f"range [{record['min_estimate']:+.5f}, {record['max_estimate']:+.5f}]"
        )

    console.rule("Role-bucket stability")
    per_role: dict[str, dict[str, dict[str, float]]] = {}
    roles = sorted(standardized["role_bucket"].dropna().unique())
    role_table = Table(title="per-role pooled estimate (within-role pairs only)")
    role_table.add_column("bucket")
    for role in roles:
        role_table.add_column(str(role), justify="right")
    role_moments = {}
    for role in roles:
        subset = standardized[standardized["role_bucket"] == role]
        role_moments[role] = pair_moments(
            subset, STATS, bootstrap=args.bootstrap, seed=args.seed
        )
    for name in ALL_BUCKETS:
        cells = []
        per_role[name] = {}
        for role in roles:
            moments = role_moments[role]
            value = bucket_entry(name, moments, moments.same_team, moments.cross_team)
            se = bucket_se(name, moments)
            per_role[name][str(role)] = {
                "estimate": value,
                "standard_error": se,
                "signal_to_se": value / se if se > 0 else float("nan"),
            }
            cells.append(f"{value:+.5f}\n±{se:.5f}")
        role_table.add_row(name, *cells)
    console.print(role_table)

    role_heterogeneity = {}
    for name in ALL_BUCKETS:
        values = np.array([per_role[name][str(r)]["estimate"] for r in roles])
        errors = np.array([per_role[name][str(r)]["standard_error"] for r in roles])
        # Is the spread across roles larger than the measurement noise?
        spread = float(values.max() - values.min())
        pooled_se = float(np.sqrt(np.sum(errors**2)))
        role_heterogeneity[name] = {
            "spread": spread,
            "combined_se": pooled_se,
            "spread_over_se": spread / pooled_se if pooled_se > 0 else float("nan"),
            "heterogeneity_supported": bool(spread > 1.96 * pooled_se),
        }
    console.print("role heterogeneity (spread vs combined SE):")
    for name, record in role_heterogeneity.items():
        verdict = "supported" if record["heterogeneity_supported"] else "not supported"
        console.print(
            f"  {name:26s} spread {record['spread']:+.5f}  "
            f"spread/SE {record['spread_over_se']:.2f}  -> {verdict}"
        )

    console.rule("Bootstrap uncertainty of the pooled estimate")
    bootstrap_ci: dict[str, dict[str, float]] = {}
    for name in ALL_BUCKETS:
        observed = diagnostics[name]["observed"]
        se = diagnostics[name]["standard_error"]
        bootstrap_ci[name] = {
            "estimate": float(observed),
            "standard_error": float(se),
            "ci95_low": float(observed - 1.96 * se),
            "ci95_high": float(observed + 1.96 * se),
            "excludes_zero": bool(abs(observed) > 1.96 * se),
        }
        console.print(
            f"  {name:26s} {observed:+.6f}  CI95 "
            f"[{bootstrap_ci[name]['ci95_low']:+.6f}, "
            f"{bootstrap_ci[name]['ci95_high']:+.6f}]  "
            f"{'excludes 0' if bootstrap_ci[name]['excludes_zero'] else 'includes 0'}"
        )

    console.rule("Rank sweep: which channel does each target lose through?")
    sweep = rank_sweep(same_thresholded, cross_thresholded, STATS)
    sweep_table = Table(title="fitted same-team entry by (k_game, r_contrast)")
    for column in (
        "k_game",
        "r_contrast",
        "reb_reb",
        "ast_ast",
        "max|dev| vs shrunk",
        "RMSE vs shrunk",
    ):
        sweep_table.add_column(column, justify="right")
    for row in sweep:
        sweep_table.add_row(
            str(row["k_game"]),
            str(row["r_contrast"]),
            f"{row['teammate_reb_reb']:+.6f}",
            f"{row['teammate_ast_ast']:+.6f}",
            f"{row['max_abs_deviation_from_shrunk']:.9f}",
            f"{row['rmse_vs_shrunk']:.6f}",
        )
    console.print(sweep_table)
    console.print(
        "At full rank the construction reproduces the shrunk block exactly, so "
        "the rank component is a parsimony cost, not a statistical one."
    )

    payload = {
        "scope": "pre-2024 training history only",
        "holdout_seasons_excluded": list(HOLDOUT_SEASONS),
        "seasons_used": seasons,
        "stats": list(STATS),
        "bootstrap_draws": int(args.bootstrap),
        "seed": int(args.seed),
        "soft_threshold_z_v1": DEFAULT_SHRINK_Z,
        "standardization_moments": {
            stat: {"mean": value[0], "sd": value[1]}
            for stat, value in moment_constants.items()
        },
        "counts": counts,
        "attenuation_decomposition": diagnostics,
        "per_season": per_season,
        "sign_stability": sign_stability,
        "per_role": per_role,
        "role_heterogeneity": role_heterogeneity,
        "bootstrap_ci": bootstrap_ci,
        "rank_sweep": sweep,
        "shrunk_same_team_1p96": same_thresholded.tolist(),
        "shrunk_cross_team_1p96": cross_thresholded.tolist(),
        "observed_same_team": pooled.same_team.tolist(),
        "observed_cross_team": pooled.cross_team.tolist(),
        "observed_same_team_se": pooled.same_team_se.tolist(),
        "observed_cross_team_se": pooled.cross_team_se.tolist(),
    }

    artifact_dir = Path(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "bucket_diagnostics.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    console.print(f"\nwrote {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
