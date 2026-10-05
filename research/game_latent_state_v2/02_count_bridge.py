#!/usr/bin/env python
"""Build the discrete Gaussian-copula count bridge on PRE-2024 data only.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Accepted V1 and the accepted bucket repair both match the *latent*
``passer_ast_teammate_pts`` bucket and both miss it in *count* space by
roughly 0.0213. This script establishes why, and converts the answer into a
target the PSD factor fit can aim at.

Three things are computed, all from seasons before 2024:

1.  **The count-space pooled moments.** The production marginals are refit on
    seasons ``< 2024`` -- the same ``fit_season`` call the validation driver
    makes, so the no-lookahead path is the existing one -- and every training
    player-game is standardized by its own analytic marginal mean and
    standard deviation. Pooling those over same-team and cross-team
    cross-player pairs gives the observed count-space correlation per bucket,
    with game-clustered standard errors.

2.  **The two bridges, per block.** For a deterministic sample of ordered
    pairs drawn the way the block itself is pooled, :mod:`bridge` integrates
    Plackett's kernel to give the exact map from latent ``rho`` to count-space
    correlation, and the exact map from latent ``rho`` to the correlation a
    *randomised-PIT* residual would measure. Both are deterministic
    quadrature, not Monte Carlo. Same-team and cross-team get their own
    curves because they pool over different pairs of margins: a same-team
    pair is two players sharing a roster, a cross-team pair is not, and the
    pooled bridge is a mean over exactly those margin pairs.

3.  **The implied latent correlation.** Inverting the count bridge at the
    shrunk observed count correlation gives ``rho_required``. Inverting the
    latent bridge at the shrunk observed latent correlation gives a second,
    independent estimate of the same quantity. If the Gaussian copula were the
    true dependence family the two would agree, and their disagreement is
    therefore the measurement that says how much of the count-space miss a
    latent-correlation change can actually repair.

The observations are shrunk before inversion, each with its own
empirical-Bayes family on its own game-clustered standard errors. Inverting a
raw moment would import its sampling noise into the fit target, which is the
mistake the whole shrinkage layer exists to avoid.

Nothing here selects a hyperparameter. The bridge weight is chosen by
``03_inner_screening.py`` on chronological pre-2024 folds.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.bridge import (  # noqa: E402
    DEFAULT_GAUSS_NODES,
    BridgeNotIdentified,
)
from nba_prop_quant.research.game_latent_state.countspace import (  # noqa: E402
    DEFAULT_BRIDGE_PAIRS,
    DEFAULT_BRIDGE_PANELS,
    DEFAULT_BRIDGE_RHO_MAX,
    FROZEN_MARGINAL_FAMILY,
    attach_production_features,
    fit_training_marginals,
    pooled_bridge_curves,
    prepare_marginals,
    sample_pairs,
    tabulate_grids,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.paths import (  # noqa: E402
    research_processed_dir,
)
from nba_prop_quant.research.game_latent_state.repair import (  # noqa: E402
    EB_FAMILY_BLOCK_DIAGONAL,
    empirical_bayes_shrink,
)

console = Console()

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)
BLOCKS: tuple[str, ...] = ("same_team", "cross_team")

#: Named buckets the report calls out. Every stat pair gets a bridge in both
#: blocks regardless; these are the ones printed.
NAMED_BUCKETS: dict[str, tuple[str, str, str]] = {
    "passer_ast_teammate_pts": ("same_team", "ast", "pts"),
    "teammate_reb_reb": ("same_team", "reb", "reb"),
    "teammate_ast_ast": ("same_team", "ast", "ast"),
    "teammate_pts_reb": ("same_team", "pts", "reb"),
    "teammate_pts_pts": ("same_team", "pts", "pts"),
    "opponent_ast_ast": ("cross_team", "ast", "ast"),
    "opponent_pts_reb": ("cross_team", "pts", "reb"),
    "opponent_fg3m_reb": ("cross_team", "fg3m", "reb"),
}


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
        default=str(PROJECT_ROOT / "research" / "game_latent_state_v2"),
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--bridge-pairs", type=int, default=DEFAULT_BRIDGE_PAIRS)
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help=(
            "smoke-test switch: keep only the first N training games. Zero "
            "uses the whole pre-2024 history, which is the only setting whose "
            "output may be committed."
        ),
    )
    args = parser.parse_args()

    console.rule("Shadow V2 count bridge (pre-2024 only)")
    residuals = pd.read_parquet(Path(args.residuals))
    residuals["season"] = residuals["season"].astype(int)
    pre = residuals[~residuals["season"].isin(HOLDOUT_SEASONS)].copy()
    leaked = sorted(set(pre["season"].unique()) & set(HOLDOUT_SEASONS))
    if leaked:
        raise AssertionError(f"holdout seasons leaked into the bridge: {leaked}")
    seasons = [int(season) for season in sorted(pre["season"].unique())]
    if args.max_games > 0:
        keep = sorted(pre["game_id"].unique())[: args.max_games]
        pre = pre[pre["game_id"].isin(keep)].copy()
        console.print(f"[yellow]smoke test: {len(keep)} games only[/yellow]")
    console.print(f"seasons used: {seasons}   rows {len(pre):,}")

    processed = (
        research_processed_dir()
        if args.data_root is None
        else research_processed_dir(args.data_root)
    )
    history = pd.read_parquet(processed / "oof_selected_means.parquet")
    history["season"] = history["season"].astype(int)
    first_holdout = min(HOLDOUT_SEASONS)

    cache_root = Path(args.cache_root) if args.cache_root else processed
    cache_root.mkdir(parents=True, exist_ok=True)

    console.rule(f"Production marginals refit on seasons < {first_holdout}")
    started = time.time()
    marginal_cache = cache_root / f"v2_training_marginals_{first_holdout}.pkl"
    if marginal_cache.exists():
        marginals = pickle.loads(marginal_cache.read_bytes())
        console.print(f"reused the cached refit from {marginal_cache}")
    else:
        marginals = fit_training_marginals(
            PROJECT_ROOT, history, first_holdout, STATS, FROZEN_MARGINAL_FAMILY
        )
        marginal_cache.write_bytes(pickle.dumps(marginals))
    console.print(
        f"{len(marginals)} marginals over "
        f"{len(history.loc[history['season'] < first_holdout]):,} rows in "
        f"{time.time() - started:.1f}s"
    )

    console.rule("Analytic marginal moments per training player-game")
    standardized, _ = standardize_residuals(pre, STATS)
    featured = attach_production_features(standardized, history, STATS)
    with_counts, grids = tabulate_grids(featured, marginals, STATS, progress=console)

    console.rule("Observed pooled moments, latent and count space")
    latent = pair_moments(with_counts, STATS, bootstrap=args.bootstrap, seed=args.seed)
    count = pair_moments(
        with_counts,
        STATS,
        value_prefix="e_",
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    observed: dict[str, dict[str, np.ndarray]] = {
        "same_team": {
            "latent": latent.same_team,
            "latent_se": latent.same_team_se,
            "count": count.same_team,
            "count_se": count.same_team_se,
        },
        "cross_team": {
            "latent": latent.cross_team,
            "latent_se": latent.cross_team_se,
            "count": count.cross_team,
            "count_se": count.cross_team_se,
        },
    }
    shrunk: dict[str, dict[str, np.ndarray]] = {}
    shrink_diagnostics: dict[str, dict[str, object]] = {}
    for block in BLOCKS:
        shrunk[block] = {}
        shrink_diagnostics[block] = {}
        for space in ("latent", "count"):
            value, diagnostics = empirical_bayes_shrink(
                observed[block][space],
                observed[block][f"{space}_se"],
                family=EB_FAMILY_BLOCK_DIAGONAL,
                is_same_team=(block == "same_team"),
            )
            shrunk[block][space] = value
            shrink_diagnostics[block][space] = diagnostics

    # Each block gets its own pooled bridge. A same-team pair and a cross-team
    # pair average over different margin pairs, and the bridge is a mean over
    # exactly the margin pairs the block pools, so sharing one curve between
    # them would mis-state whichever block did not supply the sample.
    curves: dict[str, dict[str, dict[tuple[str, str], object]]] = {}
    for block in BLOCKS:
        console.rule(
            f"{block} bridges over {args.bridge_pairs:,} deterministic pairs"
        )
        sampled = sample_pairs(with_counts, block, args.bridge_pairs, args.seed)
        prepared = prepare_marginals(grids, sampled, STATS)
        curves[block] = {
            space: pooled_bridge_curves(
                prepared,
                sampled,
                STATS,
                space=space,
                rho_max=DEFAULT_BRIDGE_RHO_MAX,
                panels=DEFAULT_BRIDGE_PANELS,
                nodes=DEFAULT_GAUSS_NODES,
                progress=console,
            )
            for space in ("count", "latent")
        }

    console.rule("Bridge-implied latent correlation per stat pair and block")
    index = {stat: position for position, stat in enumerate(STATS)}
    records: dict[str, dict[str, dict[str, object]]] = {}
    for block in BLOCKS:
        records[block] = {}
        for position, first_stat in enumerate(STATS):
            for second_stat in STATS[position:]:
                i, j = index[first_stat], index[second_stat]
                count_curve = curves[block]["count"][(first_stat, second_stat)]
                latent_curve = curves[block]["latent"][(first_stat, second_stat)]
                record: dict[str, object] = {
                    "stat_pair": [first_stat, second_stat],
                    "block": block,
                    "count_bridge": count_curve.payload(),
                    "latent_bridge": latent_curve.payload(),
                    "count_monotone": count_curve.is_monotone(),
                    "latent_monotone": latent_curve.is_monotone(),
                    "count_slope_at_zero": count_curve.slope_at_zero(),
                    "latent_slope_at_zero": latent_curve.slope_at_zero(),
                    "observed_latent": float(observed[block]["latent"][i, j]),
                    "observed_latent_se": float(observed[block]["latent_se"][i, j]),
                    "shrunk_latent": float(shrunk[block]["latent"][i, j]),
                    "observed_count": float(observed[block]["count"][i, j]),
                    "observed_count_se": float(observed[block]["count_se"][i, j]),
                    "shrunk_count": float(shrunk[block]["count"][i, j]),
                }
                for label, curve, target in (
                    (
                        "required_from_count",
                        count_curve,
                        float(shrunk[block]["count"][i, j]),
                    ),
                    (
                        "required_from_latent",
                        latent_curve,
                        float(shrunk[block]["latent"][i, j]),
                    ),
                ):
                    try:
                        record[label] = curve.invert(target)
                        record[f"{label}_identified"] = True
                        record[f"{label}_rejection"] = ""
                    except BridgeNotIdentified as error:
                        record[label] = None
                        record[f"{label}_identified"] = False
                        record[f"{label}_rejection"] = str(error)
                records[block][f"{first_stat}_{second_stat}"] = record

    table = Table(title="observed -> required latent rho, per named bucket")
    for column in (
        "bucket",
        "block",
        "latent obs",
        "count obs",
        "count slope",
        "latent slope",
        "rho from count",
        "rho from latent",
        "disagreement",
    ):
        table.add_column(column, justify="right")
    agreement: dict[str, dict[str, object]] = {}
    for name, (block, first_stat, second_stat) in NAMED_BUCKETS.items():
        key = (
            f"{first_stat}_{second_stat}"
            if f"{first_stat}_{second_stat}" in records[block]
            else f"{second_stat}_{first_stat}"
        )
        record = records[block][key]
        from_count = record["required_from_count"]
        from_latent = record["required_from_latent"]
        gap = (
            None
            if from_count is None or from_latent is None
            else float(from_count) - float(from_latent)
        )
        agreement[name] = {
            "block": block,
            "stat_pair_key": key,
            "observed_latent": record["observed_latent"],
            "observed_count": record["observed_count"],
            "shrunk_latent": record["shrunk_latent"],
            "shrunk_count": record["shrunk_count"],
            "required_from_count": from_count,
            "required_from_latent": from_latent,
            "disagreement": gap,
        }
        table.add_row(
            name,
            block,
            f"{record['observed_latent']:+.6f}",
            f"{record['observed_count']:+.6f}",
            f"{record['count_slope_at_zero']:.4f}",
            f"{record['latent_slope_at_zero']:.4f}",
            "n/a" if from_count is None else f"{float(from_count):+.6f}",
            "n/a" if from_latent is None else f"{float(from_latent):+.6f}",
            "n/a" if gap is None else f"{gap:+.6f}",
        )
    console.print(table)

    same_gaps = [
        float(value["disagreement"])
        for value in agreement.values()
        if value["block"] == "same_team" and value["disagreement"] is not None
    ]
    console.print(
        "The two rho columns are independent estimates of one latent "
        "correlation: one inverts the count-space observation, the other the "
        "randomised-PIT latent observation. A Gaussian copula with these "
        "margins would make them agree."
    )
    if same_gaps:
        console.print(
            f"They do not. The same-team disagreement is positive in "
            f"{sum(gap > 0 for gap in same_gaps)}/{len(same_gaps)} named "
            f"buckets, mean {np.mean(same_gaps):+.6f}. Count space therefore "
            "carries dependence that no single Gaussian-copula rho reproduces "
            "while also matching the latent measurement, so part of the "
            "count-space miss is copula-family misspecification rather than "
            "transform attenuation. The bridge measures how much."
        )

    payload = {
        "scope": "pre-2024 training history only",
        "holdout_seasons_excluded": list(HOLDOUT_SEASONS),
        "seasons_used": seasons,
        "stats": list(STATS),
        "blocks": list(BLOCKS),
        "marginal_family": FROZEN_MARGINAL_FAMILY,
        "marginal_training_seasons": [
            int(season)
            for season in sorted(
                history.loc[history["season"] < first_holdout, "season"].unique()
            )
        ],
        "bootstrap_draws": int(args.bootstrap),
        "seed": int(args.seed),
        "bridge_pairs": int(args.bridge_pairs),
        "bridge_panels": int(DEFAULT_BRIDGE_PANELS),
        "bridge_nodes": int(DEFAULT_GAUSS_NODES),
        "bridge_rho_max": float(DEFAULT_BRIDGE_RHO_MAX),
        "deterministic": True,
        "monte_carlo_used": False,
        "hyperparameter_selected_here": False,
        "shrinkage_diagnostics": shrink_diagnostics,
        "observed": {
            block: {
                space: observed[block][space].tolist()
                for space in ("latent", "latent_se", "count", "count_se")
            }
            for block in BLOCKS
        },
        "shrunk": {
            block: {space: shrunk[block][space].tolist() for space in ("latent", "count")}
            for block in BLOCKS
        },
        "counts": {
            "games": int(latent.games),
            "same_team_ordered_pairs": float(latent.same_team_pairs),
            "cross_team_ordered_pairs": float(latent.cross_team_pairs),
            "player_games": int(len(with_counts)),
        },
        "bridges": records,
        "named_bucket_agreement": agreement,
        "same_team_mean_disagreement": (
            float(np.mean(same_gaps)) if same_gaps else None
        ),
    }

    artifact_dir = Path(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "count_bridge.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    console.print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
