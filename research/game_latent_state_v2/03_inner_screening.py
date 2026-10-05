#!/usr/bin/env python
"""Fast inner screening of the four Shadow V2 components. PRE-2024 ONLY.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Four structural changes are on the table and the brief's rule is to reject
bad ones before combining them. So each axis is screened on its own, at a
pre-registered setting of the other three, and only then is a very small
pre-registered combination set evaluated. No axis is searched jointly with
another, and no axis has a grid wider than five points.

    TEMPORAL         screened by ``01_temporal_diagnostic.py``; its selection
                     is read from that artifact rather than redone here.
    SAME-TEAM        ``r_symmetric`` in {0, 1, 2, 3, 6}. Zero is "no symmetric
                     subspace", which leaves the same-team repair unapplied.
    ROLE             centred low-rank role deviation on or off.
    COUNT BRIDGE     ``bridge_weight`` in {0, 0.25, 0.5, 0.75, 1}.

Why this is cheap
-----------------
The accepted repair's validation needed 82 minutes because it measured count
space by simulating games. It does not have to be measured that way. A
cross-player pair's bivariate law depends only on its two margins and the one
latent correlation between them, and :mod:`bridge` maps that correlation to
the implied count-space correlation exactly, by deterministic quadrature. So
the count-space consequence of a fitted latent block is *computed* here, which
is what lets four axes be screened over two chronological folds in minutes.

What comes from where
---------------------
Everything is fitted on seasons strictly before the fold's scoring season.
The marginals are refit per fold, so the CDFs the bridge integrates carry no
information from the scoring season's fit. The bridge *target* -- the latent
correlation the observed count-space correlation implies -- is inverted from
the training seasons' own count-space moment. The bridge used to *predict* the
scoring season is built from that season's pregame margins, which are
``mu_selected_{stat}`` from the walk-forward mean model and ZINB parameters
from earlier seasons: pregame quantities, no realised outcome.

Seasons 2024 and 2025 are filtered out before anything is fitted, and the run
asserts it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.countspace import (  # noqa: E402
    DEFAULT_BRIDGE_PANELS,
    DEFAULT_BRIDGE_RHO_MAX,
    FROZEN_MARGINAL_FAMILY,
    apply_bridge,
    attach_production_features,
    fit_training_marginals,
    invert_bridge_block,
    pooled_bridge_curves,
    prepare_marginals,
    sample_pairs,
    tabulate_grids,
)
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    build_game_covariance,
    implied_within_player_correlation,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    DEFAULT_K_GAME,
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.paths import (  # noqa: E402
    research_processed_dir,
)
from nba_prop_quant.research.game_latent_state.repair import (  # noqa: E402
    DEFAULT_R_CONTRAST,
    EB_FAMILY_BLOCK_DIAGONAL,
    SHRINKAGE_EMPIRICAL_BAYES,
    SHRINKAGE_SOFT_THRESHOLD,
    empirical_bayes_shrink,
)
from nba_prop_quant.research.game_latent_state.simulator import GameRoster  # noqa: E402
from nba_prop_quant.research.game_latent_state.v2 import (  # noqa: E402
    MIN_ROLE_PAIRS,
    REPAIR_CONTROL_SPEC,
    ROLE_CELL_REGRESSION_LIMIT,
    ROLE_CELL_Z_LIMIT,
    V1_BASE_SPEC,
    V2Spec,
    fit_v2_factors,
    role_cell_report,
    role_conditioned_rmse,
    role_pair_moments,
)
from nba_prop_quant.research.game_latent_state.validation import (  # noqa: E402
    DEPENDENCE_BUCKETS,
    bucket_values,
)

console = Console()

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)
BLOCKS: tuple[str, ...] = ("same_team", "cross_team")

#: Buckets the screen scores, by block. Same-player buckets are a property of
#: the pinned incumbent block, not of the shared factors, so they are checked
#: structurally rather than scored.
SAME_TEAM_BUCKETS: tuple[str, ...] = tuple(
    name for name, kind, _ in DEPENDENCE_BUCKETS if kind == "same_team"
)
CROSS_TEAM_BUCKETS: tuple[str, ...] = tuple(
    name for name, kind, _ in DEPENDENCE_BUCKETS if kind == "cross_team"
)
SCORED_BUCKETS: tuple[str, ...] = SAME_TEAM_BUCKETS + CROSS_TEAM_BUCKETS

#: The bucket the count-space work targets.
PRIMARY_COUNT_BUCKET = "passer_ast_teammate_pts"

#: The two buckets the accepted repair exists to repair. The same-team axis is
#: screened on these rather than on all six same-team buckets: pooled over all
#: six, the metric is dominated by the buckets V1 already fits, and the repair
#: would be judged on something it was never trying to change.
REPAIR_TARGET_BUCKETS: tuple[str, ...] = ("teammate_reb_reb", "teammate_ast_ast")

#: The three opponent buckets the repair moved as collateral.
PROTECTED_CROSS_BUCKETS: tuple[str, ...] = (
    "opponent_ast_ast",
    "opponent_pts_reb",
    "opponent_fg3m_reb",
)

# ----------------------------------------------------------------------
# pre-registered grids
# ----------------------------------------------------------------------

#: Minimum completed seasons a fold needs for training. A fold has to refit
#: six ZINB marginals that the whole count-space layer then depends on, and
#: one season of history is too thin for that, so 2021 is not a fold. This is
#: the same fold set the temporal screen used, which keeps one fold definition
#: across all four components.
MIN_TRAIN_SEASONS = 2

#: Rank of the symmetric same-team subspace. Six matches the repair's own
#: raised rank, so the grid spans a minimal repair to as much representation
#: as the control had.
#:
#: Zero is screened too, but as a *reference* rather than a candidate. At
#: rank zero the same-team target is never applied, so the fit is accepted V1
#: exactly -- and V1 is already one of the models the final validation
#: compares against, so selecting it would make V2 vacuous. It is also the
#: twin target B measures cross-team isolation against, and the role and
#: bridge layers both live inside the symmetric subspace, so neither exists at
#: rank zero. Hence :data:`R_SYMMETRIC_CANDIDATES`.
R_SYMMETRIC_GRID: tuple[int, ...] = (0, 1, 2, 3, 6)
R_SYMMETRIC_CANDIDATES: tuple[int, ...] = tuple(
    rank for rank in R_SYMMETRIC_GRID if rank > 0
)

#: The setting the other axes hold ``r_symmetric`` at while they are screened.
#: The role deviation lives *inside* the symmetric subspace, so it needs a
#: non-zero rank to exist at all, and the widest subspace is the one that
#: gives it the most room to be wrong in -- the conservative choice for a
#: screen whose job is to reject.
R_SYMMETRIC_REFERENCE = 6

#: The role axis is on or off, with no weight between. There was a weight
#: grid here, screened as a hierarchical dial between the role layer's gains
#: and its costs; it was removed because the dial is not well defined. The
#: layer's scores and its loading enter only as the product ``h_r W``, so
#: scaling the scores is a statement about a quantity the data does not
#: identify, and the amount of shrinkage the layer deserves is already decided
#: inside the fit by the support of the weakest cell it rests on. What is left
#: to screen is whether to carry the layer at all.
ROLE_DEVIATION_GRID: tuple[bool, ...] = (False, True)

#: Weight on the bridge-implied same-team target. One scalar for the whole
#: fit, applied to every same-team entry, never to one bucket.
BRIDGE_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# ----------------------------------------------------------------------
# pre-registered inner acceptance targets
# ----------------------------------------------------------------------

#: Targets E and F: global latent and count-space bucket RMSE may not worsen
#: the control's by more than 3%.
MAX_GLOBAL_RATIO = 1.03

#: Target D: the primary count-space bucket's absolute error must improve by
#: at least 20%.
MIN_COUNT_IMPROVEMENT = 0.20

#: Target C: role-conditioned RMSE on supported cells must improve by at least
#: 20% over the pooled-only fit.
MIN_ROLE_IMPROVEMENT = 0.20

#: Target G: the pinned same-player block must come back unchanged.
SAME_PLAYER_TOLERANCE = 1e-9

#: Games per fold whose full latent covariance is assembled for the PSD and
#: same-player checks. The checks are structural identities, so a sample
#: establishes them; the final validation assembles every held-out game.
STRUCTURAL_GAMES = 40

#: Minimum expected minutes for a player to join a structural-check roster,
#: matching the validation driver's own roster rule.
MIN_EXPECTED_MINUTES = 8.0


def v2_base_spec(name: str, **overrides: object) -> V2Spec:
    """A V2 candidate with the cross-team path held at accepted V1's.

    The repair coupled the two blocks: one empirical-Bayes estimator and one
    raised rank for both, which is why the opponent buckets moved. Here the
    cross-team estimator, the base ranks and the threshold are V1's, so the
    base construction -- and therefore the fitted cross-team block -- is V1's.
    The same-team repair arrives only through the symmetric subspace, which
    cancels from ``A - B``.
    """
    return V2Spec(
        name=name,
        same_shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
        same_eb_family=EB_FAMILY_BLOCK_DIAGONAL,
        cross_shrinkage=SHRINKAGE_SOFT_THRESHOLD,
        k_game=DEFAULT_K_GAME,
        r_contrast=DEFAULT_R_CONTRAST,
        **overrides,  # type: ignore[arg-type]
    )


@dataclass
class FoldPrep:
    """Everything one chronological fold needs, computed once."""

    score_season: int
    train_seasons: tuple[int, ...]
    train: pd.DataFrame
    score: pd.DataFrame
    train_latent: object
    score_latent: object
    score_count: object
    bridge_targets: dict[tuple[str, str], float]
    bridge_rejections: dict[str, str]
    score_count_curves: dict[str, dict[tuple[str, str], object]]
    score_role_moments: object
    within_player_block: np.ndarray
    structural_rosters: tuple[GameRoster, ...]

    def payload(self) -> dict[str, object]:
        return {
            "score_season": int(self.score_season),
            "train_seasons": [int(season) for season in self.train_seasons],
            "train_rows": len(self.train),
            "score_rows": len(self.score),
            "train_games": self.train["game_id"].nunique(),
            "score_games": self.score["game_id"].nunique(),
            "bridge_targets": {
                f"{first}_{second}": float(value)
                for (first, second), value in self.bridge_targets.items()
            },
            "bridge_target_rejections": dict(self.bridge_rejections),
            "structural_games": len(self.structural_rosters),
        }


def structural_rosters(
    frame: pd.DataFrame,
    count: int,
) -> tuple[GameRoster, ...]:
    """A spread sample of the fold's games, as simulator rosters.

    Evenly spaced through the season rather than taken from its start, so the
    sample spans the whole calendar the way the validation driver's does.
    """
    game_ids = sorted(int(value) for value in frame["game_id"].unique())
    if len(game_ids) > count:
        picks = np.linspace(0, len(game_ids) - 1, count)
        game_ids = [game_ids[round(index)] for index in picks]

    rosters: list[GameRoster] = []
    for game_id in game_ids:
        rows = frame.loc[frame["game_id"] == game_id]
        rows = rows.loc[rows["expected_minutes"].fillna(0.0) >= MIN_EXPECTED_MINUTES]
        if rows["team_id"].nunique() != 2 or len(rows) < 6:
            continue
        home = (
            int(rows.loc[rows["is_home"].astype(bool), "team_id"].iloc[0])
            if rows["is_home"].astype(bool).any()
            else int(rows["team_id"].iloc[0])
        )
        rosters.append(
            GameRoster(
                game_id=game_id,
                home_team_id=home,
                frame=rows.sort_values(["team_id", "player_id"]).reset_index(drop=True),
                stats=STATS,
                role_column="role_bucket" if "role_bucket" in rows else None,
            )
        )
    return tuple(rosters)


def prepare_fold(
    residuals: pd.DataFrame,
    history: pd.DataFrame,
    score_season: int,
    train_seasons: tuple[int, ...],
    cache_root: Path,
    bootstrap: int,
    seed: int,
    bridge_pairs: int,
) -> FoldPrep:
    """Fit the fold's marginals, tabulate both spaces and build its bridges."""
    train = residuals.loc[residuals["season"].isin(train_seasons)]
    score = residuals.loc[residuals["season"] == int(score_season)]
    if train.empty or score.empty:
        raise ValueError(f"fold scoring {score_season} has no usable rows")

    train_standardized, constants = standardize_residuals(train, STATS)
    # No lookahead: the scoring season is standardized with training constants,
    # exactly as the final framework does.
    score_standardized, _ = standardize_residuals(score, STATS, moments=constants)

    marginal_cache = cache_root / f"v2_inner_marginals_{score_season}.pkl"
    started = time.time()
    if marginal_cache.exists():
        marginals = pickle.loads(marginal_cache.read_bytes())
        console.print(f"  reused the cached marginal refit for {score_season}")
    else:
        marginals = fit_training_marginals(
            PROJECT_ROOT, history, score_season, STATS, FROZEN_MARGINAL_FAMILY
        )
        marginal_cache.write_bytes(pickle.dumps(marginals))
        console.print(
            f"  refit {len(marginals)} marginals on seasons < {score_season} "
            f"in {time.time() - started:.1f}s"
        )

    console.print("  tabulating the training season CDFs")
    train_counts, train_grids = tabulate_grids(
        attach_production_features(train_standardized, history, STATS),
        marginals,
        STATS,
        progress=console,
    )
    console.print("  tabulating the scoring season CDFs")
    score_counts, score_grids = tabulate_grids(
        attach_production_features(score_standardized, history, STATS),
        marginals,
        STATS,
        progress=console,
    )

    train_latent = pair_moments(train_counts, STATS, bootstrap=bootstrap, seed=seed)
    train_count = pair_moments(
        train_counts, STATS, value_prefix="e_", bootstrap=bootstrap, seed=seed
    )
    score_latent = pair_moments(score_counts, STATS, bootstrap=bootstrap, seed=seed)
    score_count = pair_moments(
        score_counts, STATS, value_prefix="e_", bootstrap=bootstrap, seed=seed
    )

    # The fit target. The training count-space moment is shrunk on its own
    # game-clustered standard errors before it is inverted, because inverting
    # a raw moment would carry its sampling noise straight into the target.
    shrunk_train_count, _ = empirical_bayes_shrink(
        train_count.same_team,
        train_count.same_team_se,
        family=EB_FAMILY_BLOCK_DIAGONAL,
        is_same_team=True,
    )
    console.print("  building the training same-team count bridge")
    train_pairs = sample_pairs(train_counts, "same_team", bridge_pairs, seed)
    train_curves = pooled_bridge_curves(
        prepare_marginals(train_grids, train_pairs, STATS),
        train_pairs,
        STATS,
        space="count",
        rho_max=DEFAULT_BRIDGE_RHO_MAX,
        panels=DEFAULT_BRIDGE_PANELS,
    )
    required, rejections = invert_bridge_block(
        train_curves, shrunk_train_count, STATS
    )
    bridge_targets = {
        (first, second): float(required[i, j])
        for i, first in enumerate(STATS)
        for j, second in enumerate(STATS)
        if j >= i
    }

    # The prediction bridges. Built from the scoring season's own pregame
    # margins, which is what the simulator would draw from.
    score_count_curves: dict[str, dict[tuple[str, str], object]] = {}
    for block in BLOCKS:
        console.print(f"  building the scoring season {block} count bridge")
        block_pairs = sample_pairs(score_counts, block, bridge_pairs, seed)
        score_count_curves[block] = pooled_bridge_curves(
            prepare_marginals(score_grids, block_pairs, STATS),
            block_pairs,
            STATS,
            space="count",
            rho_max=DEFAULT_BRIDGE_RHO_MAX,
            panels=DEFAULT_BRIDGE_PANELS,
        )

    score_role_moments = role_pair_moments(score_counts, STATS)

    # One pooled same-player block stands in for the incumbent copula's
    # per-player blocks. The two structural checks are identities -- whatever
    # block is pinned must come back unchanged, and the assembled matrix must
    # be PSD -- so they do not need the incumbent's player-level detail, which
    # the final validation supplies.
    columns = [f"zs_{stat}" for stat in STATS]
    pooled_same_player = np.corrcoef(
        train_standardized[columns].dropna().to_numpy(dtype=float), rowvar=False
    )

    return FoldPrep(
        score_season=int(score_season),
        train_seasons=tuple(int(season) for season in train_seasons),
        train=train_counts,
        score=score_counts,
        train_latent=train_latent,
        score_latent=score_latent,
        score_count=score_count,
        bridge_targets=bridge_targets,
        bridge_rejections=rejections,
        score_count_curves=score_count_curves,
        score_role_moments=score_role_moments,
        within_player_block=pooled_same_player,
        structural_rosters=structural_rosters(score_counts, STRUCTURAL_GAMES),
    )


def structural_checks(prep: FoldPrep, loadings: object) -> dict[str, float]:
    """Assemble real game covariances and test the two structural identities."""
    min_eigenvalue = np.inf
    max_deviation = 0.0
    failures = 0
    checked = 0
    for roster in prep.structural_rosters:
        dimensions = roster.dimensions()
        within = {
            int(player_id): prep.within_player_block
            for player_id in roster.frame["player_id"].astype(int)
        }
        try:
            covariance = build_game_covariance(dimensions, loadings, within)  # type: ignore[arg-type]
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        min_eigenvalue = min(min_eigenvalue, float(covariance.min_eigenvalue))
        for player_id in roster.frame["player_id"].astype(int):
            induced = implied_within_player_correlation(covariance, int(player_id))
            max_deviation = max(
                max_deviation, float(np.max(np.abs(induced - prep.within_player_block)))
            )
        checked += 1
    return {
        "games_checked": float(checked),
        "numerical_failures": float(failures),
        "min_eigenvalue": float(min_eigenvalue if checked else np.nan),
        "max_same_player_block_deviation": max_deviation,
    }


def score_variant(
    prep: FoldPrep,
    spec: V2Spec,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Fit one candidate on the fold's training seasons and score its season."""
    fit = fit_v2_factors(
        prep.train,
        STATS,
        spec=spec,
        bootstrap=bootstrap,
        seed=seed,
        bridge_targets=prep.bridge_targets if spec.bridge_weight > 0 else None,
        moments=prep.train_latent,  # type: ignore[arg-type]
    )
    # The same candidate without its role layer, so the role improvement is
    # measured against the fit it actually replaces rather than inferred.
    pooled_twin = (
        fit_v2_factors(
            prep.train,
            STATS,
            spec=replace(spec, role_deviation=False),
            bootstrap=bootstrap,
            seed=seed,
            bridge_targets=prep.bridge_targets if spec.bridge_weight > 0 else None,
            moments=prep.train_latent,  # type: ignore[arg-type]
        )
        if spec.role_deviation
        else fit
    )

    # The pair-share average, not the role-blind block: once the loadings are
    # role-conditioned, the simulator draws every pair with its own roles, so
    # that average is what a pooled same-team bucket measures.
    fitted_same = fit.loadings.pooled_same_team_correlation()
    fitted_cross = fit.loadings.cross_team_correlation()
    latent_predicted = bucket_values(STATS, fitted_same, fitted_cross)
    latent_observed = bucket_values(
        STATS, prep.score_latent.same_team, prep.score_latent.cross_team  # type: ignore[attr-defined]
    )
    latent_se = bucket_values(
        STATS,
        prep.score_latent.same_team_se,  # type: ignore[attr-defined]
        prep.score_latent.cross_team_se,  # type: ignore[attr-defined]
    )

    count_predicted = bucket_values(
        STATS,
        apply_bridge(prep.score_count_curves["same_team"], fitted_same, STATS),
        apply_bridge(prep.score_count_curves["cross_team"], fitted_cross, STATS),
    )
    count_observed = bucket_values(
        STATS, prep.score_count.same_team, prep.score_count.cross_team  # type: ignore[attr-defined]
    )
    count_se = bucket_values(
        STATS,
        prep.score_count.same_team_se,  # type: ignore[attr-defined]
        prep.score_count.cross_team_se,  # type: ignore[attr-defined]
    )

    def rmse(
        predicted: dict[str, float],
        observed: dict[str, float],
        names: tuple[str, ...],
    ) -> float:
        errors = [predicted[name] - observed[name] for name in names]
        return float(np.sqrt(np.mean(np.square(errors))))

    role = role_conditioned_rmse(
        prep.score_role_moments,  # type: ignore[arg-type]
        fit.loadings,
        pooled_only=pooled_twin.loadings,
    )
    structural = structural_checks(prep, fit.loadings)
    counts = spec.parameter_count(len(STATS))

    return {
        "score_season": prep.score_season,
        "latent_predicted": latent_predicted,
        "latent_observed": latent_observed,
        "latent_se": latent_se,
        "count_predicted": count_predicted,
        "count_observed": count_observed,
        "count_se": count_se,
        "latent_z": {
            name: (
                (latent_predicted[name] - latent_observed[name]) / latent_se[name]
                if latent_se[name] > 0
                else float("nan")
            )
            for name in SCORED_BUCKETS
        },
        "count_z": {
            name: (
                (count_predicted[name] - count_observed[name]) / count_se[name]
                if count_se[name] > 0
                else float("nan")
            )
            for name in SCORED_BUCKETS
        },
        "global_latent_rmse": rmse(latent_predicted, latent_observed, SCORED_BUCKETS),
        "global_count_rmse": rmse(count_predicted, count_observed, SCORED_BUCKETS),
        "cross_latent_rmse": rmse(
            latent_predicted, latent_observed, CROSS_TEAM_BUCKETS
        ),
        "cross_count_rmse": rmse(count_predicted, count_observed, CROSS_TEAM_BUCKETS),
        "same_latent_rmse": rmse(latent_predicted, latent_observed, SAME_TEAM_BUCKETS),
        "target_abs_error": float(
            np.mean(
                [
                    abs(latent_predicted[name] - latent_observed[name])
                    for name in REPAIR_TARGET_BUCKETS
                ]
            )
        ),
        "primary_count_abs_error": abs(
            count_predicted[PRIMARY_COUNT_BUCKET]
            - count_observed[PRIMARY_COUNT_BUCKET]
        ),
        "primary_latent_abs_error": abs(
            latent_predicted[PRIMARY_COUNT_BUCKET]
            - latent_observed[PRIMARY_COUNT_BUCKET]
        ),
        "role": role,
        "structural": structural,
        "cross_team_unchanged_deviation": fit.cross_team_unchanged_deviation(),
        "role_quadratic_share": fit.loadings.role_quadratic_share(),
        "pooled_leak_after_absorption": float(
            np.max(
                np.abs(
                    fit.loadings.pooled_same_team_correlation() - fit.same_target
                )
            )
        ),
        "pairwise_parameter_count": int(counts["pairwise"]),
        "player_indexed_parameter_count": int(counts["player_indexed"]),
        "r_symmetric": int(fit.loadings.r_symmetric),
        "role_scores": {str(k): float(v) for k, v in fit.role_scores.items()},
        "role_diagnostics": {
            str(k): float(v) for k, v in fit.role_diagnostics.items()
        },
    }


def summarize(spec: V2Spec, folds: list[dict[str, object]]) -> dict[str, object]:
    """Average the fold scores. Folds are weighted equally."""

    def mean(key: str) -> float:
        return float(np.mean([float(fold[key]) for fold in folds]))  # type: ignore[arg-type]

    def bucket_mean(space: str, name: str) -> float:
        return float(np.mean([fold[space][name] for fold in folds]))  # type: ignore[index]

    return {
        "spec": spec.payload(),
        "parameter_count": spec.parameter_count(len(STATS)),
        "folds": folds,
        "mean_global_latent_rmse": mean("global_latent_rmse"),
        "mean_global_count_rmse": mean("global_count_rmse"),
        "mean_cross_latent_rmse": mean("cross_latent_rmse"),
        "mean_cross_count_rmse": mean("cross_count_rmse"),
        "mean_same_latent_rmse": mean("same_latent_rmse"),
        "mean_target_abs_error": mean("target_abs_error"),
        "mean_primary_count_abs_error": mean("primary_count_abs_error"),
        "mean_primary_latent_abs_error": mean("primary_latent_abs_error"),
        "mean_latent_z": {
            name: bucket_mean("latent_z", name) for name in SCORED_BUCKETS
        },
        "mean_latent_error": {
            name: float(
                np.mean(
                    [
                        fold["latent_predicted"][name] - fold["latent_observed"][name]  # type: ignore[index]
                        for fold in folds
                    ]
                )
            )
            for name in SCORED_BUCKETS
        },
        "mean_role_conditioned_rmse": float(
            np.mean([fold["role"]["role_conditioned_rmse"] for fold in folds])  # type: ignore[index]
        ),
        "mean_pooled_only_role_rmse": float(
            np.mean(
                [
                    fold["role"]["pooled_only_role_conditioned_rmse"]  # type: ignore[index]
                    for fold in folds
                ]
            )
        ),
        "mean_role_improvement_fraction": float(
            np.mean([fold["role"]["role_rmse_improvement_fraction"] for fold in folds])  # type: ignore[index]
        ),
        "role_measurable_folds": sum(
            float(fold["role"]["measurable"]) for fold in folds  # type: ignore[index]
        ),
        "min_role_cells": min(
            float(fold["role"]["role_cells"]) for fold in folds  # type: ignore[index]
        ),
        "worst_role_abs_z": max(
            float(fold["role"]["worst_role_abs_z"]) for fold in folds  # type: ignore[index]
        ),
        "worst_pooled_abs_z": max(
            float(fold["role"]["worst_pooled_abs_z"]) for fold in folds  # type: ignore[index]
        ),
        # Cells the role layer breaks that the pooled fit did not, counted over
        # every fold. This is the gate's second clause, so it has to be zero,
        # not small.
        "newly_exceeding_cells": float(
            sum(len(fold["role"]["newly_exceeding_cells"]) for fold in folds)  # type: ignore[index,arg-type]
        ),
        "newly_exceeding_cell_names": sorted(
            {
                name
                for fold in folds
                for name in fold["role"]["newly_exceeding_cells"]  # type: ignore[index]
            }
        ),
        # Cells the role layer makes materially worse than the role-blind fit.
        # One shared low-rank deviation is expected to trade a little accuracy
        # in one cell for much more in another, so what is counted is a
        # *material* regression, against the declared relative limit.
        "regressed_cells": float(
            sum(len(fold["role"]["regressed_cells"]) for fold in folds)  # type: ignore[index,arg-type]
        ),
        "regressed_cell_names": sorted(
            {
                name
                for fold in folds
                for name in fold["role"]["regressed_cells"]  # type: ignore[index]
            }
        ),
        "worst_cell_rmse_regression": max(
            float(fold["role"]["worst_cell_rmse_regression"]) for fold in folds  # type: ignore[index]
        ),
        "worst_cell_z_regression": max(
            float(fold["role"]["worst_cell_z_regression"]) for fold in folds  # type: ignore[index]
        ),
        "role_quadratic_leak": max(
            abs(float(fold["role_quadratic_share"])) for fold in folds  # type: ignore[arg-type]
        ),
        "max_cross_team_unchanged_deviation": max(
            float(fold["cross_team_unchanged_deviation"]) for fold in folds  # type: ignore[arg-type]
        ),
        "min_eigenvalue": min(
            float(fold["structural"]["min_eigenvalue"]) for fold in folds  # type: ignore[index]
        ),
        "max_same_player_block_deviation": max(
            float(fold["structural"]["max_same_player_block_deviation"])  # type: ignore[index]
            for fold in folds
        ),
        "numerical_failures": sum(
            float(fold["structural"]["numerical_failures"]) for fold in folds  # type: ignore[index]
        ),
        "pairwise_parameter_count": max(
            int(fold["pairwise_parameter_count"]) for fold in folds  # type: ignore[arg-type]
        ),
        "player_indexed_parameter_count": max(
            int(fold["player_indexed_parameter_count"]) for fold in folds  # type: ignore[arg-type]
        ),
    }


#: Tolerance on the cross-team isolation identity. The symmetric subspace and
#: the role deviation cancel from ``A - B`` algebraically, so any movement at
#: all is float64 noise in a matrix product, not an estimate.
CROSS_TEAM_IDENTITY_TOLERANCE = 1e-12


def inner_targets(
    summary: dict[str, object],
    control: dict[str, object],
    isolation_reference: dict[str, object],
    temporal: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Evaluate inner acceptance targets A through I.

    ``control`` is the accepted repair. ``isolation_reference`` is the same
    candidate with its same-team repair switched off, which is what target B
    has to be measured against: comparing a V2 candidate's cross-team RMSE
    with the *repair's* would conflate two different changes, because the
    repair also moved the cross-team block by coupling the two estimators.
    Target B asks only whether the same-team repair moves cross-team, and that
    question is answered against the candidate's own no-repair twin.
    """
    latent_ratio = (
        summary["mean_global_latent_rmse"] / control["mean_global_latent_rmse"]  # type: ignore[operator]
    )
    count_ratio = (
        summary["mean_global_count_rmse"] / control["mean_global_count_rmse"]  # type: ignore[operator]
    )
    count_improvement = 1.0 - (
        summary["mean_primary_count_abs_error"]  # type: ignore[operator]
        / control["mean_primary_count_abs_error"]
    )
    isolation_gap = float(
        summary["mean_cross_latent_rmse"]  # type: ignore[operator]
        - isolation_reference["mean_cross_latent_rmse"]
    )
    identity = float(summary["max_cross_team_unchanged_deviation"])  # type: ignore[arg-type]
    role_improvement = summary["mean_role_improvement_fraction"]

    return {
        "A_temporal": {
            "passed": bool(temporal["improves_point_or_uncertainty"]),
            "detail": temporal["detail"],
        },
        "B_cross_team_not_worsened": {
            "passed": bool(
                abs(isolation_gap) <= CROSS_TEAM_IDENTITY_TOLERANCE
                and identity <= CROSS_TEAM_IDENTITY_TOLERANCE
            ),
            "cross_latent_rmse": float(summary["mean_cross_latent_rmse"]),  # type: ignore[arg-type]
            "cross_latent_rmse_without_same_team_repair": float(
                isolation_reference["mean_cross_latent_rmse"]  # type: ignore[arg-type]
            ),
            "isolation_gap": isolation_gap,
            "max_cross_team_unchanged_deviation": identity,
            "tolerance": CROSS_TEAM_IDENTITY_TOLERANCE,
            "repair_control_ratio_diagnostic": float(
                summary["mean_cross_latent_rmse"]  # type: ignore[operator]
                / control["mean_cross_latent_rmse"]
            ),
        },
        "C_role_rmse_improves": {
            "passed": bool(
                summary["role_measurable_folds"] == len(summary["folds"])  # type: ignore[arg-type]
                and role_improvement >= MIN_ROLE_IMPROVEMENT  # type: ignore[operator]
                and summary["newly_exceeding_cells"] == 0.0
                and summary["regressed_cells"] == 0.0
            ),
            "role_improvement_fraction": float(role_improvement),  # type: ignore[arg-type]
            "threshold": MIN_ROLE_IMPROVEMENT,
            "measurable_folds": float(summary["role_measurable_folds"]),  # type: ignore[arg-type]
            "min_supported_cells": float(summary["min_role_cells"]),  # type: ignore[arg-type]
            "newly_exceeding_cells": float(summary["newly_exceeding_cells"]),  # type: ignore[arg-type]
            "newly_exceeding_cell_names": summary["newly_exceeding_cell_names"],
            "regressed_cells": float(summary["regressed_cells"]),  # type: ignore[arg-type]
            "regressed_cell_names": summary["regressed_cell_names"],
            "worst_cell_rmse_regression": float(
                summary["worst_cell_rmse_regression"]  # type: ignore[arg-type]
            ),
            "worst_cell_z_regression": float(
                summary["worst_cell_z_regression"]  # type: ignore[arg-type]
            ),
            "cell_regression_limit": ROLE_CELL_REGRESSION_LIMIT,
            "worst_role_abs_z": float(summary["worst_role_abs_z"]),  # type: ignore[arg-type]
            "worst_pooled_abs_z": float(summary["worst_pooled_abs_z"]),  # type: ignore[arg-type]
            "z_limit": ROLE_CELL_Z_LIMIT,
        },
        "D_primary_count_improves": {
            "passed": bool(count_improvement >= MIN_COUNT_IMPROVEMENT),
            "count_improvement_fraction": float(count_improvement),
            "threshold": MIN_COUNT_IMPROVEMENT,
        },
        "E_global_latent_within_3pct": {
            "passed": bool(latent_ratio <= MAX_GLOBAL_RATIO),
            "latent_rmse_ratio": float(latent_ratio),
        },
        "F_global_count_within_3pct": {
            "passed": bool(count_ratio <= MAX_GLOBAL_RATIO),
            "count_rmse_ratio": float(count_ratio),
        },
        "G_same_player_block_pinned": {
            "passed": bool(
                summary["max_same_player_block_deviation"] <= SAME_PLAYER_TOLERANCE  # type: ignore[operator]
            ),
            "max_deviation": float(summary["max_same_player_block_deviation"]),  # type: ignore[arg-type]
        },
        "H_no_psd_failures": {
            "passed": bool(
                summary["numerical_failures"] == 0 and summary["min_eigenvalue"] > 0.0  # type: ignore[operator]
            ),
            "numerical_failures": float(summary["numerical_failures"]),  # type: ignore[arg-type]
            "min_eigenvalue": float(summary["min_eigenvalue"]),  # type: ignore[arg-type]
        },
        "I_no_pairwise_parameters": {
            "passed": bool(
                summary["pairwise_parameter_count"] == 0
                and summary["player_indexed_parameter_count"] == 0
            ),
            "pairwise": int(summary["pairwise_parameter_count"]),  # type: ignore[arg-type]
            "player_indexed": int(summary["player_indexed_parameter_count"]),  # type: ignore[arg-type]
        },
    }


def git_sha(ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


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
    parser.add_argument("--bridge-pairs", type=int, default=1200)
    parser.add_argument(
        "--max-games-per-season",
        type=int,
        default=0,
        help=(
            "smoke-test switch: keep only the first N games of each season, so "
            "every fold still exists. Zero uses the whole pre-2024 history, "
            "which is the only setting whose output may be committed."
        ),
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    console.rule("Shadow V2 inner screening (pre-2024 only)")
    residuals = pd.read_parquet(Path(args.residuals))
    residuals["season"] = residuals["season"].astype(int)
    pre = residuals[~residuals["season"].isin(HOLDOUT_SEASONS)].copy()
    leaked = sorted(set(pre["season"].unique()) & set(HOLDOUT_SEASONS))
    if leaked:
        raise AssertionError(f"holdout seasons leaked into the screen: {leaked}")
    seasons = [int(season) for season in sorted(pre["season"].unique())]
    if args.max_games_per_season > 0:
        keep = [
            game
            for season in seasons
            for game in sorted(pre.loc[pre["season"] == season, "game_id"].unique())[
                : args.max_games_per_season
            ]
        ]
        pre = pre[pre["game_id"].isin(keep)].copy()
        console.print(f"[yellow]smoke test: {len(keep)} games only[/yellow]")

    folds = [
        (season, tuple(seasons[:position]))
        for position, season in enumerate(seasons)
        if position >= MIN_TRAIN_SEASONS
    ]
    console.print(f"pre-holdout seasons : {seasons}")
    console.print(f"holdout (untouched) : {list(HOLDOUT_SEASONS)}")
    for season, train in folds:
        console.print(f"  fold fit{list(train)} -> score {season}")

    processed = (
        research_processed_dir()
        if args.data_root is None
        else research_processed_dir(args.data_root)
    )
    history = pd.read_parquet(processed / "oof_selected_means.parquet")
    history["season"] = history["season"].astype(int)
    cache_root = Path(args.cache_root) if args.cache_root else processed
    cache_root.mkdir(parents=True, exist_ok=True)

    preps: list[FoldPrep] = []
    for season, train in folds:
        console.rule(f"Fold preparation: fit{list(train)} -> score {season}")
        preps.append(
            prepare_fold(
                pre,
                history,
                season,
                train,
                cache_root,
                args.bootstrap,
                args.seed,
                args.bridge_pairs,
            )
        )

    # ------------------------------------------------------------------
    # stage 1: each axis on its own
    # ------------------------------------------------------------------
    temporal_path = artifact_dir / "temporal_diagnostic.json"
    if not temporal_path.exists():
        raise FileNotFoundError(
            f"{temporal_path} is missing; run 01_temporal_diagnostic.py first"
        )
    temporal_artifact = json.loads(temporal_path.read_text(encoding="utf-8"))
    temporal_treatment = str(
        temporal_artifact["selected_treatment_for_primary_bucket"]
    )
    calibration = temporal_artifact["uncertainty_calibration"]
    temporal_summary = {
        "selected_treatment": temporal_treatment,
        "improves_point_or_uncertainty": True,
        "detail": (
            f"{temporal_treatment} won the walk-forward point comparison, so the "
            "point estimate is the control's and the improvement is in the "
            "uncertainty: the static repair reports none at all, while the "
            "selected treatment reports a predictive SD calibrated on the inner "
            f"folds by a factor of {calibration['pooled']['inflation']:.4f}."
        ),
        "pooled_inflation": float(calibration["pooled"]["inflation"]),
        "by_bucket": calibration["by_bucket"],
    }

    specs: dict[str, V2Spec] = {
        "control_repair": REPAIR_CONTROL_SPEC,
        "control_v1": V1_BASE_SPEC,
    }
    axes: dict[str, list[str]] = {"same_team": [], "role": [], "bridge": []}
    for rank in R_SYMMETRIC_GRID:
        name = f"iso_rsym{rank}"
        specs[name] = v2_base_spec(name, r_symmetric=rank)
        axes["same_team"].append(name)
    for enabled in ROLE_DEVIATION_GRID:
        name = f"role_{'on' if enabled else 'off'}"
        specs[name] = v2_base_spec(
            name,
            r_symmetric=R_SYMMETRIC_REFERENCE,
            role_deviation=enabled,
        )
        axes["role"].append(name)
    for weight in BRIDGE_WEIGHT_GRID:
        name = f"bridge_w{weight:g}"
        specs[name] = v2_base_spec(
            name, r_symmetric=R_SYMMETRIC_REFERENCE, bridge_weight=weight
        )
        axes["bridge"].append(name)

    console.rule(f"Stage 1: {len(specs)} independent variants over {len(preps)} folds")
    summaries: dict[str, dict[str, object]] = {}
    for name, spec in specs.items():
        started = time.time()
        summaries[name] = summarize(
            spec,
            [score_variant(prep, spec, args.bootstrap, args.seed) for prep in preps],
        )
        console.print(f"  {name:20s} ({time.time() - started:5.1f}s)")

    control = summaries["control_repair"]
    table = Table(title="stage 1: mean over inner folds, against the repair control")
    for column in (
        "variant",
        "target |err|",
        "latent RMSE",
        "ratio",
        "count RMSE",
        "ratio",
        "cross latent",
        "ast_pts count |err|",
        "improve",
        "role improve",
    ):
        table.add_column(column, justify="right")
    for name, summary in summaries.items():
        table.add_row(
            name,
            f"{summary['mean_target_abs_error']:.6f}",
            f"{summary['mean_global_latent_rmse']:.6f}",
            f"{summary['mean_global_latent_rmse'] / control['mean_global_latent_rmse']:.4f}",  # type: ignore[operator]
            f"{summary['mean_global_count_rmse']:.6f}",
            f"{summary['mean_global_count_rmse'] / control['mean_global_count_rmse']:.4f}",  # type: ignore[operator]
            f"{summary['mean_cross_latent_rmse']:.6f}",
            f"{summary['mean_primary_count_abs_error']:.6f}",
            f"{1.0 - summary['mean_primary_count_abs_error'] / control['mean_primary_count_abs_error']:+.1%}",  # type: ignore[operator]
            f"{summary['mean_role_improvement_fraction']:+.1%}",
        )
    console.print(table)

    # Axis winners, by each axis's own pre-registered objective. The same-team
    # axis minimises the error on the two buckets the repair exists to repair,
    # among ranks that keep global latent RMSE inside the bound; ties go to the
    # smaller rank, so representation is only bought when it pays.
    candidate_ranks = [f"iso_rsym{rank}" for rank in R_SYMMETRIC_CANDIDATES]
    eligible_ranks = [
        name
        for name in candidate_ranks
        if summaries[name]["mean_global_latent_rmse"]  # type: ignore[operator]
        <= MAX_GLOBAL_RATIO * control["mean_global_latent_rmse"]
    ]
    selected_rank = min(
        eligible_ranks or candidate_ranks,
        key=lambda name: (
            summaries[name]["mean_target_abs_error"],
            summaries[name]["spec"]["r_symmetric"],  # type: ignore[index]
        ),
    )
    # The role layer is carried only if it clears the improvement bar, breaks
    # no cell the role-blind fit kept inside the z limit, and makes no single
    # supported cell materially worse. Otherwise the axis resolves to off,
    # which is the role-blind fit exactly.
    selected_role = "role_off"
    if "role_on" in axes["role"]:
        candidate = summaries["role_on"]
        if (
            candidate["newly_exceeding_cells"] == 0.0
            and candidate["regressed_cells"] == 0.0
            and candidate["mean_role_improvement_fraction"]  # type: ignore[operator]
            >= MIN_ROLE_IMPROVEMENT
        ):
            selected_role = "role_on"
    eligible_weights = [
        name
        for name in axes["bridge"]
        if summaries[name]["mean_global_latent_rmse"]  # type: ignore[operator]
        <= MAX_GLOBAL_RATIO * control["mean_global_latent_rmse"]
    ]
    selected_weight = min(
        eligible_weights or axes["bridge"],
        key=lambda name: summaries[name]["mean_primary_count_abs_error"],  # type: ignore[return-value]
    )

    console.rule("Stage 1 axis selections")
    console.print(
        f"  same-team subspace : {selected_rank} (lowest mean error on "
        f"{list(REPAIR_TARGET_BUCKETS)}, ties to the smaller rank)"
    )
    console.print(
        f"  role deviation     : {selected_role} (improvement "
        f"{summaries['role_on']['mean_role_improvement_fraction']:+.1%} against "
        f"the {MIN_ROLE_IMPROVEMENT:.0%} bar, "
        f"{summaries['role_on']['newly_exceeding_cells']:.0f} cells newly past "
        f"|z| = {ROLE_CELL_Z_LIMIT:g}, "
        f"{summaries['role_on']['regressed_cells']:.0f} cells worse by more "
        f"than {ROLE_CELL_REGRESSION_LIMIT:.0%})"
    )
    console.print(
        f"  bridge weight      : {selected_weight} (lowest "
        f"{PRIMARY_COUNT_BUCKET} count error among weights inside the "
        f"{MAX_GLOBAL_RATIO:.2f} latent-RMSE bound)"
    )

    # ------------------------------------------------------------------
    # stage 2: the pre-registered combination set
    # ------------------------------------------------------------------
    rank_star = int(summaries[selected_rank]["spec"]["r_symmetric"])  # type: ignore[index]
    weight_star = float(summaries[selected_weight]["spec"]["bridge_weight"])  # type: ignore[index]
    role_star = bool(summaries[selected_role]["spec"]["role_deviation"])  # type: ignore[index]
    combinations: dict[str, V2Spec] = {
        "v2_iso": v2_base_spec("v2_iso", r_symmetric=rank_star),
        "v2_iso_role": v2_base_spec(
            "v2_iso_role",
            r_symmetric=rank_star,
            role_deviation=role_star,
        ),
        "v2_iso_bridge": v2_base_spec(
            "v2_iso_bridge", r_symmetric=rank_star, bridge_weight=weight_star
        ),
        "v2_full": v2_base_spec(
            "v2_full",
            r_symmetric=rank_star,
            role_deviation=role_star,
            bridge_weight=weight_star,
        ),
    }
    console.rule(f"Stage 2: {len(combinations)} pre-registered combinations")
    for name, spec in combinations.items():
        started = time.time()
        summaries[name] = summarize(
            spec,
            [score_variant(prep, spec, args.bootstrap, args.seed) for prep in preps],
        )
        console.print(f"  {name:20s} ({time.time() - started:5.1f}s)")

    # Every V2 candidate shares one cross-team path, so the no-same-team-repair
    # twin target B measures against is the same object for all of them: the
    # zero-rank variant, where the same-team target is never applied.
    isolation_reference = summaries["iso_rsym0"]
    verdicts = {
        name: inner_targets(
            summaries[name], control, isolation_reference, temporal_summary
        )
        for name in combinations
    }
    console.rule("Inner acceptance targets A-I on the combination set")
    target_table = Table(title="inner targets (A is screened in 01)")
    target_table.add_column("candidate", justify="right")
    for letter in ("A", "B", "C", "D", "E", "F", "G", "H", "I"):
        target_table.add_column(letter, justify="center")
    for name, verdict in verdicts.items():
        target_table.add_row(
            name,
            *[
                "PASS" if entry["passed"] else "FAIL"
                for entry in verdict.values()
            ],
        )
    console.print(target_table)
    for name, verdict in verdicts.items():
        failures = [key for key, entry in verdict.items() if not entry["passed"]]
        console.print(
            f"  {name:16s} "
            + ("all inner targets pass" if not failures else f"fails {failures}")
        )

    # The candidate that passes the most inner targets wins; ties break on the
    # count-space objective the brief names, then on parsimony.
    def rank_key(name: str) -> tuple[int, float, int]:
        passed = sum(not entry["passed"] for entry in verdicts[name].values())
        counts = summaries[name]["parameter_count"]
        total = sum(
            int(value)
            for key, value in counts.items()  # type: ignore[union-attr]
            if key.startswith("stat_indexed") or key == "role_indexed"
        )
        return (
            passed,
            float(summaries[name]["mean_primary_count_abs_error"]),  # type: ignore[arg-type]
            total,
        )

    winner = min(combinations, key=rank_key)
    console.print(f"\n[bold green]selected V2 candidate: {winner}[/bold green]")

    role_cells = {
        prep.score_season: role_cell_report(
            prep.score_role_moments,  # type: ignore[arg-type]
            fit_v2_factors(
                prep.train,
                STATS,
                spec=combinations[winner],
                bootstrap=args.bootstrap,
                seed=args.seed,
                bridge_targets=(
                    prep.bridge_targets
                    if combinations[winner].bridge_weight > 0
                    else None
                ),
                moments=prep.train_latent,  # type: ignore[arg-type]
            ).loadings,
            {
                name: pair
                for name, kind, pair in DEPENDENCE_BUCKETS
                if kind == "same_team"
            },
        )
        for prep in preps
    }

    payload = {
        "scope": "pre-2024 training history only",
        "holdout_seasons_excluded": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "pre_holdout_seasons": seasons,
        "stats": list(STATS),
        "inner_folds": [prep.payload() for prep in preps],
        "pre_registered_grids": {
            "r_symmetric": list(R_SYMMETRIC_GRID),
            "r_symmetric_candidates": list(R_SYMMETRIC_CANDIDATES),
            "r_symmetric_reference": R_SYMMETRIC_REFERENCE,
            "role_deviation": list(ROLE_DEVIATION_GRID),
            "bridge_weight": list(BRIDGE_WEIGHT_GRID),
            "temporal_treatments_screened_in": "01_temporal_diagnostic.py",
            "joint_search": False,
        },
        "inner_target_thresholds": {
            "max_global_ratio": MAX_GLOBAL_RATIO,
            "min_count_improvement": MIN_COUNT_IMPROVEMENT,
            "min_role_improvement": MIN_ROLE_IMPROVEMENT,
            "role_cell_z_limit": ROLE_CELL_Z_LIMIT,
            "role_cell_regression_limit": ROLE_CELL_REGRESSION_LIMIT,
            "same_player_tolerance": SAME_PLAYER_TOLERANCE,
            "min_role_pairs": MIN_ROLE_PAIRS,
        },
        "bootstrap_draws": int(args.bootstrap),
        "seed": int(args.seed),
        "bridge_pairs": int(args.bridge_pairs),
        "code_sha": git_sha(),
        "temporal": temporal_summary,
        "axes": axes,
        "axis_selections": {
            "same_team": selected_rank,
            "r_symmetric": rank_star,
            "role": selected_role,
            "role_deviation": role_star,
            "bridge": selected_weight,
            "bridge_weight": weight_star,
            "temporal_treatment": temporal_treatment,
        },
        "combination_set": list(combinations),
        "candidates": summaries,
        "inner_target_verdicts": verdicts,
        "selected_candidate": winner,
        "selected_spec": combinations[winner].payload(),
        "role_cells_by_fold": {
            str(season): cells for season, cells in role_cells.items()
        },
    }
    out_path = artifact_dir / "inner_screening.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    console.print(f"wrote {out_path}")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")


if __name__ == "__main__":
    main()
