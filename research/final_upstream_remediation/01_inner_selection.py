#!/usr/bin/env python
"""Inner selection for the final upstream remediation. PRE-2024 DATA ONLY.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Seasons 2024 and 2025 are filtered out of the residual frame on the first read
and never re-admitted, so nothing in this script -- no prior, no rank, no
degrees of freedom, no weight cap -- can have been chosen with knowledge of
them. The assertion that enforces it is :func:`assert_holdout_absent`, which
runs on every frame this script derives rather than once on the input.

Five of the six remediation items are decided here:

1.  the temporal treatment for the season-level level estimate,
2.  the role-scale geometry,
3.  the cross-team prior,
4.  the latent-to-count transmission weight cap,
6.  the predictive-uncertainty scale.

Item 5, the dependence temperature, needs joint-event probabilities and is
decided by ``02_dependence_temperature.py``.

Every decision uses strictly forward chronological folds: fit on seasons before
the fold's target season, score on the target season. Nothing is scored on a
season it was fitted on, and the parsimony rule -- keep the simpler model unless
the more complex one wins by more than one standard error of the *paired*
difference -- is applied mechanically by
:func:`remediation.select_within_tie_band` rather than by eye.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.artifacts import (
    git_sha,
    write_json,
)
from nba_prop_quant.research.game_latent_state.factors import (
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.remediation import (
    COVERAGE_LEVELS,
    CROSS_PRIOR_GAUSSIAN,
    CROSS_PRIOR_STUDENT_T,
    DEFAULT_HALF_LIFE_GRID,
    DEFAULT_NU_GRID,
    MIN_SEASONS_FOR_HETEROGENEITY,
    PARSIMONY_TIE_BAND_SE,
    ROLE_PAIR_CELLS,
    ROLE_SCALE_LOG_SHRUNK,
    ROLE_SCALE_RATIO,
    CONTROL_SPEC,
    RemediationSpec,
    SeasonSeries,
    cluster_mean_and_se,
    coverage_loss,
    coverage_table,
    fit_gaussian_random_effects,
    fit_log_shrunk_role_scales,
    fit_pooled,
    fit_remediated_factors,
    fit_student_t_random_effects,
    forward_fold_seasons,
    forward_predictive_records,
    huber_scale,
    median_of_fold_scales,
    paired_difference_se,
    role_pair_cell_moments,
    role_scale_components,
    select_within_tie_band,
)
from nba_prop_quant.research.game_latent_state.transmission import (
    DEFAULT_BRIDGE_ORDER,
    accumulate_per_game,
    bootstrap_source_pair,
    bridge_forward,
    combine_sources,
    homogeneity_test,
    transmission_coefficient_columns,
)
from nba_prop_quant.research.game_latent_state.validation import (
    DEPENDENCE_BUCKETS,
)

console = Console()

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS = (2024, 2025)
ROLE_COLUMN = "role_bucket"

#: The buckets the dependence model is judged on, cross-player only.
CROSS_PLAYER_BUCKETS = tuple(
    (name, kind, pair)
    for name, kind, pair in DEPENDENCE_BUCKETS
    if kind in {"same_team", "cross_team"}
)

#: Item 1's primary bucket, and item 3's protected cross-team buckets. Both
#: lists come from the brief, not from anything measured here.
PRIMARY_TEMPORAL_BUCKET = "teammate_ast_ast"
PROTECTED_OPPONENT_BUCKETS = (
    "opponent_ast_ast",
    "opponent_pts_reb",
    "opponent_fg3m_reb",
)
PROTECTED_TEAMMATE_BUCKETS = ("teammate_ast_ast", "teammate_reb_reb")

#: Training seasons a fold needs for items 2, 3 and 4. One is enough to fit a
#: factor model; item 1 needs two before a between-season spread is defined.
MIN_FOLD_TRAINING_SEASONS = 1

#: Item 4's weight-cap grid. Coarse on purpose: the cap is one scalar and the
#: inner folds cannot resolve it finely, so offering a fine grid would only
#: invite reading noise as signal.
BRIDGE_CAP_GRID = (0.0, 0.15, 0.30, 0.50, 1.00)

#: Simplicity order for each decision, simplest first. The accepted repair's
#: setting is always first, so a tie keeps the control.
TEMPORAL_ORDER = (
    "A0_pooled_empirical_bayes",
    "A1_gaussian_random_effects",
    "A2_robust_student_t",
    "A3_recency_weighted_robust",
)
UNCERTAINTY_ORDER = (
    "U0_raw",
    "U2_median_of_forward_fold_scales",
    "U1_huber_m_scale",
    "U3_student_t_predictive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        type=Path,
        default=PROJECT_ROOT / "research/game_latent_state/oof_gaussian_residuals.parquet",
    )
    parser.add_argument(
        "--count-residuals",
        type=Path,
        default=PROJECT_ROOT
        / "data/research/game_latent_state/processed/v2_bridge_count_residuals.parquet",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research/final_upstream_remediation",
    )
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--bridge-bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def assert_holdout_absent(frame: pd.DataFrame, where: str) -> pd.DataFrame:
    """Fail loudly if a holdout season reached a selection frame."""
    present = sorted(
        int(season)
        for season in frame["season"].unique()
        if int(season) in HOLDOUT_SEASONS
    )
    if present:
        raise SystemExit(f"{where} contains holdout seasons {present}")
    return frame


def bucket_index(stats):
    return {stat: position for position, stat in enumerate(stats)}


def read_buckets(
    stats,
    same: np.ndarray,
    cross: np.ndarray,
) -> dict[str, float]:
    index = bucket_index(stats)
    out: dict[str, float] = {}
    for name, kind, (first, second) in CROSS_PLAYER_BUCKETS:
        matrix = same if kind == "same_team" else cross
        out[name] = float(matrix[index[first], index[second]])
    return out


# ----------------------------------------------------------------------
# season-level series, shared by items 1 and 6
# ----------------------------------------------------------------------


def season_series(
    frame: pd.DataFrame,
    moments_by_season: dict[int, dict[str, tuple[float, float]]],
    bootstrap: int,
    seed: int,
) -> dict[str, SeasonSeries]:
    """Per-season bucket estimates with game-clustered standard errors.

    Each season is standardized with the constants of the seasons *before* it,
    which is the same no-lookahead rule the fits use. Season 2020 has no
    earlier season, so it is standardized on itself and that is recorded.
    """
    seasons = sorted(int(value) for value in frame["season"].unique())
    estimates: dict[str, list[float]] = {name: [] for name, _, _ in CROSS_PLAYER_BUCKETS}
    errors: dict[str, list[float]] = {name: [] for name, _, _ in CROSS_PLAYER_BUCKETS}

    for season in seasons:
        block = frame.loc[frame["season"] == season]
        standardized, _ = standardize_residuals(
            block, STATS, moments=moments_by_season.get(season)
        )
        pooled = pair_moments(
            standardized, STATS, bootstrap=bootstrap, seed=seed + season
        )
        values = read_buckets(STATS, pooled.same_team, pooled.cross_team)
        standard_errors = read_buckets(
            STATS, pooled.same_team_se, pooled.cross_team_se
        )
        for name in estimates:
            estimates[name].append(values[name])
            errors[name].append(standard_errors[name])

    return {
        name: SeasonSeries(
            name=name,
            seasons=tuple(seasons),
            estimates=np.array(estimates[name]),
            standard_errors=np.array(errors[name]),
        )
        for name in estimates
    }


def decide_temporal(series: dict[str, SeasonSeries]) -> dict[str, object]:
    """Item 1: which temporal treatment predicts an unseen season best.

    The treatment is one choice shared by every bucket, so it is scored on the
    pooled set of (bucket, fold) predictions with the bucket as the clustering
    unit. Scoring it on ``teammate_ast_ast`` alone would leave two
    observations, which cannot separate four candidates from each other.
    """
    nu_choices: dict[float, dict[str, float]] = {}
    for nu in DEFAULT_NU_GRID:
        per_bucket: dict[str, float] = {}
        for name, bucket in series.items():
            records = forward_predictive_records(
                bucket, lambda s, nu=nu: fit_student_t_random_effects(s, nu=nu)
            )
            if records:
                per_bucket[name] = float(np.mean([r["log_score"] for r in records]))
        nu_choices[nu] = per_bucket
    nu_scores = {
        nu: float(np.mean(list(values.values())))
        for nu, values in nu_choices.items()
        if values
    }
    best_nu = min(nu_scores, key=lambda key: nu_scores[key]) if nu_scores else 5.0

    half_life_scores: dict[float, dict[str, float]] = {}
    for half_life in DEFAULT_HALF_LIFE_GRID:
        per_bucket = {}
        for name, bucket in series.items():
            records = forward_predictive_records(
                bucket,
                lambda s, h=half_life: fit_student_t_random_effects(
                    s, nu=best_nu, half_life=h
                ),
            )
            if records:
                per_bucket[name] = float(np.mean([r["log_score"] for r in records]))
        half_life_scores[half_life] = per_bucket
    half_life_means = {
        value: float(np.mean(list(scores.values())))
        for value, scores in half_life_scores.items()
        if scores
    }
    best_half_life = (
        min(half_life_means, key=lambda key: half_life_means[key])
        if half_life_means
        else 2.0
    )

    fitters = {
        "A0_pooled_empirical_bayes": fit_pooled,
        "A1_gaussian_random_effects": fit_gaussian_random_effects,
        "A2_robust_student_t": lambda s: fit_student_t_random_effects(s, nu=best_nu),
        "A3_recency_weighted_robust": lambda s: fit_student_t_random_effects(
            s, nu=best_nu, half_life=best_half_life
        ),
    }

    per_bucket_scores: dict[str, dict[str, float]] = {}
    all_records: dict[str, list[dict[str, float]]] = {}
    for label, fitter in fitters.items():
        records: list[dict[str, float]] = []
        for bucket in series.values():
            records.extend(forward_predictive_records(bucket, fitter))
        all_records[label] = records
        per_bucket_scores[label] = {
            name: float(
                np.mean([r["log_score"] for r in records if r["bucket"] == name])
            )
            for name in series
            if any(r["bucket"] == name for r in records)
        }

    pooled_scores = {}
    pooled_se = {}
    for label, records in all_records.items():
        mean, se = cluster_mean_and_se(
            [r["log_score"] for r in records], [r["bucket"] for r in records]
        )
        pooled_scores[label] = mean
        pooled_se[label] = se

    decision = select_within_tie_band(
        pooled_scores, TEMPORAL_ORDER, per_bucket_scores, PARSIMONY_TIE_BAND_SE
    )

    selected = decision["selected"] or "A0_pooled_empirical_bayes"
    final_fits = {
        name: fitters[selected](bucket).payload() for name, bucket in series.items()
    }

    return {
        "candidates_scored": sorted(fitters),
        "nu_grid": list(DEFAULT_NU_GRID),
        "nu_selected": float(best_nu),
        "nu_forward_log_scores": {str(k): v for k, v in nu_scores.items()},
        "half_life_grid": list(DEFAULT_HALF_LIFE_GRID),
        "half_life_selected": float(best_half_life),
        "forward_fold_target_seasons": [
            int(value)
            for value in forward_fold_seasons(
                next(iter(series.values())).seasons, MIN_SEASONS_FOR_HETEROGENEITY
            )
        ],
        "pooled_forward_log_score": pooled_scores,
        "pooled_forward_log_score_se": pooled_se,
        "per_bucket_forward_log_score": per_bucket_scores,
        "paired_difference_against_control": {
            label: dict(
                zip(
                    ("mean", "standard_error"),
                    paired_difference_se(
                        per_bucket_scores[label],
                        per_bucket_scores["A0_pooled_empirical_bayes"],
                    ),
                )
            )
            for label in per_bucket_scores
        },
        "selection": decision,
        "selected": selected,
        "fits_on_all_training_seasons": final_fits,
        "primary_bucket": PRIMARY_TEMPORAL_BUCKET,
        "primary_bucket_series": {
            "seasons": list(series[PRIMARY_TEMPORAL_BUCKET].seasons),
            "estimates": series[PRIMARY_TEMPORAL_BUCKET].estimates.tolist(),
            "standard_errors": series[
                PRIMARY_TEMPORAL_BUCKET
            ].standard_errors.tolist(),
        },
        "forward_records": {
            label: records for label, records in all_records.items()
        },
    }


def decide_uncertainty(temporal: dict[str, object]) -> dict[str, object]:
    """Item 6: which predictive scale covers an unseen season correctly.

    The scale multiplier for a fold is estimated on folds strictly *earlier*
    than it, which is what makes this cross-fitted rather than a scale read off
    the same residuals it is judged on. The rejected global inflation factor
    was not cross-fitted, which is exactly how it came to double-count one
    season.
    """
    selected = temporal["selected"]
    records = pd.DataFrame(temporal["forward_records"][selected])  # type: ignore[index]
    if records.empty:
        return {"selected": "U0_raw", "reason": "no forward records to calibrate on"}

    records = records.copy()
    records["standardized"] = records["error"] / np.sqrt(
        records["predictive_sd"] ** 2 + records["observed_se"] ** 2
    )
    folds = sorted(records["target_season"].unique())

    def cross_fitted(scale_from) -> pd.DataFrame:
        """Apply, to each fold, a scale estimated only on earlier folds."""
        rows = []
        for position, fold in enumerate(folds):
            history = records.loc[records["target_season"] < fold]
            scale = 1.0 if history.empty else scale_from(history)
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            block = records.loc[records["target_season"] == fold].copy()
            block["scale"] = scale
            block["calibrated"] = block["standardized"] / scale
            block["cross_fitted"] = position > 0
            rows.append(block)
        return pd.concat(rows, ignore_index=True)

    def student_t_predictive_scale(history: pd.DataFrame) -> float:
        """Scale of a Student-t fitted to the standardized errors.

        Reported as the multiplier that makes a *normal* interval cover what
        the fitted t covers at the 68% level, because the coverage table reads
        normal critical values. Matching a quantile rather than the second
        moment is the point: a heavy-tailed error distribution has a second
        moment dominated by its tail, and forcing that to 1 is what made the
        rejected global inflation factor worse than doing nothing.
        """
        sample = history["standardized"].to_numpy(float)
        sample = sample[np.isfinite(sample)]
        if sample.size < 4:
            return 1.0
        degrees, _, scale = stats.t.fit(sample, floc=0.0)
        if not np.isfinite(degrees) or not np.isfinite(scale) or scale <= 0:
            return 1.0
        level = 0.5 + COVERAGE_LEVELS[0] / 2.0
        return float(
            scale * stats.t.ppf(level, df=degrees) / stats.norm.ppf(level)
        )

    candidates = {
        "U0_raw": lambda history: 1.0,
        "U1_huber_m_scale": lambda history: huber_scale(
            history["standardized"].to_numpy(float)
        ),
        "U2_median_of_forward_fold_scales": median_of_fold_scales,
        "U3_student_t_predictive": student_t_predictive_scale,
    }

    scored: dict[str, dict[str, object]] = {}
    per_bucket: dict[str, dict[str, float]] = {}
    losses: dict[str, float] = {}
    for label, rule in candidates.items():
        applied = cross_fitted(rule)
        usable = applied.loc[applied["cross_fitted"]]
        if usable.empty:
            usable = applied
        coverage = coverage_table(usable["calibrated"].to_numpy(float))
        loss = coverage_loss(coverage)
        scored[label] = {
            "coverage": coverage,
            "coverage_loss": loss,
            "scales_by_fold": {
                str(int(row.target_season)): float(row.scale)
                for row in applied.drop_duplicates("target_season").itertuples()
            },
            "cross_fitted_observations": int(len(usable)),
        }
        losses[label] = loss
        # Per-bucket paired score for the tie band: mean absolute coverage
        # error computed within the bucket, so the pairing is on buckets.
        per_bucket[label] = {
            str(bucket): coverage_loss(
                coverage_table(group["calibrated"].to_numpy(float))
            )
            for bucket, group in usable.groupby("bucket")
            if len(group) >= 1
        }

    decision = select_within_tie_band(
        losses, UNCERTAINTY_ORDER, per_bucket, PARSIMONY_TIE_BAND_SE
    )
    return {
        "temporal_model_used": selected,
        "coverage_levels": list(COVERAGE_LEVELS),
        "candidates": scored,
        "coverage_loss": losses,
        "selection": decision,
        "selected": decision["selected"] or "U0_raw",
        "note": (
            "The objective is mean absolute coverage error across the declared "
            "levels, not a mean squared z of one. A rejected earlier attempt "
            "optimised the second moment and produced an interval that was "
            "worse calibrated than the raw one."
        ),
    }


# ----------------------------------------------------------------------
# fold scaffolding for items 2, 3 and 4
# ----------------------------------------------------------------------


class Fold:
    """One forward chronological fold: fit before ``target``, score on it."""

    def __init__(
        self,
        frame: pd.DataFrame,
        count_frame: pd.DataFrame,
        target: int,
        bootstrap: int,
        bridge_bootstrap: int,
        seed: int,
    ) -> None:
        self.target = int(target)
        train_raw = assert_holdout_absent(
            frame.loc[frame["season"] < target], f"fold {target} training"
        )
        score_raw = assert_holdout_absent(
            frame.loc[frame["season"] == target], f"fold {target} scoring"
        )
        self.train, self.moments = standardize_residuals(train_raw, STATS)
        self.score, _ = standardize_residuals(score_raw, STATS, moments=self.moments)
        self.training_seasons = sorted(int(s) for s in train_raw["season"].unique())

        console.print(
            f"  fold {target}: train {self.training_seasons} "
            f"({len(self.train)} rows) -> score {target} ({len(self.score)} rows)"
        )

        self.train_moments = pair_moments(
            self.train, STATS, bootstrap=bootstrap, seed=seed + target
        )
        self.score_moments = pair_moments(
            self.score, STATS, bootstrap=bootstrap, seed=seed + 1000 + target
        )
        self.observed = read_buckets(
            STATS, self.score_moments.same_team, self.score_moments.cross_team
        )
        self.observed_se = read_buckets(
            STATS, self.score_moments.same_team_se, self.score_moments.cross_team_se
        )

        self.per_role_moments = {
            str(role): pair_moments(
                group, STATS, bootstrap=bootstrap, seed=seed + target, keep_draws=True
            )
            for role, group in self.train.dropna(subset=[ROLE_COLUMN]).groupby(
                ROLE_COLUMN
            )
        }
        self.score_role_cells = role_pair_cell_moments(self.score, STATS, ROLE_COLUMN)

        # Transmission inputs, built on the training seasons only.
        merged = self.train.merge(
            count_frame,
            on=["game_id", "team_id", "player_id", "season"],
            how="inner",
            suffixes=("", "_count"),
        )
        assert_holdout_absent(merged, f"fold {target} transmission")
        merged, self.coefficient_diagnostics = transmission_coefficient_columns(
            merged, STATS, order=DEFAULT_BRIDGE_ORDER
        )
        orders = range(1, DEFAULT_BRIDGE_ORDER + 1)
        usable = merged.dropna(
            subset=[f"{name}{k}_{stat}" for name in "hg" for k in orders for stat in STATS]
            + [f"zs_{stat}" for stat in STATS]
            + [f"e_{stat}" for stat in STATS]
        )
        if len(usable) != len(merged):
            raise SystemExit(
                f"fold {target}: {len(merged) - len(usable)} rows lack a "
                "transmission coefficient, which would make the control and "
                "the bridged arms read different rows"
            )
        self.latent_per_game = accumulate_per_game(
            usable, [f"zs_{stat}" for stat in STATS], STATS
        )
        self.count_per_game = accumulate_per_game(
            usable, [f"e_{stat}" for stat in STATS], STATS
        )
        self.count_gain_per_game = [
            accumulate_per_game(usable, [f"h{k}_{stat}" for stat in STATS], STATS)
            for k in orders
        ]
        self.latent_gain_per_game = [
            accumulate_per_game(usable, [f"g{k}_{stat}" for stat in STATS], STATS)
            for k in orders
        ]

        self.same_pair = bootstrap_source_pair(
            self.latent_per_game,
            self.count_per_game,
            self.latent_gain_per_game,
            self.count_gain_per_game,
            draws=bridge_bootstrap,
            seed=seed + target,
            same_team=True,
        )
        self.cross_pair = bootstrap_source_pair(
            self.latent_per_game,
            self.count_per_game,
            self.latent_gain_per_game,
            self.count_gain_per_game,
            draws=bridge_bootstrap,
            seed=seed + target,
            same_team=False,
        )
        # ``score_fit`` prices a fitted latent correlation in count space, so
        # the gains it uses are the count ones.
        self.same_gains = [moment.pooled()[0] for moment in self.count_gain_per_game]
        self.cross_gains = [moment.pooled()[1] for moment in self.count_gain_per_game]

        # Observed count-space buckets on the scoring season, for item 4's
        # objective. Built the same way, from the scoring season alone.
        score_merged = self.score.merge(
            count_frame,
            on=["game_id", "team_id", "player_id", "season"],
            how="inner",
            suffixes=("", "_count"),
        )
        assert_holdout_absent(score_merged, f"fold {target} scoring transmission")
        score_usable = score_merged.dropna(
            subset=[f"e_{stat}" for stat in STATS]
        )
        score_counts = accumulate_per_game(
            score_usable, [f"e_{stat}" for stat in STATS], STATS
        )
        same_counts, cross_counts = score_counts.pooled()
        self.observed_count = read_buckets(STATS, same_counts, cross_counts)

    def fit(self, spec: RemediationSpec, role_scale_override=None, combined=None):
        """Fit one candidate on the training seasons.

        ``combined`` replaces the *raw* pooled blocks, not the shrunk ones, so
        the transmission layer's output still passes through the same
        empirical-Bayes shrinkage the control applies. Overriding the shrunk
        targets instead would have compared a bridged-and-unshrunk arm against
        a plain-and-shrunk control and credited the difference to the bridge.
        """
        moments = self.train_moments
        if combined is not None:
            same, cross = combined
            moments = replace(
                moments,
                same_team=np.asarray(same, dtype=float),
                cross_team=np.asarray(cross, dtype=float),
            )
        return fit_remediated_factors(
            self.train,
            STATS,
            spec=spec,
            moments=moments,
            role_scale_override=role_scale_override,
        )

    def score_fit(self, fit) -> dict[str, float]:
        """Latent and count-space errors of one fit on the unseen season."""
        implied = read_buckets(
            STATS,
            fit.loadings.same_team_correlation(),
            fit.loadings.cross_team_correlation(),
        )
        latent_errors = {
            name: implied[name] - self.observed[name] for name in implied
        }
        count_implied_same = bridge_forward(
            fit.loadings.same_team_correlation(), self.same_gains
        )
        count_implied_cross = bridge_forward(
            fit.loadings.cross_team_correlation(), self.cross_gains
        )
        count_implied = read_buckets(STATS, count_implied_same, count_implied_cross)
        count_errors = {
            name: count_implied[name] - self.observed_count[name]
            for name in count_implied
        }
        return {
            "latent_rmse": float(
                np.sqrt(np.mean(np.square(list(latent_errors.values()))))
            ),
            "count_rmse": float(
                np.sqrt(np.mean(np.square(list(count_errors.values()))))
            ),
            "latent_errors": latent_errors,
            "count_errors": count_errors,
            "latent_z_errors": {
                name: latent_errors[name] / self.observed_se[name]
                if self.observed_se[name] > 0
                else np.nan
                for name in latent_errors
            },
            "implied_latent": implied,
            "implied_count": count_implied,
        }

    def role_cell_rmse(self, fit) -> dict[str, object]:
        """RMSE over the supported role-pair cells for one fit."""
        errors: dict[str, float] = {}
        supported: list[float] = []
        for first, second in ROLE_PAIR_CELLS:
            cell = self.score_role_cells.get((first, second))
            if cell is None:
                continue
            observed = np.asarray(cell["correlation"], dtype=float)
            implied = fit.loadings.same_team_correlation_for_roles(first, second)
            value = float(np.sqrt(np.mean(np.square(implied - observed))))
            errors[f"{first}+{second}"] = value
            if cell["supported"]:
                supported.append(value)
        return {
            "by_cell": errors,
            "supported_cell_rmse": float(np.sqrt(np.mean(np.square(supported))))
            if supported
            else float("nan"),
            "supported_cells": int(len(supported)),
        }


def decide_role_scale(folds: list[Fold], seed: int) -> dict[str, object]:
    """Item 2: ratio shrinkage or symmetric log-scale shrinkage."""
    results: dict[str, dict[str, object]] = {}
    per_fold_scores: dict[str, dict[str, float]] = {}

    for mode in (ROLE_SCALE_RATIO, ROLE_SCALE_LOG_SHRUNK):
        spec = RemediationSpec(name=f"role_{mode}", role_scale_mode=ROLE_SCALE_RATIO)
        fold_detail: list[dict[str, object]] = []
        scores: dict[str, float] = {}
        for fold in folds:
            base_fit = fold.fit(spec)
            base = base_fit.loadings
            if mode == ROLE_SCALE_RATIO:
                override = dict(base.role_scale)
                role_payload: dict[str, object] = {"mode": mode}
            else:
                raw, log_se, shares = role_scale_components(
                    fold.train,
                    STATS,
                    base=base,
                    role_column=ROLE_COLUMN,
                    per_role_moments=fold.per_role_moments,
                )
                shrunk = fit_log_shrunk_role_scales(raw, log_se, shares)
                override = dict(shrunk.scales)
                role_payload = shrunk.payload()
            fit = fold.fit(spec, role_scale_override=override)
            cells = fold.role_cell_rmse(fit)
            pooled_change = float(
                np.max(
                    np.abs(
                        fit.loadings.same_team_correlation()
                        - base.same_team_correlation()
                    )
                )
            )
            fold_detail.append(
                {
                    "target_season": fold.target,
                    "role_scale": {k: float(v) for k, v in override.items()},
                    "role_scale_fit": role_payload,
                    "pooled_block_max_change": pooled_change,
                    **cells,
                }
            )
            scores[str(fold.target)] = cells["supported_cell_rmse"]
        results[mode] = {"folds": fold_detail}
        per_fold_scores[mode] = scores

    pooled = {
        mode: float(np.mean(list(scores.values())))
        for mode, scores in per_fold_scores.items()
    }
    difference, se = paired_difference_se(
        per_fold_scores[ROLE_SCALE_LOG_SHRUNK], per_fold_scores[ROLE_SCALE_RATIO]
    )
    improves = difference < 0.0
    selected = ROLE_SCALE_LOG_SHRUNK if improves else ROLE_SCALE_RATIO
    return {
        "candidates": results,
        "supported_cell_rmse_by_fold": per_fold_scores,
        "pooled_supported_cell_rmse": pooled,
        "paired_difference_log_shrunk_minus_ratio": {
            "mean": difference,
            "standard_error": se,
        },
        "selected": selected,
        "reason": (
            "the log-scale form lowers supported-cell RMSE on the forward folds"
            if improves
            else "the accepted ratio form is retained: the log-scale form does "
            "not lower supported-cell RMSE on the forward folds"
        ),
    }


def decide_cross_team(folds: list[Fold]) -> dict[str, object]:
    """Item 3: the cross-team prior, judged on cross-team buckets only.

    The objective deliberately excludes same-team buckets. Letting a cross-team
    decision be scored on same-team evidence is how a cross-team repair ends up
    paid for by a same-team regression, which the brief forbids.
    """
    candidates: dict[str, RemediationSpec] = {
        "gaussian": RemediationSpec(
            name="cross_gaussian", cross_team_prior=CROSS_PRIOR_GAUSSIAN
        )
    }
    for nu in DEFAULT_NU_GRID:
        candidates[f"student_t_nu{nu:g}"] = RemediationSpec(
            name=f"cross_student_t_nu{nu:g}",
            cross_team_prior=CROSS_PRIOR_STUDENT_T,
            cross_team_nu=float(nu),
        )

    detail: dict[str, object] = {}
    per_fold: dict[str, dict[str, float]] = {}
    same_team_guard: dict[str, dict[str, float]] = {}
    for label, spec in candidates.items():
        folds_detail = []
        scores: dict[str, float] = {}
        guard: dict[str, float] = {}
        for fold in folds:
            fit = fold.fit(spec)
            scored = fold.score_fit(fit)
            cross_names = [
                name for name, kind, _ in CROSS_PLAYER_BUCKETS if kind == "cross_team"
            ]
            same_names = [
                name for name, kind, _ in CROSS_PLAYER_BUCKETS if kind == "same_team"
            ]
            cross_rmse = float(
                np.sqrt(
                    np.mean([scored["latent_errors"][n] ** 2 for n in cross_names])
                )
            )
            same_rmse = float(
                np.sqrt(
                    np.mean([scored["latent_errors"][n] ** 2 for n in same_names])
                )
            )
            folds_detail.append(
                {
                    "target_season": fold.target,
                    "cross_team_latent_rmse": cross_rmse,
                    "same_team_latent_rmse": same_rmse,
                    "protected_opponent_z": {
                        name: scored["latent_z_errors"][name]
                        for name in PROTECTED_OPPONENT_BUCKETS
                    },
                    "shrink_diagnostics": {
                        key: float(value)
                        for key, value in fit.competition_evidence.items()
                        if "cross_team" in key or key == "nu"
                    },
                }
            )
            scores[str(fold.target)] = cross_rmse
            guard[str(fold.target)] = same_rmse
        detail[label] = {"folds": folds_detail}
        per_fold[label] = scores
        same_team_guard[label] = guard

    pooled = {
        label: float(np.mean(list(scores.values()))) for label, scores in per_fold.items()
    }
    order = ["gaussian"] + [
        f"student_t_nu{nu:g}" for nu in sorted(DEFAULT_NU_GRID, reverse=True)
    ]
    decision = select_within_tie_band(pooled, order, per_fold, PARSIMONY_TIE_BAND_SE)
    selected = decision["selected"] or "gaussian"
    return {
        "candidates": detail,
        "cross_team_latent_rmse_by_fold": per_fold,
        "same_team_latent_rmse_by_fold": same_team_guard,
        "pooled_cross_team_latent_rmse": pooled,
        "simplicity_order": order,
        "selection": decision,
        "selected": selected,
        "selected_spec": (
            {"cross_team_prior": CROSS_PRIOR_GAUSSIAN}
            if selected == "gaussian"
            else {
                "cross_team_prior": CROSS_PRIOR_STUDENT_T,
                "cross_team_nu": float(selected.split("nu")[1]),
            }
        ),
    }


def decide_transmission(folds: list[Fold], cross_choice: dict[str, object]) -> dict[str, object]:
    """Item 4: how much weight the bridge-implied latent target may carry.

    The two sources are first tested for homogeneity. If they disagree by more
    than their joint uncertainty then they are not two noisy readings of one
    number, and the weight cap is what stops a refuted source from taking over.

    The selection rule is pre-registered here and is the brief's own acceptance
    test, not a free objective: among the caps whose forward global latent and
    count RMSE stay inside the 3% band, take the one that most improves the
    forward count-space error of ``passer_ast_teammate_pts``; if no non-zero
    cap qualifies, the layer stays off and the control is retained.

    The band is applied to the *forward* RMSEs, which is a stricter test than
    the confirmatory protocol those gates are finally read on, because the
    confirmatory protocol refits on the season it scores. Choosing the cap
    against an in-season objective instead would be circular: the bridge target
    is built from the same season's count moments that the count metric is
    read against, so a larger cap would win by construction.
    """
    base_kwargs = cross_choice["selected_spec"]
    homogeneity: list[dict[str, object]] = []
    for fold in folds:
        for block, pair in (("same_team", fold.same_pair), ("cross_team", fold.cross_pair)):
            test = homogeneity_test(pair)
            homogeneity.append(
                {
                    "target_season": fold.target,
                    "block": block,
                    "pooled_chi_square": test["pooled_chi_square"],
                    "degrees_of_freedom": test["degrees_of_freedom"],
                    "p_value": test["p_value"],
                    "max_disagreement_in_sigma": test["max_disagreement_in_sigma"],
                    "homogeneous": test["homogeneous"],
                }
            )

    per_cap: dict[str, dict[str, object]] = {}
    latent_by_fold: dict[str, dict[str, float]] = {}
    count_by_fold: dict[str, dict[str, float]] = {}
    target_by_fold: dict[str, dict[str, float]] = {}
    protected_by_fold: dict[str, dict[str, dict[str, float]]] = {}

    for cap in BRIDGE_CAP_GRID:
        label = f"cap_{cap:.2f}"
        spec = RemediationSpec(
            name=f"bridge_{label}",
            bridge_mode="off" if cap == 0.0 else "inverse_variance",
            bridge_weight_cap=float(cap),
            **base_kwargs,
        )
        folds_detail = []
        latent_scores: dict[str, float] = {}
        count_scores: dict[str, float] = {}
        target_scores: dict[str, float] = {}
        protected_scores: dict[str, dict[str, float]] = {}
        for fold in folds:
            if cap == 0.0:
                combined = None
                weights: dict[str, object] = {
                    "mean_bridge_weight_same_team": 0.0,
                    "mean_bridge_weight_cross_team": 0.0,
                }
            else:
                same, same_info = combine_sources(fold.same_pair, bridge_weight_cap=cap)
                cross, cross_info = combine_sources(
                    fold.cross_pair, bridge_weight_cap=cap
                )
                combined = (same, cross)
                weights = {
                    "mean_bridge_weight_same_team": same_info["mean_bridge_weight"],
                    "mean_bridge_weight_cross_team": cross_info["mean_bridge_weight"],
                    "max_bridge_weight_same_team": same_info["max_bridge_weight"],
                    "max_bridge_weight_cross_team": cross_info["max_bridge_weight"],
                    "entries_clipped_same_team": same_info["entries_clipped_by_cap"],
                    "entries_clipped_cross_team": cross_info["entries_clipped_by_cap"],
                    "bridge_model_error_sd_same_team": float(
                        np.sqrt(same_info["bridge_model_error_variance"])
                    ),
                    "bridge_model_error_sd_cross_team": float(
                        np.sqrt(cross_info["bridge_model_error_variance"])
                    ),
                }
            fit = fold.fit(spec, combined=combined)
            scored = fold.score_fit(fit)
            protected = {
                name: scored["latent_z_errors"][name]
                for name in PROTECTED_OPPONENT_BUCKETS + PROTECTED_TEAMMATE_BUCKETS
            }
            folds_detail.append(
                {
                    "target_season": fold.target,
                    "latent_rmse": scored["latent_rmse"],
                    "count_rmse": scored["count_rmse"],
                    "passer_ast_teammate_pts_count_error": scored["count_errors"][
                        "passer_ast_teammate_pts"
                    ],
                    "passer_ast_teammate_pts_latent_z": scored["latent_z_errors"][
                        "passer_ast_teammate_pts"
                    ],
                    "passer_ast_teammate_pts_implied_latent": scored[
                        "implied_latent"
                    ]["passer_ast_teammate_pts"],
                    "passer_ast_teammate_pts_implied_count": scored[
                        "implied_count"
                    ]["passer_ast_teammate_pts"],
                    "passer_ast_teammate_pts_observed_latent": fold.observed[
                        "passer_ast_teammate_pts"
                    ],
                    "passer_ast_teammate_pts_observed_count": fold.observed_count[
                        "passer_ast_teammate_pts"
                    ],
                    "protected_z": protected,
                    **weights,
                }
            )
            latent_scores[str(fold.target)] = scored["latent_rmse"]
            count_scores[str(fold.target)] = scored["count_rmse"]
            target_scores[str(fold.target)] = abs(
                scored["count_errors"]["passer_ast_teammate_pts"]
            )
            protected_scores[str(fold.target)] = protected
        per_cap[label] = {"folds": folds_detail}
        latent_by_fold[label] = latent_scores
        count_by_fold[label] = count_scores
        target_by_fold[label] = target_scores
        protected_by_fold[label] = protected_scores

    control = "cap_0.00"
    order = [f"cap_{cap:.2f}" for cap in BRIDGE_CAP_GRID]

    def pooled(by_fold: dict[str, dict[str, float]]) -> dict[str, float]:
        return {
            label: float(np.mean(list(scores.values())))
            for label, scores in by_fold.items()
        }

    pooled_latent = pooled(latent_by_fold)
    pooled_count = pooled(count_by_fold)
    pooled_target = pooled(target_by_fold)

    admissibility: dict[str, dict[str, object]] = {}
    for label in order:
        latent_ratio = pooled_latent[label] / pooled_latent[control]
        count_ratio = pooled_count[label] / pooled_count[control]
        target_gain = 1.0 - pooled_target[label] / pooled_target[control]
        worst_protected = max(
            abs(protected_by_fold[label][fold][name])
            - abs(protected_by_fold[control][fold][name])
            for fold in protected_by_fold[label]
            for name in protected_by_fold[label][fold]
        )
        admissible = (
            latent_ratio <= 1.03
            and count_ratio <= 1.03
            and worst_protected <= 0.5
        )
        admissibility[label] = {
            "forward_latent_rmse_ratio": latent_ratio,
            "forward_count_rmse_ratio": count_ratio,
            "forward_target_bucket_improvement": target_gain,
            "worst_protected_z_increase": worst_protected,
            "admissible": bool(admissible),
        }

    eligible = [
        label
        for label in order
        if label != control and admissibility[label]["admissible"]
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda label: admissibility[label]["forward_target_bucket_improvement"],
        )
        if admissibility[selected]["forward_target_bucket_improvement"] <= 0.0:
            selected = control
    else:
        selected = control
    cap = float(selected.split("_")[1])

    return {
        "homogeneity": homogeneity,
        "homogeneity_verdict": (
            "REJECTED"
            if any(not entry["homogeneous"] for entry in homogeneity)
            else "NOT_REJECTED"
        ),
        "cap_grid": list(BRIDGE_CAP_GRID),
        "candidates": per_cap,
        "latent_rmse_by_fold": latent_by_fold,
        "count_rmse_by_fold": count_by_fold,
        "target_bucket_abs_count_error_by_fold": target_by_fold,
        "pooled_forward_latent_rmse": pooled_latent,
        "pooled_forward_count_rmse": pooled_count,
        "pooled_forward_target_bucket_abs_count_error": pooled_target,
        "admissibility": admissibility,
        "selection_rule": (
            "among caps whose pooled forward global latent RMSE and forward "
            "global count RMSE stay within 1.03x the cap-0 control and whose "
            "protected buckets do not worsen by more than 0.5 z, take the cap "
            "with the largest forward improvement in the absolute count-space "
            "error of passer_ast_teammate_pts; otherwise keep the control"
        ),
        "selected": selected,
        "selected_cap": cap,
    }


def main() -> None:
    args = parse_args()
    console.rule("Final upstream remediation: inner selection (pre-2024 only)")

    frame = pd.read_parquet(args.residuals)
    frame = assert_holdout_absent(
        frame.loc[~frame["season"].isin(HOLDOUT_SEASONS)].copy(), "residual frame"
    )
    count_frame = pd.read_parquet(args.count_residuals)
    count_frame = assert_holdout_absent(
        count_frame.loc[~count_frame["season"].isin(HOLDOUT_SEASONS)].copy(),
        "count residual frame",
    )
    seasons = sorted(int(value) for value in frame["season"].unique())
    console.print(f"seasons available for selection: {seasons}")

    # Per-season standardization constants from the seasons before each one.
    moments_by_season: dict[int, dict[str, tuple[float, float]]] = {}
    for season in seasons:
        earlier = frame.loc[frame["season"] < season]
        if earlier.empty:
            continue
        _, moments = standardize_residuals(earlier, STATS)
        moments_by_season[season] = moments

    console.rule("1. temporal treatment")
    series = season_series(frame, moments_by_season, args.bootstrap, args.seed)
    temporal = decide_temporal(series)
    table = Table(title="forward predictive log score, pooled over buckets")
    table.add_column("candidate")
    table.add_column("log score", justify="right")
    table.add_column("SE", justify="right")
    table.add_column("vs A0", justify="right")
    for label in TEMPORAL_ORDER:
        paired = temporal["paired_difference_against_control"][label]  # type: ignore[index]
        table.add_row(
            label,
            f"{temporal['pooled_forward_log_score'][label]:.5f}",  # type: ignore[index]
            f"{temporal['pooled_forward_log_score_se'][label]:.5f}",  # type: ignore[index]
            f"{paired['mean']:+.5f} +/- {paired['standard_error']:.5f}",
        )
    console.print(table)
    console.print(f"[bold]temporal selected:[/bold] {temporal['selected']}")

    console.rule("6. predictive uncertainty")
    uncertainty = decide_uncertainty(temporal)
    table = Table(title="forward cross-fitted coverage")
    table.add_column("candidate")
    for level in COVERAGE_LEVELS:
        table.add_column(f"{int(level*100)}%", justify="right")
    table.add_column("mean z^2", justify="right")
    table.add_column("loss", justify="right")
    for label in UNCERTAINTY_ORDER:
        entry = uncertainty["candidates"][label]  # type: ignore[index]
        coverage = entry["coverage"]
        table.add_row(
            label,
            *[
                f"{coverage[f'coverage_{int(level*100)}']:.3f}"
                for level in COVERAGE_LEVELS
            ],
            f"{coverage['mean_squared_z']:.4f}",
            f"{entry['coverage_loss']:.4f}",
        )
    console.print(table)
    console.print(f"[bold]uncertainty selected:[/bold] {uncertainty['selected']}")

    console.rule("forward folds for items 2, 3 and 4")
    # One training season is enough to fit a factor model, so these items get
    # every fold the record supports. Item 1 needs at least two seasons before
    # a between-season heterogeneity even exists, which is why its fold set
    # (built inside ``forward_predictive_records``) starts one season later.
    targets = forward_fold_seasons(seasons, MIN_FOLD_TRAINING_SEASONS)
    folds = [
        Fold(
            frame,
            count_frame,
            target,
            bootstrap=args.bootstrap,
            bridge_bootstrap=args.bridge_bootstrap,
            seed=args.seed,
        )
        for target in targets
    ]

    console.rule("2. role scale")
    role = decide_role_scale(folds, args.seed)
    console.print(
        "supported-cell RMSE: "
        + ", ".join(
            f"{mode} {value:.6f}"
            for mode, value in role["pooled_supported_cell_rmse"].items()  # type: ignore[index]
        )
    )
    console.print(f"[bold]role scale selected:[/bold] {role['selected']}")

    console.rule("3. cross-team prior")
    cross = decide_cross_team(folds)
    table = Table(title="cross-team latent RMSE on the unseen season")
    table.add_column("prior")
    table.add_column("pooled RMSE", justify="right")
    for label in cross["simplicity_order"]:  # type: ignore[index]
        table.add_row(label, f"{cross['pooled_cross_team_latent_rmse'][label]:.6f}")  # type: ignore[index]
    console.print(table)
    console.print(f"[bold]cross-team prior selected:[/bold] {cross['selected']}")

    console.rule("4. latent-to-count transmission")
    transmission = decide_transmission(folds, cross)
    table = Table(title="transmission weight cap, pooled over forward folds")
    table.add_column("cap")
    table.add_column("latent RMSE x", justify="right")
    table.add_column("count RMSE x", justify="right")
    table.add_column("target gain", justify="right")
    table.add_column("worst prot. dz", justify="right")
    table.add_column("admissible", justify="right")
    for label in [f"cap_{cap:.2f}" for cap in BRIDGE_CAP_GRID]:
        entry = transmission["admissibility"][label]  # type: ignore[index]
        table.add_row(
            label,
            f"{entry['forward_latent_rmse_ratio']:.4f}",
            f"{entry['forward_count_rmse_ratio']:.4f}",
            f"{entry['forward_target_bucket_improvement']:+.2%}",
            f"{entry['worst_protected_z_increase']:+.3f}",
            "yes" if entry["admissible"] else "no",
        )
    console.print(table)
    console.print(
        f"homogeneity of the two sources: {transmission['homogeneity_verdict']}"
    )
    console.print(f"[bold]transmission cap selected:[/bold] {transmission['selected_cap']}")

    payload = {
        "title": "Final upstream remediation: inner selection",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "scope": "strictly pre-2024 chronological folds",
        "holdout_seasons_excluded": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "seasons_used": seasons,
        "forward_fold_target_seasons": [int(value) for value in targets],
        "stats": list(STATS),
        "seed": int(args.seed),
        "bootstrap": int(args.bootstrap),
        "bridge_bootstrap": int(args.bridge_bootstrap),
        "parsimony_tie_band_se": PARSIMONY_TIE_BAND_SE,
        "item_1_temporal": temporal,
        "item_2_role_scale": role,
        "item_3_cross_team": cross,
        "item_4_transmission": transmission,
        "item_6_uncertainty": uncertainty,
        "control_spec": CONTROL_SPEC.payload(),
        "code_sha": git_sha(PROJECT_ROOT),
    }
    path = write_json(payload, Path(args.artifact_root) / "inner_selection.json")

    console.rule("Inner selection")
    console.print(f"TEMPORAL_TREATMENT={temporal['selected']}")
    console.print(f"ROLE_SCALE_MODE={role['selected']}")
    console.print(f"CROSS_TEAM_PRIOR={cross['selected']}")
    console.print(f"TRANSMISSION_CAP={transmission['selected_cap']}")
    console.print(f"UNCERTAINTY_MODEL={uncertainty['selected']}")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")
    console.print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
