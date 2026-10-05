"""Empirical estimation of the shared latent-factor loadings.

SHADOW / RESEARCH ONLY.

The loadings are estimated from *cross-player* pairs only. Same-player pairs
are never touched, which is the estimation-side half of the no-double-count
contract: the incumbent same-player block is not re-fitted, and the new
shared factors are not allowed to re-explain it.

Per game and per team, with ``z_i`` the Gaussianized residual vector of
player ``i`` and ``n`` the number of usable players on that team::

    sum_{i != j in team}  z_i z_j^T  =  (sum_i z_i)(sum_i z_i)^T - sum_i z_i z_i^T

so both the same-team and the cross-team pair sums are available in O(n)
per team instead of O(n^2) pairs. Pooling those sums over games gives

    S_hat  = same-team   cross-player stat-by-stat correlation
    X_hat  = cross-team  cross-player stat-by-stat correlation

from which ``A = (S + X) / 2`` and ``B = (S - X) / 2`` are projected onto the
PSD cone at ranks ``k_game`` and 1 respectively (see ``covariance.py`` for
why those are the identified objects).

Shrinkage: each entry of ``S_hat`` and ``X_hat`` is soft-thresholded at
``z_crit`` game-clustered standard errors before projection, so stat pairs
with no reliable signal contribute nothing instead of contributing noise.
Because every parameter is indexed by stat (and optionally by a coarse role
bucket with partial pooling), there are no pair-specific or player-specific
dependence parameters to overfit, and the fitted layer transfers to unseen
players by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .covariance import SharedFactorLoadings, project_psd_rank

DEFAULT_K_GAME = 2

# Soft-threshold width in game-clustered standard errors. 1.96 is the usual
# two-sided 5% normal critical value; it is a declared constant of the fit,
# not a tuned one.
DEFAULT_SHRINK_Z = 1.96

# Partial-pooling strength for role multipliers, in units of games. A role
# bucket observed in `ROLE_POOLING_GAMES` games gets half its own estimate
# and half the pooled value of 1.0.
ROLE_POOLING_GAMES = 400.0


@dataclass(frozen=True)
class PairMoments:
    """Pooled cross-player second moments, in correlation units."""

    stats: tuple[str, ...]
    same_team: np.ndarray
    cross_team: np.ndarray
    same_team_pairs: float
    cross_team_pairs: float
    games: int
    same_team_se: np.ndarray
    cross_team_se: np.ndarray


def standardize_residuals(
    frame: pd.DataFrame,
    stats: Sequence[str],
    moments: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Center and scale each stat's residual before dependence estimation.

    The latent residual is exactly standard normal only when the marginal is
    exactly right. Estimating *dependence* from a residual whose empirical
    scale is 0.97 rather than 1.00 would mix a marginal-calibration error into
    the correlation estimate, so the dependence fit works in standardized
    units. The simulator keeps unit variance, and the observed mean and scale
    are reported as marginal-calibration diagnostics instead of being
    discarded. ``moments`` can be supplied so a validation season is
    standardized with training-season constants (no lookahead).
    """
    out = frame.copy()
    fitted: dict[str, tuple[float, float]] = {}
    for stat in stats:
        column = f"z_{stat}"
        values = out[column].to_numpy(dtype=float)
        if moments is not None and stat in moments:
            mean, scale = moments[stat]
        else:
            mean = float(np.nanmean(values))
            scale = float(np.nanstd(values))
        scale = scale if scale > 1e-9 else 1.0
        fitted[stat] = (mean, scale)
        out[f"zs_{stat}"] = (values - mean) / scale
    return out, fitted


def pair_moments(
    frame: pd.DataFrame,
    stats: Sequence[str],
    value_prefix: str = "zs_",
    bootstrap: int = 0,
    seed: int = 73,
) -> PairMoments:
    """Pool cross-player second moments over games.

    ``frame`` needs ``game_id``, ``team_id`` and one ``{value_prefix}{stat}``
    column per stat, with one row per usable player-game. Standard errors are
    game-clustered: games are the independent unit, so the bootstrap resamples
    whole games.
    """
    stats = tuple(stats)
    n_stats = len(stats)
    columns = [f"{value_prefix}{stat}" for stat in stats]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"residual frame is missing {missing}")

    usable = frame.dropna(subset=["game_id", "team_id", *columns])
    if usable.empty:
        raise ValueError("no usable rows for pair-moment estimation")

    per_game: list[tuple[np.ndarray, float, np.ndarray, float]] = []

    for _, game in usable.groupby("game_id", sort=True):
        team_sums: list[np.ndarray] = []
        team_counts: list[int] = []
        same = np.zeros((n_stats, n_stats), dtype=float)
        same_pairs = 0.0

        for _, team in game.groupby("team_id", sort=True):
            z = team[columns].to_numpy(dtype=float)
            count = z.shape[0]
            total = z.sum(axis=0)
            team_sums.append(total)
            team_counts.append(count)
            if count >= 2:
                same += np.outer(total, total) - z.T @ z
                same_pairs += count * (count - 1)

        cross = np.zeros((n_stats, n_stats), dtype=float)
        cross_pairs = 0.0
        for first in range(len(team_sums)):
            for second in range(first + 1, len(team_sums)):
                outer = np.outer(team_sums[first], team_sums[second])
                cross += outer + outer.T
                cross_pairs += 2.0 * team_counts[first] * team_counts[second]

        per_game.append((same, same_pairs, cross, cross_pairs))

    same_total = np.sum([entry[0] for entry in per_game], axis=0)
    same_pairs_total = float(np.sum([entry[1] for entry in per_game]))
    cross_total = np.sum([entry[2] for entry in per_game], axis=0)
    cross_pairs_total = float(np.sum([entry[3] for entry in per_game]))

    if same_pairs_total <= 0 or cross_pairs_total <= 0:
        raise ValueError("not enough same-team or cross-team pairs")

    same_hat = same_total / same_pairs_total
    cross_hat = cross_total / cross_pairs_total

    same_se = np.full((n_stats, n_stats), np.nan)
    cross_se = np.full((n_stats, n_stats), np.nan)

    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        n_games = len(per_game)
        same_draws = np.empty((bootstrap, n_stats, n_stats))
        cross_draws = np.empty((bootstrap, n_stats, n_stats))
        same_stack = np.stack([entry[0] for entry in per_game])
        same_pair_stack = np.array([entry[1] for entry in per_game])
        cross_stack = np.stack([entry[2] for entry in per_game])
        cross_pair_stack = np.array([entry[3] for entry in per_game])
        for draw in range(bootstrap):
            index = rng.integers(0, n_games, size=n_games)
            same_pairs_draw = same_pair_stack[index].sum()
            cross_pairs_draw = cross_pair_stack[index].sum()
            same_draws[draw] = same_stack[index].sum(axis=0) / max(same_pairs_draw, 1.0)
            cross_draws[draw] = cross_stack[index].sum(axis=0) / max(
                cross_pairs_draw, 1.0
            )
        same_se = same_draws.std(axis=0, ddof=1)
        cross_se = cross_draws.std(axis=0, ddof=1)

    return PairMoments(
        stats=stats,
        same_team=0.5 * (same_hat + same_hat.T),
        cross_team=0.5 * (cross_hat + cross_hat.T),
        same_team_pairs=same_pairs_total,
        cross_team_pairs=cross_pairs_total,
        games=len(per_game),
        same_team_se=0.5 * (same_se + same_se.T),
        cross_team_se=0.5 * (cross_se + cross_se.T),
    )


def soft_threshold(
    estimate: np.ndarray,
    standard_error: np.ndarray,
    z_crit: float = DEFAULT_SHRINK_Z,
) -> np.ndarray:
    """Shrink each entry toward zero by ``z_crit`` standard errors.

    ``sign(r) * max(0, |r| - z * se)``. Entries whose magnitude the data
    cannot distinguish from zero contribute exactly zero to the fitted
    structure, which is what keeps the sparse stat pairs (STL, BLK) from
    importing noise into the game-level factors.
    """
    estimate = np.asarray(estimate, dtype=float)
    error = np.asarray(standard_error, dtype=float)
    if not np.all(np.isfinite(error)):
        return estimate
    magnitude = np.clip(np.abs(estimate) - z_crit * error, 0.0, None)
    return np.sign(estimate) * magnitude


@dataclass(frozen=True)
class FactorFit:
    loadings: SharedFactorLoadings
    moments: PairMoments
    same_team_shrunk: np.ndarray
    cross_team_shrunk: np.ndarray
    game_gram_eigenvalues: np.ndarray
    contrast_gram_eigenvalues: np.ndarray
    k_game: int
    shrink_z: float

    def diagnostics(self) -> dict[str, object]:
        fitted_same = self.loadings.same_team_correlation()
        fitted_cross = self.loadings.cross_team_correlation()
        return {
            "stats": list(self.moments.stats),
            "games": int(self.moments.games),
            "same_team_pairs": float(self.moments.same_team_pairs),
            "cross_team_pairs": float(self.moments.cross_team_pairs),
            "k_game": int(self.k_game),
            "shrink_z": float(self.shrink_z),
            "observed_same_team_correlation": self.moments.same_team.tolist(),
            "observed_cross_team_correlation": self.moments.cross_team.tolist(),
            "observed_same_team_se": self.moments.same_team_se.tolist(),
            "observed_cross_team_se": self.moments.cross_team_se.tolist(),
            "shrunk_same_team_correlation": self.same_team_shrunk.tolist(),
            "shrunk_cross_team_correlation": self.cross_team_shrunk.tolist(),
            "fitted_same_team_correlation": fitted_same.tolist(),
            "fitted_cross_team_correlation": fitted_cross.tolist(),
            "game_gram_eigenvalues": self.game_gram_eigenvalues.tolist(),
            "contrast_gram_eigenvalues": self.contrast_gram_eigenvalues.tolist(),
            "same_team_fit_rmse": float(
                np.sqrt(np.mean((fitted_same - self.same_team_shrunk) ** 2))
            ),
            "cross_team_fit_rmse": float(
                np.sqrt(np.mean((fitted_cross - self.cross_team_shrunk) ** 2))
            ),
        }


def fit_shared_factors(
    frame: pd.DataFrame,
    stats: Sequence[str],
    k_game: int = DEFAULT_K_GAME,
    shrink_z: float = DEFAULT_SHRINK_Z,
    bootstrap: int = 200,
    seed: int = 73,
    value_prefix: str = "zs_",
    role_column: str | None = None,
) -> FactorFit:
    """Estimate the shared latent-factor loadings from OOF residuals."""
    stats = tuple(stats)
    moments = pair_moments(
        frame,
        stats,
        value_prefix=value_prefix,
        bootstrap=bootstrap,
        seed=seed,
    )

    same = soft_threshold(moments.same_team, moments.same_team_se, shrink_z)
    cross = soft_threshold(moments.cross_team, moments.cross_team_se, shrink_z)

    game_gram = 0.5 * (same + cross)
    contrast_gram = 0.5 * (same - cross)

    _, game_loadings = project_psd_rank(game_gram, rank=k_game)
    _, contrast_loadings = project_psd_rank(contrast_gram, rank=1)

    team_contrast = (
        contrast_loadings[:, 0]
        if contrast_loadings.shape[1] == 1
        else np.zeros(len(stats), dtype=float)
    )

    role_scale: dict[str, float] = {}
    if role_column is not None:
        role_scale = fit_role_scales(
            frame,
            stats,
            base=SharedFactorLoadings(
                stats=stats, game=game_loadings, team_contrast=team_contrast
            ),
            role_column=role_column,
            value_prefix=value_prefix,
        )

    loadings = SharedFactorLoadings(
        stats=stats,
        game=game_loadings,
        team_contrast=team_contrast,
        role_scale=role_scale,
    )

    return FactorFit(
        loadings=loadings,
        moments=moments,
        same_team_shrunk=same,
        cross_team_shrunk=cross,
        game_gram_eigenvalues=np.linalg.eigvalsh(0.5 * (game_gram + game_gram.T))[::-1],
        contrast_gram_eigenvalues=np.linalg.eigvalsh(
            0.5 * (contrast_gram + contrast_gram.T)
        )[::-1],
        k_game=int(k_game),
        shrink_z=float(shrink_z),
    )


def fit_role_scales(
    frame: pd.DataFrame,
    stats: Sequence[str],
    base: SharedFactorLoadings,
    role_column: str,
    value_prefix: str = "zs_",
) -> dict[str, float]:
    """Partially pooled multiplicative role scalars for the shared loadings.

    For each role bucket the ratio of the observed to the model-implied
    same-team cross-player covariance gives a squared scale; the square root
    is shrunk toward the pooled value of 1.0 by the number of games the bucket
    was seen in. A role the fit never saw is absent from the mapping and
    therefore falls back to 1.0.
    """
    stats = tuple(stats)
    implied = base.same_team_correlation()
    denominator = float(np.sum(implied**2))
    if denominator <= 0:
        return {}

    scales: dict[str, float] = {}
    for role, group in frame.dropna(subset=[role_column]).groupby(role_column):
        try:
            observed = pair_moments(group, stats, value_prefix=value_prefix)
        except ValueError:
            continue
        numerator = float(np.sum(observed.same_team * implied))
        if numerator <= 0:
            continue
        raw = float(np.sqrt(max(numerator / denominator, 0.0)))
        weight = observed.games / (observed.games + ROLE_POOLING_GAMES)
        scales[str(role)] = float(weight * raw + (1.0 - weight) * 1.0)
    return scales


def incumbent_within_player_blocks(
    copula: object,
    stats: Sequence[str],
    player_ids: Iterable[int],
) -> dict[int, np.ndarray]:
    """Read within-player blocks out of the incumbent ``GaussianCopula``.

    The incumbent orders its correlation matrix by ``copula.targets``; the
    shadow layer re-indexes into its own stat order without touching the
    incumbent object. Players the incumbent never fitted individually resolve
    to its global fallback, which is exactly the production behaviour and the
    sparse/unseen-player fallback this layer relies on.
    """
    targets = list(copula.targets)
    position = {stat: targets.index(stat) for stat in stats}
    order = [position[stat] for stat in stats]

    blocks: dict[int, np.ndarray] = {}
    for player_id in player_ids:
        correlation = np.asarray(
            copula.correlation_for_player(int(player_id)), dtype=float  # type: ignore[attr-defined]
        )
        blocks[int(player_id)] = correlation[np.ix_(order, order)]
    return blocks
