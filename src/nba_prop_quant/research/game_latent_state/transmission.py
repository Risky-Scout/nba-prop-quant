"""Latent-to-count transmission for the latent-state shadow layer.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The dependence model is parameterised in latent Gaussian space but a bettor is
exposed in count space, and the two are not the same number. This module says
how one maps to the other, generically, for every supported stat-pair family
rather than as an override on the one bucket that misses worst.

The bridge
----------
Under the model a count is ``Y = F^{-1}(Phi(Z))`` with ``Z`` standard normal,
so ``Y`` is a deterministic function of ``Z`` and expands in the probabilists'
Hermite basis. Writing ``h_k = E[(Y - E Y) He_k(Z)] / sd(Y)`` for the
standardized count residual, two players' standardized count residuals satisfy

    Corr(e_a, e_b) = sum_{k >= 1} rho_ab^k h_{a,k} h_{b,k} / k!

exactly, where ``rho_ab`` is their latent correlation. The ``k = 1`` term is
``rho_ab h_{a,1} h_{b,1}``, and ``h_1 <= 1`` with equality only for a
continuous margin, so the leading effect of discreteness is attenuation.

Estimating the coefficients
---------------------------
``h_{i,k}`` depends only on observation ``i``'s own marginal, and the residual
build already carries both halves of it: ``z`` is the *randomized* PIT, so
under a correct marginal ``Y`` is recovered from ``Z`` exactly, which makes the
single-observation product ``e_i He_k(z_i)`` an unbiased estimate of
``h_{i,k}``. It is far too noisy to use one observation at a time, so the
estimates are smoothed within a stat across bins of the analytic mean -- the
quantity ``h`` actually varies with, since a margin with mean 0.3 is coarse and
one with mean 20 is nearly continuous.

That estimator is empirical rather than analytic on purpose. The validation
metric standardizes by the *production* analytic mean and standard deviation,
so a marginal whose analytic variance is slightly wrong feeds that error into
the metric; measuring ``h`` the same way makes the bridge predict the quantity
being scored instead of the quantity a correctly specified model would produce.

Pair weighting
--------------
A bucket is an average over ordered player pairs, so the bridge needs
``G_k = E_pairs[h_{a,k} h_{b,k}]`` with the same pair weights the observed
bucket uses, not a product of marginal averages. The same identity that makes
the moment accumulation linear in team sums makes this one linear too:

    sum_{i != j in team} h_i h_j = (sum_i h_i)(sum_j h_j) - sum_i h_i h_i

so the per-game accumulation is O(n) in roster size and the game stays the
independent unit for every standard error taken from it.

Combining two sources of evidence
---------------------------------
A bucket's latent target can be estimated directly from ``z``, or indirectly by
inverting the bridge on the observed count moment. Both estimate the same
``rho`` under the model, so they can be pooled -- but only if they agree to
within their joint uncertainty. :func:`homogeneity_test` checks that before
:func:`combine_sources` pools them, and the pooling uses the full ``2 x 2``
covariance from a shared game-clustered bootstrap, because the two estimates
are computed from the same games and are strongly correlated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.polynomial import hermite_e
from scipy import stats

#: Hermite order the bridge is truncated at. At the magnitudes these buckets
#: occupy (``|rho| < 0.06``) the ``k = 3`` term is below 1e-7 in correlation
#: units, four orders under the standard error of any bucket, while the pooled
#: estimate of ``h_3`` is itself noisy -- so carrying it would add variance and
#: no signal.
DEFAULT_BRIDGE_ORDER = 2

#: Quantile bins of the analytic mean used to smooth the Hermite coefficients
#: within a stat.
DEFAULT_COEFFICIENT_BINS = 24

#: A bucket whose latent estimate is this close to zero has no scale to invert
#: the bridge against, so the bridge is reported as unavailable there rather
#: than dividing by it.
MIN_INVERTIBLE_GAIN = 1e-3


# ----------------------------------------------------------------------
# per-game accumulation, so every bootstrap resamples games not rows
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PerGameMoments:
    """Per-game cross-player moment sums, ready to pool or resample.

    ``same``/``cross`` have shape ``(n_games, n_stats, n_stats)`` and are sums
    over ordered distinct pairs, not averages, so a bootstrap can add games and
    divide by the resampled pair counts rather than averaging averages.
    """

    stats: tuple[str, ...]
    game_ids: np.ndarray
    same: np.ndarray
    same_pairs: np.ndarray
    cross: np.ndarray
    cross_pairs: np.ndarray

    @property
    def n_games(self) -> int:
        return len(self.game_ids)

    def pooled(self, index: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Pooled same-team and cross-team correlation matrices."""
        if index is None:
            same = self.same.sum(axis=0)
            cross = self.cross.sum(axis=0)
            same_pairs = float(self.same_pairs.sum())
            cross_pairs = float(self.cross_pairs.sum())
        else:
            same = self.same[index].sum(axis=0)
            cross = self.cross[index].sum(axis=0)
            same_pairs = float(self.same_pairs[index].sum())
            cross_pairs = float(self.cross_pairs[index].sum())
        same = same / max(same_pairs, 1.0)
        cross = cross / max(cross_pairs, 1.0)
        return 0.5 * (same + same.T), 0.5 * (cross + cross.T)


def accumulate_per_game(
    frame: pd.DataFrame,
    columns: Sequence[str],
    stats: Sequence[str],
) -> PerGameMoments:
    """Accumulate per-game pair sums for one set of value columns.

    Mirrors :func:`factors.pair_moments` exactly -- the same team-sum identity,
    the same ordered-pair counting -- but keeps the per-game terms instead of
    pooling them immediately, which is what lets several value sets share one
    bootstrap index and therefore have a measurable covariance.
    """
    columns = list(columns)
    stats = tuple(stats)
    usable = frame.dropna(subset=["game_id", "team_id", *columns])
    if usable.empty:
        raise ValueError("no usable rows for pair accumulation")

    n = len(columns)
    game_ids: list[int] = []
    same_list: list[np.ndarray] = []
    same_pairs: list[float] = []
    cross_list: list[np.ndarray] = []
    cross_pairs: list[float] = []

    for game_id, game in usable.groupby("game_id", sort=True):
        totals: list[np.ndarray] = []
        counts: list[int] = []
        same = np.zeros((n, n), dtype=float)
        same_count = 0.0
        for _, team in game.groupby("team_id", sort=True):
            values = team[columns].to_numpy(dtype=float)
            count = values.shape[0]
            total = values.sum(axis=0)
            totals.append(total)
            counts.append(count)
            if count >= 2:
                same += np.outer(total, total) - values.T @ values
                same_count += count * (count - 1)

        cross = np.zeros((n, n), dtype=float)
        cross_count = 0.0
        for first in range(len(totals)):
            for second in range(first + 1, len(totals)):
                outer = np.outer(totals[first], totals[second])
                cross += outer + outer.T
                cross_count += 2.0 * counts[first] * counts[second]

        game_ids.append(int(game_id))
        same_list.append(same)
        same_pairs.append(same_count)
        cross_list.append(cross)
        cross_pairs.append(cross_count)

    return PerGameMoments(
        stats=stats,
        game_ids=np.array(game_ids),
        same=np.stack(same_list),
        same_pairs=np.array(same_pairs, dtype=float),
        cross=np.stack(cross_list),
        cross_pairs=np.array(cross_pairs, dtype=float),
    )


# ----------------------------------------------------------------------
# Hermite coefficients and the bridge itself
# ----------------------------------------------------------------------


def _hermite_value(z: np.ndarray, order: int) -> np.ndarray:
    coefficients = np.zeros(order + 1)
    coefficients[order] = 1.0
    return hermite_e.hermeval(z, coefficients)


def conditional_hermite_moments(
    cdf_lower: np.ndarray,
    cdf_upper: np.ndarray,
    order: int = DEFAULT_BRIDGE_ORDER,
) -> list[np.ndarray]:
    """``M_k(y) = E[He_k(Z) | Y = y]`` for one observation's PIT interval.

    Under the model ``Y = y`` is exactly the event ``Z in (t_l, t_u]`` with
    ``t = Phi^{-1}(F)``, so the conditional moment is a closed-form ratio of
    Hermite-weighted normal tails:

        int He_k(z) phi(z) dz = -He_{k-1}(z) phi(z)

    which gives ``M_k = (He_{k-1}(t_l) phi(t_l) - He_{k-1}(t_u) phi(t_u)) / (F_u - F_l)``.

    This replaces the single-draw estimator ``He_k(z)``. Both are unbiased for
    the same quantity -- ``z`` is the randomized PIT, so ``z | Y`` and
    ``Z | Y`` are the same truncated normal -- but ``M_k`` is the conditional
    expectation of the one the draw estimates, so it carries none of the
    randomization's variance. Rao-Blackwell, not a different estimand.
    """
    lower = np.clip(np.asarray(cdf_lower, dtype=float), 1e-15, 1.0 - 1e-15)
    upper = np.clip(np.asarray(cdf_upper, dtype=float), 1e-15, 1.0 - 1e-15)
    t_lower = stats.norm.ppf(lower)
    t_upper = stats.norm.ppf(upper)
    mass = np.clip(upper - lower, 1e-15, None)
    density_lower = stats.norm.pdf(t_lower)
    density_upper = stats.norm.pdf(t_upper)

    hermite_lower = [np.ones_like(t_lower), t_lower]
    hermite_upper = [np.ones_like(t_upper), t_upper]
    out: list[np.ndarray] = []
    for k in range(1, order + 1):
        out.append(
            (
                hermite_lower[k - 1] * density_lower
                - hermite_upper[k - 1] * density_upper
            )
            / mass
        )
        hermite_lower.append(
            t_lower * hermite_lower[k] - (k - 1) * hermite_lower[k - 1]
        )
        hermite_upper.append(
            t_upper * hermite_upper[k] - (k - 1) * hermite_upper[k - 1]
        )
    return out


def transmission_coefficient_columns(
    frame: pd.DataFrame,
    stats: Sequence[str],
    order: int = DEFAULT_BRIDGE_ORDER,
    bins: int = DEFAULT_COEFFICIENT_BINS,
    count_prefix: str = "e_",
    mean_prefix: str = "analytic_mean_",
    lower_prefix: str = "cdf_lower_",
    upper_prefix: str = "cdf_upper_",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add the transmission coefficients of *both* observable moments.

    Two different observables are read off the same latent vector and neither
    of them is the latent vector:

    ``h{k}_{stat}``
        coefficients of the standardized count residual ``e``, estimated by
        ``e_i M_k(y_i)``. This is the bridge of the brief.

    ``g{k}_{stat}``
        coefficients of ``m(y) = E[Z | Y = y]``, which is what the *recorded*
        latent column measures. The residual build stores the randomized PIT,
        and the randomization is independent of ``Z`` given ``Y``, so

            Corr(z_a, z_b) = Corr(m(Y_a), m(Y_b)) = sum_k rho^k g_{a,k} g_{b,k} / k!

        with ``g_1 = E[m(Y)^2] < 1``. The recorded latent correlation is
        therefore an *attenuated* reading of ``rho``, not ``rho`` itself. This
        is not a modelling choice to be argued about; it follows from the
        residual build storing a randomized PIT, and it is measurable: ``g_1``
        runs from 0.93 for points down to 0.37 for blocks, because a margin
        with mean 0.3 discards far more of the latent ordering than one with
        mean 20.

    Carrying both means the two sources can be inverted onto a common ``rho``
    before they are compared, which is the only way an agreement test between
    them means anything: comparing a latent reading attenuated by ``g`` with a
    count reading attenuated by ``h`` would report a conflict wherever the two
    attenuations differ, whether or not the model was wrong.

    Both are smoothed within quantile bins of the analytic mean, which is the
    quantity they actually vary with.
    """
    out = frame.copy()
    diagnostics: dict[str, object] = {
        "bins": int(bins),
        "order": int(order),
        "by_stat": {},
    }
    for stat in stats:
        count = out[f"{count_prefix}{stat}"].to_numpy(dtype=float)
        mean = out[f"{mean_prefix}{stat}"].to_numpy(dtype=float)
        lower = out[f"{lower_prefix}{stat}"].to_numpy(dtype=float)
        upper = out[f"{upper_prefix}{stat}"].to_numpy(dtype=float)
        valid = (
            np.isfinite(count)
            & np.isfinite(mean)
            & np.isfinite(lower)
            & np.isfinite(upper)
        )
        conditional = conditional_hermite_moments(lower, upper, order=order)
        labels = np.full(len(out), -1, dtype=int)
        labels[valid] = pd.qcut(
            pd.Series(mean[valid]), bins, labels=False, duplicates="drop"
        ).to_numpy()

        per_stat: dict[str, object] = {}
        for k in range(1, order + 1):
            products = {
                "h": count * conditional[k - 1],
                "g": conditional[0] * conditional[k - 1],
            }
            for name, product in products.items():
                raw = np.where(valid, product, np.nan)
                table = pd.DataFrame({"bin": labels, "value": raw})
                means = table.loc[labels >= 0].groupby("bin")["value"].mean()
                smoothed = np.where(
                    labels >= 0, means.reindex(labels).to_numpy(), np.nan
                )
                out[f"{name}{k}_{stat}"] = smoothed
                per_stat[f"{name}{k}_pooled"] = float(np.nanmean(raw))
                per_stat[f"{name}{k}_min_bin"] = float(np.nanmin(means.to_numpy()))
                per_stat[f"{name}{k}_max_bin"] = float(np.nanmax(means.to_numpy()))
        # The latent column's own first-order attenuation, reported because it
        # is the number that decides whether the two sources can be compared.
        per_stat["latent_attenuation_g1"] = per_stat["g1_pooled"]
        diagnostics["by_stat"][stat] = per_stat  # type: ignore[index]
    return out, diagnostics


def bridge_forward(
    latent: np.ndarray,
    gains: Sequence[np.ndarray],
) -> np.ndarray:
    """``sum_k rho^k G_k / k!``: the count moment a latent target implies."""
    latent = np.asarray(latent, dtype=float)
    out = np.zeros_like(latent)
    for order, gain in enumerate(gains, start=1):
        out = out + (latent**order) * np.asarray(gain, dtype=float) / math.factorial(order)
    return out


def bridge_inverse(
    count: np.ndarray,
    gains: Sequence[np.ndarray],
    iterations: int = 60,
) -> np.ndarray:
    """Latent target whose bridge image is ``count``, entry by entry.

    The forward map is a low-order polynomial in ``rho`` that is monotone over
    the range these buckets occupy, so a Newton iteration from the first-order
    solution converges in a handful of steps. Entries whose first-order gain is
    too small to invert come back as NaN rather than as a large number produced
    by dividing by nearly nothing.
    """
    count = np.asarray(count, dtype=float)
    first = np.asarray(gains[0], dtype=float)
    invertible = np.abs(first) >= MIN_INVERTIBLE_GAIN
    guess = np.where(invertible, count / np.where(invertible, first, 1.0), np.nan)

    for _ in range(iterations):
        value = bridge_forward(guess, gains) - count
        derivative = np.zeros_like(guess)
        for order, gain in enumerate(gains, start=1):
            derivative = derivative + order * (guess ** (order - 1)) * np.asarray(
                gain, dtype=float
            ) / math.factorial(order)
        step = np.where(np.abs(derivative) > 1e-12, value / derivative, 0.0)
        guess = guess - step
        if np.nanmax(np.abs(step)) < 1e-15:
            break
    return np.where(invertible, guess, np.nan)


# ----------------------------------------------------------------------
# combining the two sources
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePair:
    """Two estimates of the same latent target, with their joint covariance."""

    latent: np.ndarray
    bridge: np.ndarray
    variance_latent: np.ndarray
    variance_bridge: np.ndarray
    covariance: np.ndarray

    def disagreement(self) -> np.ndarray:
        return self.bridge - self.latent

    def disagreement_variance(self) -> np.ndarray:
        """Variance of the *difference*, which is what homogeneity is tested on."""
        return np.maximum(
            self.variance_latent + self.variance_bridge - 2.0 * self.covariance,
            0.0,
        )


def homogeneity_test(pair: SourcePair) -> dict[str, object]:
    """Do the two sources estimate the same latent target?

    Inverse-variance pooling assumes they do. The statistic is the squared
    disagreement over its own variance, which is chi-square with one degree of
    freedom under that assumption, so a large value says the assumption is
    wrong rather than that one source is noisy. The *correlated* variance form
    matters here: the two estimates share games, so ignoring the covariance
    would overstate the disagreement variance and hide a real conflict.
    """
    difference = pair.disagreement()
    variance = pair.disagreement_variance()
    finite = np.isfinite(difference) & (variance > 0)
    statistic = np.where(finite, difference**2 / np.where(finite, variance, 1.0), np.nan)
    pooled = float(np.nansum(statistic))
    degrees = int(np.sum(finite))
    return {
        "per_entry_chi_square": statistic,
        "pooled_chi_square": pooled,
        "degrees_of_freedom": degrees,
        "p_value": float(stats.chi2.sf(pooled, degrees)) if degrees > 0 else float("nan"),
        "max_disagreement_in_sigma": float(np.nanmax(np.sqrt(statistic))),
        "homogeneous": bool(degrees > 0 and stats.chi2.sf(pooled, degrees) > 0.05),
    }


def combine_sources(
    pair: SourcePair,
    bridge_weight_cap: float = 1.0,
    heterogeneity: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Pool the two sources with the correct correlated-estimator algebra.

    With ``V`` the ``2 x 2`` covariance of ``(rho_latent, rho_bridge)`` the
    minimum-variance unbiased combination is ``w = V^{-1} 1 / (1' V^{-1} 1)``,
    which reduces to inverse-variance weighting only when the covariance is
    zero. It is not zero here -- both estimates come from the same games -- and
    with a strong positive covariance the GLS weights can leave ``[0, 1]`` and
    extrapolate beyond both estimates, so they are clipped and the clipping is
    reported rather than hidden.

    ``heterogeneity`` adds a method-of-moments model-error variance to the
    *bridge* when the two sources disagree by more than their joint sampling
    uncertainty. The asymmetry is the point. The direct estimate measures the
    latent correlation directly; the bridge-implied one infers it through a
    Gaussian copula and a measured gain, so it carries an assumption the
    direct one does not. When the two conflict beyond sampling noise it is that
    assumption which is refuted, and charging the excess to the bridge is what
    makes "no bridge target gets meaningful weight if its uncertainty is too
    large" actually bite. Splitting the excess evenly instead drives every
    weight towards one half no matter how badly the bridge is refuted, which
    reads a conflict as a reason to trust the suspect source *more*.

    The model-error variance is one scalar per block, pooled over the block's
    entries. A per-entry version would be a bucket-specific override by
    another name: each entry's own disagreement would set its own weight, so
    the layer could silently rescue or abandon any single bucket.

    ``bridge_weight_cap`` is the ceiling on the bridge's share. It is the one
    scalar this layer adds, and it is selected on pre-2024 folds.
    """
    variance_latent = np.asarray(pair.variance_latent, dtype=float)
    variance_bridge = np.asarray(pair.variance_bridge, dtype=float)
    covariance = np.asarray(pair.covariance, dtype=float)

    model_error = 0.0
    if heterogeneity:
        disagreement = pair.disagreement()
        expected = pair.disagreement_variance()
        usable_excess = np.isfinite(disagreement) & np.isfinite(expected)
        if np.any(usable_excess):
            model_error = float(
                max(
                    np.mean(
                        disagreement[usable_excess] ** 2 - expected[usable_excess]
                    ),
                    0.0,
                )
            )

    v11 = variance_latent
    v22 = variance_bridge + model_error
    v12 = covariance

    determinant = v11 * v22 - v12**2
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_bridge_weight = (v11 - v12) / (v11 + v22 - 2.0 * v12)
    singular = ~np.isfinite(raw_bridge_weight) | (determinant <= 0)
    fallback = v11 / np.maximum(v11 + v22, 1e-300)
    raw_bridge_weight = np.where(singular, fallback, raw_bridge_weight)

    clipped = np.clip(raw_bridge_weight, 0.0, float(bridge_weight_cap))
    usable = np.isfinite(pair.bridge)
    weight = np.where(usable, clipped, 0.0)

    combined = (1.0 - weight) * pair.latent + weight * np.where(
        usable, pair.bridge, 0.0
    )
    combined = 0.5 * (combined + combined.T)

    return combined, {
        "bridge_weight": weight,
        "bridge_weight_before_cap": raw_bridge_weight,
        "bridge_model_error_variance": model_error,
        "entries_using_gls_fallback": int(np.sum(singular)),
        "entries_clipped_by_cap": int(np.sum(clipped != raw_bridge_weight)),
        "entries_without_a_bridge": int(np.sum(~usable)),
        "mean_bridge_weight": float(np.mean(weight)),
        "max_bridge_weight": float(np.max(weight)),
    }


def bootstrap_source_pair(
    latent_moments: PerGameMoments,
    count_moments: PerGameMoments,
    latent_gain_moments: Sequence[PerGameMoments],
    count_gain_moments: Sequence[PerGameMoments],
    draws: int = 400,
    seed: int = 73,
    same_team: bool = True,
) -> SourcePair:
    """Joint game-clustered bootstrap of two readings of one bucket.

    Both sources are expressed in the units of the *recorded latent column*,
    which is what the factor targets and the validation metric are both stated
    in. The count moment reaches those units by a composition, not a single
    step:

        count moment --(invert h)--> rho --(apply g)--> recorded-latent units

    The round trip through ``rho`` is what makes the two comparable. Leaving
    the count reading in count units and calling it a latent target would
    report a conflict wherever the two attenuations merely differ, and
    de-attenuating the *target* instead would move the model off the scale its
    own metric is read on -- which is a different claim than the brief's, and
    one the latent gate would correctly reject.

    One resampled set of games drives every quantity. That is the only way to
    measure the covariance between the two, since they are functions of the
    same games and resampling them independently would report a covariance of
    zero and make the pooling overconfident.
    """
    block = 0 if same_team else 1
    latent_gains = [moment.pooled()[block] for moment in latent_gain_moments]
    count_gains = [moment.pooled()[block] for moment in count_gain_moments]
    direct = latent_moments.pooled()[block]
    bridge = bridge_forward(
        bridge_inverse(count_moments.pooled()[block], count_gains), latent_gains
    )

    rng = np.random.default_rng(seed)
    n_games = latent_moments.n_games
    shape = direct.shape
    latent_draws = np.empty((draws, *shape))
    bridge_draws = np.empty((draws, *shape))
    for draw in range(draws):
        index = rng.integers(0, n_games, size=n_games)
        latent_draws[draw] = latent_moments.pooled(index)[block]
        bridge_draws[draw] = bridge_forward(
            bridge_inverse(
                count_moments.pooled(index)[block],
                [moment.pooled(index)[block] for moment in count_gain_moments],
            ),
            [moment.pooled(index)[block] for moment in latent_gain_moments],
        )

    variance_latent = np.nanvar(latent_draws, axis=0, ddof=1)
    variance_bridge = np.nanvar(bridge_draws, axis=0, ddof=1)
    centred_latent = latent_draws - np.nanmean(latent_draws, axis=0)
    centred_bridge = bridge_draws - np.nanmean(bridge_draws, axis=0)
    covariance = np.nanmean(centred_latent * centred_bridge, axis=0) * draws / (draws - 1)

    return SourcePair(
        latent=direct,
        bridge=bridge,
        variance_latent=variance_latent,
        variance_bridge=variance_bridge,
        covariance=covariance,
    )


def bucket_entry(
    stats: Sequence[str],
    matrix: np.ndarray,
    pair: tuple[str, str],
) -> float:
    index = {stat: position for position, stat in enumerate(stats)}
    return float(matrix[index[pair[0]], index[pair[1]]])


def transmission_report(
    stats: Sequence[str],
    buckets: Sequence[tuple[str, str, tuple[str, str]]],
    same: Mapping[str, np.ndarray],
    cross: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Per-bucket view of whatever matrices are supplied, for the report."""
    out: dict[str, dict[str, float]] = {}
    for name, kind, pair in buckets:
        source = same if kind == "same_team" else cross
        if kind == "same_player":
            continue
        out[name] = {
            key: bucket_entry(stats, matrix, pair) for key, matrix in source.items()
        }
    return out
