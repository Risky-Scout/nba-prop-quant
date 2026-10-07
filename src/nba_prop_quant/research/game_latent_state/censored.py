"""Interval-censored Gaussian-copula dependence estimators.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Why this module exists
----------------------
The dependence layer is parameterised by a latent Gaussian correlation
``rho``, but every estimator the pipeline has used so far reads that
parameter off a *transformed* observation and therefore reads it low.

:mod:`pit` Gaussianizes a count with the randomized PIT,
``z = Phi^{-1}(F(y - 1) + v (F(y) - F(y - 1)))`` with ``v`` uniform and
independent of everything. That makes ``z`` exactly standard normal, which is
what the marginal diagnostics want, but it does not make ``Corr(z_a, z_b)``
equal to ``rho``. Conditioning on the counts,

    Cov(z_a, z_b) = E[m(Y_a) m(Y_b)],   m(y) = E[Z | Y = y]

because the two jitter draws are independent given the counts. ``m`` is the
conditional mean of the latent inside the observed cell, so it is a *smoothed*
version of the latent, and smoothing costs correlation: writing ``m``'s
Hermite expansion gives ``Corr(z_a, z_b) = sum_k rho^k g_{a,k} g_{b,k} / k!``
with ``g_1 = E[m(Y)^2] < 1``. The jitter is not noise that averages away over
seeds -- it is in the denominator, because ``Var(z) = 1`` counts it. So the
randomized-PIT reading is a *biased-low* estimator of ``rho`` no matter how
many seeds are averaged, and the bias is a pure function of how coarse the
count is.

The estimator here throws the jitter away. Observing ``Y = y`` says exactly
that the latent fell in the cell ``(Phi^{-1}(F(y - 1)), Phi^{-1}(F(y))]`` and
nothing more, so a pair of counts is an interval-censored observation of a
bivariate normal and the likelihood of one pair is the rectangle probability

    P(rho) = Phi_2(a_u, b_u; rho) - Phi_2(a_l, b_u; rho)
             - Phi_2(a_u, b_l; rho) + Phi_2(a_l, b_l; rho)

which depends on ``rho`` and the two production margins and on nothing else.
Maximising the sum of ``log P`` over the pairs a bucket averages estimates
``rho`` itself, with no attenuation to invert afterwards.

It is a *pseudo*-likelihood: a player-game appears in every pair its roster
admits, so the pair terms are not independent. The maximiser is still
consistent for ``rho`` -- each term is a correctly specified bivariate
likelihood, so the expected score vanishes at the truth -- but the curvature
is not the information, which is why the standard errors here come from a
game-clustered bootstrap and never from the Hessian.

Evaluating the rectangle
------------------------
``Phi_2`` has no closed form, but its ``rho`` derivative does, and the
rectangle's derivative telescopes to four densities::

    dP/drho = phi_2(a_u, b_u; rho) - phi_2(a_l, b_u; rho)
              - phi_2(a_u, b_l; rho) + phi_2(a_l, b_l; rho)

with ``P(0) = p_a p_b`` by independence. So the rectangle is a one-dimensional
integral of an analytic integrand from the known value at zero, which
Gauss-Legendre resolves to float64 in a handful of nodes at the correlation
magnitudes this layer carries. No bivariate CDF routine is called, and the
whole pair set is evaluated as one vectorised expression per node.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .pit import deterministic_pit_uniform, gaussianize, randomized_pit

#: Probability clip for the threshold transform. Matches ``PIT_EPSILON`` in
#: the residual builder so the two estimators see the same support.
THRESHOLD_EPSILON = 1e-12

#: Gauss-Legendre nodes for the rectangle integral. The integrand is a
#: bivariate normal density in ``rho`` over an interval no wider than the
#: correlations this layer carries, so this is far past convergence; the
#: node count is asserted sufficient in the test suite by refinement.
DEFAULT_RECTANGLE_NODES = 24

#: Smallest rectangle probability kept in the log-likelihood. A pair whose
#: joint cell is below the float64 noise floor carries no information and
#: would otherwise contribute a spurious ``-inf``.
MIN_RECTANGLE_PROBABILITY = 1e-300


def latent_thresholds(
    cdf_lower: np.ndarray,
    cdf_upper: np.ndarray,
    epsilon: float = THRESHOLD_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """``(Phi^{-1}(F(y - 1)), Phi^{-1}(F(y)))``: the observed latent cell.

    These are the *same* two numbers the randomized PIT interpolates between,
    so the two estimators read identical marginal information and differ only
    in what they do with it.
    """
    lower = np.clip(np.asarray(cdf_lower, dtype=float), epsilon, 1.0 - epsilon)
    upper = np.clip(np.asarray(cdf_upper, dtype=float), epsilon, 1.0 - epsilon)
    upper = np.maximum(upper, lower)
    return norm.ppf(lower), norm.ppf(upper)


@dataclass(frozen=True)
class PairIntervals:
    """The censoring rectangles of every ordered pair a bucket averages.

    One entry per ordered pair, so the pair set is exactly the one
    :func:`factors.pair_moments` pools its second moment over and the two
    estimators are comparable pair for pair. ``game_id`` is carried so the
    bootstrap can resample the independent unit.
    """

    a_lower: np.ndarray
    a_upper: np.ndarray
    b_lower: np.ndarray
    b_upper: np.ndarray
    game_id: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            self.a_lower.shape,
            self.a_upper.shape,
            self.b_lower.shape,
            self.b_upper.shape,
            self.game_id.shape,
        }
        if len(shapes) != 1:
            raise ValueError("every pair array must have the same shape")
        if self.a_lower.ndim != 1:
            raise ValueError("pair arrays must be one-dimensional")

    def __len__(self) -> int:
        return int(self.a_lower.size)

    @property
    def independent_probability(self) -> np.ndarray:
        """``p_a p_b``: the rectangle at ``rho = 0``, where it factorises."""
        mass_a = norm.cdf(self.a_upper) - norm.cdf(self.a_lower)
        mass_b = norm.cdf(self.b_upper) - norm.cdf(self.b_lower)
        return mass_a * mass_b

    def take(self, index: np.ndarray) -> PairIntervals:
        return PairIntervals(
            a_lower=self.a_lower[index],
            a_upper=self.a_upper[index],
            b_lower=self.b_lower[index],
            b_upper=self.b_upper[index],
            game_id=self.game_id[index],
        )


def _threshold_frame(
    frame: pd.DataFrame,
    stats: Sequence[str],
    epsilon: float,
) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    for stat in stats:
        lower, upper = latent_thresholds(
            out[f"cdf_lower_{stat}"].to_numpy(dtype=float),
            out[f"cdf_upper_{stat}"].to_numpy(dtype=float),
            epsilon=epsilon,
        )
        out[f"_t_lower_{stat}"] = lower
        out[f"_t_upper_{stat}"] = upper
    return out


def same_team_pair_intervals(
    frame: pd.DataFrame,
    stat_a: str,
    stat_b: str,
    epsilon: float = THRESHOLD_EPSILON,
) -> PairIntervals:
    """Censoring rectangles of every ordered distinct same-team pair.

    Mirrors the ``outer(total, total) - z' z`` same-team term of
    :func:`factors.pair_moments`: both orderings of a roster pair are present,
    and a player is never paired with itself.
    """
    prepared = _threshold_frame(frame, (stat_a, stat_b), epsilon)
    chunks: list[tuple[np.ndarray, ...]] = []
    for (game_id, _), team in prepared.groupby(["game_id", "team_id"], sort=True):
        size = len(team)
        if size < 2:
            continue
        rows, columns = np.meshgrid(
            np.arange(size), np.arange(size), indexing="ij"
        )
        distinct = rows != columns
        first, second = rows[distinct], columns[distinct]
        chunks.append(
            (
                team[f"_t_lower_{stat_a}"].to_numpy()[first],
                team[f"_t_upper_{stat_a}"].to_numpy()[first],
                team[f"_t_lower_{stat_b}"].to_numpy()[second],
                team[f"_t_upper_{stat_b}"].to_numpy()[second],
                np.full(first.size, int(game_id), dtype=np.int64),
            )
        )
    if not chunks:
        raise ValueError("no same-team pair survived the roster filter")
    return PairIntervals(*(np.concatenate(part) for part in zip(*chunks)))


def cross_team_pair_intervals(
    frame: pd.DataFrame,
    stat_a: str,
    stat_b: str,
    epsilon: float = THRESHOLD_EPSILON,
) -> PairIntervals:
    """Censoring rectangles of every ordered opposing-team pair.

    Mirrors the ``outer + outer.T`` cross-team term of
    :func:`factors.pair_moments`, so both ``(stat_a on one side, stat_b on the
    other)`` orientations are present and the pair count is ``2 n_1 n_2``.
    """
    prepared = _threshold_frame(frame, (stat_a, stat_b), epsilon)
    chunks: list[tuple[np.ndarray, ...]] = []
    for game_id, game in prepared.groupby("game_id", sort=True):
        sides = [team for _, team in game.groupby("team_id", sort=True)]
        if len(sides) != 2:
            continue
        for left, right in ((sides[0], sides[1]), (sides[1], sides[0])):
            rows, columns = np.meshgrid(
                np.arange(len(left)), np.arange(len(right)), indexing="ij"
            )
            first, second = rows.ravel(), columns.ravel()
            chunks.append(
                (
                    left[f"_t_lower_{stat_a}"].to_numpy()[first],
                    left[f"_t_upper_{stat_a}"].to_numpy()[first],
                    right[f"_t_lower_{stat_b}"].to_numpy()[second],
                    right[f"_t_upper_{stat_b}"].to_numpy()[second],
                    np.full(first.size, int(game_id), dtype=np.int64),
                )
            )
    if not chunks:
        raise ValueError("no cross-team pair survived the roster filter")
    return PairIntervals(*(np.concatenate(part) for part in zip(*chunks)))


def pair_intervals(
    frame: pd.DataFrame,
    kind: str,
    stat_a: str,
    stat_b: str,
    epsilon: float = THRESHOLD_EPSILON,
) -> PairIntervals:
    """Dispatch on the bucket kind, so one call site serves every bucket."""
    if kind == "same_team":
        return same_team_pair_intervals(frame, stat_a, stat_b, epsilon)
    if kind == "cross_team":
        return cross_team_pair_intervals(frame, stat_a, stat_b, epsilon)
    raise ValueError(f"unsupported bucket kind {kind!r}")


def _bivariate_density(
    first: np.ndarray,
    second: np.ndarray,
    rho: float,
) -> np.ndarray:
    complement = 1.0 - rho * rho
    exponent = (first * first - 2.0 * rho * first * second + second * second) / (
        2.0 * complement
    )
    return np.exp(-exponent) / (2.0 * np.pi * np.sqrt(complement))


def rectangle_probability(
    rho: float,
    intervals: PairIntervals,
    independent: np.ndarray | None = None,
    nodes: int = DEFAULT_RECTANGLE_NODES,
) -> np.ndarray:
    """``P(Z_a in cell_a, Z_b in cell_b; rho)`` for every pair at once.

    Integrates the analytic ``rho`` derivative from ``rho = 0``, where the
    rectangle is the product of the two cell masses. That avoids a bivariate
    normal CDF call entirely and keeps the whole evaluation vectorised.
    """
    rho = float(rho)
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly inside (-1, 1)")
    base = (
        intervals.independent_probability if independent is None else independent
    )
    if rho == 0.0:
        return np.array(base, dtype=float, copy=True)

    abscissa, weights = np.polynomial.legendre.leggauss(int(nodes))
    half = 0.5 * rho
    accumulated = np.zeros_like(intervals.a_lower)
    for node, weight in zip(abscissa, weights):
        point = half * (node + 1.0)
        accumulated += weight * (
            _bivariate_density(intervals.a_upper, intervals.b_upper, point)
            - _bivariate_density(intervals.a_lower, intervals.b_upper, point)
            - _bivariate_density(intervals.a_upper, intervals.b_lower, point)
            + _bivariate_density(intervals.a_lower, intervals.b_lower, point)
        )
    return base + half * accumulated


def censored_log_likelihood(
    rho: float,
    intervals: PairIntervals,
    independent: np.ndarray | None = None,
    nodes: int = DEFAULT_RECTANGLE_NODES,
) -> float:
    """``sum_pairs log P(rho)``: the composite likelihood of one bucket."""
    probability = rectangle_probability(rho, intervals, independent, nodes)
    clipped = np.clip(probability, MIN_RECTANGLE_PROBABILITY, None)
    return float(np.sum(np.log(clipped)))


def censored_score(
    rho: float,
    intervals: PairIntervals,
    independent: np.ndarray | None = None,
    nodes: int = DEFAULT_RECTANGLE_NODES,
) -> float:
    """``d/drho sum_pairs log P(rho)``, in closed form up to the rectangle.

    The rectangle's ``rho`` derivative telescopes to the four corner
    densities, so the score costs one extra density evaluation on top of the
    probability it divides by. That turns the maximisation into a scalar root
    find and makes a game-clustered bootstrap affordable.
    """
    probability = rectangle_probability(rho, intervals, independent, nodes)
    derivative = (
        _bivariate_density(intervals.a_upper, intervals.b_upper, float(rho))
        - _bivariate_density(intervals.a_lower, intervals.b_upper, float(rho))
        - _bivariate_density(intervals.a_upper, intervals.b_lower, float(rho))
        + _bivariate_density(intervals.a_lower, intervals.b_lower, float(rho))
    )
    return float(
        np.sum(derivative / np.clip(probability, MIN_RECTANGLE_PROBABILITY, None))
    )


def censored_copula_mle_golden(
    intervals: PairIntervals,
    bound: float = 0.60,
    tolerance: float = 1e-10,
    nodes: int = DEFAULT_RECTANGLE_NODES,
) -> float:
    """``argmax_rho sum_pairs log P(rho)`` by golden-section search.

    Derivative-free, so it is the independent check that
    :func:`censored_copula_mle`'s root find lands on the same maximiser. The
    test suite asserts the two agree; the driver uses the faster one.
    """
    independent = intervals.independent_probability
    golden = 0.5 * (np.sqrt(5.0) - 1.0)
    low, high = -abs(float(bound)), abs(float(bound))
    left = high - golden * (high - low)
    right = low + golden * (high - low)
    value_left = censored_log_likelihood(left, intervals, independent, nodes)
    value_right = censored_log_likelihood(right, intervals, independent, nodes)
    while high - low > tolerance:
        if value_left >= value_right:
            high, right, value_right = right, left, value_left
            left = high - golden * (high - low)
            value_left = censored_log_likelihood(left, intervals, independent, nodes)
        else:
            low, left, value_left = left, right, value_right
            right = low + golden * (high - low)
            value_right = censored_log_likelihood(
                right, intervals, independent, nodes
            )
    return 0.5 * (low + high)


def censored_copula_mle(
    intervals: PairIntervals,
    bound: float = 0.60,
    tolerance: float = 1e-12,
    nodes: int = DEFAULT_RECTANGLE_NODES,
) -> float:
    """``argmax_rho sum_pairs log P(rho)`` by a bracketed score root find.

    The composite log-likelihood of a bivariate normal correlation is smooth
    and unimodal in ``rho``, so its score is decreasing through a single root
    and Brent's method on the score converges in a handful of evaluations. The
    bracket is symmetric and wide relative to anything this layer carries;
    a bucket whose score does not change sign across it returns the bracket
    end, which the caller reports rather than silently treating as a fit.
    """
    independent = intervals.independent_probability
    low, high = -abs(float(bound)), abs(float(bound))
    score_low = censored_score(low, intervals, independent, nodes)
    score_high = censored_score(high, intervals, independent, nodes)
    if score_low <= 0.0:
        return low
    if score_high >= 0.0:
        return high
    # Brent on a monotone decreasing score: bisection with a secant step
    # accepted only when it stays inside the bracket, which keeps the
    # guaranteed bracketing of bisection and the speed of the secant.
    while high - low > tolerance:
        span = high - low
        secant = low + span * score_low / (score_low - score_high)
        guess = secant if low + 0.01 * span < secant < high - 0.01 * span else (
            0.5 * (low + high)
        )
        value = censored_score(guess, intervals, independent, nodes)
        if value > 0.0:
            low, score_low = guess, value
        elif value < 0.0:
            high, score_high = guess, value
        else:
            return guess
    return 0.5 * (low + high)


def censored_sandwich_se(
    rho: float,
    intervals: PairIntervals,
    step: float = 1e-4,
    nodes: int = DEFAULT_RECTANGLE_NODES,
) -> float:
    """Game-clustered sandwich standard error of :func:`censored_copula_mle`.

    A player-game sits in every pair its roster admits, so the pair scores are
    dependent and the composite-likelihood curvature is not the information.
    The clustered sandwich is the right correction: with ``S_g`` the total
    score of game ``g`` and ``J`` the negative derivative of the total score,

        Var(rho_hat) = J^{-1} (sum_g S_g^2) J^{-1}

    which treats whole games as the independent unit, exactly as the
    bootstrap behind :func:`factors.pair_moments`'s standard errors does.
    ``J`` is taken by a central difference of the analytic score, so no second
    derivative of the rectangle is needed.
    """
    independent = intervals.independent_probability
    probability = rectangle_probability(rho, intervals, independent, nodes)
    derivative = (
        _bivariate_density(intervals.a_upper, intervals.b_upper, float(rho))
        - _bivariate_density(intervals.a_lower, intervals.b_upper, float(rho))
        - _bivariate_density(intervals.a_upper, intervals.b_lower, float(rho))
        + _bivariate_density(intervals.a_lower, intervals.b_lower, float(rho))
    )
    per_pair = derivative / np.clip(probability, MIN_RECTANGLE_PROBABILITY, None)

    games, inverse = np.unique(intervals.game_id, return_inverse=True)
    per_game = np.zeros(games.size, dtype=float)
    np.add.at(per_game, inverse, per_pair)
    meat = float(np.sum(np.square(per_game)))

    forward = censored_score(rho + step, intervals, independent, nodes)
    backward = censored_score(rho - step, intervals, independent, nodes)
    bread = -(forward - backward) / (2.0 * step)
    if bread <= 0.0 or meat <= 0.0:
        return float("nan")
    return float(np.sqrt(meat) / bread)


def clustered_bootstrap_mle(
    intervals: PairIntervals,
    draws: int,
    seed: int,
    bound: float = 0.60,
    nodes: int = DEFAULT_RECTANGLE_NODES,
) -> np.ndarray:
    """Game-clustered bootstrap draws of :func:`censored_copula_mle`.

    Games are the independent unit, matching the standard errors
    :func:`factors.pair_moments` reports, so the two estimators' uncertainties
    are on the same footing.
    """
    games, inverse = np.unique(intervals.game_id, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[order], np.arange(games.size + 1))
    blocks = [order[boundaries[i] : boundaries[i + 1]] for i in range(games.size)]
    generator = np.random.default_rng(seed)
    out = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        picks = generator.integers(0, games.size, size=games.size)
        index = np.concatenate([blocks[pick] for pick in picks])
        out[draw] = censored_copula_mle(
            intervals.take(index), bound=bound, nodes=nodes
        )
    return out


def multi_seed_latent_columns(
    frame: pd.DataFrame,
    stats: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Re-Gaussianize the counts at a different deterministic jitter seed.

    Everything except the jitter is held fixed: the cell bounds come from the
    committed walk-forward marginals, so the only thing that moves between
    seeds is ``v``. That isolates the jitter's contribution to a randomized-PIT
    reading from the attenuation, which does not depend on ``v`` at all.
    """
    out = frame.reset_index(drop=True).copy()
    game_id = out["game_id"].to_numpy()
    player_id = out["player_id"].to_numpy()
    for stat in stats:
        draw = deterministic_pit_uniform(
            seed=int(seed), game_id=game_id, player_id=player_id, stat=stat
        )
        uniform = randomized_pit(
            out[f"cdf_lower_{stat}"].to_numpy(dtype=float),
            out[f"cdf_upper_{stat}"].to_numpy(dtype=float),
            draw,
        )
        out[f"z_{stat}"] = gaussianize(uniform).z
    return out
