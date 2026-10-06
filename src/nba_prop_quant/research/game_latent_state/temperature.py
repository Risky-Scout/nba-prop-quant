"""Exact joint-event probabilities for the dependence-temperature search.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The validation driver prices a same-game conjunction by counting whole-game
Monte Carlo draws. That is the right instrument for the confirmatory run --
it exercises the production inverse CDFs and the Cholesky path end to end --
but it is the wrong one for choosing a scalar, because the quantity being
compared across candidate temperatures is smaller than the Monte Carlo noise
at any affordable draw count. The accepted repair and the production
incumbent differ by about 3e-5 in Brier; resolving a difference that size by
simulation needs far more draws than a search over a grid can afford.

So this module prices the same events exactly instead.

A leg is a threshold on a count, the count is a monotone transform of one
latent normal, and therefore the leg is a threshold on that normal:

    Y = min{k : F(k) >= Phi(Z)}      (``simulator.grid_ppf``)
    Y <= m   <=>   Phi(Z) <= F(m)   <=>   Z <= Phi^{-1}(F(m))

A conjunction of such legs is an orthant of a multivariate normal, so its
probability is a Gaussian orthant integral over the sub-correlation matrix of
the legs' latent dimensions. That integral is what ``scipy`` computes with
Genz's algorithm, to a tolerance we set, and with a fixed generator it is
reproducible to the last bit -- so two temperatures priced here differ only
because their correlations differ, never because their draws differed.

The correlation matrix still comes from ``build_game_covariance``, so the
same-player pinning, the shared-factor shrink and the PSD projection all
apply exactly as they do in simulation. Only the pricing is analytic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal, norm

from .covariance import GameCovariance

#: Absolute tolerance requested of the orthant integral. Chosen against the
#: sensitivity that matters: a 10% change in one correlation entry moves a
#: four-leg probability by about 2e-3, so a 1e-7 integration tolerance leaves
#: four orders of magnitude of headroom for the comparison this drives.
ORTHANT_ABSOLUTE_TOLERANCE = 1e-7

#: Quasi-random point budget for the orthant integral. The tolerance above
#: is a request, not a guarantee; this is the ceiling on the work spent
#: trying to meet it.
ORTHANT_MAX_POINTS = 200_000

#: Fixed generator seed for the integral. Every temperature prices every event
#: with the same quasi-random stream, which removes integration noise from the
#: *comparison* even where it survives in the level.
ORTHANT_SEED = 20_260_206


@dataclass(frozen=True)
class LatentOrthant:
    """One conjunction expressed as ``W <= limit`` on correlated normals.

    ``signs`` is ``-1`` for an over leg and ``+1`` for an under leg, and
    ``W = diag(signs) Z``, so the orthant is upper-tailed in the original
    normals wherever the leg was. Degenerate legs -- ones a marginal already
    settles at probability 0 or 1 -- are recorded in ``certainty`` and removed
    from the integral rather than pushed through it as an infinite limit.
    """

    columns: tuple[int, ...]
    limits: np.ndarray
    signs: np.ndarray
    certainty: float | None

    @property
    def size(self) -> int:
        return len(self.columns)


def dimension_columns(covariance: GameCovariance) -> dict[tuple[int, str], int]:
    """Column of the game correlation matrix for each ``(player, stat)``."""
    return {
        (int(dimension.player_id), str(dimension.stat)): position
        for position, dimension in enumerate(covariance.dimensions)
    }


def latent_orthant(
    legs: Sequence[object],
    columns: Mapping[tuple[int, str], int],
    reference: Mapping[tuple[int, str], object],
) -> LatentOrthant:
    """Translate a conjunction of prop legs into a latent-normal orthant.

    The threshold uses the *same* tabulated CDF the simulator pushes uniforms
    through, so the translation is exact rather than approximate: there is no
    continuity correction and no normal approximation to a discrete count
    anywhere in it.
    """
    chosen: list[int] = []
    limits: list[float] = []
    signs: list[float] = []

    for leg in legs:
        key = (int(leg.player_id), str(leg.stat))  # type: ignore[attr-defined]
        if key not in columns or key not in reference:
            raise KeyError(f"leg {key} is not a simulated dimension")
        cdf = np.asarray(reference[key].cdf, dtype=float)  # type: ignore[attr-defined]
        floor_line = int(np.floor(float(leg.line)))  # type: ignore[attr-defined]

        if floor_line < 0:
            # Every count is at least zero, so "under 0.5-and-below" is
            # impossible and "over" is certain.
            below = 0.0
        elif floor_line >= len(cdf):
            below = 1.0
        else:
            below = float(cdf[floor_line])

        if str(leg.side) == "over":  # type: ignore[attr-defined]
            probability = 1.0 - below
            sign = -1.0
        else:
            probability = below
            sign = 1.0

        if probability <= 0.0:
            return LatentOrthant((), np.empty(0), np.empty(0), 0.0)
        if probability >= 1.0:
            continue

        chosen.append(int(columns[key]))
        limits.append(float(norm.ppf(below)))
        signs.append(sign)

    if not chosen:
        return LatentOrthant((), np.empty(0), np.empty(0), 1.0)

    return LatentOrthant(
        columns=tuple(chosen),
        limits=np.asarray(limits, dtype=float),
        signs=np.asarray(signs, dtype=float),
        certainty=None,
    )


def orthant_probability(
    orthant: LatentOrthant,
    correlation: np.ndarray,
    absolute_tolerance: float = ORTHANT_ABSOLUTE_TOLERANCE,
    max_points: int = ORTHANT_MAX_POINTS,
    seed: int = ORTHANT_SEED,
) -> float:
    """``P(conjunction)`` under one game correlation matrix, exactly.

    One dimension is a univariate normal CDF and needs no integration; two or
    more go to Genz's algorithm with a fixed generator.
    """
    if orthant.certainty is not None:
        return float(orthant.certainty)

    index = np.asarray(orthant.columns, dtype=int)
    block = np.asarray(correlation, dtype=float)[np.ix_(index, index)]
    flip = np.diag(orthant.signs)
    block = flip @ block @ flip
    upper = orthant.signs * orthant.limits

    if orthant.size == 1:
        return float(norm.cdf(upper[0]))

    value = multivariate_normal.cdf(
        upper,
        mean=np.zeros(orthant.size),
        cov=block,
        abseps=absolute_tolerance,
        maxpts=max_points,
        rng=np.random.default_rng(seed),
    )
    return float(np.clip(value, 0.0, 1.0))


def independent_product(
    orthant: LatentOrthant,
) -> float:
    """The conjunction's probability if every leg were independent.

    Reported alongside the coupled price so the share of the joint effect a
    temperature actually carries is visible, rather than inferred.
    """
    if orthant.certainty is not None:
        return float(orthant.certainty)
    upper = orthant.signs * orthant.limits
    return float(np.prod(norm.cdf(upper)))
