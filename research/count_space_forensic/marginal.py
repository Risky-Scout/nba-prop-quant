"""Discrete production margins, prepared for the transmission series.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Three helpers, carried verbatim in behaviour from the rejected V2 search
module ``game_latent_state.bridge`` and kept **here**, inside the study
directory, rather than in the installable package.

That placement is deliberate. ``tests/test_game_latent_state_shadow_clean_candidate.py``
asserts that ``game_latent_state.bridge`` and ``game_latent_state.countspace``
are not importable on this lineage, because five research components were
rejected and the only safe way to keep them rejected was to stop them being
importable at all -- "so the dials cannot be set even by mistake". This study
needs three small, pure functions out of that module and none of its dials, so
it carries the functions and leaves the invariant intact. Nothing here is
reachable from ``import nba_prop_quant``.

What the three do:

``DiscreteMarginal``
    one production margin, truncated at a negligible tail, holding the
    Gaussian cut points ``Phi^{-1}(F(x))`` and the two score-increment
    vectors the count-space and latent-space series need.

``discrete_marginal``
    builds one from the simulator's own inverse-CDF grid, so the series and
    the Monte Carlo read the same discrete law.

``mehler_scores``
    ``d_j = sum_x w_x phi(z_x) h_j(z_x)`` against the *orthonormal* Hermite
    polynomials, which is everything a margin contributes to any pair it
    appears in, at every ``rho`` at once.

Together they give the Mehler expansion of a pair bucket,

    ``T(rho) = sum_j d_j^a d_j^b rho^(j + 1) / (j + 1)``,

which is what lets this study invert a bucket statistic analytically instead
of simulating it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

#: Counts past this much cumulative mass are dropped. Their Gaussian
#: thresholds are effectively ``+inf``, where the bivariate density is zero,
#: so they contribute nothing to any series.
TAIL_MASS = 1e-12

#: A cell thinner than this cannot be resolved in float64, so its score is
#: held at its neighbour's rather than computed as a ratio of noise. That
#: makes the cell's weight exactly zero instead of arbitrary.
MIN_CELL_PROBABILITY = 1e-13

#: Hermite orders kept by default. The recurrence is the orthonormal one, so
#: a high order costs accuracy nothing.
DEFAULT_MEHLER_TERMS = 48


@dataclass(frozen=True)
class DiscreteMarginal:
    """One production marginal, truncated and prepared for the kernel.

    ``thresholds`` are the Gaussian cut points ``Phi^{-1}(F(x))`` for
    ``x = 0 .. K``, where ``K`` is the last count with ``F(K) < 1 - tail``.
    ``count_weights`` and ``latent_weights`` are the score increments the two
    spaces need; both are indexed like ``thresholds``.
    """

    thresholds: np.ndarray
    count_weights: np.ndarray
    latent_weights: np.ndarray
    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.thresholds.ndim != 1:
            raise ValueError("thresholds must be one-dimensional")
        if self.count_weights.shape != self.thresholds.shape:
            raise ValueError("count_weights must align with thresholds")
        if self.latent_weights.shape != self.thresholds.shape:
            raise ValueError("latent_weights must align with thresholds")

    @property
    def support(self) -> int:
        return int(self.thresholds.size)


def discrete_marginal(
    cdf: np.ndarray,
    tail_mass: float = TAIL_MASS,
) -> DiscreteMarginal:
    """Prepare a tabulated ``F(0..K)`` for the transmission kernels.

    ``cdf`` is the simulator's own grid (see
    :func:`simulator.tabulate_inverse_cdf`), so every series built from this
    and the Monte Carlo are reading the same discrete law.
    """
    cdf = np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)

    keep = int(np.searchsorted(cdf, 1.0 - tail_mass, side="left"))
    keep = max(min(keep, cdf.size - 1), 1)
    upper = cdf[:keep]

    thresholds = norm.ppf(upper)
    density = norm.pdf(thresholds)

    # Cell probabilities ``p_x = F(x) - F(x - 1)`` with ``F(-1) = 0``, and the
    # randomised-PIT conditional mean ``m(x) = (phi(z_{x-1}) - phi(z_x)) / p_x``
    # with ``phi(z_{-1}) = 0``. ``m`` is needed one index past ``thresholds``
    # so its increments cover every retained cut point.
    probabilities = np.diff(cdf[: keep + 1], prepend=0.0)
    lower_density = np.concatenate(([0.0], density))
    upper_density = np.concatenate((density, [norm.pdf(norm.ppf(cdf[keep]))]))
    safe = probabilities > MIN_CELL_PROBABILITY
    score = np.zeros_like(probabilities)
    score[safe] = (lower_density[safe] - upper_density[safe]) / probabilities[safe]
    for position in range(1, score.size):
        if not safe[position]:
            score[position] = score[position - 1]

    latent_weights = np.diff(score)
    count_weights = np.ones(keep, dtype=float)

    survival = 1.0 - cdf
    mean = float(np.sum(survival))
    counts = np.arange(cdf.size)
    second = float(np.sum((2.0 * counts + 1.0) * survival))
    variance = max(second - mean**2, 1e-12)

    return DiscreteMarginal(
        thresholds=thresholds,
        count_weights=count_weights,
        latent_weights=latent_weights,
        mean=mean,
        sd=float(np.sqrt(variance)),
    )


def mehler_scores(
    marginal: DiscreteMarginal,
    space: str,
    terms: int = DEFAULT_MEHLER_TERMS,
) -> np.ndarray:
    """``d_j = sum_x w_x phi(z_x) h_j(z_x)`` for ``j = 0 .. terms - 1``.

    ``h_j`` is the *orthonormal* Hermite polynomial ``He_j / sqrt(j!)``,
    evaluated by its own three-term recurrence
    ``h_j = (z h_{j-1} - sqrt(j - 1) h_{j-2}) / sqrt(j)``. Taking the
    normalisation into the recurrence is what keeps this stable: ``He_j`` and
    ``sqrt(j!)`` both overflow long before ``j = 48``, while their ratio stays
    small.
    """
    if terms < 1:
        raise ValueError("terms must be positive")
    if space not in ("count", "latent"):
        raise ValueError(f"unknown transmission space {space!r}")
    weights = (
        marginal.count_weights if space == "count" else marginal.latent_weights
    )
    thresholds = marginal.thresholds
    weighted = weights * norm.pdf(thresholds)

    out = np.empty(terms, dtype=float)
    previous = np.ones_like(thresholds)
    out[0] = float(weighted @ previous)
    if terms == 1:
        return out
    current = thresholds.copy()
    out[1] = float(weighted @ current)
    for order in range(2, terms):
        following = (
            thresholds * current - np.sqrt(order - 1.0) * previous
        ) / np.sqrt(float(order))
        previous, current = current, following
        out[order] = float(weighted @ current)
    return out
