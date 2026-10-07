"""Randomized PIT and Gaussianization for discrete count marginals.

SHADOW / RESEARCH ONLY.

The production marginals are discrete (Poisson / NB / ZINB). A plain
``F(y)`` transform of a discrete variable is not uniform, and the mid-PIT
``F(y-1) + 0.5 * pmf(y)`` transform used by the incumbent same-player copula
fit is uniform only on average: it is systematically under-dispersed, which
biases any *correlation* estimated from it toward zero.

For estimating a latent Gaussian dependence structure we therefore use the
exact randomized PIT

    u = F(y - 1) + v * (F(y) - F(y - 1)),        v ~ U(0, 1)

which is exactly Uniform(0, 1) under a correct marginal, and then

    z = Phi^{-1}(u).

``v`` must be deterministic so that the residual dataset, the fitted
loadings and every validation number are reproducible. It is derived from a
keyed BLAKE2b digest of the observation identity rather than from a stream
position, so the value attached to one (game, player, stat) observation does
not depend on row order, on the number of rows, or on how the frame was
partitioned.

Only the Gaussianization step clips, and it clips at a single documented
numerical epsilon.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# Documented numerical epsilon. Applied to ``u`` immediately before
# ``norm.ppf`` so the Gaussian residual stays finite. At this epsilon the
# clip bound is |z| <= 7.03, which is far outside any plausible NBA
# box-score residual, so the clip is a numerical guard rather than a
# modelling choice.
PIT_EPSILON = 1e-12

# Hash personalisation. Changing it changes every randomized PIT draw, so it
# is part of the artifact contract.
PIT_HASH_PERSON = b"nbaglsv1"

_UINT64_SCALE = float(1 << 64)


def _as_int_array(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    try:
        return array.astype(np.int64, copy=False)
    except (TypeError, ValueError) as error:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be integer-like") from error


def deterministic_pit_uniform(
    seed: int,
    game_id: object,
    player_id: object,
    stat: object,
) -> np.ndarray:
    """Return the locked ``v`` draw for each (game, player, stat) observation.

    The draw is a keyed digest of the observation identity, so it is stable
    under reordering, re-partitioning and re-running, and it differs across
    stats for the same player-game (the randomization must not be shared
    across the stat dimensions whose dependence we are trying to measure).
    """
    game = _as_int_array(game_id, "game_id")
    player = _as_int_array(player_id, "player_id")
    stats = np.asarray(stat, dtype=object)
    if stats.ndim == 0:
        stats = stats.reshape(1)

    length = max(len(game), len(player), len(stats))
    game = np.broadcast_to(game, (length,))
    player = np.broadcast_to(player, (length,))
    stats = np.broadcast_to(stats, (length,))

    key = int(seed).to_bytes(8, "big", signed=True)
    out = np.empty(length, dtype=float)
    for index in range(length):
        digest = hashlib.blake2b(
            b"|".join(
                (
                    key,
                    int(game[index]).to_bytes(8, "big", signed=True),
                    int(player[index]).to_bytes(8, "big", signed=True),
                    str(stats[index]).encode("utf-8"),
                )
            ),
            digest_size=8,
            person=PIT_HASH_PERSON,
        ).digest()
        # Map to the open interval (0, 1): the +0.5 offset keeps both
        # endpoints unreachable, so u can only hit 0 or 1 when the marginal
        # itself assigns the observation zero probability mass.
        out[index] = (int.from_bytes(digest, "big") + 0.5) / _UINT64_SCALE
    return out


def randomized_pit(
    cdf_lower: np.ndarray,
    cdf_upper: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """``u = F(y-1) + v * (F(y) - F(y-1))`` with no clipping.

    ``cdf_lower`` is ``F(y - 1)`` and ``cdf_upper`` is ``F(y)``. The returned
    value is the exact randomized PIT; clipping belongs to
    :func:`gaussianize`, which is the only place a numerical bound is
    applied.
    """
    lower = np.asarray(cdf_lower, dtype=float)
    upper = np.asarray(cdf_upper, dtype=float)
    draw = np.asarray(v, dtype=float)

    if not (lower.shape == upper.shape == draw.shape):
        raise ValueError("cdf_lower, cdf_upper and v must share a shape")

    mass = upper - lower
    # A correct CDF pair is monotone. Negative mass means the marginal was
    # evaluated inconsistently, which must surface rather than be absorbed.
    if np.any(mass < -1e-9):
        raise ValueError("F(y) < F(y-1): the marginal CDF is not monotone")

    return lower + draw * np.clip(mass, 0.0, None)


@dataclass(frozen=True)
class GaussianizationResult:
    z: np.ndarray
    u: np.ndarray
    clipped: np.ndarray

    @property
    def clipped_fraction(self) -> float:
        if self.clipped.size == 0:
            return 0.0
        return float(np.mean(self.clipped))


def gaussianize(u: np.ndarray, epsilon: float = PIT_EPSILON) -> GaussianizationResult:
    """``z = Phi^{-1}(clip(u, eps, 1 - eps))``.

    Returns the clipped uniform alongside the latent normal and a mask of
    which observations the epsilon actually bound, so the residual-dataset
    manifest can report the clip rate instead of hiding it.
    """
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")

    raw = np.asarray(u, dtype=float)
    if np.any(~np.isfinite(raw)):
        raise ValueError("randomized PIT produced a non-finite uniform")
    if np.any(raw < -1e-9) or np.any(raw > 1.0 + 1e-9):
        raise ValueError("randomized PIT produced a uniform outside [0, 1]")

    clipped_mask = (raw < epsilon) | (raw > 1.0 - epsilon)
    bounded = np.clip(raw, epsilon, 1.0 - epsilon)
    return GaussianizationResult(
        z=norm.ppf(bounded),
        u=bounded,
        clipped=clipped_mask,
    )


def mid_pit(cdf_lower: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    """The incumbent copula's mid-PIT transform, for comparison only.

    ``nba_prop_quant.copula.GaussianCopula.fit`` Gaussianizes with this
    transform. The shadow layer reproduces it when it needs to speak the
    incumbent's units, and never uses it to estimate a new correlation.
    """
    return np.asarray(cdf_lower, dtype=float) + 0.5 * np.asarray(pmf, dtype=float)
