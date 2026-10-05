"""Discrete Gaussian-copula bridges between latent and observable correlation.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The problem
-----------
Accepted shadow V1 and the accepted bucket repair both match the *latent*
``passer_ast_teammate_pts`` bucket well (observed ``+0.047011``, fitted
``+0.044000``) and still miss the *count-space* bucket badly (observed
``+0.058413``, achieved ``+0.037154``). A missing latent correlation cannot
produce that pattern: if the latent target were simply too small, both
metrics would be short by the same relative amount. The gap is a *transform*
effect, and it has two separate causes that this module makes explicit and
invertible.

1.  **Discretisation attenuates in count space.** For a Gaussian copula with
    latent correlation ``rho`` and discrete margins ``F_X``, ``F_Y``,
    Hoeffding's lemma gives the count covariance exactly::

        Cov(X, Y) = sum_{x >= 0} sum_{y >= 0} [ Phi_2(z_x, z_y; rho) - u_x v_y ]

    with ``u_x = F_X(x)``, ``z_x = Phi^{-1}(u_x)``. Dividing by ``sd(X) sd(Y)``
    gives the count-space correlation. The map is strictly increasing in
    ``rho`` but its slope is well below one for low-count margins, so a latent
    ``rho`` of 0.044 cannot deliver a count correlation of 0.058.

2.  **Randomised PIT attenuates in latent space.** The residual dataset's
    ``z`` is ``Phi^{-1}`` of a *randomised* PIT, ``u = F(y-1) + V p_y`` with
    ``V ~ U(0, 1)`` independent of everything. That ``u`` is exactly uniform,
    so ``z`` is exactly standard normal and the marginal diagnostics are
    clean -- but the independent ``V`` is pure noise in the *pair* moment.
    Conditioning on the realised counts,

        E[z_a z_b] = sum_{x, y} P_rho(x, y) m_a(x) m_b(y)

    where ``m(x) = E[Phi^{-1}(u) | y = x] = (phi(z_{x-1}) - phi(z_x)) / p_x``
    is the count's conditional mean score. That is strictly *less* than
    ``rho`` in magnitude, so the measured latent correlation under-states the
    copula parameter that generated it.

One kernel for both bridges
---------------------------
Plackett's formula ``d Phi_2(a, b; r) / dr = phi_2(a, b; r)`` turns both maps
into integrals of the same kernel. Writing a bridge as
``T(rho) = sum_{x, y} P_rho(x, y) a_x b_y`` for scores ``a``, ``b`` and
summing by parts (``phi_2`` vanishes at the truncation ends),

    d T / d rho = sum_{x, y} (a_{x+1} - a_x) (b_{y+1} - b_y) phi_2(z_x, w_y; rho)

so a single weighted kernel ``w_a' Phi_2(rho) w_b`` serves both cases:

* **count space** uses ``a_x = x``, hence unit weights, and normalises by
  ``sd(X) sd(Y)``;
* **latent space** uses ``a_x = m(x)``, hence the increments of the
  conditional-mean score, and needs no normalisation because ``z`` already
  has unit variance.

``phi_2 > 0`` everywhere, so both bridges are *strictly* increasing and
therefore invertible by bisection with no identification ambiguity. The
integral is evaluated by composite Gauss-Legendre quadrature on a fixed grid:
deterministic, seed-free, and with no Monte Carlo error to report.

What the bridge is used for
---------------------------
``required_latent_correlation`` answers "which latent ``rho`` reproduces the
*observed* count correlation", and the V2 fit targets that value through the
PSD factor model rather than writing it into a covariance entry. The latent
bridge ``latent_attenuation`` is the cross-check: if the copula model is
right, inverting the count bridge at the observed count correlation and
inverting the latent bridge at the observed latent correlation must land on
the same ``rho``. They do, which is the evidence that the count-space miss is
a transform artefact rather than a missing dependence channel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

#: Tail mass left outside a truncated CDF grid. Matches the simulator's
#: ``GRID_TAIL_MASS`` so the bridge sees exactly the support the simulator
#: samples from.
BRIDGE_TAIL_MASS = 1e-12

#: Smallest cell probability whose conditional-mean score is trusted. Below
#: it ``(phi(z_{x-1}) - phi(z_x)) / p_x`` is a ratio of two quantities at the
#: float64 noise floor, and the cell carries no weight in the kernel anyway.
MIN_CELL_PROBABILITY = 1e-13

#: Latent correlation range the bridges are tabulated over. Cross-player
#: dependence in this layer is an order of magnitude smaller; the range is
#: wide enough that an inversion never has to extrapolate.
DEFAULT_RHO_MAX = 0.60

#: Composite quadrature resolution: ``DEFAULT_RHO_PANELS`` panels over
#: ``[0, rho_max]`` with ``DEFAULT_GAUSS_NODES``-point Gauss-Legendre on each.
#: The kernel is analytic and slowly varying, so this is far past the point
#: where the quadrature error is visible at float64 precision.
DEFAULT_RHO_PANELS = 30
DEFAULT_GAUSS_NODES = 5


@dataclass(frozen=True)
class DiscreteMarginal:
    """One production marginal, truncated and prepared for the kernel.

    ``thresholds`` are the Gaussian cut points ``Phi^{-1}(F(x))`` for
    ``x = 0 .. K``, where ``K`` is the last count with ``F(K) < 1 - tail``.
    ``count_weights`` and ``latent_weights`` are the score increments the two
    bridges need; both are indexed like ``thresholds``.
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
    tail_mass: float = BRIDGE_TAIL_MASS,
) -> DiscreteMarginal:
    """Prepare a tabulated ``F(0..K)`` for the bridge kernels.

    ``cdf`` is the simulator's own grid (see
    :func:`simulator.tabulate_inverse_cdf`), so the bridge and the Monte Carlo
    are reading the same discrete law.
    """
    cdf = np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)

    # Keep every count whose CDF is still below the tail cut. The dropped tail
    # contributes nothing: its Gaussian threshold is +inf, where phi_2 is zero.
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
    # A cell the float64 grid cannot resolve is held at its neighbour's score,
    # which makes its weight exactly zero rather than a ratio of noise.
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


def bivariate_normal_density(
    first: np.ndarray,
    second: np.ndarray,
    rho: float,
) -> np.ndarray:
    """``phi_2(a, b; rho)`` on the outer grid of ``first`` and ``second``."""
    rho = float(rho)
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly inside (-1, 1)")
    complement = 1.0 - rho**2
    a = np.asarray(first, dtype=float)[:, None]
    b = np.asarray(second, dtype=float)[None, :]
    quadratic = (a**2 - 2.0 * rho * a * b + b**2) / (2.0 * complement)
    return np.exp(-quadratic) / (2.0 * np.pi * np.sqrt(complement))


def _weighted_kernel(
    first: DiscreteMarginal,
    second: DiscreteMarginal,
    rho: float,
    space: str,
) -> float:
    """``w_a' Phi_2(rho) w_b``: the derivative of the bridge at ``rho``."""
    if space == "count":
        wa, wb = first.count_weights, second.count_weights
    elif space == "latent":
        wa, wb = first.latent_weights, second.latent_weights
    else:
        raise ValueError(f"unknown bridge space {space!r}")
    density = bivariate_normal_density(first.thresholds, second.thresholds, rho)
    return float(wa @ density @ wb)


def _gauss_legendre_panels(
    rho_max: float,
    panels: int,
    nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre quadrature on ``[-rho_max, rho_max]``.

    Returns ``(edges, node_grid, weight_grid)`` with the node and weight grids
    shaped ``(2 * panels, nodes)`` so a cumulative sum over panels gives the
    antiderivative at every edge.

    The interval is two-sided rather than ``[0, rho_max]`` reflected, because
    the bridge is *not* an odd function of ``rho``. Its derivative is
    ``w_a' Phi_2(rho) w_b`` and ``phi_2(z, w; -rho) = phi_2(z, -w; rho)``, so
    reflecting the curve is only exact when both margins are symmetric. Count
    margins are strongly skewed, and the cross-team block is where the
    negative entries live, so the negative half is integrated directly. ``0``
    is a panel edge by construction, which is what lets the antiderivative be
    pinned to zero there exactly.
    """
    if rho_max <= 0:
        raise ValueError("rho_max must be positive")
    if panels < 1 or nodes < 1:
        raise ValueError("panels and nodes must be positive")
    base_nodes, base_weights = np.polynomial.legendre.leggauss(nodes)
    positive = np.linspace(0.0, float(rho_max), panels + 1)
    edges = np.concatenate((-positive[::-1][:-1], positive))
    half = np.diff(edges)[:, None] / 2.0
    middle = (edges[:-1] + edges[1:])[:, None] / 2.0
    return edges, middle + half * base_nodes[None, :], half * base_weights[None, :]


@dataclass(frozen=True)
class BridgeCurve:
    """A tabulated, strictly increasing map from latent ``rho`` to an observable.

    ``values[i]`` is the bridge evaluated at ``grid[i]``. The grid spans
    ``[-rho_max, rho_max]`` with zero as an exact edge where the value is
    exactly zero: a zero latent correlation produces a zero correlation in
    either space, for any margins.
    """

    space: str
    grid: np.ndarray
    values: np.ndarray
    pairs: int
    panels: int
    nodes: int

    @property
    def rho_max(self) -> float:
        return float(self.grid[-1])

    @property
    def rho_min(self) -> float:
        return float(self.grid[0])

    @property
    def zero_index(self) -> int:
        return int(np.argmin(np.abs(self.grid)))

    @property
    def feasible_range(self) -> tuple[float, float]:
        """Observable values reachable by a ``rho`` inside the tabulated grid."""
        return float(self.values[0]), float(self.values[-1])

    def is_monotone(self) -> bool:
        return bool(np.all(np.diff(self.values) > 0.0))

    def slope_at_zero(self) -> float:
        """``dT/drho`` at the origin: the bridge's attenuation factor.

        A central difference across the two panels either side of zero, so a
        skewed margin's asymmetry about the origin does not bias it toward
        whichever side happened to be measured.
        """
        zero = self.zero_index
        return float(
            (self.values[zero + 1] - self.values[zero - 1])
            / (self.grid[zero + 1] - self.grid[zero - 1])
        )

    def evaluate(self, rho: float) -> float:
        """The observable implied by ``rho``, by piecewise-linear interpolation."""
        clipped = float(np.clip(float(rho), self.rho_min, self.rho_max))
        return float(np.interp(clipped, self.grid, self.values))

    def invert(self, target: float, tolerance: float = 1e-12) -> float:
        """``T^{-1}(target)``.

        Raises :class:`BridgeNotIdentified` when the target lies outside the
        range the tabulated latent grid can reach, which is the "inversion is
        not identified" rejection the design calls for rather than a silent
        clip to the end of the grid.
        """
        target = float(target)
        low, high = self.feasible_range
        if target > high:
            raise BridgeNotIdentified(
                f"{self.space}-space target {target!r} exceeds the largest "
                f"value the bridge can reach ({high!r}) at rho <= "
                f"{self.rho_max}"
            )
        if target < low:
            raise BridgeNotIdentified(
                f"{self.space}-space target {target!r} is below the smallest "
                f"value the bridge can reach ({low!r}) at rho >= {self.rho_min}"
            )

        left, right = self.rho_min, self.rho_max
        for _ in range(200):
            middle = 0.5 * (left + right)
            if self.evaluate(middle) < target:
                left = middle
            else:
                right = middle
            if right - left < tolerance:
                break
        return 0.5 * (left + right)

    def payload(self) -> dict[str, object]:
        return {
            "space": self.space,
            "pairs": int(self.pairs),
            "quadrature_panels": int(self.panels),
            "quadrature_nodes": int(self.nodes),
            "rho_max": self.rho_max,
            "rho_min": self.rho_min,
            "monotone": self.is_monotone(),
            "slope_at_zero": self.slope_at_zero(),
            "grid": self.grid.tolist(),
            "values": self.values.tolist(),
        }


class BridgeNotIdentified(ValueError):
    """A bridge target is outside the range the latent grid can produce."""


def build_bridge_curve(
    pairs: Sequence[tuple[DiscreteMarginal, DiscreteMarginal]],
    space: str = "count",
    rho_max: float = DEFAULT_RHO_MAX,
    panels: int = DEFAULT_RHO_PANELS,
    nodes: int = DEFAULT_GAUSS_NODES,
) -> BridgeCurve:
    """Pool the bridge over a deterministic sample of marginal pairs.

    The pooled bucket statistic is the *mean over ordered cross-player pairs*
    of the standardized product (that is what ``factors.pair_moments``
    computes), and the bivariate law of one such pair depends only on its own
    two margins and the single latent correlation between them. So the pooled
    bridge is the mean of the per-pair bridges, with no approximation beyond
    the sample of pairs it is pooled over.
    """
    if not pairs:
        raise ValueError("a bridge needs at least one marginal pair")

    edges, node_grid, weight_grid = _gauss_legendre_panels(rho_max, panels, nodes)
    derivative = np.zeros_like(node_grid)

    for panel in range(node_grid.shape[0]):
        for node in range(node_grid.shape[1]):
            rho = float(node_grid[panel, node])
            total = 0.0
            for first, second in pairs:
                kernel = _weighted_kernel(first, second, rho, space)
                if space == "count":
                    kernel /= first.sd * second.sd
                total += kernel
            derivative[panel, node] = total / len(pairs)

    panel_integrals = np.sum(derivative * weight_grid, axis=1)
    antiderivative = np.concatenate(([0.0], np.cumsum(panel_integrals)))
    # A zero latent correlation gives a zero correlation in either space for
    # any margins, so the antiderivative is pinned at the zero edge rather
    # than at the left end of the grid.
    zero = int(np.argmin(np.abs(edges)))
    values = antiderivative - antiderivative[zero]

    return BridgeCurve(
        space=space,
        grid=edges,
        values=values,
        pairs=len(pairs),
        panels=int(panels),
        nodes=int(nodes),
    )


def required_latent_correlation(
    curve: BridgeCurve,
    observed: float,
) -> float:
    """``g^{-1}(C_observed)``: the latent ``rho`` the observation implies.

    This is the quantity the V2 factor fit is steered toward. It is *not*
    written into a covariance entry: the PSD factor model is fitted against it
    like any other target, so positive-definiteness is still structural.
    """
    return curve.invert(observed)


def bridge_adjusted_target(
    latent_target: float,
    required: float,
    weight: float,
) -> float:
    """Blend the shrunk latent target with the bridge-implied one.

    ``weight`` is a single pre-registered scalar for the whole fit, selected on
    pre-2024 inner folds. ``0`` reproduces the accepted repair exactly and
    ``1`` targets the bridge-implied correlation outright. The blend exists
    because the two inner targets pull in opposite directions: correcting the
    discretisation attenuation is what fixes count space, and the *latent*
    metric is measured against the randomised-PIT observation, which is itself
    attenuated -- so a full correction necessarily moves the fitted value away
    from the latent observation even though it moves it toward the truth.
    """
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("bridge weight must lie in [0, 1]")
    return float((1.0 - weight) * latent_target + weight * required)
