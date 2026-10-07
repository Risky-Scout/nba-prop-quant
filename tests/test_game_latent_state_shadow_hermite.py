"""The generic Hermite recurrence, and the claim that fixing it changed nothing.

SHADOW / RESEARCH ONLY.

``conditional_hermite_moments`` used to advance its Hermite table with
``z He_k - (k - 1) He_{k-1}`` instead of the probabilists' recurrence
``z He_k - k He_{k-1}``. Two separate things need pinning, and they pull in
opposite directions, which is why they are tested separately:

*   the orders the bridge has ever been run at -- 1 and 2 -- produce *bitwise*
    identical output before and after, because ``M_k`` reads ``He_{k-1}`` and
    the first wrong table entry is ``He_2``. The frozen candidate's
    probabilities therefore cannot have moved;

*   order 3 and above was genuinely wrong and is now right, checked against
    numpy's Hermite evaluation and against direct numerical quadrature of the
    conditional expectation the function claims to compute.

The defective recurrence is reimplemented here as ``_legacy_moments`` rather
than read out of git, so the comparison stays runnable once the history has
moved on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import hermite_e
from scipy import integrate, stats

from nba_prop_quant.research.game_latent_state.transmission import (
    DEFAULT_BRIDGE_ORDER,
    conditional_hermite_moments,
    transmission_coefficient_columns,
)

STATS = ("pts", "reb")


def _legacy_moments(
    cdf_lower: np.ndarray,
    cdf_upper: np.ndarray,
    order: int,
) -> list[np.ndarray]:
    """The recurrence as it stood, character for character in the loop body."""
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


def _intervals(count: int = 400, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """PIT intervals of a discrete margin: nested, with real mass between."""
    rng = np.random.default_rng(seed)
    lower = rng.uniform(1e-4, 1.0 - 1e-4, size=count)
    width = rng.uniform(1e-4, 0.4, size=count)
    upper = np.clip(lower + width, lower + 1e-6, 1.0 - 1e-9)
    return lower, upper


def _quadrature_moment(t_lower: float, t_upper: float, order: int) -> float:
    """``E[He_k(Z) | Z in (t_l, t_u]]`` by direct integration."""
    coefficients = np.zeros(order + 1)
    coefficients[order] = 1.0

    def integrand(z: float) -> float:
        return float(hermite_e.hermeval(z, coefficients) * stats.norm.pdf(z))

    numerator, _ = integrate.quad(integrand, t_lower, t_upper, limit=400)
    mass = stats.norm.cdf(t_upper) - stats.norm.cdf(t_lower)
    return float(numerator / mass)


# ----------------------------------------------------------------------
# the orders in use did not move
# ----------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_the_orders_the_bridge_uses_are_bitwise_unchanged(order: int) -> None:
    lower, upper = _intervals()
    fixed = conditional_hermite_moments(lower, upper, order=order)
    legacy = _legacy_moments(lower, upper, order=order)

    assert len(fixed) == len(legacy) == order
    for position, (new, old) in enumerate(zip(fixed, legacy)):
        # Bitwise, not approximately: the first table entry the correction
        # touches is He_2, and M_1 and M_2 read He_0 and He_1.
        assert np.array_equal(new, old), f"order {position + 1} moved"


def test_the_bridge_has_only_ever_been_run_at_order_two() -> None:
    """The bitwise claim above is only reassuring if nothing used order 3."""
    assert DEFAULT_BRIDGE_ORDER == 2


def test_the_fitted_transmission_columns_are_bitwise_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole coefficient build, not just the moment helper."""
    rng = np.random.default_rng(5)
    rows = 600
    frame = pd.DataFrame({"game_id": np.repeat(np.arange(rows // 10), 10)})
    for stat in STATS:
        lower, upper = _intervals(rows, seed=hash(stat) % 1000)
        frame[f"cdf_lower_{stat}"] = lower
        frame[f"cdf_upper_{stat}"] = upper
        frame[f"analytic_mean_{stat}"] = rng.uniform(0.5, 25.0, size=rows)
        frame[f"e_{stat}"] = rng.standard_normal(rows)

    fixed, fixed_diagnostics = transmission_coefficient_columns(
        frame, STATS, order=DEFAULT_BRIDGE_ORDER
    )

    import nba_prop_quant.research.game_latent_state.transmission as module

    monkeypatch.setattr(
        module,
        "conditional_hermite_moments",
        lambda lower, upper, order=DEFAULT_BRIDGE_ORDER: _legacy_moments(
            lower, upper, order
        ),
    )
    legacy, legacy_diagnostics = transmission_coefficient_columns(
        frame, STATS, order=DEFAULT_BRIDGE_ORDER
    )

    columns = [
        f"{name}{k}_{stat}"
        for name in "hg"
        for k in (1, DEFAULT_BRIDGE_ORDER)
        for stat in STATS
    ]
    for column in columns:
        assert np.array_equal(
            fixed[column].to_numpy(), legacy[column].to_numpy(), equal_nan=True
        ), f"{column} moved"
    assert fixed_diagnostics["by_stat"] == legacy_diagnostics["by_stat"]


# ----------------------------------------------------------------------
# order three and above is now right
# ----------------------------------------------------------------------


@pytest.mark.parametrize("order", [3, 4, 5])
def test_order_three_and_above_now_disagrees_with_the_defect(order: int) -> None:
    """If the two still agreed, the correction would not have corrected."""
    lower, upper = _intervals()
    fixed = conditional_hermite_moments(lower, upper, order=order)
    legacy = _legacy_moments(lower, upper, order=order)

    assert np.array_equal(fixed[0], legacy[0])
    assert np.array_equal(fixed[1], legacy[1])
    assert not np.allclose(fixed[2], legacy[2])


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7, 8])
def test_every_order_matches_direct_quadrature(order: int) -> None:
    """``M_k`` against the integral it is the closed form of."""
    lower, upper = _intervals(count=40, seed=3)
    moments = conditional_hermite_moments(lower, upper, order=order)
    t_lower = stats.norm.ppf(lower)
    t_upper = stats.norm.ppf(upper)

    for row in range(len(lower)):
        expected = _quadrature_moment(t_lower[row], t_upper[row], order)
        assert moments[order - 1][row] == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_the_recurrence_the_fix_encodes_is_the_hermite_recurrence() -> None:
    """``He_{k+1} = z He_k - k He_{k-1}`` against numpy's own evaluation.

    Stated on its own because the moment helper cannot be probed for a point
    value of ``He_k``: its output is an integral over an interval, and shrinking
    the interval to pin a point turns the numerator into a difference of two
    nearly equal numbers and loses every digit.

    The defective coefficient is shown to fail the same comparison, so this
    discriminates rather than merely passing.
    """
    grid = np.linspace(-3.0, 3.0, 61)
    table = [np.ones_like(grid), grid]
    legacy_table = [np.ones_like(grid), grid]
    for k in range(1, 8):
        table.append(grid * table[k] - k * table[k - 1])
        legacy_table.append(grid * legacy_table[k] - (k - 1) * legacy_table[k - 1])

    for degree in range(0, 9):
        coefficients = np.zeros(degree + 1)
        coefficients[degree] = 1.0
        expected = hermite_e.hermeval(grid, coefficients)
        assert table[degree] == pytest.approx(expected, rel=1e-9, abs=1e-9)
        if degree >= 2:
            assert not np.allclose(legacy_table[degree], expected)
