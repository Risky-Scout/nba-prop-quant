from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson


def pure_birth_generator(
    intensity_per_minute: float,
    max_increment: int = 80,
) -> np.ndarray:
    """
    Generator Q for a truncated Poisson pure-birth CTMC.

    State i means "i additional events from now." For i < max_increment,
    Q[i, i+1] = lambda and Q[i, i] = -lambda. The final state is absorbing
    and collects the truncated upper tail.
    """
    lam = max(float(intensity_per_minute), 0.0)
    q = np.zeros((max_increment + 1, max_increment + 1), dtype=float)

    for i in range(max_increment):
        q[i, i] = -lam
        q[i, i + 1] = lam

    return q


def transition_distribution_matrix(
    intensity_per_minute: float,
    remaining_minutes: float,
    max_increment: int = 80,
) -> np.ndarray:
    q = pure_birth_generator(
        intensity_per_minute=intensity_per_minute,
        max_increment=max_increment,
    )
    transition = expm(q * max(float(remaining_minutes), 0.0))
    return transition[0]


def transition_distribution_poisson(
    intensity_per_minute: float,
    remaining_minutes: float,
    max_increment: int = 80,
) -> np.ndarray:
    """
    Closed-form equivalent of the pure-birth CTMC above.

    The final bin collects P(N >= max_increment), avoiding lost mass.
    """
    mean = max(float(intensity_per_minute), 0.0) * max(
        float(remaining_minutes), 0.0
    )
    support = np.arange(max_increment, dtype=int)
    pmf = poisson.pmf(support, mean)
    tail = max(1.0 - float(pmf.sum()), 0.0)
    return np.concatenate([pmf, np.array([tail])])


def ctmc_over_under_push(
    current_count: int,
    line: float,
    expected_remaining_count: float,
) -> tuple[float, float, float]:
    """
    Price a live line under the pure-birth CTMC.

    expected_remaining_count is lambda * remaining_time. This lets the caller
    supply a baseline or INGARCH-adjusted integrated intensity.
    """
    current = int(current_count)
    mean = max(float(expected_remaining_count), 0.0)
    floor_needed = int(np.floor(float(line) - current))

    if float(line).is_integer():
        exact_needed = int(line) - current
        if exact_needed < 0:
            return 1.0, 0.0, 0.0

        push = float(poisson.pmf(exact_needed, mean))
        under = (
            0.0
            if exact_needed <= 0
            else float(poisson.cdf(exact_needed - 1, mean))
        )
        over = float(1.0 - poisson.cdf(exact_needed, mean))
        return over, under, push

    # Half-point line: no push.
    if floor_needed < 0:
        return 1.0, 0.0, 0.0

    under = float(poisson.cdf(floor_needed, mean))
    over = 1.0 - under
    return over, under, 0.0
