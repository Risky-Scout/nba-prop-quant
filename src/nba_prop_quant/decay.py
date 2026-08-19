from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution


DECAY_BETA_BOUNDS = (0.85, 0.999999)


@njit(cache=True)
def _decay_predictions(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    values: np.ndarray,
    beta: float,
    global_prior: float,
    prior_strength: float,
) -> np.ndarray:
    n = len(values)
    predictions = np.empty(n, dtype=np.float64)

    last_player = -1
    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        player = int(player_ids[i])

        if player != last_player:
            last_player = player
            numerator = global_prior * prior_strength
            denominator = prior_strength
        else:
            days = max(float(days_since_prev[i]), 0.0)
            decay = beta ** days
            numerator *= decay
            denominator *= decay

        predictions[i] = numerator / max(denominator, 1e-12)

        value = float(values[i])
        if np.isfinite(value):
            numerator += value
            denominator += 1.0

    return predictions


@dataclass
class DecayParams:
    beta: float
    global_prior: float
    prior_strength: float = 8.0


def decay_prior(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    values: np.ndarray,
    params: DecayParams,
) -> np.ndarray:
    return _decay_predictions(
        player_ids.astype(np.int64),
        np.nan_to_num(days_since_prev, nan=0.0).astype(np.float64),
        values.astype(np.float64),
        float(params.beta),
        float(params.global_prior),
        float(params.prior_strength),
    )


def tune_decay_beta(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    values: np.ndarray,
    history_number: np.ndarray,
    beta_bounds: tuple[float, float] = DECAY_BETA_BOUNDS,
    prior_strength: float = 8.0,
    seed: int = 73,
) -> DecayParams:
    values = values.astype(np.float64)
    finite = np.isfinite(values)
    global_prior = float(np.nanmean(values[finite]))
    score_mask = finite & (history_number >= 3)

    def objective(x: np.ndarray) -> float:
        beta = float(x[0])
        pred = _decay_predictions(
            player_ids.astype(np.int64),
            np.nan_to_num(days_since_prev, nan=0.0).astype(np.float64),
            values,
            beta,
            global_prior,
            prior_strength,
        )
        error = pred[score_mask] - values[score_mask]
        return float(np.sqrt(np.mean(error * error)))

    result = differential_evolution(
        objective,
        bounds=[beta_bounds],
        seed=seed,
        polish=True,
        updating="immediate",
        workers=1,
        maxiter=50,
        popsize=8,
        tol=1e-5,
    )
    return DecayParams(
        beta=float(result.x[0]),
        global_prior=global_prior,
        prior_strength=prior_strength,
    )
