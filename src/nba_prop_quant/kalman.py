from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution


KALMAN_Q_LOG_BOUNDS = (-16.0, 0.0)
KALMAN_R_LOG_BOUNDS = (-7.0, 6.0)


@njit(cache=True)
def _kalman_predictions(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    team_change: np.ndarray,
    values: np.ndarray,
    exposures: np.ndarray,
    global_prior: float,
    q: float,
    r: float,
    initial_variance: float,
    team_change_multiplier: float,
    epsilon: float,
) -> np.ndarray:
    n = len(values)
    predictions = np.empty(n, dtype=np.float64)

    last_player = -1
    mean_state = 0.0
    variance_state = initial_variance

    for i in range(n):
        player = int(player_ids[i])

        if player != last_player:
            last_player = player
            mean_state = np.log(max(global_prior, 0.0) + epsilon)
            variance_state = initial_variance
        else:
            days = max(float(days_since_prev[i]), 1.0)
            variance_state += q * days
            if team_change[i] > 0.5:
                variance_state *= team_change_multiplier

        predictions[i] = max(np.exp(mean_state) - epsilon, 0.0)

        value = float(values[i])
        if not np.isfinite(value):
            continue

        observation = np.log(max(value, 0.0) + epsilon)
        exposure = max(float(exposures[i]), 0.10)
        observation_variance = r / exposure

        gain = variance_state / (variance_state + observation_variance)
        mean_state = mean_state + gain * (observation - mean_state)
        variance_state = (1.0 - gain) * variance_state

    return predictions


@dataclass
class KalmanParams:
    global_prior: float
    q: float
    r: float
    initial_variance: float = 1.0
    team_change_multiplier: float = 2.0
    epsilon: float = 0.05


def kalman_prior(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    team_change: np.ndarray,
    values: np.ndarray,
    exposures: np.ndarray,
    params: KalmanParams,
) -> np.ndarray:
    return _kalman_predictions(
        player_ids.astype(np.int64),
        np.nan_to_num(days_since_prev, nan=0.0).astype(np.float64),
        np.nan_to_num(team_change, nan=0.0).astype(np.float64),
        values.astype(np.float64),
        exposures.astype(np.float64),
        float(params.global_prior),
        float(params.q),
        float(params.r),
        float(params.initial_variance),
        float(params.team_change_multiplier),
        float(params.epsilon),
    )


def tune_kalman(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    team_change: np.ndarray,
    values: np.ndarray,
    exposures: np.ndarray,
    history_number: np.ndarray,
    team_change_multiplier: float = 2.0,
    seed: int = 73,
    q_log_bounds: tuple[float, float] = KALMAN_Q_LOG_BOUNDS,
    r_log_bounds: tuple[float, float] = KALMAN_R_LOG_BOUNDS,
) -> KalmanParams:
    values = values.astype(np.float64)
    finite = np.isfinite(values)
    global_prior = float(np.nanmean(values[finite]))
    score_mask = finite & (history_number >= 3)

    def objective(x: np.ndarray) -> float:
        q = float(np.exp(x[0]))
        r = float(np.exp(x[1]))
        pred = _kalman_predictions(
            player_ids.astype(np.int64),
            np.nan_to_num(days_since_prev, nan=0.0).astype(np.float64),
            np.nan_to_num(team_change, nan=0.0).astype(np.float64),
            values,
            exposures.astype(np.float64),
            global_prior,
            q,
            r,
            1.0,
            team_change_multiplier,
            0.05,
        )
        error = pred[score_mask] - values[score_mask]
        return float(np.sqrt(np.mean(error * error)))

    result = differential_evolution(
        objective,
        bounds=[q_log_bounds, r_log_bounds],
        seed=seed,
        polish=True,
        updating="immediate",
        workers=1,
        maxiter=50,
        popsize=8,
        tol=1e-5,
    )
    return KalmanParams(
        global_prior=global_prior,
        q=float(np.exp(result.x[0])),
        r=float(np.exp(result.x[1])),
        initial_variance=1.0,
        team_change_multiplier=team_change_multiplier,
        epsilon=0.05,
    )
