import numpy as np

from nba_prop_quant.decay import DecayParams, decay_prior
from nba_prop_quant.kalman import KalmanParams, kalman_prior


def test_decay_prior_is_strictly_pregame():
    player_ids = np.array([1, 1, 1], dtype=np.int64)
    days = np.array([0.0, 1.0, 1.0])
    values = np.array([10.0, 20.0, 30.0])
    params = DecayParams(beta=1.0, global_prior=5.0, prior_strength=1.0)

    pred = decay_prior(player_ids, days, values, params)

    assert np.allclose(pred, [5.0, 7.5, 35.0 / 3.0])


def test_kalman_prior_does_not_use_current_observation():
    player_ids = np.array([1, 1], dtype=np.int64)
    days = np.array([0.0, 1.0])
    team_change = np.array([0.0, 0.0])
    values_a = np.array([2.0, 100.0])
    values_b = np.array([2.0, 1.0])
    exposure = np.ones(2)

    params = KalmanParams(global_prior=1.0, q=0.01, r=0.5)
    pred_a = kalman_prior(
        player_ids, days, team_change, values_a, exposure, params
    )
    pred_b = kalman_prior(
        player_ids, days, team_change, values_b, exposure, params
    )

    assert pred_a[1] == pred_b[1]
