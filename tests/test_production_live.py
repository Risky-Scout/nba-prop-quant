from __future__ import annotations

import numpy as np

from nba_prop_quant.production import (
    MONITORING_EDGE_GRID,
    calibrate_over_probability,
    calibrated_unconditional_probabilities,
    combine_mean_components,
)


def test_combine_mean_components_simplex() -> None:
    xgb = np.array([10.0, 20.0])
    decay = np.array([12.0, 18.0])
    kalman = np.array([8.0, 22.0])

    result = combine_mean_components(
        xgb,
        decay,
        kalman,
        {
            "xgb": 0.5,
            "decay": 0.3,
            "kalman": 0.2,
        },
    )

    expected = (
        0.5 * xgb
        + 0.3 * decay
        + 0.2 * kalman
    )

    assert np.allclose(
        result,
        expected,
    )


def test_raw_calibration_is_identity() -> None:
    policy = {
        "props": {
            "steals": {
                "selected_method": "raw",
                "production_parameters": None,
            }
        }
    }

    raw = np.array(
        [
            0.2,
            0.5,
            0.8,
        ]
    )

    calibrated, method, intercept, slope = (
        calibrate_over_probability(
            raw,
            "steals",
            policy,
        )
    )

    assert method == "raw"
    assert intercept is None
    assert slope is None
    assert np.allclose(
        calibrated,
        raw,
    )


def test_prop_calibration_matches_logit_formula() -> None:
    policy = {
        "props": {
            "points": {
                "selected_method": "prop",
                "production_parameters": {
                    "intercept": -0.1,
                    "slope": 0.5,
                },
            }
        }
    }

    raw = np.array(
        [
            0.2,
            0.5,
            0.8,
        ]
    )

    calibrated, method, intercept, slope = (
        calibrate_over_probability(
            raw,
            "points",
            policy,
        )
    )

    logits = np.log(
        raw
        / (
            1.0
            - raw
        )
    )

    expected = (
        1.0
        / (
            1.0
            + np.exp(
                -(
                    -0.1
                    + 0.5
                    * logits
                )
            )
        )
    )

    assert method == "prop"
    assert intercept == -0.1
    assert slope == 0.5
    assert np.allclose(
        calibrated,
        expected,
    )


def test_calibrated_probabilities_preserve_push_mass() -> None:
    q_over = np.array(
        [
            0.4,
            0.6,
        ]
    )

    p_push = np.array(
        [
            0.1,
            0.25,
        ]
    )

    p_over, p_under = (
        calibrated_unconditional_probabilities(
            q_over,
            p_push,
        )
    )

    assert np.allclose(
        p_over
        + p_under
        + p_push,
        1.0,
    )


def test_monitoring_grid_is_frozen_and_not_single_threshold() -> None:
    assert MONITORING_EDGE_GRID == (
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
    )
