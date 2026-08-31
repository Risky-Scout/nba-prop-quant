from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_prop_quant.gate3_v2 import (
    apply_gate3_role_state,
    build_current_role_features,
    prepare_gate3_candidate_probability_overrides,
)
from nba_prop_quant.production import (
    add_market_probability_layer,
)


class StarterResidualModel:
    def predict(
        self,
        frame,
    ):
        starter = pd.to_numeric(
            frame[
                "starter"
            ],
            errors="coerce",
        ).fillna(
            0
        ).to_numpy(
            dtype=float
        )

        return np.where(
            starter == 1,
            1.5,
            -1.0,
        )


def test_role_features_use_seed_history():
    current = pd.DataFrame(
        {
            "game_id": [100] * 5,
            "player_id": [
                1,
                2,
                3,
                4,
                5,
            ],
            "team_id": [10] * 5,
            "starter": [
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )

    seed = {
        "players": {
            "1": {
                "starter_history": [
                    0,
                    0,
                    1,
                    1,
                ]
            }
        },
        "teams": {
            "10": {
                "last_starters": [
                    1,
                    2,
                    3,
                    6,
                    7,
                ]
            }
        },
    }

    prior = pd.DataFrame(
        columns=[
            "game_id",
            "player_id",
            "team_id",
            "starter",
            "_date",
            "_captured_at",
        ]
    )

    role = (
        build_current_role_features(
            current,
            prior,
            seed,
        )
    )

    row = role.loc[
        role[
            "player_id"
        ].eq(1)
    ].iloc[0]

    assert (
        row[
            "gate3_role_ready"
        ]
        == 1
    )

    assert (
        row[
            "prev_starter"
        ]
        == 1
    )

    assert np.isclose(
        row[
            "starter_rate5"
        ],
        0.5,
    )

    assert (
        row[
            "team_starter_overlap"
        ]
        == 3
    )

    assert (
        row[
            "team_new_starters"
        ]
        == 2
    )

    assert (
        row[
            "team_lost_starters"
        ]
        == 2
    )


def test_apply_role_state_builds_ast_and_combo_deltas(
    tmp_path,
):
    slate = pd.DataFrame(
        {
            "game_id": [100],
            "player_id": [1],
            "expected_minutes": [30.0],
            "mu_selected_pts": [20.0],
            "mu_selected_reb": [6.0],
            "mu_selected_ast": [5.0],
            "availability_out": [0],
        }
    )

    current = pd.DataFrame(
        {
            "game_id": [100] * 5,
            "player_id": [
                1,
                2,
                3,
                4,
                5,
            ],
            "team_id": [10] * 5,
            "starter": [1] * 5,
        }
    )

    runtime = {
        "role_state_seed": {
            "players": {
                "1": {
                    "starter_history": [
                        1,
                        1,
                    ]
                }
            },
            "teams": {
                "10": {
                    "last_starters": [
                        1,
                        2,
                        3,
                        4,
                        5,
                    ]
                }
            },
        },
        "role_model_payload": {
            "model": (
                StarterResidualModel()
            ),
            "feature_names": [
                "expected_minutes",
                "starter",
            ],
        },
        "candidate_id": "candidate",
        "gate3_lock_commit": "lock",
        "deployment_manifest_sha256": (
            "manifest-sha"
        ),
    }

    result = apply_gate3_role_state(
        slate,
        current,
        target_date="2026-10-20",
        snapshot_dir=tmp_path,
        runtime=runtime,
    )

    assert (
        result.loc[
            0,
            "gate3_role_ready",
        ]
        == 1
    )

    assert np.isclose(
        result.loc[
            0,
            "gate3_role_minutes",
        ],
        31.5,
    )

    ratio = (
        31.5
        / 30.0
    )

    assert np.isclose(
        result.loc[
            0,
            "gate3_mu_ast",
        ],
        5.0 * ratio,
    )

    expected_pa_delta = (
        20.0
        * (
            ratio - 1.0
        )
        + 5.0
        * (
            ratio - 1.0
        )
    )

    assert np.isclose(
        result.loc[
            0,
            "gate3_delta_points_assists",
        ],
        expected_pa_delta,
    )


def test_candidate_probability_formulas_exact():
    frame = pd.DataFrame(
        {
            "prop_type": [
                "assists",
                "points_assists",
                "points_rebounds",
            ],
            "q_over_nonpush": [
                0.60,
                0.55,
                0.48,
            ],
            "gate3_role_ready": [
                1,
                1,
                1,
            ],
            "gate3_delta_points_assists": [
                np.nan,
                1.25,
                np.nan,
            ],
            "gate3_delta_points_rebounds": [
                np.nan,
                np.nan,
                -0.75,
            ],
        }
    )

    policy = {
        "props": {
            "assists": {
                "selected_method": "prop",
                "production_parameters": {
                    "intercept": -0.2,
                    "slope": 0.5,
                },
            },
            "points_assists": {
                "selected_method": "prop",
                "production_parameters": {
                    "intercept": 0.1,
                    "slope": 0.4,
                },
            },
            "points_rebounds": {
                "selected_method": "raw",
                "production_parameters": None,
            },
        }
    }

    params = {
        "assists": {
            "intercept": -0.03,
            "slope": 0.46,
        },
        "points_assists": {
            "gamma": 0.22,
            "standardization_mean": 0.03,
            "standardization_std": 1.67,
        },
        "points_rebounds": {
            "gamma": 0.24,
            "standardization_mean": 0.05,
            "standardization_std": 1.80,
        },
    }

    result = (
        prepare_gate3_candidate_probability_overrides(
            frame,
            calibration_policy=policy,
            probability_parameters=params,
        )
    )

    def sigmoid(x):
        return 1.0 / (
            1.0
            + np.exp(-x)
        )

    def logit(p):
        return np.log(
            p / (1-p)
        )

    expected_ast = sigmoid(
        -0.03
        + 0.46
        * logit(
            0.60
        )
    )

    base_pa = sigmoid(
        0.1
        + 0.4
        * logit(
            0.55
        )
    )

    expected_pa = sigmoid(
        logit(
            base_pa
        )
        + 0.22
        * (
            (
                1.25
                - 0.03
            )
            / 1.67
        )
    )

    expected_pr = sigmoid(
        logit(
            0.48
        )
        + 0.24
        * (
            (
                -0.75
                - 0.05
            )
            / 1.80
        )
    )

    assert np.isclose(
        result.loc[
            0,
            "gate3_candidate_q_over_nonpush",
        ],
        expected_ast,
    )

    assert np.isclose(
        result.loc[
            1,
            "gate3_candidate_q_over_nonpush",
        ],
        expected_pa,
    )

    assert np.isclose(
        result.loc[
            2,
            "gate3_candidate_q_over_nonpush",
        ],
        expected_pr,
    )


def test_market_layer_applies_override_and_preserves_push():
    frame = pd.DataFrame(
        {
            "prop_type": [
                "assists"
            ],
            "p_over": [
                0.36
            ],
            "p_under": [
                0.54
            ],
            "p_push": [
                0.10
            ],
            "q_over_nonpush": [
                0.40
            ],
            "q_under_nonpush": [
                0.60
            ],
            "over_odds": [
                -110
            ],
            "under_odds": [
                -110
            ],
            "gate3_candidate_q_over_nonpush": [
                0.70
            ],
            "gate3_candidate_method": [
                "v2_role_shock_calibrated"
            ],
            "gate3_candidate_intercept": [
                -0.03
            ],
            "gate3_candidate_slope": [
                0.46
            ],
        }
    )

    policy = {
        "props": {
            "assists": {
                "selected_method": "raw",
                "production_parameters": None,
            }
        }
    }

    result = (
        add_market_probability_layer(
            frame,
            calibration_policy=policy,
        )
    )

    assert bool(
        result.loc[
            0,
            "gate3_candidate_applied",
        ]
    )

    assert np.isclose(
        result.loc[
            0,
            "calibrated_q_over_nonpush",
        ],
        0.70,
    )

    assert np.isclose(
        result.loc[
            0,
            "calibrated_p_over",
        ],
        0.63,
    )

    assert np.isclose(
        result.loc[
            0,
            "calibrated_p_under",
        ],
        0.27,
    )

    assert np.isclose(
        result.loc[
            0,
            "calibrated_p_push",
        ],
        0.10,
    )

    assert np.isclose(
        result.loc[
            0,
            "calibrated_p_over",
        ]
        + result.loc[
            0,
            "calibrated_p_under",
        ]
        + result.loc[
            0,
            "calibrated_p_push",
        ],
        1.0,
    )


def test_nan_override_preserves_frozen_probability():
    base = pd.DataFrame(
        {
            "prop_type": [
                "points"
            ],
            "p_over": [
                0.45
            ],
            "p_under": [
                0.55
            ],
            "p_push": [
                0.0
            ],
            "q_over_nonpush": [
                0.45
            ],
            "q_under_nonpush": [
                0.55
            ],
            "over_odds": [
                -110
            ],
            "under_odds": [
                -110
            ],
        }
    )

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

    frozen = (
        add_market_probability_layer(
            base,
            calibration_policy=policy,
        )
    )

    with_override = base.copy()

    with_override[
        "gate3_candidate_q_over_nonpush"
    ] = np.nan

    with_override[
        "gate3_candidate_method"
    ] = "frozen_selected_v1"

    candidate = (
        add_market_probability_layer(
            with_override,
            calibration_policy=policy,
        )
    )

    for column in [
        "calibrated_q_over_nonpush",
        "calibrated_q_under_nonpush",
        "calibrated_p_over",
        "calibrated_p_under",
        "calibrated_p_push",
        "calibrated_edge_over",
        "calibrated_edge_under",
        "calibrated_ev_over",
        "calibrated_ev_under",
    ]:
        assert np.allclose(
            frozen[
                column
            ],
            candidate[
                column
            ],
        )
