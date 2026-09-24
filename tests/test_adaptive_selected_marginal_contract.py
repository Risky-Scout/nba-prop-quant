from __future__ import annotations

from pathlib import Path

import pandas as pd

import nba_prop_quant.adaptive_training as adaptive_training


def test_adaptive_marginal_contract_uses_selected_means_and_excludes_identity_metadata():
    source_path = Path(adaptive_training.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")

    # Regression for PR #9 failure:
    # the adaptive path must call the real marginal helper, never fall back
    # to None and thereby pass the entire metadata-containing DataFrame
    # into the ZINB numeric design matrix.
    assert "inflation_feature_columns" not in source
    assert "inflation_features_for" in source

    # Frozen downstream contract must be canonical selected means,
    # not raw XGB means for ensemble-selected REB/BLK.
    assert "oof_selected_means.parquet" in source
    assert 'f"mu_selected_{target}"' in source

    repo_root = source_path.parents[2]

    marginals = adaptive_training.load_script_module(
        repo_root,
        "scripts/07_fit_marginals.py",
    )

    frame = pd.DataFrame(
        {
            "expected_minutes": [30.0, 31.0, 32.0],
            "mu_selected_reb": [8.0, 9.0, 10.0],
            "decay_prior_reb_rate": [0.24, 0.25, 0.26],
            "kalman_prior_reb_rate": [0.23, 0.24, 0.25],
            "opp_pace_prior": [98.0, 100.0, 102.0],
            "days_rest": [1.0, 2.0, 1.0],
            "is_home": [1.0, 0.0, 1.0],
            "b2b": [0.0, 0.0, 1.0],
            "pos_G": [0.0, 0.0, 0.0],
            "pos_F": [1.0, 1.0, 1.0],
            "pos_C": [0.0, 0.0, 0.0],
            # Identity metadata that caused the original production failure.
            "player_first_name": ["Kyle", "Justin", "Ryan"],
            "player_last_name": ["Anderson", "Anderson", "Anderson"],
            "position": ["F", "F", "F"],
            "game_type": ["regular", "regular", "regular"],
            "min": ["30:00", "31:00", "32:00"],
        }
    )

    inflation = marginals.inflation_features_for(
        "reb",
        frame,
    )

    assert "mu_selected_reb" in inflation

    forbidden = {
        "player_first_name",
        "player_last_name",
        "position",
        "game_type",
        "min",
    }

    assert forbidden.isdisjoint(inflation)

    assert all(
        pd.api.types.is_numeric_dtype(frame[column].dtype)
        for column in inflation
    )
