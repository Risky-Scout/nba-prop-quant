from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from .model import ModelBundle
from .pricing import (
    american_implied_probability,
    american_to_decimal,
    fair_american,
)


TARGETS = ["pts", "reb", "ast", "stl", "blk", "fg3m"]

MONITORING_EDGE_GRID = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_verified_manifest_metadata(
    model_dir: Path,
    project_root: Path,
    allow_predeployment: bool = False,
) -> dict[str, str]:
    manifest_path = (
        model_dir
        / "frozen_manifests"
        / "LATEST.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing frozen manifest: {manifest_path}"
        )

    manifest = load_json(
        manifest_path
    )

    stage = str(
        manifest.get(
            "freeze_stage",
            "",
        )
    )

    if (
        stage
        != "external_test_deployment"
        and not allow_predeployment
    ):
        raise RuntimeError(
            "Production pricing requires an "
            "external_test_deployment freeze. "
            f"Current LATEST stage is {stage!r}. "
            "For integration smoke tests only, pass "
            "--allow-predeployment."
        )

    mismatches: list[str] = []
    missing: list[str] = []

    for records in manifest.get(
        "files",
        {},
    ).values():
        for record in records:
            relative = Path(
                record[
                    "path"
                ]
            )

            path = (
                project_root
                / relative
            )

            if not path.exists():
                missing.append(
                    str(
                        relative
                    )
                )
                continue

            actual = sha256_file(
                path
            )

            if actual != record[
                "sha256"
            ]:
                mismatches.append(
                    str(
                        relative
                    )
                )

    if missing or mismatches:
        parts = []

        if missing:
            parts.append(
                "missing="
                + ", ".join(
                    missing[
                        :10
                    ]
                )
            )

        if mismatches:
            parts.append(
                "hash_mismatch="
                + ", ".join(
                    mismatches[
                        :10
                    ]
                )
            )

        raise RuntimeError(
            "Frozen manifest verification failed: "
            + " | ".join(
                parts
            )
        )

    return {
        "freeze_id": str(
            manifest[
                "freeze_id"
            ]
        ),
        "freeze_stage": stage,
        "manifest_sha256": sha256_file(
            manifest_path
        ),
        "manifest_path": str(
            manifest_path
        ),
    }


def missing_model_features(
    frame: pd.DataFrame,
    bundle: ModelBundle,
) -> list[str]:
    return [
        feature
        for feature in bundle.feature_names
        if feature
        not in frame.columns
    ]


def assert_model_feature_contract(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    label: str,
    allow_missing: bool = False,
) -> list[str]:
    missing = missing_model_features(
        frame,
        bundle,
    )

    if missing and not allow_missing:
        raise RuntimeError(
            f"{label} is missing "
            f"{len(missing)} trained feature(s): "
            + ", ".join(
                missing[
                    :25
                ]
            )
            + (
                " ..."
                if len(
                    missing
                )
                > 25
                else ""
            )
            + ". Refuse to score an external-test slate "
            "with a broken feature contract."
        )

    return missing


def combine_mean_components(
    xgb: np.ndarray,
    decay: np.ndarray,
    kalman: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    xgb = np.asarray(
        xgb,
        dtype=float,
    )

    decay = np.asarray(
        decay,
        dtype=float,
    )

    kalman = np.asarray(
        kalman,
        dtype=float,
    )

    total_weight = float(
        weights.get(
            "xgb",
            0.0,
        )
        + weights.get(
            "decay",
            0.0,
        )
        + weights.get(
            "kalman",
            0.0,
        )
    )

    if not np.isclose(
        total_weight,
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "Mean-model production weights "
            f"must sum to 1; got {total_weight}"
        )

    selected = np.zeros_like(
        xgb,
        dtype=float,
    )

    for name, values in [
        (
            "xgb",
            xgb,
        ),
        (
            "decay",
            decay,
        ),
        (
            "kalman",
            kalman,
        ),
    ]:
        weight = float(
            weights.get(
                name,
                0.0,
            )
        )

        if weight <= 1e-12:
            continue

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Non-finite {name} component "
                "with positive production weight"
            )

        selected += (
            weight
            * values
        )

    return np.clip(
        selected,
        1e-6,
        None,
    )


def apply_selected_mean_policy(
    slate: pd.DataFrame,
    model_dir: Path,
    allow_missing_features: bool = False,
) -> tuple[
    pd.DataFrame,
    dict[str, list[str]],
]:
    out = slate.copy()

    policy_path = (
        model_dir
        / "mean_model_selection.json"
    )

    policy = load_json(
        policy_path
    )

    missing_by_target: dict[
        str,
        list[str],
    ] = {}

    expected_minutes = pd.to_numeric(
        out[
            "expected_minutes"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        expected_minutes
    ).all():
        raise RuntimeError(
            "Non-finite expected_minutes "
            "before target scoring"
        )

    for target in TARGETS:
        target_policy = policy[
            "targets"
        ][
            target
        ]

        bundle = ModelBundle.load(
            model_dir
            / f"{target}.joblib"
        )

        missing = (
            assert_model_feature_contract(
                out,
                bundle,
                label=(
                    f"{target} target model"
                ),
                allow_missing=(
                    allow_missing_features
                ),
            )
        )

        missing_by_target[
            target
        ] = missing

        xgb = bundle.predict(
            out
        )

        decay_col = (
            f"decay_prior_{target}_rate"
        )

        kalman_col = (
            f"kalman_prior_{target}_rate"
        )

        decay = (
            expected_minutes
            * pd.to_numeric(
                out[
                    decay_col
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
            if decay_col
            in out.columns
            else np.full(
                len(
                    out
                ),
                np.nan,
            )
        )

        kalman = (
            expected_minutes
            * pd.to_numeric(
                out[
                    kalman_col
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
            if kalman_col
            in out.columns
            else np.full(
                len(
                    out
                ),
                np.nan,
            )
        )

        weights = target_policy[
            "production_weights"
        ]

        selected = (
            combine_mean_components(
                xgb=xgb,
                decay=decay,
                kalman=kalman,
                weights=weights,
            )
        )

        out[
            f"mu_xgb_{target}"
        ] = xgb

        out[
            f"mu_decay_{target}"
        ] = decay

        out[
            f"mu_kalman_{target}"
        ] = kalman

        out[
            f"mu_selected_{target}_pre_usage"
        ] = selected

        out[
            f"mu_selected_{target}"
        ] = selected

        out[
            f"mean_model_mode_{target}"
        ] = str(
            target_policy[
                "selected_mode"
            ]
        )

    usage = pd.to_numeric(
        out.get(
            "availability_usage_multiplier",
            pd.Series(
                1.0,
                index=out.index,
            ),
        ),
        errors="coerce",
    ).fillna(
        1.0
    ).clip(
        lower=0.75,
        upper=1.30,
    )

    # Frozen production-only availability cascade. This is deliberately
    # downstream of the canonical mean-model selection so the stored
    # pre_usage columns retain the exact model-policy projection.
    out[
        "mu_selected_pts"
    ] *= (
        usage
        ** 0.55
    )

    out[
        "mu_selected_ast"
    ] *= (
        usage
        ** 0.45
    )

    out[
        "mu_selected_fg3m"
    ] *= (
        usage
        ** 0.40
    )

    availability_out = pd.to_numeric(
        out.get(
            "availability_out",
            pd.Series(
                0,
                index=out.index,
            ),
        ),
        errors="coerce",
    ).fillna(
        0
    ).astype(
        int
    )

    for target in TARGETS:
        out.loc[
            availability_out.eq(
                1
            ),
            f"mu_selected_{target}",
        ] = 1e-6

        # Backward-compatible alias for existing in-game/explanation code.
        # Production market pricing must use mu_selected_* explicitly.
        out[
            f"mu_{target}"
        ] = out[
            f"mu_selected_{target}"
        ]

    return (
        out,
        missing_by_target,
    )


def calibrate_over_probability(
    raw_q_over: float | np.ndarray,
    prop_type: str,
    calibration_policy: dict[
        str,
        Any,
    ],
) -> tuple[
    np.ndarray,
    str,
    float | None,
    float | None,
]:
    if prop_type not in calibration_policy[
        "props"
    ]:
        raise KeyError(
            "No frozen market-probability "
            f"calibration policy for {prop_type}"
        )

    entry = calibration_policy[
        "props"
    ][
        prop_type
    ]

    method = str(
        entry[
            "selected_method"
        ]
    )

    raw = np.clip(
        np.asarray(
            raw_q_over,
            dtype=float,
        ),
        1e-6,
        1.0
        - 1e-6,
    )

    if method == "raw":
        return (
            raw,
            method,
            None,
            None,
        )

    if method not in {
        "prop",
        "global",
    }:
        raise ValueError(
            f"Unknown calibration method: "
            f"{method}"
        )

    parameters = entry.get(
        "production_parameters"
    )

    if not parameters:
        raise ValueError(
            f"{prop_type}: calibration method "
            f"{method} has no production parameters"
        )

    intercept = float(
        parameters[
            "intercept"
        ]
    )

    slope = float(
        parameters[
            "slope"
        ]
    )

    logit = np.log(
        raw
        / (
            1.0
            - raw
        )
    )

    calibrated = expit(
        intercept
        + slope
        * logit
    )

    return (
        calibrated,
        method,
        intercept,
        slope,
    )


def calibrated_unconditional_probabilities(
    q_over: float | np.ndarray,
    p_push: float | np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    q_over = np.clip(
        np.asarray(
            q_over,
            dtype=float,
        ),
        0.0,
        1.0,
    )

    p_push = np.clip(
        np.asarray(
            p_push,
            dtype=float,
        ),
        0.0,
        1.0,
    )

    nonpush_mass = (
        1.0
        - p_push
    )

    p_over = (
        nonpush_mass
        * q_over
    )

    p_under = (
        nonpush_mass
        * (
            1.0
            - q_over
        )
    )

    return (
        p_over,
        p_under,
    )


def add_market_probability_layer(
    frame: pd.DataFrame,
    calibration_policy: dict[
        str,
        Any,
    ],
) -> pd.DataFrame:
    out = frame.copy()

    required = {
        "prop_type",
        "p_over",
        "p_under",
        "p_push",
        "q_over_nonpush",
        "q_under_nonpush",
        "over_odds",
        "under_odds",
    }

    missing = required - set(
        out.columns
    )

    if missing:
        raise ValueError(
            "Market probability layer missing "
            f"columns: {sorted(missing)}"
        )

    out[
        "raw_p_over"
    ] = pd.to_numeric(
        out[
            "p_over"
        ],
        errors="coerce",
    )

    out[
        "raw_p_under"
    ] = pd.to_numeric(
        out[
            "p_under"
        ],
        errors="coerce",
    )

    out[
        "raw_p_push"
    ] = pd.to_numeric(
        out[
            "p_push"
        ],
        errors="coerce",
    )

    out[
        "raw_q_over_nonpush"
    ] = pd.to_numeric(
        out[
            "q_over_nonpush"
        ],
        errors="coerce",
    )

    out[
        "raw_q_under_nonpush"
    ] = pd.to_numeric(
        out[
            "q_under_nonpush"
        ],
        errors="coerce",
    )

    out[
        "calibrated_q_over_nonpush"
    ] = np.nan

    out[
        "calibration_method"
    ] = ""

    out[
        "calibration_intercept"
    ] = np.nan

    out[
        "calibration_slope"
    ] = np.nan

    for prop_type in sorted(
        out[
            "prop_type"
        ].dropna().unique()
    ):
        mask = out[
            "prop_type"
        ].eq(
            prop_type
        )

        calibrated, method, intercept, slope = (
            calibrate_over_probability(
                out.loc[
                    mask,
                    "raw_q_over_nonpush",
                ].to_numpy(
                    dtype=float
                ),
                str(
                    prop_type
                ),
                calibration_policy,
            )
        )

        out.loc[
            mask,
            "calibrated_q_over_nonpush",
        ] = calibrated

        out.loc[
            mask,
            "calibration_method",
        ] = method

        if intercept is not None:
            out.loc[
                mask,
                "calibration_intercept",
            ] = intercept

        if slope is not None:
            out.loc[
                mask,
                "calibration_slope",
            ] = slope

    out[
        "base_calibrated_q_over_nonpush"
    ] = out[
        "calibrated_q_over_nonpush"
    ]

    out[
        "base_calibration_method"
    ] = out[
        "calibration_method"
    ]

    out[
        "base_calibration_intercept"
    ] = out[
        "calibration_intercept"
    ]

    out[
        "base_calibration_slope"
    ] = out[
        "calibration_slope"
    ]

    out[
        "gate3_candidate_applied"
    ] = False

    if (
        "gate3_candidate_q_over_nonpush"
        in out.columns
    ):
        candidate = pd.to_numeric(
            out[
                "gate3_candidate_q_over_nonpush"
            ],
            errors="coerce",
        )

        override = candidate.notna()

        if override.any():
            values = candidate.loc[
                override
            ]

            if (
                (values < 0.0).any()
                or (values > 1.0).any()
                or not np.isfinite(
                    values.to_numpy(
                        dtype=float
                    )
                ).all()
            ):
                raise RuntimeError(
                    "Invalid Gate 3 candidate probability override"
                )

            out.loc[
                override,
                "calibrated_q_over_nonpush",
            ] = values

            out.loc[
                override,
                "gate3_candidate_applied",
            ] = True

            if (
                "gate3_candidate_method"
                in out.columns
            ):
                out.loc[
                    override,
                    "calibration_method",
                ] = out.loc[
                    override,
                    "gate3_candidate_method",
                ]

            if (
                "gate3_candidate_intercept"
                in out.columns
            ):
                intercept_values = pd.to_numeric(
                    out.loc[
                        override,
                        "gate3_candidate_intercept",
                    ],
                    errors="coerce",
                )

                valid = (
                    intercept_values.notna()
                )

                if valid.any():
                    target_index = (
                        intercept_values.loc[
                            valid
                        ].index
                    )

                    out.loc[
                        target_index,
                        "calibration_intercept",
                    ] = (
                        intercept_values.loc[
                            valid
                        ]
                    )

            if (
                "gate3_candidate_slope"
                in out.columns
            ):
                slope_values = pd.to_numeric(
                    out.loc[
                        override,
                        "gate3_candidate_slope",
                    ],
                    errors="coerce",
                )

                valid = (
                    slope_values.notna()
                )

                if valid.any():
                    target_index = (
                        slope_values.loc[
                            valid
                        ].index
                    )

                    out.loc[
                        target_index,
                        "calibration_slope",
                    ] = (
                        slope_values.loc[
                            valid
                        ]
                    )

    out[
        "calibrated_q_under_nonpush"
    ] = (
        1.0
        - out[
            "calibrated_q_over_nonpush"
        ]
    )

    calibrated_p_over, calibrated_p_under = (
        calibrated_unconditional_probabilities(
            out[
                "calibrated_q_over_nonpush"
            ].to_numpy(
                dtype=float
            ),
            out[
                "raw_p_push"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    out[
        "calibrated_p_over"
    ] = calibrated_p_over

    out[
        "calibrated_p_under"
    ] = calibrated_p_under

    out[
        "calibrated_p_push"
    ] = out[
        "raw_p_push"
    ]

    over_implied = np.array(
        [
            american_implied_probability(
                value
            )
            for value in out[
                "over_odds"
            ].to_numpy()
        ],
        dtype=float,
    )

    under_implied = np.array(
        [
            american_implied_probability(
                value
            )
            for value in out[
                "under_odds"
            ].to_numpy()
        ],
        dtype=float,
    )

    probability_sum = (
        over_implied
        + under_implied
    )

    out[
        "market_raw_implied_over"
    ] = over_implied

    out[
        "market_raw_implied_under"
    ] = under_implied

    out[
        "market_hold_pct"
    ] = (
        probability_sum
        - 1.0
    ) * 100.0

    out[
        "market_devig_q_over"
    ] = (
        over_implied
        / probability_sum
    )

    out[
        "market_devig_q_under"
    ] = (
        under_implied
        / probability_sum
    )

    out[
        "raw_edge_over"
    ] = (
        out[
            "raw_q_over_nonpush"
        ]
        - out[
            "market_devig_q_over"
        ]
    )

    out[
        "raw_edge_under"
    ] = (
        out[
            "raw_q_under_nonpush"
        ]
        - out[
            "market_devig_q_under"
        ]
    )

    out[
        "calibrated_edge_over"
    ] = (
        out[
            "calibrated_q_over_nonpush"
        ]
        - out[
            "market_devig_q_over"
        ]
    )

    out[
        "calibrated_edge_under"
    ] = (
        out[
            "calibrated_q_under_nonpush"
        ]
        - out[
            "market_devig_q_under"
        ]
    )

    over_decimal = np.array(
        [
            american_to_decimal(
                value
            )
            for value in out[
                "over_odds"
            ].to_numpy()
        ],
        dtype=float,
    )

    under_decimal = np.array(
        [
            american_to_decimal(
                value
            )
            for value in out[
                "under_odds"
            ].to_numpy()
        ],
        dtype=float,
    )

    out[
        "calibrated_ev_over"
    ] = (
        out[
            "calibrated_p_over"
        ]
        * (
            over_decimal
            - 1.0
        )
        - out[
            "calibrated_p_under"
        ]
    )

    out[
        "calibrated_ev_under"
    ] = (
        out[
            "calibrated_p_under"
        ]
        * (
            under_decimal
            - 1.0
        )
        - out[
            "calibrated_p_over"
        ]
    )

    choose_over = (
        out[
            "calibrated_ev_over"
        ]
        >= out[
            "calibrated_ev_under"
        ]
    )

    out[
        "model_preferred_side"
    ] = np.where(
        choose_over,
        "over",
        "under",
    )

    out[
        "model_preferred_edge"
    ] = np.where(
        choose_over,
        out[
            "calibrated_edge_over"
        ],
        out[
            "calibrated_edge_under"
        ],
    )

    out[
        "model_preferred_ev"
    ] = np.where(
        choose_over,
        out[
            "calibrated_ev_over"
        ],
        out[
            "calibrated_ev_under"
        ],
    )

    out[
        "calibrated_fair_over_american"
    ] = [
        fair_american(
            value
        )
        for value in out[
            "calibrated_q_over_nonpush"
        ].to_numpy(
            dtype=float
        )
    ]

    out[
        "calibrated_fair_under_american"
    ] = [
        fair_american(
            value
        )
        for value in out[
            "calibrated_q_under_nonpush"
        ].to_numpy(
            dtype=float
        )
    ]

    for threshold in MONITORING_EDGE_GRID:
        label = (
            str(
                threshold
            )
            .replace(
                ".",
                "_",
            )
        )

        out[
            f"monitor_edge_ge_{label}"
        ] = (
            out[
                "model_preferred_edge"
            ]
            >= threshold
        )

    out[
        "auto_bet"
    ] = False

    out[
        "betting_threshold_policy"
    ] = (
        "monitor_only_no_threshold_frozen"
    )

    return out


def single_stat_quantile_summary(
    row: pd.Series,
    marginals: dict,
    mu_columns: dict[
        str,
        str,
    ],
) -> dict[
    str,
    float,
]:
    result: dict[
        str,
        float,
    ] = {}

    for target in TARGETS:
        mu = float(
            row[
                mu_columns[
                    target
                ]
            ]
        )

        quantiles = marginals[
            target
        ].ppf(
            np.array(
                [
                    0.10,
                    0.50,
                    0.90,
                ],
                dtype=float,
            ),
            mu,
            row,
        )

        result[
            f"{target}_mean"
        ] = mu

        result[
            f"{target}_p10"
        ] = float(
            quantiles[
                0
            ]
        )

        result[
            f"{target}_p50"
        ] = float(
            quantiles[
                1
            ]
        )

        result[
            f"{target}_p90"
        ] = float(
            quantiles[
                2
            ]
        )

    return result
