from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


PRICED_PATH = Path(
    "data/processed/market_backtest/backtest_priced_2025.parquet"
)
GAMES_PATH = Path(
    "data/raw/seasons/season=2025/games.parquet"
)
OUT_DIR = Path(
    "data/processed/market_backtest/calibration_walkforward"
)
MODEL_PATH = Path(
    "models/market_probability_calibration.json"
)

N_BLOCKS = 5
EPS = 1e-6
L2 = 1e-4


def logit_probability(
    probability: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        np.asarray(probability, dtype=float),
        EPS,
        1.0 - EPS,
    )
    return np.log(
        p / (1.0 - p)
    )


def binary_log_loss(
    probability: np.ndarray,
    outcome: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        np.asarray(probability, dtype=float),
        EPS,
        1.0 - EPS,
    )
    y = np.asarray(
        outcome,
        dtype=float,
    )

    return -(
        y * np.log(p)
        + (1.0 - y) * np.log(1.0 - p)
    )


def event_equal_weights(
    frame: pd.DataFrame,
) -> np.ndarray:
    counts = (
        frame.groupby(
            [
                "game_id",
                "player_id",
                "prop_type",
            ]
        )[
            "line_value"
        ]
        .transform("size")
        .to_numpy(dtype=float)
    )

    return 1.0 / np.clip(
        counts,
        1.0,
        None,
    )


def fit_platt(
    probability: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    x = logit_probability(
        probability
    )
    y = np.asarray(
        outcome,
        dtype=float,
    )
    w = np.asarray(
        weights,
        dtype=float,
    )

    w = w / np.mean(w)

    def objective(
        theta: np.ndarray,
    ) -> float:
        intercept = float(
            theta[0]
        )
        slope = float(
            theta[1]
        )

        calibrated = expit(
            intercept
            + slope * x
        )

        loss = binary_log_loss(
            calibrated,
            y,
        )

        penalty = L2 * (
            intercept**2
            + (
                slope - 1.0
            ) ** 2
        )

        return float(
            np.average(
                loss,
                weights=w,
            )
            + penalty
        )

    result = minimize(
        objective,
        x0=np.array(
            [
                0.0,
                1.0,
            ],
            dtype=float,
        ),
        method="L-BFGS-B",
        bounds=[
            (
                -3.0,
                3.0,
            ),
            (
                0.05,
                2.0,
            ),
        ],
    )

    if not result.success:
        raise RuntimeError(
            "Platt calibration failed: "
            f"{result.message}"
        )

    return {
        "intercept": float(
            result.x[0]
        ),
        "slope": float(
            result.x[1]
        ),
    }


def apply_platt(
    probability: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    x = logit_probability(
        probability
    )

    return expit(
        float(
            parameters[
                "intercept"
            ]
        )
        + float(
            parameters[
                "slope"
            ]
        )
        * x
    )


def build_contract_frame(
    priced: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    nonpush = priced.loc[
        ~priced[
            "actual_push"
        ]
    ].copy()

    contract = (
        nonpush.groupby(
            [
                "game_id",
                "player_id",
                "prop_type",
                "line_value",
            ],
            as_index=False,
        )
        .agg(
            actual=(
                "actual",
                "first",
            ),
            q_model=(
                "q_over_nonpush",
                "first",
            ),
            q_market=(
                "market_q_over",
                "mean",
            ),
            vendors=(
                "vendor",
                "nunique",
            ),
        )
    )

    contract[
        "actual_over"
    ] = (
        contract[
            "actual"
        ]
        > contract[
            "line_value"
        ]
    ).astype(float)

    game_dates = (
        games[
            [
                "id",
                "date",
            ]
        ]
        .rename(
            columns={
                "id": "game_id",
                "date": "game_date",
            }
        )
        .drop_duplicates(
            "game_id"
        )
    )

    game_dates[
        "game_date"
    ] = pd.to_datetime(
        game_dates[
            "game_date"
        ],
        errors="raise",
    ).dt.normalize()

    contract = contract.merge(
        game_dates,
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    if contract[
        "game_date"
    ].isna().any():
        raise RuntimeError(
            "Missing game date after games merge"
        )

    return contract


def make_chronological_blocks(
    contract: pd.DataFrame,
) -> list[np.ndarray]:
    date_order = np.sort(
        contract[
            "game_date"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    date_blocks = [
        np.asarray(block)
        for block in np.array_split(
            date_order,
            N_BLOCKS,
        )
        if len(block) > 0
    ]

    game_blocks = []

    for date_block in date_blocks:
        game_ids = (
            contract.loc[
                contract[
                    "game_date"
                ].isin(date_block),
                [
                    "game_id",
                    "game_date",
                ],
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "game_date",
                    "game_id",
                ]
            )[
                "game_id"
            ]
            .to_numpy()
        )

        game_blocks.append(
            np.asarray(game_ids)
        )

    return game_blocks


def add_losses(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    y = out[
        "actual_over"
    ].to_numpy(
        dtype=float
    )

    for label, column in [
        (
            "raw_model",
            "q_model",
        ),
        (
            "global_platt",
            "q_global_platt",
        ),
        (
            "prop_platt",
            "q_prop_platt",
        ),
        (
            "market",
            "q_market",
        ),
    ]:
        p = out[
            column
        ].to_numpy(
            dtype=float
        )

        out[
            f"brier_{label}"
        ] = (
            p - y
        ) ** 2

        out[
            f"logloss_{label}"
        ] = binary_log_loss(
            p,
            y,
        )

    return out


def event_level_summary(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    loss_columns = [
        column
        for column in scored.columns
        if column.startswith(
            "brier_"
        )
        or column.startswith(
            "logloss_"
        )
    ]

    event = (
        scored.groupby(
            [
                "game_id",
                "player_id",
                "prop_type",
            ],
            as_index=False,
        )[
            loss_columns
        ]
        .mean()
    )

    rows = []

    scopes = [
        (
            "ALL",
            event,
        )
    ]

    scopes.extend(
        (
            prop_type,
            event[
                event[
                    "prop_type"
                ].eq(
                    prop_type
                )
            ],
        )
        for prop_type in sorted(
            event[
                "prop_type"
            ].unique()
        )
    )

    for label, group in scopes:
        row = {
            "label": label,
            "events": int(
                len(
                    group
                )
            ),
        }

        for column in loss_columns:
            row[
                column
            ] = float(
                group[
                    column
                ].mean()
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    for path in [
        PRICED_PATH,
        GAMES_PATH,
    ]:
        if not path.exists():
            raise SystemExit(
                f"ERROR: missing required file: "
                f"{path}"
            )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    priced = pd.read_parquet(
        PRICED_PATH
    )

    games = pd.read_parquet(
        GAMES_PATH
    )

    contract = build_contract_frame(
        priced,
        games,
    )

    blocks = make_chronological_blocks(
        contract
    )

    print("=" * 124)
    print(
        "2025 CHRONOLOGICAL MARKET-PROBABILITY CALIBRATION"
    )
    print("=" * 124)
    print(
        f"Non-push contracts: "
        f"{len(contract):,}"
    )
    print(
        f"Unique games: "
        f"{contract['game_id'].nunique():,}"
    )
    print(
        "Design: first chronological DATE block seeds training; "
        "each later block is calibrated only from strictly earlier dates."
    )
    print(
        "Fit weights: equal total weight per "
        "game/player/prop event."
    )
    print(
        "2025 remains development data. "
        "This is internal walk-forward calibration validation."
    )

    fold_records = []
    oof_parts = []
    parameter_records = []

    for fold_index in range(
        1,
        len(
            blocks
        ),
    ):
        train_games = np.concatenate(
            blocks[
                :fold_index
            ]
        )

        validation_games = blocks[
            fold_index
        ]

        train = contract[
            contract[
                "game_id"
            ].isin(
                train_games
            )
        ].copy()

        validation = contract[
            contract[
                "game_id"
            ].isin(
                validation_games
            )
        ].copy()

        global_parameters = fit_platt(
            train[
                "q_model"
            ].to_numpy(
                dtype=float
            ),
            train[
                "actual_over"
            ].to_numpy(
                dtype=float
            ),
            event_equal_weights(
                train
            ),
        )

        validation[
            "q_global_platt"
        ] = apply_platt(
            validation[
                "q_model"
            ].to_numpy(
                dtype=float
            ),
            global_parameters,
        )

        validation[
            "q_prop_platt"
        ] = np.nan

        parameter_records.append(
            {
                "fold": fold_index,
                "scope": "GLOBAL",
                "train_games": int(
                    len(
                        np.unique(
                            train_games
                        )
                    )
                ),
                **global_parameters,
            }
        )

        for prop_type in sorted(
            contract[
                "prop_type"
            ].unique()
        ):
            train_prop = train[
                train[
                    "prop_type"
                ].eq(
                    prop_type
                )
            ].copy()

            validation_mask = (
                validation[
                    "prop_type"
                ].eq(
                    prop_type
                )
            )

            if (
                len(
                    train_prop
                )
                < 100
                or train_prop[
                    "actual_over"
                ].nunique()
                < 2
            ):
                prop_parameters = (
                    global_parameters
                )
            else:
                prop_parameters = fit_platt(
                    train_prop[
                        "q_model"
                    ].to_numpy(
                        dtype=float
                    ),
                    train_prop[
                        "actual_over"
                    ].to_numpy(
                        dtype=float
                    ),
                    event_equal_weights(
                        train_prop
                    ),
                )

            validation.loc[
                validation_mask,
                "q_prop_platt",
            ] = apply_platt(
                validation.loc[
                    validation_mask,
                    "q_model",
                ].to_numpy(
                    dtype=float
                ),
                prop_parameters,
            )

            parameter_records.append(
                {
                    "fold": fold_index,
                    "scope": prop_type,
                    "train_games": int(
                        train_prop[
                            "game_id"
                        ].nunique()
                    ),
                    **prop_parameters,
                }
            )

        if validation[
            "q_prop_platt"
        ].isna().any():
            raise RuntimeError(
                "Missing prop-specific calibrated probability"
            )

        validation[
            "fold"
        ] = fold_index

        scored = add_losses(
            validation
        )

        fold_summary = event_level_summary(
            scored
        )

        fold_summary[
            "fold"
        ] = fold_index

        fold_records.append(
            fold_summary
        )

        oof_parts.append(
            scored
        )

        all_row = fold_summary[
            fold_summary[
                "label"
            ].eq(
                "ALL"
            )
        ].iloc[
            0
        ]

        train_max_date = train["game_date"].max()
        validation_min_date = validation["game_date"].min()
        validation_max_date = validation["game_date"].max()

        if not train_max_date < validation_min_date:
            raise RuntimeError(
                "Chronological date leakage detected: "
                f"train_max_date={train_max_date}, "
                f"validation_min_date={validation_min_date}"
            )

        print(
            f"\nFold {fold_index}: "
            f"train_games={len(np.unique(train_games)):,}, "
            f"validation_games={len(validation_games):,}, "
            f"contracts={len(validation):,}"
        )

        print(
            f"  dates: train through {train_max_date.date()} | "
            f"validate {validation_min_date.date()} "
            f"through {validation_max_date.date()}"
        )

        print(
            "  event log loss: "
            f"raw={all_row['logloss_raw_model']:.6f} | "
            f"global={all_row['logloss_global_platt']:.6f} | "
            f"prop={all_row['logloss_prop_platt']:.6f} | "
            f"market={all_row['logloss_market']:.6f}"
        )

        print(
            "  event Brier:    "
            f"raw={all_row['brier_raw_model']:.6f} | "
            f"global={all_row['brier_global_platt']:.6f} | "
            f"prop={all_row['brier_prop_platt']:.6f} | "
            f"market={all_row['brier_market']:.6f}"
        )

    oof = pd.concat(
        oof_parts,
        ignore_index=True,
    )

    pooled = event_level_summary(
        oof
    )

    folds = pd.concat(
        fold_records,
        ignore_index=True,
    )

    parameters = pd.DataFrame(
        parameter_records
    )

    print("\n" + "=" * 124)
    print(
        "POOLED WALK-FORWARD EVENT-LEVEL RESULTS"
    )
    print("=" * 124)

    display_columns = [
        "label",
        "events",
        "brier_raw_model",
        "brier_global_platt",
        "brier_prop_platt",
        "brier_market",
        "logloss_raw_model",
        "logloss_global_platt",
        "logloss_prop_platt",
        "logloss_market",
    ]

    print(
        pooled[
            display_columns
        ].to_string(
            index=False,
            formatters={
                column: (
                    lambda x:
                    f"{x:.6f}"
                )
                for column in display_columns
                if column
                not in {
                    "label",
                    "events",
                }
            },
        )
    )

    print("\n" + "=" * 124)
    print(
        "CALIBRATION PARAMETER STABILITY"
    )
    print("=" * 124)

    parameter_summary = (
        parameters.groupby(
            "scope",
            as_index=False,
        )
        .agg(
            folds=(
                "fold",
                "nunique",
            ),
            intercept_mean=(
                "intercept",
                "mean",
            ),
            intercept_min=(
                "intercept",
                "min",
            ),
            intercept_max=(
                "intercept",
                "max",
            ),
            slope_mean=(
                "slope",
                "mean",
            ),
            slope_min=(
                "slope",
                "min",
            ),
            slope_max=(
                "slope",
                "max",
            ),
        )
    )

    print(
        parameter_summary.to_string(
            index=False,
            formatters={
                "intercept_mean": (
                    lambda x:
                    f"{x:+.4f}"
                ),
                "intercept_min": (
                    lambda x:
                    f"{x:+.4f}"
                ),
                "intercept_max": (
                    lambda x:
                    f"{x:+.4f}"
                ),
                "slope_mean": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "slope_min": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "slope_max": (
                    lambda x:
                    f"{x:.4f}"
                ),
            },
        )
    )

    # ----------------------------------------------------------
    # Production calibration fit on all 2025 development data.
    # Save both global and per-prop calibrators; do not decide which
    # to deploy until the walk-forward results are reviewed.
    # ----------------------------------------------------------
    production = {
        "note": (
            "Fit on all 2025 development market contracts. "
            "2026-27 is the first external evaluation."
        ),
        "global": fit_platt(
            contract[
                "q_model"
            ].to_numpy(
                dtype=float
            ),
            contract[
                "actual_over"
            ].to_numpy(
                dtype=float
            ),
            event_equal_weights(
                contract
            ),
        ),
        "by_prop": {},
    }

    for prop_type in sorted(
        contract[
            "prop_type"
        ].unique()
    ):
        prop = contract[
            contract[
                "prop_type"
            ].eq(
                prop_type
            )
        ].copy()

        production[
            "by_prop"
        ][
            prop_type
        ] = fit_platt(
            prop[
                "q_model"
            ].to_numpy(
                dtype=float
            ),
            prop[
                "actual_over"
            ].to_numpy(
                dtype=float
            ),
            event_equal_weights(
                prop
            ),
        )

    with MODEL_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            production,
            handle,
            indent=2,
            sort_keys=True,
        )

    oof.to_parquet(
        OUT_DIR
        / "oof_calibrated_contracts.parquet",
        index=False,
    )

    folds.to_csv(
        OUT_DIR
        / "fold_event_metrics.csv",
        index=False,
    )

    pooled.to_csv(
        OUT_DIR
        / "pooled_event_metrics.csv",
        index=False,
    )

    parameters.to_csv(
        OUT_DIR
        / "walkforward_parameters.csv",
        index=False,
    )

    parameter_summary.to_csv(
        OUT_DIR
        / "parameter_stability.csv",
        index=False,
    )

    print("\n" + "=" * 124)
    print("FINAL")
    print("=" * 124)

    print(
        f"Saved walk-forward calibration audit: "
        f"{OUT_DIR}"
    )

    print(
        f"Saved production candidate calibrators: "
        f"{MODEL_PATH}"
    )

    print(
        "Do not wire calibration into sportsbook pricing yet. "
        "First compare raw vs global Platt vs per-prop Platt "
        "against the de-vigged market."
    )


if __name__ == "__main__":
    main()
