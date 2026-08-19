from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FOLD_PATH = Path(
    "data/processed/market_backtest/calibration_walkforward/"
    "fold_event_metrics.csv"
)

POOLED_PATH = Path(
    "data/processed/market_backtest/calibration_walkforward/"
    "pooled_event_metrics.csv"
)

CALIBRATOR_PATH = Path(
    "models/market_probability_calibration.json"
)

POLICY_PATH = Path(
    "models/market_probability_calibration_policy.json"
)

AUDIT_PATH = Path(
    "data/processed/market_backtest/calibration_walkforward/"
    "calibration_selection_audit.csv"
)

METHOD_COLUMNS = {
    "raw": {
        "brier": "brier_raw_model",
        "logloss": "logloss_raw_model",
        "complexity": 0,
    },
    "global": {
        "brier": "brier_global_platt",
        "logloss": "logloss_global_platt",
        "complexity": 1,
    },
    "prop": {
        "brier": "brier_prop_platt",
        "logloss": "logloss_prop_platt",
        "complexity": 2,
    },
}

# Conservative calibration-selection gates.
MIN_LOGLOSS_IMPROVEMENT_VS_RAW_PCT = 0.10
MIN_LOGLOSS_FOLD_WINS_VS_RAW = 3
MIN_BRIER_IMPROVEMENT_VS_RAW_PCT = 0.00
MIN_BRIER_FOLD_WINS_VS_RAW = 2

# Extra complexity gate for prop-specific calibration if global calibration
# is already accepted.
MIN_PROP_LOGLOSS_IMPROVEMENT_VS_GLOBAL_PCT = 0.05
MIN_PROP_LOGLOSS_FOLD_WINS_VS_GLOBAL = 2


def improvement_pct(
    challenger: float,
    incumbent: float,
) -> float:
    return (
        100.0
        * (
            incumbent
            - challenger
        )
        / incumbent
    )


def fold_wins(
    folds: pd.DataFrame,
    challenger_column: str,
    incumbent_column: str,
) -> int:
    return int(
        (
            folds[
                challenger_column
            ]
            < folds[
                incumbent_column
            ]
        ).sum()
    )


def candidate_vs_raw(
    pooled_row: pd.Series,
    fold_rows: pd.DataFrame,
    method: str,
) -> dict:
    if method == "raw":
        return {
            "method": "raw",
            "accepted_vs_raw": True,
            "logloss_improvement_vs_raw_pct": 0.0,
            "logloss_fold_wins_vs_raw": int(
                fold_rows[
                    "fold"
                ].nunique()
            ),
            "brier_improvement_vs_raw_pct": 0.0,
            "brier_fold_wins_vs_raw": int(
                fold_rows[
                    "fold"
                ].nunique()
            ),
        }

    method_log = METHOD_COLUMNS[
        method
    ][
        "logloss"
    ]
    method_brier = METHOD_COLUMNS[
        method
    ][
        "brier"
    ]

    raw_log = METHOD_COLUMNS[
        "raw"
    ][
        "logloss"
    ]
    raw_brier = METHOD_COLUMNS[
        "raw"
    ][
        "brier"
    ]

    log_improvement = improvement_pct(
        float(
            pooled_row[
                method_log
            ]
        ),
        float(
            pooled_row[
                raw_log
            ]
        ),
    )

    brier_improvement = improvement_pct(
        float(
            pooled_row[
                method_brier
            ]
        ),
        float(
            pooled_row[
                raw_brier
            ]
        ),
    )

    log_wins = fold_wins(
        fold_rows,
        method_log,
        raw_log,
    )

    brier_wins = fold_wins(
        fold_rows,
        method_brier,
        raw_brier,
    )

    accepted = bool(
        log_improvement
        >= MIN_LOGLOSS_IMPROVEMENT_VS_RAW_PCT
        and log_wins
        >= MIN_LOGLOSS_FOLD_WINS_VS_RAW
        and brier_improvement
        >= MIN_BRIER_IMPROVEMENT_VS_RAW_PCT
        and brier_wins
        >= MIN_BRIER_FOLD_WINS_VS_RAW
    )

    return {
        "method": method,
        "accepted_vs_raw": accepted,
        "logloss_improvement_vs_raw_pct": float(
            log_improvement
        ),
        "logloss_fold_wins_vs_raw": int(
            log_wins
        ),
        "brier_improvement_vs_raw_pct": float(
            brier_improvement
        ),
        "brier_fold_wins_vs_raw": int(
            brier_wins
        ),
    }


def prop_beats_global_gate(
    pooled_row: pd.Series,
    fold_rows: pd.DataFrame,
) -> dict:
    prop_log = METHOD_COLUMNS[
        "prop"
    ][
        "logloss"
    ]
    global_log = METHOD_COLUMNS[
        "global"
    ][
        "logloss"
    ]

    prop_brier = METHOD_COLUMNS[
        "prop"
    ][
        "brier"
    ]
    global_brier = METHOD_COLUMNS[
        "global"
    ][
        "brier"
    ]

    log_improvement = improvement_pct(
        float(
            pooled_row[
                prop_log
            ]
        ),
        float(
            pooled_row[
                global_log
            ]
        ),
    )

    brier_improvement = improvement_pct(
        float(
            pooled_row[
                prop_brier
            ]
        ),
        float(
            pooled_row[
                global_brier
            ]
        ),
    )

    log_wins = fold_wins(
        fold_rows,
        prop_log,
        global_log,
    )

    brier_wins = fold_wins(
        fold_rows,
        prop_brier,
        global_brier,
    )

    accepted = bool(
        log_improvement
        >= MIN_PROP_LOGLOSS_IMPROVEMENT_VS_GLOBAL_PCT
        and log_wins
        >= MIN_PROP_LOGLOSS_FOLD_WINS_VS_GLOBAL
        and brier_improvement
        >= 0.0
        and brier_wins
        >= 2
    )

    return {
        "prop_vs_global_accepted": accepted,
        "prop_logloss_improvement_vs_global_pct": float(
            log_improvement
        ),
        "prop_logloss_fold_wins_vs_global": int(
            log_wins
        ),
        "prop_brier_improvement_vs_global_pct": float(
            brier_improvement
        ),
        "prop_brier_fold_wins_vs_global": int(
            brier_wins
        ),
    }


def main() -> None:
    for path in [
        FOLD_PATH,
        POOLED_PATH,
        CALIBRATOR_PATH,
    ]:
        if not path.exists():
            raise SystemExit(
                f"ERROR: missing required file: {path}"
            )

    folds = pd.read_csv(
        FOLD_PATH
    )

    pooled = pd.read_csv(
        POOLED_PATH
    )

    with CALIBRATOR_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        calibrators = json.load(
            handle
        )

    prop_types = sorted(
        label
        for label in pooled[
            "label"
        ].astype(str)
        .unique()
        if label != "ALL"
    )

    audit_rows = []
    policy = {
        "selection_basis": (
            "2025 strict-date walk-forward event-level proper scoring; "
            "market benchmark is reported but not used to choose among "
            "raw/global/prop model calibrators."
        ),
        "gates": {
            "min_logloss_improvement_vs_raw_pct": (
                MIN_LOGLOSS_IMPROVEMENT_VS_RAW_PCT
            ),
            "min_logloss_fold_wins_vs_raw": (
                MIN_LOGLOSS_FOLD_WINS_VS_RAW
            ),
            "min_brier_improvement_vs_raw_pct": (
                MIN_BRIER_IMPROVEMENT_VS_RAW_PCT
            ),
            "min_brier_fold_wins_vs_raw": (
                MIN_BRIER_FOLD_WINS_VS_RAW
            ),
            "min_prop_logloss_improvement_vs_global_pct": (
                MIN_PROP_LOGLOSS_IMPROVEMENT_VS_GLOBAL_PCT
            ),
            "min_prop_logloss_fold_wins_vs_global": (
                MIN_PROP_LOGLOSS_FOLD_WINS_VS_GLOBAL
            ),
        },
        "props": {},
    }

    print("=" * 128)
    print(
        "STRICT-DATE WALK-FORWARD PROBABILITY-CALIBRATION SELECTION"
    )
    print("=" * 128)
    print(
        "Primary score: event-level log loss."
    )
    print(
        "Brier score is a guardrail. Market probabilities are benchmark-only."
    )
    print()

    for prop_type in prop_types:
        pooled_row = pooled[
            pooled[
                "label"
            ].astype(str)
            .eq(
                prop_type
            )
        ].iloc[
            0
        ]

        fold_rows = folds[
            folds[
                "label"
            ].astype(str)
            .eq(
                prop_type
            )
        ].copy()

        raw_result = candidate_vs_raw(
            pooled_row,
            fold_rows,
            "raw",
        )

        global_result = candidate_vs_raw(
            pooled_row,
            fold_rows,
            "global",
        )

        prop_result = candidate_vs_raw(
            pooled_row,
            fold_rows,
            "prop",
        )

        prop_vs_global = prop_beats_global_gate(
            pooled_row,
            fold_rows,
        )

        selected = "raw"

        if global_result[
            "accepted_vs_raw"
        ]:
            selected = "global"

        if prop_result[
            "accepted_vs_raw"
        ]:
            if selected == "raw":
                selected = "prop"
            elif prop_vs_global[
                "prop_vs_global_accepted"
            ]:
                selected = "prop"

        selected_logloss = float(
            pooled_row[
                METHOD_COLUMNS[
                    selected
                ][
                    "logloss"
                ]
            ]
        )

        selected_brier = float(
            pooled_row[
                METHOD_COLUMNS[
                    selected
                ][
                    "brier"
                ]
            ]
        )

        market_logloss = float(
            pooled_row[
                "logloss_market"
            ]
        )

        market_brier = float(
            pooled_row[
                "brier_market"
            ]
        )

        if selected == "raw":
            parameters = None
        elif selected == "global":
            parameters = calibrators[
                "global"
            ]
        else:
            parameters = calibrators[
                "by_prop"
            ][
                prop_type
            ]

        policy[
            "props"
        ][
            prop_type
        ] = {
            "selected_method": selected,
            "production_parameters": parameters,
            "pooled_event_logloss": selected_logloss,
            "pooled_event_brier": selected_brier,
            "market_benchmark_logloss": market_logloss,
            "market_benchmark_brier": market_brier,
        }

        row = {
            "prop_type": prop_type,
            "selected_method": selected,
            "raw_logloss": float(
                pooled_row[
                    "logloss_raw_model"
                ]
            ),
            "global_logloss": float(
                pooled_row[
                    "logloss_global_platt"
                ]
            ),
            "prop_logloss": float(
                pooled_row[
                    "logloss_prop_platt"
                ]
            ),
            "market_logloss": market_logloss,
            "raw_brier": float(
                pooled_row[
                    "brier_raw_model"
                ]
            ),
            "global_brier": float(
                pooled_row[
                    "brier_global_platt"
                ]
            ),
            "prop_brier": float(
                pooled_row[
                    "brier_prop_platt"
                ]
            ),
            "market_brier": market_brier,
            **{
                key: value
                for key, value
                in global_result.items()
                if key != "method"
            },
            **{
                (
                    "prop_"
                    + key
                ): value
                for key, value
                in prop_result.items()
                if key
                not in {
                    "method",
                }
            },
            **prop_vs_global,
        }

        audit_rows.append(
            row
        )

        print(
            f"{prop_type:28s} -> "
            f"{selected.upper():6s} | "
            f"raw={row['raw_logloss']:.6f} "
            f"global={row['global_logloss']:.6f} "
            f"prop={row['prop_logloss']:.6f} "
            f"market={row['market_logloss']:.6f}"
        )

        if selected != "raw":
            chosen = (
                global_result
                if selected == "global"
                else prop_result
            )

            print(
                "  vs raw: "
                f"logloss improvement="
                f"{chosen['logloss_improvement_vs_raw_pct']:+.3f}% | "
                f"logloss wins="
                f"{chosen['logloss_fold_wins_vs_raw']}/"
                f"{fold_rows['fold'].nunique()} | "
                f"Brier improvement="
                f"{chosen['brier_improvement_vs_raw_pct']:+.3f}% | "
                f"Brier wins="
                f"{chosen['brier_fold_wins_vs_raw']}/"
                f"{fold_rows['fold'].nunique()}"
            )

    audit = pd.DataFrame(
        audit_rows
    )

    AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    with POLICY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            policy,
            handle,
            indent=2,
            sort_keys=True,
        )

    print()
    print("=" * 128)
    print("FINAL CALIBRATION POLICY")
    print("=" * 128)

    print(
        audit[
            [
                "prop_type",
                "selected_method",
                "raw_logloss",
                "global_logloss",
                "prop_logloss",
                "market_logloss",
                "raw_brier",
                "global_brier",
                "prop_brier",
                "market_brier",
            ]
        ].to_string(
            index=False,
            formatters={
                column: (
                    lambda x:
                    f"{x:.6f}"
                )
                for column in [
                    "raw_logloss",
                    "global_logloss",
                    "prop_logloss",
                    "market_logloss",
                    "raw_brier",
                    "global_brier",
                    "prop_brier",
                    "market_brier",
                ]
            },
        )
    )

    print()
    print(
        f"Saved policy: {POLICY_PATH}"
    )
    print(
        f"Saved audit:  {AUDIT_PATH}"
    )
    print()
    print(
        "Do not apply the policy to betting thresholds yet. "
        "Review which props retained RAW probabilities and which "
        "earned calibrated probabilities."
    )


if __name__ == "__main__":
    main()
