from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_prop_quant.pricing import american_to_decimal


PRICED_PATH = Path(
    "data/processed/market_backtest/backtest_priced_2025.parquet"
)

OOF_CAL_PATH = Path(
    "data/processed/market_backtest/calibration_walkforward/"
    "oof_calibrated_contracts.parquet"
)

POLICY_PATH = Path(
    "models/market_probability_calibration_policy.json"
)

OUT_DIR = Path(
    "data/processed/market_backtest/calibrated_oof"
)

PRIMARY_EDGE = 0.03
BOOTSTRAP_REPS = 10_000
SEED = 73


def binary_log_loss(
    probability: np.ndarray,
    outcome: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        np.asarray(probability, dtype=float),
        1e-9,
        1.0 - 1e-9,
    )
    y = np.asarray(outcome, dtype=float)

    return -(
        y * np.log(p)
        + (1.0 - y) * np.log(1.0 - p)
    )


def select_policy_probability(
    contracts: pd.DataFrame,
    policy: dict,
) -> pd.DataFrame:
    out = contracts.copy()

    selected_probability = np.empty(
        len(out),
        dtype=float,
    )
    selected_method = np.empty(
        len(out),
        dtype=object,
    )

    for prop_type in sorted(
        out["prop_type"].unique()
    ):
        if prop_type not in policy["props"]:
            raise RuntimeError(
                f"Calibration policy missing prop: {prop_type}"
            )

        method = policy["props"][
            prop_type
        ]["selected_method"]

        mask = out[
            "prop_type"
        ].eq(prop_type)

        if method == "raw":
            column = "q_model"
        elif method == "global":
            column = "q_global_platt"
        elif method == "prop":
            column = "q_prop_platt"
        else:
            raise RuntimeError(
                f"Unknown calibration method "
                f"{method!r} for {prop_type}"
            )

        selected_probability[
            mask.to_numpy()
        ] = out.loc[
            mask,
            column,
        ].to_numpy(dtype=float)

        selected_method[
            mask.to_numpy()
        ] = method

    out["q_selected"] = selected_probability
    out["selected_method"] = selected_method

    if not np.isfinite(
        out["q_selected"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite selected calibrated probabilities"
        )

    return out


def add_contract_scores(
    contracts: pd.DataFrame,
) -> pd.DataFrame:
    out = contracts.copy()

    y = out[
        "actual_over"
    ].to_numpy(dtype=float)

    probability_columns = {
        "raw": "q_model",
        "selected": "q_selected",
        "market": "q_market",
    }

    for label, column in probability_columns.items():
        p = out[
            column
        ].to_numpy(dtype=float)

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


def event_scores(
    contracts: pd.DataFrame,
) -> pd.DataFrame:
    score_columns = [
        "brier_raw",
        "brier_selected",
        "brier_market",
        "logloss_raw",
        "logloss_selected",
        "logloss_market",
    ]

    return (
        contracts.groupby(
            [
                "game_id",
                "player_id",
                "prop_type",
            ],
            as_index=False,
        )[
            score_columns
        ]
        .mean()
    )


def summarize_event_scores(
    event: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    scopes = [("ALL", event)]
    scopes.extend(
        (
            prop_type,
            event[
                event[
                    "prop_type"
                ].eq(prop_type)
            ],
        )
        for prop_type in sorted(
            event[
                "prop_type"
            ].unique()
        )
    )

    for label, group in scopes:
        rows.append(
            {
                "label": label,
                "events": int(len(group)),
                "brier_raw": float(
                    group["brier_raw"].mean()
                ),
                "brier_selected": float(
                    group["brier_selected"].mean()
                ),
                "brier_market": float(
                    group["brier_market"].mean()
                ),
                "logloss_raw": float(
                    group["logloss_raw"].mean()
                ),
                "logloss_selected": float(
                    group["logloss_selected"].mean()
                ),
                "logloss_market": float(
                    group["logloss_market"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def cluster_bootstrap_metric_delta(
    event: pd.DataFrame,
    selected_column: str,
    benchmark_column: str,
    reps: int,
    seed: int,
) -> dict[str, float]:
    work = event[
        [
            "game_id",
            selected_column,
            benchmark_column,
        ]
    ].copy()

    work["delta"] = (
        work[selected_column]
        - work[benchmark_column]
    )

    by_game = (
        work.groupby(
            "game_id",
            as_index=False,
        )
        .agg(
            delta_sum=("delta", "sum"),
            event_count=("delta", "size"),
        )
    )

    sums = by_game[
        "delta_sum"
    ].to_numpy(dtype=float)
    counts = by_game[
        "event_count"
    ].to_numpy(dtype=float)

    n_games = len(by_game)
    rng = np.random.default_rng(seed)

    boot = np.empty(
        reps,
        dtype=float,
    )

    for rep in range(reps):
        sampled = rng.integers(
            0,
            n_games,
            size=n_games,
        )

        boot[rep] = (
            sums[sampled].sum()
            / counts[sampled].sum()
        )

    low, high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return {
        "delta": float(
            work["delta"].mean()
        ),
        "ci_low": float(low),
        "ci_high": float(high),
        "prob_selected_better": float(
            np.mean(boot < 0.0)
        ),
    }


def build_cluster_bootstrap_table(
    event: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    scopes = [("ALL", event)]
    scopes.extend(
        (
            prop_type,
            event[
                event[
                    "prop_type"
                ].eq(prop_type)
            ],
        )
        for prop_type in sorted(
            event["prop_type"].unique()
        )
    )

    for offset, (label, group) in enumerate(scopes):
        brier_raw = cluster_bootstrap_metric_delta(
            group,
            "brier_selected",
            "brier_raw",
            BOOTSTRAP_REPS,
            SEED + 1009 * offset,
        )

        brier_market = cluster_bootstrap_metric_delta(
            group,
            "brier_selected",
            "brier_market",
            BOOTSTRAP_REPS,
            SEED + 1009 * offset + 100_003,
        )

        logloss_raw = cluster_bootstrap_metric_delta(
            group,
            "logloss_selected",
            "logloss_raw",
            BOOTSTRAP_REPS,
            SEED + 1009 * offset + 200_003,
        )

        logloss_market = cluster_bootstrap_metric_delta(
            group,
            "logloss_selected",
            "logloss_market",
            BOOTSTRAP_REPS,
            SEED + 1009 * offset + 300_007,
        )

        rows.append(
            {
                "label": label,
                "brier_vs_raw_delta": brier_raw["delta"],
                "brier_vs_raw_ci_low": brier_raw["ci_low"],
                "brier_vs_raw_ci_high": brier_raw["ci_high"],
                "brier_vs_raw_prob_selected_better": (
                    brier_raw[
                        "prob_selected_better"
                    ]
                ),
                "brier_vs_market_delta": brier_market["delta"],
                "brier_vs_market_ci_low": brier_market["ci_low"],
                "brier_vs_market_ci_high": brier_market["ci_high"],
                "brier_vs_market_prob_selected_better": (
                    brier_market[
                        "prob_selected_better"
                    ]
                ),
                "logloss_vs_raw_delta": logloss_raw["delta"],
                "logloss_vs_raw_ci_low": logloss_raw["ci_low"],
                "logloss_vs_raw_ci_high": logloss_raw["ci_high"],
                "logloss_vs_raw_prob_selected_better": (
                    logloss_raw[
                        "prob_selected_better"
                    ]
                ),
                "logloss_vs_market_delta": logloss_market["delta"],
                "logloss_vs_market_ci_low": logloss_market["ci_low"],
                "logloss_vs_market_ci_high": logloss_market["ci_high"],
                "logloss_vs_market_prob_selected_better": (
                    logloss_market[
                        "prob_selected_better"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def add_quote_calibrated_evaluation(
    priced: pd.DataFrame,
    contract_selected: pd.DataFrame,
) -> pd.DataFrame:
    lookup = contract_selected[
        [
            "game_id",
            "player_id",
            "prop_type",
            "line_value",
            "fold",
            "game_date",
            "q_selected",
            "selected_method",
        ]
    ].copy()

    merged = priced.merge(
        lookup,
        on=[
            "game_id",
            "player_id",
            "prop_type",
            "line_value",
        ],
        how="inner",
        validate="many_to_one",
    )

    if merged.empty:
        raise RuntimeError(
            "No quote rows overlap the walk-forward "
            "calibrated contracts."
        )

    merged[
        "q_selected_under"
    ] = (
        1.0
        - merged[
            "q_selected"
        ]
    )

    nonpush_mass = (
        1.0
        - merged[
            "p_push"
        ].to_numpy(dtype=float)
    )

    merged[
        "p_selected_over"
    ] = (
        nonpush_mass
        * merged[
            "q_selected"
        ].to_numpy(dtype=float)
    )

    merged[
        "p_selected_under"
    ] = (
        nonpush_mass
        * merged[
            "q_selected_under"
        ].to_numpy(dtype=float)
    )

    over_decimal = merged[
        "over_odds"
    ].map(
        american_to_decimal
    ).to_numpy(dtype=float)

    under_decimal = merged[
        "under_odds"
    ].map(
        american_to_decimal
    ).to_numpy(dtype=float)

    merged[
        "selected_ev_over"
    ] = (
        merged[
            "p_selected_over"
        ].to_numpy(dtype=float)
        * (
            over_decimal - 1.0
        )
        - merged[
            "p_selected_under"
        ].to_numpy(dtype=float)
    )

    merged[
        "selected_ev_under"
    ] = (
        merged[
            "p_selected_under"
        ].to_numpy(dtype=float)
        * (
            under_decimal - 1.0
        )
        - merged[
            "p_selected_over"
        ].to_numpy(dtype=float)
    )

    merged[
        "selected_edge_over"
    ] = (
        merged[
            "q_selected"
        ]
        - merged[
            "market_q_over"
        ]
    )

    merged[
        "selected_edge_under"
    ] = (
        merged[
            "q_selected_under"
        ]
        - merged[
            "market_q_under"
        ]
    )

    choose_over = (
        merged[
            "selected_ev_over"
        ]
        >= merged[
            "selected_ev_under"
        ]
    )

    merged[
        "selected_bet_side"
    ] = np.where(
        choose_over,
        "over",
        "under",
    )

    merged[
        "selected_bet_ev"
    ] = np.where(
        choose_over,
        merged[
            "selected_ev_over"
        ],
        merged[
            "selected_ev_under"
        ],
    )

    merged[
        "selected_bet_edge"
    ] = np.where(
        choose_over,
        merged[
            "selected_edge_over"
        ],
        merged[
            "selected_edge_under"
        ],
    )

    return merged


def grade_profit(
    row: pd.Series,
) -> float:
    if bool(row["actual_push"]):
        return 0.0

    if row["selected_bet_side"] == "over":
        won = bool(
            row["actual"]
            > row["line_value"]
        )
        decimal = american_to_decimal(
            row["over_odds"]
        )
    else:
        won = bool(
            row["actual"]
            < row["line_value"]
        )
        decimal = american_to_decimal(
            row["under_odds"]
        )

    return float(
        decimal - 1.0
        if won
        else -1.0
    )


def select_bets(
    quotes: pd.DataFrame,
    min_edge: float,
    mode: str,
) -> pd.DataFrame:
    bets = quotes[
        quotes[
            "selected_bet_ev"
        ].gt(0.0)
        & quotes[
            "selected_bet_edge"
        ].ge(min_edge)
    ].copy()

    if bets.empty:
        return bets

    bets = bets.sort_values(
        [
            "selected_bet_ev",
            "selected_bet_edge",
        ],
        ascending=False,
    )

    if mode == "all_quotes":
        pass

    elif mode == "best_vendor_event":
        bets = bets.drop_duplicates(
            subset=[
                "game_id",
                "player_id",
                "prop_type",
                "vendor",
            ],
            keep="first",
        )

    elif mode == "best_event":
        bets = bets.drop_duplicates(
            subset=[
                "game_id",
                "player_id",
                "prop_type",
            ],
            keep="first",
        )

    else:
        raise KeyError(mode)

    bets = bets.copy()

    bets[
        "profit_units"
    ] = bets.apply(
        grade_profit,
        axis=1,
    )

    return bets


def betting_summary(
    quotes: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
    ]

    modes = [
        "all_quotes",
        "best_vendor_event",
        "best_event",
    ]

    rows = []

    for threshold in thresholds:
        for mode in modes:
            bets = select_bets(
                quotes,
                threshold,
                mode,
            )

            rows.append(
                {
                    "threshold": threshold,
                    "mode": mode,
                    "bets": int(len(bets)),
                    "games": int(
                        bets["game_id"].nunique()
                    )
                    if not bets.empty
                    else 0,
                    "profit_units": float(
                        bets["profit_units"].sum()
                    )
                    if not bets.empty
                    else 0.0,
                    "roi": float(
                        bets["profit_units"].mean()
                    )
                    if not bets.empty
                    else np.nan,
                    "average_edge": float(
                        bets[
                            "selected_bet_edge"
                        ].mean()
                    )
                    if not bets.empty
                    else np.nan,
                    "average_ev": float(
                        bets[
                            "selected_bet_ev"
                        ].mean()
                    )
                    if not bets.empty
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def cluster_bootstrap_roi(
    bets: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if bets.empty:
        return {
            "roi": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "prob_roi_positive": np.nan,
        }

    by_game = (
        bets.groupby(
            "game_id",
            as_index=False,
        )
        .agg(
            profit_sum=("profit_units", "sum"),
            bet_count=("profit_units", "size"),
        )
    )

    profits = by_game[
        "profit_sum"
    ].to_numpy(dtype=float)
    counts = by_game[
        "bet_count"
    ].to_numpy(dtype=float)

    n_games = len(by_game)
    rng = np.random.default_rng(seed)

    boot = np.empty(
        reps,
        dtype=float,
    )

    for rep in range(reps):
        sampled = rng.integers(
            0,
            n_games,
            size=n_games,
        )

        boot[rep] = (
            profits[sampled].sum()
            / counts[sampled].sum()
        )

    low, high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return {
        "roi": float(
            bets["profit_units"].mean()
        ),
        "ci_low": float(low),
        "ci_high": float(high),
        "prob_roi_positive": float(
            np.mean(boot > 0.0)
        ),
    }


def main() -> None:
    for path in [
        PRICED_PATH,
        OOF_CAL_PATH,
        POLICY_PATH,
    ]:
        if not path.exists():
            raise SystemExit(
                f"ERROR: missing required file: {path}"
            )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    priced = pd.read_parquet(
        PRICED_PATH
    )

    contracts = pd.read_parquet(
        OOF_CAL_PATH
    )

    with POLICY_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        policy = json.load(handle)

    selected = select_policy_probability(
        contracts,
        policy,
    )

    selected = add_contract_scores(
        selected
    )

    event = event_scores(
        selected
    )

    summary = summarize_event_scores(
        event
    )

    bootstrap = build_cluster_bootstrap_table(
        event
    )

    quotes = add_quote_calibrated_evaluation(
        priced,
        selected,
    )

    betting = betting_summary(
        quotes
    )

    print("=" * 128)
    print(
        "STRICT-DATE OOF CALIBRATED MARKET EVALUATION"
    )
    print("=" * 128)

    print(
        f"Walk-forward contracts: "
        f"{len(selected):,}"
    )
    print(
        f"Walk-forward events:    "
        f"{len(event):,}"
    )
    print(
        f"Matched quote rows:     "
        f"{len(quotes):,}"
    )
    print(
        f"Games represented:      "
        f"{quotes['game_id'].nunique():,}"
    )

    print("\nCalibration methods:")
    print(
        selected[
            [
                "prop_type",
                "selected_method",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "prop_type"
        )
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 128)
    print(
        "EVENT-LEVEL PROPER SCORING"
    )
    print("=" * 128)

    print(
        summary.to_string(
            index=False,
            formatters={
                column: (
                    lambda x:
                    f"{x:.6f}"
                )
                for column in [
                    "brier_raw",
                    "brier_selected",
                    "brier_market",
                    "logloss_raw",
                    "logloss_selected",
                    "logloss_market",
                ]
            },
        )
    )

    print("\n" + "=" * 128)
    print(
        "GAME-CLUSTERED UNCERTAINTY — SELECTED CALIBRATION"
    )
    print("=" * 128)

    print(
        bootstrap.to_string(
            index=False,
            formatters={
                column: (
                    lambda x:
                    f"{x:+.6f}"
                )
                for column in bootstrap.columns
                if column.endswith(
                    "delta"
                )
                or column.endswith(
                    "ci_low"
                )
                or column.endswith(
                    "ci_high"
                )
            }
            | {
                column: (
                    lambda x:
                    f"{x:.3f}"
                )
                for column in bootstrap.columns
                if "prob_selected_better"
                in column
            },
        )
    )

    print("\n" + "=" * 128)
    print(
        "OOF-CALIBRATED BETTING SENSITIVITY"
    )
    print("=" * 128)

    print(
        betting.to_string(
            index=False,
            formatters={
                "threshold": (
                    lambda x:
                    f"{x:.3f}"
                ),
                "roi": (
                    lambda x:
                    f"{x:+.3%}"
                ),
                "profit_units": (
                    lambda x:
                    f"{x:+.2f}"
                ),
                "average_edge": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "average_ev": (
                    lambda x:
                    f"{x:.4f}"
                ),
            },
        )
    )

    print("\n" + "=" * 128)
    print(
        "3% EDGE GAME-CLUSTER ROI"
    )
    print("=" * 128)

    roi_rows = []

    for offset, mode in enumerate(
        [
            "best_vendor_event",
            "best_event",
        ]
    ):
        bets = select_bets(
            quotes,
            PRIMARY_EDGE,
            mode,
        )

        result = cluster_bootstrap_roi(
            bets,
            BOOTSTRAP_REPS,
            SEED
            + 1009
            * offset,
        )

        roi_rows.append(
            {
                "mode": mode,
                "bets": int(len(bets)),
                "games": int(
                    bets["game_id"].nunique()
                )
                if not bets.empty
                else 0,
                **result,
            }
        )

    roi_table = pd.DataFrame(
        roi_rows
    )

    print(
        roi_table.to_string(
            index=False,
            formatters={
                "roi": (
                    lambda x:
                    f"{x:+.3%}"
                ),
                "ci_low": (
                    lambda x:
                    f"{x:+.3%}"
                ),
                "ci_high": (
                    lambda x:
                    f"{x:+.3%}"
                ),
                "prob_roi_positive": (
                    lambda x:
                    f"{x:.3f}"
                ),
            },
        )
    )

    selected.to_parquet(
        OUT_DIR
        / "selected_oof_contracts.parquet",
        index=False,
    )

    event.to_parquet(
        OUT_DIR
        / "selected_oof_event_scores.parquet",
        index=False,
    )

    quotes.to_parquet(
        OUT_DIR
        / "selected_oof_quote_rows.parquet",
        index=False,
    )

    summary.to_csv(
        OUT_DIR
        / "event_scoring_summary.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUT_DIR
        / "event_game_cluster_bootstrap.csv",
        index=False,
    )

    betting.to_csv(
        OUT_DIR
        / "betting_sensitivity.csv",
        index=False,
    )

    roi_table.to_csv(
        OUT_DIR
        / "roi_3pct_game_cluster_bootstrap.csv",
        index=False,
    )

    print("\n" + "=" * 128)
    print("FINAL")
    print("=" * 128)

    print(
        f"Saved: {OUT_DIR}"
    )

    print(
        "This is the valid 2025 development betting diagnostic for "
        "the selected calibration policy because probabilities are "
        "walk-forward out-of-fold within the available market period."
    )

    print(
        "Do not tune the edge threshold from these results. "
        "2026-27 remains the first external market test."
    )


if __name__ == "__main__":
    main()
