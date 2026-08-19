from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_prop_quant.pricing import american_to_decimal


PRICED_PATH = Path(
    "data/processed/market_backtest/backtest_priced_2025.parquet"
)
BOOTSTRAP_PATH = Path(
    "data/processed/market_backtest/event_game_cluster_bootstrap.csv"
)
OUT_DIR = Path(
    "data/processed/market_backtest/diagnostic"
)

PRIMARY_EDGE = 0.03
BOOTSTRAP_REPS = 10_000
SEED = 73


def grade_profit(row: pd.Series) -> float:
    if bool(row["actual_push"]):
        return 0.0

    if row["bet_side"] == "over":
        won = bool(
            row["actual"] > row["line_value"]
        )
        decimal = american_to_decimal(
            row["over_odds"]
        )
    else:
        won = bool(
            row["actual"] < row["line_value"]
        )
        decimal = american_to_decimal(
            row["under_odds"]
        )

    return float(
        decimal - 1.0
        if won
        else -1.0
    )


def selected_side_probability(
    frame: pd.DataFrame,
    prefix: str,
) -> np.ndarray:
    if prefix == "model":
        over = frame[
            "q_over_nonpush"
        ].to_numpy(dtype=float)
        under = frame[
            "q_under_nonpush"
        ].to_numpy(dtype=float)
    elif prefix == "market":
        over = frame[
            "market_q_over"
        ].to_numpy(dtype=float)
        under = frame[
            "market_q_under"
        ].to_numpy(dtype=float)
    else:
        raise KeyError(prefix)

    choose_over = frame[
        "bet_side"
    ].eq("over").to_numpy()

    return np.where(
        choose_over,
        over,
        under,
    )


def realized_selected_side_win(
    frame: pd.DataFrame,
) -> np.ndarray:
    choose_over = frame[
        "bet_side"
    ].eq("over").to_numpy()

    actual = frame[
        "actual"
    ].to_numpy(dtype=float)

    line = frame[
        "line_value"
    ].to_numpy(dtype=float)

    return np.where(
        choose_over,
        actual > line,
        actual < line,
    ).astype(float)


def choose_best_event(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    candidates = frame[
        frame[
            "bet_model_ev"
        ].gt(0.0)
    ].copy()

    candidates = candidates.sort_values(
        [
            "bet_model_ev",
            "bet_edge",
        ],
        ascending=False,
    )

    return candidates.drop_duplicates(
        subset=[
            "game_id",
            "player_id",
            "prop_type",
        ],
        keep="first",
    ).copy()


def choose_best_vendor_event(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    candidates = frame[
        frame[
            "bet_model_ev"
        ].gt(0.0)
    ].copy()

    candidates = candidates.sort_values(
        [
            "bet_model_ev",
            "bet_edge",
        ],
        ascending=False,
    )

    return candidates.drop_duplicates(
        subset=[
            "game_id",
            "player_id",
            "prop_type",
            "vendor",
        ],
        keep="first",
    ).copy()


def add_profit(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    out[
        "profit_units"
    ] = out.apply(
        grade_profit,
        axis=1,
    )

    out[
        "selected_model_probability"
    ] = selected_side_probability(
        out,
        "model",
    )

    out[
        "selected_market_probability"
    ] = selected_side_probability(
        out,
        "market",
    )

    out[
        "selected_side_win"
    ] = realized_selected_side_win(
        out
    )

    return out


def calibration_table(
    frame: pd.DataFrame,
    probability_column: str,
    bins: int = 10,
) -> pd.DataFrame:
    work = frame.loc[
        ~frame[
            "actual_push"
        ]
    ].copy()

    work[
        "actual_over"
    ] = (
        work[
            "actual"
        ]
        > work[
            "line_value"
        ]
    ).astype(float)

    ranks = pd.qcut(
        work[
            probability_column
        ],
        q=bins,
        duplicates="drop",
    )

    table = (
        work.assign(
            probability_bin=ranks
        )
        .groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            n=(
                "actual_over",
                "size",
            ),
            mean_probability=(
                probability_column,
                "mean",
            ),
            actual_over_rate=(
                "actual_over",
                "mean",
            ),
        )
        .reset_index()
    )

    table[
        "calibration_error"
    ] = (
        table[
            "mean_probability"
        ]
        - table[
            "actual_over_rate"
        ]
    )

    return table


def expected_calibration_error(
    table: pd.DataFrame,
) -> float:
    if table.empty:
        return float("nan")

    total = float(
        table[
            "n"
        ].sum()
    )

    return float(
        np.sum(
            table[
                "n"
            ]
            * np.abs(
                table[
                    "calibration_error"
                ]
            )
        )
        / total
    )


def game_cluster_roi_bootstrap(
    bets: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if bets.empty:
        return {
            "bets": 0,
            "games": 0,
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
            profit_sum=(
                "profit_units",
                "sum",
            ),
            bet_count=(
                "profit_units",
                "size",
            ),
        )
    )

    profit = by_game[
        "profit_sum"
    ].to_numpy(dtype=float)

    count = by_game[
        "bet_count"
    ].to_numpy(dtype=float)

    n_games = len(
        by_game
    )

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        reps,
        dtype=float,
    )

    for rep in range(
        reps
    ):
        sampled = rng.integers(
            0,
            n_games,
            size=n_games,
        )

        boot[
            rep
        ] = (
            profit[
                sampled
            ].sum()
            / count[
                sampled
            ].sum()
        )

    low, high = np.quantile(
        boot,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "bets": int(
            len(
                bets
            )
        ),
        "games": int(
            n_games
        ),
        "roi": float(
            bets[
                "profit_units"
            ].mean()
        ),
        "ci_low": float(
            low
        ),
        "ci_high": float(
            high
        ),
        "prob_roi_positive": float(
            np.mean(
                boot > 0.0
            )
        ),
    }


def edge_bucket_table(
    bets: pd.DataFrame,
) -> pd.DataFrame:
    bins = [
        0.0,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        np.inf,
    ]

    labels = [
        "0-2%",
        "2-3%",
        "3-5%",
        "5-7.5%",
        "7.5-10%",
        "10-15%",
        "15-20%",
        "20%+",
    ]

    work = bets.copy()

    work[
        "edge_bucket"
    ] = pd.cut(
        work[
            "bet_edge"
        ],
        bins=bins,
        labels=labels,
        right=False,
    )

    return (
        work.groupby(
            "edge_bucket",
            observed=True,
        )
        .agg(
            bets=(
                "profit_units",
                "size",
            ),
            games=(
                "game_id",
                "nunique",
            ),
            avg_edge=(
                "bet_edge",
                "mean",
            ),
            avg_model_ev=(
                "bet_model_ev",
                "mean",
            ),
            avg_model_win_probability=(
                "selected_model_probability",
                "mean",
            ),
            avg_market_win_probability=(
                "selected_market_probability",
                "mean",
            ),
            realized_win_rate=(
                "selected_side_win",
                "mean",
            ),
            profit_units=(
                "profit_units",
                "sum",
            ),
            roi=(
                "profit_units",
                "mean",
            ),
        )
        .reset_index()
    )


def main() -> None:
    if not PRICED_PATH.exists():
        raise SystemExit(
            f"ERROR: missing {PRICED_PATH}. "
            "Run scripts/09_backtest.py first."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    priced = pd.read_parquet(
        PRICED_PATH
    )

    required = {
        "game_id",
        "player_id",
        "prop_type",
        "vendor",
        "line_value",
        "actual",
        "actual_push",
        "q_over_nonpush",
        "q_under_nonpush",
        "market_q_over",
        "market_q_under",
        "bet_side",
        "bet_edge",
        "bet_model_ev",
        "over_odds",
        "under_odds",
    }

    missing = required - set(
        priced.columns
    )

    if missing:
        raise SystemExit(
            f"ERROR: priced file missing columns: "
            f"{sorted(missing)}"
        )

    print("=" * 122)
    print(
        "2025 DEVELOPMENT MARKET-BACKTEST DIAGNOSTIC"
    )
    print("=" * 122)
    print(
        f"Priced quotes: {len(priced):,}"
    )
    print(
        f"Games: {priced['game_id'].nunique():,}"
    )
    print(
        "This is diagnostic/development analysis, "
        "not an external performance claim."
    )

    # --------------------------------------------------------------
    # Existing model-vs-market clustered scoring uncertainty.
    # --------------------------------------------------------------
    print("\n" + "=" * 122)
    print(
        "1. EVENT-LEVEL GAME-CLUSTERED MODEL-vs-MARKET UNCERTAINTY"
    )
    print("=" * 122)

    if BOOTSTRAP_PATH.exists():
        bootstrap = pd.read_csv(
            BOOTSTRAP_PATH
        )

        print(
            bootstrap.to_string(
                index=False,
                formatters={
                    "brier_delta": (
                        lambda x:
                        f"{x:+.6f}"
                    ),
                    "brier_ci_low": (
                        lambda x:
                        f"{x:+.6f}"
                    ),
                    "brier_ci_high": (
                        lambda x:
                        f"{x:+.6f}"
                    ),
                    "brier_prob_model_better": (
                        lambda x:
                        f"{x:.3f}"
                    ),
                    "logloss_delta": (
                        lambda x:
                        f"{x:+.6f}"
                    ),
                    "logloss_ci_low": (
                        lambda x:
                        f"{x:+.6f}"
                    ),
                    "logloss_ci_high": (
                        lambda x:
                        f"{x:+.6f}"
                    ),
                    "logloss_prob_model_better": (
                        lambda x:
                        f"{x:.3f}"
                    ),
                },
            )
        )
    else:
        print(
            f"Missing {BOOTSTRAP_PATH}"
        )

    # --------------------------------------------------------------
    # Contract-level calibration.
    # One row per game/player/prop/line prevents vendors from
    # multiplying identical model probabilities.
    # --------------------------------------------------------------
    print("\n" + "=" * 122)
    print(
        "2. CONTRACT-LEVEL CALIBRATION"
    )
    print("=" * 122)

    contract = (
        priced.groupby(
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
            actual_push=(
                "actual_push",
                "first",
            ),
            q_over_nonpush=(
                "q_over_nonpush",
                "first",
            ),
            market_q_over=(
                "market_q_over",
                "mean",
            ),
        )
    )

    model_cal = calibration_table(
        contract,
        "q_over_nonpush",
    )

    market_cal = calibration_table(
        contract,
        "market_q_over",
    )

    model_ece = expected_calibration_error(
        model_cal
    )

    market_ece = expected_calibration_error(
        market_cal
    )

    print(
        f"Model 10-bin ECE:  {model_ece:.4f}"
    )
    print(
        f"Market 10-bin ECE: {market_ece:.4f}"
    )

    print(
        "\nModel probability calibration:"
    )
    print(
        model_cal.to_string(
            index=False,
            formatters={
                "mean_probability": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "actual_over_rate": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "calibration_error": (
                    lambda x:
                    f"{x:+.4f}"
                ),
            },
        )
    )

    model_cal.to_csv(
        OUT_DIR
        / "model_probability_calibration.csv",
        index=False,
    )

    market_cal.to_csv(
        OUT_DIR
        / "market_probability_calibration.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Best available quote per player/game/prop.
    # --------------------------------------------------------------
    best_event = add_profit(
        choose_best_event(
            priced
        )
    )

    best_vendor = add_profit(
        choose_best_vendor_event(
            priced
        )
    )

    print("\n" + "=" * 122)
    print(
        "3. BEST-EVENT EDGE BUCKETS"
    )
    print("=" * 122)

    buckets = edge_bucket_table(
        best_event
    )

    print(
        buckets.to_string(
            index=False,
            formatters={
                "avg_edge": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "avg_model_ev": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "avg_model_win_probability": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "avg_market_win_probability": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "realized_win_rate": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "profit_units": (
                    lambda x:
                    f"{x:+.2f}"
                ),
                "roi": (
                    lambda x:
                    f"{x:+.3%}"
                ),
            },
        )
    )

    buckets.to_csv(
        OUT_DIR
        / "best_event_edge_buckets.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Primary 3% edge by prop.
    # --------------------------------------------------------------
    print("\n" + "=" * 122)
    print(
        "4. BEST-EVENT BETTING BY PROP @ 3% EDGE"
    )
    print("=" * 122)

    primary = best_event[
        best_event[
            "bet_edge"
        ].ge(
            PRIMARY_EDGE
        )
    ].copy()

    by_prop = (
        primary.groupby(
            "prop_type",
            as_index=False,
        )
        .agg(
            bets=(
                "profit_units",
                "size",
            ),
            games=(
                "game_id",
                "nunique",
            ),
            avg_edge=(
                "bet_edge",
                "mean",
            ),
            avg_model_win_probability=(
                "selected_model_probability",
                "mean",
            ),
            avg_market_win_probability=(
                "selected_market_probability",
                "mean",
            ),
            realized_win_rate=(
                "selected_side_win",
                "mean",
            ),
            profit_units=(
                "profit_units",
                "sum",
            ),
            roi=(
                "profit_units",
                "mean",
            ),
        )
        .sort_values(
            "roi",
            ascending=False,
        )
    )

    print(
        by_prop.to_string(
            index=False,
            formatters={
                "avg_edge": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "avg_model_win_probability": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "avg_market_win_probability": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "realized_win_rate": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "profit_units": (
                    lambda x:
                    f"{x:+.2f}"
                ),
                "roi": (
                    lambda x:
                    f"{x:+.3%}"
                ),
            },
        )
    )

    by_prop.to_csv(
        OUT_DIR
        / "best_event_3pct_by_prop.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Primary edge by side.
    # --------------------------------------------------------------
    print("\n" + "=" * 122)
    print(
        "5. BEST-EVENT BETTING BY SIDE @ 3% EDGE"
    )
    print("=" * 122)

    by_side = (
        primary.groupby(
            "bet_side",
            as_index=False,
        )
        .agg(
            bets=(
                "profit_units",
                "size",
            ),
            avg_edge=(
                "bet_edge",
                "mean",
            ),
            avg_model_win_probability=(
                "selected_model_probability",
                "mean",
            ),
            avg_market_win_probability=(
                "selected_market_probability",
                "mean",
            ),
            realized_win_rate=(
                "selected_side_win",
                "mean",
            ),
            profit_units=(
                "profit_units",
                "sum",
            ),
            roi=(
                "profit_units",
                "mean",
            ),
        )
    )

    print(
        by_side.to_string(
            index=False,
            formatters={
                "avg_edge": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "avg_model_win_probability": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "avg_market_win_probability": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "realized_win_rate": (
                    lambda x:
                    f"{x:.4f}"
                ),
                "profit_units": (
                    lambda x:
                    f"{x:+.2f}"
                ),
                "roi": (
                    lambda x:
                    f"{x:+.3%}"
                ),
            },
        )
    )

    by_side.to_csv(
        OUT_DIR
        / "best_event_3pct_by_side.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Game-clustered ROI uncertainty.
    # --------------------------------------------------------------
    print("\n" + "=" * 122)
    print(
        "6. 3% EDGE ROI — GAME-CLUSTER BOOTSTRAP"
    )
    print("=" * 122)

    roi_rows = []

    for offset, (
        name,
        frame,
    ) in enumerate(
        [
            (
                "best_event",
                best_event,
            ),
            (
                "best_vendor_event",
                best_vendor,
            ),
        ]
    ):
        bets = frame[
            frame[
                "bet_edge"
            ].ge(
                PRIMARY_EDGE
            )
        ].copy()

        result = (
            game_cluster_roi_bootstrap(
                bets,
                reps=BOOTSTRAP_REPS,
                seed=SEED
                + 1009
                * offset,
            )
        )

        result[
            "mode"
        ] = name

        roi_rows.append(
            result
        )

    roi_bootstrap = pd.DataFrame(
        roi_rows
    )

    print(
        roi_bootstrap[
            [
                "mode",
                "bets",
                "games",
                "roi",
                "ci_low",
                "ci_high",
                "prob_roi_positive",
            ]
        ].to_string(
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

    roi_bootstrap.to_csv(
        OUT_DIR
        / "roi_game_cluster_bootstrap.csv",
        index=False,
    )

    print("\n" + "=" * 122)
    print("FINAL")
    print("=" * 122)
    print(
        f"Saved diagnostics to: "
        f"{OUT_DIR}"
    )
    print(
        "Do not increase combo simulations or tune the edge threshold yet. "
        "First determine whether the model is systematically "
        "miscalibrated versus the opening market."
    )


if __name__ == "__main__":
    main()
