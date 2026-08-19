from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import (
    american_to_decimal,
    conditional_nonpush_probabilities,
    devig_two_way,
)


def add_actual_prop_outcomes(
    props: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    actual = outcomes[
        [
            "game_id",
            "player_id",
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
            "fg3m",
        ]
    ].copy()

    actual["points"] = actual["pts"]
    actual["rebounds"] = actual["reb"]
    actual["assists"] = actual["ast"]
    actual["steals"] = actual["stl"]
    actual["blocks"] = actual["blk"]
    actual["threes"] = actual["fg3m"]
    actual["points_rebounds"] = actual["pts"] + actual["reb"]
    actual["points_assists"] = actual["pts"] + actual["ast"]
    actual["rebounds_assists"] = actual["reb"] + actual["ast"]
    actual["points_rebounds_assists"] = (
        actual["pts"] + actual["reb"] + actual["ast"]
    )
    actual["stocks"] = actual["stl"] + actual["blk"]

    long = actual.melt(
        id_vars=["game_id", "player_id"],
        value_vars=[
            "points",
            "rebounds",
            "assists",
            "steals",
            "blocks",
            "threes",
            "points_rebounds",
            "points_assists",
            "rebounds_assists",
            "points_rebounds_assists",
            "stocks",
        ],
        var_name="prop_type",
        value_name="actual",
    )

    return props.merge(
        long,
        on=["game_id", "player_id", "prop_type"],
        how="left",
    )


def add_market_evaluation_columns(
    priced: pd.DataFrame,
) -> pd.DataFrame:
    df = priced.copy()

    q_over, q_under = conditional_nonpush_probabilities(
        df["p_over"].to_numpy(dtype=float),
        df["p_under"].to_numpy(dtype=float),
    )
    df["q_over_nonpush"] = q_over
    df["q_under_nonpush"] = q_under

    market_probabilities = np.array(
        [
            devig_two_way(over_odds, under_odds)
            for over_odds, under_odds in zip(
                df["over_odds"].to_numpy(),
                df["under_odds"].to_numpy(),
            )
        ],
        dtype=float,
    )

    df["market_q_over"] = market_probabilities[:, 0]
    df["market_q_under"] = market_probabilities[:, 1]
    df["edge_over"] = (
        df["q_over_nonpush"] - df["market_q_over"]
    )
    df["edge_under"] = (
        df["q_under_nonpush"] - df["market_q_under"]
    )

    df["actual_push"] = np.isclose(
        df["actual"].to_numpy(dtype=float),
        df["line_value"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    )

    df["actual_over"] = (
        df["actual"] > df["line_value"]
    ).astype(float)

    over_decimal = df["over_odds"].map(
        american_to_decimal
    )
    under_decimal = df["under_odds"].map(
        american_to_decimal
    )

    # Pushes return the stake, so unconditional bet EV is:
    # p(win)*(decimal-1) - p(loss).
    df["model_ev_over"] = (
        df["p_over"] * (over_decimal - 1.0)
        - df["p_under"]
    )
    df["model_ev_under"] = (
        df["p_under"] * (under_decimal - 1.0)
        - df["p_over"]
    )

    choose_over = (
        df["edge_over"] >= df["edge_under"]
    )

    df["bet_side"] = np.where(
        choose_over,
        "over",
        "under",
    )
    df["bet_edge"] = np.where(
        choose_over,
        df["edge_over"],
        df["edge_under"],
    )
    df["bet_model_ev"] = np.where(
        choose_over,
        df["model_ev_over"],
        df["model_ev_under"],
    )

    return df


def market_metrics(
    priced: pd.DataFrame,
) -> dict[str, float]:
    df = priced.dropna(
        subset=[
            "actual",
            "line_value",
            "p_over",
            "p_under",
            "over_odds",
            "under_odds",
        ]
    ).copy()

    if df.empty:
        return {}

    df = add_market_evaluation_columns(df)
    eval_df = df.loc[
        ~df["actual_push"]
    ].copy()

    if eval_df.empty:
        return {}

    y = eval_df["actual_over"].to_numpy(dtype=float)
    model_probability = eval_df[
        "q_over_nonpush"
    ].to_numpy(dtype=float)
    market_probability = eval_df[
        "market_q_over"
    ].to_numpy(dtype=float)

    eps = 1e-9

    brier_model = float(
        np.mean((model_probability - y) ** 2)
    )
    brier_market = float(
        np.mean((market_probability - y) ** 2)
    )

    log_loss_model = float(
        -np.mean(
            y
            * np.log(
                np.clip(
                    model_probability,
                    eps,
                    1.0 - eps,
                )
            )
            + (1.0 - y)
            * np.log(
                np.clip(
                    1.0 - model_probability,
                    eps,
                    1.0 - eps,
                )
            )
        )
    )

    log_loss_market = float(
        -np.mean(
            y
            * np.log(
                np.clip(
                    market_probability,
                    eps,
                    1.0 - eps,
                )
            )
            + (1.0 - y)
            * np.log(
                np.clip(
                    1.0 - market_probability,
                    eps,
                    1.0 - eps,
                )
            )
        )
    )

    return {
        "n": float(len(eval_df)),
        "brier_model": brier_model,
        "brier_market_devig": brier_market,
        "brier_delta_model_minus_market": (
            brier_model - brier_market
        ),
        "log_loss_model": log_loss_model,
        "log_loss_market_devig": log_loss_market,
        "log_loss_delta_model_minus_market": (
            log_loss_model - log_loss_market
        ),
    }


def flat_stake_roi(
    priced: pd.DataFrame,
    min_edge: float = 0.03,
) -> dict[str, float]:
    df = priced.dropna(
        subset=[
            "actual",
            "line_value",
            "p_over",
            "p_under",
            "over_odds",
            "under_odds",
        ]
    ).copy()

    if df.empty:
        return {
            "bets": 0.0,
            "roi": np.nan,
            "profit_units": 0.0,
        }

    df = add_market_evaluation_columns(df)
    df = df[
        (df["bet_edge"] >= float(min_edge))
        & (df["bet_model_ev"] > 0.0)
    ].copy()

    if df.empty:
        return {
            "bets": 0.0,
            "roi": np.nan,
            "profit_units": 0.0,
        }

    pnl = []

    for _, row in df.iterrows():
        if row["actual_push"]:
            pnl.append(0.0)
            continue

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

        pnl.append(
            decimal - 1.0 if won else -1.0
        )

    return {
        "bets": float(len(pnl)),
        "roi": float(np.mean(pnl)),
        "profit_units": float(np.sum(pnl)),
    }
