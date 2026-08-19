from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.market import (
    add_actual_prop_outcomes,
    add_market_evaluation_columns,
)
from nba_prop_quant.pricing import (
    COMBO_COMPONENTS,
    PROP_TO_TARGET,
    american_to_decimal,
    price_combo_lines,
    price_single_prop_frame,
)
from nba_prop_quant.storage import write_parquet_atomic


console = Console()

TARGETS = [
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "fg3m",
]

SUPPORTED_PROP_TYPES = {
    *PROP_TO_TARGET.keys(),
    *COMBO_COMPONENTS.keys(),
}

ELIGIBLE_INVENTORY = Path(
    "data/processed/market_backtest_audit/"
    "eligible_quote_inventory.parquet"
)
OOF_PATH = Path(
    "data/processed/oof_selected_means.parquet"
)
MARGINALS_PATH = Path(
    "models/marginals_pre2025.joblib"
)
COPULA_PATH = Path(
    "models/copula_pre2025.joblib"
)
POLICY_PATH = Path(
    "models/combo_dependence_policy.json"
)
OUTPUT_DIR = Path(
    "data/processed/market_backtest"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combo-simulations",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--max-hold-pct",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--primary-edge",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
    )
    return parser.parse_args()


def stable_event_seed(
    game_id: int,
    player_id: int,
    prop_type: str,
    base_seed: int,
) -> int:
    prop_code = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(prop_type)
    )
    value = (
        int(base_seed)
        + 1_000_003 * int(game_id)
        + 9_176 * int(player_id)
        + 37 * prop_code
    )
    return int(value % (2**32 - 1))


def binary_log_loss(
    probability: np.ndarray,
    outcome: np.ndarray,
) -> np.ndarray:
    probability = np.clip(
        np.asarray(probability, dtype=float),
        1e-9,
        1.0 - 1e-9,
    )
    outcome = np.asarray(outcome, dtype=float)
    return -(
        outcome * np.log(probability)
        + (1.0 - outcome) * np.log(1.0 - probability)
    )


def score_nonpush_quotes(
    priced: pd.DataFrame,
) -> pd.DataFrame:
    scored = priced.loc[
        ~priced["actual_push"]
    ].copy()

    y = scored["actual_over"].to_numpy(dtype=float)
    model_probability = scored[
        "q_over_nonpush"
    ].to_numpy(dtype=float)
    market_probability = scored[
        "market_q_over"
    ].to_numpy(dtype=float)

    scored["brier_model"] = (
        model_probability - y
    ) ** 2
    scored["brier_market"] = (
        market_probability - y
    ) ** 2
    scored["logloss_model"] = binary_log_loss(
        model_probability,
        y,
    )
    scored["logloss_market"] = binary_log_loss(
        market_probability,
        y,
    )
    return scored


def aggregate_metric_row(
    frame: pd.DataFrame,
    scope: str,
    label: str,
) -> dict:
    if frame.empty:
        return {
            "scope": scope,
            "label": label,
            "n": 0,
        }

    return {
        "scope": scope,
        "label": label,
        "n": int(len(frame)),
        "brier_model": float(
            frame["brier_model"].mean()
        ),
        "brier_market": float(
            frame["brier_market"].mean()
        ),
        "brier_delta_model_minus_market": float(
            (
                frame["brier_model"]
                - frame["brier_market"]
            ).mean()
        ),
        "logloss_model": float(
            frame["logloss_model"].mean()
        ),
        "logloss_market": float(
            frame["logloss_market"].mean()
        ),
        "logloss_delta_model_minus_market": float(
            (
                frame["logloss_model"]
                - frame["logloss_market"]
            ).mean()
        ),
    }


def build_contract_scores(
    scored_quotes: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "game_id",
        "player_id",
        "prop_type",
        "line_value",
    ]

    contract = (
        scored_quotes.groupby(
            keys,
            as_index=False,
        )
        .agg(
            actual_over=("actual_over", "first"),
            q_over_nonpush=("q_over_nonpush", "first"),
            market_q_over=("market_q_over", "mean"),
            vendors=("vendor", "nunique"),
            quote_rows=("vendor", "size"),
        )
    )

    y = contract["actual_over"].to_numpy(dtype=float)
    model_probability = contract[
        "q_over_nonpush"
    ].to_numpy(dtype=float)
    market_probability = contract[
        "market_q_over"
    ].to_numpy(dtype=float)

    contract["brier_model"] = (
        model_probability - y
    ) ** 2
    contract["brier_market"] = (
        market_probability - y
    ) ** 2
    contract["logloss_model"] = binary_log_loss(
        model_probability,
        y,
    )
    contract["logloss_market"] = binary_log_loss(
        market_probability,
        y,
    )
    return contract


def build_event_scores(
    contract: pd.DataFrame,
) -> pd.DataFrame:
    return (
        contract.groupby(
            [
                "game_id",
                "player_id",
                "prop_type",
            ],
            as_index=False,
        )
        .agg(
            contracts=("line_value", "size"),
            brier_model=("brier_model", "mean"),
            brier_market=("brier_market", "mean"),
            logloss_model=("logloss_model", "mean"),
            logloss_market=("logloss_market", "mean"),
        )
    )


def game_cluster_bootstrap_difference(
    event_scores: pd.DataFrame,
    model_column: str,
    market_column: str,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if event_scores.empty:
        return {
            "delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "prob_model_better": np.nan,
        }

    work = event_scores[
        ["game_id", model_column, market_column]
    ].copy()
    work["difference"] = (
        work[model_column] - work[market_column]
    )

    by_game = (
        work.groupby(
            "game_id",
            as_index=False,
        )
        .agg(
            difference_sum=("difference", "sum"),
            event_count=("difference", "size"),
        )
    )

    sums = by_game[
        "difference_sum"
    ].to_numpy(dtype=float)
    counts = by_game[
        "event_count"
    ].to_numpy(dtype=float)
    n_games = len(by_game)

    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)

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

    observed = float(
        work["difference"].mean()
    )
    low, high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return {
        "delta": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "prob_model_better": float(
            np.mean(boot < 0.0)
        ),
    }


def summarize_metrics(
    scored_quotes: pd.DataFrame,
    contract_scores: pd.DataFrame,
    event_scores: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = [
        aggregate_metric_row(
            scored_quotes,
            "quote",
            "ALL",
        ),
        aggregate_metric_row(
            contract_scores,
            "contract",
            "ALL",
        ),
        aggregate_metric_row(
            event_scores,
            "event",
            "ALL",
        ),
    ]

    prop_types = sorted(
        scored_quotes["prop_type"].unique()
    )

    for prop_type in prop_types:
        rows.extend(
            [
                aggregate_metric_row(
                    scored_quotes[
                        scored_quotes["prop_type"].eq(prop_type)
                    ],
                    "quote",
                    prop_type,
                ),
                aggregate_metric_row(
                    contract_scores[
                        contract_scores["prop_type"].eq(prop_type)
                    ],
                    "contract",
                    prop_type,
                ),
                aggregate_metric_row(
                    event_scores[
                        event_scores["prop_type"].eq(prop_type)
                    ],
                    "event",
                    prop_type,
                ),
            ]
        )

    summary = pd.DataFrame(rows)

    bootstrap_rows = []
    scopes = [("ALL", event_scores)]
    scopes.extend(
        (
            prop_type,
            event_scores[
                event_scores["prop_type"].eq(prop_type)
            ],
        )
        for prop_type in sorted(
            event_scores["prop_type"].unique()
        )
    )

    for offset, (label, group) in enumerate(scopes):
        brier = game_cluster_bootstrap_difference(
            group,
            "brier_model",
            "brier_market",
            bootstrap_reps,
            seed + 1009 * offset,
        )
        logloss = game_cluster_bootstrap_difference(
            group,
            "logloss_model",
            "logloss_market",
            bootstrap_reps,
            seed + 1009 * offset + 100_003,
        )

        bootstrap_rows.append(
            {
                "label": label,
                "brier_delta": brier["delta"],
                "brier_ci_low": brier["ci_low"],
                "brier_ci_high": brier["ci_high"],
                "brier_prob_model_better": (
                    brier["prob_model_better"]
                ),
                "logloss_delta": logloss["delta"],
                "logloss_ci_low": logloss["ci_low"],
                "logloss_ci_high": logloss["ci_high"],
                "logloss_prob_model_better": (
                    logloss["prob_model_better"]
                ),
            }
        )

    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT_DIR
        / "event_game_cluster_bootstrap.csv",
        index=False,
    )

    return summary


def grade_bet_profit(row: pd.Series) -> float:
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
        decimal - 1.0 if won else -1.0
    )


def select_bets(
    priced: pd.DataFrame,
    min_edge: float,
    mode: str,
) -> pd.DataFrame:
    bets = priced[
        (priced["bet_edge"] >= float(min_edge))
        & (priced["bet_model_ev"] > 0.0)
    ].copy()

    if bets.empty:
        return bets

    bets = bets.sort_values(
        ["bet_model_ev", "bet_edge"],
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
    bets["profit_units"] = bets.apply(
        grade_bet_profit,
        axis=1,
    )
    return bets


def betting_summary_row(
    bets: pd.DataFrame,
    threshold: float,
    mode: str,
    label: str,
) -> dict:
    if bets.empty:
        return {
            "threshold": threshold,
            "mode": mode,
            "label": label,
            "bets": 0,
            "roi": np.nan,
            "profit_units": 0.0,
        }

    return {
        "threshold": float(threshold),
        "mode": mode,
        "label": label,
        "bets": int(len(bets)),
        "games": int(
            bets["game_id"].nunique()
        ),
        "pushes": int(
            bets["actual_push"].sum()
        ),
        "average_edge": float(
            bets["bet_edge"].mean()
        ),
        "average_model_ev": float(
            bets["bet_model_ev"].mean()
        ),
        "profit_units": float(
            bets["profit_units"].sum()
        ),
        "roi": float(
            bets["profit_units"].mean()
        ),
    }


def build_betting_summaries(
    priced: pd.DataFrame,
    primary_edge: float,
) -> pd.DataFrame:
    thresholds = sorted(
        {
            0.01,
            0.02,
            float(primary_edge),
            0.05,
            0.075,
            0.10,
        }
    )
    modes = [
        "all_quotes",
        "best_vendor_event",
        "best_event",
    ]

    rows = []

    for threshold in thresholds:
        for mode in modes:
            selected = select_bets(
                priced,
                threshold,
                mode,
            )

            rows.append(
                betting_summary_row(
                    selected,
                    threshold,
                    mode,
                    "ALL",
                )
            )

            for prop_type in sorted(
                priced["prop_type"].unique()
            ):
                rows.append(
                    betting_summary_row(
                        selected[
                            selected["prop_type"].eq(prop_type)
                        ],
                        threshold,
                        mode,
                        prop_type,
                    )
                )

            if (
                np.isclose(
                    threshold,
                    primary_edge,
                )
                and mode
                in {
                    "best_vendor_event",
                    "best_event",
                }
            ):
                selected.to_csv(
                    OUTPUT_DIR
                    / (
                        f"bets_{mode}_"
                        f"edge_{int(round(100 * threshold)):02d}pct.csv"
                    ),
                    index=False,
                )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    for path in [
        ELIGIBLE_INVENTORY,
        OOF_PATH,
        MARGINALS_PATH,
        COPULA_PATH,
        POLICY_PATH,
    ]:
        if not path.exists():
            raise SystemExit(
                f"ERROR: missing required file: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    console.rule(
        "2025 DEVELOPMENT MARKET BACKTEST"
    )
    console.print(
        "[yellow]2025 is development/model-selection data for "
        "the dependence layer. These results are NOT a pristine "
        "external test.[/yellow]"
    )

    market = pd.read_parquet(
        ELIGIBLE_INVENTORY
    )
    market = market[
        market["prop_type"].isin(
            SUPPORTED_PROP_TYPES
        )
    ].copy()
    market = market[
        market["hold_pct"].between(
            0.0,
            float(args.max_hold_pct),
            inclusive="both",
        )
    ].copy()

    oof = pd.read_parquet(OOF_PATH)
    oof["season"] = pd.to_numeric(
        oof["season"],
        errors="raise",
    ).astype(int)

    model_2025 = oof[
        oof["season"].eq(2025)
    ].copy()

    outcomes = model_2025[
        [
            "game_id",
            "player_id",
            *TARGETS,
        ]
    ].copy()

    market = add_actual_prop_outcomes(
        market,
        outcomes,
    )

    market = market.drop(
        columns=["has_model"],
        errors="ignore",
    )

    merged = market.merge(
        model_2025,
        on=["game_id", "player_id"],
        how="inner",
        suffixes=("_market", ""),
        validate="many_to_one",
    )

    if merged.empty:
        raise SystemExit(
            "ERROR: no audited market rows overlap "
            "the 2025 OOF model frame."
        )

    marginals = joblib.load(MARGINALS_PATH)
    copula = joblib.load(COPULA_PATH)

    with POLICY_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        policy = json.load(handle)

    combo_lambda = {
        prop_type: float(
            policy["combos"][prop_type][
                "production_lambda"
            ]
        )
        for prop_type in COMBO_COMPONENTS
        if prop_type in policy["combos"]
    }

    missing_policy = (
        set(COMBO_COMPONENTS)
        - set(combo_lambda)
    )

    if missing_policy:
        raise SystemExit(
            "ERROR: dependence policy missing combo(s): "
            f"{sorted(missing_policy)}"
        )

    mu_columns = {
        target: f"mu_selected_{target}"
        for target in TARGETS
    }

    console.print(
        f"Audited quotes after hold filter: {len(merged):,}"
    )
    console.print(
        "Combo production lambdas: "
        + ", ".join(
            f"{key}={value:.2f}"
            for key, value in combo_lambda.items()
        )
    )

    priced_parts = []

    # Exact vectorized single-stat pricing.
    for prop_type, target in PROP_TO_TARGET.items():
        part = merged[
            merged["prop_type"].eq(prop_type)
        ].copy()

        if part.empty:
            continue

        probabilities = price_single_prop_frame(
            part,
            target=target,
            marginal=marginals[target],
            mu_column=mu_columns[target],
            line_column="line_value",
        )

        for column in probabilities.columns:
            part[column] = probabilities[column]

        part["dependence_lambda"] = 0.0
        priced_parts.append(part)

        console.print(
            f"priced {prop_type:28s}: "
            f"{len(part):,} quotes exact"
        )

    # Combo pricing once per player/game/prop event.
    combo_rows = merged[
        merged["prop_type"].isin(
            COMBO_COMPONENTS
        )
    ].copy()

    for prop_type in sorted(COMBO_COMPONENTS):
        prop_quotes = combo_rows[
            combo_rows["prop_type"].eq(prop_type)
        ].copy()

        if prop_quotes.empty:
            continue

        lambda_value = combo_lambda[prop_type]
        event_keys = [
            "game_id",
            "player_id",
            "prop_type",
        ]
        event_probabilities = []

        grouped = prop_quotes.groupby(
            event_keys,
            sort=False,
        )
        total_events = grouped.ngroups

        for event_index, (_, group) in enumerate(
            grouped,
            start=1,
        ):
            row = group.iloc[0]
            lines = np.sort(
                group["line_value"].unique()
            )

            seed = stable_event_seed(
                int(row["game_id"]),
                int(row["player_id"]),
                prop_type,
                args.seed,
            )

            probabilities = price_combo_lines(
                row=row,
                prop_type=prop_type,
                lines=lines,
                marginals=marginals,
                mu_columns=mu_columns,
                copula=copula,
                dependence_lambda=lambda_value,
                simulations=args.combo_simulations,
                seed=seed,
            )

            probabilities["game_id"] = int(
                row["game_id"]
            )
            probabilities["player_id"] = int(
                row["player_id"]
            )
            probabilities["prop_type"] = prop_type
            event_probabilities.append(probabilities)

            if (
                event_index % 500 == 0
                or event_index == total_events
            ):
                console.print(
                    f"  {prop_type}: "
                    f"{event_index:,}/{total_events:,} events"
                )

        lookup = pd.concat(
            event_probabilities,
            ignore_index=True,
        )

        priced = prop_quotes.merge(
            lookup,
            on=[
                "game_id",
                "player_id",
                "prop_type",
                "line_value",
            ],
            how="left",
            validate="many_to_one",
        )

        priced_parts.append(priced)

        method = (
            "exact independence convolution"
            if lambda_value <= 1e-12
            else (
                f"{args.combo_simulations:,}-draw "
                "shrunk-copula Monte Carlo"
            )
        )

        console.print(
            f"priced {prop_type:28s}: "
            f"{len(priced):,} quotes | "
            f"lambda={lambda_value:.2f} | {method}"
        )

    priced = pd.concat(
        priced_parts,
        ignore_index=True,
    )

    if priced.empty:
        raise SystemExit(
            "ERROR: no supported props were priced."
        )

    probability_sum = (
        priced["p_over"]
        + priced["p_under"]
        + priced["p_push"]
    )
    max_mass_error = float(
        np.max(
            np.abs(
                probability_sum - 1.0
            )
        )
    )

    console.print(
        f"Maximum probability-mass error: "
        f"{max_mass_error:.3e}"
    )

    if max_mass_error > 5e-3:
        raise RuntimeError(
            "Pricing probability mass error exceeds "
            "0.005; aborting market evaluation."
        )

    priced = add_market_evaluation_columns(
        priced
    )

    write_parquet_atomic(
        priced,
        OUTPUT_DIR
        / "backtest_priced_2025.parquet",
    )

    scored_quotes = score_nonpush_quotes(
        priced
    )
    contract_scores = build_contract_scores(
        scored_quotes
    )
    event_scores = build_event_scores(
        contract_scores
    )

    metrics = summarize_metrics(
        scored_quotes=scored_quotes,
        contract_scores=contract_scores,
        event_scores=event_scores,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    metrics.to_csv(
        OUTPUT_DIR
        / "market_scoring_summary.csv",
        index=False,
    )
    contract_scores.to_parquet(
        OUTPUT_DIR
        / "contract_scores.parquet",
        index=False,
    )
    event_scores.to_parquet(
        OUTPUT_DIR
        / "event_scores.parquet",
        index=False,
    )

    console.rule("MARKET SCORING")

    event_display = metrics[
        metrics["scope"].eq("event")
    ][
        [
            "label",
            "n",
            "brier_model",
            "brier_market",
            "brier_delta_model_minus_market",
            "logloss_model",
            "logloss_market",
            "logloss_delta_model_minus_market",
        ]
    ]

    console.print(
        event_display.to_string(
            index=False,
            formatters={
                "brier_model": lambda x: f"{x:.6f}",
                "brier_market": lambda x: f"{x:.6f}",
                "brier_delta_model_minus_market": (
                    lambda x: f"{x:+.6f}"
                ),
                "logloss_model": lambda x: f"{x:.6f}",
                "logloss_market": lambda x: f"{x:.6f}",
                "logloss_delta_model_minus_market": (
                    lambda x: f"{x:+.6f}"
                ),
            },
        )
    )

    betting = build_betting_summaries(
        priced,
        primary_edge=args.primary_edge,
    )

    betting.to_csv(
        OUTPUT_DIR
        / "betting_sensitivity.csv",
        index=False,
    )

    console.rule(
        f"HYPOTHETICAL BETTING @ "
        f"{100 * args.primary_edge:.1f}% EDGE"
    )

    primary = betting[
        np.isclose(
            betting["threshold"],
            args.primary_edge,
        )
        & betting["label"].eq("ALL")
    ][
        [
            "mode",
            "bets",
            "games",
            "pushes",
            "average_edge",
            "average_model_ev",
            "profit_units",
            "roi",
        ]
    ]

    console.print(
        primary.to_string(
            index=False,
            formatters={
                "average_edge": lambda x: f"{x:.4f}",
                "average_model_ev": lambda x: f"{x:.4f}",
                "profit_units": lambda x: f"{x:+.2f}",
                "roi": lambda x: f"{x:+.3%}",
            },
        )
    )

    console.rule("BACKTEST COMPLETE")
    console.print(
        f"Priced quote rows: {len(priced):,}"
    )
    console.print(
        f"Non-push scored quotes: "
        f"{len(scored_quotes):,}"
    )
    console.print(
        f"Unique scored contracts: "
        f"{len(contract_scores):,}"
    )
    console.print(
        f"Unique scored events: "
        f"{len(event_scores):,}"
    )
    console.print(
        f"Outputs: {OUTPUT_DIR}"
    )
    console.print()
    console.print(
        "[yellow]Interpretation warning: sportsbook results from "
        "this 2025 sample are development benchmarks. "
        "Do not present them as out-of-sample production performance. "
        "2026-27 is the first external market test.[/yellow]"
    )


if __name__ == "__main__":
    main()
