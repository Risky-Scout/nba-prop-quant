from __future__ import annotations

import argparse
from dataclasses import asdict
import math

import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.decay import DECAY_BETA_BOUNDS, tune_decay_beta
from nba_prop_quant.features import TARGETS
from nba_prop_quant.kalman import (
    KALMAN_Q_LOG_BOUNDS,
    KALMAN_R_LOG_BOUNDS,
    tune_kalman,
)
from nba_prop_quant.pipeline import (
    base_matrix_path,
    dynamic_params_path,
    save_json,
)
from nba_prop_quant.settings import get_settings

console = Console()


def _boundary_fraction(
    value: float,
    lower: float,
    upper: float,
) -> float:
    width = max(upper - lower, 1e-12)
    return min(
        (value - lower) / width,
        (upper - value) / width,
    )


def _boundary_warnings(
    beta: float,
    q: float,
    r: float,
    threshold: float = 0.02,
) -> list[str]:
    warnings: list[str] = []

    beta_distance = _boundary_fraction(
        beta,
        DECAY_BETA_BOUNDS[0],
        DECAY_BETA_BOUNDS[1],
    )
    if beta_distance < threshold:
        warnings.append(
            f"beta near search boundary {DECAY_BETA_BOUNDS}"
        )

    log_q = math.log(max(q, 1e-300))
    q_distance = _boundary_fraction(
        log_q,
        KALMAN_Q_LOG_BOUNDS[0],
        KALMAN_Q_LOG_BOUNDS[1],
    )
    if q_distance < threshold:
        warnings.append(
            f"log(q) near search boundary {KALMAN_Q_LOG_BOUNDS}"
        )

    log_r = math.log(max(r, 1e-300))
    r_distance = _boundary_fraction(
        log_r,
        KALMAN_R_LOG_BOUNDS[0],
        KALMAN_R_LOG_BOUNDS[1],
    )
    if r_distance < threshold:
        warnings.append(
            f"log(r) near search boundary {KALMAN_R_LOG_BOUNDS}"
        )

    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player-sample-mod",
        type=int,
        default=1,
        help="1=all players. 2 keeps about half of players for faster hyperparameter tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    df = pd.read_parquet(base_matrix_path(settings))
    df = df.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)

    if args.player_sample_mod > 1:
        keep = (df["player_id"].astype(int) % args.player_sample_mod) == 0
        df = df.loc[keep].copy().reset_index(drop=True)
        df["player_game_number"] = df.groupby("player_id").cumcount()

    pid = df["player_id"].to_numpy(dtype=np.int64)
    days = df["days_since_prev"].fillna(0).to_numpy(dtype=float)
    team_change = df["team_change"].fillna(0).to_numpy(dtype=float)
    history_number = df["player_game_number"].to_numpy(dtype=int)

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "min": (
            df["minutes"].to_numpy(dtype=float),
            np.ones(len(df), dtype=float),
        )
    }
    for target in TARGETS:
        values = (
            pd.to_numeric(df[target], errors="coerce")
            / df["minutes"].clip(lower=1.0)
        ).to_numpy(dtype=float)
        exposure = (df["minutes"].clip(lower=1.0) / 36.0).to_numpy(dtype=float)
        series[f"{target}_rate"] = (values, exposure)

    payload: dict[str, dict] = {"decay": {}, "kalman": {}}

    for name, (values, exposures) in series.items():
        console.rule(f"Tuning {name}")
        decay = tune_decay_beta(
            player_ids=pid,
            days_since_prev=days,
            values=values,
            history_number=history_number,
        )
        kalman = tune_kalman(
            player_ids=pid,
            days_since_prev=days,
            team_change=team_change,
            values=values,
            exposures=exposures,
            history_number=history_number,
        )
        payload["decay"][name] = asdict(decay)
        payload["kalman"][name] = asdict(kalman)
        console.print(
            f"beta={decay.beta:.6f}; q={kalman.q:.6g}; r={kalman.r:.6g}"
        )

        warnings = _boundary_warnings(
            beta=decay.beta,
            q=kalman.q,
            r=kalman.r,
        )
        if warnings:
            for warning in warnings:
                console.print(
                    f"[yellow]BOUNDARY WARNING[/yellow]: {warning}"
                )
        else:
            console.print(
                "[green]Bounds check: PASS[/green] "
                "(no parameter within 2% of a search boundary)"
            )

    save_json(payload, dynamic_params_path(settings))
    console.print(f"[green]Saved[/green] {dynamic_params_path(settings)}")


if __name__ == "__main__":
    main()
