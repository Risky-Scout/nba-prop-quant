from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.model import (
    fit_minutes_model,
    season_walk_forward_oof_minutes,
)
from nba_prop_quant.pipeline import feature_matrix_path, load_yaml
from nba_prop_quant.settings import get_settings
from nba_prop_quant.storage import write_parquet_atomic

console = Console()


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    error = y - pred
    return float(np.sqrt(np.mean(error * error)))


def main() -> None:
    settings = get_settings()
    config = load_yaml(Path("configs/model.yaml"))

    df = pd.read_parquet(feature_matrix_path(settings))

    model_frame = df[
        df["season"] >= config["advanced_start_season"]
    ].copy()

    model_frame = model_frame.sort_values(
        ["date", "game_id", "player_id"]
    ).reset_index(drop=True)

    console.print(f"Minutes modeling rows: {len(model_frame):,}")

    seasons = sorted(
        model_frame["season"].astype(int).unique().tolist()
    )

    console.print(
        f"Minutes walk-forward seasons: {seasons[1]}-{seasons[-1]}"
    )

    model_frame["expected_minutes"] = season_walk_forward_oof_minutes(
        model_frame,
        params=config["model"],
    )

    eval_mask = model_frame["expected_minutes"].notna()

    console.print(
        f"OOF minutes coverage: {eval_mask.mean():.2%} "
        f"({eval_mask.sum():,}/{len(model_frame):,})"
    )

    console.rule("Minutes OOF coverage by season")

    coverage = (
        model_frame.assign(has_oof=eval_mask)
        .groupby("season")["has_oof"]
        .agg(["count", "sum", "mean"])
    )
    coverage["mean"] *= 100.0
    coverage = coverage.rename(
        columns={
            "count": "rows",
            "sum": "oof_rows",
            "mean": "coverage_pct",
        }
    )
    console.print(
        coverage.to_string(
            formatters={"coverage_pct": lambda x: f"{x:.2f}"}
        )
    )

    console.rule("Minutes OOF benchmark")

    candidates = {
        "xgb_expected_minutes": "expected_minutes",
        "prior_minutes10": "prior_minutes10",
        "decay_prior_min": "decay_prior_min",
        "kalman_prior_min": "kalman_prior_min",
    }

    rows = []

    for name, column in candidates.items():
        available = eval_mask & model_frame[column].notna()

        y = model_frame.loc[
            available, "minutes"
        ].to_numpy(dtype=float)

        pred = model_frame.loc[
            available, column
        ].to_numpy(dtype=float)

        rows.append(
            {
                "model": name,
                "rows": int(available.sum()),
                "mae": _mae(y, pred),
                "rmse": _rmse(y, pred),
            }
        )

    benchmark = (
        pd.DataFrame(rows)
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    console.print(
        benchmark.to_string(
            index=False,
            formatters={
                "mae": lambda x: f"{x:.4f}",
                "rmse": lambda x: f"{x:.4f}",
            },
        )
    )

    console.rule("Minutes XGBoost OOF metrics by season")

    metric_rows = []

    for season, group in model_frame.loc[eval_mask].groupby(
        "season", sort=True
    ):
        y = group["minutes"].to_numpy(dtype=float)
        pred = group["expected_minutes"].to_numpy(dtype=float)

        metric_rows.append(
            {
                "season": int(season),
                "rows": len(group),
                "mae": _mae(y, pred),
                "rmse": _rmse(y, pred),
            }
        )

    metrics = pd.DataFrame(metric_rows)

    console.print(
        metrics.to_string(
            index=False,
            formatters={
                "mae": lambda x: f"{x:.4f}",
                "rmse": lambda x: f"{x:.4f}",
            },
        )
    )

    bundle = fit_minutes_model(
        model_frame,
        params=config["model"],
    )
    bundle.save(
        settings.nba_prop_model_dir / "minutes.joblib"
    )

    path = settings.processed_dir / "stack_training.parquet"
    write_parquet_atomic(model_frame, path)

    console.print(
        "[green]Saved minutes model and "
        "season-walk-forward stack training data.[/green]"
    )
    console.print(f"Stack training path: {path}")


if __name__ == "__main__":
    main()
