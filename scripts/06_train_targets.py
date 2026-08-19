from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.features import TARGETS
from nba_prop_quant.model import fit_target_model
from nba_prop_quant.pipeline import oof_matrix_path, load_yaml
from nba_prop_quant.settings import get_settings
from nba_prop_quant.storage import write_parquet_atomic

console = Console()


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    error = y - pred
    return float(np.sqrt(np.mean(error * error)))


def _season_walk_forward_oof_target_verbose(
    frame: pd.DataFrame,
    target: str,
    params: dict,
) -> np.ndarray:
    seasons = sorted(
        pd.to_numeric(frame["season"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )

    if len(seasons) < 2:
        raise ValueError(
            "At least two seasons are required for target walk-forward OOF."
        )

    season_values = (
        pd.to_numeric(frame["season"], errors="raise")
        .astype(int)
        .to_numpy()
    )

    oof = np.full(len(frame), np.nan, dtype=float)

    first_validation_season = seasons[1]

    for validation_season in seasons:
        if validation_season < first_validation_season:
            continue

        train_mask = season_values < validation_season
        validation_mask = season_values == validation_season

        train_rows = int(train_mask.sum())
        validation_rows = int(validation_mask.sum())

        if train_rows == 0 or validation_rows == 0:
            continue

        if frame.loc[train_mask, "expected_minutes"].isna().any():
            raise ValueError(
                f"{target}: training rows before season {validation_season} "
                "contain missing expected_minutes."
            )

        console.print(
            f"[cyan]{target} fold {validation_season}[/cyan]: "
            f"train={train_rows:,}, validate={validation_rows:,}"
        )

        started = time.perf_counter()

        bundle = fit_target_model(
            frame.loc[train_mask],
            target=target,
            params=params,
        )

        oof[validation_mask] = bundle.predict(
            frame.loc[validation_mask]
        )

        elapsed = time.perf_counter() - started

        y_fold = frame.loc[
            validation_mask,
            target,
        ].to_numpy(dtype=float)

        pred_fold = oof[validation_mask]

        console.print(
            f"  finished in {elapsed:.1f}s | "
            f"MAE={_mae(y_fold, pred_fold):.4f} "
            f"RMSE={_rmse(y_fold, pred_fold):.4f}"
        )

    return oof


def _baseline_columns(target: str) -> dict[str, str]:
    return {
        "prior10": f"prior_{target}_rate10",
        "decay": f"decay_prior_{target}_rate",
        "kalman": f"kalman_prior_{target}_rate",
    }


def _add_baseline_predictions(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    out = frame.copy()

    for label, rate_col in _baseline_columns(target).items():
        if rate_col not in out.columns:
            raise KeyError(
                f"Required baseline feature is missing: {rate_col}"
            )

        out[f"baseline_{target}_{label}"] = np.clip(
            pd.to_numeric(
                out["expected_minutes"],
                errors="coerce",
            )
            * pd.to_numeric(
                out[rate_col],
                errors="coerce",
            ),
            0.0,
            None,
        )

    return out


def _pooled_benchmark(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    model_columns = {
        "xgb": f"mu_{target}",
        "minutes_x_prior10": f"baseline_{target}_prior10",
        "minutes_x_decay": f"baseline_{target}_decay",
        "minutes_x_kalman": f"baseline_{target}_kalman",
    }

    common_mask = frame[f"mu_{target}"].notna()

    for col in model_columns.values():
        common_mask &= frame[col].notna()

    y = frame.loc[
        common_mask,
        target,
    ].to_numpy(dtype=float)

    rows = []

    for model_name, col in model_columns.items():
        pred = frame.loc[
            common_mask,
            col,
        ].to_numpy(dtype=float)

        rows.append(
            {
                "model": model_name,
                "rows": int(common_mask.sum()),
                "mae": _mae(y, pred),
                "rmse": _rmse(y, pred),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    xgb_rmse = float(
        result.loc[
            result["model"].eq("xgb"),
            "rmse",
        ].iloc[0]
    )

    baseline_only = result[
        ~result["model"].eq("xgb")
    ]

    best_baseline_row = baseline_only.loc[
        baseline_only["rmse"].idxmin()
    ]

    best_baseline_rmse = float(
        best_baseline_row["rmse"]
    )

    improvement_pct = (
        1.0
        - xgb_rmse / best_baseline_rmse
    ) * 100.0

    console.print(
        f"Best baseline: {best_baseline_row['model']} | "
        f"XGB RMSE improvement={improvement_pct:+.2f}%"
    )

    return result


def _season_benchmark(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    rows = []

    model_columns = {
        "xgb": f"mu_{target}",
        "prior10": f"baseline_{target}_prior10",
        "decay": f"baseline_{target}_decay",
        "kalman": f"baseline_{target}_kalman",
    }

    for season, group in frame.groupby(
        "season",
        sort=True,
    ):
        common_mask = group[f"mu_{target}"].notna()

        for col in model_columns.values():
            common_mask &= group[col].notna()

        if not common_mask.any():
            continue

        y = group.loc[
            common_mask,
            target,
        ].to_numpy(dtype=float)

        row = {
            "season": int(season),
            "rows": int(common_mask.sum()),
        }

        for model_name, col in model_columns.items():
            pred = group.loc[
                common_mask,
                col,
            ].to_numpy(dtype=float)

            row[f"{model_name}_mae"] = _mae(y, pred)
            row[f"{model_name}_rmse"] = _rmse(y, pred)

        best_baseline_rmse = min(
            row["prior10_rmse"],
            row["decay_rmse"],
            row["kalman_rmse"],
        )

        row["xgb_rmse_improvement_pct"] = (
            1.0
            - row["xgb_rmse"] / best_baseline_rmse
        ) * 100.0

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    settings = get_settings()
    config = load_yaml(Path("configs/model.yaml"))

    stack_path = (
        settings.processed_dir
        / "stack_training.parquet"
    )

    df = pd.read_parquet(stack_path)

    df = df[
        df["expected_minutes"].notna()
    ].copy()

    df = df.sort_values(
        ["date", "game_id", "player_id"]
    ).reset_index(drop=True)

    seasons = sorted(
        df["season"]
        .astype(int)
        .unique()
        .tolist()
    )

    console.print(
        f"Target modeling rows with OOF minutes: {len(df):,}"
    )
    console.print(
        f"Target OOF walk-forward seasons: "
        f"{seasons[1]}-{seasons[-1]}"
    )

    audit_dir = (
        settings.processed_dir
        / "target_audit"
    )
    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_started = time.perf_counter()

    for target in TARGETS:
        console.rule(
            f"Training {target}"
        )

        target_started = time.perf_counter()

        df[f"mu_{target}"] = (
            _season_walk_forward_oof_target_verbose(
                df,
                target=target,
                params=config["model"],
            )
        )

        coverage = (
            df[f"mu_{target}"]
            .notna()
            .mean()
        )

        console.print(
            f"{target} OOF coverage: "
            f"{coverage:.2%}"
        )

        df = _add_baseline_predictions(
            df,
            target=target,
        )

        console.rule(
            f"{target} pooled OOF benchmark"
        )

        pooled = _pooled_benchmark(
            df,
            target=target,
        )

        console.print(
            pooled.to_string(
                index=False,
                formatters={
                    "mae": lambda x: f"{x:.4f}",
                    "rmse": lambda x: f"{x:.4f}",
                },
            )
        )

        console.rule(
            f"{target} OOF metrics by season"
        )

        by_season = _season_benchmark(
            df,
            target=target,
        )

        display_cols = [
            "season",
            "rows",
            "xgb_mae",
            "xgb_rmse",
            "prior10_rmse",
            "decay_rmse",
            "kalman_rmse",
            "xgb_rmse_improvement_pct",
        ]

        console.print(
            by_season[
                display_cols
            ].to_string(
                index=False,
                formatters={
                    "xgb_mae": lambda x: f"{x:.4f}",
                    "xgb_rmse": lambda x: f"{x:.4f}",
                    "prior10_rmse": lambda x: f"{x:.4f}",
                    "decay_rmse": lambda x: f"{x:.4f}",
                    "kalman_rmse": lambda x: f"{x:.4f}",
                    "xgb_rmse_improvement_pct": (
                        lambda x: f"{x:+.2f}"
                    ),
                },
            )
        )

        pooled.to_csv(
            audit_dir
            / f"{target}_pooled_benchmark.csv",
            index=False,
        )

        by_season.to_csv(
            audit_dir
            / f"{target}_season_benchmark.csv",
            index=False,
        )

        console.print(
            f"[cyan]{target} final production fit[/cyan]: "
            f"train={len(df):,}"
        )

        final_started = time.perf_counter()

        final_bundle = fit_target_model(
            df,
            target=target,
            params=config["model"],
        )

        final_bundle.save(
            settings.nba_prop_model_dir
            / f"{target}.joblib"
        )

        final_elapsed = (
            time.perf_counter()
            - final_started
        )

        target_elapsed = (
            time.perf_counter()
            - target_started
        )

        console.print(
            f"{target} final fit finished in "
            f"{final_elapsed:.1f}s"
        )
        console.print(
            f"[green]{target} complete[/green] "
            f"in {target_elapsed:.1f}s"
        )

    write_parquet_atomic(
        df,
        oof_matrix_path(settings),
    )

    mu_columns = [
        f"mu_{target}"
        for target in TARGETS
    ]

    complete = (
        df[mu_columns]
        .notna()
        .all(axis=1)
        .mean()
    )

    console.rule(
        "Target OOF coverage by season"
    )

    coverage_table = (
        df.assign(
            complete_target_oof=(
                df[mu_columns]
                .notna()
                .all(axis=1)
            )
        )
        .groupby("season")[
            "complete_target_oof"
        ]
        .agg(["count", "sum", "mean"])
    )

    coverage_table["mean"] *= 100.0

    coverage_table = coverage_table.rename(
        columns={
            "count": "rows",
            "sum": "complete_oof_rows",
            "mean": "coverage_pct",
        }
    )

    console.print(
        coverage_table.to_string(
            formatters={
                "coverage_pct": lambda x: f"{x:.2f}"
            }
        )
    )

    total_elapsed = (
        time.perf_counter()
        - total_started
    )

    console.print(
        f"[green]Saved six target models.[/green] "
        f"Complete target OOF coverage={complete:.2%}"
    )
    console.print(
        f"OOF matrix: "
        f"{oof_matrix_path(settings)}"
    )
    console.print(
        f"Target audit tables: "
        f"{audit_dir}"
    )
    console.print(
        f"Total target-training elapsed: "
        f"{total_elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
