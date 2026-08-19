from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


OOF_PATH = Path("data/processed/oof_predictions.parquet")
OUT_PATH = Path("data/processed/oof_ensemble_predictions.parquet")
WEIGHTS_PATH = Path("models/ensemble_weights.json")
AUDIT_DIR = Path("data/processed/ensemble_audit")

TARGETS = ["pts", "reb", "ast", "stl", "blk", "fg3m"]


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    err = y - pred
    return float(np.sqrt(np.mean(err * err)))


def fit_simplex_weights(
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Fit nonnegative weights that sum to 1 by minimizing MSE.

    Component order:
        0 = XGBoost contextual mean
        1 = expected_minutes x decay prior rate
        2 = expected_minutes x Kalman prior rate
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (n_rows, 3)")

    def objective(w: np.ndarray) -> float:
        pred = x @ w
        err = y - pred
        return float(np.mean(err * err))

    constraints = {
        "type": "eq",
        "fun": lambda w: float(np.sum(w) - 1.0),
    }

    result = minimize(
        objective,
        x0=np.array([0.70, 0.20, 0.10], dtype=float),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Weight optimization failed: {result.message}"
        )

    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / weights.sum()

    return weights


def component_columns(target: str) -> list[str]:
    return [
        f"mu_{target}",
        f"baseline_{target}_decay",
        f"baseline_{target}_kalman",
    ]


def common_mask(
    frame: pd.DataFrame,
    target: str,
) -> pd.Series:
    mask = frame[target].notna()

    for col in component_columns(target):
        mask &= frame[col].notna()

    return mask


def evaluate_models(
    frame: pd.DataFrame,
    target: str,
    ensemble_col: str,
) -> pd.DataFrame:
    model_columns = {
        "ensemble": ensemble_col,
        "xgb": f"mu_{target}",
        "decay": f"baseline_{target}_decay",
        "kalman": f"baseline_{target}_kalman",
    }

    mask = frame[target].notna()

    for col in model_columns.values():
        mask &= frame[col].notna()

    y = frame.loc[mask, target].to_numpy(dtype=float)

    rows = []

    for name, col in model_columns.items():
        pred = frame.loc[mask, col].to_numpy(dtype=float)

        rows.append(
            {
                "model": name,
                "rows": int(mask.sum()),
                "mae": mae(y, pred),
                "rmse": rmse(y, pred),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    return result


def main() -> None:
    if not OOF_PATH.exists():
        raise SystemExit(
            f"ERROR: {OOF_PATH} does not exist. "
            "Run scripts/06_train_targets.py first."
        )

    df = pd.read_parquet(OOF_PATH)

    required = {
        "season",
        "player_id",
        "game_id",
        *TARGETS,
    }

    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"ERROR: OOF file missing columns: {sorted(missing)}"
        )

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    seasons = sorted(
        pd.to_numeric(
            df["season"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    # OOF target predictions begin in 2017.
    oof_seasons = [
        season
        for season in seasons
        if df.loc[
            df["season"].eq(season),
            "mu_pts",
        ].notna().any()
    ]

    if len(oof_seasons) < 2:
        raise SystemExit(
            "ERROR: need at least two seasons with target OOF predictions."
        )

    first_blend_validation_season = oof_seasons[1]

    print("=" * 104)
    print("LEAKAGE-SAFE CONVEX MEAN ENSEMBLE")
    print("=" * 104)
    print(
        f"Target OOF seasons available: "
        f"{oof_seasons[0]}-{oof_seasons[-1]}"
    )
    print(
        f"Strict blend evaluation seasons: "
        f"{first_blend_validation_season}-{oof_seasons[-1]}"
    )
    print()
    print(
        "Components: XGB contextual mean + decay baseline + Kalman baseline"
    )
    print(
        "Constraint: nonnegative weights summing to 1; objective = MSE/RMSE"
    )

    production_weights: dict[str, dict] = {}
    pooled_summaries = []

    for target in TARGETS:
        print("\n" + "=" * 104)
        print(f"TARGET: {target}")
        print("=" * 104)

        cols = component_columns(target)

        for col in cols:
            if col not in df.columns:
                raise SystemExit(
                    f"ERROR: required ensemble component missing: {col}"
                )

        ensemble_col = f"mu_ensemble_{target}"
        df[ensemble_col] = np.nan

        fold_rows = []

        # --------------------------------------------------------------
        # Strict season-walk-forward blend weights:
        #   2018 <- fit weights on 2017
        #   2019 <- fit weights on 2017-2018
        #   ...
        #   2025 <- fit weights on 2017-2024
        # --------------------------------------------------------------
        for validation_season in oof_seasons:
            if validation_season < first_blend_validation_season:
                continue

            train_mask = (
                df["season"].lt(validation_season)
                & common_mask(df, target)
            )

            validation_mask = (
                df["season"].eq(validation_season)
                & common_mask(df, target)
            )

            if train_mask.sum() == 0 or validation_mask.sum() == 0:
                continue

            y_train = df.loc[
                train_mask,
                target,
            ].to_numpy(dtype=float)

            x_train = df.loc[
                train_mask,
                cols,
            ].to_numpy(dtype=float)

            weights = fit_simplex_weights(
                y_train,
                x_train,
            )

            x_validation = df.loc[
                validation_mask,
                cols,
            ].to_numpy(dtype=float)

            blend_pred = x_validation @ weights

            df.loc[
                validation_mask,
                ensemble_col,
            ] = blend_pred

            y_validation = df.loc[
                validation_mask,
                target,
            ].to_numpy(dtype=float)

            xgb_pred = df.loc[
                validation_mask,
                f"mu_{target}",
            ].to_numpy(dtype=float)

            decay_pred = df.loc[
                validation_mask,
                f"baseline_{target}_decay",
            ].to_numpy(dtype=float)

            kalman_pred = df.loc[
                validation_mask,
                f"baseline_{target}_kalman",
            ].to_numpy(dtype=float)

            blend_rmse = rmse(
                y_validation,
                blend_pred,
            )

            xgb_rmse = rmse(
                y_validation,
                xgb_pred,
            )

            decay_rmse = rmse(
                y_validation,
                decay_pred,
            )

            kalman_rmse = rmse(
                y_validation,
                kalman_pred,
            )

            best_component_rmse = min(
                xgb_rmse,
                decay_rmse,
                kalman_rmse,
            )

            improvement_vs_xgb = (
                1.0 - blend_rmse / xgb_rmse
            ) * 100.0

            improvement_vs_best = (
                1.0 - blend_rmse / best_component_rmse
            ) * 100.0

            row = {
                "season": int(validation_season),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "w_xgb": float(weights[0]),
                "w_decay": float(weights[1]),
                "w_kalman": float(weights[2]),
                "ensemble_mae": mae(
                    y_validation,
                    blend_pred,
                ),
                "ensemble_rmse": blend_rmse,
                "xgb_rmse": xgb_rmse,
                "decay_rmse": decay_rmse,
                "kalman_rmse": kalman_rmse,
                "improvement_vs_xgb_pct": improvement_vs_xgb,
                "improvement_vs_best_component_pct": improvement_vs_best,
            }

            fold_rows.append(row)

            print(
                f"{validation_season}: "
                f"w=[xgb {weights[0]:.3f}, "
                f"decay {weights[1]:.3f}, "
                f"kalman {weights[2]:.3f}] | "
                f"blend RMSE={blend_rmse:.4f} | "
                f"vs XGB={improvement_vs_xgb:+.3f}% | "
                f"vs best component={improvement_vs_best:+.3f}%"
            )

        fold_table = pd.DataFrame(fold_rows)

        if fold_table.empty:
            raise RuntimeError(
                f"No ensemble validation folds produced for {target}"
            )

        fold_table.to_csv(
            AUDIT_DIR
            / f"{target}_walk_forward_weights.csv",
            index=False,
        )

        # --------------------------------------------------------------
        # Strict pooled evaluation: only seasons where weights were fitted
        # exclusively from prior seasons.
        # --------------------------------------------------------------
        strict_eval = df[
            df["season"].ge(first_blend_validation_season)
        ].copy()

        pooled = evaluate_models(
            strict_eval,
            target=target,
            ensemble_col=ensemble_col,
        )

        pooled.to_csv(
            AUDIT_DIR
            / f"{target}_pooled_ensemble_benchmark.csv",
            index=False,
        )

        print("\nPooled strict walk-forward benchmark:")
        print(
            pooled.to_string(
                index=False,
                formatters={
                    "mae": lambda x: f"{x:.4f}",
                    "rmse": lambda x: f"{x:.4f}",
                },
            )
        )

        ens_row = pooled[
            pooled["model"].eq("ensemble")
        ].iloc[0]

        xgb_row = pooled[
            pooled["model"].eq("xgb")
        ].iloc[0]

        best_component = pooled[
            pooled["model"].isin(
                ["xgb", "decay", "kalman"]
            )
        ].sort_values("rmse").iloc[0]

        pooled_vs_xgb = (
            1.0
            - float(ens_row["rmse"])
            / float(xgb_row["rmse"])
        ) * 100.0

        pooled_vs_best = (
            1.0
            - float(ens_row["rmse"])
            / float(best_component["rmse"])
        ) * 100.0

        print(
            f"Pooled ensemble improvement vs XGB: "
            f"{pooled_vs_xgb:+.3f}%"
        )
        print(
            f"Pooled ensemble improvement vs best component "
            f"({best_component['model']}): "
            f"{pooled_vs_best:+.3f}%"
        )

        # --------------------------------------------------------------
        # Final production weights:
        # fit on every row with leakage-safe 2017-2025 OOF components.
        # These are for future/live inference only, not historical scoring.
        # --------------------------------------------------------------
        prod_mask = common_mask(
            df,
            target,
        )

        y_prod = df.loc[
            prod_mask,
            target,
        ].to_numpy(dtype=float)

        x_prod = df.loc[
            prod_mask,
            cols,
        ].to_numpy(dtype=float)

        prod_weights = fit_simplex_weights(
            y_prod,
            x_prod,
        )

        production_weights[target] = {
            "components": [
                "xgb",
                "decay",
                "kalman",
            ],
            "weights": {
                "xgb": float(prod_weights[0]),
                "decay": float(prod_weights[1]),
                "kalman": float(prod_weights[2]),
            },
            "fit_rows": int(prod_mask.sum()),
            "fit_seasons": [
                int(df.loc[prod_mask, "season"].min()),
                int(df.loc[prod_mask, "season"].max()),
            ],
            "objective": "mean_squared_error",
            "constraint": "nonnegative_simplex",
            "strict_oof_evaluation_start_season": int(
                first_blend_validation_season
            ),
        }

        print(
            "Production weights: "
            f"xgb={prod_weights[0]:.4f}, "
            f"decay={prod_weights[1]:.4f}, "
            f"kalman={prod_weights[2]:.4f}"
        )

        pooled_summaries.append(
            {
                "target": target,
                "strict_eval_rows": int(
                    pooled["rows"].iloc[0]
                ),
                "ensemble_mae": float(
                    ens_row["mae"]
                ),
                "ensemble_rmse": float(
                    ens_row["rmse"]
                ),
                "xgb_rmse": float(
                    xgb_row["rmse"]
                ),
                "best_component": str(
                    best_component["model"]
                ),
                "best_component_rmse": float(
                    best_component["rmse"]
                ),
                "improvement_vs_xgb_pct": pooled_vs_xgb,
                "improvement_vs_best_component_pct": pooled_vs_best,
                "production_w_xgb": float(
                    prod_weights[0]
                ),
                "production_w_decay": float(
                    prod_weights[1]
                ),
                "production_w_kalman": float(
                    prod_weights[2]
                ),
            }
        )

    # Preserve original XGB means explicitly for downstream auditing.
    for target in TARGETS:
        xgb_col = f"mu_xgb_{target}"

        if xgb_col not in df.columns:
            df[xgb_col] = df[f"mu_{target}"]

    write_path = OUT_PATH
    write_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        write_path,
        index=False,
    )

    with WEIGHTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            production_weights,
            handle,
            indent=2,
            sort_keys=True,
        )

    summary = pd.DataFrame(
        pooled_summaries
    )

    summary.to_csv(
        AUDIT_DIR
        / "ensemble_summary.csv",
        index=False,
    )

    print("\n" + "=" * 104)
    print("FINAL ENSEMBLE SUMMARY")
    print("=" * 104)

    print(
        summary.to_string(
            index=False,
            formatters={
                "ensemble_mae": lambda x: f"{x:.4f}",
                "ensemble_rmse": lambda x: f"{x:.4f}",
                "xgb_rmse": lambda x: f"{x:.4f}",
                "best_component_rmse": lambda x: f"{x:.4f}",
                "improvement_vs_xgb_pct": lambda x: f"{x:+.3f}",
                "improvement_vs_best_component_pct": (
                    lambda x: f"{x:+.3f}"
                ),
                "production_w_xgb": lambda x: f"{x:.4f}",
                "production_w_decay": lambda x: f"{x:.4f}",
                "production_w_kalman": lambda x: f"{x:.4f}",
            },
        )
    )

    print()
    print(f"Saved ensemble OOF file: {OUT_PATH}")
    print(f"Saved production weights: {WEIGHTS_PATH}")
    print(f"Saved audit tables: {AUDIT_DIR}")
    print()
    print(
        "IMPORTANT: mu_ensemble_* is strict walk-forward OOF only "
        f"from season {first_blend_validation_season} onward."
    )
    print(
        "Do not run marginal fitting yet. Review the ensemble summary first."
    )


if __name__ == "__main__":
    main()
