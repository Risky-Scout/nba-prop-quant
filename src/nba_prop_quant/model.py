from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .features import feature_columns


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 700,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 4.0,
    "min_child_weight": 8.0,
    "random_state": 73,
    "n_jobs": -1,
    "tree_method": "hist",
}


@dataclass
class ModelBundle:
    model: XGBRegressor
    feature_names: list[str]
    target_name: str
    objective: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.reindex(columns=self.feature_names)
        prediction = self.model.predict(x)
        if self.objective == "count:poisson":
            prediction = np.clip(prediction, 1e-6, None)
        elif self.target_name == "minutes":
            prediction = np.clip(prediction, 0.0, 48.0)
        return prediction

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        return joblib.load(path)


def _fit_xgb(
    frame: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    objective: str,
    params: dict[str, Any] | None = None,
) -> XGBRegressor:
    model_params = dict(DEFAULT_XGB_PARAMS)
    if params:
        model_params.update(params)
    model = XGBRegressor(objective=objective, **model_params)
    model.fit(frame.reindex(columns=features), y)
    return model


def fit_minutes_model(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> ModelBundle:
    features = feature_columns(frame, include_expected_minutes=False)
    y = frame["minutes"].to_numpy(dtype=float)
    model = _fit_xgb(
        frame=frame,
        y=y,
        features=features,
        objective="reg:squarederror",
        params=params,
    )
    return ModelBundle(
        model=model,
        feature_names=features,
        target_name="minutes",
        objective="reg:squarederror",
    )


def fit_target_model(
    frame: pd.DataFrame,
    target: str,
    params: dict[str, Any] | None = None,
) -> ModelBundle:
    if "expected_minutes" not in frame.columns:
        raise ValueError("expected_minutes is required before fitting target models")
    features = feature_columns(frame, include_expected_minutes=True)
    y = frame[target].to_numpy(dtype=float)
    model = _fit_xgb(
        frame=frame,
        y=y,
        features=features,
        objective="count:poisson",
        params=params,
    )
    return ModelBundle(
        model=model,
        feature_names=features,
        target_name=target,
        objective="count:poisson",
    )


def expanding_time_oof_minutes(
    frame: pd.DataFrame,
    n_splits: int = 6,
    min_train_fraction: float = 0.45,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    ordered = frame.sort_values(["date", "game_id", "player_id"]).copy()
    original_index = ordered.index.to_numpy()
    dates = np.array(sorted(pd.to_datetime(ordered["date"]).dt.normalize().unique()))
    if len(dates) < n_splits + 2:
        raise ValueError("Not enough unique dates for expanding OOF predictions")

    start = max(int(len(dates) * min_train_fraction), 1)
    boundaries = np.linspace(start, len(dates), n_splits + 1, dtype=int)
    oof_sorted = np.full(len(ordered), np.nan, dtype=float)

    for fold in range(n_splits):
        val_start = boundaries[fold]
        val_end = boundaries[fold + 1]
        if val_end <= val_start:
            continue

        train_cutoff = dates[val_start]
        val_last = dates[val_end - 1]

        train_mask = pd.to_datetime(ordered["date"]).values < train_cutoff
        val_mask = (
            (pd.to_datetime(ordered["date"]).values >= train_cutoff)
            & (pd.to_datetime(ordered["date"]).values <= val_last)
        )
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        bundle = fit_minutes_model(ordered.loc[train_mask], params=params)
        oof_sorted[val_mask] = bundle.predict(ordered.loc[val_mask])

    oof = np.full(len(frame), np.nan, dtype=float)
    position = {idx: pos for pos, idx in enumerate(frame.index)}
    for sorted_pos, idx in enumerate(original_index):
        oof[position[idx]] = oof_sorted[sorted_pos]
    return oof


def expanding_time_oof_target(
    frame: pd.DataFrame,
    target: str,
    n_splits: int = 6,
    min_train_fraction: float = 0.45,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    ordered = frame.sort_values(["date", "game_id", "player_id"]).copy()
    original_index = ordered.index.to_numpy()
    dates = np.array(sorted(pd.to_datetime(ordered["date"]).dt.normalize().unique()))
    if len(dates) < n_splits + 2:
        raise ValueError("Not enough unique dates for expanding OOF predictions")

    start = max(int(len(dates) * min_train_fraction), 1)
    boundaries = np.linspace(start, len(dates), n_splits + 1, dtype=int)
    oof_sorted = np.full(len(ordered), np.nan, dtype=float)

    for fold in range(n_splits):
        val_start = boundaries[fold]
        val_end = boundaries[fold + 1]
        if val_end <= val_start:
            continue

        train_cutoff = dates[val_start]
        val_last = dates[val_end - 1]
        date_values = pd.to_datetime(ordered["date"]).values
        train_mask = date_values < train_cutoff
        val_mask = (date_values >= train_cutoff) & (date_values <= val_last)

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        bundle = fit_target_model(ordered.loc[train_mask], target=target, params=params)
        oof_sorted[val_mask] = bundle.predict(ordered.loc[val_mask])

    oof = np.full(len(frame), np.nan, dtype=float)
    position = {idx: pos for pos, idx in enumerate(frame.index)}
    for sorted_pos, idx in enumerate(original_index):
        oof[position[idx]] = oof_sorted[sorted_pos]
    return oof

def season_walk_forward_oof_minutes(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    first_validation_season: int | None = None,
) -> np.ndarray:
    """Leakage-safe season-by-season OOF minutes predictions."""
    if "season" not in frame.columns:
        raise ValueError("season column is required for walk-forward OOF")

    seasons = sorted(
        pd.to_numeric(frame["season"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if len(seasons) < 2:
        raise ValueError("At least two seasons are required")

    if first_validation_season is None:
        first_validation_season = seasons[1]

    season_values = pd.to_numeric(
        frame["season"], errors="raise"
    ).astype(int).to_numpy()

    oof = np.full(len(frame), np.nan, dtype=float)

    for validation_season in seasons:
        if validation_season < first_validation_season:
            continue

        train_mask = season_values < validation_season
        validation_mask = season_values == validation_season

        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            continue

        bundle = fit_minutes_model(
            frame.loc[train_mask],
            params=params,
        )

        oof[validation_mask] = bundle.predict(
            frame.loc[validation_mask]
        )

    return oof


def season_walk_forward_oof_target(
    frame: pd.DataFrame,
    target: str,
    params: dict[str, Any] | None = None,
    first_validation_season: int | None = None,
) -> np.ndarray:
    """Leakage-safe season-by-season OOF target predictions."""
    if "season" not in frame.columns:
        raise ValueError("season column is required for walk-forward OOF")
    if "expected_minutes" not in frame.columns:
        raise ValueError("expected_minutes is required before target OOF")

    seasons = sorted(
        pd.to_numeric(frame["season"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if len(seasons) < 2:
        raise ValueError("At least two seasons are required")

    if first_validation_season is None:
        first_validation_season = seasons[1]

    season_values = pd.to_numeric(
        frame["season"], errors="raise"
    ).astype(int).to_numpy()

    oof = np.full(len(frame), np.nan, dtype=float)

    for validation_season in seasons:
        if validation_season < first_validation_season:
            continue

        train_mask = season_values < validation_season
        validation_mask = season_values == validation_season

        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            continue

        if frame.loc[train_mask, "expected_minutes"].isna().any():
            raise ValueError(
                f"Training rows before season {validation_season} "
                "contain missing expected_minutes."
            )

        bundle = fit_target_model(
            frame.loc[train_mask],
            target=target,
            params=params,
        )

        oof[validation_mask] = bundle.predict(
            frame.loc[validation_mask]
        )

    return oof
