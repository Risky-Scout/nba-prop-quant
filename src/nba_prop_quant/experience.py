from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer


@dataclass
class ExperienceCurve:
    """
    Low-dimensional, interpretable nonlinear career-experience curve.

    BALLDONTLIE does not expose birth_date in its current NBA player schema, so
    this package models years since draft (or career-games fallback) rather than
    claiming to know exact biological age.
    """

    n_knots: int = 6
    degree: int = 3
    alpha: float = 5.0

    def __post_init__(self) -> None:
        self.model = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        n_knots=self.n_knots,
                        degree=self.degree,
                        include_bias=False,
                    ),
                ),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )

    def fit(
        self,
        experience_years: np.ndarray,
        target_rate: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ExperienceCurve":
        x = np.asarray(experience_years, dtype=float).reshape(-1, 1)
        y = np.asarray(target_rate, dtype=float)
        mask = np.isfinite(x[:, 0]) & np.isfinite(y)
        if mask.sum() < 25:
            raise ValueError("At least 25 finite rows are required for an experience curve")

        fit_params = {}
        if sample_weight is not None:
            fit_params["ridge__sample_weight"] = np.asarray(sample_weight)[mask]

        self.model.fit(x[mask], y[mask], **fit_params)
        return self

    def predict(self, experience_years: np.ndarray) -> np.ndarray:
        x = np.asarray(experience_years, dtype=float).reshape(-1, 1)
        return self.model.predict(x)


def add_prior_season_experience_curves(
    frame: pd.DataFrame,
    targets: list[str],
    min_history_rows: int = 500,
) -> pd.DataFrame:
    """
    Create leakage-safe training features.

    For rows in season S, each target's curve is fitted using seasons < S only.
    This keeps the row's current outcome and its same-season future outcomes out
    of the feature.
    """
    df = frame.copy()
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).unique())

    for target in targets:
        output = np.full(len(df), np.nan, dtype=float)
        rate = (
            pd.to_numeric(df[target], errors="coerce")
            / pd.to_numeric(df["minutes"], errors="coerce").clip(lower=1.0)
        ).to_numpy(dtype=float)

        for season in seasons:
            train_mask = (
                pd.to_numeric(df["season"], errors="coerce").to_numpy(dtype=float)
                < float(season)
            )
            score_mask = (
                pd.to_numeric(df["season"], errors="coerce").to_numpy(dtype=float)
                == float(season)
            )
            finite_train = (
                train_mask
                & np.isfinite(df["experience_years"].to_numpy(dtype=float))
                & np.isfinite(rate)
            )
            if finite_train.sum() < min_history_rows or not score_mask.any():
                continue

            curve = ExperienceCurve().fit(
                df.loc[finite_train, "experience_years"].to_numpy(dtype=float),
                rate[finite_train],
                sample_weight=(
                    df.loc[finite_train, "minutes"]
                    .clip(lower=1.0)
                    .to_numpy(dtype=float)
                ),
            )
            output[score_mask] = curve.predict(
                df.loc[score_mask, "experience_years"].to_numpy(dtype=float)
            )

        df[f"experience_curve_{target}_rate"] = output

    return df


def fit_production_experience_curves(
    frame: pd.DataFrame,
    targets: list[str],
) -> dict[str, ExperienceCurve]:
    curves: dict[str, ExperienceCurve] = {}

    for target in targets:
        rate = (
            pd.to_numeric(frame[target], errors="coerce")
            / pd.to_numeric(frame["minutes"], errors="coerce").clip(lower=1.0)
        )
        curve = ExperienceCurve().fit(
            frame["experience_years"].to_numpy(dtype=float),
            rate.to_numpy(dtype=float),
            sample_weight=frame["minutes"].clip(lower=1.0).to_numpy(dtype=float),
        )
        curves[target] = curve

    return curves


def apply_production_experience_curves(
    frame: pd.DataFrame,
    curves: dict[str, ExperienceCurve],
) -> pd.DataFrame:
    df = frame.copy()

    for target, curve in curves.items():
        df[f"experience_curve_{target}_rate"] = curve.predict(
            df["experience_years"].to_numpy(dtype=float)
        )

    return df
