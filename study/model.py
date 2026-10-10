"""2. Estimate minutes and stat means, fit distributions, and adjust for roles."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import nbinom, norm, poisson
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from xgboost import XGBRegressor

from .data import TARGETS, feature_columns

# From src/nba_prop_quant/experience.py


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
                    SplineTransformer(n_knots=self.n_knots, degree=self.degree, include_bias=False),
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
    frame: pd.DataFrame, targets: list[str], min_history_rows: int = 500
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
            train_mask = pd.to_numeric(df["season"], errors="coerce").to_numpy(dtype=float) < float(
                season
            )
            score_mask = pd.to_numeric(df["season"], errors="coerce").to_numpy(
                dtype=float
            ) == float(season)
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
                    df.loc[finite_train, "minutes"].clip(lower=1.0).to_numpy(dtype=float)
                ),
            )
            output[score_mask] = curve.predict(
                df.loc[score_mask, "experience_years"].to_numpy(dtype=float)
            )

        df[f"experience_curve_{target}_rate"] = output

    return df


def fit_production_experience_curves(
    frame: pd.DataFrame, targets: list[str]
) -> dict[str, ExperienceCurve]:
    curves: dict[str, ExperienceCurve] = {}

    for target in targets:
        rate = pd.to_numeric(frame[target], errors="coerce") / pd.to_numeric(
            frame["minutes"], errors="coerce"
        ).clip(lower=1.0)
        curve = ExperienceCurve().fit(
            frame["experience_years"].to_numpy(dtype=float),
            rate.to_numpy(dtype=float),
            sample_weight=frame["minutes"].clip(lower=1.0).to_numpy(dtype=float),
        )
        curves[target] = curve

    return curves


def apply_production_experience_curves(
    frame: pd.DataFrame, curves: dict[str, ExperienceCurve]
) -> pd.DataFrame:
    df = frame.copy()

    for target, curve in curves.items():
        df[f"experience_curve_{target}_rate"] = curve.predict(
            df["experience_years"].to_numpy(dtype=float)
        )

    return df


# From src/nba_prop_quant/model.py

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


def fit_minutes_model(frame: pd.DataFrame, params: dict[str, Any] | None = None) -> ModelBundle:
    features = feature_columns(frame, include_expected_minutes=False)
    y = frame["minutes"].to_numpy(dtype=float)
    model = _fit_xgb(
        frame=frame, y=y, features=features, objective="reg:squarederror", params=params
    )
    return ModelBundle(
        model=model, feature_names=features, target_name="minutes", objective="reg:squarederror"
    )


def fit_target_model(
    frame: pd.DataFrame, target: str, params: dict[str, Any] | None = None
) -> ModelBundle:
    if "expected_minutes" not in frame.columns:
        raise ValueError("expected_minutes is required before fitting target models")
    features = feature_columns(frame, include_expected_minutes=True)
    y = frame[target].to_numpy(dtype=float)
    model = _fit_xgb(frame=frame, y=y, features=features, objective="count:poisson", params=params)
    return ModelBundle(
        model=model, feature_names=features, target_name=target, objective="count:poisson"
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
        val_mask = (pd.to_datetime(ordered["date"]).values >= train_cutoff) & (
            pd.to_datetime(ordered["date"]).values <= val_last
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

    seasons = sorted(pd.to_numeric(frame["season"], errors="raise").astype(int).unique().tolist())
    if len(seasons) < 2:
        raise ValueError("At least two seasons are required")

    if first_validation_season is None:
        first_validation_season = seasons[1]

    season_values = pd.to_numeric(frame["season"], errors="raise").astype(int).to_numpy()

    oof = np.full(len(frame), np.nan, dtype=float)

    for validation_season in seasons:
        if validation_season < first_validation_season:
            continue

        train_mask = season_values < validation_season
        validation_mask = season_values == validation_season

        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            continue

        bundle = fit_minutes_model(frame.loc[train_mask], params=params)

        oof[validation_mask] = bundle.predict(frame.loc[validation_mask])

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

    seasons = sorted(pd.to_numeric(frame["season"], errors="raise").astype(int).unique().tolist())
    if len(seasons) < 2:
        raise ValueError("At least two seasons are required")

    if first_validation_season is None:
        first_validation_season = seasons[1]

    season_values = pd.to_numeric(frame["season"], errors="raise").astype(int).to_numpy()

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

        bundle = fit_target_model(frame.loc[train_mask], target=target, params=params)

        oof[validation_mask] = bundle.predict(frame.loc[validation_mask])

    return oof


# From src/nba_prop_quant/distributions.py


def _nb_p(mu: np.ndarray, size: float) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    size = max(float(size), 1e-9)
    return size / (size + mu)


@dataclass
class PoissonCalibrator:
    def fit(self, y: np.ndarray, mu: np.ndarray) -> "PoissonCalibrator":
        return self

    def logpmf(
        self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None
    ) -> np.ndarray:
        return poisson.logpmf(
            np.asarray(y, dtype=int), np.clip(np.asarray(mu, dtype=float), 1e-12, None)
        )

    def pmf(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None) -> np.ndarray:
        return np.exp(self.logpmf(y, mu, frame))

    def cdf(self, k: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None) -> np.ndarray:
        k = np.asarray(k)
        result = poisson.cdf(k, np.clip(np.asarray(mu, dtype=float), 1e-12, None))
        return np.where(k < 0, 0.0, result)

    def ppf(self, u: np.ndarray, mu: float, row: pd.Series | None = None) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 1e-10, 1.0 - 1e-10)
        return poisson.ppf(u, max(float(mu), 1e-12)).astype(int)

    def nll(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None) -> float:
        return float(-np.sum(self.logpmf(y, mu, frame)))


@dataclass
class NegativeBinomialCalibrator:
    size: float = 10.0

    def fit(self, y: np.ndarray, mu: np.ndarray) -> "NegativeBinomialCalibrator":
        y = np.asarray(y, dtype=int)
        mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)

        def objective(log_size: float) -> float:
            size = float(np.exp(log_size))
            p = _nb_p(mu, size)
            return float(-np.sum(nbinom.logpmf(y, size, p)))

        result = minimize_scalar(
            objective, bounds=(-6.0, 10.0), method="bounded", options={"xatol": 1e-6}
        )
        self.size = float(np.exp(result.x))
        return self

    def logpmf(
        self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None
    ) -> np.ndarray:
        y = np.asarray(y, dtype=int)
        mu = np.asarray(mu, dtype=float)
        return nbinom.logpmf(y, self.size, _nb_p(mu, self.size))

    def pmf(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None) -> np.ndarray:
        return np.exp(self.logpmf(y, mu, frame))

    def cdf(self, k: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None) -> np.ndarray:
        k = np.asarray(k)
        mu = np.asarray(mu, dtype=float)
        result = nbinom.cdf(k, self.size, _nb_p(mu, self.size))
        return np.where(k < 0, 0.0, result)

    def ppf(self, u: np.ndarray, mu: float, row: pd.Series | None = None) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 1e-10, 1 - 1e-10)
        return nbinom.ppf(u, self.size, _nb_p(np.array([mu]), self.size)[0]).astype(int)

    def nll(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None) -> float:
        return float(-np.sum(self.logpmf(y, mu, frame)))


@dataclass
class ZeroInflatedNegativeBinomialCalibrator:
    inflation_features: list[str]
    size: float = 10.0
    coef_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    l2: float = 0.01

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.reindex(columns=self.inflation_features).to_numpy(dtype=float)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ZINB calibrator has not been fit")
        x = np.where(np.isfinite(x), x, self.mean_)
        z = (x - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(z)), z])

    def _pi(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("ZINB calibrator has not been fit")
        return np.clip(expit(self._design(frame) @ self.coef_), 1e-6, 1 - 1e-6)

    @staticmethod
    def _count_component_mean(mu: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """
        Convert the canonical unconditional mean into the mean of the
        non-structural-zero Negative Binomial count component.

        For a zero-inflated mixture,

            E[Y | x] = (1 - pi(x)) * mu_count(x).

        We require the supplied canonical mean to remain the marginal
        expectation, so

            mu_count(x) = mu(x) / (1 - pi(x)).
        """
        mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
        pi = np.clip(np.asarray(pi, dtype=float), 1e-9, 1.0 - 1e-9)
        return np.clip(mu / (1.0 - pi), 1e-9, None)

    def implied_mean(self, mu: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        """
        Return the unconditional mean implied by the fitted ZINB.

        This should equal the supplied canonical mu up to floating-point
        precision.
        """
        mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
        pi = self._pi(frame)
        count_mu = self._count_component_mean(mu, pi)
        return (1.0 - pi) * count_mu

    def fit(
        self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame
    ) -> "ZeroInflatedNegativeBinomialCalibrator":
        y = np.asarray(y, dtype=int)
        mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
        x = frame.reindex(columns=self.inflation_features).to_numpy(dtype=float)

        self.mean_ = np.nanmean(x, axis=0)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        x = np.where(np.isfinite(x), x, self.mean_)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_ = np.where((self.scale_ > 1e-8) & np.isfinite(self.scale_), self.scale_, 1.0)
        z = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(z)), z])

        observed_zero = np.clip(np.mean(y == 0), 1e-4, 1 - 1e-4)
        initial_intercept = np.log(observed_zero / (1.0 - observed_zero))
        x0 = np.zeros(design.shape[1] + 1, dtype=float)
        x0[0] = initial_intercept
        x0[-1] = np.log(10.0)

        def objective(theta: np.ndarray) -> float:
            coef = theta[:-1]
            size = float(np.exp(theta[-1]))
            pi = np.clip(expit(design @ coef), 1e-6, 1 - 1e-6)
            count_mu = self._count_component_mean(mu, pi)
            p = _nb_p(count_mu, size)
            log_nb = nbinom.logpmf(y, size, p)
            log_nb0 = nbinom.logpmf(np.zeros_like(y), size, p)

            ll = np.empty(len(y), dtype=float)
            zeros = y == 0
            ll[zeros] = logsumexp(
                np.vstack([np.log(pi[zeros]), np.log1p(-pi[zeros]) + log_nb0[zeros]]), axis=0
            )
            ll[~zeros] = np.log1p(-pi[~zeros]) + log_nb[~zeros]
            penalty = self.l2 * float(np.sum(coef[1:] ** 2))
            return float(-np.sum(ll) + penalty)

        result = minimize(objective, x0, method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"ZINB calibration failed: {result.message}")

        self.coef_ = result.x[:-1]
        self.size = float(np.exp(result.x[-1]))
        return self

    def logpmf(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        y = np.asarray(y, dtype=int)
        mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
        pi = self._pi(frame)
        count_mu = self._count_component_mean(mu, pi)
        p = _nb_p(count_mu, self.size)
        log_nb = nbinom.logpmf(y, self.size, p)
        result = np.log1p(-pi) + log_nb
        zeros = y == 0
        if np.any(zeros):
            log_nb0 = nbinom.logpmf(0, self.size, p[zeros])
            result[zeros] = logsumexp(
                np.vstack([np.log(pi[zeros]), np.log1p(-pi[zeros]) + log_nb0]), axis=0
            )
        return result

    def pmf(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.logpmf(y, mu, frame))

    def cdf(self, k: np.ndarray, mu: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        k = np.asarray(k)
        mu = np.asarray(mu, dtype=float)
        pi = self._pi(frame)
        count_mu = self._count_component_mean(mu, pi)
        base = nbinom.cdf(k, self.size, _nb_p(count_mu, self.size))
        result = pi + (1.0 - pi) * base
        return np.where(k < 0, 0.0, result)

    def ppf(self, u: np.ndarray, mu: float, row: pd.Series) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 1e-10, 1 - 1e-10)
        one = row.to_frame().T
        pi = float(self._pi(one)[0])
        adjusted = (u - pi) / max(1.0 - pi, 1e-12)
        out = np.zeros(len(u), dtype=int)
        mask = u > pi
        if np.any(mask):
            count_mu = self._count_component_mean(
                np.array([mu], dtype=float), np.array([pi], dtype=float)
            )[0]
            p = _nb_p(np.array([count_mu]), self.size)[0]
            out[mask] = nbinom.ppf(np.clip(adjusted[mask], 1e-10, 1 - 1e-10), self.size, p).astype(
                int
            )
        return out

    def nll(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame) -> float:
        return float(-np.sum(self.logpmf(y, mu, frame)))


@dataclass
class FittedMarginal:
    kind: str
    model: PoissonCalibrator | NegativeBinomialCalibrator | ZeroInflatedNegativeBinomialCalibrator

    def cdf(self, k: np.ndarray, mu: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        return self.model.cdf(k, mu, frame)

    def pmf(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        return self.model.pmf(y, mu, frame)

    def ppf(self, u: np.ndarray, mu: float, row: pd.Series) -> np.ndarray:
        return self.model.ppf(u, mu, row)

    def over_under_push(self, line: float, mu: float, row: pd.Series) -> tuple[float, float, float]:
        floor_line = int(np.floor(line))
        frame = row.to_frame().T
        cdf_floor = float(self.cdf(np.array([floor_line]), np.array([mu]), frame)[0])

        if float(line).is_integer():
            pmf_line = float(self.pmf(np.array([floor_line]), np.array([mu]), frame)[0])
            under = float(self.cdf(np.array([floor_line - 1]), np.array([mu]), frame)[0])
            over = 1.0 - cdf_floor
            push = pmf_line
            return over, under, push

        over = 1.0 - cdf_floor
        under = cdf_floor
        return over, under, 0.0


def fit_best_marginal(
    y: np.ndarray,
    mu: np.ndarray,
    frame: pd.DataFrame,
    inflation_features: list[str],
    allow_zinb: bool = True,
) -> FittedMarginal:
    nb = NegativeBinomialCalibrator().fit(y, mu)
    nb_nll = nb.nll(y, mu, frame)

    if not allow_zinb or np.mean(np.asarray(y) == 0) < 0.10:
        return FittedMarginal(kind="nb", model=nb)

    zinb = ZeroInflatedNegativeBinomialCalibrator(inflation_features=inflation_features).fit(
        y, mu, frame
    )
    zinb_nll = zinb.nll(y, mu, frame)

    # Require a real NLL improvement so the extra zero-inflation parameters earn their keep.
    if zinb_nll + 2.0 * (len(inflation_features) + 1) < nb_nll:
        return FittedMarginal(kind="zinb", model=zinb)
    return FittedMarginal(kind="nb", model=nb)


# From src/nba_prop_quant/copula.py

DEFAULT_COPULA_TARGETS = ["pts", "reb", "ast", "stl", "blk", "fg3m"]


def _cov_to_corr(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    corr = covariance / np.outer(scale, scale)
    corr = 0.5 * (corr + corr.T)

    # Numerical clipping of pairwise correlations can make a matrix indefinite.
    # Project back to the positive-semidefinite cone before simulation.
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    renorm = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    corr = corr / np.outer(renorm, renorm)
    corr = np.clip(corr, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    return corr


@dataclass
class GaussianCopula:
    targets: list[str] = field(default_factory=lambda: list(DEFAULT_COPULA_TARGETS))
    global_corr: np.ndarray | None = None
    player_corr: dict[int, np.ndarray] = field(default_factory=dict)

    def fit(
        self,
        frame: pd.DataFrame,
        marginals: dict[str, FittedMarginal],
        mu_columns: dict[str, str],
        min_player_games: int = 40,
        shrink_games: float = 100.0,
    ) -> "GaussianCopula":
        z_columns = []

        for target in self.targets:
            y = frame[target].to_numpy(dtype=int)
            mu = frame[mu_columns[target]].to_numpy(dtype=float)
            marginal = marginals[target]

            lower = marginal.cdf(y - 1, mu, frame)
            mass = marginal.pmf(y, mu, frame)
            u = np.clip(lower + 0.5 * mass, 1e-5, 1 - 1e-5)
            z_columns.append(norm.ppf(u))

        z = np.column_stack(z_columns)
        valid = np.all(np.isfinite(z), axis=1)
        if valid.sum() < len(self.targets) + 5:
            raise ValueError("Not enough valid rows to fit Gaussian copula")

        covariance = LedoitWolf().fit(z[valid]).covariance_
        self.global_corr = _cov_to_corr(covariance)

        player_ids = frame["player_id"].to_numpy()
        for player_id in np.unique(player_ids[valid]):
            mask = valid & (player_ids == player_id)
            n = int(mask.sum())
            if n < min_player_games:
                continue
            empirical = np.corrcoef(z[mask], rowvar=False)
            if not np.all(np.isfinite(empirical)):
                continue
            weight = n / (n + shrink_games)
            shrunk = weight * empirical + (1.0 - weight) * self.global_corr
            self.player_corr[int(player_id)] = _cov_to_corr(shrunk)

        return self

    def correlation_for_player(self, player_id: int | None) -> np.ndarray:
        if self.global_corr is None:
            raise RuntimeError("Copula has not been fit")
        if player_id is None:
            return self.global_corr
        return self.player_corr.get(int(player_id), self.global_corr)

    def simulate(
        self,
        row: pd.Series,
        marginals: dict[str, FittedMarginal],
        mu_columns: dict[str, str],
        simulations: int = 20_000,
        seed: int = 73,
    ) -> pd.DataFrame:
        corr = self.correlation_for_player(
            int(row["player_id"]) if pd.notna(row.get("player_id")) else None
        )
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(mean=np.zeros(len(self.targets)), cov=corr, size=simulations)
        u = norm.cdf(z)

        samples: dict[str, np.ndarray] = {}
        for j, target in enumerate(self.targets):
            mu = float(row[mu_columns[target]])
            samples[target] = marginals[target].ppf(u[:, j], mu, row)

        out = pd.DataFrame(samples)
        out["points_rebounds"] = out["pts"] + out["reb"]
        out["points_assists"] = out["pts"] + out["ast"]
        out["rebounds_assists"] = out["reb"] + out["ast"]
        out["points_rebounds_assists"] = out["pts"] + out["reb"] + out["ast"]
        out["stocks"] = out["stl"] + out["blk"]
        return out


# From src/nba_prop_quant/production.py


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def missing_model_features(frame: pd.DataFrame, bundle: ModelBundle) -> list[str]:
    return [feature for feature in bundle.feature_names if feature not in frame.columns]


def assert_model_feature_contract(
    frame: pd.DataFrame, bundle: ModelBundle, label: str, allow_missing: bool = False
) -> list[str]:
    missing = missing_model_features(frame, bundle)

    if missing and not allow_missing:
        raise RuntimeError(
            f"{label} is missing "
            f"{len(missing)} trained feature(s): "
            + ", ".join(missing[:25])
            + (" ..." if len(missing) > 25 else "")
            + ". Refuse to score an external-test slate "
            "with a broken feature contract."
        )

    return missing


def combine_mean_components(
    xgb: np.ndarray, decay: np.ndarray, kalman: np.ndarray, weights: dict[str, float]
) -> np.ndarray:
    xgb = np.asarray(xgb, dtype=float)

    decay = np.asarray(decay, dtype=float)

    kalman = np.asarray(kalman, dtype=float)

    total_weight = float(
        weights.get("xgb", 0.0) + weights.get("decay", 0.0) + weights.get("kalman", 0.0)
    )

    if not np.isclose(total_weight, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("Mean-model production weights " f"must sum to 1; got {total_weight}")

    selected = np.zeros_like(xgb, dtype=float)

    for name, values in [("xgb", xgb), ("decay", decay), ("kalman", kalman)]:
        weight = float(weights.get(name, 0.0))

        if weight <= 1e-12:
            continue

        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} component " "with positive production weight")

        selected += weight * values

    return np.clip(selected, 1e-6, None)


def apply_selected_mean_policy(
    slate: pd.DataFrame, model_dir: Path, allow_missing_features: bool = False
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = slate.copy()

    policy_path = model_dir / "mean_model_selection.json"

    policy = load_json(policy_path)

    missing_by_target: dict[str, list[str]] = {}

    expected_minutes = pd.to_numeric(out["expected_minutes"], errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(expected_minutes).all():
        raise RuntimeError("Non-finite expected_minutes " "before target scoring")

    for target in TARGETS:
        target_policy = policy["targets"][target]

        bundle = ModelBundle.load(model_dir / f"{target}.joblib")

        missing = assert_model_feature_contract(
            out, bundle, label=(f"{target} target model"), allow_missing=(allow_missing_features)
        )

        missing_by_target[target] = missing

        xgb = bundle.predict(out)

        decay_col = f"decay_prior_{target}_rate"

        kalman_col = f"kalman_prior_{target}_rate"

        decay = (
            expected_minutes * pd.to_numeric(out[decay_col], errors="coerce").to_numpy(dtype=float)
            if decay_col in out.columns
            else np.full(len(out), np.nan)
        )

        kalman = (
            expected_minutes * pd.to_numeric(out[kalman_col], errors="coerce").to_numpy(dtype=float)
            if kalman_col in out.columns
            else np.full(len(out), np.nan)
        )

        weights = target_policy["production_weights"]

        selected = combine_mean_components(xgb=xgb, decay=decay, kalman=kalman, weights=weights)

        out[f"mu_xgb_{target}"] = xgb

        out[f"mu_decay_{target}"] = decay

        out[f"mu_kalman_{target}"] = kalman

        out[f"mu_selected_{target}_pre_usage"] = selected

        out[f"mu_selected_{target}"] = selected

        out[f"mean_model_mode_{target}"] = str(target_policy["selected_mode"])

    usage = (
        pd.to_numeric(
            out.get("availability_usage_multiplier", pd.Series(1.0, index=out.index)),
            errors="coerce",
        )
        .fillna(1.0)
        .clip(lower=0.75, upper=1.30)
    )

    # Frozen production-only availability cascade. This is deliberately
    # downstream of the canonical mean-model selection so the stored
    # pre_usage columns retain the exact model-policy projection.
    out["mu_selected_pts"] *= usage**0.55

    out["mu_selected_ast"] *= usage**0.45

    out["mu_selected_fg3m"] *= usage**0.40

    availability_out = (
        pd.to_numeric(out.get("availability_out", pd.Series(0, index=out.index)), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    for target in TARGETS:
        out.loc[availability_out.eq(1), f"mu_selected_{target}"] = 1e-6

        # Backward-compatible alias for existing in-game/explanation code.
        # Production market pricing must use mu_selected_* explicitly.
        out[f"mu_{target}"] = out[f"mu_selected_{target}"]

    return (out, missing_by_target)


def single_stat_quantile_summary(
    row: pd.Series, marginals: dict, mu_columns: dict[str, str]
) -> dict[str, float]:
    result: dict[str, float] = {}

    for target in TARGETS:
        mu = float(row[mu_columns[target]])

        quantiles = marginals[target].ppf(np.array([0.10, 0.50, 0.90], dtype=float), mu, row)

        result[f"{target}_mean"] = mu

        result[f"{target}_p10"] = float(quantiles[0])

        result[f"{target}_p50"] = float(quantiles[1])

        result[f"{target}_p90"] = float(quantiles[2])

    return result


# From src/nba_prop_quant/gate3_v2.py

GATE3_CHANGED_PROPS = frozenset({"assists", "points_assists", "points_rebounds"})

EPS = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _default_artifact_dir() -> Path:
    raise ValueError("Pass the exported study artifacts/role directory explicitly.")


def resolve_gate3_snapshot_dir(default_snapshot_dir: Path) -> Path:
    override = os.getenv("NBA_PROP_GATE3_SNAPSHOT_DIR")

    if override:
        return Path(override).expanduser().resolve()

    return Path(default_snapshot_dir).expanduser().resolve()


def _verify_artifacts(artifact_dir: Path) -> None:
    checksum_path = artifact_dir / "SHA256SUMS.txt"

    if not checksum_path.exists():
        raise RuntimeError("Gate 3 deployment checksums missing: " f"{checksum_path}")

    for raw in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue

        expected, filename = raw.split(None, 1)

        path = artifact_dir / filename.strip()

        if not path.exists():
            raise RuntimeError("Gate 3 deployment artifact missing: " f"{path}")

        actual = _sha256(path)

        if actual != expected:
            raise RuntimeError("Gate 3 deployment artifact hash mismatch: " f"{path.name}")


def load_gate3_runtime(artifact_dir: Path | None = None) -> dict[str, Any]:
    directory = (
        Path(artifact_dir) if artifact_dir is not None else _default_artifact_dir()
    ).resolve()

    _verify_artifacts(directory)

    manifest_path = directory / "deployment_manifest.json"

    parameter_path = directory / "probability_parameters.json"

    seed_path = directory / "role_state_seed.json"

    model_path = directory / "role_minutes_model.joblib"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    params = json.loads(parameter_path.read_text(encoding="utf-8"))

    seed = json.loads(seed_path.read_text(encoding="utf-8"))

    model_payload = joblib.load(model_path)

    required_policy = {
        "points": "frozen_selected_v1",
        "rebounds": "frozen_selected_v1",
        "assists": "v2_role_shock_calibrated",
        "steals": "frozen_selected_v1",
        "blocks": "frozen_selected_v1",
        "threes": "frozen_selected_v1",
        "points_assists": "v2_role_increment",
        "points_rebounds": "v2_role_increment",
        "rebounds_assists": "frozen_selected_v1",
        "points_rebounds_assists": "frozen_selected_v1",
    }

    if manifest.get("gate3_policy") != required_policy:
        raise RuntimeError("Gate 3 deployment policy mismatch")

    lock_commit = str(manifest["gate3_lock_commit"])

    candidate_id = "nba_prop_quant_v2_gate3_" + lock_commit[:12]

    return {
        "artifact_dir": directory,
        "manifest": manifest,
        "probability_parameters": params,
        "role_state_seed": seed,
        "role_model_payload": model_payload,
        "candidate_id": candidate_id,
        "gate3_lock_commit": lock_commit,
        "deployment_manifest_sha256": _sha256(manifest_path),
        "probability_parameters_sha256": _sha256(parameter_path),
        "role_minutes_model_sha256": _sha256(model_path),
        "role_state_seed_sha256": _sha256(seed_path),
    }


def _parse_snapshot_payload(
    payload: dict[str, Any], snapshot_date: str, captured_at: str
) -> dict[str, Any] | None:
    player = payload.get("player") or {}

    team = payload.get("team") or {}

    player_id = player.get("id")

    team_id = team.get("id", player.get("team_id"))

    game_id = payload.get("game_id")

    starter = payload.get("starter")

    if player_id is None or team_id is None or game_id is None or starter is None:
        return None

    return {
        "game_id": int(game_id),
        "player_id": int(player_id),
        "team_id": int(team_id),
        "starter": int(bool(starter)),
        "_date": snapshot_date,
        "_captured_at": str(captured_at),
    }


def _load_prior_lineup_snapshots(snapshot_dir: Path, target_date: str) -> pd.DataFrame:
    lineup_dir = Path(snapshot_dir) / "lineups"

    columns = ["game_id", "player_id", "team_id", "starter", "_date", "_captured_at"]

    if not lineup_dir.exists():
        return pd.DataFrame(columns=columns)

    target = date.fromisoformat(target_date)

    rows: list[dict[str, Any]] = []

    for path in sorted(lineup_dir.glob("*.jsonl")):
        try:
            snapshot_date = date.fromisoformat(path.stem)
        except ValueError:
            continue

        if snapshot_date >= target:
            continue

        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not raw.strip():
                continue

            try:
                envelope = json.loads(raw)
            except json.JSONDecodeError:
                continue

            payload = envelope.get("payload") or {}

            row = _parse_snapshot_payload(
                payload, snapshot_date.isoformat(), str(envelope.get("captured_at", ""))
            )

            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)

    frame = frame.sort_values(["_date", "_captured_at", "game_id", "player_id"])

    frame = frame.drop_duplicates(["game_id", "player_id"], keep="last")

    return frame


def _current_lineup_frame(current_lineups: pd.DataFrame) -> pd.DataFrame:
    columns = ["game_id", "player_id", "team_id", "starter"]

    if current_lineups is None or current_lineups.empty:
        return pd.DataFrame(columns=columns)

    missing = set(columns) - set(current_lineups.columns)

    if missing:
        raise RuntimeError("Current lineup frame missing columns: " f"{sorted(missing)}")

    frame = current_lineups[columns].dropna(subset=["game_id", "player_id", "team_id"]).copy()

    frame["game_id"] = frame["game_id"].astype(int)

    frame["player_id"] = frame["player_id"].astype(int)

    frame["team_id"] = frame["team_id"].astype(int)

    frame["starter"] = pd.to_numeric(frame["starter"], errors="coerce")

    return frame.drop_duplicates(["game_id", "player_id"], keep="last")


def _player_history_map(seed: dict[str, Any], prior: pd.DataFrame) -> dict[int, list[int]]:
    history: dict[int, list[int]] = {}

    for key, value in seed.get("players", {}).items():
        values = [int(x) for x in value.get("starter_history", [])]

        history[int(key)] = values[-10:]

    if not prior.empty:
        ordered = prior.sort_values(["_date", "game_id"])

        for player_id, group in ordered.groupby("player_id", sort=False):
            values = history.get(int(player_id), [])

            values = values + group["starter"].astype(int).tolist()

            history[int(player_id)] = values[-10:]

    return history


def _team_prior_starters(seed: dict[str, Any], prior: pd.DataFrame) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}

    for key, value in seed.get("teams", {}).items():
        result[int(key)] = {int(x) for x in value.get("last_starters", [])}

    if prior.empty:
        return result

    team_games = (
        prior[["team_id", "_date", "game_id"]]
        .drop_duplicates(["team_id", "game_id"])
        .sort_values(["team_id", "_date", "game_id"])
    )

    for team_id, games in team_games.groupby("team_id", sort=False):
        latest = games.iloc[-1]

        game_id = int(latest["game_id"])

        starters = set(
            prior.loc[
                prior["game_id"].eq(game_id)
                & prior["team_id"].eq(int(team_id))
                & prior["starter"].eq(1),
                "player_id",
            ]
            .astype(int)
            .tolist()
        )

        result[int(team_id)] = starters

    return result


def build_current_role_features(
    current_lineups: pd.DataFrame, prior_snapshots: pd.DataFrame, seed: dict[str, Any]
) -> pd.DataFrame:
    current = _current_lineup_frame(current_lineups)

    output_columns = [
        "game_id",
        "player_id",
        "starter",
        "prev_starter",
        "starter_rate5",
        "starter_rate10",
        "starter_surprise",
        "promoted_to_starter",
        "demoted_to_bench",
        "team_starter_overlap",
        "team_new_starters",
        "team_lost_starters",
        "team_lineup_rows",
        "gate3_role_ready",
    ]

    if current.empty:
        return pd.DataFrame(columns=output_columns)

    player_history = _player_history_map(seed, prior_snapshots)

    team_prior = _team_prior_starters(seed, prior_snapshots)

    team_summary = current.groupby(["game_id", "team_id"], as_index=False).agg(
        team_lineup_rows=("player_id", "size"),
        team_starter_count=(
            "starter",
            lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).sum()),
        ),
    )

    current = current.merge(
        team_summary, on=["game_id", "team_id"], how="left", validate="many_to_one"
    )

    rows = []

    for row in current.itertuples(index=False):
        player_id = int(row.player_id)

        team_id = int(row.team_id)

        starter_value = np.nan if pd.isna(row.starter) else int(row.starter)

        history = player_history.get(player_id, [])

        prev_starter = float(history[-1]) if history else np.nan

        starter_rate5 = float(np.mean(history[-5:])) if history else np.nan

        starter_rate10 = float(np.mean(history[-10:])) if history else np.nan

        current_team = current.loc[
            current["game_id"].eq(int(row.game_id))
            & current["team_id"].eq(team_id)
            & current["starter"].eq(1),
            "player_id",
        ]

        current_starters = {int(x) for x in current_team.dropna().tolist()}

        prior_starters = team_prior.get(team_id)

        if prior_starters is None:
            overlap = np.nan
            new_starters = np.nan
            lost_starters = np.nan
        else:
            overlap = float(len(current_starters & prior_starters))

            new_starters = float(len(current_starters - prior_starters))

            lost_starters = float(len(prior_starters - current_starters))

        team_complete = int(row.team_starter_count) == 5

        ready = bool(team_complete and not pd.isna(starter_value))

        rows.append(
            {
                "game_id": int(row.game_id),
                "player_id": player_id,
                "starter": starter_value,
                "prev_starter": prev_starter,
                "starter_rate5": starter_rate5,
                "starter_rate10": starter_rate10,
                "starter_surprise": (
                    starter_value - starter_rate10
                    if (not pd.isna(starter_value) and not pd.isna(starter_rate10))
                    else np.nan
                ),
                "promoted_to_starter": int(ready and starter_value == 1 and prev_starter == 0),
                "demoted_to_bench": int(ready and starter_value == 0 and prev_starter == 1),
                "team_starter_overlap": overlap,
                "team_new_starters": new_starters,
                "team_lost_starters": lost_starters,
                "team_lineup_rows": int(row.team_lineup_rows),
                "gate3_role_ready": int(ready),
            }
        )

    return pd.DataFrame(rows)


def apply_gate3_role_state(
    slate: pd.DataFrame,
    current_lineups: pd.DataFrame,
    *,
    target_date: str,
    snapshot_dir: Path,
    runtime: dict[str, Any],
) -> pd.DataFrame:
    out = slate.copy()

    prior = _load_prior_lineup_snapshots(snapshot_dir, target_date)

    role = build_current_role_features(current_lineups, prior, runtime["role_state_seed"])

    role_columns = [
        "starter",
        "prev_starter",
        "starter_rate5",
        "starter_rate10",
        "starter_surprise",
        "promoted_to_starter",
        "demoted_to_bench",
        "team_starter_overlap",
        "team_new_starters",
        "team_lost_starters",
        "team_lineup_rows",
        "gate3_role_ready",
    ]

    for column in role_columns:
        if column in out.columns:
            out = out.drop(columns=[column])

    if role.empty:
        for column in role_columns:
            out[column] = 0 if column == "gate3_role_ready" else np.nan
    else:
        out = out.merge(role, on=["game_id", "player_id"], how="left", validate="one_to_one")

        out["gate3_role_ready"] = (
            pd.to_numeric(out["gate3_role_ready"], errors="coerce").fillna(0).astype(int)
        )

    payload = runtime["role_model_payload"]

    feature_names = list(payload["feature_names"])

    missing = [feature for feature in feature_names if feature not in out.columns]

    if missing:
        raise RuntimeError("Gate 3 role model missing live features: " f"{missing}")

    expected_minutes = pd.to_numeric(out["expected_minutes"], errors="coerce")

    availability_out = (
        pd.to_numeric(out.get("availability_out", pd.Series(0, index=out.index)), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    ready = (
        out["gate3_role_ready"].eq(1)
        & expected_minutes.notna()
        & expected_minutes.gt(0)
        & availability_out.eq(0)
    )

    out["gate3_role_minutes"] = np.nan

    if ready.any():
        X = out.loc[ready, feature_names].apply(pd.to_numeric, errors="coerce")

        residual = payload["model"].predict(X)

        out.loc[ready, "gate3_role_minutes"] = np.maximum(
            expected_minutes.loc[ready].to_numpy(dtype=float) + residual, 0.0
        )

    out["gate3_role_ratio"] = (
        pd.to_numeric(out["gate3_role_minutes"], errors="coerce") / expected_minutes
    )

    out["gate3_mu_ast"] = (
        pd.to_numeric(out["mu_selected_ast"], errors="coerce") * out["gate3_role_ratio"]
    )

    out["gate3_delta_pts"] = pd.to_numeric(out["mu_selected_pts"], errors="coerce") * (
        out["gate3_role_ratio"] - 1.0
    )

    out["gate3_delta_reb"] = pd.to_numeric(out["mu_selected_reb"], errors="coerce") * (
        out["gate3_role_ratio"] - 1.0
    )

    out["gate3_delta_ast"] = pd.to_numeric(out["mu_selected_ast"], errors="coerce") * (
        out["gate3_role_ratio"] - 1.0
    )

    out["gate3_delta_points_assists"] = out["gate3_delta_pts"] + out["gate3_delta_ast"]

    out["gate3_delta_points_rebounds"] = out["gate3_delta_pts"] + out["gate3_delta_reb"]

    out["gate3_candidate_policy_id"] = runtime["candidate_id"]

    out["gate3_policy_lock_commit"] = runtime["gate3_lock_commit"]

    out["gate3_deployment_manifest_sha256"] = runtime["deployment_manifest_sha256"]

    return out
