from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import nbinom, poisson


def _nb_p(mu: np.ndarray, size: float) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    size = max(float(size), 1e-9)
    return size / (size + mu)


@dataclass
class PoissonCalibrator:
    def fit(
        self,
        y: np.ndarray,
        mu: np.ndarray,
    ) -> "PoissonCalibrator":
        return self

    def logpmf(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        return poisson.logpmf(
            np.asarray(y, dtype=int),
            np.clip(
                np.asarray(mu, dtype=float),
                1e-12,
                None,
            ),
        )

    def pmf(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        return np.exp(
            self.logpmf(
                y,
                mu,
                frame,
            )
        )

    def cdf(
        self,
        k: np.ndarray,
        mu: np.ndarray,
        frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        k = np.asarray(k)
        result = poisson.cdf(
            k,
            np.clip(
                np.asarray(mu, dtype=float),
                1e-12,
                None,
            ),
        )
        return np.where(
            k < 0,
            0.0,
            result,
        )

    def ppf(
        self,
        u: np.ndarray,
        mu: float,
        row: pd.Series | None = None,
    ) -> np.ndarray:
        u = np.clip(
            np.asarray(u, dtype=float),
            1e-10,
            1.0 - 1e-10,
        )
        return poisson.ppf(
            u,
            max(
                float(mu),
                1e-12,
            ),
        ).astype(int)

    def nll(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        frame: pd.DataFrame | None = None,
    ) -> float:
        return float(
            -np.sum(
                self.logpmf(
                    y,
                    mu,
                    frame,
                )
            )
        )


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
            objective,
            bounds=(-6.0, 10.0),
            method="bounded",
            options={"xatol": 1e-6},
        )
        self.size = float(np.exp(result.x))
        return self

    def logpmf(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame | None = None) -> np.ndarray:
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
    def _count_component_mean(
        mu: np.ndarray,
        pi: np.ndarray,
    ) -> np.ndarray:
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

    def implied_mean(
        self,
        mu: np.ndarray,
        frame: pd.DataFrame,
    ) -> np.ndarray:
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
        self,
        y: np.ndarray,
        mu: np.ndarray,
        frame: pd.DataFrame,
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
                np.vstack(
                    [
                        np.log(pi[zeros]),
                        np.log1p(-pi[zeros]) + log_nb0[zeros],
                    ]
                ),
                axis=0,
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
                np.vstack(
                    [
                        np.log(pi[zeros]),
                        np.log1p(-pi[zeros]) + log_nb0,
                    ]
                ),
                axis=0,
            )
        return result

    def pmf(self, y: np.ndarray, mu: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.logpmf(y, mu, frame))

    def cdf(self, k: np.ndarray, mu: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        k = np.asarray(k)
        mu = np.asarray(mu, dtype=float)
        pi = self._pi(frame)
        count_mu = self._count_component_mean(mu, pi)
        base = nbinom.cdf(
            k,
            self.size,
            _nb_p(count_mu, self.size),
        )
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
                np.array([mu], dtype=float),
                np.array([pi], dtype=float),
            )[0]
            p = _nb_p(
                np.array([count_mu]),
                self.size,
            )[0]
            out[mask] = nbinom.ppf(
                np.clip(adjusted[mask], 1e-10, 1 - 1e-10),
                self.size,
                p,
            ).astype(int)
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

    def over_under_push(
        self,
        line: float,
        mu: float,
        row: pd.Series,
    ) -> tuple[float, float, float]:
        floor_line = int(np.floor(line))
        frame = row.to_frame().T
        cdf_floor = float(self.cdf(np.array([floor_line]), np.array([mu]), frame)[0])

        if float(line).is_integer():
            pmf_line = float(
                self.pmf(np.array([floor_line]), np.array([mu]), frame)[0]
            )
            under = float(
                self.cdf(np.array([floor_line - 1]), np.array([mu]), frame)[0]
            )
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

    zinb = ZeroInflatedNegativeBinomialCalibrator(
        inflation_features=inflation_features
    ).fit(y, mu, frame)
    zinb_nll = zinb.nll(y, mu, frame)

    # Require a real NLL improvement so the extra zero-inflation parameters earn their keep.
    if zinb_nll + 2.0 * (len(inflation_features) + 1) < nb_nll:
        return FittedMarginal(kind="zinb", model=zinb)
    return FittedMarginal(kind="nb", model=nb)
