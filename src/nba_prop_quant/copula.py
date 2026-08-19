from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from .distributions import FittedMarginal


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
        z = rng.multivariate_normal(
            mean=np.zeros(len(self.targets)),
            cov=corr,
            size=simulations,
        )
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
