import numpy as np
import pandas as pd

from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.distributions import FittedMarginal, NegativeBinomialCalibrator


def test_gaussian_copula_simulates_combo_columns():
    rng = np.random.default_rng(19)
    n = 300
    frame = pd.DataFrame(
        {
            "player_id": np.repeat([1, 2, 3], n // 3),
            "pts": rng.poisson(20, n),
            "reb": rng.poisson(7, n),
            "ast": rng.poisson(5, n),
            "stl": rng.poisson(1, n),
            "blk": rng.poisson(1, n),
            "fg3m": rng.poisson(2, n),
        }
    )
    targets = ["pts", "reb", "ast", "stl", "blk", "fg3m"]
    marginals = {}
    mu_columns = {}

    for target in targets:
        frame[f"mu_{target}"] = max(float(frame[target].mean()), 0.1)
        mu_columns[target] = f"mu_{target}"
        nb = NegativeBinomialCalibrator().fit(
            frame[target].to_numpy(),
            frame[f"mu_{target}"].to_numpy(),
        )
        marginals[target] = FittedMarginal(kind="nb", model=nb)

    copula = GaussianCopula(targets=targets).fit(
        frame,
        marginals=marginals,
        mu_columns=mu_columns,
        min_player_games=500,
    )
    samples = copula.simulate(
        frame.iloc[0],
        marginals=marginals,
        mu_columns=mu_columns,
        simulations=1000,
    )

    assert "points_rebounds_assists" in samples.columns
    assert "stocks" in samples.columns
    assert len(samples) == 1000
