import numpy as np
import pandas as pd

from nba_prop_quant.distributions import (
    NegativeBinomialCalibrator,
    fit_best_marginal,
)


def test_negative_binomial_probabilities_are_valid():
    rng = np.random.default_rng(7)
    mu = np.full(500, 8.0)
    size = 4.0
    p = size / (size + mu)
    y = rng.negative_binomial(size, p)

    model = NegativeBinomialCalibrator().fit(y, mu)
    frame = pd.DataFrame({"expected_minutes": np.full(len(y), 30.0)})

    cdf = model.cdf(np.array([0, 5, 20]), np.array([8.0, 8.0, 8.0]), frame)
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))
    assert np.all(np.diff(cdf) >= 0.0)


def test_zero_heavy_data_can_be_calibrated():
    rng = np.random.default_rng(11)
    n = 800
    minutes = rng.uniform(8, 36, size=n)
    mu = 0.02 * minutes
    structural = rng.random(n) < (0.55 - 0.01 * minutes)
    base = rng.poisson(mu)
    y = np.where(structural, 0, base)

    frame = pd.DataFrame(
        {
            "expected_minutes": minutes,
            "days_rest": rng.integers(0, 4, size=n),
        }
    )
    fitted = fit_best_marginal(
        y=y,
        mu=np.clip(mu, 1e-4, None),
        frame=frame,
        inflation_features=["expected_minutes", "days_rest"],
        allow_zinb=True,
    )

    probs = fitted.pmf(
        np.array([0, 1]),
        np.array([0.5, 0.5]),
        frame.iloc[:2],
    )
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))
