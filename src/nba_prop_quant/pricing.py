from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .copula import GaussianCopula
from .distributions import FittedMarginal


PROP_TO_TARGET = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "threes": "fg3m",
}

COMBO_COLUMNS = {
    "points_rebounds": "points_rebounds",
    "points_assists": "points_assists",
    "rebounds_assists": "rebounds_assists",
    "points_rebounds_assists": "points_rebounds_assists",
    "stocks": "stocks",
}

COMBO_COMPONENTS = {
    "points_rebounds": ("pts", "reb"),
    "points_assists": ("pts", "ast"),
    "rebounds_assists": ("reb", "ast"),
    "points_rebounds_assists": ("pts", "reb", "ast"),
    "stocks": ("stl", "blk"),
}


def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_implied_probability(odds: int | float) -> float:
    return 1.0 / american_to_decimal(odds)


def devig_two_way(
    over_odds: int | float,
    under_odds: int | float,
) -> tuple[float, float]:
    over = american_implied_probability(over_odds)
    under = american_implied_probability(under_odds)
    total = over + under
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid two-way market probabilities")
    return over / total, under / total


def empirical_over_under_push(
    values: np.ndarray,
    line: float,
) -> tuple[float, float, float]:
    values = np.asarray(values)
    return (
        float(np.mean(values > line)),
        float(np.mean(values < line)),
        float(np.mean(values == line)),
    )


def fair_american(probability: float) -> int:
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    if probability >= 0.5:
        return int(round(-100.0 * probability / (1.0 - probability)))
    return int(round(100.0 * (1.0 - probability) / probability))


def conditional_nonpush_probabilities(
    p_over: float | np.ndarray,
    p_under: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    over = np.asarray(p_over, dtype=float)
    under = np.asarray(p_under, dtype=float)
    denominator = over + under

    q_over = np.divide(
        over,
        denominator,
        out=np.full_like(over, np.nan, dtype=float),
        where=denominator > 0,
    )
    q_under = np.divide(
        under,
        denominator,
        out=np.full_like(under, np.nan, dtype=float),
        where=denominator > 0,
    )

    if np.ndim(q_over) == 0:
        return float(q_over), float(q_under)

    return q_over, q_under


def price_single_prop_frame(
    frame: pd.DataFrame,
    target: str,
    marginal: FittedMarginal,
    mu_column: str,
    line_column: str = "line_value",
) -> pd.DataFrame:
    """Vectorized exact pricing for a single-stat discrete prop."""
    if frame.empty:
        return pd.DataFrame(
            index=frame.index,
            columns=[
                "p_over",
                "p_under",
                "p_push",
                "q_over_nonpush",
                "q_under_nonpush",
            ],
        )

    mu = frame[mu_column].to_numpy(dtype=float)
    line = frame[line_column].to_numpy(dtype=float)
    floor_line = np.floor(line).astype(int)

    cdf_floor = marginal.cdf(floor_line, mu, frame)
    is_integer = np.isclose(
        line,
        np.round(line),
        atol=1e-12,
        rtol=0.0,
    )

    p_over = 1.0 - cdf_floor
    p_under = cdf_floor.copy()
    p_push = np.zeros(len(frame), dtype=float)

    if np.any(is_integer):
        integer_frame = frame.loc[is_integer]
        integer_mu = mu[is_integer]
        integer_k = floor_line[is_integer]

        p_push[is_integer] = marginal.pmf(
            integer_k,
            integer_mu,
            integer_frame,
        )
        p_under[is_integer] = marginal.cdf(
            integer_k - 1,
            integer_mu,
            integer_frame,
        )

    q_over, q_under = conditional_nonpush_probabilities(
        p_over,
        p_under,
    )

    return pd.DataFrame(
        {
            "p_over": p_over,
            "p_under": p_under,
            "p_push": p_push,
            "q_over_nonpush": q_over,
            "q_under_nonpush": q_under,
        },
        index=frame.index,
    )


def _minimal_repeated_frame(
    row: pd.Series,
    marginal: FittedMarginal,
    n: int,
) -> pd.DataFrame:
    features = list(
        getattr(marginal.model, "inflation_features", [])
    )
    if not features:
        return pd.DataFrame(index=np.arange(n))

    return pd.DataFrame(
        {
            feature: np.repeat(row.get(feature, np.nan), n)
            for feature in features
        }
    )


def _marginal_pmf_for_row(
    marginal: FittedMarginal,
    mu: float,
    row: pd.Series,
    tail_probability: float = 1e-10,
) -> np.ndarray:
    max_k = int(
        marginal.ppf(
            np.array([1.0 - tail_probability], dtype=float),
            float(mu),
            row,
        )[0]
    )
    max_k = max(max_k, 0)
    support = np.arange(max_k + 1, dtype=int)
    repeated = _minimal_repeated_frame(
        row,
        marginal,
        len(support),
    )

    probabilities = marginal.pmf(
        support,
        np.full(len(support), float(mu), dtype=float),
        repeated,
    )
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        0.0,
        1.0,
    )

    mass = float(probabilities.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Invalid marginal PMF mass")

    probabilities /= mass
    return probabilities


def independent_combo_pmf(
    row: pd.Series,
    prop_type: str,
    marginals: dict[str, FittedMarginal],
    mu_columns: dict[str, str],
) -> np.ndarray:
    if prop_type not in COMBO_COMPONENTS:
        raise KeyError(f"Unsupported combo prop: {prop_type}")

    result = np.array([1.0], dtype=float)

    for target in COMBO_COMPONENTS[prop_type]:
        component = _marginal_pmf_for_row(
            marginals[target],
            float(row[mu_columns[target]]),
            row,
        )
        result = np.convolve(result, component)

    result = np.clip(result, 0.0, None)
    result /= float(result.sum())
    return result


def _pmf_over_under_push(
    pmf: np.ndarray,
    line: float,
) -> tuple[float, float, float]:
    pmf = np.asarray(pmf, dtype=float)
    floor_line = int(math.floor(float(line)))

    if floor_line < 0:
        return 1.0, 0.0, 0.0

    cumulative = np.cumsum(pmf)

    if floor_line >= len(pmf):
        cdf_floor = 1.0
    else:
        cdf_floor = float(cumulative[floor_line])

    if float(line).is_integer():
        k = int(line)
        push = float(pmf[k]) if 0 <= k < len(pmf) else 0.0
        under = (
            float(cumulative[k - 1])
            if k > 0 and k - 1 < len(cumulative)
            else 0.0
        )
        over = max(0.0, 1.0 - under - push)
        return over, under, push

    under = cdf_floor
    over = max(0.0, 1.0 - under)
    return over, under, 0.0


def _shrunk_combo_correlation(
    copula: GaussianCopula,
    row: pd.Series,
    prop_type: str,
    dependence_lambda: float,
) -> np.ndarray:
    dependence_lambda = float(
        np.clip(dependence_lambda, 0.0, 1.0)
    )

    player_id = (
        int(row["player_id"])
        if pd.notna(row.get("player_id"))
        else None
    )

    full_corr = copula.correlation_for_player(player_id)
    target_index = {
        target: index
        for index, target in enumerate(copula.targets)
    }
    indices = [
        target_index[target]
        for target in COMBO_COMPONENTS[prop_type]
    ]
    sub_corr = full_corr[np.ix_(indices, indices)]

    k = len(indices)
    corr = (
        (1.0 - dependence_lambda) * np.eye(k)
        + dependence_lambda * sub_corr
    )
    corr = 0.5 * (corr + corr.T)

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    corr = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    scale = np.sqrt(
        np.clip(np.diag(corr), 1e-12, None)
    )
    corr = corr / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)

    return corr


def simulate_combo_values(
    row: pd.Series,
    prop_type: str,
    marginals: dict[str, FittedMarginal],
    mu_columns: dict[str, str],
    copula: GaussianCopula,
    dependence_lambda: float,
    simulations: int,
    seed: int,
) -> np.ndarray:
    if simulations < 100:
        raise ValueError("simulations must be at least 100")

    components = COMBO_COMPONENTS[prop_type]
    corr = _shrunk_combo_correlation(
        copula,
        row,
        prop_type,
        dependence_lambda,
    )

    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(
        mean=np.zeros(len(components)),
        cov=corr,
        size=int(simulations),
    )
    u = norm.cdf(z)

    total = np.zeros(int(simulations), dtype=int)

    for index, target in enumerate(components):
        mu = float(row[mu_columns[target]])
        total += marginals[target].ppf(
            u[:, index],
            mu,
            row,
        )

    return total


def price_combo_lines(
    row: pd.Series,
    prop_type: str,
    lines: list[float] | np.ndarray,
    marginals: dict[str, FittedMarginal],
    mu_columns: dict[str, str],
    copula: GaussianCopula,
    dependence_lambda: float,
    simulations: int = 20_000,
    seed: int = 73,
) -> pd.DataFrame:
    lines = np.asarray(lines, dtype=float)

    if dependence_lambda <= 1e-12:
        pmf = independent_combo_pmf(
            row,
            prop_type,
            marginals,
            mu_columns,
        )
        probabilities = [
            _pmf_over_under_push(pmf, float(line))
            for line in lines
        ]
    else:
        values = simulate_combo_values(
            row=row,
            prop_type=prop_type,
            marginals=marginals,
            mu_columns=mu_columns,
            copula=copula,
            dependence_lambda=dependence_lambda,
            simulations=simulations,
            seed=seed,
        )
        probabilities = [
            empirical_over_under_push(values, float(line))
            for line in lines
        ]

    output = pd.DataFrame(
        probabilities,
        columns=["p_over", "p_under", "p_push"],
    )
    output["line_value"] = lines

    q_over, q_under = conditional_nonpush_probabilities(
        output["p_over"].to_numpy(dtype=float),
        output["p_under"].to_numpy(dtype=float),
    )
    output["q_over_nonpush"] = q_over
    output["q_under_nonpush"] = q_under
    output["dependence_lambda"] = float(dependence_lambda)

    return output


def price_player_prop(
    row: pd.Series,
    prop_type: str,
    line: float,
    marginals: dict[str, FittedMarginal],
    mu_columns: dict[str, str],
    copula: GaussianCopula | None = None,
    simulations: int = 20_000,
    dependence_lambda: float = 1.0,
    seed: int = 73,
) -> dict[str, float | int | str]:
    if prop_type in PROP_TO_TARGET:
        target = PROP_TO_TARGET[prop_type]
        marginal = marginals[target]
        mu = float(row[mu_columns[target]])
        over, under, push = marginal.over_under_push(
            line,
            mu,
            row,
        )

    elif prop_type in COMBO_COLUMNS:
        if copula is None:
            raise ValueError(
                "A fitted copula is required for combo props"
            )

        combo = price_combo_lines(
            row=row,
            prop_type=prop_type,
            lines=[float(line)],
            marginals=marginals,
            mu_columns=mu_columns,
            copula=copula,
            dependence_lambda=dependence_lambda,
            simulations=simulations,
            seed=seed,
        ).iloc[0]

        over = float(combo["p_over"])
        under = float(combo["p_under"])
        push = float(combo["p_push"])

    else:
        raise KeyError(
            f"Unsupported prop_type: {prop_type}"
        )

    q_over, q_under = conditional_nonpush_probabilities(
        over,
        under,
    )

    return {
        "prop_type": prop_type,
        "line": float(line),
        "p_over": over,
        "p_under": under,
        "p_push": push,
        "q_over_nonpush": q_over,
        "q_under_nonpush": q_under,
        "dependence_lambda": (
            float(dependence_lambda)
            if prop_type in COMBO_COLUMNS
            else 0.0
        ),
        "fair_over_american": fair_american(q_over),
        "fair_under_american": fair_american(q_under),
    }


def projection_summary(
    row: pd.Series,
    marginals: dict[str, FittedMarginal],
    mu_columns: dict[str, str],
    copula: GaussianCopula,
    simulations: int = 20_000,
) -> dict[str, float]:
    samples = copula.simulate(
        row=row,
        marginals=marginals,
        mu_columns=mu_columns,
        simulations=simulations,
    )

    result: dict[str, float] = {}

    for col in samples.columns:
        values = samples[col].to_numpy()
        result[f"{col}_mean"] = float(np.mean(values))
        result[f"{col}_p10"] = float(np.quantile(values, 0.10))
        result[f"{col}_p50"] = float(np.quantile(values, 0.50))
        result[f"{col}_p90"] = float(np.quantile(values, 0.90))

    return result
