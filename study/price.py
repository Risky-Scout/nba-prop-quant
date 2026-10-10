"""3. Convert distributions into over/under/push probabilities and fair prices."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from .model import EPS, GATE3_CHANGED_PROPS, FittedMarginal, GaussianCopula

MONITORING_EDGE_GRID = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)

# From src/nba_prop_quant/pricing.py

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


def devig_two_way(over_odds: int | float, under_odds: int | float) -> tuple[float, float]:
    over = american_implied_probability(over_odds)
    under = american_implied_probability(under_odds)
    total = over + under
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid two-way market probabilities")
    return over / total, under / total


def empirical_over_under_push(values: np.ndarray, line: float) -> tuple[float, float, float]:
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
    p_over: float | np.ndarray, p_under: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    over = np.asarray(p_over, dtype=float)
    under = np.asarray(p_under, dtype=float)
    denominator = over + under

    q_over = np.divide(
        over, denominator, out=np.full_like(over, np.nan, dtype=float), where=denominator > 0
    )
    q_under = np.divide(
        under, denominator, out=np.full_like(under, np.nan, dtype=float), where=denominator > 0
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
            columns=["p_over", "p_under", "p_push", "q_over_nonpush", "q_under_nonpush"],
        )

    mu = frame[mu_column].to_numpy(dtype=float)
    line = frame[line_column].to_numpy(dtype=float)
    floor_line = np.floor(line).astype(int)

    cdf_floor = marginal.cdf(floor_line, mu, frame)
    is_integer = np.isclose(line, np.round(line), atol=1e-12, rtol=0.0)

    p_over = 1.0 - cdf_floor
    p_under = cdf_floor.copy()
    p_push = np.zeros(len(frame), dtype=float)

    if np.any(is_integer):
        integer_frame = frame.loc[is_integer]
        integer_mu = mu[is_integer]
        integer_k = floor_line[is_integer]

        p_push[is_integer] = marginal.pmf(integer_k, integer_mu, integer_frame)
        p_under[is_integer] = marginal.cdf(integer_k - 1, integer_mu, integer_frame)

    q_over, q_under = conditional_nonpush_probabilities(p_over, p_under)

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


def _minimal_repeated_frame(row: pd.Series, marginal: FittedMarginal, n: int) -> pd.DataFrame:
    features = list(getattr(marginal.model, "inflation_features", []))
    if not features:
        return pd.DataFrame(index=np.arange(n))

    return pd.DataFrame({feature: np.repeat(row.get(feature, np.nan), n) for feature in features})


def _marginal_pmf_for_row(
    marginal: FittedMarginal, mu: float, row: pd.Series, tail_probability: float = 1e-10
) -> np.ndarray:
    max_k = int(marginal.ppf(np.array([1.0 - tail_probability], dtype=float), float(mu), row)[0])
    max_k = max(max_k, 0)
    support = np.arange(max_k + 1, dtype=int)
    repeated = _minimal_repeated_frame(row, marginal, len(support))

    probabilities = marginal.pmf(support, np.full(len(support), float(mu), dtype=float), repeated)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)

    mass = float(probabilities.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Invalid marginal PMF mass")

    probabilities /= mass
    return probabilities


def independent_combo_pmf(
    row: pd.Series, prop_type: str, marginals: dict[str, FittedMarginal], mu_columns: dict[str, str]
) -> np.ndarray:
    if prop_type not in COMBO_COMPONENTS:
        raise KeyError(f"Unsupported combo prop: {prop_type}")

    result = np.array([1.0], dtype=float)

    for target in COMBO_COMPONENTS[prop_type]:
        component = _marginal_pmf_for_row(marginals[target], float(row[mu_columns[target]]), row)
        result = np.convolve(result, component)

    result = np.clip(result, 0.0, None)
    result /= float(result.sum())
    return result


def _pmf_over_under_push(pmf: np.ndarray, line: float) -> tuple[float, float, float]:
    pmf = np.asarray(pmf, dtype=float)
    floor_line = int(math.floor(float(line)))

    if floor_line < 0:
        return 1.0, 0.0, 0.0

    cumulative = np.cumsum(pmf)

    if floor_line >= len(pmf):
        # Every represented outcome is below this line, including integer lines.
        return (0.0, 1.0, 0.0)
    else:
        cdf_floor = float(cumulative[floor_line])

    if float(line).is_integer():
        k = int(line)
        push = float(pmf[k]) if 0 <= k < len(pmf) else 0.0
        under = float(cumulative[k - 1]) if k > 0 and k - 1 < len(cumulative) else 0.0
        over = max(0.0, 1.0 - under - push)
        return over, under, push

    under = cdf_floor
    over = max(0.0, 1.0 - under)
    return over, under, 0.0


def _shrunk_combo_correlation(
    copula: GaussianCopula, row: pd.Series, prop_type: str, dependence_lambda: float
) -> np.ndarray:
    dependence_lambda = float(np.clip(dependence_lambda, 0.0, 1.0))

    player_id = int(row["player_id"]) if pd.notna(row.get("player_id")) else None

    full_corr = copula.correlation_for_player(player_id)
    target_index = {target: index for index, target in enumerate(copula.targets)}
    indices = [target_index[target] for target in COMBO_COMPONENTS[prop_type]]
    sub_corr = full_corr[np.ix_(indices, indices)]

    k = len(indices)
    corr = (1.0 - dependence_lambda) * np.eye(k) + dependence_lambda * sub_corr
    corr = 0.5 * (corr + corr.T)

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    scale = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
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
    corr = _shrunk_combo_correlation(copula, row, prop_type, dependence_lambda)

    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(mean=np.zeros(len(components)), cov=corr, size=int(simulations))
    u = norm.cdf(z)

    total = np.zeros(int(simulations), dtype=int)

    for index, target in enumerate(components):
        mu = float(row[mu_columns[target]])
        total += marginals[target].ppf(u[:, index], mu, row)

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
        pmf = independent_combo_pmf(row, prop_type, marginals, mu_columns)
        probabilities = [_pmf_over_under_push(pmf, float(line)) for line in lines]
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
        probabilities = [empirical_over_under_push(values, float(line)) for line in lines]

    output = pd.DataFrame(probabilities, columns=["p_over", "p_under", "p_push"])
    output["line_value"] = lines

    q_over, q_under = conditional_nonpush_probabilities(
        output["p_over"].to_numpy(dtype=float), output["p_under"].to_numpy(dtype=float)
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
        over, under, push = marginal.over_under_push(line, mu, row)

    elif prop_type in COMBO_COLUMNS:
        if copula is None:
            raise ValueError("A fitted copula is required for combo props")

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
        raise KeyError(f"Unsupported prop_type: {prop_type}")

    q_over, q_under = conditional_nonpush_probabilities(over, under)

    return {
        "prop_type": prop_type,
        "line": float(line),
        "p_over": over,
        "p_under": under,
        "p_push": push,
        "q_over_nonpush": q_over,
        "q_under_nonpush": q_under,
        "dependence_lambda": (float(dependence_lambda) if prop_type in COMBO_COLUMNS else 0.0),
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
        row=row, marginals=marginals, mu_columns=mu_columns, simulations=simulations
    )

    result: dict[str, float] = {}

    for col in samples.columns:
        values = samples[col].to_numpy()
        result[f"{col}_mean"] = float(np.mean(values))
        result[f"{col}_p10"] = float(np.quantile(values, 0.10))
        result[f"{col}_p50"] = float(np.quantile(values, 0.50))
        result[f"{col}_p90"] = float(np.quantile(values, 0.90))

    return result


# From src/nba_prop_quant/production.py


def calibrate_over_probability(
    raw_q_over: float | np.ndarray, prop_type: str, calibration_policy: dict[str, Any]
) -> tuple[np.ndarray, str, float | None, float | None]:
    if prop_type not in calibration_policy["props"]:
        raise KeyError("No frozen market-probability " f"calibration policy for {prop_type}")

    entry = calibration_policy["props"][prop_type]

    method = str(entry["selected_method"])

    raw = np.clip(np.asarray(raw_q_over, dtype=float), 1e-6, 1.0 - 1e-6)

    if method == "raw":
        return (raw, method, None, None)

    if method not in {"prop", "global"}:
        raise ValueError(f"Unknown calibration method: " f"{method}")

    parameters = entry.get("production_parameters")

    if not parameters:
        raise ValueError(
            f"{prop_type}: calibration method " f"{method} has no production parameters"
        )

    intercept = float(parameters["intercept"])

    slope = float(parameters["slope"])

    logit = np.log(raw / (1.0 - raw))

    calibrated = expit(intercept + slope * logit)

    return (calibrated, method, intercept, slope)


def calibrated_unconditional_probabilities(
    q_over: float | np.ndarray, p_push: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    q_over = np.clip(np.asarray(q_over, dtype=float), 0.0, 1.0)

    p_push = np.clip(np.asarray(p_push, dtype=float), 0.0, 1.0)

    nonpush_mass = 1.0 - p_push

    p_over = nonpush_mass * q_over

    p_under = nonpush_mass * (1.0 - q_over)

    return (p_over, p_under)


def add_market_probability_layer(
    frame: pd.DataFrame, calibration_policy: dict[str, Any]
) -> pd.DataFrame:
    out = frame.copy()

    required = {
        "prop_type",
        "p_over",
        "p_under",
        "p_push",
        "q_over_nonpush",
        "q_under_nonpush",
        "over_odds",
        "under_odds",
    }

    missing = required - set(out.columns)

    if missing:
        raise ValueError("Market probability layer missing " f"columns: {sorted(missing)}")

    out["raw_p_over"] = pd.to_numeric(out["p_over"], errors="coerce")

    out["raw_p_under"] = pd.to_numeric(out["p_under"], errors="coerce")

    out["raw_p_push"] = pd.to_numeric(out["p_push"], errors="coerce")

    out["raw_q_over_nonpush"] = pd.to_numeric(out["q_over_nonpush"], errors="coerce")

    out["raw_q_under_nonpush"] = pd.to_numeric(out["q_under_nonpush"], errors="coerce")

    out["calibrated_q_over_nonpush"] = np.nan

    out["calibration_method"] = ""

    out["calibration_intercept"] = np.nan

    out["calibration_slope"] = np.nan

    for prop_type in sorted(out["prop_type"].dropna().unique()):
        mask = out["prop_type"].eq(prop_type)

        calibrated, method, intercept, slope = calibrate_over_probability(
            out.loc[mask, "raw_q_over_nonpush"].to_numpy(dtype=float),
            str(prop_type),
            calibration_policy,
        )

        out.loc[mask, "calibrated_q_over_nonpush"] = calibrated

        out.loc[mask, "calibration_method"] = method

        if intercept is not None:
            out.loc[mask, "calibration_intercept"] = intercept

        if slope is not None:
            out.loc[mask, "calibration_slope"] = slope

    out["base_calibrated_q_over_nonpush"] = out["calibrated_q_over_nonpush"]

    out["base_calibration_method"] = out["calibration_method"]

    out["base_calibration_intercept"] = out["calibration_intercept"]

    out["base_calibration_slope"] = out["calibration_slope"]

    out["gate3_candidate_applied"] = False

    if "gate3_candidate_q_over_nonpush" in out.columns:
        candidate = pd.to_numeric(out["gate3_candidate_q_over_nonpush"], errors="coerce")

        override = candidate.notna()

        if override.any():
            values = candidate.loc[override]

            if (
                (values < 0.0).any()
                or (values > 1.0).any()
                or not np.isfinite(values.to_numpy(dtype=float)).all()
            ):
                raise RuntimeError("Invalid Gate 3 candidate probability override")

            out.loc[override, "calibrated_q_over_nonpush"] = values

            out.loc[override, "gate3_candidate_applied"] = True

            if "gate3_candidate_method" in out.columns:
                out.loc[override, "calibration_method"] = out.loc[
                    override, "gate3_candidate_method"
                ]

            if "gate3_candidate_intercept" in out.columns:
                intercept_values = pd.to_numeric(
                    out.loc[override, "gate3_candidate_intercept"], errors="coerce"
                )

                valid = intercept_values.notna()

                if valid.any():
                    target_index = intercept_values.loc[valid].index

                    out.loc[target_index, "calibration_intercept"] = intercept_values.loc[valid]

            if "gate3_candidate_slope" in out.columns:
                slope_values = pd.to_numeric(
                    out.loc[override, "gate3_candidate_slope"], errors="coerce"
                )

                valid = slope_values.notna()

                if valid.any():
                    target_index = slope_values.loc[valid].index

                    out.loc[target_index, "calibration_slope"] = slope_values.loc[valid]

    out["calibrated_q_under_nonpush"] = 1.0 - out["calibrated_q_over_nonpush"]

    calibrated_p_over, calibrated_p_under = calibrated_unconditional_probabilities(
        out["calibrated_q_over_nonpush"].to_numpy(dtype=float),
        out["raw_p_push"].to_numpy(dtype=float),
    )

    out["calibrated_p_over"] = calibrated_p_over

    out["calibrated_p_under"] = calibrated_p_under

    out["calibrated_p_push"] = out["raw_p_push"]

    over_implied = np.array(
        [american_implied_probability(value) for value in out["over_odds"].to_numpy()], dtype=float
    )

    under_implied = np.array(
        [american_implied_probability(value) for value in out["under_odds"].to_numpy()], dtype=float
    )

    probability_sum = over_implied + under_implied

    out["market_raw_implied_over"] = over_implied

    out["market_raw_implied_under"] = under_implied

    out["market_hold_pct"] = (probability_sum - 1.0) * 100.0

    out["market_devig_q_over"] = over_implied / probability_sum

    out["market_devig_q_under"] = under_implied / probability_sum

    out["raw_edge_over"] = out["raw_q_over_nonpush"] - out["market_devig_q_over"]

    out["raw_edge_under"] = out["raw_q_under_nonpush"] - out["market_devig_q_under"]

    out["calibrated_edge_over"] = out["calibrated_q_over_nonpush"] - out["market_devig_q_over"]

    out["calibrated_edge_under"] = out["calibrated_q_under_nonpush"] - out["market_devig_q_under"]

    over_decimal = np.array(
        [american_to_decimal(value) for value in out["over_odds"].to_numpy()], dtype=float
    )

    under_decimal = np.array(
        [american_to_decimal(value) for value in out["under_odds"].to_numpy()], dtype=float
    )

    out["calibrated_ev_over"] = (
        out["calibrated_p_over"] * (over_decimal - 1.0) - out["calibrated_p_under"]
    )

    out["calibrated_ev_under"] = (
        out["calibrated_p_under"] * (under_decimal - 1.0) - out["calibrated_p_over"]
    )

    choose_over = out["calibrated_ev_over"] >= out["calibrated_ev_under"]

    out["model_preferred_side"] = np.where(choose_over, "over", "under")

    out["model_preferred_edge"] = np.where(
        choose_over, out["calibrated_edge_over"], out["calibrated_edge_under"]
    )

    out["model_preferred_ev"] = np.where(
        choose_over, out["calibrated_ev_over"], out["calibrated_ev_under"]
    )

    out["calibrated_fair_over_american"] = [
        fair_american(value) for value in out["calibrated_q_over_nonpush"].to_numpy(dtype=float)
    ]

    out["calibrated_fair_under_american"] = [
        fair_american(value) for value in out["calibrated_q_under_nonpush"].to_numpy(dtype=float)
    ]

    for threshold in MONITORING_EDGE_GRID:
        label = str(threshold).replace(".", "_")

        out[f"monitor_edge_ge_{label}"] = out["model_preferred_edge"] >= threshold

    out["auto_bet"] = False

    out["betting_threshold_policy"] = "monitor_only_no_threshold_frozen"

    return out


# From src/nba_prop_quant/gate3_v2.py


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)

    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -40.0, 40.0)

    return 1.0 / (1.0 + np.exp(-value))


def _frozen_calibrate(
    raw_q: np.ndarray, prop_type: str, calibration_policy: dict[str, Any]
) -> np.ndarray:
    entry = calibration_policy["props"][prop_type]

    method = str(entry["selected_method"])

    raw = np.clip(np.asarray(raw_q, dtype=float), EPS, 1.0 - EPS)

    if method == "raw":
        return raw

    if method not in {"prop", "global"}:
        raise RuntimeError("Unsupported frozen calibration method " f"for {prop_type}: {method}")

    parameters = entry.get("production_parameters")

    if not parameters:
        raise RuntimeError("Missing frozen calibration parameters " f"for {prop_type}")

    intercept = float(parameters["intercept"])

    slope = float(parameters["slope"])

    return _sigmoid(intercept + slope * _logit(raw))


def prepare_gate3_candidate_probability_overrides(
    frame: pd.DataFrame,
    *,
    calibration_policy: dict[str, Any],
    probability_parameters: dict[str, Any],
) -> pd.DataFrame:
    out = frame.copy()

    out["gate3_candidate_q_over_nonpush"] = np.nan

    out["gate3_candidate_method"] = "frozen_selected_v1"

    out["gate3_candidate_gamma"] = np.nan

    out["gate3_candidate_standardization_mean"] = np.nan

    out["gate3_candidate_standardization_std"] = np.nan

    out["gate3_candidate_intercept"] = np.nan

    out["gate3_candidate_slope"] = np.nan

    changed = out["prop_type"].isin(GATE3_CHANGED_PROPS)

    if changed.any():
        if "gate3_role_ready" not in out.columns:
            raise RuntimeError("Gate 3 changed prop rows missing " "gate3_role_ready")

        ready = pd.to_numeric(out["gate3_role_ready"], errors="coerce").fillna(0).astype(int).eq(1)

        if (changed & ~ready).any():
            raise RuntimeError("Gate 3 changed prop reached pricing " "without ready role state")

    assists_mask = out["prop_type"].eq("assists")

    if assists_mask.any():
        params = probability_parameters["assists"]

        raw = pd.to_numeric(out.loc[assists_mask, "q_over_nonpush"], errors="coerce").to_numpy(
            dtype=float
        )

        candidate = _sigmoid(float(params["intercept"]) + float(params["slope"]) * _logit(raw))

        out.loc[assists_mask, "gate3_candidate_q_over_nonpush"] = candidate

        out.loc[assists_mask, "gate3_candidate_method"] = "v2_role_shock_calibrated"

        out.loc[assists_mask, "gate3_candidate_intercept"] = float(params["intercept"])

        out.loc[assists_mask, "gate3_candidate_slope"] = float(params["slope"])

    combo_config = {
        "points_assists": ("gate3_delta_points_assists"),
        "points_rebounds": ("gate3_delta_points_rebounds"),
    }

    for prop_type, delta_column in combo_config.items():
        mask = out["prop_type"].eq(prop_type)

        if not mask.any():
            continue

        if delta_column not in out.columns:
            raise RuntimeError(f"{prop_type} missing {delta_column}")

        params = probability_parameters[prop_type]

        raw = pd.to_numeric(out.loc[mask, "q_over_nonpush"], errors="coerce").to_numpy(dtype=float)

        base = _frozen_calibrate(raw, prop_type, calibration_policy)

        delta = pd.to_numeric(out.loc[mask, delta_column], errors="coerce").to_numpy(dtype=float)

        mean = float(params["standardization_mean"])

        std = float(params["standardization_std"])

        gamma = float(params["gamma"])

        if not np.isfinite(std) or std <= 0:
            raise RuntimeError(f"{prop_type}: invalid Gate 3 standardization std")

        z = (delta - mean) / std

        candidate = _sigmoid(_logit(base) + gamma * z)

        out.loc[mask, "gate3_candidate_q_over_nonpush"] = candidate

        out.loc[mask, "gate3_candidate_method"] = "v2_role_increment"

        out.loc[mask, "gate3_candidate_gamma"] = gamma

        out.loc[mask, "gate3_candidate_standardization_mean"] = mean

        out.loc[mask, "gate3_candidate_standardization_std"] = std

    return out


def stable_event_seed(game_id: int, player_id: int, prop_type: str, base_seed: int) -> int:
    prop_code = sum(((index + 1) * ord(char) for index, char in enumerate(prop_type)))
    return int(
        (int(base_seed) + 1000003 * int(game_id) + 9176 * int(player_id) + 37 * prop_code)
        % (2**32 - 1)
    )


def add_quote_filter_fields(
    markets: pd.DataFrame, max_hold_pct: float, supported_props: set[str]
) -> pd.DataFrame:
    out = markets.copy()
    out["line_value"] = pd.to_numeric(out["line_value"], errors="coerce")
    out["over_odds"] = pd.to_numeric(out["over_odds"], errors="coerce")
    out["under_odds"] = pd.to_numeric(out["under_odds"], errors="coerce")
    out["quote_filter_reason"] = ""
    out.loc[~out["market_type"].eq("over_under"), "quote_filter_reason"] = "unsupported_market_type"
    out.loc[
        out["market_type"].eq("over_under") & ~out["prop_type"].isin(supported_props),
        "quote_filter_reason",
    ] = "unsupported_or_unfrozen_prop"
    required_numeric = (
        out["line_value"].notna()
        & out["over_odds"].notna()
        & out["under_odds"].notna()
        & out["over_odds"].ne(0)
        & out["under_odds"].ne(0)
    )
    out.loc[out["quote_filter_reason"].eq("") & ~required_numeric, "quote_filter_reason"] = (
        "missing_or_invalid_line_odds"
    )
    valid_numeric = out["quote_filter_reason"].eq("")
    if valid_numeric.any():
        from nba_prop_quant.pricing import american_implied_probability

        over_imp = np.array(
            [
                american_implied_probability(value)
                for value in out.loc[valid_numeric, "over_odds"].to_numpy()
            ]
        )
        under_imp = np.array(
            [
                american_implied_probability(value)
                for value in out.loc[valid_numeric, "under_odds"].to_numpy()
            ]
        )
        hold = (over_imp + under_imp - 1.0) * 100.0
        out.loc[valid_numeric, "preprice_hold_pct"] = hold
        bad_hold = (hold < 0.0) | (hold > float(max_hold_pct))
        bad_indices = out.loc[valid_numeric].index[bad_hold]
        out.loc[bad_indices, "quote_filter_reason"] = "hold_outside_frozen_range"
    out["quote_eligible"] = out["quote_filter_reason"].eq("")
    return out


# The two public entry points below are the complete study prediction workflow.
# Inputs are already-normalized tables. There are no API calls or scheduler here.


def project_slate(history, games, players, advanced, injuries, lineups, artifacts, snapshots):
    """Historical box scores -> features -> minutes -> six means -> lineup adjustment.

    This is an offline study run. Inputs must all have been known before tip.
    ``artifacts`` is created by ``python -m study.check export``.
    """
    from . import data, model

    artifacts = Path(artifacts)
    dates = pd.to_datetime(games["date"]).dt.normalize().unique()
    if len(dates) != 1:
        raise ValueError("Provide exactly one NBA slate date")
    slate = data.build_upcoming_slate_features(
        history, games, players, advanced, model.load_json(artifacts / "dynamic_params.json")
    )
    if slate.empty:
        raise ValueError("No players matched the scheduled teams")
    curves = joblib.load(artifacts / "experience_curves.joblib")
    slate = model.apply_production_experience_curves(slate, curves)

    minutes = model.ModelBundle.load(artifacts / "minutes.joblib")
    model.assert_model_feature_contract(slate, minutes, "minutes model")
    slate["expected_minutes_model"] = minutes.predict(slate)
    slate["expected_minutes"] = slate["expected_minutes_model"]
    slate = data.apply_current_injury_adjustment(slate, injuries)
    slate["expected_minutes"] = slate["availability_expected_minutes"]
    slate.loc[slate["availability_out"].eq(1), "expected_minutes"] = 0.0
    slate, _ = model.apply_selected_mean_policy(slate, artifacts)

    role = model.load_gate3_runtime(artifacts / "role")
    slate = model.apply_gate3_role_state(
        slate,
        lineups,
        target_date=str(pd.Timestamp(dates[0]).date()),
        snapshot_dir=Path(snapshots),
        runtime=role,
    )
    slate["study_only"] = True
    return slate


def price_markets(projections, quotes, artifacts, simulations=20_000, seed=73):
    """Price ten supported markets. Return (priced rows, excluded quotes).

    Quotes must come from the same pregame information set as projections.
    This function does not certify collection time or produce production records.
    """
    from . import data, model

    if simulations < 200:
        raise ValueError("simulations must be at least 200")
    artifacts = Path(artifacts)
    marginals = joblib.load(artifacts / "marginals.joblib")
    copula = joblib.load(artifacts / "copula.joblib")
    dependence = model.load_json(artifacts / "combo_dependence_policy.json")
    calibration = model.load_json(artifacts / "market_probability_calibration_policy.json")
    role = model.load_gate3_runtime(artifacts / "role")
    supported = (set(PROP_TO_TARGET) | set(COMBO_COMPONENTS)) & set(calibration["props"])
    mu_columns = {target: f"mu_selected_{target}" for target in data.TARGETS}
    quoted = add_quote_filter_fields(quotes, 20.0, supported)
    rejected = quoted.loc[~quoted["quote_eligible"]].copy()
    eligible = quoted.loc[quoted["quote_eligible"]].copy()
    merged = eligible.merge(
        projections,
        on=["game_id", "player_id"],
        how="left",
        suffixes=("_market", ""),
        validate="many_to_one",
        indicator=True,
    )
    # Retain unmatched rows as explicit rejections so the learner can inspect them.
    merged.loc[merged["_merge"].ne("both"), "quote_filter_reason"] = "missing_projection"
    merged.loc[merged["availability_out"].eq(1), "quote_filter_reason"] = "player_currently_out"
    needs_role = merged["prop_type"].isin(model.GATE3_CHANGED_PROPS)
    unavailable = needs_role & ~merged["gate3_role_ready"].eq(1)
    merged.loc[unavailable & merged["quote_filter_reason"].eq(""), "quote_filter_reason"] = (
        "gate3_role_state_unavailable"
    )
    bad = merged["quote_filter_reason"].ne("")
    rejected = pd.concat([rejected, merged.loc[bad]], ignore_index=True)
    merged = merged.loc[~bad].drop(columns="_merge")
    if merged.empty:
        return pd.DataFrame(), rejected

    pieces = []
    for prop, target in PROP_TO_TARGET.items():
        part = merged.loc[merged["prop_type"].eq(prop)].copy()
        if part.empty:
            continue
        mean_column = "gate3_mu_ast" if prop == "assists" else mu_columns[target]
        probabilities = price_single_prop_frame(part, target, marginals[target], mean_column)
        for column in probabilities:
            part[column] = probabilities[column]
        part["dependence_lambda"] = 0.0
        pieces.append(part)

    for prop in sorted(COMBO_COMPONENTS):
        part = merged.loc[merged["prop_type"].eq(prop)].copy()
        if part.empty:
            continue
        strength = float(dependence["combos"][prop]["production_lambda"])
        for (game_id, player_id), group in part.groupby(["game_id", "player_id"], sort=False):
            probabilities = price_combo_lines(
                group.iloc[0],
                prop,
                np.sort(group["line_value"].unique()),
                marginals,
                mu_columns,
                copula,
                strength,
                simulations,
                stable_event_seed(game_id, player_id, prop, seed),
            )
            pieces.append(group.merge(probabilities, on="line_value", validate="many_to_one"))

    priced = pd.concat(pieces, ignore_index=True)
    priced = prepare_gate3_candidate_probability_overrides(
        priced,
        calibration_policy=calibration,
        probability_parameters=role["probability_parameters"],
    )
    mass = priced["p_over"] + priced["p_under"] + priced["p_push"]
    if not np.isfinite(mass).all() or np.max(np.abs(mass - 1)) > 0.005:
        raise ValueError("Over, under and push probabilities do not sum to one")
    priced = add_market_probability_layer(priced, calibration)
    priced["study_only"] = True
    return priced, rejected
