from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from rich.console import Console

from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.settings import get_settings


TARGETS = ["pts", "reb", "ast", "stl", "blk", "fg3m"]

COMBOS = {
    "points_rebounds": ("pts", "reb"),
    "points_assists": ("pts", "ast"),
    "rebounds_assists": ("reb", "ast"),
    "points_rebounds_assists": ("pts", "reb", "ast"),
    "stocks": ("stl", "blk"),
}

INPUT_PATH = Path(
    "data/processed/selected_means_distribution_split.parquet"
)

PRE2025_MARGINALS_PATH = Path(
    "models/marginals_pre2025.joblib"
)

PRODUCTION_MARGINALS_PATH = Path(
    "models/marginals.joblib"
)

PRE2025_COPULA_PATH = Path(
    "models/copula_pre2025.joblib"
)

PRODUCTION_COPULA_PATH = Path(
    "models/copula.joblib"
)

AUDIT_DIR = Path(
    "data/processed/copula_audit"
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2000,
        help=(
            "Number of 2025 holdout player-games used for "
            "Monte Carlo combo-distribution CRPS evaluation."
        ),
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=1500,
        help=(
            "Monte Carlo draws per sampled player-game for "
            "copula-vs-independence combo evaluation."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=73,
    )

    return parser.parse_args()


def mid_pit_z(
    frame: pd.DataFrame,
    marginals: dict,
    mu_columns: dict[str, str],
) -> np.ndarray:
    z_columns = []

    for target in TARGETS:
        y = frame[target].to_numpy(
            dtype=int
        )

        mu = frame[
            mu_columns[target]
        ].to_numpy(
            dtype=float
        )

        marginal = marginals[target]

        lower = marginal.cdf(
            y - 1,
            mu,
            frame,
        )

        mass = marginal.pmf(
            y,
            mu,
            frame,
        )

        u = np.clip(
            lower + 0.5 * mass,
            1e-6,
            1.0 - 1e-6,
        )

        z_columns.append(
            norm.ppf(u)
        )

    return np.column_stack(
        z_columns
    )


def copula_log_density(
    z: np.ndarray,
    corr: np.ndarray,
) -> np.ndarray:
    """
    Gaussian-copula density contribution relative to independence:

        log c(u) =
          -0.5 log|R|
          -0.5 z' (R^{-1} - I) z

    This is a dependence pseudo-log score for our discrete mid-PIT
    transform, not the exact joint discrete count likelihood.
    """
    corr = np.asarray(
        corr,
        dtype=float,
    )

    sign, logdet = np.linalg.slogdet(
        corr
    )

    if sign <= 0:
        raise ValueError(
            "Correlation matrix is not positive definite."
        )

    inv = np.linalg.inv(
        corr
    )

    adjustment = (
        inv
        - np.eye(
            corr.shape[0]
        )
    )

    quad = np.einsum(
        "ij,jk,ik->i",
        z,
        adjustment,
        z,
    )

    return (
        -0.5 * logdet
        - 0.5 * quad
    )


def player_specific_log_score(
    frame: pd.DataFrame,
    z: np.ndarray,
    copula: GaussianCopula,
) -> np.ndarray:
    player_ids = frame[
        "player_id"
    ].to_numpy()

    result = np.empty(
        len(frame),
        dtype=float,
    )

    unique_players = np.unique(
        player_ids
    )

    for player_id in unique_players:
        mask = (
            player_ids
            == player_id
        )

        corr = (
            copula.correlation_for_player(
                int(player_id)
            )
        )

        result[mask] = (
            copula_log_density(
                z[mask],
                corr,
            )
        )

    return result


def empirical_crps(
    samples: np.ndarray,
    observed: float,
) -> float:
    """
    CRPS for an empirical predictive sample.

    Uses the O(n log n) sorted-sample identity instead of
    explicitly constructing all pairwise distances.
    """
    x = np.sort(
        np.asarray(
            samples,
            dtype=float,
        )
    )

    n = len(x)

    if n == 0:
        return float("nan")

    first_term = float(
        np.mean(
            np.abs(
                x - float(observed)
            )
        )
    )

    index = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    coefficients = (
        2.0 * index
        - n
        - 1.0
    )

    half_pairwise_term = float(
        np.sum(
            coefficients * x
        )
        / (n * n)
    )

    return (
        first_term
        - half_pairwise_term
    )


def realized_combo(
    row: pd.Series,
    components: tuple[str, ...],
) -> int:
    return int(
        sum(
            int(
                row[target]
            )
            for target in components
        )
    )


def summarize_combo_metrics(
    per_row: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for combo, group in per_row.groupby(
        "combo",
        sort=False,
    ):
        copula_crps = float(
            group[
                "copula_crps"
            ].mean()
        )

        independence_crps = float(
            group[
                "independence_crps"
            ].mean()
        )

        improvement = (
            1.0
            - copula_crps
            / independence_crps
        ) * 100.0

        rows.append(
            {
                "combo": combo,
                "rows": len(group),
                "copula_crps": copula_crps,
                "independence_crps": independence_crps,
                "crps_improvement_pct": improvement,
                "copula_80_coverage": float(
                    group[
                        "copula_80_contains"
                    ].mean()
                ),
                "independence_80_coverage": float(
                    group[
                        "independence_80_contains"
                    ].mean()
                ),
                "copula_80_width": float(
                    group[
                        "copula_80_width"
                    ].mean()
                ),
                "independence_80_width": float(
                    group[
                        "independence_80_width"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def correlation_error(
    fitted: np.ndarray,
    observed: np.ndarray,
) -> float:
    mask = ~np.eye(
        fitted.shape[0],
        dtype=bool,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    fitted[mask]
                    - observed[mask]
                )
                ** 2
            )
        )
    )


def main() -> None:
    args = parse_args()

    settings = get_settings()

    for path in [
        INPUT_PATH,
        PRE2025_MARGINALS_PATH,
        PRODUCTION_MARGINALS_PATH,
    ]:
        if not path.exists():
            raise SystemExit(
                f"ERROR: missing required file: {path}"
            )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_parquet(
        INPUT_PATH
    )

    frame["season"] = pd.to_numeric(
        frame["season"],
        errors="raise",
    ).astype(int)

    mu_columns = {
        target: f"mu_selected_{target}"
        for target in TARGETS
    }

    required_columns = {
        "season",
        "player_id",
        "game_id",
        *TARGETS,
        *mu_columns.values(),
    }

    missing = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise SystemExit(
            f"ERROR: input missing columns: {sorted(missing)}"
        )

    development = frame[
        frame["season"].between(
            2018,
            2024,
        )
    ].copy()

    holdout = frame[
        frame["season"].eq(
            2025
        )
    ].copy()

    production = frame[
        frame["season"].between(
            2018,
            2025,
        )
    ].copy()

    development = development[
        development[
            list(
                mu_columns.values()
            )
        ]
        .notna()
        .all(axis=1)
    ].copy()

    holdout = holdout[
        holdout[
            list(
                mu_columns.values()
            )
        ]
        .notna()
        .all(axis=1)
    ].copy()

    production = production[
        production[
            list(
                mu_columns.values()
            )
        ]
        .notna()
        .all(axis=1)
    ].copy()

    pre2025_marginals = joblib.load(
        PRE2025_MARGINALS_PATH
    )

    production_marginals = joblib.load(
        PRODUCTION_MARGINALS_PATH
    )

    console.rule(
        "Gaussian copula dependence validation"
    )

    console.print(
        f"Copula development rows (2018-2024): "
        f"{len(development):,}"
    )

    console.print(
        f"Untouched dependence holdout rows (2025): "
        f"{len(holdout):,}"
    )

    console.print(
        "Dependence benchmark: fitted Gaussian copula "
        "vs independence."
    )

    console.print(
        "NOTE: after this run, 2025 becomes the "
        "copula-selection holdout and should not be treated "
        "as untouched for later dependence-model selection."
    )

    # --------------------------------------------------------------
    # 1. Fit pre-2025 copula.
    # --------------------------------------------------------------
    pre2025_copula = GaussianCopula(
        targets=list(
            TARGETS
        )
    ).fit(
        development,
        marginals=pre2025_marginals,
        mu_columns=mu_columns,
    )

    joblib.dump(
        pre2025_copula,
        PRE2025_COPULA_PATH,
    )

    # --------------------------------------------------------------
    # 2. Full 2025 dependence pseudo-log score.
    # --------------------------------------------------------------
    z_holdout = mid_pit_z(
        holdout,
        marginals=pre2025_marginals,
        mu_columns=mu_columns,
    )

    copula_log_score = (
        player_specific_log_score(
            holdout,
            z_holdout,
            pre2025_copula,
        )
    )

    dependence_summary = {
        "holdout_rows": int(
            len(holdout)
        ),
        "mean_copula_log_density_relative_to_independence": float(
            np.mean(
                copula_log_score
            )
        ),
        "median_copula_log_density_relative_to_independence": float(
            np.median(
                copula_log_score
            )
        ),
        "positive_log_density_fraction": float(
            np.mean(
                copula_log_score
                > 0.0
            )
        ),
    }

    console.rule(
        "2025 dependence pseudo-log score"
    )

    console.print(
        "Independence reference log-copula density: 0.000000"
    )

    console.print(
        f"Mean fitted-copula log density: "
        f"{dependence_summary['mean_copula_log_density_relative_to_independence']:+.6f}"
    )

    console.print(
        f"Median fitted-copula log density: "
        f"{dependence_summary['median_copula_log_density_relative_to_independence']:+.6f}"
    )

    console.print(
        f"Rows where fitted dependence scores above independence: "
        f"{dependence_summary['positive_log_density_fraction']:.2%}"
    )

    # --------------------------------------------------------------
    # 3. Correlation-matrix stability.
    # --------------------------------------------------------------
    observed_corr_2025 = np.corrcoef(
        z_holdout,
        rowvar=False,
    )

    fitted_global_corr = (
        pre2025_copula.global_corr
    )

    corr_rmse = correlation_error(
        fitted_global_corr,
        observed_corr_2025,
    )

    console.rule(
        "Global latent-correlation stability"
    )

    console.print(
        f"Off-diagonal RMSE, fitted 2018-2024 vs observed 2025: "
        f"{corr_rmse:.4f}"
    )

    corr_names = list(
        TARGETS
    )

    fitted_corr_df = pd.DataFrame(
        fitted_global_corr,
        index=corr_names,
        columns=corr_names,
    )

    observed_corr_df = pd.DataFrame(
        observed_corr_2025,
        index=corr_names,
        columns=corr_names,
    )

    console.print(
        "\nFitted 2018-2024 global latent correlation:"
    )

    console.print(
        fitted_corr_df.round(
            3
        ).to_string()
    )

    console.print(
        "\nObserved 2025 latent correlation:"
    )

    console.print(
        observed_corr_df.round(
            3
        ).to_string()
    )

    fitted_corr_df.to_csv(
        AUDIT_DIR
        / "fitted_global_corr_2018_2024.csv"
    )

    observed_corr_df.to_csv(
        AUDIT_DIR
        / "observed_latent_corr_2025.csv"
    )

    # --------------------------------------------------------------
    # 4. Direct combo-distribution validation via Monte Carlo CRPS.
    # --------------------------------------------------------------
    if args.sample_rows <= 0:
        raise SystemExit(
            "--sample-rows must be positive"
        )

    if args.simulations < 200:
        raise SystemExit(
            "--simulations must be at least 200"
        )

    sample_n = min(
        args.sample_rows,
        len(holdout),
    )

    sampled = holdout.sample(
        n=sample_n,
        random_state=args.seed,
    ).reset_index(
        drop=True
    )

    independence = GaussianCopula(
        targets=list(
            TARGETS
        ),
        global_corr=np.eye(
            len(
                TARGETS
            )
        ),
        player_corr={},
    )

    console.rule(
        "2025 combo-distribution CRPS"
    )

    console.print(
        f"Monte Carlo sample rows: {sample_n:,}"
    )

    console.print(
        f"Simulations per row/model: {args.simulations:,}"
    )

    per_row_metrics = []

    for row_index, row in sampled.iterrows():
        seed = (
            args.seed
            + int(
                row_index
            )
            * 1009
        )

        copula_samples = (
            pre2025_copula.simulate(
                row=row,
                marginals=pre2025_marginals,
                mu_columns=mu_columns,
                simulations=args.simulations,
                seed=seed,
            )
        )

        independent_samples = (
            independence.simulate(
                row=row,
                marginals=pre2025_marginals,
                mu_columns=mu_columns,
                simulations=args.simulations,
                seed=seed,
            )
        )

        for combo, components in COMBOS.items():
            observed = realized_combo(
                row,
                components,
            )

            copula_values = (
                copula_samples[
                    combo
                ]
                .to_numpy(
                    dtype=float
                )
            )

            independence_values = (
                independent_samples[
                    combo
                ]
                .to_numpy(
                    dtype=float
                )
            )

            copula_q10 = float(
                np.quantile(
                    copula_values,
                    0.10,
                )
            )

            copula_q90 = float(
                np.quantile(
                    copula_values,
                    0.90,
                )
            )

            independence_q10 = float(
                np.quantile(
                    independence_values,
                    0.10,
                )
            )

            independence_q90 = float(
                np.quantile(
                    independence_values,
                    0.90,
                )
            )

            per_row_metrics.append(
                {
                    "game_id": int(
                        row[
                            "game_id"
                        ]
                    ),
                    "player_id": int(
                        row[
                            "player_id"
                        ]
                    ),
                    "combo": combo,
                    "observed": observed,
                    "copula_crps": empirical_crps(
                        copula_values,
                        observed,
                    ),
                    "independence_crps": empirical_crps(
                        independence_values,
                        observed,
                    ),
                    "copula_80_contains": int(
                        copula_q10
                        <= observed
                        <= copula_q90
                    ),
                    "independence_80_contains": int(
                        independence_q10
                        <= observed
                        <= independence_q90
                    ),
                    "copula_80_width": (
                        copula_q90
                        - copula_q10
                    ),
                    "independence_80_width": (
                        independence_q90
                        - independence_q10
                    ),
                }
            )

        if (
            row_index + 1
        ) % 250 == 0:
            console.print(
                f"  evaluated "
                f"{row_index + 1:,}/"
                f"{sample_n:,} rows"
            )

    per_row_df = pd.DataFrame(
        per_row_metrics
    )

    combo_summary = (
        summarize_combo_metrics(
            per_row_df
        )
    )

    console.print()
    console.print(
        combo_summary.to_string(
            index=False,
            formatters={
                "copula_crps": (
                    lambda x:
                    f"{x:.5f}"
                ),
                "independence_crps": (
                    lambda x:
                    f"{x:.5f}"
                ),
                "crps_improvement_pct": (
                    lambda x:
                    f"{x:+.3f}"
                ),
                "copula_80_coverage": (
                    lambda x:
                    f"{x:.3f}"
                ),
                "independence_80_coverage": (
                    lambda x:
                    f"{x:.3f}"
                ),
                "copula_80_width": (
                    lambda x:
                    f"{x:.3f}"
                ),
                "independence_80_width": (
                    lambda x:
                    f"{x:.3f}"
                ),
            },
        )
    )

    per_row_df.to_csv(
        AUDIT_DIR
        / "combo_crps_2025_per_row.csv",
        index=False,
    )

    combo_summary.to_csv(
        AUDIT_DIR
        / "combo_crps_2025_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # 5. Fit production copula on all 2018-2025.
    #    We save it now, but downstream use is conditional on reviewing
    #    the 2025 dependence/CRPS evidence printed above.
    # --------------------------------------------------------------
    production_copula = GaussianCopula(
        targets=list(
            TARGETS
        )
    ).fit(
        production,
        marginals=production_marginals,
        mu_columns=mu_columns,
    )

    joblib.dump(
        production_copula,
        PRODUCTION_COPULA_PATH,
    )

    # --------------------------------------------------------------
    # 6. Save machine-readable audit summary.
    # --------------------------------------------------------------
    summary_payload = {
        "development_seasons": [
            2018,
            2024,
        ],
        "dependence_holdout_season": 2025,
        "holdout_rows": int(
            len(holdout)
        ),
        "monte_carlo_sample_rows": int(
            sample_n
        ),
        "simulations_per_row": int(
            args.simulations
        ),
        "dependence_pseudo_log_score": dependence_summary,
        "global_latent_correlation_rmse_2025": float(
            corr_rmse
        ),
        "combo_crps": (
            combo_summary
            .set_index(
                "combo"
            )
            .to_dict(
                orient="index"
            )
        ),
    }

    with (
        AUDIT_DIR
        / "copula_validation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary_payload,
            handle,
            indent=2,
            sort_keys=True,
        )

    console.rule(
        "COPULA VALIDATION COMPLETE"
    )

    console.print(
        f"Saved development copula: "
        f"{PRE2025_COPULA_PATH}"
    )

    console.print(
        f"Saved production copula: "
        f"{PRODUCTION_COPULA_PATH}"
    )

    console.print(
        f"Saved audit outputs: "
        f"{AUDIT_DIR}"
    )

    console.print()
    console.print(
        "Do not proceed to combo pricing yet. "
        "Review whether the copula beats independence "
        "on the 2025 CRPS results."
    )


if __name__ == "__main__":
    main()
