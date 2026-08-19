from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm


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
COPULA_PATH = Path("models/copula_pre2025.joblib")
MARGINALS_PATH = Path("models/marginals_pre2025.joblib")
OUT_DIR = Path("data/processed/copula_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1000,
        help="2025 player-games used for the shrinkage grid.",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Monte Carlo samples per player-game / lambda.",
    )

    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.75, 1.0],
        help=(
            "Dependence-strength grid. "
            "0=independence, 1=full fitted copula."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=73,
    )

    return parser.parse_args()


def nearest_psd_correlation(
    corr: np.ndarray,
) -> np.ndarray:
    corr = np.asarray(corr, dtype=float)
    corr = 0.5 * (corr + corr.T)

    values, vectors = np.linalg.eigh(corr)
    values = np.clip(values, 1e-10, None)

    corr = (
        vectors
        @ np.diag(values)
        @ vectors.T
    )

    scale = np.sqrt(
        np.clip(
            np.diag(corr),
            1e-12,
            None,
        )
    )

    corr = corr / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)

    return corr


def empirical_crps(
    samples: np.ndarray,
    observed: float,
) -> float:
    x = np.sort(
        np.asarray(
            samples,
            dtype=float,
        )
    )

    n = len(x)

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

    return first_term - half_pairwise_term


def combo_observed(
    row: pd.Series,
    components: tuple[str, ...],
) -> int:
    return int(
        sum(
            int(row[target])
            for target in components
        )
    )


def simulate_combo_common_random_numbers(
    row: pd.Series,
    components: tuple[str, ...],
    target_indices: dict[str, int],
    copula,
    marginals: dict,
    lambda_value: float,
    standard_normals: np.ndarray,
) -> np.ndarray:
    player_id = int(row["player_id"])

    full_corr = (
        copula.correlation_for_player(
            player_id
        )
    )

    indices = [
        target_indices[target]
        for target in components
    ]

    sub_corr = full_corr[
        np.ix_(
            indices,
            indices,
        )
    ]

    k = len(components)

    shrunk = (
        (1.0 - lambda_value)
        * np.eye(k)
        + lambda_value
        * sub_corr
    )

    shrunk = nearest_psd_correlation(
        shrunk
    )

    chol = np.linalg.cholesky(
        shrunk
        + 1e-12 * np.eye(k)
    )

    z = (
        standard_normals
        @ chol.T
    )

    u = norm.cdf(z)

    total = np.zeros(
        len(u),
        dtype=int,
    )

    for j, target in enumerate(
        components
    ):
        mu = float(
            row[
                f"mu_selected_{target}"
            ]
        )

        total += marginals[
            target
        ].ppf(
            u[:, j],
            mu,
            row,
        )

    return total


def main() -> None:
    args = parse_args()

    for path in [
        INPUT_PATH,
        COPULA_PATH,
        MARGINALS_PATH,
    ]:
        if not path.exists():
            raise SystemExit(
                f"ERROR: missing required file: {path}"
            )

    if args.sample_rows <= 0:
        raise SystemExit(
            "--sample-rows must be positive"
        )

    if args.simulations < 200:
        raise SystemExit(
            "--simulations must be at least 200"
        )

    lambda_grid = sorted(
        set(
            float(x)
            for x in args.lambdas
        )
    )

    if (
        min(lambda_grid) < 0.0
        or max(lambda_grid) > 1.0
    ):
        raise SystemExit(
            "All lambdas must lie in [0, 1]."
        )

    frame = pd.read_parquet(
        INPUT_PATH
    )

    frame["season"] = pd.to_numeric(
        frame["season"],
        errors="raise",
    ).astype(int)

    holdout = frame[
        frame["season"].eq(2025)
    ].copy()

    required_means = [
        f"mu_selected_{target}"
        for target in TARGETS
    ]

    holdout = holdout[
        holdout[
            required_means
        ]
        .notna()
        .all(axis=1)
    ].copy()

    sample_n = min(
        args.sample_rows,
        len(holdout),
    )

    sampled = (
        holdout.sample(
            n=sample_n,
            random_state=args.seed,
        )
        .reset_index(drop=True)
    )

    copula = joblib.load(
        COPULA_PATH
    )

    marginals = joblib.load(
        MARGINALS_PATH
    )

    target_indices = {
        target: i
        for i, target in enumerate(
            TARGETS
        )
    }

    print("=" * 112)
    print("COPULA DEPENDENCE-STRENGTH SHRINKAGE GRID")
    print("=" * 112)
    print(
        f"2025 selection rows: {sample_n:,}"
    )
    print(
        f"Simulations per row/lambda: {args.simulations:,}"
    )
    print(
        "Lambda = 0 -> independence"
    )
    print(
        "Lambda = 1 -> full fitted copula"
    )
    print(
        "Common random numbers are reused across lambdas "
        "to reduce Monte Carlo comparison noise."
    )
    print()
    print(
        "IMPORTANT: 2025 is now development/selection data. "
        "This run is for production shrinkage calibration, "
        "not an untouched performance claim."
    )

    records = []

    for combo_index, (
        combo,
        components,
    ) in enumerate(
        COMBOS.items()
    ):
        print("\n" + "-" * 112)
        print(
            f"{combo}: components={components}"
        )
        print("-" * 112)

        accum = {
            lam: {
                "crps": [],
                "contains_80": [],
                "width_80": [],
            }
            for lam in lambda_grid
        }

        k = len(components)

        for row_index, row in sampled.iterrows():
            seed = (
                args.seed
                + 1_000_003 * combo_index
                + 1009 * int(row_index)
            )

            rng = np.random.default_rng(
                seed
            )

            standard_normals = (
                rng.standard_normal(
                    size=(
                        args.simulations,
                        k,
                    )
                )
            )

            observed = combo_observed(
                row,
                components,
            )

            for lam in lambda_grid:
                values = (
                    simulate_combo_common_random_numbers(
                        row=row,
                        components=components,
                        target_indices=target_indices,
                        copula=copula,
                        marginals=marginals,
                        lambda_value=lam,
                        standard_normals=standard_normals,
                    )
                )

                q10 = float(
                    np.quantile(
                        values,
                        0.10,
                    )
                )
                q90 = float(
                    np.quantile(
                        values,
                        0.90,
                    )
                )

                accum[
                    lam
                ][
                    "crps"
                ].append(
                    empirical_crps(
                        values,
                        observed,
                    )
                )

                accum[
                    lam
                ][
                    "contains_80"
                ].append(
                    int(
                        q10
                        <= observed
                        <= q90
                    )
                )

                accum[
                    lam
                ][
                    "width_80"
                ].append(
                    q90
                    - q10
                )

            if (
                row_index + 1
            ) % 250 == 0:
                print(
                    f"  evaluated "
                    f"{row_index + 1:,}/"
                    f"{sample_n:,} rows"
                )

        combo_rows = []

        independence_crps = float(
            np.mean(
                accum[
                    0.0
                ][
                    "crps"
                ]
            )
        )

        for lam in lambda_grid:
            mean_crps = float(
                np.mean(
                    accum[
                        lam
                    ][
                        "crps"
                    ]
                )
            )

            relative = (
                100.0
                * (
                    independence_crps
                    - mean_crps
                )
                / independence_crps
            )

            row = {
                "combo": combo,
                "lambda": lam,
                "rows": sample_n,
                "mean_crps": mean_crps,
                "relative_improvement_vs_independence_pct": (
                    relative
                ),
                "coverage_80": float(
                    np.mean(
                        accum[
                            lam
                        ][
                            "contains_80"
                        ]
                    )
                ),
                "width_80": float(
                    np.mean(
                        accum[
                            lam
                        ][
                            "width_80"
                        ]
                    )
                ),
            }

            records.append(row)
            combo_rows.append(row)

        combo_table = pd.DataFrame(
            combo_rows
        )

        best_row = combo_table.loc[
            combo_table[
                "mean_crps"
            ].idxmin()
        ]

        print()
        print(
            combo_table.to_string(
                index=False,
                formatters={
                    "lambda": (
                        lambda x:
                        f"{x:.2f}"
                    ),
                    "mean_crps": (
                        lambda x:
                        f"{x:.5f}"
                    ),
                    "relative_improvement_vs_independence_pct": (
                        lambda x:
                        f"{x:+.4f}"
                    ),
                    "coverage_80": (
                        lambda x:
                        f"{x:.3f}"
                    ),
                    "width_80": (
                        lambda x:
                        f"{x:.3f}"
                    ),
                },
            )
        )

        print(
            f"Grid minimum for {combo}: "
            f"lambda={best_row['lambda']:.2f}, "
            f"CRPS={best_row['mean_crps']:.5f}, "
            f"vs independence="
            f"{best_row['relative_improvement_vs_independence_pct']:+.4f}%"
        )

    results = pd.DataFrame(
        records
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_path = (
        OUT_DIR
        / "copula_shrinkage_grid_2025.csv"
    )

    results.to_csv(
        grid_path,
        index=False,
    )

    best = (
        results.loc[
            results.groupby(
                "combo"
            )[
                "mean_crps"
            ].idxmin()
        ]
        .sort_values(
            "combo"
        )
        .reset_index(
            drop=True
        )
    )

    print("\n" + "=" * 112)
    print("GRID MINIMA")
    print("=" * 112)

    print(
        best[
            [
                "combo",
                "lambda",
                "mean_crps",
                "relative_improvement_vs_independence_pct",
                "coverage_80",
                "width_80",
            ]
        ].to_string(
            index=False,
            formatters={
                "lambda": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "mean_crps": (
                    lambda x:
                    f"{x:.5f}"
                ),
                "relative_improvement_vs_independence_pct": (
                    lambda x:
                    f"{x:+.4f}"
                ),
                "coverage_80": (
                    lambda x:
                    f"{x:.3f}"
                ),
                "width_80": (
                    lambda x:
                    f"{x:.3f}"
                ),
            },
        )
    )

    summary_path = (
        OUT_DIR
        / "copula_shrinkage_grid_minima_2025.json"
    )

    payload = {
        row["combo"]: {
            "grid_best_lambda": float(
                row["lambda"]
            ),
            "mean_crps": float(
                row["mean_crps"]
            ),
            "relative_improvement_vs_independence_pct": float(
                row[
                    "relative_improvement_vs_independence_pct"
                ]
            ),
            "coverage_80": float(
                row["coverage_80"]
            ),
            "width_80": float(
                row["width_80"]
            ),
        }
        for _, row in best.iterrows()
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )

    print()
    print(
        f"Saved full grid: {grid_path}"
    )
    print(
        f"Saved grid minima: {summary_path}"
    )
    print()
    print(
        "Do not lock these lambdas yet. "
        "If an interior grid point wins, refine around it next."
    )


if __name__ == "__main__":
    main()
