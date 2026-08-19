from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupKFold


TARGETS = ["pts", "reb", "ast", "stl", "blk", "fg3m"]

COMBO_COMPONENTS = {
    "points_rebounds": ("pts", "reb"),
    "points_assists": ("pts", "ast"),
    "rebounds_assists": ("reb", "ast"),
    "points_rebounds_assists": ("pts", "reb", "ast"),
    "stocks": ("stl", "blk"),
}

# Targeted refinement grids based on the coarse 2025 search.
LAMBDA_GRIDS = {
    "points_rebounds": [
        0.00, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70,
    ],
    "points_assists": [
        0.00, 0.10, 0.20, 0.30,
        0.40, 0.50, 0.60,
    ],
    "rebounds_assists": [
        0.00, 0.70, 0.75, 0.80, 0.85,
        0.90, 0.95, 1.00,
    ],
    "points_rebounds_assists": [
        0.00, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    ],
    "stocks": [
        0.00, 0.10, 0.20, 0.30,
    ],
}

INPUT_PATH = Path(
    "data/processed/selected_means_distribution_split.parquet"
)
COPULA_PATH = Path("models/copula_pre2025.joblib")
MARGINALS_PATH = Path("models/marginals_pre2025.joblib")
OUT_DIR = Path("data/processed/copula_audit")
POLICY_PATH = Path("models/combo_dependence_policy.json")

# Conservative deployment gate.
MIN_CV_IMPROVEMENT_PCT = 0.05
MIN_NONNEGATIVE_FOLDS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
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

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)

    corr = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
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
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)

    first = float(
        np.mean(np.abs(x - float(observed)))
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

    half_pairwise = float(
        np.sum(coefficients * x)
        / (n * n)
    )

    return first - half_pairwise


def observed_combo(
    row: pd.Series,
    components: tuple[str, ...],
) -> int:
    return int(
        sum(
            int(row[target])
            for target in components
        )
    )


def simulate_combo(
    row: pd.Series,
    components: tuple[str, ...],
    target_indices: dict[str, int],
    copula,
    marginals: dict,
    lambda_value: float,
    base_normals: np.ndarray,
) -> np.ndarray:
    player_id = int(row["player_id"])

    full_corr = copula.correlation_for_player(
        player_id
    )

    indices = [
        target_indices[target]
        for target in components
    ]

    sub_corr = full_corr[
        np.ix_(indices, indices)
    ]

    k = len(components)

    shrunk = (
        (1.0 - lambda_value) * np.eye(k)
        + lambda_value * sub_corr
    )

    shrunk = nearest_psd_correlation(
        shrunk
    )

    chol = np.linalg.cholesky(
        shrunk + 1e-12 * np.eye(k)
    )

    z = base_normals @ chol.T
    u = norm.cdf(z)

    total = np.zeros(
        len(u),
        dtype=int,
    )

    for j, target in enumerate(components):
        mu = float(
            row[f"mu_selected_{target}"]
        )

        total += marginals[
            target
        ].ppf(
            u[:, j],
            mu,
            row,
        )

    return total


def build_crps_grid(
    sampled: pd.DataFrame,
    combo: str,
    components: tuple[str, ...],
    lambdas: list[float],
    copula,
    marginals: dict,
    target_indices: dict[str, int],
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    records = []
    k = len(components)

    for row_index, row in sampled.iterrows():
        rng = np.random.default_rng(
            seed + 1009 * int(row_index)
        )

        base_normals = rng.standard_normal(
            size=(simulations, k)
        )

        observed = observed_combo(
            row,
            components,
        )

        for lam in lambdas:
            values = simulate_combo(
                row=row,
                components=components,
                target_indices=target_indices,
                copula=copula,
                marginals=marginals,
                lambda_value=float(lam),
                base_normals=base_normals,
            )

            records.append(
                {
                    "game_id": int(row["game_id"]),
                    "player_id": int(row["player_id"]),
                    "combo": combo,
                    "lambda": float(lam),
                    "crps": empirical_crps(
                        values,
                        observed,
                    ),
                }
            )

        if (
            row_index + 1
        ) % 250 == 0:
            print(
                f"  {combo}: evaluated "
                f"{row_index + 1:,}/"
                f"{len(sampled):,} rows"
            )

    return pd.DataFrame(records)


def grouped_cv_select(
    grid: pd.DataFrame,
    n_splits: int,
) -> tuple[pd.DataFrame, dict]:
    unique_rows = (
        grid[
            ["game_id", "player_id"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    groups = unique_rows[
        "game_id"
    ].to_numpy()

    splitter = GroupKFold(
        n_splits=n_splits
    )

    fold_rows = []
    lambdas = sorted(
        grid["lambda"].unique().tolist()
    )

    for fold_number, (
        train_idx,
        validation_idx,
    ) in enumerate(
        splitter.split(
            unique_rows,
            groups=groups,
        ),
        start=1,
    ):
        train_keys = unique_rows.iloc[
            train_idx
        ]

        validation_keys = unique_rows.iloc[
            validation_idx
        ]

        train = grid.merge(
            train_keys.assign(_train=1),
            on=["game_id", "player_id"],
            how="inner",
        )

        validation = grid.merge(
            validation_keys.assign(_validation=1),
            on=["game_id", "player_id"],
            how="inner",
        )

        train_means = (
            train.groupby("lambda")["crps"]
            .mean()
        )

        selected_lambda = float(
            train_means.idxmin()
        )

        validation_means = (
            validation.groupby("lambda")["crps"]
            .mean()
        )

        selected_crps = float(
            validation_means.loc[
                selected_lambda
            ]
        )

        independence_crps = (
            float(
                validation_means.loc[0.0]
            )
            if 0.0 in validation_means.index
            else np.nan
        )

        if np.isfinite(
            independence_crps
        ):
            improvement = (
                100.0
                * (
                    independence_crps
                    - selected_crps
                )
                / independence_crps
            )
        else:
            improvement = np.nan

        fold_rows.append(
            {
                "fold": fold_number,
                "selected_lambda": selected_lambda,
                "validation_crps": selected_crps,
                "independence_crps": independence_crps,
                "improvement_vs_independence_pct": improvement,
                "validation_rows": int(
                    len(validation_keys)
                ),
            }
        )

    folds = pd.DataFrame(
        fold_rows
    )

    overall_means = (
        grid.groupby("lambda")["crps"]
        .mean()
        .sort_index()
    )

    full_sample_best_lambda = float(
        overall_means.idxmin()
    )

    result = {
        "full_sample_best_lambda": (
            full_sample_best_lambda
        ),
        "fold_selected_lambdas": (
            folds[
                "selected_lambda"
            ]
            .tolist()
        ),
        "mean_cv_improvement_pct": float(
            folds[
                "improvement_vs_independence_pct"
            ].mean()
        )
        if folds[
            "improvement_vs_independence_pct"
        ].notna().any()
        else np.nan,
        "nonnegative_cv_folds": int(
            (
                folds[
                    "improvement_vs_independence_pct"
                ]
                >= 0.0
            ).sum()
        )
        if folds[
            "improvement_vs_independence_pct"
        ].notna().any()
        else 0,
    }

    return folds, result


def main() -> None:
    args = parse_args()

    for path in [
        INPUT_PATH,
        COPULA_PATH,
        MARGINALS_PATH,
    ]:
        if not path.exists():
            raise SystemExit(
                f"ERROR: missing {path}"
            )

    if args.folds < 2:
        raise SystemExit(
            "--folds must be at least 2"
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

    mean_cols = [
        f"mu_selected_{target}"
        for target in TARGETS
    ]

    holdout = holdout[
        holdout[mean_cols]
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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 116)
    print(
        "GAME-GROUPED CROSS-VALIDATED COPULA SHRINKAGE REFINEMENT"
    )
    print("=" * 116)
    print(
        f"2025 selection rows: {sample_n:,}"
    )
    print(
        f"Simulations per row/lambda: "
        f"{args.simulations:,}"
    )
    print(
        f"GroupKFold splits by game_id: "
        f"{args.folds}"
    )
    print(
        "2025 is development data for lambda calibration; "
        "2026-27 is the first external dependence test."
    )

    policy = {
        "selection_season": 2025,
        "external_test_season": 2026,
        "minimum_cv_improvement_pct": (
            MIN_CV_IMPROVEMENT_PCT
        ),
        "minimum_nonnegative_cv_folds": (
            MIN_NONNEGATIVE_FOLDS
        ),
        "combos": {},
    }

    summary_rows = []

    for combo_index, (
        combo,
        components,
    ) in enumerate(
        COMBO_COMPONENTS.items()
    ):
        print("\n" + "-" * 116)
        print(
            f"{combo}: grid="
            f"{LAMBDA_GRIDS[combo]}"
        )
        print("-" * 116)

        grid = build_crps_grid(
            sampled=sampled,
            combo=combo,
            components=components,
            lambdas=LAMBDA_GRIDS[combo],
            copula=copula,
            marginals=marginals,
            target_indices=target_indices,
            simulations=args.simulations,
            seed=(
                args.seed
                + 1_000_003
                * combo_index
            ),
        )

        grid_path = (
            OUT_DIR
            / f"{combo}_lambda_refinement_per_row.csv"
        )

        grid.to_csv(
            grid_path,
            index=False,
        )

        folds, result = grouped_cv_select(
            grid,
            n_splits=args.folds,
        )

        folds.to_csv(
            OUT_DIR
            / f"{combo}_lambda_group_cv.csv",
            index=False,
        )

        print()
        print(
            folds.to_string(
                index=False,
                formatters={
                    "selected_lambda": (
                        lambda x:
                        f"{x:.2f}"
                    ),
                    "validation_crps": (
                        lambda x:
                        f"{x:.5f}"
                    ),
                    "independence_crps": (
                        lambda x:
                        (
                            ""
                            if pd.isna(x)
                            else f"{x:.5f}"
                        )
                    ),
                    "improvement_vs_independence_pct": (
                        lambda x:
                        (
                            ""
                            if pd.isna(x)
                            else f"{x:+.4f}"
                        )
                    ),
                },
            )
        )

        full_best = float(
            result[
                "full_sample_best_lambda"
            ]
        )

        mean_cv_improvement = float(
            result[
                "mean_cv_improvement_pct"
            ]
        )

        nonnegative_folds = int(
            result[
                "nonnegative_cv_folds"
            ]
        )

        deploy_dependence = bool(
            mean_cv_improvement
            >= MIN_CV_IMPROVEMENT_PCT
            and nonnegative_folds
            >= MIN_NONNEGATIVE_FOLDS
        )

        production_lambda = (
            full_best
            if deploy_dependence
            else 0.0
        )

        policy[
            "combos"
        ][combo] = {
            "components": list(components),
            "refinement_grid": (
                LAMBDA_GRIDS[combo]
            ),
            "fold_selected_lambdas": (
                result[
                    "fold_selected_lambdas"
                ]
            ),
            "full_sample_best_lambda": (
                full_best
            ),
            "mean_cv_improvement_vs_independence_pct": (
                mean_cv_improvement
            )
            if np.isfinite(
                mean_cv_improvement
            )
            else None,
            "nonnegative_cv_folds": (
                nonnegative_folds
            ),
            "production_lambda": (
                production_lambda
            ),
        }

        summary_rows.append(
            {
                "combo": combo,
                "full_sample_best_lambda": (
                    full_best
                ),
                "fold_selected_lambdas": (
                    ",".join(
                        f"{x:.2f}"
                        for x in result[
                            "fold_selected_lambdas"
                        ]
                    )
                ),
                "mean_cv_improvement_pct": (
                    mean_cv_improvement
                ),
                "nonnegative_cv_folds": (
                    nonnegative_folds
                ),
                "production_lambda": (
                    production_lambda
                ),
            }
        )

        print()
        print(
            f"Full-sample refined minimum: "
            f"lambda={full_best:.2f}"
        )
        print(
            f"Fold-selected lambdas: "
            f"{result['fold_selected_lambdas']}"
        )

        print(
            f"Mean grouped-CV improvement vs "
            f"independence: "
            f"{mean_cv_improvement:+.4f}%"
        )
        print(
            f"Nonnegative CV folds: "
            f"{nonnegative_folds}/"
            f"{args.folds}"
        )

        print(
            f"Provisional production lambda: "
            f"{production_lambda:.2f}"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print("\n" + "=" * 116)
    print("REFINED DEPENDENCE POLICY")
    print("=" * 116)

    print(
        summary.to_string(
            index=False,
            formatters={
                "full_sample_best_lambda": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "mean_cv_improvement_pct": (
                    lambda x:
                    (
                        ""
                        if pd.isna(x)
                        else f"{x:+.4f}"
                    )
                ),
                "production_lambda": (
                    lambda x:
                    f"{x:.2f}"
                ),
            },
        )
    )

    with POLICY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            policy,
            handle,
            indent=2,
            sort_keys=True,
        )

    summary.to_csv(
        OUT_DIR
        / "combo_dependence_policy_summary.csv",
        index=False,
    )

    print()
    print(
        f"Saved provisional production policy: "
        f"{POLICY_PATH}"
    )
    print(
        "Review this output before wiring the lambdas into "
        "combo pricing."
    )


if __name__ == "__main__":
    main()
