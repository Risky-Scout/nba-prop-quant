from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.distributions import (
    FittedMarginal,
    NegativeBinomialCalibrator,
    PoissonCalibrator,
    ZeroInflatedNegativeBinomialCalibrator,
)
from nba_prop_quant.settings import get_settings
from nba_prop_quant.storage import write_parquet_atomic

console = Console()

TARGETS = ["pts", "reb", "ast", "stl", "blk", "fg3m"]

INPUT_PATH = Path("data/processed/oof_selected_means.parquet")
TRAIN_START_SEASON = 2018
VALIDATION_SEASONS = [2020, 2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025

NB_MIN_IMPROVEMENT_PCT = 0.05
ZINB_MIN_IMPROVEMENT_PCT = 0.10
MIN_POSITIVE_VALIDATION_FOLDS = 3


def inflation_features_for(
    target: str,
    frame: pd.DataFrame,
) -> list[str]:
    candidates = [
        "expected_minutes",
        f"mu_selected_{target}",
        f"decay_prior_{target}_rate",
        f"kalman_prior_{target}_rate",
        "opp_pace_prior",
        "days_rest",
        "is_home",
        "b2b",
        "pos_G",
        "pos_F",
        "pos_C",
    ]
    return [
        col
        for col in candidates
        if col in frame.columns
    ]


def fit_candidate(
    kind: str,
    y: np.ndarray,
    mu: np.ndarray,
    frame: pd.DataFrame,
    inflation_features: list[str],
) -> FittedMarginal:
    if kind == "poisson":
        model = PoissonCalibrator().fit(
            y,
            mu,
        )
        return FittedMarginal(
            kind="poisson",
            model=model,
        )

    if kind == "nb":
        model = NegativeBinomialCalibrator().fit(
            y,
            mu,
        )
        return FittedMarginal(
            kind="nb",
            model=model,
        )

    if kind == "zinb":
        model = ZeroInflatedNegativeBinomialCalibrator(
            inflation_features=inflation_features
        ).fit(
            y,
            mu,
            frame,
        )
        return FittedMarginal(
            kind="zinb",
            model=model,
        )

    raise KeyError(kind)


def score_candidate(
    fitted: FittedMarginal,
    y: np.ndarray,
    mu: np.ndarray,
    frame: pd.DataFrame,
) -> dict[str, float]:
    y = np.asarray(
        y,
        dtype=int,
    )
    mu = np.asarray(
        mu,
        dtype=float,
    )

    logpmf = fitted.model.logpmf(
        y,
        mu,
        frame,
    )

    if not np.isfinite(logpmf).all():
        raise RuntimeError(
            f"{fitted.kind}: non-finite log probabilities encountered"
        )

    zero_y = (
        y == 0
    ).astype(float)

    zero_prob = fitted.model.pmf(
        np.zeros_like(y),
        mu,
        frame,
    )

    return {
        "rows": float(len(y)),
        "nll_total": float(
            -np.sum(logpmf)
        ),
        "nll_per_row": float(
            -np.mean(logpmf)
        ),
        "zero_brier": float(
            np.mean(
                (
                    zero_y
                    - zero_prob
                )
                ** 2
            )
        ),
        "actual_zero_rate": float(
            np.mean(zero_y)
        ),
        "pred_zero_rate": float(
            np.mean(zero_prob)
        ),
    }


def aggregate_validation(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for candidate, group in metrics.groupby(
        "candidate",
        sort=False,
    ):
        total_rows = float(
            group["rows"].sum()
        )

        rows.append(
            {
                "candidate": candidate,
                "folds": int(
                    len(group)
                ),
                "rows": int(
                    total_rows
                ),
                "nll_total": float(
                    group[
                        "nll_total"
                    ].sum()
                ),
                "nll_per_row": float(
                    group[
                        "nll_total"
                    ].sum()
                    / total_rows
                ),
                "zero_brier": float(
                    np.average(
                        group[
                            "zero_brier"
                        ],
                        weights=group[
                            "rows"
                        ],
                    )
                ),
                "actual_zero_rate": float(
                    np.average(
                        group[
                            "actual_zero_rate"
                        ],
                        weights=group[
                            "rows"
                        ],
                    )
                ),
                "pred_zero_rate": float(
                    np.average(
                        group[
                            "pred_zero_rate"
                        ],
                        weights=group[
                            "rows"
                        ],
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "nll_per_row"
        )
        .reset_index(
            drop=True
        )
    )


def candidate_fold_wins(
    metrics: pd.DataFrame,
    challenger: str,
    incumbent: str,
) -> int:
    pivot = metrics.pivot(
        index="validation_season",
        columns="candidate",
        values="nll_per_row",
    )

    required = {
        challenger,
        incumbent,
    }

    if not required.issubset(
        pivot.columns
    ):
        return 0

    return int(
        (
            pivot[
                challenger
            ]
            < pivot[
                incumbent
            ]
        )
        .sum()
    )


def select_distribution(
    aggregate: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> dict:
    by_candidate = (
        aggregate
        .set_index(
            "candidate"
        )
    )

    selected = "poisson"
    decisions = []

    for challenger in [
        "nb",
        "zinb",
    ]:
        incumbent = selected

        challenger_nll = float(
            by_candidate.loc[
                challenger,
                "nll_per_row",
            ]
        )

        incumbent_nll = float(
            by_candidate.loc[
                incumbent,
                "nll_per_row",
            ]
        )

        improvement_pct = (
            1.0
            - challenger_nll
            / incumbent_nll
        ) * 100.0

        positive_folds = (
            candidate_fold_wins(
                fold_metrics,
                challenger=challenger,
                incumbent=incumbent,
            )
        )

        threshold = (
            NB_MIN_IMPROVEMENT_PCT
            if challenger
            == "nb"
            else ZINB_MIN_IMPROVEMENT_PCT
        )

        accepted = bool(
            improvement_pct
            >= threshold
            and positive_folds
            >= MIN_POSITIVE_VALIDATION_FOLDS
        )

        decisions.append(
            {
                "challenger": challenger,
                "incumbent": incumbent,
                "improvement_pct": float(
                    improvement_pct
                ),
                "positive_folds": int(
                    positive_folds
                ),
                "minimum_improvement_pct": float(
                    threshold
                ),
                "minimum_positive_folds": int(
                    MIN_POSITIVE_VALIDATION_FOLDS
                ),
                "accepted": accepted,
            }
        )

        if accepted:
            selected = challenger

    return {
        "selected": selected,
        "decisions": decisions,
    }


def candidate_frame(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    mean_col = (
        f"mu_selected_{target}"
    )

    return frame[
        frame[
            mean_col
        ].notna()
        & frame[
            target
        ].notna()
    ].copy()


def fit_all_candidates(
    train: pd.DataFrame,
    score: pd.DataFrame,
    target: str,
) -> list[dict]:
    mean_col = (
        f"mu_selected_{target}"
    )

    inflation_features = (
        inflation_features_for(
            target,
            train,
        )
    )

    y_train = train[
        target
    ].to_numpy(
        dtype=int
    )

    mu_train = train[
        mean_col
    ].to_numpy(
        dtype=float
    )

    y_score = score[
        target
    ].to_numpy(
        dtype=int
    )

    mu_score = score[
        mean_col
    ].to_numpy(
        dtype=float
    )

    rows = []

    for kind in [
        "poisson",
        "nb",
        "zinb",
    ]:
        started = (
            time.perf_counter()
        )

        try:
            fitted = fit_candidate(
                kind,
                y=y_train,
                mu=mu_train,
                frame=train,
                inflation_features=inflation_features,
            )

            metrics = score_candidate(
                fitted,
                y=y_score,
                mu=mu_score,
                frame=score,
            )

            metrics[
                "status"
            ] = "ok"

        except Exception as exc:
            metrics = {
                "rows": float(
                    len(score)
                ),
                "nll_total": np.inf,
                "nll_per_row": np.inf,
                "zero_brier": np.inf,
                "actual_zero_rate": float(
                    np.mean(
                        y_score
                        == 0
                    )
                ),
                "pred_zero_rate": np.nan,
                "status": (
                    f"failed: {exc}"
                ),
            }

        metrics[
            "candidate"
        ] = kind

        metrics[
            "elapsed_seconds"
        ] = (
            time.perf_counter()
            - started
        )

        rows.append(
            metrics
        )

    return rows


def main() -> None:
    settings = (
        get_settings()
    )

    if not INPUT_PATH.exists():
        raise SystemExit(
            f"ERROR: missing "
            f"{INPUT_PATH}. "
            "Run "
            "scripts/06c_select_mean_models.py "
            "first."
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    df["season"] = (
        pd.to_numeric(
            df[
                "season"
            ],
            errors="raise",
        )
        .astype(int)
    )

    strict = df[
        df[
            "season"
        ].between(
            TRAIN_START_SEASON,
            HOLDOUT_SEASON,
        )
    ].copy()

    audit_dir = (
        settings.processed_dir
        / "marginal_audit"
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_payload = {
        "train_start_season": TRAIN_START_SEASON,
        "validation_seasons": VALIDATION_SEASONS,
        "holdout_season": HOLDOUT_SEASON,
        "selection_gate": {
            "nb_min_improvement_pct": NB_MIN_IMPROVEMENT_PCT,
            "zinb_min_improvement_pct": ZINB_MIN_IMPROVEMENT_PCT,
            "min_positive_validation_folds": MIN_POSITIVE_VALIDATION_FOLDS,
        },
        "targets": {},
    }

    selected_pre2025 = {}
    selected_production = {}

    all_validation_rows = []
    all_holdout_rows = []

    console.rule(
        "Distribution selection"
    )

    console.print(
        "Strict selected means: "
        "2018-2025"
    )
    console.print(
        "Validation folds: "
        "2020-2024 expanding"
    )
    console.print(
        "Untouched selection holdout: "
        "2025"
    )
    console.print(
        "Candidates: "
        "Poisson, Negative Binomial, "
        "Zero-Inflated Negative Binomial"
    )

    for target in TARGETS:
        console.rule(
            target
        )

        target_frame = (
            candidate_frame(
                strict,
                target,
            )
        )

        fold_rows = []

        for validation_season in (
            VALIDATION_SEASONS
        ):
            train = target_frame[
                (
                    target_frame[
                        "season"
                    ]
                    >= TRAIN_START_SEASON
                )
                & (
                    target_frame[
                        "season"
                    ]
                    < validation_season
                )
            ].copy()

            validation = (
                target_frame[
                    target_frame[
                        "season"
                    ].eq(
                        validation_season
                    )
                ]
                .copy()
            )

            if (
                train.empty
                or validation.empty
            ):
                raise RuntimeError(
                    f"{target}: empty "
                    f"train/validation fold "
                    f"for {validation_season}"
                )

            console.print(
                f"{validation_season}: "
                f"train={len(train):,}, "
                f"validate={len(validation):,}"
            )

            candidate_rows = (
                fit_all_candidates(
                    train,
                    validation,
                    target,
                )
            )

            for row in (
                candidate_rows
            ):
                row[
                    "target"
                ] = target

                row[
                    "validation_season"
                ] = int(
                    validation_season
                )

                row[
                    "train_rows"
                ] = int(
                    len(train)
                )

                fold_rows.append(
                    row
                )

                console.print(
                    f"  {row['candidate']:7s} "
                    f"NLL/row="
                    f"{row['nll_per_row']:.6f} "
                    f"zeroBrier="
                    f"{row['zero_brier']:.6f} "
                    f"time="
                    f"{row['elapsed_seconds']:.1f}s"
                )

        fold_metrics = (
            pd.DataFrame(
                fold_rows
            )
        )

        aggregate = (
            aggregate_validation(
                fold_metrics
            )
        )

        selection = (
            select_distribution(
                aggregate,
                fold_metrics,
            )
        )

        selected_kind = (
            selection[
                "selected"
            ]
        )

        console.print()
        console.print(
            aggregate[
                [
                    "candidate",
                    "rows",
                    "nll_per_row",
                    "zero_brier",
                    "actual_zero_rate",
                    "pred_zero_rate",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "nll_per_row": (
                        lambda x:
                        f"{x:.6f}"
                    ),
                    "zero_brier": (
                        lambda x:
                        f"{x:.6f}"
                    ),
                    "actual_zero_rate": (
                        lambda x:
                        f"{x:.4f}"
                    ),
                    "pred_zero_rate": (
                        lambda x:
                        f"{x:.4f}"
                    ),
                },
            )
        )

        console.print(
            f"[green]Selected "
            f"{target}: "
            f"{selected_kind}[/green]"
        )

        for decision in (
            selection[
                "decisions"
            ]
        ):
            console.print(
                "  "
                f"{decision['challenger']} "
                f"vs "
                f"{decision['incumbent']}: "
                f"improvement="
                f"{decision['improvement_pct']:+.4f}% | "
                f"wins="
                f"{decision['positive_folds']}/"
                f"{len(VALIDATION_SEASONS)} | "
                f"accepted="
                f"{decision['accepted']}"
            )

        fold_metrics.to_csv(
            audit_dir
            / f"{target}_validation_folds.csv",
            index=False,
        )

        aggregate.to_csv(
            audit_dir
            / f"{target}_validation_aggregate.csv",
            index=False,
        )

        all_validation_rows.extend(
            fold_rows
        )

        # ----------------------------------------------------------
        # Untouched 2025 holdout:
        # fit every candidate on 2018-2024,
        # score on 2025, but DO NOT use holdout
        # results to change the selected type.
        # ----------------------------------------------------------
        train_pre2025 = (
            target_frame[
                target_frame[
                    "season"
                ].between(
                    TRAIN_START_SEASON,
                    HOLDOUT_SEASON - 1,
                )
            ]
            .copy()
        )

        holdout = (
            target_frame[
                target_frame[
                    "season"
                ].eq(
                    HOLDOUT_SEASON
                )
            ]
            .copy()
        )

        holdout_rows = (
            fit_all_candidates(
                train_pre2025,
                holdout,
                target,
            )
        )

        console.print(
            f"2025 untouched holdout "
            f"(selection remains "
            f"{selected_kind}):"
        )

        for row in (
            holdout_rows
        ):
            row[
                "target"
            ] = target
            row[
                "holdout_season"
            ] = HOLDOUT_SEASON
            row[
                "selected_by_validation"
            ] = bool(
                row[
                    "candidate"
                ]
                == selected_kind
            )

            all_holdout_rows.append(
                row
            )

            console.print(
                f"  {row['candidate']:7s} "
                f"NLL/row="
                f"{row['nll_per_row']:.6f} "
                f"zeroBrier="
                f"{row['zero_brier']:.6f}"
            )

        holdout_table = (
            pd.DataFrame(
                holdout_rows
            )
        )

        holdout_table.to_csv(
            audit_dir
            / f"{target}_holdout_2025.csv",
            index=False,
        )

        # Fit the selected distribution on 2018-2024.
        inflation_features = (
            inflation_features_for(
                target,
                train_pre2025,
            )
        )

        y_pre = (
            train_pre2025[
                target
            ]
            .to_numpy(
                dtype=int
            )
        )

        mu_pre = (
            train_pre2025[
                f"mu_selected_{target}"
            ]
            .to_numpy(
                dtype=float
            )
        )

        selected_pre2025[
            target
        ] = fit_candidate(
            selected_kind,
            y=y_pre,
            mu=mu_pre,
            frame=train_pre2025,
            inflation_features=inflation_features,
        )

        # Fit production distribution on all 2018-2025.
        production_frame = (
            target_frame.copy()
        )

        production_features = (
            inflation_features_for(
                target,
                production_frame,
            )
        )

        selected_production[
            target
        ] = fit_candidate(
            selected_kind,
            y=production_frame[
                target
            ].to_numpy(
                dtype=int
            ),
            mu=production_frame[
                f"mu_selected_{target}"
            ].to_numpy(
                dtype=float
            ),
            frame=production_frame,
            inflation_features=production_features,
        )

        selection_payload[
            "targets"
        ][target] = {
            "selected_distribution": selected_kind,
            "validation_aggregate": (
                aggregate
                .set_index(
                    "candidate"
                )
                .to_dict(
                    orient="index"
                )
            ),
            "decisions": selection[
                "decisions"
            ],
            "holdout_2025": (
                pd.DataFrame(
                    holdout_rows
                )
                .set_index(
                    "candidate"
                )
                .to_dict(
                    orient="index"
                )
            ),
            "inflation_features": (
                production_features
                if selected_kind
                == "zinb"
                else []
            ),
        }

    # --------------------------------------------------------------
    # Save artifacts.
    # --------------------------------------------------------------
    joblib.dump(
        selected_pre2025,
        settings.nba_prop_model_dir
        / "marginals_pre2025.joblib",
    )

    joblib.dump(
        selected_production,
        settings.nba_prop_model_dir
        / "marginals.joblib",
    )

    with (
        settings.nba_prop_model_dir
        / "marginal_selection.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            selection_payload,
            handle,
            indent=2,
            sort_keys=True,
        )

    validation_table = (
        pd.DataFrame(
            all_validation_rows
        )
    )

    holdout_table = (
        pd.DataFrame(
            all_holdout_rows
        )
    )

    validation_table.to_csv(
        audit_dir
        / "all_validation_candidate_metrics.csv",
        index=False,
    )

    holdout_table.to_csv(
        audit_dir
        / "all_holdout_2025_candidate_metrics.csv",
        index=False,
    )

    split_frame = strict.copy()
    split_frame[
        "distribution_split"
    ] = np.where(
        split_frame[
            "season"
        ].eq(
            HOLDOUT_SEASON
        ),
        "holdout_2025",
        "development_2018_2024",
    )

    write_parquet_atomic(
        split_frame,
        settings.processed_dir
        / "selected_means_distribution_split.parquet",
    )

    console.rule(
        "FINAL DISTRIBUTION SELECTION"
    )

    for target in TARGETS:
        selected = (
            selection_payload[
                "targets"
            ][target][
                "selected_distribution"
            ]
        )

        console.print(
            f"{target.upper():5s}: "
            f"{selected}"
        )

    console.print()
    console.print(
        "[green]Saved[/green] "
        "models/marginals_pre2025.joblib"
    )
    console.print(
        "[green]Saved[/green] "
        "models/marginals.joblib"
    )
    console.print(
        "[green]Saved[/green] "
        "models/marginal_selection.json"
    )
    console.print(
        "[green]Saved[/green] "
        "data/processed/"
        "selected_means_distribution_split.parquet"
    )
    console.print()
    console.print(
        "Do not run the copula step yet. "
        "Review validation and untouched 2025 "
        "distribution results first."
    )


if __name__ == "__main__":
    main()
