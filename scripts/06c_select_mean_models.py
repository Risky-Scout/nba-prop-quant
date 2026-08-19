from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ENSEMBLE_OOF_PATH = Path(
    "data/processed/oof_ensemble_predictions.parquet"
)
ENSEMBLE_AUDIT_DIR = Path(
    "data/processed/ensemble_audit"
)
OUT_PATH = Path(
    "data/processed/oof_selected_means.parquet"
)
SELECTION_PATH = Path(
    "models/mean_model_selection.json"
)

TARGETS = [
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "fg3m",
]

# Conservative model-selection gate.
#
# We only accept the extra ensemble complexity when:
#   1) strict pooled RMSE improves by at least 0.05%;
#   2) at least 6 of 8 strict walk-forward seasons improve vs XGB;
#   3) mean improvement over the recent 2023-2025 period is positive.
MIN_POOLED_IMPROVEMENT_PCT = 0.05
MIN_POSITIVE_FOLDS = 6
RECENT_SEASONS = {2023, 2024, 2025}


def main() -> None:
    if not ENSEMBLE_OOF_PATH.exists():
        raise SystemExit(
            f"ERROR: missing {ENSEMBLE_OOF_PATH}. "
            "Run scripts/06b_fit_mean_ensemble.py first."
        )

    summary_path = (
        ENSEMBLE_AUDIT_DIR
        / "ensemble_summary.csv"
    )

    if not summary_path.exists():
        raise SystemExit(
            f"ERROR: missing {summary_path}."
        )

    df = pd.read_parquet(
        ENSEMBLE_OOF_PATH
    )

    summary = pd.read_csv(
        summary_path
    )

    selection: dict[str, dict] = {}

    print("=" * 104)
    print("CANONICAL MEAN-MODEL SELECTION")
    print("=" * 104)
    print(
        f"Gate: pooled improvement >= "
        f"{MIN_POOLED_IMPROVEMENT_PCT:.3f}%"
    )
    print(
        f"      positive strict folds >= "
        f"{MIN_POSITIVE_FOLDS}/8"
    )
    print(
        "      mean 2023-2025 improvement > 0"
    )

    for target in TARGETS:
        row = summary.loc[
            summary["target"].eq(target)
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Expected one summary row for {target}, "
                f"found {len(row)}"
            )

        row = row.iloc[0]

        fold_path = (
            ENSEMBLE_AUDIT_DIR
            / f"{target}_walk_forward_weights.csv"
        )

        if not fold_path.exists():
            raise RuntimeError(
                f"Missing fold audit: {fold_path}"
            )

        folds = pd.read_csv(
            fold_path
        )

        pooled_improvement = float(
            row["improvement_vs_xgb_pct"]
        )

        positive_folds = int(
            (
                folds[
                    "improvement_vs_xgb_pct"
                ]
                > 0.0
            ).sum()
        )

        recent = folds[
            folds["season"].isin(
                RECENT_SEASONS
            )
        ]

        if len(recent) != len(
            RECENT_SEASONS
        ):
            raise RuntimeError(
                f"{target}: expected all recent seasons "
                f"{sorted(RECENT_SEASONS)}"
            )

        recent_mean_improvement = float(
            recent[
                "improvement_vs_xgb_pct"
            ].mean()
        )

        use_ensemble = bool(
            pooled_improvement
            >= MIN_POOLED_IMPROVEMENT_PCT
            and positive_folds
            >= MIN_POSITIVE_FOLDS
            and recent_mean_improvement
            > 0.0
        )

        selected_mode = (
            "ensemble"
            if use_ensemble
            else "xgb"
        )

        selected_col = (
            f"mu_ensemble_{target}"
            if use_ensemble
            else f"mu_xgb_{target}"
        )

        if selected_col not in df.columns:
            raise RuntimeError(
                f"Missing selected mean column: "
                f"{selected_col}"
            )

        canonical_col = (
            f"mu_selected_{target}"
        )

        # Strict common downstream evaluation begins in 2018,
        # because that is the first season with leakage-safe
        # ensemble weights trained only on earlier OOF seasons.
        strict_mask = (
            pd.to_numeric(
                df["season"],
                errors="raise",
            )
            .astype(int)
            .ge(2018)
        )

        df[canonical_col] = np.nan

        df.loc[
            strict_mask,
            canonical_col,
        ] = df.loc[
            strict_mask,
            selected_col,
        ]

        prod_weights = {
            "xgb": 1.0,
            "decay": 0.0,
            "kalman": 0.0,
        }

        if use_ensemble:
            prod_weights = {
                "xgb": float(
                    row[
                        "production_w_xgb"
                    ]
                ),
                "decay": float(
                    row[
                        "production_w_decay"
                    ]
                ),
                "kalman": float(
                    row[
                        "production_w_kalman"
                    ]
                ),
            }

        selection[target] = {
            "selected_mode": selected_mode,
            "strict_oof_source_column": selected_col,
            "canonical_oof_column": canonical_col,
            "pooled_improvement_vs_xgb_pct": pooled_improvement,
            "positive_strict_folds": positive_folds,
            "strict_fold_count": int(
                len(folds)
            ),
            "recent_2023_2025_mean_improvement_pct": (
                recent_mean_improvement
            ),
            "production_weights": prod_weights,
            "gate": {
                "min_pooled_improvement_pct": (
                    MIN_POOLED_IMPROVEMENT_PCT
                ),
                "min_positive_folds": (
                    MIN_POSITIVE_FOLDS
                ),
                "recent_mean_improvement_must_be_positive": (
                    True
                ),
            },
        }

        print()
        print(
            f"{target.upper():5s} -> "
            f"{selected_mode.upper():8s} | "
            f"pooled={pooled_improvement:+.3f}% | "
            f"positive folds={positive_folds}/{len(folds)} | "
            f"recent mean={recent_mean_improvement:+.3f}%"
        )

        if use_ensemble:
            print(
                "       production weights: "
                f"xgb={prod_weights['xgb']:.4f}, "
                f"decay={prod_weights['decay']:.4f}, "
                f"kalman={prod_weights['kalman']:.4f}"
            )

    canonical_cols = [
        f"mu_selected_{target}"
        for target in TARGETS
    ]

    strict_rows = (
        df[canonical_cols]
        .notna()
        .all(axis=1)
    )

    strict_seasons = sorted(
        df.loc[
            strict_rows,
            "season",
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if strict_seasons != list(
        range(2018, 2026)
    ):
        raise RuntimeError(
            "Canonical selected means do not cover "
            "exactly seasons 2018-2025. "
            f"Observed: {strict_seasons}"
        )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUT_PATH,
        index=False,
    )

    SELECTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "strict_oof_start_season": 2018,
        "strict_oof_end_season": 2025,
        "targets": selection,
    }

    with SELECTION_PATH.open(
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
    print("=" * 104)
    print("FINAL SELECTION")
    print("=" * 104)

    for target in TARGETS:
        item = selection[target]
        print(
            f"{target.upper():5s}: "
            f"{item['selected_mode']}"
        )

    print()
    print(
        f"Strict all-target OOF rows: "
        f"{strict_rows.sum():,}"
    )
    print(
        f"Strict OOF seasons: "
        f"{strict_seasons[0]}-"
        f"{strict_seasons[-1]}"
    )
    print()
    print(
        f"Saved canonical OOF means: "
        f"{OUT_PATH}"
    )
    print(
        f"Saved production selection: "
        f"{SELECTION_PATH}"
    )
    print()
    print(
        "Do not fit marginals yet. "
        "Patch the distribution stage to consume "
        "mu_selected_* next."
    )


if __name__ == "__main__":
    main()
