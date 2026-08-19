from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/copula_audit/combo_crps_2025_per_row.csv"
)
OUTPUT_PATH = Path(
    "data/processed/copula_audit/combo_crps_2025_cluster_bootstrap.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
    )
    return parser.parse_args()


def cluster_bootstrap(
    frame: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict[str, float]:
    # Positive paired difference means the copula has LOWER CRPS.
    work = frame.copy()
    work["paired_gain"] = (
        work["independence_crps"]
        - work["copula_crps"]
    )

    by_game = (
        work.groupby("game_id", as_index=False)
        .agg(
            gain_sum=("paired_gain", "sum"),
            independence_sum=("independence_crps", "sum"),
            row_count=("paired_gain", "size"),
        )
    )

    gains = by_game["gain_sum"].to_numpy(dtype=float)
    independence = by_game[
        "independence_sum"
    ].to_numpy(dtype=float)
    counts = by_game["row_count"].to_numpy(dtype=float)

    n_clusters = len(by_game)
    rng = np.random.default_rng(seed)

    boot_gain = np.empty(reps, dtype=float)
    boot_relative = np.empty(reps, dtype=float)

    for rep in range(reps):
        sampled = rng.integers(
            0,
            n_clusters,
            size=n_clusters,
        )

        total_gain = float(gains[sampled].sum())
        total_independence = float(
            independence[sampled].sum()
        )
        total_rows = float(counts[sampled].sum())

        mean_gain = total_gain / total_rows
        mean_independence = (
            total_independence / total_rows
        )

        boot_gain[rep] = mean_gain
        boot_relative[rep] = (
            100.0
            * mean_gain
            / mean_independence
        )

    observed_gain = float(
        work["paired_gain"].mean()
    )
    observed_independence = float(
        work["independence_crps"].mean()
    )
    observed_relative = (
        100.0
        * observed_gain
        / observed_independence
    )

    gain_ci = np.quantile(
        boot_gain,
        [0.025, 0.975],
    )
    relative_ci = np.quantile(
        boot_relative,
        [0.025, 0.975],
    )

    probability_better = float(
        np.mean(
            boot_gain > 0.0
        )
    )

    if gain_ci[0] > 0:
        verdict = "copula_supported"
    elif gain_ci[1] < 0:
        verdict = "independence_supported"
    else:
        verdict = "inconclusive"

    return {
        "rows": int(len(work)),
        "game_clusters": int(n_clusters),
        "copula_crps": float(
            work["copula_crps"].mean()
        ),
        "independence_crps": float(
            work["independence_crps"].mean()
        ),
        "paired_gain_crps": observed_gain,
        "relative_improvement_pct": observed_relative,
        "gain_ci_low": float(gain_ci[0]),
        "gain_ci_high": float(gain_ci[1]),
        "relative_ci_low_pct": float(
            relative_ci[0]
        ),
        "relative_ci_high_pct": float(
            relative_ci[1]
        ),
        "bootstrap_probability_copula_better": (
            probability_better
        ),
        "verdict": verdict,
    }


def main() -> None:
    args = parse_args()

    if not INPUT_PATH.exists():
        raise SystemExit(
            f"ERROR: missing {INPUT_PATH}. "
            "Run scripts/08_fit_copula.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "game_id",
        "combo",
        "copula_crps",
        "independence_crps",
    }

    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"ERROR: input missing columns: "
            f"{sorted(missing)}"
        )

    rows = []

    for offset, (combo, group) in enumerate(
        df.groupby("combo", sort=False)
    ):
        result = cluster_bootstrap(
            group,
            reps=args.bootstrap_reps,
            seed=args.seed + 1009 * offset,
        )
        result["combo"] = combo
        rows.append(result)

    out = pd.DataFrame(rows)

    columns = [
        "combo",
        "rows",
        "game_clusters",
        "copula_crps",
        "independence_crps",
        "relative_improvement_pct",
        "relative_ci_low_pct",
        "relative_ci_high_pct",
        "bootstrap_probability_copula_better",
        "verdict",
    ]

    print("=" * 120)
    print("2025 COPULA-vs-INDEPENDENCE PAIRED CLUSTER BOOTSTRAP")
    print("=" * 120)
    print(
        f"Bootstrap replicates: {args.bootstrap_reps:,}"
    )
    print(
        "Cluster unit: game_id "
        "(accounts for within-game dependence among sampled player rows)"
    )
    print(
        "Positive relative improvement = lower CRPS for the copula."
    )
    print()

    print(
        out[columns].to_string(
            index=False,
            formatters={
                "copula_crps": lambda x: f"{x:.5f}",
                "independence_crps": (
                    lambda x: f"{x:.5f}"
                ),
                "relative_improvement_pct": (
                    lambda x: f"{x:+.4f}"
                ),
                "relative_ci_low_pct": (
                    lambda x: f"{x:+.4f}"
                ),
                "relative_ci_high_pct": (
                    lambda x: f"{x:+.4f}"
                ),
                "bootstrap_probability_copula_better": (
                    lambda x: f"{x:.3f}"
                ),
            },
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print(
        "Decision rule:"
        "\n  CI entirely > 0  -> copula supported"
        "\n  CI entirely < 0  -> independence supported"
        "\n  CI crosses 0     -> inconclusive; calibrate/shrink dependence "
        "rather than forcing a binary choice."
    )


if __name__ == "__main__":
    main()
