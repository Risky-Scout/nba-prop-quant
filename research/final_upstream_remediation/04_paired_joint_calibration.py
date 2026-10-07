#!/usr/bin/env python
"""Paired remediated-vs-control joint calibration on the untouched holdout.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Gates 9 and 10 ask whether the remediated candidate's multi-leg Brier score is
worse than the accepted bucket repair's. Reading two ``validation_report.json``
files side by side answers that only to within two independent Monte Carlo
errors, which at four legs is an order of magnitude wider than the 0.0005
tolerance the gates use.

The two runs are paired by construction: same held-out games, same seed, same
simulation count, same residual dataset byte for byte, and two baselines that
do not depend on the fitted loadings at all. So the graded conjunctions line up
row for row and a game-clustered paired bootstrap with common random numbers
differences out the shared simulation noise. The pairing is asserted rather
than assumed: the realized outcomes and both baseline probability columns must
agree exactly before any difference is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.validation import (
    brier_score,
    clustered_bootstrap_ci,
)

ARTIFACT_DIR = PROJECT_ROOT / "research" / "final_upstream_remediation"
GRADES_NAME = "joint_event_grades.parquet"

KEY = ["game_id", "family", "n_legs", "event_index"]
BASELINE_COLUMNS = ("p_baseline_independence", "p_baseline_production")

BOOTSTRAP_DRAWS = 4000
SEED = 73

#: Gates 9 and 10. The remediated candidate may not worsen the control's Brier
#: at any leg count by more than this.
MAX_BRIER_DEGRADATION = 0.0005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-grades",
        type=Path,
        required=True,
        help="joint_event_grades.parquet from the accepted bucket repair run",
    )
    parser.add_argument(
        "--candidate-grades",
        type=Path,
        default=ARTIFACT_DIR / GRADES_NAME,
    )
    parser.add_argument(
        "--out", type=Path, default=ARTIFACT_DIR / "paired_joint_calibration.json"
    )
    return parser.parse_args()


def load_grades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing graded conjunctions: {path}")
    frame = pd.read_parquet(path).reset_index(drop=True)
    # The driver emits graded events in a deterministic order per game and
    # family, so position within the group is a stable join key that neither
    # run invents.
    frame["event_index"] = frame.groupby(["game_id", "family", "n_legs"]).cumcount()
    return frame


def assert_paired(control: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if len(control) != len(candidate):
        raise SystemExit(
            f"graded event counts differ: control {len(control)} vs candidate "
            f"{len(candidate)}; the runs are not paired"
        )
    merged = control.merge(
        candidate,
        on=KEY,
        how="inner",
        suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(control):
        raise SystemExit(
            f"only {len(merged)} of {len(control)} graded events matched on "
            f"{KEY}; the runs are not paired"
        )
    if not (merged["realized_control"] == merged["realized_candidate"]).all():
        raise SystemExit("realized outcomes differ between the two runs")
    for column in BASELINE_COLUMNS:
        deviation = float(
            np.abs(merged[f"{column}_control"] - merged[f"{column}_candidate"]).max()
        )
        if deviation > 0.0:
            raise SystemExit(
                f"{column} differs between runs by {deviation:.3e}; the shared "
                "baselines must be identical for the pairing to hold"
            )
    return merged


def paired_delta(group: pd.DataFrame, left: str, right: str) -> dict[str, object]:
    """Brier(left) - Brier(right) with a game-clustered paired bootstrap."""
    columns = ["game_id", left, right, "realized_control"]

    def statistic(sample: pd.DataFrame) -> float:
        outcome = sample["realized_control"].to_numpy(dtype=float)
        return brier_score(
            sample[left].to_numpy(dtype=float), outcome
        ) - brier_score(sample[right].to_numpy(dtype=float), outcome)

    point = statistic(group)
    low, high = clustered_bootstrap_ci(
        group[columns],
        cluster_column="game_id",
        statistic=statistic,
        draws=BOOTSTRAP_DRAWS,
        seed=SEED,
    )
    return {"delta": float(point), "ci95": [float(low), float(high)]}


def main() -> None:
    args = parse_args()
    control = load_grades(args.control_grades)
    candidate = load_grades(args.candidate_grades)
    merged = assert_paired(control, candidate)

    report: dict[str, object] = {
        "title": "paired remediated-vs-control joint calibration",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "scope": "untouched 2024-2025 holdout",
        "control_artifact": str(args.control_grades),
        "candidate_artifact": str(args.candidate_grades),
        "paired": True,
        "events": int(len(merged)),
        "games": int(merged["game_id"].nunique()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "max_brier_degradation": MAX_BRIER_DEGRADATION,
        "by_legs": {},
    }

    verdicts: dict[str, bool] = {}
    for legs, group in merged.groupby("n_legs"):
        outcome = group["realized_control"].to_numpy(dtype=float)
        entry: dict[str, object] = {
            "events": int(len(group)),
            "games": int(group["game_id"].nunique()),
            "base_rate": float(outcome.mean()),
            "brier_control": brier_score(
                group["p_candidate_control"].to_numpy(dtype=float), outcome
            ),
            "brier_candidate": brier_score(
                group["p_candidate_candidate"].to_numpy(dtype=float), outcome
            ),
            "brier_independence": brier_score(
                group["p_baseline_independence_control"].to_numpy(dtype=float),
                outcome,
            ),
            "brier_production": brier_score(
                group["p_baseline_production_control"].to_numpy(dtype=float), outcome
            ),
            "candidate_minus_control": paired_delta(
                group, "p_candidate_candidate", "p_candidate_control"
            ),
            "control_minus_independence": paired_delta(
                group, "p_candidate_control", "p_baseline_independence_control"
            ),
            "candidate_minus_independence": paired_delta(
                group, "p_candidate_candidate", "p_baseline_independence_control"
            ),
        }
        degradation = entry["candidate_minus_control"]["delta"]  # type: ignore[index]
        entry["within_tolerance"] = bool(degradation <= MAX_BRIER_DEGRADATION)
        verdicts[str(int(legs))] = bool(entry["within_tolerance"])
        report["by_legs"][str(int(legs))] = entry  # type: ignore[index]

    report["all_leg_counts_within_tolerance"] = all(verdicts.values())
    report["within_tolerance_by_legs"] = verdicts

    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"{'legs':>4s} {'brier ctrl':>12s} {'brier cand':>12s} "
        f"{'cand-ctrl':>12s} {'ci95 low':>12s} {'ci95 high':>12s}  ok"
    )
    for legs, entry in report["by_legs"].items():  # type: ignore[union-attr]
        delta = entry["candidate_minus_control"]
        print(
            f"{legs:>4s} {entry['brier_control']:12.8f} "
            f"{entry['brier_candidate']:12.8f} {delta['delta']:+12.8f} "
            f"{delta['ci95'][0]:+12.8f} {delta['ci95'][1]:+12.8f}  "
            f"{entry['within_tolerance']}"
        )
    print()
    print(f"{'legs':>4s} {'ctrl - indep':>14s} {'cand - indep':>14s}")
    for legs, entry in report["by_legs"].items():  # type: ignore[union-attr]
        print(
            f"{legs:>4s} {entry['control_minus_independence']['delta']:+14.8f} "
            f"{entry['candidate_minus_independence']['delta']:+14.8f}"
        )
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
