#!/usr/bin/env python3
"""Paired repair-vs-V1 joint calibration comparison on the untouched holdout.

REPAIR GATES 9 and 10 ask whether the repair candidate's multi-leg Brier score
is worse than accepted shadow V1's. Reading the two ``validation_report.json``
files side by side answers that only to within two independent Monte Carlo
errors, which at 4 legs is far wider than the 0.0005 tolerance the gates use.

The two runs are paired by construction: same held-out games, same seed, same
simulation count, same residual dataset, and an independence baseline that does
not depend on the fitted loadings at all. So the graded conjunctions line up
row for row, and a paired game-clustered bootstrap with common random numbers
differences out the shared simulation noise. That is the only comparison tight
enough to decide a 0.0005 threshold, and it needs no re-run of V1.

The script asserts the pairing rather than assuming it: the realized outcomes
and both baseline probability columns must agree between the two frames before
any difference is reported.

SHADOW / RESEARCH ONLY. Reads two committed artifacts and writes one JSON
report. Touches no production code and cannot promote anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.validation import (  # noqa: E402
    brier_score,
    clustered_bootstrap_ci,
)

CONTROL_DIR = PROJECT_ROOT / "research" / "game_latent_state"
REPAIR_DIR = PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"
GRADES_NAME = "joint_event_grades.parquet"

KEY = ["game_id", "family", "n_legs", "event_index"]
BASELINE_COLUMNS = ("p_baseline_independence", "p_baseline_production")

BOOTSTRAP_DRAWS = 4000
SEED = 73

# REPAIR GATE 10: 3-leg and 4-leg Brier may not worsen beyond accepted V1 by
# more than this, and the repair must not make V1's existing small 4-leg
# regression against independence worse.
MAX_MULTI_LEG_BRIER_DEGRADATION = 0.0005


def load_grades(directory: Path) -> pd.DataFrame:
    path = directory / GRADES_NAME
    if not path.exists():
        raise SystemExit(f"missing graded conjunctions: {path}")
    frame = pd.read_parquet(path)
    # The driver emits graded events in a deterministic order per game and
    # family, so position within the group is a stable join key. It is not a
    # column either run invents, which is why it is derived here instead.
    frame = frame.reset_index(drop=True)
    frame["event_index"] = frame.groupby(["game_id", "family", "n_legs"]).cumcount()
    return frame


def assert_paired(control: pd.DataFrame, repair: pd.DataFrame) -> pd.DataFrame:
    """Merge on the event key and verify everything shared is bit-identical."""
    if len(control) != len(repair):
        raise SystemExit(
            f"graded event counts differ: control {len(control)} vs repair "
            f"{len(repair)}; the runs are not paired"
        )
    merged = control.merge(
        repair,
        on=KEY,
        how="inner",
        suffixes=("_control", "_repair"),
        validate="one_to_one",
    )
    if len(merged) != len(control):
        raise SystemExit(
            f"only {len(merged)} of {len(control)} graded events matched on "
            f"{KEY}; the runs are not paired"
        )
    if not (merged["realized_control"] == merged["realized_repair"]).all():
        raise SystemExit("realized outcomes differ between the two runs")
    for column in BASELINE_COLUMNS:
        deviation = float(
            np.abs(merged[f"{column}_control"] - merged[f"{column}_repair"]).max()
        )
        if deviation > 0.0:
            raise SystemExit(
                f"{column} differs between runs by {deviation:.3e}; the shared "
                "baselines must be identical for the pairing to hold"
            )
    return merged


def paired_delta(group: pd.DataFrame, left: str, right: str) -> dict[str, float]:
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
    control = load_grades(CONTROL_DIR)
    repair = load_grades(REPAIR_DIR)
    merged = assert_paired(control, repair)

    report: dict[str, object] = {
        "scope": "paired repair-vs-V1 joint calibration, untouched 2024-2025",
        "control_artifact": str(CONTROL_DIR.relative_to(PROJECT_ROOT) / GRADES_NAME),
        "repair_artifact": str(REPAIR_DIR.relative_to(PROJECT_ROOT) / GRADES_NAME),
        "paired": True,
        "events": int(len(merged)),
        "games": int(merged["game_id"].nunique()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "max_multi_leg_brier_degradation": MAX_MULTI_LEG_BRIER_DEGRADATION,
        "by_legs": {},
    }

    verdicts: dict[str, bool] = {}
    for legs, group in merged.groupby("n_legs"):
        outcome = group["realized_control"].to_numpy(dtype=float)
        entry: dict[str, object] = {
            "events": int(len(group)),
            "games": int(group["game_id"].nunique()),
            "base_rate": float(outcome.mean()),
            "brier_v1": brier_score(
                group["p_candidate_control"].to_numpy(dtype=float), outcome
            ),
            "brier_repair": brier_score(
                group["p_candidate_repair"].to_numpy(dtype=float), outcome
            ),
            "brier_independence": brier_score(
                group["p_baseline_independence_control"].to_numpy(dtype=float),
                outcome,
            ),
            "brier_production": brier_score(
                group["p_baseline_production_control"].to_numpy(dtype=float), outcome
            ),
            "repair_minus_v1": paired_delta(
                group, "p_candidate_repair", "p_candidate_control"
            ),
            "v1_minus_independence": paired_delta(
                group, "p_candidate_control", "p_baseline_independence_control"
            ),
            "repair_minus_independence": paired_delta(
                group, "p_candidate_repair", "p_baseline_independence_control"
            ),
        }
        degradation = entry["repair_minus_v1"]["delta"]  # type: ignore[index]
        entry["within_tolerance"] = bool(
            degradation <= MAX_MULTI_LEG_BRIER_DEGRADATION
        )
        verdicts[str(int(legs))] = entry["within_tolerance"]
        report["by_legs"][str(int(legs))] = entry  # type: ignore[index]

    report["all_leg_counts_within_tolerance"] = all(verdicts.values())

    out = REPAIR_DIR / "paired_joint_calibration.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"{'legs':>4s} {'brier V1':>11s} {'brier rep':>11s} {'rep-V1':>11s} "
          f"{'ci95 low':>11s} {'ci95 high':>11s}  ok")
    for legs, entry in report["by_legs"].items():  # type: ignore[union-attr]
        delta = entry["repair_minus_v1"]
        print(
            f"{legs:>4s} {entry['brier_v1']:11.8f} {entry['brier_repair']:11.8f} "
            f"{delta['delta']:+11.8f} {delta['ci95'][0]:+11.8f} "
            f"{delta['ci95'][1]:+11.8f}  {entry['within_tolerance']}"
        )
    print()
    print(f"{'legs':>4s} {'V1 - indep':>12s} {'repair - indep':>15s}")
    for legs, entry in report["by_legs"].items():  # type: ignore[union-attr]
        print(
            f"{legs:>4s} {entry['v1_minus_independence']['delta']:+12.8f} "
            f"{entry['repair_minus_independence']['delta']:+15.8f}"
        )
    print()
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
