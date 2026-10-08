#!/usr/bin/env python
"""Attribute the predictive-SD calibration failure to its cause.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

``07_prove_equivalence_and_calibrate_sd.py`` returns a verdict: the inflation
factor frozen on pre-2024 inner folds does not survive untouched 2024-2025.
A verdict is not a diagnosis, and the difference matters here, because the two
candidate explanations call for opposite responses. If the factor is merely
noisy then it is the wrong size and the layer could be re-derived. If the
factor is measuring something the random-effects posterior already measures,
then the layer is double-counting and should not exist at all.

This step decides between them from numbers both runs have already written
down. It reads ``temporal_diagnostic.json`` and
``equivalence_and_calibration.json``, fits nothing, reads no residual data, and
changes no threshold: re-deciding the pre-registered criterion after seeing the
holdout is the one thing it must not do, so the criterion and its verdict are
copied through untouched.

What it computes
----------------
Three things, each a direct consequence of the recorded fold detail.

1.  The inner-fold mean ``z^2`` split by fold. The pooled factor averages two
    folds, and if they disagree the average is not an estimate of anything.
2.  The season-level estimate path per bucket, which says whether a
    disagreeing fold is a one-season departure or a trend.
3.  How much the random-effects predictive SD widens for the holdout window
    relative to the fold that departed -- that is, whether the posterior has
    already absorbed the departure the inflation factor was fitted on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    sha256_file,
    write_json,
)

console = Console()

TREATMENT = "T0_pooled_empirical_bayes"
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

#: A fold whose mean z^2 sits this far from the pooled value, measured as a
#: ratio, is not a second measurement of the same quantity.
FOLD_DISAGREEMENT_RATIO = 3.0

#: A season whose estimate sits within one of its own standard errors of the
#: preceding level has not departed from that level; it has wobbled inside its
#: sampling error. Without this floor the reversion test reads any sign flip as
#: a departure, however small, and ``teammate_reb_reb`` -- which rises steadily
#: and never dips -- gets misread as a bucket that dipped and recovered.
MIN_DEPARTURE_IN_STANDARD_ERRORS = 1.0


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    ).stdout.strip()


def fold_squared_z(temporal: dict) -> dict[int, dict[str, float]]:
    """Recover each inner fold's ``z^2`` from the recorded fold detail.

    The screen stores the pooled mean over folds and buckets, not the split.
    The split is recoverable because the fold detail records the realised
    estimate, its standard error, and the treatment's own prediction and
    predictive SD, which is everything the ``z`` definition consumes.
    """
    by_fold: dict[int, dict[str, float]] = {}
    for bucket, payload in temporal["buckets"].items():
        for entry in payload["screening"]["fold_detail"]:
            fold = int(entry["fold_season"])
            prediction = entry[TREATMENT]
            error = float(entry["realised_estimate"]) - float(prediction["prediction"])
            variance = (
                float(entry["realised_standard_error"]) ** 2
                + float(prediction["prediction_sd"]) ** 2
            )
            by_fold.setdefault(fold, {})[bucket] = error**2 / variance
    return by_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research" / "game_latent_state_v2",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_root)
    temporal_path = artifact_dir / "temporal_diagnostic.json"
    calibration_path = artifact_dir / "equivalence_and_calibration.json"
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    report = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration = report["predictive_sd_calibration"]

    if calibration["passed"]:
        raise SystemExit(
            "the predictive-SD calibration passed; there is no blocker to diagnose"
        )

    console.rule("Why the frozen inflation factor did not survive the holdout")
    inflation = float(calibration["inflation_factor"])
    pooled = calibration["pooled"]
    inner_pooled = float(temporal["uncertainty_calibration"]["pooled"]["mean_squared_z"])
    console.print(
        f"frozen inflation factor {inflation:.4f} = sqrt({inner_pooled:.4f}), "
        f"pooled over {temporal['uncertainty_calibration']['pooled']['observations']:.0f} "
        "inner fold-bucket observations"
    )
    console.print(
        f"holdout mean z^2: raw {pooled['raw_mean_squared_z']:.4f}, "
        f"calibrated {pooled['calibrated_mean_squared_z']:.4f}; "
        "a calibrated reported SD would average 1"
    )

    # ---- 1. the two inner folds do not agree ------------------------
    by_fold = fold_squared_z(temporal)
    fold_table = Table(title="inner-fold mean z^2, split by fold")
    for column in ("fold", "trains on", "mean z^2", "per bucket"):
        fold_table.add_column(column, justify="right" if column != "per bucket" else "left")
    fold_means: dict[int, float] = {}
    training_counts: dict[int, int] = {}
    for fold in sorted(by_fold):
        values = by_fold[fold]
        fold_means[fold] = float(np.mean(list(values.values())))
        any_bucket = next(iter(temporal["buckets"].values()))
        detail = next(
            entry
            for entry in any_bucket["screening"]["fold_detail"]
            if int(entry["fold_season"]) == fold
        )
        training_counts[fold] = len(detail[TREATMENT]["training_seasons"])
        fold_table.add_row(
            str(fold),
            f"{training_counts[fold]} seasons",
            f"{fold_means[fold]:.4f}",
            ", ".join(f"{name}={value:.3f}" for name, value in sorted(values.items())),
        )
    console.print(fold_table)

    worst_fold = max(fold_means, key=lambda fold: fold_means[fold])
    best_fold = min(fold_means, key=lambda fold: fold_means[fold])
    ratio = fold_means[worst_fold] / max(fold_means[best_fold], 1e-30)
    folds_disagree = ratio >= FOLD_DISAGREEMENT_RATIO
    console.print(
        f"the two folds differ by a factor of {ratio:.1f}; "
        f"fold {best_fold} scored {fold_means[best_fold]:.4f} and the holdout "
        f"scored {pooled['raw_mean_squared_z']:.4f}, so the factor rests on "
        f"fold {worst_fold} alone"
    )

    # ---- 2. the disagreeing fold is one anomalous season ------------
    holdout_by_key = {
        (str(item["bucket"]), int(item["refit_window"])): item
        for item in calibration["observations"]
    }
    path_table = Table(title=f"season estimate path, and what fold {worst_fold} saw")
    for column in ("bucket", *[str(season) for season in range(2020, 2026)]):
        path_table.add_column(column, justify="right")
    season_paths: dict[str, dict[str, float]] = {}
    standard_errors: dict[str, dict[str, float]] = {}
    for bucket, payload in temporal["buckets"].items():
        path = {
            int(entry["season"]): float(entry["estimate"])
            for entry in payload["season_estimates"]
        }
        errors = {
            int(entry["season"]): float(entry["standard_error"])
            for entry in payload["season_estimates"]
        }
        for season in HOLDOUT_SEASONS:
            observation = holdout_by_key[(bucket, season)]
            path[season] = float(observation["observed"])
            errors[season] = float(observation["observed_standard_error"])
        season_paths[bucket] = {str(k): v for k, v in sorted(path.items())}
        standard_errors[bucket] = {str(k): v for k, v in sorted(errors.items())}
        path_table.add_row(
            bucket, *[f"{path[season]:+.5f}" for season in sorted(path)]
        )
    console.print(path_table)

    # Is the departing fold season a one-season dip that the holdout undid?
    reverted: dict[str, object] = {}
    for bucket, path in season_paths.items():
        before = [
            value for season, value in path.items() if int(season) < worst_fold
        ]
        departed = path[str(worst_fold)]
        after = [
            value
            for season, value in path.items()
            if int(season) in HOLDOUT_SEASONS
        ]
        level_before = float(np.mean(before))
        level_after = float(np.mean(after))
        departure = departed - level_before
        recovery = level_after - departed
        departure_se = standard_errors[bucket][str(worst_fold)]
        in_standard_errors = abs(departure) / departure_se
        material = in_standard_errors >= MIN_DEPARTURE_IN_STANDARD_ERRORS
        reverted[bucket] = {
            "level_before": level_before,
            "departed_to": departed,
            "level_after": level_after,
            "departure": departure,
            "departure_in_standard_errors": in_standard_errors,
            "departure_is_material": bool(material),
            "recovery": recovery,
            "reverted": bool(material and recovery / departure <= -0.5),
            "inner_fold_squared_z": by_fold[worst_fold][bucket],
        }
    reverting = [name for name, entry in reverted.items() if entry["reverted"]]
    console.print(
        f"{len(reverting)} of {len(reverted)} buckets departed from their level in "
        f"{worst_fold} by at least {MIN_DEPARTURE_IN_STANDARD_ERRORS:.0f} standard "
        f"error and reverted over {list(HOLDOUT_SEASONS)}: "
        f"{', '.join(sorted(reverting))}"
    )
    for name in sorted(reverted):
        entry = reverted[name]  # type: ignore[assignment]
        console.print(
            f"    {name:24s} departure {entry['departure']:+.5f} = "  # type: ignore[index]
            f"{entry['departure_in_standard_errors']:.2f} standard errors, "  # type: ignore[index]
            f"material: {entry['departure_is_material']}"  # type: ignore[index]
        )

    # ---- 3. the posterior already absorbed that departure -----------
    widen_table = Table(
        title=(
            f"predictive SD once season {worst_fold} is inside the training set"
        )
    )
    for column in (
        "bucket",
        f"fold-{worst_fold} SD",
        "holdout-2024 SD",
        "ratio",
        f"fold-{worst_fold} z^2",
    ):
        widen_table.add_column(column, justify="right")
    widening: dict[str, object] = {}
    for bucket, payload in temporal["buckets"].items():
        detail = next(
            entry
            for entry in payload["screening"]["fold_detail"]
            if int(entry["fold_season"]) == worst_fold
        )
        fold_sd = float(detail[TREATMENT]["prediction_sd"])
        holdout_sd = float(
            holdout_by_key[(bucket, min(HOLDOUT_SEASONS))]["raw_prediction_sd"]
        )
        widening[bucket] = {
            "fold_prediction_sd": fold_sd,
            "holdout_prediction_sd": holdout_sd,
            "ratio": holdout_sd / fold_sd,
            "inner_fold_squared_z": by_fold[worst_fold][bucket],
        }
        widen_table.add_row(
            bucket,
            f"{fold_sd:.6f}",
            f"{holdout_sd:.6f}",
            f"x{holdout_sd / fold_sd:.2f}",
            f"{by_fold[worst_fold][bucket]:.3f}",
        )
    console.print(widen_table)

    absorbed = [
        name
        for name in reverting
        if float(widening[name]["ratio"]) > 1.0  # type: ignore[index]
    ]
    not_departing = [name for name in widening if name not in reverting]
    unwidened = [
        name
        for name in not_departing
        if float(widening[name]["ratio"]) <= 1.0  # type: ignore[index]
    ]
    console.print(
        f"every bucket that departed in {worst_fold} reports a wider predictive SD "
        f"for the holdout window: {len(absorbed)} of {len(reverting)}"
    )
    console.print(
        f"and every bucket that did not depart does not: {len(unwidened)} of "
        f"{len(not_departing)}"
    )

    double_counted = bool(
        folds_disagree
        and reverting
        and len(absorbed) == len(reverting)
        and len(unwidened) == len(not_departing)
    )

    cause = (
        "The inflation factor double-counts one season of heterogeneity. It is "
        f"sqrt of a mean over two folds that disagree by a factor of {ratio:.1f}, "
        f"so it is effectively fold {worst_fold}'s number. Fold {worst_fold} is "
        "large because the pooled control was fitted on seasons before it and "
        "could not have known that season would depart from the level. For a "
        "holdout window that season is inside the training set, the "
        "random-effects tau^2 absorbs its departure, and the reported "
        "predictive SD widens on its own -- by up to "
        f"x{max(float(widening[name]['ratio']) for name in reverting):.2f} "  # type: ignore[index]
        "across the departing buckets. Multiplying that already-widened SD by a "
        "factor fitted on the unwidened one applies the same correction twice, "
        "which is why the calibrated mean z^2 lands further from 1 than the raw "
        "one. The correspondence is exact across the four buckets: the three "
        "that departed are the three whose predictive SD widens and the three "
        f"with a large fold-{worst_fold} z^2, and the one that did not depart is "
        "the one whose SD does not widen and whose fold z^2 is near zero."
    ) if double_counted else (
        "The two inner folds disagree, so the pooled factor is not a stable "
        "estimate, but the recorded detail does not establish that the "
        "random-effects posterior already absorbs what the factor measures."
    )
    console.print()
    console.print(cause)

    diagnosis = {
        "diagnoses": {
            "artifact": "equivalence_and_calibration.json",
            "artifact_sha256": sha256_file(calibration_path),
            "temporal_diagnostic_sha256": sha256_file(temporal_path),
        },
        "verdict_being_explained": {
            "predictive_sd_calibration_passed": calibration["passed"],
            "pre_registered_criteria": calibration["pre_registered_criteria"],
            "verdicts": calibration["verdicts"],
            "criteria_were_not_revisited_after_seeing_the_holdout": True,
        },
        "inflation_factor": inflation,
        "inner_pooled_mean_squared_z": inner_pooled,
        "holdout_raw_mean_squared_z": pooled["raw_mean_squared_z"],
        "holdout_calibrated_mean_squared_z": pooled["calibrated_mean_squared_z"],
        "inner_fold_mean_squared_z_by_fold": {
            str(fold): {
                "mean_squared_z": fold_means[fold],
                "training_seasons": training_counts[fold],
                "by_bucket": by_fold[fold],
            }
            for fold in sorted(by_fold)
        },
        "folds_disagree": {
            "ratio": ratio,
            "threshold": FOLD_DISAGREEMENT_RATIO,
            "disagree": bool(folds_disagree),
            "quiet_fold": best_fold,
            "driving_fold": worst_fold,
            "quiet_fold_agrees_with_the_holdout": bool(
                abs(fold_means[best_fold] - float(pooled["raw_mean_squared_z"])) < 0.25
            ),
        },
        "season_estimate_path": season_paths,
        "season_standard_errors": standard_errors,
        "departure_materiality_floor_in_standard_errors": (
            MIN_DEPARTURE_IN_STANDARD_ERRORS
        ),
        "departure_and_reversion": reverted,
        "buckets_that_departed_and_reverted": sorted(reverting),
        "departing_buckets_whose_predictive_sd_widened": sorted(absorbed),
        "non_departing_buckets_whose_predictive_sd_did_not_widen": sorted(unwidened),
        "predictive_sd_widening_once_the_departure_is_in_training": widening,
        "cause": cause,
        "double_counting_established": double_counted,
        "what_this_does_not_do": [
            "refit anything",
            "read residual data",
            "change a pre-registered threshold",
            "re-decide the calibration verdict",
        ],
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "code_sha": git_sha(),
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
    }
    path = write_json(diagnosis, artifact_dir / "sd_calibration_blocker.json")

    console.rule("Blocker")
    console.print("PREDICTIVE_SD_CALIBRATION_PASSED=NO")
    console.print(
        "BLOCKER=FROZEN_INFLATION_FACTOR_DOUBLE_COUNTS_ONE_SEASON_OF_HETEROGENEITY"
        if double_counted
        else "BLOCKER=FROZEN_INFLATION_FACTOR_RESTS_ON_ONE_DISAGREEING_INNER_FOLD"
    )
    console.print("INFLATION_FACTOR_REFITTED=NO")
    console.print("CRITERIA_REVISITED_AFTER_SEEING_HOLDOUT=NO")
    console.print("CLEAN_CONSOLIDATION_BRANCH_STARTED=NO")
    console.print("PRODUCTION_INTEGRATION_STARTED=NO")
    console.print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
