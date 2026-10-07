#!/usr/bin/env python
"""Final audit: uncertainty, evaluation universes, count RMSE and provenance.

SHADOW / RESEARCH ONLY. AUDIT ONLY -- this driver fits nothing, selects
nothing and simulates nothing. It reads committed artifacts, reproduces the
stored statistics from their stored inputs, and reports where two numbers that
look comparable are not.

The only arithmetic performed here that is not already in an artifact is:

* the leave-one-out recomputation of the item-6 coverage table, which reuses
  the same stored standardized errors;
* log loss from the stored joint-event probabilities and outcomes;
* the two held-out game subsets, from the validator's own selection rule;
* a deterministic equivalence check on the half-life dial.

No model parameter is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nba_prop_quant.research.game_latent_state.artifacts import git_sha
from nba_prop_quant.research.game_latent_state.remediation import (
    COVERAGE_LEVELS,
    MIN_CELL_GAMES,
    MIN_CELL_PAIRS,
    SeasonSeries,
    coverage_loss,
    coverage_table,
    fit_student_t_random_effects,
)

ARTIFACT_DIR = PROJECT_ROOT / "research" / "final_upstream_remediation"
REPAIR_DIR = PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"
SHADOW_DIR = PROJECT_ROOT / "research" / "game_latent_state"
V2_DIR = PROJECT_ROOT / "research" / "game_latent_state_v2"

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS = (2024, 2025)

ROLE_COLUMN = "role_bucket"

LOG_LOSS_CLIP = 1e-6
BOOTSTRAP_DRAWS = 4000
CELL_BOOTSTRAP_DRAWS = 400
SEED = 73

#: The four buckets the V2 structural round's SD-calibration blocker scored.
#: Quoted here only to show that the mean z^2 the brief cites comes from a
#: different universe than the coverage table it is being compared against.
V2_BLOCKER_BUCKETS = (
    "passer_ast_teammate_pts",
    "teammate_ast_ast",
    "teammate_pts_reb",
    "teammate_reb_reb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--control-grades",
        type=Path,
        default=Path("/tmp/control_validation/joint_event_grades.parquet"),
    )
    parser.add_argument(
        "--residuals", type=Path, default=SHADOW_DIR / "oof_gaussian_residuals.parquet"
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    if not path.exists():
        return "ABSENT"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rule(title: str) -> None:
    print()
    print(f"=== {title} ===")


# ======================================================================
# 1. uncertainty
# ======================================================================


def standardized_records(inner: dict) -> pd.DataFrame:
    """Reproduce item 6's standardized errors from the stored forward records.

    ``decide_uncertainty`` divides each forward error by the root of the sum of
    the predictive variance for an unseen season and the sampling variance of
    the observed next-season estimate, then scores only the folds that had an
    earlier fold to calibrate on. Both steps are reproduced here rather than
    re-derived, so a mismatch against the stored table is a finding.
    """
    item1 = inner["item_1_temporal"]
    selected = item1["selected"]
    frame = pd.DataFrame(item1["forward_records"][selected]).copy()
    frame["denominator"] = np.sqrt(
        frame["predictive_sd"] ** 2 + frame["observed_se"] ** 2
    )
    frame["standardized"] = frame["error"] / frame["denominator"]
    folds = sorted(frame["target_season"].unique())
    frame["fold_position"] = frame["target_season"].map(
        {fold: position for position, fold in enumerate(folds)}
    )
    frame["cross_fitted"] = frame["fold_position"] > 0
    frame["scale_u0_raw"] = 1.0
    frame["calibrated"] = frame["standardized"] / frame["scale_u0_raw"]
    return frame


def uncertainty_audit(inner: dict) -> dict[str, object]:
    frame = standardized_records(inner)
    scored = frame.loc[frame["cross_fitted"]].reset_index(drop=True)
    z = scored["calibrated"].to_numpy(float)

    reproduced = coverage_table(z)
    stored = inner["item_6_uncertainty"]["candidates"]["U0_raw"]["coverage"]
    matches = {
        key: bool(np.isclose(reproduced[key], stored[key], rtol=0.0, atol=1e-12))
        for key in stored
    }

    # Leave-one-out over the scored observations.
    loo: list[dict[str, object]] = []
    for position in range(len(z)):
        kept = np.delete(z, position)
        table = coverage_table(kept)
        loo.append(
            {
                "removed_bucket": scored.loc[position, "bucket"],
                "removed_z": float(z[position]),
                "mean_squared_z": table["mean_squared_z"],
                "coverage_loss": coverage_loss(table),
                **{
                    f"coverage_{int(round(level * 100))}": table[
                        f"coverage_{int(round(level * 100))}"
                    ]
                    for level in COVERAGE_LEVELS
                },
            }
        )
    loo_frame = pd.DataFrame(loo)

    absolute = np.abs(z)
    # The coverage estimate at each level is a binomial proportion over the
    # scored observations. Its standard error is what decides whether the
    # shortfall is distinguishable from sampling noise at this sample size --
    # and only if the observations were independent, which they are not.
    binomial: dict[str, dict[str, float]] = {}
    for level in COVERAGE_LEVELS:
        key = f"coverage_{int(round(level * 100))}"
        observed = float(reproduced[key])
        se = float(np.sqrt(level * (1.0 - level) / len(z)))
        binomial[key] = {
            "nominal": level,
            "observed": observed,
            "successes": float(observed * len(z)),
            "trials": float(len(z)),
            "binomial_se_if_independent": se,
            "z_against_nominal": (observed - level) / se if se > 0 else float("nan"),
            "nominal_inside_95_interval": bool(abs(observed - level) <= 1.96 * se),
        }

    denominators = scored["denominator"].to_numpy(float)
    subset = scored.loc[scored["bucket"].isin(V2_BLOCKER_BUCKETS)]
    both_folds = frame.loc[frame["bucket"].isin(V2_BLOCKER_BUCKETS)]

    return {
        "unit_of_observation": (
            "one (dependence bucket, forward target season) pair: a "
            "bucket-level seasonal estimate. Not a game, not a player-game."
        ),
        "z_definition": (
            "z = (observed next-season bucket - prediction from strictly "
            "earlier seasons) / sqrt(predictive_sd^2 + observed_se^2)"
        ),
        "predictive_sd_in_u0_raw": (
            "the temporal model's predictive SD for an unseen season, "
            "sqrt(var(level) + tau^2 * nu/(nu-2)) under A2, combined in "
            "quadrature with the game-clustered sampling SE of the observed "
            "next-season estimate. U0_raw multiplies it by exactly 1.0."
        ),
        "interval_formula": (
            "prediction +/- Phi^-1(0.5 + level/2) * sqrt(predictive_sd^2 + "
            "observed_se^2); covered iff |z| <= Phi^-1(0.5 + level/2)"
        ),
        "mean_squared_z_formula": "mean over the scored observations of z^2",
        "coverage_and_mean_z2_share_observations": True,
        "coverage_levels": list(COVERAGE_LEVELS),
        "critical_values": {
            f"coverage_{int(round(level * 100))}": float(
                stats.norm.ppf(0.5 + level / 2.0)
            )
            for level in COVERAGE_LEVELS
        },
        "forward_records_total": int(len(frame)),
        "folds_present": [int(value) for value in sorted(frame["target_season"].unique())],
        "folds_scored": [
            int(value) for value in sorted(scored["target_season"].unique())
        ],
        "folds_dropped_as_not_cross_fitted": [
            int(value)
            for value in sorted(
                frame.loc[~frame["cross_fitted"], "target_season"].unique()
            )
        ],
        "scored_observations": int(len(scored)),
        "scored_buckets": sorted(str(value) for value in scored["bucket"].unique()),
        "scored_bucket_count": int(scored["bucket"].nunique()),
        "training_seasons_behind_the_scored_fold": sorted(
            int(value) for value in scored["training_seasons"].unique()
        ),
        "independence": (
            "NOT independent. All scored observations share one target season "
            "and therefore the same games; the buckets are different stat "
            "pairs measured on overlapping player pairs within those games. "
            "The effective sample size in the time dimension is one season, "
            "not twelve observations."
        ),
        "effective_independent_seasons": 1,
        "reproduced_coverage": reproduced,
        "stored_coverage": stored,
        "reproduction_matches_stored": matches,
        "all_reproduced": all(matches.values()),
        "binomial_reading": binomial,
        "z_by_observation": [
            {
                "bucket": str(row.bucket),
                "target_season": int(row.target_season),
                "observed": float(row.observed),
                "prediction": float(row.prediction),
                "error": float(row.error),
                "predictive_sd": float(row.predictive_sd),
                "observed_se": float(row.observed_se),
                "denominator": float(row.denominator),
                "z": float(row.calibrated),
                "squared_z": float(row.calibrated**2),
            }
            for row in scored.itertuples()
        ],
        "z_distribution": {
            "median_abs_z": float(np.median(absolute)),
            "p90_abs_z": float(np.percentile(absolute, 90)),
            "max_abs_z": float(np.max(absolute)),
            "mean_squared_z": float(np.mean(z**2)),
            "median_squared_z": float(np.median(z**2)),
            "mean_to_median_squared_z_ratio": float(
                np.mean(z**2) / np.median(z**2)
            ),
            "share_of_sum_squares_from_largest": float(
                np.max(z**2) / np.sum(z**2)
            ),
            "share_of_sum_squares_from_largest_two": float(
                np.sum(np.sort(z**2)[-2:]) / np.sum(z**2)
            ),
        },
        "predictive_sd_heterogeneity": {
            "min_denominator": float(denominators.min()),
            "max_denominator": float(denominators.max()),
            "ratio_max_over_min": float(denominators.max() / denominators.min()),
        },
        "leave_one_out": loo,
        "leave_one_out_envelope": {
            "mean_squared_z": [
                float(loo_frame["mean_squared_z"].min()),
                float(loo_frame["mean_squared_z"].max()),
            ],
            "coverage_loss": [
                float(loo_frame["coverage_loss"].min()),
                float(loo_frame["coverage_loss"].max()),
            ],
            **{
                f"coverage_{int(round(level * 100))}": [
                    float(loo_frame[f"coverage_{int(round(level * 100))}"].min()),
                    float(loo_frame[f"coverage_{int(round(level * 100))}"].max()),
                ]
                for level in COVERAGE_LEVELS
            },
        },
        "mean_z2_crosses_one_under_any_single_removal": bool(
            (loo_frame["mean_squared_z"] < 1.0).any()
        ),
        "the_quoted_0_665_is_a_different_universe": {
            "quoted_raw_mean_squared_z": 0.6652266156010309,
            "quoted_calibrated_mean_squared_z": 0.390769082372168,
            "source_artifact": "research/game_latent_state_v2/sd_calibration_blocker.json",
            "source_seasons": "2024-2025 holdout",
            "source_buckets": list(V2_BLOCKER_BUCKETS),
            "source_bucket_count": len(V2_BLOCKER_BUCKETS),
            "source_round": "V2 structural refinement (rejected inflation factor)",
            "final_item_6_seasons": "pre-2024 forward fold 2023",
            "final_item_6_bucket_count": int(scored["bucket"].nunique()),
            "same_observations": False,
            "note": (
                "The coverage table the brief quotes and the mean z^2 of "
                "0.665 are not computed on the same observations, the same "
                "seasons, the same buckets or by the same code. Within the "
                "final artifact, coverage and mean z^2 are computed on the "
                "same scored observations and both say the same thing."
            ),
        },
        "same_four_buckets_on_the_scored_fold": {
            "observations": int(len(subset)),
            "mean_squared_z": float(np.mean(subset["calibrated"].to_numpy(float) ** 2)),
        },
        "same_four_buckets_both_folds": {
            "observations": int(len(both_folds)),
            "mean_squared_z": float(
                np.mean(both_folds["standardized"].to_numpy(float) ** 2)
            ),
        },
        "all_twelve_buckets_both_folds": {
            "observations": int(len(frame)),
            "mean_squared_z": float(
                np.mean(frame["standardized"].to_numpy(float) ** 2)
            ),
        },
        "tension_resolution": {
            "apparent_tension": (
                "mean z^2 of 0.665 reads as intervals that are too wide, "
                "while coverage of 50/66.7/75/83.3 against 68/80/90/95 reads "
                "as intervals that are too narrow"
            ),
            "resolution": "DIFFERENT EVALUATION UNIVERSES",
            "explanation": (
                "the two numbers were never computed on the same "
                "observations. 0.665 is the V2 structural round's raw mean "
                "z^2 on the 2024-2025 holdout over four buckets; the "
                "coverage table is the final round's item 6 on the pre-2024 "
                "cross-fitted fold over twelve buckets. The final artifact's "
                "own mean z^2 on the observations behind the coverage table "
                "is 2.670186, which is greater than one and therefore agrees "
                "with the under-coverage instead of contradicting it. There "
                "is no internal inconsistency to resolve."
            ),
            "ruled_out": {
                "small_sample": (
                    "contributing but not causal: the sign of the "
                    "miscalibration is stable under every single-observation "
                    "removal, with leave-one-out mean z^2 never below one"
                ),
                "heavy_tails": (
                    "present and material: mean z^2 is 3.8 times the median "
                    "and two of twelve observations carry 72% of the sum of "
                    "squares"
                ),
                "one_or_two_extreme_folds_or_buckets": (
                    "present: passer_ast_teammate_pts at z -3.77 and "
                    "teammate_ast_ast at z -2.97 dominate"
                ),
                "unequal_predictive_sds": (
                    "not causal: the twelve denominators span 0.00217 to "
                    "0.00350, a factor of 1.6, far too narrow to manufacture "
                    "a factor-of-four discrepancy"
                ),
                "dependence": (
                    "material for the standard errors: all twelve "
                    "observations come from one target season, so the "
                    "binomial standard errors quoted against the coverage "
                    "levels are optimistic and the effective number of "
                    "independent seasons is one"
                ),
                "different_evaluation_universes": (
                    "this is the cause of the quoted tension"
                ),
            },
            "direction_within_the_final_artifact": (
                "intervals are too NARROW on the scored fold: mean z^2 "
                "2.670186 implies the predictive standard deviation is "
                "understated by about a factor of 1.63, which is exactly the "
                "direction the coverage shortfall reports"
            ),
            "direction_is_not_stable_across_universes": (
                "the same statistic is 0.665 on the 2024-2025 holdout and "
                "3.118 on the inner pooled fit. The sign of the error flips "
                "between universes, which is the clearest evidence that "
                "twelve bucket-seasonal observations cannot pin a scale"
            ),
        },
        "verdict": (
            "UNCERTAINTY AUDIT CAUTION -- RAW INTERVALS BEST AVAILABLE BUT "
            "SAMPLE TOO SMALL / TAIL-SENSITIVE"
        ),
        "verdict_basis": {
            "not_pass_because": (
                "mean z^2 is 2.670186 on the scored observations and every "
                "nominal coverage level is missed low, so the intervals are "
                "not demonstrably adequate"
            ),
            "not_fail_because": (
                "no single coverage shortfall is significant even against "
                "the optimistic independent-binomial standard error, the "
                "scale is driven by two of twelve observations, the twelve "
                "observations are one season's worth of correlated bucket "
                "estimates, and the three fitted alternatives all scored "
                "worse on the same folds, so no better-calibrated option was "
                "available to select"
            ),
            "no_recalibration_performed": True,
            "no_replacement_calibration_invented": True,
        },
    }


# ======================================================================
# 2. the two held-out game universes
# ======================================================================


def evenly_spaced(game_ids: list[int], count: int) -> list[int]:
    if count and len(game_ids) > count:
        picks = np.linspace(0, len(game_ids) - 1, count)
        return [game_ids[round(index)] for index in picks]
    return list(game_ids)


def game_universes(residuals: pd.DataFrame) -> dict[str, object]:
    per_season: dict[str, object] = {}
    small: set[int] = set()
    large: set[int] = set()
    available_total: set[int] = set()
    for season in HOLDOUT_SEASONS:
        available = sorted(
            int(value)
            for value in residuals.loc[residuals["season"] == season, "game_id"].unique()
        )
        picked_small = evenly_spaced(available, 200)
        picked_large = evenly_spaced(available, 300)
        per_season[str(season)] = {
            "games_available": len(available),
            "picked_200": len(picked_small),
            "picked_300": len(picked_large),
            "intersection": len(set(picked_small) & set(picked_large)),
            "first_five_of_200": picked_small[:5],
            "first_five_of_300": picked_large[:5],
        }
        small |= set(picked_small)
        large |= set(picked_large)
        available_total |= set(available)
    return {
        "rule_location": (
            "research/game_latent_state/04_validate_shadow_v1.py, the "
            "--games-per-season branch inside simulate_season"
        ),
        "rule": (
            "sorted held-out game ids of the season, subsampled at "
            "numpy.linspace(0, n-1, games_per_season) rounded to an index"
        ),
        "rule_depends_only_on": ["season", "game_id ordering", "games_per_season"],
        "rule_depends_on_model_or_performance": False,
        "driver_default_games_per_season": 200,
        "published_repair_invocation": 300,
        "paired_runs_invocation": "default (200), not overridden",
        "total_600_universe": len(large),
        "total_400_universe": len(small),
        "intersection": len(small & large),
        "in_400_not_in_600": len(small - large),
        "in_600_not_in_400": len(large - small),
        "400_is_subset_of_600": bool(small <= large),
        "by_season": per_season,
        "exclusions_grouped_by_reason": {
            "not_selected_by_the_subsample_rule": {
                "count": len(available_total - small),
                "reason": (
                    "the season's held-out game ids are sorted and sampled at "
                    "evenly spaced indices; a game outside those indices is "
                    "not simulated. The rule reads only the season, the id "
                    "ordering and the requested count"
                ),
                "performance_based": False,
                "model_specific": False,
            },
            "failed_the_eligibility_filter": {
                "count": 0,
                "reason": (
                    "a game is skipped when fewer than two players clear the "
                    "8.0 expected-minute floor; games_skipped is 0 in all "
                    "three runs, so this excluded nothing"
                ),
                "performance_based": False,
                "model_specific": False,
            },
        },
        "games_available_in_the_holdout_seasons": len(available_total),
        "subsample_density_provenance": subsample_provenance(),
        "classification": {
            "label": "A",
            "statement": (
                "A -- pre-defined paired joint-event subset. The 400-game set "
                "is the validator's own deterministic subsample at the "
                "driver's default density, used identically by candidate and "
                "control, and fixed before the candidate existed."
            ),
            "not_B_accidental_reduction": (
                "the density is the driver's declared default, not a silent "
                "drop; both runs record games_simulated 400 and "
                "games_skipped 0"
            ),
            "not_C_different_event_generation": (
                "both universes use the same families, the same two events "
                "per family, the same probability levels and the same line "
                "rule; only the set of games differs"
            ),
            "caveat": (
                "the 400-game set is NOT a nested subset of the 600-game "
                "set. Evenly spaced indices at n=200 and n=300 over the same "
                "sorted pool coincide rarely, so the two are differently "
                "spaced samples of one pool, not a sample and its subsample. "
                "That is why a count-space statistic measured on the "
                "simulated games moves between the two runs."
            ),
        },
    }


def subsample_provenance() -> dict[str, object]:
    """When the 200-per-season default was fixed, relative to the candidate."""
    probe = subprocess.run(
        [
            "git",
            "log",
            "--format=%H %cI",
            "-S",
            'default=200,',
            "--",
            "research/game_latent_state/04_validate_shadow_v1.py",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in probe.stdout.splitlines() if line.strip()]
    introduced = lines[-1].split() if lines else ["", ""]
    fork = subprocess.run(
        ["git", "rev-parse", "research/nba-game-latent-state-shadow-clean"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", introduced[0] or "HEAD", fork or "HEAD"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "default_introduced_in": introduced[0],
        "default_introduced_at": introduced[1] if len(introduced) > 1 else None,
        "remediation_fork_point": fork,
        "default_predates_the_remediation_fork": ancestor.returncode == 0,
        "statement": (
            "the 200-games-per-season default was committed before the branch "
            "the remediation candidate was built on, so it cannot have been "
            "chosen in response to the candidate's performance"
        ),
    }


def grades_universe(candidate: pd.DataFrame, control: pd.DataFrame) -> dict[str, object]:
    candidate_games = sorted(int(value) for value in candidate["game_id"].unique())
    control_games = sorted(int(value) for value in control["game_id"].unique())
    identical_keys = bool(
        candidate[["game_id", "family", "n_legs", "realized"]].equals(
            control[["game_id", "family", "n_legs", "realized"]]
        )
    )
    return {
        "candidate_events": int(len(candidate)),
        "control_events": int(len(control)),
        "candidate_games": len(candidate_games),
        "control_games": len(control_games),
        "game_sets_identical": candidate_games == control_games,
        "event_keys_identical_row_for_row": identical_keys,
        "realized_identical": bool(candidate["realized"].equals(control["realized"])),
        "baselines_identical": bool(
            np.allclose(
                candidate["p_baseline_production"], control["p_baseline_production"]
            )
            and np.allclose(
                candidate["p_baseline_independence"],
                control["p_baseline_independence"],
            )
        ),
        "events_per_game": int(len(candidate) / len(candidate_games)),
        "families": sorted(str(value) for value in candidate["family"].unique()),
        "family_count": int(candidate["family"].nunique()),
        "events_by_legs": {
            str(int(legs)): int((candidate["n_legs"] == legs).sum())
            for legs in sorted(candidate["n_legs"].unique())
        },
    }


# ======================================================================
# 3. count RMSE definitions
# ======================================================================


def count_rmse_definitions(
    repair: dict, control: dict, candidate: dict
) -> dict[str, object]:
    runs = {
        "published_repair_600": repair,
        "paired_control_400": control,
        "paired_candidate_400": candidate,
    }
    recomputation: dict[str, object] = {}
    for name, report in runs.items():
        residual = report["residual_dependence"]
        entry: dict[str, object] = {
            "buckets": sorted(residual["observed_buckets"]),
            "bucket_count": len(residual["observed_buckets"]),
            "games_simulated": report["games_simulated"],
            "simulations_per_game": report["simulations_per_game"],
            "validation_seasons": report["validation_seasons"],
            "by_model": {},
        }
        for model, block in residual["by_model"].items():
            errors = np.array(
                [block["bucket_errors"][key] for key in sorted(block["bucket_errors"])]
            )
            recomputed = float(np.sqrt(np.mean(np.square(errors))))
            stored = float(residual["count_space_cross_player_rmse"][model])
            entry["by_model"][model] = {  # type: ignore[index]
                "recomputed_unweighted_rmse": recomputed,
                "stored": stored,
                "matches": bool(np.isclose(recomputed, stored, rtol=0.0, atol=1e-15)),
            }
        recomputation[name] = entry

    observed = {
        name: report["residual_dependence"]["observed_buckets"]
        for name, report in runs.items()
    }
    latent_observed = {
        name: report["latent_dependence"]["observed_buckets"]
        for name, report in runs.items()
    }
    return {
        "definition": {
            "statistic": (
                "unweighted root mean square of the 12 cross-player bucket "
                "errors in count space"
            ),
            "bucket_count": 12,
            "weighting": "none; every bucket enters with weight 1",
            "se_weighting": False,
            "role_weighting": False,
            "same_and_cross_team_both_included": True,
            "count_transform": (
                "standardized count residuals e_<stat>, re-prefixed to zs_ and "
                "pooled by the same game-clustered pair_moments accumulator "
                "the latent space uses"
            ),
            "observed_estimator": (
                "observed_count_pair_moments over the simulated held-out games"
            ),
            "implied_estimator": (
                "simulated_pair_moments over the model's own draws for the "
                "same games"
            ),
            "random_pit_enters": False,
            "identical_in_both_runs": True,
        },
        "by_run": recomputation,
        "observed_count_buckets_identical": {
            "repair_600_vs_control_400": observed["published_repair_600"]
            == observed["paired_control_400"],
            "control_400_vs_candidate_400": observed["paired_control_400"]
            == observed["paired_candidate_400"],
        },
        "observed_count_bucket_difference_600_minus_400": {
            key: observed["published_repair_600"][key]
            - observed["paired_control_400"][key]
            for key in sorted(observed["published_repair_600"])
        },
        "latent_observed_buckets_identical": {
            "repair_600_vs_control_400": latent_observed["published_repair_600"]
            == latent_observed["paired_control_400"],
        },
        "latent_control_rmse_identical_across_universes": (
            repair["residual_dependence"]["cross_player_rmse"]["candidate"]
            == control["residual_dependence"]["cross_player_rmse"]["candidate"]
        ),
        # The latent column is measured on every held-out game rather than on
        # the simulated subset, so it is the one reading of the control model
        # that the subsample cannot move. Bitwise agreement across the two
        # code revisions is therefore the equivalence proof for the additive
        # role machinery that landed in covariance.py between them.
        "control_latent_path_unchanged_by_the_refactor": {
            "statement": (
                "the latent dependence summary reads all held-out games, not "
                "the simulated subset, so it isolates the control model from "
                "the universe change"
            ),
            "observed_buckets_identical": latent_observed["published_repair_600"]
            == latent_observed["paired_control_400"],
            "implied_buckets_identical": (
                repair["latent_dependence"]["by_model"]["candidate"][
                    "implied_buckets"
                ]
                == control["latent_dependence"]["by_model"]["candidate"][
                    "implied_buckets"
                ]
            ),
            "rmse_repair_600": repair["residual_dependence"]["cross_player_rmse"][
                "candidate"
            ],
            "rmse_control_400": control["residual_dependence"]["cross_player_rmse"][
                "candidate"
            ],
        },
        "why_the_values_differ": (
            "the definition is identical in both runs and reproduces both "
            "stored numbers exactly, and the control model's latent reading "
            "is bitwise unchanged. The count-space statistic is measured on "
            "the simulated games only, and the two runs simulate differently "
            "spaced 400- and 600-game samples of the same held-out pool, so "
            "the observed count-space buckets themselves differ. The move "
            "from 0.007861206 to 0.006404740 is entirely the evaluation "
            "universe, not a changed metric and not a changed model."
        ),
        "are_values_directly_comparable": "NO",
        "is_this_a_bug": (
            "No. A bug would require the same observations to produce two "
            "answers. Here the definition is provably one definition and the "
            "observations are provably two different samples. The reporting "
            "defect -- quoting 0.007861206 and 0.006404740 side by side as "
            "though they were a control-versus-candidate comparison -- is "
            "real, and the paired control number 0.006404740 is the only one "
            "comparable with the candidate's 0.006078041."
        ),
    }


def driver_equivalence() -> dict[str, object]:
    """Blob identity of the files behind the two count-RMSE numbers."""
    repair_sha = json.loads(
        (REPAIR_DIR / "manifest.validation.json").read_text(encoding="utf-8")
    )["code_sha"]
    paired_sha = json.loads(
        (ARTIFACT_DIR / "manifest.validation.json").read_text(encoding="utf-8")
    )["code_sha"]
    files = (
        "research/game_latent_state/04_validate_shadow_v1.py",
        "src/nba_prop_quant/research/game_latent_state/factors.py",
        "src/nba_prop_quant/research/game_latent_state/validation.py",
        "src/nba_prop_quant/research/game_latent_state/covariance.py",
    )
    out: dict[str, object] = {
        "published_repair_code_sha": repair_sha,
        "paired_runs_code_sha": paired_sha,
        "by_file": {},
    }
    for path in files:
        try:
            before = subprocess.run(
                ["git", "rev-parse", f"{repair_sha}:{path}"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            before = "ABSENT"
        after = subprocess.run(
            ["git", "rev-parse", f"{paired_sha}:{path}"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--numstat", repair_sha, paired_sha, "--", path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        out["by_file"][path] = {  # type: ignore[index]
            "blob_at_repair_run": before,
            "blob_at_paired_runs": after,
            "identical": before == after,
            "numstat": diff,
        }
    return out


# ======================================================================
# 4. proper scores
# ======================================================================


def log_loss(probability: np.ndarray, realized: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    return -(realized * np.log(clipped) + (1.0 - realized) * np.log(1.0 - clipped))


def clustered_ci(
    per_event: np.ndarray, games: np.ndarray, draws: int, seed: int
) -> dict[str, float]:
    unique = np.unique(games)
    index = {game: position for position, game in enumerate(unique)}
    sums = np.zeros(len(unique))
    counts = np.zeros(len(unique))
    for value, game in zip(per_event, games, strict=True):
        position = index[game]
        sums[position] += float(value)
        counts[position] += 1.0
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        picks = rng.integers(0, len(unique), size=len(unique))
        means[draw] = sums[picks].sum() / counts[picks].sum()
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "delta": float(per_event.mean()),
        "standard_error": float(np.std(means, ddof=1)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "significant": bool(high < 0.0 or low > 0.0),
    }


def proper_scores(
    candidate: pd.DataFrame, control: pd.DataFrame
) -> dict[str, object]:
    realized = candidate["realized"].to_numpy(float)
    games = candidate["game_id"].to_numpy()
    probability = {
        "candidate": candidate["p_candidate"].to_numpy(float),
        "control": control["p_candidate"].to_numpy(float),
        "production": candidate["p_baseline_production"].to_numpy(float),
        "independence": candidate["p_baseline_independence"].to_numpy(float),
    }
    losses = {name: log_loss(value, realized) for name, value in probability.items()}
    briers = {
        name: (value - realized) ** 2 for name, value in probability.items()
    }

    by_legs: dict[str, object] = {}
    for legs in sorted(candidate["n_legs"].unique()):
        mask = (candidate["n_legs"] == legs).to_numpy()
        entry: dict[str, object] = {
            "events": int(mask.sum()),
            "games": int(len(np.unique(games[mask]))),
            "base_rate": float(realized[mask].mean()),
            "probability_range": [
                float(probability["candidate"][mask].min()),
                float(probability["candidate"][mask].max()),
            ],
        }
        for name in probability:
            entry[f"log_loss_{name}"] = float(losses[name][mask].mean())
            entry[f"brier_{name}"] = float(briers[name][mask].mean())
        for label, (left, right) in (
            ("log_loss_candidate_minus_control", ("candidate", "control")),
            ("log_loss_candidate_minus_production", ("candidate", "production")),
            ("brier_candidate_minus_control", ("candidate", "control")),
            ("brier_candidate_minus_production", ("candidate", "production")),
        ):
            source = losses if label.startswith("log_loss") else briers
            entry[label] = clustered_ci(
                source[left][mask] - source[right][mask],
                games[mask],
                BOOTSTRAP_DRAWS,
                SEED,
            )
        by_legs[str(int(legs))] = entry
    return {
        "universe": {
            "seasons": list(HOLDOUT_SEASONS),
            "games": int(len(np.unique(games))),
            "events": int(len(candidate)),
            "events_per_game": int(len(candidate) / len(np.unique(games))),
            "event_generation_rule": (
                "8 declared families x 2 events per family per game, players "
                "stats sides and probability levels drawn from a per-game "
                "locked RNG"
            ),
            "line_rule": (
                "half-integer line placed at the predictive marginal's "
                "quantile of the drawn probability level, plus 0.5; the "
                "realized box score enters only when grading"
            ),
            "pushes_possible": False,
            "pushes_reason": (
                "counts are integers and every line is a half-integer, so "
                "realized == line cannot occur"
            ),
            "probability_source": "Monte Carlo, 20,000 draws per game",
            "blk_excluded_from_legs": True,
            "leg_eligibility_min_expected_minutes": 12.0,
        },
        "clip": LOG_LOSS_CLIP,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "by_legs": by_legs,
    }


def temperature_universe(temperature: dict) -> dict[str, object]:
    return {
        "seasons": temperature["fold_target_seasons"],
        "holdout_seasons_excluded": temperature["holdout_seasons_excluded"],
        "holdout_used_for_selection": temperature["holdout_used_for_selection"],
        "games": temperature["selection"]["games"],
        "events": temperature["scores"]["events"],
        "events_by_legs": temperature["events_by_legs"],
        "events_by_fold": temperature["events_by_fold"],
        "probability_source": "exact Gaussian orthant integration, not Monte Carlo",
        "pricing_note": temperature["pricing"],
        "selected_temperature": temperature["selected_temperature"],
        "score": "game-clustered log loss",
    }


# ======================================================================
# 6. half life
# ======================================================================


def half_life_equivalence(inner: dict, diagnostics: dict) -> dict[str, object]:
    """A deterministic equivalence check, not a refit.

    ``temporal_fitter`` dispatches on the treatment name. The A2 branch does
    not forward ``half_life``, so calling the fitter the way A2 calls it with
    any half life must return the identical object. This runs the estimator on
    the stored series rather than on data, so no model is refitted.
    """
    from upstream_spec import TEMPORAL_STUDENT_T, temporal_fitter

    recorded = inner["item_1_temporal"]["primary_bucket_series"]
    series = SeasonSeries(
        name=inner["item_1_temporal"]["primary_bucket"],
        seasons=tuple(int(value) for value in recorded["seasons"]),
        estimates=np.asarray(recorded["estimates"], dtype=float),
        standard_errors=np.asarray(recorded["standard_errors"], dtype=float),
    )
    nu = float(inner["item_1_temporal"]["nu_selected"])

    probes = [0.5, 1.0, 2.0, 4.0, 8.0, 1000.0]
    results = []
    for value in probes:
        fit = temporal_fitter(TEMPORAL_STUDENT_T, nu=nu, half_life=value)(series)
        results.append(
            {
                "half_life_passed": value,
                "model": fit.model,
                "half_life_recorded": fit.half_life,
                "posterior_mean_hex": float(fit.posterior_mean).hex(),
                "posterior_mean": float(fit.posterior_mean),
                "posterior_sd_hex": float(fit.posterior_sd).hex(),
                "tau_hex": float(fit.tau).hex(),
                "weights_hex": [float(w).hex() for w in fit.weights],
            }
        )
    first = results[0]
    bitwise = all(
        entry["posterior_mean_hex"] == first["posterior_mean_hex"]
        and entry["posterior_sd_hex"] == first["posterior_sd_hex"]
        and entry["tau_hex"] == first["tau_hex"]
        and entry["weights_hex"] == first["weights_hex"]
        for entry in results
    )

    a3 = temporal_fitter(
        "A3_recency_weighted_robust", nu=nu, half_life=4.0
    )(series)
    per_bucket = {
        bucket: entry.get("half_life")
        for bucket, entry in diagnostics["temporal"]["by_bucket"].items()
    }
    return {
        "selected_treatment": diagnostics["temporal"]["treatment"],
        "half_life_carried_in_frozen_choices": inner["item_1_temporal"][
            "half_life_selected"
        ],
        "half_life_grid": inner["item_1_temporal"]["half_life_grid"],
        "half_life_recorded_in_every_bucket_fit": per_bucket,
        "half_life_null_in_every_bucket_fit": all(
            value is None for value in per_bucket.values()
        ),
        "probes": results,
        "bitwise_identical_across_every_probe": bitwise,
        "a3_posterior_mean_for_contrast": float(a3.posterior_mean),
        "a3_differs_from_a2": bool(
            not np.isclose(a3.posterior_mean, first["posterior_mean"])
        ),
        "dispatch_reason": (
            "temporal_fitter returns "
            "lambda series: fit_student_t_random_effects(series, nu=nu) for "
            "A2 and forwards half_life only on the A3 branch, where "
            "half_life=None is documented as the A2 case"
        ),
        "half_life_status": "INACTIVE_NOT_APPLICABLE",
    }


# ======================================================================
# 5. provenance
# ======================================================================


def provenance(inner: dict, temperature: dict, frozen: dict, gates: dict) -> dict:
    manifest = load(ARTIFACT_DIR / "manifest.validation.json")
    factor_manifest = load(ARTIFACT_DIR / "manifest.json")
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/production/wizardofodds-integration"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    fork = subprocess.run(
        [
            "git",
            "merge-base",
            "HEAD",
            "origin/research/nba-game-latent-state-shadow-clean",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "01_clean_control_base_sha": merge_base,
        "02_remediation_branch_fork_sha": fork,
        "03_inner_selection_code_sha": inner["code_sha"],
        "04_inner_selection_artifact_sha256": sha256(
            ARTIFACT_DIR / "inner_selection.json"
        ),
        "05_dependence_temperature_code_sha": temperature["code_sha"],
        "06_dependence_temperature_artifact_sha256": sha256(
            ARTIFACT_DIR / "dependence_temperature.json"
        ),
        "07_frozen_spec_code_sha": frozen["code_sha"],
        "08_frozen_spec_file_sha256": sha256(ARTIFACT_DIR / "frozen_spec.json"),
        "09_candidate_factor_spec_hash": frozen["factor_spec_hash"],
        "09b_factor_spec_file_sha256": sha256(ARTIFACT_DIR / "factor_spec.json"),
        "09c_factor_spec_sha256_in_manifest": factor_manifest["outputs"].get(
            "factor_spec.json"
        ),
        "10_confirmatory_validation_code_sha": manifest["code_sha"],
        "11_validation_artifact_sha256": sha256(
            ARTIFACT_DIR / "validation_report.json"
        ),
        "11b_validation_sha256_in_manifest": manifest["outputs"].get(
            "validation_report.json"
        ),
        "12_gate_evaluator_code_sha": gates["code_sha"],
        "13_gate_report_sha256": sha256(ARTIFACT_DIR / "gate_report.json"),
        # Both rows are HEAD at the moment the audit ran, which is one commit
        # behind the commit that carries this file. A row cannot name the
        # commit that contains it, so the number is stated for what it is.
        "14_report_generator_sha": git_sha(PROJECT_ROOT),
        "15_final_branch_head_sha": git_sha(PROJECT_ROOT),
        "15_note": (
            "HEAD when the audit ran. The commit carrying final_audit.json is "
            "its child, since a recorded hash cannot name the commit that "
            "contains it."
        ),
        "control_factor_spec_hash": frozen["control"]["factor_spec_hash"],
        "control_factor_spec_file_sha256": sha256(REPAIR_DIR / "factor_spec.json"),
    }


def post_selection_refactors(inner: dict) -> list[dict[str, object]]:
    """Every commit that touched a selection input after its artifact was written."""
    recorded = inner["code_sha"]
    head = git_sha(PROJECT_ROOT)
    changed = subprocess.run(
        ["git", "diff", "--name-only", recorded, head],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [
        {
            "artifact": "inner_selection.json",
            "recorded_code_sha": recorded,
            "head": head,
            "files_changed_since": sorted(changed),
            "behaviour_affecting": [
                path
                for path in sorted(changed)
                if path.endswith(".py")
                and not path.startswith("tests/")
                and "05_evaluate_gates" not in path
                and "06_write_report" not in path
                and "07_audit" not in path
                and "08_final_audit" not in path
            ],
            "equivalence_evidence": (
                "research/final_upstream_remediation/01_inner_selection.py lost "
                "its local read_buckets/season_series/CROSS_PLAYER_BUCKETS to "
                "upstream_spec.py so a test could import them. The shared "
                "season_series reproduces the recorded primary-bucket series "
                "to 0.0 absolute difference in estimates and standard errors; "
                "the regression test is "
                "test_the_shared_season_series_reproduces_the_recorded_one in "
                "tests/test_game_latent_state_shadow_final_audit.py."
            ),
            "per_file": [
                {
                    "path": "research/final_upstream_remediation/01_inner_selection.py",
                    "change": (
                        "the local read_buckets, season_series and "
                        "CROSS_PLAYER_BUCKETS definitions were deleted and the "
                        "names imported from upstream_spec instead"
                    ),
                    "behaviour_unchanged_because": (
                        "the moved bodies are the same expressions over the "
                        "same inputs; the estimator was not edited, only "
                        "relocated"
                    ),
                    "proof": (
                        "test_the_shared_season_series_reproduces_the_recorded_one "
                        "recomputes the recorded teammate_ast_ast series from "
                        "the shared helper at the artifact's own bootstrap and "
                        "seed and asserts 0.0 absolute difference in both the "
                        "estimates and the standard errors"
                    ),
                },
                {
                    "path": "research/final_upstream_remediation/upstream_spec.py",
                    "change": (
                        "gained read_buckets, season_series and "
                        "CROSS_PLAYER_BUCKETS as the single definition the "
                        "drivers and the tests share"
                    ),
                    "behaviour_unchanged_because": (
                        "the additions are the relocated bodies; no existing "
                        "function in the module was altered"
                    ),
                    "proof": (
                        "the same regression test, plus "
                        "test_the_uniform_weighted_student_t_fit_is_what_the_"
                        "artifact_recorded, which reproduces the recorded "
                        "posterior mean for the primary bucket exactly"
                    ),
                },
            ],
        }
    ]


def findings() -> list[dict[str, object]]:
    """Every discrepancy this audit found, with its disposition.

    Listed explicitly rather than left in prose so the verdict can be read
    off the artifact.
    """
    return [
        {
            "id": "F1",
            "severity": "NON_BLOCKING",
            "what": (
                "the README recorded item 1's selection as "
                "A0_pooled_empirical_bayes while every artifact records "
                "A2_robust_student_t, and the accompanying sentence counted "
                "four of six items landing on control behaviour instead of "
                "three"
            ),
            "where": "research/final_upstream_remediation/README.md",
            "is_a_model_change": False,
            "disposition": (
                "the prose was corrected to match the artifacts. No "
                "parameter, artifact or hash moved; the candidate spec hash "
                "is unchanged"
            ),
        },
        {
            "id": "F2",
            "severity": "NON_BLOCKING",
            "what": (
                "the count-space control RMSE was quoted as 0.007861206 in "
                "the repair round and 0.006404740 in the paired round as "
                "though the two were comparable. They are the same statistic "
                "on two differently spaced samples of the held-out pool"
            ),
            "where": "the repair and remediation validation reports",
            "is_a_model_change": False,
            "disposition": (
                "a reporting defect, not a bug. Only 0.006404740 is "
                "comparable with the candidate's 0.006078041, and gate 4 "
                "already read the paired pair"
            ),
        },
        {
            "id": "F3",
            "severity": "BLOCKING_FOR_THE_ORIGINAL_BRIEF",
            "what": (
                "the brief's >=20% count-space improvement on "
                "passer_ast_teammate_pts was never encoded as a gate and is "
                "not met: the absolute error falls 5.1997%"
            ),
            "where": "research/final_upstream_remediation/05_evaluate_gates.py",
            "is_a_model_change": False,
            "disposition": (
                "reported as a failure. Not reinterpreted and not replaced "
                "with the no-regression gate that was written"
            ),
        },
        {
            "id": "F4",
            "severity": "NON_BLOCKING",
            "what": (
                "the brief asked for an uncertainty calibration improvement "
                "over the raw predictive interval. No gate was written and "
                "no improvement was found; U0_raw was kept because the three "
                "alternatives scored worse on the same folds"
            ),
            "where": "item 6 of the inner selection",
            "is_a_model_change": False,
            "disposition": (
                "reported as NO CALIBRATION IMPROVEMENT -- RAW RETAINED. The "
                "raw intervals are too narrow on the scored fold and the "
                "sample is twelve correlated bucket-seasonal observations, "
                "which is the CAUTION verdict"
            ),
        },
        {
            "id": "F5",
            "severity": "NON_BLOCKING",
            "what": (
                "the brief asked that no supported role-pair cell regress. "
                "No gate reads the six held-out cells, and starter+bench "
                "deteriorates by 4.777e-04 in RMSE and 0.134 in rms z"
            ),
            "where": "research/final_upstream_remediation/05_evaluate_gates.py",
            "is_a_model_change": False,
            "disposition": (
                "reported as a failure of the original requirement. The "
                "pooled supported-cell RMSE still improves and four of six "
                "cells improve, but the per-cell no-regression requirement "
                "as stated is not met"
            ),
        },
        {
            "id": "F6",
            "severity": "INFORMATIONAL",
            "what": (
                "half_life is carried in the frozen upstream choices at 4.0 "
                "but has no numerical effect, because recency weighting "
                "lives on the A3 branch and item 1 selected A2"
            ),
            "where": "upstream_spec.temporal_fitter",
            "is_a_model_change": False,
            "disposition": (
                "marked INACTIVE_NOT_APPLICABLE and proven bitwise inert "
                "across six half-life probes"
            ),
        },
    ]


# ======================================================================
# 7. role scale
# ======================================================================


def per_game_role_cells(
    frame: pd.DataFrame, stats: tuple[str, ...]
) -> dict[tuple[str, str], dict[str, object]]:
    """Per-game cell contributions, so a clustered SE and an ESS are available.

    Identical accumulation to ``remediation.role_pair_cell_moments``: the
    team-sum identity inside each ordered role pair with the diagonal
    self-products removed. The per-game terms are kept rather than only their
    total because the audit needs each cell's clustered standard error and its
    effective number of games, and the library version returns the pooled
    matrix alone.
    """
    columns = [f"zs_{stat}" for stat in stats]
    usable = frame.dropna(subset=["game_id", "team_id", ROLE_COLUMN, *columns])
    roles = sorted(str(value) for value in usable[ROLE_COLUMN].unique())
    width = len(stats)

    blocks: dict[tuple[str, str], list[np.ndarray]] = {
        (a, b): [] for a in roles for b in roles
    }
    counts: dict[tuple[str, str], list[float]] = {key: [] for key in blocks}

    for _, game in usable.groupby("game_id", sort=True):
        game_block: dict[tuple[str, str], np.ndarray] = {}
        game_count: dict[tuple[str, str], float] = {}
        for _, team in game.groupby("team_id", sort=True):
            totals: dict[str, np.ndarray] = {}
            own: dict[str, np.ndarray] = {}
            size: dict[str, int] = {}
            for role, members in team.groupby(ROLE_COLUMN, sort=True):
                values = members[columns].to_numpy(dtype=float)
                label = str(role)
                totals[label] = values.sum(axis=0)
                own[label] = values.T @ values
                size[label] = values.shape[0]
            for first in totals:
                for second in totals:
                    block = np.outer(totals[first], totals[second])
                    count = size[first] * size[second]
                    if first == second:
                        block = block - own[first]
                        count -= size[first]
                    if count <= 0:
                        continue
                    key = (first, second)
                    game_block[key] = game_block.get(key, np.zeros((width, width)))
                    game_block[key] = game_block[key] + block
                    game_count[key] = game_count.get(key, 0.0) + count
        for key, block in game_block.items():
            blocks[key].append(block)
            counts[key].append(game_count[key])

    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, terms in blocks.items():
        if not terms:
            continue
        stacked = np.stack(terms)
        count = np.asarray(counts[key], dtype=float)
        total = float(count.sum())
        if total <= 0:
            continue
        matrix = stacked.sum(axis=0) / total
        out[key] = {
            "correlation": 0.5 * (matrix + matrix.T),
            "per_game_blocks": stacked,
            "per_game_counts": count,
            "pairs": total,
            "games": int(len(count)),
            # Kish effective number of clusters for a pair-count-weighted mean
            # over games. The cell's estimate is one such mean, so this is the
            # number of equally sized games that would carry the same weight
            # concentration -- not a claim about independence across stats.
            "effective_games": float(total**2 / float(np.sum(np.square(count)))),
            "supported": bool(
                len(count) >= MIN_CELL_GAMES and total >= MIN_CELL_PAIRS
            ),
        }
    return out


def cell_standard_errors(
    cell: dict[str, object], draws: int, seed: int
) -> np.ndarray:
    """Game-clustered bootstrap SE of every entry of one cell's matrix."""
    blocks = np.asarray(cell["per_game_blocks"])
    counts = np.asarray(cell["per_game_counts"], dtype=float)
    rng = np.random.default_rng(seed)
    games = len(counts)
    drawn = np.empty((draws, blocks.shape[1], blocks.shape[2]))
    for draw in range(draws):
        picks = rng.integers(0, games, size=games)
        matrix = blocks[picks].sum(axis=0) / counts[picks].sum()
        drawn[draw] = 0.5 * (matrix + matrix.T)
    return drawn.std(axis=0, ddof=1)


def role_scale_audit(
    residuals: pd.DataFrame,
    candidate_spec: dict,
    control_spec: dict,
    recorded_fit: dict,
) -> dict[str, object]:
    """Report the six role-pair cells and the identification constraint.

    Nothing is redesigned or refitted. The role scales are read from the
    frozen spec; the cells are measured on the held-out seasons with the
    spec's own training moments, which is the same standardization the
    validator applies.
    """
    from nba_prop_quant.research.game_latent_state.covariance import (
        SharedFactorLoadings,
    )
    from nba_prop_quant.research.game_latent_state.factors import (
        standardize_residuals,
    )
    from nba_prop_quant.research.game_latent_state.remediation import (
        ROLE_PAIR_CELLS,
    )

    moments = {
        stat: (float(value["mean"]), float(value["sd"]))
        for stat, value in candidate_spec["standardization_moments"].items()
    }
    if (
        control_spec["standardization_moments"]
        != candidate_spec["standardization_moments"]
    ):
        raise SystemExit(
            "the two specs standardize differently, so their cells are not "
            "measured on the same scale"
        )

    held_out = residuals.loc[residuals["season"].isin(list(HOLDOUT_SEASONS))]
    standardized, _ = standardize_residuals(held_out, STATS, moments=moments)
    observed = per_game_role_cells(standardized, STATS)

    candidate = SharedFactorLoadings.from_payload(candidate_spec["loadings"])
    control = SharedFactorLoadings.from_payload(control_spec["loadings"])

    by_cell: dict[str, object] = {}
    worst_label: str | None = None
    worst_delta = -float("inf")
    worst_z_label: str | None = None
    worst_z_delta = -float("inf")

    for position, (first, second) in enumerate(ROLE_PAIR_CELLS):
        label = f"{first}+{second}"
        cell = observed.get((first, second))
        if cell is None:
            by_cell[label] = {"present": False}
            continue
        truth = np.asarray(cell["correlation"], dtype=float)
        standard_error = cell_standard_errors(cell, CELL_BOOTSTRAP_DRAWS, SEED + position)
        safe = np.where(standard_error > 0.0, standard_error, np.nan)
        diagonal = np.arange(len(STATS))

        entry: dict[str, object] = {
            "present": True,
            "supported": bool(cell["supported"]),
            "games": cell["games"],
            "pairs": cell["pairs"],
            "effective_games": cell["effective_games"],
            # The scalar the role layer actually moves: the mean same-stat
            # same-team cross-player correlation inside the role pair.
            "observed_scalar": float(np.mean(truth[diagonal, diagonal])),
        }
        errors: dict[str, float] = {}
        z_errors: dict[str, float] = {}
        scalar_z: dict[str, float] = {}
        for name, loadings in (("control", control), ("candidate", candidate)):
            implied = loadings.same_team_correlation_for_roles(first, second)
            residual = implied - truth
            errors[name] = float(np.sqrt(np.mean(np.square(residual))))
            z_errors[name] = float(np.sqrt(np.nanmean(np.square(residual / safe))))
            scalar = float(np.mean(np.asarray(implied)[diagonal, diagonal]))
            scalar_se = float(np.mean(standard_error[diagonal, diagonal]))
            entry[f"{name}_scalar"] = scalar
            entry[f"{name}_scalar_z"] = (
                float("nan")
                if scalar_se <= 0.0
                else (scalar - float(entry["observed_scalar"])) / scalar_se
            )
            scalar_z[name] = float(entry[f"{name}_scalar_z"])
            entry[f"{name}_rmse"] = errors[name]
            entry[f"{name}_rms_z"] = z_errors[name]
            entry[f"{name}_max_abs_error"] = float(np.max(np.abs(residual)))
        entry["scalar_se"] = float(np.mean(standard_error[diagonal, diagonal]))
        entry["delta_rmse"] = errors["candidate"] - errors["control"]
        entry["delta_rms_z"] = z_errors["candidate"] - z_errors["control"]
        entry["delta_abs_scalar_z"] = abs(scalar_z["candidate"]) - abs(
            scalar_z["control"]
        )
        by_cell[label] = entry

        if cell["supported"]:
            if float(entry["delta_rmse"]) > worst_delta:
                worst_delta = float(entry["delta_rmse"])
                worst_label = label
            if float(entry["delta_rms_z"]) > worst_z_delta:
                worst_z_delta = float(entry["delta_rms_z"])
                worst_z_label = label

    supported = [
        value
        for value in by_cell.values()
        if isinstance(value, dict) and value.get("supported")
    ]

    fit = dict(recorded_fit)
    recorded = dict(candidate_spec["loadings"].get("role_scale", {}))
    return {
        "role_scale_was_redesigned": False,
        "role_scale_was_refitted": False,
        "definition": (
            "per cell, the root mean square over the 6x6 stat block of "
            "(model-implied same-team correlation for the ordered role pair "
            "minus the held-out observed one); rms z divides each entry by "
            "its game-clustered bootstrap standard error; the scalar column "
            "is the mean of the six same-stat diagonal entries"
        ),
        "effective_sample_size_definition": (
            "Kish effective number of games for the pair-count-weighted mean "
            "over games, (sum c_g)^2 / sum c_g^2"
        ),
        "support_threshold": {
            "min_cell_games": MIN_CELL_GAMES,
            "min_cell_pairs": MIN_CELL_PAIRS,
            "rule": "a cell is supported when games >= 200 and pairs >= 2000",
        },
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "cell_bootstrap_draws": CELL_BOOTSTRAP_DRAWS,
        "cells_total": len(ROLE_PAIR_CELLS),
        "cells_present": len(
            [v for v in by_cell.values() if isinstance(v, dict) and v.get("present")]
        ),
        "cells_supported": len(supported),
        "cells_shrunk_to_pooled": len(ROLE_PAIR_CELLS) - len(supported),
        "cells_shrunk_to_pooled_note": (
            "a cell falls back to the pooled same-team block when it misses "
            "either support threshold; every one of the six clears both, so "
            "no cell is pooled away"
        ),
        "by_cell": by_cell,
        "worst_supported_cell_deterioration_rmse": {
            "cell": worst_label,
            "delta_rmse": None if worst_label is None else worst_delta,
        },
        "worst_supported_cell_deterioration_rms_z": {
            "cell": worst_z_label,
            "delta_rms_z": None if worst_z_label is None else worst_z_delta,
        },
        "pooled_supported_rmse": {
            "control": float(
                np.sqrt(np.mean([v["control_rmse"] ** 2 for v in supported]))
            ),
            "candidate": float(
                np.sqrt(np.mean([v["candidate_rmse"] ** 2 for v in supported]))
            ),
        },
        "identification": role_scale_identification(recorded, fit),
    }


def role_scale_identification(
    recorded: dict[str, float], fit: dict[str, object]
) -> dict[str, object]:
    """Independently recompute the player-share-weighted mean scale."""
    shares = {
        str(key): float(value)
        for key, value in dict(fit.get("player_shares", {})).items()
    }
    roles = sorted(recorded)
    if shares:
        total = sum(shares.get(role, 0.0) for role in roles)
        weights = np.array([shares.get(role, 0.0) / total for role in roles])
    else:
        weights = np.full(len(roles), 1.0 / max(len(roles), 1))
    scales = np.array([float(recorded[role]) for role in roles])
    weighted_mean = float(np.sum(weights * scales))

    raw = {
        str(key): float(value)
        for key, value in dict(fit.get("raw_scales", {})).items()
    }
    log_se = {
        str(key): float(value)
        for key, value in dict(fit.get("log_standard_errors", {})).items()
    }
    tau_log = float(fit.get("tau_log", 0.0) or 0.0)
    shrinkage = {
        role: (
            float("nan")
            if tau_log <= 0.0
            else tau_log**2 / (tau_log**2 + log_se.get(role, 0.0) ** 2)
        )
        for role in roles
    }
    return {
        "roles": roles,
        "scales": {role: float(recorded[role]) for role in roles},
        "raw_scales": raw,
        "log_standard_errors": log_se,
        "tau_log": tau_log,
        "player_share_weights": {
            role: float(weight) for role, weight in zip(roles, weights)
        },
        "shrinkage_factor_towards_pooled": shrinkage,
        "roles_fully_shrunk_to_pooled": [
            role for role, value in shrinkage.items() if value == 0.0
        ],
        "recomputed_weighted_mean_scale": weighted_mean,
        "recomputed_weighted_mean_scale_minus_one": weighted_mean - 1.0,
        "recorded_weighted_mean_scale_minus_one": float(
            fit.get("weighted_mean_scale_minus_one", float("nan"))
        ),
        "tolerance": 1e-12,
        "constraint_holds_exactly_within_tolerance": bool(
            abs(weighted_mean - 1.0) <= 1e-12
        ),
    }


# ======================================================================
# 8. original gate versus implemented gate
# ======================================================================


def gate_mapping(
    gates: dict, reconciliation: dict, uncertainty: dict, scores: dict
) -> list[dict[str, object]]:
    """Map every acceptance requirement in the brief onto what was built.

    The ``original`` column is the requirement as the brief stated it. The
    ``implemented`` column is the gate that actually ran. Where they differ the
    difference is stated exactly, and the original requirement is evaluated on
    its own terms whether or not the implemented gate passed.
    """
    implemented = {entry["gate"]: entry for entry in gates["gates"]}
    count = reconciliation["count_target_gate"]
    unc = reconciliation["uncertainty"]
    cells = reconciliation["role_cells"]

    def row(
        original: str,
        gate: object,
        exact_difference: str,
        original_passes: object,
        evidence: object,
    ) -> dict[str, object]:
        entry = implemented.get(gate) if gate is not None else None
        return {
            "original_gate": original,
            "final_implemented_gate": (
                "NOT IMPLEMENTED" if entry is None else f"gate {gate}: {entry['name']}"
            ),
            "identical": exact_difference == "none",
            "exact_difference": exact_difference,
            "implemented_gate_passed": None if entry is None else bool(entry["passed"]),
            "would_the_original_gate_pass": original_passes,
            "evidence": evidence,
        }

    rows = [
        row(
            "passer_ast_teammate_pts improves by at least 20% in count space "
            "against the control",
            None,
            "the brief asked for a >=20% count-space absolute-error reduction "
            "on one named bucket; no gate encoding that threshold was ever "
            "written. Gate 4 is a different statistic: it asks that the "
            "12-bucket count-space RMSE not worsen by more than 5%, which is "
            "a no-regression test and not a 20% improvement test",
            False,
            {
                "observed": count["observed"],
                "control_predicted": count["control_predicted"],
                "candidate_predicted": count["candidate_predicted"],
                "control_abs_error": count["control_abs_error"],
                "candidate_abs_error": count["candidate_abs_error"],
                "percentage_error_reduction": count["percentage_error_reduction"],
                "required_percentage_error_reduction": 100.0
                * count["original_requirement"],
                "shortfall_in_percentage_points": 100.0
                * count["original_requirement"]
                - count["percentage_error_reduction"],
            },
        ),
        row(
            "predictive uncertainty calibration improves on the raw "
            "predictive interval",
            None,
            "no gate was written for item 6 at all. The inner selection "
            "scored four uncertainty models and kept the raw one because it "
            "had the lowest coverage loss, so the brief's improvement was "
            "sought and not found rather than skipped",
            False,
            {
                "selected": unc["selected"],
                "raw_coverage_loss": unc["raw_loss"],
                "coverage_loss_by_candidate": unc["coverage_loss"],
                "candidates_better_than_raw": unc["candidates_better_than_raw"],
                "statement": unc["statement"],
                "scored_observations": uncertainty["scored_observations"],
            },
        ),
        row(
            "final 2-leg, 3-leg and 4-leg log loss is reported",
            None,
            "gates 9 and 10 priced the multi-leg comparison in Brier score, "
            "not log loss. Log loss is recomputed here from the stored "
            "probabilities, so the quantity exists; it was simply never a "
            "gate",
            True,
            {
                "by_legs": {
                    key: {
                        "log_loss_control": value["log_loss_control"],
                        "log_loss_candidate": value["log_loss_candidate"],
                        "log_loss_candidate_minus_control": value[
                            "log_loss_candidate_minus_control"
                        ],
                    }
                    for key, value in scores["by_legs"].items()
                },
                "note": (
                    "reported as required; no threshold was attached in the "
                    "brief, so there is nothing to fail"
                ),
            },
        ),
        row(
            "no supported role-pair cell regresses",
            None,
            "gate 12 checks that the role layer is pooled-neutral and "
            "identified, which is an identification test, not a per-cell "
            "no-regression test. No gate reads the six held-out cells",
            False,
            {
                "supported_cells": cells["supported_cells"],
                "worst_supported_cell_deterioration": cells[
                    "worst_supported_cell_deterioration"
                ],
                "note": (
                    "starter+bench deteriorates: rmse "
                    f"{cells['by_cell']['starter+bench']['control_rmse']:.9f} -> "
                    f"{cells['by_cell']['starter+bench']['candidate_rmse']:.9f}, "
                    "rms z "
                    f"{cells['by_cell']['starter+bench']['control_rms_z']:.4f} -> "
                    f"{cells['by_cell']['starter+bench']['candidate_rms_z']:.4f}"
                ),
            },
        ),
        row(
            "no protected opponent (cross-team) bucket worsens",
            5,
            "none",
            True,
            implemented[5]["evidence"],
        ),
        row(
            "the repaired target buckets are not given back",
            1,
            "none",
            True,
            implemented[1]["evidence"],
        ),
        row(
            "no target bucket overshoots the observed value",
            2,
            "none",
            True,
            implemented[2]["evidence"],
        ),
        row(
            "global latent-space bucket RMSE does not worsen materially",
            3,
            "none",
            True,
            implemented[3]["evidence"],
        ),
        row(
            "count-space cross-player RMSE does not worsen materially",
            4,
            "none",
            True,
            implemented[4]["evidence"],
        ),
        row(
            "every same-player block stays pinned at the incumbent's",
            6,
            "none",
            True,
            implemented[6]["evidence"],
        ),
        row(
            "every held-out game covariance is PSD",
            7,
            "none",
            True,
            implemented[7]["evidence"],
        ),
        row(
            "marginal calibration is preserved",
            8,
            "none",
            True,
            {
                "candidate_max_abs_mean_z": implemented[8]["evidence"][
                    "candidate_gate_a"
                ]["evidence"]["max_abs_mean_z"],
                "bonferroni_z_critical": implemented[8]["evidence"][
                    "candidate_gate_a"
                ]["evidence"]["bonferroni_z_critical"],
            },
        ),
        row(
            "multi-leg joint calibration does not worsen beyond tolerance",
            9,
            "none",
            True,
            implemented[9]["evidence"],
        ),
        row(
            "3-leg and 4-leg joint calibration does not worsen beyond tolerance",
            10,
            "none",
            True,
            implemented[10]["evidence"],
        ),
        row(
            "no pairwise or player-indexed parameter is introduced",
            11,
            "none",
            True,
            implemented[11]["evidence"]["parameter_counts"],
        ),
        row(
            "the multiplicative role layer is pooled-neutral and identified",
            12,
            "none",
            True,
            implemented[12]["evidence"],
        ),
        row(
            "every selection used pre-2024 folds only and the global "
            "inflation factor was not revived",
            13,
            "none",
            True,
            {
                "inner_selection_holdout_used": implemented[13]["evidence"][
                    "inner_selection_holdout_used"
                ],
                "temperature_holdout_used": implemented[13]["evidence"][
                    "temperature_holdout_used"
                ],
                "parameters_equal_to_the_forbidden_factor": implemented[13][
                    "evidence"
                ]["parameters_equal_to_it"],
            },
        ),
        row(
            "the production surface is untouched",
            14,
            "none",
            True,
            {
                "paths_outside_the_research_surface": implemented[14]["evidence"][
                    "paths_outside_the_research_surface"
                ]
            },
        ),
    ]
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root)

    inner = load(root / "inner_selection.json")
    temperature = load(root / "dependence_temperature.json")
    frozen = load(root / "frozen_spec.json")
    diagnostics = load(root / "covariance_diagnostics.json")
    gates = load(root / "gate_report.json")
    candidate_report = load(root / "validation_report.json")
    control_report = load(root / "control_validation_report.json")
    repair_report = load(REPAIR_DIR / "validation_report.json")
    candidate_spec_payload = load(root / "factor_spec.json")

    print("FINAL AUDIT: uncertainty, universes, count RMSE, provenance")
    print("AUDIT ONLY -- no fit, no selection, no simulation.")

    rule("1. UNCERTAINTY")
    uncertainty = uncertainty_audit(inner)
    print(f"  unit            {uncertainty['unit_of_observation']}")
    print(f"  forward records {uncertainty['forward_records_total']}")
    print(f"  folds present   {uncertainty['folds_present']}")
    print(f"  folds scored    {uncertainty['folds_scored']}")
    print(f"  dropped folds   {uncertainty['folds_dropped_as_not_cross_fitted']}")
    print(f"  observations    {uncertainty['scored_observations']}")
    print(f"  buckets         {uncertainty['scored_bucket_count']}")
    print(f"  reproduces stored coverage table: {uncertainty['all_reproduced']}")
    print(f"  {'level':>6s} {'nominal':>8s} {'observed':>9s} {'k/n':>7s} {'se':>7s} {'z':>7s} inside95")
    for key, entry in uncertainty["binomial_reading"].items():  # type: ignore[union-attr]
        print(
            f"  {key.split('_')[1]:>6s} {entry['nominal']:8.2f} "
            f"{entry['observed']:9.4f} "
            f"{int(entry['successes'])}/{int(entry['trials']):<4d} "
            f"{entry['binomial_se_if_independent']:7.4f} "
            f"{entry['z_against_nominal']:+7.2f} "
            f"{entry['nominal_inside_95_interval']}"
        )
    dist = uncertainty["z_distribution"]
    print(
        f"  median |z| {dist['median_abs_z']:.4f}   p90 |z| {dist['p90_abs_z']:.4f}   "
        f"max |z| {dist['max_abs_z']:.4f}"
    )
    print(
        f"  mean z^2 {dist['mean_squared_z']:.6f}   median z^2 "
        f"{dist['median_squared_z']:.6f}   ratio {dist['mean_to_median_squared_z_ratio']:.2f}"
    )
    print(
        f"  largest single z^2 is {100 * dist['share_of_sum_squares_from_largest']:.1f}% "
        f"of the total; largest two are "
        f"{100 * dist['share_of_sum_squares_from_largest_two']:.1f}%"
    )
    print("  z by observation:")
    for entry in uncertainty["z_by_observation"]:  # type: ignore[union-attr]
        print(
            f"    {entry['bucket']:<26s} z {entry['z']:+8.4f}  z^2 "
            f"{entry['squared_z']:8.4f}  sd {entry['denominator']:.6f}"
        )
    envelope = uncertainty["leave_one_out_envelope"]
    print(
        f"  leave-one-out mean z^2 range [{envelope['mean_squared_z'][0]:.4f}, "
        f"{envelope['mean_squared_z'][1]:.4f}]; crosses 1 = "
        f"{uncertainty['mean_z2_crosses_one_under_any_single_removal']}"
    )
    for level in COVERAGE_LEVELS:
        key = f"coverage_{int(round(level * 100))}"
        print(f"  leave-one-out {key} range {envelope[key]}")
    universe_note = uncertainty["the_quoted_0_665_is_a_different_universe"]
    print(
        f"  the quoted raw mean z^2 {universe_note['quoted_raw_mean_squared_z']:.6f} "
        f"comes from {universe_note['source_artifact']}"
    )
    print(
        f"    its universe: {universe_note['source_seasons']}, "
        f"{universe_note['source_bucket_count']} buckets; "
        f"same observations as the coverage table = {universe_note['same_observations']}"
    )
    print(
        f"    same four buckets on the scored fold: mean z^2 "
        f"{uncertainty['same_four_buckets_on_the_scored_fold']['mean_squared_z']:.6f}"
    )
    print(
        f"    all twelve buckets both folds:       mean z^2 "
        f"{uncertainty['all_twelve_buckets_both_folds']['mean_squared_z']:.6f}"
    )
    print(f"  tension: {uncertainty['tension_resolution']['resolution']}")
    print(f"  VERDICT: {uncertainty['verdict']}")

    rule("2. HELD-OUT GAME UNIVERSES")
    residuals = pd.read_parquet(args.residuals)
    universes = game_universes(residuals)
    print(f"  rule          {universes['rule']}")
    print(f"  encoded in    {universes['rule_location']}")
    print(f"  depends on    {universes['rule_depends_only_on']}")
    print(f"  model/performance dependent: {universes['rule_depends_on_model_or_performance']}")
    print(
        f"  600 universe {universes['total_600_universe']}  "
        f"400 universe {universes['total_400_universe']}  "
        f"intersection {universes['intersection']}"
    )
    print(
        f"  in 400 not in 600: {universes['in_400_not_in_600']}   "
        f"in 600 not in 400: {universes['in_600_not_in_400']}   "
        f"nested: {universes['400_is_subset_of_600']}"
    )
    provenance_of_n = universes["subsample_density_provenance"]
    print(
        f"  the 200/season default was committed in "
        f"{provenance_of_n['default_introduced_in'][:12]} at "
        f"{provenance_of_n['default_introduced_at']}; predates the "
        f"remediation fork: {provenance_of_n['default_predates_the_remediation_fork']}"
    )
    for reason, entry in universes["exclusions_grouped_by_reason"].items():  # type: ignore[union-attr]
        print(
            f"  excluded by {reason}: {entry['count']} "
            f"(performance based: {entry['performance_based']})"
        )
    print(f"  CLASSIFICATION {universes['classification']['statement']}")
    candidate_grades = pd.read_parquet(root / "joint_event_grades.parquet")
    control_grades = pd.read_parquet(args.control_grades)
    paired = grades_universe(candidate_grades, control_grades)
    print(
        f"  candidate and control grades: same games "
        f"{paired['game_sets_identical']}, same event keys row for row "
        f"{paired['event_keys_identical_row_for_row']}, same baselines "
        f"{paired['baselines_identical']}"
    )
    skipped = {
        "published_repair_600": repair_report["games_skipped"],
        "paired_control_400": control_report["games_skipped"],
        "paired_candidate_400": candidate_report["games_skipped"],
    }
    print(f"  games skipped by the eligibility filter: {skipped}")

    rule("3. COUNT RMSE")
    counts = count_rmse_definitions(repair_report, control_report, candidate_report)
    print(f"  statistic: {counts['definition']['statistic']}")
    print(
        f"  buckets {counts['definition']['bucket_count']}, weighting "
        f"{counts['definition']['weighting']}, SE weighting "
        f"{counts['definition']['se_weighting']}, role weighting "
        f"{counts['definition']['role_weighting']}"
    )
    for name, entry in counts["by_run"].items():  # type: ignore[union-attr]
        model = entry["by_model"]["candidate"]
        print(
            f"  {name:<24s} games {entry['games_simulated']:>4d}  buckets "
            f"{entry['bucket_count']}  recomputed {model['recomputed_unweighted_rmse']:.9f} "
            f"stored {model['stored']:.9f} match {model['matches']}"
        )
    print(
        f"  observed count buckets identical 600 vs 400: "
        f"{counts['observed_count_buckets_identical']['repair_600_vs_control_400']}"
    )
    print(
        f"  observed count buckets identical control vs candidate (400): "
        f"{counts['observed_count_buckets_identical']['control_400_vs_candidate_400']}"
    )
    print(
        f"  latent observed buckets identical 600 vs 400: "
        f"{counts['latent_observed_buckets_identical']['repair_600_vs_control_400']}"
    )
    print(
        f"  ARE_VALUES_DIRECTLY_COMPARABLE: {counts['are_values_directly_comparable']}\n"
        f"  latent control RMSE identical across universes: "
        f"{counts['latent_control_rmse_identical_across_universes']}"
    )
    equivalence = driver_equivalence()
    for path, entry in equivalence["by_file"].items():  # type: ignore[union-attr]
        print(
            f"  {Path(path).name:<28s} blob identical across the two runs: "
            f"{entry['identical']}  {entry['numstat'] or '(no diff)'}"
        )

    rule("4. PROPER SCORES")
    scores = proper_scores(candidate_grades, control_grades)
    inner_universe = temperature_universe(temperature)
    print(
        f"  temperature inner: {inner_universe['events']:,} events, "
        f"{inner_universe['games']} games, seasons "
        f"{inner_universe['seasons']}, {inner_universe['probability_source']}"
    )
    print(
        f"  holdout used for lambda selection: "
        f"{inner_universe['holdout_used_for_selection']}"
    )
    print(
        f"  final: {scores['universe']['events']:,} events over "
        f"{scores['universe']['games']} games, seasons "
        f"{scores['universe']['seasons']}, "
        f"{scores['universe']['probability_source']}, pushes possible "
        f"{scores['universe']['pushes_possible']}"
    )
    print(
        f"  {'legs':>4s} {'LL ctrl':>10s} {'LL cand':>10s} {'LL prod':>10s} "
        f"{'cand-ctrl':>12s} {'cand-prod':>12s}"
    )
    for legs, entry in scores["by_legs"].items():  # type: ignore[union-attr]
        print(
            f"  {legs:>4s} {entry['log_loss_control']:10.6f} "
            f"{entry['log_loss_candidate']:10.6f} "
            f"{entry['log_loss_production']:10.6f} "
            f"{entry['log_loss_candidate_minus_control']['delta']:+12.3e} "
            f"{entry['log_loss_candidate_minus_production']['delta']:+12.3e}"
        )
    for legs, entry in scores["by_legs"].items():  # type: ignore[union-attr]
        for label in (
            "log_loss_candidate_minus_control",
            "log_loss_candidate_minus_production",
        ):
            margin = entry[label]
            print(
                f"  {legs}-leg {label[9:]:<28s} {margin['delta']:+.6e} "
                f"[{margin['ci95_low']:+.6e}, {margin['ci95_high']:+.6e}] "
                f"significant={margin['significant']}"
            )

    rule("5. PROVENANCE")
    lineage = provenance(inner, temperature, frozen, gates)
    for key, value in lineage.items():
        print(f"  {key:<42s} {value}")
    refactors = post_selection_refactors(inner)
    for entry in refactors:
        print(
            f"  post-selection refactor of {entry['artifact']}: "
            f"{len(entry['files_changed_since'])} files changed since "
            f"{entry['recorded_code_sha'][:8]}"
        )
        for path in entry["behaviour_affecting"]:  # type: ignore[union-attr]
            print(f"    behaviour-relevant: {path}")

    rule("6. HALF LIFE")
    half = half_life_equivalence(inner, diagnostics)
    print(f"  selected treatment {half['selected_treatment']}")
    print(f"  carried half life  {half['half_life_carried_in_frozen_choices']}")
    print(
        f"  null in every bucket fit: {half['half_life_null_in_every_bucket_fit']}"
    )
    for entry in half["probes"]:  # type: ignore[union-attr]
        print(
            f"    half_life={entry['half_life_passed']:<8} -> model "
            f"{entry['model']}, recorded {entry['half_life_recorded']}, "
            f"mean {entry['posterior_mean_hex']}"
        )
    print(
        f"  bitwise identical across every probe: "
        f"{half['bitwise_identical_across_every_probe']}"
    )
    print(f"  status {half['half_life_status']}")

    rule("7. ROLE SCALE")
    control_spec = load(
        PROJECT_ROOT
        / "research"
        / "game_latent_state_bucket_repair"
        / "factor_spec.json"
    )
    roles = role_scale_audit(
        residuals,
        candidate_spec_payload,
        control_spec,
        diagnostics.get("role_scale", {}),
    )
    identification = roles["identification"]
    print(f"  redesigned {roles['role_scale_was_redesigned']}  refitted {roles['role_scale_was_refitted']}")
    print(
        f"  support rule: games >= {MIN_CELL_GAMES} and pairs >= {MIN_CELL_PAIRS}; "
        f"{roles['cells_supported']}/{roles['cells_total']} supported, "
        f"{roles['cells_shrunk_to_pooled']} shrunk to pooled"
    )
    print(
        f"  {'cell':<20s} {'games':>6s} {'eff':>8s} {'obs':>9s} {'ctrl':>9s} "
        f"{'cand':>9s} {'z ctrl':>8s} {'z cand':>8s} {'d|z|':>8s}"
    )
    for label, entry in roles["by_cell"].items():  # type: ignore[union-attr]
        if not entry.get("present"):
            print(f"  {label:<20s} ABSENT")
            continue
        print(
            f"  {label:<20s} {entry['games']:6d} {entry['effective_games']:8.1f} "
            f"{entry['observed_scalar']:+9.6f} {entry['control_scalar']:+9.6f} "
            f"{entry['candidate_scalar']:+9.6f} {entry['control_scalar_z']:+8.3f} "
            f"{entry['candidate_scalar_z']:+8.3f} {entry['delta_abs_scalar_z']:+8.3f}"
        )
    print(
        f"  worst supported deterioration (rmse): "
        f"{roles['worst_supported_cell_deterioration_rmse']}"
    )
    print(
        f"  worst supported deterioration (rms z): "
        f"{roles['worst_supported_cell_deterioration_rms_z']}"
    )
    print(f"  role scales {identification['scales']}")
    print(f"  shrinkage towards pooled {identification['shrinkage_factor_towards_pooled']}")
    print(
        f"  weighted mean scale recomputed {identification['recomputed_weighted_mean_scale']!r}"
        f"  minus one {identification['recomputed_weighted_mean_scale_minus_one']!r}"
        f"  holds {identification['constraint_holds_exactly_within_tolerance']}"
    )

    rule("8. ORIGINAL GATE VERSUS IMPLEMENTED GATE")
    reconciliation = load(root / "audit_reconciliation.json")
    mapping = gate_mapping(gates, reconciliation, uncertainty, scores)
    for entry in mapping:
        status = (
            "PASS" if entry["would_the_original_gate_pass"] is True else "FAIL"
        )
        print(
            f"  [{status}] identical={str(entry['identical']):<5s} "
            f"{entry['original_gate'][:64]}"
        )
        if not entry["identical"]:
            print(f"          implemented: {entry['final_implemented_gate']}")
    failing = [
        entry
        for entry in mapping
        if entry["would_the_original_gate_pass"] is not True
    ]
    print(f"  original gates that would fail: {len(failing)} of {len(mapping)}")
    for entry in failing:
        print(f"    - {entry['original_gate']}")

    rule("9. FINDINGS")
    for entry in findings():
        print(f"  {entry['id']} [{entry['severity']}] {entry['what']}")

    out = {
        "title": "Final audit: uncertainty, universes, count RMSE, provenance",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "audit_only": True,
        "model_changes_made": "NONE",
        "candidate_factor_spec_hash": frozen["factor_spec_hash"],
        "uncertainty": uncertainty,
        "game_universes": universes,
        "paired_grades_universe": paired,
        "games_skipped_by_run": skipped,
        "count_rmse": counts,
        "driver_equivalence": equivalence,
        "proper_scores": scores,
        "temperature_universe": inner_universe,
        "provenance": lineage,
        "post_selection_refactors": refactors,
        "half_life": half,
        "findings": findings(),
        "role_scale": roles,
        "original_versus_implemented_gates": mapping,
        "original_gates_that_would_fail": [
            entry["original_gate"] for entry in failing
        ],
        "support_threshold": {
            "min_cell_games": MIN_CELL_GAMES,
            "min_cell_pairs": MIN_CELL_PAIRS,
        },
        "code_sha": git_sha(PROJECT_ROOT),
    }
    path = root / "final_audit.json"
    path.write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
