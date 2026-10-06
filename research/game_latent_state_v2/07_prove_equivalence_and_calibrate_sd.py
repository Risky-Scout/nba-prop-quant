#!/usr/bin/env python
"""Prove the frozen Shadow V2 model *is* the accepted repair, then score only the new part.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The inner screen rejected all three structural covariance refinements, so the
frozen candidate's loadings are the accepted repair's. That turns the expensive
holdout into two separate jobs, and only one of them needs computing.

What is proved rather than simulated
------------------------------------
A graded probability out of this framework is a deterministic function of six
things: the roster, the production marginals, the shared loadings, the
incumbent within-player blocks, the simulation count and the seed. Five of the
six are model-independent by construction -- the roster comes from the history
frame, the marginals and the within-player blocks come from ``fit_season``,
which takes no factor-model argument, and the count and seed are run-level
constants shared by every model in a paired run. The sixth, the loadings, is
the only place a dependence model can enter.

So if the frozen candidate's loadings are *bitwise* the repair's, then on the
same games with the same seed the two models' latent draws are bitwise equal,
every count they produce is bitwise equal, and every probability graded from
them is bitwise equal. Re-running the Monte Carlo would be a provable no-op:
20,000 draws per game across 600 games to reproduce numbers already committed
to ``joint_event_grades.parquet``. This driver establishes the premise instead,
by checking the loadings array by array, by checking the derived correlation
blocks, by checking the assembled game covariance and its Cholesky factor on
the real graded rosters in both refit windows, and by asserting the function
signatures that make the factorization above true.

What is actually computed
-------------------------
1.  The 2-, 3- and 4-leg log losses the brief asks for, from the committed
    joint-event grades, through the same audited ``summarize_joint_events`` the
    validator would have called. V1's grade file is row-aligned with the
    repair's -- same games, same order, bitwise identical baselines -- so the
    paired comparisons against V1, the repair and both baselines all come out
    of existing grades with no simulation.
2.  The one thing Shadow V2 adds over the static repair: a calibrated
    predictive SD. That has never been scored out of sample, so it is scored
    here on untouched 2024 and 2025, in both refit windows, using the
    inflation factor frozen on pre-2024 inner folds. The factor is *read*, not
    refitted; fitting it against the holdout is what this driver must not do.

Season 2024 enters the 2025 refit window's training set, which is what a
walk-forward evaluation means and is why that window exists. It cannot reach
the frozen candidate: nothing here writes a hyperparameter, and the freeze
record's ``holdout_used_for_selection`` stays false.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state import temporal as T  # noqa: E402
from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    ArtifactManifest,
    finalize_manifest,
    sha256_file,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    GameDimension,
    SharedFactorLoadings,
    build_game_covariance,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.paths import (  # noqa: E402
    DEFAULT_RESEARCH_DATA_ROOT,
    FACTOR_SPEC_NAME,
    RESIDUAL_DATASET_NAME,
    research_processed_dir,
)
from nba_prop_quant.research.game_latent_state.simulator import (  # noqa: E402
    GameRoster,
    simulate_game,
)
from nba_prop_quant.research.game_latent_state.v2 import (  # noqa: E402
    REPAIR_CONTROL_SPEC,
    fit_v2_factors,
)

console = Console()

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

BRANCH = "research/nba-game-latent-state-shadow-v2-structural"

#: Buckets the temporal treatment was screened on, in the diagnostic's order.
#: The predictive SD is calibrated for these four and for nothing else.
BUCKETS: dict[str, tuple[str, str, str]] = {
    "teammate_ast_ast": ("same_team", "ast", "ast"),
    "teammate_reb_reb": ("same_team", "reb", "reb"),
    "passer_ast_teammate_pts": ("same_team", "ast", "pts"),
    "teammate_pts_reb": ("same_team", "pts", "reb"),
}

#: Bitwise is bitwise. These comparisons are over float64 arrays that either
#: came out of the same arithmetic or did not, so the tolerance is zero and
#: anything above it is a finding rather than noise.
EXACT = 0.0

# ---------------------------------------------------------------------
# pre-registered criteria for the predictive-SD calibration
# ---------------------------------------------------------------------
#
# Declared here, before any 2024 or 2025 bucket value is read, and judged on
# the *same* statistic the inner folds were judged on: ``mean(z^2)`` where
# ``z`` is the prediction error over the claimed predictive SD combined with
# the realised season's own sampling error. The inner folds scored 3.1183 for
# the selected treatment, which is what the inflation factor repairs.
#
# These criteria judge a reported uncertainty, not a fitted parameter of the
# probability model. The frozen candidate cannot move in response to them: the
# inflation factor is read from the pre-2024 diagnostic and is not refitted.
#
#: Eight fold-bucket observations (four buckets, two refit windows). Under
#: correct calibration ``mean(z^2)`` has mean 1 and standard deviation
#: ``sqrt(2/8) = 0.5``, so two sampling deviations above one is 2.0.
SD_MAX_CALIBRATED_MEAN_SQUARED_Z = 2.00

#: And the symmetric guard against declaring success by making the interval
#: absurdly wide: a claimed SD more than about twice too large would land here.
SD_MIN_CALIBRATED_MEAN_SQUARED_Z = 0.25

#: The calibration has to help. Stated as a strict improvement in distance
#: from one, so a factor that overshot past one would have to overshoot by
#: less than it started out short.
SD_MUST_BEAT_RAW = True


def git_sha(ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_numbered_script(relative: str):
    """Import a numbered driver for its pure functions.

    The validator's joint-event summariser is the audited definition of the
    log losses this driver reports, and calling it is how the numbers stay the
    ones the gate evaluator expects rather than a second implementation that
    agrees today. Every driver guards its entry point, so importing runs
    nothing.
    """
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------
# part 1: the equivalence proof
# ---------------------------------------------------------------------


def loading_arrays(loadings: SharedFactorLoadings) -> dict[str, np.ndarray | None]:
    """Every array a game covariance can read out of a loadings object."""
    return {
        "game": loadings.game,
        "team_contrast": loadings.contrast_matrix,
        "competition": loadings.competition,
        "symmetric": loadings.symmetric,
        "role_deviation": loadings.role_deviation,
    }


def derived_blocks(loadings: SharedFactorLoadings) -> dict[str, np.ndarray]:
    """The correlation blocks the covariance assembly is built from."""
    return {
        "pooled_same_team_correlation": loadings.pooled_same_team_correlation(),
        "same_team_correlation": loadings.same_team_correlation(),
        "cross_team_correlation": loadings.cross_team_correlation(),
    }


def compare_arrays(
    left: dict[str, np.ndarray | None],
    right: dict[str, np.ndarray | None],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if a is None and b is None:
            out[key] = {"both_absent": True, "identical": True}
            continue
        if a is None or b is None:
            out[key] = {
                "both_absent": False,
                "identical": False,
                "detail": f"present on one side only: left={a is not None}",
            }
            continue
        same_shape = a.shape == b.shape
        out[key] = {
            "shape": list(a.shape),
            "shapes_match": bool(same_shape),
            "bitwise_identical": bool(same_shape and np.array_equal(a, b)),
            "max_abs_difference": (
                float(np.max(np.abs(a - b))) if same_shape else float("inf")
            ),
            "identical": bool(same_shape and np.array_equal(a, b)),
        }
    return out


def assert_model_enters_only_through_loadings(validator) -> dict[str, object]:
    """Machine-check the factorization the proof rests on.

    If ``fit_season`` ever grew a loadings argument, or ``simulate_game`` grew
    a second model-bearing one, the argument that the repair's grades are the
    candidate's would stop following from the loadings being equal. So the
    signatures are asserted rather than described.
    """
    fit_params = tuple(inspect.signature(validator.fit_season).parameters)
    sim_params = tuple(inspect.signature(simulate_game).parameters)
    cov_params = tuple(inspect.signature(build_game_covariance).parameters)

    checks = {
        "fit_season_parameters": list(fit_params),
        "fit_season_is_model_independent": fit_params == ("history", "season", "stats"),
        "simulate_game_parameters": list(sim_params),
        "simulate_game_model_bearing_parameters": ["loadings"],
        "simulate_game_has_exactly_one_model_input": sim_params
        == (
            "roster",
            "marginals",
            "loadings",
            "within_player",
            "simulations",
            "seed",
            "tabulated_inverse_cdf",
        ),
        "build_game_covariance_parameters": list(cov_params),
        "build_game_covariance_has_exactly_one_model_input": cov_params
        == ("dimensions", "loadings", "within_player"),
    }
    checks["all_signature_checks_pass"] = all(
        value for key, value in checks.items() if isinstance(value, bool)
    )
    return checks


def graded_roster_geometries(
    residuals: pd.DataFrame,
    history: pd.DataFrame,
    season: int,
    games_per_season: int,
    min_expected_minutes: float,
) -> dict[int, GameRoster]:
    """Rebuild the rosters the committed grades were produced on.

    Same selection the validator makes: the season's games, thinned to
    ``games_per_season`` on an even index grid, players above the minutes
    floor, sorted by team then player. Rebuilding it here is what lets the
    covariance identity be checked on the games that were actually graded
    rather than on a convenient subset, and the game-id set is asserted
    against the grade file afterwards.
    """
    season_rows = residuals.loc[residuals["season"] == season]
    game_ids = sorted(int(value) for value in season_rows["game_id"].unique())
    if games_per_season and len(game_ids) > games_per_season:
        picks = np.linspace(0, len(game_ids) - 1, games_per_season)
        game_ids = [game_ids[round(index)] for index in picks]

    out: dict[int, GameRoster] = {}
    for game_id in game_ids:
        observations = season_rows.loc[season_rows["game_id"] == game_id]
        observations = observations.loc[
            observations["expected_minutes"].fillna(0.0) >= min_expected_minutes
        ]
        if observations["team_id"].nunique() != 2 or len(observations) < 6:
            continue
        roster_frame = history.loc[
            (history["game_id"] == game_id)
            & history["player_id"].isin(observations["player_id"])
        ].copy()
        if len(roster_frame) != len(observations):
            continue
        home_team = int(
            observations.loc[observations["is_home"].astype(bool), "team_id"].iloc[0]
            if observations["is_home"].astype(bool).any()
            else observations["team_id"].iloc[0]
        )
        roster_frame = roster_frame.sort_values(["team_id", "player_id"]).reset_index(
            drop=True
        )
        out[game_id] = GameRoster(
            game_id=game_id,
            home_team_id=home_team,
            frame=roster_frame,
            stats=STATS,
            role_column="role_bucket" if "role_bucket" in roster_frame else None,
        )
    return out


def within_player_variants(
    dimensions: tuple[GameDimension, ...],
    seed: int,
) -> list[dict[int, np.ndarray]]:
    """Deterministic stand-ins for the incumbent within-player blocks.

    The incumbent block is model-independent -- it comes from the production
    copula refit, which never sees a factor model -- so the covariance identity
    has to hold for *every* admissible block, not just for the one the graded
    run happened to use. Checking it against several valid blocks is how that
    is demonstrated without refitting the production copula, which would be
    recomputing an input both models share.
    """
    players = sorted({dim.player_id for dim in dimensions})
    n = len(STATS)
    variants: list[dict[int, np.ndarray]] = [
        {player: np.eye(n) for player in players}
    ]
    rng = np.random.default_rng(seed)
    for _ in range(2):
        block: dict[int, np.ndarray] = {}
        for player in players:
            factor = rng.normal(size=(n, n))
            gram = factor @ factor.T + n * np.eye(n)
            scale = np.sqrt(np.diag(gram))
            block[player] = gram / np.outer(scale, scale)
        variants.append(block)
    return variants


def prove_covariance_identity(
    rosters: dict[int, GameRoster],
    candidate: SharedFactorLoadings,
    repair: SharedFactorLoadings,
    seed: int,
) -> dict[str, object]:
    """Assemble each graded game's covariance under both models and compare."""
    worst_correlation = 0.0
    worst_cholesky = 0.0
    offenders: list[dict[str, object]] = []
    games = 0
    blocks_checked = 0
    dimensions_max = 0

    for game_id, roster in sorted(rosters.items()):
        dimensions = roster.dimensions()
        dimensions_max = max(dimensions_max, len(dimensions))
        for variant_index, within in enumerate(
            within_player_variants(dimensions, seed + game_id)
        ):
            left = build_game_covariance(
                dimensions=dimensions, loadings=candidate, within_player=within
            )
            right = build_game_covariance(
                dimensions=dimensions, loadings=repair, within_player=within
            )
            correlation_gap = float(
                np.max(np.abs(left.correlation - right.correlation))
            )
            cholesky_gap = float(np.max(np.abs(left.cholesky - right.cholesky)))
            worst_correlation = max(worst_correlation, correlation_gap)
            worst_cholesky = max(worst_cholesky, cholesky_gap)
            blocks_checked += 1
            if correlation_gap > EXACT or cholesky_gap > EXACT:
                offenders.append(
                    {
                        "game_id": int(game_id),
                        "within_player_variant": variant_index,
                        "correlation_gap": correlation_gap,
                        "cholesky_gap": cholesky_gap,
                    }
                )
        games += 1

    return {
        "games_checked": games,
        "covariance_assemblies_compared": blocks_checked,
        "max_dimensions": dimensions_max,
        "within_player_variants_per_game": 3,
        "max_abs_correlation_difference": worst_correlation,
        "max_abs_cholesky_difference": worst_cholesky,
        "bitwise_identical_everywhere": not offenders,
        "offenders": offenders[:20],
        "note": (
            "the correlation matrix and its Cholesky factor are the only two "
            "objects the simulator reads out of the covariance, and the "
            "Cholesky is what multiplies the seeded normal draw, so equality "
            "here is equality of the draws"
        ),
    }


# ---------------------------------------------------------------------
# part 2: the predictive-SD calibration, on untouched 2024 and 2025
# ---------------------------------------------------------------------


def pooled_same_team_block(
    standardized: pd.DataFrame,
    target_season: int,
    bootstrap: int,
    seed: int,
) -> np.ndarray:
    """``T0``'s fitted same-team block for ``target_season``.

    ``T0`` is what the accepted repair does: pool every completed season into
    one empirical-Bayes fit and read the fitted same-team entry. Refitting it
    on seasons strictly before the target is what makes the 2025 window a
    refit window rather than a second read of the 2024 one. One fit serves
    every bucket, because every bucket is an entry of the same block.
    """
    history = standardized.loc[standardized["season"] < target_season]
    fit = fit_v2_factors(
        history,
        STATS,
        spec=replace(REPAIR_CONTROL_SPEC, role_column=None),
        bootstrap=bootstrap,
        seed=seed,
    )
    return fit.loadings.same_team_correlation()


def calibrate_predictive_sd(
    standardized: pd.DataFrame,
    inflation: float,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Score the calibrated predictive SD on both untouched refit windows."""
    windows = tuple(int(season) for season in sorted(HOLDOUT_SEASONS))
    index = {stat: position for position, stat in enumerate(STATS)}
    pooled_blocks = {
        window: pooled_same_team_block(
            standardized, target_season=window, bootstrap=bootstrap, seed=seed
        )
        for window in windows
    }
    by_bucket: dict[str, object] = {}
    raw_squared: list[float] = []
    calibrated_squared: list[float] = []
    observations: list[dict[str, object]] = []

    for name, (block, first_stat, second_stat) in BUCKETS.items():
        estimates = T.season_bucket_estimates(
            standardized,
            STATS,
            (first_stat, second_stat),
            block=block,
            bootstrap=bootstrap,
            seed=seed,
        )
        by_season = {int(item.season): item for item in estimates}
        entry: dict[str, object] = {"by_window": {}}
        for window in windows:
            realised = by_season.get(window)
            if realised is None:
                raise SystemExit(f"no realised {name} estimate for season {window}")
            pooled = float(
                pooled_blocks[window][index[first_stat], index[second_stat]]
            )
            prediction = T.treatment_prediction(
                T.TREATMENT_POOLED,
                estimates,
                target_season=window,
                pooled_estimate=pooled,
            )
            error = float(realised.estimate - prediction.prediction)
            observed_variance = float(realised.standard_error) ** 2
            raw_sd = float(prediction.prediction_sd)
            calibrated_sd = raw_sd * float(inflation)
            raw_total = observed_variance + raw_sd**2
            calibrated_total = observed_variance + calibrated_sd**2
            raw_z2 = error**2 / raw_total if raw_total > 0 else float("nan")
            calibrated_z2 = (
                error**2 / calibrated_total if calibrated_total > 0 else float("nan")
            )
            raw_squared.append(raw_z2)
            calibrated_squared.append(calibrated_z2)
            record = {
                "bucket": name,
                "refit_window": window,
                "training_seasons": list(prediction.training_seasons),
                "prediction": prediction.prediction,
                "raw_prediction_sd": raw_sd,
                "calibrated_prediction_sd": calibrated_sd,
                "observed": float(realised.estimate),
                "observed_standard_error": float(realised.standard_error),
                "observed_games": float(realised.games),
                "error": error,
                "raw_z": float(np.sign(error) * np.sqrt(raw_z2)),
                "calibrated_z": float(np.sign(error) * np.sqrt(calibrated_z2)),
                "raw_squared_z": raw_z2,
                "calibrated_squared_z": calibrated_z2,
                "inside_one_calibrated_sd_interval": bool(calibrated_z2 <= 1.0),
                "inside_two_calibrated_sd_interval": bool(calibrated_z2 <= 4.0),
            }
            entry["by_window"][str(window)] = record  # type: ignore[index]
            observations.append(record)
        by_bucket[name] = entry

    raw_mean = float(np.nanmean(raw_squared))
    calibrated_mean = float(np.nanmean(calibrated_squared))
    improved = abs(calibrated_mean - 1.0) < abs(raw_mean - 1.0)
    inside_ceiling = calibrated_mean <= SD_MAX_CALIBRATED_MEAN_SQUARED_Z
    inside_floor = calibrated_mean >= SD_MIN_CALIBRATED_MEAN_SQUARED_Z
    passed = bool(inside_ceiling and inside_floor and (improved or not SD_MUST_BEAT_RAW))

    return {
        "scope": "untouched 2024 and 2025, both refit windows",
        "refit_windows": list(windows),
        "inflation_factor_source": (
            "frozen in temporal_diagnostic.json from pre-2024 inner folds; "
            "read here, never refitted"
        ),
        "inflation_factor": float(inflation),
        "z_definition": (
            "z^2 = (observed - prediction)^2 / (observed_se^2 + prediction_sd^2), "
            "identical to screen_temporal_treatments so the holdout number is "
            "comparable with the inner-fold one"
        ),
        "by_bucket": by_bucket,
        "observations": observations,
        "pooled": {
            "observations": float(len(calibrated_squared)),
            "raw_mean_squared_z": raw_mean,
            "calibrated_mean_squared_z": calibrated_mean,
            "raw_implied_inflation_still_needed": float(np.sqrt(max(raw_mean, 0.0))),
            "calibrated_distance_from_one": abs(calibrated_mean - 1.0),
            "raw_distance_from_one": abs(raw_mean - 1.0),
        },
        "pre_registered_criteria": {
            "max_calibrated_mean_squared_z": SD_MAX_CALIBRATED_MEAN_SQUARED_Z,
            "min_calibrated_mean_squared_z": SD_MIN_CALIBRATED_MEAN_SQUARED_Z,
            "calibrated_must_beat_raw": SD_MUST_BEAT_RAW,
            "declared": (
                "in this driver, before any 2024 or 2025 bucket value was "
                "read, and judging a reported uncertainty rather than a fitted "
                "parameter of the probability model"
            ),
        },
        "verdicts": {
            "calibrated_inside_ceiling": bool(inside_ceiling),
            "calibrated_inside_floor": bool(inside_floor),
            "calibration_beats_raw": bool(improved),
        },
        "passed": passed,
    }


def explain_published_prediction(
    standardized: pd.DataFrame,
    temporal: dict,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Account for the gap between the published and the refitted ``T0`` mean.

    ``01_temporal_diagnostic.py`` reports a ``T0`` prediction for the first
    holdout season, but it reuses the *last inner fold's* pooled control --
    fitted on seasons ``< 2023`` -- rather than refitting on seasons
    ``< 2024``. That is the right choice there, where the point of the number
    is to extend the inner walk-forward by one step with the fold machinery
    intact. It is the wrong choice here: a 2024 holdout prediction may use
    every completed season before 2024, and the accepted repair's own
    validation manifest refits per validation season on seasons strictly
    before it.

    So the two numbers are expected to differ, and the difference has to be
    attributable to the refit window and nothing else. Two bitwise checks
    pin that down: the published value is the recorded ``< 2023`` fold
    control, and this driver's pooled-block code reproduces that same
    ``< 2023`` control when it is handed the same window.
    """
    index = {stat: position for position, stat in enumerate(STATS)}
    last_inner_fold = max(
        int(fold)
        for bucket in temporal["buckets"].values()
        for fold in bucket["pooled_control_by_fold"]
    )
    recomputed_block = pooled_same_team_block(
        standardized, target_season=last_inner_fold, bootstrap=bootstrap, seed=seed
    )
    published = temporal["uncertainty_calibration"]["by_bucket"]

    by_bucket: dict[str, object] = {}
    provenance_exact = True
    implementation_exact = True
    for name, (_block, first_stat, second_stat) in BUCKETS.items():
        recorded = float(
            temporal["buckets"][name]["pooled_control_by_fold"][str(last_inner_fold)]
        )
        recomputed = float(recomputed_block[index[first_stat], index[second_stat]])
        published_prediction = float(published[name]["prediction"])
        provenance = published_prediction == recorded
        implementation = recomputed == recorded
        provenance_exact &= provenance
        implementation_exact &= implementation
        by_bucket[name] = {
            "published_prediction": published_prediction,
            "recorded_inner_fold_control": recorded,
            "published_is_the_inner_fold_control": bool(provenance),
            "this_driver_reproduces_that_control": bool(implementation),
            "recomputed_inner_fold_control": recomputed,
        }

    return {
        "last_inner_fold": last_inner_fold,
        "published_prediction_trained_on": f"seasons < {last_inner_fold}",
        "holdout_prediction_trained_on": "seasons < the refit window",
        "published_prediction_is_the_last_inner_fold_control": bool(provenance_exact),
        "this_driver_reproduces_the_inner_fold_control_bitwise": bool(
            implementation_exact
        ),
        "difference_is_attributable_to_the_refit_window_alone": bool(
            provenance_exact and implementation_exact
        ),
        "by_bucket": by_bucket,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESEARCH_DATA_ROOT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research" / "game_latent_state_v2",
    )
    parser.add_argument(
        "--v1-artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research" / "game_latent_state",
    )
    parser.add_argument(
        "--repair-artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research" / "game_latent_state_bucket_repair",
    )
    # The inner-fold mean z^2 this holdout number is compared against was
    # scored at 800 draws. The empirical-Bayes weights are functions of the
    # bootstrap standard errors, so a different draw count moves the pooled
    # control and makes the two numbers incomparable.
    parser.add_argument("--bootstrap", type=int, default=800)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--games-per-season", type=int, default=300)
    parser.add_argument("--min-expected-minutes", type=float, default=8.0)
    parser.add_argument(
        "--equivalence-games",
        type=int,
        default=0,
        help="cap the covariance identity check (0 = every graded game)",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_root)
    repair_dir = Path(args.repair_artifact_root)
    v1_dir = Path(args.v1_artifact_root)

    v2_spec = json.loads((artifact_dir / FACTOR_SPEC_NAME).read_text(encoding="utf-8"))
    repair_spec = json.loads((repair_dir / FACTOR_SPEC_NAME).read_text(encoding="utf-8"))
    freeze = json.loads(
        (artifact_dir / "shadow_v2_candidate.json").read_text(encoding="utf-8")
    )
    temporal = json.loads(
        (artifact_dir / "temporal_diagnostic.json").read_text(encoding="utf-8")
    )
    if not freeze.get("frozen"):
        raise SystemExit("the candidate record does not say it is frozen")
    if freeze["factor_spec_hash"] != v2_spec["spec_hash"]:
        raise SystemExit("the factor spec no longer matches the frozen record")
    if freeze.get("holdout_used_for_selection", True):
        raise SystemExit("the freeze claims the holdout was used for selection")
    if temporal.get("holdout_used_for_selection", True):
        raise SystemExit("the temporal diagnostic claims the holdout was used")

    console.rule("Shadow V2: equivalence proof and predictive-SD calibration")
    console.print(f"frozen candidate : {freeze['candidate_name']}")
    console.print(
        "structural dials : "
        f"r_symmetric={freeze['hyperparameters']['r_symmetric']}, "
        f"role_deviation={freeze['hyperparameters']['role_deviation']}, "
        f"bridge_weight={freeze['hyperparameters']['bridge_weight']}"
    )

    # ---- part 1: the equivalence proof -----------------------------
    candidate_loadings = SharedFactorLoadings.from_payload(v2_spec["loadings"])
    repair_loadings = SharedFactorLoadings.from_payload(repair_spec["loadings"])

    console.rule("Part 1: the frozen candidate is the accepted repair")
    payload_identical = v2_spec["loadings"] == repair_spec["loadings"]
    array_report = compare_arrays(
        loading_arrays(candidate_loadings), loading_arrays(repair_loadings)
    )
    derived_report = compare_arrays(
        derived_blocks(candidate_loadings), derived_blocks(repair_loadings)
    )
    moments_identical = (
        v2_spec["standardization_moments"] == repair_spec["standardization_moments"]
    )
    stats_identical = list(v2_spec["stats"]) == list(repair_spec["stats"])

    table = Table(title="loadings, array by array")
    for column in ("array", "shape", "bitwise identical", "max |difference|"):
        table.add_column(column, justify="right")
    for key, entry in array_report.items():
        table.add_row(
            key,
            str(entry.get("shape", "absent")),
            "yes" if entry["identical"] else "NO",
            (
                f"{entry['max_abs_difference']:.3e}"
                if "max_abs_difference" in entry
                else "-"
            ),
        )
    for key, entry in derived_report.items():
        table.add_row(
            key,
            str(entry.get("shape", "absent")),
            "yes" if entry["identical"] else "NO",
            (
                f"{entry['max_abs_difference']:.3e}"
                if "max_abs_difference" in entry
                else "-"
            ),
        )
    console.print(table)

    validator = load_numbered_script(
        "research/game_latent_state_v2/05_validate_shadow_v2.py"
    )
    signature_checks = assert_model_enters_only_through_loadings(validator)
    console.print(
        "signature checks  : "
        + ("all pass" if signature_checks["all_signature_checks_pass"] else "FAILED")
    )

    repair_grades_path = repair_dir / "joint_event_grades.parquet"
    repair_report_path = repair_dir / "validation_report.json"
    v1_grades_path = v1_dir / "joint_event_grades.parquet"
    repair_manifest = json.loads(
        (repair_dir / "manifest.validation.json").read_text(encoding="utf-8")
    )
    grades_fingerprint_ok = repair_manifest["outputs"][
        "joint_event_grades.parquet"
    ] == sha256_file(repair_grades_path)
    report_fingerprint_ok = repair_manifest["outputs"][
        "validation_report.json"
    ] == sha256_file(repair_report_path)
    console.print(
        "reused grade files match the repair's own manifest fingerprints: "
        f"{grades_fingerprint_ok and report_fingerprint_ok}"
    )

    residuals = pd.read_parquet(v1_dir / RESIDUAL_DATASET_NAME)
    residuals["season"] = residuals["season"].astype(int)
    history = pd.read_parquet(
        research_processed_dir(args.data_root) / "oof_selected_means.parquet"
    )
    history["season"] = history["season"].astype(int)

    repair_grades = pd.read_parquet(repair_grades_path)
    graded_game_ids = {int(value) for value in repair_grades["game_id"].unique()}

    rosters: dict[int, GameRoster] = {}
    for season in HOLDOUT_SEASONS:
        rosters.update(
            graded_roster_geometries(
                residuals,
                history,
                season=int(season),
                games_per_season=args.games_per_season,
                min_expected_minutes=args.min_expected_minutes,
            )
        )
    reconstruction_matches = set(rosters) == graded_game_ids
    console.print(
        f"rebuilt {len(rosters)} roster geometries; they are exactly the "
        f"{len(graded_game_ids)} graded games: {reconstruction_matches}"
    )
    if not reconstruction_matches:
        console.print(
            f"  only in reconstruction: {sorted(set(rosters) - graded_game_ids)[:8]}"
        )
        console.print(
            f"  only in grade file    : {sorted(graded_game_ids - set(rosters))[:8]}"
        )

    checked = dict(sorted(rosters.items()))
    if args.equivalence_games:
        checked = dict(list(checked.items())[: args.equivalence_games])
    console.print(
        f"assembling the game covariance under both models on {len(checked)} games"
    )
    covariance_report = prove_covariance_identity(
        checked, candidate_loadings, repair_loadings, seed=args.seed
    )
    console.print(
        "  max |correlation difference| = "
        f"{covariance_report['max_abs_correlation_difference']:.3e}   "
        "max |Cholesky difference| = "
        f"{covariance_report['max_abs_cholesky_difference']:.3e}"
    )

    equivalence_passed = bool(
        payload_identical
        and all(entry["identical"] for entry in array_report.values())
        and all(entry["identical"] for entry in derived_report.values())
        and moments_identical
        and stats_identical
        and signature_checks["all_signature_checks_pass"]
        and grades_fingerprint_ok
        and report_fingerprint_ok
        and reconstruction_matches
        and covariance_report["bitwise_identical_everywhere"]
    )
    console.print(
        "\n[bold]POINT_PROBABILITY_MODEL_IDENTICAL_TO_ACCEPTED_REPAIR="
        + ("YES" if equivalence_passed else "NO")
        + "[/bold]"
    )

    # ---- part 2: the leg-count log losses, from existing grades ----
    console.rule("Part 2: 2/3/4-leg log loss from the committed joint-event grades")
    v1_grades = pd.read_parquet(v1_grades_path)
    key = ["game_id", "family", "n_legs"]
    aligned = repair_grades[key].equals(v1_grades[key]) and bool(
        (repair_grades["realized"].to_numpy() == v1_grades["realized"].to_numpy()).all()
    )
    baselines_identical = np.array_equal(
        repair_grades["p_baseline_independence"].to_numpy(),
        v1_grades["p_baseline_independence"].to_numpy(),
    ) and np.array_equal(
        repair_grades["p_baseline_production"].to_numpy(),
        v1_grades["p_baseline_production"].to_numpy(),
    )
    console.print(
        f"V1 and repair grade files are row-aligned: {aligned}; their shared "
        f"baselines are bitwise identical: {baselines_identical}"
    )
    if not (aligned and baselines_identical):
        raise SystemExit(
            "the two committed grade files are not a paired run, so a paired "
            "log-loss comparison across them would not be meaningful"
        )

    events = repair_grades.copy()
    # ``p_candidate`` in the repair's grade file is the frozen candidate's own
    # probability: part 1 proved the two models produce bitwise identical
    # draws on these games at this seed. ``p_repair`` is the same column under
    # the name the validator's paired comparisons use, so the candidate-minus-
    # repair delta comes out as the exact zero it is rather than being omitted.
    events["p_repair"] = events["p_candidate"]
    events["p_v1"] = v1_grades["p_candidate"].to_numpy()

    joint_events = validator.summarize_joint_events(
        events, bootstrap=args.bootstrap, seed=args.seed
    )

    leg_table = Table(title="log loss by leg count (lower is better)")
    for column in (
        "legs",
        "events",
        "base rate",
        "candidate",
        "repair",
        "V1",
        "independence",
        "production",
        "cand - prod",
    ):
        leg_table.add_column(column, justify="right")
    for legs in sorted(joint_events["by_legs"], key=int):
        entry = joint_events["by_legs"][legs]
        delta = entry["paired"].get("candidate_minus_baseline_production", {})
        leg_table.add_row(
            legs,
            str(entry["events"]),
            f"{entry['base_rate']:.4f}",
            f"{entry['candidate']['log_loss']:.6f}",
            f"{entry['repair']['log_loss']:.6f}",
            f"{entry['v1']['log_loss']:.6f}",
            f"{entry['baseline_independence']['log_loss']:.6f}",
            f"{entry['baseline_production']['log_loss']:.6f}",
            (
                f"{delta['log_loss']['delta']:+.6f}"
                if delta and "log_loss" in delta
                else "-"
            ),
        )
    console.print(leg_table)

    # ---- part 3: the predictive-SD calibration ---------------------
    console.rule("Part 3: predictive-SD calibration on untouched 2024 and 2025")
    training_moments = {
        stat: (float(value["mean"]), float(value["sd"]))
        for stat, value in v2_spec["standardization_moments"].items()
    }
    # Every season is standardized with the *frozen training* constants, so the
    # predictions and the realised holdout estimates are in one set of units.
    # Restandardizing per window would change the units between the two
    # windows and make their z scores incomparable.
    standardized, applied = standardize_residuals(residuals, STATS, training_moments)
    for stat, (mean, scale) in applied.items():
        if not np.isclose(mean, training_moments[stat][0], rtol=0, atol=0) or not (
            np.isclose(scale, training_moments[stat][1], rtol=0, atol=0)
        ):
            raise SystemExit(f"{stat} was not standardized with the frozen constants")
    console.print(
        "standardized every season with the frozen 2020-2023 constants "
        f"({len(training_moments)} stats)"
    )

    inflation = float(temporal["uncertainty_calibration"]["pooled"]["inflation"])
    console.print(f"frozen inflation factor from pre-2024 inner folds: {inflation:.4f}")
    calibration = calibrate_predictive_sd(
        standardized, inflation, bootstrap=args.bootstrap, seed=args.seed
    )

    sd_table = Table(title="predictive SD out of sample, by bucket and refit window")
    for column in (
        "bucket",
        "window",
        "trained on",
        "prediction",
        "observed",
        "raw SD",
        "calib SD",
        "raw z",
        "calib z",
    ):
        sd_table.add_column(column, justify="right")
    for record in calibration["observations"]:  # type: ignore[union-attr]
        sd_table.add_row(
            str(record["bucket"]),
            str(record["refit_window"]),
            str(record["training_seasons"]),
            f"{record['prediction']:+.6f}",
            f"{record['observed']:+.6f}",
            f"{record['raw_prediction_sd']:.6f}",
            f"{record['calibrated_prediction_sd']:.6f}",
            f"{record['raw_z']:+.3f}",
            f"{record['calibrated_z']:+.3f}",
        )
    console.print(sd_table)
    pooled = calibration["pooled"]  # type: ignore[index]
    console.print(
        f"  pooled mean z^2: raw {pooled['raw_mean_squared_z']:.4f} -> "
        f"calibrated {pooled['calibrated_mean_squared_z']:.4f} "
        f"over {pooled['observations']:.0f} observations "
        f"(inner folds scored {temporal['uncertainty_calibration']['pooled']['mean_squared_z']:.4f} raw)"
    )

    # Self-check: the predictive SD is the quantity under test, and the frozen
    # temporal diagnostic already published its 2024-window value, so this
    # driver's arithmetic has to reproduce it bitwise. The accompanying point
    # prediction is expected to differ, because the published one reuses the
    # last inner fold's pooled control; `explain_published_prediction` is what
    # holds that difference to the refit window and nothing else.
    published = temporal["uncertainty_calibration"]["by_bucket"]
    reproduction: dict[str, object] = {}
    for name in BUCKETS:
        window = calibration["by_bucket"][name]["by_window"]["2024"]  # type: ignore[index]
        reproduction[name] = {
            "published_prediction": published[name]["prediction"],
            "recomputed_prediction": window["prediction"],
            "prediction_difference": abs(
                float(published[name]["prediction"]) - float(window["prediction"])
            ),
            "published_raw_sd": published[name]["raw_prediction_sd"],
            "recomputed_raw_sd": window["raw_prediction_sd"],
            "raw_sd_difference": abs(
                float(published[name]["raw_prediction_sd"])
                - float(window["raw_prediction_sd"])
            ),
        }
    worst_sd_reproduction = max(
        float(entry["raw_sd_difference"])
        for entry in reproduction.values()  # type: ignore[union-attr]
    )
    console.print(
        "  2024-window predictive SD reproduces the frozen diagnostic to "
        f"{worst_sd_reproduction:.3e}"
    )

    provenance = explain_published_prediction(
        standardized, temporal, bootstrap=args.bootstrap, seed=args.seed
    )
    console.print(
        "  the published 2024 point prediction is the "
        f"< {provenance['last_inner_fold']} fold control "
        f"(bitwise: {provenance['published_prediction_is_the_last_inner_fold_control']}), "
        "which this driver reproduces bitwise: "
        f"{provenance['this_driver_reproduces_the_inner_fold_control_bitwise']}; "
        "the holdout refits on seasons < the window"
    )
    if not provenance["difference_is_attributable_to_the_refit_window_alone"]:
        raise SystemExit(
            "the published T0 prediction is not the recorded inner-fold control, "
            "so the gap against the refitted one is unexplained"
        )
    if worst_sd_reproduction != 0.0:
        raise SystemExit(
            "the 2024-window predictive SD does not reproduce the frozen "
            f"diagnostic bitwise (worst {worst_sd_reproduction:.3e}); the "
            "holdout mean z^2 is not comparable with the inner-fold one"
        )

    console.print(
        "\n[bold]PREDICTIVE_SD_CALIBRATION_PASSED="
        + ("YES" if calibration["passed"] else "NO")
        + "[/bold]"
    )

    # ---- the record -------------------------------------------------
    report = {
        "scope": (
            "equivalence proof plus the only two quantities the frozen "
            "candidate does not inherit from the accepted repair"
        ),
        "monte_carlo_run": False,
        "point_probabilities_resimulated": False,
        "factor_model_refitted": False,
        "equivalence": {
            "passed": equivalence_passed,
            "loadings_payload_bitwise_identical": bool(payload_identical),
            "loading_arrays": array_report,
            "derived_correlation_blocks": derived_report,
            "standardization_moments_identical": bool(moments_identical),
            "stat_order_identical": bool(stats_identical),
            "model_enters_only_through_loadings": signature_checks,
            "graded_roster_reconstruction_matches_grade_file": bool(
                reconstruction_matches
            ),
            "game_covariance_identity": covariance_report,
            "reused_artifact_fingerprints": {
                "joint_event_grades.parquet": sha256_file(repair_grades_path),
                "validation_report.json": sha256_file(repair_report_path),
                "matches_repair_manifest": bool(
                    grades_fingerprint_ok and report_fingerprint_ok
                ),
            },
            "conclusion": (
                "The frozen candidate and the accepted repair differ in no "
                "input the simulator reads, so on the graded games at the "
                "graded seed their draws are bitwise equal and every "
                "probability graded from them is bitwise equal. The repair's "
                "committed grades are the candidate's grades."
            ),
        },
        "reused_point_metrics": {
            "source": "research/game_latent_state_bucket_repair/validation_report.json",
            "source_sha256": sha256_file(repair_report_path),
            "justified_by": "equivalence.conclusion",
            "games_simulated_in_the_reused_run": json.loads(
                repair_report_path.read_text(encoding="utf-8")
            )["games_simulated"],
            "sections": [
                "latent_dependence",
                "residual_dependence",
                "marginal_preservation",
                "same_player_contract",
                "stability",
            ],
            **{
                section: json.loads(repair_report_path.read_text(encoding="utf-8"))[
                    section
                ]
                for section in (
                    "latent_dependence",
                    "residual_dependence",
                    "marginal_preservation",
                    "same_player_contract",
                    "stability",
                )
            },
        },
        "joint_events": joint_events,
        "joint_events_provenance": {
            "computed_from": [
                "research/game_latent_state_bucket_repair/joint_event_grades.parquet",
                "research/game_latent_state/joint_event_grades.parquet",
            ],
            "repair_grades_sha256": sha256_file(repair_grades_path),
            "v1_grades_sha256": sha256_file(v1_grades_path),
            "grade_files_row_aligned": bool(aligned),
            "shared_baselines_bitwise_identical": bool(baselines_identical),
            "candidate_column_is": (
                "the repair's p_candidate, which part 1 proves is the frozen "
                "candidate's own probability on these games at this seed"
            ),
            "nothing_was_simulated": True,
        },
        "predictive_sd_calibration": calibration,
        "predictive_sd_2024_window_reproduces_frozen_diagnostic": {
            "by_bucket": reproduction,
            "worst_raw_sd_difference": worst_sd_reproduction,
            "predictive_sd_exact": bool(worst_sd_reproduction == 0.0),
            "point_prediction_gap_explained": provenance,
        },
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "frozen_candidate": freeze["candidate_name"],
        "frozen_hyperparameters": freeze["hyperparameters"],
        "factor_spec_hash": v2_spec["spec_hash"],
        "parent_shadow_v1_sha": freeze["parent_shadow_v1_sha"],
        "parent_bucket_repair_sha": freeze["parent_bucket_repair_sha"],
        "branch": BRANCH,
        "seed": int(args.seed),
        "bootstrap_draws": int(args.bootstrap),
        "code_sha": git_sha(),
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "production_integration_started": False,
    }
    report_path = write_json(report, artifact_dir / "equivalence_and_calibration.json")

    manifest = ArtifactManifest(
        artifact_name="game_latent_state_shadow_v2_equivalence_and_calibration",
        source_production_sha=git_sha("origin/production/wizardofodds-integration"),
        source_production_ref="origin/production/wizardofodds-integration",
        code_sha=git_sha(),
        branch=BRANCH,
        seed=int(args.seed),
        training_cutoff=f"season<{min(HOLDOUT_SEASONS)} for the candidate",
        seasons_used=[int(season) for season in sorted(residuals["season"].unique())],
        training_seasons=list(v2_spec["training_seasons"]),
        validation_seasons=list(HOLDOUT_SEASONS),
        input_fingerprints={
            RESIDUAL_DATASET_NAME: sha256_file(v1_dir / RESIDUAL_DATASET_NAME),
            "v2_factor_spec.json": sha256_file(artifact_dir / FACTOR_SPEC_NAME),
            "repair_factor_spec.json": sha256_file(repair_dir / FACTOR_SPEC_NAME),
            "repair_joint_event_grades.parquet": sha256_file(repair_grades_path),
            "repair_validation_report.json": sha256_file(repair_report_path),
            "v1_joint_event_grades.parquet": sha256_file(v1_grades_path),
            "temporal_diagnostic.json": sha256_file(
                artifact_dir / "temporal_diagnostic.json"
            ),
        },
        parameters={
            "bootstrap_draws": int(args.bootstrap),
            "games_per_season": int(args.games_per_season),
            "monte_carlo_run": False,
            "inflation_factor": inflation,
            "pre_registered_sd_criteria": calibration["pre_registered_criteria"],  # type: ignore[index]
        },
        notes=[
            "No game was simulated. The point probabilities are the accepted "
            "repair's committed grades, reused because the frozen candidate's "
            "loadings are bitwise the repair's.",
            "Season 2024 enters the 2025 refit window's training set, which is "
            "what a walk-forward predictive-SD test means; it reaches no "
            "hyperparameter of the frozen candidate.",
        ],
    )
    finalize_manifest(
        manifest,
        artifact_dir,
        outputs={"equivalence_and_calibration.json": report_path},
        manifest_name="manifest.equivalence.json",
        checksum_name="SHA256SUMS.equivalence.txt",
    )

    console.rule("Verdict")
    console.print(
        "POINT_PROBABILITY_MODEL_IDENTICAL_TO_ACCEPTED_REPAIR="
        + ("YES" if equivalence_passed else "NO")
    )
    console.print(
        "PREDICTIVE_SD_CALIBRATION_PASSED="
        + ("YES" if calibration["passed"] else "NO")
    )
    console.print("MONTE_CARLO_HOLDOUT_RUN=NO")
    console.print("POINT_PROBABILITIES_RESIMULATED=NO")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")
    console.print("PRODUCTION_INTEGRATION_STARTED=NO")
    console.print(f"\nwrote {report_path}")
    if not (equivalence_passed and calibration["passed"]):
        raise SystemExit("equivalence or calibration did not pass; see the report")


if __name__ == "__main__":
    main()
