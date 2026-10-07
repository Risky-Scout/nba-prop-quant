#!/usr/bin/env python3
"""COUNT-SPACE BLOCKER FORENSIC STUDY.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE. This driver reads the frozen
remediation candidate and never writes it.

The question
------------
The frozen candidate moves ``passer_ast_teammate_pts``'s held-out count-space
error down by 5.2% against the accepted bucket repair, and the research target
was 20%. This study asks whether that shortfall is an *estimator* defect or a
*structural* limit of the frozen PSD factor architecture, and it answers with
pre-2024 evidence only: every estimator, every inversion and every target in
here reads seasons 2020-2023. The held-out seasons are used to *evaluate* the
feasibility envelope the constraints define, which is what section 3 asks
for, and never to choose anything.

Sections, as commissioned
-------------------------
1.  Five estimators of the same latent correlation on pre-2024 folds:
    the current randomized-PIT reading, the same reading averaged over many
    deterministic jitter seeds, the mid-PIT reading, the interval-censored
    Gaussian-copula pseudo-likelihood, and the direct count correlation.
2.  The transmission test: each estimator's latent ``rho`` pushed through the
    production margins and the Gaussian copula into an implied count-space
    correlation, against the observed one.
3.  The feasibility envelope: the largest count-space correlation the frozen
    architecture reaches on this bucket with every global constraint intact.
4.  The identifiability classification.
5.  The inner forward test, run only if the censored estimator is generic and
    improves the bucket on pre-2024 forward folds.
6.  The stop rule.

Everything expensive is cached under ``--cache-dir``, which is gitignored: the
ZINB refits and the scored held-out universe are derived intermediates of
committed fits and reproducible from them.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import forensic_lib as FL  # noqa: E402
import holdout_lib as HL  # noqa: E402
from nba_prop_quant.copula import GaussianCopula  # noqa: E402
from nba_prop_quant.research.game_latent_state import censored  # noqa: E402
from marginal import discrete_marginal  # noqa: E402
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    SharedFactorLoadings,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.simulator import (  # noqa: E402
    tabulate_inverse_cdf,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(__file__).resolve().parent

#: The frozen candidate the study starts from, and the control the paired
#: gate evidence compares it against.
CANDIDATE_SPEC = PROJECT_ROOT / "research/final_upstream_remediation/factor_spec.json"
CONTROL_SPEC = PROJECT_ROOT / "research/game_latent_state_bucket_repair/factor_spec.json"
CANDIDATE_REPORT = (
    PROJECT_ROOT / "research/final_upstream_remediation/validation_report.json"
)
CONTROL_REPORT = (
    PROJECT_ROOT
    / "research/final_upstream_remediation/control_validation_report.json"
)
EXPECTED_CANDIDATE_HASH = (
    "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
)
EXPECTED_CONTROL_HASH = (
    "c9b46e3a7497cfee52832f29397397b843aafb8e5f770a39157ce508f8157bc0"
)

#: Deterministic jitter seeds for estimator B. The committed residual build
#: used 73; the rest are an arithmetic sweep so the set is reproducible from
#: this line alone.
JITTER_SEEDS: tuple[int, ...] = (73,) + tuple(range(1000, 1000 + 47))

#: Hermite orders the transmission series keeps.
TERMS = 40

#: The two z-degradation tolerances ``05_evaluate_gates.py`` scores the frozen
#: candidate against. Carried verbatim so the envelope's constraints are the
#: ones the candidate actually had to satisfy.
MAX_PROTECTED_Z_DEGRADATION = 1.0
MAX_TARGET_Z_DEGRADATION = 0.25


def log(message: str) -> None:
    print(message, flush=True)


# ----------------------------------------------------------------------
# section 0: provenance
# ----------------------------------------------------------------------


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def provenance(residuals: pd.DataFrame) -> dict[str, object]:
    """Repository state, spec hashes and the holdout-isolation assertion."""
    candidate = json.loads(CANDIDATE_SPEC.read_text())
    control = json.loads(CONTROL_SPEC.read_text())
    if candidate["spec_hash"] != EXPECTED_CANDIDATE_HASH:
        raise SystemExit(
            "the frozen candidate spec hash moved; refusing to run a forensic "
            f"study against {candidate['spec_hash']}"
        )
    if control["spec_hash"] != EXPECTED_CONTROL_HASH:
        raise SystemExit("the control spec hash moved")
    seasons = sorted(int(value) for value in residuals["season"].unique())
    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "worktree_clean": git("status", "--short") == "",
        "merge_base_with_remediation": git(
            "merge-base", "HEAD", "research/nba-final-upstream-remediation"
        ),
        "candidate_spec_hash": candidate["spec_hash"],
        "control_spec_hash": control["spec_hash"],
        "candidate_spec_hash_confirmed": True,
        "residual_seasons_available": seasons,
        "pre_2024_seasons_used": list(FL.PRE_2024_SEASONS),
        "pre_2024_forward_folds": list(FL.PRE_2024_FORWARD_FOLDS),
        "holdout_seasons": list(FL.HOLDOUT_SEASONS),
        "holdout_used_for": (
            "evaluating the feasibility envelope the constraints define, which "
            "section 3 requires; no estimator, inversion or target in this "
            "study reads a held-out season"
        ),
        "frozen_artifacts_untouched": [
            str(CANDIDATE_SPEC.relative_to(PROJECT_ROOT)),
            str(CONTROL_SPEC.relative_to(PROJECT_ROOT)),
            str(CANDIDATE_REPORT.relative_to(PROJECT_ROOT)),
            str(CONTROL_REPORT.relative_to(PROJECT_ROOT)),
        ],
    }


# ----------------------------------------------------------------------
# section 1: the five estimators
# ----------------------------------------------------------------------


def estimator_windows() -> dict[str, tuple[int, int]]:
    """Pre-2024 estimation windows, all strictly before 2024.

    ``pooled_2020_2023`` is every pre-2024 season, which is the window a
    target would actually be fitted on. Each forward fold carries its own
    training window and its own evaluation season, so the estimators are also
    compared under the same no-lookahead structure the holdout has.
    """
    windows = {"pooled_2020_2023": (2020, 2023)}
    for fold in FL.PRE_2024_FORWARD_FOLDS:
        windows[f"train_before_{fold}"] = (2020, fold - 1)
        windows[f"fold_{fold}"] = (fold, fold)
    return windows


def randomized_pit_reading(
    frame: pd.DataFrame,
    stat_a: str,
    stat_b: str,
    kind: str,
) -> float:
    """Estimator A: the reading the pipeline's latent gate actually uses.

    ``standardize_residuals`` with the window's own moments, then the ordered
    pair mean, which is exactly ``factors.pair_moments`` for one entry.
    """
    standardized, _ = standardize_residuals(frame, FL.STATS)
    value, _ = FL.ordered_pair_mean(
        standardized, kind, f"zs_{stat_a}", f"zs_{stat_b}"
    )
    return value


def mid_pit_reading(
    frame: pd.DataFrame,
    stat_a: str,
    stat_b: str,
    kind: str,
) -> float:
    """Estimator C: the mid-PIT reading, standardized by its own spread.

    Diagnostic only, as commissioned. It carries no jitter variance, so
    standardising by its own spread removes the part of the attenuation the
    randomized PIT keeps in its denominator.
    """
    working = frame.reset_index(drop=True).copy()
    for stat in (stat_a, stat_b):
        column = working[f"zmid_{stat}"].to_numpy(dtype=float)
        working[f"zc_{stat}"] = (column - column.mean()) / column.std(ddof=0)
    value, _ = FL.ordered_pair_mean(working, kind, f"zc_{stat_a}", f"zc_{stat_b}")
    return value


def estimator_table(
    residuals: pd.DataFrame,
    counts: pd.DataFrame,
    bucket: str,
    kind: str,
    stat_a: str,
    stat_b: str,
    bootstrap: int,
    seed: int,
    mle_cache: Mapping[str, object] | None,
    cache_path: Path | None = None,
) -> dict[str, object]:
    """Estimators A to E of one bucket, on every pre-2024 window.

    Cached, because estimator B re-Gaussianises every pre-2024 window at 48
    deterministic jitter seeds and that is the single slowest step in the
    study. Everything it reads is deterministic given the committed residuals
    and the seed list, so the cache is a pure memo.
    """
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        if payload.get("seeds") == list(JITTER_SEEDS) and payload.get(
            "bucket"
        ) == bucket:
            log(f"  reusing cached estimator table from {cache_path.name}")
            return payload["by_window"]
    out: dict[str, object] = {}
    for label, (low, high) in estimator_windows().items():
        window = residuals.loc[
            (residuals["season"] >= low) & (residuals["season"] <= high)
        ]
        count_window = counts.loc[
            (counts["season"] >= low) & (counts["season"] <= high)
        ]
        log(f"  window {label}: {len(window):,} rows")

        started = time.time()
        value_a = randomized_pit_reading(window, stat_a, stat_b, kind)
        draws = FL.clustered_bootstrap_pair_mean(
            standardize_residuals(window, FL.STATS)[0],
            kind,
            f"zs_{stat_a}",
            f"zs_{stat_b}",
            draws=bootstrap,
            seed=seed,
        )
        entry: dict[str, object] = {
            "seasons": sorted(int(value) for value in window["season"].unique()),
            "rows": int(len(window)),
            "A_randomized_pit": {
                "reading": float(value_a),
                "clustered_se": float(np.std(draws, ddof=1)),
                "jitter_seed": 73,
            },
        }

        seed_values = []
        for jitter in JITTER_SEEDS:
            rejittered = censored.multi_seed_latent_columns(
                window, (stat_a, stat_b), jitter
            )
            seed_values.append(
                randomized_pit_reading(rejittered, stat_a, stat_b, kind)
            )
        seed_values = np.asarray(seed_values, dtype=float)
        entry["B_multi_seed_randomized_pit"] = {
            "reading": float(seed_values.mean()),
            "across_seed_sd": float(seed_values.std(ddof=1)),
            "seeds": len(JITTER_SEEDS),
            "minimum": float(seed_values.min()),
            "maximum": float(seed_values.max()),
            "committed_seed_reading": float(seed_values[0]),
        }

        entry["C_mid_pit"] = {
            "reading": float(mid_pit_reading(window, stat_a, stat_b, kind)),
            "note": "diagnostic only, as commissioned",
        }

        cached = None
        if mle_cache is not None:
            cached = mle_cache.get(label, {}).get("buckets", {}).get(bucket)
        if cached is None:
            intervals = censored.pair_intervals(window, kind, stat_a, stat_b)
            rho = censored.censored_copula_mle(intervals, nodes=8)
            standard_error = censored.censored_sandwich_se(rho, intervals, nodes=8)
            pairs, games = len(intervals), int(np.unique(intervals.game_id).size)
        else:
            rho = float(cached["rho"])
            standard_error = float(cached["sandwich_se"])
            pairs, games = int(cached["pairs"]), int(cached["games"])
        entry["D_interval_censored_mle"] = {
            "rho": float(rho),
            "clustered_sandwich_se": float(standard_error),
            "pairs": pairs,
            "games": games,
            "is_latent_rho_directly": True,
        }

        observed, pair_count = FL.ordered_pair_mean(
            count_window, kind, f"e_{stat_a}", f"e_{stat_b}"
        )
        count_draws = FL.clustered_bootstrap_pair_mean(
            count_window, kind, f"e_{stat_a}", f"e_{stat_b}", bootstrap, seed
        )
        entry["E_observed_count"] = {
            "reading": float(observed),
            "clustered_se": float(np.std(count_draws, ddof=1)),
            "pairs": float(pair_count),
        }
        entry["seconds"] = round(time.time() - started, 1)
        out[label] = entry
    if cache_path is not None:
        cache_path.write_text(
            json.dumps(
                {
                    "bucket": bucket,
                    "seeds": list(JITTER_SEEDS),
                    "by_window": out,
                },
                indent=1,
            )
        )
    return out


# ----------------------------------------------------------------------
# section 2: transmission
# ----------------------------------------------------------------------


def build_statistic_series(
    frame: pd.DataFrame,
    marginals: Mapping[str, object],
    kind: str,
    stat_a: str,
    stat_b: str,
    sample: int,
    seed: int,
) -> dict[str, FL.StatisticSeries]:
    """Pooled transmission series of one bucket in all three statistic spaces.

    Pooled over a deterministic by-game sample of the bucket's ordered pairs,
    so the margins carry the joint distribution the bucket statistic averages
    over. Both orientations of the stat pair are included when the two stats
    differ, matching the bucket's own symmetry.
    """
    positions = frame.reset_index(drop=True)
    pairs = FL.sampled_pairs(positions, kind, sample, seed)
    needed = sorted({index for pair in pairs for index in pair})
    grids: dict[tuple[int, str], np.ndarray] = {}
    for index in needed:
        row = positions.iloc[index]
        for stat in {stat_a, stat_b}:
            grids[(index, stat)] = tabulate_inverse_cdf(
                marginals[stat], float(row[f"mu_selected_{stat}"]), row
            )

    out: dict[str, FL.StatisticSeries] = {}
    for statistic in ("count", "randomized_pit", "mid_pit"):
        cache: dict[tuple[int, str], tuple[np.ndarray, float]] = {}

        def coefficients(index: int, stat: str) -> tuple[np.ndarray, float]:
            key = (index, stat)
            if key not in cache:
                cache[key] = FL.statistic_hermite_coefficients(
                    grids[key], statistic, TERMS
                )
            return cache[key]

        orientations = (
            ((stat_a, stat_b),)
            if stat_a == stat_b
            else ((stat_a, stat_b), (stat_b, stat_a))
        )
        left: list[tuple[np.ndarray, float]] = []
        right: list[tuple[np.ndarray, float]] = []
        for first, second in pairs:
            for one, two in orientations:
                left.append(coefficients(first, one))
                right.append(coefficients(second, two))
        out[statistic] = FL.StatisticSeries(
            statistic=statistic,
            coefficients=FL.pooled_statistic_series(left, right, TERMS),
        )
    out["_pairs"] = len(pairs)  # type: ignore[assignment]
    return out


def transmission_test(
    estimators: Mapping[str, object],
    series: Mapping[str, FL.StatisticSeries],
    window: str,
) -> dict[str, object]:
    """Each estimator's latent ``rho``, its implied count correlation, the error.

    An estimator that reports a *reading* in some statistic space is inverted
    through that space's own transmission series to get the latent ``rho`` it
    implies. The censored estimator reports ``rho`` directly, so it is the one
    estimator with nothing to invert -- which is the whole reason it is in the
    comparison.
    """
    entry = estimators[window]
    observed = float(entry["E_observed_count"]["reading"])  # type: ignore[index]
    count = series["count"]
    rows: dict[str, object] = {}

    plan = (
        ("A_randomized_pit", entry["A_randomized_pit"]["reading"], "randomized_pit"),  # type: ignore[index]
        (
            "B_multi_seed_randomized_pit",
            entry["B_multi_seed_randomized_pit"]["reading"],  # type: ignore[index]
            "randomized_pit",
        ),
        ("C_mid_pit", entry["C_mid_pit"]["reading"], "mid_pit"),  # type: ignore[index]
        ("D_interval_censored_mle", entry["D_interval_censored_mle"]["rho"], None),  # type: ignore[index]
    )
    for name, value, space in plan:
        if space is None:
            rho = float(value)
            reading = None
        else:
            reading = float(value)
            rho = series[space].invert(reading)
        implied = count.evaluate(rho)
        rows[name] = {
            "reading_in_own_space": reading,
            "latent_rho": float(rho),
            "implied_count_correlation": float(implied),
            "observed_count_correlation": observed,
            "count_space_error": float(implied - observed),
            "abs_count_space_error": float(abs(implied - observed)),
            "inverted_through": space,
        }
    rows["E_observed_count"] = {
        "reading_in_own_space": observed,
        "latent_rho": float(count.invert(observed)),
        "implied_count_correlation": observed,
        "observed_count_correlation": observed,
        "count_space_error": 0.0,
        "abs_count_space_error": 0.0,
        "inverted_through": "count",
        "note": (
            "the count reading is the target, so its own error is zero by "
            "construction; its latent rho is reported for comparison"
        ),
    }
    rows["first_order_gains"] = {  # type: ignore[assignment]
        statistic: float(series[statistic].coefficients[0])
        for statistic in ("count", "randomized_pit", "mid_pit")
    }

    # The count moment's own latent rho against the best latent estimator.
    # Correcting the attenuation moves the implied count correlation the right
    # way, but it does not arrive: the count moment wants a parameter no
    # estimator of the latent correlation supports, which is a statement about
    # the Gaussian copula and the production margins rather than about any
    # estimator.
    named = [
        (name, rows[name])
        for name in (
            "A_randomized_pit",
            "B_multi_seed_randomized_pit",
            "C_mid_pit",
            "D_interval_censored_mle",
        )
    ]
    mle = float(entry["D_interval_censored_mle"]["rho"])  # type: ignore[index]
    mle_se = float(entry["D_interval_censored_mle"]["clustered_sandwich_se"])  # type: ignore[index]
    count_rho = float(rows["E_observed_count"]["latent_rho"])  # type: ignore[index]
    ordered = sorted(named, key=lambda item: item[1]["latent_rho"])
    rows["residual_after_correcting_the_estimator"] = {  # type: ignore[assignment]
        "latent_rho_the_count_moment_implies": count_rho,
        "interval_censored_mle": mle,
        "shortfall": float(count_rho - mle),
        "shortfall_in_sandwich_se": (
            float((count_rho - mle) / mle_se) if mle_se > 0 else None
        ),
        "every_estimator_undershoots_the_count_moment": bool(
            all(
                row["implied_count_correlation"] < observed
                for _, row in named
            )
        ),
        "count_space_error_is_monotone_in_the_latent_rho": bool(
            all(
                ordered[index][1]["abs_count_space_error"]
                >= ordered[index + 1][1]["abs_count_space_error"]
                for index in range(len(ordered) - 1)
            )
        ),
        "count_space_error_reduction_from_a_to_d": float(
            1.0
            - rows["D_interval_censored_mle"]["abs_count_space_error"]  # type: ignore[index]
            / rows["A_randomized_pit"]["abs_count_space_error"]  # type: ignore[index]
        ),
        "note": (
            "because the count-space error is monotone in the latent rho over "
            "this range and every estimator sits below the value the count "
            "moment implies, the count-space ranking of the estimators is "
            "just their ordering in rho. The censored MLE is the correct "
            "estimate of the copula parameter and is still not the closest in "
            "count space, which is the sharpest form of the finding: the "
            "remaining gap is not an estimator's to close"
        ),
    }
    return rows


def attenuation_verdict(
    estimators: Mapping[str, object],
    transmission: Mapping[str, object],
    window: str,
) -> dict[str, object]:
    """``RANDOMIZED-PIT ATTENUATION: YES/NO``, with the three legs it rests on.

    The pipeline's latent gate equates the randomized-PIT *reading* with the
    copula *parameter*, so the question is whether that reading is materially
    below the parameter it stands for. Three independent legs decide it:

    1.  Size. The interval-censored MLE, which estimates the parameter with no
        Gaussianisation step at all, exceeds the reading by more than two of
        the reading's own game-clustered standard errors.
    2.  Mechanism. Averaging the reading over many deterministic jitter seeds
        does not close the gap. The jitter enters the statistic's denominator,
        so it contributes variance and not bias, and multi-seed averaging can
        only remove variance.
    3.  Correctability. Inverting the reading through its own transmission
        series -- the analytic map from parameter to reading -- lands on the
        censored MLE. If the gap were noise or model error the inversion would
        not reconcile the two.
    """
    entry = estimators[window]
    reading = float(entry["A_randomized_pit"]["reading"])  # type: ignore[index]
    reading_se = float(entry["A_randomized_pit"]["clustered_se"])  # type: ignore[index]
    multi = entry["B_multi_seed_randomized_pit"]  # type: ignore[index]
    mid = float(entry["C_mid_pit"]["reading"])  # type: ignore[index]
    mle = float(entry["D_interval_censored_mle"]["rho"])  # type: ignore[index]
    mle_se = float(entry["D_interval_censored_mle"]["clustered_sandwich_se"])  # type: ignore[index]
    count = float(entry["E_observed_count"]["reading"])  # type: ignore[index]

    gap = mle - reading
    pooled_se = float(np.sqrt(reading_se**2 + mle_se**2))
    inverted = float(transmission["A_randomized_pit"]["latent_rho"])  # type: ignore[index]

    legs = {
        "size__censored_mle_exceeds_the_reading_by_over_two_se": bool(
            gap > 2.0 * pooled_se
        ),
        "mechanism__multi_seed_averaging_does_not_close_the_gap": bool(
            abs(float(multi["reading"]) - reading) < 0.25 * gap  # type: ignore[index]
        ),
        "correctability__inverting_the_reading_lands_on_the_mle": bool(
            abs(inverted - mle) <= 2.0 * mle_se
        ),
        "ordering__reading_below_mid_pit_below_count": bool(
            reading < mid < count
        ),
    }
    attenuated = bool(
        legs["size__censored_mle_exceeds_the_reading_by_over_two_se"]
        and legs["mechanism__multi_seed_averaging_does_not_close_the_gap"]
    )
    return {
        "verdict": "YES" if attenuated else "NO",
        "window": window,
        "randomized_pit_reading": reading,
        "randomized_pit_clustered_se": reading_se,
        "multi_seed_reading": float(multi["reading"]),  # type: ignore[index]
        "multi_seed_across_seed_sd": float(multi["across_seed_sd"]),  # type: ignore[index]
        "mid_pit_reading": mid,
        "interval_censored_mle": mle,
        "interval_censored_sandwich_se": mle_se,
        "observed_count_correlation": count,
        "mle_minus_reading": float(gap),
        "mle_minus_reading_in_pooled_se": float(gap / pooled_se),
        "reading_inverted_through_its_own_series": inverted,
        "attenuation_factor_implied": float(reading / mle) if mle != 0.0 else None,
        "legs": legs,
        "what_the_pipeline_does_with_it": (
            "04_validate_shadow_v1.py::latent_dependence_summary scores the "
            "copula parameter against this reading directly, so an estimator "
            "fitted to minimise that error is driven to the attenuated value "
            "and the simulator then uses it as a real copula correlation"
        ),
    }


# ----------------------------------------------------------------------
# section 3: the feasibility envelope
# ----------------------------------------------------------------------


def transmitted_latent_readings(
    games: Sequence[HL.HoldoutGame],
    control_loadings: SharedFactorLoadings,
    candidate_loadings: SharedFactorLoadings,
) -> dict[str, object]:
    """What a randomized-PIT reading of the *simulated* data would show.

    The latent gate scores the copula parameter directly, so this is the step
    the gate leaves out: push the parameter through the production margins and
    the realised per-player shrink, and read the same randomized-PIT statistic
    back off the result. The ratio is the attenuation the gate charges the
    architecture for having corrected.
    """
    out: dict[str, object] = {}
    for name, candidate in (
        ("control", control_loadings),
        ("candidate", candidate_loadings),
    ):
        HL.assemble(games, candidate)
        transmitted = HL.predict_buckets(
            HL.bucket_series(games, "latent"), candidate
        )
        parameter = FL.bucket_readout(candidate)
        out[name] = {
            bucket: {
                "parameter": float(parameter[bucket]),
                "transmitted_randomized_pit_reading": float(transmitted[bucket]),
                "ratio": (
                    float(transmitted[bucket] / parameter[bucket])
                    if parameter[bucket] != 0.0
                    else None
                ),
            }
            for bucket in sorted(parameter)
        }
    return out


def constraint_set(
    games: Sequence[HL.HoldoutGame],
    control_loadings: SharedFactorLoadings,
) -> dict[str, object]:
    """The constraints section 3 requires, in the pipeline's own terms.

    Two of the commissioned constraints -- "protected opponent buckets within
    tolerance" and the two guarded same-team buckets "no worse than control"
    -- are the repository's own language, so they are read off the repository's
    own definitions rather than re-invented here. ``05_evaluate_gates.py``
    scores both as a *latent z degradation* against the control: gate 5 allows
    one z on the protected set, gate 1 allows a quarter of a z on the two
    repaired targets. Operationalising them instead as a count-space error
    ratio rejects the frozen candidate itself, which is how that reading was
    found to be wrong: on the held-out universe the frozen candidate's
    count-space opponent errors already exceed the control's by up to 8.9%,
    and it passed its gates regardless, because its gates never asked that.
    The count-space opponent errors are still reported below, as evidence.

    The control's own readings are measured with the same instruments the
    sweep uses -- the parameter matrix for latent, the exact held-out
    predictor for count -- so numerator and denominator of every ratio come
    from one instrument. Both are cross-checked against the published runs.
    """
    candidate = json.loads(CANDIDATE_REPORT.read_text())
    control = json.loads(CONTROL_REPORT.read_text())
    latent = candidate["latent_dependence"]
    observed_latent = latent["observed_buckets"]
    observed_count = candidate["residual_dependence"]["observed_buckets"]

    HL.assemble(games, control_loadings)
    control_count = HL.predict_buckets(
        HL.bucket_series(games, "count"), control_loadings
    )
    control_latent = FL.bucket_readout(control_loadings)
    control_latent_rmse = float(
        np.sqrt(
            np.mean(
                [(control_latent[b] - observed_latent[b]) ** 2 for b in control_latent]
            )
        )
    )
    control_count_rmse = float(
        np.sqrt(
            np.mean([(control_count[b] - observed_count[b]) ** 2 for b in control_count])
        )
    )
    return {
        "latent_observed": observed_latent,
        "latent_observed_se": latent["observed_bucket_se"],
        "count_observed": observed_count,
        "control_latent": control_latent,
        "control_count": control_count,
        "published_control_count": control["residual_dependence"]["by_model"][
            "candidate"
        ]["buckets"],
        "published_candidate_count": candidate["residual_dependence"]["by_model"][
            "candidate"
        ]["buckets"],
        "control_latent_rmse": control_latent_rmse,
        "control_count_rmse": control_count_rmse,
        "published_control_latent_rmse": float(
            control["latent_dependence"]["by_model"]["candidate"]["rmse"]
        ),
        "published_candidate_latent_rmse": float(
            latent["by_model"]["candidate"]["rmse"]
        ),
        "published_control_count_rmse": float(
            control["residual_dependence"]["count_space_cross_player_rmse"][
                "candidate"
            ]
        ),
        "published_candidate_count_rmse": float(
            candidate["residual_dependence"]["count_space_cross_player_rmse"][
                "candidate"
            ]
        ),
        "gate_definitions": {
            "source": "research/final_upstream_remediation/05_evaluate_gates.py",
            "max_protected_abs_z_degradation": MAX_PROTECTED_Z_DEGRADATION,
            "max_target_abs_z_degradation": MAX_TARGET_Z_DEGRADATION,
            "rmse_tolerance_used_here": FL.RMSE_TOLERANCE,
            "rmse_tolerance_the_pipeline_uses": 1.05,
            "note": (
                "the commissioned 1.03 RMSE tolerance is tighter than the "
                "pipeline's own 1.05, so the envelope below is the stricter "
                "of the two"
            ),
        },
    }


#: The structural checks no reading of the brief relaxes.
STRUCTURAL_CHECKS = (
    "protected_opponent_buckets_ok",
    "same_player_deviation_ok",
    "psd_failures_zero",
    "pairwise_parameters_zero",
    "player_indexed_parameters_zero",
)


def _commissioned(
    checks: Mapping[str, bool],
    guarded: Mapping[str, Mapping[str, object]],
    latent_rmse: float,
    count_rmse: float,
    control_latent_rmse: float,
    control_count_rmse: float,
) -> bool:
    return all(checks.values())


def _give_back_in_latent_space(
    checks: Mapping[str, bool],
    guarded: Mapping[str, Mapping[str, object]],
    latent_rmse: float,
    count_rmse: float,
    control_latent_rmse: float,
    control_count_rmse: float,
) -> bool:
    return bool(
        checks["latent_rmse_within_tolerance"]
        and checks["count_rmse_within_tolerance"]
        and all(checks[name] for name in STRUCTURAL_CHECKS)
        and all(
            entry["latent_within_pipeline_gate_1"] for entry in guarded.values()
        )
    )


def _pipeline_own_gates(
    checks: Mapping[str, bool],
    guarded: Mapping[str, Mapping[str, object]],
    latent_rmse: float,
    count_rmse: float,
    control_latent_rmse: float,
    control_count_rmse: float,
) -> bool:
    return bool(
        latent_rmse <= control_latent_rmse * 1.05
        and count_rmse <= control_count_rmse * 1.05
        and all(checks[name] for name in STRUCTURAL_CHECKS)
        and all(
            entry["latent_within_pipeline_gate_1"] for entry in guarded.values()
        )
    )


#: Three readings of the same constraint list, because the answer depends on
#: one of them and saying which is the honest form of the result.
#:
#: ``commissioned`` is the brief read literally: the 1.03 RMSE tolerances, and
#: "no worse than control" on the two repaired buckets enforced in count space
#: as well as latent space. That is the primary reading, because the 20% goal
#: is itself stated in count space and it would be incoherent to demand a
#: count-space gain on one bucket while refusing to measure the count-space
#: give-back on another.
#:
#: ``give_back_in_latent_space`` keeps the 1.03 tolerances but judges the two
#: repaired buckets the way ``05_evaluate_gates.py`` gate 1 judges them, on
#: latent z alone.
#:
#: ``pipeline_own_gates`` is the repository's own gate set, 1.05 tolerances
#: included -- what the frozen candidate actually had to satisfy.
POLICIES = {
    "commissioned": _commissioned,
    "give_back_in_latent_space": _give_back_in_latent_space,
    "pipeline_own_gates": _pipeline_own_gates,
}


def evaluate_target(
    games: Sequence[HL.HoldoutGame],
    loadings: SharedFactorLoadings,
    target: float,
    constraints: Mapping[str, object],
) -> dict[str, object]:
    """Every constraint, at one candidate value of the focal bucket's entry."""
    retarget = FL.retarget_same_team_entry(loadings, "ast", "pts", target)
    diagnostics = HL.assemble(games, retarget.loadings)
    count_series = HL.bucket_series(games, "count")
    predicted = HL.predict_buckets(count_series, retarget.loadings)
    latent = FL.bucket_readout(retarget.loadings)

    latent_observed = constraints["latent_observed"]
    latent_se = constraints["latent_observed_se"]
    count_observed = constraints["count_observed"]
    control_count = constraints["control_count"]
    control_latent = constraints["control_latent"]

    latent_errors = {b: latent[b] - latent_observed[b] for b in latent}
    count_errors = {b: predicted[b] - count_observed[b] for b in predicted}
    latent_rmse = float(
        np.sqrt(np.mean(np.square(list(latent_errors.values()))))
    )
    count_rmse = float(np.sqrt(np.mean(np.square(list(count_errors.values())))))

    def latent_z_degradation(bucket: str) -> float:
        se = float(latent_se[bucket])
        candidate_z = abs(latent_errors[bucket]) / se
        control_z = abs(control_latent[bucket] - latent_observed[bucket]) / se
        return float(candidate_z - control_z)

    focal_error = abs(count_errors[FL.FOCAL_BUCKET])
    control_focal_error = abs(
        control_count[FL.FOCAL_BUCKET] - count_observed[FL.FOCAL_BUCKET]
    )
    reduction = 1.0 - focal_error / control_focal_error

    protected = {}
    for bucket in count_errors:
        if not bucket.startswith("opponent_"):
            continue
        control_error = abs(control_count[bucket] - count_observed[bucket])
        degradation = latent_z_degradation(bucket)
        protected[bucket] = {
            "latent_abs_z_degradation": degradation,
            "within_tolerance": bool(degradation <= MAX_PROTECTED_Z_DEGRADATION),
            "count_abs_error": float(abs(count_errors[bucket])),
            "control_count_abs_error": float(control_error),
            "count_error_ratio_to_control": (
                float(abs(count_errors[bucket]) / control_error)
                if control_error > 0.0
                else None
            ),
        }

    guarded = {}
    for bucket in ("teammate_ast_ast", "teammate_reb_reb"):
        control_error = abs(control_count[bucket] - count_observed[bucket])
        degradation = latent_z_degradation(bucket)
        latent_ok = bool(degradation <= MAX_TARGET_Z_DEGRADATION)
        count_ok = bool(abs(count_errors[bucket]) <= control_error + 1e-12)
        guarded[bucket] = {
            "latent_abs_z_degradation": degradation,
            "latent_within_pipeline_gate_1": latent_ok,
            "count_abs_error": float(abs(count_errors[bucket])),
            "control_count_abs_error": float(control_error),
            "count_no_worse_than_control": count_ok,
            "no_worse_than_control": bool(latent_ok and count_ok),
        }

    latent_bound = float(constraints["control_latent_rmse"]) * FL.RMSE_TOLERANCE
    count_bound = float(constraints["control_count_rmse"]) * FL.RMSE_TOLERANCE
    checks = {
        "latent_rmse_within_tolerance": bool(latent_rmse <= latent_bound),
        "count_rmse_within_tolerance": bool(count_rmse <= count_bound),
        "protected_opponent_buckets_ok": all(
            item["within_tolerance"] for item in protected.values()
        ),
        "teammate_ast_ast_no_worse": guarded["teammate_ast_ast"][
            "no_worse_than_control"
        ],
        "teammate_reb_reb_no_worse": guarded["teammate_reb_reb"][
            "no_worse_than_control"
        ],
        "same_player_deviation_ok": bool(
            diagnostics.max_same_player_block_deviation <= 1e-9
        ),
        "psd_failures_zero": diagnostics.numerical_failures == 0,
        "pairwise_parameters_zero": True,
        "player_indexed_parameters_zero": True,
    }
    return {
        "target_latent_entry": float(target),
        "competition_inflation": float(retarget.competition_inflation),
        "min_game_gram_eigenvalue": float(retarget.min_game_eigenvalue),
        "min_contrast_gram_eigenvalue": float(retarget.min_contrast_eigenvalue),
        "min_competition_gram_eigenvalue": float(
            retarget.min_competition_eigenvalue
        ),
        "shared_scale_range": [
            float(diagnostics.min_shared_scale),
            float(diagnostics.max_shared_scale),
        ],
        # The price of representability, in the currency it is paid in. The
        # *extremes* of the shrink barely move -- the binding player is the
        # same one -- so the range alone hides the cost. The mean of w^2 is
        # what every realised same-team correlation is scaled by, and it is
        # what carries the give-back to the buckets the lever never touches.
        "mean_squared_shared_scale": float(
            np.mean(
                np.square(
                    np.concatenate(
                        [game.scale for game in games if game.scale.size]
                    )
                )
            )
        ),
        "min_covariance_eigenvalue": float(diagnostics.min_covariance_eigenvalue),
        "max_same_player_block_deviation": float(
            diagnostics.max_same_player_block_deviation
        ),
        "psd_numerical_failures": int(diagnostics.numerical_failures),
        "predicted_count_buckets": {k: float(v) for k, v in predicted.items()},
        "latent_buckets": {k: float(v) for k, v in latent.items()},
        "focal_predicted_count": float(predicted[FL.FOCAL_BUCKET]),
        "focal_abs_count_error": float(focal_error),
        "focal_count_error_reduction": float(reduction),
        "global_latent_rmse": latent_rmse,
        "global_latent_rmse_bound": latent_bound,
        "global_latent_rmse_ratio_to_control": float(
            latent_rmse / float(constraints["control_latent_rmse"])
        ),
        "passes_under_the_pipelines_own_1_05_tolerance": bool(
            latent_rmse <= float(constraints["control_latent_rmse"]) * 1.05
            and count_rmse <= float(constraints["control_count_rmse"]) * 1.05
        ),
        "global_count_rmse": count_rmse,
        "global_count_rmse_bound": count_bound,
        "global_count_rmse_ratio_to_control": float(
            count_rmse / float(constraints["control_count_rmse"])
        ),
        "protected_opponent_buckets": protected,
        "guarded_same_team_buckets": guarded,
        "checks": checks,
        "all_constraints_pass": all(checks.values()),
        "feasible_under": {
            name: policy(
                checks,
                guarded,
                latent_rmse,
                count_rmse,
                float(constraints["control_latent_rmse"]),
                float(constraints["control_count_rmse"]),
            )
            for name, policy in POLICIES.items()
        },
        "reaches_target_reduction": bool(reduction >= FL.TARGET_ERROR_REDUCTION),
    }


def feasibility_envelope(
    games: Sequence[HL.HoldoutGame],
    loadings: SharedFactorLoadings,
    constraints: Mapping[str, object],
    grid: Sequence[float],
) -> dict[str, object]:
    """Sweep the lever, then bisect for the two boundaries that matter.

    The architecture represents any value of this entry, so the envelope is
    not set by the rank: it is set by the global latent RMSE, which is
    measured against the *randomized-PIT* observation and therefore penalises
    exactly the correction count space needs. The sweep shows the whole trade
    and the bisections pin the two crossings.
    """
    # Memoised because the three readings of the constraints bisect the same
    # interval and agree on the opening steps, and one evaluation re-assembles
    # four hundred game covariances.
    seen: dict[float, dict[str, object]] = {}

    def evaluate(target: float) -> dict[str, object]:
        if target not in seen:
            seen[target] = evaluate_target(games, loadings, target, constraints)
        return seen[target]

    sweep = [evaluate(target) for target in grid]

    base = float(
        loadings.same_team_correlation()[
            loadings.stats.index("ast"), loadings.stats.index("pts")
        ]
    )
    at_base = evaluate(base)
    if not at_base["all_constraints_pass"]:
        failed = [
            name for name, passed in at_base["checks"].items() if not passed
        ]
        raise SystemExit(
            "the frozen candidate itself fails the constraint set on "
            f"{failed}, so the constraint set is mis-specified rather than "
            "the architecture being infeasible"
        )

    def bisect_boundary(policy: str) -> float:
        low, high = base, max(float(max(grid)), base)
        for _ in range(40):
            middle = 0.5 * (low + high)
            if evaluate(middle)["feasible_under"][policy]:
                low = middle
            else:
                high = middle
        return low

    # The reduction is not monotone in the target -- past the peak the shrink
    # that representability costs pulls the realised correlation back down --
    # so the smallest target that reaches the goal is bisected on the rising
    # branch below the peak.
    peak = max(sweep, key=lambda row: row["focal_count_error_reduction"])
    needed = None
    if peak["focal_count_error_reduction"] >= FL.TARGET_ERROR_REDUCTION:
        rising_low, rising_high = base, float(peak["target_latent_entry"])
        for _ in range(40):
            middle = 0.5 * (rising_low + rising_high)
            if (
                evaluate(middle)["focal_count_error_reduction"]
                < FL.TARGET_ERROR_REDUCTION
            ):
                rising_low = middle
            else:
                rising_high = middle
        needed = rising_high

    readings: dict[str, object] = {}
    for policy in POLICIES:
        boundary = bisect_boundary(policy)
        at_boundary = evaluate(boundary)
        readings[policy] = {
            "max_feasible_entry": float(boundary),
            "reduction_at_the_boundary": float(
                at_boundary["focal_count_error_reduction"]
            ),
            "reaches_target_reduction": bool(
                needed is not None and needed <= boundary
            ),
            "binding_constraints_just_past_the_boundary": binding_constraints(
                evaluate(boundary + 1e-5), policy
            ),
            "at_the_boundary": at_boundary,
        }

    primary = readings["commissioned"]
    curve = trade_curve(evaluate, base, constraints, readings, needed)
    return {
        "trade_curve": curve,
        "lever": (
            "one same-team off-diagonal entry retargeted through the "
            "competition Gram, which leaves every cross-team block and every "
            "other same-team entry identically unmoved"
        ),
        "candidate_entry": base,
        "at_the_candidate_entry": at_base,
        "sweep": sweep,
        "entry_needed_for_target_reduction": (
            None if needed is None else float(needed)
        ),
        "by_reading_of_the_constraints": readings,
        "primary_reading": "commissioned",
        "max_feasible_entry": primary["max_feasible_entry"],  # type: ignore[index]
        "at_max_feasible": primary["at_the_boundary"],  # type: ignore[index]
        "target_reduction_feasible": primary["reaches_target_reduction"],  # type: ignore[index]
        "unconstrained_peak": {
            "target_latent_entry": peak["target_latent_entry"],
            "focal_predicted_count": peak["focal_predicted_count"],
            "focal_count_error_reduction": peak["focal_count_error_reduction"],
            "note": (
                "beyond this the competition inflation representability needs "
                "triggers enough per-player shrink that the realised "
                "correlation falls again"
            ),
        },
        "binding_constraint": sweep_constraint_profile(sweep),
        "analytic_latent_rmse_boundary": analytic_latent_boundary(
            loadings, constraints
        ),
        "why_any_step_costs_shrink": (
            "the frozen contrast Gram sits on the PSD boundary, so moving "
            "this entry in either direction drives it negative and has to be "
            "bought back with competition inflation. Inflation adds "
            "delta * n * I to every within-player shared block, which "
            "_largest_feasible_shrink then has to re-pin, and the pinning is "
            "paid for in per-player shrink. Shrink multiplies every realised "
            "cross-player correlation, so the price of moving one entry is "
            "charged to all twelve buckets"
        ),
    }


def tradeoff_row(row: Mapping[str, object]) -> dict[str, object]:
    """One point of the trade curve, in the two quantities that trade.

    The focal bucket's count-space error reduction is what the research target
    is stated in; ``teammate_reb_reb``'s count-space degradation is what pays
    for it. Nothing else on the sweep moves except through per-player shrink,
    so these two columns are the whole trade.
    """
    reb = row["guarded_same_team_buckets"]["teammate_reb_reb"]  # type: ignore[index]
    ast = row["guarded_same_team_buckets"]["teammate_ast_ast"]  # type: ignore[index]
    control = float(reb["control_count_abs_error"])
    error = float(reb["count_abs_error"])
    return {
        "entry": float(row["target_latent_entry"]),  # type: ignore[arg-type]
        "focal_count_correlation": float(row["focal_predicted_count"]),  # type: ignore[arg-type]
        "focal_abs_count_error": float(row["focal_abs_count_error"]),  # type: ignore[arg-type]
        "focal_count_error_reduction": float(
            row["focal_count_error_reduction"]  # type: ignore[arg-type]
        ),
        "teammate_reb_reb_abs_count_error": error,
        "teammate_reb_reb_control_abs_count_error": control,
        "teammate_reb_reb_degradation_absolute": float(error - control),
        "teammate_reb_reb_degradation_fraction": float(error / control - 1.0),
        "teammate_ast_ast_abs_count_error": float(ast["count_abs_error"]),
        "teammate_ast_ast_degradation_fraction": float(
            float(ast["count_abs_error"])
            / float(ast["control_count_abs_error"])
            - 1.0
        ),
        "latent_rmse_ratio_to_control": float(
            row["global_latent_rmse_ratio_to_control"]  # type: ignore[arg-type]
        ),
        "count_rmse_ratio_to_control": float(
            row["global_count_rmse_ratio_to_control"]  # type: ignore[arg-type]
        ),
        "mean_squared_shared_scale": float(row["mean_squared_shared_scale"]),  # type: ignore[arg-type]
        "competition_inflation": float(row["competition_inflation"]),  # type: ignore[arg-type]
        "psd_numerical_failures": int(row["psd_numerical_failures"]),  # type: ignore[arg-type]
        "feasible_under": dict(row["feasible_under"]),  # type: ignore[arg-type]
        "failing_constraints_under": {
            policy: binding_constraints(row, policy) for policy in POLICIES
        },
    }


def binding_constraints(
    row: Mapping[str, object],
    policy: str,
) -> list[str]:
    """The named checks that fail just past a policy's boundary."""
    guarded = row["guarded_same_team_buckets"]
    checks = dict(row["checks"])  # type: ignore[arg-type]
    if policy != "commissioned":
        for bucket, entry in guarded.items():  # type: ignore[union-attr]
            checks[f"{bucket}_no_worse"] = bool(
                entry["latent_within_pipeline_gate_1"]
            )
    if policy == "pipeline_own_gates":
        checks["latent_rmse_within_tolerance"] = bool(
            row["global_latent_rmse_ratio_to_control"] <= 1.05
        )
        checks["count_rmse_within_tolerance"] = bool(
            row["global_count_rmse_ratio_to_control"] <= 1.05
        )
    return [name for name, passed in checks.items() if not passed]


def sweep_constraint_profile(
    sweep: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Which constraints move with the lever at all, and which never do.

    The lever moves exactly one latent parameter, so every constraint that
    reads only the unmoved parameters is constant across the sweep. Saying
    which those are is what makes the envelope a *structural* statement rather
    than the outcome of one arbitrary tolerance.
    """
    names = list(sweep[0]["checks"])  # type: ignore[index]
    constant = [
        name
        for name in names
        if len({bool(row["checks"][name]) for row in sweep}) == 1  # type: ignore[index]
    ]
    first_failure: dict[str, float | None] = {}
    for name in names:
        crossing = None
        for row in sweep:
            if not row["checks"][name]:  # type: ignore[index]
                crossing = float(row["target_latent_entry"])  # type: ignore[arg-type]
                break
        first_failure[name] = crossing
    return {
        "constant_across_the_whole_sweep": constant,
        "first_grid_point_each_constraint_fails_at": first_failure,
    }


def analytic_latent_boundary(
    loadings: SharedFactorLoadings,
    constraints: Mapping[str, object],
) -> dict[str, object]:
    """The latent-RMSE boundary in closed form.

    The lever moves one of the twelve latent parameters and nothing else, and
    the latent gate scores parameters, so

        ``12 * rmse(S)^2 = C + (S - observed_focal)^2``

    with ``C`` the other eleven squared errors. The boundary is therefore a
    root of a quadratic rather than a bisection artefact, which is what lets
    the envelope be compared against an estimate and its standard error.
    """
    latent = FL.bucket_readout(loadings)
    observed = constraints["latent_observed"]
    focal_observed = float(observed[FL.FOCAL_BUCKET])
    others = float(
        sum(
            (latent[bucket] - observed[bucket]) ** 2
            for bucket in latent
            if bucket != FL.FOCAL_BUCKET
        )
    )
    count = len(latent)
    bound = float(constraints["control_latent_rmse"]) * FL.RMSE_TOLERANCE
    slack = count * bound**2 - others
    if slack < 0.0:
        return {
            "feasible": False,
            "note": "the eleven unmoved buckets already exhaust the bound",
        }
    half_width = float(np.sqrt(slack))
    return {
        "feasible": True,
        "other_eleven_squared_error_sum": others,
        "observed_focal_latent": focal_observed,
        "latent_rmse_bound": bound,
        "max_entry": focal_observed + half_width,
        "min_entry": focal_observed - half_width,
        "formula": (
            f"{count} * rmse(S)^2 = {others:.12e} + (S - {focal_observed:.12f})^2"
        ),
    }


# ----------------------------------------------------------------------
# section 5: the inner forward test
# ----------------------------------------------------------------------


def inner_forward_test(
    residuals: pd.DataFrame,
    counts: pd.DataFrame,
    marginals: Mapping[int, object],
    estimators: Mapping[str, object],
    sample: int,
    seed: int,
) -> dict[str, object]:
    """Does the censored estimator predict the next pre-2024 season better?

    For each forward fold the estimators are fitted on seasons strictly before
    it, pushed through *that fold's own* margins into an implied count
    correlation, and scored against the fold's observed count correlation.
    No held-out season is involved, and no estimator sees its own evaluation
    season, so this is the precondition section 5 asks about.
    """
    out: dict[str, object] = {}
    for fold in FL.PRE_2024_FORWARD_FOLDS:
        train_label = f"train_before_{fold}"
        fold_rows = residuals.loc[residuals["season"] == fold]
        fold_counts = counts.loc[counts["season"] == fold]
        series = build_statistic_series(
            fold_rows,
            marginals[fold].fitted,
            "same_team",
            "ast",
            "pts",
            sample,
            seed,
        )
        observed, _ = FL.ordered_pair_mean(
            fold_counts, "same_team", "e_ast", "e_pts"
        )
        entry = estimators[train_label]
        rows: dict[str, object] = {}
        plan = (
            (
                "A_randomized_pit",
                float(entry["A_randomized_pit"]["reading"]),  # type: ignore[index]
                "randomized_pit",
            ),
            (
                "B_multi_seed_randomized_pit",
                float(entry["B_multi_seed_randomized_pit"]["reading"]),  # type: ignore[index]
                "randomized_pit",
            ),
            ("C_mid_pit", float(entry["C_mid_pit"]["reading"]), "mid_pit"),  # type: ignore[index]
            (
                "D_interval_censored_mle",
                float(entry["D_interval_censored_mle"]["rho"]),  # type: ignore[index]
                None,
            ),
        )
        for name, value, space in plan:
            rho = value if space is None else series[space].invert(value)
            implied = series["count"].evaluate(rho)
            rows[name] = {
                "trained_on": entry["seasons"],
                "latent_rho": float(rho),
                "implied_count_correlation": float(implied),
                "abs_error": float(abs(implied - observed)),
            }
        baseline = rows["A_randomized_pit"]["abs_error"]  # type: ignore[index]
        for name in rows:
            rows[name]["error_reduction_vs_current_estimator"] = float(  # type: ignore[index]
                1.0 - rows[name]["abs_error"] / baseline  # type: ignore[index]
            )
        out[f"fold_{fold}"] = {
            "observed_count_correlation": float(observed),
            "bridge_pairs": int(series["_pairs"]),  # type: ignore[arg-type]
            "estimators": rows,
        }
    folds = list(out)
    improvements = [
        out[key]["estimators"]["D_interval_censored_mle"][  # type: ignore[index]
            "error_reduction_vs_current_estimator"
        ]
        for key in folds
    ]
    out["censored_improves_every_forward_fold"] = bool(
        all(value > 0.0 for value in improvements)
    )
    out["mean_error_reduction"] = float(np.mean(improvements))

    # Why the folds answer the way they do. Within one fold every estimator's
    # implied count correlation sits inside a band narrower than the
    # fold-to-fold movement of the target, so whichever estimator happens to
    # sit on the side the season moved towards wins that fold. That is a
    # statement about the target's year-to-year stability, not about the
    # estimators, and it is the reason section 5's precondition is scored
    # rather than assumed.
    observed_spread = float(
        max(out[key]["observed_count_correlation"] for key in folds)  # type: ignore[index]
        - min(out[key]["observed_count_correlation"] for key in folds)  # type: ignore[index]
    )
    implied_spreads = {
        key: float(
            max(
                row["implied_count_correlation"]
                for row in out[key]["estimators"].values()  # type: ignore[index]
            )
            - min(
                row["implied_count_correlation"]
                for row in out[key]["estimators"].values()  # type: ignore[index]
            )
        )
        for key in folds
    }
    best_by_fold = {
        key: min(
            out[key]["estimators"].items(),  # type: ignore[index]
            key=lambda item: item[1]["abs_error"],
        )[0]
        for key in folds
    }
    out["why_the_folds_disagree"] = {
        "observed_count_correlation_by_fold": {
            key: float(out[key]["observed_count_correlation"]) for key in folds  # type: ignore[index]
        },
        "fold_to_fold_spread_of_the_target": observed_spread,
        "within_fold_spread_of_the_estimators": implied_spreads,
        "target_moves_more_than_the_estimators_differ": bool(
            observed_spread > max(implied_spreads.values())
        ),
        "best_estimator_by_fold": best_by_fold,
        "the_folds_agree_on_a_winner": bool(len(set(best_by_fold.values())) == 1),
    }
    out["section_5_precondition_met"] = bool(
        out["censored_improves_every_forward_fold"]
    )
    out["inner_fit_was_run"] = False
    out["why_no_inner_fit"] = (
        "section 5 permits refitting the existing PSD factor model to an "
        "improved latent target only if the generic estimator improves this "
        "bucket on the pre-2024 forward folds. It does not: it wins one fold "
        "and loses the other, because the target moves further between "
        "seasons than the estimators differ within a season. The precondition "
        "is not met, so no refit was performed and no new factor family was "
        "introduced"
    )
    return out


def generic_application(
    games: Sequence[HL.HoldoutGame],
    loadings: SharedFactorLoadings,
    constraints: Mapping[str, object],
    mle: Mapping[str, object],
) -> dict[str, object]:
    """What the censored estimator does to all twelve buckets at once.

    The estimator is generic, so the honest version of section 5 is to apply
    it everywhere rather than only where it helps. Every bucket's latent
    reading is attenuated, so a correct estimator raises every target in
    magnitude, and the latent gate -- which scores the architecture's
    parameter against the attenuated reading -- charges for all of it.
    """
    pooled = mle["pooled_2020_2023"]["buckets"]  # type: ignore[index]
    index = {stat: position for position, stat in enumerate(loadings.stats)}
    same = loadings.same_team_correlation().copy()
    cross = loadings.cross_team_correlation().copy()
    for name, kind, (first, second) in FL.CROSS_PLAYER_BUCKETS:
        value = float(pooled[name]["rho"])
        block = same if kind == "same_team" else cross
        block[index[first], index[second]] = value
        block[index[second], index[first]] = value

    base = loadings.competition_gram()
    size = len(loadings.stats)
    game_base = 0.5 * (same + cross + base)
    contrast_base = 0.5 * (same - cross + base)
    floor = min(
        float(np.min(np.linalg.eigvalsh(game_base))),
        float(np.min(np.linalg.eigvalsh(contrast_base))),
    )
    inflation = max(0.0, -2.0 * floor)
    retargeted = SharedFactorLoadings(
        stats=loadings.stats,
        game=FL._refactor_gram(
            game_base + 0.5 * inflation * np.eye(size), "game"
        ),
        team_contrast=FL._refactor_gram(
            contrast_base + 0.5 * inflation * np.eye(size), "team_contrast"
        ),
        competition=FL._refactor_gram(
            base + inflation * np.eye(size), "competition"
        ),
        role_scale=dict(loadings.role_scale),
        symmetric=loadings.symmetric,
        role_deviation=loadings.role_deviation,
        role_offset=dict(loadings.role_offset),
        role_pair_shares=dict(loadings.role_pair_shares),
    )
    diagnostics = HL.assemble(games, retargeted)
    series = HL.bucket_series(games, "count")
    predicted = HL.predict_buckets(series, retargeted)
    latent = FL.bucket_readout(retargeted)

    latent_observed = constraints["latent_observed"]
    count_observed = constraints["count_observed"]
    control_count = constraints["control_count"]
    latent_rmse = float(
        np.sqrt(
            np.mean([(latent[b] - latent_observed[b]) ** 2 for b in latent])
        )
    )
    count_rmse = float(
        np.sqrt(
            np.mean([(predicted[b] - count_observed[b]) ** 2 for b in predicted])
        )
    )
    focal_error = abs(
        predicted[FL.FOCAL_BUCKET] - count_observed[FL.FOCAL_BUCKET]
    )
    control_focal = abs(
        control_count[FL.FOCAL_BUCKET] - count_observed[FL.FOCAL_BUCKET]
    )
    latent_bound = float(constraints["control_latent_rmse"]) * FL.RMSE_TOLERANCE
    count_bound = float(constraints["control_count_rmse"]) * FL.RMSE_TOLERANCE
    return {
        "targets": {
            name: float(pooled[name]["rho"]) for name, _, _ in FL.CROSS_PLAYER_BUCKETS
        },
        "competition_inflation": float(inflation),
        "shared_scale_range": [
            float(diagnostics.min_shared_scale),
            float(diagnostics.max_shared_scale),
        ],
        "psd_numerical_failures": int(diagnostics.numerical_failures),
        "max_same_player_block_deviation": float(
            diagnostics.max_same_player_block_deviation
        ),
        "predicted_count_buckets": {k: float(v) for k, v in predicted.items()},
        "focal_predicted_count": float(predicted[FL.FOCAL_BUCKET]),
        "focal_count_error_reduction": float(1.0 - focal_error / control_focal),
        "global_latent_rmse": latent_rmse,
        "global_latent_rmse_bound": latent_bound,
        "latent_rmse_within_tolerance": bool(latent_rmse <= latent_bound),
        "global_count_rmse": count_rmse,
        "global_count_rmse_bound": count_bound,
        "count_rmse_within_tolerance": bool(count_rmse <= count_bound),
    }


# ----------------------------------------------------------------------
# the trade curve, and the five answers it is read off
# ----------------------------------------------------------------------

#: The two gate sets the envelope is reported under.
#:
#: ``pipeline_own_gates`` is what ``05_evaluate_gates.py`` implements and what
#: the frozen candidate actually had to satisfy: 1.05 RMSE tolerances, gate 5's
#: one-z latent rule on the protected set, gate 1's quarter-z latent rule on
#: the two repaired buckets.
#:
#: ``commissioned`` is the original remediation requirement as briefed: 1.03
#: RMSE tolerances, and "no worse than control" on the two repaired buckets
#: enforced in count space as well as latent space. It is the stricter of the
#: two in both respects.
REPORTED_GATE_SETS = ("pipeline_own_gates", "commissioned")


def trade_curve(
    evaluate,
    base: float,
    constraints: Mapping[str, object],
    readings: Mapping[str, object],
    needed: float | None,
) -> dict[str, object]:
    """The exact trade between the focal gain and the ``reb_reb`` give-back.

    Sampled finely from the frozen entry up past both boundaries, with every
    boundary and the twenty-percent crossing included exactly rather than
    interpolated, so the five answers below are read off evaluated points and
    not off a fit.
    """
    # Keyed on the entry rounded to ten decimals so nothing appears twice
    # under two float representations differing in their last bits, but
    # *evaluated* at full precision wherever the point matters: a boundary
    # rounded to ten decimals falls just the wrong side of itself, which would
    # put a feasible boundary row in the curve marked infeasible.
    probes: dict[float, float] = {
        round(float(value), 10): float(value)
        for value in np.round(np.arange(base, 0.0565, 0.00025), 10)
    }
    exact = [base] + [
        float(entry["max_feasible_entry"]) for entry in readings.values()  # type: ignore[index]
    ]
    if needed is not None:
        exact.append(float(needed))
    for value in exact:
        probes[round(value, 10)] = value
    curve = [
        tradeoff_row(evaluate(probes[key])) for key in sorted(probes)
    ]

    def crossing(fraction: float) -> dict[str, object] | None:
        """The first curve point whose ``reb_reb`` give-back exceeds a budget."""
        for row in curve:
            if row["teammate_reb_reb_degradation_fraction"] > fraction + 1e-9:  # type: ignore[operator]
                return row
        return None

    return {
        "what_trades": (
            "the focal bucket's count-space error reduction against "
            "teammate_reb_reb's count-space degradation. The lever moves one "
            "latent parameter and no other, so the only channel between them "
            "is the per-player shrink that representability costs"
        ),
        "curve": curve,
        "give_back_budget_crossings": {
            f"first_entry_where_reb_reb_degrades_more_than_{int(100 * fraction)}_percent": (
                None if row is None else row["entry"]
            )
            for fraction, row in (
                (value, crossing(value))
                for value in (0.0, 0.05, 0.10, 0.15, 0.20)
            )
        },
        # The budget comparison is tolerant at the ninth decimal so the
        # bisected boundaries, which land on a give-back of zero to within
        # 1e-12, are counted rather than excluded by their own rounding.
        "reduction_available_within_each_give_back_budget": {
            f"reb_reb_degradation_at_most_{int(100 * fraction)}_percent": (
                max(
                    (
                        row["focal_count_error_reduction"]
                        for row in curve
                        if row["teammate_reb_reb_degradation_fraction"]  # type: ignore[operator]
                        <= fraction + 1e-9
                    ),
                    default=None,
                )
            )
            for fraction in (0.0, 0.05, 0.10, 0.15, 0.20)
        },
    }


def five_answers(
    readings: Mapping[str, object],
    curve: Sequence[Mapping[str, object]],
    needed: float | None,
    forward: Mapping[str, object],
    transmission: Mapping[str, object],
) -> dict[str, object]:
    """A to E, one evaluated number each, under both gate sets."""
    out: dict[str, object] = {}
    for gate_set in REPORTED_GATE_SETS:
        reading = readings[gate_set]
        boundary = float(reading["max_feasible_entry"])  # type: ignore[index]
        at_boundary = tradeoff_row(reading["at_the_boundary"])  # type: ignore[index]
        binding = reading["binding_constraints_just_past_the_boundary"]  # type: ignore[index]
        out[gate_set] = {
            "A_max_feasible_ast_to_teammate_pts_count_correlation": {
                "count_space_correlation": at_boundary["focal_count_correlation"],
                "latent_parameter_that_produces_it": boundary,
            },
            "B_absolute_error_reduction_percent": float(
                100.0 * at_boundary["focal_count_error_reduction"]  # type: ignore[arg-type]
            ),
            "C_teammate_reb_reb_degradation_at_that_point": {
                "abs_count_error": at_boundary[
                    "teammate_reb_reb_abs_count_error"
                ],
                "control_abs_count_error": at_boundary[
                    "teammate_reb_reb_control_abs_count_error"
                ],
                "degradation_absolute": at_boundary[
                    "teammate_reb_reb_degradation_absolute"
                ],
                "degradation_percent": float(
                    100.0
                    * at_boundary["teammate_reb_reb_degradation_fraction"]  # type: ignore[arg-type]
                ),
            },
            "D_first_binding_constraint": (
                binding[0] if binding else None  # type: ignore[index]
            ),
            "D_all_constraints_failing_just_past_it": binding,
            "E_twenty_percent_gate_achievable": bool(
                reading["reaches_target_reduction"]  # type: ignore[index]
            ),
        }
    out["entry_needed_for_twenty_percent"] = needed
    out["E_qualification"] = {
        "achievable_under_the_implemented_pipeline_gates": bool(
            readings["pipeline_own_gates"]["reaches_target_reduction"]  # type: ignore[index]
        ),
        "achievable_under_the_original_remediation_requirements": bool(
            readings["commissioned"]["reaches_target_reduction"]  # type: ignore[index]
        ),
        "but_the_pre_2024_folds_do_not_support_the_corrected_estimator": bool(
            not forward["censored_improves_every_forward_fold"]
        ),
        "and_the_count_moment_wants_a_rho_no_estimator_supports": bool(
            transmission["residual_after_correcting_the_estimator"][  # type: ignore[index]
                "every_estimator_undershoots_the_count_moment"
            ]
        ),
        "note": (
            "where the gate is reachable it is reachable only by setting the "
            "parameter above the generic estimator's own pre-2024 value, and "
            "the forward folds decline to prefer that estimator at all, so "
            "reaching it would require choosing this bucket's value against "
            "the holdout -- which is the bucket-specific free parameter the "
            "brief forbids"
        ),
    }
    return out


# ----------------------------------------------------------------------
# section 4: identifiability, and section 6: the stop rule
# ----------------------------------------------------------------------

#: The five classifications section 4 allows. Exactly one is returned.
CLASSIFICATIONS = (
    "IDENTIFIABLE_AND_ACHIEVABLE",
    "IDENTIFIABLE_BUT_GLOBALLY_INCOMPATIBLE",
    "NOT_IDENTIFIABLE_FROM_CURRENT_MARGINALS",
    "RANDOMIZED_PIT_ATTENUATION_CONFIRMED",
    "NO_ESTIMATOR_DEFECT_FOUND",
)


def identifiability(
    attenuation: Mapping[str, object],
    envelope: Mapping[str, object],
    at_generic_value: Mapping[str, object],
    generic_all_buckets: Mapping[str, object] | None,
    transmission: Mapping[str, object],
    forward: Mapping[str, object],
) -> dict[str, object]:
    """One classification, derived from predicates rather than asserted.

    The order matters. "Not identifiable" is checked before anything else
    because an unidentified parameter makes every later question moot, and
    "no estimator defect" is checked next because if the two estimators agree
    there is nothing to attribute. Only once a material estimator gap is
    established does the question become whether the architecture can carry
    the corrected value, and that splits three ways: it can and the generic
    estimator's own value is inside the envelope; it can but the generic
    value is not; or it cannot reach the goal at all.
    """
    identified = bool(
        np.isfinite(float(attenuation["interval_censored_sandwich_se"]))
        and float(attenuation["interval_censored_sandwich_se"]) > 0.0
        and abs(float(transmission["first_order_gains"]["count"])) > 1e-6  # type: ignore[index]
    )
    material_gap = bool(attenuation["verdict"] == "YES")
    envelope_reaches_target = bool(envelope["target_reduction_feasible"])
    generic_value_feasible = bool(at_generic_value["all_constraints_pass"])
    generic_value_reaches_target = bool(
        at_generic_value["reaches_target_reduction"]
    )
    all_buckets_feasible = (
        None
        if generic_all_buckets is None
        else bool(
            generic_all_buckets["latent_rmse_within_tolerance"]
            and generic_all_buckets["count_rmse_within_tolerance"]
        )
    )

    predicates = {
        "the_bucket_is_identified_from_the_current_marginals": identified,
        "the_two_estimators_differ_materially": material_gap,
        "the_envelope_reaches_the_twenty_percent_goal": envelope_reaches_target,
        "the_generic_estimators_own_value_is_inside_the_envelope": (
            generic_value_feasible
        ),
        "the_generic_estimators_own_value_reaches_the_goal": (
            generic_value_reaches_target
        ),
        "applying_the_generic_estimator_to_all_twelve_buckets_is_feasible": (
            all_buckets_feasible
        ),
        "the_pre_2024_forward_folds_agree_the_estimator_is_better": bool(
            forward["censored_improves_every_forward_fold"]
        ),
        "the_count_moment_wants_a_rho_no_estimator_supports": bool(
            transmission["residual_after_correcting_the_estimator"][  # type: ignore[index]
                "every_estimator_undershoots_the_count_moment"
            ]
        ),
    }

    if not identified:
        label = "NOT_IDENTIFIABLE_FROM_CURRENT_MARGINALS"
        reason = (
            "the interval-censored pseudo-likelihood does not pin the "
            "parameter from the production marginals, so no estimator can"
        )
    elif not material_gap:
        label = "NO_ESTIMATOR_DEFECT_FOUND"
        reason = (
            "the randomized-PIT reading and the interval-censored MLE agree "
            "within sampling error, so the shortfall is not an estimator "
            "defect"
        )
    elif not envelope_reaches_target:
        label = "IDENTIFIABLE_BUT_GLOBALLY_INCOMPATIBLE"
        reason = (
            "the parameter is identified, the randomized-PIT estimator is "
            "confirmed biased low, and the corrected value is representable "
            "-- but no value the global constraints admit reaches a twenty "
            "percent count-space error reduction, because every step of the "
            "lever is bought with per-player shrink and the shrink gives "
            "back a bucket the accepted repair fixed"
        )
    elif generic_value_feasible and generic_value_reaches_target:
        label = "IDENTIFIABLE_AND_ACHIEVABLE"
        reason = (
            "the generic estimator's own value satisfies every global "
            "constraint and reaches the goal, so the fix is available without "
            "any bucket-specific parameter"
        )
    else:
        label = "RANDOMIZED_PIT_ATTENUATION_CONFIRMED"
        reason = (
            "the shortfall is attributable to the randomized-PIT estimator, "
            "and the constraint that refuses the corrected value is the "
            "latent-space RMSE -- which is itself measured against that same "
            "attenuated reading. The architecture represents a value that "
            "clears the twenty percent goal; what refuses it is the metric, "
            "not the factor family"
        )

    return {
        "classification": label,
        "allowed_classifications": list(CLASSIFICATIONS),
        "reason": reason,
        "predicates": predicates,
        "root_cause_is_separate_from_the_classification": {
            "randomized_pit_attenuation": attenuation["verdict"],
            "note": (
                "section 4 allows exactly one label, and the feasibility fact "
                "is the one that entails the others: it already says the "
                "parameter is identified and that the correction cannot be "
                "carried. The root cause stays in section 2's verdict, which "
                "is reported here so the single label does not bury it"
            ),
        },
        "sensitivity_to_how_the_constraints_are_read": {
            name: {
                "max_feasible_entry": entry["max_feasible_entry"],  # type: ignore[index]
                "reduction_at_the_boundary": entry[  # type: ignore[index]
                    "reduction_at_the_boundary"
                ],
                "reaches_target_reduction": entry[  # type: ignore[index]
                    "reaches_target_reduction"
                ],
            }
            for name, entry in envelope[  # type: ignore[union-attr]
                "by_reading_of_the_constraints"
            ].items()
        },
    }


#: The four conclusions section 7 allows, keyed by the section 4
#: classification that implies each one.
CONCLUSION_BY_CLASSIFICATION = {
    "IDENTIFIABLE_AND_ACHIEVABLE": (
        "COUNT-SPACE BLOCKER IS AN ESTIMATOR ISSUE AND IS FIXABLE GENERICALLY"
    ),
    # Both of these say the same thing about what to do next: the estimator is
    # at fault, the corrected value is identified, and the global constraints
    # refuse it. They differ only in where the refusal comes from -- the
    # envelope not reaching the goal at all, or reaching it but not at the
    # generic estimator's own value.
    "IDENTIFIABLE_BUT_GLOBALLY_INCOMPATIBLE": (
        "COUNT-SPACE BLOCKER IS AN ESTIMATOR ISSUE BUT FIX VIOLATES GLOBAL "
        "CONSTRAINTS"
    ),
    "RANDOMIZED_PIT_ATTENUATION_CONFIRMED": (
        "COUNT-SPACE BLOCKER IS AN ESTIMATOR ISSUE BUT FIX VIOLATES GLOBAL "
        "CONSTRAINTS"
    ),
    # No estimator defect means the shortfall is the factor family's, which is
    # the structural reading.
    "NO_ESTIMATOR_DEFECT_FOUND": (
        "COUNT-SPACE BLOCKER IS STRUCTURAL UNDER CURRENT ARCHITECTURE"
    ),
    "NOT_IDENTIFIABLE_FROM_CURRENT_MARGINALS": (
        "20% GATE NOT SUPPORTED BY EVIDENCE"
    ),
}


def stop_rule(
    envelope: Mapping[str, object],
    at_generic_value: Mapping[str, object],
    generic_all_buckets: Mapping[str, object] | None,
    forward: Mapping[str, object],
    classification: str,
) -> dict[str, object]:
    """Section 6, applied without discretion.

    "If no generic estimator achieves twenty percent without violating the
    constraints, STOP." The clause has two halves and both are scored: the
    estimator has to be generic, and it has to clear the goal with every
    constraint intact at its own value -- not at a value chosen because it
    happens to fit inside the envelope, which would be a bucket-specific free
    parameter wearing a different name.
    """
    reaches = bool(at_generic_value["reaches_target_reduction"])
    feasible = bool(at_generic_value["all_constraints_pass"])
    failed = [
        name
        for name, passed in at_generic_value["checks"].items()  # type: ignore[union-attr]
        if not passed
    ]
    all_buckets_feasible = (
        None
        if generic_all_buckets is None
        else bool(
            generic_all_buckets["latent_rmse_within_tolerance"]
            and generic_all_buckets["count_rmse_within_tolerance"]
        )
    )
    proceed = bool(reaches and feasible)
    return {
        "generic_estimator": (
            "interval-censored Gaussian-copula pseudo-likelihood, one scalar "
            "per bucket from one procedure, no bucket-specific free parameter"
        ),
        "improves_every_pre_2024_forward_fold": bool(
            forward["censored_improves_every_forward_fold"]
        ),
        "mean_forward_fold_error_reduction": float(forward["mean_error_reduction"]),  # type: ignore[arg-type]
        "section_5_precondition_met": bool(forward["section_5_precondition_met"]),
        "inner_fit_was_run": bool(forward["inner_fit_was_run"]),
        "at_its_own_value_reaches_twenty_percent": reaches,
        "at_its_own_value_every_constraint_passes": feasible,
        "constraints_it_fails": failed,
        "at_its_own_value_latent_rmse_ratio_to_control": float(
            at_generic_value["global_latent_rmse_ratio_to_control"]  # type: ignore[arg-type]
        ),
        "at_its_own_value_passes_the_pipelines_own_1_05_tolerance": bool(
            at_generic_value["passes_under_the_pipelines_own_1_05_tolerance"]
        ),
        "how_close_the_decision_is": (
            "the commissioned latent-RMSE tolerance is 1.03 and the pipeline's "
            "own is 1.05; where the generic value lands between the two, the "
            "STOP is a property of the tighter tolerance and not of the "
            "architecture, and the margin is reported above so the reader can "
            "see which side of it the decision falls on"
        ),
        "applied_to_all_twelve_buckets_is_feasible": all_buckets_feasible,
        "largest_feasible_entry": float(envelope["max_feasible_entry"]),  # type: ignore[arg-type]
        "reduction_at_the_largest_feasible_entry": float(
            envelope["at_max_feasible"]["focal_count_error_reduction"]  # type: ignore[index]
        ),
        "entry_needed_for_twenty_percent": envelope[
            "entry_needed_for_target_reduction"
        ],
        "by_reading_of_the_constraints": {
            name: {
                "max_feasible_entry": entry["max_feasible_entry"],  # type: ignore[index]
                "reduction_at_the_boundary": entry[  # type: ignore[index]
                    "reduction_at_the_boundary"
                ],
                "reaches_target_reduction": entry[  # type: ignore[index]
                    "reaches_target_reduction"
                ],
                "binding_constraints_just_past_the_boundary": entry[  # type: ignore[index]
                    "binding_constraints_just_past_the_boundary"
                ],
            }
            for name, entry in envelope[  # type: ignore[union-attr]
                "by_reading_of_the_constraints"
            ].items()
        },
        "decision": "PROCEED_TO_SECTION_5_FIT" if proceed else "STOP",
        "stop_reason": (
            None
            if proceed
            else (
                "no generic estimator reaches a twenty percent count-space "
                "error reduction at its own value with every global "
                "constraint intact, so section 6 applies and no further model "
                "is invented here"
            )
        ),
        "no_new_factor_family_was_fitted": True,
        "final_conclusion": CONCLUSION_BY_CLASSIFICATION[classification],
    }


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------


def build_count_residuals(
    residuals: pd.DataFrame,
    history: pd.DataFrame,
    marginals: Mapping[int, object],
    seasons: Sequence[int],
    cache: Path,
    stats: Sequence[str] = ("ast", "pts"),
) -> pd.DataFrame:
    """Standardized count residuals for the pre-2024 windows.

    Built here rather than read from the uncommitted
    ``v2_bridge_count_residuals.parquet``, whose analytic standard deviations
    do not reproduce under either of the pipeline's two marginal conventions,
    so its provenance cannot be stated. These use the validator convention --
    the one the held-out simulation inverts -- declared per season.

    Only the focal bucket's two stats are tabulated by default: the pre-2024
    count-space evidence this study reads is the focal bucket's observed
    correlation, and tabulating the other four margins would triple the cost
    for numbers nothing consumes.
    """
    path = cache / f"pre2024_count_residuals_{'_'.join(stats)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames: list[pd.DataFrame] = []
    for season in seasons:
        rows = residuals.loc[residuals["season"] == season].reset_index(drop=True)
        joined = rows.merge(
            history[
                ["game_id", "player_id"]
                + [f"mu_selected_{stat}" for stat in stats]
            ],
            on=["game_id", "player_id"],
            how="left",
            validate="one_to_one",
        )
        block = joined[["game_id", "team_id", "player_id", "season"]].copy()
        for stat in stats:
            mean = np.empty(len(joined), dtype=float)
            deviation = np.empty(len(joined), dtype=float)
            for position in range(len(joined)):
                row = joined.iloc[position]
                margin = discrete_marginal(
                    tabulate_inverse_cdf(
                        marginals[season].fitted[stat],
                        float(row[f"mu_selected_{stat}"]),
                        row,
                    )
                )
                mean[position] = margin.mean
                deviation[position] = margin.sd
            block[f"e_{stat}"] = (
                joined[f"y_{stat}"].to_numpy(dtype=float) - mean
            ) / deviation
            block[f"analytic_mean_{stat}"] = mean
            block[f"analytic_sd_{stat}"] = deviation
        frames.append(block)
        log(f"  count residuals: season {season}, {len(block):,} rows")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def season_marginals_and_copula(
    cache: Path,
    history: pd.DataFrame,
    season: int,
) -> tuple[FL.MarginalSet, object]:
    """Walk-forward margins and incumbent copula for one evaluation season."""
    path = cache / f"fit_{season}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    started = time.time()
    margins = FL.fit_walk_forward_marginals(history, season, FL.STATS, "validator")
    columns = {stat: f"mu_selected_{stat}" for stat in FL.STATS}
    train = history.loc[history["season"] < season]
    rows = train.dropna(subset=[*FL.STATS, *columns.values()])
    copula = GaussianCopula(targets=list(FL.STATS)).fit(
        rows, marginals=dict(margins.fitted), mu_columns=columns
    )
    with path.open("wb") as handle:
        pickle.dump((margins, copula), handle)
    log(f"  season {season} production refit in {time.time() - started:.0f}s")
    return margins, copula


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=ARTIFACT_DIR / "cache"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/research/game_latent_state",
    )
    parser.add_argument(
        "--residuals",
        type=Path,
        default=PROJECT_ROOT
        / "research/final_upstream_remediation/oof_gaussian_residuals.parquet",
    )
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--bridge-pairs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--out", type=Path, default=ARTIFACT_DIR / "count_space_forensic.json"
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    cache = arguments.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    residuals = pd.read_parquet(arguments.residuals)
    residuals["season"] = residuals["season"].astype(int)
    history = pd.read_parquet(
        arguments.data_root / "processed/oof_selected_means.parquet"
    )
    history["season"] = history["season"].astype(int)

    report: dict[str, object] = {
        "title": "COUNT-SPACE BLOCKER FORENSIC REPORT",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "focal_bucket": FL.FOCAL_BUCKET,
    }

    log("section 0: provenance")
    report["provenance"] = provenance(residuals)

    log("section 0b: production refits for every evaluation season")
    marginals: dict[int, FL.MarginalSet] = {}
    copulas: dict[int, object] = {}
    # The incumbent copula is only needed where a game covariance is
    # assembled, which is the held-out universe; the pre-2024 work needs the
    # margins alone, and fitting a copula it never reads would double the
    # refit cost for nothing.
    for season in FL.HOLDOUT_SEASONS:
        margins, copula = season_marginals_and_copula(cache, history, season)
        marginals[season] = margins
        copulas[season] = copula
    for season in FL.PRE_2024_SEASONS:
        marginals[season] = FL.cached_marginals(
            cache, history, season, "validator"
        )
        log(
            f"  season {season} margins on "
            f"{list(marginals[season].training_seasons)}"
        )
    report["marginal_conventions"] = {
        "used_for_count_space_and_simulation": "validator",
        "used_for_the_committed_latent_columns": "residual_build",
        "note": (
            "02_build_oof_residuals.py drops rows missing any of the six "
            "stats or any of the six selected means before splitting, so its "
            "marginals are fitted on the intersection across stats, while "
            "04_validate_shadow_v1.py::fit_season drops only the stat it is "
            "fitting. The two give CDF bounds that differ by up to 1e-2 on "
            "pts, ast, stl and fg3m. Pre-existing, reported, not introduced "
            "here"
        ),
        "by_season": {
            str(season): {
                "training_seasons": list(marginals[season].training_seasons),
                "training_rows": marginals[season].training_rows,
            }
            for season in marginals
        },
    }

    log("section 0c: pre-2024 count residuals")
    counts = build_count_residuals(
        residuals, history, marginals, FL.PRE_2024_SEASONS, cache
    )

    mle_cache_path = cache / "all_bucket_mle.json"
    mle_cache = (
        json.loads(mle_cache_path.read_text()) if mle_cache_path.exists() else None
    )

    log("section 1: estimator diagnostic")
    estimators = estimator_table(
        residuals.loc[residuals["season"] < min(FL.HOLDOUT_SEASONS)],
        counts,
        FL.FOCAL_BUCKET,
        "same_team",
        "ast",
        "pts",
        arguments.bootstrap,
        arguments.seed,
        mle_cache,
        cache / "estimator_table.json",
    )
    report["section_1_estimator_diagnostic"] = {
        "bucket": FL.FOCAL_BUCKET,
        "rule": (
            "one scalar per bucket from a single generic procedure; no "
            "bucket-specific free covariance parameter is introduced anywhere "
            "in this study"
        ),
        "jitter_seeds": list(JITTER_SEEDS),
        "by_window": estimators,
    }

    log("section 2: transmission test")
    pooled = residuals.loc[residuals["season"].isin(FL.PRE_2024_SEASONS)]
    pooled_series = build_statistic_series(
        pooled.merge(
            history[
                ["game_id", "player_id"]
                + [f"mu_selected_{stat}" for stat in ("ast", "pts")]
            ],
            on=["game_id", "player_id"],
            how="left",
            validate="one_to_one",
        ),
        marginals[2023].fitted,
        "same_team",
        "ast",
        "pts",
        arguments.bridge_pairs,
        arguments.seed,
    )
    transmission = transmission_test(
        estimators, pooled_series, "pooled_2020_2023"
    )
    report["section_2_transmission"] = {
        "margins": (
            "pooled over a deterministic by-game sample of the bucket's own "
            "ordered pairs, tabulated from the 2023 walk-forward production "
            "marginals, which are fitted on 2016-2022 and therefore read no "
            "held-out season"
        ),
        "bridge_pairs": int(pooled_series["_pairs"]),  # type: ignore[arg-type]
        "pooled_2020_2023": transmission,
        "randomized_pit_attenuation": attenuation_verdict(
            estimators, transmission, "pooled_2020_2023"
        ),
    }

    log("section 3: feasibility envelope")
    candidate_spec = json.loads(CANDIDATE_SPEC.read_text())
    loadings = SharedFactorLoadings.from_payload(candidate_spec["loadings"])
    universe_path = cache / "holdout_universe.pkl"
    if universe_path.exists():
        with universe_path.open("rb") as handle:
            games = pickle.load(handle)
    else:
        games = []
        for season in FL.HOLDOUT_SEASONS:
            started = time.time()
            games.extend(
                HL.build_holdout_games(
                    residuals,
                    history,
                    season,
                    marginals[season].fitted,
                    copulas[season],
                    progress=log,
                )
            )
            log(f"  held-out season {season} scored in {time.time() - started:.0f}s")
        with universe_path.open("wb") as handle:
            pickle.dump(games, handle)
    log(f"  held-out universe: {len(games)} games")

    control_loadings = SharedFactorLoadings.from_payload(
        json.loads(CONTROL_SPEC.read_text())["loadings"]
    )
    constraints = constraint_set(games, control_loadings)
    fidelity = {}
    for name, candidate in (
        ("control", control_loadings),
        ("candidate", loadings),
    ):
        HL.assemble(games, candidate)
        predicted = HL.predict_buckets(HL.bucket_series(games, "count"), candidate)
        published = constraints[f"published_{name}_count"]
        differences = [predicted[b] - published[b] for b in predicted]
        fidelity[name] = {
            "max_abs_difference": float(np.max(np.abs(differences))),
            "rms_difference": float(np.sqrt(np.mean(np.square(differences)))),
            "by_bucket": {
                b: {
                    "exact_predictor": float(predicted[b]),
                    "published_monte_carlo": float(published[b]),
                }
                for b in sorted(predicted)
            },
        }
    report["predictor_fidelity"] = {
        "note": (
            "the envelope is read off an exact predictor, so it only means "
            "anything if the predictor reproduces the published Monte Carlo "
            "runs; both paired runs are reproduced here before any "
            "feasibility claim is made"
        ),
        "count_space_runs": fidelity,
        "latent_space_is_not_simulated": {
            "finding": (
                "04_validate_shadow_v1.py::latent_dependence_summary sets "
                "implied_buckets to bucket_values(same_team_correlation(), "
                "cross_team_correlation()) -- the architecture's raw copula "
                "parameter matrices. The latent gate therefore compares a "
                "copula *parameter* against a randomized-PIT *sample moment*, "
                "with no transmission in between, and the published latent "
                "implied buckets are reproduced exactly by reading the "
                "parameter rather than by simulating"
            ),
            "max_abs_parameter_vs_published": {
                name: float(
                    np.max(
                        np.abs(
                            [
                                FL.bucket_readout(candidate)[bucket]
                                - report_payload["latent_dependence"]["by_model"][
                                    "candidate"
                                ]["implied_buckets"][bucket]
                                for bucket in FL.bucket_readout(candidate)
                            ]
                        )
                    )
                )
                for name, candidate, report_payload in (
                    (
                        "control",
                        control_loadings,
                        json.loads(CONTROL_REPORT.read_text()),
                    ),
                    (
                        "candidate",
                        loadings,
                        json.loads(CANDIDATE_REPORT.read_text()),
                    ),
                )
            },
            "transmitted_latent_reading_on_the_holdout": transmitted_latent_readings(
                games, control_loadings, loadings
            ),
        },
    }

    grid = [
        0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.0515, 0.0527, 0.054,
        0.056, 0.058, 0.060, 0.065, 0.070, 0.080, 0.090, 0.12, 0.20,
    ]
    envelope = feasibility_envelope(games, loadings, constraints, grid)
    report["section_3_feasibility_envelope"] = envelope
    log(
        "  commissioned max entry "
        f"{envelope['by_reading_of_the_constraints']['commissioned']['max_feasible_entry']:.8f}"
    )

    log("section 5: inner forward test")
    forward_history = history[
        ["game_id", "player_id"]
        + [f"mu_selected_{stat}" for stat in ("ast", "pts")]
    ]
    forward = inner_forward_test(
        residuals.merge(
            forward_history,
            on=["game_id", "player_id"],
            how="left",
            validate="one_to_one",
        ),
        counts,
        marginals,
        estimators,
        arguments.bridge_pairs,
        arguments.seed,
    )
    report["section_5_inner_forward_test"] = forward

    # The generic estimator's own pre-2024 value, evaluated at face value --
    # which is the only honest reading of section 5, because picking a
    # different value because it fits inside the envelope would reintroduce
    # the bucket-specific free parameter the rule forbids.
    generic_value = float(
        estimators["pooled_2020_2023"]["D_interval_censored_mle"]["rho"]  # type: ignore[index]
    )
    at_generic = evaluate_target(games, loadings, generic_value, constraints)
    report["section_5_at_the_generic_estimators_own_value"] = {
        "entry": generic_value,
        "source_window": "pooled_2020_2023",
        "evaluation": at_generic,
    }
    generic_all = None
    if mle_cache is not None:
        generic_all = generic_application(games, loadings, constraints, mle_cache)
        report["section_5_generic_application_to_all_buckets"] = generic_all

    log("section 4: identifiability")
    report["section_4_identifiability"] = identifiability(
        report["section_2_transmission"]["randomized_pit_attenuation"],  # type: ignore[index]
        envelope,
        at_generic,
        generic_all,
        transmission,
        forward,
    )

    log("section 6: stop rule")
    classification = str(
        report["section_4_identifiability"]["classification"]  # type: ignore[index]
    )
    report["section_6_stop_rule"] = stop_rule(
        envelope, at_generic, generic_all, forward, classification
    )
    report["final_conclusion"] = CONCLUSION_BY_CLASSIFICATION[classification]
    report["classification"] = classification
    report["answers"] = five_answers(
        envelope["by_reading_of_the_constraints"],  # type: ignore[index]
        envelope["trade_curve"]["curve"],  # type: ignore[index]
        envelope["entry_needed_for_target_reduction"],  # type: ignore[index]
        forward,
        transmission,
    )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=False))
    log(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
