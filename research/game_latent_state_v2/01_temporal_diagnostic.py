#!/usr/bin/env python
"""Is teammate_ast_ast genuinely time-varying, or is 2023 sampling noise?

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The accepted repair undershoots the held-out ``teammate_ast_ast`` by 1.89
game-clustered standard errors, and one of its four training seasons carries
the opposite sign. The temptation is to hand the bucket a free time-varying
parameter. This script refuses to do that before answering the prior
question, and it answers it on pre-2024 data only.

Per training season it reports the point estimate, its game-clustered standard
error, a percentile bootstrap interval, the effective game and pair counts,
and the posterior sign probability. Then it fits

    observed_r_s ~ Normal(theta_s, se_s^2),   theta_s ~ Normal(mu, tau_time^2)

and reads three things off it: Cochran's ``Q`` (is the between-season spread
larger than the within-season noise?), ``tau_time`` against ``|mu|`` (do the
seasons differ by as much as the level itself?), and ``P(mu > 0)`` (is the
*level* confidently signed?). The order matters: a significant ``Q`` sitting
on top of a season interval that contains zero is evidence of noise, not of a
regime change.

Four treatments are then screened walk-forward on the training seasons alone:
the pooled control the repair uses, the random-effects posterior mean, a
recency-weighted variant, and a one-state random walk. The walk has to clear
a pre-registered serial-dependence bar before it is even compared, because
with a handful of seasons a dynamic model wins on flexibility and loses out of
sample.

Nothing here loads 2024 or 2025.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state import temporal as T  # noqa: E402
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.v2 import (  # noqa: E402
    REPAIR_CONTROL_SPEC,
    fit_v2_factors,
)

console = Console()

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

#: Buckets the temporal model is run on. The two repair targets plus the two
#: protected same-team neighbours, so a temporal treatment cannot be adopted
#: for one bucket without its effect on the others being visible.
BUCKETS: dict[str, tuple[str, str, str]] = {
    "teammate_ast_ast": ("same_team", "ast", "ast"),
    "teammate_reb_reb": ("same_team", "reb", "reb"),
    "passer_ast_teammate_pts": ("same_team", "ast", "pts"),
    "teammate_pts_reb": ("same_team", "pts", "reb"),
}

#: The bucket the brief names. Its selected treatment is the one that enters
#: the frozen candidate; the others are reported for contrast.
PRIMARY_BUCKET = "teammate_ast_ast"


def pooled_control_by_fold(
    frame: pd.DataFrame,
    folds: tuple[int, ...],
    bucket: tuple[str, str],
    bootstrap: int,
    seed: int,
) -> dict[int, float]:
    """The ``T0`` control's prediction for each fold season.

    ``T0`` is what the accepted repair does: pool every completed season into
    one empirical-Bayes fit and read the fitted same-team entry. Refitting it
    per fold on seasons ``< S`` is what makes the comparison with the temporal
    treatments a like-for-like walk-forward one.
    """
    index = {stat: position for position, stat in enumerate(STATS)}
    i, j = index[bucket[0]], index[bucket[1]]
    spec = REPAIR_CONTROL_SPEC.__class__(
        **{**REPAIR_CONTROL_SPEC.payload(), "role_column": None}
    )
    out: dict[int, float] = {}
    for fold in folds:
        history = frame.loc[frame["season"] < fold]
        fit = fit_v2_factors(
            history, STATS, spec=spec, bootstrap=bootstrap, seed=seed
        )
        out[fold] = float(fit.loadings.same_team_correlation()[i, j])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        default=str(
            PROJECT_ROOT
            / "research"
            / "game_latent_state"
            / "oof_gaussian_residuals.parquet"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state_v2"),
    )
    parser.add_argument("--bootstrap", type=int, default=800)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    console.rule("Shadow V2 temporal diagnostic (pre-2024 only)")
    residuals = pd.read_parquet(Path(args.residuals))
    residuals["season"] = residuals["season"].astype(int)
    pre = residuals[~residuals["season"].isin(HOLDOUT_SEASONS)].copy()
    leaked = sorted(set(pre["season"].unique()) & set(HOLDOUT_SEASONS))
    if leaked:
        raise AssertionError(f"holdout seasons leaked into the diagnostic: {leaked}")
    seasons = tuple(int(season) for season in sorted(pre["season"].unique()))
    console.print(f"seasons used: {list(seasons)}   rows {len(pre):,}")
    standardized, _ = standardize_residuals(pre, STATS)

    payload: dict[str, object] = {
        "scope": "pre-2024 training history only",
        "holdout_seasons_excluded": list(HOLDOUT_SEASONS),
        "seasons_used": list(seasons),
        "stats": list(STATS),
        "bootstrap_draws": int(args.bootstrap),
        "seed": int(args.seed),
        "primary_bucket": PRIMARY_BUCKET,
        "pre_registered_constants": {
            "sign_confidence": T.SIGN_CONFIDENCE,
            "heterogeneity_level": T.HETEROGENEITY_LEVEL,
            "time_varying_tau_ratio": T.TIME_VARYING_TAU_RATIO,
            "recency_half_life_seasons": T.RECENCY_HALF_LIFE_SEASONS,
            "ar1_support_threshold": T.AR1_SUPPORT_THRESHOLD,
            "min_seasons_for_dynamic": T.MIN_SEASONS_FOR_DYNAMIC,
            "min_seasons_for_prediction": T.MIN_SEASONS_FOR_PREDICTION,
            "selection_tie_band": T.SELECTION_TIE_BAND,
        },
        "buckets": {},
    }

    screenings: list[T.TemporalScreening] = []
    predictions: dict[str, dict[str, dict[str, float]]] = {}

    for name, (block, first_stat, second_stat) in BUCKETS.items():
        console.rule(f"{name}: season-by-season")
        estimates = T.season_bucket_estimates(
            standardized,
            STATS,
            (first_stat, second_stat),
            block=block,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        table = Table(title=f"{name}: per-season game-clustered estimates")
        for column in (
            "season",
            "estimate",
            "SE",
            "bootstrap CI95",
            "games",
            "ordered pairs",
            "P(>0)",
        ):
            table.add_column(column, justify="right")
        for item in estimates:
            table.add_row(
                str(item.season),
                f"{item.estimate:+.6f}",
                f"{item.standard_error:.6f}",
                f"[{item.bootstrap_low:+.6f}, {item.bootstrap_high:+.6f}]",
                f"{item.games:,}",
                f"{item.pairs:,.0f}",
                f"{item.posterior_sign_probability:.4f}",
            )
        console.print(table)

        crosses_zero = [
            item.season
            for item in estimates
            if item.bootstrap_low < 0.0 < item.bootstrap_high
        ]
        opposite_sign = [
            item.season
            for item in estimates
            if np.sign(item.estimate) != np.sign(np.mean([e.estimate for e in estimates]))
        ]
        if opposite_sign:
            console.print(
                f"seasons with the opposite sign: {opposite_sign}; of those, "
                f"{[s for s in opposite_sign if s in crosses_zero]} have a "
                "bootstrap interval containing zero, which is uncertainty "
                "rather than a regime switch."
            )

        random_effects = T.fit_random_effects(estimates)
        recency = T.fit_random_effects(
            estimates, decay_half_life=T.RECENCY_HALF_LIFE_SEASONS
        )
        walk = T.fit_random_walk(estimates, random_effects.lag1_autocorrelation)

        console.print(
            f"random effects: mu={random_effects.posterior_mean:+.6f} "
            f"(sd {random_effects.posterior_sd:.6f})  "
            f"tau_time={random_effects.tau_time:.6f}  "
            f"Q={random_effects.q_statistic:.3f} on df "
            f"{random_effects.degrees_of_freedom} (p="
            f"{random_effects.q_p_value:.4f})  I^2="
            f"{random_effects.i_squared:.3f}  "
            f"P(mu>0)={random_effects.posterior_sign_probability:.4f}  "
            f"lag1={random_effects.lag1_autocorrelation:+.4f}"
        )
        console.print(f"classification: [bold]{random_effects.classification}[/bold]")
        console.print(f"random walk: {walk.support_reason}")

        folds = tuple(
            int(item.season)
            for item in estimates
            if len([other for other in estimates if other.season < item.season])
            >= T.MIN_SEASONS_FOR_PREDICTION
        )
        console.print(f"walk-forward folds: {list(folds)}")
        pooled = pooled_control_by_fold(
            standardized,
            folds,
            (first_stat, second_stat),
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        console.print(
            "pooled T0 control per fold: "
            + ", ".join(f"{fold}: {value:+.6f}" for fold, value in pooled.items())
        )
        screening = T.screen_temporal_treatments(estimates, pooled, bucket=name)
        screen_table = Table(title=f"{name}: walk-forward treatment comparison")
        for column in ("treatment", "predictive RMSE", "mean z^2", "folds"):
            screen_table.add_column(column, justify="right")
        for treatment, scores in screening.scores.items():
            screen_table.add_row(
                treatment,
                f"{scores['predictive_rmse']:.6f}",
                f"{scores['mean_squared_z']:.4f}",
                f"{scores['folds']:.0f}",
            )
        console.print(screen_table)
        console.print(f"selected: [bold]{screening.selected}[/bold]")
        console.print(screening.selection_reason)
        screenings.append(screening)

        full_history_prediction = {
            treatment: T.treatment_prediction(
                treatment,
                estimates,
                target_season=min(HOLDOUT_SEASONS),
                pooled_estimate=float(
                    pooled[max(folds)] if folds else random_effects.posterior_mean
                ),
            ).payload()
            for treatment in (
                T.TREATMENT_POOLED,
                T.TREATMENT_RANDOM_EFFECTS,
                T.TREATMENT_RECENCY,
                T.TREATMENT_DYNAMIC,
            )
        }
        predictions[name] = full_history_prediction

        payload["buckets"][name] = {  # type: ignore[index]
            "block": block,
            "stat_pair": [first_stat, second_stat],
            "season_estimates": [item.payload() for item in estimates],
            "seasons_crossing_zero": crosses_zero,
            "seasons_opposite_sign": opposite_sign,
            "random_effects": random_effects.payload(),
            "recency_weighted": recency.payload(),
            "random_walk": walk.payload(),
            "pooled_control_by_fold": {
                str(fold): value for fold, value in pooled.items()
            },
            "screening": screening.payload(),
            "prediction_for_first_holdout_season": full_history_prediction,
        }

    selected = payload["buckets"][PRIMARY_BUCKET]["screening"]["selected_treatment"]  # type: ignore[index]
    payload["selected_treatment_for_primary_bucket"] = selected
    payload["selected_treatment_applies_to"] = [PRIMARY_BUCKET]
    payload["holdout_used_for_selection"] = False

    # The selected treatment supplies the point estimate; its predictive
    # standard deviation is the uncertainty the static repair never reported at
    # all. That SD is over-confident out of sample -- the posterior treats the
    # seasons as exchangeable around a level that is itself only estimated from
    # four of them -- so it is multiplied by one inflation scalar measured on
    # the inner folds. Pooled over the four screened buckets because two folds
    # apiece cannot support a per-bucket factor.
    console.rule("Predictive-uncertainty calibration for the selected treatment")
    inflation = T.pooled_uncertainty_inflation(screenings, selected)
    calibration_table = Table(title="inner-fold calibration of the predictive SD")
    for column in ("bucket", "mean z^2", "folds", "raw SD", "calibrated SD"):
        calibration_table.add_column(column, justify="right")
    calibrated: dict[str, dict[str, float]] = {}
    for name in BUCKETS:
        scores = payload["buckets"][name]["screening"]["scores"][selected]  # type: ignore[index]
        raw_sd = float(predictions[name][selected]["prediction_sd"])
        calibrated[name] = {
            "prediction": float(predictions[name][selected]["prediction"]),
            "raw_prediction_sd": raw_sd,
            "calibrated_prediction_sd": raw_sd * inflation["inflation"],
            "inner_fold_mean_squared_z": float(scores["mean_squared_z"]),
        }
        calibration_table.add_row(
            name,
            f"{scores['mean_squared_z']:.4f}",
            f"{scores['folds']:.0f}",
            f"{raw_sd:.6f}",
            f"{calibrated[name]['calibrated_prediction_sd']:.6f}",
        )
    console.print(calibration_table)
    console.print(
        f"pooled inflation factor {inflation['inflation']:.4f} from "
        f"mean z^2 {inflation['mean_squared_z']:.4f} over "
        f"{inflation['observations']:.0f} fold-bucket observations. Two folds "
        "per bucket is thin support for this scalar and it is reported as a "
        "calibration, not as a precise variance estimate."
    )
    payload["uncertainty_calibration"] = {
        "treatment": selected,
        "pooled": inflation,
        "by_bucket": calibrated,
        "estimated_on": "pre-2024 inner walk-forward folds only",
    }

    artifact_dir = Path(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "temporal_diagnostic.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    console.print(f"\nwrote {out_path.relative_to(PROJECT_ROOT)}")
    console.print(f"TEMPORAL_TREATMENT_SELECTED={selected}")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")


if __name__ == "__main__":
    main()
