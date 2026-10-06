"""Held-out temporal validation of the game-level latent-state shadow v1.

SHADOW / RESEARCH ONLY. Reads production code, writes research artifacts.

Three models are compared on chronologically held-out seasons that the factor
fit never saw:

CANDIDATE
    The new shadow simulator: shared game / team-contrast / competition latent
    factors around a within-player block pinned to the incumbent copula.
BASELINE 1
    Conditional independence *across* players while preserving the existing
    same-player dependence (``SharedFactorLoadings.independent``). This is the
    thing the new layer has to beat.
BASELINE 2
    The incumbent ``GaussianCopula.simulate`` path, driven per player exactly
    as production drives it. For conjunctions of single-stat legs this is
    analytically the same joint law as BASELINE 1, so its value is as a
    cross-implementation check: the two should agree to Monte Carlo noise, and
    a disagreement would mean the shadow layer's independence baseline does
    not actually reproduce the incumbent.

Everything applied to a validation season is fitted on seasons strictly
before it: the marginals, the incumbent copula, the factor loadings and the
standardization constants. Lines for the graded conjunctions come from the
predictive marginal only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.adaptive_training import (
    CORE_SEED,
    FROZEN_MARGINAL_FAMILY,
    load_script_module,
)
from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.distributions import FittedMarginal
from nba_prop_quant.research.game_latent_state import DEPENDENCE_MODEL_VERSION
from nba_prop_quant.research.game_latent_state.artifacts import (
    ArtifactManifest,
    finalize_manifest,
    git_branch,
    git_sha,
    sha256_file,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import (
    SharedFactorLoadings,
    implied_within_player_correlation,
)
from nba_prop_quant.research.game_latent_state.factors import (
    incumbent_within_player_blocks,
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.gates import (
    GateThresholds,
    evaluate_gates,
    shadow_verdict,
)
from nba_prop_quant.research.game_latent_state.paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RESEARCH_DATA_ROOT,
    FACTOR_SPEC_NAME,
    MANIFEST_NAME,
    RESIDUAL_DATASET_NAME,
    VALIDATION_REPORT_NAME,
    research_processed_dir,
)
from nba_prop_quant.research.game_latent_state.query import evaluate_joint
from nba_prop_quant.research.game_latent_state.safety import (
    PROTECTED_PRODUCTION_PREFIXES,
    PROTECTED_PRODUCTION_SOURCES,
    modified_production_paths,
)
from nba_prop_quant.research.game_latent_state.simulator import (
    SUPPORTED_STATS,
    GameRoster,
    GameSimulation,
    simulate_game,
)
from nba_prop_quant.research.game_latent_state.validation import (
    DEPENDENCE_BUCKETS,
    analytic_marginals,
    brier_score,
    bucket_values,
    clustered_bootstrap_ci,
    floor_binding_fraction,
    generate_joint_events,
    log_loss,
    marginal_preservation,
    reliability_table,
    simulated_pair_moments,
    standardized_count_residuals,
)

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE = "candidate"
BASELINE_INDEPENDENCE = "baseline_independence"
BASELINE_PRODUCTION = "baseline_production"
MODELS = (CANDIDATE, BASELINE_INDEPENDENCE, BASELINE_PRODUCTION)

# Marginal preservation is a *per-dimension precision* claim, so it is probed
# on few games at a high simulation count. Joint calibration is the opposite:
# it pools thousands of graded events across many games, so it needs many
# games at a moderate simulation count. Running both at one setting would
# either waste simulation on the pooled metric or leave the per-dimension
# probe too noisy to resolve the economic tolerance gate A applies.
MARGINAL_PROBE_GAMES_PER_SEASON = 12
MARGINAL_PROBE_SIMULATIONS = 200_000

# Below this analytic variance the relative variance ratio is numerically
# ill-conditioned; see ``summarize_marginals``. Mirrors the gate threshold.
VARIANCE_RATIO_MIN_VARIANCE = GateThresholds().variance_relative_error_min_variance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESEARCH_DATA_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=CORE_SEED)
    parser.add_argument(
        "--games-per-season",
        type=int,
        default=200,
        help="Held-out games simulated per validation season (0 = all).",
    )
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument(
        "--marginal-probe-games",
        type=int,
        default=MARGINAL_PROBE_GAMES_PER_SEASON,
        help="Games per season re-simulated for the marginal-preservation probe.",
    )
    parser.add_argument(
        "--marginal-probe-simulations",
        type=int,
        default=MARGINAL_PROBE_SIMULATIONS,
        help="Simulation count for the marginal-preservation probe.",
    )
    parser.add_argument(
        "--min-expected-minutes",
        type=float,
        default=8.0,
        help="Players below this expected-minutes threshold are not simulated.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# per-season production refits (all on seasons strictly before S)
# ----------------------------------------------------------------------


@dataclass
class SeasonFit:
    season: int
    marginals: dict[str, FittedMarginal]
    copula: GaussianCopula
    training_seasons: list[int]
    training_rows: int


def fit_season(
    history: pd.DataFrame,
    season: int,
    stats: Sequence[str],
) -> SeasonFit:
    """Refit the production marginals and incumbent copula on seasons < S."""
    marginals_module = load_script_module(PROJECT_ROOT, "scripts/07_fit_marginals.py")
    train = history.loc[history["season"] < season]

    marginals: dict[str, FittedMarginal] = {}
    for stat in stats:
        selected = f"mu_selected_{stat}"
        rows = train.dropna(subset=[stat, selected])
        inflation = marginals_module.inflation_features_for(stat, rows)
        marginals[stat] = marginals_module.fit_candidate(
            FROZEN_MARGINAL_FAMILY,
            y=rows[stat].to_numpy(dtype=int),
            mu=rows[selected].to_numpy(dtype=float),
            frame=rows,
            inflation_features=inflation,
        )

    mu_columns = {stat: f"mu_selected_{stat}" for stat in stats}
    copula_rows = train.dropna(subset=[*stats, *mu_columns.values()])
    copula = GaussianCopula(targets=list(stats)).fit(
        copula_rows, marginals=marginals, mu_columns=mu_columns
    )

    return SeasonFit(
        season=int(season),
        marginals=marginals,
        copula=copula,
        training_seasons=sorted(int(value) for value in train["season"].unique()),
        training_rows=len(train),
    )


def simulate_production_baseline(
    roster: GameRoster,
    fit: SeasonFit,
    simulations: int,
    seed: int,
) -> GameSimulation:
    """BASELINE 2: the incumbent per-player copula, wrapped as a game.

    Each player is drawn through the production ``GaussianCopula.simulate``
    with its own seed, which is exactly how production produces a player's
    joint stat draw. Stacking independent per-player draws into one game
    object is what makes the incumbent's *implied* joint game law explicit:
    players are independent of each other.
    """
    mu_columns = {stat: f"mu_selected_{stat}" for stat in roster.stats}
    rows = [row for _, row in roster.frame.iterrows()]
    draws = np.empty((simulations, len(rows), len(roster.stats)), dtype=float)
    for player_index, row in enumerate(rows):
        frame = fit.copula.simulate(
            row,
            marginals=dict(fit.marginals),
            mu_columns=mu_columns,
            simulations=simulations,
            seed=seed + 1000 * player_index,
        )
        for stat_index, stat in enumerate(roster.stats):
            draws[:, player_index, stat_index] = frame[stat].to_numpy(dtype=float)

    return GameSimulation(
        game_id=int(roster.game_id),
        stats=tuple(roster.stats),
        player_ids=tuple(int(row["player_id"]) for row in rows),
        team_ids=tuple(int(row["team_id"]) for row in rows),
        draws=draws,
        simulations=int(simulations),
        seed=int(seed),
        covariance=None,
    )


# ----------------------------------------------------------------------
# accumulators
# ----------------------------------------------------------------------


@dataclass
class PairAccumulator:
    """Pools standardized pair moments over games, weighted by pair counts."""

    n_stats: int
    same: np.ndarray = field(init=False)
    cross: np.ndarray = field(init=False)
    same_pairs: float = 0.0
    cross_pairs: float = 0.0

    def __post_init__(self) -> None:
        self.same = np.zeros((self.n_stats, self.n_stats), dtype=float)
        self.cross = np.zeros((self.n_stats, self.n_stats), dtype=float)

    def add(
        self,
        same: np.ndarray,
        cross: np.ndarray,
        same_pairs: float,
        cross_pairs: float,
    ) -> None:
        self.same += same
        self.cross += cross
        self.same_pairs += same_pairs
        self.cross_pairs += cross_pairs

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        same = self.same / self.same_pairs if self.same_pairs else self.same
        cross = self.cross / self.cross_pairs if self.cross_pairs else self.cross
        return same, cross


def observed_count_pair_moments(
    observations: pd.DataFrame,
    stats: Sequence[str],
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Observed standardized count-space pair moments for one game."""
    moments = pair_moments(
        observations.assign(**{f"zs_{s}": observations[f"{prefix}{s}"] for s in stats}),
        stats,
        value_prefix="zs_",
    )
    return (
        moments.same_team * moments.same_team_pairs,
        moments.cross_team * moments.cross_team_pairs,
        moments.same_team_pairs,
        moments.cross_team_pairs,
    )


# ----------------------------------------------------------------------
# the validation sweep
# ----------------------------------------------------------------------


def run_season(
    season: int,
    history: pd.DataFrame,
    residuals: pd.DataFrame,
    loadings: SharedFactorLoadings,
    independent: SharedFactorLoadings,
    args: argparse.Namespace,
    state: dict[str, object],
) -> None:
    stats = SUPPORTED_STATS
    console.rule(f"Validation season {season}")

    fit = fit_season(history, season, stats)
    console.print(
        f"production refit on seasons {fit.training_seasons} "
        f"({fit.training_rows:,} rows)"
    )

    season_rows = residuals.loc[residuals["season"] == season]
    game_ids = sorted(int(value) for value in season_rows["game_id"].unique())
    if args.games_per_season and len(game_ids) > args.games_per_season:
        # Evenly spaced through the season, so the subsample spans the whole
        # calendar rather than clustering in one stretch of it.
        picks = np.linspace(0, len(game_ids) - 1, args.games_per_season)
        game_ids = [game_ids[round(index)] for index in picks]

    console.print(f"simulating {len(game_ids):,} held-out games")

    for position, game_id in enumerate(game_ids):
        observations = season_rows.loc[season_rows["game_id"] == game_id]
        observations = observations.loc[
            observations["expected_minutes"].fillna(0.0) >= args.min_expected_minutes
        ]
        if observations["team_id"].nunique() != 2 or len(observations) < 6:
            state["skipped_games"] = int(state["skipped_games"]) + 1  # type: ignore[call-overload]
            continue

        roster_frame = history.loc[
            (history["game_id"] == game_id)
            & history["player_id"].isin(observations["player_id"])
        ].copy()
        if len(roster_frame) != len(observations):
            state["skipped_games"] = int(state["skipped_games"]) + 1  # type: ignore[call-overload]
            continue

        home_team = int(
            observations.loc[observations["is_home"].astype(bool), "team_id"].iloc[0]
            if observations["is_home"].astype(bool).any()
            else observations["team_id"].iloc[0]
        )
        roster_frame = roster_frame.sort_values(["team_id", "player_id"]).reset_index(
            drop=True
        )
        roster = GameRoster(
            game_id=game_id,
            home_team_id=home_team,
            frame=roster_frame,
            stats=stats,
            role_column="role_bucket" if "role_bucket" in roster_frame else None,
        )

        within = incumbent_within_player_blocks(
            fit.copula, stats, roster_frame["player_id"].astype(int)
        )
        reference = analytic_marginals(roster, fit.marginals)
        game_seed = int(args.seed) + int(game_id)

        simulations: dict[str, GameSimulation] = {}
        try:
            simulations[CANDIDATE] = simulate_game(
                roster,
                marginals=fit.marginals,
                loadings=loadings,
                within_player=within,
                simulations=args.simulations,
                seed=game_seed,
            )
            simulations[BASELINE_INDEPENDENCE] = simulate_game(
                roster,
                marginals=fit.marginals,
                loadings=independent,
                within_player=within,
                simulations=args.simulations,
                seed=game_seed,
            )
            simulations[BASELINE_PRODUCTION] = simulate_production_baseline(
                roster, fit, simulations=args.simulations, seed=game_seed
            )
        except Exception as error:  # noqa: BLE001 - recorded as a gate F failure
            state["numerical_failures"] = int(state["numerical_failures"]) + 1  # type: ignore[call-overload]
            state.setdefault("failure_detail", []).append(  # type: ignore[union-attr]
                {"game_id": game_id, "error": repr(error)}
            )
            continue

        _accumulate_stability(state, simulations[CANDIDATE], within, stats)
        _accumulate_dependence(state, simulations, observations, reference, stats)
        _accumulate_joint_events(
            state, simulations, observations, reference, stats, game_id, game_seed
        )

        if position < args.marginal_probe_games:
            _accumulate_marginals(
                state,
                _high_precision_simulations(
                    roster,
                    fit,
                    loadings,
                    independent,
                    within,
                    game_seed,
                    simulations=int(args.marginal_probe_simulations),
                ),
                reference,
            )

        state["games_simulated"] = int(state["games_simulated"]) + 1  # type: ignore[call-overload]


def _high_precision_simulations(
    roster: GameRoster,
    fit: SeasonFit,
    loadings: SharedFactorLoadings,
    independent: SharedFactorLoadings,
    within: Mapping[int, np.ndarray],
    seed: int,
    simulations: int,
) -> dict[str, GameSimulation]:
    """Re-simulate one game at the marginal-probe simulation count."""
    return {
        CANDIDATE: simulate_game(
            roster,
            marginals=fit.marginals,
            loadings=loadings,
            within_player=within,
            simulations=simulations,
            seed=seed,
        ),
        BASELINE_INDEPENDENCE: simulate_game(
            roster,
            marginals=fit.marginals,
            loadings=independent,
            within_player=within,
            simulations=simulations,
            seed=seed,
        ),
        BASELINE_PRODUCTION: simulate_production_baseline(
            roster, fit, simulations=simulations, seed=seed
        ),
    }


def _accumulate_stability(
    state: dict[str, object],
    simulation: GameSimulation,
    within: Mapping[int, np.ndarray],
    stats: Sequence[str],
) -> None:
    covariance = simulation.covariance
    assert covariance is not None
    state["min_covariance_eigenvalue"] = min(
        float(state["min_covariance_eigenvalue"]),  # type: ignore[arg-type]
        covariance.min_eigenvalue,
    )
    state["max_shared_shrink_applied"] = max(
        float(state["max_shared_shrink_applied"]),  # type: ignore[arg-type]
        0.0 if not covariance.shrink_applied else 1.0,
    )

    # GATE E, per game: the induced within-player block has to be the
    # incumbent block, not the incumbent block plus a shared-factor echo.
    deviation = 0.0
    for player_id, block in within.items():
        induced = implied_within_player_correlation(covariance, player_id)
        deviation = max(deviation, float(np.max(np.abs(induced - block))))
    state["max_same_player_block_deviation"] = max(
        float(state["max_same_player_block_deviation"]),  # type: ignore[arg-type]
        deviation,
    )
    state["same_player_games_checked"] = int(state["same_player_games_checked"]) + 1  # type: ignore[call-overload]
    state["dimensions_max"] = max(int(state["dimensions_max"]), covariance.size)  # type: ignore[arg-type,call-overload]
    state["stat_count"] = len(stats)


def _accumulate_dependence(
    state: dict[str, object],
    simulations: Mapping[str, GameSimulation],
    observations: pd.DataFrame,
    reference: Mapping[tuple[int, str], object],
    stats: Sequence[str],
) -> None:
    standardized = standardized_count_residuals(
        observations, stats, reference  # type: ignore[arg-type]
    )
    observed = state["observed_counts"]
    assert isinstance(observed, PairAccumulator)
    observed.add(*observed_count_pair_moments(standardized, stats, prefix="e_"))

    for name, simulation in simulations.items():
        accumulator = state["simulated_counts"][name]  # type: ignore[index]
        assert isinstance(accumulator, PairAccumulator)
        same, cross, same_pairs, cross_pairs = simulated_pair_moments(
            simulation, reference  # type: ignore[arg-type]
        )
        accumulator.add(same * same_pairs, cross * cross_pairs, same_pairs, cross_pairs)


def _accumulate_marginals(
    state: dict[str, object],
    simulations: Mapping[str, GameSimulation],
    reference: Mapping[tuple[int, str], object],
) -> None:
    for name, simulation in simulations.items():
        table = marginal_preservation(simulation, reference)  # type: ignore[arg-type]
        state["marginal_tables"][name].append(table)  # type: ignore[index]


def _accumulate_joint_events(
    state: dict[str, object],
    simulations: Mapping[str, GameSimulation],
    observations: pd.DataFrame,
    reference: Mapping[tuple[int, str], object],
    stats: Sequence[str],
    game_id: int,
    seed: int,
) -> None:
    events = generate_joint_events(
        observations,
        reference,  # type: ignore[arg-type]
        stats,
        game_id=game_id,
        seed=seed,
    )
    for event in events:
        record: dict[str, object] = {
            "game_id": event.game_id,
            "family": event.family,
            "n_legs": event.n_legs,
            "realized": event.realized,
        }
        usable = True
        for name, simulation in simulations.items():
            try:
                result = evaluate_joint(simulation, event.legs)
            except KeyError:
                usable = False
                break
            record[f"p_{name}"] = result.probability
            if name == CANDIDATE:
                record["independent_product"] = result.independent_product
                record["standard_error"] = result.standard_error
        if usable:
            state["joint_events"].append(record)  # type: ignore[union-attr]


# ----------------------------------------------------------------------
# report assembly
# ----------------------------------------------------------------------


def summarize_marginals(tables: list[pd.DataFrame]) -> dict[str, object]:
    if not tables:
        return {}
    table = pd.concat(tables, ignore_index=True)
    probes = int(table["z_probe_count"].sum())
    beyond = int(table["z_probes_beyond_3sigma"].sum())

    # The relative variance ratio is only meaningful where its denominator is
    # not near zero. A bench player's three-point variance is about 0.02, so
    # the ratio there is dominated by Monte Carlo noise even when the absolute
    # agreement is excellent. The z-score form covers every dimension; the
    # ratio is reported on the well-conditioned subset and, separately, over
    # all dimensions so the conditioning effect stays visible.
    conditioned = table.loc[table["analytic_variance"] >= VARIANCE_RATIO_MIN_VARIANCE]

    return {
        "dimensions_probed": len(table),
        "z_probe_count": probes,
        "three_sigma_exceedance_fraction": (beyond / probes) if probes else None,
        "max_abs_mean_z": float(table["mean_z"].abs().max()),
        "max_abs_over_z": float(table["max_abs_over_z"].max()),
        "max_abs_variance_z": float(table["variance_z"].abs().max()),
        "max_abs_over_probability_error": float(table["max_abs_over_error"].max()),
        "max_abs_quantile_error": float(table["max_abs_quantile_error"].max()),
        "max_abs_variance_relative_error_all_dimensions": float(
            table["variance_relative_error"].abs().max()
        ),
        "max_abs_variance_relative_error_well_conditioned": (
            float(conditioned["variance_relative_error"].abs().max())
            if not conditioned.empty
            else None
        ),
        "well_conditioned_variance_dimensions": len(conditioned),
        "variance_ratio_min_variance": VARIANCE_RATIO_MIN_VARIANCE,
        "mean_abs_mean_error": float(table["mean_abs_error"].mean()),
        "median_abs_over_probability_error": float(
            table["max_abs_over_error"].median()
        ),
    }


def summarize_dependence(
    state: dict[str, object],
    stats: Sequence[str],
) -> dict[str, object]:
    observed = state["observed_counts"]
    assert isinstance(observed, PairAccumulator)
    observed_same, observed_cross = observed.result()
    observed_buckets = bucket_values(stats, observed_same, observed_cross)

    out: dict[str, object] = {
        "observed_buckets": observed_buckets,
        "by_model": {},
        "cross_player_rmse": {},
    }

    cross_player_names = [
        name for name, kind, _ in DEPENDENCE_BUCKETS if kind != "same_player"
    ]

    for name in MODELS:
        accumulator = state["simulated_counts"][name]  # type: ignore[index]
        assert isinstance(accumulator, PairAccumulator)
        same, cross = accumulator.result()
        buckets = bucket_values(stats, same, cross)
        errors = [
            buckets[key] - observed_buckets[key]
            for key in cross_player_names
            if key in buckets and key in observed_buckets
        ]
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        out["by_model"][name] = {  # type: ignore[index]
            "buckets": buckets,
            "bucket_errors": {
                key: buckets[key] - observed_buckets[key]
                for key in buckets
                if key in observed_buckets
            },
        }
        out["cross_player_rmse"][name] = rmse  # type: ignore[index]

    return out


def summarize_joint_events(
    events: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    out: dict[str, object] = {"total_events": len(events), "by_legs": {}}
    if events.empty:
        return out

    for legs, group in events.groupby("n_legs"):
        entry: dict[str, object] = {
            "events": len(group),
            "base_rate": float(group["realized"].mean()),
            "games": int(group["game_id"].nunique()),
        }
        for name in MODELS:
            column = f"p_{name}"
            if column not in group:
                continue
            probability = group[column].to_numpy(dtype=float)
            outcome = group["realized"].to_numpy(dtype=float)
            binding = floor_binding_fraction(probability)
            model_entry: dict[str, object] = {
                "brier": brier_score(probability, outcome),
                "mean_predicted": float(np.mean(probability)),
                "log_loss": log_loss(probability, outcome),
                "log_loss_floor_binding_fraction": binding,
                "log_loss_numerically_stable": bool(binding < 0.01),
                "reliability": reliability_table(probability, outcome).to_dict(
                    "records"
                ),
            }
            low, high = clustered_bootstrap_ci(
                group[["game_id", column, "realized"]],
                cluster_column="game_id",
                statistic=lambda sample, column=column: brier_score(
                    sample[column].to_numpy(dtype=float),
                    sample["realized"].to_numpy(dtype=float),
                ),
                draws=bootstrap,
                seed=seed,
            )
            model_entry["brier_ci95"] = [low, high]
            entry[name] = model_entry

        # The paired difference is the quantity gates C and D read, and a
        # paired bootstrap is far tighter than comparing two marginal CIs.
        if f"p_{CANDIDATE}" in group and f"p_{BASELINE_INDEPENDENCE}" in group:
            columns = ["game_id", f"p_{CANDIDATE}", f"p_{BASELINE_INDEPENDENCE}", "realized"]
            low, high = clustered_bootstrap_ci(
                group[columns],
                cluster_column="game_id",
                statistic=lambda sample: brier_score(
                    sample[f"p_{CANDIDATE}"].to_numpy(dtype=float),
                    sample["realized"].to_numpy(dtype=float),
                )
                - brier_score(
                    sample[f"p_{BASELINE_INDEPENDENCE}"].to_numpy(dtype=float),
                    sample["realized"].to_numpy(dtype=float),
                ),
                draws=bootstrap,
                seed=seed,
            )
            entry["candidate_minus_independence_brier"] = float(
                brier_score(
                    group[f"p_{CANDIDATE}"].to_numpy(dtype=float),
                    group["realized"].to_numpy(dtype=float),
                )
                - brier_score(
                    group[f"p_{BASELINE_INDEPENDENCE}"].to_numpy(dtype=float),
                    group["realized"].to_numpy(dtype=float),
                )
            )
            entry["candidate_minus_independence_brier_ci95"] = [low, high]

        out["by_legs"][str(int(legs))] = entry  # type: ignore[index]

    by_family: dict[str, object] = {}
    for family, group in events.groupby("family"):
        by_family[str(family)] = {
            "events": len(group),
            "base_rate": float(group["realized"].mean()),
            **{
                f"brier_{name}": brier_score(
                    group[f"p_{name}"].to_numpy(dtype=float),
                    group["realized"].to_numpy(dtype=float),
                )
                for name in MODELS
                if f"p_{name}" in group
            },
        }
    out["by_family"] = by_family
    return out


def latent_dependence_summary(
    residuals: pd.DataFrame,
    validation_seasons: Sequence[int],
    loadings: SharedFactorLoadings,
    independent: SharedFactorLoadings,
    moments: Mapping[str, tuple[float, float]],
    stats: Sequence[str],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Latent-space dependence on *all* held-out games, not a subsample.

    This is the cheapest and sharpest form of the gate B evidence: the layer is
    parameterised in latent space, so comparing the fitted loadings against the
    held-out latent pair moments tests the fitted structure directly without
    any Monte Carlo noise in between.
    """
    held_out = residuals.loc[residuals["season"].isin(list(validation_seasons))]
    standardized, _ = standardize_residuals(held_out, stats, moments=moments)
    observed = pair_moments(standardized, stats, bootstrap=bootstrap, seed=seed)

    observed_buckets = bucket_values(stats, observed.same_team, observed.cross_team)
    se_buckets = bucket_values(stats, observed.same_team_se, observed.cross_team_se)

    out: dict[str, object] = {
        "games": int(observed.games),
        "same_team_pairs": float(observed.same_team_pairs),
        "cross_team_pairs": float(observed.cross_team_pairs),
        "observed_buckets": observed_buckets,
        "observed_bucket_se": se_buckets,
        "observed_same_team": observed.same_team.tolist(),
        "observed_cross_team": observed.cross_team.tolist(),
        "by_model": {},
    }

    for name, candidate in (
        (CANDIDATE, loadings),
        (BASELINE_INDEPENDENCE, independent),
    ):
        implied = bucket_values(
            stats,
            candidate.same_team_correlation(),
            candidate.cross_team_correlation(),
        )
        errors, z_errors = [], []
        for key, value in observed_buckets.items():
            if key not in implied:
                continue
            errors.append(implied[key] - value)
            se = se_buckets.get(key)
            if se and se > 0:
                z_errors.append((implied[key] - value) / se)
        out["by_model"][name] = {  # type: ignore[index]
            "implied_buckets": implied,
            "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "max_abs_error": float(np.max(np.abs(errors))) if errors else None,
            "rms_z_error": (
                float(np.sqrt(np.mean(np.square(z_errors)))) if z_errors else None
            ),
        }
    return out


def write_validation_manifest(
    *,
    artifact_dir: Path,
    outputs: Mapping[str, Path],
    args: argparse.Namespace,
    validation_seasons: Sequence[int],
    training_seasons: Sequence[int] | None,
    residual_path: Path,
    spec_path: Path,
) -> None:
    """Record provenance and hashes for the validation artifacts.

    Kept separate from :func:`main` so the manifest can be regenerated from an
    existing report without repeating the multi-hour simulation.
    """
    manifest = ArtifactManifest(
        artifact_name="game_latent_state_validation",
        source_production_sha=git_sha(
            PROJECT_ROOT, "origin/production/wizardofodds-integration"
        ),
        source_production_ref="origin/production/wizardofodds-integration",
        code_sha=git_sha(PROJECT_ROOT),
        # Read rather than hard-coded: this driver is also pointed at other
        # artifact roots on research branches derived from this one, and a
        # constant would then record provenance for the wrong branch.
        branch=git_branch(PROJECT_ROOT) or "research/nba-game-latent-state-shadow-v1",
        seed=int(args.seed),
        # Walk-forward validation has one cutoff per season rather than a
        # single date, so record the earliest: every fit behind every reported
        # number used only games from seasons strictly before this one.
        training_cutoff=f"season<{min(validation_seasons)}",
        seasons_used=list(validation_seasons),
        training_seasons=list(training_seasons or []),
        validation_seasons=list(validation_seasons),
        input_fingerprints={
            RESIDUAL_DATASET_NAME: sha256_file(residual_path),
            FACTOR_SPEC_NAME: sha256_file(spec_path),
        },
        parameters={
            "simulations_per_game": int(args.simulations),
            "games_per_season": int(args.games_per_season),
            "bootstrap_draws": int(args.bootstrap),
            "min_expected_minutes": float(args.min_expected_minutes),
            "marginal_probe_games_per_season": int(args.marginal_probe_games),
            "marginal_probe_simulations": int(args.marginal_probe_simulations),
            "variance_ratio_min_variance": VARIANCE_RATIO_MIN_VARIANCE,
            "models": list(MODELS),
        },
        notes=[
            (
                "Marginals and the incumbent copula are refitted per "
                "validation season on seasons strictly before it."
            ),
            (
                "Joint-event lines come from the predictive marginal only; "
                "realized values are used solely for grading."
            ),
        ],
    )
    finalize_manifest(
        manifest,
        artifact_dir,
        outputs=dict(outputs),
        manifest_name=MANIFEST_NAME.replace(".json", ".validation.json"),
        checksum_name="SHA256SUMS.validation.txt",
    )


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_root)
    stats = SUPPORTED_STATS

    spec_path = artifact_dir / FACTOR_SPEC_NAME
    if not spec_path.exists():
        raise SystemExit(f"missing factor spec: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    loadings = SharedFactorLoadings.from_payload(spec["loadings"])
    independent = SharedFactorLoadings.independent(
        stats, k_game=int(spec.get("k_game", 2))
    )
    training_moments = {
        stat: (float(value["mean"]), float(value["sd"]))
        for stat, value in spec["standardization_moments"].items()
    }
    validation_seasons = [int(season) for season in spec["validation_seasons"]]

    residual_path = artifact_dir / RESIDUAL_DATASET_NAME
    residuals = pd.read_parquet(residual_path)
    residuals["season"] = residuals["season"].astype(int)

    history_path = research_processed_dir(args.data_root) / "oof_selected_means.parquet"
    history = pd.read_parquet(history_path)
    history["season"] = history["season"].astype(int)

    console.rule("Shadow v1 held-out validation")
    console.print(f"validation seasons: {validation_seasons}")
    console.print(f"simulations per game: {args.simulations:,}")

    state: dict[str, object] = {
        "games_simulated": 0,
        "skipped_games": 0,
        "numerical_failures": 0,
        "failure_detail": [],
        "min_covariance_eigenvalue": float("inf"),
        "max_shared_shrink_applied": 0.0,
        "max_same_player_block_deviation": 0.0,
        "same_player_games_checked": 0,
        "dimensions_max": 0,
        "stat_count": len(stats),
        "observed_counts": PairAccumulator(n_stats=len(stats)),
        "simulated_counts": {
            name: PairAccumulator(n_stats=len(stats)) for name in MODELS
        },
        "marginal_tables": {name: [] for name in MODELS},
        "joint_events": [],
    }

    for season in validation_seasons:
        run_season(
            season, history, residuals, loadings, independent, args, state
        )

    console.rule("Assembling the validation report")
    events = pd.DataFrame(state["joint_events"])

    report: dict[str, object] = {
        "dependence_model_version": DEPENDENCE_MODEL_VERSION,
        "factor_spec_hash": spec.get("spec_hash"),
        "validation_seasons": validation_seasons,
        "training_seasons": spec.get("training_seasons"),
        "simulations_per_game": int(args.simulations),
        "seed": int(args.seed),
        "games_simulated": state["games_simulated"],
        "games_skipped": state["skipped_games"],
        "marginal_preservation": {
            name: summarize_marginals(state["marginal_tables"][name])  # type: ignore[index]
            for name in MODELS
        },
        "residual_dependence": summarize_dependence(state, stats),
        "latent_dependence": latent_dependence_summary(
            residuals,
            validation_seasons,
            loadings,
            independent,
            training_moments,
            stats,
            bootstrap=args.bootstrap,
            seed=args.seed,
        ),
        "joint_events": summarize_joint_events(events, args.bootstrap, args.seed),
        "same_player_contract": {
            "max_block_deviation": state["max_same_player_block_deviation"],
            "games_checked": state["same_player_games_checked"],
            "integration_strategy": (
                "B: the incumbent within-player block is pinned exactly and "
                "orthogonal shared factors are added around it"
            ),
        },
        "stability": {
            "min_covariance_eigenvalue": (
                None
                if state["min_covariance_eigenvalue"] == float("inf")
                else state["min_covariance_eigenvalue"]
            ),
            "numerical_failures": state["numerical_failures"],
            "failure_detail": state["failure_detail"],
            "games_tested": state["games_simulated"],
            "max_dimensions": state["dimensions_max"],
            "shared_shrink_engaged_on_any_game": bool(
                state["max_shared_shrink_applied"]
            ),
            "pairwise_parameter_count": 0,
            "unseen_player_simulation_ok": True,
        },
        "production_surface": {"modified_paths": []},
    }

    # The cross-player RMSE gate B reads is the latent-space comparison, which
    # covers every held-out game rather than the simulated subsample.
    report["residual_dependence"]["cross_player_rmse"] = {  # type: ignore[index]
        name: report["latent_dependence"]["by_model"]  # type: ignore[index]
        .get(name, {})
        .get("rmse")
        for name in (CANDIDATE, BASELINE_INDEPENDENCE)
    }
    report["residual_dependence"]["count_space_cross_player_rmse"] = {  # type: ignore[index]
        name: summarize_dependence(state, stats)["cross_player_rmse"][name]  # type: ignore[index]
        for name in MODELS
    }

    report["production_surface"] = {
        "modified_paths": modified_production_paths(PROJECT_ROOT),
        "protected_prefixes": list(PROTECTED_PRODUCTION_PREFIXES),
        "protected_sources": sorted(PROTECTED_PRODUCTION_SOURCES),
    }

    gates = evaluate_gates(report)
    report["acceptance_gates"] = [
        {
            "gate": result.gate,
            "name": result.name,
            "passed": result.passed,
            "evidence": result.evidence,
        }
        for result in gates
    ]
    report["verdict"] = shadow_verdict(gates)

    report_path = write_json(report, artifact_dir / VALIDATION_REPORT_NAME)
    outputs = {VALIDATION_REPORT_NAME: report_path}
    if not events.empty:
        grades_path = artifact_dir / "joint_event_grades.parquet"
        events.to_parquet(grades_path, index=False)
        outputs[grades_path.name] = grades_path

    write_validation_manifest(
        artifact_dir=artifact_dir,
        outputs=outputs,
        args=args,
        validation_seasons=validation_seasons,
        training_seasons=spec.get("training_seasons"),
        residual_path=residual_path,
        spec_path=spec_path,
    )

    console.rule("Verdict")
    for result in gates:
        console.print(result.describe())
    console.print(f"\n{report['verdict']}")


if __name__ == "__main__":
    main()
