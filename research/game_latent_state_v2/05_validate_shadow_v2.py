#!/usr/bin/env python
"""The single untouched 2024-2025 validation of the frozen Shadow V2 candidate.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE. RUN ONCE.

Five models, one set of held-out games, one seed:

CANDIDATE            the frozen Shadow V2 fit.
REPAIR               the accepted bucket repair, verbatim from its committed
                     factor spec.
V1                   accepted shadow V1, verbatim from its committed spec.
BASELINE_INDEPENDENCE
                     cross-player independence with the incumbent same-player
                     block preserved. What any dependence layer has to beat.
BASELINE_PRODUCTION  the incumbent per-player ``GaussianCopula.simulate``
                     path, driven exactly as production drives it.

Why all five in one run rather than differencing four reports
-------------------------------------------------------------
At three and four legs the Monte Carlo error on a single run's Brier score is
an order of magnitude wider than the tolerance the gates apply, so comparing
two independent runs cannot resolve the difference the gates ask about. Here
every model sees the same games, the same rosters, the same production refits
and the same per-game seed, so every comparison is paired and the shared
Monte Carlo error cancels. The accepted repair's run established that the
observed held-out moments reproduce to the last digit under this protocol,
and this driver asserts that again against the committed control.

Nothing is fitted here. The three loading sets are read from committed factor
specs, the marginals and the incumbent copula are refitted per validation
season on seasons strictly before it, and the lines for every graded
conjunction come from the predictive marginal alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.adaptive_training import (  # noqa: E402
    FROZEN_MARGINAL_FAMILY,
    load_script_module,
)
from nba_prop_quant.copula import GaussianCopula  # noqa: E402
from nba_prop_quant.distributions import FittedMarginal  # noqa: E402
from nba_prop_quant.research.game_latent_state import (  # noqa: E402
    DEPENDENCE_MODEL_VERSION,
)
from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    ArtifactManifest,
    finalize_manifest,
    git_branch,
    git_sha,
    sha256_file,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    SharedFactorLoadings,
    implied_within_player_correlation,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    incumbent_within_player_blocks,
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.gates import (  # noqa: E402
    GateThresholds,
    evaluate_gates,
    shadow_verdict,
)
from nba_prop_quant.research.game_latent_state.paths import (  # noqa: E402
    DEFAULT_RESEARCH_DATA_ROOT,
    FACTOR_SPEC_NAME,
    MANIFEST_NAME,
    RESIDUAL_DATASET_NAME,
    VALIDATION_REPORT_NAME,
    research_processed_dir,
)
from nba_prop_quant.research.game_latent_state.query import evaluate_joint  # noqa: E402
from nba_prop_quant.research.game_latent_state.safety import (  # noqa: E402
    PROTECTED_PRODUCTION_PREFIXES,
    PROTECTED_PRODUCTION_SOURCES,
    modified_production_paths,
)
from nba_prop_quant.research.game_latent_state.simulator import (  # noqa: E402
    SUPPORTED_STATS,
    GameRoster,
    GameSimulation,
    simulate_game,
)
from nba_prop_quant.research.game_latent_state.v2 import (  # noqa: E402
    role_cell_report,
    role_conditioned_rmse,
    role_pair_moments,
)
from nba_prop_quant.research.game_latent_state.validation import (  # noqa: E402
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

CANDIDATE = "candidate"
REPAIR = "repair"
V1 = "v1"
BASELINE_INDEPENDENCE = "baseline_independence"
BASELINE_PRODUCTION = "baseline_production"

#: Order matters only for reporting. ``candidate`` is first because the gate
#: evaluator reads it by name.
MODELS: tuple[str, ...] = (
    CANDIDATE,
    REPAIR,
    V1,
    BASELINE_INDEPENDENCE,
    BASELINE_PRODUCTION,
)

#: The models carrying fitted shared factors, i.e. the ones whose implied
#: latent blocks can be compared with the held-out latent moments directly.
FITTED_MODELS: tuple[str, ...] = (CANDIDATE, REPAIR, V1, BASELINE_INDEPENDENCE)

#: Paired comparisons the report carries per leg count. The first element is
#: always the candidate, so a negative delta is always "V2 is better".
PAIRED_COMPARISONS: tuple[tuple[str, str], ...] = (
    (CANDIDATE, V1),
    (CANDIDATE, REPAIR),
    (CANDIDATE, BASELINE_INDEPENDENCE),
)

HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

# Same split of effort as the accepted control's run: marginal preservation is
# a per-dimension precision claim, so it is probed on few games at a high
# simulation count, while joint calibration pools thousands of graded events
# and needs many games at a moderate count.
MARGINAL_PROBE_GAMES_PER_SEASON = 12
MARGINAL_PROBE_SIMULATIONS = 200_000

VARIANCE_RATIO_MIN_VARIANCE = GateThresholds().variance_relative_error_min_variance

BRANCH = "research/nba-game-latent-state-shadow-v2-structural"


def parse_args() -> argparse.Namespace:
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
    # The control's own settings. Changing either would break the pairing with
    # the committed control baseline, so they are asserted against it below.
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--games-per-season", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument(
        "--marginal-probe-games", type=int, default=MARGINAL_PROBE_GAMES_PER_SEASON
    )
    parser.add_argument(
        "--marginal-probe-simulations", type=int, default=MARGINAL_PROBE_SIMULATIONS
    )
    parser.add_argument("--min-expected-minutes", type=float, default=8.0)
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
    """The incumbent per-player copula, stacked into one game object."""
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
    loadings: Mapping[str, SharedFactorLoadings],
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
        picks = np.linspace(0, len(game_ids) - 1, args.games_per_season)
        game_ids = [game_ids[round(index)] for index in picks]

    console.print(f"simulating {len(game_ids):,} held-out games across {len(MODELS)} models")

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
            for name in FITTED_MODELS:
                simulations[name] = simulate_game(
                    roster,
                    marginals=fit.marginals,
                    loadings=loadings[name],
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
                {
                    name: simulate_game(
                        roster,
                        marginals=fit.marginals,
                        loadings=loadings[name],
                        within_player=within,
                        simulations=int(args.marginal_probe_simulations),
                        seed=game_seed,
                    )
                    for name in FITTED_MODELS
                }
                | {
                    BASELINE_PRODUCTION: simulate_production_baseline(
                        roster,
                        fit,
                        simulations=int(args.marginal_probe_simulations),
                        seed=game_seed,
                    )
                },
                reference,
            )

        state["games_simulated"] = int(state["games_simulated"]) + 1  # type: ignore[call-overload]
        if (position + 1) % 25 == 0:
            console.print(
                f"  {position + 1:,}/{len(game_ids):,} games "
                f"({int(state['games_simulated']):,} usable)"
            )


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
        out["by_model"][name] = {  # type: ignore[index]
            "buckets": buckets,
            "bucket_errors": {
                key: buckets[key] - observed_buckets[key]
                for key in buckets
                if key in observed_buckets
            },
        }
        out["cross_player_rmse"][name] = (  # type: ignore[index]
            float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        )

    return out


def paired_delta(
    group: pd.DataFrame,
    first: str,
    second: str,
    statistic,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """One paired, game-clustered difference of a proper scoring rule.

    The bootstrap resamples *games*, not events, because a game contributes
    many overlapping conjunctions and treating them as independent would
    understate the interval. The same resample is applied to both models,
    which is what makes the shared Monte Carlo and sampling error cancel.
    """
    left = f"p_{first}"
    right = f"p_{second}"
    outcome = group["realized"].to_numpy(dtype=float)
    delta = float(
        statistic(group[left].to_numpy(dtype=float), outcome)
        - statistic(group[right].to_numpy(dtype=float), outcome)
    )
    low, high = clustered_bootstrap_ci(
        group[["game_id", left, right, "realized"]],
        cluster_column="game_id",
        statistic=lambda sample: statistic(
            sample[left].to_numpy(dtype=float),
            sample["realized"].to_numpy(dtype=float),
        )
        - statistic(
            sample[right].to_numpy(dtype=float),
            sample["realized"].to_numpy(dtype=float),
        ),
        draws=bootstrap,
        seed=seed,
    )
    return {
        "delta": delta,
        "ci95": [low, high],
        "indistinguishable_from_zero": bool(low <= 0.0 <= high),
        "better": bool(high < 0.0),
        "worse": bool(low > 0.0),
    }


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

        # Both proper scoring rules, paired. Brier is the control's currency
        # and log loss is the brief's: Brier is bounded and insensitive to the
        # extreme tail a four-leg conjunction lives in, while log loss is
        # unbounded there, so a model can trade one for the other and the
        # gates read both.
        paired: dict[str, object] = {}
        for first, second in PAIRED_COMPARISONS:
            if f"p_{first}" not in group or f"p_{second}" not in group:
                continue
            paired[f"{first}_minus_{second}"] = {
                "brier": paired_delta(
                    group, first, second, brier_score, bootstrap, seed
                ),
                "log_loss": paired_delta(
                    group, first, second, log_loss, bootstrap, seed
                ),
            }
        entry["paired"] = paired
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
    loadings: Mapping[str, SharedFactorLoadings],
    role_blind_twin: SharedFactorLoadings,
    moments: Mapping[str, tuple[float, float]],
    stats: Sequence[str],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Latent-space dependence on *every* held-out game, not a subsample.

    The layer is parameterised in latent space, so comparing the fitted
    loadings against the held-out latent pair moments tests the fitted
    structure with no Monte Carlo noise in between. Role-conditioned models
    are read through their pair-share average, which is what a pooled bucket
    measures once the simulator draws every pair with its own roles.
    """
    held_out = residuals.loc[residuals["season"].isin(list(validation_seasons))]
    standardized, _ = standardize_residuals(held_out, stats, moments=moments)
    observed = pair_moments(standardized, stats, bootstrap=bootstrap, seed=seed)

    observed_buckets = bucket_values(stats, observed.same_team, observed.cross_team)
    se_buckets = bucket_values(
        stats, observed.same_team_se, observed.cross_team_se
    )

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

    for name in FITTED_MODELS:
        candidate = loadings[name]
        implied = bucket_values(
            stats,
            candidate.pooled_same_team_correlation(),
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
            "bucket_z": {
                key: (implied[key] - observed_buckets[key]) / se_buckets[key]
                for key in implied
                if key in observed_buckets and se_buckets.get(key, 0.0) > 0
            },
        }

    # The role-conditioned claim, on the holdout: how well each model's
    # same-team block fits the *role cells* rather than the pooled average.
    # The candidate is additionally compared with its own role-blind twin,
    # which is the fit its role layer replaces; every other model has no role
    # layer, so for them there is nothing to compare against.
    role_moments = role_pair_moments(
        standardized, stats, bootstrap=bootstrap, seed=seed
    )
    same_team_pairs = {
        name: pair for name, kind, pair in DEPENDENCE_BUCKETS if kind == "same_team"
    }
    out["role_conditioned"] = {
        name: {
            **role_conditioned_rmse(
                role_moments,
                loadings[name],
                pooled_only=role_blind_twin if name == CANDIDATE else None,
            ),
            "cells": role_cell_report(role_moments, loadings[name], same_team_pairs),
        }
        for name in FITTED_MODELS
    }
    return out


def load_loadings(path: Path) -> tuple[SharedFactorLoadings, dict]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    return SharedFactorLoadings.from_payload(spec["loadings"]), spec


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_root)
    stats = SUPPORTED_STATS

    v2_loadings, v2_spec = load_loadings(Path(args.artifact_root) / FACTOR_SPEC_NAME)
    repair_loadings, repair_spec = load_loadings(
        Path(args.repair_artifact_root) / FACTOR_SPEC_NAME
    )
    v1_loadings, v1_spec = load_loadings(
        Path(args.v1_artifact_root) / FACTOR_SPEC_NAME
    )

    freeze_path = artifact_dir / "shadow_v2_candidate.json"
    if not freeze_path.exists():
        raise SystemExit(f"freeze the candidate first: {freeze_path} missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("frozen"):
        raise SystemExit("the candidate record does not say it is frozen")
    if freeze["factor_spec_hash"] != v2_spec["spec_hash"]:
        raise SystemExit(
            "the factor spec does not match the frozen record; the candidate "
            "was modified after it was frozen"
        )

    control_path = artifact_dir / "repair_control_baseline.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if (
        int(control["simulations_per_game"]) != int(args.simulations)
        or int(control["seed"]) != int(args.seed)
    ):
        raise SystemExit(
            "this run's simulation count or seed differs from the control's; "
            "the comparison would not be paired"
        )

    validation_seasons = [int(season) for season in v2_spec["validation_seasons"]]
    if sorted(validation_seasons) != sorted(HOLDOUT_SEASONS):
        raise SystemExit(f"unexpected validation seasons: {validation_seasons}")
    for name, spec in (("v1", v1_spec), ("repair", repair_spec)):
        if sorted(int(s) for s in spec["validation_seasons"]) != sorted(
            validation_seasons
        ):
            raise SystemExit(f"{name} holds out different seasons")
        if sorted(int(s) for s in spec["training_seasons"]) != sorted(
            int(s) for s in v2_spec["training_seasons"]
        ):
            raise SystemExit(f"{name} was fitted on different training seasons")

    # All three specs standardize with their own training constants. They were
    # fitted on the same seasons, so the constants have to agree exactly; if
    # they did not, the held-out latent moments below would be measured in
    # three different units and the comparison would be meaningless.
    training_moments = {
        stat: (float(value["mean"]), float(value["sd"]))
        for stat, value in v2_spec["standardization_moments"].items()
    }
    for name, spec in (("v1", v1_spec), ("repair", repair_spec)):
        for stat, value in spec["standardization_moments"].items():
            if not np.isclose(
                value["mean"], training_moments[stat][0], rtol=0, atol=1e-12
            ) or not np.isclose(
                value["sd"], training_moments[stat][1], rtol=0, atol=1e-12
            ):
                raise SystemExit(
                    f"{name} standardizes {stat} differently from the candidate"
                )

    role_blind_twin = SharedFactorLoadings.from_payload(
        v2_spec["role_blind_twin_loadings"]
    )
    # The candidate's own base path with the symmetric subspace switched off:
    # the reference the cross-team isolation claim is stated against.
    no_repair_twin_loadings = SharedFactorLoadings.from_payload(
        v2_spec["no_repair_twin_loadings"]
    )
    screening = json.loads(
        (artifact_dir / "inner_screening.json").read_text(encoding="utf-8")
    )
    if screening["holdout_used_for_selection"]:
        raise SystemExit("the inner screen reports that it used the holdout")
    loadings: dict[str, SharedFactorLoadings] = {
        CANDIDATE: v2_loadings,
        REPAIR: repair_loadings,
        V1: v1_loadings,
        BASELINE_INDEPENDENCE: SharedFactorLoadings.independent(
            stats, k_game=int(v1_spec.get("k_game", 2))
        ),
    }

    residual_path = Path(args.v1_artifact_root) / RESIDUAL_DATASET_NAME
    residuals = pd.read_parquet(residual_path)
    residuals["season"] = residuals["season"].astype(int)

    history_path = research_processed_dir(args.data_root) / "oof_selected_means.parquet"
    history = pd.read_parquet(history_path)
    history["season"] = history["season"].astype(int)

    console.rule("Shadow V2 untouched holdout validation")
    console.print(f"validation seasons  : {validation_seasons}")
    console.print(f"simulations per game: {args.simulations:,}")
    console.print(f"models              : {list(MODELS)}")
    console.print(f"frozen candidate    : {freeze['candidate_name']}")
    console.print(
        "cross-team identity at the frozen fit: "
        f"{freeze['structural_identities_at_the_frozen_fit']['cross_team_unchanged_deviation']:.3e}"
    )

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
        run_season(season, history, residuals, loadings, args, state)

    console.rule("Assembling the validation report")
    events = pd.DataFrame(state["joint_events"])
    dependence = summarize_dependence(state, stats)
    latent = latent_dependence_summary(
        residuals,
        validation_seasons,
        loadings,
        role_blind_twin,
        training_moments,
        stats,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    report: dict[str, object] = {
        "dependence_model_version": DEPENDENCE_MODEL_VERSION,
        "scope": "single untouched 2024-2025 validation of the frozen Shadow V2 candidate",
        "factor_spec_hash": v2_spec.get("spec_hash"),
        "repair_factor_spec_hash": repair_spec.get("spec_hash"),
        "v1_factor_spec_hash": v1_spec.get("spec_hash"),
        "frozen_candidate": freeze["candidate_name"],
        "frozen_hyperparameters": freeze["hyperparameters"],
        "models": list(MODELS),
        "validation_seasons": validation_seasons,
        "training_seasons": v2_spec.get("training_seasons"),
        "simulations_per_game": int(args.simulations),
        "games_per_season": int(args.games_per_season),
        "seed": int(args.seed),
        "games_simulated": state["games_simulated"],
        "games_skipped": state["skipped_games"],
        "marginal_preservation": {
            name: summarize_marginals(state["marginal_tables"][name])  # type: ignore[index]
            for name in MODELS
        },
        "residual_dependence": dependence,
        "latent_dependence": latent,
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
            "pairwise_parameter_count": int(
                freeze["parameter_counts"]["pairwise"]
            ),
            "player_indexed_parameter_count": int(
                freeze["parameter_counts"]["player_indexed"]
            ),
            "unseen_player_simulation_ok": True,
        },
        "structural_identities": {
            # Re-read from the committed loadings rather than from the freeze
            # record, so the artifact the validation consumed is the one
            # audited.
            #
            # The claim is that the *same-team* work does not move the
            # cross-team block, so the reference is the candidate's own base
            # path with the symmetric subspace switched off -- the no-repair
            # twin frozen alongside the candidate -- and not some other model.
            # Earlier this was measured against accepted V1's block, which is
            # only the right reference when the candidate is on V1's cross-team
            # path; the screen selects that path on its own merits and selected
            # the repair's.
            "cross_team_block_moved_by_the_same_team_work": float(
                np.max(
                    np.abs(
                        v2_loadings.cross_team_correlation()
                        - no_repair_twin_loadings.cross_team_correlation()
                    )
                )
            ),
            "cross_team_block_versus_v1_path": float(
                np.max(
                    np.abs(
                        v2_loadings.cross_team_correlation()
                        - (
                            v1_loadings.game @ v1_loadings.game.T
                            - v1_loadings.contrast_matrix
                            @ v1_loadings.contrast_matrix.T
                        )
                    )
                )
            ),
            "cross_team_block_versus_repair_control": float(
                np.max(
                    np.abs(
                        v2_loadings.cross_team_correlation()
                        - repair_loadings.cross_team_correlation()
                    )
                )
            ),
            "symmetric_subspace_rank": int(v2_loadings.r_symmetric),
            "symmetric_mode": freeze["hyperparameters"]["symmetric_mode"],
            "role_deviation_carried": bool(
                v2_loadings.role_deviation is not None
            ),
            "role_quadratic_share": v2_loadings.role_quadratic_share(),
            # Where the identity has content: every same-team grid point the
            # inner screen fitted, not just the one the freeze took.
            "screened_same_team_points": {
                name: screening["candidates"][name][
                    "max_cross_team_unchanged_deviation"
                ]
                for name in screening["axes"]["same_team"]
            },
        },
        "production_surface": {
            "modified_paths": modified_production_paths(PROJECT_ROOT),
            "protected_prefixes": list(PROTECTED_PRODUCTION_PREFIXES),
            "protected_sources": sorted(PROTECTED_PRODUCTION_SOURCES),
        },
    }

    # Gate B reads the latent-space comparison, which covers every held-out
    # game rather than the simulated subsample.
    report["residual_dependence"]["cross_player_rmse"] = {  # type: ignore[index]
        name: latent["by_model"].get(name, {}).get("rmse")  # type: ignore[union-attr]
        for name in FITTED_MODELS
    }
    report["residual_dependence"]["count_space_cross_player_rmse"] = {  # type: ignore[index]
        name: dependence["cross_player_rmse"][name]  # type: ignore[index]
        for name in MODELS
    }

    # The control measured the same held-out games with the same seed, so its
    # observed moments must come back identical. A mismatch means the two runs
    # are not comparable and every paired statement built on them is void.
    control_observed = control["latent_buckets"]
    drift = {
        name: abs(
            float(latent["observed_buckets"][name])  # type: ignore[index]
            - float(entry["observed"])
        )
        for name, entry in control_observed.items()
        if name in latent["observed_buckets"]  # type: ignore[operator]
    }
    report["control_pairing"] = {
        "control_is": control["control_is"],
        "max_observed_bucket_drift": max(drift.values()) if drift else None,
        "observed_bucket_drift": drift,
        "paired": bool(drift and max(drift.values()) <= 1e-12),
        "control_games_simulated": control["games_simulated"],
        "games_simulated": state["games_simulated"],
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
    report["framework_verdict"] = shadow_verdict(gates)

    report_path = write_json(report, artifact_dir / VALIDATION_REPORT_NAME)
    outputs = {VALIDATION_REPORT_NAME: report_path}
    if not events.empty:
        grades_path = artifact_dir / "joint_event_grades.parquet"
        events.to_parquet(grades_path, index=False)
        outputs[grades_path.name] = grades_path

    manifest = ArtifactManifest(
        artifact_name="game_latent_state_shadow_v2_validation",
        source_production_sha=git_sha(
            PROJECT_ROOT, "origin/production/wizardofodds-integration"
        ),
        source_production_ref="origin/production/wizardofodds-integration",
        code_sha=git_sha(PROJECT_ROOT),
        branch=git_branch(PROJECT_ROOT) or BRANCH,
        seed=int(args.seed),
        training_cutoff=f"season<{min(validation_seasons)}",
        seasons_used=list(validation_seasons),
        training_seasons=[int(s) for s in v2_spec.get("training_seasons", [])],
        validation_seasons=list(validation_seasons),
        input_fingerprints={
            RESIDUAL_DATASET_NAME: sha256_file(residual_path),
            FACTOR_SPEC_NAME: sha256_file(artifact_dir / FACTOR_SPEC_NAME),
            "shadow_v2_candidate.json": sha256_file(freeze_path),
        },
        parameters={
            "simulations_per_game": int(args.simulations),
            "games_per_season": int(args.games_per_season),
            "bootstrap_draws": int(args.bootstrap),
            "min_expected_minutes": float(args.min_expected_minutes),
            "marginal_probe_games_per_season": int(args.marginal_probe_games),
            "marginal_probe_simulations": int(args.marginal_probe_simulations),
            "models": list(MODELS),
        },
        notes=[
            "Run once, after the candidate was frozen.",
            "Marginals and the incumbent copula are refitted per validation "
            "season on seasons strictly before it.",
            "All five models see the same games, rosters and per-game seed, "
            "so every reported difference is paired.",
        ],
    )
    finalize_manifest(
        manifest,
        artifact_dir,
        outputs=dict(outputs),
        manifest_name=MANIFEST_NAME.replace(".json", ".validation.json"),
        checksum_name="SHA256SUMS.validation.txt",
    )

    console.rule("Framework gates A-H")
    for result in gates:
        console.print(result.describe())
    console.print(f"\n{report['framework_verdict']}")
    console.print(
        "\nlatent cross-player RMSE: "
        + "  ".join(
            f"{name}={latent['by_model'][name]['rmse']:.6f}"  # type: ignore[index]
            for name in FITTED_MODELS
        )
    )
    console.print(
        "count-space cross-player RMSE: "
        + "  ".join(
            f"{name}={dependence['cross_player_rmse'][name]:.6f}"  # type: ignore[index]
            for name in MODELS
        )
    )
    console.print(
        f"\nobserved-bucket drift against the control: "
        f"{report['control_pairing']['max_observed_bucket_drift']:.3e}"  # type: ignore[index]
    )
    console.print("VALIDATION_RUN_COUNT=1")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")


if __name__ == "__main__":
    main()
