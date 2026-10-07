#!/usr/bin/env python
"""Item 5: the scalar dependence temperature. PRE-2024 DATA ONLY.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

``lambda`` scales every shared cross-player loading by ``sqrt(lambda)``, so
every cross-player covariance block is multiplied by exactly ``lambda`` while
each player's own block stays pinned at the incumbent's. ``lambda = 0`` is
cross-player conditional independence, which is the production incumbent's
joint behaviour; ``lambda = 1`` is the accepted repaired dependence model.

How the events are priced
-------------------------
Not by Monte Carlo. The quantity being compared across temperatures is of the
same order as the difference the brief already records between the accepted
repair and the production incumbent -- about 3e-5 in Brier -- and resolving a
difference that size by counting draws needs more draws than a grid search can
afford, so a simulated search would be reading its own noise.

Every leg is a threshold on a count, every count is a monotone transform of
one latent normal, so every conjunction is an orthant of a multivariate
normal and its probability is an orthant integral. ``temperature.py`` carries
the translation and the integral; this driver supplies the games, the events
and the correlation matrices. The correlation matrices still come from
``build_game_covariance``, so the same-player pinning, the shared-factor
shrink and the PSD projection all apply exactly as they do in simulation.

What is held out
----------------
Seasons 2024 and 2025 are filtered out on the first read and never
re-admitted. Each fold refits the production marginals and the incumbent
copula on seasons strictly before its target, fits the shared loadings on the
same seasons, and prices events in the target season alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nba_prop_quant.research.game_latent_state.artifacts import (
    git_sha,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import (
    build_game_covariance,
)
from nba_prop_quant.research.game_latent_state.factors import (
    incumbent_within_player_blocks,
)
from nba_prop_quant.research.game_latent_state.remediation import (
    PARSIMONY_TIE_BAND_SE,
    cluster_mean_and_se,
    paired_difference_se,
    temper_loadings,
)
from nba_prop_quant.research.game_latent_state.simulator import (
    SUPPORTED_STATS,
    GameRoster,
)
from nba_prop_quant.research.game_latent_state.temperature import (
    dimension_columns,
    independent_product,
    latent_orthant,
    orthant_probability,
)
from nba_prop_quant.research.game_latent_state.validation import (
    analytic_marginals,
    brier_score,
    generate_joint_events,
    log_loss,
)
from upstream_spec import UpstreamChoices, upstream_fit

console = Console()

STATS = SUPPORTED_STATS
HOLDOUT_SEASONS = (2024, 2025)

#: Pre-registered temperature grid. Deliberately tiny: the layer is one
#: scalar, the folds cannot resolve it finely, and offering a fine grid would
#: only invite reading fold noise as a preferred temperature. ``1.0`` is the
#: accepted repair and ``0.0`` is the incumbent's joint behaviour, so both
#: ends of the grid are models that already exist.
TEMPERATURE_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)

#: Target seasons of the forward folds. Each refits the production marginals
#: on the seasons before it, so the earliest usable target is the third
#: season of the record -- a marginal refit on one season is not a fair
#: stand-in for the production fit this is meant to sit beside.
FOLD_TARGET_SEASONS = (2022, 2023)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        type=Path,
        default=PROJECT_ROOT / "research/game_latent_state/oof_gaussian_residuals.parquet",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=PROJECT_ROOT
        / "data/research/game_latent_state/processed/oof_selected_means.parquet",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research/final_upstream_remediation",
    )
    parser.add_argument(
        "--count-residuals",
        type=Path,
        default=PROJECT_ROOT
        / "data/research/game_latent_state/processed/v2_bridge_count_residuals.parquet",
    )
    parser.add_argument(
        "--inner-selection",
        type=Path,
        default=None,
        help="inner_selection.json; defaults to the one under --artifact-root",
    )
    parser.add_argument("--games-per-fold", type=int, default=220)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--bridge-bootstrap", type=int, default=2000)
    parser.add_argument("--min-expected-minutes", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def assert_holdout_absent(frame: pd.DataFrame, where: str) -> pd.DataFrame:
    present = sorted(
        int(season)
        for season in frame["season"].unique()
        if int(season) in HOLDOUT_SEASONS
    )
    if present:
        raise SystemExit(f"{where} contains holdout seasons {present}")
    return frame


def load_validation_driver():
    """Load the unmodified validation driver for its production refit.

    Loading rather than copying keeps one definition of the refit: this
    driver's temperatures are selected against the same marginals and the same
    incumbent copula the confirmatory run will price them with, and it cannot
    drift from that code because it calls it.
    """
    path = PROJECT_ROOT / "research/game_latent_state/04_validate_shadow_v1.py"
    spec = importlib.util.spec_from_file_location("shadow_validation_driver", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def price_fold(
    target: int,
    history: pd.DataFrame,
    residuals: pd.DataFrame,
    loadings,
    driver,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Every generated event in one target season, priced at every temperature."""
    fit = driver.fit_season(history, target, STATS)
    console.print(
        f"  fold {target}: production refit on {fit.training_seasons} "
        f"({fit.training_rows:,} rows)"
    )

    tempered = {
        temperature: temper_loadings(loadings, temperature)
        for temperature in TEMPERATURE_GRID
    }

    season_rows = residuals.loc[residuals["season"] == target]
    game_ids = sorted(int(value) for value in season_rows["game_id"].unique())
    if args.games_per_fold and len(game_ids) > args.games_per_fold:
        picks = np.linspace(0, len(game_ids) - 1, args.games_per_fold)
        game_ids = [game_ids[round(index)] for index in picks]

    records: list[dict[str, object]] = []
    skipped = 0
    for game_id in game_ids:
        observations = season_rows.loc[season_rows["game_id"] == game_id]
        observations = observations.loc[
            observations["expected_minutes"].fillna(0.0) >= args.min_expected_minutes
        ]
        if observations["team_id"].nunique() != 2 or len(observations) < 6:
            skipped += 1
            continue

        roster_frame = history.loc[
            (history["game_id"] == game_id)
            & history["player_id"].isin(observations["player_id"])
        ].copy()
        if len(roster_frame) != len(observations):
            skipped += 1
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
            stats=STATS,
            role_column="role_bucket" if "role_bucket" in roster_frame else None,
        )
        within = incumbent_within_player_blocks(
            fit.copula, STATS, roster_frame["player_id"].astype(int)
        )
        reference = analytic_marginals(roster, fit.marginals)
        game_seed = int(args.seed) + int(game_id)

        dimensions = roster.dimensions()
        try:
            correlations = {
                temperature: build_game_covariance(
                    dimensions=dimensions,
                    loadings=value,
                    within_player=within,
                )
                for temperature, value in tempered.items()
            }
        except Exception as error:
            console.print(f"    game {game_id}: {error!r}")
            skipped += 1
            continue

        columns = dimension_columns(correlations[TEMPERATURE_GRID[0]])
        events = generate_joint_events(
            observations, reference, STATS, game_id=game_id, seed=game_seed
        )
        for position, event in enumerate(events):
            try:
                orthant = latent_orthant(event.legs, columns, reference)
            except KeyError:
                continue
            record: dict[str, object] = {
                "game_id": int(game_id),
                "family": event.family,
                "n_legs": int(event.n_legs),
                "realized": int(event.realized),
                "independent_product": independent_product(orthant),
            }
            for temperature, covariance in correlations.items():
                record[f"p_{temperature:.2f}"] = orthant_probability(
                    orthant,
                    covariance.correlation,
                    seed=game_seed * 1_000 + position,
                )
            records.append(record)

    console.print(
        f"  fold {target}: {len(records):,} events over "
        f"{len({r['game_id'] for r in records}):,} games ({skipped} games skipped)"
    )
    return records


def score(events: pd.DataFrame) -> dict[str, object]:
    """Game-clustered log loss and Brier, pooled and by leg count."""
    outcome = events["realized"].to_numpy(float)
    out: dict[str, object] = {"events": int(len(events)), "by_temperature": {}}
    for temperature in TEMPERATURE_GRID:
        column = f"p_{temperature:.2f}"
        probability = events[column].to_numpy(float)
        entry: dict[str, object] = {
            "log_loss": log_loss(probability, outcome),
            "brier": brier_score(probability, outcome),
            "mean_probability": float(np.mean(probability)),
        }
        for legs in sorted(events["n_legs"].unique()):
            mask = events["n_legs"] == legs
            entry[f"log_loss_{int(legs)}leg"] = log_loss(
                probability[mask.to_numpy()], outcome[mask.to_numpy()]
            )
            entry[f"brier_{int(legs)}leg"] = brier_score(
                probability[mask.to_numpy()], outcome[mask.to_numpy()]
            )
            entry[f"events_{int(legs)}leg"] = int(mask.sum())
        out["by_temperature"][f"{temperature:.2f}"] = entry  # type: ignore[index]
    return out


def per_game_log_loss(events: pd.DataFrame, temperature: float) -> pd.Series:
    """Per-game mean log loss, the clustering unit for every standard error."""
    probability = np.clip(events[f"p_{temperature:.2f}"].to_numpy(float), 1e-6, 1 - 1e-6)
    outcome = events["realized"].to_numpy(float)
    loss = -(outcome * np.log(probability) + (1.0 - outcome) * np.log1p(-probability))
    return pd.Series(loss, index=events.index).groupby(events["game_id"]).mean()


def select(events: pd.DataFrame) -> dict[str, object]:
    """Choose the temperature on game-clustered log loss over all leg counts.

    The tie band keeps ``lambda = 1`` -- the accepted repaired dependence
    model -- unless another temperature beats it by more than one standard
    error of the *paired* per-game difference. Pairing on games is what makes
    that standard error small enough to mean anything: the same games are
    priced at every temperature, so the between-game variance that dominates
    the level cancels in the difference.
    """
    per_game = {
        temperature: per_game_log_loss(events, temperature)
        for temperature in TEMPERATURE_GRID
    }
    pooled: dict[str, float] = {}
    pooled_se: dict[str, float] = {}
    for temperature, series in per_game.items():
        mean, standard_error = cluster_mean_and_se(
            series.to_numpy(float), series.index.to_numpy()
        )
        pooled[f"{temperature:.2f}"] = mean
        pooled_se[f"{temperature:.2f}"] = standard_error

    reference = "1.00"
    against_full: dict[str, dict[str, float]] = {}
    for temperature in TEMPERATURE_GRID:
        label = f"{temperature:.2f}"
        difference, standard_error = paired_difference_se(
            {str(game): float(value) for game, value in per_game[temperature].items()},
            {
                str(game): float(value)
                for game, value in per_game[float(reference)].items()
            },
        )
        against_full[label] = {"mean": difference, "standard_error": standard_error}

    best = min(pooled, key=lambda key: pooled[key])
    if best == reference:
        selected = reference
        reason = "the full dependence model has the best game-clustered log loss"
        improvement = 0.0
        improvement_se = 0.0
    else:
        margin = against_full[best]
        # ``against_full`` is candidate minus full and log loss is a loss, so a
        # candidate that beats the full model has a negative difference.
        improvement = -margin["mean"]
        improvement_se = margin["standard_error"]
        if improvement > PARSIMONY_TIE_BAND_SE * improvement_se:
            selected = best
            reason = (
                f"temperature {best} beats the full model by "
                f"{improvement:.3e} +/- {improvement_se:.3e} in game-clustered "
                "log loss, more than one standard error of the paired per-game "
                "difference"
            )
        else:
            selected = reference
            reason = (
                f"temperature {best} has the lower score but only by "
                f"{improvement:.3e} +/- {improvement_se:.3e}, inside one "
                "standard error of the paired per-game difference, so the full "
                "dependence model is retained"
            )

    return {
        "pooled_game_clustered_log_loss": pooled,
        "pooled_game_clustered_log_loss_se": pooled_se,
        "paired_difference_against_full_model": against_full,
        "best_unconditional": best,
        "best_improvement_over_full_model": improvement,
        "best_improvement_standard_error": improvement_se,
        "parsimony_tie_band_se": PARSIMONY_TIE_BAND_SE,
        "selected": selected,
        "reason": reason,
        "games": int(len(next(iter(per_game.values())))),
    }


def main() -> None:
    args = parse_args()
    console.rule("Final upstream remediation: dependence temperature (pre-2024 only)")

    choice_path = (
        Path(args.inner_selection)
        if args.inner_selection is not None
        else Path(args.artifact_root) / "inner_selection.json"
    )
    if not choice_path.exists():
        raise SystemExit(f"run 01_inner_selection.py first: {choice_path} is missing")
    inner = json.loads(choice_path.read_text(encoding="utf-8"))

    residuals = pd.read_parquet(args.residuals)
    residuals["season"] = residuals["season"].astype(int)
    residuals = assert_holdout_absent(
        residuals.loc[~residuals["season"].isin(HOLDOUT_SEASONS)].copy(),
        "residual frame",
    )
    history = pd.read_parquet(args.history)
    history["season"] = history["season"].astype(int)
    history = assert_holdout_absent(
        history.loc[~history["season"].isin(HOLDOUT_SEASONS)].copy(), "history frame"
    )

    driver = load_validation_driver()

    # The upstream choices are fixed before the temperature is searched, which
    # is what makes this one scalar rather than a joint search over six.
    # The fit is built at the full dependence model and the candidates come
    # from ``temper_loadings``, so every temperature shares one upstream fit.
    choices = UpstreamChoices.from_artifacts(inner, temperature=1.0)
    count_frame = pd.read_parquet(args.count_residuals)
    count_frame["season"] = count_frame["season"].astype(int)
    count_frame = assert_holdout_absent(
        count_frame.loc[~count_frame["season"].isin(HOLDOUT_SEASONS)].copy(),
        "count residual frame",
    )

    folds: dict[int, list[dict[str, object]]] = {}
    for target in FOLD_TARGET_SEASONS:
        training = assert_holdout_absent(
            residuals.loc[residuals["season"] < target], f"fold {target} training"
        )
        fit = upstream_fit(
            training,
            STATS,
            choices,
            count_frame=count_frame.loc[count_frame["season"] < target],
            bootstrap=args.bootstrap,
            bridge_bootstrap=args.bridge_bootstrap,
            seed=args.seed,
        )
        folds[target] = price_fold(
            target, history, residuals, fit.loadings, driver, args
        )

    events = pd.DataFrame(
        [{**record, "target_season": target} for target, rows in folds.items() for record in rows]
    )
    if events.empty:
        raise SystemExit("no events were priced")

    scored = score(events)
    decision = select(events)

    table = Table(title="dependence temperature, pooled over forward folds")
    table.add_column("lambda")
    table.add_column("log loss", justify="right")
    table.add_column("Brier", justify="right")
    for legs in (2, 3, 4):
        table.add_column(f"{legs}-leg LL", justify="right")
    table.add_column("vs lambda=1", justify="right")
    for temperature in TEMPERATURE_GRID:
        label = f"{temperature:.2f}"
        entry = scored["by_temperature"][label]  # type: ignore[index]
        margin = decision["paired_difference_against_full_model"][label]  # type: ignore[index]
        table.add_row(
            label,
            f"{entry['log_loss']:.6f}",
            f"{entry['brier']:.6f}",
            *[f"{entry[f'log_loss_{legs}leg']:.6f}" for legs in (2, 3, 4)],
            f"{margin['mean']:+.3e} +/- {margin['standard_error']:.3e}",
        )
    console.print(table)
    console.print(f"[bold]{decision['reason']}[/bold]")

    payload = {
        "title": "Final upstream remediation: dependence temperature",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "scope": "strictly pre-2024 chronological folds",
        "holdout_seasons_excluded": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "pricing": (
            "exact Gaussian orthant probabilities, not Monte Carlo: every leg "
            "is a threshold on one latent normal, so every conjunction is an "
            "orthant whose probability is an integral. The compared quantity "
            "is smaller than the Monte Carlo noise of any affordable draw "
            "count, so a simulated search would read its own noise."
        ),
        "temperature_grid": list(TEMPERATURE_GRID),
        "fold_target_seasons": list(FOLD_TARGET_SEASONS),
        "events_by_fold": {
            str(target): int(len(rows)) for target, rows in folds.items()
        },
        "events_by_legs": {
            str(int(legs)): int(count)
            for legs, count in events["n_legs"].value_counts().items()
        },
        "scores": scored,
        "selection": decision,
        "selected_temperature": float(decision["selected"]),  # type: ignore[arg-type]
        "upstream_choices": {
            "temporal": inner["item_1_temporal"]["selected"],
            "role_scale": inner["item_2_role_scale"]["selected"],
            "cross_team_prior": inner["item_3_cross_team"]["selected"],
            "transmission_cap": inner["item_4_transmission"]["selected_cap"],
            "uncertainty": inner["item_6_uncertainty"]["selected"],
        },
        "seed": int(args.seed),
        "code_sha": git_sha(PROJECT_ROOT),
    }
    path = write_json(payload, Path(args.artifact_root) / "dependence_temperature.json")

    console.rule("Dependence temperature")
    console.print(f"DEPENDENCE_TEMPERATURE={decision['selected']}")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")
    console.print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
