"""Controlled production shadow: a replay of real slates through both models.

SHADOW ONLY. The incumbent remains the published authority. Nothing this
script writes reaches WizardOfOdds, and nothing it writes can promote
anything.

What it does
------------

For each shadowed season it refits the *production* marginals and the
*incumbent* copula on seasons strictly before it -- the same objects the live
pricing path builds, from ``scripts/07_fit_marginals.py`` and
``GaussianCopula`` -- then for each held-out game it:

1.  builds the full-game latent covariance from the frozen factor spec,
2.  answers arbitrary same-game joint queries (2- to 4-leg conjunctions
    spanning same-player, same-team, cross-team and mixed shapes),
3.  records the candidate, the incumbent and a cross-player independence
    reference side by side on one row, with immutable provenance attached,
4.  grades all three against the realized box score once it has them,
5.  records dependence and PSD/numerical diagnostics, and
6.  attempts a publish, so the refusal is in the artifact rather than assumed.

Why a replay rather than a live slate
-------------------------------------

There is no live feed in this environment. A replay of the graded holdout
seasons is the honest substitute: it exercises every runtime path end to end
on real production marginals and real rosters, and it produces the side-by-side
log and the grading report the live shadow would. It is explicitly *not*
additional model-selection evidence -- the selection is frozen, and the grades
here are the already-accepted holdout grades recomputed by a different caller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from nba_prop_quant.adaptive_training import CORE_SEED
from nba_prop_quant.research.game_latent_state.artifacts import (
    git_sha,
    sha256_file,
    write_checksums,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import SharedFactorLoadings
from nba_prop_quant.research.game_latent_state.factors import (
    incumbent_within_player_blocks,
)
from nba_prop_quant.research.game_latent_state.paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RESEARCH_DATA_ROOT,
    FACTOR_SPEC_NAME,
    RESIDUAL_DATASET_NAME,
    research_processed_dir,
)
from nba_prop_quant.research.game_latent_state.shadow_runtime import (
    PUBLISHING_SWITCH_PATH,
    ShadowConfig,
    ShadowPublishingDisabled,
    build_provenance,
    evaluate_shadow_game,
    grade_shadow_log,
    publish_shadow_probabilities,
    read_publishing_switch,
    shadow_log_frame,
)
from nba_prop_quant.research.game_latent_state.simulator import (
    SUPPORTED_STATS,
    GameRoster,
)
from nba_prop_quant.research.game_latent_state.validation import (
    analytic_marginals,
    generate_joint_events,
)

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SHADOW_DIR = DEFAULT_ARTIFACT_ROOT / "shadow"

OUTPUT_NAME = "production_shadow_report.json"
LOG_NAME = "production_shadow_log.csv"
CHECKSUM_NAME = "SHA256SUMS.shadow.txt"

FINAL_MODEL_SPEC_PATH = Path("research/final_model/final_model_spec.json")

#: The seasons the final model was graded on. Replaying them is a runtime
#: exercise, not new selection evidence.
DEFAULT_SHADOW_SEASONS = (2024, 2025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESEARCH_DATA_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=SHADOW_DIR)
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=list(DEFAULT_SHADOW_SEASONS)
    )
    parser.add_argument("--games-per-season", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=CORE_SEED)
    parser.add_argument("--min-expected-minutes", type=float, default=12.0)
    parser.add_argument("--events-per-family", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_root)
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_path = artifact_dir / FACTOR_SPEC_NAME
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    loadings = SharedFactorLoadings.from_payload(spec["loadings"])

    final_spec_path = PROJECT_ROOT / FINAL_MODEL_SPEC_PATH
    switch_path = PROJECT_ROOT / PUBLISHING_SWITCH_PATH

    residuals = pd.read_parquet(artifact_dir / RESIDUAL_DATASET_NAME)
    residuals["season"] = residuals["season"].astype(int)
    history = pd.read_parquet(
        research_processed_dir(args.data_root) / "oof_selected_means.parquet"
    )
    history["season"] = history["season"].astype(int)

    config = ShadowConfig(
        simulations=int(args.simulations),
        seed=int(args.seed),
        stats=SUPPORTED_STATS,
    )

    driver = _validation_driver()
    results = []
    realized: dict[str, int] = {}
    provenance = None
    per_season: dict[str, object] = {}

    for season in sorted(int(value) for value in args.seasons):
        console.rule(f"Shadowing season {season}")
        fit = driver.fit_season(history, season, SUPPORTED_STATS)
        console.print(
            f"production marginals + incumbent copula refit on seasons "
            f"{fit.training_seasons} ({fit.training_rows:,} rows)"
        )

        if provenance is None:
            # One provenance for the whole run: the model, the artifacts and
            # the code are the same across seasons, and a fingerprint that
            # moved per season would make the rows incomparable for no reason.
            provenance = build_provenance(
                loadings=loadings,
                factor_spec=spec,
                factor_spec_path=spec_path,
                marginal_source=(
                    "scripts/07_fit_marginals.py refit on seasons strictly "
                    "before each shadowed season (the production marginal path)"
                ),
                copula_source=(
                    "nba_prop_quant.copula.GaussianCopula refit on the same "
                    "training seasons (the incumbent)"
                ),
                simulations=config.simulations,
                seed=config.seed,
                final_model_spec_path=final_spec_path,
                code_sha=git_sha(PROJECT_ROOT),
            )
            console.print(f"provenance fingerprint {provenance.fingerprint}")

        season_rows = residuals.loc[residuals["season"] == season]
        game_ids = sorted(int(value) for value in season_rows["game_id"].unique())
        if args.games_per_season and len(game_ids) > args.games_per_season:
            picks = np.linspace(0, len(game_ids) - 1, args.games_per_season)
            game_ids = [game_ids[round(index)] for index in picks]

        season_events = 0
        season_games = 0
        for game_id in game_ids:
            prepared = _prepare_game(
                game_id, season_rows, history, float(args.min_expected_minutes)
            )
            if prepared is None:
                continue
            roster, observations = prepared

            reference = analytic_marginals(roster, fit.marginals)
            joint_events = generate_joint_events(
                observations,
                reference,
                SUPPORTED_STATS,
                game_id=game_id,
                seed=int(args.seed) + int(game_id),
                events_per_family=int(args.events_per_family),
                min_expected_minutes=float(args.min_expected_minutes),
            )
            if not joint_events:
                continue

            events = {}
            for index, event in enumerate(joint_events):
                event_id = f"{season}-{game_id}-{event.family}-{index}"
                events[event_id] = event.legs
                realized[event_id] = int(event.realized)

            within = incumbent_within_player_blocks(
                fit.copula, SUPPORTED_STATS, roster.frame["player_id"].astype(int)
            )
            result = evaluate_shadow_game(
                roster,
                events=events,
                marginals=fit.marginals,
                copula=fit.copula,
                loadings=loadings,
                provenance=provenance,
                config=config,
                within_player=within,
            )
            results.append(result)
            season_games += 1
            season_events += len(result.decisions)

        per_season[str(season)] = {
            "games_shadowed": season_games,
            "events_priced": season_events,
            "training_seasons": fit.training_seasons,
            "training_rows": fit.training_rows,
        }
        console.print(f"{season_games} games, {season_events} events")

    if provenance is None:
        raise SystemExit("no season produced a usable game")

    log = shadow_log_frame(results, provenance)
    grades = grade_shadow_log(log, realized)

    switch = read_publishing_switch(switch_path)
    publish_attempt: dict[str, object]
    try:
        publish_shadow_probabilities(
            [decision for result in results for decision in result.decisions], switch
        )
    except ShadowPublishingDisabled as error:
        publish_attempt = {"raised": type(error).__name__, "message": str(error)}
    else:  # pragma: no cover - unreachable while the switch is disabled
        raise SystemExit(
            "a publish attempt did not raise; the shadow must never publish"
        )

    report = {
        "study": "controlled_production_shadow_v1",
        "mode": "SHADOW_ONLY__INCUMBENT_REMAINS_THE_PUBLISHED_AUTHORITY",
        "replay_not_live": (
            "there is no live feed in this environment, so this is a replay of "
            "the graded holdout seasons through the live runtime paths; it is "
            "runtime evidence, not new model-selection evidence"
        ),
        "provenance": provenance.payload(),
        "provenance_fingerprint": provenance.fingerprint,
        "factor_spec_sha256": sha256_file(spec_path),
        "final_model_spec_sha256": (
            sha256_file(final_spec_path) if final_spec_path.exists() else None
        ),
        "config": {
            "simulations": config.simulations,
            "seed": config.seed,
            "stats": list(config.stats),
            "games_per_season": args.games_per_season,
            "events_per_family": args.events_per_family,
            "min_expected_minutes": args.min_expected_minutes,
        },
        "by_season": per_season,
        "games_shadowed": len(results),
        "games_that_fell_back": sum(1 for result in results if result.fell_back),
        "fallback_reasons": sorted(
            {result.failure_reason for result in results if result.failure_reason}
        ),
        "grading": grades,
        "numerical_diagnostics": _summarize_numerical(results),
        "dependence_diagnostics": _summarize_dependence(results),
        "publishing": {
            "switch": switch.payload(),
            "switch_path": str(PUBLISHING_SWITCH_PATH),
            "switch_sha256": sha256_file(switch_path) if switch_path.exists() else None,
            "publish_attempt": publish_attempt,
            "rows_published": 0,
        },
        "promotion_authority": grades["promotion_authority"],
    }

    _print_summary(report)

    log_path = output_dir / LOG_NAME
    log.to_csv(log_path, index=False)
    report_path = write_json(report, output_dir / OUTPUT_NAME)
    write_checksums([report_path, log_path], output_dir / CHECKSUM_NAME)
    console.print(f"\nwrote {report_path}\nwrote {log_path}")


def _validation_driver():
    """Load ``04_validate_shadow_v1.py`` for its per-season production refit.

    Reusing its ``fit_season`` is deliberate: "the marginals the shadow sees
    are the production marginals" is only true if there is exactly one
    definition of that refit, and this is it.
    """
    import importlib.util
    import sys

    path = PROJECT_ROOT / "research" / "game_latent_state" / "04_validate_shadow_v1.py"
    spec = importlib.util.spec_from_file_location("shadow_validation_driver", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_game(
    game_id: int,
    season_rows: pd.DataFrame,
    history: pd.DataFrame,
    min_minutes: float,
) -> tuple[GameRoster, pd.DataFrame] | None:
    observations = season_rows.loc[season_rows["game_id"] == game_id]
    observations = observations.loc[
        observations["expected_minutes"].fillna(0.0) >= min_minutes
    ]
    if observations["team_id"].nunique() != 2 or len(observations) < 6:
        return None

    frame = history.loc[
        (history["game_id"] == game_id)
        & history["player_id"].isin(observations["player_id"])
    ].copy()
    if len(frame) != len(observations):
        return None

    home = int(
        observations.loc[observations["is_home"].astype(bool), "team_id"].iloc[0]
        if observations["is_home"].astype(bool).any()
        else observations["team_id"].iloc[0]
    )
    frame = frame.sort_values(["team_id", "player_id"]).reset_index(drop=True)
    roster = GameRoster(
        game_id=int(game_id),
        home_team_id=home,
        frame=frame,
        stats=SUPPORTED_STATS,
        role_column="role_bucket" if "role_bucket" in frame else None,
    )
    return roster, observations


def _summarize_numerical(results) -> dict[str, object]:
    eigenvalues = [
        float(result.numerical["min_eigenvalue"])
        for result in results
        if "min_eigenvalue" in result.numerical
    ]
    deviations = [
        float(result.numerical["same_player_max_block_deviation"])
        for result in results
        if "same_player_max_block_deviation" in result.numerical
    ]
    return {
        "games_with_diagnostics": len(eigenvalues),
        "min_covariance_eigenvalue": min(eigenvalues) if eigenvalues else None,
        "psd_failures": sum(1 for value in eigenvalues if value < 0.0),
        "same_player_max_block_deviation": max(deviations) if deviations else None,
        "games_where_shared_shrink_engaged": sum(
            1
            for result in results
            if bool(result.numerical.get("shared_shrink_engaged", False))
        ),
    }


def _summarize_dependence(results) -> dict[str, object]:
    """Average each declared bucket over games, per arm and for the model."""
    implied: dict[str, list[float]] = {}
    simulated: dict[str, dict[str, list[float]]] = {}
    for result in results:
        if not result.dependence:
            continue
        for bucket, value in result.dependence["model_implied"].items():
            implied.setdefault(bucket, []).append(float(value))
        for model, values in result.dependence["simulated_by_model"].items():
            for bucket, value in values.items():
                if bucket.startswith("_"):
                    continue
                simulated.setdefault(model, {}).setdefault(bucket, []).append(
                    float(value)
                )
    return {
        "games_with_diagnostics": sum(1 for r in results if r.dependence),
        "model_implied": {
            bucket: float(np.mean(values)) for bucket, values in implied.items()
        },
        "simulated_mean_by_model": {
            model: {
                bucket: float(np.mean(values)) for bucket, values in buckets.items()
            }
            for model, buckets in simulated.items()
        },
    }


def _print_summary(report: dict[str, object]) -> None:
    grades = report["grading"]  # type: ignore[index]
    console.rule("Side-by-side grading")
    table = Table(show_header=True)
    table.add_column("model")
    table.add_column("events", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("log loss", justify="right")
    table.add_column("mean predicted", justify="right")
    for model, scores in grades["by_model"].items():  # type: ignore[index]
        if scores is None:
            continue
        table.add_row(
            model,
            f"{scores['events']:,}",
            f"{scores['brier']:.6f}",
            f"{scores['log_loss']:.6f}",
            f"{scores['mean_predicted']:.4f}",
        )
    console.print(table)
    console.print(f"base rate {grades['by_model']['incumbent']['base_rate']:.4f}")  # type: ignore[index]

    console.rule("Runtime status")
    console.print(f"served model: {grades['served_model']}")  # type: ignore[index]
    console.print(f"published rows: {report['publishing']['rows_published']}")  # type: ignore[index]
    console.print(f"promotion authority: {report['promotion_authority']}")
    console.print(f"candidate fallbacks: {grades['events_where_the_candidate_fell_back']}")  # type: ignore[index]
    console.print(
        f"min covariance eigenvalue: "
        f"{report['numerical_diagnostics']['min_covariance_eigenvalue']}"  # type: ignore[index]
    )
    console.print(
        f"same-player max block deviation: "
        f"{report['numerical_diagnostics']['same_player_max_block_deviation']}"  # type: ignore[index]
    )


if __name__ == "__main__":
    main()
