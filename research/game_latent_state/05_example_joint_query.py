"""Worked example of the joint same-game query contract.

SHADOW / RESEARCH ONLY. No sportsbook pricing, no vig, no publishing.

Takes one held-out game, builds it from the real production marginals and the
real fitted loadings, and answers a four-leg same-game conjunction of the form
the brief asks for:

    P(A PTS > l1, A AST > l2, B REB > l3, C PTS < l4)

The point is to show the full returned contract -- probability, Monte Carlo
standard error, simulation count, seed, game identifier, artifact version and
dependence-model version -- and to show the gap against the multiplied
marginals that the incumbent per-player path is limited to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console

from nba_prop_quant.adaptive_training import CORE_SEED
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
from nba_prop_quant.research.game_latent_state.query import PropLeg, evaluate_joint
from nba_prop_quant.research.game_latent_state.simulator import (
    SUPPORTED_STATS,
    GameRoster,
    simulate_game,
)
from nba_prop_quant.research.game_latent_state.validation import analytic_marginals

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_NAME = "example_joint_query.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESEARCH_DATA_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--simulations", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=CORE_SEED)
    parser.add_argument("--min-expected-minutes", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_root)

    spec = json.loads(
        (artifact_dir / FACTOR_SPEC_NAME).read_text(encoding="utf-8")
    )
    loadings = SharedFactorLoadings.from_payload(spec["loadings"])
    artifact_version = str(spec["spec_hash"])

    residuals = pd.read_parquet(artifact_dir / RESIDUAL_DATASET_NAME)
    residuals["season"] = residuals["season"].astype(int)
    history = pd.read_parquet(
        research_processed_dir(args.data_root) / "oof_selected_means.parquet"
    )
    history["season"] = history["season"].astype(int)

    season = max(int(value) for value in spec["validation_seasons"])

    console.rule(f"Refitting production marginals and copula on seasons < {season}")
    fit = _validation_driver().fit_season(history, season, SUPPORTED_STATS)

    season_rows = residuals.loc[residuals["season"] == season]

    roster, observations = _first_usable_game(
        season_rows, history, float(args.min_expected_minutes)
    )
    console.print(
        f"game {roster.game_id}: {len(roster.frame)} players with at least "
        f"{args.min_expected_minutes:g} expected minutes"
    )

    within = incumbent_within_player_blocks(
        fit.copula, SUPPORTED_STATS, roster.frame["player_id"].astype(int)
    )
    reference = analytic_marginals(roster, fit.marginals)

    simulation = simulate_game(
        roster,
        marginals=fit.marginals,
        loadings=loadings,
        within_player=within,
        simulations=int(args.simulations),
        seed=int(args.seed),
    )

    legs = _example_legs(roster, reference)
    result = evaluate_joint(simulation, legs, artifact_version=artifact_version)
    payload = result.to_payload()

    # The incumbent per-player path can only multiply marginals across
    # players, so the ratio below is exactly the correction this layer adds.
    product = payload["independent_product"]
    payload["dependence_lift_vs_independent_product"] = (
        float(payload["probability"]) / product if product else None
    )
    payload["realized_outcome"] = _grade(legs, observations)
    payload["legs_detail"] = [
        {
            "player_id": leg.player_id,
            "stat": leg.stat,
            "side": leg.side,
            "line": leg.line,
            "marginal_probability": marginal,
        }
        for leg, marginal in zip(
            legs, payload["leg_marginal_probabilities"], strict=True
        )
    ]

    console.rule("Joint query result")
    console.print_json(json.dumps(payload, indent=2))

    output = artifact_dir / OUTPUT_NAME
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"\nwrote {output}")


def _validation_driver():
    """Load the validation driver for its per-season production refit.

    The driver is a script, not a module, so it is loaded by path rather than
    imported. It has to be registered in ``sys.modules`` before execution or
    the dataclasses it defines cannot resolve their own module. Reusing its
    ``fit_season`` keeps one definition of "refit production on seasons
    strictly before S" instead of a second copy that could drift.
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


def _first_usable_game(
    season_rows: pd.DataFrame,
    history: pd.DataFrame,
    min_minutes: float,
) -> tuple[GameRoster, pd.DataFrame]:
    """The first held-out game with at least three usable players per team."""
    for game_id in sorted(int(value) for value in season_rows["game_id"].unique()):
        observations = season_rows.loc[
            (season_rows["game_id"] == game_id)
            & (season_rows["expected_minutes"].fillna(0.0) >= min_minutes)
        ]
        if observations["team_id"].nunique() != 2:
            continue
        counts = observations["team_id"].value_counts()
        if counts.min() < 3:
            continue

        frame = history.loc[
            (history["game_id"] == game_id)
            & history["player_id"].isin(observations["player_id"])
        ]
        if len(frame) != len(observations):
            continue
        frame = frame.sort_values(["team_id", "player_id"]).reset_index(drop=True)
        home = int(
            observations.loc[observations["is_home"].astype(bool), "team_id"].iloc[0]
            if observations["is_home"].astype(bool).any()
            else observations["team_id"].iloc[0]
        )
        roster = GameRoster(
            game_id=game_id,
            home_team_id=home,
            frame=frame,
            stats=SUPPORTED_STATS,
            role_column="role_bucket" if "role_bucket" in frame else None,
        )
        return roster, observations

    raise SystemExit("no held-out game had enough high-minute players")


def _example_legs(roster: GameRoster, reference) -> tuple[PropLeg, ...]:
    """A, A, B on one team and C on the other, lines at the median.

    Lines come from the predictive marginal only, placed at the median so each
    leg is a near-coin-flip and the joint effect is not swamped by one leg.
    """
    frame = roster.frame
    home = frame.loc[frame["team_id"] == roster.home_team_id, "player_id"].tolist()
    away = frame.loc[frame["team_id"] != roster.home_team_id, "player_id"].tolist()

    def line(player_id: int, stat: str) -> float:
        return reference[(int(player_id), stat)].quantile(0.5) + 0.5

    player_a, player_b = int(home[0]), int(home[1])
    player_c = int(away[0])
    return (
        PropLeg(player_a, "pts", "over", line(player_a, "pts")),
        PropLeg(player_a, "ast", "over", line(player_a, "ast")),
        PropLeg(player_b, "reb", "over", line(player_b, "reb")),
        PropLeg(player_c, "pts", "under", line(player_c, "pts")),
    )


def _grade(legs, observations: pd.DataFrame) -> dict[str, object]:
    lookup = {
        (int(row["player_id"]), stat): float(row[f"y_{stat}"])
        for _, row in observations.iterrows()
        for stat in SUPPORTED_STATS
    }
    detail = []
    satisfied = True
    for leg in legs:
        value = lookup.get((leg.player_id, leg.stat))
        hit = (
            None
            if value is None
            else (value > leg.line if leg.side == "over" else value < leg.line)
        )
        satisfied = satisfied and bool(hit)
        detail.append({"leg": leg.describe(), "realized": value, "hit": hit})
    return {"all_legs_hit": satisfied, "legs": detail}


if __name__ == "__main__":
    main()
