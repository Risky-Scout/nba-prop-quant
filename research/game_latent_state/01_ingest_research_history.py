"""Ingest the raw BDL history this shadow research branch needs.

Shadow/research only. Writes into a research-scoped data root so the
production ``data/raw`` tree, the Step 3B rolling state and the Step 3D
automation are never touched.

The window is deliberately narrower than the production
``history_start_season`` because this branch only needs enough seasons to
produce genuinely out-of-fold predictive distributions for the residual
dependence study. The reduced window is recorded in the artifact manifest.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console

from nba_prop_quant.api import BDLClient
from nba_prop_quant.ingest import (
    ingest_players,
    ingest_season_advanced,
    ingest_season_games_and_stats,
)
from nba_prop_quant.research.game_latent_state.paths import (
    DEFAULT_RESEARCH_DATA_ROOT,
    research_raw_dir,
)

console = Console()

DEFAULT_FIRST_SEASON = 2015
DEFAULT_LAST_SEASON = 2025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=DEFAULT_FIRST_SEASON)
    parser.add_argument("--last-season", type=int, default=DEFAULT_LAST_SEASON)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_RESEARCH_DATA_ROOT,
        help="Research-scoped data root. Never the production data root.",
    )
    parser.add_argument("--skip-advanced", action="store_true")
    parser.add_argument("--skip-players", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("BDL_API_KEY")
    if not api_key:
        raise SystemExit("BDL_API_KEY is not set")

    raw_dir = research_raw_dir(args.data_root)
    raw_dir.mkdir(parents=True, exist_ok=True)

    console.rule("Shadow research ingest")
    console.print(f"raw dir: {raw_dir}")
    console.print(f"seasons: {args.first_season}-{args.last_season}")

    with BDLClient(api_key=api_key, requests_per_minute=600) as client:
        if not args.skip_players:
            players_path = raw_dir / "players.parquet"
            if players_path.exists():
                console.print("players.parquet present; skipping")
            else:
                ingest_players(client, raw_dir)

        for season in range(args.first_season, args.last_season + 1):
            season_dir = raw_dir / "seasons" / f"season={season}"
            if (season_dir / "stats.parquet").exists() and (
                season_dir / "games.parquet"
            ).exists():
                console.print(f"season {season}: box history present; skipping")
            else:
                ingest_season_games_and_stats(client, raw_dir, season)

            if args.skip_advanced:
                continue

            advanced_path = (
                raw_dir / "advanced" / f"season={season}" / "advanced.parquet"
            )
            if advanced_path.exists():
                console.print(f"advanced {season}: present; skipping")
                continue
            ingest_season_advanced(client, raw_dir, season)

    console.rule("Shadow research ingest complete")


if __name__ == "__main__":
    main()
