from __future__ import annotations

import argparse

from nba_prop_quant.api import BDLClient
from nba_prop_quant.ingest import (
    ingest_opening_props,
    ingest_players,
    ingest_season_advanced,
    ingest_season_games_and_stats,
    ingest_season_lineups,
    ingest_season_plays,
)
from nba_prop_quant.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2001)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--include-advanced", action="store_true")
    parser.add_argument("--include-lineups", action="store_true")
    parser.add_argument("--include-plays", action="store_true")
    parser.add_argument("--include-opening-props", action="store_true")
    parser.add_argument("--skip-players", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:
        if not args.skip_players:
            ingest_players(client, settings.raw_dir)

        for season in range(args.start_season, args.end_season + 1):
            ingest_season_games_and_stats(client, settings.raw_dir, season)

            if args.include_advanced and season >= settings.advanced_start_season:
                ingest_season_advanced(client, settings.raw_dir, season)

            if args.include_lineups and season >= settings.play_by_play_start_season:
                ingest_season_lineups(client, settings.raw_dir, season)

            if args.include_plays and season >= settings.play_by_play_start_season:
                ingest_season_plays(client, settings.raw_dir, season)

            if args.include_opening_props:
                ingest_opening_props(client, settings.raw_dir, season)


if __name__ == "__main__":
    main()
