from __future__ import annotations

import argparse

from rich.console import Console

from nba_prop_quant.api import BDLClient
from nba_prop_quant.settings import get_settings
from nba_prop_quant.storage import timestamped_jsonl_append

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:
        games = list(client.games(dates=[args.date]))
        team_ids = sorted(
            {
                int(team["id"])
                for game in games
                for team in (game.get("home_team", {}), game.get("visitor_team", {}))
                if team and team.get("id") is not None
            }
        )
        injuries = list(client.injuries(team_ids=team_ids))
        timestamped_jsonl_append(
            injuries,
            settings.snapshot_dir / "injuries" / f"{args.date}.jsonl",
            snapshot_type="injury",
        )

        for game in games:
            game_id = int(game["id"])
            try:
                props = client.live_player_props(game_id)
            except RuntimeError:
                props = []
            if props:
                timestamped_jsonl_append(
                    props,
                    settings.snapshot_dir / "player_props" / f"{args.date}.jsonl",
                    snapshot_type="live_player_prop",
                )

        console.print(
            f"[green]Captured snapshots[/green]: {len(injuries)} injury rows, {len(games)} games"
        )


if __name__ == "__main__":
    main()
