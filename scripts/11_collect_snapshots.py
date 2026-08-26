from __future__ import annotations

import argparse
from datetime import datetime, timezone

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
    captured_at = datetime.now(timezone.utc).isoformat()

    errors: list[dict[str, object]] = []

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:
        games = list(client.games(dates=[args.date]))

        game_ids = sorted(
            int(game["id"])
            for game in games
            if game.get("id") is not None
        )

        team_ids = sorted(
            {
                int(team["id"])
                for game in games
                for team in (
                    game.get("home_team", {}),
                    game.get("visitor_team", {}),
                )
                if team and team.get("id") is not None
            }
        )

        if games:
            timestamped_jsonl_append(
                games,
                settings.snapshot_dir / "games" / f"{args.date}.jsonl",
                snapshot_type="game",
                captured_at=captured_at,
            )

        try:
            injuries = list(
                client.injuries(team_ids=team_ids)
            ) if team_ids else []
        except RuntimeError as exc:
            injuries = []
            errors.append(
                {
                    "component": "injuries",
                    "error": str(exc),
                }
            )
            console.print(
                f"[yellow]Injury capture unavailable[/yellow]: {exc}"
            )

        if injuries:
            timestamped_jsonl_append(
                injuries,
                settings.snapshot_dir / "injuries" / f"{args.date}.jsonl",
                snapshot_type="injury",
                captured_at=captured_at,
            )

        try:
            lineups = list(
                client.lineups(game_ids)
            ) if game_ids else []
        except RuntimeError as exc:
            lineups = []
            errors.append(
                {
                    "component": "lineups",
                    "error": str(exc),
                }
            )
            console.print(
                f"[yellow]Lineup capture unavailable[/yellow]: {exc}"
            )

        if lineups:
            timestamped_jsonl_append(
                lineups,
                settings.snapshot_dir / "lineups" / f"{args.date}.jsonl",
                snapshot_type="lineup",
                captured_at=captured_at,
            )

        props_count = 0

        for game in games:
            game_id = int(game["id"])

            try:
                props = client.live_player_props(game_id)
            except RuntimeError as exc:
                props = []
                errors.append(
                    {
                        "component": "player_props",
                        "game_id": game_id,
                        "error": str(exc),
                    }
                )

            if props:
                props_count += len(props)

                timestamped_jsonl_append(
                    props,
                    settings.snapshot_dir
                    / "player_props"
                    / f"{args.date}.jsonl",
                    snapshot_type="live_player_prop",
                    captured_at=captured_at,
                )

        capture_run = {
            "date": args.date,
            "games_count": len(games),
            "game_ids": game_ids,
            "team_ids": team_ids,
            "injuries_count": len(injuries),
            "lineups_count": len(lineups),
            "player_props_count": props_count,
            "error_count": len(errors),
            "errors": errors,
        }

        timestamped_jsonl_append(
            [capture_run],
            settings.snapshot_dir
            / "capture_runs"
            / f"{args.date}.jsonl",
            snapshot_type="capture_run",
            captured_at=captured_at,
        )

        console.print(
            "[green]Captured point-in-time snapshot[/green]: "
            f"{len(games)} games, "
            f"{len(injuries)} injuries, "
            f"{len(lineups)} lineups, "
            f"{props_count} player props, "
            f"{len(errors)} errors"
        )


if __name__ == "__main__":
    main()
