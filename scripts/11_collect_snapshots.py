from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from nba_prop_quant.api import BDLClient
from nba_prop_quant.settings import get_settings
from nba_prop_quant.snapshot_schedule import due_windows
from nba_prop_quant.storage import timestamped_jsonl_append
from nba_prop_quant.prospective_snapshot import (
    build_capture_id,
    canonical_records_sha256,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        required=True,
        help="YYYY-MM-DD",
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
    )
    parser.add_argument(
        "--now-utc",
    )
    parser.add_argument(
        "--grace-minutes",
        type=int,
        default=5,
    )
    return parser.parse_args()


def parse_now_utc(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(timezone.utc)

    text = raw

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    value = datetime.fromisoformat(text)

    if value.tzinfo is None:
        raise ValueError(
            "--now-utc must include timezone information"
        )

    return value.astimezone(timezone.utc)


def completed_window_ids(
    capture_run_path: Path,
) -> set[str]:
    if not capture_run_path.exists():
        return set()

    completed: set[str] = set()

    for line in capture_run_path.read_text(
        encoding="utf-8",
    ).splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row.get("snapshot_type") != "capture_run":
            continue

        payload = row.get("payload") or {}

        for window_id in payload.get(
            "window_ids",
            [],
        ):
            completed.add(
                str(window_id)
            )

    return completed


def main() -> None:
    args = parse_args()
    settings = get_settings()
    now_utc = parse_now_utc(
        args.now_utc
    )

    errors: list[dict[str, object]] = []

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:
        games = list(
            client.games(
                dates=[args.date]
            )
        )

        window_ids: list[str] = []
        due_game_ids: list[int] = []

        if args.scheduled:
            due = due_windows(
                games,
                now_utc,
                grace_minutes=args.grace_minutes,
            )

            capture_run_path = (
                settings.snapshot_dir
                / "capture_runs"
                / f"{args.date}.jsonl"
            )

            completed = completed_window_ids(
                capture_run_path
            )

            unseen = [
                row
                for row in due
                if row["window_id"]
                not in completed
            ]

            if not unseen:
                console.print(
                    "[cyan]No unseen scheduled "
                    "capture windows due[/cyan]"
                )
                return

            window_ids = sorted(
                {
                    str(
                        row["window_id"]
                    )
                    for row in unseen
                }
            )

            due_game_ids = sorted(
                {
                    int(
                        row["game_id"]
                    )
                    for row in unseen
                }
            )

        captured_at = (
            now_utc.isoformat()
        )

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
                    game.get(
                        "home_team",
                        {},
                    ),
                    game.get(
                        "visitor_team",
                        {},
                    ),
                )
                if team
                and team.get("id")
                is not None
            }
        )

        active_player_method = getattr(
            client,
            "active_players",
            None,
        )

        if active_player_method is None:
            active_players = []
        else:
            try:
                active_players = list(
                    active_player_method(
                        team_ids=team_ids
                    )
                )
            except RuntimeError as exc:
                active_players = []

                errors.append(
                    {
                        "component": "active_players",
                        "error": str(exc),
                    }
                )

                console.print(
                    "[yellow]Active-player capture "
                    f"unavailable[/yellow]: {exc}"
                )

        if games:
            timestamped_jsonl_append(
                games,
                settings.snapshot_dir
                / "games"
                / f"{args.date}.jsonl",
                snapshot_type="game",
                captured_at=captured_at,
            )

        if active_players:
            timestamped_jsonl_append(
                active_players,
                settings.snapshot_dir
                / "active_players"
                / f"{args.date}.jsonl",
                snapshot_type="active_player",
                captured_at=captured_at,
            )


        try:
            injuries = (
                list(
                    client.injuries(
                        team_ids=team_ids
                    )
                )
                if team_ids
                else []
            )
        except RuntimeError as exc:
            injuries = []

            errors.append(
                {
                    "component": "injuries",
                    "error": str(exc),
                }
            )

            console.print(
                "[yellow]Injury capture "
                f"unavailable[/yellow]: {exc}"
            )

        if injuries:
            timestamped_jsonl_append(
                injuries,
                settings.snapshot_dir
                / "injuries"
                / f"{args.date}.jsonl",
                snapshot_type="injury",
                captured_at=captured_at,
            )

        try:
            lineups = (
                list(
                    client.lineups(
                        game_ids
                    )
                )
                if game_ids
                else []
            )
        except RuntimeError as exc:
            lineups = []

            errors.append(
                {
                    "component": "lineups",
                    "error": str(exc),
                }
            )

            console.print(
                "[yellow]Lineup capture "
                f"unavailable[/yellow]: {exc}"
            )

        if lineups:
            timestamped_jsonl_append(
                lineups,
                settings.snapshot_dir
                / "lineups"
                / f"{args.date}.jsonl",
                snapshot_type="lineup",
                captured_at=captured_at,
            )

        props_count = 0
        all_props = []

        for game in games:
            game_id = int(
                game["id"]
            )

            try:
                props = (
                    client.live_player_props(
                        game_id
                    )
                )
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
                props_count += len(
                    props
                )

                all_props.extend(
                    props
                )

                timestamped_jsonl_append(
                    props,
                    settings.snapshot_dir
                    / "player_props"
                    / f"{args.date}.jsonl",
                    snapshot_type="live_player_prop",
                    captured_at=captured_at,
                )

        component_records = {
            "games": games,
            "active_players": active_players,
            "injuries": injuries,
            "lineups": lineups,
            "player_props": all_props,
        }

        component_sha256 = {
            name: canonical_records_sha256(
                records
            )
            for name, records in (
                component_records.items()
            )
        }

        capture_id = build_capture_id(
            date=args.date,
            captured_at=captured_at,
            window_ids=window_ids,
            due_game_ids=due_game_ids,
            component_sha256=component_sha256,
        )

        capture_run = {
            "date": args.date,
            "capture_reason": (
                "scheduled"
                if args.scheduled
                else "manual"
            ),
            "window_ids": window_ids,
            "due_game_ids": due_game_ids,
            "grace_minutes": (
                args.grace_minutes
                if args.scheduled
                else None
            ),
            "games_count": len(
                games
            ),
            "active_players_count": len(
                active_players
            ),
            "game_ids": game_ids,
            "team_ids": team_ids,
            "injuries_count": len(
                injuries
            ),
            "lineups_count": len(
                lineups
            ),
            "player_props_count": (
                props_count
            ),
            "error_count": len(
                errors
            ),
            "errors": errors,
            "component_sha256": component_sha256,
            "capture_id": capture_id,
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
            "[green]Captured point-in-time "
            "snapshot[/green]: "
            f"{len(games)} games, "
            f"{len(injuries)} injuries, "
            f"{len(lineups)} lineups, "
            f"{props_count} player props, "
            f"{len(errors)} errors, "
            f"{len(window_ids)} scheduled windows"
        )


if __name__ == "__main__":
    main()
