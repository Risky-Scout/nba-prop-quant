from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console

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

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe BALLDONTLIE NBA historical ingestion. "
            "Completed components are validated and skipped on rerun."
        )
    )
    parser.add_argument("--start-season", type=int, default=2001)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--include-advanced", action="store_true")
    parser.add_argument("--include-lineups", action="store_true")
    parser.add_argument("--include-plays", action="store_true")
    parser.add_argument("--include-opening-props", action="store_true")
    parser.add_argument("--skip-players", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload components even when valid local files already exist.",
    )
    return parser.parse_args()


def parquet_is_valid(
    path: Path,
    required_columns: set[str],
    allow_empty: bool = False,
) -> tuple[bool, int, str]:
    if not path.exists():
        return False, 0, "missing"

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return False, 0, f"unreadable: {exc}"

    missing = required_columns - set(frame.columns)
    if missing:
        return False, len(frame), f"missing columns: {sorted(missing)}"

    if not allow_empty and frame.empty:
        return False, 0, "empty"

    return True, len(frame), "ok"


def marker_path(
    raw_dir: Path,
    component: str,
    season: int,
) -> Path:
    return (
        raw_dir
        / "_resume_markers"
        / f"season={season}"
        / f"{component}.json"
    )


def write_marker(
    raw_dir: Path,
    component: str,
    season: int,
    payload: dict,
) -> None:
    path = marker_path(raw_dir, component, season)
    path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "component": component,
        "season": season,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }

    path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def marker_exists(
    raw_dir: Path,
    component: str,
    season: int,
) -> bool:
    return marker_path(raw_dir, component, season).exists()


def standard_component_valid(
    raw_dir: Path,
    season: int,
) -> tuple[bool, dict]:
    season_dir = raw_dir / "seasons" / f"season={season}"

    games_path = season_dir / "games.parquet"
    stats_path = season_dir / "stats.parquet"

    games_ok, games_rows, games_reason = parquet_is_valid(
        games_path,
        required_columns={
            "id",
            "date",
            "season",
            "home_team_id",
            "visitor_team_id",
        },
    )

    stats_ok, stats_rows, stats_reason = parquet_is_valid(
        stats_path,
        required_columns={
            "game_id",
            "player_id",
            "team_id",
            "minutes",
            "pts",
            "reb",
            "ast",
        },
    )

    payload = {
        "games_rows": games_rows,
        "stats_rows": stats_rows,
        "games_reason": games_reason,
        "stats_reason": stats_reason,
    }

    return games_ok and stats_ok, payload


def advanced_component_valid(
    raw_dir: Path,
    season: int,
) -> tuple[bool, dict]:
    path = (
        raw_dir
        / "advanced"
        / f"season={season}"
        / "advanced.parquet"
    )

    ok, rows, reason = parquet_is_valid(
        path,
        required_columns={
            "game_id",
            "player_id",
            "team_id",
        },
    )

    return ok, {
        "advanced_rows": rows,
        "reason": reason,
    }


def lineups_component_valid(
    raw_dir: Path,
    season: int,
) -> tuple[bool, dict]:
    path = (
        raw_dir
        / "lineups"
        / f"season={season}"
        / "lineups.parquet"
    )

    ok, rows, reason = parquet_is_valid(
        path,
        required_columns={
            "game_id",
            "player_id",
            "team_id",
        },
    )

    return ok, {
        "lineup_rows": rows,
        "reason": reason,
    }


def players_component_valid(
    raw_dir: Path,
) -> tuple[bool, dict]:
    path = raw_dir / "players.parquet"

    ok, rows, reason = parquet_is_valid(
        path,
        required_columns={
            "id",
            "first_name",
            "last_name",
        },
    )

    return ok, {
        "player_rows": rows,
        "reason": reason,
    }


def should_skip(
    valid: bool,
    force: bool,
) -> bool:
    return valid and not force


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if args.start_season > args.end_season:
        raise SystemExit(
            "--start-season must be less than or equal to --end-season"
        )

    console.rule("Resume-safe BDL ingestion")
    console.print(
        {
            "start_season": args.start_season,
            "end_season": args.end_season,
            "include_advanced": args.include_advanced,
            "include_lineups": args.include_lineups,
            "include_plays": args.include_plays,
            "include_opening_props": args.include_opening_props,
            "force": args.force,
        }
    )

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:

        # ------------------------------------------------------------
        # Player master
        # ------------------------------------------------------------
        if not args.skip_players:
            valid, info = players_component_valid(settings.raw_dir)

            if should_skip(valid, args.force):
                console.print(
                    f"[cyan]SKIP players[/cyan]: "
                    f"valid local file ({info['player_rows']:,} rows)"
                )
            else:
                console.print("[yellow]FETCH players[/yellow]")
                ingest_players(client, settings.raw_dir)

                valid, info = players_component_valid(settings.raw_dir)
                if not valid:
                    raise RuntimeError(
                        f"players.parquet failed validation after download: {info}"
                    )

                console.print(
                    f"[green]PASS players[/green]: "
                    f"{info['player_rows']:,} rows"
                )

        # ------------------------------------------------------------
        # Season loop
        # ------------------------------------------------------------
        for season in range(
            args.start_season,
            args.end_season + 1,
        ):
            console.rule(f"Season {season}")

            # Standard games + player box scores
            valid, info = standard_component_valid(
                settings.raw_dir,
                season,
            )

            if should_skip(valid, args.force):
                console.print(
                    f"[cyan]SKIP standard {season}[/cyan]: "
                    f"{info['games_rows']:,} games, "
                    f"{info['stats_rows']:,} stat rows"
                )
            else:
                console.print(
                    f"[yellow]FETCH standard {season}[/yellow]"
                )

                try:
                    ingest_season_games_and_stats(
                        client,
                        settings.raw_dir,
                        season,
                    )
                except Exception:
                    console.print(
                        f"[red]FAILED standard {season}[/red]. "
                        "Completed prior seasons remain on disk. "
                        "Rerun the same command to resume."
                    )
                    raise

                valid, info = standard_component_valid(
                    settings.raw_dir,
                    season,
                )

                if not valid:
                    raise RuntimeError(
                        f"Season {season} standard files failed "
                        f"post-download validation: {info}"
                    )

                write_marker(
                    settings.raw_dir,
                    "standard",
                    season,
                    info,
                )

                console.print(
                    f"[green]PASS standard {season}[/green]: "
                    f"{info['games_rows']:,} games, "
                    f"{info['stats_rows']:,} stat rows"
                )

            # Advanced stats: BDL availability begins at the configured
            # advanced_start_season.
            if (
                args.include_advanced
                and season >= settings.advanced_start_season
            ):
                valid, info = advanced_component_valid(
                    settings.raw_dir,
                    season,
                )

                if should_skip(valid, args.force):
                    console.print(
                        f"[cyan]SKIP advanced {season}[/cyan]: "
                        f"{info['advanced_rows']:,} rows"
                    )
                else:
                    console.print(
                        f"[yellow]FETCH advanced {season}[/yellow]"
                    )

                    try:
                        ingest_season_advanced(
                            client,
                            settings.raw_dir,
                            season,
                        )
                    except Exception:
                        console.print(
                            f"[red]FAILED advanced {season}[/red]. "
                            "Completed prior components remain on disk. "
                            "Rerun the same command to resume."
                        )
                        raise

                    valid, info = advanced_component_valid(
                        settings.raw_dir,
                        season,
                    )

                    if not valid:
                        raise RuntimeError(
                            f"Season {season} advanced file failed "
                            f"post-download validation: {info}"
                        )

                    write_marker(
                        settings.raw_dir,
                        "advanced",
                        season,
                        info,
                    )

                    console.print(
                        f"[green]PASS advanced {season}[/green]: "
                        f"{info['advanced_rows']:,} rows"
                    )

            # 2025+ lineups
            if (
                args.include_lineups
                and season >= settings.play_by_play_start_season
            ):
                valid, info = lineups_component_valid(
                    settings.raw_dir,
                    season,
                )

                if should_skip(valid, args.force):
                    console.print(
                        f"[cyan]SKIP lineups {season}[/cyan]: "
                        f"{info['lineup_rows']:,} rows"
                    )
                else:
                    console.print(
                        f"[yellow]FETCH lineups {season}[/yellow]"
                    )

                    try:
                        ingest_season_lineups(
                            client,
                            settings.raw_dir,
                            season,
                        )
                    except Exception:
                        console.print(
                            f"[red]FAILED lineups {season}[/red]. "
                            "Rerun the same command to resume."
                        )
                        raise

                    valid, info = lineups_component_valid(
                        settings.raw_dir,
                        season,
                    )

                    if not valid:
                        raise RuntimeError(
                            f"Season {season} lineups file failed "
                            f"post-download validation: {info}"
                        )

                    write_marker(
                        settings.raw_dir,
                        "lineups",
                        season,
                        info,
                    )

            # 2025+ plays can span many per-game files, so a successful
            # completion marker is the resume boundary.
            if (
                args.include_plays
                and season >= settings.play_by_play_start_season
            ):
                component = "plays"

                if (
                    marker_exists(
                        settings.raw_dir,
                        component,
                        season,
                    )
                    and not args.force
                ):
                    console.print(
                        f"[cyan]SKIP plays {season}[/cyan]: "
                        "completion marker exists"
                    )
                else:
                    console.print(
                        f"[yellow]FETCH plays {season}[/yellow]"
                    )

                    try:
                        ingest_season_plays(
                            client,
                            settings.raw_dir,
                            season,
                        )
                    except Exception:
                        console.print(
                            f"[red]FAILED plays {season}[/red]. "
                            "Existing per-game files remain on disk, but "
                            "the original play ingester will revisit the "
                            "season on rerun until a completion marker exists."
                        )
                        raise

                    write_marker(
                        settings.raw_dir,
                        component,
                        season,
                        {"status": "complete"},
                    )

            # Opening props can legitimately return no rows, so use a
            # successful completion marker instead of requiring a parquet file.
            if args.include_opening_props:
                component = "opening_props"

                if (
                    marker_exists(
                        settings.raw_dir,
                        component,
                        season,
                    )
                    and not args.force
                ):
                    console.print(
                        f"[cyan]SKIP opening props {season}[/cyan]: "
                        "completion marker exists"
                    )
                else:
                    console.print(
                        f"[yellow]FETCH opening props {season}[/yellow]"
                    )

                    try:
                        ingest_opening_props(
                            client,
                            settings.raw_dir,
                            season,
                        )
                    except Exception:
                        console.print(
                            f"[red]FAILED opening props {season}[/red]. "
                            "Rerun the same command to resume."
                        )
                        raise

                    write_marker(
                        settings.raw_dir,
                        component,
                        season,
                        {"status": "complete"},
                    )

    console.rule("COMPLETE")
    console.print(
        "[green]Resume-safe ingestion completed successfully.[/green]"
    )


if __name__ == "__main__":
    main()
