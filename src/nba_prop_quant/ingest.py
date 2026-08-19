from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import pandas as pd
from rich.console import Console

from .api import BDLClient
from .normalize import (
    normalize_advanced,
    normalize_games,
    normalize_lineups,
    normalize_players,
    normalize_props,
    normalize_stats,
)
from .storage import upsert_parquet, write_parquet_atomic

T = TypeVar("T")
console = Console()


def chunks(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def ingest_players(client: BDLClient, raw_dir: Path) -> None:
    rows = list(client.players())
    df = normalize_players(rows)
    write_parquet_atomic(df, raw_dir / "players.parquet")
    console.print(f"[green]players[/green]: {len(df):,}")


def ingest_season_games_and_stats(
    client: BDLClient,
    raw_dir: Path,
    season: int,
) -> None:
    games = normalize_games(list(client.games(seasons=[season])))
    stats = normalize_stats(list(client.stats(seasons=[season], period=0)))

    season_dir = raw_dir / "seasons" / f"season={season}"
    write_parquet_atomic(games, season_dir / "games.parquet")
    write_parquet_atomic(stats, season_dir / "stats.parquet")
    console.print(
        f"[green]season {season}[/green]: {len(games):,} games, {len(stats):,} player-games"
    )


def ingest_season_advanced(
    client: BDLClient,
    raw_dir: Path,
    season: int,
) -> None:
    advanced = normalize_advanced(
        list(client.advanced_stats(seasons=[season], period=0))
    )
    path = raw_dir / "advanced" / f"season={season}" / "advanced.parquet"
    write_parquet_atomic(advanced, path)
    console.print(f"[green]advanced {season}[/green]: {len(advanced):,}")


def ingest_season_lineups(
    client: BDLClient,
    raw_dir: Path,
    season: int,
) -> None:
    game_path = raw_dir / "seasons" / f"season={season}" / "games.parquet"
    games = pd.read_parquet(game_path)
    rows = []
    for game_ids in chunks(games["id"].dropna().astype(int).tolist(), 25):
        rows.extend(client.lineups(game_ids))
    lineups = normalize_lineups(rows)
    path = raw_dir / "lineups" / f"season={season}" / "lineups.parquet"
    write_parquet_atomic(lineups, path)
    console.print(f"[green]lineups {season}[/green]: {len(lineups):,}")


def ingest_season_plays(
    client: BDLClient,
    raw_dir: Path,
    season: int,
) -> None:
    game_path = raw_dir / "seasons" / f"season={season}" / "games.parquet"
    games = pd.read_parquet(game_path)
    for index, game_id in enumerate(games["id"].dropna().astype(int), start=1):
        rows = client.plays(int(game_id))
        if not rows:
            continue
        frame = pd.json_normalize(rows, sep=".")
        path = raw_dir / "plays" / f"season={season}" / f"game_id={game_id}.parquet"
        write_parquet_atomic(frame, path)
        if index % 100 == 0:
            console.print(f"plays {season}: {index:,}/{len(games):,} games")


def ingest_opening_props(
    client: BDLClient,
    raw_dir: Path,
    season: int,
) -> None:
    game_path = raw_dir / "seasons" / f"season={season}" / "games.parquet"
    games = pd.read_parquet(game_path)
    frames = []
    for index, game_id in enumerate(games["id"].dropna().astype(int), start=1):
        rows = client.opening_player_props(int(game_id))
        if rows:
            frames.append(normalize_props(rows))
        if index % 100 == 0:
            console.print(f"opening props {season}: {index:,}/{len(games):,} games")
    if not frames:
        console.print(f"[yellow]No opening props returned for season {season}[/yellow]")
        return
    props = pd.concat(frames, ignore_index=True)
    path = raw_dir / "opening_props" / f"season={season}" / "props.parquet"
    upsert_parquet(props, path, key_columns=["id"])
    console.print(f"[green]opening props {season}[/green]: {len(props):,}")
