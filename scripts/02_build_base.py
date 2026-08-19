from __future__ import annotations

from rich.console import Console

from nba_prop_quant.features import build_base_frame
from nba_prop_quant.pipeline import (
    base_matrix_path,
    load_advanced,
    load_history_box_stats,
    load_history_games,
    load_players,
)
from nba_prop_quant.settings import get_settings
from nba_prop_quant.storage import write_parquet_atomic

console = Console()


def main() -> None:
    settings = get_settings()
    stats = load_history_box_stats(settings)
    games = load_history_games(settings)
    players = load_players(settings)
    advanced = load_advanced(settings)

    console.print(f"games rows: {len(games):,}")
    console.print(f"stats rows: {len(stats):,}")
    console.print(f"advanced rows: {len(advanced):,}")

    base = build_base_frame(
        stats,
        players=players,
        advanced=advanced,
        games=games,
    )
    write_parquet_atomic(base, base_matrix_path(settings))
    console.print(
        f"[green]Wrote base matrix[/green] {base_matrix_path(settings)} "
        f"with {len(base):,} rows and {len(base.columns):,} columns"
    )


if __name__ == "__main__":
    main()
