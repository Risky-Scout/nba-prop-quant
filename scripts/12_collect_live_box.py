from __future__ import annotations

import argparse
import time

import pandas as pd
from rich.console import Console

from nba_prop_quant.api import BDLClient
from nba_prop_quant.live import flatten_live_box_scores
from nba_prop_quant.settings import get_settings
from nba_prop_quant.storage import timestamped_jsonl_append

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:
        for iteration in range(args.iterations):
            raw = client.live_box_scores()
            flat = flatten_live_box_scores(raw)

            if not flat.empty:
                date = str(flat["date"].dropna().iloc[0])
                games = list(client.games(dates=[date]))
                lookup = {
                    (
                        int(game["home_team"]["id"]),
                        int(game["visitor_team"]["id"]),
                    ): int(game["id"])
                    for game in games
                }
                missing = flat["game_id"].isna()
                flat.loc[missing, "game_id"] = flat.loc[missing].apply(
                    lambda row: lookup.get(
                        (int(row["home_team_id"]), int(row["visitor_team_id"]))
                    ),
                    axis=1,
                )
                timestamped_jsonl_append(
                    flat.to_dict("records"),
                    settings.snapshot_dir / "live_box" / f"{date}.jsonl",
                    snapshot_type="live_box",
                )
                console.print(
                    f"iteration {iteration + 1}: captured {len(flat):,} player rows"
                )

            if iteration + 1 < args.iterations:
                time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
