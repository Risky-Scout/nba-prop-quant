from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from nba_prop_quant.api import BDLClient
from nba_prop_quant.ctmc import ctmc_over_under_push
from nba_prop_quant.live import (
    archived_snapshots_to_bins,
    flatten_live_box_scores,
    live_mean_projection,
    regulation_minutes_remaining,
    remaining_minutes_projection,
)
from nba_prop_quant.normalize import minutes_to_float
from nba_prop_quant.settings import get_settings

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", type=int, required=True)
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument("--target", choices=["pts", "reb", "ast", "stl", "blk", "fg3m"], required=True)
    parser.add_argument("--line", type=float, default=None)
    return parser.parse_args()


def load_live_archive(date: str) -> pd.DataFrame:
    settings = get_settings()
    path = settings.snapshot_dir / "live_box" / f"{date}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            envelope = json.loads(line)
            payload = dict(envelope["payload"])
            payload["captured_at"] = envelope["captured_at"]
            rows.append(payload)
    return pd.DataFrame(rows)


def latest_projection_for_game(game_id: int) -> pd.DataFrame:
    settings = get_settings()
    files = sorted((settings.processed_dir / "projections").glob("*.parquet"), reverse=True)
    for path in files:
        frame = pd.read_parquet(path)
        hit = frame[frame["game_id"].eq(game_id)]
        if not hit.empty:
            return hit
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    projections = latest_projection_for_game(args.game_id)
    if projections.empty:
        raise RuntimeError("No saved pregame projection found for this game")
    pregame = projections[projections["player_id"].eq(args.player_id)]
    if pregame.empty:
        raise RuntimeError("Player not found in saved pregame projection")
    pregame_row = pregame.iloc[0]

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:
        raw = client.live_box_scores()
        flat = flatten_live_box_scores(raw)
        if flat.empty:
            raise RuntimeError("No live box scores returned")

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

    current = flat[
        flat["game_id"].eq(args.game_id) & flat["player_id"].eq(args.player_id)
    ]
    if current.empty:
        raise RuntimeError("Player is not present in the current live box score")
    live_row = current.iloc[0]

    current_count = int(live_row[args.target] or 0)
    minutes_played = minutes_to_float(live_row["min"])
    expected_minutes = float(pregame_row["expected_minutes"])
    clock_remaining = regulation_minutes_remaining(
        int(live_row["period"] or 0),
        live_row["clock"],
    )
    score_margin = float(live_row["home_team_score"] - live_row["visitor_team_score"])
    remaining_minutes = remaining_minutes_projection(
        expected_minutes,
        minutes_played,
        score_margin=score_margin,
        regulation_minutes_remaining_value=clock_remaining,
    )

    pregame_mu = float(pregame_row[f"mu_{args.target}"])
    pregame_rate = pregame_mu / max(expected_minutes, 1.0)

    ingarch_models = {}
    ingarch_path = settings.nba_prop_model_dir / "live_ingarch.joblib"
    if ingarch_path.exists():
        ingarch_models = joblib.load(ingarch_path)
    ingarch = ingarch_models.get(args.target)

    recent_counts = None
    archive = load_live_archive(date)
    if not archive.empty:
        archive = archive[
            archive["game_id"].eq(args.game_id)
            & archive["player_id"].eq(args.player_id)
        ]
        if not archive.empty:
            binned = archived_snapshots_to_bins(
                archive, target=args.target, bin_seconds=60
            )
            recent_counts = binned["count"].to_numpy(dtype=float)[-8:]

    live_mean = live_mean_projection(
        current_count=current_count,
        remaining_minutes=remaining_minutes,
        pregame_rate_per_minute=pregame_rate,
        ingarch=ingarch,
        recent_counts_per_bin=recent_counts,
        bin_minutes=1.0,
    )
    console.print(
        {
            "current": current_count,
            "minutes_played": round(minutes_played, 2),
            "projected_remaining_minutes": round(remaining_minutes, 2),
            "live_mean": round(live_mean, 3),
        }
    )

    if args.line is not None:
        remaining_mean = max(live_mean - current_count, 0.0)
        p_over, p_under, p_push = ctmc_over_under_push(
            current_count=current_count,
            line=args.line,
            expected_remaining_count=remaining_mean,
        )
        console.print(
            {
                "line": args.line,
                "p_over_ctmc": round(p_over, 4),
                "p_under_ctmc": round(p_under, 4),
                "p_push_ctmc": round(p_push, 4),
            }
        )


if __name__ == "__main__":
    main()
