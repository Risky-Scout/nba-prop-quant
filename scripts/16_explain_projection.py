from __future__ import annotations

import argparse

import pandas as pd
from rich.console import Console

from nba_prop_quant.interpret import local_shap_explanation
from nba_prop_quant.model import ModelBundle
from nba_prop_quant.settings import get_settings

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument(
        "--target",
        choices=["pts", "reb", "ast", "stl", "blk", "fg3m"],
        required=True,
    )
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    projection_path = settings.processed_dir / "projections" / f"{args.date}.parquet"
    frame = pd.read_parquet(projection_path)
    hit = frame[frame["player_id"].eq(args.player_id)]
    if hit.empty:
        raise RuntimeError("Player is not in the saved slate projection")

    row = hit.iloc[0]
    bundle = ModelBundle.load(settings.nba_prop_model_dir / f"{args.target}.joblib")
    explanation = local_shap_explanation(bundle, row, top_n=args.top_n)

    console.print(
        {
            "player": row.get("player_name"),
            "target": args.target,
            "projected_mean": float(row[f"mu_{args.target}"]),
            "expected_minutes": float(row["expected_minutes"]),
        }
    )
    console.print(
        "\nSHAP contributions below are on the XGBoost model's raw score scale.\n"
    )
    console.print(explanation.to_string(index=False))


if __name__ == "__main__":
    main()
