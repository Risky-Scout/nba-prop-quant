from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .settings import Settings
from .storage import read_parquet_tree


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_history_stats(settings: Settings) -> pd.DataFrame:
    return read_parquet_tree(settings.raw_dir / "seasons").query(
        "minutes == minutes"
    )


def load_history_games(settings: Settings) -> pd.DataFrame:
    frames = []
    for file in sorted((settings.raw_dir / "seasons").rglob("games.parquet")):
        frames.append(pd.read_parquet(file))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_history_box_stats(settings: Settings) -> pd.DataFrame:
    frames = []
    for file in sorted((settings.raw_dir / "seasons").rglob("stats.parquet")):
        frames.append(pd.read_parquet(file))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_advanced(settings: Settings) -> pd.DataFrame:
    return read_parquet_tree(settings.raw_dir / "advanced")


def load_players(settings: Settings) -> pd.DataFrame:
    path = settings.raw_dir / "players.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def dynamic_params_path(settings: Settings) -> Path:
    return settings.nba_prop_model_dir / "dynamic_params.json"


def feature_matrix_path(settings: Settings) -> Path:
    return settings.processed_dir / "features.parquet"


def base_matrix_path(settings: Settings) -> Path:
    return settings.processed_dir / "base_player_games.parquet"


def oof_matrix_path(settings: Settings) -> Path:
    return settings.processed_dir / "oof_predictions.parquet"
