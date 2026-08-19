from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BOX_FIELDS = [
    "min",
    "fgm",
    "fga",
    "fg_pct",
    "fg3m",
    "fg3a",
    "fg3_pct",
    "ftm",
    "fta",
    "ft_pct",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "turnover",
    "pf",
    "pts",
    "plus_minus",
]


def minutes_to_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    if ":" in text:
        minute, second = text.split(":", maxsplit=1)
        return float(minute) + float(second) / 60.0
    return float(text)


def normalize_players(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, sep=".")
    rename = {
        "team.id": "current_team_id",
        "team.abbreviation": "current_team_abbreviation",
    }
    return df.rename(columns=rename)


def normalize_games(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, sep=".")
    rename = {
        "home_team.id": "home_team_id",
        "visitor_team.id": "visitor_team_id",
        "home_team.abbreviation": "home_team_abbreviation",
        "visitor_team.abbreviation": "visitor_team_abbreviation",
    }
    df = df.rename(columns=rename)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def normalize_stats(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    for row in rows:
        player = row.get("player", {}) or {}
        team = row.get("team", {}) or {}
        game = row.get("game", {}) or {}
        flat = {field: row.get(field) for field in BOX_FIELDS}
        flat.update(
            {
                "stat_id": row.get("id"),
                "player_id": player.get("id"),
                "player_first_name": player.get("first_name"),
                "player_last_name": player.get("last_name"),
                "position": player.get("position"),
                "draft_year": player.get("draft_year"),
                "team_id": team.get("id"),
                "game_id": game.get("id"),
                "date": game.get("date"),
                "season": game.get("season"),
                "postseason": game.get("postseason"),
                "home_team_id": (game.get("home_team") or {}).get("id", game.get("home_team_id")),
                "visitor_team_id": (game.get("visitor_team") or {}).get(
                    "id", game.get("visitor_team_id")
                ),
            }
        )
        out.append(flat)
    df = pd.DataFrame(out)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["minutes"] = df["min"].map(minutes_to_float)
    return df


def normalize_advanced(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"player", "team", "game"}
        }
        flat["player_id"] = (row.get("player") or {}).get("id")
        flat["team_id"] = (row.get("team") or {}).get("id")
        flat["game_id"] = (row.get("game") or {}).get("id")
        flat["date"] = (row.get("game") or {}).get("date")
        flat["season"] = (row.get("game") or {}).get("season")
        out.append(flat)
    df = pd.DataFrame(out)
    if not df.empty and "date" in df:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def normalize_injuries(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = []
    for row in rows:
        player = row.get("player", {}) or {}
        team = player.get("team", {}) or {}
        out.append(
            {
                "player_id": player.get("id"),
                "team_id": team.get("id"),
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "status": row.get("status"),
                "description": row.get("description"),
                "return_date": row.get("return_date"),
            }
        )
    return pd.DataFrame(out)


def normalize_lineups(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = []
    for row in rows:
        player = row.get("player", {}) or {}
        team = row.get("team", {}) or {}
        out.append(
            {
                "lineup_id": row.get("id"),
                "game_id": row.get("game_id"),
                "starter": row.get("starter"),
                "listed_position": row.get("position"),
                "player_id": player.get("id"),
                "team_id": team.get("id", player.get("team_id")),
            }
        )
    return pd.DataFrame(out)


def normalize_props(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = []
    for row in rows:
        market = row.get("market", {}) or {}
        out.append(
            {
                "id": row.get("id"),
                "game_id": row.get("game_id"),
                "player_id": row.get("player_id"),
                "vendor": row.get("vendor"),
                "prop_type": row.get("prop_type"),
                "line_value": pd.to_numeric(row.get("line_value"), errors="coerce"),
                "market_type": market.get("type"),
                "over_odds": market.get("over_odds"),
                "under_odds": market.get("under_odds"),
                "milestone_odds": market.get("odds"),
                "opened_at": row.get("opened_at"),
                "updated_at": row.get("updated_at"),
            }
        )
    return pd.DataFrame(out)


def normalize_plays(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, sep=".")
    if "wallclock" in df:
        df["wallclock"] = pd.to_datetime(df["wallclock"], utc=True, errors="coerce")
    return df
