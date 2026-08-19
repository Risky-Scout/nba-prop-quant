from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import ADVANCED_FEATURES, TARGETS, add_dynamic_priors


def _final_ewm_by_player(
    frame: pd.DataFrame,
    value_col: str,
    span: int = 10,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for player_id, group in frame.groupby("player_id", sort=False):
        values = pd.to_numeric(group[value_col], errors="coerce")
        ewm = values.ewm(span=span, adjust=False, min_periods=1).mean()
        finite = ewm.dropna()
        if not finite.empty:
            result[int(player_id)] = float(finite.iloc[-1])
    return result


def _latest_team_context(stats: pd.DataFrame) -> dict[int, dict[str, float]]:
    sum_cols = [
        "pts", "reb", "ast", "stl", "blk", "fg3m", "fga", "fta", "oreb", "turnover"
    ]
    team = (
        stats.groupby(
            ["game_id", "date", "season", "team_id", "home_team_id", "visitor_team_id"],
            as_index=False,
        )[sum_cols]
        .sum(min_count=1)
        .sort_values(["date", "game_id", "team_id"])
    )
    team["opponent_id"] = np.where(
        team["team_id"].eq(team["home_team_id"]),
        team["visitor_team_id"],
        team["home_team_id"],
    )

    opp = team[
        ["game_id", "team_id", "pts", "reb", "ast", "stl", "blk", "fg3m", "fga", "fta", "oreb", "turnover"]
    ].rename(
        columns={
            "team_id": "opponent_id",
            "pts": "opp_pts",
            "reb": "opp_reb",
            "ast": "opp_ast",
            "stl": "opp_stl",
            "blk": "opp_blk",
            "fg3m": "opp_fg3m",
            "fga": "opp_fga",
            "fta": "opp_fta",
            "oreb": "opp_oreb",
            "turnover": "opp_turnover",
        }
    )
    team = team.merge(opp, on=["game_id", "opponent_id"], how="left")
    poss = (team["fga"] - team["oreb"] + team["turnover"] + 0.44 * team["fta"]).clip(lower=1)
    opp_poss = (
        team["opp_fga"] - team["opp_oreb"] + team["opp_turnover"] + 0.44 * team["opp_fta"]
    ).clip(lower=1)
    team["pace"] = 0.5 * (poss + opp_poss)
    team["off_rtg"] = 100.0 * team["pts"] / poss
    team["def_rtg"] = 100.0 * team["opp_pts"] / opp_poss
    team["reb_allowed"] = team["opp_reb"]
    team["ast_allowed"] = team["opp_ast"]
    team["stl_allowed"] = team["opp_stl"]
    team["blk_allowed"] = team["opp_blk"]
    team["fg3m_allowed"] = team["opp_fg3m"]

    metrics = {
        "pace": "team_pace_prior",
        "off_rtg": "team_off_rtg_prior",
        "def_rtg": "team_def_rtg_prior",
        "reb_allowed": "team_reb_allowed_prior",
        "ast_allowed": "team_ast_allowed_prior",
        "stl_allowed": "team_stl_allowed_prior",
        "blk_allowed": "team_blk_allowed_prior",
        "fg3m_allowed": "team_3pm_allowed_prior",
    }

    result: dict[int, dict[str, float]] = {}
    for team_id, group in team.groupby("team_id", sort=False):
        payload = {}
        group = group.sort_values(["date", "game_id"])
        for source, target in metrics.items():
            values = group[source].ewm(span=10, adjust=False, min_periods=1).mean()
            payload[target] = float(values.iloc[-1])
        result[int(team_id)] = payload
    return result


def _advanced_latest(advanced: pd.DataFrame) -> dict[int, dict[str, float]]:
    if advanced is None or advanced.empty:
        return {}

    advanced = advanced.sort_values(["player_id", "date", "game_id"])
    result: dict[int, dict[str, float]] = {}
    for player_id, group in advanced.groupby("player_id", sort=False):
        payload: dict[str, float] = {}
        for col in ADVANCED_FEATURES:
            if col not in group.columns:
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            ewm = values.ewm(span=10, adjust=False, min_periods=1).mean().dropna()
            if not ewm.empty:
                payload[f"adv_prior_{col}"] = float(ewm.iloc[-1])
        result[int(player_id)] = payload
    return result


def _dynamic_next_for_player(
    history: pd.DataFrame,
    future_game_id: int,
    future_date: pd.Timestamp,
    future_team_id: int,
    params: dict[str, Any],
) -> dict[str, float]:
    required = [
        "player_id", "game_id", "date", "team_id", "minutes",
        "pts", "reb", "ast", "stl", "blk", "fg3m"
    ]
    h = history[required].sort_values(["date", "game_id"]).copy()
    player_id = int(h["player_id"].iloc[-1]) if not h.empty else -1

    future = {
        "player_id": player_id,
        "game_id": int(future_game_id),
        "date": pd.Timestamp(future_date).normalize(),
        "team_id": int(future_team_id),
        "minutes": np.nan,
        "pts": np.nan,
        "reb": np.nan,
        "ast": np.nan,
        "stl": np.nan,
        "blk": np.nan,
        "fg3m": np.nan,
    }
    temp = pd.concat([h, pd.DataFrame([future])], ignore_index=True)
    temp = temp.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    temp["days_since_prev"] = temp.groupby("player_id")["date"].diff().dt.days
    previous_team = temp.groupby("player_id")["team_id"].shift(1)
    temp["team_change"] = (
        previous_team.notna() & temp["team_id"].ne(previous_team)
    ).astype(int)
    dynamic = add_dynamic_priors(temp, params)
    last = dynamic.iloc[-1]
    return {
        col: float(last[col])
        for col in dynamic.columns
        if col.startswith("decay_prior_") or col.startswith("kalman_prior_")
    }


def build_upcoming_slate_features(
    history_stats: pd.DataFrame,
    upcoming_games: pd.DataFrame,
    active_players: pd.DataFrame,
    advanced: pd.DataFrame,
    dynamic_params: dict[str, Any],
) -> pd.DataFrame:
    stats = history_stats.copy()
    stats["date"] = pd.to_datetime(stats["date"]).dt.normalize()
    upcoming = upcoming_games.copy()
    upcoming["date"] = pd.to_datetime(upcoming["date"]).dt.normalize()

    team_context = _latest_team_context(stats)
    advanced_latest = _advanced_latest(advanced)

    player_rate_maps = {}
    for stat in ["pts", "reb", "ast", "stl", "blk", "fg3m", "fga", "fg3a", "fta", "turnover", "pf"]:
        temp_col = f"_{stat}_rate"
        stats[temp_col] = pd.to_numeric(stats[stat], errors="coerce") / stats["minutes"].clip(lower=1.0)
        player_rate_maps[stat] = _final_ewm_by_player(stats, temp_col)
    prior_minutes = _final_ewm_by_player(stats, "minutes")

    last_rows = (
        stats.sort_values(["player_id", "date", "game_id"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .set_index("player_id")
    )
    career_games = stats.groupby("player_id").size().to_dict()

    if "current_team_id" not in active_players.columns and "team.id" in active_players.columns:
        active_players = active_players.rename(columns={"team.id": "current_team_id"})

    league_prior_minutes = float(stats["minutes"].mean())
    league_rate_prior = {
        stat: float(
            (pd.to_numeric(stats[stat], errors="coerce") / stats["minutes"].clip(lower=1.0)).mean()
        )
        for stat in TARGETS
    }

    rows = []
    for _, game in upcoming.iterrows():
        game_id = int(game["id"])
        game_date = pd.Timestamp(game["date"]).normalize()
        season = int(game["season"])
        home_team_id = int(game["home_team_id"])
        visitor_team_id = int(game["visitor_team_id"])

        for team_id, opponent_id, is_home in [
            (home_team_id, visitor_team_id, 1),
            (visitor_team_id, home_team_id, 0),
        ]:
            roster = active_players[
                pd.to_numeric(active_players["current_team_id"], errors="coerce").eq(team_id)
            ]
            for _, player in roster.iterrows():
                player_id = int(player["id"])
                historical = stats[stats["player_id"].eq(player_id)]
                last = last_rows.loc[player_id] if player_id in last_rows.index else None

                last_date = pd.Timestamp(last["date"]) if last is not None else pd.NaT
                days_since_prev = (
                    (game_date - last_date).days if pd.notna(last_date) else np.nan
                )
                prior_team_id = int(last["team_id"]) if last is not None else team_id
                team_change = int(prior_team_id != team_id)

                draft_year = pd.to_numeric(player.get("draft_year"), errors="coerce")
                games_prior = int(career_games.get(player_id, 0))
                if pd.notna(draft_year) and 0 <= season - float(draft_year) <= 30:
                    experience = season - float(draft_year)
                else:
                    experience = games_prior / 82.0

                position = str(player.get("position") or "").upper()
                row = {
                    "game_id": game_id,
                    "date": game_date,
                    "season": season,
                    "player_id": player_id,
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "home_team_id": home_team_id,
                    "visitor_team_id": visitor_team_id,
                    "is_home": is_home,
                    "postseason": int(bool(game.get("postseason", False))),
                    "days_since_prev": days_since_prev,
                    "days_rest": np.clip(days_since_prev - 1, 0, 14)
                    if pd.notna(days_since_prev)
                    else np.nan,
                    "b2b": int(days_since_prev == 1) if pd.notna(days_since_prev) else 0,
                    "team_change": team_change,
                    "career_games_prior": games_prior,
                    "experience_years": experience,
                    "pos_G": int("G" in position),
                    "pos_F": int("F" in position),
                    "pos_C": int("C" in position),
                    "prior_minutes10": prior_minutes.get(player_id, league_prior_minutes),
                }

                current_season_games = stats[
                    stats["team_id"].eq(team_id) & stats["season"].eq(season)
                ]["game_id"].nunique()
                row["season_progress"] = np.clip(current_season_games / 82.0, 0.0, 1.5)

                for stat, mapping in player_rate_maps.items():
                    row[f"prior_{stat}_rate10"] = mapping.get(
                        player_id,
                        league_rate_prior.get(stat, np.nan),
                    )

                row.update(team_context.get(team_id, {}))
                opp_context = team_context.get(opponent_id, {})
                for key, value in opp_context.items():
                    if key.startswith("team_"):
                        row[key.replace("team_", "opp_", 1)] = value

                row.update(advanced_latest.get(player_id, {}))

                if not historical.empty:
                    row.update(
                        _dynamic_next_for_player(
                            historical,
                            future_game_id=game_id,
                            future_date=game_date,
                            future_team_id=team_id,
                            params=dynamic_params,
                        )
                    )
                else:
                    row["decay_prior_min"] = dynamic_params.get("decay", {}).get(
                        "min", {}
                    ).get("global_prior", league_prior_minutes)
                    row["kalman_prior_min"] = dynamic_params.get("kalman", {}).get(
                        "min", {}
                    ).get("global_prior", league_prior_minutes)
                    for target in TARGETS:
                        name = f"{target}_rate"
                        row[f"decay_prior_{name}"] = dynamic_params.get("decay", {}).get(
                            name, {}
                        ).get("global_prior", league_rate_prior[target])
                        row[f"kalman_prior_{name}"] = dynamic_params.get("kalman", {}).get(
                            name, {}
                        ).get("global_prior", league_rate_prior[target])

                rows.append(row)

    return pd.DataFrame(rows)
