"""1. Turn completed games into information available before the next game."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import differential_evolution

# From src/nba_prop_quant/normalize.py

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
    rename = {"team.id": "current_team_id", "team.abbreviation": "current_team_abbreviation"}
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
        flat = {key: value for key, value in row.items() if key not in {"player", "team", "game"}}
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


# From src/nba_prop_quant/game_context.py

REGULAR_SEASON_GAME_TYPES = {
    "regular",
    "nba_cup_group",
    "nba_cup_quarterfinal",
    "nba_cup_semifinal",
    "nba_cup_other",
}

LEGACY_CUP_CHAMPIONSHIP_DATES = {2023: pd.Timestamp("2023-12-09"), 2024: pd.Timestamp("2024-12-17")}

LEGACY_PLAY_IN_WINDOWS = {
    2019: (pd.Timestamp("2020-08-15"), pd.Timestamp("2020-08-15")),
    2020: (pd.Timestamp("2021-05-18"), pd.Timestamp("2021-05-21")),
    2021: (pd.Timestamp("2022-04-12"), pd.Timestamp("2022-04-15")),
}


def _legacy_cup_championship_mask(games: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(games["date"], errors="coerce").dt.normalize()

    seasons = pd.to_numeric(games["season"], errors="coerce")

    postseason = games["postseason"].fillna(False).astype(bool)

    mask = pd.Series(False, index=games.index, dtype=bool)

    for season, championship_date in LEGACY_CUP_CHAMPIONSHIP_DATES.items():
        mask |= seasons.eq(season) & dates.eq(championship_date) & (~postseason)

    return mask


def _legacy_play_in_mask(games: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(games["date"], errors="coerce").dt.normalize()

    seasons = pd.to_numeric(games["season"], errors="coerce")

    mask = pd.Series(False, index=games.index, dtype=bool)

    for season, (start_date, end_date) in LEGACY_PLAY_IN_WINDOWS.items():
        mask |= seasons.eq(season) & dates.between(start_date, end_date, inclusive="both")

    return mask


def _normalize_stage(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    return str(value).strip().lower()


def _cup_game_type(stage: str) -> str | None:
    """
    Convert BALLDONTLIE ist_stage text into a stable model-facing label.
    """
    if not stage:
        return None

    if "championship" in stage:
        return "nba_cup_championship"

    if "quarterfinal" in stage:
        return "nba_cup_quarterfinal"

    if "semifinal" in stage:
        return "nba_cup_semifinal"

    if "group" in stage:
        return "nba_cup_group"

    return "nba_cup_other"


def infer_schedule_lengths(games: pd.DataFrame) -> dict[int, int]:
    """
    Infer each completed season's normal regular-season schedule length.

    Examples:
        82 games in a normal NBA season
        72 games in 2020-21
        66 games in 2011-12

    The calculation excludes:
        - playoff games
        - the NBA Cup Championship

    It intentionally retains:
        - ordinary regular-season games
        - NBA Cup Group Play
        - NBA Cup Quarterfinals
        - NBA Cup Semifinals
        - Play-In games

    Play-In teams therefore finish above the modal schedule length.
    The mode across all 30 teams identifies the normal schedule length
    without hard-coding 82.
    """
    required = {"season", "home_team_id", "visitor_team_id", "postseason"}

    missing = required - set(games.columns)

    if missing:
        raise ValueError(f"Games table is missing required columns: {sorted(missing)}")

    g = games.copy()

    if "ist_stage" not in g.columns:
        g["ist_stage"] = pd.NA

    g["season"] = pd.to_numeric(g["season"], errors="raise").astype(int)

    g["postseason"] = g["postseason"].fillna(False).astype(bool)

    g["_stage"] = g["ist_stage"].map(_normalize_stage)

    is_cup_championship = g["_stage"].str.contains(
        "championship", regex=False
    ) | _legacy_cup_championship_mask(g)

    is_legacy_play_in = _legacy_play_in_mask(g)

    candidates = g[(~g["postseason"]) & (~is_cup_championship) & (~is_legacy_play_in)].copy()

    home = candidates[["season", "home_team_id"]].rename(columns={"home_team_id": "team_id"})

    visitor = candidates[["season", "visitor_team_id"]].rename(
        columns={"visitor_team_id": "team_id"}
    )

    appearances = pd.concat([home, visitor], ignore_index=True)

    team_counts = (
        appearances.groupby(["season", "team_id"], as_index=False)
        .size()
        .rename(columns={"size": "games"})
    )

    schedule_lengths: dict[int, int] = {}

    for season, season_counts in team_counts.groupby("season", sort=True):
        values = season_counts["games"].astype(int)

        frequencies = values.value_counts()

        max_frequency = frequencies.max()

        modes = frequencies[frequencies.eq(max_frequency)].index.to_numpy(dtype=int)

        # If an unusual historical season produces a tie,
        # choose the median modal value and flag it later in QA.
        schedule_length = int(np.round(np.median(modes)))

        schedule_lengths[int(season)] = schedule_length

    return schedule_lengths


def classify_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Add a canonical competition regime to the BALLDONTLIE games table.

    Output game_type values:
        regular
        nba_cup_group
        nba_cup_quarterfinal
        nba_cup_semifinal
        nba_cup_championship
        nba_cup_other
        play_in
        playoffs

    Play-In detection is structural:

    1. Infer the completed season's normal schedule length from the
       modal number of non-playoff, non-Cup-Championship team games.

    2. Process games chronologically.

    3. Once a team has already completed the inferred number of
       standings-counting regular-season games, any subsequent
       non-playoff/non-Cup game is classified as Play-In.

    This avoids hard-coding individual Play-In game IDs.
    """
    required = {"id", "date", "season", "home_team_id", "visitor_team_id", "postseason"}

    missing = required - set(games.columns)

    if missing:
        raise ValueError(f"Games table is missing required columns: {sorted(missing)}")

    g = games.copy().reset_index(drop=True)

    if "ist_stage" not in g.columns:
        g["ist_stage"] = pd.NA

    g["date"] = pd.to_datetime(g["date"], errors="raise").dt.normalize()

    g["season"] = pd.to_numeric(g["season"], errors="raise").astype(int)

    g["home_team_id"] = pd.to_numeric(g["home_team_id"], errors="raise").astype(int)

    g["visitor_team_id"] = pd.to_numeric(g["visitor_team_id"], errors="raise").astype(int)

    g["postseason"] = g["postseason"].fillna(False).astype(bool)

    g["_stage"] = g["ist_stage"].map(_normalize_stage)

    g["_legacy_cup_championship"] = _legacy_cup_championship_mask(g)

    g["_legacy_play_in"] = _legacy_play_in_mask(g)

    schedule_lengths = infer_schedule_lengths(g)

    g["inferred_schedule_length"] = g["season"].map(schedule_lengths).astype("Int64")

    g["game_type"] = pd.NA

    g["home_regular_games_before"] = 0
    g["visitor_regular_games_before"] = 0

    for season in sorted(g["season"].unique()):
        season_rows = g[g["season"].eq(season)].sort_values(["date", "id"])

        schedule_length = schedule_lengths[int(season)]

        regular_games_played: dict[int, int] = {}

        for idx, row in season_rows.iterrows():
            home_id = int(row["home_team_id"])
            visitor_id = int(row["visitor_team_id"])

            home_before = regular_games_played.get(home_id, 0)

            visitor_before = regular_games_played.get(visitor_id, 0)

            g.at[idx, "home_regular_games_before"] = home_before

            g.at[idx, "visitor_regular_games_before"] = visitor_before

            stage = row["_stage"]

            cup_type = _cup_game_type(stage)

            if bool(row["_legacy_cup_championship"]):
                cup_type = "nba_cup_championship"

            if bool(row["_legacy_play_in"]):
                game_type = "play_in"

            elif bool(row["postseason"]):
                game_type = "playoffs"

            elif cup_type == "nba_cup_championship":
                game_type = "nba_cup_championship"

            elif cup_type is not None:
                game_type = cup_type

            elif int(season) in LEGACY_PLAY_IN_WINDOWS:
                game_type = "regular"

            elif home_before >= schedule_length or visitor_before >= schedule_length:
                game_type = "play_in"

            else:
                game_type = "regular"

            g.at[idx, "game_type"] = game_type

            # These games count toward the official regular-season
            # schedule. The Cup Championship, Play-In and playoffs do not.
            if game_type in REGULAR_SEASON_GAME_TYPES:
                regular_games_played[home_id] = home_before + 1

                regular_games_played[visitor_id] = visitor_before + 1

    g["counts_in_regular_season"] = g["game_type"].isin(REGULAR_SEASON_GAME_TYPES).astype(int)

    g["is_regular_season"] = g["counts_in_regular_season"].astype(int)

    g["is_nba_cup"] = g["game_type"].astype(str).str.startswith("nba_cup_").astype(int)

    g["is_nba_cup_championship"] = g["game_type"].eq("nba_cup_championship").astype(int)

    g["is_play_in"] = g["game_type"].eq("play_in").astype(int)

    g["is_playoffs"] = g["game_type"].eq("playoffs").astype(int)

    schedule_length = g["inferred_schedule_length"].astype(float).replace(0.0, np.nan)

    g["home_season_progress"] = (g["home_regular_games_before"] / schedule_length).clip(
        lower=0.0, upper=1.0
    )

    g["visitor_season_progress"] = (g["visitor_regular_games_before"] / schedule_length).clip(
        lower=0.0, upper=1.0
    )

    return (
        g.drop(columns=["_stage", "_legacy_cup_championship", "_legacy_play_in"])
        .sort_values(["date", "id"])
        .reset_index(drop=True)
    )


# From src/nba_prop_quant/decay.py

DECAY_BETA_BOUNDS = (0.85, 0.999999)


@njit(cache=True)
def _decay_predictions(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    values: np.ndarray,
    beta: float,
    global_prior: float,
    prior_strength: float,
) -> np.ndarray:
    n = len(values)
    predictions = np.empty(n, dtype=np.float64)

    last_player = -1
    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        player = int(player_ids[i])

        if player != last_player:
            last_player = player
            numerator = global_prior * prior_strength
            denominator = prior_strength
        else:
            days = max(float(days_since_prev[i]), 0.0)
            decay = beta**days
            numerator *= decay
            denominator *= decay

        predictions[i] = numerator / max(denominator, 1e-12)

        value = float(values[i])
        if np.isfinite(value):
            numerator += value
            denominator += 1.0

    return predictions


@dataclass
class DecayParams:
    beta: float
    global_prior: float
    prior_strength: float = 8.0


def decay_prior(
    player_ids: np.ndarray, days_since_prev: np.ndarray, values: np.ndarray, params: DecayParams
) -> np.ndarray:
    return _decay_predictions(
        player_ids.astype(np.int64),
        np.nan_to_num(days_since_prev, nan=0.0).astype(np.float64),
        values.astype(np.float64),
        float(params.beta),
        float(params.global_prior),
        float(params.prior_strength),
    )


def tune_decay_beta(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    values: np.ndarray,
    history_number: np.ndarray,
    beta_bounds: tuple[float, float] = DECAY_BETA_BOUNDS,
    prior_strength: float = 8.0,
    seed: int = 73,
) -> DecayParams:
    values = values.astype(np.float64)
    finite = np.isfinite(values)
    global_prior = float(np.nanmean(values[finite]))
    score_mask = finite & (history_number >= 3)

    def objective(x: np.ndarray) -> float:
        beta = float(x[0])
        pred = _decay_predictions(
            player_ids.astype(np.int64),
            np.nan_to_num(days_since_prev, nan=0.0).astype(np.float64),
            values,
            beta,
            global_prior,
            prior_strength,
        )
        error = pred[score_mask] - values[score_mask]
        return float(np.sqrt(np.mean(error * error)))

    result = differential_evolution(
        objective,
        bounds=[beta_bounds],
        seed=seed,
        polish=True,
        updating="immediate",
        workers=1,
        maxiter=50,
        popsize=8,
        tol=1e-5,
    )
    return DecayParams(
        beta=float(result.x[0]), global_prior=global_prior, prior_strength=prior_strength
    )


# From src/nba_prop_quant/kalman.py

KALMAN_Q_LOG_BOUNDS = (-16.0, 0.0)

KALMAN_R_LOG_BOUNDS = (-7.0, 6.0)


@njit(cache=True)
def _kalman_predictions(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    team_change: np.ndarray,
    values: np.ndarray,
    exposures: np.ndarray,
    global_prior: float,
    q: float,
    r: float,
    initial_variance: float,
    team_change_multiplier: float,
    epsilon: float,
) -> np.ndarray:
    n = len(values)
    predictions = np.empty(n, dtype=np.float64)

    last_player = -1
    mean_state = 0.0
    variance_state = initial_variance

    for i in range(n):
        player = int(player_ids[i])

        if player != last_player:
            last_player = player
            mean_state = np.log(max(global_prior, 0.0) + epsilon)
            variance_state = initial_variance
        else:
            days = max(float(days_since_prev[i]), 1.0)
            variance_state += q * days
            if team_change[i] > 0.5:
                variance_state *= team_change_multiplier

        predictions[i] = max(np.exp(mean_state) - epsilon, 0.0)

        value = float(values[i])
        if not np.isfinite(value):
            continue

        observation = np.log(max(value, 0.0) + epsilon)
        exposure = max(float(exposures[i]), 0.10)
        observation_variance = r / exposure

        gain = variance_state / (variance_state + observation_variance)
        mean_state = mean_state + gain * (observation - mean_state)
        variance_state = (1.0 - gain) * variance_state

    return predictions


@dataclass
class KalmanParams:
    global_prior: float
    q: float
    r: float
    initial_variance: float = 1.0
    team_change_multiplier: float = 2.0
    epsilon: float = 0.05


def kalman_prior(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    team_change: np.ndarray,
    values: np.ndarray,
    exposures: np.ndarray,
    params: KalmanParams,
) -> np.ndarray:
    return _kalman_predictions(
        player_ids.astype(np.int64),
        np.nan_to_num(days_since_prev, nan=0.0).astype(np.float64),
        np.nan_to_num(team_change, nan=0.0).astype(np.float64),
        values.astype(np.float64),
        exposures.astype(np.float64),
        float(params.global_prior),
        float(params.q),
        float(params.r),
        float(params.initial_variance),
        float(params.team_change_multiplier),
        float(params.epsilon),
    )


def tune_kalman(
    player_ids: np.ndarray,
    days_since_prev: np.ndarray,
    team_change: np.ndarray,
    values: np.ndarray,
    exposures: np.ndarray,
    history_number: np.ndarray,
    team_change_multiplier: float = 2.0,
    seed: int = 73,
    q_log_bounds: tuple[float, float] = KALMAN_Q_LOG_BOUNDS,
    r_log_bounds: tuple[float, float] = KALMAN_R_LOG_BOUNDS,
) -> KalmanParams:
    values = values.astype(np.float64)
    finite = np.isfinite(values)
    global_prior = float(np.nanmean(values[finite]))
    score_mask = finite & (history_number >= 3)

    def objective(x: np.ndarray) -> float:
        q = float(np.exp(x[0]))
        r = float(np.exp(x[1]))
        pred = _kalman_predictions(
            player_ids.astype(np.int64),
            np.nan_to_num(days_since_prev, nan=0.0).astype(np.float64),
            np.nan_to_num(team_change, nan=0.0).astype(np.float64),
            values,
            exposures.astype(np.float64),
            global_prior,
            q,
            r,
            1.0,
            team_change_multiplier,
            0.05,
        )
        error = pred[score_mask] - values[score_mask]
        return float(np.sqrt(np.mean(error * error)))

    result = differential_evolution(
        objective,
        bounds=[q_log_bounds, r_log_bounds],
        seed=seed,
        polish=True,
        updating="immediate",
        workers=1,
        maxiter=50,
        popsize=8,
        tol=1e-5,
    )
    return KalmanParams(
        global_prior=global_prior,
        q=float(np.exp(result.x[0])),
        r=float(np.exp(result.x[1])),
        initial_variance=1.0,
        team_change_multiplier=team_change_multiplier,
        epsilon=0.05,
    )


# From src/nba_prop_quant/features.py

TARGETS = ["pts", "reb", "ast", "stl", "blk", "fg3m"]

VOLUME_STATS = ["pts", "reb", "ast", "stl", "blk", "fg3m", "fga", "fg3a", "fta", "turnover", "pf"]

ADVANCED_FEATURES = [
    "usage_percentage",
    "estimated_usage_percentage",
    "assist_percentage",
    "rebound_percentage",
    "true_shooting_percentage",
    "effective_field_goal_percentage",
    "pace",
    "estimated_pace",
    "possessions",
    "touches",
    "passes",
    "secondary_assists",
    "rebound_chances_total",
    "deflections",
    "contested_shots",
    "defended_at_rim_fga",
    "defended_at_rim_fg_pct",
    "pct_fga_3pt",
    "pct_3pa",
    "free_throw_attempt_rate",
    "fouls_drawn",
    "speed",
    "distance",
    "turnover_ratio",
]

FEATURE_EXACT = [
    "is_home",
    "postseason",
    "is_regular_season",
    "is_nba_cup",
    "is_nba_cup_championship",
    "is_play_in",
    "is_playoffs",
    "days_rest",
    "b2b",
    "season_progress",
    "career_games_prior",
    "experience_years",
    "team_change",
    "pos_G",
    "pos_F",
    "pos_C",
    "team_pace_prior",
    "team_off_rtg_prior",
    "team_def_rtg_prior",
    "team_reb_allowed_prior",
    "team_ast_allowed_prior",
    "team_stl_allowed_prior",
    "team_blk_allowed_prior",
    "team_3pm_allowed_prior",
    "opp_pace_prior",
    "opp_off_rtg_prior",
    "opp_def_rtg_prior",
    "opp_reb_allowed_prior",
    "opp_ast_allowed_prior",
    "opp_stl_allowed_prior",
    "opp_blk_allowed_prior",
    "opp_3pm_allowed_prior",
]

FEATURE_PREFIXES = (
    "prior_",
    "decay_prior_",
    "kalman_prior_",
    "adv_prior_",
    "experience_curve_",
    "availability_",
)


def _lagged_ewm(frame: pd.DataFrame, group_col: str, value_col: str, span: int) -> pd.Series:
    return frame.groupby(group_col, sort=False)[value_col].transform(
        lambda s: s.shift(1).ewm(span=span, adjust=False, min_periods=1).mean()
    )


def _prepare_team_context(stats: pd.DataFrame) -> pd.DataFrame:
    sum_cols = ["pts", "reb", "ast", "stl", "blk", "fg3m", "fga", "fg3a", "fta", "oreb", "turnover"]
    team = (
        stats.groupby(
            ["game_id", "date", "season", "team_id", "home_team_id", "visitor_team_id"],
            as_index=False,
        )[sum_cols]
        .sum(min_count=1)
        .sort_values(["date", "game_id", "team_id"])
    )

    opponent = team[
        [
            "game_id",
            "team_id",
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
            "fg3m",
            "fga",
            "fta",
            "oreb",
            "turnover",
        ]
    ].rename(
        columns={
            "team_id": "opponent_id",
            "pts": "opp_actual_pts",
            "reb": "opp_actual_reb",
            "ast": "opp_actual_ast",
            "stl": "opp_actual_stl",
            "blk": "opp_actual_blk",
            "fg3m": "opp_actual_fg3m",
            "fga": "opp_actual_fga",
            "fta": "opp_actual_fta",
            "oreb": "opp_actual_oreb",
            "turnover": "opp_actual_turnover",
        }
    )

    team["opponent_id"] = np.where(
        team["team_id"].eq(team["home_team_id"]), team["visitor_team_id"], team["home_team_id"]
    )
    team = team.merge(opponent, on=["game_id", "opponent_id"], how="left")

    team["possessions_proxy"] = (
        team["fga"] - team["oreb"] + team["turnover"] + 0.44 * team["fta"]
    ).clip(lower=1.0)
    team["opp_possessions_proxy"] = (
        team["opp_actual_fga"]
        - team["opp_actual_oreb"]
        + team["opp_actual_turnover"]
        + 0.44 * team["opp_actual_fta"]
    ).clip(lower=1.0)
    team["pace_game"] = 0.5 * (team["possessions_proxy"] + team["opp_possessions_proxy"])
    team["off_rtg_game"] = 100.0 * team["pts"] / team["possessions_proxy"]
    team["def_rtg_game"] = 100.0 * team["opp_actual_pts"] / team["opp_possessions_proxy"]
    team["reb_allowed_game"] = team["opp_actual_reb"]
    team["ast_allowed_game"] = team["opp_actual_ast"]
    team["stl_allowed_game"] = team["opp_actual_stl"]
    team["blk_allowed_game"] = team["opp_actual_blk"]
    team["fg3m_allowed_game"] = team["opp_actual_fg3m"]

    context_map = {
        "pace_game": "team_pace_prior",
        "off_rtg_game": "team_off_rtg_prior",
        "def_rtg_game": "team_def_rtg_prior",
        "reb_allowed_game": "team_reb_allowed_prior",
        "ast_allowed_game": "team_ast_allowed_prior",
        "stl_allowed_game": "team_stl_allowed_prior",
        "blk_allowed_game": "team_blk_allowed_prior",
        "fg3m_allowed_game": "team_3pm_allowed_prior",
    }
    for source, target in context_map.items():
        team[target] = _lagged_ewm(team, "team_id", source, span=10)

    own_columns = ["game_id", "team_id"] + list(context_map.values())
    own = team[own_columns]

    opp = own.rename(
        columns={
            "team_id": "opponent_id",
            **{col: col.replace("team_", "opp_", 1) for col in context_map.values()},
        }
    )
    return (
        team[["game_id", "team_id", "opponent_id"]]
        .merge(own, on=["game_id", "team_id"], how="left")
        .merge(opp, on=["game_id", "opponent_id"], how="left")
    )


def _advanced_priors(advanced: pd.DataFrame) -> pd.DataFrame:
    if advanced is None or advanced.empty:
        return pd.DataFrame(columns=["game_id", "player_id"])

    adv = advanced.copy()
    adv["date"] = pd.to_datetime(adv["date"]).dt.normalize()
    adv = adv.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)

    keep = ["game_id", "player_id"]
    for col in ADVANCED_FEATURES:
        if col not in adv.columns:
            continue
        numeric = pd.to_numeric(adv[col], errors="coerce")
        adv[col] = numeric
        prior_col = f"adv_prior_{col}"
        adv[prior_col] = _lagged_ewm(adv, "player_id", col, span=10)
        keep.append(prior_col)

    return adv[keep].drop_duplicates(["game_id", "player_id"], keep="last")


def build_base_frame(
    stats: pd.DataFrame,
    players: pd.DataFrame | None = None,
    advanced: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = stats.copy()
    if df.empty:
        raise ValueError("stats data is empty")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if games is not None and not games.empty:
        classified_games = classify_games(games)

        context_columns = [
            "id",
            "game_type",
            "counts_in_regular_season",
            "is_regular_season",
            "is_nba_cup",
            "is_nba_cup_championship",
            "is_play_in",
            "is_playoffs",
            "inferred_schedule_length",
            "home_season_progress",
            "visitor_season_progress",
        ]

        game_context = classified_games[context_columns].rename(columns={"id": "game_id"})

        overlapping = [
            col for col in game_context.columns if col != "game_id" and col in df.columns
        ]

        if overlapping:
            df = df.drop(columns=overlapping)

        df = df.merge(game_context, on="game_id", how="left", validate="many_to_one")

        if df["game_type"].isna().any():
            missing_games = int(df.loc[df["game_type"].isna(), "game_id"].nunique())

            raise ValueError(
                f"{missing_games} stat games did not match " "the classified games table"
            )

    df = df[df["minutes"].fillna(0) > 0].copy()
    df = df.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)

    if players is not None and not players.empty:
        player_cols = [col for col in ["id", "draft_year", "position"] if col in players.columns]
        p = players[player_cols].rename(columns={"id": "player_id"})
        p = p.drop_duplicates("player_id")
        df = df.merge(p, on="player_id", how="left", suffixes=("", "_master"))
        if "draft_year_master" in df.columns:
            df["draft_year"] = df["draft_year"].fillna(df["draft_year_master"])
            df = df.drop(columns=["draft_year_master"])
        if "position_master" in df.columns:
            df["position"] = df["position"].fillna(df["position_master"])
            df = df.drop(columns=["position_master"])

    df["opponent_id"] = np.where(
        df["team_id"].eq(df["home_team_id"]), df["visitor_team_id"], df["home_team_id"]
    )
    df["is_home"] = df["team_id"].eq(df["home_team_id"]).astype(int)
    df["postseason"] = df["postseason"].fillna(False).astype(int)

    previous_date = df.groupby("player_id")["date"].shift(1)
    df["days_since_prev"] = (df["date"] - previous_date).dt.days
    df["days_rest"] = (df["days_since_prev"] - 1).clip(lower=0, upper=14)
    df["b2b"] = df["days_since_prev"].eq(1).astype(int)

    previous_team = df.groupby("player_id")["team_id"].shift(1)
    df["team_change"] = (previous_team.notna() & df["team_id"].ne(previous_team)).astype(int)

    df["player_game_number"] = df.groupby("player_id").cumcount()
    df["career_games_prior"] = df["player_game_number"]

    team_order = df[["game_id", "date", "season", "team_id"]].drop_duplicates()

    team_order = team_order.sort_values(["team_id", "season", "date", "game_id"])

    team_order["team_game_number"] = team_order.groupby(["team_id", "season"]).cumcount() + 1

    df = df.merge(
        team_order[["game_id", "team_id", "team_game_number"]],
        on=["game_id", "team_id"],
        how="left",
    )

    fallback_progress = ((df["team_game_number"] - 1) / 82.0).clip(0.0, 1.5)

    if {"home_season_progress", "visitor_season_progress"}.issubset(df.columns):

        classified_progress = np.where(
            df["is_home"].eq(1), df["home_season_progress"], df["visitor_season_progress"]
        )

        df["season_progress"] = (
            pd.to_numeric(pd.Series(classified_progress, index=df.index), errors="coerce")
            .fillna(fallback_progress)
            .clip(0.0, 1.0)
        )

    else:
        df["season_progress"] = fallback_progress

    draft_year = pd.to_numeric(df["draft_year"], errors="coerce")
    experience = pd.to_numeric(df["season"], errors="coerce") - draft_year
    fallback = df["career_games_prior"] / 82.0
    df["experience_years"] = experience.where(experience.between(0, 30), fallback)

    position = df["position"].fillna("").astype(str).str.upper()
    df["pos_G"] = position.str.contains("G").astype(int)
    df["pos_F"] = position.str.contains("F").astype(int)
    df["pos_C"] = position.str.contains("C").astype(int)

    for stat in VOLUME_STATS:
        if stat not in df.columns:
            continue
        df[f"{stat}_rate_game"] = pd.to_numeric(df[stat], errors="coerce") / df["minutes"].clip(
            lower=1.0
        )
        df[f"prior_{stat}_rate10"] = _lagged_ewm(df, "player_id", f"{stat}_rate_game", span=10)

    df["prior_minutes10"] = _lagged_ewm(df, "player_id", "minutes", span=10)

    team_context = _prepare_team_context(df)
    df = df.merge(team_context, on=["game_id", "team_id", "opponent_id"], how="left")

    advanced_for_priors = advanced.copy() if advanced is not None else pd.DataFrame()

    if not advanced_for_priors.empty:
        # BDL can return advanced rows for DNP / zero-minute
        # player-games. Restrict advanced history to the exact
        # played-player population used by the box-score model.
        played_keys = df[["game_id", "player_id"]].drop_duplicates()

        advanced_for_priors = advanced_for_priors.merge(
            played_keys, on=["game_id", "player_id"], how="inner", validate="one_to_one"
        )

    adv_prior = _advanced_priors(advanced_for_priors)

    if not adv_prior.empty:
        df = df.merge(adv_prior, on=["game_id", "player_id"], how="left", validate="one_to_one")

    return df.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)


def _params_from_dict_decay(payload: dict[str, Any], fallback_prior: float) -> DecayParams:
    return DecayParams(
        beta=float(payload.get("beta", 0.99)),
        global_prior=float(payload.get("global_prior", fallback_prior)),
        prior_strength=float(payload.get("prior_strength", 8.0)),
    )


def _params_from_dict_kalman(payload: dict[str, Any], fallback_prior: float) -> KalmanParams:
    return KalmanParams(
        global_prior=float(payload.get("global_prior", fallback_prior)),
        q=float(payload.get("q", 0.002)),
        r=float(payload.get("r", 0.4)),
        initial_variance=float(payload.get("initial_variance", 1.0)),
        team_change_multiplier=float(payload.get("team_change_multiplier", 2.0)),
        epsilon=float(payload.get("epsilon", 0.05)),
    )


def add_dynamic_priors(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    df = frame.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True).copy()
    pid = df["player_id"].to_numpy(dtype=np.int64)
    days = df["days_since_prev"].fillna(0).to_numpy(dtype=float)
    team_change = df["team_change"].fillna(0).to_numpy(dtype=float)

    series_map: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "min": (df["minutes"].to_numpy(dtype=float), np.ones(len(df), dtype=float))
    }
    for target in TARGETS:
        rate = (
            pd.to_numeric(df[target], errors="coerce") / df["minutes"].clip(lower=1.0)
        ).to_numpy(dtype=float)
        exposure = (df["minutes"].clip(lower=1.0) / 36.0).to_numpy(dtype=float)
        series_map[f"{target}_rate"] = (rate, exposure)

    for name, (values, exposure) in series_map.items():
        fallback_prior = float(np.nanmean(values))
        decay_payload = params.get("decay", {}).get(name, {})
        kalman_payload = params.get("kalman", {}).get(name, {})

        dparams = _params_from_dict_decay(decay_payload, fallback_prior)
        kparams = _params_from_dict_kalman(kalman_payload, fallback_prior)

        df[f"decay_prior_{name}"] = decay_prior(pid, days, values, dparams)
        df[f"kalman_prior_{name}"] = kalman_prior(pid, days, team_change, values, exposure, kparams)

    return df


def feature_columns(frame: pd.DataFrame, include_expected_minutes: bool = False) -> list[str]:
    columns = []
    for col in frame.columns:
        if col in FEATURE_EXACT or col.startswith(FEATURE_PREFIXES):
            columns.append(col)
    if include_expected_minutes and "expected_minutes" in frame.columns:
        columns.append("expected_minutes")
    return sorted(set(columns))


# From src/nba_prop_quant/slate.py

NBA_SLATE_TIMEZONE = "America/New_York"

NBA_SLATE_ZONE = ZoneInfo(NBA_SLATE_TIMEZONE)


class SlateDateError(ValueError):
    """A value could not be resolved to an unambiguous NBA slate date."""


class HistoryLeakageError(ValueError):
    """Historical inputs are not strictly older than the slate being predicted."""


def slate_date_from_schedule_date(value: Any) -> date:
    """Return the authoritative schedule date carried by a date-only value.

    BALLDONTLIE game payloads carry a ``date`` field that already *is* the NBA
    scheduled game date. It is preserved verbatim: treating it as UTC midnight
    and converting it to Eastern would silently shift every slate back a day.

    A midnight-normalized naive timestamp is accepted because it carries no
    time of day and is how this codebase already represents a slate date. A
    naive timestamp with a real time of day is refused rather than guessed at.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as error:
            raise SlateDateError(
                f"schedule date must be an exact YYYY-MM-DD value; got {value!r}"
            ) from error

    if isinstance(value, (datetime, pd.Timestamp)):
        moment = pd.Timestamp(value)

        if moment.tzinfo is not None:
            raise SlateDateError(
                "an offset-aware timestamp is not a schedule date; use "
                "slate_date_from_tip_timestamp so the NBA slate timezone is "
                "applied"
            )

        if moment != moment.normalize():
            raise SlateDateError(
                "a naive timestamp with a time of day is an ambiguous slate "
                f"date; supply an offset or a date-only value, got {value!r}"
            )

        return moment.date()

    raise SlateDateError(f"unsupported schedule date value: {value!r}")


def slate_date_from_tip_timestamp(value: Any) -> date:
    """Return the NBA slate date a tip-off timestamp belongs to.

    The timestamp must carry an offset. A naive timestamp is refused rather than
    assumed to be UTC or local, because guessing is what moves a late West-coast
    game onto the wrong slate.
    """
    if isinstance(value, str):
        text = value.strip()

        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"

        try:
            moment: datetime = datetime.fromisoformat(text)
        except ValueError as error:
            raise SlateDateError(f"tip timestamp is not ISO 8601: {value!r}") from error

    elif isinstance(value, pd.Timestamp):
        moment = value.to_pydatetime()

    elif isinstance(value, datetime):
        moment = value

    else:
        raise SlateDateError(f"unsupported tip timestamp value: {value!r}")

    if moment.tzinfo is None or moment.utcoffset() is None:
        raise SlateDateError(
            "tip timestamp must include timezone information so the NBA slate "
            f"date is unambiguous; got {value!r}"
        )

    return moment.astimezone(NBA_SLATE_ZONE).date()


def resolve_slate_date(value: Any) -> date:
    """Resolve either a schedule date or a tip timestamp to an NBA slate date.

    Only an offset-aware timestamp is converted through the NBA slate timezone.
    Date-only values are preserved exactly as scheduled.
    """
    if isinstance(value, (datetime, pd.Timestamp)):
        aware = pd.Timestamp(value).tzinfo is not None

        return (
            slate_date_from_tip_timestamp(value) if aware else slate_date_from_schedule_date(value)
        )

    if isinstance(value, str):
        text = value.strip()

        return (
            slate_date_from_schedule_date(text)
            if len(text) == 10
            else slate_date_from_tip_timestamp(text)
        )

    return slate_date_from_schedule_date(value)


def _latest_history_date(frame: pd.DataFrame, label: str, source: Any) -> pd.Timestamp:
    """Return max(date), refusing a frame whose dates cannot all be trusted."""
    if "date" not in getattr(frame, "columns", []):
        raise HistoryLeakageError(f"{label} has no date column: {source}")

    dates = pd.to_datetime(frame["date"], errors="coerce")

    unparseable = int(dates.isna().sum())

    if unparseable:
        raise HistoryLeakageError(f"{label} has {unparseable} unparseable date value(s): {source}")

    latest = dates.dt.normalize().max()

    if pd.isna(latest):
        raise HistoryLeakageError(f"{label} has no usable date: {source}")

    return latest


def assert_history_precedes_slate(
    frame: pd.DataFrame, label: str, slate_date: Any, source: Any = ""
) -> pd.Timestamp:
    """Refuse historical inputs that are not strictly older than the slate.

    Training consumes a shifted lagged EWM state while live inference consumes
    the latest historical EWM state. The two are equivalent only while history
    ends strictly before the slate date, so this is a train/serve-skew guard as
    much as a leakage guard.

    Returns the observed maximum date when the inputs are safe.
    """
    required = pd.Timestamp(resolve_slate_date(slate_date)).normalize()

    latest = _latest_history_date(frame, label, source)

    if latest >= required:
        raise HistoryLeakageError(
            f"{label} would leak same-or-later-day results into the "
            f"information set: max date {latest.date()} is not strictly "
            f"before slate date {required.date()}"
        )

    return latest


def slate_date_of_upcoming_games(upcoming_games: pd.DataFrame) -> pd.Timestamp:
    """Return the slate date the upcoming games belong to.

    The earliest scheduled game date is used, because history must already be
    complete before the first tip of the slate. BALLDONTLIE supplies this as a
    date field, so it is the authoritative schedule date and is not reinterpreted
    through any timezone.
    """
    if "date" not in getattr(upcoming_games, "columns", []):
        raise SlateDateError("upcoming games have no date column")

    dates = pd.to_datetime(upcoming_games["date"], errors="coerce")

    earliest = dates.dt.normalize().min()

    if pd.isna(earliest):
        raise SlateDateError("upcoming games have no usable date")

    return earliest


def assert_slate_inputs_precede_slate(
    history_stats: pd.DataFrame, advanced: pd.DataFrame, upcoming_games: pd.DataFrame
) -> pd.Timestamp:
    """Fail closed when slate inputs are not strictly older than the slate.

    This runs inside the slate builder rather than in the calling script so the
    check cannot be bypassed by a caller and so every consumer of upcoming-slate
    features inherits it. It is deliberately placed before any feature is
    derived, which puts it before all model inference.

    Only the historical inputs are checked. upcoming_games legitimately holds
    future scheduled rows and is the source of the slate date itself.
    """
    slate_date = slate_date_of_upcoming_games(upcoming_games)

    assert_history_precedes_slate(history_stats, "historical stats", slate_date)

    if not advanced.empty:
        assert_history_precedes_slate(advanced, "historical advanced", slate_date)

    return slate_date


def _final_ewm_by_player(frame: pd.DataFrame, value_col: str, span: int = 10) -> dict[int, float]:
    result: dict[int, float] = {}
    for player_id, group in frame.groupby("player_id", sort=False):
        values = pd.to_numeric(group[value_col], errors="coerce")
        ewm = values.ewm(span=span, adjust=False, min_periods=1).mean()
        finite = ewm.dropna()
        if not finite.empty:
            result[int(player_id)] = float(finite.iloc[-1])
    return result


def _latest_team_context(stats: pd.DataFrame) -> dict[int, dict[str, float]]:
    sum_cols = ["pts", "reb", "ast", "stl", "blk", "fg3m", "fga", "fta", "oreb", "turnover"]
    team = (
        stats.groupby(
            ["game_id", "date", "season", "team_id", "home_team_id", "visitor_team_id"],
            as_index=False,
        )[sum_cols]
        .sum(min_count=1)
        .sort_values(["date", "game_id", "team_id"])
    )
    team["opponent_id"] = np.where(
        team["team_id"].eq(team["home_team_id"]), team["visitor_team_id"], team["home_team_id"]
    )

    opp = team[
        [
            "game_id",
            "team_id",
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
            "fg3m",
            "fga",
            "fta",
            "oreb",
            "turnover",
        ]
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
        "player_id",
        "game_id",
        "date",
        "team_id",
        "minutes",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "fg3m",
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
    temp["team_change"] = (previous_team.notna() & temp["team_id"].ne(previous_team)).astype(int)
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

    assert_slate_inputs_precede_slate(stats, advanced, upcoming)

    team_context = _latest_team_context(stats)
    advanced_latest = _advanced_latest(advanced)

    player_rate_maps = {}
    for stat in ["pts", "reb", "ast", "stl", "blk", "fg3m", "fga", "fg3a", "fta", "turnover", "pf"]:
        temp_col = f"_{stat}_rate"
        stats[temp_col] = pd.to_numeric(stats[stat], errors="coerce") / stats["minutes"].clip(
            lower=1.0
        )
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
                days_since_prev = (game_date - last_date).days if pd.notna(last_date) else np.nan
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
                    "days_rest": (
                        np.clip(days_since_prev - 1, 0, 14) if pd.notna(days_since_prev) else np.nan
                    ),
                    "b2b": int(days_since_prev == 1) if pd.notna(days_since_prev) else 0,
                    "team_change": team_change,
                    # build_base_frame assigns player_game_number as the
                    # player's 0-indexed cumcount and then defines
                    # career_games_prior as exactly that value. The upcoming
                    # game sits after all history, so its cumcount is the
                    # number of games already played. The fitted Gate 3 role
                    # model asks for the first name, so both are exposed.
                    "player_game_number": games_prior,
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
                # build_base_frame numbers each team's games within a season
                # from 1, so the upcoming game is one past those already
                # played, and season_progress stays (team_game_number - 1)/82.
                row["team_game_number"] = current_season_games + 1
                row["season_progress"] = np.clip(current_season_games / 82.0, 0.0, 1.5)

                for stat, mapping in player_rate_maps.items():
                    row[f"prior_{stat}_rate10"] = mapping.get(
                        player_id, league_rate_prior.get(stat, np.nan)
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
                    row["decay_prior_min"] = (
                        dynamic_params.get("decay", {})
                        .get("min", {})
                        .get("global_prior", league_prior_minutes)
                    )
                    row["kalman_prior_min"] = (
                        dynamic_params.get("kalman", {})
                        .get("min", {})
                        .get("global_prior", league_prior_minutes)
                    )
                    for target in TARGETS:
                        name = f"{target}_rate"
                        row[f"decay_prior_{name}"] = (
                            dynamic_params.get("decay", {})
                            .get(name, {})
                            .get("global_prior", league_rate_prior[target])
                        )
                        row[f"kalman_prior_{name}"] = (
                            dynamic_params.get("kalman", {})
                            .get(name, {})
                            .get("global_prior", league_rate_prior[target])
                        )

                rows.append(row)

    return pd.DataFrame(rows)


# From src/nba_prop_quant/availability.py

OUT_PATTERNS = ("out", "inactive", "suspended")


def _is_out(status: object) -> bool:
    text = str(status or "").strip().lower()
    return any(pattern in text for pattern in OUT_PATTERNS)


def apply_current_injury_adjustment(
    slate: pd.DataFrame,
    injuries: pd.DataFrame,
    expected_minutes_col: str = "expected_minutes",
    max_minutes: float = 42.0,
) -> pd.DataFrame:
    """
    Production-only availability adjustment.

    BDL's injuries endpoint is a current snapshot, not a historical snapshot archive.
    Therefore this function should not be used in a historical backtest unless the
    injury record was captured before tip-off by your own collector.
    """
    df = slate.copy()
    injury_map = {}
    if injuries is not None and not injuries.empty:
        injury_map = (
            injuries.dropna(subset=["player_id"]).set_index("player_id")["status"].to_dict()
        )

    df["availability_status"] = df["player_id"].map(injury_map).fillna("")
    df["availability_out"] = df["availability_status"].map(_is_out).astype(int)
    df["availability_base_minutes"] = df[expected_minutes_col].clip(lower=0.0)

    adjusted_groups = []
    for team_id, team in df.groupby("team_id", sort=False):
        team = team.copy()
        base = team["availability_base_minutes"].to_numpy(dtype=float)
        out = team["availability_out"].to_numpy(dtype=bool)

        adjusted = base.copy()
        freed = float(adjusted[out].sum())
        adjusted[out] = 0.0

        healthy = ~out
        if freed > 0 and healthy.any():
            # Elasticity favors players with a real rotation role but leaves more room
            # for players who are not already projected near a star-level minutes cap.
            weights = np.sqrt(np.clip(base, 0.0, None) + 0.5) * np.clip(
                1.0 - base / 48.0, 0.05, 1.0
            )
            weights[~healthy] = 0.0

            remaining = freed
            for _ in range(6):
                room = np.clip(max_minutes - adjusted, 0.0, None)
                eligible_weight = weights * (room > 1e-9)
                total_weight = float(eligible_weight.sum())
                if remaining <= 1e-6 or total_weight <= 0:
                    break
                allocation = remaining * eligible_weight / total_weight
                allocation = np.minimum(allocation, room)
                adjusted += allocation
                remaining -= float(allocation.sum())

        team["availability_expected_minutes"] = adjusted
        team["availability_minutes_delta"] = adjusted - base

        if "adv_prior_usage_percentage" in team.columns:
            usage = (
                team["adv_prior_usage_percentage"]
                .fillna(team["adv_prior_usage_percentage"].median())
                .fillna(0.20)
            )
            lost_usage_minutes = float((base[out] * usage.to_numpy()[out]).sum())
            healthy_minutes = np.clip(adjusted, 0.0, None)
            denom = float((healthy_minutes[healthy] * usage.to_numpy()[healthy]).sum())
            multiplier = np.ones(len(team), dtype=float)
            if lost_usage_minutes > 0 and denom > 0:
                share = healthy_minutes * usage.to_numpy() / max(denom, 1e-12)
                multiplier += 0.35 * lost_usage_minutes * share / np.maximum(healthy_minutes, 1.0)
            multiplier[out] = 0.0
            team["availability_usage_multiplier"] = multiplier
        else:
            team["availability_usage_multiplier"] = np.where(out, 0.0, 1.0)

        adjusted_groups.append(team)

    return pd.concat(adjusted_groups, ignore_index=True)
