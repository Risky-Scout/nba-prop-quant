from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .decay import DecayParams, decay_prior
from .game_context import classify_games
from .kalman import KalmanParams, kalman_prior


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


def _lagged_ewm(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    span: int,
) -> pd.Series:
    return frame.groupby(group_col, sort=False)[value_col].transform(
        lambda s: s.shift(1).ewm(span=span, adjust=False, min_periods=1).mean()
    )


def _prepare_team_context(stats: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "fg3m",
        "fga",
        "fg3a",
        "fta",
        "oreb",
        "turnover",
    ]
    team = (
        stats.groupby(
            ["game_id", "date", "season", "team_id", "home_team_id", "visitor_team_id"],
            as_index=False,
        )[sum_cols]
        .sum(min_count=1)
        .sort_values(["date", "game_id", "team_id"])
    )

    opponent = team[
        ["game_id", "team_id", "pts", "reb", "ast", "stl", "blk", "fg3m", "fga", "fta", "oreb", "turnover"]
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
        team["team_id"].eq(team["home_team_id"]),
        team["visitor_team_id"],
        team["home_team_id"],
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
    team["pace_game"] = 0.5 * (
        team["possessions_proxy"] + team["opp_possessions_proxy"]
    )
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
    return team[["game_id", "team_id", "opponent_id"]].merge(
        own, on=["game_id", "team_id"], how="left"
    ).merge(
        opp, on=["game_id", "opponent_id"], how="left"
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

        game_context = (
            classified_games[context_columns]
            .rename(columns={"id": "game_id"})
        )

        overlapping = [
            col
            for col in game_context.columns
            if col != "game_id" and col in df.columns
        ]

        if overlapping:
            df = df.drop(columns=overlapping)

        df = df.merge(
            game_context,
            on="game_id",
            how="left",
            validate="many_to_one",
        )

        if df["game_type"].isna().any():
            missing_games = int(
                df.loc[
                    df["game_type"].isna(),
                    "game_id",
                ].nunique()
            )

            raise ValueError(
                f"{missing_games} stat games did not match "
                "the classified games table"
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
        df["team_id"].eq(df["home_team_id"]),
        df["visitor_team_id"],
        df["home_team_id"],
    )
    df["is_home"] = df["team_id"].eq(df["home_team_id"]).astype(int)
    df["postseason"] = df["postseason"].fillna(False).astype(int)

    previous_date = df.groupby("player_id")["date"].shift(1)
    df["days_since_prev"] = (df["date"] - previous_date).dt.days
    df["days_rest"] = (df["days_since_prev"] - 1).clip(lower=0, upper=14)
    df["b2b"] = df["days_since_prev"].eq(1).astype(int)

    previous_team = df.groupby("player_id")["team_id"].shift(1)
    df["team_change"] = (
        previous_team.notna() & df["team_id"].ne(previous_team)
    ).astype(int)

    df["player_game_number"] = df.groupby("player_id").cumcount()
    df["career_games_prior"] = df["player_game_number"]

    team_order = df[["game_id", "date", "season", "team_id"]].drop_duplicates()

    team_order = team_order.sort_values(
        ["team_id", "season", "date", "game_id"]
    )

    team_order["team_game_number"] = (
        team_order
        .groupby(["team_id", "season"])
        .cumcount()
        + 1
    )

    df = df.merge(
        team_order[
            ["game_id", "team_id", "team_game_number"]
        ],
        on=["game_id", "team_id"],
        how="left",
    )

    fallback_progress = (
        (df["team_game_number"] - 1) / 82.0
    ).clip(
        0.0,
        1.5,
    )

    if {
        "home_season_progress",
        "visitor_season_progress",
    }.issubset(df.columns):

        classified_progress = np.where(
            df["is_home"].eq(1),
            df["home_season_progress"],
            df["visitor_season_progress"],
        )

        df["season_progress"] = (
            pd.to_numeric(
                pd.Series(
                    classified_progress,
                    index=df.index,
                ),
                errors="coerce",
            )
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
        df[f"{stat}_rate_game"] = (
            pd.to_numeric(df[stat], errors="coerce") / df["minutes"].clip(lower=1.0)
        )
        df[f"prior_{stat}_rate10"] = _lagged_ewm(
            df, "player_id", f"{stat}_rate_game", span=10
        )

    df["prior_minutes10"] = _lagged_ewm(df, "player_id", "minutes", span=10)

    team_context = _prepare_team_context(df)
    df = df.merge(
        team_context,
        on=["game_id", "team_id", "opponent_id"],
        how="left",
    )

    advanced_for_priors = (
        advanced.copy()
        if advanced is not None
        else pd.DataFrame()
    )

    if not advanced_for_priors.empty:
        # BDL can return advanced rows for DNP / zero-minute
        # player-games. Restrict advanced history to the exact
        # played-player population used by the box-score model.
        played_keys = (
            df[
                ["game_id", "player_id"]
            ]
            .drop_duplicates()
        )

        advanced_for_priors = advanced_for_priors.merge(
            played_keys,
            on=["game_id", "player_id"],
            how="inner",
            validate="one_to_one",
        )

    adv_prior = _advanced_priors(
        advanced_for_priors
    )

    if not adv_prior.empty:
        df = df.merge(
            adv_prior,
            on=["game_id", "player_id"],
            how="left",
            validate="one_to_one",
        )

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


def add_dynamic_priors(
    frame: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    df = frame.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True).copy()
    pid = df["player_id"].to_numpy(dtype=np.int64)
    days = df["days_since_prev"].fillna(0).to_numpy(dtype=float)
    team_change = df["team_change"].fillna(0).to_numpy(dtype=float)

    series_map: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "min": (
            df["minutes"].to_numpy(dtype=float),
            np.ones(len(df), dtype=float),
        )
    }
    for target in TARGETS:
        rate = (
            pd.to_numeric(df[target], errors="coerce")
            / df["minutes"].clip(lower=1.0)
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
        df[f"kalman_prior_{name}"] = kalman_prior(
            pid,
            days,
            team_change,
            values,
            exposure,
            kparams,
        )

    return df


def feature_columns(
    frame: pd.DataFrame,
    include_expected_minutes: bool = False,
) -> list[str]:
    columns = []
    for col in frame.columns:
        if col in FEATURE_EXACT or col.startswith(FEATURE_PREFIXES):
            columns.append(col)
    if include_expected_minutes and "expected_minutes" in frame.columns:
        columns.append("expected_minutes")
    return sorted(set(columns))
