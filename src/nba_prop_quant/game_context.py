from __future__ import annotations

import numpy as np
import pandas as pd


REGULAR_SEASON_GAME_TYPES = {
    "regular",
    "nba_cup_group",
    "nba_cup_quarterfinal",
    "nba_cup_semifinal",
    "nba_cup_other",
}


# BALLDONTLIE only populates ist_stage beginning in season 2025.
# These are the two NBA Cup / In-Season Tournament championship
# dates that predate that field.
LEGACY_CUP_CHAMPIONSHIP_DATES = {
    2023: pd.Timestamp("2023-12-09"),
    2024: pd.Timestamp("2024-12-17"),
}


# Season labels are NBA season start years.
LEGACY_PLAY_IN_WINDOWS = {
    2019: (
        pd.Timestamp("2020-08-15"),
        pd.Timestamp("2020-08-15"),
    ),
    2020: (
        pd.Timestamp("2021-05-18"),
        pd.Timestamp("2021-05-21"),
    ),
    2021: (
        pd.Timestamp("2022-04-12"),
        pd.Timestamp("2022-04-15"),
    ),
}


def _legacy_cup_championship_mask(
    games: pd.DataFrame,
) -> pd.Series:
    dates = pd.to_datetime(
        games["date"],
        errors="coerce",
    ).dt.normalize()

    seasons = pd.to_numeric(
        games["season"],
        errors="coerce",
    )

    postseason = (
        games["postseason"]
        .fillna(False)
        .astype(bool)
    )

    mask = pd.Series(
        False,
        index=games.index,
        dtype=bool,
    )

    for season, championship_date in (
        LEGACY_CUP_CHAMPIONSHIP_DATES.items()
    ):
        mask |= (
            seasons.eq(season)
            & dates.eq(championship_date)
            & (~postseason)
        )

    return mask


def _legacy_play_in_mask(
    games: pd.DataFrame,
) -> pd.Series:
    dates = pd.to_datetime(
        games["date"],
        errors="coerce",
    ).dt.normalize()

    seasons = pd.to_numeric(
        games["season"],
        errors="coerce",
    )

    mask = pd.Series(
        False,
        index=games.index,
        dtype=bool,
    )

    for season, (start_date, end_date) in (
        LEGACY_PLAY_IN_WINDOWS.items()
    ):
        mask |= (
            seasons.eq(season)
            & dates.between(
                start_date,
                end_date,
                inclusive="both",
            )
        )

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


def infer_schedule_lengths(
    games: pd.DataFrame,
) -> dict[int, int]:
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
    required = {
        "season",
        "home_team_id",
        "visitor_team_id",
        "postseason",
    }

    missing = required - set(games.columns)

    if missing:
        raise ValueError(
            f"Games table is missing required columns: {sorted(missing)}"
        )

    g = games.copy()

    if "ist_stage" not in g.columns:
        g["ist_stage"] = pd.NA

    g["season"] = pd.to_numeric(
        g["season"],
        errors="raise",
    ).astype(int)

    g["postseason"] = (
        g["postseason"]
        .fillna(False)
        .astype(bool)
    )

    g["_stage"] = g["ist_stage"].map(_normalize_stage)

    is_cup_championship = (
        g["_stage"].str.contains(
            "championship",
            regex=False,
        )
        | _legacy_cup_championship_mask(g)
    )

    is_legacy_play_in = _legacy_play_in_mask(g)

    candidates = g[
        (~g["postseason"])
        & (~is_cup_championship)
        & (~is_legacy_play_in)
    ].copy()

    home = candidates[
        ["season", "home_team_id"]
    ].rename(
        columns={
            "home_team_id": "team_id",
        }
    )

    visitor = candidates[
        ["season", "visitor_team_id"]
    ].rename(
        columns={
            "visitor_team_id": "team_id",
        }
    )

    appearances = pd.concat(
        [home, visitor],
        ignore_index=True,
    )

    team_counts = (
        appearances
        .groupby(
            ["season", "team_id"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "games",
            }
        )
    )

    schedule_lengths: dict[int, int] = {}

    for season, season_counts in team_counts.groupby(
        "season",
        sort=True,
    ):
        values = season_counts["games"].astype(int)

        frequencies = values.value_counts()

        max_frequency = frequencies.max()

        modes = (
            frequencies[
                frequencies.eq(max_frequency)
            ]
            .index
            .to_numpy(dtype=int)
        )

        # If an unusual historical season produces a tie,
        # choose the median modal value and flag it later in QA.
        schedule_length = int(
            np.round(
                np.median(modes)
            )
        )

        schedule_lengths[int(season)] = schedule_length

    return schedule_lengths


def classify_games(
    games: pd.DataFrame,
) -> pd.DataFrame:
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
    required = {
        "id",
        "date",
        "season",
        "home_team_id",
        "visitor_team_id",
        "postseason",
    }

    missing = required - set(games.columns)

    if missing:
        raise ValueError(
            f"Games table is missing required columns: {sorted(missing)}"
        )

    g = games.copy().reset_index(drop=True)

    if "ist_stage" not in g.columns:
        g["ist_stage"] = pd.NA

    g["date"] = pd.to_datetime(
        g["date"],
        errors="raise",
    ).dt.normalize()

    g["season"] = pd.to_numeric(
        g["season"],
        errors="raise",
    ).astype(int)

    g["home_team_id"] = pd.to_numeric(
        g["home_team_id"],
        errors="raise",
    ).astype(int)

    g["visitor_team_id"] = pd.to_numeric(
        g["visitor_team_id"],
        errors="raise",
    ).astype(int)

    g["postseason"] = (
        g["postseason"]
        .fillna(False)
        .astype(bool)
    )

    g["_stage"] = g["ist_stage"].map(
        _normalize_stage
    )

    g["_legacy_cup_championship"] = (
        _legacy_cup_championship_mask(g)
    )

    g["_legacy_play_in"] = (
        _legacy_play_in_mask(g)
    )

    schedule_lengths = infer_schedule_lengths(g)

    g["inferred_schedule_length"] = (
        g["season"]
        .map(schedule_lengths)
        .astype("Int64")
    )

    g["game_type"] = pd.NA

    g["home_regular_games_before"] = 0
    g["visitor_regular_games_before"] = 0

    for season in sorted(g["season"].unique()):
        season_rows = (
            g[
                g["season"].eq(season)
            ]
            .sort_values(
                ["date", "id"]
            )
        )

        schedule_length = schedule_lengths[int(season)]

        regular_games_played: dict[int, int] = {}

        for idx, row in season_rows.iterrows():
            home_id = int(row["home_team_id"])
            visitor_id = int(row["visitor_team_id"])

            home_before = regular_games_played.get(
                home_id,
                0,
            )

            visitor_before = regular_games_played.get(
                visitor_id,
                0,
            )

            g.at[
                idx,
                "home_regular_games_before",
            ] = home_before

            g.at[
                idx,
                "visitor_regular_games_before",
            ] = visitor_before

            stage = row["_stage"]

            cup_type = _cup_game_type(stage)

            if bool(
                row["_legacy_cup_championship"]
            ):
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

            elif (
                home_before >= schedule_length
                or visitor_before >= schedule_length
            ):
                game_type = "play_in"

            else:
                game_type = "regular"

            g.at[
                idx,
                "game_type",
            ] = game_type

            # These games count toward the official regular-season
            # schedule. The Cup Championship, Play-In and playoffs do not.
            if game_type in REGULAR_SEASON_GAME_TYPES:
                regular_games_played[home_id] = (
                    home_before + 1
                )

                regular_games_played[visitor_id] = (
                    visitor_before + 1
                )

    g["counts_in_regular_season"] = (
        g["game_type"]
        .isin(REGULAR_SEASON_GAME_TYPES)
        .astype(int)
    )

    g["is_regular_season"] = (
        g["counts_in_regular_season"]
        .astype(int)
    )

    g["is_nba_cup"] = (
        g["game_type"]
        .astype(str)
        .str.startswith("nba_cup_")
        .astype(int)
    )

    g["is_nba_cup_championship"] = (
        g["game_type"]
        .eq("nba_cup_championship")
        .astype(int)
    )

    g["is_play_in"] = (
        g["game_type"]
        .eq("play_in")
        .astype(int)
    )

    g["is_playoffs"] = (
        g["game_type"]
        .eq("playoffs")
        .astype(int)
    )

    schedule_length = (
        g["inferred_schedule_length"]
        .astype(float)
        .replace(0.0, np.nan)
    )

    g["home_season_progress"] = (
        g["home_regular_games_before"]
        / schedule_length
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    g["visitor_season_progress"] = (
        g["visitor_regular_games_before"]
        / schedule_length
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    return (
        g.drop(
            columns=[
                "_stage",
                "_legacy_cup_championship",
                "_legacy_play_in",
            ]
        )
        .sort_values(
            ["date", "id"]
        )
        .reset_index(drop=True)
    )
