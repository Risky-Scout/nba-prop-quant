from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .features import ADVANCED_FEATURES, TARGETS, add_dynamic_priors
from .game_context import classify_games


# The NBA schedules and reports its slates in Eastern time. A 10:30pm PT tip is
# still part of that evening's Eastern slate even though its UTC timestamp has
# already rolled over to the next calendar day, so any slate date derived from a
# tip timestamp must be taken in this zone rather than in UTC or local time.
NBA_SLATE_TIMEZONE = "America/New_York"

NBA_SLATE_ZONE = ZoneInfo(NBA_SLATE_TIMEZONE)

# The competition-regime flags the fitted models name in their feature lists.
# classify_games derives all five from one game_type and build_base_frame
# merges them onto the training frame by game id, so a live slate has to carry
# the same five under the same definitions or the model refuses to score it.
GAME_CONTEXT_FLAGS: tuple[str, ...] = (
    "is_regular_season",
    "is_nba_cup",
    "is_nba_cup_championship",
    "is_play_in",
    "is_playoffs",
)

# What classify_games needs to classify a game at all, plus the NBA Cup stage
# label, which is the only source of a Cup regime from season 2025 onwards.
GAME_CONTEXT_INPUT_COLUMNS: tuple[str, ...] = (
    "id",
    "date",
    "season",
    "home_team_id",
    "visitor_team_id",
    "postseason",
    "ist_stage",
)


class SlateDateError(ValueError):
    """A value could not be resolved to an unambiguous NBA slate date."""


class HistoryLeakageError(ValueError):
    """Historical inputs are not strictly older than the slate being predicted."""


class SlateGameContextError(ValueError):
    """An upcoming slate's competition regime could not be classified."""


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
            raise SlateDateError(
                f"tip timestamp is not ISO 8601: {value!r}"
            ) from error

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
            slate_date_from_tip_timestamp(value)
            if aware
            else slate_date_from_schedule_date(value)
        )

    if isinstance(value, str):
        text = value.strip()

        return (
            slate_date_from_schedule_date(text)
            if len(text) == 10
            else slate_date_from_tip_timestamp(text)
        )

    return slate_date_from_schedule_date(value)


def _latest_history_date(
    frame: pd.DataFrame,
    label: str,
    source: Any,
) -> pd.Timestamp:
    """Return max(date), refusing a frame whose dates cannot all be trusted."""
    if "date" not in getattr(frame, "columns", []):
        raise HistoryLeakageError(f"{label} has no date column: {source}")

    dates = pd.to_datetime(frame["date"], errors="coerce")

    unparseable = int(dates.isna().sum())

    if unparseable:
        raise HistoryLeakageError(
            f"{label} has {unparseable} unparseable date value(s): {source}"
        )

    latest = dates.dt.normalize().max()

    if pd.isna(latest):
        raise HistoryLeakageError(f"{label} has no usable date: {source}")

    return latest


def assert_history_precedes_slate(
    frame: pd.DataFrame,
    label: str,
    slate_date: Any,
    source: Any = "",
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
    history_stats: pd.DataFrame,
    advanced: pd.DataFrame,
    upcoming_games: pd.DataFrame,
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
        assert_history_precedes_slate(
            advanced,
            "historical advanced",
            slate_date,
        )

    return slate_date


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


def _game_context_input(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Reduce a games table to exactly what the classifier reads."""
    columns = list(getattr(frame, "columns", []))

    required = [
        name for name in GAME_CONTEXT_INPUT_COLUMNS if name != "ist_stage"
    ]

    missing = [name for name in required if name not in columns]

    if missing:
        raise SlateGameContextError(
            f"{label} cannot be classified: missing column(s) {missing}"
        )

    reduced = frame[[name for name in GAME_CONTEXT_INPUT_COLUMNS if name in columns]].copy()

    if "ist_stage" not in reduced.columns:
        reduced["ist_stage"] = pd.NA

    reduced["id"] = pd.to_numeric(reduced["id"], errors="raise").astype("int64")
    reduced["season"] = pd.to_numeric(reduced["season"], errors="raise").astype("int64")
    reduced["date"] = pd.to_datetime(reduced["date"], errors="raise").dt.normalize()

    return reduced[list(GAME_CONTEXT_INPUT_COLUMNS)]


def classify_upcoming_games(
    upcoming_games: pd.DataFrame,
    history_games: pd.DataFrame,
) -> pd.DataFrame:
    """Return the five canonical game-context flags for each upcoming game.

    ``classify_games`` is the single definition of those flags and
    ``build_base_frame`` was the only thing that had ever called it, so the
    live slate carried none of them and the fitted models refused every real
    slate on a column-presence contract. It is reused here rather than
    reimplemented: a second copy of the rules is a second thing that has to
    stay in step with the feature definitions the models were fitted on.

    It is handed the season's whole games table rather than the slate alone,
    because Play-In detection is structural rather than labelled. A Play-In
    game carries no ``postseason`` flag and no ``ist_stage``; the classifier
    recognises it by a team having already completed the season's inferred
    schedule length, and both the inference and the count need the season's
    other games. Given the slate by itself the inferred schedule length is the
    size of the slate, and every answer that follows from it is meaningless.

    Narrowing the input to the seasons the slate belongs to is exact rather
    than an approximation. ``classify_games`` infers a schedule length per
    season and walks each season's games against that season's own counter, so
    a season is classified identically whether or not the others are present.
    """
    upcoming = _game_context_input(upcoming_games, "upcoming games")

    if upcoming.empty:
        return pd.DataFrame(columns=["game_id", *GAME_CONTEXT_FLAGS])

    seasons = sorted(set(upcoming["season"].tolist()))

    if history_games is None or len(history_games) == 0:
        raise SlateGameContextError(
            "the slate's competition regime needs the season's games table "
            f"and none was supplied; seasons {seasons} are unclassifiable "
            "because Play-In detection counts a team's completed games"
        )

    schedule = _game_context_input(history_games, "historical games")

    schedule = schedule[schedule["season"].isin(seasons)]

    absent = sorted(set(seasons) - set(schedule["season"].tolist()))

    if absent:
        raise SlateGameContextError(
            f"the games table holds no season {absent} schedule, so the "
            "slate's competition regime cannot be established; Play-In "
            "detection counts a team's completed games against the season's "
            "inferred schedule length"
        )

    # The upcoming rows are the live authority on the games they describe, so
    # they are taken last. last() keeps the last *non-null* value per column,
    # which is also how a capture that omits ist_stage inherits the stage the
    # stored season schedule recorded for the same game.
    combined = (
        pd.concat([schedule, upcoming], ignore_index=True)
        .groupby("id", as_index=False, sort=False)
        .last()
    )

    classified = classify_games(combined)

    wanted = set(upcoming["id"].tolist())

    context = classified[classified["id"].isin(wanted)]

    unclassified = sorted(wanted - set(context["id"].tolist()))

    if unclassified:
        raise SlateGameContextError(
            f"the classifier returned no competition regime for game(s) "
            f"{unclassified}"
        )

    return (
        context[["id", *GAME_CONTEXT_FLAGS]]
        .rename(columns={"id": "game_id"})
        .astype({name: "int64" for name in GAME_CONTEXT_FLAGS})
        .reset_index(drop=True)
    )


def build_upcoming_slate_features(
    history_stats: pd.DataFrame,
    upcoming_games: pd.DataFrame,
    active_players: pd.DataFrame,
    advanced: pd.DataFrame,
    dynamic_params: dict[str, Any],
    history_games: pd.DataFrame,
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

    context = classify_upcoming_games(upcoming, history_games)

    flags = {
        int(record["game_id"]): {
            name: int(record[name]) for name in GAME_CONTEXT_FLAGS
        }
        for record in context.to_dict("records")
    }

    for row in rows:
        row.update(flags[row["game_id"]])

    return pd.DataFrame(rows)
