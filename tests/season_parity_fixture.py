"""A season a feature can be built from twice: once as training, once as live.

``season_progress`` is the one feature whose live and training definitions
disagreed, and the only test that settles a disagreement like that is to build
the same game both ways from the same historical state and compare. That needs
a fixture with three properties the smaller slate fixtures do not have.

It needs enough teams that the inferred schedule length is a real mode rather
than whatever the slate's own teams happen to have played: the classifier
infers the denominator from the modal team appearance count, so the teams that
are *not* in the game under test are what fix the number.

It needs games the standings do not count -- a Cup Championship, a Play-In
game, a playoff game -- because that is exactly where the two definitions
diverge. A team whose every game counts in the standings produces the same
number either way, so a fixture without them cannot fail.

And it needs box scores it can hand out two ways: for the games before the
slate, which is what live serving has, and for those plus the slate itself,
which is what training has once the game is played.
"""

from __future__ import annotations

import pandas as pd

#: Six teams, which is the smallest fixture in which the modal appearance
#: count survives two teams playing extra games. The two in the game under
#: test may run ahead; the other four hold the mode where it belongs.
PARITY_TEAMS: tuple[int, ...] = (11, 22, 33, 44, 55, 66)

HOME = PARITY_TEAMS[0]
AWAY = PARITY_TEAMS[1]

#: Season late enough that BALLDONTLIE populates ``ist_stage``, so a Cup
#: regime comes from the stage label rather than from the legacy date table.
SEASON = 2026

#: Where a fixture's standings rounds begin. Opening night, near enough.
SEASON_START = "2026-10-21"

#: One line of a box score, held constant so nothing a parity test compares
#: can move for a reason other than the feature under test.
BOX_SCORE_LINE: dict[str, object] = {
    "min": "30:00",
    "minutes": 30.0,
    "pts": 12.0,
    "reb": 4.0,
    "ast": 3.0,
    "stl": 1.0,
    "blk": 0.0,
    "fg3m": 1.0,
    "fga": 10.0,
    "fg3a": 4.0,
    "fta": 2.0,
    "oreb": 1.0,
    "dreb": 3.0,
    "turnover": 1.0,
    "pf": 2.0,
}

#: The slate builder reads advanced rows only for the leakage guard and the
#: latest-row lookup, and neither is what these tests are about.
EMPTY_ADVANCED_COLUMNS: tuple[str, ...] = ("player_id", "game_id", "date")


def game(
    identifier: int,
    moment: object,
    *,
    home_team_id: int = HOME,
    visitor_team_id: int = AWAY,
    postseason: bool = False,
    ist_stage: object = None,
    season: int = SEASON,
) -> dict:
    """One row of the games table, in the shape BALLDONTLIE delivers it."""
    return {
        "id": int(identifier),
        "date": pd.Timestamp(moment),
        "season": int(season),
        "home_team_id": int(home_team_id),
        "visitor_team_id": int(visitor_team_id),
        "postseason": bool(postseason),
        "ist_stage": ist_stage,
    }


def standings_rounds(
    count: int,
    *,
    teams: tuple[int, ...] = PARITY_TEAMS,
    start: str = SEASON_START,
    season: int = SEASON,
    first_id: int = 800_000,
) -> list[dict]:
    """``count`` ordinary rounds, each giving every team one appearance.

    Pairing the teams off means the modal appearance count after ``count``
    rounds is exactly ``count``, which is what makes a fixture's inferred
    schedule length predictable.
    """
    if len(teams) % 2:
        raise ValueError("a round needs an even number of teams")

    pairs = [(teams[index], teams[index + 1]) for index in range(0, len(teams), 2)]

    rows: list[dict] = []

    for index, moment in enumerate(
        pd.date_range(start, periods=count, freq="D")
    ):
        for offset, (home_team_id, visitor_team_id) in enumerate(pairs):
            rows.append(
                game(
                    first_id + index * len(pairs) + offset,
                    moment,
                    home_team_id=home_team_id,
                    visitor_team_id=visitor_team_id,
                    season=season,
                )
            )

    return rows


def day_after(games: list[dict], days: int = 1) -> pd.Timestamp:
    """The date ``days`` after the latest game in ``games``."""
    return max(entry["date"] for entry in games) + pd.Timedelta(days=days)


def box_scores(games: list[dict]) -> pd.DataFrame:
    """Two players per team for every game in ``games``.

    Player ids are derived from the team so a roster is stable across the two
    builds: ``build_base_frame`` groups by player and the live builder looks
    up each player's history, and both have to be talking about the same one.
    """
    rows: list[dict] = []

    for entry in games:
        for team_id in (entry["home_team_id"], entry["visitor_team_id"]):
            for slot in (1, 2):
                rows.append(
                    {
                        "player_id": int(team_id) * 10 + slot,
                        "team_id": int(team_id),
                        "game_id": int(entry["id"]),
                        "date": entry["date"],
                        "season": int(entry["season"]),
                        "postseason": bool(entry["postseason"]),
                        "home_team_id": int(entry["home_team_id"]),
                        "visitor_team_id": int(entry["visitor_team_id"]),
                        "position": "G" if slot == 1 else "F",
                        "draft_year": 2020,
                        **BOX_SCORE_LINE,
                    }
                )

    return pd.DataFrame(rows)


def active_players(*team_ids: int) -> pd.DataFrame:
    """The rosters the slate builder will build rows for."""
    return pd.DataFrame(
        [
            {
                "id": int(team_id) * 10 + slot,
                "current_team_id": int(team_id),
                "position": "G" if slot == 1 else "F",
                "draft_year": 2020,
                "first_name": f"P{team_id}{slot}",
                "last_name": "Test",
            }
            for team_id in team_ids
            for slot in (1, 2)
        ]
    )


def empty_advanced() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EMPTY_ADVANCED_COLUMNS))
