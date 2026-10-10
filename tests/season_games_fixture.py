"""A season games table in the shape the production one actually has.

``data/raw/seasons/season=<n>/games.parquet`` is written from
``client.games(seasons=[season])``, which is the season's *whole* schedule:
every fixture the league has published, played or not. ``build_base_frame``
and the adaptive fit both pass that table to ``classify_games`` unfiltered --
"games keeps its future rows only as schedule context" is how
``adaptive_training`` puts it -- and the live slate builder now does the same.

A test fixture that supplies only the games it also has box scores for is
therefore not a smaller version of the production table, it is a different
one: ``classify_games`` infers each season's schedule length from the modal
number of team appearances, so a four-game fixture declares a four-game
season and every later game in it is a Play-In game. This pads a fixture out
to a real schedule length so the regime a test asserts is the regime
production would produce.
"""

from __future__ import annotations

import pandas as pd

#: A normal NBA season. Not read as a model constant anywhere: the classifier
#: infers the schedule length from the table rather than being told it, and
#: this is only how much schedule a fixture has to carry for that inference to
#: land on a normal season.
FULL_SCHEDULE_LENGTH = 82


def team_appearances(games: pd.DataFrame) -> dict[int, int]:
    """How many of ``games`` each team appears in."""
    counts: dict[int, int] = {}

    for column in ("home_team_id", "visitor_team_id"):
        for team_id in pd.to_numeric(games[column], errors="raise").astype(int):
            counts[int(team_id)] = counts.get(int(team_id), 0) + 1

    return counts


def pad_season_schedule(
    games: pd.DataFrame,
    schedule_length: int = FULL_SCHEDULE_LENGTH,
    first_id: int = 500_000,
) -> pd.DataFrame:
    """Return ``games`` plus the rest of its season's scheduled fixtures.

    The padding is dated strictly after everything in ``games``, so it is
    future schedule rather than history: nothing reads an outcome from it, and
    the slate builder's leakage guard looks at box scores and advanced rows,
    neither of which this touches.

    Fixtures pair the two least-played teams repeatedly, which is enough to
    bring every team to ``schedule_length`` appearances for the two- and
    four-team fixtures the tests use.
    """
    frame = games.copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()

    if "postseason" not in frame.columns:
        raise ValueError("a games fixture must say whether each game is postseason")

    rows: list[dict] = []
    next_id = first_id

    for season, season_games in frame.groupby("season", sort=True):
        counts = team_appearances(season_games)
        moment = season_games["date"].max()

        if len(counts) < 2:
            raise ValueError(
                f"season {season} has {len(counts)} team(s), so there is no "
                "fixture to schedule"
            )

        while any(played < schedule_length for played in counts.values()):
            ranked = sorted(counts.items(), key=lambda item: (item[1], item[0]))
            (home_id, _), (visitor_id, _) = ranked[0], ranked[1]

            moment = moment + pd.Timedelta(days=1)

            rows.append(
                {
                    "id": next_id,
                    "date": moment,
                    "season": int(season),
                    "home_team_id": home_id,
                    "visitor_team_id": visitor_id,
                    "postseason": False,
                }
            )

            counts[home_id] += 1
            counts[visitor_id] += 1
            next_id += 1

    if not rows:
        return frame.reset_index(drop=True)

    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
