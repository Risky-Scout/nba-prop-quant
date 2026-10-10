"""A non-empty slate the real serving scripts can actually price.

The repository had no fixture that ran ``scripts/10_predict_slate.py`` and
``scripts/15_price_markets.py`` end to end to non-empty parquets. Every
existing test either replaced both scripts with stubs or built one of their
inputs in isolation, which is why the frozen manifest being resolved against
the working directory survived: nothing ever executed the two scripts against
a verified frozen runtime bundle and a slate with rows in it, so the failure
had nowhere to show up except production.

So this builds the smallest tree that makes both scripts run their real code
paths: a prospective capture bundle for one game, ten players with enough
history for the rolling features to exist, and over/under quotes on props the
frozen calibration policy covers.

The models are the frozen ones. Nothing here trains, fits or calibrates
anything, and nothing here is a model artifact: the fixture supplies inputs,
and the verified bundle supplies every parameter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from nba_prop_quant.prospective_snapshot import (
    build_capture_id,
    canonical_records_sha256,
)
from nba_prop_quant.storage import timestamped_jsonl_append

#: The slate being served, and the capture twenty minutes before tip. Both are
#: fixed rather than derived from today, so the fixture's own feature values do
#: not drift with the calendar and the "prediction output is unchanged" check
#: compares two runs of the same thing.
SLATE_DATE = "2026-10-20"
TIP_OFF = "2026-10-20T23:00:00+00:00"
CAPTURED_AT = "2026-10-20T22:40:00+00:00"

SEASON = 2026
GAME_ID = 9901
HOME_TEAM_ID = 11
AWAY_TEAM_ID = 22

#: Ten players, five a side. Enough that a dropped row leaves the slate
#: non-empty, so a run that silently loses players is visible as a count rather
#: than as an empty-frame failure.
ROSTER: tuple[tuple[int, int, str, int], ...] = (
    (1101, HOME_TEAM_ID, "G", 2017),
    (1102, HOME_TEAM_ID, "G", 2019),
    (1103, HOME_TEAM_ID, "F", 2015),
    (1104, HOME_TEAM_ID, "F", 2021),
    (1105, HOME_TEAM_ID, "C", 2018),
    (2201, AWAY_TEAM_ID, "G", 2016),
    (2202, AWAY_TEAM_ID, "G", 2020),
    (2203, AWAY_TEAM_ID, "F", 2014),
    (2204, AWAY_TEAM_ID, "F", 2022),
    (2205, AWAY_TEAM_ID, "C", 2019),
)

#: Quotes on props the frozen calibration policy covers, deliberately none of
#: them a Gate 3 changed prop. ``assists``, ``points_assists`` and
#: ``points_rebounds`` are dropped unless the projection carries
#: ``gate3_role_ready == 1``, and a fixture that priced zero rows because of a
#: role-readiness filter would prove nothing about the manifest root.
QUOTED_PROPS: tuple[tuple[int, str, str], ...] = (
    (1101, "points", "18.5"),
    (1101, "rebounds", "4.5"),
    (1102, "points", "14.5"),
    (1103, "rebounds", "7.5"),
    (1104, "threes", "1.5"),
    (1105, "rebounds", "9.5"),
    (2201, "points", "20.5"),
    (2202, "points", "12.5"),
    (2203, "rebounds", "6.5"),
    (2205, "blocks", "1.5"),
)

#: How many prior games each player has. The rolling windows, decay states and
#: Kalman filters need a run of games to produce a prior at all; ten days of
#: them is enough for every window the slate builder asks for.
HISTORY_GAMES = 14


def _history_rows() -> list[dict[str, Any]]:
    """Box scores for every roster player, all strictly before the slate.

    The slate builder refuses history that reaches the slate date, which is
    the leakage guard, so the last prior game is two days before tip.
    """
    rows: list[dict[str, Any]] = []

    dates = pd.date_range(
        end=pd.Timestamp(SLATE_DATE) - pd.Timedelta(days=2),
        periods=HISTORY_GAMES,
        freq="2D",
    )

    for index, moment in enumerate(dates):
        game_id = 9000 + index

        home_team_id = HOME_TEAM_ID if index % 2 == 0 else AWAY_TEAM_ID
        visitor_team_id = AWAY_TEAM_ID if index % 2 == 0 else HOME_TEAM_ID

        for slot, (player_id, team_id, position, draft_year) in enumerate(ROSTER):
            # Deterministic, mildly varying box scores. The values only have to
            # be plausible and stable: every parameter that turns them into a
            # projection comes from the frozen bundle.
            minutes = 22.0 + ((slot * 3 + index) % 11)

            rows.append(
                {
                    "stat_id": len(rows) + 1,
                    "player_id": player_id,
                    "team_id": team_id,
                    "game_id": game_id,
                    "date": moment,
                    "season": SEASON,
                    "postseason": False,
                    "home_team_id": home_team_id,
                    "visitor_team_id": visitor_team_id,
                    "position": position,
                    "draft_year": draft_year,
                    "min": f"{int(minutes)}:00",
                    "minutes": minutes,
                    "pts": 8.0 + ((slot * 2 + index) % 15),
                    "reb": 3.0 + ((slot + index) % 8),
                    "ast": 2.0 + ((slot + index) % 6),
                    "stl": float((slot + index) % 3),
                    "blk": float((slot + index) % 2),
                    "fg3m": float((slot + index) % 4),
                    "fga": 10.0 + ((slot + index) % 9),
                    "fg3a": 3.0 + ((slot + index) % 5),
                    "fta": 2.0 + ((slot + index) % 4),
                    "oreb": 1.0 + ((slot + index) % 3),
                    "dreb": 2.0 + ((slot + index) % 6),
                    "turnover": 1.0 + ((slot + index) % 3),
                    "pf": 1.0 + ((slot + index) % 4),
                }
            )

    return rows


def _advanced_rows(history: pd.DataFrame) -> list[dict[str, Any]]:
    """Advanced tracking rows covering every metric the frozen models use.

    All of ``features.ADVANCED_FEATURES``, because the slate builder derives an
    ``adv_prior_*`` feature only from a column that is actually present, and
    the frozen minutes model's contract names all twenty-four. A fixture
    carrying a convenient subset fails the contract on the missing ones, which
    says nothing about the manifest root under test.
    """
    rows: list[dict[str, Any]] = []

    for row in history.itertuples():
        player_id = int(row.player_id)
        spread = player_id % 9

        rows.append(
            {
                "player_id": player_id,
                "game_id": int(row.game_id),
                "date": row.date,
                "season": SEASON,
                "usage_percentage": 18.0 + spread,
                "estimated_usage_percentage": 17.5 + spread,
                "assist_percentage": 12.0 + spread,
                "rebound_percentage": 9.0 + (spread / 2.0),
                "true_shooting_percentage": 0.54 + (spread / 200.0),
                "effective_field_goal_percentage": 0.51 + (spread / 200.0),
                "pace": 99.0 + (spread / 3.0),
                "estimated_pace": 98.5 + (spread / 3.0),
                "possessions": 62.0 + spread,
                "touches": 48.0 + spread,
                "passes": 38.0 + spread,
                "secondary_assists": 0.8 + (spread / 10.0),
                "rebound_chances_total": 7.0 + (spread / 2.0),
                "deflections": 1.6 + (spread / 10.0),
                "contested_shots": 5.4 + (spread / 4.0),
                "defended_at_rim_fga": 3.1 + (spread / 5.0),
                "defended_at_rim_fg_pct": 0.56 + (spread / 200.0),
                "pct_fga_3pt": 0.34 + (spread / 150.0),
                "pct_3pa": 0.33 + (spread / 150.0),
                "free_throw_attempt_rate": 0.21 + (spread / 200.0),
                "fouls_drawn": 2.2 + (spread / 10.0),
                "speed": 4.3 + (spread / 50.0),
                "distance": 2.4 + (spread / 50.0),
                "turnover_ratio": 11.0 + (spread / 4.0),
            }
        )

    return rows


def write_history(data_root: Path) -> int:
    """The prior-season box scores and advanced rows the slate is built from."""
    history = pd.DataFrame(_history_rows())

    seasons = Path(data_root) / "raw" / "seasons" / str(SEASON)
    seasons.mkdir(parents=True, exist_ok=True)
    history.to_parquet(seasons / "stats.parquet", index=False)

    advanced = Path(data_root) / "raw" / "advanced" / str(SEASON)
    advanced.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_advanced_rows(history)).to_parquet(
        advanced / "advanced.parquet", index=False
    )

    return len(history)


def write_capture(snapshot_root: Path) -> str:
    """One scheduled pre-tip capture, in the layout the client demands.

    The capture is the production input path: ``--input-source snapshot`` is
    what the lifecycle serves from, and the client checks the component hashes
    against the recorded run and refuses a capture taken after tip-off. So the
    fixture builds a real one rather than a loose pile of JSONL.
    """
    game = {
        "id": GAME_ID,
        "date": SLATE_DATE,
        "season": SEASON,
        "datetime": TIP_OFF,
        "status": "Scheduled",
        "home_team": {"id": HOME_TEAM_ID, "abbreviation": "HOM"},
        "visitor_team": {"id": AWAY_TEAM_ID, "abbreviation": "AWY"},
    }

    active_players = [
        {
            "id": player_id,
            "first_name": f"P{player_id}",
            "last_name": "Fixture",
            "position": position,
            "draft_year": draft_year,
            "team": {"id": team_id},
        }
        for player_id, team_id, position, draft_year in ROSTER
    ]

    lineups = [
        {
            "id": index + 1,
            "game_id": GAME_ID,
            "starter": index % 5 < 5,
            "position": position,
            "player": {"id": player_id},
            "team": {"id": team_id},
        }
        for index, (player_id, team_id, position, _) in enumerate(ROSTER)
    ]

    props = [
        {
            "id": 7000 + index,
            "game_id": GAME_ID,
            "player_id": player_id,
            "vendor": "fixture",
            "prop_type": prop_type,
            "line_value": line_value,
            "opened_at": CAPTURED_AT,
            "updated_at": CAPTURED_AT,
            "market": {
                "type": "over_under",
                # -110 both ways is a 4.5% hold, inside the 20% default
                # ceiling, so the quote filter keeps the row and the fixture
                # tests pricing rather than rejection.
                "over_odds": -110,
                "under_odds": -110,
            },
        }
        for index, (player_id, prop_type, line_value) in enumerate(QUOTED_PROPS)
    ]

    components: dict[str, list[dict[str, Any]]] = {
        "games": [game],
        "active_players": active_players,
        "injuries": [],
        "lineups": lineups,
        "player_props": props,
    }

    directories = {
        "games": ("games", "game"),
        "active_players": ("active_players", "active_player"),
        "injuries": ("injuries", "injury"),
        "lineups": ("lineups", "lineup"),
        "player_props": ("player_props", "live_player_prop"),
    }

    root = Path(snapshot_root)

    for name, records in components.items():
        if not records:
            continue

        directory, snapshot_type = directories[name]

        timestamped_jsonl_append(
            records,
            root / directory / f"{SLATE_DATE}.jsonl",
            snapshot_type,
            captured_at=CAPTURED_AT,
        )

    component_sha256 = {
        name: canonical_records_sha256(records)
        for name, records in components.items()
    }

    window_ids = [f"{GAME_ID}:T-20m"]
    due_game_ids = [GAME_ID]

    capture_id = build_capture_id(
        date=SLATE_DATE,
        captured_at=CAPTURED_AT,
        window_ids=window_ids,
        due_game_ids=due_game_ids,
        component_sha256=component_sha256,
    )

    run = {
        "date": SLATE_DATE,
        "capture_reason": "scheduled",
        "window_ids": window_ids,
        "due_game_ids": due_game_ids,
        "grace_minutes": 5,
        "games_count": 1,
        "active_players_count": len(active_players),
        "game_ids": [GAME_ID],
        "team_ids": [HOME_TEAM_ID, AWAY_TEAM_ID],
        "injuries_count": 0,
        "lineups_count": len(lineups),
        "player_props_count": len(props),
        "error_count": 0,
        "errors": [],
        "component_sha256": component_sha256,
        "capture_id": capture_id,
    }

    timestamped_jsonl_append(
        [run],
        root / "capture_runs" / f"{SLATE_DATE}.jsonl",
        "capture_run",
        captured_at=CAPTURED_AT,
    )

    return capture_id


def write_prior_lineups(snapshot_root: Path) -> int:
    """Lineup captures from before the slate, which Gate 3 reads for role state.

    Gate 3 seeds its role state from prior lineups, and with none the roles are
    unseeded. Supplied so the fixture exercises the role path the production
    projection actually takes.
    """
    root = Path(snapshot_root)

    written = 0

    for offset in (2, 4, 6):
        date = (pd.Timestamp(SLATE_DATE) - pd.Timedelta(days=offset)).date().isoformat()

        records = [
            {
                "id": index + 1,
                "game_id": 9000 + offset,
                "starter": index % 5 < 5,
                "position": position,
                "player": {"id": player_id},
                "team": {"id": team_id},
            }
            for index, (player_id, team_id, position, _) in enumerate(ROSTER)
        ]

        timestamped_jsonl_append(
            records,
            root / "lineups" / f"{date}.jsonl",
            "lineup",
            captured_at=f"{date}T22:40:00+00:00",
        )

        written += len(records)

    return written


def write_refresh_status(path: Path, outcome: str = "REFRESHED") -> Path:
    """The refresh classifier's verdict the serving caller reads readiness from."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "outcome": outcome,
                "slate_date": SLATE_DATE,
                "writes_performed": outcome == "REFRESHED",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def build(root: Path) -> dict[str, Any]:
    """Assemble the whole fixture under ``root`` and report what it holds."""
    base = Path(root)

    data_root = base / "data"
    snapshot_root = data_root / "snapshots"

    history_rows = write_history(data_root)
    capture_id = write_capture(snapshot_root)
    prior_lineups = write_prior_lineups(snapshot_root)
    refresh_status = write_refresh_status(base / "run" / "refresh.json")

    return {
        "capture_id": capture_id,
        "data_root": data_root,
        "history_rows": history_rows,
        "prior_lineup_rows": prior_lineups,
        "quoted_props": len(QUOTED_PROPS),
        "refresh_status": refresh_status,
        "roster": len(ROSTER),
        "slate_date": SLATE_DATE,
        "snapshot_root": snapshot_root,
    }
