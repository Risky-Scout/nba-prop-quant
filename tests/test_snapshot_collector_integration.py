from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace


def load_collector():
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts/11_collect_snapshots.py"

    spec = importlib.util.spec_from_file_location(
        "snapshot_collector_under_test",
        script,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load snapshot collector")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]


def make_settings(tmp_path: Path):
    return SimpleNamespace(
        bdl_api_key="test-key",
        bdl_base_url="https://example.invalid",
        bdl_requests_per_minute=600,
        snapshot_dir=tmp_path / "snapshots",
    )


def test_collector_captures_complete_point_in_time_run(
    tmp_path,
    monkeypatch,
):
    collector = load_collector()
    settings = make_settings(tmp_path)

    games = [
        {
            "id": 101,
            "home_team": {"id": 1},
            "visitor_team": {"id": 2},
        },
        {
            "id": 102,
            "home_team": {"id": 3},
            "visitor_team": {"id": 4},
        },
    ]

    injuries = [
        {
            "player": {
                "id": 501,
                "team": {"id": 1},
            },
            "status": "Out",
        }
    ]

    lineups = [
        {
            "id": 9001,
            "game_id": 101,
            "starter": True,
            "position": "G",
            "player": {"id": 601},
            "team": {"id": 1},
        },
        {
            "id": 9002,
            "game_id": 101,
            "starter": False,
            "position": "F",
            "player": {"id": 602},
            "team": {"id": 1},
        },
    ]

    props_by_game = {
        101: [
            {
                "game_id": 101,
                "player_id": 601,
                "prop_type": "points",
            }
        ],
        102: [
            {
                "game_id": 102,
                "player_id": 701,
                "prop_type": "assists",
            }
        ],
    }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def games(self, dates):
            assert dates == ["2026-10-20"]
            return iter(games)

        def injuries(self, team_ids=None, player_ids=None):
            assert team_ids == [1, 2, 3, 4]
            return iter(injuries)

        def lineups(self, game_ids):
            assert game_ids == [101, 102]
            return iter(lineups)

        def live_player_props(self, game_id):
            return props_by_game[game_id]

    monkeypatch.setattr(
        collector,
        "get_settings",
        lambda: settings,
    )

    monkeypatch.setattr(
        collector,
        "BDLClient",
        FakeClient,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "11_collect_snapshots.py",
            "--date",
            "2026-10-20",
        ],
    )

    collector.main()

    base = settings.snapshot_dir

    game_rows = read_jsonl(
        base / "games/2026-10-20.jsonl"
    )

    injury_rows = read_jsonl(
        base / "injuries/2026-10-20.jsonl"
    )

    lineup_rows = read_jsonl(
        base / "lineups/2026-10-20.jsonl"
    )

    prop_rows = read_jsonl(
        base / "player_props/2026-10-20.jsonl"
    )

    run_rows = read_jsonl(
        base / "capture_runs/2026-10-20.jsonl"
    )

    assert len(game_rows) == 2
    assert len(injury_rows) == 1
    assert len(lineup_rows) == 2
    assert len(prop_rows) == 2
    assert len(run_rows) == 1

    all_rows = (
        game_rows
        + injury_rows
        + lineup_rows
        + prop_rows
        + run_rows
    )

    timestamps = {
        row["captured_at"]
        for row in all_rows
    }

    assert len(timestamps) == 1

    captured = datetime.fromisoformat(
        next(iter(timestamps))
    )

    assert captured.utcoffset() == timedelta(0)

    assert {
        row["snapshot_type"]
        for row in game_rows
    } == {"game"}

    assert {
        row["snapshot_type"]
        for row in injury_rows
    } == {"injury"}

    assert {
        row["snapshot_type"]
        for row in lineup_rows
    } == {"lineup"}

    assert {
        row["snapshot_type"]
        for row in prop_rows
    } == {"live_player_prop"}

    run = run_rows[0]["payload"]

    assert run["date"] == "2026-10-20"
    assert run["games_count"] == 2
    assert run["game_ids"] == [101, 102]
    assert run["team_ids"] == [1, 2, 3, 4]
    assert run["injuries_count"] == 1
    assert run["lineups_count"] == 2
    assert run["player_props_count"] == 2
    assert run["error_count"] == 0
    assert run["errors"] == []


def test_collector_records_partial_failures(
    tmp_path,
    monkeypatch,
):
    collector = load_collector()
    settings = make_settings(tmp_path)

    games = [
        {
            "id": 101,
            "home_team": {"id": 1},
            "visitor_team": {"id": 2},
        },
        {
            "id": 102,
            "home_team": {"id": 3},
            "visitor_team": {"id": 4},
        },
    ]

    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def games(self, dates):
            return iter(games)

        def injuries(self, team_ids=None, player_ids=None):
            raise RuntimeError("injury endpoint unavailable")

        def lineups(self, game_ids):
            raise RuntimeError("lineup endpoint unavailable")

        def live_player_props(self, game_id):
            if game_id == 102:
                raise RuntimeError(
                    "props unavailable for game 102"
                )

            return [
                {
                    "game_id": 101,
                    "player_id": 601,
                    "prop_type": "points",
                }
            ]

    monkeypatch.setattr(
        collector,
        "get_settings",
        lambda: settings,
    )

    monkeypatch.setattr(
        collector,
        "BDLClient",
        FailingClient,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "11_collect_snapshots.py",
            "--date",
            "2026-10-20",
        ],
    )

    collector.main()

    base = settings.snapshot_dir

    assert (
        base / "games/2026-10-20.jsonl"
    ).exists()

    assert not (
        base / "injuries/2026-10-20.jsonl"
    ).exists()

    assert not (
        base / "lineups/2026-10-20.jsonl"
    ).exists()

    assert (
        base / "player_props/2026-10-20.jsonl"
    ).exists()

    run_rows = read_jsonl(
        base / "capture_runs/2026-10-20.jsonl"
    )

    assert len(run_rows) == 1

    run = run_rows[0]["payload"]

    assert run["games_count"] == 2
    assert run["injuries_count"] == 0
    assert run["lineups_count"] == 0
    assert run["player_props_count"] == 1
    assert run["error_count"] == 3

    components = [
        error["component"]
        for error in run["errors"]
    ]

    assert components == [
        "injuries",
        "lineups",
        "player_props",
    ]

    assert run["errors"][2]["game_id"] == 102
