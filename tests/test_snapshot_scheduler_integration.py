from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def load_collector():
    project = Path(
        __file__
    ).resolve().parents[1]

    script = (
        project
        / "scripts/11_collect_snapshots.py"
    )

    spec = (
        importlib.util.spec_from_file_location(
            "scheduled_snapshot_collector",
            script,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to load collector"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def settings_for(tmp_path):
    return SimpleNamespace(
        bdl_api_key="test-key",
        bdl_base_url=(
            "https://example.invalid"
        ),
        bdl_requests_per_minute=600,
        snapshot_dir=(
            tmp_path / "snapshots"
        ),
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def test_scheduled_window_is_idempotent(
    tmp_path,
    monkeypatch,
):
    collector = load_collector()
    settings = settings_for(
        tmp_path
    )

    calls = {
        "games": 0,
        "injuries": 0,
        "lineups": 0,
        "props": 0,
    }

    games = [
        {
            "id": 101,
            "datetime": (
                "2025-10-22T23:00:00.000Z"
            ),
            "home_team": {
                "id": 1,
            },
            "visitor_team": {
                "id": 2,
            },
        }
    ]

    class FakeClient:
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            pass

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            return False

        def games(
            self,
            dates,
        ):
            calls["games"] += 1
            return iter(games)

        def injuries(
            self,
            team_ids=None,
            player_ids=None,
        ):
            calls[
                "injuries"
            ] += 1

            return iter(
                [
                    {
                        "player": {
                            "id": 501,
                            "team": {
                                "id": 1,
                            },
                        },
                        "status": "Out",
                    }
                ]
            )

        def lineups(
            self,
            game_ids,
        ):
            calls[
                "lineups"
            ] += 1

            return iter(
                [
                    {
                        "id": 1,
                        "game_id": 101,
                        "starter": True,
                        "player": {
                            "id": 601,
                        },
                        "team": {
                            "id": 1,
                        },
                    }
                ]
            )

        def live_player_props(
            self,
            game_id,
        ):
            calls[
                "props"
            ] += 1

            return [
                {
                    "game_id": 101,
                    "player_id": 601,
                    "prop_type": (
                        "points"
                    ),
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
        FakeClient,
    )

    argv = [
        "11_collect_snapshots.py",
        "--date",
        "2025-10-22",
        "--scheduled",
        "--now-utc",
        "2025-10-22T22:56:00Z",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )

    collector.main()

    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )

    collector.main()

    run_path = (
        settings.snapshot_dir
        / "capture_runs"
        / "2025-10-22.jsonl"
    )

    runs = read_jsonl(
        run_path
    )

    assert len(runs) == 1

    payload = (
        runs[0]["payload"]
    )

    assert (
        payload["capture_reason"]
        == "scheduled"
    )

    assert (
        payload["window_ids"]
        == ["101:T-5m"]
    )

    assert (
        payload["due_game_ids"]
        == [101]
    )

    assert calls["games"] == 2
    assert calls["injuries"] == 1
    assert calls["lineups"] == 1
    assert calls["props"] == 1


def test_not_due_creates_no_capture(
    tmp_path,
    monkeypatch,
):
    collector = load_collector()
    settings = settings_for(
        tmp_path
    )

    games = [
        {
            "id": 101,
            "datetime": (
                "2025-10-22T23:00:00.000Z"
            ),
            "home_team": {
                "id": 1,
            },
            "visitor_team": {
                "id": 2,
            },
        }
    ]

    class FakeClient:
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            pass

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            return False

        def games(
            self,
            dates,
        ):
            return iter(games)

        def injuries(
            self,
            team_ids=None,
            player_ids=None,
        ):
            raise AssertionError(
                "injuries should not be called"
            )

        def lineups(
            self,
            game_ids,
        ):
            raise AssertionError(
                "lineups should not be called"
            )

        def live_player_props(
            self,
            game_id,
        ):
            raise AssertionError(
                "props should not be called"
            )

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
            "2025-10-22",
            "--scheduled",
            "--now-utc",
            "2025-10-22T22:35:00Z",
        ],
    )

    collector.main()

    assert not (
        settings.snapshot_dir
        / "capture_runs"
        / "2025-10-22.jsonl"
    ).exists()


def test_completed_window_reader(
    tmp_path,
):
    collector = load_collector()

    path = (
        tmp_path
        / "capture_runs.jsonl"
    )

    rows = [
        {
            "captured_at": (
                "2025-10-22T22:56:00+00:00"
            ),
            "snapshot_type": (
                "capture_run"
            ),
            "payload": {
                "window_ids": [
                    "101:T-5m",
                    "102:T-20m",
                ]
            },
        },
        {
            "captured_at": (
                "2025-10-22T22:57:00+00:00"
            ),
            "snapshot_type": (
                "capture_run"
            ),
            "payload": {
                "window_ids": [
                    "101:T-5m",
                ]
            },
        },
    ]

    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        collector.completed_window_ids(
            path
        )
        == {
            "101:T-5m",
            "102:T-20m",
        }
    )
