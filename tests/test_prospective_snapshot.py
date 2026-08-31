from __future__ import annotations

from pathlib import Path
import json

import pytest

from nba_prop_quant.prospective_snapshot import (
    ProspectiveSnapshotClient,
    build_capture_id,
    canonical_records_sha256,
)
from nba_prop_quant.storage import (
    timestamped_jsonl_append,
)


DATE = "2026-10-20"
CAPTURED_AT = (
    "2026-10-20T22:40:00+00:00"
)


def write_capture(
    root: Path,
) -> str:
    game = {
        "id": 999,
        "date": DATE,
        "datetime": (
            "2026-10-20T23:00:00+00:00"
        ),
        "home_team": {
            "id": 1,
        },
        "visitor_team": {
            "id": 2,
        },
    }

    active_players = [
        {
            "id": 101,
            "first_name": "A",
            "last_name": "One",
            "team": {
                "id": 1,
            },
        },
        {
            "id": 201,
            "first_name": "B",
            "last_name": "One",
            "team": {
                "id": 2,
            },
        },
    ]

    injuries = []

    lineups = [
        {
            "id": 1,
            "game_id": 999,
            "starter": True,
            "position": "G",
            "player": {
                "id": 101,
            },
            "team": {
                "id": 1,
            },
        },
        {
            "id": 2,
            "game_id": 999,
            "starter": True,
            "position": "G",
            "player": {
                "id": 201,
            },
            "team": {
                "id": 2,
            },
        },
    ]

    props = [
        {
            "id": 5001,
            "game_id": 999,
            "player_id": 101,
            "vendor": "test",
            "prop_type": "assists",
            "line_value": "5.5",
            "market": {
                "type": "over_under",
                "over_odds": -110,
                "under_odds": -110,
            },
        }
    ]

    components = {
        "games": [
            game
        ],
        "active_players": (
            active_players
        ),
        "injuries": injuries,
        "lineups": lineups,
        "player_props": props,
    }

    specs = {
        "games": (
            "games",
            "game",
        ),
        "active_players": (
            "active_players",
            "active_player",
        ),
        "injuries": (
            "injuries",
            "injury",
        ),
        "lineups": (
            "lineups",
            "lineup",
        ),
        "player_props": (
            "player_props",
            "live_player_prop",
        ),
    }

    for name, records in (
        components.items()
    ):
        directory, snapshot_type = (
            specs[
                name
            ]
        )

        if records:
            timestamped_jsonl_append(
                records,
                root
                / directory
                / f"{DATE}.jsonl",
                snapshot_type,
                captured_at=CAPTURED_AT,
            )

    component_sha256 = {
        name: (
            canonical_records_sha256(
                records
            )
        )
        for name, records in (
            components.items()
        )
    }

    window_ids = [
        "999:T-20m"
    ]

    due_game_ids = [
        999
    ]

    capture_id = build_capture_id(
        date=DATE,
        captured_at=CAPTURED_AT,
        window_ids=window_ids,
        due_game_ids=due_game_ids,
        component_sha256=(
            component_sha256
        ),
    )

    run = {
        "date": DATE,
        "capture_reason": "scheduled",
        "window_ids": window_ids,
        "due_game_ids": (
            due_game_ids
        ),
        "grace_minutes": 5,
        "games_count": 1,
        "active_players_count": 2,
        "game_ids": [
            999
        ],
        "team_ids": [
            1,
            2,
        ],
        "injuries_count": 0,
        "lineups_count": 2,
        "player_props_count": 1,
        "error_count": 0,
        "errors": [],
        "component_sha256": (
            component_sha256
        ),
        "capture_id": capture_id,
    }

    timestamped_jsonl_append(
        [
            run
        ],
        root
        / "capture_runs"
        / f"{DATE}.jsonl",
        "capture_run",
        captured_at=CAPTURED_AT,
    )

    return capture_id


def test_snapshot_client_uses_exact_capture(
    tmp_path,
):
    capture_id = write_capture(
        tmp_path
    )

    client = (
        ProspectiveSnapshotClient(
            snapshot_dir=tmp_path,
            target_date=DATE,
            offset_minutes=20,
        )
    )

    games = client.games(
        dates=[
            DATE
        ]
    )

    assert len(
        games
    ) == 1

    assert games[
        0
    ][
        "id"
    ] == 999

    assert len(
        client.active_players(
            team_ids=[
                1,
                2,
            ]
        )
    ) == 2

    assert len(
        client.lineups(
            [
                999
            ]
        )
    ) == 2

    assert len(
        client.live_player_props(
            999
        )
    ) == 1

    lineage = (
        client.lineage_for_game(
            999
        )
    )

    assert (
        lineage[
            "gate3_capture_id"
        ]
        == capture_id
    )

    assert (
        lineage[
            "gate3_capture_window_id"
        ]
        == "999:T-20m"
    )

    assert (
        lineage[
            "gate3_input_source"
        ]
        == "scheduled_snapshot"
    )


def test_snapshot_hash_tampering_fails(
    tmp_path,
):
    write_capture(
        tmp_path
    )

    path = (
        tmp_path
        / "lineups"
        / f"{DATE}.jsonl"
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "captured_at": (
                        CAPTURED_AT
                    ),
                    "snapshot_type": (
                        "lineup"
                    ),
                    "payload": {
                        "id": 9999,
                        "game_id": 999,
                        "starter": False,
                        "player": {
                            "id": 777,
                        },
                        "team": {
                            "id": 1,
                        },
                    },
                }
            )
            + "\n"
        )

    with pytest.raises(
        RuntimeError,
        match=(
            "component hash mismatch"
        ),
    ):
        ProspectiveSnapshotClient(
            snapshot_dir=tmp_path,
            target_date=DATE,
            offset_minutes=20,
        )
