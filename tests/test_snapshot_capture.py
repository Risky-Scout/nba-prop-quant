from __future__ import annotations

import json
from datetime import datetime, timedelta

from nba_prop_quant.storage import timestamped_jsonl_append


def test_timestamped_jsonl_append_preserves_multiple_snapshots(tmp_path):
    path = tmp_path / "snapshots" / "games" / "2026-10-20.jsonl"

    first = "2026-10-20T16:00:00+00:00"
    second = "2026-10-20T22:30:00+00:00"

    timestamped_jsonl_append(
        [{"id": 1}],
        path,
        snapshot_type="game",
        captured_at=first,
    )

    timestamped_jsonl_append(
        [{"id": 2}],
        path,
        snapshot_type="game",
        captured_at=second,
    )

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 2
    assert rows[0]["captured_at"] == first
    assert rows[1]["captured_at"] == second
    assert rows[0]["snapshot_type"] == "game"
    assert rows[1]["snapshot_type"] == "game"
    assert rows[0]["payload"] == {"id": 1}
    assert rows[1]["payload"] == {"id": 2}


def test_timestamped_jsonl_append_default_timestamp_is_utc(tmp_path):
    path = tmp_path / "injuries.jsonl"

    timestamped_jsonl_append(
        [{"player_id": 10}],
        path,
        snapshot_type="injury",
    )

    row = json.loads(
        path.read_text(encoding="utf-8").strip()
    )

    captured = datetime.fromisoformat(row["captured_at"])

    assert captured.utcoffset() == timedelta(0)
    assert row["snapshot_type"] == "injury"
    assert row["payload"] == {"player_id": 10}
