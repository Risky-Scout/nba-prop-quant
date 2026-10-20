"""Point-in-time captures in exactly the shape the collector writes them.

``scripts/11_collect_snapshots.py`` does not write a snapshot, it writes six
append-only JSONL components plus a ``capture_run`` envelope whose payload
carries a per-component ``canonical_records_sha256`` and a ``capture_id``
derived from all of them. ``select_capture_bundle`` then recomputes every one
of those digests before pricing is allowed to read the capture.

A hand-rolled fixture that wrote plausible-looking JSON would therefore prove
nothing: it would be eligible or refused for reasons of its own rather than
for the reasons a real capture is. Everything here is built with the
repository's own hashing and append helpers, so a capture written by this
module is accepted by exactly the checks that accept a production one, and a
capture deliberately corrupted here is refused by exactly the check that
would have caught the corruption in production.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from nba_prop_quant.prospective_snapshot import (
    COMPONENT_SPECS,
    build_capture_id,
    canonical_records_sha256,
)
from nba_prop_quant.storage import timestamped_jsonl_append

#: An opening-week slate date. Only the shape of the date matters; it is
#: fixed so that every test in the suite talks about the same day.
SLATE = "2026-10-20"


def at(hour: int, minute: int = 0, *, day: int = 20) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


def game(game_id: int, tip_utc: datetime, *, home: int = 1, away: int = 2):
    return {
        "id": game_id,
        "date": SLATE,
        "datetime": tip_utc.isoformat().replace("+00:00", "Z"),
        "home_team": {"id": home, "abbreviation": "HHH"},
        "visitor_team": {"id": away, "abbreviation": "VVV"},
    }


def prop(game_id: int, player_id: int):
    return {
        "game_id": game_id,
        "player_id": player_id,
        "prop_type": "points",
        "line": 24.5,
    }


def write_capture(
    snapshot_dir: Path,
    *,
    date: str,
    captured_at: str,
    window_ids: list[str],
    due_game_ids: list[int],
    games: list[dict],
    props: list[dict] | None = None,
    reason: str = "scheduled",
    error_count: int = 0,
    component_overrides: dict[str, str] | None = None,
) -> str:
    """One append-only capture, in exactly the shape the collector writes.

    ``reason``, ``error_count`` and ``component_overrides`` exist so a test
    can produce the specific kinds of ineligible capture production has to
    refuse rather than only the eligible kind.
    """
    components = {
        "games": list(games),
        "active_players": [],
        "injuries": [],
        "lineups": [],
        "player_props": list(props or []),
    }

    for name, (directory, snapshot_type) in COMPONENT_SPECS.items():
        rows = components[name]

        if rows:
            timestamped_jsonl_append(
                rows,
                snapshot_dir / directory / f"{date}.jsonl",
                snapshot_type=snapshot_type,
                captured_at=captured_at,
            )

    component_sha256 = {
        name: canonical_records_sha256(rows)
        for name, rows in components.items()
    }

    if component_overrides:
        component_sha256.update(component_overrides)

    capture_id = build_capture_id(
        date=date,
        captured_at=captured_at,
        window_ids=window_ids,
        due_game_ids=due_game_ids,
        component_sha256=component_sha256,
    )

    timestamped_jsonl_append(
        [
            {
                "date": date,
                "capture_reason": reason,
                "window_ids": window_ids,
                "due_game_ids": due_game_ids,
                "grace_minutes": 5,
                "games_count": len(components["games"]),
                "active_players_count": 0,
                "game_ids": sorted(
                    int(row["id"]) for row in components["games"]
                ),
                "team_ids": [1, 2],
                "injuries_count": 0,
                "lineups_count": 0,
                "player_props_count": len(components["player_props"]),
                "error_count": error_count,
                "errors": [],
                "component_sha256": component_sha256,
                "capture_id": capture_id,
            }
        ],
        snapshot_dir / "capture_runs" / f"{date}.jsonl",
        snapshot_type="capture_run",
        captured_at=captured_at,
    )

    return capture_id


def eligible_game_ids(snapshot_dir: Path, date: str) -> list[int]:
    """Every game with a scheduled T-20 window recorded for this slate."""
    path = snapshot_dir / "capture_runs" / f"{date}.jsonl"

    found: set[int] = set()

    if not path.exists():
        return []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        envelope = json.loads(line)

        payload = envelope.get("payload") or {}

        if payload.get("capture_reason") != "scheduled":
            continue

        for window_id in payload.get("window_ids", []):
            if str(window_id).endswith(":T-20m"):
                found.add(int(str(window_id).split(":", 1)[0]))

    return sorted(found)
