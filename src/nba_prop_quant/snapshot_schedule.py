from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


CAPTURE_OFFSETS_MINUTES = (
    1440,
    480,
    180,
    90,
    45,
    20,
    5,
)


def parse_tip_utc(game: dict[str, Any]) -> datetime:
    raw = game.get("datetime")

    if not raw:
        raise ValueError("game payload missing datetime")

    text = str(raw)

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    tip = datetime.fromisoformat(text)

    if tip.tzinfo is None:
        raise ValueError(
            "game datetime must include timezone information"
        )

    return tip.astimezone(timezone.utc)


def capture_schedule_for_game(
    game: dict[str, Any],
) -> list[dict[str, Any]]:
    game_id = game.get("id")

    if game_id is None:
        raise ValueError("game payload missing id")

    tip = parse_tip_utc(game)

    rows: list[dict[str, Any]] = []

    for offset_minutes in CAPTURE_OFFSETS_MINUTES:
        target = tip - timedelta(
            minutes=offset_minutes
        )

        rows.append(
            {
                "game_id": int(game_id),
                "tip_utc": tip,
                "offset_minutes": offset_minutes,
                "capture_at_utc": target,
                "window_id": (
                    f"{int(game_id)}:"
                    f"T-{offset_minutes}m"
                ),
            }
        )

    return rows


def capture_schedule(
    games: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for game in games:
        rows.extend(
            capture_schedule_for_game(game)
        )

    rows.sort(
        key=lambda row: (
            row["capture_at_utc"],
            row["game_id"],
            row["offset_minutes"],
        )
    )

    return rows


def due_windows(
    games: list[dict[str, Any]],
    now_utc: datetime,
    *,
    grace_minutes: int = 5,
) -> list[dict[str, Any]]:
    if now_utc.tzinfo is None:
        raise ValueError(
            "now_utc must be timezone-aware"
        )

    now = now_utc.astimezone(timezone.utc)

    if grace_minutes <= 0:
        raise ValueError(
            "grace_minutes must be positive"
        )

    grace = timedelta(
        minutes=grace_minutes
    )

    due: list[dict[str, Any]] = []

    for row in capture_schedule(games):
        target = row["capture_at_utc"]

        if target <= now < target + grace:
            due.append(row)

    return due
