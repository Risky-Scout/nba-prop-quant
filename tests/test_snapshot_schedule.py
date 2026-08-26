from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nba_prop_quant.snapshot_schedule import (
    CAPTURE_OFFSETS_MINUTES,
    capture_schedule,
    capture_schedule_for_game,
    due_windows,
    parse_tip_utc,
)


def sample_game():
    return {
        "id": 18446821,
        "date": "2025-10-22",
        "datetime": "2025-10-22T23:00:00.000Z",
    }


def test_parse_tip_utc():
    tip = parse_tip_utc(sample_game())

    assert tip == datetime(
        2025,
        10,
        22,
        23,
        0,
        tzinfo=timezone.utc,
    )


def test_capture_schedule_contains_all_offsets():
    rows = capture_schedule_for_game(
        sample_game()
    )

    assert len(rows) == len(
        CAPTURE_OFFSETS_MINUTES
    )

    assert {
        row["offset_minutes"]
        for row in rows
    } == set(CAPTURE_OFFSETS_MINUTES)

    assert all(
        row["capture_at_utc"]
        < row["tip_utc"]
        for row in rows
    )


def test_five_minute_window():
    rows = capture_schedule_for_game(
        sample_game()
    )

    five = next(
        row
        for row in rows
        if row["offset_minutes"] == 5
    )

    assert five["capture_at_utc"] == datetime(
        2025,
        10,
        22,
        22,
        55,
        tzinfo=timezone.utc,
    )

    assert five["window_id"] == (
        "18446821:T-5m"
    )


def test_due_window_identification():
    now = datetime(
        2025,
        10,
        22,
        22,
        56,
        tzinfo=timezone.utc,
    )

    due = due_windows(
        [sample_game()],
        now,
        grace_minutes=5,
    )

    assert len(due) == 1
    assert due[0]["offset_minutes"] == 5


def test_not_due_outside_grace():
    now = datetime(
        2025,
        10,
        22,
        22,
        54,
        tzinfo=timezone.utc,
    )

    assert due_windows(
        [sample_game()],
        now,
        grace_minutes=5,
    ) == []


def test_schedule_sorts_multiple_games():
    games = [
        {
            "id": 2,
            "datetime": "2025-10-23T01:30:00.000Z",
        },
        {
            "id": 1,
            "datetime": "2025-10-22T23:00:00.000Z",
        },
    ]

    rows = capture_schedule(games)

    times = [
        row["capture_at_utc"]
        for row in rows
    ]

    assert times == sorted(times)


def test_missing_datetime_fails_closed():
    with pytest.raises(
        ValueError,
        match="missing datetime",
    ):
        parse_tip_utc({"id": 1})


def test_naive_now_fails_closed():
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        due_windows(
            [sample_game()],
            datetime(2025, 10, 22, 22, 55),
        )
