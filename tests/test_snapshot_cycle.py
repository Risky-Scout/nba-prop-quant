from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path


def load_cycle():
    project = Path(
        __file__
    ).resolve().parents[1]

    path = (
        project
        / "scripts/12_run_snapshot_cycle.py"
    )

    spec = importlib.util.spec_from_file_location(
        "snapshot_cycle_under_test",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load snapshot cycle"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def test_target_dates_regular_evening():
    cycle = load_cycle()

    now = datetime(
        2026,
        10,
        20,
        22,
        0,
        tzinfo=timezone.utc,
    )

    assert cycle.target_dates(now) == [
        "2026-10-20",
        "2026-10-21",
    ]


def test_target_dates_after_midnight_utc():
    cycle = load_cycle()

    now = datetime(
        2026,
        10,
        21,
        2,
        0,
        tzinfo=timezone.utc,
    )

    assert cycle.target_dates(now) == [
        "2026-10-20",
        "2026-10-21",
    ]


def test_builds_two_scheduled_commands():
    cycle = load_cycle()

    now = datetime(
        2026,
        10,
        20,
        22,
        0,
        tzinfo=timezone.utc,
    )

    collector = Path(
        "/tmp/11_collect_snapshots.py"
    )

    commands = cycle.build_commands(
        collector,
        now,
        5,
    )

    assert len(commands) == 2

    assert commands[0][1] == str(
        collector
    )

    assert "--scheduled" in commands[0]
    assert "--scheduled" in commands[1]

    assert commands[0][
        commands[0].index("--date") + 1
    ] == "2026-10-20"

    assert commands[1][
        commands[1].index("--date") + 1
    ] == "2026-10-21"

    assert commands[0][
        commands[0].index(
            "--grace-minutes"
        )
        + 1
    ] == "5"
