from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


NBA_TZ = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--now-utc",
        help="ISO UTC timestamp for deterministic testing",
    )
    parser.add_argument(
        "--grace-minutes",
        type=int,
        default=5,
    )
    return parser.parse_args()


def parse_now_utc(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(timezone.utc)

    text = raw

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    value = datetime.fromisoformat(text)

    if value.tzinfo is None:
        raise ValueError(
            "--now-utc must include timezone information"
        )

    return value.astimezone(timezone.utc)


def target_dates(now_utc: datetime) -> list[str]:
    local_now = now_utc.astimezone(NBA_TZ)

    return [
        local_now.date().isoformat(),
        (
            local_now.date()
            + timedelta(days=1)
        ).isoformat(),
    ]


def build_commands(
    collector: Path,
    now_utc: datetime,
    grace_minutes: int,
) -> list[list[str]]:
    now_text = now_utc.isoformat()

    commands = []

    for date in target_dates(now_utc):
        commands.append(
            [
                sys.executable,
                str(collector),
                "--date",
                date,
                "--scheduled",
                "--now-utc",
                now_text,
                "--grace-minutes",
                str(grace_minutes),
            ]
        )

    return commands


def main() -> None:
    args = parse_args()

    if args.grace_minutes <= 0:
        raise ValueError(
            "--grace-minutes must be positive"
        )

    now_utc = parse_now_utc(
        args.now_utc
    )

    collector = (
        Path(__file__).resolve().parent
        / "11_collect_snapshots.py"
    )

    failures = []

    for command in build_commands(
        collector,
        now_utc,
        args.grace_minutes,
    ):
        result = subprocess.run(
            command,
            check=False,
        )

        if result.returncode != 0:
            failures.append(
                {
                    "command": command,
                    "returncode": result.returncode,
                }
            )

    if failures:
        raise SystemExit(
            f"Snapshot cycle failures: {failures}"
        )


if __name__ == "__main__":
    main()
