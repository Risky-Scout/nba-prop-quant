from __future__ import annotations

import argparse
import fcntl
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from nba_prop_quant.api import BDLClient
from nba_prop_quant.settings import get_settings
from nba_prop_quant.snapshot_schedule import (
    CAPTURE_OFFSETS_MINUTES,
    due_windows,
)
from nba_prop_quant.storage import timestamped_jsonl_append
from nba_prop_quant.prospective_snapshot import (
    build_capture_id,
    canonical_records_sha256,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        required=True,
        help="YYYY-MM-DD",
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
    )
    parser.add_argument(
        "--now-utc",
    )
    parser.add_argument(
        "--grace-minutes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--offset-minutes",
        action="append",
        type=int,
        default=None,
        help=(
            "Capture only the scheduled windows at this offset. "
            "Repeatable. Omitting it selects every offset the capture "
            "schedule defines, which is what research and manual "
            "callers already get."
        ),
    )
    return parser.parse_args()


def selected_offsets(
    raw: list[int] | None,
) -> tuple[int, ...] | None:
    """The offsets a caller narrowed the scheduled capture to.

    ``None`` means no narrowing, which is the behaviour every existing
    caller has. A named offset the schedule does not define is refused
    rather than silently matching nothing: a production caller that asked
    for a window that cannot exist would otherwise capture nothing and
    report success.
    """
    if raw is None:
        return None

    requested = {
        int(value)
        for value in raw
    }

    unknown = sorted(
        requested
        - set(
            CAPTURE_OFFSETS_MINUTES
        )
    )

    if unknown:
        raise ValueError(
            "--offset-minutes must name offsets the capture schedule "
            f"defines; {unknown} are not among "
            f"{list(CAPTURE_OFFSETS_MINUTES)}"
        )

    return tuple(
        sorted(
            requested
        )
    )


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


def completed_window_ids(
    capture_run_path: Path,
) -> set[str]:
    if not capture_run_path.exists():
        return set()

    completed: set[str] = set()

    for line in capture_run_path.read_text(
        encoding="utf-8",
    ).splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row.get("snapshot_type") != "capture_run":
            continue

        payload = row.get("payload") or {}

        for window_id in payload.get(
            "window_ids",
            [],
        ):
            completed.add(
                str(window_id)
            )

    return completed


#: How often a sweep retries the claim while another sweep holds it.
CAPTURE_CLAIM_POLL_SECONDS = 0.05


@contextmanager
def capture_window_claim(
    capture_run_path: Path,
    *,
    timeout_seconds: float,
):
    """The exclusive right to sweep one slate, held across the whole sweep.

    ``completed_window_ids`` is the authority on which windows have already
    been captured, and the collector consults it, collects, and then records
    its own answer. Those three steps are a read-modify-write of an
    append-only log with nothing holding the file between them, so two
    sweeps whose collections overlap both read the window as unclaimed and
    both append a scheduled capture naming it. ``select_capture_bundle``
    then refuses that window for being ambiguous -- permanently, because the
    log is append-only and both captures are genuine -- and the game cannot
    be priced at all. That is what happened on 2026-10-23: three games
    tipping at 02:00 UTC were swept twice inside one grace window and none
    of the three was ever served.

    The claim is an exclusive ``flock`` on a lock file beside the log, held
    from before the read until after the append, so the second sweep reads
    the claim set the first one left rather than the one it found. It covers
    the component appends too, and that is not incidental: a capture's
    component digests are recomputed from every row carrying its
    ``captured_at``, so two sweeps invoked with the same ``--now-utc`` write
    rows into one another's digests and corrupt both captures even if only
    one of them claims the window.

    The wait is bounded by the capture grace window, which is the span in
    which a due window may legitimately be captured at all: a sweep that
    cannot get the claim inside it could not produce a capture pricing would
    accept, so it refuses and exits non-zero rather than appending a second
    claim to a window the holder is in the middle of claiming. The lock is a
    file descriptor, so a sweep that crashes releases it.
    """
    lock_path = (
        capture_run_path.parent
        / f"{capture_run_path.name}.claim"
    )

    lock_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    deadline = (
        time.monotonic()
        + float(timeout_seconds)
    )

    with lock_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        while True:
            try:
                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX
                    | fcntl.LOCK_NB,
                )
            except OSError:
                if (
                    time.monotonic()
                    >= deadline
                ):
                    raise RuntimeError(
                        "another capture sweep has held "
                        f"{lock_path} for more than "
                        f"{float(timeout_seconds):g}s; refusing "
                        "rather than recording a second "
                        "scheduled capture of a window it may "
                        "already have claimed"
                    ) from None

                time.sleep(
                    CAPTURE_CLAIM_POLL_SECONDS
                )

                continue

            break

        try:
            yield lock_path
        finally:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_UN,
            )


def main() -> None:
    args = parse_args()
    settings = get_settings()
    now_utc = parse_now_utc(
        args.now_utc
    )

    offsets = selected_offsets(
        args.offset_minutes
    )

    if (
        offsets is not None
        and not args.scheduled
    ):
        raise ValueError(
            "--offset-minutes narrows which scheduled capture windows "
            "are due, so it has no meaning without --scheduled"
        )

    errors: list[dict[str, object]] = []

    with (
        BDLClient(
            api_key=settings.bdl_api_key,
            base_url=settings.bdl_base_url,
            requests_per_minute=settings.bdl_requests_per_minute,
        ) as client,
        capture_window_claim(
            settings.snapshot_dir
            / "capture_runs"
            / f"{args.date}.jsonl",
            timeout_seconds=(
                args.grace_minutes * 60
            ),
        ),
    ):
        games = list(
            client.games(
                dates=[args.date]
            )
        )

        window_ids: list[str] = []
        due_game_ids: list[int] = []

        if args.scheduled:
            due = due_windows(
                games,
                now_utc,
                grace_minutes=args.grace_minutes,
            )

            if offsets is not None:
                due = [
                    row
                    for row in due
                    if int(
                        row[
                            "offset_minutes"
                        ]
                    )
                    in offsets
                ]

            capture_run_path = (
                settings.snapshot_dir
                / "capture_runs"
                / f"{args.date}.jsonl"
            )

            completed = completed_window_ids(
                capture_run_path
            )

            unseen = [
                row
                for row in due
                if row["window_id"]
                not in completed
            ]

            if not unseen:
                console.print(
                    "[cyan]No unseen scheduled "
                    "capture windows due[/cyan]"
                )
                return

            window_ids = sorted(
                {
                    str(
                        row["window_id"]
                    )
                    for row in unseen
                }
            )

            due_game_ids = sorted(
                {
                    int(
                        row["game_id"]
                    )
                    for row in unseen
                }
            )

        captured_at = (
            now_utc.isoformat()
        )

        game_ids = sorted(
            int(game["id"])
            for game in games
            if game.get("id") is not None
        )

        team_ids = sorted(
            {
                int(team["id"])
                for game in games
                for team in (
                    game.get(
                        "home_team",
                        {},
                    ),
                    game.get(
                        "visitor_team",
                        {},
                    ),
                )
                if team
                and team.get("id")
                is not None
            }
        )

        active_player_method = getattr(
            client,
            "active_players",
            None,
        )

        if active_player_method is None:
            active_players = []
        else:
            try:
                active_players = list(
                    active_player_method(
                        team_ids=team_ids
                    )
                )
            except RuntimeError as exc:
                active_players = []

                errors.append(
                    {
                        "component": "active_players",
                        "error": str(exc),
                    }
                )

                console.print(
                    "[yellow]Active-player capture "
                    f"unavailable[/yellow]: {exc}"
                )

        if games:
            timestamped_jsonl_append(
                games,
                settings.snapshot_dir
                / "games"
                / f"{args.date}.jsonl",
                snapshot_type="game",
                captured_at=captured_at,
            )

        if active_players:
            timestamped_jsonl_append(
                active_players,
                settings.snapshot_dir
                / "active_players"
                / f"{args.date}.jsonl",
                snapshot_type="active_player",
                captured_at=captured_at,
            )


        try:
            injuries = (
                list(
                    client.injuries(
                        team_ids=team_ids
                    )
                )
                if team_ids
                else []
            )
        except RuntimeError as exc:
            injuries = []

            errors.append(
                {
                    "component": "injuries",
                    "error": str(exc),
                }
            )

            console.print(
                "[yellow]Injury capture "
                f"unavailable[/yellow]: {exc}"
            )

        if injuries:
            timestamped_jsonl_append(
                injuries,
                settings.snapshot_dir
                / "injuries"
                / f"{args.date}.jsonl",
                snapshot_type="injury",
                captured_at=captured_at,
            )

        try:
            lineups = (
                list(
                    client.lineups(
                        game_ids
                    )
                )
                if game_ids
                else []
            )
        except RuntimeError as exc:
            lineups = []

            errors.append(
                {
                    "component": "lineups",
                    "error": str(exc),
                }
            )

            console.print(
                "[yellow]Lineup capture "
                f"unavailable[/yellow]: {exc}"
            )

        if lineups:
            timestamped_jsonl_append(
                lineups,
                settings.snapshot_dir
                / "lineups"
                / f"{args.date}.jsonl",
                snapshot_type="lineup",
                captured_at=captured_at,
            )

        props_count = 0
        all_props = []

        for game in games:
            game_id = int(
                game["id"]
            )

            try:
                props = (
                    client.live_player_props(
                        game_id
                    )
                )
            except RuntimeError as exc:
                props = []

                errors.append(
                    {
                        "component": "player_props",
                        "game_id": game_id,
                        "error": str(exc),
                    }
                )

            if props:
                props_count += len(
                    props
                )

                all_props.extend(
                    props
                )

                timestamped_jsonl_append(
                    props,
                    settings.snapshot_dir
                    / "player_props"
                    / f"{args.date}.jsonl",
                    snapshot_type="live_player_prop",
                    captured_at=captured_at,
                )

        component_records = {
            "games": games,
            "active_players": active_players,
            "injuries": injuries,
            "lineups": lineups,
            "player_props": all_props,
        }

        component_sha256 = {
            name: canonical_records_sha256(
                records
            )
            for name, records in (
                component_records.items()
            )
        }

        capture_id = build_capture_id(
            date=args.date,
            captured_at=captured_at,
            window_ids=window_ids,
            due_game_ids=due_game_ids,
            component_sha256=component_sha256,
        )

        capture_run = {
            "date": args.date,
            "capture_reason": (
                "scheduled"
                if args.scheduled
                else "manual"
            ),
            "window_ids": window_ids,
            "due_game_ids": due_game_ids,
            "grace_minutes": (
                args.grace_minutes
                if args.scheduled
                else None
            ),
            "games_count": len(
                games
            ),
            "active_players_count": len(
                active_players
            ),
            "game_ids": game_ids,
            "team_ids": team_ids,
            "injuries_count": len(
                injuries
            ),
            "lineups_count": len(
                lineups
            ),
            "player_props_count": (
                props_count
            ),
            "error_count": len(
                errors
            ),
            "errors": errors,
            "component_sha256": component_sha256,
            "capture_id": capture_id,
        }

        timestamped_jsonl_append(
            [capture_run],
            settings.snapshot_dir
            / "capture_runs"
            / f"{args.date}.jsonl",
            snapshot_type="capture_run",
            captured_at=captured_at,
        )

        console.print(
            "[green]Captured point-in-time "
            "snapshot[/green]: "
            f"{len(games)} games, "
            f"{len(injuries)} injuries, "
            f"{len(lineups)} lineups, "
            f"{props_count} player props, "
            f"{len(errors)} errors, "
            f"{len(window_ids)} scheduled windows"
        )


if __name__ == "__main__":
    main()
