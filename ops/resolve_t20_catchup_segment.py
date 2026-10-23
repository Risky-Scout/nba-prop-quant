#!/usr/bin/env python3
"""The segment a catch-up run owns: from now to the end of the current one.

A catch-up exists because a scheduled segment refused and abandoned windows it
owned. It must not simply re-own that whole segment, for two reasons that pull
in opposite directions and are both satisfied by starting at *now*:

``a window that already passed cannot be captured``
    The controller fails closed when a window inside its segment is past the
    capture grace, because no later snapshot may be labelled T-20. That is
    correct, and it means a catch-up that re-owned the whole segment would
    refuse the entire evening over one window nothing can ever capture again.
    Starting at now puts the lost window outside the segment, so it stays
    uncaptured and visibly absent from the evidence record -- the honest
    outcome -- while the windows still ahead are served.

``no window may be owned twice``
    So the end is the end of the scheduled segment containing now, taken from
    the controller's own boundaries rather than restated here. The next
    scheduled segment still owns everything after it, and the shared
    ``nba-t20-production`` concurrency group serialises the overlap of a
    catch-up that is still sleeping when the next segment fires.

If now falls outside every scheduled segment -- between 05:25 and 14:25 UTC,
when no NBA T-20 window exists -- there is nothing to catch up and this says
so rather than inventing a segment.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from run_t20_production_controller import (  # noqa: E402
    ControllerRefused,
    resolve_segment,
    slate_date_for,
)

EXIT_OK = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="resolve_t20_catchup_segment",
        description=(
            "Resolve the half-open UTC segment a T-20 catch-up owns: from now "
            "to the end of the scheduled segment that contains now."
        ),
    )

    parser.add_argument(
        "--now-utc",
        default=None,
        help="override the clock, for tests",
    )
    parser.add_argument("--github-output", type=pathlib.Path, default=None)
    parser.add_argument("--summary-path", type=pathlib.Path, default=None)

    return parser.parse_args(argv)


def resolve(*, now_utc: datetime) -> dict[str, object]:
    """Segment bounds for a catch-up starting now, or a recorded no-op."""
    now = now_utc.replace(microsecond=0)

    try:
        _, segment_end = resolve_segment(now_utc=now)

    except ControllerRefused as error:
        return {
            "catchup": False,
            "now_utc": now.isoformat(),
            "reason": (
                "now falls in no scheduled T-20 segment, so there is no "
                f"evening to catch up: {error}"
            ),
        }

    return {
        "catchup": True,
        "now_utc": now.isoformat(),
        "reason": (
            f"a catch-up starting {now.isoformat()} owns every T-20 window up "
            f"to {segment_end.isoformat()}, which is where the scheduled "
            "segment containing now ends"
        ),
        "segment_end_utc": segment_end.isoformat(),
        "segment_start_utc": now.isoformat(),
        "slate_date": slate_date_for(now),
    }


def _write(path: pathlib.Path | None, text: str, *, append: bool = False) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    now = (
        datetime.fromisoformat(args.now_utc).astimezone(UTC)
        if args.now_utc
        else datetime.now(UTC)
    )

    resolved = resolve(now_utc=now)

    lines = [f"catchup={'true' if resolved['catchup'] else 'false'}"]

    if resolved["catchup"]:
        lines.extend(
            [
                f"segment_start_utc={resolved['segment_start_utc']}",
                f"segment_end_utc={resolved['segment_end_utc']}",
                f"slate_date={resolved['slate_date']}",
            ]
        )

    _write(args.github_output, "\n".join(lines) + "\n", append=True)

    _write(
        args.summary_path,
        f"## T-20 catch-up segment\n\n{resolved['reason']}\n",
        append=True,
    )

    print(json.dumps(resolved, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
