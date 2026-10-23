#!/usr/bin/env python3
"""What evidence actually exists for a slate, read from the production host.

Three evenings of this season have no published predictions, and the question
that has to be answered before anything is reconstructed is not "what went
wrong" -- the workflow logs answer that -- but "what, if anything, did the
production machine actually keep". Only the machine can answer it, so this is
the step that asks, and it asks by reading.

WHY THIS EXISTS RATHER THAN A RECONSTRUCTION
--------------------------------------------

A prediction is prospective evidence only if it was made from inputs that
existed before tip-off and was recorded before the result was known. There
are exactly three states a missing slate can be in, and they license three
different actions:

    AUTHENTIC_PREDICTIONS
        A T-20 capture, a served slate and a published release all exist.
        There is nothing to recover; the evidence is already there.

    PREGAME_SOURCE_ONLY
        T-20 capture bundles exist and are provably pre-tip, but no
        prediction artifact was ever produced from them. The inputs are
        immutable and were frozen before the game, so running the accepted
        model against them is a strict as-of reconstruction: it is not
        prospective evidence, because nobody published it at the time, but it
        is an honest statement of what the model would have said.

    NO_PREGAME_SOURCE
        No capture was taken. There is no immutable pre-tip input to
        reconstruct from, and anything produced for that date would be built
        from a market snapshot taken after the game or from a schedule
        endpoint that now carries the result. Nothing may be generated.

The last case is the common one and it is the reason this script reports
rather than repairs. A reconstruction built on a post-tip source would look
exactly like one built on a pre-tip source in every artifact downstream of
it, which is precisely why the distinction has to be established here, from
the capture log, and recorded.

WHAT PRE-TIP MEANS HERE
-----------------------

Each capture-run envelope records its own ``captured_at`` and the games it
was taken for, and the games component captured alongside it records those
games' scheduled start times as they stood at capture time. So pre-tip is
decided from the capture's own contents and not from a schedule read today --
today's schedule read would return a completed game.

WHAT IT DOES NOT DO
-------------------

It reads. It imports nothing from the package, so it runs on the probe's two
wheels and costs nothing; it writes one receipt and makes no decision that
anything downstream is obliged to honour.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import UTC, datetime
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Standard library only, like the segment probe's imports, so this runs
# before the scientific environment is installed.
from nba_prop_quant.snapshot_schedule import parse_tip_utc  # noqa: E402

#: Where the collector appends, relative to the snapshot directory.
CAPTURE_RUNS = "capture_runs"

#: The games component, which is what carries the tip times a capture saw.
GAMES_COMPONENT = "games"

#: The primary production window. A capture at any other offset is recorded
#: but is not what pricing consumes.
PRODUCTION_OFFSET_MINUTES = 20

CLASSIFICATION_AUTHENTIC = "AUTHENTIC_PREDICTIONS"
CLASSIFICATION_SOURCE_ONLY = "PREGAME_SOURCE_ONLY"
CLASSIFICATION_NONE = "NO_PREGAME_SOURCE"


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    rows: list[dict[str, Any]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"snapshot_type": "UNREADABLE_RECORD", "raw": line})

    return rows


def parse_utc(raw: object) -> datetime | None:
    if not raw:
        return None

    try:
        moment = datetime.fromisoformat(str(raw))
    except ValueError:
        return None

    return (
        moment.replace(tzinfo=UTC)
        if moment.tzinfo is None
        else moment.astimezone(UTC)
    )


def file_record(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path)}

    stat = path.stat()

    return {
        "bytes": stat.st_size,
        "exists": True,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=UTC
        ).isoformat(),
        "path": str(path),
    }


def tip_times(snapshot_dir: pathlib.Path, slate_date: str) -> dict[int, str]:
    """Scheduled starts as the captures themselves recorded them.

    Read from the captured games component rather than from the API, because
    a schedule read today returns a finished game and would make every
    capture look pre-tip or post-tip according to today's clock.
    """
    tips: dict[int, str] = {}

    path = snapshot_dir / GAMES_COMPONENT / f"{slate_date}.jsonl"

    for envelope in read_jsonl(path):
        payload = envelope.get("payload")

        if not isinstance(payload, dict) or payload.get("id") is None:
            continue

        # The schedule module's own parser, so a tip is read here exactly as
        # the capture scheduler read it when it computed the T-20 time.
        try:
            tips[int(payload["id"])] = parse_tip_utc(payload).isoformat()
        except (ValueError, TypeError):
            continue

    return tips


def capture_inventory(
    snapshot_dir: pathlib.Path, slate_date: str
) -> dict[str, Any]:
    runs = [
        row
        for row in read_jsonl(
            snapshot_dir / CAPTURE_RUNS / f"{slate_date}.jsonl"
        )
        if row.get("snapshot_type") == "capture_run"
    ]

    tips = tip_times(snapshot_dir, slate_date)

    captures: list[dict[str, Any]] = []

    for envelope in runs:
        payload = envelope.get("payload") or {}

        captured_at = parse_utc(envelope.get("captured_at"))

        due = [int(value) for value in (payload.get("due_game_ids") or [])]

        # Pre-tip is decided per game, against the tip the capture itself
        # recorded. A capture is only usable as a pregame source for the
        # games it was actually taken before.
        pregame = []

        posttip = []

        for game_id in due:
            tip = parse_utc(tips.get(game_id))

            if captured_at is None or tip is None:
                posttip.append(game_id)

                continue

            (pregame if captured_at < tip else posttip).append(game_id)

        captures.append(
            {
                "capture_reason": payload.get("capture_reason"),
                "captured_at_utc": (
                    captured_at.isoformat() if captured_at else None
                ),
                "due_game_ids": due,
                "error_count": payload.get("error_count"),
                "games_count": payload.get("games_count"),
                "post_tip_game_ids": sorted(posttip),
                "pregame_game_ids": sorted(pregame),
                "window_ids": [
                    str(value) for value in (payload.get("window_ids") or [])
                ],
            }
        )

    production_windows = sorted(
        {
            window_id
            for capture in captures
            for window_id in capture["window_ids"]
            if window_id.endswith(f"T-{PRODUCTION_OFFSET_MINUTES}m")
        }
    )

    pregame_games = sorted(
        {
            game_id
            for capture in captures
            for game_id in capture["pregame_game_ids"]
        }
    )

    return {
        "capture_count": len(captures),
        "captures": captures,
        "pregame_game_ids": pregame_games,
        "production_t20_windows_captured": production_windows,
        "tip_times_recorded_by_captures": {
            str(key): value for key, value in sorted(tips.items())
        },
    }


def classify(
    *,
    captures: dict[str, Any],
    projections: dict[str, Any],
    priced_markets: dict[str, Any],
    releases: list[dict[str, Any]],
) -> dict[str, str]:
    has_pregame_source = bool(captures["pregame_game_ids"]) and bool(
        captures["production_t20_windows_captured"]
    )

    published = [
        record for record in releases if record.get("rows_published")
    ]

    if published and projections["exists"] and priced_markets["exists"]:
        return {
            "classification": CLASSIFICATION_AUTHENTIC,
            "reason": (
                f"{len(published)} published release(s) exist alongside the "
                "slate's projections and priced markets, so the prospective "
                "record for this slate is already complete"
            ),
        }

    if has_pregame_source:
        return {
            "classification": CLASSIFICATION_SOURCE_ONLY,
            "reason": (
                f"{len(captures['production_t20_windows_captured'])} "
                f"T-{PRODUCTION_OFFSET_MINUTES} window(s) were captured "
                f"pre-tip for {len(captures['pregame_game_ids'])} game(s) and "
                "no publication was made from them, so a strict as-of "
                "reconstruction from these immutable inputs is possible"
            ),
        }

    return {
        "classification": CLASSIFICATION_NONE,
        "reason": (
            "no pre-tip capture exists for this slate, so there is no "
            "immutable pregame input to reconstruct from and nothing may be "
            "generated for it"
        ),
    }


def inventory(
    *, data_root: pathlib.Path, slate_date: str
) -> dict[str, Any]:
    data_root = pathlib.Path(data_root).expanduser()

    snapshot_dir = data_root / "snapshots"

    processed = data_root / "processed"

    captures = capture_inventory(snapshot_dir, slate_date)

    projections = file_record(
        processed / "projections" / f"{slate_date}.parquet"
    )

    priced_markets = file_record(
        processed / "priced_markets" / f"{slate_date}.parquet"
    )

    releases = release_inventory(data_root, slate_date)

    return {
        "data_root": str(data_root),
        "distributions": file_record(
            processed / "distributions" / f"{slate_date}.parquet"
        ),
        "evidence_ledger": ledger_inventory(processed, slate_date),
        "grading_receipt": file_record(
            processed / "incumbent_grading" / f"{slate_date}.json"
        ),
        "priced_markets": priced_markets,
        "projections": projections,
        "releases": releases,
        "serving_receipt": serving_receipt(processed, slate_date),
        "slate_date": slate_date,
        "snapshot_dir": str(snapshot_dir),
        "t20_captures": captures,
        **classify(
            captures=captures,
            projections=projections,
            priced_markets=priced_markets,
            releases=releases,
        ),
    }


def serving_receipt(
    processed: pathlib.Path, slate_date: str
) -> dict[str, Any]:
    path = processed / "incumbent_serving" / f"{slate_date}.json"

    record = file_record(path)

    if not record["exists"]:
        return record

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {**record, "unreadable": str(error)}

    return {
        **record,
        "outcome": document.get("outcome"),
        "production_code_sha": document.get("production_code_sha"),
        "slate_readiness": document.get("slate_readiness"),
    }


def ledger_inventory(
    processed: pathlib.Path, slate_date: str
) -> dict[str, Any]:
    path = processed / "t20_evidence" / f"{slate_date}.jsonl"

    record = file_record(path)

    if not record["exists"]:
        return record

    entries = read_jsonl(path)

    return {
        **record,
        "events": [str(entry.get("event")) for entry in entries],
        "outcomes": sorted(
            {
                str((entry.get("detail") or {}).get("outcome"))
                for entry in entries
                if (entry.get("detail") or {}).get("outcome")
            }
        ),
    }


def release_inventory(
    data_root: pathlib.Path, slate_date: str
) -> list[dict[str, Any]]:
    """Every release minted for this slate, canary or live.

    Found by directory name rather than by reading an index, because the
    question being asked is what is on disk.
    """
    found: list[dict[str, Any]] = []

    for root in sorted(data_root.glob("**/releases")):
        for release in sorted(root.glob(f"*__{slate_date}__*")):
            if not release.is_dir():
                continue

            rows = release / "rows.json"

            found.append(
                {
                    "files": sorted(
                        path.name
                        for path in release.iterdir()
                        if path.is_file()
                    ),
                    "release_id": release.name,
                    "root": str(root),
                    "rows_published": rows.is_file(),
                }
            )

    return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory the evidence the production host holds for one or "
            "more slate dates. Read-only."
        )
    )

    parser.add_argument("--data-root", required=True, type=pathlib.Path)

    parser.add_argument(
        "--slate-date",
        action="append",
        dest="slate_dates",
        required=True,
        help="Repeatable.",
    )

    parser.add_argument("--receipt-path", type=pathlib.Path)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "slates": [
            inventory(data_root=args.data_root, slate_date=slate_date)
            for slate_date in args.slate_dates
        ],
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"

    if args.receipt_path is not None:
        args.receipt_path.parent.mkdir(parents=True, exist_ok=True)

        args.receipt_path.write_text(text, encoding="utf-8")

    print(text, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
