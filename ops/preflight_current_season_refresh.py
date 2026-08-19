from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nba_prop_quant.api import BDLClient
from nba_prop_quant.normalize import normalize_advanced, normalize_stats
from nba_prop_quant.settings import get_settings


ROOT = Path.cwd()
EASTERN = ZoneInfo("America/New_York")
OPENING_DAY_BY_SEASON = {
    2026: "2026-10-20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only BALLDONTLIE availability preflight before the "
            "strict current-season history refresh. This script never "
            "writes raw data, markers, processed data, models, or "
            "external-test artifacts."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--target-date",
        default=None,
        help=(
            "Slate date YYYY-MM-DD. Defaults to today's date in "
            "America/New_York."
        ),
    )

    parser.add_argument(
        "--opening-day",
        default=None,
        help=(
            "Override opening day YYYY-MM-DD. "
            "Season 2026 defaults to 2026-10-20."
        ),
    )

    parser.add_argument(
        "--require-advanced",
        action="store_true",
        help=(
            "Require advanced-stat coverage for every game on the "
            "latest completed post-opening-day date."
        ),
    )

    return parser.parse_args()


def exact_date(value: str, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise SystemExit(
            f"ERROR: invalid {label} {value!r}: {exc}"
        )

    if parsed.strftime("%Y-%m-%d") != value:
        raise SystemExit(
            f"ERROR: {label} must be exact YYYY-MM-DD; got {value!r}"
        )

    return parsed.normalize()


def row_game_date(row: dict) -> pd.Timestamp | None:
    value = row.get("date")

    if value is None:
        game = row.get("game") or {}
        value = game.get("date")

    if value is None:
        return None

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return pd.Timestamp(parsed).normalize()


def is_final_game(row: dict) -> bool:
    postponed_raw = row.get(
        "postponed",
        False,
    )

    postponed = (
        bool(postponed_raw)
        if pd.notna(postponed_raw)
        else False
    )

    if postponed:
        return False

    status = str(
        row.get(
            "status",
            "",
        )
    ).strip().lower()

    state = str(
        row.get(
            "status_state",
            "",
        )
    ).strip().lower()

    if "final" in status:
        return True

    if state in {
        "post",
        "final",
        "completed",
        "complete",
    }:
        return True

    return False


def finish(
    *,
    action: str,
    code: int,
    message: str,
    payload: dict,
) -> None:
    print("=" * 118)
    print("CURRENT-SEASON HISTORY REFRESH PREFLIGHT — READ ONLY")
    print("=" * 118)
    print(f"Action: {action}")
    print(message)
    print()
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    print()
    print("Writes performed: NONE")

    raise SystemExit(code)


def main() -> None:
    args = parse_args()

    target_text = (
        args.target_date
        or datetime.now(EASTERN).strftime(
            "%Y-%m-%d"
        )
    )

    target = exact_date(
        target_text,
        "target date",
    )

    opening_text = (
        args.opening_day
        or OPENING_DAY_BY_SEASON.get(
            args.season
        )
    )

    if opening_text is None:
        raise SystemExit(
            "ERROR: no locked opening day is configured for this season. "
            "Pass --opening-day YYYY-MM-DD."
        )

    opening = exact_date(
        opening_text,
        "opening day",
    )

    common = {
        "season": int(args.season),
        "target_date": target_text,
        "opening_day": opening_text,
        "require_advanced": bool(
            args.require_advanced
        ),
    }

    if target < opening:
        finish(
            action="PRESEASON_BLOCK",
            code=20,
            message=(
                f"{target_text} is before opening day {opening_text}. "
                "Current-season regular-season player stats are not expected "
                "yet. Do NOT run the strict season-2026 refresh."
            ),
            payload=common,
        )

    if target == opening:
        finish(
            action="SKIP_REFRESH",
            code=10,
            message=(
                "Opening-day exception: no completed 2026-27 "
                "regular-season games belong in the historical information "
                "set before the first official slate capture."
            ),
            payload=common,
        )

    previous_day = (
        target
        - pd.Timedelta(
            days=1
        )
    ).normalize()

    settings = get_settings()

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=settings.bdl_requests_per_minute,
    ) as client:
        # READ ONLY. Restrict to dates on/after regular-season opening day,
        # which excludes preseason games from this readiness decision.
        game_rows = list(
            client.games(
                start_date=opening_text,
                end_date=previous_day.strftime(
                    "%Y-%m-%d"
                ),
            )
        )

        final_rows = []

        for row in game_rows:
            date = row_game_date(
                row
            )

            if date is None:
                continue

            if not (
                opening
                <= date
                < target
            ):
                continue

            if is_final_game(row):
                final_rows.append(
                    row
                )

        if not final_rows:
            finish(
                action="SKIP_REFRESH",
                code=10,
                message=(
                    "BDL reports no final post-opening-day games strictly "
                    "before this slate date. There is no newly completed "
                    "regular-season history to require."
                ),
                payload={
                    **common,
                    "games_returned": len(
                        game_rows
                    ),
                    "final_games": 0,
                    "query_end_date": previous_day.strftime(
                        "%Y-%m-%d"
                    ),
                },
            )

        latest_final_date = max(
            date
            for date in (
                row_game_date(
                    row
                )
                for row in final_rows
            )
            if date is not None
        )

        latest_final_rows = [
            row
            for row in final_rows
            if row_game_date(
                row
            )
            == latest_final_date
        ]

        latest_game_ids = sorted(
            {
                int(
                    row["id"]
                )
                for row in latest_final_rows
                if row.get(
                    "id"
                )
                is not None
            }
        )

        if not latest_game_ids:
            finish(
                action="HOLD",
                code=30,
                message=(
                    "Final games were found, but no usable game IDs were "
                    "resolved. Do not run the strict history refresh."
                ),
                payload={
                    **common,
                    "latest_final_date": str(
                        latest_final_date.date()
                    ),
                },
            )

        standard_rows = list(
            client.stats(
                game_ids=latest_game_ids,
                period=0,
            )
        )

        standard = normalize_stats(
            standard_rows
        )

        standard_game_ids = (
            set(
                pd.to_numeric(
                    standard[
                        "game_id"
                    ],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
            )
            if (
                not standard.empty
                and "game_id"
                in standard.columns
            )
            else set()
        )

        missing_standard = sorted(
            set(
                latest_game_ids
            )
            - standard_game_ids
        )

        if (
            not standard_rows
            or missing_standard
        ):
            finish(
                action="HOLD_STANDARD_NOT_READY",
                code=30,
                message=(
                    "Final games exist, but BDL standard player stats are "
                    "missing or partial for the latest completed date. "
                    "Wait and rerun this read-only preflight later."
                ),
                payload={
                    **common,
                    "latest_final_date": str(
                        latest_final_date.date()
                    ),
                    "latest_final_game_ids": latest_game_ids,
                    "standard_rows_returned": len(
                        standard
                    ),
                    "standard_game_ids_seen": sorted(
                        standard_game_ids
                    ),
                    "missing_standard_game_ids": missing_standard,
                },
            )

        advanced_rows = []
        advanced_game_ids: set[int] = set()

        if args.require_advanced:
            advanced_rows = list(
                client.advanced_stats(
                    game_ids=latest_game_ids,
                    period=0,
                )
            )

            advanced = normalize_advanced(
                advanced_rows
            )

            advanced_game_ids = (
                set(
                    pd.to_numeric(
                        advanced[
                            "game_id"
                        ],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                    .unique()
                )
                if (
                    not advanced.empty
                    and "game_id"
                    in advanced.columns
                )
                else set()
            )

            missing_advanced = sorted(
                set(
                    latest_game_ids
                )
                - advanced_game_ids
            )

            if (
                not advanced_rows
                or missing_advanced
            ):
                finish(
                    action="HOLD_ADVANCED_NOT_READY",
                    code=31,
                    message=(
                        "Standard stats are available, but BDL advanced "
                        "stats are missing or partial for the latest "
                        "completed date. Because the normal production "
                        "refresh requests advanced data, wait and rerun "
                        "this preflight later."
                    ),
                    payload={
                        **common,
                        "latest_final_date": str(
                            latest_final_date.date()
                        ),
                        "latest_final_game_ids": latest_game_ids,
                        "standard_rows_returned": len(
                            standard
                        ),
                        "advanced_rows_returned": len(
                            advanced
                        ),
                        "advanced_game_ids_seen": sorted(
                            advanced_game_ids
                        ),
                        "missing_advanced_game_ids": missing_advanced,
                    },
                )

        finish(
            action="READY_TO_REFRESH",
            code=0,
            message=(
                "BDL has complete standard"
                + (
                    " and advanced"
                    if args.require_advanced
                    else ""
                )
                + " player-stat coverage for every game on the latest "
                "completed post-opening-day date. It is appropriate to "
                "run the existing strict current-season refresh."
            ),
            payload={
                **common,
                "latest_final_date": str(
                    latest_final_date.date()
                ),
                "latest_final_game_ids": latest_game_ids,
                "standard_rows_returned": len(
                    standard
                ),
                "advanced_rows_returned": (
                    len(
                        advanced_rows
                    )
                    if args.require_advanced
                    else None
                ),
            },
        )


if __name__ == "__main__":
    main()
