#!/usr/bin/env python3
"""Decide what one current-season refresh exit code means for the lifecycle.

GitHub orchestrates; Python decides. The refresh wrapper
(`ops/refresh_current_season_state.py`) returns the readiness preflight's own
exit code unchanged, and that code is the only machine contract either script
honours. This module is the single place that maps such a code onto the two
questions the workflow needs answered:

    * may the lifecycle report success?
    * may the adaptive daily fit run?

Exactly three answers exist:

    exit 0
        The rolling state advanced. The lifecycle continues normally and the
        adaptive daily protocol runs.

    exit 20, independently verified as preseason
        The slate date precedes the locked NBA opening day, so there is no
        2026-27 regular-season history to fetch and no newly completed
        information to fit. Nothing was written. This is a safe no-op: the
        lifecycle succeeds, the adaptive daily fit is skipped and the
        incumbent production fit is retained untouched.

    anything else
        Fail closed. The refresh exit code is returned unchanged so the job
        carries the real cause, including an exit 20 whose preseason claim
        could not be verified here.

Verification does not read the preflight's human-readable banner. It
re-derives the preseason condition — slate date strictly before opening day —
from the opening-day table the preflight itself owns, so this module cannot
disagree with it and cannot invent an opening day of its own.

Nothing here fetches data, fits a model, promotes a fit or contacts an
external service.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The preflight owns the locked opening day per season. It is read from there
# rather than restated here so the two can never drift apart.
PREFLIGHT_MODULE_PATH = ROOT / "ops" / "preflight_current_season_refresh.py"

OPENING_DAY_TABLE_NAME = "OPENING_DAY_BY_SEASON"

DEFAULT_SEASON = 2026

EXIT_OK = 0

# ops/preflight_current_season_refresh.py exits 20 for PRESEASON_BLOCK and
# ops/refresh_current_season_state.py propagates it unchanged.
REFRESH_EXIT_PRESEASON_BLOCK = 20

# Returned when the refresh reported success but this module was asked to
# classify an impossible pairing, so there is no refresh code to re-raise.
EXIT_UNCLASSIFIED = 69

OUTCOME_REFRESHED = "REFRESHED"
OUTCOME_PRESEASON_BLOCK = "PRESEASON_BLOCK"
OUTCOME_REFRESH_FAILED = "REFRESH_FAILED"

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


class PreseasonUnverifiable(Exception):
    """The preseason claim behind an exit 20 could not be confirmed."""


def load_opening_day_table() -> dict[int, str]:
    """Read the preflight's locked opening-day table.

    Imported rather than copied: a second table would be a second opening day,
    and this module is not allowed to have an opinion about the schedule.
    """
    spec = importlib.util.spec_from_file_location(
        "preflight_current_season_refresh_opening_days",
        PREFLIGHT_MODULE_PATH,
    )

    if spec is None or spec.loader is None:
        raise PreseasonUnverifiable(
            f"cannot load the opening-day table from {PREFLIGHT_MODULE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise PreseasonUnverifiable(
            f"cannot load the opening-day table from "
            f"{PREFLIGHT_MODULE_PATH}: {type(error).__name__}: {error}"
        ) from error

    table = getattr(module, OPENING_DAY_TABLE_NAME, None)

    if not isinstance(table, dict) or not table:
        raise PreseasonUnverifiable(
            f"{PREFLIGHT_MODULE_PATH.name} declares no "
            f"{OPENING_DAY_TABLE_NAME}"
        )

    return {int(season): str(day) for season, day in table.items()}


def exact_date(value: str, label: str) -> date:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as error:
        raise PreseasonUnverifiable(
            f"{label} must be exact YYYY-MM-DD; got {value!r} ({error})"
        ) from error

    return parsed


def opening_day_for(
    season: int,
    override: str | None = None,
    table: dict[int, str] | None = None,
) -> date:
    if override:
        return exact_date(override, "opening day")

    resolved = load_opening_day_table() if table is None else table

    if season not in resolved:
        raise PreseasonUnverifiable(
            f"no locked opening day is configured for season {season}"
        )

    return exact_date(resolved[season], "opening day")


def verify_preseason(
    *,
    season: int,
    slate_date: str,
    opening_day: str | None = None,
    table: dict[int, str] | None = None,
) -> tuple[date, date]:
    """Confirm the slate date precedes the locked opening day.

    Raises rather than returning False: an exit 20 this module cannot explain
    must fail closed, and a bare False invites a caller to treat "unverified"
    and "not preseason" as the same thing.
    """
    slate = exact_date(slate_date, "slate date")
    opening = opening_day_for(season, opening_day, table)

    if slate >= opening:
        raise PreseasonUnverifiable(
            f"the refresh reported PRESEASON_BLOCK for slate date "
            f"{slate.isoformat()}, but that is not before the locked "
            f"season-{season} opening day {opening.isoformat()}"
        )

    return slate, opening


def classify(
    *,
    refresh_exit_code: int,
    season: int = DEFAULT_SEASON,
    slate_date: str = "",
    opening_day: str | None = None,
    table: dict[int, str] | None = None,
) -> dict:
    """Map one refresh exit code onto the lifecycle decision."""
    code = int(refresh_exit_code)

    decision: dict = {
        "lifecycle_status": STATUS_FAILED,
        "opening_day": None,
        "outcome": OUTCOME_REFRESH_FAILED,
        "refresh_exit_code": code,
        "run_adaptive_fit": False,
        "season": int(season),
        "slate_date": slate_date or None,
        "verified": False,
        "writes_performed": None,
    }

    if code == EXIT_OK:
        decision.update(
            {
                "detail": (
                    "the current-season rolling state was refreshed and "
                    "committed; the normal lifecycle continues"
                ),
                "lifecycle_status": STATUS_SUCCESS,
                "outcome": OUTCOME_REFRESHED,
                "run_adaptive_fit": True,
                "verified": True,
                "writes_performed": True,
            }
        )

        return decision

    if code == REFRESH_EXIT_PRESEASON_BLOCK:
        try:
            slate, opening = verify_preseason(
                season=season,
                slate_date=slate_date,
                opening_day=opening_day,
                table=table,
            )

        except PreseasonUnverifiable as error:
            decision["detail"] = (
                f"refresh exit {code} claims PRESEASON_BLOCK but it could "
                f"not be verified here, so the lifecycle fails closed: "
                f"{error}"
            )

            return decision

        decision.update(
            {
                "detail": (
                    f"{slate.isoformat()} is before the locked season-"
                    f"{season} opening day {opening.isoformat()}, so no "
                    "2026-27 regular-season history exists to refresh. "
                    "Nothing was written, the adaptive daily fit is skipped "
                    "and the incumbent production fit is retained."
                ),
                "lifecycle_status": STATUS_SUCCESS,
                "opening_day": opening.isoformat(),
                "outcome": OUTCOME_PRESEASON_BLOCK,
                "run_adaptive_fit": False,
                "slate_date": slate.isoformat(),
                "verified": True,
                "writes_performed": False,
            }
        )

        return decision

    decision["detail"] = (
        f"the current-season refresh exited {code}, which is not a verified "
        "safe no-op; the lifecycle fails closed and the incumbent production "
        "fit is retained"
    )

    return decision


def render(decision: dict) -> str:
    rows = [
        ("refresh exit code", str(decision["refresh_exit_code"])),
        ("refresh outcome", decision["outcome"]),
        ("slate date", decision["slate_date"] or "n/a"),
        ("opening day", decision["opening_day"] or "n/a"),
        (
            "adaptive daily fit",
            "runs" if decision["run_adaptive_fit"] else "skipped",
        ),
        (
            "data writes",
            {True: "committed", False: "none", None: "unknown"}[
                decision["writes_performed"]
            ],
        ),
        ("lifecycle", decision["lifecycle_status"]),
    ]

    lines = [
        "## Current-season refresh outcome",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    lines += [f"| {label} | {value} |" for label, value in rows]

    lines += ["", decision["detail"]]

    return "\n".join(lines) + "\n"


def github_env_lines(decision: dict) -> str:
    """The two facts the workflow needs, decided here and only here."""
    return (
        f"REFRESH_OUTCOME={decision['outcome']}\n"
        f"RUN_ADAPTIVE={'true' if decision['run_adaptive_fit'] else 'false'}\n"
    )


def exit_code_for(decision: dict) -> int:
    if decision["lifecycle_status"] == STATUS_SUCCESS:
        return EXIT_OK

    code = int(decision["refresh_exit_code"])

    # A failed classification must never return success, even if it was
    # somehow reached with a zero refresh code.
    return code if code != EXIT_OK else EXIT_UNCLASSIFIED


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="classify_refresh_outcome",
        description=(
            "Classify one current-season refresh exit code into the "
            "production lifecycle decision. Reports a verified "
            "PRESEASON_BLOCK as a safe no-op and fails closed otherwise."
        ),
    )

    parser.add_argument(
        "--refresh-exit-code",
        type=int,
        required=True,
        help="Exit code returned by ops/refresh_current_season_state.py.",
    )

    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)

    parser.add_argument(
        "--slate-date",
        default="",
        help="Slate date YYYY-MM-DD the refresh was invoked with.",
    )

    parser.add_argument(
        "--opening-day",
        default=None,
        help=(
            "Opening-day override, for parity with the refresh wrapper. "
            "Defaults to the preflight's locked table."
        ),
    )

    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)

    parser.add_argument(
        "--github-env",
        type=Path,
        default=None,
        help="Append REFRESH_OUTCOME and RUN_ADAPTIVE to this file.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    decision = classify(
        refresh_exit_code=args.refresh_exit_code,
        season=args.season,
        slate_date=args.slate_date,
        opening_day=args.opening_day,
    )

    decision["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    rendered = render(decision)

    print(rendered)

    if args.status_path:
        args.status_path.parent.mkdir(parents=True, exist_ok=True)

        args.status_path.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.github_env:
        with args.github_env.open("a", encoding="utf-8") as handle:
            handle.write(github_env_lines(decision))

    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(rendered)

    code = exit_code_for(decision)

    if code != EXIT_OK:
        print(decision["detail"], file=sys.stderr)

    return code


if __name__ == "__main__":
    raise SystemExit(main())
