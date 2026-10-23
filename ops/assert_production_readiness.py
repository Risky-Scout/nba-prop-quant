#!/usr/bin/env python3
"""Prove the durable readiness receipt makes the T-20 controller eligible.

A refresh step exiting 0 proves the step ran. It does not prove the thing the
evening depends on, which is that ``run_t20_production_controller.py`` will now
accept the receipt for this slate under this production SHA.

So this asks the controller. It calls the controller's own
:func:`morning_readiness` -- the exact function that gates serving, not a
re-implementation of it -- and reports what that function decided. If the gate
would refuse, this refuses, with the gate's own words.

That is deliberately circular in one direction and not in the other: it cannot
tell the controller to accept anything, it can only find out whether the
controller does. A test pins that this module does no readiness checking of
its own.
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
    morning_readiness,
)

OUTCOME_ELIGIBLE = "T20_CONTROLLER_ELIGIBLE"
OUTCOME_INELIGIBLE = "T20_CONTROLLER_INELIGIBLE"

EXIT_OK = 0
EXIT_INELIGIBLE = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assert_production_readiness",
        description=(
            "Report whether the T-20 controller's readiness gate accepts the "
            "durable receipt for this slate under this production SHA."
        ),
    )

    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--production-sha", required=True)
    parser.add_argument("--status-path", type=pathlib.Path, default=None)
    parser.add_argument("--summary-path", type=pathlib.Path, default=None)

    return parser.parse_args(argv)


def _write(path: pathlib.Path | None, text: str, *, append: bool = False) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        readiness = morning_readiness(
            data_root=args.data_root,
            slate_date=args.slate_date,
            production_sha=args.production_sha,
        )

    except (ControllerRefused, OSError) as error:
        status = {
            "generated_at": datetime.now(UTC).isoformat(),
            "outcome": OUTCOME_INELIGIBLE,
            "production_code_sha": args.production_sha,
            "reason": str(error),
            "refused_with": getattr(error, "outcome", type(error).__name__),
            "slate_date": args.slate_date,
        }

        _write(
            args.status_path,
            json.dumps(status, indent=2, sort_keys=True) + "\n",
        )

        _write(
            args.summary_path,
            f"## T-20 controller eligibility\n\n{OUTCOME_INELIGIBLE}: "
            f"{error}\n",
            append=True,
        )

        print(json.dumps(status, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_INELIGIBLE

    status = {
        "generated_at": datetime.now(UTC).isoformat(),
        "outcome": OUTCOME_ELIGIBLE,
        "production_code_sha": args.production_sha,
        "readiness": readiness,
        "reason": (
            "the controller's own readiness gate accepts the receipt for "
            f"{args.slate_date} under production SHA {args.production_sha}"
        ),
        "slate_date": args.slate_date,
    }

    _write(args.status_path, json.dumps(status, indent=2, sort_keys=True) + "\n")

    _write(
        args.summary_path,
        f"## T-20 controller eligibility\n\n{OUTCOME_ELIGIBLE}: readiness "
        f"receipt for {args.slate_date} was produced by "
        f"{readiness['production_code_sha']}, which is the code about to "
        f"serve. morning outcome {readiness['morning_outcome']}, refresh "
        f"outcome {readiness['refresh_outcome']}.\n",
        append=True,
    )

    print(json.dumps(status, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
