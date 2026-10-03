#!/usr/bin/env python3
"""Turn one production lifecycle run into a readable summary and a status file.

The summary answers, in one screen, what the run decided and whether
production changed. A no-op is reported as a successful outcome, because a day
with no newly completed NBA information genuinely requires no new model. That
covers both kinds: an in-season day whose completed information has not
changed, and a preseason day that has no regular-season history at all.

No DataFrame is rendered here and no secret is read.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


OUTCOME_LABELS = {
    "COMPLETED": "candidate fitted and registered",
    "NO_NEW_TRAINING_DATA": "no new completed NBA information; incumbent retained",
    "ALREADY_RUNNING": "another production lifecycle holds the lock",
    "DRY_RUN_OK": "plan verified; nothing fitted",
}

# Refresh outcomes are classified by ops/classify_refresh_outcome.py. They are
# reported here, not re-decided here.
REFRESH_OUTCOME_LABELS = {
    "REFRESHED": "rolling current-season state advanced",
    "PRESEASON_BLOCK": "before NBA opening day; no refresh was due",
    "REFRESH_FAILED": "refresh failed closed; nothing was published",
}

PRESEASON_BLOCK = "PRESEASON_BLOCK"

# What the adaptive stage means when the refresh was a verified safe no-op.
PRESEASON_ADAPTIVE_DETAIL = (
    "skipped: preseason no-op; incumbent fit retained"
)


def read_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return None

    try:
        return json.loads(text)
    except ValueError:
        # The adaptive CLI writes JSON on stdout; a tee'd log may hold other
        # lines around it. Take the outermost object rather than fail the
        # summary over formatting.
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end <= start:
            return None

        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            return None


def field(payload: dict | None, *path, default=None):
    node = payload

    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default

        node = node[key]

    return node


def build_status(
    preflight: dict | None,
    adaptive: dict | None,
    production_sha: str,
    trigger: str,
    mode: str,
    refresh: dict | None = None,
) -> dict:
    outcome = field(adaptive, "outcome")

    preflight_ok = field(preflight, "status") == "ok"

    blockers = field(preflight, "blockers", default=[]) or []

    refresh_outcome = field(refresh, "outcome")

    refresh_ok = refresh is None or (
        field(refresh, "lifecycle_status") == "success"
    )

    # A verified PRESEASON_BLOCK means the adaptive daily fit was deliberately
    # not run, so its absence is the intended outcome rather than a failure.
    preseason_no_op = refresh_ok and refresh_outcome == PRESEASON_BLOCK

    if not preflight_ok:
        failure_stage = "preflight"
    elif not refresh_ok:
        failure_stage = "refresh"
    elif adaptive is None and mode == "production" and not preseason_no_op:
        failure_stage = "adaptive_protocol"
    else:
        failure_stage = None

    promoted = bool(field(adaptive, "promoted", default=False))

    adaptive_detail = OUTCOME_LABELS.get(outcome, outcome)

    if adaptive is None and preseason_no_op:
        adaptive_detail = PRESEASON_ADAPTIVE_DETAIL

    return {
        "adaptive_action": outcome,
        "adaptive_action_detail": adaptive_detail,
        "blockers": blockers,
        "candidate_fit_id": field(adaptive, "fit_id"),
        "current_good_fit_id_after": field(
            adaptive, "promotion_state", "current_good_fit_id"
        ),
        "failure_stage": failure_stage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "preflight_status": field(preflight, "status", default="unknown"),
        "production_code_sha": production_sha,
        "promoted": promoted,
        "refresh_outcome": refresh_outcome,
        "refresh_outcome_detail": REFRESH_OUTCOME_LABELS.get(
            refresh_outcome, field(refresh, "detail", default=refresh_outcome)
        ),
        # In a preseason no-op there is no adaptive plan, so the slate date
        # the refresh was invoked with is the only one the run has.
        "slate_date": field(adaptive, "plan", "slate_date")
        or field(refresh, "slate_date"),
        "training_cutoff": field(adaptive, "plan", "training_cutoff"),
        "trigger": trigger,
        "validation_checks_recorded": len(
            field(adaptive, "validation_checks", default={}) or {}
        ),
        "validation_checks_deferred": field(
            adaptive, "deferred_validation_checks", default=[]
        ),
        "runtime_seconds": field(adaptive, "benchmark", "total_seconds"),
        "training_seasons": field(
            adaptive, "benchmark", "details", "training_seasons"
        ),
    }


def render(status: dict) -> str:
    rows = [
        ("production code SHA", status["production_code_sha"] or "unknown"),
        ("trigger", status["trigger"]),
        ("mode", status["mode"]),
        ("slate date", status["slate_date"] or "n/a"),
        ("training cutoff", status["training_cutoff"] or "n/a"),
        ("preflight", status["preflight_status"]),
        (
            "refresh outcome",
            status.get("refresh_outcome") or "not reported",
        ),
        ("adaptive action", status["adaptive_action_detail"] or "not reached"),
        ("candidate fit id", status["candidate_fit_id"] or "none created"),
        ("promoted", "yes" if status["promoted"] else "no"),
        (
            "current good fit id",
            status["current_good_fit_id_after"] or "unchanged",
        ),
        (
            "validation checks recorded",
            str(status["validation_checks_recorded"]),
        ),
        ("runtime seconds", str(status["runtime_seconds"] or "n/a")),
    ]

    lines = ["## NBA production lifecycle run", "", "| field | value |", "| --- | --- |"]

    lines += [f"| {label} | {value} |" for label, value in rows]

    if status["training_seasons"]:
        seasons = status["training_seasons"]

        lines += [
            "",
            f"Training corpus spanned {len(seasons)} seasons "
            f"({seasons[0]}–{seasons[-1]}).",
        ]

    if status["failure_stage"]:
        lines += ["", f"**Failed at stage: `{status['failure_stage']}`**"]

    if status["blockers"]:
        lines += ["", "### Blockers", ""]
        lines += [f"- `{blocker}`" for blocker in status["blockers"]]

    if status["adaptive_action"] == "NO_NEW_TRAINING_DATA":
        lines += [
            "",
            "This is a successful outcome. No newly completed NBA "
            "information was available, so the incumbent production fit "
            "remains current and no model version was manufactured.",
        ]

    if status.get("refresh_outcome") == PRESEASON_BLOCK:
        lines += [
            "",
            "This is a successful outcome. The slate date precedes NBA "
            "opening day, so there was no regular-season history to refresh: "
            "no data was written, the adaptive daily fit was skipped, and "
            "the incumbent production fit is retained unchanged.",
        ]

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="summarise_production_run",
        description="Summarise one automated production lifecycle run.",
    )

    parser.add_argument("--preflight", type=Path, default=None)
    parser.add_argument("--refresh", type=Path, default=None)
    parser.add_argument("--adaptive", type=Path, default=None)
    parser.add_argument("--production-sha", default="")
    parser.add_argument("--trigger", default="unknown")
    parser.add_argument("--mode", default="unknown")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--status-path", type=Path, default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    status = build_status(
        read_json(args.preflight),
        read_json(args.adaptive),
        args.production_sha,
        args.trigger,
        args.mode,
        refresh=read_json(args.refresh),
    )

    rendered = render(status)

    print(rendered)

    if args.status_path:
        args.status_path.parent.mkdir(parents=True, exist_ok=True)

        args.status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(rendered)

    # The summary always succeeds; the job's failure is decided by the stage
    # that failed, not by the reporter.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
