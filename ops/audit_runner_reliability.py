#!/usr/bin/env python3
"""Decide whether the self-hosted production runner is actually reliable.

A runner that eventually answers is not a healthy runner. The audited evidence
is a scheduled run created at 09:49:20Z that started at 15:36:22Z -- five
hours forty-seven minutes later -- plus two runs that failed with "the
self-hosted runner lost communication with the server". Both shapes look like
a laptop asleep, and neither is visible in a pass/fail column: the five-hour
run succeeded.

So reliability is defined here as a measured property of a window of
consecutive scheduled runs, and the closure criterion is mechanical rather
than a judgement:

    * every run in the window started within the allowed wait of being created
    * no run in the window carries a lost-communication annotation
    * no run in the window failed because the runner was offline

Below that, the state is not PASS. It is either observation still pending,
when there have not yet been enough consecutive runs, or a named failure when
the window contains one. Nothing here can make a runner healthy; it can only
decline to call one healthy.

The input is the GitHub API's own answer, passed in as JSON so this is
auditable and so the decision does not depend on who ran it:

    gh api repos/<owner>/<repo>/actions/workflows/<file>/runs \\
      --jq '[.workflow_runs[] | select(.event == "schedule")]' > runs.json

    for each run id: gh api repos/<owner>/<repo>/actions/runs/<id>/jobs

Both shapes are accepted; see ``load_window``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

#: The brief's closure criterion. Named constants rather than literals in the
#: comparison, because these are the thresholds a reader has to be able to
#: check against the brief.
REQUIRED_CONSECUTIVE_RUNS = 7

MAXIMUM_QUEUE_WAIT = timedelta(minutes=5)

#: GitHub's own wording. Matched on a substring rather than parsed, because
#: this is the annotation text the runner service emits.
LOST_COMMUNICATION_PHRASE = "lost communication with the server"

#: What a run looks like when the runner was simply not there.
OFFLINE_PHRASES: tuple[str, ...] = (
    "No runner matched",
    "Waiting for a runner to pick up this job",
    "runner is offline",
)

STATUS_PASS = "PASS_PROVEN"
STATUS_PENDING = "IMPROVED_BUT_OBSERVATION_PENDING"
STATUS_FAIL = "FAIL"

EXIT_OK = 0
EXIT_NOT_PROVEN = 1


@dataclass
class Observation:
    """One scheduled run, reduced to the three facts that decide health."""

    run_id: int
    created_at: str
    started_at: str | None
    conclusion: str | None
    runner_name: str | None
    annotations: tuple[str, ...] = ()

    @property
    def wait(self) -> timedelta | None:
        if self.started_at is None:
            return None

        return _moment(self.started_at) - _moment(self.created_at)

    @property
    def started_promptly(self) -> bool:
        wait = self.wait

        return wait is not None and wait <= MAXIMUM_QUEUE_WAIT

    @property
    def lost_communication(self) -> bool:
        return any(
            LOST_COMMUNICATION_PHRASE in annotation
            for annotation in self.annotations
        )

    @property
    def runner_was_offline(self) -> bool:
        return any(
            phrase in annotation
            for annotation in self.annotations
            for phrase in OFFLINE_PHRASES
        )

    def payload(self) -> dict[str, Any]:
        wait = self.wait

        return {
            "conclusion": self.conclusion,
            "created_at": self.created_at,
            "lost_communication": self.lost_communication,
            "run_id": self.run_id,
            "runner_name": self.runner_name,
            "runner_was_offline": self.runner_was_offline,
            "started_at": self.started_at,
            "started_promptly": self.started_promptly,
            "wait_seconds": None if wait is None else round(wait.total_seconds()),
        }


def _moment(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def load_window(payload: Any) -> list[Observation]:
    """Read observations from whatever shape the caller collected.

    Accepts a bare list, a ``{"runs": [...]}`` wrapper, or GitHub's
    ``{"workflow_runs": [...]}``. Each entry may carry its job-level
    ``started_at``, ``runner_name`` and ``annotations``; the run-level
    ``run_started_at`` is used only as a fallback, and is noted as weaker
    evidence because it is not the moment the job reached the runner.
    """
    if isinstance(payload, dict):
        entries = payload.get("runs") or payload.get("workflow_runs") or []

    else:
        entries = payload or []

    observations: list[Observation] = []

    for entry in entries:
        annotations = entry.get("annotations") or []

        observations.append(
            Observation(
                run_id=int(entry.get("run_id") or entry.get("id") or 0),
                created_at=str(entry["created_at"]),
                started_at=(
                    entry.get("started_at") or entry.get("run_started_at")
                ),
                conclusion=entry.get("conclusion"),
                runner_name=entry.get("runner_name"),
                annotations=tuple(
                    annotation
                    if isinstance(annotation, str)
                    else str(annotation.get("message", ""))
                    for annotation in annotations
                ),
            )
        )

    # Newest first from the API; the window is the most recent runs.
    return sorted(observations, key=lambda item: _moment(item.created_at))


def assess(observations: list[Observation]) -> dict[str, Any]:
    """Classify the window. PASS requires evidence, not absence of evidence."""
    window = observations[-REQUIRED_CONSECUTIVE_RUNS:]

    slow = [item.run_id for item in window if not item.started_promptly]

    lost = [item.run_id for item in window if item.lost_communication]

    offline = [item.run_id for item in window if item.runner_was_offline]

    waits = [
        item.wait.total_seconds() for item in window if item.wait is not None
    ]

    if slow or lost or offline:
        status = STATUS_FAIL

        detail = (
            f"{len(slow)} of {len(window)} runs waited longer than "
            f"{int(MAXIMUM_QUEUE_WAIT.total_seconds() // 60)} minutes to "
            f"start, {len(lost)} lost communication with the server and "
            f"{len(offline)} found no runner. The runner is not reliable yet."
        )

    elif len(window) < REQUIRED_CONSECUTIVE_RUNS:
        status = STATUS_PENDING

        detail = (
            f"{len(window)} of the required {REQUIRED_CONSECUTIVE_RUNS} "
            "consecutive clean scheduled runs have been observed. Reliability "
            "is improved but not proven; a runner that has answered promptly "
            "a few times has not yet answered promptly seven times."
        )

    else:
        status = STATUS_PASS

        detail = (
            f"{REQUIRED_CONSECUTIVE_RUNS} consecutive scheduled runs each "
            "started within "
            f"{int(MAXIMUM_QUEUE_WAIT.total_seconds() // 60)} minutes, with "
            "no lost-communication annotation and no runner-offline failure."
        )

    return {
        "detail": detail,
        "longest_wait_seconds": round(max(waits)) if waits else None,
        "maximum_allowed_wait_seconds": int(
            MAXIMUM_QUEUE_WAIT.total_seconds()
        ),
        "observations": [item.payload() for item in window],
        "required_consecutive_runs": REQUIRED_CONSECUTIVE_RUNS,
        "runs_observed": len(window),
        "runs_that_lost_communication": lost,
        "runs_that_started_slowly": slow,
        "runs_with_no_runner": offline,
        "status": status,
    }


def render(assessment: dict[str, Any]) -> str:
    lines = [
        "## Self-hosted runner reliability",
        "",
        f"Status: **{assessment['status']}**",
        "",
        "| run | created | started | wait | prompt | lost comms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for item in assessment["observations"]:
        wait = item["wait_seconds"]

        lines.append(
            f"| `{item['run_id']}` | {item['created_at']} | "
            f"{item['started_at'] or 'never'} | "
            f"{'n/a' if wait is None else f'{wait}s'} | "
            f"{'yes' if item['started_promptly'] else 'NO'} | "
            f"{'YES' if item['lost_communication'] else 'no'} |"
        )

    lines += ["", assessment["detail"]]

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="audit_runner_reliability",
        description=(
            "Decide whether the self-hosted production runner meets the "
            "closure criterion. Cannot make a runner healthy."
        ),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="JSON collected from the GitHub Actions API",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    assessment = assess(
        load_window(
            json.loads(args.observations.read_text(encoding="utf-8"))
        )
    )

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)

        args.report_path.write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    text = render(assessment)

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    print(text, end="")

    return EXIT_OK if assessment["status"] == STATUS_PASS else EXIT_NOT_PROVEN


if __name__ == "__main__":
    raise SystemExit(main())
