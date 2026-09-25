#!/usr/bin/env python3
"""Static validation of the GitHub workflow definitions.

Catches the mistakes that are expensive to find by pushing and waiting: a
workflow that parses but schedules from the wrong branch, forgets concurrency
on a job that mutates production, grants more permission than it needs, or
interpolates a secret into a place that ends up in a log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

WORKFLOW_DIR = ROOT / ".github" / "workflows"

PRODUCTION_WORKFLOW = "nba_production_lifecycle.yml"

CI_WORKFLOW = "ci.yml"

AUTHORITATIVE_PRODUCTION_BRANCH = "production/wizardofodds-integration"


def load(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a mapping")

    return payload


def triggers(workflow: dict) -> dict:
    # PyYAML resolves the bare key `on` to the boolean True.
    return workflow.get("on") or workflow.get(True) or {}


def problems_for(name: str, workflow: dict) -> list[str]:
    found: list[str] = []

    on = triggers(workflow)

    if not on:
        found.append(f"{name}: no triggers")

    jobs = workflow.get("jobs") or {}

    if not jobs:
        found.append(f"{name}: no jobs")

    if "permissions" not in workflow:
        found.append(f"{name}: no explicit permissions block")

    elif workflow["permissions"] not in ({"contents": "read"},):
        # Anything broader than read is allowed only deliberately; say so.
        found.append(
            f"{name}: permissions are broader than contents:read "
            f"({workflow['permissions']!r})"
        )

    if name == PRODUCTION_WORKFLOW:
        found += production_problems(name, on, workflow, jobs)

    if name == CI_WORKFLOW:
        if "pull_request" not in on:
            found.append(f"{name}: does not report on pull requests")

    return found


def production_problems(
    name: str, on: dict, workflow: dict, jobs: dict
) -> list[str]:
    found: list[str] = []

    if "schedule" not in on:
        found.append(f"{name}: no schedule trigger")

    else:
        for entry in on["schedule"]:
            cron = str(entry.get("cron", ""))

            minute = cron.split()[0] if cron.split() else ""

            if minute in ("0", "00", "*"):
                found.append(
                    f"{name}: schedule {cron!r} runs at the top of the hour, "
                    "where GitHub's scheduler is most congested"
                )

    if "workflow_dispatch" not in on:
        found.append(f"{name}: no workflow_dispatch trigger")

    concurrency = workflow.get("concurrency")

    if not concurrency:
        found.append(
            f"{name}: no concurrency group, so production runs could overlap"
        )

    elif concurrency.get("cancel-in-progress") is True:
        found.append(
            f"{name}: cancel-in-progress would interrupt a production "
            "lifecycle mid-write"
        )

    text = (WORKFLOW_DIR / name).read_text(encoding="utf-8")

    if AUTHORITATIVE_PRODUCTION_BRANCH not in text:
        found.append(
            f"{name}: never names the authoritative production branch, so a "
            "scheduled run would use the default branch"
        )

    for job_name, job in jobs.items():
        steps = job.get("steps") or []

        checkout = [
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/checkout")
        ]

        if not checkout:
            found.append(f"{name}: job {job_name} has no checkout step")

        for step in checkout:
            if not (step.get("with") or {}).get("ref"):
                found.append(
                    f"{name}: job {job_name} checks out without an explicit "
                    "ref; a scheduled run would take the default branch"
                )

    return found


def secret_leak_problems(name: str) -> list[str]:
    """Refuse a secret interpolated anywhere that reaches a log."""
    found: list[str] = []

    for number, line in enumerate(
        (WORKFLOW_DIR / name).read_text(encoding="utf-8").splitlines(), 1
    ):
        if "secrets." not in line:
            continue

        stripped = line.strip()

        # A secret may only be bound to an environment variable.
        if stripped.startswith("#"):
            continue

        if "echo" in stripped or "printf" in stripped:
            found.append(
                f"{name}:{number}: a secret is interpolated into a command "
                "that writes to the log"
            )

    return found


def validate() -> list[str]:
    if not WORKFLOW_DIR.is_dir():
        return [f"missing {WORKFLOW_DIR.relative_to(ROOT)}"]

    found: list[str] = []

    names = sorted(
        path.name
        for path in WORKFLOW_DIR.iterdir()
        if path.suffix in (".yml", ".yaml")
    )

    if PRODUCTION_WORKFLOW not in names:
        found.append(f"missing {PRODUCTION_WORKFLOW}")

    if CI_WORKFLOW not in names:
        found.append(f"missing {CI_WORKFLOW}")

    for name in names:
        path = WORKFLOW_DIR / name

        try:
            workflow = load(path)
        except Exception as error:
            found.append(f"{name}: {type(error).__name__}: {error}")
            continue

        found += problems_for(name, workflow)
        found += secret_leak_problems(name)

    return found


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        prog="validate_workflows",
        description="Statically validate the GitHub workflow definitions.",
    ).parse_args(argv)

    found = validate()

    if found:
        print("workflow validation FAILED:")

        for problem in found:
            print(f"  - {problem}")

        return 1

    print("workflow validation passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
