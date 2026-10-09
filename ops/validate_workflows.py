#!/usr/bin/env python3
"""Static validation of the GitHub workflow definitions.

Catches the mistakes that are expensive to find by pushing and waiting: a
workflow that parses but schedules from the wrong branch, forgets concurrency
on a job that mutates production, grants more permission than it needs,
interpolates a secret into a place that ends up in a log, or puts pull-request
code on the self-hosted machine that holds production state.
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

# The production lifecycle mutates durable state that only exists on one
# machine, so it must land on that machine and nowhere else.
PRODUCTION_RUNNER_LABELS = ("self-hosted", "nba-production")

SELF_HOSTED_LABEL = "self-hosted"

# A pull request can carry arbitrary code from a contributor. Running it on
# the self-hosted runner would hand that code the production data root, the
# fit registry and the runner's credentials.
UNTRUSTED_TRIGGERS = ("pull_request", "pull_request_target")

# What a refresh exit code means for the lifecycle is decided in Python. The
# workflow must call the classifier and must gate the adaptive daily fit on
# the variable the classifier writes, so a verified PRESEASON_BLOCK is a safe
# no-op and every other nonzero refresh code still fails closed.
REFRESH_CLASSIFIER = "ops/classify_refresh_outcome.py"

ADAPTIVE_ENTRY_POINT = "ops/run_adaptive_daily_fit.py"

ADAPTIVE_GATE_VARIABLE = "RUN_ADAPTIVE"


def load(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a mapping")

    return payload


def triggers(workflow: dict) -> dict:
    # PyYAML resolves the bare key `on` to the boolean True.
    return workflow.get("on") or workflow.get(True) or {}


def runner_labels(job: dict) -> list[str]:
    """Every label a job requires, whatever form `runs-on` was written in."""
    runs_on = job.get("runs-on")

    if isinstance(runs_on, str):
        return [runs_on]

    if isinstance(runs_on, list):
        return [str(label) for label in runs_on]

    if isinstance(runs_on, dict):
        labels = runs_on.get("labels") or []

        if isinstance(labels, str):
            labels = [labels]

        return [str(label) for label in labels]

    return []


def untrusted_runner_problems(name: str, on: dict, jobs: dict) -> list[str]:
    if not any(trigger in on for trigger in UNTRUSTED_TRIGGERS):
        return []

    return [
        f"{name}: job {job_name} runs pull-request code on a self-hosted "
        "runner, which holds production state"
        for job_name, job in jobs.items()
        if SELF_HOSTED_LABEL in runner_labels(job)
    ]


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

    found += untrusted_runner_problems(name, on, jobs)

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

    found += refresh_classification_problems(name, jobs)

    for job_name, job in jobs.items():
        labels = runner_labels(job)

        absent = [
            label for label in PRODUCTION_RUNNER_LABELS if label not in labels
        ]

        if absent:
            found.append(
                f"{name}: job {job_name} does not require "
                f"{', '.join(absent)}; production state lives on the "
                "self-hosted runner and does not survive a hosted one"
            )

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


def refresh_classification_problems(name: str, jobs: dict) -> list[str]:
    """Require the Python-owned refresh classification, not a YAML guess.

    A raw refresh invocation fails the job on every nonzero exit, including the
    preseason no-op; a YAML conditional that special-cased exit 20 itself
    would put the rule in two places. So the classifier must be called and the
    adaptive daily fit must be gated on the variable it writes.
    """
    found: list[str] = []

    for job_name, job in jobs.items():
        steps = job.get("steps") or []

        commands = "\n".join(str(step.get("run", "")) for step in steps)

        adaptive_steps = [
            step
            for step in steps
            if ADAPTIVE_ENTRY_POINT in str(step.get("run", ""))
        ]

        if not adaptive_steps:
            continue

        if REFRESH_CLASSIFIER not in commands:
            found.append(
                f"{name}: job {job_name} runs the adaptive daily fit without "
                f"calling {REFRESH_CLASSIFIER}, so a preseason refresh would "
                "fail the run instead of being a safe no-op"
            )

        for step in adaptive_steps:
            if ADAPTIVE_GATE_VARIABLE not in str(step.get("if", "")):
                found.append(
                    f"{name}: job {job_name} does not gate the adaptive "
                    f"daily fit on env.{ADAPTIVE_GATE_VARIABLE}, so it could "
                    "run after a refresh that produced no new state"
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
