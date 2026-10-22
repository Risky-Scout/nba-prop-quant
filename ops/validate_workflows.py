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

#: The scheduler for the one market snapshot production prices from. Separate
#: from the lifecycle because it runs on a different clock: the lifecycle is
#: the once-daily heavy path and this fires three bounded segments covering
#: the evening's T-20 capture windows.
T20_WORKFLOW = "nba_t20_production.yml"

CI_WORKFLOW = "ci.yml"

AUTHORITATIVE_PRODUCTION_BRANCH = "production/wizardofodds-integration"

PRODUCTION_ENVIRONMENT = "wizardofodds-production"

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

# The publication chain, in the only order that is safe on a served slate.
#
# Serving produces the priced artifact and the receipt that provenances it;
# the feed publishes those exact bytes; the page is built from the release the
# feed just published. Each step depends on the one before it, and the
# dependency that matters most is the last: a page built before its feed, or
# from a release the feed did not publish, renders and verifies and reports
# success while showing a slate from before.
SERVING_ENTRY_POINT = "ops/run_incumbent_production_serving.py"

PUBLICATION_ENTRY_POINT = "ops/run_wizardofodds_publication.py"

FEED_STAGE_FLAG = "--stage feed"

PAGE_STAGE_FLAG = "--stage page"

# The page stage is handed the feed stage's receipt. Named here because that
# argument is the dependency: without it the page would be built from whatever
# the destination happened to be serving.
PAGE_PIN_ARGUMENT = "--feed-receipt"

# Live publication is a separate reviewed decision taken through the committed
# switch and two environment variables. None of them belongs in a workflow
# file: a schedule that carried the approval would make every scheduled run an
# activation.
PUBLICATION_ACTIVATION_TOKENS = (
    "NBA_PROP_WOO_PUBLISH",
    "NBA_PROP_WOO_PUBLISH_APPROVAL",
    "--mode live",
    "--stage live",
)

# ----------------------------------------------------------------------
# the T-20 capture scheduler
# ----------------------------------------------------------------------

#: The three segments the T-20 workflow schedules, and the only ones. Pinned
#: here because the segment boundaries are a correctness property rather than
#: a preference: they are half-open and contiguous, so every T-20 time in the
#: supported window is owned by exactly one invocation. A fourth cron would
#: make two invocations fight over the same window; a missing one would leave
#: a span of the evening with no owner at all.
T20_SCHEDULE_CRONS: tuple[str, ...] = (
    "25 0 * * *",
    "25 14 * * *",
    "25 19 * * *",
)

T20_ENTRY_POINT = "ops/run_t20_production_controller.py"

#: Once-daily heavy and state-maintenance work. The T-20 workflow fires up to
#: three times a day and repeats none of it: the refresh, the adaptive
#: protocol, the frozen-package install, grading, the shadow and the shadow
#: monitor all belong to the lifecycle, and running any of them per tip group
#: is the cost this design exists to avoid.
T20_FORBIDDEN_ENTRY_POINTS: tuple[str, ...] = (
    "ops/refresh_current_season_state.py",
    "ops/run_adaptive_daily_fit.py",
    "ops/install_frozen_model_artifacts.py",
    "ops/grade_incumbent_production_slate.py",
    "ops/run_production_shadow.py",
    "ops/monitor_live_shadow_evidence.py",
)

# ----------------------------------------------------------------------
# what a workflow may install, and with what
# ----------------------------------------------------------------------

#: The version each official GitHub action must name, and the only one.
#:
#: These three generations are the Node 24 ones. The Node 20 runtime is
#: deprecated: a workflow still on it logs a deprecation warning on every run
#: and stops working on a date GitHub chooses rather than one this repository
#: does. Pinned to a single version rather than a floor because `uses:` has no
#: range syntax -- a workflow names exactly one ref, so "at least v7" is not a
#: thing a workflow file can say, and the only way to state the requirement is
#: to state the version.
#:
#: Executing a Node 24 action needs a runner that can: support landed in
#: actions/runner v2.327.0. The self-hosted production runner reports 2.337 or
#: newer, so it satisfies that; GitHub-hosted runners are always current.
OFFICIAL_ACTION_VERSIONS: dict[str, str] = {
    "actions/checkout": "v7",
    "actions/setup-python": "v7",
    "actions/upload-artifact": "v6",
}

#: A serving workflow installs what it serves with. The dev extra is pytest
#: and ruff, which serve nothing -- and a production environment carrying its
#: own test runner is one that can be made to run tests. Upgrading the
#: installer on every trigger is a download nothing asked for that changes
#: the installer under the run.
PRODUCTION_FORBIDDEN_INSTALL_TOKENS: tuple[str, ...] = (
    ".[dev]",
    "pip install --upgrade pip",
)

#: The version set production and CI are known to work under.
#:
#: pyproject.toml states floors, which is what the code needs and a different
#: question from what production is proven under. Without this file every
#: fresh environment re-resolved the whole tree against that morning's PyPI,
#: so a numpy, pandas, scipy or shap published hours earlier could land in a
#: run that serves predictions with no reviewed decision anywhere saying it
#: should.
PRODUCTION_CONSTRAINT_FILE = "constraints/py312-production.txt"

#: The workflows whose every ``pip install`` must go through that file.
#:
#: The two production workflows, because they serve, and CI, because a suite
#: that passes under versions production will not install is not a check on
#: production. ``default_branch_guard.yml`` is deliberately absent: it is the
#: one workflow that also runs from the default branch, which carries the
#: schedulers and their validators rather than the production dependency set,
#: so naming a production constraint file there would fail on the branch that
#: job exists to check.
CONSTRAINED_INSTALL_WORKFLOWS: tuple[str, ...] = (
    PRODUCTION_WORKFLOW,
    T20_WORKFLOW,
    CI_WORKFLOW,
)


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


def steps_of(jobs: dict) -> list[tuple[str, dict]]:
    """Every step in the workflow, paired with the job it belongs to."""
    return [
        (job_name, step)
        for job_name, job in jobs.items()
        for step in (job.get("steps") or [])
    ]


def official_action_problems(name: str, jobs: dict) -> list[str]:
    """Refuse an official action on a version other than the pinned one.

    Only the official actions, and only the ones named in
    :data:`OFFICIAL_ACTION_VERSIONS`: a third-party action's version is that
    project's own release cadence and is not this rule's business.
    """
    found: list[str] = []

    for job_name, step in steps_of(jobs):
        uses = str(step.get("uses", ""))

        if "@" not in uses:
            continue

        action, _, version = uses.partition("@")

        expected = OFFICIAL_ACTION_VERSIONS.get(action)

        if expected is None or version == expected:
            continue

        found.append(
            f"{name}: job {job_name} uses {uses}, and {action} is pinned to "
            f"{expected}; an older generation runs on the deprecated Node 20 "
            "runtime"
        )

    return found


def install_problems(name: str, jobs: dict) -> list[str]:
    """Refuse an install that is not reproducible, or not production-only.

    Two separate requirements that happen to live in the same commands. Every
    install must name the constraint file, so the environment a run builds is
    one a previous run proved rather than whatever was published that morning.
    And a production workflow must not install the dev extra or upgrade pip.
    """
    found: list[str] = []

    production = name in (PRODUCTION_WORKFLOW, T20_WORKFLOW)

    for job_name, step in steps_of(jobs):
        run = str(step.get("run", ""))

        if production:
            for token in PRODUCTION_FORBIDDEN_INSTALL_TOKENS:
                if token in run:
                    found.append(
                        f"{name}: job {job_name} runs {token!r}, which a "
                        "production serving workflow has no use for"
                    )

        if name not in CONSTRAINED_INSTALL_WORKFLOWS:
            continue

        for command in shell_commands(run):
            if "pip install" not in command:
                continue

            if PRODUCTION_CONSTRAINT_FILE in command:
                continue

            found.append(
                f"{name}: job {job_name} installs with {command.strip()!r}, "
                f"which does not go through {PRODUCTION_CONSTRAINT_FILE}, so "
                "the versions it resolves are whatever was published that day"
            )

    return found


def shell_commands(script: str) -> list[str]:
    """One entry per command in a ``run`` block, continuations joined.

    A long install is written over several lines with trailing backslashes,
    and reading those lines separately would see a ``pip install`` whose
    arguments are somewhere else.
    """
    commands: list[str] = []

    for line in script.splitlines():
        if commands and commands[-1].rstrip().endswith("\\"):
            commands[-1] = (
                commands[-1].rstrip().removesuffix("\\").rstrip()
                + " "
                + line.strip()
            )
        else:
            commands.append(line)

    return commands


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

    found += official_action_problems(name, jobs)

    found += install_problems(name, jobs)

    if name == PRODUCTION_WORKFLOW:
        found += production_problems(name, on, workflow, jobs)

    if name == T20_WORKFLOW:
        found += t20_problems(name, on, jobs)

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

    found += publication_order_problems(name, jobs)

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


def t20_problems(name: str, on: dict, jobs: dict) -> list[str]:
    """Require the T-20 scheduler to stay a scheduler.

    Three things are checked here rather than left to review. The segment
    schedule, because its contiguity is what makes every capture window owned
    exactly once. The runner, environment and checkout ref, because this job
    writes the snapshot tree and publishes, and a hosted runner or an
    inherited default-branch checkout would do that against the wrong state.
    And the absence of the lifecycle's once-daily work, because the whole
    point of a three-times-a-day scheduler is that it does not refresh, fit,
    recalibrate, grade, shadow or reinstall the frozen package per tip group.
    """
    found: list[str] = []

    scheduled = sorted(
        str(entry.get("cron", "")) for entry in (on.get("schedule") or [])
    )

    if scheduled != sorted(T20_SCHEDULE_CRONS):
        found.append(
            f"{name}: schedules {scheduled or 'nothing'} rather than the "
            f"contiguous segments {sorted(T20_SCHEDULE_CRONS)}, so a T-20 "
            "window would be owned by two invocations or by none"
        )

    if "workflow_dispatch" not in on:
        found.append(f"{name}: no workflow_dispatch trigger")

    text = (WORKFLOW_DIR / name).read_text(encoding="utf-8")

    if AUTHORITATIVE_PRODUCTION_BRANCH not in text:
        found.append(
            f"{name}: never names the authoritative production branch, so a "
            "scheduled run would use the default branch"
        )

    for token in PUBLICATION_ACTIVATION_TOKENS:
        for number, line in enumerate(text.splitlines(), 1):
            if token in line and not line.strip().startswith("#"):
                found.append(
                    f"{name}:{number}: names {token!r}, which is live "
                    "publication activation material and belongs to a "
                    "reviewed decision rather than to a schedule"
                )

    for job_name, job in jobs.items():
        labels = runner_labels(job)

        absent = [
            label for label in PRODUCTION_RUNNER_LABELS if label not in labels
        ]

        if absent:
            found.append(
                f"{name}: job {job_name} does not require "
                f"{', '.join(absent)}; the snapshot tree, the fit registry "
                "and the installed frozen bundle live on the self-hosted "
                "runner and do not survive a hosted one"
            )

        if job.get("environment") != PRODUCTION_ENVIRONMENT:
            found.append(
                f"{name}: job {job_name} does not declare environment "
                f"{PRODUCTION_ENVIRONMENT}, so its secrets and variables "
                "would not resolve"
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

        commands = "\n".join(str(step.get("run", "")) for step in steps)

        if T20_ENTRY_POINT not in commands:
            found.append(
                f"{name}: job {job_name} never runs {T20_ENTRY_POINT}, so "
                "nothing would capture the snapshot production prices from"
            )

        for entry_point in T20_FORBIDDEN_ENTRY_POINTS:
            if entry_point in commands:
                found.append(
                    f"{name}: job {job_name} runs {entry_point}, which is "
                    "the lifecycle's once-daily work and must not be "
                    "repeated per T-20 segment"
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


def publication_order_problems(name: str, jobs: dict) -> list[str]:
    """Require the publication chain in the order that cannot publish a lie.

    Checked here rather than left to review because step order in YAML is the
    only thing that sequences these, and the failure it prevents is silent: a
    page built before or beside its feed publishes successfully and shows the
    wrong slate.

    A workflow that publishes nothing is not held to any of this. Only one
    does, and it is the one that serves.
    """
    found: list[str] = []

    for job_name, job in jobs.items():
        steps = job.get("steps") or []

        commands = [str(step.get("run", "")) for step in steps]

        if not any(PUBLICATION_ENTRY_POINT in command for command in commands):
            continue

        def index_of(fragment: str) -> int | None:
            for position, command in enumerate(commands):
                if fragment in command:
                    return position

            return None

        serving = index_of(SERVING_ENTRY_POINT)
        feed = index_of(FEED_STAGE_FLAG)
        page = index_of(PAGE_STAGE_FLAG)

        if serving is None:
            found.append(
                f"{name}: job {job_name} publishes without running "
                f"{SERVING_ENTRY_POINT}, so it would publish an artifact this "
                "run did not produce"
            )

        for label, position in (("feed", feed), ("page", page)):
            if position is None:
                found.append(
                    f"{name}: job {job_name} runs {PUBLICATION_ENTRY_POINT} "
                    f"without its {label} stage, so the publication chain is "
                    "incomplete"
                )

        if serving is not None and feed is not None and feed < serving:
            found.append(
                f"{name}: job {job_name} publishes the feed before serving "
                "the slate, so it would publish the previous run's artifact"
            )

        if feed is not None and page is not None and page < feed:
            found.append(
                f"{name}: job {job_name} builds the page before publishing "
                "the feed, so the page would be built from whatever release "
                "the destination was already serving"
            )

        for position, step in enumerate(steps):
            run = str(step.get("run", ""))

            if PUBLICATION_ENTRY_POINT not in run:
                continue

            if step.get("continue-on-error") is True:
                found.append(
                    f"{name}: job {job_name} step {step.get('name')!r} may "
                    "fail without failing the job, so a later publication "
                    "stage could claim success after an earlier one broke"
                )

            if PAGE_STAGE_FLAG in run and PAGE_PIN_ARGUMENT not in run:
                found.append(
                    f"{name}: job {job_name} builds the page without "
                    f"{PAGE_PIN_ARGUMENT}, so it is not pinned to the release "
                    "this run published"
                )

    text = (WORKFLOW_DIR / name).read_text(encoding="utf-8")

    for token in PUBLICATION_ACTIVATION_TOKENS:
        for number, line in enumerate(text.splitlines(), 1):
            if token in line and not line.strip().startswith("#"):
                found.append(
                    f"{name}:{number}: names {token!r}, which is live "
                    "publication activation material and belongs to a "
                    "reviewed decision rather than to a schedule"
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

    if T20_WORKFLOW not in names:
        found.append(f"missing {T20_WORKFLOW}")

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
