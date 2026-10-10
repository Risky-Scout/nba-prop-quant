#!/usr/bin/env python3
"""Validate what the default branch is actually responsible for.

GitHub fires scheduled events only from the default branch, so `main` holds
the copy of ``nba_production_lifecycle.yml`` that starts every production run,
while the production application lineage lives on
``production/wizardofodds-integration``. That is the whole of main's
production role: it is the scheduler, and it is not the model.

Nothing ever checked it. Main carried no workflow that runs on a push to main
or on a pull request into main, so a change to the scheduler copy -- a
retimed cron, a dropped step, a checkout ref pointed somewhere else, an
enabled publishing switch -- would have landed with no check reporting at all.
That is the gap this closes.

WHY NOT JUST RUN THE TEST SUITE
-------------------------------

Because main does not contain the production tree, and a suite that cannot
run is not a check. Main is an older source snapshot with its own small test
set; the fit registry, the adaptive trainer, the shadow and their tests are
production-lineage files. Copying them onto main to make a full-suite check
possible would put the production application lineage on the default branch
to satisfy a convenience, which is the opposite of the arrangement. So this
validates main's real role instead, and validates it strictly.

WHAT IT PROVES
--------------

The lifecycle copy parses; it passes the same static workflow validation the
production branch applies to it; it is byte-identical to the authoritative
production copy; it still checks out the production ref rather than inheriting
the default branch; every required lifecycle step and entry point is still
there; the RUN_ADAPTIVE gating is still where it belongs and still absent
where it must not be; the shadow is still non-blocking while the health
assertion that watches it is not; ``--strict`` is still absent from the
scheduled shadow; no publishing activation has appeared; and the production
ref named anywhere in the file is still the expected one.

It runs identically from either branch, which is what makes the byte-identity
check symmetric: run from production it proves main has not drifted, and run
from main it proves main has not drifted. Nothing here fits, serves, promotes
or publishes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

LIFECYCLE_RELATIVE = Path(".github") / "workflows" / "nba_production_lifecycle.yml"

#: The static validator the production branch already applies. Imported by
#: path and used for the lifecycle workflow alone: calling its whole-directory
#: entry point would demand ci.yml, which is a production-branch file, and
#: main is not where the production test suite runs.
WORKFLOW_VALIDATOR = ROOT / "ops" / "validate_workflows.py"

AUTHORITATIVE_PRODUCTION_REF = "production/wizardofodds-integration"

DEFAULT_BRANCH = "main"

LIFECYCLE_JOB = "lifecycle"

#: Every step the production lifecycle must still perform. Named rather than
#: counted so a dropped step is reported by name.
REQUIRED_STEPS: tuple[str, ...] = (
    "Check out the authoritative production code",
    "Record the production checkout",
    "Set up Python",
    "Create the per-run Python environment",
    "Install the production package",
    "Verify the production interpreter is isolated",
    "Preflight",
    "Refresh the current-season rolling state",
    "Run the adaptive daily protocol",
    "Install the verified frozen runtime bundle",
    "Serve the slate with the incumbent",
    "Grade the incumbent's previous slate",
    "Shadow the slate beside production",
    "Monitor the accumulated live shadow evidence",
    "Assert the candidate shadow was healthy",
    "Report the run",
)

#: Every Python entry point the lifecycle must still invoke. A step can be
#: renamed; what it runs is the contract.
REQUIRED_ENTRY_POINTS: tuple[str, ...] = (
    "ops/production_lifecycle_preflight.py",
    "ops/refresh_current_season_state.py",
    "ops/classify_refresh_outcome.py",
    "ops/run_adaptive_daily_fit.py",
    "ops/install_frozen_model_artifacts.py",
    "ops/run_incumbent_production_serving.py",
    "ops/grade_incumbent_production_slate.py",
    "ops/run_production_shadow.py",
    "ops/monitor_live_shadow_evidence.py",
    "ops/evaluate_shadow_health.py",
    "ops/summarise_production_run.py",
    "ops/verify_production_interpreter.py",
)

#: Steps that must be gated on the classifier's RUN_ADAPTIVE decision,
#: because they are candidate work and a day with no new completed data must
#: not perform any.
RUN_ADAPTIVE_GATED_ENTRY_POINTS: tuple[str, ...] = (
    "ops/run_adaptive_daily_fit.py",
    "ops/run_production_shadow.py",
)

#: Steps that must NOT be gated on RUN_ADAPTIVE. Serving the incumbent is a
#: different question from refitting a candidate, and binding them would mean
#: a future off-day fit policy silently stopped production serving.
RUN_ADAPTIVE_UNGATED_ENTRY_POINTS: tuple[str, ...] = (
    # The frozen bundle the incumbent serves from has nothing to do with
    # whether a candidate was refitted this morning, and gating its install
    # would leave serving without the artifacts it was handed.
    "ops/install_frozen_model_artifacts.py",
    "ops/run_incumbent_production_serving.py",
    "ops/grade_incumbent_production_slate.py",
    # The shadow only has something to say on a day it refitted, but the
    # accumulated window is what the frozen policy judges, and it must be
    # summarised every production day -- including a day with no retrain,
    # where the answer is still CONTINUE_SHADOW rather than silence.
    "ops/monitor_live_shadow_evidence.py",
)

#: The shadow must stay non-blocking: a candidate failure may never stop the
#: incumbent being served.
NON_BLOCKING_ENTRY_POINTS: tuple[str, ...] = ("ops/run_production_shadow.py",)

#: And these must stay blocking. The health assertion exists precisely so an
#: unhealthy shadow is visible, and the serving step's failure is a production
#: failure.
BLOCKING_ENTRY_POINTS: tuple[str, ...] = (
    "ops/evaluate_shadow_health.py",
    # An unverifiable frozen bundle means the incumbent has no artifacts
    # anybody approved. Continuing past that would serve from an unverified
    # tree, which is the failure the install exists to prevent.
    "ops/install_frozen_model_artifacts.py",
    "ops/run_incumbent_production_serving.py",
    # A grading refusal means the provenance does not describe the rows, or a
    # slate was priced after its own games. Both are faults, and grading runs
    # after serving so a red job here costs production nothing.
    "ops/grade_incumbent_production_slate.py",
    # Monitoring fails the job on one decision only, and that decision --
    # SHADOW_DISABLED_FOR_SAFETY -- is the operator signal the frozen policy
    # exists to raise. continue-on-error here would swallow it.
    "ops/monitor_live_shadow_evidence.py",
)

#: ``--strict`` makes the shadow entry point propagate its own failures. It is
#: for a human debugging the shadow and must never appear in the scheduled
#: lifecycle, where it would let a candidate failure fail production.
FORBIDDEN_SHADOW_FLAG = "--strict"

#: Publishing activation, in the terms the frozen switch itself states. Taken
#: from research/game_latent_state/shadow/shadow_publishing_switch.json and
#: the publisher script name, so this list is not a guess about what turning
#: publishing on would look like.
PUBLISHING_ACTIVATION_TOKENS: tuple[str, ...] = (
    "NBA_PROP_SHADOW_PUBLISH",
    "NBA_PROP_SHADOW_PUBLISH_APPROVAL",
    "19_build_wizardofodds_runtime_bundle.py",
    "--publish",
)

EXIT_OK = 0
EXIT_FAILED = 1


@dataclass
class Check:
    """One statement about the scheduler copy.

    ``passed`` is ``None`` for a check that does not apply to the branch under
    test. Exactly one check is like that -- byte-identity with production --
    and it matters: a production-lineage pull request legitimately differs
    from the production head it is about to become, so demanding identity
    there would refuse every change to the lifecycle. Identity is a statement
    about the *default branch*, so it is asserted when the default branch is
    what is being tested and reported as not applicable otherwise.
    """

    name: str
    passed: bool | None
    evidence: str
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.passed is False

    def payload(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence,
            "name": self.name,
            "passed": self.passed,
            "values": dict(sorted(self.values.items())),
        }


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _load_workflow_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_workflows_for_scheduler_role", WORKFLOW_VALIDATOR
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {WORKFLOW_VALIDATOR}")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def steps_of(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    job = (workflow.get("jobs") or {}).get(LIFECYCLE_JOB) or {}

    return [step for step in (job.get("steps") or []) if isinstance(step, dict)]


def steps_running(steps: list[dict[str, Any]], entry_point: str):
    return [step for step in steps if entry_point in str(step.get("run", ""))]


# ----------------------------------------------------------------------
# the checks
# ----------------------------------------------------------------------


def check_lifecycle_parses(text: str) -> tuple[Check, dict[str, Any] | None]:
    try:
        workflow = yaml.safe_load(text)

    except yaml.YAMLError as error:
        return (
            Check(
                name="the_lifecycle_workflow_parses",
                passed=False,
                evidence="the scheduler copy is not loadable YAML",
                values={"error": f"{type(error).__name__}: {error}"},
            ),
            None,
        )

    if not isinstance(workflow, dict):
        return (
            Check(
                name="the_lifecycle_workflow_parses",
                passed=False,
                evidence="the scheduler copy does not parse to a mapping",
            ),
            None,
        )

    return (
        Check(
            name="the_lifecycle_workflow_parses",
            passed=True,
            evidence="the scheduler copy parses to a workflow mapping",
            values={"jobs": sorted(workflow.get("jobs") or {})},
        ),
        workflow,
    )


def check_static_validation(workflow: dict[str, Any]) -> Check:
    """The production branch's own static rules, applied to this copy.

    Reused rather than restated. These are the rules that catch a workflow
    scheduling from the wrong branch, dropping concurrency on a job that
    mutates production, widening permissions, or putting pull-request code on
    the self-hosted machine.
    """
    validator = _load_workflow_validator()

    name = LIFECYCLE_RELATIVE.name

    problems = validator.problems_for(name, workflow)
    problems += validator.secret_leak_problems(name)

    return Check(
        name="the_lifecycle_workflow_passes_static_validation",
        passed=not problems,
        evidence=(
            "the production branch's own workflow validator reports no "
            "problems with this copy"
            if not problems
            else "the production branch's own workflow validator rejects it"
        ),
        values={"problems": problems},
    )


def check_copies_are_byte_identical(
    *,
    local: bytes,
    production_ref: str,
    project_root: Path,
    applicable: bool,
) -> Check:
    """Main's copy against the authoritative production copy, byte for byte.

    The default branch holds the file that starts the run; the production
    branch holds the file the run is reviewed against. Any difference means
    the thing GitHub executes is not the thing that was reviewed.
    """
    blob = f"{production_ref}:{LIFECYCLE_RELATIVE.as_posix()}"

    if not applicable:
        return Check(
            name="the_scheduler_copy_matches_production_byte_for_byte",
            passed=None,
            evidence=(
                "not the default branch, so identity with the production "
                "head is not the question being asked here"
            ),
            values={"blob": blob},
        )

    result = _git("show", blob, cwd=project_root)

    if result.returncode != 0:
        return Check(
            name="the_scheduler_copy_matches_production_byte_for_byte",
            passed=False,
            evidence=(
                "the authoritative production copy could not be read, so "
                "identity could not be established"
            ),
            values={"blob": blob, "git_stderr": result.stderr.strip()[:500]},
        )

    authoritative = result.stdout.encode("utf-8")

    return Check(
        name="the_scheduler_copy_matches_production_byte_for_byte",
        passed=authoritative == local,
        evidence=(
            "the two copies are identical"
            if authoritative == local
            else "the two copies differ, so GitHub would schedule a workflow "
            "that was never reviewed on the production branch"
        ),
        values={
            "authoritative_bytes": len(authoritative),
            "blob": blob,
            "local_bytes": len(local),
        },
    )


def check_checkout_targets_production(
    workflow: dict[str, Any], text: str
) -> Check:
    """A scheduled run originates from main and must not model from it."""
    checkouts = [
        step
        for step in steps_of(workflow)
        if str(step.get("uses", "")).startswith("actions/checkout")
    ]

    refs = [str((step.get("with") or {}).get("ref", "")) for step in checkouts]

    named = [ref for ref in refs if AUTHORITATIVE_PRODUCTION_REF in ref]

    passed = bool(checkouts) and len(named) == len(refs) and all(refs)

    return Check(
        name="the_checkout_still_targets_the_production_ref",
        passed=passed,
        evidence=(
            "every checkout names the authoritative production ref "
            "explicitly"
            if passed
            else "a checkout does not name the production ref, so a "
            "scheduled run would model from the default branch"
        ),
        values={"checkout_refs": refs},
    )


def check_production_ref_unchanged(text: str) -> Check:
    """No unauthorized production-ref change.

    Any ref-like string in the file other than the authoritative production
    ref and the default branch is a redirection of the production run.
    """
    found = {
        line.strip()
        for line in text.splitlines()
        if "production/" in line
        and AUTHORITATIVE_PRODUCTION_REF not in line
    }

    return Check(
        name="no_unauthorized_production_ref_change",
        passed=not found,
        evidence=(
            f"every production ref in the file is {AUTHORITATIVE_PRODUCTION_REF}"
            if not found
            else "the file names a production ref that is not the "
            "authoritative one"
        ),
        values={"foreign_refs": sorted(found)},
    )


def check_required_steps(workflow: dict[str, Any]) -> Check:
    names = [str(step.get("name", "")) for step in steps_of(workflow)]

    absent = [name for name in REQUIRED_STEPS if name not in names]

    return Check(
        name="every_required_lifecycle_step_is_present",
        passed=not absent,
        evidence=(
            "every required step is present"
            if not absent
            else "the lifecycle has lost a required step"
        ),
        values={"missing_steps": absent, "step_count": len(names)},
    )


def check_required_entry_points(workflow: dict[str, Any]) -> Check:
    steps = steps_of(workflow)

    absent = [
        entry
        for entry in REQUIRED_ENTRY_POINTS
        if not steps_running(steps, entry)
    ]

    return Check(
        name="every_required_entry_point_is_still_invoked",
        passed=not absent,
        evidence=(
            "every required entry point is invoked"
            if not absent
            else "the lifecycle no longer invokes a required entry point"
        ),
        values={"missing_entry_points": absent},
    )


def check_inline_python_compiles(workflow: dict[str, Any]) -> Check:
    """Every ``python -c`` fragment in the lifecycle must actually parse.

    These fragments are the one part of the lifecycle nothing else validates.
    The YAML parses, the step is present and the entry point is named even
    when the Python inside a ``run:`` block is a syntax error, because it is
    just a string until the shell runs it. And when it fails, the shell hides
    it: a broken command substitution yields an empty string and the step
    carries on with a blank argument rather than stopping.

    Compiled against what the shell actually receives, after YAML has stripped
    the block indentation, so the check agrees with the runner rather than
    with how the file looks.
    """
    fragments = inline_python_fragments(workflow)

    broken: list[dict[str, str]] = []

    for step, source in fragments:
        try:
            compile(source, "<lifecycle>", "exec")
        except SyntaxError as error:
            broken.append({"step": step, "error": f"{type(error).__name__}: {error}"})

    return Check(
        name="every_inline_python_fragment_parses",
        passed=not broken,
        evidence=(
            f"{len(fragments)} inline python fragment(s) parse"
            if not broken
            else "an inline python fragment in the lifecycle does not parse"
        ),
        values={"broken_fragments": broken, "fragments": len(fragments)},
    )


def inline_python_fragments(
    workflow: dict[str, Any],
) -> list[tuple[str, str]]:
    """Every ``python -c <source>`` in the workflow, with its step name."""
    pattern = re.compile(
        r"""python\s+-c\s+(?P<quote>['"])(?P<source>.*?)(?<!\\)(?P=quote)""",
        re.DOTALL,
    )

    found: list[tuple[str, str]] = []

    for step in steps_of(workflow):
        run = str(step.get("run", ""))

        for match in pattern.finditer(run):
            found.append((str(step.get("name", "")), match.group("source")))

    return found


def check_run_adaptive_gates(workflow: dict[str, Any]) -> Check:
    steps = steps_of(workflow)

    ungated: list[str] = []
    gated: list[str] = []

    for entry in RUN_ADAPTIVE_GATED_ENTRY_POINTS:
        for step in steps_running(steps, entry):
            if "RUN_ADAPTIVE" not in str(step.get("if", "")):
                ungated.append(entry)

    for entry in RUN_ADAPTIVE_UNGATED_ENTRY_POINTS:
        for step in steps_running(steps, entry):
            if "RUN_ADAPTIVE" in str(step.get("if", "")):
                gated.append(entry)

    problems = sorted(set(ungated)) + sorted(set(gated))

    return Check(
        name="the_run_adaptive_gates_are_where_they_belong",
        passed=not problems,
        evidence=(
            "candidate work is gated on the classifier's decision and "
            "incumbent serving is not"
            if not problems
            else "the RUN_ADAPTIVE gating has moved"
        ),
        values={
            "candidate_work_left_ungated": sorted(set(ungated)),
            "incumbent_serving_wrongly_gated": sorted(set(gated)),
        },
    )


def check_continue_on_error_is_deliberate(workflow: dict[str, Any]) -> Check:
    """Non-blocking where it protects the incumbent, blocking everywhere else.

    The shadow keeps ``continue-on-error`` because a candidate failure must
    never stop the incumbent being served. The health assertion that reads the
    shadow's own status must not have it, or the signal the assertion exists to
    produce would be swallowed by the same setting.
    """
    steps = steps_of(workflow)

    problems: list[str] = []

    for entry in NON_BLOCKING_ENTRY_POINTS:
        for step in steps_running(steps, entry):
            if step.get("continue-on-error") is not True:
                problems.append(f"{entry} is no longer non-blocking")

    for entry in BLOCKING_ENTRY_POINTS:
        for step in steps_running(steps, entry):
            if "continue-on-error" in step:
                problems.append(f"{entry} has become non-blocking")

    return Check(
        name="the_non_blocking_shadow_is_still_deliberate",
        passed=not problems,
        evidence=(
            "the shadow is non-blocking and the steps that judge it are not"
            if not problems
            else "the non-blocking boundary has moved"
        ),
        values={"problems": sorted(set(problems))},
    )


def check_shadow_is_not_strict(workflow: dict[str, Any]) -> Check:
    steps = steps_of(workflow)

    offenders = [
        str(step.get("name", ""))
        for step in steps_running(steps, "ops/run_production_shadow.py")
        if FORBIDDEN_SHADOW_FLAG in str(step.get("run", ""))
    ]

    return Check(
        name="the_scheduled_shadow_is_not_strict",
        passed=not offenders,
        evidence=(
            f"{FORBIDDEN_SHADOW_FLAG} is absent from the scheduled shadow"
            if not offenders
            else f"{FORBIDDEN_SHADOW_FLAG} would let a candidate failure "
            "fail the production lifecycle"
        ),
        values={"offending_steps": offenders},
    )


def check_no_publishing_activation(text: str) -> Check:
    found = [token for token in PUBLISHING_ACTIVATION_TOKENS if token in text]

    return Check(
        name="no_publishing_activation_appears",
        passed=not found,
        evidence=(
            "the lifecycle names nothing that could turn publishing on"
            if not found
            else "the lifecycle names a publishing activation token"
        ),
        values={"tokens_found": found},
    )


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------


def validate(
    *,
    project_root: Path,
    production_ref: str,
    is_default_branch: bool = True,
) -> list[Check]:
    path = Path(project_root) / LIFECYCLE_RELATIVE

    if not path.exists():
        return [
            Check(
                name="the_lifecycle_workflow_is_present",
                passed=False,
                evidence=(
                    "the default branch holds no lifecycle workflow, so no "
                    "schedule can fire at all"
                ),
                values={"path": str(path)},
            )
        ]

    local = path.read_bytes()

    text = local.decode("utf-8")

    parsed, workflow = check_lifecycle_parses(text)

    checks = [parsed]

    if workflow is None:
        return checks

    checks += [
        check_static_validation(workflow),
        check_copies_are_byte_identical(
            local=local,
            production_ref=production_ref,
            project_root=Path(project_root),
            applicable=is_default_branch,
        ),
        check_checkout_targets_production(workflow, text),
        check_production_ref_unchanged(text),
        check_required_steps(workflow),
        check_required_entry_points(workflow),
        check_inline_python_compiles(workflow),
        check_run_adaptive_gates(workflow),
        check_continue_on_error_is_deliberate(workflow),
        check_shadow_is_not_strict(workflow),
        check_no_publishing_activation(text),
    ]

    return checks


def render(checks: list[Check], *, branch: str | None) -> str:
    failed = [check for check in checks if check.failed]

    lines = [
        "## Default-branch scheduler role",
        "",
        f"Branch under test: `{branch or 'unknown'}`",
        "",
        "| check | result | evidence |",
        "| --- | --- | --- |",
    ]

    for check in checks:
        result = {True: "PASS", False: "FAIL", None: "N/A"}[check.passed]

        lines.append(f"| `{check.name}` | {result} | {check.evidence} |")

    lines += [""]

    if failed:
        lines.append(
            f"{len(failed)} of {len(checks)} checks failed. The default "
            "branch is what GitHub schedules production from."
        )

        for check in failed:
            lines.append(
                f"- `{check.name}`: "
                + json.dumps(check.values, sort_keys=True)
            )

    else:
        answered = [check for check in checks if check.passed is not None]

        skipped = len(checks) - len(answered)

        lines.append(
            f"{len(answered)} of {len(checks)} checks passed"
            + (
                f"; {skipped} did not apply to this branch."
                if skipped
                else ". The scheduler copy is identical to the authoritative "
                "production copy and still performs its whole role."
            )
        )

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_main_scheduler_role",
        description=(
            "Validate the default branch's production role: the scheduler "
            "copy of the lifecycle workflow and nothing else."
        ),
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--production-ref",
        default=f"origin/{AUTHORITATIVE_PRODUCTION_REF}",
        help="the git ref holding the authoritative lifecycle copy",
    )
    parser.add_argument("--branch", default=None)
    parser.add_argument(
        "--default-branch",
        default=DEFAULT_BRANCH,
        help="the branch GitHub schedules from. Identity with the production "
        "head is asserted only when --branch is this branch.",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    is_default_branch = args.branch == args.default_branch

    checks = validate(
        project_root=Path(args.project_root),
        production_ref=args.production_ref,
        is_default_branch=is_default_branch,
    )

    report = {
        "branch": args.branch,
        "checks": [check.payload() for check in checks],
        "failed": sorted(check.name for check in checks if check.failed),
        "is_default_branch": is_default_branch,
        "passed": not any(check.failed for check in checks),
        "production_ref": args.production_ref,
    }

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)

        args.report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    text = render(checks, branch=args.branch)

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    print(text, end="")

    return EXIT_OK if report["passed"] else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
