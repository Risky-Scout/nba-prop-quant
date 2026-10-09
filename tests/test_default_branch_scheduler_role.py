"""Tests for the default branch's reported check.

What is under test is a *check*, so the tests that matter are the ones that
break it on purpose. Each weakening below is a change somebody could land on
main with no reviewer noticing -- a dropped lifecycle step, a redirected
checkout, a strict shadow, a publishing token, a gate moved onto the serving
step -- and each must turn exactly the corresponding named check false.

Nothing here runs a workflow, trains, serves, promotes or publishes.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

VALIDATOR = REPO / "ops" / "validate_main_scheduler_role.py"

GUARD_WORKFLOW = REPO / ".github" / "workflows" / "default_branch_guard.yml"

LIFECYCLE_RELATIVE = (
    Path(".github") / "workflows" / "nba_production_lifecycle.yml"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_main_scheduler_role", VALIDATOR
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


role = _load()


# ----------------------------------------------------------------------
# a throwaway two-branch repository
# ----------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A git repository holding the lifecycle workflow on two refs.

    Built rather than mocked, because the check this module exists for is a
    comparison against a git ref and a fake ref would not exercise it.
    """
    root = tmp_path / "repo"

    (root / LIFECYCLE_RELATIVE.parent).mkdir(parents=True)

    # The validator imports the production branch's own static validator by
    # path, so that file has to be where it expects it.
    (root / "ops").mkdir()

    (root / "ops" / "validate_workflows.py").write_bytes(
        (REPO / "ops" / "validate_workflows.py").read_bytes()
    )

    (root / LIFECYCLE_RELATIVE).write_bytes(
        (REPO / LIFECYCLE_RELATIVE).read_bytes()
    )

    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "pytest@example.invalid", cwd=root)
    _git("config", "user.name", "pytest", cwd=root)
    _git("add", "-A", cwd=root)
    _git("commit", "-qm", "scheduler copy", cwd=root)
    _git(
        "branch",
        "origin/production/wizardofodds-integration",
        cwd=root,
    )

    return root


def validate(repository: Path, *, is_default_branch: bool = True):
    return role.validate(
        project_root=repository,
        production_ref="origin/production/wizardofodds-integration",
        is_default_branch=is_default_branch,
    )


def named(checks, name):
    for check in checks:
        if check.name == name:
            return check

    raise AssertionError(f"no check named {name!r}: {[c.name for c in checks]}")


def failures(checks) -> list[str]:
    return sorted(check.name for check in checks if check.failed)


def rewrite(repository: Path, transform) -> None:
    """Apply ``transform`` to the scheduler copy only, leaving production."""
    path = repository / LIFECYCLE_RELATIVE

    path.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")


def rewrite_both(repository: Path, transform) -> None:
    """Weaken the lifecycle on both refs, so identity stays satisfied.

    Used for the checks that are about the lifecycle's content rather than
    about drift between the copies: otherwise every content test would also
    trip the byte-identity check and prove less.
    """
    rewrite(repository, transform)

    _git("add", "-A", cwd=repository)
    _git("commit", "-qm", "weakened", cwd=repository)
    _git(
        "branch",
        "-f",
        "origin/production/wizardofodds-integration",
        "HEAD",
        cwd=repository,
    )


# ----------------------------------------------------------------------
# the healthy case
# ----------------------------------------------------------------------


def test_an_identical_scheduler_copy_passes_every_check(repository: Path):
    checks = validate(repository)

    assert failures(checks) == []
    assert named(checks, "the_scheduler_copy_matches_production_byte_for_byte").passed


def test_the_repository_as_it_stands_has_a_sound_lifecycle():
    """The real file, checked for everything except drift.

    Identity cannot be asserted from inside a remediation branch -- the
    production head is what this branch is about to become -- so that one
    check is reported as not applicable here and asserted on the default
    branch by the workflow.
    """
    checks = role.validate(
        project_root=REPO,
        production_ref="origin/production/wizardofodds-integration",
        is_default_branch=False,
    )

    assert failures(checks) == []


def test_identity_is_not_demanded_of_a_production_lineage_branch(
    repository: Path,
):
    """Otherwise the check would refuse every change to the lifecycle.

    A production pull request differs from the production head by
    construction. Demanding identity there would mean the lifecycle could
    never be changed again, which is a broken check rather than a strict one.
    """
    rewrite(repository, lambda text: text + "\n# a production-lineage change\n")

    on_default = validate(repository, is_default_branch=True)

    assert failures(on_default) == [
        "the_scheduler_copy_matches_production_byte_for_byte"
    ]

    on_branch = validate(repository, is_default_branch=False)

    assert failures(on_branch) == []
    assert (
        named(
            on_branch, "the_scheduler_copy_matches_production_byte_for_byte"
        ).passed
        is None
    )


# ----------------------------------------------------------------------
# negative controls: each weakening fails its own check
# ----------------------------------------------------------------------


def test_unparseable_yaml_fails_the_parse_check(repository: Path):
    rewrite_both(repository, lambda text: text + "\n  : : not yaml : :\n")

    checks = validate(repository)

    assert failures(checks) == ["the_lifecycle_workflow_parses"]


def test_a_missing_scheduler_copy_is_reported(repository: Path):
    (repository / LIFECYCLE_RELATIVE).unlink()

    checks = validate(repository)

    assert failures(checks) == ["the_lifecycle_workflow_is_present"]


def test_a_dropped_step_fails_the_required_step_check(repository: Path):
    rewrite_both(
        repository,
        lambda text: text.replace(
            "      - name: Assert the candidate shadow was healthy",
            "      - name: Something else entirely",
        ),
    )

    checks = validate(repository)

    missing = named(checks, "every_required_lifecycle_step_is_present")

    assert missing.passed is False
    assert missing.values["missing_steps"] == [
        "Assert the candidate shadow was healthy"
    ]


def test_a_dropped_entry_point_fails_its_own_check(repository: Path):
    rewrite_both(
        repository,
        lambda text: text.replace(
            "python ops/run_incumbent_production_serving.py",
            "echo skipping incumbent serving #",
        ),
    )

    checks = validate(repository)

    dropped = named(checks, "every_required_entry_point_is_still_invoked")

    assert dropped.passed is False
    assert dropped.values["missing_entry_points"] == [
        "ops/run_incumbent_production_serving.py"
    ]


def test_a_redirected_checkout_fails_the_ref_check(repository: Path):
    rewrite_both(
        repository,
        lambda text: text.replace(
            "ref: ${{ github.event_name == 'schedule' && "
            "'production/wizardofodds-integration' || "
            "(inputs.production_ref || 'production/wizardofodds-integration') }}",
            "ref: main",
        ),
    )

    checks = validate(repository)

    assert "the_checkout_still_targets_the_production_ref" in failures(checks)


def test_a_foreign_production_ref_is_reported(repository: Path):
    rewrite_both(
        repository,
        lambda text: text.replace(
            "default: production/wizardofodds-integration",
            "default: production/some-other-lineage",
        ),
    )

    checks = validate(repository)

    foreign = named(checks, "no_unauthorized_production_ref_change")

    assert foreign.passed is False
    assert foreign.values["foreign_refs"]


def test_ungating_the_adaptive_fit_fails_the_gate_check(repository: Path):
    rewrite_both(
        repository,
        lambda text: text.replace(
            "        if: env.MODE == 'production' && env.RUN_ADAPTIVE == 'true'\n"
            "        env:\n"
            "          NBA_PROP_DATA_DIR: ${{ vars.NBA_PROP_DATA_DIR }}\n"
            "          NBA_PROP_FIT_REGISTRY_DIR: "
            "${{ vars.NBA_PROP_FIT_REGISTRY_DIR }}\n"
            "          NBA_PROP_WORK_DIR: ${{ vars.NBA_PROP_WORK_DIR }}",
            "        if: env.MODE == 'production'\n"
            "        env:\n"
            "          NBA_PROP_DATA_DIR: ${{ vars.NBA_PROP_DATA_DIR }}\n"
            "          NBA_PROP_FIT_REGISTRY_DIR: "
            "${{ vars.NBA_PROP_FIT_REGISTRY_DIR }}\n"
            "          NBA_PROP_WORK_DIR: ${{ vars.NBA_PROP_WORK_DIR }}",
        ),
    )

    checks = validate(repository)

    gates = named(checks, "the_run_adaptive_gates_are_where_they_belong")

    assert gates.passed is False
    assert gates.values["candidate_work_left_ungated"] == [
        "ops/run_adaptive_daily_fit.py"
    ]


def test_gating_incumbent_serving_on_run_adaptive_fails_the_gate_check(
    repository: Path,
):
    """The other direction, and the one the brief asks for by name."""
    rewrite_both(
        repository,
        lambda text: text.replace(
            "      - name: Serve the slate with the incumbent\n"
            "        if: env.MODE == 'production'",
            "      - name: Serve the slate with the incumbent\n"
            "        if: env.MODE == 'production' && env.RUN_ADAPTIVE == 'true'",
        ),
    )

    checks = validate(repository)

    gates = named(checks, "the_run_adaptive_gates_are_where_they_belong")

    assert gates.passed is False
    assert gates.values["incumbent_serving_wrongly_gated"] == [
        "ops/run_incumbent_production_serving.py"
    ]


def test_making_the_shadow_blocking_fails_the_deliberateness_check(
    repository: Path,
):
    """The shadow's non-blocking setting is a safety property, not laziness."""
    rewrite_both(
        repository,
        lambda text: text.replace(
            "      - name: Shadow the slate beside production\n"
            "        if: env.MODE == 'production' && env.RUN_ADAPTIVE == 'true'\n"
            "        continue-on-error: true",
            "      - name: Shadow the slate beside production\n"
            "        if: env.MODE == 'production' && env.RUN_ADAPTIVE == 'true'",
        ),
    )

    checks = validate(repository)

    assert "the_non_blocking_shadow_is_still_deliberate" in failures(checks)


def test_hiding_the_health_assertion_fails_the_deliberateness_check(
    repository: Path,
):
    """The exact bug the health assertion was added to fix, reintroduced.

    continue-on-error on the step that judges the shadow would swallow the
    one signal that step exists to produce, leaving GitHub green again.
    """
    rewrite_both(
        repository,
        lambda text: text.replace(
            "      - name: Assert the candidate shadow was healthy\n"
            "        if: always() && env.MODE == 'production' && "
            "env.RUN_ADAPTIVE == 'true'",
            "      - name: Assert the candidate shadow was healthy\n"
            "        continue-on-error: true\n"
            "        if: always() && env.MODE == 'production' && "
            "env.RUN_ADAPTIVE == 'true'",
        ),
    )

    checks = validate(repository)

    deliberate = named(checks, "the_non_blocking_shadow_is_still_deliberate")

    assert deliberate.passed is False
    assert deliberate.values["problems"] == [
        "ops/evaluate_shadow_health.py has become non-blocking"
    ]


def test_hiding_incumbent_serving_fails_the_deliberateness_check(
    repository: Path,
):
    rewrite_both(
        repository,
        lambda text: text.replace(
            "      - name: Serve the slate with the incumbent\n"
            "        if: env.MODE == 'production'",
            "      - name: Serve the slate with the incumbent\n"
            "        continue-on-error: true\n"
            "        if: env.MODE == 'production'",
        ),
    )

    checks = validate(repository)

    deliberate = named(checks, "the_non_blocking_shadow_is_still_deliberate")

    assert deliberate.passed is False
    assert deliberate.values["problems"] == [
        "ops/run_incumbent_production_serving.py has become non-blocking"
    ]


def test_a_strict_scheduled_shadow_fails_its_own_check(repository: Path):
    rewrite_both(
        repository,
        lambda text: text.replace(
            "          python ops/run_production_shadow.py \\\n"
            "            --slate-date \"$slate\" \\",
            "          python ops/run_production_shadow.py \\\n"
            "            --strict \\\n"
            "            --slate-date \"$slate\" \\",
        ),
    )

    checks = validate(repository)

    strict = named(checks, "the_scheduled_shadow_is_not_strict")

    assert strict.passed is False
    assert strict.values["offending_steps"] == [
        "Shadow the slate beside production"
    ]


@pytest.mark.parametrize(
    "token",
    (
        "NBA_PROP_SHADOW_PUBLISH",
        "19_build_wizardofodds_runtime_bundle.py",
        "--publish",
    ),
)
def test_a_publishing_activation_token_fails_its_own_check(
    token: str, repository: Path
):
    rewrite_both(
        repository, lambda text: text.replace("      - name: Report the run", f"      - name: Publish\n        run: echo {token}\n\n      - name: Report the run")
    )

    checks = validate(repository)

    publishing = named(checks, "no_publishing_activation_appears")

    assert publishing.passed is False
    assert token in publishing.values["tokens_found"]


def test_a_statically_invalid_workflow_fails_static_validation(
    repository: Path,
):
    """The production branch's own rules, still applied here."""
    rewrite_both(
        repository, lambda text: text.replace("- cron: '37 9 * * *'", "- cron: '0 9 * * *'")
    )

    checks = validate(repository)

    static = named(checks, "the_lifecycle_workflow_passes_static_validation")

    assert static.passed is False
    assert any("top of the hour" in problem for problem in static.values["problems"])


# ----------------------------------------------------------------------
# the exit code and the workflow wiring
# ----------------------------------------------------------------------


def test_the_cli_exits_nonzero_and_records_the_failed_check(
    repository: Path, tmp_path: Path
):
    rewrite(repository, lambda text: text + "\n# drift\n")

    report_path = tmp_path / "report.json"

    code = role.main(
        [
            "--project-root",
            str(repository),
            "--production-ref",
            "origin/production/wizardofodds-integration",
            "--branch",
            "main",
            "--report-path",
            str(report_path),
        ]
    )

    assert code == role.EXIT_FAILED

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["is_default_branch"]
    assert report["failed"] == [
        "the_scheduler_copy_matches_production_byte_for_byte"
    ]


def test_the_cli_exits_zero_on_a_sound_scheduler_copy(
    repository: Path, tmp_path: Path
):
    code = role.main(
        [
            "--project-root",
            str(repository),
            "--production-ref",
            "origin/production/wizardofodds-integration",
            "--branch",
            "main",
            "--summary-path",
            str(tmp_path / "summary.md"),
        ]
    )

    assert code == role.EXIT_OK

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")

    assert "Default-branch scheduler role" in summary


def guard_workflow() -> dict:
    return yaml.safe_load(GUARD_WORKFLOW.read_text(encoding="utf-8"))


def test_the_guard_reports_on_a_push_to_the_default_branch():
    """The gap being closed: main had no check triggered by a push to main."""
    triggers = guard_workflow().get("on") or guard_workflow().get(True)

    assert "main" in triggers["push"]["branches"]
    assert "main" in triggers["pull_request"]["branches"]


def test_the_guard_never_runs_on_the_production_runner():
    """It may execute pull-request code, so it must stay GitHub-hosted."""
    for job in guard_workflow()["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest"


def test_the_guard_asks_for_no_more_permission_than_reading():
    assert guard_workflow()["permissions"] == {"contents": "read"}


def test_the_guard_fetches_the_ref_it_compares_against():
    """Otherwise the identity check would fail for want of a ref, not drift."""
    commands = "\n".join(
        str(step.get("run", ""))
        for step in guard_workflow()["jobs"]["scheduler-role"]["steps"]
    )

    assert "git fetch" in commands
    assert "origin production/wizardofodds-integration" in commands
    assert "ops/validate_main_scheduler_role.py" in commands


def test_the_guard_tests_the_base_branch_of_a_pull_request():
    """A pull request's merge result is what would become the base branch.

    Passing the head ref instead would mean a pull request into main was
    never checked for identity, which is the whole point.
    """
    commands = "\n".join(
        str(step.get("run", ""))
        for step in guard_workflow()["jobs"]["scheduler-role"]["steps"]
    )

    assert "github.base_ref || github.ref_name" in commands


def test_the_validator_needs_nothing_from_the_production_tree():
    """What makes this synchronisable to main at all.

    Main deliberately does not hold the production application lineage. A
    validator that imported the package could not run there, so this one
    imports only the standard library, PyYAML and the workflow validator
    beside it.
    """
    source = VALIDATOR.read_text(encoding="utf-8")

    assert "nba_prop_quant" not in source
    assert "import pandas" not in source
    assert "import numpy" not in source
