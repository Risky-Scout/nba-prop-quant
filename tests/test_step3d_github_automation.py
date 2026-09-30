"""Step 3D: GitHub automation of the production lifecycle.

These tests prove orchestration safety, not statistics. The Step 3C suite
already covers the adaptive behaviour; nothing here retrains, refits,
promotes, or contacts an external service.

The property that matters most: a scheduled run originates from the default
branch, which is not the production branch, so the workflow must name and
verify the authoritative production ref rather than inherit whatever the
default branch carries.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

import pytest
import yaml


PROJECT = Path(__file__).resolve().parents[1]

WORKFLOW_DIR = PROJECT / ".github" / "workflows"

PRODUCTION_WORKFLOW = WORKFLOW_DIR / "nba_production_lifecycle.yml"

CI_WORKFLOW = WORKFLOW_DIR / "ci.yml"

AUTHORITATIVE_BRANCH = "production/wizardofodds-integration"


def head_sha() -> str:
    """The checkout under test, whatever shape CI gave it."""
    return preflight.git("rev-parse", "HEAD")

GITHUB_DEFAULT_BRANCH = "main"


def load_ops(name: str):
    spec = importlib.util.spec_from_file_location(
        f"ops_{name}", PROJECT / "ops" / f"{name}.py"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


preflight = load_ops("production_lifecycle_preflight")

validator = load_ops("validate_workflows")

summariser = load_ops("summarise_production_run")


@pytest.fixture(autouse=True)
def block_all_network(monkeypatch):
    def deny(*args, **kwargs):
        raise RuntimeError(
            f"network access is forbidden in this test module: {args!r}"
        )

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)

    yield


def workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def triggers(payload: dict) -> dict:
    # PyYAML reads the bare key `on` as the boolean True.
    return payload.get("on") or payload.get(True) or {}


# ----------------------------------------------------------------------
# workflow parsing and static validation
# ----------------------------------------------------------------------


def test_workflows_exist_and_parse():
    assert PRODUCTION_WORKFLOW.is_file()
    assert CI_WORKFLOW.is_file()

    for path in (PRODUCTION_WORKFLOW, CI_WORKFLOW):
        assert isinstance(workflow(path), dict)


def test_static_validation_passes():
    assert validator.validate() == []


def test_static_validation_catches_a_missing_checkout_ref(tmp_path, monkeypatch):
    """The validator must actually fail on the mistake it exists to catch."""
    broken = tmp_path / "workflows"
    broken.mkdir()

    for path in (PRODUCTION_WORKFLOW, CI_WORKFLOW):
        payload = path.read_text(encoding="utf-8")

        if path is PRODUCTION_WORKFLOW:
            payload = payload.replace(
                "          ref: ${{ github.event_name == 'schedule'"
                " && 'production/wizardofodds-integration' ||"
                " (inputs.production_ref ||"
                " 'production/wizardofodds-integration') }}\n",
                "",
            )

        (broken / path.name).write_text(payload, encoding="utf-8")

    monkeypatch.setattr(validator, "WORKFLOW_DIR", broken)

    problems = validator.validate()

    assert any("without an explicit ref" in problem for problem in problems)


def test_static_validation_catches_top_of_hour_schedule(tmp_path, monkeypatch):
    broken = tmp_path / "workflows"
    broken.mkdir()

    for path in (PRODUCTION_WORKFLOW, CI_WORKFLOW):
        payload = path.read_text(encoding="utf-8")

        if path is PRODUCTION_WORKFLOW:
            payload = payload.replace("cron: '37 9 * * *'", "cron: '0 9 * * *'")

        (broken / path.name).write_text(payload, encoding="utf-8")

    monkeypatch.setattr(validator, "WORKFLOW_DIR", broken)

    assert any(
        "top of the hour" in problem for problem in validator.validate()
    )


# ----------------------------------------------------------------------
# triggers and default-branch behaviour
# ----------------------------------------------------------------------


def test_production_workflow_has_both_triggers():
    on = triggers(workflow(PRODUCTION_WORKFLOW))

    assert "workflow_dispatch" in on
    assert "schedule" in on


def test_schedule_avoids_the_top_of_the_hour():
    on = triggers(workflow(PRODUCTION_WORKFLOW))

    minutes = [str(entry["cron"]).split()[0] for entry in on["schedule"]]

    for minute in minutes:
        assert minute not in ("0", "00", "*")


def test_scheduled_run_checks_out_the_authoritative_branch():
    """A schedule fires from the default branch, which is not production."""
    text = PRODUCTION_WORKFLOW.read_text(encoding="utf-8")

    assert GITHUB_DEFAULT_BRANCH != AUTHORITATIVE_BRANCH

    job = workflow(PRODUCTION_WORKFLOW)["jobs"]["lifecycle"]

    checkout = next(
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout")
    )

    ref = checkout["with"]["ref"]

    assert AUTHORITATIVE_BRANCH in ref
    assert "schedule" in ref

    # And it proves the checkout rather than trusting it.
    assert "--expected-ref" in text


def test_manual_and_scheduled_share_one_orchestration_path():
    """Both triggers run the same job and the same Python entry points."""
    job = workflow(PRODUCTION_WORKFLOW)["jobs"]["lifecycle"]

    assert len(workflow(PRODUCTION_WORKFLOW)["jobs"]) == 1

    commands = "\n".join(
        str(step.get("run", "")) for step in job["steps"]
    )

    assert "ops/production_lifecycle_preflight.py" in commands
    assert "ops/run_adaptive_daily_fit.py" in commands
    assert "ops/refresh_current_season_state.py" in commands


def test_adaptive_policy_is_not_reimplemented_in_yaml():
    """The workflow orchestrates; the policy decides."""
    text = PRODUCTION_WORKFLOW.read_text(encoding="utf-8")

    for banned in (
        "expanding_time_oof",
        "select_distribution",
        "fit_platt",
        "GaussianCopula",
        "--benchmark-only",
    ):
        assert banned not in text

    # No YAML conditional decides whether to retrain.
    assert "if: ${{ steps" not in text


# ----------------------------------------------------------------------
# concurrency and permissions
# ----------------------------------------------------------------------


def test_single_writer_concurrency():
    payload = workflow(PRODUCTION_WORKFLOW)

    concurrency = payload["concurrency"]

    # A fixed group, so a scheduled and a manual run queue against each other
    # rather than running concurrently.
    assert "${{" not in concurrency["group"]
    assert concurrency["cancel-in-progress"] is False


def test_repository_locks_are_not_replaced_by_github_concurrency():
    """Step 3C's flock remains the real guard; GitHub only supplements it."""
    from nba_prop_quant.adaptive_training import training_lock

    assert callable(training_lock)

    text = PRODUCTION_WORKFLOW.read_text(encoding="utf-8")

    assert "does not replace" in text


def test_least_privilege_permissions():
    for path in (PRODUCTION_WORKFLOW, CI_WORKFLOW):
        assert workflow(path)["permissions"] == {"contents": "read"}


# ----------------------------------------------------------------------
# runner placement
# ----------------------------------------------------------------------


def test_production_job_requires_the_self_hosted_production_runner():
    """Production state only exists on that machine's filesystem."""
    job = workflow(PRODUCTION_WORKFLOW)["jobs"]["lifecycle"]

    labels = validator.runner_labels(job)

    assert "self-hosted" in labels
    assert "nba-production" in labels


def test_pull_request_validation_stays_on_github_hosted_runners():
    """Contributor code must never execute beside the production data root."""
    for job in workflow(CI_WORKFLOW)["jobs"].values():
        assert validator.runner_labels(job) == ["ubuntu-latest"]

    assert "pull_request" not in triggers(workflow(PRODUCTION_WORKFLOW))
    assert "pull_request_target" not in triggers(workflow(PRODUCTION_WORKFLOW))


def test_validator_refuses_pull_request_code_on_the_production_runner():
    problems = validator.untrusted_runner_problems(
        "hypothetical.yml",
        {"pull_request": {"branches": ["main"]}},
        {"test": {"runs-on": ["self-hosted", "nba-production"]}},
    )

    assert any("pull-request code" in problem for problem in problems)


def test_validator_refuses_a_hosted_production_runner(tmp_path, monkeypatch):
    broken = tmp_path / "workflows"
    broken.mkdir()

    for path in (PRODUCTION_WORKFLOW, CI_WORKFLOW):
        payload = path.read_text(encoding="utf-8")

        if path is PRODUCTION_WORKFLOW:
            payload = payload.replace(
                "    runs-on:\n      - self-hosted\n      - nba-production\n",
                "    runs-on: ubuntu-latest\n",
            )

        (broken / path.name).write_text(payload, encoding="utf-8")

    monkeypatch.setattr(validator, "WORKFLOW_DIR", broken)

    assert any(
        "does not require self-hosted" in problem
        for problem in validator.validate()
    )


# ----------------------------------------------------------------------
# frozen-artifact dependency compatibility
# ----------------------------------------------------------------------


def test_production_pins_the_scikit_learn_the_artifacts_were_built_under():
    """A pin, not a floor: CI installed 1.9.1 against 1.9.0 artifacts."""
    required = preflight.frozen_sklearn_version()

    assert required == "1.9.0"

    pyproject = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")

    assert f'"scikit-learn=={required}"' in pyproject


def test_incompatible_sklearn_blocks_production():
    result = preflight.PreflightResult(mode=preflight.MODE_PRODUCTION)

    preflight.check_sklearn_version(result, installed="1.9.1")

    assert not result.checks[0].passed
    assert preflight.BLOCKER_SKLEARN_VERSION in result.blockers
    assert preflight.exit_code_for(result) == (
        preflight.EXIT_INCOMPATIBLE_DEPENDENCY
    )

    # The message has to be actionable, not just a failure.
    assert "scikit-learn==1.9.0" in result.checks[0].detail


def test_absent_sklearn_blocks_production():
    result = preflight.PreflightResult(mode=preflight.MODE_PRODUCTION)

    preflight.check_sklearn_version(result, installed=None)

    assert not result.checks[0].passed
    assert preflight.BLOCKER_SKLEARN_VERSION in result.blockers


def test_matching_sklearn_satisfies_the_check():
    result = preflight.PreflightResult(mode=preflight.MODE_PRODUCTION)

    preflight.check_sklearn_version(result, installed="1.9.0")

    assert result.checks[0].passed


def test_production_mode_checks_sklearn_before_it_touches_state():
    result = preflight.run_preflight(
        preflight.MODE_PRODUCTION,
        expected_ref=head_sha(),
        environ={
            "BDL_API_KEY": "present",
            "NBA_PROP_DATA_DIR": "/srv/nba-prop/data",
            "NBA_PROP_FIT_REGISTRY_DIR": "/srv/nba-prop/fits",
            "NBA_PROP_WORK_DIR": "/srv/nba-prop/work",
        },
    )

    names = [check.name for check in result.checks]

    assert names.index("sklearn_version") < names.index(
        "durable_state_backend"
    )


def test_the_installed_runtime_matches_the_frozen_artifacts():
    assert (
        preflight.installed_sklearn_version()
        == preflight.frozen_sklearn_version()
    )


# ----------------------------------------------------------------------
# secrets
# ----------------------------------------------------------------------


def test_required_secret_names_are_declared_not_invented():
    assert preflight.REQUIRED_PRODUCTION_SECRETS == ("BDL_API_KEY",)

    # The name must come from production code, not from this workflow.
    settings = (PROJECT / "src" / "nba_prop_quant" / "settings.py").read_text(
        encoding="utf-8"
    )

    assert "bdl_api_key" in settings


def test_missing_secret_fails_preflight_before_any_mutation():
    result = preflight.run_preflight(
        preflight.MODE_PRODUCTION,
        expected_ref=head_sha(),
        environ={
            "NBA_PROP_DATA_DIR": "/srv/nba/data",
            "NBA_PROP_FIT_REGISTRY_DIR": "/srv/nba/fits",
            "NBA_PROP_WORK_DIR": "/srv/nba/work",
        },
    )

    assert not result.ok
    assert preflight.exit_code_for(result) == preflight.EXIT_MISSING_SECRET

    names = {check.name for check in result.checks if not check.passed}

    assert names == {"required_secrets"}


def test_preflight_never_returns_a_secret_value():
    result = preflight.run_preflight(
        preflight.MODE_PRODUCTION,
        expected_ref=head_sha(),
        environ={
            "BDL_API_KEY": "super-secret-value",
            "NBA_PROP_DATA_DIR": "/srv/nba/data",
            "NBA_PROP_FIT_REGISTRY_DIR": "/srv/nba/fits",
            "NBA_PROP_WORK_DIR": "/srv/nba/work",
        },
    )

    assert "super-secret-value" not in json.dumps(result.as_dict())
    assert "super-secret-value" not in preflight.render_summary(result, "sha")


def test_no_workflow_writes_a_secret_to_the_log():
    for path in (PRODUCTION_WORKFLOW, CI_WORKFLOW):
        assert validator.secret_leak_problems(path.name) == []


def env_bindings(path: Path, name: str) -> list[str]:
    """Every value the workflow binds to `name`, job level and step level."""
    found = []

    for job in workflow(path)["jobs"].values():
        blocks = [job.get("env") or {}]

        blocks += [step.get("env") or {} for step in job.get("steps") or []]

        for block in blocks:
            if name in block:
                found.append(str(block[name]))

    return found


def test_production_paths_come_from_github_environment_variables():
    """The three durable roots are configuration, not literals in the YAML."""
    text = PRODUCTION_WORKFLOW.read_text(encoding="utf-8")

    for name in preflight.STATE_BACKEND_VARIABLES:
        bindings = env_bindings(PRODUCTION_WORKFLOW, name)

        assert bindings, f"{name} is never bound"

        for binding in bindings:
            assert binding == "${{ vars." + name + " }}", binding

        # And nothing hard-codes a path under the same name.
        assert f"{name}: /" not in text


def test_api_credentials_come_only_from_github_secrets():
    credentials = (
        preflight.REQUIRED_PRODUCTION_SECRETS
        + preflight.OPTIONAL_PRODUCTION_SECRETS
    )

    for name in credentials:
        bindings = env_bindings(PRODUCTION_WORKFLOW, name)

        assert bindings, f"{name} is never bound"

        for binding in bindings:
            assert binding == "${{ secrets." + name + " }}", binding

        # A credential must never be offered as a `vars` value, which is
        # readable by anyone who can read the repository settings.
        assert "vars." + name not in PRODUCTION_WORKFLOW.read_text(
            encoding="utf-8"
        )


def test_production_job_runs_in_the_configured_github_environment():
    """Environment-scoped vars and secrets only resolve inside it."""
    job = workflow(PRODUCTION_WORKFLOW)["jobs"]["lifecycle"]

    assert job["environment"] == "wizardofodds-production"


def test_secrets_are_only_bound_to_environment_variables():
    payload = workflow(PRODUCTION_WORKFLOW)

    text = PRODUCTION_WORKFLOW.read_text(encoding="utf-8")

    for line in text.splitlines():
        if "secrets." in line and not line.strip().startswith("#"):
            # `NAME: ${{ secrets.X }}` under an env: block only.
            assert line.strip().split(":")[0].isupper(), line

    assert payload["jobs"]["lifecycle"]["environment"] == (
        "wizardofodds-production"
    )


# ----------------------------------------------------------------------
# authoritative production ref resolution
# ----------------------------------------------------------------------


def test_authoritative_checkout_accepts_the_expected_ref():
    """Deterministic: CI checks out a PR merge commit, not a branch tip."""
    result = preflight.PreflightResult(mode="validate-only")

    preflight.check_authoritative_checkout(result, head_sha())

    assert result.checks[0].passed, result.checks[0].detail


def test_authoritative_checkout_rejects_a_foreign_ref():
    result = preflight.PreflightResult(mode="validate-only")

    preflight.check_authoritative_checkout(result, "0" * 40)

    assert not result.checks[0].passed
    assert preflight.exit_code_for(result) == (
        preflight.EXIT_NOT_AUTHORITATIVE
    )


def test_frozen_contracts_are_verified_before_production_work():
    result = preflight.run_preflight(
        preflight.MODE_VALIDATE_ONLY, expected_ref=head_sha()
    )

    names = [check.name for check in result.checks]

    assert names.index("authoritative_checkout") < names.index(
        "frozen_contracts"
    )

    assert result.ok


def test_validate_only_does_not_demand_a_production_checkout():
    """A production PR is not yet on production; CI must still report green."""
    result = preflight.run_preflight(preflight.MODE_VALIDATE_ONLY)

    assert [check.name for check in result.checks] == ["frozen_contracts"]
    assert result.ok


def test_production_mode_always_proves_its_checkout():
    result = preflight.run_preflight(
        preflight.MODE_PRODUCTION, environ={}
    )

    assert "authoritative_checkout" in {
        check.name for check in result.checks
    }


# ----------------------------------------------------------------------
# durable state backend
# ----------------------------------------------------------------------


def test_unconfigured_state_backend_blocks_production():
    result = preflight.run_preflight(
        preflight.MODE_PRODUCTION,
        expected_ref=head_sha(),
        environ={"BDL_API_KEY": "present"},
    )

    assert not result.ok
    assert preflight.BLOCKER_DURABLE_STATE in result.blockers
    assert preflight.exit_code_for(result) == (
        preflight.EXIT_NO_STATE_BACKEND
    )


def test_runner_local_state_is_refused(tmp_path):
    """Runner-local paths would silently reset production every night."""
    result = preflight.run_preflight(
        preflight.MODE_PRODUCTION,
        expected_ref=head_sha(),
        environ={
            "BDL_API_KEY": "present",
            "RUNNER_TEMP": str(tmp_path),
            "NBA_PROP_DATA_DIR": str(tmp_path / "data"),
            "NBA_PROP_FIT_REGISTRY_DIR": str(tmp_path / "fits"),
            "NBA_PROP_WORK_DIR": str(tmp_path / "work"),
        },
    )

    assert not result.ok
    assert preflight.BLOCKER_DURABLE_STATE in result.blockers


def test_in_repository_state_is_refused():
    result = preflight.run_preflight(
        preflight.MODE_PRODUCTION,
        expected_ref=head_sha(),
        environ={
            "BDL_API_KEY": "present",
            "NBA_PROP_DATA_DIR": str(PROJECT / "data"),
            "NBA_PROP_FIT_REGISTRY_DIR": str(PROJECT / "fits"),
            "NBA_PROP_WORK_DIR": str(PROJECT / "work"),
        },
    )

    assert not result.ok
    assert preflight.BLOCKER_DURABLE_STATE in result.blockers


def test_durable_backend_satisfies_the_check():
    result = preflight.run_preflight(
        preflight.MODE_PRODUCTION,
        expected_ref=head_sha(),
        environ={
            "BDL_API_KEY": "present",
            "NBA_PROP_DATA_DIR": "/srv/nba-prop/data",
            "NBA_PROP_FIT_REGISTRY_DIR": "/srv/nba-prop/fits",
            "NBA_PROP_WORK_DIR": "/srv/nba-prop/work",
        },
    )

    assert result.ok
    assert result.blockers == []


def test_production_steps_are_gated_behind_the_preflight():
    """Nothing that mutates state may run before the preflight passes."""
    steps = workflow(PRODUCTION_WORKFLOW)["jobs"]["lifecycle"]["steps"]

    names = [step["name"] for step in steps]

    preflight_index = names.index("Preflight")

    for mutating in (
        "Refresh the current-season rolling state",
        "Run the adaptive daily protocol",
    ):
        assert names.index(mutating) > preflight_index

        step = steps[names.index(mutating)]

        assert step["if"] == "env.MODE == 'production'"


# ----------------------------------------------------------------------
# run reporting: no-op, success and failure
# ----------------------------------------------------------------------


def test_no_op_is_reported_as_success():
    status = summariser.build_status(
        {"status": "ok", "blockers": []},
        {
            "outcome": "NO_NEW_TRAINING_DATA",
            "plan": {"slate_date": "2026-11-15", "training_cutoff": "2026-11-14"},
        },
        "f02df30",
        "schedule",
        "production",
    )

    assert status["failure_stage"] is None
    assert status["promoted"] is False
    assert status["candidate_fit_id"] is None

    rendered = summariser.render(status)

    assert "successful outcome" in rendered
    assert "incumbent production fit remains current" in rendered


def test_candidate_success_is_reported_without_promotion():
    status = summariser.build_status(
        {"status": "ok", "blockers": []},
        {
            "outcome": "COMPLETED",
            "fit_id": "nba_prop_quant_fit_20261115_abcdef0123456789",
            "promoted": False,
            "plan": {"slate_date": "2026-11-15", "training_cutoff": "2026-11-14"},
            "validation_checks": {"training_completed": True},
            "deferred_validation_checks": ["t20_protocol_compatible"],
            "benchmark": {"total_seconds": 1234.5},
        },
        "f02df30",
        "workflow_dispatch",
        "production",
    )

    assert status["candidate_fit_id"].startswith("nba_prop_quant_fit_")
    assert status["promoted"] is False
    assert status["failure_stage"] is None


def test_preflight_failure_names_the_failed_stage():
    status = summariser.build_status(
        {
            "status": "failed",
            "blockers": [preflight.BLOCKER_DURABLE_STATE],
        },
        None,
        "f02df30",
        "schedule",
        "production",
    )

    assert status["failure_stage"] == "preflight"
    assert preflight.BLOCKER_DURABLE_STATE in status["blockers"]

    rendered = summariser.render(status)

    assert "Failed at stage" in rendered
    assert preflight.BLOCKER_DURABLE_STATE in rendered


def test_candidate_failure_retains_the_incumbent():
    """A failed adaptive stage must not report a promotion."""
    status = summariser.build_status(
        {"status": "ok", "blockers": []},
        None,
        "f02df30",
        "schedule",
        "production",
    )

    assert status["failure_stage"] == "adaptive_protocol"
    assert status["promoted"] is False
    assert status["current_good_fit_id_after"] is None


def test_summary_reports_every_required_observability_field():
    status = summariser.build_status(
        {"status": "ok", "blockers": []},
        {
            "outcome": "COMPLETED",
            "fit_id": "nba_prop_quant_fit_20261115_abcdef0123456789",
            "promoted": False,
            "plan": {"slate_date": "2026-11-15", "training_cutoff": "2026-11-14"},
            "benchmark": {"total_seconds": 10.0},
        },
        "f02df30",
        "schedule",
        "production",
    )

    for field in (
        "production_code_sha",
        "trigger",
        "slate_date",
        "training_cutoff",
        "adaptive_action",
        "candidate_fit_id",
        "promoted",
        "current_good_fit_id_after",
        "runtime_seconds",
        "failure_stage",
    ):
        assert field in status, field


def test_summary_renders_no_dataframe_dump():
    rendered = summariser.render(
        summariser.build_status(
            {"status": "ok", "blockers": []},
            {"outcome": "NO_NEW_TRAINING_DATA", "plan": {}},
            "f02df30",
            "schedule",
            "production",
        )
    )

    assert len(rendered.splitlines()) < 40


# ----------------------------------------------------------------------
# Step 3C is not regressed
# ----------------------------------------------------------------------


def test_step3d_adds_no_promotion_path():
    """Promotion remains a Step 3C/3D-governed registry operation."""
    for path in (
        PRODUCTION_WORKFLOW,
        PROJECT / "ops" / "production_lifecycle_preflight.py",
        PROJECT / "ops" / "summarise_production_run.py",
        PROJECT / "ops" / "validate_workflows.py",
    ):
        text = path.read_text(encoding="utf-8")

        assert "--rollback" not in text
        assert ".promote(" not in text


def test_step3c_entry_point_is_unchanged():
    from nba_prop_quant.adaptive_training import (
        MODE_REGISTER_CANDIDATE,
        OUTCOME_NO_NEW_TRAINING_DATA,
    )

    assert MODE_REGISTER_CANDIDATE == "register-candidate"
    assert OUTCOME_NO_NEW_TRAINING_DATA == "NO_NEW_TRAINING_DATA"

    assert (PROJECT / "ops" / "run_adaptive_daily_fit.py").is_file()


def test_ci_workflow_reports_on_production_pull_requests():
    on = triggers(workflow(CI_WORKFLOW))

    assert AUTHORITATIVE_BRANCH in on["pull_request"]["branches"]

    steps = workflow(CI_WORKFLOW)["jobs"]["test"]["steps"]

    commands = "\n".join(str(step.get("run", "")) for step in steps)

    assert "pytest" in commands
    assert "validate_workflows.py" in commands
