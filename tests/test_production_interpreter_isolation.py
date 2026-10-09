"""The production run must use its own Python environment, and prove it.

The self-hosted runner's defect was that ``actions/setup-python`` reported a
tool-cache interpreter while the install step resolved to
``/Library/Frameworks/Python.framework/.../site-packages``. The frozen
scikit-learn pin still held, so nothing failed; the environment simply was not
reproducible, and a package installed by hand between runs was
indistinguishable from one the lifecycle installed.

These tests cover the guard that refuses such a run, and the workflow wiring
that gives it something to verify.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]

GUARD = PROJECT / "ops" / "verify_production_interpreter.py"

WORKFLOW = PROJECT / ".github" / "workflows" / "nba_production_lifecycle.yml"


sys.path.insert(0, str(PROJECT / "ops"))

import verify_production_interpreter as guard  # noqa: E402


# ----------------------------------------------------------------------
# the guard, against the interpreter actually running these tests
# ----------------------------------------------------------------------


def test_the_guard_reports_every_field_the_remediation_requires():
    facts = guard.environment_facts()

    for field in (
        "architecture",
        "pip_executable",
        "python_executable",
        "python_version",
        "scikit_learn_version",
        "venv_prefix",
    ):
        assert field in facts, f"{field} is not recorded"

    assert facts["python_executable"] == sys.executable


def test_the_guard_passes_against_the_environment_it_is_run_in():
    """The suite runs in a virtual environment, so this is the positive case."""
    if Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve():
        pytest.skip("this interpreter is not in a virtual environment")

    report = guard.evaluate(Path(sys.prefix))

    assert report.failed == [], report.payload()
    assert report.passed is True


def test_the_guard_fails_when_the_interpreter_is_outside_the_expected_venv(
    tmp_path,
):
    """The defect, in its own terms: a different environment than the named one."""
    report = guard.evaluate(tmp_path / "some-other-venv")

    assert "interpreter_resolves_inside_the_per_run_environment" in report.failed
    assert report.passed is False


def test_the_guard_fails_when_installs_would_land_elsewhere(tmp_path):
    report = guard.evaluate(tmp_path / "some-other-venv")

    assert "installs_target_the_per_run_environment" in report.failed


def test_the_guard_fails_on_a_machine_global_site_packages(monkeypatch):
    """A site-packages under the macOS framework prefix is the audit's finding."""
    monkeypatch.setattr(
        guard,
        "site_package_paths",
        lambda: [
            "/Library/Frameworks/Python.framework/Versions/3.12/lib/"
            "python3.12/site-packages"
        ],
    )

    report = guard.evaluate(Path(sys.prefix))

    assert "no_machine_global_site_packages_are_importable" in report.failed

    offending = next(
        check
        for check in report.checks
        if check.name == "no_machine_global_site_packages_are_importable"
    )

    assert offending.values["offenders"]


def test_a_global_package_directory_is_found_under_either_spelling(monkeypatch):
    """``dist-packages`` is the same defect as ``site-packages``.

    The production runner is a Mac and says ``site-packages``, so keying on
    that one name passed every test written here while leaving a differently
    packaged runner's global directory invisible to the check -- it would not
    appear in the path list at all, and absence reads as cleanliness.
    """
    monkeypatch.setattr(
        guard,
        "site_package_paths",
        lambda: ["/usr/lib/python3/dist-packages"],
    )

    report = guard.evaluate(Path(sys.prefix))

    assert "no_machine_global_site_packages_are_importable" in report.failed


def test_both_package_directory_spellings_are_collected(monkeypatch):
    monkeypatch.setattr(
        guard.sys,
        "path",
        [
            "",
            "/tmp/venv/lib/python3.12/site-packages",
            "/usr/lib/python3/dist-packages",
            "/workspace/src",
        ],
    )

    assert guard.site_package_paths() == [
        "/tmp/venv/lib/python3.12/site-packages",
        "/usr/lib/python3/dist-packages",
    ]


def test_the_guard_fails_on_a_translated_interpreter(monkeypatch):
    """An x86_64 Python on an arm64 host is a different numerical stack."""
    monkeypatch.setattr(guard.platform, "machine", lambda: "x86_64")

    report = guard.evaluate(Path(sys.prefix), expected_architecture="arm64")

    assert "interpreter_architecture_matches_the_runner" in report.failed


def test_the_guard_reads_the_sklearn_pin_rather_than_restating_it():
    """One place decides which scikit-learn production serves under."""
    from production_lifecycle_preflight import frozen_sklearn_version

    assert guard._frozen_sklearn() == frozen_sklearn_version(PROJECT)
    assert guard._frozen_sklearn()


def test_the_guard_fails_on_a_foreign_sklearn(monkeypatch):
    monkeypatch.setattr(guard, "_installed_sklearn", lambda: "0.0.0")

    report = guard.evaluate(Path(sys.prefix))

    assert "scikit_learn_matches_the_frozen_pin" in report.failed


def test_the_guard_requires_the_frozen_python_version():
    assert guard.REQUIRED_PYTHON == (3, 12)


# ----------------------------------------------------------------------
# the guard as the workflow invokes it
# ----------------------------------------------------------------------


def run_guard(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GUARD), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT),
    )


def test_the_cli_exits_zero_and_writes_a_receipt(tmp_path):
    if Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve():
        pytest.skip("this interpreter is not in a virtual environment")

    status = tmp_path / "interpreter.json"

    completed = run_guard(
        "--expected-venv",
        sys.prefix,
        "--status-path",
        str(status),
        "--production-sha",
        "a" * 40,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr

    payload = json.loads(status.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["production_sha"] == "a" * 40
    assert payload["environment"]["python_executable"] == sys.executable


def test_the_cli_fails_the_workflow_on_the_wrong_environment(tmp_path):
    completed = run_guard("--expected-venv", str(tmp_path / "nope"))

    assert completed.returncode == 1


def test_the_cli_refuses_to_verify_nothing():
    completed = run_guard("--expected-venv", "   ")

    assert completed.returncode == 2


# ----------------------------------------------------------------------
# the workflow wiring
# ----------------------------------------------------------------------


def lifecycle_steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    return workflow["jobs"]["lifecycle"]["steps"]


def step_index(predicate) -> int:
    for index, step in enumerate(lifecycle_steps()):
        if predicate(step):
            return index

    raise AssertionError("no matching step")


def test_the_lifecycle_creates_a_per_run_environment():
    steps = lifecycle_steps()

    creating = [
        step for step in steps if "-m venv" in str(step.get("run", ""))
    ]

    assert creating, "no lifecycle step creates a virtual environment"

    for step in creating:
        run = str(step["run"])
        assert "$RUNNER_TEMP" in run, (
            "the environment must live under RUNNER_TEMP so it is discarded "
            "with the run rather than accumulating on the runner"
        )
        assert "$GITHUB_PATH" in run, (
            "the environment must go on PATH, which is what makes every later "
            "python and pip in the job resolve to it"
        )


def test_the_environment_is_created_before_anything_is_installed():
    venv = step_index(lambda step: "-m venv" in str(step.get("run", "")))

    install = step_index(
        lambda step: "pip install -e" in str(step.get("run", ""))
    )

    assert venv < install, (
        "GITHUB_PATH takes effect in later steps only, so the environment "
        "must exist before the install or the install goes somewhere else"
    )


def test_the_interpreter_is_verified_after_the_install():
    install = step_index(
        lambda step: "pip install -e" in str(step.get("run", ""))
    )

    verify = step_index(
        lambda step: "verify_production_interpreter.py" in str(step.get("run", ""))
    )

    assert install < verify, (
        "the question is where the dependencies actually landed, not where "
        "they were meant to"
    )


def test_the_interpreter_check_can_fail_the_production_job():
    """An isolation guard that cannot stop the run has not guarded anything."""
    steps = lifecycle_steps()

    verifying = [
        step
        for step in steps
        if "verify_production_interpreter.py" in str(step.get("run", ""))
    ]

    assert verifying

    for step in verifying:
        assert step.get("continue-on-error") is not True


def test_the_interpreter_check_runs_before_any_production_state_is_touched():
    verify = step_index(
        lambda step: "verify_production_interpreter.py" in str(step.get("run", ""))
    )

    for name in (
        "refresh_current_season_state.py",
        "run_adaptive_daily_fit.py",
    ):
        mutating = step_index(lambda step, n=name: n in str(step.get("run", "")))

        assert verify < mutating, (
            f"{name} must not run before the interpreter has been verified"
        )
