"""An unhealthy shadow must stop looking green.

The shadow step is ``continue-on-error: true`` and has to stay that way: a
candidate that can fail the production lifecycle is worse than no candidate.
The cost was that GitHub reported success whether the shadow worked or not. A
run whose candidate raised on every game, refused to assemble a covariance, or
wrote nothing at all was indistinguishable from a clean one.

These tests cover the health evaluation that closes that gap. The four
negative controls the remediation requires are here by name: a forced candidate
exception, a forced PSD refusal, a missing status file after an expected run,
and a healthy shadow. In all of them the incumbent stays the served authority
and nothing is published -- failing the job is the signal, not a change to what
production served.

Thresholds are never written here. Every gate is read from the frozen live
policy, and a test pins that the evaluator reads it rather than restating it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]

EVALUATOR = PROJECT / "ops" / "evaluate_shadow_health.py"

WORKFLOW = PROJECT / ".github" / "workflows" / "nba_production_lifecycle.yml"

POLICY = PROJECT / "research" / "final_model" / "live_shadow_promotion_policy.json"


sys.path.insert(0, str(PROJECT / "ops"))

import evaluate_shadow_health as health  # noqa: E402


def policy() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# synthetic shadow statuses
# ----------------------------------------------------------------------


def healthy_status(**overrides) -> dict:
    """What the shadow entry point writes after a clean slate."""
    status = {
        "entry_point": "ops/run_production_shadow.py",
        "mode": "SHADOW_ONLY",
        "started_at": "2026-11-15T09:40:00+00:00",
        "finished_at": "2026-11-15T09:52:00+00:00",
        "slate_date": "2026-11-15",
        "outcome": "SHADOWED",
        "served_authority": "incumbent",
        "rows_published": 0,
        "production_serving_was_affected": False,
        "authority": {
            "promotion_attempt": "ShadowPromotionRefused",
            "publish_attempt": "ShadowPublishingDisabled",
            "published_authority": "incumbent",
            "rows_published": 0,
        },
        "shadow": {
            "games_offered": 8,
            "games_shadowed": 8,
            "games_skipped_for_too_few_players_or_events": 0,
            "games_that_fell_back": 0,
            "fallbacks": [],
            "events_shadowed": 96,
            "rows_written": 192,
            "rows_published": 0,
            "psd_failures": 0,
            "psd_eigenvalue_floor": 1e-10,
            "min_covariance_eigenvalue": 0.004,
            "same_player_max_block_deviation": 0.0,
            "provenance_fingerprint": "f" * 64,
            "provenance": {
                "final_model_spec_sha256": "1" * 64,
                "factor_spec_hash": "3" * 64,
                "marginal_source_sha256": "a" * 64,
                "copula_source_sha256": "b" * 64,
                "code_sha": "c" * 40,
            },
        },
    }

    status.update(overrides)

    return status


def write(path: Path, status: dict) -> Path:
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")

    return path


def run_evaluator(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EVALUATOR), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT),
    )


# ----------------------------------------------------------------------
# thresholds come from the frozen policy, not from this code
# ----------------------------------------------------------------------


def test_the_gates_are_read_from_the_frozen_policy():
    gates = policy()["operational_gates"]

    assert gates["O1_shadow_publishing_violations"]["max"] == 0
    assert gates["O2_psd_failures"]["max"] == 0
    assert gates["O3_same_player_block_deviation"]["max"] == 1e-9
    assert gates["O4_candidate_fallback_rate"]["max"] == 0.005


def test_the_evaluator_names_the_policy_it_reads():
    assert health.POLICY_RELATIVE_PATH == Path(
        "research/final_model/live_shadow_promotion_policy.json"
    )
    assert (PROJECT / health.POLICY_RELATIVE_PATH).is_file()


def test_the_evaluator_states_no_threshold_of_its_own():
    """A number here would be a second opinion about a frozen gate."""
    source = EVALUATOR.read_text(encoding="utf-8")

    for literal in ("0.005", "1e-09", "1e-9"):
        assert literal not in source, (
            f"{literal} is a frozen policy threshold and must be read, not "
            "written here"
        )


def test_every_gate_the_evaluator_cites_exists_in_the_policy():
    report = health.evaluate(healthy_status(), policy())

    gates = set(policy()["operational_gates"])

    cited = {finding.gate for finding in report.findings if finding.gate}

    assert cited
    assert cited <= gates, f"unknown gates cited: {sorted(cited - gates)}"


# ----------------------------------------------------------------------
# negative control 1: a healthy shadow is healthy
# ----------------------------------------------------------------------


def test_a_healthy_shadow_reports_success(tmp_path):
    report = health.evaluate(healthy_status(), policy())

    assert report.failures == [], report.payload()
    assert report.healthy is True

    completed = run_evaluator(
        "--status-path",
        str(write(tmp_path / "shadow.json", healthy_status())),
        "--shadow-was-expected",
        "--health-path",
        str(tmp_path / "health.json"),
    )

    assert completed.returncode == 0, completed.stdout

    payload = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))

    assert payload["healthy"] is True


# ----------------------------------------------------------------------
# negative control 2: a candidate that raised
# ----------------------------------------------------------------------


def test_a_candidate_exception_is_reported_as_unhealthy(tmp_path):
    """The shadow catches its own failures and exits 0. This must not."""
    broken = healthy_status(
        outcome="SHADOW_FAILED",
        error="LinAlgError: candidate factor assembly failed",
        traceback="Traceback (most recent call last): ...",
    )
    broken.pop("shadow")

    report = health.evaluate(broken, policy())

    assert "shadow_recorded_no_error" in report.failures
    assert report.healthy is False

    completed = run_evaluator(
        "--status-path",
        str(write(tmp_path / "shadow.json", broken)),
        "--shadow-was-expected",
    )

    assert completed.returncode == 1


def test_a_candidate_exception_leaves_the_incumbent_unaffected(tmp_path):
    """The whole point of the non-blocking design is preserved."""
    broken = healthy_status(
        outcome="SHADOW_FAILED",
        error="LinAlgError: candidate factor assembly failed",
    )
    broken.pop("shadow")

    report = health.evaluate(broken, policy())

    by_name = {finding.name: finding for finding in report.findings}

    assert by_name["served_model_remained_the_incumbent"].status == "PASS"
    assert by_name["production_serving_was_not_affected"].status == "PASS"
    assert by_name["nothing_was_published"].status == "PASS"

    # And the evaluator writes nothing except its own report.
    health_path = tmp_path / "health.json"

    run_evaluator(
        "--status-path",
        str(write(tmp_path / "shadow.json", broken)),
        "--shadow-was-expected",
        "--health-path",
        str(health_path),
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "health.json",
        "shadow.json",
    ]


# ----------------------------------------------------------------------
# negative control 3: a PSD refusal
# ----------------------------------------------------------------------


def test_a_psd_refusal_is_reported_as_unhealthy(tmp_path):
    status = healthy_status()
    status["shadow"].update(
        {
            "psd_failures": 2,
            "min_covariance_eigenvalue": -3.0e-07,
            "games_that_fell_back": 2,
            "fallbacks": [
                {
                    "game_id": 1,
                    "failure_reason": "covariance is not positive semi-definite",
                    "was_a_psd_refusal": True,
                },
                {
                    "game_id": 2,
                    "failure_reason": "covariance is not positive semi-definite",
                    "was_a_psd_refusal": True,
                },
            ],
            "outcome": "SHADOWED_WITH_FALLBACKS",
        }
    )
    status["outcome"] = "SHADOWED_WITH_FALLBACKS"

    report = health.evaluate(status, policy())

    assert "no_psd_failures" in report.failures
    assert "minimum_covariance_eigenvalue_is_above_the_floor" in report.failures

    # The policy asks for review on a repeated root cause even inside O4, so
    # that is a warning beside the failures rather than silence.
    assert "no_repeated_fallback_root_cause" in report.warnings

    assert run_evaluator(
        "--status-path",
        str(write(tmp_path / "shadow.json", status)),
        "--shadow-was-expected",
    ).returncode == 1


def test_a_psd_refusal_leaves_the_incumbent_unaffected():
    status = healthy_status()
    status["shadow"]["psd_failures"] = 1

    report = health.evaluate(status, policy())

    by_name = {finding.name: finding for finding in report.findings}

    assert by_name["served_model_remained_the_incumbent"].status == "PASS"
    assert by_name["nothing_was_published"].status == "PASS"


# ----------------------------------------------------------------------
# negative control 4: no status file at all
# ----------------------------------------------------------------------


def test_a_missing_status_after_an_expected_run_is_reported(tmp_path):
    """The case a step conclusion could never distinguish from success."""
    completed = run_evaluator(
        "--status-path",
        str(tmp_path / "absent.json"),
        "--shadow-was-expected",
        "--health-path",
        str(tmp_path / "health.json"),
    )

    assert completed.returncode == 2

    payload = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))

    assert payload["status_present"] is False
    assert payload["healthy"] is False
    assert payload["failures"] == ["shadow_status_is_missing"]


def test_a_missing_status_when_no_shadow_was_expected_is_not_a_failure(tmp_path):
    completed = run_evaluator("--status-path", str(tmp_path / "absent.json"))

    assert completed.returncode == 0


# ----------------------------------------------------------------------
# the remaining required fields
# ----------------------------------------------------------------------


def test_a_published_row_is_reported_as_unhealthy():
    report = health.evaluate(healthy_status(rows_published=1), policy())

    assert "nothing_was_published" in report.failures


def test_a_served_candidate_is_reported_as_unhealthy():
    report = health.evaluate(
        healthy_status(served_authority="candidate"), policy()
    )

    assert "served_model_remained_the_incumbent" in report.failures


def test_affected_production_serving_is_reported_as_unhealthy():
    report = health.evaluate(
        healthy_status(production_serving_was_affected=True), policy()
    )

    assert "production_serving_was_not_affected" in report.failures


def test_a_block_deviation_above_the_gate_is_reported_as_unhealthy():
    status = healthy_status()
    status["shadow"]["same_player_max_block_deviation"] = 1e-6

    report = health.evaluate(status, policy())

    assert "same_player_block_deviation_is_within_the_gate" in report.failures


def test_a_fallback_rate_above_the_gate_is_reported_as_unhealthy():
    status = healthy_status()
    status["shadow"].update(
        {
            "games_shadowed": 10,
            "games_that_fell_back": 1,
            "fallbacks": [{"game_id": 1, "failure_reason": "no minutes"}],
        }
    )

    report = health.evaluate(status, policy())

    assert "candidate_fallback_rate_is_within_the_gate" in report.failures


def test_incomplete_provenance_is_reported_as_unhealthy():
    status = healthy_status()
    status["shadow"]["provenance"].pop("code_sha")

    report = health.evaluate(status, policy())

    assert "provenance_is_complete" in report.failures

    finding = next(
        f for f in report.findings if f.name == "provenance_is_complete"
    )

    assert finding.values["unresolved"] == ["production_code_sha"]


def test_a_shadow_that_never_started_is_reported_as_unhealthy():
    status = healthy_status()
    status.pop("started_at")
    status.pop("entry_point")

    report = health.evaluate(status, policy())

    assert "shadow_actually_started" in report.failures


def test_a_shadow_that_never_finished_is_reported_as_unhealthy():
    status = healthy_status()
    status.pop("finished_at")

    report = health.evaluate(status, policy())

    assert "shadow_reached_its_own_end" in report.failures


def test_a_refused_publish_is_required():
    status = healthy_status()
    status["authority"]["publish_attempt"] = "DID NOT RAISE"

    report = health.evaluate(status, policy())

    assert "the_shadow_refused_to_publish" in report.failures


def test_a_refused_promotion_is_required():
    status = healthy_status()
    status["authority"]["promotion_attempt"] = "DID NOT RAISE"

    report = health.evaluate(status, policy())

    assert "the_shadow_refused_to_promote" in report.failures


@pytest.mark.parametrize(
    "outcome",
    sorted(health.NOTHING_TO_GRADE_OUTCOMES),
)
def test_a_slate_with_nothing_to_grade_is_healthy(outcome):
    """Before a settled slate exists this is the normal state, not a failure."""
    status = healthy_status(outcome=outcome)
    status.pop("shadow")

    report = health.evaluate(status, policy())

    assert report.failures == [], report.payload()

    numerical = next(
        f for f in report.findings if f.name == "numerical_gates"
    )

    assert numerical.status == "NOT_APPLICABLE"


def test_a_missing_shadow_block_on_a_graded_outcome_is_unhealthy():
    status = healthy_status(outcome="SHADOWED")
    status.pop("shadow")

    report = health.evaluate(status, policy())

    assert "numerical_gates" in report.failures


def test_an_unrecognised_outcome_is_reported_as_unhealthy():
    report = health.evaluate(healthy_status(outcome="SOMETHING_NEW"), policy())

    assert "shadow_outcome_is_recognised" in report.failures


# ----------------------------------------------------------------------
# the workflow wiring
# ----------------------------------------------------------------------


def lifecycle_steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    return workflow["jobs"]["lifecycle"]["steps"]


def health_steps() -> list[dict]:
    return [
        step
        for step in lifecycle_steps()
        if "evaluate_shadow_health.py" in str(step.get("run", ""))
    ]


def test_the_lifecycle_asserts_shadow_health():
    assert health_steps(), "no lifecycle step evaluates shadow health"


def test_the_shadow_step_is_still_non_blocking():
    """The health check is additional to the safety design, not a change to it."""
    shadow = [
        step
        for step in lifecycle_steps()
        if "run_production_shadow.py" in str(step.get("run", ""))
    ]

    assert shadow

    for step in shadow:
        assert step.get("continue-on-error") is True
        assert "--strict" not in str(step["run"])


def test_the_health_step_can_fail_the_job():
    """A health assertion that cannot go red asserts nothing."""
    for step in health_steps():
        assert step.get("continue-on-error") is not True


def test_the_health_step_runs_even_when_the_shadow_step_did_not_finish():
    """always(), so a missing status file is reported rather than skipped."""
    for step in health_steps():
        condition = str(step.get("if", ""))
        assert "always()" in condition
        assert "RUN_ADAPTIVE" in condition, (
            "health is only asserted where the shadow was expected to run"
        )


def test_the_health_step_reads_the_persisted_status_not_the_step_conclusion():
    for step in health_steps():
        run = str(step["run"])
        assert "--status-path" in run
        assert "shadow.json" in run
        assert "--shadow-was-expected" in run


def test_the_health_step_runs_after_the_shadow():
    steps = lifecycle_steps()

    def index(needle: str) -> int:
        for position, step in enumerate(steps):
            if needle in str(step.get("run", "")):
                return position

        raise AssertionError(needle)

    assert index("run_production_shadow.py") < index("evaluate_shadow_health.py")


def test_the_health_step_activates_no_publishing():
    source = EVALUATOR.read_text(encoding="utf-8")

    for forbidden in ("publish_shadow_probabilities", "wizardofodds"):
        assert forbidden not in source.lower().replace(
            "research/final_model/live_shadow_promotion_policy.json", ""
        )
