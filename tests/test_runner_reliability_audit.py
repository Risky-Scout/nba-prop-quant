"""Tests for the self-hosted runner reliability criterion.

The thing worth pinning is that this cannot be talked into saying PASS. A
runner that answered eventually, a runner that answered promptly five times, a
runner that answered promptly seven times but dropped its connection once --
none of those is a reliable runner, and each has to come back as something
other than PASS_PROVEN.

Nothing here contacts GitHub. The observations are the shape the API returns,
including the real audited window, which is committed as a regression case
precisely because it is the window that must not be called healthy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

AUDITOR = REPO / "ops" / "audit_runner_reliability.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "audit_runner_reliability", AUDITOR
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


audit = _load()


def run_entry(
    index: int,
    *,
    wait_seconds: int = 3,
    annotations: tuple[str, ...] = (),
    conclusion: str = "success",
) -> dict:
    created = datetime(2026, 11, 10 + index, 9, 37, tzinfo=UTC)

    started = created + timedelta(seconds=wait_seconds)

    return {
        "run_id": 1000 + index,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conclusion": conclusion,
        "runner_name": "Josephs-MacBook-Pro",
        "annotations": list(annotations),
    }


def window(count: int, **overrides) -> list[dict]:
    return [run_entry(index, **overrides) for index in range(count)]


# ----------------------------------------------------------------------
# the only way to PASS
# ----------------------------------------------------------------------


def test_seven_consecutive_clean_runs_is_the_only_pass():
    assessment = audit.assess(audit.load_window(window(7)))

    assert assessment["status"] == audit.STATUS_PASS
    assert assessment["runs_observed"] == audit.REQUIRED_CONSECUTIVE_RUNS


def test_six_clean_runs_is_not_yet_a_pass():
    """One short is not nearly enough, and must not round up."""
    assessment = audit.assess(audit.load_window(window(6)))

    assert assessment["status"] == audit.STATUS_PENDING
    assert "6 of the required 7" in assessment["detail"]


def test_the_window_is_the_most_recent_runs_not_the_best_ones():
    """Otherwise an old clean stretch would excuse a current bad one."""
    entries = window(7)

    entries.append(run_entry(7, wait_seconds=3600))

    assessment = audit.assess(audit.load_window(entries))

    assert assessment["status"] == audit.STATUS_FAIL
    assert assessment["runs_that_started_slowly"] == [1007]


# ----------------------------------------------------------------------
# each disqualifying shape, on its own
# ----------------------------------------------------------------------


def test_one_slow_start_in_the_window_fails():
    entries = window(7)

    entries[3] = run_entry(3, wait_seconds=400)

    assessment = audit.assess(audit.load_window(entries))

    assert assessment["status"] == audit.STATUS_FAIL
    assert assessment["runs_that_started_slowly"] == [1003]


def test_a_run_exactly_at_the_limit_still_counts_as_prompt():
    """The boundary is stated in the brief as "< 5 min"; five minutes
    exactly is treated as meeting it rather than failing it, so the
    classification does not hinge on a rounding of the API's timestamps."""
    entries = window(7)

    entries[0] = run_entry(0, wait_seconds=300)

    assert audit.assess(audit.load_window(entries))["status"] == audit.STATUS_PASS


def test_one_lost_communication_annotation_fails():
    """The run that lost communication may still have concluded success."""
    entries = window(7)

    entries[5] = run_entry(
        5,
        annotations=(
            "The self-hosted runner lost communication with the server. "
            "Verify the machine is running and has a healthy network "
            "connection.",
        ),
    )

    assessment = audit.assess(audit.load_window(entries))

    assert assessment["status"] == audit.STATUS_FAIL
    assert assessment["runs_that_lost_communication"] == [1005]


def test_a_run_that_found_no_runner_fails():
    entries = window(7)

    entries[2] = run_entry(
        2, annotations=("No runner matched the labels for this job",)
    )

    assessment = audit.assess(audit.load_window(entries))

    assert assessment["status"] == audit.STATUS_FAIL
    assert assessment["runs_with_no_runner"] == [1002]


def test_a_run_that_never_started_is_not_prompt():
    entries = window(7)

    entries[1] = run_entry(1)

    entries[1]["started_at"] = None

    assessment = audit.assess(audit.load_window(entries))

    assert assessment["status"] == audit.STATUS_FAIL
    assert assessment["runs_that_started_slowly"] == [1001]


def test_a_successful_conclusion_does_not_buy_health():
    """The whole point. The five-hour run concluded success.

    A column that reads "success" is why this went unnoticed: the job did
    finish, hours after the slate it was scheduled for.
    """
    entries = window(7)

    entries[6] = run_entry(6, wait_seconds=20_822, conclusion="success")

    assessment = audit.assess(audit.load_window(entries))

    assert assessment["status"] == audit.STATUS_FAIL
    assert assessment["longest_wait_seconds"] == 20_822


# ----------------------------------------------------------------------
# the real audited window
# ----------------------------------------------------------------------


AUDITED_WINDOW = [
    {
        "run_id": 37113947924,
        "created_at": "2026-10-03T09:42:19Z",
        "started_at": "2026-10-03T10:02:09Z",
        "conclusion": "failure",
        "runner_name": "Josephs-MacBook-Pro",
        "annotations": [],
    },
    {
        "run_id": 37197716819,
        "created_at": "2026-10-04T11:08:31Z",
        "started_at": "2026-10-04T11:36:58Z",
        "conclusion": "failure",
        "runner_name": "Josephs-MacBook-Pro",
        "annotations": [],
    },
    {
        "run_id": 37235586601,
        "created_at": "2026-10-04T21:18:50Z",
        "started_at": "2026-10-04T21:18:53Z",
        "conclusion": "success",
        "runner_name": "Josephs-MacBook-Pro",
        "annotations": [],
    },
    {
        "run_id": 37293057379,
        "created_at": "2026-10-05T09:54:01Z",
        "started_at": "2026-10-05T09:54:05Z",
        "conclusion": "success",
        "runner_name": "Josephs-MacBook-Pro",
        "annotations": [],
    },
    {
        "run_id": 37445390061,
        "created_at": "2026-10-06T09:49:20Z",
        "started_at": "2026-10-06T10:17:02Z",
        "conclusion": "failure",
        "runner_name": "Josephs-MacBook-Pro",
        "annotations": [
            "The self-hosted runner lost communication with the server."
        ],
    },
    {
        "run_id": 37603138394,
        "created_at": "2026-10-07T09:48:31Z",
        "started_at": "2026-10-07T10:20:13Z",
        "conclusion": "failure",
        "runner_name": "Josephs-MacBook-Pro",
        "annotations": [
            "The self-hosted runner lost communication with the server."
        ],
    },
    {
        "run_id": 37759297910,
        "created_at": "2026-10-08T09:49:20Z",
        "started_at": "2026-10-08T15:36:22Z",
        "conclusion": "success",
        "runner_name": "Josephs-MacBook-Pro",
        "annotations": [],
    },
]


def test_the_audited_window_is_not_a_healthy_runner():
    """The regression case: this exact window must never read PASS."""
    assessment = audit.assess(audit.load_window(AUDITED_WINDOW))

    assert assessment["status"] == audit.STATUS_FAIL
    assert len(assessment["runs_that_started_slowly"]) == 5
    assert len(assessment["runs_that_lost_communication"]) == 2

    # 5h47m02s, the longest wait on record.
    assert assessment["longest_wait_seconds"] == 20_822


# ----------------------------------------------------------------------
# input shapes and the exit code
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    (
        AUDITED_WINDOW,
        {"runs": AUDITED_WINDOW},
        {"workflow_runs": AUDITED_WINDOW},
    ),
)
def test_every_collection_shape_is_accepted(payload):
    assert audit.assess(audit.load_window(payload))["runs_observed"] == 7


def test_the_run_level_start_is_accepted_as_a_fallback():
    """GitHub exposes both; the job-level one is the real answer.

    The run-level ``run_started_at`` is weaker evidence because it is not the
    moment the job reached the runner, but refusing it would mean a collection
    that only has it produces no assessment at all.
    """
    entries = [
        {
            "id": 7,
            "created_at": "2026-11-10T09:37:00Z",
            "run_started_at": "2026-11-10T09:37:02Z",
            "conclusion": "success",
        }
    ]

    observations = audit.load_window(entries)

    assert observations[0].run_id == 7
    assert observations[0].wait == timedelta(seconds=2)


def test_the_cli_exits_nonzero_until_reliability_is_proven(tmp_path: Path):
    path = tmp_path / "observations.json"

    path.write_text(json.dumps(AUDITED_WINDOW), encoding="utf-8")

    report_path = tmp_path / "report.json"

    assert (
        audit.main(
            ["--observations", str(path), "--report-path", str(report_path)]
        )
        == audit.EXIT_NOT_PROVEN
    )

    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == (
        audit.STATUS_FAIL
    )


def test_the_cli_exits_zero_on_a_proven_window(tmp_path: Path):
    path = tmp_path / "observations.json"

    path.write_text(json.dumps(window(7)), encoding="utf-8")

    assert audit.main(["--observations", str(path)]) == audit.EXIT_OK


def test_the_criterion_matches_the_brief():
    """Stated once, as constants, so a reader can check them against it."""
    assert audit.REQUIRED_CONSECUTIVE_RUNS == 7
    assert audit.MAXIMUM_QUEUE_WAIT == timedelta(minutes=5)


def test_the_auditor_cannot_change_the_runner():
    """It observes. It has no way to administer anything."""
    source = AUDITOR.read_text(encoding="utf-8")

    for forbidden in ("subprocess.run", "pmset", "svc.sh", "requests."):
        assert forbidden not in source
