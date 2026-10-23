"""Tests for re-establishing production readiness after an accepted merge.

The property under test is the one that cost three evenings of forecasts. The
morning lifecycle writes the readiness receipt once; the T-20 controller
refuses to serve unless that receipt was produced by the exact production SHA
it is running; and so any merge to the production branch after the morning run
silently killed the rest of that day's slate.

The acceptance scenario below is the whole sequence, deterministically: a
morning receipt at SHA A, an accepted merge to SHA B, a readiness refresh, and
then the controller's own gate accepting B. The git history is real -- a
throwaway repository built in a temporary directory -- because the refresh is
authorised on ancestry and a fake would not test the thing that matters.

Nothing here trains, fits, serves or publishes, and nothing reaches the
network. The refresh gatekeeper imports no model code, which is also pinned.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]

WORKFLOW = (
    PROJECT / ".github" / "workflows" / "nba_production_readiness_refresh.yml"
)

SLATE = "2026-10-23"

MANIFEST_RELATIVE = "models/frozen_manifests/LATEST.json"

CONTRACT_RELATIVE = (
    "models/frozen_manifests/"
    "nba_prop_quant_v2_adaptive_serving_source_contract.json"
)

PROTOCOL_RELATIVE = (
    "models/frozen_manifests/nba_prop_quant_v2_adaptive_update_protocol.json"
)

#: The two fitted artifacts the synthetic repository pins. The real manifest
#: pins nineteen; the number is not the property under test, the comparison is.
FITTED = {
    "models/pts.joblib": "pts-bytes",
    "models/minutes.joblib": "minutes-bytes",
}

#: One locked serving source file, enough to exercise the governance check.
LOCKED = {"src/nba_prop_quant/slate.py": "def serve():\n    return 1\n"}


def load_ops(name: str):
    spec = importlib.util.spec_from_file_location(
        f"ops_{name}", PROJECT / "ops" / f"{name}.py"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


refresher = load_ops("refresh_production_readiness")

controller = load_ops("run_t20_production_controller")

catchup = load_ops("resolve_t20_catchup_segment")


@pytest.fixture(autouse=True)
def block_all_network(monkeypatch):
    """No test in this module may reach the network."""
    import socket

    def refuse(*_args, **_kwargs):
        raise AssertionError("this test may not reach the network")

    monkeypatch.setattr(socket, "socket", refuse)


# ----------------------------------------------------------------------
# a real, throwaway production history
# ----------------------------------------------------------------------


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(text, encoding="utf-8")


def _governance(root: Path, *, fitted: dict[str, str]) -> None:
    """Write a manifest, a contract that describes the files it locks, and a
    protocol that points at that contract. Written in that order because each
    digest depends on the content written before it."""
    for relative, body in LOCKED.items():
        _write(root, relative, body)

    _write(
        root,
        MANIFEST_RELATIVE,
        json.dumps(
            {
                "files": {
                    "model_artifacts": [
                        {
                            "bytes": len(body),
                            "path": relative,
                            "sha256": hashlib.sha256(
                                body.encode()
                            ).hexdigest(),
                        }
                        for relative, body in sorted(fitted.items())
                    ],
                    "source_files": [],
                },
                "freeze_id": "nba_prop_quant_20260818T205213Z",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    contract = json.dumps(
        {
            "locked_serving_source_files": {
                relative: {
                    "current_sha256": hashlib.sha256(
                        (root / relative).read_bytes()
                    ).hexdigest(),
                    "serving_role": "slate assembly",
                }
                for relative in sorted(LOCKED)
            }
        },
        indent=2,
        sort_keys=True,
    ) + "\n"

    _write(root, CONTRACT_RELATIVE, contract)

    _write(
        root,
        PROTOCOL_RELATIVE,
        json.dumps(
            {
                "adaptive_serving_source_contract_sha256": hashlib.sha256(
                    contract.encode()
                ).hexdigest()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


@pytest.fixture
def history(tmp_path):
    """A production history with an accepted merge: SHA A, then SHA B."""
    root = tmp_path / "repo"

    root.mkdir()

    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")

    _governance(root, fitted=FITTED)

    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "morning production")

    sha_a = _git(root, "rev-parse", "HEAD")

    # The accepted merge: source code moves, fitted artifacts do not.
    _write(root, "ops/anything.py", "# accepted repair\n")

    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "accepted production merge")

    sha_b = _git(root, "rev-parse", "HEAD")

    assert sha_a != sha_b

    return {"root": root, "sha_a": sha_a, "sha_b": sha_b}


@pytest.fixture
def data_root(tmp_path):
    """A durable data root with the directories a receipt names."""
    root = tmp_path / "data"

    for relative in ("bundle/models", "processed/incumbent_serving"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    return root


def write_receipt(
    data_root: Path,
    *,
    production_sha: str,
    slate_date: str = SLATE,
    outcome: str = "NO_GAMES_ON_SLATE",
    refresh_outcome: str = "REFRESHED",
    ready: bool = True,
) -> Path:
    """The receipt the serving entry point durably writes."""
    path = data_root / "processed" / "incumbent_serving" / f"{slate_date}.json"

    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "incumbent_authority": "frozen_deployment_bundle",
                "incumbent_fit_id": None,
                "incumbent_version": "nba_prop_quant_20260818T205213Z",
                "model_authority": "incumbent",
                "outcome": outcome,
                "production_code_sha": production_sha,
                "serving_bundle_root": str(data_root / "bundle"),
                "serving_model_dir": str(data_root / "bundle" / "models"),
                "slate_date": slate_date,
                "slate_readiness": {
                    "ready": ready,
                    "reason": "the current-season rolling state advanced",
                    "refresh_outcome": refresh_outcome,
                },
            }
        ),
        encoding="utf-8",
    )

    return path


def authorise(history, data_root, *, sha: str | None = None):
    return refresher.authorise(
        data_root=data_root,
        slate_date=SLATE,
        production_sha=sha or history["sha_b"],
        project_root=history["root"],
    )


def gate(history, data_root, *, sha: str | None = None):
    return controller.morning_readiness(
        data_root=data_root,
        slate_date=SLATE,
        production_sha=sha or history["sha_b"],
        project_root=history["root"],
    )


# ----------------------------------------------------------------------
# the acceptance scenario
# ----------------------------------------------------------------------


def test_merge_then_refresh_restores_controller_eligibility(
    history, data_root
):
    """Morning receipt at A, accepted merge to B, refresh, controller accepts B.

    The four steps the operational failure needs in order, asserted in order,
    so a regression in any one of them fails here rather than on a game night.
    """
    # 1. the morning lifecycle proved readiness under SHA A.
    write_receipt(data_root, production_sha=history["sha_a"])

    assert gate(history, data_root, sha=history["sha_a"])[
        "production_code_sha"
    ] == history["sha_a"]

    # 2. accepted code is merged. The receipt is now stale, and the controller
    #    refuses -- but it refuses with the outcome that names the remedy.
    with pytest.raises(controller.ControllerRefused) as refusal:
        gate(history, data_root)

    assert (
        refusal.value.outcome == controller.OUTCOME_MORNING_STALE_AFTER_MERGE
    )

    assert controller.READINESS_REFRESH_WORKFLOW in str(refusal.value)

    # 3. the refresh is authorised, and it carries the morning run's own
    #    recorded refresh classification rather than inventing one.
    decision = authorise(history, data_root)

    assert decision["outcome"] == refresher.OUTCOME_AUTHORISED
    assert decision["recorded_production_code_sha"] == history["sha_a"]
    assert decision["production_code_sha"] == history["sha_b"]
    assert decision["fitted_artifacts"]["fitted_artifacts_identical"] is True

    carried = refresher.carried_refresh_status(decision)

    assert carried["outcome"] == "REFRESHED"
    assert carried["slate_date"] == SLATE
    assert carried["carried_forward_from"]["production_code_sha"] == history[
        "sha_a"
    ]

    # 4. the receipt is re-issued under SHA B by the serving entry point, and
    #    the controller's own gate now accepts it.
    write_receipt(data_root, production_sha=history["sha_b"])

    readiness = gate(history, data_root)

    assert readiness["production_code_sha"] == history["sha_b"]
    assert readiness["morning_outcome"] == "NO_GAMES_ON_SLATE"
    assert readiness["refresh_outcome"] == "REFRESHED"


def test_a_receipt_already_at_this_sha_needs_no_refresh(history, data_root):
    write_receipt(data_root, production_sha=history["sha_b"])

    decision = authorise(history, data_root)

    assert decision["outcome"] == refresher.OUTCOME_NOT_REQUIRED


# ----------------------------------------------------------------------
# fail closed
# ----------------------------------------------------------------------


def test_changed_fitted_artifacts_fail_closed(history, data_root):
    """A refreshed receipt may never describe a model nobody verified."""
    root = history["root"]

    _governance(root, fitted={**FITTED, "models/pts.joblib": "refitted"})

    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "a fitted artifact moved")

    refitted = _git(root, "rev-parse", "HEAD")

    write_receipt(data_root, production_sha=history["sha_a"])

    with pytest.raises(refresher.RefreshRefused) as refusal:
        authorise(history, data_root, sha=refitted)

    assert "fitted artifacts are not the ones" in str(refusal.value)
    assert "models/pts.joblib" in str(refusal.value)


def test_a_sideways_sha_is_refused(history, data_root):
    """Only a strict fast-forward is a merge; anything else is other code."""
    root = history["root"]

    _git(root, "checkout", "--quiet", "-b", "sideways", history["sha_a"])

    _write(root, "ops/elsewhere.py", "# not on production\n")

    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "a different line of development")

    sideways = _git(root, "rev-parse", "HEAD")

    # The receipt is from B, and the run is on a commit that is not a
    # descendant of B. Re-issuing here would launder one revision's proof on
    # to another.
    write_receipt(data_root, production_sha=history["sha_b"])

    with pytest.raises(refresher.RefreshRefused) as refusal:
        authorise(history, data_root, sha=sideways)

    assert "is not an ancestor of" in str(refusal.value)


def test_a_broken_governance_contract_is_refused(history, data_root):
    """A refresh is not a route around the serving source byte-pinning."""
    root = history["root"]

    relative = next(iter(LOCKED))

    _write(root, relative, "def serve():\n    return 2  # unlocked edit\n")

    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "edit a locked serving source file")

    unlocked = _git(root, "rev-parse", "HEAD")

    write_receipt(data_root, production_sha=history["sha_a"])

    with pytest.raises(refresher.RefreshRefused) as refusal:
        authorise(history, data_root, sha=unlocked)

    assert "no longer describes the files it locks" in str(refusal.value)


def test_a_morning_run_that_never_proved_readiness_cannot_be_carried(
    history, data_root
):
    write_receipt(
        data_root,
        production_sha=history["sha_a"],
        outcome="NO_PRODUCTION_SLATE",
    )

    with pytest.raises(refresher.RefreshRefused) as refusal:
        authorise(history, data_root)

    assert "did not prove readiness" in str(refusal.value)


def test_an_unservable_refresh_outcome_cannot_be_carried(history, data_root):
    write_receipt(
        data_root,
        production_sha=history["sha_a"],
        refresh_outcome="NO_NEW_DATA",
    )

    with pytest.raises(refresher.RefreshRefused) as refusal:
        authorise(history, data_root)

    assert "not one a slate may be served under" in str(refusal.value)


def test_a_missing_receipt_is_refused_rather_than_invented(
    history, data_root
):
    with pytest.raises(refresher.RefreshRefused) as refusal:
        authorise(history, data_root)

    assert "no proof to carry forward" in str(refusal.value)


def test_the_refresh_gatekeeper_imports_no_model_code():
    """A source SHA moving forward is not evidence about the model."""
    source = (PROJECT / "ops" / "refresh_production_readiness.py").read_text()

    for forbidden in (
        "nba_prop_quant",
        "sklearn",
        "xgboost",
        "joblib",
        "pandas",
        "numpy",
    ):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


# ----------------------------------------------------------------------
# the catch-up segment
# ----------------------------------------------------------------------


def test_the_catchup_segment_starts_now_and_ends_on_the_schedule():
    """A window already past is outside the segment; the boundary is not moved."""
    resolved = catchup.resolve(
        now_utc=datetime(2026, 10, 23, 20, 10, tzinfo=UTC)
    )

    assert resolved["catchup"] is True
    assert resolved["segment_start_utc"] == "2026-10-23T20:10:00+00:00"
    # The 19:25 segment's own end, so the 00:25 segment still owns its windows.
    assert resolved["segment_end_utc"] == "2026-10-24T00:25:00+00:00"
    assert resolved["slate_date"] == SLATE


def test_the_catchup_segment_is_a_no_op_outside_every_segment():
    resolved = catchup.resolve(
        now_utc=datetime(2026, 10, 23, 8, 0, tzinfo=UTC)
    )

    assert resolved["catchup"] is False
    assert "segment_start_utc" not in resolved


def test_catchup_bounds_exclude_a_window_that_already_passed():
    """The reason the start is now: a lost window must not abort the evening."""
    now = datetime(2026, 10, 23, 20, 10, tzinfo=UTC)

    resolved = catchup.resolve(now_utc=now)

    start = datetime.fromisoformat(resolved["segment_start_utc"])
    end = datetime.fromisoformat(resolved["segment_end_utc"])

    missed = datetime(2026, 10, 23, 19, 40, tzinfo=UTC)
    ahead = datetime(2026, 10, 23, 22, 40, tzinfo=UTC)

    assert not start <= missed < end
    assert start <= ahead < end


# ----------------------------------------------------------------------
# the workflow wiring
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def test_the_refresh_is_triggered_by_a_push_to_production(workflow):
    """The whole point: a merge landing must refresh readiness by itself."""
    # PyYAML reads the bare key `on` as True.
    triggers = workflow[True]

    assert "production/wizardofodds-integration" in triggers["push"]["branches"]
    assert "workflow_dispatch" in triggers

    assert "schedule" not in triggers, (
        "the refresh reacts to merges; the once-daily readiness run is the "
        "lifecycle's job and must not be duplicated here"
    )


def test_the_refresh_runs_on_the_durable_production_runner(workflow):
    for name in ("readiness", "t20_catchup"):
        labels = workflow["jobs"][name]["runs-on"]

        assert "self-hosted" in labels
        assert "nba-production" in labels


def test_the_refresh_cannot_race_the_morning_lifecycle(workflow):
    group = workflow["jobs"]["readiness"]["concurrency"]

    assert group["group"] == "nba-production-lifecycle"
    assert group["cancel-in-progress"] is False


def test_the_catchup_cannot_race_a_scheduled_segment(workflow):
    """Concurrency groups are repository-wide, so sharing the name serialises."""
    group = workflow["jobs"]["t20_catchup"]["concurrency"]

    assert group["group"] == "nba-t20-production"
    assert group["cancel-in-progress"] is False


def test_the_catchup_only_runs_after_readiness_succeeds(workflow):
    job = workflow["jobs"]["t20_catchup"]

    assert job["needs"] == "readiness"
    assert "needs.readiness.result == 'success'" in job["if"]


def test_the_refresh_neither_refits_nor_promotes(workflow):
    """A readiness refresh is not a lifecycle run and must not become one.

    Read from what the jobs actually execute rather than from the file text,
    so the comment explaining that none of this is run here does not itself
    trip the check.
    """
    commands = "\n".join(
        step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job["steps"]
    )

    for forbidden in (
        "run_adaptive_daily_protocol",
        "refresh_current_season",
        "promote_",
        "20_fit",
        "21_fit",
    ):
        assert forbidden not in commands, (
            f"{forbidden} belongs to the lifecycle, not to a readiness refresh"
        )


def test_the_refresh_proves_the_receipt_it_wrote(workflow):
    """Exiting zero is not the property; the controller accepting it is."""
    steps = workflow["jobs"]["readiness"]["steps"]

    commands = "\n".join(step.get("run", "") for step in steps)

    assert "ops/refresh_production_readiness.py" in commands
    assert "ops/run_incumbent_production_serving.py" in commands
    assert "ops/assert_production_readiness.py" in commands


def test_the_refresh_publishes_the_receipt_where_the_controller_reads_it(
    workflow,
):
    """The serving entry point writes to a temporary directory.

    In the lifecycle the durable copy is made by the grading step. A refresh
    runs no grading step, so it must make the copy itself or the re-issued
    receipt is invisible to the controller and the refresh is a silent no-op.
    """
    steps = workflow["jobs"]["readiness"]["steps"]

    commands = "\n".join(step.get("run", "") for step in steps)

    assert "processed/incumbent_serving" in commands

    publish = next(
        step
        for step in steps
        if "durable data root" in step.get("name", "")
    )

    # Ordered: published before it is asserted, and asserted at all.
    names = [step.get("name", "") for step in steps]

    assert names.index(publish["name"]) < next(
        index
        for index, step in enumerate(steps)
        if "assert_production_readiness.py" in step.get("run", "")
    )


def test_the_eligibility_assertion_does_no_checking_of_its_own():
    """It must ask the controller's gate, not re-implement it."""
    source = (PROJECT / "ops" / "assert_production_readiness.py").read_text()

    assert "from run_t20_production_controller import" in source

    for invented in ("production_code_sha\"]", "OUTCOMES_THAT_PROVE"):
        assert invented not in source


# ----------------------------------------------------------------------
# the evidence this host holds
# ----------------------------------------------------------------------


def test_every_job_that_lands_on_the_host_inventories_what_it_kept(workflow):
    """Only the machine can answer it, so every job on it asks."""
    steps = workflow["jobs"]["readiness"]["steps"]

    inventory = next(
        step
        for step in steps
        if "ops/inventory_slate_evidence.py" in step.get("run", "")
    )

    # Read-only diagnostics, so a failed refresh still reports the state of
    # the host rather than reporting nothing.
    assert inventory["if"] == "always()"


def test_both_t20_paths_collect_the_durable_evidence_record():
    for path in (
        WORKFLOW,
        PROJECT / ".github" / "workflows" / "nba_t20_production.yml",
    ):
        definition = yaml.safe_load(path.read_text())

        collectors = [
            step
            for job in definition["jobs"].values()
            for step in job["steps"]
            if "ops/collect_t20_evidence.sh" in step.get("run", "")
        ]

        assert len(collectors) == 1, path.name

        # A cancelled or refused segment is exactly when the record matters.
        assert collectors[0]["if"] == "always()", path.name


def test_the_controller_records_the_ledger_before_it_consults_the_gate():
    """The 2026-10-21 and 2026-10-23 evenings recorded nothing at all."""
    source = (
        PROJECT / "ops" / "run_t20_production_controller.py"
    ).read_text()

    opened = source.index("EVENT_SEGMENT_OPENED,")

    gate = source.index("readiness = morning_readiness(")

    assert opened < gate


def test_the_ledger_writes_only_to_the_data_root():
    source = (PROJECT / "ops" / "t20_evidence_ledger.py").read_text()

    assert 'LEDGER_RELATIVE = pathlib.Path("processed") / "t20_evidence"' in (
        source
    )

    # Not RUNNER_TEMP, which is what a cancelled job loses.
    assert "RUNNER_TEMP" not in source
