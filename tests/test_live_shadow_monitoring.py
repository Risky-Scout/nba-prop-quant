"""The daily monitoring step: accumulation, idempotence and the decision.

The frozen policy's evaluator is tested on its own in
``test_live_promotion_policy_evaluator.py``. What is tested here is the wiring
around it: that a day's shadow artifacts become durable accumulated evidence,
that accumulating the same day twice does not say the evidence doubled, that
absent evidence reaches the policy as absent rather than as a zero, and that
the one decision which must page an operator is the only one that fails the
lifecycle.

The synthetic evidence generators are imported from the evaluator's own test
module rather than rewritten. They are the shapes the gates were pinned
against, and a second set here could drift into a window the policy would
read differently.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
ENTRY_POINT = PROJECT / "ops" / "monitor_live_shadow_evidence.py"
POLICY_PATH = PROJECT / "research" / "final_model" / "live_shadow_promotion_policy.json"
LIFECYCLE = PROJECT / ".github" / "workflows" / "nba_production_lifecycle.yml"

CONTINUE = "CONTINUE_SHADOW"
ELIGIBLE = "PROMOTION_REVIEW_ELIGIBLE"
REJECTED = "SHADOW_MODEL_REJECTED"
DISABLED = "SHADOW_DISABLED_FOR_SAFETY"

FINGERPRINT = "c0ffee" * 10


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a module that defines dataclasses needs to
    # find itself in sys.modules while its class bodies run.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def monitor():
    return _load("monitor_live_shadow_evidence", ENTRY_POINT)


@pytest.fixture(scope="module")
def evidence():
    """The evaluator's own synthetic generators."""
    return _load(
        "live_policy_evidence",
        PROJECT / "tests" / "test_live_promotion_policy_evaluator.py",
    )


@pytest.fixture(scope="module")
def policy() -> dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


# ======================================================================
# the day's shadow artifacts, in the shapes the shadow actually writes
# ======================================================================


def provenance_payload(policy: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """A provenance preimage in ``ShadowProvenance.payload()`` shape.

    The pinned identities are the frozen policy's own, read from the policy
    rather than typed out, because a run whose provenance disagrees with the
    policy is refused and a hard-coded copy here would silently stop testing
    agreement the day the policy's identity block changed.
    """
    identity = policy["identity"]

    payload = {
        "dependence_model_version": identity["dependence_model_version"],
        "factor_spec_hash": identity["factor_spec_hash"],
        "factor_spec_sha256": identity["factor_spec_hash"],
        "final_model_spec_sha256": identity["final_model_spec_sha256"],
        # The producer's names, not the policy's. Resolving one vocabulary to
        # the other is the monitor's job and is tested for below.
        "code_sha": "a81360bcce760b0e5df39e0d56c564d15b474e40",
        "marginal_source": "/production/models/marginals.joblib",
        "marginal_source_sha256": "ab" * 32,
        "copula_source": "/production/models/copula.joblib",
        "copula_source_sha256": "cd" * 32,
        "role_scale": {"pts": 1.0},
        "stats": ["pts", "reb", "ast"],
        "simulations": 20000,
        "seed": 73,
        "built_at": "2026-10-20T09:41:00+00:00",
        "promotion_authority": "NONE",
        "published": False,
    }
    payload.update(overrides)
    return {name: value for name, value in payload.items() if value is not None}


def write_day(
    run_dir: Path,
    *,
    slate_date: str,
    rows: pd.DataFrame,
    dependence: list[dict[str, Any]],
    provenance: dict[str, Any],
    fingerprint: str = FINGERPRINT,
    status_overrides: dict[str, Any] | None = None,
    shadow_overrides: dict[str, Any] | None = None,
    write_log: bool = True,
    write_status: bool = True,
    write_grading: bool = True,
) -> dict[str, Path]:
    """One day's RUNNER_TEMP artifacts, as ``run_production_shadow`` leaves them."""
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "shadow_log.jsonl"
    status_path = run_dir / "shadow.json"
    grading_path = run_dir / "shadow_grading.json"

    if write_log:
        with log_path.open("w", encoding="utf-8") as handle:
            for record in rows.to_dict(orient="records"):
                record["slate_date"] = slate_date
                record["provenance_fingerprint"] = fingerprint
                handle.write(json.dumps(record, default=str) + "\n")

    shadow = {
        "psd_failures": 0,
        "same_player_max_block_deviation": 0.0,
        "min_covariance_eigenvalue": 1e-6,
        "games_shadowed": int(rows["game_id"].nunique()) if len(rows) else 0,
        "games_that_fell_back": 0,
        "events_shadowed": int(len(rows)),
        "dependence_diagnostics_recorded": len(dependence),
        "provenance_fingerprint": fingerprint,
        "provenance": provenance,
        "rows_published": 0,
        "slate_has_settled_outcomes": True,
    }
    shadow.update(shadow_overrides or {})

    status = {
        "entry_point": "ops/run_production_shadow.py",
        "mode": "SHADOW_ONLY",
        "slate_date": slate_date,
        "outcome": "SHADOWED",
        "served_authority": "incumbent",
        "rows_published": 0,
        "production_serving_was_affected": False,
        "shadow": shadow,
    }
    status.update(status_overrides or {})

    if write_status:
        status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

    if write_grading:
        grading_path.write_text(
            json.dumps(
                {
                    "slate_date": slate_date,
                    "served_model": "incumbent",
                    "published": False,
                    "provenance_fingerprint": fingerprint,
                    "dependence_diagnostics": dependence,
                    "numerical_diagnostics": [],
                    "fallbacks": [],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    return {
        "shadow_log": log_path,
        "shadow_status": status_path,
        "shadow_grading": grading_path,
    }


def invoke(
    monitor,
    *,
    data_root: Path,
    slate_date: str,
    artifacts: dict[str, Path] | None = None,
    status_path: Path | None = None,
    extra: tuple[str, ...] = (),
) -> int:
    argv = [
        "--slate-date",
        slate_date,
        "--data-root",
        str(data_root),
        "--production-sha",
        "a81360bcce760b0e5df39e0d56c564d15b474e40",
    ]
    for flag in ("shadow_log", "shadow_status", "shadow_grading"):
        if artifacts and flag in artifacts:
            argv += [f"--{flag.replace('_', '-')}", str(artifacts[flag])]
    if status_path is not None:
        argv += ["--status-path", str(status_path)]
    argv += list(extra)
    return monitor.main(argv)


def read_status(monitor, data_root: Path) -> dict[str, Any]:
    path = (
        data_root / monitor.STATE_RELATIVE / "live_shadow_monitoring.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def days_of(rows: pd.DataFrame) -> list[str]:
    return sorted(rows["slate_date"].astype(str).unique())


def accumulate(
    monitor,
    evidence,
    policy,
    *,
    data_root: Path,
    run_root: Path,
    rows: pd.DataFrame,
    dependence: list[dict[str, Any]],
    status_overrides: dict[str, Any] | None = None,
    shadow_overrides: dict[str, Any] | None = None,
    evaluate_each_day: bool = False,
) -> int:
    """Ingest a multi-day window one day at a time, as the lifecycle would.

    ``evaluate_each_day`` is off by default because the point of most tests is
    the accumulated total, and the policy's bootstrap would otherwise be run
    once per simulated day for no extra assurance.
    """
    payload = provenance_payload(policy)
    exit_code = 0

    for index, slate in enumerate(days_of(rows)):
        day_rows = rows.loc[rows["slate_date"].astype(str) == slate]
        # The dependence entries are per game; give each day the slice that
        # matches its share so the pooled reading is over the whole window.
        share = dependence[index :: max(len(days_of(rows)), 1)]
        artifacts = write_day(
            run_root / slate,
            slate_date=slate,
            rows=day_rows,
            dependence=share,
            provenance=payload,
            status_overrides=status_overrides,
            shadow_overrides=shadow_overrides,
        )
        if evaluate_each_day or slate == days_of(rows)[-1]:
            exit_code = invoke(
                monitor,
                data_root=data_root,
                slate_date=slate,
                artifacts=artifacts,
            )
        else:
            monitor.ingest(
                state_root=data_root / monitor.STATE_RELATIVE,
                slate_date=slate,
                policy=policy,
                shadow_log=artifacts["shadow_log"],
                shadow_status=artifacts["shadow_status"],
                shadow_grading=artifacts["shadow_grading"],
                production_sha="a81360bcce760b0e5df39e0d56c564d15b474e40",
            )

    return exit_code


@pytest.fixture
def eligible(evidence):
    """A window sized and shaped to clear every frozen gate and minimum."""
    return {
        "rows": evidence.synthetic_events(
            games=500, days=30, per_leg_count=6, candidate_advantage=0.06
        ),
        "dependence": evidence.synthetic_dependence(
            games=60, candidate_error=0.01, incumbent_error=0.05
        ),
    }


@pytest.fixture
def small(evidence):
    """A healthy but far-too-small window: three days, nine games."""
    return {
        "rows": evidence.synthetic_events(
            games=9, days=3, per_leg_count=2, candidate_advantage=0.06
        ),
        "dependence": evidence.synthetic_dependence(
            games=6, candidate_error=0.01, incumbent_error=0.05
        ),
    }


# ======================================================================
# 1. insufficient evidence -> CONTINUE_SHADOW
# ======================================================================


def test_an_empty_state_continues_the_shadow(monitor, tmp_path):
    assert invoke(monitor, data_root=tmp_path / "data", slate_date="2026-10-20") == 0

    status = read_status(monitor, tmp_path / "data")

    assert status["decision"] == CONTINUE
    assert status["minimum_live_evidence"]["satisfied"] is False
    assert status["sample_counts"]["graded_games"] == 0
    assert status["sample_counts"]["total_joint_events"] == 0
    assert status["sample_counts"]["regular_season_calendar_days"] == 0
    assert status["sample_counts"]["by_leg_count"] == {"2": 0, "3": 0, "4": 0}


def test_insufficient_evidence_continues_the_shadow(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"

    exit_code = accumulate(
        monitor,
        evidence,
        policy,
        data_root=data_root,
        run_root=tmp_path / "runs",
        **small,
    )

    status = read_status(monitor, data_root)

    assert exit_code == 0
    assert status["decision"] == CONTINUE
    assert status["minimum_live_evidence"]["satisfied"] is False
    assert status["sample_counts"]["graded_games"] == 9
    assert status["sample_counts"]["regular_season_calendar_days"] == 3


def test_every_frozen_minimum_is_reported_as_its_own_unmet_requirement(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"
    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )

    unmet = read_status(monitor, data_root)["minimum_live_evidence"]["unmet"]

    assert set(unmet) == {
        "A_games",
        "B_joint_events",
        "B_joint_events_2_leg",
        "B_joint_events_3_leg",
        "B_joint_events_4_leg",
        "C_calendar",
    }


def test_no_gate_verdict_is_rendered_before_the_minimum_is_met(
    monitor, evidence, policy, small, tmp_path
):
    """The policy forbids a verdict early, however good the small sample is."""
    data_root = tmp_path / "data"
    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )

    status = read_status(monitor, data_root)

    for name, gate in status["proper_scores"]["gates"].items():
        assert gate["passed"] is None, f"{name} rendered a verdict early"
    for name, gate in status["dependence"]["gates"].items():
        assert gate["passed"] is None, f"{name} rendered a verdict early"


# ======================================================================
# 2 and 3. rerunning the same evidence, and new evidence arriving
# ======================================================================


def test_rerunning_the_same_day_does_not_double_the_evidence(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"
    payload = provenance_payload(policy)
    slate = days_of(small["rows"])[0]
    rows = small["rows"].loc[small["rows"]["slate_date"] == slate]

    artifacts = write_day(
        tmp_path / "run",
        slate_date=slate,
        rows=rows,
        dependence=small["dependence"],
        provenance=payload,
    )

    counts = []
    for _ in range(3):
        assert (
            invoke(
                monitor, data_root=data_root, slate_date=slate, artifacts=artifacts
            )
            == 0
        )
        counts.append(read_status(monitor, data_root)["sample_counts"])

    assert counts[0] == counts[1] == counts[2]
    assert counts[0]["total_joint_events"] == len(rows)
    assert counts[0]["graded_games"] == int(rows["game_id"].nunique())
    assert counts[0]["regular_season_calendar_days"] == 1
    # One run record and one dependence file per slate, not one per attempt.
    assert counts[0]["shadow_runs_accumulated"] == 1
    assert counts[0]["dependence_games_accumulated"] == len(small["dependence"])


def test_the_stored_event_rows_are_unchanged_by_a_rerun(
    monitor, evidence, policy, small, tmp_path
):
    """Idempotent down to the stored frame, not just the counts."""
    data_root = tmp_path / "data"
    slate = days_of(small["rows"])[0]
    artifacts = write_day(
        tmp_path / "run",
        slate_date=slate,
        rows=small["rows"].loc[small["rows"]["slate_date"] == slate],
        dependence=small["dependence"],
        provenance=provenance_payload(policy),
    )

    invoke(monitor, data_root=data_root, slate_date=slate, artifacts=artifacts)
    before = monitor.accumulated_events(data_root / monitor.STATE_RELATIVE)

    invoke(monitor, data_root=data_root, slate_date=slate, artifacts=artifacts)
    after = monitor.accumulated_events(data_root / monitor.STATE_RELATIVE)

    pd.testing.assert_frame_equal(
        before.sort_values("event_key").reset_index(drop=True),
        after.sort_values("event_key").reset_index(drop=True),
    )


def test_an_event_identity_is_derived_from_the_event_and_nothing_else(monitor):
    """Two ingestions of one event must collide; two events must not."""
    first = monitor.event_key("2026-10-20", 22600001, "evt-a")
    again = monitor.event_key("2026-10-20", 22600001, "evt-a")

    assert first == again
    assert first != monitor.event_key("2026-10-21", 22600001, "evt-a")
    assert first != monitor.event_key("2026-10-20", 22600002, "evt-a")
    assert first != monitor.event_key("2026-10-20", 22600001, "evt-b")


def test_new_settled_evidence_advances_the_counts_exactly_once(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"
    payload = provenance_payload(policy)
    rows = small["rows"]
    observed: list[dict[str, Any]] = []

    for index, slate in enumerate(days_of(rows)):
        day = rows.loc[rows["slate_date"].astype(str) == slate]
        artifacts = write_day(
            tmp_path / "run" / slate,
            slate_date=slate,
            rows=day,
            dependence=small["dependence"][index : index + 2],
            provenance=payload,
        )
        # Twice per day: the second pass must move nothing.
        invoke(monitor, data_root=data_root, slate_date=slate, artifacts=artifacts)
        first = read_status(monitor, data_root)["sample_counts"]
        invoke(monitor, data_root=data_root, slate_date=slate, artifacts=artifacts)
        second = read_status(monitor, data_root)["sample_counts"]

        assert first == second, f"re-ingesting {slate} moved the counts"
        observed.append(second)

    assert [entry["regular_season_calendar_days"] for entry in observed] == [1, 2, 3]
    assert [entry["shadow_runs_accumulated"] for entry in observed] == [1, 2, 3]

    expected = 0
    for index, slate in enumerate(days_of(rows)):
        expected += int((rows["slate_date"].astype(str) == slate).sum())
        assert observed[index]["total_joint_events"] == expected


def test_an_interrupted_write_leaves_the_previous_state_readable(
    monitor, evidence, policy, small, tmp_path
):
    """Atomic replacement: a crash mid-write cannot leave a parsed half-file."""
    data_root = tmp_path / "data"
    slate = days_of(small["rows"])[0]
    artifacts = write_day(
        tmp_path / "run",
        slate_date=slate,
        rows=small["rows"].loc[small["rows"]["slate_date"] == slate],
        dependence=small["dependence"],
        provenance=provenance_payload(policy),
    )
    invoke(monitor, data_root=data_root, slate_date=slate, artifacts=artifacts)

    status_path = data_root / monitor.STATE_RELATIVE / "live_shadow_monitoring.json"
    good = status_path.read_text(encoding="utf-8")
    before = sorted(path.name for path in status_path.parent.iterdir())

    # Fails partway through serialisation: the keys cannot be ordered, which
    # the writer only discovers once it has already opened its temporary file.
    with pytest.raises(TypeError):
        monitor.write_json_atomic(status_path, {"decision": "x", 1: "unsortable"})

    assert status_path.read_text(encoding="utf-8") == good
    # And no temporary file is left behind to be mistaken for state.
    assert sorted(path.name for path in status_path.parent.iterdir()) == before


# ======================================================================
# 4. missing optional evidence -> explicit not-evaluable, never a number
# ======================================================================


def test_a_day_with_no_shadow_run_is_a_recorded_absence_not_a_failure(
    monitor, tmp_path
):
    """A production day with no retrain. The window is still reported."""
    data_root = tmp_path / "data"

    assert (
        invoke(
            monitor,
            data_root=data_root,
            slate_date="2026-10-20",
            artifacts={
                "shadow_log": tmp_path / "absent" / "shadow_log.jsonl",
                "shadow_status": tmp_path / "absent" / "shadow.json",
                "shadow_grading": tmp_path / "absent" / "shadow_grading.json",
            },
        )
        == 0
    )

    status = read_status(monitor, data_root)

    assert status["decision"] == CONTINUE
    assert status["ingestion"]["shadow_evidence_present"] is False
    assert set(status["ingestion"]["absent"]) == {
        "shadow status",
        "shadow log",
        "shadow grading report",
    }
    assert status["ingestion"]["events_ingested"] == 0


def test_absent_dependence_readings_are_not_evaluable_rather_than_zero(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"

    accumulate(
        monitor,
        evidence,
        policy,
        data_root=data_root,
        run_root=tmp_path / "runs",
        rows=small["rows"],
        dependence=[],
    )

    status = read_status(monitor, data_root)

    for space in ("latent", "count"):
        block = status["dependence"][space]
        assert block["candidate_cross_player_rmse"] is None
        assert block["incumbent_cross_player_rmse"] is None
    for name in ("D1_global_latent_dependence_rmse", "D2_count_space_dependence_rmse"):
        gate = status["dependence"]["gates"][name]
        assert gate["passed"] is None
        # Both arms absent, reported as absent. Not a zero, which would read
        # as the two models having measured the same dependence.
        assert gate["observed"] == {"candidate": None, "incumbent": None}


def test_a_slice_with_no_paired_rows_reports_null_and_not_a_tie(
    monitor, evidence, policy, tmp_path
):
    """A fabricated zero here would read as the two arms being equal."""
    data_root = tmp_path / "data"
    rows = evidence.synthetic_events(
        games=4, days=1, per_leg_count=1, candidate_advantage=0.05
    )
    rows = rows.loc[rows["leg_count"] != 4]

    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", rows=rows, dependence=[],
    )

    status = read_status(monitor, data_root)
    four_leg = status["proper_scores"]["by_leg_count"]["4"]

    assert four_leg["paired_events"] == 0
    assert four_leg["candidate_brier"] is None
    assert four_leg["incumbent_brier"] is None
    assert four_leg["brier_delta"] is None
    assert status["proper_scores"]["by_leg_count"]["2"]["paired_events"] > 0


def test_a_run_that_recorded_no_block_deviation_leaves_o3_unevaluable(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"

    accumulate(
        monitor,
        evidence,
        policy,
        data_root=data_root,
        run_root=tmp_path / "runs",
        shadow_overrides={"same_player_max_block_deviation": None},
        **small,
    )

    gate = read_status(monitor, data_root)["operational_gates"][
        "O3_same_player_block_deviation"
    ]

    assert gate["passed"] is None
    assert gate["observed"] is None


# ======================================================================
# 5. provenance mismatch -> fail closed for that evidence
# ======================================================================


@pytest.mark.parametrize(
    "field",
    ["dependence_model_version", "factor_spec_hash", "final_model_spec_sha256"],
)
def test_a_run_whose_provenance_contradicts_the_policy_is_refused(
    monitor, evidence, policy, small, tmp_path, field
):
    """Evidence from a model the policy was not frozen against is not pooled."""
    data_root = tmp_path / "data"
    slate = days_of(small["rows"])[0]
    artifacts = write_day(
        tmp_path / "run",
        slate_date=slate,
        rows=small["rows"].loc[small["rows"]["slate_date"] == slate],
        dependence=small["dependence"],
        provenance=provenance_payload(policy, **{field: "something-else"}),
    )

    assert invoke(monitor, data_root=data_root, slate_date=slate, artifacts=artifacts) == 0

    status = read_status(monitor, data_root)

    assert status["ingestion"]["refused"] is not None
    assert field in status["ingestion"]["refused"]["disagreements"]
    assert status["ingestion"]["events_ingested"] == 0
    assert status["sample_counts"]["total_joint_events"] == 0
    assert status["decision"] == CONTINUE
    # Nothing was written, so nothing can be read back.
    assert not list(
        (data_root / monitor.STATE_RELATIVE / "events").glob("*.parquet")
    )


def test_a_refused_run_does_not_disturb_evidence_already_accumulated(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"
    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )
    before = read_status(monitor, data_root)["sample_counts"]

    foreign = write_day(
        tmp_path / "foreign",
        slate_date="2026-11-01",
        rows=small["rows"].assign(slate_date="2026-11-01"),
        dependence=small["dependence"],
        provenance=provenance_payload(policy, factor_spec_hash="00" * 32),
        fingerprint="dead" * 15,
    )
    assert (
        invoke(
            monitor, data_root=data_root, slate_date="2026-11-01", artifacts=foreign
        )
        == 0
    )

    after = read_status(monitor, data_root)

    assert after["sample_counts"] == before
    assert after["ingestion"]["refused"] is not None


def test_a_log_filed_under_the_wrong_slate_is_refused(
    monitor, evidence, policy, small, tmp_path
):
    """Pooling it would break the calendar-day count the minimum is stated in."""
    data_root = tmp_path / "data"
    artifacts = write_day(
        tmp_path / "run",
        slate_date="2026-10-20",
        rows=small["rows"],
        dependence=small["dependence"],
        provenance=provenance_payload(policy),
    )

    with pytest.raises(monitor.MonitoringRefused, match="not the slate being ingested"):
        invoke(
            monitor, data_root=data_root, slate_date="2026-10-21", artifacts=artifacts
        )


def test_a_log_carrying_one_event_twice_is_refused(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"
    slate = days_of(small["rows"])[0]
    day = small["rows"].loc[small["rows"]["slate_date"] == slate]
    artifacts = write_day(
        tmp_path / "run",
        slate_date=slate,
        rows=pd.concat([day, day.head(1)], ignore_index=True),
        dependence=small["dependence"],
        provenance=provenance_payload(policy),
    )

    with pytest.raises(monitor.MonitoringRefused, match="same event more than once"):
        invoke(monitor, data_root=data_root, slate_date=slate, artifacts=artifacts)


def test_rows_with_no_recorded_preimage_fail_the_provenance_gate(
    monitor, evidence, policy, small, tmp_path
):
    """Fingerprint on the rows, no payload behind it: O5 fails, nothing is faked."""
    data_root = tmp_path / "data"
    slate = days_of(small["rows"])[0]
    artifacts = write_day(
        tmp_path / "run",
        slate_date=slate,
        rows=small["rows"].loc[small["rows"]["slate_date"] == slate],
        dependence=small["dependence"],
        provenance=provenance_payload(policy),
        shadow_overrides={"provenance": {}},
    )

    assert invoke(monitor, data_root=data_root, slate_date=slate, artifacts=artifacts) == 0

    status = read_status(monitor, data_root)
    gate = status["operational_gates"]["O5_provenance_completeness"]

    assert gate["passed"] is False
    assert status["decision"] == CONTINUE


def test_the_policys_four_identities_resolve_from_the_producers_own_names(
    monitor, policy
):
    """``code_sha`` and the artifact hashes are what the shadow actually writes."""
    resolved = monitor.resolved_provenance(provenance_payload(policy))

    for required in policy["operational_gates"]["O5_provenance_completeness"][
        "required_fields"
    ]:
        assert str(resolved.get(required) or "").strip(), required

    assert resolved["production_code_sha"] == resolved["code_sha"]
    assert resolved["artifact_hash_or_version"] == resolved["marginal_source_sha256"]
    # Resolution only: an identity the run did not record stays unrecorded.
    without = monitor.resolved_provenance(
        {
            name: value
            for name, value in provenance_payload(policy).items()
            if name != "code_sha"
        }
    )
    assert "production_code_sha" not in without


def test_the_resolution_map_is_the_health_assertions_own(monitor):
    """One mapping between the two vocabularies, not two."""
    health = _load(
        "shadow_health_for_test", PROJECT / "ops" / "evaluate_shadow_health.py"
    )

    assert monitor._shadow_health().PROVENANCE_FIELD_SOURCES == (
        health.PROVENANCE_FIELD_SOURCES
    )


# ======================================================================
# 6. an operational safety violation -> SHADOW_DISABLED_FOR_SAFETY
# ======================================================================


@pytest.mark.parametrize(
    "overrides, gate",
    [
        ({"rows_published": 3}, "O1_shadow_publishing_violations"),
        (
            {"production_serving_was_affected": True},
            "O6_incumbent_serving_unaffected",
        ),
    ],
)
def test_a_safety_violation_disables_the_shadow_and_fails_the_step(
    monitor, evidence, policy, small, tmp_path, overrides, gate
):
    data_root = tmp_path / "data"

    exit_code = accumulate(
        monitor,
        evidence,
        policy,
        data_root=data_root,
        run_root=tmp_path / "runs",
        status_overrides=overrides,
        **small,
    )

    status = read_status(monitor, data_root)

    assert status["decision"] == DISABLED
    assert status["operational_gates"][gate]["passed"] is False
    assert status["safety_signal_raised"] is True
    assert exit_code == monitor.EXIT_SHADOW_DISABLED_FOR_SAFETY
    assert exit_code != 0


def test_a_safety_failure_changes_nothing_about_the_incumbent(
    monitor, evidence, policy, small, tmp_path
):
    """The failure is a signal. It promotes nothing and publishes nothing."""
    data_root = tmp_path / "data"

    accumulate(
        monitor,
        evidence,
        policy,
        data_root=data_root,
        run_root=tmp_path / "runs",
        status_overrides={"rows_published": 3},
        **small,
    )

    status = read_status(monitor, data_root)

    assert status["decision"] == DISABLED
    assert status["published_authority"] == "incumbent"
    assert status["publishing_switch"] == "DISABLED"
    assert status["promotion_authority"] == "NONE"
    assert status["autonomous_promotion_authority"] == "NONE"
    assert status["published"] is False
    assert status["mode"] == "MONITORING_ONLY"


def test_monitoring_writes_nothing_outside_its_own_state_tree(
    monitor, evidence, policy, small, tmp_path
):
    """Serving, pricing and the registry are untouched by construction."""
    data_root = tmp_path / "data"
    for relative in (
        "processed/priced_markets",
        "processed/incumbent_serving",
        "processed/incumbent_grades",
        "raw/seasons",
    ):
        (data_root / relative).mkdir(parents=True, exist_ok=True)

    before = {
        path: path.stat().st_mtime_ns
        for path in sorted(data_root.rglob("*"))
        if path.is_file()
    }

    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )

    after = {
        path: path.stat().st_mtime_ns
        for path in sorted(data_root.rglob("*"))
        if path.is_file()
    }

    created = sorted(set(after) - set(before))
    assert before == {path: after[path] for path in before}
    assert created, "monitoring wrote nothing at all"
    for path in created:
        assert monitor.STATE_RELATIVE.as_posix() in path.as_posix(), path


def test_a_shadow_quality_failure_is_not_a_safety_failure(
    monitor, evidence, policy, small, tmp_path
):
    """O2 is real and blocks eligibility, but it does not page anyone."""
    data_root = tmp_path / "data"

    exit_code = accumulate(
        monitor,
        evidence,
        policy,
        data_root=data_root,
        run_root=tmp_path / "runs",
        shadow_overrides={"psd_failures": 2},
        **small,
    )

    status = read_status(monitor, data_root)

    assert status["operational_gates"]["O2_psd_failures"]["passed"] is False
    assert status["decision"] == CONTINUE
    assert status["safety_signal_raised"] is False
    assert exit_code == 0


# ======================================================================
# 7. healthy but insufficient -> green lifecycle, CONTINUE_SHADOW
# ======================================================================


def test_a_healthy_small_window_keeps_the_lifecycle_green(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"

    exit_code = accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )

    status = read_status(monitor, data_root)

    assert exit_code == 0
    assert status["decision"] == CONTINUE
    assert status["safety_signal_raised"] is False
    assert status["lifecycle_exit_code"] == 0
    for name, gate in status["operational_gates"].items():
        assert gate["passed"] is not False, f"{name} failed on a healthy window"


# ======================================================================
# 8. the frozen minimum satisfied -> PROMOTION_REVIEW_ELIGIBLE, no promotion
# ======================================================================


def test_a_window_satisfying_every_frozen_gate_is_review_eligible(
    monitor, evidence, policy, eligible, tmp_path
):
    data_root = tmp_path / "data"

    exit_code = accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **eligible,
    )

    status = read_status(monitor, data_root)
    minimum = policy["minimum_live_evidence"]
    counts = status["sample_counts"]

    assert exit_code == 0
    assert status["decision"] == ELIGIBLE
    assert status["minimum_live_evidence"]["satisfied"] is True
    assert counts["graded_games"] >= minimum["A_games"]["min_live_nba_games_graded"]
    assert (
        counts["total_joint_events"]
        >= minimum["B_joint_events"]["min_live_graded_joint_events"]
    )
    for legs, floor in minimum["B_joint_events"]["by_leg_count"].items():
        assert counts["by_leg_count"][legs] >= floor
    assert (
        counts["regular_season_calendar_days"]
        >= minimum["C_calendar"]["min_regular_season_days"]
    )
    for name, gate in status["operational_gates"].items():
        assert gate["passed"] is True, f"{name} did not pass: {gate['detail']}"
    for name, gate in status["proper_scores"]["gates"].items():
        assert gate["passed"] is True, f"{name} did not pass: {gate['detail']}"
    for name, gate in status["dependence"]["gates"].items():
        assert gate["passed"] is True, f"{name} did not pass: {gate['detail']}"


def test_review_eligible_promotes_nothing_and_publishes_nothing(
    monitor, evidence, policy, eligible, tmp_path
):
    data_root = tmp_path / "data"
    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **eligible,
    )

    status = read_status(monitor, data_root)

    assert status["decision"] == ELIGIBLE
    assert status["promotion_authority"] == "NONE"
    assert status["autonomous_promotion_authority"] == "NONE"
    assert status["published_authority"] == "incumbent"
    assert status["publishing_switch"] == "DISABLED"
    assert status["published"] is False
    assert status["rows_published"] == 0
    assert status["production_serving_was_affected"] is False
    # No registry, no promotion state, nothing but the monitoring tree.
    assert sorted(
        path.name for path in (data_root / "processed").iterdir()
    ) == ["live_shadow_monitoring"]


def test_a_degraded_candidate_on_a_full_window_is_rejected(
    monitor, evidence, policy, tmp_path
):
    """The other end of the same wiring: the decision is not pinned to one word."""
    data_root = tmp_path / "data"

    exit_code = accumulate(
        monitor,
        evidence,
        policy,
        data_root=data_root,
        run_root=tmp_path / "runs",
        rows=evidence.synthetic_events(
            games=500, days=30, per_leg_count=6, candidate_advantage=-0.06
        ),
        dependence=evidence.synthetic_dependence(
            games=60, candidate_error=0.09, incumbent_error=0.05
        ),
    )

    status = read_status(monitor, data_root)

    assert status["decision"] == REJECTED
    assert exit_code == 0, "a rejected candidate is a result, not an outage"
    assert status["published_authority"] == "incumbent"


# ======================================================================
# 9. no publishing activation appears
# ======================================================================


def test_the_monitoring_entry_point_names_no_publishing_activation():
    source = ENTRY_POINT.read_text(encoding="utf-8")

    for token in (
        "NBA_PROP_SHADOW_PUBLISH",
        "NBA_PROP_SHADOW_PUBLISH_APPROVAL",
        "19_build_wizardofodds_runtime_bundle.py",
        "publish_shadow_probabilities",
        "--publish",
    ):
        assert token not in source, token


def test_the_monitoring_step_names_no_publishing_activation():
    workflow = yaml.safe_load(LIFECYCLE.read_text(encoding="utf-8"))
    steps = [
        step
        for step in workflow["jobs"]["lifecycle"]["steps"]
        if "monitor_live_shadow_evidence.py" in str(step.get("run", ""))
    ]

    assert len(steps) == 1

    rendered = yaml.safe_dump(steps[0])
    for token in (
        "NBA_PROP_SHADOW_PUBLISH",
        "NBA_PROP_SHADOW_PUBLISH_APPROVAL",
        "19_build_wizardofodds_runtime_bundle.py",
        "--publish",
    ):
        assert token not in rendered, token


def test_monitoring_defines_no_promotion_or_publishing_entry_point():
    source = ENTRY_POINT.read_text(encoding="utf-8")

    for forbidden in ("def promote", "def publish", "def register", "def activate"):
        assert forbidden not in source, forbidden


# ======================================================================
# the monitor states no threshold of its own
# ======================================================================


def test_every_bound_the_monitor_reports_came_from_the_frozen_policy(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"
    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )
    status = read_status(monitor, data_root)

    operational = policy["operational_gates"]
    assert status["operational_gates"]["O1_shadow_publishing_violations"][
        "threshold"
    ] == operational["O1_shadow_publishing_violations"]["max"]
    assert status["operational_gates"]["O4_candidate_fallback_rate"][
        "threshold"
    ] == operational["O4_candidate_fallback_rate"]["max"]
    assert status["proper_scores"]["gates"]["aggregate_brier"]["threshold"] == (
        policy["proper_score_gates"]["aggregate"]["brier"][
            "upper_95_ci_of_delta_max"
        ]
    )
    assert status["decision_states"] == policy["decision_states"]["states"]


def test_a_policy_missing_a_field_is_refused_rather_than_defaulted(
    monitor, policy, tmp_path
):
    broken = dict(policy)
    broken.pop("served_authority")

    root = tmp_path / "project"
    (root / "research" / "final_model").mkdir(parents=True)
    (root / "research" / "final_model" / "live_shadow_promotion_policy.json").write_text(
        json.dumps(broken), encoding="utf-8"
    )

    with pytest.raises(Exception) as caught:
        monitor.main(
            [
                "--slate-date",
                "2026-10-20",
                "--data-root",
                str(tmp_path / "data"),
                "--project-root",
                str(root),
            ]
        )

    assert "served_authority" in str(caught.value)


def test_the_status_artifact_records_the_policy_it_was_evaluated_against(
    monitor, policy, tmp_path
):
    import hashlib

    data_root = tmp_path / "data"
    invoke(monitor, data_root=data_root, slate_date="2026-10-20")

    status = read_status(monitor, data_root)
    expected = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()

    assert status["policy_sha256"] == expected
    assert status["policy_identity"] == policy["identity"]


# ======================================================================
# candidate and incumbent evidence stay distinguishable
# ======================================================================


def test_the_incumbents_own_graded_record_is_reported_separately(
    monitor, evidence, policy, small, tmp_path
):
    """Single-leg prop grades are the incumbent's record, not the gated arm."""
    data_root = tmp_path / "data"
    grades = data_root / "processed" / "incumbent_grades" / "season=2027"
    grades.mkdir(parents=True)
    pd.DataFrame.from_records(
        [
            {
                "prediction_id": "a" * 64,
                "slate_date": "2026-10-20",
                "settlement_status": "GRADED",
                "brier_contribution": 0.16,
                "log_loss_contribution": 0.51,
            },
            {
                "prediction_id": "b" * 64,
                "slate_date": "2026-10-20",
                "settlement_status": "PENDING_SETTLEMENT",
                "brier_contribution": None,
                "log_loss_contribution": None,
            },
        ]
    ).to_parquet(grades / "date=2026-10-20.parquet", index=False)

    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )

    status = read_status(monitor, data_root)
    record = status["incumbent_production_grades"]

    assert record["available"] is True
    assert record["graded_rows"] == 1
    assert record["pending_rows"] == 1
    assert record["brier"] == pytest.approx(0.16)
    assert "joint event" in record["comparison_note"]
    # And it has not been mistaken for the gated incumbent arm, which is read
    # off the shadow rows and is a different number.
    assert status["proper_scores"]["aggregate"]["incumbent_brier"] != pytest.approx(
        0.16
    )


def test_an_absent_incumbent_grade_store_is_reported_as_absent(
    monitor, tmp_path
):
    data_root = tmp_path / "data"
    invoke(monitor, data_root=data_root, slate_date="2026-10-20")

    record = read_status(monitor, data_root)["incumbent_production_grades"]

    assert record["available"] is False
    assert record["brier"] is None
    assert record["graded_rows"] == 0


def test_the_candidate_identity_is_read_from_the_recorded_preimage(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"
    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )

    identity = read_status(monitor, data_root)["candidate_identity"]

    assert identity["provenance_fingerprints"] == [FINGERPRINT]
    assert identity["dependence_model_version"] == (
        policy["identity"]["dependence_model_version"]
    )
    assert identity["factor_spec_hash"] == policy["identity"]["factor_spec_hash"]
    assert identity["final_model_spec_sha256"] == (
        policy["identity"]["final_model_spec_sha256"]
    )
    assert identity["pinned_by_policy"]["factor_spec_hash"] == (
        policy["identity"]["factor_spec_hash"]
    )


def test_the_incumbent_identity_comes_from_the_durable_serving_receipt(
    monitor, tmp_path
):
    data_root = tmp_path / "data"
    receipts = data_root / "processed" / "incumbent_serving"
    receipts.mkdir(parents=True)
    (receipts / "2026-10-20.json").write_text(
        json.dumps(
            {
                "slate_date": "2026-10-20",
                "incumbent_authority": "frozen_deployment_bundle",
                "incumbent_fit_id": None,
                "incumbent_version": "freeze-2026-10-01",
                "input_state_fingerprint": "d4f1" * 16,
                "prediction_artifact": {"sha256": "ab" * 32},
                "production_code_sha": "a81360b",
            }
        ),
        encoding="utf-8",
    )

    invoke(monitor, data_root=data_root, slate_date="2026-10-21")

    identity = read_status(monitor, data_root)["incumbent_identity"]

    assert identity["available"] is True
    assert identity["authority"] == "frozen_deployment_bundle"
    assert identity["version"] == "freeze-2026-10-01"
    assert identity["prediction_artifact_sha256"] == "ab" * 32


# ======================================================================
# the required contents of the status artifact
# ======================================================================


def test_the_status_artifact_carries_everything_an_operator_needs(
    monitor, evidence, policy, small, tmp_path
):
    data_root = tmp_path / "data"
    accumulate(
        monitor, evidence, policy, data_root=data_root,
        run_root=tmp_path / "runs", **small,
    )
    status = read_status(monitor, data_root)

    for key in (
        "generated_at",
        "production_code_sha",
        "candidate_identity",
        "incumbent_identity",
        "sample_counts",
        "operational_gates",
        "proper_scores",
        "dependence",
        "decision",
        "decision_reason",
        "decision_states",
    ):
        assert key in status, key

    for key in (
        "graded_games",
        "total_joint_events",
        "by_leg_count",
        "regular_season_calendar_days",
    ):
        assert key in status["sample_counts"], key

    for index in range(1, 7):
        assert any(
            name.startswith(f"O{index}_") for name in status["operational_gates"]
        ), f"O{index} is not reported"

    aggregate = status["proper_scores"]["aggregate"]
    for key in (
        "candidate_brier",
        "incumbent_brier",
        "candidate_log_loss",
        "incumbent_log_loss",
    ):
        assert key in aggregate, key

    for legs in ("2", "3", "4"):
        block = status["proper_scores"]["by_leg_count"][legs]
        assert "candidate_brier" in block and "candidate_log_loss" in block

    for space in ("latent", "count"):
        block = status["dependence"][space]
        assert "candidate_cross_player_rmse" in block
        assert "incumbent_cross_player_rmse" in block

    assert set(status["dependence"]["all_declared_buckets"]) == set(
        policy["tracked_metrics"]["dependence_buckets"]
    )
    assert set(status["dependence"]["protected_buckets"]) == set(
        policy["dependence_gates"]["D3_protected_buckets_may_not_worsen"][
            "protected_buckets"
        ]
    )
    assert set(status["decision_states"]) == {CONTINUE, ELIGIBLE, REJECTED, DISABLED}


def test_the_status_artifact_is_written_where_the_next_run_reads_it(
    monitor, tmp_path
):
    data_root = tmp_path / "data"
    invoke(monitor, data_root=data_root, slate_date="2026-10-20")

    assert (
        data_root / monitor.STATE_RELATIVE / "live_shadow_monitoring.json"
    ).is_file()


def test_an_explicit_status_path_is_honoured(monitor, tmp_path):
    data_root = tmp_path / "data"
    elsewhere = tmp_path / "run" / "live_shadow_monitoring.json"

    invoke(
        monitor,
        data_root=data_root,
        slate_date="2026-10-20",
        status_path=elsewhere,
    )

    assert elsewhere.is_file()
    assert json.loads(elsewhere.read_text(encoding="utf-8"))["decision"] == CONTINUE


# ======================================================================
# the lifecycle wiring
# ======================================================================


@pytest.fixture(scope="module")
def lifecycle() -> dict[str, Any]:
    return yaml.safe_load(LIFECYCLE.read_text(encoding="utf-8"))


def monitoring_step(lifecycle: dict[str, Any]) -> dict[str, Any]:
    for step in lifecycle["jobs"]["lifecycle"]["steps"]:
        if "monitor_live_shadow_evidence.py" in str(step.get("run", "")):
            return step
    raise AssertionError("the lifecycle does not invoke the monitoring entry point")


def step_names(lifecycle: dict[str, Any]) -> list[str]:
    return [str(step.get("name", "")) for step in lifecycle["jobs"]["lifecycle"]["steps"]]


def test_the_lifecycle_runs_the_monitoring_step(lifecycle):
    assert "Monitor the accumulated live shadow evidence" in step_names(lifecycle)


def test_monitoring_is_not_gated_on_a_retrain_happening(lifecycle):
    """The accumulated window must be summarised on a day with no retrain."""
    condition = str(monitoring_step(lifecycle).get("if", ""))

    assert "RUN_ADAPTIVE" not in condition
    assert "production" in condition


def test_monitoring_runs_after_the_evidence_it_reads_is_produced(lifecycle):
    names = step_names(lifecycle)
    monitoring = names.index("Monitor the accumulated live shadow evidence")

    for producer in (
        "Serve the slate with the incumbent",
        "Grade the incumbent's previous slate",
        "Shadow the slate beside production",
    ):
        assert names.index(producer) < monitoring, producer


def test_monitoring_runs_before_the_blocking_health_assertion(lifecycle):
    """Behind it, monitoring would be skipped on exactly the unhealthy days."""
    names = step_names(lifecycle)

    assert names.index("Monitor the accumulated live shadow evidence") < names.index(
        "Assert the candidate shadow was healthy"
    )


def test_monitoring_can_fail_the_job(lifecycle):
    """A swallowed SHADOW_DISABLED_FOR_SAFETY would be no signal at all."""
    assert "continue-on-error" not in monitoring_step(lifecycle)


def test_incumbent_serving_is_not_gated_on_monitoring(lifecycle):
    """Monitoring is downstream of serving and cannot reach back."""
    steps = lifecycle["jobs"]["lifecycle"]["steps"]
    serving = next(
        step
        for step in steps
        if "run_incumbent_production_serving.py" in str(step.get("run", ""))
    )

    assert "monitor" not in str(serving.get("if", "")).lower()
    assert "monitor" not in str(serving.get("run", ""))


def test_incumbent_grading_is_not_gated_on_the_candidate_being_healthy(lifecycle):
    steps = lifecycle["jobs"]["lifecycle"]["steps"]
    grading = next(
        step
        for step in steps
        if "grade_incumbent_production_slate.py" in str(step.get("run", ""))
    )
    condition = str(grading.get("if", ""))

    assert "shadow" not in condition.lower()
    assert "monitor" not in condition.lower()
    assert "RUN_ADAPTIVE" not in condition


def test_the_monitoring_step_reads_the_days_shadow_artifacts(lifecycle):
    run = str(monitoring_step(lifecycle)["run"])

    for flag in ("--shadow-log", "--shadow-status", "--shadow-grading"):
        assert flag in run, flag
    assert "--data-root" in run
    assert "--production-sha" in run


def test_the_monitoring_step_persists_its_status_durably(lifecycle):
    """Under the data root, not RUNNER_TEMP, which dies with the job."""
    run = str(monitoring_step(lifecycle)["run"])

    assert "NBA_PROP_DATA_DIR" in run
    assert "live_shadow_monitoring.json" in run
    assert "--status-path" in run


def test_the_lifecycle_change_adds_lines_and_removes_none():
    """The additive-only contract, checked against the production head."""
    base = subprocess.run(
        ["git", "rev-parse", "origin/production/wizardofodds-integration"],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    if base.returncode != 0:
        pytest.skip("the production ref is not available in this checkout")

    relative = ".github/workflows/nba_production_lifecycle.yml"
    before = subprocess.run(
        ["git", "show", f"{base.stdout.strip()}:{relative}"],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    ).stdout.splitlines()
    after = (PROJECT / relative).read_text(encoding="utf-8").splitlines()

    assert [line for line in before if line not in after] == []
