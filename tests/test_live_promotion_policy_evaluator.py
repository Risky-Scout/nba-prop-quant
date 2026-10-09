"""The live promotion-policy evaluator, exercised on synthetic evidence.

WHY SYNTHETIC
-------------

No live regular-season slate has run, so there is no real accumulated window
to evaluate. That is the point of writing these tests now rather than later:
every one of them constructs evidence that is *known* to sit on a particular
side of a frozen gate, and asserts the evaluator agrees. Written after live
evidence arrived, the same assertions could be quietly chosen to match
whatever the live sample happened to show.

WHAT IS PROVEN HERE
-------------------

* All four frozen decision states are reachable, and only those four.
* Every frozen gate has a negative control: a window that passes everything
  else and fails exactly that gate, with the decision the policy's own table
  says that failure produces.
* Eligibility cannot be inferred before the minimum evidence requirement is
  met, including when the candidate is overwhelmingly better.
* No threshold, minimum, bucket list, draw count or seed comes from the
  evaluator. Deleting the policy field makes the evaluator refuse rather than
  substitute a default.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nba_prop_quant.research.game_latent_state import live_policy as lp
from nba_prop_quant.research.game_latent_state.live_policy import (
    EvidenceWindow,
    PolicyIncomplete,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The four states the frozen policy allows and nothing else.
CONTINUE = "CONTINUE_SHADOW"
ELIGIBLE = "PROMOTION_REVIEW_ELIGIBLE"
REJECTED = "SHADOW_MODEL_REJECTED"
DISABLED = "SHADOW_DISABLED_FOR_SAFETY"


@pytest.fixture(scope="module")
def policy() -> dict[str, Any]:
    return dict(lp.load_policy(PROJECT_ROOT))


# ======================================================================
# synthetic evidence
# ======================================================================


def _probabilities(
    realized: np.ndarray, error: np.ndarray
) -> np.ndarray:
    """Probabilities that are ``error`` away from the realized outcome.

    Both gated scores are monotone in this distance -- Brier by construction,
    log loss because the probability assigned to what happened is ``1 -
    error`` -- so an arm with a smaller per-row error is better on both, on
    every row. That is what makes the dominance constructions below exact
    rather than statistical: every bootstrap draw inherits the ordering, so
    the test does not depend on the interval happening to land somewhere.
    """
    return np.where(realized > 0.5, 1.0 - error, error)


def synthetic_events(
    *,
    games: int,
    days: int,
    per_leg_count: int,
    candidate_advantage: float,
    seed: int = 20261020,
    worse_leg_counts: tuple[int, ...] = (),
) -> pd.DataFrame:
    """One graded shadow row per joint event over ``games`` games.

    ``candidate_advantage`` is how much closer to the realized outcome the
    candidate sits than the incumbent, per row. Positive means the candidate
    dominates every row; negative means the incumbent does. ``worse_leg_counts``
    flips the sign for those leg counts only, which is how the by-leg gates
    get a control that leaves the aggregate intact.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for game in range(games):
        slate = pd.Timestamp("2026-10-20") + pd.Timedelta(days=game % days)
        for leg_count in (2, 3, 4):
            for leg in range(per_leg_count):
                realized = float(rng.integers(0, 2))
                incumbent_error = float(rng.uniform(0.18, 0.34))
                advantage = (
                    -candidate_advantage
                    if leg_count in worse_leg_counts
                    else candidate_advantage
                )
                candidate_error = float(
                    np.clip(incumbent_error - advantage, 0.02, 0.60)
                )
                probability = _probabilities(
                    np.array([realized]),
                    np.array([candidate_error]),
                )[0]
                incumbent_probability = _probabilities(
                    np.array([realized]),
                    np.array([incumbent_error]),
                )[0]
                records.append(
                    {
                        "event_id": f"g{game:04d}-{leg_count}-{leg}",
                        "game_id": f"002260{game:05d}",
                        "slate_date": slate.strftime("%Y-%m-%d"),
                        "leg_count": leg_count,
                        "legs": leg_count,
                        "candidate_probability": float(probability),
                        "incumbent_probability": float(incumbent_probability),
                        "independence_probability": float(incumbent_probability),
                        "realized": realized,
                        "fell_back": False,
                        "published": False,
                        "served_model": "incumbent",
                        "provenance_fingerprint": "f" * 16,
                    }
                )

    return pd.DataFrame.from_records(records)


#: The buckets the dependence readings below carry. The evaluator pools and
#: scores exactly the cross-player ones, and reads the pair counts to do it.
def synthetic_dependence(
    *,
    games: int,
    candidate_error: float,
    incumbent_error: float,
    seed: int = 73,
    worsened_buckets: tuple[str, ...] = (),
    spaces: tuple[str, ...] = (lp.LATENT_SPACE, lp.COUNT_SPACE),
) -> list[dict[str, Any]]:
    """Per-game bucket readings for observed, candidate, incumbent, independence.

    The observed reading is the target; each arm sits a fixed distance from
    it, so the pooled cross-player RMSE of each arm is that distance and the
    ordering the gates read is exact. ``worsened_buckets`` pushes the
    candidate further from observed than the incumbent on those buckets only,
    which is the protected-bucket control.
    """
    rng = np.random.default_rng(seed)
    entries: list[dict[str, Any]] = []

    for game in range(games):
        entry: dict[str, Any] = {"game_id": f"002260{game:05d}"}
        for space in spaces:
            observed = {
                bucket: float(rng.uniform(-0.05, 0.15))
                for bucket in lp.CROSS_PLAYER_BUCKETS
            }
            pairs = {"_same_team_pairs": 12.0, "_cross_team_pairs": 20.0}

            def arm(distance: float, flipped: tuple[str, ...] = ()) -> dict[str, float]:
                return {
                    bucket: observed[bucket]
                    + (incumbent_error * 2.0 if bucket in flipped else distance)
                    for bucket in lp.CROSS_PLAYER_BUCKETS
                } | pairs

            entry[space] = {
                "observed": observed | pairs,
                lp.CANDIDATE: arm(candidate_error, worsened_buckets),
                lp.INCUMBENT: arm(incumbent_error),
                lp.INDEPENDENCE: arm(incumbent_error * 1.5),
            }
        entries.append(entry)

    return entries


def synthetic_runs(
    *,
    days: int,
    rows_published: float = 0.0,
    psd_failures: float = 0.0,
    same_player_max_block_deviation: float = 0.0,
    production_serving_was_affected: bool = False,
) -> list[dict[str, Any]]:
    return [
        {
            "slate_date": (
                pd.Timestamp("2026-10-20") + pd.Timedelta(days=day)
            ).strftime("%Y-%m-%d"),
            "rows_published": rows_published,
            "psd_failures": psd_failures,
            "same_player_max_block_deviation": same_player_max_block_deviation,
            "production_serving_was_affected": production_serving_was_affected,
            "served_authority": "incumbent",
        }
        for day in range(days)
    ]


def synthetic_provenance(
    *, fingerprint: str = "f" * 16, drop: tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    payload = {
        "final_model_spec_sha256": "1541096581279" + "2" * 51,
        "factor_spec_hash": "3229bbc83a438" + "8" * 51,
        "artifact_hash_or_version": "game-latent-state-shadow-v1",
        "production_code_sha": "a81360bcce760b0e5df39e0d56c564d15b474e40",
    }
    for name in drop:
        payload.pop(name, None)
    return {fingerprint: payload}


def eligible_window(**overrides: Any) -> EvidenceWindow:
    """A window that clears every frozen gate and every minimum.

    Sized from the frozen minimums with headroom: 500 games is the floor, and
    three leg counts at 6 events per game each gives 3000 of every leg count
    and 9000 joint events against floors of 2000, 2000, 1200 and 8000.
    """
    events = overrides.pop(
        "events",
        synthetic_events(
            games=500,
            days=30,
            per_leg_count=6,
            candidate_advantage=0.06,
        ),
    )
    window = EvidenceWindow(
        events=events,
        provenance=overrides.pop("provenance", synthetic_provenance()),
        dependence=overrides.pop(
            "dependence",
            synthetic_dependence(
                games=40, candidate_error=0.01, incumbent_error=0.05
            ),
        ),
        runs=overrides.pop("runs", synthetic_runs(days=30)),
    )
    assert not overrides, f"unused overrides: {sorted(overrides)}"
    return window


# ======================================================================
# the frozen policy is the only source of every number
# ======================================================================


def test_the_evaluator_reads_the_frozen_policy_from_the_repository(policy):
    assert policy["AUTONOMOUS_PROMOTION_AUTHORITY"] == "NONE"
    assert set(lp.decision_vocabulary(policy)) == {
        CONTINUE,
        ELIGIBLE,
        REJECTED,
        DISABLED,
    }


def test_the_frozen_policy_file_is_the_one_the_identity_block_names(policy):
    """The evaluator must be reading the policy the model freeze recorded."""
    spec = json.loads(
        (PROJECT_ROOT / "research" / "final_model" / "final_model_spec.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        policy["identity"]["factor_spec_hash"] == spec["factor_spec_hash"]
    )


@pytest.mark.parametrize(
    "path",
    [
        ("operational_gates", "O1_shadow_publishing_violations", "max"),
        ("operational_gates", "O2_psd_failures", "max"),
        ("operational_gates", "O3_same_player_block_deviation", "max"),
        ("operational_gates", "O4_candidate_fallback_rate", "max"),
        ("operational_gates", "O5_provenance_completeness", "required_fields"),
        ("operational_gates", "O6_incumbent_serving_unaffected", "requirement"),
        ("minimum_live_evidence", "A_games", "min_live_nba_games_graded"),
        ("minimum_live_evidence", "B_joint_events", "min_live_graded_joint_events"),
        ("minimum_live_evidence", "B_joint_events", "by_leg_count"),
        ("minimum_live_evidence", "C_calendar", "min_regular_season_days"),
        ("proper_score_gates", "aggregate", "brier", "upper_95_ci_of_delta_max"),
        ("proper_score_gates", "aggregate", "log_loss", "upper_95_ci_of_delta_max"),
        ("proper_score_gates", "by_leg_count", "leg_counts"),
        ("proper_score_gates", "by_leg_count", "upper_95_ci_of_brier_delta_max"),
        ("proper_score_gates", "by_leg_count", "upper_95_ci_of_log_loss_delta_max"),
        ("proper_score_gates", "bootstrap"),
        ("dependence_gates", "D1_global_latent_dependence_rmse", "requirement"),
        ("dependence_gates", "D2_count_space_dependence_rmse", "requirement"),
        ("dependence_gates", "D3_protected_buckets_may_not_worsen", "max_worsening_z"),
        ("dependence_gates", "D3_protected_buckets_may_not_worsen", "protected_buckets"),
        ("dependence_gates", "D4_explicitly_reported_not_gated", "buckets"),
    ],
)
def test_removing_a_policy_field_makes_the_evaluator_refuse(policy, path):
    """No field has a default, because a default here is an invented threshold."""
    mutilated = copy.deepcopy(policy)
    node = mutilated
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]

    with pytest.raises(PolicyIncomplete) as raised:
        lp.evaluate(mutilated, eligible_window())
    assert ".".join(path) in str(raised.value) or path[-1] in str(raised.value)


def test_the_evaluator_refuses_to_run_without_the_no_autonomy_clause(policy):
    mutilated = copy.deepcopy(policy)
    mutilated["AUTONOMOUS_PROMOTION_AUTHORITY"] = "SHADOW"
    with pytest.raises(PolicyIncomplete):
        lp.evaluate(mutilated, eligible_window())


def test_the_bootstrap_settings_are_the_frozen_ones(policy):
    settings = lp._bootstrap_settings(policy)
    assert settings["draws"] == 5000
    assert settings["seed"] == 73
    assert settings["ci"].startswith("95")

    report = lp.evaluate(policy, eligible_window())
    assert {reading.draws for reading in report.readings} == {5000}


def test_an_unpaired_or_differently_clustered_bootstrap_is_refused(policy):
    for change in ({"paired": False}, {"cluster": "slate_date"}):
        mutilated = copy.deepcopy(policy)
        mutilated["proper_score_gates"]["bootstrap"].update(change)
        with pytest.raises(PolicyIncomplete):
            lp._bootstrap_settings(mutilated)


# ======================================================================
# the four decision states
# ======================================================================


def test_an_empty_window_continues_the_shadow(policy):
    report = lp.evaluate(policy, EvidenceWindow())
    assert report.decision == CONTINUE
    assert "minimum live-evidence" in report.reason
    assert all(gate.passed is None for gate in report.scoring)
    assert all(gate.passed is None for gate in report.dependence)


def test_a_clean_full_window_reaches_promotion_review_eligible(policy):
    report = lp.evaluate(policy, eligible_window())

    assert report.decision == ELIGIBLE
    assert report.evidence.satisfied
    assert [gate.name for gate in report.operational if not gate.passed] == []
    assert [gate.name for gate in report.scoring if not gate.passed] == []
    assert [gate.name for gate in report.dependence if not gate.passed] == []


def test_promotion_review_eligible_promotes_nothing(policy):
    """The eligible report is a report. It carries no promotion authority."""
    report = lp.evaluate(policy, eligible_window())
    payload = report.payload()

    assert report.decision == ELIGIBLE
    assert payload["promotion_authority"] == "NONE"
    assert "not a promotion" in report.reason
    # The evaluator has no way to act: it exposes a decision string and
    # readings, and nothing that mutates authority, publishes or promotes.
    assert set(payload) == {
        "decision",
        "decision_reason",
        "dependence_detail",
        "dependence_gates",
        "minimum_live_evidence",
        "operational_gates",
        "policy_identity",
        "promotion_authority",
        "proper_score_gates",
        "score_readings",
    }


def test_a_worse_candidate_is_rejected(policy):
    window = eligible_window(
        events=synthetic_events(
            games=500, days=30, per_leg_count=6, candidate_advantage=-0.06
        )
    )
    report = lp.evaluate(policy, window)

    assert report.decision == REJECTED
    failed = {gate.name for gate in report.scoring if gate.failed}
    assert "aggregate_brier" in failed
    assert "aggregate_log_loss" in failed


def test_a_published_shadow_row_disables_the_shadow_for_safety(policy):
    window = eligible_window(runs=synthetic_runs(days=30, rows_published=1.0))
    report = lp.evaluate(policy, window)

    assert report.decision == DISABLED
    assert not report.safe


def test_a_decision_outside_the_frozen_vocabulary_cannot_be_returned(policy):
    """``decide`` routes every answer through the frozen state list."""
    mutilated = copy.deepcopy(policy)
    mutilated["decision_states"]["states"] = [CONTINUE]
    with pytest.raises(PolicyIncomplete):
        lp.decide(
            mutilated,
            operational=[
                lp.Gate(
                    name="O1_shadow_publishing_violations",
                    passed=False,
                    requirement="",
                )
            ],
            evidence=lp.MinimumEvidence(satisfied=True, requirements=[]),
            scoring=[],
            dependence=[],
        )


# ======================================================================
# eligibility cannot be inferred early
# ======================================================================


@pytest.mark.parametrize(
    "shortfall, unmet",
    [
        ({"games": 60, "days": 30, "per_leg_count": 6}, "A_games"),
        ({"games": 500, "days": 30, "per_leg_count": 1}, "B_joint_events"),
        ({"games": 500, "days": 7, "per_leg_count": 6}, "C_calendar"),
    ],
)
def test_an_overwhelmingly_better_candidate_is_still_not_eligible_early(
    policy, shortfall, unmet
):
    """The candidate dominates every row and the answer is still CONTINUE."""
    window = eligible_window(
        events=synthetic_events(candidate_advantage=0.12, **shortfall),
        runs=synthetic_runs(days=shortfall["days"]),
    )
    report = lp.evaluate(policy, window)

    assert report.decision == CONTINUE
    assert unmet in report.evidence.payload()["unmet"]
    # Not merely a different decision: no verdict is rendered at all.
    assert all(gate.passed is None for gate in report.scoring)
    assert all(gate.passed is None for gate in report.dependence)


def test_each_minimum_is_a_separate_blocking_requirement(policy):
    """Nine events per leg count, then 4-leg removed: only its floor is unmet."""
    window = eligible_window(
        events=synthetic_events(
            games=500, days=30, per_leg_count=9, candidate_advantage=0.06
        ).query("leg_count != 4")
    )
    report = lp.evaluate(policy, window)

    assert report.decision == CONTINUE
    assert report.evidence.payload()["unmet"] == ["B_joint_events_4_leg"]


def test_a_window_one_day_short_of_the_calendar_minimum_is_not_eligible(policy):
    """The boundary, because an off-by-one here would grant eligibility early."""
    short = eligible_window(
        events=synthetic_events(
            games=500, days=29, per_leg_count=6, candidate_advantage=0.06
        ),
        runs=synthetic_runs(days=29),
    )
    assert lp.evaluate(policy, short).decision == CONTINUE
    assert lp.evaluate(policy, eligible_window()).decision == ELIGIBLE


# ======================================================================
# negative controls: operational gates
# ======================================================================


@pytest.mark.parametrize(
    "run_change, gate_name, decision",
    [
        (
            {"rows_published": 1.0},
            "O1_shadow_publishing_violations",
            DISABLED,
        ),
        ({"psd_failures": 1.0}, "O2_psd_failures", CONTINUE),
        (
            {"same_player_max_block_deviation": 1e-6},
            "O3_same_player_block_deviation",
            CONTINUE,
        ),
        (
            {"production_serving_was_affected": True},
            "O6_incumbent_serving_unaffected",
            DISABLED,
        ),
    ],
)
def test_an_operational_gate_control(policy, run_change, gate_name, decision):
    window = eligible_window(runs=synthetic_runs(days=30, **run_change))
    report = lp.evaluate(policy, window)

    tripped = {gate.name for gate in report.operational if gate.failed}
    assert tripped == {gate_name}
    assert report.decision == decision


def test_the_candidate_fallback_rate_gate_control(policy):
    events = synthetic_events(
        games=500, days=30, per_leg_count=6, candidate_advantage=0.06
    )
    # One per cent of rows, against a frozen bound of 0.005.
    fell_back = events.index[: len(events) // 100]
    events.loc[fell_back, "fell_back"] = True
    report = lp.evaluate(policy, eligible_window(events=events))

    tripped = {gate.name for gate in report.operational if gate.failed}
    assert tripped == {"O4_candidate_fallback_rate"}
    assert report.decision == CONTINUE


def test_a_fallback_row_is_never_scored_as_a_candidate_row(policy):
    """A row the candidate did not price may not be credited to the candidate."""
    events = synthetic_events(
        games=500, days=30, per_leg_count=6, candidate_advantage=0.06
    )
    scored = lp.evaluate(policy, eligible_window(events=events))

    marked = events.copy()
    victims = marked.index[:1000]
    marked.loc[victims, "fell_back"] = True
    # The served column on a fallback row carries the incumbent's number.
    marked.loc[victims, "candidate_probability"] = marked.loc[
        victims, "incumbent_probability"
    ]
    report = lp.evaluate(policy, eligible_window(events=marked))

    aggregate = next(
        reading
        for reading in report.readings
        if reading.slice_name == "aggregate" and reading.metric == "brier"
    )
    assert aggregate.events == len(events) - len(victims)
    assert aggregate.candidate != pytest.approx(
        next(
            reading.candidate
            for reading in scored.readings
            if reading.slice_name == "aggregate" and reading.metric == "brier"
        )
    )


@pytest.mark.parametrize(
    "dropped",
    [
        "final_model_spec_sha256",
        "factor_spec_hash",
        "artifact_hash_or_version",
        "production_code_sha",
    ],
)
def test_an_incomplete_provenance_preimage_fails_the_provenance_gate(policy, dropped):
    window = eligible_window(provenance=synthetic_provenance(drop=(dropped,)))
    report = lp.evaluate(policy, window)

    gate = next(
        gate for gate in report.operational if gate.name == "O5_provenance_completeness"
    )
    assert gate.failed
    assert dropped in gate.detail
    assert report.decision == CONTINUE


def test_a_fingerprint_with_no_recorded_preimage_fails_the_provenance_gate(policy):
    """A column is not provenance. The fingerprint must resolve to a payload."""
    window = eligible_window(provenance={})
    report = lp.evaluate(policy, window)

    gate = next(
        gate for gate in report.operational if gate.name == "O5_provenance_completeness"
    )
    assert gate.failed
    assert "no recorded preimage" in gate.detail
    assert report.decision == CONTINUE


def test_a_row_served_by_something_other_than_the_incumbent_is_a_safety_failure(
    policy,
):
    events = synthetic_events(
        games=500, days=30, per_leg_count=6, candidate_advantage=0.06
    )
    events.loc[events.index[0], "served_model"] = "candidate"
    report = lp.evaluate(policy, eligible_window(events=events))

    assert report.decision == DISABLED
    gate = next(
        gate
        for gate in report.operational
        if gate.name == "O6_incumbent_serving_unaffected"
    )
    assert gate.observed["served_model_values_other_than_incumbent"] == ["candidate"]


def test_a_row_flagged_published_is_a_publishing_violation(policy):
    events = synthetic_events(
        games=500, days=30, per_leg_count=6, candidate_advantage=0.06
    )
    events.loc[events.index[0], "published"] = True
    report = lp.evaluate(policy, eligible_window(events=events))

    assert report.decision == DISABLED


def test_a_run_that_recorded_no_same_player_deviation_does_not_pass_that_gate(policy):
    """Silence is not evidence the blocks were preserved."""
    runs = synthetic_runs(days=30)
    for run in runs:
        run["same_player_max_block_deviation"] = None
    report = lp.evaluate(policy, eligible_window(runs=runs))

    gate = next(
        gate
        for gate in report.operational
        if gate.name == "O3_same_player_block_deviation"
    )
    assert gate.passed is None
    assert not gate.failed
    assert report.decision == CONTINUE
    assert "O3_same_player_block_deviation" in report.reason


# ======================================================================
# negative controls: proper score gates
# ======================================================================


@pytest.mark.parametrize("leg_count", [2, 3, 4])
def test_a_single_worse_leg_count_rejects_the_candidate(policy, leg_count):
    """Non-inferiority is required on every declared leg count separately."""
    window = eligible_window(
        events=synthetic_events(
            games=500,
            days=30,
            per_leg_count=6,
            candidate_advantage=0.06,
            worse_leg_counts=(leg_count,),
        )
    )
    report = lp.evaluate(policy, window)

    assert report.decision == REJECTED
    failed = {gate.name for gate in report.scoring if gate.failed}
    assert failed >= {f"{leg_count}_leg_brier", f"{leg_count}_leg_log_loss"}
    for other in {2, 3, 4} - {leg_count}:
        assert f"{other}_leg_brier" not in failed


def test_an_equal_candidate_fails_the_aggregate_point_estimate(policy):
    """The aggregate gate is improvement, not non-inferiority.

    The frozen standard states the point estimate must satisfy ``candidate <=
    incumbent``; an interval that merely contains zero is not enough. A
    candidate a hair worse than the incumbent on every row has an upper bound
    well inside the tolerance and must still fail.
    """
    window = eligible_window(
        events=synthetic_events(
            games=500, days=30, per_leg_count=6, candidate_advantage=-0.0005
        )
    )
    report = lp.evaluate(policy, window)

    aggregate = next(
        gate for gate in report.scoring if gate.name == "aggregate_brier"
    )
    assert aggregate.observed["delta"] > 0.0
    assert aggregate.failed
    assert report.decision == REJECTED


def test_the_score_slices_are_exactly_the_frozen_ones(policy):
    report = lp.evaluate(policy, eligible_window())
    assert {(r.metric, r.slice_name) for r in report.readings} == {
        ("brier", "aggregate"),
        ("log_loss", "aggregate"),
        ("brier", "2_leg"),
        ("log_loss", "2_leg"),
        ("brier", "3_leg"),
        ("log_loss", "3_leg"),
        ("brier", "4_leg"),
        ("log_loss", "4_leg"),
    }


def test_the_bootstrap_resamples_games_and_not_rows(policy):
    """Clustering matters: rows inside a game share that game's latent state.

    Resampling rows would understate the interval, so the same evidence
    arranged as one game must give a wider interval than when it is arranged
    as many -- which is the observable signature of the cluster being honoured.
    """
    rows = synthetic_events(
        games=60, days=30, per_leg_count=6, candidate_advantage=0.02
    )
    clustered = lp.score_reading(
        rows,
        metric="brier",
        slice_name="aggregate",
        draws=2000,
        seed=73,
        level=0.95,
        floor=1e-6,
    )
    one_game = rows.assign(game_id="002260" + "0" * 5)
    collapsed = lp.score_reading(
        one_game,
        metric="brier",
        slice_name="aggregate",
        draws=2000,
        seed=73,
        level=0.95,
        floor=1e-6,
    )

    assert collapsed.delta == pytest.approx(clustered.delta)
    # One cluster cannot vary: every draw is the whole sample.
    assert collapsed.delta_ci_lower == pytest.approx(collapsed.delta_ci_upper)
    assert clustered.delta_ci_upper > clustered.delta_ci_lower


def test_the_bootstrap_is_deterministic_under_the_frozen_seed(policy):
    first = lp.evaluate(policy, eligible_window())
    second = lp.evaluate(policy, eligible_window())
    assert [reading.payload() for reading in first.readings] == [
        reading.payload() for reading in second.readings
    ]


# ======================================================================
# negative controls: dependence gates
# ======================================================================


@pytest.mark.parametrize(
    "space, gate_name",
    [
        (lp.LATENT_SPACE, "D1_global_latent_dependence_rmse"),
        (lp.COUNT_SPACE, "D2_count_space_dependence_rmse"),
    ],
)
def test_a_worse_dependence_rmse_rejects_the_candidate(policy, space, gate_name):
    good = synthetic_dependence(games=40, candidate_error=0.01, incumbent_error=0.05)
    bad = synthetic_dependence(games=40, candidate_error=0.09, incumbent_error=0.05)
    merged = [
        {**entry, space: bad[index][space]} for index, entry in enumerate(good)
    ]
    report = lp.evaluate(policy, eligible_window(dependence=merged))

    failed = {gate.name for gate in report.dependence if gate.failed}
    assert gate_name in failed
    assert report.decision == REJECTED


def test_an_equal_dependence_rmse_is_not_an_improvement(policy):
    """D1 and D2 are strict improvements, not non-inferiority."""
    equal = synthetic_dependence(games=40, candidate_error=0.05, incumbent_error=0.05)
    report = lp.evaluate(policy, eligible_window(dependence=equal))

    assert {gate.name for gate in report.dependence if gate.failed} >= {
        "D1_global_latent_dependence_rmse",
        "D2_count_space_dependence_rmse",
    }
    assert report.decision == REJECTED


@pytest.mark.parametrize(
    "bucket",
    [
        "opponent_ast_ast",
        "opponent_pts_reb",
        "opponent_fg3m_reb",
        "passer_ast_teammate_pts",
        "teammate_pts_reb",
    ],
)
def test_worsening_one_protected_bucket_rejects_the_candidate(policy, bucket):
    window = eligible_window(
        dependence=synthetic_dependence(
            games=40,
            candidate_error=0.01,
            incumbent_error=0.05,
            worsened_buckets=(bucket,),
        )
    )
    report = lp.evaluate(policy, window)

    gate = next(
        gate
        for gate in report.dependence
        if gate.name == "D3_protected_buckets_may_not_worsen"
    )
    assert gate.failed
    assert bucket in gate.detail
    assert report.decision == REJECTED


def test_the_protected_bucket_list_is_the_frozen_one(policy):
    expected = set(
        policy["dependence_gates"]["D3_protected_buckets_may_not_worsen"][
            "protected_buckets"
        ]
    )
    report = lp.evaluate(policy, eligible_window())
    assert set(report.dependence_detail["protected_buckets"]) == expected


def test_the_reported_but_ungated_buckets_are_reported(policy):
    report = lp.evaluate(policy, eligible_window())
    reported = report.dependence_detail["explicitly_reported_not_gated"]

    assert set(reported) == {"teammate_ast_ast", "teammate_reb_reb"}
    for bucket, spaces in reported.items():
        for space in (lp.LATENT_SPACE, lp.COUNT_SPACE):
            assert spaces[space] is not None, bucket
            assert spaces[space]["worsening_z"] is not None


def test_worsening_an_ungated_bucket_does_not_reject_the_candidate(policy):
    """D4's buckets are reported and not gated, which must be observable."""
    window = eligible_window(
        dependence=synthetic_dependence(
            games=40,
            candidate_error=0.001,
            incumbent_error=0.002,
            worsened_buckets=("teammate_ast_ast",),
        )
    )
    report = lp.evaluate(policy, window)

    gate = next(
        gate
        for gate in report.dependence
        if gate.name == "D3_protected_buckets_may_not_worsen"
    )
    assert gate.passed is True
    reported = report.dependence_detail["explicitly_reported_not_gated"]
    assert reported["teammate_ast_ast"][lp.LATENT_SPACE]["worsening_z"] > 0.0


def test_all_twelve_declared_cross_player_buckets_are_evaluated(policy):
    declared = set(policy["tracked_metrics"]["dependence_buckets"])
    report = lp.evaluate(policy, eligible_window())

    assert set(report.dependence_detail["all_declared_buckets"]) == declared
    assert len(declared) == 12
    for space in (lp.LATENT_SPACE, lp.COUNT_SPACE):
        errors = report.dependence_detail[space]["by_model"][lp.CANDIDATE][
            "bucket_errors"
        ]
        assert set(errors) == declared


def test_a_missing_dependence_space_leaves_its_gate_unevaluated(policy):
    """No reading is not a pass, and an unevaluated gate blocks eligibility."""
    window = eligible_window(
        dependence=synthetic_dependence(
            games=40,
            candidate_error=0.01,
            incumbent_error=0.05,
            spaces=(lp.COUNT_SPACE,),
        )
    )
    report = lp.evaluate(policy, window)

    gate = next(
        gate
        for gate in report.dependence
        if gate.name == "D1_global_latent_dependence_rmse"
    )
    assert gate.passed is None
    assert report.decision == CONTINUE
    assert "D1_global_latent_dependence_rmse" in report.reason


def test_buckets_pool_by_pair_count_across_games(policy):
    """A game contributing twice the pairs carries twice the weight."""
    entries = [
        {
            lp.LATENT_SPACE: {
                lp.CANDIDATE: {
                    bucket: 0.0 for bucket in lp.CROSS_PLAYER_BUCKETS
                }
                | {"_same_team_pairs": 1.0, "_cross_team_pairs": 1.0}
            }
        },
        {
            lp.LATENT_SPACE: {
                lp.CANDIDATE: {
                    bucket: 0.3 for bucket in lp.CROSS_PLAYER_BUCKETS
                }
                | {"_same_team_pairs": 3.0, "_cross_team_pairs": 3.0}
            }
        },
    ]
    pooled = lp.pooled_buckets(entries, space=lp.LATENT_SPACE, arm=lp.CANDIDATE)
    for bucket in lp.CROSS_PLAYER_BUCKETS:
        assert pooled[bucket] == pytest.approx(0.225)


def test_a_bucket_z_needs_more_than_one_game(policy):
    single = synthetic_dependence(
        games=1, candidate_error=0.01, incumbent_error=0.05
    )
    report = lp.evaluate(policy, eligible_window(dependence=single))

    gate = next(
        gate
        for gate in report.dependence
        if gate.name == "D3_protected_buckets_may_not_worsen"
    )
    assert gate.passed is None
    assert report.decision == CONTINUE


# ======================================================================
# no lookahead
# ======================================================================


def test_an_ungraded_row_is_not_evidence(policy):
    """Rows whose games have not settled must not count toward any minimum."""
    events = synthetic_events(
        games=500, days=30, per_leg_count=6, candidate_advantage=0.06
    )
    pending = events.copy()
    pending["realized"] = np.nan

    window = eligible_window(events=pending)
    report = lp.evaluate(policy, window)

    assert report.decision == CONTINUE
    assert report.evidence.payload()["unmet"] == [
        "A_games",
        "B_joint_events",
        "B_joint_events_2_leg",
        "B_joint_events_3_leg",
        "B_joint_events_4_leg",
        "C_calendar",
    ]


def test_a_partly_settled_window_counts_only_the_settled_part(policy):
    events = synthetic_events(
        games=500, days=30, per_leg_count=6, candidate_advantage=0.06
    )
    half = events.copy()
    half.loc[half.index[len(half) // 2 :], "realized"] = np.nan

    window = eligible_window(events=half)
    report = lp.evaluate(policy, window)

    assert len(window.graded) == len(events) // 2
    assert report.decision == CONTINUE
