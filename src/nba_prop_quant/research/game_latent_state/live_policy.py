"""Evaluate accumulated live shadow evidence against the frozen policy.

SHADOW / MONITORING ONLY. This module renders a decision about what live
evidence *would* justify opening a human promotion review. It promotes
nothing, publishes nothing, and no caller may act on its output beyond
reporting it.

WHY THIS EXISTS BEFORE THERE IS ANY LIVE EVIDENCE
-------------------------------------------------

``research/final_model/live_shadow_promotion_policy.json`` was frozen before a
single live slate ran, which is what stops a threshold from being tuned to
whatever the live sample happens to show. The policy is only half the
guarantee: an evaluator written *after* the evidence arrives can reach any
conclusion it likes while still quoting frozen numbers, by choosing which
rows to count, how to pair them, or which gate to treat as inapplicable.
Writing it now, against synthetic evidence, fixes those choices while nobody
knows which way they would cut.

THIS MODULE STATES NO THRESHOLD OF ITS OWN
------------------------------------------

Every bound, every minimum, the bootstrap draw count, the seed, the gate list,
the bucket lists and the decision vocabulary are read from the frozen policy.
A missing field raises :class:`PolicyIncomplete` rather than falling back to a
default, because a default here would be a threshold this module invented. The
tests assert that no numeric literal in this file is a gate bound.

WHAT A GATE CAN ANSWER
----------------------

Three answers, not two. ``True`` and ``False`` mean the gate was evaluated.
``None`` means the evidence to evaluate it is not present -- an empty window,
a leg count with no graded rows, a dependence reading the shadow did not
record. ``None`` is never read as a pass. Before the minimum live-evidence
requirement is met the policy forbids rendering a gate verdict at all, so the
scoring and dependence gates are reported as readings with ``None`` verdicts
and the decision is ``CONTINUE_SHADOW``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .validation import DEPENDENCE_BUCKETS, brier_score, log_loss

#: Where the frozen policy lives, relative to the project root.
POLICY_RELATIVE = Path("research") / "final_model" / "live_shadow_promotion_policy.json"

#: The arms a shadow row carries. ``independence`` is reference only: the
#: policy gates the candidate against the incumbent and says so explicitly.
CANDIDATE = "candidate"
INCUMBENT = "incumbent"
INDEPENDENCE = "independence"

#: Probability column per arm on an accumulated shadow row.
PROBABILITY_COLUMNS: Mapping[str, str] = {
    CANDIDATE: "candidate_probability",
    INCUMBENT: "incumbent_probability",
    INDEPENDENCE: "independence_probability",
}

#: The cross-player buckets. The same-player blocks are deliberately excluded:
#: the policy tracks them separately under O3 because they must be preserved
#: exactly rather than improved, so scoring them as a dependence improvement
#: would be reporting the incumbent's own pinned numbers as a candidate win.
CROSS_PLAYER_BUCKETS: tuple[str, ...] = tuple(
    name for name, kind, _ in DEPENDENCE_BUCKETS if kind != "same_player"
)

#: Which pair-count weight pools a bucket across games.
BUCKET_PAIR_WEIGHT: Mapping[str, str] = {
    name: ("_same_team_pairs" if kind == "same_team" else "_cross_team_pairs")
    for name, kind, _ in DEPENDENCE_BUCKETS
    if kind != "same_player"
}

#: The two dependence spaces the policy gates, D1 and D2 respectively.
LATENT_SPACE = "latent"
COUNT_SPACE = "count"


class PolicyIncomplete(RuntimeError):
    """The frozen policy does not carry a field this evaluator must read.

    Raised rather than defaulted. The policy's own immutability clause calls
    "a reference to a field the shadow log does not carry" a correctness bug,
    and the same applies in the other direction: an evaluator that silently
    substitutes its own number for a missing policy field has stopped being an
    evaluator of that policy.
    """


# ======================================================================
# the frozen policy, read strictly
# ======================================================================


def load_policy(project_root: Path | str = ".") -> Mapping[str, Any]:
    path = Path(project_root) / POLICY_RELATIVE
    if not path.is_file():
        raise PolicyIncomplete(f"no frozen policy at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(policy: Mapping[str, Any], *keys: str) -> Any:
    """Walk ``keys`` or raise. Never returns a default."""
    node: Any = policy
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(node, Mapping) or key not in node:
            raise PolicyIncomplete(
                "the frozen policy does not carry "
                + ".".join(walked)
                + ", so this gate cannot be evaluated as written"
            )
        node = node[key]
    return node


def decision_vocabulary(policy: Mapping[str, Any]) -> tuple[str, ...]:
    """The only decisions a report may reach, in the policy's own words."""
    states = _require(policy, "decision_states", "states")
    return tuple(str(state) for state in states)


# ======================================================================
# the accumulated evidence window
# ======================================================================


@dataclass(frozen=True)
class EvidenceWindow:
    """Everything the live monitoring state has accumulated so far.

    ``events`` is one row per graded joint event, pooled over every slate the
    shadow has run. ``provenance`` maps a fingerprint to the payload it is the
    digest of, which is what lets O5 resolve a row's identities after the fact
    rather than trusting a column. ``dependence`` is one entry per evaluated
    game. ``runs`` is one entry per shadow run, which is where the
    publishing, PSD and serving-safety readings live.
    """

    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    provenance: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    dependence: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    runs: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    @property
    def graded(self) -> pd.DataFrame:
        """Rows with a realized outcome. An ungraded row is not evidence."""
        if self.events.empty or "realized" not in self.events.columns:
            return self.events.iloc[0:0]
        return self.events.loc[self.events["realized"].notna()].copy()

    @property
    def graded_games(self) -> int:
        frame = self.graded
        if frame.empty or "game_id" not in frame.columns:
            return 0
        return int(frame["game_id"].nunique())

    @property
    def calendar_days(self) -> int:
        frame = self.graded
        if frame.empty or "slate_date" not in frame.columns:
            return 0
        return int(frame["slate_date"].nunique())

    def graded_by_leg_count(self) -> dict[int, int]:
        frame = self.graded
        if frame.empty or "leg_count" not in frame.columns:
            return {}
        counts = frame["leg_count"].dropna().astype(int).value_counts()
        return {int(legs): int(count) for legs, count in counts.items()}


# ======================================================================
# gates
# ======================================================================


@dataclass
class Gate:
    """One statement the policy makes, evaluated or declared unevaluable."""

    name: str
    passed: bool | None
    requirement: str
    observed: Any = None
    threshold: Any = None
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.passed is False

    @property
    def evaluated(self) -> bool:
        return self.passed is not None

    def payload(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "name": self.name,
            "observed": self.observed,
            "passed": self.passed,
            "requirement": self.requirement,
            "threshold": self.threshold,
        }


def _sum_over_runs(runs: Iterable[Mapping[str, Any]], *keys: str) -> float:
    total = 0.0
    for run in runs:
        for key in keys:
            value = run.get(key)
            if value is not None:
                total += float(value)
                break
    return total


# ----------------------------------------------------------------------
# operational gates O1 - O6
# ----------------------------------------------------------------------


def operational_gates(
    policy: Mapping[str, Any], window: EvidenceWindow
) -> list[Gate]:
    """The six gates that decide whether the shadow is safe to keep running.

    These are evaluated on whatever evidence exists, including none: a
    publishing violation or a PSD failure is a safety fact and the policy does
    not make it wait for the minimum sample.
    """
    gates = _require(policy, "operational_gates")
    runs = list(window.runs)
    graded = window.graded

    out: list[Gate] = []

    # --- O1 publishing violations -------------------------------------
    limit = float(_require(gates, "O1_shadow_publishing_violations", "max"))
    published_rows = _sum_over_runs(runs, "rows_published")
    published_flags = (
        int(graded["published"].fillna(False).astype(bool).sum())
        if not graded.empty and "published" in graded.columns
        else 0
    )
    violations = published_rows + published_flags
    out.append(
        Gate(
            name="O1_shadow_publishing_violations",
            passed=violations <= limit,
            requirement=f"published rows must not exceed {limit:g}",
            observed=violations,
            threshold=limit,
            detail=(
                f"{published_rows:g} rows reported published across "
                f"{len(runs)} run(s) and {published_flags} row(s) flagged "
                "published in the accumulated log"
            ),
        )
    )

    # --- O2 PSD failures -----------------------------------------------
    limit = float(_require(gates, "O2_psd_failures", "max"))
    psd_failures = _sum_over_runs(runs, "psd_failures")
    out.append(
        Gate(
            name="O2_psd_failures",
            passed=psd_failures <= limit,
            requirement=f"PSD failures must not exceed {limit:g}",
            observed=psd_failures,
            threshold=limit,
            detail=f"{psd_failures:g} PSD failure(s) across {len(runs)} run(s)",
        )
    )

    # --- O3 same-player block deviation --------------------------------
    limit = float(_require(gates, "O3_same_player_block_deviation", "max"))
    deviations = [
        float(run["same_player_max_block_deviation"])
        for run in runs
        if run.get("same_player_max_block_deviation") is not None
    ]
    worst = max(deviations) if deviations else None
    out.append(
        Gate(
            name="O3_same_player_block_deviation",
            # Scoped to every evaluated game, not an average, so the maximum
            # is the reading. With no reading the gate is unevaluable rather
            # than passed: a run that recorded no deviation at all did not
            # prove the blocks were preserved.
            passed=None if worst is None else worst <= limit,
            requirement=(
                f"every evaluated game must hold the same-player blocks to "
                f"within {limit:g}"
            ),
            observed=worst,
            threshold=limit,
            detail=(
                "no run recorded a same-player deviation"
                if worst is None
                else f"worst deviation {worst:.3e} over {len(deviations)} run(s)"
            ),
        )
    )

    # --- O4 candidate fallback rate ------------------------------------
    limit = float(_require(gates, "O4_candidate_fallback_rate", "max"))
    if graded.empty or "fell_back" not in graded.columns:
        rate: float | None = None
        fell_back = 0
    else:
        fell_back = int(graded["fell_back"].fillna(False).astype(bool).sum())
        rate = fell_back / len(graded)
    out.append(
        Gate(
            name="O4_candidate_fallback_rate",
            passed=None if rate is None else rate <= limit,
            requirement=f"candidate fallback rate must not exceed {limit:g}",
            observed=rate,
            threshold=limit,
            detail=(
                "no graded rows, so there is no rate to read"
                if rate is None
                else f"{fell_back} fallback(s) in {len(graded)} graded row(s)"
            ),
        )
    )

    # --- O5 provenance completeness ------------------------------------
    required_fields = tuple(
        str(name)
        for name in _require(policy, "operational_gates", "O5_provenance_completeness", "required_fields")
    )
    fraction_required = float(
        _require(
            policy,
            "operational_gates",
            "O5_provenance_completeness",
            "fraction_of_graded_candidate_rows_required",
        )
    )
    out.append(
        _provenance_gate(
            window=window,
            required_fields=required_fields,
            fraction_required=fraction_required,
        )
    )

    # --- O6 incumbent serving unaffected -------------------------------
    requirement = str(
        _require(policy, "operational_gates", "O6_incumbent_serving_unaffected", "requirement")
    )
    affected = [
        run for run in runs if run.get("production_serving_was_affected") is True
    ]
    served_other = (
        sorted(
            str(value)
            for value in graded["served_model"].dropna().unique()
            if str(value).lower() != INCUMBENT
        )
        if not graded.empty and "served_model" in graded.columns
        else []
    )
    out.append(
        Gate(
            name="O6_incumbent_serving_unaffected",
            passed=not affected and not served_other,
            requirement=requirement,
            observed={
                "runs_reporting_affected_serving": len(affected),
                "served_model_values_other_than_incumbent": served_other,
            },
            threshold=0,
            detail=(
                "every run reports the incumbent served and unaffected"
                if not affected and not served_other
                else "a run reported production serving was affected, or a "
                "row was served by something other than the incumbent"
            ),
        )
    )

    return out


def _provenance_gate(
    *,
    window: EvidenceWindow,
    required_fields: tuple[str, ...],
    fraction_required: float,
) -> Gate:
    """Every graded candidate row must resolve to all four identities.

    The policy is explicit that the gate is "all four resolve for every graded
    candidate row, not that all four appear as literal columns": the row
    carries a fingerprint, and the fingerprint must have a recorded preimage
    carrying the identities. A row whose fingerprint has no preimage fails,
    because its identities cannot be established after the fact.
    """
    graded = window.graded
    if graded.empty:
        return Gate(
            name="O5_provenance_completeness",
            passed=None,
            requirement=(
                f"{fraction_required:g} of graded candidate rows must resolve "
                f"to {', '.join(required_fields)}"
            ),
            observed=None,
            threshold=fraction_required,
            detail="no graded rows, so there is no provenance to resolve",
        )

    if "fell_back" in graded.columns:
        rows = graded.loc[~graded["fell_back"].fillna(False).astype(bool)]
    else:
        rows = graded

    if rows.empty:
        return Gate(
            name="O5_provenance_completeness",
            passed=None,
            requirement=(
                f"{fraction_required:g} of graded candidate rows must resolve "
                f"to {', '.join(required_fields)}"
            ),
            observed=None,
            threshold=fraction_required,
            detail="every graded row fell back, so no candidate row was produced",
        )

    unresolved: dict[str, int] = {}
    resolved = 0
    for fingerprint, block in rows.groupby(
        rows.get("provenance_fingerprint", pd.Series(dtype=object)).fillna("")
    ):
        payload = window.provenance.get(str(fingerprint))
        if not payload:
            unresolved["no recorded preimage"] = unresolved.get(
                "no recorded preimage", 0
            ) + len(block)
            continue
        absent = [
            name
            for name in required_fields
            if not str(payload.get(name) or "").strip()
        ]
        if absent:
            key = "missing " + ", ".join(absent)
            unresolved[key] = unresolved.get(key, 0) + len(block)
            continue
        resolved += len(block)

    fraction = resolved / len(rows)
    return Gate(
        name="O5_provenance_completeness",
        passed=fraction >= fraction_required,
        requirement=(
            f"{fraction_required:g} of graded candidate rows must resolve to "
            f"{', '.join(required_fields)}"
        ),
        observed=fraction,
        threshold=fraction_required,
        detail=(
            f"{resolved} of {len(rows)} graded candidate row(s) resolve"
            + (f"; unresolved: {unresolved}" if unresolved else "")
        ),
    )


# ----------------------------------------------------------------------
# minimum live evidence
# ----------------------------------------------------------------------


@dataclass
class MinimumEvidence:
    satisfied: bool
    requirements: list[Gate]

    def payload(self) -> dict[str, Any]:
        return {
            "requirements": [gate.payload() for gate in self.requirements],
            "satisfied": self.satisfied,
            "unmet": sorted(
                gate.name for gate in self.requirements if not gate.passed
            ),
        }


def minimum_evidence(
    policy: Mapping[str, Any], window: EvidenceWindow
) -> MinimumEvidence:
    """A, B and C must all hold before any gate verdict is rendered."""
    minimum = _require(policy, "minimum_live_evidence")

    games_required = int(_require(minimum, "A_games", "min_live_nba_games_graded"))
    events_required = int(
        _require(minimum, "B_joint_events", "min_live_graded_joint_events")
    )
    by_leg_required = {
        int(legs): int(count)
        for legs, count in _require(
            minimum, "B_joint_events", "by_leg_count"
        ).items()
    }
    days_required = int(_require(minimum, "C_calendar", "min_regular_season_days"))

    graded_events = len(window.graded)
    by_leg = window.graded_by_leg_count()

    requirements = [
        Gate(
            name="A_games",
            passed=window.graded_games >= games_required,
            requirement=f"at least {games_required} graded live NBA games",
            observed=window.graded_games,
            threshold=games_required,
        ),
        Gate(
            name="B_joint_events",
            passed=graded_events >= events_required,
            requirement=f"at least {events_required} graded joint events",
            observed=graded_events,
            threshold=events_required,
        ),
    ]
    for legs in sorted(by_leg_required):
        observed = by_leg.get(legs, 0)
        requirements.append(
            Gate(
                name=f"B_joint_events_{legs}_leg",
                passed=observed >= by_leg_required[legs],
                requirement=(
                    f"at least {by_leg_required[legs]} graded {legs}-leg events"
                ),
                observed=observed,
                threshold=by_leg_required[legs],
            )
        )
    requirements.append(
        Gate(
            name="C_calendar",
            passed=window.calendar_days >= days_required,
            requirement=f"at least {days_required} regular-season days",
            observed=window.calendar_days,
            threshold=days_required,
        )
    )

    return MinimumEvidence(
        satisfied=all(bool(gate.passed) for gate in requirements),
        requirements=requirements,
    )


# ----------------------------------------------------------------------
# proper scores
# ----------------------------------------------------------------------


@dataclass
class ScoreReading:
    """Candidate and incumbent on one score over one slice of the window."""

    metric: str
    slice_name: str
    events: int
    candidate: float | None
    incumbent: float | None
    delta: float | None
    delta_ci_lower: float | None
    delta_ci_upper: float | None
    draws: int

    def payload(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "delta": self.delta,
            "delta_ci_lower": self.delta_ci_lower,
            "delta_ci_upper": self.delta_ci_upper,
            "draws": self.draws,
            "events": self.events,
            "incumbent": self.incumbent,
            "metric": self.metric,
            "slice": self.slice_name,
        }


def _paired_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows where both arms answered, which is the only paired unit.

    A row where the candidate fell back carries the incumbent's number in the
    served column and no candidate number. Pairing it would credit the
    candidate with the incumbent's score on exactly the rows it failed, which
    is the one way this comparison could lie.
    """
    needed = [
        PROBABILITY_COLUMNS[CANDIDATE],
        PROBABILITY_COLUMNS[INCUMBENT],
        "realized",
        "game_id",
    ]
    absent = [column for column in needed if column not in frame.columns]
    if absent:
        return frame.iloc[0:0]
    rows = frame.dropna(subset=needed).copy()
    if "fell_back" in rows.columns:
        rows = rows.loc[~rows["fell_back"].fillna(False).astype(bool)]
    return rows


def _row_losses(
    rows: pd.DataFrame, metric: str, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row loss for each arm.

    Both scores the policy gates are means of a per-row contribution, which is
    what makes the game-clustered bootstrap exact without resampling the frame
    itself: a resampled score is the resampled sum of these divided by the
    resampled row count. The whole-array scores are computed through
    :func:`brier_score` and :func:`log_loss` so the single definition of each
    metric still lives in the validation layer.
    """
    outcome = rows["realized"].to_numpy(dtype=float)
    candidate = rows[PROBABILITY_COLUMNS[CANDIDATE]].to_numpy(dtype=float)
    incumbent = rows[PROBABILITY_COLUMNS[INCUMBENT]].to_numpy(dtype=float)

    if metric == "brier":
        return (candidate - outcome) ** 2, (incumbent - outcome) ** 2

    def losses(probability: np.ndarray) -> np.ndarray:
        p = np.clip(probability, floor, 1.0 - floor)
        return -(outcome * np.log(p) + (1.0 - outcome) * np.log1p(-p))

    return losses(candidate), losses(incumbent)


def _scores(rows: pd.DataFrame, metric: str, floor: float) -> tuple[float, float]:
    outcome = rows["realized"].to_numpy(dtype=float)
    candidate = rows[PROBABILITY_COLUMNS[CANDIDATE]].to_numpy(dtype=float)
    incumbent = rows[PROBABILITY_COLUMNS[INCUMBENT]].to_numpy(dtype=float)
    if metric == "brier":
        return brier_score(candidate, outcome), brier_score(incumbent, outcome)
    return (
        log_loss(candidate, outcome, floor=floor),
        log_loss(incumbent, outcome, floor=floor),
    )


def score_reading(
    rows: pd.DataFrame,
    *,
    metric: str,
    slice_name: str,
    draws: int,
    seed: int,
    level: float,
    floor: float,
) -> ScoreReading:
    """One paired, game-clustered bootstrap reading of candidate minus incumbent.

    Games are the resampling unit because joint-event rows inside one game
    share that game's latent state and are not independent. The pairing is
    what makes the interval a statement about the difference rather than about
    two separately noisy numbers.
    """
    if rows.empty:
        return ScoreReading(
            metric=metric,
            slice_name=slice_name,
            events=0,
            candidate=None,
            incumbent=None,
            delta=None,
            delta_ci_lower=None,
            delta_ci_upper=None,
            draws=0,
        )

    candidate, incumbent = _scores(rows, metric, floor)

    # The resampled difference in closed form. A score is the sum of its
    # per-row losses over the row count, so a resampled score is the sum of
    # the drawn games' loss sums over the sum of their row counts. Collapsing
    # each game to three numbers once means the 5000 draws are array work
    # instead of 5000 frame rebuilds, which is the difference between a
    # monitoring step that finishes and one that does not.
    candidate_loss, incumbent_loss = _row_losses(rows, metric, floor)
    cluster = rows.groupby("game_id", sort=True).ngroup().to_numpy()
    game_count = int(cluster.max()) + 1
    candidate_sum = np.bincount(cluster, weights=candidate_loss, minlength=game_count)
    incumbent_sum = np.bincount(cluster, weights=incumbent_loss, minlength=game_count)
    row_count = np.bincount(cluster, minlength=game_count).astype(float)

    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, game_count, size=(draws, game_count))
    deltas = (
        candidate_sum[drawn].sum(axis=1) - incumbent_sum[drawn].sum(axis=1)
    ) / row_count[drawn].sum(axis=1)

    tail = (1.0 - level) / 2.0
    lower, upper = np.percentile(deltas, [100.0 * tail, 100.0 * (1.0 - tail)])

    return ScoreReading(
        metric=metric,
        slice_name=slice_name,
        events=len(rows),
        candidate=candidate,
        incumbent=incumbent,
        delta=candidate - incumbent,
        delta_ci_lower=float(lower),
        delta_ci_upper=float(upper),
        draws=draws,
    )


def _bootstrap_settings(policy: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = _require(policy, "proper_score_gates", "bootstrap")
    for name in ("draws", "seed", "paired", "cluster", "ci"):
        if name not in bootstrap:
            raise PolicyIncomplete(
                f"the frozen policy's bootstrap block does not carry {name}"
            )
    if not bootstrap["paired"]:
        raise PolicyIncomplete(
            "the frozen policy's bootstrap is not paired, which this "
            "evaluator cannot honour as written"
        )
    if str(bootstrap["cluster"]) != "game":
        raise PolicyIncomplete(
            "the frozen policy clusters the bootstrap on "
            f"{bootstrap['cluster']!r}, which this evaluator does not implement"
        )
    return {
        "draws": int(bootstrap["draws"]),
        "seed": int(bootstrap["seed"]),
        # "95 per cent, two-sided, percentile" -- the level is stated in the
        # policy's prose, and 95 is the only number in it.
        "level": 0.95,
        "ci": str(bootstrap["ci"]),
    }


def proper_score_gates(
    policy: Mapping[str, Any],
    window: EvidenceWindow,
    *,
    render_verdicts: bool,
    log_loss_floor: float,
) -> tuple[list[Gate], list[ScoreReading]]:
    """Aggregate improvement plus leg-level non-inferiority.

    ``render_verdicts`` is the minimum-evidence switch. Before the requirement
    is met the readings are still computed and reported -- that is what makes
    the report monitoring -- but every verdict is ``None``, because the policy
    says a report produced before the requirement is met may not render one.
    """
    aggregate = _require(policy, "proper_score_gates", "aggregate")
    by_leg = _require(policy, "proper_score_gates", "by_leg_count")
    settings = _bootstrap_settings(policy)

    rows = _paired_rows(window.graded)

    readings: list[ScoreReading] = []
    gates: list[Gate] = []

    for metric in ("brier", "log_loss"):
        bound = float(_require(aggregate, metric, "upper_95_ci_of_delta_max"))
        reading = score_reading(
            rows,
            metric=metric,
            slice_name="aggregate",
            draws=settings["draws"],
            seed=settings["seed"],
            level=settings["level"],
            floor=log_loss_floor,
        )
        readings.append(reading)

        if reading.delta is None or not render_verdicts:
            passed: bool | None = None
            detail = (
                "no paired rows"
                if reading.delta is None
                else "minimum live evidence not met, so no verdict is rendered"
            )
        else:
            # Both halves of the frozen standard: the point estimate must not
            # be worse, and the upper bound of the paired interval must sit
            # inside the frozen tolerance.
            passed = (
                reading.delta <= 0.0
                and reading.delta_ci_upper is not None
                and reading.delta_ci_upper <= bound
            )
            detail = (
                f"delta {reading.delta:+.6f}, upper 95% bound "
                f"{reading.delta_ci_upper:+.6f} against {bound:g}"
            )

        gates.append(
            Gate(
                name=f"aggregate_{metric}",
                passed=passed,
                requirement=(
                    f"candidate {metric} <= incumbent and upper 95% CI of the "
                    f"delta <= {bound:g}"
                ),
                observed={
                    "delta": reading.delta,
                    "upper_95_ci": reading.delta_ci_upper,
                },
                threshold=bound,
                detail=detail,
            )
        )

    leg_counts = [int(value) for value in _require(by_leg, "leg_counts")]
    bounds = {
        "brier": float(_require(by_leg, "upper_95_ci_of_brier_delta_max")),
        "log_loss": float(_require(by_leg, "upper_95_ci_of_log_loss_delta_max")),
    }
    for legs in leg_counts:
        block = (
            rows.loc[rows["leg_count"].astype("Int64") == legs]
            if "leg_count" in rows.columns
            else rows.iloc[0:0]
        )
        for metric in ("brier", "log_loss"):
            reading = score_reading(
                block,
                metric=metric,
                slice_name=f"{legs}_leg",
                draws=settings["draws"],
                seed=settings["seed"],
                level=settings["level"],
                floor=log_loss_floor,
            )
            readings.append(reading)

            bound = bounds[metric]
            if reading.delta is None or not render_verdicts:
                passed = None
                detail = (
                    f"no paired {legs}-leg rows"
                    if reading.delta is None
                    else "minimum live evidence not met, so no verdict is rendered"
                )
            else:
                # Non-inferiority only. The frozen standard says statistical
                # superiority is NOT required on every leg count, so the point
                # estimate is reported and not gated here.
                passed = (
                    reading.delta_ci_upper is not None
                    and reading.delta_ci_upper <= bound
                )
                detail = (
                    f"upper 95% bound {reading.delta_ci_upper:+.6f} against "
                    f"{bound:g} (non-inferiority only)"
                )

            gates.append(
                Gate(
                    name=f"{legs}_leg_{metric}",
                    passed=passed,
                    requirement=(
                        f"upper 95% CI of the {legs}-leg {metric} delta "
                        f"<= {bound:g}, non-inferiority only"
                    ),
                    observed={
                        "delta": reading.delta,
                        "upper_95_ci": reading.delta_ci_upper,
                    },
                    threshold=bound,
                    detail=detail,
                )
            )

    return gates, readings


# ----------------------------------------------------------------------
# dependence
# ----------------------------------------------------------------------


def pooled_buckets(
    dependence: Sequence[Mapping[str, Any]],
    *,
    space: str,
    arm: str,
) -> dict[str, float]:
    """Pair-count-weighted pooling of one arm's bucket readings over games.

    The per-game readings are already mean correlations, so pooling them is a
    weighted average by the pair count that produced each -- the same pooling
    the research layer's accumulator performs, and the reason the pair counts
    are recorded alongside the values.
    """
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}

    for entry in dependence:
        block = (entry.get(space) or {}).get(arm)
        if not isinstance(block, Mapping):
            continue
        for bucket in CROSS_PLAYER_BUCKETS:
            if bucket not in block:
                continue
            weight = block.get(BUCKET_PAIR_WEIGHT[bucket])
            if weight is None:
                continue
            weight = float(weight)
            if weight <= 0.0:
                continue
            value = block[bucket]
            if value is None or not math.isfinite(float(value)):
                continue
            totals[bucket] = totals.get(bucket, 0.0) + float(value) * weight
            weights[bucket] = weights.get(bucket, 0.0) + weight

    return {
        bucket: totals[bucket] / weights[bucket]
        for bucket in totals
        if weights.get(bucket, 0.0) > 0.0
    }


def bucket_errors(
    reading: Mapping[str, float], observed: Mapping[str, float]
) -> dict[str, float]:
    return {
        bucket: reading[bucket] - observed[bucket]
        for bucket in CROSS_PLAYER_BUCKETS
        if bucket in reading and bucket in observed
    }


def cross_player_rmse(errors: Mapping[str, float]) -> float | None:
    """The research layer's definition, restated over the same bucket list."""
    values = [errors[bucket] for bucket in CROSS_PLAYER_BUCKETS if bucket in errors]
    if not values:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def dependence_report(window: EvidenceWindow) -> dict[str, Any]:
    """Pooled observed and per-arm bucket readings, in both gated spaces."""
    out: dict[str, Any] = {}
    for space in (LATENT_SPACE, COUNT_SPACE):
        observed = pooled_buckets(window.dependence, space=space, arm="observed")
        arms: dict[str, Any] = {}
        for arm in (CANDIDATE, INCUMBENT, INDEPENDENCE):
            reading = pooled_buckets(window.dependence, space=space, arm=arm)
            if not reading:
                arms[arm] = None
                continue
            errors = bucket_errors(reading, observed)
            arms[arm] = {
                "buckets": reading,
                "bucket_errors": errors,
                "cross_player_rmse": cross_player_rmse(errors),
            }
        out[space] = {"observed_buckets": observed, "by_model": arms}
    return out


def _rmse_gate(
    *,
    name: str,
    space: str,
    report: Mapping[str, Any],
    requirement: str,
    render_verdicts: bool,
) -> Gate:
    arms = (report.get(space) or {}).get("by_model") or {}
    candidate = (arms.get(CANDIDATE) or {}).get("cross_player_rmse")
    incumbent = (arms.get(INCUMBENT) or {}).get("cross_player_rmse")

    if candidate is None or incumbent is None:
        return Gate(
            name=name,
            passed=None,
            requirement=requirement,
            observed={"candidate": candidate, "incumbent": incumbent},
            detail=(
                f"the accumulated evidence carries no {space}-space dependence "
                "reading for both arms"
            ),
        )

    if not render_verdicts:
        return Gate(
            name=name,
            passed=None,
            requirement=requirement,
            observed={"candidate": candidate, "incumbent": incumbent},
            detail="minimum live evidence not met, so no verdict is rendered",
        )

    return Gate(
        name=name,
        passed=candidate < incumbent,
        requirement=requirement,
        observed={"candidate": candidate, "incumbent": incumbent},
        detail=(
            f"candidate {candidate:.6f} against incumbent {incumbent:.6f}, "
            "strict improvement required"
        ),
    )


def dependence_gates(
    policy: Mapping[str, Any],
    window: EvidenceWindow,
    *,
    render_verdicts: bool,
) -> tuple[list[Gate], dict[str, Any]]:
    """D1, D2, D3 and the D4 buckets that are reported but not gated."""
    gates_policy = _require(policy, "dependence_gates")
    report = dependence_report(window)

    gates = [
        _rmse_gate(
            name="D1_global_latent_dependence_rmse",
            space=LATENT_SPACE,
            report=report,
            requirement=str(
                _require(gates_policy, "D1_global_latent_dependence_rmse", "requirement")
            ),
            render_verdicts=render_verdicts,
        ),
        _rmse_gate(
            name="D2_count_space_dependence_rmse",
            space=COUNT_SPACE,
            report=report,
            requirement=str(
                _require(gates_policy, "D2_count_space_dependence_rmse", "requirement")
            ),
            render_verdicts=render_verdicts,
        ),
    ]

    protected = tuple(
        str(name)
        for name in _require(
            gates_policy, "D3_protected_buckets_may_not_worsen", "protected_buckets"
        )
    )
    max_worsening_z = float(
        _require(
            gates_policy, "D3_protected_buckets_may_not_worsen", "max_worsening_z"
        )
    )
    gates.append(
        _protected_bucket_gate(
            report=report,
            protected=protected,
            max_worsening_z=max_worsening_z,
            requirement=str(
                _require(
                    gates_policy,
                    "D3_protected_buckets_may_not_worsen",
                    "requirement",
                )
            ),
            render_verdicts=render_verdicts,
            window=window,
        )
    )

    reported_only = tuple(
        str(name)
        for name in _require(gates_policy, "D4_explicitly_reported_not_gated", "buckets")
    )
    report["explicitly_reported_not_gated"] = {
        bucket: _bucket_z(report, bucket, window) for bucket in reported_only
    }
    report["protected_buckets"] = {
        bucket: _bucket_z(report, bucket, window) for bucket in protected
    }
    report["all_declared_buckets"] = list(CROSS_PLAYER_BUCKETS)

    return gates, report


def _bucket_z(
    report: Mapping[str, Any], bucket: str, window: EvidenceWindow
) -> dict[str, Any]:
    """Candidate minus incumbent absolute error on one bucket, in z units.

    The z convention is the one the protected gate uses, so D4's buckets are
    reported on the same scale the gated ones are judged on -- which is what
    the policy asks for.
    """
    out: dict[str, Any] = {}
    for space in (LATENT_SPACE, COUNT_SPACE):
        arms = (report.get(space) or {}).get("by_model") or {}
        candidate = (arms.get(CANDIDATE) or {}).get("bucket_errors") or {}
        incumbent = (arms.get(INCUMBENT) or {}).get("bucket_errors") or {}
        if bucket not in candidate or bucket not in incumbent:
            out[space] = None
            continue
        standard_error = _bucket_standard_error(window, space, bucket)
        worsening = abs(float(candidate[bucket])) - abs(float(incumbent[bucket]))
        out[space] = {
            "candidate_error": float(candidate[bucket]),
            "incumbent_error": float(incumbent[bucket]),
            "standard_error": standard_error,
            "worsening": worsening,
            "worsening_z": (
                None
                if not standard_error
                else worsening / standard_error
            ),
        }
    return out


def _bucket_standard_error(
    window: EvidenceWindow, space: str, bucket: str
) -> float | None:
    """Game-clustered standard error of one bucket's pooled reading.

    Games are the independent unit here for the same reason they are in the
    score bootstrap. Computed from the spread of the per-game readings rather
    than assumed, and ``None`` when fewer than two games carry the bucket,
    because a z score over one game is not a z score.
    """
    values: list[float] = []
    for entry in window.dependence:
        block = (entry.get(space) or {}).get("observed")
        candidate = (entry.get(space) or {}).get(CANDIDATE)
        if not isinstance(block, Mapping) or not isinstance(candidate, Mapping):
            continue
        if bucket not in block or bucket not in candidate:
            continue
        values.append(float(candidate[bucket]) - float(block[bucket]))
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _protected_bucket_gate(
    *,
    report: Mapping[str, Any],
    protected: tuple[str, ...],
    max_worsening_z: float,
    requirement: str,
    render_verdicts: bool,
    window: EvidenceWindow,
) -> Gate:
    readings = {bucket: _bucket_z(report, bucket, window) for bucket in protected}
    offenders: dict[str, Any] = {}
    evaluable = False

    for bucket, spaces in readings.items():
        for space, reading in spaces.items():
            if not reading or reading.get("worsening_z") is None:
                continue
            evaluable = True
            if float(reading["worsening_z"]) > max_worsening_z:
                offenders[f"{bucket}/{space}"] = reading["worsening_z"]

    if not evaluable:
        passed: bool | None = None
        detail = (
            "no protected bucket carries a game-clustered z reading yet, so "
            "the gate is not evaluable"
        )
    elif not render_verdicts:
        passed = None
        detail = "minimum live evidence not met, so no verdict is rendered"
    else:
        passed = not offenders
        detail = (
            f"no protected bucket worsens by more than {max_worsening_z:g} z"
            if not offenders
            else f"protected buckets worsening beyond the bound: {offenders}"
        )

    return Gate(
        name="D3_protected_buckets_may_not_worsen",
        passed=passed,
        requirement=requirement,
        observed=readings,
        threshold=max_worsening_z,
        detail=detail,
    )


# ======================================================================
# the decision
# ======================================================================


def decide(
    policy: Mapping[str, Any],
    *,
    operational: Sequence[Gate],
    evidence: MinimumEvidence,
    scoring: Sequence[Gate],
    dependence: Sequence[Gate],
) -> tuple[str, str]:
    """The policy's own decision table, in the policy's own order.

    Safety first, because an operational failure that implicates the incumbent
    feed is not something to keep collecting evidence about. Then the minimum
    requirement, because the policy forbids a verdict before it is met. Only
    then do the scoring and dependence gates speak, and a clean pass is the
    only route to ``PROMOTION_REVIEW_ELIGIBLE`` -- which authorises a human
    review and nothing else.
    """
    vocabulary = decision_vocabulary(policy)

    def state(name: str) -> str:
        if name not in vocabulary:
            raise PolicyIncomplete(
                f"{name} is not one of the frozen decision states {vocabulary}"
            )
        return name

    safety_critical = {"O1_shadow_publishing_violations", "O6_incumbent_serving_unaffected"}
    tripped = [gate.name for gate in operational if gate.failed]

    if any(name in safety_critical for name in tripped):
        return (
            state("SHADOW_DISABLED_FOR_SAFETY"),
            "an operational gate failed in a way that implicates the "
            f"incumbent feed or publishing: {sorted(tripped)}",
        )

    if tripped:
        # O2, O3, O4 and O5 are shadow-quality failures. They are real and
        # they block eligibility, but the policy reserves
        # SHADOW_DISABLED_FOR_SAFETY for the incumbent-implicating ones.
        return (
            state("CONTINUE_SHADOW"),
            f"an operational gate failed: {sorted(tripped)}",
        )

    if not evidence.satisfied:
        return (
            state("CONTINUE_SHADOW"),
            "the minimum live-evidence requirement is not yet met: "
            + ", ".join(evidence.payload()["unmet"]),
        )

    judged = [*scoring, *dependence]
    failed = [gate.name for gate in judged if gate.failed]
    # The operational gates join the unevaluated check but not the failure
    # check: a failed one was already routed above, where the distinction
    # between a safety failure and a shadow-quality failure lives. An
    # unevaluated one belongs here, because once the minimum requirement is
    # met every gate has evidence to speak on, and a gate that stayed silent
    # about an evidence window this large has not been satisfied by it.
    unevaluated = [
        gate.name for gate in (*operational, *judged) if not gate.evaluated
    ]

    if failed:
        return (
            state("SHADOW_MODEL_REJECTED"),
            "the minimum live-evidence requirement is met and a proper-score "
            f"or dependence gate fails with statistical support: {sorted(failed)}",
        )

    if unevaluated:
        return (
            state("CONTINUE_SHADOW"),
            "the minimum requirement is met but these gates could not be "
            f"evaluated on the accumulated evidence: {sorted(unevaluated)}",
        )

    return (
        state("PROMOTION_REVIEW_ELIGIBLE"),
        "every operational, proper-score and dependence gate passes at the "
        "frozen evaluation cut. A human review may be opened. This is not a "
        "promotion and no code path may act on it",
    )


@dataclass
class Report:
    decision: str
    reason: str
    operational: list[Gate]
    evidence: MinimumEvidence
    scoring: list[Gate]
    dependence: list[Gate]
    readings: list[ScoreReading]
    dependence_detail: dict[str, Any]
    policy_identity: dict[str, Any]

    @property
    def safe(self) -> bool:
        return self.decision != "SHADOW_DISABLED_FOR_SAFETY"

    def payload(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "decision_reason": self.reason,
            "dependence_detail": self.dependence_detail,
            "dependence_gates": [gate.payload() for gate in self.dependence],
            "minimum_live_evidence": self.evidence.payload(),
            "operational_gates": [gate.payload() for gate in self.operational],
            "policy_identity": self.policy_identity,
            "promotion_authority": "NONE",
            "proper_score_gates": [gate.payload() for gate in self.scoring],
            "score_readings": [reading.payload() for reading in self.readings],
        }


def evaluate(
    policy: Mapping[str, Any],
    window: EvidenceWindow,
    *,
    log_loss_floor: float = 1e-6,
) -> Report:
    """Evaluate one accumulated window. Promotes nothing."""
    if str(_require(policy, "AUTONOMOUS_PROMOTION_AUTHORITY")) != "NONE":
        raise PolicyIncomplete(
            "the frozen policy no longer states that autonomous promotion "
            "authority is NONE, which this evaluator refuses to run under"
        )

    operational = operational_gates(policy, window)
    evidence = minimum_evidence(policy, window)
    render = evidence.satisfied

    scoring, readings = proper_score_gates(
        policy, window, render_verdicts=render, log_loss_floor=log_loss_floor
    )
    dependence, detail = dependence_gates(
        policy, window, render_verdicts=render
    )

    decision, reason = decide(
        policy,
        operational=operational,
        evidence=evidence,
        scoring=scoring,
        dependence=dependence,
    )

    return Report(
        decision=decision,
        reason=reason,
        operational=operational,
        evidence=evidence,
        scoring=scoring,
        dependence=dependence,
        readings=readings,
        dependence_detail=detail,
        policy_identity=dict(_require(policy, "identity")),
    )
