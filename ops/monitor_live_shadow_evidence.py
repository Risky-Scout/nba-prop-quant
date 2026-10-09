#!/usr/bin/env python3
"""Accumulate live shadow evidence and read the frozen policy over it.

MONITORING ONLY. This entry point promotes nothing, publishes nothing,
refits nothing and never touches serving authority. It reads evidence that
other steps already produced, keeps a cumulative copy of it, and reports what
the frozen policy says about the total.

WHY A SEPARATE ACCUMULATOR IS NEEDED AT ALL
-------------------------------------------
The shadow step writes its log, its grading report and its status into the
run's temporary directory, which GitHub destroys when the job ends. The
frozen policy's minimum evidence requirement is 500 graded games over 30
regular-season days, so every number it gates on is a sum over runs that no
longer exist. Without a durable accumulation there is no window to evaluate,
and the evaluator -- which is already written and already tested -- has
nothing to read.

So this step copies the day's shadow evidence out of the run directory into
the durable production state root, beside the incumbent's own grades, and then
evaluates the accumulated total. The copy is made here rather than by pointing
the shadow step at a durable path, because the shadow step is pre-existing
lifecycle and this lineage is permitted to add steps, never to rewrite one.

NO THRESHOLD LIVES HERE
-----------------------
Every bound, minimum, bucket list, seed and decision word comes from
``research/final_model/live_shadow_promotion_policy.json`` by way of
``live_policy``. This module decides one thing the policy does not state: that
the lifecycle goes red when the policy's decision is
``SHADOW_DISABLED_FOR_SAFETY`` and stays green otherwise. That is an operator
signalling choice, not a threshold, and it is the only reason this process
can exit nonzero on a healthy run's evidence.

WHAT IDEMPOTENCE MEANS HERE
---------------------------
An accumulated window is a count, so a step that ran twice must not say the
evidence doubled. Every joint event is stored under an identity derived from
the event itself, never from when it was ingested, so re-ingesting a slate
replaces its rows. Run records, per-game dependence readings and provenance
preimages are keyed by slate date and fingerprint for the same reason. Every
write is atomic, so an interrupted run leaves the previous state intact rather
than a half-written one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state import live_policy  # noqa: E402
from nba_prop_quant.storage import upsert_parquet  # noqa: E402


def _shadow_health():
    """The health assertion module, imported for its provenance field map.

    O5 names four identities a graded row must resolve to, and the shadow's
    provenance payload carries them under its own field names -- ``code_sha``
    for the production SHA, the two artifact hashes for the artifact identity.
    That mapping is already written down in
    ``ops/evaluate_shadow_health.py``, which is the step that judges a single
    run against the same gate. Imported rather than restated: two mappings
    between the same two vocabularies would be two answers to whether a row's
    provenance is complete.
    """
    import importlib.util

    name = "_shadow_health_for_monitor"

    cached = sys.modules.get(name)
    if cached is not None:
        return cached

    path = PROJECT_ROOT / "ops" / "evaluate_shadow_health.py"
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:  # pragma: no cover - packaging fault
        raise MonitoringRefused(f"cannot import the shadow health module at {path}")

    module = importlib.util.module_from_spec(spec)
    # Registered before execution, not after: that module defines dataclasses,
    # and dataclasses resolves a field's annotation by looking its own module
    # up in sys.modules while the class body is still being executed.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module

# ----------------------------------------------------------------------
# where the durable state lives
# ----------------------------------------------------------------------

#: Under the data root, beside ``processed/incumbent_grades`` and
#: ``processed/incumbent_serving``, because this is production state with the
#: same durability requirement and the same backup story.
STATE_RELATIVE = Path("processed") / "live_shadow_monitoring"

#: One file per slate under each of these, so a slate is re-ingested by
#: rewriting its own file and nothing else. Not partitioned by season: the
#: grade store partitions that way because it mirrors the box-score tree,
#: and inventing a second answer to which season a date belongs to -- for a
#: store that mirrors nothing -- would be a second answer too many.
EVENTS_RELATIVE = Path("events")
RUNS_RELATIVE = Path("runs")
DEPENDENCE_RELATIVE = Path("dependence")
PROVENANCE_RELATIVE = Path("provenance")

#: Columns that must come back from parquet as numbers. An accumulated window
#: spans many files and an early slate can leave a column entirely null, which
#: parquet stores untyped; concatenating that with a later slate's floats gives
#: an object column, and the evaluator's arithmetic would then depend on which
#: day happened to be first.
NUMERIC_EVENT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "leg_count",
    "realized",
    "candidate_probability",
    "candidate_standard_error",
    "incumbent_probability",
    "independence_probability",
    "candidate_independent_product",
    "push_fraction",
    "served_probability",
    "min_eigenvalue",
    "same_player_max_block_deviation",
)

#: Columns the evaluator reads as booleans, for the same reason.
BOOLEAN_EVENT_COLUMNS: tuple[str, ...] = ("fell_back", "game_fell_back", "published")

#: Where the incumbent's own graded prop rows are, written by
#: ``ops/grade_incumbent_production_slate.py``.
INCUMBENT_GRADES_RELATIVE = Path("processed") / "incumbent_grades"
INCUMBENT_SERVING_RELATIVE = Path("processed") / "incumbent_serving"

#: The identity of a stored joint event. Derived from the event, so ingesting
#: the same slate again lands on the same rows.
EVENT_KEY = ["event_key"]

#: Columns the evaluator reads off an accumulated event row. A row that is
#: missing one is stored with it null rather than dropped, because an absent
#: reading must reach the evaluator as absent -- that is what makes a gate
#: report itself unevaluable instead of quietly passing.
EVENT_COLUMNS: tuple[str, ...] = (
    "event_key",
    "slate_date",
    "game_id",
    "event_id",
    "legs",
    "leg_count",
    "realized",
    "candidate_probability",
    "candidate_standard_error",
    "incumbent_probability",
    "independence_probability",
    "candidate_independent_product",
    "push_fraction",
    "served_model",
    "served_probability",
    "fell_back",
    "game_fell_back",
    "failure_reason",
    "published",
    "provenance_fingerprint",
    "dependence_model_version",
    "factor_spec_hash",
    "min_eigenvalue",
    "same_player_max_block_deviation",
)

#: Readings carried forward from a shadow run's status file. These are what
#: the operational gates sum over, and the names are the ones
#: :func:`live_policy.operational_gates` looks for.
RUN_COLUMNS: tuple[str, ...] = (
    "slate_date",
    "outcome",
    "rows_published",
    "psd_failures",
    "same_player_max_block_deviation",
    "min_covariance_eigenvalue",
    "production_serving_was_affected",
    "served_authority",
    "games_shadowed",
    "games_that_fell_back",
    "events_shadowed",
    "dependence_diagnostics_recorded",
    "provenance_fingerprint",
    "production_code_sha",
    "ingested_at",
)

#: The identities a run's provenance must agree with before its evidence is
#: accumulated. Read from the frozen policy's own identity block, so this is
#: not a second statement of what the model is.
PINNED_IDENTITY_FIELDS: tuple[str, ...] = (
    "dependence_model_version",
    "factor_spec_hash",
    "final_model_spec_sha256",
)

EXIT_OK = 0
EXIT_SHADOW_DISABLED_FOR_SAFETY = 1

#: The decision that makes this step fail the lifecycle. Not a threshold: the
#: policy decides the state, and this names which state an operator must be
#: paged about.
SAFETY_DECISION = "SHADOW_DISABLED_FOR_SAFETY"


class MonitoringRefused(RuntimeError):
    """The monitoring step cannot honestly report on this evidence."""


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Replace ``path`` in one step, or leave the previous file untouched.

    The accumulated state is read by the next day's run, so a half-written
    file is worse than no file: it would be parsed and believed. Written to a
    temporary name in the same directory and moved into place, which is atomic
    on every filesystem this runs on, so an interrupted run loses the write
    rather than corrupting the state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    )
    temp_path = Path(handle.name)

    try:
        with handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def event_key(slate_date: str, game_id: Any, event_id: Any) -> str:
    """The identity of one joint event, derived from the event alone.

    Not from the run, the ingestion time or the fingerprint. A slate that is
    shadowed twice -- a retry, a catch-up, a refit the same morning -- must
    land on the same rows, because the accumulated window is a count and the
    policy's minimum sample is a bound on that count. Including anything
    run-specific here would let a retry inflate the evidence.
    """
    payload = "|".join(
        (
            str(slate_date),
            str(game_id),
            str(event_id),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


# ======================================================================
# ingestion
# ======================================================================


def pinned_identity(policy: Mapping[str, Any]) -> dict[str, str]:
    identity = dict(live_policy.require(policy, "identity"))
    return {
        field: str(identity[field])
        for field in PINNED_IDENTITY_FIELDS
        if field in identity
    }


def resolved_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The run's provenance, with O5's four identities resolvable by name.

    The policy is explicit that the gate is "all four resolve for every graded
    candidate row, not that all four appear as literal columns", and it says
    where they are carried: the remaining identities are recorded once per run
    in the provenance payload. The payload's own names are not the policy's
    names -- it writes ``code_sha`` and two artifact hashes -- so the stored
    preimage carries both, the producer's names and the policy's.

    Resolution only. A field whose sources are all absent stays absent, so a
    run that genuinely did not record its code SHA fails O5 rather than being
    handed one. Nothing here is invented and nothing is overwritten: a payload
    that already carries a policy name keeps its own value.
    """
    sources = _shadow_health().PROVENANCE_FIELD_SOURCES

    resolved = dict(payload)

    for required, candidates in sources.items():
        if str(resolved.get(required) or "").strip():
            continue
        for source in candidates:
            value = payload.get(source)
            if value is not None and str(value).strip():
                resolved[required] = value
                break

    return resolved


def identity_disagreements(
    *, provenance: Mapping[str, Any], pinned: Mapping[str, str]
) -> dict[str, dict[str, str | None]]:
    """Where a run's provenance contradicts the frozen policy's identity.

    An absent field is not a disagreement: the provenance payload does not
    have to restate everything, and O5 is the gate that judges completeness.
    A field that is present and different is a disagreement, and it is why
    the run's evidence is refused rather than pooled -- evidence from a model
    the policy was not frozen against is not evidence about this candidate,
    and pooling it would quietly change what the gates are reading.
    """
    out: dict[str, dict[str, str | None]] = {}

    for field, expected in pinned.items():
        observed = provenance.get(field)
        if observed is None or not str(observed).strip():
            continue
        if str(observed) != expected:
            out[field] = {"expected": expected, "observed": str(observed)}

    return out


def shadow_log_rows(path: Path, *, slate_date: str) -> pd.DataFrame:
    """Read the shadow's JSONL log into accumulated-event shape."""
    records: list[dict[str, Any]] = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise MonitoringRefused(
                f"the shadow log at {path} carries a line that is not JSON, so "
                f"the day's evidence cannot be read: {error}"
            ) from error
        records.append(record)

    if not records:
        return pd.DataFrame(columns=list(EVENT_COLUMNS))

    frame = pd.DataFrame.from_records(records)

    # The log's own slate_date wins where present; the argument is the
    # fallback for a log written before that column existed.
    if "slate_date" not in frame.columns:
        frame["slate_date"] = slate_date
    frame["slate_date"] = frame["slate_date"].fillna(slate_date).astype(str)

    foreign = sorted(set(frame["slate_date"].unique()) - {str(slate_date)})
    if foreign:
        raise MonitoringRefused(
            f"the shadow log at {path} carries rows for {foreign}, not the "
            f"slate being ingested ({slate_date}). Pooling them would file one "
            "slate's evidence under another's date and break the calendar-day "
            "count the frozen minimum is stated in"
        )

    if "legs" in frame.columns:
        frame["legs"] = frame["legs"].apply(
            lambda value: "|".join(value) if isinstance(value, (list, tuple)) else value
        )

    for column in EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    frame["event_key"] = [
        event_key(slate, game, event)
        for slate, game, event in zip(
            frame["slate_date"], frame["game_id"], frame["event_id"], strict=True
        )
    ]

    duplicated = frame["event_key"].duplicated(keep=False)
    if bool(duplicated.any()):
        offenders = sorted(
            frame.loc[duplicated, "event_id"].astype(str).unique()
        )[:10]
        raise MonitoringRefused(
            f"the shadow log at {path} carries the same event more than once "
            f"for {slate_date}: {offenders}. Accumulating it would count one "
            "joint event twice toward the frozen minimum"
        )

    return frame.loc[:, list(EVENT_COLUMNS)].copy()


def run_record(
    *,
    status: Mapping[str, Any],
    slate_date: str,
    production_sha: str | None,
) -> dict[str, Any]:
    """One accumulated run row, from the shadow's own status file.

    Readings live at two levels in that file: the authority and serving facts
    are stated at the top, and the numerical diagnostics sit under ``shadow``.
    Both are flattened here into the names the operational gates read.
    """
    shadow = status.get("shadow") or {}

    record = {
        "slate_date": str(slate_date),
        "outcome": status.get("outcome"),
        "rows_published": _number(
            status.get("rows_published")
            if status.get("rows_published") is not None
            else shadow.get("rows_published")
        ),
        "psd_failures": _number(shadow.get("psd_failures")),
        "same_player_max_block_deviation": _number(
            shadow.get("same_player_max_block_deviation")
        ),
        "min_covariance_eigenvalue": _number(shadow.get("min_covariance_eigenvalue")),
        "production_serving_was_affected": bool(
            status.get("production_serving_was_affected") is True
        ),
        "served_authority": status.get("served_authority"),
        "games_shadowed": _number(shadow.get("games_shadowed")),
        "games_that_fell_back": _number(shadow.get("games_that_fell_back")),
        "events_shadowed": _number(shadow.get("events_shadowed")),
        "dependence_diagnostics_recorded": _number(
            shadow.get("dependence_diagnostics_recorded")
        ),
        "provenance_fingerprint": shadow.get("provenance_fingerprint"),
        "production_code_sha": production_sha,
        "ingested_at": utc_now(),
    }

    return {name: record.get(name) for name in RUN_COLUMNS}


def ingest(
    *,
    state_root: Path,
    slate_date: str,
    policy: Mapping[str, Any],
    shadow_log: Path | None,
    shadow_status: Path | None,
    shadow_grading: Path | None,
    production_sha: str | None,
) -> dict[str, Any]:
    """Copy one day's shadow evidence into the durable accumulated state.

    Returns what was taken and what was refused. Absence is normal and is not
    a failure: a day with no retrain ran no shadow, and the accumulated window
    is still the thing to report on.
    """
    outcome: dict[str, Any] = {
        "slate_date": str(slate_date),
        "shadow_evidence_present": False,
        "events_ingested": 0,
        "dependence_games_ingested": 0,
        "run_record_written": False,
        "provenance_recorded": [],
        "refused": None,
        "absent": [],
    }

    status: Mapping[str, Any] | None = None

    if shadow_status is None or not Path(shadow_status).is_file():
        outcome["absent"].append("shadow status")
    else:
        status = read_json(shadow_status)

    if shadow_log is None or not Path(shadow_log).is_file():
        outcome["absent"].append("shadow log")

    if shadow_grading is None or not Path(shadow_grading).is_file():
        outcome["absent"].append("shadow grading report")

    if status is None:
        # Without the run's status there is no provenance payload to check the
        # rows against and no operational reading to carry forward. Ingesting
        # the rows anyway would add graded events toward the frozen minimum
        # whose identities can never be resolved, so the day is skipped and
        # said so.
        return outcome

    outcome["shadow_evidence_present"] = True

    shadow = status.get("shadow") or {}
    provenance = shadow.get("provenance") or {}
    fingerprint = shadow.get("provenance_fingerprint")

    disagreements = identity_disagreements(
        provenance=provenance, pinned=pinned_identity(policy)
    )

    if disagreements:
        outcome["refused"] = {
            "reason": "the run's provenance contradicts the frozen policy identity",
            "disagreements": disagreements,
        }
        # Fail closed for this evidence only. The accumulated state is left
        # exactly as it was, so a run against the wrong model cannot move a
        # single gate, and the refusal is reported rather than swallowed.
        return outcome

    if fingerprint and provenance:
        write_json_atomic(
            state_root / PROVENANCE_RELATIVE / f"{fingerprint}.json",
            resolved_provenance(provenance),
        )
        outcome["provenance_recorded"] = [str(fingerprint)]

    if shadow_log is not None and Path(shadow_log).is_file():
        rows = shadow_log_rows(Path(shadow_log), slate_date=slate_date)
        if not rows.empty:
            upsert_parquet(
                rows,
                state_root / EVENTS_RELATIVE / f"{slate_date}.parquet",
                EVENT_KEY,
            )
        outcome["events_ingested"] = int(len(rows))

    if shadow_grading is not None and Path(shadow_grading).is_file():
        grading = read_json(shadow_grading)
        entries = [
            entry
            for entry in (grading.get("dependence_diagnostics") or [])
            if isinstance(entry, Mapping)
        ]
        write_json_atomic(
            state_root / DEPENDENCE_RELATIVE / f"{slate_date}.json",
            {
                "slate_date": str(slate_date),
                "ingested_at": utc_now(),
                "provenance_fingerprint": grading.get("provenance_fingerprint"),
                "games": entries,
            },
        )
        outcome["dependence_games_ingested"] = len(entries)

    write_json_atomic(
        state_root / RUNS_RELATIVE / f"{slate_date}.json",
        run_record(
            status=status, slate_date=slate_date, production_sha=production_sha
        ),
    )
    outcome["run_record_written"] = True

    return outcome


# ======================================================================
# the accumulated window
# ======================================================================


def accumulated_events(state_root: Path) -> pd.DataFrame:
    files = sorted((state_root / EVENTS_RELATIVE).glob("*.parquet"))

    if not files:
        return pd.DataFrame(columns=list(EVENT_COLUMNS))

    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    for column in NUMERIC_EVENT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in BOOLEAN_EVENT_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna(False).astype(bool)

    # Defence in depth. The per-slate files cannot overlap by construction,
    # but a window that double-counted an event would misreport the frozen
    # minimum, which is the one number nothing else here would catch.
    return frame.drop_duplicates(subset=EVENT_KEY, keep="last")


def accumulated_runs(state_root: Path) -> list[dict[str, Any]]:
    return [
        read_json(path)
        for path in sorted((state_root / RUNS_RELATIVE).glob("*.json"))
    ]


def accumulated_dependence(state_root: Path) -> list[dict[str, Any]]:
    """Per-game dependence entries, pooled over every ingested slate."""
    entries: list[dict[str, Any]] = []
    for path in sorted((state_root / DEPENDENCE_RELATIVE).glob("*.json")):
        payload = read_json(path)
        for entry in payload.get("games") or []:
            if isinstance(entry, Mapping):
                entries.append(dict(entry))
    return entries


def accumulated_provenance(state_root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: read_json(path)
        for path in sorted((state_root / PROVENANCE_RELATIVE).glob("*.json"))
    }


def evidence_window(state_root: Path) -> live_policy.EvidenceWindow:
    return live_policy.EvidenceWindow(
        events=accumulated_events(state_root),
        provenance=accumulated_provenance(state_root),
        dependence=tuple(accumulated_dependence(state_root)),
        runs=tuple(accumulated_runs(state_root)),
    )


# ======================================================================
# the incumbent's own graded record
# ======================================================================


def incumbent_grade_summary(grades_root: Path) -> dict[str, Any]:
    """The incumbent's realized prop record, reported beside the gated arms.

    Deliberately kept apart from the policy's comparison. These are
    single-leg prop grades from ``ops/grade_incumbent_production_slate.py``;
    the arm the policy gates the candidate against is the incumbent's
    probability on the same joint event, which lives on the shadow row. Mixing
    the two would compare the candidate's joint events against the
    incumbent's marginals and call the difference a model result.
    """
    files = sorted(Path(grades_root).rglob("*.parquet"))

    if not files:
        return {
            "available": False,
            "detail": f"no incumbent grade rows under {grades_root}",
            "graded_rows": 0,
            "pending_rows": 0,
            "slates": 0,
            "brier": None,
            "log_loss": None,
        }

    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    if "prediction_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["prediction_id"], keep="last")

    status = frame.get("settlement_status")
    graded = (
        frame.loc[status == "GRADED"] if status is not None else frame.iloc[0:0]
    )
    pending = (
        frame.loc[status == "PENDING_SETTLEMENT"]
        if status is not None
        else frame.iloc[0:0]
    )
    slates = (
        int(frame["slate_date"].nunique()) if "slate_date" in frame.columns else 0
    )

    def mean(column: str) -> float | None:
        if graded.empty or column not in graded.columns:
            return None
        values = pd.to_numeric(graded[column], errors="coerce").dropna()
        return float(values.mean()) if len(values) else None

    return {
        "available": True,
        "detail": (
            f"{len(graded)} graded and {len(pending)} pending incumbent prop "
            f"row(s) over {slates} slate(s)"
        ),
        "graded_rows": int(len(graded)),
        "pending_rows": int(len(pending)),
        "slates": slates,
        "brier": mean("brier_contribution"),
        "log_loss": mean("log_loss_contribution"),
        "comparison_note": (
            "single-leg prop grades. Reported for the incumbent's own record; "
            "the arm the frozen policy gates against is the incumbent "
            "probability on the same joint event, carried on the shadow row"
        ),
    }


def incumbent_identity(serving_root: Path) -> dict[str, Any]:
    """Who served, from the most recent durable serving receipt."""
    receipts = sorted(Path(serving_root).glob("*.json"))

    if not receipts:
        return {
            "available": False,
            "detail": f"no incumbent serving receipt under {serving_root}",
        }

    payload = read_json(receipts[-1])
    artifact = payload.get("prediction_artifact") or {}

    return {
        "available": True,
        "slate_date": payload.get("slate_date"),
        "authority": payload.get("incumbent_authority"),
        "fit_id": payload.get("incumbent_fit_id"),
        "version": payload.get("incumbent_version"),
        "input_state_fingerprint": payload.get("input_state_fingerprint"),
        "prediction_artifact_sha256": artifact.get("sha256"),
        "production_code_sha": payload.get("production_code_sha"),
        "receipt": str(receipts[-1]),
    }


def candidate_identity(
    window: live_policy.EvidenceWindow, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """What produced the candidate rows, resolved through the fingerprints.

    Read off the recorded preimages rather than restated from the policy, so a
    window whose rows were produced by something else reports that rather than
    echoing the identity it was supposed to have.
    """
    fingerprints = sorted(window.provenance)

    identities: dict[str, Any] = {
        "provenance_fingerprints": fingerprints,
        "pinned_by_policy": pinned_identity(policy),
    }

    for field in (
        "dependence_model_version",
        "factor_spec_hash",
        "final_model_spec_sha256",
        "artifact_hash_or_version",
        "production_code_sha",
    ):
        observed = sorted(
            {
                str(payload.get(field))
                for payload in window.provenance.values()
                if payload.get(field) is not None
                and str(payload.get(field)).strip()
            }
        )
        identities[field] = (
            observed[0] if len(observed) == 1 else (observed or None)
        )

    return identities


# ======================================================================
# the status artifact
# ======================================================================


def _reading(readings: Sequence[live_policy.ScoreReading], metric: str, slice_name: str):
    for reading in readings:
        if reading.metric == metric and reading.slice_name == slice_name:
            return reading
    return None


def _score_block(
    readings: Sequence[live_policy.ScoreReading], slice_name: str
) -> dict[str, Any]:
    """Candidate and incumbent on one slice, with nothing invented.

    A slice with no paired rows reports ``null`` for every number. That is the
    point: a fabricated zero here would read as a tie between the arms.
    """
    out: dict[str, Any] = {}

    for metric in ("brier", "log_loss"):
        reading = _reading(readings, metric, slice_name)
        out[f"candidate_{metric}"] = None if reading is None else reading.candidate
        out[f"incumbent_{metric}"] = None if reading is None else reading.incumbent
        out[f"{metric}_delta"] = None if reading is None else reading.delta
        out[f"{metric}_delta_ci_lower"] = (
            None if reading is None else reading.delta_ci_lower
        )
        out[f"{metric}_delta_ci_upper"] = (
            None if reading is None else reading.delta_ci_upper
        )

    events = _reading(readings, "brier", slice_name)
    out["paired_events"] = 0 if events is None else int(events.events)

    return out


def _dependence_block(detail: Mapping[str, Any], space: str) -> dict[str, Any]:
    block = detail.get(space) or {}
    arms = block.get("by_model") or {}

    def rmse(arm: str) -> float | None:
        reading = arms.get(arm) or {}
        return reading.get("cross_player_rmse") if isinstance(reading, Mapping) else None

    def buckets(arm: str) -> Any:
        reading = arms.get(arm) or {}
        return reading.get("buckets") if isinstance(reading, Mapping) else None

    return {
        "candidate_cross_player_rmse": rmse(live_policy.CANDIDATE),
        "incumbent_cross_player_rmse": rmse(live_policy.INCUMBENT),
        "independence_cross_player_rmse": rmse(live_policy.INDEPENDENCE),
        "observed_buckets": block.get("observed_buckets"),
        "candidate_buckets": buckets(live_policy.CANDIDATE),
        "incumbent_buckets": buckets(live_policy.INCUMBENT),
    }


def status_payload(
    *,
    report: live_policy.Report,
    window: live_policy.EvidenceWindow,
    policy: Mapping[str, Any],
    policy_sha256: str,
    slate_date: str,
    production_sha: str | None,
    ingestion: Mapping[str, Any],
    incumbent_grades: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    state_root: Path,
) -> dict[str, Any]:
    """The machine-readable monitoring artifact.

    Every number here is either read from the accumulated evidence or copied
    from the evaluator's own report. Nothing is recomputed on a second
    definition, because two definitions of the candidate's Brier score would
    be two answers to the policy's question.
    """
    by_leg = window.graded_by_leg_count()
    served = live_policy.require(policy, "served_authority")
    operational = {gate.name: gate.payload() for gate in report.operational}

    # O1's own reading, not a second count of the same thing. The gate
    # already sums published rows across runs and published flags across
    # rows, and a separate tally here could disagree with the gate the policy
    # actually judges on.
    published_rows = (operational.get("O1_shadow_publishing_violations") or {}).get(
        "observed"
    )

    return {
        "entry_point": "ops/monitor_live_shadow_evidence.py",
        "mode": "MONITORING_ONLY",
        "generated_at": utc_now(),
        "slate_date": str(slate_date),
        "production_code_sha": production_sha,
        "state_root": str(state_root),
        # --- authority, restated from the policy and from the evidence ---
        "promotion_authority": "NONE",
        "autonomous_promotion_authority": str(
            live_policy.require(policy, "AUTONOMOUS_PROMOTION_AUTHORITY")
        ),
        "publishing_switch": str(served["shadow_publishing_switch"]),
        "published_authority": str(served["published_authority"]),
        "published": False,
        "rows_published": published_rows,
        "production_serving_was_affected": any(
            run.get("production_serving_was_affected") is True for run in window.runs
        ),
        # --- identity ---
        "candidate_identity": candidate_identity(window, policy),
        "incumbent_identity": dict(incumbent),
        "policy_identity": report.policy_identity,
        "policy_sha256": policy_sha256,
        # --- sample counts ---
        "sample_counts": {
            "graded_games": window.graded_games,
            "total_joint_events": int(len(window.graded)),
            "events_accumulated_including_ungraded": int(len(window.events)),
            "by_leg_count": {
                str(legs): int(by_leg.get(legs, 0)) for legs in (2, 3, 4)
            },
            "regular_season_calendar_days": window.calendar_days,
            "shadow_runs_accumulated": len(window.runs),
            "dependence_games_accumulated": len(window.dependence),
        },
        # --- operational ---
        "operational_gates": operational,
        "minimum_live_evidence": report.evidence.payload(),
        # --- proper scores ---
        "proper_scores": {
            "aggregate": _score_block(report.readings, "aggregate"),
            "by_leg_count": {
                str(legs): _score_block(report.readings, f"{legs}_leg")
                for legs in (2, 3, 4)
            },
            "gates": {gate.name: gate.payload() for gate in report.scoring},
            "delta_convention": str(
                live_policy.require(policy, "evaluation", "loss_delta_convention")
            ),
        },
        # --- dependence ---
        "dependence": {
            "latent": _dependence_block(report.dependence_detail, live_policy.LATENT_SPACE),
            "count": _dependence_block(report.dependence_detail, live_policy.COUNT_SPACE),
            "all_declared_buckets": report.dependence_detail.get(
                "all_declared_buckets"
            ),
            "protected_buckets": report.dependence_detail.get("protected_buckets"),
            "explicitly_reported_not_gated": report.dependence_detail.get(
                "explicitly_reported_not_gated"
            ),
            "gates": {gate.name: gate.payload() for gate in report.dependence},
        },
        # --- the incumbent's own record, kept distinguishable ---
        "incumbent_production_grades": dict(incumbent_grades),
        # --- what this run took in ---
        "ingestion": dict(ingestion),
        # --- the decision ---
        "decision": report.decision,
        "decision_reason": report.reason,
        "decision_states": list(live_policy.decision_vocabulary(policy)),
        "safety_signal_raised": report.decision == SAFETY_DECISION,
        "lifecycle_exit_code": (
            EXIT_SHADOW_DISABLED_FOR_SAFETY
            if report.decision == SAFETY_DECISION
            else EXIT_OK
        ),
    }


def render(payload: Mapping[str, Any]) -> str:
    counts = payload["sample_counts"]
    minimum = payload["minimum_live_evidence"]
    aggregate = payload["proper_scores"]["aggregate"]

    def number(value: Any, spec: str = ".6f") -> str:
        return "not evaluable" if value is None else format(float(value), spec)

    lines = [
        "## Live shadow monitoring",
        "",
        f"Decision: **{payload['decision']}**",
        "",
        payload["decision_reason"],
        "",
        "| accumulated evidence | observed |",
        "| --- | --- |",
        f"| graded games | {counts['graded_games']} |",
        f"| graded joint events | {counts['total_joint_events']} |",
        f"| 2-leg / 3-leg / 4-leg | "
        f"{counts['by_leg_count']['2']} / {counts['by_leg_count']['3']} / "
        f"{counts['by_leg_count']['4']} |",
        f"| regular-season days | {counts['regular_season_calendar_days']} |",
        f"| shadow runs | {counts['shadow_runs_accumulated']} |",
        f"| minimum requirement satisfied | {minimum['satisfied']} |",
        "",
        "| operational gate | verdict | observed |",
        "| --- | --- | --- |",
    ]

    for name, gate in payload["operational_gates"].items():
        verdict = (
            "not evaluable"
            if gate["passed"] is None
            else ("PASS" if gate["passed"] else "FAIL")
        )
        lines.append(f"| `{name}` | {verdict} | {gate['observed']} |")

    lines += [
        "",
        "| aggregate score | candidate | incumbent |",
        "| --- | --- | --- |",
        f"| Brier | {number(aggregate['candidate_brier'])} | "
        f"{number(aggregate['incumbent_brier'])} |",
        f"| log loss | {number(aggregate['candidate_log_loss'])} | "
        f"{number(aggregate['incumbent_log_loss'])} |",
        "",
        f"Published authority: **{payload['published_authority']}**. "
        f"Publishing switch: **{payload['publishing_switch']}**. "
        f"Autonomous promotion authority: "
        f"**{payload['autonomous_promotion_authority']}**.",
        "",
        "Monitoring only. This step promotes nothing, publishes nothing and "
        "leaves serving authority untouched.",
        "",
    ]

    return "\n".join(lines)


# ======================================================================
# entry point
# ======================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Accumulate live shadow evidence durably and report what the "
            "frozen promotion policy says about the total. Promotes nothing."
        )
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help=f"defaults to <data-root>/{STATE_RELATIVE} .",
    )
    parser.add_argument(
        "--shadow-log",
        type=Path,
        default=None,
        help="the day's shadow JSONL log, if the shadow ran.",
    )
    parser.add_argument(
        "--shadow-status",
        type=Path,
        default=None,
        help="the day's shadow status file, if the shadow ran.",
    )
    parser.add_argument(
        "--shadow-grading",
        type=Path,
        default=None,
        help="the day's shadow grading report, if the shadow ran.",
    )
    parser.add_argument(
        "--incumbent-grades-root",
        type=Path,
        default=None,
        help=f"defaults to <data-root>/{INCUMBENT_GRADES_RELATIVE} .",
    )
    parser.add_argument(
        "--incumbent-serving-root",
        type=Path,
        default=None,
        help=f"defaults to <data-root>/{INCUMBENT_SERVING_RELATIVE} .",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--production-sha", default=None)
    parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help=(
            "where to write live_shadow_monitoring.json. Defaults to the "
            "accumulated state root, which is where the next run reads it."
        ),
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    state_root = args.state_root or (Path(args.data_root) / STATE_RELATIVE)
    grades_root = args.incumbent_grades_root or (
        Path(args.data_root) / INCUMBENT_GRADES_RELATIVE
    )
    serving_root = args.incumbent_serving_root or (
        Path(args.data_root) / INCUMBENT_SERVING_RELATIVE
    )
    status_path = args.status_path or (state_root / "live_shadow_monitoring.json")

    policy_path = Path(args.project_root) / live_policy.POLICY_RELATIVE
    policy = live_policy.load_policy(args.project_root)

    ingestion = ingest(
        state_root=state_root,
        slate_date=args.slate_date,
        policy=policy,
        shadow_log=args.shadow_log,
        shadow_status=args.shadow_status,
        shadow_grading=args.shadow_grading,
        production_sha=args.production_sha,
    )

    window = evidence_window(state_root)
    report = live_policy.evaluate(policy, window)

    payload = status_payload(
        report=report,
        window=window,
        policy=policy,
        policy_sha256=sha256_file(policy_path),
        slate_date=args.slate_date,
        production_sha=args.production_sha,
        ingestion=ingestion,
        incumbent_grades=incumbent_grade_summary(grades_root),
        incumbent=incumbent_identity(serving_root),
        state_root=state_root,
    )

    write_json_atomic(status_path, payload)

    if args.summary_path is not None:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.summary_path).open("a", encoding="utf-8") as handle:
            handle.write(render(payload))

    print(render(payload))

    if payload["safety_signal_raised"]:
        # The one case that fails the lifecycle. Reaching this state means the
        # policy found a publishing violation or an operational failure that
        # implicates the incumbent feed, and an operator has to see it. The
        # failure costs production nothing: serving and grading are already
        # done by this point, this step wrote no prediction and no serving
        # authority, and the incumbent never depended on the shadow.
        print(
            f"\nSHADOW DISABLED FOR SAFETY: {report.reason}\n"
            "The incumbent is unaffected and remains the served and published "
            "authority. Nothing was promoted and nothing was published.",
            file=sys.stderr,
        )
        return EXIT_SHADOW_DISABLED_FOR_SAFETY

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
