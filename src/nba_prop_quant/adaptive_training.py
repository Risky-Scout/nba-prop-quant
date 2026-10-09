"""Daily adaptive fitting under a frozen architecture.

This module orchestrates one production day's refit. It re-estimates numerical
parameters and nothing else: the model family, routing, feature definitions,
hyperparameters, seeds and policies are all frozen by the Step 3A adaptive
architecture contract and are verified before and after every stage.

The distinction it enforces is between FIT and SELECTION. Fitting re-estimates
parameters under a decision that was already made and certified. Selection
makes that decision. A nightly job that quietly re-selects a model family, a
calibration method or a dependence lambda is no longer running the certified
model, so every known selector entry point is denied here and the frozen policy
files are hashed before and after training.

Nothing in this module promotes a fit. Registration is immutable and leaves the
candidate REGISTERED / NOT PROMOTED; Step 3D owns live validation and
promotion.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import resource
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .adaptive_fit_registry import (
    ARCHITECTURE_REFERENCE_SHA,
    CONFIG_RELATIVE_PATH,
    CONTRACT_RELATIVE_PATH,
    FROZEN_POLICY_SOURCES,
    REQUIRED_VALIDATION_CHECKS,
    ArchitectureContract,
    FitMetadata,
    FitRegistry,
    canonical_json,
    config_hash,
    feature_schema_hash,
    frozen_policy_digests,
    gate3_candidate_policy_id,
    load_architecture_contract,
    sha256_file,
    write_json_atomic,
)
from .adaptive_validation import compute_validation_report
from .slate import resolve_slate_date


# --------------------------------------------------------------------------
# frozen protocol and contract locations
# --------------------------------------------------------------------------

UPDATE_PROTOCOL_RELATIVE_PATH = (
    Path("models")
    / "frozen_manifests"
    / "nba_prop_quant_v2_adaptive_update_protocol.json"
)

SERVING_SOURCE_CONTRACT_RELATIVE_PATH = (
    Path("models")
    / "frozen_manifests"
    / "nba_prop_quant_v2_adaptive_serving_source_contract.json"
)

PROTOCOL_VERSION = 1

TRAINING_MANIFEST_SCHEMA_VERSION = 1

BENCHMARK_SCHEMA_VERSION = 1

# Step 3B writes this after a successful rolling refresh. The trainer refuses
# to spend hours fitting against a rolling tree it has not verified first.
ROLLING_STATE_RELATIVE_PATH = Path(".state") / "current_season_state.json"

LOCK_RELATIVE_PATH = Path(".locks") / "adaptive_daily_fit.lock"


# --------------------------------------------------------------------------
# frozen training window
# --------------------------------------------------------------------------

# Step 3A freezes the window's shape, not its end. Raw ingest starts here so
# career and experience features resolve; the mean models train from the
# advanced floor onward. The end expands through each day's verified cutoff.
HISTORY_START_SEASON = 2001

ADVANCED_START_SEASON = 2015

FROZEN_MEAN_ROUTES = {
    "pts": "xgb",
    "reb": "ensemble",
    "ast": "xgb",
    "stl": "xgb",
    "blk": "ensemble",
    "fg3m": "xgb",
}

FROZEN_MARGINAL_FAMILY = "zinb"

FROZEN_DEPENDENCE_LAMBDA = {
    "points_assists": 0.0,
    "points_rebounds": 0.0,
    "points_rebounds_assists": 0.0,
    "rebounds_assists": 0.85,
    "stocks": 0.0,
}

FROZEN_CALIBRATION_ROUTES = {
    "assists": "prop",
    "blocks": "raw",
    "points": "prop",
    "points_assists": "prop",
    "points_rebounds": "prop",
    "points_rebounds_assists": "prop",
    "rebounds": "prop",
    "rebounds_assists": "prop",
    "steals": "raw",
    "threes": "prop",
}

CORE_SEED = 73

GATE3_SEED = 20260830

EFFECTIVE_N_JOBS = 2

# A fitted calibration unit needs enough signal to be worth trusting.
CALIBRATION_MIN_ROWS = 100

CALIBRATION_MIN_CLASSES = 2


# --------------------------------------------------------------------------
# selector prohibition
# --------------------------------------------------------------------------

# Research entry points that decide architecture. None may run during a daily
# fit. Running one would silently turn the nightly job into model selection.
FORBIDDEN_SELECTOR_SCRIPTS = frozenset(
    {
        "scripts/03_tune_dynamic_priors.py",
        "scripts/06c_select_mean_models.py",
        "scripts/08c_tune_copula_shrinkage.py",
        "scripts/08d_refine_copula_shrinkage_cv.py",
        "scripts/09d_select_probability_calibration.py",
        "scripts/13_gate2_certify_v2.py",
    }
)

# Callables that perform a search or a family decision. The daily DAG must
# reach none of them.
FORBIDDEN_SELECTOR_CALLABLES = (
    ("nba_prop_quant.model", "expanding_time_oof_target"),
    ("nba_prop_quant.model", "expanding_time_oof_minutes"),
    ("nba_prop_quant.decay", "tune_decay_beta"),
    ("nba_prop_quant.kalman", "tune_kalman"),
)

# The only OOF methodology production adaptive fitting may use. The expanding
# time variants split on row position rather than season and are research
# tools; they are denied above.
REQUIRED_OOF_FUNCTIONS = (
    "season_walk_forward_oof_minutes",
    "season_walk_forward_oof_target",
)


# --------------------------------------------------------------------------
# outcomes and errors
# --------------------------------------------------------------------------

OUTCOME_COMPLETED = "COMPLETED"
OUTCOME_NO_NEW_TRAINING_DATA = "NO_NEW_TRAINING_DATA"
OUTCOME_ALREADY_RUNNING = "ALREADY_RUNNING"
OUTCOME_DRY_RUN = "DRY_RUN_OK"

MODE_DRY_RUN = "dry-run"
MODE_BENCHMARK_ONLY = "benchmark-only"
MODE_REGISTER_CANDIDATE = "register-candidate"


class AdaptiveTrainingError(RuntimeError):
    """Base class for every fail-closed refusal in the daily fitter."""


class FrozenPolicyViolation(AdaptiveTrainingError):
    """A frozen policy, contract or protocol changed."""


class SelectorInvocationRefused(AdaptiveTrainingError):
    """The daily DAG attempted to reach a research selector."""


class DataStateError(AdaptiveTrainingError):
    """The rolling data state is missing, unverified or inconsistent."""


class TrainingLocked(AdaptiveTrainingError):
    """Another daily fit holds the exclusive training lock."""


class CandidateIncomplete(AdaptiveTrainingError):
    """The candidate artifact tree is missing or unusable."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_canonical(payload: Any) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def peak_rss_bytes() -> int:
    """Peak resident set size via the stdlib, in bytes.

    ru_maxrss is kilobytes on Linux and bytes on macOS.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return int(usage) if platform.system() == "Darwin" else int(usage) * 1024


def directory_size_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def package_versions() -> dict[str, str]:
    """Versions of the libraries whose numerics the fit depends on."""
    import importlib

    versions: dict[str, str] = {}

    for name in (
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "sklearn",
        "xgboost",
    ):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue

        versions[name] = str(getattr(module, "__version__", "unknown"))

    return versions


# --------------------------------------------------------------------------
# frozen policy write guard
# --------------------------------------------------------------------------


def frozen_guard_paths(project_root: Path) -> dict[str, Path]:
    """Every file whose content the daily fit must leave untouched."""
    paths = {
        "architecture_contract": project_root / CONTRACT_RELATIVE_PATH,
        "adaptive_update_protocol": (
            project_root / UPDATE_PROTOCOL_RELATIVE_PATH
        ),
        "adaptive_serving_source_contract": (
            project_root / SERVING_SOURCE_CONTRACT_RELATIVE_PATH
        ),
        "model_config": project_root / CONFIG_RELATIVE_PATH,
    }

    for name, source in FROZEN_POLICY_SOURCES.items():
        paths[name] = project_root / source["path"]

    return paths


class FrozenPolicyGuard:
    """Hash the frozen inputs, then prove they did not move.

    A daily fit that rewrites a selection or policy file has performed
    selection, whatever its intent. Verifying after every stage localises the
    stage that did it rather than reporting it only at the end.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.baseline: dict[str, str] = {}
        self.checkpoints: list[str] = []

    def snapshot(self) -> dict[str, str]:
        self.baseline = {}

        for name, path in sorted(frozen_guard_paths(self.project_root).items()):
            if not path.exists():
                raise FrozenPolicyViolation(
                    f"frozen input {name} is missing at {path}"
                )

            self.baseline[name] = sha256_file(path)

        return dict(self.baseline)

    def verify(self, stage: str) -> None:
        if not self.baseline:
            raise FrozenPolicyViolation(
                "frozen policy guard was verified before it was taken"
            )

        moved: list[str] = []

        for name, expected in sorted(self.baseline.items()):
            path = frozen_guard_paths(self.project_root)[name]

            if not path.exists():
                moved.append(f"{name} (deleted)")
                continue

            if sha256_file(path) != expected:
                moved.append(name)

        if moved:
            raise FrozenPolicyViolation(
                f"frozen policy changed during stage {stage!r}: "
                + ", ".join(moved)
                + ". A daily fit may re-estimate numerical parameters only."
            )

        self.checkpoints.append(stage)


@contextmanager
def selector_guard() -> Iterator[None]:
    """Make every known research selector raise if the DAG reaches it.

    Auditing the call graph proves a selector is not reached today. This proves
    it at run time, so a future edit that wires one in fails immediately rather
    than silently re-selecting the architecture.
    """
    import importlib

    patched: list[tuple[Any, str, Any]] = []

    for module_name, attribute in FORBIDDEN_SELECTOR_CALLABLES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        if not hasattr(module, attribute):
            continue

        original = getattr(module, attribute)

        def refuse(*args, _name=f"{module_name}.{attribute}", **kwargs):
            raise SelectorInvocationRefused(
                f"{_name} is a research selector and must never run during a "
                "daily adaptive fit"
            )

        setattr(module, attribute, refuse)
        patched.append((module, attribute, original))

    try:
        yield
    finally:
        for module, attribute, original in patched:
            setattr(module, attribute, original)


def assert_selector_script_not_requested(script: str) -> None:
    normalised = str(script).replace("\\", "/")

    for forbidden in sorted(FORBIDDEN_SELECTOR_SCRIPTS):
        if normalised.endswith(forbidden) or normalised == forbidden:
            raise SelectorInvocationRefused(
                f"{forbidden} performs model or policy selection and must "
                "never run during a daily adaptive fit"
            )


# --------------------------------------------------------------------------
# rolling data state
# --------------------------------------------------------------------------


def load_rolling_state(data_root: Path) -> dict[str, Any]:
    path = Path(data_root) / ROLLING_STATE_RELATIVE_PATH

    if not path.exists():
        raise DataStateError(
            f"no rolling-state record at {path}. Run the Step 3B refresh "
            "before fitting; the trainer does not fetch data itself."
        )

    state = _load_json(path)

    if not isinstance(state, dict) or "datasets" not in state:
        raise DataStateError(f"rolling-state record is malformed: {path}")

    return state


def verify_rolling_state(
    data_root: Path,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-derive the semantic fingerprints before spending hours fitting.

    The record is only worth trusting if the rolling tree still matches it, so
    every dataset is re-read and re-fingerprinted here rather than taken on
    faith.
    """
    # Imported lazily so the guards above stay usable without pandas.
    import pandas as pd

    from .storage import sort_by_keys

    data_root = Path(data_root)

    if state is None:
        state = load_rolling_state(data_root)

    datasets = state.get("datasets")

    if not isinstance(datasets, dict) or not datasets:
        raise DataStateError("rolling-state record lists no datasets")

    for label, record in sorted(datasets.items()):
        relative = str(record["relative_path"])
        path = data_root / relative

        if not path.exists():
            raise DataStateError(
                f"rolling dataset {label} is recorded at {relative} but the "
                "file is missing"
            )

        frame = pd.read_parquet(path)

        if len(frame) != int(record["row_count"]):
            raise DataStateError(
                f"rolling dataset {label} holds {len(frame)} rows but the "
                f"state record says {record['row_count']}"
            )

        observed_columns = sorted(str(column) for column in frame.columns)

        if observed_columns != list(record["columns"]):
            raise DataStateError(
                f"rolling dataset {label} columns differ from the state record"
            )

        key_columns = list(record["key_columns"])

        ordered = sort_by_keys(frame, key_columns)
        ordered = ordered[sorted(ordered.columns)]

        canonical = ordered.copy()

        for column in canonical.columns:
            values = canonical[column]

            if pd.api.types.is_datetime64_any_dtype(values):
                canonical[column] = values.dt.strftime("%Y-%m-%dT%H:%M:%S")

        payload = canonical.to_csv(index=False, float_format="%.12g")

        observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        if observed != str(record["semantic_fingerprint"]):
            raise DataStateError(
                f"rolling dataset {label} no longer matches its semantic "
                "fingerprint; the refresh and the fit disagree about the data"
            )

    return state


def derive_training_cutoff(
    state: dict[str, Any],
    slate_date: Any,
) -> date:
    """Latest verified completed data date, and strictly before the slate.

    The caller never supplies the cutoff. It is read from what the verified
    rolling state actually contains, so a caller cannot widen the information
    set by asking for a later one.
    """
    resolved_slate = resolve_slate_date(slate_date)

    observed: list[date] = []

    for label, record in sorted(state["datasets"].items()):
        # games legitimately holds future scheduled rows, so it cannot bound
        # the completed-information cutoff.
        if label == "games":
            continue

        maximum = record.get("max_date")

        if maximum:
            observed.append(date.fromisoformat(str(maximum)))

    if not observed:
        raise DataStateError(
            "rolling state records no completed history date, so no training "
            "cutoff can be derived"
        )

    cutoff = min(observed)

    if cutoff >= resolved_slate:
        raise DataStateError(
            f"verified history ends {cutoff}, which is not strictly before "
            f"slate date {resolved_slate}. Training on it would leak "
            "same-or-later-day results into the information set."
        )

    return cutoff


def assert_cutoff_not_widened(requested: Any, derived: date) -> None:
    if requested is None:
        return

    asked = resolve_slate_date(requested)

    if asked > derived:
        raise DataStateError(
            f"requested training cutoff {asked} exceeds the verified data "
            f"cutoff {derived}; the trainer never accepts a cutoff the data "
            "does not support"
        )


# --------------------------------------------------------------------------
# no-new-data detection
# --------------------------------------------------------------------------


def parent_fit_information(registry: FitRegistry) -> dict[str, Any] | None:
    """The most recent registered fit, preferring the promoted good one."""
    current = registry.current().get("current_good_fit_id")

    if current:
        try:
            return registry.load_manifest(current)
        except Exception:
            pass

    fits = registry.list_fits()

    if not fits:
        return None

    latest: dict[str, Any] | None = None

    for fit_id in fits:
        try:
            manifest = registry.load_manifest(fit_id)
        except Exception:
            continue

        if latest is None or str(manifest.get("training_cutoff", "")) > str(
            latest.get("training_cutoff", "")
        ):
            latest = manifest

    return latest


def has_new_training_data(
    parent: dict[str, Any] | None,
    training_cutoff: date,
    training_manifest_hash: str,
) -> bool:
    """True when today's information set is genuinely new.

    An NBA off day produces no newly completed games, so both the cutoff and
    the training-data manifest are unchanged. Refitting then would mint a
    different fit_id describing the same information, which is noise in the
    registry and wasted compute.

    The comparison is against the manifest hash rather than the rolling
    fingerprint alone, because the manifest also covers the contracts, the
    protocol and the source commit: a code change with unchanged data is
    genuinely a new fit.
    """
    if parent is None:
        return True

    if str(parent.get("training_cutoff", "")) != training_cutoff.isoformat():
        return True

    identity = parent.get("identity_inputs") or {}

    recorded = str(
        identity.get("training_data_manifest_hash")
        or parent.get("training_data_manifest_hash")
        or ""
    )

    return recorded != training_manifest_hash


# --------------------------------------------------------------------------
# isolated workspace and lock
# --------------------------------------------------------------------------


@contextmanager
def training_lock(work_root: Path, blocking: bool = False) -> Iterator[None]:
    path = Path(work_root) / LOCK_RELATIVE_PATH

    path.parent.mkdir(parents=True, exist_ok=True)

    handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)

    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB

    try:
        try:
            fcntl.flock(handle, flags)
        except OSError as error:
            raise TrainingLocked(
                f"another adaptive daily fit holds {path}"
            ) from error

        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


@dataclass(frozen=True)
class TrainingWorkspace:
    """A unique isolated staging tree for one candidate fit.

    Existing training code may write its ordinary models/ and processed/ paths,
    provided those resolve inside here. Nothing in the repository, the serving
    tree or the registry's finalized fits is writable from this workspace.
    """

    root: Path

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def candidate(self) -> Path:
        return self.root / "candidate"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def create(self) -> "TrainingWorkspace":
        for path in (
            self.inputs,
            self.models,
            self.processed,
            self.candidate,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)

        return self


def new_workspace(work_root: Path, slate_date: Any) -> TrainingWorkspace:
    root = Path(work_root) / "staging"

    root.mkdir(parents=True, exist_ok=True)

    stamp = resolve_slate_date(slate_date).isoformat().replace("-", "")

    return TrainingWorkspace(
        root=Path(
            tempfile.mkdtemp(prefix=f"fit_{stamp}_", dir=root)
        )
    ).create()


def assert_workspace_is_isolated(
    workspace: TrainingWorkspace,
    project_root: Path,
    registry: FitRegistry | None,
) -> None:
    """Refuse a workspace that could overwrite serving or registry state."""
    root = workspace.root.resolve()
    repository = Path(project_root).resolve()

    if root == repository or repository in root.parents:
        raise AdaptiveTrainingError(
            f"training workspace {root} is inside the repository at "
            f"{repository}; daily fitting must never write over the serving "
            "tree"
        )

    if registry is not None:
        fits = registry.fits_dir.resolve()

        if root == fits or fits in root.parents or root in fits.parents:
            raise AdaptiveTrainingError(
                f"training workspace {root} overlaps the registry's immutable "
                f"fits at {fits}"
            )


# --------------------------------------------------------------------------
# immutable training input snapshot
# --------------------------------------------------------------------------


SEASON_DIRECTORY = "season="


def _season_of(directory: Path) -> int | None:
    if not directory.is_dir() or not directory.name.startswith(
        SEASON_DIRECTORY
    ):
        return None

    try:
        return int(directory.name.split("=", 1)[1])
    except (IndexError, ValueError):
        return None


def eligible_history_files(
    data_root: Path,
    training_cutoff: date,
) -> list[str]:
    """Every historical file the frozen training pipeline reads.

    The Step 3B rolling state authenticates the latest coherent partition; it
    does not define the training corpus. Fitting needs the whole expanding
    window, so the seasons are enumerated from the frozen floors up to the
    season containing the cutoff. Row-level eligibility is enforced separately,
    after assembly.
    """
    data_root = Path(data_root)

    relatives: list[str] = []

    seasons_root = data_root / "raw" / "seasons"

    if seasons_root.is_dir():
        for directory in sorted(seasons_root.iterdir()):
            season = _season_of(directory)

            if season is None:
                continue

            if season < HISTORY_START_SEASON or season > training_cutoff.year:
                continue

            for name in ("stats.parquet", "games.parquet"):
                path = directory / name

                if path.exists():
                    relatives.append(
                        path.relative_to(data_root).as_posix()
                    )

    advanced_root = data_root / "raw" / "advanced"

    if advanced_root.is_dir():
        for directory in sorted(advanced_root.iterdir()):
            season = _season_of(directory)

            if season is None:
                continue

            # The advanced floor is frozen at 2015 and is preserved here
            # rather than narrowed to whatever the rolling state holds.
            if season < ADVANCED_START_SEASON or season > training_cutoff.year:
                continue

            path = directory / "advanced.parquet"

            if path.exists():
                relatives.append(path.relative_to(data_root).as_posix())

    players = data_root / "raw" / "players.parquet"

    if players.exists():
        relatives.append(players.relative_to(data_root).as_posix())

    return sorted(set(relatives))


def eligible_content_hash(path: Path, training_cutoff: date) -> str:
    """Hash only the rows this fit may actually learn from.

    A file carrying future scheduled rows changes whenever the schedule
    changes. Keying the information set on raw bytes would make a schedule
    edit look like newly completed NBA data and trigger a pointless retrain, so
    eligibility is measured on the cutoff-filtered content instead.
    """
    import pandas as pd

    frame = pd.read_parquet(path)

    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")

        frame = frame.loc[
            dates.dt.normalize() <= pd.Timestamp(training_cutoff)
        ]

    ordered = frame[sorted(frame.columns)].copy()

    for column in ordered.columns:
        if pd.api.types.is_datetime64_any_dtype(ordered[column]):
            ordered[column] = ordered[column].dt.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

    if "date" in ordered.columns:
        ordered = ordered.sort_values(
            by=sorted(ordered.columns), kind="stable"
        )

    payload = ordered.to_csv(index=False, float_format="%.12g")

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_training_corpus(
    data_root: Path,
    training_cutoff: date,
) -> tuple[dict[str, str], dict[str, str]]:
    """Raw and cutoff-eligible hashes for the whole historical corpus."""
    data_root = Path(data_root)

    raw: dict[str, str] = {}
    eligible: dict[str, str] = {}

    for relative in eligible_history_files(data_root, training_cutoff):
        path = data_root / relative

        raw[relative] = sha256_file(path)

        if path.suffix == ".parquet":
            eligible[relative] = eligible_content_hash(path, training_cutoff)

    return raw, eligible


def snapshot_training_inputs(
    data_root: Path,
    workspace: TrainingWorkspace,
    state: dict[str, Any],
    training_cutoff: date,
) -> dict[str, str]:
    """Copy the whole eligible historical corpus into the workspace.

    A long fit must not read files that can change underneath it, so every
    load-bearing input is copied once and hashed here, and every later stage
    reads the copy rather than the mutable data root.
    """
    data_root = Path(data_root)

    hashes: dict[str, str] = {}

    for relative in eligible_history_files(data_root, training_cutoff):
        source = data_root / relative
        destination = workspace.inputs / relative

        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(source, destination)

        hashes[relative] = sha256_file(destination)

    # The rolling state authenticates the latest partition, so those files must
    # be present in the snapshot and must still match what it recorded.
    for label, record in sorted(state["datasets"].items()):
        relative = str(record["relative_path"])

        if relative not in hashes:
            raise DataStateError(
                f"the rolling state's {label} partition at {relative} is not "
                "part of the snapshotted training corpus; the corpus and the "
                "authenticated rolling generation disagree"
            )

        if hashes[relative] != sha256_file(data_root / relative):
            raise DataStateError(
                f"{relative} changed while it was being snapshotted"
            )

    if not hashes:
        raise DataStateError(
            f"no eligible historical training files found under {data_root}"
        )

    return hashes


def build_training_data_manifest(
    *,
    project_root: Path,
    state: dict[str, Any],
    slate_date: date,
    training_cutoff: date,
    input_hashes: dict[str, str],
    eligible_hashes: dict[str, str],
    contract: ArchitectureContract,
    source_commit_sha: str,
) -> dict[str, Any]:
    """Describe exactly the information set this fit is allowed to see.

    Two hash sets are recorded because they answer different questions.
    input_sha256 is the lineage of the exact bytes fitting consumed.
    eligible_input_sha256 is the cutoff-filtered content, and it is what the
    fitting information digest is taken over, so a schedule edit that adds no
    completed game cannot masquerade as new training data.

    Deterministic: no wall-clock field, so two days seeing the same information
    produce the same digest, which is what lets an off day be recognised
    without fitting anything.
    """
    project_root = Path(project_root)

    manifest = {
        "advanced_start_season": ADVANCED_START_SEASON,
        "architecture_contract_sha256": contract.sha256,
        "architecture_reference_sha": ARCHITECTURE_REFERENCE_SHA,
        "adaptive_serving_source_contract_sha256": sha256_file(
            project_root / SERVING_SOURCE_CONTRACT_RELATIVE_PATH
        ),
        "adaptive_update_protocol_sha256": sha256_file(
            project_root / UPDATE_PROTOCOL_RELATIVE_PATH
        ),
        "config_hash": config_hash(project_root),
        "dependency_environment": {
            "packages": package_versions(),
            "python_version": platform.python_version(),
        },
        "feature_schema_hash": feature_schema_hash(project_root),
        "frozen_policy_digests": frozen_policy_digests(project_root),
        "history_start_season": HISTORY_START_SEASON,
        # Relative to the data root, so the manifest stays portable.
        "input_sha256": dict(sorted(input_hashes.items())),
        "eligible_input_sha256": dict(sorted(eligible_hashes.items())),
        "training_corpus_file_count": len(input_hashes),
        "rolling_state_datasets": {
            label: {
                "max_date": record.get("max_date"),
                "min_date": record.get("min_date"),
                "relative_path": record["relative_path"],
                "row_count": record["row_count"],
                "semantic_fingerprint": record["semantic_fingerprint"],
                "unique_key_count": record.get("unique_key_count"),
            }
            for label, record in sorted(state["datasets"].items())
        },
        "rolling_state_fingerprint": state["datasets_fingerprint"],
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "slate_date": slate_date.isoformat(),
        "source_commit_sha": source_commit_sha,
        "training_cutoff": training_cutoff.isoformat(),
    }

    # What the fit may actually learn from, plus the frozen inputs that decide
    # how. Deliberately excludes input_sha256 so schedule churn beyond the
    # cutoff cannot look like new completed information.
    manifest["fitting_information_digest"] = sha256_canonical(
        {
            "adaptive_update_protocol_sha256": manifest[
                "adaptive_update_protocol_sha256"
            ],
            "architecture_contract_sha256": manifest[
                "architecture_contract_sha256"
            ],
            "config_hash": manifest["config_hash"],
            "eligible_input_sha256": manifest["eligible_input_sha256"],
            "feature_schema_hash": manifest["feature_schema_hash"],
            "frozen_policy_digests": manifest["frozen_policy_digests"],
            "source_commit_sha": manifest["source_commit_sha"],
            "training_cutoff": manifest["training_cutoff"],
        }
    )

    return manifest


# --------------------------------------------------------------------------
# benchmark instrumentation
# --------------------------------------------------------------------------

DAG_STAGES = (
    "input_snapshot",
    "feature_build",
    "minutes_fit",
    "target_fit",
    "ensemble_weight_fit",
    "marginal_fit",
    "dependence_fit",
    "calibration_fit",
    "gate3_fit",
    "candidate_assembly",
    "validation",
    "registration",
)


@dataclass
class BenchmarkRecorder:
    """Times the real production path, not a parallel benchmark path."""

    started_at: str = field(default_factory=_utc_now)
    started_monotonic: float = field(default_factory=time.monotonic)
    stages: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in DAG_STAGES:
            raise AdaptiveTrainingError(f"unknown benchmark stage: {name}")

        begin = time.monotonic()

        try:
            yield
        finally:
            self.stages[name] = round(time.monotonic() - begin, 6)

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + int(amount)

    def record(self, name: str, value: Any) -> None:
        self.details[name] = value

    def report(
        self,
        *,
        mode: str,
        outcome: str,
        workspace: TrainingWorkspace | None = None,
    ) -> dict[str, Any]:
        return {
            "counters": dict(sorted(self.counters.items())),
            "cpu_count": os.cpu_count(),
            "details": dict(sorted(self.details.items())),
            "effective_n_jobs": EFFECTIVE_N_JOBS,
            "ended_at_utc": _utc_now(),
            "mode": mode,
            "outcome": outcome,
            "package_versions": package_versions(),
            "peak_rss_bytes": peak_rss_bytes(),
            "python_version": platform.python_version(),
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "stage_seconds": dict(sorted(self.stages.items())),
            "started_at_utc": self.started_at,
            "total_seconds": round(
                time.monotonic() - self.started_monotonic, 6
            ),
            "candidate_artifact_bytes": (
                directory_size_bytes(workspace.candidate)
                if workspace is not None and workspace.candidate.exists()
                else None
            ),
        }


# --------------------------------------------------------------------------
# fit engine
# --------------------------------------------------------------------------


class FitEngine(Protocol):
    """The numerical work of one daily fit.

    Production and benchmark runs both use ProductionFitEngine, so the
    benchmark exercises exactly the functions production uses. Tests supply a
    small deterministic engine so the orchestration, guards and registry
    integration can be exercised without an hours-long historical fit.
    """

    def build_features(self, context: "FitContext") -> None: ...

    def fit_minutes(self, context: "FitContext") -> None: ...

    def fit_targets(self, context: "FitContext") -> None: ...

    def fit_ensemble_weights(self, context: "FitContext") -> None: ...

    def fit_marginals(self, context: "FitContext") -> None: ...

    def fit_dependence(self, context: "FitContext") -> None: ...

    def fit_calibration(self, context: "FitContext") -> None: ...

    def fit_gate3(self, context: "FitContext") -> None: ...

    def assemble_candidate(self, context: "FitContext") -> None: ...


@dataclass
class FitContext:
    """Everything a stage needs, and the only place it may write."""

    project_root: Path
    data_root: Path
    workspace: TrainingWorkspace
    contract: ArchitectureContract
    slate_date: date
    training_cutoff: date
    benchmark: BenchmarkRecorder
    state: dict[str, Any]
    training_manifest: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    calibration_fallbacks: dict[str, str] = field(default_factory=dict)

    @property
    def mean_routes(self) -> dict[str, str]:
        return dict(FROZEN_MEAN_ROUTES)

    @property
    def marginal_family(self) -> str:
        return FROZEN_MARGINAL_FAMILY

    @property
    def dependence_lambda(self) -> dict[str, float]:
        return dict(FROZEN_DEPENDENCE_LAMBDA)

    @property
    def calibration_routes(self) -> dict[str, str]:
        return dict(FROZEN_CALIBRATION_ROUTES)

    def model_params(self) -> dict[str, Any]:
        """Frozen hyperparameters, never searched."""
        import yaml

        config = yaml.safe_load(
            (self.project_root / CONFIG_RELATIVE_PATH).read_text(
                encoding="utf-8"
            )
        )

        params = dict(config["model"])

        params["random_state"] = CORE_SEED
        params["n_jobs"] = EFFECTIVE_N_JOBS

        return params


def calibration_unit_is_fittable(rows: int, classes: int) -> bool:
    return rows >= CALIBRATION_MIN_ROWS and classes >= CALIBRATION_MIN_CLASSES


def resolve_calibration_parameters(
    prop_type: str,
    rows: int,
    classes: int,
    fitter: Callable[[], dict[str, float]],
    prior_good: dict[str, Any] | None,
) -> tuple[dict[str, float] | None, str]:
    """Fit a PROP unit, or fall back to a compatible prior-good parameter.

    RAW stays RAW: this is never called for a raw route, because inventing a
    Platt pair for one would silently change the calibration methodology.
    """
    route = FROZEN_CALIBRATION_ROUTES.get(prop_type)

    if route != "prop":
        raise FrozenPolicyViolation(
            f"{prop_type} is routed {route!r}, not 'prop'; a raw route has no "
            "fitted calibration parameters"
        )

    if calibration_unit_is_fittable(rows, classes):
        return dict(fitter()), "fitted"

    if not prior_good:
        raise AdaptiveTrainingError(
            f"{prop_type} has {rows} eligible rows and {classes} outcome "
            "class(es), below the production minimum, and no prior good "
            "parameter is available. Refusing to substitute an uncalibrated "
            "probability."
        )

    if str(prior_good.get("selected_method")) != "prop":
        raise AdaptiveTrainingError(
            f"{prop_type} prior good parameter was fitted under method "
            f"{prior_good.get('selected_method')!r} and is not compatible "
            "with the frozen 'prop' route"
        )

    parameters = prior_good.get("production_parameters")

    if not parameters:
        raise AdaptiveTrainingError(
            f"{prop_type} prior good record carries no production parameters"
        )

    return dict(parameters), "reused_prior_good"


# --------------------------------------------------------------------------
# candidate verification and validation
# --------------------------------------------------------------------------


def assert_candidate_tree_usable(candidate: Path) -> dict[str, str]:
    """Refuse symlinks, special files and an empty or unreadable tree."""
    import stat as stat_module

    if not candidate.is_dir():
        raise CandidateIncomplete(f"candidate tree is missing: {candidate}")

    hashes: dict[str, str] = {}

    for path in sorted(candidate.rglob("*")):
        relative = path.relative_to(candidate).as_posix()

        if path.is_symlink():
            raise CandidateIncomplete(
                f"symlink rejected in candidate tree: {relative}"
            )

        mode = os.lstat(path).st_mode

        if stat_module.S_ISDIR(mode):
            continue

        if not stat_module.S_ISREG(mode):
            raise CandidateIncomplete(
                f"special file rejected in candidate tree: {relative}"
            )

        hashes[relative] = sha256_file(path)

    if not hashes:
        raise CandidateIncomplete(f"candidate tree is empty: {candidate}")

    return hashes


# A candidate that carries only its own provenance has fitted nothing, so the
# serving artifacts are required explicitly rather than inferred from the tree
# being non-empty.
REQUIRED_CANDIDATE_PREFIXES = ("models/",)

REQUIRED_CANDIDATE_FILES = ("training_data_manifest.json",)


def assert_required_candidate_artifacts(
    artifact_hashes: dict[str, str],
) -> None:
    missing: list[str] = [
        name
        for name in REQUIRED_CANDIDATE_FILES
        if name not in artifact_hashes
    ]

    for prefix in REQUIRED_CANDIDATE_PREFIXES:
        if not any(name.startswith(prefix) for name in artifact_hashes):
            missing.append(f"{prefix}*")

    if missing:
        raise CandidateIncomplete(
            "candidate is missing required artifacts: "
            + ", ".join(sorted(missing))
        )


def structured_values_are_finite(candidate: Path) -> tuple[bool, list[str]]:
    """Check JSON numerical parameters for NaN and infinity.

    Only structured files are inspected. Serialized estimators are opaque here
    and are covered by the prediction smoke test instead.
    """
    offenders: list[str] = []

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{trail}{key}.")

        elif isinstance(node, list):
            for item in node:
                walk(item, trail)

        elif isinstance(node, float):
            if node != node or node in (float("inf"), float("-inf")):
                offenders.append(trail.rstrip("."))

    for path in sorted(candidate.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            offenders.append(f"{path.name} (unreadable)")
            continue

        walk(payload, f"{path.relative_to(candidate).as_posix()}:")

    return (not offenders), offenders


# Checks Step 3C can establish truthfully from an offline fit. The remaining
# two need a live captured lineup snapshot and a real T-20 capture, so Step 3C
# leaves them unset and the candidate is therefore not promotable.
STEP3C_VALIDATION_CHECKS = (
    "training_completed",
    "data_refresh_valid",
    "history_regression_check",
    "advanced_coverage_check",
    "required_artifacts_present",
    "artifact_hashes_valid",
    "finite_values_check",
    "feature_schema_match",
    "architecture_contract_match",
    "source_lineage_match",
    "marginal_fit_valid",
    "calibration_valid",
    "prediction_smoke_test",
)

STEP3D_VALIDATION_CHECKS = (
    "gate3_role_readiness",
    "t20_protocol_compatible",
)


def deferred_validation_checks() -> tuple[str, ...]:
    return tuple(
        name
        for name in REQUIRED_VALIDATION_CHECKS
        if name not in STEP3C_VALIDATION_CHECKS
    )


# The stages that actually fit something, which is what training_completed
# asks about. validation and registration are the stages that observe and
# record, so requiring them here would be circular.
FIT_STAGES = tuple(
    stage
    for stage in DAG_STAGES
    if stage not in ("validation", "registration")
)


# Provenance filename -> working-tree source. Every file a candidate carries in
# its provenance directory must be declared here, so source_lineage_match can
# verify it instead of skipping it.
def lineage_sources() -> dict[str, Path]:
    sources: dict[str, Path] = {
        Path(relative).name: Path(relative)
        for relative in (
            UPDATE_PROTOCOL_RELATIVE_PATH,
            CONTRACT_RELATIVE_PATH,
            SERVING_SOURCE_CONTRACT_RELATIVE_PATH,
            CONFIG_RELATIVE_PATH,
        )
    }

    for name, source in FROZEN_POLICY_SOURCES.items():
        sources[f"{name}.json"] = Path(source["path"])

    return sources


def snapshot_directory(data_root: Path) -> Path:
    """Where the T-20 captures live, by the same rule Settings uses.

    Settings derives snapshot_dir from the data directory. Deriving it the same
    way here keeps the two checks that need a capture reading the directory
    production writes, without the trainer needing an API key to construct a
    Settings object.
    """
    return Path(data_root) / "snapshots"


# --------------------------------------------------------------------------
# the daily fit
# --------------------------------------------------------------------------


def run_daily_fit(
    *,
    project_root: Path,
    data_root: Path,
    work_root: Path,
    slate_date: Any,
    registry: FitRegistry | None = None,
    engine: FitEngine | None = None,
    mode: str = MODE_DRY_RUN,
    source_commit_sha: str = "",
    requested_cutoff: Any = None,
    keep_workspace: bool = True,
) -> dict[str, Any]:
    """Run one day's adaptive fit under the frozen architecture.

    The same DAG serves every mode. dry-run stops after planning, benchmark-only
    and register-candidate run the identical fit path and differ only in whether
    the candidate is registered. No mode promotes.
    """
    if mode not in (MODE_DRY_RUN, MODE_BENCHMARK_ONLY, MODE_REGISTER_CANDIDATE):
        raise AdaptiveTrainingError(f"unknown mode: {mode}")

    project_root = Path(project_root)
    data_root = Path(data_root)
    work_root = Path(work_root)

    benchmark = BenchmarkRecorder()

    resolved_slate = resolve_slate_date(slate_date)

    guard = FrozenPolicyGuard(project_root)
    guard.snapshot()

    contract = load_architecture_contract(project_root)

    protocol = _load_json(project_root / UPDATE_PROTOCOL_RELATIVE_PATH)

    assert_protocol_matches_tree(protocol, project_root, contract)

    state = verify_rolling_state(data_root)

    training_cutoff = derive_training_cutoff(state, resolved_slate)

    assert_cutoff_not_widened(requested_cutoff, training_cutoff)

    plan = {
        "advanced_start_season": ADVANCED_START_SEASON,
        "architecture_contract_sha256": contract.sha256,
        "calibration_routes": dict(FROZEN_CALIBRATION_ROUTES),
        "dependence_lambda": dict(FROZEN_DEPENDENCE_LAMBDA),
        "history_start_season": HISTORY_START_SEASON,
        "marginal_family": FROZEN_MARGINAL_FAMILY,
        "mean_routes": dict(FROZEN_MEAN_ROUTES),
        "mode": mode,
        "oof_methodology": list(REQUIRED_OOF_FUNCTIONS),
        "rolling_state_fingerprint": state["datasets_fingerprint"],
        "slate_date": resolved_slate.isoformat(),
        "stages": list(DAG_STAGES),
        "training_cutoff": training_cutoff.isoformat(),
        "update_protocol_sha256": sha256_file(
            project_root / UPDATE_PROTOCOL_RELATIVE_PATH
        ),
    }

    if mode == MODE_DRY_RUN:
        guard.verify("dry_run")

        return {
            "benchmark": benchmark.report(mode=mode, outcome=OUTCOME_DRY_RUN),
            "outcome": OUTCOME_DRY_RUN,
            "plan": plan,
        }

    if engine is None:
        raise AdaptiveTrainingError(
            "a fit engine is required for benchmark-only and "
            "register-candidate runs"
        )

    try:
        lock = training_lock(work_root)
        lock.__enter__()
    except TrainingLocked as error:
        return {
            "message": str(error),
            "outcome": OUTCOME_ALREADY_RUNNING,
            "plan": plan,
        }

    workspace: TrainingWorkspace | None = None

    try:
        parent = (
            parent_fit_information(registry) if registry is not None else None
        )

        # Built before any fitting, from the data root itself, so an off day
        # costs a few hashes rather than a full retrain.
        corpus_hashes, eligible_hashes = hash_training_corpus(
            data_root, training_cutoff
        )

        prospective_manifest = build_training_data_manifest(
            project_root=project_root,
            state=state,
            slate_date=resolved_slate,
            training_cutoff=training_cutoff,
            input_hashes=corpus_hashes,
            eligible_hashes=eligible_hashes,
            contract=contract,
            source_commit_sha=source_commit_sha,
        )

        manifest_hash = prospective_manifest["fitting_information_digest"]

        if not has_new_training_data(parent, training_cutoff, manifest_hash):
            guard.verify("no_new_training_data")

            return {
                "benchmark": benchmark.report(
                    mode=mode, outcome=OUTCOME_NO_NEW_TRAINING_DATA
                ),
                "outcome": OUTCOME_NO_NEW_TRAINING_DATA,
                "parent_fit_id": (parent or {}).get("fit_id"),
                "plan": plan,
                "reason": (
                    "no newly completed eligible NBA information since the "
                    f"parent fit; cutoff is still "
                    f"{training_cutoff.isoformat()}"
                ),
                "training_data_manifest_sha256": manifest_hash,
            }

        workspace = new_workspace(work_root, resolved_slate)

        assert_workspace_is_isolated(workspace, project_root, registry)

        context = FitContext(
            project_root=project_root,
            data_root=data_root,
            workspace=workspace,
            contract=contract,
            slate_date=resolved_slate,
            training_cutoff=training_cutoff,
            benchmark=benchmark,
            state=state,
        )

        with selector_guard():
            with benchmark.stage("input_snapshot"):
                snapshotted = snapshot_training_inputs(
                    data_root, workspace, state, training_cutoff
                )

                if snapshotted != prospective_manifest["input_sha256"]:
                    raise DataStateError(
                        "training inputs changed while they were being "
                        "snapshotted; refusing to fit a moving information set"
                    )

                context.training_manifest = prospective_manifest

                benchmark.record(
                    "training_corpus_files", len(snapshotted)
                )

                write_json_atomic(
                    prospective_manifest,
                    workspace.candidate / "training_data_manifest.json",
                )

            guard.verify("input_snapshot")

            for stage_name, step in (
                ("feature_build", engine.build_features),
                ("minutes_fit", engine.fit_minutes),
                ("target_fit", engine.fit_targets),
                ("ensemble_weight_fit", engine.fit_ensemble_weights),
                ("marginal_fit", engine.fit_marginals),
                ("dependence_fit", engine.fit_dependence),
                ("calibration_fit", engine.fit_calibration),
                ("gate3_fit", engine.fit_gate3),
                ("candidate_assembly", engine.assemble_candidate),
            ):
                with benchmark.stage(stage_name):
                    step(context)

                guard.verify(stage_name)

        with benchmark.stage("validation"):
            artifact_hashes = assert_candidate_tree_usable(workspace.candidate)

            assert_required_candidate_artifacts(artifact_hashes)

            finite, offenders = structured_values_are_finite(
                workspace.candidate
            )

            if not finite:
                raise CandidateIncomplete(
                    "candidate holds non-finite structured values: "
                    + ", ".join(sorted(offenders)[:10])
                )

            report = compute_validation_report(
                project_root=project_root,
                candidate=workspace.candidate,
                contract_object=contract,
                manifest=context.training_manifest,
                state=state,
                parent_manifest=parent,
                artifact_hashes=artifact_hashes,
                finite=finite,
                finite_offenders=offenders,
                required_stages=FIT_STAGES,
                observed_stages=tuple(benchmark.stages),
                required_prefixes=REQUIRED_CANDIDATE_PREFIXES,
                required_files=REQUIRED_CANDIDATE_FILES,
                lineage_sources=lineage_sources(),
                targets=tuple(sorted(FROZEN_MEAN_ROUTES)),
                frozen_marginal_family=FROZEN_MARGINAL_FAMILY,
                calibration_routes=dict(FROZEN_CALIBRATION_ROUTES),
                calibration_hashes=dict(
                    context.notes.get("calibration_hashes") or {}
                ),
                calibration_fallbacks=dict(context.calibration_fallbacks),
                advanced_start_season=ADVANCED_START_SEASON,
                role_state_hash=context.notes.get("role_state_hash"),
                slate_date=resolved_slate.isoformat(),
                snapshot_dir=snapshot_directory(data_root),
            )

            # Every check is computed, so a failure here is a measurement and
            # the candidate must not be recorded as validated.
            if not report.passed:
                raise CandidateIncomplete(
                    "candidate failed computed validation checks: "
                    + "; ".join(
                        f"{name}: "
                        f"{report.by_name[name].error or 'no reason recorded'}"
                        for name in report.failed()
                    )
                )

            checks = report.recorded_checks()

            benchmark.record("candidate_artifact_count", len(artifact_hashes))

            benchmark.record("validation_checks_computed", len(report.results))

        guard.verify("validation")

        result: dict[str, Any] = {
            "outcome": OUTCOME_COMPLETED,
            "plan": plan,
            "training_data_manifest_sha256": context.training_manifest[
                "fitting_information_digest"
            ],
            "validation_checks": checks,
            "validation_report": report.payload(),
            "deferred_validation_checks": report.deferred(),
            "calibration_fallbacks": dict(context.calibration_fallbacks),
            "workspace": str(workspace.root),
        }

        if mode == MODE_REGISTER_CANDIDATE:
            if registry is None:
                raise AdaptiveTrainingError(
                    "register-candidate requires a registry root"
                )

            with benchmark.stage("registration"):
                metadata = FitMetadata(
                    fit_date=resolved_slate.isoformat(),
                    training_cutoff=training_cutoff.isoformat(),
                    source_commit_sha=source_commit_sha,
                    training_data_manifest_hash=result[
                        "training_data_manifest_sha256"
                    ],
                    python_version=platform.python_version(),
                    package_versions=package_versions(),
                    core_seed=CORE_SEED,
                    gate3_seed=GATE3_SEED,
                    effective_n_jobs=EFFECTIVE_N_JOBS,
                    gate3_candidate_policy_id=gate3_candidate_policy_id(
                        project_root
                    ),
                    calibration_hashes=context.notes.get(
                        "calibration_hashes", {}
                    ),
                    role_state_hash=context.notes.get("role_state_hash"),
                    training_started_at=benchmark.started_at,
                    training_completed_at=_utc_now(),
                    validation_lineage_reference=(
                        f"step3c:{resolved_slate.isoformat()}"
                    ),
                )

                fit_id = registry.register(
                    workspace.candidate, metadata, contract=contract
                )

                registry.record_validation(
                    fit_id, checks, actor="adaptive_daily_fit"
                )

            guard.verify("registration")

            result["fit_id"] = fit_id
            result["promoted"] = False

        result["benchmark"] = benchmark.report(
            mode=mode, outcome=OUTCOME_COMPLETED, workspace=workspace
        )

        write_json_atomic(
            result["benchmark"],
            Path(work_root) / "reports" / f"benchmark_{resolved_slate}.json",
        )

        return result

    finally:
        if (
            workspace is not None
            and not keep_workspace
            and workspace.root.exists()
        ):
            shutil.rmtree(workspace.root, ignore_errors=True)

        lock.__exit__(None, None, None)


def assert_protocol_matches_tree(
    protocol: dict[str, Any],
    project_root: Path,
    contract: ArchitectureContract,
) -> None:
    """Refuse a protocol that does not describe this working tree."""
    project_root = Path(project_root)

    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise FrozenPolicyViolation(
            f"unsupported adaptive update protocol version "
            f"{protocol.get('protocol_version')!r}"
        )

    expected = {
        "architecture_contract_sha256": contract.sha256,
        "architecture_reference_sha": ARCHITECTURE_REFERENCE_SHA,
        "adaptive_serving_source_contract_sha256": sha256_file(
            project_root / SERVING_SOURCE_CONTRACT_RELATIVE_PATH
        ),
        "config_hash": config_hash(project_root),
        "feature_schema_hash": feature_schema_hash(project_root),
    }

    for key, value in sorted(expected.items()):
        observed = protocol.get(key)

        if observed != value:
            raise FrozenPolicyViolation(
                f"adaptive update protocol {key} is {observed!r} but the tree "
                f"derives {value!r}"
            )

    recorded = protocol.get("frozen_policy_digests")
    derived = frozen_policy_digests(project_root)

    if recorded != derived:
        raise FrozenPolicyViolation(
            "adaptive update protocol frozen policy digests do not match the "
            "working tree"
        )


# --------------------------------------------------------------------------
# production fit engine
# --------------------------------------------------------------------------


def load_script_module(project_root: Path, relative: str):
    """Import a pipeline script for its pure fitting functions.

    scripts/ is not a package, so the functions are reached the same way
    scripts/14_build_gate3_deployment_artifacts.py already reaches the Gate 2
    module. Loading rather than copying keeps one definition of the
    mathematics: the daily fit cannot drift from the certified code because it
    calls that code. Every such script guards its entry point behind
    __main__, so importing runs no pipeline.
    """
    import importlib.util

    assert_selector_script_not_requested(relative)

    path = Path(project_root) / relative

    if not path.exists():
        raise AdaptiveTrainingError(f"pipeline script missing: {path}")

    spec = importlib.util.spec_from_file_location(
        f"adaptive_fit_source_{path.stem}", path
    )

    if spec is None or spec.loader is None:
        raise AdaptiveTrainingError(f"cannot load {path}")

    module = importlib.util.module_from_spec(spec)

    import sys as _sys

    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


@dataclass(frozen=True)
class SnapshotSettings:
    """Minimal settings shim so the canonical loaders read the snapshot.

    nba_prop_quant.pipeline's loaders take a settings object and use only
    raw_dir. Pointing them at the workspace copy is what keeps the daily fit
    reading immutable bytes while still using the certified assembly code.
    """

    raw_dir: Path


class ProductionFitEngine:
    """The real daily fit.

    Every stage calls the existing certified fitting function and supplies the
    frozen decision rather than deriving one. Benchmark runs use this engine
    too, so a benchmark measures the production path.

    This engine has not been exercised end to end in an environment without
    full historical NBA data. The required local full-data benchmark is what
    validates it; see docs/wizardofodds/ADAPTIVE_DAILY_TRAINING.md.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)

    # -- helpers --------------------------------------------------------

    def _eligible(self, frame, context: "FitContext", column: str = "date"):
        """Rows strictly before the training cutoff.

        Applied at every stage rather than once, so no later merge can
        reintroduce a row the cutoff excluded.
        """
        import pandas as pd

        dates = pd.to_datetime(frame[column], errors="coerce")

        cutoff = pd.Timestamp(context.training_cutoff)

        return frame.loc[dates.dt.normalize() <= cutoff].copy()

    # -- stages ---------------------------------------------------------

    def build_features(self, context: "FitContext") -> None:
        """Assemble the full expanding historical training frame.

        Uses the canonical loaders and build_base_frame, pointed at the
        immutable snapshot rather than the mutable data root, so the frame is
        the same one the certified pipeline builds. It spans every eligible
        season, not just the latest rolling partition.
        """
        from .features import add_dynamic_priors, build_base_frame
        from .pipeline import (
            load_advanced,
            load_history_box_stats,
            load_history_games,
            load_json,
            load_players,
        )

        snapshot = SnapshotSettings(raw_dir=context.workspace.inputs / "raw")

        stats = load_history_box_stats(snapshot)
        games = load_history_games(snapshot)
        players = load_players(snapshot)
        advanced = load_advanced(snapshot)

        if stats.empty:
            raise CandidateIncomplete(
                "the snapshotted training corpus holds no box-score history"
            )

        # Applied before assembly so a future scheduled row can never become a
        # training outcome. games keeps its future rows only as schedule
        # context, which build_base_frame joins by game id.
        eligible_stats = self._eligible(stats, context)

        eligible_advanced = (
            self._eligible(advanced, context)
            if not advanced.empty and "date" in advanced.columns
            else advanced
        )

        base = build_base_frame(
            eligible_stats,
            players=players,
            advanced=eligible_advanced,
            games=games,
        )

        params = load_json(
            self.project_root / "models" / "dynamic_params.json"
        )

        features = add_dynamic_priors(base, params)

        # Defence in depth: nothing downstream may see a post-cutoff row.
        features = self._eligible(features, context)

        seasons = sorted(
            int(season) for season in features["season"].dropna().unique()
        )

        if len(seasons) < 2:
            raise CandidateIncomplete(
                "the training frame spans "
                f"{len(seasons)} season(s) {seasons}; the frozen pipeline "
                "needs the expanding historical window, so the snapshot did "
                "not capture the full corpus"
            )

        context.benchmark.record("feature_rows", int(len(features)))
        context.benchmark.record(
            "feature_columns", int(len(features.columns))
        )
        context.benchmark.record("training_seasons", seasons)
        context.benchmark.record("training_season_count", len(seasons))

        features.to_parquet(
            context.workspace.processed / "features.parquet", index=False
        )

    def fit_minutes(self, context: "FitContext") -> None:
        import pandas as pd

        from .model import fit_minutes_model, season_walk_forward_oof_minutes

        frame = pd.read_parquet(
            context.workspace.processed / "features.parquet"
        )

        frame = frame.loc[
            frame["season"] >= ADVANCED_START_SEASON
        ].sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)

        params = context.model_params()

        frame["expected_minutes"] = season_walk_forward_oof_minutes(
            frame, params=params
        )

        context.benchmark.count(
            "xgboost_fits", int(frame["season"].nunique())
        )

        bundle = fit_minutes_model(frame, params=params)

        context.benchmark.count("xgboost_fits")

        bundle.save(context.workspace.models / "minutes.joblib")

        frame.to_parquet(
            context.workspace.processed / "stack_training.parquet",
            index=False,
        )

        context.benchmark.record("minutes_training_rows", int(len(frame)))

    def fit_targets(self, context: "FitContext") -> None:
        import pandas as pd

        from .features import TARGETS
        from .model import fit_target_model, season_walk_forward_oof_target

        frame = pd.read_parquet(
            context.workspace.processed / "stack_training.parquet"
        )

        frame = frame.loc[frame["expected_minutes"].notna()].copy()

        params = context.model_params()

        seasons = int(frame["season"].nunique())

        for target in TARGETS:
            frame[f"mu_{target}"] = season_walk_forward_oof_target(
                frame, target=target, params=params
            )

            context.benchmark.count("xgboost_fits", seasons)

            bundle = fit_target_model(frame, target=target, params=params)

            context.benchmark.count("xgboost_fits")

            bundle.save(context.workspace.models / f"{target}.joblib")

        frame.to_parquet(
            context.workspace.processed / "oof_predictions.parquet",
            index=False,
        )

        context.benchmark.record("oof_rows", int(len(frame)))

    def fit_ensemble_weights(self, context: "FitContext") -> None:
        """Refit weights for the two ensemble routes only.

        PTS, AST, STL and 3PM are frozen to the XGB route and have no weights
        to fit. No alternative family is evaluated for any target.
        """
        import json as _json

        import numpy as np
        import pandas as pd

        ensemble = load_script_module(
            self.project_root, "scripts/06b_fit_mean_ensemble.py"
        )

        frame = pd.read_parquet(
            context.workspace.processed / "oof_predictions.parquet"
        )

        weights: dict[str, dict[str, float]] = {}

        for target, route in sorted(FROZEN_MEAN_ROUTES.items()):
            if route != "ensemble":
                continue

            columns = [
                f"mu_{target}",
                f"decay_prior_{target}_rate",
                f"kalman_prior_{target}_rate",
            ]

            usable = frame.dropna(subset=[target, *columns])

            fitted = ensemble.fit_simplex_weights(
                usable[target].to_numpy(dtype=float),
                usable[columns].to_numpy(dtype=float),
            )

            weights[target] = {
                "xgb": float(fitted[0]),
                "decay": float(fitted[1]),
                "kalman": float(fitted[2]),
            }

            context.benchmark.count("ensemble_weight_fits")

        (context.workspace.models / "ensemble_weights.json").write_text(
            _json.dumps(weights, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        context.notes["ensemble_weights"] = weights

        _ = np

    def fit_marginals(self, context: "FitContext") -> None:
        """Refit frozen ZINB marginals from canonical selected means."""
        import joblib
        import pandas as pd

        marginals_module = load_script_module(
            self.project_root, "scripts/07_fit_marginals.py"
        )

        if not hasattr(marginals_module, "inflation_features_for"):
            raise RuntimeError(
                "scripts/07_fit_marginals.py is missing required "
                "inflation_features_for(target, frame)"
            )

        frame = pd.read_parquet(
            context.workspace.processed / "oof_predictions.parquet"
        )

        ensemble_weights = context.notes.get("ensemble_weights", {})

        # Materialize the frozen canonical selected-mean contract:
        # PTS/AST/STL/FG3M -> XGB
        # REB/BLK          -> fitted convex ensemble
        for target, route in sorted(FROZEN_MEAN_ROUTES.items()):
            xgb_col = f"mu_{target}"
            selected_col = f"mu_selected_{target}"

            if xgb_col not in frame.columns:
                raise RuntimeError(
                    f"Missing XGB OOF mean column for target={target}: "
                    f"{xgb_col}"
                )

            if route == "ensemble":
                weights = ensemble_weights.get(target)

                if not weights:
                    raise RuntimeError(
                        f"Missing fitted ensemble weights for target={target}"
                    )

                required_weight_keys = {"xgb", "decay", "kalman"}
                missing_weight_keys = (
                    required_weight_keys - set(weights)
                )

                if missing_weight_keys:
                    raise RuntimeError(
                        f"Incomplete ensemble weights for target={target}: "
                        f"missing {sorted(missing_weight_keys)}"
                    )

                decay_col = f"decay_prior_{target}_rate"
                kalman_col = f"kalman_prior_{target}_rate"

                missing_columns = [
                    column
                    for column in (decay_col, kalman_col)
                    if column not in frame.columns
                ]

                if missing_columns:
                    raise RuntimeError(
                        f"Missing ensemble source columns for target={target}: "
                        f"{missing_columns}"
                    )

                frame[selected_col] = (
                    float(weights["xgb"]) * frame[xgb_col]
                    + float(weights["decay"]) * frame[decay_col]
                    + float(weights["kalman"]) * frame[kalman_col]
                )

            elif route == "xgb":
                frame[selected_col] = frame[xgb_col]

            else:
                raise RuntimeError(
                    f"Unsupported frozen mean route for target={target}: "
                    f"{route!r}"
                )

        # Persist the canonical OOF selected means so every downstream
        # distribution/dependence stage consumes the same mean contract.
        frame.to_parquet(
            context.workspace.processed / "oof_selected_means.parquet",
            index=False,
        )

        fitted: dict[str, Any] = {}
        convergence: dict[str, bool] = {}

        for target in sorted(FROZEN_MEAN_ROUTES):
            selected_col = f"mu_selected_{target}"

            usable = frame.dropna(
                subset=[target, selected_col]
            ).copy()

            inflation = marginals_module.inflation_features_for(
                target,
                usable,
            )

            if not inflation:
                raise RuntimeError(
                    f"No ZINB inflation features available for "
                    f"target={target}"
                )

            non_numeric = [
                column
                for column in inflation
                if not pd.api.types.is_numeric_dtype(
                    usable[column].dtype
                )
            ]

            if non_numeric:
                raise TypeError(
                    f"Non-numeric ZINB inflation features for "
                    f"target={target}: {non_numeric}"
                )

            model = marginals_module.fit_candidate(
                FROZEN_MARGINAL_FAMILY,
                y=usable[target].to_numpy(dtype=int),
                mu=usable[selected_col].to_numpy(dtype=float),
                frame=usable,
                inflation_features=inflation,
            )

            fitted[target] = model
            convergence[target] = True

            context.benchmark.count("marginal_fits")

        joblib.dump(
            fitted,
            context.workspace.models / "marginals.joblib",
        )

        context.notes["marginal_convergence"] = convergence

    def fit_dependence(self, context: "FitContext") -> None:
        """Re-estimate the copula correlation, then apply the frozen lambda."""
        import joblib
        import pandas as pd

        from .copula import GaussianCopula

        frame = pd.read_parquet(
            context.workspace.processed / "oof_selected_means.parquet"
        )

        marginals = joblib.load(
            context.workspace.models / "marginals.joblib"
        )

        copula = GaussianCopula(targets=sorted(FROZEN_MEAN_ROUTES)).fit(
            frame,
            marginals=marginals,
            mu_columns={
                target: f"mu_selected_{target}" for target in sorted(FROZEN_MEAN_ROUTES)
            },
        )

        joblib.dump(copula, context.workspace.models / "copula.joblib")

        context.benchmark.count("copula_fits")

        # The frozen policy is applied, never searched.
        context.notes["dependence_lambda"] = dict(FROZEN_DEPENDENCE_LAMBDA)

    def fit_calibration(self, context: "FitContext") -> None:
        """Refit Platt parameters for PROP routes. RAW stays RAW."""
        import json as _json

        import pandas as pd

        calibration = load_script_module(
            self.project_root, "scripts/09c_fit_probability_calibration.py"
        )

        prior_policy = _load_json(
            self.project_root
            / "models"
            / "market_probability_calibration_policy.json"
        )

        contracts_path = (
            context.workspace.inputs / "market_calibration_contracts.parquet"
        )

        contracts = (
            pd.read_parquet(contracts_path)
            if contracts_path.exists()
            else pd.DataFrame()
        )

        policy: dict[str, Any] = {"props": {}}

        for prop_type, route in sorted(FROZEN_CALIBRATION_ROUTES.items()):
            if route == "raw":
                # A raw route carries no fitted parameters. Manufacturing a
                # Platt pair here would change the calibration methodology.
                policy["props"][prop_type] = {"selected_method": "raw"}
                continue

            eligible = (
                contracts.loc[contracts["prop_type"].eq(prop_type)]
                if not contracts.empty
                else pd.DataFrame()
            )

            if not eligible.empty:
                eligible = self._eligible(eligible, context, "game_date")

            rows = int(len(eligible))

            classes = (
                int(eligible["actual_over"].nunique()) if rows else 0
            )

            def fitter(frame=eligible):
                return calibration.fit_platt(
                    frame["model_probability"].to_numpy(dtype=float),
                    frame["actual_over"].to_numpy(dtype=float),
                    frame["weight"].to_numpy(dtype=float)
                    if "weight" in frame.columns
                    else None,
                )

            parameters, origin = resolve_calibration_parameters(
                prop_type,
                rows,
                classes,
                fitter,
                (prior_policy.get("props") or {}).get(prop_type),
            )

            policy["props"][prop_type] = {
                "selected_method": "prop",
                "production_parameters": parameters,
            }

            if origin != "fitted":
                context.calibration_fallbacks[prop_type] = origin
            else:
                context.benchmark.count("calibration_fits")

        (
            context.workspace.models
            / "market_probability_calibration_policy.json"
        ).write_text(
            _json.dumps(policy, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        context.notes["calibration_hashes"] = {
            prop_type: sha256_canonical(entry)
            for prop_type, entry in sorted(policy["props"].items())
        }

    def fit_gate3(self, context: "FitContext") -> None:
        """Install the checksum-verified frozen Gate 3 deployment artifacts."""
        import hashlib
        import shutil

        import joblib

        source_dir = (
            self.project_root
            / "research"
            / "v2_gate3_deployment_artifacts"
        )

        required = (
            "SHA256SUMS.txt",
            "deployment_manifest.json",
            "probability_parameters.json",
            "role_minutes_model.joblib",
            "role_state_seed.json",
        )

        missing = [
            name
            for name in required
            if not (source_dir / name).is_file()
        ]

        if missing:
            raise CandidateIncomplete(
                "Frozen Gate 3 deployment bundle is incomplete: "
                f"{missing}"
            )

        checksums: dict[str, str] = {}

        for raw_line in (
            source_dir / "SHA256SUMS.txt"
        ).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                raise CandidateIncomplete(
                    "Malformed frozen Gate 3 SHA256SUMS.txt entry: "
                    f"{raw_line!r}"
                )

            digest, label = parts
            label = label.strip()

            if label.startswith("*"):
                label = label[1:]

            checksums[Path(label).name] = digest.lower()

        verified_hashes: dict[str, str] = {}

        for name in required[1:]:
            expected = checksums.get(name)

            if not expected:
                raise CandidateIncomplete(
                    "Frozen Gate 3 checksum missing for "
                    f"{name}"
                )

            actual = hashlib.sha256(
                (source_dir / name).read_bytes()
            ).hexdigest()

            if actual != expected:
                raise CandidateIncomplete(
                    "Frozen Gate 3 checksum mismatch for "
                    f"{name}: expected {expected}, got {actual}"
                )

            verified_hashes[name] = actual

        manifest = _load_json(
            source_dir / "deployment_manifest.json"
        )

        role_state = _load_json(
            source_dir / "role_state_seed.json"
        )

        role_model_payload = joblib.load(
            source_dir / "role_minutes_model.joblib"
        )

        if not isinstance(role_model_payload, dict):
            raise CandidateIncomplete(
                "Frozen Gate 3 role_minutes_model.joblib "
                "does not contain the expected payload"
            )

        manifest_features = list(
            manifest.get("role_minutes_features") or []
        )

        payload_features = list(
            role_model_payload.get("feature_names") or []
        )

        if not manifest_features:
            raise CandidateIncomplete(
                "Frozen Gate 3 deployment manifest has no "
                "role_minutes_features"
            )

        if payload_features != manifest_features:
            raise CandidateIncomplete(
                "Frozen Gate 3 role-model feature contract does not "
                "match the deployment manifest"
            )

        expected_rows = manifest.get(
            "role_minutes_training_rows"
        )

        payload_rows = role_model_payload.get(
            "training_rows"
        )

        if (
            expected_rows is not None
            and payload_rows is not None
            and int(payload_rows) != int(expected_rows)
        ):
            raise CandidateIncomplete(
                "Frozen Gate 3 role-model training row count does "
                "not match the deployment manifest: "
                f"{payload_rows} != {expected_rows}"
            )

        # These are the two artifacts this stage historically produced.
        # Preserve that output contract, but install the frozen certified
        # versions rather than refitting from prospective/adaptive data.
        shutil.copy2(
            source_dir / "role_minutes_model.joblib",
            context.workspace.models / "role_minutes_model.joblib",
        )

        shutil.copy2(
            source_dir / "role_state_seed.json",
            context.workspace.models / "role_state_seed.json",
        )

        context.notes["role_state_hash"] = sha256_canonical(
            role_state
        )

        context.notes["gate3_role_model_source"] = (
            "frozen_deployment_artifact"
        )

        context.notes["gate3_deployment_artifact"] = (
            manifest.get("artifact")
        )

        context.notes["gate3_deployment_artifact_version"] = (
            manifest.get("artifact_version")
        )

        context.notes["gate3_verified_hashes"] = (
            verified_hashes
        )

    def assemble_candidate(self, context: "FitContext") -> None:
        """Collect the serving artifacts plus the provenance they were fit under."""
        candidate = context.workspace.candidate

        (candidate / "models").mkdir(parents=True, exist_ok=True)
        (candidate / "provenance").mkdir(parents=True, exist_ok=True)

        for item in sorted(context.workspace.models.iterdir()):
            if item.is_file():
                shutil.copyfile(item, candidate / "models" / item.name)

        # The protocol and the frozen policies the fit ran under travel with
        # the candidate, so the fit_id digest covers them.
        for relative in (
            UPDATE_PROTOCOL_RELATIVE_PATH,
            CONTRACT_RELATIVE_PATH,
            SERVING_SOURCE_CONTRACT_RELATIVE_PATH,
            CONFIG_RELATIVE_PATH,
        ):
            source = self.project_root / relative

            shutil.copyfile(
                source, candidate / "provenance" / Path(relative).name
            )

        for name, source in sorted(FROZEN_POLICY_SOURCES.items()):
            shutil.copyfile(
                self.project_root / source["path"],
                candidate / "provenance" / f"{name}.json",
            )
