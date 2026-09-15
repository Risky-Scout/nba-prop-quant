"""Immutable adaptive fit registry, fail-closed promotion and last-good rollback.

This module is infrastructure. It stores the numerical parameters produced by a
daily fit, and it refuses to store anything that changed the model
architecture. It never trains, refits, recalibrates or selects a model, and it
contains no predictive mathematics.

Two categories are held strictly apart.

FROZEN CHOICE
    Mean-model routing, marginal family, calibration family, dependence
    lambda, Gate 3 routing, the feature schema, the hyperparameter config and
    both seed families. These are pinned by the adaptive architecture contract
    and registration is refused when any of them moves.

DAILY FITTED VALUE
    Fitted boosters, ensemble weights, ZINB parameters, copula correlation,
    Platt coefficients, experience-curve coefficients and role-model weights.
    These are the registry payload and are expected to differ every day.

Daily fitting is therefore not daily model selection: a registered fit is
proof that only the second category moved.
"""

from __future__ import annotations

import ast
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import MISSING, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


# --------------------------------------------------------------------------
# identity and location constants
# --------------------------------------------------------------------------

FIT_ID_PREFIX = "nba_prop_quant_fit_"

FIT_ID_DIGEST_CHARS = 16

REGISTRY_ENV_VAR = "NBA_PROP_FIT_REGISTRY_DIR"

ARCHITECTURE_REFERENCE_SHA = (
    "4def8ad33ccc56016fb19a97fceca6e027c9612a"
)

CONTRACT_RELATIVE_PATH = (
    Path("models")
    / "frozen_manifests"
    / "nba_prop_quant_v2_adaptive_architecture_contract.json"
)

CONFIG_RELATIVE_PATH = Path("configs") / "model.yaml"

FEATURES_RELATIVE_PATH = (
    Path("src") / "nba_prop_quant" / "features.py"
)

CONTRACT_VERSION = 1

# configs/model.yaml holds frozen hyperparameters and operational season
# boundaries in the same file. The end of the eligible training history has to
# advance as seasons complete, so hashing the file whole would make a lawful
# season advancement look identical to architecture drift and would refuse
# every fit from the next season onward. The config digest is therefore taken
# over the hyperparameter projection with these operational fields pruned.
CONFIG_OPERATIONAL_FIELDS: tuple[str, ...] = (
    "advanced_start_season",
    "history_start_season",
    "play_by_play_start_season",
    "production_train_end_season",
)

# The two window boundaries that are genuinely frozen. They leave the config
# digest with the other season fields, so the contract pins them by value
# instead and they are checked against the tree at registration.
TRAINING_WINDOW_FROZEN_BOUNDARIES: tuple[str, ...] = (
    "advanced_start_season",
    "history_start_season",
)

TRAINING_WINDOW_POLICY_KEY = "training_window_policy"

TRAINING_END_POLICY = "expanding_through_training_cutoff"

TRAINING_CUTOFF_RELATION = "strictly_before_slate_date"

# A terminal training season may never re-enter the frozen choices: a daily
# adaptive fit has to be able to reach every season that completes after the
# architecture was frozen.
PROHIBITED_TRAINING_END_KEYS: frozenset[str] = frozenset(
    {
        "production_train_end_season",
        "train_end_season",
        "training_end_season",
    }
)

MANIFEST_SCHEMA_VERSION = 1

PROMOTION_STATE_SCHEMA_VERSION = 1

VALIDATION_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# frozen policy sources
# --------------------------------------------------------------------------

# The declared feature schema, read as literals rather than imported so the
# registry stays usable on a host with no modelling stack installed.
FEATURE_SCHEMA_NAMES = (
    "FEATURE_EXACT",
    "ADVANCED_FEATURES",
    "FEATURE_PREFIXES",
    "VOLUME_STATS",
    "TARGETS",
)


# Several policy documents carry a frozen decision and a daily fitted value in
# the same file. Hashing such a file whole would freeze numbers the contract
# explicitly allows to move every day, so each source declares the paths that
# must be pruned before its frozen-policy digest is taken. Dotted paths, with
# "*" matching any single mapping key.
FROZEN_POLICY_SOURCES: dict[str, dict[str, Any]] = {
    "mean_model_selection": {
        "path": Path("models") / "mean_model_selection.json",
        # Ensemble weights are refit daily.
        "exclude": ("targets.*.production_weights",),
    },
    "marginal_selection": {
        "path": Path("models") / "marginal_selection.json",
        "exclude": (),
    },
    "market_probability_calibration_policy": {
        "path": Path("models")
        / "market_probability_calibration_policy.json",
        # Platt intercept and slope are refit daily.
        "exclude": ("props.*.production_parameters",),
    },
    "combo_dependence_policy": {
        "path": Path("models")
        / "combo_dependence_policy.json",
        "exclude": (),
    },
    "gate3_deployment_policy": {
        "path": Path("research")
        / "v2_gate3_deployment_artifacts"
        / "deployment_manifest.json",
        # Everything describing one particular role-model build moves when
        # the role model is refit; only the policy itself is frozen.
        "exclude": (
            "builder_commit",
            "deployment_parameter_fit",
            "gate2_evidence_commit",
            "generated_at_utc",
            "input_sha256",
            "role_minutes_training_games",
            "role_minutes_training_rows",
            "role_state_seed_players",
            "role_state_seed_teams",
        ),
    },
}


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

# Checks an adaptive orchestrator must report before a fit may be promoted.
# The registry records the outcomes; it does not run the model-side checks.
REQUIRED_VALIDATION_CHECKS = (
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
    "gate3_role_readiness",
    "t20_protocol_compatible",
)


# --------------------------------------------------------------------------
# secret refusal
# --------------------------------------------------------------------------

_SECRET_KEY_PATTERN = re.compile(
    r"api[_-]?key|secret|token|password|passwd|credential"
    r"|authorization|bearer|private[_-]?key|ssh",
    re.IGNORECASE,
)

_SECRET_VALUE_PATTERN = re.compile(
    r"BDL_API_KEY|ODDS_API_KEY|BEGIN [A-Z ]*PRIVATE KEY"
    r"|ssh-rsa|Bearer\s+\S",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class RegistryError(RuntimeError):
    """Base class for every fail-closed registry refusal."""


class RegistryRootError(RegistryError):
    """The registry root was unspecified or unsafe."""


class ArchitectureContractViolation(RegistryError):
    """Candidate metadata or the tree conflicts with the frozen contract."""


class FitAlreadyExists(RegistryError):
    """A finalized fit exists and must never be replaced."""


class FitNotFound(RegistryError):
    """The referenced fit is not registered."""


class IntegrityError(RegistryError):
    """Stored bytes disagree with the recorded hashes."""


class StagedTreeError(RegistryError):
    """The staged candidate tree is unusable."""


class ValidationRefused(RegistryError):
    """Validation state is missing, malformed, mismatched or failing."""


class PromotionRefused(RegistryError):
    """Promotion or rollback preconditions were not met."""


class PromotionLocked(RegistryError):
    """Another promotion or rollback holds the exclusive lock."""


# --------------------------------------------------------------------------
# deterministic serialization and hashing
# --------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    """Serialize deterministically: sorted keys, no incidental whitespace."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_canonical(payload: Any) -> str:
    return sha256_bytes(
        canonical_json(payload).encode("utf-8")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(payload: Any, path: Path) -> None:
    """Write readable deterministic JSON through a same-directory replace."""
    path.parent.mkdir(parents=True, exist_ok=True)

    text = (
        json.dumps(payload, sort_keys=True, indent=2)
        + "\n"
    )

    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )

    temp_path = Path(handle.name)

    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# frozen-choice derivation from the working tree
# --------------------------------------------------------------------------


def prune_paths(payload: Any, exclude: tuple[str, ...]) -> Any:
    """Return a deep copy with the dotted paths removed.

    A "*" segment matches any single mapping key, which lets one rule prune a
    per-target or per-prop fitted field across the whole document.
    """
    if not exclude:
        return payload

    def strip(node: Any, trail: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            out = {}

            for key, value in node.items():
                path = trail + (str(key),)

                if any(
                    _path_matches(path, rule)
                    for rule in exclude
                ):
                    continue

                out[key] = strip(value, path)

            return out

        if isinstance(node, list):
            return [strip(item, trail) for item in node]

        return node

    return strip(payload, ())


def _path_matches(
    path: tuple[str, ...], rule: str
) -> bool:
    parts = rule.split(".")

    if len(parts) != len(path):
        return False

    return all(
        part == "*" or part == actual
        for part, actual in zip(parts, path)
    )


def frozen_policy_projection(
    project_root: Path, name: str
) -> Any:
    """Load one policy document with its daily fitted fields pruned away."""
    source = FROZEN_POLICY_SOURCES[name]

    path = project_root / source["path"]

    if not path.exists():
        raise ArchitectureContractViolation(
            f"frozen policy file missing: {path}"
        )

    return prune_paths(
        _load_json(path), source["exclude"]
    )


def frozen_policy_digests(
    project_root: Path,
) -> dict[str, str]:
    return {
        name: sha256_canonical(
            frozen_policy_projection(project_root, name)
        )
        for name in sorted(FROZEN_POLICY_SOURCES)
    }


def frozen_routing(
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """Extract the human-readable routing decisions the contract pins."""
    mean = frozen_policy_projection(
        project_root, "mean_model_selection"
    )

    marginal = frozen_policy_projection(
        project_root, "marginal_selection"
    )

    calibration = frozen_policy_projection(
        project_root,
        "market_probability_calibration_policy",
    )

    dependence = frozen_policy_projection(
        project_root, "combo_dependence_policy"
    )

    gate3 = frozen_policy_projection(
        project_root, "gate3_deployment_policy"
    )

    return {
        "mean_model_routing": {
            target: str(entry["selected_mode"])
            for target, entry in sorted(
                mean["targets"].items()
            )
        },
        "marginal_family_routing": {
            target: str(entry["selected_distribution"])
            for target, entry in sorted(
                marginal["targets"].items()
            )
        },
        "calibration_family_routing": {
            prop: str(entry["selected_method"])
            for prop, entry in sorted(
                calibration["props"].items()
            )
        },
        "dependence_production_lambda": {
            combo: float(entry["production_lambda"])
            for combo, entry in sorted(
                dependence["combos"].items()
            )
        },
        "gate3_routing": {
            prop: str(route)
            for prop, route in sorted(
                gate3["gate3_policy"].items()
            )
        },
    }


def feature_schema_identity(
    project_root: Path,
) -> dict[str, list[str]]:
    """Read the declared feature schema by parsing, not importing."""
    path = project_root / FEATURES_RELATIVE_PATH

    if not path.exists():
        raise ArchitectureContractViolation(
            f"feature module missing: {path}"
        )

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    found: dict[str, list[str]] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1:
            continue

        target = node.targets[0]

        if not isinstance(target, ast.Name):
            continue

        if target.id not in FEATURE_SCHEMA_NAMES:
            continue

        try:
            value = ast.literal_eval(node.value)
        except ValueError as error:
            raise ArchitectureContractViolation(
                f"{target.id} is no longer a literal in "
                f"{FEATURES_RELATIVE_PATH}: {error}"
            ) from error

        found[target.id] = [str(item) for item in value]

    missing = sorted(
        set(FEATURE_SCHEMA_NAMES) - set(found)
    )

    if missing:
        raise ArchitectureContractViolation(
            "feature schema declarations missing from "
            f"{FEATURES_RELATIVE_PATH}: "
            + ", ".join(missing)
        )

    # Declared order is preserved because column order is part of identity.
    return {
        name.lower(): found[name]
        for name in FEATURE_SCHEMA_NAMES
    }


def feature_schema_hash(project_root: Path) -> str:
    return sha256_canonical(
        feature_schema_identity(project_root)
    )


def load_model_config(
    project_root: Path,
) -> dict[str, Any]:
    path = project_root / CONFIG_RELATIVE_PATH

    if not path.exists():
        raise ArchitectureContractViolation(
            f"model config missing: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ArchitectureContractViolation(
            f"model config is not a mapping: {path}"
        )

    return payload


def config_projection(
    project_root: Path,
) -> dict[str, Any]:
    """Return the model config with the operational season fields pruned.

    Targets, distribution policy, dynamic priors, the booster
    hyperparameters and the seeds all stay in the projection, so a
    hyperparameter or target-set change still cannot register. Only the
    season boundaries leave, because the training end expands as seasons
    complete.
    """
    return prune_paths(
        load_model_config(project_root),
        CONFIG_OPERATIONAL_FIELDS,
    )


def config_hash(project_root: Path) -> str:
    return sha256_canonical(
        config_projection(project_root)
    )


def training_window_boundaries(
    project_root: Path,
) -> dict[str, int]:
    """Read the frozen window boundaries the config digest no longer covers."""
    config = load_model_config(project_root)

    boundaries: dict[str, int] = {}

    for name in TRAINING_WINDOW_FROZEN_BOUNDARIES:
        if name not in config:
            raise ArchitectureContractViolation(
                f"model config is missing {name!r}"
            )

        boundaries[name] = int(config[name])

    return boundaries


def gate3_candidate_policy_id(project_root: Path) -> str:
    """Mirror the identifier gate3_v2.load_gate3_runtime derives."""
    manifest = frozen_policy_projection(
        project_root, "gate3_deployment_policy"
    )

    lock_commit = str(manifest["gate3_lock_commit"])

    return "nba_prop_quant_v2_gate3_" + lock_commit[:12]


def environment_fingerprint(
    python_version: str,
    package_versions: dict[str, str],
) -> str:
    return sha256_canonical(
        {
            "packages": {
                str(name): str(version)
                for name, version in package_versions.items()
            },
            "python_version": str(python_version),
        }
    )


# --------------------------------------------------------------------------
# architecture contract
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchitectureContract:
    path: Path
    payload: dict[str, Any]
    sha256: str

    @property
    def architecture_reference_sha(self) -> str:
        return str(
            self.payload["architecture_reference_sha"]
        )

    @property
    def frozen_choices(self) -> dict[str, Any]:
        return self.payload["frozen_choices"]

    @property
    def policy_digests(self) -> dict[str, str]:
        return self.payload["frozen_policy_digests"]


def load_architecture_contract(
    project_root: Path,
    contract_path: Path | None = None,
) -> ArchitectureContract:
    path = (
        contract_path
        if contract_path is not None
        else project_root / CONTRACT_RELATIVE_PATH
    )

    if not path.exists():
        raise ArchitectureContractViolation(
            f"adaptive architecture contract missing: {path}"
        )

    payload = _load_json(path)

    observed_reference = payload.get(
        "architecture_reference_sha"
    )

    if observed_reference != ARCHITECTURE_REFERENCE_SHA:
        raise ArchitectureContractViolation(
            "contract architecture_reference_sha is "
            f"{observed_reference!r}, expected "
            f"{ARCHITECTURE_REFERENCE_SHA!r}"
        )

    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ArchitectureContractViolation(
            "unsupported contract_version "
            f"{payload.get('contract_version')!r}"
        )

    for required in (
        "frozen_choices",
        "frozen_policy_digests",
    ):
        if required not in payload:
            raise ArchitectureContractViolation(
                f"contract is missing {required!r}"
            )

    _assert_training_window_policy(payload["frozen_choices"])

    return ArchitectureContract(
        path=path,
        payload=payload,
        sha256=sha256_file(path),
    )


def _find_prohibited_training_end(
    node: Any, trail: str
) -> str | None:
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{trail}.{key}"

            if str(key) in PROHIBITED_TRAINING_END_KEYS:
                return path

            found = _find_prohibited_training_end(
                value, path
            )

            if found is not None:
                return found

    return None


def _assert_training_window_policy(
    frozen: dict[str, Any],
) -> None:
    """Refuse a contract that pins a terminal production training season.

    The architecture is frozen; the end of the eligible training history is
    not. A contract naming a final season would stop every season completed
    after the freeze from ever entering a daily adaptive fit.
    """
    prohibited = _find_prohibited_training_end(
        frozen, "frozen_choices"
    )

    if prohibited is not None:
        raise ArchitectureContractViolation(
            f"{prohibited} pins a terminal production "
            "training season; the training end expands "
            "through the daily training_cutoff and is not a "
            "frozen architecture choice"
        )

    policy = frozen.get(TRAINING_WINDOW_POLICY_KEY)

    if not isinstance(policy, dict):
        raise ArchitectureContractViolation(
            "contract is missing frozen_choices."
            f"{TRAINING_WINDOW_POLICY_KEY}"
        )

    for key, expected in (
        ("end_policy", TRAINING_END_POLICY),
        ("cutoff_relation", TRAINING_CUTOFF_RELATION),
    ):
        observed = policy.get(key)

        if observed != expected:
            raise ArchitectureContractViolation(
                f"{TRAINING_WINDOW_POLICY_KEY}.{key} is "
                f"{observed!r}, expected {expected!r}"
            )

    for name in TRAINING_WINDOW_FROZEN_BOUNDARIES:
        if name not in policy:
            raise ArchitectureContractViolation(
                f"{TRAINING_WINDOW_POLICY_KEY} is missing "
                f"{name!r}"
            )


def verify_contract_against_tree(
    contract: ArchitectureContract,
    project_root: Path,
) -> None:
    """Refuse when the working tree has drifted from the frozen choices.

    This is what stops a nightly fitting job from quietly becoming model
    selection, feature search, hyperparameter search or architecture
    research.
    """
    frozen = contract.frozen_choices

    observed_features = feature_schema_hash(project_root)

    if observed_features != str(
        frozen["feature_schema_hash"]
    ):
        raise ArchitectureContractViolation(
            "feature schema hash drifted: tree "
            f"{observed_features}, contract "
            f"{frozen['feature_schema_hash']}"
        )

    observed_config = config_hash(project_root)

    if observed_config != str(frozen["config_hash"]):
        raise ArchitectureContractViolation(
            "hyperparameter config hash drifted: tree "
            f"{observed_config}, contract "
            f"{frozen['config_hash']}"
        )

    policy = frozen[TRAINING_WINDOW_POLICY_KEY]

    observed_window = training_window_boundaries(
        project_root
    )

    for name in TRAINING_WINDOW_FROZEN_BOUNDARIES:
        expected_boundary = int(policy[name])

        if observed_window[name] != expected_boundary:
            raise ArchitectureContractViolation(
                f"frozen training window {name} drifted: "
                f"tree {observed_window[name]}, contract "
                f"{expected_boundary}"
            )

    observed_digests = frozen_policy_digests(project_root)

    for name, expected in sorted(
        contract.policy_digests.items()
    ):
        observed = observed_digests.get(name)

        if observed != expected:
            raise ArchitectureContractViolation(
                f"frozen policy {name} drifted: tree "
                f"{observed}, contract {expected}"
            )

    observed_routing = frozen_routing(project_root)

    for name, expected_map in sorted(
        frozen["routing"].items()
    ):
        if observed_routing.get(name) != expected_map:
            raise ArchitectureContractViolation(
                f"frozen routing {name} drifted from contract"
            )

    observed_gate3 = gate3_candidate_policy_id(project_root)

    if observed_gate3 != str(
        frozen["gate3_candidate_policy_id"]
    ):
        raise ArchitectureContractViolation(
            "Gate 3 candidate policy id drifted: tree "
            f"{observed_gate3}, contract "
            f"{frozen['gate3_candidate_policy_id']}"
        )


# --------------------------------------------------------------------------
# candidate metadata
# --------------------------------------------------------------------------


@dataclass
class FitMetadata:
    """Caller-supplied provenance for one immutable daily fit."""

    fit_date: str
    training_cutoff: str
    source_commit_sha: str
    training_data_manifest_hash: str
    python_version: str
    package_versions: dict[str, str]
    core_seed: int
    gate3_seed: int
    effective_n_jobs: int
    architecture_reference_sha: str = (
        ARCHITECTURE_REFERENCE_SHA
    )
    gate3_candidate_policy_id: str | None = None
    calibration_hashes: dict[str, str] = field(
        default_factory=dict
    )
    role_state_hash: str | None = None
    training_started_at: str | None = None
    training_completed_at: str | None = None
    validation_lineage_reference: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "FitMetadata":
        if not isinstance(payload, dict):
            raise ArchitectureContractViolation(
                "fit metadata must be a JSON object"
            )

        fields = cls.__dataclass_fields__

        unknown = sorted(set(payload) - set(fields))

        if unknown:
            raise ArchitectureContractViolation(
                "unknown metadata fields: "
                + ", ".join(unknown)
            )

        missing = sorted(
            name
            for name, spec in fields.items()
            if spec.default is MISSING
            and spec.default_factory is MISSING
            and name not in payload
        )

        if missing:
            raise ArchitectureContractViolation(
                "missing metadata fields: "
                + ", ".join(missing)
            )

        return cls(**payload)


def _assert_iso_date(value: Any, label: str) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureContractViolation(
            f"{label} must be an ISO date, got {value!r}"
        ) from error


def assert_no_secrets(
    payload: Any, trail: str = ""
) -> None:
    """Refuse credential-shaped keys and values, and absolute paths.

    Absolute paths are refused as well: a manifest that embeds a training-host
    or WizardOfOdds path is not portable, which defeats the deployment seam.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _SECRET_KEY_PATTERN.search(str(key)):
                raise ArchitectureContractViolation(
                    "refusing to store credential-shaped field "
                    f"{trail}{key!r}"
                )

            assert_no_secrets(value, f"{trail}{key}.")

        return

    if isinstance(payload, (list, tuple)):
        for item in payload:
            assert_no_secrets(item, trail)

        return

    if isinstance(payload, str):
        if _SECRET_VALUE_PATTERN.search(payload):
            raise ArchitectureContractViolation(
                "refusing to store credential-shaped value at "
                f"{trail or '<root>'}"
            )

        if payload.startswith("/") and len(payload) > 1:
            raise ArchitectureContractViolation(
                "refusing to store an absolute path at "
                f"{trail or '<root>'}; manifests must stay "
                "portable"
            )


# --------------------------------------------------------------------------
# staged artifact scanning
# --------------------------------------------------------------------------


def scan_artifact_tree(root: Path) -> dict[str, str]:
    """Hash every regular file, refusing anything that is not one."""
    if not root.is_dir():
        raise StagedTreeError(
            f"artifact tree is not a directory: {root}"
        )

    hashes: dict[str, str] = {}

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()

        if path.is_symlink():
            raise StagedTreeError(
                f"symlink rejected in artifact tree: {relative}"
            )

        mode = os.lstat(path).st_mode

        if stat.S_ISDIR(mode):
            continue

        if not stat.S_ISREG(mode):
            raise StagedTreeError(
                "special file rejected in artifact tree: "
                f"{relative}"
            )

        hashes[relative] = sha256_file(path)

    if not hashes:
        raise StagedTreeError(
            f"artifact tree is empty: {root}"
        )

    return hashes


def render_checksums(hashes: dict[str, str]) -> str:
    """Render a sha256sum-compatible inventory, as used elsewhere here."""
    lines = [
        f"{hashes[relative]}  {relative}"
        for relative in sorted(hashes)
    ]

    return "\n".join(lines) + "\n"


def parse_checksums(text: str) -> dict[str, str]:
    out: dict[str, str] = {}

    for line in text.splitlines():
        if not line.strip():
            continue

        digest, separator, relative = line.partition("  ")

        if not separator or not relative:
            raise IntegrityError(
                f"malformed checksum line: {line!r}"
            )

        out[relative] = digest

    return out


# --------------------------------------------------------------------------
# fit identity
# --------------------------------------------------------------------------


def build_identity(
    metadata: FitMetadata,
    contract: ArchitectureContract,
    project_root: Path,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    """Assemble the immutable inputs the fit_id digest is taken over.

    Promotion state is deliberately absent: identity has to be settled before
    anything decides whether the fit is good. Wall-clock fields are recorded
    in the manifest but kept out of the digest so identity stays deterministic.
    """
    return {
        "architecture_contract_sha256": contract.sha256,
        "architecture_reference_sha": (
            metadata.architecture_reference_sha
        ),
        "artifact_hashes": dict(
            sorted(artifact_hashes.items())
        ),
        "calibration_hashes": dict(
            sorted(metadata.calibration_hashes.items())
        ),
        "config_hash": config_hash(project_root),
        "core_seed": int(metadata.core_seed),
        "dependency_environment_fingerprint": (
            environment_fingerprint(
                metadata.python_version,
                metadata.package_versions,
            )
        ),
        "effective_n_jobs": int(metadata.effective_n_jobs),
        "feature_schema_hash": feature_schema_hash(
            project_root
        ),
        "fit_date": metadata.fit_date,
        "gate3_candidate_policy_id": (
            metadata.gate3_candidate_policy_id
        ),
        "gate3_seed": int(metadata.gate3_seed),
        "role_state_hash": metadata.role_state_hash,
        "source_commit_sha": metadata.source_commit_sha,
        "training_cutoff": metadata.training_cutoff,
        "training_data_manifest_hash": (
            metadata.training_data_manifest_hash
        ),
    }


def derive_fit_id(identity: dict[str, Any]) -> str:
    digest = sha256_canonical(identity)

    compact_date = str(identity["fit_date"]).replace(
        "-", ""
    )

    return (
        f"{FIT_ID_PREFIX}{compact_date}_"
        f"{digest[:FIT_ID_DIGEST_CHARS]}"
    )


_FIT_ID_RE = re.compile(
    re.escape(FIT_ID_PREFIX)
    + r"\d{8}_[0-9a-f]{"
    + str(FIT_ID_DIGEST_CHARS)
    + r"}$"
)


def is_fit_id(value: str) -> bool:
    return bool(_FIT_ID_RE.fullmatch(str(value)))


# --------------------------------------------------------------------------
# registry root
# --------------------------------------------------------------------------


def resolve_registry_root(
    explicit: Path | str | None,
    project_root: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Path:
    """Resolve the registry root, failing closed when it is unspecified."""
    env = os.environ if environ is None else environ

    candidate = (
        explicit
        if explicit is not None
        else env.get(REGISTRY_ENV_VAR)
    )

    if candidate is None or str(candidate).strip() == "":
        raise RegistryRootError(
            "no registry root supplied; pass --registry-root or set "
            f"{REGISTRY_ENV_VAR}. A production registry is never "
            "placed inside the repository by default."
        )

    root = Path(candidate).expanduser()

    root = (
        root.resolve()
        if root.is_absolute()
        else (Path.cwd() / root).resolve()
    )

    if project_root is not None:
        resolved = project_root.resolve()

        if root == resolved or resolved in root.parents:
            raise RegistryRootError(
                f"registry root {root} is inside the repository at "
                f"{resolved}; fits are production state and must "
                "live outside version control"
            )

    return root


@contextmanager
def exclusive_lock(
    path: Path, blocking: bool = True
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = os.open(
        path, os.O_CREAT | os.O_RDWR, 0o644
    )

    flags = fcntl.LOCK_EX

    if not blocking:
        flags |= fcntl.LOCK_NB

    try:
        try:
            fcntl.flock(handle, flags)
        except OSError as error:
            raise PromotionLocked(
                f"another promotion or rollback holds {path}"
            ) from error

        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


class FitRegistry:
    """Filesystem registry of immutable daily fits and promotion state."""

    def __init__(
        self, root: Path, project_root: Path
    ) -> None:
        self.root = Path(root)
        self.project_root = Path(project_root)

    # -- layout ---------------------------------------------------------

    @property
    def fits_dir(self) -> Path:
        return self.root / "fits"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def validations_dir(self) -> Path:
        return self.state_dir / "validations"

    @property
    def locks_dir(self) -> Path:
        return self.root / "locks"

    @property
    def staging_dir(self) -> Path:
        return self.root / "staging"

    @property
    def promotion_state_path(self) -> Path:
        return self.state_dir / "promotion_state.json"

    @property
    def promotion_lock_path(self) -> Path:
        return self.locks_dir / "promotion.lock"

    def fit_dir(self, fit_id: str) -> Path:
        if not is_fit_id(fit_id):
            raise FitNotFound(
                f"not a well-formed fit_id: {fit_id!r}"
            )

        return self.fits_dir / fit_id

    def validation_path(self, fit_id: str) -> Path:
        if not is_fit_id(fit_id):
            raise FitNotFound(
                f"not a well-formed fit_id: {fit_id!r}"
            )

        return self.validations_dir / f"{fit_id}.json"

    def initialise(self) -> None:
        for path in (
            self.fits_dir,
            self.state_dir,
            self.validations_dir,
            self.locks_dir,
            self.staging_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def list_fits(self) -> list[str]:
        if not self.fits_dir.is_dir():
            return []

        return sorted(
            path.name
            for path in self.fits_dir.iterdir()
            if path.is_dir() and is_fit_id(path.name)
        )

    # -- registration ---------------------------------------------------

    def register(
        self,
        staged_dir: Path,
        metadata: FitMetadata,
        contract: ArchitectureContract | None = None,
    ) -> str:
        """Register a staged candidate tree as an immutable fit.

        Registration never promotes. The fit begins as
        REGISTERED / NOT PROMOTED.
        """
        self.initialise()

        if contract is None:
            contract = load_architecture_contract(
                self.project_root
            )

        self._assert_architecture_lock(metadata, contract)

        artifact_hashes = scan_artifact_tree(staged_dir)

        identity = build_identity(
            metadata,
            contract,
            self.project_root,
            artifact_hashes,
        )

        fit_id = derive_fit_id(identity)

        manifest = self._build_manifest(
            fit_id, metadata, contract, identity
        )

        assert_no_secrets(manifest)

        final_dir = self.fit_dir(fit_id)

        if final_dir.exists():
            raise FitAlreadyExists(
                f"fit {fit_id} is already registered at "
                f"{final_dir}; a finalized fit is never "
                "replaced, merged into or modified"
            )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".staging_{fit_id}_",
                dir=self.staging_dir,
            )
        )

        try:
            self._materialise(
                staging,
                staged_dir,
                artifact_hashes,
                manifest,
            )

            try:
                # A non-empty destination makes rename fail, so this is
                # atomic even against a concurrent registration that won
                # the race between the check above and here.
                os.rename(staging, final_dir)
            except OSError as error:
                if error.errno in (
                    errno.ENOTEMPTY,
                    errno.EEXIST,
                    errno.ENOTDIR,
                ):
                    raise FitAlreadyExists(
                        f"fit {fit_id} was registered "
                        "concurrently; the existing fit is "
                        "left untouched"
                    ) from error

                raise
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return fit_id

    def _assert_architecture_lock(
        self,
        metadata: FitMetadata,
        contract: ArchitectureContract,
    ) -> None:
        _assert_iso_date(metadata.fit_date, "fit_date")

        _assert_iso_date(
            metadata.training_cutoff, "training_cutoff"
        )

        if metadata.training_cutoff >= metadata.fit_date:
            raise ArchitectureContractViolation(
                f"training_cutoff {metadata.training_cutoff} "
                "must be strictly before fit_date "
                f"{metadata.fit_date}"
            )

        if (
            metadata.architecture_reference_sha
            != contract.architecture_reference_sha
        ):
            raise ArchitectureContractViolation(
                "architecture_reference_sha mismatch: metadata "
                f"{metadata.architecture_reference_sha}, "
                "contract "
                f"{contract.architecture_reference_sha}"
            )

        verify_contract_against_tree(
            contract, self.project_root
        )

        frozen = contract.frozen_choices

        for key, observed in (
            ("core_seed", int(metadata.core_seed)),
            ("gate3_seed", int(metadata.gate3_seed)),
            (
                "effective_n_jobs",
                int(metadata.effective_n_jobs),
            ),
        ):
            expected = int(frozen[key])

            if observed != expected:
                raise ArchitectureContractViolation(
                    f"{key} mismatch: metadata {observed}, "
                    f"contract {expected}"
                )

        if metadata.gate3_candidate_policy_id is not None:
            expected_gate3 = str(
                frozen["gate3_candidate_policy_id"]
            )

            if (
                metadata.gate3_candidate_policy_id
                != expected_gate3
            ):
                raise ArchitectureContractViolation(
                    "gate3_candidate_policy_id mismatch: "
                    "metadata "
                    f"{metadata.gate3_candidate_policy_id}, "
                    f"contract {expected_gate3}"
                )

    def _build_manifest(
        self,
        fit_id: str,
        metadata: FitMetadata,
        contract: ArchitectureContract,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "architecture_contract_sha256": contract.sha256,
            "architecture_reference_sha": identity[
                "architecture_reference_sha"
            ],
            "artifact_hashes": identity["artifact_hashes"],
            "artifact_root": "artifacts",
            "calibration_hashes": identity[
                "calibration_hashes"
            ],
            "config_hash": identity["config_hash"],
            "core_seed": identity["core_seed"],
            "created_at": _utc_now(),
            "dependency_environment_fingerprint": identity[
                "dependency_environment_fingerprint"
            ],
            "effective_n_jobs": identity[
                "effective_n_jobs"
            ],
            "feature_schema_hash": identity[
                "feature_schema_hash"
            ],
            "fit_date": identity["fit_date"],
            "fit_id": fit_id,
            "gate3_candidate_policy_id": identity[
                "gate3_candidate_policy_id"
            ],
            "gate3_seed": identity["gate3_seed"],
            "identity_inputs": identity,
            "manifest_schema_version": (
                MANIFEST_SCHEMA_VERSION
            ),
            "notes": metadata.notes,
            "package_versions": dict(
                sorted(metadata.package_versions.items())
            ),
            "python_version": metadata.python_version,
            "role_state_hash": identity["role_state_hash"],
            "source_commit_sha": identity[
                "source_commit_sha"
            ],
            "training_completed_at": (
                metadata.training_completed_at
            ),
            "training_cutoff": identity["training_cutoff"],
            "training_data_manifest_hash": identity[
                "training_data_manifest_hash"
            ],
            "training_started_at": (
                metadata.training_started_at
            ),
            "validation_lineage_reference": (
                metadata.validation_lineage_reference
            ),
        }

    def _materialise(
        self,
        staging: Path,
        staged_dir: Path,
        artifact_hashes: dict[str, str],
        manifest: dict[str, Any],
    ) -> None:
        artifacts = staging / "artifacts"

        for relative in sorted(artifact_hashes):
            destination = artifacts / relative

            destination.parent.mkdir(
                parents=True, exist_ok=True
            )

            shutil.copyfile(
                staged_dir / relative, destination
            )

        # Verify the copied bytes before anything is sealed.
        for relative, expected in sorted(
            artifact_hashes.items()
        ):
            observed = sha256_file(artifacts / relative)

            if observed != expected:
                raise IntegrityError(
                    "staged copy hash mismatch for "
                    f"{relative}: {observed} != {expected}"
                )

        manifest_path = staging / "manifest.json"

        write_json_atomic(manifest, manifest_path)

        inventory = {
            f"artifacts/{relative}": digest
            for relative, digest in artifact_hashes.items()
        }

        inventory["manifest.json"] = sha256_file(
            manifest_path
        )

        (staging / "SHA256SUMS").write_text(
            render_checksums(inventory), encoding="utf-8"
        )

    # -- verification ---------------------------------------------------

    def load_manifest(self, fit_id: str) -> dict[str, Any]:
        path = self.fit_dir(fit_id) / "manifest.json"

        if not path.exists():
            raise FitNotFound(
                f"fit {fit_id} has no manifest at {path}"
            )

        return _load_json(path)

    def verify(
        self,
        fit_id: str,
        contract: ArchitectureContract | None = None,
    ) -> dict[str, Any]:
        """Re-derive identity and re-hash every stored byte.

        fit_id is a digest over the manifest's own identity inputs, so an
        edited manifest derives a different fit_id than the directory holding
        it, and an edited artifact fails its recorded hash. Tampering with
        either is detected without a separate signature.
        """
        fit_dir = self.fit_dir(fit_id)

        if not fit_dir.is_dir():
            raise FitNotFound(
                f"fit {fit_id} is not registered at {fit_dir}"
            )

        manifest = self.load_manifest(fit_id)

        if manifest.get("fit_id") != fit_id:
            raise IntegrityError(
                f"manifest fit_id {manifest.get('fit_id')!r} "
                f"does not match directory {fit_id!r}"
            )

        identity = manifest.get("identity_inputs")

        if not isinstance(identity, dict):
            raise IntegrityError(
                f"fit {fit_id} manifest has no identity_inputs"
            )

        recomputed = derive_fit_id(identity)

        if recomputed != fit_id:
            raise IntegrityError(
                "manifest tampering detected: identity inputs "
                f"derive {recomputed}, directory is {fit_id}"
            )

        expected = {
            str(key): str(value)
            for key, value in identity[
                "artifact_hashes"
            ].items()
        }

        observed = scan_artifact_tree(fit_dir / "artifacts")

        missing = sorted(set(expected) - set(observed))

        if missing:
            raise IntegrityError(
                f"fit {fit_id} is missing artifacts: "
                + ", ".join(missing)
            )

        extra = sorted(set(observed) - set(expected))

        if extra:
            raise IntegrityError(
                f"fit {fit_id} has unrecorded artifacts: "
                + ", ".join(extra)
            )

        for relative in sorted(expected):
            if observed[relative] != expected[relative]:
                raise IntegrityError(
                    f"artifact tampering detected in {fit_id}: "
                    f"{relative} hashes {observed[relative]}, "
                    f"manifest records {expected[relative]}"
                )

        self._verify_checksums(fit_id, fit_dir, expected)

        if contract is None:
            contract = load_architecture_contract(
                self.project_root
            )

        recorded_contract = manifest.get(
            "architecture_contract_sha256"
        )

        if recorded_contract != contract.sha256:
            raise ArchitectureContractViolation(
                f"fit {fit_id} was registered against contract "
                f"{recorded_contract}, current contract is "
                f"{contract.sha256}"
            )

        return manifest

    def _verify_checksums(
        self,
        fit_id: str,
        fit_dir: Path,
        expected_artifacts: dict[str, str],
    ) -> None:
        path = fit_dir / "SHA256SUMS"

        if not path.exists():
            raise IntegrityError(
                f"fit {fit_id} has no SHA256SUMS inventory"
            )

        inventory = parse_checksums(
            path.read_text(encoding="utf-8")
        )

        recorded_manifest = inventory.get("manifest.json")

        if recorded_manifest is None:
            raise IntegrityError(
                f"fit {fit_id} inventory omits manifest.json"
            )

        observed_manifest = sha256_file(
            fit_dir / "manifest.json"
        )

        if observed_manifest != recorded_manifest:
            raise IntegrityError(
                f"manifest tampering detected in {fit_id}: "
                f"inventory records {recorded_manifest}, file "
                f"hashes {observed_manifest}"
            )

        for relative, digest in sorted(
            expected_artifacts.items()
        ):
            if (
                inventory.get(f"artifacts/{relative}")
                != digest
            ):
                raise IntegrityError(
                    f"fit {fit_id} inventory disagrees with "
                    f"the manifest for {relative}"
                )

    # -- validation -----------------------------------------------------

    def record_validation(
        self,
        fit_id: str,
        checks: dict[str, bool],
        actor: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Record validation results outside the immutable fit directory."""
        if not self.fit_dir(fit_id).is_dir():
            raise FitNotFound(
                "cannot record validation for unregistered fit "
                f"{fit_id}"
            )

        if not isinstance(checks, dict):
            raise ValidationRefused(
                "validation checks must be a JSON object"
            )

        unknown = sorted(
            set(checks) - set(REQUIRED_VALIDATION_CHECKS)
        )

        if unknown:
            raise ValidationRefused(
                "unknown validation checks: "
                + ", ".join(unknown)
            )

        for name, value in sorted(checks.items()):
            if not isinstance(value, bool):
                raise ValidationRefused(
                    f"validation check {name} must be a "
                    f"boolean, got {type(value).__name__}"
                )

        normalised = dict(sorted(checks.items()))

        missing = sorted(
            set(REQUIRED_VALIDATION_CHECKS)
            - set(normalised)
        )

        failed = sorted(
            name
            for name, value in normalised.items()
            if not value
        )

        record = {
            "actor": actor,
            "checks": normalised,
            "failed_checks": failed,
            "fit_id": fit_id,
            "missing_checks": missing,
            "notes": notes,
            "passed": not missing and not failed,
            "recorded_at": _utc_now(),
            "required_checks": list(
                REQUIRED_VALIDATION_CHECKS
            ),
            "schema_version": VALIDATION_SCHEMA_VERSION,
        }

        assert_no_secrets(record)

        self.validations_dir.mkdir(
            parents=True, exist_ok=True
        )

        write_json_atomic(
            record, self.validation_path(fit_id)
        )

        return record

    def load_validation(
        self, fit_id: str
    ) -> dict[str, Any] | None:
        path = self.validation_path(fit_id)

        if not path.exists():
            return None

        return _load_json(path)

    def assert_validation_pass(self, fit_id: str) -> None:
        """Refuse a missing, malformed, mismatched or failing record."""
        record = self.load_validation(fit_id)

        if record is None:
            raise ValidationRefused(
                f"fit {fit_id} has no validation record; a "
                "registered fit is never production by default"
            )

        if not isinstance(record, dict):
            raise ValidationRefused(
                f"fit {fit_id} has a malformed validation record"
            )

        recorded_fit = record.get("fit_id")

        if recorded_fit != fit_id:
            raise ValidationRefused(
                "validation record belongs to "
                f"{recorded_fit!r}, not {fit_id!r}"
            )

        if (
            record.get("schema_version")
            != VALIDATION_SCHEMA_VERSION
        ):
            raise ValidationRefused(
                f"fit {fit_id} validation record has "
                "unsupported schema_version "
                f"{record.get('schema_version')!r}"
            )

        checks = record.get("checks")

        if not isinstance(checks, dict):
            raise ValidationRefused(
                f"fit {fit_id} validation record has no checks"
            )

        missing = sorted(
            name
            for name in REQUIRED_VALIDATION_CHECKS
            if name not in checks
        )

        if missing:
            raise ValidationRefused(
                f"fit {fit_id} is missing required validation "
                "checks: " + ", ".join(missing)
            )

        non_boolean = sorted(
            name
            for name in REQUIRED_VALIDATION_CHECKS
            if not isinstance(checks[name], bool)
        )

        if non_boolean:
            raise ValidationRefused(
                f"fit {fit_id} has non-boolean validation "
                "checks: " + ", ".join(non_boolean)
            )

        failed = sorted(
            name
            for name in REQUIRED_VALIDATION_CHECKS
            if not checks[name]
        )

        if failed:
            raise ValidationRefused(
                f"fit {fit_id} failed required validation "
                "checks: " + ", ".join(failed)
            )

    # -- promotion state ------------------------------------------------

    def empty_promotion_state(self) -> dict[str, Any]:
        return {
            "current_good_fit_id": None,
            "history": [],
            "previous_good_fit_id": None,
            "promoted_at": None,
            "promotion_actor": None,
            "promotion_reason": None,
            "schema_version": (
                PROMOTION_STATE_SCHEMA_VERSION
            ),
        }

    def load_promotion_state(self) -> dict[str, Any]:
        if not self.promotion_state_path.exists():
            return self.empty_promotion_state()

        return _load_json(self.promotion_state_path)

    def _write_promotion_state(
        self, state: dict[str, Any]
    ) -> None:
        assert_no_secrets(state)

        self.state_dir.mkdir(parents=True, exist_ok=True)

        write_json_atomic(
            state, self.promotion_state_path
        )

    def _transition(
        self,
        action: str,
        target_fit_id: str,
        reason: str,
        actor: str | None,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        current = state.get("current_good_fit_id")

        moment = _utc_now()

        history = list(state.get("history", []))

        history.append(
            {
                "action": action,
                "actor": actor,
                "at": moment,
                "fit_id": target_fit_id,
                "previous_fit_id": current,
                "reason": reason,
            }
        )

        return {
            "current_good_fit_id": target_fit_id,
            "history": history,
            "previous_good_fit_id": current,
            "promoted_at": moment,
            "promotion_actor": actor,
            "promotion_reason": reason,
            "schema_version": (
                PROMOTION_STATE_SCHEMA_VERSION
            ),
        }

    def promote(
        self,
        fit_id: str,
        reason: str,
        actor: str | None = None,
        blocking: bool = True,
    ) -> dict[str, Any]:
        """Promote a validated fit, leaving current-good intact on any failure.

        Every check runs before the single atomic state replacement, so a
        refusal at any point leaves the previously promoted fit serving.
        """
        with exclusive_lock(
            self.promotion_lock_path, blocking=blocking
        ):
            contract = load_architecture_contract(
                self.project_root
            )

            self.verify(fit_id, contract=contract)

            verify_contract_against_tree(
                contract, self.project_root
            )

            self.assert_validation_pass(fit_id)

            state = self.load_promotion_state()

            if state.get("current_good_fit_id") == fit_id:
                raise PromotionRefused(
                    f"fit {fit_id} is already the current good "
                    "fit"
                )

            new_state = self._transition(
                "promote", fit_id, reason, actor, state
            )

            self._write_promotion_state(new_state)

            return new_state

    def rollback(
        self,
        reason: str,
        actor: str | None = None,
        blocking: bool = True,
    ) -> dict[str, Any]:
        """Restore the previous good fit after re-verifying it end to end."""
        with exclusive_lock(
            self.promotion_lock_path, blocking=blocking
        ):
            state = self.load_promotion_state()

            target = state.get("previous_good_fit_id")

            if target is None:
                raise PromotionRefused(
                    "no previous good fit is recorded; there "
                    "is nothing to roll back to"
                )

            contract = load_architecture_contract(
                self.project_root
            )

            # Fail closed: a missing or corrupt rollback target must never
            # displace the fit that is currently serving.
            self.verify(target, contract=contract)

            self.assert_validation_pass(target)

            new_state = self._transition(
                "rollback", target, reason, actor, state
            )

            self._write_promotion_state(new_state)

            return new_state

    def current(self) -> dict[str, Any]:
        state = self.load_promotion_state()

        return {
            key: state.get(key)
            for key in (
                "current_good_fit_id",
                "previous_good_fit_id",
                "promoted_at",
                "promotion_actor",
                "promotion_reason",
            )
        }
