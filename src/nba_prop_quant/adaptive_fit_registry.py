"""Immutable adaptive fit registry, fail-closed promotion and last-good rollback.

This module stores daily *fitted parameters* produced under a frozen model
architecture. It does not train, refit, recalibrate or select models, and it
never changes predictive mathematics.

Two ideas are kept strictly apart:

FROZEN CHOICE
    Which mean mode each target uses, which marginal family, which calibration
    family, which dependence lambda, which Gate 3 route, the feature schema and
    the hyperparameter config. These live in the adaptive architecture contract
    and a registration is refused when they move.

DAILY FITTED VALUE
    Boosters, ensemble weights, ZINB parameters, Platt coefficients, copula
    correlation and role-model weights. These are the registry payload and are
    expected to differ from day to day.

A registration therefore proves that only the second category changed.
"""

from __future__ import annotations

import ast
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


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

FIT_ID_PREFIX = "nba_prop_quant_fit_"

REGISTRY_ENV_VAR = "NBA_PROP_FIT_REGISTRY_DIR"

ARCHITECTURE_REFERENCE_SHA = (
    "4def8ad33ccc56016fb19a97fceca6e027c9612a"
)

CONTRACT_RELATIVE_PATH = Path(
    "models"
) / "frozen_manifests" / (
    "nba_prop_quant_v2_adaptive_architecture_contract.json"
)

CONFIG_RELATIVE_PATH = Path("configs") / "model.yaml"

FEATURES_RELATIVE_PATH = (
    Path("src") / "nba_prop_quant" / "features.py"
)

CONTRACT_VERSION = 1

MANIFEST_SCHEMA_VERSION = 1

PROMOTION_STATE_SCHEMA_VERSION = 1

FIT_ID_DIGEST_CHARS = 16


# Canonical feature-schema declarations parsed out of features.py. These are
# the static schema identity; feature_columns() composes the runtime order
# from them.
FEATURE_SCHEMA_NAMES = (
    "FEATURE_EXACT",
    "ADVANCED_FEATURES",
    "FEATURE_PREFIXES",
    "VOLUME_STATS",
    "TARGETS",
)


# Frozen-choice groups the contract pins and registration re-derives from the
# working tree. Each entry maps a contract key to the policy file it is read
# from.
ROUTING_SOURCES = {
    "mean_model_routing": Path("models")
    / "mean_model_selection.json",
    "marginal_family_routing": Path("models")
    / "marginal_selection.json",
    "calibration_family_routing": Path("models")
    / "market_probability_calibration_policy.json",
    "dependence_production_lambda": Path("models")
    / "combo_dependence_policy.json",
    "gate3_routing": Path("research")
    / "v2_gate3_deployment_artifacts"
    / "deployment_manifest.json",
}


# Validation checks an adaptive orchestrator must report before a fit may be
# promoted. The registry records them; it does not execute the model-side
# checks itself.
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


# Metadata keys whose names or values must never reach the manifest.
_SECRET_KEY_PATTERN = re.compile(
    r"api[_-]?key|secret|token|password|passwd|credential"
    r"|authorization|bearer",
    re.IGNORECASE,
)

_SECRET_VALUE_PATTERN = re.compile(
    r"BDL_API_KEY|ODDS_API_KEY", re.IGNORECASE
)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class RegistryError(RuntimeError):
    """Base class for every fail-closed registry refusal."""


class RegistryRootError(RegistryError):
    """The registry root was absent, ambiguous or unsafe."""


class ArchitectureContractViolation(RegistryError):
    """Supplied metadata does not match the frozen architecture contract."""


class FitAlreadyExists(RegistryError):
    """A final fit directory already exists and must never be replaced."""


class FitNotFound(RegistryError):
    """The referenced fit is not present in the registry."""


class IntegrityError(RegistryError):
    """Stored bytes do not match the recorded hashes."""


class StagedTreeError(RegistryError):
    """The staged candidate tree is unusable."""


class ValidationRefused(RegistryError):
    """Validation state is missing, incomplete or failed."""


class PromotionRefused(RegistryError):
    """Promotion or rollback preconditions were not met."""


class PromotionLocked(RegistryError):
    """Another promotion or rollback holds the exclusive lock."""


# --------------------------------------------------------------------------
# deterministic hashing helpers
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
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json_atomic(payload: Any, path: Path) -> None:
    """Write readable but deterministic JSON via a same-directory replace."""
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# frozen-choice derivation from the working tree
# --------------------------------------------------------------------------


def feature_schema_identity(
    project_root: Path,
) -> dict[str, list[str]]:
    """Read the declared feature schema without importing the model stack.

    Parsing rather than importing keeps the registry usable on a serving host
    that has no pandas or xgboost, which matters for portability.
    """
    path = project_root / FEATURES_RELATIVE_PATH

    if not path.exists():
        raise ArchitectureContractViolation(
            f"feature module missing: {path}"
        )

    tree = ast.parse(
        path.read_text(encoding="utf-8"), filename=str(path)
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
            f"{FEATURES_RELATIVE_PATH}: {', '.join(missing)}"
        )

    # Declared order is preserved; column order is part of schema identity.
    return {
        name.lower(): found[name]
        for name in FEATURE_SCHEMA_NAMES
    }


def feature_schema_hash(project_root: Path) -> str:
    return sha256_canonical(
        feature_schema_identity(project_root)
    )


def config_hash(project_root: Path) -> str:
    path = project_root / CONFIG_RELATIVE_PATH

    if not path.exists():
        raise ArchitectureContractViolation(
            f"model config missing: {path}"
        )

    return sha256_file(path)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def routing_maps(project_root: Path) -> dict[str, dict]:
    """Extract only the frozen routing decisions from the policy files.

    The policy files also carry daily fitted values -- ensemble weights and
    Platt coefficients among them -- so whole-file hashes would conflate the
    frozen choice with the value that is meant to move every day. Only the
    routing maps are extracted here, and only these are enforced.
    """
    out: dict[str, dict] = {}

    for key, relative in ROUTING_SOURCES.items():
        path = project_root / relative

        if not path.exists():
            raise ArchitectureContractViolation(
                f"policy file missing: {path}"
            )

        payload = _load_json(path)

        if key == "mean_model_routing":
            out[key] = {
                target: str(entry["selected_mode"])
                for target, entry in sorted(
                    payload["targets"].items()
                )
            }

        elif key == "marginal_family_routing":
            out[key] = {
                target: str(
                    entry["selected_distribution"]
                )
                for target, entry in sorted(
                    payload["targets"].items()
                )
            }

        elif key == "calibration_family_routing":
            # Only the selected method is frozen; the sibling
            # production_parameters are daily fitted Platt coefficients.
            out[key] = {
                prop: str(entry["selected_method"])
                for prop, entry in sorted(
                    payload["props"].items()
                )
            }

        elif key == "dependence_production_lambda":
            out[key] = {
                combo: float(
                    entry["production_lambda"]
                )
                for combo, entry in sorted(
                    payload["combos"].items()
                )
            }

        elif key == "gate3_routing":
            out[key] = {
                prop: str(route)
                for prop, route in sorted(
                    payload["gate3_policy"].items()
                )
            }

    return out


def routing_digests(project_root: Path) -> dict[str, str]:
    return {
        key: sha256_canonical(value)
        for key, value in routing_maps(project_root).items()
    }


def gate3_candidate_policy_id(project_root: Path) -> str:
    """Mirror the identifier gate3_v2.load_gate3_runtime derives."""
    path = (
        project_root / ROUTING_SOURCES["gate3_routing"]
    )

    manifest = _load_json(path)

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
    def routing_digests(self) -> dict[str, str]:
        return self.payload["frozen_routing_digests"]


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

    if (
        payload.get("architecture_reference_sha")
        != ARCHITECTURE_REFERENCE_SHA
    ):
        raise ArchitectureContractViolation(
            "contract architecture_reference_sha is "
            f"{payload.get('architecture_reference_sha')!r}, "
            f"expected {ARCHITECTURE_REFERENCE_SHA!r}"
        )

    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ArchitectureContractViolation(
            "unsupported contract_version "
            f"{payload.get('contract_version')!r}"
        )

    return ArchitectureContract(
        path=path,
        payload=payload,
        sha256=sha256_file(path),
    )


def verify_contract_against_tree(
    contract: ArchitectureContract,
    project_root: Path,
) -> None:
    """Refuse when the working tree has drifted from the frozen choices.

    This is the guard that stops a nightly fitting job from quietly becoming
    model selection, feature search or hyperparameter search.
    """
    observed_features = feature_schema_hash(project_root)

    expected_features = str(
        contract.frozen_choices["feature_schema_hash"]
    )

    if observed_features != expected_features:
        raise ArchitectureContractViolation(
            "feature schema hash drifted: tree "
            f"{observed_features}, contract {expected_features}"
        )

    observed_config = config_hash(project_root)

    expected_config = str(
        contract.frozen_choices["config_hash"]
    )

    if observed_config != expected_config:
        raise ArchitectureContractViolation(
            "hyperparameter config hash drifted: tree "
            f"{observed_config}, contract {expected_config}"
        )

    observed_routing = routing_digests(project_root)

    for key, expected in sorted(
        contract.routing_digests.items()
    ):
        observed = observed_routing.get(key)

        if observed != expected:
            raise ArchitectureContractViolation(
                f"frozen routing {key} drifted: tree "
                f"{observed}, contract {expected}"
            )

    observed_gate3 = gate3_candidate_policy_id(project_root)

    expected_gate3 = str(
        contract.frozen_choices[
            "gate3_candidate_policy_id"
        ]
    )

    if observed_gate3 != expected_gate3:
        raise ArchitectureContractViolation(
            "Gate 3 candidate policy id drifted: tree "
            f"{observed_gate3}, contract {expected_gate3}"
        )


# --------------------------------------------------------------------------
# fit metadata
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
        known = {
            f.name for f in cls.__dataclass_fields__.values()
        }

        unknown = sorted(set(payload) - known)

        if unknown:
            raise ArchitectureContractViolation(
                "unknown metadata fields: "
                + ", ".join(unknown)
            )

        missing = sorted(
            name
            for name, spec in cls.__dataclass_fields__.items()
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


def _assert_iso_date(value: str, label: str) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureContractViolation(
            f"{label} must be an ISO date, got {value!r}"
        ) from error


def _assert_no_secrets(payload: Any, trail: str = "") -> None:
    """Refuse credential-shaped keys, values and absolute paths."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _SECRET_KEY_PATTERN.search(str(key)):
                raise ArchitectureContractViolation(
                    "refusing to store credential-shaped field "
                    f"{trail}{key!r}"
                )

            _assert_no_secrets(
                value, f"{trail}{key}."
            )

        return

    if isinstance(payload, (list, tuple)):
        for item in payload:
            _assert_no_secrets(item, trail)

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


def scan_staged_tree(staged_dir: Path) -> dict[str, str]:
    """Hash every regular file, refusing anything that is not one."""
    if not staged_dir.is_dir():
        raise StagedTreeError(
            f"staged artifact tree is not a directory: {staged_dir}"
        )

    hashes: dict[str, str] = {}

    for path in sorted(staged_dir.rglob("*")):
        relative = path.relative_to(staged_dir).as_posix()

        if path.is_symlink():
            raise StagedTreeError(
                f"symlink rejected in staged tree: {relative}"
            )

        mode = os.lstat(path).st_mode

        if stat.S_ISDIR(mode):
            continue

        if not stat.S_ISREG(mode):
            raise StagedTreeError(
                "non-regular file rejected in staged tree: "
                f"{relative}"
            )

        hashes[relative] = sha256_file(path)

    if not hashes:
        raise StagedTreeError(
            f"staged artifact tree is empty: {staged_dir}"
        )

    return hashes


def checksum_inventory(hashes: dict[str, str]) -> str:
    """Render a sha256sum-compatible inventory, matching repo convention."""
    lines = [
        f"{hashes[relative]}  {relative}"
        for relative in sorted(hashes)
    ]

    return "\n".join(lines) + "\n"


def parse_checksum_inventory(text: str) -> dict[str, str]:
    out: dict[str, str] = {}

    for line in text.splitlines():
        if not line.strip():
            continue

        digest, _, relative = line.partition("  ")

        if not relative:
            raise IntegrityError(
                f"malformed checksum line: {line!r}"
            )

        out[relative] = digest

    return out


# --------------------------------------------------------------------------
# fit identity
# --------------------------------------------------------------------------


def fit_identity_inputs(
    metadata: FitMetadata,
    contract: ArchitectureContract,
    project_root: Path,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    """Assemble the immutable inputs the fit_id digest is taken over.

    Promotion state is deliberately absent: identity must be settled before
    anything decides whether the fit is good.
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
        "effective_n_jobs": int(
            metadata.effective_n_jobs
        ),
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


def is_fit_id(value: str) -> bool:
    return bool(
        re.fullmatch(
            re.escape(FIT_ID_PREFIX)
            + r"\d{8}_[0-9a-f]{"
            + str(FIT_ID_DIGEST_CHARS)
            + r"}",
            value,
        )
    )


# --------------------------------------------------------------------------
# registry
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
            f"{REGISTRY_ENV_VAR}. The registry is never defaulted into "
            "the repository."
        )

    root = Path(candidate).expanduser()

    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()

    if project_root is not None:
        resolved_project = project_root.resolve()

        if root == resolved_project or resolved_project in root.parents:
            raise RegistryRootError(
                f"registry root {root} is inside the repository at "
                f"{resolved_project}; fits are production state and "
                "must live outside version control"
            )

    return root


@contextmanager
def _exclusive_lock(
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
                f"another promotion holds {path}"
            ) from error

        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


class FitRegistry:
    """Filesystem registry of immutable daily fits and promotion state."""

    def __init__(
        self,
        root: Path,
        project_root: Path,
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
    def validation_dir(self) -> Path:
        return self.state_dir / "validation"

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

    @property
    def registration_lock_path(self) -> Path:
        return self.locks_dir / "registration.lock"

    def fit_dir(self, fit_id: str) -> Path:
        if not is_fit_id(fit_id):
            raise FitNotFound(
                f"not a well-formed fit_id: {fit_id!r}"
            )

        return self.fits_dir / fit_id

    def validation_path(self, fit_id: str) -> Path:
        return self.validation_dir / f"{fit_id}.json"

    def initialise(self) -> None:
        for path in (
            self.fits_dir,
            self.state_dir,
            self.validation_dir,
            self.locks_dir,
            self.staging_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # -- registration ---------------------------------------------------

    def register(
        self,
        staged_dir: Path,
        metadata: FitMetadata,
        contract: ArchitectureContract | None = None,
    ) -> str:
        """Register a staged candidate tree as an immutable fit.

        Registration never promotes. The returned fit begins life as
        REGISTERED / NOT PROMOTED.
        """
        self.initialise()

        if contract is None:
            contract = load_architecture_contract(
                self.project_root
            )

        self._assert_architecture_lock(metadata, contract)

        artifact_hashes = scan_staged_tree(staged_dir)

        identity = fit_identity_inputs(
            metadata,
            contract,
            self.project_root,
            artifact_hashes,
        )

        fit_id = derive_fit_id(identity)

        manifest = self._build_manifest(
            fit_id, metadata, contract, identity
        )

        _assert_no_secrets(manifest)

        with _exclusive_lock(self.registration_lock_path):
            final_dir = self.fit_dir(fit_id)

            if final_dir.exists():
                raise FitAlreadyExists(
                    f"fit {fit_id} is already registered at "
                    f"{final_dir}; immutable fits are never "
                    "replaced or merged into"
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

                # os.rename refuses a non-empty destination, so this is a
                # second guard behind the existence check above.
                os.rename(staging, final_dir)
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
                "training_cutoff "
                f"{metadata.training_cutoff} must be strictly "
                f"before fit_date {metadata.fit_date}"
            )

        if (
            metadata.architecture_reference_sha
            != contract.architecture_reference_sha
        ):
            raise ArchitectureContractViolation(
                "architecture_reference_sha mismatch: metadata "
                f"{metadata.architecture_reference_sha}, contract "
                f"{contract.architecture_reference_sha}"
            )

        verify_contract_against_tree(
            contract, self.project_root
        )

        frozen = contract.frozen_choices

        expected_seeds = {
            "core_seed": int(frozen["core_seed"]),
            "gate3_seed": int(frozen["gate3_seed"]),
            "effective_n_jobs": int(
                frozen["effective_n_jobs"]
            ),
        }

        observed_seeds = {
            "core_seed": int(metadata.core_seed),
            "gate3_seed": int(metadata.gate3_seed),
            "effective_n_jobs": int(
                metadata.effective_n_jobs
            ),
        }

        for key, expected in expected_seeds.items():
            if observed_seeds[key] != expected:
                raise ArchitectureContractViolation(
                    f"{key} mismatch: metadata "
                    f"{observed_seeds[key]}, contract {expected}"
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
                    "gate3_candidate_policy_id mismatch: metadata "
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
            "architecture_reference_sha": (
                metadata.architecture_reference_sha
            ),
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
            source = staged_dir / relative
            destination = artifacts / relative

            destination.parent.mkdir(
                parents=True, exist_ok=True
            )

            shutil.copyfile(source, destination)

        manifest_path = staging / "manifest.json"

        write_json_atomic(manifest, manifest_path)

        inventory = dict(artifact_hashes)

        # Verify the copied bytes before the inventory is sealed.
        for relative, expected in sorted(inventory.items()):
            observed = sha256_file(artifacts / relative)

            if observed != expected:
                raise IntegrityError(
                    "staged copy hash mismatch for "
                    f"{relative}: {observed} != {expected}"
                )

        sealed = {
            f"artifacts/{relative}": digest
            for relative, digest in inventory.items()
        }

        sealed["manifest.json"] = sha256_file(manifest_path)

        (staging / "SHA256SUMS").write_text(
            checksum_inventory(sealed), encoding="utf-8"
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

        Because fit_id is a digest over the manifest's own identity inputs,
        an edited manifest yields a different fit_id than the directory it
        sits in, and an edited artifact fails its recorded hash. Tampering
        with either is therefore detected without a separate signature.
        """
        fit_dir = self.fit_dir(fit_id)

        if not fit_dir.is_dir():
            raise FitNotFound(
                f"fit {fit_id} is not registered at {fit_dir}"
            )

        manifest = self.load_manifest(fit_id)

        if manifest.get("fit_id") != fit_id:
            raise IntegrityError(
                "manifest fit_id "
                f"{manifest.get('fit_id')!r} does not match "
                f"directory {fit_id!r}"
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

        artifacts = fit_dir / "artifacts"

        expected = {
            str(k): str(v)
            for k, v in identity["artifact_hashes"].items()
        }

        observed = scan_staged_tree(artifacts)

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
                    "artifact tampering detected in "
                    f"{fit_id}: {relative} hashes "
                    f"{observed[relative]}, manifest records "
                    f"{expected[relative]}"
                )

        self._verify_inventory(fit_id, fit_dir, expected)

        if contract is None:
            contract = load_architecture_contract(
                self.project_root
            )

        if (
            manifest.get("architecture_contract_sha256")
            != contract.sha256
        ):
            raise ArchitectureContractViolation(
                f"fit {fit_id} was registered against contract "
                f"{manifest.get('architecture_contract_sha256')}, "
                f"current contract is {contract.sha256}"
            )

        return manifest

    def _verify_inventory(
        self,
        fit_id: str,
        fit_dir: Path,
        expected_artifacts: dict[str, str],
    ) -> None:
        inventory_path = fit_dir / "SHA256SUMS"

        if not inventory_path.exists():
            raise IntegrityError(
                f"fit {fit_id} has no SHA256SUMS inventory"
            )

        inventory = parse_checksum_inventory(
            inventory_path.read_text(encoding="utf-8")
        )

        manifest_digest = inventory.get("manifest.json")

        if manifest_digest is None:
            raise IntegrityError(
                f"fit {fit_id} inventory omits manifest.json"
            )

        observed_manifest = sha256_file(
            fit_dir / "manifest.json"
        )

        if observed_manifest != manifest_digest:
            raise IntegrityError(
                "manifest tampering detected in "
                f"{fit_id}: inventory records "
                f"{manifest_digest}, file hashes "
                f"{observed_manifest}"
            )

        for relative, digest in sorted(
            expected_artifacts.items()
        ):
            key = f"artifacts/{relative}"

            if inventory.get(key) != digest:
                raise IntegrityError(
                    f"fit {fit_id} inventory disagrees with "
                    f"manifest for {relative}"
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
                f"cannot record validation for unregistered fit {fit_id}"
            )

        unknown = sorted(
            set(checks) - set(REQUIRED_VALIDATION_CHECKS)
        )

        if unknown:
            raise ValidationRefused(
                "unknown validation checks: "
                + ", ".join(unknown)
            )

        normalised = {
            name: bool(value)
            for name, value in sorted(checks.items())
        }

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
        }

        _assert_no_secrets(record)

        self.validation_dir.mkdir(
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
        record = self.load_validation(fit_id)

        if record is None:
            raise ValidationRefused(
                f"fit {fit_id} has no validation record; a "
                "registered fit is never production by default"
            )

        checks = record.get("checks", {})

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

        failed = sorted(
            name
            for name in REQUIRED_VALIDATION_CHECKS
            if not bool(checks[name])
        )

        if failed:
            raise ValidationRefused(
                f"fit {fit_id} failed required validation "
                "checks: " + ", ".join(failed)
            )

    # -- promotion state ------------------------------------------------

    def load_promotion_state(self) -> dict[str, Any]:
        path = self.promotion_state_path

        if not path.exists():
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

        return _load_json(path)

    def _write_promotion_state(
        self, state: dict[str, Any]
    ) -> None:
        _assert_no_secrets(state)

        self.state_dir.mkdir(parents=True, exist_ok=True)

        write_json_atomic(
            state, self.promotion_state_path
        )

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
        with _exclusive_lock(
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

            current = state.get("current_good_fit_id")

            if current == fit_id:
                raise PromotionRefused(
                    f"fit {fit_id} is already the current good fit"
                )

            promoted_at = _utc_now()

            history = list(state.get("history", []))

            history.append(
                {
                    "action": "promote",
                    "actor": actor,
                    "at": promoted_at,
                    "fit_id": fit_id,
                    "previous_fit_id": current,
                    "reason": reason,
                }
            )

            new_state = {
                "current_good_fit_id": fit_id,
                "history": history,
                "previous_good_fit_id": current,
                "promoted_at": promoted_at,
                "promotion_actor": actor,
                "promotion_reason": reason,
                "schema_version": (
                    PROMOTION_STATE_SCHEMA_VERSION
                ),
            }

            self._write_promotion_state(new_state)

            return new_state

    def rollback(
        self,
        reason: str,
        actor: str | None = None,
        blocking: bool = True,
    ) -> dict[str, Any]:
        """Restore the previous good fit after re-verifying it end to end."""
        with _exclusive_lock(
            self.promotion_lock_path, blocking=blocking
        ):
            state = self.load_promotion_state()

            current = state.get("current_good_fit_id")

            target = state.get("previous_good_fit_id")

            if target is None:
                raise PromotionRefused(
                    "no previous good fit is recorded; nothing "
                    "to roll back to"
                )

            contract = load_architecture_contract(
                self.project_root
            )

            # Fail closed: a corrupt or absent rollback target must never
            # displace a fit that is currently serving.
            self.verify(target, contract=contract)

            self.assert_validation_pass(target)

            rolled_at = _utc_now()

            history = list(state.get("history", []))

            history.append(
                {
                    "action": "rollback",
                    "actor": actor,
                    "at": rolled_at,
                    "fit_id": target,
                    "previous_fit_id": current,
                    "reason": reason,
                }
            )

            new_state = {
                "current_good_fit_id": target,
                "history": history,
                "previous_good_fit_id": current,
                "promoted_at": rolled_at,
                "promotion_actor": actor,
                "promotion_reason": reason,
                "schema_version": (
                    PROMOTION_STATE_SCHEMA_VERSION
                ),
            }

            self._write_promotion_state(new_state)

            return new_state

    def current(self) -> dict[str, Any]:
        state = self.load_promotion_state()

        return {
            "current_good_fit_id": state.get(
                "current_good_fit_id"
            ),
            "previous_good_fit_id": state.get(
                "previous_good_fit_id"
            ),
            "promoted_at": state.get("promoted_at"),
            "promotion_actor": state.get(
                "promotion_actor"
            ),
            "promotion_reason": state.get(
                "promotion_reason"
            ),
        }

    def list_fits(self) -> list[str]:
        if not self.fits_dir.is_dir():
            return []

        return sorted(
            path.name
            for path in self.fits_dir.iterdir()
            if path.is_dir() and is_fit_id(path.name)
        )
