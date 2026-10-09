"""Computed validation checks for one adaptive daily fit.

Every name in the registry's ``REQUIRED_VALIDATION_CHECKS`` is answered here by
reading the candidate tree, the verified rolling state and the frozen policy
sources. Nothing is asserted by constant.

Two of the names cannot be answered from an offline fit at all. The repository
is explicit about why: ``gate3_role_readiness`` needs a live captured lineup
snapshot and ``t20_protocol_compatible`` needs a real T-20 capture, and
``docs/wizardofodds/ADAPTIVE_DAILY_TRAINING.md`` records that "fabricating
either would be asserting something about a capture that never happened". So
they are computed when a capture is present and reported ``NOT_EVALUABLE`` when
one is not. A ``NOT_EVALUABLE`` check is not recorded, which leaves it in the
registry's missing set, which is what keeps the candidate unpromotable. That is
the frozen behaviour; this module makes it a measurement rather than an
omission.

No check in this module changes a model parameter, a threshold, a calibration
family or a promotion rule. A check may only observe.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .adaptive_fit_registry import (
    REQUIRED_VALIDATION_CHECKS,
    ArchitectureContract,
    config_hash,
    feature_schema_hash,
    frozen_policy_digests,
    sha256_file,
    verify_contract_against_tree,
)


# The serving artifacts the prediction smoke test exercises.
MARGINALS_RELATIVE = Path("models") / "marginals.joblib"

COPULA_RELATIVE = Path("models") / "copula.joblib"

TRAINING_MANIFEST_RELATIVE = Path("training_data_manifest.json")

PROVENANCE_DIRECTORY = "provenance"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# Probability identities the smoke test holds the serving marginals to. These
# are arithmetic facts about a distribution, not model thresholds.
PROBABILITY_SUM_TOLERANCE = 1e-9

CORRELATION_DIAGONAL_TOLERANCE = 1e-9

# The smallest eigenvalue a serving correlation matrix may have. Mirrors the
# shadow runtime's PSD floor convention: a correlation matrix that is not
# positive semi-definite cannot be simulated from.
MINIMUM_EIGENVALUE = -1e-8

# Lines and means the smoke test prices at. Deliberately ordinary NBA values
# chosen to exercise an integer line (where a push is possible) and a
# half-point line (where it is not). They parameterise a smoke test; they are
# not model inputs and no fitted value depends on them.
SMOKE_TEST_LINES = (0.5, 1.5, 4.5, 6.0, 20.5)

SMOKE_TEST_MU = 11.0

NOT_EVALUABLE = "NOT_EVALUABLE"


class ValidationComputationError(RuntimeError):
    """A check could not be computed for a reason that is itself a failure."""


@dataclass
class CheckResult:
    """One validation answer, with the evidence it was derived from."""

    name: str
    passed: bool | None
    evidence: str
    values: dict[str, Any] = field(default_factory=dict)
    contract: str | None = None
    error: str | None = None

    @property
    def evaluable(self) -> bool:
        return self.passed is not None

    def payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "error": self.error,
            "evidence": self.evidence,
            "name": self.name,
            "passed": self.passed,
            "values": self.values,
        }


@dataclass
class ValidationReport:
    """Every computed answer for one fit, plus the two derived views of it."""

    results: tuple[CheckResult, ...]

    @property
    def by_name(self) -> dict[str, CheckResult]:
        return {result.name: result for result in self.results}

    def recorded_checks(self) -> dict[str, bool]:
        """The boolean map the registry records. Omits NOT_EVALUABLE names."""
        return {
            result.name: bool(result.passed)
            for result in self.results
            if result.evaluable
        }

    def deferred(self) -> list[str]:
        """Names that could not be evaluated, so the registry stays blocked."""
        return sorted(
            result.name for result in self.results if not result.evaluable
        )

    def failed(self) -> list[str]:
        return sorted(
            result.name
            for result in self.results
            if result.evaluable and not result.passed
        )

    @property
    def passed(self) -> bool:
        return not self.failed()

    def payload(self) -> dict[str, Any]:
        return {
            "checks": [result.payload() for result in self.results],
            "deferred": self.deferred(),
            "failed": self.failed(),
            "passed": self.passed,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_PATTERN.match(value))


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number)


def _guarded(
    name: str,
    contract: str,
    computation: Callable[[], CheckResult],
) -> CheckResult:
    """Run one check so that an exception becomes a failure, never a crash.

    A check that raises has not proved anything, and a fit whose validation
    crashed must not be recorded as validated. So the exception is captured as
    ``passed=False`` with its text, which is a recorded failure rather than a
    silent gap.
    """
    try:
        return computation()
    except Exception as error:  # noqa: BLE001 - deliberately total
        return CheckResult(
            name=name,
            passed=False,
            evidence="the check raised while being computed",
            contract=contract,
            error=f"{type(error).__name__}: {error}",
        )


def _load_joblib(path: Path) -> Any:
    import joblib

    return joblib.load(path)


# --------------------------------------------------------------------------
# the individual checks
# --------------------------------------------------------------------------


def check_training_completed(
    *,
    required_stages: tuple[str, ...],
    observed_stages: tuple[str, ...],
) -> CheckResult:
    """Every fit stage in the frozen DAG actually ran."""
    contract = "every stage in DAG_STAGES up to candidate_assembly ran"

    def compute() -> CheckResult:
        missing = [
            stage for stage in required_stages if stage not in observed_stages
        ]

        return CheckResult(
            name="training_completed",
            passed=not missing,
            evidence=(
                f"{len(observed_stages)} stage(s) recorded; "
                f"{len(missing)} required stage(s) missing"
            ),
            values={
                "missing_stages": missing,
                "observed_stages": list(observed_stages),
                "required_stages": list(required_stages),
            },
            contract=contract,
            error=(
                "stages did not run: " + ", ".join(missing) if missing else None
            ),
        )

    return _guarded("training_completed", contract, compute)


def check_data_refresh_valid(
    *,
    state: dict[str, Any],
    plan_fingerprint: str,
) -> CheckResult:
    """The verified rolling state is the one this fit planned against."""
    contract = (
        "the re-verified rolling state fingerprint equals the planned one and "
        "every rolling dataset carries a sha256 semantic fingerprint"
    )

    def compute() -> CheckResult:
        observed = str(state.get("datasets_fingerprint", ""))

        datasets = state.get("datasets") or {}

        unfingerprinted = sorted(
            label
            for label, record in datasets.items()
            if not _is_sha256(record.get("semantic_fingerprint"))
        )

        empty = sorted(
            label
            for label, record in datasets.items()
            if int(record.get("row_count", 0)) <= 0
        )

        agrees = bool(observed) and observed == str(plan_fingerprint)

        passed = agrees and not unfingerprinted and not empty

        reasons = []

        if not agrees:
            reasons.append(
                "rolling-state fingerprint moved between planning and "
                "validation"
            )

        if unfingerprinted:
            reasons.append(
                "datasets without a sha256 fingerprint: "
                + ", ".join(unfingerprinted)
            )

        if empty:
            reasons.append("empty datasets: " + ", ".join(empty))

        return CheckResult(
            name="data_refresh_valid",
            passed=passed,
            evidence=(
                f"{len(datasets)} rolling dataset(s); fingerprint "
                f"{observed[:12] or 'absent'}"
            ),
            values={
                "dataset_count": len(datasets),
                "observed_fingerprint": observed,
                "planned_fingerprint": str(plan_fingerprint),
                "row_counts": {
                    label: int(record.get("row_count", 0))
                    for label, record in sorted(datasets.items())
                },
            },
            contract=contract,
            error="; ".join(reasons) or None,
        )

    return _guarded("data_refresh_valid", contract, compute)


def check_history_regression_check(
    *,
    state: dict[str, Any],
    parent_manifest: dict[str, Any] | None,
) -> CheckResult:
    """No rolling dataset lost rows relative to the parent fit.

    History may only grow. A refresh that replaced good live state with a
    smaller partition is the failure mode this exists to catch, and it is
    checked against the previous fit's own recorded counts rather than against
    a number written here.
    """
    contract = "every rolling dataset row count is >= the parent fit's count"

    def compute() -> CheckResult:
        datasets = state.get("datasets") or {}

        observed = {
            label: int(record.get("row_count", 0))
            for label, record in sorted(datasets.items())
        }

        if not parent_manifest:
            return CheckResult(
                name="history_regression_check",
                passed=True,
                evidence=(
                    "no parent fit is registered, so there is no earlier "
                    "corpus this one could have regressed against"
                ),
                values={"observed_row_counts": observed, "parent": None},
                contract=contract,
            )

        recorded = parent_manifest.get("rolling_state_datasets") or {}

        previous = {
            label: int(record.get("row_count", 0))
            for label, record in recorded.items()
        }

        regressed = sorted(
            f"{label}: {observed[label]} < {previous[label]}"
            for label in sorted(set(observed) & set(previous))
            if observed[label] < previous[label]
        )

        dropped = sorted(set(previous) - set(observed))

        passed = not regressed and not dropped

        reasons = []

        if regressed:
            reasons.append("datasets lost rows: " + ", ".join(regressed))

        if dropped:
            reasons.append("datasets disappeared: " + ", ".join(dropped))

        return CheckResult(
            name="history_regression_check",
            passed=passed,
            evidence=(
                f"compared {len(previous)} parent dataset count(s) against "
                f"{len(observed)} current one(s)"
            ),
            values={
                "dropped_datasets": dropped,
                "observed_row_counts": observed,
                "parent_row_counts": previous,
                "regressed": regressed,
            },
            contract=contract,
            error="; ".join(reasons) or None,
        )

    return _guarded("history_regression_check", contract, compute)


def check_advanced_coverage_check(
    *,
    manifest: dict[str, Any],
    advanced_start_season: int,
) -> CheckResult:
    """No season in the training corpus has box scores but no advanced data.

    The frozen advanced floor is ADVANCED_START_SEASON, so a season below it
    legitimately has no advanced partition. Above it, a season that contributed
    stats without contributing advanced data would be fit on a silently
    narrower feature set than the schema declares, which is the failure this
    catches. The required set is derived from the corpus the fit actually
    consumed rather than from a season list written here, so it cannot drift
    away from the data.
    """
    contract = (
        "every season at or above ADVANCED_START_SEASON that contributed "
        "stats.parquet to the eligible corpus also contributed advanced.parquet"
    )

    def compute() -> CheckResult:
        eligible = manifest.get("eligible_input_sha256") or {}

        stats_seasons: set[int] = set()
        advanced_seasons: set[int] = set()

        for relative in eligible:
            text = str(relative)

            stats = re.search(r"seasons/season=(\d{4})/stats\.parquet$", text)

            if stats:
                stats_seasons.add(int(stats.group(1)))

            advanced = re.search(
                r"advanced/season=(\d{4})/advanced\.parquet$", text
            )

            if advanced:
                advanced_seasons.add(int(advanced.group(1)))

        floor = int(advanced_start_season)

        required = {season for season in stats_seasons if season >= floor}

        missing = sorted(required - advanced_seasons)

        orphaned = sorted(
            season
            for season in advanced_seasons - stats_seasons
            if season >= floor
        )

        passed = not missing and not orphaned

        reasons = []

        if missing:
            reasons.append(
                "seasons with stats but no advanced data: "
                + ", ".join(str(season) for season in missing)
            )

        if orphaned:
            reasons.append(
                "seasons with advanced data but no stats: "
                + ", ".join(str(season) for season in orphaned)
            )

        return CheckResult(
            name="advanced_coverage_check",
            passed=passed,
            evidence=(
                f"{len(advanced_seasons)} advanced season(s) cover "
                f"{len(required)} season(s) at or above the {floor} floor"
            ),
            values={
                "advanced_seasons": sorted(advanced_seasons),
                "advanced_start_season": floor,
                "missing_seasons": missing,
                "orphaned_advanced_seasons": orphaned,
                "stats_seasons": sorted(stats_seasons),
            },
            contract=contract,
            error="; ".join(reasons) or None,
        )

    return _guarded("advanced_coverage_check", contract, compute)


def check_required_artifacts_present(
    *,
    artifact_hashes: dict[str, str],
    required_prefixes: tuple[str, ...],
    required_files: tuple[str, ...],
) -> CheckResult:
    """The candidate carries serving artifacts, not just its own provenance."""
    contract = (
        "every REQUIRED_CANDIDATE_FILES entry is present and every "
        "REQUIRED_CANDIDATE_PREFIXES prefix matches at least one artifact"
    )

    def compute() -> CheckResult:
        missing = [name for name in required_files if name not in artifact_hashes]

        for prefix in required_prefixes:
            if not any(name.startswith(prefix) for name in artifact_hashes):
                missing.append(f"{prefix}*")

        serving = sorted(
            name
            for name in (
                MARGINALS_RELATIVE.as_posix(),
                COPULA_RELATIVE.as_posix(),
            )
            if name in artifact_hashes
        )

        return CheckResult(
            name="required_artifacts_present",
            passed=not missing,
            evidence=(
                f"{len(artifact_hashes)} artifact(s) in the candidate tree; "
                f"serving objects present: {', '.join(serving) or 'none'}"
            ),
            values={
                "artifact_count": len(artifact_hashes),
                "missing": sorted(missing),
                "serving_artifacts": serving,
            },
            contract=contract,
            error="missing: " + ", ".join(sorted(missing)) if missing else None,
        )

    return _guarded("required_artifacts_present", contract, compute)


def check_artifact_hashes_valid(
    *,
    candidate: Path,
    artifact_hashes: dict[str, str],
) -> CheckResult:
    """Re-hash the candidate tree and prove it did not move during validation.

    The hashes were taken when the tree was first walked. Taking them again is
    what distinguishes "these are plausible digests" from "these are still the
    digests of the bytes on disk".
    """
    contract = (
        "every recorded digest is a sha256 and still equals the file's "
        "re-computed digest"
    )

    def compute() -> CheckResult:
        root = Path(candidate)

        malformed = sorted(
            name
            for name, digest in artifact_hashes.items()
            if not _is_sha256(digest)
        )

        mismatched: list[str] = []
        vanished: list[str] = []

        for name, digest in sorted(artifact_hashes.items()):
            path = root / name

            if not path.is_file():
                vanished.append(name)
                continue

            if sha256_file(path) != digest:
                mismatched.append(name)

        passed = not malformed and not mismatched and not vanished

        reasons = []

        if malformed:
            reasons.append("malformed digests: " + ", ".join(malformed))

        if vanished:
            reasons.append("files disappeared: " + ", ".join(vanished))

        if mismatched:
            reasons.append("files changed: " + ", ".join(mismatched))

        return CheckResult(
            name="artifact_hashes_valid",
            passed=passed,
            evidence=f"re-hashed {len(artifact_hashes)} artifact(s)",
            values={
                "malformed": malformed,
                "mismatched": mismatched,
                "rehashed_count": len(artifact_hashes),
                "vanished": vanished,
            },
            contract=contract,
            error="; ".join(reasons) or None,
        )

    return _guarded("artifact_hashes_valid", contract, compute)


def check_finite_values_check(
    *,
    finite: bool,
    offenders: list[str],
) -> CheckResult:
    """No structured numerical parameter is NaN or infinite."""
    contract = "no JSON numeric value in the candidate tree is NaN or infinite"

    def compute() -> CheckResult:
        return CheckResult(
            name="finite_values_check",
            passed=bool(finite) and not offenders,
            evidence=f"{len(offenders)} non-finite structured value(s)",
            values={"offenders": sorted(offenders)[:50]},
            contract=contract,
            error=(
                "non-finite values: " + ", ".join(sorted(offenders)[:10])
                if offenders
                else None
            ),
        )

    return _guarded("finite_values_check", contract, compute)


def check_feature_schema_match(
    *,
    project_root: Path,
    manifest: dict[str, Any],
) -> CheckResult:
    """The feature schema the fit recorded is still the tree's schema."""
    contract = (
        "feature_schema_hash(project_root) equals the manifest's recorded "
        "feature_schema_hash, and so does config_hash"
    )

    def compute() -> CheckResult:
        observed_schema = feature_schema_hash(Path(project_root))
        observed_config = config_hash(Path(project_root))

        recorded_schema = str(manifest.get("feature_schema_hash", ""))
        recorded_config = str(manifest.get("config_hash", ""))

        schema_agrees = observed_schema == recorded_schema
        config_agrees = observed_config == recorded_config

        reasons = []

        if not schema_agrees:
            reasons.append("feature schema hash differs from the manifest")

        if not config_agrees:
            reasons.append("model config hash differs from the manifest")

        return CheckResult(
            name="feature_schema_match",
            passed=schema_agrees and config_agrees,
            evidence=(
                f"schema {observed_schema[:12]} vs recorded "
                f"{recorded_schema[:12] or 'absent'}"
            ),
            values={
                "observed_config_hash": observed_config,
                "observed_feature_schema_hash": observed_schema,
                "recorded_config_hash": recorded_config,
                "recorded_feature_schema_hash": recorded_schema,
            },
            contract=contract,
            error="; ".join(reasons) or None,
        )

    return _guarded("feature_schema_match", contract, compute)


def check_architecture_contract_match(
    *,
    project_root: Path,
    contract_object: ArchitectureContract,
    manifest: dict[str, Any],
) -> CheckResult:
    """The architecture contract still describes this working tree."""
    description = (
        "verify_contract_against_tree passes and the manifest's recorded "
        "contract sha256 equals the loaded contract's"
    )

    def compute() -> CheckResult:
        verify_contract_against_tree(contract_object, Path(project_root))

        recorded = str(manifest.get("architecture_contract_sha256", ""))

        agrees = recorded == contract_object.sha256

        return CheckResult(
            name="architecture_contract_match",
            passed=agrees,
            evidence=(
                "contract verified against the tree; recorded sha256 "
                f"{recorded[:12] or 'absent'}"
            ),
            values={
                "loaded_contract_sha256": contract_object.sha256,
                "recorded_contract_sha256": recorded,
                "reference_sha": contract_object.architecture_reference_sha,
            },
            contract=description,
            error=(
                None
                if agrees
                else "the manifest records a different architecture contract"
            ),
        )

    return _guarded("architecture_contract_match", description, compute)


def check_source_lineage_match(
    *,
    project_root: Path,
    candidate: Path,
    manifest: dict[str, Any],
    lineage_sources: dict[str, Path],
) -> CheckResult:
    """Every provenance copy in the candidate is byte-identical to its source.

    The candidate carries the protocol, the contracts, the config and the
    frozen policies it was fit under. If a copy has drifted from the working
    tree, the fit_id digest describes a lineage that no longer exists.

    ``lineage_sources`` maps a provenance filename to its working-tree path.
    Every file in the provenance directory must appear in that map: an
    undeclared provenance artifact is a lineage claim nothing can verify, so it
    fails rather than being skipped.
    """
    description = (
        "every file in candidate/provenance is a declared lineage source with "
        "the same sha256 as its working-tree original, the directory is not "
        "empty, and the manifest's frozen policy digests still match the tree"
    )

    def compute() -> CheckResult:
        provenance = Path(candidate) / PROVENANCE_DIRECTORY

        if not provenance.is_dir():
            return CheckResult(
                name="source_lineage_match",
                passed=False,
                evidence="the candidate has no provenance directory",
                contract=description,
                error="candidate/provenance is absent",
            )

        copied = sorted(
            path for path in provenance.rglob("*") if path.is_file()
        )

        if not copied:
            return CheckResult(
                name="source_lineage_match",
                passed=False,
                evidence="the candidate provenance directory is empty",
                contract=description,
                error="the candidate records no lineage at all",
            )

        mismatched: list[str] = []
        undeclared: list[str] = []
        unresolvable: list[str] = []

        for path in copied:
            name = path.name

            relative = lineage_sources.get(name)

            if relative is None:
                undeclared.append(name)
                continue

            source = Path(project_root) / relative

            if not source.is_file():
                unresolvable.append(str(relative))
                continue

            if sha256_file(path) != sha256_file(source):
                mismatched.append(name)

        observed_digests = frozen_policy_digests(Path(project_root))

        recorded_digests = manifest.get("frozen_policy_digests") or {}

        drifted = sorted(
            name
            for name in sorted(set(observed_digests) | set(recorded_digests))
            if observed_digests.get(name) != recorded_digests.get(name)
        )

        passed = (
            not mismatched and not undeclared and not unresolvable and not drifted
        )

        reasons = []

        if undeclared:
            reasons.append(
                "undeclared provenance artifacts: " + ", ".join(undeclared)
            )

        if unresolvable:
            reasons.append(
                "declared sources missing from the tree: "
                + ", ".join(unresolvable)
            )

        if mismatched:
            reasons.append(
                "provenance copies differ from source: " + ", ".join(mismatched)
            )

        if drifted:
            reasons.append(
                "frozen policy digests drifted: " + ", ".join(drifted)
            )

        return CheckResult(
            name="source_lineage_match",
            passed=passed,
            evidence=(
                f"compared {len(copied)} provenance copy/copies and "
                f"{len(observed_digests)} frozen policy digest(s)"
            ),
            values={
                "compared": [path.name for path in copied],
                "drifted_policy_digests": drifted,
                "mismatched": mismatched,
                "undeclared": undeclared,
                "unresolvable_sources": unresolvable,
            },
            contract=description,
            error="; ".join(reasons) or None,
        )

    return _guarded("source_lineage_match", description, compute)


def check_marginal_fit_valid(
    *,
    candidate: Path,
    targets: tuple[str, ...],
    frozen_family: str,
) -> CheckResult:
    """The serving marginals load and cover every frozen target.

    ``frozen_family`` is the family the architecture chose and is read, not
    decided, here: a fit may report ``nb`` where zero inflation did not earn
    its parameters, so the family is checked against the allowed set the
    frozen selector itself can return rather than against one literal.
    """
    description = (
        "models/marginals.joblib loads as a mapping holding one fitted marginal "
        "per frozen target, each of a kind the frozen family admits"
    )

    allowed_kinds = {"nb", "zinb"} if frozen_family == "zinb" else {frozen_family}

    def compute() -> CheckResult:
        path = Path(candidate) / MARGINALS_RELATIVE

        if not path.is_file():
            return CheckResult(
                name="marginal_fit_valid",
                passed=False,
                evidence=f"{MARGINALS_RELATIVE.as_posix()} is absent",
                contract=description,
                error="the candidate has no serving marginals",
            )

        marginals = _load_joblib(path)

        if not isinstance(marginals, dict):
            return CheckResult(
                name="marginal_fit_valid",
                passed=False,
                evidence=(
                    "the serving marginals deserialised as "
                    f"{type(marginals).__name__}, not a mapping"
                ),
                contract=description,
                error="serving marginals are not a target-keyed mapping",
            )

        missing = sorted(set(targets) - set(marginals))

        kinds = {
            str(target): str(getattr(marginal, "kind", "unknown"))
            for target, marginal in sorted(marginals.items())
        }

        wrong_family = sorted(
            f"{target}: {kind}"
            for target, kind in kinds.items()
            if kind not in allowed_kinds
        )

        uncallable = sorted(
            str(target)
            for target, marginal in sorted(marginals.items())
            if not callable(getattr(marginal, "over_under_push", None))
        )

        passed = not missing and not wrong_family and not uncallable

        reasons = []

        if missing:
            reasons.append("targets without a marginal: " + ", ".join(missing))

        if wrong_family:
            reasons.append(
                "marginals outside the frozen family: " + ", ".join(wrong_family)
            )

        if uncallable:
            reasons.append(
                "marginals that cannot price an over/under: "
                + ", ".join(uncallable)
            )

        return CheckResult(
            name="marginal_fit_valid",
            passed=passed,
            evidence=f"{len(marginals)} fitted marginal(s) loaded",
            values={
                "allowed_kinds": sorted(allowed_kinds),
                "frozen_family": frozen_family,
                "kinds": kinds,
                "missing_targets": missing,
            },
            contract=description,
            error="; ".join(reasons) or None,
        )

    return _guarded("marginal_fit_valid", description, compute)


def check_calibration_valid(
    *,
    calibration_hashes: dict[str, Any],
    calibration_fallbacks: dict[str, str],
    calibration_routes: dict[str, str],
) -> CheckResult:
    """Every PROP-routed prop has a calibration digest, and RAW stayed RAW.

    The route table is frozen. This check proves the fit respected it: a prop
    routed ``prop`` must carry a calibration digest, and a prop routed ``raw``
    must not have acquired one, because that would be a silent methodology
    change.
    """
    description = (
        "every prop routed 'prop' carries a sha256 calibration digest, no prop "
        "routed 'raw' carries one, and every fallback names a known route"
    )

    def compute() -> CheckResult:
        prop_routed = sorted(
            name for name, route in calibration_routes.items() if route == "prop"
        )

        raw_routed = sorted(
            name for name, route in calibration_routes.items() if route == "raw"
        )

        missing = sorted(
            name
            for name in prop_routed
            if not _is_sha256(calibration_hashes.get(name))
        )

        unexpected = sorted(
            name for name in raw_routed if name in calibration_hashes
        )

        unknown_fallbacks = sorted(
            name for name in calibration_fallbacks if name not in calibration_routes
        )

        raw_fallbacks = sorted(
            name for name in calibration_fallbacks if name in raw_routed
        )

        passed = (
            not missing
            and not unexpected
            and not unknown_fallbacks
            and not raw_fallbacks
        )

        reasons = []

        if missing:
            reasons.append(
                "prop-routed props without a calibration digest: "
                + ", ".join(missing)
            )

        if unexpected:
            reasons.append(
                "raw-routed props that acquired a calibration digest: "
                + ", ".join(unexpected)
            )

        if unknown_fallbacks:
            reasons.append(
                "fallbacks for props outside the frozen route table: "
                + ", ".join(unknown_fallbacks)
            )

        if raw_fallbacks:
            reasons.append(
                "raw-routed props recorded a calibration fallback: "
                + ", ".join(raw_fallbacks)
            )

        return CheckResult(
            name="calibration_valid",
            passed=passed,
            evidence=(
                f"{len(prop_routed)} prop-routed and {len(raw_routed)} "
                f"raw-routed prop(s); {len(calibration_fallbacks)} fallback(s)"
            ),
            values={
                "fallbacks": dict(sorted(calibration_fallbacks.items())),
                "missing_digests": missing,
                "prop_routed": prop_routed,
                "raw_routed": raw_routed,
                "unexpected_digests": unexpected,
            },
            contract=description,
            error="; ".join(reasons) or None,
        )

    return _guarded("calibration_valid", description, compute)


def check_prediction_smoke_test(
    *,
    candidate: Path,
    targets: tuple[str, ...],
) -> CheckResult:
    """Price something with the candidate's own serving objects.

    This is the check that covers the serialized estimators, which
    ``structured_values_are_finite`` explicitly cannot see. It deserialises the
    marginals and the copula the live pricing path would load, asks each
    marginal for an over/under/push at ordinary NBA lines, and asks the copula
    for a correlation matrix. It asserts only arithmetic identities -- every
    probability finite and in [0, 1], over + under + push == 1, a unit
    diagonal, and a positive semi-definite matrix. It does not assert any
    particular number, because asserting a number here would freeze a value the
    daily fit is allowed to re-estimate.
    """
    description = (
        "the candidate's marginals and copula deserialise and produce finite "
        "probabilities in [0, 1] summing to 1, over a PSD unit-diagonal "
        "correlation matrix"
    )

    def compute() -> CheckResult:
        import numpy as np
        import pandas as pd

        marginal_path = Path(candidate) / MARGINALS_RELATIVE
        copula_path = Path(candidate) / COPULA_RELATIVE

        absent = sorted(
            relative.as_posix()
            for relative, path in (
                (MARGINALS_RELATIVE, marginal_path),
                (COPULA_RELATIVE, copula_path),
            )
            if not path.is_file()
        )

        if absent:
            return CheckResult(
                name="prediction_smoke_test",
                passed=False,
                evidence="serving objects absent: " + ", ".join(absent),
                values={"absent": absent},
                contract=description,
                error="nothing could be priced; the candidate cannot serve",
            )

        marginals = _load_joblib(marginal_path)
        copula = _load_joblib(copula_path)

        if not isinstance(marginals, dict) or not marginals:
            return CheckResult(
                name="prediction_smoke_test",
                passed=False,
                evidence="the serving marginals are not a non-empty mapping",
                contract=description,
                error="serving marginals are unusable",
            )

        priced = 0
        offenders: list[str] = []

        for target in sorted(set(targets) & set(marginals)):
            marginal = marginals[target]

            row = pd.Series(
                {
                    "player_id": 1,
                    "minutes": 30.0,
                    "mu": SMOKE_TEST_MU,
                    target: SMOKE_TEST_MU,
                }
            )

            for line in SMOKE_TEST_LINES:
                over, under, push = marginal.over_under_push(
                    float(line), SMOKE_TEST_MU, row
                )

                priced += 1

                for label, value in (
                    ("over", over),
                    ("under", under),
                    ("push", push),
                ):
                    if not _finite(value) or not 0.0 <= float(value) <= 1.0:
                        offenders.append(
                            f"{target}@{line}:{label}={value!r}"
                        )

                total = float(over) + float(under) + float(push)

                if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
                    offenders.append(f"{target}@{line}:sum={total!r}")

        if priced == 0:
            return CheckResult(
                name="prediction_smoke_test",
                passed=False,
                evidence="no frozen target had a loadable marginal to price",
                values={"targets": sorted(targets)},
                contract=description,
                error="the candidate priced nothing",
            )

        correlation = copula.correlation_for_player(None)

        matrix = np.asarray(correlation, dtype=float)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            offenders.append(f"copula correlation shape {matrix.shape!r}")

            eigenvalue = float("nan")

        else:
            if not np.all(np.isfinite(matrix)):
                offenders.append("copula correlation holds non-finite entries")

            diagonal_error = float(
                np.max(np.abs(np.diag(matrix) - 1.0)) if matrix.size else 1.0
            )

            if diagonal_error > CORRELATION_DIAGONAL_TOLERANCE:
                offenders.append(
                    f"copula correlation diagonal departs by {diagonal_error!r}"
                )

            eigenvalue = (
                float(np.min(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)))
                if np.all(np.isfinite(matrix))
                else float("nan")
            )

            if not _finite(eigenvalue) or eigenvalue < MINIMUM_EIGENVALUE:
                offenders.append(
                    f"copula correlation minimum eigenvalue {eigenvalue!r}"
                )

        return CheckResult(
            name="prediction_smoke_test",
            passed=not offenders,
            evidence=(
                f"priced {priced} over/under/push triple(s) across "
                f"{len(set(targets) & set(marginals))} target(s)"
            ),
            values={
                "minimum_eigenvalue": eigenvalue,
                "offenders": offenders[:20],
                "priced_triples": priced,
                "smoke_test_lines": list(SMOKE_TEST_LINES),
            },
            contract=description,
            error="; ".join(offenders[:5]) or None,
        )

    return _guarded("prediction_smoke_test", description, compute)


# --------------------------------------------------------------------------
# the two checks that need a live capture
# --------------------------------------------------------------------------


def _not_evaluable(name: str, contract: str, reason: str) -> CheckResult:
    return CheckResult(
        name=name,
        passed=None,
        evidence=reason,
        values={"status": NOT_EVALUABLE},
        contract=contract,
        error=None,
    )


def check_gate3_role_readiness(
    *,
    project_root: Path,
    candidate: Path,
    role_state_hash: Any,
    slate_date: Any,
    snapshot_dir: Path | None,
) -> CheckResult:
    """Prove the candidate satisfies the Gate 3 role/readiness contract.

    The contract is recovered from the repository, not invented here.
    ``docs/wizardofodds/ADAPTIVE_DAILY_TRAINING.md`` records that this check
    "needs a live captured lineup snapshot". ``prospective_snapshot`` defines
    what one is: the ``lineups`` component of a verified capture bundle. And
    ``gate3_v2.load_gate3_runtime`` defines a usable Gate 3 runtime --
    checksum-verified deployment artifacts whose ``gate3_policy`` equals the
    frozen per-prop policy.

    So readiness means three things together: a real lineup capture exists for
    the slate, the Gate 3 runtime the candidate was fit against still verifies,
    and the candidate recorded a role state derived from it.

    With no capture for the slate, the answer is NOT_EVALUABLE. Returning True
    without a capture would be asserting something about a capture that never
    happened, which is exactly what the frozen design forbids.
    """
    description = (
        "a verified capture bundle for the slate carries lineup records, the "
        "frozen Gate 3 deployment artifacts verify, and the candidate records "
        "a sha256 role-state hash"
    )

    def compute() -> CheckResult:
        from .gate3_v2 import load_gate3_runtime, resolve_gate3_snapshot_dir
        # See check_t20_protocol_compatible on why _capture_runs is used by
        # its private name: prospective_snapshot is hash-locked.
        from .prospective_snapshot import (
            PRIMARY_OFFSET_MINUTES,
            _capture_runs,
            select_capture_bundle,
        )

        resolved = (
            resolve_gate3_snapshot_dir(Path(snapshot_dir))
            if snapshot_dir is not None
            else None
        )

        target = str(slate_date)

        offset = int(PRIMARY_OFFSET_MINUTES)

        suffix = f":T-{offset}m"

        advertised: list[str] = []

        if resolved is not None and resolved.is_dir():
            try:
                runs = _capture_runs(resolved, target)
            except Exception:  # noqa: BLE001
                runs = []

            advertised = sorted(
                {
                    str(window)
                    for envelope in runs
                    if (envelope.get("payload") or {}).get("capture_reason")
                    == "scheduled"
                    for window in ((envelope.get("payload") or {}).get(
                        "window_ids"
                    ) or [])
                    if str(window).endswith(suffix)
                }
            )

        # The capture gate comes first on purpose. With no capture there is
        # nothing to be ready for, so the honest answer is "not measured" --
        # neither a pass nor a failure of the candidate.
        if not advertised:
            return _not_evaluable(
                "gate3_role_readiness",
                description,
                "no live captured lineup snapshot is available for "
                f"{target}, so Gate 3 role readiness cannot be measured; the "
                "frozen design defers rather than asserts",
            )

        lineup_records = 0

        refused: list[str] = []
        without_lineups: list[str] = []

        for window in advertised:
            game_id = int(window.split(":", 1)[0])

            try:
                bundle = select_capture_bundle(
                    resolved,
                    date=target,
                    game_id=game_id,
                    offset_minutes=offset,
                )
            except Exception as error:  # noqa: BLE001
                refused.append(f"{window}: {type(error).__name__}: {error}")
                continue

            lineups = bundle.records.get("lineups") or []

            if not lineups:
                without_lineups.append(window)

            lineup_records += len(lineups)

        runtime = load_gate3_runtime(
            Path(project_root) / "research" / "v2_gate3_deployment_artifacts"
        )

        installed = sorted(
            path.name
            for path in (Path(candidate) / "models").glob("*")
            if path.is_file() and "role" in path.name
        )

        hash_ok = _is_sha256(role_state_hash)

        reasons = []

        if refused:
            reasons.append(
                "captures that would not verify: " + "; ".join(refused[:5])
            )

        if without_lineups:
            reasons.append(
                "captures carrying no lineup records: "
                + ", ".join(without_lineups[:10])
            )

        if not hash_ok:
            reasons.append("role_state_hash is absent or malformed")

        return CheckResult(
            name="gate3_role_readiness",
            passed=not reasons,
            evidence=(
                f"Gate 3 runtime {runtime['candidate_id']} verified against "
                f"{len(advertised)} captured T-{offset}m window(s) for "
                f"{target} carrying {lineup_records} lineup record(s)"
            ),
            values={
                "advertised_windows": advertised[:20],
                "candidate_id": runtime["candidate_id"],
                "captures_without_lineups": without_lineups[:10],
                "deployment_manifest_sha256": runtime[
                    "deployment_manifest_sha256"
                ],
                "gate3_lock_commit": runtime["gate3_lock_commit"],
                "installed_role_artifacts": installed,
                "lineup_record_count": lineup_records,
                "refused_captures": refused[:10],
                "role_state_hash": str(role_state_hash),
                "role_state_seed_sha256": runtime["role_state_seed_sha256"],
                "slate_date": target,
            },
            contract=description,
            error="; ".join(reasons) or None,
        )

    return _guarded("gate3_role_readiness", description, compute)


def check_t20_protocol_compatible(
    *,
    slate_date: Any,
    snapshot_dir: Path | None,
) -> CheckResult:
    """Prove the slate's captures satisfy the live T-20 protocol.

    The contract is recovered from ``prospective_snapshot``, which owns the
    protocol. ``ProspectiveSnapshotClient`` -- the object the live prediction
    path reads its slate from -- derives the scheduled T-20 windows for a date
    and calls ``select_capture_bundle`` once per game in them. This check does
    the same two steps, so what it proves is what the live path needs:
    ``select_capture_bundle`` refuses a bundle whose capture carried errors,
    whose component hashes are incomplete, or whose component records do not
    re-hash to what the capture claimed.

    With no capture for the slate, the answer is NOT_EVALUABLE, for the same
    reason as ``gate3_role_readiness``: the documented contract is "needs a
    real T-20 capture", and there is no honest way to answer it without one.
    """
    description = (
        "every scheduled T-20 window the slate advertises resolves to a "
        "verified capture bundle at the frozen PRIMARY_OFFSET_MINUTES offset"
    )

    def compute() -> CheckResult:
        # _capture_runs is the reader ProspectiveSnapshotClient itself uses to
        # find the windows a slate advertises. It is reached for by its private
        # name deliberately: prospective_snapshot is a hash-locked serving
        # source under the frozen serving-source contract, so it cannot acquire
        # a public alias, and duplicating its on-disk layout here would be a
        # second definition of the protocol that could drift from the first.
        from .prospective_snapshot import (
            PRIMARY_OFFSET_MINUTES,
            _capture_runs,
            select_capture_bundle,
        )
        from .gate3_v2 import resolve_gate3_snapshot_dir

        resolved = (
            resolve_gate3_snapshot_dir(Path(snapshot_dir))
            if snapshot_dir is not None
            else None
        )

        if resolved is None or not resolved.is_dir():
            return _not_evaluable(
                "t20_protocol_compatible",
                description,
                "no snapshot directory is configured, so there is no T-20 "
                "capture to validate the output contract against",
            )

        target = str(slate_date)

        offset = int(PRIMARY_OFFSET_MINUTES)

        suffix = f":T-{offset}m"

        # Deciding evaluability before construction keeps the two outcomes
        # distinct: a slate that advertises no scheduled T-20 window has not
        # been captured, while a slate that advertises one and cannot produce a
        # verified bundle for it has failed the protocol.
        try:
            runs = _capture_runs(resolved, target)
        except Exception as error:  # noqa: BLE001
            return _not_evaluable(
                "t20_protocol_compatible",
                description,
                f"no readable capture run exists for {target}: "
                f"{type(error).__name__}: {error}",
            )

        advertised = sorted(
            {
                str(window)
                for envelope in runs
                if (envelope.get("payload") or {}).get("capture_reason")
                == "scheduled"
                for window in ((envelope.get("payload") or {}).get(
                    "window_ids"
                ) or [])
                if str(window).endswith(suffix)
            }
        )

        if not advertised:
            return _not_evaluable(
                "t20_protocol_compatible",
                description,
                f"no scheduled T-{offset}m capture window exists for {target}",
            )

        verified = 0

        refused: list[str] = []
        wrong_offset: list[str] = []

        for window in advertised:
            game_id = int(window.split(":", 1)[0])

            try:
                bundle = select_capture_bundle(
                    resolved,
                    date=target,
                    game_id=game_id,
                    offset_minutes=offset,
                )
            except Exception as error:  # noqa: BLE001
                refused.append(f"{window}: {type(error).__name__}: {error}")
                continue

            verified += 1

            if int(bundle.offset_minutes) != offset:
                wrong_offset.append(
                    f"{window}: T-{int(bundle.offset_minutes)}m"
                )

        passed = not refused and not wrong_offset

        reasons = []

        if refused:
            reasons.append(
                "advertised windows without a verified bundle: "
                + "; ".join(refused[:5])
            )

        if wrong_offset:
            reasons.append(
                "bundles outside the frozen offset: " + ", ".join(wrong_offset)
            )

        return CheckResult(
            name="t20_protocol_compatible",
            passed=passed,
            evidence=(
                f"verified {verified} of {len(advertised)} advertised "
                f"T-{offset}m capture window(s) for {target}"
            ),
            values={
                "advertised_windows": advertised[:20],
                "protocol_offset_minutes": offset,
                "refused_windows": refused[:10],
                "slate_date": target,
                "verified_bundles": verified,
                "wrong_offset": wrong_offset,
            },
            contract=description,
            error="; ".join(reasons) or None,
        )

    return _guarded("t20_protocol_compatible", description, compute)


# --------------------------------------------------------------------------
# the whole report
# --------------------------------------------------------------------------


def compute_validation_report(
    *,
    project_root: Path,
    candidate: Path,
    contract_object: ArchitectureContract,
    manifest: dict[str, Any],
    state: dict[str, Any],
    parent_manifest: dict[str, Any] | None,
    artifact_hashes: dict[str, str],
    finite: bool,
    finite_offenders: list[str],
    required_stages: tuple[str, ...],
    observed_stages: tuple[str, ...],
    required_prefixes: tuple[str, ...],
    required_files: tuple[str, ...],
    lineage_sources: dict[str, Path],
    targets: tuple[str, ...],
    frozen_marginal_family: str,
    calibration_routes: dict[str, str],
    calibration_hashes: dict[str, Any],
    calibration_fallbacks: dict[str, str],
    advanced_start_season: int,
    role_state_hash: Any,
    slate_date: Any,
    snapshot_dir: Path | None,
) -> ValidationReport:
    """Answer every required validation check for one candidate."""
    results = (
        check_training_completed(
            required_stages=required_stages,
            observed_stages=observed_stages,
        ),
        check_data_refresh_valid(
            state=state,
            plan_fingerprint=str(manifest.get("rolling_state_fingerprint", "")),
        ),
        check_history_regression_check(
            state=state,
            parent_manifest=parent_manifest,
        ),
        check_advanced_coverage_check(
            manifest=manifest,
            advanced_start_season=advanced_start_season,
        ),
        check_required_artifacts_present(
            artifact_hashes=artifact_hashes,
            required_prefixes=required_prefixes,
            required_files=required_files,
        ),
        check_artifact_hashes_valid(
            candidate=candidate,
            artifact_hashes=artifact_hashes,
        ),
        check_finite_values_check(finite=finite, offenders=finite_offenders),
        check_feature_schema_match(
            project_root=project_root,
            manifest=manifest,
        ),
        check_architecture_contract_match(
            project_root=project_root,
            contract_object=contract_object,
            manifest=manifest,
        ),
        check_source_lineage_match(
            project_root=project_root,
            candidate=candidate,
            manifest=manifest,
            lineage_sources=lineage_sources,
        ),
        check_marginal_fit_valid(
            candidate=candidate,
            targets=targets,
            frozen_family=frozen_marginal_family,
        ),
        check_calibration_valid(
            calibration_hashes=calibration_hashes,
            calibration_fallbacks=calibration_fallbacks,
            calibration_routes=calibration_routes,
        ),
        check_prediction_smoke_test(candidate=candidate, targets=targets),
        check_gate3_role_readiness(
            project_root=project_root,
            candidate=candidate,
            role_state_hash=role_state_hash,
            slate_date=slate_date,
            snapshot_dir=snapshot_dir,
        ),
        check_t20_protocol_compatible(
            slate_date=slate_date,
            snapshot_dir=snapshot_dir,
        ),
    )

    report = ValidationReport(results=results)

    assert_every_required_check_is_answered(report)

    return report


def assert_every_required_check_is_answered(
    report: ValidationReport,
) -> None:
    """Refuse a report that silently omits a name the registry requires."""
    answered = {result.name for result in report.results}

    missing = sorted(set(REQUIRED_VALIDATION_CHECKS) - answered)

    if missing:
        raise ValidationComputationError(
            "the validation report does not answer every required check: "
            + ", ".join(missing)
        )

    unknown = sorted(answered - set(REQUIRED_VALIDATION_CHECKS))

    if unknown:
        raise ValidationComputationError(
            "the validation report answers checks the registry does not "
            "require: " + ", ".join(unknown)
        )
