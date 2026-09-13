"""Tests for the immutable adaptive fit registry and fail-closed promotion.

Nothing here trains, refits or recalibrates a model, and nothing touches the
network. Every registry lives in a pytest temporary directory; the repository
is only ever read from.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nba_prop_quant.adaptive_fit_registry import (
    ARCHITECTURE_REFERENCE_SHA,
    CONTRACT_RELATIVE_PATH,
    FIT_ID_PREFIX,
    REGISTRY_ENV_VAR,
    REQUIRED_VALIDATION_CHECKS,
    ArchitectureContractViolation,
    FitAlreadyExists,
    FitMetadata,
    FitNotFound,
    FitRegistry,
    IntegrityError,
    PromotionLocked,
    PromotionRefused,
    RegistryRootError,
    StagedTreeError,
    ValidationRefused,
    canonical_json,
    derive_fit_id,
    feature_schema_hash,
    is_fit_id,
    load_architecture_contract,
    parse_checksum_inventory,
    resolve_registry_root,
    routing_maps,
)


REPO = Path(__file__).resolve().parents[1]

CLI = REPO / "ops" / "adaptive_fit_registry.py"


# Files the registry reads when it re-derives the frozen choices.
PROJECT_FILES = (
    "configs/model.yaml",
    "models/combo_dependence_policy.json",
    "models/marginal_selection.json",
    "models/market_probability_calibration_policy.json",
    "models/mean_model_selection.json",
    "research/v2_gate3_deployment_artifacts/deployment_manifest.json",
    "src/nba_prop_quant/features.py",
    str(CONTRACT_RELATIVE_PATH.as_posix()),
)


# --------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture
def project_root() -> Path:
    """The real repository, read-only."""
    return REPO


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    """A throwaway copy of just the files the registry derives from.

    Contract and schema drift can then be simulated without ever writing
    inside the repository.
    """
    root = tmp_path / "project"

    for relative in PROJECT_FILES:
        source = REPO / relative
        destination = root / relative

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    return root


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    return make_staged(tmp_path / "staged")


def make_staged(
    path: Path, marker: bytes = b"booster-day-1"
) -> Path:
    (path / "models").mkdir(parents=True, exist_ok=True)

    (path / "models" / "target_pts.joblib").write_bytes(
        marker
    )

    (path / "models" / "marginals.joblib").write_bytes(
        b"zinb-parameters"
    )

    (path / "calibration.json").write_text(
        json.dumps({"points": {"slope": 0.41}}),
        encoding="utf-8",
    )

    return path


def make_metadata(**overrides) -> FitMetadata:
    payload = {
        "fit_date": "2026-11-15",
        "training_cutoff": "2026-11-14",
        "source_commit_sha": (
            "b7bb2e41a9efa296e776c619ed286a7f42f5e44d"
        ),
        "training_data_manifest_hash": "a" * 64,
        "python_version": "3.14.5",
        "package_versions": {
            "pandas": "3.0.5",
            "scikit-learn": "1.9.0",
            "xgboost": "3.4.1",
        },
        "core_seed": 73,
        "gate3_seed": 20260830,
        "effective_n_jobs": 2,
        "gate3_candidate_policy_id": (
            "nba_prop_quant_v2_gate3_dd2d394b6def"
        ),
    }

    payload.update(overrides)

    return FitMetadata(**payload)


def make_registry(
    tmp_path: Path,
    project_root: Path,
    name: str = "registry",
) -> FitRegistry:
    return FitRegistry(
        root=tmp_path / name, project_root=project_root
    )


def passing_checks() -> dict[str, bool]:
    return {
        name: True for name in REQUIRED_VALIDATION_CHECKS
    }


def register_and_validate(
    registry: FitRegistry,
    staged_dir: Path,
    metadata: FitMetadata,
    checks: dict[str, bool] | None = None,
) -> str:
    fit_id = registry.register(staged_dir, metadata)

    registry.record_validation(
        fit_id,
        passing_checks() if checks is None else checks,
        actor="pytest",
    )

    return fit_id


def read_contract(root: Path) -> dict:
    return json.loads(
        (root / CONTRACT_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )


def write_contract(root: Path, payload: dict) -> None:
    (root / CONTRACT_RELATIVE_PATH).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# architecture contract
# --------------------------------------------------------------------------


def test_architecture_contract_loads(project_root: Path):
    contract = load_architecture_contract(project_root)

    assert (
        contract.architecture_reference_sha
        == ARCHITECTURE_REFERENCE_SHA
    )

    assert len(contract.sha256) == 64

    frozen = contract.frozen_choices

    assert frozen["core_seed"] == 73
    assert frozen["gate3_seed"] == 20260830
    assert frozen["effective_n_jobs"] == 2


def test_contract_matches_the_working_tree(
    project_root: Path,
):
    """The committed contract must be re-derivable from the repository."""
    contract = load_architecture_contract(project_root)

    observed = routing_maps(project_root)

    assert (
        observed["mean_model_routing"]
        == contract.frozen_choices["mean_model_routing"]
    )

    assert (
        observed["gate3_routing"]
        == contract.frozen_choices["gate3_routing"]
    )

    assert contract.frozen_choices[
        "feature_schema_hash"
    ] == feature_schema_hash(project_root)


def test_contract_pins_the_audited_routing(
    project_root: Path,
):
    frozen = load_architecture_contract(
        project_root
    ).frozen_choices

    assert frozen["mean_model_routing"] == {
        "ast": "xgb",
        "blk": "ensemble",
        "fg3m": "xgb",
        "pts": "xgb",
        "reb": "ensemble",
        "stl": "xgb",
    }

    assert set(
        frozen["marginal_family_routing"].values()
    ) == {"zinb"}

    assert (
        frozen["gate3_routing"]["assists"]
        == "v2_role_shock_calibrated"
    )

    assert (
        frozen["dependence_production_lambda"][
            "rebounds_assists"
        ]
        == 0.85
    )

    assert (
        frozen["calibration_family_routing"]["blocks"]
        == "raw"
    )


def test_contract_does_not_freeze_daily_fitted_values(
    project_root: Path,
):
    """Booster, ZINB, Platt and role weights must not be pinned here."""
    payload = json.loads(
        (project_root / CONTRACT_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )

    banned = {
        "artifact_hashes",
        "booster_hashes",
        "calibration_coefficients",
        "production_parameters",
        "production_weights",
        "role_model_weights",
    }

    def walk(node, trail: str):
        if isinstance(node, dict):
            for key, value in node.items():
                assert (
                    key not in banned
                ), f"{trail}{key} is a daily fitted value"

                walk(value, f"{trail}{key}.")

        elif isinstance(node, list):
            for item in node:
                walk(item, trail)

    walk(payload["frozen_choices"], "frozen_choices.")
    walk(
        payload["frozen_routing_digests"],
        "frozen_routing_digests.",
    )


def test_wrong_architecture_reference_in_contract_rejected(
    fake_project: Path,
):
    payload = read_contract(fake_project)

    payload["architecture_reference_sha"] = "0" * 40

    write_contract(fake_project, payload)

    with pytest.raises(ArchitectureContractViolation):
        load_architecture_contract(fake_project)


def test_wrong_architecture_reference_in_metadata_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    metadata = make_metadata(
        architecture_reference_sha="0" * 40
    )

    with pytest.raises(ArchitectureContractViolation):
        registry.register(staged, metadata)


def test_wrong_contract_hash_rejected_at_verify(
    tmp_path: Path, fake_project: Path, staged: Path
):
    """A fit registered under one contract is unusable under another."""
    registry = make_registry(tmp_path, fake_project)

    fit_id = registry.register(staged, make_metadata())

    payload = read_contract(fake_project)

    payload["purpose"] = "mutated after registration"

    write_contract(fake_project, payload)

    with pytest.raises(ArchitectureContractViolation):
        registry.verify(fit_id)


def test_feature_schema_drift_rejected(
    tmp_path: Path, fake_project: Path, staged: Path
):
    features = (
        fake_project / "src" / "nba_prop_quant" / "features.py"
    )

    text = features.read_text(encoding="utf-8")

    text = text.replace(
        '"is_home",', '"is_home", "invented_feature",', 1
    )

    features.write_text(text, encoding="utf-8")

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation, match="feature schema"
    ):
        registry.register(staged, make_metadata())


def test_hyperparameter_drift_rejected(
    tmp_path: Path, fake_project: Path, staged: Path
):
    config = fake_project / "configs" / "model.yaml"

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "max_depth: 5", "max_depth: 7", 1
        ),
        encoding="utf-8",
    )

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation, match="config hash"
    ):
        registry.register(staged, make_metadata())


def test_model_selection_drift_rejected(
    tmp_path: Path, fake_project: Path, staged: Path
):
    """Re-running selection nightly must be refused, not absorbed."""
    path = (
        fake_project / "models" / "mean_model_selection.json"
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    payload["targets"]["pts"]["selected_mode"] = "ensemble"

    path.write_text(
        json.dumps(payload), encoding="utf-8"
    )

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation,
        match="mean_model_routing",
    ):
        registry.register(staged, make_metadata())


def test_seed_and_n_jobs_mismatch_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    for override in (
        {"core_seed": 7},
        {"gate3_seed": 1},
        {"effective_n_jobs": -1},
    ):
        with pytest.raises(
            ArchitectureContractViolation
        ):
            registry.register(
                staged, make_metadata(**override)
            )


def test_gate3_candidate_policy_mismatch_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(ArchitectureContractViolation):
        registry.register(
            staged,
            make_metadata(
                gate3_candidate_policy_id="nba_prop_quant_v2_gate3_dead"
            ),
        )


# --------------------------------------------------------------------------
# fit identity
# --------------------------------------------------------------------------


def test_fit_id_namespace_is_distinct(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    assert fit_id.startswith(FIT_ID_PREFIX)
    assert is_fit_id(fit_id)

    # Must not collide with the model freeze or bundle freeze namespaces.
    assert not fit_id.startswith("nba_prop_quant_2")
    assert "gate3" not in fit_id

    manifest = registry.load_manifest(fit_id)

    assert "freeze_id" not in manifest


def test_fit_identity_is_deterministic(
    tmp_path: Path, project_root: Path, staged: Path
):
    first = make_registry(tmp_path, project_root, "a")
    second = make_registry(tmp_path, project_root, "b")

    metadata = make_metadata()

    assert first.register(
        staged, metadata
    ) == second.register(staged, metadata)


def test_identity_excludes_promotion_state(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    before = registry.load_manifest(fit_id)[
        "identity_inputs"
    ]

    registry.promote(fit_id, reason="unit test")

    after = registry.load_manifest(fit_id)[
        "identity_inputs"
    ]

    assert before == after
    assert derive_fit_id(after) == fit_id

    for key in (
        "current_good_fit_id",
        "promoted_at",
        "promotion_reason",
    ):
        assert key not in after


def test_different_artifact_bytes_change_fit_id(
    tmp_path: Path, project_root: Path
):
    first_dir = make_staged(
        tmp_path / "one", marker=b"booster-day-1"
    )

    second_dir = make_staged(
        tmp_path / "two", marker=b"booster-day-2"
    )

    metadata = make_metadata()

    first = make_registry(
        tmp_path, project_root, "reg_a"
    ).register(first_dir, metadata)

    second = make_registry(
        tmp_path, project_root, "reg_b"
    ).register(second_dir, metadata)

    assert first != second


def test_different_training_cutoff_changes_fit_id(
    tmp_path: Path, project_root: Path, staged: Path
):
    first = make_registry(
        tmp_path, project_root, "reg_a"
    ).register(staged, make_metadata())

    second = make_registry(
        tmp_path, project_root, "reg_b"
    ).register(
        staged,
        make_metadata(training_cutoff="2026-11-13"),
    )

    assert first != second


def test_training_cutoff_must_precede_fit_date(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        ArchitectureContractViolation,
        match="strictly before",
    ):
        registry.register(
            staged,
            make_metadata(training_cutoff="2026-11-15"),
        )


# --------------------------------------------------------------------------
# registration and immutability
# --------------------------------------------------------------------------


def test_registration_succeeds_once(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    fit_dir = registry.fit_dir(fit_id)

    assert (fit_dir / "manifest.json").is_file()
    assert (fit_dir / "SHA256SUMS").is_file()
    assert (
        fit_dir / "artifacts" / "models" / "target_pts.joblib"
    ).is_file()

    assert registry.list_fits() == [fit_id]


def test_duplicate_registration_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    metadata = make_metadata()

    fit_id = registry.register(staged, metadata)

    before = (
        registry.fit_dir(fit_id) / "manifest.json"
    ).read_bytes()

    with pytest.raises(FitAlreadyExists):
        registry.register(staged, metadata)

    after = (
        registry.fit_dir(fit_id) / "manifest.json"
    ).read_bytes()

    assert before == after


def test_no_staging_residue_after_duplicate(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    metadata = make_metadata()

    registry.register(staged, metadata)

    with pytest.raises(FitAlreadyExists):
        registry.register(staged, metadata)

    assert list(registry.staging_dir.iterdir()) == []


def test_symlink_in_staged_tree_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")

    (staged / "link.joblib").symlink_to(target)

    registry = make_registry(tmp_path, project_root)

    with pytest.raises(StagedTreeError, match="symlink"):
        registry.register(staged, make_metadata())

    assert registry.list_fits() == []


def test_fifo_in_staged_tree_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    os.mkfifo(staged / "pipe")

    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        StagedTreeError, match="non-regular"
    ):
        registry.register(staged, make_metadata())


def test_empty_staged_tree_rejected(
    tmp_path: Path, project_root: Path
):
    empty = tmp_path / "empty"
    empty.mkdir()

    registry = make_registry(tmp_path, project_root)

    with pytest.raises(StagedTreeError, match="empty"):
        registry.register(empty, make_metadata())


def test_checksum_inventory_created(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    text = (
        registry.fit_dir(fit_id) / "SHA256SUMS"
    ).read_text(encoding="utf-8")

    inventory = parse_checksum_inventory(text)

    assert "manifest.json" in inventory

    assert (
        "artifacts/models/target_pts.joblib" in inventory
    )

    for line in text.splitlines():
        digest, _, relative = line.partition("  ")

        assert len(digest) == 64
        assert relative


def test_registered_bytes_are_preserved(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    stored = (
        registry.fit_dir(fit_id)
        / "artifacts"
        / "models"
        / "target_pts.joblib"
    )

    assert (
        stored.read_bytes()
        == (
            staged / "models" / "target_pts.joblib"
        ).read_bytes()
    )


def test_artifact_tampering_detected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    (
        registry.fit_dir(fit_id)
        / "artifacts"
        / "models"
        / "target_pts.joblib"
    ).write_bytes(b"tampered")

    with pytest.raises(
        IntegrityError, match="artifact tampering"
    ):
        registry.verify(fit_id)


def test_extra_artifact_detected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    (
        registry.fit_dir(fit_id)
        / "artifacts"
        / "smuggled.joblib"
    ).write_bytes(b"extra")

    with pytest.raises(
        IntegrityError, match="unrecorded"
    ):
        registry.verify(fit_id)


def test_missing_artifact_detected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    (
        registry.fit_dir(fit_id)
        / "artifacts"
        / "calibration.json"
    ).unlink()

    with pytest.raises(IntegrityError, match="missing"):
        registry.verify(fit_id)


def test_manifest_tampering_detected(
    tmp_path: Path, project_root: Path, staged: Path
):
    """Editing identity inputs re-derives a different fit_id."""
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    manifest_path = (
        registry.fit_dir(fit_id) / "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    manifest["identity_inputs"]["source_commit_sha"] = (
        "f" * 40
    )

    manifest_path.write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(
        IntegrityError, match="manifest tampering"
    ):
        registry.verify(fit_id)


def test_manifest_relabelling_detected(
    tmp_path: Path, project_root: Path, staged: Path
):
    """Rewriting only the recorded hash still fails the inventory."""
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    manifest_path = (
        registry.fit_dir(fit_id) / "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    manifest["notes"] = "quietly edited"

    manifest_path.write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(IntegrityError):
        registry.verify(fit_id)


def test_manifest_paths_are_relative(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    manifest = registry.load_manifest(fit_id)

    for relative in manifest["artifact_hashes"]:
        assert not relative.startswith("/")

    assert manifest["artifact_root"] == "artifacts"

    text = canonical_json(manifest)

    assert str(tmp_path) not in text
    assert str(project_root) not in text


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------


def test_secret_shaped_metadata_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    metadata = make_metadata(
        package_versions={"bdl_api_key": "abc123"}
    )

    with pytest.raises(
        ArchitectureContractViolation,
        match="credential-shaped",
    ):
        registry.register(staged, metadata)


def test_secret_shaped_value_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        ArchitectureContractViolation,
        match="credential-shaped",
    ):
        registry.register(
            staged,
            make_metadata(notes="BDL_API_KEY=secretvalue"),
        )


def test_absolute_path_in_metadata_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        ArchitectureContractViolation,
        match="absolute path",
    ):
        registry.register(
            staged,
            make_metadata(
                notes="/srv/wizardofodds/runtime/bundle"
            ),
        )


def test_no_secret_fields_written(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    registry.promote(fit_id, reason="unit test")

    for path in sorted(registry.root.rglob("*")):
        if not path.is_file() or path.suffix == ".lock":
            continue

        text = path.read_text(
            encoding="utf-8", errors="ignore"
        ).lower()

        for banned in (
            "api_key",
            "apikey",
            "bdl_api_key",
            "password",
            "authorization",
            "bearer ",
        ):
            assert banned not in text


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_registration_does_not_promote(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    assert (
        registry.current()["current_good_fit_id"] is None
    )

    assert registry.load_validation(fit_id) is None

    assert not registry.promotion_state_path.exists()


def test_promotion_without_validation_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    with pytest.raises(
        ValidationRefused, match="no validation record"
    ):
        registry.promote(fit_id, reason="unit test")

    assert (
        registry.current()["current_good_fit_id"] is None
    )


def test_promotion_with_missing_check_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    checks = passing_checks()

    del checks["calibration_valid"]

    fit_id = register_and_validate(
        registry, staged, make_metadata(), checks=checks
    )

    with pytest.raises(
        ValidationRefused, match="calibration_valid"
    ):
        registry.promote(fit_id, reason="unit test")

    assert (
        registry.current()["current_good_fit_id"] is None
    )


def test_promotion_with_failed_check_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    checks = passing_checks()

    checks["gate3_role_readiness"] = False

    fit_id = register_and_validate(
        registry, staged, make_metadata(), checks=checks
    )

    with pytest.raises(
        ValidationRefused, match="gate3_role_readiness"
    ):
        registry.promote(fit_id, reason="unit test")

    assert (
        registry.current()["current_good_fit_id"] is None
    )


def test_validation_rejects_unknown_check(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    with pytest.raises(
        ValidationRefused, match="unknown validation"
    ):
        registry.record_validation(
            fit_id, {"beats_the_market": True}
        )


def test_validation_for_unregistered_fit_rejected(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)
    registry.initialise()

    with pytest.raises(FitNotFound):
        registry.record_validation(
            f"{FIT_ID_PREFIX}20261115_" + "0" * 16,
            passing_checks(),
        )


def test_validation_record_lives_outside_the_fit(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    assert registry.validation_path(fit_id).is_file()

    fit_files = {
        path.relative_to(registry.fit_dir(fit_id)).as_posix()
        for path in registry.fit_dir(fit_id).rglob("*")
        if path.is_file()
    }

    assert not any(
        "validation" in name for name in fit_files
    )


def test_no_superiority_criterion_in_required_checks():
    """Promotion is not model selection."""
    joined = " ".join(REQUIRED_VALIDATION_CHECKS)

    for banned in (
        "superiority",
        "beats",
        "better_than",
        "selection",
        "outperform",
    ):
        assert banned not in joined


# --------------------------------------------------------------------------
# promotion, last-good and rollback
# --------------------------------------------------------------------------


def test_valid_promotion_sets_current_good(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    state = registry.promote(
        fit_id, reason="nightly", actor="orchestrator"
    )

    assert state["current_good_fit_id"] == fit_id
    assert state["previous_good_fit_id"] is None

    current = registry.current()

    assert current["current_good_fit_id"] == fit_id
    assert current["promotion_reason"] == "nightly"
    assert current["promotion_actor"] == "orchestrator"


def test_second_promotion_moves_previous_good(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(first, reason="day one")

    second = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    state = registry.promote(second, reason="day two")

    assert state["current_good_fit_id"] == second
    assert state["previous_good_fit_id"] == first


def test_repromoting_current_fit_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    registry.promote(fit_id, reason="first")

    with pytest.raises(
        PromotionRefused, match="already the current"
    ):
        registry.promote(fit_id, reason="again")

    assert (
        registry.current()["previous_good_fit_id"] is None
    )


def test_failed_second_candidate_preserves_current_good(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    good = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(good, reason="day one")

    checks = passing_checks()
    checks["marginal_fit_valid"] = False

    bad = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
        checks=checks,
    )

    with pytest.raises(ValidationRefused):
        registry.promote(bad, reason="day two")

    assert (
        registry.current()["current_good_fit_id"] == good
    )

    # The failed fit is evidence and is never deleted.
    assert registry.fit_dir(bad).is_dir()


def test_corrupt_candidate_preserves_current_good(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    good = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(good, reason="day one")

    bad = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    (
        registry.fit_dir(bad)
        / "artifacts"
        / "calibration.json"
    ).write_text("corrupted", encoding="utf-8")

    with pytest.raises(IntegrityError):
        registry.promote(bad, reason="day two")

    assert (
        registry.current()["current_good_fit_id"] == good
    )


def test_rollback_restores_previous_good(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(first, reason="day one")

    second = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    registry.promote(second, reason="day two")

    state = registry.rollback(reason="regression found")

    assert state["current_good_fit_id"] == first
    assert state["previous_good_fit_id"] == second

    # Neither fit is deleted by a rollback.
    assert registry.fit_dir(first).is_dir()
    assert registry.fit_dir(second).is_dir()


def test_rollback_without_previous_good_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    registry.promote(fit_id, reason="first")

    with pytest.raises(
        PromotionRefused, match="no previous good"
    ):
        registry.rollback(reason="nothing to do")

    assert (
        registry.current()["current_good_fit_id"] == fit_id
    )


def test_rollback_to_corrupt_fit_rejected(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(first, reason="day one")

    second = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    registry.promote(second, reason="day two")

    (
        registry.fit_dir(first)
        / "artifacts"
        / "calibration.json"
    ).write_text("corrupted", encoding="utf-8")

    with pytest.raises(IntegrityError):
        registry.rollback(reason="attempted")

    assert (
        registry.current()["current_good_fit_id"] == second
    )


def test_rollback_to_missing_fit_rejected(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(first, reason="day one")

    second = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    registry.promote(second, reason="day two")

    shutil.rmtree(registry.fit_dir(first))

    with pytest.raises(FitNotFound):
        registry.rollback(reason="attempted")

    assert (
        registry.current()["current_good_fit_id"] == second
    )


def test_promotion_history_is_recorded(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(first, reason="day one")

    second = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    registry.promote(second, reason="day two")
    registry.rollback(reason="regression")

    history = registry.load_promotion_state()["history"]

    assert [entry["action"] for entry in history] == [
        "promote",
        "promote",
        "rollback",
    ]


# --------------------------------------------------------------------------
# atomicity and locking
# --------------------------------------------------------------------------


def test_promotion_state_update_is_atomic(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A failure at the replace step must leave current-good untouched."""
    registry = make_registry(tmp_path, project_root)

    first = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(first, reason="day one")

    before = registry.promotion_state_path.read_bytes()

    second = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    real_replace = os.replace

    def exploding_replace(src, dst, *args, **kwargs):
        if str(dst).endswith("promotion_state.json"):
            raise OSError("simulated crash before replace")

        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", exploding_replace)

    with pytest.raises(OSError):
        registry.promote(second, reason="day two")

    monkeypatch.undo()

    assert (
        registry.promotion_state_path.read_bytes() == before
    )

    assert (
        registry.current()["current_good_fit_id"] == first
    )

    leftovers = [
        path.name
        for path in registry.state_dir.iterdir()
        if path.suffix == ".tmp"
    ]

    assert leftovers == []


def test_promotion_lock_prevents_conflicting_promotion(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    registry.locks_dir.mkdir(parents=True, exist_ok=True)

    handle = os.open(
        registry.promotion_lock_path,
        os.O_CREAT | os.O_RDWR,
        0o644,
    )

    try:
        fcntl.flock(handle, fcntl.LOCK_EX)

        with pytest.raises(PromotionLocked):
            registry.promote(
                fit_id, reason="blocked", blocking=False
            )
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)

    assert (
        registry.current()["current_good_fit_id"] is None
    )

    # The lock is released, so promotion now succeeds.
    registry.promote(fit_id, reason="after unlock")

    assert (
        registry.current()["current_good_fit_id"] == fit_id
    )


def test_rollback_is_lock_protected(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    registry.promote(first, reason="day one")

    second = register_and_validate(
        registry,
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    registry.promote(second, reason="day two")

    handle = os.open(
        registry.promotion_lock_path,
        os.O_CREAT | os.O_RDWR,
        0o644,
    )

    try:
        fcntl.flock(handle, fcntl.LOCK_EX)

        with pytest.raises(PromotionLocked):
            registry.rollback(
                reason="blocked", blocking=False
            )
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)

    assert (
        registry.current()["current_good_fit_id"] == second
    )


# --------------------------------------------------------------------------
# registry root safety
# --------------------------------------------------------------------------


def test_registry_root_required(tmp_path: Path):
    with pytest.raises(
        RegistryRootError, match="no registry root"
    ):
        resolve_registry_root(None, environ={})


def test_registry_root_from_environment(tmp_path: Path):
    root = resolve_registry_root(
        None,
        environ={REGISTRY_ENV_VAR: str(tmp_path / "r")},
    )

    assert root == (tmp_path / "r").resolve()


def test_registry_root_inside_repository_rejected(
    project_root: Path,
):
    for candidate in (
        project_root / "registry",
        project_root / "models" / "fits",
        project_root,
    ):
        with pytest.raises(
            RegistryRootError, match="inside the repository"
        ):
            resolve_registry_root(
                candidate, project_root=project_root
            )


def test_no_registry_artifacts_written_into_repository(
    project_root: Path,
):
    for name in (
        "fits",
        "registry",
        "staging",
        "locks",
    ):
        assert not (project_root / name).exists()

    assert not (
        project_root / "state" / "promotion_state.json"
    ).exists()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *argv],
        capture_output=True,
        text=True,
    )


def test_cli_requires_registry_root():
    result = run_cli("current")

    assert result.returncode != 0

    assert (
        "no registry root supplied" in result.stderr
    )


def test_cli_refuses_registry_inside_repository():
    result = run_cli(
        "current",
        "--registry-root",
        str(REPO / "registry"),
    )

    assert result.returncode != 0

    assert "inside the repository" in result.stderr


def test_cli_register_verify_promote_rollback(
    tmp_path: Path, project_root: Path
):
    root = tmp_path / "registry"

    metadata_path = tmp_path / "metadata.json"

    checks_path = tmp_path / "checks.json"

    checks_path.write_text(
        json.dumps(passing_checks()), encoding="utf-8"
    )

    fit_ids = []

    for day, marker in (
        ("2026-11-15", b"day-1"),
        ("2026-11-16", b"day-2"),
    ):
        staged_dir = make_staged(
            tmp_path / f"staged_{day}", marker
        )

        cutoff = (
            "2026-11-14"
            if day == "2026-11-15"
            else "2026-11-15"
        )

        metadata_path.write_text(
            json.dumps(
                {
                    "core_seed": 73,
                    "effective_n_jobs": 2,
                    "fit_date": day,
                    "gate3_candidate_policy_id": (
                        "nba_prop_quant_v2_gate3_dd2d394b6def"
                    ),
                    "gate3_seed": 20260830,
                    "package_versions": {
                        "xgboost": "3.4.1"
                    },
                    "python_version": "3.14.5",
                    "source_commit_sha": "b" * 40,
                    "training_cutoff": cutoff,
                    "training_data_manifest_hash": "a" * 64,
                }
            ),
            encoding="utf-8",
        )

        registered = run_cli(
            "register",
            "--registry-root",
            str(root),
            "--staged-dir",
            str(staged_dir),
            "--metadata",
            str(metadata_path),
        )

        assert registered.returncode == 0, registered.stderr

        payload = json.loads(registered.stdout)

        assert payload["promoted"] is False
        assert (
            payload["status"] == "registered_not_promoted"
        )

        fit_id = payload["fit_id"]
        fit_ids.append(fit_id)

        verified = run_cli(
            "verify",
            "--registry-root",
            str(root),
            "--fit-id",
            fit_id,
        )

        assert verified.returncode == 0

        assert (
            json.loads(verified.stdout)["integrity"] == "ok"
        )

        recorded = run_cli(
            "record-validation",
            "--registry-root",
            str(root),
            "--fit-id",
            fit_id,
            "--checks",
            str(checks_path),
        )

        assert recorded.returncode == 0

        assert json.loads(recorded.stdout)["passed"] is True

        promoted = run_cli(
            "promote",
            "--registry-root",
            str(root),
            "--fit-id",
            fit_id,
            "--reason",
            "cli test",
        )

        assert promoted.returncode == 0

    current = run_cli(
        "current", "--registry-root", str(root)
    )

    assert current.returncode == 0

    state = json.loads(current.stdout)

    assert state["current_good_fit_id"] == fit_ids[1]
    assert state["previous_good_fit_id"] == fit_ids[0]

    rolled = run_cli(
        "rollback",
        "--registry-root",
        str(root),
        "--reason",
        "cli rollback",
    )

    assert rolled.returncode == 0

    assert (
        json.loads(rolled.stdout)["current_good_fit_id"]
        == fit_ids[0]
    )


def test_cli_promote_without_validation_is_non_zero(
    tmp_path: Path, project_root: Path
):
    root = tmp_path / "registry"

    registry = FitRegistry(
        root=root, project_root=project_root
    )

    fit_id = registry.register(
        make_staged(tmp_path / "staged"), make_metadata()
    )

    result = run_cli(
        "promote",
        "--registry-root",
        str(root),
        "--fit-id",
        fit_id,
        "--reason",
        "should fail",
    )

    assert result.returncode != 0

    assert (
        json.loads(result.stderr)["error"]
        == "ValidationRefused"
    )

    assert (
        registry.current()["current_good_fit_id"] is None
    )
