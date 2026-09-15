"""Tests for the immutable adaptive fit registry and fail-closed promotion.

No test trains, refits or recalibrates a model, and nothing here touches the
network. Every registry lives in a pytest temporary directory; the repository
itself is only ever read from.
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
    CONFIG_OPERATIONAL_FIELDS,
    CONFIG_RELATIVE_PATH,
    CONTRACT_RELATIVE_PATH,
    FEATURES_RELATIVE_PATH,
    FIT_ID_PREFIX,
    FROZEN_POLICY_SOURCES,
    PROHIBITED_TRAINING_END_KEYS,
    REGISTRY_ENV_VAR,
    REQUIRED_VALIDATION_CHECKS,
    TRAINING_CUTOFF_RELATION,
    TRAINING_END_POLICY,
    TRAINING_WINDOW_POLICY_KEY,
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
    _find_prohibited_training_end,
    canonical_json,
    config_hash,
    config_projection,
    derive_fit_id,
    feature_schema_hash,
    frozen_policy_digests,
    frozen_routing,
    is_fit_id,
    load_architecture_contract,
    parse_checksums,
    prune_paths,
    resolve_registry_root,
    training_window_boundaries,
)


REPO = Path(__file__).resolve().parents[1]

CLI = REPO / "ops" / "adaptive_fit_registry.py"


# Everything the registry re-derives the frozen choices from.
DERIVATION_SOURCES = tuple(
    sorted(
        {
            CONFIG_RELATIVE_PATH.as_posix(),
            FEATURES_RELATIVE_PATH.as_posix(),
            CONTRACT_RELATIVE_PATH.as_posix(),
            *(
                source["path"].as_posix()
                for source in FROZEN_POLICY_SOURCES.values()
            ),
        }
    )
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
    """A throwaway copy of only the files the registry derives from.

    Contract and architecture drift can then be simulated without ever
    writing inside the repository.
    """
    root = tmp_path / "project"

    for relative in DERIVATION_SOURCES:
        destination = root / relative

        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        shutil.copyfile(REPO / relative, destination)

    return root


def make_staged(
    path: Path, marker: bytes = b"fitted-booster-day-1"
) -> Path:
    (path / "models").mkdir(parents=True, exist_ok=True)

    (path / "models" / "target_pts.joblib").write_bytes(
        marker
    )

    (path / "models" / "marginals.joblib").write_bytes(
        b"zinb-fitted-parameters"
    )

    (path / "calibration.json").write_text(
        json.dumps({"points": {"slope": 0.41}}),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    return make_staged(tmp_path / "staged")


def metadata_payload(**overrides) -> dict:
    payload = {
        "fit_date": "2026-11-15",
        "training_cutoff": "2026-11-14",
        "source_commit_sha": (
            "b7bb2e41a9efa296e776c619ed286a7f42f5e44d"
        ),
        "training_data_manifest_hash": "a" * 64,
        "python_version": "3.12.3",
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

    return payload


def make_metadata(**overrides) -> FitMetadata:
    return FitMetadata(**metadata_payload(**overrides))


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


def promote_two_fits(
    registry: FitRegistry, tmp_path: Path
) -> tuple[str, str]:
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

    return first, second


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2)
        + "\n",
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

    assert (
        frozen["primary_certification_capture"][
            "offset_minutes_before_tip"
        ]
        == 20
    )


def test_contract_is_rederivable_from_the_tree(
    project_root: Path,
):
    """The committed contract must match what registration recomputes."""
    contract = load_architecture_contract(project_root)

    assert (
        contract.frozen_choices["routing"]
        == frozen_routing(project_root)
    )

    assert (
        contract.policy_digests
        == frozen_policy_digests(project_root)
    )

    assert contract.frozen_choices[
        "feature_schema_hash"
    ] == feature_schema_hash(project_root)


def test_contract_pins_the_production_routing(
    project_root: Path,
):
    routing = load_architecture_contract(
        project_root
    ).frozen_choices["routing"]

    assert routing["mean_model_routing"] == {
        "ast": "xgb",
        "blk": "ensemble",
        "fg3m": "xgb",
        "pts": "xgb",
        "reb": "ensemble",
        "stl": "xgb",
    }

    assert set(
        routing["marginal_family_routing"].values()
    ) == {"zinb"}

    assert routing["gate3_routing"] == {
        "assists": "v2_role_shock_calibrated",
        "blocks": "frozen_selected_v1",
        "points": "frozen_selected_v1",
        "points_assists": "v2_role_increment",
        "points_rebounds": "v2_role_increment",
        "points_rebounds_assists": "frozen_selected_v1",
        "rebounds": "frozen_selected_v1",
        "rebounds_assists": "frozen_selected_v1",
        "steals": "frozen_selected_v1",
        "threes": "frozen_selected_v1",
    }

    assert (
        routing["dependence_production_lambda"][
            "rebounds_assists"
        ]
        == 0.85
    )

    assert (
        routing["calibration_family_routing"]["blocks"]
        == "raw"
    )


def test_gate3_routing_matches_the_runtime_assertion(
    project_root: Path,
):
    """The contract must agree with what gate3_v2 hard-asserts."""
    source = (
        project_root
        / "src"
        / "nba_prop_quant"
        / "gate3_v2.py"
    ).read_text(encoding="utf-8")

    routing = load_architecture_contract(
        project_root
    ).frozen_choices["routing"]["gate3_routing"]

    for prop, route in routing.items():
        assert f'"{prop}": "{route}"' in source


def test_contract_does_not_freeze_daily_fitted_values(
    project_root: Path,
):
    payload = read_json(
        project_root / CONTRACT_RELATIVE_PATH
    )

    banned = {
        "artifact_hashes",
        "booster_hashes",
        "fitted_parameters",
        "production_parameters",
        "production_weights",
        "role_model_weights",
    }

    def walk(node, trail: str):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in banned, (
                    f"{trail}{key} is a daily fitted value "
                    "and must not be frozen"
                )

                walk(value, f"{trail}{key}.")

        elif isinstance(node, list):
            for item in node:
                walk(item, trail)

    walk(payload["frozen_choices"], "frozen_choices.")
    walk(
        payload["frozen_policy_digests"],
        "frozen_policy_digests.",
    )


# --------------------------------------------------------------------------
# frozen digest separation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative,mutate",
    [
        pytest.param(
            "models/mean_model_selection.json",
            lambda d: d["targets"]["reb"].update(
                production_weights={
                    "xgb": 0.11,
                    "decay": 0.44,
                    "kalman": 0.45,
                }
            ),
            id="ensemble_weights_refit",
        ),
        pytest.param(
            "models/market_probability_calibration_policy.json",
            lambda d: d["props"]["points"].update(
                production_parameters={
                    "intercept": -0.9,
                    "slope": 0.77,
                }
            ),
            id="platt_coefficients_refit",
        ),
        pytest.param(
            "research/v2_gate3_deployment_artifacts"
            "/deployment_manifest.json",
            lambda d: (
                d["input_sha256"].update(
                    {
                        "models/marginals_pre2025.joblib": (
                            "f" * 64
                        )
                    }
                ),
                d.update(
                    role_minutes_training_rows=31000,
                    generated_at_utc="2026-11-15T00:00:00+00:00",
                ),
            ),
            id="role_model_rebuild",
        ),
    ],
)
def test_daily_refit_does_not_break_frozen_digests(
    fake_project: Path, relative, mutate
):
    """Refitting numbers under a frozen architecture must stay registrable."""
    before = frozen_policy_digests(fake_project)

    path = fake_project / relative

    payload = read_json(path)
    mutate(payload)
    write_json(path, payload)

    assert frozen_policy_digests(fake_project) == before


@pytest.mark.parametrize(
    "relative,mutate,expect_changed",
    [
        pytest.param(
            "models/mean_model_selection.json",
            lambda d: d["targets"]["pts"].update(
                selected_mode="ensemble"
            ),
            "mean_model_selection",
            id="mean_model_reselection",
        ),
        pytest.param(
            "models/marginal_selection.json",
            lambda d: d["targets"]["pts"].update(
                selected_distribution="nb"
            ),
            "marginal_selection",
            id="marginal_family_change",
        ),
        pytest.param(
            "models/market_probability_calibration_policy.json",
            lambda d: d["props"]["blocks"].update(
                selected_method="prop"
            ),
            "market_probability_calibration_policy",
            id="calibration_family_change",
        ),
        pytest.param(
            "models/combo_dependence_policy.json",
            lambda d: d["combos"][
                "rebounds_assists"
            ].update(production_lambda=0.5),
            "combo_dependence_policy",
            id="dependence_lambda_change",
        ),
        pytest.param(
            "research/v2_gate3_deployment_artifacts"
            "/deployment_manifest.json",
            lambda d: d["gate3_policy"].update(
                assists="frozen_selected_v1"
            ),
            "gate3_deployment_policy",
            id="gate3_route_change",
        ),
    ],
)
def test_architecture_change_breaks_frozen_digest(
    fake_project: Path, relative, mutate, expect_changed
):
    """Model selection dressed up as a daily fit must be refused."""
    before = frozen_policy_digests(fake_project)

    path = fake_project / relative

    payload = read_json(path)
    mutate(payload)
    write_json(path, payload)

    after = frozen_policy_digests(fake_project)

    assert after[expect_changed] != before[expect_changed]


def test_prune_paths_supports_wildcards():
    payload = {
        "targets": {
            "pts": {"selected_mode": "xgb", "w": [1, 2]},
            "reb": {"selected_mode": "ensemble", "w": [3]},
        },
        "keep": 1,
    }

    pruned = prune_paths(payload, ("targets.*.w",))

    assert pruned == {
        "targets": {
            "pts": {"selected_mode": "xgb"},
            "reb": {"selected_mode": "ensemble"},
        },
        "keep": 1,
    }

    # The original must not be mutated.
    assert "w" in payload["targets"]["pts"]


# --------------------------------------------------------------------------
# architecture lock
# --------------------------------------------------------------------------


def test_wrong_architecture_reference_in_contract_rejected(
    fake_project: Path,
):
    payload = read_json(
        fake_project / CONTRACT_RELATIVE_PATH
    )

    payload["architecture_reference_sha"] = "0" * 40

    write_json(
        fake_project / CONTRACT_RELATIVE_PATH, payload
    )

    with pytest.raises(
        ArchitectureContractViolation,
        match="architecture_reference_sha",
    ):
        load_architecture_contract(fake_project)


def test_wrong_architecture_reference_in_metadata_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        ArchitectureContractViolation,
        match="architecture_reference_sha mismatch",
    ):
        registry.register(
            staged,
            make_metadata(
                architecture_reference_sha="0" * 40
            ),
        )

    assert registry.list_fits() == []


def test_wrong_contract_hash_rejected(
    tmp_path: Path, fake_project: Path, staged: Path
):
    """A fit registered under one contract is unusable under another."""
    registry = make_registry(tmp_path, fake_project)

    fit_id = registry.register(staged, make_metadata())

    contract_path = fake_project / CONTRACT_RELATIVE_PATH

    payload = read_json(contract_path)

    payload["purpose"] = "mutated after registration"

    write_json(contract_path, payload)

    with pytest.raises(
        ArchitectureContractViolation,
        match="registered against contract",
    ):
        registry.verify(fit_id)


def test_feature_schema_drift_rejected(
    tmp_path: Path, fake_project: Path, staged: Path
):
    features = fake_project / FEATURES_RELATIVE_PATH

    features.write_text(
        features.read_text(encoding="utf-8").replace(
            '"is_home",',
            '"is_home", "invented_feature",',
            1,
        ),
        encoding="utf-8",
    )

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation,
        match="feature schema",
    ):
        registry.register(staged, make_metadata())


def test_hyperparameter_drift_rejected(
    tmp_path: Path, fake_project: Path, staged: Path
):
    config = fake_project / CONFIG_RELATIVE_PATH

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "max_depth: 5", "max_depth: 7", 1
        ),
        encoding="utf-8",
    )

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation,
        match="config hash",
    ):
        registry.register(staged, make_metadata())


def test_model_reselection_rejected_at_registration(
    tmp_path: Path, fake_project: Path, staged: Path
):
    """A nightly job that re-ran selection cannot register."""
    path = (
        fake_project
        / "models"
        / "mean_model_selection.json"
    )

    payload = read_json(path)

    payload["targets"]["pts"]["selected_mode"] = "ensemble"

    write_json(path, payload)

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation,
        match="mean_model_selection",
    ):
        registry.register(staged, make_metadata())


def test_gate3_routing_change_rejected_at_registration(
    tmp_path: Path, fake_project: Path, staged: Path
):
    path = (
        fake_project
        / FROZEN_POLICY_SOURCES["gate3_deployment_policy"][
            "path"
        ]
    )

    payload = read_json(path)

    payload["gate3_policy"]["assists"] = (
        "frozen_selected_v1"
    )

    write_json(path, payload)

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation,
        match="gate3_deployment_policy",
    ):
        registry.register(staged, make_metadata())


def test_daily_refit_still_registers(
    tmp_path: Path, fake_project: Path, staged: Path
):
    """The lock must not block a legitimate adaptive fit."""
    path = (
        fake_project
        / "models"
        / "market_probability_calibration_policy.json"
    )

    payload = read_json(path)

    payload["props"]["points"]["production_parameters"] = {
        "intercept": -0.31,
        "slope": 0.52,
    }

    write_json(path, payload)

    registry = make_registry(tmp_path, fake_project)

    fit_id = registry.register(staged, make_metadata())

    assert is_fit_id(fit_id)


@pytest.mark.parametrize(
    "override",
    [
        {"core_seed": 7},
        {"gate3_seed": 1},
        {"effective_n_jobs": -1},
    ],
)
def test_seed_and_n_jobs_mismatch_rejected(
    tmp_path: Path,
    project_root: Path,
    staged: Path,
    override,
):
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(ArchitectureContractViolation):
        registry.register(
            staged, make_metadata(**override)
        )


def test_gate3_candidate_policy_mismatch_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        ArchitectureContractViolation,
        match="gate3_candidate_policy_id",
    ):
        registry.register(
            staged,
            make_metadata(
                gate3_candidate_policy_id="nba_prop_quant_v2_gate3_dead"
            ),
        )


def test_unknown_metadata_field_rejected():
    with pytest.raises(
        ArchitectureContractViolation, match="unknown"
    ):
        FitMetadata.from_dict(
            metadata_payload(surprise="value")
        )


def test_missing_metadata_field_rejected():
    payload = metadata_payload()

    del payload["training_cutoff"]

    with pytest.raises(
        ArchitectureContractViolation, match="missing"
    ):
        FitMetadata.from_dict(payload)


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
# adaptive training window
#
# The architecture is frozen; the end of the eligible training history is not.
# These tests hold that line in both directions: a season completing after the
# freeze must be able to enter a daily fit, and a hyperparameter change must
# still be refused.
# --------------------------------------------------------------------------


def test_no_terminal_training_season_is_frozen(
    project_root: Path,
):
    """2025, or any other season, is never the frozen production end."""
    contract = load_architecture_contract(project_root)

    assert (
        _find_prohibited_training_end(
            contract.frozen_choices, "frozen_choices"
        )
        is None
    )

    policy = contract.frozen_choices[
        TRAINING_WINDOW_POLICY_KEY
    ]

    assert policy["end_policy"] == TRAINING_END_POLICY

    assert (
        policy["cutoff_relation"]
        == TRAINING_CUTOFF_RELATION
    )

    assert "training_window" not in contract.frozen_choices


@pytest.mark.parametrize(
    "key",
    sorted(PROHIBITED_TRAINING_END_KEYS),
)
def test_contract_pinning_a_terminal_season_rejected(
    fake_project: Path, key: str
):
    """Re-freezing a terminal season is refused, not silently honoured."""
    contract_path = fake_project / CONTRACT_RELATIVE_PATH

    payload = read_json(contract_path)

    payload["frozen_choices"][
        TRAINING_WINDOW_POLICY_KEY
    ][key] = 2025

    write_json(contract_path, payload)

    with pytest.raises(
        ArchitectureContractViolation,
        match="terminal production training season",
    ):
        load_architecture_contract(fake_project)


def test_legacy_nested_training_window_rejected(
    fake_project: Path,
):
    """The exact shape this correction removed cannot come back."""
    contract_path = fake_project / CONTRACT_RELATIVE_PATH

    payload = read_json(contract_path)

    payload["frozen_choices"]["training_window"] = {
        "advanced_start_season": 2015,
        "history_start_season": 2001,
        "production_train_end_season": 2025,
    }

    write_json(contract_path, payload)

    with pytest.raises(
        ArchitectureContractViolation,
        match="terminal production training season",
    ):
        load_architecture_contract(fake_project)


def test_missing_training_window_policy_rejected(
    fake_project: Path,
):
    contract_path = fake_project / CONTRACT_RELATIVE_PATH

    payload = read_json(contract_path)

    del payload["frozen_choices"][
        TRAINING_WINDOW_POLICY_KEY
    ]

    write_json(contract_path, payload)

    with pytest.raises(
        ArchitectureContractViolation,
        match=TRAINING_WINDOW_POLICY_KEY,
    ):
        load_architecture_contract(fake_project)


@pytest.mark.parametrize(
    "key",
    ["end_policy", "cutoff_relation"],
)
def test_training_window_policy_semantics_pinned(
    fake_project: Path, key: str
):
    """The policy cannot quietly become a different policy."""
    contract_path = fake_project / CONTRACT_RELATIVE_PATH

    payload = read_json(contract_path)

    payload["frozen_choices"][
        TRAINING_WINDOW_POLICY_KEY
    ][key] = "fixed_at_2025"

    write_json(contract_path, payload)

    with pytest.raises(
        ArchitectureContractViolation, match=key
    ):
        load_architecture_contract(fake_project)


def test_future_training_cutoff_registers(
    tmp_path: Path, project_root: Path, staged: Path
):
    """A cutoff in a season completed long after the freeze is registrable."""
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(
        staged,
        make_metadata(
            fit_date="2031-01-15",
            training_cutoff="2031-01-14",
        ),
    )

    manifest = registry.load_manifest(fit_id)

    assert manifest["training_cutoff"] == "2031-01-14"

    assert manifest["fit_date"] == "2031-01-15"

    registry.verify(fit_id)


@pytest.mark.parametrize(
    "fit_date,training_cutoff",
    [
        ("2031-01-15", "2031-01-15"),
        ("2031-01-15", "2031-01-16"),
        ("2026-11-15", "2027-06-01"),
    ],
)
def test_training_cutoff_not_strictly_before_rejected(
    tmp_path: Path,
    project_root: Path,
    staged: Path,
    fit_date: str,
    training_cutoff: str,
):
    """An expanding window is still never allowed to reach the slate date."""
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        ArchitectureContractViolation,
        match="strictly before",
    ):
        registry.register(
            staged,
            make_metadata(
                fit_date=fit_date,
                training_cutoff=training_cutoff,
            ),
        )

    assert registry.list_fits() == []


def test_frozen_window_boundaries_are_2001_and_2015(
    project_root: Path,
):
    contract = load_architecture_contract(project_root)

    policy = contract.frozen_choices[
        TRAINING_WINDOW_POLICY_KEY
    ]

    assert policy["history_start_season"] == 2001

    assert policy["advanced_start_season"] == 2015

    assert training_window_boundaries(project_root) == {
        "advanced_start_season": 2015,
        "history_start_season": 2001,
    }


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("history_start_season: 2001", "history_start_season: 2002"),
        (
            "advanced_start_season: 2015",
            "advanced_start_season: 2016",
        ),
    ],
)
def test_frozen_window_boundary_drift_rejected(
    tmp_path: Path,
    fake_project: Path,
    staged: Path,
    field: str,
    replacement: str,
):
    """Pruning the boundaries from the digest did not unprotect them."""
    config = fake_project / CONFIG_RELATIVE_PATH

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            field, replacement, 1
        ),
        encoding="utf-8",
    )

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation,
        match="frozen training window",
    ):
        registry.register(staged, make_metadata())

    assert registry.list_fits() == []


@pytest.mark.parametrize(
    "field,replacement",
    [
        (
            "production_train_end_season: 2025",
            "production_train_end_season: 2026",
        ),
        (
            "production_train_end_season: 2025",
            "production_train_end_season: 2031",
        ),
        (
            "play_by_play_start_season: 2025",
            "play_by_play_start_season: 2026",
        ),
    ],
)
def test_season_advancement_is_not_architecture_drift(
    tmp_path: Path,
    fake_project: Path,
    staged: Path,
    field: str,
    replacement: str,
):
    """Advancing an operational season leaves the architecture lock intact."""
    config = fake_project / CONFIG_RELATIVE_PATH

    before = config_hash(fake_project)

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            field, replacement, 1
        ),
        encoding="utf-8",
    )

    assert config_hash(fake_project) == before

    registry = make_registry(tmp_path, fake_project)

    fit_id = registry.register(
        staged,
        make_metadata(
            fit_date="2031-01-15",
            training_cutoff="2031-01-14",
        ),
    )

    assert registry.list_fits() == [fit_id]

    registry.verify(fit_id)


def test_operational_fields_are_outside_the_config_digest(
    project_root: Path,
):
    projection = config_projection(project_root)

    for name in CONFIG_OPERATIONAL_FIELDS:
        assert name not in projection

    # The hyperparameters, seeds and target set stay inside it.
    assert projection["model"]["max_depth"] == 5

    assert projection["model"]["random_state"] == 73

    assert projection["model"]["n_jobs"] == 2

    assert projection["distribution"]["simulations"] == 20000

    assert projection["targets"] == [
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "fg3m",
    ]


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("n_estimators: 700", "n_estimators: 900"),
        ("learning_rate: 0.03", "learning_rate: 0.05"),
        ("max_depth: 5", "max_depth: 7"),
        ("subsample: 0.85", "subsample: 0.7"),
        ("reg_lambda: 4.0", "reg_lambda: 2.0"),
        ("min_child_weight: 8.0", "min_child_weight: 4.0"),
        ("random_state: 73", "random_state: 99"),
        ("simulations: 20000", "simulations: 40000"),
        ("prior_strength: 8.0", "prior_strength: 3.0"),
        ("calibration_fraction: 0.15", "calibration_fraction: 0.3"),
    ],
)
def test_hyperparameter_drift_still_rejected(
    tmp_path: Path,
    fake_project: Path,
    staged: Path,
    field: str,
    replacement: str,
):
    """Projecting the config did not loosen hyperparameter locking."""
    config = fake_project / CONFIG_RELATIVE_PATH

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            field, replacement, 1
        ),
        encoding="utf-8",
    )

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation, match="config hash"
    ):
        registry.register(staged, make_metadata())

    assert registry.list_fits() == []


def test_target_set_change_still_rejected(
    tmp_path: Path, fake_project: Path, staged: Path
):
    config = fake_project / CONFIG_RELATIVE_PATH

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "- fg3m\n", "- fg3m\n- tov\n", 1
        ),
        encoding="utf-8",
    )

    registry = make_registry(tmp_path, fake_project)

    with pytest.raises(
        ArchitectureContractViolation, match="config hash"
    ):
        registry.register(staged, make_metadata())


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

    # It must not be mistakable for either freeze_id namespace.
    assert "gate3" not in fit_id
    assert not is_fit_id("nba_prop_quant_20260818T205213Z")
    assert not is_fit_id(
        "nba_prop_quant_v2_gate3_dd2d394b6def"
    )

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


def test_promotion_state_does_not_affect_identity(
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


@pytest.mark.parametrize(
    "overrides,label",
    [
        ({"training_cutoff": "2026-11-13"}, "cutoff"),
        ({"source_commit_sha": "c" * 40}, "commit"),
        (
            {"training_data_manifest_hash": "b" * 64},
            "data manifest",
        ),
        ({"python_version": "3.13.0"}, "environment"),
        ({"role_state_hash": "d" * 64}, "role state"),
    ],
)
def test_identity_inputs_change_fit_id(
    tmp_path: Path,
    project_root: Path,
    staged: Path,
    overrides,
    label,
):
    baseline = make_registry(
        tmp_path, project_root, "base"
    ).register(staged, make_metadata())

    variant = make_registry(
        tmp_path, project_root, f"var_{label.split()[0]}"
    ).register(staged, make_metadata(**overrides))

    assert baseline != variant


def test_different_artifact_bytes_change_fit_id(
    tmp_path: Path, project_root: Path
):
    first = make_registry(
        tmp_path, project_root, "reg_a"
    ).register(
        make_staged(tmp_path / "one", b"booster-day-1"),
        make_metadata(),
    )

    second = make_registry(
        tmp_path, project_root, "reg_b"
    ).register(
        make_staged(tmp_path / "two", b"booster-day-2"),
        make_metadata(),
    )

    assert first != second


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
        fit_dir
        / "artifacts"
        / "models"
        / "target_pts.joblib"
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

    assert (
        registry.fit_dir(fit_id) / "manifest.json"
    ).read_bytes() == before

    assert list(registry.staging_dir.iterdir()) == []


def test_registered_bytes_are_preserved_exactly(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    for relative in (
        "models/target_pts.joblib",
        "models/marginals.joblib",
        "calibration.json",
    ):
        assert (
            registry.fit_dir(fit_id)
            / "artifacts"
            / relative
        ).read_bytes() == (
            staged / relative
        ).read_bytes()


def test_symlink_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    (staged / "link.joblib").symlink_to(outside)

    registry = make_registry(tmp_path, project_root)

    with pytest.raises(StagedTreeError, match="symlink"):
        registry.register(staged, make_metadata())

    assert registry.list_fits() == []


def test_special_file_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    os.mkfifo(staged / "pipe")

    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        StagedTreeError, match="special file"
    ):
        registry.register(staged, make_metadata())

    assert registry.list_fits() == []


def test_empty_staged_tree_rejected(
    tmp_path: Path, project_root: Path
):
    empty = tmp_path / "empty"
    empty.mkdir()

    registry = make_registry(tmp_path, project_root)

    with pytest.raises(StagedTreeError, match="empty"):
        registry.register(empty, make_metadata())


def test_checksum_inventory_generated(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    text = (
        registry.fit_dir(fit_id) / "SHA256SUMS"
    ).read_text(encoding="utf-8")

    inventory = parse_checksums(text)

    assert "manifest.json" in inventory
    assert (
        "artifacts/models/target_pts.joblib" in inventory
    )

    # Deterministic ordering, sha256sum-compatible format.
    relatives = [
        line.partition("  ")[2]
        for line in text.splitlines()
    ]

    assert relatives == sorted(relatives)

    for line in text.splitlines():
        digest, _, relative = line.partition("  ")

        assert len(digest) == 64
        assert relative


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


def test_added_artifact_detected(
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


def test_removed_artifact_detected(
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

    path = registry.fit_dir(fit_id) / "manifest.json"

    manifest = read_json(path)

    manifest["identity_inputs"]["source_commit_sha"] = (
        "f" * 40
    )

    write_json(path, manifest)

    with pytest.raises(
        IntegrityError, match="manifest tampering"
    ):
        registry.verify(fit_id)


def test_manifest_edit_outside_identity_detected(
    tmp_path: Path, project_root: Path, staged: Path
):
    """Even a non-identity edit fails the sealed inventory."""
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    path = registry.fit_dir(fit_id) / "manifest.json"

    manifest = read_json(path)

    manifest["notes"] = "quietly edited"

    write_json(path, manifest)

    with pytest.raises(IntegrityError):
        registry.verify(fit_id)


def test_missing_checksum_inventory_detected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    (registry.fit_dir(fit_id) / "SHA256SUMS").unlink()

    with pytest.raises(
        IntegrityError, match="SHA256SUMS"
    ):
        registry.verify(fit_id)


def test_manifest_is_portable(
    tmp_path: Path, project_root: Path, staged: Path
):
    """A promoted fit must move to another host without a rewrite."""
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    manifest = registry.load_manifest(fit_id)

    assert manifest["artifact_root"] == "artifacts"

    for relative in manifest["artifact_hashes"]:
        assert not relative.startswith("/")

    text = canonical_json(manifest)

    assert str(tmp_path) not in text
    assert str(project_root) not in text
    assert "/workspace" not in text


def test_fit_directory_survives_relocation(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    relocated = tmp_path / "elsewhere"

    shutil.copytree(registry.root, relocated)

    moved = FitRegistry(
        root=relocated, project_root=project_root
    )

    assert moved.verify(fit_id)["fit_id"] == fit_id


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,pattern",
    [
        (
            {"package_versions": {"bdl_api_key": "abc123"}},
            "credential-shaped field",
        ),
        (
            {"notes": "BDL_API_KEY=supersecretvalue"},
            "credential-shaped value",
        ),
        (
            {"notes": "ssh-rsa AAAAB3NzaC1yc2E"},
            "credential-shaped value",
        ),
        (
            {"notes": "/srv/wizardofodds/runtime/bundle"},
            "absolute path",
        ),
    ],
)
def test_secret_shaped_input_is_refused(
    tmp_path: Path,
    project_root: Path,
    staged: Path,
    overrides,
    pattern,
):
    registry = make_registry(tmp_path, project_root)

    with pytest.raises(
        ArchitectureContractViolation, match=pattern
    ):
        registry.register(
            staged, make_metadata(**overrides)
        )

    assert registry.list_fits() == []


def test_no_secret_fields_written_anywhere(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    promote_two_fits(registry, tmp_path)

    registry.rollback(reason="unit test")

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
            "ssh-rsa",
            "bearer ",
            "private key",
        ):
            assert banned not in text, f"{banned} in {path}"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_registration_does_not_auto_promote(
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


def test_validation_record_for_wrong_fit_rejected(
    tmp_path: Path, project_root: Path
):
    """A record copied from another fit must not authorise promotion."""
    registry = make_registry(tmp_path, project_root)

    good = register_and_validate(
        registry,
        make_staged(tmp_path / "d1", b"day-1"),
        make_metadata(),
    )

    target = registry.register(
        make_staged(tmp_path / "d2", b"day-2"),
        make_metadata(
            fit_date="2026-11-16",
            training_cutoff="2026-11-15",
        ),
    )

    shutil.copyfile(
        registry.validation_path(good),
        registry.validation_path(target),
    )

    with pytest.raises(
        ValidationRefused,
        match="validation record belongs to",
    ):
        registry.promote(target, reason="unit test")

    assert (
        registry.current()["current_good_fit_id"] is None
    )


def test_malformed_validation_record_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    write_json(
        registry.validation_path(fit_id),
        {"fit_id": fit_id, "schema_version": 1},
    )

    with pytest.raises(
        ValidationRefused, match="no checks"
    ):
        registry.promote(fit_id, reason="unit test")


def test_unsupported_validation_schema_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    record = read_json(registry.validation_path(fit_id))

    record["schema_version"] = 99

    write_json(registry.validation_path(fit_id), record)

    with pytest.raises(
        ValidationRefused, match="schema_version"
    ):
        registry.promote(fit_id, reason="unit test")


def test_non_boolean_validation_check_rejected(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = registry.register(staged, make_metadata())

    checks = passing_checks()
    checks["finite_values_check"] = "yes"

    with pytest.raises(
        ValidationRefused, match="must be a boolean"
    ):
        registry.record_validation(fit_id, checks)


def test_unknown_validation_check_rejected(
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


def test_validation_lives_outside_the_fit_directory(
    tmp_path: Path, project_root: Path, staged: Path
):
    registry = make_registry(tmp_path, project_root)

    fit_id = register_and_validate(
        registry, staged, make_metadata()
    )

    assert registry.validation_path(fit_id).is_file()

    assert (
        registry.validations_dir
        == registry.state_dir / "validations"
    )

    names = {
        path.name
        for path in registry.fit_dir(fit_id).rglob("*")
    }

    assert not any(
        "validation" in name for name in names
    )


def test_no_model_selection_criterion_is_required():
    """Promotion is not model selection."""
    joined = " ".join(REQUIRED_VALIDATION_CHECKS)

    for banned in (
        "superiority",
        "beats",
        "better_than",
        "selection",
        "outperform",
        "profit",
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
    assert current["promoted_at"]


def test_second_promotion_moves_previous_good(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first, second = promote_two_fits(registry, tmp_path)

    current = registry.current()

    assert current["current_good_fit_id"] == second
    assert current["previous_good_fit_id"] == first


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


def test_failed_candidate_preserves_current_good(
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

    # A failed candidate is immutable production evidence.
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

    first, second = promote_two_fits(registry, tmp_path)

    state = registry.rollback(
        reason="regression found", actor="operator"
    )

    assert state["current_good_fit_id"] == first
    assert state["previous_good_fit_id"] == second

    # Rollback deletes nothing.
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


def test_rollback_to_missing_fit_rejected(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first, second = promote_two_fits(registry, tmp_path)

    shutil.rmtree(registry.fit_dir(first))

    with pytest.raises(FitNotFound):
        registry.rollback(reason="attempted")

    assert (
        registry.current()["current_good_fit_id"] == second
    )


def test_rollback_to_corrupt_fit_rejected(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    first, second = promote_two_fits(registry, tmp_path)

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


def test_promotion_history_is_recorded(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    promote_two_fits(registry, tmp_path)

    registry.rollback(reason="regression")

    history = registry.load_promotion_state()["history"]

    assert [entry["action"] for entry in history] == [
        "promote",
        "promote",
        "rollback",
    ]

    assert all(entry["at"] for entry in history)


# --------------------------------------------------------------------------
# atomicity and locking
# --------------------------------------------------------------------------


def test_promotion_state_replacement_is_atomic(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A crash at the replace step must leave current-good untouched."""
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
        registry.promotion_state_path.read_bytes()
        == before
    )

    assert (
        registry.current()["current_good_fit_id"] == first
    )

    # No partial temporary state is left behind.
    assert [
        path.name
        for path in registry.state_dir.iterdir()
        if path.suffix == ".tmp"
    ] == []


def test_exclusive_promotion_lock(
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

    # Once released the same promotion succeeds.
    registry.promote(fit_id, reason="after unlock")

    assert (
        registry.current()["current_good_fit_id"] == fit_id
    )


def test_rollback_is_lock_protected(
    tmp_path: Path, project_root: Path
):
    registry = make_registry(tmp_path, project_root)

    _, second = promote_two_fits(registry, tmp_path)

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


def test_registry_root_required():
    with pytest.raises(
        RegistryRootError, match="no registry root"
    ):
        resolve_registry_root(None, environ={})

    with pytest.raises(RegistryRootError):
        resolve_registry_root(
            None, environ={REGISTRY_ENV_VAR: "   "}
        )


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
        project_root,
        project_root / "registry",
        project_root / "models" / "fits",
    ):
        with pytest.raises(
            RegistryRootError,
            match="inside the repository",
        ):
            resolve_registry_root(
                candidate, project_root=project_root
            )


def test_no_registry_written_into_repository(
    project_root: Path,
):
    for name in ("fits", "registry", "staging", "locks"):
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

    assert "no registry root supplied" in result.stderr


def test_cli_refuses_registry_inside_repository():
    result = run_cli(
        "current",
        "--registry-root",
        str(REPO / "registry"),
    )

    assert result.returncode != 0

    assert "inside the repository" in result.stderr


def test_cli_full_lifecycle(
    tmp_path: Path, project_root: Path
):
    root = tmp_path / "registry"

    checks_path = tmp_path / "checks.json"

    checks_path.write_text(
        json.dumps(passing_checks()), encoding="utf-8"
    )

    metadata_path = tmp_path / "metadata.json"

    fit_ids = []

    for day, cutoff, marker in (
        ("2026-11-15", "2026-11-14", b"day-1"),
        ("2026-11-16", "2026-11-15", b"day-2"),
    ):
        staged_dir = make_staged(
            tmp_path / f"staged_{day}", marker
        )

        metadata_path.write_text(
            json.dumps(
                metadata_payload(
                    fit_date=day, training_cutoff=cutoff
                )
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

        assert (
            registered.returncode == 0
        ), registered.stderr

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
        assert (
            json.loads(recorded.stdout)["passed"] is True
        )

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


def test_cli_duplicate_registration_is_non_zero(
    tmp_path: Path, project_root: Path
):
    root = tmp_path / "registry"

    staged_dir = make_staged(tmp_path / "staged")

    metadata_path = tmp_path / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata_payload()), encoding="utf-8"
    )

    args = (
        "register",
        "--registry-root",
        str(root),
        "--staged-dir",
        str(staged_dir),
        "--metadata",
        str(metadata_path),
    )

    assert run_cli(*args).returncode == 0

    repeated = run_cli(*args)

    assert repeated.returncode != 0
    assert (
        json.loads(repeated.stderr)["error"]
        == "FitAlreadyExists"
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
