"""Every Step 3C/3D validation check is a measurement, not a label.

The daily fit used to record its validation checks as constant ``True``. A
constant cannot fail, so the registry's validation contract was satisfied by
assertion. These tests exist to prove the opposite property of the replacement:
for every required check there is a deliberately broken input that drives that
specific check -- and only that check -- to ``False``.

The two checks the frozen design defers, ``gate3_role_readiness`` and
``t20_protocol_compatible``, get both halves: a real captured slate makes them
answer, and no capture makes them report NOT_EVALUABLE rather than inventing an
answer about a capture that never happened.

Nothing here trains a model, reaches the network, or writes outside a pytest
temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest

from nba_prop_quant.adaptive_fit_registry import (
    REQUIRED_VALIDATION_CHECKS,
    config_hash,
    feature_schema_hash,
    frozen_policy_digests,
    load_architecture_contract,
    sha256_file,
)
from nba_prop_quant.adaptive_training import (
    ADVANCED_START_SEASON,
    FIT_STAGES,
    FROZEN_CALIBRATION_ROUTES,
    FROZEN_MARGINAL_FAMILY,
    FROZEN_MEAN_ROUTES,
    REQUIRED_CANDIDATE_FILES,
    REQUIRED_CANDIDATE_PREFIXES,
    lineage_sources,
)
from nba_prop_quant.adaptive_validation import (
    NOT_EVALUABLE,
    ValidationComputationError,
    ValidationReport,
    check_advanced_coverage_check,
    check_architecture_contract_match,
    check_artifact_hashes_valid,
    check_calibration_valid,
    check_data_refresh_valid,
    check_feature_schema_match,
    check_finite_values_check,
    check_gate3_role_readiness,
    check_history_regression_check,
    check_marginal_fit_valid,
    check_prediction_smoke_test,
    check_required_artifacts_present,
    check_source_lineage_match,
    check_t20_protocol_compatible,
    check_training_completed,
    compute_validation_report,
)
from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.distributions import FittedMarginal, NegativeBinomialCalibrator
from nba_prop_quant.prospective_snapshot import (
    PRIMARY_OFFSET_MINUTES,
    build_capture_id,
    canonical_records_sha256,
)
from nba_prop_quant.storage import timestamped_jsonl_append


PROJECT = Path(__file__).resolve().parents[1]

SLATE = "2026-11-15"

TARGETS = tuple(sorted(FROZEN_MEAN_ROUTES))

GAME_ID = 999

CAPTURED_AT = "2026-11-15T22:40:00+00:00"


@pytest.fixture(autouse=True)
def no_snapshot_override(monkeypatch):
    """The Gate 3 snapshot override must not leak in from the environment."""
    monkeypatch.delenv("NBA_PROP_GATE3_SNAPSHOT_DIR", raising=False)

    yield


# ----------------------------------------------------------------------
# a candidate tree that passes, and the knobs to break it
# ----------------------------------------------------------------------


def marginals() -> dict[str, FittedMarginal]:
    built: dict[str, FittedMarginal] = {}

    for index, target in enumerate(TARGETS):
        model = NegativeBinomialCalibrator()
        model.size = 6.0 + index
        built[target] = FittedMarginal(kind="nb", model=model)

    return built


def copula(correlation: np.ndarray | None = None) -> GaussianCopula:
    if correlation is None:
        correlation = np.full((len(TARGETS), len(TARGETS)), 0.2, dtype=float)
        np.fill_diagonal(correlation, 1.0)

    fitted = GaussianCopula(targets=list(TARGETS))
    fitted.global_corr = correlation

    return fitted


def eligible_inputs(seasons: list[int], advanced: list[int]) -> dict[str, str]:
    """A manifest's eligible-input digest map for a synthetic corpus."""
    inputs: dict[str, str] = {}

    for season in seasons:
        key = f"raw/seasons/season={season}/stats.parquet"
        inputs[key] = hashlib.sha256(key.encode("utf-8")).hexdigest()

    for season in advanced:
        key = f"raw/advanced/season={season}/advanced.parquet"
        inputs[key] = hashlib.sha256(key.encode("utf-8")).hexdigest()

    return inputs


def manifest_payload(
    *,
    seasons: list[int] | None = None,
    advanced: list[int] | None = None,
) -> dict[str, object]:
    seasons = [2024, 2025, 2026] if seasons is None else seasons
    advanced = list(seasons) if advanced is None else advanced

    contract = load_architecture_contract(PROJECT)

    return {
        "architecture_contract_sha256": contract.sha256,
        "config_hash": config_hash(PROJECT),
        "eligible_input_sha256": eligible_inputs(seasons, advanced),
        "feature_schema_hash": feature_schema_hash(PROJECT),
        "frozen_policy_digests": frozen_policy_digests(PROJECT),
        "rolling_state_fingerprint": "f" * 64,
    }


def build_candidate(root: Path) -> Path:
    """A candidate tree that satisfies every computable check."""
    candidate = root / "candidate"

    (candidate / "models").mkdir(parents=True, exist_ok=True)
    (candidate / "provenance").mkdir(parents=True, exist_ok=True)

    joblib.dump(marginals(), candidate / "models" / "marginals.joblib")
    joblib.dump(copula(), candidate / "models" / "copula.joblib")

    (candidate / "models" / "parameters.json").write_text(
        json.dumps({"slope": 0.41}, indent=2, sort_keys=True), encoding="utf-8"
    )

    (candidate / "training_data_manifest.json").write_text(
        json.dumps(manifest_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for name, relative in lineage_sources().items():
        shutil.copyfile(PROJECT / relative, candidate / "provenance" / name)

    return candidate


def artifact_hashes(candidate: Path) -> dict[str, str]:
    return {
        path.relative_to(candidate).as_posix(): sha256_file(path)
        for path in sorted(candidate.rglob("*"))
        if path.is_file()
    }


def rolling_state(fingerprint: str = "f" * 64) -> dict[str, object]:
    return {
        "datasets_fingerprint": fingerprint,
        "datasets": {
            "games": {
                "row_count": 1230,
                "semantic_fingerprint": "a" * 64,
            },
            "stats": {
                "row_count": 29000,
                "semantic_fingerprint": "b" * 64,
            },
        },
    }


def calibration_digests() -> dict[str, str]:
    return {
        prop: hashlib.sha256(prop.encode("utf-8")).hexdigest()
        for prop, route in FROZEN_CALIBRATION_ROUTES.items()
        if route == "prop"
    }


# ----------------------------------------------------------------------
# a captured slate, and the knobs to break it
# ----------------------------------------------------------------------


def write_capture(
    snapshot_dir: Path,
    *,
    date: str = SLATE,
    offset_minutes: int = PRIMARY_OFFSET_MINUTES,
    lineups: bool = True,
    error_count: int = 0,
) -> None:
    """A single scheduled capture for one game, in the on-disk layout.

    The knobs are the three ways a capture can be present but unusable: it was
    taken at the wrong offset, it carries no lineup records, or it recorded
    errors. Each one is a negative control for a Step 3D check.
    """
    game = {
        "id": GAME_ID,
        "date": date,
        "datetime": f"{date}T23:00:00+00:00",
        "home_team": {"id": 1},
        "visitor_team": {"id": 2},
    }

    components: dict[str, list[dict[str, object]]] = {
        "games": [game],
        "active_players": [
            {
                "id": 101,
                "first_name": "A",
                "last_name": "One",
                "team": {"id": 1},
            },
            {
                "id": 201,
                "first_name": "B",
                "last_name": "One",
                "team": {"id": 2},
            },
        ],
        "injuries": [],
        "lineups": (
            [
                {
                    "id": 1,
                    "game_id": GAME_ID,
                    "starter": True,
                    "position": "G",
                    "player": {"id": 101},
                    "team": {"id": 1},
                },
                {
                    "id": 2,
                    "game_id": GAME_ID,
                    "starter": True,
                    "position": "G",
                    "player": {"id": 201},
                    "team": {"id": 2},
                },
            ]
            if lineups
            else []
        ),
        "player_props": [
            {
                "id": 5001,
                "game_id": GAME_ID,
                "player_id": 101,
                "vendor": "test",
                "prop_type": "assists",
                "line_value": "5.5",
                "market": {
                    "type": "over_under",
                    "over_odds": -110,
                    "under_odds": -110,
                },
            }
        ],
    }

    directories = {
        "games": ("games", "game"),
        "active_players": ("active_players", "active_player"),
        "injuries": ("injuries", "injury"),
        "lineups": ("lineups", "lineup"),
        "player_props": ("player_props", "live_player_prop"),
    }

    for name, records in components.items():
        directory, snapshot_type = directories[name]

        if records:
            timestamped_jsonl_append(
                records,
                snapshot_dir / directory / f"{date}.jsonl",
                snapshot_type,
                captured_at=CAPTURED_AT,
            )

    digests = {
        name: canonical_records_sha256(records)
        for name, records in components.items()
    }

    window_ids = [f"{GAME_ID}:T-{int(offset_minutes)}m"]

    run = {
        "date": date,
        "capture_reason": "scheduled",
        "window_ids": window_ids,
        "due_game_ids": [GAME_ID],
        "grace_minutes": 5,
        "games_count": 1,
        "active_players_count": 2,
        "game_ids": [GAME_ID],
        "team_ids": [1, 2],
        "injuries_count": 0,
        "lineups_count": len(components["lineups"]),
        "player_props_count": 1,
        "error_count": int(error_count),
        "errors": [],
        "component_sha256": digests,
        "capture_id": build_capture_id(
            date=date,
            captured_at=CAPTURED_AT,
            window_ids=window_ids,
            due_game_ids=[GAME_ID],
            component_sha256=digests,
        ),
    }

    timestamped_jsonl_append(
        [run],
        snapshot_dir / "capture_runs" / f"{date}.jsonl",
        "capture_run",
        captured_at=CAPTURED_AT,
    )


# ----------------------------------------------------------------------
# the whole report, over a candidate that should pass
# ----------------------------------------------------------------------


def report_for(
    candidate: Path,
    *,
    snapshot_dir: Path | None = None,
    **overrides,
) -> ValidationReport:
    manifest = overrides.pop("manifest", None) or json.loads(
        (candidate / "training_data_manifest.json").read_text(encoding="utf-8")
    )

    arguments: dict[str, object] = {
        "project_root": PROJECT,
        "candidate": candidate,
        "contract_object": load_architecture_contract(PROJECT),
        "manifest": manifest,
        "state": rolling_state(),
        "parent_manifest": None,
        "artifact_hashes": artifact_hashes(candidate),
        "finite": True,
        "finite_offenders": [],
        "required_stages": FIT_STAGES,
        "observed_stages": FIT_STAGES,
        "required_prefixes": REQUIRED_CANDIDATE_PREFIXES,
        "required_files": REQUIRED_CANDIDATE_FILES,
        "lineage_sources": lineage_sources(),
        "targets": TARGETS,
        "frozen_marginal_family": FROZEN_MARGINAL_FAMILY,
        "calibration_routes": dict(FROZEN_CALIBRATION_ROUTES),
        "calibration_hashes": calibration_digests(),
        "calibration_fallbacks": {},
        "advanced_start_season": ADVANCED_START_SEASON,
        "role_state_hash": hashlib.sha256(b"role").hexdigest(),
        "slate_date": SLATE,
        "snapshot_dir": snapshot_dir,
    }

    arguments.update(overrides)

    return compute_validation_report(**arguments)


def test_a_healthy_candidate_passes_every_computable_check(tmp_path):
    report = report_for(build_candidate(tmp_path))

    assert report.failed() == []
    assert report.passed is True


def test_the_report_answers_every_name_the_registry_requires(tmp_path):
    report = report_for(build_candidate(tmp_path))

    assert set(report.by_name) == set(REQUIRED_VALIDATION_CHECKS)


def test_without_a_capture_only_the_two_step3d_checks_are_deferred(tmp_path):
    report = report_for(build_candidate(tmp_path))

    assert report.deferred() == [
        "gate3_role_readiness",
        "t20_protocol_compatible",
    ]

    recorded = report.recorded_checks()

    assert len(recorded) == len(REQUIRED_VALIDATION_CHECKS) - 2
    assert all(recorded.values())


def test_a_deferred_check_is_not_recorded_as_a_boolean(tmp_path):
    report = report_for(build_candidate(tmp_path))

    for name in report.deferred():
        assert name not in report.recorded_checks()
        assert report.by_name[name].values["status"] == NOT_EVALUABLE
        assert report.by_name[name].passed is None


def test_a_report_missing_a_required_check_is_refused():
    with pytest.raises(ValidationComputationError, match="does not answer"):
        from nba_prop_quant.adaptive_validation import (
            assert_every_required_check_is_answered,
        )

        assert_every_required_check_is_answered(ValidationReport(results=()))


# ----------------------------------------------------------------------
# negative controls: one deliberately broken input per check
# ----------------------------------------------------------------------


def test_training_completed_is_false_when_a_stage_did_not_run():
    result = check_training_completed(
        required_stages=FIT_STAGES,
        observed_stages=tuple(
            stage for stage in FIT_STAGES if stage != "calibration_fit"
        ),
    )

    assert result.passed is False
    assert "calibration_fit" in result.values["missing_stages"]


def test_data_refresh_valid_is_false_when_the_fingerprint_moved():
    result = check_data_refresh_valid(
        state=rolling_state(fingerprint="e" * 64),
        plan_fingerprint="f" * 64,
    )

    assert result.passed is False
    assert "moved between planning and validation" in result.error


def test_data_refresh_valid_is_false_for_an_unfingerprinted_dataset():
    state = rolling_state()
    state["datasets"]["stats"]["semantic_fingerprint"] = "not-a-digest"

    result = check_data_refresh_valid(state=state, plan_fingerprint="f" * 64)

    assert result.passed is False
    assert "stats" in result.error


def test_history_regression_is_false_when_a_dataset_lost_rows():
    result = check_history_regression_check(
        state=rolling_state(),
        parent_manifest={
            "rolling_state_datasets": {
                "games": {"row_count": 1230},
                "stats": {"row_count": 30000},
            }
        },
    )

    assert result.passed is False
    assert result.values["regressed"] == ["stats: 29000 < 30000"]


def test_history_regression_is_false_when_a_dataset_disappeared():
    result = check_history_regression_check(
        state=rolling_state(),
        parent_manifest={
            "rolling_state_datasets": {
                "games": {"row_count": 1},
                "stats": {"row_count": 1},
                "advanced": {"row_count": 1},
            }
        },
    )

    assert result.passed is False
    assert result.values["dropped_datasets"] == ["advanced"]


def test_history_regression_passes_the_first_fit_with_no_parent():
    result = check_history_regression_check(
        state=rolling_state(), parent_manifest=None
    )

    assert result.passed is True
    assert result.values["parent"] is None


def test_advanced_coverage_is_false_when_a_season_has_stats_but_no_advanced():
    result = check_advanced_coverage_check(
        manifest=manifest_payload(
            seasons=[2024, 2025, 2026], advanced=[2024, 2026]
        ),
        advanced_start_season=ADVANCED_START_SEASON,
    )

    assert result.passed is False
    assert result.values["missing_seasons"] == [2025]


def test_advanced_coverage_is_false_for_orphaned_advanced_data():
    result = check_advanced_coverage_check(
        manifest=manifest_payload(seasons=[2026], advanced=[2025, 2026]),
        advanced_start_season=ADVANCED_START_SEASON,
    )

    assert result.passed is False
    assert result.values["orphaned_advanced_seasons"] == [2025]


def test_advanced_coverage_does_not_require_data_below_the_frozen_floor():
    below = ADVANCED_START_SEASON - 1

    result = check_advanced_coverage_check(
        manifest=manifest_payload(seasons=[below, 2026], advanced=[2026]),
        advanced_start_season=ADVANCED_START_SEASON,
    )

    assert result.passed is True
    assert below in result.values["stats_seasons"]


def test_required_artifacts_present_is_false_without_a_serving_model(tmp_path):
    candidate = build_candidate(tmp_path)

    hashes = artifact_hashes(candidate)

    result = check_required_artifacts_present(
        artifact_hashes={
            name: digest
            for name, digest in hashes.items()
            if not name.startswith("models/")
        },
        required_prefixes=REQUIRED_CANDIDATE_PREFIXES,
        required_files=REQUIRED_CANDIDATE_FILES,
    )

    assert result.passed is False
    assert "models/*" in result.values["missing"]


def test_artifact_hashes_valid_is_false_when_a_file_changed(tmp_path):
    candidate = build_candidate(tmp_path)

    hashes = artifact_hashes(candidate)

    (candidate / "models" / "parameters.json").write_text(
        json.dumps({"slope": 0.42}), encoding="utf-8"
    )

    result = check_artifact_hashes_valid(
        candidate=candidate, artifact_hashes=hashes
    )

    assert result.passed is False
    assert result.values["mismatched"] == ["models/parameters.json"]


def test_artifact_hashes_valid_is_false_when_a_file_disappeared(tmp_path):
    candidate = build_candidate(tmp_path)

    hashes = artifact_hashes(candidate)

    (candidate / "models" / "parameters.json").unlink()

    result = check_artifact_hashes_valid(
        candidate=candidate, artifact_hashes=hashes
    )

    assert result.passed is False
    assert result.values["vanished"] == ["models/parameters.json"]


def test_finite_values_is_false_for_a_non_finite_parameter():
    result = check_finite_values_check(
        finite=False, offenders=["models/parameters.json:slope"]
    )

    assert result.passed is False
    assert "slope" in result.error


def test_feature_schema_match_is_false_when_the_manifest_disagrees(tmp_path):
    manifest = manifest_payload()
    manifest["feature_schema_hash"] = "0" * 64

    result = check_feature_schema_match(project_root=PROJECT, manifest=manifest)

    assert result.passed is False
    assert "feature schema hash differs" in result.error


def test_feature_schema_match_is_false_when_the_config_hash_disagrees():
    manifest = manifest_payload()
    manifest["config_hash"] = "0" * 64

    result = check_feature_schema_match(project_root=PROJECT, manifest=manifest)

    assert result.passed is False
    assert "model config hash differs" in result.error


def test_architecture_contract_match_is_false_for_a_different_contract():
    manifest = manifest_payload()
    manifest["architecture_contract_sha256"] = "0" * 64

    result = check_architecture_contract_match(
        project_root=PROJECT,
        contract_object=load_architecture_contract(PROJECT),
        manifest=manifest,
    )

    assert result.passed is False
    assert "different architecture contract" in result.error


def test_source_lineage_match_is_false_for_a_drifted_provenance_copy(tmp_path):
    candidate = build_candidate(tmp_path)

    target = candidate / "provenance" / "model.yaml"

    target.write_text(
        target.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8"
    )

    result = check_source_lineage_match(
        project_root=PROJECT,
        candidate=candidate,
        manifest=manifest_payload(),
        lineage_sources=lineage_sources(),
    )

    assert result.passed is False
    assert result.values["mismatched"] == ["model.yaml"]


def test_source_lineage_match_is_false_for_an_undeclared_artifact(tmp_path):
    candidate = build_candidate(tmp_path)

    (candidate / "provenance" / "something_else.json").write_text(
        "{}", encoding="utf-8"
    )

    result = check_source_lineage_match(
        project_root=PROJECT,
        candidate=candidate,
        manifest=manifest_payload(),
        lineage_sources=lineage_sources(),
    )

    assert result.passed is False
    assert result.values["undeclared"] == ["something_else.json"]


def test_source_lineage_match_is_false_with_no_lineage_at_all(tmp_path):
    candidate = build_candidate(tmp_path)

    shutil.rmtree(candidate / "provenance")
    (candidate / "provenance").mkdir()

    result = check_source_lineage_match(
        project_root=PROJECT,
        candidate=candidate,
        manifest=manifest_payload(),
        lineage_sources=lineage_sources(),
    )

    assert result.passed is False
    assert "records no lineage" in result.error


def test_source_lineage_match_is_false_when_policy_digests_drifted(tmp_path):
    candidate = build_candidate(tmp_path)

    manifest = manifest_payload()
    manifest["frozen_policy_digests"] = {
        name: "0" * 64 for name in manifest["frozen_policy_digests"]
    }

    result = check_source_lineage_match(
        project_root=PROJECT,
        candidate=candidate,
        manifest=manifest,
        lineage_sources=lineage_sources(),
    )

    assert result.passed is False
    assert result.values["drifted_policy_digests"]


def test_marginal_fit_valid_is_false_when_a_target_has_no_marginal(tmp_path):
    candidate = build_candidate(tmp_path)

    partial = marginals()
    partial.pop("pts")

    joblib.dump(partial, candidate / "models" / "marginals.joblib")

    result = check_marginal_fit_valid(
        candidate=candidate,
        targets=TARGETS,
        frozen_family=FROZEN_MARGINAL_FAMILY,
    )

    assert result.passed is False
    assert result.values["missing_targets"] == ["pts"]


def test_marginal_fit_valid_is_false_outside_the_frozen_family(tmp_path):
    candidate = build_candidate(tmp_path)

    wrong = marginals()
    wrong["pts"] = FittedMarginal(kind="poisson", model=wrong["pts"].model)

    joblib.dump(wrong, candidate / "models" / "marginals.joblib")

    result = check_marginal_fit_valid(
        candidate=candidate,
        targets=TARGETS,
        frozen_family=FROZEN_MARGINAL_FAMILY,
    )

    assert result.passed is False
    assert "pts: poisson" in result.error


def test_marginal_fit_valid_is_false_when_the_object_is_not_a_mapping(tmp_path):
    candidate = build_candidate(tmp_path)

    joblib.dump(["not", "a", "mapping"], candidate / "models" / "marginals.joblib")

    result = check_marginal_fit_valid(
        candidate=candidate,
        targets=TARGETS,
        frozen_family=FROZEN_MARGINAL_FAMILY,
    )

    assert result.passed is False
    assert "not a target-keyed mapping" in result.error


def test_calibration_valid_is_false_when_a_prop_route_lost_its_digest():
    digests = calibration_digests()
    digests.pop("assists")

    result = check_calibration_valid(
        calibration_hashes=digests,
        calibration_fallbacks={},
        calibration_routes=dict(FROZEN_CALIBRATION_ROUTES),
    )

    assert result.passed is False
    assert result.values["missing_digests"] == ["assists"]


def test_calibration_valid_is_false_when_a_raw_route_acquired_parameters():
    digests = calibration_digests()
    digests["blocks"] = hashlib.sha256(b"blocks").hexdigest()

    result = check_calibration_valid(
        calibration_hashes=digests,
        calibration_fallbacks={},
        calibration_routes=dict(FROZEN_CALIBRATION_ROUTES),
    )

    assert result.passed is False
    assert result.values["unexpected_digests"] == ["blocks"]


def test_calibration_valid_is_false_for_a_fallback_outside_the_route_table():
    result = check_calibration_valid(
        calibration_hashes=calibration_digests(),
        calibration_fallbacks={"not_a_prop": "prior_good"},
        calibration_routes=dict(FROZEN_CALIBRATION_ROUTES),
    )

    assert result.passed is False
    assert "not_a_prop" in result.error


def test_calibration_valid_is_false_for_a_raw_routed_fallback():
    result = check_calibration_valid(
        calibration_hashes=calibration_digests(),
        calibration_fallbacks={"steals": "prior_good"},
        calibration_routes=dict(FROZEN_CALIBRATION_ROUTES),
    )

    assert result.passed is False
    assert "steals" in result.error


def test_calibration_valid_accepts_a_prop_routed_fallback():
    result = check_calibration_valid(
        calibration_hashes=calibration_digests(),
        calibration_fallbacks={"assists": "prior_good"},
        calibration_routes=dict(FROZEN_CALIBRATION_ROUTES),
    )

    assert result.passed is True


# ----------------------------------------------------------------------
# the smoke test, which used to be the most load-bearing constant
# ----------------------------------------------------------------------


def test_prediction_smoke_test_prices_a_real_candidate(tmp_path):
    result = check_prediction_smoke_test(
        candidate=build_candidate(tmp_path), targets=TARGETS
    )

    assert result.passed is True
    assert result.values["priced_triples"] > 0
    assert result.values["minimum_eigenvalue"] > 0.0


def test_prediction_smoke_test_is_false_for_unloadable_marginals(tmp_path):
    candidate = build_candidate(tmp_path)

    (candidate / "models" / "marginals.joblib").write_bytes(
        b"zinb-fitted-parameters"
    )

    result = check_prediction_smoke_test(candidate=candidate, targets=TARGETS)

    assert result.passed is False
    assert result.error


def test_prediction_smoke_test_is_false_when_a_serving_object_is_absent(
    tmp_path,
):
    candidate = build_candidate(tmp_path)

    (candidate / "models" / "copula.joblib").unlink()

    result = check_prediction_smoke_test(candidate=candidate, targets=TARGETS)

    assert result.passed is False
    assert result.values["absent"] == ["models/copula.joblib"]


def test_prediction_smoke_test_is_false_for_an_indefinite_copula(tmp_path):
    candidate = build_candidate(tmp_path)

    # A unit diagonal with off-diagonals this large is not a correlation
    # matrix: it has a negative eigenvalue, so nothing can be simulated from
    # it. The structured-value check cannot see this, which is why the smoke
    # test has to.
    indefinite = np.full((len(TARGETS), len(TARGETS)), 0.95, dtype=float)
    np.fill_diagonal(indefinite, 1.0)
    indefinite[0, 1] = indefinite[1, 0] = -0.99

    joblib.dump(
        copula(indefinite), candidate / "models" / "copula.joblib"
    )

    result = check_prediction_smoke_test(candidate=candidate, targets=TARGETS)

    assert result.passed is False
    assert result.values["minimum_eigenvalue"] < 0.0


def test_prediction_smoke_test_is_false_for_a_non_unit_diagonal(tmp_path):
    candidate = build_candidate(tmp_path)

    scaled = np.zeros((len(TARGETS), len(TARGETS)), dtype=float)
    np.fill_diagonal(scaled, 4.0)

    joblib.dump(copula(scaled), candidate / "models" / "copula.joblib")

    result = check_prediction_smoke_test(candidate=candidate, targets=TARGETS)

    assert result.passed is False
    assert "diagonal departs" in result.error


def test_prediction_smoke_test_is_false_when_a_marginal_cannot_price(tmp_path):
    candidate = build_candidate(tmp_path)

    broken = marginals()
    broken["pts"].model.size = float("nan")

    joblib.dump(broken, candidate / "models" / "marginals.joblib")

    result = check_prediction_smoke_test(candidate=candidate, targets=TARGETS)

    assert result.passed is False
    assert any("pts" in offender for offender in result.values["offenders"])


# ----------------------------------------------------------------------
# the two Step 3D checks: positive, negative and deferred
# ----------------------------------------------------------------------


def test_t20_is_not_evaluable_without_a_snapshot_directory():
    result = check_t20_protocol_compatible(slate_date=SLATE, snapshot_dir=None)

    assert result.passed is None
    assert result.values["status"] == NOT_EVALUABLE


def test_t20_is_not_evaluable_when_the_slate_was_never_captured(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    result = check_t20_protocol_compatible(
        slate_date=SLATE, snapshot_dir=snapshots
    )

    assert result.passed is None
    assert result.values["status"] == NOT_EVALUABLE


def test_t20_passes_against_a_real_capture_at_the_frozen_offset(tmp_path):
    snapshots = tmp_path / "snapshots"

    write_capture(snapshots)

    result = check_t20_protocol_compatible(
        slate_date=SLATE, snapshot_dir=snapshots
    )

    assert result.passed is True
    assert result.values["verified_bundles"] == 1
    assert result.values["protocol_offset_minutes"] == PRIMARY_OFFSET_MINUTES


def test_t20_is_not_evaluable_when_the_only_capture_is_off_protocol(tmp_path):
    snapshots = tmp_path / "snapshots"

    write_capture(snapshots, offset_minutes=PRIMARY_OFFSET_MINUTES + 25)

    result = check_t20_protocol_compatible(
        slate_date=SLATE, snapshot_dir=snapshots
    )

    # A capture at another offset advertises no T-20 window at all, so the
    # frozen protocol was never exercised: that is deferred, not satisfied.
    assert result.passed is None
    assert result.values["status"] == NOT_EVALUABLE


def test_t20_is_false_when_an_advertised_capture_will_not_verify(tmp_path):
    snapshots = tmp_path / "snapshots"

    write_capture(snapshots, error_count=3)

    result = check_t20_protocol_compatible(
        slate_date=SLATE, snapshot_dir=snapshots
    )

    assert result.passed is False
    assert result.values["verified_bundles"] == 0
    assert "contains errors" in result.error


def test_gate3_is_not_evaluable_without_a_capture(tmp_path):
    result = check_gate3_role_readiness(
        project_root=PROJECT,
        candidate=build_candidate(tmp_path),
        role_state_hash=hashlib.sha256(b"role").hexdigest(),
        slate_date=SLATE,
        snapshot_dir=None,
    )

    assert result.passed is None
    assert result.values["status"] == NOT_EVALUABLE


def test_gate3_passes_against_a_real_lineup_capture(tmp_path):
    snapshots = tmp_path / "snapshots"

    write_capture(snapshots)

    result = check_gate3_role_readiness(
        project_root=PROJECT,
        candidate=build_candidate(tmp_path),
        role_state_hash=hashlib.sha256(b"role").hexdigest(),
        slate_date=SLATE,
        snapshot_dir=snapshots,
    )

    assert result.passed is True
    assert result.values["lineup_record_count"] == 2
    assert result.values["candidate_id"]


def test_gate3_is_false_when_the_capture_carries_no_lineups(tmp_path):
    snapshots = tmp_path / "snapshots"

    write_capture(snapshots, lineups=False)

    result = check_gate3_role_readiness(
        project_root=PROJECT,
        candidate=build_candidate(tmp_path),
        role_state_hash=hashlib.sha256(b"role").hexdigest(),
        slate_date=SLATE,
        snapshot_dir=snapshots,
    )

    assert result.passed is False
    assert "no lineup records" in result.error


def test_gate3_is_false_without_a_recorded_role_state(tmp_path):
    snapshots = tmp_path / "snapshots"

    write_capture(snapshots)

    result = check_gate3_role_readiness(
        project_root=PROJECT,
        candidate=build_candidate(tmp_path),
        role_state_hash=None,
        slate_date=SLATE,
        snapshot_dir=snapshots,
    )

    assert result.passed is False
    assert "role_state_hash" in result.error


def test_a_captured_slate_leaves_nothing_deferred(tmp_path):
    """The one condition under which the registry's missing set empties."""
    snapshots = tmp_path / "snapshots"

    write_capture(snapshots)

    report = report_for(build_candidate(tmp_path), snapshot_dir=snapshots)

    assert report.deferred() == []
    assert report.failed() == []
    assert set(report.recorded_checks()) == set(REQUIRED_VALIDATION_CHECKS)


# ----------------------------------------------------------------------
# a check that raises is a failure, never a silent omission
# ----------------------------------------------------------------------


def test_a_raising_check_is_recorded_as_a_failure(tmp_path):
    candidate = build_candidate(tmp_path)

    # A list where a digest mapping belongs makes the check raise rather than
    # answer. A fit whose validation crashed has proved nothing, so the answer
    # must be a recorded failure and not a silently absent check.
    report = report_for(candidate, artifact_hashes=["models/marginals.joblib"])

    result = report.by_name["artifact_hashes_valid"]

    assert result.passed is False
    assert result.evidence == "the check raised while being computed"
    assert "AttributeError" in result.error


def test_a_malformed_digest_is_a_failure_not_a_crash(tmp_path):
    candidate = build_candidate(tmp_path)

    report = report_for(
        candidate, artifact_hashes={"models/marginals.joblib": None}
    )

    result = report.by_name["artifact_hashes_valid"]

    assert result.passed is False
    assert result.values["malformed"] == ["models/marginals.joblib"]
