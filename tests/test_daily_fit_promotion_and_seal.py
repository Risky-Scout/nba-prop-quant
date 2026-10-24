"""The daily fit has to be promotable in the morning, and then sealed.

Four claims, and they are the four that made the lifecycle impossible:

    a daily fit can be promoted from information that exists when the fit
    exists
        no check required to promote may need a capture taken hours later.

    no safety invariant disappeared in making that true
        the two checks that moved are refused at the serving window, where
        the capture they ask about exists, and a recorded failure still
        blocks promotion.

    a promoted fit becomes a verified runtime bundle automatically
        and the bundle holds exactly the fit's bytes, which is what
        ``resolve_incumbent`` compares.

    a bundle that will not verify never becomes the serving authority
        the previous bundle is still there and no pointer moved.

Nothing here trains, refits, prices or publishes. Every registry and bundle
lives in a pytest temporary directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nba_prop_quant.adaptive_fit_registry import (
    FIT_TIME_VALIDATION_CHECKS,
    REQUIRED_VALIDATION_CHECKS,
    SERVING_WINDOW_VALIDATION_CHECKS,
    FitMetadata,
    FitRegistry,
    ValidationRefused,
)

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO / "ops"))

import seal_promoted_fit_bundle as sealer  # noqa: E402

DERIVATION_SOURCES = (
    "configs/model.yaml",
    "configs/features.yaml",
    "models/frozen_manifests/nba_prop_quant_v2_adaptive_architecture_contract.json",
    "models/frozen_manifests/nba_prop_quant_v2_adaptive_serving_source_contract.json",
    "models/frozen_manifests/nba_prop_quant_v2_adaptive_update_protocol.json",
    "models/marginal_selection.json",
    "models/mean_model_selection.json",
    "models/market_probability_calibration_policy.json",
    "models/combo_dependence_policy.json",
    "research/v2_gate3_deployment_artifacts/deployment_manifest.json",
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"

    for relative in DERIVATION_SOURCES:
        source = REPO / relative

        if not source.exists():
            continue

        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    return root


def make_staged(path: Path, marker: bytes = b"fitted-day-1") -> Path:
    (path / "models").mkdir(parents=True, exist_ok=True)
    (path / "models" / "target_pts.joblib").write_bytes(marker)
    (path / "models" / "marginals.joblib").write_bytes(b"zinb-" + marker)
    (path / "calibration.json").write_text(
        json.dumps({"points": {"slope": 0.41}}), encoding="utf-8"
    )

    return path


def make_metadata(**overrides) -> FitMetadata:
    payload = {
        "fit_date": "2026-11-15",
        "training_cutoff": "2026-11-14",
        "source_commit_sha": "b7bb2e41a9efa296e776c619ed286a7f42f5e44d",
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
        "gate3_candidate_policy_id": "nba_prop_quant_v2_gate3_dd2d394b6def",
    }
    payload.update(overrides)

    return FitMetadata(**payload)


@pytest.fixture
def registry(tmp_path: Path) -> FitRegistry:
    return FitRegistry(root=tmp_path / "registry", project_root=REPO)


def fit_time_only() -> dict[str, bool]:
    """What a morning fit can honestly report: thirteen answers, not fifteen."""
    return {name: True for name in FIT_TIME_VALIDATION_CHECKS}


# --------------------------------------------------------------------------
# the partition itself
# --------------------------------------------------------------------------


def test_the_canonical_gate_set_is_unchanged_and_exactly_partitioned():
    assert len(REQUIRED_VALIDATION_CHECKS) == 15
    assert len(FIT_TIME_VALIDATION_CHECKS) == 13
    assert len(SERVING_WINDOW_VALIDATION_CHECKS) == 2

    assert set(FIT_TIME_VALIDATION_CHECKS) | set(
        SERVING_WINDOW_VALIDATION_CHECKS
    ) == set(REQUIRED_VALIDATION_CHECKS)

    assert not set(FIT_TIME_VALIDATION_CHECKS) & set(
        SERVING_WINDOW_VALIDATION_CHECKS
    )


def test_the_window_gates_are_the_two_that_need_a_capture():
    assert SERVING_WINDOW_VALIDATION_CHECKS == (
        "gate3_role_readiness",
        "t20_protocol_compatible",
    )


# --------------------------------------------------------------------------
# promotion does not require future information
# --------------------------------------------------------------------------


def test_a_morning_fit_promotes_on_the_thirteen_it_can_answer(
    registry: FitRegistry, tmp_path: Path
):
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())

    registry.record_validation(fit_id, fit_time_only(), actor="pytest")

    registry.promote(fit_id, reason="morning fit")

    assert registry.current()["current_good_fit_id"] == fit_id


def test_promotion_asks_for_no_check_that_needs_a_capture(
    registry: FitRegistry, tmp_path: Path
):
    """The regression that made the daily lifecycle impossible."""
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())

    registry.record_validation(fit_id, fit_time_only(), actor="pytest")

    record = registry.load_validation(fit_id)

    assert record["serving_window_missing"] == sorted(
        SERVING_WINDOW_VALIDATION_CHECKS
    )
    assert record["fit_time_passed"] is True

    registry.assert_validation_pass(fit_id)


@pytest.mark.parametrize("missing", FIT_TIME_VALIDATION_CHECKS)
def test_every_fit_time_gate_is_still_required_to_promote(
    registry: FitRegistry, tmp_path: Path, missing: str
):
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())

    checks = fit_time_only()
    checks.pop(missing)

    registry.record_validation(fit_id, checks, actor="pytest")

    with pytest.raises(ValidationRefused, match=missing):
        registry.assert_validation_pass(fit_id)


@pytest.mark.parametrize("failing", FIT_TIME_VALIDATION_CHECKS)
def test_a_failed_fit_time_gate_still_refuses_promotion(
    registry: FitRegistry, tmp_path: Path, failing: str
):
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())

    checks = fit_time_only()
    checks[failing] = False

    registry.record_validation(fit_id, checks, actor="pytest")

    with pytest.raises(ValidationRefused, match=failing):
        registry.assert_validation_pass(fit_id)


@pytest.mark.parametrize("failing", SERVING_WINDOW_VALIDATION_CHECKS)
def test_a_measured_window_failure_still_refuses_promotion(
    registry: FitRegistry, tmp_path: Path, failing: str
):
    """Moving a check to a later stage is not permission to ignore a failure."""
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())

    checks = fit_time_only()
    checks[failing] = False

    registry.record_validation(fit_id, checks, actor="pytest")

    with pytest.raises(ValidationRefused, match=failing):
        registry.assert_validation_pass(fit_id)


# --------------------------------------------------------------------------
# the serving window still refuses, fail-closed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SERVING_WINDOW_VALIDATION_CHECKS)
def test_the_serving_window_refuses_an_unanswered_window_gate(
    registry: FitRegistry, tmp_path: Path, name: str
):
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())

    checks = {name_: True for name_ in REQUIRED_VALIDATION_CHECKS}
    checks.pop(name)

    registry.record_validation(fit_id, checks, actor="pytest")

    # Promotion is content: the fit itself is sound.
    registry.assert_validation_pass(fit_id)

    # The window is not. Absent is a refusal here, which is the invariant
    # promotion used to carry.
    with pytest.raises(ValidationRefused, match=name):
        registry.assert_serving_window_validation_pass(fit_id)


@pytest.mark.parametrize("name", SERVING_WINDOW_VALIDATION_CHECKS)
def test_the_serving_window_refuses_a_failed_window_gate(
    registry: FitRegistry, tmp_path: Path, name: str
):
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())

    checks = {name_: True for name_ in REQUIRED_VALIDATION_CHECKS}
    checks[name] = False

    registry.record_validation(fit_id, checks, actor="pytest")

    with pytest.raises(ValidationRefused, match=name):
        registry.assert_serving_window_validation_pass(fit_id)


def test_the_serving_window_passes_only_when_both_are_recorded_true(
    registry: FitRegistry, tmp_path: Path
):
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())

    registry.record_validation(
        fit_id,
        {name: True for name in REQUIRED_VALIDATION_CHECKS},
        actor="pytest",
    )

    registry.assert_serving_window_validation_pass(fit_id)


def test_the_window_gate_entry_point_refuses_a_not_evaluable_answer():
    """NOT_EVALUABLE at a window that has just captured is a refusal."""
    sys.path.insert(0, str(REPO / "ops"))

    import assert_serving_window_gates as gates

    measurement = {
        "checks": {
            "t20_protocol_compatible": {
                "error": None,
                "evidence": "no scheduled T-20m capture window exists",
                "passed": None,
            }
        },
        "enforced": ["t20_protocol_compatible"],
        "slate_date": "2026-10-24",
    }

    with pytest.raises(gates.WindowGateRefused, match="cannot also have no"):
        gates.assert_window_gates(measurement)


def test_the_window_gate_entry_point_refuses_a_failed_answer():
    import assert_serving_window_gates as gates

    measurement = {
        "checks": {
            "gate3_role_readiness": {
                "error": "role_state_hash is absent or malformed",
                "evidence": "measured",
                "passed": False,
            }
        },
        "enforced": ["gate3_role_readiness"],
        "slate_date": "2026-10-24",
    }

    with pytest.raises(gates.WindowGateRefused, match="role_state_hash"):
        gates.assert_window_gates(measurement)


def test_the_window_gate_entry_point_passes_a_measured_pass():
    import assert_serving_window_gates as gates

    gates.assert_window_gates(
        {
            "checks": {
                "t20_protocol_compatible": {
                    "error": None,
                    "evidence": "verified 4 of 4",
                    "passed": True,
                }
            },
            "enforced": ["t20_protocol_compatible"],
            "slate_date": "2026-10-24",
        }
    )


# --------------------------------------------------------------------------
# sealing
# --------------------------------------------------------------------------


def make_base_bundle(root: Path) -> Path:
    """A minimal bundle shaped exactly like the one serving verifies."""
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)

    binaries = {
        name: f"frozen-{name}".encode()
        for name in sealer.REQUIRED_MODEL_BINARIES
    }

    for name, payload in binaries.items():
        (models / name).write_bytes(payload)

    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "model.yaml").write_text("seed: 73\n", encoding="utf-8")

    records = [
        {
            "path": f"models/{name}",
            "sha256": sealer.sha256_file(models / name),
        }
        for name in binaries
    ]

    manifest = {
        "freeze_id": "nba_prop_quant_20260818T205213Z",
        "freeze_stage": "external_test_deployment",
        "files": {
            "model_artifacts": records,
            "configs": [
                {
                    "path": "configs/model.yaml",
                    "sha256": sealer.sha256_file(
                        root / "configs" / "model.yaml"
                    ),
                }
            ],
        },
    }

    manifest_path = models / "frozen_manifests" / "LATEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return root


def seal_one(
    registry: FitRegistry, tmp_path: Path, marker: bytes = b"day-1"
) -> tuple[str, dict]:
    staged = tmp_path / "staged"
    (staged / "models").mkdir(parents=True, exist_ok=True)

    # A daily fit re-estimates the binaries the frozen manifest already
    # records, so the overlay is a digest correction rather than an addition.
    for name in ("marginals.joblib", "copula.joblib"):
        (staged / "models" / name).write_bytes(b"refit-" + marker + name.encode())

    (staged / "calibration.json").write_text("{}", encoding="utf-8")

    fit_id = registry.register(staged, make_metadata())

    registry.record_validation(fit_id, fit_time_only(), actor="pytest")
    registry.promote(fit_id, reason="test")

    base = make_base_bundle(tmp_path / "base")

    receipt = sealer.seal(
        registry=registry,
        fit_id=fit_id,
        base_bundle_root=base,
        runtime_root=tmp_path / "runtime",
    )

    return fit_id, receipt


def test_a_promoted_fit_seals_into_a_bundle_that_verifies(
    registry: FitRegistry, tmp_path: Path
):
    fit_id, receipt = seal_one(registry, tmp_path)

    assert receipt["outcome"] == sealer.OUTCOME_SEALED
    assert receipt["fit_id"] == fit_id

    # The architecture freeze is unchanged: a daily fit moves parameters
    # under a frozen architecture, and the publishing switch's
    # approved_freeze_id has to stay true about the bundle.
    assert receipt["freeze_id"] == "nba_prop_quant_20260818T205213Z"
    assert receipt["freeze_stage"] == "external_test_deployment"

    sealer.verify_frozen_bundle(Path(receipt["bundle_root"]))
    sealer.verify_runtime_model_artifacts(Path(receipt["model_dir"]))


def test_the_sealed_bundle_holds_exactly_the_promoted_fits_bytes(
    registry: FitRegistry, tmp_path: Path
):
    """The comparison ``resolve_incumbent`` makes, made against the seal."""
    fit_id, receipt = seal_one(registry, tmp_path)

    manifest = registry.load_manifest(fit_id)

    model_dir = Path(receipt["model_dir"])

    for relative, digest in (manifest["artifact_hashes"] or {}).items():
        if Path(relative).parts[:1] != ("models",):
            continue

        served = model_dir / Path(*Path(relative).parts[1:])

        assert served.exists(), relative
        assert sealer.sha256_file(served) == digest, relative


def test_the_sealed_manifest_names_the_fit_it_was_sealed_from(
    registry: FitRegistry, tmp_path: Path
):
    fit_id, receipt = seal_one(registry, tmp_path)

    manifest = json.loads(
        (
            Path(receipt["model_dir"]) / "frozen_manifests" / "LATEST.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["daily_fit"]["fit_id"] == fit_id
    assert manifest["daily_fit"]["training_cutoff"] == "2026-11-14"
    assert manifest["daily_fit"]["corrected_artifacts"]


def test_sealing_the_same_fit_twice_reuses_and_re_verifies(
    registry: FitRegistry, tmp_path: Path
):
    fit_id, first = seal_one(registry, tmp_path)

    second = sealer.seal(
        registry=registry,
        fit_id=fit_id,
        base_bundle_root=tmp_path / "base",
        runtime_root=tmp_path / "runtime",
    )

    assert second["outcome"] == sealer.OUTCOME_REUSED
    assert second["bundle_root"] == first["bundle_root"]


def test_a_seal_that_cannot_verify_publishes_nothing(
    registry: FitRegistry, tmp_path: Path
):
    """A failed seal leaves the previous bundle and moves no pointer."""
    staged = tmp_path / "staged"
    (staged / "models").mkdir(parents=True, exist_ok=True)
    (staged / "models" / "marginals.joblib").write_bytes(b"refit")
    (staged / "calibration.json").write_text("{}", encoding="utf-8")

    fit_id = registry.register(staged, make_metadata())
    registry.record_validation(fit_id, fit_time_only(), actor="pytest")
    registry.promote(fit_id, reason="test")

    base = make_base_bundle(tmp_path / "base")

    # Corrupt the registry's own copy after registration, so the seal's
    # re-hash of the source disagrees with the manifest.
    target = registry.fit_dir(fit_id) / "artifacts" / "models" / "marginals.joblib"
    target.chmod(0o644)
    target.write_bytes(b"tampered")

    with pytest.raises((sealer.SealRefused, Exception)):
        sealer.seal(
            registry=registry,
            fit_id=fit_id,
            base_bundle_root=base,
            runtime_root=tmp_path / "runtime",
        )

    sealed = tmp_path / "runtime" / sealer.SEALED_RELATIVE / fit_id

    assert not (sealed / sealer.READY_RELATIVE).exists()

    # And the base bundle is untouched and still verifies.
    sealer.verify_frozen_bundle(base)


def test_an_unpromoted_registry_seals_nothing(tmp_path: Path):
    registry = FitRegistry(root=tmp_path / "registry", project_root=REPO)
    registry.initialise()

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "ops" / "seal_promoted_fit_bundle.py"),
            "--registry-root",
            str(tmp_path / "registry"),
            "--base-bundle-root",
            str(make_base_bundle(tmp_path / "base")),
            "--runtime-root",
            str(tmp_path / "runtime"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert sealer.OUTCOME_NO_PROMOTED_FIT in result.stdout


# --------------------------------------------------------------------------
# the promotion entry point
# --------------------------------------------------------------------------


def run_promotion(tmp_path: Path, registry_root: Path, **extra) -> dict:
    receipt = tmp_path / "promotion.json"

    command = [
        sys.executable,
        str(REPO / "ops" / "run_daily_fit_promotion.py"),
        "--slate-date",
        extra.pop("slate_date", "2026-11-15"),
        "--registry-root",
        str(registry_root),
        "--receipt-path",
        str(receipt),
    ]

    for key, value in extra.items():
        command += [f"--{key.replace('_', '-')}", str(value)]

    result = subprocess.run(command, capture_output=True, text=True)

    assert receipt.exists(), result.stdout + result.stderr

    return json.loads(receipt.read_text(encoding="utf-8"))


def test_the_entry_point_promotes_a_validated_morning_fit(
    registry: FitRegistry, tmp_path: Path
):
    fit_id = registry.register(make_staged(tmp_path / "d1"), make_metadata())
    registry.record_validation(fit_id, fit_time_only(), actor="pytest")

    receipt = run_promotion(tmp_path, registry.root)

    assert receipt["outcome"] == "PROMOTED"
    assert receipt["candidate_fit_id"] == fit_id
    assert receipt["training_cutoff"] == "2026-11-14"
    assert receipt["fallback_active"] is False
    assert receipt["gates"]["canonical_gate_count"] == 15
    assert receipt["gates"]["fit_time_gates_pass"] is True
    assert receipt["fit_artifact_digest"]

    assert registry.current()["current_good_fit_id"] == fit_id


def test_the_entry_point_keeps_the_last_good_fit_when_validation_fails(
    registry: FitRegistry, tmp_path: Path
):
    good = registry.register(make_staged(tmp_path / "d1", b"good"), make_metadata())
    registry.record_validation(good, fit_time_only(), actor="pytest")
    registry.promote(good, reason="yesterday")

    checks = fit_time_only()
    checks["calibration_valid"] = False

    bad = registry.register(
        make_staged(tmp_path / "d2", b"bad"),
        make_metadata(fit_date="2026-11-16", training_cutoff="2026-11-15"),
    )
    registry.record_validation(bad, checks, actor="pytest")

    receipt = run_promotion(
        tmp_path, registry.root, slate_date="2026-11-16", fit_id=bad
    )

    assert receipt["outcome"] == "PROMOTION_REFUSED"
    assert receipt["fallback_active"] is True
    assert receipt["fallback_reason"] == "DAILY_FIT_VALIDATION_FAILED"
    assert receipt["fallback_fit_id"] == good

    # Yesterday's promoted daily fit is the fallback. Not the frozen bundle.
    assert registry.current()["current_good_fit_id"] == good


def test_the_entry_point_reports_an_absent_fit_as_an_absent_fit(
    registry: FitRegistry, tmp_path: Path
):
    registry.initialise()

    receipt = run_promotion(tmp_path, registry.root)

    assert receipt["outcome"] == "NO_DAILY_FIT"
    assert receipt["fallback_reason"] == "NO_DAILY_FIT_REGISTERED"


def test_the_entry_point_never_reaches_a_fit_from_another_day(
    registry: FitRegistry, tmp_path: Path
):
    """A day whose fit is missing must not promote yesterday's as today's."""
    yesterday = registry.register(make_staged(tmp_path / "d1"), make_metadata())
    registry.record_validation(yesterday, fit_time_only(), actor="pytest")

    receipt = run_promotion(tmp_path, registry.root, slate_date="2026-11-16")

    assert receipt["outcome"] == "NO_DAILY_FIT"
    assert receipt["candidate_fit_id"] is None
