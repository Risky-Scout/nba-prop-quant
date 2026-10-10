"""Tests for installing the frozen model binaries the incumbent serves from.

The defect under test is a deployment one, not a numerical one. The frozen
manifest records twelve ``.joblib`` files that prediction and pricing load;
``.gitignore`` excludes ``models/**/*.joblib`` because the frozen package is
distributed as a release asset; so verifying the frozen bundle against a Git
checkout asked a question no checkout can answer. The property established
here is that the already-published package is what supplies them, that every
installed byte is checked against the hashes the manifest already records, and
that a bundle which is incomplete, drifted or half-written never serves.

Nothing here builds a model package, regenerates an artifact, trains, promotes
or publishes. Every package is a few hundred bytes of fixture, every runtime
root is a pytest temporary directory, and the repository is only read from.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from nba_prop_quant.adaptive_fit_registry import (
    CONFIG_RELATIVE_PATH,
    CONTRACT_RELATIVE_PATH,
    FEATURES_RELATIVE_PATH,
    FROZEN_POLICY_SOURCES,
    REQUIRED_VALIDATION_CHECKS,
    FitMetadata,
    FitRegistry,
)
from nba_prop_quant.research.game_latent_state.safety import (
    SERVING_ENTRY_POINTS,
    serving_reachable_sources,
)

REPO = Path(__file__).resolve().parents[1]

INSTALLER_PATH = REPO / "ops" / "install_frozen_model_artifacts.py"

SERVING_PATH = REPO / "ops" / "run_incumbent_production_serving.py"

VALIDATOR_PATH = REPO / "ops" / "validate_main_scheduler_role.py"

LIFECYCLE_WORKFLOW = (
    REPO / ".github" / "workflows" / "nba_production_lifecycle.yml"
)

FROZEN_MANIFEST = REPO / "models" / "frozen_manifests" / "LATEST.json"

INSTALL_STEP_NAME = "Install the verified frozen runtime bundle"

SERVING_STEP_NAME = "Serve the slate with the incumbent"

#: The 2025 out-of-fold and market-backtest outputs the freeze recorded as
#: provenance. They are training and audit evidence: no serving entry point
#: and nothing in its import closure reads them, which is why they are not a
#: prerequisite for serving tonight's slate.
AUDIT_ONLY_DATA_FILES: tuple[str, ...] = (
    "data/processed/oof_selected_means.parquet",
    "data/processed/selected_means_distribution_split.parquet",
    "data/processed/market_backtest/calibration_walkforward/"
    "pooled_event_metrics.csv",
    "data/processed/market_backtest/calibration_walkforward/"
    "calibration_selection_audit.csv",
    "data/processed/market_backtest/calibrated_oof/"
    "event_scoring_summary.csv",
    "data/processed/market_backtest/calibrated_oof/"
    "event_game_cluster_bootstrap.csv",
)

#: Everything the registry re-derives the frozen choices from, so a throwaway
#: project is enough to register a candidate.
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


installer = _load("install_frozen_model_artifacts_under_test", INSTALLER_PATH)

serving = _load("run_incumbent_production_serving_under_test", SERVING_PATH)

validator = _load("validate_main_scheduler_role_under_test", VALIDATOR_PATH)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ----------------------------------------------------------------------
# a frozen package of the shape ops/build_full_model_package.py produces
# ----------------------------------------------------------------------


PACKAGE_NAME = "nba_prop_quant_20261009T000000Z_production_model_package"

FREEZE_ID = "nba_prop_quant_20261009T000000Z"


def _payloads() -> dict[str, bytes]:
    """The model tree a frozen package carries: the binaries plus the JSON."""
    payloads = {
        name: f"frozen-{name}".encode("utf-8")
        for name in installer.REQUIRED_MODEL_BINARIES
    }

    payloads["dynamic_params.json"] = b'{"role_scale": "frozen"}\n'
    payloads["ensemble_weights.json"] = b'{"weights": "frozen"}\n'

    return payloads


def _manifest_bytes(payloads: dict[str, bytes]) -> bytes:
    """A frozen manifest over those artifacts, plus the freeze's provenance.

    The ``source_files`` and ``data_audit_files`` groups carry hashes nothing
    in the runtime root can satisfy, deliberately: they are what the real
    manifest also carries, and the serving verification must not treat them as
    runtime prerequisites.
    """
    manifest = {
        "created_utc": "2026-10-09T00:00:00+00:00",
        "files": {
            "data_audit_files": [
                {"path": relative, "sha256": "a" * 64}
                for relative in AUDIT_ONLY_DATA_FILES
            ],
            "model_artifacts": [
                {
                    "path": f"{installer.MODEL_PREFIX}/{name}",
                    "sha256": sha256_bytes(payload),
                }
                for name, payload in sorted(payloads.items())
            ],
            "source_files": [
                {"path": "src/nba_prop_quant/production.py", "sha256": "b" * 64},
                {"path": "src/nba_prop_quant/slate.py", "sha256": "c" * 64},
            ],
            "scripts": [
                {"path": "scripts/10_predict_slate.py", "sha256": "d" * 64},
                {"path": "scripts/15_price_markets.py", "sha256": "e" * 64},
            ],
        },
        "freeze_id": FREEZE_ID,
        "freeze_stage": installer.REQUIRED_FREEZE_STAGE,
    }

    return (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


@pytest.fixture
def frozen_release(tmp_path: Path):
    """A factory for a frozen package and the manifest that describes it."""

    def build(
        *,
        omit: str | None = None,
        corrupt: str | None = None,
        label: str = "release",
    ) -> dict[str, Path]:
        payloads = _payloads()

        manifest_bytes = _manifest_bytes(payloads)

        root = tmp_path / label

        root.mkdir(parents=True, exist_ok=True)

        # The checkout's own copy: the authoritative manifest, which is all a
        # clean checkout holds. No model binary beside it, exactly as Git has
        # it.
        checkout_manifest = (
            root
            / "checkout"
            / installer.MODEL_PREFIX
            / installer.MANIFEST_RELATIVE
        )

        checkout_manifest.parent.mkdir(parents=True, exist_ok=True)

        checkout_manifest.write_bytes(manifest_bytes)

        if omit is not None:
            payloads.pop(omit)

        if corrupt is not None:
            payloads[corrupt] = b"tampered"

        package = root / f"{PACKAGE_NAME}.zip"

        prefix = (
            f"{PACKAGE_NAME}/{installer.PACKAGE_PROJECT_PREFIX}/"
            f"{installer.MODEL_PREFIX}"
        )

        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(f"{prefix}/{name}", payload)

            archive.writestr(
                f"{prefix}/{installer.MANIFEST_RELATIVE.as_posix()}",
                manifest_bytes,
            )

        return {
            "manifest": checkout_manifest,
            "package": package,
            "runtime_root": root / "work",
        }

    return build


def install(release: dict[str, Path]) -> dict:
    return installer.install(
        runtime_root=release["runtime_root"],
        manifest_path=release["manifest"],
        package=release["package"],
        allow_download=False,
    )


# ----------------------------------------------------------------------
# 1. a clean checkout has no model binaries, and hydration supplies them
# ----------------------------------------------------------------------


def test_a_clean_checkout_cannot_hold_the_binaries_and_hydration_supplies_them(
    frozen_release,
):
    """The original failure's cause, and the fix for it, in one statement.

    The repository under test is a real clean checkout, and the twelve model
    binaries the frozen manifest requires are absent from it and ignored by
    design. The installed bundle holds all twelve, outside the checkout.
    """
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")

    assert "models/**/*.joblib" in ignore

    absent = [
        name
        for name in installer.REQUIRED_MODEL_BINARIES
        if not (REPO / installer.MODEL_PREFIX / name).exists()
    ]

    assert absent == list(installer.REQUIRED_MODEL_BINARIES), (
        "this test states what a clean checkout looks like; a checkout "
        "carrying model binaries is not the case under test"
    )

    release = frozen_release()

    receipt = install(release)

    assert receipt["outcome"] == installer.OUTCOME_INSTALLED

    model_dir = Path(receipt["model_dir"])

    assert REPO not in model_dir.parents

    for name in installer.REQUIRED_MODEL_BINARIES:
        assert (model_dir / name).exists(), name


# ----------------------------------------------------------------------
# 2. a complete, correct bundle passes serving verification
# ----------------------------------------------------------------------


def test_a_complete_bundle_passes_the_serving_verification(frozen_release):
    release = frozen_release()

    receipt = install(release)

    metadata = installer.verify_runtime_model_artifacts(receipt["model_dir"])

    assert metadata["freeze_id"] == FREEZE_ID
    assert metadata["freeze_stage"] == installer.REQUIRED_FREEZE_STAGE
    assert metadata["verified_artifacts"] == len(_payloads())
    assert metadata["manifest_sha256"] == receipt["manifest_sha256"]


# ----------------------------------------------------------------------
# 3. a required artifact the package does not carry is a refusal
# ----------------------------------------------------------------------


def test_a_missing_required_model_binary_fails_closed(frozen_release):
    """Fail closed, and closed means nothing is installed to serve from."""
    release = frozen_release(omit="pts.joblib")

    with pytest.raises(RuntimeError) as caught:
        install(release)

    assert "pts.joblib" in str(caught.value)

    bundles = release["runtime_root"] / installer.BUNDLES_RELATIVE

    assert not [path for path in bundles.glob("*") if path.name != ".staging"]


# ----------------------------------------------------------------------
# 4. a required artifact whose bytes drifted is a refusal
# ----------------------------------------------------------------------


def test_a_corrupted_required_model_binary_fails_closed(frozen_release):
    """The hashes the manifest already records are what decides this."""
    release = frozen_release(corrupt="copula.joblib")

    with pytest.raises(RuntimeError) as caught:
        install(release)

    message = str(caught.value)

    assert "hash_mismatch" in message
    assert "copula.joblib" in message

    bundles = release["runtime_root"] / installer.BUNDLES_RELATIVE

    assert not [path for path in bundles.glob("*") if path.name != ".staging"]


# ----------------------------------------------------------------------
# 5. a half-written runtime directory is never treated as valid
# ----------------------------------------------------------------------


def test_a_partial_runtime_directory_is_never_treated_as_valid(
    frozen_release,
):
    """Both shapes an interrupted install could leave behind.

    A directory with no readiness marker is debris, and a marked directory
    that has since lost an artifact is no longer the bundle it claims to be.
    Each is rebuilt from the frozen package rather than served.
    """
    release = frozen_release()

    receipt = install(release)

    bundle = Path(receipt["bundle_root"])

    expected = {
        "bundle_key": receipt["bundle_key"],
        "freeze_id": receipt["freeze_id"],
        "manifest_sha256": receipt["manifest_sha256"],
        "package_sha256": receipt["frozen_package"]["sha256"],
    }

    assert installer.installed_bundle_is_valid(bundle, expected)

    (bundle / installer.READY_RELATIVE).unlink()

    assert not installer.installed_bundle_is_valid(bundle, expected)

    rebuilt = install(release)

    assert rebuilt["outcome"] == installer.OUTCOME_INSTALLED
    assert installer.installed_bundle_is_valid(bundle, expected)

    (Path(rebuilt["model_dir"]) / "minutes.joblib").unlink()

    assert not installer.installed_bundle_is_valid(bundle, expected)

    with pytest.raises(RuntimeError, match="minutes.joblib"):
        installer.verify_runtime_model_artifacts(rebuilt["model_dir"])

    assert install(release)["outcome"] == installer.OUTCOME_INSTALLED
    assert installer.installed_bundle_is_valid(bundle, expected)


# ----------------------------------------------------------------------
# 6. an already verified copy is re-verified and reused
# ----------------------------------------------------------------------


def test_an_already_verified_bundle_is_reused_rather_than_rebuilt(
    frozen_release,
):
    """Idempotent, and keyed to what is installed rather than to a timestamp."""
    release = frozen_release()

    first = install(release)

    stamp = (Path(first["model_dir"]) / "pts.joblib").stat().st_mtime_ns

    second = install(release)

    assert second["outcome"] == installer.OUTCOME_REUSED
    assert second["bundle_root"] == first["bundle_root"]
    assert second["bundle_key"] == first["bundle_key"]
    assert second["verified_artifacts"] == first["verified_artifacts"]

    assert (
        Path(second["model_dir"]) / "pts.joblib"
    ).stat().st_mtime_ns == stamp, "the reused copy was re-extracted"

    assert FREEZE_ID in first["bundle_key"]
    assert first["manifest_sha256"][:16] in first["bundle_key"]
    assert first["frozen_package"]["sha256"][:16] in first["bundle_key"]


# ----------------------------------------------------------------------
# 7. the incumbent is handed the verified model directory explicitly
# ----------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"

    for relative in DERIVATION_SOURCES:
        destination = root / relative

        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(REPO / relative, destination)

    return root


def _preseason_refresh(tmp_path: Path) -> Path:
    path = tmp_path / "refresh.json"

    path.write_text(
        json.dumps(
            {
                "outcome": "PRESEASON_BLOCK",
                "run_adaptive_fit": False,
                "writes_performed": False,
            }
        ),
        encoding="utf-8",
    )

    return path


def test_incumbent_serving_receives_the_explicit_verified_model_dir(
    tmp_path: Path, project: Path, frozen_release
):
    """No implicit default, and the receipt names the bundle it verified."""
    with pytest.raises(SystemExit):
        serving.parse_args(
            [
                "--slate-date",
                "2026-11-15",
                "--data-root",
                str(tmp_path / "data"),
            ]
        )

    release = frozen_release()

    receipt = install(release)

    model_dir = Path(receipt["model_dir"])

    served = serving.run(
        serving.parse_args(
            [
                "--slate-date",
                "2026-11-15",
                "--data-root",
                str(tmp_path / "data"),
                "--model-dir",
                str(model_dir),
                "--project-root",
                str(project),
                "--registry-root",
                str(tmp_path / "registry"),
                "--refresh-status",
                str(_preseason_refresh(tmp_path)),
            ]
        )
    )

    assert served["serving_model_dir"] == str(model_dir)
    assert served["incumbent_version"] == FREEZE_ID
    assert served["model_authority"] == serving.MODEL_AUTHORITY
    assert project not in model_dir.parents

    step = _lifecycle_step(SERVING_STEP_NAME)

    assert f"--model-dir \"${installer.MODEL_DIR_VARIABLE}\"" in step["run"]

    install_step = _lifecycle_step(INSTALL_STEP_NAME)

    assert "--github-env" in install_step["run"]
    assert installer.MODEL_DIR_VARIABLE == "FROZEN_RUNTIME_MODEL_DIR"

    names = [entry.get("name") for entry in _lifecycle_steps()]

    assert names.index(INSTALL_STEP_NAME) < names.index(SERVING_STEP_NAME)

    assert "continue-on-error" not in install_step


# ----------------------------------------------------------------------
# 8. an unpromoted candidate is still never what serves
# ----------------------------------------------------------------------


def test_an_unpromoted_candidate_is_still_never_the_incumbent(
    tmp_path: Path, project: Path, frozen_release
):
    """The authority property the new model directory must not loosen."""
    release = frozen_release()

    model_dir = Path(install(release)["model_dir"])

    registry = FitRegistry(root=tmp_path / "registry", project_root=project)

    staged = tmp_path / "candidate"

    (staged / "models").mkdir(parents=True)

    (staged / "models" / "marginals.joblib").write_bytes(b"candidate")

    (staged / "training_data_manifest.json").write_text(
        json.dumps({"fitting_information_digest": "b" * 64}),
        encoding="utf-8",
    )

    candidate = registry.register(
        staged,
        FitMetadata(
            fit_date="2026-11-15",
            training_cutoff="2026-11-14",
            source_commit_sha="b7bb2e41a9efa296e776c619ed286a7f42f5e44d",
            training_data_manifest_hash="a" * 64,
            python_version="3.12.3",
            package_versions={
                "pandas": "3.0.5",
                "scikit-learn": "1.9.0",
                "xgboost": "3.4.1",
            },
            core_seed=73,
            gate3_seed=20260830,
            effective_n_jobs=2,
            gate3_candidate_policy_id="nba_prop_quant_v2_gate3_dd2d394b6def",
        ),
    )

    registry.record_validation(
        candidate,
        {name: True for name in REQUIRED_VALIDATION_CHECKS},
        actor="pytest",
    )

    incumbent = serving.resolve_incumbent(
        registry=registry, model_dir=model_dir
    )

    assert incumbent["authority"] == serving.AUTHORITY_FROZEN_BUNDLE
    assert incumbent["fit_id"] is None
    assert incumbent["version"] == FREEZE_ID
    assert candidate not in json.dumps(incumbent)


# ----------------------------------------------------------------------
# 9. the audit-only data files are not a serving prerequisite
# ----------------------------------------------------------------------


def test_the_audit_only_data_files_do_not_block_serving(frozen_release):
    """They are provenance for the freeze, not inputs to tonight's slate.

    Their hashes stay in the frozen manifest and are not removed; the serving
    verification simply does not resolve them, because no serving entry point
    or module in its import closure reads them.
    """
    recorded = {
        record["path"]
        for record in json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))[
            "files"
        ]["data_audit_files"]
    }

    assert set(AUDIT_ONLY_DATA_FILES) == recorded

    closure = [
        REPO / relative
        for relative in (
            *SERVING_ENTRY_POINTS,
            *sorted(serving_reachable_sources(REPO)),
        )
    ]

    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in closure if path.is_file()
    )

    for relative in AUDIT_ONLY_DATA_FILES:
        assert Path(relative).name not in sources, relative

    release = frozen_release()

    receipt = install(release)

    for relative in AUDIT_ONLY_DATA_FILES:
        assert not (release["runtime_root"] / relative).exists()

    metadata = installer.verify_runtime_model_artifacts(receipt["model_dir"])

    assert metadata["verified_artifacts"] == len(_payloads())


# ----------------------------------------------------------------------
# 10. no publishing or promotion authority is introduced
# ----------------------------------------------------------------------


def _lifecycle_steps() -> list[dict]:
    workflow = yaml.safe_load(LIFECYCLE_WORKFLOW.read_text(encoding="utf-8"))

    return [
        step
        for step in workflow["jobs"]["lifecycle"]["steps"]
        if isinstance(step, dict)
    ]


def _lifecycle_step(name: str) -> dict:
    for step in _lifecycle_steps():
        if step.get("name") == name:
            return step

    raise AssertionError(f"no {name!r} step in the lifecycle")


def test_the_install_introduces_no_publishing_or_promotion_authority():
    source = INSTALLER_PATH.read_text(encoding="utf-8")

    for forbidden in (
        ".promote(",
        ".rollback(",
        "publish_",
        "_publish",
        "--publish",
        "promotion_state",
        "current_good_fit_id",
        *validator.PUBLISHING_ACTIVATION_TOKENS,
    ):
        assert forbidden not in source, forbidden

    run = _lifecycle_step(INSTALL_STEP_NAME)["run"]

    for forbidden in validator.PUBLISHING_ACTIVATION_TOKENS:
        assert forbidden not in run, forbidden

    assert "--runtime-root" in run

    # The install writes into the durable production work root and nowhere
    # else. A model binary copied into the checkout would be an ignored file
    # the next `actions/checkout` deletes, and a committed one would be the
    # thing the distribution plan exists to prevent.
    assert str(installer.BUNDLES_RELATIVE) in source
    assert "$NBA_PROP_WORK_DIR" in run


def test_the_install_step_is_a_required_blocking_ungated_lifecycle_step():
    """The guard that watches the scheduler copy knows about the new step."""
    entry_point = "ops/install_frozen_model_artifacts.py"

    assert INSTALL_STEP_NAME in validator.REQUIRED_STEPS
    assert entry_point in validator.REQUIRED_ENTRY_POINTS
    assert entry_point in validator.BLOCKING_ENTRY_POINTS
    assert entry_point in validator.RUN_ADAPTIVE_UNGATED_ENTRY_POINTS
    assert entry_point not in validator.NON_BLOCKING_ENTRY_POINTS
    assert entry_point not in validator.RUN_ADAPTIVE_GATED_ENTRY_POINTS

    condition = str(_lifecycle_step(INSTALL_STEP_NAME).get("if", ""))

    assert "MODE" in condition
    assert "RUN_ADAPTIVE" not in condition
