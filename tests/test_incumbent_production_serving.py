"""Tests for the automated incumbent prediction and pricing path.

The property under test is an authority property, not a numerical one: the
step that prices tonight's slate must use the fit somebody approved, and must
refuse rather than substitute when it cannot establish which fit that is.

Nothing here trains, refits, promotes or publishes. Every registry and
serving tree lives in a pytest temporary directory; the repository is only
read from, and the two serving scripts are replaced by stubs so the
orchestration can be exercised without the historical corpus.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
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
    sha256_file,
)

REPO = Path(__file__).resolve().parents[1]

ENTRY_POINT = REPO / "ops" / "run_incumbent_production_serving.py"

LIFECYCLE_WORKFLOW = (
    REPO / ".github" / "workflows" / "nba_production_lifecycle.yml"
)

SERVING_STEP_NAME = "Serve the slate with the incumbent"

#: Everything the registry re-derives the frozen choices from. A throwaway
#: copy of exactly these files is enough to register a fit.
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


def _load_entry_point():
    """Import the ops script by path, the way the workflow invokes it."""
    spec = importlib.util.spec_from_file_location(
        "run_incumbent_production_serving", ENTRY_POINT
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


serving = _load_entry_point()


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


PREDICT_STUB = """
import sys
from pathlib import Path

date = sys.argv[sys.argv.index("--date") + 1]
root = Path(__import__("os").environ["NBA_PROP_DATA_DIR"])
out = root / "processed" / "projections" / f"{date}.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(b"projections")
"""

PRICE_STUB = """
import sys
from pathlib import Path

date = sys.argv[sys.argv.index("--date") + 1]
root = Path(__import__("os").environ["NBA_PROP_DATA_DIR"])
out = root / "processed" / "priced_markets" / f"{date}.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(b"priced-markets")
"""

NO_GAMES_STUB = """
print("No games found")
"""

FAILING_STUB = """
import sys

print("the slate could not be projected", file=sys.stderr)
raise SystemExit(3)
"""


def _write_stub(project: Path, relative: Path, body: str) -> None:
    path = project / relative

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(body, encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A throwaway project: the registry's derivation sources plus stubs."""
    root = tmp_path / "project"

    for relative in DERIVATION_SOURCES:
        destination = root / relative

        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(REPO / relative, destination)

    _write_stub(root, serving.PREDICT_SCRIPT, PREDICT_STUB)
    _write_stub(root, serving.PRICE_SCRIPT, PRICE_STUB)

    return root


def serving_tree(project: Path, *, artifacts: dict[str, bytes]) -> Path:
    """A frozen deployment bundle holding ``artifacts``.

    Built to the shape ``load_verified_manifest_metadata`` verifies: a
    LATEST.json at the external_test_deployment stage whose every named file
    hashes to what it records.
    """
    model_dir = project / "models"

    records = []

    for relative, payload in sorted(artifacts.items()):
        path = model_dir / relative

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_bytes(payload)

        records.append(
            {
                "path": f"models/{relative}",
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "created_utc": "2026-10-08T00:00:00+00:00",
        "files": {"model_artifacts": records},
        "freeze_id": "nba_prop_quant_20261008T000000Z",
        "freeze_stage": "external_test_deployment",
    }

    manifest_path = model_dir / "frozen_manifests" / "LATEST.json"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    return model_dir


@pytest.fixture
def registry(tmp_path: Path, project: Path) -> FitRegistry:
    return FitRegistry(root=tmp_path / "registry", project_root=project)


def metadata(**overrides) -> FitMetadata:
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


def staged_fit(path: Path, marker: bytes) -> Path:
    (path / "models").mkdir(parents=True, exist_ok=True)

    (path / "models" / "marginals.joblib").write_bytes(marker)

    (path / "training_data_manifest.json").write_text(
        json.dumps({"fitting_information_digest": "b" * 64}),
        encoding="utf-8",
    )

    return path


def register(
    registry: FitRegistry, path: Path, marker: bytes, **overrides
) -> str:
    fit_id = registry.register(staged_fit(path, marker), metadata(**overrides))

    registry.record_validation(
        fit_id,
        {name: True for name in REQUIRED_VALIDATION_CHECKS},
        actor="pytest",
    )

    return fit_id


def refreshed_data_root(tmp_path: Path) -> Path:
    """A data root the refresh has already fingerprinted.

    A REFRESHED day always leaves this behind, which is why the receipt is
    allowed to require the fingerprint.
    """
    root = tmp_path / "data"

    path = root / serving.SEASON_STATE_RELATIVE

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps({"datasets_fingerprint": "d" * 64}), encoding="utf-8"
    )

    return root


def refresh_status(tmp_path: Path, outcome: str) -> Path:
    path = tmp_path / f"refresh_{outcome}.json"

    path.write_text(
        json.dumps(
            {
                "outcome": outcome,
                "run_adaptive_fit": outcome == "REFRESHED",
                "writes_performed": outcome == "REFRESHED",
            }
        ),
        encoding="utf-8",
    )

    return path


def arguments(**overrides):
    defaults = {
        "data_root": None,
        "frozen_bundle_root": None,
        "model_dir": None,
        "production_sha": "c" * 40,
        "project_root": None,
        "receipt_path": None,
        "refresh_status": None,
        "registry_root": None,
        "slate_date": "2026-11-15",
        "summary_path": None,
    }

    defaults.update(overrides)

    return serving.parse_args(
        [
            argument
            for key, value in defaults.items()
            if value is not None
            for argument in (f"--{key.replace('_', '-')}", str(value))
        ]
    )


# ----------------------------------------------------------------------
# the incumbent is what gets served
# ----------------------------------------------------------------------


def test_the_frozen_bundle_is_the_incumbent_before_any_promotion(
    project: Path, registry: FitRegistry
):
    """The state production is actually in today.

    No adaptive fit has been promoted, so the sealed deployment bundle is the
    approved serving authority and the receipt says so by name.
    """
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    incumbent = serving.resolve_incumbent(
        registry=registry, model_dir=model_dir, bundle_root=model_dir.parent
    )

    assert incumbent["authority"] == serving.AUTHORITY_FROZEN_BUNDLE
    assert incumbent["fit_id"] is None
    assert incumbent["version"] == "nba_prop_quant_20261008T000000Z"


def test_a_registered_but_unpromoted_candidate_is_never_the_incumbent(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """Registration is not approval.

    This is the exact shape of the risk: the lifecycle registers a candidate
    every morning, and a serving path that resolved "the fit that exists"
    would start serving it the same day with nobody having approved anything.
    """
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    candidate = register(registry, tmp_path / "candidate", b"candidate-day-1")

    assert registry.list_fits() == [candidate]
    assert registry.current()["current_good_fit_id"] is None

    incumbent = serving.resolve_incumbent(
        registry=registry, model_dir=model_dir, bundle_root=model_dir.parent
    )

    assert incumbent["fit_id"] is None
    assert incumbent["version"] != candidate


def test_the_newest_fit_is_not_the_incumbent(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """Resolution is by promotion state, not by recency.

    The promoted fit is the older one and the serving tree holds its
    artifacts, so a resolver that sorted by timestamp or took the last
    directory would name the wrong one.
    """
    promoted = register(registry, tmp_path / "older", b"approved-weights")

    newer = register(
        registry,
        tmp_path / "newer",
        b"unapproved-weights",
        fit_date="2026-11-16",
        training_cutoff="2026-11-15",
    )

    registry.promote(promoted, reason="human review")

    model_dir = serving_tree(
        project, artifacts={"marginals.joblib": b"approved-weights"}
    )

    incumbent = serving.resolve_incumbent(
        registry=registry, model_dir=model_dir, bundle_root=model_dir.parent
    )

    assert incumbent["authority"] == serving.AUTHORITY_PROMOTED_FIT
    assert incumbent["fit_id"] == promoted
    assert incumbent["fit_id"] != newer
    assert newer not in json.dumps(incumbent)


def test_a_promoted_fit_must_actually_be_what_the_bundle_holds(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """A label is not evidence, so the digests are compared.

    Promotion approves a fit; it does not move any bytes. Serving the old
    bundle while reporting the promoted fit as the authority would be a
    provenance lie, so the mismatch is a refusal that names the human step.
    """
    promoted = register(registry, tmp_path / "fit", b"approved-weights")

    registry.promote(promoted, reason="human review")

    model_dir = serving_tree(
        project, artifacts={"marginals.joblib": b"something-else"}
    )

    with pytest.raises(serving.ServingRefused) as caught:
        serving.resolve_incumbent(
            registry=registry,
            model_dir=model_dir,
            bundle_root=model_dir.parent,
        )

    assert "has not been sealed" in str(caught.value)


def test_a_corrupt_promoted_fit_refuses_rather_than_substituting(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """Fail closed, and closed means nothing serves.

    The tempting failure mode is to notice the promoted fit is unusable and
    quietly fall back to the frozen bundle, which would serve an authority
    nobody asked for under a run that reported success.
    """
    promoted = register(registry, tmp_path / "fit", b"approved-weights")

    registry.promote(promoted, reason="human review")

    model_dir = serving_tree(
        project, artifacts={"marginals.joblib": b"approved-weights"}
    )

    artifact = (
        registry.fit_dir(promoted) / "artifacts" / "models" / "marginals.joblib"
    )

    artifact.write_bytes(b"tampered")

    with pytest.raises(serving.ServingRefused) as caught:
        serving.resolve_incumbent(
            registry=registry,
            model_dir=model_dir,
            bundle_root=model_dir.parent,
        )

    assert "did not verify" in str(caught.value)
    assert "nothing else may serve in its place" in str(caught.value)


def test_a_missing_incumbent_fails_closed(
    project: Path, registry: FitRegistry
):
    """No verifiable serving authority at all is a refusal."""
    with pytest.raises(FileNotFoundError):
        serving.resolve_incumbent(
            registry=registry,
            model_dir=project / "models",
            bundle_root=project,
        )


def test_a_corrupt_serving_bundle_fails_closed(
    project: Path, registry: FitRegistry
):
    """The bundle is re-hashed, not trusted."""
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    (model_dir / "marginals.joblib").write_bytes(b"drifted")

    with pytest.raises(RuntimeError, match="manifest verification failed"):
        serving.resolve_incumbent(
            registry=registry,
            model_dir=model_dir,
            bundle_root=model_dir.parent,
        )


# ----------------------------------------------------------------------
# the candidate and the shadow cannot reach this path
# ----------------------------------------------------------------------


def test_the_serving_path_never_reads_the_candidate_or_shadow_status():
    """Structural, because the risk is an accidental convenience read.

    ``adaptive.json`` names the candidate the lifecycle just registered and
    sits in the same run directory. Reading it for a slate date or a workspace
    is how a candidate identifier ends up in a served receipt.
    """
    source = ENTRY_POINT.read_text(encoding="utf-8")

    for forbidden in (
        "adaptive.json",
        "adaptive_status",
        "shadow.json",
        "shadow_status",
        "list_fits",
    ):
        assert forbidden not in source


def test_a_candidate_failure_cannot_change_what_the_incumbent_serves(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """An adaptive status claiming a failed candidate changes nothing.

    The behavioural half of the test above: the candidate's own record is
    present on disk, and the receipt still names only the approved authority.
    """
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    candidate = register(registry, tmp_path / "candidate", b"candidate")

    run_dir = tmp_path / "run"

    run_dir.mkdir()

    (run_dir / "adaptive.json").write_text(
        json.dumps({"fit_id": candidate, "outcome": "COMPLETED"}),
        encoding="utf-8",
    )

    (run_dir / "shadow.json").write_text(
        json.dumps({"outcome": "SHADOW_FAILED", "error": "candidate blew up"}),
        encoding="utf-8",
    )

    receipt = serving.run(
        arguments(
            data_root=refreshed_data_root(tmp_path),
            model_dir=model_dir,
            frozen_bundle_root=model_dir.parent,
            project_root=project,
            refresh_status=refresh_status(tmp_path, "REFRESHED"),
            registry_root=registry.root,
        )
    )

    assert receipt["outcome"] == serving.OUTCOME_SERVED
    assert receipt["incumbent_fit_id"] is None
    assert candidate not in json.dumps(receipt)


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------


def test_a_served_run_carries_complete_provenance(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """Every field the brief requires, computed rather than declared."""
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    data_root = tmp_path / "data"

    (data_root / ".state").mkdir(parents=True)

    (data_root / serving.SEASON_STATE_RELATIVE).write_text(
        json.dumps({"datasets_fingerprint": "d" * 64}), encoding="utf-8"
    )

    receipt = serving.run(
        arguments(
            data_root=data_root,
            model_dir=model_dir,
            frozen_bundle_root=model_dir.parent,
            project_root=project,
            refresh_status=refresh_status(tmp_path, "REFRESHED"),
            registry_root=registry.root,
        )
    )

    assert serving.missing_provenance(receipt) == []
    assert receipt["model_authority"] == "incumbent"
    assert receipt["production_code_sha"] == "c" * 40
    assert receipt["input_state_fingerprint"] == "d" * 64
    assert receipt["slate_date"] == "2026-11-15"
    assert receipt["prediction_artifact"]["sha256"]
    assert receipt["priced_market_artifact"]["sha256"]
    assert receipt["generated_at"].startswith("20")


def test_an_incomplete_receipt_is_a_failure_not_a_thinner_record(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """The provenance requirement is enforced, not merely attempted."""
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    receipt = serving.build_receipt(
        slate_date="2026-11-15",
        production_sha=None,
        incumbent=serving.resolve_incumbent(
            registry=registry,
            model_dir=model_dir,
            bundle_root=model_dir.parent,
        ),
        readiness={"ready": True, "reason": "", "refresh_outcome": "REFRESHED"},
        fingerprint=None,
        served={
            "outcome": serving.OUTCOME_SERVED,
            "prediction_artifact": {"sha256": "e" * 64},
            "priced_market_artifact": {"sha256": "f" * 64},
            "reason": "",
        },
    )

    assert serving.missing_provenance(receipt) == [
        "input_state_fingerprint",
        "production_code_sha",
    ]

    stripped = {
        key: value
        for key, value in receipt.items()
        if key != "incumbent_fit_id"
    }

    # Required to be present, not required to be non-null.
    assert "incumbent_fit_id" in serving.missing_provenance(stripped)


def test_a_no_slate_day_is_not_asked_for_artifact_provenance(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """Otherwise the preseason no-op would report a provenance failure."""
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    receipt = serving.run(
        arguments(
            data_root=tmp_path / "data",
            model_dir=model_dir,
            frozen_bundle_root=model_dir.parent,
            project_root=project,
            refresh_status=refresh_status(tmp_path, "PRESEASON_BLOCK"),
            registry_root=registry.root,
        )
    )

    assert receipt["outcome"] == serving.OUTCOME_NO_PRODUCTION_SLATE
    assert serving.missing_provenance(receipt) == []
    assert receipt["prediction_artifact"] is None
    assert receipt["incumbent_authority"] == serving.AUTHORITY_FROZEN_BUNDLE


# ----------------------------------------------------------------------
# readiness, and what counts as a failure
# ----------------------------------------------------------------------


def test_readiness_comes_from_the_refresh_outcome(tmp_path: Path):
    assert serving.slate_readiness(refresh_status(tmp_path, "REFRESHED"))[
        "ready"
    ]

    assert not serving.slate_readiness(
        refresh_status(tmp_path, "PRESEASON_BLOCK")
    )["ready"]


def test_a_missing_refresh_status_is_a_refusal_not_an_assumption(
    tmp_path: Path,
):
    """Readiness unknown is not readiness false, and not readiness true."""
    with pytest.raises(serving.ServingRefused):
        serving.slate_readiness(tmp_path / "absent.json")


def test_an_empty_regular_season_slate_is_not_a_failure(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """A date with no games written is the prediction script's own no-op."""
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    _write_stub(project, serving.PREDICT_SCRIPT, NO_GAMES_STUB)

    receipt = serving.run(
        arguments(
            data_root=tmp_path / "data",
            model_dir=model_dir,
            frozen_bundle_root=model_dir.parent,
            project_root=project,
            refresh_status=refresh_status(tmp_path, "REFRESHED"),
            registry_root=registry.root,
        )
    )

    assert receipt["outcome"] == serving.OUTCOME_NO_GAMES_ON_SLATE
    assert receipt["priced_market_artifact"] is None


@pytest.mark.parametrize("script", ("PREDICT_SCRIPT", "PRICE_SCRIPT"))
def test_a_failed_serving_script_fails_the_lifecycle(
    script: str, tmp_path: Path, project: Path, registry: FitRegistry
):
    """The brief's hard requirement: this may not be hidden."""
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    _write_stub(project, getattr(serving, script), FAILING_STUB)

    with pytest.raises(serving.ServingFailed):
        serving.run(
            arguments(
                data_root=tmp_path / "data",
                model_dir=model_dir,
                frozen_bundle_root=model_dir.parent,
                project_root=project,
                refresh_status=refresh_status(tmp_path, "REFRESHED"),
                registry_root=registry.root,
            )
        )


def test_the_entry_point_exits_nonzero_and_records_why(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    """What the workflow actually observes."""
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    _write_stub(project, serving.PRICE_SCRIPT, FAILING_STUB)

    receipt_path = tmp_path / "receipt.json"

    code = serving.main(
        [
            "--slate-date",
            "2026-11-15",
            "--data-root",
            str(tmp_path / "data"),
            "--model-dir",
            str(model_dir),
            "--frozen-bundle-root",
            str(model_dir.parent),
            "--project-root",
            str(project),
            "--refresh-status",
            str(refresh_status(tmp_path, "REFRESHED")),
            "--registry-root",
            str(registry.root),
            "--receipt-path",
            str(receipt_path),
        ]
    )

    assert code == serving.EXIT_FAILED

    recorded = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert recorded["outcome"] == serving.OUTCOME_FAILED
    assert recorded["error"] == "ServingFailed"


def test_a_served_run_exits_zero_and_writes_its_receipt(
    tmp_path: Path, project: Path, registry: FitRegistry
):
    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    receipt_path = tmp_path / "receipt.json"

    summary_path = tmp_path / "summary.md"

    code = serving.main(
        [
            "--slate-date",
            "2026-11-15",
            "--data-root",
            str(refreshed_data_root(tmp_path)),
            "--model-dir",
            str(model_dir),
            "--frozen-bundle-root",
            str(model_dir.parent),
            "--project-root",
            str(project),
            "--production-sha",
            "c" * 40,
            "--refresh-status",
            str(refresh_status(tmp_path, "REFRESHED")),
            "--registry-root",
            str(registry.root),
            "--receipt-path",
            str(receipt_path),
            "--summary-path",
            str(summary_path),
        ]
    )

    assert code == serving.EXIT_OK

    recorded = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert recorded["outcome"] == serving.OUTCOME_SERVED

    summary = summary_path.read_text(encoding="utf-8")

    assert "Incumbent production serving" in summary
    assert "incumbent" in summary


def test_an_unspecified_registry_root_is_a_refusal(
    tmp_path: Path, project: Path, monkeypatch
):
    """Serving without the promotion state would be serving without authority."""
    monkeypatch.delenv("NBA_PROP_FIT_REGISTRY_DIR", raising=False)

    model_dir = serving_tree(project, artifacts={"marginals.joblib": b"frozen"})

    code = serving.main(
        [
            "--slate-date",
            "2026-11-15",
            "--data-root",
            str(tmp_path / "data"),
            "--model-dir",
            str(model_dir),
            "--frozen-bundle-root",
            str(model_dir.parent),
            "--project-root",
            str(project),
            "--refresh-status",
            str(refresh_status(tmp_path, "REFRESHED")),
        ]
    )

    assert code == serving.EXIT_FAILED


# ----------------------------------------------------------------------
# the lifecycle wiring
# ----------------------------------------------------------------------


def lifecycle_steps() -> list[dict]:
    workflow = yaml.safe_load(LIFECYCLE_WORKFLOW.read_text(encoding="utf-8"))

    return workflow["jobs"]["lifecycle"]["steps"]


def serving_step() -> dict:
    for step in lifecycle_steps():
        if step.get("name") == SERVING_STEP_NAME:
            return step

    raise AssertionError(f"no {SERVING_STEP_NAME!r} step in the lifecycle")


def test_the_serving_step_is_not_gated_on_run_adaptive():
    """The brief's requirement, and a real decoupling.

    They agree today, but "should the incumbent price tonight" and "was a
    candidate refitted this morning" are different questions. Binding them
    means any future off-day fit policy silently stops production serving.
    """
    condition = serving_step()["if"]

    assert "RUN_ADAPTIVE" not in condition
    assert "MODE" in condition


def test_the_serving_step_cannot_be_hidden_behind_continue_on_error():
    assert "continue-on-error" not in serving_step()


def test_the_incumbent_serves_before_the_candidate_is_shadowed():
    """Ordering is part of the safety argument.

    Serving must be complete before any candidate work is observed, so that
    a shadow failure cannot arrive mid-serve.
    """
    names = [step.get("name") for step in lifecycle_steps()]

    assert names.index("Run the adaptive daily protocol") < names.index(
        SERVING_STEP_NAME
    )

    assert names.index(SERVING_STEP_NAME) < names.index(
        "Shadow the slate beside production"
    )


def test_the_serving_step_resolves_the_registry_it_is_told_about():
    run = serving_step()["run"]

    assert "--registry-root" in run
    assert "--refresh-status" in run
    assert "--production-sha" in run


def test_nothing_in_the_serving_path_publishes_or_promotes():
    source = ENTRY_POINT.read_text(encoding="utf-8")

    for forbidden in (
        ".promote(",
        ".rollback(",
        "publish_",
        "_publish",
        "wizardofodds",
        "WizardOfOdds",
    ):
        assert forbidden not in source
