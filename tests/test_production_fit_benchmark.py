"""Tests for the full-data production fit benchmark harness.

The harness is a wrapper, so what is worth testing is what it refuses and
what it notices. It must not be able to register, promote or publish; it must
stage outside version control; and it must report a production-state mutation
as a failure of the benchmark even when the fit itself succeeded.

Nothing here runs a real fit. The fit entry point is replaced by a stub that
emits the same JSON shape, which is what lets the isolation assertions be
exercised at all -- a real full-corpus fit takes hours and is run by the
operator on the production runner.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

HARNESS = REPO / "ops" / "benchmark_production_fit.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "benchmark_production_fit", HARNESS
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


harness = _load()


# ----------------------------------------------------------------------
# a fit result of the shape the real path emits
# ----------------------------------------------------------------------


def fit_result(**overrides) -> dict:
    stages = {stage: 1.5 for stage in harness.REQUIRED_STAGES}

    result = {
        "outcome": "COMPLETED",
        "training_data_manifest_sha256": "a" * 64,
        "calibration_fallbacks": {},
        "deferred_validation_checks": [],
        "validation_checks": {
            "advanced_coverage_check": True,
            "architecture_contract_match": True,
            "artifact_hashes_valid": True,
            "calibration_valid": True,
            "data_refresh_valid": True,
            "feature_schema_match": True,
            "finite_values_check": True,
            "history_regression_check": True,
            "marginal_fit_valid": True,
            "prediction_smoke_test": True,
            "required_artifacts_present": True,
            "source_lineage_match": True,
            "training_completed": True,
        },
        "validation_report": {
            "results": [
                {
                    "name": "prediction_smoke_test",
                    "passed": True,
                    "evidence": "the candidate priced five lines",
                    "values": {"lines": 5},
                }
            ]
        },
        "workspace": "/scratch/workspace",
        "benchmark": {
            "candidate_artifact_bytes": 1024,
            "details": {"candidate_artifact_count": 9},
            "ended_at_utc": "2026-11-15T04:00:00+00:00",
            "peak_rss_bytes": 2_000_000_000,
            "stage_seconds": stages,
            "started_at_utc": "2026-11-15T02:00:00+00:00",
            "total_seconds": 7200.0,
        },
    }

    result.update(overrides)

    return result


STUB_TEMPLATE = """
import json
import sys

{mutation}

print(json.dumps({result!r}))
"""


@pytest.fixture
def staged(tmp_path: Path):
    """A fake project whose fit entry point is a stub, plus scratch roots."""
    project = tmp_path / "project"

    (project / harness.FIT_ENTRY_POINT.parent).mkdir(parents=True)

    for relative in harness.FROZEN_IDENTIFIER_PATHS:
        path = project / relative

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(json.dumps({"frozen": relative}), encoding="utf-8")

    model_dir = project / "models"

    model_dir.mkdir()

    (model_dir / "marginals.joblib").write_bytes(b"served-marginals")

    registry = tmp_path / "registry"

    (registry / "state").mkdir(parents=True)

    (registry / "state" / "promotion_state.json").write_text(
        json.dumps({"current_good_fit_id": None}), encoding="utf-8"
    )

    return {
        "data": tmp_path / "data",
        "model_dir": model_dir,
        "project": project,
        "registry": registry,
        "work": tmp_path / "scratch",
    }


def install_stub(staged: dict, *, result: dict, mutation: str = "") -> None:
    (staged["project"] / harness.FIT_ENTRY_POINT).write_text(
        STUB_TEMPLATE.format(mutation=mutation, result=result),
        encoding="utf-8",
    )


def arguments(staged: dict, tmp_path: Path, **overrides):
    defaults = {
        "slate_date": "2026-11-15",
        "data_root": staged["data"],
        "work_root": staged["work"],
        "model_dir": staged["model_dir"],
        "registry_root": staged["registry"],
        "project_root": staged["project"],
        "production_sha": "c" * 40,
        "receipt_path": tmp_path / "receipt.json",
    }

    defaults.update(overrides)

    argv: list[str] = []

    for key, value in defaults.items():
        if value is None:
            continue

        argv += [f"--{key.replace('_', '-')}", str(value)]

    return harness.parse_args(argv)


# ----------------------------------------------------------------------
# the healthy case
# ----------------------------------------------------------------------


def test_a_clean_benchmark_passes_and_records_every_stage(
    staged: dict, tmp_path: Path
):
    install_stub(staged, result=fit_result())

    receipt = harness.run(arguments(staged, tmp_path))

    assert receipt["passed"]
    assert receipt["stages"]["every_required_stage_completed"]
    assert receipt["stages"]["registration_was_not_reached"]
    assert receipt["validation"]["every_computed_check_passed"]
    assert receipt["validation"]["prediction_smoke_test"]["passed"]
    assert receipt["calibration"]["stage_completed"]
    assert receipt["runtime"]["total_seconds"] == 7200.0
    assert receipt["peak_rss_bytes"] == 2_000_000_000
    assert receipt["production_state"]["mutation_occurred"] is False
    assert receipt["promotion_performed"] is False
    assert receipt["publishing_performed"] is False


def test_the_receipt_stays_small_and_names_the_workspace_rather_than_carrying_it(
    staged: dict, tmp_path: Path
):
    """Fitted binaries are not evidence and must not be committed."""
    install_stub(staged, result=fit_result())

    receipt_path = tmp_path / "receipt.json"

    harness.main(
        [
            "--slate-date",
            "2026-11-15",
            "--data-root",
            str(staged["data"]),
            "--work-root",
            str(staged["work"]),
            "--model-dir",
            str(staged["model_dir"]),
            "--project-root",
            str(staged["project"]),
            "--receipt-path",
            str(receipt_path),
        ]
    )

    assert receipt_path.stat().st_size < 32_768

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["workspace"]["path"] == "/scratch/workspace"
    assert receipt["workspace"]["candidate_artifact_bytes"] == 1024


# ----------------------------------------------------------------------
# isolation: the benchmark cannot change production state
# ----------------------------------------------------------------------


def test_the_fit_is_never_given_a_registry_root(staged: dict, tmp_path: Path):
    """The structural half of "registers nothing".

    benchmark-only with no registry root has no registry object at all, so
    registration is not refused at runtime -- there is nothing to refuse to.
    """
    command = harness.fit_command(
        project_root=staged["project"],
        data_root=staged["data"],
        work_root=staged["work"],
        slate_date="2026-11-15",
        training_cutoff=None,
    )

    assert "--benchmark-only" in command
    assert "--registry-root" not in command
    assert "--register-candidate" not in command


def test_a_mutated_registry_fails_the_benchmark(staged: dict, tmp_path: Path):
    """Checked, not assumed, and a failure of the benchmark either way.

    The fit succeeded in this case. The benchmark still fails, because a run
    that moved production state has not measured production -- it has changed
    it.
    """
    install_stub(
        staged,
        result=fit_result(),
        mutation=(
            "from pathlib import Path\n"
            f"Path({str(staged['registry'] / 'state' / 'promotion_state.json')!r})"
            '.write_text(\'{"current_good_fit_id": "nba_prop_quant_fit_x"}\')'
        ),
    )

    receipt = harness.run(arguments(staged, tmp_path))

    assert receipt["outcome"] == "COMPLETED"
    assert receipt["passed"] is False
    assert receipt["production_state"]["mutation_occurred"]
    assert receipt["production_state"]["mutated"] == ["registry_tree"]


def test_a_mutated_serving_tree_fails_the_benchmark(
    staged: dict, tmp_path: Path
):
    install_stub(
        staged,
        result=fit_result(),
        mutation=(
            "from pathlib import Path\n"
            f"Path({str(staged['model_dir'] / 'marginals.joblib')!r})"
            ".write_bytes(b'overwritten')"
        ),
    )

    receipt = harness.run(arguments(staged, tmp_path))

    assert receipt["passed"] is False
    assert receipt["production_state"]["mutated"] == ["model_tree"]


@pytest.mark.parametrize("relative", harness.FROZEN_IDENTIFIER_PATHS)
def test_a_moved_frozen_identifier_fails_the_benchmark(
    relative: str, staged: dict, tmp_path: Path
):
    install_stub(
        staged,
        result=fit_result(),
        mutation=(
            "from pathlib import Path\n"
            f"Path({str(staged['project'] / relative)!r})"
            ".write_text('{\"frozen\": \"moved\"}')"
        ),
    )

    receipt = harness.run(arguments(staged, tmp_path))

    assert receipt["passed"] is False
    assert receipt["production_state"]["mutated"] == [relative]


def test_staging_inside_the_repository_is_refused(
    staged: dict, tmp_path: Path
):
    """Otherwise a benchmark would write model binaries into version control."""
    install_stub(staged, result=fit_result())

    with pytest.raises(harness.BenchmarkRefused, match="inside the repository"):
        harness.run(
            arguments(
                staged, tmp_path, work_root=staged["project"] / "scratch"
            )
        )


# ----------------------------------------------------------------------
# what counts as a failed benchmark
# ----------------------------------------------------------------------


def test_reaching_registration_fails_the_benchmark(
    staged: dict, tmp_path: Path
):
    """The one stage a benchmark run must not have."""
    result = fit_result()

    result["benchmark"]["stage_seconds"][harness.FORBIDDEN_STAGE] = 0.5

    install_stub(staged, result=result)

    receipt = harness.run(arguments(staged, tmp_path))

    assert receipt["passed"] is False
    assert receipt["stages"]["registration_was_not_reached"] is False


def test_a_missing_stage_fails_the_benchmark(staged: dict, tmp_path: Path):
    result = fit_result()

    del result["benchmark"]["stage_seconds"]["calibration_fit"]

    install_stub(staged, result=result)

    receipt = harness.run(arguments(staged, tmp_path))

    assert receipt["passed"] is False
    assert receipt["stages"]["missing_stages"] == ["calibration_fit"]
    assert receipt["calibration"]["stage_completed"] is False


def test_a_failed_computed_check_fails_the_benchmark(
    staged: dict, tmp_path: Path
):
    """The validation the brief requires is the real computed one."""
    result = fit_result()

    result["validation_checks"]["prediction_smoke_test"] = False

    install_stub(staged, result=result)

    receipt = harness.run(arguments(staged, tmp_path))

    assert receipt["passed"] is False
    assert receipt["validation"]["failed_checks"] == ["prediction_smoke_test"]


def test_a_calibration_fallback_is_recorded_without_being_hidden(
    staged: dict, tmp_path: Path
):
    """A fallback origin is a real observation about the calibration stage."""
    install_stub(
        staged,
        result=fit_result(calibration_fallbacks={"blocks": "parent"}),
    )

    receipt = harness.run(arguments(staged, tmp_path))

    assert receipt["calibration"]["fallback_origins"] == {"blocks": "parent"}
    assert receipt["calibration"]["every_route_fitted"] is False


def test_a_nonzero_fit_exit_is_a_refusal_with_the_reason(
    staged: dict, tmp_path: Path
):
    (staged["project"] / harness.FIT_ENTRY_POINT).write_text(
        "import sys\n"
        "print('FrozenPolicyViolation: calibration family moved',"
        " file=sys.stderr)\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )

    with pytest.raises(harness.BenchmarkRefused, match="FrozenPolicyViolation"):
        harness.run(arguments(staged, tmp_path))


def test_the_cli_exits_nonzero_when_the_benchmark_failed(
    staged: dict, tmp_path: Path
):
    result = fit_result()

    result["validation_checks"]["calibration_valid"] = False

    install_stub(staged, result=result)

    receipt_path = tmp_path / "receipt.json"

    code = harness.main(
        [
            "--slate-date",
            "2026-11-15",
            "--data-root",
            str(staged["data"]),
            "--work-root",
            str(staged["work"]),
            "--model-dir",
            str(staged["model_dir"]),
            "--project-root",
            str(staged["project"]),
            "--receipt-path",
            str(receipt_path),
        ]
    )

    assert code == harness.EXIT_FAILED

    assert json.loads(receipt_path.read_text(encoding="utf-8"))["passed"] is False


def test_the_harness_states_no_model_parameter_of_its_own():
    """It measures the production path; it must not be able to change it."""
    source = HARNESS.read_text(encoding="utf-8")

    for forbidden in (
        "--register-candidate",
        ".promote(",
        "role_scale",
        "lambda_",
        "zinb",
    ):
        assert forbidden not in source
