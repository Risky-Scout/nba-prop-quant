from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]

BUILDER_PATH = PROJECT / "scripts/19_build_wizardofodds_runtime_bundle.py"

REAL_CONTRACT_PATH = (
    PROJECT
    / "models/frozen_manifests/nba_prop_quant_v2_gate3_runtime_contract.json"
)

FROZEN_MODEL_COMMIT = "4def8ad33ccc56016fb19a97fceca6e027c9612a"

GATE3_POLICY = {
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

V1_FREEZE_MANIFESTS = [
    "models/frozen_manifests/nba_prop_quant_20260818T202318Z.json",
    "models/frozen_manifests/nba_prop_quant_20260818T202318Z.md",
    "models/frozen_manifests/nba_prop_quant_20260818T202318Z_pip_freeze.txt",
    "models/frozen_manifests/nba_prop_quant_20260818T205213Z.json",
    "models/frozen_manifests/nba_prop_quant_20260818T205213Z.md",
    "models/frozen_manifests/nba_prop_quant_20260818T205213Z_pip_freeze.txt",
    "models/frozen_manifests/LATEST.json",
    "models/frozen_manifests/LATEST.md",
]


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "runtime_bundle_builder_under_test",
        BUILDER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the runtime bundle builder")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


builder = load_builder()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=runtime-bundle-test",
            "-c",
            "user.email=runtime-bundle-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_sha256sums(directory: Path, filenames: list[str]) -> None:
    lines = [
        f"{sha256_bytes((directory / name).read_bytes())}  {name}"
        for name in sorted(filenames)
    ]

    write(directory / "SHA256SUMS.txt", "\n".join(lines) + "\n")


def synthetic_contract(model_source_commit: str) -> dict:
    return {
        "schema_version": 2,
        "contract_name": "synthetic_runtime_contract",
        "contract_version": 1,
        "runtime_name": "synthetic_runtime",
        "runtime_version": 1,
        "runtime_manifest_basename": "synthetic_runtime_manifest.json",
        "manifest_freeze_stage": "external_test_deployment",
        "model_source_commit": model_source_commit,
        "gate3_policy": dict(GATE3_POLICY),
        "gate3_policy_lock_commit": "dd2d394b6def1e146a193b9000536158d1ec383d",
        "gate3_candidate_policy_id": "nba_prop_quant_v2_gate3_dd2d394b6def",
        "primary_certification_window": "T-20m",
        "primary_certification_offset_minutes": 20,
        "prospective_claim_allowed": False,
        "auto_bet": False,
        "frozen_model_source_files": [
            "src/nba_prop_quant/production.py",
            "src/nba_prop_quant/prospective_snapshot.py",
            "scripts/10_predict_slate.py",
            "scripts/15_price_markets.py",
        ],
        "sha256sums_groups": [
            "research/v2_gate3_capture_lock/SHA256SUMS.txt",
            "research/v2_gate3_deployment_artifacts/SHA256SUMS.txt",
        ],
        "forbidden_path_globs": [
            ".env",
            ".env.*",
            "*.env",
            "*.pem",
            "*.key",
            "credentials*",
            "secrets*",
        ],
        "forbidden_content_patterns": [
            "-----BEGIN [A-Z ]*PRIVATE KEY-----",
            "(?im)^[ \\t]*(export[ \\t]+)?BDL_API_KEY[ \\t]*=",
            "(?im)^[ \\t]*(export[ \\t]+)?FTP_PASSWORD[ \\t]*=",
        ],
        "excluded_development_resources": [],
        "frozen_integrity_groups": [
            "source_files",
            "scripts",
            "model_artifacts",
            "model_provenance_artifacts",
            "gate3_deployment_artifacts",
            "gate3_policy_locks",
            "packaging",
        ],
        "rolling_integrity_groups": [
            "runtime_data",
        ],
        "groups": {
            "source_files": {
                "root": "project_root",
                "required": True,
                "files": [
                    "src/nba_prop_quant/__init__.py",
                    "src/nba_prop_quant/production.py",
                    "src/nba_prop_quant/prospective_snapshot.py",
                ],
            },
            "scripts": {
                "root": "project_root",
                "required": True,
                "files": [
                    "scripts/10_predict_slate.py",
                    "scripts/15_price_markets.py",
                ],
            },
            "packaging": {
                "root": "project_root",
                "required": True,
                "files": ["pyproject.toml"],
            },
            "gate3_deployment_artifacts": {
                "root": "project_root",
                "required": True,
                "files": [
                    "research/v2_gate3_deployment_artifacts/SHA256SUMS.txt",
                    (
                        "research/v2_gate3_deployment_artifacts/"
                        "deployment_manifest.json"
                    ),
                    (
                        "research/v2_gate3_deployment_artifacts/"
                        "probability_parameters.json"
                    ),
                ],
            },
            "gate3_policy_locks": {
                "root": "project_root",
                "required": True,
                "files": [
                    "research/v2_gate3_capture_lock/SHA256SUMS.txt",
                    (
                        "research/v2_gate3_capture_lock/"
                        "GATE3_CAPTURE_POLICY.json"
                    ),
                ],
            },
            "model_artifacts": {
                "root": "model_dir",
                "stage_prefix": "models",
                "required": True,
                "files": [
                    "marginals.joblib",
                    "mean_model_selection.json",
                    "pts.joblib",
                ],
            },
            "model_provenance_artifacts": {
                "root": "model_dir",
                "stage_prefix": "models",
                "required": False,
                "files": ["ensemble_weights.json"],
            },
            "runtime_data": {
                "root": "data_dir",
                "stage_prefix": "data",
                "required": True,
                "patterns": [
                    {
                        "glob": "raw/seasons/**/stats.parquet",
                        "min_files": 1,
                        "read_by": "load_history_box_stats",
                    },
                    {
                        "glob": "raw/advanced/**/*.parquet",
                        "min_files": 1,
                        "read_by": "load_advanced",
                    },
                ],
            },
        },
    }


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    """A minimal synthetic project root, model dir and data dir."""

    project = tmp_path / "project"
    model_dir = tmp_path / "frozen_models"
    data_dir = tmp_path / "runtime_data"
    output_dir = tmp_path / "out"

    write(project / "src/nba_prop_quant/__init__.py", "")

    write(
        project / "src/nba_prop_quant/production.py",
        "TARGETS = ['pts']\n",
    )

    write(
        project / "src/nba_prop_quant/prospective_snapshot.py",
        "PRIMARY_OFFSET_MINUTES = 20\n",
    )

    write(
        project / "src/nba_prop_quant/gate3_v2.py",
        "def load_gate3_runtime():\n"
        "    required_policy = "
        + repr(dict(sorted(GATE3_POLICY.items())))
        + "\n    return required_policy\n",
    )

    write(project / "scripts/10_predict_slate.py", "# predict\n")
    write(project / "scripts/15_price_markets.py", "# price\n")

    write(
        project / "pyproject.toml",
        '[project]\nname = "synthetic"\nversion = "0.0.1"\n',
    )

    gate3 = project / "research/v2_gate3_deployment_artifacts"

    write(
        gate3 / "deployment_manifest.json",
        json.dumps(
            {
                "artifact": "synthetic_gate3_deployment",
                "gate3_lock_commit": (
                    "dd2d394b6def1e146a193b9000536158d1ec383d"
                ),
                "gate3_policy": dict(GATE3_POLICY),
                "prospective_claim_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    write(
        gate3 / "probability_parameters.json",
        json.dumps({"assists": {"intercept": 0.0, "slope": 1.0}}) + "\n",
    )

    write_sha256sums(
        gate3,
        ["deployment_manifest.json", "probability_parameters.json"],
    )

    capture = project / "research/v2_gate3_capture_lock"

    write(
        capture / "GATE3_CAPTURE_POLICY.json",
        json.dumps(
            {
                "policy_name": "synthetic_capture_policy",
                "primary_certification_window": {
                    "fallback_allowed": False,
                    "offset_minutes_before_tip": 20,
                    "window_label": "T-20m",
                },
                "retuning_from_prospective_results": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    write_sha256sums(capture, ["GATE3_CAPTURE_POLICY.json"])

    for name in [
        "pts.joblib",
        "marginals.joblib",
        "ensemble_weights.json",
    ]:
        write(model_dir / name, f"synthetic-{name}\n")

    write(
        model_dir / "mean_model_selection.json",
        json.dumps({"targets": {"pts": {"selected_mode": "xgb"}}}) + "\n",
    )

    write(
        data_dir / "raw/seasons/season=2024/stats.parquet",
        "synthetic-stats\n",
    )

    write(
        data_dir / "raw/advanced/season=2024/advanced.parquet",
        "synthetic-advanced\n",
    )

    git(project, "init", "-q", "-b", "main")
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "synthetic frozen model")

    model_commit = git(project, "rev-parse", "HEAD")

    write(project / "OPERATIONS.md", "# operational wrapper\n")
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "operational packaging")

    contract_path = project / "runtime_contract.json"

    write(
        contract_path,
        json.dumps(synthetic_contract(model_commit), indent=2, sort_keys=True)
        + "\n",
    )

    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "runtime contract")

    return {
        "project": project,
        "model_dir": model_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "contract_path": contract_path,
        "model_commit": model_commit,
        "gate3_artifact_dir": gate3,
    }


def build(workspace: dict, **overrides):
    kwargs = {
        "project_root": workspace["project"],
        "model_dir": workspace["model_dir"],
        "data_dir": workspace["data_dir"],
        "output_dir": workspace["output_dir"],
        "contract_path": workspace["contract_path"],
        "gate3_artifact_dir": workspace["gate3_artifact_dir"],
    }

    kwargs.update(overrides)

    return builder.build_runtime_bundle(**kwargs)


def patch_contract(workspace: dict, mutate) -> None:
    contract = json.loads(
        workspace["contract_path"].read_text(encoding="utf-8")
    )

    mutate(contract)

    workspace["contract_path"].write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "adjust contract")


HISTORICAL_STAGED_PATHS = [
    "data/raw/seasons/season=2024/stats.parquet",
    "data/raw/advanced/season=2024/advanced.parquet",
]


def extract(workspace: dict, result: dict, name: str = "extracted") -> Path:
    destination = workspace["output_dir"] / name

    with tarfile.open(result["archive_path"], "r:gz") as tar:
        tar.extractall(destination, filter="data")

    return destination / result["staging_name"]


def verify_frozen_manifest(bundle: Path) -> dict:
    load_verified_manifest_metadata = builder.import_verifier(PROJECT)

    return load_verified_manifest_metadata(
        model_dir=bundle / "models",
        project_root=bundle,
    )


# ---------------------------------------------------------------------------
# Real repository contract
# ---------------------------------------------------------------------------


def test_real_contract_schema_is_complete():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    required_fields = {
        "schema_version",
        "runtime_name",
        "runtime_version",
        "model_source_commit",
        "gate3_candidate_policy_id",
        "gate3_policy_lock_commit",
        "primary_certification_window",
        "prospective_claim_allowed",
        "auto_bet",
        "manifest_freeze_stage",
        "groups",
        "frozen_model_source_files",
        "forbidden_path_globs",
        "excluded_development_resources",
    }

    assert required_fields <= set(contract)

    assert contract["model_source_commit"] == FROZEN_MODEL_COMMIT
    assert contract["primary_certification_window"] == "T-20m"
    assert contract["primary_certification_offset_minutes"] == 20
    assert contract["prospective_claim_allowed"] is False
    assert contract["auto_bet"] is False
    assert contract["gate3_policy"] == GATE3_POLICY
    assert contract["manifest_freeze_stage"] == "external_test_deployment"

    derived = (
        "nba_prop_quant_v2_gate3_"
        + contract["gate3_policy_lock_commit"][:12]
    )

    assert contract["gate3_candidate_policy_id"] == derived


def test_real_contract_declares_two_integrity_domains():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    frozen = contract["frozen_integrity_groups"]
    rolling = contract["rolling_integrity_groups"]

    assert frozen == [
        "source_files",
        "scripts",
        "model_artifacts",
        "model_provenance_artifacts",
        "gate3_deployment_artifacts",
        "gate3_policy_locks",
        "claim_policy",
        "configs",
        "packaging",
    ]

    assert rolling == ["runtime_data"]

    assert "runtime_data" in rolling
    assert "runtime_data" not in frozen
    assert set(frozen) & set(rolling) == set()

    groups = set(contract["groups"])

    assert set(frozen) <= groups
    assert set(rolling) <= groups
    assert set(frozen) | set(rolling) == groups


def test_real_contract_version_tracks_the_domain_split():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["contract_version"] == 2
    assert contract["schema_version"] == 2
    assert contract["runtime_version"] == 1


def test_real_contract_preserves_published_runtime_properties():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["model_source_commit"] == FROZEN_MODEL_COMMIT
    assert contract["primary_certification_window"] == "T-20m"
    assert contract["prospective_claim_allowed"] is False
    assert contract["auto_bet"] is False


def test_real_contract_policy_matches_frozen_gate3_source():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    assert builder.source_gate3_policy(PROJECT) == contract["gate3_policy"]
    assert builder.source_gate3_policy(PROJECT) == GATE3_POLICY


def test_real_contract_covers_runtime_import_closure():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    bundled_sources = set(contract["groups"]["source_files"]["files"])

    for module in [
        "gate3_v2",
        "pipeline",
        "pricing",
        "production",
        "prospective_snapshot",
        "settings",
        "slate",
        "snapshot_schedule",
        "storage",
    ]:
        assert f"src/nba_prop_quant/{module}.py" in bundled_sources

    scripts = set(contract["groups"]["scripts"]["files"])

    assert "scripts/10_predict_slate.py" in scripts
    assert "scripts/15_price_markets.py" in scripts

    model_artifacts = set(contract["groups"]["model_artifacts"]["files"])

    for artifact in [
        "ast.joblib",
        "blk.joblib",
        "combo_dependence_policy.json",
        "copula.joblib",
        "dynamic_params.json",
        "experience_curves.joblib",
        "fg3m.joblib",
        "marginals.joblib",
        "market_probability_calibration_policy.json",
        "mean_model_selection.json",
        "minutes.joblib",
        "pts.joblib",
        "reb.joblib",
        "stl.joblib",
    ]:
        assert artifact in model_artifacts


def test_real_contract_excludes_mutable_runtime_state():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    serialized = json.dumps(contract["groups"])

    for fragment in [
        "data/snapshots",
        "capture_runs",
        "priced_markets",
        "projections",
    ]:
        assert fragment not in serialized

    excluded = {
        entry["path"]
        for entry in contract["excluded_development_resources"]
    }

    assert "data/snapshots/**" in excluded
    assert "research/v2_gate2_certification_outputs/**" in excluded


ADAPTIVE_SERVING_CONTRACT_PATH = (
    PROJECT
    / "models"
    / "frozen_manifests"
    / "nba_prop_quant_v2_adaptive_serving_source_contract.json"
)


def adaptive_serving_contract() -> dict:
    return json.loads(
        ADAPTIVE_SERVING_CONTRACT_PATH.read_text(encoding="utf-8")
    )


def test_frozen_mathematical_sources_match_anchor():
    """Every serving source still matches the anchor unless declared otherwise.

    Adaptive production may need a serving-correctness fix in a file the
    historical contract byte-pins. Such a file is enumerated in the adaptive
    serving source contract with a justification; everything else must still
    be byte-identical to the frozen commit, and an undeclared difference still
    fails here.
    """
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
    adaptive = adaptive_serving_contract()

    declared = set(adaptive["diverged_from_historical_reference"])

    unchanged = [
        relative
        for relative in contract["frozen_model_source_files"]
        if relative not in declared
    ]

    verified = builder.verify_frozen_model_sources(
        PROJECT,
        FROZEN_MODEL_COMMIT,
        unchanged,
    )

    assert set(verified) == set(unchanged)

    # A declared divergence must be a real one, so the exemption cannot be
    # used to quietly wave through an unchanged file.
    for relative in sorted(declared):
        assert relative in set(contract["frozen_model_source_files"])

        with pytest.raises(builder.BuildError):
            builder.verify_frozen_model_sources(
                PROJECT,
                FROZEN_MODEL_COMMIT,
                [relative],
            )


def test_historical_reference_bytes_are_still_intact():
    """The anchor itself is unchanged, including for the diverged file.

    The adaptive contract records the historical SHA256 of every locked file.
    Those must still match the blobs at the frozen commit, so the historical
    evidence remains verifiable even where serving has moved on.
    """
    adaptive = adaptive_serving_contract()

    assert (
        adaptive["historical_architecture_reference"] == FROZEN_MODEL_COMMIT
    )

    historical_contract_sha = hashlib.sha256(
        REAL_CONTRACT_PATH.read_bytes()
    ).hexdigest()

    assert (
        adaptive["historical_contract_sha256"] == historical_contract_sha
    )

    for relative, entry in sorted(
        adaptive["locked_serving_source_files"].items()
    ):
        blob = subprocess.run(
            ["git", "cat-file", "blob", f"{FROZEN_MODEL_COMMIT}:{relative}"],
            cwd=PROJECT,
            capture_output=True,
            check=True,
        )

        assert (
            hashlib.sha256(blob.stdout).hexdigest()
            == entry["historical_reference_sha256"]
        ), f"{relative} anchor bytes moved"


def test_adaptive_serving_source_matches_its_contract():
    """What adaptive serving runs is pinned, not merely exempted."""
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
    adaptive = adaptive_serving_contract()

    locked = adaptive["locked_serving_source_files"]

    assert set(locked) == set(contract["frozen_model_source_files"])

    for relative, entry in sorted(locked.items()):
        observed = hashlib.sha256(
            (PROJECT / relative).read_bytes()
        ).hexdigest()

        assert observed == entry["current_sha256"], (
            f"{relative} does not match the adaptive serving source contract"
        )

        matches = entry["matches_historical_reference"]

        assert matches == (
            entry["current_sha256"] == entry["historical_reference_sha256"]
        )

        if not matches:
            assert entry["divergence_reason"].strip()


def test_adaptive_serving_contract_asserts_no_model_change():
    invariants = adaptive_serving_contract()["invariants"]

    for name in (
        "model_mathematics_changed",
        "gate3_routing_changed",
        "calibration_methodology_changed",
        "dependence_methodology_changed",
        "marginal_family_changed",
        "mean_model_routing_changed",
        "feature_definitions_changed",
        "t20_certification_protocol_changed",
        "fitted_model_artifacts_changed",
    ):
        assert invariants[name] is False


def test_v1_freeze_manifests_are_unchanged():
    for relative in V1_FREEZE_MANIFESTS:
        blob = subprocess.run(
            ["git", "cat-file", "blob", f"{FROZEN_MODEL_COMMIT}:{relative}"],
            cwd=PROJECT,
            capture_output=True,
            check=True,
        ).stdout

        assert sha256_bytes(blob) == builder.sha256_file(PROJECT / relative), (
            f"{relative} changed relative to the frozen model anchor"
        )


def test_real_gate3_checksum_groups_verify():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    for relative in contract["sha256sums_groups"]:
        assert builder.verify_sha256sums_group(PROJECT, relative) > 0


def test_builder_performs_no_network_access():
    source = BUILDER_PATH.read_text(encoding="utf-8")

    for forbidden in ["httpx", "requests", "urllib.request", "socket"]:
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Synthetic end-to-end build
# ---------------------------------------------------------------------------


def test_bundle_builds_and_round_trips(workspace):
    result = build(workspace)
    manifest = result["manifest"]

    assert manifest["freeze_stage"] == "external_test_deployment"
    assert manifest["auto_bet"] is False
    assert manifest["prospective_claim_allowed"] is False
    assert manifest["primary_certification_window"] == "T-20m"
    assert manifest["gate3_policy"] == GATE3_POLICY
    assert (
        manifest["gate3_candidate_policy_id"]
        == "nba_prop_quant_v2_gate3_dd2d394b6def"
    )

    assert result["archive_path"].exists()
    assert result["roundtrip"]["verified_files"] > 0
    assert result["roundtrip"]["freeze_id"] == manifest["freeze_id"]
    assert result["roundtrip"]["freeze_stage"] == "external_test_deployment"


def test_manifest_separates_model_and_production_commits(workspace):
    manifest = build(workspace)["manifest"]

    assert manifest["model_source_commit"] == workspace["model_commit"]
    assert (
        manifest["production_source_commit"] != manifest["model_source_commit"]
    )
    assert manifest["production_source_dirty"] is False
    assert (
        manifest["frozen_model_source_verification"]["commit"]
        == workspace["model_commit"]
    )


def test_manifest_records_required_runtime_metadata(workspace):
    manifest = build(workspace)["manifest"]

    for field in [
        "schema_version",
        "runtime_name",
        "runtime_version",
        "model_source_commit",
        "production_source_commit",
        "gate3_candidate_policy_id",
        "gate3_policy_lock_commit",
        "model_artifact_hashes",
        "source_file_hashes",
        "runtime_data_hashes",
        "build_timestamp_utc",
        "dependency_lock",
        "prospective_claim_allowed",
        "auto_bet",
    ]:
        assert field in manifest, field

    assert manifest["environment"]["python_version"]
    assert manifest["dependency_lock"]["declared_dependencies_sha256"]
    assert manifest["runtime_data_hashes"]
    assert "models/pts.joblib" in manifest["model_artifact_hashes"]


def test_manifest_never_records_absolute_build_paths(workspace):
    manifest = build(workspace)["manifest"]
    serialized = json.dumps(manifest)

    assert "/Users/" not in serialized
    assert str(workspace["project"]) not in serialized


def test_generated_manifest_preserves_published_runtime_properties(workspace):
    manifest = build(workspace)["manifest"]
    real = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    assert real["model_source_commit"] == FROZEN_MODEL_COMMIT
    assert manifest["primary_certification_window"] == "T-20m"
    assert manifest["prospective_claim_allowed"] is False
    assert manifest["auto_bet"] is False


def test_real_contract_generates_metadata_pinned_to_the_model_anchor(tmp_path):
    """The real contract, not just a fixture, must still pin the anchor.

    A full build needs the release-asset model artifacts and historical
    parquets, so the manifest is generated from the real contract with empty
    groups: enough to prove contract fields reach the runtime metadata.
    """

    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    frozen_groups, rolling_groups = builder.resolve_integrity_domains(contract)

    manifest = builder.build_manifest(
        contract=contract,
        resolved={name: [] for name in contract["groups"]},
        staging=tmp_path,
        provenance={
            "model_source_commit": contract["model_source_commit"],
            "production_source_commit": "0" * 40,
            "production_source_branch": "production/wizardofodds-integration",
            "production_source_dirty": False,
        },
        gate3={
            "candidate_id": contract["gate3_candidate_policy_id"],
            "gate3_lock_commit": contract["gate3_policy_lock_commit"],
            "deployment_manifest_sha256": "0" * 64,
        },
        frozen_sources={},
        absent_optional=[],
        dependency_lock={},
        runtime_version=int(contract["runtime_version"]),
        frozen_groups=frozen_groups,
        rolling_groups=rolling_groups,
        created=builder.datetime(2026, 9, 4, tzinfo=builder.timezone.utc),
    )

    assert manifest["model_source_commit"] == FROZEN_MODEL_COMMIT
    assert manifest["primary_certification_window"] == "T-20m"
    assert manifest["primary_certification_offset_minutes"] == 20
    assert manifest["prospective_claim_allowed"] is False
    assert manifest["auto_bet"] is False
    assert manifest["gate3_policy"] == GATE3_POLICY
    assert manifest["contract_version"] == contract["contract_version"]

    assert manifest["rolling_integrity_groups"] == ["runtime_data"]
    assert "runtime_data" not in manifest["frozen_integrity_groups"]
    assert "runtime_data" not in manifest["files"]


# ---------------------------------------------------------------------------
# Frozen versus rolling integrity domains
# ---------------------------------------------------------------------------


def test_generated_manifest_records_both_integrity_domains(workspace):
    manifest = build(workspace)["manifest"]

    frozen = manifest["frozen_integrity_groups"]
    rolling = manifest["rolling_integrity_groups"]

    assert "runtime_data" in rolling
    assert "runtime_data" not in frozen
    assert set(frozen) & set(rolling) == set()
    assert set(manifest["files"]) <= set(frozen)


def test_runtime_data_is_bundled_under_data_prefix(workspace):
    result = build(workspace)

    staged = {stage for stage, _ in result["resolved"]["runtime_data"]}

    assert set(HISTORICAL_STAGED_PATHS) <= staged

    with tarfile.open(result["archive_path"], "r:gz") as tar:
        archived = set(tar.getnames())

    for relative in HISTORICAL_STAGED_PATHS:
        assert f"{result['staging_name']}/{relative}" in archived

    bundle = extract(workspace, result)

    for relative in HISTORICAL_STAGED_PATHS:
        assert (bundle / relative).is_file()


def test_runtime_data_hashes_cover_every_staged_historical_file(workspace):
    result = build(workspace, keep_staging=True)
    manifest = result["manifest"]
    staging = result["staging_path"]

    staged = {stage for stage, _ in result["resolved"]["runtime_data"]}

    assert staged
    assert set(HISTORICAL_STAGED_PATHS) <= staged
    assert set(manifest["runtime_data_hashes"]) == staged

    for relative, digest in manifest["runtime_data_hashes"].items():
        assert digest == builder.sha256_file(staging / relative)


def test_runtime_sha256sums_still_covers_historical_data(workspace):
    result = build(workspace, keep_staging=True)
    staging = result["staging_path"]

    listed = {
        line.split(None, 1)[1].strip(): line.split(None, 1)[0]
        for line in (
            staging / builder.RUNTIME_SHA256SUMS_NAME
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    staged = {stage for stage, _ in result["resolved"]["runtime_data"]}

    assert staged <= set(listed)

    for relative in staged:
        assert listed[relative] == builder.sha256_file(staging / relative)


def test_frozen_manifest_excludes_runtime_data(workspace):
    manifest = build(workspace)["manifest"]

    assert "runtime_data" not in manifest["files"]

    frozen_paths = {
        record["path"]
        for records in manifest["files"].values()
        for record in records
    }

    assert frozen_paths
    assert not any(path.startswith("data/") for path in frozen_paths)
    assert not any(path.endswith(".parquet") for path in frozen_paths)

    for relative in HISTORICAL_STAGED_PATHS:
        assert relative not in frozen_paths


def test_rolling_data_mutation_does_not_break_frozen_verifier(workspace):
    result = build(workspace)
    bundle = extract(workspace, result)

    assert verify_frozen_manifest(bundle)["freeze_id"] == (
        result["manifest"]["freeze_id"]
    )

    rolling = bundle / "data/raw/seasons/season=2024/stats.parquet"
    before = rolling.read_bytes()

    rolling.write_text(
        "synthetic-stats\nsynthetic-postgame-refresh\n",
        encoding="utf-8",
    )

    assert rolling.read_bytes() != before

    metadata = verify_frozen_manifest(bundle)

    assert metadata["freeze_id"] == result["manifest"]["freeze_id"]
    assert metadata["freeze_stage"] == "external_test_deployment"


def test_frozen_file_mutation_still_fails_frozen_verifier(workspace):
    result = build(workspace)
    bundle = extract(workspace, result)

    assert verify_frozen_manifest(bundle)["freeze_id"] == (
        result["manifest"]["freeze_id"]
    )

    frozen_paths = {
        record["path"]
        for records in result["manifest"]["files"].values()
        for record in records
    }

    assert "scripts/10_predict_slate.py" in frozen_paths

    (bundle / "scripts/10_predict_slate.py").write_text(
        "# tampered frozen entry point\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hash_mismatch"):
        verify_frozen_manifest(bundle)


def test_missing_frozen_file_still_fails_frozen_verifier(workspace):
    result = build(workspace)
    bundle = extract(workspace, result)

    (bundle / "scripts/10_predict_slate.py").unlink()

    with pytest.raises(RuntimeError, match="missing="):
        verify_frozen_manifest(bundle)


def test_undeclared_contract_group_fails_the_build(workspace):
    def mutate(contract):
        contract["frozen_integrity_groups"].remove("packaging")

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="assigned to no integrity domain",
    ):
        build(workspace)


def test_group_in_both_integrity_domains_fails_the_build(workspace):
    def mutate(contract):
        contract["frozen_integrity_groups"].append("runtime_data")

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="declared both frozen and rolling",
    ):
        build(workspace)


def test_unknown_integrity_group_fails_the_build(workspace):
    def mutate(contract):
        contract["frozen_integrity_groups"].append("not_a_group")

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="is not a contract group",
    ):
        build(workspace)


def test_missing_integrity_domain_declaration_fails_the_build(workspace):
    def mutate(contract):
        contract.pop("rolling_integrity_groups")

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="does not declare rolling_integrity_groups",
    ):
        build(workspace)


def test_sha256sums_covers_every_bundled_file(workspace):
    result = build(workspace, keep_staging=True)
    staging = result["staging_path"]

    listed = {
        line.split(None, 1)[1].strip()
        for line in (
            staging / builder.RUNTIME_SHA256SUMS_NAME
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    on_disk = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    }

    assert on_disk - listed == {builder.RUNTIME_SHA256SUMS_NAME}
    assert builder.RUNTIME_MANIFEST_NAME in listed
    assert "models/frozen_manifests/LATEST.json" in listed


def test_bundle_latest_pointer_matches_runtime_manifest(workspace):
    result = build(workspace, keep_staging=True)
    staging = result["staging_path"]

    canonical = (staging / builder.RUNTIME_MANIFEST_NAME).read_bytes()

    for relative in result["manifest_targets"]:
        assert (staging / relative).read_bytes() == canonical


def test_archive_members_are_normalized(workspace):
    result = build(workspace)

    with tarfile.open(result["archive_path"], "r:gz") as tar:
        members = tar.getmembers()

    names = [member.name for member in members]

    assert names == sorted(names)

    for member in members:
        assert member.isfile()
        assert member.mtime == 0
        assert member.uid == 0
        assert member.gid == 0
        assert member.uname == ""
        assert member.gname == ""
        assert member.name.startswith(f"{result['staging_name']}/")


def test_archive_round_trip_detects_tampering(workspace):
    result = build(workspace)

    with tarfile.open(result["archive_path"], "r:gz") as tar:
        tar.extractall(workspace["output_dir"] / "extracted", filter="data")

    extracted = (
        workspace["output_dir"] / "extracted" / result["staging_name"]
    )

    assert builder.verify_sha256sums_tree(extracted) > 0

    (extracted / "scripts/10_predict_slate.py").write_text(
        "# tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(builder.BuildError, match="hash mismatch"):
        builder.verify_sha256sums_tree(extracted)


def test_unlisted_extra_file_fails_verification(workspace):
    result = build(workspace)

    with tarfile.open(result["archive_path"], "r:gz") as tar:
        tar.extractall(workspace["output_dir"] / "extracted", filter="data")

    extracted = (
        workspace["output_dir"] / "extracted" / result["staging_name"]
    )

    (extracted / "smuggled.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(builder.BuildError, match="absent from"):
        builder.verify_sha256sums_tree(extracted)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_model_artifact_fails(workspace):
    (workspace["model_dir"] / "pts.joblib").unlink()

    with pytest.raises(
        builder.BuildError,
        match="Required runtime resource missing",
    ):
        build(workspace)


def test_missing_historical_data_fails(workspace):
    (
        workspace["data_dir"] / "raw/seasons/season=2024/stats.parquet"
    ).unlink()

    with pytest.raises(
        builder.BuildError,
        match="Required historical runtime data missing",
    ):
        build(workspace)


def test_absent_optional_artifact_is_recorded_not_fatal(workspace):
    (workspace["model_dir"] / "ensemble_weights.json").unlink()

    manifest = build(workspace)["manifest"]

    assert (
        "model_provenance_artifacts:ensemble_weights.json"
        in manifest["absent_optional_files"]
    )


def test_changed_mathematical_source_fails(workspace):
    write(
        workspace["project"] / "src/nba_prop_quant/production.py",
        "TARGETS = ['pts', 'reb']  # unauthorised model change\n",
    )

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "change model source")

    with pytest.raises(
        builder.BuildError,
        match="Frozen mathematical model source verification failed",
    ):
        build(workspace)


def test_tampered_gate3_artifact_fails_checksum_group(workspace):
    path = (
        workspace["gate3_artifact_dir"] / "probability_parameters.json"
    )

    path.write_text('{"assists": {"slope": 9.0}}\n', encoding="utf-8")

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "tamper artifact")

    with pytest.raises(builder.BuildError, match="hash mismatch"):
        build(workspace)


def test_deployment_manifest_policy_mismatch_fails(workspace):
    manifest_path = (
        workspace["gate3_artifact_dir"] / "deployment_manifest.json"
    )

    deployment = json.loads(manifest_path.read_text(encoding="utf-8"))
    deployment["gate3_policy"]["assists"] = "frozen_selected_v1"

    manifest_path.write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_sha256sums(
        manifest_path.parent,
        ["deployment_manifest.json", "probability_parameters.json"],
    )

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "shift deployed policy")

    with pytest.raises(
        builder.BuildError,
        match="Gate 3 10-prop policy mismatch",
    ):
        build(workspace)


def test_contract_policy_must_match_frozen_source_policy(workspace):
    def mutate(contract):
        contract["gate3_policy"]["points"] = "v2_role_increment"

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="does not match the policy",
    ):
        build(workspace)


def test_gate3_candidate_id_mismatch_fails(workspace):
    def mutate(contract):
        contract["gate3_candidate_policy_id"] = "nba_prop_quant_v2_gate3_bad"

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="Gate 3 candidate policy ID mismatch",
    ):
        build(workspace)


def test_t20_window_mismatch_fails(workspace):
    policy_path = (
        workspace["project"]
        / "research/v2_gate3_capture_lock/GATE3_CAPTURE_POLICY.json"
    )

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["primary_certification_window"]["window_label"] = "T-5m"
    policy["primary_certification_window"]["offset_minutes_before_tip"] = 5

    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_sha256sums(policy_path.parent, ["GATE3_CAPTURE_POLICY.json"])

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "shift capture window")

    with pytest.raises(
        builder.BuildError,
        match="capture window label mismatch",
    ):
        build(workspace)


def test_runtime_source_offset_must_match_locked_window(workspace):
    write(
        workspace["project"] / "src/nba_prop_quant/prospective_snapshot.py",
        "PRIMARY_OFFSET_MINUTES = 5\n",
    )

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "shift runtime offset")

    def mutate(contract):
        contract["frozen_model_source_files"] = [
            "scripts/10_predict_slate.py",
        ]

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="does not match the locked",
    ):
        build(workspace)


def test_non_ancestor_model_commit_fails(workspace):
    unrelated = "0" * 39 + "1"

    def mutate(contract):
        contract["model_source_commit"] = unrelated

    patch_contract(workspace, mutate)

    with pytest.raises(builder.BuildError):
        build(workspace)


def test_dirty_worktree_fails_without_override(workspace):
    write(
        workspace["project"] / "scripts/10_predict_slate.py",
        "# uncommitted edit\n",
    )

    with pytest.raises(
        builder.BuildError,
        match="uncommitted tracked changes",
    ):
        build(workspace)


# ---------------------------------------------------------------------------
# Security: credential exclusion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [".env", ".env.txt", "runtime.env", "credentials.json", "ftp.key"],
)
def test_secret_filenames_are_rejected(workspace, filename):
    write(workspace["project"] / filename, "PLACEHOLDER=1\n")

    def mutate(contract):
        contract["groups"]["packaging"]["files"].append(filename)

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="Refusing to package credential-bearing files",
    ):
        build(workspace)


@pytest.mark.parametrize(
    "payload",
    [
        "BDL_API_KEY=fixture-not-a-real-key-0000\n",
        "FTP_PASSWORD=fixture-not-a-real-password\n",
        "-----BEGIN RSA PRIVATE KEY-----\nfixture\n",
    ],
)
def test_secret_content_is_rejected(workspace, payload):
    target = workspace["project"] / "scripts/10_predict_slate.py"

    target.write_text(f"# predict\n{payload}", encoding="utf-8")

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "add fixture secret")

    def mutate(contract):
        contract["frozen_model_source_files"] = [
            "src/nba_prop_quant/production.py",
        ]

    patch_contract(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="content matched secret pattern",
    ):
        build(workspace)


def test_clean_bundle_contains_no_secret_filenames(workspace):
    result = build(workspace, keep_staging=True)
    staging = result["staging_path"]

    for path in staging.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()

        assert name != ".env"
        assert not name.endswith(".env")
        assert not name.endswith(".key")
        assert not name.endswith(".pem")
        assert "credential" not in name
        assert "__pycache__" not in path.as_posix()
