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


ADAPTIVE_INVARIANTS = {
    name: False for name in builder.REQUIRED_ADAPTIVE_INVARIANTS
}


def synthetic_adaptive_contract(
    project: Path,
    *,
    anchor: str,
    contract_path: Path,
    historically_pinned: list[str],
    locked: list[str],
    diverged: list[str],
) -> dict:
    """An adaptive serving source contract describing a synthetic worktree.

    Hashes are read from the worktree and from the anchor's blobs rather than
    restated, so a test that changes a locked file and regenerates this gets a
    contract that is internally consistent and still has to satisfy every
    cross-check the builder performs.
    """

    pinned = set(historically_pinned)

    entries = {}

    for relative in sorted(locked):
        current = sha256_bytes((project / relative).read_bytes())

        historical = sha256_bytes(
            subprocess.run(
                ["git", "cat-file", "blob", f"{anchor}:{relative}"],
                cwd=project,
                capture_output=True,
                check=True,
            ).stdout
        )

        entries[relative] = {
            "current_sha256": current,
            "historical_reference_sha256": historical,
            "historically_byte_pinned": relative in pinned,
            "matches_historical_reference": current == historical,
        }

        if current != historical:
            entries[relative]["divergence_reason"] = (
                "synthetic serving-correctness fix"
            )

    return {
        "contract_name": "synthetic_adaptive_serving_source_contract",
        "contract_version": 1,
        "diverged_from_historical_reference": sorted(diverged),
        "divergence_count": len(diverged),
        "historical_architecture_reference": anchor,
        "historical_contract_sha256": sha256_bytes(contract_path.read_bytes()),
        "invariants": dict(ADAPTIVE_INVARIANTS),
        "locked_serving_source_files": entries,
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

    adaptive_contract_path = project / "adaptive_serving_contract.json"

    pinned = synthetic_contract(model_commit)["frozen_model_source_files"]

    write(
        adaptive_contract_path,
        json.dumps(
            synthetic_adaptive_contract(
                project,
                anchor=model_commit,
                contract_path=contract_path,
                historically_pinned=pinned,
                locked=[*pinned, "src/nba_prop_quant/__init__.py"],
                diverged=[],
            ),
            indent=2,
            sort_keys=True,
        )
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
        "adaptive_contract_path": adaptive_contract_path,
        "historically_pinned": pinned,
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
        "adaptive_contract_path": workspace["adaptive_contract_path"],
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

    # The adaptive contract hashes the historical contract it promises not to
    # amend, so it has to be reissued whenever that file is rewritten or the
    # build would fail on the stale hash rather than on what the test changed.
    relock(workspace)


def patch_adaptive(workspace: dict, mutate) -> None:
    adaptive = json.loads(
        workspace["adaptive_contract_path"].read_text(encoding="utf-8")
    )

    mutate(adaptive)

    workspace["adaptive_contract_path"].write_text(
        json.dumps(adaptive, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "adjust adaptive contract")


def relock(workspace: dict, *, diverged: list[str] | None = None) -> dict:
    """Reissue the adaptive contract from the worktree as it stands now."""

    locked = sorted(
        json.loads(
            workspace["adaptive_contract_path"].read_text(encoding="utf-8")
        )["locked_serving_source_files"]
    )

    if diverged is None:
        diverged = [
            relative
            for relative in locked
            if sha256_bytes((workspace["project"] / relative).read_bytes())
            != sha256_bytes(
                subprocess.run(
                    [
                        "git",
                        "cat-file",
                        "blob",
                        f"{workspace['model_commit']}:{relative}",
                    ],
                    cwd=workspace["project"],
                    capture_output=True,
                    check=True,
                ).stdout
            )
        ]

    adaptive = synthetic_adaptive_contract(
        workspace["project"],
        anchor=workspace["model_commit"],
        contract_path=workspace["contract_path"],
        # Read from the contract as it stands rather than from the fixture's
        # original list: historically_byte_pinned has to agree with whatever
        # the runtime contract pins right now, and a test that narrowed it is
        # testing something else.
        historically_pinned=json.loads(
            workspace["contract_path"].read_text(encoding="utf-8")
        )["frozen_model_source_files"],
        locked=locked,
        diverged=diverged,
    )

    workspace["adaptive_contract_path"].write_text(
        json.dumps(adaptive, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "relock serving sources")

    return adaptive


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


def anchor_blob_sha256(relative: str) -> str:
    blob = subprocess.run(
        ["git", "cat-file", "blob", f"{FROZEN_MODEL_COMMIT}:{relative}"],
        cwd=PROJECT,
        capture_output=True,
        check=True,
    )

    return hashlib.sha256(blob.stdout).hexdigest()


def adaptive_serving_problems(read_bytes=None) -> list[str]:
    """Verify every locked serving source against the adaptive contract.

    read_bytes lets a test substitute the content of one file so tampering can
    be exercised without touching the worktree.
    """
    if read_bytes is None:

        def read_bytes(relative: str) -> bytes:
            return (PROJECT / relative).read_bytes()

    problems: list[str] = []

    for relative, entry in sorted(
        adaptive_serving_contract()["locked_serving_source_files"].items()
    ):
        observed = hashlib.sha256(read_bytes(relative)).hexdigest()

        if observed != entry["current_sha256"]:
            problems.append(
                f"{relative}: {observed} does not match the locked "
                f"{entry['current_sha256']}"
            )

    return problems


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

    # A declared divergence among the historically pinned files must be a real
    # one, so the exemption cannot quietly wave through an unchanged file.
    for relative in sorted(declared & set(contract["frozen_model_source_files"])):
        with pytest.raises(builder.BuildError):
            builder.verify_frozen_model_sources(
                PROJECT,
                FROZEN_MODEL_COMMIT,
                [relative],
            )


def test_every_declared_divergence_genuinely_differs():
    """Covers the locked files the historical contract never pinned too."""
    adaptive = adaptive_serving_contract()

    declared = adaptive["diverged_from_historical_reference"]

    assert declared, "the contract declares no divergence"
    assert adaptive["divergence_count"] == len(declared)

    for relative in declared:
        entry = adaptive["locked_serving_source_files"][relative]

        assert entry["current_sha256"] != entry["historical_reference_sha256"]
        assert entry["matches_historical_reference"] is False
        assert entry["divergence_reason"].strip()


def test_every_declared_non_divergence_matches_the_anchor():
    adaptive = adaptive_serving_contract()

    declared = set(adaptive["diverged_from_historical_reference"])

    for relative, entry in sorted(
        adaptive["locked_serving_source_files"].items()
    ):
        if relative in declared:
            continue

        assert entry["matches_historical_reference"] is True

        observed = hashlib.sha256(
            (PROJECT / relative).read_bytes()
        ).hexdigest()

        assert observed == anchor_blob_sha256(relative)


def test_live_serving_modules_are_locked():
    """slate.py and storage.py are on the serving path and must be pinned.

    The historical contract stages both into the bundle but never listed them
    in frozen_model_source_files, so their content shipped unpinned. Step 3B
    changed both, which is why the adaptive contract has to cover them.
    """
    locked = set(adaptive_serving_contract()["locked_serving_source_files"])

    assert "src/nba_prop_quant/slate.py" in locked
    assert "src/nba_prop_quant/storage.py" in locked

    predict = (PROJECT / "scripts" / "10_predict_slate.py").read_text(
        encoding="utf-8"
    )
    price = (PROJECT / "scripts" / "15_price_markets.py").read_text(
        encoding="utf-8"
    )

    # The import evidence that puts them on the serving path.
    assert "from nba_prop_quant.slate import" in predict
    assert "from nba_prop_quant.storage import" in predict
    assert "from nba_prop_quant.storage import" in price


def test_locked_set_covers_the_historical_pinned_set():
    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))
    locked = adaptive_serving_contract()["locked_serving_source_files"]

    assert set(locked) >= set(contract["frozen_model_source_files"])

    for relative, entry in sorted(locked.items()):
        assert entry["historically_byte_pinned"] == (
            relative in set(contract["frozen_model_source_files"])
        )


def test_adaptive_serving_sources_match_their_locked_hashes():
    assert adaptive_serving_problems() == []


def test_changing_a_locked_serving_source_fails_verification():
    """A silent edit to any locked serving source must be detected."""
    adaptive = adaptive_serving_contract()

    for target in (
        "src/nba_prop_quant/slate.py",
        "src/nba_prop_quant/storage.py",
        "scripts/15_price_markets.py",
    ):
        assert target in adaptive["locked_serving_source_files"]

        def tampered(relative: str, _target=target) -> bytes:
            content = (PROJECT / relative).read_bytes()

            return content + b"\n# silently edited\n" if relative == _target else content

        problems = adaptive_serving_problems(tampered)

        assert len(problems) == 1
        assert problems[0].startswith(target)


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
        assert (
            anchor_blob_sha256(relative)
            == entry["historical_reference_sha256"]
        ), f"{relative} anchor bytes moved"


def test_adaptive_serving_source_matches_its_contract():
    """What adaptive serving runs is pinned, not merely exempted.

    The locked set is a superset of the historically pinned nine, because
    serving also depends on modules that contract never pinned.
    """
    adaptive = adaptive_serving_contract()

    locked = adaptive["locked_serving_source_files"]

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
        "model_family_or_routing_changed",
        "gate3_routing_changed",
        "calibration_methodology_changed",
        "dependence_methodology_changed",
        "marginal_family_changed",
        "mean_model_routing_changed",
        "training_feature_definitions_changed",
        "feature_definitions_changed",
        "t20_certification_protocol_changed",
        "fitted_model_artifacts_changed",
    ):
        assert invariants[name] is False


def test_runtime_bundle_limitation_stays_documented():
    """Expanding the lock must not quietly drop the deferred obligation."""
    limitation = adaptive_serving_contract()["known_limitation"]

    assert "19_build_wizardofodds_runtime_bundle.py" in limitation
    assert "Step 3D" in limitation

    addendum = (
        PROJECT
        / "docs"
        / "wizardofodds"
        / "V2_REFERENCE_PROVENANCE_ADDENDUM.md"
    ).read_text(encoding="utf-8")

    assert "19_build_wizardofodds_runtime_bundle.py" in addendum


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
        adaptive_sources=builder.verify_adaptive_serving_sources(
            PROJECT,
            adaptive_serving_contract(),
            anchor_commit=contract["model_source_commit"],
            historical_contract_path=REAL_CONTRACT_PATH,
            historically_pinned=contract["frozen_model_source_files"],
        ),
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

    adaptive = manifest["adaptive_serving_source_verification"]

    assert adaptive["verified"] is True
    assert adaptive["historical_architecture_reference"] == FROZEN_MODEL_COMMIT
    assert set(adaptive["files"]) >= set(contract["frozen_model_source_files"])

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


# ---------------------------------------------------------------------------
# The adaptive serving source lock the builder now reads
# ---------------------------------------------------------------------------


def diverge_entry_point(workspace: dict) -> str:
    """Change a pinned entry point after the anchor, the way production did."""

    relative = "scripts/15_price_markets.py"

    write(
        workspace["project"] / relative,
        "# price\n# --frozen-bundle-root, so the bundle can be used at all\n",
    )

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "serving correction")

    return relative


def test_a_diverged_entry_point_can_be_packaged_once_it_is_declared(workspace):
    """The blocker: the anchor check alone cannot pass on a serving tree.

    Two of production's real divergences add ``--frozen-bundle-root``, without
    which the entry points resolve the frozen manifest against the working
    directory and no bundle is usable. The anchor-only check therefore refused
    the only source able to consume its own output.
    """

    relative = diverge_entry_point(workspace)

    with pytest.raises(builder.BuildError, match="locked"):
        build(workspace)

    relock(workspace)

    manifest = build(workspace)["manifest"]

    adaptive = manifest["adaptive_serving_source_verification"]

    assert adaptive["verified"] is True
    assert adaptive["diverged_from_historical_reference"] == [relative]

    # Still pinned, just not to the anchor: the lock carries this file's bytes.
    assert adaptive["files"][relative] == sha256_bytes(
        (workspace["project"] / relative).read_bytes()
    )

    # And the anchor block no longer claims to have verified it.
    anchored = manifest["frozen_model_source_verification"]["files"]

    assert relative not in anchored
    assert "src/nba_prop_quant/production.py" in anchored


def test_every_pinned_file_stays_pinned_by_one_block_or_the_other(workspace):
    diverge_entry_point(workspace)

    relock(workspace)

    manifest = build(workspace)["manifest"]

    contract = json.loads(
        workspace["contract_path"].read_text(encoding="utf-8")
    )

    anchored = set(manifest["frozen_model_source_verification"]["files"])

    locked = set(manifest["adaptive_serving_source_verification"]["files"])

    assert set(contract["frozen_model_source_files"]) <= anchored | locked

    # The lock is the wider of the two: it also covers serving sources the
    # historical contract staged into the bundle but never byte-pinned.
    assert locked > set(contract["frozen_model_source_files"])


def test_an_undeclared_divergence_still_fails_the_build(workspace):
    relative = diverge_entry_point(workspace)

    relock(workspace, diverged=[])

    with pytest.raises(
        builder.BuildError,
        match="differs from the anchor without being declared",
    ):
        build(workspace)

    assert relative


def test_declaring_an_unchanged_file_as_diverged_fails_the_build(workspace):
    def mutate(adaptive):
        adaptive["diverged_from_historical_reference"] = [
            "scripts/10_predict_slate.py"
        ]
        adaptive["divergence_count"] = 1

    patch_adaptive(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="declared a divergence but is byte-identical to the anchor",
    ):
        build(workspace)


def test_a_divergence_without_a_justification_fails_the_build(workspace):
    relative = diverge_entry_point(workspace)

    relock(workspace)

    def mutate(adaptive):
        adaptive["locked_serving_source_files"][relative][
            "divergence_reason"
        ] = "   "

    patch_adaptive(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="diverges from the anchor with no justification",
    ):
        build(workspace)


def test_a_lock_anchored_elsewhere_fails_the_build(workspace):
    def mutate(adaptive):
        adaptive["historical_architecture_reference"] = "0" * 40

    patch_adaptive(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="adaptive serving contract is anchored to",
    ):
        build(workspace)


def test_a_lock_that_does_not_hash_the_contract_it_cites_fails(workspace):
    """The lock promises not to amend the historical contract.

    It records that contract's hash to make the promise checkable, so a lock
    carried over to a rewritten historical contract must be refused rather
    than silently reinterpreting it.
    """

    def mutate(adaptive):
        adaptive["historical_contract_sha256"] = "0" * 64

    patch_adaptive(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="records historical contract sha256",
    ):
        build(workspace)


def test_recorded_anchor_bytes_must_really_be_the_anchors(workspace):
    def mutate(adaptive):
        adaptive["locked_serving_source_files"][
            "src/nba_prop_quant/production.py"
        ]["historical_reference_sha256"] = "0" * 64

    patch_adaptive(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="recorded anchor bytes are not the bytes at",
    ):
        build(workspace)


def test_a_lock_that_misses_a_pinned_file_fails_the_build(workspace):
    def mutate(adaptive):
        del adaptive["locked_serving_source_files"][
            "src/nba_prop_quant/production.py"
        ]

    patch_adaptive(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="does not lock every historically pinned source",
    ):
        build(workspace)


def test_a_lock_that_stops_asserting_the_mathematics_held_fails(workspace):
    for name in builder.REQUIRED_ADAPTIVE_INVARIANTS:

        def mutate(adaptive, _name=name):
            adaptive["invariants"][_name] = True

        workspace_copy = dict(workspace)

        patch_adaptive(workspace_copy, mutate)

        with pytest.raises(
            builder.BuildError,
            match=f"does not assert {name} is false",
        ):
            build(workspace_copy)

        def restore(adaptive, _name=name):
            adaptive["invariants"][_name] = False

        patch_adaptive(workspace_copy, restore)


def test_a_miscounted_divergence_list_fails_the_build(workspace):
    def mutate(adaptive):
        adaptive["divergence_count"] = 7

    patch_adaptive(workspace, mutate)

    with pytest.raises(
        builder.BuildError,
        match="declares 0 divergence\\(s\\) but counts 7",
    ):
        build(workspace)


def test_a_missing_lock_fails_the_build(workspace):
    workspace["adaptive_contract_path"].unlink()

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "drop the lock")

    with pytest.raises(
        builder.BuildError,
        match="Adaptive serving source contract missing",
    ):
        build(workspace)


def test_the_bundle_records_which_lock_packaged_it(workspace):
    diverge_entry_point(workspace)

    adaptive_contract = relock(workspace)

    result = build(workspace)

    bundle = extract(workspace, result)

    reloaded = json.loads(
        (bundle / builder.RUNTIME_MANIFEST_NAME).read_text(encoding="utf-8")
    )

    recorded = reloaded["adaptive_serving_source_verification"]

    assert recorded["contract_name"] == adaptive_contract["contract_name"]
    assert recorded["contract_version"] == adaptive_contract["contract_version"]
    assert recorded["historical_contract_sha256"] == sha256_bytes(
        workspace["contract_path"].read_bytes()
    )

    # The frozen verifier still accepts the bundle: the new block is metadata
    # beside the hash set, not a change to it.
    assert verify_frozen_manifest(bundle)["freeze_id"] == (
        result["manifest"]["freeze_id"]
    )


def test_the_real_lock_lets_the_real_contract_be_packaged():
    """The blocker, on the real tree rather than a fixture.

    The anchor-only check fails on the three serving scripts production
    corrected. Splitting the same nine files between the two pins leaves every
    one of them byte-verified and nothing unaccounted for.
    """

    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    anchor = contract["model_source_commit"]

    pinned = contract["frozen_model_source_files"]

    with pytest.raises(builder.BuildError) as refusal:
        builder.verify_frozen_model_sources(PROJECT, anchor, pinned)

    for relative in (
        "scripts/10_predict_slate.py",
        "scripts/11_collect_snapshots.py",
        "scripts/15_price_markets.py",
    ):
        assert relative in str(refusal.value)

    adaptive = builder.verify_adaptive_serving_sources(
        PROJECT,
        adaptive_serving_contract(),
        anchor_commit=anchor,
        historical_contract_path=REAL_CONTRACT_PATH,
        historically_pinned=pinned,
    )

    diverged = set(adaptive["diverged_from_historical_reference"])

    anchored = builder.verify_frozen_model_sources(
        PROJECT,
        anchor,
        [relative for relative in pinned if relative not in diverged],
    )

    assert set(pinned) <= set(anchored) | set(adaptive["files"])
    assert set(anchored) & diverged == set()
    assert len(adaptive["files"]) > len(pinned)


def test_the_real_lock_diverges_only_on_command_line_drivers():
    """The mathematics is still anchor-pinned and cannot be relocked."""

    contract = json.loads(REAL_CONTRACT_PATH.read_text(encoding="utf-8"))

    pinned = set(contract["frozen_model_source_files"])

    adaptive = adaptive_serving_contract()

    for relative in adaptive["diverged_from_historical_reference"]:
        if relative in pinned:
            assert relative.startswith(builder.DIVERGEABLE_PINNED_PREFIX)

    mathematics = sorted(
        relative
        for relative in pinned
        if not relative.startswith(builder.DIVERGEABLE_PINNED_PREFIX)
    )

    assert mathematics == [
        "src/nba_prop_quant/distributions.py",
        "src/nba_prop_quant/gate3_v2.py",
        "src/nba_prop_quant/pricing.py",
        "src/nba_prop_quant/production.py",
        "src/nba_prop_quant/prospective_snapshot.py",
    ]

    builder.verify_frozen_model_sources(
        PROJECT,
        contract["model_source_commit"],
        mathematics,
    )


def test_the_deferred_obligation_is_recorded_as_discharged():
    """The limitation note must track reality, in both directions.

    Version 9 recorded the builder as unable to package this source. The
    builder now reads the contract, so a note still saying otherwise would be
    the misleading half of the same problem.
    """

    adaptive = adaptive_serving_contract()

    limitation = adaptive["known_limitation"]

    assert "19_build_wizardofodds_runtime_bundle.py" in limitation
    assert "Step 3D" in limitation
    assert "discharged" in limitation

    history = adaptive["contract_version_history"]

    assert str(adaptive["contract_version"]) in history
    assert "Step 3D" in history[str(adaptive["contract_version"])]

    builder_source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "nba_prop_quant_v2_adaptive_serving_source_contract" in builder_source

    addendum = (
        PROJECT
        / "docs"
        / "wizardofodds"
        / "V2_REFERENCE_PROVENANCE_ADDENDUM.md"
    ).read_text(encoding="utf-8")

    assert "19_build_wizardofodds_runtime_bundle.py" in addendum
    assert "adaptive_serving_source_verification" in addendum


#: Digest of every lock-bearing field of the adaptive serving source contract
#: as version 9 recorded them, with the collector's own lock entry left out.
#:
#: The pin used to cover the whole subset, because at the time the only
#: contract edit in flight was the note and the version. Version 10 then
#: relocked :data:`THE_ONE_LOCK_ANOTHER_CHANGE_MOVED` as part of the
#: capture-concurrency correction -- a real lock move, separately declared and
#: recorded in the version history -- so a whole-subset literal taken before
#: it would now assert something false, and widening it to the new value would
#: assert nothing at all.
#:
#: So the pin is narrowed to what this test was always for: discharging the
#: Step 3D obligation moved nothing. It covers every lock-bearing field except
#: the single entry another reviewed change is allowed to have moved, and the
#: test accounts for that entry field by field rather than exempting it.
#: Pinned as a literal rather than read from a git ref so it stays meaningful
#: after this lands and the ref moves on.
ADAPTIVE_LOCK_DIGEST_AT_VERSION_NINE_WITHOUT_THE_COLLECTOR = (
    "2492dd67ff7b012aae25116722f026c70d684928230eae0c0d7c6ed1902c7573"
)

#: The locked file the capture-concurrency correction relocked, and the only
#: one any change landing beside this one may have moved.
THE_ONE_LOCK_ANOTHER_CHANGE_MOVED = "scripts/11_collect_snapshots.py"

#: That file's anchor hash, which a relock must never touch: a relock records
#: what the working tree now is, never what the architecture reference was.
COLLECTOR_HISTORICAL_REFERENCE = (
    "dd0c3ed393ec04bbef521c4ff0d719bf3744ee04ef6405e83bc19b99233b6535"
)

ADAPTIVE_LOCK_BEARING_FIELDS = (
    "coverage_basis",
    "diverged_from_historical_reference",
    "divergence_count",
    "historical_architecture_reference",
    "historical_contract_sha256",
    "invariant_basis",
    "invariants",
    "locked_serving_source_files",
)


def test_discharging_the_obligation_moved_no_lock():
    """The contract edit must be the note and the version, nothing else.

    Every lock, hash, divergence and invariant has to be what version 9
    recorded, so teaching the builder to read this contract cannot have
    quietly relocked anything on the way. The one exception is the collector,
    which a separate reviewed change relocked, and the exception is read out
    field by field below rather than taken on trust.
    """

    adaptive = adaptive_serving_contract()

    subset = {
        field: adaptive[field] for field in ADAPTIVE_LOCK_BEARING_FIELDS
    }

    locked = dict(subset["locked_serving_source_files"])
    collector = locked.pop(THE_ONE_LOCK_ANOTHER_CHANGE_MOVED)
    subset["locked_serving_source_files"] = locked

    digest = hashlib.sha256(
        json.dumps(subset, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    assert digest == ADAPTIVE_LOCK_DIGEST_AT_VERSION_NINE_WITHOUT_THE_COLLECTOR

    # The collector's entry may have been relocked, but only its current hash
    # and the reason recording why. Its anchor, its pinned status and its
    # divergence from the anchor are the reviewed version 9 values.
    assert collector["historical_reference_sha256"] == (
        COLLECTOR_HISTORICAL_REFERENCE
    )
    assert collector["historically_byte_pinned"] is True
    assert collector["matches_historical_reference"] is False

    # And a moved lock is only legitimate if the history says it moved, so the
    # relock has to be named by an entry past the version the pin is taken at.
    assert any(
        THE_ONE_LOCK_ANOTHER_CHANGE_MOVED in entry
        for version, entry in adaptive["contract_version_history"].items()
        if int(version) > 9
    )

    # And the version did move, so the edit is not passing itself off as the
    # reviewed version 9 contract.
    assert adaptive["contract_version"] > 9


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
    """An edit to a mathematical source is refused, declared or not.

    Undeclared, the serving lock catches it, because that lock covers more
    files than the anchor one does. Declared and relocked -- the loophole the
    adaptive mechanism would otherwise open -- the builder still refuses,
    because a library module under ``src/`` that the runtime contract
    byte-pins is the mathematics and nothing may relock it.
    """

    write(
        workspace["project"] / "src/nba_prop_quant/production.py",
        "TARGETS = ['pts', 'reb']  # unauthorised model change\n",
    )

    git(workspace["project"], "add", "-A")
    git(workspace["project"], "commit", "-q", "-m", "change model source")

    with pytest.raises(
        builder.BuildError,
        match="differs from the adaptive serving contract's locked",
    ):
        build(workspace)

    relock(workspace)

    with pytest.raises(
        builder.BuildError,
        match="may not relock them, because they are the mathematics",
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
