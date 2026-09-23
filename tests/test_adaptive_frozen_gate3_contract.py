from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import joblib

import nba_prop_quant.adaptive_training as adaptive_training


def _checksums(path: Path) -> dict[str, str]:
    result = {}

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if not line:
            continue

        digest, label = line.split(maxsplit=1)
        label = label.strip()

        if label.startswith("*"):
            label = label[1:]

        result[Path(label).name] = digest.lower()

    return result


def _concrete_fit_gate3_source() -> str:
    source_path = Path(adaptive_training.__file__).resolve()
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        if node.name != "fit_gate3":
            continue

        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and node.body[0].value.value is Ellipsis
        ):
            continue

        matches.append(
            ast.get_source_segment(text, node) or ""
        )

    assert len(matches) == 1
    return matches[0]


def test_adaptive_gate3_installs_frozen_certified_artifacts():
    segment = _concrete_fit_gate3_source()

    assert "HistGradientBoostingRegressor" not in segment
    assert "model.fit(" not in segment
    assert "stack_training.parquet" not in segment

    assert "SHA256SUMS.txt" in segment
    assert "role_minutes_model.joblib" in segment
    assert "role_state_seed.json" in segment
    assert "shutil.copy2" in segment
    assert "frozen_deployment_artifact" in segment


def test_frozen_gate3_bundle_hashes_and_feature_contract_match():
    source_path = Path(adaptive_training.__file__).resolve()
    repo_root = source_path.parents[2]

    bundle = (
        repo_root
        / "research"
        / "v2_gate3_deployment_artifacts"
    )

    required = (
        "deployment_manifest.json",
        "probability_parameters.json",
        "role_minutes_model.joblib",
        "role_state_seed.json",
    )

    checksums = _checksums(
        bundle / "SHA256SUMS.txt"
    )

    for name in required:
        assert name in checksums

        actual = hashlib.sha256(
            (bundle / name).read_bytes()
        ).hexdigest()

        assert actual == checksums[name]

    manifest = json.loads(
        (bundle / "deployment_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    payload = joblib.load(
        bundle / "role_minutes_model.joblib"
    )

    assert isinstance(payload, dict)

    assert list(payload["feature_names"]) == list(
        manifest["role_minutes_features"]
    )

    if (
        "training_rows" in payload
        and "role_minutes_training_rows" in manifest
    ):
        assert int(payload["training_rows"]) == int(
            manifest["role_minutes_training_rows"]
        )
