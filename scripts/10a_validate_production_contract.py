from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from nba_prop_quant.model import ModelBundle
from nba_prop_quant.production import (
    MONITORING_EDGE_GRID,
    TARGETS,
    load_json,
    load_verified_manifest_metadata,
)
from nba_prop_quant.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--allow-predeployment",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = get_settings()
    project_root = Path.cwd()

    manifest = load_verified_manifest_metadata(
        model_dir=settings.nba_prop_model_dir,
        project_root=project_root,
        allow_predeployment=args.allow_predeployment,
    )

    mean_policy = load_json(
        settings.nba_prop_model_dir
        / "mean_model_selection.json"
    )

    marginal_policy = load_json(
        settings.nba_prop_model_dir
        / "marginal_selection.json"
    )

    dependence_policy = load_json(
        settings.nba_prop_model_dir
        / "combo_dependence_policy.json"
    )

    calibration_policy = load_json(
        settings.nba_prop_model_dir
        / "market_probability_calibration_policy.json"
    )

    marginals = joblib.load(
        settings.nba_prop_model_dir
        / "marginals.joblib"
    )

    minutes = ModelBundle.load(
        settings.nba_prop_model_dir
        / "minutes.joblib"
    )

    if minutes.target_name != "minutes":
        raise RuntimeError(
            "minutes.joblib has wrong target_name"
        )

    for target in TARGETS:
        bundle = ModelBundle.load(
            settings.nba_prop_model_dir
            / f"{target}.joblib"
        )

        if bundle.target_name != target:
            raise RuntimeError(
                f"{target}.joblib target mismatch: "
                f"{bundle.target_name}"
            )

        if target not in mean_policy["targets"]:
            raise RuntimeError(
                f"Mean policy missing {target}"
            )

        if target not in marginal_policy["targets"]:
            raise RuntimeError(
                f"Marginal policy missing {target}"
            )

        if target not in marginals:
            raise RuntimeError(
                f"Production marginals missing {target}"
            )

        selected_distribution = (
            marginal_policy[
                "targets"
            ][target][
                "selected_distribution"
            ]
        )

        if selected_distribution != "zinb":
            raise RuntimeError(
                f"{target}: expected frozen ZINB, "
                f"got {selected_distribution}"
            )

        if marginals[target].kind != "zinb":
            raise RuntimeError(
                f"{target}: marginals.joblib kind is "
                f"{marginals[target].kind}"
            )

    expected_modes = {
        "pts": "xgb",
        "reb": "ensemble",
        "ast": "xgb",
        "stl": "xgb",
        "blk": "ensemble",
        "fg3m": "xgb",
    }

    observed_modes = {
        target: mean_policy[
            "targets"
        ][target][
            "selected_mode"
        ]
        for target in TARGETS
    }

    if observed_modes != expected_modes:
        raise RuntimeError(
            "Frozen mean-model policy mismatch: "
            f"{observed_modes}"
        )

    expected_lambdas = {
        "points_rebounds": 0.0,
        "points_assists": 0.0,
        "rebounds_assists": 0.85,
        "points_rebounds_assists": 0.0,
        "stocks": 0.0,
    }

    observed_lambdas = {
        key: float(
            dependence_policy[
                "combos"
            ][key][
                "production_lambda"
            ]
        )
        for key in expected_lambdas
    }

    if observed_lambdas != expected_lambdas:
        raise RuntimeError(
            "Frozen dependence policy mismatch: "
            f"{observed_lambdas}"
        )

    expected_calibration = {
        "assists": "prop",
        "blocks": "raw",
        "points": "prop",
        "points_assists": "prop",
        "points_rebounds": "prop",
        "points_rebounds_assists": "prop",
        "rebounds": "prop",
        "rebounds_assists": "prop",
        "steals": "raw",
        "threes": "prop",
    }

    observed_calibration = {
        key: calibration_policy[
            "props"
        ][key][
            "selected_method"
        ]
        for key in expected_calibration
    }

    if observed_calibration != expected_calibration:
        raise RuntimeError(
            "Frozen market-calibration policy mismatch: "
            f"{observed_calibration}"
        )

    print("=" * 118)
    print("PRODUCTION CONTRACT VALIDATION")
    print("=" * 118)
    print(
        f"Manifest: {manifest['freeze_id']} "
        f"({manifest['freeze_stage']})"
    )
    print()
    print("Mean models:")
    for target in TARGETS:
        print(
            f"  {target.upper():5s} "
            f"{observed_modes[target]}"
        )

    print()
    print("Marginals:")
    for target in TARGETS:
        print(
            f"  {target.upper():5s} ZINB"
        )

    print()
    print("Combo lambdas:")
    for key, value in observed_lambdas.items():
        print(
            f"  {key:28s} {value:.2f}"
        )

    print()
    print("Market calibration:")
    for key, value in observed_calibration.items():
        print(
            f"  {key:28s} {value}"
        )

    print()
    print(
        "Monitoring edge grid: "
        + ", ".join(
            f"{100*x:g}%"
            for x in MONITORING_EDGE_GRID
        )
    )

    print(
        "Auto betting threshold: NONE"
    )

    print()
    print(
        "PASS: frozen production policy/artifact contract is coherent."
    )


if __name__ == "__main__":
    main()
