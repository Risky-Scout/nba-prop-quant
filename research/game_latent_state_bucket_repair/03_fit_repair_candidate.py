#!/usr/bin/env python
"""Freeze the selected bucket-repair candidate and emit its factor spec.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Reads the winner chosen by ``02_inner_validation.py`` -- which never saw the
2024/2025 holdout -- refits it on the same training seasons accepted V1 used,
and writes a ``factor_spec.json`` in the format the *unmodified* V1 validation
driver consumes. That is what lets the final comparison run through the exact
same framework as the accepted control rather than a re-implementation.

Also writes ``bucket_repair_candidate.json``, the freeze record. Once that file
exists the candidate is closed: nothing downstream may retune it using holdout
results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state import (  # noqa: E402
    DEPENDENCE_MODEL_VERSION,
)
from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    ArtifactManifest,
    finalize_manifest,
    sha256_canonical,
    sha256_file,
    write_json,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.paths import (  # noqa: E402
    COVARIANCE_DIAGNOSTICS_NAME,
    FACTOR_LOADINGS_NAME,
    FACTOR_SPEC_NAME,
    RESIDUAL_DATASET_NAME,
)
from nba_prop_quant.research.game_latent_state.repair import (  # noqa: E402
    RepairSpec,
    fit_repaired_factors,
)

console = Console()

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

BRANCH = "research/nba-game-latent-state-shadow-v1-bucket-repair"
PARENT_V1_SHA = "1c5b8c93569ee25afd4eb4222158300702bd9471"

FACTOR_FAMILIES: tuple[str, ...] = (
    "game_pace_volume",
    "game_rebound_environment",
    "team_contrast_own_minus_opponent",
    "within_team_zero_sum_competition",
    "player_within_block_incumbent",
    "idiosyncratic_residual",
)


def git_sha(ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def spec_from_inner_validation(inner: dict, name: str | None = None) -> RepairSpec:
    """Rebuild the winning :class:`RepairSpec` from the inner-validation record."""
    selected = name or inner["selected_candidate"]
    payload = inner["candidates"][selected]["spec"]
    return RepairSpec(
        name=str(payload["name"]),
        family=str(payload["family"]),
        shrinkage=str(payload["shrinkage"]),
        shrink_z=float(payload["shrink_z"]),
        eb_family=str(payload["eb_family"]),
        k_game=int(payload["k_game"]),
        r_contrast=int(payload["r_contrast"]),
        role_column=payload["role_column"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"),
    )
    parser.add_argument(
        "--v1-artifact-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state"),
    )
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--candidate",
        default=None,
        help="override the inner-validation winner (research use only)",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_root)
    v1_dir = Path(args.v1_artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    inner_path = artifact_dir / "inner_validation.json"
    if not inner_path.exists():
        raise SystemExit(f"run 02_inner_validation.py first: {inner_path} missing")
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    if inner.get("holdout_used_for_selection", True):
        raise SystemExit("inner validation claims the holdout was used; refusing")

    spec_obj = spec_from_inner_validation(inner, args.candidate)
    console.rule(f"Freezing repair candidate: {spec_obj.name}")
    console.print(spec_obj.payload())

    # The accepted V1 spec fixes the training/validation split. Reusing it
    # verbatim is what makes the final comparison paired.
    v1_spec = json.loads((v1_dir / FACTOR_SPEC_NAME).read_text(encoding="utf-8"))
    training_seasons = [int(season) for season in v1_spec["training_seasons"]]
    validation_seasons = [int(season) for season in v1_spec["validation_seasons"]]
    if sorted(validation_seasons) != sorted(HOLDOUT_SEASONS):
        raise SystemExit(
            f"unexpected holdout seasons in the V1 spec: {validation_seasons}"
        )
    if set(training_seasons) & set(HOLDOUT_SEASONS):
        raise SystemExit("training seasons overlap the holdout")
    console.print(f"training seasons  : {training_seasons}")
    console.print(f"holdout seasons   : {validation_seasons} (never fitted)")

    residual_path = v1_dir / RESIDUAL_DATASET_NAME
    residuals = pd.read_parquet(residual_path)
    residuals["season"] = residuals["season"].astype(int)
    training = residuals[residuals["season"].isin(training_seasons)]
    if training.empty:
        raise SystemExit("no training rows")

    standardized, moment_constants = standardize_residuals(training, STATS)
    fit = fit_repaired_factors(
        standardized, STATS, spec_obj, bootstrap=args.bootstrap, seed=args.seed
    )

    diagnostics = fit.diagnostics()
    diagnostics["standardization_moments"] = {
        stat: {"mean": value[0], "sd": value[1]}
        for stat, value in moment_constants.items()
    }
    diagnostics["repair_spec"] = spec_obj.payload()
    diagnostics["r_contrast"] = int(fit.loadings.r_contrast)

    spec = {
        "dependence_model_version": DEPENDENCE_MODEL_VERSION,
        "stats": list(STATS),
        "factor_families": list(FACTOR_FAMILIES),
        "k_game": int(fit.k_game),
        "r_contrast": int(fit.loadings.r_contrast),
        "shrink_z": float(spec_obj.shrink_z),
        "shrinkage": spec_obj.shrinkage,
        "eb_family": spec_obj.eb_family,
        "role_column": spec_obj.role_column,
        "training_seasons": training_seasons,
        "validation_seasons": validation_seasons,
        "standardization_moments": diagnostics["standardization_moments"],
        "loadings": fit.loadings.to_payload(),
        "repair_candidate": spec_obj.payload(),
        "parent_shadow_v1_sha": PARENT_V1_SHA,
        "identification_note": (
            "Only the antisymmetric part of the own-team/opponent-team "
            "loadings is separately identified; the symmetric part is "
            "absorbed into the game-level factors. See covariance.py."
        ),
        "within_player_block_source": (
            "incumbent nba_prop_quant.copula.GaussianCopula; pinned, not refitted"
        ),
        "rank_note": (
            "The team-contrast Gram B is a full PSD Gram in general; V1's "
            "rank-1 truncation was a parsimony choice, not an identification "
            "constraint. Raising k_game and r_contrast lets the construction "
            "reproduce the shrunk blocks exactly, which is where roughly a "
            "quarter of the two target buckets was being lost."
        ),
    }
    spec["spec_hash"] = sha256_canonical(spec)

    spec_path = write_json(spec, artifact_dir / FACTOR_SPEC_NAME)
    diagnostics_path = write_json(
        diagnostics, artifact_dir / COVARIANCE_DIAGNOSTICS_NAME
    )

    contrast = fit.loadings.contrast_matrix
    competition = fit.loadings.competition
    loadings_frame = pd.DataFrame(
        {
            "stat": list(STATS),
            **{
                f"game_factor_{index + 1}": fit.loadings.game[:, index]
                for index in range(fit.loadings.k_game)
            },
            **{
                f"team_contrast_{index + 1}": contrast[:, index]
                for index in range(contrast.shape[1])
            },
            **(
                {
                    f"competition_factor_{index + 1}": competition[:, index]
                    for index in range(competition.shape[1])
                }
                if competition is not None
                else {}
            ),
        }
    )
    loadings_path = artifact_dir / FACTOR_LOADINGS_NAME
    loadings_frame.to_parquet(loadings_path, index=False)
    console.print(loadings_frame.round(4).to_string(index=False))

    selected_summary = inner["candidates"][spec_obj.name]
    freeze = {
        "frozen": True,
        "candidate_family": spec_obj.family,
        "candidate_name": spec_obj.name,
        "hyperparameters": spec_obj.payload(),
        "parameter_counts": spec_obj.parameter_count(len(STATS)),
        "inner_temporal_folds": inner["inner_folds"],
        "inner_selection_rule": inner["selection_rule"],
        "inner_metrics": {
            "mean_global_rmse": selected_summary["mean_global_rmse"],
            "mean_global_rms_z": selected_summary["mean_global_rms_z"],
            "mean_target_abs_error": selected_summary["mean_target_abs_error"],
            "mean_target_abs_z": selected_summary["mean_target_abs_z"],
            "mean_bucket_error": selected_summary["mean_bucket_error"],
            "mean_bucket_z_error": selected_summary["mean_bucket_z_error"],
            "target_overshoot_z": selected_summary["target_overshoot_z"],
            "target_overshoot_z_max_fold": selected_summary.get(
                "target_overshoot_z_max_fold"
            ),
        },
        "inner_metrics_control": {
            "mean_global_rmse": inner["candidates"]["v1_control"]["mean_global_rmse"],
            "mean_target_abs_error": inner["candidates"]["v1_control"][
                "mean_target_abs_error"
            ],
            "mean_target_abs_z": inner["candidates"]["v1_control"][
                "mean_target_abs_z"
            ],
            "mean_bucket_error": inner["candidates"]["v1_control"][
                "mean_bucket_error"
            ],
        },
        "selection_rationale": (
            f"Selected mechanically by the pre-registered rule in "
            f"02_inner_validation.py from {len(inner['candidates'])} candidates "
            f"scored on three strictly temporal pre-2024 folds. The winner "
            f"minimises the mean absolute inner-fold error of the two target "
            f"buckets among candidates that keep global latent bucket RMSE "
            f"within {inner['selection_rule']['max_global_rmse_ratio']}x the "
            f"control, keep both protected buckets within "
            f"{inner['selection_rule']['max_protected_z_degradation']} z of the "
            f"control, and do not overshoot either target bucket by more than "
            f"{inner['selection_rule']['overshoot_z']} fold-averaged z."
        ),
        "diagnosis": (
            "The attenuation is two independent mechanisms measured on pre-2024 "
            "data: a fixed-width soft threshold that costs 1.96/|z| of the "
            "signal (52.9% for teammate_reb_reb at z=3.70, 56.4% for "
            "teammate_ast_ast at z=3.47, against 6.3% for "
            "passer_ast_teammate_pts at z=31.15), and rank truncation that "
            "costs a further ~24-25%. The two targets lose through different "
            "rank channels: reb_reb through the rank-1 contrast factor, ast_ast "
            "through k_game=2."
        ),
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "parent_shadow_v1_sha": PARENT_V1_SHA,
        "source_production_sha": git_sha("origin/production/wizardofodds-integration"),
        "branch": BRANCH,
        "seed": int(args.seed),
        "bootstrap_draws": int(args.bootstrap),
        "code_sha": git_sha(),
        "factor_spec_hash": spec["spec_hash"],
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "note": (
            "Frozen before the 2024-2025 holdout was run. Retuning against "
            "holdout results after this point would invalidate the evaluation."
        ),
    }
    freeze_path = write_json(freeze, artifact_dir / "bucket_repair_candidate.json")

    manifest = ArtifactManifest(
        artifact_name="game_latent_state_bucket_repair_factor_model",
        source_production_sha=git_sha("origin/production/wizardofodds-integration"),
        source_production_ref="origin/production/wizardofodds-integration",
        code_sha=git_sha(),
        branch=BRANCH,
        seed=int(args.seed),
        training_cutoff=f"season<{min(HOLDOUT_SEASONS)}",
        seasons_used=training_seasons,
        training_seasons=training_seasons,
        validation_seasons=validation_seasons,
        input_fingerprints={RESIDUAL_DATASET_NAME: sha256_file(residual_path)},
        parameters={
            "repair_spec": spec_obj.payload(),
            "parameter_counts": spec_obj.parameter_count(len(STATS)),
            "bootstrap_draws": int(args.bootstrap),
            "spec_hash": spec["spec_hash"],
            "parent_shadow_v1_sha": PARENT_V1_SHA,
        },
        notes=[
            "Candidate selected on pre-2024 inner folds only.",
            "Loadings estimated from cross-player pairs only; the incumbent "
            "same-player block is never re-estimated.",
        ],
    )
    finalize_manifest(
        manifest,
        artifact_dir,
        outputs={
            FACTOR_SPEC_NAME: spec_path,
            FACTOR_LOADINGS_NAME: loadings_path,
            COVARIANCE_DIAGNOSTICS_NAME: diagnostics_path,
            "bucket_repair_candidate.json": freeze_path,
        },
        manifest_name="manifest.json",
        checksum_name="SHA256SUMS.factors.txt",
    )

    console.rule("Frozen")
    console.print("FINAL_CANDIDATE_FROZEN=YES")
    console.print(f"FINAL_CANDIDATE_FAMILY={spec_obj.family}")
    console.print(f"FINAL_CANDIDATE_PARAMETERS={json.dumps(spec_obj.payload())}")
    console.print("INNER_VALIDATION_COMPLETE=YES")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")
    console.print(f"\nk_game={fit.k_game}  r_contrast={fit.loadings.r_contrast}  "
                  f"r_competition={fit.r_competition}")
    same = fit.loadings.same_team_correlation()
    index = {stat: position for position, stat in enumerate(STATS)}
    console.print(
        f"implied teammate_reb_reb = {same[index['reb'], index['reb']]:+.6f}   "
        f"teammate_ast_ast = {same[index['ast'], index['ast']]:+.6f}"
    )
    console.print(
        f"min eig of implied same-team block = "
        f"{float(np.min(np.linalg.eigvalsh(0.5 * (same + same.T)))):+.6f}"
    )


if __name__ == "__main__":
    main()
