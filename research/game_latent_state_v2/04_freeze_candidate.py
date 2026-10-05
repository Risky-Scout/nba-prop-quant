#!/usr/bin/env python
"""Freeze exactly one Shadow V2 candidate and emit its factor spec.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Reads the winner ``03_inner_screening.py`` selected on pre-2024 folds, refits
it on the same training seasons accepted V1 and the accepted repair used, and
writes a ``factor_spec.json`` in the format the validation framework consumes.
Fitting on the control's own training seasons is what makes the final
comparison paired: five models, one set of held-out games, one seed.

Nothing here chooses anything. Every hyperparameter is read from the screening
artifact, every season boundary from the accepted V1 spec, and every bridge
target from ``count_bridge.json``, which was built on the training seasons
only. The one quantity this driver computes that the screen did not is the
fit itself, on the union of the folds rather than on each fold.

Once ``shadow_v2_candidate.json`` exists the candidate is closed. Retuning it
against anything the 2024-2025 validation reports would invalidate the
evaluation, and the freeze record says so in a field the gate evaluator
reads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
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
from nba_prop_quant.research.game_latent_state.v2 import (  # noqa: E402
    V2Spec,
    fit_v2_factors,
    role_cell_report,
    role_pair_moments,
)
from nba_prop_quant.research.game_latent_state.validation import (  # noqa: E402
    DEPENDENCE_BUCKETS,
)

console = Console()

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

BRANCH = "research/nba-game-latent-state-shadow-v2-structural"
PARENT_V1_SHA = "1c5b8c93569ee25afd4eb4222158300702bd9471"
PARENT_REPAIR_SHA = "c965451de2662b953f45c791157f3826207f753f"

#: Unchanged from V1. The symmetric subspace is not a new *family* -- it is a
#: pair of loading columns inside the existing game-level and team-contrast
#: families, which is precisely why it cancels from the cross-team block.
FACTOR_FAMILIES: tuple[str, ...] = (
    "game_pace_volume",
    "game_rebound_environment",
    "team_contrast_own_minus_opponent",
    "within_team_zero_sum_competition",
    "symmetric_same_team_subspace",
    "role_conditioned_symmetric_deviation",
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


def spec_from_screening(
    screening: dict,
    name: str | None = None,
    resolution: dict | None = None,
) -> V2Spec:
    """Rebuild the winning :class:`V2Spec` from the screening record.

    ``resolution`` is the axis-indifference resolution ``03b`` wrote, if one
    exists. It can only substitute the same-team axis's two fields, and only
    with values that axis's own grid already contained, so reading it cannot
    introduce a setting the screen did not score.
    """
    selected = name or screening["selected_candidate"]
    payload = screening["candidates"][selected]["spec"]
    r_symmetric = int(payload["r_symmetric"])
    symmetric_mode = str(payload["symmetric_mode"])
    if resolution is not None and resolution["stage_two_winner"] == selected:
        r_symmetric = int(resolution["resolved_r_symmetric"])
        symmetric_mode = str(resolution["resolved_symmetric_mode"])
    return V2Spec(
        name=str(payload["name"]),
        same_shrinkage=str(payload["same_shrinkage"]),
        same_eb_family=str(payload["same_eb_family"]),
        cross_shrinkage=str(payload["cross_shrinkage"]),
        cross_eb_family=str(payload["cross_eb_family"]),
        shrink_z=float(payload["shrink_z"]),
        k_game=int(payload["k_game"]),
        r_contrast=int(payload["r_contrast"]),
        r_symmetric=r_symmetric,
        symmetric_mode=symmetric_mode,
        role_deviation=bool(payload["role_deviation"]),
        role_column=payload["role_column"],
        bridge_weight=float(payload["bridge_weight"]),
        temporal_treatment=str(payload["temporal_treatment"]),
    )


def bridge_targets_from_artifact(
    bridge: dict,
    stats: tuple[str, ...],
) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    """The same-team latent correlations the training count moments imply.

    Read from ``count_bridge.json`` rather than recomputed, so the targets
    behind the frozen fit are the audited ones. A pair whose inversion was not
    identified -- the observed count-space moment lies outside the bridge's
    range on the admissible interval -- is dropped and reported, exactly as
    :func:`countspace.invert_bridge_block` does per fold.
    """
    curves = bridge["bridges"]["same_team"]
    targets: dict[tuple[str, str], float] = {}
    rejections: dict[str, str] = {}
    for position, first in enumerate(stats):
        for second in stats[position:]:
            key = f"{first}_{second}"
            entry = curves.get(key) or curves.get(f"{second}_{first}")
            if entry is None:
                rejections[key] = "no bridge curve in the artifact"
                continue
            if not entry["required_from_count_identified"]:
                rejections[key] = str(entry["required_from_count_rejection"])
                continue
            targets[(first, second)] = float(entry["required_from_count"])
    return targets, rejections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state_v2"),
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
        help="override the screening winner (research use only)",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_root)
    v1_dir = Path(args.v1_artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    screening_path = artifact_dir / "inner_screening.json"
    if not screening_path.exists():
        raise SystemExit(f"run 03_inner_screening.py first: {screening_path} missing")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    if screening.get("holdout_used_for_selection", True):
        raise SystemExit("the screening claims the holdout was used; refusing")

    temporal_path = artifact_dir / "temporal_diagnostic.json"
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    if temporal.get("holdout_used_for_selection", True):
        raise SystemExit("the temporal diagnostic claims the holdout was used")

    bridge_path = artifact_dir / "count_bridge.json"
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    if bridge.get("hyperparameter_selected_here", True):
        raise SystemExit("the count bridge claims it selected a hyperparameter")

    # The same-team axis's indifference resolution, when ``03b`` has written
    # one. It is a decision rule re-applied to the metrics the screen already
    # persisted, so it substitutes a value from that axis's own grid without
    # anything being refitted.
    resolution_path = artifact_dir / "axis_indifference_resolution.json"
    resolution = (
        json.loads(resolution_path.read_text(encoding="utf-8"))
        if resolution_path.exists()
        else None
    )
    if resolution is not None:
        if resolution["holdout_used_for_selection"]:
            raise SystemExit("the axis resolution claims the holdout was used")
        if resolution["resolves"]["artifact_sha256"] != sha256_file(screening_path):
            raise SystemExit(
                "the axis resolution was computed against a different "
                "inner_screening.json; rerun 03b_resolve_axis_indifference.py"
            )
        if resolution["refitted_anything"]:
            raise SystemExit("the axis resolution claims it refitted something")

    spec = spec_from_screening(screening, args.candidate, resolution)
    console.rule(f"Freezing Shadow V2 candidate: {spec.name}")
    console.print(spec.payload())
    if resolution is not None and resolution["selection_changed"]:
        console.print(
            f"  same-team axis resolved to {resolution['resolved_selection']} "
            f"(screen recorded {resolution['recorded_selection']}): "
            f"{resolution['reason']}"
        )

    v1_spec = json.loads((v1_dir / FACTOR_SPEC_NAME).read_text(encoding="utf-8"))
    training_seasons = [int(season) for season in v1_spec["training_seasons"]]
    validation_seasons = [int(season) for season in v1_spec["validation_seasons"]]
    if sorted(validation_seasons) != sorted(HOLDOUT_SEASONS):
        raise SystemExit(f"unexpected holdout seasons: {validation_seasons}")
    if set(training_seasons) & set(HOLDOUT_SEASONS):
        raise SystemExit("training seasons overlap the holdout")
    if sorted(bridge["seasons_used"]) != sorted(training_seasons):
        raise SystemExit(
            "the count bridge was built on "
            f"{bridge['seasons_used']}, not the training seasons "
            f"{training_seasons}"
        )
    console.print(f"training seasons : {training_seasons}")
    console.print(f"holdout seasons  : {validation_seasons} (never fitted)")

    residual_path = v1_dir / RESIDUAL_DATASET_NAME
    residuals = pd.read_parquet(residual_path)
    residuals["season"] = residuals["season"].astype(int)
    training = residuals[residuals["season"].isin(training_seasons)]
    if training.empty:
        raise SystemExit("no training rows")
    leaked = sorted(set(training["season"].unique()) & set(HOLDOUT_SEASONS))
    if leaked:
        raise SystemExit(f"holdout seasons leaked into the fit: {leaked}")

    standardized, moment_constants = standardize_residuals(training, STATS)
    targets, rejections = bridge_targets_from_artifact(bridge, STATS)
    if spec.bridge_weight > 0 and not targets:
        raise SystemExit("the candidate wants bridge targets and none are identified")
    console.print(
        f"bridge targets   : {len(targets)} identified, "
        f"{len(rejections)} rejected "
        f"(applied at weight {spec.bridge_weight:g})"
    )

    role_moments = role_pair_moments(
        standardized, STATS, bootstrap=args.bootstrap, seed=args.seed
    )
    fit = fit_v2_factors(
        standardized,
        STATS,
        spec=spec,
        bootstrap=args.bootstrap,
        seed=args.seed,
        bridge_targets=targets if spec.bridge_weight > 0 else None,
        role_moments=role_moments,
    )

    # The same candidate with its role layer switched off. It is never
    # simulated and never promoted; it exists so the holdout can measure the
    # role layer against the fit it actually replaces rather than against
    # accepted V1, which differs from it in three other ways.
    role_blind_twin = (
        fit_v2_factors(
            standardized,
            STATS,
            spec=replace(spec, role_deviation=False),
            bootstrap=args.bootstrap,
            seed=args.seed,
            bridge_targets=targets if spec.bridge_weight > 0 else None,
            role_moments=role_moments,
        )
        if spec.role_deviation
        else fit
    )

    # And the same candidate with the symmetric subspace switched off, so the
    # same-team target is never applied at all. This is the reference the
    # cross-team isolation claim is stated against: whatever the same-team work
    # does, ``A - B`` has to come back as this twin's. Also never simulated.
    no_repair_twin = (
        fit_v2_factors(
            standardized,
            STATS,
            spec=replace(spec, r_symmetric=0, role_deviation=False),
            bootstrap=args.bootstrap,
            seed=args.seed,
            bridge_targets=targets if spec.bridge_weight > 0 else None,
            role_moments=role_moments,
        )
        if spec.r_symmetric > 0 or spec.role_deviation
        else fit
    )

    diagnostics = fit.diagnostics()
    diagnostics["role_blind_twin"] = role_blind_twin.diagnostics()
    diagnostics["no_repair_twin"] = no_repair_twin.diagnostics()
    diagnostics["standardization_moments"] = {
        stat: {"mean": value[0], "sd": value[1]}
        for stat, value in moment_constants.items()
    }
    diagnostics["bridge_target_rejections"] = rejections
    diagnostics["role_cells"] = role_cell_report(
        role_moments,
        fit.loadings,
        {name: pair for name, kind, pair in DEPENDENCE_BUCKETS if kind == "same_team"},
    )

    loadings_payload = fit.loadings.to_payload()
    factor_spec = {
        "dependence_model_version": DEPENDENCE_MODEL_VERSION,
        "stats": list(STATS),
        "factor_families": list(FACTOR_FAMILIES),
        "k_game": int(fit.loadings.k_game),
        "r_contrast": int(fit.loadings.r_contrast),
        "r_symmetric": int(fit.loadings.r_symmetric),
        "symmetric_mode": spec.symmetric_mode,
        "role_deviation": bool(spec.role_deviation),
        "shrink_z": float(spec.shrink_z),
        "same_shrinkage": spec.same_shrinkage,
        "cross_shrinkage": spec.cross_shrinkage,
        "bridge_weight": float(spec.bridge_weight),
        "temporal_treatment": spec.temporal_treatment,
        "role_column": spec.role_column,
        "training_seasons": training_seasons,
        "validation_seasons": validation_seasons,
        "standardization_moments": diagnostics["standardization_moments"],
        "loadings": loadings_payload,
        # Diagnostic only: the role-blind twin is the comparison the role
        # layer is measured against on the holdout. Nothing simulates it.
        "role_blind_twin_loadings": role_blind_twin.loadings.to_payload(),
        # Diagnostic only: the no-symmetric-subspace twin the cross-team
        # isolation identity is measured against. Nothing simulates it either.
        "no_repair_twin_loadings": no_repair_twin.loadings.to_payload(),
        "v2_candidate": spec.payload(),
        "parent_shadow_v1_sha": PARENT_V1_SHA,
        "parent_bucket_repair_sha": PARENT_REPAIR_SHA,
        "identification_note": (
            "A loading column appended to both the game-level and the "
            "team-contrast block enters S = A + B - Q twice and cancels from "
            "X = A - B exactly, so the same-team repair is algebraically "
            "unable to move the opponent buckets. See covariance.py."
        ),
        "role_note": (
            "The role layer is an additive deviation U_i = U + h(role_i) W "
            "inside that subspace, with h weighted-centred over same-team "
            "ordered pairs so the pooled block is unmoved. No parameter is "
            "indexed by a player or a player pair, so an unseen role falls "
            "back to h = 0, which is the pooled loading."
        ),
        "bridge_note": (
            "The same-team target is blended with the latent correlation the "
            "training count-space moment implies through the discrete "
            "Gaussian-copula bridge, inverted per stat pair. The bridge is "
            "deterministic: no simulation enters the target."
        ),
        "within_player_block_source": (
            "incumbent nba_prop_quant.copula.GaussianCopula; pinned, not refitted"
        ),
    }
    factor_spec["spec_hash"] = sha256_canonical(factor_spec)

    spec_path = write_json(factor_spec, artifact_dir / FACTOR_SPEC_NAME)
    diagnostics_path = write_json(
        diagnostics, artifact_dir / COVARIANCE_DIAGNOSTICS_NAME
    )

    contrast = fit.loadings.contrast_matrix
    competition = fit.loadings.competition
    symmetric = fit.loadings.symmetric
    deviation = fit.loadings.role_deviation
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
            **(
                {
                    f"symmetric_{index + 1}": symmetric[:, index]
                    for index in range(symmetric.shape[1])
                }
                if symmetric is not None
                else {}
            ),
            **(
                {
                    f"role_deviation_{index + 1}": deviation[:, index]
                    for index in range(deviation.shape[1])
                }
                if deviation is not None
                else {}
            ),
        }
    )
    loadings_path = artifact_dir / FACTOR_LOADINGS_NAME
    loadings_frame.to_parquet(loadings_path, index=False)
    console.print(loadings_frame.round(4).to_string(index=False))

    # Inner metrics come from the scored row whose spec *is* the frozen spec.
    # Normally that is the winner's own row. When the axis resolution moved the
    # same-team selection, the frozen spec coincides with a different row the
    # screen also scored, and reading the winner's row would report the fold
    # metrics of a fit the freeze did not take.
    scored_as = spec.name
    if resolution is not None and resolution["selection_changed"]:
        equivalent = [
            name
            for name in resolution["resolved_winner_already_scored_as"]
            if name in screening["candidates"]
        ]
        if not equivalent:
            raise SystemExit(
                "the resolved candidate's spec matches no scored screening row, "
                "so its inner fold metrics are not known without rerunning 03"
            )
        scored_as = equivalent[0]
        console.print(
            f"  inner fold metrics read from {sorted(equivalent)}, whose spec "
            "is the frozen spec field for field"
        )
    selected = screening["candidates"][scored_as]
    control = screening["candidates"]["control_repair"]
    calibration = temporal["uncertainty_calibration"]
    freeze = {
        "frozen": True,
        "candidate_name": spec.name,
        "hyperparameters": spec.payload(),
        "parameter_counts": spec.parameter_count(len(STATS)),
        "structural_components": {
            "symmetric_same_team_subspace_rank": int(fit.loadings.r_symmetric),
            "role_deviation_carried": bool(fit.loadings.role_deviation is not None),
            "role_scores": {
                str(key): float(value) for key, value in fit.role_scores.items()
            },
            "bridge_weight": float(spec.bridge_weight),
            "temporal_treatment": spec.temporal_treatment,
            "temporal_predictive_sd_inflation": float(
                calibration["pooled"]["inflation"]
            ),
        },
        "axis_selections": {
            **screening["axis_selections"],
            **(
                {
                    "same_team": resolution["resolved_selection"],
                    "r_symmetric": resolution["resolved_r_symmetric"],
                    "symmetric_mode": resolution["resolved_symmetric_mode"],
                }
                if resolution is not None
                else {}
            ),
        },
        "axis_selections_as_the_screen_recorded_them": screening["axis_selections"],
        "axis_indifference_resolution": resolution,
        "inner_folds": screening["inner_folds"],
        "inner_target_verdicts": screening["inner_target_verdicts"][spec.name],
        "inner_metrics_scored_as": scored_as,
        "inner_metrics": {
            key: selected[key]
            for key in (
                "mean_global_latent_rmse",
                "mean_global_count_rmse",
                "mean_cross_latent_rmse",
                "mean_same_latent_rmse",
                "mean_target_abs_error",
                "mean_primary_count_abs_error",
                "mean_role_conditioned_rmse",
                "mean_pooled_only_role_rmse",
                "mean_role_improvement_fraction",
                "max_cross_team_unchanged_deviation",
                "min_eigenvalue",
                "max_same_player_block_deviation",
            )
        },
        "inner_metrics_control_repair": {
            key: control[key]
            for key in (
                "mean_global_latent_rmse",
                "mean_global_count_rmse",
                "mean_cross_latent_rmse",
                "mean_same_latent_rmse",
                "mean_target_abs_error",
                "mean_primary_count_abs_error",
            )
        },
        "selection_rationale": (
            "Selected mechanically by the pre-registered rule in "
            "03_inner_screening.py: each structural axis screened on its own "
            "over two strictly temporal pre-2024 folds, then one "
            "pre-registered combination set scored on inner targets A-I. No "
            "axis was searched jointly with another and no grid was wider "
            "than five points."
            + (
                " The same-team axis's selection was then resolved by "
                "03b_resolve_axis_indifference.py, which re-applied the "
                "indifference rule to the fold metrics the screening run had "
                "already written down. It refitted nothing and could only "
                "substitute a point from that axis's own grid."
                if resolution is not None and resolution["selection_changed"]
                else ""
            )
        ),
        "fitted_on": {
            "seasons": training_seasons,
            "games": int(fit.moments.games),
            "same_team_ordered_pairs": float(fit.moments.same_team_pairs),
            "cross_team_ordered_pairs": float(fit.moments.cross_team_pairs),
        },
        "structural_identities_at_the_frozen_fit": {
            "cross_team_unchanged_deviation": fit.cross_team_unchanged_deviation(),
            "role_quadratic_share": fit.loadings.role_quadratic_share(),
            "pooled_same_team_vs_target_max_abs": float(
                np.max(
                    np.abs(
                        fit.loadings.pooled_same_team_correlation() - fit.same_target
                    )
                )
            ),
        },
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "parent_shadow_v1_sha": PARENT_V1_SHA,
        "parent_bucket_repair_sha": PARENT_REPAIR_SHA,
        "source_production_sha": git_sha("origin/production/wizardofodds-integration"),
        "branch": BRANCH,
        "seed": int(args.seed),
        "bootstrap_draws": int(args.bootstrap),
        "code_sha": git_sha(),
        "factor_spec_hash": factor_spec["spec_hash"],
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "production_integration_started": False,
        "note": (
            "Frozen before the 2024-2025 holdout was run. Retuning against "
            "holdout results after this point would invalidate the evaluation."
        ),
    }
    freeze_path = write_json(freeze, artifact_dir / "shadow_v2_candidate.json")

    manifest = ArtifactManifest(
        artifact_name="game_latent_state_shadow_v2_factor_model",
        source_production_sha=git_sha("origin/production/wizardofodds-integration"),
        source_production_ref="origin/production/wizardofodds-integration",
        code_sha=git_sha(),
        branch=BRANCH,
        seed=int(args.seed),
        training_cutoff=f"season<{min(HOLDOUT_SEASONS)}",
        seasons_used=training_seasons,
        training_seasons=training_seasons,
        validation_seasons=validation_seasons,
        input_fingerprints={
            RESIDUAL_DATASET_NAME: sha256_file(residual_path),
            "inner_screening.json": sha256_file(screening_path),
            **(
                {"axis_indifference_resolution.json": sha256_file(resolution_path)}
                if resolution is not None
                else {}
            ),
            "temporal_diagnostic.json": sha256_file(temporal_path),
            "count_bridge.json": sha256_file(bridge_path),
        },
        parameters={
            "v2_spec": spec.payload(),
            "parameter_counts": spec.parameter_count(len(STATS)),
            "bootstrap_draws": int(args.bootstrap),
            "spec_hash": factor_spec["spec_hash"],
            "parent_shadow_v1_sha": PARENT_V1_SHA,
            "parent_bucket_repair_sha": PARENT_REPAIR_SHA,
        },
        notes=[
            "Candidate selected on pre-2024 inner folds only.",
            "Loadings estimated from cross-player pairs only; the incumbent "
            "same-player block is never re-estimated.",
            "The symmetric subspace cancels from the cross-team block "
            "algebraically, so no same-team hyperparameter can reach it.",
        ],
    )
    finalize_manifest(
        manifest,
        artifact_dir,
        outputs={
            FACTOR_SPEC_NAME: spec_path,
            FACTOR_LOADINGS_NAME: loadings_path,
            COVARIANCE_DIAGNOSTICS_NAME: diagnostics_path,
            "shadow_v2_candidate.json": freeze_path,
        },
        manifest_name="manifest.json",
        checksum_name="SHA256SUMS.factors.txt",
    )

    index = {stat: position for position, stat in enumerate(STATS)}
    pooled = fit.loadings.pooled_same_team_correlation()
    cross = fit.loadings.cross_team_correlation()
    console.rule("Frozen")
    console.print("SHADOW_V2_CANDIDATE_FROZEN=YES")
    console.print(f"SHADOW_V2_CANDIDATE_NAME={spec.name}")
    console.print(f"SHADOW_V2_CANDIDATE_PARAMETERS={json.dumps(spec.payload())}")
    console.print("INNER_SCREENING_COMPLETE=YES")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")
    console.print("PRODUCTION_INTEGRATION_STARTED=NO")
    console.print(
        f"\nk_game={fit.loadings.k_game}  "
        f"r_contrast={fit.loadings.r_contrast}  "
        f"r_symmetric={fit.loadings.r_symmetric}  "
        f"r_competition={fit.loadings.r_competition}"
    )
    console.print(
        f"implied teammate_ast_ast = {pooled[index['ast'], index['ast']]:+.6f}   "
        f"teammate_reb_reb = {pooled[index['reb'], index['reb']]:+.6f}   "
        f"passer_ast_teammate_pts = {pooled[index['pts'], index['ast']]:+.6f}"
    )
    console.print(
        f"opponent_ast_ast = {cross[index['ast'], index['ast']]:+.6f}   "
        f"cross-team identity deviation = "
        f"{fit.cross_team_unchanged_deviation():.3e}"
    )
    console.print(
        "min eig of implied same-team block = "
        f"{float(np.min(np.linalg.eigvalsh(0.5 * (pooled + pooled.T)))):+.6f}"
    )


if __name__ == "__main__":
    main()
