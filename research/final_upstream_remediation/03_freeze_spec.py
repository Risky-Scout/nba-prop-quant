#!/usr/bin/env python
"""Freeze the six remediation decisions and emit the spec the validator reads.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Nothing is selected here. The five inner items come from
``inner_selection.json`` and the dependence temperature comes from
``dependence_temperature.json``, both written by drivers that saw only
pre-2024 seasons. This refits the chosen model on the same training seasons
the accepted V1 lineage used and writes a ``factor_spec.json`` in the format
the *unmodified* V1 validation driver consumes, which is what lets the
confirmatory run go through the same framework as the control instead of a
re-implementation.

Once ``frozen_spec.json`` exists the candidate is closed. Any retuning after
this point would be retuning against holdout results.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nba_prop_quant.research.game_latent_state import DEPENDENCE_MODEL_VERSION
from nba_prop_quant.research.game_latent_state.artifacts import (
    ArtifactManifest,
    finalize_manifest,
    git_sha,
    sha256_canonical,
    sha256_file,
    write_json,
)
from nba_prop_quant.research.game_latent_state.paths import (
    COVARIANCE_DIAGNOSTICS_NAME,
    FACTOR_LOADINGS_NAME,
    FACTOR_SPEC_NAME,
    RESIDUAL_DATASET_NAME,
    research_processed_dir,
)
from nba_prop_quant.research.game_latent_state.simulator import SUPPORTED_STATS
from upstream_spec import HOLDOUT_SEASONS, UpstreamChoices, upstream_fit

console = Console()

STATS = SUPPORTED_STATS
BRANCH = "research/nba-final-upstream-remediation"

#: The accepted bucket repair this candidate must be compared against. Its
#: validation report is the control for every gate.
CONTROL_ARTIFACT_DIR = PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"
V1_ARTIFACT_DIR = PROJECT_ROOT / "research" / "game_latent_state"

FACTOR_FAMILIES: tuple[str, ...] = (
    "game_pace_volume",
    "game_rebound_environment",
    "team_contrast_own_minus_opponent",
    "within_team_zero_sum_competition",
    "player_within_block_incumbent",
    "idiosyncratic_residual",
)

#: The six items of the brief, in its order, mapped onto the artifacts that
#: decided them. The report and the gate driver both read this.
ITEMS: tuple[tuple[str, str, str], ...] = (
    ("1", "temporal_ast_ast_season_effects", "item_1_temporal"),
    ("2", "multiplicative_role_scale", "item_2_role_scale"),
    ("3", "cross_team_shrinkage_prior", "item_3_cross_team"),
    ("4", "latent_to_count_transmission_bridge", "item_4_transmission"),
    ("5", "cross_player_dependence_temperature", None),
    ("6", "predictive_uncertainty_calibration", "item_6_uncertainty"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research/final_upstream_remediation",
    )
    parser.add_argument(
        "--residuals",
        type=Path,
        default=V1_ARTIFACT_DIR / RESIDUAL_DATASET_NAME,
    )
    parser.add_argument(
        "--count-residuals",
        type=Path,
        default=research_processed_dir(PROJECT_ROOT / "data/research/game_latent_state")
        / "v2_bridge_count_residuals.parquet",
    )
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--bridge-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def refuse_if_holdout_was_used(name: str, payload: dict) -> None:
    if payload.get("holdout_used_for_selection", True):
        raise SystemExit(f"{name} does not certify that the holdout was unused")
    excluded = [int(season) for season in payload.get("holdout_seasons_excluded", [])]
    if sorted(excluded) != sorted(HOLDOUT_SEASONS):
        raise SystemExit(
            f"{name} excluded {excluded}, not {list(HOLDOUT_SEASONS)}"
        )


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    console.rule("Final upstream remediation: freeze")

    inner = load(artifact_dir / "inner_selection.json")
    temperature = load(artifact_dir / "dependence_temperature.json")
    refuse_if_holdout_was_used("inner_selection.json", inner)
    refuse_if_holdout_was_used("dependence_temperature.json", temperature)

    choices = UpstreamChoices.from_artifacts(
        inner, temperature=float(temperature["selected_temperature"])
    )
    spec_obj = choices.spec()

    table = Table(title="the six decisions, all taken on pre-2024 folds")
    table.add_column("item")
    table.add_column("layer")
    table.add_column("selected")
    table.add_column("decided by")
    payload = choices.payload()
    keys = {
        "1": "temporal_treatment",
        "2": "role_scale_mode",
        "3": "cross_team_prior",
        "4": "transmission_weight_cap",
        "5": "dependence_temperature",
        "6": "uncertainty_model",
    }
    for number, layer, source in ITEMS:
        table.add_row(
            number,
            layer,
            str(payload[keys[number]]),
            "01_inner_selection.py" if source else "02_dependence_temperature.py",
        )
    console.print(table)

    # The accepted V1 spec fixes the training/validation split; reusing it
    # verbatim is what keeps the confirmatory comparison paired.
    v1_spec = load(V1_ARTIFACT_DIR / FACTOR_SPEC_NAME)
    training_seasons = [int(season) for season in v1_spec["training_seasons"]]
    validation_seasons = [int(season) for season in v1_spec["validation_seasons"]]
    if sorted(validation_seasons) != sorted(HOLDOUT_SEASONS):
        raise SystemExit(f"unexpected holdout in the V1 spec: {validation_seasons}")
    if set(training_seasons) & set(HOLDOUT_SEASONS):
        raise SystemExit("training seasons overlap the holdout")

    control_spec = load(CONTROL_ARTIFACT_DIR / FACTOR_SPEC_NAME)
    if [int(s) for s in control_spec["training_seasons"]] != training_seasons:
        raise SystemExit("the control was fitted on different training seasons")

    residuals = pd.read_parquet(args.residuals)
    residuals["season"] = residuals["season"].astype(int)
    training = residuals.loc[residuals["season"].isin(training_seasons)]
    if training.empty:
        raise SystemExit("no training rows")
    if set(training["season"].unique()) & set(HOLDOUT_SEASONS):
        raise SystemExit("holdout seasons reached the fit")
    console.print(f"training seasons: {training_seasons} ({len(training):,} rows)")
    console.print(f"holdout seasons : {validation_seasons} (never fitted)")

    count_frame = None
    if choices.transmission_cap > 0.0:
        count_frame = pd.read_parquet(args.count_residuals)
        count_frame["season"] = count_frame["season"].astype(int)
        count_frame = count_frame.loc[count_frame["season"].isin(training_seasons)]

    fit = upstream_fit(
        training,
        STATS,
        choices,
        count_frame=count_frame,
        bootstrap=args.bootstrap,
        bridge_bootstrap=args.bridge_bootstrap,
        seed=args.seed,
    )
    loadings = fit.loadings

    diagnostics: dict[str, object] = dict(fit.diagnostics)
    diagnostics["remediation_spec"] = spec_obj.payload()
    diagnostics["r_contrast"] = int(loadings.r_contrast)

    spec = {
        "dependence_model_version": DEPENDENCE_MODEL_VERSION,
        "stats": list(STATS),
        "factor_families": list(FACTOR_FAMILIES),
        "k_game": int(spec_obj.k_game),
        "r_contrast": int(loadings.r_contrast),
        "eb_family": spec_obj.eb_family,
        "role_column": spec_obj.role_column,
        "training_seasons": training_seasons,
        "validation_seasons": validation_seasons,
        "standardization_moments": diagnostics["standardization_moments"],
        "loadings": loadings.to_payload(),
        "remediation_spec": spec_obj.payload(),
        "upstream_choices": choices.payload(),
        "control_factor_spec_hash": control_spec["spec_hash"],
        "identification_note": (
            "Only the antisymmetric part of the own-team/opponent-team "
            "loadings is separately identified; the symmetric part is absorbed "
            "into the game-level factors. See covariance.py."
        ),
        "within_player_block_source": (
            "incumbent nba_prop_quant.copula.GaussianCopula; pinned, not refitted"
        ),
        "role_scale_identification": (
            "multiplicative, one scale per role bucket, constrained to a "
            "player-share-weighted mean of exactly one, so the pooled "
            "same-team block is the same object the unroled fit produced"
        ),
        "dependence_temperature_note": (
            "lambda multiplies every shared cross-player loading by its square "
            "root, so every cross-player block is multiplied by exactly lambda "
            "while each player's own block stays pinned at the incumbent's."
        ),
    }
    spec["spec_hash"] = sha256_canonical(spec)

    spec_path = write_json(spec, artifact_dir / FACTOR_SPEC_NAME)
    diagnostics_path = write_json(
        diagnostics, artifact_dir / COVARIANCE_DIAGNOSTICS_NAME
    )

    contrast = loadings.contrast_matrix
    competition = loadings.competition
    loadings_frame = pd.DataFrame(
        {
            "stat": list(STATS),
            **{
                f"game_factor_{index + 1}": loadings.game[:, index]
                for index in range(loadings.game.shape[1])
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

    # The validation driver reads the residual dataset from its artifact root.
    # Copying rather than re-deriving is what makes the comparison paired: the
    # candidate and the control are scored on the same rows, byte for byte.
    local_residuals = artifact_dir / RESIDUAL_DATASET_NAME
    if not local_residuals.exists():
        shutil.copy2(args.residuals, local_residuals)
    if sha256_file(local_residuals) != sha256_file(args.residuals):
        raise SystemExit("the copied residual dataset does not match its source")

    role_scale = dict(loadings.role_scale or {})
    pooled_neutrality = None
    role_diagnostics = diagnostics.get("role_scale")
    if isinstance(role_diagnostics, dict):
        pooled_neutrality = role_diagnostics.get("weighted_mean_scale_minus_one")

    frozen = {
        "title": "Final upstream remediation: frozen spec",
        "frozen": True,
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "candidate_name": spec_obj.name,
        "upstream_choices": choices.payload(),
        "remediation_spec": spec_obj.payload(),
        "parameter_counts": spec_obj.parameter_count(len(STATS)),
        "items": [
            {
                "item": number,
                "layer": layer,
                "selected": payload[keys[number]],
                "decided_by": (
                    "01_inner_selection.py" if source else "02_dependence_temperature.py"
                ),
                "selection_record": (
                    inner[source]["selection_rule"]
                    if source and "selection_rule" in inner[source]
                    else (
                        inner[source].get("reason")
                        if source
                        else temperature["selection"]["reason"]
                    )
                ),
            }
            for number, layer, source in ITEMS
        ],
        "role_scale": role_scale,
        "role_scale_weighted_mean_minus_one": pooled_neutrality,
        "transmission": (
            {
                "cap": choices.transmission_cap,
                "homogeneity_verdict": inner["item_4_transmission"][
                    "homogeneity_verdict"
                ],
                "admissibility": inner["item_4_transmission"]["admissibility"][
                    f"cap_{choices.transmission_cap:.2f}"
                ],
            }
            if choices.transmission_cap > 0.0
            else {"cap": 0.0, "note": "the bridge is off"}
        ),
        "training_seasons": training_seasons,
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "holdout_seasons_excluded": list(HOLDOUT_SEASONS),
        "control": {
            "name": "accepted bucket repair",
            "artifact_dir": str(CONTROL_ARTIFACT_DIR.relative_to(PROJECT_ROOT)),
            "factor_spec_hash": control_spec["spec_hash"],
        },
        "factor_spec_hash": spec["spec_hash"],
        "inner_selection_code_sha": inner.get("code_sha"),
        "dependence_temperature_code_sha": temperature.get("code_sha"),
        "code_sha": git_sha(PROJECT_ROOT),
        "branch": BRANCH,
        "seed": int(args.seed),
        "bootstrap_draws": int(args.bootstrap),
        "bridge_bootstrap_draws": int(args.bridge_bootstrap),
        "note": (
            "Frozen before the 2024-2025 holdout was run. Retuning against "
            "holdout results after this point would invalidate the evaluation."
        ),
    }
    frozen_path = write_json(frozen, artifact_dir / "frozen_spec.json")

    manifest = ArtifactManifest(
        artifact_name="final_upstream_remediation_factor_model",
        source_production_sha=git_sha(
            PROJECT_ROOT, "origin/production/wizardofodds-integration"
        ),
        source_production_ref="origin/production/wizardofodds-integration",
        code_sha=git_sha(PROJECT_ROOT),
        branch=BRANCH,
        seed=int(args.seed),
        training_cutoff=f"season<{min(HOLDOUT_SEASONS)}",
        seasons_used=training_seasons,
        training_seasons=training_seasons,
        validation_seasons=validation_seasons,
        input_fingerprints={RESIDUAL_DATASET_NAME: sha256_file(args.residuals)},
        parameters={
            "upstream_choices": choices.payload(),
            "remediation_spec": spec_obj.payload(),
            "parameter_counts": spec_obj.parameter_count(len(STATS)),
            "bootstrap_draws": int(args.bootstrap),
            "spec_hash": spec["spec_hash"],
        },
        notes=[
            "All six decisions taken on strictly pre-2024 chronological folds.",
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
            "frozen_spec.json": frozen_path,
        },
        manifest_name="manifest.json",
        checksum_name="SHA256SUMS.factors.txt",
    )

    same = loadings.same_team_correlation()
    index = {stat: position for position, stat in enumerate(STATS)}
    console.rule("Frozen")
    console.print("FINAL_UPSTREAM_SPEC_FROZEN=YES")
    for number, _, _ in ITEMS:
        console.print(f"ITEM_{number}={payload[keys[number]]}")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")
    console.print(
        f"\nimplied teammate_ast_ast = {same[index['ast'], index['ast']]:+.6f}   "
        f"teammate_reb_reb = {same[index['reb'], index['reb']]:+.6f}"
    )
    console.print(
        "min eig of implied same-team block = "
        f"{float(np.min(np.linalg.eigvalsh(0.5 * (same + same.T)))):+.6f}"
    )
    console.print(f"\nwrote {frozen_path}")


if __name__ == "__main__":
    main()
