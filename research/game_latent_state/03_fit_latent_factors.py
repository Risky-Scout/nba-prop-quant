"""Fit the hierarchical latent-state factor loadings. Chronological split only.

SHADOW / RESEARCH ONLY.

Reads ``oof_gaussian_residuals.parquet``, splits it chronologically, estimates
the shared factor loadings on the training seasons *only*, and writes the
factor spec, loadings, covariance diagnostics and a hashed manifest.

The validation seasons are never used to fit anything here, including the
standardization constants.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.adaptive_training import CORE_SEED
from nba_prop_quant.research.game_latent_state import DEPENDENCE_MODEL_VERSION
from nba_prop_quant.research.game_latent_state.artifacts import (
    ArtifactManifest,
    finalize_manifest,
    git_sha,
    sha256_canonical,
)
from nba_prop_quant.research.game_latent_state.covariance import min_eigenvalue
from nba_prop_quant.research.game_latent_state.factors import (
    DEFAULT_K_GAME,
    DEFAULT_SHRINK_Z,
    fit_shared_factors,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.paths import (
    COVARIANCE_DIAGNOSTICS_NAME,
    DEFAULT_ARTIFACT_ROOT,
    FACTOR_LOADINGS_NAME,
    FACTOR_SPEC_NAME,
    MANIFEST_NAME,
    RESIDUAL_DATASET_NAME,
)
from nba_prop_quant.research.game_latent_state.simulator import SUPPORTED_STATS

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Named factor families the fit reports. The game-level block is rank
# `k_game`; see covariance.py for why the own-team / opponent-team loadings
# are identified only through their difference.
FACTOR_FAMILIES = (
    "game_pace_volume",
    "game_rebound_environment",
    "team_contrast_own_minus_opponent",
    "within_team_zero_sum_competition",
    "player_within_block_incumbent",
    "idiosyncratic_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--k-game", type=int, default=DEFAULT_K_GAME)
    parser.add_argument("--shrink-z", type=float, default=DEFAULT_SHRINK_Z)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=CORE_SEED)
    parser.add_argument(
        "--holdout-seasons",
        type=int,
        default=2,
        help="Number of most recent seasons reserved for validation.",
    )
    parser.add_argument(
        "--role-column",
        type=str,
        default="role_bucket",
        help="Set to 'none' to disable role modulation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_root)
    residual_path = artifact_dir / RESIDUAL_DATASET_NAME
    if not residual_path.exists():
        raise SystemExit(f"missing residual dataset: {residual_path}")

    residuals = pd.read_parquet(residual_path)
    residuals["season"] = residuals["season"].astype(int)
    # Plain ints, not numpy scalars: a list of numpy scalars renders as a rich
    # markup tag and is swallowed by the console.
    seasons = [int(season) for season in sorted(residuals["season"].unique())]
    if len(seasons) <= args.holdout_seasons:
        raise SystemExit("not enough residual seasons for a chronological split")

    validation_seasons = seasons[-int(args.holdout_seasons) :]
    training_seasons = seasons[: -int(args.holdout_seasons)]

    console.rule("Chronological split")
    console.print(f"training seasons:   {training_seasons}")
    console.print(f"validation seasons: {validation_seasons} (never fitted)")

    training = residuals.loc[residuals["season"].isin(training_seasons)].copy()
    standardized, moments = standardize_residuals(training, SUPPORTED_STATS)

    role_column = None if args.role_column.lower() == "none" else args.role_column

    console.rule("Shared factor estimation (cross-player pairs only)")
    fit = fit_shared_factors(
        standardized,
        SUPPORTED_STATS,
        k_game=int(args.k_game),
        shrink_z=float(args.shrink_z),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        role_column=role_column,
    )

    diagnostics = fit.diagnostics()
    same = fit.loadings.same_team_correlation()
    cross = fit.loadings.cross_team_correlation()
    diagnostics["game_gram_min_eigenvalue"] = min_eigenvalue(0.5 * (same + cross))
    diagnostics["contrast_gram_min_eigenvalue"] = min_eigenvalue(0.5 * (same - cross))
    diagnostics["same_team_min_eigenvalue"] = min_eigenvalue(same)
    diagnostics["standardization_moments"] = {
        stat: {"mean": value[0], "sd": value[1]} for stat, value in moments.items()
    }
    diagnostics["rank_selection"] = _rank_sweep(fit)

    console.rule("Observed cross-player correlation (training)")
    _print_matrix("same team", fit.moments.same_team)
    _print_matrix("cross team", fit.moments.cross_team)
    console.print(
        f"game-level Gram eigenvalues: "
        f"{np.round(fit.game_gram_eigenvalues, 5).tolist()}"
    )
    console.print(
        f"team-contrast Gram eigenvalues: "
        f"{np.round(fit.contrast_gram_eigenvalues, 5).tolist()}"
    )

    spec = {
        "dependence_model_version": DEPENDENCE_MODEL_VERSION,
        "stats": list(SUPPORTED_STATS),
        "factor_families": list(FACTOR_FAMILIES),
        "k_game": int(args.k_game),
        "shrink_z": float(args.shrink_z),
        "role_column": role_column,
        "training_seasons": [int(season) for season in training_seasons],
        "validation_seasons": [int(season) for season in validation_seasons],
        "standardization_moments": diagnostics["standardization_moments"],
        "loadings": fit.loadings.to_payload(),
        "identification_note": (
            "Only the antisymmetric part of the own-team/opponent-team "
            "loadings is separately identified; the symmetric part is "
            "absorbed into the game-level factors. See covariance.py."
        ),
        "within_player_block_source": (
            "incumbent nba_prop_quant.copula.GaussianCopula; pinned, not refitted"
        ),
        "competition_rank_note": (
            "The competition Gram is Q = (A + B) - S by construction, so it is "
            "carried at whatever rank represents that difference exactly; "
            "truncating it would stop the model from reproducing the observed "
            "same-team block. Its eigenvalue spectrum reports the effective "
            "rank. The family is activated only when the bias-corrected "
            "bootstrap bound on min eig(S_hat) is negative."
        ),
    }
    spec["spec_hash"] = sha256_canonical(spec)

    spec_path = _write_json(spec, artifact_dir / FACTOR_SPEC_NAME)
    diagnostics_path = _write_json(
        diagnostics, artifact_dir / COVARIANCE_DIAGNOSTICS_NAME
    )

    competition = fit.loadings.competition
    loadings_frame = pd.DataFrame(
        {
            "stat": list(SUPPORTED_STATS),
            **{
                f"game_factor_{index + 1}": fit.loadings.game[:, index]
                for index in range(fit.loadings.k_game)
            },
            "team_contrast": fit.loadings.team_contrast,
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

    manifest = ArtifactManifest(
        artifact_name="game_latent_state_factor_model",
        source_production_sha=git_sha(
            PROJECT_ROOT, "origin/production/wizardofodds-integration"
        ),
        source_production_ref="origin/production/wizardofodds-integration",
        code_sha=git_sha(PROJECT_ROOT),
        branch="research/nba-game-latent-state-shadow-v1",
        seed=int(args.seed),
        training_cutoff=str(
            pd.to_datetime(training["game_date"]).max().date()
        ),
        seasons_used=[int(season) for season in seasons],
        training_seasons=[int(season) for season in training_seasons],
        validation_seasons=[int(season) for season in validation_seasons],
        input_fingerprints={RESIDUAL_DATASET_NAME: _sha(residual_path)},
        parameters={
            "k_game": int(args.k_game),
            "shrink_z": float(args.shrink_z),
            "bootstrap_draws": int(args.bootstrap),
            "role_column": role_column,
            "stats": list(SUPPORTED_STATS),
            "factor_families": list(FACTOR_FAMILIES),
            "spec_hash": spec["spec_hash"],
        },
        notes=[
            (
                "Loadings estimated from cross-player pairs only; the "
                "incumbent same-player block is never re-estimated."
            ),
            (
                "Validation seasons were excluded from every fitted "
                "quantity, including the standardization constants."
            ),
        ],
    )
    finalize_manifest(
        manifest,
        artifact_dir,
        outputs={
            FACTOR_SPEC_NAME: spec_path,
            FACTOR_LOADINGS_NAME: loadings_path,
            COVARIANCE_DIAGNOSTICS_NAME: diagnostics_path,
        },
        manifest_name=MANIFEST_NAME,
        checksum_name="SHA256SUMS.factors.txt",
    )
    console.rule("Factor model written")


def _rank_sweep(fit) -> dict[str, object]:
    """Report how much of the game-level Gram each rank explains."""
    eigenvalues = np.clip(fit.game_gram_eigenvalues, 0.0, None)
    total = float(np.sum(eigenvalues))
    cumulative = (
        (np.cumsum(eigenvalues) / total).tolist() if total > 0 else []
    )
    return {
        "game_gram_eigenvalues": eigenvalues.tolist(),
        "cumulative_explained": cumulative,
        "selected_k_game": int(fit.k_game),
    }


def _print_matrix(label: str, matrix: np.ndarray) -> None:
    frame = pd.DataFrame(matrix, index=list(SUPPORTED_STATS), columns=list(SUPPORTED_STATS))
    console.print(f"\n{label}:")
    console.print(frame.round(4).to_string())


def _write_json(payload: object, path: Path) -> Path:
    from nba_prop_quant.research.game_latent_state.artifacts import write_json

    return write_json(payload, path)


def _sha(path: Path) -> str:
    from nba_prop_quant.research.game_latent_state.artifacts import sha256_file

    return sha256_file(path)


if __name__ == "__main__":
    main()
