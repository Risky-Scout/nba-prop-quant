"""Research-scoped filesystem layout for the game-level latent-state shadow.

Every path here is separate from the production data root, the production
model directory and the Step 3B rolling state tree. The shadow study reads
production *code* and writes only research artifacts.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_RESEARCH_DATA_ROOT = Path("data/research/game_latent_state")

DEFAULT_ARTIFACT_ROOT = Path("research/game_latent_state")

RESIDUAL_DATASET_NAME = "oof_gaussian_residuals.parquet"

FACTOR_SPEC_NAME = "factor_spec.json"

FACTOR_LOADINGS_NAME = "factor_loadings.parquet"

COVARIANCE_DIAGNOSTICS_NAME = "covariance_diagnostics.json"

VALIDATION_REPORT_NAME = "validation_report.json"

MANIFEST_NAME = "manifest.json"

CHECKSUM_NAME = "SHA256SUMS.txt"


def research_raw_dir(data_root: Path = DEFAULT_RESEARCH_DATA_ROOT) -> Path:
    return Path(data_root) / "raw"


def research_processed_dir(data_root: Path = DEFAULT_RESEARCH_DATA_ROOT) -> Path:
    return Path(data_root) / "processed"


def research_model_dir(data_root: Path = DEFAULT_RESEARCH_DATA_ROOT) -> Path:
    return Path(data_root) / "models"
