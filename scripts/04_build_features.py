from __future__ import annotations

import joblib
import pandas as pd
from rich.console import Console

from nba_prop_quant.experience import (
    add_prior_season_experience_curves,
    fit_production_experience_curves,
)
from nba_prop_quant.features import TARGETS, add_dynamic_priors, feature_columns
from nba_prop_quant.pipeline import (
    base_matrix_path,
    dynamic_params_path,
    feature_matrix_path,
    load_json,
)
from nba_prop_quant.settings import get_settings
from nba_prop_quant.storage import write_parquet_atomic

console = Console()


def main() -> None:
    settings = get_settings()
    base = pd.read_parquet(base_matrix_path(settings))
    params = load_json(dynamic_params_path(settings))

    features = add_dynamic_priors(base, params)
    features = add_prior_season_experience_curves(
        features,
        targets=list(TARGETS),
        min_history_rows=500,
    )

    production_curves = fit_production_experience_curves(
        features,
        targets=list(TARGETS),
    )
    joblib.dump(
        production_curves,
        settings.nba_prop_model_dir / "experience_curves.joblib",
    )

    write_parquet_atomic(features, feature_matrix_path(settings))
    console.print(
        f"[green]Wrote features[/green]: {len(features):,} rows; "
        f"{len(feature_columns(features))} permitted pregame features"
    )
    console.print("[green]Saved target-specific production experience curves[/green]")


if __name__ == "__main__":
    main()
