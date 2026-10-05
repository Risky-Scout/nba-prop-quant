"""Scratch probe: does the reserved symmetric subspace hold its identities?

Not part of the pipeline and not referenced by any artifact. Deleted before
the branch is consolidated.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.v2 import (  # noqa: E402
    REPAIR_CONTROL_SPEC,
    SYMMETRIC_MODE_RESERVED,
    SYMMETRIC_MODE_RESIDUAL,
    V1_BASE_SPEC,
    fit_v2_factors,
    role_pair_moments,
)

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT = (2024, 2025)


def main() -> None:
    residuals = pd.read_parquet(
        PROJECT_ROOT
        / "research"
        / "game_latent_state"
        / "oof_gaussian_residuals.parquet"
    )
    residuals["season"] = residuals["season"].astype(int)
    pre = residuals[~residuals["season"].isin(HOLDOUT)].copy()
    standardized, _ = standardize_residuals(pre, STATS)
    role_moments = role_pair_moments(standardized, STATS, bootstrap=200, seed=73)

    for path_name, path_spec in (
        ("repair k6/r6", REPAIR_CONTROL_SPEC),
        ("v1 k2/r1", replace(V1_BASE_SPEC, same_shrinkage="empirical_bayes")),
    ):
        print(f"\n################ {path_name} ################")
        control = fit_v2_factors(
            standardized,
            STATS,
            spec=replace(path_spec, name="control", r_symmetric=0),
            bootstrap=200,
            seed=73,
            role_moments=role_moments,
        )
        control_cross = control.loadings.cross_team_correlation()
        for mode in (SYMMETRIC_MODE_RESIDUAL, SYMMETRIC_MODE_RESERVED):
            for role in (False, True):
                fit = fit_v2_factors(
                    standardized,
                    STATS,
                    spec=replace(
                        path_spec,
                        name=f"{mode}_role{int(role)}",
                        r_symmetric=6,
                        symmetric_mode=mode,
                        role_deviation=role,
                    ),
                    bootstrap=200,
                    seed=73,
                    role_moments=role_moments,
                )
                loadings = fit.loadings
                symmetric = loadings.symmetric
                pooled = loadings.pooled_same_team_correlation()
                scale = loadings.scale_for_role(None)
                print(
                    f"  mode={mode:<8} role={int(role)}  "
                    f"Ucols={0 if symmetric is None else symmetric.shape[1]}  "
                    f"same max|err| {np.max(np.abs(pooled / scale - fit.same_target)):.3e}  "
                    f"X vs own base {fit.cross_team_unchanged_deviation():.3e}  "
                    f"X vs control {np.max(np.abs(loadings.cross_team_correlation() - control_cross)):.3e}  "
                    f"W={'yes' if loadings.role_deviation is not None else 'no '}  "
                    f"leak {fit.role_diagnostics.get('pooled_leak_absorbed', 0.0):+.4f}  "
                    f"scores {[round(v, 3) for v in sorted(fit.role_scores.values())]}"
                )


if __name__ == "__main__":
    main()
