from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from .model import ModelBundle


def global_permutation_importance(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    y: np.ndarray,
    scoring: str = "neg_mean_absolute_error",
    repeats: int = 5,
) -> pd.DataFrame:
    x = frame.reindex(columns=bundle.feature_names)
    result = permutation_importance(
        bundle.model,
        x,
        y,
        scoring=scoring,
        n_repeats=repeats,
        random_state=73,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": bundle.feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def local_shap_explanation(
    bundle: ModelBundle,
    row: pd.Series,
    top_n: int = 12,
) -> pd.DataFrame:
    x = row.reindex(bundle.feature_names).to_frame().T
    explainer = shap.TreeExplainer(bundle.model)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[0]
    contributions = np.asarray(values).reshape(-1)
    out = pd.DataFrame(
        {
            "feature": bundle.feature_names,
            "value": x.iloc[0].to_numpy(),
            "shap_contribution": contributions,
        }
    )
    out["abs_contribution"] = out["shap_contribution"].abs()
    return out.sort_values("abs_contribution", ascending=False).head(top_n)
