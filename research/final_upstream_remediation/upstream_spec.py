#!/usr/bin/env python
"""One definition of the remediated upstream fit.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The inner selection driver decides five of the six items; the temperature
driver decides the sixth; the freeze driver records all of them and the gate
driver scores them. All four need to build *the same* fit from those choices,
so the construction lives here once rather than four times. A second copy
would be a second model, and the difference between them would show up as a
gate result nobody could attribute.

Nothing here selects anything. It takes decisions already made and applies
them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from nba_prop_quant.research.game_latent_state.factors import (
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.remediation import (
    BRIDGE_COMBINED,
    BRIDGE_OFF,
    CROSS_PRIOR_GAUSSIAN,
    CROSS_PRIOR_STUDENT_T,
    ROLE_SCALE_LOG_SHRUNK,
    ROLE_SCALE_RATIO,
    RemediationSpec,
    fit_log_shrunk_role_scales,
    fit_remediated_factors,
    role_scale_components,
)
from nba_prop_quant.research.game_latent_state.transmission import (
    DEFAULT_BRIDGE_ORDER,
    accumulate_per_game,
    bootstrap_source_pair,
    combine_sources,
    homogeneity_test,
    transmission_coefficient_columns,
)

ROLE_COLUMN = "role_bucket"
HOLDOUT_SEASONS = (2024, 2025)


@dataclass(frozen=True)
class UpstreamChoices:
    """The six decisions, as selected on pre-2024 folds."""

    temporal: str
    role_scale_mode: str
    cross_team_prior: str
    cross_team_nu: float | None
    transmission_cap: float
    uncertainty: str
    dependence_temperature: float

    @classmethod
    def from_artifacts(
        cls,
        inner: dict,
        temperature: float | None = None,
    ) -> UpstreamChoices:
        cross = inner["item_3_cross_team"]["selected"]
        if cross == "gaussian":
            prior, nu = CROSS_PRIOR_GAUSSIAN, None
        else:
            prior, nu = CROSS_PRIOR_STUDENT_T, float(cross.split("nu")[1])
        return cls(
            temporal=str(inner["item_1_temporal"]["selected"]),
            role_scale_mode=str(inner["item_2_role_scale"]["selected"]),
            cross_team_prior=prior,
            cross_team_nu=nu,
            transmission_cap=float(inner["item_4_transmission"]["selected_cap"]),
            uncertainty=str(inner["item_6_uncertainty"]["selected"]),
            dependence_temperature=1.0 if temperature is None else float(temperature),
        )

    def spec(self, name: str = "final_upstream_candidate") -> RemediationSpec:
        return RemediationSpec(
            name=name,
            role_scale_mode=ROLE_SCALE_RATIO,
            cross_team_prior=self.cross_team_prior,
            cross_team_nu=self.cross_team_nu,
            bridge_mode=BRIDGE_OFF if self.transmission_cap <= 0.0 else BRIDGE_COMBINED,
            bridge_weight_cap=self.transmission_cap,
            dependence_temperature=self.dependence_temperature,
        )

    def payload(self) -> dict[str, object]:
        return {
            "temporal_treatment": self.temporal,
            "role_scale_mode": self.role_scale_mode,
            "cross_team_prior": self.cross_team_prior,
            "cross_team_nu": self.cross_team_nu,
            "transmission_weight_cap": self.transmission_cap,
            "uncertainty_model": self.uncertainty,
            "dependence_temperature": self.dependence_temperature,
        }


@dataclass
class UpstreamFit:
    loadings: object
    spec: RemediationSpec
    choices: UpstreamChoices
    moments: object
    combined_targets: tuple[np.ndarray, np.ndarray] | None
    diagnostics: dict[str, object]


def read_choices(
    artifact_root: Path,
    inner_name: str = "inner_selection.json",
    temperature_name: str = "dependence_temperature.json",
) -> UpstreamChoices:
    """The selected choices, with the temperature if it has been decided."""
    inner = json.loads(
        (Path(artifact_root) / inner_name).read_text(encoding="utf-8")
    )
    path = Path(artifact_root) / temperature_name
    temperature = None
    if path.exists():
        temperature = float(
            json.loads(path.read_text(encoding="utf-8"))["selected_temperature"]
        )
    return UpstreamChoices.from_artifacts(inner, temperature)


def transmission_targets(
    standardized: pd.DataFrame,
    count_frame: pd.DataFrame,
    stats: Sequence[str],
    cap: float,
    bridge_bootstrap: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, object]]:
    """Combined raw pooled blocks, in the recorded latent column's units."""
    merged = standardized.merge(
        count_frame,
        on=["game_id", "team_id", "player_id", "season"],
        how="inner",
        suffixes=("", "_count"),
    )
    merged, coefficient_diagnostics = transmission_coefficient_columns(
        merged, stats, order=DEFAULT_BRIDGE_ORDER
    )
    orders = range(1, DEFAULT_BRIDGE_ORDER + 1)
    required = (
        [f"{name}{k}_{stat}" for name in "hg" for k in orders for stat in stats]
        + [f"zs_{stat}" for stat in stats]
        + [f"e_{stat}" for stat in stats]
    )
    usable = merged.dropna(subset=required)
    if len(usable) != len(merged):
        raise SystemExit(
            f"{len(merged) - len(usable)} rows lack a transmission coefficient, "
            "which would make the control and the bridged arm read different rows"
        )

    latent = accumulate_per_game(usable, [f"zs_{stat}" for stat in stats], stats)
    count = accumulate_per_game(usable, [f"e_{stat}" for stat in stats], stats)
    count_gains = [
        accumulate_per_game(usable, [f"h{k}_{stat}" for stat in stats], stats)
        for k in orders
    ]
    latent_gains = [
        accumulate_per_game(usable, [f"g{k}_{stat}" for stat in stats], stats)
        for k in orders
    ]

    out: dict[str, object] = {"coefficients": coefficient_diagnostics}
    blocks: list[np.ndarray] = []
    for same_team in (True, False):
        pair = bootstrap_source_pair(
            latent,
            count,
            latent_gains,
            count_gains,
            draws=bridge_bootstrap,
            seed=seed,
            same_team=same_team,
        )
        test = homogeneity_test(pair)
        block, info = combine_sources(pair, bridge_weight_cap=cap)
        blocks.append(block)
        label = "same_team" if same_team else "cross_team"
        out[label] = {
            "direct": pair.latent.tolist(),
            "bridge_in_latent_units": pair.bridge.tolist(),
            "combined": block.tolist(),
            "bridge_weight": np.asarray(info["bridge_weight"]).tolist(),
            "mean_bridge_weight": info["mean_bridge_weight"],
            "max_bridge_weight": info["max_bridge_weight"],
            "bridge_model_error_variance": info["bridge_model_error_variance"],
            "entries_clipped_by_cap": info["entries_clipped_by_cap"],
            "homogeneity_p_value": test["p_value"],
            "homogeneity_rejected": not test["homogeneous"],
            "max_disagreement_in_sigma": test["max_disagreement_in_sigma"],
        }
    return (blocks[0], blocks[1]), out


def upstream_fit(
    frame: pd.DataFrame,
    stats: Sequence[str],
    choices: UpstreamChoices,
    count_frame: pd.DataFrame | None = None,
    bootstrap: int = 400,
    bridge_bootstrap: int = 2000,
    seed: int = 73,
    moments=None,
    standardization_moments=None,
) -> UpstreamFit:
    """Fit the remediated loadings from decisions already made.

    ``frame`` is a *raw* residual frame; it is standardized here so the
    standardization constants and the fit always come from the same rows.
    """
    stats = tuple(stats)
    standardized, used_moments = standardize_residuals(
        frame, stats, moments=standardization_moments
    )
    if moments is None:
        moments = pair_moments(standardized, stats, bootstrap=bootstrap, seed=seed)

    spec = choices.spec()
    diagnostics: dict[str, object] = {}

    combined = None
    if choices.transmission_cap > 0.0:
        if count_frame is None:
            raise ValueError("the transmission layer needs the count residual frame")
        combined, diagnostics["transmission"] = transmission_targets(
            standardized,
            count_frame,
            stats,
            cap=choices.transmission_cap,
            bridge_bootstrap=bridge_bootstrap,
            seed=seed,
        )
        moments = replace(
            moments,
            same_team=np.asarray(combined[0], dtype=float),
            cross_team=np.asarray(combined[1], dtype=float),
        )

    role_override = None
    if choices.role_scale_mode == ROLE_SCALE_LOG_SHRUNK:
        base = fit_remediated_factors(
            standardized, stats, spec=spec, moments=moments
        ).loadings
        raw, log_se, shares = role_scale_components(
            standardized,
            stats,
            base=base,
            role_column=ROLE_COLUMN,
        )
        shrunk = fit_log_shrunk_role_scales(raw, log_se, shares)
        role_override = dict(shrunk.scales)
        diagnostics["role_scale"] = shrunk.payload()

    fit = fit_remediated_factors(
        standardized,
        stats,
        spec=spec,
        moments=moments,
        role_scale_override=role_override,
    )
    diagnostics["standardization_moments"] = {
        stat: {"mean": float(mean), "sd": float(sd)}
        for stat, (mean, sd) in used_moments.items()
    }
    return UpstreamFit(
        loadings=fit.loadings,
        spec=spec,
        choices=choices,
        moments=moments,
        combined_targets=combined,
        diagnostics={**diagnostics, "fit": fit.payload()}
        if hasattr(fit, "payload")
        else diagnostics,
    )
