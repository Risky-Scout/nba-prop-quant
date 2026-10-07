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
    SeasonSeries,
    fit_gaussian_random_effects,
    fit_log_shrunk_role_scales,
    fit_pooled,
    fit_remediated_factors,
    fit_student_t_random_effects,
    role_scale_components,
    temper_loadings,
)
from nba_prop_quant.research.game_latent_state.transmission import (
    DEFAULT_BRIDGE_ORDER,
    accumulate_per_game,
    bootstrap_source_pair,
    combine_sources,
    homogeneity_test,
    transmission_coefficient_columns,
)
from nba_prop_quant.research.game_latent_state.validation import DEPENDENCE_BUCKETS

ROLE_COLUMN = "role_bucket"
HOLDOUT_SEASONS = (2024, 2025)

#: The buckets the dependence model is judged on, cross-player only. Item 1's
#: season pooling moves exactly these entries of the two blocks.
CROSS_PLAYER_BUCKETS = tuple(
    (name, kind, pair)
    for name, kind, pair in DEPENDENCE_BUCKETS
    if kind in {"same_team", "cross_team"}
)

#: Item 1's four treatments. ``A0`` is the accepted repair's fixed effect, so
#: selecting it leaves the pooled targets exactly where the control put them.
TEMPORAL_POOLED = "A0_pooled_empirical_bayes"
TEMPORAL_GAUSSIAN = "A1_gaussian_random_effects"
TEMPORAL_STUDENT_T = "A2_robust_student_t"
TEMPORAL_RECENCY = "A3_recency_weighted_robust"


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
    #: Item 1's own hyperparameters, chosen on the same pre-2024 folds. Only
    #: the Student-t treatments read them.
    temporal_nu: float = 5.0
    temporal_half_life: float = 2.0

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
        temporal = inner["item_1_temporal"]
        return cls(
            temporal=str(temporal["selected"]),
            role_scale_mode=str(inner["item_2_role_scale"]["selected"]),
            cross_team_prior=prior,
            cross_team_nu=nu,
            transmission_cap=float(inner["item_4_transmission"]["selected_cap"]),
            uncertainty=str(inner["item_6_uncertainty"]["selected"]),
            dependence_temperature=1.0 if temperature is None else float(temperature),
            temporal_nu=float(temporal["nu_selected"]),
            temporal_half_life=float(temporal["half_life_selected"]),
        )

    def spec(self, name: str = "final_upstream_candidate") -> RemediationSpec:
        return RemediationSpec(
            name=name,
            role_scale_mode=self.role_scale_mode,
            cross_team_prior=self.cross_team_prior,
            cross_team_nu=self.cross_team_nu,
            bridge_mode=BRIDGE_OFF if self.transmission_cap <= 0.0 else BRIDGE_COMBINED,
            bridge_weight_cap=self.transmission_cap,
            dependence_temperature=self.dependence_temperature,
        )

    def payload(self) -> dict[str, object]:
        return {
            "temporal_treatment": self.temporal,
            "temporal_nu": self.temporal_nu,
            "temporal_half_life": self.temporal_half_life,
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


def read_buckets(
    stats: Sequence[str],
    same: np.ndarray,
    cross: np.ndarray,
) -> dict[str, float]:
    """The named cross-player buckets read out of the two blocks."""
    index = {stat: position for position, stat in enumerate(stats)}
    out: dict[str, float] = {}
    for name, kind, (first, second) in CROSS_PLAYER_BUCKETS:
        matrix = same if kind == "same_team" else cross
        out[name] = float(matrix[index[first], index[second]])
    return out


def expanding_season_moments(
    frame: pd.DataFrame,
    stats: Sequence[str],
) -> dict[int, dict[str, tuple[float, float]]]:
    """Standardization constants for each season, from the seasons before it.

    The earliest season has nothing before it, so it is standardized on
    itself. Every later season uses only its own past, which is the same
    no-lookahead rule the fits follow.
    """
    seasons = sorted(int(value) for value in frame["season"].unique())
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for season in seasons:
        earlier = frame.loc[frame["season"] < season]
        if earlier.empty:
            earlier = frame.loc[frame["season"] == season]
        _, moments = standardize_residuals(earlier, stats)
        out[season] = moments
    return out


def season_series(
    frame: pd.DataFrame,
    stats: Sequence[str],
    moments_by_season: dict[int, dict[str, tuple[float, float]]] | None = None,
    bootstrap: int = 400,
    seed: int = 73,
) -> dict[str, SeasonSeries]:
    """Per-season bucket estimates with game-clustered standard errors."""
    stats = tuple(stats)
    if moments_by_season is None:
        moments_by_season = expanding_season_moments(frame, stats)
    seasons = sorted(int(value) for value in frame["season"].unique())
    estimates: dict[str, list[float]] = {
        name: [] for name, _, _ in CROSS_PLAYER_BUCKETS
    }
    errors: dict[str, list[float]] = {name: [] for name, _, _ in CROSS_PLAYER_BUCKETS}

    for season in seasons:
        block = frame.loc[frame["season"] == season]
        standardized, _ = standardize_residuals(
            block, stats, moments=moments_by_season.get(season)
        )
        pooled = pair_moments(
            standardized, stats, bootstrap=bootstrap, seed=seed + season
        )
        values = read_buckets(stats, pooled.same_team, pooled.cross_team)
        standard_errors = read_buckets(
            stats, pooled.same_team_se, pooled.cross_team_se
        )
        for name in estimates:
            estimates[name].append(values[name])
            errors[name].append(standard_errors[name])

    return {
        name: SeasonSeries(
            name=name,
            seasons=tuple(seasons),
            estimates=np.array(estimates[name]),
            standard_errors=np.array(errors[name]),
        )
        for name in estimates
    }


def temporal_fitter(treatment: str, nu: float, half_life: float):
    if treatment == TEMPORAL_POOLED:
        return fit_pooled
    if treatment == TEMPORAL_GAUSSIAN:
        return fit_gaussian_random_effects
    if treatment == TEMPORAL_STUDENT_T:
        return lambda series: fit_student_t_random_effects(series, nu=nu)
    if treatment == TEMPORAL_RECENCY:
        return lambda series: fit_student_t_random_effects(
            series, nu=nu, half_life=half_life
        )
    raise ValueError(f"unknown temporal treatment {treatment!r}")


def temporal_shifts(
    frame: pd.DataFrame,
    stats: Sequence[str],
    treatment: str,
    nu: float,
    half_life: float,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, object]]:
    """How far the chosen season pooling moves each bucket off the control's.

    The layer enters as a *shift*, not as a replacement. The control pools
    rows; the four treatments pool seasons; and the question item 1 asks is
    whether modelling season heterogeneity moves the estimate, not whether
    pooling seasons reproduces pooling rows. Taking the difference between the
    chosen treatment and the fixed effect on the same per-season series answers
    exactly that question and makes ``A0`` an exact no-op, so selecting the
    incumbent treatment leaves the control's targets untouched.

    A shift also composes with the transmission bridge, which is the other
    layer that moves these entries. A replacement would overwrite whatever the
    bridge contributed to the same bucket.
    """
    series = season_series(frame, stats, bootstrap=bootstrap, seed=seed)
    chosen = temporal_fitter(treatment, nu, half_life)
    shifts: dict[str, float] = {}
    detail: dict[str, object] = {}
    for name, bucket in series.items():
        baseline = fit_pooled(bucket)
        fit = chosen(bucket)
        shifts[name] = float(fit.posterior_mean - baseline.posterior_mean)
        detail[name] = {
            "seasons": list(bucket.seasons),
            "per_season_estimates": bucket.estimates.tolist(),
            "per_season_standard_errors": bucket.standard_errors.tolist(),
            "fixed_effect_posterior_mean": float(baseline.posterior_mean),
            "selected_posterior_mean": float(fit.posterior_mean),
            "shift": shifts[name],
            "tau": float(fit.tau),
            "weights": list(fit.weights),
            "q_statistic": float(fit.q_statistic),
            "q_p_value": float(fit.q_p_value),
            "i_squared": float(fit.i_squared),
        }
    return shifts, {
        "treatment": treatment,
        "nu": float(nu),
        "half_life": float(half_life),
        "shift_is_zero_by_construction_under_a0": treatment == TEMPORAL_POOLED,
        "max_abs_shift": max((abs(value) for value in shifts.values()), default=0.0),
        "shifts": shifts,
        "by_bucket": detail,
    }


def apply_bucket_shifts(
    same: np.ndarray,
    cross: np.ndarray,
    shifts: dict[str, float],
    stats: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Add a per-bucket shift to the two pooled blocks, symmetrically.

    ``pair_moments`` symmetrises both blocks, so a shift has to land on both
    entries of an off-diagonal bucket or the block stops being symmetric and
    the PSD construction downstream silently reads the average instead.
    """
    index = {stat: position for position, stat in enumerate(stats)}
    same = np.array(same, dtype=float, copy=True)
    cross = np.array(cross, dtype=float, copy=True)
    for name, kind, (first, second) in CROSS_PLAYER_BUCKETS:
        shift = shifts.get(name)
        if not shift:
            continue
        matrix = same if kind == "same_team" else cross
        row, column = index[first], index[second]
        matrix[row, column] += shift
        if row != column:
            matrix[column, row] += shift
    return same, cross


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
    # Everything is estimated at the full dependence model and tempered once,
    # at the end. Estimating under the temperature instead would feed the
    # shrunken cross-player loadings back into the role-scale projection, so
    # lambda would no longer be the one scalar multiplying the cross-player
    # blocks and nothing else.
    fit_spec = replace(spec, dependence_temperature=1.0)
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

    # Item 1 lands last among the target layers and only on the named buckets,
    # so it moves the bridged estimate rather than discarding it. Under A0 the
    # shift is identically zero and this is a no-op.
    if choices.temporal != TEMPORAL_POOLED:
        shifts, diagnostics["temporal"] = temporal_shifts(
            frame,
            stats,
            treatment=choices.temporal,
            nu=choices.temporal_nu,
            half_life=choices.temporal_half_life,
            bootstrap=bootstrap,
            seed=seed,
        )
        shifted_same, shifted_cross = apply_bucket_shifts(
            moments.same_team, moments.cross_team, shifts, stats
        )
        moments = replace(
            moments, same_team=shifted_same, cross_team=shifted_cross
        )
    else:
        diagnostics["temporal"] = {
            "treatment": choices.temporal,
            "shift_is_zero_by_construction_under_a0": True,
            "max_abs_shift": 0.0,
            "shifts": {name: 0.0 for name, _, _ in CROSS_PLAYER_BUCKETS},
        }

    role_override = None
    if choices.role_scale_mode == ROLE_SCALE_LOG_SHRUNK:
        # The projection reads only the pooled same-team block, which carries
        # no role layer, so the base is fitted without one.
        base = fit_remediated_factors(
            standardized,
            stats,
            spec=replace(
                fit_spec, role_scale_mode=ROLE_SCALE_RATIO, role_column=None
            ),
            moments=moments,
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
        spec=fit_spec,
        moments=moments,
        role_scale_override=role_override,
    )
    diagnostics["standardization_moments"] = {
        stat: {"mean": float(mean), "sd": float(sd)}
        for stat, (mean, sd) in used_moments.items()
    }
    return UpstreamFit(
        loadings=temper_loadings(fit.loadings, choices.dependence_temperature),
        spec=spec,
        choices=choices,
        moments=moments,
        combined_targets=combined,
        diagnostics={**diagnostics, "fit": fit.payload()}
        if hasattr(fit, "payload")
        else diagnostics,
    )
