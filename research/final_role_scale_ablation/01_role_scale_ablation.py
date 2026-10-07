#!/usr/bin/env python
"""The one role-scale ablation: R0 ratio against R1 log-shrunk. PRE-2024 ONLY.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Two candidates, no third:

    R0  the accepted bucket-repair's multiplicative role scale: the within-role
        projection partially pooled toward one by a declared game count, then
        renormalised to a player-share-weighted mean of one.

    R1  the same multiplicative projection shrunk symmetrically in the log
        ratio by a measured standard error, then renormalised the same way.

Everything else is held fixed. Both arms are fitted by one call to
``fit_remediated_factors`` with ``role_scale_override`` set, on moments that
were computed once per fold, so the only thing that differs between the two
fits is the three-element role-scale vector. The pooled same-team block, the
cross-team block, the competition factors, the standardization constants, the
scored rows and the cell support flags are shared by construction rather than
by inspection.

Seasons 2024 and 2025 are dropped on the first read and never re-admitted;
:func:`assert_holdout_absent` runs on every frame derived here rather than once
on the input. The holdout plays no part in this decision.

THE SELECTION RULE IS FROZEN IN THIS FILE BEFORE ANY NUMBER WAS COMPUTED.
It is stated once, in ``FROZEN_RULE`` below, and applied mechanically by
:func:`apply_frozen_rule`. The commit that introduces this file introduces the
rule; the run that produces the numbers comes afterwards, so the rule cannot
have been fitted to the result it selects.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "final_upstream_remediation"))

from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    git_sha,
    sha256_file,
    write_checksums,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    GameDimension,
    build_game_covariance,
    implied_within_player_correlation,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    fit_role_scales,
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.remediation import (  # noqa: E402
    MIN_CELL_GAMES,
    MIN_CELL_PAIRS,
    ROLE_PAIR_CELLS,
    ROLE_SCALE_LOG_SHRUNK,
    ROLE_SCALE_RATIO,
    RemediationSpec,
    cell_standard_errors,
    fit_log_shrunk_role_scales,
    fit_remediated_factors,
    forward_fold_seasons,
    paired_difference_se,
    per_game_role_cells,
    role_scale_components,
)
from upstream_spec import (  # noqa: E402
    TEMPORAL_POOLED,
    apply_bucket_shifts,
    read_choices,
    temporal_shifts,
    transmission_targets,
)

console = Console()

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS = (2024, 2025)
ROLE_COLUMN = "role_bucket"

#: One training season is enough to fit a factor model, so the ablation gets
#: every forward fold the pre-2024 record supports.
MIN_FOLD_TRAINING_SEASONS = 1

R0 = "R0_ratio"
R1 = "R1_log_shrunk"
ARMS = (R0, R1)

#: The two architectures the ablation is run under. ``frozen`` is the final
#: model's own architecture and is the arm the rule reads; ``control`` is the
#: accepted repair's, carried only so the reader can see whether the answer
#: depends on the layers that sit upstream of the role scale.
PRIMARY_UNIVERSE = "frozen_architecture"
ROBUSTNESS_UNIVERSE = "control_architecture"
UNIVERSES = (PRIMARY_UNIVERSE, ROBUSTNESS_UNIVERSE)

# ======================================================================
# the frozen rule
# ======================================================================

#: Written before the first run. Nothing below reads a number and then decides
#: what to compare it against.
FROZEN_RULE: dict[str, object] = {
    "statement": (
        "select R1 only if all three criteria hold on the pre-2024 forward "
        "folds of the primary universe; otherwise select R0"
    ),
    "primary_universe": PRIMARY_UNIVERSE,
    "criterion_A": (
        "the support-weighted role-cell RMSE improves by more than one "
        "standard error of the paired per-fold difference, i.e. "
        "mean(R1 - R0) < -1.0 * se(R1 - R0) with a finite positive se"
    ),
    "criterion_B": (
        "no supported role cell worsens by more than 0.5 in rms z, where a "
        "cell's deterioration is the mean over folds of "
        "(rms z of R1 - rms z of R0)"
    ),
    "criterion_C": (
        "pooled neutrality, same-player preservation and PSD all hold for "
        "both arms in every fold"
    ),
    "improvement_standard_errors": 1.0,
    "max_cell_deterioration_rms_z": 0.5,
    "primary_objective": (
        "support-weighted RMSE over supported cells: "
        "sqrt(sum_c pairs_c * rmse_c^2 / sum_c pairs_c)"
    ),
    "secondary_objective_reported_not_decisive": (
        "equal-weighted RMSE over supported cells: sqrt(mean_c rmse_c^2), "
        "the convention the stored inner selection used"
    ),
    "pairing_unit": "forward fold",
    "tie_default": "R0, the accepted repair's treatment",
    "holdout_role": (
        "none; 2024 and 2025 are excluded from every frame this file touches"
    ),
}

#: Criterion C's tolerances. All three come from the original brief, not from
#: anything measured here.
WEIGHTED_MEAN_TOLERANCE = 1e-12
POOLED_BLOCK_TOLERANCE = 1e-10
SAME_PLAYER_TOLERANCE = 1e-9

#: Games sampled per fold for the PSD and same-player checks. The check is a
#: contract, not an estimate, so a sample that exercises every roster size and
#: every role mix is enough; it is the maximum deviation that is reported.
PSD_GAMES_PER_FOLD = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        type=Path,
        default=PROJECT_ROOT
        / "research/game_latent_state/oof_gaussian_residuals.parquet",
    )
    parser.add_argument(
        "--count-residuals",
        type=Path,
        default=PROJECT_ROOT
        / "data/research/game_latent_state/processed/v2_bridge_count_residuals.parquet",
    )
    parser.add_argument(
        "--choices-root",
        type=Path,
        default=PROJECT_ROOT / "research/final_upstream_remediation",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research/final_role_scale_ablation",
    )
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--bridge-bootstrap", type=int, default=400)
    parser.add_argument("--cell-bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def assert_holdout_absent(frame: pd.DataFrame, where: str) -> pd.DataFrame:
    """Fail loudly if a holdout season reached a selection frame."""
    present = sorted(
        int(season)
        for season in frame["season"].unique()
        if int(season) in HOLDOUT_SEASONS
    )
    if present:
        raise SystemExit(f"{where} contains holdout seasons {present}")
    return frame


# ======================================================================
# one fold, both arms
# ======================================================================


class RoleFold:
    """One forward chronological fold, fitted both ways on shared moments."""

    def __init__(
        self,
        frame: pd.DataFrame,
        count_frame: pd.DataFrame,
        target: int,
        universe: str,
        choices,
        bootstrap: int,
        bridge_bootstrap: int,
        cell_bootstrap: int,
        seed: int,
    ) -> None:
        self.target = int(target)
        self.universe = universe
        train_raw = assert_holdout_absent(
            frame.loc[frame["season"] < target], f"fold {target} training"
        )
        score_raw = assert_holdout_absent(
            frame.loc[frame["season"] == target], f"fold {target} scoring"
        )
        self.training_seasons = sorted(int(s) for s in train_raw["season"].unique())

        # One standardization, taken from the training seasons and applied to
        # both. Standardizing the scored season on itself would move the
        # target the two arms are compared against.
        self.train, self.standardization = standardize_residuals(train_raw, STATS)
        self.score, _ = standardize_residuals(
            score_raw, STATS, moments=self.standardization
        )
        console.print(
            f"  [{universe}] fold {target}: train {self.training_seasons} "
            f"({len(self.train):,} rows) -> score {target} ({len(self.score):,} rows)"
        )

        self.layers: dict[str, object] = {}
        moments = pair_moments(
            self.train, STATS, bootstrap=bootstrap, seed=seed + target
        )

        if universe == PRIMARY_UNIVERSE:
            spec = choices.spec()
            if choices.transmission_cap > 0.0:
                combined, bridge = transmission_targets(
                    self.train,
                    count_frame,
                    STATS,
                    cap=choices.transmission_cap,
                    bridge_bootstrap=bridge_bootstrap,
                    seed=seed + target,
                )
                moments = replace(
                    moments,
                    same_team=np.asarray(combined[0], dtype=float),
                    cross_team=np.asarray(combined[1], dtype=float),
                )
                self.layers["transmission_cap"] = float(choices.transmission_cap)
                self.layers["transmission_mean_bridge_weight"] = bridge["same_team"][
                    "mean_bridge_weight"
                ]
            if choices.temporal != TEMPORAL_POOLED:
                shifts, _ = temporal_shifts(
                    self.train,
                    STATS,
                    treatment=choices.temporal,
                    nu=choices.temporal_nu,
                    half_life=choices.temporal_half_life,
                    bootstrap=bootstrap,
                    seed=seed + target,
                )
                shifted_same, shifted_cross = apply_bucket_shifts(
                    moments.same_team, moments.cross_team, shifts, STATS
                )
                moments = replace(
                    moments, same_team=shifted_same, cross_team=shifted_cross
                )
                self.layers["temporal_treatment"] = choices.temporal
                self.layers["temporal_max_abs_shift"] = float(
                    max(abs(value) for value in shifts.values())
                )
        else:
            spec = RemediationSpec(name="control", role_scale_mode=ROLE_SCALE_RATIO)

        self.moments = moments
        # The role layer is fitted at the full dependence model and the
        # temperature is applied once at the end, so the projection never reads
        # tempered loadings. The frozen temperature is 1 and this is a no-op,
        # but the ordering is the fitted model's and is kept.
        self.fit_spec = replace(spec, dependence_temperature=1.0)

        # The projection reads only the pooled same-team block, which carries
        # no role layer, so the base both arms project onto is fitted without
        # one. Both arms therefore see the identical base.
        self.base_fit = fit_remediated_factors(
            self.train,
            STATS,
            spec=replace(self.fit_spec, role_column=None),
            moments=moments,
        )
        self.base = self.base_fit.loadings

        self.per_role_moments = {
            str(role): pair_moments(
                group,
                STATS,
                bootstrap=bootstrap,
                seed=seed + target,
                keep_draws=True,
            )
            for role, group in self.train.dropna(subset=[ROLE_COLUMN]).groupby(
                ROLE_COLUMN
            )
        }
        self.player_rows = {
            str(role): float(len(group))
            for role, group in self.train.dropna(subset=[ROLE_COLUMN]).groupby(
                ROLE_COLUMN
            )
        }

        # Observed cells on the unseen season, with their game-clustered
        # standard errors. Measured once, so both arms are judged against the
        # same numbers and the same support flags.
        self.cells = per_game_role_cells(self.score, STATS, ROLE_COLUMN)
        self.cell_se = {
            key: cell_standard_errors(
                cell, draws=cell_bootstrap, seed=seed + 7 * position + self.target
            )
            for position, (key, cell) in enumerate(sorted(self.cells.items()))
        }

        self.scales = {R0: self._ratio_scales(), R1: self._log_shrunk_scales()}
        self.fits = {
            arm: fit_remediated_factors(
                self.train,
                STATS,
                spec=self.fit_spec,
                moments=moments,
                role_scale_override=self.scales[arm]["scales"],
            )
            for arm in ARMS
        }

    # -- the two role-scale treatments ---------------------------------

    def _ratio_scales(self) -> dict[str, object]:
        """R0: the accepted repair's game-count-pooled projection."""
        scales = fit_role_scales(
            self.train, STATS, base=self.base, role_column=ROLE_COLUMN
        )
        return {
            "mode": ROLE_SCALE_RATIO,
            "scales": {role: float(value) for role, value in scales.items()},
            "payload": {"mode": ROLE_SCALE_RATIO},
        }

    def _log_shrunk_scales(self) -> dict[str, object]:
        """R1: the same projection shrunk in the log ratio by its own SE."""
        raw, log_se, shares = role_scale_components(
            self.train,
            STATS,
            base=self.base,
            role_column=ROLE_COLUMN,
            per_role_moments=self.per_role_moments,
        )
        shrunk = fit_log_shrunk_role_scales(raw, log_se, shares)
        return {
            "mode": ROLE_SCALE_LOG_SHRUNK,
            "scales": {role: float(value) for role, value in shrunk.scales.items()},
            "payload": shrunk.payload(),
        }

    # -- scoring -------------------------------------------------------

    def score_cells(self, arm: str) -> dict[str, object]:
        """Per-cell RMSE and rms z of one arm on the unseen season."""
        loadings = self.fits[arm].loadings
        by_cell: dict[str, object] = {}
        weights: list[float] = []
        squared: list[float] = []
        for first, second in ROLE_PAIR_CELLS:
            label = f"{first}+{second}"
            cell = self.cells.get((first, second))
            if cell is None:
                by_cell[label] = {"present": False}
                continue
            observed = np.asarray(cell["correlation"], dtype=float)
            implied = loadings.same_team_correlation_for_roles(first, second)
            residual = implied - observed
            standard_error = self.cell_se[(first, second)]
            safe = np.where(standard_error > 0.0, standard_error, np.nan)
            diagonal = np.arange(len(STATS))
            by_cell[label] = {
                "present": True,
                "supported": bool(cell["supported"]),
                "games": int(cell["games"]),
                "pairs": float(cell["pairs"]),
                "effective_games": float(cell["effective_games"]),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "rms_z": float(np.sqrt(np.nanmean(np.square(residual / safe)))),
                "max_abs_error": float(np.max(np.abs(residual))),
                "observed_scalar": float(np.mean(observed[diagonal, diagonal])),
                "implied_scalar": float(
                    np.mean(np.asarray(implied)[diagonal, diagonal])
                ),
            }
            if cell["supported"]:
                weights.append(float(cell["pairs"]))
                squared.append(float(np.mean(np.square(residual))))

        total = float(sum(weights))
        return {
            "by_cell": by_cell,
            "supported_cells": len(weights),
            "support_weighted_rmse": float(
                np.sqrt(np.sum(np.asarray(weights) * np.asarray(squared)) / total)
            )
            if total > 0
            else float("nan"),
            "equal_weighted_rmse": float(np.sqrt(np.mean(squared)))
            if squared
            else float("nan"),
        }

    # -- criterion C ---------------------------------------------------

    def identification(self, arm: str) -> dict[str, object]:
        """The identification and contract checks criterion C reads."""
        loadings = self.fits[arm].loadings
        scales = self.scales[arm]["scales"]
        roles = sorted(scales)

        total = sum(self.player_rows.get(role, 0.0) for role in roles)
        weights = (
            {role: self.player_rows.get(role, 0.0) / total for role in roles}
            if total > 0
            else {role: 1.0 / len(roles) for role in roles}
        )
        weighted_mean = float(sum(weights[role] * scales[role] for role in roles))

        pooled_change = float(
            np.max(
                np.abs(
                    loadings.same_team_correlation()
                    - self.base.same_team_correlation()
                )
            )
        )
        cross_change = float(
            np.max(
                np.abs(
                    loadings.cross_team_correlation()
                    - self.base.cross_team_correlation()
                )
            )
        )
        contract = self._same_player_and_psd(arm)
        return {
            "role_scale": {role: float(scales[role]) for role in roles},
            "player_share_weights": weights,
            "weighted_mean_scale": weighted_mean,
            "weighted_mean_scale_minus_one": weighted_mean - 1.0,
            "weighted_mean_holds": bool(
                abs(weighted_mean - 1.0) <= WEIGHTED_MEAN_TOLERANCE
            ),
            "pooled_same_team_block_max_change": pooled_change,
            "pooled_cross_team_block_max_change": cross_change,
            "pooled_neutrality_holds": bool(
                max(pooled_change, cross_change) <= POOLED_BLOCK_TOLERANCE
            ),
            "role_deviation_layer": False,
            "pairwise_parameter_count": 0,
            "player_indexed": 0,
            "role_scale_parameter_count": len(roles),
            **contract,
        }

    def _same_player_and_psd(self, arm: str) -> dict[str, object]:
        """Assemble real rosters and read the pinned block back out.

        The role scale multiplies the shared cross-player factors, so the one
        way it could break the integration contract is by leaving a
        shared-factor echo inside a player's own block. That is read off the
        assembled covariance rather than argued about.
        """
        loadings = self.fits[arm].loadings
        within = {}
        pooled_block = self._pooled_same_player_block()
        deviation = 0.0
        min_eigenvalue = float("inf")
        failures = 0
        shrink_engaged = 0
        games_checked = 0
        failure_detail: list[str] = []

        for game_id in self._psd_game_ids:
            rows = self._psd_rows.loc[self._psd_rows["game_id"] == game_id]
            dimensions: list[GameDimension] = []
            home = int(rows["team_id"].min())
            for row in rows.itertuples():
                for stat in STATS:
                    dimensions.append(
                        GameDimension(
                            player_id=int(row.player_id),
                            team_id=int(row.team_id),
                            stat=stat,
                            side=1 if int(row.team_id) == home else -1,
                            role=None
                            if pd.isna(getattr(row, ROLE_COLUMN))
                            else str(getattr(row, ROLE_COLUMN)),
                        )
                    )
            within = {
                int(row.player_id): pooled_block.copy() for row in rows.itertuples()
            }
            try:
                covariance = build_game_covariance(
                    dimensions=tuple(dimensions),
                    loadings=loadings,
                    within_player=within,
                )
            except Exception as error:  # pragma: no cover - recorded, not hidden
                failures += 1
                if len(failure_detail) < 5:
                    failure_detail.append(f"game {game_id}: {error}")
                continue
            games_checked += 1
            min_eigenvalue = min(min_eigenvalue, float(covariance.min_eigenvalue))
            if covariance.shrink_applied:
                shrink_engaged += 1
            for player_id, block in within.items():
                induced = implied_within_player_correlation(covariance, player_id)
                deviation = max(deviation, float(np.max(np.abs(induced - block))))

        return {
            "same_player_games_checked": games_checked,
            "same_player_max_block_deviation": deviation,
            "same_player_holds": bool(
                games_checked > 0 and deviation <= SAME_PLAYER_TOLERANCE
            ),
            "min_covariance_eigenvalue": None
            if min_eigenvalue == float("inf")
            else min_eigenvalue,
            "psd_failures": failures,
            "psd_failure_detail": failure_detail,
            "psd_holds": bool(
                games_checked > 0
                and failures == 0
                and min_eigenvalue >= -SAME_PLAYER_TOLERANCE
            ),
            "shared_shrink_engaged_games": shrink_engaged,
        }

    def prepare_contract_sample(self, seed: int) -> None:
        """Pick the games the contract checks run on, once, before both arms."""
        columns = ["game_id", "team_id", "player_id", ROLE_COLUMN]
        rows = self.score.dropna(subset=columns)[columns].drop_duplicates(
            subset=["game_id", "player_id"]
        )
        games = np.sort(rows["game_id"].unique())
        rng = np.random.default_rng(seed + self.target)
        take = min(PSD_GAMES_PER_FOLD, len(games))
        chosen = rng.choice(games, size=take, replace=False) if take else []
        self._psd_game_ids = tuple(int(value) for value in np.sort(chosen))
        self._psd_rows = rows.loc[rows["game_id"].isin(self._psd_game_ids)]

    def _pooled_same_player_block(self) -> np.ndarray:
        """The incumbent same-player block, pooled over the training rows.

        The production copula's within-player block is not available to a
        research driver, and the contract this checks is a structural one: the
        assembled covariance must hand back whatever block it was given. A
        pooled empirical same-player correlation is a real one and exercises
        the same code path.
        """
        if getattr(self, "_pooled_block", None) is None:
            columns = [f"zs_{stat}" for stat in STATS]
            values = self.train.dropna(subset=columns)[columns].to_numpy(float)
            block = np.corrcoef(values, rowvar=False)
            block = 0.5 * (block + block.T)
            np.fill_diagonal(block, 1.0)
            self._pooled_block = block
        return self._pooled_block

    _pooled_block: np.ndarray | None = None
    _psd_game_ids: tuple[int, ...] = ()
    _psd_rows: pd.DataFrame | None = None


# ======================================================================
# the ablation
# ======================================================================


def run_universe(
    frame: pd.DataFrame,
    count_frame: pd.DataFrame,
    targets: tuple[int, ...],
    universe: str,
    choices,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Both arms on every fold of one universe."""
    folds = []
    for target in targets:
        fold = RoleFold(
            frame,
            count_frame,
            target,
            universe=universe,
            choices=choices,
            bootstrap=args.bootstrap,
            bridge_bootstrap=args.bridge_bootstrap,
            cell_bootstrap=args.cell_bootstrap,
            seed=args.seed,
        )
        fold.prepare_contract_sample(args.seed)
        folds.append(fold)

    by_arm: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        per_fold = []
        for fold in folds:
            scored = fold.score_cells(arm)
            per_fold.append(
                {
                    "target_season": fold.target,
                    "training_seasons": fold.training_seasons,
                    "role_scale_fit": fold.scales[arm]["payload"],
                    "identification": fold.identification(arm),
                    "upstream_layers": fold.layers,
                    **scored,
                }
            )
        by_arm[arm] = {
            "mode": fold.scales[arm]["mode"],
            "folds": per_fold,
            "support_weighted_rmse_by_fold": {
                str(entry["target_season"]): entry["support_weighted_rmse"]
                for entry in per_fold
            },
            "equal_weighted_rmse_by_fold": {
                str(entry["target_season"]): entry["equal_weighted_rmse"]
                for entry in per_fold
            },
            "pooled_support_weighted_rmse": float(
                np.mean([entry["support_weighted_rmse"] for entry in per_fold])
            ),
            "pooled_equal_weighted_rmse": float(
                np.mean([entry["equal_weighted_rmse"] for entry in per_fold])
            ),
        }

    return {
        "universe": universe,
        "folds_scored": [fold.target for fold in folds],
        "arms": by_arm,
        "comparison": compare_arms(by_arm),
    }


def compare_arms(by_arm: dict[str, dict[str, object]]) -> dict[str, object]:
    """Paired differences and per-cell deteriorations, R1 against R0."""
    primary = {
        arm: by_arm[arm]["support_weighted_rmse_by_fold"] for arm in ARMS  # type: ignore[index]
    }
    secondary = {
        arm: by_arm[arm]["equal_weighted_rmse_by_fold"] for arm in ARMS  # type: ignore[index]
    }
    difference, se = paired_difference_se(primary[R1], primary[R0])
    secondary_difference, secondary_se = paired_difference_se(
        secondary[R1], secondary[R0]
    )

    # Pairing on (fold, cell) as well. Three folds give the fold-level
    # difference three observations, which is enough for the rule the brief
    # names and little else, so the finer pairing is reported beside it.
    cell_pairs: dict[str, float] = {}
    cell_reference: dict[str, float] = {}
    by_cell: dict[str, dict[str, object]] = {}
    for position, (first, second) in enumerate(ROLE_PAIR_CELLS):
        label = f"{first}+{second}"
        deltas_z: list[float] = []
        deltas_rmse: list[float] = []
        supported_everywhere = True
        for index, entry in enumerate(by_arm[R0]["folds"]):  # type: ignore[index]
            cell0 = entry["by_cell"].get(label, {})  # type: ignore[union-attr]
            cell1 = by_arm[R1]["folds"][index]["by_cell"].get(label, {})  # type: ignore[index]
            if not cell0.get("present") or not cell1.get("present"):
                supported_everywhere = False
                continue
            if not cell0.get("supported"):
                supported_everywhere = False
                continue
            deltas_z.append(float(cell1["rms_z"]) - float(cell0["rms_z"]))
            deltas_rmse.append(float(cell1["rmse"]) - float(cell0["rmse"]))
            key = f"{label}@{entry['target_season']}"
            cell_pairs[key] = float(cell1["rmse"])
            cell_reference[key] = float(cell0["rmse"])
        by_cell[label] = {
            "supported_in_every_fold": supported_everywhere,
            "folds_compared": len(deltas_z),
            "delta_rms_z_by_fold": deltas_z,
            "mean_delta_rms_z": float(np.mean(deltas_z)) if deltas_z else None,
            "worst_fold_delta_rms_z": float(np.max(deltas_z)) if deltas_z else None,
            "mean_delta_rmse": float(np.mean(deltas_rmse)) if deltas_rmse else None,
        }

    fine_difference, fine_se = paired_difference_se(cell_pairs, cell_reference)
    deteriorations = {
        label: entry["mean_delta_rms_z"]
        for label, entry in by_cell.items()
        if entry["mean_delta_rms_z"] is not None
    }
    worst = max(deteriorations, key=lambda key: deteriorations[key])
    return {
        "paired_difference_r1_minus_r0": {
            "objective": "support-weighted supported-cell RMSE",
            "pairing_unit": "forward fold",
            "mean": difference,
            "standard_error": se,
            "standard_errors_of_improvement": (
                None if not (np.isfinite(se) and se > 0) else float(-difference / se)
            ),
        },
        "paired_difference_equal_weighted": {
            "mean": secondary_difference,
            "standard_error": secondary_se,
        },
        "paired_difference_by_fold_and_cell": {
            "pairing_unit": "(fold, supported cell)",
            "observations": len(cell_pairs),
            "mean": fine_difference,
            "standard_error": fine_se,
        },
        "by_cell": by_cell,
        "worst_supported_cell_deterioration_rms_z": {
            "cell": worst,
            "mean_delta_rms_z": deteriorations[worst],
        },
    }


def apply_frozen_rule(primary: dict[str, object]) -> dict[str, object]:
    """The rule in ``FROZEN_RULE``, applied mechanically and once."""
    comparison = primary["comparison"]  # type: ignore[index]
    paired = comparison["paired_difference_r1_minus_r0"]  # type: ignore[index]
    difference = float(paired["mean"])
    se = float(paired["standard_error"])

    criterion_a = bool(
        np.isfinite(se)
        and se > 0.0
        and difference < -float(FROZEN_RULE["improvement_standard_errors"]) * se
    )

    worst = comparison["worst_supported_cell_deterioration_rms_z"]  # type: ignore[index]
    criterion_b = bool(
        float(worst["mean_delta_rms_z"])
        <= float(FROZEN_RULE["max_cell_deterioration_rms_z"])
    )

    checks: list[dict[str, object]] = []
    for arm in ARMS:
        for entry in primary["arms"][arm]["folds"]:  # type: ignore[index]
            identification = entry["identification"]
            checks.append(
                {
                    "arm": arm,
                    "target_season": entry["target_season"],
                    "weighted_mean_holds": identification["weighted_mean_holds"],
                    "pooled_neutrality_holds": identification[
                        "pooled_neutrality_holds"
                    ],
                    "same_player_holds": identification["same_player_holds"],
                    "psd_holds": identification["psd_holds"],
                }
            )
    criterion_c = all(
        check["weighted_mean_holds"]
        and check["pooled_neutrality_holds"]
        and check["same_player_holds"]
        and check["psd_holds"]
        for check in checks
    )

    selected = R1 if (criterion_a and criterion_b and criterion_c) else R0
    return {
        "rule": FROZEN_RULE,
        "criterion_A_rmse_improves_by_more_than_one_paired_se": criterion_a,
        "criterion_B_no_cell_worsens_by_more_than_half_a_z": criterion_b,
        "criterion_C_identification_and_contracts_hold": criterion_c,
        "identification_checks": checks,
        "selected": selected,
        "reason": (
            "all three criteria hold, so the log-shrunk form is selected"
            if selected == R1
            else "at least one criterion fails, so the accepted ratio form is "
            "retained: "
            + ", ".join(
                name
                for name, ok in (
                    ("A", criterion_a),
                    ("B", criterion_b),
                    ("C", criterion_c),
                )
                if not ok
            )
            + " did not hold"
        ),
        "ROLE_SCALE_SELECTION_FROZEN": "YES",
        "2024_2025_NOT_USED_FOR_ROLE_SELECTION": "YES",
    }


def render(report: dict[str, object]) -> None:
    for universe in UNIVERSES:
        block = report["universes"][universe]  # type: ignore[index]
        table = Table(
            title=f"{universe}: supported-cell RMSE by fold",
            show_header=True,
            header_style="bold",
        )
        table.add_column("arm")
        for target in block["folds_scored"]:
            table.add_column(str(target), justify="right")
        table.add_column("pooled", justify="right")
        for arm in ARMS:
            scores = block["arms"][arm]["support_weighted_rmse_by_fold"]
            table.add_row(
                arm,
                *[f"{scores[str(target)]:.6f}" for target in block["folds_scored"]],
                f"{block['arms'][arm]['pooled_support_weighted_rmse']:.6f}",
            )
        console.print(table)

        paired = block["comparison"]["paired_difference_r1_minus_r0"]
        console.print(
            f"  paired R1-R0: {paired['mean']:+.3e} "
            f"+/- {paired['standard_error']:.3e} "
            f"({paired['standard_errors_of_improvement']} SE of improvement)"
        )

    cells = Table(
        title="per-cell rms z, primary universe (mean over folds)",
        show_header=True,
        header_style="bold",
    )
    cells.add_column("cell")
    cells.add_column("supported", justify="center")
    cells.add_column("R0 rms z", justify="right")
    cells.add_column("R1 rms z", justify="right")
    cells.add_column("delta", justify="right")
    primary = report["universes"][PRIMARY_UNIVERSE]  # type: ignore[index]
    for first, second in ROLE_PAIR_CELLS:
        label = f"{first}+{second}"
        entry = primary["comparison"]["by_cell"][label]
        values = {}
        for arm in ARMS:
            folds = primary["arms"][arm]["folds"]
            observations = [
                f["by_cell"][label]["rms_z"]
                for f in folds
                if f["by_cell"][label].get("present")
            ]
            values[arm] = float(np.mean(observations)) if observations else float("nan")
        cells.add_row(
            label,
            "yes" if entry["supported_in_every_fold"] else "no",
            f"{values[R0]:.4f}",
            f"{values[R1]:.4f}",
            f"{entry['mean_delta_rms_z']:+.4f}"
            if entry["mean_delta_rms_z"] is not None
            else "-",
        )
    console.print(cells)

    decision = report["decision"]
    console.print(
        f"\n[bold]criterion A[/bold] "
        f"{decision['criterion_A_rmse_improves_by_more_than_one_paired_se']}  "
        f"[bold]criterion B[/bold] "
        f"{decision['criterion_B_no_cell_worsens_by_more_than_half_a_z']}  "
        f"[bold]criterion C[/bold] "
        f"{decision['criterion_C_identification_and_contracts_hold']}"
    )
    console.print(f"[bold]ROLE SCALE SELECTED: {decision['selected']}[/bold]")
    console.print("ROLE_SCALE_SELECTION_FROZEN=YES")
    console.print("2024_2025_NOT_USED_FOR_ROLE_SELECTION=YES")


def main() -> int:
    args = parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    console.rule("loading")
    frame = pd.read_parquet(args.residuals)
    seasons_available = sorted(int(value) for value in frame["season"].unique())
    frame = frame.loc[~frame["season"].isin(list(HOLDOUT_SEASONS))].copy()
    assert_holdout_absent(frame, "residual frame")
    count_frame = pd.read_parquet(args.count_residuals)
    count_frame = count_frame.loc[
        ~count_frame["season"].isin(list(HOLDOUT_SEASONS))
    ].copy()
    assert_holdout_absent(count_frame, "count residual frame")

    seasons = sorted(int(value) for value in frame["season"].unique())
    targets = forward_fold_seasons(seasons, MIN_FOLD_TRAINING_SEASONS)
    console.print(f"seasons used: {seasons}; forward folds: {list(targets)}")

    choices = read_choices(args.choices_root)
    console.print(f"frozen upstream choices: {choices.payload()}")

    universes: dict[str, object] = {}
    for universe in UNIVERSES:
        console.rule(universe)
        universes[universe] = run_universe(
            frame, count_frame, targets, universe, choices, args
        )

    decision = apply_frozen_rule(universes[PRIMARY_UNIVERSE])  # type: ignore[arg-type]

    report: dict[str, object] = {
        "title": "final role-scale ablation: R0 ratio against R1 log-shrunk",
        "scope": "SHADOW / RESEARCH ONLY. NOT PROMOTABLE.",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "code_sha": git_sha(PROJECT_ROOT),
        "stats": list(STATS),
        "role_column": ROLE_COLUMN,
        "seasons_available_in_the_residual_frame": seasons_available,
        "seasons_used": seasons,
        "holdout_seasons_excluded": list(HOLDOUT_SEASONS),
        "holdout_used_for_role_selection": False,
        "forward_fold_target_seasons": [int(value) for value in targets],
        "min_fold_training_seasons": MIN_FOLD_TRAINING_SEASONS,
        "arms": {
            R0: "accepted bucket-repair multiplicative role scale (ratio)",
            R1: "log-shrunk multiplicative role scale",
        },
        "third_candidate_offered": False,
        "support_threshold": {
            "min_cell_games": MIN_CELL_GAMES,
            "min_cell_pairs": MIN_CELL_PAIRS,
        },
        "role_pair_cells": [f"{a}+{b}" for a, b in ROLE_PAIR_CELLS],
        "bootstrap": args.bootstrap,
        "bridge_bootstrap": args.bridge_bootstrap,
        "cell_bootstrap": args.cell_bootstrap,
        "psd_games_per_fold": PSD_GAMES_PER_FOLD,
        "seed": args.seed,
        "frozen_upstream_choices": choices.payload(),
        "tolerances": {
            "weighted_mean_scale": WEIGHTED_MEAN_TOLERANCE,
            "pooled_block": POOLED_BLOCK_TOLERANCE,
            "same_player_block": SAME_PLAYER_TOLERANCE,
        },
        "universes": universes,
        "decision": decision,
        "ROLE_SCALE_SELECTION_FROZEN": "YES",
        "2024_2025_NOT_USED_FOR_ROLE_SELECTION": "YES",
    }

    destination = args.artifact_root / "role_scale_ablation.json"
    write_json(report, destination)
    write_checksums(
        [
            destination,
            Path(__file__).resolve(),
        ],
        args.artifact_root / "SHA256SUMS.ablation.txt",
    )
    console.rule("result")
    render(report)
    console.print(f"\nwrote {destination} ({sha256_file(destination)[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
