#!/usr/bin/env python
"""Confirm the selected role treatment on the holdout without regrading it.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Runs after :mod:`01_role_scale_ablation` and cannot change what it selected.

The ablation selected R1, the log-shrunk multiplicative role scale, which is
the treatment the already-validated frozen candidate
``3229bbc8...`` carries. So the final point-probability model is not a new
model and there is nothing to grade: the loadings are the same file, and every
module that turns loadings into a probability is byte-identical to the commit
that graded them. This script proves that rather than asserting it, and then
re-reads the holdout role cells, which is linear algebra on stored loadings and
needs no simulation at all.

Three things are established here:

1.  the selected treatment is the frozen candidate's recorded treatment, and
    the frozen candidate's role scale is reproduced bitwise from its own
    recorded components;

2.  every file on the path from (factor spec, marginals, rosters) to a joint
    probability is byte-identical to the grading commit, so the stored Brier
    and log loss describe the final model;

3.  all six role cells on the identical holdout observations, for the selected
    treatment and for an R0 counterfactual that differs only in the
    three-element role-scale vector.

Monte Carlo is deliberately not re-run. ``MONTE_CARLO_RERUN`` is reported as
``NO`` with the equivalence that licenses it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    git_sha,
    sha256_canonical,
    sha256_file,
    write_checksums,
    write_json,
)
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    SharedFactorLoadings,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    fit_role_scales,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.remediation import (  # noqa: E402
    ROLE_PAIR_CELLS,
    ROLE_SCALE_LOG_SHRUNK,
    cell_standard_errors,
    fit_log_shrunk_role_scales,
    per_game_role_cells,
)

console = Console()

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS = (2024, 2025)
ROLE_COLUMN = "role_bucket"

#: The authoritative candidate, from the brief. Not read from an artifact,
#: so an artifact that disagrees is caught rather than believed.
AUTHORITATIVE_SPEC_HASH = (
    "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
)

#: The commit whose run produced the graded joint events.
GRADING_CODE_SHA = "9343de38107feca14090697dabbd85f6a895d223"

#: Everything on the path from a stored factor spec to a joint probability.
#: A change to any of these would mean the stored grades no longer describe
#: what the final model would produce, and the confirmation would have to be
#: re-run instead of reused.
PROBABILITY_PATH_FILES = (
    "src/nba_prop_quant/research/game_latent_state/covariance.py",
    "src/nba_prop_quant/research/game_latent_state/simulator.py",
    "src/nba_prop_quant/research/game_latent_state/pit.py",
    "src/nba_prop_quant/research/game_latent_state/query.py",
    "src/nba_prop_quant/research/game_latent_state/factors.py",
    "src/nba_prop_quant/research/game_latent_state/validation.py",
    "src/nba_prop_quant/research/game_latent_state/gates.py",
    "research/game_latent_state/04_validate_shadow_v1.py",
    "research/final_upstream_remediation/04_paired_joint_calibration.py",
    "research/final_upstream_remediation/factor_spec.json",
)

#: Modules that were touched after the grading commit. Each has to be shown to
#: be unreachable from the probability path, which is checked by import rather
#: than claimed.
MODULES_CHANGED_AFTER_GRADING = (
    "src/nba_prop_quant/research/game_latent_state/remediation.py",
    "src/nba_prop_quant/research/game_latent_state/transmission.py",
)

CELL_BOOTSTRAP_DRAWS = 400
SEED = 73


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residuals",
        type=Path,
        default=PROJECT_ROOT
        / "research/game_latent_state/oof_gaussian_residuals.parquet",
    )
    parser.add_argument(
        "--remediation-root",
        type=Path,
        default=PROJECT_ROOT / "research/final_upstream_remediation",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=PROJECT_ROOT
        / "research/final_role_scale_ablation/role_scale_ablation.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "research/final_role_scale_ablation",
    )
    return parser.parse_args()


def blob_hash(revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}:{path}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


# ======================================================================
# 1. the selected treatment is the frozen candidate's
# ======================================================================


def selection_matches_frozen_candidate(
    ablation: dict, spec: dict, diagnostics: dict
) -> dict[str, object]:
    """The selected arm, the spec's recorded mode and the stored fit agree."""
    selected = str(ablation["decision"]["selected"])
    selected_mode = (
        ROLE_SCALE_LOG_SHRUNK if selected.startswith("R1") else "ratio"
    )
    recorded_mode = str(spec["remediation_spec"]["role_scale_mode"])

    # The frozen role scale is reproduced from its own recorded components,
    # which is an exact arithmetic check rather than a refit: the shrinkage is
    # a deterministic function of (raw scale, log SE, player share) and all
    # three were written down.
    fit = diagnostics["role_scale"]
    reproduced = fit_log_shrunk_role_scales(
        {k: float(v) for k, v in fit["raw_scales"].items()},
        {k: float(v) for k, v in fit["log_standard_errors"].items()},
        {k: float(v) for k, v in fit["player_shares"].items()},
    )
    recorded_scales = {k: float(v) for k, v in spec["loadings"]["role_scale"].items()}
    bitwise = all(
        float(reproduced.scales[role]) == recorded_scales[role]
        for role in recorded_scales
    )
    deviation = max(
        abs(float(reproduced.scales[role]) - recorded_scales[role])
        for role in recorded_scales
    )

    stored = dict(spec)
    stored_hash = stored.pop("spec_hash")
    return {
        "selected_arm": selected,
        "selected_role_scale_mode": selected_mode,
        "frozen_candidate_recorded_mode": recorded_mode,
        "selection_is_the_frozen_candidates_treatment": selected_mode == recorded_mode,
        "frozen_role_scale": recorded_scales,
        "reproduced_role_scale": {
            k: float(v) for k, v in reproduced.scales.items()
        },
        "role_scale_reproduced_bitwise": bitwise,
        "role_scale_max_abs_deviation": deviation,
        "tau_log_recorded": float(fit["tau_log"]),
        "tau_log_reproduced": float(reproduced.tau_log),
        "recorded_spec_hash": stored_hash,
        "recomputed_spec_hash": sha256_canonical(stored),
        "spec_hash_matches_recorded": sha256_canonical(stored) == stored_hash,
        "spec_hash_matches_authoritative": stored_hash == AUTHORITATIVE_SPEC_HASH,
        "authoritative_spec_hash": AUTHORITATIVE_SPEC_HASH,
    }


# ======================================================================
# 2. the probability path has not moved since grading
# ======================================================================


def probability_path_equivalence() -> dict[str, object]:
    """Byte-identity of every file that maps a spec to a probability."""
    by_file: dict[str, object] = {}
    identical = True
    for path in PROBABILITY_PATH_FILES:
        at_grading = blob_hash(GRADING_CODE_SHA, path)
        at_head = blob_hash("HEAD", path)
        same = at_grading is not None and at_grading == at_head
        identical = identical and same
        by_file[path] = {
            "blob_at_grading_commit": at_grading,
            "blob_at_head": at_head,
            "identical": same,
        }

    # The two modules that did change are only reachable from the *fitting*
    # path. Asking an interpreter that has imported nothing else what the
    # probability path pulled in is stronger than reading the import lines,
    # because it sees transitive imports; it has to run in a fresh process,
    # since this one imports the fitting path itself.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})\n"
            "for name in ('simulator', 'query', 'covariance', 'pit'):\n"
            "    __import__('nba_prop_quant.research.game_latent_state.' + name)\n"
            "import json\n"
            "print(json.dumps(sorted(\n"
            "    name for name in sys.modules\n"
            "    if name.startswith('nba_prop_quant.research.game_latent_state.')\n"
            ")))\n",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = set(json.loads(probe.stdout))
    reachable = {
        path: f"nba_prop_quant.research.game_latent_state.{Path(path).stem}" in loaded
        for path in MODULES_CHANGED_AFTER_GRADING
    }

    return {
        "grading_code_sha": GRADING_CODE_SHA,
        "head_sha": git_sha(PROJECT_ROOT),
        "by_file": by_file,
        "every_probability_path_file_is_byte_identical": identical,
        "modules_changed_after_grading": list(MODULES_CHANGED_AFTER_GRADING),
        "probability_path_imports": sorted(loaded),
        "changed_module_reachable_from_probability_path": reachable,
        "no_changed_module_is_on_the_probability_path": not any(reachable.values()),
        "conclusion": (
            "the final point-probability model is bitwise identical to the "
            "graded one: the same factor spec file, the same simulator, the "
            "same inverse-CDF tabulation and the same event generator"
            if identical and not any(reachable.values())
            else "the probability path has moved; the confirmation cannot be "
            "reused and has to be re-run"
        ),
    }


# ======================================================================
# 3. the six role cells on the identical holdout observations
# ======================================================================


def holdout_role_cells(
    residuals: pd.DataFrame,
    spec: dict,
) -> dict[str, object]:
    """All six cells for the selected treatment and an R0 counterfactual.

    The counterfactual is the final loadings with R0's role-scale vector
    substituted. Nothing else differs -- not the game factors, not the
    contrast factors, not the competition block, not the standardization --
    so the comparison prices the role layer and only the role layer.
    """
    moments = {
        stat: (float(value["mean"]), float(value["sd"]))
        for stat, value in spec["standardization_moments"].items()
    }
    selected = SharedFactorLoadings.from_payload(spec["loadings"])

    # R0 on the same training rows the frozen candidate used. fit_role_scales
    # reads only the pooled same-team block, which carries no role layer, so
    # the frozen loadings serve directly as the base.
    training = residuals.loc[~residuals["season"].isin(list(HOLDOUT_SEASONS))]
    standardized_training, _ = standardize_residuals(training, STATS, moments=moments)
    counterfactual_scales = fit_role_scales(
        standardized_training, STATS, base=selected, role_column=ROLE_COLUMN
    )
    counterfactual = SharedFactorLoadings(
        stats=selected.stats,
        game=selected.game,
        team_contrast=selected.team_contrast,
        competition=selected.competition,
        role_scale={k: float(v) for k, v in counterfactual_scales.items()},
    )

    held_out = residuals.loc[residuals["season"].isin(list(HOLDOUT_SEASONS))]
    standardized, _ = standardize_residuals(held_out, STATS, moments=moments)
    observed = per_game_role_cells(standardized, STATS, ROLE_COLUMN)

    arms = {"R1_log_shrunk_selected": selected, "R0_ratio_counterfactual": counterfactual}
    by_cell: dict[str, object] = {}
    squared = {name: [] for name in arms}
    weights: list[float] = []
    worst_label: str | None = None
    worst_delta = -float("inf")

    for position, (first, second) in enumerate(ROLE_PAIR_CELLS):
        label = f"{first}+{second}"
        cell = observed.get((first, second))
        if cell is None:
            by_cell[label] = {"present": False}
            continue
        truth = np.asarray(cell["correlation"], dtype=float)
        standard_error = cell_standard_errors(
            cell, draws=CELL_BOOTSTRAP_DRAWS, seed=SEED + position
        )
        safe = np.where(standard_error > 0.0, standard_error, np.nan)
        diagonal = np.arange(len(STATS))
        entry: dict[str, object] = {
            "present": True,
            "supported": bool(cell["supported"]),
            "games": int(cell["games"]),
            "pairs": float(cell["pairs"]),
            "effective_games": float(cell["effective_games"]),
            "observed_scalar": float(np.mean(truth[diagonal, diagonal])),
            "scalar_se": float(np.mean(standard_error[diagonal, diagonal])),
        }
        for name, loadings in arms.items():
            implied = loadings.same_team_correlation_for_roles(first, second)
            residual = implied - truth
            entry[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(residual))))
            entry[f"{name}_rms_z"] = float(
                np.sqrt(np.nanmean(np.square(residual / safe)))
            )
            entry[f"{name}_max_abs_error"] = float(np.max(np.abs(residual)))
            entry[f"{name}_scalar"] = float(
                np.mean(np.asarray(implied)[diagonal, diagonal])
            )
            if cell["supported"]:
                squared[name].append(float(np.mean(np.square(residual))))
        entry["delta_rmse_selected_minus_counterfactual"] = (
            entry["R1_log_shrunk_selected_rmse"]
            - entry["R0_ratio_counterfactual_rmse"]
        )
        entry["delta_rms_z_selected_minus_counterfactual"] = (
            entry["R1_log_shrunk_selected_rms_z"]
            - entry["R0_ratio_counterfactual_rms_z"]
        )
        if cell["supported"]:
            weights.append(float(cell["pairs"]))
            if float(entry["delta_rms_z_selected_minus_counterfactual"]) > worst_delta:
                worst_delta = float(
                    entry["delta_rms_z_selected_minus_counterfactual"]
                )
                worst_label = label
        by_cell[label] = entry

    total = float(sum(weights))
    pooled = {
        name: {
            "support_weighted_rmse": float(
                np.sqrt(
                    np.sum(np.asarray(weights) * np.asarray(squared[name])) / total
                )
            ),
            "equal_weighted_rmse": float(np.sqrt(np.mean(squared[name]))),
        }
        for name in arms
    }

    return {
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "observations_are_identical_across_arms": True,
        "monte_carlo_used": False,
        "definition": (
            "per cell, the root mean square over the 6x6 stat block of "
            "(model-implied same-team correlation for the ordered role pair "
            "minus the held-out observed one); rms z divides each entry by "
            "its game-clustered bootstrap standard error"
        ),
        "counterfactual_definition": (
            "the final loadings with R0's role-scale vector substituted, "
            "fitted on the same pre-2024 training rows; nothing else differs"
        ),
        "selected_role_scale": {
            k: float(v) for k, v in selected.role_scale.items()
        },
        "counterfactual_role_scale": {
            k: float(v) for k, v in counterfactual.role_scale.items()
        },
        "cells_supported": len(weights),
        "cell_bootstrap_draws": CELL_BOOTSTRAP_DRAWS,
        "by_cell": by_cell,
        "pooled": pooled,
        "worst_supported_cell_deterioration_rms_z": {
            "cell": worst_label,
            "delta_rms_z": None if worst_label is None else worst_delta,
        },
        "confirmation_may_change_the_selection": False,
    }


def stored_grades(remediation_root: Path) -> dict[str, object]:
    """The graded proper scores, read rather than recomputed."""
    validation = json.loads(
        (remediation_root / "validation_report.json").read_text(encoding="utf-8")
    )
    paired = json.loads(
        (remediation_root / "paired_joint_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    by_legs = {
        legs: {
            "events": int(entry["events"]),
            "games": int(entry["games"]),
            "base_rate": float(entry["base_rate"]),
            "brier_candidate": float(entry["brier_candidate"]),
            "brier_control": float(entry["brier_control"]),
            "brier_independence": float(entry["brier_independence"]),
            "brier_production": float(entry["brier_production"]),
            "candidate_minus_control": entry["candidate_minus_control"],
            "candidate_minus_independence": entry["candidate_minus_independence"],
        }
        for legs, entry in paired["by_legs"].items()
    }
    log_loss = {
        legs: {
            name: float(entry[name]["log_loss"])
            for name in ("candidate", "baseline_independence", "baseline_production")
            if name in entry
        }
        for legs, entry in validation["joint_events"]["by_legs"].items()
    }
    return {
        "source": {
            "validation_report.json": sha256_file(
                remediation_root / "validation_report.json"
            ),
            "paired_joint_calibration.json": sha256_file(
                remediation_root / "paired_joint_calibration.json"
            ),
            "joint_event_grades.parquet": sha256_file(
                remediation_root / "joint_event_grades.parquet"
            ),
        },
        "validation_seasons": validation["validation_seasons"],
        "factor_spec_hash_in_validation_report": validation["factor_spec_hash"],
        "verdict": validation["verdict"],
        "brier_by_legs": by_legs,
        "log_loss_by_legs": log_loss,
        "regraded_here": False,
    }


def render(report: dict[str, object]) -> None:
    equivalence = report["probability_path_equivalence"]
    console.print(
        f"[bold]probability path byte-identical to grading commit:[/bold] "
        f"{equivalence['every_probability_path_file_is_byte_identical']}"
    )
    console.print(
        f"[bold]no changed module on the probability path:[/bold] "
        f"{equivalence['no_changed_module_is_on_the_probability_path']}"
    )
    console.print(f"[bold]MONTE_CARLO_RERUN:[/bold] {report['MONTE_CARLO_RERUN']}")

    cells = report["holdout_role_cells"]
    table = Table(
        title="holdout role cells, identical observations, no simulation",
        show_header=True,
        header_style="bold",
    )
    for column in (
        "cell",
        "sup",
        "games",
        "pairs",
        "R1 rmse",
        "R0 rmse",
        "R1 rms z",
        "R0 rms z",
        "delta rms z",
    ):
        table.add_column(column, justify="right" if column != "cell" else "left")
    for first, second in ROLE_PAIR_CELLS:
        label = f"{first}+{second}"
        entry = cells["by_cell"][label]
        table.add_row(
            label,
            "yes" if entry["supported"] else "no",
            f"{entry['games']:,}",
            f"{int(entry['pairs']):,}",
            f"{entry['R1_log_shrunk_selected_rmse']:.6f}",
            f"{entry['R0_ratio_counterfactual_rmse']:.6f}",
            f"{entry['R1_log_shrunk_selected_rms_z']:.4f}",
            f"{entry['R0_ratio_counterfactual_rms_z']:.4f}",
            f"{entry['delta_rms_z_selected_minus_counterfactual']:+.4f}",
        )
    console.print(table)
    console.print(
        "  pooled support-weighted RMSE: "
        f"R1 {cells['pooled']['R1_log_shrunk_selected']['support_weighted_rmse']:.6f}, "
        f"R0 {cells['pooled']['R0_ratio_counterfactual']['support_weighted_rmse']:.6f}"
    )


def main() -> int:
    args = parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    ablation = json.loads(args.ablation.read_text(encoding="utf-8"))
    spec = json.loads(
        (args.remediation_root / "factor_spec.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (args.remediation_root / "covariance_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )

    console.rule("1. the selected treatment is the frozen candidate's")
    selection = selection_matches_frozen_candidate(ablation, spec, diagnostics)
    for key in (
        "selection_is_the_frozen_candidates_treatment",
        "role_scale_reproduced_bitwise",
        "spec_hash_matches_recorded",
        "spec_hash_matches_authoritative",
    ):
        console.print(f"  {key}: {selection[key]}")

    console.rule("2. the probability path has not moved")
    equivalence = probability_path_equivalence()

    console.rule("3. holdout role cells")
    residuals = pd.read_parquet(args.residuals)
    cells = holdout_role_cells(residuals, spec)

    already_graded = bool(
        selection["selection_is_the_frozen_candidates_treatment"]
        and selection["spec_hash_matches_authoritative"]
        and equivalence["every_probability_path_file_is_byte_identical"]
        and equivalence["no_changed_module_is_on_the_probability_path"]
    )

    report: dict[str, object] = {
        "title": "holdout confirmation of the selected role treatment",
        "scope": "SHADOW / RESEARCH ONLY. NOT PROMOTABLE.",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "code_sha": git_sha(PROJECT_ROOT),
        "runs_after": "01_role_scale_ablation.py",
        "may_change_the_selection": False,
        "selected_arm": selection["selected_arm"],
        "selection_equivalence": selection,
        "probability_path_equivalence": equivalence,
        "final_model_is_bitwise_identical_to_an_already_graded_model": already_graded,
        "MONTE_CARLO_RERUN": "NO" if already_graded else "REQUIRED",
        "monte_carlo_rerun_justification": (
            "the selected role treatment is the frozen candidate's, the frozen "
            "factor spec file is unchanged and hashes to the authoritative "
            "value, and every module on the path from that file to a joint "
            "probability is byte-identical to the commit that graded it; "
            "re-running the simulation would reproduce the stored grades draw "
            "for draw, because the seed, the game universe and the inverse-CDF "
            "tabulation are all the same"
            if already_graded
            else "equivalence could not be established"
        ),
        "stored_holdout_grades": stored_grades(args.remediation_root),
        "holdout_role_cells": cells,
    }

    destination = args.artifact_root / "holdout_confirmation.json"
    write_json(report, destination)
    write_checksums(
        [destination, Path(__file__).resolve()],
        args.artifact_root / "SHA256SUMS.confirmation.txt",
    )
    console.rule("result")
    render(report)
    console.print(f"\nwrote {destination} ({sha256_file(destination)[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
