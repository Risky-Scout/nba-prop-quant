#!/usr/bin/env python
"""Apply the same-team axis indifference rule to a completed screening run.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The indifference rule is arithmetic over fold metrics the screen has already
written down, so it does not need the fits to be repeated. That is the whole
reason it lives in :func:`resolve_axis_indifference` rather than inline in
``03_inner_screening.py``: the third screening pass completed, recorded every
same-team grid point's fold metrics, and selected its same-team point with the
ordinary tie-break, which was the rule in force at the time. The rule was then
corrected. Re-running the pass to re-apply a decision rule to numbers the pass
already persisted would burn an expensive fitting run to recompute inputs that
are byte-identical, so this step reads them instead.

What it does
------------
Loads ``inner_screening.json``, takes the admissible same-team points exactly
as that artifact recorded them -- the ones that cleared the global latent bound
and the cross-team identity tolerance -- and compares each with the axis's null
value on the metrics the axis is judged by. If none of them moves a judged
metric by more than :data:`AXIS_INDIFFERENCE_TOLERANCE` relatively, the axis
resolves to its null value and the symmetric subspace is not carried.

It then resolves the stage-2 winner's spec at the resolved rank. This is a
substitution, not a re-selection: the combination set's specs are a
pre-registered function of the axis selections, so correcting one axis
selection determines the winner's spec without any further choice. When the
resolved spec coincides with a combination the pass already scored, that
combination's fold metrics are the resolved candidate's fold metrics, and the
run asserts the coincidence by comparing the specs field by field.

No residual data is read, no model is fitted, and the holdout is not touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.artifacts import (  # noqa: E402
    sha256_file,
    write_json,
)
from nba_prop_quant.research.game_latent_state.v2 import (  # noqa: E402
    AXIS_INDIFFERENCE_TOLERANCE,
    resolve_axis_indifference,
)

console = Console()

HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

#: The axis this step resolves, its null value, and the metrics it is judged by.
SAME_TEAM_AXIS = "same_team"
SAME_TEAM_NULL_POINT = "iso_rsym0"
SAME_TEAM_JUDGED_METRICS: tuple[str, ...] = (
    "mean_target_abs_error",
    "mean_global_latent_rmse",
    "mean_global_count_rmse",
)

#: The stage-2 row the freeze may take. The reference twin is excluded from
#: selection in the screen and stays excluded here.
STAGE_TWO_WINNER = "v2_iso"

#: Spec fields that identify a fit. ``name`` is excluded: two rows of the
#: combination set can be the same model under different labels, and that is
#: exactly the coincidence this step needs to be able to detect.
IDENTIFYING_SPEC_FIELDS: tuple[str, ...] = (
    "same_shrinkage",
    "same_eb_family",
    "cross_shrinkage",
    "cross_eb_family",
    "shrink_z",
    "k_game",
    "r_contrast",
    "r_symmetric",
    "symmetric_mode",
    "role_deviation",
    "role_column",
    "bridge_weight",
    "temporal_treatment",
)


def git_sha(ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def identifying_spec(spec: dict) -> dict:
    return {field: spec[field] for field in IDENTIFYING_SPEC_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(PROJECT_ROOT / "research" / "game_latent_state_v2"),
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_root)
    screening_path = artifact_dir / "inner_screening.json"
    if not screening_path.exists():
        raise SystemExit(f"run 03_inner_screening.py first: {screening_path} missing")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    if screening.get("holdout_used_for_selection", True):
        raise SystemExit("the screening claims the holdout was used; refusing")

    candidates = screening["candidates"]
    guards = screening["guards"]
    recorded_point = screening["axis_selections"][SAME_TEAM_AXIS]

    console.rule(
        f"Resolving the {SAME_TEAM_AXIS} axis on pass {screening['screening_pass']}"
    )
    console.print(f"  recorded selection : {recorded_point}")

    # Prefer the screen's own admissible list. A run that predates the rule did
    # not write one, but it did write both guard thresholds and every point's
    # metrics, so the set it admitted is recoverable exactly -- with the
    # thresholds read from the artifact rather than restated here, so this step
    # cannot widen or narrow the set the screen actually used.
    if "r_symmetric_eligible_points" in guards:
        admissible = sorted(guards["r_symmetric_eligible_points"])
        admissible_source = "recorded by the screening run"
    else:
        latent_bound = float(guards["global_latent_rmse_bound"])
        identity_tolerance = float(guards["cross_team_identity_tolerance"])
        admissible = sorted(
            name
            for name in screening["axes"][SAME_TEAM_AXIS]
            if name != SAME_TEAM_NULL_POINT
            and float(candidates[name]["mean_global_latent_rmse"]) <= latent_bound
            and float(candidates[name]["max_cross_team_unchanged_deviation"])
            <= identity_tolerance
        )
        admissible_source = (
            "reconstructed by applying the guard thresholds the screening run "
            f"recorded (latent bound {latent_bound:.9g}, cross-team identity "
            f"tolerance {identity_tolerance:g}) to the metrics it recorded"
        )
    console.print(f"  admissible points  : {admissible}")
    console.print(f"  admissible set     : {admissible_source}")

    indifference = resolve_axis_indifference(
        candidates,
        admissible,
        null_point=SAME_TEAM_NULL_POINT,
        metrics=SAME_TEAM_JUDGED_METRICS,
    )

    table = Table(title=f"deviation from {SAME_TEAM_NULL_POINT}, relative")
    table.add_column("point", justify="right")
    for metric in SAME_TEAM_JUDGED_METRICS:
        table.add_column(metric, justify="right")
    table.add_column("worst", justify="right")
    table.add_column("verdict", justify="center")
    null = candidates[SAME_TEAM_NULL_POINT]

    def relative(name: str, metric: str) -> float:
        baseline = float(null[metric])
        return abs(float(candidates[name][metric]) - baseline) / max(
            abs(baseline), 1e-30
        )

    for name in admissible:
        table.add_row(
            name,
            *[
                f"{relative(name, metric):.3e}"
                for metric in SAME_TEAM_JUDGED_METRICS
            ],
            f"{indifference.relative_deviation[name]:.3e}",
            "same model" if name in indifference.indistinguishable else "distinct",
        )
    console.print(table)

    if indifference.axis_is_indifferent:
        resolved_point = SAME_TEAM_NULL_POINT
        reason = (
            f"every admissible point reproduced {SAME_TEAM_NULL_POINT} to within "
            f"{AXIS_INDIFFERENCE_TOLERANCE:g} relatively on "
            f"{list(SAME_TEAM_JUDGED_METRICS)}, so the subspace is a "
            "reparameterisation at this base and cannot earn its parameters"
        )
    elif not admissible:
        resolved_point = SAME_TEAM_NULL_POINT
        reason = "no point satisfied the axis guards, so the axis carries nothing"
    else:
        resolved_point = recorded_point
        reason = (
            f"{len(indifference.distinguishable)} admissible point(s) moved a "
            "judged metric, so the screen's tie-break stands"
        )
    console.print(f"\n  resolved selection : {resolved_point}\n  because {reason}")

    resolved_rank = int(candidates[resolved_point]["spec"]["r_symmetric"])
    resolved_mode = str(candidates[resolved_point]["spec"]["symmetric_mode"])
    recorded_winner_spec = candidates[STAGE_TWO_WINNER]["spec"]
    resolved_winner_spec = {
        **identifying_spec(recorded_winner_spec),
        "r_symmetric": resolved_rank,
        "symmetric_mode": resolved_mode,
    }

    # The resolved winner's spec is a pre-registered function of the axis
    # selections, so if a scored combination matches it field for field, that
    # combination's fold metrics are the resolved candidate's fold metrics and
    # nothing has to be refitted to know them.
    equivalent = sorted(
        name
        for name, entry in candidates.items()
        if identifying_spec(entry["spec"]) == resolved_winner_spec
    )
    console.print(
        f"  resolved winner spec already scored as: {equivalent or 'nothing'}"
    )

    resolution = {
        "resolves": {
            "artifact": "inner_screening.json",
            "artifact_sha256": sha256_file(screening_path),
            "screening_pass": int(screening["screening_pass"]),
            "axis": SAME_TEAM_AXIS,
        },
        "rule": {
            "null_point": SAME_TEAM_NULL_POINT,
            "judged_metrics": list(SAME_TEAM_JUDGED_METRICS),
            "tolerance": AXIS_INDIFFERENCE_TOLERANCE,
            "statement": (
                "An admissible grid point that reproduces the axis's null value "
                "on every judged metric to within the tolerance relatively is "
                "the same model written differently, so it cannot buy "
                "parameters and the axis resolves to its null value."
            ),
        },
        "admissible_points": admissible,
        "admissible_points_source": admissible_source,
        "relative_deviation_from_null": indifference.relative_deviation,
        "indistinguishable_from_null": list(indifference.indistinguishable),
        "distinguishable_from_null": list(indifference.distinguishable),
        "axis_is_indifferent": indifference.axis_is_indifferent,
        "recorded_selection": recorded_point,
        "resolved_selection": resolved_point,
        "selection_changed": resolved_point != recorded_point,
        "reason": reason,
        "resolved_r_symmetric": resolved_rank,
        "resolved_symmetric_mode": resolved_mode,
        "stage_two_winner": STAGE_TWO_WINNER,
        "recorded_winner_spec": identifying_spec(recorded_winner_spec),
        "resolved_winner_spec": resolved_winner_spec,
        "resolved_winner_already_scored_as": equivalent,
        "refitted_anything": False,
        "residual_data_read": False,
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "holdout_used_for_selection": False,
        "code_sha": git_sha(),
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
    }
    path = write_json(resolution, artifact_dir / "axis_indifference_resolution.json")
    console.rule("Resolved")
    console.print(f"AXIS_INDIFFERENCE_RESOLVED={resolved_point}")
    console.print(f"RESOLVED_R_SYMMETRIC={resolved_rank}")
    changed = "YES" if resolution["selection_changed"] else "NO"
    console.print(f"SELECTION_CHANGED={changed}")
    console.print("SCREENING_RERUN_REQUIRED=NO")
    console.print("2024_2025_NOT_USED_FOR_SELECTION=YES")
    console.print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
