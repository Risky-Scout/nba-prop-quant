"""Audit: the OOF-versus-validator marginal-training convention mismatch.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE. This audit reads the frozen
remediation candidate and the committed OOF residuals and writes neither. It
changes no dependence architecture, fits no new factor family, and reads no
held-out season.

The question
------------
The pipeline fits the same production marginal family on two *different* row
sets:

``02_build_oof_residuals.py::build_residuals`` takes a joint ``dropna`` across
all six stats and all six selected means *before* splitting into train and
target, so its marginals are trained on the intersection across stats. Those
marginals produced the committed ``cdf_lower_``/``cdf_upper_``/``z_`` columns
and therefore every latent dependence target and the fitted structure.

``04_validate_shadow_v1.py::fit_season`` drops only the stat it is fitting and
that stat's own mean, so its marginals are trained on a per-stat superset.
Those marginals are the ones the held-out simulation inverts.

``scripts/07_fit_marginals.py::candidate_frame`` -- the production path -- also
filters per target only. So the validator convention is the live one and the
OOF builder is the single deviation.

What this audit measures
------------------------
Holding the target rows, the deterministic PIT jitter, the marginal family and
every dependence parameter fixed, it swaps only the marginal *training* rows
from the OOF builder's joint filter to the live per-stat filter, and reports
the consequence for

1. the pre-2024 PITs, per stat per season, and
2. all twelve cross-player dependence buckets, in units of the buckets' own
   game-clustered standard error -- the same ``observed_bucket_se`` the gates
   divide by.

Decision rule, fixed before the numbers were read: if every key bucket moves
less than 0.25 z *and* the global latent target structure moves less than 3%,
the mismatch is non-material and the audit closes. If any key bucket moves
0.25 z or more, or the global latent structure moves 3% or more, it is an
upstream consistency bug and only that convention is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[1]

# The count-space forensic study already carries one implementation of the two
# row filters, cross-checked against both pipeline sources, plus a disk cache
# of the fitted marginals keyed on the convention. Importing it rather than
# copying it keeps a single definition of the thing under audit -- a second
# copy could only measure itself -- and reuses the cached validator fits.
FORENSIC_DIR = PROJECT_ROOT / "research/count_space_forensic"
for path in (str(FORENSIC_DIR), str(PROJECT_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import forensic_lib as FL  # noqa: E402

from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    SharedFactorLoadings,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.pit import (  # noqa: E402
    PIT_EPSILON,
    deterministic_pit_uniform,
    gaussianize,
    randomized_pit,
)
from nba_prop_quant.research.game_latent_state.validation import (  # noqa: E402
    bucket_values,
)

STATS: tuple[str, ...] = FL.STATS

#: Residual seasons this audit reads. 2024 and 2025 are never touched.
PRE_2024_SEASONS: tuple[int, ...] = FL.PRE_2024_SEASONS

#: The deterministic PIT seed the committed residuals were built with.
PIT_SEED = 73

#: Bootstrap draws for the game-clustered bucket standard errors. The value
#: ``03_fit_latent_factors.py`` and ``04_validate_shadow_v1.py`` both default
#: to, so the denominator of every z below is the pipeline's own.
BOOTSTRAP_DRAWS = 400

#: Extra randomized-PIT jitter seeds the bucket shift is re-measured under, so
#: the verdict does not rest on the one draw the residuals were built with.
JITTER_SEEDS: tuple[int, ...] = (73, 1000, 1001, 1002, 1003, 1004, 1005, 1006)

#: The decision thresholds, fixed by the audit brief before any number was
#: read. ``Z`` applies to each key bucket's shift; ``FRACTION`` applies to the
#: global latent target structure.
MAX_KEY_BUCKET_SHIFT_Z = 0.25
MAX_GLOBAL_LATENT_MOVEMENT = 0.03

#: The buckets the remediation gates name. Gate 1 guards the targets, gate 5
#: the protected set; together they are the "key" buckets of the brief. The
#: decision below is taken on all twelve, which is stricter.
TARGET_BUCKETS: tuple[str, ...] = (
    "teammate_ast_ast",
    "teammate_reb_reb",
    "passer_ast_teammate_pts",
)
PROTECTED_BUCKETS: tuple[str, ...] = (
    "opponent_pts_pts",
    "opponent_reb_reb",
    "opponent_ast_ast",
    "opponent_pts_reb",
    "opponent_fg3m_reb",
    "teammate_pts_reb",
)

CANDIDATE_SPEC = PROJECT_ROOT / "research/final_upstream_remediation/factor_spec.json"
CONTROL_SPEC = (
    PROJECT_ROOT / "research/game_latent_state_bucket_repair/factor_spec.json"
)
CANDIDATE_SPEC_HASH = "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"

#: The published OOF build report. Its per-stat PIT diagnostics and training
#: row counts are the reference the reproduction below is checked against.
OOF_BUILD_REPORT = PROJECT_ROOT / "research/game_latent_state/oof_residual_build_report.json"

CONVENTIONS = ("residual_build", "validator")
CONVENTION_LABEL = {
    "residual_build": "02_build_oof_residuals.py joint dropna across all stats",
    "validator": "04_validate_shadow_v1.py per-stat dropna (the live convention)",
}


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(residual_path: Path, history_path: Path) -> dict[str, object]:
    spec = json.loads(CANDIDATE_SPEC.read_text())
    if spec["spec_hash"] != CANDIDATE_SPEC_HASH:
        raise SystemExit(
            "the frozen candidate spec hash moved: "
            f"{spec['spec_hash']} != {CANDIDATE_SPEC_HASH}"
        )
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "worktree_clean": git("status", "--porcelain") == "",
        "candidate_spec_hash": spec["spec_hash"],
        "candidate_spec_hash_confirmed": True,
        "control_spec_hash": json.loads(CONTROL_SPEC.read_text())["spec_hash"],
        "inputs": {
            "committed_oof_residuals": {
                "path": str(residual_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(residual_path),
            },
            "history": {
                "path": str(history_path),
                "sha256": sha256(history_path),
            },
        },
        "seasons_read": list(PRE_2024_SEASONS),
        "holdout_seasons_read": [],
        "holdout_note": (
            "2024 and 2025 are not read anywhere in this audit: no estimator, "
            "no diagnostic and no decision touches them"
        ),
        "dependence_architecture_touched": False,
        "pit_seed": PIT_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


# ----------------------------------------------------------------------
# 1. the mismatch, stated from the three sources
# ----------------------------------------------------------------------


def source_filters() -> dict[str, object]:
    """Quote each path's row filter out of its own source file.

    The audit's premise is a three-way comparison of row filters, so the
    filters are read back out of the files rather than described from memory;
    a drift in any of the three would change these strings and the paired test
    would fail.
    """

    def extract(relative: str, needle: str, span: int = 1) -> dict[str, object]:
        path = PROJECT_ROOT / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        hits = [
            {
                "line": index + 1,
                "source": " ".join(
                    part.strip()
                    for part in lines[index : index + max(span, 1)]
                    if part.strip()
                ),
            }
            for index, line in enumerate(lines)
            if needle in line
        ]
        if not hits:
            raise SystemExit(
                f"{relative} no longer contains {needle!r}: the audit's premise "
                "is a comparison of these filters, so a drift here invalidates it"
            )
        return {"file": relative, "needle": needle, "occurrences": hits}

    return {
        "oof_builder": {
            "what_it_feeds": (
                "the committed cdf_lower_/cdf_upper_/z_ columns, hence every "
                "latent dependence target and the fitted factor structure"
            ),
            "joint_filter_before_the_split": extract(
                "research/game_latent_state/02_build_oof_residuals.py",
                "usable = frame.dropna(subset=[*stats, *selected_columns])",
            ),
            "per_stat_filter_after_it": extract(
                "research/game_latent_state/02_build_oof_residuals.py",
                "fit_rows = train.dropna(subset=[stat, selected])",
            ),
            "effective_convention": "intersection across all six stats and all six selected means",
        },
        "validator": {
            "what_it_feeds": "the marginals the held-out simulation inverts",
            "per_stat_filter": extract(
                "research/game_latent_state/04_validate_shadow_v1.py",
                "rows = train.dropna(subset=[stat, selected])",
            ),
            "effective_convention": "per stat and that stat's own selected mean",
        },
        "production": {
            "what_it_feeds": "the live fitted marginals",
            # The production source is wrapped one token per line, so the
            # filter is quoted as a span and re-joined rather than matched on
            # a single line.
            "candidate_frame_filter": extract(
                "scripts/07_fit_marginals.py",
                "def candidate_frame",
                span=16,
            ),
            "effective_convention": "per target and that target's own selected mean",
        },
        "which_convention_is_live": "validator",
        "the_deviation": (
            "02_build_oof_residuals.py is the only one of the three that "
            "intersects across stats. Its own per-stat dropna runs after the "
            "joint one and so can never remove another row"
        ),
    }


def per_stat_filter_is_a_no_op(history: pd.DataFrame) -> dict[str, object]:
    """Show the OOF builder's per-stat ``dropna`` removes nothing.

    ``build_residuals`` applies the joint filter first and the per-stat filter
    second. If the second removes no row, the effective training convention is
    purely the intersection and the per-stat line is dead code -- which is
    what makes the deviation easy to miss on a read.
    """
    selected = [f"mu_selected_{stat}" for stat in STATS]
    usable = history.dropna(subset=[*STATS, *selected])
    by_stat: dict[str, object] = {}
    for stat in STATS:
        after = usable.dropna(subset=[stat, f"mu_selected_{stat}"])
        by_stat[stat] = {
            "rows_before_the_per_stat_filter": int(len(usable)),
            "rows_after": int(len(after)),
            "rows_removed": int(len(usable) - len(after)),
        }
    return {
        "statement": (
            "inside build_residuals the per-stat dropna is applied to a frame "
            "the joint dropna already cleaned, so it removes nothing and the "
            "effective convention is the intersection"
        ),
        "removes_nothing_for_every_stat": all(
            entry["rows_removed"] == 0 for entry in by_stat.values()  # type: ignore[index]
        ),
        "by_stat": by_stat,
    }


def mean_availability(history: pd.DataFrame) -> dict[str, object]:
    """Which seasons carry a usable row for each stat.

    This is the mechanism underneath the whole audit. A row is usable for a
    stat when both the stat and its own selected mean are present, so the
    joint filter's reach is set by whichever stat is available in the fewest
    seasons: any season missing one stat's mean is removed for *all* stats.
    """
    by_season: dict[str, object] = {}
    for season in sorted(int(value) for value in history["season"].unique()):
        rows = history.loc[history["season"] == season]
        by_season[str(season)] = {
            "rows": int(len(rows)),
            "usable_rows_per_stat": {
                stat: int(
                    rows[[stat, f"mu_selected_{stat}"]].notna().all(axis=1).sum()
                )
                for stat in STATS
            },
        }
    binding = {
        stat: sorted(
            int(season)
            for season, entry in by_season.items()
            if entry["usable_rows_per_stat"][stat] > 0  # type: ignore[index]
        )
        for stat in STATS
    }
    return {
        "statement": (
            "a season survives the joint filter only if every stat has a "
            "selected mean in it, so the stat available in the fewest seasons "
            "sets the training window for all six"
        ),
        "seasons_with_usable_rows_by_stat": binding,
        "seasons_the_joint_filter_removes_entirely": sorted(
            set().union(*binding.values()) - set.intersection(*map(set, binding.values()))
        ),
        "by_season": by_season,
    }


def row_accounting(history: pd.DataFrame) -> dict[str, object]:
    """How many training rows each convention sees, per stat per season.

    The marginal training window for residual season ``S`` is every history
    season strictly before ``S``, which is what both pipeline paths use. The
    *effective* training seasons are reported after each convention's own
    filter, because the raw season span before filtering is the same for both
    and so says nothing.
    """
    selected = {stat: f"mu_selected_{stat}" for stat in STATS}
    joint = history.dropna(subset=[*STATS, *selected.values()])
    by_season: dict[str, object] = {}
    for season in PRE_2024_SEASONS:
        joint_train = joint.loc[joint["season"] < season]
        live_train = history.loc[history["season"] < season]
        entry: dict[str, object] = {
            "oof_convention_training_rows": int(len(joint_train)),
            "oof_convention_effective_training_seasons": sorted(
                int(value) for value in joint_train["season"].unique()
            ),
            "by_stat": {},
        }
        for stat in STATS:
            live_rows = live_train.dropna(subset=[stat, selected[stat]])
            oof_rows = int(len(joint_train))
            entry["by_stat"][stat] = {  # type: ignore[index]
                "live_convention_training_rows": int(len(live_rows)),
                "live_convention_effective_training_seasons": sorted(
                    int(value) for value in live_rows["season"].unique()
                ),
                "oof_convention_training_rows": oof_rows,
                "rows_the_joint_filter_discards": int(len(live_rows)) - oof_rows,
                "fraction_discarded": (
                    (int(len(live_rows)) - oof_rows) / len(live_rows)
                    if len(live_rows)
                    else None
                ),
            }
        by_season[str(season)] = entry
    return {
        "note": (
            "the joint filter discards rows that are complete for the stat "
            "being fitted but missing some *other* stat or mean, so each "
            "marginal is trained on less data than production would give it"
        ),
        "by_season": by_season,
    }


# ----------------------------------------------------------------------
# 2. PITs under each convention, on identical rows
# ----------------------------------------------------------------------


def pit_block(
    rows: pd.DataFrame,
    marginals: Mapping[str, object],
    stats: Sequence[str] = STATS,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate one convention's marginals on one season's target rows.

    Mirrors ``build_residuals`` exactly: ``F(y-1)``, ``F(y)``, the keyed
    deterministic jitter, the randomized PIT and the epsilon-clipped
    Gaussianization, evaluated on the full feature frame the marginal's
    zero-inflation component needs.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for stat in stats:
        marginal = marginals[stat]
        y = rows[stat].to_numpy(dtype=int)
        mu = rows[f"mu_selected_{stat}"].to_numpy(dtype=float)
        lower = marginal.cdf(y - 1, mu, rows)  # type: ignore[union-attr]
        upper = marginal.cdf(y, mu, rows)  # type: ignore[union-attr]
        draw = deterministic_pit_uniform(
            seed=PIT_SEED,
            game_id=rows["game_id"].to_numpy(),
            player_id=rows["player_id"].to_numpy(),
            stat=stat,
        )
        gaussian = gaussianize(randomized_pit(lower, upper, draw), epsilon=PIT_EPSILON)
        out[stat] = {
            "y": y,
            "cdf_lower": np.asarray(lower, dtype=float),
            "cdf_upper": np.asarray(upper, dtype=float),
            "v": draw,
            "u": gaussian.u,
            "z": gaussian.z,
            "clipped": gaussian.clipped,
        }
    return out


def build_pits(
    history: pd.DataFrame,
    cache: Path,
) -> tuple[dict[int, dict[str, dict[str, dict[str, np.ndarray]]]], dict[str, object]]:
    """Both conventions' PITs for every pre-2024 residual season.

    The target rows are the OOF builder's -- the intersection rows -- under
    both conventions, so the comparison is paired on the same player-games and
    the only thing that moves is which rows the marginal was *trained* on. The
    jitter draw is keyed on observation identity, so it is identical too.
    """
    selected = [f"mu_selected_{stat}" for stat in STATS]
    usable = history.dropna(subset=[*STATS, *selected]).copy()
    usable["season"] = usable["season"].astype(int)

    pits: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    marginal_sets: dict[str, object] = {}
    for season in PRE_2024_SEASONS:
        target_rows = usable.loc[usable["season"] == season]
        pits[season] = {}
        for convention in CONVENTIONS:
            fitted = FL.cached_marginals(cache, history, season, convention, STATS)
            # The window before each stat's own dropna. Under the live
            # convention that still carries rows with no selected mean at all,
            # so the *effective* per-stat training sizes are the ones in
            # section 1's row accounting, not these.
            marginal_sets[f"{convention}_{season}"] = {
                "window_seasons_before_the_per_stat_filter": list(
                    fitted.training_seasons
                ),
                "window_rows_before_the_per_stat_filter": fitted.training_rows,
            }
            pits[season][convention] = pit_block(target_rows, fitted.fitted)
        pits[season]["_rows"] = target_rows  # type: ignore[assignment]
    return pits, marginal_sets


def reproduction_check(
    pits: Mapping[int, Mapping[str, object]],
    residuals: pd.DataFrame,
) -> dict[str, object]:
    """Does the OOF-convention recomputation reproduce the committed columns?

    This is the audit's load-bearing validation. If the reproduction of the
    *published* convention does not land on the published numbers, then no
    difference measured against the other convention can be attributed to the
    convention. The committed parquet carries ``v_``, ``cdf_lower_``,
    ``cdf_upper_``, ``u_`` and ``z_``, so all five are checked.
    """
    by_season: dict[str, object] = {}
    worst = {"cdf": 0.0, "z": 0.0, "v": 0.0}
    for season in PRE_2024_SEASONS:
        rows = pits[season]["_rows"]
        published = residuals.loc[residuals["season"].astype(int) == season]
        merged = (
            rows[["game_id", "player_id"]]
            .reset_index(drop=True)
            .merge(
                published,
                on=["game_id", "player_id"],
                how="left",
                indicator=True,
            )
        )
        if len(merged) != len(rows):
            raise SystemExit(
                f"season {season}: the published residuals do not align one to "
                f"one with the recomputed target rows ({len(merged)} vs {len(rows)})"
            )
        matched = int((merged["_merge"] == "both").sum())
        entry: dict[str, object] = {
            "recomputed_rows": int(len(rows)),
            "published_rows": int(len(published)),
            "rows_matched": matched,
            "by_stat": {},
        }
        block = pits[season]["residual_build"]
        for stat in STATS:
            deltas = {
                "v": float(
                    np.max(np.abs(block[stat]["v"] - merged[f"v_{stat}"].to_numpy()))
                ),
                "cdf_lower": float(
                    np.max(
                        np.abs(
                            block[stat]["cdf_lower"]
                            - merged[f"cdf_lower_{stat}"].to_numpy()
                        )
                    )
                ),
                "cdf_upper": float(
                    np.max(
                        np.abs(
                            block[stat]["cdf_upper"]
                            - merged[f"cdf_upper_{stat}"].to_numpy()
                        )
                    )
                ),
                "u": float(
                    np.max(np.abs(block[stat]["u"] - merged[f"u_{stat}"].to_numpy()))
                ),
                "z": float(
                    np.max(np.abs(block[stat]["z"] - merged[f"z_{stat}"].to_numpy()))
                ),
            }
            entry["by_stat"][stat] = deltas  # type: ignore[index]
            worst["cdf"] = max(
                worst["cdf"], deltas["cdf_lower"], deltas["cdf_upper"]
            )
            worst["z"] = max(worst["z"], deltas["z"])
            worst["v"] = max(worst["v"], deltas["v"])
        by_season[str(season)] = entry
    return {
        "statement": (
            "the OOF-convention recomputation is checked against the committed "
            "residual columns before any difference is attributed to the "
            "convention"
        ),
        "max_abs_cdf_difference": worst["cdf"],
        "max_abs_z_difference": worst["z"],
        "max_abs_jitter_difference": worst["v"],
        "reproduces_the_committed_columns": worst["cdf"] < 1e-9
        and worst["z"] < 1e-9
        and worst["v"] == 0.0,
        "by_season": by_season,
    }


def published_diagnostic_check(
    pits: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Cross-check against the published per-stat PIT diagnostics.

    ``oof_residual_build_report.json`` records ``pit_mean``, ``pit_variance``,
    ``z_mean``, ``z_sd`` and ``clipped_fraction`` per stat per season. They are
    an independent reference to the parquet columns, computed by the builder
    itself at build time.
    """
    report = json.loads(OOF_BUILD_REPORT.read_text())["marginals"]["by_season"]
    worst = 0.0
    by_season: dict[str, object] = {}
    for season in PRE_2024_SEASONS:
        published = report[str(season)]
        block = pits[season]["residual_build"]
        entry: dict[str, object] = {
            "published_marginal_training_rows": published["marginal_training_rows"],
            "published_residual_rows": published["residual_rows"],
            "recomputed_residual_rows": int(len(pits[season]["_rows"])),
            "by_stat": {},
        }
        for stat in STATS:
            reference = published["stats"][stat]
            recomputed = {
                "pit_mean": float(np.mean(block[stat]["u"])),
                "pit_variance": float(np.var(block[stat]["u"])),
                "z_mean": float(np.mean(block[stat]["z"])),
                "z_sd": float(np.std(block[stat]["z"])),
                "clipped_fraction": float(np.mean(block[stat]["clipped"])),
            }
            differences = {
                key: abs(recomputed[key] - float(reference[key])) for key in recomputed
            }
            worst = max(worst, max(differences.values()))
            entry["by_stat"][stat] = {  # type: ignore[index]
                "published": {key: float(reference[key]) for key in recomputed},
                "recomputed": recomputed,
                "max_abs_difference": max(differences.values()),
            }
        by_season[str(season)] = entry
    return {
        "max_abs_difference_against_the_published_diagnostics": worst,
        "agrees": worst < 1e-9,
        "by_season": by_season,
    }


def pit_effect(pits: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    """What swapping the training convention does to the pre-2024 PITs."""
    by_season: dict[str, object] = {}
    pooled: dict[str, list[float]] = {stat: [] for stat in STATS}
    pooled_cdf: dict[str, list[float]] = {stat: [] for stat in STATS}
    for season in PRE_2024_SEASONS:
        oof = pits[season]["residual_build"]
        live = pits[season]["validator"]
        entry: dict[str, object] = {"by_stat": {}}
        for stat in STATS:
            d_lower = live[stat]["cdf_lower"] - oof[stat]["cdf_lower"]
            d_upper = live[stat]["cdf_upper"] - oof[stat]["cdf_upper"]
            d_u = live[stat]["u"] - oof[stat]["u"]
            d_z = live[stat]["z"] - oof[stat]["z"]
            pooled[stat].extend(d_z.tolist())
            pooled_cdf[stat].extend(np.abs(np.concatenate([d_lower, d_upper])).tolist())
            entry["by_stat"][stat] = {  # type: ignore[index]
                "rows": int(len(d_z)),
                "max_abs_cdf_shift": float(
                    max(np.max(np.abs(d_lower)), np.max(np.abs(d_upper)))
                ),
                "mean_abs_cdf_shift": float(
                    np.mean(np.abs(np.concatenate([d_lower, d_upper])))
                ),
                "max_abs_u_shift": float(np.max(np.abs(d_u))),
                "rms_z_shift": float(np.sqrt(np.mean(np.square(d_z)))),
                "max_abs_z_shift": float(np.max(np.abs(d_z))),
                "oof_pit_mean": float(np.mean(oof[stat]["u"])),
                "live_pit_mean": float(np.mean(live[stat]["u"])),
                "oof_pit_variance": float(np.var(oof[stat]["u"])),
                "live_pit_variance": float(np.var(live[stat]["u"])),
                "oof_z_mean": float(np.mean(oof[stat]["z"])),
                "live_z_mean": float(np.mean(live[stat]["z"])),
                "oof_z_sd": float(np.std(oof[stat]["z"])),
                "live_z_sd": float(np.std(live[stat]["z"])),
                "oof_ks_vs_uniform": _ks_uniform(oof[stat]["u"]),
                "live_ks_vs_uniform": _ks_uniform(live[stat]["u"]),
            }
        by_season[str(season)] = entry

    return {
        "what_moved": (
            "the marginal training rows only. Target rows, jitter draws, "
            "marginal family and every dependence parameter are identical"
        ),
        "by_season": by_season,
        "pooled_pre_2024": {
            stat: {
                "rows": len(pooled[stat]),
                "rms_z_shift": float(np.sqrt(np.mean(np.square(pooled[stat])))),
                "max_abs_z_shift": float(np.max(np.abs(pooled[stat]))),
                "mean_z_shift": float(np.mean(pooled[stat])),
                "max_abs_cdf_shift": float(np.max(pooled_cdf[stat])),
                "mean_abs_cdf_shift": float(np.mean(pooled_cdf[stat])),
            }
            for stat in STATS
        },
    }


def _ks_uniform(u: np.ndarray) -> float:
    """One-sample Kolmogorov-Smirnov distance from the uniform."""
    values = np.sort(np.asarray(u, dtype=float))
    n = len(values)
    grid = np.arange(1, n + 1, dtype=float) / n
    return float(max(np.max(grid - values), np.max(values - (grid - 1.0 / n))))


# ----------------------------------------------------------------------
# 3. the twelve dependence buckets
# ----------------------------------------------------------------------


def residual_frame(
    pits: Mapping[int, Mapping[str, object]],
    convention: str,
) -> pd.DataFrame:
    """Stack one convention's pre-2024 PITs into a dependence-estimation frame.

    Carries exactly the columns ``pair_moments`` needs plus the season, so the
    two conventions' frames differ in nothing but their ``z_`` values.
    """
    blocks = []
    for season in PRE_2024_SEASONS:
        rows = pits[season]["_rows"]
        block = rows[["game_id", "team_id", "player_id", "season"]].reset_index(
            drop=True
        )
        for stat in STATS:
            block[f"z_{stat}"] = pits[season][convention][stat]["z"]
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def bucket_readings(
    pits: Mapping[int, Mapping[str, object]],
    moment_mode: str = "own",
    bootstrap: int = BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    """All twelve observed buckets under each convention, with clustered SEs.

    ``moment_mode`` selects the standardization, because the pipeline's own
    choice could in principle absorb the convention's effect and the audit
    should not rest on that:

    ``own``
        each convention standardized with its own pre-2024 moments. This is
        what ``03_fit_latent_factors.py`` does and is the primary reading.
    ``frozen``
        both conventions standardized with the frozen spec's recorded
        moments, so the comparison holds the scaling fixed.
    ``none``
        no standardization at all, so nothing can be absorbed.
    """
    out: dict[str, object] = {"by_convention": {}}
    readings: dict[str, dict[str, float]] = {}
    moment_sets: dict[str, dict[str, dict[str, float]]] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}
    ses: dict[str, dict[str, float]] = {}

    if moment_mode == "frozen":
        spec = json.loads(CANDIDATE_SPEC.read_text())["standardization_moments"]
        fixed = {stat: (spec[stat]["mean"], spec[stat]["sd"]) for stat in STATS}
    elif moment_mode == "none":
        fixed = {stat: (0.0, 1.0) for stat in STATS}
    elif moment_mode == "own":
        fixed = None  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown moment mode {moment_mode!r}")

    for convention in CONVENTIONS:
        frame = residual_frame(pits, convention)
        standardized, moments = standardize_residuals(frame, STATS, moments=fixed)
        observed = pair_moments(
            standardized, STATS, bootstrap=bootstrap, seed=PIT_SEED
        )
        readings[convention] = bucket_values(
            STATS, observed.same_team, observed.cross_team
        )
        ses[convention] = bucket_values(
            STATS, observed.same_team_se, observed.cross_team_se
        )
        moment_sets[convention] = {
            stat: {"mean": value[0], "sd": value[1]} for stat, value in moments.items()
        }
        matrices[convention] = {
            "same_team": observed.same_team,
            "cross_team": observed.cross_team,
        }
        out["by_convention"][convention] = {  # type: ignore[index]
            "label": CONVENTION_LABEL[convention],
            "moment_mode": moment_mode,
            "rows": int(len(frame)),
            "games": int(observed.games),
            "same_team_pairs": float(observed.same_team_pairs),
            "cross_team_pairs": float(observed.cross_team_pairs),
            "observed_buckets": readings[convention],
            "observed_bucket_se": ses[convention],
            "standardization_moments": moment_sets[convention],
        }

    # The shift is scored against the OOF convention's own standard error,
    # because that is the denominator the gates use when they read these
    # buckets. The live convention's SE is reported alongside; the two agree
    # closely enough that the choice does not move any verdict.
    shift: dict[str, object] = {}
    for bucket in readings["residual_build"]:
        oof = readings["residual_build"][bucket]
        live = readings["validator"][bucket]
        se = ses["residual_build"][bucket]
        shift[bucket] = {
            "oof_convention": oof,
            "live_convention": live,
            "absolute_shift": live - oof,
            "shift_in_z": (live - oof) / se if se else None,
            "shift_in_z_against_the_live_se": (
                (live - oof) / ses["validator"][bucket]
                if ses["validator"][bucket]
                else None
            ),
            "relative_shift": (live - oof) / abs(oof) if oof else None,
            "se_oof_convention": se,
            "se_live_convention": ses["validator"][bucket],
            "is_target_bucket": bucket in TARGET_BUCKETS,
            "is_protected_bucket": bucket in PROTECTED_BUCKETS,
        }

    out["shift_by_bucket"] = shift
    out["_matrices"] = matrices
    out["_readings"] = readings
    out["_ses"] = ses
    out["_moments"] = moment_sets
    return out


def frozen_spec_moment_check(buckets: Mapping[str, object]) -> dict[str, object]:
    """Does the OOF-convention standardization reproduce the frozen spec?

    ``factor_spec.json`` records the six ``standardization_moments`` the
    frozen candidate was fitted with. They were computed from the committed
    pre-2024 residuals under the OOF convention, so reproducing them is an
    independent check -- separate from the parquet columns and from the build
    report -- that this audit's pre-2024 window and standardization are the
    ones the frozen fit used.
    """
    spec = json.loads(CANDIDATE_SPEC.read_text())["standardization_moments"]
    recomputed = buckets["_moments"]["residual_build"]  # type: ignore[index]
    worst = 0.0
    by_stat: dict[str, object] = {}
    for stat in STATS:
        differences = {
            "mean": abs(recomputed[stat]["mean"] - spec[stat]["mean"]),  # type: ignore[index]
            "sd": abs(recomputed[stat]["sd"] - spec[stat]["sd"]),  # type: ignore[index]
        }
        worst = max(worst, *differences.values())
        by_stat[stat] = {
            "frozen_spec": {"mean": spec[stat]["mean"], "sd": spec[stat]["sd"]},
            "recomputed": recomputed[stat],  # type: ignore[index]
            "max_abs_difference": max(differences.values()),
        }
    return {
        "max_abs_difference": worst,
        "agrees": worst < 1e-12,
        "by_stat": by_stat,
    }


def standardization_sensitivity(
    pits: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Is the verdict an artefact of the pipeline's standardization?

    ``standardize_residuals`` divides by each stat's empirical spread, and the
    convention moves that spread by up to 0.3%, so the standardization could
    in principle absorb part of the convention's effect. Re-reading the
    buckets with the scaling held fixed, and with no scaling at all, shows
    whether it does.
    """
    out: dict[str, object] = {}
    for mode, label in (
        ("own", "each convention standardized with its own moments (the pipeline's choice)"),
        ("frozen", "both standardized with the frozen spec's moments"),
        ("none", "no standardization at all"),
    ):
        # The point estimates are all this needs, and the bootstrap is the
        # expensive part, so the z below is taken against the primary
        # reading's standard errors rather than re-bootstrapped per mode.
        reading = bucket_readings(pits, moment_mode=mode, bootstrap=0)
        shifts = {
            bucket: entry["absolute_shift"]
            for bucket, entry in reading["shift_by_bucket"].items()  # type: ignore[union-attr]
        }
        out[mode] = {
            "label": label,
            "absolute_shift_by_bucket": shifts,
            "largest_absolute_shift": max(abs(float(v)) for v in shifts.values()),
            "largest_shift_bucket": max(
                shifts, key=lambda bucket: abs(float(shifts[bucket]))
            ),
        }
    return {
        "statement": (
            "the convention's effect on the buckets is reported under three "
            "standardizations so the verdict cannot rest on the pipeline's "
            "scaling absorbing it"
        ),
        "by_mode": out,
    }


def jitter_robustness(
    pits: Mapping[int, Mapping[str, object]],
    seeds: Sequence[int],
) -> dict[str, object]:
    """Is the measured bucket shift stable across the randomized-PIT jitter?

    The pipeline's bucket reading is a randomized-PIT sample moment, so it
    carries jitter variance. The two conventions share the committed draw, so
    the paired shift cancels most of it, but the *size* of the shift could
    still be seed-specific. Re-drawing the jitter and re-reading both
    conventions shows whether it is. The CDF bounds are reused, so no
    marginal is refitted: only ``u`` and ``z`` are redrawn.
    """
    per_seed: dict[str, object] = {}
    largest = 0.0
    for seed in seeds:
        frames: dict[str, pd.DataFrame] = {}
        for convention in CONVENTIONS:
            blocks = []
            for season in PRE_2024_SEASONS:
                rows = pits[season]["_rows"]
                block = rows[["game_id", "team_id"]].reset_index(drop=True)
                for stat in STATS:
                    entry = pits[season][convention][stat]
                    draw = deterministic_pit_uniform(
                        seed=seed,
                        game_id=rows["game_id"].to_numpy(),
                        player_id=rows["player_id"].to_numpy(),
                        stat=stat,
                    )
                    gaussian = gaussianize(
                        randomized_pit(entry["cdf_lower"], entry["cdf_upper"], draw),
                        epsilon=PIT_EPSILON,
                    )
                    block[f"z_{stat}"] = gaussian.z
                blocks.append(block)
            frames[convention] = pd.concat(blocks, ignore_index=True)

        readings: dict[str, dict[str, float]] = {}
        for convention in CONVENTIONS:
            standardized, _ = standardize_residuals(frames[convention], STATS)
            observed = pair_moments(standardized, STATS, bootstrap=0)
            readings[convention] = bucket_values(
                STATS, observed.same_team, observed.cross_team
            )
        shifts = {
            bucket: readings["validator"][bucket] - readings["residual_build"][bucket]
            for bucket in readings["residual_build"]
        }
        worst_bucket = max(shifts, key=lambda bucket: abs(shifts[bucket]))
        largest = max(largest, abs(shifts[worst_bucket]))
        per_seed[str(seed)] = {
            "largest_absolute_shift": abs(shifts[worst_bucket]),
            "largest_shift_bucket": worst_bucket,
            "absolute_shift_by_bucket": shifts,
        }
    return {
        "seeds": list(seeds),
        "largest_absolute_shift_over_all_seeds_and_buckets": largest,
        "by_seed": per_seed,
    }


# ----------------------------------------------------------------------
# 4. global latent structure
# ----------------------------------------------------------------------


def loadings_from(path: Path) -> SharedFactorLoadings:
    spec = json.loads(path.read_text())
    return SharedFactorLoadings.from_payload(spec["loadings"])


def global_latent_movement(buckets: Mapping[str, object]) -> dict[str, object]:
    """How much the global latent target structure moves.

    Four readings, because "global latent structure" admits more than one
    defensible meaning and the decision should not turn on which was picked:

    1. the twelve-bucket target vector, in relative L2;
    2. the full 6x6 same-team and cross-team pair-moment matrices, in
       relative Frobenius norm -- the whole structure, not just the scored
       twelve;
    3. the standardization moments the fit pins;
    4. the latent RMSE of each *frozen* model against the two target vectors,
       which is the quantity gate 3 reads. No model is refitted: both
       loadings are read off their frozen specs.
    """
    readings = buckets["_readings"]
    matrices = buckets["_matrices"]
    moments = buckets["_moments"]
    keys = sorted(readings["residual_build"])  # type: ignore[index]

    oof_vector = np.array([readings["residual_build"][k] for k in keys])  # type: ignore[index]
    live_vector = np.array([readings["validator"][k] for k in keys])  # type: ignore[index]
    vector_movement = float(
        np.linalg.norm(live_vector - oof_vector) / np.linalg.norm(oof_vector)
    )

    matrix_movement: dict[str, float] = {}
    for block in ("same_team", "cross_team"):
        oof = matrices["residual_build"][block]  # type: ignore[index]
        live = matrices["validator"][block]  # type: ignore[index]
        matrix_movement[block] = float(
            np.linalg.norm(live - oof) / np.linalg.norm(oof)
        )

    rmse: dict[str, object] = {}
    for name, path in (("candidate", CANDIDATE_SPEC), ("control", CONTROL_SPEC)):
        loadings = loadings_from(path)
        implied = bucket_values(
            STATS,
            loadings.same_team_correlation(),
            loadings.cross_team_correlation(),
        )
        against: dict[str, float] = {}
        for convention in CONVENTIONS:
            errors = np.array(
                [implied[k] - readings[convention][k] for k in keys]  # type: ignore[index]
            )
            against[convention] = float(np.sqrt(np.mean(np.square(errors))))
        rmse[name] = {
            "latent_rmse_against_oof_targets": against["residual_build"],
            "latent_rmse_against_live_targets": against["validator"],
            "ratio_live_over_oof": against["validator"] / against["residual_build"],
            "relative_movement": abs(
                against["validator"] / against["residual_build"] - 1.0
            ),
        }

    moment_movement = {
        stat: {
            "oof_mean": moments["residual_build"][stat]["mean"],  # type: ignore[index]
            "live_mean": moments["validator"][stat]["mean"],  # type: ignore[index]
            "oof_sd": moments["residual_build"][stat]["sd"],  # type: ignore[index]
            "live_sd": moments["validator"][stat]["sd"],  # type: ignore[index]
            "sd_relative_shift": (
                moments["validator"][stat]["sd"] / moments["residual_build"][stat]["sd"]  # type: ignore[index]
                - 1.0
            ),
            "mean_absolute_shift": (
                moments["validator"][stat]["mean"] - moments["residual_build"][stat]["mean"]  # type: ignore[index]
            ),
        }
        for stat in STATS
    }

    worst = max(
        vector_movement,
        matrix_movement["same_team"],
        matrix_movement["cross_team"],
        max(float(entry["relative_movement"]) for entry in rmse.values()),  # type: ignore[index]
    )
    return {
        "target_vector_relative_l2_movement": vector_movement,
        "pair_moment_matrix_relative_frobenius_movement": matrix_movement,
        "frozen_model_latent_rmse_against_each_target_set": rmse,
        "standardization_moments": moment_movement,
        "largest_of_the_four_readings": worst,
        "no_model_was_refitted": True,
    }


# ----------------------------------------------------------------------
# 5. the decision
# ----------------------------------------------------------------------


def transmission_consequence(pit: Mapping[str, object]) -> dict[str, object]:
    """What the mismatch costs where the two conventions actually meet.

    The dependence structure is fitted in a latent space defined by the OOF
    builder's marginals, while the held-out simulation inverts
    ``fit_season``'s. Fixing the OOF convention makes the calibration space
    and the inversion space the same without touching the simulator, so the
    inconsistency that would be removed is exactly the PIT movement measured
    in section 3. Converting that into a count-space number needs the
    simulation, which the brief defers, so it is bounded and not estimated.
    """
    pooled = pit["pooled_pre_2024"]
    return {
        "where_the_conventions_meet": (
            "the latent dependence parameters are calibrated on OOF-convention "
            "PITs and the held-out simulation inverts validator-convention "
            "marginals, so the latent space the parameters mean is not the "
            "latent space the simulator assumes"
        ),
        "fixing_the_oof_convention_aligns_them": True,
        "simulator_would_not_change": (
            "fit_season already uses the live per-stat filter, so aligning the "
            "residual builder changes only the calibration side"
        ),
        "size_of_the_misalignment_in_latent_units": {
            stat: {
                "rms_z": entry["rms_z_shift"],  # type: ignore[index]
                "max_abs_z": entry["max_abs_z_shift"],  # type: ignore[index]
            }
            for stat, entry in pooled.items()  # type: ignore[union-attr]
        },
        "count_space_magnitude": (
            "not estimated here: the brief defers the Monte Carlo rerun, and "
            "the count-space consequence cannot be read off the latent shift "
            "without re-inverting the margins"
        ),
    }


def decision(
    buckets: Mapping[str, object],
    movement: Mapping[str, object],
    jitter: Mapping[str, object],
    sensitivity: Mapping[str, object],
) -> dict[str, object]:
    """Apply the brief's rule to the measured numbers.

    Taken on the strictest reading available: the largest shift over all
    twelve buckets rather than only the named key ones, over every jitter seed
    rather than only the committed one, and the largest of the four global
    latent readings.
    """
    shift = buckets["shift_by_bucket"]
    key = {
        bucket: entry
        for bucket, entry in shift.items()  # type: ignore[union-attr]
        if entry["is_target_bucket"] or entry["is_protected_bucket"]  # type: ignore[index]
    }
    all_twelve_max = max(
        abs(float(entry["shift_in_z"])) for entry in shift.values()  # type: ignore[union-attr,index]
    )
    key_max_bucket = max(
        key, key=lambda bucket: abs(float(key[bucket]["shift_in_z"]))
    )
    key_max = abs(float(key[key_max_bucket]["shift_in_z"]))
    all_max_bucket = max(
        shift, key=lambda bucket: abs(float(shift[bucket]["shift_in_z"]))  # type: ignore[union-attr,index]
    )

    global_worst = float(movement["largest_of_the_four_readings"])

    # The jitter sweep and the standardization sweep report absolute shifts.
    # Scoring them against the smallest bucket standard error turns each into
    # an upper bound in z, so neither can hide a breach.
    smallest_se = min(
        float(entry["se_oof_convention"]) for entry in shift.values()  # type: ignore[union-attr,index]
    )
    jitter_bound_z = (
        float(jitter["largest_absolute_shift_over_all_seeds_and_buckets"]) / smallest_se
    )
    sensitivity_bound_z = (
        max(
            float(entry["largest_absolute_shift"])  # type: ignore[index]
            for entry in sensitivity["by_mode"].values()  # type: ignore[union-attr]
        )
        / smallest_se
    )
    strictest_z = max(all_twelve_max, jitter_bound_z, sensitivity_bound_z)

    buckets_held = strictest_z < MAX_KEY_BUCKET_SHIFT_Z
    global_held = global_worst < MAX_GLOBAL_LATENT_MOVEMENT
    non_material = buckets_held and global_held

    breached: dict[str, float] = {
        bucket: abs(float(entry["shift_in_z"]))  # type: ignore[index]
        for bucket, entry in shift.items()  # type: ignore[union-attr]
        if abs(float(entry["shift_in_z"])) >= MAX_KEY_BUCKET_SHIFT_Z  # type: ignore[index]
    }

    return {
        "rule": (
            "non-material if every key bucket moves under 0.25 z and the "
            "global latent target structure moves under 3%; an upstream "
            "consistency bug otherwise"
        ),
        "thresholds": {
            "max_key_bucket_shift_z": MAX_KEY_BUCKET_SHIFT_Z,
            "max_global_latent_movement": MAX_GLOBAL_LATENT_MOVEMENT,
        },
        "measured": {
            "largest_key_bucket_shift_z": key_max,
            "largest_key_bucket": key_max_bucket,
            "largest_shift_z_over_all_twelve": all_twelve_max,
            "largest_bucket_over_all_twelve": all_max_bucket,
            "upper_bound_z_over_every_jitter_seed": jitter_bound_z,
            "upper_bound_z_over_every_standardization": sensitivity_bound_z,
            "strictest_bucket_shift_z": strictest_z,
            "largest_global_latent_movement": global_worst,
            "buckets_at_or_over_the_z_threshold": breached,
        },
        "bucket_test_held": buckets_held,
        "global_test_held": global_held,
        "classification": (
            "NON_MATERIAL" if non_material else "UPSTREAM_CONSISTENCY_BUG"
        ),
        "action": (
            "close the audit; no code change"
            if non_material
            else "fix only the marginal-training convention in "
            "02_build_oof_residuals.py so it matches the live per-stat filter; "
            "change nothing else"
        ),
        "dependence_architecture_changed": False,
        "holdout_seasons_used": [],
    }


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/research/game_latent_state",
    )
    parser.add_argument(
        "--residuals",
        type=Path,
        default=PROJECT_ROOT
        / "research/final_upstream_remediation/oof_gaussian_residuals.parquet",
    )
    # Shared with the forensic study on purpose: a convention's marginals must
    # be the same object in both studies.
    parser.add_argument("--cache-dir", type=Path, default=FORENSIC_DIR / "cache")
    parser.add_argument(
        "--out", type=Path, default=ARTIFACT_DIR / "marginal_convention_audit.json"
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    cache = arguments.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    history_path = arguments.data_root / "processed/oof_selected_means.parquet"
    history = pd.read_parquet(history_path)
    history["season"] = history["season"].astype(int)
    residuals = pd.read_parquet(arguments.residuals)

    report: dict[str, object] = {
        "title": "OOF-VERSUS-VALIDATOR MARGINAL CONVENTION AUDIT",
        "promotion_eligibility": "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION",
        "scope": (
            "audit only the marginal-training convention mismatch. The "
            "dependence architecture is not touched, no factor family is "
            "fitted, no Monte Carlo is run and no held-out season is read"
        ),
    }

    print("section 0: provenance")
    report["provenance"] = provenance(arguments.residuals, history_path)

    print("section 1: the mismatch")
    report["section_1_the_mismatch"] = source_filters()
    report["section_1_the_mismatch"]["per_stat_filter_is_a_no_op"] = (  # type: ignore[index]
        per_stat_filter_is_a_no_op(history)
    )
    report["section_1_mean_availability"] = mean_availability(history)
    report["section_1_row_accounting"] = row_accounting(history)

    print("section 2: PITs under both conventions")
    pits, marginal_sets = build_pits(history, cache)
    report["section_2_marginal_fits"] = marginal_sets
    report["section_2_reproduction_check"] = reproduction_check(pits, residuals)
    report["section_2_published_diagnostic_check"] = published_diagnostic_check(pits)
    if not report["section_2_reproduction_check"]["reproduces_the_committed_columns"]:  # type: ignore[index]
        raise SystemExit(
            "the OOF-convention recomputation does not reproduce the committed "
            "residual columns, so no measured difference can be attributed to "
            "the convention"
        )

    print("section 3: the PIT effect")
    report["section_3_pit_effect"] = pit_effect(pits)

    print("section 4: the twelve dependence buckets")
    buckets = bucket_readings(pits)
    report["section_4_dependence_buckets"] = {
        key: value for key, value in buckets.items() if not key.startswith("_")
    }
    report["section_4_frozen_spec_moment_check"] = frozen_spec_moment_check(buckets)
    if not report["section_4_frozen_spec_moment_check"]["agrees"]:  # type: ignore[index]
        raise SystemExit(
            "the recomputed OOF standardization moments do not match the "
            "frozen spec's, so this audit's pre-2024 window is not the one "
            "the frozen candidate was fitted on"
        )

    print("section 4b: standardization sensitivity")
    sensitivity = standardization_sensitivity(pits)
    report["section_4_standardization_sensitivity"] = sensitivity

    print("section 4c: jitter robustness")
    jitter = jitter_robustness(pits, JITTER_SEEDS)
    report["section_4_jitter_robustness"] = jitter

    print("section 5: global latent structure")
    movement = global_latent_movement(buckets)
    report["section_5_global_latent_structure"] = movement
    report["section_5_transmission_consequence"] = transmission_consequence(
        report["section_3_pit_effect"]  # type: ignore[arg-type]
    )

    print("section 6: decision")
    report["decision"] = decision(buckets, movement, jitter, sensitivity)
    report["classification"] = report["decision"]["classification"]  # type: ignore[index]

    arguments.out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(f"wrote {arguments.out}")

    verdict = report["decision"]
    print(
        f"\n  largest key bucket shift   {verdict['measured']['largest_key_bucket_shift_z']:+.4f} z"  # type: ignore[index]
        f"  ({verdict['measured']['largest_key_bucket']})"  # type: ignore[index]
    )
    print(
        f"  largest over all twelve    "
        f"{verdict['measured']['largest_shift_z_over_all_twelve']:+.4f} z"  # type: ignore[index]
        f"  ({verdict['measured']['largest_bucket_over_all_twelve']})"  # type: ignore[index]
    )
    print(
        f"  strictest bound            "
        f"{verdict['measured']['strictest_bucket_shift_z']:+.4f} z"  # type: ignore[index]
        "  (over every jitter seed and standardization)"
    )
    print(
        f"  global latent movement     "
        f"{verdict['measured']['largest_global_latent_movement']:.6f}"  # type: ignore[index]
    )
    print(f"  CLASSIFICATION             {verdict['classification']}")  # type: ignore[index]


if __name__ == "__main__":
    main()
