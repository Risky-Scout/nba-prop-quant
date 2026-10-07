"""Locks for the OOF-versus-validator marginal convention audit.

SHADOW / RESEARCH ONLY. These tests do not re-run the audit driver, which
refits ZINB marginals over four walk-forward windows. They check three
different things:

1. that the audit's *premise* still holds -- the three pipeline row filters
   are the ones it compares, re-derived from their own source files;
2. that its *instrument* can detect an effect, on synthetic data with a known
   answer, because a "no material effect" verdict is worthless from a dead
   measurement; and
3. that its *published numbers* are internally consistent and that the
   verdict follows from them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "research/marginal_convention_audit"
REPORT_PATH = AUDIT_DIR / "marginal_convention_audit.json"

for path in (
    str(PROJECT_ROOT / "research/count_space_forensic"),
    str(PROJECT_ROOT / "src"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.validation import (  # noqa: E402
    bucket_values,
)

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")

#: Stats whose per-stat usable rows coincide with the intersection across all
#: six, so the convention change cannot move them. Derived in the audit from
#: the selected-mean availability table, and asserted here as a property of
#: the data rather than taken on faith.
UNAFFECTED_STATS = ("reb", "blk")

#: The buckets built only from unaffected stats, which must therefore not move
#: at all.
UNAFFECTED_BUCKETS = ("teammate_reb_reb", "opponent_reb_reb")


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip(f"missing audit artifact: {REPORT_PATH}")
    return json.loads(REPORT_PATH.read_text())


# ----------------------------------------------------------------------
# 1. the premise: the three row filters
# ----------------------------------------------------------------------


def test_the_oof_builder_still_takes_a_joint_dropna_before_splitting() -> None:
    """The deviation under audit, read out of its own source.

    If this line moves or changes shape the audit is measuring something that
    no longer exists, so the premise is pinned rather than described.
    """
    source = (
        PROJECT_ROOT / "research/game_latent_state/02_build_oof_residuals.py"
    ).read_text()
    assert "usable = frame.dropna(subset=[*stats, *selected_columns])" in source
    # And the per-stat filter that follows it, which is the one the audit
    # shows cannot remove anything.
    assert "fit_rows = train.dropna(subset=[stat, selected])" in source
    joint = source.index("usable = frame.dropna(subset=[*stats, *selected_columns])")
    per_stat = source.index("fit_rows = train.dropna(subset=[stat, selected])")
    assert joint < per_stat, "the joint filter must still run first"


def test_the_validator_and_production_paths_both_filter_per_stat() -> None:
    """The live convention, read out of both sources that implement it."""
    validator = (
        PROJECT_ROOT / "research/game_latent_state/04_validate_shadow_v1.py"
    ).read_text()
    assert "rows = train.dropna(subset=[stat, selected])" in validator
    assert "usable = frame.dropna(subset=[*stats" not in validator

    production = (PROJECT_ROOT / "scripts/07_fit_marginals.py").read_text()
    # candidate_frame is wrapped one token per line, so the filter is matched
    # on its collapsed form.
    collapsed = " ".join(production.split())
    assert "def candidate_frame( frame: pd.DataFrame, target: str, )" in collapsed
    assert (
        "return frame[ frame[ mean_col ].notna() & frame[ target ].notna() ].copy()"
        in collapsed
    )


def test_the_per_stat_filter_cannot_remove_a_row_the_joint_filter_left() -> None:
    """The no-op, proved on a frame built to make it fail if it could.

    Every stat has a row the others are missing, so a per-stat filter applied
    first would keep different rows for each stat. Applied after the joint
    filter it keeps the same two rows for all of them.
    """
    rows = {"season": [2019] * (len(STATS) + 2)}
    for stat in STATS:
        rows[stat] = [1.0] * (len(STATS) + 2)
        rows[f"mu_selected_{stat}"] = [1.0] * (len(STATS) + 2)
    frame = pd.DataFrame(rows)
    for index, stat in enumerate(STATS):
        frame.loc[index, stat] = np.nan

    selected = [f"mu_selected_{stat}" for stat in STATS]
    usable = frame.dropna(subset=[*STATS, *selected])
    assert len(usable) == 2

    for stat in STATS:
        after = usable.dropna(subset=[stat, f"mu_selected_{stat}"])
        assert len(after) == len(usable)


def test_the_audit_reports_the_no_op_for_every_stat(report: dict) -> None:
    no_op = report["section_1_the_mismatch"]["per_stat_filter_is_a_no_op"]
    assert no_op["removes_nothing_for_every_stat"] is True
    for stat, entry in no_op["by_stat"].items():
        assert entry["rows_removed"] == 0, stat


# ----------------------------------------------------------------------
# 2. the instrument: can it see an effect at all?
# ----------------------------------------------------------------------


def _synthetic_games(
    rho_same: float,
    games: int = 900,
    players_per_team: int = 8,
    seed: int = 11,
) -> pd.DataFrame:
    """Two teams per game, one shared game factor per team.

    Within a team every player's ``pts`` and ``ast`` share a team-level
    normal with weight ``sqrt(rho_same)``, so the same-team cross-player
    correlation is ``rho_same`` by construction for both the pts/pts and the
    ast/pts bucket, and the cross-team correlation is zero.
    """
    rng = np.random.default_rng(seed)
    weight = np.sqrt(rho_same)
    residual = np.sqrt(1.0 - rho_same)
    blocks = []
    for game in range(games):
        for team in range(2):
            shared = rng.standard_normal()
            n = players_per_team
            block = pd.DataFrame(
                {
                    "game_id": np.full(n, game),
                    "team_id": np.full(n, 2 * game + team),
                }
            )
            for stat in STATS:
                block[f"z_{stat}"] = (
                    weight * shared + residual * rng.standard_normal(n)
                )
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


@pytest.mark.parametrize("rho", [0.0, 0.05, 0.12])
def test_the_bucket_instrument_recovers_a_known_same_team_correlation(
    rho: float,
) -> None:
    """The measurement chain the audit reads the buckets through.

    ``standardize_residuals`` then ``pair_moments`` then ``bucket_values`` is
    exactly the chain ``03_fit_latent_factors.py`` and
    ``latent_dependence_summary`` use. On data with a known same-team
    correlation it has to return that correlation.
    """
    frame = _synthetic_games(rho)
    standardized, _ = standardize_residuals(frame, STATS)
    observed = pair_moments(standardized, STATS, bootstrap=0)
    buckets = bucket_values(STATS, observed.same_team, observed.cross_team)

    assert buckets["teammate_pts_pts"] == pytest.approx(rho, abs=0.02)
    assert buckets["passer_ast_teammate_pts"] == pytest.approx(rho, abs=0.02)
    # The construction shares one factor per team, so cross-team pairs are
    # independent and that bucket must sit at zero.
    assert buckets["opponent_pts_pts"] == pytest.approx(0.0, abs=0.02)


def test_the_bucket_instrument_resolves_a_shift_far_below_the_threshold() -> None:
    """A deliberate shift the audit's threshold would call material.

    The audit concludes that nothing moves as much as 0.25 of a bucket
    standard error. That is only evidence if the instrument would have seen
    such a move. Two universes differing by a known 0.02 of correlation --
    roughly thirteen standard errors at the audit's pre-2024 sample size --
    are resolved far more sharply than the threshold.
    """
    low = _synthetic_games(0.05)
    high = _synthetic_games(0.07)

    readings = []
    for frame in (low, high):
        standardized, _ = standardize_residuals(frame, STATS)
        observed = pair_moments(standardized, STATS, bootstrap=200, seed=73)
        readings.append(
            (
                bucket_values(STATS, observed.same_team, observed.cross_team),
                bucket_values(
                    STATS, observed.same_team_se, observed.cross_team_se
                ),
            )
        )

    shift = (
        readings[1][0]["passer_ast_teammate_pts"]
        - readings[0][0]["passer_ast_teammate_pts"]
    )
    se = readings[0][1]["passer_ast_teammate_pts"]
    assert shift == pytest.approx(0.02, abs=0.01)
    assert abs(shift) / se > 1.0, "a real shift must clear the audit's own z scale"


# ----------------------------------------------------------------------
# 3. the published numbers
# ----------------------------------------------------------------------


def test_the_oof_reproduction_is_exact(report: dict) -> None:
    """The audit's load-bearing validation.

    Every difference the audit attributes to the convention is a difference
    against a recomputation of the published convention, so that
    recomputation has to land on the published numbers exactly.
    """
    check = report["section_2_reproduction_check"]
    assert check["reproduces_the_committed_columns"] is True
    assert check["max_abs_cdf_difference"] == 0.0
    assert check["max_abs_z_difference"] == 0.0
    assert check["max_abs_jitter_difference"] == 0.0


def test_the_reproduction_also_matches_the_published_build_diagnostics(
    report: dict,
) -> None:
    """A second, independent reference to the parquet columns."""
    check = report["section_2_published_diagnostic_check"]
    assert check["agrees"] is True
    assert check["max_abs_difference_against_the_published_diagnostics"] == 0.0


def test_the_pre_2024_window_is_the_one_the_frozen_candidate_was_fitted_on(
    report: dict,
) -> None:
    """The frozen spec's standardization moments, reproduced.

    A third reference, and the one that pins the *window*: these six numbers
    were computed from the committed pre-2024 residuals when the candidate
    was frozen.
    """
    check = report["section_4_frozen_spec_moment_check"]
    assert check["agrees"] is True
    assert check["max_abs_difference"] < 1e-12


def test_the_convention_is_a_no_op_for_the_stats_that_bind_the_intersection(
    report: dict,
) -> None:
    """``reb`` and ``blk`` must not move, and must be shown not to.

    The joint filter's reach is set by whichever stat is usable in the fewest
    seasons. For those stats the per-stat row set already *is* the
    intersection, so their marginals, PITs and buckets are identical under
    both conventions. Exact zeros here are what rule out a plumbing error in
    which one convention silently overwrote the other.
    """
    pooled = report["section_3_pit_effect"]["pooled_pre_2024"]
    for stat in UNAFFECTED_STATS:
        assert pooled[stat]["max_abs_cdf_shift"] == 0.0, stat
        assert pooled[stat]["rms_z_shift"] == 0.0, stat

    affected = [stat for stat in STATS if stat not in UNAFFECTED_STATS]
    for stat in affected:
        assert pooled[stat]["max_abs_cdf_shift"] > 0.0, stat
        assert pooled[stat]["rms_z_shift"] > 0.0, stat


def test_the_buckets_built_only_from_unaffected_stats_do_not_move(
    report: dict,
) -> None:
    shift = report["section_4_dependence_buckets"]["shift_by_bucket"]
    for bucket in UNAFFECTED_BUCKETS:
        assert shift[bucket]["absolute_shift"] == 0.0, bucket
        assert shift[bucket]["shift_in_z"] == 0.0, bucket


def test_every_one_of_the_twelve_buckets_is_scored(report: dict) -> None:
    shift = report["section_4_dependence_buckets"]["shift_by_bucket"]
    assert len(shift) == 12
    named = [
        bucket
        for bucket, entry in shift.items()
        if entry["is_target_bucket"] or entry["is_protected_bucket"]
    ]
    assert len(named) == 9, "three gate-1 targets plus six gate-5 protected buckets"


def test_only_the_training_channel_is_in_scope_and_the_other_is_empty(
    report: dict,
) -> None:
    """The convention has two channels; the data closes the second.

    ``build_residuals`` filters the training rows and the target rows with
    the same joint ``dropna``. Inside the residual seasons every row is
    already complete, so the target-row channel carries nothing, and
    ``pair_moments`` would drop an incomplete row before any bucket saw it
    regardless.
    """
    scope = report["section_1_target_row_scope"]
    assert scope["the_target_row_channel_is_empty"] is True
    assert scope["extra_target_rows_a_per_stat_filter_would_emit"] == 0
    assert scope["rows_surviving_the_joint_filter"] == scope["pre_2024_history_rows"]


def test_the_audit_reads_no_held_out_season(report: dict) -> None:
    """The brief forbids 2024/2025 tuning; the audit reads them not at all."""
    provenance = report["provenance"]
    assert provenance["seasons_read"] == [2020, 2021, 2022, 2023]
    assert provenance["holdout_seasons_read"] == []
    assert provenance["dependence_architecture_touched"] is False
    assert report["decision"]["holdout_seasons_used"] == []
    assert report["decision"]["dependence_architecture_changed"] is False


def test_the_frozen_candidate_spec_hash_is_unmoved(report: dict) -> None:
    assert report["provenance"]["candidate_spec_hash"] == (
        "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
    )
    assert report["provenance"]["candidate_spec_hash_confirmed"] is True


def test_no_model_was_refitted_for_the_global_reading(report: dict) -> None:
    """The global latent reading uses frozen loadings, not a refit."""
    assert report["section_5_global_latent_structure"]["no_model_was_refitted"] is True


def test_the_verdict_follows_from_the_thresholds(report: dict) -> None:
    """The classification is a function of the measured numbers.

    Recomputed from the reported measurements rather than trusted, so a
    threshold edited in one place and not the other fails here.
    """
    decision = report["decision"]
    measured = decision["measured"]
    thresholds = decision["thresholds"]

    assert thresholds["max_key_bucket_shift_z"] == 0.25
    assert thresholds["max_global_latent_movement"] == 0.03

    buckets_held = (
        measured["strictest_bucket_shift_z"] < thresholds["max_key_bucket_shift_z"]
    )
    global_held = (
        measured["largest_global_latent_movement"]
        < thresholds["max_global_latent_movement"]
    )
    assert decision["bucket_test_held"] is buckets_held
    assert decision["global_test_held"] is global_held
    assert decision["classification"] == (
        "NON_MATERIAL" if buckets_held and global_held else "UPSTREAM_CONSISTENCY_BUG"
    )


def test_the_strictest_bound_dominates_every_reported_reading(report: dict) -> None:
    """The decision is taken on the largest of the measurements, not the first."""
    measured = report["decision"]["measured"]
    for key in (
        "largest_shift_z_over_all_twelve",
        "largest_matched_shift_z_over_every_jitter_seed",
        "largest_matched_shift_z_over_every_standardization",
        "largest_matched_shift_z_anywhere",
        "conservative_bound_z_over_every_jitter_seed",
        "conservative_bound_z_over_every_standardization",
    ):
        assert measured["strictest_bucket_shift_z"] >= measured[key], key
    assert measured["largest_matched_shift_z_anywhere"] >= measured[
        "largest_shift_z_over_all_twelve"
    ]
    assert measured["largest_shift_z_over_all_twelve"] >= measured[
        "largest_key_bucket_shift_z"
    ]


def test_the_unstandardized_reading_is_the_one_the_verdict_rests_on(
    report: dict,
) -> None:
    """Standardization absorbs part of the effect, so it must not be relied on.

    The pipeline rescales each stat by its own empirical spread, and the
    convention moves that spread, so the standardized reading understates the
    convention's effect. The audit therefore has to clear its threshold on
    the unstandardized row too.
    """
    modes = report["section_4_standardization_sensitivity"]["by_mode"]
    assert (
        modes["none"]["largest_matched_shift_z"]
        >= modes["own"]["largest_matched_shift_z"]
    ), "if standardization did not absorb anything this premise is wrong"
    threshold = report["decision"]["thresholds"]["max_key_bucket_shift_z"]
    if report["decision"]["classification"] == "NON_MATERIAL":
        assert modes["none"]["largest_matched_shift_z"] < threshold
        assert modes["none"]["conservative_bound_z"] < threshold


def test_the_action_matches_the_classification(report: dict) -> None:
    decision = report["decision"]
    if decision["classification"] == "NON_MATERIAL":
        assert "close" in decision["action"]
        assert decision["measured"]["buckets_at_or_over_the_z_threshold"] == {}
    else:
        assert "02_build_oof_residuals.py" in decision["action"]
        assert decision["measured"]["buckets_at_or_over_the_z_threshold"]


def test_the_jitter_sweep_includes_the_committed_draw(report: dict) -> None:
    """The committed seed must be one of the seeds the sweep reports."""
    jitter = report["section_4_jitter_robustness"]
    assert report["provenance"]["pit_seed"] in jitter["seeds"]
    assert len(jitter["seeds"]) >= 4
    assert len(jitter["by_seed"]) == len(jitter["seeds"])


def test_the_standardization_sensitivity_covers_all_three_modes(
    report: dict,
) -> None:
    """Including the unstandardized reading, which can absorb nothing."""
    modes = report["section_4_standardization_sensitivity"]["by_mode"]
    assert set(modes) == {"own", "frozen", "none"}
    for mode, entry in modes.items():
        assert entry["largest_absolute_shift"] >= 0.0, mode


def test_the_row_accounting_reproduces_the_published_training_sizes(
    report: dict,
) -> None:
    """The OOF convention's training rows match the published build report."""
    published = json.loads(
        (
            PROJECT_ROOT / "research/game_latent_state/oof_residual_build_report.json"
        ).read_text()
    )["marginals"]["by_season"]
    accounting = report["section_1_row_accounting"]["by_season"]
    for season, entry in accounting.items():
        assert entry["oof_convention_training_rows"] == (
            published[season]["marginal_training_rows"]
        ), season
        assert entry["oof_convention_effective_training_seasons"] == (
            published[season]["marginal_training_seasons"]
        ), season


def test_the_joint_filter_discards_a_whole_season_for_four_of_six_stats(
    report: dict,
) -> None:
    """The mechanism, asserted as the shape of the row accounting.

    The intersection is binding through a season in which four stats have a
    selected mean and two do not, so those four lose that season entirely
    while the other two lose nothing.
    """
    availability = report["section_1_mean_availability"]
    removed = availability["seasons_the_joint_filter_removes_entirely"]
    assert removed, "the audit's mechanism requires at least one such season"

    accounting = report["section_1_row_accounting"]["by_season"]
    for season, entry in accounting.items():
        discarding = {
            stat: value["rows_the_joint_filter_discards"]
            for stat, value in entry["by_stat"].items()
        }
        for stat in UNAFFECTED_STATS:
            assert discarding[stat] == 0, (season, stat)
        affected = [stat for stat in STATS if stat not in UNAFFECTED_STATS]
        assert all(discarding[stat] > 0 for stat in affected), season
        # One shared season, so every affected stat loses the same count.
        assert len(set(discarding[stat] for stat in affected)) == 1, season
