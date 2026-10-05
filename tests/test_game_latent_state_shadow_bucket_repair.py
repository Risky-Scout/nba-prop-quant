"""Tests for the latent-state bucket-repair research branch.

SHADOW / RESEARCH ONLY.

The load-bearing property is that accepted shadow V1 remains the control: the
repair code is additive, and :data:`V1_CONTROL_SPEC` must reproduce
:func:`factors.fit_shared_factors` exactly, including against the committed V1
artifact. Everything else here guards the inner-selection contract (no holdout
leakage, deterministic selection, a frozen manifest) and the invariants the
repair is not allowed to break (PSD, same-player block pinning, no
player-pair parameters, no promotion path).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from nba_prop_quant.research.game_latent_state.covariance import (
    GameDimension,
    SharedFactorLoadings,
    build_game_covariance,
    implied_within_player_correlation,
    project_psd_rank,
)
from nba_prop_quant.research.game_latent_state.factors import (
    fit_shared_factors,
    pair_moments,
    soft_threshold,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.repair import (
    EB_FAMILY_BLOCK,
    EB_FAMILY_BLOCK_DIAGONAL,
    EB_FAMILY_GLOBAL,
    SHRINKAGE_EMPIRICAL_BAYES,
    V1_CONTROL_SPEC,
    RepairSpec,
    attenuation_report,
    empirical_bayes_shrink,
    fit_repaired_factors,
    shrink_blocks,
)

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")
REPAIR_DIR = PROJECT / "research" / "game_latent_state_bucket_repair"
V1_DIR = PROJECT / "research" / "game_latent_state"
HOLDOUT_SEASONS = (2024, 2025)


def load_driver(name: str):
    """Import a numeric-prefixed driver by path.

    Registered in ``sys.modules`` before execution because the drivers define
    dataclasses, which cannot resolve their own module otherwise.
    """
    path = REPAIR_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# synthetic residual frame
# ----------------------------------------------------------------------


def synthetic_residuals(
    seasons: tuple[int, ...] = (2020, 2021, 2022, 2023),
    games_per_season: int = 60,
    team_size: int = 8,
    seed: int = 7,
    shared_sd: float = 0.25,
    teammate_stats: tuple[str, ...] = ("reb", "ast"),
    teammate_sd: float = 0.3,
) -> pd.DataFrame:
    """Residuals with a known shared game factor plus a teammate-only factor.

    The teammate factor loads on ``teammate_stats`` for one team at a time, so
    it creates genuine same-team dependence in exactly the two stats the repair
    targets without touching cross-team pairs.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    game_id = 0
    roles = ("starter", "rotation", "bench")
    for season in seasons:
        for _ in range(games_per_season):
            game_id += 1
            game_factor = rng.normal(0.0, 1.0)
            for side, team_id in enumerate((1, 2)):
                team_factor = rng.normal(0.0, 1.0)
                for member in range(team_size):
                    row: dict[str, object] = {
                        "game_id": game_id,
                        "game_date": pd.Timestamp("2020-01-01")
                        + pd.Timedelta(days=game_id),
                        "season": season,
                        "team_id": team_id,
                        "opponent_id": 2 if team_id == 1 else 1,
                        "is_home": 1 - side,
                        "player_id": 1000 * team_id + member,
                        "expected_minutes": 20.0,
                        "role_bucket": roles[member % len(roles)],
                    }
                    for stat in STATS:
                        value = rng.normal(0.0, 1.0) + shared_sd * game_factor
                        if stat in teammate_stats:
                            value += teammate_sd * team_factor
                        row[f"z_{stat}"] = value
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def standardized_synthetic() -> pd.DataFrame:
    frame = synthetic_residuals()
    standardized, _ = standardize_residuals(frame, STATS)
    return standardized


# ----------------------------------------------------------------------
# the accepted V1 fit remains the control
# ----------------------------------------------------------------------


def test_control_spec_reproduces_the_accepted_v1_fit(standardized_synthetic):
    """The repair path with the control spec must be the V1 path exactly.

    This is what makes V1 the benchmark rather than a re-derivation: if this
    drifts, every before/after number on the branch is comparing two different
    models.
    """
    v1 = fit_shared_factors(
        standardized_synthetic,
        STATS,
        k_game=2,
        shrink_z=1.96,
        bootstrap=120,
        seed=73,
        role_column="role_bucket",
    )
    control = fit_repaired_factors(
        standardized_synthetic, STATS, V1_CONTROL_SPEC, bootstrap=120, seed=73
    )

    assert np.array_equal(v1.loadings.game, control.loadings.game)
    assert np.array_equal(v1.loadings.team_contrast, control.loadings.team_contrast)
    assert v1.loadings.role_scale == control.loadings.role_scale
    assert np.array_equal(v1.same_team_shrunk, control.same_team_shrunk)
    assert np.array_equal(v1.cross_team_shrunk, control.cross_team_shrunk)
    assert np.array_equal(
        v1.loadings.same_team_correlation(), control.loadings.same_team_correlation()
    )
    assert np.array_equal(
        v1.loadings.cross_team_correlation(), control.loadings.cross_team_correlation()
    )
    if v1.loadings.competition is None:
        assert control.loadings.competition is None
    else:
        assert np.array_equal(v1.loadings.competition, control.loadings.competition)


def test_control_spec_keeps_the_rank_one_contrast_wire_format(standardized_synthetic):
    """A rank-1 contrast must still serialise as a flat vector.

    The committed V1 artifacts store it that way, so a shape change here would
    silently invalidate them.
    """
    control = fit_repaired_factors(
        standardized_synthetic, STATS, V1_CONTROL_SPEC, bootstrap=60, seed=73
    )
    payload = control.loadings.to_payload()
    assert np.shape(payload["team_contrast_loadings"]) == (len(STATS),)
    assert payload["r_contrast"] == 1
    restored = SharedFactorLoadings.from_payload(payload)
    assert np.array_equal(restored.team_contrast, control.loadings.team_contrast)


@pytest.mark.skipif(
    not (V1_DIR / "factor_spec.json").exists(),
    reason="committed V1 factor spec not present",
)
def test_multi_rank_contrast_round_trips_without_disturbing_the_v1_payload():
    """Raising the contrast rank must not change how V1's payload deserialises."""
    v1_spec = json.loads((V1_DIR / "factor_spec.json").read_text(encoding="utf-8"))
    loadings = SharedFactorLoadings.from_payload(v1_spec["loadings"])
    assert loadings.r_contrast == 1
    assert loadings.team_contrast.ndim == 1

    # The rank-1 Gram and the matrix Gram must agree entry for entry.
    assert np.allclose(
        loadings.contrast_matrix @ loadings.contrast_matrix.T,
        np.outer(loadings.team_contrast, loadings.team_contrast),
        atol=0,
        rtol=0,
    )


# ----------------------------------------------------------------------
# shrinkage estimator correctness
# ----------------------------------------------------------------------


def test_empirical_bayes_shrinks_multiplicatively_not_by_a_fixed_width():
    """The whole point of family B: bias proportional to the estimate.

    A fixed-width soft threshold removes ``z * se`` from every entry, so it
    costs a strong signal almost nothing and a modest signal most of its value.
    The posterior mean removes a constant *fraction* instead, which is why it
    keeps modest-but-real buckets alive.
    """
    estimate = np.diag([0.04, 0.006, 0.0005])
    error = np.full((3, 3), 0.0015)
    shrunk, diagnostics = empirical_bayes_shrink(
        estimate, error, family=EB_FAMILY_GLOBAL
    )

    tau2 = diagnostics["tau2_all"]
    for position in range(3):
        expected = estimate[position, position] * tau2 / (tau2 + 0.0015**2)
        assert shrunk[position, position] == pytest.approx(expected, rel=1e-12)

    # Retained fraction is identical across entries, unlike a soft threshold.
    retained = [
        shrunk[i, i] / estimate[i, i] for i in range(3) if estimate[i, i] != 0
    ]
    assert np.allclose(retained, retained[0], atol=1e-12)

    thresholded = soft_threshold(estimate, error, 1.96)
    strong = thresholded[0, 0] / estimate[0, 0]
    modest = thresholded[1, 1] / estimate[1, 1]
    assert strong > modest + 0.4, "soft threshold should tax the modest entry harder"


def test_empirical_bayes_drives_a_pure_noise_family_to_near_zero():
    """With no signal above the noise floor, ``tau^2`` collapses and so does the fit."""
    rng = np.random.default_rng(3)
    error = np.full((4, 4), 0.002)
    noise = rng.normal(0.0, 0.002, size=(4, 4))
    estimate = 0.5 * (noise + noise.T)
    shrunk, diagnostics = empirical_bayes_shrink(
        estimate, error, family=EB_FAMILY_GLOBAL
    )
    assert diagnostics["tau2_all"] < 1e-5
    assert np.max(np.abs(shrunk)) < np.max(np.abs(estimate))


def test_empirical_bayes_output_is_symmetric_and_preserves_sign():
    rng = np.random.default_rng(11)
    raw = rng.normal(0.0, 0.01, size=(5, 5))
    estimate = 0.5 * (raw + raw.T)
    error = np.full((5, 5), 0.001)
    shrunk, _ = empirical_bayes_shrink(estimate, error, family=EB_FAMILY_BLOCK)
    assert np.allclose(shrunk, shrunk.T, atol=1e-15)
    nonzero = np.abs(shrunk) > 1e-15
    assert np.all(np.sign(shrunk[nonzero]) == np.sign(estimate[nonzero]))


def test_empirical_bayes_pools_only_within_the_declared_family():
    """Diagonal and off-diagonal entries must get their own ``tau``.

    Pooling them lets one dominant off-diagonal entry inflate the prior
    variance for every diagonal entry, which would under-shrink noise.
    """
    estimate = np.array(
        [[0.004, 0.05, 0.0], [0.05, 0.004, 0.0], [0.0, 0.0, 0.004]], dtype=float
    )
    error = np.full((3, 3), 0.002)
    _, split = empirical_bayes_shrink(
        estimate, error, family=EB_FAMILY_BLOCK_DIAGONAL, is_same_team=True
    )
    _, pooled = empirical_bayes_shrink(
        estimate, error, family=EB_FAMILY_BLOCK, is_same_team=True
    )
    assert split["tau2_same_team_diagonal"] < split["tau2_same_team_offdiagonal"]
    assert split["tau2_same_team_diagonal"] < pooled["tau2_same_team"]


def test_empirical_bayes_without_standard_errors_does_not_invent_shrinkage():
    estimate = np.eye(3) * 0.01
    error = np.full((3, 3), np.nan)
    shrunk, diagnostics = empirical_bayes_shrink(estimate, error)
    assert np.array_equal(shrunk, estimate)
    assert diagnostics["tau2_unavailable"] == 1.0


def test_unknown_shrinkage_and_family_are_rejected():
    moments_frame = synthetic_residuals(seasons=(2020,), games_per_season=20)
    standardized, _ = standardize_residuals(moments_frame, STATS)
    moments = pair_moments(standardized, STATS, bootstrap=40, seed=1)
    with pytest.raises(ValueError, match="unknown shrinkage"):
        shrink_blocks(moments, RepairSpec(name="x", family="x", shrinkage="nope"))
    with pytest.raises(ValueError, match="unknown empirical-Bayes family"):
        empirical_bayes_shrink(np.eye(2), np.full((2, 2), 0.1), family="nope")


# ----------------------------------------------------------------------
# target-bucket unit correctness
# ----------------------------------------------------------------------


def test_target_bucket_entries_read_the_same_team_diagonal(standardized_synthetic):
    """``teammate_reb_reb`` is the same-team block's reb/reb entry, nothing else."""
    fit = fit_repaired_factors(
        standardized_synthetic, STATS, V1_CONTROL_SPEC, bootstrap=60, seed=73
    )
    from nba_prop_quant.research.game_latent_state.validation import bucket_values

    buckets = bucket_values(
        STATS,
        fit.loadings.same_team_correlation(),
        fit.loadings.cross_team_correlation(),
    )
    same = fit.loadings.same_team_correlation()
    index = {stat: position for position, stat in enumerate(STATS)}
    assert buckets["teammate_reb_reb"] == pytest.approx(
        same[index["reb"], index["reb"]], rel=1e-15
    )
    assert buckets["teammate_ast_ast"] == pytest.approx(
        same[index["ast"], index["ast"]], rel=1e-15
    )


def test_attenuation_report_splits_shrinkage_from_rank_loss(standardized_synthetic):
    """The decomposition must add up and attribute each loss to one mechanism."""
    moments = pair_moments(standardized_synthetic, STATS, bootstrap=120, seed=73)
    shrunk = soft_threshold(moments.same_team, moments.same_team_se, 1.96)
    fit = fit_repaired_factors(
        standardized_synthetic, STATS, V1_CONTROL_SPEC, bootstrap=120, seed=73
    )
    report = attenuation_report(
        moments,
        shrunk,
        fit.loadings.same_team_correlation(),
        {"teammate_reb_reb": ("reb", "reb"), "teammate_ast_ast": ("ast", "ast")},
        moments.same_team_se,
    )
    for record in report.values():
        assert record["shrinkage_loss_fraction"] + record[
            "rank_loss_fraction"
        ] == pytest.approx(record["total_loss_fraction"], abs=1e-12)


def test_full_rank_reproduces_the_identified_block_exactly(standardized_synthetic):
    """At full rank the construction is exact, so rank loss is pure parsimony.

    ``X = A - B`` is the only block the loadings reproduce unconditionally: it
    carries no competition term, so it is a pure statement about the two
    representation ranks. The same-team block ``S = A + B - Q`` closes exactly
    only when the competition gate admits ``Q``; on this synthetic frame it
    does not, because the generator has no competition structure to detect.
    The real-data counterpart is asserted against the committed rank sweep in
    :func:`test_real_data_rank_sweep_closes_at_full_rank`.
    """
    full = fit_repaired_factors(
        standardized_synthetic,
        STATS,
        RepairSpec(name="full", family="C_rank", k_game=6, r_contrast=6),
        bootstrap=120,
        seed=73,
    )
    assert np.allclose(
        full.loadings.cross_team_correlation(), full.cross_team_shrunk, atol=1e-12
    )

    truncated = fit_repaired_factors(
        standardized_synthetic, STATS, V1_CONTROL_SPEC, bootstrap=120, seed=73
    )
    assert np.max(
        np.abs(
            truncated.loadings.cross_team_correlation() - truncated.cross_team_shrunk
        )
    ) > np.max(
        np.abs(full.loadings.cross_team_correlation() - full.cross_team_shrunk)
    )


def test_raising_the_contrast_rank_closes_the_representation_gap(
    standardized_synthetic,
):
    """A rank-1 contrast cannot express every team-contrast direction."""
    errors: list[float] = []
    for r_contrast in (1, 2, 3, 6):
        fit = fit_repaired_factors(
            standardized_synthetic,
            STATS,
            RepairSpec(
                name=f"r{r_contrast}",
                family="C_rank",
                k_game=6,
                r_contrast=r_contrast,
            ),
            bootstrap=120,
            seed=73,
        )
        assert fit.loadings.r_contrast == r_contrast
        errors.append(
            float(
                np.abs(
                    fit.loadings.cross_team_correlation() - fit.cross_team_shrunk
                ).max()
            )
        )

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-12


def test_real_data_rank_sweep_closes_at_full_rank():
    """On the real pre-2024 frame the full-rank fit is exact to machine precision.

    This is the measurement behind family C: the residual same-team gap left
    after shrinkage is a representation limit, not noise.
    """
    path = REPAIR_DIR / "bucket_diagnostics.json"
    if not path.exists():
        pytest.skip("bucket diagnostics have not been generated")
    sweep = {
        (row["k_game"], row["r_contrast"]): row
        for row in json.loads(path.read_text())["rank_sweep"]
    }
    n_stats = len(STATS)
    assert sweep[(n_stats, n_stats)]["max_abs_deviation_from_shrunk"] < 1e-12
    assert sweep[(2, 1)]["max_abs_deviation_from_shrunk"] > 1e-4


# ----------------------------------------------------------------------
# PSD construction and same-player invariance under the repair
# ----------------------------------------------------------------------


@pytest.mark.parametrize("r_contrast", [1, 2, 6])
@pytest.mark.parametrize("k_game", [2, 6])
def test_repaired_game_covariance_is_psd_with_unit_diagonal(
    standardized_synthetic, k_game, r_contrast
):
    fit = fit_repaired_factors(
        standardized_synthetic,
        STATS,
        RepairSpec(
            name=f"k{k_game}r{r_contrast}",
            family="C_rank",
            shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
            k_game=k_game,
            r_contrast=r_contrast,
        ),
        bootstrap=120,
        seed=73,
    )
    dimensions = []
    for team_id, side in ((1, 1), (2, -1)):
        for member in range(5):
            for stat in STATS:
                dimensions.append(
                    GameDimension(
                        player_id=100 * team_id + member,
                        team_id=team_id,
                        stat=stat,
                        side=side,
                        role="starter",
                    )
                )
    within = {
        dim.player_id: np.eye(len(STATS)) for dim in dimensions
    }
    covariance = build_game_covariance(dimensions, fit.loadings, within)
    assert covariance.min_eigenvalue >= -1e-10
    assert np.allclose(np.diag(covariance.correlation), 1.0, atol=1e-12)
    assert covariance.cholesky.shape == (len(dimensions), len(dimensions))


@pytest.mark.parametrize("r_contrast", [1, 2, 6])
def test_same_player_block_stays_pinned_under_the_repair(
    standardized_synthetic, r_contrast
):
    """Gate E must hold for every repair rank, not just V1's."""
    fit = fit_repaired_factors(
        standardized_synthetic,
        STATS,
        RepairSpec(
            name=f"r{r_contrast}",
            family="C_rank",
            shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
            k_game=6,
            r_contrast=r_contrast,
        ),
        bootstrap=120,
        seed=73,
    )
    rng = np.random.default_rng(5)
    within = {}
    dimensions = []
    for team_id, side in ((1, 1), (2, -1)):
        for member in range(4):
            player_id = 100 * team_id + member
            noise = rng.normal(0.0, 0.25, size=(len(STATS), len(STATS)))
            block = np.eye(len(STATS)) + 0.5 * (noise + noise.T)
            block, _ = project_psd_rank(block, rank=len(STATS))
            scale = np.sqrt(np.diag(block))
            block = block / np.outer(scale, scale)
            within[player_id] = block
            for stat in STATS:
                dimensions.append(
                    GameDimension(
                        player_id=player_id,
                        team_id=team_id,
                        stat=stat,
                        side=side,
                        role="rotation",
                    )
                )

    covariance = build_game_covariance(dimensions, fit.loadings, within)
    worst = 0.0
    for player_id, block in within.items():
        implied = implied_within_player_correlation(covariance, player_id)
        worst = max(worst, float(np.max(np.abs(implied - block))))
    assert worst <= 1e-9, f"same-player block drifted by {worst}"


def test_no_dependence_parameter_is_indexed_by_player_or_pair():
    """REPAIR GATE 11: ``pairwise_parameter_count`` must stay zero."""
    for k_game, r_contrast in ((2, 1), (6, 6), (4, 4)):
        spec = RepairSpec(
            name="x", family="C_rank", k_game=k_game, r_contrast=r_contrast
        )
        counts = spec.parameter_count(len(STATS))
        assert counts["pairwise"] == 0
        assert counts["player_indexed"] == 0
        assert counts["role_indexed"] <= 3


def test_repair_transfers_to_an_unseen_player_without_refitting(
    standardized_synthetic,
):
    """A roster the fit never saw must simulate from the same stat-level loadings."""
    fit = fit_repaired_factors(
        standardized_synthetic,
        STATS,
        RepairSpec(
            name="full",
            family="C_rank",
            shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
            k_game=6,
            r_contrast=6,
        ),
        bootstrap=120,
        seed=73,
    )
    dimensions = [
        GameDimension(
            player_id=999_999 + member,
            team_id=1 if member < 3 else 2,
            stat=stat,
            side=1 if member < 3 else -1,
            role="never_seen_role",
        )
        for member in range(6)
        for stat in STATS
    ]
    within = {dim.player_id: np.eye(len(STATS)) for dim in dimensions}
    covariance = build_game_covariance(dimensions, fit.loadings, within)
    assert covariance.min_eigenvalue >= -1e-10
    # An unseen role falls back to the pooled scale of exactly 1.0.
    assert fit.loadings.scale_for_role("never_seen_role") == 1.0


# ----------------------------------------------------------------------
# inner-selection contract
# ----------------------------------------------------------------------


def test_inner_folds_are_strictly_temporal_and_exclude_the_holdout():
    driver = load_driver("02_inner_validation.py")
    folds = driver.build_folds([2020, 2021, 2022, 2023])
    assert [fold.score_season for fold in folds] == [2021, 2022, 2023]
    for fold in folds:
        assert max(fold.train_seasons) < fold.score_season, "training must precede"
        assert not set(fold.train_seasons) & set(HOLDOUT_SEASONS)
        assert fold.score_season not in HOLDOUT_SEASONS


def test_inner_fold_construction_never_leaks_a_later_season_backwards():
    driver = load_driver("02_inner_validation.py")
    folds = driver.build_folds([2018, 2019, 2020, 2021])
    for fold in folds:
        later = [s for s in fold.train_seasons if s >= fold.score_season]
        assert later == [], f"fold {fold.label} trains on {later}"


def test_scoring_a_fold_uses_training_standardisation_constants():
    """No lookahead: the inner validation season must not set its own scale."""
    driver = load_driver("02_inner_validation.py")
    frame = synthetic_residuals(seasons=(2020, 2021), games_per_season=40)
    fold = driver.InnerFold(train_seasons=(2020,), score_season=2021)
    result = driver.score_fold(frame, fold, V1_CONTROL_SPEC, bootstrap=40, seed=73)

    train = frame[frame["season"] == 2020]
    _, train_moments = standardize_residuals(train, STATS)
    score = frame[frame["season"] == 2021]
    with_train, _ = standardize_residuals(score, STATS, moments=train_moments)
    own, _ = standardize_residuals(score, STATS)
    # The two standardisations genuinely differ, so the choice is observable.
    assert not np.allclose(
        with_train["zs_pts"].to_numpy(), own["zs_pts"].to_numpy(), atol=1e-9
    )

    expected = pair_moments(with_train, STATS, bootstrap=40, seed=73)
    from nba_prop_quant.research.game_latent_state.validation import bucket_values

    expected_buckets = bucket_values(
        STATS, expected.same_team, expected.cross_team
    )
    assert result["observed_buckets"]["teammate_reb_reb"] == pytest.approx(
        expected_buckets["teammate_reb_reb"], rel=1e-12
    )


def test_candidate_grid_is_pre_registered_and_keeps_the_v1_role_setting():
    driver = load_driver("02_inner_validation.py")
    for supported in (True, False):
        specs = driver.candidate_grid(supported)
        names = [spec.name for spec in specs]
        assert names[0] == "v1_control", "the control must be scored"
        assert len(names) == len(set(names)), "candidate names must be unique"
        assert all(spec.role_column == "role_bucket" for spec in specs)
        families = {spec.family for spec in specs}
        assert {"A_threshold", "B_empirical_bayes", "C_rank"} <= families


def test_candidate_selection_is_deterministic_and_rule_driven():
    """Selection must be a function of the scores, not of iteration order."""
    driver = load_driver("02_inner_validation.py")

    def summary(name, target_error, rmse, protected_z, overshoot, params=12):
        return {
            "spec": {"name": name, "family": "x"},
            "parameter_count": {"stat_indexed_game": params, "role_indexed": 0},
            "mean_global_rmse": rmse,
            "mean_target_abs_error": target_error,
            "mean_target_abs_z": 1.0,
            "mean_bucket_z_error": {
                "passer_ast_teammate_pts": protected_z,
                "teammate_pts_reb": 0.0,
            },
            "target_overshoot_z": {
                "teammate_reb_reb": 0.0,
                "teammate_ast_ast": overshoot,
            },
        }

    summaries = {
        "v1_control": summary("v1_control", 0.006, 0.005, 0.5, 0.0),
        "good": summary("good", 0.003, 0.004, 0.6, 0.5),
        "overshoots": summary("overshoots", 0.001, 0.004, 0.6, 9.0),
        "wrecks_global": summary("wrecks_global", 0.001, 0.010, 0.6, 0.5),
        "wrecks_protected": summary("wrecks_protected", 0.001, 0.004, 5.0, 0.5),
    }
    winner, verdicts = driver.select_candidate(summaries)
    assert winner == "good"
    assert verdicts["overshoots"]["eligible"] is False
    assert verdicts["wrecks_global"]["eligible"] is False
    assert verdicts["wrecks_protected"]["eligible"] is False

    # Order independence.
    reordered = dict(reversed(list(summaries.items())))
    assert driver.select_candidate(reordered)[0] == "good"


def test_selection_breaks_ties_toward_the_parsimonious_candidate():
    driver = load_driver("02_inner_validation.py")

    def summary(name, params):
        return {
            "spec": {"name": name, "family": "x"},
            "parameter_count": {"stat_indexed_game": params, "role_indexed": 0},
            "mean_global_rmse": 0.004,
            "mean_target_abs_error": 0.003,
            "mean_target_abs_z": 1.0,
            "mean_bucket_z_error": {
                "passer_ast_teammate_pts": 0.5,
                "teammate_pts_reb": 0.0,
            },
            "target_overshoot_z": {
                "teammate_reb_reb": 0.0,
                "teammate_ast_ast": 0.0,
            },
        }

    summaries = {
        "v1_control": summary("v1_control", 40),
        "rich": summary("rich", 36),
        "lean": summary("lean", 18),
    }
    assert driver.select_candidate(summaries)[0] == "lean"


def test_selection_refuses_when_nothing_satisfies_the_constraints():
    driver = load_driver("02_inner_validation.py")
    summaries = {
        "v1_control": {
            "spec": {"name": "v1_control", "family": "x"},
            "parameter_count": {"stat_indexed_game": 12},
            "mean_global_rmse": 0.004,
            "mean_target_abs_error": 0.006,
            "mean_target_abs_z": 1.0,
            "mean_bucket_z_error": {
                "passer_ast_teammate_pts": 0.5,
                "teammate_pts_reb": 0.0,
            },
            "target_overshoot_z": {
                "teammate_reb_reb": 99.0,
                "teammate_ast_ast": 99.0,
            },
        }
    }
    with pytest.raises(RuntimeError, match="no candidate satisfied"):
        driver.select_candidate(summaries)


def test_diagnostics_driver_refuses_holdout_rows():
    driver = load_driver("01_diagnose_buckets.py")
    assert driver.HOLDOUT_SEASONS == HOLDOUT_SEASONS
    frame = synthetic_residuals(seasons=(2020, 2024), games_per_season=5)
    path = REPAIR_DIR / "_pytest_tmp_residuals.parquet"
    try:
        frame.to_parquet(path, index=False)
        loaded = driver.load_pre_holdout(path)
        assert set(loaded["season"].unique()) == {2020}
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# frozen candidate manifest
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not (REPAIR_DIR / "bucket_repair_candidate.json").exists(),
    reason="candidate not frozen yet",
)
def test_frozen_candidate_manifest_records_the_full_selection_contract():
    freeze = json.loads(
        (REPAIR_DIR / "bucket_repair_candidate.json").read_text(encoding="utf-8")
    )
    for key in (
        "candidate_family",
        "hyperparameters",
        "inner_temporal_folds",
        "inner_metrics",
        "selection_rationale",
        "parameter_counts",
        "seed",
        "code_sha",
    ):
        assert key in freeze, f"frozen candidate is missing {key}"
    assert freeze["frozen"] is True
    assert freeze["holdout_used_for_selection"] is False
    assert freeze["parameter_counts"]["pairwise"] == 0
    assert freeze["parent_shadow_v1_sha"] == "1c5b8c93569ee25afd4eb4222158300702bd9471"
    assert (
        freeze["promotion_eligibility"]
        == "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION"
    )
    assert sorted(freeze["holdout_seasons"]) == sorted(HOLDOUT_SEASONS)


@pytest.mark.skipif(
    not (REPAIR_DIR / "inner_validation.json").exists(),
    reason="inner validation has not run",
)
def test_inner_validation_record_never_touches_the_holdout():
    """The selection record must prove, on its own face, that it is clean."""
    inner = json.loads(
        (REPAIR_DIR / "inner_validation.json").read_text(encoding="utf-8")
    )
    assert inner["holdout_used_for_selection"] is False
    assert sorted(inner["holdout_seasons_excluded"]) == sorted(HOLDOUT_SEASONS)
    assert not set(inner["pre_holdout_seasons"]) & set(HOLDOUT_SEASONS)
    for fold in inner["inner_folds"]:
        assert not set(fold["train_seasons"]) & set(HOLDOUT_SEASONS)
        assert fold["score_season"] not in HOLDOUT_SEASONS
        assert max(fold["train_seasons"]) < fold["score_season"]


@pytest.mark.skipif(
    not (REPAIR_DIR / "factor_spec.json").exists(),
    reason="repair factor spec not written yet",
)
def test_repair_factor_spec_excludes_the_holdout_from_every_fitted_quantity():
    spec = json.loads((REPAIR_DIR / "factor_spec.json").read_text(encoding="utf-8"))
    assert not set(spec["training_seasons"]) & set(HOLDOUT_SEASONS)
    assert sorted(spec["validation_seasons"]) == sorted(HOLDOUT_SEASONS)
    assert spec["parent_shadow_v1_sha"] == "1c5b8c93569ee25afd4eb4222158300702bd9471"
    loadings = SharedFactorLoadings.from_payload(spec["loadings"])
    assert loadings.r_contrast >= 1
    assert len(loadings.stats) == len(STATS)


# ----------------------------------------------------------------------
# containment: the repair may not promote or touch production
# ----------------------------------------------------------------------


def test_repair_module_defines_no_promotion_or_publishing_entry_point():
    import ast

    source = (
        PROJECT / "src" / "nba_prop_quant" / "research" / "game_latent_state"
        / "repair.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = ("promote", "register_fit", "publish", "deploy")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert not any(token in node.name.lower() for token in forbidden), node.name


def test_repair_drivers_declare_themselves_non_promotable():
    for name in (
        "01_diagnose_buckets.py",
        "02_inner_validation.py",
        "03_fit_repair_candidate.py",
    ):
        text = (REPAIR_DIR / name).read_text(encoding="utf-8")
        assert "NOT PROMOTABLE" in text, f"{name} must declare itself non-promotable"
        assert "wizardofodds" not in text.lower().replace(
            "origin/production/wizardofodds-integration", ""
        ), f"{name} must not reference a publishing surface"


def test_repair_branch_modifies_no_protected_production_path():
    from nba_prop_quant.research.game_latent_state.safety import (
        modified_production_paths,
    )

    offenders = modified_production_paths(PROJECT)
    assert offenders == [], f"repair branch touched production: {offenders}"


def test_repair_changes_live_only_in_research_and_test_namespaces():
    from nba_prop_quant.research.game_latent_state.safety import changed_paths

    allowed = (
        "research/",
        "src/nba_prop_quant/research/",
        "tests/test_game_latent_state_shadow",
    )
    offenders = [path for path in changed_paths(PROJECT) if not path.startswith(allowed)]
    assert offenders == [], f"unexpected paths on the repair branch: {offenders}"


@pytest.mark.skipif(
    not (REPAIR_DIR / "bucket_repair_candidate.json").exists(),
    reason="candidate not frozen yet",
)
def test_a_repair_candidate_cannot_promote_even_when_every_gate_passes():
    from nba_prop_quant.research.game_latent_state.gates import (
        ShadowPromotionRefused,
        assert_promotable,
    )

    freeze = json.loads(
        (REPAIR_DIR / "bucket_repair_candidate.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ShadowPromotionRefused):
        assert_promotable({"verdict": "anything", "frozen_candidate": freeze})
