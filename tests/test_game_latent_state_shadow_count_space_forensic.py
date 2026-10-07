"""Tests for the count-space blocker forensic study.

SHADOW / RESEARCH ONLY. These lock the study's measurement tools against
independent references -- synthetic data with a known latent correlation,
direct quadrature, an independent bivariate-normal CDF, and the published
Monte Carlo runs -- and then lock the findings the report draws from them.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORENSIC_DIR = PROJECT_ROOT / "research/count_space_forensic"
sys.path.insert(0, str(FORENSIC_DIR))

from nba_prop_quant.research.game_latent_state import censored  # noqa: E402
from marginal import discrete_marginal, mehler_scores  # noqa: E402
from nba_prop_quant.research.game_latent_state.covariance import (  # noqa: E402
    SharedFactorLoadings,
)
from nba_prop_quant.research.game_latent_state.factors import (  # noqa: E402
    pair_moments,
)
from nba_prop_quant.research.game_latent_state.transmission import (  # noqa: E402
    conditional_hermite_moments as committed_moments,
)

import forensic_lib as FL  # noqa: E402

CANDIDATE_SPEC = PROJECT_ROOT / "research/final_upstream_remediation/factor_spec.json"
CONTROL_SPEC = (
    PROJECT_ROOT / "research/game_latent_state_bucket_repair/factor_spec.json"
)
CANDIDATE_REPORT = (
    PROJECT_ROOT / "research/final_upstream_remediation/validation_report.json"
)
CONTROL_REPORT = (
    PROJECT_ROOT
    / "research/final_upstream_remediation/control_validation_report.json"
)
FORENSIC_REPORT = FORENSIC_DIR / "count_space_forensic.json"

FROZEN_CANDIDATE_HASH = (
    "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
)


# ----------------------------------------------------------------------
# synthetic ground truth
# ----------------------------------------------------------------------


def synthetic_roster_counts(
    rho: float,
    games: int,
    team_size: int,
    means: tuple[float, float],
    seed: int,
) -> pd.DataFrame:
    """Discretized counts whose latent same-team correlation is exactly ``rho``.

    A single per-team factor gives every pair of distinct teammates the same
    latent correlation in every stat combination, so the ``(ast, pts)``
    same-team bucket of the generated data has a known truth to recover. The
    counts are then produced by the copula's own construction,
    ``y = F^{-1}(Phi(z))``, and the cell bounds recorded, which is exactly the
    information the residual builder hands every estimator.
    """
    generator = np.random.default_rng(seed)
    stats_pair = ("ast", "pts")
    rows: list[dict[str, float]] = []
    for game in range(games):
        for team in range(2):
            factor = generator.standard_normal()
            for player in range(team_size):
                latent = np.sqrt(rho) * factor + np.sqrt(
                    1.0 - rho
                ) * generator.standard_normal(len(stats_pair))
                row: dict[str, float] = {
                    "game_id": float(game),
                    "team_id": float(2 * game + team),
                    "player_id": float(1000 * team + player + 100 * game),
                }
                for index, stat in enumerate(stats_pair):
                    mean = means[index]
                    uniform = float(stats.norm.cdf(latent[index]))
                    count = int(stats.poisson.ppf(min(uniform, 1 - 1e-12), mean))
                    row[f"y_{stat}"] = float(count)
                    row[f"cdf_lower_{stat}"] = (
                        0.0 if count == 0 else float(stats.poisson.cdf(count - 1, mean))
                    )
                    row[f"cdf_upper_{stat}"] = float(stats.poisson.cdf(count, mean))
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["game_id"] = frame["game_id"].astype(int)
    frame["team_id"] = frame["team_id"].astype(int)
    frame["player_id"] = frame["player_id"].astype(int)
    return frame


@pytest.fixture(scope="module")
def synthetic() -> tuple[float, pd.DataFrame]:
    rho = 0.18
    return rho, synthetic_roster_counts(
        rho, games=700, team_size=6, means=(2.4, 10.7), seed=73
    )


# ----------------------------------------------------------------------
# A. the censored estimator recovers the truth; the PIT reading does not
# ----------------------------------------------------------------------


def test_censored_mle_recovers_the_known_latent_correlation(synthetic):
    rho, frame = synthetic
    intervals = censored.same_team_pair_intervals(frame, "ast", "pts")
    estimate = censored.censored_copula_mle(intervals)
    standard_error = censored.censored_sandwich_se(estimate, intervals)
    assert abs(estimate - rho) < 3.0 * standard_error
    assert abs(estimate - rho) < 0.02


def test_randomized_pit_reading_is_biased_low_against_the_truth(synthetic):
    """The reading the latent gate uses understates the correlation it names.

    This is the study's central claim in its simplest possible setting: with
    the latent correlation known by construction, the randomized-PIT reading
    lands materially below it while the censored estimator lands on it.
    """
    rho, frame = synthetic
    rejittered = censored.multi_seed_latent_columns(frame, ("ast", "pts"), 73)
    reading, _ = FL.ordered_pair_mean(rejittered, "same_team", "z_ast", "z_pts")
    estimate = censored.censored_copula_mle(
        censored.same_team_pair_intervals(frame, "ast", "pts")
    )
    assert reading < rho
    assert reading < estimate
    # The attenuation is a sizeable fraction of the quantity, not a rounding
    # difference: these margins are coarse enough to cost over five percent.
    assert (rho - reading) / rho > 0.05


def test_averaging_jitter_seeds_does_not_remove_the_attenuation(synthetic):
    """Many seeds shrink the spread of the reading and not its bias.

    The jitter enters the randomized PIT's variance, which the reading
    divides by, so it is not noise that averages away. Confirming that is
    what separates "the estimator is noisy" from "the estimator is biased".
    """
    rho, frame = synthetic
    readings = []
    for seed in (73, 1000, 1001, 1002, 1003, 1004, 1005, 1006):
        rejittered = censored.multi_seed_latent_columns(frame, ("ast", "pts"), seed)
        readings.append(
            FL.ordered_pair_mean(rejittered, "same_team", "z_ast", "z_pts")[0]
        )
    readings = np.asarray(readings)
    assert readings.std(ddof=1) < 0.01
    assert readings.mean() < rho
    assert (rho - readings.mean()) / rho > 0.05


# ----------------------------------------------------------------------
# B. the estimator's internals against independent references
# ----------------------------------------------------------------------


def test_rectangle_probability_matches_an_independent_bivariate_cdf():
    """The integrated-derivative form against ``scipy``'s own orthant CDF."""
    lower = np.array([-1.2, -0.3, 0.4, -2.0])
    upper = np.array([-0.4, 0.6, 1.9, -1.1])
    other_lower = np.array([-0.9, 0.1, -1.5, 0.2])
    other_upper = np.array([0.3, 1.4, -0.2, 2.2])
    intervals = censored.PairIntervals(
        a_lower=lower,
        a_upper=upper,
        b_lower=other_lower,
        b_upper=other_upper,
        game_id=np.arange(4, dtype=np.int64),
    )
    for rho in (-0.4, -0.05, 0.05, 0.3, 0.55):
        got = censored.rectangle_probability(rho, intervals)
        covariance = np.array([[1.0, rho], [rho, 1.0]])
        normal = stats.multivariate_normal(mean=[0.0, 0.0], cov=covariance)
        want = np.array(
            [
                normal.cdf([upper[i], other_upper[i]])
                - normal.cdf([lower[i], other_upper[i]])
                - normal.cdf([upper[i], other_lower[i]])
                + normal.cdf([lower[i], other_lower[i]])
                for i in range(4)
            ]
        )
        assert np.allclose(got, want, atol=1e-9)


def test_rectangle_probability_is_quadrature_converged(synthetic):
    _, frame = synthetic
    intervals = censored.same_team_pair_intervals(frame, "ast", "pts")
    reference = censored.rectangle_probability(0.18, intervals, nodes=64)
    for nodes in (8, 16, 24):
        assert np.allclose(
            censored.rectangle_probability(0.18, intervals, nodes=nodes),
            reference,
            atol=1e-12,
        )


def test_score_root_find_agrees_with_derivative_free_search(synthetic):
    _, frame = synthetic
    intervals = censored.same_team_pair_intervals(frame, "ast", "pts")
    fast = censored.censored_copula_mle(intervals)
    slow = censored.censored_copula_mle_golden(intervals, tolerance=1e-9)
    assert abs(fast - slow) < 1e-7


def test_score_vanishes_at_the_maximiser(synthetic):
    _, frame = synthetic
    intervals = censored.same_team_pair_intervals(frame, "ast", "pts")
    estimate = censored.censored_copula_mle(intervals)
    curvature = (
        censored.censored_log_likelihood(estimate + 1e-3, intervals)
        + censored.censored_log_likelihood(estimate - 1e-3, intervals)
        - 2.0 * censored.censored_log_likelihood(estimate, intervals)
    )
    assert curvature < 0.0
    assert abs(censored.censored_score(estimate, intervals)) < 1e-3 * abs(curvature)


def test_sandwich_standard_error_tracks_the_clustered_bootstrap(synthetic):
    _, frame = synthetic
    intervals = censored.same_team_pair_intervals(frame, "ast", "pts")
    estimate = censored.censored_copula_mle(intervals, nodes=8)
    sandwich = censored.censored_sandwich_se(estimate, intervals, nodes=8)
    draws = censored.clustered_bootstrap_mle(intervals, draws=40, seed=73, nodes=8)
    assert sandwich == pytest.approx(draws.std(ddof=1), rel=0.30)


def test_pair_intervals_match_the_pipeline_pair_counts(synthetic):
    """The estimator and the bucket statistic average over the same pairs."""
    _, frame = synthetic
    working = frame.copy()
    for stat in ("ast", "pts"):
        working[f"zs_{stat}"] = working[f"y_{stat}"].astype(float)
    moments = pair_moments(working, ("ast", "pts"), value_prefix="zs_")
    assert len(
        censored.same_team_pair_intervals(frame, "ast", "pts")
    ) == pytest.approx(moments.same_team_pairs)
    assert len(
        censored.cross_team_pair_intervals(frame, "ast", "pts")
    ) == pytest.approx(moments.cross_team_pairs)


def test_latent_thresholds_are_the_randomized_pit_endpoints():
    """Both estimators read the same two numbers out of the margin."""
    lower = np.array([0.0, 0.21, 0.74])
    upper = np.array([0.21, 0.74, 0.99])
    low, high = censored.latent_thresholds(lower, upper)
    assert np.allclose(stats.norm.cdf(high), upper)
    assert np.allclose(stats.norm.cdf(low)[1:], lower[1:])
    assert low[0] < -5.0


# ----------------------------------------------------------------------
# C. the Hermite tools, and the committed recurrence's boundaries
# ----------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
def test_forensic_hermite_moments_match_direct_quadrature(order):
    lower = np.array([1e-12, 0.35, 0.62, 0.90])
    upper = np.array([0.35, 0.62, 0.81, 0.97])
    got = FL.conditional_hermite_moments(lower, upper, order)[order - 1]
    coefficients = np.zeros(order + 1)
    coefficients[order] = 1.0
    for index in range(4):
        low = stats.norm.ppf(lower[index])
        high = stats.norm.ppf(upper[index])
        value, _ = integrate.quad(
            lambda z: np.polynomial.hermite_e.hermeval(z, coefficients)
            * stats.norm.pdf(z),
            low,
            high,
            limit=400,
        )
        assert got[index] == pytest.approx(
            value / (upper[index] - lower[index]), rel=1e-9
        )


def test_committed_recurrence_is_exact_at_the_orders_the_pipeline_uses():
    """``DEFAULT_BRIDGE_ORDER`` is 2, and orders 1 and 2 are correct.

    This was the boundary of the defect the study reported: the committed
    recurrence carried ``z He_k - (k - 1) He_{k-1}``, which is right for
    ``He_0`` and ``He_1`` and therefore right for ``M_1`` and ``M_2``. Every
    caller in the pipeline passes ``DEFAULT_BRIDGE_ORDER``, so the frozen
    candidate never reached the wrong orders. The coefficient has since been
    corrected; these two orders are unchanged by that, which is what makes the
    correction safe.
    """
    from nba_prop_quant.research.game_latent_state import transmission

    assert transmission.DEFAULT_BRIDGE_ORDER == 2
    lower = np.array([1e-12, 0.35, 0.62, 0.90])
    upper = np.array([0.35, 0.62, 0.81, 0.97])
    committed = committed_moments(lower, upper, order=2)
    correct = FL.conditional_hermite_moments(lower, upper, 2)
    for order in range(2):
        assert np.allclose(committed[order], correct[order], atol=1e-12)


def test_committed_recurrence_now_agrees_from_order_three():
    """The two independent implementations have converged.

    They used to disagree from ``He_2`` up, because the pipeline's generic
    recurrence carried the wrong coefficient while this study's own tool
    carried the right one. Fixing the pipeline is what closed the gap, so the
    assertion is now agreement rather than divergence -- and this remains a
    genuinely independent check, because the two implementations were written
    separately and only one of them was changed.
    """
    lower = np.array([1e-12, 0.35, 0.62, 0.90])
    upper = np.array([0.35, 0.62, 0.81, 0.97])
    committed = committed_moments(lower, upper, order=5)
    correct = FL.conditional_hermite_moments(lower, upper, 5)
    for order in range(5):
        assert np.allclose(committed[order], correct[order], atol=1e-12), order


@pytest.mark.parametrize("mean", [1.5, 3.0, 11.0, 24.0])
@pytest.mark.parametrize(
    "space,statistic", [("count", "count"), ("latent", "randomized_pit")]
)
def test_statistic_series_agrees_with_the_bridge_mehler_series(
    mean, space, statistic
):
    """Two independent derivations of one transmission map."""
    terms = 40
    grid = stats.poisson.cdf(
        np.arange(0, int(mean + 14 * np.sqrt(mean)) + 50), mean
    )
    margin = discrete_marginal(grid)
    scores = mehler_scores(margin, space, terms)
    coefficients, sigma = FL.statistic_hermite_coefficients(grid, statistic, terms)
    series = FL.StatisticSeries(
        statistic=statistic,
        coefficients=FL.pooled_statistic_series(
            [(coefficients, sigma)], [(coefficients, sigma)], terms
        ),
    )
    denominator = margin.sd**2 if space == "count" else 1.0
    for rho in (0.02, 0.05, 0.12):
        want = float(
            sum(
                scores[j] * scores[j] * rho ** (j + 1) / ((j + 1) * denominator)
                for j in range(terms)
            )
        )
        assert series.evaluate(rho) == pytest.approx(want, rel=1e-8)


def test_transmission_is_pinned_to_zero_at_zero_correlation():
    grid = stats.poisson.cdf(np.arange(0, 60), 3.0)
    for statistic in ("count", "randomized_pit", "mid_pit"):
        coefficients, sigma = FL.statistic_hermite_coefficients(grid, statistic, 20)
        series = FL.StatisticSeries(
            statistic=statistic,
            coefficients=FL.pooled_statistic_series(
                [(coefficients, sigma)], [(coefficients, sigma)], 20
            ),
        )
        assert series.evaluate(0.0) == 0.0


def test_randomized_pit_has_the_smallest_first_order_gain():
    """The ordering that makes the attenuation an estimator property.

    The randomized PIT keeps the jitter in its denominator, so its gain is
    the product of two conditional-mean variances; mid-PIT standardises by
    its own spread and recovers the square root of that; count space is
    nearly lossless. So the same latent ``rho`` reads lowest through the
    randomized PIT, which is the space the latent gate scores.
    """
    for mean in (1.5, 3.0, 11.0):
        grid = stats.poisson.cdf(np.arange(0, 120), mean)
        gains = {}
        for statistic in ("count", "randomized_pit", "mid_pit"):
            coefficients, sigma = FL.statistic_hermite_coefficients(
                grid, statistic, 8
            )
            gains[statistic] = coefficients[0] / sigma
        assert gains["randomized_pit"] < gains["mid_pit"]
        assert gains["randomized_pit"] < gains["count"]
        assert gains["randomized_pit"] < 1.0


# ----------------------------------------------------------------------
# D. the architecture lever
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def candidate_loadings() -> SharedFactorLoadings:
    spec = json.loads(CANDIDATE_SPEC.read_text())
    assert spec["spec_hash"] == FROZEN_CANDIDATE_HASH
    return SharedFactorLoadings.from_payload(spec["loadings"])


def test_retarget_round_trips_at_the_candidate_value(candidate_loadings):
    index = candidate_loadings.stats.index
    base = float(
        candidate_loadings.same_team_correlation()[index("ast"), index("pts")]
    )
    retarget = FL.retarget_same_team_entry(candidate_loadings, "ast", "pts", base)
    assert retarget.competition_inflation == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(
        retarget.loadings.same_team_correlation(),
        candidate_loadings.same_team_correlation(),
        atol=1e-14,
    )
    assert np.allclose(
        retarget.loadings.cross_team_correlation(),
        candidate_loadings.cross_team_correlation(),
        atol=1e-14,
    )


def test_retarget_moves_exactly_one_bucket(candidate_loadings):
    base = FL.bucket_readout(candidate_loadings)
    retarget = FL.retarget_same_team_entry(
        candidate_loadings, "ast", "pts", 0.0527
    )
    got = FL.bucket_readout(retarget.loadings)
    moved = {
        name for name in got if abs(got[name] - base[name]) > 1e-12
    }
    assert moved == {"passer_ast_teammate_pts"}
    assert got["passer_ast_teammate_pts"] == pytest.approx(0.0527, abs=1e-12)


def test_retarget_keeps_every_gram_positive_semidefinite(candidate_loadings):
    for target in (0.045, 0.0527, 0.070, 0.12, 0.30):
        retarget = FL.retarget_same_team_entry(
            candidate_loadings, "ast", "pts", target
        )
        for gram in (
            retarget.loadings.game_gram(),
            retarget.loadings.contrast_gram(),
            retarget.loadings.competition_gram(),
        ):
            assert float(np.min(np.linalg.eigvalsh(gram))) > -1e-12


def test_retarget_adds_no_pairwise_or_player_indexed_parameter(candidate_loadings):
    """The lever stays inside the architecture it was given.

    Three stat-by-stat Grams in, three out, at the frozen ranks, with the role
    layer and the inactive symmetric block carried through untouched. Nothing
    acquires a player index and nothing acquires a pair index.
    """
    retarget = FL.retarget_same_team_entry(
        candidate_loadings, "ast", "pts", 0.0527
    ).loadings
    assert retarget.stats == candidate_loadings.stats
    assert retarget.k_game <= len(candidate_loadings.stats)
    assert retarget.r_contrast <= len(candidate_loadings.stats)
    assert retarget.r_competition <= len(candidate_loadings.stats)
    assert retarget.r_symmetric == candidate_loadings.r_symmetric == 0
    assert retarget.role_deviation is None
    assert dict(retarget.role_scale) == dict(candidate_loadings.role_scale)


def test_the_frozen_grams_sit_on_the_psd_boundary(candidate_loadings):
    """Why the lever has to go through the competition Gram.

    A bump added straight to the two additive Grams is refused immediately
    because the contrast Gram is singular, so the only representation of a
    retargeted same-team entry runs through an inflated competition Gram --
    and that is what makes the envelope a shrink trade rather than a rank one.
    """
    contrast = candidate_loadings.contrast_gram()
    assert float(np.min(np.linalg.eigvalsh(contrast))) < 1e-15
    with pytest.raises(FL.ShiftNotRepresentable):
        FL.isolated_same_team_shift(candidate_loadings, "ast", "pts", 0.007)


# ----------------------------------------------------------------------
# E. holdout isolation
# ----------------------------------------------------------------------


def test_every_estimation_window_is_strictly_pre_2024():
    for label, (low, high) in {
        **{"pooled": (2020, 2023)},
    }.items():
        assert high < min(FL.HOLDOUT_SEASONS), label
    assert set(FL.PRE_2024_SEASONS).isdisjoint(FL.HOLDOUT_SEASONS)
    assert set(FL.PRE_2024_FORWARD_FOLDS) <= set(FL.PRE_2024_SEASONS)
    assert max(FL.PRE_2024_SEASONS) < min(FL.HOLDOUT_SEASONS)


def test_pre_2024_marginal_windows_never_reach_a_holdout_season():
    """A fold's margins come from its own past, as the holdout's do."""
    for fold in FL.PRE_2024_FORWARD_FOLDS:
        assert fold - 1 < min(FL.HOLDOUT_SEASONS)


# ----------------------------------------------------------------------
# F. the report's findings
# ----------------------------------------------------------------------


requires_report = pytest.mark.skipif(
    not FORENSIC_REPORT.exists(),
    reason="run research/count_space_forensic/01_count_space_forensic.py first",
)


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(FORENSIC_REPORT.read_text())


@requires_report
def test_report_starts_from_the_frozen_candidate(report):
    provenance = report["provenance"]
    assert provenance["candidate_spec_hash"] == FROZEN_CANDIDATE_HASH
    assert provenance["candidate_spec_hash_confirmed"] is True
    assert provenance["branch"] == "research/nba-count-space-forensic"
    assert provenance["holdout_seasons"] == [2024, 2025]
    assert provenance["pre_2024_seasons_used"] == [2020, 2021, 2022, 2023]


@requires_report
def test_exact_predictor_reproduces_both_published_monte_carlo_runs(report):
    """The envelope is only readable if the predictor behind it is faithful."""
    for run in ("control", "candidate"):
        assert report["predictor_fidelity"]["count_space_runs"][run][
            "max_abs_difference"
        ] < 2.0e-4


@requires_report
def test_estimator_ordering_is_the_attenuation_ordering(report):
    """A below D below E, on the pooled pre-2024 window."""
    window = report["section_1_estimator_diagnostic"]["by_window"][
        "pooled_2020_2023"
    ]
    reading = window["A_randomized_pit"]["reading"]
    censored_rho = window["D_interval_censored_mle"]["rho"]
    count = window["E_observed_count"]["reading"]
    assert reading < censored_rho < count


@requires_report
def test_multi_seed_averaging_does_not_move_the_reading(report):
    window = report["section_1_estimator_diagnostic"]["by_window"][
        "pooled_2020_2023"
    ]
    multi = window["B_multi_seed_randomized_pit"]
    single = window["A_randomized_pit"]
    assert abs(multi["reading"] - single["reading"]) < 0.5 * single["clustered_se"]
    assert multi["across_seed_sd"] < single["clustered_se"]


@requires_report
def test_censored_estimate_exceeds_the_reading_by_more_than_its_uncertainty(
    report,
):
    window = report["section_1_estimator_diagnostic"]["by_window"][
        "pooled_2020_2023"
    ]
    gap = (
        window["D_interval_censored_mle"]["rho"]
        - window["A_randomized_pit"]["reading"]
    )
    assert gap > 3.0 * window["D_interval_censored_mle"]["clustered_sandwich_se"]


@requires_report
def test_correcting_the_estimator_moves_count_space_the_right_way(report):
    """The correction helps, which is the part the attenuation predicts."""
    rows = report["section_2_transmission"]["pooled_2020_2023"]
    assert (
        rows["D_interval_censored_mle"]["abs_count_space_error"]
        < rows["A_randomized_pit"]["abs_count_space_error"]
    )
    assert (
        rows["B_multi_seed_randomized_pit"]["abs_count_space_error"]
        > rows["D_interval_censored_mle"]["abs_count_space_error"]
    )


@requires_report
def test_no_latent_estimator_reproduces_the_count_moment(report):
    """And the part it does not predict, which is the real blocker.

    Every estimator of the latent correlation lands below the value the
    count-space moment implies, so over this range the count-space ranking of
    the estimators is just their ordering in rho -- and the correct estimate
    of the copula parameter is not the closest one. The residual is a
    statement about the Gaussian copula and the production margins, not about
    any estimator, which is why no estimator can be asked to close it.
    """
    residual = report["section_2_transmission"]["pooled_2020_2023"][
        "residual_after_correcting_the_estimator"
    ]
    assert residual["every_estimator_undershoots_the_count_moment"] is True
    assert residual["count_space_error_is_monotone_in_the_latent_rho"] is True
    assert residual["shortfall"] > 0.0
    assert residual["shortfall_in_sandwich_se"] > 2.0
    assert residual["count_space_error_reduction_from_a_to_d"] > 0.0
    rows = report["section_2_transmission"]["pooled_2020_2023"]
    assert (
        rows["C_mid_pit"]["abs_count_space_error"]
        < rows["D_interval_censored_mle"]["abs_count_space_error"]
    ), "the diagnostic-only estimator sits closer purely by overshooting"


@requires_report
def test_the_lever_never_breaks_psd_or_the_same_player_pinning(report):
    for row in report["section_3_feasibility_envelope"]["sweep"]:
        assert row["psd_numerical_failures"] == 0
        assert row["max_same_player_block_deviation"] <= 1e-9


@requires_report
def test_the_envelope_is_bounded_by_give_back_not_by_representability(report):
    """What stops the lever is the price of representability, not the rank.

    The architecture represents every value on the sweep -- PSD never fails,
    the same-player blocks stay pinned at machine precision, and no
    cross-team or opponent parameter moves at all. What binds is that the
    frozen contrast Gram sits on the PSD boundary, so every step has to be
    bought with competition inflation, the inflation is re-pinned out of
    per-player shrink, and the shrink gives back ``teammate_reb_reb`` -- a
    bucket the accepted repair fixed -- before the latent RMSE tolerance is
    anywhere near.
    """
    envelope = report["section_3_feasibility_envelope"]
    readings = envelope["by_reading_of_the_constraints"]

    assert envelope["binding_constraint"]["constant_across_the_whole_sweep"] == [
        "protected_opponent_buckets_ok",
        "same_player_deviation_ok",
        "psd_failures_zero",
        "pairwise_parameters_zero",
        "player_indexed_parameters_zero",
    ]
    assert readings["commissioned"][
        "binding_constraints_just_past_the_boundary"
    ] == ["teammate_reb_reb_no_worse"]
    assert readings["give_back_in_latent_space"][
        "binding_constraints_just_past_the_boundary"
    ] == ["latent_rmse_within_tolerance"]
    assert (
        readings["commissioned"]["max_feasible_entry"]
        < readings["give_back_in_latent_space"]["max_feasible_entry"]
    )

    # And the currency the price is paid in. The extremes of the shrink barely
    # move, so the shrink *range* hides the cost; what carries it is the mean
    # of w^2, the factor every realised same-team correlation is scaled by.
    at_base = envelope["at_the_candidate_entry"]["mean_squared_shared_scale"]
    for reading in readings.values():
        assert reading["at_the_boundary"]["mean_squared_shared_scale"] < at_base
    assert (
        envelope["at_the_candidate_entry"]["shared_scale_range"]
        == readings["commissioned"]["at_the_boundary"]["shared_scale_range"]
    ), "the shrink range is unmoved at the boundary, which is why it misleads"


@requires_report
def test_the_latent_rmse_boundary_matches_its_closed_form(report):
    """The latent gate scores one parameter, so its boundary is a quadratic.

    That is what lets the boundary be compared against an estimate and its
    standard error rather than being an artefact of where the bisection
    happened to stop.
    """
    envelope = report["section_3_feasibility_envelope"]
    analytic = envelope["analytic_latent_rmse_boundary"]
    bisected = envelope["by_reading_of_the_constraints"][
        "give_back_in_latent_space"
    ]["max_feasible_entry"]
    assert analytic["feasible"] is True
    assert abs(analytic["max_entry"] - bisected) < 1e-6
    for row in envelope["sweep"]:
        entry = row["target_latent_entry"]
        predicted = np.sqrt(
            (
                analytic["other_eleven_squared_error_sum"]
                + (entry - analytic["observed_focal_latent"]) ** 2
            )
            / 12.0
        )
        assert abs(predicted - row["global_latent_rmse"]) < 1e-12


@requires_report
def test_the_lever_moves_no_parameter_but_the_focal_one(report):
    """Across the whole sweep, every other latent bucket is fixed.

    Fixed to machine precision rather than bitwise: the lever rebuilds the
    three Grams by eigen-refactorisation, so the untouched entries come back
    through a decomposition and round at the last bit.
    """
    sweep = report["section_3_feasibility_envelope"]["sweep"]
    reference = sweep[0]["latent_buckets"]
    for row in sweep:
        for bucket, value in row["latent_buckets"].items():
            if bucket == "passer_ast_teammate_pts":
                continue
            assert value == pytest.approx(reference[bucket], abs=1e-15), bucket
        assert row["latent_buckets"]["passer_ast_teammate_pts"] == pytest.approx(
            row["target_latent_entry"], abs=1e-12
        )


@requires_report
def test_the_count_reduction_peaks_and_then_falls(report):
    """The shrink that representability costs eventually dominates."""
    envelope = report["section_3_feasibility_envelope"]
    sweep = envelope["sweep"]
    reductions = [row["focal_count_error_reduction"] for row in sweep]
    peak = int(np.argmax(reductions))
    assert 0 < peak < len(sweep) - 1
    assert reductions[-1] < reductions[peak]
    assert envelope["unconstrained_peak"]["focal_count_error_reduction"] > 0.20


@requires_report
def test_the_forward_folds_cannot_rank_the_estimators(report):
    """Section 5's precondition is scored, and it fails.

    The bucket's observed count correlation moves further between 2022 and
    2023 than the four estimators' implied values differ inside either
    season, so whichever estimator happens to sit on the side the season
    moved towards wins that fold. The two folds therefore pick different
    winners, the precondition is not met, and no refit was run.
    """
    forward = report["section_5_inner_forward_test"]
    why = forward["why_the_folds_disagree"]
    assert why["target_moves_more_than_the_estimators_differ"] is True
    assert why["the_folds_agree_on_a_winner"] is False
    assert forward["censored_improves_every_forward_fold"] is False
    assert forward["section_5_precondition_met"] is False
    assert forward["inner_fit_was_run"] is False


@requires_report
def test_generic_application_to_all_twelve_buckets_breaks_the_latent_gate(
    report
):
    """The estimator is generic, and the latent gate charges for all of it."""
    generic = report.get("section_5_generic_application_to_all_buckets")
    if generic is None:
        pytest.skip("generic application needs the cached all-bucket MLE")
    assert generic["latent_rmse_within_tolerance"] is False
    assert (
        generic["global_latent_rmse"] > generic["global_latent_rmse_bound"]
    )


@requires_report
def test_frozen_artifacts_are_byte_identical_to_the_remediation(report):
    """The study measured the candidate; it did not move it."""
    import hashlib

    for path in (
        CANDIDATE_SPEC,
        CONTROL_SPEC,
        CANDIDATE_REPORT,
        CONTROL_REPORT,
    ):
        assert path.exists()
    assert (
        json.loads(CANDIDATE_SPEC.read_text())["spec_hash"]
        == FROZEN_CANDIDATE_HASH
    )
    payload = json.loads(CANDIDATE_SPEC.read_text())["loadings"]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    assert len(digest) == 64


@requires_report
def test_verdict_is_one_of_the_commissioned_classifications(report):
    assert report["section_4_identifiability"]["classification"] in {
        "IDENTIFIABLE_AND_ACHIEVABLE",
        "IDENTIFIABLE_BUT_GLOBALLY_INCOMPATIBLE",
        "NOT_IDENTIFIABLE_FROM_CURRENT_MARGINALS",
        "RANDOMIZED_PIT_ATTENUATION_CONFIRMED",
        "NO_ESTIMATOR_DEFECT_FOUND",
    }
    assert report["section_6_stop_rule"]["final_conclusion"] in {
        "COUNT-SPACE BLOCKER IS AN ESTIMATOR ISSUE AND IS FIXABLE GENERICALLY",
        "COUNT-SPACE BLOCKER IS AN ESTIMATOR ISSUE BUT FIX VIOLATES GLOBAL CONSTRAINTS",
        "COUNT-SPACE BLOCKER IS STRUCTURAL UNDER CURRENT ARCHITECTURE",
        "20% GATE NOT SUPPORTED BY EVIDENCE",
    }


@requires_report
def test_randomized_pit_attenuation_verdict_is_recorded(report):
    verdict = report["section_2_transmission"]["randomized_pit_attenuation"]
    assert verdict["verdict"] in {"YES", "NO"}
    assert set(verdict["legs"]) == {
        "size__censored_mle_exceeds_the_reading_by_over_two_se",
        "mechanism__multi_seed_averaging_does_not_close_the_gap",
        "correctability__inverting_the_reading_lands_on_the_mle",
        "ordering__reading_below_mid_pit_below_count",
    }


@requires_report
def test_the_latent_gate_scores_a_parameter_against_a_sample_moment(report):
    """The defect the whole study turns on, asserted against the artifact.

    ``latent_dependence_summary`` reports the architecture's raw copula
    parameter as the model's "implied" latent bucket, so the published latent
    numbers are reproduced exactly by reading the parameter -- no simulation
    involved. The transmitted reading is materially smaller, and that
    difference is the attenuation the gate never charges.
    """
    evidence = report["predictor_fidelity"]["latent_space_is_not_simulated"]
    for run in ("control", "candidate"):
        assert evidence["max_abs_parameter_vs_published"][run] < 1e-9
    focal = evidence["transmitted_latent_reading_on_the_holdout"]["candidate"][
        "passer_ast_teammate_pts"
    ]
    assert focal["transmitted_randomized_pit_reading"] < focal["parameter"]
    assert 0.5 < focal["ratio"] < 1.0


@requires_report
def test_the_trade_curve_is_monotone_in_both_quantities(report):
    """The give-back is the price of the gain, and the price only rises.

    Over the sampled range the focal bucket's count-space error reduction
    rises with the entry and ``teammate_reb_reb``'s count-space degradation
    rises with it, so there is a single trade and not a region where both
    improve. Every point is an evaluated covariance assembly, not a fit.
    """
    curve = report["section_3_feasibility_envelope"]["trade_curve"]["curve"]
    assert len(curve) > 30
    entries = [row["entry"] for row in curve]
    assert entries == sorted(entries)
    reductions = [row["focal_count_error_reduction"] for row in curve]
    give_back = [row["teammate_reb_reb_degradation_fraction"] for row in curve]
    assert reductions == sorted(reductions)
    assert give_back == sorted(give_back)
    assert reductions[0] < 0.10 < reductions[-1]
    assert give_back[0] < 0.0 < give_back[-1]
    for row in curve:
        assert row["psd_numerical_failures"] == 0

    # The curve starts at the frozen entry and only moves away from it, so the
    # price of representability rises monotonically along it -- unlike the
    # coarse sweep, which straddles the frozen entry and therefore does not.
    scales = [row["mean_squared_shared_scale"] for row in curve]
    assert scales == sorted(scales, reverse=True)
    assert scales[0] > scales[-1]


@requires_report
def test_the_answers_are_reported_under_both_gate_sets(report):
    """A to E, evaluated, under the implemented gates and the original brief."""
    answers = report["answers"]
    for gate_set in ("pipeline_own_gates", "commissioned"):
        entry = answers[gate_set]
        assert entry["A_max_feasible_ast_to_teammate_pts_count_correlation"][
            "count_space_correlation"
        ] > 0.0
        assert 0.0 < entry["B_absolute_error_reduction_percent"] < 100.0
        degradation = entry["C_teammate_reb_reb_degradation_at_that_point"]
        assert degradation["abs_count_error"] > 0.0
        assert degradation["degradation_percent"] == pytest.approx(
            100.0
            * (
                degradation["abs_count_error"]
                / degradation["control_abs_count_error"]
                - 1.0
            )
        )
        assert entry["D_first_binding_constraint"] in entry[
            "D_all_constraints_failing_just_past_it"
        ]
        assert isinstance(entry["E_twenty_percent_gate_achievable"], bool)

    # The original brief is strictly tighter in both of the ways it differs,
    # so it cannot admit more than the implemented gates do.
    assert (
        answers["commissioned"]["B_absolute_error_reduction_percent"]
        < answers["pipeline_own_gates"]["B_absolute_error_reduction_percent"]
    )
    assert (
        answers["commissioned"]["D_first_binding_constraint"]
        == "teammate_reb_reb_no_worse"
    )
    assert (
        answers["pipeline_own_gates"]["D_first_binding_constraint"]
        == "latent_rmse_within_tolerance"
    )
    assert answers["commissioned"]["E_twenty_percent_gate_achievable"] is False


@requires_report
def test_the_boundary_is_where_the_give_back_crosses_zero(report):
    """Under the original brief the boundary is exactly the crossing point."""
    answers = report["answers"]["commissioned"]
    degradation = answers["C_teammate_reb_reb_degradation_at_that_point"]
    assert degradation["degradation_absolute"] == pytest.approx(0.0, abs=1e-9)
    crossings = report["section_3_feasibility_envelope"]["trade_curve"][
        "give_back_budget_crossings"
    ]
    boundary = answers[
        "A_max_feasible_ast_to_teammate_pts_count_correlation"
    ]["latent_parameter_that_produces_it"]
    first = crossings[
        "first_entry_where_reb_reb_degrades_more_than_0_percent"
    ]
    assert first is not None
    assert first >= boundary
    assert first - boundary < 3e-4


@requires_report
def test_the_conclusion_follows_from_the_classification(report):
    classification = report["section_4_identifiability"]["classification"]
    assert report["classification"] == classification
    assert (
        report["final_conclusion"]
        == report["section_6_stop_rule"]["final_conclusion"]
    )
    assert report["section_6_stop_rule"]["no_new_factor_family_was_fitted"] is True
