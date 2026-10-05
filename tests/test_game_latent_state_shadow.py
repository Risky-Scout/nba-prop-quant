"""Tests for the game-level latent-state dependence shadow model, v1.

Hermetic: synthetic marginals and synthetic rosters only, no network, no
parquet inputs, no fitted production binaries. The shadow research scripts
that need real history are exercised separately and are not imported here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.distributions import (
    FittedMarginal,
    NegativeBinomialCalibrator,
    ZeroInflatedNegativeBinomialCalibrator,
)
from nba_prop_quant.research.game_latent_state import DEPENDENCE_MODEL_VERSION
from nba_prop_quant.research.game_latent_state.covariance import (
    SharedFactorLoadings,
    build_game_covariance,
    implied_within_player_correlation,
    min_eigenvalue,
    project_psd_rank,
)
from nba_prop_quant.research.game_latent_state.factors import (
    competition_gate,
    fit_shared_factors,
    incumbent_within_player_blocks,
    pair_moments,
    soft_threshold,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.pit import (
    PIT_EPSILON,
    deterministic_pit_uniform,
    gaussianize,
    randomized_pit,
)
from nba_prop_quant.research.game_latent_state.query import (
    PropLeg,
    evaluate_joint,
)
from nba_prop_quant.research.game_latent_state.simulator import (
    SUPPORTED_STATS,
    GameRoster,
    grid_ppf,
    simulate_game,
    tabulate_inverse_cdf,
)
from nba_prop_quant.research.game_latent_state.validation import (
    analytic_marginals,
    generate_joint_events,
    marginal_preservation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATS = ("pts", "reb", "ast")

MU = {"pts": 18.0, "reb": 6.5, "ast": 4.0, "stl": 1.0, "blk": 0.6, "fg3m": 2.0}


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


def nb_marginal(size: float = 8.0) -> FittedMarginal:
    model = NegativeBinomialCalibrator()
    model.size = size
    return FittedMarginal(kind="nb", model=model)


@pytest.fixture(scope="module")
def marginals() -> dict[str, FittedMarginal]:
    return {stat: nb_marginal(6.0 + index) for index, stat in enumerate(SUPPORTED_STATS)}


def make_roster(
    stats: tuple[str, ...] = SUPPORTED_STATS,
    players: tuple[tuple[int, int], ...] = ((10, 1), (11, 1), (12, 1), (20, 2), (21, 2), (22, 2)),
    game_id: int = 9001,
    home_team_id: int = 1,
) -> GameRoster:
    records = []
    for index, (player_id, team_id) in enumerate(players):
        record: dict[str, object] = {
            "player_id": player_id,
            "team_id": team_id,
            "expected_minutes": 30.0 - index,
            "is_home": int(team_id == home_team_id),
            "days_rest": 1,
            "b2b": 0,
            "pos_G": 1,
            "pos_F": 0,
            "pos_C": 0,
            "opp_pace_prior": 100.0,
            "role_bucket": "starter" if index < 4 else "bench",
        }
        for stat in stats:
            record[f"mu_selected_{stat}"] = MU[stat] * (1.0 - 0.05 * index)
            record[f"decay_prior_{stat}_rate"] = MU[stat] / 36.0
            record[f"kalman_prior_{stat}_rate"] = MU[stat] / 36.0
        records.append(record)
    return GameRoster(
        game_id=game_id,
        home_team_id=home_team_id,
        frame=pd.DataFrame(records),
        stats=stats,
    )


def make_loadings(stats: tuple[str, ...] = SUPPORTED_STATS) -> SharedFactorLoadings:
    rng = np.random.default_rng(11)
    game = np.round(rng.uniform(0.05, 0.25, size=(len(stats), 2)), 3)
    game[:, 1] *= np.where(np.arange(len(stats)) % 2 == 0, 1.0, -1.0)
    contrast = np.round(rng.uniform(-0.12, 0.12, size=len(stats)), 3)
    return SharedFactorLoadings(stats=stats, game=game, team_contrast=contrast)


def incumbent_block(stats: tuple[str, ...] = SUPPORTED_STATS, rho: float = 0.3) -> np.ndarray:
    n = len(stats)
    block = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(block, 1.0)
    return block


def within_player_map(
    roster: GameRoster, rho: float = 0.3
) -> dict[int, np.ndarray]:
    return {
        int(player_id): incumbent_block(roster.stats, rho)
        for player_id in roster.frame["player_id"]
    }


# ----------------------------------------------------------------------
# randomized PIT
# ----------------------------------------------------------------------


def test_randomized_pit_is_uniform_for_a_correct_discrete_marginal():
    """``u`` must be exactly Uniform(0,1), unlike the mid-PIT transform."""
    marginal = nb_marginal(7.0)
    rng = np.random.default_rng(5)
    mu = np.full(60_000, 11.0)
    frame = pd.DataFrame({"mu": mu})
    y = marginal.model.ppf(rng.random(len(mu)), 11.0)

    lower = marginal.cdf(y - 1, mu, frame)
    upper = marginal.cdf(y, mu, frame)
    v = rng.random(len(mu))
    u = randomized_pit(lower, upper, v)

    assert u.min() >= 0.0 and u.max() <= 1.0
    assert abs(float(np.mean(u)) - 0.5) < 0.01
    assert abs(float(np.var(u)) - 1.0 / 12.0) < 0.002

    # The mid-PIT transform the incumbent copula uses is under-dispersed on
    # the same data, which is why the shadow layer does not estimate
    # correlations from it.
    mid = lower + 0.5 * marginal.pmf(y, mu, frame)
    assert float(np.var(mid)) < float(np.var(u))


def test_randomized_pit_matches_its_closed_form():
    lower = np.array([0.1, 0.4, 0.0])
    upper = np.array([0.3, 0.9, 0.05])
    v = np.array([0.0, 0.5, 1.0])
    expected = lower + v * (upper - lower)
    assert np.allclose(randomized_pit(lower, upper, v), expected)


def test_randomized_pit_rejects_a_non_monotone_cdf_pair():
    with pytest.raises(ValueError, match="not monotone"):
        randomized_pit(np.array([0.5]), np.array([0.2]), np.array([0.5]))


def test_pit_seed_is_deterministic_and_order_independent():
    games = np.array([101, 102, 103, 104])
    players = np.array([7, 8, 9, 10])
    first = deterministic_pit_uniform(73, games, players, "pts")
    again = deterministic_pit_uniform(73, games, players, "pts")
    assert np.array_equal(first, again)

    order = np.array([2, 0, 3, 1])
    shuffled = deterministic_pit_uniform(73, games[order], players[order], "pts")
    assert np.allclose(shuffled, first[order])

    assert not np.allclose(
        deterministic_pit_uniform(74, games, players, "pts"), first
    )
    assert not np.allclose(
        deterministic_pit_uniform(73, games, players, "reb"), first
    )


def test_pit_draws_are_uniform_across_identities():
    games = np.repeat(np.arange(1_000), 20)
    players = np.tile(np.arange(20), 1_000)
    draws = deterministic_pit_uniform(73, games, players, "pts")
    assert draws.min() > 0.0 and draws.max() < 1.0
    assert abs(float(np.mean(draws)) - 0.5) < 0.01
    assert abs(float(np.var(draws)) - 1.0 / 12.0) < 0.002


# ----------------------------------------------------------------------
# Gaussianization bounds
# ----------------------------------------------------------------------


def test_gaussianize_bounds_and_reports_clipping():
    u = np.array([0.0, 1e-20, 0.5, 1.0 - 1e-20, 1.0])
    result = gaussianize(u)
    assert np.all(np.isfinite(result.z))
    assert abs(result.z[2]) < 1e-12
    assert result.clipped.tolist() == [True, True, False, True, True]
    assert result.clipped_fraction == pytest.approx(0.8)
    # The epsilon bound is |z| <= 7.04, a numerical guard rather than a
    # modelling choice. The two tails differ in the last few digits because
    # ``1 - eps`` loses precision relative to ``eps``, so the bound is taken
    # over both tails.
    bound = max(
        abs(float(norm.ppf(PIT_EPSILON))),
        abs(float(norm.ppf(1.0 - PIT_EPSILON))),
    )
    assert np.max(np.abs(result.z)) <= bound
    assert bound < 8.0


def test_gaussianize_is_exact_inside_the_epsilon():
    u = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    result = gaussianize(u)
    assert np.allclose(result.z, norm.ppf(u))
    assert not result.clipped.any()


def test_gaussianize_rejects_out_of_range_input():
    with pytest.raises(ValueError, match="outside"):
        gaussianize(np.array([1.5]))
    with pytest.raises(ValueError, match="non-finite"):
        gaussianize(np.array([np.nan]))


# ----------------------------------------------------------------------
# PSD covariance construction
# ----------------------------------------------------------------------


def test_game_covariance_is_psd_with_unit_diagonal():
    roster = make_roster()
    loadings = make_loadings()
    covariance = build_game_covariance(
        roster.dimensions(), loadings, within_player_map(roster)
    )
    assert covariance.size == len(roster.frame) * len(SUPPORTED_STATS)
    assert np.allclose(np.diag(covariance.correlation), 1.0)
    assert min_eigenvalue(covariance.correlation) > 0.0
    assert np.allclose(
        covariance.cholesky @ covariance.cholesky.T,
        covariance.correlation,
        atol=1e-8,
    )


def test_game_covariance_stays_psd_under_aggressive_loadings():
    """Even loadings far larger than any fitted value stay PSD by shrink."""
    roster = make_roster()
    loadings = SharedFactorLoadings(
        stats=SUPPORTED_STATS,
        game=np.full((len(SUPPORTED_STATS), 2), 0.65),
        team_contrast=np.full(len(SUPPORTED_STATS), 0.5),
    )
    covariance = build_game_covariance(
        roster.dimensions(), loadings, within_player_map(roster, rho=0.05)
    )
    assert covariance.shrink_applied
    assert covariance.min_residual_eigenvalue >= -1e-9
    assert min_eigenvalue(covariance.correlation) >= -1e-9
    assert np.allclose(np.diag(covariance.correlation), 1.0)


def test_project_psd_rank_truncates_and_factorizes():
    base = np.array([[0.3, 0.1, 0.0], [0.1, 0.2, 0.05], [0.0, 0.05, -0.4]])
    approximation, loadings = project_psd_rank(base, rank=2)
    assert loadings.shape == (3, 2)
    assert np.allclose(approximation, loadings @ loadings.T)
    assert min_eigenvalue(approximation) >= -1e-12
    assert np.linalg.matrix_rank(approximation, tol=1e-9) <= 2


def test_cross_player_blocks_match_the_fitted_pair_correlations():
    roster = make_roster()
    loadings = make_loadings()
    covariance = build_game_covariance(
        roster.dimensions(), loadings, within_player_map(roster)
    )
    dims = covariance.dimensions

    def block(player_a: int, player_b: int) -> np.ndarray:
        rows = [i for i, d in enumerate(dims) if d.player_id == player_a]
        cols = [i for i, d in enumerate(dims) if d.player_id == player_b]
        return covariance.correlation[np.ix_(rows, cols)]

    assert np.allclose(block(10, 11), loadings.same_team_correlation())
    assert np.allclose(block(10, 20), loadings.cross_team_correlation())


def test_game_covariance_requires_a_within_player_block():
    roster = make_roster()
    blocks = within_player_map(roster)
    blocks.pop(12)
    with pytest.raises(KeyError, match="player 12"):
        build_game_covariance(roster.dimensions(), make_loadings(), blocks)


# ----------------------------------------------------------------------
# same-player no-double-count contract
# ----------------------------------------------------------------------


def test_same_player_dependence_is_not_applied_twice():
    """The within-player block equals the incumbent copula block exactly.

    This is the Gate E contract. The shared game/team factors add cross-player
    covariance only; a player's own stat-by-stat correlation is the incumbent
    one to float64 precision, for every player in the game and for every
    shared-loading magnitude.
    """
    roster = make_roster()
    incumbent = within_player_map(roster, rho=0.35)

    for scale in (0.0, 0.5, 1.0, 2.0):
        loadings = SharedFactorLoadings(
            stats=SUPPORTED_STATS,
            game=scale * make_loadings().game,
            team_contrast=scale * make_loadings().team_contrast,
        )
        covariance = build_game_covariance(roster.dimensions(), loadings, incumbent)
        for player_id in roster.frame["player_id"]:
            implied = implied_within_player_correlation(covariance, int(player_id))
            assert np.allclose(implied, incumbent[int(player_id)], atol=1e-12), (
                f"same-player block drifted at scale={scale} for {player_id}"
            )


def test_adding_shared_factors_leaves_the_same_player_copula_untouched():
    """A same-player conjunction must price identically with and without the layer."""
    roster = make_roster(players=((10, 1), (11, 1), (20, 2), (21, 2)))
    incumbent = within_player_map(roster, rho=0.4)
    marginals = {stat: nb_marginal(7.0) for stat in SUPPORTED_STATS}

    independent = simulate_game(
        roster,
        marginals,
        SharedFactorLoadings.independent(SUPPORTED_STATS),
        incumbent,
        simulations=40_000,
        seed=4242,
    )
    coupled = simulate_game(
        roster,
        marginals,
        make_loadings(),
        incumbent,
        simulations=40_000,
        seed=4242,
    )

    legs = (
        PropLeg(10, "pts", "over", 17.5),
        PropLeg(10, "ast", "over", 3.5),
    )
    without = evaluate_joint(independent, legs)
    with_layer = evaluate_joint(coupled, legs)
    tolerance = 4.0 * np.hypot(without.standard_error, with_layer.standard_error)
    assert abs(without.probability - with_layer.probability) < tolerance


def test_incumbent_within_player_blocks_reindex_without_mutating_the_copula():
    copula = GaussianCopula(targets=list(SUPPORTED_STATS))
    rng = np.random.default_rng(3)
    base = rng.normal(size=(len(SUPPORTED_STATS), len(SUPPORTED_STATS)))
    global_corr = np.corrcoef(base @ base.T)
    copula.global_corr = global_corr
    copula.player_corr = {77: np.eye(len(SUPPORTED_STATS))}

    order = ("reb", "pts", "ast", "stl", "blk", "fg3m")
    blocks = incumbent_within_player_blocks(copula, order, [77, 78])

    assert np.allclose(blocks[77], np.eye(len(order)))
    assert np.allclose(blocks[78][0, 1], global_corr[1, 0])
    assert np.allclose(copula.global_corr, global_corr)
    assert set(copula.player_corr) == {77}


# ----------------------------------------------------------------------
# simulator
# ----------------------------------------------------------------------


def test_tabulated_inverse_cdf_matches_the_production_ppf_exactly():
    """The fast path is an evaluation-order change, not a distribution change."""
    roster = make_roster()
    row = roster.frame.iloc[0]
    u = np.concatenate(
        [
            np.linspace(1e-9, 1 - 1e-9, 20_001),
            np.array([1e-10, 0.5, 1.0 - 1e-10]),
        ]
    )

    nb = nb_marginal(9.0)
    cdf = tabulate_inverse_cdf(nb, 14.0, row)
    assert np.array_equal(grid_ppf(cdf, u), nb.ppf(u, 14.0, row).astype(float))

    zinb_frame = pd.DataFrame(
        {
            "expected_minutes": np.linspace(4.0, 34.0, 400),
            "is_home": np.tile([0, 1], 200),
        }
    )
    rng = np.random.default_rng(1)
    mu = np.clip(0.04 * zinb_frame["expected_minutes"].to_numpy(), 1e-3, None)
    y = rng.poisson(mu)
    zinb_model = ZeroInflatedNegativeBinomialCalibrator(
        inflation_features=["expected_minutes", "is_home"]
    ).fit(y, mu, zinb_frame)
    zinb = FittedMarginal(kind="zinb", model=zinb_model)

    zinb_row = zinb_frame.iloc[10]
    zinb_cdf = tabulate_inverse_cdf(zinb, 1.2, zinb_row)
    assert np.array_equal(grid_ppf(zinb_cdf, u), zinb.ppf(u, 1.2, zinb_row).astype(float))


def test_simulation_is_deterministic_in_the_seed(marginals):
    roster = make_roster()
    loadings = make_loadings()
    incumbent = within_player_map(roster)

    first = simulate_game(roster, marginals, loadings, incumbent, simulations=600, seed=99)
    again = simulate_game(roster, marginals, loadings, incumbent, simulations=600, seed=99)
    other = simulate_game(roster, marginals, loadings, incumbent, simulations=600, seed=100)

    assert np.array_equal(first.draws, again.draws)
    assert not np.array_equal(first.draws, other.draws)
    assert first.draws.shape == (600, len(roster.frame), len(SUPPORTED_STATS))


def test_simulation_shape_and_accessors(marginals):
    roster = make_roster()
    simulation = simulate_game(
        roster, marginals, make_loadings(), within_player_map(roster), simulations=400, seed=7
    )
    assert simulation.values(10, "pts").shape == (400,)
    assert np.allclose(
        simulation.combo(10, ("pts", "reb")),
        simulation.values(10, "pts") + simulation.values(10, "reb"),
    )
    with pytest.raises(KeyError):
        simulation.values(999, "pts")
    with pytest.raises(KeyError):
        simulation.values(10, "tov")
    long = simulation.to_long_frame()
    assert len(long) == 400 * len(roster.frame) * len(SUPPORTED_STATS)


def test_simulator_refuses_targets_production_does_not_model():
    """TOV has no production marginal, so it cannot enter a shadow roster."""
    assert "tov" not in SUPPORTED_STATS

    roster_frame = make_roster().frame
    with pytest.raises(KeyError, match="mu_selected_tov"):
        GameRoster(
            game_id=1,
            home_team_id=1,
            frame=roster_frame,
            stats=("pts", "tov"),
        )


def test_tabulated_and_exact_routes_agree_on_whole_games(marginals):
    roster = make_roster()
    kwargs = {
        "roster": roster,
        "marginals": marginals,
        "loadings": make_loadings(),
        "within_player": within_player_map(roster),
        "simulations": 500,
        "seed": 31,
    }
    fast = simulate_game(**kwargs, tabulated_inverse_cdf=True)
    exact = simulate_game(**kwargs, tabulated_inverse_cdf=False)
    assert np.array_equal(fast.draws, exact.draws)


def test_shared_factors_induce_cross_player_dependence(marginals):
    roster = make_roster()
    incumbent = within_player_map(roster)
    loadings = SharedFactorLoadings(
        stats=SUPPORTED_STATS,
        game=np.tile(np.array([[0.45, 0.0]]), (len(SUPPORTED_STATS), 1)),
        team_contrast=np.zeros(len(SUPPORTED_STATS)),
    )

    coupled = simulate_game(roster, marginals, loadings, incumbent, simulations=20_000, seed=8)
    independent = simulate_game(
        roster,
        marginals,
        SharedFactorLoadings.independent(SUPPORTED_STATS),
        incumbent,
        simulations=20_000,
        seed=8,
    )

    def teammate_corr(simulation):
        return float(
            np.corrcoef(simulation.values(10, "pts"), simulation.values(11, "pts"))[0, 1]
        )

    assert teammate_corr(coupled) > 0.12
    assert abs(teammate_corr(independent)) < 0.03


# ----------------------------------------------------------------------
# marginal preservation
# ----------------------------------------------------------------------


def test_marginal_preservation_holds_with_and_without_the_layer(marginals):
    roster = make_roster()
    incumbent = within_player_map(roster)
    reference = analytic_marginals(roster, marginals)

    for loadings in (make_loadings(), SharedFactorLoadings.independent(SUPPORTED_STATS)):
        simulation = simulate_game(
            roster, marginals, loadings, incumbent, simulations=30_000, seed=55
        )
        table = marginal_preservation(simulation, reference)
        assert len(table) == len(roster.frame) * len(SUPPORTED_STATS)
        assert table["max_abs_over_z"].max() < 5.0
        assert table["mean_z"].abs().max() < 5.0
        assert table["max_abs_quantile_error"].max() <= 1.0
        assert table["variance_relative_error"].abs().max() < 0.1


# ----------------------------------------------------------------------
# joint query engine
# ----------------------------------------------------------------------


def test_joint_query_counts_draws_satisfying_every_leg(marginals):
    roster = make_roster()
    simulation = simulate_game(
        roster,
        marginals,
        make_loadings(),
        within_player_map(roster),
        simulations=20_000,
        seed=2026,
    )
    legs = (
        PropLeg(10, "pts", "over", 14.5),
        PropLeg(10, "ast", "over", 2.5),
        PropLeg(11, "reb", "over", 4.5),
        PropLeg(20, "pts", "under", 19.5),
    )
    result = evaluate_joint(simulation, legs, artifact_version="test")

    manual = np.ones(simulation.simulations, dtype=bool)
    for leg in legs:
        values = simulation.values(leg.player_id, leg.stat)
        manual &= values > leg.line if leg.side == "over" else values < leg.line

    assert result.satisfying_draws == int(manual.sum())
    assert result.probability == pytest.approx(float(manual.mean()))
    assert result.standard_error > 0.0
    assert result.simulations == 20_000
    assert result.seed == 2026
    assert result.game_id == roster.game_id
    assert result.dependence_model_version == DEPENDENCE_MODEL_VERSION
    assert result.artifact_version == "test"
    assert len(result.legs) == 4
    payload = result.to_payload()
    assert json.loads(json.dumps(payload))["simulations"] == 20_000


def test_joint_query_supports_arbitrary_leg_counts(marginals):
    roster = make_roster()
    simulation = simulate_game(
        roster,
        marginals,
        make_loadings(),
        within_player_map(roster),
        simulations=8_000,
        seed=3,
    )
    pool = (
        PropLeg(10, "pts", "over", 10.5),
        PropLeg(11, "reb", "over", 3.5),
        PropLeg(20, "ast", "under", 6.5),
        PropLeg(21, "pts", "over", 8.5),
        PropLeg(12, "reb", "under", 9.5),
    )
    previous = 1.1
    for count in range(1, len(pool) + 1):
        result = evaluate_joint(simulation, pool[:count])
        assert 0.0 <= result.probability <= previous + 1e-12
        previous = result.probability


def test_joint_query_rejects_an_empty_conjunction(marginals):
    roster = make_roster()
    simulation = simulate_game(
        roster, marginals, make_loadings(), within_player_map(roster), simulations=100, seed=1
    )
    with pytest.raises(ValueError, match="at least one leg"):
        evaluate_joint(simulation, [])


def test_integer_lines_are_reported_as_pushes(marginals):
    roster = make_roster()
    simulation = simulate_game(
        roster, marginals, make_loadings(), within_player_map(roster), simulations=5_000, seed=6
    )
    pushable = evaluate_joint(simulation, (PropLeg(10, "pts", "over", 18.0),))
    assert pushable.push_fraction > 0.0
    half = evaluate_joint(simulation, (PropLeg(10, "pts", "over", 18.5),))
    assert half.push_fraction == 0.0


def test_combo_legs_use_production_component_sums(marginals):
    roster = make_roster()
    simulation = simulate_game(
        roster, marginals, make_loadings(), within_player_map(roster), simulations=4_000, seed=12
    )
    result = evaluate_joint(simulation, (PropLeg(10, "points_rebounds", "over", 24.5),))
    expected = float(
        np.mean(simulation.combo(10, ("pts", "reb")) > 24.5)
    )
    assert result.probability == pytest.approx(expected)


# ----------------------------------------------------------------------
# factor estimation, leakage, sparse fallback
# ----------------------------------------------------------------------


def synthetic_residual_frame(
    games: int = 400,
    seed: int = 17,
    game_sd: float = 0.3,
    contrast_sd: float = 0.2,
    competition_sd: float = 0.0,
    competition_stats: Sequence[str] = ("reb",),
    team_size: int = 8,
) -> pd.DataFrame:
    """Residuals generated from a known shared-factor structure.

    ``competition_sd`` injects the within-team zero-sum family: each team draws
    one value per slot, centers them, and rescales by ``sqrt(team_size)`` so the
    induced teammate covariance is ``-competition_sd ** 2`` regardless of roster
    size. It loads only on ``competition_stats`` -- the rebound-competition
    story -- because a loading spread evenly over every stat leaves the
    same-team block rank-one positive and therefore still PSD. Concentrating it
    on one stat is what drives that stat's teammate correlation negative, and
    that is the only thing here that can make the same-team block indefinite.
    """
    rng = np.random.default_rng(seed)
    records = []
    for game in range(games):
        shared = rng.normal()
        contrast = rng.normal()
        for team_index, team in enumerate((1, 2)):
            side = 1.0 if team_index == 0 else -1.0
            if competition_sd > 0.0:
                draw = rng.normal(size=team_size)
                competition = (
                    competition_sd * np.sqrt(team_size) * (draw - draw.mean())
                )
            else:
                competition = np.zeros(team_size)
            for slot in range(team_size):
                base = {
                    "game_id": 1000 + game,
                    "season": 2018 + game % 5,
                    "team_id": team,
                    "player_id": team * 100 + slot,
                    "role_bucket": "starter" if slot < 5 else "bench",
                }
                for stat in STATS:
                    loading = 1.0 if stat in competition_stats else 0.0
                    base[f"z_{stat}"] = (
                        game_sd * shared
                        + side * contrast_sd * contrast
                        + loading * competition[slot]
                        + rng.normal()
                    )
                records.append(base)
    return pd.DataFrame(records)


def test_pair_moments_recovers_a_known_shared_structure():
    frame = synthetic_residual_frame()
    standardized, _ = standardize_residuals(frame, STATS)
    moments = pair_moments(standardized, STATS, bootstrap=0)

    expected_same = 0.3**2 + 0.2**2
    expected_cross = 0.3**2 - 0.2**2
    scale = 1.0 + 0.3**2 + 0.2**2

    assert moments.same_team[0, 1] == pytest.approx(expected_same / scale, abs=0.02)
    assert moments.cross_team[0, 1] == pytest.approx(expected_cross / scale, abs=0.02)
    assert moments.games == 400
    assert moments.same_team_pairs > 0 and moments.cross_team_pairs > 0


def test_pair_moments_never_touch_same_player_pairs():
    """A player whose own stats are perfectly correlated must not move S or X."""
    frame = synthetic_residual_frame(games=150, seed=4)
    baseline, _ = standardize_residuals(frame, STATS)
    base_moments = pair_moments(baseline, STATS)

    inflated = frame.copy()
    anchor = inflated["z_pts"].to_numpy()
    for stat in STATS:
        inflated[f"z_{stat}"] = anchor
    inflated, _ = standardize_residuals(inflated, STATS)
    inflated_moments = pair_moments(inflated, STATS)

    # Same-player correlation is now 1.0 everywhere, yet the cross-player
    # statistics only reflect what pts alone contributed.
    assert np.allclose(
        inflated_moments.same_team,
        np.full((3, 3), base_moments.same_team[0, 0]),
        atol=1e-9,
    )


def test_fitted_loadings_reproduce_the_shrunk_pair_matrices():
    frame = synthetic_residual_frame(games=500, seed=21)
    standardized, _ = standardize_residuals(frame, STATS)
    fit = fit_shared_factors(standardized, STATS, bootstrap=50, seed=5)

    diagnostics = fit.diagnostics()
    assert diagnostics["same_team_fit_rmse"] < 0.01
    assert diagnostics["cross_team_fit_rmse"] < 0.01
    assert fit.loadings.k_game == 2
    assert fit.loadings.game.shape == (len(STATS), 2)

    # The structural PSD requirements sit on the three factor Grams A, B and
    # Q, each of which is a loading matrix times its transpose and so is PSD
    # by construction. The observable blocks S = A + B - Q and X = A - B are
    # not themselves required to be PSD: S is a cross-player block bounded
    # below by the roster-size constraint, and X is an off-diagonal block.
    game_gram = fit.loadings.game @ fit.loadings.game.T
    contrast_gram = np.outer(fit.loadings.team_contrast, fit.loadings.team_contrast)
    assert min_eigenvalue(game_gram) >= -1e-10
    assert min_eigenvalue(contrast_gram) >= -1e-10
    assert min_eigenvalue(fit.loadings.competition_gram()) >= -1e-10


def test_competition_family_stays_off_when_teammates_do_not_compete():
    """A purely additive structure must not be given a competition factor.

    The same-team block's smallest eigenvalue reads as negative here purely
    from estimation noise, so a naive `min eig(S_hat) < 0` test would activate
    the family and then fit the noise exactly. The bias-corrected bound has to
    see through that.
    """
    frame = synthetic_residual_frame(games=500, seed=21, competition_sd=0.0)
    standardized, _ = standardize_residuals(frame, STATS)
    moments = pair_moments(standardized, STATS, bootstrap=200, seed=5)
    shrunk = soft_threshold(moments.same_team, moments.same_team_se)

    active, evidence = competition_gate(moments, shrunk)

    assert evidence["min_eigenvalue_point_estimate"] < 0.0
    assert evidence["min_eigenvalue_pivotal_upper_bound"] > 0.0
    assert active is False

    fit = fit_shared_factors(standardized, STATS, bootstrap=200, seed=5)
    assert fit.r_competition == 0
    assert np.allclose(fit.loadings.competition_gram(), 0.0, atol=1e-12)
    assert fit.diagnostics()["competition_family_active"] is False


def test_competition_family_activates_on_genuine_rebound_competition():
    """Negative teammate rebound correlation must be representable."""
    frame = synthetic_residual_frame(
        games=500, seed=21, competition_sd=0.55, competition_stats=("reb",)
    )
    standardized, _ = standardize_residuals(frame, STATS)
    moments = pair_moments(standardized, STATS, bootstrap=200, seed=5)
    reb = STATS.index("reb")

    # The injected structure really does make teammate rebounds compete.
    assert moments.same_team[reb, reb] < 0.0

    shrunk = soft_threshold(moments.same_team, moments.same_team_se)
    active, evidence = competition_gate(moments, shrunk)
    assert evidence["min_eigenvalue_pivotal_upper_bound"] < 0.0
    assert active is True

    fit = fit_shared_factors(standardized, STATS, bootstrap=200, seed=5)
    assert fit.r_competition > 0
    assert min_eigenvalue(fit.loadings.competition_gram()) >= -1e-10

    # A purely additive model cannot produce a negative same-team entry, so
    # reproducing this block is exactly what the competition family buys.
    assert fit.loadings.same_team_correlation()[reb, reb] < 0.0
    assert fit.diagnostics()["same_team_fit_rmse"] < 0.01


def test_competition_gate_abstains_without_enough_bootstrap_draws():
    """Too few draws is not evidence; the family stays off."""
    frame = synthetic_residual_frame(
        games=200, seed=21, competition_sd=0.55, competition_stats=("reb",)
    )
    standardized, _ = standardize_residuals(frame, STATS)
    moments = pair_moments(standardized, STATS, bootstrap=0)
    shrunk = soft_threshold(moments.same_team, moments.same_team_se)

    active, evidence = competition_gate(moments, shrunk)
    assert active is False
    assert evidence["available_draws"] == 0.0


def test_soft_threshold_zeroes_insignificant_entries():
    estimate = np.array([[0.10, 0.01], [0.01, -0.20]])
    error = np.array([[0.01, 0.02], [0.02, 0.01]])
    shrunk = soft_threshold(estimate, error, z_crit=1.96)
    assert shrunk[0, 1] == 0.0
    assert shrunk[0, 0] == pytest.approx(0.10 - 1.96 * 0.01)
    assert shrunk[1, 1] == pytest.approx(-(0.20 - 1.96 * 0.01))


def test_independent_loadings_are_the_conditional_independence_baseline():
    loadings = SharedFactorLoadings.independent(SUPPORTED_STATS)
    assert np.allclose(loadings.same_team_correlation(), 0.0)
    assert np.allclose(loadings.cross_team_correlation(), 0.0)


def test_no_dependence_parameter_is_indexed_by_player_or_pair():
    """Gate G: the fitted structure cannot overfit a specific player pair."""
    frame = synthetic_residual_frame(games=200, seed=9)
    standardized, _ = standardize_residuals(frame, STATS)
    fit = fit_shared_factors(standardized, STATS, bootstrap=0)
    payload = fit.loadings.to_payload()

    assert set(payload) == {
        "stats",
        "k_game",
        "r_competition",
        "game_loadings",
        "team_contrast_loadings",
        "competition_loadings",
        "role_scale",
    }
    assert np.shape(payload["game_loadings"]) == (len(STATS), 2)
    assert np.shape(payload["team_contrast_loadings"]) == (len(STATS),)
    assert payload["role_scale"] == {}
    if payload["competition_loadings"] is not None:
        assert np.shape(payload["competition_loadings"])[0] == len(STATS)


def test_unseen_and_sparse_players_fall_back_without_refitting(marginals):
    """A roster of players the fit never saw still simulates coherently."""
    copula = GaussianCopula(targets=list(SUPPORTED_STATS))
    rng = np.random.default_rng(2)
    base = rng.normal(size=(len(SUPPORTED_STATS), 4 * len(SUPPORTED_STATS)))
    copula.global_corr = np.corrcoef(base)
    copula.player_corr = {}

    roster = make_roster(
        players=((9_000_001, 1), (9_000_002, 1), (9_000_003, 2), (9_000_004, 2)),
        game_id=424_242,
    )
    blocks = incumbent_within_player_blocks(
        copula, SUPPORTED_STATS, roster.frame["player_id"].astype(int)
    )
    assert all(np.allclose(block, copula.global_corr) for block in blocks.values())

    frame = synthetic_residual_frame(games=200, seed=33)
    standardized, _ = standardize_residuals(frame, STATS)
    fit = fit_shared_factors(standardized, STATS, bootstrap=0)
    loadings = SharedFactorLoadings(
        stats=SUPPORTED_STATS,
        game=np.vstack([fit.loadings.game, np.zeros((3, 2))]),
        team_contrast=np.concatenate([fit.loadings.team_contrast, np.zeros(3)]),
    )

    simulation = simulate_game(
        roster, marginals, loadings, blocks, simulations=2_000, seed=77
    )
    assert np.all(np.isfinite(simulation.draws))
    assert simulation.covariance.min_eigenvalue > 0.0
    assert not simulation.covariance.shrink_applied


def test_zero_minute_player_simulates_without_numerical_failure(marginals):
    roster = make_roster(players=((10, 1), (11, 1), (20, 2), (21, 2)))
    roster.frame.loc[1, "expected_minutes"] = 0.0
    for stat in SUPPORTED_STATS:
        roster.frame.loc[1, f"mu_selected_{stat}"] = 1e-9

    simulation = simulate_game(
        roster,
        marginals,
        make_loadings(),
        within_player_map(roster),
        simulations=1_000,
        seed=4,
    )
    bench = simulation.draws[:, 1, :]
    assert np.all(np.isfinite(bench))
    assert np.all(bench >= 0.0)
    assert bench.max() == 0.0


def test_roster_change_does_not_change_the_fitted_parameters(marginals):
    """Adding, removing or reordering players reuses the same loadings."""
    loadings = make_loadings()
    incumbent_rho = 0.3

    small = make_roster(players=((10, 1), (11, 1), (20, 2), (21, 2)))
    large = make_roster(
        players=((10, 1), (11, 1), (12, 1), (13, 1), (20, 2), (21, 2), (22, 2))
    )
    for roster in (small, large):
        simulation = simulate_game(
            roster,
            marginals,
            loadings,
            within_player_map(roster, incumbent_rho),
            simulations=500,
            seed=21,
        )
        assert simulation.covariance.min_eigenvalue > 0.0
        assert np.all(np.isfinite(simulation.draws))

    reordered = make_roster(players=((21, 2), (10, 1), (20, 2), (11, 1)))
    reordered_sim = simulate_game(
        reordered,
        marginals,
        loadings,
        within_player_map(reordered, incumbent_rho),
        simulations=500,
        seed=21,
    )
    base = simulate_game(
        small,
        marginals,
        loadings,
        within_player_map(small, incumbent_rho),
        simulations=500,
        seed=21,
    )
    def first_dimension(simulation, player_id: int) -> int:
        return next(
            index
            for index, dimension in enumerate(simulation.covariance.dimensions)
            if dimension.player_id == player_id
        )

    # Different dimension order means a different draw assignment, but the
    # implied correlation between a given pair of coordinates is identical.
    assert np.isclose(
        reordered_sim.covariance.correlation[
            first_dimension(reordered_sim, 10), first_dimension(reordered_sim, 11)
        ],
        base.covariance.correlation[0, len(SUPPORTED_STATS)],
    )


def test_standardization_can_be_locked_to_training_moments():
    """A validation season must not be standardized with its own moments."""
    frame = synthetic_residual_frame(games=60, seed=6)
    _, moments = standardize_residuals(frame, STATS)
    shifted = frame.copy()
    for stat in STATS:
        shifted[f"z_{stat}"] = shifted[f"z_{stat}"] + 5.0
    locked, reused = standardize_residuals(shifted, STATS, moments=moments)
    assert reused == moments
    assert float(np.mean(locked["zs_pts"])) > 4.0


# ----------------------------------------------------------------------
# joint event generation
# ----------------------------------------------------------------------


def test_generated_events_cover_every_leg_count_and_grade_from_box_scores(marginals):
    roster = make_roster()
    reference = analytic_marginals(roster, marginals)

    observations = roster.frame.copy()
    rng = np.random.default_rng(13)
    for stat in SUPPORTED_STATS:
        observations[f"y_{stat}"] = rng.poisson(
            observations[f"mu_selected_{stat}"].to_numpy()
        )

    events = generate_joint_events(
        observations,
        reference,
        SUPPORTED_STATS,
        game_id=roster.game_id,
        seed=5,
        events_per_family=3,
    )
    assert events
    assert {event.n_legs for event in events} == {2, 3, 4}
    assert all(event.realized in (0, 1) for event in events)
    assert all(
        float(leg.line) % 1 == 0.5 for event in events for leg in event.legs
    )

    for event in events:
        expected = 1
        for leg in event.legs:
            value = float(
                observations.loc[
                    observations["player_id"] == leg.player_id, f"y_{leg.stat}"
                ].iloc[0]
            )
            expected &= int(value > leg.line) if leg.side == "over" else int(value < leg.line)
        assert event.realized == expected


def test_generated_event_lines_do_not_depend_on_realized_values(marginals):
    """Changing the box score must not change a single generated line."""
    roster = make_roster()
    reference = analytic_marginals(roster, marginals)
    observations = roster.frame.copy()
    rng = np.random.default_rng(21)
    for stat in SUPPORTED_STATS:
        observations[f"y_{stat}"] = rng.poisson(
            observations[f"mu_selected_{stat}"].to_numpy()
        )

    first = generate_joint_events(
        observations, reference, SUPPORTED_STATS, game_id=1, seed=8
    )

    perturbed = observations.copy()
    for stat in SUPPORTED_STATS:
        perturbed[f"y_{stat}"] = perturbed[f"y_{stat}"] + 7
    second = generate_joint_events(
        perturbed, reference, SUPPORTED_STATS, game_id=1, seed=8
    )

    assert [leg for event in first for leg in event.legs] == [
        leg for event in second for leg in event.legs
    ]
    assert [event.realized for event in first] != [event.realized for event in second]
