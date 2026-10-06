"""Tests for the final upstream remediation layer.

SHADOW / RESEARCH ONLY.

The claims worth a test here are the ones the gate report rests on and that a
reader cannot check by eye:

* the control spec reproduces the accepted repair *bitwise*, so an A/B against
  it measures the remediation and not a pipeline difference,
* the dependence temperature multiplies every cross-player block by exactly
  ``lambda`` while leaving each player's own block exactly alone,
* the analytic conjunction price agrees with the Monte Carlo the confirmatory
  run uses, because the temperature search replaces one with the other,
* the transmission coefficients are the conditional moments they claim to be,
* the combination charges between-source disagreement to the bridge.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nba_prop_quant.research.game_latent_state.covariance import (
    SharedFactorLoadings,
    build_game_covariance,
)
from nba_prop_quant.research.game_latent_state.factors import (
    pair_moments,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.query import PropLeg, evaluate_joint
from nba_prop_quant.research.game_latent_state.remediation import (
    CONTROL_SPEC,
    RemediationSpec,
    coverage_table,
    fit_log_shrunk_role_scales,
    fit_remediated_factors,
    huber_scale,
    select_within_tie_band,
    temper_loadings,
)
from nba_prop_quant.research.game_latent_state.repair import (
    SHRINKAGE_EMPIRICAL_BAYES,
    RepairSpec,
    fit_repaired_factors,
)
from nba_prop_quant.research.game_latent_state.simulator import (
    GameRoster,
    simulate_game,
)
from nba_prop_quant.research.game_latent_state.temperature import (
    dimension_columns,
    independent_product,
    latent_orthant,
    orthant_probability,
)
from nba_prop_quant.research.game_latent_state.transmission import (
    SourcePair,
    accumulate_per_game,
    bridge_forward,
    bridge_inverse,
    combine_sources,
    conditional_hermite_moments,
    homogeneity_test,
    transmission_coefficient_columns,
)

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")


# ----------------------------------------------------------------------
# synthetic residual frame
# ----------------------------------------------------------------------


def _synthetic_residuals(games: int = 160, seed: int = 11) -> pd.DataFrame:
    """A small game-structured residual frame with real cross-player signal."""
    rng = np.random.default_rng(seed)
    rows = []
    roles = ("starter", "rotation", "bench")
    for game in range(games):
        for team in range(2):
            shared = rng.normal(size=len(STATS)) * 0.25
            for slot in range(5):
                player = team * 100 + slot
                own = rng.normal(size=len(STATS))
                values = 0.3 * shared + np.sqrt(1.0 - 0.09) * own
                rows.append(
                    {
                        "game_id": game,
                        "team_id": team,
                        "player_id": player,
                        "season": 2020 + game % 2,
                        "role_bucket": roles[slot % 3],
                        **{f"z_{stat}": values[index] for index, stat in enumerate(STATS)},
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def standardized() -> pd.DataFrame:
    frame, _ = standardize_residuals(_synthetic_residuals(), STATS)
    return frame


@pytest.fixture(scope="module")
def moments(standardized: pd.DataFrame):
    return pair_moments(standardized, STATS, bootstrap=40, seed=5)


# ----------------------------------------------------------------------
# the control really is the control
# ----------------------------------------------------------------------


def test_the_control_spec_reproduces_the_accepted_repair_bitwise(
    standardized: pd.DataFrame,
    moments,
) -> None:
    """Every remediation A/B is read against this, so it must be exact.

    An approximate match would leave every gate margin partly attributable to
    a pipeline difference rather than to the remediation being measured.
    """
    # Both pool their own moments from the same frame at the same bootstrap
    # and seed, so the comparison covers the moment step too.
    remediated = fit_remediated_factors(
        standardized, STATS, spec=CONTROL_SPEC, bootstrap=40, seed=5
    )
    repaired = fit_repaired_factors(
        standardized,
        STATS,
        spec=RepairSpec(
            # The accepted candidate, from
            # research/game_latent_state_bucket_repair/bucket_repair_candidate.json
            name="C_rank_k6_r6_eb_block_diagonal",
            family="C_rank",
            shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
            k_game=CONTROL_SPEC.k_game,
            r_contrast=CONTROL_SPEC.r_contrast,
            eb_family=CONTROL_SPEC.eb_family,
            role_column=CONTROL_SPEC.role_column,
        ),
        bootstrap=40,
        seed=5,
    )

    assert remediated.loadings.game == pytest.approx(repaired.loadings.game, abs=0.0)
    assert remediated.loadings.team_contrast == pytest.approx(
        repaired.loadings.team_contrast, abs=0.0
    )
    assert remediated.loadings.role_scale == repaired.loadings.role_scale


# ----------------------------------------------------------------------
# item 5: the temperature is exactly multiplicative
# ----------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_temperature_scales_every_cross_player_block_by_exactly_lambda(
    standardized: pd.DataFrame,
    moments,
    temperature: float,
) -> None:
    full = fit_remediated_factors(
        standardized, STATS, spec=CONTROL_SPEC, moments=moments
    ).loadings
    tempered = temper_loadings(full, temperature)

    for role_a in (None, "starter", "bench"):
        for role_b in (None, "starter", "bench"):
            assert tempered.same_team_correlation_for_roles(
                role_a, role_b
            ) == pytest.approx(
                temperature * full.same_team_correlation_for_roles(role_a, role_b),
                abs=1e-15,
            )
    assert tempered.cross_team_correlation() == pytest.approx(
        temperature * full.cross_team_correlation(), abs=1e-15
    )


@pytest.mark.parametrize("temperature", [0.0, 0.4, 1.0])
def test_the_temperature_leaves_each_player_own_block_exactly_pinned(
    standardized: pd.DataFrame,
    moments,
    temperature: float,
) -> None:
    """The brief's constraint: no temperature may move a player's own margin.

    ``build_game_covariance`` subtracts the shared gram from the incumbent
    block before adding the shared factors back, so the diagonal block is
    algebraically the incumbent's at every temperature. This is the assertion
    that the algebra holds numerically too.
    """
    full = fit_remediated_factors(
        standardized, STATS, spec=CONTROL_SPEC, moments=moments
    ).loadings
    tempered = temper_loadings(full, temperature)

    rng = np.random.default_rng(3)
    base = rng.normal(size=(len(STATS), len(STATS)))
    gram = base @ base.T + np.eye(len(STATS)) * 3.0
    block = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
    within = {player: block for player in (1, 2, 3, 4)}

    # The dimensions come through the public roster path rather than being
    # built by hand, so their ordering is the one the simulator uses.
    roster = GameRoster(
        game_id=1,
        home_team_id=0,
        frame=pd.DataFrame(
            {
                "player_id": list(within),
                "team_id": [0, 0, 1, 1],
                **{f"mu_selected_{stat}": [4.0, 5.0, 6.0, 7.0] for stat in STATS},
            }
        ),
        stats=STATS,
    )
    covariance = build_game_covariance(
        dimensions=roster.dimensions(),
        loadings=tempered,
        within_player=within,
    )
    size = len(STATS)
    for index in range(len(within)):
        start = index * size
        observed = covariance.correlation[start : start + size, start : start + size]
        assert observed == pytest.approx(block, abs=1e-9)


# ----------------------------------------------------------------------
# item 5: the analytic price is the Monte Carlo price
# ----------------------------------------------------------------------


def test_the_analytic_conjunction_price_matches_the_monte_carlo_one() -> None:
    """The temperature search swaps simulation for an orthant integral.

    That swap is only legitimate if the two price the *same* event, so this
    simulates a game and checks the counted frequency against the integral to
    inside the counting standard error. The simulator is the reference because
    it is what the confirmatory run uses.
    """
    pytest.importorskip("scipy")

    from nba_prop_quant.research.game_latent_state.validation import (
        AnalyticMarginal,
    )

    size = len(STATS)
    rng = np.random.default_rng(17)
    base = rng.normal(size=(size, size)) * 0.4
    gram = base @ base.T
    block = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
    within = {player: block for player in (10, 11, 20, 21)}

    loadings = SharedFactorLoadings(
        stats=STATS,
        game=np.full((size, 1), 0.28),
        team_contrast=np.full((size, 1), 0.16),
    )
    roster = GameRoster(
        game_id=7,
        home_team_id=0,
        frame=pd.DataFrame(
            {
                "player_id": [10, 11, 20, 21],
                "team_id": [0, 0, 1, 1],
                **{f"mu_selected_{stat}": [8.0, 6.0, 7.0, 5.0] for stat in STATS},
            }
        ),
        stats=STATS,
    )

    # A Poisson-like tabulated CDF per player-stat, standing in for the
    # production marginal. The orthant translation only ever reads this table,
    # so a stand-in exercises exactly the same code path.
    from scipy.stats import poisson

    reference: dict[tuple[int, str], AnalyticMarginal] = {}
    cdfs: dict[tuple[int, str], np.ndarray] = {}
    for player, mu in zip((10, 11, 20, 21), (8.0, 6.0, 7.0, 5.0)):
        for stat in STATS:
            grid = np.arange(0, 60)
            cdf = np.clip(poisson.cdf(grid, mu), 0.0, 1.0)
            reference[(player, stat)] = AnalyticMarginal(cdf=cdf)
            cdfs[(player, stat)] = cdf

    covariance = build_game_covariance(
        dimensions=roster.dimensions(), loadings=loadings, within_player=within
    )
    columns = dimension_columns(covariance)

    class _Marginal:
        """Minimal stand-in exposing the production ``ppf`` contract."""

        def __init__(self, table: dict[int, np.ndarray], stat: str) -> None:
            self.table = table
            self.stat = stat

        def ppf(self, u, mu, frame):
            player = int(frame["player_id"])
            return np.searchsorted(
                self.table[(player, self.stat)], np.asarray(u), side="left"
            ).astype(float)

    marginals = {stat: _Marginal(cdfs, stat) for stat in STATS}
    simulation = simulate_game(
        roster,
        marginals=marginals,  # type: ignore[arg-type]
        loadings=loadings,
        within_player=within,
        simulations=400_000,
        seed=99,
        tabulated_inverse_cdf=False,
    )

    cases = [
        (PropLeg(10, "pts", "over", 7.5), PropLeg(11, "reb", "over", 5.5)),
        (PropLeg(10, "pts", "over", 7.5), PropLeg(20, "ast", "under", 7.5)),
        (
            PropLeg(10, "pts", "over", 7.5),
            PropLeg(10, "reb", "under", 8.5),
            PropLeg(11, "ast", "over", 5.5),
        ),
        (
            PropLeg(10, "pts", "over", 8.5),
            PropLeg(11, "reb", "over", 5.5),
            PropLeg(20, "ast", "under", 7.5),
            PropLeg(21, "fg3m", "over", 4.5),
        ),
    ]
    for legs in cases:
        counted = evaluate_joint(simulation, legs)
        orthant = latent_orthant(legs, columns, reference)
        analytic = orthant_probability(orthant, covariance.correlation)
        tolerance = max(4.0 * counted.standard_error, 2e-3)
        assert analytic == pytest.approx(counted.probability, abs=tolerance)


def test_the_independent_product_matches_the_counted_leg_marginals() -> None:
    """A sanity rail on the threshold translation itself.

    If the per-leg thresholds were wrong, the product of the leg marginals
    would disagree with the simulator's even when a coupled probability
    happened to land close by luck.
    """
    from nba_prop_quant.research.game_latent_state.validation import (
        AnalyticMarginal,
    )
    from scipy.stats import poisson

    grid = np.arange(0, 60)
    reference = {
        (1, "pts"): AnalyticMarginal(cdf=np.clip(poisson.cdf(grid, 9.0), 0, 1)),
        (1, "reb"): AnalyticMarginal(cdf=np.clip(poisson.cdf(grid, 4.0), 0, 1)),
    }
    columns = {(1, "pts"): 0, (1, "reb"): 1}
    legs = (PropLeg(1, "pts", "over", 8.5), PropLeg(1, "reb", "under", 4.5))
    orthant = latent_orthant(legs, columns, reference)

    expected = float(
        (1.0 - poisson.cdf(8, 9.0)) * poisson.cdf(4, 4.0)
    )
    assert independent_product(orthant) == pytest.approx(expected, abs=1e-12)
    assert orthant_probability(orthant, np.eye(2)) == pytest.approx(expected, abs=1e-9)


# ----------------------------------------------------------------------
# item 4: the transmission coefficients
# ----------------------------------------------------------------------


def test_the_conditional_hermite_moment_is_the_truncated_normal_moment() -> None:
    """``M_k`` is checked against a direct numerical integral.

    ``M_1`` also has a closed form that a reader can verify independently --
    the truncated normal mean -- so both are asserted.
    """
    lower = np.array([0.05, 0.30, 0.62])
    upper = np.array([0.20, 0.55, 0.95])
    first, second = conditional_hermite_moments(lower, upper, order=2)

    t_lower = norm.ppf(lower)
    t_upper = norm.ppf(upper)
    closed_form = (norm.pdf(t_lower) - norm.pdf(t_upper)) / (upper - lower)
    assert first == pytest.approx(closed_form, abs=1e-12)

    for index in range(len(lower)):
        grid = np.linspace(t_lower[index], t_upper[index], 200_001)
        density = norm.pdf(grid)
        mass = np.trapezoid(density, grid)
        assert first[index] == pytest.approx(
            np.trapezoid(grid * density, grid) / mass, rel=1e-6
        )
        assert second[index] == pytest.approx(
            np.trapezoid((grid**2 - 1.0) * density, grid) / mass, rel=1e-6
        )


def test_the_recorded_latent_column_is_an_attenuated_reading_of_rho() -> None:
    """``g_1 < 1``, and more so for a coarser margin.

    This is the measurement fact the transmission layer is built on: the
    residual build stores a randomized PIT, so the recorded latent
    correlation is ``g_a g_b rho`` rather than ``rho``. If this ever came back
    equal to one the layer would be solving a problem that did not exist.
    """
    from scipy.stats import poisson

    coarse = 0.35
    fine = 18.0
    attenuation = {}
    for label, mu in (("coarse", coarse), ("fine", fine)):
        counts = np.arange(0, 120)
        probability = poisson.pmf(counts, mu)
        lower = np.clip(poisson.cdf(counts - 1, mu), 0.0, 1.0)
        upper = np.clip(poisson.cdf(counts, mu), 0.0, 1.0)
        keep = probability > 1e-12
        first, _ = conditional_hermite_moments(lower[keep], upper[keep], order=2)
        attenuation[label] = float(np.sum(probability[keep] * first**2))

    assert 0.0 < attenuation["coarse"] < attenuation["fine"] < 1.0
    assert attenuation["coarse"] < 0.7
    assert attenuation["fine"] > 0.95


def test_the_bridge_inverse_undoes_the_bridge_forward() -> None:
    gains = [np.full((3, 3), 0.9), np.full((3, 3), 0.25)]
    latent = np.array(
        [[0.04, -0.02, 0.0], [-0.02, 0.05, 0.01], [0.0, 0.01, -0.03]]
    )
    assert bridge_inverse(bridge_forward(latent, gains), gains) == pytest.approx(
        latent, abs=1e-12
    )


def test_a_bridge_with_no_scale_to_invert_comes_back_missing() -> None:
    """Rather than as a large number produced by dividing by nearly nothing."""
    gains = [np.full((2, 2), 1e-6), np.zeros((2, 2))]
    out = bridge_inverse(np.full((2, 2), 0.01), gains)
    assert np.all(np.isnan(out))


def test_transmission_coefficient_columns_carry_both_readings() -> None:
    frame = pd.DataFrame(
        {
            "cdf_lower_pts": [0.1, 0.4, 0.6, 0.2],
            "cdf_upper_pts": [0.3, 0.5, 0.9, 0.45],
            "e_pts": [-0.5, 0.2, 1.1, -0.3],
            "analytic_mean_pts": [3.0, 8.0, 12.0, 5.0],
        }
    )
    out, diagnostics = transmission_coefficient_columns(
        frame, ("pts",), order=2, bins=2
    )
    for name in ("h1_pts", "h2_pts", "g1_pts", "g2_pts"):
        assert name in out
        assert np.all(np.isfinite(out[name].to_numpy(float)))
    assert diagnostics["by_stat"]["pts"]["latent_attenuation_g1"] == pytest.approx(
        diagnostics["by_stat"]["pts"]["g1_pooled"]
    )


# ----------------------------------------------------------------------
# item 4: the combination
# ----------------------------------------------------------------------


def _pair(disagreement: float, model_error_free: bool = False) -> SourcePair:
    size = 2
    latent = np.full((size, size), 0.02)
    bridge = latent + disagreement
    variance = np.full((size, size), 4e-6)
    return SourcePair(
        latent=latent,
        bridge=bridge,
        variance_latent=variance,
        variance_bridge=variance,
        covariance=np.full((size, size), 2e-6),
    )


def test_disagreement_beyond_sampling_noise_is_charged_to_the_bridge() -> None:
    """The weight must *fall* as the conflict grows, not drift to one half.

    Splitting the excess evenly between the two sources drives every weight
    towards one half however badly the copula assumption is refuted, which
    reads a conflict as a reason to trust the suspect source more.
    """
    weights = []
    for disagreement in (0.0, 0.002, 0.01, 0.05):
        _, info = combine_sources(_pair(disagreement))
        weights.append(float(info["mean_bridge_weight"]))

    assert weights == sorted(weights, reverse=True)
    assert weights[-1] < 0.02
    assert weights[0] > weights[-1] * 5


def test_the_cap_is_a_ceiling_and_reports_when_it_binds() -> None:
    _, uncapped = combine_sources(_pair(0.0))
    _, capped = combine_sources(_pair(0.0), bridge_weight_cap=0.1)
    assert uncapped["mean_bridge_weight"] > 0.1
    assert capped["mean_bridge_weight"] == pytest.approx(0.1)
    assert int(capped["entries_clipped_by_cap"]) == 4
    assert int(uncapped["entries_clipped_by_cap"]) == 0


def test_the_combined_target_never_leaves_the_interval_the_sources_span() -> None:
    for disagreement in (-0.03, 0.0, 0.03):
        combined, _ = combine_sources(_pair(disagreement))
        pair = _pair(disagreement)
        low = np.minimum(pair.latent, pair.bridge)
        high = np.maximum(pair.latent, pair.bridge)
        assert np.all(combined >= low - 1e-12)
        assert np.all(combined <= high + 1e-12)


def test_homogeneity_uses_the_correlated_variance_of_the_difference() -> None:
    """Ignoring the shared games would overstate the variance and hide a conflict."""
    pair = _pair(0.01)
    independent = SourcePair(
        latent=pair.latent,
        bridge=pair.bridge,
        variance_latent=pair.variance_latent,
        variance_bridge=pair.variance_bridge,
        covariance=np.zeros_like(pair.covariance),
    )
    assert homogeneity_test(pair)["pooled_chi_square"] > homogeneity_test(
        independent
    )["pooled_chi_square"]


# ----------------------------------------------------------------------
# items 2 and 6
# ----------------------------------------------------------------------


def test_the_shrunk_role_scales_leave_the_pooled_block_unchanged() -> None:
    """``sum_r p_r s_r == 1``, which is what pooled-neutral means.

    Without it the role layer would move the pooled same-team block, and a
    role refinement would be paying for itself out of the bucket targets the
    repair was accepted on.
    """
    raw = {"starter": 0.92, "rotation": 0.84, "bench": 1.24}
    errors = {"starter": 0.05, "rotation": 0.09, "bench": 0.22}
    shares = {"starter": 0.45, "rotation": 0.35, "bench": 0.20}

    fit = fit_log_shrunk_role_scales(raw, errors, shares)
    pooled = sum(shares[role] * fit.scales[role] for role in shares)
    assert pooled == pytest.approx(1.0, abs=1e-12)


def test_a_sparse_role_shrinks_further_than_a_well_measured_one() -> None:
    """Only the standard error differs between the two fits below.

    The brief asks for sparse cells to shrink to the pooled value, so the same
    raw scale must end up closer to one when it is badly measured.
    """
    raw = {"starter": 0.95, "rotation": 0.90, "bench": 1.35}
    shares = {"starter": 0.40, "rotation": 0.40, "bench": 0.20}

    precise = fit_log_shrunk_role_scales(
        raw, {"starter": 0.02, "rotation": 0.02, "bench": 0.02}, shares
    )
    sparse = fit_log_shrunk_role_scales(
        raw, {"starter": 0.02, "rotation": 0.02, "bench": 0.80}, shares
    )
    assert abs(np.log(sparse.scales["bench"])) < abs(
        np.log(precise.scales["bench"])
    )


def test_the_huber_scale_ignores_one_wild_season() -> None:
    clean = np.array([0.9, 1.1, 0.95, 1.05, 1.0, 0.98])
    contaminated = np.concatenate([clean, [40.0]])
    assert huber_scale(contaminated) == pytest.approx(huber_scale(clean), rel=0.25)
    assert np.std(contaminated) > 5.0 * np.std(clean)


def test_coverage_is_reported_at_every_declared_level() -> None:
    rng = np.random.default_rng(4)
    sample = rng.normal(size=20_000)
    table = coverage_table(sample)
    assert table["coverage_68"] == pytest.approx(0.68, abs=0.02)
    assert table["coverage_95"] == pytest.approx(0.95, abs=0.01)
    assert table["mean_squared_z"] == pytest.approx(1.0, abs=0.05)


def test_the_tie_band_keeps_the_simpler_candidate() -> None:
    """Parsimony is applied mechanically, not by eye."""
    scores = {"simple": 1.000, "complex": 0.990}
    per_unit = {
        "simple": {"a": 0.9, "b": 1.0, "c": 1.1},
        "complex": {"a": 0.80, "b": 1.02, "c": 1.15},
    }
    decision = select_within_tie_band(scores, ("simple", "complex"), per_unit, 1.0)
    assert decision["selected"] == "simple"

    decisive = {"simple": 1.0, "complex": 0.2}
    per_unit_decisive = {
        "simple": {"a": 0.9, "b": 1.0, "c": 1.1},
        "complex": {"a": 0.1, "b": 0.2, "c": 0.3},
    }
    assert (
        select_within_tie_band(
            decisive, ("simple", "complex"), per_unit_decisive, 1.0
        )["selected"]
        == "complex"
    )


# ----------------------------------------------------------------------
# accounting the gate report asserts
# ----------------------------------------------------------------------


def _load_driver(name: str, module_name: str):
    path = (
        Path(__file__).resolve().parents[1]
        / "research/final_upstream_remediation"
        / name
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_rejected_inflation_factor_is_looked_for_in_the_parameters() -> None:
    """A text search would have failed on the evidence of its own rejection.

    The files that record why the global inflation factor was rejected name
    the number, so gate 13 reads the carried spec's parameters instead.
    """
    gates = _load_driver("05_evaluate_gates.py", "gate_driver")

    clean = {"spec": {"bridge_weight_cap": 0.15, "loadings": [[0.1, 1.7035]]}}
    assert gates.inflation_factor_in_parameters(clean) == []

    revived = {
        "spec": {"global_inflation": gates.FORBIDDEN_INFLATION_FACTOR},
        "prose": {"note": "rejected the 1.7659 factor"},
    }
    offenders = gates.inflation_factor_in_parameters(revived)
    assert [entry["path"] for entry in offenders] == [".global_inflation"]


def test_a_new_global_dial_cannot_enter_the_spec_unannounced() -> None:
    gates = _load_driver("05_evaluate_gates.py", "gate_driver")
    upstream = _load_driver("upstream_spec.py", "upstream_spec_under_test")

    choices = upstream.UpstreamChoices(
        temporal="A0_pooled_empirical_bayes",
        role_scale_mode="log_shrunk",
        cross_team_prior="gaussian",
        cross_team_nu=None,
        transmission_cap=0.15,
        uncertainty="U0_raw",
        dependence_temperature=1.0,
    )
    assert set(choices.spec().payload()) == set(gates.DECLARED_SPEC_KEYS)


def test_a_gaussian_cross_team_prior_reports_no_degrees_of_freedom() -> None:
    spec = RemediationSpec(name="g", cross_team_prior="gaussian", cross_team_nu=None)
    assert spec.payload()["cross_team_nu"] is None
    assert spec.parameter_count(len(STATS))["scalar_hyperparameters"] == 2


def _temperature_driver():
    path = (
        Path(__file__).resolve().parents[1]
        / "research/final_upstream_remediation/02_dependence_temperature.py"
    )
    spec = importlib.util.spec_from_file_location("temperature_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _priced_events(gap: float, games: int = 80) -> pd.DataFrame:
    """Events whose log loss falls linearly in lambda by ``gap`` per game.

    The outcome is always realised, so the loss at each temperature is just
    ``-log p`` and the per-game ranking is exactly the sign of ``gap``.
    """
    driver = _temperature_driver()
    rows = []
    for game in range(games):
        record: dict[str, object] = {"game_id": game, "n_legs": 2, "realized": 1}
        for temperature in driver.TEMPERATURE_GRID:
            record[f"p_{temperature:.2f}"] = float(
                np.exp(-(0.30 + gap * temperature))
            )
        rows.append(record)
    return pd.DataFrame(rows)


def test_a_cooler_temperature_is_taken_only_when_it_decisively_wins() -> None:
    """Guards the sign and the reference of the temperature decision.

    The candidate must be compared against the full model, not the full model
    against itself, and a lower loss must read as a win rather than a loss.
    """
    driver = _temperature_driver()

    decisive = driver.select(_priced_events(gap=0.05))
    assert decisive["best_unconditional"] == "0.00"
    assert decisive["selected"] == "0.00"
    assert decisive["best_improvement_over_full_model"] == pytest.approx(0.05)

    hotter = driver.select(_priced_events(gap=-0.05))
    assert hotter["best_unconditional"] == "1.00"
    assert hotter["selected"] == "1.00"


def test_a_temperature_inside_the_tie_band_leaves_the_full_model_in_place() -> None:
    driver = _temperature_driver()
    frame = _priced_events(gap=0.05)
    # One game that strongly prefers the full model, enough to swamp the
    # standard error without reversing the pooled ranking.
    noisy = frame.copy()
    for temperature in driver.TEMPERATURE_GRID:
        column = f"p_{temperature:.2f}"
        noisy.loc[0, column] = float(
            np.exp(-(3.0 * (1.0 - temperature) + 0.001))
        )

    decision = driver.select(noisy)
    assert decision["best_unconditional"] == "0.00"
    assert decision["selected"] == "1.00"
    assert "inside one standard error" in str(decision["reason"])


def test_no_remediation_spec_introduces_a_pairwise_or_player_parameter() -> None:
    for spec in (
        CONTROL_SPEC,
        RemediationSpec(name="t", cross_team_nu=5.0, cross_team_prior="student_t"),
        RemediationSpec(name="b", bridge_mode="inverse_variance", bridge_weight_cap=0.3),
        RemediationSpec(name="l", dependence_temperature=0.5),
    ):
        counts = spec.parameter_count(len(STATS))
        assert counts["pairwise"] == 0
        assert counts["player_indexed"] == 0
        assert counts["active_role_deviation_parameters"] == 0
        assert counts["active_symmetric_subspace_parameters"] == 0


def test_the_per_game_accumulator_matches_the_pooled_pair_moments(
    standardized: pd.DataFrame,
) -> None:
    """The transmission layer's accumulator must be the same estimator.

    It exists only so several value sets can share one bootstrap index; if it
    were a different estimator the bridge would be comparing its own moment
    against the one the factor fit targets.
    """
    pooled = pair_moments(standardized, STATS, bootstrap=0, seed=1)
    accumulated = accumulate_per_game(
        standardized, [f"zs_{stat}" for stat in STATS], STATS
    )
    same, cross = accumulated.pooled()
    assert same == pytest.approx(pooled.same_team, abs=0.0)
    assert cross == pytest.approx(pooled.cross_team, abs=0.0)
    assert accumulated.n_games == pooled.games
