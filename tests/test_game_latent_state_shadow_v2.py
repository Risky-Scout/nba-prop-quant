"""Tests for the Shadow V2 structural refinement.

SHADOW / RESEARCH ONLY.

Four structural claims carry the branch, and each one is an identity rather
than a tolerance on an estimate, so each is tested as one:

1.  **The controls are reproduced, not re-derived.** :data:`V1_BASE_SPEC` must
    give byte-identical loadings to :func:`factors.fit_shared_factors` and
    :data:`REPAIR_CONTROL_SPEC` to :func:`repair.fit_repaired_factors`. If
    that drifts, every before-and-after number on the branch is comparing two
    different models.
2.  **The same-team repair cannot reach the cross-team block.** Appending the
    same loading to both ``A`` and ``B`` cancels from ``X = A - B``
    algebraically, so the fitted cross-team block must come back *bitwise*
    unchanged, not merely close.
3.  **The role layer cannot move the pooled block and cannot break PSD.** The
    scores are centred on the pair shares, so the linear term averages to
    zero; the quadratic term does not, and is absorbed into the symmetric
    block. ``U_r U_r'`` is a Gram for every role, so ``A`` and ``B`` stay PSD
    role by role with no eigenvalue repair.
4.  **The count bridge is exact.** Mehler's series and Gauss-Legendre
    quadrature are two evaluations of one integral and must agree to float64
    noise, and both must agree with a Monte Carlo draw from the copula they
    describe.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from nba_prop_quant.research.game_latent_state.bridge import (
    BRIDGE_METHOD_MEHLER,
    BRIDGE_METHOD_QUADRATURE,
    DEFAULT_MEHLER_TERMS,
    BridgeNotIdentified,
    build_bridge_curve,
    discrete_marginal,
    mehler_scores,
)
from nba_prop_quant.research.game_latent_state.covariance import (
    SharedFactorLoadings,
    project_psd_rank,
)
from nba_prop_quant.research.game_latent_state.factors import (
    COMPETITION_GATE_MIN_DRAWS,
    fit_shared_factors,
    standardize_residuals,
)
from nba_prop_quant.research.game_latent_state.repair import (
    DEFAULT_R_CONTRAST,
    EB_FAMILY_BLOCK_DIAGONAL,
    SHRINKAGE_EMPIRICAL_BAYES,
    SHRINKAGE_SOFT_THRESHOLD,
    RepairSpec,
    fit_repaired_factors,
)
from nba_prop_quant.research.game_latent_state.v2 import (
    AXIS_INDIFFERENCE_TOLERANCE,
    MIN_ROLE_PAIRS,
    REPAIR_CONTROL_SPEC,
    SYMMETRIC_EIGENVALUE_FLOOR_FRACTION,
    SYMMETRIC_MODE_RESERVED,
    SYMMETRIC_MODE_RESIDUAL,
    V1_BASE_SPEC,
    V2Spec,
    build_base_loadings,
    canonical_role_split,
    effective_symmetric_block,
    fit_role_layer,
    fit_v2_factors,
    initial_role_scores,
    resolve_axis_indifference,
    role_conditioned_rmse,
    role_pair_moments,
    role_pair_shares,
    symmetric_lift,
)

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")
ROLES: tuple[str, ...] = ("starter", "rotation", "bench")

#: Same-team loading of the team factor, by role. Deliberately unequal: a
#: uniform loading would leave the role layer nothing to find, so a test built
#: on one could not tell a working role layer from a disabled one.
#: Bootstrap draws for a fit whose base must be *exact*. The competition
#: family is gated on a bootstrap lower bound and declines below
#: ``COMPETITION_GATE_MIN_DRAWS`` draws; without it the additive Gram
#: dominates the same-team target instead of equalling it, so even a
#: full-rank base overshoots by ``Q`` and nothing below is exact.
FULL_RANK_BOOTSTRAP = COMPETITION_GATE_MIN_DRAWS + 20

ROLE_TEAMMATE_SD: dict[str, float] = {
    "starter": 0.15,
    "rotation": 0.30,
    "bench": 0.55,
}


def synthetic_residuals(
    seasons: tuple[int, ...] = (2020, 2021, 2022, 2023),
    games_per_season: int = 250,
    team_size: int = 12,
    seed: int = 7,
    shared_sd: float = 0.25,
    teammate_stats: tuple[str, ...] = ("reb", "ast"),
) -> pd.DataFrame:
    """Residuals with a shared game factor and a role-dependent team factor.

    The team factor loads on ``teammate_stats`` for one team at a time, with a
    per-role amplitude, so there is genuine same-team dependence that varies
    by role pair and no cross-team dependence beyond the game factor.

    Sized so that every role cell clears :data:`MIN_ROLE_PAIRS`. Four players
    per role per team gives twelve within-role ordered pairs per team-game, so
    a thousand games puts the *weakest* cell -- a within-role one -- at
    twenty-four thousand pairs. A smaller frame leaves every cell below the
    threshold, the role layer correctly declines to fit, and the tests then
    pass for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    game_id = 0
    for season in seasons:
        for _ in range(games_per_season):
            game_id += 1
            game_factor = rng.normal(0.0, 1.0)
            for side, team_id in enumerate((1, 2)):
                team_factor = rng.normal(0.0, 1.0)
                for member in range(team_size):
                    role = ROLES[member % len(ROLES)]
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
                        "role_bucket": role,
                    }
                    for stat in STATS:
                        value = rng.normal(0.0, 1.0) + shared_sd * game_factor
                        if stat in teammate_stats:
                            value += ROLE_TEAMMATE_SD[role] * team_factor
                        row[f"z_{stat}"] = value
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def standardized() -> pd.DataFrame:
    frame, _ = standardize_residuals(synthetic_residuals(), STATS)
    return frame


@pytest.fixture(scope="module")
def role_moments(standardized: pd.DataFrame):
    return role_pair_moments(standardized, STATS, bootstrap=80, seed=73)


def v2_spec(name: str, **overrides: object) -> V2Spec:
    """A V2 candidate with the cross-team path held at accepted V1's.

    Accepted V1's ranks are ``k_game = 2`` and one contrast factor, which for
    six stats is a *truncated* base. That is the regime the residual symmetric
    subspace is defined for, so it is the default here.
    """
    return V2Spec(
        name=name,
        same_shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
        same_eb_family=EB_FAMILY_BLOCK_DIAGONAL,
        cross_shrinkage=SHRINKAGE_SOFT_THRESHOLD,
        k_game=2,
        r_contrast=DEFAULT_R_CONTRAST,
        **overrides,  # type: ignore[arg-type]
    )


def v2_full_rank_spec(name: str, **overrides: object) -> V2Spec:
    """A V2 candidate on the accepted bucket repair's cross-team path.

    Six game factors and six contrast factors is full rank for six stats, so
    the base represents both targets exactly. That is the regime the *reserved*
    symmetric subspace is defined for, and the regime in which the residual one
    is vacuous.
    """
    return V2Spec(
        name=name,
        same_shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
        same_eb_family=EB_FAMILY_BLOCK_DIAGONAL,
        cross_shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
        cross_eb_family=EB_FAMILY_BLOCK_DIAGONAL,
        k_game=len(STATS),
        r_contrast=len(STATS),
        **overrides,  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------
# 1. the controls are reproduced, not re-derived
# ----------------------------------------------------------------------


def test_v1_base_spec_reproduces_fit_shared_factors(standardized):
    """V2's own code path must give accepted V1 back exactly."""
    control = fit_shared_factors(
        standardized,
        STATS,
        k_game=2,
        bootstrap=80,
        seed=73,
        role_column="role_bucket",
    )
    reproduced = fit_v2_factors(
        standardized,
        STATS,
        spec=V2Spec(
            name="v1_base",
            same_shrinkage=V1_BASE_SPEC.same_shrinkage,
            cross_shrinkage=V1_BASE_SPEC.cross_shrinkage,
            k_game=2,
            r_contrast=V1_BASE_SPEC.r_contrast,
        ),
        bootstrap=80,
        seed=73,
    )
    assert reproduced.loadings.to_payload() == control.loadings.to_payload()


def test_repair_control_spec_reproduces_the_accepted_repair(standardized):
    """And the accepted bucket repair, through the same path."""
    spec = RepairSpec(
        name="C_rank_k6_r6_eb_block_diagonal",
        family="C_rank",
        shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
        eb_family=EB_FAMILY_BLOCK_DIAGONAL,
        k_game=2,
        r_contrast=6,
    )
    control = fit_repaired_factors(standardized, STATS, spec, bootstrap=80, seed=73)
    reproduced = fit_v2_factors(
        standardized,
        STATS,
        spec=V2Spec(
            name="repair_control",
            same_shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
            cross_shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
            k_game=2,
            r_contrast=6,
        ),
        bootstrap=80,
        seed=73,
    )
    assert reproduced.loadings.to_payload() == control.loadings.to_payload()


def test_the_controls_carry_no_symmetric_subspace_or_role_deviation():
    """The controls are the controls: neither V2 layer may be switched on."""
    for spec in (V1_BASE_SPEC, REPAIR_CONTROL_SPEC):
        assert spec.r_symmetric == 0
        assert spec.role_deviation is False
        assert spec.bridge_weight == 0.0
        assert spec.symmetric_mode == SYMMETRIC_MODE_RESIDUAL


@pytest.mark.parametrize(
    "mode", [SYMMETRIC_MODE_RESIDUAL, SYMMETRIC_MODE_RESERVED]
)
def test_the_symmetric_mode_is_inert_at_rank_zero(standardized, role_moments, mode):
    """Rank zero is the same fit in either mode, and it is the repair control.

    This is what makes a frozen candidate at rank zero unambiguous. The mode
    field still has to be recorded and round-tripped -- a spec that silently
    dropped it would fit a different model at any positive rank -- but at the
    axis's null value it can have no effect, so the recorded value cannot
    change what was frozen.
    """
    kwargs = {"bootstrap": 80, "seed": 73, "role_moments": role_moments}
    control = fit_v2_factors(standardized, STATS, spec=REPAIR_CONTROL_SPEC, **kwargs)
    at_null = fit_v2_factors(
        standardized,
        STATS,
        spec=replace(REPAIR_CONTROL_SPEC, name="frozen", symmetric_mode=mode),
        **kwargs,
    )
    assert at_null.loadings.r_symmetric == 0
    assert at_null.loadings.symmetric is None
    assert at_null.loadings.role_deviation is None
    assert at_null.loadings.to_payload() == control.loadings.to_payload()


def test_a_spec_round_trips_its_symmetric_mode(standardized, role_moments):
    """The mode survives ``payload``, which is what the freeze record stores.

    The freeze driver rebuilds the winning spec from the screening artifact
    field by field, so a field missing from either side of that round trip is a
    model substituted silently.
    """
    spec = v2_full_rank_spec(
        "round_trip", r_symmetric=2, symmetric_mode=SYMMETRIC_MODE_RESERVED
    )
    payload = spec.payload()
    assert payload["symmetric_mode"] == SYMMETRIC_MODE_RESERVED
    rebuilt = V2Spec(
        name=str(payload["name"]),
        same_shrinkage=str(payload["same_shrinkage"]),
        same_eb_family=str(payload["same_eb_family"]),
        cross_shrinkage=str(payload["cross_shrinkage"]),
        cross_eb_family=str(payload["cross_eb_family"]),
        shrink_z=float(payload["shrink_z"]),
        k_game=int(payload["k_game"]),
        r_contrast=int(payload["r_contrast"]),
        r_symmetric=int(payload["r_symmetric"]),
        symmetric_mode=str(payload["symmetric_mode"]),
        role_deviation=bool(payload["role_deviation"]),
        role_column=payload["role_column"],
        bridge_weight=float(payload["bridge_weight"]),
        temporal_treatment=str(payload["temporal_treatment"]),
    )
    assert rebuilt == spec


# ----------------------------------------------------------------------
# 1b. the two symmetric modes, and which base each one is for
# ----------------------------------------------------------------------


def representable_blocks(seed: int = 19) -> tuple[np.ndarray, np.ndarray]:
    """A same-team and cross-team pair the three-family model can fit exactly.

    Built the way the model reads them -- ``S = A + B - Q`` and ``X = A - B``
    from three PSD Grams -- so ``S`` is genuinely indefinite, which is the case
    the competition family exists for and the case the real moments are in. The
    synthetic residual frame is not in that case: its same-team block is PSD,
    the competition gate correctly declines, and then the additive Gram
    dominates the target instead of equalling it. So the exactness premise is
    asserted here, on the algebra, rather than on a frame that does not have
    the property.
    """
    rng = np.random.default_rng(seed)
    n = len(STATS)

    def gram(scale: float) -> np.ndarray:
        factor = rng.normal(size=(n, n)) * scale
        return factor @ factor.T

    a, b, q = gram(0.30), gram(0.20), gram(0.15)
    return a + b - q, a - b


@pytest.mark.parametrize("rank", [1, 2, 6])
def test_a_full_rank_base_represents_both_targets_exactly(rank):
    """The premise the reserved mode rests on, asserted rather than assumed.

    At six game and six contrast factors the construction is an exact solution
    of ``S = A + B - Q`` and ``X = A - B`` for six stats, so there is no
    residual at all. Everything else about the two modes follows from this: the
    residual subspace has nothing to carry here, and reserving a piece of the
    target out of the base is free.
    """
    same, cross = representable_blocks()
    assert np.min(np.linalg.eigvalsh(same)) < 0.0, "S must be indefinite"
    base = build_base_loadings(
        STATS,
        same,
        cross,
        v2_full_rank_spec("exact", r_symmetric=rank),
        competition_allowed=True,
    )
    assert np.allclose(base.same_team_correlation(), same, atol=1e-12)
    assert np.allclose(base.cross_team_correlation(), cross, atol=1e-12)


def test_the_residual_subspace_is_vacuous_on_an_exact_base():
    """No residual, no subspace -- and no role layer riding on rounding error.

    This is why the reserved mode exists. Before the eigenvalue floor, the
    1e-17 gap left by an exact base still produced a carrier out of whichever
    of its eigenvalues landed positive, and the role layer fitted onto it: the
    quadratic leak reached 4.4e5 before the layer's own drop rule caught it.
    """
    same, cross = representable_blocks()
    spec = v2_full_rank_spec("exact", r_symmetric=6)
    base = build_base_loadings(STATS, same, cross, spec, competition_allowed=True)
    gap = same - base.same_team_correlation()
    assert np.max(np.abs(gap)) < 1e-12
    scale = float(np.max(np.abs(same)))
    assert effective_symmetric_block(0.5 * gap, 6, scale=scale) is None
    # Without the scale the noise still yields a carrier, which is the bug the
    # floor fixes and the reason the argument is not optional in the fit.
    assert effective_symmetric_block(0.5 * gap, 6) is not None


@pytest.mark.parametrize("rank", [1, 2, 6])
def test_the_reserved_subspace_is_free_at_full_rank(rank):
    """It carries a real loading and leaves both fitted blocks where they were.

    This is the reparameterisation claim, and it is stated against the base's
    own fit rather than against the target so that it does not quietly depend
    on the base being exact.
    """
    same, cross = representable_blocks()
    spec = v2_full_rank_spec("reserved", r_symmetric=rank)
    reference = build_base_loadings(
        STATS, same, cross, spec, competition_allowed=True
    )
    symmetric = effective_symmetric_block(
        0.5 * same, rank, scale=float(np.max(np.abs(same)))
    )
    assert symmetric is not None
    assert symmetric.shape[1] >= 1
    # A real loading, not rounding error.
    assert np.max(np.abs(symmetric)) > 1e-4
    lift = symmetric_lift(symmetric)
    reserved = build_base_loadings(
        STATS, same - lift, cross, spec, competition_allowed=True
    )
    assert np.allclose(
        reserved.same_team_correlation() + lift,
        reference.same_team_correlation(),
        atol=1e-12,
    )
    assert np.allclose(
        reserved.cross_team_correlation(),
        reference.cross_team_correlation(),
        atol=1e-12,
    )


@pytest.mark.parametrize("rank", [1, 2, 6])
def test_the_reserved_subspace_leaves_the_cross_team_block_alone_on_real_shapes(
    standardized, role_moments, rank
):
    """And the same through the whole fit, where the gate decides for itself."""
    reference = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_full_rank_spec("exact", r_symmetric=0),
        bootstrap=FULL_RANK_BOOTSTRAP,
        seed=73,
    )
    for role_deviation in (False, True):
        fit = fit_v2_factors(
            standardized,
            STATS,
            spec=v2_full_rank_spec(
                f"reserved{rank}",
                r_symmetric=rank,
                symmetric_mode=SYMMETRIC_MODE_RESERVED,
                role_deviation=role_deviation,
            ),
            bootstrap=FULL_RANK_BOOTSTRAP,
            seed=73,
            role_moments=role_moments,
        )
        assert fit.loadings.symmetric is not None
        assert np.allclose(
            fit.loadings.cross_team_correlation(),
            reference.loadings.cross_team_correlation(),
            atol=1e-12,
        )
        assert fit.cross_team_unchanged_deviation() < 1e-12


def test_the_reserved_subspace_gives_the_role_layer_a_carrier(
    standardized, role_moments
):
    """On an exact base the role deviation exists in reserved mode only."""
    fits = {
        mode: fit_v2_factors(
            standardized,
            STATS,
            spec=v2_full_rank_spec(
                mode, r_symmetric=6, symmetric_mode=mode, role_deviation=True
            ),
            bootstrap=FULL_RANK_BOOTSTRAP,
            seed=73,
            role_moments=role_moments,
        )
        for mode in (SYMMETRIC_MODE_RESIDUAL, SYMMETRIC_MODE_RESERVED)
    }
    assert fits[SYMMETRIC_MODE_RESIDUAL].loadings.role_deviation is None
    reserved = fits[SYMMETRIC_MODE_RESERVED]
    assert reserved.loadings.role_deviation is not None
    assert reserved.role_scores
    # Centred, and the leak taken out of the base rather than the subspace.
    shares = reserved.loadings.role_pair_shares
    assert shares
    assert reserved.role_diagnostics["leak_absorbed_into_base"] == 1.0
    assert abs(reserved.role_diagnostics["pooled_leak_absorbed"]) < 1.0


def test_the_reserved_subspace_is_not_an_identity_on_a_truncated_base(
    standardized,
):
    """And the screen must therefore guard the identity rather than assume it.

    Reserving a piece of the same-team target changes the additive Gram the
    base is built from, and a *truncated* ``A - B`` depends on that Gram. The
    accepted V1 ranks are truncated, so the mode is not admissible there --
    recorded as a test because it is the reason the same-team axis screens the
    cross-team identity directly instead of trusting the mode's name.
    """
    fit = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec(
            "reserved_truncated",
            r_symmetric=6,
            symmetric_mode=SYMMETRIC_MODE_RESERVED,
        ),
        bootstrap=FULL_RANK_BOOTSTRAP,
        seed=73,
    )
    assert fit.cross_team_unchanged_deviation() > 1e-6


def test_an_unknown_symmetric_mode_is_refused(standardized):
    with pytest.raises(ValueError, match="unknown symmetric_mode"):
        fit_v2_factors(
            standardized,
            STATS,
            spec=v2_spec("bogus", r_symmetric=2, symmetric_mode="halfway"),
            bootstrap=80,
            seed=73,
        )


# ----------------------------------------------------------------------
# 2. the same-team repair cannot reach the cross-team block
# ----------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2, 6])
def test_symmetric_subspace_leaves_cross_team_bitwise_unchanged(
    standardized, role_moments, rank
):
    """``A - B`` is algebraically free of the symmetric block.

    Asserted at zero rather than at a tolerance: ``U U' - U U'`` is not a
    small number, it is the same floating-point value subtracted from itself.
    """
    reference = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("rsym0", r_symmetric=0),
        bootstrap=80,
        seed=73,
    )
    for role_deviation in (False, True):
        fitted = fit_v2_factors(
            standardized,
            STATS,
            spec=v2_spec(
                f"rsym{rank}", r_symmetric=rank, role_deviation=role_deviation
            ),
            bootstrap=80,
            seed=73,
            role_moments=role_moments,
        )
        assert np.array_equal(
            fitted.loadings.cross_team_correlation(),
            reference.loadings.cross_team_correlation(),
        )
        assert fitted.cross_team_unchanged_deviation() == 0.0


def test_symmetric_subspace_lifts_the_same_team_block_by_twice_its_gram(
    standardized,
):
    """``S`` gains exactly ``2 U U'`` and nothing else."""
    reference = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("rsym0", r_symmetric=0),
        bootstrap=80,
        seed=73,
    )
    fitted = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("rsym3", r_symmetric=3),
        bootstrap=80,
        seed=73,
    )
    symmetric = fitted.loadings.symmetric
    assert symmetric is not None
    # Both fits carry V1's multiplicative role layer, which is estimated on
    # the block the symmetric subspace changes, so the comparison is made on
    # the role-blind blocks the subspace is defined against.
    lift = fitted.loadings.same_team_correlation() / fitted.loadings.scale_for_role(
        None
    ) - reference.loadings.same_team_correlation() / reference.loadings.scale_for_role(
        None
    )
    assert np.allclose(lift, 2.0 * symmetric @ symmetric.T, atol=1e-12)


def test_the_symmetric_subspace_closes_the_same_team_gap(standardized):
    """Raising the rank may not make the same-team fit worse."""
    errors = []
    for rank in (0, 1, 2, 6):
        fitted = fit_v2_factors(
            standardized,
            STATS,
            spec=v2_spec(f"rsym{rank}", r_symmetric=rank),
            bootstrap=80,
            seed=73,
        )
        errors.append(float(fitted.diagnostics()["same_team_fit_rmse"]))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]


# ----------------------------------------------------------------------
# 3. the role layer
# ----------------------------------------------------------------------


def test_role_scores_are_centred_on_the_pair_shares(standardized, role_moments):
    """``sum_r p_r h_r == 0``, so the linear term cannot move the pooled block."""
    fitted = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("role", r_symmetric=6, role_deviation=True),
        bootstrap=80,
        seed=73,
        role_moments=role_moments,
    )
    assert fitted.role_scores, "the fit saw labelled roles and should report scores"
    shares = role_pair_shares(role_moments)
    centre = sum(shares[role] * fitted.role_scores[role] for role in shares)
    assert abs(centre) < 1e-10
    assert abs(float(fitted.role_diagnostics["pooled_linear_leakage"])) < 1e-10


def test_the_role_layers_quadratic_leak_is_absorbed(standardized, role_moments):
    """The pair-share average of ``S`` may not drift from the target.

    The quadratic term ``sum p_{r r'} h_r h_r'`` is *not* zeroed by centring,
    so it is absorbed into the symmetric block. What this checks is the
    consequence: switching the role layer on must not push the pair-share
    averaged same-team block further from its target than the role-blind fit
    already is.
    """
    blind = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("blind", r_symmetric=6),
        bootstrap=80,
        seed=73,
    )
    role = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("role", r_symmetric=6, role_deviation=True),
        bootstrap=80,
        seed=73,
        role_moments=role_moments,
    )
    assert role.loadings.role_quadratic_share() != 0.0
    blind_gap = float(
        np.max(np.abs(blind.loadings.pooled_same_team_correlation() - blind.same_target))
    )
    role_gap = float(
        np.max(np.abs(role.loadings.pooled_same_team_correlation() - role.same_target))
    )
    assert role_gap <= blind_gap + 1e-12
    assert float(role.role_diagnostics["pooled_leak_absorbed"]) == pytest.approx(
        role.loadings.role_quadratic_share(), rel=1e-9
    )


def test_a_role_the_fit_never_saw_resolves_to_the_pooled_loading(
    standardized, role_moments
):
    """No role table: an unseen label is predicted, from ``h = 0``."""
    fitted = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("role", r_symmetric=6, role_deviation=True),
        bootstrap=80,
        seed=73,
        role_moments=role_moments,
    )
    loadings = fitted.loadings
    assert np.allclose(
        loadings.symmetric_block("a_role_that_does_not_exist"),
        loadings.symmetric,
        atol=0.0,
    )
    unseen = loadings.same_team_correlation_for_roles("nope", "nope")
    assert np.allclose(unseen, loadings.same_team_correlation(), atol=1e-12)


def test_the_role_layer_keeps_every_role_block_psd(standardized, role_moments):
    """``A`` and ``B`` are Grams for every role, so PSD needs no repair."""
    fitted = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("role", r_symmetric=6, role_deviation=True),
        bootstrap=80,
        seed=73,
        role_moments=role_moments,
    )
    loadings = fitted.loadings
    game = loadings.game @ loadings.game.T
    contrast = loadings.contrast_matrix @ loadings.contrast_matrix.T
    for role in (*ROLES, None, "unseen"):
        block = loadings.symmetric_block(role)
        gram = block @ block.T
        assert np.min(np.linalg.eigvalsh(gram)) >= -1e-12
        assert np.min(np.linalg.eigvalsh(game + gram)) >= -1e-12
        assert np.min(np.linalg.eigvalsh(contrast + gram)) >= -1e-12


def test_role_cells_improve_without_breaking_a_fitted_cell(
    standardized, role_moments
):
    """The layer's purpose, and the no-regression clause that bounds it."""
    blind = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("blind", r_symmetric=6),
        bootstrap=80,
        seed=73,
    )
    role = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("role", r_symmetric=6, role_deviation=True),
        bootstrap=80,
        seed=73,
        role_moments=role_moments,
    )
    scored = role_conditioned_rmse(
        role_moments, role.loadings, pooled_only=blind.loadings
    )
    assert scored["measurable"] == 1.0
    assert scored["role_cells"] >= 3.0
    assert scored["role_rmse_improvement_fraction"] > 0.2
    assert scored["newly_exceeding_cells"] == []


def test_the_role_split_is_canonical_and_the_model_is_invariant_to_it(
    role_moments, standardized
):
    """``(h / c, c W)`` is the same model, so only the canonical split is reported."""
    blind = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("blind", r_symmetric=6),
        bootstrap=80,
        seed=73,
    )
    symmetric = blind.loadings.symmetric
    assert symmetric is not None
    pooled = blind.loadings.same_team_correlation()
    scores, deviation, _ = fit_role_layer(
        role_moments, symmetric, pooled, scales=blind.loadings.role_scale
    )
    shares = role_pair_shares(role_moments)
    norm = float(np.linalg.norm(symmetric))

    reference = SharedFactorLoadings(
        stats=STATS,
        game=blind.loadings.game,
        team_contrast=blind.loadings.team_contrast,
        competition=blind.loadings.competition,
        symmetric=symmetric,
        role_deviation=deviation,
        role_offset=dict(scores),
    )
    for factor in (0.25, 4.0, -2.0):
        rescaled = SharedFactorLoadings(
            stats=STATS,
            game=blind.loadings.game,
            team_contrast=blind.loadings.team_contrast,
            competition=blind.loadings.competition,
            symmetric=symmetric,
            role_deviation=deviation / factor,
            role_offset={role: value * factor for role, value in scores.items()},
        )
        for first in ROLES:
            for second in ROLES:
                assert np.allclose(
                    rescaled.same_team_correlation_for_roles(first, second),
                    reference.same_team_correlation_for_roles(first, second),
                    atol=1e-12,
                )
        # And the canonicalisation maps every one of them to the same split.
        canonical_scores, canonical_deviation = canonical_role_split(
            {role: value * factor for role, value in scores.items()},
            deviation / factor,
            shares,
            norm,
        )
        assert canonical_scores == pytest.approx(scores, rel=1e-8, abs=1e-12)
        assert np.allclose(canonical_deviation, deviation, atol=1e-10)


def test_the_role_fit_is_independent_of_its_starting_point(
    role_moments, standardized
):
    """The pinned score removes the flat direction, so the optimum is found."""
    blind = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("blind", r_symmetric=6),
        bootstrap=80,
        seed=73,
    )
    symmetric = blind.loadings.symmetric
    assert symmetric is not None
    pooled = blind.loadings.same_team_correlation()
    scales = blind.loadings.role_scale
    start = initial_role_scores(role_moments, pooled, scales=scales)

    def fit_from(multiple: float):
        scores, deviation, diagnostics = fit_role_layer(
            role_moments,
            symmetric,
            pooled,
            scales=scales,
            start={role: multiple * value for role, value in start.items()},
        )
        assert diagnostics["role_layer_fitted"] == 1.0
        return scores, deviation, float(diagnostics["role_fit_cost"])

    # The starting scores are rescaled into the pinned role's units, so any
    # nonzero multiple of them is the same starting direction and has to give
    # the identical answer -- not merely a close one.
    reference_scores, reference_deviation, reference_cost = fit_from(1.0)
    for multiple in (0.5, -2.0, 3.0):
        scores, deviation, cost = fit_from(multiple)
        assert cost == pytest.approx(reference_cost, rel=1e-6)
        assert scores == pytest.approx(reference_scores, abs=1e-4)
        assert np.allclose(deviation, reference_deviation, atol=1e-4)

    # A start that says nothing at all is a *different* direction: the pinned
    # score is one and every other is zero. It reaches the same answer to
    # three decimal places and stops 0.2% higher on the objective, which is
    # the honest statement of how flat the optimum is rather than evidence of
    # two different fits.
    scores, _, cost = fit_from(0.0)
    assert cost == pytest.approx(reference_cost, rel=0.01)
    assert scores == pytest.approx(reference_scores, abs=0.01)


def test_the_role_layer_is_shrunk_by_the_support_of_its_weakest_cell(role_moments):
    """Hierarchical shrinkage, read off the data rather than tuned."""
    weakest = min(
        role_moments.support(first, second)
        for index, first in enumerate(role_moments.roles)
        for second in role_moments.roles[index:]
        if role_moments.support(first, second) >= MIN_ROLE_PAIRS
    )
    expected = weakest / (weakest + MIN_ROLE_PAIRS)
    loadings = SharedFactorLoadings(
        stats=STATS,
        game=np.full((len(STATS), 1), 0.2),
        team_contrast=np.full(len(STATS), 0.1),
    )
    _, _, diagnostics = fit_role_layer(
        role_moments,
        np.full((len(STATS), 2), 0.1),
        loadings.same_team_correlation(),
    )
    assert float(diagnostics["role_shrinkage_factor"]) == pytest.approx(expected)


# ----------------------------------------------------------------------
# no pairwise and no player-indexed parameters, ever
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        V1_BASE_SPEC,
        REPAIR_CONTROL_SPEC,
        V2Spec(name="full", r_symmetric=6, role_deviation=True, bridge_weight=0.5),
    ],
)
def test_no_spec_has_a_pairwise_or_player_indexed_parameter(spec):
    counts = spec.parameter_count(len(STATS))
    assert counts["pairwise"] == 0
    assert counts["player_indexed"] == 0
    # The role layer's cost is one loading block plus ``len(roles) - 1``
    # scores, not a cell per role pair.
    assert counts["stat_indexed_role_deviation"] in (
        0,
        len(STATS) * spec.r_symmetric,
    )


def test_loadings_round_trip_through_their_payload(standardized, role_moments):
    """Everything V2 adds has to survive serialisation."""
    fitted = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("full", r_symmetric=4, role_deviation=True),
        bootstrap=80,
        seed=73,
        role_moments=role_moments,
    )
    restored = SharedFactorLoadings.from_payload(fitted.loadings.to_payload())
    assert restored.to_payload() == fitted.loadings.to_payload()
    assert 0 < restored.r_symmetric <= 4
    for first in ROLES:
        for second in ROLES:
            assert np.allclose(
                restored.same_team_correlation_for_roles(first, second),
                fitted.loadings.same_team_correlation_for_roles(first, second),
                atol=0.0,
            )
    assert restored.role_quadratic_share() == pytest.approx(
        fitted.loadings.role_quadratic_share()
    )


# ----------------------------------------------------------------------
# 4. the count bridge
# ----------------------------------------------------------------------


def negative_binomial_cdf(mean: float, size: float, limit: int = 400) -> np.ndarray:
    from scipy.stats import nbinom

    return nbinom.cdf(np.arange(limit), size, size / (size + mean))


#: Means spanning the six supported stats, from a blocks-per-game margin to a
#: points margin. The low-count end is where discretisation attenuates most,
#: so a bridge test on high-count margins only would miss the effect.
BRIDGE_MEANS: tuple[float, ...] = (0.4, 0.9, 1.8, 2.4, 5.0, 12.0, 24.0)


@pytest.fixture(scope="module")
def bridge_margins():
    return [discrete_marginal(negative_binomial_cdf(mean, 4.0)) for mean in BRIDGE_MEANS]


@pytest.fixture(scope="module")
def bridge_pairs(bridge_margins):
    return [
        (bridge_margins[i], bridge_margins[j])
        for i in range(len(bridge_margins))
        for j in range(len(bridge_margins))
    ]


@pytest.mark.parametrize("space", ["count", "latent"])
def test_the_series_and_the_quadrature_are_the_same_integral(bridge_pairs, space):
    """Two evaluations of one integral, so they must agree to float64 noise."""
    series = build_bridge_curve(
        bridge_pairs, space=space, method=BRIDGE_METHOD_MEHLER
    )
    quadrature = build_bridge_curve(
        bridge_pairs, space=space, method=BRIDGE_METHOD_QUADRATURE
    )
    scale = float(np.max(np.abs(quadrature.values)))
    assert np.max(np.abs(series.values - quadrature.values)) < 1e-12 * max(scale, 1.0)
    assert series.is_monotone()
    assert quadrature.is_monotone()

    # The tabulated curves agree to float64 noise, so any central difference
    # of them does too.
    zero = quadrature.zero_index
    step = quadrature.grid[zero + 1] - quadrature.grid[zero - 1]
    difference = (series.values[zero + 1] - series.values[zero - 1]) / step
    assert difference == pytest.approx(quadrature.slope_at_zero(), rel=1e-12)
    # The series reports its slope from the leading coefficient instead, which
    # is exact; the quadrature has only the central difference, whose own
    # truncation error is the gap between the two.
    assert series.slope_at_zero() == pytest.approx(
        quadrature.slope_at_zero(), rel=1e-4
    )


@pytest.mark.parametrize("space", ["count", "latent"])
def test_the_bridge_is_pinned_and_invertible(bridge_pairs, space):
    curve = build_bridge_curve(bridge_pairs, space=space)
    assert curve.evaluate(0.0) == pytest.approx(0.0, abs=1e-15)
    assert curve.grid[curve.zero_index] == pytest.approx(0.0, abs=1e-15)
    for rho in (-0.4, -0.1, 0.05, 0.25, 0.5):
        assert curve.invert(curve.evaluate(rho)) == pytest.approx(rho, abs=1e-9)
    low, high = curve.feasible_range
    with pytest.raises(BridgeNotIdentified):
        curve.invert(high + 0.05)
    with pytest.raises(BridgeNotIdentified):
        curve.invert(low - 0.05)


def test_the_count_bridge_attenuates(bridge_pairs):
    """Discretisation can only shrink a correlation, never inflate it."""
    curve = build_bridge_curve(bridge_pairs, space="count")
    assert 0.0 < curve.slope_at_zero() < 1.0
    for rho in (0.05, 0.2, 0.4):
        assert 0.0 < curve.evaluate(rho) < rho
        assert curve.invert(curve.evaluate(rho)) == pytest.approx(rho, abs=1e-9)


def test_the_count_bridge_is_not_an_odd_function(bridge_margins):
    """Which is why the quadrature is two-sided rather than reflected.

    Reflecting ``[0, rho_max]`` into the negative half assumes
    ``T(-rho) == -T(rho)``, which holds only for symmetric margins. Count
    margins are strongly skewed, and the cross-team block is where the
    negative entries live, so the asymmetry is load-bearing rather than
    cosmetic.
    """
    skewed = [(bridge_margins[0], bridge_margins[0])]
    curve = build_bridge_curve(skewed, space="count")
    asymmetry = max(
        abs(curve.evaluate(-rho) + curve.evaluate(rho)) for rho in (0.1, 0.3, 0.5)
    )
    assert asymmetry > 1e-3


def test_the_mehler_scores_are_bounded_at_high_order(bridge_margins):
    """``He_j`` and ``sqrt(j!)`` both overflow; their ratio must not."""
    for marginal in bridge_margins:
        for space in ("count", "latent"):
            scores = mehler_scores(marginal, space, DEFAULT_MEHLER_TERMS)
            assert scores.shape == (DEFAULT_MEHLER_TERMS,)
            assert np.all(np.isfinite(scores))
            assert np.max(np.abs(scores)) < 1e3


@pytest.mark.parametrize("space", ["count", "latent"])
def test_the_series_term_count_is_past_convergence(bridge_pairs, space):
    """Dropping a third of the terms must not move the curve."""
    full = build_bridge_curve(bridge_pairs, space=space, terms=DEFAULT_MEHLER_TERMS)
    short = build_bridge_curve(bridge_pairs, space=space, terms=32)
    assert np.max(np.abs(full.values - short.values)) < 1e-10


def test_the_count_bridge_matches_a_monte_carlo_draw_from_the_copula():
    """The series describes a copula; sampling that copula must agree.

    One pair of margins, one latent correlation, four million draws. The
    comparison is against the Monte Carlo standard error, because that is the
    only thing uncertain here: the bridge itself is deterministic.
    """
    from scipy.stats import norm as normal

    margins = [
        discrete_marginal(negative_binomial_cdf(mean, 4.0)) for mean in (1.2, 9.0)
    ]
    cdfs = [negative_binomial_cdf(mean, 4.0) for mean in (1.2, 9.0)]
    curve = build_bridge_curve([(margins[0], margins[1])], space="count")

    rng = np.random.default_rng(404)
    draws = 4_000_000
    for rho in (-0.3, 0.25):
        first = rng.standard_normal(draws)
        second = rho * first + np.sqrt(1.0 - rho**2) * rng.standard_normal(draws)
        x = np.searchsorted(cdfs[0], normal.cdf(first), side="left")
        y = np.searchsorted(cdfs[1], normal.cdf(second), side="left")
        product = (x - x.mean()) * (y - y.mean())
        correlation = product.mean() / (x.std() * y.std())
        error = product.std() / np.sqrt(draws) / (x.std() * y.std())
        assert abs(curve.evaluate(rho) - correlation) <= 4.0 * error


# ----------------------------------------------------------------------
# role-pair moments
# ----------------------------------------------------------------------


def test_role_pair_moments_pool_to_the_overall_same_team_moment(
    standardized, role_moments
):
    """The role cells are a partition of the same-team pairs, so they average."""
    from nba_prop_quant.research.game_latent_state.factors import pair_moments

    pooled = pair_moments(standardized, STATS)
    total = sum(role_moments.pairs.values())
    assert total == pytest.approx(pooled.same_team_pairs, rel=1e-9)

    blended = np.zeros((len(STATS), len(STATS)))
    for (first, second), count in role_moments.pairs.items():
        blended += count * role_moments.blocks[(first, second)]
    assert np.allclose(blended / total, pooled.same_team, atol=1e-10)


def test_role_pair_moments_report_game_clustered_standard_errors(role_moments):
    assert role_moments.bootstrap_draws == 80
    for index, first in enumerate(role_moments.roles):
        for second in role_moments.roles[index:]:
            error = role_moments.standard_error(first, second)
            assert error is not None
            assert error.shape == (len(STATS), len(STATS))
            assert np.all(error > 0.0)
            # Symmetric cells, read in either order.
            assert np.array_equal(
                error, role_moments.standard_error(second, first)
            )


def test_role_pair_shares_sum_to_one(role_moments):
    shares = role_pair_shares(role_moments)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert all(share > 0.0 for share in shares.values())


def test_the_role_moment_cells_recover_the_synthetic_role_ordering(role_moments):
    """The synthetic team factor is strongest for ``bench``; the cells see it."""
    index = STATS.index("reb")
    within = {
        role: float(role_moments.symmetrised(role, role)[index, index])
        for role in ROLES
    }
    assert within["bench"] > within["rotation"] > within["starter"]


# ----------------------------------------------------------------------
# the design rows reproduce the blocks they are built from
# ----------------------------------------------------------------------


def test_design_rows_reproduce_the_role_conditioned_blocks(
    standardized, role_moments
):
    """The simulator reads ``design``; it must agree with the block algebra.

    Same team is one side, cross team the other, and the symmetric block
    appears twice -- unsigned and signed -- which is what makes it double in
    ``S`` and cancel in ``X``.
    """
    fitted = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("full", r_symmetric=4, role_deviation=True),
        bootstrap=80,
        seed=73,
        role_moments=role_moments,
    )
    loadings = fitted.loadings
    competition = loadings.competition_gram()
    for first in (*ROLES, "unseen"):
        for second in (*ROLES, "unseen"):
            same = loadings.design(1, first) @ loadings.design(1, second).T
            cross = loadings.design(1, first) @ loadings.design(-1, second).T
            expected_same = 0.5 * (same + same.T) - competition * (
                loadings.scale_for_role(first) * loadings.scale_for_role(second)
            )
            assert np.allclose(
                expected_same,
                loadings.same_team_correlation_for_roles(first, second),
                atol=1e-12,
            )
            # The cross-team block is role-free apart from the multiplicative
            # layer: the symmetric block, deviation and all, cancels.
            assert np.allclose(
                0.5 * (cross + cross.T),
                loadings.cross_team_correlation()
                * loadings.scale_for_role(first)
                * loadings.scale_for_role(second),
                atol=1e-12,
            )


# ----------------------------------------------------------------------
# the rank the symmetric block claims is the rank it uses
# ----------------------------------------------------------------------


@pytest.mark.parametrize("requested", [2, 3, 6])
def test_the_symmetric_block_is_trimmed_to_its_positive_rank(requested):
    """A requested rank above the gap's positive rank yields no extra columns.

    ``project_psd_rank`` pads with structurally zero columns, which the role
    deviation would then carry as unidentified directions of ``W``.
    """
    basis = np.linalg.qr(np.random.default_rng(11).normal(size=(len(STATS), 2)))[0]
    # Two positive eigenvalues, the rest strictly negative, so the positive
    # rank is two whatever rank is asked for.
    gap = (
        basis[:, [0]] @ basis[:, [0]].T * 3.0
        + basis[:, [1]] @ basis[:, [1]].T * 1.0
        - np.eye(len(STATS)) * 0.0
    )
    gap = gap - 0.5 * (np.eye(len(STATS)) - basis @ basis.T)

    block = effective_symmetric_block(gap, requested)
    assert block is not None
    assert block.shape == (len(STATS), 2)
    assert np.all(np.any(np.abs(block) > 0.0, axis=0))
    # Trimming drops zeros, so the Gram is untouched by it.
    _, padded = project_psd_rank(gap, rank=requested)
    assert np.allclose(block @ block.T, padded @ padded.T, atol=1e-12)


def test_the_symmetric_block_declines_a_gap_it_cannot_represent():
    """Rank zero and a negative-definite gap both give no block at all."""
    gap = np.eye(len(STATS)) * 2.0
    assert effective_symmetric_block(gap, 0) is None
    assert effective_symmetric_block(gap, -1) is None
    assert effective_symmetric_block(-gap, 6) is None


def test_the_symmetric_block_declines_a_gap_that_is_only_rounding_error():
    """The floor is relative to the target's scale, so it is scale-free."""
    scale = 0.03
    noise = np.eye(len(STATS)) * (0.1 * SYMMETRIC_EIGENVALUE_FLOOR_FRACTION * scale)
    assert effective_symmetric_block(noise, 6) is not None
    assert effective_symmetric_block(noise, 6, scale=scale) is None
    # An order above the floor survives it.
    real = np.eye(len(STATS)) * (10.0 * SYMMETRIC_EIGENVALUE_FLOOR_FRACTION * scale)
    block = effective_symmetric_block(real, 6, scale=scale)
    assert block is not None
    assert block.shape == (len(STATS), len(STATS))


# ----------------------------------------------------------------------
# an axis whose points are the same model resolves to its null value
# ----------------------------------------------------------------------

#: Metrics the same-team axis is judged by, as the screening drivers name them.
JUDGED = ("mean_target_abs_error", "mean_global_latent_rmse", "mean_global_count_rmse")


def axis_summaries(**points: dict[str, float]) -> dict[str, dict[str, object]]:
    """A screening-shaped summary table for the judged metrics only."""
    return {name: dict(metrics) for name, metrics in points.items()}


def test_an_axis_whose_points_all_reproduce_the_null_is_indifferent():
    """Float reassociation is not a model difference.

    These are the magnitudes the third screening pass actually recorded: every
    reserved rank matched rank zero to the sixteenth significant digit.
    """
    null = {key: value for key, value in zip(JUDGED, (3.7479e-3, 3.6404e-3, 4.4154e-3))}
    summaries = axis_summaries(
        null=null,
        reassociated={key: value * (1.0 + 6.5e-15) for key, value in null.items()},
        bitwise=dict(null),
    )
    verdict = resolve_axis_indifference(
        summaries, ("reassociated", "bitwise"), null_point="null", metrics=JUDGED
    )
    assert verdict.axis_is_indifferent
    assert verdict.distinguishable == ()
    assert set(verdict.indistinguishable) == {"reassociated", "bitwise"}
    assert verdict.relative_deviation["bitwise"] == 0.0
    assert verdict.relative_deviation["reassociated"] < AXIS_INDIFFERENCE_TOLERANCE


def test_a_point_that_moves_one_judged_metric_is_distinguishable():
    """One metric is enough: the rule is an ``all``, so it cannot be diluted."""
    null = {key: 1.0 for key in JUDGED}
    moved = dict(null)
    moved["mean_global_count_rmse"] = 1.0 - 1e-6
    summaries = axis_summaries(null=null, earns_it=moved, noise_only=dict(null))
    verdict = resolve_axis_indifference(
        summaries, ("earns_it", "noise_only"), null_point="null", metrics=JUDGED
    )
    assert not verdict.axis_is_indifferent
    assert verdict.distinguishable == ("earns_it",)
    assert verdict.indistinguishable == ("noise_only",)


def test_an_axis_with_no_admissible_point_is_not_called_indifferent():
    """Indifference and an unsatisfiable guard are different findings.

    Both resolve the axis to its null value, but only the first one says the
    points were measured and found equivalent.
    """
    null = {key: 1.0 for key in JUDGED}
    verdict = resolve_axis_indifference(
        axis_summaries(null=null), (), null_point="null", metrics=JUDGED
    )
    assert not verdict.axis_is_indifferent
    assert verdict.distinguishable == ()
    assert verdict.indistinguishable == ()


def test_the_indifference_comparison_is_relative_not_absolute():
    """A tiny metric and a large one are held to the same relative standard."""
    for magnitude in (1e-8, 1.0, 1e8):
        inside_of = 1.0 + 0.1 * AXIS_INDIFFERENCE_TOLERANCE
        outside_of = 1.0 + 10.0 * AXIS_INDIFFERENCE_TOLERANCE
        null = {key: magnitude for key in JUDGED}
        inside = {key: magnitude * inside_of for key in JUDGED}
        outside = {key: magnitude * outside_of for key in JUDGED}
        verdict = resolve_axis_indifference(
            axis_summaries(null=null, inside=inside, outside=outside),
            ("inside", "outside"),
            null_point="null",
            metrics=JUDGED,
        )
        assert verdict.indistinguishable == ("inside",), magnitude
        assert verdict.distinguishable == ("outside",), magnitude


def test_a_null_metric_of_zero_does_not_divide_by_zero():
    """The relative denominator is floored, so an exact zero stays comparable."""
    summaries = axis_summaries(
        null={key: 0.0 for key in JUDGED},
        also_zero={key: 0.0 for key in JUDGED},
        nonzero={key: 1e-12 for key in JUDGED},
    )
    verdict = resolve_axis_indifference(
        summaries, ("also_zero", "nonzero"), null_point="null", metrics=JUDGED
    )
    assert verdict.indistinguishable == ("also_zero",)
    assert verdict.distinguishable == ("nonzero",)


# ----------------------------------------------------------------------
# the bridge is a same-team dial with an exact null
# ----------------------------------------------------------------------


def bridge_target_probe() -> dict[tuple[str, str], float]:
    """Bridge-implied latent targets, deliberately far from the moments.

    Chosen large and of both signs so that a weight that is meant to be
    ignored cannot be ignored by accident.
    """
    return {
        ("ast", "pts"): 0.40,
        ("reb", "reb"): -0.20,
        ("pts", "fg3m"): 0.35,
    }


def test_bridge_weight_zero_is_the_no_bridge_fit_exactly(standardized, role_moments):
    """The bridge's null is an identity, not a small perturbation."""
    kwargs = {"bootstrap": 80, "seed": 73, "role_moments": role_moments}
    spec = v2_spec("bridge0", r_symmetric=4, role_deviation=True, bridge_weight=0.0)
    without = fit_v2_factors(standardized, STATS, spec=spec, **kwargs)
    with_targets = fit_v2_factors(
        standardized,
        STATS,
        spec=spec,
        bridge_targets=bridge_target_probe(),
        **kwargs,
    )
    assert np.array_equal(with_targets.same_target, without.same_target)
    assert with_targets.loadings.to_payload() == without.loadings.to_payload()


def test_the_bridge_moves_the_same_team_target_and_only_its_own_entries(
    standardized, role_moments
):
    """At weight one the named entries land on the target; the rest do not move."""
    kwargs = {"bootstrap": 80, "seed": 73, "role_moments": role_moments}
    targets = bridge_target_probe()
    reference = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("nobridge", r_symmetric=4),
        **kwargs,
    )
    bridged = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("bridge1", r_symmetric=4, bridge_weight=1.0),
        bridge_targets=targets,
        **kwargs,
    )
    named = {
        tuple(sorted((STATS.index(first), STATS.index(second))))
        for first, second in targets
    }
    for (first, second), required in targets.items():
        i, j = STATS.index(first), STATS.index(second)
        assert bridged.same_target[i, j] == pytest.approx(required, abs=1e-12)
        assert bridged.same_target[j, i] == pytest.approx(required, abs=1e-12)
    for i in range(len(STATS)):
        for j in range(len(STATS)):
            if tuple(sorted((i, j))) in named:
                continue
            assert bridged.same_target[i, j] == reference.same_target[i, j]


@pytest.mark.parametrize("weight", [0.25, 1.0])
def test_the_bridge_cannot_reach_the_cross_team_block(
    standardized, role_moments, weight
):
    """A same-team target is a same-team target: ``A - B`` is bitwise unchanged."""
    kwargs = {"bootstrap": 80, "seed": 73, "role_moments": role_moments}
    reference = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec("nobridge", r_symmetric=4),
        **kwargs,
    )
    bridged = fit_v2_factors(
        standardized,
        STATS,
        spec=v2_spec(
            f"bridge{weight}",
            r_symmetric=4,
            role_deviation=True,
            bridge_weight=weight,
        ),
        bridge_targets=bridge_target_probe(),
        **kwargs,
    )
    assert np.array_equal(bridged.cross_target, reference.cross_target)
    assert np.array_equal(
        bridged.loadings.cross_team_correlation(),
        reference.loadings.cross_team_correlation(),
    )
    assert bridged.cross_team_unchanged_deviation() == 0.0


# ----------------------------------------------------------------------
# the temporal treatment replaces named entries and nothing else
# ----------------------------------------------------------------------


def test_temporal_overrides_replace_only_the_entries_they_name(
    standardized, role_moments
):
    """One stat pair is substituted, symmetrically, and the rest is the control."""
    kwargs = {"bootstrap": 80, "seed": 73, "role_moments": role_moments}
    spec = v2_spec("temporal", r_symmetric=4, role_deviation=True)
    reference = fit_v2_factors(standardized, STATS, spec=spec, **kwargs)
    index = STATS.index("ast")
    posterior = float(reference.same_target[index, index]) + 0.05
    overridden = fit_v2_factors(
        standardized,
        STATS,
        spec=spec,
        temporal_overrides={("ast", "ast"): posterior},
        **kwargs,
    )
    assert overridden.same_target[index, index] == pytest.approx(posterior, abs=1e-12)
    others = np.ones((len(STATS), len(STATS)), dtype=bool)
    others[index, index] = False
    assert np.array_equal(overridden.same_target[others], reference.same_target[others])
    # The override is a same-team statement; the cross-team block is untouched.
    assert np.array_equal(overridden.cross_target, reference.cross_target)
    assert np.array_equal(
        overridden.loadings.cross_team_correlation(),
        reference.loadings.cross_team_correlation(),
    )
    # And the base construction, which reads the cross-team estimator on both
    # blocks, is upstream of the override.
    assert np.array_equal(overridden.base_same_target, reference.base_same_target)


def test_temporal_overrides_ignore_a_stat_outside_the_model(standardized, role_moments):
    """An override naming a stat the fit does not carry is dropped, not an error."""
    kwargs = {"bootstrap": 80, "seed": 73, "role_moments": role_moments}
    spec = v2_spec("temporal_unknown", r_symmetric=4)
    reference = fit_v2_factors(standardized, STATS, spec=spec, **kwargs)
    overridden = fit_v2_factors(
        standardized,
        STATS,
        spec=spec,
        temporal_overrides={("ast", "tov"): 0.5},
        **kwargs,
    )
    assert np.array_equal(overridden.same_target, reference.same_target)
