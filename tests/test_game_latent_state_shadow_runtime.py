"""Tests for the controlled production shadow runtime.

Hermetic: synthetic marginals, synthetic rosters, a hand-built incumbent
copula. No parquet, no fitted binaries, no network.

These tests are about the *contract*, not the model. The model is frozen and
graded elsewhere. What is tested here is the set of things the shadow must be
unable to do: serve the candidate, publish, promote itself, fabricate an
incumbent number, credit itself on a row it did not answer, or report a
probability without saying which model and which artifacts produced it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.distributions import FittedMarginal, NegativeBinomialCalibrator
from nba_prop_quant.research.game_latent_state.covariance import SharedFactorLoadings
from nba_prop_quant.research.game_latent_state.query import PropLeg
from nba_prop_quant.research.game_latent_state.shadow_runtime import (
    CANDIDATE,
    INCUMBENT,
    INDEPENDENCE,
    PROMOTION_AUTHORITY,
    PUBLISH_APPROVAL_ENV_VAR,
    PUBLISH_ENV_VAR,
    PUBLISHING_DISABLED,
    PUBLISHING_ENABLED,
    PUBLISHING_SWITCH_PATH,
    ShadowConfig,
    ShadowPromotionRefused,
    ShadowProvenance,
    ShadowPublishingDisabled,
    assert_no_promotion_authority,
    build_provenance,
    evaluate_shadow_game,
    grade_shadow_log,
    incumbent_game_simulation,
    psd_diagnostics,
    publish_shadow_probabilities,
    read_publishing_switch,
    shadow_log_frame,
)
from nba_prop_quant.research.game_latent_state.simulator import (
    SUPPORTED_STATS,
    GameRoster,
    simulate_game,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MU = {"pts": 18.0, "reb": 6.5, "ast": 4.0, "stl": 1.0, "blk": 0.6, "fg3m": 2.0}

SIMULATIONS = 2_000


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


@pytest.fixture(scope="module")
def roster() -> GameRoster:
    players = ((10, 1), (11, 1), (12, 1), (20, 2), (21, 2), (22, 2))
    records = []
    for index, (player_id, team_id) in enumerate(players):
        record: dict[str, object] = {
            "player_id": player_id,
            "team_id": team_id,
            "expected_minutes": 30.0 - index,
            "is_home": int(team_id == 1),
            "role_bucket": "starter" if index < 2 else "rotation" if index < 4 else "bench",
        }
        for stat in SUPPORTED_STATS:
            record[f"mu_selected_{stat}"] = MU[stat] * (1.0 - 0.05 * index)
        records.append(record)
    return GameRoster(
        game_id=9001,
        home_team_id=1,
        frame=pd.DataFrame(records),
        stats=SUPPORTED_STATS,
        role_column="role_bucket",
    )


@pytest.fixture(scope="module")
def copula() -> GaussianCopula:
    n = len(SUPPORTED_STATS)
    block = np.full((n, n), 0.25, dtype=float)
    np.fill_diagonal(block, 1.0)
    fitted = GaussianCopula(targets=list(SUPPORTED_STATS))
    fitted.global_corr = block
    return fitted


@pytest.fixture(scope="module")
def loadings() -> SharedFactorLoadings:
    stats = SUPPORTED_STATS
    rng = np.random.default_rng(11)
    game = np.round(rng.uniform(0.05, 0.22, size=(len(stats), 2)), 3)
    game[:, 1] *= np.where(np.arange(len(stats)) % 2 == 0, 1.0, -1.0)
    contrast = np.round(rng.uniform(-0.10, 0.10, size=len(stats)), 3)
    return SharedFactorLoadings(
        stats=stats,
        game=game,
        team_contrast=contrast,
        role_scale={"starter": 0.9112, "rotation": 0.8251, "bench": 1.2567},
    )


@pytest.fixture(scope="module")
def provenance(loadings) -> ShadowProvenance:
    return ShadowProvenance(
        dependence_model_version="game-latent-state-shadow-v1",
        factor_spec_hash="3229bbc8",
        factor_spec_sha256="a" * 64,
        final_model_spec_sha256="b" * 64,
        code_sha="c" * 40,
        marginal_source="synthetic",
        marginal_source_sha256=None,
        copula_source="synthetic",
        copula_source_sha256=None,
        role_scale=dict(loadings.role_scale),
        stats=SUPPORTED_STATS,
        simulations=SIMULATIONS,
        seed=73,
        built_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture(scope="module")
def events(roster) -> dict[str, tuple[PropLeg, ...]]:
    return {
        "same_player_2leg": (
            PropLeg(10, "pts", "over", 17.5),
            PropLeg(10, "ast", "over", 3.5),
        ),
        "same_team_2leg": (
            PropLeg(10, "pts", "over", 17.5),
            PropLeg(11, "reb", "over", 5.5),
        ),
        "cross_team_3leg": (
            PropLeg(10, "pts", "over", 17.5),
            PropLeg(20, "reb", "under", 6.5),
            PropLeg(21, "ast", "over", 2.5),
        ),
        "four_leg_mixed": (
            PropLeg(10, "pts", "over", 15.5),
            PropLeg(10, "reb", "over", 4.5),
            PropLeg(12, "ast", "over", 2.5),
            PropLeg(22, "pts", "under", 20.5),
        ),
    }


@pytest.fixture(scope="module")
def within(roster, copula) -> dict[int, np.ndarray]:
    from nba_prop_quant.research.game_latent_state.factors import (
        incumbent_within_player_blocks,
    )

    return incumbent_within_player_blocks(
        copula, SUPPORTED_STATS, roster.frame["player_id"].astype(int)
    )


@pytest.fixture(scope="module")
def shadow(roster, events, marginals, copula, loadings, provenance, within):
    return evaluate_shadow_game(
        roster,
        events=events,
        marginals=marginals,
        copula=copula,
        loadings=loadings,
        provenance=provenance,
        config=ShadowConfig(simulations=SIMULATIONS, seed=73),
        within_player=within,
    )


# ----------------------------------------------------------------------
# no promotion authority
# ----------------------------------------------------------------------


def test_the_shadow_has_no_promotion_authority():
    with pytest.raises(ShadowPromotionRefused) as raised:
        assert_no_promotion_authority()
    assert PROMOTION_AUTHORITY in str(raised.value)


@pytest.mark.parametrize("context", ["", "a caller with a good reason", "operator"])
def test_no_argument_makes_the_promotion_check_return(context):
    """There is no reason good enough. The signature has no success path."""
    with pytest.raises(ShadowPromotionRefused):
        assert_no_promotion_authority(context)


def test_grading_refuses_to_render_a_promotion_verdict(shadow, provenance):
    log = shadow_log_frame([shadow], provenance)
    realized = dict.fromkeys(log["event_id"], 1)
    with pytest.raises(ShadowPromotionRefused):
        grade_shadow_log(log, realized, promotion_verdict=True)


def test_the_grading_report_states_it_has_no_promotion_authority(shadow, provenance):
    log = shadow_log_frame([shadow], provenance)
    grades = grade_shadow_log(log, dict.fromkeys(log["event_id"], 1))
    assert grades["promotion_authority"] == PROMOTION_AUTHORITY
    assert grades["served_model"] == INCUMBENT
    assert grades["published"] is False


# ----------------------------------------------------------------------
# the publishing switch
# ----------------------------------------------------------------------


def test_the_committed_switch_is_disabled():
    path = PROJECT_ROOT / PUBLISHING_SWITCH_PATH
    assert path.exists(), f"the declared switch must be committed at {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == PUBLISHING_DISABLED
    assert payload["published_authority"] == "incumbent"
    assert payload["promotion_authority"] == PROMOTION_AUTHORITY


def test_the_committed_switch_reads_as_disabled_even_with_the_env_set():
    """The file is the authority. Environment alone cannot enable it."""
    path = PROJECT_ROOT / PUBLISHING_SWITCH_PATH
    token = json.loads(path.read_text(encoding="utf-8"))["approval_token"]
    switch = read_publishing_switch(
        path,
        environment={
            PUBLISH_ENV_VAR: PUBLISHING_ENABLED,
            PUBLISH_APPROVAL_ENV_VAR: token,
        },
    )
    assert switch.state == PUBLISHING_DISABLED
    assert switch.enabled is False


def test_a_missing_switch_reads_as_disabled_not_as_permission(tmp_path):
    for path in (None, tmp_path / "nope.json"):
        switch = read_publishing_switch(path, environment={})
        assert switch.state == PUBLISHING_DISABLED
        assert switch.enabled is False
        assert "disabled" in switch.reason


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {PUBLISH_ENV_VAR: PUBLISHING_ENABLED},
        {PUBLISH_APPROVAL_ENV_VAR: "token"},
        {PUBLISH_ENV_VAR: PUBLISHING_DISABLED, PUBLISH_APPROVAL_ENV_VAR: "token"},
        {PUBLISH_ENV_VAR: PUBLISHING_ENABLED, PUBLISH_APPROVAL_ENV_VAR: "wrong"},
    ],
)
def test_enabling_needs_all_three_independent_conditions(tmp_path, environment):
    path = tmp_path / "switch.json"
    path.write_text(
        json.dumps({"state": PUBLISHING_ENABLED, "approval_token": "token"}),
        encoding="utf-8",
    )
    switch = read_publishing_switch(path, environment=environment)
    assert switch.enabled is False
    assert switch.blockers


def test_publishing_is_refused_while_the_switch_is_disabled(shadow, tmp_path):
    switch = read_publishing_switch(tmp_path / "absent.json", environment={})
    with pytest.raises(ShadowPublishingDisabled) as raised:
        publish_shadow_probabilities(shadow.decisions, switch)
    message = str(raised.value)
    assert "disabled" in message
    assert "incumbent feed is unaffected" in message


def test_publishing_is_refused_even_when_all_three_conditions_are_met(
    shadow, tmp_path
):
    """Flipping the switch cannot by itself publish anything.

    Activation is a follow-up change that has to build a publisher. Until it
    does, an operator who satisfies every gate still gets a refusal rather
    than a surprise feed.
    """
    path = tmp_path / "switch.json"
    path.write_text(
        json.dumps({"state": PUBLISHING_ENABLED, "approval_token": "token"}),
        encoding="utf-8",
    )
    switch = read_publishing_switch(
        path,
        environment={
            PUBLISH_ENV_VAR: PUBLISHING_ENABLED,
            PUBLISH_APPROVAL_ENV_VAR: "token",
        },
    )
    assert switch.enabled is True
    with pytest.raises(ShadowPublishingDisabled) as raised:
        publish_shadow_probabilities(shadow.decisions, switch)
    assert "no shadow publisher has been built" in str(raised.value)


# ----------------------------------------------------------------------
# the incumbent is what is served
# ----------------------------------------------------------------------


def test_every_row_serves_the_incumbent(shadow):
    assert shadow.decisions
    assert shadow.fell_back is False
    for decision in shadow.decisions:
        assert decision.served_model == INCUMBENT
        assert decision.served_probability == decision.incumbent_probability


def test_the_candidate_number_is_recorded_but_never_served(shadow):
    """Both numbers are on the row; only one of them is the served one."""
    differing = [
        decision
        for decision in shadow.decisions
        if decision.candidate_probability is not None
        and decision.candidate_probability != decision.incumbent_probability
    ]
    assert differing, "the two models should disagree on at least one event"
    for decision in differing:
        assert decision.served_probability != decision.candidate_probability
        assert decision.served_probability == decision.incumbent_probability


def test_the_candidate_moves_cross_player_events_off_the_independence_arm(
    roster, marginals, within
):
    """Sanity: the shadow is actually exercising the dependence layer.

    Tested paired rather than per-event. The candidate and the independence
    arm share a seed, a within-player block and a dimension order, so the two
    draw sets are coupled and the difference of their indicators carries far
    less noise than either probability does on its own. Differencing the two
    *unpaired* probabilities off a shadow row at a realistic simulation count
    mostly measures Monte Carlo error, which is why those rows carry their
    standard errors instead.

    Loadings local to this test, not the module fixture: this asserts that a
    same-team conjunction moves when there *is* same-team dependence, so it
    needs loadings whose teammate correlation is unambiguously non-zero. The
    frozen model's magnitudes are measured in the ablation and the holdout
    artifacts, not here.
    """
    from nba_prop_quant.research.game_latent_state.query import leg_mask

    stats = SUPPORTED_STATS
    coupled = SharedFactorLoadings(
        stats=stats,
        game=np.full((len(stats), 1), 0.45),
        team_contrast=np.zeros(len(stats)),
    )
    assert coupled.same_team_correlation()[0, 0] > 0.15

    legs = (PropLeg(10, "pts", "over", 17.5), PropLeg(11, "pts", "over", 15.5))
    simulations = 20_000
    indicators = {}
    for name, arm in {
        "candidate": coupled,
        "independence": SharedFactorLoadings.independent(stats, k_game=1),
    }.items():
        simulation = simulate_game(
            roster,
            marginals=marginals,
            loadings=arm,
            within_player=within,
            simulations=simulations,
            seed=73,
        )
        satisfied = np.ones(simulations, dtype=bool)
        for leg in legs:
            satisfied &= leg_mask(simulation, leg)
        indicators[name] = satisfied.astype(float)

    paired = indicators["candidate"] - indicators["independence"]
    standard_error = float(np.std(paired, ddof=1) / np.sqrt(simulations))
    # Positive same-team coupling raises a two-over conjunction.
    assert float(np.mean(paired)) > 5.0 * standard_error


def test_the_shadow_prices_arbitrary_leg_counts(shadow):
    assert {len(decision.legs) for decision in shadow.decisions} == {2, 3, 4}


def test_an_empty_event_set_produces_no_rows(
    roster, marginals, copula, loadings, provenance
):
    result = evaluate_shadow_game(
        roster,
        events={},
        marginals=marginals,
        copula=copula,
        loadings=loadings,
        provenance=provenance,
        config=ShadowConfig(simulations=SIMULATIONS),
    )
    assert result.decisions == []


# ----------------------------------------------------------------------
# failing closed
# ----------------------------------------------------------------------


def test_a_broken_candidate_path_falls_back_to_the_incumbent(
    roster, events, marginals, copula, provenance, within, monkeypatch
):
    """Every row still carries a probability, and it is the incumbent's."""
    import nba_prop_quant.research.game_latent_state.shadow_runtime as runtime

    def explode(*args, **kwargs):
        raise np.linalg.LinAlgError("the covariance is not decomposable")

    monkeypatch.setattr(runtime, "simulate_game", explode)

    result = evaluate_shadow_game(
        roster,
        events=events,
        marginals=marginals,
        copula=copula,
        loadings=SharedFactorLoadings.independent(SUPPORTED_STATS, k_game=2),
        provenance=provenance,
        config=ShadowConfig(simulations=SIMULATIONS),
        within_player=within,
    )

    assert result.fell_back is True
    assert "LinAlgError" in result.failure_reason
    assert len(result.decisions) == len(events)
    for decision in result.decisions:
        assert decision.fell_back is True
        assert decision.candidate_probability is None
        assert decision.served_model == INCUMBENT
        assert 0.0 <= decision.served_probability <= 1.0
        assert decision.served_probability == decision.incumbent_probability
        assert "LinAlgError" in decision.failure_reason


def test_a_broken_incumbent_path_raises_rather_than_inventing_a_number(
    roster, events, marginals, loadings, provenance, within
):
    """There is nothing to fall back *to*, so this must not be swallowed."""

    class BrokenCopula:
        targets: ClassVar[list[str]] = list(SUPPORTED_STATS)

        def correlation_for_player(self, player_id):
            return np.eye(len(SUPPORTED_STATS))

        def simulate(self, *args, **kwargs):
            raise RuntimeError("the incumbent copula is unavailable")

    with pytest.raises(RuntimeError, match="incumbent copula is unavailable"):
        evaluate_shadow_game(
            roster,
            events=events,
            marginals=marginals,
            copula=BrokenCopula(),
            loadings=loadings,
            provenance=provenance,
            config=ShadowConfig(simulations=SIMULATIONS),
            within_player=within,
        )


def test_a_fallback_row_is_not_graded_for_the_candidate(
    roster, events, marginals, copula, provenance, within, monkeypatch
):
    """The one way this report could lie is by crediting a fallback row."""
    import nba_prop_quant.research.game_latent_state.shadow_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "simulate_game",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("down")),
    )
    broken = evaluate_shadow_game(
        roster,
        events=events,
        marginals=marginals,
        copula=copula,
        loadings=SharedFactorLoadings.independent(SUPPORTED_STATS, k_game=2),
        provenance=provenance,
        config=ShadowConfig(simulations=SIMULATIONS),
        within_player=within,
    )
    monkeypatch.undo()

    log = shadow_log_frame([broken], provenance)
    grades = grade_shadow_log(log, dict.fromkeys(log["event_id"], 1))

    assert grades["by_model"][CANDIDATE] is None
    assert grades["by_model"][INCUMBENT]["events"] == len(events)
    assert grades["events_where_the_candidate_fell_back"] == len(events)
    assert grades["candidate_fallback_rate"] == 1.0


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------


def test_the_provenance_is_immutable(provenance):
    """Both halves: the attributes and the mapping behind one of them.

    ``frozen=True`` only stops attribute rebinding. The role scale arrives as
    an ordinary dict from most callers, so the dataclass has to copy it into a
    read-only mapping itself -- otherwise a caller could edit provenance after
    it had already been fingerprinted and logged.
    """
    with pytest.raises(dataclasses.FrozenInstanceError):
        provenance.factor_spec_hash = "tampered"  # type: ignore[misc]
    with pytest.raises(TypeError):
        provenance.role_scale["bench"] = 2.0  # type: ignore[index]
    assert isinstance(provenance.role_scale, MappingProxyType)


def test_the_provenance_copies_the_mapping_it_was_handed(loadings):
    """Editing the caller's dict afterwards must not move the fingerprint."""
    mutable = dict(loadings.role_scale)
    built = ShadowProvenance(
        dependence_model_version="v1",
        factor_spec_hash="h",
        factor_spec_sha256="a" * 64,
        final_model_spec_sha256=None,
        code_sha=None,
        marginal_source="m",
        marginal_source_sha256=None,
        copula_source="c",
        copula_source_sha256=None,
        role_scale=mutable,
        stats=SUPPORTED_STATS,
        simulations=1,
        seed=1,
        built_at="2026-01-01T00:00:00+00:00",
    )
    before = built.fingerprint
    mutable["bench"] = 99.0
    assert built.role_scale["bench"] != 99.0
    assert built.fingerprint == before


def test_the_provenance_fingerprint_moves_when_anything_it_covers_moves(provenance):
    import dataclasses

    baseline = provenance.fingerprint
    for field_name, value in [
        ("factor_spec_hash", "different"),
        ("code_sha", "d" * 40),
        ("simulations", 10_000),
        ("seed", 74),
        ("final_model_spec_sha256", "e" * 64),
        ("marginal_source", "something else"),
    ]:
        moved = dataclasses.replace(provenance, **{field_name: value})
        assert moved.fingerprint != baseline, field_name


def test_the_fingerprint_is_stable_for_the_same_configuration(provenance):
    import dataclasses

    assert dataclasses.replace(provenance).fingerprint == provenance.fingerprint


def test_build_provenance_hashes_its_inputs_rather_than_naming_them(
    tmp_path, loadings
):
    spec_path = tmp_path / "factor_spec.json"
    spec = {
        "spec_hash": "3229bbc8",
        "dependence_model_version": "game-latent-state-shadow-v1",
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    final_path = tmp_path / "final_model_spec.json"
    final_path.write_text('{"a": 1}', encoding="utf-8")

    built = build_provenance(
        loadings=loadings,
        factor_spec=spec,
        factor_spec_path=spec_path,
        marginal_source="production refit",
        copula_source="incumbent",
        simulations=SIMULATIONS,
        seed=73,
        final_model_spec_path=final_path,
        code_sha="abc",
        built_at="2026-01-01T00:00:00+00:00",
    )

    assert len(built.factor_spec_sha256) == 64
    assert len(built.final_model_spec_sha256) == 64
    assert built.factor_spec_hash == "3229bbc8"
    assert built.promotion_authority == PROMOTION_AUTHORITY
    assert built.published is False
    assert built.role_scale == {
        key: float(value) for key, value in dict(loadings.role_scale).items()
    }

    # A missing optional artifact resolves to None rather than to a hash of
    # nothing, so an absent input cannot masquerade as a recorded one.
    assert (
        build_provenance(
            loadings=loadings,
            factor_spec=spec,
            factor_spec_path=spec_path,
            marginal_source="x",
            copula_source="y",
            simulations=1,
            seed=1,
            final_model_spec_path=tmp_path / "absent.json",
        ).final_model_spec_sha256
        is None
    )


def test_every_logged_row_carries_the_provenance_fingerprint(shadow, provenance):
    log = shadow_log_frame([shadow], provenance)
    assert set(log["provenance_fingerprint"]) == {provenance.fingerprint}
    assert set(log["factor_spec_hash"]) == {provenance.factor_spec_hash}
    assert set(log["dependence_model_version"]) == {
        provenance.dependence_model_version
    }
    assert not log["published"].any()


# ----------------------------------------------------------------------
# side-by-side logging and grading
# ----------------------------------------------------------------------


def test_the_log_holds_all_three_arms_on_one_row(shadow, provenance):
    log = shadow_log_frame([shadow], provenance)
    assert len(log) == len(shadow.decisions)
    for column in (
        "candidate_probability",
        "incumbent_probability",
        "independence_probability",
        "served_probability",
        "served_model",
        "leg_count",
        "min_eigenvalue",
        "same_player_max_block_deviation",
    ):
        assert column in log.columns
    assert log["candidate_probability"].notna().all()
    assert log["incumbent_probability"].notna().all()
    assert log["independence_probability"].notna().all()


def test_grading_reports_brier_and_log_loss_for_every_arm(shadow, provenance):
    log = shadow_log_frame([shadow], provenance)
    rng = np.random.default_rng(5)
    realized = {
        event_id: int(rng.random() < 0.4) for event_id in log["event_id"]
    }
    grades = grade_shadow_log(log, realized)

    assert grades["events_graded"] == len(log)
    assert grades["events_without_a_realized_outcome"] == 0
    for model in (CANDIDATE, INCUMBENT, INDEPENDENCE):
        scores = grades["by_model"][model]
        assert 0.0 <= scores["brier"] <= 1.0
        assert scores["log_loss"] > 0.0
        assert scores["events"] == len(log)
    assert set(grades["by_legs"]) == {"2", "3", "4"}
    assert grades["provenance_fingerprints"] == [provenance.fingerprint]


def test_an_ungraded_row_is_counted_rather_than_dropped_silently(shadow, provenance):
    log = shadow_log_frame([shadow], provenance)
    partial = {log["event_id"].iloc[0]: 1}
    grades = grade_shadow_log(log, partial)
    assert grades["events_logged"] == len(log)
    assert grades["events_graded"] == 1
    assert grades["events_without_a_realized_outcome"] == len(log) - 1


def test_grading_an_empty_log_is_not_an_error():
    grades = grade_shadow_log(pd.DataFrame(), {})
    assert grades["events_graded"] == 0
    assert grades["promotion_authority"] == PROMOTION_AUTHORITY


def test_the_log_loss_floor_is_applied_and_declared(shadow, provenance):
    """A zero-probability row that happens must not produce an infinity."""
    log = shadow_log_frame([shadow], provenance)
    log.loc[:, "candidate_probability"] = 0.0
    grades = grade_shadow_log(log, dict.fromkeys(log["event_id"], 1), floor=1e-6)
    assert np.isfinite(grades["by_model"][CANDIDATE]["log_loss"])
    assert grades["log_loss_floor"] == 1e-6


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------


def test_the_psd_diagnostics_are_recorded_per_game(shadow):
    numerical = shadow.numerical
    assert numerical["min_eigenvalue"] > 0.0
    assert numerical["min_residual_eigenvalue"] >= -1e-10
    assert numerical["players_checked"] == 6
    assert numerical["dimensions"] == 6 * len(SUPPORTED_STATS)
    assert "correlation_min_eigenvalue_recomputed" in numerical


def test_the_same_player_block_is_read_back_out_of_the_assembled_covariance(
    roster, marginals, loadings, within
):
    """The integration contract, verified against the matrix, not the intent.

    The role scale multiplies shared cross-player loadings. The one way it
    could break the contract is by leaving a shared-factor echo inside a
    player's own block, so the check reads the block back rather than
    trusting that it was pinned.
    """
    simulation = simulate_game(
        roster,
        marginals=marginals,
        loadings=loadings,
        within_player=within,
        simulations=64,
        seed=73,
    )
    diagnostics = psd_diagnostics(simulation.covariance, within)
    assert diagnostics["same_player_max_block_deviation"] < 1e-9


def test_the_eigenvalue_spectrum_is_a_declared_switch(
    roster, events, marginals, copula, loadings, provenance, within
):
    cheap = evaluate_shadow_game(
        roster,
        events=events,
        marginals=marginals,
        copula=copula,
        loadings=loadings,
        provenance=provenance,
        config=ShadowConfig(simulations=256, collect_eigenvalues=False),
        within_player=within,
    )
    assert "correlation_min_eigenvalue_recomputed" not in cheap.numerical
    assert "min_eigenvalue" in cheap.numerical


def test_the_dependence_diagnostics_report_the_model_and_every_arm(shadow):
    dependence = shadow.dependence
    assert dependence["buckets"]
    assert set(dependence["simulated_by_model"]) == {
        CANDIDATE,
        INCUMBENT,
        INDEPENDENCE,
    }
    for bucket in dependence["buckets"]:
        assert bucket in dependence["model_implied"]
        assert bucket in dependence["simulated_by_model"][CANDIDATE]

    # The incumbent and the independence reference are both cross-player
    # independent, so their cross-team buckets sit at zero while the
    # candidate's do not. That contrast is the point of carrying three arms.
    candidate = dependence["simulated_by_model"][CANDIDATE]
    incumbent = dependence["simulated_by_model"][INCUMBENT]
    cross = [b for b in dependence["buckets"] if "cross" in b or "opponent" in b]
    assert cross
    assert max(abs(incumbent[b]) for b in cross) < 0.05
    assert max(abs(candidate[b]) for b in cross) > max(
        abs(incumbent[b]) for b in cross
    )


def test_the_dependence_diagnostics_carry_the_role_scale(shadow, loadings):
    assert shadow.dependence["role_scale"] == {
        key: float(value) for key, value in dict(loadings.role_scale).items()
    }


def test_the_independence_arm_can_be_switched_off(
    roster, events, marginals, copula, loadings, provenance, within
):
    result = evaluate_shadow_game(
        roster,
        events=events,
        marginals=marginals,
        copula=copula,
        loadings=loadings,
        provenance=provenance,
        config=ShadowConfig(simulations=256, evaluate_independence=False),
        within_player=within,
    )
    assert all(
        decision.independence_probability is None for decision in result.decisions
    )
    assert INDEPENDENCE not in result.dependence["simulated_by_model"]


# ----------------------------------------------------------------------
# the incumbent arm is the production path
# ----------------------------------------------------------------------


def test_the_incumbent_arm_is_the_production_per_player_copula(
    roster, marginals, copula
):
    """Stacked per-player production draws, not a reimplementation."""
    stacked = incumbent_game_simulation(
        roster, marginals=marginals, copula=copula, simulations=512, seed=73
    )
    assert stacked.covariance is None
    assert stacked.player_ids == tuple(roster.frame["player_id"].astype(int))

    mu_columns = {stat: f"mu_selected_{stat}" for stat in roster.stats}
    for player_index, (_, row) in enumerate(roster.frame.iterrows()):
        direct = copula.simulate(
            row,
            marginals=dict(marginals),
            mu_columns=mu_columns,
            simulations=512,
            seed=73 + 1000 * player_index,
        )
        for stat_index, stat in enumerate(roster.stats):
            assert np.array_equal(
                stacked.draws[:, player_index, stat_index],
                direct[stat].to_numpy(dtype=float),
            )


def test_the_incumbent_arm_is_deterministic_in_the_seed(roster, marginals, copula):
    first = incumbent_game_simulation(
        roster, marginals=marginals, copula=copula, simulations=256, seed=73
    )
    again = incumbent_game_simulation(
        roster, marginals=marginals, copula=copula, simulations=256, seed=73
    )
    assert np.array_equal(first.draws, again.draws)


def test_the_shadow_is_deterministic_in_the_seed(
    roster, events, marginals, copula, loadings, provenance, within
):
    def run():
        return evaluate_shadow_game(
            roster,
            events=events,
            marginals=marginals,
            copula=copula,
            loadings=loadings,
            provenance=provenance,
            config=ShadowConfig(simulations=512, seed=73),
            within_player=within,
        )

    first, again = run(), run()
    assert [d.payload() for d in first.decisions] == [
        d.payload() for d in again.decisions
    ]
