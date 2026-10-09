"""Contracts for the shadow's production entry point.

Hermetic: synthetic marginals, a hand-built incumbent copula, a six-player
slate written to a temporary directory. No fitted binaries, no parquet from
the repository, no network.

These tests are about what the *deployed* path may and may not do. The model
is frozen and graded elsewhere. What is checked here is that deploying it
beside production cannot change what production serves: the incumbent is the
served number on every row and in every failure mode, the publishing switch
stays disabled, a deliberate publish attempt refuses, a candidate failure is
recorded rather than propagated, an incumbent failure propagates rather than
being replaced by a fabricated number, and nothing in the path can promote
anything.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.distributions import FittedMarginal, NegativeBinomialCalibrator
from nba_prop_quant.research.game_latent_state import shadow_runtime as runtime
from nba_prop_quant.research.game_latent_state.safety import (
    DECLARED_INTEGRATION_PATHS,
    SHADOW_OWNED_PRODUCTION_PATHS,
    stale_integration_declarations,
)
from nba_prop_quant.research.game_latent_state.simulator import SUPPORTED_STATS

PROJECT = Path(__file__).resolve().parents[1]

ENTRY_POINT_RELATIVE = "ops/run_production_shadow.py"
ENTRY_POINT = PROJECT / ENTRY_POINT_RELATIVE

FINAL_MODEL_SPEC_SHA256 = (
    "154109658127920ebeac381bd21e8bb29390505384124419899236cce37fdc64"
)
FACTOR_SPEC_HASH = "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"

MU = {"pts": 18.0, "reb": 6.5, "ast": 4.0, "stl": 1.0, "blk": 0.6, "fg3m": 2.0}

SIMULATIONS = 1_500


@pytest.fixture(scope="module")
def shadow_ops():
    """The entry point, imported as a module.

    Loaded by path because ``ops/`` is a script directory rather than a
    package, which is also how the lifecycle invokes it.
    """
    spec = importlib.util.spec_from_file_location("run_production_shadow", ENTRY_POINT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nb_marginal(size: float) -> FittedMarginal:
    model = NegativeBinomialCalibrator()
    model.size = size
    return FittedMarginal(kind="nb", model=model)


def synthetic_marginals() -> dict[str, FittedMarginal]:
    return {
        stat: nb_marginal(6.0 + index) for index, stat in enumerate(SUPPORTED_STATS)
    }


def synthetic_copula() -> GaussianCopula:
    n = len(SUPPORTED_STATS)
    block = np.full((n, n), 0.25, dtype=float)
    np.fill_diagonal(block, 1.0)
    fitted = GaussianCopula(targets=list(SUPPORTED_STATS))
    fitted.global_corr = block
    return fitted


def synthetic_slate() -> pd.DataFrame:
    players = ((10, 1), (11, 1), (12, 1), (13, 1), (20, 2), (21, 2), (22, 2), (23, 2))
    rng = np.random.default_rng(5)
    records = []
    for index, (player_id, team_id) in enumerate(players):
        record: dict[str, object] = {
            "game_id": 9001,
            "date": "2026-04-02",
            "player_id": player_id,
            "team_id": team_id,
            "is_home": int(team_id == 1),
            "expected_minutes": 32.0 - index,
            "role_bucket": ["starter", "starter", "rotation", "bench"][index % 4],
        }
        for stat in SUPPORTED_STATS:
            mu = MU[stat] * (1.0 - 0.04 * index)
            record[f"mu_selected_{stat}"] = mu
            record[stat] = int(rng.poisson(mu))
        records.append(record)
    return pd.DataFrame(records)


@pytest.fixture
def deployment(tmp_path):
    """A production-shaped fit workspace plus the status file naming it.

    The layout is the one ``ops/run_adaptive_daily_fit.py`` leaves behind, so
    the entry point resolves its inputs here exactly as it does in the
    lifecycle rather than through a test-only shortcut.
    """
    import joblib

    workspace = tmp_path / "staging"
    models = workspace / "candidate" / "models"
    processed = workspace / "processed"
    models.mkdir(parents=True)
    processed.mkdir(parents=True)

    joblib.dump(synthetic_marginals(), models / "marginals.joblib")
    joblib.dump(synthetic_copula(), models / "copula.joblib")
    synthetic_slate().to_parquet(processed / "oof_selected_means.parquet", index=False)

    status = tmp_path / "adaptive.json"
    status.write_text(
        json.dumps({"outcome": "completed", "workspace": str(workspace)}),
        encoding="utf-8",
    )
    return {
        "workspace": workspace,
        "models": models,
        "adaptive_status": status,
        "root": tmp_path,
    }


def invoke(shadow_ops, deployment, *extra: str) -> dict:
    status_path = (
        deployment["root"] / f"shadow-{len(list(deployment['root'].iterdir()))}.json"
    )
    code = shadow_ops.main(
        [
            "--slate-date",
            "2026-04-02",
            "--adaptive-status",
            str(deployment["adaptive_status"]),
            "--simulations",
            str(SIMULATIONS),
            "--log-path",
            str(deployment["root"] / "shadow_log.jsonl"),
            "--grading-path",
            str(deployment["root"] / "grading.json"),
            "--status-path",
            str(status_path),
            *extra,
        ]
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["exit_code"] = code
    return payload


# ----------------------------------------------------------------------
# the deployment itself
# ----------------------------------------------------------------------


def test_the_entry_point_is_an_owned_production_path():
    """The entry point stays on the record whether or not it is being changed.

    Ownership and permission are two different questions, which is why they
    are two mappings. Being on the owned registry is permanently true and buys
    nothing: a branch that changes this file still needs a fresh declaration
    and the review that comes with it. A declaration, when there is one, is
    about one pending change and expires with it --
    ``stale_integration_declarations`` is what enforces that, by requiring
    every declared path to be a path the branch actually modifies.

    So this asserts ownership unconditionally, and says nothing about whether
    a declaration happens to exist right now. It used to assert there was
    none, which read correctly on the branch that first deployed the shadow
    and then made the file permanently unextendable: adding the realized
    dependence reading the frozen policy's gates need a target for would have
    had to choose between an undeclared change to a protected path and a
    declaration this test forbade.
    """
    assert ENTRY_POINT.exists()
    assert ENTRY_POINT_RELATIVE in SHADOW_OWNED_PRODUCTION_PATHS
    assert SHADOW_OWNED_PRODUCTION_PATHS[ENTRY_POINT_RELATIVE].strip()

    if ENTRY_POINT_RELATIVE in DECLARED_INTEGRATION_PATHS:
        assert DECLARED_INTEGRATION_PATHS[ENTRY_POINT_RELATIVE].strip(), (
            "an empty declaration is permission with no reason attached"
        )
        assert ENTRY_POINT_RELATIVE not in stale_integration_declarations(PROJECT), (
            "the entry point is declared but this branch does not change it, "
            "which is standing permission rather than a pending change"
        )


def test_the_deployment_is_pinned_to_the_frozen_final_model(shadow_ops):
    """A shadow running a model nobody froze produces an uninterpretable log."""
    args = shadow_ops.parse_args(["--slate-date", "2026-04-02"])
    pinned = shadow_ops.check_the_model_on_disk_is_the_frozen_one(args)
    assert pinned["passed"]
    assert pinned["final_model_spec_sha256"] == FINAL_MODEL_SPEC_SHA256
    assert pinned["factor_spec_hash"] == FACTOR_SPEC_HASH
    assert pinned["factor_spec_hash_expected"] == FACTOR_SPEC_HASH
    assert pinned["final_model_status"] == "FROZEN"


def test_a_specification_that_is_not_the_frozen_one_is_refused(shadow_ops, tmp_path):
    spec = json.loads(
        (PROJECT / "research/final_upstream_remediation/factor_spec.json").read_text(
            encoding="utf-8"
        )
    )
    spec["spec_hash"] = "0" * 64
    tampered = tmp_path / "factor_spec.json"
    tampered.write_text(json.dumps(spec), encoding="utf-8")

    args = shadow_ops.parse_args(
        ["--slate-date", "2026-04-02", "--factor-spec", str(tampered)]
    )
    pinned = shadow_ops.check_the_model_on_disk_is_the_frozen_one(args)
    assert not pinned["passed"]
    assert not pinned["factor_spec_matches_the_final_model_specification"]


def test_a_declare_only_run_proves_the_wiring_without_a_slate(shadow_ops, tmp_path):
    status_path = tmp_path / "declare.json"
    code = shadow_ops.main(
        ["--slate-date", "2026-04-02", "--status-path", str(status_path)]
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert code == 0
    assert status["outcome"] == "DECLARE_ONLY_NO_SLATE_RESOLVED"
    assert status["pinned"]["passed"]
    assert status["authority"]["passed"]
    assert status["rows_published"] == 0


# ----------------------------------------------------------------------
# the served number
# ----------------------------------------------------------------------


def test_every_logged_row_serves_the_incumbent(shadow_ops, deployment):
    status = invoke(shadow_ops, deployment)
    assert status["outcome"] == "SHADOWED"
    assert status["shadow"]["events_shadowed"] > 0

    rows = [
        json.loads(line)
        for line in (deployment["root"] / "shadow_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows
    for row in rows:
        assert row["served_model"] == "incumbent"
        assert row["served_probability"] == row["incumbent_probability"]
        assert row["published"] is False


def test_every_logged_row_carries_the_provenance_of_what_produced_it(
    shadow_ops, deployment
):
    status = invoke(shadow_ops, deployment)
    fingerprint = status["shadow"]["provenance_fingerprint"]
    rows = [
        json.loads(line)
        for line in (deployment["root"] / "shadow_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in rows:
        assert row["provenance_fingerprint"] == fingerprint
        assert row["factor_spec_hash"] == FACTOR_SPEC_HASH
        assert row["dependence_model_version"]
    provenance = status["shadow"]["provenance"]
    assert provenance["final_model_spec_sha256"] == FINAL_MODEL_SPEC_SHA256
    assert provenance["marginal_source_sha256"]
    assert provenance["copula_source_sha256"]
    assert provenance["promotion_authority"] == runtime.PROMOTION_AUTHORITY
    assert provenance["published"] is False


def test_the_shadow_reads_the_artifacts_the_fit_registered(shadow_ops, deployment):
    """Not a parallel fit of its own: the same files, by digest."""
    from nba_prop_quant.research.game_latent_state.artifacts import sha256_file

    status = invoke(shadow_ops, deployment)
    shadow = status["shadow"]
    assert shadow["model_dir_resolved_from"] == "--adaptive-status workspace"
    assert shadow["slate_resolved_from"] == "--adaptive-status workspace"
    assert shadow["marginal_source_sha256"] == sha256_file(
        deployment["models"] / "marginals.joblib"
    )
    assert shadow["copula_source_sha256"] == sha256_file(
        deployment["models"] / "copula.joblib"
    )


# ----------------------------------------------------------------------
# the publishing switch
# ----------------------------------------------------------------------


def test_the_publishing_switch_is_disabled_after_deployment(shadow_ops, deployment):
    status = invoke(shadow_ops, deployment)
    authority = status["authority"]
    assert authority["switch"]["state"] == runtime.PUBLISHING_DISABLED
    assert authority["switch"]["enabled"] is False
    assert authority["published_authority"] == "incumbent"
    assert authority["rows_published"] == 0
    assert status["rows_published"] == 0
    assert status["served_authority"] == "incumbent"


def test_no_approval_has_been_granted_on_the_committed_switch():
    """The committed token is a statement that no approval exists.

    The field is not absent, and leaving it absent would be weaker: a reader
    could not tell a switch that was never approved from one whose token was
    dropped by accident. It holds a sentinel instead, and supplying that
    sentinel still does not enable publishing, because the state is disabled
    and the state is a separate reviewed commit.
    """
    payload = json.loads(
        (PROJECT / runtime.PUBLISHING_SWITCH_PATH).read_text(encoding="utf-8")
    )
    assert payload["state"] == runtime.PUBLISHING_DISABLED
    assert payload["approval_token"] == "no-approval-has-been-granted"
    assert payload["published_authority"] == "incumbent"
    assert payload["promotion_authority"] == runtime.PROMOTION_AUTHORITY

    switch = runtime.read_publishing_switch(
        PROJECT / runtime.PUBLISHING_SWITCH_PATH,
        environment={
            runtime.PUBLISH_ENV_VAR: runtime.PUBLISHING_ENABLED,
            runtime.PUBLISH_APPROVAL_ENV_VAR: payload["approval_token"],
        },
    )
    assert switch.approval_supplied is True
    assert switch.enabled is False
    assert "switch file state is DISABLED" in switch.blockers


def test_the_publish_environment_variable_cannot_enable_publishing_by_itself():
    """Three independent conditions, so no single mistake turns it on."""
    switch = runtime.read_publishing_switch(
        PROJECT / runtime.PUBLISHING_SWITCH_PATH,
        environment={runtime.PUBLISH_ENV_VAR: runtime.PUBLISHING_ENABLED},
    )
    assert switch.enabled is False
    switch = runtime.read_publishing_switch(
        PROJECT / runtime.PUBLISHING_SWITCH_PATH,
        environment={
            runtime.PUBLISH_ENV_VAR: runtime.PUBLISHING_DISABLED,
            runtime.PUBLISH_APPROVAL_ENV_VAR: "anything",
        },
    )
    assert switch.enabled is False


def test_a_deliberate_publish_attempt_refuses_and_the_refusal_is_recorded(
    shadow_ops, deployment
):
    status = invoke(shadow_ops, deployment)
    authority = status["authority"]
    assert authority["publish_attempt"] == "ShadowPublishingDisabled"
    assert "publishing is disabled" in authority["publish_refusal"]
    assert "The incumbent feed is unaffected." in authority["publish_refusal"]
    assert authority["promotion_attempt"] == "ShadowPromotionRefused"
    assert runtime.PROMOTION_AUTHORITY in authority["promotion_refusal"]


def test_no_autonomous_promotion_action_exists_in_the_deployed_path(shadow_ops):
    """Every promotion-shaped call reachable from the entry point refuses."""
    for context in ("", "the deployed lifecycle", "an operator"):
        with pytest.raises(runtime.ShadowPromotionRefused):
            runtime.assert_no_promotion_authority(context)
    record = shadow_ops.check_the_shadow_has_no_authority()
    assert record["passed"]
    assert record["promotion_authority"] == runtime.PROMOTION_AUTHORITY
    assert record["rows_published"] == 0


# ----------------------------------------------------------------------
# fallback: the served number survives every candidate failure
# ----------------------------------------------------------------------


def served_rows(deployment) -> list[dict]:
    return [
        json.loads(line)
        for line in (deployment["root"] / "shadow_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_fallback_a_a_healthy_candidate_still_serves_the_incumbent(
    shadow_ops, deployment
):
    status = invoke(shadow_ops, deployment)
    assert status["shadow"]["games_that_fell_back"] == 0
    rows = served_rows(deployment)
    assert all(row["fell_back"] is False for row in rows)
    assert all(row["candidate_probability"] is not None for row in rows)
    assert all(
        row["served_probability"] == row["incumbent_probability"] for row in rows
    )


def test_fallback_b_a_candidate_exception_is_recorded_not_propagated(
    shadow_ops, deployment, monkeypatch
):
    import nba_prop_quant.research.game_latent_state.shadow_runtime as module

    def explode(*args, **kwargs):
        raise RuntimeError("forced candidate failure")

    monkeypatch.setattr(module, "simulate_game", explode)
    status = invoke(shadow_ops, deployment)

    assert status["exit_code"] == 0
    assert status["outcome"] == "SHADOWED_WITH_FALLBACKS"
    assert status["shadow"]["games_that_fell_back"] == 1
    rows = served_rows(deployment)
    assert rows
    for row in rows:
        assert row["fell_back"] is True
        assert row["candidate_probability"] is None
        assert "forced candidate failure" in row["failure_reason"]
        # The whole contract: the candidate failing changes nothing about the
        # number that was served.
        assert row["served_model"] == "incumbent"
        assert row["served_probability"] == row["incumbent_probability"]


def test_fallback_c_a_psd_refusal_is_recorded_not_propagated(
    shadow_ops, deployment, monkeypatch
):
    """The covariance refusing to assemble is a fallback, not an outage."""
    import nba_prop_quant.research.game_latent_state.shadow_runtime as module

    def refuse(*args, **kwargs):
        raise ValueError(
            "player 10 cannot absorb any shared structure; the incumbent "
            "within-player block is numerically degenerate"
        )

    monkeypatch.setattr(module, "simulate_game", refuse)
    status = invoke(shadow_ops, deployment)

    assert status["exit_code"] == 0
    assert status["shadow"]["psd_failures"] == 1
    assert status["shadow"]["fallbacks"][0]["was_a_psd_refusal"] is True
    for row in served_rows(deployment):
        assert row["served_probability"] == row["incumbent_probability"]
        assert row["candidate_probability"] is None


def test_fallback_d_a_missing_candidate_artifact_does_not_fail_the_lifecycle(
    shadow_ops, deployment
):
    (deployment["models"] / "marginals.joblib").unlink()
    status = invoke(shadow_ops, deployment)

    assert status["exit_code"] == 0
    assert status["outcome"] == "SHADOW_FAILED"
    assert "marginals.joblib" in status["error"]
    assert "incumbent is unaffected" in status["error"]
    assert status["rows_published"] == 0
    assert status["production_serving_was_affected"] is False
    assert not (deployment["root"] / "shadow_log.jsonl").exists()


def test_strict_is_the_only_way_a_shadow_failure_becomes_a_nonzero_exit(
    shadow_ops, deployment
):
    (deployment["models"] / "copula.joblib").unlink()
    assert invoke(shadow_ops, deployment)["exit_code"] == 0
    assert invoke(shadow_ops, deployment, "--strict")["exit_code"] == 1


def test_an_incumbent_failure_propagates_rather_than_fabricating_a_number(
    deployment, monkeypatch
):
    """The one failure the shadow must not absorb.

    Falling back to the incumbent is only meaningful if the incumbent number
    is real. So when the incumbent arm itself breaks there is nothing to fall
    back to, and ``evaluate_shadow_game`` lets the exception out instead of
    serving a candidate number or a placeholder.
    """
    import joblib

    import nba_prop_quant.research.game_latent_state.shadow_runtime as module
    from nba_prop_quant.research.game_latent_state.covariance import (
        SharedFactorLoadings,
    )
    from nba_prop_quant.research.game_latent_state.factors import (
        incumbent_within_player_blocks,
    )
    from nba_prop_quant.research.game_latent_state.query import PropLeg
    from nba_prop_quant.research.game_latent_state.simulator import GameRoster

    marginals = joblib.load(deployment["models"] / "marginals.joblib")
    copula = joblib.load(deployment["models"] / "copula.joblib")
    frame = synthetic_slate()
    roster = GameRoster(
        game_id=9001,
        home_team_id=1,
        frame=frame,
        stats=SUPPORTED_STATS,
        role_column="role_bucket",
    )
    loadings = SharedFactorLoadings(
        stats=SUPPORTED_STATS,
        game=np.full((len(SUPPORTED_STATS), 2), 0.1),
        team_contrast=np.full(len(SUPPORTED_STATS), 0.05),
        role_scale={"starter": 0.9112, "rotation": 0.8251, "bench": 1.2567},
    )
    provenance = module.build_provenance(
        loadings=loadings,
        factor_spec={"spec_hash": FACTOR_SPEC_HASH},
        factor_spec_path=PROJECT / "research/final_model/final_model_spec.json",
        marginal_source="synthetic",
        copula_source="synthetic",
        simulations=SIMULATIONS,
        seed=73,
    )
    within = incumbent_within_player_blocks(
        copula, SUPPORTED_STATS, frame["player_id"].astype(int)
    )

    def explode(*args, **kwargs):
        raise RuntimeError("the incumbent arm is unavailable")

    monkeypatch.setattr(module, "incumbent_game_simulation", explode)

    with pytest.raises(RuntimeError, match="the incumbent arm is unavailable"):
        module.evaluate_shadow_game(
            roster,
            events={"probe": (PropLeg(10, "pts", "over", 17.5),)},
            marginals=marginals,
            copula=copula,
            loadings=loadings,
            provenance=provenance,
            config=module.ShadowConfig(simulations=SIMULATIONS, seed=73),
            within_player=within,
        )


# ----------------------------------------------------------------------
# automatic grading
# ----------------------------------------------------------------------


def test_the_deployed_path_persists_everything_grading_needs(shadow_ops, deployment):
    """Brier, log loss, leg breakdown and diagnostics, from the written files."""
    invoke(shadow_ops, deployment)
    report = json.loads(
        (deployment["root"] / "grading.json").read_text(encoding="utf-8")
    )

    assert report["served_model"] == "incumbent"
    assert report["published"] is False
    grading = report["grading"]
    for arm in ("candidate", "incumbent", "independence"):
        assert set(grading["by_model"][arm]) >= {"brier", "log_loss", "events"}
    assert set(grading["by_legs"]) >= {"2", "3", "4"}
    assert "candidate_fallback_rate" in grading
    assert grading["promotion_authority"] == runtime.PROMOTION_AUTHORITY

    assert report["numerical_diagnostics"]
    numerical = report["numerical_diagnostics"][0]
    assert "min_eigenvalue" in numerical
    assert "same_player_max_block_deviation" in numerical
    assert report["dependence_diagnostics"]
    assert "buckets" in json.dumps(report["dependence_diagnostics"])


def test_the_persisted_log_can_be_regraded_from_disk_alone(shadow_ops, deployment):
    """A report nobody can recompute is not a monitoring record.

    The log is re-read and regraded here, which is the operation a monitoring
    job would perform, so the columns it needs are proven present rather than
    assumed.
    """
    invoke(shadow_ops, deployment)
    rows = served_rows(deployment)
    log = pd.DataFrame(rows)
    realized = {
        row["event_id"]: row["realized"] for row in rows if row["realized"] is not None
    }
    assert realized

    regraded = runtime.grade_shadow_log(log, realized)
    assert regraded["events_graded"] == len(realized)
    assert regraded["served_model"] == "incumbent"
    assert regraded["published"] is False
    for arm in ("candidate", "incumbent", "independence"):
        assert regraded["by_model"][arm]["events"] > 0


def test_grading_never_credits_the_candidate_for_a_row_it_did_not_answer(
    shadow_ops, deployment, monkeypatch
):
    import nba_prop_quant.research.game_latent_state.shadow_runtime as module

    def explode(*args, **kwargs):
        raise RuntimeError("forced candidate failure")

    monkeypatch.setattr(module, "simulate_game", explode)
    status = invoke(shadow_ops, deployment)

    grading = status["shadow"]["grading"]
    assert grading["by_model"]["candidate"] is None
    assert grading["by_model"]["incumbent"]["events"] > 0
    assert grading["candidate_fallback_rate"] == 1.0


def test_a_grading_verdict_is_refused_even_when_the_numbers_are_available(
    shadow_ops, deployment
):
    invoke(shadow_ops, deployment)
    rows = served_rows(deployment)
    log = pd.DataFrame(rows)
    realized = {row["event_id"]: row["realized"] for row in rows}
    with pytest.raises(runtime.ShadowPromotionRefused):
        runtime.grade_shadow_log(log, realized, promotion_verdict=True)
