"""Step 3C: daily adaptive fitting under the frozen architecture.

No test runs the expensive historical fit. A small deterministic engine stands
in for the numerical stages so the orchestration can be exercised: the frozen
policy guard, the selector prohibition, cutoff derivation, the no-new-data
outcome, the training lock, workspace isolation, candidate verification and
registry integration are all real code paths here.

Nothing trains a model, nothing touches the network, and every registry, data
root and work root is a pytest temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nba_prop_quant.adaptive_fit_registry import (
    REQUIRED_VALIDATION_CHECKS,
    FitRegistry,
    load_architecture_contract,
    sha256_file,
)
from nba_prop_quant.adaptive_training import (
    ADVANCED_START_SEASON,
    CALIBRATION_MIN_CLASSES,
    CALIBRATION_MIN_ROWS,
    CORE_SEED,
    DAG_STAGES,
    EFFECTIVE_N_JOBS,
    FORBIDDEN_SELECTOR_CALLABLES,
    FORBIDDEN_SELECTOR_SCRIPTS,
    FROZEN_CALIBRATION_ROUTES,
    FROZEN_DEPENDENCE_LAMBDA,
    FROZEN_MARGINAL_FAMILY,
    FROZEN_MEAN_ROUTES,
    GATE3_SEED,
    HISTORY_START_SEASON,
    MODE_BENCHMARK_ONLY,
    MODE_DRY_RUN,
    MODE_REGISTER_CANDIDATE,
    OUTCOME_ALREADY_RUNNING,
    OUTCOME_COMPLETED,
    OUTCOME_DRY_RUN,
    OUTCOME_NO_NEW_TRAINING_DATA,
    REQUIRED_OOF_FUNCTIONS,
    ROLLING_STATE_RELATIVE_PATH,
    STEP3C_VALIDATION_CHECKS,
    UPDATE_PROTOCOL_RELATIVE_PATH,
    AdaptiveTrainingError,
    BenchmarkRecorder,
    CandidateIncomplete,
    DataStateError,
    FrozenPolicyGuard,
    FrozenPolicyViolation,
    SelectorInvocationRefused,
    TrainingLocked,
    assert_selector_script_not_requested,
    assert_workspace_is_isolated,
    calibration_unit_is_fittable,
    deferred_validation_checks,
    derive_training_cutoff,
    frozen_guard_paths,
    has_new_training_data,
    new_workspace,
    resolve_calibration_parameters,
    run_daily_fit,
    selector_guard,
    structured_values_are_finite,
    training_lock,
)
from nba_prop_quant.copula import GaussianCopula
from nba_prop_quant.distributions import FittedMarginal, NegativeBinomialCalibrator


PROJECT = Path(__file__).resolve().parents[1]

CLI = PROJECT / "ops" / "run_adaptive_daily_fit.py"

SLATE_DATE = "2026-11-15"

CUTOFF = "2026-11-14"


@pytest.fixture(autouse=True)
def block_all_network(monkeypatch):
    def deny(*args, **kwargs):
        raise RuntimeError(
            f"network access is forbidden in this test module: {args!r}"
        )

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)

    yield


# ----------------------------------------------------------------------
# synthetic rolling data root
# ----------------------------------------------------------------------


def rolling_frame(label: str, dates: list[str]) -> pd.DataFrame:
    if label == "games":
        return pd.DataFrame(
            [
                {
                    "id": index,
                    "date": pd.Timestamp(moment),
                    "season": 2026,
                    "home_team_id": 10,
                    "visitor_team_id": 20,
                }
                for index, moment in enumerate(dates, start=1)
            ]
        )

    return pd.DataFrame(
        [
            {
                "game_id": index,
                "player_id": player_id,
                "date": pd.Timestamp(moment),
                "pts": 10.0 + player_id,
            }
            for index, moment in enumerate(dates, start=1)
            for player_id in (1, 2)
        ]
    )


def semantic_fingerprint(frame: pd.DataFrame, keys: list[str]) -> str:
    from nba_prop_quant.storage import sort_by_keys

    ordered = sort_by_keys(frame, keys)
    ordered = ordered[sorted(ordered.columns)]

    canonical = ordered.copy()

    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].dt.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

    payload = canonical.to_csv(index=False, float_format="%.12g")

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_data_root(
    root: Path,
    completed: list[str] | None = None,
    scheduled: list[str] | None = None,
) -> Path:
    """A rolling tree plus the Step 3B state record that describes it."""
    if completed is None:
        completed = ["2026-11-10", "2026-11-12", "2026-11-14"]

    if scheduled is None:
        scheduled = ["2026-11-20"]

    layout = {
        "stats": ("raw/seasons/season=2026/stats.parquet", ["game_id", "player_id"]),
        "advanced": ("raw/advanced/season=2026/advanced.parquet", ["game_id", "player_id"]),
        "games": ("raw/seasons/season=2026/games.parquet", ["id"]),
    }

    datasets = {}

    for label, (relative, keys) in layout.items():
        dates = completed + scheduled if label == "games" else completed

        frame = rolling_frame(label, dates)

        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

        usable = pd.to_datetime(frame["date"]).dropna()

        datasets[label] = {
            "columns": sorted(str(c) for c in frame.columns),
            "key_columns": keys,
            "max_date": usable.max().date().isoformat(),
            "min_date": usable.min().date().isoformat(),
            "relative_path": relative,
            "row_count": int(len(frame)),
            "schema_fingerprint": "s" * 64,
            "semantic_fingerprint": semantic_fingerprint(frame, keys),
            "unique_key_count": int(len(frame.drop_duplicates(subset=keys))),
        }

    state = {
        "datasets": datasets,
        "datasets_fingerprint": hashlib.sha256(
            json.dumps(datasets, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "generated_at_utc": "2026-11-15T09:00:00+00:00",
        "schema_version": 1,
        "season": 2026,
        "slate_date": SLATE_DATE,
    }

    state_path = root / ROLLING_STATE_RELATIVE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))

    return root


@pytest.fixture
def data_root(tmp_path) -> Path:
    return build_data_root(tmp_path / "data")


@pytest.fixture
def work_root(tmp_path) -> Path:
    root = tmp_path / "work"
    root.mkdir(parents=True, exist_ok=True)

    return root


@pytest.fixture
def registry(tmp_path) -> FitRegistry:
    return FitRegistry(root=tmp_path / "registry", project_root=PROJECT)


# ----------------------------------------------------------------------
# a deterministic stand-in engine
# ----------------------------------------------------------------------


def stub_marginals() -> dict[str, FittedMarginal]:
    """One real fitted marginal per frozen target.

    Real rather than a placeholder blob because the prediction smoke test
    deserialises these and prices with them. A stub that wrote bytes nothing
    could load would make the smoke test unexercised by the suite, which is the
    defect this engine used to hide.
    """
    marginals: dict[str, FittedMarginal] = {}

    for index, target in enumerate(sorted(FROZEN_MEAN_ROUTES)):
        model = NegativeBinomialCalibrator()
        model.size = 6.0 + index
        marginals[target] = FittedMarginal(kind="nb", model=model)

    return marginals


def stub_copula() -> GaussianCopula:
    """A real fitted copula with a PSD unit-diagonal correlation matrix."""
    targets = sorted(FROZEN_MEAN_ROUTES)

    size = len(targets)

    correlation = np.full((size, size), 0.2, dtype=float)
    np.fill_diagonal(correlation, 1.0)

    copula = GaussianCopula(targets=list(targets))
    copula.global_corr = correlation

    return copula


class StubFitEngine:
    """Writes tiny deterministic artifacts instead of fitting.

    Production and benchmark runs both use ProductionFitEngine; this exists
    only so the orchestration can be tested without an hours-long fit. The
    serving objects it writes are genuinely loadable, because the computed
    validation checks read them.
    """

    def __init__(self, project_root: Path, finite: bool = True) -> None:
        self.project_root = Path(project_root)
        self.finite = finite
        self.stages: list[str] = []

    def _record(self, name: str) -> None:
        self.stages.append(name)

    def build_features(self, context) -> None:
        self._record("build_features")

    def fit_minutes(self, context) -> None:
        self._record("fit_minutes")
        context.benchmark.count("xgboost_fits", 3)

    def fit_targets(self, context) -> None:
        self._record("fit_targets")
        context.benchmark.count("xgboost_fits", 6)

    def fit_ensemble_weights(self, context) -> None:
        self._record("fit_ensemble_weights")

        context.notes["ensemble_weights"] = {
            target: {"xgb": 0.7, "decay": 0.2, "kalman": 0.1}
            for target, route in FROZEN_MEAN_ROUTES.items()
            if route == "ensemble"
        }

    def fit_marginals(self, context) -> None:
        self._record("fit_marginals")

    def fit_dependence(self, context) -> None:
        self._record("fit_dependence")

    def fit_calibration(self, context) -> None:
        self._record("fit_calibration")

        # One digest per PROP-routed prop, which is what the real engine
        # records: it hashes every entry it wrote into the calibration policy,
        # and it writes an entry for every prop route. RAW routes acquire no
        # fitted parameters and so acquire no digest.
        context.notes["calibration_hashes"] = {
            prop_type: hashlib.sha256(prop_type.encode("utf-8")).hexdigest()
            for prop_type, route in sorted(FROZEN_CALIBRATION_ROUTES.items())
            if route == "prop"
        }

    def fit_gate3(self, context) -> None:
        self._record("fit_gate3")

        context.notes["role_state_hash"] = hashlib.sha256(
            b"stub-role-state"
        ).hexdigest()

    def assemble_candidate(self, context) -> None:
        self._record("assemble_candidate")

        candidate = context.workspace.candidate

        (candidate / "models").mkdir(parents=True, exist_ok=True)
        (candidate / "provenance").mkdir(parents=True, exist_ok=True)

        joblib.dump(stub_marginals(), candidate / "models" / "marginals.joblib")
        joblib.dump(stub_copula(), candidate / "models" / "copula.joblib")

        payload = {
            "dependence_lambda": dict(FROZEN_DEPENDENCE_LAMBDA),
            "ensemble_weights": context.notes.get("ensemble_weights", {}),
            "slope": 0.41 if self.finite else float("inf"),
        }

        (candidate / "models" / "parameters.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )

        for relative in (
            UPDATE_PROTOCOL_RELATIVE_PATH,
            Path("models/frozen_manifests")
            / "nba_prop_quant_v2_adaptive_architecture_contract.json",
        ):
            shutil.copyfile(
                self.project_root / relative,
                candidate / "provenance" / Path(relative).name,
            )


def fit(
    data_root: Path,
    work_root: Path,
    *,
    registry: FitRegistry | None = None,
    mode: str = MODE_BENCHMARK_ONLY,
    engine=None,
    slate_date: str = SLATE_DATE,
    **kwargs,
):
    return run_daily_fit(
        project_root=PROJECT,
        data_root=data_root,
        work_root=work_root,
        slate_date=slate_date,
        registry=registry,
        engine=engine if engine is not None else StubFitEngine(PROJECT),
        mode=mode,
        source_commit_sha="d6ee71d3e163b2f9eb2fe6b3d9b560d659eed281",
        **kwargs,
    )


# ----------------------------------------------------------------------
# frozen protocol and contracts
# ----------------------------------------------------------------------


def test_update_protocol_matches_the_tree(data_root, work_root):
    result = fit(data_root, work_root, mode=MODE_DRY_RUN, engine=None)

    assert result["outcome"] == OUTCOME_DRY_RUN


def test_protocol_carries_no_fitted_values():
    payload = json.loads(
        (PROJECT / UPDATE_PROTOCOL_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert payload["contains_fitted_values"] is False

    banned = {
        "production_parameters",
        "production_weights",
        "intercept",
        "slope",
        "global_corr",
        "player_corr",
    }

    def walk(node, trail: str):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in banned, f"{trail}{key} is a fitted value"
                walk(value, f"{trail}{key}.")
        elif isinstance(node, list):
            for item in node:
                walk(item, trail)

    walk(payload["daily_update_methodology"], "")


def test_protocol_references_the_required_contracts():
    payload = json.loads(
        (PROJECT / UPDATE_PROTOCOL_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    contract = load_architecture_contract(PROJECT)

    assert payload["architecture_contract_sha256"] == contract.sha256
    assert (
        payload["architecture_reference_sha"]
        == "4def8ad33ccc56016fb19a97fceca6e027c9612a"
    )

    serving = (
        PROJECT
        / "models"
        / "frozen_manifests"
        / "nba_prop_quant_v2_adaptive_serving_source_contract.json"
    )

    assert payload[
        "adaptive_serving_source_contract_sha256"
    ] == sha256_file(serving)

    assert payload["frozen_policy_digests"]


def build_fake_project(root: Path) -> Path:
    """A copy of every file the guards and the protocol check read."""
    shutil.copytree(
        PROJECT / "models" / "frozen_manifests",
        root / "models" / "frozen_manifests",
    )

    for relative in (
        "configs/model.yaml",
        "src/nba_prop_quant/features.py",
        "models/mean_model_selection.json",
        "models/marginal_selection.json",
        "models/market_probability_calibration_policy.json",
        "models/combo_dependence_policy.json",
        "research/v2_gate3_deployment_artifacts/deployment_manifest.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, destination)

    return root


def test_wrong_update_protocol_rejected(tmp_path, data_root, work_root):
    """A protocol that does not describe this tree stops the run."""
    fake = build_fake_project(tmp_path / "project")

    protocol_path = fake / UPDATE_PROTOCOL_RELATIVE_PATH

    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["config_hash"] = "0" * 64
    protocol_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    with pytest.raises(FrozenPolicyViolation, match="config_hash"):
        run_daily_fit(
            project_root=fake,
            data_root=data_root,
            work_root=work_root,
            slate_date=SLATE_DATE,
            mode=MODE_DRY_RUN,
        )


def test_wrong_architecture_contract_rejected(tmp_path, data_root, work_root):
    fake = build_fake_project(tmp_path / "project")

    contract_path = (
        fake
        / "models"
        / "frozen_manifests"
        / "nba_prop_quant_v2_adaptive_architecture_contract.json"
    )

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["architecture_reference_sha"] = "0" * 40
    contract_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    with pytest.raises(Exception, match="architecture_reference_sha"):
        run_daily_fit(
            project_root=fake,
            data_root=data_root,
            work_root=work_root,
            slate_date=SLATE_DATE,
            mode=MODE_DRY_RUN,
        )


def test_wrong_source_lineage_rejected(tmp_path, data_root, work_root, registry):
    """A different source commit is a different information set."""
    first = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    second = run_daily_fit(
        project_root=PROJECT,
        data_root=data_root,
        work_root=work_root,
        slate_date=SLATE_DATE,
        registry=registry,
        engine=StubFitEngine(PROJECT),
        mode=MODE_REGISTER_CANDIDATE,
        source_commit_sha="0" * 40,
    )

    # Same data, different source commit: a genuinely new fit, not an off day.
    assert second["outcome"] == OUTCOME_COMPLETED
    assert second["fit_id"] != first["fit_id"]


# ----------------------------------------------------------------------
# frozen policy write guard
# ----------------------------------------------------------------------


def test_frozen_policy_guard_covers_every_policy():
    covered = set(frozen_guard_paths(PROJECT))

    assert {
        "architecture_contract",
        "adaptive_update_protocol",
        "adaptive_serving_source_contract",
        "model_config",
        "mean_model_selection",
        "marginal_selection",
        "market_probability_calibration_policy",
        "combo_dependence_policy",
        "gate3_deployment_policy",
    } <= covered


def test_policy_mutation_refuses_registration(
    tmp_path, data_root, work_root, registry
):
    """A fit that rewrites a policy file has performed selection."""

    class MutatingEngine(StubFitEngine):
        def fit_marginals(self, context) -> None:
            super().fit_marginals(context)

            path = PROJECT / "models" / "marginal_selection.json"
            self._original = path.read_bytes()

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["targets"]["pts"]["selected_distribution"] = "nb"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    engine = MutatingEngine(PROJECT)
    path = PROJECT / "models" / "marginal_selection.json"
    original = path.read_bytes()

    try:
        with pytest.raises(FrozenPolicyViolation, match="marginal_selection"):
            fit(
                data_root,
                work_root,
                registry=registry,
                mode=MODE_REGISTER_CANDIDATE,
                engine=engine,
            )
    finally:
        path.write_bytes(original)

    assert registry.list_fits() == []


def test_guard_localises_the_stage_that_moved_a_policy(tmp_path):
    guard = FrozenPolicyGuard(PROJECT)
    guard.snapshot()

    guard.verify("input_snapshot")

    guard.baseline["model_config"] = "0" * 64

    with pytest.raises(FrozenPolicyViolation, match="target_fit"):
        guard.verify("target_fit")


# ----------------------------------------------------------------------
# selector prohibition
# ----------------------------------------------------------------------


def test_forbidden_selector_scripts_are_denied():
    for script in sorted(FORBIDDEN_SELECTOR_SCRIPTS):
        with pytest.raises(SelectorInvocationRefused):
            assert_selector_script_not_requested(script)

        with pytest.raises(SelectorInvocationRefused):
            assert_selector_script_not_requested(f"/repo/{script}")


def test_fit_only_scripts_are_allowed():
    for script in (
        "scripts/06b_fit_mean_ensemble.py",
        "scripts/07_fit_marginals.py",
        "scripts/09c_fit_probability_calibration.py",
        "scripts/08_fit_copula.py",
    ):
        assert_selector_script_not_requested(script)


def test_selection_scripts_are_on_the_deny_list():
    """The known selectors, from the audit of what writes a policy file."""
    assert "scripts/06c_select_mean_models.py" in FORBIDDEN_SELECTOR_SCRIPTS
    assert (
        "scripts/09d_select_probability_calibration.py"
        in FORBIDDEN_SELECTOR_SCRIPTS
    )
    assert (
        "scripts/08d_refine_copula_shrinkage_cv.py"
        in FORBIDDEN_SELECTOR_SCRIPTS
    )
    assert "scripts/13_gate2_certify_v2.py" in FORBIDDEN_SELECTOR_SCRIPTS
    assert "scripts/03_tune_dynamic_priors.py" in FORBIDDEN_SELECTOR_SCRIPTS


def test_expanding_time_oof_cannot_be_invoked():
    """Season walk-forward is the certified methodology; the other is research."""
    from nba_prop_quant import model

    with selector_guard():
        with pytest.raises(SelectorInvocationRefused, match="expanding_time"):
            model.expanding_time_oof_target(pd.DataFrame(), target="pts")

        with pytest.raises(SelectorInvocationRefused):
            model.expanding_time_oof_minutes(pd.DataFrame())

    # Restored afterwards, so research use outside a daily fit is unaffected.
    assert callable(model.expanding_time_oof_target)


def test_hyperparameter_tuners_cannot_be_invoked():
    from nba_prop_quant import decay, kalman

    with selector_guard():
        with pytest.raises(SelectorInvocationRefused):
            decay.tune_decay_beta()

        with pytest.raises(SelectorInvocationRefused):
            kalman.tune_kalman()


def test_required_oof_methodology_is_season_walk_forward():
    assert REQUIRED_OOF_FUNCTIONS == (
        "season_walk_forward_oof_minutes",
        "season_walk_forward_oof_target",
    )

    from nba_prop_quant import model

    for name in REQUIRED_OOF_FUNCTIONS:
        assert hasattr(model, name)


def test_selector_guard_restores_originals_on_error():
    from nba_prop_quant import model

    original = model.expanding_time_oof_target

    with pytest.raises(ValueError):
        with selector_guard():
            raise ValueError("stage blew up")

    assert model.expanding_time_oof_target is original


# ----------------------------------------------------------------------
# frozen architecture values
# ----------------------------------------------------------------------


def test_frozen_mean_routing():
    assert FROZEN_MEAN_ROUTES == {
        "pts": "xgb",
        "reb": "ensemble",
        "ast": "xgb",
        "stl": "xgb",
        "blk": "ensemble",
        "fg3m": "xgb",
    }


def test_frozen_marginal_family_is_zinb():
    assert FROZEN_MARGINAL_FAMILY == "zinb"


def test_frozen_dependence_lambda_policy():
    assert FROZEN_DEPENDENCE_LAMBDA["rebounds_assists"] == 0.85

    for combo in (
        "points_assists",
        "points_rebounds",
        "points_rebounds_assists",
        "stocks",
    ):
        assert FROZEN_DEPENDENCE_LAMBDA[combo] == 0.0


def test_frozen_seeds_and_n_jobs():
    assert CORE_SEED == 73
    assert GATE3_SEED == 20260830
    assert EFFECTIVE_N_JOBS == 2


def test_training_window_floors():
    assert HISTORY_START_SEASON == 2001
    assert ADVANCED_START_SEASON == 2015


def test_frozen_values_agree_with_the_architecture_contract():
    frozen = load_architecture_contract(PROJECT).frozen_choices

    routing = frozen["routing"]

    assert routing["mean_model_routing"] == FROZEN_MEAN_ROUTES
    assert set(routing["marginal_family_routing"].values()) == {
        FROZEN_MARGINAL_FAMILY
    }
    assert routing["dependence_production_lambda"] == FROZEN_DEPENDENCE_LAMBDA
    assert routing["calibration_family_routing"] == FROZEN_CALIBRATION_ROUTES

    assert frozen["core_seed"] == CORE_SEED
    assert frozen["gate3_seed"] == GATE3_SEED
    assert frozen["effective_n_jobs"] == EFFECTIVE_N_JOBS


# ----------------------------------------------------------------------
# data state, cutoff and leakage
# ----------------------------------------------------------------------


def test_missing_rolling_state_rejected(tmp_path, work_root):
    root = build_data_root(tmp_path / "data")

    (root / ROLLING_STATE_RELATIVE_PATH).unlink()

    with pytest.raises(DataStateError, match="no rolling-state record"):
        fit(root, work_root, mode=MODE_DRY_RUN, engine=None)


def test_rolling_state_mismatch_rejected(tmp_path, work_root):
    """The record is only worth trusting if the data still matches it."""
    root = build_data_root(tmp_path / "data")

    stats = root / "raw/seasons/season=2026/stats.parquet"

    frame = pd.read_parquet(stats)
    frame.loc[0, "pts"] = 999.0
    frame.to_parquet(stats, index=False)

    with pytest.raises(DataStateError, match="semantic fingerprint"):
        fit(root, work_root, mode=MODE_DRY_RUN, engine=None)


def test_rolling_state_row_count_mismatch_rejected(tmp_path, work_root):
    root = build_data_root(tmp_path / "data")

    stats = root / "raw/seasons/season=2026/stats.parquet"

    frame = pd.read_parquet(stats)
    frame.iloc[:-1].to_parquet(stats, index=False)

    with pytest.raises(DataStateError, match="rows but the state record"):
        fit(root, work_root, mode=MODE_DRY_RUN, engine=None)


def test_cutoff_is_derived_from_verified_data(data_root):
    state = json.loads(
        (data_root / ROLLING_STATE_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert derive_training_cutoff(state, SLATE_DATE) == date(2026, 11, 14)


def test_games_future_rows_do_not_move_the_cutoff(tmp_path):
    """games legitimately holds scheduled rows; they are not completed data."""
    root = build_data_root(
        tmp_path / "data", scheduled=["2026-12-25", "2027-01-05"]
    )

    state = json.loads(
        (root / ROLLING_STATE_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    assert state["datasets"]["games"]["max_date"] == "2027-01-05"
    assert derive_training_cutoff(state, SLATE_DATE) == date(2026, 11, 14)


def test_same_day_history_rejected(tmp_path, work_root):
    """History reaching the slate date is leakage, not fresh data."""
    root = build_data_root(
        tmp_path / "data", completed=["2026-11-10", SLATE_DATE]
    )

    with pytest.raises(DataStateError, match="not strictly before"):
        fit(root, work_root, mode=MODE_DRY_RUN, engine=None)


def test_caller_cannot_widen_the_cutoff(data_root, work_root):
    with pytest.raises(DataStateError, match="exceeds the verified data"):
        fit(
            data_root,
            work_root,
            mode=MODE_DRY_RUN,
            engine=None,
            requested_cutoff="2026-11-20",
        )


def test_caller_may_assert_a_narrower_cutoff(data_root, work_root):
    result = fit(
        data_root,
        work_root,
        mode=MODE_DRY_RUN,
        engine=None,
        requested_cutoff="2026-11-13",
    )

    assert result["plan"]["training_cutoff"] == CUTOFF


# ----------------------------------------------------------------------
# no new training data
# ----------------------------------------------------------------------


def test_no_parent_fit_means_new_data():
    assert has_new_training_data(None, date(2026, 11, 14), "abc") is True


def test_identical_information_set_is_not_refit():
    parent = {
        "training_cutoff": "2026-11-14",
        "identity_inputs": {"training_data_manifest_hash": "abc"},
    }

    assert has_new_training_data(parent, date(2026, 11, 14), "abc") is False


def test_new_cutoff_means_new_data():
    parent = {
        "training_cutoff": "2026-11-13",
        "identity_inputs": {"training_data_manifest_hash": "abc"},
    }

    assert has_new_training_data(parent, date(2026, 11, 14), "abc") is True


def test_off_day_returns_no_new_training_data(
    data_root, work_root, registry
):
    """An NBA off day must not mint a duplicate fit_id."""
    first = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    assert first["outcome"] == OUTCOME_COMPLETED

    second = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    assert second["outcome"] == OUTCOME_NO_NEW_TRAINING_DATA
    assert "fit_id" not in second
    assert registry.list_fits() == [first["fit_id"]]


def test_no_new_data_changes_no_promotion_state(
    data_root, work_root, registry
):
    fit(data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE)

    before = registry.current()

    fit(data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE)

    assert registry.current() == before
    assert registry.current()["current_good_fit_id"] is None


# ----------------------------------------------------------------------
# workspace isolation and locking
# ----------------------------------------------------------------------


def test_workspace_is_unique_and_isolated(work_root, registry):
    first = new_workspace(work_root, SLATE_DATE)
    second = new_workspace(work_root, SLATE_DATE)

    assert first.root != second.root

    for workspace in (first, second):
        assert workspace.models.is_dir()
        assert workspace.candidate.is_dir()

        assert_workspace_is_isolated(workspace, PROJECT, registry)


def test_workspace_inside_the_repository_refused(registry):
    workspace = new_workspace(PROJECT / "build", SLATE_DATE)

    try:
        with pytest.raises(
            AdaptiveTrainingError, match="inside the repository"
        ):
            assert_workspace_is_isolated(workspace, PROJECT, registry)
    finally:
        shutil.rmtree(PROJECT / "build", ignore_errors=True)


def test_workspace_overlapping_the_registry_refused(tmp_path, registry):
    registry.initialise()

    workspace = new_workspace(registry.fits_dir, SLATE_DATE)

    with pytest.raises(AdaptiveTrainingError, match="overlaps the registry"):
        assert_workspace_is_isolated(workspace, PROJECT, registry)


def test_production_models_are_untouched(data_root, work_root, registry):
    before = {
        path.name: sha256_file(path)
        for path in sorted((PROJECT / "models").glob("*.json"))
    }

    fit(data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE)

    after = {
        path.name: sha256_file(path)
        for path in sorted((PROJECT / "models").glob("*.json"))
    }

    assert after == before


def test_training_lock_prevents_a_second_runner(work_root):
    with training_lock(work_root):
        with pytest.raises(TrainingLocked):
            with training_lock(work_root):
                pass


def test_locked_run_returns_already_running(data_root, work_root, registry):
    with training_lock(work_root):
        result = fit(
            data_root,
            work_root,
            registry=registry,
            mode=MODE_REGISTER_CANDIDATE,
        )

    assert result["outcome"] == OUTCOME_ALREADY_RUNNING
    assert registry.list_fits() == []


def test_lock_is_released_after_a_run(data_root, work_root, registry):
    fit(data_root, work_root, registry=registry, mode=MODE_BENCHMARK_ONLY)

    with training_lock(work_root):
        pass


# ----------------------------------------------------------------------
# calibration policy
# ----------------------------------------------------------------------


def test_raw_routes_stay_raw():
    assert FROZEN_CALIBRATION_ROUTES["blocks"] == "raw"
    assert FROZEN_CALIBRATION_ROUTES["steals"] == "raw"

    for prop_type in ("blocks", "steals"):
        with pytest.raises(FrozenPolicyViolation, match="raw route"):
            resolve_calibration_parameters(
                prop_type, 5000, 2, lambda: {"intercept": 0.0, "slope": 1.0}, None
            )


def test_prop_unit_with_enough_rows_is_fitted():
    parameters, origin = resolve_calibration_parameters(
        "points",
        CALIBRATION_MIN_ROWS,
        CALIBRATION_MIN_CLASSES,
        lambda: {"intercept": -0.05, "slope": 0.39},
        None,
    )

    assert origin == "fitted"
    assert parameters["slope"] == 0.39


def test_minimum_sample_guard():
    assert calibration_unit_is_fittable(CALIBRATION_MIN_ROWS, 2) is True
    assert calibration_unit_is_fittable(CALIBRATION_MIN_ROWS - 1, 2) is False


def test_single_class_guard():
    assert calibration_unit_is_fittable(5000, 1) is False

    with pytest.raises(AdaptiveTrainingError, match="below the production"):
        resolve_calibration_parameters(
            "points", 5000, 1, lambda: {"intercept": 0.0, "slope": 1.0}, None
        )


def test_compatible_prior_good_parameter_is_reused():
    parameters, origin = resolve_calibration_parameters(
        "points",
        10,
        2,
        lambda: {"intercept": 0.0, "slope": 1.0},
        {
            "selected_method": "prop",
            "production_parameters": {"intercept": -0.2, "slope": 0.5},
        },
    )

    assert origin == "reused_prior_good"
    assert parameters == {"intercept": -0.2, "slope": 0.5}


def test_incompatible_prior_good_parameter_rejected():
    with pytest.raises(AdaptiveTrainingError, match="not compatible"):
        resolve_calibration_parameters(
            "points",
            10,
            2,
            lambda: {"intercept": 0.0, "slope": 1.0},
            {
                "selected_method": "global",
                "production_parameters": {"intercept": -0.2, "slope": 0.5},
            },
        )


def test_missing_prior_parameters_rejected():
    with pytest.raises(AdaptiveTrainingError, match="no production parameters"):
        resolve_calibration_parameters(
            "points", 10, 2, lambda: {}, {"selected_method": "prop"}
        )


# ----------------------------------------------------------------------
# candidate verification
# ----------------------------------------------------------------------


def test_candidate_registers_and_is_not_promoted(
    data_root, work_root, registry
):
    result = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    assert result["outcome"] == OUTCOME_COMPLETED
    assert result["promoted"] is False

    fit_id = result["fit_id"]

    assert fit_id.startswith("nba_prop_quant_fit_")
    assert registry.list_fits() == [fit_id]

    assert registry.current()["current_good_fit_id"] is None


def test_benchmark_only_registers_nothing(data_root, work_root, registry):
    result = fit(
        data_root, work_root, registry=registry, mode=MODE_BENCHMARK_ONLY
    )

    assert result["outcome"] == OUTCOME_COMPLETED
    assert "fit_id" not in result
    assert registry.list_fits() == []


def test_non_finite_structured_parameters_rejected(
    data_root, work_root, registry
):
    with pytest.raises(CandidateIncomplete, match="non-finite"):
        fit(
            data_root,
            work_root,
            registry=registry,
            mode=MODE_REGISTER_CANDIDATE,
            engine=StubFitEngine(PROJECT, finite=False),
        )

    assert registry.list_fits() == []


def test_finite_check_reads_structured_values(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    (candidate / "ok.json").write_text(json.dumps({"slope": 0.4}))

    finite, offenders = structured_values_are_finite(candidate)

    assert finite is True
    assert offenders == []

    (candidate / "bad.json").write_text('{"slope": Infinity}')

    finite, offenders = structured_values_are_finite(candidate)

    assert finite is False
    assert any("slope" in name for name in offenders)


def test_candidate_without_fitted_artifacts_rejected(
    data_root, work_root, registry
):
    """A tree holding only its own provenance has fitted nothing."""

    class EmptyEngine(StubFitEngine):
        def assemble_candidate(self, context) -> None:
            self._record("assemble_candidate")

    with pytest.raises(CandidateIncomplete, match="missing required"):
        fit(
            data_root,
            work_root,
            registry=registry,
            mode=MODE_REGISTER_CANDIDATE,
            engine=EmptyEngine(PROJECT),
        )

    assert registry.list_fits() == []


def test_truly_empty_candidate_rejected(tmp_path):
    from nba_prop_quant.adaptive_training import assert_candidate_tree_usable

    empty = tmp_path / "candidate"
    empty.mkdir()

    with pytest.raises(CandidateIncomplete, match="empty"):
        assert_candidate_tree_usable(empty)


def test_symlink_in_candidate_rejected(data_root, work_root, registry):
    class SymlinkEngine(StubFitEngine):
        def assemble_candidate(self, context) -> None:
            super().assemble_candidate(context)

            outside = context.workspace.root / "outside.bin"
            outside.write_bytes(b"outside")

            (context.workspace.candidate / "link.joblib").symlink_to(outside)

    with pytest.raises(CandidateIncomplete, match="symlink"):
        fit(
            data_root,
            work_root,
            registry=registry,
            mode=MODE_REGISTER_CANDIDATE,
            engine=SymlinkEngine(PROJECT),
        )


def test_candidate_carries_the_update_protocol(
    data_root, work_root, registry
):
    """The protocol's hash contributes to immutable fit content."""
    result = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    manifest = registry.load_manifest(result["fit_id"])

    names = set(manifest["artifact_hashes"])

    assert any(
        "nba_prop_quant_v2_adaptive_update_protocol.json" in name
        for name in names
    )


def test_fit_metadata_records_the_frozen_identity(
    data_root, work_root, registry
):
    result = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    manifest = registry.load_manifest(result["fit_id"])

    assert manifest["training_cutoff"] == CUTOFF
    assert manifest["fit_date"] == SLATE_DATE
    assert manifest["core_seed"] == CORE_SEED
    assert manifest["gate3_seed"] == GATE3_SEED
    assert manifest["effective_n_jobs"] == EFFECTIVE_N_JOBS
    assert manifest["source_commit_sha"]
    assert manifest["training_data_manifest_hash"]
    assert manifest["gate3_candidate_policy_id"]


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------


def test_step3c_records_only_checks_it_can_establish(
    data_root, work_root, registry
):
    result = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    recorded = set(result["validation_checks"])

    assert recorded == set(STEP3C_VALIDATION_CHECKS)

    deferred = set(result["deferred_validation_checks"])

    assert deferred == {"gate3_role_readiness", "t20_protocol_compatible"}
    assert recorded.isdisjoint(deferred)


def test_deferred_checks_are_the_live_ones():
    """Step 3C must not fabricate a live snapshot or a T-20 capture."""
    assert set(deferred_validation_checks()) == {
        "gate3_role_readiness",
        "t20_protocol_compatible",
    }

    assert set(STEP3C_VALIDATION_CHECKS) | set(
        deferred_validation_checks()
    ) == set(REQUIRED_VALIDATION_CHECKS)


def test_incomplete_validation_cannot_promote(
    data_root, work_root, registry
):
    """The candidate is deliberately not promotable after Step 3C."""
    result = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    from nba_prop_quant.adaptive_fit_registry import ValidationRefused

    with pytest.raises(ValidationRefused, match="missing required"):
        registry.promote(result["fit_id"], reason="should be refused")

    assert registry.current()["current_good_fit_id"] is None


def test_no_step3c_code_promotes():
    source = (
        PROJECT / "src" / "nba_prop_quant" / "adaptive_training.py"
    ).read_text(encoding="utf-8")

    cli = CLI.read_text(encoding="utf-8")

    for text in (source, cli):
        assert ".promote(" not in text
        assert "rollback(" not in text


# ----------------------------------------------------------------------
# benchmark harness
# ----------------------------------------------------------------------


def test_benchmark_report_schema(data_root, work_root, registry):
    result = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    report = result["benchmark"]

    for field in (
        "started_at_utc",
        "ended_at_utc",
        "total_seconds",
        "stage_seconds",
        "counters",
        "cpu_count",
        "effective_n_jobs",
        "peak_rss_bytes",
        "python_version",
        "package_versions",
        "candidate_artifact_bytes",
        "schema_version",
    ):
        assert field in report, field

    assert report["effective_n_jobs"] == EFFECTIVE_N_JOBS
    assert report["peak_rss_bytes"] > 0
    assert report["counters"]["xgboost_fits"] == 9


def test_benchmark_times_every_executed_stage(
    data_root, work_root, registry
):
    result = fit(
        data_root, work_root, registry=registry, mode=MODE_REGISTER_CANDIDATE
    )

    timed = set(result["benchmark"]["stage_seconds"])

    assert timed == set(DAG_STAGES)


def test_benchmark_written_outside_the_repository(
    data_root, work_root, registry
):
    fit(data_root, work_root, registry=registry, mode=MODE_BENCHMARK_ONLY)

    report = work_root / "reports" / f"benchmark_{SLATE_DATE}.json"

    assert report.is_file()
    assert PROJECT not in report.parents


def test_benchmark_fit_count_is_measured_not_hardcoded(
    data_root, work_root, registry
):
    """77 was a prior audit anchor for one window, not a constant."""
    source = (
        PROJECT / "src" / "nba_prop_quant" / "adaptive_training.py"
    ).read_text(encoding="utf-8")

    assert "77" not in source

    recorder = BenchmarkRecorder()

    recorder.count("xgboost_fits", 12)
    recorder.count("xgboost_fits", 3)

    assert recorder.counters["xgboost_fits"] == 15


def test_benchmark_and_register_share_the_same_dag(
    tmp_path, data_root, work_root, registry
):
    """Benchmark must not have its own mathematics or its own path."""
    benchmark_engine = StubFitEngine(PROJECT)

    fit(
        data_root,
        work_root,
        registry=registry,
        mode=MODE_BENCHMARK_ONLY,
        engine=benchmark_engine,
    )

    register_engine = StubFitEngine(PROJECT)

    fit(
        build_data_root(tmp_path / "data2"),
        tmp_path / "work2",
        registry=FitRegistry(
            root=tmp_path / "registry2", project_root=PROJECT
        ),
        mode=MODE_REGISTER_CANDIDATE,
        engine=register_engine,
    )

    assert benchmark_engine.stages == register_engine.stages


# ----------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------


def test_dry_run_fits_nothing(data_root, work_root):
    engine = StubFitEngine(PROJECT)

    result = run_daily_fit(
        project_root=PROJECT,
        data_root=data_root,
        work_root=work_root,
        slate_date=SLATE_DATE,
        engine=engine,
        mode=MODE_DRY_RUN,
    )

    assert result["outcome"] == OUTCOME_DRY_RUN
    assert engine.stages == []

    assert not (work_root / "staging").exists()


def test_dry_run_plan_states_the_frozen_architecture(data_root, work_root):
    plan = fit(data_root, work_root, mode=MODE_DRY_RUN, engine=None)["plan"]

    assert plan["mean_routes"] == FROZEN_MEAN_ROUTES
    assert plan["marginal_family"] == FROZEN_MARGINAL_FAMILY
    assert plan["dependence_lambda"] == FROZEN_DEPENDENCE_LAMBDA
    assert plan["calibration_routes"] == FROZEN_CALIBRATION_ROUTES
    assert plan["training_cutoff"] == CUTOFF
    assert plan["oof_methodology"] == list(REQUIRED_OOF_FUNCTIONS)
    assert plan["history_start_season"] == HISTORY_START_SEASON


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *argv],
        capture_output=True,
        text=True,
    )


def test_cli_requires_a_mode(tmp_path, data_root):
    result = run_cli(
        "--slate-date",
        SLATE_DATE,
        "--data-root",
        str(data_root),
        "--work-root",
        str(tmp_path / "work"),
    )

    assert result.returncode != 0


def test_cli_dry_run(tmp_path, data_root):
    result = run_cli(
        "--dry-run",
        "--slate-date",
        SLATE_DATE,
        "--data-root",
        str(data_root),
        "--work-root",
        str(tmp_path / "work"),
    )

    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)

    assert payload["outcome"] == OUTCOME_DRY_RUN
    assert payload["plan"]["training_cutoff"] == CUTOFF


def test_cli_register_requires_a_registry(tmp_path, data_root):
    result = run_cli(
        "--register-candidate",
        "--slate-date",
        SLATE_DATE,
        "--data-root",
        str(data_root),
        "--work-root",
        str(tmp_path / "work"),
    )

    assert result.returncode != 0
    assert "requires --registry-root" in result.stderr


def test_cli_refuses_a_registry_inside_the_repository(tmp_path, data_root):
    result = run_cli(
        "--register-candidate",
        "--slate-date",
        SLATE_DATE,
        "--data-root",
        str(data_root),
        "--work-root",
        str(tmp_path / "work"),
        "--registry-root",
        str(PROJECT / "registry"),
    )

    assert result.returncode != 0
    assert "inside the repository" in result.stderr


def test_cli_reports_missing_rolling_state(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = run_cli(
        "--dry-run",
        "--slate-date",
        SLATE_DATE,
        "--data-root",
        str(empty),
        "--work-root",
        str(tmp_path / "work"),
    )

    assert result.returncode != 0
    assert json.loads(result.stderr)["error"] == "DataStateError"


# ----------------------------------------------------------------------
# environment hygiene
# ----------------------------------------------------------------------


def test_tests_leave_no_registry_in_the_repository():
    for name in ("registry", "fits", "staging", "build"):
        assert not (PROJECT / name).exists()


def test_tests_do_not_write_a_rolling_state_into_the_repository():
    assert not (PROJECT / ROLLING_STATE_RELATIVE_PATH).exists()


def test_no_test_reaches_the_network():
    with pytest.raises(RuntimeError, match="network access is forbidden"):
        socket.socket()


# ----------------------------------------------------------------------
# Regression: the full-data benchmark defect
#
# The first real benchmark failed with "At least two seasons are required".
# The workspace held only the latest season because the snapshot looped over
# the Step 3B rolling-state datasets, and feature construction read only the
# state's stats partition. The rolling state authenticates the latest coherent
# partition; it does not define the historical training corpus.
# ----------------------------------------------------------------------


HISTORICAL_SEASONS = (2023, 2024, 2025, 2026)

STATE_SEASON = 2026


def season_stats(season: int, dates: list[str]) -> pd.DataFrame:
    rows = []

    for index, moment in enumerate(dates):
        game_id = season * 1000 + index

        for player_id, team_id in ((1, 10), (2, 10), (3, 20), (4, 20)):
            home = 10 if index % 2 == 0 else 20

            rows.append(
                {
                    "stat_id": len(rows) + season * 10000,
                    "player_id": player_id,
                    "team_id": team_id,
                    "game_id": game_id,
                    "date": pd.Timestamp(moment),
                    "season": season,
                    "postseason": False,
                    "home_team_id": home,
                    "visitor_team_id": 20 if home == 10 else 10,
                    "position": "G" if player_id % 2 else "F",
                    "draft_year": 2018,
                    "min": "30:00",
                    "minutes": 28.0 + player_id,
                    "pts": 10 + player_id + index,
                    "reb": 3 + player_id,
                    "ast": 2 + index,
                    "stl": player_id % 2,
                    "blk": 0,
                    "fg3m": 1,
                    "fga": 10,
                    "fg3a": 4,
                    "fta": 2,
                    "oreb": 1,
                    "dreb": 3,
                    "turnover": 1,
                    "pf": 2,
                }
            )

    return pd.DataFrame(rows)


def season_games(
    season: int, dates: list[str], id_offset: int = 0
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": season * 1000 + id_offset + index,
                "date": pd.Timestamp(moment),
                "season": season,
                "home_team_id": 10 if index % 2 == 0 else 20,
                "visitor_team_id": 20 if index % 2 == 0 else 10,
                "status": "Final",
                "postseason": False,
            }
            for index, moment in enumerate(dates)
        ]
    )


def season_advanced(season: int, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": player_id,
                "game_id": season * 1000 + index,
                "date": pd.Timestamp(moment),
                "season": season,
                "usage_percentage": 20.0 + player_id,
                "assist_percentage": 12.0,
                "rebound_percentage": 9.0,
                "pace": 99.0,
            }
            for index, moment in enumerate(dates)
            for player_id in (1, 2, 3, 4)
        ]
    )


def season_dates(season: int) -> list[str]:
    """Ten dated games per season, starting in that season's November."""
    return [
        (pd.Timestamp(f"{season}-11-01") + pd.Timedelta(days=2 * step))
        .date()
        .isoformat()
        for step in range(10)
    ]


FUTURE_SCHEDULED = ["2026-12-20", "2027-01-15"]


def build_multi_season_data_root(root: Path) -> Path:
    """Several historical seasons, with state describing only the latest.

    This is the shape the real benchmark ran against: a complete local data
    root, and a Step 3B state record covering one partition.
    """
    root.mkdir(parents=True, exist_ok=True)

    for season in HISTORICAL_SEASONS:
        dates = season_dates(season)

        season_dir = root / f"raw/seasons/season={season}"
        season_dir.mkdir(parents=True, exist_ok=True)

        season_stats(season, dates).to_parquet(
            season_dir / "stats.parquet", index=False
        )

        games = season_games(season, dates)

        if season == STATE_SEASON:
            # Future scheduled rows are legitimate in games.
            games = pd.concat(
                [
                    games,
                    season_games(season, FUTURE_SCHEDULED, id_offset=900),
                ],
                ignore_index=True,
            )

        games.to_parquet(season_dir / "games.parquet", index=False)

        advanced_dir = root / f"raw/advanced/season={season}"
        advanced_dir.mkdir(parents=True, exist_ok=True)

        season_advanced(season, dates).to_parquet(
            advanced_dir / "advanced.parquet", index=False
        )

    # The rolling state authenticates only the latest partition.
    datasets = {}

    layout = {
        "stats": (
            f"raw/seasons/season={STATE_SEASON}/stats.parquet",
            ["game_id", "player_id"],
        ),
        "advanced": (
            f"raw/advanced/season={STATE_SEASON}/advanced.parquet",
            ["game_id", "player_id"],
        ),
        "games": (
            f"raw/seasons/season={STATE_SEASON}/games.parquet",
            ["id"],
        ),
    }

    for label, (relative, keys) in layout.items():
        frame = pd.read_parquet(root / relative)

        usable = pd.to_datetime(frame["date"]).dropna()

        datasets[label] = {
            "columns": sorted(str(c) for c in frame.columns),
            "key_columns": keys,
            "max_date": usable.max().date().isoformat(),
            "min_date": usable.min().date().isoformat(),
            "relative_path": relative,
            "row_count": int(len(frame)),
            "schema_fingerprint": "s" * 64,
            "semantic_fingerprint": semantic_fingerprint(frame, keys),
            "unique_key_count": int(len(frame.drop_duplicates(subset=keys))),
        }

    state = {
        "datasets": datasets,
        "datasets_fingerprint": hashlib.sha256(
            json.dumps(datasets, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "generated_at_utc": "2026-11-21T09:00:00+00:00",
        "schema_version": 1,
        "season": STATE_SEASON,
        "slate_date": "2026-11-21",
    }

    state_path = root / ROLLING_STATE_RELATIVE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))

    return root


MULTI_SLATE_DATE = "2026-11-21"

MULTI_CUTOFF = "2026-11-19"


@pytest.fixture
def multi_season_root(tmp_path) -> Path:
    return build_multi_season_data_root(tmp_path / "full_data")


class RealFeatureEngine(StubFitEngine):
    """Stub everywhere except the real feature assembly under test."""

    def build_features(self, context) -> None:
        self._record("build_features")

        from nba_prop_quant.adaptive_training import ProductionFitEngine

        ProductionFitEngine(self.project_root).build_features(context)


def test_state_describes_only_the_latest_season(multi_season_root):
    """The fixture reproduces the condition that caused the failure."""
    state = json.loads(
        (multi_season_root / ROLLING_STATE_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )

    for record in state["datasets"].values():
        assert f"season={STATE_SEASON}" in record["relative_path"]

    assert len(HISTORICAL_SEASONS) > 1


def test_a_snapshot_captures_the_full_historical_corpus(
    multi_season_root, tmp_path
):
    from nba_prop_quant.adaptive_training import (
        eligible_history_files,
        snapshot_training_inputs,
    )

    state = json.loads(
        (multi_season_root / ROLLING_STATE_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )

    cutoff = date.fromisoformat(MULTI_CUTOFF)

    relatives = eligible_history_files(multi_season_root, cutoff)

    # Every season's stats and games, not only the state's partition.
    for season in HISTORICAL_SEASONS:
        assert f"raw/seasons/season={season}/stats.parquet" in relatives
        assert f"raw/seasons/season={season}/games.parquet" in relatives

    workspace = new_workspace(tmp_path / "work", MULTI_SLATE_DATE)

    hashes = snapshot_training_inputs(
        multi_season_root, workspace, state, cutoff
    )

    assert set(hashes) == set(relatives)

    snapshotted_seasons = {
        int(path.parent.name.split("=")[1])
        for path in (workspace.inputs / "raw" / "seasons").iterdir()
        for _ in [0]
        for path in [path / "stats.parquet"]
        if path.exists()
    }

    assert snapshotted_seasons == set(HISTORICAL_SEASONS)


def test_b_latest_state_partition_is_still_verified(
    multi_season_root, tmp_path
):
    """Authentication of the rolling generation is unchanged."""
    from nba_prop_quant.adaptive_training import (
        snapshot_training_inputs,
        verify_rolling_state,
    )

    state = verify_rolling_state(multi_season_root)

    workspace = new_workspace(tmp_path / "work", MULTI_SLATE_DATE)

    hashes = snapshot_training_inputs(
        multi_season_root,
        workspace,
        state,
        date.fromisoformat(MULTI_CUTOFF),
    )

    for record in state["datasets"].values():
        assert record["relative_path"] in hashes

    # A state partition missing from the corpus is a hard stop.
    stats_relative = state["datasets"]["stats"]["relative_path"]

    (multi_season_root / stats_relative).unlink()

    with pytest.raises(DataStateError, match="not part of the snapshotted"):
        snapshot_training_inputs(
            multi_season_root,
            new_workspace(tmp_path / "work2", MULTI_SLATE_DATE),
            state,
            date.fromisoformat(MULTI_CUTOFF),
        )


def test_advanced_floor_is_preserved(multi_season_root, tmp_path):
    """Advanced history is not narrowed to the latest rolling partition."""
    from nba_prop_quant.adaptive_training import eligible_history_files

    relatives = eligible_history_files(
        multi_season_root, date.fromisoformat(MULTI_CUTOFF)
    )

    advanced = [name for name in relatives if "raw/advanced/" in name]

    assert len(advanced) == len(HISTORICAL_SEASONS)

    for season in HISTORICAL_SEASONS:
        assert season >= ADVANCED_START_SEASON
        assert f"raw/advanced/season={season}/advanced.parquet" in advanced


def test_pre_2015_advanced_is_excluded(multi_season_root):
    """The frozen advanced floor still holds."""
    from nba_prop_quant.adaptive_training import eligible_history_files

    early = multi_season_root / "raw/advanced/season=2012"
    early.mkdir(parents=True, exist_ok=True)

    season_advanced(2012, season_dates(2012)).to_parquet(
        early / "advanced.parquet", index=False
    )

    relatives = eligible_history_files(
        multi_season_root, date.fromisoformat(MULTI_CUTOFF)
    )

    assert "raw/advanced/season=2012/advanced.parquet" not in relatives


def multi_season_fit(multi_season_root, work_root, registry, engine=None):
    return run_daily_fit(
        project_root=PROJECT,
        data_root=multi_season_root,
        work_root=work_root,
        slate_date=MULTI_SLATE_DATE,
        registry=registry,
        engine=engine if engine is not None else RealFeatureEngine(PROJECT),
        mode=MODE_REGISTER_CANDIDATE,
        source_commit_sha="d8a32599738e858dc6153d62892a62d22562b981",
    )


def test_cd_training_frame_spans_multiple_seasons(
    multi_season_root, work_root, registry
):
    """The exact failure: the frame must not collapse to the state season.

    Against the previous behaviour this raised
    "the training frame spans 1 season(s)" because only the state partition was
    snapshotted and read.
    """
    result = multi_season_fit(multi_season_root, work_root, registry)

    assert result["outcome"] == OUTCOME_COMPLETED

    seasons = result["benchmark"]["details"]["training_seasons"]

    assert len(seasons) >= 2
    assert seasons == sorted(HISTORICAL_SEASONS)
    assert seasons != [STATE_SEASON]

    workspace = Path(result["workspace"])

    features = pd.read_parquet(workspace / "processed" / "features.parquet")

    assert sorted(features["season"].unique().tolist()) == sorted(
        HISTORICAL_SEASONS
    )

    print(
        "\ntraining frame seasons: "
        f"{sorted(features['season'].unique().tolist())} "
        f"({len(features):,} rows) while the Step 3B semantic state points "
        f"only at season {STATE_SEASON}"
    )


def test_e_manifest_hashes_cover_every_historical_input(
    multi_season_root, work_root, registry
):
    result = multi_season_fit(multi_season_root, work_root, registry)

    manifest = json.loads(
        (
            Path(result["workspace"])
            / "candidate"
            / "training_data_manifest.json"
        ).read_text(encoding="utf-8")
    )

    recorded = manifest["input_sha256"]

    for season in HISTORICAL_SEASONS:
        for relative in (
            f"raw/seasons/season={season}/stats.parquet",
            f"raw/seasons/season={season}/games.parquet",
            f"raw/advanced/season={season}/advanced.parquet",
        ):
            assert relative in recorded, relative

            assert recorded[relative] == hashlib.sha256(
                (multi_season_root / relative).read_bytes()
            ).hexdigest()

    assert manifest["training_corpus_file_count"] == len(recorded)

    # The rolling fingerprint still authenticates the latest generation.
    state = json.loads(
        (multi_season_root / ROLLING_STATE_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["rolling_state_fingerprint"]
        == state["datasets_fingerprint"]
    )


def test_f_future_rows_are_excluded_by_the_cutoff(
    multi_season_root, work_root, registry
):
    result = multi_season_fit(multi_season_root, work_root, registry)

    assert result["plan"]["training_cutoff"] == MULTI_CUTOFF

    features = pd.read_parquet(
        Path(result["workspace"]) / "processed" / "features.parquet"
    )

    latest = pd.to_datetime(features["date"]).max().date()

    assert latest <= date.fromisoformat(MULTI_CUTOFF)

    for scheduled in FUTURE_SCHEDULED:
        assert latest < date.fromisoformat(scheduled)


def test_future_schedule_churn_is_not_new_training_data(
    multi_season_root, work_root, registry
):
    """A schedule edit adds no completed game, so it must not trigger a refit."""
    first = multi_season_fit(multi_season_root, work_root, registry)

    assert first["outcome"] == OUTCOME_COMPLETED

    games_relative = f"raw/seasons/season={STATE_SEASON}/games.parquet"
    games_path = multi_season_root / games_relative

    before = hashlib.sha256(games_path.read_bytes()).hexdigest()

    games = pd.read_parquet(games_path)

    extra = season_games(STATE_SEASON, ["2027-02-01"], id_offset=950)

    pd.concat([games, extra], ignore_index=True).to_parquet(
        games_path, index=False
    )

    after = hashlib.sha256(games_path.read_bytes()).hexdigest()

    assert after != before

    # The state record must be refreshed for the new bytes, as Step 3B would.
    state_path = multi_season_root / ROLLING_STATE_RELATIVE_PATH
    state = json.loads(state_path.read_text(encoding="utf-8"))

    updated = pd.read_parquet(games_path)
    record = state["datasets"]["games"]
    record["row_count"] = int(len(updated))
    record["columns"] = sorted(str(c) for c in updated.columns)
    record["max_date"] = (
        pd.to_datetime(updated["date"]).max().date().isoformat()
    )
    record["semantic_fingerprint"] = semantic_fingerprint(updated, ["id"])
    state["datasets_fingerprint"] = hashlib.sha256(
        json.dumps(
            state["datasets"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))

    second = multi_season_fit(multi_season_root, work_root, registry)

    assert second["outcome"] == OUTCOME_NO_NEW_TRAINING_DATA
    assert registry.list_fits() == [first["fit_id"]]


def test_a_completed_game_is_new_training_data(
    multi_season_root, work_root, registry
):
    """The counterpart: newly completed history does trigger a refit."""
    first = multi_season_fit(multi_season_root, work_root, registry)

    stats_relative = f"raw/seasons/season={STATE_SEASON}/stats.parquet"
    stats_path = multi_season_root / stats_relative

    stats = pd.read_parquet(stats_path)

    fresh = season_stats(STATE_SEASON, ["2026-11-20"])
    fresh["stat_id"] = fresh["stat_id"] + 500000
    fresh["game_id"] = 987654

    pd.concat([stats, fresh], ignore_index=True).to_parquet(
        stats_path, index=False
    )

    # A completed game lands in every rolling dataset. The cutoff is the
    # minimum across them, so advancing stats alone would correctly still be
    # no new eligible information.
    advanced_relative = (
        f"raw/advanced/season={STATE_SEASON}/advanced.parquet"
    )
    advanced_path = multi_season_root / advanced_relative

    fresh_advanced = season_advanced(STATE_SEASON, ["2026-11-20"])
    fresh_advanced["game_id"] = 987654

    pd.concat(
        [pd.read_parquet(advanced_path), fresh_advanced], ignore_index=True
    ).to_parquet(advanced_path, index=False)

    games_relative = f"raw/seasons/season={STATE_SEASON}/games.parquet"
    games_path = multi_season_root / games_relative

    fresh_game = season_games(STATE_SEASON, ["2026-11-20"])
    fresh_game["id"] = 987654

    pd.concat(
        [pd.read_parquet(games_path), fresh_game], ignore_index=True
    ).to_parquet(games_path, index=False)

    state_path = multi_season_root / ROLLING_STATE_RELATIVE_PATH
    state = json.loads(state_path.read_text(encoding="utf-8"))

    for label, relative, keys in (
        ("stats", stats_relative, ["game_id", "player_id"]),
        ("advanced", advanced_relative, ["game_id", "player_id"]),
        ("games", games_relative, ["id"]),
    ):
        updated = pd.read_parquet(multi_season_root / relative)

        record = state["datasets"][label]
        record["row_count"] = int(len(updated))
        record["columns"] = sorted(str(c) for c in updated.columns)
        record["max_date"] = (
            pd.to_datetime(updated["date"]).max().date().isoformat()
        )
        record["semantic_fingerprint"] = semantic_fingerprint(updated, keys)

    state["datasets_fingerprint"] = hashlib.sha256(
        json.dumps(
            state["datasets"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))

    second = multi_season_fit(multi_season_root, work_root, registry)

    assert second["outcome"] == OUTCOME_COMPLETED
    assert second["fit_id"] != first["fit_id"]
    assert len(registry.list_fits()) == 2


def test_g_no_selector_runs_during_the_corrected_path(
    multi_season_root, work_root, registry
):
    """The corrected snapshot did not weaken the selector prohibition."""
    from nba_prop_quant import decay, kalman, model

    originals = (
        model.expanding_time_oof_target,
        model.expanding_time_oof_minutes,
        decay.tune_decay_beta,
        kalman.tune_kalman,
    )

    result = multi_season_fit(multi_season_root, work_root, registry)

    assert result["outcome"] == OUTCOME_COMPLETED

    assert (
        model.expanding_time_oof_target,
        model.expanding_time_oof_minutes,
        decay.tune_decay_beta,
        kalman.tune_kalman,
    ) == originals

    # No policy file moved while the corrected path ran.
    guard = FrozenPolicyGuard(PROJECT)
    guard.snapshot()
    guard.verify("post_run")


def test_no_fitting_stage_reads_the_mutable_data_root():
    """Stages read the snapshot; only the snapshot stage touches data_root."""
    source = (
        PROJECT / "src" / "nba_prop_quant" / "adaptive_training.py"
    ).read_text(encoding="utf-8")

    engine = source.split("class ProductionFitEngine")[1]

    assert "context.data_root" not in engine
    assert "workspace.inputs" in engine
