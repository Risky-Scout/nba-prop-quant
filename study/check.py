"""Check the study engine, export trusted fitted artifacts, or run a small lesson.

From the repository root:
  python -m study.check
  python -m study.check lesson
  PYTHONPATH=src python -m study.check export /path/to/extracted/runtime

The lesson and default numerical checks use explicitly synthetic inputs.
Export requires the original runtime's dependency versions and trusted pickles.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from . import data, model, price

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODULES = {
    "data": ["normalize", "game_context", "decay", "kalman", "features", "slate", "availability"],
    "model": ["experience", "model", "distributions", "copula", "production", "gate3_v2"],
    "price": ["pricing", "production", "gate3_v2"],
}


def definitions(path):
    tree = ast.parse(path.read_text())
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def check_source_equivalence():
    """Compare executable syntax, ignoring whitespace and comments."""
    count = 0
    for destination, originals in SOURCE_MODULES.items():
        copied = definitions(ROOT / "study" / f"{destination}.py")
        for original in originals:
            source = definitions(ROOT / "src/nba_prop_quant" / f"{original}.py")
            for name in copied.keys() & source.keys():
                if name in {"_default_artifact_dir", "_pmf_over_under_push"}:
                    continue  # Explicit artifact path; documented upper-support bug fix.
                before = ast.dump(source[name], include_attributes=False)
                after = ast.dump(copied[name], include_attributes=False)
                if before != after:
                    raise AssertionError(f"Changed calculation: {destination}.{name}")
                count += 1
    return count


def fake_marginals(module):
    """Known synthetic fitted parameters; no real player claims."""
    result = {}
    for i, target in enumerate(data.TARGETS):
        fitted = module.ZeroInflatedNegativeBinomialCalibrator(["expected_minutes"])
        fitted.size = 5.0 + i
        fitted.coef_ = np.array([-2.5, -0.2])
        fitted.mean_ = np.array([25.0])
        fitted.scale_ = np.array([8.0])
        result[target] = module.FittedMarginal("zinb", fitted)
    return result


def fixture_history():
    rows = []
    # Two teams, six players each, twenty historical games, then a future fixture.
    for game in range(20):
        for team in [1, 2]:
            for player in range(6):
                row = dict(
                    game_id=game + 1,
                    date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=2 * game),
                    season=2024,
                    team_id=team,
                    home_team_id=1,
                    visitor_team_id=2,
                    player_id=team * 100 + player,
                    minutes=15.0 + player * 4,
                    postseason=False,
                    draft_year=2018,
                    position="G" if player < 3 else "F",
                )
                for j, stat in enumerate(data.VOLUME_STATS + ["oreb"]):
                    row[stat] = (game + player + j) % (4 if stat in ["stl", "blk"] else 12)
                rows.append(row)
    return pd.DataFrame(rows)


def check_numerical_equivalence():
    sys.path.insert(0, str(ROOT / "src"))
    original = {
        n: importlib.import_module(f"nba_prop_quant.{n}")
        for n in [
            "distributions",
            "pricing",
            "production",
            "features",
            "slate",
            "copula",
            "gate3_v2",
            "availability",
        ]
    }
    row = pd.Series(
        dict(
            player_id=101,
            game_id=99,
            expected_minutes=28.0,
            **{
                f"mu_selected_{t}": v for t, v in zip(data.TARGETS, [24.0, 8.0, 6.0, 1.3, 0.8, 2.7])
            },
        )
    )
    marginals = fake_marginals(model)
    old_marginals = fake_marginals(original["distributions"])
    mu_columns = {t: f"mu_selected_{t}" for t in data.TARGETS}
    corr = np.eye(6)
    corr[1, 2] = corr[2, 1] = 0.35
    copula = model.GaussianCopula(global_corr=corr)
    old_copula = original["copula"].GaussianCopula(global_corr=corr)
    comparisons = 0
    for target in data.TARGETS:
        mu = row[mu_columns[target]]
        frame = pd.DataFrame({"expected_minutes": np.repeat(28.0, 201)})
        support = np.arange(201)
        means = np.repeat(mu, 201)
        new = marginals[target]
        old = old_marginals[target]
        assert_allclose(
            new.pmf(support, means, frame), old.pmf(support, means, frame), rtol=0, atol=0
        )
        assert_allclose(
            new.cdf(support, means, frame), old.cdf(support, means, frame), rtol=0, atol=0
        )
        u = np.linspace(0.001, 0.999, 1000)
        assert_allclose(new.ppf(u, mu, row), old.ppf(u, mu, row), rtol=0, atol=0)
        assert_allclose(new.model.implied_mean(means, frame), means, rtol=1e-14)
        probabilities = new.pmf(support, means, frame)
        assert_allclose(probabilities.sum(), 1.0, atol=1e-8)
        assert_allclose(probabilities @ support, mu, atol=1e-6)
        comparisons += 6
    for prop in [
        "points_rebounds",
        "points_assists",
        "rebounds_assists",
        "points_rebounds_assists",
    ]:
        strength = 0.85 if prop == "rebounds_assists" else 0.0
        kwargs = dict(
            row=row,
            prop_type=prop,
            lines=[0.0, 4.5, 8.0, 20.5, 30.0],
            mu_columns=mu_columns,
            dependence_lambda=strength,
            simulations=2000,
            seed=73,
        )
        new = price.price_combo_lines(**kwargs, marginals=marginals, copula=copula)
        old = original["pricing"].price_combo_lines(
            **kwargs, marginals=old_marginals, copula=old_copula
        )
        assert_frame_equal(new, old, check_exact=True)
        assert_allclose(new[["p_over", "p_under", "p_push"]].sum(axis=1), 1.0, atol=1e-12)
        comparisons += 2
    history = fixture_history()
    params = model.load_json(ROOT / "models/dynamic_params.json")
    new_features = data.add_dynamic_priors(data.build_base_frame(history), params)
    old_features = original["features"].add_dynamic_priors(
        original["features"].build_base_frame(history), params
    )
    assert_frame_equal(new_features, old_features, check_exact=True)
    players = history.drop_duplicates("player_id").rename(
        columns={"player_id": "id", "team_id": "current_team_id"}
    )
    games = pd.DataFrame(
        [
            dict(
                id=99,
                date="2025-03-01",
                season=2024,
                home_team_id=1,
                visitor_team_id=2,
                postseason=False,
            )
        ]
    )
    kwargs = dict(
        history_stats=history,
        upcoming_games=games,
        active_players=players,
        advanced=pd.DataFrame(),
        dynamic_params=params,
    )
    slate = data.build_upcoming_slate_features(**kwargs)
    assert_frame_equal(
        slate, original["slate"].build_upcoming_slate_features(**kwargs), check_exact=True
    )
    slate["expected_minutes"] = 28.0
    injuries = pd.DataFrame([dict(player_id=101, status="Out")])
    assert_frame_equal(
        data.apply_current_injury_adjustment(slate, injuries),
        original["availability"].apply_current_injury_adjustment(slate, injuries),
        check_exact=True,
    )
    try:
        data.assert_history_precedes_slate(history, "test", "2025-01-01")
    except data.HistoryLeakageError:
        pass
    else:
        raise AssertionError("Same-day history was accepted")
    comparisons += 4
    policy = model.load_json(ROOT / "models/market_probability_calibration_policy.json")
    frame = pd.DataFrame(
        [
            dict(
                prop_type=p,
                p_over=0.45,
                p_under=0.5,
                p_push=0.05,
                q_over_nonpush=0.45 / 0.95,
                q_under_nonpush=0.5 / 0.95,
                over_odds=-110,
                under_odds=-110,
                gate3_role_ready=1,
                gate3_delta_points_assists=2.0,
                gate3_delta_points_rebounds=-1.0,
            )
            for p in policy["props"]
        ]
    )
    parameters = model.load_json(
        ROOT / "research/v2_gate3_deployment_artifacts/probability_parameters.json"
    )
    new_role = price.prepare_gate3_candidate_probability_overrides(
        frame, calibration_policy=policy, probability_parameters=parameters
    )
    old_role = original["gate3_v2"].prepare_gate3_candidate_probability_overrides(
        frame, calibration_policy=policy, probability_parameters=parameters
    )
    assert_frame_equal(new_role, old_role, check_exact=True)
    new = price.add_market_probability_layer(new_role, policy)
    old = original["production"].add_market_probability_layer(old_role, policy)
    assert_frame_equal(new, old, check_exact=True)
    assert_allclose(
        new[["calibrated_p_over", "calibrated_p_under", "calibrated_p_push"]].sum(axis=1),
        1.0,
        atol=1e-12,
    )
    assert not new["auto_bet"].any()
    # Regression: old code incorrectly returned over=1 for integer lines above support.
    assert price._pmf_over_under_push(np.array([0.2, 0.8]), 5.0) == (0.0, 1.0, 0.0)
    assert price._pmf_over_under_push(np.array([0.2, 0.8]), 5.5) == (0.0, 1.0, 0.0)
    return comparisons + 6


def check_offline_workflow():
    """Exercise both new entry points using small, explicitly synthetic fitted models."""
    import hashlib

    from sklearn.dummy import DummyRegressor

    history = fixture_history()
    params = model.load_json(ROOT / "models/dynamic_params.json")
    features = data.add_dynamic_priors(data.build_base_frame(history), params)
    curves = model.fit_production_experience_curves(features, data.TARGETS)
    features = model.apply_production_experience_curves(features, curves)
    training_params = {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}
    minutes = model.fit_minutes_model(features, training_params)
    features["expected_minutes"] = minutes.predict(features)
    players = history.drop_duplicates("player_id").rename(
        columns={"player_id": "id", "team_id": "current_team_id"}
    )
    games = pd.DataFrame(
        [
            dict(
                id=99,
                date="2025-03-01",
                season=2024,
                home_team_id=1,
                visitor_team_id=2,
                postseason=False,
            )
        ]
    )
    lineups = pd.DataFrame(
        [
            dict(game_id=99, player_id=team * 100 + i, team_id=team, starter=i < 5)
            for team in [1, 2]
            for i in range(6)
        ]
    )
    with tempfile.TemporaryDirectory() as temp:
        artifacts = Path(temp)
        joblib.dump(curves, artifacts / "experience_curves.joblib")
        minutes.save(artifacts / "minutes.joblib")
        for target in data.TARGETS:
            model.fit_target_model(features, target, training_params).save(
                artifacts / f"{target}.joblib"
            )
        joblib.dump(fake_marginals(model), artifacts / "marginals.joblib")
        joblib.dump(model.GaussianCopula(global_corr=np.eye(6)), artifacts / "copula.joblib")
        for name in [
            "dynamic_params",
            "mean_model_selection",
            "combo_dependence_policy",
            "market_probability_calibration_policy",
        ]:
            shutil.copy2(ROOT / "models" / f"{name}.json", artifacts / f"{name}.json")
        role = artifacts / "role"
        role.mkdir()
        source_role = ROOT / "research/v2_gate3_deployment_artifacts"
        for name in [
            "deployment_manifest.json",
            "probability_parameters.json",
            "role_state_seed.json",
        ]:
            shutil.copy2(source_role / name, role / name)
        dummy = DummyRegressor(strategy="constant", constant=0.0).fit(
            features[["expected_minutes"]], np.zeros(len(features))
        )
        joblib.dump(
            {"model": dummy, "feature_names": ["expected_minutes"]},
            role / "role_minutes_model.joblib",
        )
        # Only this synthetic temporary fixture has synthetic checksums.
        checksums = "".join(
            f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n"
            for f in sorted(role.iterdir())
        )
        (role / "SHA256SUMS.txt").write_text(checksums)
        projections = price.project_slate(
            history,
            games,
            players,
            pd.DataFrame(),
            pd.DataFrame(),
            lineups,
            artifacts,
            artifacts / "snapshots",
        )
        assert len(projections) == 12
        assert projections["gate3_role_ready"].eq(1).all()
        assert projections["expected_minutes"].ge(0).all()
        quotes = pd.DataFrame(
            [
                dict(
                    game_id=99,
                    player_id=101,
                    prop_type=prop,
                    line_value=line,
                    market_type="over_under",
                    over_odds=-110,
                    under_odds=-110,
                )
                for prop in model.load_json(
                    artifacts / "market_probability_calibration_policy.json"
                )["props"]
                for line in [4.0, 4.5]
            ]
        )
        priced, rejected = price.price_markets(projections, quotes, artifacts, simulations=200)
        assert len(priced) == 20 and rejected.empty
        assert priced["study_only"].all() and not priced["auto_bet"].any()
        assert_allclose(
            priced[["calibrated_p_over", "calibrated_p_under", "calibrated_p_push"]].sum(axis=1),
            1.0,
            atol=1e-12,
        )
        # Missing role information must remove exactly the three role-dependent props.
        projections["gate3_role_ready"] = 0
        ready, refused = price.price_markets(projections, quotes, artifacts, simulations=200)
        assert len(ready) == 14 and len(refused) == 6
        assert set(refused["quote_filter_reason"]) == {"gate3_role_state_unavailable"}
    return 8


def convert_artifact(value):
    """Copy the project's small dataclasses into the study module's class names.

    External fitted estimators and arrays retain their original fitted values.
    No monkeypatching and no dependence on old source during study prediction.
    """
    if isinstance(value, dict):
        return {k: convert_artifact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_artifact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(convert_artifact(v) for v in value)
    if type(value).__module__.startswith("nba_prop_quant."):
        cls = getattr(model, type(value).__name__)
        copied = cls.__new__(cls)
        copied.__dict__.update({k: convert_artifact(v) for k, v in vars(value).items()})
        return copied
    return value


def export_artifacts(runtime, destination):
    """Only run on a trusted extracted runtime, using its matching dependencies."""
    runtime = Path(runtime).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    sys.path.insert(0, str(ROOT / "src"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staging = Path(temp) / "artifacts"
        staging.mkdir()
        for name in ["minutes", *data.TARGETS, "marginals", "copula", "experience_curves"]:
            obj = joblib.load(runtime / "models" / f"{name}.joblib")
            joblib.dump(convert_artifact(obj), staging / f"{name}.joblib")
        for name in [
            "dynamic_params",
            "mean_model_selection",
            "combo_dependence_policy",
            "market_probability_calibration_policy",
        ]:
            shutil.copy2(runtime / "models" / f"{name}.json", staging / f"{name}.json")
        shutil.copytree(runtime / "research/v2_gate3_deployment_artifacts", staging / "role")
        (staging / "STUDY_ONLY.txt").write_text(
            "Offline learning artifacts. Not certified production artifacts.\n"
        )
        staging.rename(destination)
    print(f"Exported study artifacts to {destination}")


def check_runtime_objects(runtime):
    """Compare available published objects on synthetic inputs; report load failures."""
    sys.path.insert(0, str(ROOT / "src"))
    runtime = Path(runtime)
    passed, blocked = [], {}
    for name in ["minutes", *data.TARGETS, "marginals", "copula", "experience_curves"]:
        try:
            original = joblib.load(runtime / "models" / f"{name}.joblib")
            copied = convert_artifact(original)
            if name in ["minutes", *data.TARGETS]:
                rng = np.random.default_rng(73)
                inputs = pd.DataFrame(
                    rng.uniform(0.1, 10.0, (20, len(original.feature_names))),
                    columns=original.feature_names,
                )
                assert_allclose(original.predict(inputs), copied.predict(inputs), rtol=0, atol=0)
            elif name == "marginals":
                for target in data.TARGETS:
                    features = original[target].model.inflation_features
                    inputs = pd.DataFrame(1.0, index=range(50), columns=features)
                    counts, means = np.arange(50), np.repeat(5.0, 50)
                    assert_allclose(
                        original[target].pmf(counts, means, inputs),
                        copied[target].pmf(counts, means, inputs),
                        rtol=0,
                        atol=0,
                    )
            elif name == "copula":
                assert_allclose(
                    original.correlation_for_player(None),
                    copied.correlation_for_player(None),
                    rtol=0,
                    atol=0,
                )
            elif name == "experience_curves":
                for target in data.TARGETS:
                    assert_allclose(
                        original[target].predict(np.array([1.0, 5.0, 10.0])),
                        copied[target].predict(np.array([1.0, 5.0, 10.0])),
                        rtol=0,
                        atol=0,
                    )
            passed.append(name)
        except (ImportError, AttributeError, ValueError) as error:
            blocked[name] = f"{type(error).__name__}: {error}"
    print(
        json.dumps(
            {
                "compared_on_synthetic_inputs": passed,
                "blocked": blocked,
                "production_replay_certified": False,
            },
            indent=2,
        )
    )


def lesson():
    """First exercise: one player's distribution, one line, one fair price."""
    marginal = fake_marginals(model)["pts"]
    row = pd.Series({"expected_minutes": 28.0})
    mean = 24.0
    line = 24.5
    over, under, push = marginal.over_under_push(line, mean, row)
    q_over = over / (over + under)
    print("SYNTHETIC LEARNING EXAMPLE; not a player forecast")
    print(f"Mean = {mean}, line = {line}")
    print(f"Over = {over:.6f}, under = {under:.6f}, push = {push:.6f}")
    print(f"Fair over odds = {price.fair_american(q_over):+d}")
    print("Exercise: change line to 24.0, then explain why push is positive.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["check", "lesson", "export", "runtime-check"],
        default="check",
        nargs="?",
    )
    parser.add_argument("runtime", nargs="?")
    parser.add_argument("--output", default=str(ROOT / "study/artifacts"))
    args = parser.parse_args()
    if args.command == "lesson":
        lesson()
    elif args.command == "runtime-check":
        if not args.runtime:
            parser.error("runtime-check needs the extracted runtime directory")
        check_runtime_objects(args.runtime)
    elif args.command == "export":
        if not args.runtime:
            parser.error("export needs the extracted runtime directory")
        export_artifacts(args.runtime, args.output)
    else:
        source_checks = check_source_equivalence()
        numerical_checks = check_numerical_equivalence() + check_offline_workflow()
        print(
            f"PASS: {source_checks} unchanged function/class syntax comparisons; {numerical_checks} numerical/invariant checks"
        )
        print("Scope: core mathematics and synthetic offline inputs; not a full production replay.")


if __name__ == "__main__":
    main()
