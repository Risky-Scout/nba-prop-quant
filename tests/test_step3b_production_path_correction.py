"""Step 3B production-path correction: Gate 3 live features and pricing lineage.

Two serving-path defects are covered here.

The fitted Gate 3 role model asks for ``player_game_number`` and
``team_game_number``. Historical feature construction produces both, the live
upcoming-slate path did not, and ``apply_gate3_role_state`` fails closed on any
absent feature, so assists, points+assists and points+rebounds could not be
priced. The live values must reproduce the training definitions exactly.

The pricing emitter read ``external_test_record`` before creating it, and no
upstream stage produces that column, so the lineage tail raised ``KeyError``.

Nothing here trains, refits or recalibrates a model, and nothing touches the
network. The repository is read from; every write goes to a pytest temporary
directory.
"""

from __future__ import annotations

import ast
import json
import socket
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from nba_prop_quant.features import build_base_frame
from nba_prop_quant.gate3_v2 import (
    GATE3_CHANGED_PROPS,
    apply_gate3_role_state,
    load_gate3_runtime,
)
from nba_prop_quant.slate import build_upcoming_slate_features

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_features import synthetic_stats  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]

GATE3_ARTIFACTS = PROJECT / "research" / "v2_gate3_deployment_artifacts"

ROLE_MODEL_PATH = GATE3_ARTIFACTS / "role_minutes_model.joblib"

PRICING_SCRIPT = PROJECT / "scripts" / "15_price_markets.py"

SLATE_DATE = "2025-10-27"

# The eleven role columns are merged in by apply_gate3_role_state itself from
# lineup data, and expected_minutes is produced by the minutes step that runs
# between the slate builder and the Gate 3 step. Neither is the slate builder's
# responsibility, so neither counts as a slate gap.
ROLE_COLUMNS_FROM_GATE3_STEP = frozenset(
    {
        "starter",
        "prev_starter",
        "starter_rate5",
        "starter_rate10",
        "starter_surprise",
        "promoted_to_starter",
        "demoted_to_bench",
        "team_starter_overlap",
        "team_new_starters",
        "team_lost_starters",
        "team_lineup_rows",
    }
)

COLUMNS_FROM_MINUTES_STEP = frozenset({"expected_minutes"})


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
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def role_model_features() -> list[str]:
    return list(joblib.load(ROLE_MODEL_PATH)["feature_names"])


@pytest.fixture(scope="module")
def dynamic_params() -> dict:
    return json.loads(
        (PROJECT / "models" / "dynamic_params.json").read_text(
            encoding="utf-8"
        )
    )


def history_frame() -> pd.DataFrame:
    """Four players over four game days, two teams, ending 2025-10-26."""
    return synthetic_stats()


def advanced_frame(features: list[str]) -> pd.DataFrame:
    """Advanced history carrying the adv_prior_* inputs the role model wants."""
    wanted = [
        name.removeprefix("adv_prior_")
        for name in features
        if name.startswith("adv_prior_")
    ]

    return pd.DataFrame(
        [
            {
                "player_id": player_id,
                "game_id": game_id,
                "date": moment,
                **{name: 20.0 + player_id for name in wanted},
            }
            for game_id, moment in enumerate(
                pd.date_range("2025-10-20", periods=4, freq="2D"), start=1
            )
            for player_id in (1, 2, 3, 4)
        ]
    )


def upcoming_games(slate_date: str = SLATE_DATE) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 99,
                "date": pd.Timestamp(slate_date),
                "season": 2025,
                "home_team_id": 10,
                "visitor_team_id": 20,
                "postseason": False,
            }
        ]
    )


def active_players() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": player_id,
                "current_team_id": team_id,
                "position": "G" if player_id % 2 else "F",
                "draft_year": 2020,
                "first_name": f"P{player_id}",
                "last_name": "Test",
            }
            for player_id, team_id in [(1, 10), (2, 10), (3, 20), (4, 20)]
        ]
    )


def live_slate(
    dynamic_params: dict,
    features: list[str],
    slate_date: str = SLATE_DATE,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_upcoming_slate_features(
        history_stats=history_frame(),
        upcoming_games=upcoming_games(slate_date) if games is None else games,
        active_players=active_players(),
        advanced=advanced_frame(features),
        dynamic_params=dynamic_params,
    )


# ----------------------------------------------------------------------
# Gate 3 live-feature parity
# ----------------------------------------------------------------------


def test_live_slate_supplies_every_required_role_model_feature(
    dynamic_params, role_model_features
):
    """The gap that made the live Gate 3 path fail closed.

    Before this correction the live slate was missing player_game_number and
    team_game_number, which are the only two the slate builder owns and did
    not produce.
    """
    slate = live_slate(dynamic_params, role_model_features)

    missing = sorted(
        name
        for name in role_model_features
        if name not in slate.columns
        and name not in ROLE_COLUMNS_FROM_GATE3_STEP
        and name not in COLUMNS_FROM_MINUTES_STEP
    )

    assert missing == [], (
        "the live slate does not supply role-model features it owns: "
        f"{missing}"
    )


def test_the_two_corrected_features_are_present(
    dynamic_params, role_model_features
):
    slate = live_slate(dynamic_params, role_model_features)

    assert "player_game_number" in slate.columns
    assert "team_game_number" in slate.columns

    assert set(role_model_features) >= {
        "player_game_number",
        "team_game_number",
    }


def historical_next_game_row(history: pd.DataFrame, game: dict) -> pd.Series:
    """Run the training construction over history plus the next game.

    build_base_frame assigns player_game_number and team_game_number for every
    row, so appending the upcoming game and reading its row gives the values
    training semantics would assign to it. The appended row's box-score values
    are irrelevant: both features are pure counts of prior games.
    """
    appended = history.iloc[[-1]].copy()

    for column, value in game.items():
        appended[column] = value

    combined = pd.concat([history, appended], ignore_index=True)

    frame = build_base_frame(combined)

    match = frame[
        frame["player_id"].eq(game["player_id"])
        & frame["game_id"].eq(game["game_id"])
    ]

    assert len(match) == 1

    return match.iloc[0]


@pytest.mark.parametrize("player_id,team_id", [(1, 10), (3, 20)])
def test_player_game_number_matches_training_semantics(
    dynamic_params, role_model_features, player_id, team_id
):
    """Live value equals what build_base_frame assigns the same next game."""
    slate = live_slate(dynamic_params, role_model_features)

    live = slate.loc[slate["player_id"].eq(player_id)].iloc[0]

    historical = historical_next_game_row(
        history_frame(),
        {
            "player_id": player_id,
            "team_id": team_id,
            "game_id": 99,
            "date": pd.Timestamp(SLATE_DATE),
            "season": 2025,
            "home_team_id": 10,
            "visitor_team_id": 20,
        },
    )

    assert int(live["player_game_number"]) == int(
        historical["player_game_number"]
    )


@pytest.mark.parametrize("player_id,team_id", [(1, 10), (3, 20)])
def test_team_game_number_matches_training_semantics(
    dynamic_params, role_model_features, player_id, team_id
):
    slate = live_slate(dynamic_params, role_model_features)

    live = slate.loc[slate["player_id"].eq(player_id)].iloc[0]

    historical = historical_next_game_row(
        history_frame(),
        {
            "player_id": player_id,
            "team_id": team_id,
            "game_id": 99,
            "date": pd.Timestamp(SLATE_DATE),
            "season": 2025,
            "home_team_id": 10,
            "visitor_team_id": 20,
        },
    )

    assert int(live["team_game_number"]) == int(
        historical["team_game_number"]
    )


def test_player_game_number_is_career_games_prior(
    dynamic_params, role_model_features
):
    """build_base_frame defines them as the same value; live must agree."""
    slate = live_slate(dynamic_params, role_model_features)

    assert (
        slate["player_game_number"].tolist()
        == slate["career_games_prior"].tolist()
    )


def test_team_game_number_is_consistent_with_season_progress(
    dynamic_params, role_model_features
):
    """season_progress is (team_game_number - 1) / 82 in training."""
    slate = live_slate(dynamic_params, role_model_features)

    expected = (slate["team_game_number"] - 1) / 82.0

    pd.testing.assert_series_equal(
        slate["season_progress"].astype(float),
        expected.astype(float),
        check_names=False,
    )


def test_role_features_count_only_history_before_the_slate(
    dynamic_params, role_model_features
):
    """No same-day or future result may influence either value."""
    history = history_frame()

    slate = live_slate(dynamic_params, role_model_features)

    row = slate.loc[slate["player_id"].eq(1)].iloc[0]

    prior_games = int(history["player_id"].eq(1).sum())

    assert int(row["player_game_number"]) == prior_games

    team_games = int(
        history.loc[
            history["team_id"].eq(10) & history["season"].eq(2025),
            "game_id",
        ].nunique()
    )

    assert int(row["team_game_number"]) == team_games + 1


def test_future_scheduled_games_do_not_inflate_the_counts(
    dynamic_params, role_model_features
):
    """games.parquet legitimately holds future rows; they are not history."""
    baseline = live_slate(dynamic_params, role_model_features)

    with_future = live_slate(
        dynamic_params,
        role_model_features,
        games=pd.concat(
            [
                upcoming_games(SLATE_DATE),
                upcoming_games("2025-11-05").assign(id=100),
            ],
            ignore_index=True,
        ),
    )

    first_slate_game = with_future.loc[with_future["game_id"].eq(99)]

    assert (
        first_slate_game["player_game_number"].tolist()
        == baseline["player_game_number"].tolist()
    )
    assert (
        first_slate_game["team_game_number"].tolist()
        == baseline["team_game_number"].tolist()
    )


# ----------------------------------------------------------------------
# Gate 3 end-to-end offline path
# ----------------------------------------------------------------------


def prepared_slate(dynamic_params, features) -> pd.DataFrame:
    """Mirror the prediction script's stages up to the Gate 3 step.

    The minutes model and the mean policy are not what is under test, and
    loading them would test those artifacts instead of the role-feature
    contract. Their outputs are stood in for with the slate's own prior
    features, which is enough for the Gate 3 step to run. The role-model
    feature check this exercises reads column names, not values.
    """
    slate = live_slate(dynamic_params, features)

    slate["expected_minutes"] = slate["prior_minutes10"]

    for target in ("pts", "reb", "ast", "stl", "blk", "fg3m"):
        mean = slate[f"prior_{target}_rate10"] * slate["expected_minutes"]

        slate[f"mu_selected_{target}"] = mean
        slate[f"mu_{target}"] = mean

    return slate


def current_lineups(slate: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": int(row["game_id"]),
                "player_id": int(row["player_id"]),
                "team_id": int(row["team_id"]),
                "starter": 1,
            }
            for _, row in slate.iterrows()
        ]
    )


def test_slate_passes_through_apply_gate3_role_state(
    tmp_path, dynamic_params, role_model_features
):
    """The live path reaches Gate 3 role state without the missing-feature stop."""
    slate = prepared_slate(dynamic_params, role_model_features)

    result = apply_gate3_role_state(
        slate,
        current_lineups(slate),
        target_date=SLATE_DATE,
        snapshot_dir=tmp_path / "snapshots",
        runtime=load_gate3_runtime(),
    )

    assert len(result) == len(slate)

    for column in (
        "gate3_role_ready",
        "gate3_role_minutes",
        "gate3_mu_ast",
        "gate3_candidate_policy_id",
    ):
        assert column in result.columns


def test_absent_role_feature_still_fails_closed(
    tmp_path, dynamic_params, role_model_features
):
    """The feature-name check must not have been weakened."""
    slate = prepared_slate(dynamic_params, role_model_features)

    slate = slate.drop(columns=["team_game_number"])

    with pytest.raises(RuntimeError, match="missing live features"):
        apply_gate3_role_state(
            slate,
            current_lineups(slate),
            target_date=SLATE_DATE,
            snapshot_dir=tmp_path / "snapshots",
            runtime=load_gate3_runtime(),
        )


def test_gate3_changed_props_are_the_ones_this_unblocks():
    assert set(GATE3_CHANGED_PROPS) == {
        "assists",
        "points_assists",
        "points_rebounds",
    }


# ----------------------------------------------------------------------
# Pricing lineage
# ----------------------------------------------------------------------


def projection_assigned_columns() -> set[str]:
    """Columns the projection emitter assigns onto its output frame."""
    tree = ast.parse(
        (PROJECT / "scripts" / "10_predict_slate.py").read_text(
            encoding="utf-8"
        )
    )

    return {
        target.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "slate"
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    }


def test_external_test_record_is_not_supplied_upstream():
    """Outcome B: no upstream stage guarantees the column.

    This is the evidence for correcting the emitter rather than documenting a
    guarantee. The projection feed carries gate3_external_test_candidate.
    """
    assigned = projection_assigned_columns()

    assert "external_test_record" not in assigned
    assert "gate3_external_test_candidate" in assigned

    sources = [
        path
        for path in list((PROJECT / "src").rglob("*.py"))
        + list((PROJECT / "scripts").rglob("*.py"))
        if "external_test_record" in path.read_text(encoding="utf-8")
    ]

    assert [path.name for path in sources] == ["15_price_markets.py"]


def pricing_lineage_tail() -> ast.Module:
    """Extract the emitter's own lineage statements, so the real code runs.

    Everything from priced_at_utc up to the output path is taken verbatim from
    scripts/15_price_markets.py rather than reimplemented, so a regression in
    the shipped ordering fails this test.
    """
    tree = ast.parse(PRICING_SCRIPT.read_text(encoding="utf-8"))

    main_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def assigns(node: ast.stmt, name: str, key: str | None) -> bool:
        if not isinstance(node, ast.Assign):
            return False

        target = node.targets[0]

        if key is None:
            return isinstance(target, ast.Name) and target.id == name

        return (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == name
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == key
        )

    body: list[ast.stmt] = []
    capturing = False

    for node in main_function.body:
        if assigns(node, "priced", "priced_at_utc"):
            capturing = True

        if capturing:
            if assigns(node, "out_dir", None):
                break

            body.append(node)

    assert body, "could not locate the pricing lineage tail"

    return ast.Module(body=body, type_ignores=[])


def run_pricing_lineage_tail(priced: pd.DataFrame) -> pd.DataFrame:
    namespace = {
        "priced": priced,
        "priced_at": "2026-11-15T23:05:00+00:00",
        "manifest": {
            "freeze_id": "nba_prop_quant_20260818T205213Z",
            "freeze_stage": "external_test_deployment",
            "manifest_sha256": "a" * 64,
        },
        "gate3_runtime": {
            "candidate_id": "nba_prop_quant_v2_gate3_dd2d394b6def",
            "gate3_lock_commit": "dd2d394b6def1e146a193b9000536158d1ec383d",
            "deployment_manifest_sha256": "b" * 64,
        },
    }

    exec(  # noqa: S102 - executing the shipped emitter statements is the point
        compile(pricing_lineage_tail(), str(PRICING_SCRIPT), "exec"),
        namespace,
    )

    return namespace["priced"]


def priced_frame(candidate: list[bool] | None = None) -> pd.DataFrame:
    """A merged pricing frame as the emitter would hold it at the tail."""
    if candidate is None:
        candidate = [True, True, True]

    return pd.DataFrame(
        {
            "game_id": [1, 1, 2],
            "player_id": [10, 11, 12],
            "prop_type": ["assists", "points", "points_rebounds"],
            "line_value": [6.5, 24.5, 31.5],
            "calibrated_q_over_nonpush": [0.52, 0.48, 0.55],
            "gate3_external_test_candidate": candidate,
            "gate3_capture_id": ["cap-1", "cap-1", "cap-2"],
            "gate3_capture_window_id": ["w-1", "w-1", "w-2"],
        }
    )


def test_pricing_lineage_tail_no_longer_raises():
    """Reproduces the original failure mode and proves it is resolved."""
    result = run_pricing_lineage_tail(priced_frame())

    assert "external_test_record" in result.columns
    assert "gate3_external_test_record" in result.columns


def test_external_test_record_semantics():
    """Run level: is this pricing run an external-test deployment."""
    result = run_pricing_lineage_tail(priced_frame())

    assert result["external_test_record"].tolist() == [True, True, True]

    namespace_frame = priced_frame()

    tail = pricing_lineage_tail()

    scope = {
        "priced": namespace_frame,
        "priced_at": "2026-11-15T23:05:00+00:00",
        "manifest": {
            "freeze_id": "nba_prop_quant_20260818T205213Z",
            "freeze_stage": "production_deployment",
            "manifest_sha256": "a" * 64,
        },
        "gate3_runtime": {
            "candidate_id": "nba_prop_quant_v2_gate3_dd2d394b6def",
            "gate3_lock_commit": "dd2d394b6def1e146a193b9000536158d1ec383d",
            "deployment_manifest_sha256": "b" * 64,
        },
    }

    exec(  # noqa: S102
        compile(tail, str(PRICING_SCRIPT), "exec"),
        scope,
    )

    # A non external-test freeze stage makes both flags false.
    assert scope["priced"]["external_test_record"].tolist() == [
        False,
        False,
        False,
    ]
    assert scope["priced"]["gate3_external_test_record"].tolist() == [
        False,
        False,
        False,
    ]


def test_gate3_external_test_record_semantics():
    """Gate 3 level: candidate on the projection AND an external-test run."""
    result = run_pricing_lineage_tail(
        priced_frame(candidate=[True, False, True])
    )

    assert result["gate3_external_test_record"].tolist() == [
        True,
        False,
        True,
    ]


def test_the_two_lineage_fields_remain_distinct():
    result = run_pricing_lineage_tail(
        priced_frame(candidate=[True, False, True])
    )

    assert (
        result["external_test_record"].tolist()
        != result["gate3_external_test_record"].tolist()
    )


def test_pricing_tail_preserves_identity_fields():
    """Gate 3 candidate identity and capture identity survive the tail."""
    result = run_pricing_lineage_tail(priced_frame())

    assert result["gate3_candidate_policy_id"].tolist() == [
        "nba_prop_quant_v2_gate3_dd2d394b6def"
    ] * 3

    assert result["gate3_policy_lock_commit"].tolist() == [
        "dd2d394b6def1e146a193b9000536158d1ec383d"
    ] * 3

    assert result["gate3_deployment_manifest_sha256"].tolist() == [
        "b" * 64
    ] * 3

    assert result["gate3_capture_id"].tolist() == ["cap-1", "cap-1", "cap-2"]
    assert result["gate3_capture_window_id"].tolist() == ["w-1", "w-1", "w-2"]

    assert result["market_pricing_schema_version"].tolist() == [3, 3, 3]
    assert result["freeze_id"].tolist() == [
        "nba_prop_quant_20260818T205213Z"
    ] * 3


def test_pricing_tail_depends_on_the_projection_candidacy_column():
    """The dependency is explicit, so a future upstream drop fails loudly."""
    frame = priced_frame().drop(columns=["gate3_external_test_candidate"])

    with pytest.raises(KeyError, match="gate3_external_test_candidate"):
        run_pricing_lineage_tail(frame)
