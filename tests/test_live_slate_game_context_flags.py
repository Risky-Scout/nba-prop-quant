"""The five game-context flags a live slate owes the fitted models.

``FEATURE_EXACT`` names ``is_regular_season``, ``is_nba_cup``,
``is_nba_cup_championship``, ``is_play_in`` and ``is_playoffs``, and
``assert_model_feature_contract`` is a column-presence check with no
allowance, so a slate missing any of them is refused before a single
projection is produced. ``build_upcoming_slate_features`` built its rows from
scratch and emitted none of the five: they reached the training frame only
through ``classify_games`` via ``build_base_frame``, which no live path ever
called.

These tests are about that one defect. They check each competition regime the
classifier distinguishes, that the live answer is the classifier's own answer
rather than a second implementation of its rules, that the builder emits all
five, and that the feature contract then passes with no diagnostic override.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nba_prop_quant.features import FEATURE_EXACT, build_base_frame
from nba_prop_quant.game_context import classify_games
from nba_prop_quant.production import assert_model_feature_contract
from nba_prop_quant.slate import (
    GAME_CONTEXT_FLAGS,
    SlateGameContextError,
    build_upcoming_slate_features,
    classify_upcoming_games,
)

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from season_games_fixture import FULL_SCHEDULE_LENGTH, pad_season_schedule  # noqa: E402

#: A season late enough that BALLDONTLIE populates ``ist_stage``, so the Cup
#: regimes come from the stage label rather than from the legacy date table.
SEASON = 2026

HOME = 11
AWAY = 22

#: Two more teams, so a fixture can schedule a game that does not involve the
#: pair the slate is about.
OTHER_HOME = 33
OTHER_AWAY = 44

SLATE_ID = 9901

LIFECYCLE_WORKFLOW = PROJECT / ".github" / "workflows" / "nba_production_lifecycle.yml"

PRODUCTION_REF = "production/wizardofodds-integration"


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


def game(
    identifier: int,
    moment: str,
    *,
    home_team_id: int = HOME,
    visitor_team_id: int = AWAY,
    postseason: bool = False,
    ist_stage: object = None,
    season: int = SEASON,
) -> dict:
    return {
        "id": identifier,
        "date": pd.Timestamp(moment),
        "season": season,
        "home_team_id": home_team_id,
        "visitor_team_id": visitor_team_id,
        "postseason": postseason,
        "ist_stage": ist_stage,
    }


def season_schedule(
    *extra: dict,
    schedule_length: int = FULL_SCHEDULE_LENGTH,
) -> pd.DataFrame:
    """A season's whole games table, in the shape ``games.parquet`` has."""
    return pad_season_schedule(
        pd.DataFrame(
            [
                game(600_000 + index, moment)
                for index, moment in enumerate(
                    pd.date_range("2026-10-21", periods=4, freq="2D")
                )
            ]
            + list(extra)
        ),
        schedule_length=schedule_length,
    )


#: Six teams, which is the smallest fixture in which a completed schedule can
#: be told apart from a team that has played one game more than it. The league
#: has thirty for the same reason: the inferred schedule length is the *modal*
#: team appearance count, so the teams that do not reach the Play-In are what
#: establishes the number the Play-In teams exceed. A two-team fixture cannot
#: express that -- both teams play the extra game, so the mode moves with them.
PLAY_IN_TEAMS = (HOME, AWAY, OTHER_HOME, OTHER_AWAY, 55, 66)


def completed_season(
    team_games: int = FULL_SCHEDULE_LENGTH,
    *,
    teams: tuple[int, ...] = PLAY_IN_TEAMS,
    last_date: str = "2027-04-05",
) -> list[dict]:
    """A finished season: ``team_games`` fixtures for every team in ``teams``.

    Each round pairs the teams off so every one of them gains exactly one
    appearance, and the rounds run on consecutive days up to ``last_date``.
    """
    if len(teams) % 2:
        raise ValueError("a round needs an even number of teams")

    pairs = [
        (teams[index], teams[index + 1]) for index in range(0, len(teams), 2)
    ]

    rows: list[dict] = []

    dates = pd.date_range(end=last_date, periods=team_games, freq="D")

    for round_index, moment in enumerate(dates):
        for pair_index, (home_team_id, visitor_team_id) in enumerate(pairs):
            rows.append(
                game(
                    700_000 + round_index * len(pairs) + pair_index,
                    moment,
                    home_team_id=home_team_id,
                    visitor_team_id=visitor_team_id,
                )
            )

    return rows


# ----------------------------------------------------------------------
# 1-5: each competition regime the classifier distinguishes
# ----------------------------------------------------------------------


def flags_for(slate: dict, schedule: pd.DataFrame) -> dict[str, int]:
    context = classify_upcoming_games(pd.DataFrame([slate]), schedule)

    assert len(context) == 1
    assert int(context["game_id"].iloc[0]) == slate["id"]

    return {name: int(context[name].iloc[0]) for name in GAME_CONTEXT_FLAGS}


def test_an_ordinary_regular_season_game_is_regular_season():
    """Regime 1: no stage label, not postseason, schedule not yet complete."""
    slate = game(SLATE_ID, "2026-11-04")

    assert flags_for(slate, season_schedule(slate)) == {
        "is_regular_season": 1,
        "is_nba_cup": 0,
        "is_nba_cup_championship": 0,
        "is_play_in": 0,
        "is_playoffs": 0,
    }


def test_an_nba_cup_group_game_is_an_nba_cup_game():
    """Regime 2: Cup group play is a Cup game and still counts in the season.

    ``REGULAR_SEASON_GAME_TYPES`` holds group, quarterfinal and semifinal, so
    ``is_regular_season`` and ``is_nba_cup`` are both 1 here. That is the
    classifier's definition and the live path reproduces it rather than
    picking one of the two.
    """
    slate = game(SLATE_ID, "2026-11-28", ist_stage="Group Play")

    assert flags_for(slate, season_schedule(slate)) == {
        "is_regular_season": 1,
        "is_nba_cup": 1,
        "is_nba_cup_championship": 0,
        "is_play_in": 0,
        "is_playoffs": 0,
    }


@pytest.mark.parametrize("stage", ["Quarterfinals", "Semifinals"])
def test_the_other_nba_cup_rounds_are_nba_cup_games(stage: str):
    """The knockout rounds short of the final are Cup games that still count."""
    slate = game(SLATE_ID, "2026-12-09", ist_stage=stage)

    assert flags_for(slate, season_schedule(slate)) == {
        "is_regular_season": 1,
        "is_nba_cup": 1,
        "is_nba_cup_championship": 0,
        "is_play_in": 0,
        "is_playoffs": 0,
    }


def test_the_nba_cup_championship_is_a_championship_and_not_a_season_game():
    """Regime 3: the final is a Cup game that does not count in the season."""
    slate = game(SLATE_ID, "2026-12-16", ist_stage="Championship")

    assert flags_for(slate, season_schedule(slate)) == {
        "is_regular_season": 0,
        "is_nba_cup": 1,
        "is_nba_cup_championship": 1,
        "is_play_in": 0,
        "is_playoffs": 0,
    }


def test_a_play_in_game_is_a_play_in_game():
    """Regime 4: structural, from the season's own inferred schedule length.

    Nothing labels a Play-In game. Every team has completed a full schedule by
    the slate date, so the next non-playoff, non-Cup game between two of them
    is Play-In -- and the four teams that are not in it are what fixes the
    schedule length the two in it exceed.
    """
    slate = game(SLATE_ID, "2027-04-15")

    schedule = pd.DataFrame([*completed_season(), slate])

    assert flags_for(slate, schedule) == {
        "is_regular_season": 0,
        "is_nba_cup": 0,
        "is_nba_cup_championship": 0,
        "is_play_in": 1,
        "is_playoffs": 0,
    }


def test_the_same_game_one_round_earlier_is_still_a_regular_season_game():
    """The Play-In answer is the count, not the date.

    Same six-team season, same late-April slate date, except the slate game is
    the two teams' last *scheduled* fixture rather than an extra one: their
    eighty-second meeting is removed and the slate takes its place. Everyone
    still finishes on eighty-two, so the classifier says regular season.
    Without this the test above would pass against a classifier that called
    every late-April game a Play-In game.
    """
    played = completed_season()

    final_meeting = max(
        row["id"]
        for row in played
        if {row["home_team_id"], row["visitor_team_id"]} == {HOME, AWAY}
    )

    slate = game(SLATE_ID, "2027-04-15")

    schedule = pd.DataFrame(
        [row for row in played if row["id"] != final_meeting] + [slate]
    )

    assert flags_for(slate, schedule) == {
        "is_regular_season": 1,
        "is_nba_cup": 0,
        "is_nba_cup_championship": 0,
        "is_play_in": 0,
        "is_playoffs": 0,
    }


def test_a_playoff_game_is_a_playoff_game():
    """Regime 5: the postseason flag the API carries."""
    slate = game(SLATE_ID, "2027-04-25", postseason=True)

    assert flags_for(slate, season_schedule(slate)) == {
        "is_regular_season": 0,
        "is_nba_cup": 0,
        "is_nba_cup_championship": 0,
        "is_play_in": 0,
        "is_playoffs": 1,
    }


# ----------------------------------------------------------------------
# 6: the live answer is the training path's answer
# ----------------------------------------------------------------------


REPRESENTATIVE_SLATES = {
    "regular": game(SLATE_ID, "2026-11-04"),
    "cup_group": game(SLATE_ID, "2026-11-28", ist_stage="Group Play"),
    "cup_quarterfinal": game(SLATE_ID, "2026-12-09", ist_stage="Quarterfinals"),
    "cup_championship": game(SLATE_ID, "2026-12-16", ist_stage="Championship"),
    "playoffs": game(SLATE_ID, "2027-04-25", postseason=True),
    "other_teams": game(
        SLATE_ID,
        "2026-11-04",
        home_team_id=OTHER_HOME,
        visitor_team_id=OTHER_AWAY,
    ),
}


@pytest.mark.parametrize("label", sorted(REPRESENTATIVE_SLATES))
def test_the_live_classification_is_the_training_path_classification(label: str):
    """The live flags equal ``classify_games``' own flags for the same game.

    This is the parity the fix rests on: the live path must not be a second
    implementation of the rules. It is checked against the classifier run over
    the whole table, which is what ``build_base_frame`` passes it, so the
    seasons narrowing the live path does cannot change an answer.
    """
    slate = REPRESENTATIVE_SLATES[label]

    schedule = season_schedule(
        slate,
        game(650_000, "2026-11-04", home_team_id=OTHER_HOME, visitor_team_id=OTHER_AWAY),
    )

    live = flags_for(slate, schedule)

    classified = classify_games(schedule)
    row = classified[classified["id"].eq(slate["id"])].iloc[0]

    training = {name: int(row[name]) for name in GAME_CONTEXT_FLAGS}

    assert live == training, label


def test_narrowing_to_the_slates_seasons_changes_no_answer():
    """Seasons are independent, so the narrowing is exact.

    ``classify_games`` infers a schedule length per season and walks each
    season against its own counter, which is why the live path may hand it one
    season rather than twenty. Checked rather than asserted: an earlier
    season is added and every flag stays put.
    """
    slate = game(SLATE_ID, "2026-11-04")

    current = season_schedule(slate)

    earlier = pad_season_schedule(
        pd.DataFrame(
            [
                game(800_000 + index, moment, season=SEASON - 1)
                for index, moment in enumerate(
                    pd.date_range("2025-10-21", periods=4, freq="2D")
                )
            ]
        )
    )

    one_season = flags_for(slate, current)
    two_seasons = flags_for(slate, pd.concat([earlier, current], ignore_index=True))

    assert one_season == two_seasons


def test_a_capture_without_a_stage_label_inherits_the_schedules_stage():
    """The stored schedule fills a field the live capture omitted.

    The season games table carries ``ist_stage`` from the API. A capture of the
    same game that does not carry it must not turn a Cup final into an
    ordinary game, so the two rows for one game id are combined rather than
    one simply replacing the other.
    """
    scheduled = game(SLATE_ID, "2026-12-16", ist_stage="Championship")

    captured = {
        key: value for key, value in scheduled.items() if key != "ist_stage"
    }

    assert flags_for(captured, season_schedule(scheduled)) == {
        "is_regular_season": 0,
        "is_nba_cup": 1,
        "is_nba_cup_championship": 1,
        "is_play_in": 0,
        "is_playoffs": 0,
    }


def test_a_live_capture_overrides_a_stale_scheduled_postseason_flag():
    """Where both rows carry a value, the live capture is the authority."""
    scheduled = game(SLATE_ID, "2027-04-25", postseason=False)

    captured = game(SLATE_ID, "2027-04-25", postseason=True)

    assert flags_for(captured, season_schedule(scheduled))["is_playoffs"] == 1


# ----------------------------------------------------------------------
# the slate's regime cannot be guessed
# ----------------------------------------------------------------------


def test_a_missing_season_schedule_is_refused_rather_than_defaulted():
    """Without the season's games the regime is unknowable, so it refuses.

    Handed the slate alone the classifier infers a one-game season, and every
    Play-In answer that follows is meaningless. Defaulting the five flags to
    zero would be worse: the models would be scored on a regime the slate is
    not in, silently.
    """
    slate = game(SLATE_ID, "2026-11-04")

    with pytest.raises(SlateGameContextError, match="no season"):
        classify_upcoming_games(
            pd.DataFrame([slate]),
            pad_season_schedule(
                pd.DataFrame(
                    [game(900_001, "2025-10-21", season=SEASON - 1)]
                    + [game(900_002, "2025-10-23", season=SEASON - 1)]
                )
            ),
        )


def test_an_empty_games_table_is_refused_rather_than_defaulted():
    slate = game(SLATE_ID, "2026-11-04")

    with pytest.raises(SlateGameContextError, match="none was supplied"):
        classify_upcoming_games(pd.DataFrame([slate]), pd.DataFrame())


def test_a_slate_without_a_postseason_flag_is_refused():
    """``postseason`` decides ``is_playoffs``, so it is required, not assumed."""
    slate = {
        key: value
        for key, value in game(SLATE_ID, "2027-04-25").items()
        if key != "postseason"
    }

    with pytest.raises(SlateGameContextError, match="postseason"):
        classify_upcoming_games(pd.DataFrame([slate]), season_schedule())


def test_no_flag_is_hard_coded_in_the_slate_builder():
    """The five values are read from the classifier, not written literally.

    A constant would survive every test above that happens to expect zero, so
    the source is checked directly: the only place a flag name may appear
    beside a literal is the declaration of the tuple itself.
    """
    source = (PROJECT / "src" / "nba_prop_quant" / "slate.py").read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    offenders: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        rendered = ast.unparse(node)

        for flag in GAME_CONTEXT_FLAGS:
            if f'"{flag}"' not in rendered and f"'{flag}'" not in rendered:
                continue
            if rendered.startswith("GAME_CONTEXT_FLAGS"):
                continue
            offenders.append(rendered[:160])

    assert offenders == [], f"a game-context flag is assigned literally: {offenders}"


# ----------------------------------------------------------------------
# 7: the builder emits all five
# ----------------------------------------------------------------------


def synthetic_history(games: pd.DataFrame) -> pd.DataFrame:
    """Box scores for four players over the games dated before the slate."""
    rows = []

    played = games[games["date"].lt(pd.Timestamp("2026-11-04"))]

    for index, row in enumerate(played.itertuples()):
        for player_id, team_id in ((1, HOME), (2, HOME), (3, AWAY), (4, AWAY)):
            rows.append(
                {
                    "player_id": player_id,
                    "team_id": team_id,
                    "game_id": int(row.id),
                    "date": row.date,
                    "season": SEASON,
                    "postseason": False,
                    "home_team_id": int(row.home_team_id),
                    "visitor_team_id": int(row.visitor_team_id),
                    "position": "G" if player_id % 2 else "F",
                    "draft_year": 2020,
                    "min": "30:00",
                    "minutes": 30.0,
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


def synthetic_active() -> pd.DataFrame:
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
            for player_id, team_id in ((1, HOME), (2, HOME), (3, AWAY), (4, AWAY))
        ]
    )


@pytest.fixture(scope="module")
def dynamic_params() -> dict:
    return json.loads(
        (PROJECT / "models" / "dynamic_params.json").read_text(encoding="utf-8")
    )


def live_slate(slate: dict, dynamic_params: dict) -> pd.DataFrame:
    schedule = season_schedule(slate)

    return build_upcoming_slate_features(
        history_stats=synthetic_history(schedule),
        upcoming_games=pd.DataFrame([slate]),
        active_players=synthetic_active(),
        advanced=pd.DataFrame(columns=["player_id", "game_id", "date"]),
        dynamic_params=dynamic_params,
        history_games=schedule,
    )


def test_the_builder_emits_every_required_game_context_column(dynamic_params):
    """The defect itself: all five present on every row, correctly typed."""
    frame = live_slate(game(SLATE_ID, "2026-11-04"), dynamic_params)

    assert not frame.empty

    for flag in GAME_CONTEXT_FLAGS:
        assert flag in frame.columns, flag
        assert frame[flag].notna().all(), flag
        assert pd.api.types.is_integer_dtype(frame[flag]), flag
        assert set(frame[flag].unique()) <= {0, 1}, flag

    assert (frame["is_regular_season"] == 1).all()


def test_the_five_flags_are_the_ones_the_feature_list_names():
    """The tuple is the model's own list, not a parallel one."""
    assert set(GAME_CONTEXT_FLAGS) == {
        name
        for name in FEATURE_EXACT
        if name.startswith("is_") and name != "is_home"
    }


def test_the_builder_carries_each_regime_through_to_its_rows(dynamic_params):
    """A Cup final slate reaches the model rows as a Cup final."""
    frame = live_slate(
        game(SLATE_ID, "2026-12-16", ist_stage="Championship"), dynamic_params
    )

    assert (frame["is_nba_cup"] == 1).all()
    assert (frame["is_nba_cup_championship"] == 1).all()
    assert (frame["is_regular_season"] == 0).all()


def test_the_live_rows_match_the_training_frames_rows_for_the_same_game(
    dynamic_params,
):
    """Same game, same five values, built both ways.

    ``build_base_frame`` is the training path and merges the classifier's
    output by game id. Running both over the same schedule and comparing the
    rows for one game is the end-to-end form of the parity above.
    """
    schedule = season_schedule(game(SLATE_ID, "2026-11-04"))

    history = synthetic_history(schedule)

    trained = build_base_frame(history, games=schedule)

    played = int(history["game_id"].iloc[0])

    training_row = trained[trained["game_id"].eq(played)].iloc[0]

    live = live_slate(game(SLATE_ID, "2026-11-04"), dynamic_params).iloc[0]

    for flag in GAME_CONTEXT_FLAGS:
        assert int(live[flag]) == int(training_row[flag]), flag


# ----------------------------------------------------------------------
# 8: the feature contract passes with no diagnostic override
# ----------------------------------------------------------------------


class FakeBundle:
    """The minimal shape ``assert_model_feature_contract`` reads."""

    def __init__(self, features: list[str]) -> None:
        self.feature_names = features


def test_the_feature_contract_passes_with_no_diagnostic_override(dynamic_params):
    """The contract is satisfied by the slate, not waived.

    ``allow_missing`` is left at its default, so the call raises unless every
    one of the five is present. The frozen models' own feature lists are
    exercised by the end-to-end bundle tests; this pins the five in isolation
    so the reason a contract failure appears cannot be mistaken.
    """
    frame = live_slate(game(SLATE_ID, "2026-11-04"), dynamic_params)

    missing = assert_model_feature_contract(
        frame,
        FakeBundle(list(GAME_CONTEXT_FLAGS)),
        label="game-context flags",
    )

    assert missing == []


def test_the_contract_still_refuses_a_slate_without_the_flags(dynamic_params):
    """The check that failed before the fix still fails without them.

    Otherwise this suite would pass against a contract that had been weakened
    rather than a slate that had been corrected.
    """
    frame = live_slate(game(SLATE_ID, "2026-11-04"), dynamic_params).drop(
        columns=list(GAME_CONTEXT_FLAGS)
    )

    with pytest.raises(RuntimeError, match="missing 5 trained feature"):
        assert_model_feature_contract(
            frame,
            FakeBundle(list(GAME_CONTEXT_FLAGS)),
            label="game-context flags",
        )


# ----------------------------------------------------------------------
# 9: production never passes the diagnostic override
# ----------------------------------------------------------------------


def test_the_production_lifecycle_passes_no_diagnostic_override():
    """``--allow-missing-model-features`` may not appear in the lifecycle."""
    text = LIFECYCLE_WORKFLOW.read_text(encoding="utf-8")

    assert "--allow-missing-model-features" not in text

    # Parsed as well as scanned, so a step cannot smuggle it in through a
    # differently formatted command.
    workflow = yaml.safe_load(text)

    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            assert "allow-missing-model-features" not in (step.get("run") or "")


def test_the_serving_caller_passes_no_diagnostic_override():
    """Nor may the incumbent serving caller, which builds the command."""
    text = (PROJECT / "ops" / "run_incumbent_production_serving.py").read_text(
        encoding="utf-8"
    )

    assert "--allow-missing-model-features" not in text


def test_the_override_is_absent_from_every_test_that_serves_the_real_scripts():
    """The strict proof must stay strict.

    The real-slate suite passed the override while the flags were missing. A
    reintroduction would make a green run say nothing about the contract, so
    it is pinned here rather than left to review.
    """
    text = (PROJECT / "tests" / "test_real_slate_serving_root.py").read_text(
        encoding="utf-8"
    )

    assert "--allow-missing-model-features" not in text
    assert "allow_missing_model_features" not in text


# ----------------------------------------------------------------------
# 10: no model, prediction or pricing mathematics moved
# ----------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def production_base() -> str | None:
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return git("merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue
    return None


#: Every module that decides a projection, a quantile, a probability or a
#: price. None of them may differ from production on this branch.
NUMERICAL_MODULES = (
    "src/nba_prop_quant/copula.py",
    "src/nba_prop_quant/decay.py",
    "src/nba_prop_quant/distributions.py",
    "src/nba_prop_quant/experience.py",
    "src/nba_prop_quant/features.py",
    "src/nba_prop_quant/game_context.py",
    "src/nba_prop_quant/gate3_v2.py",
    "src/nba_prop_quant/kalman.py",
    "src/nba_prop_quant/model.py",
    "src/nba_prop_quant/pricing.py",
    "src/nba_prop_quant/production.py",
    "scripts/15_price_markets.py",
)


def test_no_model_prediction_or_pricing_module_changed():
    """The fix is one slate-construction function and one call site.

    ``game_context.py`` is in this list deliberately: the whole point is that
    the classifier is reused, so it must be byte-identical to production. A
    change there would mean the rules had been rewritten rather than called.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    changed = [
        path
        for path in git("diff", "--name-only", base, "HEAD").splitlines()
        if path in NUMERICAL_MODULES
    ]

    assert changed == [], f"numerical modules changed: {changed}"


def test_no_model_artifact_or_freeze_manifest_changed():
    """No fitted artifact, no freeze manifest, no frozen policy."""
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    offenders = [
        path
        for path in git("diff", "--name-only", base, "HEAD").splitlines()
        if path.endswith(".joblib")
        or path
        in (
            "models/frozen_manifests/LATEST.json",
            "models/frozen_manifests/nba_prop_quant_20260818T205213Z.json",
            "models/frozen_manifests/nba_prop_quant_v2_gate3_runtime_contract.json",
            "models/dynamic_params.json",
            "research/final_model/final_model_spec.json",
            "research/final_model/live_shadow_promotion_policy.json",
        )
    ]

    assert offenders == [], f"frozen state changed: {offenders}"


def test_the_prediction_script_only_gained_the_games_table_lookup():
    """The projection script's change is one import and one argument.

    Compared as a line set against production, so an edited computation would
    show up as a removed line even if the file got longer.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    relative = "scripts/10_predict_slate.py"

    before = git("show", f"{base}:{relative}").splitlines()
    after = git("show", f"HEAD:{relative}").splitlines()

    removed = [line for line in before if line not in after]

    assert removed == [], f"{relative} removed or edited lines: {removed}"

    added = [line for line in after if line not in before]

    assert all(
        "history_games" in line or "load_history_games" in line for line in added
    ), f"{relative} added lines beyond the games-table lookup: {added}"


def test_the_slate_module_only_gained_the_classification_path():
    """Every pre-existing function in ``slate.py`` is unchanged.

    ``build_upcoming_slate_features`` is the one function the fix extends.
    Everything else -- the slate-date semantics, the leakage guard, the prior
    builders -- is compared by syntax tree against production.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    relative = "src/nba_prop_quant/slate.py"

    def functions(source: str) -> dict[str, ast.FunctionDef]:
        return {
            node.name: node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        }

    before = functions(git("show", f"{base}:{relative}"))
    after = functions(git("show", f"HEAD:{relative}"))

    assert set(before) <= set(after), (
        f"{relative} dropped functions: {sorted(set(before) - set(after))}"
    )

    for name, node in before.items():
        if name == "build_upcoming_slate_features":
            continue
        assert ast.dump(node) == ast.dump(after[name]), (
            f"{relative}:{name} is not the function production has"
        )

    # And the one function that may differ only gained statements.
    statements = {
        ast.dump(node) for node in before["build_upcoming_slate_features"].body
    }

    added = [
        node
        for node in after["build_upcoming_slate_features"].body
        if ast.dump(node) not in statements
    ]

    for node in added:
        rendered = ast.unparse(node)
        assert "context" in rendered or "flags" in rendered, (
            "the builder gained a statement that is not the game-context "
            f"plumbing: {rendered[:160]}"
        )
