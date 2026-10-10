"""``season_progress`` means the same thing live as it does in training.

``FEATURE_EXACT`` names ``season_progress``, so the fitted models were shown
one definition of it and must be served that same definition. There were two.

``classify_games`` derives ``home_season_progress`` and
``visitor_season_progress`` as a team's *standings-counting* games before the
game over the season's *inferred* schedule length, and ``build_base_frame``
picks one of the two per row by ``is_home``. That is what the models were fit
on. The live slate builder instead counted a team's distinct box-score games
against a hard-coded 82. Those two agree only while every game a team has
played counts in the standings, and diverge from that team's first Cup
Championship, Play-In or playoff game onwards, because the classifier excludes
exactly those from the numerator and the box-score count does not.

The tests here build the same game twice from the same historical state --
once through ``build_base_frame`` as training does, once through
``build_upcoming_slate_features`` as serving does -- and require one number.
The pre-fix builder is loaded out of Git alongside them, so each case also
records the divergence it closes and pins that ``season_progress`` is the only
feature whose value moved.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nba_prop_quant import slate as slate_module
from nba_prop_quant.features import FEATURE_EXACT, build_base_frame
from nba_prop_quant.game_context import classify_games
from nba_prop_quant.production import assert_model_feature_contract
from nba_prop_quant.slate import (
    GAME_CONTEXT_FLAGS,
    GAME_CONTEXT_PROGRESS,
    SlateGameContextError,
    build_upcoming_slate_features,
    canonical_season_progress,
    classify_upcoming_games,
    upcoming_season_progress,
)

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from season_games_fixture import (  # noqa: E402
    FULL_SCHEDULE_LENGTH,
    pad_season_schedule,
)
from season_parity_fixture import (  # noqa: E402
    AWAY,
    HOME,
    PARITY_TEAMS,
    SEASON,
    active_players,
    box_scores,
    day_after,
    empty_advanced,
    game,
    standings_rounds,
)

PRODUCTION_REF = "production/wizardofodds-integration"

LIFECYCLE_WORKFLOW = PROJECT / ".github" / "workflows" / "nba_production_lifecycle.yml"

SERVING_CALLER = PROJECT / "ops" / "run_incumbent_production_serving.py"

DIAGNOSTIC_OVERRIDE = "--allow-" + "missing-model-features"

SLATE_ID = 9901

#: The standings-counting games the measured regression case puts behind the
#: home team. 40 over an inferred 82 is the 0.487805 the fix has to produce.
MEASURED_STANDINGS_GAMES = 40


# ----------------------------------------------------------------------
# the pre-fix builder, loaded out of Git
# ----------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def blob_text(ref: str, relative: str) -> str:
    """One file's exact bytes at ``ref``, trailing newline and all.

    ``git`` above strips its output, which is right for a revision but turns
    a byte comparison into an almost-byte comparison.
    """
    return subprocess.run(
        ["git", "show", f"{ref}:{relative}"],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def production_base() -> str | None:
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return git("merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue
    return None


SLATE_MODULE = "src/nba_prop_quant/slate.py"

#: The function whose presence in the production base means the fix has landed.
FIX_MARKER_FUNCTION = "canonical_season_progress"

#: The production head this fix branched from, and the one it landed in. Named
#: so the guard below can be checked against both states rather than trusted.
PRE_FIX_PRODUCTION_COMMIT = "536615e58bb7f2f2b61afdac4baffeb817f28644"
POST_FIX_PRODUCTION_COMMIT = "b0e8b490886d0ebb08751fd53bb9dc5e5eb88ee8"


def pre_fix_slate_source() -> str:
    """The slate module as it was before this fix, or skip.

    Everything differential in this file is measured against the production
    merge base, which exists only while the fix has not landed. Once it has,
    the base carries the fix too and "the pre-fix builder" is the post-fix
    builder under another name: a test asserting the two differ is then
    asserting something false about a correct tree rather than catching
    anything.

    The condition is whether the base already defines the fix, not whether
    the file changed. Those differ on the branch that matters -- the next one
    to touch this module for an unrelated reason -- where the file differs
    from a base that already has the fix, and comparing them would measure
    that branch's change against this fix's expectations and fail for a
    reason that is nothing to do with either.

    The absolute tests here, live against training, do not come through this
    and keep running forever. They are the ones worth keeping.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    before = blob_text(base, SLATE_MODULE)

    landed = {
        node.name
        for node in ast.parse(before).body
        if isinstance(node, ast.FunctionDef)
    }

    if FIX_MARKER_FUNCTION in landed:
        pytest.skip(
            "production already carries this fix, so there is no pre-fix "
            "state left to compare against"
        )

    return before


@pytest.fixture(scope="module")
def pre_fix_builder():
    """``build_upcoming_slate_features`` as production has it today.

    Loaded from the merge base rather than reconstructed, so "what the old
    definition produced" is the old code's answer and not a test's idea of it.
    The module is given a name inside the package so its relative imports
    resolve against the real ``nba_prop_quant``: it is the same slate module,
    reading the same features, classifier and priors.
    """
    source = pre_fix_slate_source()

    path = Path(tempfile.mkdtemp()) / "production_slate.py"
    path.write_text(source, encoding="utf-8")

    name = "nba_prop_quant._pre_fix_slate_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module.build_upcoming_slate_features


def test_the_guard_tells_the_pre_fix_state_from_the_post_fix_one():
    """The skip condition is checked against both states, not assumed.

    A guard that silently always skipped would make every differential test
    in this file vacuous, which is the failure mode worth one test of its own.
    The two commits are the production head this fix branched from and the one
    it landed in.
    """
    def defines_the_fix(ref: str) -> bool:
        try:
            source = blob_text(ref, SLATE_MODULE)
        except subprocess.CalledProcessError:
            pytest.skip(f"{ref} is not available in this checkout")

        return FIX_MARKER_FUNCTION in {
            node.name
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        }

    assert not defines_the_fix(PRE_FIX_PRODUCTION_COMMIT)
    assert defines_the_fix(POST_FIX_PRODUCTION_COMMIT)


@pytest.fixture(scope="module")
def dynamic_params() -> dict:
    return json.loads(
        (PROJECT / "models" / "dynamic_params.json").read_text(encoding="utf-8")
    )


# ----------------------------------------------------------------------
# building one game both ways
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A slate game, the games played before it, and the season's schedule."""

    slate: dict
    played: list[dict]
    schedule: pd.DataFrame

    @property
    def teams(self) -> tuple[int, int]:
        return self.slate["home_team_id"], self.slate["visitor_team_id"]


def scenario(slate: dict, played: list[dict], *, future: list[dict] = ()) -> Scenario:
    """Pad the fixture to a real schedule and hand back the three inputs.

    ``games.parquet`` is the season's whole published schedule, so the
    schedule covers the played games, the slate and whatever else the season
    holds. Without the padding the modal appearance count is the fixture's own
    size and every inferred denominator is wrong.
    """
    return Scenario(
        slate=slate,
        played=list(played),
        schedule=pad_season_schedule(
            pd.DataFrame([*played, slate, *future]),
            schedule_length=FULL_SCHEDULE_LENGTH,
        ),
    )


def build_live(case: Scenario, dynamic_params: dict, builder=None) -> pd.DataFrame:
    """The slate as serving builds it: history behind, the game ahead."""
    builder = builder or build_upcoming_slate_features

    return builder(
        history_stats=box_scores(case.played),
        upcoming_games=pd.DataFrame([case.slate]),
        active_players=active_players(*case.teams),
        advanced=empty_advanced(),
        dynamic_params=dynamic_params,
        history_games=case.schedule,
    )


def build_training(case: Scenario) -> pd.DataFrame:
    """The same game as training builds it, once it has been played."""
    trained = build_base_frame(
        box_scores([*case.played, case.slate]),
        games=case.schedule,
    )

    rows = trained[trained["game_id"].eq(case.slate["id"])]

    assert not rows.empty, "the training frame does not hold the game under test"

    return rows


def feature_by_side(frame: pd.DataFrame, case: Scenario, name: str) -> dict[int, float]:
    """One value of ``name`` per side of the game, keyed by ``is_home``."""
    out: dict[int, float] = {}

    for is_home, team_id in ((1, case.teams[0]), (0, case.teams[1])):
        side = frame[frame["team_id"].eq(team_id)]

        assert not side.empty, f"no rows for team {team_id}"

        values = {float(value) for value in side[name]}

        assert len(values) == 1, (
            f"{name} differs within one side of the game: {sorted(values)}"
        )

        out[is_home] = values.pop()

    return out


def assert_parity(case: Scenario, dynamic_params: dict) -> dict[int, float]:
    """The same game, built both ways, agrees on ``season_progress``."""
    live = feature_by_side(
        build_live(case, dynamic_params), case, "season_progress"
    )

    trained = feature_by_side(build_training(case), case, "season_progress")

    assert live == trained, (
        f"live {live} and training {trained} disagree on season_progress"
    )

    return live


# ----------------------------------------------------------------------
# the five historical states
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def ordinary_case() -> Scenario:
    """Mid-season, every game played so far counting in the standings."""
    played = standings_rounds(12)

    return scenario(game(SLATE_ID, day_after(played)), played)


@pytest.fixture(scope="module")
def cup_championship_case() -> Scenario:
    """The measured case: 40 standings games plus a Cup Championship.

    The Cup final is a game the team played and the standings do not count, so
    the classifier's numerator stays at 40 while a box-score count reaches 41.
    """
    played = standings_rounds(MEASURED_STANDINGS_GAMES)

    final = game(SLATE_ID - 1, day_after(played), ist_stage="Championship")

    return scenario(
        game(SLATE_ID, day_after(played, days=2)),
        [*played, final],
    )


@pytest.fixture(scope="module")
def play_in_case() -> Scenario:
    """A completed schedule plus a Play-In game, with a playoff game ahead.

    A Play-In game is the other thing a team plays that the standings do not
    count, and the classifier recognises it structurally rather than from a
    label: the home team has already completed the inferred schedule length.
    """
    played = standings_rounds(FULL_SCHEDULE_LENGTH)

    play_in = game(SLATE_ID - 1, day_after(played))

    return scenario(
        game(SLATE_ID, day_after(played, days=2), postseason=True),
        [*played, play_in],
    )


@pytest.fixture(scope="module")
def playoff_case() -> Scenario:
    """Deeper in: a Play-In game and a playoff game already behind the team."""
    played = standings_rounds(FULL_SCHEDULE_LENGTH)

    play_in = game(SLATE_ID - 2, day_after(played))

    first_round = game(
        SLATE_ID - 1, day_after(played, days=3), postseason=True
    )

    return scenario(
        game(SLATE_ID, day_after(played, days=5), postseason=True),
        [*played, play_in, first_round],
    )


@pytest.fixture(scope="module")
def multiple_season_case() -> Scenario:
    """Two seasons in one games table, the slate in the later one.

    Production's games table holds whatever seasons have been ingested, and
    the classifier walks each one against its own counter. A fixture with only
    one season cannot show that the earlier one neither shifts the inferred
    denominator nor leaks into the count.
    """
    earlier = standings_rounds(
        FULL_SCHEDULE_LENGTH,
        start="2025-10-21",
        season=SEASON - 1,
        first_id=300_000,
    )

    later = standings_rounds(MEASURED_STANDINGS_GAMES)

    final = game(SLATE_ID - 1, day_after(later), ist_stage="Championship")

    return scenario(
        game(SLATE_ID, day_after(later, days=2)),
        [*earlier, *later, final],
    )


# ----------------------------------------------------------------------
# 1-5: live equals training in every historical state
# ----------------------------------------------------------------------


def test_an_ordinary_regular_season_state_agrees(ordinary_case, dynamic_params):
    """State 1: nothing played that the standings do not count.

    This is the case the old definition got right, and it is here because a
    fix that changes it would be changing a feature value for no reason.
    """
    progress = assert_parity(ordinary_case, dynamic_params)

    assert progress == {1: 12 / 82, 0: 12 / 82}


def test_a_cup_championship_in_the_history_agrees(
    cup_championship_case, dynamic_params
):
    """State 2: the Cup final the standings do not count."""
    progress = assert_parity(cup_championship_case, dynamic_params)

    assert progress == {
        1: MEASURED_STANDINGS_GAMES / FULL_SCHEDULE_LENGTH,
        0: MEASURED_STANDINGS_GAMES / FULL_SCHEDULE_LENGTH,
    }


def test_a_play_in_game_in_the_history_agrees(play_in_case, dynamic_params):
    """State 3: a completed schedule and a Play-In game behind the team."""
    progress = assert_parity(play_in_case, dynamic_params)

    assert progress == {1: 1.0, 0: 1.0}


def test_playoff_history_agrees(playoff_case, dynamic_params):
    """State 4: playoff games behind the team as well."""
    progress = assert_parity(playoff_case, dynamic_params)

    assert progress == {1: 1.0, 0: 1.0}


def test_multiple_seasons_in_the_games_table_agree(
    multiple_season_case, dynamic_params
):
    """State 5: an earlier season present changes neither side's answer."""
    progress = assert_parity(multiple_season_case, dynamic_params)

    assert progress == {
        1: MEASURED_STANDINGS_GAMES / FULL_SCHEDULE_LENGTH,
        0: MEASURED_STANDINGS_GAMES / FULL_SCHEDULE_LENGTH,
    }


def test_the_earlier_season_does_not_change_the_later_seasons_answer(
    multiple_season_case, cup_championship_case, dynamic_params
):
    """The two-season fixture answers exactly what the one-season one does.

    The live path narrows the games table to the slate's seasons before
    classifying. That is exact rather than approximate only because the
    classifier is season-partitioned, and this is the statement of it.
    """
    assert feature_by_side(
        build_live(multiple_season_case, dynamic_params),
        multiple_season_case,
        "season_progress",
    ) == feature_by_side(
        build_live(cup_championship_case, dynamic_params),
        cup_championship_case,
        "season_progress",
    )


# ----------------------------------------------------------------------
# 6: the exact measured regression case
# ----------------------------------------------------------------------


def test_the_measured_divergence_is_closed(
    cup_championship_case, dynamic_params, pre_fix_builder
):
    """training 0.487805 / live 0.500000, measured and then reconciled.

    This is the case that was reported: a team 40 standings games into the
    season that has also played the Cup Championship. ``40 / 82`` is what
    training says. ``41 / 82`` is what counting box scores said. The
    assertions below are the two numbers and the gap, from the real pre-fix
    code rather than from arithmetic repeated in a test.
    """
    expected = MEASURED_STANDINGS_GAMES / FULL_SCHEDULE_LENGTH

    assert expected == pytest.approx(0.487805, abs=5e-7)

    before = feature_by_side(
        build_live(cup_championship_case, dynamic_params, pre_fix_builder),
        cup_championship_case,
        "season_progress",
    )

    after = feature_by_side(
        build_live(cup_championship_case, dynamic_params),
        cup_championship_case,
        "season_progress",
    )

    trained = feature_by_side(
        build_training(cup_championship_case),
        cup_championship_case,
        "season_progress",
    )

    assert before == {1: 0.5, 0: 0.5}
    assert trained == {1: expected, 0: expected}
    assert after == trained

    assert before[1] - trained[1] == pytest.approx(0.012195, abs=5e-7)


def test_every_state_that_diverged_is_now_identical(
    ordinary_case,
    cup_championship_case,
    play_in_case,
    playoff_case,
    multiple_season_case,
    dynamic_params,
    pre_fix_builder,
):
    """The gap the fix closes, state by state, and that it is now zero.

    Recorded rather than only asserted away: the ordinary state never
    diverged, and the three states with a standings-exempt game behind the
    team all did, which is the shape the root cause predicts.
    """
    diverged = {}

    for name, case in (
        ("ordinary", ordinary_case),
        ("cup_championship", cup_championship_case),
        ("play_in", play_in_case),
        ("playoff", playoff_case),
        ("multiple_season", multiple_season_case),
    ):
        trained = feature_by_side(build_training(case), case, "season_progress")

        before = feature_by_side(
            build_live(case, dynamic_params, pre_fix_builder),
            case,
            "season_progress",
        )

        after = feature_by_side(
            build_live(case, dynamic_params), case, "season_progress"
        )

        assert after == trained, name

        diverged[name] = before != trained

    assert diverged == {
        "ordinary": False,
        "cup_championship": True,
        "play_in": True,
        "playoff": True,
        "multiple_season": True,
    }


# ----------------------------------------------------------------------
# 7: the feature contract passes
# ----------------------------------------------------------------------


class FakeBundle:
    """The minimal shape ``assert_model_feature_contract`` reads."""

    def __init__(self, features: list[str]) -> None:
        self.feature_names = features


def test_the_feature_contract_passes_over_the_whole_exact_feature_list(
    cup_championship_case, dynamic_params
):
    """``FEATURE_EXACT`` is satisfied by the slate, with nothing waived.

    ``allow_missing`` is left at its default, so the call raises unless every
    exact feature is present. Asserted over the whole list rather than over
    ``season_progress`` alone, because a fix that produced one feature by
    dropping another would still pass the narrower check.
    """
    frame = build_live(cup_championship_case, dynamic_params)

    missing = assert_model_feature_contract(
        frame,
        FakeBundle(list(FEATURE_EXACT)),
        label="exact feature list",
    )

    assert missing == []

    assert "season_progress" in FEATURE_EXACT
    assert frame["season_progress"].notna().all()


# ----------------------------------------------------------------------
# 8: the five game-context flags are untouched
# ----------------------------------------------------------------------


def test_the_five_game_context_flags_still_match_the_classifier(
    ordinary_case,
    cup_championship_case,
    play_in_case,
    playoff_case,
    multiple_season_case,
    dynamic_params,
):
    """The previous fix still holds in every state this one exercises."""
    for case in (
        ordinary_case,
        cup_championship_case,
        play_in_case,
        playoff_case,
        multiple_season_case,
    ):
        live = build_live(case, dynamic_params)
        trained = build_training(case)

        for flag in GAME_CONTEXT_FLAGS:
            live_values = {int(value) for value in live[flag]}
            training_values = {int(value) for value in trained[flag]}

            assert len(live_values) == 1
            assert live_values == training_values, flag


def test_the_flags_the_pre_fix_builder_produced_are_unchanged(
    cup_championship_case, play_in_case, dynamic_params, pre_fix_builder
):
    """This change does not move a flag the last one established."""
    for case in (cup_championship_case, play_in_case):
        before = build_live(case, dynamic_params, pre_fix_builder)
        after = build_live(case, dynamic_params)

        for flag in GAME_CONTEXT_FLAGS:
            assert before[flag].equals(after[flag]), flag


# ----------------------------------------------------------------------
# 9: no diagnostic override
# ----------------------------------------------------------------------


def test_the_production_lifecycle_passes_no_diagnostic_override():
    """The override is absent from the workflow's text and from its steps."""
    text = LIFECYCLE_WORKFLOW.read_text(encoding="utf-8")

    assert DIAGNOSTIC_OVERRIDE not in text

    workflow = yaml.safe_load(text)

    rendered = json.dumps(workflow)

    assert DIAGNOSTIC_OVERRIDE not in rendered


def test_the_serving_caller_passes_no_diagnostic_override():
    """The incumbent serving entry point waives nothing either."""
    assert DIAGNOSTIC_OVERRIDE not in SERVING_CALLER.read_text(encoding="utf-8")


def test_nothing_this_change_touches_introduces_an_override():
    """Neither the slate module nor these tests reach for the override."""
    for path in (
        PROJECT / "src" / "nba_prop_quant" / "slate.py",
        Path(__file__),
        PROJECT / "tests" / "season_parity_fixture.py",
    ):
        text = path.read_text(encoding="utf-8")

        assert text.count(DIAGNOSTIC_OVERRIDE) == 0, path
        assert ("allow_missing" + "=True") not in text, path


# ----------------------------------------------------------------------
# 10: nothing else moved
# ----------------------------------------------------------------------


def test_season_progress_is_the_only_feature_whose_value_changed(
    ordinary_case,
    cup_championship_case,
    play_in_case,
    playoff_case,
    multiple_season_case,
    dynamic_params,
    pre_fix_builder,
):
    """Every other column of the slate frame is what production produced.

    The whole frame is compared against the pre-fix builder's, column by
    column, over every state above. This is the test that makes "no other
    feature changes" a measurement rather than an intention: a prior, a
    rolling window, an experience curve or a team-context value that moved
    would show up here by name.

    ``season_progress`` itself is allowed to move and not required to: the
    ordinary state is one the old definition already got right. Which states
    move is pinned separately, so neither question is answered loosely here.
    """
    for case in (
        ordinary_case,
        cup_championship_case,
        play_in_case,
        playoff_case,
        multiple_season_case,
    ):
        before = build_live(case, dynamic_params, pre_fix_builder)
        after = build_live(case, dynamic_params)

        assert list(before.columns) == list(after.columns)
        assert len(before) == len(after)

        moved = [
            column
            for column in before.columns
            if not before[column].equals(after[column])
        ]

        assert moved in ([], ["season_progress"]), moved


def disagreeing_exact_features(
    live: pd.DataFrame, trained: pd.DataFrame
) -> dict[str, tuple[float, float]]:
    """Exact features whose value differs between the two frames."""
    out: dict[str, tuple[float, float]] = {}

    shared = sorted(set(live["player_id"]) & set(trained["player_id"]))

    for name in FEATURE_EXACT:
        for player_id in shared:
            a = live[live["player_id"].eq(player_id)][name]
            b = trained[trained["player_id"].eq(player_id)][name]

            if a.empty or b.empty:
                continue

            first, second = float(a.iloc[0]), float(b.iloc[0])

            if pd.isna(first) and pd.isna(second):
                continue

            if pd.isna(first) != pd.isna(second) or abs(first - second) > 1e-9:
                out[name] = (first, second)
                break

    return out


def test_no_exact_feature_disagrees_between_live_and_training(
    ordinary_case,
    cup_championship_case,
    play_in_case,
    playoff_case,
    multiple_season_case,
    dynamic_params,
):
    """Every name in ``FEATURE_EXACT``, not just the one that was reported.

    ``season_progress`` was found because somebody measured it. The claim this
    fix closes the live/training feature gap is only worth making if the whole
    exact feature list has been checked the same way, so it is: every name,
    every player, every state above, live against training.
    """
    for name, case in (
        ("ordinary", ordinary_case),
        ("cup_championship", cup_championship_case),
        ("play_in", play_in_case),
        ("playoff", playoff_case),
        ("multiple_season", multiple_season_case),
    ):
        live = build_live(case, dynamic_params)
        trained = build_training(case)

        assert set(FEATURE_EXACT) <= set(live.columns), name
        assert set(FEATURE_EXACT) <= set(trained.columns), name

        assert disagreeing_exact_features(live, trained) == {}, name


def test_the_reported_feature_was_the_only_one_that_disagreed(
    cup_championship_case, dynamic_params, pre_fix_builder
):
    """Before the fix, exactly one exact feature disagreed. Now none does.

    The complement of the test above: it establishes that the sweep is capable
    of finding a divergence, by finding the one this change removes.
    """
    trained = build_training(cup_championship_case)

    before = disagreeing_exact_features(
        build_live(cup_championship_case, dynamic_params, pre_fix_builder), trained
    )

    after = disagreeing_exact_features(
        build_live(cup_championship_case, dynamic_params), trained
    )

    assert sorted(before) == ["season_progress"]
    assert after == {}


def test_the_slate_frame_still_carries_every_column_production_produced(
    cup_championship_case, dynamic_params, pre_fix_builder
):
    """No column is added or dropped, so no model feature appears or vanishes."""
    before = build_live(cup_championship_case, dynamic_params, pre_fix_builder)
    after = build_live(cup_championship_case, dynamic_params)

    assert set(before.columns) == set(after.columns)


# ----------------------------------------------------------------------
# the live path reads the classifier rather than restating it
# ----------------------------------------------------------------------


def test_the_two_live_classification_paths_agree(
    ordinary_case,
    cup_championship_case,
    play_in_case,
    playoff_case,
    multiple_season_case,
):
    """``upcoming_season_progress`` classifies what the flag path classifies.

    The two functions marshal their inputs separately -- the serving source
    contract pins the flag path syntax-tree-identical to production, so it
    could not be refactored to share the marshalling. This is the guard
    against the two copies drifting: both are run over the same slate and
    required to produce the same classification.
    """
    for case in (
        ordinary_case,
        cup_championship_case,
        play_in_case,
        playoff_case,
        multiple_season_case,
    ):
        upcoming = pd.DataFrame([case.slate])

        flags = classify_upcoming_games(upcoming, case.schedule)
        progress = upcoming_season_progress(upcoming, case.schedule)

        assert list(flags["game_id"]) == list(progress["game_id"])

        combined = classify_games(case.schedule)
        row = combined[combined["id"].eq(case.slate["id"])].iloc[0]

        for name in GAME_CONTEXT_FLAGS:
            assert int(flags[name].iloc[0]) == int(row[name]), name

        for name in GAME_CONTEXT_PROGRESS:
            assert float(progress[name].iloc[0]) == pytest.approx(
                float(row[name])
            ), name


def test_the_progress_columns_are_the_classifiers_own(
    cup_championship_case, dynamic_params
):
    """The live value is one of the classifier's two columns, picked by side."""
    progress = upcoming_season_progress(
        pd.DataFrame([cup_championship_case.slate]), cup_championship_case.schedule
    )

    record = progress.to_dict("records")[0]

    live = feature_by_side(
        build_live(cup_championship_case, dynamic_params),
        cup_championship_case,
        "season_progress",
    )

    assert live[1] == pytest.approx(float(record["home_season_progress"]))
    assert live[0] == pytest.approx(float(record["visitor_season_progress"]))


def test_the_progress_names_are_the_classifiers_output_columns():
    """The tuple is read off ``classify_games``, not maintained in parallel."""
    classified = classify_games(
        pd.DataFrame(
            [
                game(1, "2026-10-21"),
                game(2, "2026-10-23"),
            ]
        )
    )

    for name in GAME_CONTEXT_PROGRESS:
        assert name in classified.columns

    assert set(GAME_CONTEXT_PROGRESS) == {
        name for name in classified.columns if name.endswith("_season_progress")
    }


def test_no_season_progress_denominator_is_written_into_the_builder():
    """The builder states no schedule length of its own.

    ``canonical_season_progress`` keeps training's ``/ 82`` fallback, which is
    training's own and is reached only when a season has no inferable schedule
    length. What must not exist is a denominator in the builder deciding the
    answer, so the builder's statements are checked for one.
    """
    source = (PROJECT / "src" / "nba_prop_quant" / "slate.py").read_text(
        encoding="utf-8"
    )

    builder = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_upcoming_slate_features"
    )

    assignments = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "season_progress"
            for target in node.targets
        )
    ]

    assert assignments, "the builder no longer assigns season_progress"

    final = max(assignments, key=lambda node: node.lineno)

    assert isinstance(final.value, ast.Call)
    assert isinstance(final.value.func, ast.Name)
    assert final.value.func.id == "canonical_season_progress"

    # The earlier assignment survives as training's fallback and is passed in
    # as one, so it must not be what the row ends up carrying.
    assert len(assignments) == 2


# ----------------------------------------------------------------------
# the canonical reproduction, in isolation
# ----------------------------------------------------------------------


def test_the_side_is_picked_the_way_training_picks_it():
    """``is_home`` selects the home column, anything else the visitor one."""
    record = {"home_season_progress": 0.25, "visitor_season_progress": 0.75}

    assert canonical_season_progress(record, is_home=1, fallback=0.0) == 0.25
    assert canonical_season_progress(record, is_home=0, fallback=0.0) == 0.75


def test_the_fallback_is_only_reached_when_the_classifier_has_no_answer():
    """Training's fallback, kept because training keeps it."""
    answered = {"home_season_progress": 0.25, "visitor_season_progress": 0.75}

    assert canonical_season_progress(answered, is_home=1, fallback=0.9) == 0.25

    unanswered = {
        "home_season_progress": float("nan"),
        "visitor_season_progress": float("nan"),
    }

    assert canonical_season_progress(unanswered, is_home=1, fallback=0.9) == 0.9


def test_the_result_is_clipped_the_way_training_clips_it():
    """``[0, 1]``, which is tighter than the 1.5 ceiling the live path had."""
    record = {"home_season_progress": 1.4, "visitor_season_progress": -0.2}

    assert canonical_season_progress(record, is_home=1, fallback=0.0) == 1.0
    assert canonical_season_progress(record, is_home=0, fallback=0.0) == 0.0

    assert canonical_season_progress(
        {"home_season_progress": float("nan")}, is_home=1, fallback=1.4
    ) == 1.0


def test_the_training_definition_is_the_one_being_reproduced():
    """``build_base_frame`` picks by ``is_home``, coalesces, clips ``[0, 1]``.

    Read off the training module rather than asserted from memory, so a change
    to the definition this fix reproduces fails here instead of silently
    leaving the live path behind again.
    """
    source = (PROJECT / "src" / "nba_prop_quant" / "features.py").read_text(
        encoding="utf-8"
    )

    base = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "build_base_frame"
    )

    rendered = ast.unparse(base)

    assert "df['is_home'].eq(1)" in rendered
    assert "df['home_season_progress']" in rendered
    assert "df['visitor_season_progress']" in rendered
    assert "fillna(fallback_progress)" in rendered
    assert ".clip(0.0, 1.0)" in rendered


# ----------------------------------------------------------------------
# refusals: the inputs the canonical definition needs
# ----------------------------------------------------------------------


def test_season_progress_refuses_a_slate_with_no_games_table(cup_championship_case):
    """No games table means no inferable denominator, so it refuses."""
    with pytest.raises(SlateGameContextError, match="games table"):
        upcoming_season_progress(
            pd.DataFrame([cup_championship_case.slate]),
            pd.DataFrame(columns=list(cup_championship_case.schedule.columns)),
        )


def test_season_progress_refuses_a_games_table_without_the_slates_season(
    cup_championship_case,
):
    """A table holding only other seasons is refused rather than guessed at."""
    other = cup_championship_case.schedule.copy()
    other["season"] = SEASON - 3

    with pytest.raises(SlateGameContextError, match="no season"):
        upcoming_season_progress(pd.DataFrame([cup_championship_case.slate]), other)


def test_season_progress_refuses_a_games_table_missing_a_required_column(
    cup_championship_case,
):
    """The classifier's inputs are checked, not defaulted."""
    with pytest.raises(SlateGameContextError, match="missing column"):
        upcoming_season_progress(
            pd.DataFrame([cup_championship_case.slate]),
            cup_championship_case.schedule.drop(columns=["postseason"]),
        )


def test_an_empty_slate_has_no_progress_rows(cup_championship_case):
    """An empty slate is empty, not an error and not a fabricated row."""
    empty = upcoming_season_progress(
        pd.DataFrame(columns=list(cup_championship_case.schedule.columns)),
        cup_championship_case.schedule,
    )

    assert empty.empty
    assert list(empty.columns) == ["game_id", *GAME_CONTEXT_PROGRESS]


def test_the_builder_refuses_rather_than_serving_an_unclassifiable_slate(
    cup_championship_case, dynamic_params
):
    """A slate whose season is absent is refused before anything is scored."""
    other = cup_championship_case.schedule.copy()
    other["season"] = SEASON - 3

    with pytest.raises(SlateGameContextError):
        build_upcoming_slate_features(
            history_stats=box_scores(cup_championship_case.played),
            upcoming_games=pd.DataFrame([cup_championship_case.slate]),
            active_players=active_players(*cup_championship_case.teams),
            advanced=empty_advanced(),
            dynamic_params=dynamic_params,
            history_games=other,
        )


# ----------------------------------------------------------------------
# nothing in the model, the training definition or pricing moved
# ----------------------------------------------------------------------

#: Every module that decides a projection, a probability or a price. The fix is
#: a feature-construction change in the slate builder and must not reach these.
NUMERICAL_MODULES: tuple[str, ...] = (
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
    "src/nba_prop_quant/prospective_snapshot.py",
)


def test_no_model_probability_or_pricing_module_changed():
    """The twelve numerical modules are byte-identical to production.

    ``features.py`` and ``game_context.py`` are in this list deliberately:
    this fix makes the live path match the training definition, so the
    training definition itself must not move. A fix that edited either would
    be redefining the feature rather than reproducing it.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    for relative in NUMERICAL_MODULES:
        assert blob_text(base, relative) == (
            PROJECT / relative
        ).read_text(encoding="utf-8"), relative


def test_no_model_artifact_or_freeze_manifest_changed():
    """No fitted binary, frozen config or freeze manifest is in the diff."""
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    # Against the working tree rather than HEAD, and with untracked files
    # included, so an artifact that has not been committed yet is still seen.
    changed = [
        line
        for line in (
            git("diff", "--name-only", base).splitlines()
            + git("ls-files", "--others", "--exclude-standard").splitlines()
        )
        if line
    ]

    offenders = [
        path
        for path in changed
        if path.endswith((".joblib", ".pkl", ".npy"))
        or path.startswith(("configs/", "data/", "research/"))
        or (
            path.startswith("models/")
            and "adaptive_serving_source_contract" not in path
            and "adaptive_update_protocol" not in path
        )
    ]

    assert offenders == [], f"model surface changed: {offenders}"


def test_the_slate_module_only_gained_the_season_progress_path():
    """Every other function in the slate module is production's own.

    The slate module holds the leakage guards, the slate-date rules, the
    rolling priors and the team-context lookups as well as the builder. Only
    the builder may differ, and the two functions this change adds are new
    rather than edits.
    """
    source = pre_fix_slate_source()

    def functions(text: str) -> dict[str, ast.FunctionDef]:
        return {
            node.name: node
            for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef)
        }

    before = functions(source)
    after = functions((PROJECT / SLATE_MODULE).read_text(encoding="utf-8"))

    assert set(before) <= set(after)

    assert set(after) - set(before) == {
        "upcoming_season_progress",
        "canonical_season_progress",
    }

    for name, node in before.items():
        if name == "build_upcoming_slate_features":
            continue

        assert ast.dump(node) == ast.dump(after[name]), name

    assert slate_module.build_upcoming_slate_features is not None
