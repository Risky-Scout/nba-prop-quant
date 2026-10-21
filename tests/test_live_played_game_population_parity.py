"""Live priors are built over the player-games training built them over.

``build_base_frame`` opens with ``df[df["minutes"].fillna(0) > 0]``. Every
rolling prior, decay state and Kalman state the fitted models name is defined
over what survives that line, because that line is the first thing training
does. BALLDONTLIE's stats endpoint returns a row per rostered player, so a
player who did not play arrives carrying zero or null minutes, and the live
slate builder read the table whole.

So a missed game entered every per-minute EWM as a zero and the minutes EWM as
a zero. A feature the models were fit on over appearances was served
averaged over appearances *and* absences, and because every target here is a
count, the error has one direction: down. It falls hardest on the players
whose most recent rows are absences, which on opening night is every player
whose previous season ended in an injury -- and it leaves a player who never
misses a game untouched, which is why the resulting slate reads as a model
that likes the under rather than as a model that is broken.

The advanced priors had the same defect from the same cause.
``build_base_frame`` restricts the advanced history by an inner merge on the
played keys and says why in place; the live builder read it unrestricted, so
each player's usage, true-shooting and pace EWM ran over rows training had
excluded.

The tests here build the same game twice from the same historical state --
once through ``build_base_frame`` as training does, once through
``build_upcoming_slate_features`` as serving does -- and require one number
for every feature in the fitted feature list rather than for the handful
somebody thought to check. The pre-fix builder is loaded out of Git alongside
them, so the deflation each case closes is recorded as a measurement.
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

from nba_prop_quant.features import (
    add_dynamic_priors,
    build_base_frame,
    feature_columns,
)
from nba_prop_quant.slate import (
    HistoryLeakageError,
    build_upcoming_slate_features,
    played_advanced_rows,
    played_player_games,
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
    SEASON,
    active_players,
    box_scores,
    day_after,
    empty_advanced,
    game,
    standings_rounds,
)

PRODUCTION_REF = "production/wizardofodds-integration"

SLATE_MODULE = "src/nba_prop_quant/slate.py"

#: The function whose presence in the production base means this fix landed.
FIX_MARKER_FUNCTION = "played_player_games"

SLATE_ID = 9801

#: The rounds of ordinary season behind every case below. Enough that an EWM
#: with a span of ten has reached its steady state, so a deflation measured
#: here is the deflation the window produces rather than a startup transient.
ROUNDS_BEHIND_THE_SLATE = 20

#: How many of those rounds the returning player misses. Four is the shape the
#: opening-night slate had -- a star whose last few appearances before the
#: slate are absences -- and is deliberately a minority of the history, so the
#: deflation cannot be explained as the player simply having no history.
ROUNDS_MISSED = 4

#: Rows BALLDONTLIE returns for a player who did not play. ``minutes`` is the
#: column every feature divides by and the one ``build_base_frame`` filters
#: on; the rest of the line is zeros, which is what makes the contamination
#: one-directional rather than merely noisy.
DNP_LINE: dict[str, object] = {
    "min": "0:00",
    "minutes": 0.0,
    "pts": 0.0,
    "reb": 0.0,
    "ast": 0.0,
    "stl": 0.0,
    "blk": 0.0,
    "fg3m": 0.0,
    "fga": 0.0,
    "fg3a": 0.0,
    "fta": 0.0,
    "oreb": 0.0,
    "dreb": 0.0,
    "turnover": 0.0,
    "pf": 0.0,
}

#: The other spelling. A BDL row for an inactive player can carry a null
#: minutes value rather than ``"0:00"``, and ``minutes_to_float`` maps that to
#: NaN -- which is why training's filter is ``fillna(0) > 0`` and not ``> 0``.
NULL_MINUTES_DNP_LINE: dict[str, object] = {
    **DNP_LINE,
    "min": None,
    "minutes": float("nan"),
}

#: The player who misses games, and the one who plays every one of them. Both
#: are on the home team and produce the identical line whenever they play, so
#: any difference between their served features is the absences and nothing
#: else.
ABSENTEE = HOME * 10 + 1
IRON_MAN = HOME * 10 + 2

#: Advanced values held constant for the same reason the box-score line is.
ADVANCED_LINE: dict[str, float] = {
    "usage_percentage": 0.28,
    "estimated_usage_percentage": 0.27,
    "assist_percentage": 0.18,
    "rebound_percentage": 0.12,
    "true_shooting_percentage": 0.58,
    "effective_field_goal_percentage": 0.54,
    "pace": 99.0,
    "estimated_pace": 98.5,
    "possessions": 68.0,
    "touches": 55.0,
    "passes": 40.0,
    "secondary_assists": 1.0,
    "rebound_chances_total": 8.0,
    "deflections": 2.0,
    "contested_shots": 6.0,
    "defended_at_rim_fga": 3.0,
    "defended_at_rim_fg_pct": 0.5,
    "pct_fga_3pt": 0.4,
    "pct_3pa": 0.4,
    "free_throw_attempt_rate": 0.2,
    "fouls_drawn": 2.0,
    "speed": 4.2,
    "distance": 2.4,
    "turnover_ratio": 9.0,
}

#: What BDL reports for the same fields on a row the player did not play. All
#: zero, which is the whole point: averaged in, they halve a usage prior.
ADVANCED_DNP_LINE: dict[str, float] = {name: 0.0 for name in ADVANCED_LINE}


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


def pre_fix_slate_source() -> str:
    """The slate module as production has it, or skip.

    The differential cases here are measured against the production merge
    base, which carries the pre-fix builder only while this fix has not
    landed. Once it has, "the pre-fix builder" is this builder under another
    name and a test asserting the two differ is asserting something false
    about a correct tree. The condition is whether the base already defines
    the fix rather than whether the file changed, so the next branch to touch
    this module for an unrelated reason does not have its own change measured
    against this fix's expectations.

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
    population produced" is the old code's answer and not a test's idea of it.
    """
    source = pre_fix_slate_source()

    path = Path(tempfile.mkdtemp()) / "production_slate.py"
    path.write_text(source, encoding="utf-8")

    name = "nba_prop_quant._pre_fix_population_slate_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module.build_upcoming_slate_features


def test_the_fix_marker_names_a_function_the_module_really_defines():
    """The skip condition is anchored to something real, not assumed.

    A marker naming nothing would make every differential test in this file
    skip forever, which is the failure mode worth one test of its own. It
    deliberately does not also assert that the merge base lacks the marker:
    once this fix lands the base carries it, and that is exactly when the
    differential tests are meant to stop running.
    """
    assert FIX_MARKER_FUNCTION in {
        node.name
        for node in ast.parse(
            (PROJECT / SLATE_MODULE).read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef)
    }


# ----------------------------------------------------------------------
# building one game both ways, from a history that holds absences
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def dynamic_params() -> dict:
    return json.loads(
        (PROJECT / "models" / "dynamic_params.json").read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class Scenario:
    """A slate game, the games behind it, and who was absent for which."""

    slate: dict
    played: list[dict]
    schedule: pd.DataFrame
    missed: frozenset[int]
    dnp_line: dict

    @property
    def teams(self) -> tuple[int, int]:
        return self.slate["home_team_id"], self.slate["visitor_team_id"]

    def box_scores(self, games: list[dict]) -> pd.DataFrame:
        """``games``' box scores, with the absentee's missed rows as DNP rows."""
        return with_dnp_rows(
            box_scores(games), ABSENTEE, self.missed, self.dnp_line
        )

    @property
    def history(self) -> pd.DataFrame:
        """What live serving has: the games behind the slate."""
        return self.box_scores(self.played)

    @property
    def trained_history(self) -> pd.DataFrame:
        """What training has: those games plus the slate, once it is played."""
        return self.box_scores([*self.played, self.slate])


def with_dnp_rows(
    frame: pd.DataFrame,
    player_id: int,
    missed_game_ids: frozenset[int],
    dnp_line: dict,
) -> pd.DataFrame:
    """Replace one player's rows for ``missed_game_ids`` with DNP rows.

    The rows are replaced rather than dropped, because that is what the
    production stats tree holds: BDL reports the player as rostered and not
    playing, so the row exists and carries zeros.
    """
    out = frame.copy()

    rows = out.index[
        out["player_id"].eq(player_id) & out["game_id"].isin(missed_game_ids)
    ]

    assert len(rows) == len(missed_game_ids), (
        "the fixture does not hold a row for every missed game"
    )

    for column, value in dnp_line.items():
        out.loc[rows, column] = value

    return out


def home_games(games: list[dict]) -> list[dict]:
    """The games in ``games`` the home team appears in."""
    return [
        entry
        for entry in games
        if HOME in (entry["home_team_id"], entry["visitor_team_id"])
    ]


def scenario(
    *,
    rounds: int = ROUNDS_BEHIND_THE_SLATE,
    missed_rounds: int = ROUNDS_MISSED,
    from_the_start: bool = False,
    dnp_line: dict = DNP_LINE,
) -> Scenario:
    """A season of ordinary rounds with the absentee missing some of them.

    ``from_the_start`` puts the absences at the beginning of the history
    instead of the end, which separates "the EWM saw zeros" from "the EWM saw
    zeros recently": the span-ten window has largely forgotten the opening
    rounds by the slate, so that case is the control for recency.
    """
    played = standings_rounds(rounds)

    appearances = home_games(played)

    missed = (
        []
        if missed_rounds == 0
        else appearances[:missed_rounds]
        if from_the_start
        else appearances[-missed_rounds:]
    )

    slate = game(
        SLATE_ID, day_after(played), home_team_id=HOME, visitor_team_id=AWAY
    )

    return Scenario(
        slate=slate,
        played=played,
        schedule=pad_season_schedule(
            pd.DataFrame([*played, slate]),
            schedule_length=FULL_SCHEDULE_LENGTH,
        ),
        missed=frozenset(int(entry["id"]) for entry in missed),
        dnp_line=dnp_line,
    )


def advanced_rows(case: Scenario, games: list[dict]) -> pd.DataFrame:
    """An advanced table for ``games``, zeroed on the absentee's missed ones."""
    rows: list[dict] = []

    for entry in games:
        for team_id in (entry["home_team_id"], entry["visitor_team_id"]):
            for slot in (1, 2):
                player_id = int(team_id) * 10 + slot

                line = (
                    ADVANCED_DNP_LINE
                    if player_id == ABSENTEE
                    and int(entry["id"]) in case.missed
                    else ADVANCED_LINE
                )

                rows.append(
                    {
                        "player_id": player_id,
                        "team_id": int(team_id),
                        "game_id": int(entry["id"]),
                        "date": entry["date"],
                        "season": int(entry["season"]),
                        **line,
                    }
                )

    return pd.DataFrame(rows)


def build_live(
    case: Scenario,
    dynamic_params: dict,
    builder=None,
    advanced: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The slate as serving builds it: history behind, the game ahead."""
    builder = builder or build_upcoming_slate_features

    return builder(
        history_stats=case.history,
        upcoming_games=pd.DataFrame([case.slate]),
        active_players=active_players(*case.teams),
        advanced=empty_advanced() if advanced is None else advanced,
        dynamic_params=dynamic_params,
        history_games=case.schedule,
    )


def build_training(
    case: Scenario,
    dynamic_params: dict,
    advanced: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The same game as training builds it, once it has been played."""
    trained = add_dynamic_priors(
        build_base_frame(
            case.trained_history,
            games=case.schedule,
            advanced=advanced,
        ),
        dynamic_params,
    )

    rows = trained[trained["game_id"].eq(case.slate["id"])]

    assert not rows.empty, "the training frame does not hold the game under test"

    return rows


def fitted_features(trained: pd.DataFrame, live: pd.DataFrame) -> list[str]:
    """The feature names both frames carry, as ``feature_columns`` names them.

    ``feature_columns`` is what ``fit_minutes_model`` and ``fit_target_model``
    pass to XGBoost, so this is the list the fitted models were shown rather
    than a list of features somebody remembered to check.
    """
    return sorted(set(feature_columns(trained)) & set(live.columns))


def disagreements(
    live: pd.DataFrame,
    trained: pd.DataFrame,
    names: list[str],
    player_ids: tuple[int, ...] = (ABSENTEE, IRON_MAN),
) -> dict[str, dict[int, tuple[float, float]]]:
    """Features whose live value differs from training's, per player."""
    out: dict[str, dict[int, tuple[float, float]]] = {}

    for name in names:
        for player_id in player_ids:
            served = live[live["player_id"].eq(player_id)][name]
            fitted = trained[trained["player_id"].eq(player_id)][name]

            if served.empty or fitted.empty:
                continue

            first, second = float(served.iloc[0]), float(fitted.iloc[0])

            if pd.isna(first) and pd.isna(second):
                continue

            if pd.isna(first) != pd.isna(second) or abs(first - second) > 1e-9:
                out.setdefault(name, {})[player_id] = (first, second)

    return out


# ----------------------------------------------------------------------
# 1: live equals training over the whole fitted feature list
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def recent_absence_case() -> Scenario:
    """The opening-night shape: the absences are the most recent rows."""
    return scenario()


@pytest.fixture(scope="module")
def early_absence_case() -> Scenario:
    """The same absences, at the start of the history instead of the end."""
    return scenario(from_the_start=True)


@pytest.fixture(scope="module")
def null_minutes_case() -> Scenario:
    """The absences spelled with a null minutes value rather than ``"0:00"``."""
    return scenario(dnp_line=NULL_MINUTES_DNP_LINE)


def test_the_fitted_feature_list_agrees_between_live_and_training(
    recent_absence_case,
    early_absence_case,
    null_minutes_case,
    dynamic_params,
):
    """Every feature the models were fitted on, for a player who misses games.

    This is the absolute claim: ``feature_columns`` is the list XGBoost is
    handed, and over a history that holds absences it must mean the same thing
    live as it does in training. It is swept rather than spot-checked because
    the defect this closes was in the ``prior_``, ``decay_prior_`` and
    ``kalman_prior_`` families, which the exact-feature sweep the previous
    parity fix installed does not reach.
    """
    for label, case in (
        ("recent_absence", recent_absence_case),
        ("early_absence", early_absence_case),
        ("null_minutes", null_minutes_case),
    ):
        live = build_live(case, dynamic_params)
        trained = build_training(case, dynamic_params)

        names = fitted_features(trained, live)

        # The sweep has to reach the families the defect was in, or it
        # would pass by covering nothing.
        for prefix in ("prior_", "decay_prior_", "kalman_prior_"):
            assert any(name.startswith(prefix) for name in names), (
                f"{label}: the sweep covers no {prefix} feature"
            )

        assert "prior_minutes10" in names
        assert "days_rest" in names
        assert "career_games_prior" in names

        assert disagreements(live, trained, names) == {}, label


def test_the_advanced_priors_agree_too(recent_absence_case, dynamic_params):
    """The advanced families, over an advanced table that holds DNP rows.

    ``build_base_frame`` restricts the advanced history to the played keys and
    the live builder did not, so this is a second population to get right and
    it needs an advanced table to show it. Swept the same way.
    """
    case = recent_absence_case

    live_advanced = advanced_rows(case, case.played)
    trained_advanced = advanced_rows(case, [*case.played, case.slate])

    live = build_live(case, dynamic_params, advanced=live_advanced)
    trained = build_training(case, dynamic_params, advanced=trained_advanced)

    names = fitted_features(trained, live)

    assert any(name.startswith("adv_prior_") for name in names), (
        "the sweep covers no advanced prior"
    )

    for column in ("usage_percentage", "true_shooting_percentage", "pace"):
        assert f"adv_prior_{column}" in names

    assert disagreements(live, trained, names) == {}


# ----------------------------------------------------------------------
# 2: the population is the one build_base_frame keeps
# ----------------------------------------------------------------------


def test_the_served_population_is_the_one_training_keeps(recent_absence_case):
    """``played_player_games`` keeps exactly what ``build_base_frame`` keeps.

    The rule is reproduced in the slate module rather than read out of
    ``features.py``, for the reason the function's own docstring gives, so
    this pins the two against each other directly as well as through the
    feature parity above. ``build_base_frame`` is asked about the same frame,
    and the surviving player-game keys must be the same set.
    """
    case = recent_absence_case

    history = case.history

    served = played_player_games(history)

    trained = build_base_frame(history, games=case.schedule)

    def keys(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return {
            (int(game_id), int(player_id))
            for game_id, player_id in zip(
                frame["game_id"], frame["player_id"], strict=True
            )
        }

    assert keys(served) == keys(trained)

    # And it is a real restriction, so the test is not comparing a frame to
    # itself.
    assert len(served) == len(history) - ROUNDS_MISSED
    assert len(history) > len(served)


def test_the_served_population_is_in_training_order(recent_absence_case):
    """Chronological per player, which is the order training's EWM runs in.

    ``_final_ewm_by_player`` groups by player and takes the last value of an
    EWM over the group as it finds it, so the order of the frame it is handed
    is part of the feature definition. Training sorts by
    ``["player_id", "date", "game_id"]`` before any prior is computed; the
    live builder was handed whatever order the season files concatenated in.
    """
    shuffled = recent_absence_case.history.sample(
        frac=1.0, random_state=73
    ).reset_index(drop=True)

    served = played_player_games(shuffled)

    expected = served.sort_values(
        ["player_id", "date", "game_id"]
    ).reset_index(drop=True)

    assert served.equals(expected)


def test_an_empty_advanced_table_is_passed_through(recent_absence_case):
    """The restriction has nothing to do when there are no advanced rows."""
    empty = empty_advanced()

    assert played_advanced_rows(empty, recent_absence_case.history) is empty
    assert played_advanced_rows(None, recent_absence_case.history) is None


def test_the_advanced_restriction_drops_only_the_unplayed_rows(
    recent_absence_case,
):
    """And keeps every played key, so it cannot silently empty the table."""
    case = recent_absence_case

    advanced = advanced_rows(case, case.played)

    restricted = played_advanced_rows(
        advanced, played_player_games(case.history)
    )

    assert len(restricted) == len(advanced) - ROUNDS_MISSED

    dropped = {
        (int(game_id), int(player_id))
        for game_id, player_id in zip(
            advanced["game_id"], advanced["player_id"], strict=True
        )
    } - {
        (int(game_id), int(player_id))
        for game_id, player_id in zip(
            restricted["game_id"], restricted["player_id"], strict=True
        )
    }

    assert dropped == {
        (game_id, ABSENTEE) for game_id in sorted(case.missed)
    }


# ----------------------------------------------------------------------
# 3: the leakage guard still reads the unrestricted history
# ----------------------------------------------------------------------


def test_a_same_day_dnp_row_still_fails_the_leakage_guard(
    recent_absence_case, dynamic_params
):
    """The restriction is taken after the guard, and that ordering matters.

    ``assert_slate_inputs_precede_slate`` refuses historical inputs that are
    not strictly older than the slate. Restricting the population first would
    let a same-day or later-day row evade it by being a row for a player who
    did not play, which is precisely the row a late-arriving box score starts
    life as.
    """
    case = recent_absence_case

    leaking = pd.concat(
        [
            case.history,
            pd.DataFrame(
                [
                    {
                        **case.history.iloc[0].to_dict(),
                        **DNP_LINE,
                        "game_id": int(case.slate["id"]),
                        "date": pd.Timestamp(case.slate["date"]),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(HistoryLeakageError):
        build_upcoming_slate_features(
            history_stats=leaking,
            upcoming_games=pd.DataFrame([case.slate]),
            active_players=active_players(*case.teams),
            advanced=empty_advanced(),
            dynamic_params=dynamic_params,
            history_games=case.schedule,
        )


# ----------------------------------------------------------------------
# 4: what the pre-fix builder served, measured
# ----------------------------------------------------------------------


def test_the_pre_fix_builder_deflated_the_absentees_priors(
    recent_absence_case, dynamic_params, pre_fix_builder
):
    """The defect, measured, and the sweep's ability to find it.

    The complement of the parity sweep above: it is only worth asserting that
    nothing disagrees now if the same sweep finds the disagreement this change
    removes. Every family the fix covers has to show up, and the direction has
    to be down for each one -- a deflation rather than noise is what turns a
    slate of fair prices into a slate of unders.
    """
    case = recent_absence_case

    trained = build_training(case, dynamic_params)
    before = build_live(case, dynamic_params, pre_fix_builder)

    names = fitted_features(trained, before)

    moved = disagreements(before, trained, names, player_ids=(ABSENTEE,))

    for family in ("prior_minutes10", "decay_prior_min", "kalman_prior_min"):
        assert family in moved, family

    for prefix in ("prior_", "decay_prior_", "kalman_prior_"):
        assert any(name.startswith(prefix) for name in moved), prefix

    # Every rolling prior the models read was served low, not merely wrong.
    deflated = {
        name: values[ABSENTEE]
        for name, values in moved.items()
        if name.startswith(("prior_", "decay_prior_", "kalman_prior_"))
    }

    assert deflated, "no rolling prior moved"

    for name, (served, fitted) in sorted(deflated.items()):
        if fitted == 0.0:
            continue

        assert served < fitted, f"{name} was not served low: {served} vs {fitted}"


def test_the_pre_fix_builder_left_a_player_who_never_misses_alone(
    recent_absence_case, dynamic_params, pre_fix_builder
):
    """Which is why the defect reads as a view rather than as a failure.

    The iron man's served features were already right, so a slate built with
    the defect is a slate where the durable players are priced correctly and
    the absence-prone ones are marked down. Nothing in the output says so.
    """
    case = recent_absence_case

    trained = build_training(case, dynamic_params)
    before = build_live(case, dynamic_params, pre_fix_builder)

    names = fitted_features(trained, before)

    assert disagreements(before, trained, names, player_ids=(IRON_MAN,)) == {}


def test_the_minutes_prior_the_pre_fix_builder_served_was_less_than_half(
    recent_absence_case, dynamic_params, pre_fix_builder
):
    """The size of it, for the feature every count is scaled by.

    ``expected_minutes`` is the minutes model's output and the multiplier the
    ensemble's decay and Kalman components are formed with, so a minutes prior
    served at half its value is a count mean served at half or less. Pinned as
    a number because "a bias" and "the minutes feature was halved" are
    different claims and only the second one is checkable.
    """
    case = recent_absence_case

    trained = build_training(case, dynamic_params)
    before = build_live(case, dynamic_params, pre_fix_builder)

    def minutes_prior(frame: pd.DataFrame, name: str) -> float:
        return float(frame[frame["player_id"].eq(ABSENTEE)][name].iloc[0])

    fitted = minutes_prior(trained, "prior_minutes10")
    served = minutes_prior(before, "prior_minutes10")

    assert served < 0.5 * fitted

    # And the Kalman minutes state, which collapses rather than merely halves.
    assert minutes_prior(before, "kalman_prior_min") < 0.1 * minutes_prior(
        trained, "kalman_prior_min"
    )


def test_the_days_rest_the_pre_fix_builder_served_was_the_absence(
    recent_absence_case, dynamic_params, pre_fix_builder
):
    """A returning player was served the rest since a game he did not play.

    ``days_rest`` is in ``FEATURE_EXACT`` and training defines it as the gap
    since the player's last *appearance*. Serving measured it from his last
    roster row, so every player returning from an absence arrived looking
    fully rested in the one direction that does not help -- no rest at all.
    """
    case = recent_absence_case

    trained = build_training(case, dynamic_params)
    before = build_live(case, dynamic_params, pre_fix_builder)
    after = build_live(case, dynamic_params)

    def rest(frame: pd.DataFrame) -> float:
        return float(frame[frame["player_id"].eq(ABSENTEE)]["days_rest"].iloc[0])

    assert rest(before) != rest(trained)
    assert rest(after) == rest(trained)


# ----------------------------------------------------------------------
# 5: nothing moves for a history with no absences
# ----------------------------------------------------------------------


def test_the_fix_changes_nothing_when_nobody_missed_a_game(
    dynamic_params, pre_fix_builder
):
    """Column by column, over a clean history, the whole frame is unchanged.

    This is what makes "only the contaminated rows moved" a measurement. A
    rolling window, an experience input or a team-context value that this
    change moved for its own reasons would show up here by name.
    """
    case = scenario(missed_rounds=0)

    assert case.missed == frozenset()

    before = build_live(case, dynamic_params, pre_fix_builder)
    after = build_live(case, dynamic_params)

    assert list(before.columns) == list(after.columns)
    assert len(before) == len(after)

    moved = [
        column
        for column in before.columns
        if not before[column].equals(after[column])
    ]

    assert moved == [], moved


def test_the_fix_moves_nothing_for_the_other_players_on_the_slate(
    recent_absence_case, dynamic_params, pre_fix_builder
):
    """Only the absentee's rows move, on a slate where one player missed games.

    The restriction is a population change, so the question worth asking is
    whether it reached anybody it should not have. Every player other than
    the absentee is compared whole.
    """
    case = recent_absence_case

    before = build_live(case, dynamic_params, pre_fix_builder)
    after = build_live(case, dynamic_params)

    assert list(before.columns) == list(after.columns)

    untouched = before["player_id"].ne(ABSENTEE)

    assert untouched.any()

    for column in before.columns:
        assert (
            before.loc[untouched, column]
            .reset_index(drop=True)
            .equals(after.loc[untouched, column].reset_index(drop=True))
        ), column


# ----------------------------------------------------------------------
# 6: the training definition did not move
# ----------------------------------------------------------------------


def test_the_training_feature_definitions_are_untouched():
    """``features.py`` is byte-identical to production.

    The authority here is the training definition, so the fix is only a fix
    if the training definition is where it was. ``decay.py`` and ``kalman.py``
    are included because the decay and Kalman priors are among the families
    whose served value this change moves.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    for relative in (
        "src/nba_prop_quant/features.py",
        "src/nba_prop_quant/decay.py",
        "src/nba_prop_quant/kalman.py",
        "src/nba_prop_quant/game_context.py",
        "src/nba_prop_quant/model.py",
        "src/nba_prop_quant/production.py",
        "src/nba_prop_quant/pricing.py",
        "src/nba_prop_quant/distributions.py",
    ):
        assert blob_text(base, relative) == (
            PROJECT / relative
        ).read_text(encoding="utf-8"), relative


def test_the_advanced_restriction_is_the_one_training_states():
    """``build_base_frame`` restricts advanced rows the same way, and says so.

    Pinned against the training module's own source rather than against a
    paraphrase of it, so the comment that documents the rule cannot be
    removed from training while the live builder goes on claiming to
    reproduce it.
    """
    features = (
        PROJECT / "src" / "nba_prop_quant" / "features.py"
    ).read_text(encoding="utf-8")

    assert "DNP / zero-minute" in features
    assert 'df[df["minutes"].fillna(0) > 0]' in features
    assert 'on=["game_id", "player_id"]' in features


def test_the_slate_module_reaches_for_no_diagnostic_override():
    """Neither the module nor this file waives the feature contract."""
    override = "--allow-" + "missing-model-features"

    for path in (PROJECT / SLATE_MODULE, Path(__file__)):
        text = path.read_text(encoding="utf-8")

        assert override not in text, path
        assert ("allow_missing" + "=True") not in text, path
