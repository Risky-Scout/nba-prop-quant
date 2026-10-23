"""What an appearance-only prior is worth for a player with no appearances.

The population argument's last stop. ``played_player_games`` settled which
player-games a prior averages over and ``prior_participation_rate`` settled
what an absence frees; both took the appearance-only priors as the authority.
This is the row where that authority has nothing to say: a player making his
first NBA appearance has no games to average, so ``prior_minutes10`` and all
eleven ``prior_<stat>_rate10`` columns have no value.

``build_base_frame`` says so. It takes every one of them through
``_lagged_ewm``, which shifts by one inside the player group, so the first row
of every player in the fitted frame carries NaN in all twelve -- 2,542 of the
corpus's 667,556 rows -- and nothing fills them. ``ModelBundle.predict``
reindexes onto the feature names and hands the frame to an ``XGBRegressor``,
which learns a default branch for a missing value, so the fitted minutes model
holds a branch fit on exactly those rows.

The live builder substituted the played population's mean minutes instead.
That is not a neutral stand-in: it is what a rotation regular's prior looks
like, so a player nobody had seen was scored as one with twenty-three minutes
a night behind him. It refused nothing and announced nothing, which is the
same way the ``season_progress`` and played-population divergences went
unnoticed.

The tests here build the same first appearance twice from the same historical
state -- once through ``build_base_frame`` as training does, once through
``build_upcoming_slate_features`` as serving does -- and require the twelve
priors to agree. The pre-fix builder is loaded out of Git beside them, so the
substitution each case closes is recorded as a measurement rather than as a
claim.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_prop_quant.features import add_dynamic_priors, build_base_frame
from nba_prop_quant.slate import NO_APPEARANCE_PRIOR, build_upcoming_slate_features

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from season_games_fixture import (  # noqa: E402
    FULL_SCHEDULE_LENGTH,
    pad_season_schedule,
)
from season_parity_fixture import (  # noqa: E402
    AWAY,
    BOX_SCORE_LINE,
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

#: The module constant whose presence in the production base means this fix
#: landed. A constant rather than a function because this fix adds no
#: function: the one it needed, ``played_player_games``, is already there.
FIX_MARKER_CONSTANT = "NO_APPEARANCE_PRIOR"

SLATE_ID = 9802

#: Rounds of ordinary season behind the slate. Enough that a span-ten EWM has
#: reached steady state, so the established players' priors are a flat 30.0
#: and any movement in them is the change under test rather than a transient.
ROUNDS_BEHIND_THE_SLATE = 20

#: The player every roster in the fixture has and no box score does: he is on
#: the active roster for the slate game and has never played an NBA game, which
#: is the serving condition. Slots 1 and 2 are the fixture's established
#: players, so slot 3 is his.
DEBUTANT = HOME * 10 + 3

#: Every appearance-only prior ``build_base_frame`` derives from a player's own
#: games through ``_lagged_ewm``. All twelve are NaN on a first row there, and
#: the first eleven are what the live builder was substituting a league average
#: into. Listed in full rather than matched by prefix so a column added to
#: either side has to be accounted for here.
APPEARANCE_ONLY_PRIORS: tuple[str, ...] = (
    "prior_minutes10",
    "prior_pts_rate10",
    "prior_reb_rate10",
    "prior_ast_rate10",
    "prior_stl_rate10",
    "prior_blk_rate10",
    "prior_fg3m_rate10",
    "prior_fga_rate10",
    "prior_fg3a_rate10",
    "prior_fta_rate10",
    "prior_turnover_rate10",
    "prior_pf_rate10",
)

#: The two dynamic priors that are *not* in the list above, and must not move.
#: ``decay_prior`` opens each player with ``global_prior * prior_strength`` over
#: ``prior_strength``, which is ``global_prior``, and the live builder's
#: no-history branch reads the same ``global_prior`` out of the fitted
#: parameters. Those two already agree, so this fix has no business touching
#: them and these assertions are what says so.
GLOBAL_PRIOR_FALLBACKS: tuple[str, ...] = ("decay_prior_min", "kalman_prior_min")


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


def module_constants(source: str) -> set[str]:
    names: set[str] = set()

    for node in ast.parse(source).body:
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign) else
            list(node.targets) if isinstance(node, ast.Assign) else []
        )
        names.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )

    return names


def pre_fix_slate_source() -> str:
    """The slate module as production has it, or skip.

    The differential cases here are measured against the production merge
    base, which carries the pre-fix builder only while this fix has not
    landed. Once it has, "the pre-fix builder" is this builder under another
    name, and a test asserting the two differ asserts something false about a
    correct tree. The condition is whether the base already defines the fix,
    not whether the file changed, so the next branch to touch this module for
    an unrelated reason does not have its change measured against this fix's
    expectations.

    The absolute cases here, live against training, do not come through this
    and keep running forever. They are the ones worth keeping.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    before = blob_text(base, SLATE_MODULE)

    if FIX_MARKER_CONSTANT in module_constants(before):
        pytest.skip(
            "production already carries this fix, so there is no pre-fix "
            "state left to compare against"
        )

    return before


@pytest.fixture(scope="module")
def pre_fix_builder():
    """``build_upcoming_slate_features`` as production has it today.

    Loaded from the merge base rather than reconstructed, so "what the old
    fallback served" is the old code's answer and not a test's idea of it.
    """
    source = pre_fix_slate_source()

    path = Path(tempfile.mkdtemp()) / "production_slate.py"
    path.write_text(source, encoding="utf-8")

    name = "nba_prop_quant._pre_fix_no_appearance_slate_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module.build_upcoming_slate_features


def test_the_fix_marker_names_a_constant_the_module_really_defines():
    """The skip condition is anchored to something real, not assumed.

    A marker naming nothing would make every differential case in this file
    skip forever, which is the failure mode worth one test of its own. It
    deliberately does not also assert that the merge base lacks the marker:
    once this fix lands the base carries it, and that is exactly when the
    differential cases are meant to stop running.
    """
    assert FIX_MARKER_CONSTANT in module_constants(
        (PROJECT / SLATE_MODULE).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def dynamic_params() -> dict:
    return json.loads(
        (PROJECT / "models" / "dynamic_params.json").read_text(encoding="utf-8")
    )


def debutant_line(slate: dict, minutes: float = 14.0) -> dict:
    """The debutant's own box score for the slate game, once it is played.

    Training only ever sees him through this row, and it is his first, so
    every appearance-only prior on it is the shifted EWM of an empty history.
    """
    return {
        "player_id": DEBUTANT,
        "team_id": HOME,
        "game_id": int(slate["id"]),
        "date": slate["date"],
        "season": int(slate["season"]),
        "postseason": bool(slate["postseason"]),
        "home_team_id": int(slate["home_team_id"]),
        "visitor_team_id": int(slate["visitor_team_id"]),
        "position": "C",
        "draft_year": 2026,
        **{**BOX_SCORE_LINE, "min": "14:00", "minutes": float(minutes)},
    }


class Case:
    """A slate game with a never-seen player on one roster.

    Built once per module: the history is twenty ordinary rounds in which the
    fixture's two players per team play every game and the debutant plays
    none, and the slate is the game after them.
    """

    def __init__(self) -> None:
        self.played = standings_rounds(ROUNDS_BEHIND_THE_SLATE)
        self.slate = game(
            SLATE_ID, day_after(self.played), home_team_id=HOME, visitor_team_id=AWAY
        )
        self.schedule = pad_season_schedule(
            pd.DataFrame([*self.played, self.slate]),
            schedule_length=FULL_SCHEDULE_LENGTH,
        )

    @property
    def history(self) -> pd.DataFrame:
        """What serving has: the games behind the slate, debutant absent."""
        return box_scores(self.played)

    @property
    def trained_history(self) -> pd.DataFrame:
        """What training has: those games, the slate, and his first line."""
        return pd.concat(
            [
                box_scores([*self.played, self.slate]),
                pd.DataFrame([debutant_line(self.slate)]),
            ],
            ignore_index=True,
        )

    @property
    def roster(self) -> pd.DataFrame:
        return pd.concat(
            [
                active_players(HOME, AWAY),
                pd.DataFrame(
                    [
                        {
                            "id": DEBUTANT,
                            "current_team_id": HOME,
                            "position": "C",
                            "draft_year": 2026,
                            "first_name": "Never",
                            "last_name": "Played",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )


@pytest.fixture(scope="module")
def case() -> Case:
    return Case()


def build_live(case: Case, dynamic_params: dict, builder=None) -> pd.DataFrame:
    builder = builder or build_upcoming_slate_features

    return builder(
        history_stats=case.history,
        upcoming_games=pd.DataFrame([case.slate]),
        active_players=case.roster,
        advanced=empty_advanced(),
        dynamic_params=dynamic_params,
        history_games=case.schedule,
    )


def build_training(case: Case, dynamic_params: dict) -> pd.DataFrame:
    trained = add_dynamic_priors(
        build_base_frame(case.trained_history, games=case.schedule),
        dynamic_params,
    )

    rows = trained[trained["game_id"].eq(case.slate["id"])]

    assert not rows.empty, "the training frame does not hold the game under test"

    return rows


def row_for(frame: pd.DataFrame, player_id: int) -> pd.Series:
    rows = frame[frame["player_id"].eq(player_id)]

    assert len(rows) == 1, f"expected one row for {player_id}, got {len(rows)}"

    return rows.iloc[0]


# ----------------------------------------------------------------------
# what training states
# ----------------------------------------------------------------------


def test_training_leaves_every_appearance_only_prior_missing_on_a_first_row(
    case, dynamic_params
):
    """``build_base_frame``, not this test, is the authority on the value.

    ``_lagged_ewm`` shifts by one inside the player group, so a player's first
    row in the fitted frame has no prior to average. This is the fact the live
    builder has to reproduce, so it is asserted directly rather than inferred
    from the two agreeing.
    """
    trained = build_training(case, dynamic_params)
    served = row_for(trained, DEBUTANT)

    assert served["career_games_prior"] == 0

    for column in APPEARANCE_ONLY_PRIORS:
        assert pd.isna(served[column]), (
            f"build_base_frame gave a first row a value for {column}"
        )


def test_the_constant_the_builder_serves_is_the_value_training_leaves():
    """``NO_APPEARANCE_PRIOR`` states no number of its own.

    The point of the fix is that there was no value to choose. A constant that
    had drifted to a number would be a second definition of a prior, which is
    the thing the serving-source carve-out exists to prevent.
    """
    assert isinstance(NO_APPEARANCE_PRIOR, float)
    assert np.isnan(NO_APPEARANCE_PRIOR)


# ----------------------------------------------------------------------
# what serving now serves
# ----------------------------------------------------------------------


def test_the_live_builder_serves_training_s_value_for_a_player_with_no_games(
    case, dynamic_params
):
    """The absolute case, and the one worth keeping.

    Live against training on the same twelve columns, for the same player, in
    the same game. It does not go through the pre-fix builder and so keeps
    running after this fix is the base.
    """
    live = row_for(build_live(case, dynamic_params), DEBUTANT)
    trained = row_for(build_training(case, dynamic_params), DEBUTANT)

    for column in APPEARANCE_ONLY_PRIORS:
        assert pd.isna(live[column]), (
            f"the live builder served a value for {column} to a player with "
            "no appearances"
        )
        assert pd.isna(trained[column])


def test_a_player_with_one_appearance_keeps_his_own_prior(case, dynamic_params):
    """The condition is "no games", not "few games".

    A single appearance is enough for the EWM to have a value, and serving it
    is correct. This is the boundary the fix must not cross, so it is tested
    on its own rather than left to the established players below.
    """
    played_once = box_scores([case.played[0]])
    newcomer = played_once[played_once["player_id"].eq(HOME * 10 + 1)].copy()
    newcomer["player_id"] = DEBUTANT
    newcomer["team_id"] = HOME
    newcomer["minutes"] = 21.0
    newcomer["min"] = "21:00"

    history = pd.concat([case.history, newcomer], ignore_index=True)

    live = build_upcoming_slate_features(
        history_stats=history,
        upcoming_games=pd.DataFrame([case.slate]),
        active_players=case.roster,
        advanced=empty_advanced(),
        dynamic_params=dynamic_params,
        history_games=case.schedule,
    )

    served = row_for(live, DEBUTANT)

    assert served["career_games_prior"] == 1
    assert served["prior_minutes10"] == pytest.approx(21.0)

    for column in APPEARANCE_ONLY_PRIORS:
        assert pd.notna(served[column]), (
            f"a player with one appearance was served no {column}"
        )


def test_the_global_prior_fallbacks_are_untouched(case, dynamic_params):
    """The two dynamic priors that already agreed still do.

    ``decay_prior`` opens a player at ``global_prior`` and the live builder's
    no-history branch reads the same ``global_prior`` from the fitted
    parameters, so those two were never the defect. A fix that zeroed or
    blanked them would be changing a feature that was correct.
    """
    live = row_for(build_live(case, dynamic_params), DEBUTANT)
    trained = row_for(build_training(case, dynamic_params), DEBUTANT)

    for column in GLOBAL_PRIOR_FALLBACKS:
        assert pd.notna(live[column]), f"{column} lost its global prior"
        assert live[column] == pytest.approx(trained[column], rel=1e-9), (
            f"{column} no longer matches what training opens a player at"
        )


def test_players_with_history_are_served_exactly_what_they_were(
    case, dynamic_params, pre_fix_builder
):
    """Mechanically unchanged, column by column, for everyone else.

    The differential that matters most: the fix is conditioned on
    ``games_prior``, so every row for a player with any appearance has to come
    out of the new builder identical to the old one. Compared over the whole
    frame rather than over the twelve priors, because a condition written
    wrongly could move anything.
    """
    before = build_live(case, dynamic_params, builder=pre_fix_builder)
    after = build_live(case, dynamic_params)

    established = sorted(set(before["player_id"]) - {DEBUTANT})

    assert len(established) == 4, "the fixture should hold four played players"

    columns = [
        column
        for column in before.columns
        if column != "projection_generated_at_utc"
    ]

    pd.testing.assert_frame_equal(
        before[before["player_id"].isin(established)][columns]
        .sort_values("player_id")
        .reset_index(drop=True),
        after[after["player_id"].isin(established)][columns]
        .sort_values("player_id")
        .reset_index(drop=True),
    )


def test_the_pre_fix_builder_served_the_played_populations_mean_minutes(
    case, dynamic_params, pre_fix_builder
):
    """What the defect was, measured rather than described.

    The fixture's every played line is 30 minutes, so the played population's
    mean is exactly 30.0 and the old builder served a player nobody had seen a
    prior saying he averages a full starter's night. The eleven rate priors
    went the same way for the six target stats and were already missing for
    the other five, which is the asymmetry that made this hard to see.
    """
    before = row_for(build_live(case, dynamic_params, builder=pre_fix_builder), DEBUTANT)
    after = row_for(build_live(case, dynamic_params), DEBUTANT)

    assert before["prior_minutes10"] == pytest.approx(30.0)
    assert pd.isna(after["prior_minutes10"])

    substituted = [
        column
        for column in APPEARANCE_ONLY_PRIORS
        if pd.notna(before[column])
    ]

    assert substituted == [
        "prior_minutes10",
        "prior_pts_rate10",
        "prior_reb_rate10",
        "prior_ast_rate10",
        "prior_stl_rate10",
        "prior_blk_rate10",
        "prior_fg3m_rate10",
    ], substituted

    for column in substituted:
        assert pd.isna(after[column])


def test_the_fitted_minutes_model_still_accepts_the_slate(case, dynamic_params):
    """A missing value is not a missing column.

    ``assert_model_feature_contract`` is a column-presence check with no
    allowance, and it is what refuses a slate outright. Serving NaN keeps the
    column, which is the difference between the fix and dropping the feature.
    """
    from nba_prop_quant.production import assert_model_feature_contract

    class Bundle:
        feature_names = list(APPEARANCE_ONLY_PRIORS)

    live = build_live(case, dynamic_params)

    assert assert_model_feature_contract(live, Bundle(), label="minutes model") == []
