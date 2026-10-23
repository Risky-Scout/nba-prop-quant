"""Freed minutes are the ones the survivors' priors still assign to the absentee.

``apply_current_injury_adjustment`` took every OUT player's projected minutes as
opportunity to hand back to his team-mates. That is right for a player who was
playing until today and wrong for one who has not played in weeks: the minutes
EWMs the survivors carry are built over appearances, so a team-mate who was
already absent across that window is already absent from those priors, and his
rotation replacement's elevated minutes are already in them. Crediting the
absence again adds minutes the team never had, and it does so on exactly the
teams carrying long-term injuries.

The October 20 slate held the degenerate form of this. Three OUT players had
never played an NBA game, so their minutes were the league fallback rather than
any observed workload, and 55.63 minutes of opportunity that had never existed
was redistributed to their team-mates.

The repair weights each OUT player's freed minutes by his share of the recent
window the priors are built over: unchanged for a new absence, nothing for an
established one, prorated between. The tests here pin each of those three cases,
pin that OUT zeroing and the elasticity allocation are untouched, and pin that
the weighting is the only thing that moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_prop_quant.availability import apply_current_injury_adjustment
from nba_prop_quant.slate import (
    _prior_window_participation,
    build_upcoming_slate_features,
)

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from season_parity_fixture import (  # noqa: E402
    HOME,
    active_players,
    empty_advanced,
)

TEAM = 10
ABSENTEE = 1
SURVIVOR_A = 2
SURVIVOR_B = 3


def slate_frame(
    *,
    participation: dict[int, float] | None = None,
    minutes: dict[int, float] | None = None,
) -> pd.DataFrame:
    """One team: an absentee and two survivors with room to absorb minutes."""
    minutes = minutes or {ABSENTEE: 24.0, SURVIVOR_A: 20.0, SURVIVOR_B: 18.0}

    frame = pd.DataFrame(
        {
            "player_id": list(minutes),
            "team_id": TEAM,
            "expected_minutes": [minutes[pid] for pid in minutes],
        }
    )

    if participation is not None:
        frame["prior_participation_rate"] = frame["player_id"].map(participation)

    return frame


def out_feed(*player_ids: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"player_id": list(player_ids), "status": ["Out"] * len(player_ids)}
    )


def adjust(frame: pd.DataFrame, *out_ids: int) -> pd.DataFrame:
    return apply_current_injury_adjustment(frame, injuries=out_feed(*out_ids))


def uplift(result: pd.DataFrame) -> float:
    """Minutes handed to the survivors."""
    healthy = result[result["availability_out"].eq(0)]
    return float(healthy["availability_minutes_delta"].sum())


# ----------------------------------------------------------------------
# the three absence cases
# ----------------------------------------------------------------------


def test_persistently_absent_out_player_frees_no_minutes() -> None:
    """Absent across the prior window, so the priors already moved on."""
    result = adjust(
        slate_frame(participation={ABSENTEE: 0.0, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0}),
        ABSENTEE,
    )

    assert uplift(result) == pytest.approx(0.0)

    survivors = result[result["availability_out"].eq(0)]

    assert survivors["availability_expected_minutes"].tolist() == pytest.approx(
        survivors["availability_base_minutes"].tolist()
    )


def test_never_played_out_player_frees_no_phantom_minutes() -> None:
    """A fallback-imputed workload is not opportunity: it was never observed.

    The minutes are the league prior rather than anything this player did, so
    weighting them by a participation share of zero is what keeps them from
    being handed to his team-mates as if he had vacated a role.
    """
    result = adjust(
        slate_frame(
            participation={ABSENTEE: 0.0, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0},
            minutes={ABSENTEE: 23.32, SURVIVOR_A: 20.0, SURVIVOR_B: 18.0},
        ),
        ABSENTEE,
    )

    assert result.loc[
        result["player_id"].eq(ABSENTEE), "availability_base_minutes"
    ].item() == pytest.approx(23.32)

    assert uplift(result) == pytest.approx(0.0)


def test_newly_absent_out_player_still_redistributes() -> None:
    """Present for the whole window, so the priors still hold his minutes."""
    result = adjust(
        slate_frame(participation={ABSENTEE: 1.0, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0}),
        ABSENTEE,
    )

    assert uplift(result) == pytest.approx(24.0)


def test_partial_participation_is_prorated() -> None:
    """Half the window present, so half the minutes are still his to free."""
    result = adjust(
        slate_frame(participation={ABSENTEE: 0.5, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0}),
        ABSENTEE,
    )

    assert uplift(result) == pytest.approx(12.0)


def test_survivor_participation_does_not_gate_receiving() -> None:
    """The weighting asks who vacated minutes, not who may receive them."""
    full = adjust(
        slate_frame(participation={ABSENTEE: 1.0, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0}),
        ABSENTEE,
    )

    thin = adjust(
        slate_frame(participation={ABSENTEE: 1.0, SURVIVOR_A: 0.1, SURVIVOR_B: 0.0}),
        ABSENTEE,
    )

    assert uplift(thin) == pytest.approx(uplift(full))


# ----------------------------------------------------------------------
# what the repair must not have moved
# ----------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_out_zeroing_preserved(share: float) -> None:
    """An OUT player is held at zero minutes whatever he freed."""
    result = adjust(
        slate_frame(participation={ABSENTEE: share, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0}),
        ABSENTEE,
    )

    absentee = result[result["player_id"].eq(ABSENTEE)]

    assert absentee["availability_expected_minutes"].item() == 0.0
    assert absentee["availability_out"].item() == 1
    assert absentee["availability_usage_multiplier"].item() == 0.0


def test_elasticity_allocation_unchanged_for_eligible_minutes() -> None:
    """Fully represented absences reproduce the unweighted allocation exactly.

    The column is absent from the second frame, which is the pre-repair input,
    so this is the numerical statement that the elasticity loop itself did not
    move: same freed total in, same minutes out, to the bit.
    """
    weighted = adjust(
        slate_frame(participation={ABSENTEE: 1.0, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0}),
        ABSENTEE,
    )

    unweighted = adjust(slate_frame(), ABSENTEE)

    assert "prior_participation_rate" not in unweighted.columns

    for column in (
        "availability_expected_minutes",
        "availability_minutes_delta",
        "availability_usage_multiplier",
    ):
        assert weighted[column].to_numpy() == pytest.approx(
            unweighted[column].to_numpy(), abs=0.0
        )


def test_missing_column_is_the_unweighted_path() -> None:
    """No participation column means every OUT player counts in full."""
    result = adjust(slate_frame(), ABSENTEE)

    assert uplift(result) == pytest.approx(24.0)


def test_no_absence_is_untouched_either_way() -> None:
    """Nothing is freed and nothing moves when the feed lists nobody out."""
    for frame in (
        slate_frame(),
        slate_frame(participation={ABSENTEE: 1.0, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0}),
    ):
        result = apply_current_injury_adjustment(
            frame, injuries=pd.DataFrame(columns=["player_id", "status"])
        )

        assert result["availability_minutes_delta"].abs().max() == pytest.approx(0.0)
        assert result["availability_out"].sum() == 0


def test_cap_still_bounds_the_allocation() -> None:
    """Freed minutes beyond the cap are dropped, not forced onto survivors."""
    result = adjust(
        slate_frame(
            participation={ABSENTEE: 1.0, SURVIVOR_A: 1.0, SURVIVOR_B: 1.0},
            minutes={ABSENTEE: 40.0, SURVIVOR_A: 40.0, SURVIVOR_B: 40.0},
        ),
        ABSENTEE,
    )

    assert result["availability_expected_minutes"].max() <= 42.0


# ----------------------------------------------------------------------
# the share itself
# ----------------------------------------------------------------------


def window_history(appearances: dict[int, list[int]], span: int = 12) -> pd.DataFrame:
    """One team's box scores: ``appearances`` maps a player to the games he played."""
    rows = [
        {
            "game_id": game_id,
            "player_id": player_id,
            "team_id": TEAM,
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=game_id),
            "minutes": 24.0,
        }
        for player_id, games in appearances.items()
        for game_id in games
    ]

    return pd.DataFrame(rows)


def test_participation_share_counts_the_recent_window_only() -> None:
    """The window is the last ten team games, which is the EWM's span."""
    every_game = list(range(1, 21))

    share = _prior_window_participation(
        window_history(
            {
                SURVIVOR_A: every_game,
                # Played only the first half, so absent from the last ten.
                ABSENTEE: every_game[:10],
            }
        )
    )

    assert share[(TEAM, SURVIVOR_A)] == pytest.approx(1.0)

    # Absent from every game in the window, so he holds no entry at all, which
    # is the same answer the slate builder's default turns into a zero share.
    assert (TEAM, ABSENTEE) not in share


def test_participation_share_is_prorated_within_the_window() -> None:
    every_game = list(range(1, 11))

    share = _prior_window_participation(
        window_history({SURVIVOR_A: every_game, ABSENTEE: every_game[:5]})
    )

    assert share[(TEAM, ABSENTEE)] == pytest.approx(0.5)


def test_participation_share_is_absent_for_a_player_who_never_played() -> None:
    share = _prior_window_participation(
        window_history({SURVIVOR_A: list(range(1, 11))})
    )

    assert (TEAM, ABSENTEE) not in share


def test_participation_share_is_per_team() -> None:
    """A player's share belongs to the team whose window it was measured in."""
    history = pd.concat(
        [
            window_history({SURVIVOR_A: list(range(1, 11))}),
            window_history({SURVIVOR_A: list(range(1, 11))}).assign(team_id=TEAM + 1),
        ],
        ignore_index=True,
    )

    share = _prior_window_participation(history)

    assert share[(TEAM, SURVIVOR_A)] == pytest.approx(1.0)
    assert share[(TEAM + 1, SURVIVOR_A)] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# through the slate builder, where the column is actually produced
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def dynamic_params() -> dict:
    import json

    return json.loads(
        (PROJECT / "models" / "dynamic_params.json").read_text(encoding="utf-8")
    )


def test_slate_builder_emits_a_zero_share_for_a_player_with_no_history(
    dynamic_params: dict,
) -> None:
    """The never-played case end to end: no minutes prior, zero share.

    This is the October 20 shape. The roster carries a player the history has
    never seen, so his participation share is zero because he appears in no
    game the survivors' priors were built over.

    His minutes prior is missing rather than the league mean, which is the
    separate no-history fix one question further down and is asserted here
    because the two have to compose: the share is what stops a never-played
    player freeing minutes at all, and it has to keep doing that whatever his
    own prior says. He still reaches the availability layer with a projected
    total -- the fitted model scores the row from the rest of its features --
    so the guard is still the thing standing between that total and his
    team-mates.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from season_games_fixture import pad_season_schedule  # noqa: E402
    from season_parity_fixture import (  # noqa: E402
        AWAY,
        box_scores,
        day_after,
        game,
        standings_rounds,
    )

    played = standings_rounds(12)
    slate = game(99_001, day_after(played), home_team_id=HOME, visitor_team_id=AWAY)

    roster = active_players(HOME, AWAY)
    newcomer = roster.iloc[0].copy()
    newcomer["id"] = 987_654
    newcomer["first_name"] = "Never"
    roster = pd.concat([roster, newcomer.to_frame().T], ignore_index=True)

    live = build_upcoming_slate_features(
        history_stats=box_scores(played),
        upcoming_games=pd.DataFrame([slate]),
        active_players=roster,
        advanced=empty_advanced(),
        dynamic_params=dynamic_params,
        history_games=pad_season_schedule(pd.DataFrame([*played, slate])),
    )

    assert "prior_participation_rate" in live.columns

    row = live[live["player_id"].eq(987_654)]

    assert not row.empty, "the slate does not hold the roster newcomer"
    assert row["prior_participation_rate"].item() == pytest.approx(0.0)
    assert pd.isna(row["prior_minutes10"].item())

    established = live[live["player_id"].eq(HOME * 10 + 1)]

    assert established["prior_participation_rate"].item() == pytest.approx(1.0)


def test_slate_builder_share_is_bounded(dynamic_params: dict) -> None:
    """Whatever the history, the share is a share."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from season_games_fixture import pad_season_schedule  # noqa: E402
    from season_parity_fixture import (  # noqa: E402
        AWAY,
        box_scores,
        day_after,
        game,
        standings_rounds,
    )

    played = standings_rounds(12)
    slate = game(99_002, day_after(played), home_team_id=HOME, visitor_team_id=AWAY)

    live = build_upcoming_slate_features(
        history_stats=box_scores(played),
        upcoming_games=pd.DataFrame([slate]),
        active_players=active_players(HOME, AWAY),
        advanced=empty_advanced(),
        dynamic_params=dynamic_params,
        history_games=pad_season_schedule(pd.DataFrame([*played, slate])),
    )

    share = live["prior_participation_rate"]

    assert share.notna().all()
    assert share.between(0.0, 1.0).all()
