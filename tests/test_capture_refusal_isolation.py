"""What one game's unservable capture may cost the rest of the slate.

The claim held across this suite is one sentence: a refusal costs the game it
belongs to, and no other game.

``select_capture_bundle`` decides one game's capture and refuses per game.
``ProspectiveSnapshotClient.__init__`` then selected every game of the slate
in a single dict comprehension, so the first game that refused aborted
construction and no game in the slate was priced. The blast radius of a
per-game refusal was the whole slate.

That is not hypothetical and it reframes 2026-10-23. Exactly three games lost
their predictions that evening and the crash was in this client's
construction inside ``scripts/10_predict_slate.py``, so one conflicted window
is sufficient to explain all three losses, with the other two as collateral
rather than as three independent duplicates. Neither reading has been
confirmed against the production capture log, which this suite does not read
and does not need to: the mechanism is reproducible here from first
principles.

The sibling suite ``test_duplicate_capture_resolution`` establishes when a
second record of one window is one capture recorded twice and when it is two
captures of one window. The second case must keep failing closed, and this
suite is about what that refusal is allowed to cost. So the two cases this
file separates are:

  - a game whose capture cannot be selected, for any of the permanent reasons
    production refuses for, which must not be served and must be named;
  - every other game of the same slate, which is unaffected and must be
    served exactly as it is served when no game of the slate refuses.

A game excluded this way must not be a quietly shorter slate. It is recorded
with the reason ``select_capture_bundle`` gave, reachable through
``excluded_games``, and the serving entry point refuses the run rather than
writing a release that is missing a game and does not say so.

Every capture here is built with ``t20_capture_fixture``, which writes the
components and the ``capture_run`` envelope exactly as the collector writes
them. Nothing in this file reaches balldontlie, writes outside ``tmp_path``,
or runs any model, pricing or publication code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nba_prop_quant.prospective_snapshot import (
    ProspectiveSnapshotClient,
    select_capture_bundle,
)

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from t20_capture_fixture import (  # noqa: E402
    SLATE,
    at,
    game,
    prop,
    write_capture,
)

#: The game whose capture is made unservable, and the one that is left alone.
CONFLICTED = 101
CLEAN = 202

#: Both games tip after every capture instant used here, so nothing in this
#: file is refused for being post-tip unless a test asks for that.
TIP = at(23, 0)


def write_clean_game(snapshot_dir, *, game_id=CLEAN, captured_at=at(22, 42)):
    """One eligible scheduled T-20 capture for a game nothing is wrong with.

    Captured at its own instant rather than sharing one with another game's
    capture: two captures written under one ``captured_at`` interleave their
    component rows, and because ``_component_records`` matches rows by
    ``captured_at`` equality each would then recompute the other's content
    into its own digests and both would fail the digest check. That is the
    capture claim's reasoning showing up from the other side, and a test that
    shared an instant would be asserting digest corruption rather than
    isolation.
    """
    return write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=captured_at.isoformat(),
        window_ids=[f"{game_id}:T-20m"],
        due_game_ids=[game_id],
        games=[game(game_id, TIP)],
        props=[prop(game_id, 9000 + game_id)],
    )


def write_conflicting_pair(snapshot_dir, *, game_id=CONFLICTED):
    """Two captures of one window, a minute apart, over a market that moved.

    The permanent refusal this suite is mostly about. They disagree on the
    instant and on the ``player_props`` digest, so they are two point-in-time
    snapshots of one window and nothing can decide which of them T-20 pricing
    was supposed to read.
    """
    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 40).isoformat(),
        window_ids=[f"{game_id}:T-20m"],
        due_game_ids=[game_id],
        games=[game(game_id, TIP)],
        props=[prop(game_id, 1)],
    )

    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 41).isoformat(),
        window_ids=[f"{game_id}:T-20m"],
        due_game_ids=[game_id],
        games=[game(game_id, TIP)],
        props=[prop(game_id, 2)],
    )


def write_post_tip_capture(snapshot_dir, *, game_id=CONFLICTED):
    """One capture, taken after the game it names had already tipped.

    The second permanent refusal. A later run cannot repair it either: the
    tip is in the past and no capture of this window will ever be pre-tip
    again, so it isolates or it costs the slate forever.
    """
    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(23, 30).isoformat(),
        window_ids=[f"{game_id}:T-20m"],
        due_game_ids=[game_id],
        games=[game(game_id, at(23, 0))],
        props=[prop(game_id, 3)],
    )


def client_for(snapshot_dir):
    return ProspectiveSnapshotClient(
        snapshot_dir=snapshot_dir,
        target_date=SLATE,
        offset_minutes=20,
    )


# ----------------------------------------------------------------------
# the defect: one game's refusal cost every game of the slate
# ----------------------------------------------------------------------


def test_a_clean_game_is_served_alongside_a_conflicted_one(tmp_path):
    """The reproduction. One conflicted game, one clean game, one slate.

    Selected on its own the clean game is servable, and that is not in
    question. What this asserts is that it is still servable in company:
    building the client over a slate that also holds an unservable game
    returns the clean game's bundle rather than raising the other game's
    error.
    """
    snapshot_dir = tmp_path / "snapshots"

    write_conflicting_pair(snapshot_dir)

    write_clean_game(snapshot_dir)

    # Servable on its own, which is the premise the rest of this rests on.
    alone = select_capture_bundle(
        snapshot_dir,
        date=SLATE,
        game_id=CLEAN,
        offset_minutes=20,
    )

    client = client_for(snapshot_dir)

    assert sorted(client.lineage_frame()["game_id"]) == [CLEAN]

    served = client.lineage_for_game(CLEAN)

    assert served["gate3_capture_id"] == alone.capture_id

    assert [int(row["id"]) for row in client.games()] == [CLEAN]


def test_the_conflicted_game_is_named_with_the_reason_selection_gave(tmp_path):
    """An excluded game is a visible, attributable absence.

    A quietly shorter slate is a worse failure than the one being fixed, so
    the exclusion carries the whole refusal ``select_capture_bundle`` raised
    -- the window, how many records it found, how many captures they denote
    and the fields they differ on -- not a summary of it.
    """
    snapshot_dir = tmp_path / "snapshots"

    write_conflicting_pair(snapshot_dir)

    write_clean_game(snapshot_dir)

    excluded = client_for(snapshot_dir).excluded_games()

    assert sorted(excluded) == [CONFLICTED]

    reason = excluded[CONFLICTED]

    assert reason.startswith(
        f"{CONFLICTED}:T-20m: expected exactly one scheduled capture, found 2"
    )

    assert "2 different point-in-time captures" in reason

    assert "the player_props digest" in reason

    assert "captured_at" in reason


def test_nothing_prices_the_conflicted_game(tmp_path):
    """Isolation is not permission. The excluded game is still refused.

    Every route a caller has to a game's records has to stay shut for a game
    whose capture could not be selected, or isolating the refusal would have
    turned "this game cannot be priced" into "this game is priced off
    whatever happened to be readable".
    """
    snapshot_dir = tmp_path / "snapshots"

    write_conflicting_pair(snapshot_dir)

    write_clean_game(snapshot_dir)

    client = client_for(snapshot_dir)

    with pytest.raises(RuntimeError):
        client.lineage_for_game(CONFLICTED)

    with pytest.raises(RuntimeError):
        client.live_player_props(CONFLICTED)

    with pytest.raises(RuntimeError):
        client.lineups([CONFLICTED])

    assert CONFLICTED not in set(client.lineage_frame()["game_id"])

    assert CONFLICTED not in {int(row["id"]) for row in client.games()}

    # Nothing of the excluded game's content reaches the slate's records
    # through the collection accessors either.
    assert client.live_player_props(CLEAN) != []

    for row in client.lineups([CLEAN]):
        assert int(row["game_id"]) == CLEAN


def test_a_post_tip_capture_is_isolated_the_same_way(tmp_path):
    """The second permanent refusal isolates like the first.

    A duplicate conflict and a post-tip capture are unrelated defects with
    one thing in common: neither is repairable by a later run, so under the
    old construction either one cost the slate permanently. The isolation is
    a property of how the client selects, not of which refusal was raised,
    and this asserts that by reaching it through a different check.
    """
    snapshot_dir = tmp_path / "snapshots"

    write_post_tip_capture(snapshot_dir)

    write_clean_game(snapshot_dir)

    client = client_for(snapshot_dir)

    assert sorted(client.excluded_games()) == [CONFLICTED]

    assert "tip" in client.excluded_games()[CONFLICTED]

    assert [int(row["id"]) for row in client.games()] == [CLEAN]

    with pytest.raises(RuntimeError):
        client.live_player_props(CONFLICTED)


def test_two_different_refusals_in_one_slate_are_both_named(tmp_path):
    """Three games, two unservable for unrelated reasons, one served.

    The isolation is per game rather than "skip the first bad one", so a
    slate holding both permanent refusals still serves what it can and names
    both of what it cannot, each with its own reason.
    """
    snapshot_dir = tmp_path / "snapshots"

    write_conflicting_pair(snapshot_dir, game_id=CONFLICTED)

    write_post_tip_capture(snapshot_dir, game_id=303)

    write_clean_game(snapshot_dir)

    excluded = client_for(snapshot_dir).excluded_games()

    assert sorted(excluded) == [CONFLICTED, 303]

    assert "expected exactly one scheduled capture" in excluded[CONFLICTED]

    assert "tip" in excluded[303]

    assert [
        int(row["id"]) for row in client_for(snapshot_dir).games()
    ] == [CLEAN]


# ----------------------------------------------------------------------
# a slate with nothing wrong with it is unaffected
# ----------------------------------------------------------------------


def test_a_slate_with_no_refusal_behaves_exactly_as_before(tmp_path):
    """The common case is not changed by the uncommon one.

    Asserted field for field over the whole lineage frame and the whole
    bundle mapping rather than by spot check, because "the clean path still
    works" is the claim that stops an isolation fix from becoming a serving
    change.
    """
    snapshot_dir = tmp_path / "snapshots"

    first = write_clean_game(snapshot_dir, game_id=CLEAN)

    second = write_clean_game(
        snapshot_dir,
        game_id=303,
        captured_at=at(22, 43),
    )

    client = client_for(snapshot_dir)

    assert client.excluded_games() == {}

    assert sorted(client.lineage_frame()["game_id"]) == [CLEAN, 303]

    assert client.lineage_for_game(CLEAN)["gate3_capture_id"] == first

    assert client.lineage_for_game(303)["gate3_capture_id"] == second

    for game_id in (CLEAN, 303):
        direct = select_capture_bundle(
            snapshot_dir,
            date=SLATE,
            game_id=game_id,
            offset_minutes=20,
        )

        through = client.lineage_for_game(game_id)

        assert through["gate3_capture_id"] == direct.capture_id

        assert through["gate3_captured_at_utc"] == direct.captured_at

        assert through["gate3_capture_window_id"] == direct.window_id

        assert client.live_player_props(game_id) == [
            row
            for row in direct.records["player_props"]
            if int(row["game_id"]) == game_id
        ]


def test_an_empty_slate_still_has_nothing_to_exclude(tmp_path):
    """No capture log at all is not an exclusion, it is an empty slate."""
    client = client_for(tmp_path / "snapshots")

    assert client.excluded_games() == {}

    assert client.games() == []

    assert client.lineage_frame().empty


# ----------------------------------------------------------------------
# what isolation may not do
# ----------------------------------------------------------------------


def test_every_eligibility_check_still_runs_over_the_resolved_record(
    tmp_path,
):
    """Isolation does not make an ineligible capture eligible.

    One game per refusal production states, all in one slate, with one clean
    game among them. Each is excluded rather than served, which is the
    difference between "the refusal costs only its own game" and "the refusal
    was relaxed".
    """
    snapshot_dir = tmp_path / "snapshots"

    # A sweep that recorded an error.
    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 30).isoformat(),
        window_ids=["401:T-20m"],
        due_game_ids=[401],
        games=[game(401, TIP)],
        props=[prop(401, 4)],
        error_count=1,
    )

    # A capture whose recorded digest is not the content on disk.
    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 31).isoformat(),
        window_ids=["402:T-20m"],
        due_game_ids=[402],
        games=[game(402, TIP)],
        props=[prop(402, 5)],
        component_overrides={"player_props": "0" * 64},
    )

    # A capture whose games payload does not contain the target game.
    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 32).isoformat(),
        window_ids=["403:T-20m"],
        due_game_ids=[403],
        games=[game(404, TIP)],
        props=[prop(403, 6)],
    )

    # A capture taken after its game tipped.
    write_post_tip_capture(snapshot_dir, game_id=405)

    write_clean_game(snapshot_dir)

    client = client_for(snapshot_dir)

    assert sorted(client.excluded_games()) == [401, 402, 403, 405]

    assert [int(row["id"]) for row in client.games()] == [CLEAN]


def test_an_unreadable_log_is_not_isolated_into_one_game(tmp_path):
    """A log that cannot be read is not a per-game problem.

    ``select_capture_bundle`` raises ``RuntimeError`` for every reason a
    capture is ineligible, and those are the refusals that belong to one
    game. Anything else -- a truncated line, a window id that is not a
    number, a component file that is not JSON -- says the log itself cannot
    be trusted, and swallowing that per game would turn a corrupt slate into
    a quietly short one.
    """
    snapshot_dir = tmp_path / "snapshots"

    write_clean_game(snapshot_dir)

    path = snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"

    path.write_text(
        path.read_text(encoding="utf-8") + "{ this is not json\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception) as caught:
        client_for(snapshot_dir)

    assert not isinstance(caught.value, RuntimeError)


def test_isolating_a_refusal_writes_nothing(tmp_path):
    """The capture log is evidence. Reading it does not repair it.

    No capture is relabelled, re-timed, merged, synthesized or widened, and
    the append-only log a later audit reads is byte-for-byte the log the
    sweeps left.
    """
    snapshot_dir = tmp_path / "snapshots"

    write_conflicting_pair(snapshot_dir)

    write_clean_game(snapshot_dir)

    before = {
        path: path.read_bytes()
        for path in sorted(snapshot_dir.rglob("*.jsonl"))
    }

    client = client_for(snapshot_dir)

    client.excluded_games()

    client.lineage_frame()

    after = {
        path: path.read_bytes()
        for path in sorted(snapshot_dir.rglob("*.jsonl"))
    }

    assert after == before


def test_the_excluded_mapping_is_ordered_and_is_a_copy(tmp_path):
    """Named absences read the same way every time and cannot be edited away.

    Sorted by game id so a receipt or a log line naming them is stable, and
    returned as a copy so a caller cannot quietly drop a game from the record
    of what was not served.
    """
    snapshot_dir = tmp_path / "snapshots"

    write_post_tip_capture(snapshot_dir, game_id=303)

    write_conflicting_pair(snapshot_dir, game_id=CONFLICTED)

    write_post_tip_capture(snapshot_dir, game_id=202)

    client = client_for(snapshot_dir)

    assert list(client.excluded_games()) == [CONFLICTED, 202, 303]

    taken = client.excluded_games()

    taken.clear()

    assert list(client.excluded_games()) == [CONFLICTED, 202, 303]
