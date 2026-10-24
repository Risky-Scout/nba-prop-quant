"""What a second record of one T-20 capture window may and may not mean.

The claim held across this suite is one sentence: duplicate records of the
same legitimate capture resolve deterministically, and genuinely conflicting
captures still fail closed.

``select_capture_bundle`` required exactly one scheduled capture per window
and refused two, which was right about the case it was written for and wrong
about the one it could not distinguish. On 2026-10-23 that refusal cost three
games every prediction they would have had, and it was permanent: the capture
log is append-only, so a window with two records stays a window with two
records for the rest of the slate's life.

The exclusive capture claim stops two sweeps from recording one window again.
It does nothing for the records already on disk, and nothing for a second
record arriving by a route no claim covers -- a replayed log, a restored
backup, a manual sweep, two hosts, an NFS lock that silently does not lock.
So the refusal has to be able to tell two cases apart:

  - two records of the *same* capture, which denote one point-in-time
    snapshot and are interchangeable, so serving may proceed;
  - two records of *different* captures of one window, which denote two
    snapshots and are not, so serving must refuse rather than price a game
    off whichever of them a reader happened to see first.

Every capture here is built with ``t20_capture_fixture``, which writes the
components and the ``capture_run`` envelope exactly as the collector writes
them, so a capture accepted here is accepted by the checks that accept a
production one. Nothing in this file reaches balldontlie, writes outside
``tmp_path``, or runs any model, pricing or publication code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from nba_prop_quant.prospective_snapshot import (
    PRIMARY_OFFSET_MINUTES,
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

#: The one game most tests here capture, and the window it is captured for.
GAME_ID = 101

WINDOW_ID = f"{GAME_ID}:T-20m"


def capture_run_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "capture_runs" / f"{SLATE}.jsonl"


def capture_run_lines(snapshot_dir: Path) -> list[str]:
    return [
        line
        for line in capture_run_path(snapshot_dir)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def rewrite_capture_log(snapshot_dir: Path, rows: list[dict]) -> None:
    capture_run_path(snapshot_dir).write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def one_capture(
    snapshot_dir: Path,
    *,
    captured_at: str | None = None,
    tip: tuple[int, int] = (23, 0),
    props: list[dict] | None = None,
) -> str:
    """One eligible scheduled T-20 capture of ``GAME_ID``."""
    return write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=captured_at or at(22, 40).isoformat(),
        window_ids=[WINDOW_ID],
        due_game_ids=[GAME_ID],
        games=[game(GAME_ID, at(*tip))],
        props=[prop(GAME_ID, 601)] if props is None else props,
    )


def replay_the_capture_log(snapshot_dir: Path) -> None:
    """Append every capture-run row of this slate again, byte for byte.

    The duplication route that has nothing to do with capturing twice. The
    components are untouched and only the envelope is repeated, which is what
    a replayed log, a restored backup concatenated onto the live file and an
    append retried after a partial write all leave behind: two records of one
    capture, carrying identical instants, window sets, due-game sets and
    component digests, because they *are* one capture.
    """
    lines = capture_run_lines(snapshot_dir)

    with capture_run_path(snapshot_dir).open(
        "a", encoding="utf-8"
    ) as handle:
        for line in lines:
            handle.write(line + "\n")


def rewrite_the_capture_log_backwards(snapshot_dir: Path) -> None:
    """The same rows, in the other order.

    Not a thing production does -- the log is append-only and nothing here
    asks it to be anything else. It is how a test states "the selection does
    not depend on which record is read first", because the only other way to
    say it would be to assert something about the implementation.
    """
    lines = capture_run_lines(snapshot_dir)

    capture_run_path(snapshot_dir).write_text(
        "\n".join(reversed(lines)) + "\n", encoding="utf-8"
    )


def bundle_of(snapshot_dir: Path):
    return select_capture_bundle(
        snapshot_dir,
        date=SLATE,
        game_id=GAME_ID,
        offset_minutes=PRIMARY_OFFSET_MINUTES,
    )


# ----------------------------------------------------------------------
# the same capture, recorded twice
# ----------------------------------------------------------------------


def test_one_capture_recorded_twice_is_served_rather_than_refused(tmp_path):
    """The production symptom, over records that denote one snapshot.

    Before this change the second record made the window unpriceable with
    ``101:T-20m: expected exactly one scheduled capture, found 2`` -- the
    message that cost 2026-10-23 three games, with 101 in place of the game
    id. The two records agree on the instant, on the windows, on the due
    games and on all five component digests, so they are not two captures
    about which something has to be decided. They are one capture, written
    down twice.
    """
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir)

    replay_the_capture_log(snapshot_dir)

    assert len(capture_run_lines(snapshot_dir)) == 2

    bundle = bundle_of(snapshot_dir)

    assert bundle.window_id == WINDOW_ID

    assert bundle.captured_at == at(22, 40).isoformat()

    assert bundle.game["id"] == GAME_ID


def test_the_resolved_capture_is_the_capture_itself(tmp_path):
    """Resolving a duplicate returns the capture, not a reconstruction.

    Nothing may be relabelled, re-timed, merged or synthesized on the way
    out: the bundle a duplicated log produces has to be the bundle the single
    record produces, field for field, including the component rows recomputed
    off disk and the capture id the lineage is provenanced with.
    """
    single = tmp_path / "single" / "snapshots"

    duplicated = tmp_path / "duplicated" / "snapshots"

    capture_id = one_capture(single)

    assert one_capture(duplicated) == capture_id

    replay_the_capture_log(duplicated)

    assert bundle_of(duplicated) == bundle_of(single)

    assert bundle_of(duplicated).capture_id == capture_id


def test_the_chosen_capture_does_not_depend_on_the_order_of_the_log(tmp_path):
    """Determinism, stated the only way a test can state it.

    Reversing the rows changes which record a reader sees first and changes
    nothing else. If the selection were "the first candidate" this would be
    the same answer by luck; it is the same answer because the candidates are
    ordered by their own canonical content rather than by arrival.
    """
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir)

    replay_the_capture_log(snapshot_dir)

    forwards = bundle_of(snapshot_dir)

    rewrite_the_capture_log_backwards(snapshot_dir)

    assert bundle_of(snapshot_dir) == forwards


def test_reading_the_same_log_twice_resolves_it_the_same_way(tmp_path):
    """Idempotency at the layer serving reads the capture through.

    Two ``ProspectiveSnapshotClient`` constructions over one unchanged log
    discover the same games, resolve the same capture and carry the same
    provenance, so two serving runs over a duplicated log price the same
    snapshot and publish the same release rather than two.
    """
    snapshot_dir = tmp_path / "snapshots"

    capture_id = one_capture(snapshot_dir)

    replay_the_capture_log(snapshot_dir)

    lineages = []

    for _ in range(2):
        client = ProspectiveSnapshotClient(
            snapshot_dir=snapshot_dir,
            target_date=SLATE,
            offset_minutes=PRIMARY_OFFSET_MINUTES,
        )

        assert [row["id"] for row in client.games()] == [GAME_ID]

        lineages.append(client.lineage_for_game(GAME_ID))

    assert lineages[0] == lineages[1]

    assert lineages[0]["gate3_capture_id"] == capture_id


def test_a_whole_slate_of_duplicated_records_resolves(tmp_path):
    """Three games sharing one window, which is 2026-10-23's shape.

    The three games that tipped at 02:00 UTC shared a T-20 capture naming all
    three windows, so a second record of that capture refused all three games
    at once rather than one of them.
    """
    snapshot_dir = tmp_path / "snapshots"

    game_ids = [21717943, 21717944, 21717945]

    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 40).isoformat(),
        window_ids=[f"{game_id}:T-20m" for game_id in game_ids],
        due_game_ids=game_ids,
        games=[
            game(game_id, at(23, 0), home=index, away=index + 10)
            for index, game_id in enumerate(game_ids, start=1)
        ],
        props=[
            prop(game_id, 600 + index)
            for index, game_id in enumerate(game_ids)
        ],
    )

    replay_the_capture_log(snapshot_dir)

    client = ProspectiveSnapshotClient(
        snapshot_dir=snapshot_dir,
        target_date=SLATE,
        offset_minutes=PRIMARY_OFFSET_MINUTES,
    )

    assert sorted(row["id"] for row in client.games()) == game_ids


# ----------------------------------------------------------------------
# two captures, one window
# ----------------------------------------------------------------------


def test_two_snapshots_of_one_window_are_still_refused(tmp_path):
    """The case the refusal exists for, and it still refuses.

    Two sweeps a minute apart produce two genuine point-in-time captures of
    one window. Both are real data, both verify, and nothing in the data
    model says which of them T-20 pricing is entitled to read -- so serving
    either would be pricing a game off a snapshot nobody chose. The refusal
    is the correct outcome and is unchanged.
    """
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir)

    one_capture(snapshot_dir, captured_at=at(22, 41).isoformat())

    with pytest.raises(RuntimeError) as error:
        bundle_of(snapshot_dir)

    message = " ".join(str(error.value).split())

    # The refusal production already had, intact.
    assert message.startswith(
        f"{WINDOW_ID}: expected exactly one scheduled capture, found 2"
    )

    # And what it could not say before: that the two are two captures, and
    # which recorded facts about them differ.
    assert "2 different point-in-time captures" in message

    assert "captured_at" in message

    assert at(22, 40).isoformat() in message

    assert at(22, 41).isoformat() in message

    assert "component_sha256" in message

    assert "capture_id" in message


def test_two_captures_of_one_instant_with_different_content_are_refused(
    tmp_path,
):
    """Same instant, different content, which is the harder conflict.

    A reader comparing only timestamps would call these one capture. They
    carry different props and therefore different ``player_props`` digests,
    so they are two snapshots of the same moment and the refusal has to name
    the digest rather than the instant.
    """
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir, props=[prop(GAME_ID, 601)])

    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 40).isoformat(),
        window_ids=[WINDOW_ID],
        due_game_ids=[GAME_ID],
        games=[game(GAME_ID, at(23, 0))],
        props=[prop(GAME_ID, 601), prop(GAME_ID, 602)],
    )

    with pytest.raises(RuntimeError) as error:
        bundle_of(snapshot_dir)

    message = " ".join(str(error.value).split())

    assert "2 different point-in-time captures" in message

    assert "component_sha256" in message


def test_a_record_that_copied_a_capture_id_onto_other_fields_is_refused(
    tmp_path,
):
    """Equivalence is the preimage, not the digest the row claims.

    ``capture_id`` is a claim an envelope makes about itself. A record
    carrying a copied id over a different instant would read as a duplicate
    of the record it copied from if the id were what was compared, and
    something would then have chosen between two snapshots before the
    capture-id check ever ran. The fields the id is derived from are what is
    compared, so this is a conflict.
    """
    snapshot_dir = tmp_path / "snapshots"

    capture_id = one_capture(snapshot_dir)

    one_capture(snapshot_dir, captured_at=at(22, 41).isoformat())

    rows = [json.loads(line) for line in capture_run_lines(snapshot_dir)]

    rows[1]["payload"]["capture_id"] = capture_id

    rewrite_capture_log(snapshot_dir, rows)

    with pytest.raises(RuntimeError) as error:
        bundle_of(snapshot_dir)

    message = " ".join(str(error.value).split())

    assert "2 different point-in-time captures" in message

    assert "captured_at" in message


def test_a_duplicate_that_disagrees_about_collection_errors_is_refused(
    tmp_path,
):
    """The one field outside the digest that still decides eligibility.

    ``select_capture_bundle`` refuses a capture whose sweep recorded any
    error, and ``build_capture_id`` does not digest ``error_count``. So two
    records can agree on every byte of content and disagree about whether the
    sweep that produced it was clean, and exactly one of them is servable.
    Which one cannot be decided here, so neither is.
    """
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir)

    rows = [json.loads(line) for line in capture_run_lines(snapshot_dir)]

    rows.append(json.loads(json.dumps(rows[0])))

    rows[1]["payload"]["error_count"] = 1

    rewrite_capture_log(snapshot_dir, rows)

    with pytest.raises(RuntimeError) as error:
        bundle_of(snapshot_dir)

    assert "error_count" in " ".join(str(error.value).split())


def test_three_records_of_two_captures_are_refused(tmp_path):
    """A duplicate alongside a conflict is still a conflict.

    Collapsing the equivalent records and then serving the survivor would be
    choosing between two snapshots by majority, which is not a thing the data
    model supports.
    """
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir)

    one_capture(snapshot_dir, captured_at=at(22, 41).isoformat())

    replay_the_capture_log(snapshot_dir)

    assert len(capture_run_lines(snapshot_dir)) == 4

    with pytest.raises(RuntimeError) as error:
        bundle_of(snapshot_dir)

    message = " ".join(str(error.value).split())

    assert "found 4" in message

    assert "2 different point-in-time captures" in message


# ----------------------------------------------------------------------
# everything the resolution does not reach
# ----------------------------------------------------------------------


def test_a_window_with_no_scheduled_capture_still_reports_none(tmp_path):
    """The backstop production has, unmoved.

    Resolution only ever narrows two records of one capture to that capture.
    A window with no scheduled capture is refused by the same count check
    that has always refused it, with the same message, because nothing about
    duplicates has anything to say about absence.
    """
    snapshot_dir = tmp_path / "snapshots"

    write_capture(
        snapshot_dir,
        date=SLATE,
        captured_at=at(22, 40).isoformat(),
        window_ids=[WINDOW_ID],
        due_game_ids=[GAME_ID],
        games=[game(GAME_ID, at(23, 0))],
        props=[prop(GAME_ID, 601)],
        reason="manual",
    )

    with pytest.raises(RuntimeError) as error:
        bundle_of(snapshot_dir)

    assert "expected exactly one scheduled capture, found 0" in " ".join(
        str(error.value).split()
    )


def test_a_duplicated_post_tip_capture_is_still_post_tip(tmp_path):
    """Resolution cannot make an ineligible capture eligible.

    Every check after the selection runs over the capture that was resolved,
    so a duplicate of a capture pricing would refuse is refused for the
    reason the capture itself is refused. A later snapshot does not become a
    T-20 capture by being written down twice.
    """
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir, tip=(22, 30))

    replay_the_capture_log(snapshot_dir)

    with pytest.raises(RuntimeError, match="is not pre-tip"):
        bundle_of(snapshot_dir)


def test_a_duplicated_capture_still_has_its_digests_recomputed(tmp_path):
    """The component check is not weakened by the record count either."""
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir)

    replay_the_capture_log(snapshot_dir)

    props_path = snapshot_dir / "player_props" / f"{SLATE}.jsonl"

    with props_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "captured_at": at(22, 40).isoformat(),
                    "snapshot_type": "live_player_prop",
                    "payload": prop(GAME_ID, 999),
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    with pytest.raises(RuntimeError, match="component hash mismatch"):
        bundle_of(snapshot_dir)


def test_the_capture_log_is_not_rewritten_by_resolving_it(tmp_path):
    """Both records are still there afterwards.

    The log is the append-only record of what was captured. Resolution is a
    reading of it and may not become a repair of it: a run that quietly
    dropped the second row would destroy the evidence that the duplicate ever
    happened, which is the thing an operator needs most.
    """
    snapshot_dir = tmp_path / "snapshots"

    one_capture(snapshot_dir)

    replay_the_capture_log(snapshot_dir)

    before = capture_run_path(snapshot_dir).read_bytes()

    bundle_of(snapshot_dir)

    assert capture_run_path(snapshot_dir).read_bytes() == before
