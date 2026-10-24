"""Tests for the per-game live composite: every game at its own window.

The live slate used to be published through one forward-only pointer. It
recorded the closest-to-tip window the *slate* had reached and declined any
later refresh at a larger offset, which reads as a sensible ordering rule
until you notice that an offset is measured from a game's own tip and games
do not tip together. On a staggered evening the windows interleave: the first
game's T-20 capture is taken before the last game's T-180 capture, so one
game reaching T-20 silenced every earlier window of every later game for
the rest of the night.

:func:`test_the_one_pointer_would_have_declined_fourteen_of_the_twenty_four`
measures that on a reconstruction of the real 2026-10-24 slate rather than
asserting it, and the rest of this module is about the design that replaced
it.

*Each game stands at its own latest validated window.* No card before its
first capture; T-180 when T-180 validates; T-90, T-45 and T-20 as each of
those validates; and all of that simultaneously across the slate, so a
composite really does hold one game at T-20, another at T-90 and a third at
T-180 at the same moment.

*A game is never downgraded.* A refresh at a larger offset than the window a
game is already serving leaves that game exactly as it is, and does not stop
the refresh from advancing the games it is actually for.

*A game nobody advanced is never altered.* Advancing one game leaves every
other game's rows byte-identical -- asserted on the bytes at the destination,
not on a count of them.

*Composition is selection, never computation.* Every row is copied out of a
release a canary already read back and attested, and the composite proves it
by re-reading the stored source bytes, recomputing their digest, putting each
one through the publisher's own canary gate and looking every composed row up
by the publisher's own logical key in what came back.
``COMPOSITE_UNATTESTED_ROWS``, ``LIVE_SOURCE_MISMATCH_COUNT`` and
``COMPOSITE_UNATTESTED_DISTRIBUTIONS`` all have to be zero.

*The last known-good composite survives every failure.* A missing source, a
source whose bytes were edited, an attestation of other bytes, a game that
has tipped, a public readback that serves something else: each is injected
separately against a destination that is already serving a valid composite,
and in every case the bytes a reader is served afterwards are byte-for-byte
the ones they were served before.

*The post-tip guard is per game.* A started game is refused new publication
and the rest of the slate still goes, and the card that game had before it
started is carried unchanged rather than retracted.

The publisher is real throughout, the canary half is published by the real
publication orchestrator against the committed switch, and the destinations
are local directories in a pytest temporary directory. Nothing here can
reach the public site: the live destination is a directory, and the live
approval is a token forged for these tests rather than the committed one.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "ops"))

sys.path.insert(0, str(PROJECT / "tests"))

import live_slate_composite as composite  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import run_progressive_live_refresh as live  # noqa: E402
import run_wizardofodds_publication as publication  # noqa: E402
import wizardofodds_publisher_fixture as fixture  # noqa: E402

#: The live approval this repository cannot grant itself. Forged, and
#: deliberately not the committed token: a test that passed because it reused
#: the real one would be testing the switch file rather than the gate.
TOKEN = "the-approval-this-composite-test-forged"

#: The evening under test. Eastern is UTC-4 on this date -- daylight time runs
#: to 1 November 2026 -- and the offset is written out rather than looked up so
#: that these tests do not depend on a timezone database.
SLATE_DATE = "2026-10-24"

EASTERN = timezone(timedelta(hours=-4))

#: Two published names per game. Shaped like people, because the publisher
#: refuses a name that is an identifier wearing a person's slot and these
#: tests are not about that refusal.
SEAT_NAMES = ("Alpha Example", "Beta Example")

#: How many releases a real advisory canary destination keeps. Read off the
#: committed destinations record rather than restated, because the whole
#: reason the attested source store is durable is that this number is small:
#: a cumulative offset publishes one release per tip group, so by the third
#: publication the release the first group's card came out of is gone from
#: the canary host and only a local copy can still produce it.
CANARY_RETENTION: int = json.loads(
    (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
)["named_destinations"]["wizardofodds_nba_advisory_t180m_canary"][
    "retained_releases"
]


@dataclass(frozen=True)
class Game:
    """One game of the slate: who is playing and when it starts."""

    game_id: int
    away: str
    home: str
    tip_et: str

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def tip(self) -> datetime:
        hour, minute = (int(part) for part in self.tip_et.split(":"))

        return datetime(
            2026, 10, 24, hour, minute, tzinfo=EASTERN
        ).astimezone(UTC)

    def window(self, offset_minutes: int) -> str:
        return f"{self.game_id}:T-{int(offset_minutes)}m"

    def captured_at(self, offset_minutes: int) -> datetime:
        """When this game's window at ``offset_minutes`` is captured."""
        return self.tip - timedelta(minutes=int(offset_minutes))


#: A reconstruction of the real 2026-10-24 slate: eight games over six tip
#: times from 6:00 PM to 10:00 PM Eastern. The tip times are what the defect
#: is about -- two pairs of games share a tip, which is why eight games make
#: twenty-four window groups rather than thirty-two -- so they are the part of
#: the reconstruction that has to be faithful. The teams and the prices are
#: not, and are the fixture's own.
SLATE: tuple[Game, ...] = (
    Game(4001, "BOS", "NYK", "18:00"),
    Game(4002, "MIA", "ORL", "19:00"),
    Game(4003, "ATL", "CHI", "19:00"),
    Game(4004, "DET", "CLE", "19:30"),
    Game(4005, "TOR", "PHI", "19:30"),
    Game(4006, "MEM", "DAL", "20:00"),
    Game(4007, "DEN", "PHX", "20:30"),
    Game(4008, "LAL", "GSW", "22:00"),
)

#: Three of them, for the tests that are about one invariant rather than about
#: the whole evening. Chosen so that their tips are within three hours of each
#: other, which is what lets one refresh name all three of them before any has
#: started: a game's T-180 capture is taken three hours before its own tip, so
#: a trio spread further than that has no moment at which every one of its
#: T-180 windows is pregame. The slate as a whole does not have that property
#: and is not meant to -- that interleaving is the defect -- but a test about
#: one invariant should not have to pick its way around it.
TRIO: tuple[Game, ...] = (SLATE[1], SLATE[3], SLATE[5])


def priced_frame_for(
    games: tuple[Game, ...], *, odds: int
) -> pd.DataFrame:
    """A priced slate of these games, at this price.

    Every column is the fixture's, taken off one of its rows and then given
    this game's identity, so a column the integration contract gains has to be
    added in one place rather than two. The price is the caller's because a
    window's release has to differ from the window before it for a replacement
    to be a real replacement.
    """
    template = fixture.priced_frame().iloc[0].to_dict()

    rows: list[dict[str, Any]] = []

    for game in games:
        for seat in (0, 1):
            row = dict(template)

            row.update(
                {
                    "id": game.game_id * 100 + seat,
                    "vendor": "fixturebook",
                    "game_id": game.game_id,
                    "date": pd.Timestamp(SLATE_DATE),
                    "player_id": game.game_id * 10 + seat + 1,
                    "player_name": SEAT_NAMES[seat],
                    "team_id": game.game_id * 2 + seat,
                    "opponent_id": game.game_id * 2 + (1 - seat),
                    "is_home": 1 - seat,
                    "home_team_abbreviation": game.home,
                    "visitor_team_abbreviation": game.away,
                    "player_team_abbreviation": (
                        game.home if seat == 0 else game.away
                    ),
                    "tip_off_utc": game.tip.isoformat(),
                    "prop_type": "points",
                    "line_value": 18.5 + seat,
                    "model_projection": 18.9 + seat,
                    "over_odds": odds,
                    "under_odds": -odds - 20,
                }
            )

            rows.append(row)

    return pd.DataFrame(rows)


class Slate:
    """One evening: N canary destinations and one live destination.

    The canary half is published by the real publication orchestrator against
    the committed switch, which enables canary. The live half is invoked
    through :func:`run_progressive_live_refresh.main` exactly as the controller
    invokes it, so what is under test includes the receipt it writes, the exit
    code it returns and the per-game ledger it appends to.
    """

    def __init__(self, root: Path, games: tuple[Game, ...] = SLATE) -> None:
        self.root = Path(root)

        self.games = games

        self.by_id = {str(game.game_id): game for game in games}

        self.live_root = self.root / "live"

        self.slate_root = self.root / "slate"

        self.switch_path = self.root / "switch.json"

        self.destinations_path = self.root / "destinations.json"

        self.built: dict[str, Any] | None = None

        self.serves = 0

        self._write_switch()

        self._write_destinations()

    # -- the forged live approval -------------------------------------

    def _write_switch(self) -> None:
        document = json.loads(
            (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
        )

        document["modes"][publisher.MODE_LIVE] = {
            "approval_token": TOKEN,
            "reason": "forged for this test only",
            "state": publisher.STATE_ENABLED,
        }

        self.switch_path.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _write_destinations(self) -> None:
        document = json.loads(
            (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(
                encoding="utf-8"
            )
        )

        document["destinations"][publisher.MODE_LIVE] = {
            "release_prefix": "nba_prop_quant",
            "root": str(self.live_root),
            "transport": publisher.TRANSPORT_LOCAL_DIRECTORY,
        }

        self.destinations_path.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )

    def approve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(publisher.PUBLISH_ENV_VAR, publisher.STATE_ENABLED)

        monkeypatch.setenv(publisher.PUBLISH_APPROVAL_ENV_VAR, TOKEN)

    # -- the canary half ----------------------------------------------

    def canary_root(self, offset_minutes: int) -> Path:
        return self.root / f"canary_t{int(offset_minutes)}"

    def serve(self, games: tuple[Game, ...], *, odds: int) -> dict[str, Any]:
        """Price exactly these games at this price, as one serving run would.

        The priced parquet is rewritten, which is what a serve does: one
        invocation at an offset prices every game with an eligible capture at
        that offset and the file it leaves behind is the whole of that.
        """
        self.serves += 1

        self.built = fixture.build(
            self.slate_root,
            frame=priced_frame_for(games, odds=odds),
            slate_date=SLATE_DATE,
        )

        return self.built

    def canary(
        self,
        offset_minutes: int,
        games: tuple[Game, ...],
        *,
        odds: int | None = None,
    ) -> Path:
        """Serve these games at this window and publish the canary of them.

        Returns the feed receipt the publication wrote, which is what
        authorises the live refresh that follows: the live half's whole claim
        is that the bytes it publishes are bytes a canary read back.
        """
        built = self.serve(
            games,
            odds=(
                -100 - int(offset_minutes) - self.serves
                if odds is None
                else odds
            ),
        )

        receipt = publication.publish_feed(
            mode=publisher.MODE_CANARY,
            slate_date=built["slate_date"],
            data_root=built["data_root"],
            project_root=PROJECT,
            serving_receipt=built["receipt_path"],
            destination_name=None,
            destination_root=self.canary_root(offset_minutes),
        )

        path = self.root / f"canary_feed_t{int(offset_minutes)}.json"

        path.write_text(
            json.dumps(receipt, default=str, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # The retention a real advisory canary runs under. Applied here
        # because the harness addresses the canary by root rather than by
        # name, and the unnamed local destination retains everything -- which
        # would quietly make every source readable off the canary host and
        # leave the durable store untested. The publisher's own pruning, with
        # the production number.
        publisher.prune_releases(
            publisher.LocalDirectoryTransport(self.canary_root(offset_minutes)),
            retained=CANARY_RETENTION,
        )

        return path

    # -- the live half ------------------------------------------------

    def argv(
        self,
        offset_minutes: int,
        games: tuple[Game, ...],
        *,
        now: datetime,
        captures: dict[str, str] | None = None,
        receipt_path: Path | None = None,
    ) -> list[str]:
        built = self.built

        assert built is not None

        argv = [
            "--slate-date",
            SLATE_DATE,
            "--data-root",
            str(built["data_root"]),
            "--snapshot-offset-minutes",
            str(int(offset_minutes)),
            "--canary-feed-receipt",
            str(self.root / f"canary_feed_t{int(offset_minutes)}.json"),
            "--serving-receipt",
            str(built["receipt_path"]),
            "--canary-destination-root",
            str(self.canary_root(offset_minutes)),
            "--live-destination-root",
            str(self.live_root),
            "--switch-path",
            str(self.switch_path),
            "--destinations-path",
            str(self.destinations_path),
            "--project-root",
            str(PROJECT),
            "--now-utc",
            now.isoformat(),
            "--receipt-path",
            str(receipt_path if receipt_path is not None else self.receipt()),
        ]

        for game in games:
            window_id = game.window(offset_minutes)

            argv += ["--window-tip-utc", f"{window_id}={game.tip.isoformat()}"]

            identity = (captures or {}).get(
                window_id,
                f"capture-{game.game_id}-t{int(offset_minutes)}"
                f"@{game.captured_at(offset_minutes).isoformat()}",
            )

            argv += ["--window-capture", f"{window_id}={identity}"]

        return argv

    def receipt(self) -> Path:
        return self.root / "live_refresh.json"

    def refresh(
        self,
        offset_minutes: int,
        games: tuple[Game, ...],
        *,
        now: datetime | None = None,
        **overrides: Any,
    ) -> tuple[int, dict[str, Any]]:
        """Refresh live for these games at this window.

        ``now`` defaults to a minute after the last of their captures, which is
        when the controller reaches the live refresh: after the window's canary
        has published and validated, and before any of the games has tipped.
        """
        moment = (
            now
            if now is not None
            else max(game.captured_at(offset_minutes) for game in games)
            + timedelta(minutes=1)
        )

        path = Path(overrides.get("receipt_path") or self.receipt())

        code = live.main(
            self.argv(offset_minutes, games, now=moment, **overrides)
        )

        return code, json.loads(path.read_text(encoding="utf-8"))

    def window(
        self,
        offset_minutes: int,
        games: tuple[Game, ...],
        *,
        canary_games: tuple[Game, ...] | None = None,
        **overrides: Any,
    ) -> tuple[int, dict[str, Any]]:
        """Canary this window and refresh live from it.

        ``canary_games`` is the whole cumulative set the serve prices, which
        is not always the set the refresh is for: a release at an offset holds
        every game captured at that offset so far, and the refresh advances
        only the tip group it was handed.
        """
        self.canary(
            offset_minutes, canary_games if canary_games is not None else games
        )

        return self.refresh(offset_minutes, games, **overrides)

    # -- what the destination holds -----------------------------------

    def pointer(
        self, name: str = publisher.CURRENT_POINTER
    ) -> dict[str, Any] | None:
        path = self.live_root / name

        if not path.exists():
            return None

        return json.loads(path.read_text(encoding="utf-8"))

    def served_bytes(self) -> bytes:
        """The rows document a reader opening the live page is served."""
        current = self.pointer()

        assert current is not None

        return (self.live_root / str(current["rows_relative"])).read_bytes()

    def served_rows(self) -> list[dict[str, Any]]:
        return json.loads(self.served_bytes())["rows"]

    def rows_of(self, game_id: int | str) -> bytes:
        """One game's rows as the destination serves them, as bytes.

        Serialised with the publisher's own canonical serialiser, so "game B
        was not altered" is a claim about bytes rather than about whether two
        dictionaries happen to compare equal.
        """
        return publisher.serialise(
            [
                row
                for row in self.served_rows()
                if str(row.get("game_id")) == str(game_id)
            ]
        )

    def live_windows(self) -> dict[str, str]:
        """Which window each game is live at, read off the destination.

        Out of the composite manifest beside the release rather than out of a
        receipt, because the question is what the destination says it is
        serving.
        """
        current = self.pointer()

        assert current is not None

        manifest = json.loads(
            (
                self.live_root
                / composite.manifest_relative(str(current["release_id"]))
            ).read_text(encoding="utf-8")
        )

        return {
            str(entry["GAME_ID"]): str(entry["SOURCE_WINDOW"])
            for entry in manifest["games"]
        }

    def ledger(self) -> list[dict[str, Any]]:
        built = self.built

        assert built is not None

        return live.read_evidence(
            live.evidence_path(built["data_root"], SLATE_DATE)
        )


@pytest.fixture
def slate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Slate:
    built = Slate(tmp_path, SLATE)

    built.approve(monkeypatch)

    return built


@pytest.fixture
def trio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Slate:
    built = Slate(tmp_path, TRIO)

    built.approve(monkeypatch)

    return built


# ======================================================================
# what the one pointer did
# ======================================================================


@dataclass(frozen=True)
class WindowGroup:
    """One capture of one tip group at one offset: a live refresh's input."""

    captured_at: datetime
    offset_minutes: int
    tip: datetime
    games: tuple[Game, ...]

    @property
    def label(self) -> str:
        return (
            f"{self.captured_at.astimezone(EASTERN).strftime('%H:%M')} ET "
            f"T-{self.offset_minutes} "
            + "/".join(str(game.game_id) for game in self.games)
        )


def window_groups(games: tuple[Game, ...]) -> list[WindowGroup]:
    """Every window group of the evening, in the order they are captured.

    A group is one tip time at one offset, because that is what one capture,
    one serve and one live refresh are about. Games that tip together are
    captured together and so belong to the same group, which is why eight
    games over six tip times make twenty-four groups rather than thirty-two.

    Ties are broken by tip time, earliest first. Two groups share a capture
    minute only when one game's T-90 falls on another's T-180, and the
    controller reaches the earlier game first because it has been awake for
    it since the afternoon.
    """
    tips: dict[datetime, list[Game]] = {}

    for game in games:
        tips.setdefault(game.tip, []).append(game)

    groups = [
        WindowGroup(
            captured_at=tip - timedelta(minutes=offset),
            offset_minutes=offset,
            tip=tip,
            games=tuple(sorted(members, key=lambda game: game.game_id)),
        )
        for tip, members in tips.items()
        for offset in live.PROGRESSIVE_OFFSETS_MINUTES
    ]

    return sorted(groups, key=lambda group: (group.captured_at, group.tip))


def declined_by_one_pointer(groups: list[WindowGroup]) -> list[WindowGroup]:
    """Which groups a single slate-wide forward-only pointer would refuse.

    The rule as it was: one "closest offset the slate has reached" for the
    whole evening, a group at a smaller or equal offset accepted, a group at a
    larger one declined. Reimplemented here rather than kept alive in the
    entry point, so the defect is measured against the real slate's own
    schedule instead of being described in a commit message.
    """
    reached: int | None = None

    declined: list[WindowGroup] = []

    for group in groups:
        if reached is None or group.offset_minutes <= reached:
            reached = (
                group.offset_minutes
                if reached is None
                else min(reached, group.offset_minutes)
            )

            continue

        declined.append(group)

    return declined


def test_the_slate_is_eight_games_over_six_tips_and_twenty_four_groups():
    """The reconstruction's own shape, pinned before anything rests on it."""
    assert len(SLATE) == 8

    assert len({game.tip for game in SLATE}) == 6

    groups = window_groups(SLATE)

    assert len(groups) == 24

    # Four windows per tip time, and every game in exactly one group per
    # offset -- which is the "owned exactly once per offset" property the
    # whole design depends on, asserted on the schedule before it is asserted
    # on the destination.
    for offset in live.PROGRESSIVE_OFFSETS_MINUTES:
        owners = [
            game.game_id
            for group in groups
            if group.offset_minutes == offset
            for game in group.games
        ]

        assert sorted(owners) == sorted(game.game_id for game in SLATE)

    # The defect's premise: the first game's T-20 capture is taken before the
    # last game's T-180 capture.
    assert SLATE[0].captured_at(20) < SLATE[-1].captured_at(180)


def test_the_one_pointer_would_have_declined_fourteen_of_the_twenty_four():
    """The measurement the per-game composite exists because of.

    Not a story about a hypothetical slate: this is the real evening's own
    schedule run through the rule as it was, and it refuses fourteen of the
    twenty-four window groups. Every one of them is a window whose canary had
    published and validated, whose digests all agreed and whose readback would
    have passed.
    """
    groups = window_groups(SLATE)

    declined = declined_by_one_pointer(groups)

    assert len(declined) == 14, [group.label for group in declined]

    assert len(groups) - len(declined) == 10

    # And the shape of the damage: once the earliest game reaches T-20 at 5:40
    # PM -- twenty minutes before the first tip of the night -- every single
    # remaining T-180, T-90 and T-45 group of the evening is refused. The
    # later games are not merely late to the page, they cannot be refreshed
    # on it at all until twenty minutes before their own tips.
    first_t20 = SLATE[0].captured_at(20)

    assert first_t20.astimezone(EASTERN).strftime("%H:%M") == "17:40"

    after = [
        group
        for group in groups
        if group.captured_at > first_t20 and group.offset_minutes != 20
    ]

    assert after, "the reconstruction has to have later windows to suppress"

    assert set(after).issubset(set(declined))


# ======================================================================
# the two invariants
# ======================================================================


def test_a_game_already_closer_to_its_tip_is_never_downgraded(trio):
    """The no-downgrade invariant, with another game advancing beside it.

    The one-pointer rule's decline was all-or-nothing: a game being ahead
    stopped the whole refresh. Here the first game is at T-20 and a T-180
    refresh arrives for it and for a game that has no card at all, and both
    halves have to hold at once -- the one that is ahead keeps its T-20 card,
    and the one that is behind gets published.
    """
    first, middle, last = trio.games

    code, _ = trio.window(20, (first,))

    assert code == live.EXIT_OK

    assert trio.live_windows() == {str(first.game_id): "T-20m"}

    ahead = trio.rows_of(first.game_id)

    # A replay of the T-180 window after the first game has gone to T-20,
    # which is the operational case this guard is for: the live half can
    # refuse all evening while the canary publishes -- it does exactly that
    # until both approval variables are set -- so an evening's windows get
    # replayed in order from the receipts the canary stage left behind, and
    # the earlier ones must not walk back what the later ones achieved.
    code, receipt = trio.window(180, (first, last), canary_games=(first, last))

    assert code == live.EXIT_OK, receipt

    assert receipt["live_outcome"] == live.OUTCOME_REPLACED

    assert receipt["advancement"]["advancing_game_ids"] == [
        str(last.game_id)
    ]

    assert [entry["game_id"] for entry in receipt["advancement"]["declined"]] == [
        str(first.game_id)
    ]

    # The game that was ahead is still at T-20, and its rows are the bytes it
    # was already serving.
    assert trio.live_windows() == {
        str(first.game_id): "T-20m",
        str(last.game_id): "T-180m",
    }

    assert trio.rows_of(first.game_id) == ahead

    assert str(middle.game_id) not in trio.live_windows()


def test_advancing_one_game_leaves_every_other_game_byte_identical(trio):
    """The second invariant, asserted on the bytes at the destination.

    Three games at T-180, then one of them advanced to T-90. The advanced
    game's rows have to change -- it is a different window's prices -- and the
    other two have to be byte-for-byte what they were, which is the strongest
    available form of the claim because their components are reassembled out
    of stored source bytes this refresh never looked at.
    """
    first, middle, last = trio.games

    code, _ = trio.window(180, trio.games)

    assert code == live.EXIT_OK

    before = {
        game.game_id: trio.rows_of(game.game_id) for game in trio.games
    }

    served_before = trio.served_bytes()

    code, receipt = trio.window(
        90, (middle,), canary_games=trio.games
    )

    assert code == live.EXIT_OK, receipt

    assert receipt["advancement"]["advancing_game_ids"] == [
        str(middle.game_id)
    ]

    assert trio.live_windows() == {
        str(first.game_id): "T-180m",
        str(middle.game_id): "T-90m",
        str(last.game_id): "T-180m",
    }

    for game in (first, last):
        assert trio.rows_of(game.game_id) == before[game.game_id]

    assert trio.rows_of(middle.game_id) != before[middle.game_id]

    # And the document as a whole did move, so the test is not passing
    # because nothing was published.
    assert trio.served_bytes() != served_before


def test_three_games_stand_at_three_windows_at_the_same_time(trio):
    """One composite, three games, three different windows, simultaneously.

    The property the one pointer made unrepresentable. There is no slate-wide
    window any more: there are three cards, each at the latest window its own
    game validated, all live at once and all traceable to three different
    canary releases.
    """
    first, middle, last = trio.games

    assert trio.window(180, trio.games)[0] == live.EXIT_OK

    assert trio.window(90, (first, middle), canary_games=trio.games)[0] == (
        live.EXIT_OK
    )

    code, receipt = trio.window(20, (first,), canary_games=trio.games)

    assert code == live.EXIT_OK, receipt

    assert trio.live_windows() == {
        str(first.game_id): "T-20m",
        str(middle.game_id): "T-90m",
        str(last.game_id): "T-180m",
    }

    manifest = receipt["composite"]

    assert sorted(
        entry["SOURCE_WINDOW"] for entry in manifest["games"]
    ) == ["T-180m", "T-20m", "T-90m"]

    # Three windows, three distinct canary releases behind them, and every
    # row of the composite traced back to one of them.
    assert len(manifest["source_releases"]) == 3

    assert manifest[composite.UNATTESTED_ROWS_FIELD] == 0

    assert manifest[composite.SOURCE_MISMATCH_FIELD] == 0

    assert manifest["attested_rows"] == len(trio.served_rows())

    # Every game is owned exactly once. Not a property of a dictionary here:
    # asserted against the rows the destination serves, so two components
    # claiming the same game would show up as a game with twice its rows.
    assert sorted(
        str(row["game_id"]) for row in trio.served_rows()
    ) == sorted(
        str(game.game_id)
        for game in trio.games
        for _ in range(2)
    )


# ======================================================================
# the composite manifest and its attestation counts
# ======================================================================


def test_the_manifest_names_the_source_of_every_game_field_by_field(trio):
    """Every field the composite manifest is required to record, per game.

    Checked for presence and for being populated rather than for particular
    values, except where the value is a fact about the source that can be
    compared against the source: the window, the capture the controller named,
    the release the rows came out of and the digest of those rows.
    """
    first = trio.games[0]

    assert trio.window(180, trio.games)[0] == live.EXIT_OK

    code, receipt = trio.window(90, (first,), canary_games=trio.games)

    assert code == live.EXIT_OK, receipt

    manifest = receipt["composite"]

    assert manifest["contract_version"] == (
        composite.COMPOSITE_MANIFEST_CONTRACT_VERSION
    )

    assert manifest["slate_date"] == SLATE_DATE

    assert manifest["mode"] == publisher.MODE_LIVE

    assert manifest["model_authority"] == publisher.MODEL_AUTHORITY

    assert manifest["rows_sha256"] == receipt["release"]["rows_sha256"]

    entries = {entry["GAME_ID"]: entry for entry in manifest["games"]}

    assert sorted(entries) == sorted(
        str(game.game_id) for game in trio.games
    )

    for game_id, entry in entries.items():
        for name in (
            "CAPTURE_ID",
            "CAPTURE_TIMESTAMP",
            "FACTOR_ID",
            "GAME_ID",
            "MODEL_ID",
            "POLICY_ID",
            "PRODUCTION_SHA",
            "SOURCE_RELEASE_ID",
            "SOURCE_ROWS_SHA256",
            "SOURCE_WINDOW",
        ):
            assert entry.get(name), (game_id, name)

        # FIT_ID is the one that may legitimately be absent: the incumbent is
        # a frozen bundle rather than a promoted fit, so there is no fit to
        # name and inventing one would be the lie the field exists to prevent.
        assert "FIT_ID" in entry

        offset = 90 if game_id == str(first.game_id) else 180

        assert entry["SOURCE_WINDOW"] == f"T-{offset}m"

        assert entry["CAPTURE_ID"] == f"capture-{game_id}-t{offset}"

        assert entry["CAPTURE_TIMESTAMP"] == (
            trio.by_id[game_id].captured_at(offset).isoformat()
        )

    # The digest every entry names is the digest of the release it names,
    # recomputed off the stored bytes rather than read out of a field.
    built = trio.built

    assert built is not None

    for entry in manifest["games"]:
        bundle = composite.read_source_bundle(
            built["data_root"], SLATE_DATE, entry["SOURCE_RELEASE_ID"]
        )

        assert bundle.rows_sha256 == entry["SOURCE_ROWS_SHA256"]

        assert entry["GAME_ID"] in {
            str(row.get("game_id")) for row in bundle.rows()
        }


def test_a_row_no_source_holds_is_counted_and_refuses_to_publish(trio):
    """``COMPOSITE_UNATTESTED_ROWS`` is computed from the source bytes.

    Asserted by putting a row into a composite that no attested source
    holds, which is the only way to see the count do anything: every composite
    the entry point assembles is made of copied rows, so the count is zero on
    every honest path and a test that only ever saw zero would be testing
    nothing.
    """
    assert trio.window(180, trio.games)[0] == live.EXIT_OK

    built = trio.built

    assert built is not None

    release_id = str(trio.pointer()["release_id"])

    components = composite.recorded_components(trio.ledger()[-1]["games"])

    bundles = {
        component.source_release_id: composite.read_source_bundle(
            built["data_root"], SLATE_DATE, component.source_release_id
        )
        for component in components.values()
    }

    assert release_id

    honest = composite.compose(components, bundles, slate_date=SLATE_DATE)

    composite.assert_every_row_is_attested(honest.audit)

    # One row of somebody else's, slipped in. Same game, a line nobody quoted.
    forged = dict(honest.rows[0])

    forged["line_value"] = 99.5

    audited = composite.audit_composite(
        [*honest.rows, forged],
        honest.distributions,
        components=components,
        bundles=bundles,
        slate_date=SLATE_DATE,
    )

    assert audited[composite.UNATTESTED_ROWS_FIELD] == 1

    assert audited[composite.SOURCE_MISMATCH_FIELD] == 0

    assert audited["attested_rows"] == len(honest.rows)

    with pytest.raises(composite.CompositeRefused) as refusal:
        composite.assert_every_row_is_attested(audited)

    assert composite.UNATTESTED_ROWS_FIELD in str(refusal.value)

    # And an edited row -- one whose logical key a source does hold and whose
    # bytes it does not -- is the other count, because a row from nowhere and
    # a row changed on the way are different failures.
    edited = dict(honest.rows[0])

    edited["calibrated_q_over_nonpush"] = 0.99

    mismatched = composite.audit_composite(
        [edited, *honest.rows[1:]],
        honest.distributions,
        components=components,
        bundles=bundles,
        slate_date=SLATE_DATE,
    )

    assert mismatched[composite.SOURCE_MISMATCH_FIELD] == 1

    assert mismatched[composite.UNATTESTED_ROWS_FIELD] == 0

    with pytest.raises(composite.CompositeRefused) as refusal:
        composite.assert_every_row_is_attested(mismatched)

    assert composite.SOURCE_MISMATCH_FIELD in str(refusal.value)


def test_every_published_composite_reports_both_counts_as_zero(trio):
    """The counts are reported on the happy path too, and in the ledger.

    A gate that is only visible when it fires is a gate nobody can audit, so
    the receipt, the manifest written beside the release and the durable
    ledger line all carry both numbers.
    """
    first, _, _ = trio.games

    assert trio.window(180, trio.games)[0] == live.EXIT_OK

    code, receipt = trio.window(45, (first,), canary_games=trio.games)

    assert code == live.EXIT_OK, receipt

    current = trio.pointer()

    assert current is not None

    written = json.loads(
        (
            trio.live_root
            / composite.manifest_relative(str(current["release_id"]))
        ).read_text(encoding="utf-8")
    )

    recorded = trio.ledger()[-1]["composite_audit"]

    for document in (receipt["composite"], written, recorded):
        assert document[composite.UNATTESTED_ROWS_FIELD] == 0

        assert document[composite.SOURCE_MISMATCH_FIELD] == 0

        assert document[composite.UNATTESTED_DISTRIBUTIONS_FIELD] == 0

    assert written["rows_sha256"] == str(current["rows_sha256"])


# ======================================================================
# the prior valid composite survives every failure
# ======================================================================


@pytest.fixture
def composed(trio) -> tuple[Slate, bytes, dict[str, Any], dict[str, str]]:
    """A valid composite, live, with every game at its own window.

    What every failure below is injected against. Returned with the bytes a
    reader is served and the pointer that names them, so "the previous good
    composite is still being served" is asserted on the bytes rather than on
    the release id.
    """
    first, middle, _ = trio.games

    assert trio.window(180, trio.games)[0] == live.EXIT_OK

    assert trio.window(90, (first, middle), canary_games=trio.games)[0] == (
        live.EXIT_OK
    )

    assert trio.window(20, (first,), canary_games=trio.games)[0] == (
        live.EXIT_OK
    )

    return (
        trio,
        trio.served_bytes(),
        dict(trio.pointer() or {}),
        trio.live_windows(),
    )


def assert_the_previous_composite_is_still_served(
    slate: Slate,
    served: bytes,
    pointer: dict[str, Any],
    windows: dict[str, str],
) -> None:
    """Nothing a reader can see has changed, in any of the three ways."""
    assert slate.pointer() == pointer

    assert slate.served_bytes() == served

    assert slate.live_windows() == windows


def test_a_source_the_store_has_lost_leaves_the_previous_composite_serving(
    composed,
):
    """A held game whose stored source is gone refuses the whole refresh.

    And deliberately refuses rather than publishing the games it can: a
    composite missing a game is a reader's page silently losing a card, which
    is the failure mode this design exists to end. The previous composite is
    still complete, so it stays.
    """
    slate, served, pointer, windows = composed

    _, middle, last = slate.games

    built = slate.built

    assert built is not None

    # The release the third game is holding its T-180 card out of. The
    # refresh is for the second game, which is the point: the source that goes
    # missing belongs to a game this refresh is not even about.
    held = composite.recorded_components(slate.ledger()[-1]["games"])[
        str(last.game_id)
    ]

    directory = composite.source_directory(
        built["data_root"], SLATE_DATE, held.source_release_id
    )

    assert directory.is_dir()

    (directory / composite.SOURCE_ROWS_NAME).unlink()

    code, receipt = slate.window(45, (middle,), canary_games=slate.games)

    assert code == live.EXIT_REFUSED, receipt

    assert held.source_release_id in receipt["message"]

    assert_the_previous_composite_is_still_served(
        slate, served, pointer, windows
    )


def test_a_source_whose_bytes_were_edited_leaves_the_previous_one_serving(
    composed,
):
    """The digest is recomputed on every read, so an edited source is caught.

    The recorded ``SOURCE_ROWS_SHA256`` is not trusted and could not be: it is
    a field in the same store as the bytes it describes. The refusal comes
    from re-digesting the bytes and comparing them against the attestation the
    canary signed, which is a statement nothing in the store can forge.
    """
    slate, served, pointer, windows = composed

    _, middle, last = slate.games

    built = slate.built

    assert built is not None

    held = composite.recorded_components(slate.ledger()[-1]["games"])[
        str(last.game_id)
    ]

    rows_path = (
        composite.source_directory(
            built["data_root"], SLATE_DATE, held.source_release_id
        )
        / composite.SOURCE_ROWS_NAME
    )

    document = json.loads(rows_path.read_text(encoding="utf-8"))

    document["rows"][0]["calibrated_q_over_nonpush"] = 0.99

    rows_path.write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )

    code, receipt = slate.window(45, (middle,), canary_games=slate.games)

    assert code == live.EXIT_REFUSED, receipt

    assert held.source_rows_sha256 in receipt["message"]

    assert_the_previous_composite_is_still_served(
        slate, served, pointer, windows
    )


def test_a_canary_attestation_of_other_bytes_leaves_the_previous_one_serving(
    composed,
):
    """The publisher's own canary gate, applied once per source.

    A composite of three releases runs that gate three times rather than once,
    which is strictly more than the single-release chain ever did. Here one of
    the three stored attestations is made to cover a different release, and
    nothing is published.
    """
    slate, served, pointer, windows = composed

    _, middle, last = slate.games

    built = slate.built

    assert built is not None

    held = composite.recorded_components(slate.ledger()[-1]["games"])[
        str(last.game_id)
    ]

    attestation_path = (
        composite.source_directory(
            built["data_root"], SLATE_DATE, held.source_release_id
        )
        / composite.SOURCE_ATTESTATION_NAME
    )

    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))

    attestation["release_id"] = "nba_prop_quant__2026-10-24__cafecafecafecafe"

    attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True), encoding="utf-8"
    )

    code, receipt = slate.window(45, (middle,), canary_games=slate.games)

    assert code == live.EXIT_REFUSED, receipt

    assert "cafecafecafecafe" in receipt["message"]

    assert_the_previous_composite_is_still_served(
        slate, served, pointer, windows
    )


def test_an_advancing_window_with_no_named_capture_refuses(composed):
    """A card whose snapshot nobody could find again is not published.

    The capture identity is required and never defaulted. A composite entry
    with an invented ``CAPTURE_ID`` would be a row on the served root with no
    snapshot behind it that anybody could trace, and that is exactly the
    thing this chain may never do -- so the refusal names the flag and the
    window instead, and the previous composite stays.
    """
    slate, served, pointer, windows = composed

    _, middle, _ = slate.games

    slate.canary(45, slate.games)

    given = slate.argv(
        45, (middle,), now=middle.captured_at(45) + timedelta(minutes=1)
    )

    at = given.index("--window-capture")

    argv = given[:at] + given[at + 2 :]

    assert "--window-capture" not in argv

    code = live.main(argv)

    assert code == live.EXIT_REFUSED

    receipt = json.loads(slate.receipt().read_text(encoding="utf-8"))

    assert "--window-capture" in receipt["message"]

    assert middle.window(45) in receipt["message"]

    assert_the_previous_composite_is_still_served(
        slate, served, pointer, windows
    )


def test_a_public_readback_of_other_bytes_leaves_the_previous_one_serving(
    composed, monkeypatch
):
    """The HTTPS verification the destination exists for, failing.

    ``verify_release`` runs before ``promote``, which is the publisher's own
    ordering and the reason this is survivable at all: the new release's files
    are written, the readback disagrees, and the pointer is never moved. What
    a reader is served is still the previous composite, byte for byte.

    The local transport is given the public-readback methods an HTTPS one has
    so the real ``verify_public_readback`` runs over them. Patched after the
    canary has published, so the only publication it reaches is the live one.
    """
    slate, served, pointer, windows = composed

    _, middle, _ = slate.games

    slate.canary(45, slate.games)

    monkeypatch.setattr(
        publisher.LocalDirectoryTransport,
        "public_url",
        lambda self, relative: f"https://example.invalid/{relative}",
        raising=False,
    )

    monkeypatch.setattr(
        publisher.LocalDirectoryTransport,
        "public_readback",
        lambda self, relative: b'{"rows": []}',
        raising=False,
    )

    code, receipt = slate.refresh(45, (middle,))

    assert code == live.EXIT_REFUSED, receipt

    assert "https://example.invalid/" in receipt["message"]

    assert "not promoted" in receipt["message"]

    assert_the_previous_composite_is_still_served(
        slate, served, pointer, windows
    )


def test_a_destination_that_will_not_talk_leaves_the_previous_one_serving(
    composed, monkeypatch
):
    """The first byte of the write failing, which is the earliest it can.

    Nothing of the new composite reached the destination, so there is nothing
    to roll back and the pointer still names the composite that validated.
    """
    slate, served, pointer, windows = composed

    _, middle, _ = slate.games

    slate.canary(45, slate.games)

    def refuse(self, relative, payload):
        raise OSError("the destination is not reachable")

    monkeypatch.setattr(
        publisher.LocalDirectoryTransport, "write_bytes", refuse
    )

    code, receipt = slate.refresh(45, (middle,))

    assert code == live.EXIT_REFUSED, receipt

    monkeypatch.undo()

    assert_the_previous_composite_is_still_served(
        slate, served, pointer, windows
    )


# ======================================================================
# the post-tip guard, per game
# ======================================================================


def test_a_game_that_has_tipped_is_refused_and_the_slate_still_goes(composed):
    """One game started; the other two are published anyway.

    The guard is about the game, not the slate. The started game gets no new
    card and keeps the one it had -- which was published before it started and
    is a correct pregame price -- and the two that have not started advance.
    """
    slate, _, _, windows = composed

    first, middle, last = slate.games

    held = windows[str(first.game_id)]

    slate.canary(45, slate.games)

    # After the first game's tip and before the other two's: the only moment
    # at which this is a mixed refresh rather than an ordinary one.
    moment = first.tip + timedelta(minutes=5)

    assert moment < middle.tip

    code, receipt = slate.refresh(45, slate.games, now=moment)

    assert code == live.EXIT_OK, receipt

    assert receipt["live_outcome"] == live.OUTCOME_REPLACED

    assert receipt["advancement"]["advancing_game_ids"] == sorted(
        [str(middle.game_id), str(last.game_id)]
    )

    refused = receipt["advancement"]["post_tip_refused"]

    assert [entry["game_id"] for entry in refused] == [str(first.game_id)]

    assert "not newly published after its own tip" in refused[0]["reason"]

    # Refused new publication, not retracted: its card is still live and still
    # at the window it validly reached before it started.
    assert slate.live_windows() == {
        str(first.game_id): held,
        str(middle.game_id): "T-45m",
        str(last.game_id): "T-45m",
    }


def test_a_refresh_in_which_every_game_has_tipped_refuses_and_keeps_the_card(
    composed,
):
    """Nothing left to publish, and nothing taken away either."""
    slate, served, pointer, windows = composed

    slate.canary(20, slate.games)

    code, receipt = slate.refresh(
        20, slate.games, now=slate.games[-1].tip + timedelta(minutes=1)
    )

    assert code == live.EXIT_REFUSED, receipt

    assert "have already started" in receipt["message"]

    assert_the_previous_composite_is_still_served(
        slate, served, pointer, windows
    )


# ======================================================================
# the whole evening
# ======================================================================


def drive_the_evening(slate: Slate) -> dict[str, Any]:
    """Run all twenty-four window groups of the slate, in capture order.

    Each group is the whole chain: the capture is named, the cumulative set of
    games captured at that offset is served and published to that offset's
    canary, and the live refresh is handed that group's windows and nothing
    else -- which is exactly what the controller does.

    Returns what every group did, and what the live destination held after it,
    so the matrix can be rendered from observations rather than from
    intentions.
    """
    groups = window_groups(slate.games)

    captured: dict[int, list[Game]] = {}

    history: list[dict[str, Any]] = []

    for index, group in enumerate(groups):
        offset = group.offset_minutes

        for game in group.games:
            captured.setdefault(offset, []).append(game)

        cumulative = tuple(
            sorted(captured[offset], key=lambda game: game.game_id)
        )

        code, receipt = slate.window(
            offset,
            group.games,
            canary_games=cumulative,
            receipt_path=slate.root / f"live_{index:02d}.json",
        )

        history.append(
            {
                "canary_release_id": str(
                    (receipt.get("canary") or {}).get("canary_release_id")
                    or ""
                ),
                "canary_rows": len(cumulative) * 2,
                "code": code,
                "group": group,
                "live_outcome": str(receipt.get("live_outcome") or ""),
                "live_windows": slate.live_windows(),
                "receipt": receipt,
            }
        )

    return {"groups": groups, "history": history}


def render_matrix(slate: Slate, evening: dict[str, Any]) -> str:
    """The eight-by-four window matrix, as a table.

    One row per game and three cells per offset -- capture, canary, live --
    because the claim being evidenced is that all three happened for all
    thirty-two of them, and a single collapsed verdict would hide which stage
    a missing one failed at.
    """
    history = evening["history"]

    offsets = list(live.PROGRESSIVE_OFFSETS_MINUTES)

    cells: dict[tuple[str, int], dict[str, str]] = {}

    for game in slate.games:
        for offset in offsets:
            cells[(str(game.game_id), offset)] = {
                "capture": "-",
                "canary": "-",
                "live": "-",
            }

    for record in history:
        group = record["group"]

        offset = group.offset_minutes

        for game in group.games:
            cell = cells[(str(game.game_id), offset)]

            cell["capture"] = "taken"

            cell["canary"] = (
                "published" if record["canary_release_id"] else "-"
            )

            cell["live"] = (
                "served"
                if record["live_windows"].get(str(game.game_id))
                == f"T-{offset}m"
                else record["live_outcome"] or "refused"
            )

    width = 28

    header = (
        f"{'GAME_ID':<9}{'MATCHUP':<13}{'TIP_ET':<9}"
        + "".join(f"T-{offset}".ljust(width) for offset in offsets)
    )

    lines = [
        "NBA per-game live composite -- window matrix",
        (
            f"slate {SLATE_DATE}, {len(slate.games)} games, "
            f"{len(offsets)} windows each, "
            f"{len(evening['groups'])} window groups"
        ),
        "",
        "each cell is capture / canary / live",
        "",
        header,
        "-" * len(header),
    ]

    for game in slate.games:
        row = (
            f"{game.game_id:<9}{game.matchup:<13}"
            f"{game.tip_et + ' ET':<9}"
        )

        for offset in offsets:
            cell = cells[(str(game.game_id), offset)]

            row += (
                f"{cell['capture']} / {cell['canary']} / "
                f"{cell['live']}"
            ).ljust(width)

        lines.append(row)

    lines += ["", "final live composite, per game:"]

    for game_id, window in sorted(slate.live_windows().items()):
        lines.append(f"  {game_id}  {window}")

    declined = declined_by_one_pointer(evening["groups"])

    lines += [
        "",
        (
            "what the one slate-wide forward-only pointer would have done "
            "with the same evening:"
        ),
        (
            f"  declined {len(declined)} of "
            f"{len(evening['groups'])} window groups"
        ),
    ]

    lines += [f"    {group.label}" for group in declined]

    declined_now = sum(
        1
        for record in history
        if record["live_outcome"] == live.OUTCOME_ALREADY_CLOSER_TO_TIP
    )

    published_now = sum(
        1
        for record in history
        if record["live_outcome"] in live.OUTCOMES_THAT_ARE_LIVE
    )

    lines += [
        "",
        "what the per-game composite did:",
        f"  declined {declined_now} of {len(evening['groups'])} window groups",
        (
            f"  published {published_now} of "
            f"{len(evening['groups'])} window groups"
        ),
        "",
        (
            "the evening group by group -- every game live after each "
            "refresh, at its own window:"
        ),
        "",
        (
            f"{'#':<4}{'CAPTURED':<12}{'WINDOW':<9}{'ADVANCED':<12}"
            f"{'OUTCOME':<18}LIVE SLATE"
        ),
        "-" * 118,
    ]

    for index, record in enumerate(history, start=1):
        group = record["group"]

        lines.append(
            f"{index:<4}"
            + f"{group.captured_at.astimezone(EASTERN).strftime('%H:%M ET'):<12}"
            + f"{'T-' + str(group.offset_minutes):<9}"
            + f"{'/'.join(str(game.game_id) for game in group.games):<12}"
            + f"{record['live_outcome']:<18}"
            + " ".join(
                f"{game_id}@{window.removeprefix('T-').removesuffix('m')}"
                for game_id, window in sorted(record["live_windows"].items())
            )
        )

    lines.append("")

    return "\n".join(lines) + "\n"


def test_the_whole_eight_game_slate_reaches_every_window_live(slate, tmp_path):
    """The end-to-end claim, on the real slate's own schedule.

    Twenty-four window groups driven in capture order, every one of them
    capture to canary to live composite. Nothing is declined, every game is
    served at every one of its four windows, and no game ever disappears from
    the page because another game advanced.

    The matrix is written to an artifact rather than only asserted, because
    the interesting part is which cells are populated and that reads far
    better as a table than as a list of assertions.
    """
    evening = drive_the_evening(slate)

    history = evening["history"]

    assert len(history) == 24

    # Not one group refused, and not one declined. Under the rule this
    # replaced, fourteen of these were refused.
    for record in history:
        assert record["code"] == live.EXIT_OK, (
            record["group"].label,
            record["receipt"].get("message"),
        )

        assert record["live_outcome"] in live.OUTCOMES_THAT_ARE_LIVE, (
            record["group"].label,
            record["live_outcome"],
        )

    # Every game reached every window live, which is thirty-two cells and all
    # of them served.
    reached: dict[str, set[int]] = {
        str(game.game_id): set() for game in slate.games
    }

    for record in history:
        offset = record["group"].offset_minutes

        for game in record["group"].games:
            assert record["live_windows"][str(game.game_id)] == (
                f"T-{offset}m"
            ), (record["group"].label, game.game_id)

            reached[str(game.game_id)].add(offset)

    for game_id, offsets in reached.items():
        assert offsets == set(live.PROGRESSIVE_OFFSETS_MINUTES), game_id

    # Every game is owned exactly once per offset: each game appears in
    # exactly one group at each offset, and each group published it.
    for offset in live.PROGRESSIVE_OFFSETS_MINUTES:
        owners = [
            game.game_id
            for record in history
            if record["group"].offset_minutes == offset
            for game in record["group"].games
        ]

        assert sorted(owners) == sorted(game.game_id for game in slate.games)

    # The defect itself, gone. The first game reaches T-20 at group nine,
    # and every one of the seven later games is still on the page afterwards
    # at its own window -- which is the thing the one pointer made impossible.
    first = slate.games[0]

    after_the_first_t20 = [
        record
        for record in history
        if record["group"].captured_at > first.captured_at(20)
    ]

    assert len(after_the_first_t20) == 15

    for record in after_the_first_t20:
        assert record["live_windows"][str(first.game_id)] == "T-20m"

    # By the end every game stands at T-20, each out of its own capture.
    assert slate.live_windows() == {
        str(game.game_id): "T-20m" for game in slate.games
    }

    final = history[-1]["receipt"]["composite"]

    assert len(final["games"]) == 8

    assert final[composite.UNATTESTED_ROWS_FIELD] == 0

    assert final[composite.SOURCE_MISMATCH_FIELD] == 0

    assert final[composite.UNATTESTED_DISTRIBUTIONS_FIELD] == 0

    assert len({entry["CAPTURE_ID"] for entry in final["games"]}) == 8

    # The matrix, for a reader.
    matrix = render_matrix(slate, evening)

    for game in slate.games:
        assert game.matchup in matrix

    assert matrix.count("taken / published / served") == 32

    artifacts = Path("/opt/cursor/artifacts")

    if artifacts.is_dir():
        (artifacts / "composite_window_matrix.txt").write_text(
            matrix, encoding="utf-8"
        )
    (tmp_path / "composite_window_matrix.txt").write_text(
        matrix, encoding="utf-8"
    )


def test_a_game_held_from_a_pruned_canary_release_is_still_reassembled(slate):
    """The reason the attested source store is durable rather than borrowed.

    A canary destination retains two releases: ``current.json`` and
    ``previous.json``. A cumulative offset publishes one release per tip
    group, so by the fourth T-180 publication the release the first tip
    group's card came out of has been pruned off the canary host -- and that
    card is still on the page and still has to be reassembled byte for byte.

    Driven as the evening drives it and then checked against the canary host,
    so this is a property of the real retention rather than of a simulated
    one.
    """
    groups = [
        group
        for group in window_groups(slate.games)
        if group.offset_minutes == 180
    ]

    assert len(groups) == 6

    captured: list[Game] = []

    sources: dict[str, str] = {}

    for group in groups:
        captured += list(group.games)

        code, receipt = slate.window(
            180,
            group.games,
            canary_games=tuple(
                sorted(captured, key=lambda game: game.game_id)
            ),
        )

        assert code == live.EXIT_OK, receipt

        for entry in receipt["composite"]["games"]:
            sources[str(entry["GAME_ID"])] = str(entry["SOURCE_RELEASE_ID"])

    # Eight games, six distinct source releases, and the earliest of them is
    # no longer on the canary host.
    assert len(sources) == 8

    assert len(set(sources.values())) == 6

    on_the_host = {
        path.name
        for path in (
            slate.canary_root(180) / publisher.RELEASES_DIRECTORY
        ).iterdir()
    }

    pruned = set(sources.values()) - on_the_host

    assert pruned, (
        "the canary host still holds every source, so this test is not "
        "exercising the retention it is about"
    )

    # And the card from the pruned release is still being served, out of the
    # durable store.
    built = slate.built

    assert built is not None

    for game_id, release_id in sources.items():
        if release_id not in pruned:
            continue

        bundle = composite.read_source_bundle(
            built["data_root"], SLATE_DATE, release_id
        )

        expected = publisher.serialise(
            composite.rows_of_game(bundle, game_id)
        )

        assert slate.rows_of(game_id) == expected
