from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

import pandas as pd


COMPONENT_SPECS = {
    "games": (
        "games",
        "game",
    ),
    "active_players": (
        "active_players",
        "active_player",
    ),
    "injuries": (
        "injuries",
        "injury",
    ),
    "lineups": (
        "lineups",
        "lineup",
    ),
    "player_props": (
        "player_props",
        "live_player_prop",
    ),
}

PRIMARY_OFFSET_MINUTES = 20


def _canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )


def canonical_records_sha256(
    records: list[dict[str, Any]],
) -> str:
    lines = sorted(
        _canonical_json(
            row
        )
        for row in records
    )

    payload = (
        "\n".join(
            lines
        )
        + (
            "\n"
            if lines
            else ""
        )
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


def build_capture_id(
    *,
    date: str,
    captured_at: str,
    window_ids: list[str],
    due_game_ids: list[int],
    component_sha256: dict[str, str],
) -> str:
    payload = {
        "date": str(
            date
        ),
        "captured_at": str(
            captured_at
        ),
        "window_ids": sorted(
            str(x)
            for x in window_ids
        ),
        "due_game_ids": sorted(
            int(x)
            for x in due_game_ids
        ),
        "component_sha256": {
            str(key): str(
                component_sha256[
                    key
                ]
            )
            for key in sorted(
                component_sha256
            )
        },
    }

    return hashlib.sha256(
        _canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _parse_utc(
    raw: str,
) -> datetime:
    text = str(
        raw
    )

    if text.endswith(
        "Z"
    ):
        text = (
            text[:-1]
            + "+00:00"
        )

    value = datetime.fromisoformat(
        text
    )

    if value.tzinfo is None:
        raise RuntimeError(
            "Snapshot timestamp is timezone-naive"
        )

    return value.astimezone(
        timezone.utc
    )


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []

    for raw in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not raw.strip():
            continue

        rows.append(
            json.loads(
                raw
            )
        )

    return rows


def _component_records(
    snapshot_dir: Path,
    *,
    component: str,
    date: str,
    captured_at: str,
) -> list[dict[str, Any]]:
    directory, snapshot_type = (
        COMPONENT_SPECS[
            component
        ]
    )

    path = (
        snapshot_dir
        / directory
        / f"{date}.jsonl"
    )

    records = []

    for envelope in _read_jsonl(
        path
    ):
        if (
            str(
                envelope.get(
                    "captured_at"
                )
            )
            != str(
                captured_at
            )
        ):
            continue

        if (
            envelope.get(
                "snapshot_type"
            )
            != snapshot_type
        ):
            continue

        payload = envelope.get(
            "payload"
        )

        if isinstance(
            payload,
            dict,
        ):
            records.append(
                payload
            )

    return records


@dataclass(frozen=True)
class CaptureBundle:
    game_id: int
    date: str
    window_id: str
    offset_minutes: int
    captured_at: str
    capture_id: str
    component_sha256: dict[str, str]
    records: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ]
    game: dict[str, Any]


def _capture_runs(
    snapshot_dir: Path,
    date: str,
) -> list[dict[str, Any]]:
    path = (
        snapshot_dir
        / "capture_runs"
        / f"{date}.jsonl"
    )

    return [
        row
        for row in _read_jsonl(
            path
        )
        if row.get(
            "snapshot_type"
        )
        == "capture_run"
    ]


def _capture_identity(
    envelope: dict[str, Any],
) -> tuple[
    tuple[str, str],
    ...,
]:
    """Which capture a ``capture_run`` row says it is.

    A capture is content-addressed. :func:`build_capture_id` digests the
    slate date, the instant, the window ids, the due game ids and the five
    component digests, and :func:`select_capture_bundle` recomputes that
    digest off the envelope before pricing may read the capture -- so those
    fields, and only those, are what distinguishes one point-in-time
    snapshot from another. Two rows agreeing on every one of them denote the
    same capture: the same instant, over the same windows and the same due
    games, with byte-identical content in every component.

    What is compared is the preimage rather than the stored ``capture_id``,
    because the stored id is a claim the envelope makes about itself. A row
    carrying a copied id over a different instant would otherwise read as a
    duplicate of the row it copied from, and something would have chosen
    between two snapshots before the capture-id check below ever ran.
    Comparing the fields the id is derived from cannot be fooled that way,
    and on a row the collector wrote it is the same comparison, since equal
    fields hash to equal ids.

    ``error_count`` is here as well and is the one field outside the digest.
    It is not part of the capture's identity in the data model, but
    :func:`select_capture_bundle` refuses a capture whose sweep recorded any
    error, so two rows agreeing on all the content and disagreeing about
    whether that sweep was clean are not interchangeable: exactly one of
    them is servable and nothing here can say which. The stored
    ``capture_id`` is carried for the same reason -- a row whose id does not
    describe it is refused below, and whether a window is refused may not
    depend on which of two rows was read.

    Every remaining key of the envelope -- the five counts, ``game_ids``,
    ``team_ids``, ``errors``, ``grace_minutes``, the payload's own ``date``
    -- is either derived from content already compared here or is read by
    nothing on the serving path, so a row this collector wrote cannot differ
    in one of them while agreeing on all of these, and none of them can
    reach what is served.

    Rendered as ordered ``(name, value)`` pairs of strings so that the
    comparison cannot depend on dict ordering and so that a refusal can name
    the fields that differ. The two id lists are sorted for the reason
    :func:`build_capture_id` sorts them -- a list written in another order
    is the same set of windows -- and sorted as recorded rather than as
    parsed, so a row this function cannot interpret is a difference rather
    than an exception.
    """
    payload = (
        envelope.get(
            "payload"
        )
        or {}
    )

    component_hashes = (
        payload.get(
            "component_sha256"
        )
        or {}
    )

    return (
        (
            "captured_at",
            str(
                envelope.get(
                    "captured_at"
                )
            ),
        ),
        (
            "window_ids",
            _canonical_json(
                sorted(
                    str(x)
                    for x in payload.get(
                        "window_ids",
                        [],
                    )
                )
            ),
        ),
        (
            "due_game_ids",
            _canonical_json(
                sorted(
                    str(x)
                    for x in payload.get(
                        "due_game_ids",
                        [],
                    )
                )
            ),
        ),
        (
            "component_sha256",
            _canonical_json(
                {
                    str(key): str(
                        value
                    )
                    for key, value in (
                        component_hashes.items()
                    )
                }
            ),
        ),
        (
            "error_count",
            str(
                payload.get(
                    "error_count"
                )
            ),
        ),
        (
            "capture_id",
            str(
                payload.get(
                    "capture_id",
                    "",
                )
            ),
        ),
    )


def _describe_capture_conflict(
    identities: list[
        tuple[
            tuple[str, str],
            ...,
        ]
    ],
) -> str:
    """Field by field, what the captures of one window disagree about.

    A refusal that says only how many records were found is a refusal nobody
    can act on. The 2026-10-23 one said "found 2", and establishing whether
    those two were one capture written twice or two snapshots taken minutes
    apart meant reading the log by hand -- on the evening the window was
    still open. So the difference is named: every field the distinct
    captures do not agree on, with the values they carry, in field order and
    with the values sorted, so one conflict always reads the same way.

    The component digests are reported per component rather than as the maps
    they are compared as. The comparison has to be over the whole map,
    because that is what ``build_capture_id`` digests, but a reader of the
    refusal needs to know which component moved and would otherwise be left
    diffing two five-entry maps by eye.
    """
    parts = []

    for index, name in enumerate(
        name
        for name, _ in identities[
            0
        ]
    ):
        values = sorted(
            {
                identity[index][1]
                for identity in identities
            }
        )

        if len(
            values
        ) < 2:
            continue

        if (
            name
            == "component_sha256"
        ):
            maps = [
                json.loads(
                    value
                )
                for value in values
            ]

            for component in sorted(
                {
                    key
                    for row in maps
                    for key in row
                }
            ):
                digests = sorted(
                    {
                        str(
                            row.get(
                                component
                            )
                        )
                        for row in maps
                    }
                )

                if len(
                    digests
                ) < 2:
                    continue

                parts.append(
                    f"the {component} "
                    "digest "
                    + " vs ".join(
                        digests
                    )
                )

            continue

        parts.append(
            f"{name} "
            + " vs ".join(
                values
            )
        )

    return "; ".join(
        parts
    )


def _resolve_duplicate_scheduled_captures(
    window_id: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Two records of one capture are that capture; of two, are a refusal.

    Requiring exactly one scheduled capture per window is right about what
    it was written for and cannot distinguish the only two things a second
    record can mean. On 2026-10-23 it refused three games permanently,
    because the capture log is append-only: the second record never goes
    away, so the window was unpriceable for the rest of the slate's life.
    The exclusive claim the collector now takes stops two sweeps recording
    one window again and can do nothing for a record already written, nor
    for one arriving by a route no claim covers -- a replayed log, a
    restored backup, a manual sweep, a second host, a network lock that
    silently does not lock.

    So the two cases are separated, and only they are:

      - records that :func:`_capture_identity` cannot tell apart denote one
        point-in-time snapshot. Whichever is read, every field
        :func:`select_capture_bundle` goes on to use is the same field, so
        the bundle it returns is the same bundle and the capture it verifies
        is the same capture -- which is why picking either is sound and why
        nothing here needs to decide which record is "the" record. It is
        still picked by a stable total order rather than by arrival, the
        lexicographic minimum of the canonical encoding: file order, read
        order and dict order cannot reach it, so two runs over one log
        resolve it identically and a log read backwards resolves to the same
        row.
      - records that it can tell apart denote different snapshots of one
        window. Both are genuine data, both may verify, and the data model
        says nothing about which one T-20 pricing is entitled to read, so
        serving either would be pricing a game off a snapshot nobody chose.
        That refuses, and it refuses with the message production already had
        plus the thing it could not say: how many captures the records
        denote and exactly which recorded facts differ between them.

    Nothing is relabelled, re-timed, merged, synthesized or widened here and
    no capture is written: the return value is a sublist of the candidates
    read off disk, so the only thing a resolution can produce is a record
    the log already held. The log is not rewritten either -- it is the
    append-only evidence that the duplicate happened, which is what an
    operator needs most -- and every eligibility check below runs over the
    resolved record unchanged, so a duplicate of a capture pricing would
    refuse is still refused for the capture's own reason.
    """
    if len(
        candidates
    ) < 2:
        return candidates

    identities = {}

    for envelope in candidates:
        identities.setdefault(
            _capture_identity(
                envelope
            ),
            [],
        ).append(
            envelope
        )

    if len(
        identities
    ) != 1:
        raise RuntimeError(
            f"{window_id}: expected exactly one "
            f"scheduled capture, found {len(candidates)}"
            f", recording {len(identities)} "
            "different point-in-time captures of "
            "this window; nothing can decide which "
            "of them T-20 pricing must read, and "
            "they differ in "
            + _describe_capture_conflict(
                sorted(
                    identities
                )
            )
        )

    return [
        min(
            candidates,
            key=_canonical_json,
        )
    ]


def select_capture_bundle(
    snapshot_dir: Path,
    *,
    date: str,
    game_id: int,
    offset_minutes: int = PRIMARY_OFFSET_MINUTES,
) -> CaptureBundle:
    snapshot_dir = Path(
        snapshot_dir
    )

    window_id = (
        f"{int(game_id)}:"
        f"T-{int(offset_minutes)}m"
    )

    candidates = []

    for envelope in _capture_runs(
        snapshot_dir,
        date,
    ):
        payload = (
            envelope.get(
                "payload"
            )
            or {}
        )

        if (
            window_id
            not in payload.get(
                "window_ids",
                [],
            )
        ):
            continue

        if (
            payload.get(
                "capture_reason"
            )
            != "scheduled"
        ):
            continue

        candidates.append(
            envelope
        )

    candidates = (
        _resolve_duplicate_scheduled_captures(
            window_id,
            candidates,
        )
    )

    if len(
        candidates
    ) != 1:
        raise RuntimeError(
            f"{window_id}: expected exactly one "
            f"scheduled capture, found {len(candidates)}"
        )

    envelope = candidates[
        0
    ]

    payload = (
        envelope.get(
            "payload"
        )
        or {}
    )

    captured_at = str(
        envelope.get(
            "captured_at"
        )
    )

    if int(
        payload.get(
            "error_count",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            f"{window_id}: capture contains errors"
        )

    component_hashes = (
        payload.get(
            "component_sha256"
        )
        or {}
    )

    missing_hashes = (
        set(
            COMPONENT_SPECS
        )
        - set(
            component_hashes
        )
    )

    if missing_hashes:
        raise RuntimeError(
            f"{window_id}: missing component hashes "
            f"{sorted(missing_hashes)}"
        )

    records = {}

    for component in (
        COMPONENT_SPECS
    ):
        values = (
            _component_records(
                snapshot_dir,
                component=component,
                date=date,
                captured_at=captured_at,
            )
        )

        actual_hash = (
            canonical_records_sha256(
                values
            )
        )

        expected_hash = str(
            component_hashes[
                component
            ]
        )

        if (
            actual_hash
            != expected_hash
        ):
            raise RuntimeError(
                f"{window_id}: {component} "
                "component hash mismatch"
            )

        records[
            component
        ] = values

    expected_capture_id = (
        build_capture_id(
            date=date,
            captured_at=captured_at,
            window_ids=[
                str(x)
                for x in payload.get(
                    "window_ids",
                    [],
                )
            ],
            due_game_ids=[
                int(x)
                for x in payload.get(
                    "due_game_ids",
                    [],
                )
            ],
            component_sha256={
                str(k): str(v)
                for k, v in (
                    component_hashes.items()
                )
            },
        )
    )

    stored_capture_id = str(
        payload.get(
            "capture_id",
            "",
        )
    )

    if (
        expected_capture_id
        != stored_capture_id
    ):
        raise RuntimeError(
            f"{window_id}: capture ID mismatch"
        )

    matching_games = [
        game
        for game in records[
            "games"
        ]
        if int(
            game.get(
                "id",
                -1,
            )
        )
        == int(
            game_id
        )
    ]

    if len(
        matching_games
    ) != 1:
        raise RuntimeError(
            f"{window_id}: captured games payload "
            "does not contain exactly one target game"
        )

    game = matching_games[
        0
    ]

    tip_raw = game.get(
        "datetime"
    )

    if not tip_raw:
        raise RuntimeError(
            f"{window_id}: game missing scheduled datetime"
        )

    captured_dt = (
        _parse_utc(
            captured_at
        )
    )

    tip_dt = (
        _parse_utc(
            str(
                tip_raw
            )
        )
    )

    if not (
        captured_dt
        < tip_dt
    ):
        raise RuntimeError(
            f"{window_id}: capture is not pre-tip"
        )

    return CaptureBundle(
        game_id=int(
            game_id
        ),
        date=str(
            date
        ),
        window_id=window_id,
        offset_minutes=int(
            offset_minutes
        ),
        captured_at=captured_at,
        capture_id=stored_capture_id,
        component_sha256={
            str(k): str(v)
            for k, v in (
                component_hashes.items()
            )
        },
        records=records,
        game=game,
    )


def _team_ids_for_game(
    game: dict[str, Any],
) -> set[int]:
    result = set()

    for name in [
        "home_team",
        "visitor_team",
    ]:
        team = (
            game.get(
                name
            )
            or {}
        )

        if team.get(
            "id"
        ) is not None:
            result.add(
                int(
                    team[
                        "id"
                    ]
                )
            )

    return result


def _player_team_id(
    player: dict[str, Any],
) -> int | None:
    team = (
        player.get(
            "team"
        )
        or {}
    )

    value = team.get(
        "id",
        player.get(
            "team_id"
        ),
    )

    if value is None:
        return None

    return int(
        value
    )


def _dedupe(
    records: list[dict[str, Any]],
    key_function,
) -> list[dict[str, Any]]:
    output = {}

    for record in records:
        key = key_function(
            record
        )

        if key is None:
            continue

        output[
            key
        ] = record

    return [
        output[key]
        for key in sorted(
            output,
            key=lambda x:
                str(x),
        )
    ]


def _isolated_capture_refusals(
    snapshot_dir: Path,
    *,
    date: str,
    game_ids: list[int],
    offset_minutes: int,
) -> dict[int, str]:
    """Which games of a slate cannot be served, decided one game at a time.

    :func:`select_capture_bundle` decides one game and refuses one game. The
    slate was assembled by selecting every game of it in a single
    comprehension, so the first game that refused aborted the assembly and
    no game in the slate was priced: a conflict in one window cost every
    other window of the evening, and the refusals it fires on are permanent
    ones. Two captures of a window stay two captures, a capture taken after
    its game tipped stays post-tip, and the capture log is append-only, so
    for those games "try again later" is not a thing that exists.

    A refusal has to cost the game it belongs to and no other game, so the
    selection is attempted per game here and the games that refuse are
    recorded with the reason they gave, verbatim. Nothing is decided about
    them beyond that they cannot be served: this returns refusals rather
    than bundles precisely so that it cannot become a second place where a
    capture is chosen.

    Only ``RuntimeError`` is caught, which is every refusal the selection
    states -- the scheduled-reason filter finding nothing, two records that
    are two captures, a recorded error, a missing hash, a component digest
    that is not the content on disk, a capture id that is not the envelope's
    own, a games payload without the target game, a capture past the tip.
    Anything else means the log could not be read at all rather than that
    one game is unservable, and a truncated line or a window id that is not
    a number has to keep aborting the slate: swallowing it per game would
    turn a corrupt log into a quietly short release, which is the failure
    this whole path exists to prevent.
    """
    refusals = {}

    for game_id in sorted(
        {
            int(x)
            for x in game_ids
        }
    ):
        try:
            select_capture_bundle(
                snapshot_dir,
                date=date,
                game_id=game_id,
                offset_minutes=(
                    offset_minutes
                ),
            )
        except RuntimeError as error:
            refusals[
                game_id
            ] = str(
                error
            )

    return refusals


class ProspectiveSnapshotClient:
    def __init__(
        self,
        *,
        snapshot_dir: Path,
        target_date: str,
        offset_minutes: int = PRIMARY_OFFSET_MINUTES,
    ):
        self.snapshot_dir = Path(
            snapshot_dir
        ).expanduser().resolve()

        self.target_date = str(
            target_date
        )

        self.offset_minutes = int(
            offset_minutes
        )

        game_ids = set()

        suffix = (
            f":T-{self.offset_minutes}m"
        )

        for envelope in _capture_runs(
            self.snapshot_dir,
            self.target_date,
        ):
            payload = (
                envelope.get(
                    "payload"
                )
                or {}
            )

            if (
                payload.get(
                    "capture_reason"
                )
                != "scheduled"
            ):
                continue

            for raw_window in payload.get(
                "window_ids",
                [],
            ):
                window = str(
                    raw_window
                )

                if not window.endswith(
                    suffix
                ):
                    continue

                prefix = window.split(
                    ":",
                    1,
                )[0]

                game_ids.add(
                    int(
                        prefix
                    )
                )

        # One game's refusal is one game's problem. The comprehension below
        # selects every game of the slate, so before it runs the games that
        # cannot be selected are found and set aside with the reason
        # selection gave. What is left is the slate that can be served, and
        # the comprehension builds it from exactly the games that can.
        #
        # Selecting twice is deliberate and is the cheaper half of the
        # trade. The selection is a pure read of an append-only log, so the
        # two passes cannot disagree, and the second is production's own
        # line-for-line selection: every eligibility check -- the
        # scheduled-reason filter, the error count, the missing hashes, the
        # component digests recomputed from the rows on disk, the capture id
        # recomputed from the envelope, the target game, the pre-tip bound
        # -- runs inside the comprehension over the resolved record exactly
        # as it always did. No game is served from anything this pass
        # decided. On a slate heavier than production has ever had -- twelve
        # games and a full evening of sweeps in one day's component logs --
        # the extra read costs under half a second, which is a fair price
        # for "the selection pricing reads is still the unchanged one".
        self._refusals = (
            _isolated_capture_refusals(
                self.snapshot_dir,
                date=self.target_date,
                game_ids=sorted(
                    game_ids
                ),
                offset_minutes=(
                    self.offset_minutes
                ),
            )
        )

        game_ids = sorted(
            set(game_ids)
            - set(
                self._refusals
            )
        )

        self._bundles = {
            game_id: (
                select_capture_bundle(
                    self.snapshot_dir,
                    date=self.target_date,
                    game_id=game_id,
                    offset_minutes=(
                        self.offset_minutes
                    ),
                )
            )
            for game_id in sorted(
                game_ids
            )
        }

    def excluded_games(
        self,
    ) -> dict[int, str]:
        """The games of this slate that cannot be served, and why.

        An excluded game must not be a quietly shorter slate. A release that
        is missing a game and does not say so is a worse failure than the
        one the isolation fixes, so every game left out is reachable here,
        by game id, carrying the refusal :func:`select_capture_bundle`
        raised verbatim rather than a summary of it -- which window, how
        many records it found, how many captures those records denote and
        the fields they differ on. A caller that writes a release is
        expected to read this and refuse rather than publish a short slate.

        Sorted by game id so that a log line or a receipt naming them reads
        the same way on every run, and a copy so that a caller cannot drop a
        game from the record of what was not served.
        """
        return dict(
            sorted(
                self._refusals.items()
            )
        )

    def __enter__(
        self,
    ):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    def games(
        self,
        dates=None,
    ):
        if (
            dates is not None
            and self.target_date
            not in {
                str(x)
                for x in dates
            }
        ):
            return []

        return [
            self._bundles[
                game_id
            ].game
            for game_id in sorted(
                self._bundles
            )
        ]

    def active_players(
        self,
        team_ids=None,
    ):
        requested = (
            None
            if team_ids is None
            else {
                int(x)
                for x in team_ids
            }
        )

        rows = []

        for bundle in (
            self._bundles.values()
        ):
            game_teams = (
                _team_ids_for_game(
                    bundle.game
                )
            )

            for player in bundle.records[
                "active_players"
            ]:
                team_id = (
                    _player_team_id(
                        player
                    )
                )

                if team_id is None:
                    continue

                if (
                    team_id
                    not in game_teams
                ):
                    continue

                if (
                    requested
                    is not None
                    and team_id
                    not in requested
                ):
                    continue

                rows.append(
                    player
                )

        return _dedupe(
            rows,
            lambda row:
                row.get(
                    "id"
                ),
        )

    def injuries(
        self,
        team_ids=None,
    ):
        requested = (
            None
            if team_ids is None
            else {
                int(x)
                for x in team_ids
            }
        )

        rows = []

        for bundle in (
            self._bundles.values()
        ):
            game_teams = (
                _team_ids_for_game(
                    bundle.game
                )
            )

            for injury in bundle.records[
                "injuries"
            ]:
                player = (
                    injury.get(
                        "player"
                    )
                    or {}
                )

                team_id = (
                    _player_team_id(
                        player
                    )
                )

                if team_id is None:
                    continue

                if team_id not in (
                    game_teams
                ):
                    continue

                if (
                    requested
                    is not None
                    and team_id
                    not in requested
                ):
                    continue

                rows.append(
                    injury
                )

        return _dedupe(
            rows,
            lambda row:
                (
                    row.get(
                        "player"
                    )
                    or {}
                ).get(
                    "id"
                ),
        )

    def lineups(
        self,
        game_ids,
    ):
        rows = []

        for game_id in sorted(
            {
                int(x)
                for x in game_ids
            }
        ):
            bundle = self._bundles.get(
                game_id
            )

            if bundle is None:
                raise RuntimeError(
                    f"{game_id}: no eligible "
                    "prospective capture"
                )

            rows.extend(
                row
                for row in bundle.records[
                    "lineups"
                ]
                if int(
                    row.get(
                        "game_id",
                        -1,
                    )
                )
                == game_id
            )

        return rows

    def live_player_props(
        self,
        game_id: int,
    ):
        game_id = int(
            game_id
        )

        bundle = self._bundles.get(
            game_id
        )

        if bundle is None:
            raise RuntimeError(
                f"{game_id}: no eligible "
                "prospective capture"
            )

        return [
            row
            for row in bundle.records[
                "player_props"
            ]
            if int(
                row.get(
                    "game_id",
                    -1,
                )
            )
            == game_id
        ]

    def lineage_for_game(
        self,
        game_id: int,
    ) -> dict[str, Any]:
        game_id = int(
            game_id
        )

        bundle = self._bundles.get(
            game_id
        )

        if bundle is None:
            raise RuntimeError(
                f"{game_id}: capture lineage unavailable"
            )

        return {
            "game_id": game_id,
            "gate3_capture_id": (
                bundle.capture_id
            ),
            "gate3_capture_window_id": (
                bundle.window_id
            ),
            "gate3_capture_offset_minutes": (
                bundle.offset_minutes
            ),
            "gate3_captured_at_utc": (
                bundle.captured_at
            ),
            "gate3_input_source": (
                "scheduled_snapshot"
            ),
            "gate3_component_sha256_json": (
                _canonical_json(
                    bundle.component_sha256
                )
            ),
        }

    def lineage_frame(
        self,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                self.lineage_for_game(
                    game_id
                )
                for game_id in sorted(
                    self._bundles
                )
            ]
        )
