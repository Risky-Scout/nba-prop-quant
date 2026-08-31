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
