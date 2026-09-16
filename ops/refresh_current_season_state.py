#!/usr/bin/env python3
"""Fail-closed operational wrapper around the existing whole-season
current-season history refresh.

The frozen NBA Prop Quant model reads three rolling current-season files:

    <data-dir>/raw/seasons/season=<season>/stats.parquet
    <data-dir>/raw/seasons/season=<season>/games.parquet
    <data-dir>/raw/advanced/season=<season>/advanced.parquet

Those files legitimately advance as 2026-27 games complete, so they are the
rolling integrity domain of the Gate 3 runtime contract rather than part of
the frozen model. This wrapper adds no statistics and changes no prediction
mathematics. It only guarantees that a bad refresh can never replace good
live state:

    * one writer at a time, enforced before anything else happens;
    * readiness decided solely by the existing preflight's process exit
      code, never by its human-readable banner text;
    * every byte fetched into a disposable shadow data root, never into the
      live historical tree;
    * schema, leakage and non-regression validation on the staged files;
    * commit of the exact staged bytes, with post-commit hash verification
      and rollback of the previously live bytes on any caught failure.

Three separate os.replace calls are three separate atomic operations, not
one atomic three-file transaction. A process kill or machine loss between
them can still leave a mixed set on disk; the post-commit hash check and
rollback cover caught in-process failures only.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_prop_quant.slate import (
    HistoryLeakageError,
    assert_history_precedes_slate,
)
from nba_prop_quant.storage import sort_by_keys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREFLIGHT_SCRIPT = PROJECT_ROOT / "ops" / "preflight_current_season_refresh.py"
INGEST_SCRIPT = PROJECT_ROOT / "scripts" / "01_ingest_history_resume.py"

LOCK_RELATIVE_PATH = Path(".locks") / "current_season_refresh.lock"

SHADOW_PREFIX = ".refresh_shadow_"
ROLLBACK_PREFIX = ".refresh_rollback_"

# Durable multi-file refresh journal. Three os.replace calls cannot be one
# POSIX atomic operation, so crash recoverability is provided by a journal
# instead: the pre-transaction bytes and both content hashes of every target
# are recorded durably before the first live replacement, and an interrupted
# transaction is resolved to a single coherent generation on the next run.
TRANSACTIONS_RELATIVE_PATH = Path(".refresh_transactions")

TRANSACTION_PREFIX = "txn_"

PREPARED_MARKER = "PREPARED"
COMMITTED_MARKER = "COMMITTED"

TRANSACTION_SCHEMA_VERSION = 1

# Deterministic semantic integrity record for the mutable rolling tree. Raw
# parquet SHA256 answers "did these bytes change"; this answers "did the
# logical records change", which is the question that survives a rewrite.
STATE_RELATIVE_PATH = Path(".state") / "current_season_state.json"

STATE_SCHEMA_VERSION = 1

# Logical identity of each rolling dataset. Row order is not part of identity,
# so every semantic measurement below is taken on the key-sorted frame.
ROLLING_KEY_COLUMNS = {
    "stats": ("game_id", "player_id"),
    "games": ("id",),
    "advanced": ("game_id", "player_id"),
}

ROLLING_DATE_COLUMN = "date"

DEFAULT_SEASON = 2026

EXIT_OK = 0

# Readiness outcomes owned by ops/preflight_current_season_refresh.py. They
# are propagated unchanged so the caller sees the preflight's own decision.
PREFLIGHT_STOP_CODES = frozenset({10, 20, 30, 31})

EXIT_PREFLIGHT_UNEXPECTED = 70
EXIT_INGEST_FAILED = 71
EXIT_VALIDATION_FAILED = 72
EXIT_COMMIT_ROLLED_BACK = 73
EXIT_POST_COMMIT_ROLLED_BACK = 74
EXIT_LOCK_BUSY = 75
EXIT_ROLLBACK_FAILED = 78
EXIT_RECOVERY_FAILED = 79

STATS_REQUIRED_COLUMNS = (
    "player_id",
    "game_id",
    "date",
    "season",
    "team_id",
    "home_team_id",
    "visitor_team_id",
    "minutes",
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "fg3m",
    "fga",
    "fta",
    "oreb",
    "turnover",
    "fg3a",
    "pf",
)

ADVANCED_REQUIRED_COLUMNS = (
    "player_id",
    "game_id",
    "date",
)

# scripts/01_ingest_history_resume.py validates every games.parquet it writes
# against these columns, and ops/capture_external_test_day.py refuses to build
# its game-date map without id and date. "id" is the normalized game
# identifier that stats.game_id and advanced.game_id resolve against.
GAMES_REQUIRED_COLUMNS = (
    "id",
    "date",
    "season",
    "home_team_id",
    "visitor_team_id",
)

GAMES_ID_COLUMN = "id"

ADVANCED_SEASON_DIR = re.compile(r"^season=\d+$")

BANNER = "=" * 118


class RefreshFailure(Exception):
    """A fail-closed stop with the process exit code to return."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class LockBusy(Exception):
    """Another writer already holds the exclusive refresh lock."""


@dataclass(frozen=True)
class SeasonPaths:
    root: Path
    advanced_root: Path
    stats: Path
    games: Path
    advanced: Path

    def as_items(self) -> list[tuple[str, Path]]:
        return [
            ("stats", self.stats),
            ("games", self.games),
            ("advanced", self.advanced),
        ]


def season_paths(data_root: Path, season: int) -> SeasonPaths:
    raw = data_root / "raw"
    season_dir = raw / "seasons" / f"season={season}"
    advanced_root = raw / "advanced"

    return SeasonPaths(
        root=data_root,
        advanced_root=advanced_root,
        stats=season_dir / "stats.parquet",
        games=season_dir / "games.parquet",
        advanced=advanced_root / f"season={season}" / "advanced.parquet",
    )


def exact_date(value: str, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"invalid {label} {value!r}: {exc}",
        )

    if pd.isna(parsed) or parsed.strftime("%Y-%m-%d") != value:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"{label} must be exact YYYY-MM-DD; got {value!r}",
        )

    return parsed.normalize()


def resolve_data_dir(raw_value: str | None) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()

    from_env = os.environ.get("NBA_PROP_DATA_DIR")

    if from_env:
        return Path(from_env).expanduser().resolve()

    return (PROJECT_ROOT / "data").resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _atomic_replace(source: Path, destination: Path) -> None:
    """Single replacement seam so every rename in this module is auditable."""
    os.replace(source, destination)


def _temp_sibling(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.refresh_{uuid.uuid4().hex}"


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_durable(payload: dict, path: Path) -> None:
    """Write deterministic JSON so a crash leaves either old or new content."""
    path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"

    temporary = _temp_sibling(path)

    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    # Deliberately not _atomic_replace: that seam is reserved for replacements
    # of live rolling data, and journal and state metadata are neither.
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def write_marker(path: Path) -> None:
    """Create a durable status marker.

    Recovery reads these markers after a crash, so the marker must be on disk
    before the step it authorises is allowed to proceed.
    """
    path.write_text("", encoding="utf-8")
    _fsync_file(path)
    _fsync_dir(path.parent)


class WriterLock:
    """Exclusive, non-blocking, advisory whole-refresh writer lock.

    fcntl.flock is available on Linux and macOS and is released by the kernel
    if this process dies, so a crashed refresh cannot wedge the next run.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)

            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                raise LockBusy(str(self.path))

            raise

        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
        os.fsync(fd)

        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def run_preflight(
    *,
    season: int,
    slate_date: str,
    opening_day: str | None,
    data_dir: Path,
) -> int:
    """Run the unmodified read-only preflight and return its exit code.

    The preflight prints a human-readable banner for operators. That text is
    deliberately neither captured nor inspected here: the process exit code is
    the only machine contract this wrapper honours.
    """
    args = [
        sys.executable,
        str(PREFLIGHT_SCRIPT),
        "--season",
        str(season),
        "--target-date",
        slate_date,
        "--require-advanced",
    ]

    if opening_day:
        args.extend(["--opening-day", opening_day])

    env = dict(os.environ)
    env["NBA_PROP_DATA_DIR"] = str(data_dir)

    completed = subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )

    return int(completed.returncode)


def run_shadow_ingest(*, season: int, shadow_root: Path) -> int:
    """Run the unmodified resume-safe ingester against a disposable root.

    NBA_PROP_DATA_DIR is the only thing that tells the existing ingester where
    to write, so pointing it at the shadow root is what keeps the live
    historical tree untouched while data is being fetched.
    """
    args = [
        sys.executable,
        str(INGEST_SCRIPT),
        "--start-season",
        str(season),
        "--end-season",
        str(season),
        "--include-advanced",
        "--skip-players",
        "--force",
    ]

    env = dict(os.environ)
    env["NBA_PROP_DATA_DIR"] = str(shadow_root)

    completed = subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )

    return int(completed.returncode)


def read_parquet_or_fail(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"{label} is missing: {path}",
        )

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"{label} is not readable parquet ({path}): {exc}",
        )


def require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
    path: Path,
) -> None:
    missing = [name for name in columns if name not in frame.columns]

    if missing:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"{label} is missing required columns {missing}: {path}",
        )


def parsed_dates(frame: pd.DataFrame, label: str, path: Path) -> pd.Series:
    dates = pd.to_datetime(frame["date"], errors="coerce")

    if dates.isna().any():
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"{label} has {int(dates.isna().sum())} unparseable date "
            f"value(s): {path}",
        )

    return dates.dt.normalize()


def numeric_id_set(frame: pd.DataFrame, column: str) -> set[int]:
    if column not in frame.columns:
        return set()

    values = pd.to_numeric(frame[column], errors="coerce").dropna()

    return set(values.astype("int64").tolist())


def assert_advanced_layout(advanced_root: Path, label: str) -> None:
    """Advanced parquet may exist only at season=<digits>/advanced.parquet.

    Anything else under the advanced tree would be silently concatenated by
    nba_prop_quant.pipeline.load_advanced, so an unexpected file is a stop,
    never a cleanup. Nothing is deleted here.
    """
    if not advanced_root.exists():
        return

    unexpected = []

    for path in sorted(advanced_root.rglob("*.parquet")):
        relative = path.relative_to(advanced_root)

        if (
            len(relative.parts) == 2
            and ADVANCED_SEASON_DIR.match(relative.parts[0])
            and relative.parts[1] == "advanced.parquet"
        ):
            continue

        unexpected.append(str(relative))

    if unexpected:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"{label} advanced tree contains unexpected parquet "
            f"{unexpected} under {advanced_root}. Nothing was deleted; "
            "resolve this by hand before refreshing.",
        )


def validate_staged_stats(path: Path) -> pd.DataFrame:
    frame = read_parquet_or_fail(path, "staged stats")

    if frame.empty:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"staged stats is empty: {path}",
        )

    require_columns(frame, STATS_REQUIRED_COLUMNS, "staged stats", path)
    parsed_dates(frame, "staged stats", path)

    if frame["game_id"].isna().all():
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"staged stats game_id is entirely null: {path}",
        )

    if frame["player_id"].isna().all():
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"staged stats player_id is entirely null: {path}",
        )

    return frame


def validate_staged_advanced(path: Path) -> pd.DataFrame:
    frame = read_parquet_or_fail(path, "staged advanced")

    if frame.empty:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"staged advanced is empty: {path}",
        )

    require_columns(frame, ADVANCED_REQUIRED_COLUMNS, "staged advanced", path)
    parsed_dates(frame, "staged advanced", path)

    return frame


def validate_staged_games(path: Path) -> pd.DataFrame:
    frame = read_parquet_or_fail(path, "staged games")
    require_columns(frame, GAMES_REQUIRED_COLUMNS, "staged games", path)

    return frame


def assert_games_cover(
    child: pd.DataFrame,
    games: pd.DataFrame,
    label: str,
) -> None:
    game_ids = numeric_id_set(games, GAMES_ID_COLUMN)
    referenced = numeric_id_set(child, "game_id")
    missing = sorted(referenced - game_ids)

    if missing:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"staged {label} references {len(missing)} game id(s) that are "
            f"absent from staged games: {missing[:20]}",
        )


def assert_no_leakage(
    frame: pd.DataFrame,
    label: str,
    path: Path,
    slate_date: pd.Timestamp,
) -> pd.Timestamp:
    """Delegate to the shared cutoff rule so staging and prediction agree.

    nba_prop_quant.slate owns the single definition of "history must end
    strictly before the slate date". Keeping a second copy here would let the
    staging gate and the prediction gate drift apart.
    """
    try:
        return assert_history_precedes_slate(
            frame,
            f"staged {label}",
            slate_date,
            path,
        )

    except HistoryLeakageError as exc:
        raise RefreshFailure(EXIT_VALIDATION_FAILED, str(exc)) from exc


def live_counts(paths: SeasonPaths) -> dict[str, dict[str, int]]:
    """Row and distinct-game counts of existing non-empty live state.

    A live file that exists but cannot be read is a stop, not an implicit
    floor of zero: an unreadable predecessor makes non-regression impossible
    to establish.
    """
    counts: dict[str, dict[str, int]] = {}

    for label, path, id_column in (
        ("stats", paths.stats, "game_id"),
        ("advanced", paths.advanced, "game_id"),
        ("games", paths.games, GAMES_ID_COLUMN),
    ):
        if not path.exists():
            continue

        frame = read_parquet_or_fail(path, f"live {label}")

        if frame.empty:
            continue

        counts[label] = {
            "rows": int(len(frame)),
            "games": len(numeric_id_set(frame, id_column)),
        }

    return counts


def assert_no_regression(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> None:
    checks = (
        ("stats", "rows", "row count"),
        ("stats", "games", "distinct game count"),
        ("advanced", "rows", "row count"),
        ("advanced", "games", "distinct game count"),
        ("games", "games", "distinct game count"),
    )

    regressions = []

    for label, metric, description in checks:
        if label not in before:
            continue

        previous = before[label][metric]
        staged = after[label][metric]

        if staged < previous:
            regressions.append(
                f"{label} {description} fell from {previous} to {staged}"
            )

    if regressions:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            "staged current-season state regresses against live state: "
            + "; ".join(regressions)
            + ". A decrease is never auto-accepted as a correction.",
        )


def print_counts(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> None:
    print()
    print(f"{'file':<10}{'live rows':>12}{'staged rows':>14}"
          f"{'live games':>13}{'staged games':>15}")

    for label in ("stats", "advanced", "games"):
        live = before.get(label, {})
        staged = after.get(label, {})

        print(
            f"{label:<10}"
            f"{live.get('rows', '-'):>12}"
            f"{staged.get('rows', '-'):>14}"
            f"{live.get('games', '-'):>13}"
            f"{staged.get('games', '-'):>15}"
        )

    print()


def validate_staged_tree(
    staged: SeasonPaths,
    live: SeasonPaths,
    slate_date: pd.Timestamp,
) -> dict[str, dict[str, int]]:
    assert_advanced_layout(staged.advanced_root, "staged")

    stats = validate_staged_stats(staged.stats)
    advanced = validate_staged_advanced(staged.advanced)
    games = validate_staged_games(staged.games)

    assert_games_cover(stats, games, "stats")
    assert_games_cover(advanced, games, "advanced")

    stats_max = assert_no_leakage(stats, "stats", staged.stats, slate_date)
    advanced_max = assert_no_leakage(
        advanced,
        "advanced",
        staged.advanced,
        slate_date,
    )

    print(
        f"latest staged stats date {stats_max.date()}, "
        f"latest staged advanced date {advanced_max.date()}, "
        f"slate date {slate_date.date()}"
    )

    before = live_counts(live)
    after = {
        "stats": {
            "rows": int(len(stats)),
            "games": len(numeric_id_set(stats, "game_id")),
        },
        "advanced": {
            "rows": int(len(advanced)),
            "games": len(numeric_id_set(advanced, "game_id")),
        },
        "games": {
            "rows": int(len(games)),
            "games": len(numeric_id_set(games, GAMES_ID_COLUMN)),
        },
    }

    print_counts(before, after)
    assert_no_regression(before, after)

    return after


def semantic_fingerprint(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> str:
    """Hash the logical content of a rolling dataset.

    Deliberately independent of parquet byte layout. The frame is ordered by
    its key columns and its columns are ordered by name, so rewriting the same
    records with a different row order, a different compression setting or a
    newer parquet writer yields the same fingerprint, while any changed,
    added or removed record changes it.
    """
    ordered = sort_by_keys(frame, list(key_columns))
    ordered = ordered[sorted(ordered.columns)]

    canonical = ordered.copy()

    for column in canonical.columns:
        values = canonical[column]

        if pd.api.types.is_datetime64_any_dtype(values):
            # Normalize to a stable textual form so timezone or unit changes
            # in the storage layer cannot move the fingerprint.
            canonical[column] = values.dt.strftime("%Y-%m-%dT%H:%M:%S")

    payload = canonical.to_csv(index=False, float_format="%.12g")

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_state_record(
    frame: pd.DataFrame,
    label: str,
    path: Path,
    data_root: Path,
) -> dict:
    """Describe one rolling dataset semantically, with no host-private paths."""
    key_columns = ROLLING_KEY_COLUMNS[label]

    record: dict = {
        "columns": sorted(str(column) for column in frame.columns),
        "key_columns": list(key_columns),
        "relative_path": path.relative_to(data_root).as_posix(),
        "row_count": int(len(frame)),
        "schema_fingerprint": hashlib.sha256(
            json.dumps(
                {
                    str(column): str(frame[column].dtype)
                    for column in sorted(frame.columns)
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "semantic_fingerprint": semantic_fingerprint(frame, key_columns),
        "unique_key_count": int(
            len(frame.drop_duplicates(subset=list(key_columns)))
        ),
    }

    if ROLLING_DATE_COLUMN in frame.columns:
        dates = pd.to_datetime(frame[ROLLING_DATE_COLUMN], errors="coerce")
        usable = dates.dropna()

        record["min_date"] = (
            usable.min().date().isoformat() if not usable.empty else None
        )
        record["max_date"] = (
            usable.max().date().isoformat() if not usable.empty else None
        )

    game_column = GAMES_ID_COLUMN if label == "games" else "game_id"

    if game_column in frame.columns:
        record["game_count"] = len(numeric_id_set(frame, game_column))

    return record


def build_state_record(
    live: SeasonPaths,
    data_root: Path,
    season: int,
    slate_date: pd.Timestamp,
) -> dict:
    """Read the committed live tree back and describe it semantically."""
    datasets = {
        label: dataset_state_record(
            read_parquet_or_fail(path, f"committed {label}"),
            label,
            path,
            data_root,
        )
        for label, path in live.as_items()
    }

    return {
        "datasets": datasets,
        # Fingerprint of the datasets block alone, so an identical live tree
        # always produces an identical value regardless of when it was written.
        "datasets_fingerprint": hashlib.sha256(
            json.dumps(datasets, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": STATE_SCHEMA_VERSION,
        "season": int(season),
        "slate_date": slate_date.date().isoformat(),
    }


def write_state_record(
    live: SeasonPaths,
    data_root: Path,
    season: int,
    slate_date: pd.Timestamp,
    expected_counts: dict[str, dict[str, int]],
) -> dict:
    """Validate the committed tree semantically, then record it atomically."""
    record = build_state_record(live, data_root, season, slate_date)

    for label, expected in expected_counts.items():
        observed = record["datasets"][label]["row_count"]

        if observed != expected["rows"]:
            raise RefreshFailure(
                EXIT_VALIDATION_FAILED,
                f"committed {label} holds {observed} rows but validation "
                f"approved {expected['rows']}",
            )

    write_json_durable(record, data_root / STATE_RELATIVE_PATH)

    return record


# ----------------------------------------------------------------------
# crash-recoverable multi-file transaction
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TransactionTarget:
    label: str
    relative_path: str
    existed: bool
    old_sha256: str | None
    new_sha256: str

    def as_dict(self) -> dict:
        return {
            "existed": self.existed,
            "label": self.label,
            "new_sha256": self.new_sha256,
            "old_sha256": self.old_sha256,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TransactionTarget":
        return cls(
            label=str(payload["label"]),
            relative_path=str(payload["relative_path"]),
            existed=bool(payload["existed"]),
            old_sha256=payload["old_sha256"],
            new_sha256=str(payload["new_sha256"]),
        )


@dataclass(frozen=True)
class RefreshTransaction:
    root: Path
    data_root: Path
    targets: tuple[TransactionTarget, ...]

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def backups_dir(self) -> Path:
        return self.root / "backups"

    @property
    def prepared_marker(self) -> Path:
        return self.root / PREPARED_MARKER

    @property
    def committed_marker(self) -> Path:
        return self.root / COMMITTED_MARKER

    def backup_path(self, label: str) -> Path:
        return self.backups_dir / f"{label}.parquet"

    def destination(self, target: TransactionTarget) -> Path:
        return self.data_root / target.relative_path


def transactions_root(data_root: Path) -> Path:
    return data_root / TRANSACTIONS_RELATIVE_PATH


def prepare_transaction(
    staged: SeasonPaths,
    live: SeasonPaths,
    data_root: Path,
    season: int,
    slate_date: pd.Timestamp,
) -> RefreshTransaction:
    """Record every target and preserve the bytes needed to undo the change.

    Nothing live may be replaced until this returns: the PREPARED marker is
    what tells a later recovery that a replacement may have started.
    """
    root = transactions_root(data_root) / (
        f"{TRANSACTION_PREFIX}{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_"
        f"{uuid.uuid4().hex[:12]}"
    )

    backups = root / "backups"
    backups.mkdir(parents=True, exist_ok=True)

    staged_paths = dict(staged.as_items())

    targets: list[TransactionTarget] = []

    for label, destination in live.as_items():
        staged_path = staged_paths[label]
        existed = destination.exists()

        old_hash = None

        if existed:
            backup = backups / f"{label}.parquet"
            shutil.copyfile(destination, backup)
            _fsync_file(backup)

            old_hash = sha256_file(backup)

            if old_hash != sha256_file(destination):
                raise RefreshFailure(
                    EXIT_VALIDATION_FAILED,
                    f"live {label} changed while it was being backed up; "
                    "refusing to start a refresh transaction.",
                )

        targets.append(
            TransactionTarget(
                label=label,
                relative_path=destination.relative_to(data_root).as_posix(),
                existed=existed,
                old_sha256=old_hash,
                new_sha256=sha256_file(staged_path),
            )
        )

    _fsync_dir(backups)

    manifest = {
        "data_root_is_relative": True,
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "season": int(season),
        "slate_date": slate_date.date().isoformat(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": [target.as_dict() for target in targets],
    }

    transaction = RefreshTransaction(
        root=root,
        data_root=data_root,
        targets=tuple(targets),
    )

    write_json_durable(manifest, transaction.manifest_path)
    write_marker(transaction.prepared_marker)

    return transaction


def load_transaction(root: Path, data_root: Path) -> RefreshTransaction | None:
    """Rebuild a transaction from its journal, or None if it is unreadable."""
    manifest_path = root / "manifest.json"

    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        targets = tuple(
            TransactionTarget.from_dict(entry)
            for entry in manifest["targets"]
        )

    except (OSError, ValueError, KeyError, TypeError):
        return None

    return RefreshTransaction(root=root, data_root=data_root, targets=targets)


def discard_transaction(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def restore_target(
    transaction: RefreshTransaction,
    target: TransactionTarget,
) -> None:
    """Put one target back to its pre-transaction state, verifying hashes."""
    destination = transaction.destination(target)

    if not target.existed:
        if destination.exists():
            destination.unlink()

        if destination.exists():
            raise RefreshFailure(
                EXIT_RECOVERY_FAILED,
                f"could not remove {destination}, which did not exist before "
                "the interrupted refresh",
            )

        return

    backup = transaction.backup_path(target.label)

    if not backup.exists():
        raise RefreshFailure(
            EXIT_RECOVERY_FAILED,
            f"backup for {target.label} is missing at {backup}; the previous "
            "generation cannot be restored automatically.",
        )

    if sha256_file(backup) != target.old_sha256:
        raise RefreshFailure(
            EXIT_RECOVERY_FAILED,
            f"backup for {target.label} is corrupt: {backup} does not match "
            f"the recorded hash {target.old_sha256}. The live current-season "
            "tree needs manual inspection before any prediction or pricing "
            "run.",
        )

    temporary = _temp_sibling(destination)

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(backup, temporary)
        _atomic_replace(temporary, destination)

    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)

    if sha256_file(destination) != target.old_sha256:
        raise RefreshFailure(
            EXIT_RECOVERY_FAILED,
            f"restoring {destination} did not reproduce the pre-transaction "
            "bytes",
        )


def recover_transaction(transaction: RefreshTransaction) -> str:
    """Resolve one interrupted transaction to a single coherent generation.

    Rolling back is preferred. Rolling forward is only allowed when every
    target already matches the validated new generation, which proves the
    replacements all completed and only the commit marker was lost.
    """
    observed: dict[str, str | None] = {}

    for target in transaction.targets:
        destination = transaction.destination(target)

        observed[target.label] = (
            sha256_file(destination) if destination.exists() else None
        )

    complete_new = all(
        observed[target.label] == target.new_sha256
        for target in transaction.targets
    )

    if complete_new:
        return (
            "rolled forward: every target already matched the validated new "
            "generation, so only the commit marker was missing"
        )

    for target in transaction.targets:
        restore_target(transaction, target)

    return (
        "rolled back to the complete pre-transaction generation: "
        + ", ".join(
            f"{target.label}="
            + ("restored" if target.existed else "removed")
            for target in transaction.targets
        )
    )


def recover_incomplete_transactions(data_root: Path) -> list[str]:
    """Resolve any interrupted refresh before a new one is allowed to start."""
    root = transactions_root(data_root)

    if not root.is_dir():
        return []

    outcomes: list[str] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith(TRANSACTION_PREFIX):
            continue

        if (entry / COMMITTED_MARKER).exists():
            discard_transaction(entry)
            continue

        if not (entry / PREPARED_MARKER).exists():
            # The marker is written before the first replacement, so its
            # absence proves no live file was touched.
            discard_transaction(entry)
            outcomes.append(
                f"{entry.name}: discarded, no live file had been replaced"
            )
            continue

        transaction = load_transaction(entry, data_root)

        if transaction is None:
            raise RefreshFailure(
                EXIT_RECOVERY_FAILED,
                f"interrupted refresh {entry.name} is prepared but its "
                "manifest is unreadable; the live current-season tree needs "
                "manual inspection before any prediction or pricing run.",
            )

        outcome = recover_transaction(transaction)

        discard_transaction(entry)
        outcomes.append(f"{entry.name}: {outcome}")

    return outcomes


def rollback(
    entries: list[tuple[str, Path, Path, str]],
    backups: dict[str, Path],
    original_hashes: dict[str, str],
) -> list[str]:
    """Return the live tree to its pre-commit bytes.

    Returns the list of paths whose restoration could not be verified. An
    empty list means every previously existing destination is byte-identical
    to its pre-commit state and every newly created destination is gone.
    """
    failures: list[str] = []

    for label, _staged, destination, _staged_hash in entries:
        try:
            if label in backups:
                temporary = _temp_sibling(destination)
                shutil.copyfile(backups[label], temporary)

                try:
                    _atomic_replace(temporary, destination)
                finally:
                    if temporary.exists():
                        temporary.unlink(missing_ok=True)

                if sha256_file(destination) != original_hashes[label]:
                    failures.append(str(destination))

            else:
                if destination.exists():
                    destination.unlink()

                if destination.exists():
                    failures.append(str(destination))

        except Exception as exc:
            failures.append(f"{destination} ({exc})")

    return failures


def commit_staged_bytes(
    staged: SeasonPaths,
    live: SeasonPaths,
    transaction: RefreshTransaction,
) -> dict[str, str]:
    """Copy the exact staged bytes over the live destinations.

    The staged files are never rewritten from a DataFrame, so row order,
    column order, compression and parquet metadata are exactly whatever the
    existing writer produced.

    The prepared transaction already holds the pre-replacement bytes and both
    content hashes of every target, so an in-process failure rolls back here
    and a process loss is resolved by recovery on the next run.
    """
    staged_paths = dict(staged.as_items())
    live_paths = dict(live.as_items())

    entries: list[tuple[str, Path, Path, str]] = [
        (
            target.label,
            staged_paths[target.label],
            live_paths[target.label],
            target.new_sha256,
        )
        for target in transaction.targets
    ]

    backups: dict[str, Path] = {
        target.label: transaction.backup_path(target.label)
        for target in transaction.targets
        if target.existed
    }

    original_hashes: dict[str, str] = {
        target.label: str(target.old_sha256)
        for target in transaction.targets
        if target.existed
    }

    for label, staged_path, destination, staged_hash in entries:
        temporary = _temp_sibling(destination)

        try:
            # Refuse a staged file that changed after it was validated and
            # recorded, rather than publishing unvalidated bytes.
            observed = sha256_file(staged_path)

            if observed != staged_hash:
                raise RuntimeError(
                    f"staged {label} changed after validation: {observed} "
                    f"does not match the prepared hash {staged_hash}"
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged_path, temporary)
            _atomic_replace(temporary, destination)

        except Exception as exc:
            failures = rollback(entries, backups, original_hashes)

            if failures:
                raise RefreshFailure(
                    EXIT_ROLLBACK_FAILED,
                    f"replacing {destination} failed ({exc}) AND rollback "
                    f"could not be verified for {failures}. The live "
                    "current-season tree needs manual inspection before any "
                    "prediction or pricing run.",
                )

            raise RefreshFailure(
                EXIT_COMMIT_ROLLED_BACK,
                f"replacing {destination} failed ({exc}); the previous live "
                "bytes were restored and verified.",
            )

        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

        print(f"committed {label}: {destination} ({staged_hash})")

    mismatched = [
        str(destination)
        for _label, _staged_path, destination, staged_hash in entries
        if sha256_file(destination) != staged_hash
    ]

    if mismatched:
        failures = rollback(entries, backups, original_hashes)

        if failures:
            raise RefreshFailure(
                EXIT_ROLLBACK_FAILED,
                f"post-commit hash verification failed for {mismatched} AND "
                f"rollback could not be verified for {failures}. The live "
                "current-season tree needs manual inspection before any "
                "prediction or pricing run.",
            )

        raise RefreshFailure(
            EXIT_POST_COMMIT_ROLLED_BACK,
            f"post-commit hash verification failed for {mismatched}; the "
            "previous live bytes were restored and verified.",
        )

    return {
        label: staged_hash
        for label, _staged_path, _destination, staged_hash in entries
    }


def refresh(
    *,
    season: int,
    slate_date: pd.Timestamp,
    data_dir: Path,
) -> int:
    live = season_paths(data_dir, season)

    assert_advanced_layout(live.advanced_root, "live")

    data_dir.mkdir(parents=True, exist_ok=True)

    shadow_root = Path(tempfile.mkdtemp(prefix=SHADOW_PREFIX, dir=data_dir))

    try:
        print(f"shadow data root: {shadow_root}")

        code = run_shadow_ingest(season=season, shadow_root=shadow_root)

        if code != 0:
            raise RefreshFailure(
                EXIT_INGEST_FAILED,
                f"shadow ingestion exited {code}; the live current-season "
                "tree was not touched.",
            )

        staged = season_paths(shadow_root, season)
        approved = validate_staged_tree(staged, live, slate_date)

        transaction = prepare_transaction(
            staged,
            live,
            data_dir,
            season,
            slate_date,
        )

        print(f"refresh transaction: {transaction.root.name}")

        try:
            committed = commit_staged_bytes(staged, live, transaction)

        except RefreshFailure as exc:
            # commit_staged_bytes rolls back in process and reports the
            # outcome. The journal is only kept when that rollback could not
            # be verified, because then its backups are the remaining way
            # back to the previous generation.
            if exc.code != EXIT_ROLLBACK_FAILED:
                discard_transaction(transaction.root)

            raise

        try:
            state = write_state_record(
                live,
                data_dir,
                season,
                slate_date,
                approved,
            )

        except RefreshFailure:
            for target in transaction.targets:
                restore_target(transaction, target)

            discard_transaction(transaction.root)
            raise

        write_marker(transaction.committed_marker)
        discard_transaction(transaction.root)

        print()
        print("refresh committed; live files match staged bytes:")

        for label, digest in committed.items():
            print(f"  {label}: {digest}")

        print()
        print(
            "rolling-state integrity record: "
            f"{STATE_RELATIVE_PATH.as_posix()} "
            f"({state['datasets_fingerprint']})"
        )

        return EXIT_OK

    finally:
        shutil.rmtree(shadow_root, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed wrapper around the existing whole-season "
            "current-season history refresh. Fetches into a disposable "
            "shadow root, validates the staged files, and only then "
            "replaces live state with the exact staged bytes."
        )
    )

    parser.add_argument(
        "--slate-date",
        required=True,
        help="Slate date YYYY-MM-DD. Staged history must end strictly before it.",
    )

    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Live data root. Defaults to NBA_PROP_DATA_DIR, then to the "
            "repository's data directory."
        ),
    )

    parser.add_argument(
        "--opening-day",
        default=None,
        help="Optional opening-day override forwarded to the preflight.",
    )

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    slate_date = exact_date(args.slate_date, "--slate-date")
    season = int(args.season)
    data_dir = resolve_data_dir(args.data_dir)

    print(BANNER)
    print("CURRENT-SEASON ROLLING STATE REFRESH — FAIL CLOSED")
    print(BANNER)
    print(f"season:     {season}")
    print(f"slate date: {args.slate_date}")
    print(f"data root:  {data_dir}")
    print()

    lock = WriterLock(data_dir / LOCK_RELATIVE_PATH)

    try:
        lock.acquire()
    except LockBusy:
        print(
            "ANOTHER REFRESH IS ALREADY RUNNING. No preflight was run, no "
            "data was fetched, and nothing was modified.",
            file=sys.stderr,
        )
        return EXIT_LOCK_BUSY

    try:
        # An interrupted refresh is resolved to one coherent generation before
        # anything else is allowed to touch the rolling tree.
        for outcome in recover_incomplete_transactions(data_dir):
            print(f"recovered interrupted refresh {outcome}")
            print()

        code = run_preflight(
            season=season,
            slate_date=args.slate_date,
            opening_day=args.opening_day,
            data_dir=data_dir,
        )

        if code in PREFLIGHT_STOP_CODES:
            print()
            print(
                f"preflight exit code {code} withholds readiness; no data "
                "was fetched and no historical file was modified."
            )
            return code

        if code != EXIT_OK:
            print()
            print(
                f"preflight exited with the unexpected code {code}; failing "
                "closed without fetching anything.",
                file=sys.stderr,
            )
            return EXIT_PREFLIGHT_UNEXPECTED

        return refresh(
            season=season,
            slate_date=slate_date,
            data_dir=data_dir,
        )

    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        return run(args)
    except RefreshFailure as exc:
        print()
        print(f"REFRESH FAILED (exit {exc.code}): {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
