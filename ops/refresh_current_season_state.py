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
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREFLIGHT_SCRIPT = PROJECT_ROOT / "ops" / "preflight_current_season_refresh.py"
INGEST_SCRIPT = PROJECT_ROOT / "scripts" / "01_ingest_history_resume.py"

LOCK_RELATIVE_PATH = Path(".locks") / "current_season_refresh.lock"

SHADOW_PREFIX = ".refresh_shadow_"
ROLLBACK_PREFIX = ".refresh_rollback_"

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
    latest = parsed_dates(frame, f"staged {label}", path).max()

    if pd.isna(latest):
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"staged {label} has no usable date: {path}",
        )

    if latest >= slate_date:
        raise RefreshFailure(
            EXIT_VALIDATION_FAILED,
            f"staged {label} would leak same-or-later-day results into the "
            f"information set: max date {latest.date()} is not strictly "
            f"before slate date {slate_date.date()}",
        )

    return latest


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
    rollback_dir: Path,
) -> dict[str, str]:
    """Copy the exact staged bytes over the live destinations.

    The staged files are never rewritten from a DataFrame, so row order,
    column order, compression and parquet metadata are exactly whatever the
    existing writer produced.
    """
    entries: list[tuple[str, Path, Path, str]] = []

    for label, staged_path in staged.as_items():
        destination = dict(live.as_items())[label]
        entries.append(
            (label, staged_path, destination, sha256_file(staged_path))
        )

    backups: dict[str, Path] = {}
    original_hashes: dict[str, str] = {}

    for label, _staged_path, destination, _staged_hash in entries:
        if not destination.exists():
            continue

        backup = rollback_dir / f"{label}.parquet"
        shutil.copyfile(destination, backup)

        backups[label] = backup
        original_hashes[label] = sha256_file(backup)

    for label, staged_path, destination, staged_hash in entries:
        temporary = _temp_sibling(destination)

        try:
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
    rollback_dir = Path(
        tempfile.mkdtemp(prefix=ROLLBACK_PREFIX, dir=data_dir)
    )

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
        validate_staged_tree(staged, live, slate_date)

        committed = commit_staged_bytes(staged, live, rollback_dir)

        print()
        print("refresh committed; live files match staged bytes:")

        for label, digest in committed.items():
            print(f"  {label}: {digest}")

        return EXIT_OK

    finally:
        shutil.rmtree(shadow_root, ignore_errors=True)
        shutil.rmtree(rollback_dir, ignore_errors=True)


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
