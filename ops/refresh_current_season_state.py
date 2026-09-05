"""Safe, unattended current-season historical-state refresh (Stage 2).

This wrapper does not implement a second ingestion path. It orchestrates the
existing, authoritative components:

    ops/preflight_current_season_refresh.py   readiness gate (read only)
    scripts/01_ingest_history_resume.py       whole-season writer

The existing writers keep their whole-season replacement semantics and their
exact physical row order. This wrapper only decides *whether* their output is
allowed to become the live current-season historical state, by fetching into a
throwaway shadow data root, validating the resulting files there, and then
replacing the three live files with the validated bytes.

Guarantee boundary (see also STAGE 2 TRANSACTION LIMITATION below):

    * one writer at a time (exclusive lock held across the whole operation)
    * every candidate file fully validated before any live file is touched
    * per-file atomic replacement on the same filesystem
    * rollback of already-replaced files when a commit step raises
    * post-commit SHA-256 verification of every destination

STAGE 2 TRANSACTION LIMITATION: replacing three files with three separate
os.replace() calls is NOT a single atomic three-file transaction. Each file is
replaced atomically on its own. A process kill or machine loss between two
replacements can still leave the three live files mutually inconsistent. That
residual failure mode is deliberately out of scope for Stage 2 and is not
solved here.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREFLIGHT_SCRIPT = PROJECT_ROOT / "ops/preflight_current_season_refresh.py"
INGEST_SCRIPT = PROJECT_ROOT / "scripts/01_ingest_history_resume.py"

READY_ACTION = "READY_TO_REFRESH"

# Inference-required standard columns established by the audit. These are a
# strictly stronger acceptance criterion than the resume script's seven-column
# post-download check, which is deliberately not reused here.
REQUIRED_STATS_COLUMNS = (
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

REQUIRED_ADVANCED_COLUMNS = (
    "player_id",
    "game_id",
    "date",
)

# ops/capture_external_test_day.py::latest_raw_history_date() and
# ops/grade_external_test_capture.py read the normalized games frame by "id"
# and "date". The game-id column in games.parquet is "id", not "game_id".
REQUIRED_GAMES_COLUMNS = (
    "id",
    "date",
)

# grade_external_test_capture.py::is_final_game() decides finality from these
# two columns, so at least one of them must be present for grading to work.
GAMES_FINAL_STATUS_COLUMNS = (
    "status",
    "status_state",
)

GAMES_ID_COLUMN = "id"

EXIT_OK = 0
EXIT_VALIDATION_FAILED = 40
EXIT_PREFLIGHT_UNRECOGNIZED = 41
EXIT_INGEST_FAILED = 50
EXIT_COMMIT_FAILED = 60
EXIT_ROLLBACK_FAILED = 61
EXIT_ALREADY_LOCKED = 75


class RefreshError(RuntimeError):
    """A refresh was refused or failed. No live file was left unvalidated."""


class ValidationError(RefreshError):
    """A staged candidate file failed an acceptance check."""


class AlreadyLockedError(RefreshError):
    """Another refresh process already holds the exclusive lock."""


class CommitError(RefreshError):
    """A live replacement failed after rollback was attempted."""


class RollbackError(RefreshError):
    """A live replacement failed and rollback could not be verified."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def exact_date(value: str, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise SystemExit(f"ERROR: invalid {label} {value!r}: {exc}")

    if parsed.strftime("%Y-%m-%d") != value:
        raise SystemExit(
            f"ERROR: {label} must be exact YYYY-MM-DD; got {value!r}"
        )

    return parsed.normalize()


def stats_path_for(data_dir: Path, season: int) -> Path:
    return data_dir / "raw" / "seasons" / f"season={season}" / "stats.parquet"


def games_path_for(data_dir: Path, season: int) -> Path:
    return data_dir / "raw" / "seasons" / f"season={season}" / "games.parquet"


def advanced_path_for(data_dir: Path, season: int) -> Path:
    return data_dir / "raw" / "advanced" / f"season={season}" / "advanced.parquet"


def advanced_root_for(data_dir: Path) -> Path:
    return data_dir / "raw" / "advanced"


def numeric_id_set(frame: pd.DataFrame, column: str) -> set[int]:
    if column not in frame.columns:
        return set()

    values = pd.to_numeric(frame[column], errors="coerce").dropna()

    return {int(value) for value in values.tolist()}


def read_parquet_or_fail(path: Path, label: str) -> pd.DataFrame:
    """Read a candidate file for validation only. Never written back out."""

    if not path.exists():
        raise ValidationError(f"{label}: missing file {path}")

    if path.stat().st_size == 0:
        raise ValidationError(f"{label}: zero-byte file {path}")

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise ValidationError(f"{label}: unreadable parquet {path}: {exc}")


def parseable_dates(
    frame: pd.DataFrame,
    label: str,
) -> pd.Series:
    try:
        dates = pd.to_datetime(frame["date"], errors="coerce")
    except Exception as exc:
        raise ValidationError(f"{label}: date column is not parseable: {exc}")

    if len(dates) == 0 or not bool(dates.notna().all()):
        raise ValidationError(
            f"{label}: date column contains values that do not parse as dates"
        )

    return pd.to_datetime(dates).dt.normalize()


# ---------------------------------------------------------------------------
# single-writer lock
# ---------------------------------------------------------------------------


def lock_path_for(data_dir: Path) -> Path:
    return data_dir / ".locks" / "current_season_refresh.lock"


@contextmanager
def refresh_lock(lock_path: Path) -> Iterator[Path]:
    """Exclusive, non-blocking, stdlib-only writer lock (fcntl.flock).

    A second concurrent refresh fails closed with AlreadyLockedError instead of
    waiting, so two processes can never fetch or commit historical state at the
    same time. Readers (prediction) are intentionally not required to take it.
    """

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    handle = lock_path.open("a+")

    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise AlreadyLockedError(
                    f"ALREADY_LOCKED: another refresh holds {lock_path}"
                )
            raise

        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
        except OSError:
            pass

        yield lock_path

    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


# ---------------------------------------------------------------------------
# preflight reuse (the preflight script itself is not modified)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightResult:
    returncode: int
    action: str | None
    payload: dict
    stdout: str = ""
    stderr: str = ""

    @property
    def ready(self) -> bool:
        return self.returncode == 0 and self.action == READY_ACTION

    @property
    def final_game_ids(self) -> set[int] | None:
        raw = self.payload.get("latest_final_game_ids")

        if not isinstance(raw, list) or not raw:
            return None

        try:
            return {int(value) for value in raw}
        except (TypeError, ValueError):
            return None

    @property
    def advanced_required(self) -> bool:
        return bool(self.payload.get("require_advanced"))


def parse_preflight_output(text: str) -> tuple[str | None, dict]:
    """Read the preflight's own structured status line and JSON payload.

    The preflight emits `Action: <STATUS>` followed by a json.dumps() payload.
    Only those machine-emitted fields are consumed; no human-readable prose is
    interpreted.
    """

    action: str | None = None

    for line in text.splitlines():
        if line.startswith("Action: "):
            action = line[len("Action: ") :].strip()
            break

    payload: dict = {}
    decoder = json.JSONDecoder()

    for index, character in enumerate(text):
        if character != "{":
            continue

        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue

        if isinstance(candidate, dict):
            payload = candidate
            break

    return action, payload


def run_preflight(
    *,
    season: int,
    slate_date: str,
    opening_day: str | None = None,
    require_advanced: bool = True,
    python_executable: str | None = None,
) -> PreflightResult:
    """Invoke the existing read-only preflight exactly as production does."""

    if not PREFLIGHT_SCRIPT.exists():
        raise RefreshError(f"missing preflight script: {PREFLIGHT_SCRIPT}")

    command = [
        python_executable or sys.executable,
        str(PREFLIGHT_SCRIPT),
        "--season",
        str(season),
        "--target-date",
        slate_date,
    ]

    if opening_day:
        command += ["--opening-day", opening_day]

    if require_advanced:
        command.append("--require-advanced")

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    action, payload = parse_preflight_output(completed.stdout)

    return PreflightResult(
        returncode=completed.returncode,
        action=action,
        payload=payload,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


# ---------------------------------------------------------------------------
# shadow ingestion (the ingestion scripts themselves are not modified)
# ---------------------------------------------------------------------------


def subprocess_env(shadow_data_dir: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)

    source_dir = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        source_dir if not existing else f"{source_dir}{os.pathsep}{existing}"
    )

    if shadow_data_dir is not None:
        env["NBA_PROP_DATA_DIR"] = str(shadow_data_dir)

    return env


def run_shadow_ingest(
    *,
    shadow_data_dir: Path,
    season: int,
    python_executable: str | None = None,
) -> None:
    """Run the existing resume-safe ingester against a throwaway data root.

    Settings.nba_prop_data_dir is populated from NBA_PROP_DATA_DIR, and
    environment variables outrank the repository .env file, so the existing
    script writes every artifact under the shadow root. No live path is passed
    to the ingester and no live file is opened for writing during the fetch.
    """

    if not INGEST_SCRIPT.exists():
        raise RefreshError(f"missing ingestion script: {INGEST_SCRIPT}")

    command = [
        python_executable or sys.executable,
        str(INGEST_SCRIPT),
        "--start-season",
        str(season),
        "--end-season",
        str(season),
        "--include-advanced",
        "--skip-players",
        "--force",
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=subprocess_env(shadow_data_dir),
        check=False,
    )

    if completed.returncode != 0:
        raise RefreshError(
            "shadow ingestion failed with exit code "
            f"{completed.returncode}; live files were not touched"
        )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@dataclass
class StoreSummary:
    path: Path
    rows: int
    game_ids: set[int] = field(default_factory=set)
    max_date: pd.Timestamp | None = None
    sha256: str = ""

    @property
    def distinct_games(self) -> int:
        return len(self.game_ids)


def validate_stats(path: Path, slate_date: pd.Timestamp) -> StoreSummary:
    label = "staged stats.parquet"
    frame = read_parquet_or_fail(path, label)

    if frame.empty:
        raise ValidationError(f"{label}: empty; refusing to replace live state")

    missing = [
        column
        for column in REQUIRED_STATS_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValidationError(
            f"{label}: missing inference-required columns {sorted(missing)}"
        )

    dates = parseable_dates(frame, label)

    game_ids = numeric_id_set(frame, "game_id")

    if not game_ids:
        raise ValidationError(f"{label}: game_id is entirely null")

    if not numeric_id_set(frame, "player_id"):
        raise ValidationError(f"{label}: player_id is entirely null")

    max_date = pd.Timestamp(dates.max()).normalize()

    if max_date >= slate_date:
        raise ValidationError(
            f"{label}: leakage guard failed; max date {max_date.date()} is not "
            f"strictly before slate date {slate_date.date()}"
        )

    return StoreSummary(
        path=path,
        rows=len(frame),
        game_ids=game_ids,
        max_date=max_date,
        sha256=sha256_file(path),
    )


def validate_advanced(path: Path, slate_date: pd.Timestamp) -> StoreSummary:
    label = "staged advanced.parquet"
    frame = read_parquet_or_fail(path, label)

    if frame.empty:
        raise ValidationError(f"{label}: empty; refusing to replace live state")

    missing = [
        column
        for column in REQUIRED_ADVANCED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValidationError(
            f"{label}: missing inference-required columns {sorted(missing)}"
        )

    dates = parseable_dates(frame, label)

    game_ids = numeric_id_set(frame, "game_id")

    if not game_ids:
        raise ValidationError(f"{label}: game_id is entirely null")

    if not numeric_id_set(frame, "player_id"):
        raise ValidationError(f"{label}: player_id is entirely null")

    max_date = pd.Timestamp(dates.max()).normalize()

    if max_date >= slate_date:
        raise ValidationError(
            f"{label}: leakage guard failed; max date {max_date.date()} is not "
            f"strictly before slate date {slate_date.date()}"
        )

    return StoreSummary(
        path=path,
        rows=len(frame),
        game_ids=game_ids,
        max_date=max_date,
        sha256=sha256_file(path),
    )


def validate_games(path: Path) -> StoreSummary:
    """Games are not leakage-guarded: future scheduled games belong here."""

    label = "staged games.parquet"
    frame = read_parquet_or_fail(path, label)

    if frame.empty:
        raise ValidationError(f"{label}: empty; refusing to replace live state")

    missing = [
        column
        for column in REQUIRED_GAMES_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValidationError(
            f"{label}: missing operational columns {sorted(missing)} required "
            "by latest_raw_history_date() and grade_external_test_capture.py"
        )

    if not any(
        column in frame.columns for column in GAMES_FINAL_STATUS_COLUMNS
    ):
        raise ValidationError(
            f"{label}: has neither {GAMES_FINAL_STATUS_COLUMNS[0]} nor "
            f"{GAMES_FINAL_STATUS_COLUMNS[1]}; grading cannot confirm finality"
        )

    parseable_dates(frame, label)

    game_ids = numeric_id_set(frame, GAMES_ID_COLUMN)

    if not game_ids:
        raise ValidationError(f"{label}: {GAMES_ID_COLUMN} is entirely null")

    return StoreSummary(
        path=path,
        rows=len(frame),
        game_ids=game_ids,
        sha256=sha256_file(path),
    )


def assert_referenced_games_present(
    *,
    stats: StoreSummary,
    advanced: StoreSummary,
    games: StoreSummary,
) -> None:
    missing_from_stats = sorted(stats.game_ids - games.game_ids)
    missing_from_advanced = sorted(advanced.game_ids - games.game_ids)

    if missing_from_stats:
        raise ValidationError(
            "staged stats.parquet references game ids absent from staged "
            f"games.parquet: {missing_from_stats[:20]}"
        )

    if missing_from_advanced:
        raise ValidationError(
            "staged advanced.parquet references game ids absent from staged "
            f"games.parquet: {missing_from_advanced[:20]}"
        )


def assert_canonical_advanced_layout(advanced_root: Path, label: str) -> None:
    """load_advanced() recursively globs every *.parquet under raw/advanced.

    Anything outside data/raw/advanced/season=*/advanced.parquet would silently
    join the model's advanced inputs, so an unexpected file fails closed. The
    file is reported, never deleted.
    """

    root = Path(advanced_root)

    if not root.exists():
        return

    unexpected: list[str] = []

    for candidate in sorted(root.rglob("*.parquet")):
        relative = candidate.relative_to(root)
        parts = relative.parts

        canonical = (
            len(parts) == 2
            and parts[0].startswith("season=")
            and parts[0][len("season=") :].isdigit()
            and parts[1] == "advanced.parquet"
        )

        if not canonical:
            unexpected.append(str(candidate))

    if unexpected:
        raise ValidationError(
            f"{label}: unexpected parquet file(s) under {root} would be read "
            f"by load_advanced(): {unexpected}. Nothing was deleted; resolve "
            "this by hand before refreshing."
        )


def previous_store_summary(
    path: Path,
    id_column: str,
) -> StoreSummary | None:
    """Baseline counts for the regression guards, or None when unusable."""

    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None

    if frame.empty:
        return None

    return StoreSummary(
        path=path,
        rows=len(frame),
        game_ids=numeric_id_set(frame, id_column),
    )


def assert_no_regression(
    *,
    label: str,
    previous: StoreSummary | None,
    staged: StoreSummary,
    compare_rows: bool = True,
) -> None:
    if previous is None:
        return

    if compare_rows and staged.rows < previous.rows:
        raise ValidationError(
            f"{label}: row-count regression {previous.rows} -> {staged.rows}. "
            "A decrease is not accepted as a correction; investigate by hand."
        )

    if staged.distinct_games < previous.distinct_games:
        raise ValidationError(
            f"{label}: distinct-game regression {previous.distinct_games} -> "
            f"{staged.distinct_games}. A decrease is not accepted as a "
            "correction; investigate by hand."
        )


def assert_final_game_coverage(
    *,
    preflight: PreflightResult,
    stats: StoreSummary,
    advanced: StoreSummary,
) -> str:
    """Reuse the preflight's own structured final-game set when it exposes one."""

    expected = preflight.final_game_ids

    if expected is None:
        return (
            "NOT AVAILABLE WITHOUT PREFLIGHT INTERFACE CHANGE "
            "(no latest_final_game_ids in preflight payload)"
        )

    missing_standard = sorted(expected - stats.game_ids)

    if missing_standard:
        raise ValidationError(
            "staged stats.parquet is missing games the preflight declared "
            f"ready: {missing_standard[:20]}"
        )

    if preflight.advanced_required:
        missing_advanced = sorted(expected - advanced.game_ids)

        if missing_advanced:
            raise ValidationError(
                "staged advanced.parquet is missing games the preflight "
                f"declared ready: {missing_advanced[:20]}"
            )

    return f"ENFORCED for {len(expected)} preflight final game id(s)"


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------


@dataclass
class CommitCandidate:
    name: str
    staged: Path
    live: Path
    staged_sha256: str


def replace_file(source: Path, destination: Path) -> None:
    """Atomically put `source` bytes at `destination` on the same filesystem.

    The staged bytes are copied verbatim into a sibling temporary file and then
    os.replace()d over the destination. No parquet is re-serialized.
    """

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        prefix=".refresh_commit_",
        suffix=".parquet",
        dir=str(destination.parent),
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()

    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def commit_candidates(
    candidates: Sequence[CommitCandidate],
    rollback_dir: Path,
) -> list[str]:
    """Replace each live file, rolling back everything already replaced on error.

    NOT a three-file transaction: each replacement is individually atomic, and
    rollback only covers exceptions this process actually catches.
    """

    rollback_dir = Path(rollback_dir)
    rollback_dir.mkdir(parents=True, exist_ok=True)

    backups: dict[str, Path | None] = {}
    original_sha: dict[str, str | None] = {}

    for candidate in candidates:
        if candidate.live.exists():
            backup = rollback_dir / f"{candidate.name}.previous.parquet"
            shutil.copy2(candidate.live, backup)
            backups[candidate.name] = backup
            original_sha[candidate.name] = sha256_file(candidate.live)
        else:
            backups[candidate.name] = None
            original_sha[candidate.name] = None

    replaced: list[CommitCandidate] = []
    messages: list[str] = []

    def rollback(reason: str) -> None:
        failures: list[str] = []

        for done in reversed(replaced):
            backup = backups[done.name]

            try:
                if backup is None:
                    if done.live.exists():
                        done.live.unlink()
                else:
                    replace_file(backup, done.live)
            except Exception as exc:
                failures.append(f"{done.name}: {exc}")
                continue

            expected = original_sha[done.name]

            if expected is None:
                if done.live.exists():
                    failures.append(
                        f"{done.name}: new destination could not be removed"
                    )
            else:
                if not done.live.exists():
                    failures.append(f"{done.name}: rollback file missing")
                elif sha256_file(done.live) != expected:
                    failures.append(f"{done.name}: rollback hash mismatch")

        if failures:
            raise RollbackError(
                f"{reason}; ROLLBACK INCOMPLETE: {failures}. Live current-"
                "season state must be repaired by hand before the next run."
            )

        raise CommitError(f"{reason}; rollback verified, live state restored")

    for candidate in candidates:
        try:
            replace_file(candidate.staged, candidate.live)
        except Exception as exc:
            rollback(f"commit failed for {candidate.name}: {exc}")

        replaced.append(candidate)

    for candidate in candidates:
        try:
            committed = sha256_file(candidate.live)
        except Exception as exc:
            rollback(f"could not hash committed {candidate.name}: {exc}")

        if committed != candidate.staged_sha256:
            rollback(
                f"post-commit hash mismatch for {candidate.name}: "
                f"staged {candidate.staged_sha256} != live {committed}"
            )

        messages.append(f"{candidate.name}: sha256 {committed}")

    return messages


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def report(lines: Sequence[str]) -> None:
    for line in lines:
        print(line)


def describe_counts(
    label: str,
    previous: StoreSummary | None,
    staged: StoreSummary,
) -> str:
    before_rows = "none" if previous is None else f"{previous.rows:,}"
    before_games = (
        "none" if previous is None else f"{previous.distinct_games:,}"
    )

    return (
        f"  {label:<9} rows {before_rows} -> {staged.rows:,} | "
        f"distinct games {before_games} -> {staged.distinct_games:,}"
    )


def refresh(
    *,
    slate_date: str,
    season: int,
    data_dir: Path,
    opening_day: str | None = None,
    require_advanced: bool = True,
) -> int:
    data_dir = Path(data_dir).resolve()
    slate = exact_date(slate_date, "slate date")

    print("=" * 78)
    print("SAFE CURRENT-SEASON HISTORICAL-STATE REFRESH (STAGE 2)")
    print("=" * 78)
    print(f"slate date : {slate_date}")
    print(f"season     : {season}")
    print(f"data dir   : {data_dir}")

    try:
        with refresh_lock(lock_path_for(data_dir)) as lock:
            print(f"lock       : {lock} (exclusive, non-blocking)")
            return _locked_refresh(
                slate=slate,
                slate_date=slate_date,
                season=season,
                data_dir=data_dir,
                opening_day=opening_day,
                require_advanced=require_advanced,
            )
    except AlreadyLockedError as exc:
        print(f"Status: ALREADY_LOCKED\n{exc}")
        print("No preflight, no fetch, no historical-data modification.")
        return EXIT_ALREADY_LOCKED


def _locked_refresh(
    *,
    slate: pd.Timestamp,
    slate_date: str,
    season: int,
    data_dir: Path,
    opening_day: str | None,
    require_advanced: bool,
) -> int:
    live_stats = stats_path_for(data_dir, season)
    live_games = games_path_for(data_dir, season)
    live_advanced = advanced_path_for(data_dir, season)

    preflight = run_preflight(
        season=season,
        slate_date=slate_date,
        opening_day=opening_day,
        require_advanced=require_advanced,
    )

    print()
    print(f"Preflight action    : {preflight.action}")
    print(f"Preflight exit code : {preflight.returncode}")

    if not preflight.ready:
        print(
            "Status: PREFLIGHT_NOT_READY — no fetch, no staging, no "
            "historical-data modification. Preserving the preflight outcome."
        )
        if preflight.stdout:
            print(preflight.stdout.rstrip())
        if preflight.stderr:
            print(preflight.stderr.rstrip(), file=sys.stderr)

        if preflight.returncode != 0:
            return preflight.returncode

        return EXIT_PREFLIGHT_UNRECOGNIZED

    # An unexpected extra parquet already sitting in the live advanced tree
    # poisons load_advanced() regardless of what we stage, so check it first.
    assert_canonical_advanced_layout(
        advanced_root_for(data_dir),
        "live advanced tree",
    )

    previous_stats = previous_store_summary(live_stats, "game_id")
    previous_advanced = previous_store_summary(live_advanced, "game_id")
    previous_games = previous_store_summary(live_games, GAMES_ID_COLUMN)

    shadow_root = Path(
        tempfile.mkdtemp(
            prefix=".nba_prop_refresh_shadow_",
            dir=str(data_dir.parent),
        )
    )

    try:
        shadow_data_dir = shadow_root / "data"
        shadow_data_dir.mkdir(parents=True, exist_ok=True)

        print(f"Shadow data root    : {shadow_data_dir}")
        print("Live files are not passed to the ingester during the fetch.")

        run_shadow_ingest(shadow_data_dir=shadow_data_dir, season=season)

        staged_stats_path = stats_path_for(shadow_data_dir, season)
        staged_games_path = games_path_for(shadow_data_dir, season)
        staged_advanced_path = advanced_path_for(shadow_data_dir, season)

        assert_canonical_advanced_layout(
            advanced_root_for(shadow_data_dir),
            "staged advanced tree",
        )

        stats = validate_stats(staged_stats_path, slate)
        advanced = validate_advanced(staged_advanced_path, slate)
        games = validate_games(staged_games_path)

        assert_referenced_games_present(
            stats=stats,
            advanced=advanced,
            games=games,
        )

        assert_no_regression(
            label="stats",
            previous=previous_stats,
            staged=stats,
        )
        assert_no_regression(
            label="advanced",
            previous=previous_advanced,
            staged=advanced,
        )
        assert_no_regression(
            label="games",
            previous=previous_games,
            staged=games,
            compare_rows=False,
        )

        coverage = assert_final_game_coverage(
            preflight=preflight,
            stats=stats,
            advanced=advanced,
        )

        print()
        print("Validated staged state (live -> staged):")
        report(
            [
                describe_counts("stats", previous_stats, stats),
                describe_counts("advanced", previous_advanced, advanced),
                describe_counts("games", previous_games, games),
            ]
        )
        print(
            f"  stats max date {stats.max_date.date()} < slate {slate.date()}"
        )
        print(
            f"  advanced max date {advanced.max_date.date()} < slate "
            f"{slate.date()}"
        )
        print(f"  final-game coverage: {coverage}")

        candidates = [
            CommitCandidate(
                name="stats",
                staged=staged_stats_path,
                live=live_stats,
                staged_sha256=stats.sha256,
            ),
            CommitCandidate(
                name="games",
                staged=staged_games_path,
                live=live_games,
                staged_sha256=games.sha256,
            ),
            CommitCandidate(
                name="advanced",
                staged=staged_advanced_path,
                live=live_advanced,
                staged_sha256=advanced.sha256,
            ),
        ]

        committed = commit_candidates(candidates, shadow_root / ".rollback")

        print()
        print("Committed live current-season files (staged sha == live sha):")
        report([f"  {line}" for line in committed])
        print(
            "  NOTE: three separate atomic replacements are NOT one atomic "
            "three-file transaction."
        )
        print()
        print("Status: REFRESH_COMMITTED")

        return EXIT_OK

    finally:
        shutil.rmtree(shadow_root, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely refresh the current-season standard and advanced "
            "historical state. Fetches into a shadow data root through the "
            "existing ingester, validates it, and only then atomically "
            "replaces the live current-season files."
        )
    )

    parser.add_argument(
        "--slate-date",
        required=True,
        help="Slate date YYYY-MM-DD. Staged history must end strictly before it.",
    )

    parser.add_argument("--season", type=int, default=2026)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Live data root. Defaults to NBA_PROP_DATA_DIR, else <repo>/data. "
            "API credentials are never accepted on the command line."
        ),
    )

    parser.add_argument(
        "--opening-day",
        default=None,
        help="Optional opening-day override forwarded to the preflight.",
    )

    parser.add_argument(
        "--no-require-advanced",
        dest="require_advanced",
        action="store_false",
        help=(
            "Do not require advanced readiness in the preflight. Advanced "
            "staged data is validated either way."
        ),
    )
    parser.set_defaults(require_advanced=True)

    return parser.parse_args(argv)


def default_data_dir() -> Path:
    configured = os.environ.get("NBA_PROP_DATA_DIR")

    if configured:
        return Path(configured)

    return PROJECT_ROOT / "data"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    data_dir = args.data_dir or default_data_dir()

    try:
        return refresh(
            slate_date=args.slate_date,
            season=args.season,
            data_dir=data_dir,
            opening_day=args.opening_day,
            require_advanced=args.require_advanced,
        )
    except ValidationError as exc:
        print(f"Status: VALIDATION_FAILED\n{exc}", file=sys.stderr)
        print("Live current-season files were not modified.", file=sys.stderr)
        return EXIT_VALIDATION_FAILED
    except RollbackError as exc:
        print(f"Status: ROLLBACK_FAILED\n{exc}", file=sys.stderr)
        return EXIT_ROLLBACK_FAILED
    except CommitError as exc:
        print(f"Status: COMMIT_FAILED\n{exc}", file=sys.stderr)
        return EXIT_COMMIT_FAILED
    except RefreshError as exc:
        print(f"Status: REFRESH_FAILED\n{exc}", file=sys.stderr)
        return EXIT_INGEST_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
