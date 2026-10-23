#!/usr/bin/env python3
"""The append-only record of what a T-20 segment saw, served and published.

Three evenings of this season have no evidence at all, and in every case the
reason was the same shape: the controller's only durable output was the
artifact it wrote at the *end*, so anything that stopped it earlier left
nothing behind.

    2026-10-21
        The morning lifecycle failed, so no serving receipt existed and the
        evening segment refused with MORNING_PRODUCTION_NOT_READY. Because
        ``morning_readiness`` is consulted before any capture, the segment
        exited having recorded nothing -- not even that it had tried.

    2026-10-22
        Readiness was sound and the controller slept correctly for three
        hours. The job was cancelled at 22:29:23Z, eleven minutes before the
        first T-20 window. A cancelled job runs no later step, so the
        diagnostics upload never happened; the run's own artifact upload then
        failed with "403 Forbidden: job is completed".

    2026-10-23
        Accepted production code merged after the morning receipt was
        written, so the receipt named an older SHA and the segment refused.
        Twelve games, nothing captured, nothing recorded.

The readiness refresh fixes the third cause and nothing can fix the second,
because no process can promise to be alive. What can be fixed is the evidence:
a segment that writes what it knows *as it learns it*, into the durable data
root rather than into a runner's temporary directory, leaves a record even
when it is killed between two sleeps.

WHY THE DATA ROOT AND NOT AN ARTIFACT
-------------------------------------

Workflow artifacts are the wrong store for this twice over. They expire --
the T-20 workflow asks for fourteen days -- and they are uploaded by a step,
which means they exist only if the job reaches that step. The snapshot tree,
the parquet artifacts and the serving receipts already live on the production
machine and already survive the job. The ledger joins them.

WHY APPEND-ONLY JSONL
---------------------

Each record is one line, written and fsynced on its own. There is no document
to re-serialise, so a later record cannot corrupt an earlier one and a process
killed mid-write loses at most the line it was writing. This is the same
shape, and for the same reason, as the capture-run log the collector appends
to.

WHY THE PREDICTIONS ARE COPIED
------------------------------

``10_predict_slate.py`` and ``15_price_markets.py`` write one file per slate
date: ``processed/projections/<slate>.parquet`` and
``processed/priced_markets/<slate>.parquet``. The controller serves the slate
once per *capture group*, so on a slate with tips at 19:00, 19:30 and 22:00
the second group's serving overwrites the first group's predictions and the
third overwrites the second. The published releases are immutable, but the
model's own output for the earlier groups was not recoverable from the data
root at all.

So each served group's two artifacts are copied here under their own content
digest before the next group can overwrite them. They are small -- the
2026-10-23 pair is 170 KB and 276 KB -- and the copy is keyed by digest, so
re-serving identical bytes costs nothing and stores nothing new.

WHAT IT DOES NOT DO
-------------------

It does not decide anything. Nothing reads the ledger to gate serving, no
outcome depends on what it contains, and it never refuses: it is a witness,
not a check. Its failures are failures of the data root, which is the same
filesystem serving is about to write parquet to, so they are raised rather
than swallowed -- a ledger that silently stopped recording would recreate the
exact problem it exists to end.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

LEDGER_RELATIVE = pathlib.Path("processed") / "t20_evidence"

EVENT_SEGMENT_OPENED = "SEGMENT_OPENED"
EVENT_SEGMENT_PLANNED = "SEGMENT_PLANNED"
EVENT_SEGMENT_CLOSED = "SEGMENT_CLOSED"
EVENT_SEGMENT_REFUSED = "SEGMENT_REFUSED"
EVENT_READINESS_PROVEN = "READINESS_PROVEN"
EVENT_WINDOW_CAPTURED = "WINDOW_CAPTURED"
EVENT_SLATE_SERVED = "SLATE_SERVED"
EVENT_DISTRIBUTIONS_DERIVED = "DISTRIBUTIONS_DERIVED"
EVENT_FEED_PUBLISHED = "FEED_PUBLISHED"
EVENT_PAGE_PUBLISHED = "PAGE_PUBLISHED"
EVENT_PUBLICATION_SKIPPED = "PUBLICATION_SKIPPED"

# Read in 1 MiB blocks rather than whole-file, because the artifacts copied
# here are parquet and nothing guarantees they stay in the hundreds of KB.
DIGEST_BLOCK_BYTES = 1024 * 1024


def digest_of(path: pathlib.Path) -> tuple[str, int]:
    """The SHA-256 and byte count of a file on disk."""
    sha = hashlib.sha256()

    size = 0

    with path.open("rb") as handle:
        while True:
            block = handle.read(DIGEST_BLOCK_BYTES)

            if not block:
                break

            size += len(block)

            sha.update(block)

    return sha.hexdigest(), size


class EvidenceLedger:
    """One slate's append-only evidence tree, under the durable data root."""

    def __init__(
        self,
        *,
        data_root: pathlib.Path,
        slate_date: str,
        production_sha: str,
        segment_start_utc: str,
        segment_end_utc: str,
        run_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.data_root = pathlib.Path(data_root)
        self.slate_date = str(slate_date)
        self.production_sha = str(production_sha)
        self.segment_start_utc = str(segment_start_utc)
        self.segment_end_utc = str(segment_end_utc)

        # Identifies the invocation inside a slate that several segments and
        # several reruns all append to. The workflow run id is the natural
        # one and is present on every production invocation; outside Actions
        # the start of the segment is unambiguous enough.
        self.run_id = str(
            run_id
            if run_id is not None
            else os.environ.get("GITHUB_RUN_ID") or self.segment_start_utc
        )

        self.clock = clock or (lambda: datetime.now(UTC))

        self.recorded: list[dict[str, Any]] = []

    # -- locations -----------------------------------------------------

    @property
    def slate_dir(self) -> pathlib.Path:
        return self.data_root / LEDGER_RELATIVE / self.slate_date

    @property
    def ledger_path(self) -> pathlib.Path:
        return self.data_root / LEDGER_RELATIVE / f"{self.slate_date}.jsonl"

    @property
    def artifact_dir(self) -> pathlib.Path:
        return self.slate_dir / "artifacts"

    @property
    def receipt_dir(self) -> pathlib.Path:
        return self.slate_dir / "receipts"

    # -- writing -------------------------------------------------------

    def record(self, event: str, detail: dict[str, Any]) -> dict[str, Any]:
        """Append one event. Returns the record, for the caller's receipt."""
        entry = {
            "detail": detail,
            "event": str(event),
            "production_sha": self.production_sha,
            "recorded_at": self.clock().isoformat(),
            "run_id": self.run_id,
            "segment_end_utc": self.segment_end_utc,
            "segment_start_utc": self.segment_start_utc,
            "slate_date": self.slate_date,
        }

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"

        # Opened, written, flushed and fsynced per record. The cost is one
        # sync per event -- a dozen or so an evening -- and it buys the
        # property the whole module exists for: the record survives whatever
        # happens to this process next.
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

            handle.flush()

            os.fsync(handle.fileno())

        self.recorded.append(entry)

        return entry

    def preserve_receipt(
        self, label: str, document: dict[str, Any]
    ) -> pathlib.Path:
        """Keep a full receipt, not just the ledger's summary of it."""
        self.receipt_dir.mkdir(parents=True, exist_ok=True)

        stamp = self.clock().strftime("%Y%m%dT%H%M%SZ")

        path = self.receipt_dir / f"{stamp}__{self.run_id}__{label}.json"

        self._write_atomically(
            path, json.dumps(document, indent=2, sort_keys=True) + "\n"
        )

        return path

    def preserve_artifact(
        self, label: str, source: pathlib.Path
    ) -> dict[str, Any]:
        """Copy one prediction artifact under its own content digest.

        Keyed by digest, so the copy is idempotent: a group that re-serves
        identical bytes resolves to the file already here and no second copy
        is made.
        """
        source = pathlib.Path(source)

        if not source.is_file():
            return {
                "label": str(label),
                "preserved": False,
                "reason": f"{source} is not a file, so there was none to keep",
                "source": str(source),
            }

        sha256, size = digest_of(source)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        path = self.artifact_dir / f"{label}__{sha256[:16]}{source.suffix}"

        already = path.is_file()

        if not already:
            temporary = path.with_name(path.name + ".tmp")

            shutil.copyfile(source, temporary)

            os.replace(temporary, path)

        return {
            "bytes": size,
            "already_preserved": already,
            "label": str(label),
            "path": str(path),
            "preserved": True,
            "sha256": sha256,
            "source": str(source),
        }

    # -- reading -------------------------------------------------------

    def read(self) -> list[dict[str, Any]]:
        """Every event recorded for this slate, by anyone, in order."""
        if not self.ledger_path.is_file():
            return []

        entries: list[dict[str, Any]] = []

        with self.ledger_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A line torn by a kill is the one thing this format can
                    # lose, and losing it must not hide the lines around it.
                    entries.append({"event": "UNREADABLE_RECORD", "raw": line})

                    continue

                entries.append(entry)

        return entries

    # -- internals -----------------------------------------------------

    @staticmethod
    def _write_atomically(path: pathlib.Path, text: str) -> None:
        temporary = path.with_name(path.name + ".tmp")

        temporary.write_text(text, encoding="utf-8")

        os.replace(temporary, path)
