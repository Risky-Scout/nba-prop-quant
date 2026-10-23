"""Tests for the durable T-20 evidence ledger.

The property worth testing is survival: a record written before a segment
dies has to still be there afterwards, in the data root rather than in a
runner's temporary directory, and readable by a process that shares nothing
with the one that wrote it. So these tests write with one ledger instance and
read with another, and they assert where the bytes landed rather than what an
in-memory list happens to hold.

Nothing here captures, serves, publishes or reaches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load_ops(name: str):
    spec = importlib.util.spec_from_file_location(
        f"ops_{name}", PROJECT / "ops" / f"{name}.py"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


ledger_module = load_ops("t20_evidence_ledger")

SLATE = "2026-10-23"

SHA = "c7246b8b9f2172875a0c70874ea5bfa81ecb4d8e"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 23, 22, 40, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now = self.now + timedelta(seconds=1)

        return self.now


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def make(data_root: Path, *, run_id: str = "40016015624"):
    return ledger_module.EvidenceLedger(
        data_root=data_root,
        slate_date=SLATE,
        production_sha=SHA,
        segment_start_utc="2026-10-23T19:25:00+00:00",
        segment_end_utc="2026-10-24T00:25:00+00:00",
        run_id=run_id,
        clock=Clock(),
    )


# ----------------------------------------------------------------------
# where it writes
# ----------------------------------------------------------------------


def test_the_ledger_lives_in_the_data_root(data_root):
    book = make(data_root)

    assert book.ledger_path == (
        data_root / "processed" / "t20_evidence" / f"{SLATE}.jsonl"
    )

    assert book.slate_dir == data_root / "processed" / "t20_evidence" / SLATE


def test_constructing_a_ledger_writes_nothing(data_root):
    make(data_root)

    # The segment probe builds a controller three times a day and must not
    # leave a directory behind for a slate it never acts on.
    assert not data_root.exists()


# ----------------------------------------------------------------------
# survival
# ----------------------------------------------------------------------


def test_a_record_is_readable_by_a_ledger_that_did_not_write_it(data_root):
    make(data_root).record("SEGMENT_OPENED", {"offset_minutes": 20})

    read_back = make(data_root).read()

    assert [entry["event"] for entry in read_back] == ["SEGMENT_OPENED"]

    assert read_back[0]["detail"] == {"offset_minutes": 20}


def test_records_accumulate_in_the_order_they_were_written(data_root):
    book = make(data_root)

    for event in ("SEGMENT_OPENED", "WINDOW_CAPTURED", "SEGMENT_CLOSED"):
        book.record(event, {})

    assert [entry["event"] for entry in make(data_root).read()] == [
        "SEGMENT_OPENED",
        "WINDOW_CAPTURED",
        "SEGMENT_CLOSED",
    ]


def test_a_second_segment_appends_rather_than_replacing_the_first(data_root):
    make(data_root, run_id="first").record("SEGMENT_CLOSED", {})

    make(data_root, run_id="second").record("SEGMENT_OPENED", {})

    assert [entry["run_id"] for entry in make(data_root).read()] == [
        "first",
        "second",
    ]


def test_every_record_names_the_slate_the_sha_and_the_segment(data_root):
    entry = make(data_root).record("SEGMENT_OPENED", {})

    assert entry["production_sha"] == SHA

    assert entry["slate_date"] == SLATE

    assert entry["segment_start_utc"] == "2026-10-23T19:25:00+00:00"

    assert entry["segment_end_utc"] == "2026-10-24T00:25:00+00:00"

    assert entry["recorded_at"].startswith("2026-10-23T22:40:")


def test_a_torn_line_does_not_hide_the_records_around_it(data_root):
    book = make(data_root)

    book.record("SEGMENT_OPENED", {})

    with book.ledger_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "WINDOW_CAPT')

        handle.write("\n")

    book.record("SEGMENT_CLOSED", {})

    assert [entry["event"] for entry in make(data_root).read()] == [
        "SEGMENT_OPENED",
        "UNREADABLE_RECORD",
        "SEGMENT_CLOSED",
    ]


def test_reading_a_slate_nobody_recorded_is_empty_rather_than_an_error(
    data_root,
):
    assert make(data_root).read() == []


# ----------------------------------------------------------------------
# preserved predictions
# ----------------------------------------------------------------------


def test_an_artifact_is_copied_under_its_own_digest(data_root, tmp_path):
    source = tmp_path / "priced_markets" / f"{SLATE}.parquet"

    source.parent.mkdir(parents=True)

    source.write_bytes(b"priced markets for the first tip group")

    record = make(data_root).preserve_artifact("group-01-priced", source)

    assert record["preserved"] is True

    assert record["already_preserved"] is False

    assert record["bytes"] == len(b"priced markets for the first tip group")

    kept = Path(record["path"])

    assert kept.name == f"group-01-priced__{record['sha256'][:16]}.parquet"

    assert kept.read_bytes() == b"priced markets for the first tip group"


def test_the_preserved_copy_outlives_the_source_being_overwritten(
    data_root, tmp_path
):
    source = tmp_path / f"{SLATE}.parquet"

    source.write_bytes(b"first group")

    first = make(data_root).preserve_artifact("group-01-priced", source)

    # What serving does to the earlier group's predictions when the next
    # group's tips come due.
    source.write_bytes(b"second group")

    second = make(data_root).preserve_artifact("group-02-priced", source)

    assert Path(first["path"]).read_bytes() == b"first group"

    assert Path(second["path"]).read_bytes() == b"second group"


def test_preserving_identical_bytes_twice_stores_one_copy(data_root, tmp_path):
    source = tmp_path / f"{SLATE}.parquet"

    source.write_bytes(b"unchanged")

    book = make(data_root)

    first = book.preserve_artifact("group-01-priced", source)

    second = book.preserve_artifact("group-01-priced", source)

    assert second["already_preserved"] is True

    assert first["path"] == second["path"]

    assert len(list(book.artifact_dir.iterdir())) == 1


def test_an_absent_artifact_is_reported_rather_than_raised(data_root, tmp_path):
    record = make(data_root).preserve_artifact(
        "group-01-priced", tmp_path / "never-written.parquet"
    )

    assert record["preserved"] is False

    assert "is not a file" in record["reason"]


def test_a_preserved_receipt_keeps_the_whole_document(data_root):
    book = make(data_root)

    document = {
        "input_state_fingerprint": "b5472a3d",
        "outcome": "SERVED",
        "priced_market_artifact": {"bytes": 275956, "sha256": "997391b4"},
    }

    path = book.preserve_receipt("incumbent_serving", document)

    assert json.loads(path.read_text(encoding="utf-8")) == document

    assert path.parent == book.receipt_dir

    assert path.name.endswith("__40016015624__incumbent_serving.json")


def test_two_receipts_of_the_same_label_do_not_overwrite_each_other(data_root):
    book = make(data_root)

    first = book.preserve_receipt("incumbent_serving", {"group": 1})

    second = book.preserve_receipt("incumbent_serving", {"group": 2})

    assert first != second

    assert json.loads(first.read_text(encoding="utf-8")) == {"group": 1}

    assert json.loads(second.read_text(encoding="utf-8")) == {"group": 2}


def test_no_temporary_file_is_left_beside_a_preserved_artifact(
    data_root, tmp_path
):
    source = tmp_path / f"{SLATE}.parquet"

    source.write_bytes(b"predictions")

    book = make(data_root)

    book.preserve_artifact("group-01-priced", source)

    book.preserve_receipt("incumbent_serving", {"outcome": "SERVED"})

    leftovers = [
        path
        for path in book.slate_dir.rglob("*")
        if path.name.endswith(".tmp")
    ]

    assert leftovers == []
