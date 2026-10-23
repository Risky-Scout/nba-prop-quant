"""Tests for the read-only slate evidence inventory.

The inventory's one job is to be right about which of three states a slate is
in, because the three states license three different actions and the most
important of them is "generate nothing". So these tests build real capture
trees with the collector's own fixture writer and assert the classification,
including the case that matters most: a capture taken after tip-off is not a
pregame source.

Nothing here serves, publishes, trains or reaches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from t20_capture_fixture import (  # noqa: E402
    SLATE,
    at,
    game,
    prop,
    write_capture,
)


def load_ops(name: str):
    spec = importlib.util.spec_from_file_location(
        f"ops_{name}", PROJECT / "ops" / f"{name}.py"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


inventory_module = load_ops("inventory_slate_evidence")


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def capture(
    data_root: Path, *, captured_at, games, window_ids, due_game_ids
) -> None:
    write_capture(
        data_root / "snapshots",
        date=SLATE,
        captured_at=captured_at.isoformat(),
        window_ids=window_ids,
        due_game_ids=due_game_ids,
        games=games,
        props=[prop(int(row["id"]), 600 + int(row["id"])) for row in games],
    )


def take(data_root: Path, slate_date: str = SLATE) -> dict:
    return inventory_module.inventory(
        data_root=data_root, slate_date=slate_date
    )


def write_parquet(data_root: Path, folder: str) -> None:
    path = data_root / "processed" / folder / f"{SLATE}.parquet"

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_bytes(b"artifact")


def write_release(data_root: Path, release_id: str) -> None:
    path = data_root / "publication" / "canary" / "releases" / release_id

    path.mkdir(parents=True, exist_ok=True)

    (path / "rows.json").write_text("[]", encoding="utf-8")


# ----------------------------------------------------------------------
# nothing to recover from
# ----------------------------------------------------------------------


def test_a_slate_nothing_was_captured_for_licenses_no_reconstruction(
    data_root,
):
    report = take(data_root)

    assert report["classification"] == inventory_module.CLASSIFICATION_NONE

    assert "nothing may be generated" in report["reason"]

    assert report["t20_captures"]["capture_count"] == 0

    assert report["projections"]["exists"] is False


def test_a_capture_taken_after_tip_off_is_not_a_pregame_source(data_root):
    games = [game(101, at(23, 0))]

    # Half an hour into the game. This is the shape a well-meaning backfill
    # would produce, and it must not be mistaken for a frozen pregame input.
    capture(
        data_root,
        captured_at=at(23, 30),
        games=games,
        window_ids=["101:T-20m"],
        due_game_ids=[101],
    )

    report = take(data_root)

    assert report["classification"] == inventory_module.CLASSIFICATION_NONE

    assert report["t20_captures"]["capture_count"] == 1

    assert report["t20_captures"]["pregame_game_ids"] == []

    assert report["t20_captures"]["captures"][0]["post_tip_game_ids"] == [101]


# ----------------------------------------------------------------------
# a frozen pregame input and no prediction made from it
# ----------------------------------------------------------------------


def test_a_pretip_t20_capture_with_no_prediction_licenses_reconstruction(
    data_root,
):
    games = [game(101, at(23, 0)), game(102, at(23, 0), home=3, away=4)]

    capture(
        data_root,
        captured_at=at(22, 40),
        games=games,
        window_ids=["101:T-20m", "102:T-20m"],
        due_game_ids=[101, 102],
    )

    report = take(data_root)

    assert (
        report["classification"]
        == inventory_module.CLASSIFICATION_SOURCE_ONLY
    )

    assert report["t20_captures"]["pregame_game_ids"] == [101, 102]

    assert report["t20_captures"]["production_t20_windows_captured"] == [
        "101:T-20m",
        "102:T-20m",
    ]

    assert "strict as-of reconstruction" in report["reason"]


def test_predictions_without_a_publication_are_still_only_a_source(data_root):
    """2026-10-20's shape: served, then the rsync transport refused."""
    games = [game(101, at(23, 0))]

    capture(
        data_root,
        captured_at=at(22, 40),
        games=games,
        window_ids=["101:T-20m"],
        due_game_ids=[101],
    )

    write_parquet(data_root, "projections")

    write_parquet(data_root, "priced_markets")

    report = take(data_root)

    assert (
        report["classification"]
        == inventory_module.CLASSIFICATION_SOURCE_ONLY
    )


# ----------------------------------------------------------------------
# already complete
# ----------------------------------------------------------------------


def test_a_published_slate_needs_no_recovery(data_root):
    games = [game(101, at(23, 0))]

    capture(
        data_root,
        captured_at=at(22, 40),
        games=games,
        window_ids=["101:T-20m"],
        due_game_ids=[101],
    )

    write_parquet(data_root, "projections")

    write_parquet(data_root, "priced_markets")

    write_release(data_root, f"nba_prop_quant__{SLATE}__dc684fd869a8e9f5")

    report = take(data_root)

    assert (
        report["classification"] == inventory_module.CLASSIFICATION_AUTHENTIC
    )

    assert [record["release_id"] for record in report["releases"]] == [
        f"nba_prop_quant__{SLATE}__dc684fd869a8e9f5"
    ]


# ----------------------------------------------------------------------
# what else it reports
# ----------------------------------------------------------------------


def test_tip_times_come_from_the_capture_and_not_from_todays_schedule(
    data_root,
):
    capture(
        data_root,
        captured_at=at(22, 40),
        games=[game(101, at(23, 0))],
        window_ids=["101:T-20m"],
        due_game_ids=[101],
    )

    report = take(data_root)

    assert report["t20_captures"]["tip_times_recorded_by_captures"] == {
        "101": at(23, 0).isoformat()
    }


def test_the_serving_receipt_is_summarised_rather_than_inlined(data_root):
    path = data_root / "processed" / "incumbent_serving" / f"{SLATE}.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            {
                "outcome": "NO_GAMES_ON_SLATE",
                "production_code_sha": "6336611",
                "serving_bundle_root": "/a/very/long/path",
                "slate_readiness": {"ready": True},
            }
        ),
        encoding="utf-8",
    )

    report = take(data_root)

    assert report["serving_receipt"]["outcome"] == "NO_GAMES_ON_SLATE"

    assert report["serving_receipt"]["production_code_sha"] == "6336611"

    assert "serving_bundle_root" not in report["serving_receipt"]


def test_an_evidence_ledger_is_reported_as_its_event_sequence(data_root):
    path = data_root / "processed" / "t20_evidence" / f"{SLATE}.jsonl"

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        "\n".join(
            json.dumps(entry)
            for entry in (
                {"event": "SEGMENT_OPENED", "detail": {}},
                {
                    "event": "SEGMENT_REFUSED",
                    "detail": {"outcome": "MORNING_PRODUCTION_NOT_READY"},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = take(data_root)

    assert report["evidence_ledger"]["events"] == [
        "SEGMENT_OPENED",
        "SEGMENT_REFUSED",
    ]

    assert report["evidence_ledger"]["outcomes"] == [
        "MORNING_PRODUCTION_NOT_READY"
    ]


def test_the_inventory_writes_nothing_into_the_data_root(data_root):
    capture(
        data_root,
        captured_at=at(22, 40),
        games=[game(101, at(23, 0))],
        window_ids=["101:T-20m"],
        due_game_ids=[101],
    )

    before = sorted(str(path) for path in data_root.rglob("*"))

    take(data_root)

    assert sorted(str(path) for path in data_root.rglob("*")) == before


def test_the_inventory_imports_no_model_code():
    source = (PROJECT / "ops" / "inventory_slate_evidence.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "import pandas",
        "import numpy",
        "import xgboost",
        "import sklearn",
        "prospective_snapshot",
    ):
        assert forbidden not in source, forbidden
