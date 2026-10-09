"""Automatic grading of the incumbent's served slate.

Every case here builds a complete fake data root -- priced markets, serving
receipt, box scores and game statuses -- so the grader is exercised through
its real entry point against real parquet files rather than through mocks of
the things most likely to be wrong.

The cases the brief names are each a separate test: a correctly settled
grade, a game that has not finished, a slate that is only partly settled, a
rerun that must not duplicate anything, corrupted provenance, a prediction
and a settlement that do not describe the same slate, and the two directions
of lookahead.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
ENTRY_POINT = PROJECT / "ops" / "grade_incumbent_production_slate.py"

SLATE = "2026-11-04"
SEASON = 2026


def load_grader():
    spec = importlib.util.spec_from_file_location(
        "grade_incumbent_production_slate", ENTRY_POINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def grader():
    return load_grader()


@pytest.fixture(scope="module")
def rules(grader):
    return grader.load_settlement_rules()


# ======================================================================
# a fake served slate
# ======================================================================


def priced_frame(records: list[dict[str, Any]], *, priced_at: str) -> pd.DataFrame:
    """Priced markets in the shape ``scripts/15_price_markets.py`` writes."""
    rows = []
    for record in records:
        over = float(record["q_over"])
        rows.append(
            {
                "game_id": int(record["game_id"]),
                "player_id": int(record["player_id"]),
                "prop_type": record.get("prop_type", "points"),
                "line_value": float(record["line"]),
                "vendor": record.get("vendor", "wizardofodds"),
                "calibrated_q_over_nonpush": over,
                "calibrated_q_under_nonpush": 1.0 - over,
                "model_preferred_side": record.get("side", "over"),
                "priced_at_utc": priced_at,
                "market_pricing_schema_version": 3,
            }
        )
    return pd.DataFrame.from_records(rows)


def stats_frame(records: list[dict[str, Any]], *, date: str) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "game_id": int(record["game_id"]),
                "player_id": int(record["player_id"]),
                "date": pd.Timestamp(record.get("date", date)),
                "season": SEASON,
                "minutes": float(record.get("minutes", 32.5)),
                "pts": float(record.get("pts", 0.0)),
                "reb": float(record.get("reb", 0.0)),
                "ast": float(record.get("ast", 0.0)),
                "stl": float(record.get("stl", 0.0)),
                "blk": float(record.get("blk", 0.0)),
                "fg3m": float(record.get("fg3m", 0.0)),
            }
        )
    return pd.DataFrame.from_records(rows)


def games_frame(records: list[dict[str, Any]], *, date: str) -> pd.DataFrame:
    rows = []
    for record in records:
        final = record.get("final", True)
        rows.append(
            {
                "id": int(record["game_id"]),
                "date": pd.Timestamp(record.get("date", date)),
                "season": SEASON,
                "status": "Final" if final else "7:30 PM ET",
                "status_state": "post" if final else "pre",
                "postponed": False,
            }
        )
    return pd.DataFrame.from_records(rows)


def build_root(
    tmp_path: Path,
    *,
    priced: pd.DataFrame,
    stats: pd.DataFrame | None,
    games: pd.DataFrame | None,
    slate_date: str = SLATE,
    receipt_overrides: dict[str, Any] | None = None,
    receipt_slate: str | None = None,
) -> dict[str, Path]:
    """A durable data root holding exactly what grading reads."""
    data_root = tmp_path / "data"

    priced_path = data_root / "processed" / "priced_markets" / f"{slate_date}.parquet"
    priced_path.parent.mkdir(parents=True, exist_ok=True)
    priced.to_parquet(priced_path, index=False)

    if stats is not None and games is not None:
        season_dir = data_root / "raw" / "seasons" / f"season={SEASON}"
        season_dir.mkdir(parents=True, exist_ok=True)
        stats.to_parquet(season_dir / "stats.parquet", index=False)
        games.to_parquet(season_dir / "games.parquet", index=False)

    grader = load_grader()

    receipt = {
        "generated_at": "2026-11-04T22:05:00+00:00",
        "incumbent_authority": "frozen_deployment_bundle",
        "incumbent_fit_id": None,
        "incumbent_version": "freeze-2026-10-01",
        "input_state_fingerprint": "d4f1" * 16,
        "model_authority": "incumbent",
        "outcome": "SERVED",
        "prediction_artifact": {"path": "projections", "sha256": "ab" * 32},
        "priced_market_artifact": {
            "path": str(priced_path),
            "sha256": grader.sha256_file(priced_path),
        },
        "production_code_sha": "a81360bcce760b0e5df39e0d56c564d15b474e40",
        "slate_date": receipt_slate or slate_date,
    }
    receipt.update(receipt_overrides or {})

    # Written where the lifecycle durably keeps it, which is how a slate
    # graded on a later day still finds its own provenance.
    receipt_path = (
        data_root
        / grader.SERVING_RECEIPTS_RELATIVE
        / f"{receipt_slate or slate_date}.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    return {
        "data_root": data_root,
        "priced_path": priced_path,
        "serving_receipt": receipt_path,
        "grades_root": tmp_path / "grades",
        "receipt_path": tmp_path / "grading.json",
    }


def invoke(grader, paths: dict[str, Path], *extra: str, slate_date: str = SLATE) -> int:
    return grader.main(
        [
            "--slate-date",
            slate_date,
            "--data-root",
            str(paths["data_root"]),
            "--serving-receipt",
            str(paths["serving_receipt"]),
            "--grades-root",
            str(paths["grades_root"]),
            "--receipt-path",
            str(paths["receipt_path"]),
            "--catch-up-days",
            "0",
            *extra,
        ]
    )


def stored_grades(paths: dict[str, Path]) -> pd.DataFrame:
    files = sorted(paths["grades_root"].rglob("*.parquet"))
    assert files, "grading wrote no grade rows"
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


# ======================================================================
# a correctly settled grade
# ======================================================================


@pytest.fixture
def settled(tmp_path, grader):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62},
                {"game_id": 1, "player_id": 12, "line": 18.5, "q_over": 0.40},
                {
                    "game_id": 2,
                    "player_id": 21,
                    "line": 8.5,
                    "q_over": 0.55,
                    "prop_type": "assists",
                    "side": "under",
                },
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame(
            [
                {"game_id": 1, "player_id": 11, "pts": 31.0},
                {"game_id": 1, "player_id": 12, "pts": 12.0},
                {"game_id": 2, "player_id": 21, "ast": 4.0},
            ],
            date=SLATE,
        ),
        games=games_frame([{"game_id": 1}, {"game_id": 2}], date=SLATE),
    )
    return paths


def test_a_settled_slate_grades_every_prediction(grader, settled):
    assert invoke(grader, settled) == 0

    grades = stored_grades(settled)
    assert len(grades) == 3
    assert set(grades["settlement_status"]) == {grader.STATUS_GRADED}

    by_player = grades.set_index("player_id")

    # 31 points against a 24.5 line, priced over at 0.62: the over hit.
    over_hit = by_player.loc[11]
    assert over_hit["side"] == "over"
    assert over_hit["realized_value"] == 24.5 + 6.5
    assert over_hit["realized_result"] == 1.0
    assert over_hit["model_probability"] == pytest.approx(0.62)
    assert over_hit["brier_contribution"] == pytest.approx((0.62 - 1.0) ** 2)

    # 12 points against 18.5, priced over at 0.40: the over missed.
    over_missed = by_player.loc[12]
    assert over_missed["realized_result"] == 0.0
    assert over_missed["brier_contribution"] == pytest.approx(0.40**2)

    # 4 assists against 8.5 priced *under*: the under hit, and the graded
    # probability is the under column rather than one minus something.
    under_hit = by_player.loc[21]
    assert under_hit["side"] == "under"
    assert under_hit["probability_column"] == "calibrated_q_under_nonpush"
    assert under_hit["model_probability"] == pytest.approx(0.45)
    assert under_hit["realized_result"] == 1.0


def test_every_field_the_policy_requires_is_recorded(grader, settled):
    assert invoke(grader, settled) == 0
    grades = stored_grades(settled)

    for column in (
        "prediction_id",
        "game_id",
        "player_id",
        "stat",
        "side",
        "line",
        "model_probability",
        "realized_result",
        "brier_contribution",
        "log_loss_contribution",
        "incumbent_fit_id",
        "production_code_sha",
        "input_state_fingerprint",
        "prediction_timestamp",
        "settlement_timestamp",
    ):
        assert column in grades.columns, column

    assert grades["production_code_sha"].nunique() == 1
    assert grades["input_state_fingerprint"].nunique() == 1
    assert grades["prediction_artifact_sha256"].nunique() == 1
    # Null is the true value before the first human promotion, so the column
    # must be present and may be empty.
    assert "incumbent_fit_id" in grades.columns
    assert grades["prediction_timestamp"].notna().all()
    assert grades["settlement_timestamp"].notna().all()


def test_the_log_loss_contribution_matches_its_declared_floor(grader, settled):
    assert invoke(grader, settled) == 0
    grades = stored_grades(settled)

    for _, row in grades.iterrows():
        assert row["log_loss_contribution"] == pytest.approx(
            grader.log_loss_contribution(
                row["model_probability"], row["realized_result"]
            )
        )
    assert set(grades["log_loss_floor"]) == {grader.LOG_LOSS_FLOOR}


def test_the_run_receipt_reports_the_slate_scores(grader, settled):
    assert invoke(grader, settled) == 0
    receipt = json.loads(settled["receipt_path"].read_text(encoding="utf-8"))

    assert receipt["outcome"] == grader.OUTCOME_GRADED
    assert receipt["scores"]["events"] == 3
    assert receipt["counts"] == {grader.STATUS_GRADED: 3}
    assert receipt["published"] is False
    assert receipt["promotion_authority"] == "NONE"
    assert receipt["production_serving_was_affected"] is False


# ======================================================================
# pending settlement
# ======================================================================


def test_a_game_that_is_not_final_is_pending_not_graded(grader, tmp_path):
    """The one case where a box score exists and must not be read.

    A live game's partial box score is already in the stats tree, which is
    exactly how a half-played game becomes a fabricated grade. The status
    check comes first for that reason.
    """
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1, "final": False}], date=SLATE),
    )

    assert invoke(grader, paths) == 0

    grades = stored_grades(paths)
    row = grades.iloc[0]
    assert row["settlement_status"] == grader.STATUS_PENDING
    assert row["realized_result"] is None or pd.isna(row["realized_result"])
    assert row["realized_value"] is None or pd.isna(row["realized_value"])
    assert row["brier_contribution"] is None or pd.isna(row["brier_contribution"])
    assert "not confirmed final" in row["settlement_detail"]

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_PENDING
    assert receipt["scores"]["events"] == 0


def test_an_absent_box_score_source_is_pending_not_a_failure(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=None,
        games=None,
    )

    assert invoke(grader, paths) == 0
    grades = stored_grades(paths)
    assert set(grades["settlement_status"]) == {grader.STATUS_PENDING}
    assert "no box-score source" in grades.iloc[0]["settlement_detail"]


def test_a_partly_settled_slate_grades_only_the_settled_games(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62},
                {"game_id": 2, "player_id": 21, "line": 20.5, "q_over": 0.50},
                {"game_id": 3, "player_id": 31, "line": 15.5, "q_over": 0.44},
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame(
            [
                {"game_id": 1, "player_id": 11, "pts": 31.0},
                {"game_id": 2, "player_id": 21, "pts": 14.0},
                {"game_id": 3, "player_id": 31, "pts": 8.0},
            ],
            date=SLATE,
        ),
        games=games_frame(
            [{"game_id": 1}, {"game_id": 2}, {"game_id": 3, "final": False}],
            date=SLATE,
        ),
    )

    assert invoke(grader, paths) == 0

    grades = stored_grades(paths).set_index("game_id")
    assert grades.loc[1, "settlement_status"] == grader.STATUS_GRADED
    assert grades.loc[2, "settlement_status"] == grader.STATUS_GRADED
    assert grades.loc[3, "settlement_status"] == grader.STATUS_PENDING

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_GRADED
    assert receipt["counts"] == {
        grader.STATUS_GRADED: 2,
        grader.STATUS_PENDING: 1,
    }
    assert receipt["scores"]["events"] == 2


def test_a_pending_game_settles_in_place_on_a_later_run(grader, tmp_path):
    """The pending row is the thing a later run updates, not a second row."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1, "final": False}], date=SLATE),
    )
    assert invoke(grader, paths) == 0
    first = stored_grades(paths)
    assert set(first["settlement_status"]) == {grader.STATUS_PENDING}

    season_dir = paths["data_root"] / "raw" / "seasons" / f"season={SEASON}"
    games_frame([{"game_id": 1}], date=SLATE).to_parquet(
        season_dir / "games.parquet", index=False
    )

    assert invoke(grader, paths) == 0
    second = stored_grades(paths)

    assert len(second) == len(first) == 1
    assert second.iloc[0]["prediction_id"] == first.iloc[0]["prediction_id"]
    assert second.iloc[0]["settlement_status"] == grader.STATUS_GRADED


# ======================================================================
# idempotence
# ======================================================================


def test_rerunning_the_same_grading_does_not_duplicate_rows(grader, settled):
    assert invoke(grader, settled) == 0
    first = stored_grades(settled)

    for _ in range(3):
        assert invoke(grader, settled) == 0

    again = stored_grades(settled)

    assert len(again) == len(first) == 3
    assert again["prediction_id"].is_unique
    assert set(again["prediction_id"]) == set(first["prediction_id"])

    # Every column, including graded_at: a rerun that reaches the same grade
    # from the same inputs decided nothing, so it changes nothing.
    pd.testing.assert_frame_equal(
        first.sort_values("prediction_id").reset_index(drop=True),
        again.sort_values("prediction_id").reset_index(drop=True),
    )


def test_the_stored_rows_are_byte_identical_on_a_rerun(grader, settled):
    """Idempotent down to the artifact, which is what the sha256 attests."""
    assert invoke(grader, settled) == 0
    files = sorted(settled["grades_root"].rglob("*.parquet"))
    assert len(files) == 1
    before = grader.sha256_file(files[0])

    assert invoke(grader, settled) == 0

    assert grader.sha256_file(files[0]) == before


def test_a_rerun_keeps_the_time_the_grade_was_determined(grader, settled):
    """``graded_at`` is when the grade was reached, not when grading last ran.

    The run receipt already records when grading ran. If the stored row moved
    its timestamp too, the field would say nothing the receipt does not, and
    every row would look freshly decided on a morning when nothing was.
    """
    assert invoke(grader, settled) == 0
    before = stored_grades(settled).set_index("prediction_id")["graded_at"]

    assert invoke(grader, settled) == 0
    after = stored_grades(settled).set_index("prediction_id")["graded_at"]

    pd.testing.assert_series_equal(before.sort_index(), after.sort_index())


def test_settling_a_pending_row_advances_the_time_it_was_graded(grader, tmp_path):
    """The converse: a row whose grade did change was decided just now."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T13:30:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1, "final": False}], date=SLATE),
    )
    assert invoke(grader, paths) == 0
    pending = stored_grades(paths).iloc[0]
    assert pending["settlement_status"] == grader.STATUS_PENDING

    season_dir = paths["data_root"] / "raw" / "seasons" / f"season={SEASON}"
    games_frame([{"game_id": 1}], date=SLATE).to_parquet(
        season_dir / "games.parquet", index=False
    )

    assert invoke(grader, paths) == 0
    graded = stored_grades(paths).iloc[0]

    assert graded["settlement_status"] == grader.STATUS_GRADED
    assert graded["prediction_id"] == pending["prediction_id"]
    assert pd.Timestamp(graded["graded_at"]) > pd.Timestamp(pending["graded_at"])


def test_the_prediction_id_is_a_digest_of_what_was_predicted(grader, settled):
    assert invoke(grader, settled) == 0
    grades = stored_grades(settled)

    for _, row in grades.iterrows():
        assert row["prediction_id"] == grader.prediction_id(
            slate_date=row["slate_date"],
            game_id=row["game_id"],
            player_id=row["player_id"],
            prop_type=row["stat"],
            line_value=row["line"],
            vendor=row["vendor"],
        )

    # Two different lines on the same player are two different predictions.
    assert grader.prediction_id(
        slate_date=SLATE,
        game_id=1,
        player_id=11,
        prop_type="points",
        line_value=24.5,
        vendor="w",
    ) != grader.prediction_id(
        slate_date=SLATE,
        game_id=1,
        player_id=11,
        prop_type="points",
        line_value=25.5,
        vendor="w",
    )


# ======================================================================
# provenance
# ======================================================================


def test_priced_markets_that_changed_after_serving_are_refused(grader, tmp_path):
    """Corrupted provenance: the receipt no longer describes these rows."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    # Reprice the same slate without reissuing the receipt.
    priced_frame(
        [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.95}],
        priced_at="2026-11-04T22:05:00+00:00",
    ).to_parquet(paths["priced_path"], index=False)

    assert invoke(grader, paths) == 1

    assert not list(paths["grades_root"].rglob("*.parquet"))
    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_REFUSED
    assert "changed after they were served" in receipt["reason"]


def test_a_receipt_for_a_different_slate_is_refused(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
        receipt_slate="2026-11-03",
    )

    assert invoke(grader, paths) == 1
    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert "does not provenance these predictions" in receipt["reason"]


@pytest.mark.parametrize(
    "field", ["production_code_sha", "input_state_fingerprint"]
)
def test_a_receipt_missing_a_required_identity_is_refused(grader, tmp_path, field):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
        receipt_overrides={field: ""},
    )

    assert invoke(grader, paths) == 1
    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert field in receipt["reason"]


def test_priced_markets_with_no_receipt_at_all_are_refused(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )
    paths["serving_receipt"].unlink()

    assert invoke(grader, paths) == 1
    assert not list(paths["grades_root"].rglob("*.parquet"))


def test_a_slate_whose_games_are_scheduled_elsewhere_is_refused(grader, tmp_path):
    """Mismatched prediction and settlement identity.

    One slate's predictions scored against another slate's outcomes is the
    single most damaging thing a grader can do quietly, because the rows look
    perfectly well formed afterwards.
    """
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame(
            [{"game_id": 1, "player_id": 11, "pts": 31.0, "date": "2026-11-01"}],
            date="2026-11-01",
        ),
        games=games_frame([{"game_id": 1, "date": "2026-11-01"}], date="2026-11-01"),
    )

    assert invoke(grader, paths) == 1
    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert "dates elsewhere" in receipt["reason"]


# ======================================================================
# no lookahead
# ======================================================================


def test_a_slate_priced_after_its_own_date_is_refused(grader, tmp_path):
    """A price stamped the morning after could have been told the score."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-05T14:00:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 1

    assert not list(paths["grades_root"].rglob("*.parquet"))
    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert "cannot be scored as out-of-sample" in receipt["reason"]


def test_the_lookahead_check_is_refused_before_any_outcome_is_opened(grader, tmp_path):
    """A leaked slate must be refused even with no settlement tree present."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-06T09:00:00+00:00",
        ),
        stats=None,
        games=None,
    )

    assert invoke(grader, paths) == 1


def test_the_lookahead_boundary_is_the_slate_s_own_date(grader):
    """Pricing on game day is fine; pricing the next day is not.

    Midnight UTC on the following day is the earliest any game on the slate
    can be final, which is what makes it the boundary rather than a margin
    somebody chose.
    """
    for stamp in (
        "2026-11-03T18:00:00+00:00",
        "2026-11-04T00:00:00+00:00",
        "2026-11-04T23:59:59+00:00",
    ):
        grader.refuse_on_lookahead(prediction_timestamp=stamp, slate_date=SLATE)

    for stamp in ("2026-11-05T00:00:00+00:00", "2026-11-06T09:00:00+00:00"):
        with pytest.raises(grader.GradingRefused):
            grader.refuse_on_lookahead(prediction_timestamp=stamp, slate_date=SLATE)


def test_the_lookahead_check_reads_the_latest_price_on_the_slate(grader, tmp_path):
    """One late row contaminates the slate, so the maximum is the reading."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62},
                {"game_id": 1, "player_id": 12, "line": 18.5, "q_over": 0.40},
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame(
            [
                {"game_id": 1, "player_id": 11, "pts": 31.0},
                {"game_id": 1, "player_id": 12, "pts": 12.0},
            ],
            date=SLATE,
        ),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )
    assert invoke(grader, paths) == 0

    priced = pd.read_parquet(paths["priced_path"])
    priced.loc[1, "priced_at_utc"] = "2026-11-05T15:00:00+00:00"
    priced.to_parquet(paths["priced_path"], index=False)

    grader_module = load_grader()
    receipt = json.loads(paths["serving_receipt"].read_text(encoding="utf-8"))
    receipt["priced_market_artifact"]["sha256"] = grader_module.sha256_file(
        paths["priced_path"]
    )
    paths["serving_receipt"].write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    assert invoke(grader, paths) == 1


def test_no_serving_script_can_read_the_grade_store(grader):
    """Grading must be unable to feed back into prediction.

    The strongest structural statement available: the tree grades are written
    to is not named anywhere a serving entry point can reach.
    """
    tree = grader.GRADES_RELATIVE.as_posix()
    assert tree == "processed/incumbent_grades"

    for relative in (
        "scripts/10_predict_slate.py",
        "scripts/15_price_markets.py",
        "src/nba_prop_quant/production.py",
        "src/nba_prop_quant/pricing.py",
        "src/nba_prop_quant/slate.py",
        "src/nba_prop_quant/features.py",
        "src/nba_prop_quant/model.py",
    ):
        text = (PROJECT / relative).read_text(encoding="utf-8")
        assert "incumbent_grades" not in text, relative
        assert "grade_incumbent" not in text, relative


# ======================================================================
# settlement edge cases
# ======================================================================


def test_a_push_is_recorded_and_not_scored(grader, tmp_path):
    """The incumbent prices a non-push conditional, so a push has no score."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 25.0, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 25.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 0
    row = stored_grades(paths).iloc[0]

    assert row["settlement_status"] == grader.STATUS_PUSH
    assert row["realized_value"] == 25.0
    assert row["realized_result"] is None or pd.isna(row["realized_result"])
    assert row["brier_contribution"] is None or pd.isna(row["brier_contribution"])


def test_a_player_who_did_not_play_a_final_game_is_void(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62},
                {"game_id": 1, "player_id": 99, "line": 10.5, "q_over": 0.50},
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame(
            [
                {"game_id": 1, "player_id": 11, "pts": 31.0},
                {"game_id": 1, "player_id": 12, "minutes": 0.0, "pts": 0.0},
            ],
            date=SLATE,
        ),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 0
    grades = stored_grades(paths).set_index("player_id")

    assert grades.loc[11, "settlement_status"] == grader.STATUS_GRADED
    # Priced but absent from a final game's box score.
    assert grades.loc[99, "settlement_status"] == grader.STATUS_VOID_DID_NOT_PLAY


def test_a_prop_type_with_no_box_score_components_is_unsupported(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {
                    "game_id": 1,
                    "player_id": 11,
                    "line": 0.5,
                    "q_over": 0.62,
                    "prop_type": "double_double",
                }
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 0
    row = stored_grades(paths).iloc[0]
    assert row["settlement_status"] == grader.STATUS_UNSUPPORTED
    assert row["brier_contribution"] is None or pd.isna(row["brier_contribution"])


def test_a_combination_prop_sums_its_components(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {
                    "game_id": 1,
                    "player_id": 11,
                    "line": 36.5,
                    "q_over": 0.60,
                    "prop_type": "points_rebounds_assists",
                }
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame(
            [{"game_id": 1, "player_id": 11, "pts": 24.0, "reb": 9.0, "ast": 7.0}],
            date=SLATE,
        ),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 0
    row = stored_grades(paths).iloc[0]
    assert row["realized_value"] == 40.0
    assert row["settlement_status"] == grader.STATUS_GRADED
    assert row["realized_result"] == 1.0


def test_duplicate_box_score_rows_are_refused_rather_than_aggregated(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame(
            [
                {"game_id": 1, "player_id": 11, "pts": 31.0},
                {"game_id": 1, "player_id": 11, "pts": 12.0},
            ],
            date=SLATE,
        ),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 1
    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert "ambiguous settlement" in receipt["reason"]


# ======================================================================
# an unserved day is not a failure
# ======================================================================


def test_a_day_the_incumbent_did_not_serve_is_a_clean_no_op(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=None,
        games=None,
    )

    assert invoke(grader, paths, slate_date="2026-11-05") == 0

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_NOT_SERVED
    assert not list(paths["grades_root"].rglob("*.parquet"))


def test_an_empty_priced_file_is_a_clean_no_op(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame([], priced_at="2026-11-04T22:05:00+00:00"),
        stats=None,
        games=None,
    )

    assert invoke(grader, paths) == 0
    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_NOT_SERVED


# ======================================================================
# catch-up
# ======================================================================


def test_an_earlier_pending_slate_is_revisited(grader, tmp_path):
    """Yesterday being unsettleable must not mean it is never graded.

    The lifecycle runs once a day, so a slate that had not finished when it
    ran would otherwise stay pending for ever.
    """
    yesterday = "2026-11-03"

    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 7, "player_id": 71, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-03T22:05:00+00:00",
        ),
        stats=stats_frame(
            [{"game_id": 7, "player_id": 71, "pts": 31.0, "date": yesterday}],
            date=yesterday,
        ),
        games=games_frame(
            [{"game_id": 7, "final": False, "date": yesterday}], date=yesterday
        ),
        slate_date=yesterday,
    )

    assert invoke(grader, paths, slate_date=yesterday) == 0
    assert set(stored_grades(paths)["settlement_status"]) == {grader.STATUS_PENDING}

    # Today: a new slate, and yesterday's game has since gone final.
    season_dir = paths["data_root"] / "raw" / "seasons" / f"season={SEASON}"
    games_frame([{"game_id": 7, "date": yesterday}], date=yesterday).to_parquet(
        season_dir / "games.parquet", index=False
    )

    assert (
        grader.main(
            [
                "--slate-date",
                SLATE,
                "--data-root",
                str(paths["data_root"]),
                "--grades-root",
                str(paths["grades_root"]),
                "--receipt-path",
                str(paths["receipt_path"]),
                "--catch-up-days",
                "7",
            ]
        )
        == 0
    )

    grades = stored_grades(paths)
    assert len(grades) == 1
    assert grades.iloc[0]["settlement_status"] == grader.STATUS_GRADED

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_NOT_SERVED
    assert [entry["slate_date"] for entry in receipt["caught_up"]] == [yesterday]


def test_catch_up_only_looks_at_days_with_unsettled_rows(grader, settled):
    assert invoke(grader, settled) == 0

    assert (
        grader.dates_needing_another_attempt(
            grades_root=settled["grades_root"],
            slate_date="2026-11-10",
            days=14,
        )
        == []
    )


def test_a_refused_earlier_slate_does_not_cost_the_current_one(grader, tmp_path):
    """Today's grading is never held hostage by an earlier day."""
    yesterday = "2026-11-03"

    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 7, "player_id": 71, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-03T22:05:00+00:00",
        ),
        stats=stats_frame(
            [{"game_id": 7, "player_id": 71, "pts": 31.0, "date": yesterday}],
            date=yesterday,
        ),
        games=games_frame(
            [{"game_id": 7, "final": False, "date": yesterday}], date=yesterday
        ),
        slate_date=yesterday,
    )
    assert invoke(grader, paths, slate_date=yesterday) == 0

    # Today the incumbent served nothing, and yesterday's receipt is gone.
    paths["serving_receipt"].unlink()

    assert (
        grader.main(
            [
                "--slate-date",
                SLATE,
                "--data-root",
                str(paths["data_root"]),
                "--grades-root",
                str(paths["grades_root"]),
                "--receipt-path",
                str(paths["receipt_path"]),
                "--catch-up-days",
                "7",
            ]
        )
        == 0
    )

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["caught_up"][0]["outcome"] == grader.OUTCOME_REFUSED
    assert "no serving receipt" in receipt["caught_up"][0]["reason"]


def test_a_caught_up_slate_cannot_borrow_another_slate_s_receipt(grader, tmp_path):
    """Provenance is per slate, so an explicit override is not inherited.

    The one thing catch-up must not do is reach for whatever receipt happens
    to be at hand: a grade row carrying another run's fit id, production SHA
    and input fingerprint would read as fully provenanced to the frozen
    policy's own gate.
    """
    yesterday = "2026-11-03"

    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 7, "player_id": 71, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-03T22:05:00+00:00",
        ),
        stats=stats_frame(
            [{"game_id": 7, "player_id": 71, "pts": 31.0, "date": yesterday}],
            date=yesterday,
        ),
        games=games_frame(
            [{"game_id": 7, "final": False, "date": yesterday}], date=yesterday
        ),
        slate_date=yesterday,
    )
    assert invoke(grader, paths, slate_date=yesterday) == 0

    elsewhere = tmp_path / "some_other_receipt.json"
    elsewhere.write_text(
        paths["serving_receipt"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    paths["serving_receipt"].unlink()

    assert (
        grader.main(
            [
                "--slate-date",
                SLATE,
                "--data-root",
                str(paths["data_root"]),
                "--serving-receipt",
                str(elsewhere),
                "--grades-root",
                str(paths["grades_root"]),
                "--receipt-path",
                str(paths["receipt_path"]),
                "--catch-up-days",
                "7",
            ]
        )
        == 0
    )

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["caught_up"][0]["outcome"] == grader.OUTCOME_REFUSED
    assert set(stored_grades(paths)["settlement_status"]) == {grader.STATUS_PENDING}


# ======================================================================
# the settlement rules are shared, not copied
# ======================================================================


def test_the_settlement_rules_come_from_the_immutable_grader(grader, rules):
    source = ENTRY_POINT.read_text(encoding="utf-8")

    assert "grade_external_test_capture.py" in source
    # None of the shared rules may be restated here.
    assert "PROP_COMPONENTS = {" not in source
    assert "def is_final_game" not in source
    assert "def played_indicator" not in source
    assert "def prop_actual" not in source
    assert "def target_season" not in source

    assert "points_rebounds_assists" in rules.PROP_COMPONENTS


def test_the_preferred_side_column_the_incumbent_actually_writes_is_first(grader):
    """The gap the static contract audit has been reporting.

    ``add_market_probability_layer`` writes ``model_preferred_side``, which
    the external grader's historical candidate list never learned, so a
    grader that only reused that list would find no side to grade on a real
    production file.
    """
    assert grader.SIDE_COLUMNS[0] == "model_preferred_side"

    pricing = (PROJECT / "src" / "nba_prop_quant" / "production.py").read_text(
        encoding="utf-8"
    )
    assert "model_preferred_side" in pricing
    assert "calibrated_q_over_nonpush" in pricing
    assert "calibrated_q_under_nonpush" in pricing
    assert grader.PROBABILITY_PAIRS[0] == (
        "calibrated_q_over_nonpush",
        "calibrated_q_under_nonpush",
    )
