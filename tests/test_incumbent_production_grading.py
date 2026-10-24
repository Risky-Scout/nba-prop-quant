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

import ast
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


def controller_constants(*names: str) -> dict[str, Any]:
    """Module-level constants the T-20 controller declares, read statically.

    The grader may not import the controller -- that would pull the fit
    registry and the frozen-bundle installer in behind it, which is the same
    reason the controller restates the grader's constants rather than
    importing them. So the agreement between the two is checked from the
    source text instead.
    """
    source = (PROJECT / "ops" / "run_t20_production_controller.py").read_text(
        encoding="utf-8"
    )

    wanted = set(names)
    found: dict[str, Any] = {}

    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue

            if isinstance(node.value, ast.Constant):
                found[target.id] = node.value.value
            else:
                # Path("processed") / "name", read as the set of its parts:
                # ast.walk does not promise the order of a BinOp's operands.
                found[target.id] = {
                    part.value
                    for part in ast.walk(node.value)
                    if isinstance(part, ast.Constant)
                }

    missing = wanted - set(found)

    assert not missing, f"the controller declares no {sorted(missing)}"

    return found


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
    """Priced markets in the shape ``scripts/15_price_markets.py`` writes.

    ``market_over`` is optional for the same reason the grader treats the
    de-vigged market pair as optional: a day whose capture carried no market
    probability is still a gradeable day, and the fixtures that predate the
    benchmark are what prove it.
    """
    rows = []
    for record in records:
        over = float(record["q_over"])
        row = {
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

        # Optional for the same reason the market pair is: the breakdowns
        # narrow when a column is absent rather than refusing the day.
        for name in ("minutes", "over_odds", "under_odds"):
            if name in record:
                row["expected_minutes" if name == "minutes" else name] = (
                    record[name]
                )

        if "market_over" in record:
            market_over = record["market_over"]
            row["market_devig_q_over"] = (
                None if market_over is None else float(market_over)
            )
            row["market_devig_q_under"] = (
                None if market_over is None else 1.0 - float(market_over)
            )

        rows.append(row)

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
# the benchmark: the same rows, scored against the market
# ======================================================================


@pytest.fixture
def settled_against_the_market(tmp_path):
    """A settled slate whose capture carried the de-vigged market too.

    The three wagers are chosen so the comparison has a known direction on
    each: the model is nearer the outcome on the first, the market is nearer
    on the second, and the two are level on the third.
    """
    return build_root(
        tmp_path,
        priced=priced_frame(
            [
                # realized 31 over 24.5, so "over" happened. Model 0.62,
                # market 0.50: the model was nearer.
                {
                    "game_id": 1,
                    "player_id": 11,
                    "line": 24.5,
                    "q_over": 0.62,
                    "market_over": 0.50,
                },
                # realized 12 under 18.5, so "over" did not happen. Model
                # 0.40, market 0.20: the market was nearer.
                {
                    "game_id": 1,
                    "player_id": 12,
                    "line": 18.5,
                    "q_over": 0.40,
                    "market_over": 0.20,
                },
                # identical probabilities, so this row contributes nothing to
                # either delta and proves the pairing is per-row.
                {
                    "game_id": 2,
                    "player_id": 21,
                    "line": 8.5,
                    "q_over": 0.55,
                    "market_over": 0.55,
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


def test_the_market_is_scored_on_the_identical_outcome_and_side(
    grader, settled_against_the_market
):
    """Both sides of every row come from one capture and one realization."""
    assert invoke(grader, settled_against_the_market) == 0

    grades = stored_grades(settled_against_the_market).set_index("player_id")

    assert grades["market_probability"].notna().all()
    assert grades["market_brier_contribution"].notna().all()
    assert grades["market_log_loss_contribution"].notna().all()

    for player_id, market_over in ((11, 0.50), (12, 0.20), (21, 0.55)):
        row = grades.loc[player_id]

        # The under row reads the under column, which is the same convention
        # the model's own probability is read on.
        expected = market_over if row["side"] == "over" else 1.0 - market_over

        assert row["market_probability"] == pytest.approx(expected)

        assert row["market_brier_contribution"] == pytest.approx(
            grader.brier_contribution(expected, row["realized_result"])
        )
        assert row["market_log_loss_contribution"] == pytest.approx(
            grader.log_loss_contribution(expected, row["realized_result"])
        )

        assert row["market_probability_column"] == (
            "market_devig_q_over" if row["side"] == "over" else "market_devig_q_under"
        )


def test_the_receipt_reports_the_delta_against_the_market(
    grader, settled_against_the_market
):
    """Model minus market, so a negative delta is the model ahead."""
    assert invoke(grader, settled_against_the_market) == 0

    receipt = json.loads(
        settled_against_the_market["receipt_path"].read_text(encoding="utf-8")
    )

    scores = receipt["scores"]

    assert scores["events"] == 3
    assert scores["market_events"] == 3

    grades = stored_grades(settled_against_the_market)

    assert scores["market_brier"] == pytest.approx(
        grades["market_brier_contribution"].mean()
    )
    assert scores["market_log_loss"] == pytest.approx(
        grades["market_log_loss_contribution"].mean()
    )

    assert scores["brier_delta_model_minus_market"] == pytest.approx(
        grades["brier_contribution"].mean()
        - grades["market_brier_contribution"].mean()
    )
    assert scores["log_loss_delta_model_minus_market"] == pytest.approx(
        grades["log_loss_contribution"].mean()
        - grades["market_log_loss_contribution"].mean()
    )

    # The sign is the whole point of reporting a delta at all, so it is
    # pinned rather than inferred: on this slate the market wins, because the
    # row it is nearer on is further from the model than the row the model
    # wins is from the market.
    assert scores["brier_delta_model_minus_market"] > 0


def test_a_row_the_market_priced_better_is_recorded_not_hidden(
    grader, settled_against_the_market
):
    """A grader that could only record favourable rows would be worthless."""
    assert invoke(grader, settled_against_the_market) == 0

    grades = stored_grades(settled_against_the_market).set_index("player_id")

    model_ahead = grades.loc[11]
    market_ahead = grades.loc[12]
    level = grades.loc[21]

    assert model_ahead["brier_contribution"] < model_ahead["market_brier_contribution"]
    assert (
        market_ahead["brier_contribution"] > market_ahead["market_brier_contribution"]
    )
    assert level["brier_contribution"] == pytest.approx(
        level["market_brier_contribution"]
    )


def test_a_capture_without_a_market_probability_still_grades_the_model(
    grader, settled
):
    """The benchmark is optional; the model's own score is not.

    ``settled`` carries no market columns at all, which is the shape every
    slate priced before this benchmark existed has.
    """
    assert invoke(grader, settled) == 0

    grades = stored_grades(settled)

    assert grades["brier_contribution"].notna().all()
    assert grades["market_probability"].isna().all()
    assert grades["market_brier_contribution"].isna().all()
    assert grades["market_log_loss_contribution"].isna().all()
    assert grades["market_probability_column"].isna().all()

    receipt = json.loads(settled["receipt_path"].read_text(encoding="utf-8"))

    assert receipt["scores"]["events"] == 3
    assert receipt["scores"]["market_events"] == 0
    assert receipt["scores"]["brier_delta_model_minus_market"] is None


def test_the_delta_is_taken_over_the_rows_that_carry_both_sides(grader, tmp_path):
    """A delta over two different row sets is not a delta.

    One of these three wagers has no market probability, so the comparison
    covers two rows while the slate grades three, and the reported delta must
    be the two-row one rather than a slate-wide model figure differenced
    against a two-row market figure.
    """
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {
                    "game_id": 1,
                    "player_id": 11,
                    "line": 24.5,
                    "q_over": 0.62,
                    "market_over": 0.50,
                },
                {
                    "game_id": 1,
                    "player_id": 12,
                    "line": 18.5,
                    "q_over": 0.40,
                    "market_over": 0.20,
                },
                # Priced, gradeable, and quoted with no market probability.
                {
                    "game_id": 2,
                    "player_id": 21,
                    "line": 8.5,
                    "q_over": 0.99,
                    "market_over": None,
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

    assert invoke(grader, paths) == 0

    grades = stored_grades(paths)
    scores = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))["scores"]

    assert scores["events"] == 3
    assert scores["market_events"] == 2

    paired = grades.loc[grades["market_brier_contribution"].notna()]

    assert len(paired) == 2

    assert scores["brier_delta_model_minus_market"] == pytest.approx(
        paired["brier_contribution"].mean()
        - paired["market_brier_contribution"].mean()
    )

    # The slate-wide model Brier is not the paired one, which is exactly the
    # confusion this pins against.
    assert scores["brier"] != pytest.approx(paired["brier_contribution"].mean())


def test_an_unsettled_row_carries_no_market_score_either(grader, tmp_path):
    """A benchmark scored before the outcome exists would be a fabrication."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {
                    "game_id": 1,
                    "player_id": 11,
                    "line": 24.5,
                    "q_over": 0.62,
                    "market_over": 0.50,
                }
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame(
            [{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE
        ),
        games=games_frame([{"game_id": 1, "final": False}], date=SLATE),
    )

    assert invoke(grader, paths) == 0

    grades = stored_grades(paths)

    assert set(grades["settlement_status"]) == {grader.STATUS_PENDING}

    # The market's probability is a fact about the capture and is recorded.
    # Its score is a fact about the outcome and is not.
    assert grades["market_probability"].notna().all()
    assert grades["market_brier_contribution"].isna().all()
    assert grades["market_log_loss_contribution"].isna().all()


def test_the_report_states_the_delta_and_its_direction(
    grader, settled_against_the_market
):
    assert invoke(grader, settled_against_the_market) == 0

    receipt = json.loads(
        settled_against_the_market["receipt_path"].read_text(encoding="utf-8")
    )

    report = grader.render(receipt)

    assert "market brier" in report
    assert "market-paired events : 3" in report
    assert "negative is the model ahead" in report


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
# the daily record, cut by the dimensions it is kept by
# ======================================================================
#
# Overall scores alone cannot answer the question the daily record exists to
# answer: the model's edge is not uniform across props, across rotation
# minutes, across the side it took or across how confident it was, and a
# single Brier hides exactly the place where it is losing.


@pytest.fixture
def many(tmp_path, grader):
    """A settled slate wide enough to cut four ways.

    Deliberately not uniform: two props, both sides, three minutes bands and
    three confidence tiers, with the realised results arranged so a reader
    could tell the groups apart from each other.
    """
    records = [
        {
            "game_id": 1,
            "player_id": 11,
            "line": 24.5,
            "q_over": 0.68,
            "minutes": 35.0,
            "over_odds": -110,
        },
        {
            "game_id": 1,
            "player_id": 12,
            "line": 18.5,
            "q_over": 0.44,
            "side": "under",
            "minutes": 27.0,
            "under_odds": 105,
        },
        {
            "game_id": 1,
            "player_id": 13,
            "line": 6.5,
            "q_over": 0.57,
            "prop_type": "assists",
            "minutes": 19.0,
            "over_odds": 120,
        },
        {
            "game_id": 2,
            "player_id": 21,
            "line": 8.5,
            "q_over": 0.38,
            "prop_type": "assists",
            "side": "under",
            "minutes": 33.0,
            "under_odds": -130,
        },
    ]

    paths = build_root(
        tmp_path,
        priced=priced_frame(records, priced_at="2026-11-04T22:05:00+00:00"),
        stats=stats_frame(
            [
                {"game_id": 1, "player_id": 11, "pts": 31.0},
                {"game_id": 1, "player_id": 12, "pts": 12.0},
                {"game_id": 1, "player_id": 13, "ast": 9.0},
                {"game_id": 2, "player_id": 21, "ast": 4.0},
            ],
            date=SLATE,
        ),
        games=games_frame([{"game_id": 1}, {"game_id": 2}], date=SLATE),
    )

    assert invoke(grader, paths) == 0

    return paths, json.loads(paths["receipt_path"].read_text(encoding="utf-8"))


def test_the_record_is_kept_by_every_dimension_it_declares(grader, many):
    _, receipt = many

    breakdowns = receipt["breakdowns"]

    for name, _ in grader.BREAKDOWN_DIMENSIONS:
        assert name in breakdowns, f"{name} is declared and not reported"

        assert breakdowns[name], f"{name} was reported with no groups"


def test_each_group_carries_the_same_figures_the_slate_does(grader, many):
    _, receipt = many

    for name, _ in grader.BREAKDOWN_DIMENSIONS:
        for label, group in receipt["breakdowns"][name].items():
            for field in ("events", "brier", "log_loss", "base_rate"):
                assert group[field] is not None, f"{name}/{label} has no {field}"

            assert "market_events" in group

            assert "flat_stake" in group


def test_every_dimension_accounts_for_every_settled_row(grader, many):
    """A cut that drops rows is a cut a reader cannot reconcile."""
    _, receipt = many

    settled = receipt["scores"]["events"]

    for name, _ in grader.BREAKDOWN_DIMENSIONS:
        total = sum(
            group["events"] for group in receipt["breakdowns"][name].values()
        )

        assert total == settled, (
            f"{name} accounts for {total} of {settled} settled rows"
        )


def test_the_breakdowns_are_the_cuts_the_daily_record_is_asked_for(grader):
    """Named rather than inferred, so the set cannot quietly shrink."""
    assert {name for name, _ in grader.BREAKDOWN_DIMENSIONS} == {
        "by_prop",
        "by_expected_minutes_band",
        "by_side",
        "by_confidence_tier",
    }


def test_the_prop_cut_is_by_the_prop_actually_priced(grader, many):
    _, receipt = many

    assert set(receipt["breakdowns"]["by_prop"]) == {"points", "assists"}

    assert receipt["breakdowns"]["by_prop"]["points"]["events"] == 2

    assert receipt["breakdowns"]["by_prop"]["assists"]["events"] == 2


def test_the_side_cut_separates_over_from_under(grader, many):
    _, receipt = many

    assert set(receipt["breakdowns"]["by_side"]) == {"over", "under"}


def test_the_minutes_cut_uses_the_declared_bands(grader, many):
    _, receipt = many

    bands = set(receipt["breakdowns"]["by_expected_minutes_band"])

    assert bands <= {label for label, _, _ in grader.EXPECTED_MINUTES_BANDS}

    # 19 is a rotation player, 27 a starter, 33 and 35 leading starters.
    assert bands == {"12-24", "24-32", "32+"}

    assert receipt["breakdowns"]["by_expected_minutes_band"]["32+"]["events"] == 2


def test_the_bands_cover_the_line_without_overlapping(grader):
    """Half-open and contiguous, so nothing falls between two bands."""
    for bands in (grader.EXPECTED_MINUTES_BANDS, grader.CONFIDENCE_TIERS):
        for (_, _, upper), (_, lower, _) in zip(bands, bands[1:]):
            assert upper == lower

        assert bands[-1][2] == float("inf")

        for value in (0.0, 0.499, 11.9, 12.0, 23.99, 31.0, 55.0, 1000.0):
            matched = [
                label for label, low, high in bands if low <= value < high
            ]

            assert len(matched) == 1, f"{value} matched {matched}"


def test_the_confidence_cut_is_by_the_probability_that_was_scored(grader, many):
    """Not by a published tier label, which could disagree with the score."""
    _, receipt = many

    tiers = receipt["breakdowns"]["by_confidence_tier"]

    # 0.68 over, 0.56 under, 0.57 over, 0.62 under.
    assert set(tiers) == {"55-60", "60-65", "65+"}

    assert tiers["55-60"]["events"] == 2


def test_an_absent_dimension_is_reported_empty_rather_than_missing(
    grader, tmp_path
):
    """"No row carried this" and "this was never asked" are different."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 0

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))

    # The fixture prices no minutes column and no odds.
    assert receipt["breakdowns"]["by_expected_minutes_band"] == {}

    assert receipt["breakdowns"]["by_prop"]["points"]["events"] == 1

    assert receipt["scores"]["flat_stake"]["staked_events"] == 0

    assert receipt["scores"]["flat_stake"]["return_on_stake"] is None


def test_the_flat_stake_return_is_one_unit_at_the_quoted_price(grader, many):
    """Arithmetic, checked by hand rather than against the implementation."""
    _, receipt = many

    stake = receipt["scores"]["flat_stake"]

    assert stake["staked_events"] == 4

    assert stake["staked"] == pytest.approx(4.0)

    # over 24.5 at -110 on 31 points: won, +100/110.
    # under 18.5 at +105 on 12 points: won, +1.05.
    # over 6.5 at +120 on 9 assists: won, +1.20.
    # under 8.5 at -130 on 4 assists: won, +100/130.
    expected = 100 / 110 + 1.05 + 1.20 + 100 / 130

    assert stake["profit"] == pytest.approx(expected)

    assert stake["return_on_stake"] == pytest.approx(expected / 4.0)


def test_a_losing_prediction_costs_its_whole_stake(grader, tmp_path):
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {
                    "game_id": 1,
                    "player_id": 11,
                    "line": 24.5,
                    "q_over": 0.62,
                    "over_odds": 150,
                }
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 12.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 0

    stake = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))["scores"][
        "flat_stake"
    ]

    assert stake["profit"] == pytest.approx(-1.0)

    assert stake["return_on_stake"] == pytest.approx(-1.0)


def test_a_push_is_neither_staked_nor_scored(grader, tmp_path):
    """Settlement removes a push from the comparison; the stake follows it."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {
                    "game_id": 1,
                    "player_id": 11,
                    "line": 24.0,
                    "q_over": 0.62,
                    "over_odds": -110,
                }
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 24.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 0

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))

    assert receipt["counts"].get(grader.STATUS_PUSH) == 1

    assert receipt["scores"]["flat_stake"]["staked_events"] == 0


@pytest.mark.parametrize(
    "odds,profit",
    [(100, 1.0), (-100, 1.0), (150, 1.5), (-200, 0.5), (0, None), (None, None)],
)
def test_an_american_price_converts_to_the_profit_it_pays(grader, odds, profit):
    """Zero is not a price and may not be read as even money."""
    assert grader.american_profit(odds) == (
        None if profit is None else pytest.approx(profit)
    )


def test_a_row_with_no_price_for_the_side_taken_is_not_staked(grader, tmp_path):
    """The model took under; only the over was quoted."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [
                {
                    "game_id": 1,
                    "player_id": 11,
                    "line": 24.5,
                    "q_over": 0.38,
                    "side": "under",
                    "over_odds": -110,
                }
            ],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 12.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    assert invoke(grader, paths) == 0

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))

    assert receipt["scores"]["events"] == 1

    assert receipt["scores"]["flat_stake"]["staked_events"] == 0


def test_the_breakdowns_reach_nothing_that_serves(grader, many):
    """Reported, never gated. The outcome is decided before any of this."""
    paths, receipt = many

    assert receipt["outcome"] == grader.OUTCOME_GRADED

    assert receipt["production_serving_was_affected"] is False

    assert receipt["promotion_authority"] == "NONE"

    assert receipt["published"] is False

    report = grader.render(receipt)

    assert "by prop" in report

    assert "flat-stake return" in report

    assert "leaves serving authority untouched" in report


def test_the_stored_rows_carry_the_dimensions_the_report_cuts_by(grader, many):
    """So a reader can recompute every cut from the grade store itself."""
    paths, _ = many

    rows = stored_grades(paths)

    for column in (
        "expected_minutes",
        "expected_minutes_band",
        "confidence_tier",
        "flat_stake_odds",
        "flat_stake_profit_if_won",
    ):
        assert column in rows.columns, column

    settled = rows[rows["settlement_status"] == grader.STATUS_GRADED]

    assert settled["expected_minutes_band"].notna().all()

    assert settled["confidence_tier"].notna().all()


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


# ======================================================================
# provenance for a slate that was served more than once
# ======================================================================
#
# A real slate is served several times: the morning lifecycle serves it, then
# every advisory window and every T-20 segment serves it again, and each serve
# rewrites the same priced-markets file. Only the morning receipt is kept in
# ``processed/incumbent_serving``, so from 2026-10-23 onwards the bytes that
# survived the night were provenanced by nothing the grader looked at and
# every published slate was refused. The controller now records each serve's
# receipt in ``processed/incumbent_serving_provenance/<slate>.jsonl`` and these
# are the tests that the grader reads it without relaxing anything.


def served_again(
    paths: dict[str, Path],
    grader,
    *,
    priced: pd.DataFrame | None = None,
    slate_date: str = SLATE,
    overrides: dict[str, Any] | None = None,
    record_type: str | None = "incumbent_serving_provenance",
) -> dict[str, Any]:
    """Reprice the slate and append the receipt for the new bytes."""
    if priced is not None:
        priced.to_parquet(paths["priced_path"], index=False)

    row = {
        "generated_at": "2026-11-04T23:40:00+00:00",
        "incumbent_authority": "frozen_deployment_bundle",
        "incumbent_fit_id": None,
        "incumbent_version": "freeze-2026-10-01",
        "input_state_fingerprint": "e5c2" * 16,
        "model_authority": "incumbent",
        "offset_minutes": 20,
        "outcome": "SERVED",
        "prediction_artifact": {"path": "projections", "sha256": "cd" * 32},
        "priced_market_artifact": {
            "path": str(paths["priced_path"]),
            "sha256": grader.sha256_file(paths["priced_path"]),
        },
        "production_code_sha": "ee63d6c36b53ac92584ca6a79cf734d413a151b8",
        "recorded_at_utc": "2026-11-04T23:40:02+00:00",
        "served_window_ids": ["1:T-20m"],
        "slate_date": slate_date,
    }

    if record_type is not None:
        row["record_type"] = record_type

    row.update(overrides or {})

    ledger = (
        paths["data_root"]
        / grader.SERVING_PROVENANCE_RELATIVE
        / f"{slate_date}.jsonl"
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)

    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

    return row


def by_convention(grader, paths: dict[str, Path], slate_date: str = SLATE) -> int:
    """Grade without naming a receipt, which is how production calls it.

    ``invoke`` passes ``--serving-receipt`` explicitly and so deliberately
    opts out of the ledger. The lifecycle does not pass it, so these tests
    must not either.
    """
    return grader.main(
        [
            "--slate-date",
            slate_date,
            "--data-root",
            str(paths["data_root"]),
            "--grades-root",
            str(paths["grades_root"]),
            "--receipt-path",
            str(paths["receipt_path"]),
            "--catch-up-days",
            "0",
        ]
    )


@pytest.fixture
def reserved(tmp_path, grader):
    """A settled slate whose evening serve replaced the morning's bytes."""
    paths = build_root(
        tmp_path,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.62}],
            priced_at="2026-11-04T22:05:00+00:00",
        ),
        stats=stats_frame([{"game_id": 1, "player_id": 11, "pts": 31.0}], date=SLATE),
        games=games_frame([{"game_id": 1}], date=SLATE),
    )

    row = served_again(
        paths,
        grader,
        priced=priced_frame(
            [{"game_id": 1, "player_id": 11, "line": 24.5, "q_over": 0.71}],
            priced_at="2026-11-04T23:40:00+00:00",
        ),
    )

    return paths, row


def test_the_morning_receipt_alone_cannot_grade_a_reserved_slate(
    grader, reserved
):
    """The defect, stated as the state it leaves behind.

    Without the ledger this is all the grader had, and it is the exact
    refusal run 40082999575 recorded for 2026-10-23.
    """
    paths, _ = reserved

    assert invoke(grader, paths) == 1

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_REFUSED
    assert "changed after they were served" in receipt["reason"]


def test_the_evening_serve_s_own_receipt_provenances_the_surviving_bytes(
    grader, reserved
):
    paths, _ = reserved

    assert by_convention(grader, paths) == 0

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_GRADED

    rows = stored_grades(paths)
    assert len(rows) == 1


def test_the_grade_records_the_identities_of_the_serve_that_wrote_the_rows(
    grader, reserved
):
    """Not the morning's. A grade row's provenance has to be its own."""
    paths, row = reserved

    assert by_convention(grader, paths) == 0

    rows = stored_grades(paths)

    assert set(rows["production_code_sha"]) == {row["production_code_sha"]}
    assert set(rows["input_state_fingerprint"]) == {
        row["input_state_fingerprint"]
    }
    assert set(rows["prediction_artifact_sha256"]) == {
        row["priced_market_artifact"]["sha256"]
    }


def test_the_serve_that_wrote_the_bytes_is_used_even_with_a_later_serve_logged(
    grader, reserved
):
    """Newest first, but correctness decides: only one digest can match.

    A segment that served and then refused before repricing leaves a newer
    row describing bytes that are not on disk. It must be passed over rather
    than refuse the slate.
    """
    paths, row = reserved

    served_again(
        paths,
        grader,
        overrides={
            "priced_market_artifact": {
                "path": str(paths["priced_path"]),
                "sha256": "f0" * 32,
            },
            "production_code_sha": "0" * 40,
            "recorded_at_utc": "2026-11-04T23:55:02+00:00",
        },
    )

    assert by_convention(grader, paths) == 0

    rows = stored_grades(paths)
    assert set(rows["production_code_sha"]) == {row["production_code_sha"]}


def test_a_ledger_that_describes_none_of_the_bytes_is_refused(grader, reserved):
    """Fail closed, and say so. No candidate means no grade."""
    paths, _ = reserved

    ledger = (
        paths["data_root"] / grader.SERVING_PROVENANCE_RELATIVE / f"{SLATE}.jsonl"
    )
    ledger.write_text("", encoding="utf-8")

    served_again(
        paths,
        grader,
        overrides={
            "priced_market_artifact": {
                "path": str(paths["priced_path"]),
                "sha256": "f0" * 32,
            }
        },
    )

    assert by_convention(grader, paths) == 1

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_REFUSED
    assert "none of the 2 serving receipts" in receipt["reason"]
    assert not list(paths["grades_root"].rglob("*.parquet"))


def test_a_ledger_row_for_another_slate_is_not_provenance(grader, reserved):
    """verify_receipt still decides, and it still checks the slate."""
    paths, _ = reserved

    ledger = (
        paths["data_root"] / grader.SERVING_PROVENANCE_RELATIVE / f"{SLATE}.jsonl"
    )
    ledger.write_text("", encoding="utf-8")

    served_again(paths, grader, overrides={"slate_date": "2026-11-03"})

    assert by_convention(grader, paths) == 1

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert "does not provenance these predictions" in receipt["reason"]


@pytest.mark.parametrize(
    "field", ["production_code_sha", "input_state_fingerprint"]
)
def test_a_ledger_row_missing_a_required_identity_is_not_provenance(
    grader, reserved, field
):
    paths, _ = reserved

    ledger = (
        paths["data_root"] / grader.SERVING_PROVENANCE_RELATIVE / f"{SLATE}.jsonl"
    )
    ledger.write_text("", encoding="utf-8")

    served_again(paths, grader, overrides={field: ""})

    assert by_convention(grader, paths) == 1

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert field in receipt["reason"]


def test_a_row_without_the_ledger_s_record_type_is_not_read(grader, reserved):
    """A later writer's rows in the same tree are skipped, not guessed at."""
    paths, _ = reserved

    ledger = (
        paths["data_root"] / grader.SERVING_PROVENANCE_RELATIVE / f"{SLATE}.jsonl"
    )
    ledger.write_text("", encoding="utf-8")

    served_again(paths, grader, record_type=None)

    assert by_convention(grader, paths) == 1

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert "changed after they were served" in receipt["reason"]


def test_an_advisory_serve_cannot_provenance_a_production_grade(
    grader, reserved
):
    """An advisory run may not contaminate the production record.

    The advisory serves rewrite the same priced-markets file, so their
    receipts are in the ledger and do describe real bytes -- but they
    published to an advisory destination. A night whose T-20 serve never
    landed has to refuse rather than score advisory prices as the
    incumbent's.
    """
    paths, _ = reserved

    ledger = (
        paths["data_root"] / grader.SERVING_PROVENANCE_RELATIVE / f"{SLATE}.jsonl"
    )
    ledger.write_text("", encoding="utf-8")

    served_again(
        paths,
        grader,
        overrides={
            grader.ADVISORY_RECEIPT_MARKER_FIELD: True,
            "advisory_capture_offset_minutes": 45,
            "offset_minutes": 45,
        },
    )

    assert by_convention(grader, paths) == 1

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_REFUSED
    assert not list(paths["grades_root"].rglob("*.parquet"))


def test_the_advisory_marker_the_grader_skips_is_the_one_that_is_written(
    grader,
):
    """Two modules naming the same field, pinned equal rather than imported.

    The grader may not import the controller -- it would pull the fit
    registry and the frozen-bundle installer behind it -- so the constant is
    restated and this is what keeps the two honest.
    """
    declared = controller_constants(
        "ADVISORY_RECEIPT_MARKER_FIELD",
        "SERVING_PROVENANCE_RECORD_TYPE",
        "SERVING_PROVENANCE_RELATIVE",
    )

    assert declared["ADVISORY_RECEIPT_MARKER_FIELD"] == (
        grader.ADVISORY_RECEIPT_MARKER_FIELD
    )

    assert declared["SERVING_PROVENANCE_RECORD_TYPE"] == (
        grader.SERVING_PROVENANCE_RECORD_TYPE
    )

    # Read from the source as the set of path segments the controller joins,
    # which is what Path("processed") / "incumbent_serving_provenance" is.
    assert declared["SERVING_PROVENANCE_RELATIVE"] == set(
        grader.SERVING_PROVENANCE_RELATIVE.parts
    )


def test_an_unreadable_ledger_line_does_not_cost_the_slate_its_grade(
    grader, reserved
):
    paths, _ = reserved

    ledger = (
        paths["data_root"] / grader.SERVING_PROVENANCE_RELATIVE / f"{SLATE}.jsonl"
    )

    ledger.write_text(
        "not json\n\n[1, 2, 3]\n" + ledger.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert by_convention(grader, paths) == 0


def test_an_explicitly_named_receipt_is_not_joined_by_the_ledger(
    grader, reserved
):
    """``--serving-receipt`` names one receipt and means it.

    Letting the ledger stand in would make the flag advisory, and the flag is
    how an operator provenances a slate deliberately.
    """
    paths, _ = reserved

    assert invoke(grader, paths) == 1

    receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
    assert receipt["outcome"] == grader.OUTCOME_REFUSED


def test_every_caught_up_slate_is_offered_its_own_ledger(grader, reserved):
    """The catch-up path grades slates served last night, so it needs it too.

    A slate is caught up a day or more after it was served, which is exactly
    when the morning receipt is least likely to describe the surviving bytes.
    Asserted on the call rather than by building a settled-then-pending slate,
    because the question here is only which provenance each date is offered.
    """
    paths, _ = reserved

    calls: list[tuple[str, object]] = []

    def record(**kwargs):
        calls.append((kwargs["slate_date"], kwargs.get("provenance_ledger")))

        return {"outcome": grader.OUTCOME_GRADED, "slate_date": kwargs["slate_date"]}

    grader_grade_slate = grader.grade_slate
    grader_dates = grader.dates_needing_another_attempt

    grader.grade_slate = record
    grader.dates_needing_another_attempt = lambda **_: ["2026-11-03"]

    try:
        grader.run(
            grader.parse_args(
                [
                    "--slate-date",
                    SLATE,
                    "--data-root",
                    str(paths["data_root"]),
                    "--grades-root",
                    str(paths["grades_root"]),
                    "--catch-up-days",
                    "7",
                ]
            )
        )
    finally:
        grader.grade_slate = grader_grade_slate
        grader.dates_needing_another_attempt = grader_dates

    assert [date for date, _ in calls] == [SLATE, "2026-11-03"]

    for date, ledger in calls:
        assert ledger == grader.serving_provenance_path(
            data_root=paths["data_root"], slate_date=date
        ), f"{date} was graded without its own serving provenance"


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


# ======================================================================
# the immutable priced artifact
# ======================================================================
#
# The exact failure of 2026-10-24: the morning grader refused 2026-10-23
# because the mutable date-keyed parquet held the evening's bytes and the
# serving receipt provenanced the morning's. Nothing had gone wrong with the
# model, the capture or the digest check. The artifact the immutable receipt
# pointed at was mutable, which is a lifecycle defect and not a provenance
# one, so the fix is to stop overwriting the thing being provenanced.


def seal_immutable(paths: dict[str, Path], grader, *, slate_date: str = SLATE) -> Path:
    """Copy the priced bytes where the serving entry point now copies them."""
    priced_path = paths["priced_path"]

    digest = grader.sha256_file(priced_path)

    sealed = (
        paths["data_root"]
        / "processed"
        / "served_priced_markets"
        / slate_date
        / f"{digest}.parquet"
    )
    sealed.parent.mkdir(parents=True, exist_ok=True)
    sealed.write_bytes(priced_path.read_bytes())

    receipt = json.loads(paths["serving_receipt"].read_text(encoding="utf-8"))

    receipt["immutable_priced_market_artifact"] = {
        "path": str(sealed),
        "relative": str(
            Path("processed")
            / "served_priced_markets"
            / slate_date
            / f"{digest}.parquet"
        ),
        "sha256": digest,
    }

    paths["serving_receipt"].write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )

    return sealed


def test_grading_follows_the_receipt_when_the_convenience_file_moves_on(
    grader, settled
):
    """Serve A, receipt A, later artifact B overwrites the date-keyed file.

    The grader must still load A, its digest must still match, and the grade
    must still be written -- which is what the production run of
    2026-10-24 could not do.
    """
    sealed = seal_immutable(settled, grader)

    original = sealed.read_bytes()

    # A second serve of the same slate rewrites the convenience path with
    # different bytes. This is the ordinary evening behaviour, not a fault.
    priced_frame(
        [{"game_id": 1, "player_id": 11, "line": 26.5, "q_over": 0.71}],
        priced_at="2026-11-04T23:40:00+00:00",
    ).to_parquet(settled["priced_path"], index=False)

    assert grader.sha256_file(settled["priced_path"]) != grader.sha256_file(sealed)

    assert invoke(grader, settled) == 0

    receipt = json.loads(settled["receipt_path"].read_text(encoding="utf-8"))

    assert receipt["outcome"] == grader.OUTCOME_GRADED
    assert receipt["priced_source_is_immutable"] is True
    assert receipt["priced_market_artifact"] == str(sealed)

    # Three rows, which is the sealed artifact's row count and not the one
    # row the overwritten convenience file now holds.
    assert receipt["scores"]["events"] == 3

    # And nothing touched the immutable bytes.
    assert sealed.read_bytes() == original


def test_the_digest_check_still_refuses_a_tampered_immutable_artifact(
    grader, settled
):
    """Preferring the immutable artifact is not trusting it."""
    sealed = seal_immutable(settled, grader)

    priced_frame(
        [{"game_id": 1, "player_id": 11, "line": 26.5, "q_over": 0.71}],
        priced_at="2026-11-04T23:40:00+00:00",
    ).to_parquet(sealed, index=False)

    assert invoke(grader, settled) == grader.EXIT_REFUSED

    receipt = json.loads(settled["receipt_path"].read_text(encoding="utf-8"))

    assert receipt["outcome"] == grader.OUTCOME_REFUSED
    assert "changed after they were served" in receipt["reason"]


def test_a_receipt_with_no_immutable_artifact_still_grades_the_old_way(
    grader, settled
):
    """Receipts written before the store existed are not orphaned by it."""
    assert invoke(grader, settled) == 0

    receipt = json.loads(settled["receipt_path"].read_text(encoding="utf-8"))

    assert receipt["outcome"] == grader.OUTCOME_GRADED
    assert receipt["priced_source_is_immutable"] is False
    assert receipt["priced_market_artifact"] == str(settled["priced_path"])


def test_the_immutable_artifact_is_found_when_the_data_root_moved(
    grader, settled, tmp_path
):
    """The recorded absolute path is from the machine that served."""
    sealed = seal_immutable(settled, grader)

    receipt = json.loads(settled["serving_receipt"].read_text(encoding="utf-8"))
    receipt["immutable_priced_market_artifact"]["path"] = (
        "/some/other/machine/served_priced_markets/x.parquet"
    )
    settled["serving_receipt"].write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )

    priced_frame(
        [{"game_id": 1, "player_id": 11, "line": 26.5, "q_over": 0.71}],
        priced_at="2026-11-04T23:40:00+00:00",
    ).to_parquet(settled["priced_path"], index=False)

    assert invoke(grader, settled) == 0

    graded = json.loads(settled["receipt_path"].read_text(encoding="utf-8"))

    assert graded["priced_market_artifact"] == str(sealed)
