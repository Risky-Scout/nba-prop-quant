#!/usr/bin/env python3
"""Grade the incumbent's served slate against realized box scores.

WHAT THIS IS FOR
----------------

The incumbent serves every production slate and its projections and prices
are recorded durably, but nothing read them back against what happened. Until
that loop closes there is no incumbent baseline, and without an incumbent
baseline the frozen promotion policy's central comparison -- candidate
against incumbent on Brier and log loss -- has only one side.

WHAT IT MAY AND MAY NOT DO
--------------------------

It reads the priced markets the incumbent already wrote, the serving receipt
that provenanced them, and the settled box scores in the durable rolling
tree. It writes grade rows and its own receipt. It does not predict, price,
promote, publish, refit or touch serving authority, and it writes nothing any
serving script reads.

NO LOOKAHEAD, IN BOTH DIRECTIONS
--------------------------------

Grading reads outcomes; prediction must not. Two things enforce that here:

* A game that is not confirmed final produces ``PENDING_SETTLEMENT`` even
  when a box score for it is already sitting in the stats tree. An
  unconfirmed outcome is not an outcome, and treating one as settled is how a
  partially-played game becomes a fabricated grade.
* The settlement source must have been written *after* the prediction. If the
  realized outcomes were already on disk when the slate was priced, the
  prediction had the opportunity to read them, and this run refuses rather
  than recording a score it cannot vouch for.

IDEMPOTENCE
-----------

A grade is keyed by the identity of the prediction it grades, so rerunning a
day replaces its rows instead of appending a second copy. Pending rows are
written too, which is what lets a later run settle them in place: the
cumulative evidence window is a count of graded rows, and a day that silently
wrote nothing would be indistinguishable from a day with no games.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.storage import upsert_parquet  # noqa: E402

#: The immutable external-test grader. Imported rather than copied: it already
#: owns how a prop type maps to box-score components, what counts as a final
#: game, whether a player played, and which season a date belongs to. A second
#: copy of any of those would be a second settlement rule.
SETTLEMENT_MODULE_PATH = PROJECT_ROOT / "ops" / "grade_external_test_capture.py"

#: Where the incumbent's serving artifacts live, relative to the data root.
PRICED_MARKETS_RELATIVE = Path("processed") / "priced_markets"
SEASONS_RELATIVE = Path("raw") / "seasons"

#: Where grades live. Under ``processed`` because they are derived, and in
#: their own tree because no serving script may read them.
GRADES_RELATIVE = Path("processed") / "incumbent_grades"
RECEIPTS_RELATIVE = GRADES_RELATIVE / "_receipts"

#: Where the serving receipt for a given slate lives, relative to the data
#: root. Durable rather than in the run's temporary directory, because a
#: slate is graded on a later day than it was served and a receipt that died
#: with the run would leave every grade unprovenanced.
SERVING_RECEIPTS_RELATIVE = Path("processed") / "incumbent_serving"

#: The grade store's primary key. Replacing on it is what makes a rerun
#: idempotent: one row per prediction, whatever happens to it afterwards.
GRADE_KEY = ["prediction_id"]

#: Settled outcomes.
STATUS_GRADED = "GRADED"
STATUS_PUSH = "PUSH"
STATUS_VOID_DID_NOT_PLAY = "VOID_PLAYER_DID_NOT_PLAY"
STATUS_UNSUPPORTED = "UNSUPPORTED_PROP_TYPE"

#: Not an outcome. The one answer allowed when settlement is absent or
#: incomplete, and deliberately not a score.
STATUS_PENDING = "PENDING_SETTLEMENT"

#: Run outcomes.
OUTCOME_GRADED = "GRADED"
OUTCOME_PENDING = "PENDING_SETTLEMENT"
OUTCOME_NOT_SERVED = "NO_SERVED_SLATE"
OUTCOME_REFUSED = "GRADING_REFUSED"

EXIT_OK = 0
EXIT_REFUSED = 1

#: The log-loss probability floor. The same declared bound the shadow's own
#: grading uses, so the two arms are scored on one convention.
LOG_LOSS_FLOOR = 1e-6

#: The incumbent's own runtime column names, ahead of the external grader's
#: historical candidates. ``model_preferred_side`` is what
#: ``add_market_probability_layer`` actually writes, and the static
#: pricing/grading contract audit has been reporting it as unevidenced
#: precisely because the historical list never learned the runtime name.
SIDE_COLUMNS: tuple[str, ...] = (
    "model_preferred_side",
    "preferred_side_calibrated",
    "preferred_side",
    "selected_bet_side",
    "bet_side",
    "monitor_side",
)

#: Over/under probability pairs, calibrated first. The incumbent prices a
#: non-push conditional probability, which is the convention the grade's
#: push handling assumes: a push is removed from the comparison rather than
#: scored against a probability that never contemplated it.
PROBABILITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("calibrated_q_over_nonpush", "calibrated_q_under_nonpush"),
    ("q_over_calibrated", "q_under_calibrated"),
    ("raw_q_over_nonpush", "raw_q_under_nonpush"),
    ("p_over_calibrated", "p_under_calibrated"),
)

#: Identity every priced row must carry to be gradeable at all.
REQUIRED_PRICED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "prop_type",
    "line_value",
)

#: Box-score columns settlement needs.
REQUIRED_STAT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "date",
    "minutes",
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "fg3m",
)


class GradingRefused(RuntimeError):
    """Grading cannot proceed without recording something untrue."""


# ----------------------------------------------------------------------
# shared settlement rules
# ----------------------------------------------------------------------


def load_settlement_rules() -> Any:
    """The external-test grader's settlement vocabulary, imported."""
    spec = importlib.util.spec_from_file_location(
        "incumbent_grading_settlement_rules", SETTLEMENT_MODULE_PATH
    )

    if spec is None or spec.loader is None:
        raise GradingRefused(
            f"cannot load the settlement rules from {SETTLEMENT_MODULE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001
        raise GradingRefused(
            f"cannot load the settlement rules from "
            f"{SETTLEMENT_MODULE_PATH}: {type(error).__name__}: {error}"
        ) from error

    for name in (
        "PROP_COMPONENTS",
        "is_final_game",
        "normalize_side",
        "played_indicator",
        "prop_actual",
        "target_season",
    ):
        if not hasattr(module, name):
            raise GradingRefused(
                f"{SETTLEMENT_MODULE_PATH} no longer provides {name}, so the "
                "settlement rules cannot be shared with it"
            )

    return module


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modified_utc(path: Path) -> str:
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _first_present(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    present = set(columns)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if stamp is None or pd.isna(stamp):
        return None
    return stamp


def prediction_id(
    *,
    slate_date: str,
    game_id: Any,
    player_id: Any,
    prop_type: Any,
    line_value: Any,
    vendor: Any,
) -> str:
    """A stable digest of what was predicted, not of the file it came from.

    Keyed on identity so a reserved slate updates its grade rather than
    appending a second one, and so a rerun of the same day replaces exactly
    the rows it wrote before. The artifact digest is recorded alongside as a
    column instead, which keeps the join to the prediction artifact immutable
    without making the key depend on bytes that may legitimately be rewritten.
    """
    payload = "|".join(
        str(part)
        for part in (slate_date, game_id, player_id, prop_type, line_value, vendor)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------


def verify_receipt(
    receipt: dict[str, Any], *, slate_date: str, priced_path: Path
) -> dict[str, Any]:
    """Check the serving receipt actually provenances this priced file.

    A receipt that names a different slate, or a different set of bytes than
    the file on disk holds, is not provenance for this grade. Recording the
    identities anyway would produce grade rows whose incumbent fit id,
    production SHA and input fingerprint belong to some other run, which is
    worse than having no grade: the frozen policy's provenance gate would
    read them as resolved.
    """
    recorded_date = str(receipt.get("slate_date") or "")

    if recorded_date != slate_date:
        raise GradingRefused(
            f"the serving receipt is for slate {recorded_date or 'unknown'}, "
            f"not {slate_date}, so it does not provenance these predictions"
        )

    artifact = receipt.get("priced_market_artifact") or {}
    expected = str(artifact.get("sha256") or "")

    if not expected:
        raise GradingRefused(
            "the serving receipt records no priced-market artifact digest, so "
            "the predictions being graded cannot be tied to the run that "
            "produced them"
        )

    actual = sha256_file(priced_path)

    if actual != expected:
        raise GradingRefused(
            f"{priced_path} holds {actual} but the serving receipt provenances "
            f"{expected}. The priced markets changed after they were served, "
            "so the receipt's identities do not describe these rows"
        )

    missing = [
        name
        for name in ("production_code_sha", "input_state_fingerprint")
        if not str(receipt.get(name) or "").strip()
    ]

    if missing:
        raise GradingRefused(
            "the serving receipt does not carry "
            + ", ".join(missing)
            + ", so a grade row cannot record the provenance the frozen "
            "policy requires"
        )

    return {
        "incumbent_authority": receipt.get("incumbent_authority"),
        "incumbent_fit_id": receipt.get("incumbent_fit_id"),
        "incumbent_version": receipt.get("incumbent_version"),
        "input_state_fingerprint": receipt.get("input_state_fingerprint"),
        "prediction_artifact_sha256": expected,
        "production_code_sha": receipt.get("production_code_sha"),
        "served_at": receipt.get("generated_at"),
    }


def refuse_on_lookahead(
    *, prediction_timestamp: str | None, slate_date: str
) -> None:
    """Refuse if the slate was priced after its own games could have finished.

    The question is whether the outcome was available to the prediction, and
    the only evidence for that inside the artifacts is when they were written
    relative to when the games were played. The earliest any game on a slate
    can be final is after midnight UTC on the following day -- a seven o'clock
    eastern tip ends around two in the morning UTC -- so a price stamped later
    than the slate's own date is a price that could have been informed by a
    result.

    Deliberately not a comparison against the settlement file's modification
    time. That is when the refresh last rewrote the season, not when the
    outcome became knowable, and in a durable tree rewritten every day it says
    nothing about any particular game. It is recorded on the grade row as the
    settlement timestamp, which is what it honestly is, and gated on nothing.
    """
    predicted = _as_timestamp(prediction_timestamp)

    if predicted is None:
        return

    boundary = pd.Timestamp(slate_date, tz="UTC")

    if predicted.normalize() > boundary:
        raise GradingRefused(
            f"the slate was priced at {predicted.isoformat()}, later than its "
            f"own date of {slate_date}. Games on that slate had already begun "
            "finishing, so these predictions cannot be scored as "
            "out-of-sample"
        )


# ----------------------------------------------------------------------
# settlement
# ----------------------------------------------------------------------


def settlement_sources(data_root: Path, slate_date: str, rules: Any) -> dict[str, Any]:
    """Locate and describe the box-score tree for a slate's season."""
    season = int(rules.target_season(pd.Timestamp(slate_date)))
    season_dir = Path(data_root) / SEASONS_RELATIVE / f"season={season}"

    stats_path = season_dir / "stats.parquet"
    games_path = season_dir / "games.parquet"

    absent = [str(path) for path in (stats_path, games_path) if not path.is_file()]

    if absent:
        return {
            "season": season,
            "available": False,
            "absent": absent,
            "games_path": str(games_path),
            "stats_path": str(stats_path),
            "settlement_timestamp": None,
        }

    return {
        "season": season,
        "available": True,
        "absent": [],
        "games_path": str(games_path),
        "games_sha256": sha256_file(games_path),
        "stats_path": str(stats_path),
        "stats_sha256": sha256_file(stats_path),
        # The moment the outcomes became durable. Not the moment the game
        # ended, which the box-score feed does not record, so the receipt says
        # which of the two it is rather than implying the other.
        "settlement_timestamp": max(_modified_utc(stats_path), _modified_utc(games_path)),
    }


def final_games(games: pd.DataFrame, game_ids: set[int], rules: Any) -> dict[int, bool]:
    """Which of a slate's games are confirmed final."""
    if "id" not in games.columns:
        raise GradingRefused(
            "the game-status source carries no id column, so no game can be "
            "confirmed final"
        )

    identifiers = pd.to_numeric(games["id"], errors="coerce")
    rows = games.loc[identifiers.isin(game_ids)]

    return {
        int(identifier): bool(rules.is_final_game(row))
        for identifier, (_, row) in zip(
            pd.to_numeric(rows["id"], errors="coerce").astype("Int64"),
            rows.iterrows(),
            strict=True,
        )
        if pd.notna(identifier)
    }


def game_dates(games: pd.DataFrame, game_ids: set[int]) -> dict[int, str | None]:
    if "date" not in games.columns:
        return {}
    identifiers = pd.to_numeric(games["id"], errors="coerce")
    rows = games.loc[identifiers.isin(game_ids)]
    dates = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    return {
        int(identifier): (None if pd.isna(date) else date.strftime("%Y-%m-%d"))
        for identifier, date in zip(
            pd.to_numeric(rows["id"], errors="coerce").astype("Int64"),
            dates,
            strict=True,
        )
        if pd.notna(identifier)
    }


def outcome_rows(stats: pd.DataFrame, game_ids: set[int]) -> pd.DataFrame:
    """One box-score row per player per game, for this slate's games only."""
    absent = [name for name in REQUIRED_STAT_COLUMNS if name not in stats.columns]

    if absent:
        raise GradingRefused(
            f"the outcome source is missing required columns: {sorted(absent)}"
        )

    rows = stats.loc[
        pd.to_numeric(stats["game_id"], errors="coerce").isin(game_ids)
    ].copy()

    duplicates = (
        rows.groupby(["game_id", "player_id"], dropna=False)
        .size()
        .loc[lambda series: series > 1]
    )

    if not duplicates.empty:
        raise GradingRefused(
            "the outcome source carries more than one row for the same game "
            "and player, which is ambiguous settlement rather than a result: "
            f"{duplicates.head(5).to_dict()}"
        )

    return rows.set_index(
        [
            pd.to_numeric(rows["game_id"], errors="coerce").astype("Int64"),
            pd.to_numeric(rows["player_id"], errors="coerce").astype("Int64"),
        ]
    )


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------


def brier_contribution(probability: float, realized: float) -> float:
    return float((probability - realized) ** 2)


def log_loss_contribution(
    probability: float, realized: float, floor: float = LOG_LOSS_FLOOR
) -> float:
    bounded = min(max(probability, floor), 1.0 - floor)
    return float(
        -(realized * math.log(bounded) + (1.0 - realized) * math.log1p(-bounded))
    )


def grade_rows(
    *,
    priced: pd.DataFrame,
    outcomes: pd.DataFrame,
    finals: dict[int, bool],
    slate_date: str,
    provenance: dict[str, Any],
    settlement: dict[str, Any],
    rules: Any,
    graded_at: str,
) -> pd.DataFrame:
    """One grade row per priced prediction, settled or pending."""
    absent = [name for name in REQUIRED_PRICED_COLUMNS if name not in priced.columns]

    if absent:
        raise GradingRefused(
            f"the priced markets are missing required settlement columns: "
            f"{sorted(absent)}"
        )

    side_column = _first_present(priced.columns, SIDE_COLUMNS)

    if side_column is None:
        raise GradingRefused(
            "the priced markets carry no recognised preferred-side column, so "
            f"there is no side to grade. Looked for: {list(SIDE_COLUMNS)}"
        )

    pair = next(
        (
            (over, under)
            for over, under in PROBABILITY_PAIRS
            if over in priced.columns and under in priced.columns
        ),
        None,
    )

    if pair is None:
        raise GradingRefused(
            "the priced markets carry no recognised over/under probability "
            f"pair, so there is no model probability to grade. Looked for: "
            f"{[list(candidate) for candidate in PROBABILITY_PAIRS]}"
        )

    over_column, under_column = pair
    prediction_timestamp_column = _first_present(
        priced.columns, ("priced_at_utc", "generated_at_utc")
    )

    records: list[dict[str, Any]] = []

    for _, row in priced.iterrows():
        game_id = pd.to_numeric(row["game_id"], errors="coerce")
        player_id = pd.to_numeric(row["player_id"], errors="coerce")
        prop_type = str(row["prop_type"]).strip().lower()
        line_value = pd.to_numeric(row["line_value"], errors="coerce")
        side = rules.normalize_side(row[side_column])

        probability = pd.to_numeric(
            row[over_column] if side == "over" else row[under_column],
            errors="coerce",
        )

        prediction_timestamp = (
            row[prediction_timestamp_column]
            if prediction_timestamp_column is not None
            else provenance.get("served_at")
        )

        record: dict[str, Any] = {
            "prediction_id": prediction_id(
                slate_date=slate_date,
                game_id=row["game_id"],
                player_id=row["player_id"],
                prop_type=prop_type,
                line_value=row["line_value"],
                vendor=row.get("vendor"),
            ),
            "slate_date": slate_date,
            "game_id": None if pd.isna(game_id) else int(game_id),
            "player_id": None if pd.isna(player_id) else int(player_id),
            "stat": prop_type,
            "side": side,
            "line": None if pd.isna(line_value) else float(line_value),
            "vendor": None if row.get("vendor") is None else str(row.get("vendor")),
            "model_probability": None if pd.isna(probability) else float(probability),
            "realized_value": None,
            "realized_result": None,
            "brier_contribution": None,
            "log_loss_contribution": None,
            "settlement_status": STATUS_PENDING,
            "settlement_detail": "",
            "prediction_timestamp": (
                None
                if prediction_timestamp is None or pd.isna(prediction_timestamp)
                else str(prediction_timestamp)
            ),
            "settlement_timestamp": settlement.get("settlement_timestamp"),
            "side_column": side_column,
            "probability_column": over_column if side == "over" else under_column,
            "log_loss_floor": LOG_LOSS_FLOOR,
            "graded_at": graded_at,
            **{key: provenance[key] for key in sorted(provenance)},
        }

        records.append(
            _settle(
                record,
                row=row,
                outcomes=outcomes,
                finals=finals,
                rules=rules,
                side=side,
                probability=probability,
                line_value=line_value,
                prop_type=prop_type,
                settlement_available=bool(settlement["available"]),
            )
        )

    return pd.DataFrame.from_records(records)


def _settle(
    record: dict[str, Any],
    *,
    row: pd.Series,
    outcomes: pd.DataFrame,
    finals: dict[int, bool],
    rules: Any,
    side: str | None,
    probability: Any,
    line_value: Any,
    prop_type: str,
    settlement_available: bool,
) -> dict[str, Any]:
    """Resolve one prediction, or declare it unsettled. Never both."""
    if not settlement_available:
        record["settlement_detail"] = "no box-score source for this season yet"
        return record

    if prop_type not in rules.PROP_COMPONENTS:
        record["settlement_status"] = STATUS_UNSUPPORTED
        record["settlement_detail"] = (
            f"{prop_type} has no declared box-score components, so it cannot "
            "be settled from a box score"
        )
        return record

    if side is None or pd.isna(probability):
        record["settlement_detail"] = (
            "the priced row carries no resolvable side or probability, so "
            "there is nothing to score"
        )
        return record

    game_id = record["game_id"]

    if game_id is None:
        record["settlement_detail"] = "the priced row carries no game id"
        return record

    if not finals.get(game_id, False):
        # Deliberately checked before any box score is read. A box score may
        # already exist for a game still in progress, and reading it here is
        # exactly the fabricated grade this status exists to prevent.
        record["settlement_detail"] = (
            "the game is not confirmed final, so its box score is not an "
            "outcome yet"
        )
        return record

    key = (record["game_id"], record["player_id"])

    if key not in outcomes.index:
        record["settlement_status"] = STATUS_VOID_DID_NOT_PLAY
        record["settlement_detail"] = (
            "the game is final and carries no box-score row for this player"
        )
        return record

    outcome = outcomes.loc[key]

    if isinstance(outcome, pd.DataFrame):
        outcome = outcome.iloc[0]

    if not rules.played_indicator(outcome):
        record["settlement_status"] = STATUS_VOID_DID_NOT_PLAY
        record["settlement_detail"] = "the player did not play in a final game"
        return record

    actual = rules.prop_actual(outcome, prop_type)

    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        record["settlement_detail"] = (
            "a component of this prop type is absent from the box-score row"
        )
        return record

    record["realized_value"] = float(actual)

    if float(actual) == float(line_value):
        record["settlement_status"] = STATUS_PUSH
        record["settlement_detail"] = (
            "the realized value equals the line. The incumbent prices a "
            "non-push conditional probability, so a push is removed from the "
            "comparison rather than scored against it"
        )
        return record

    went_over = float(actual) > float(line_value)
    realized = float(went_over if side == "over" else not went_over)

    record["realized_result"] = realized
    record["brier_contribution"] = brier_contribution(float(probability), realized)
    record["log_loss_contribution"] = log_loss_contribution(
        float(probability), realized
    )
    record["settlement_status"] = STATUS_GRADED
    record["settlement_detail"] = (
        f"realized {float(actual):g} against line {float(line_value):g}"
    )

    return record


# ----------------------------------------------------------------------
# one slate
# ----------------------------------------------------------------------


def grade_slate(
    *,
    slate_date: str,
    data_root: Path,
    receipt_path: Path | None,
    rules: Any,
    grades_root: Path,
) -> dict[str, Any]:
    """Grade one served slate and persist its rows. Returns the receipt."""
    graded_at = _utc_now()
    priced_path = Path(data_root) / PRICED_MARKETS_RELATIVE / f"{slate_date}.parquet"

    base: dict[str, Any] = {
        "entry_point": "ops/grade_incumbent_production_slate.py",
        "graded_at": graded_at,
        "slate_date": slate_date,
        "data_root": str(data_root),
        "priced_market_artifact": str(priced_path),
        "production_serving_was_affected": False,
        "promotion_authority": "NONE",
        "published": False,
    }

    if not priced_path.is_file():
        return {
            **base,
            "outcome": OUTCOME_NOT_SERVED,
            "reason": (
                f"the incumbent wrote no priced markets for {slate_date}, so "
                "there is nothing to grade"
            ),
            "counts": {},
            "grade_artifact": None,
        }

    if receipt_path is None or not Path(receipt_path).is_file():
        raise GradingRefused(
            f"priced markets exist for {slate_date} but no serving receipt is "
            f"at {receipt_path}. A grade with no provenance cannot satisfy "
            "the frozen policy's provenance gate, so none is written"
        )

    provenance = verify_receipt(
        _load_json(Path(receipt_path)), slate_date=slate_date, priced_path=priced_path
    )

    priced = pd.read_parquet(priced_path)

    if priced.empty:
        return {
            **base,
            "outcome": OUTCOME_NOT_SERVED,
            "reason": f"the priced markets for {slate_date} carry no rows",
            "counts": {},
            "grade_artifact": None,
        }

    # Checked before any outcome is opened, and on the latest price rather
    # than the earliest: a slate is out of sample only if every row in it is.
    prediction_timestamps = (
        pd.to_datetime(priced["priced_at_utc"], errors="coerce", utc=True).dropna()
        if "priced_at_utc" in priced.columns
        else pd.Series(dtype="datetime64[ns, UTC]")
    )

    refuse_on_lookahead(
        prediction_timestamp=(
            str(prediction_timestamps.max())
            if not prediction_timestamps.empty
            else provenance.get("served_at")
        ),
        slate_date=slate_date,
    )

    settlement = settlement_sources(Path(data_root), slate_date, rules)

    game_ids = {
        int(value)
        for value in pd.to_numeric(priced["game_id"], errors="coerce")
        .dropna()
        .unique()
    }

    finals: dict[int, bool] = {}
    outcomes = pd.DataFrame().set_index(
        pd.MultiIndex.from_arrays([[], []], names=["game_id", "player_id"])
    )

    if settlement["available"]:
        games = pd.read_parquet(settlement["games_path"])
        stats = pd.read_parquet(settlement["stats_path"])

        finals = final_games(games, game_ids, rules)

        mismatched = sorted(
            f"{identifier}:{date}"
            for identifier, date in game_dates(games, game_ids).items()
            if date is not None and date != slate_date
        )

        if mismatched:
            raise GradingRefused(
                f"the priced markets for {slate_date} name games the "
                f"schedule dates elsewhere: {mismatched[:10]}. Grading them "
                "would score one slate's predictions against another slate's "
                "outcomes"
            )

        outcomes = outcome_rows(stats, game_ids)

    graded = grade_rows(
        priced=priced,
        outcomes=outcomes,
        finals=finals,
        slate_date=slate_date,
        provenance=provenance,
        settlement=settlement,
        rules=rules,
        graded_at=graded_at,
    )

    grade_path = (
        Path(grades_root)
        / f"season={settlement['season']}"
        / f"date={slate_date}.parquet"
    )

    graded = carry_graded_at(graded, grade_path)

    upsert_parquet(graded, grade_path, GRADE_KEY)

    counts = {
        status: int(count)
        for status, count in graded["settlement_status"].value_counts().items()
    }

    settled = int((graded["settlement_status"] == STATUS_GRADED).sum())

    return {
        **base,
        "outcome": OUTCOME_GRADED if settled else OUTCOME_PENDING,
        "reason": (
            f"{settled} of {len(graded)} prediction(s) for {slate_date} are "
            "settled and scored"
        ),
        "counts": counts,
        "rows": int(len(graded)),
        "grade_artifact": {
            "path": str(grade_path),
            "rows": int(len(graded)),
            "sha256": sha256_file(grade_path),
        },
        "incumbent": {key: provenance[key] for key in sorted(provenance)},
        "scores": _slate_scores(graded),
        "settlement": settlement,
    }


def carry_graded_at(graded: pd.DataFrame, grade_path: Path) -> pd.DataFrame:
    """Keep the original ``graded_at`` on rows whose grade has not changed.

    ``graded_at`` says when a row's grade was determined, so a rerun that
    recomputes the same grade from the same inputs must not move it. Without
    this, the field would mean "when grading last ran", which is already on
    the run receipt and would make every stored row differ from the last run
    whether anything was decided or not.

    A row whose grade did change -- the usual case being a PENDING_SETTLEMENT
    row that has since gone final -- takes the new timestamp, because that is
    when the new grade was in fact determined.
    """
    if not grade_path.is_file():
        return graded

    stored = pd.read_parquet(grade_path)

    if stored.empty or "graded_at" not in stored.columns:
        return graded

    # The key columns are equal by construction -- they are what the rows
    # were matched on -- and setting the index moves them off the row, so
    # comparing them would read a missing field and never carry anything.
    skipped = {"graded_at", *GRADE_KEY}

    compared = [
        column
        for column in graded.columns
        if column not in skipped and column in stored.columns
    ]

    previous = stored.set_index(GRADE_KEY)

    carried = graded.copy()
    timestamps = carried["graded_at"].tolist()

    for position, (_, row) in enumerate(carried.iterrows()):
        key = tuple(row[name] for name in GRADE_KEY)
        key = key[0] if len(key) == 1 else key

        if key not in previous.index:
            continue

        before = previous.loc[key]

        if isinstance(before, pd.DataFrame):  # duplicated key in a prior write
            continue

        unchanged = all(
            _same_value(before.get(column), row[column]) for column in compared
        )

        if unchanged:
            timestamps[position] = before["graded_at"]

    carried["graded_at"] = timestamps

    return carried


def _same_value(before: Any, after: Any) -> bool:
    if pd.isna(before) and pd.isna(after):
        return True
    if pd.isna(before) or pd.isna(after):
        return False
    if isinstance(before, float) or isinstance(after, float):
        return math.isclose(float(before), float(after), rel_tol=0.0, abs_tol=1e-12)
    return bool(before == after)


def _slate_scores(graded: pd.DataFrame) -> dict[str, Any]:
    """The slate's own Brier and log loss. Reported, never gated here."""
    settled = graded.loc[graded["settlement_status"] == STATUS_GRADED]

    if settled.empty:
        return {"events": 0, "brier": None, "log_loss": None, "base_rate": None}

    return {
        "events": int(len(settled)),
        "brier": float(settled["brier_contribution"].mean()),
        "log_loss": float(settled["log_loss_contribution"].mean()),
        "base_rate": float(settled["realized_result"].mean()),
    }


# ----------------------------------------------------------------------
# catch-up
# ----------------------------------------------------------------------


def dates_needing_another_attempt(
    *, grades_root: Path, slate_date: str, days: int
) -> list[str]:
    """Earlier slates whose stored grades still carry unsettled rows.

    A day whose games had not finished when the lifecycle ran is the normal
    case, not a failure, so it has to be revisited. Idempotence is what makes
    revisiting free: the rows are replaced, so a day that settles in place
    costs nothing and a day that does not stays pending.
    """
    if days <= 0:
        return []

    target = pd.Timestamp(slate_date)
    window = {
        (target - timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(1, days + 1)
    }

    pending: set[str] = set()

    for path in sorted(Path(grades_root).rglob("date=*.parquet")):
        date = path.stem.split("=", 1)[1]

        if date not in window:
            continue

        frame = pd.read_parquet(path, columns=["settlement_status"])

        if (frame["settlement_status"] == STATUS_PENDING).any():
            pending.add(date)

    return sorted(pending)


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grade the incumbent's served production slate against realized "
            "box scores. Writes grade rows and a receipt; promotes nothing, "
            "publishes nothing and changes no serving authority."
        )
    )
    parser.add_argument(
        "--slate-date",
        required=True,
        help="The served slate to grade, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="The durable data root holding processed/ and raw/.",
    )
    parser.add_argument(
        "--serving-receipt",
        type=Path,
        default=None,
        help=(
            "The receipt ops/run_incumbent_production_serving.py wrote for "
            "this slate. Defaults to "
            f"<data-root>/{SERVING_RECEIPTS_RELATIVE}/<slate-date>.json, "
            "which is also how a caught-up earlier slate's receipt is found."
        ),
    )
    parser.add_argument(
        "--grades-root",
        type=Path,
        default=None,
        help=(
            "Where grade rows are stored. Defaults to "
            f"<data-root>/{GRADES_RELATIVE} ."
        ),
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=None,
        help=(
            "Where this run's receipt is written. Defaults to "
            f"<data-root>/{RECEIPTS_RELATIVE}/<slate-date>.json ."
        ),
    )
    parser.add_argument(
        "--catch-up-days",
        type=int,
        default=7,
        help=(
            "Also re-attempt earlier slates within this many days whose "
            "stored grades still carry unsettled rows. 0 disables."
        ),
    )
    return parser.parse_args(argv)


def render(receipt: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "INCUMBENT PRODUCTION GRADING",
        "=" * 72,
        f"outcome              : {receipt['outcome']}",
        f"slate date           : {receipt['slate_date']}",
        f"reason               : {receipt['reason']}",
    ]

    incumbent = receipt.get("incumbent") or {}

    if incumbent:
        lines += [
            f"incumbent authority  : {incumbent.get('incumbent_authority')}",
            f"incumbent version    : {incumbent.get('incumbent_version')}",
            f"incumbent fit id     : {incumbent.get('incumbent_fit_id') or 'none promoted'}",
            f"production code SHA  : {incumbent.get('production_code_sha')}",
            f"input fingerprint    : {incumbent.get('input_state_fingerprint')}",
        ]

    for status, count in sorted((receipt.get("counts") or {}).items()):
        lines.append(f"  {status:<26}: {count}")

    scores = receipt.get("scores") or {}

    if scores.get("events"):
        lines += [
            f"settled events       : {scores['events']}",
            f"brier                : {scores['brier']:.6f}",
            f"log loss             : {scores['log_loss']:.6f}",
            f"base rate            : {scores['base_rate']:.6f}",
        ]

    artifact = receipt.get("grade_artifact")

    if artifact:
        lines.append(f"grades               : {artifact['path']}")

    lines += [
        "",
        "Grading only. This run promotes nothing, publishes nothing and "
        "leaves serving authority untouched.",
    ]

    return "\n".join(lines)


def serving_receipt_path(
    *, data_root: Path, slate_date: str, override: Path | None = None
) -> Path:
    """Where this slate's serving provenance is, by convention or by argument."""
    if override is not None:
        return Path(override)

    return Path(data_root) / SERVING_RECEIPTS_RELATIVE / f"{slate_date}.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    rules = load_settlement_rules()

    grades_root = args.grades_root or (Path(args.data_root) / GRADES_RELATIVE)

    receipt = grade_slate(
        slate_date=args.slate_date,
        data_root=Path(args.data_root),
        receipt_path=serving_receipt_path(
            data_root=Path(args.data_root),
            slate_date=args.slate_date,
            override=args.serving_receipt,
        ),
        rules=rules,
        grades_root=Path(grades_root),
    )

    caught_up: list[dict[str, Any]] = []

    for date in dates_needing_another_attempt(
        grades_root=Path(grades_root),
        slate_date=args.slate_date,
        days=int(args.catch_up_days),
    ):
        try:
            caught_up.append(
                grade_slate(
                    slate_date=date,
                    data_root=Path(args.data_root),
                    # By convention only. An explicit --serving-receipt names
                    # one slate's provenance and is not provenance for another.
                    receipt_path=serving_receipt_path(
                        data_root=Path(args.data_root), slate_date=date
                    ),
                    rules=rules,
                    grades_root=Path(grades_root),
                )
            )
        except GradingRefused as error:
            # One unsettleable earlier day must not cost today's grading. The
            # refusal is recorded so it is still visible in the receipt.
            caught_up.append(
                {
                    "slate_date": date,
                    "outcome": OUTCOME_REFUSED,
                    "reason": str(error),
                }
            )

    receipt["caught_up"] = caught_up

    return receipt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    receipt_path = args.receipt_path or (
        Path(args.data_root) / RECEIPTS_RELATIVE / f"{args.slate_date}.json"
    )

    try:
        receipt = run(args)
    except GradingRefused as error:
        receipt = {
            "entry_point": "ops/grade_incumbent_production_slate.py",
            "graded_at": _utc_now(),
            "outcome": OUTCOME_REFUSED,
            "production_serving_was_affected": False,
            "promotion_authority": "NONE",
            "published": False,
            "reason": str(error),
            "slate_date": args.slate_date,
        }
        _write_receipt(receipt_path, receipt)
        print(f"GRADING REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED

    _write_receipt(receipt_path, receipt)
    print(render(receipt))

    return EXIT_OK


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(receipt, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
