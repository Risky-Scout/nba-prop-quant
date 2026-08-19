from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
ARCHIVE_ROOT = ROOT / "data/external_test/season=2026"
GRADE_ROOT = ARCHIVE_ROOT / "grades"
GRADE_INDEX = GRADE_ROOT / "grade_index.jsonl"

MONITORING_GRID = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
)

PROP_COMPONENTS = {
    "points": ("pts",),
    "rebounds": ("reb",),
    "assists": ("ast",),
    "steals": ("stl",),
    "blocks": ("blk",),
    "threes": ("fg3m",),
    "points_rebounds": ("pts", "reb"),
    "points_assists": ("pts", "ast"),
    "rebounds_assists": ("reb", "ast"),
    "points_rebounds_assists": ("pts", "reb", "ast"),
    "stocks": ("stl", "blk"),
}

SIDE_COLUMNS = (
    "preferred_side",
    "preferred_side_calibrated",
    "selected_bet_side",
    "bet_side",
    "monitor_side",
)

EDGE_COLUMNS = (
    "preferred_edge_calibrated",
    "preferred_edge",
    "selected_edge_calibrated",
    "selected_bet_edge",
    "bet_edge",
    "monitor_edge",
    "calibrated_edge_selected",
    "selected_edge",
)

EV_COLUMNS = (
    "preferred_ev_calibrated",
    "preferred_ev",
    "selected_ev_calibrated",
    "selected_bet_ev",
    "bet_model_ev",
    "monitor_ev",
    "calibrated_ev_selected",
    "selected_ev",
)

OVER_ODDS_COLUMNS = (
    "over_odds",
    "market_over_odds",
)

UNDER_ODDS_COLUMNS = (
    "under_odds",
    "market_under_odds",
)

CALIBRATED_Q_OVER_COLUMNS = (
    "q_over_calibrated",
    "q_selected",
    "selected_q_over",
    "calibrated_q_over",
    "q_over_selected",
    "model_q_over_calibrated",
)

RAW_Q_OVER_COLUMNS = (
    "q_over_raw",
    "q_over_nonpush",
    "raw_q_over",
    "model_q_over_raw",
)

MARKET_Q_OVER_COLUMNS = (
    "market_q_over",
    "market_devig_p_over",
    "market_q_over_nonpush",
    "market_devig_over",
)

CALIBRATED_P_PAIRS = (
    ("p_over_calibrated", "p_under_calibrated"),
    ("p_selected_over", "p_selected_under"),
    ("calibrated_p_over", "calibrated_p_under"),
)

RAW_P_PAIRS = (
    ("p_over_raw", "p_under_raw"),
    ("p_over", "p_under"),
    ("raw_p_over", "raw_p_under"),
)

MARKET_P_PAIRS = (
    ("market_devig_p_over", "market_devig_p_under"),
    ("market_q_over", "market_q_under"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grade one immutable NBA external-test capture using final "
            "local BDL standard box scores. Original capture files are "
            "never modified."
        )
    )

    parser.add_argument(
        "--capture",
        type=Path,
        help=(
            "Capture directory containing capture_manifest.json. "
            "Required unless --self-test is used."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic grading-unit checks without touching project data.",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    present = set(columns)

    for candidate in candidates:
        if candidate in present:
            return candidate

    return None


def conditional_nonpush(
    over: pd.Series,
    under: pd.Series,
) -> pd.Series:
    over_num = pd.to_numeric(over, errors="coerce")
    under_num = pd.to_numeric(under, errors="coerce")
    denom = over_num + under_num

    result = over_num / denom
    result = result.where(denom > 0.0)

    return result.clip(0.0, 1.0)


def resolve_probability(
    frame: pd.DataFrame,
    direct_candidates: tuple[str, ...],
    pair_candidates: tuple[tuple[str, str], ...],
) -> tuple[pd.Series, str | None]:
    direct = first_existing(
        frame.columns,
        direct_candidates,
    )

    if direct is not None:
        values = pd.to_numeric(
            frame[direct],
            errors="coerce",
        ).clip(0.0, 1.0)

        return values, direct

    present = set(frame.columns)

    for over_col, under_col in pair_candidates:
        if over_col in present and under_col in present:
            return (
                conditional_nonpush(
                    frame[over_col],
                    frame[under_col],
                ),
                f"{over_col}/({over_col}+{under_col})",
            )

    return (
        pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        ),
        None,
    )


def american_profit_units(odds: float) -> float:
    value = float(odds)

    if not np.isfinite(value) or value == 0.0:
        return math.nan

    if value > 0.0:
        return value / 100.0

    return 100.0 / abs(value)


def normalize_side(value: object) -> str | None:
    text = str(value).strip().lower()

    if text in {"over", "o"}:
        return "over"

    if text in {"under", "u"}:
        return "under"

    return None


def prop_actual(row: pd.Series, prop_type: str) -> float:
    components = PROP_COMPONENTS.get(
        str(prop_type).strip().lower()
    )

    if components is None:
        return math.nan

    values = []

    for column in components:
        value = pd.to_numeric(
            pd.Series([row.get(column)]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value):
            return math.nan

        values.append(float(value))

    return float(sum(values))


def played_indicator(row: pd.Series) -> bool:
    minutes = pd.to_numeric(
        pd.Series([row.get("minutes")]),
        errors="coerce",
    ).iloc[0]

    if pd.notna(minutes):
        return bool(float(minutes) > 0.0)

    # Fallback only if a normalized minutes field is unavailable.
    stats = [
        row.get("pts"),
        row.get("reb"),
        row.get("ast"),
        row.get("stl"),
        row.get("blk"),
        row.get("fg3m"),
    ]

    numeric = pd.to_numeric(
        pd.Series(stats),
        errors="coerce",
    )

    return bool(numeric.notna().any())


def is_final_game(row: pd.Series) -> bool:
    status = str(
        row.get("status", "")
    ).strip().lower()

    state = str(
        row.get("status_state", "")
    ).strip().lower()

    postponed_raw = row.get(
        "postponed",
        False,
    )

    postponed = (
        bool(postponed_raw)
        if pd.notna(postponed_raw)
        else False
    )

    if postponed:
        return False

    if "final" in status:
        return True

    if state in {
        "post",
        "final",
        "completed",
        "complete",
    }:
        return True

    return False


def verify_capture(capture: Path) -> None:
    verifier = ROOT / "ops/verify_external_test_capture.py"

    if not verifier.exists():
        raise SystemExit(
            f"ERROR: missing capture verifier: {verifier}"
        )

    subprocess.run(
        [
            sys.executable,
            str(verifier),
            str(capture),
        ],
        cwd=ROOT,
        check=True,
    )


def target_season(capture_date: pd.Timestamp) -> int:
    # NBA season label follows the calendar year in which the season begins.
    # 2026-10 through 2027 playoffs are season=2026.
    if capture_date.month >= 7:
        return int(capture_date.year)

    return int(capture_date.year - 1)


def source_file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def write_checksums(directory: Path) -> None:
    lines = []

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(
                f"{sha256_file(path)}  {path.relative_to(directory)}"
            )

    (
        directory
        / "checksums.sha256"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def empty_grade(
    *,
    capture: Path,
    capture_manifest: dict,
    status: str,
    now_utc: datetime,
) -> Path:
    date = str(capture_manifest["date"])
    capture_id = str(capture_manifest["capture_id"])
    grade_id = now_utc.strftime("%Y%m%dT%H%M%SZ")

    grade_dir = (
        GRADE_ROOT
        / f"date={date}"
        / f"capture_id={capture_id}"
        / grade_id
    )

    grade_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    source_manifest = capture / "capture_manifest.json"

    metadata = {
        "schema_version": 1,
        "grade_id": grade_id,
        "graded_at_utc": now_utc.isoformat(),
        "date": date,
        "capture_id": capture_id,
        "capture_mode": capture_manifest.get("mode"),
        "freeze_id": capture_manifest.get("freeze_id"),
        "status": status,
        "capture_manifest_sha256": sha256_file(source_manifest),
        "priced_markets_sha256": None,
        "outcome_source": None,
        "graded_quote_rows": 0,
        "settled_rows": 0,
        "unresolved_rows": 0,
        "monitoring_policy": {
            "auto_bet": False,
            "threshold_frozen": False,
            "monitoring_edge_grid": list(MONITORING_GRID),
        },
    }

    (
        grade_dir
        / "grade_manifest.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    shutil.copy2(
        Path(__file__).resolve(),
        grade_dir / "grade_external_test_capture.py",
    )

    write_checksums(grade_dir)

    GRADE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with GRADE_INDEX.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "grade_id": grade_id,
                    "graded_at_utc": now_utc.isoformat(),
                    "date": date,
                    "capture_id": capture_id,
                    "capture_mode": capture_manifest.get("mode"),
                    "freeze_id": capture_manifest.get("freeze_id"),
                    "status": status,
                    "grade_manifest": str(
                        (
                            grade_dir
                            / "grade_manifest.json"
                        ).relative_to(ROOT)
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )

    return grade_dir


def self_test() -> None:
    stat = pd.Series(
        {
            "minutes": 31.5,
            "pts": 24,
            "reb": 8,
            "ast": 7,
            "stl": 2,
            "blk": 1,
            "fg3m": 4,
        }
    )

    expected = {
        "points": 24.0,
        "rebounds": 8.0,
        "assists": 7.0,
        "steals": 2.0,
        "blocks": 1.0,
        "threes": 4.0,
        "points_rebounds": 32.0,
        "points_assists": 31.0,
        "rebounds_assists": 15.0,
        "points_rebounds_assists": 39.0,
        "stocks": 3.0,
    }

    for prop_type, actual in expected.items():
        observed = prop_actual(stat, prop_type)
        assert observed == actual, (
            prop_type,
            observed,
            actual,
        )

    assert played_indicator(stat)
    assert american_profit_units(-110) == 100.0 / 110.0
    assert american_profit_units(150) == 1.5

    probs, source = resolve_probability(
        pd.DataFrame(
            {
                "p_over_calibrated": [0.45],
                "p_under_calibrated": [0.50],
            }
        ),
        CALIBRATED_Q_OVER_COLUMNS,
        CALIBRATED_P_PAIRS,
    )

    assert source is not None
    assert abs(float(probs.iloc[0]) - (0.45 / 0.95)) < 1e-12

    print("PASS: external-test grading deterministic self-test.")


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    if args.capture is None:
        raise SystemExit(
            "ERROR: --capture is required unless --self-test is used."
        )

    capture = args.capture

    if not capture.is_absolute():
        capture = ROOT / capture

    capture = capture.resolve()

    capture_manifest_path = capture / "capture_manifest.json"

    if not capture_manifest_path.exists():
        raise SystemExit(
            f"ERROR: missing capture manifest: {capture_manifest_path}"
        )

    verify_capture(capture)

    capture_manifest = load_json(capture_manifest_path)
    date_text = str(capture_manifest["date"])
    capture_date = pd.Timestamp(date_text).normalize()
    capture_id = str(capture_manifest["capture_id"])
    now_utc = datetime.now(timezone.utc)

    priced_path = capture / "priced_markets.parquet"

    if not priced_path.exists():
        grade_dir = empty_grade(
            capture=capture,
            capture_manifest=capture_manifest,
            status="no_priced_markets",
            now_utc=now_utc,
        )

        print(
            "PASS: capture has no priced markets; "
            "immutable no-market grade archived."
        )

        print(
            f"Grade archive: {grade_dir}"
        )

        return

    priced = pd.read_parquet(priced_path)

    required_priced = {
        "game_id",
        "player_id",
        "prop_type",
        "line_value",
    }

    missing_priced = required_priced - set(priced.columns)

    if missing_priced:
        raise SystemExit(
            "ERROR: priced market schema is missing required settlement "
            f"columns: {sorted(missing_priced)}. "
            f"Available columns: {list(priced.columns)}"
        )

    season = target_season(capture_date)

    stats_path = (
        ROOT
        / "data/raw/seasons"
        / f"season={season}"
        / "stats.parquet"
    )

    games_path = (
        ROOT
        / "data/raw/seasons"
        / f"season={season}"
        / "games.parquet"
    )

    if not stats_path.exists():
        raise SystemExit(
            f"ERROR: missing final-stat source: {stats_path}. "
            "Refresh completed history before grading."
        )

    if not games_path.exists():
        raise SystemExit(
            f"ERROR: missing game-status source: {games_path}. "
            "Refresh completed history before grading."
        )

    stats = pd.read_parquet(stats_path)
    games = pd.read_parquet(games_path)

    if stats.empty:
        raise SystemExit(
            f"ERROR: {stats_path} is empty; outcomes are not available yet."
        )

    if "date" not in stats.columns:
        raise SystemExit(
            f"ERROR: non-empty stats source lacks date: {stats_path}"
        )

    if "id" not in games.columns:
        raise SystemExit(
            f"ERROR: games source lacks id: {games_path}"
        )

    game_ids = set(
        pd.to_numeric(
            priced["game_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    games_for_capture = games[
        pd.to_numeric(
            games["id"],
            errors="coerce",
        )
        .isin(game_ids)
    ].copy()

    found_game_ids = set(
        pd.to_numeric(
            games_for_capture["id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    missing_games = sorted(
        game_ids - found_game_ids
    )

    if missing_games:
        raise SystemExit(
            "ERROR: final-status source does not contain captured game_id(s): "
            f"{missing_games[:20]}"
        )

    not_final = []

    for _, game in games_for_capture.iterrows():
        if not is_final_game(game):
            not_final.append(
                {
                    "game_id": int(game["id"]),
                    "status": game.get("status"),
                    "status_state": game.get("status_state"),
                    "postponed": game.get("postponed"),
                }
            )

    if not_final:
        raise SystemExit(
            "ERROR: at least one captured game is not confirmed final. "
            "Do not grade yet. Examples: "
            + json.dumps(not_final[:10], default=str)
        )

    stat_date = pd.to_datetime(
        stats["date"],
        errors="coerce",
    ).dt.normalize()

    outcomes = stats[
        stat_date.eq(capture_date)
        & pd.to_numeric(
            stats["game_id"],
            errors="coerce",
        ).isin(game_ids)
    ].copy()

    needed_stats = {
        "game_id",
        "player_id",
        "minutes",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "fg3m",
    }

    missing_outcome_columns = needed_stats - set(outcomes.columns)

    if missing_outcome_columns:
        raise SystemExit(
            "ERROR: outcome source missing required columns: "
            f"{sorted(missing_outcome_columns)}"
        )

    duplicate_keys = (
        outcomes.groupby(
            [
                "game_id",
                "player_id",
            ],
            dropna=False,
        )
        .size()
        .loc[
            lambda series:
            series > 1
        ]
    )

    if not duplicate_keys.empty:
        raise SystemExit(
            "ERROR: duplicate game/player outcome rows detected; "
            "settlement aborted rather than aggregating ambiguous records. "
            f"Examples: {duplicate_keys.head(10).to_dict()}"
        )

    outcome_columns = [
        "game_id",
        "player_id",
        "minutes",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "fg3m",
    ]

    for optional in [
        "date",
        "player_first_name",
        "player_last_name",
        "team_id",
        "stat_id",
    ]:
        if optional in outcomes.columns:
            outcome_columns.append(optional)

    outcomes_small = outcomes[
        outcome_columns
    ].copy()

    renamed = {
        column: f"actual_{column}"
        for column in outcome_columns
        if column not in {
            "game_id",
            "player_id",
        }
    }

    outcomes_join = outcomes_small.rename(
        columns=renamed
    )

    graded = priced.merge(
        outcomes_join,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
        validate="many_to_one",
    )

    def played_from_graded(row: pd.Series) -> bool:
        proxy = pd.Series(
            {
                "minutes": row.get("actual_minutes"),
                "pts": row.get("actual_pts"),
                "reb": row.get("actual_reb"),
                "ast": row.get("actual_ast"),
                "stl": row.get("actual_stl"),
                "blk": row.get("actual_blk"),
                "fg3m": row.get("actual_fg3m"),
            }
        )

        return played_indicator(proxy)

    graded["outcome_row_found"] = graded[
        "actual_minutes"
    ].notna()

    # Some feeds may retain DNP rows with NaN minutes but stat identifiers.
    # Treat any joined player row as found even if minutes is NaN.
    joined_stat_columns = [
        column
        for column in [
            "actual_pts",
            "actual_reb",
            "actual_ast",
            "actual_stl",
            "actual_blk",
            "actual_fg3m",
        ]
        if column in graded.columns
    ]

    if joined_stat_columns:
        graded["outcome_row_found"] = (
            graded["outcome_row_found"]
            | graded[
                joined_stat_columns
            ].notna().any(axis=1)
        )

    graded["played"] = graded.apply(
        played_from_graded,
        axis=1,
    )

    def compute_actual(row: pd.Series) -> float:
        proxy = pd.Series(
            {
                "pts": row.get("actual_pts"),
                "reb": row.get("actual_reb"),
                "ast": row.get("actual_ast"),
                "stl": row.get("actual_stl"),
                "blk": row.get("actual_blk"),
                "fg3m": row.get("actual_fg3m"),
            }
        )

        return prop_actual(
            proxy,
            str(row["prop_type"]),
        )

    graded["actual"] = graded.apply(
        compute_actual,
        axis=1,
    )

    line = pd.to_numeric(
        graded["line_value"],
        errors="coerce",
    )

    valid_prop = graded[
        "prop_type"
    ].astype(str).str.lower().isin(
        PROP_COMPONENTS
    )

    settled = (
        graded["outcome_row_found"]
        & graded["played"]
        & valid_prop
        & graded["actual"].notna()
        & line.notna()
    )

    graded["settlement_status"] = "unresolved"

    graded.loc[
        ~graded["outcome_row_found"],
        "settlement_status",
    ] = "player_missing_final_game_void_candidate"

    graded.loc[
        graded["outcome_row_found"]
        & ~graded["played"],
        "settlement_status",
    ] = "dnp_void_candidate"

    graded.loc[
        ~valid_prop,
        "settlement_status",
    ] = "unsupported_prop_type"

    graded.loc[
        settled
        & graded["actual"].gt(line),
        "settlement_status",
    ] = "over"

    graded.loc[
        settled
        & graded["actual"].lt(line),
        "settlement_status",
    ] = "under"

    graded.loc[
        settled
        & graded["actual"].eq(line),
        "settlement_status",
    ] = "push"

    graded["actual_push"] = graded[
        "settlement_status"
    ].eq("push")

    graded["actual_over"] = np.where(
        graded[
            "settlement_status"
        ].eq("over"),
        1.0,
        np.where(
            graded[
                "settlement_status"
            ].eq("under"),
            0.0,
            np.nan,
        ),
    )

    over_odds_col = first_existing(
        graded.columns,
        OVER_ODDS_COLUMNS,
    )

    under_odds_col = first_existing(
        graded.columns,
        UNDER_ODDS_COLUMNS,
    )

    side_col = first_existing(
        graded.columns,
        SIDE_COLUMNS,
    )

    edge_col = first_existing(
        graded.columns,
        EDGE_COLUMNS,
    )

    ev_col = first_existing(
        graded.columns,
        EV_COLUMNS,
    )

    if side_col is not None:
        graded["monitor_side"] = graded[
            side_col
        ].map(normalize_side)

    else:
        graded["monitor_side"] = None

    if edge_col is not None:
        graded["monitor_edge"] = pd.to_numeric(
            graded[edge_col],
            errors="coerce",
        )

    else:
        graded["monitor_edge"] = np.nan

    if ev_col is not None:
        graded["monitor_ev"] = pd.to_numeric(
            graded[ev_col],
            errors="coerce",
        )

    else:
        graded["monitor_ev"] = np.nan

    calibrated_q, calibrated_source = resolve_probability(
        graded,
        CALIBRATED_Q_OVER_COLUMNS,
        CALIBRATED_P_PAIRS,
    )

    raw_q, raw_source = resolve_probability(
        graded,
        RAW_Q_OVER_COLUMNS,
        RAW_P_PAIRS,
    )

    market_q, market_source = resolve_probability(
        graded,
        MARKET_Q_OVER_COLUMNS,
        MARKET_P_PAIRS,
    )

    graded["grade_q_over_calibrated"] = calibrated_q
    graded["grade_q_over_raw"] = raw_q
    graded["grade_market_q_over"] = market_q

    nonpush = graded[
        "settlement_status"
    ].isin(
        [
            "over",
            "under",
        ]
    )

    eps = 1e-12

    for label, probability in [
        ("calibrated", calibrated_q),
        ("raw", raw_q),
        ("market", market_q),
    ]:
        y = graded["actual_over"]

        brier = (
            probability - y
        ) ** 2

        logloss = -(
            y * np.log(
                probability.clip(eps, 1.0 - eps)
            )
            + (
                1.0 - y
            )
            * np.log(
                (
                    1.0 - probability
                ).clip(eps, 1.0 - eps)
            )
        )

        graded[
            f"brier_{label}"
        ] = brier.where(nonpush)

        graded[
            f"logloss_{label}"
        ] = logloss.where(nonpush)

    graded["monitor_profit_units"] = np.nan

    for index, row in graded.iterrows():
        side = row.get("monitor_side")
        status = row.get("settlement_status")

        if side not in {
            "over",
            "under",
        }:
            continue

        if status in {
            "push",
        }:
            graded.at[
                index,
                "monitor_profit_units",
            ] = 0.0

            continue

        if status not in {
            "over",
            "under",
        }:
            continue

        won = side == status

        if side == "over":
            odds_value = (
                row.get(over_odds_col)
                if over_odds_col is not None
                else np.nan
            )

        else:
            odds_value = (
                row.get(under_odds_col)
                if under_odds_col is not None
                else np.nan
            )

        odds_numeric = pd.to_numeric(
            pd.Series([odds_value]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(odds_numeric):
            continue

        graded.at[
            index,
            "monitor_profit_units",
        ] = (
            american_profit_units(
                float(odds_numeric)
            )
            if won
            else -1.0
        )

    for threshold in MONITORING_GRID:
        label = (
            str(threshold)
            .replace(
                ".",
                "_",
            )
        )

        graded[
            f"monitor_edge_ge_{label}"
        ] = (
            pd.to_numeric(
                graded["monitor_edge"],
                errors="coerce",
            )
            .ge(threshold)
            & graded[
                "monitor_side"
            ].isin(
                [
                    "over",
                    "under",
                ]
            )
        )

    contract_keys = [
        "game_id",
        "player_id",
        "prop_type",
        "line_value",
    ]

    contract_rows = []

    for _, group in graded.groupby(
        contract_keys,
        dropna=False,
        sort=False,
    ):
        row = {
            key: group.iloc[0][key]
            for key in contract_keys
        }

        for column in [
            "actual",
            "actual_push",
            "actual_over",
            "settlement_status",
            "grade_q_over_calibrated",
            "grade_q_over_raw",
        ]:
            row[column] = group.iloc[0][column]

        row["grade_market_q_over"] = pd.to_numeric(
            group[
                "grade_market_q_over"
            ],
            errors="coerce",
        ).mean()

        row["vendors"] = int(
            group[
                "vendor"
            ].nunique()
            if "vendor" in group.columns
            else len(group)
        )

        contract_rows.append(row)

    contracts = pd.DataFrame(
        contract_rows
    )

    monitoring_rows = []

    for threshold in MONITORING_GRID:
        eligible_mask = (
            graded[
                "monitor_side"
            ].isin(
                [
                    "over",
                    "under",
                ]
            )
            & pd.to_numeric(
                graded[
                    "monitor_edge"
                ],
                errors="coerce",
            ).ge(threshold)
            & graded[
                "monitor_profit_units"
            ].notna()
        )

        if ev_col is not None:
            eligible_mask = (
                eligible_mask
                & pd.to_numeric(
                    graded[
                        "monitor_ev"
                    ],
                    errors="coerce",
                ).gt(0.0)
            )

        eligible = graded[
            eligible_mask
        ].copy()

        for scope in [
            "all_quotes",
            "best_vendor_event",
            "best_event",
        ]:
            scoped = eligible.copy()

            if scope == "best_vendor_event" and not scoped.empty:
                subset = [
                    "game_id",
                    "player_id",
                    "prop_type",
                ]

                if "vendor" in scoped.columns:
                    subset.append(
                        "vendor"
                    )

                sort_cols = [
                    column
                    for column in [
                        "monitor_ev",
                        "monitor_edge",
                    ]
                    if column in scoped.columns
                ]

                if sort_cols:
                    scoped = scoped.sort_values(
                        sort_cols,
                        ascending=False,
                    )

                scoped = scoped.drop_duplicates(
                    subset=subset,
                    keep="first",
                )

            elif scope == "best_event" and not scoped.empty:
                sort_cols = [
                    column
                    for column in [
                        "monitor_ev",
                        "monitor_edge",
                    ]
                    if column in scoped.columns
                ]

                if sort_cols:
                    scoped = scoped.sort_values(
                        sort_cols,
                        ascending=False,
                    )

                scoped = scoped.drop_duplicates(
                    subset=[
                        "game_id",
                        "player_id",
                        "prop_type",
                    ],
                    keep="first",
                )

            profits = pd.to_numeric(
                scoped[
                    "monitor_profit_units"
                ],
                errors="coerce",
            ).dropna()

            monitoring_rows.append(
                {
                    "threshold": threshold,
                    "scope": scope,
                    "eligible_rows": int(len(scoped)),
                    "games": int(
                        scoped[
                            "game_id"
                        ].nunique()
                        if not scoped.empty
                        else 0
                    ),
                    "profit_units": float(
                        profits.sum()
                    )
                    if len(profits)
                    else 0.0,
                    "roi": float(
                        profits.mean()
                    )
                    if len(profits)
                    else np.nan,
                    "note": (
                        "prospective monitoring only; "
                        "no betting threshold is frozen"
                    ),
                }
            )

    monitoring = pd.DataFrame(
        monitoring_rows
    )

    grade_id = now_utc.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    grade_dir = (
        GRADE_ROOT
        / f"date={date_text}"
        / f"capture_id={capture_id}"
        / grade_id
    )

    grade_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    outcomes_small.to_parquet(
        grade_dir
        / "outcome_snapshot.parquet",
        index=False,
    )

    graded.to_parquet(
        grade_dir
        / "graded_quotes.parquet",
        index=False,
    )

    contracts.to_parquet(
        grade_dir
        / "graded_contracts.parquet",
        index=False,
    )

    monitoring.to_csv(
        grade_dir
        / "monitoring_summary.csv",
        index=False,
    )

    source_records = {
        "stats": source_file_record(
            stats_path
        ),
        "games": source_file_record(
            games_path
        ),
    }

    metadata = {
        "schema_version": 1,
        "grade_id": grade_id,
        "graded_at_utc": now_utc.isoformat(),
        "date": date_text,
        "season": season,
        "capture_id": capture_id,
        "capture_mode": capture_manifest.get("mode"),
        "freeze_id": capture_manifest.get("freeze_id"),
        "status": "complete",
        "capture_manifest_sha256": sha256_file(
            capture_manifest_path
        ),
        "priced_markets_sha256": sha256_file(
            priced_path
        ),
        "outcome_source": source_records,
        "graded_quote_rows": int(
            len(graded)
        ),
        "graded_contract_rows": int(
            len(contracts)
        ),
        "settled_rows": int(
            graded[
                "settlement_status"
            ].isin(
                [
                    "over",
                    "under",
                    "push",
                ]
            ).sum()
        ),
        "unresolved_rows": int(
            (
                ~graded[
                    "settlement_status"
                ].isin(
                    [
                        "over",
                        "under",
                        "push",
                    ]
                )
            ).sum()
        ),
        "settlement_policy": {
            "played_definition": (
                "normalized minutes > 0; fallback to observed stat row only "
                "when minutes is unavailable"
            ),
            "dnp_policy": (
                "DNP/player-missing is a void candidate and excluded from "
                "ROI/proper scoring; sportsbook-specific settlement rules "
                "are not assumed"
            ),
            "push_policy": (
                "actual == line is push; profit 0; excluded from binary "
                "Brier/logloss"
            ),
        },
        "probability_resolution": {
            "calibrated_q_over_source": calibrated_source,
            "raw_q_over_source": raw_source,
            "market_q_over_source": market_source,
        },
        "monitoring_resolution": {
            "side_source": side_col,
            "edge_source": edge_col,
            "ev_source": ev_col,
            "over_odds_source": over_odds_col,
            "under_odds_source": under_odds_col,
        },
        "monitoring_policy": {
            "auto_bet": False,
            "threshold_frozen": False,
            "monitoring_edge_grid": list(
                MONITORING_GRID
            ),
        },
        "original_capture_mutated": False,
    }

    (
        grade_dir
        / "grade_manifest.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    shutil.copy2(
        Path(__file__).resolve(),
        grade_dir
        / "grade_external_test_capture.py",
    )

    write_checksums(
        grade_dir
    )

    GRADE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with GRADE_INDEX.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "grade_id": grade_id,
                    "graded_at_utc": now_utc.isoformat(),
                    "date": date_text,
                    "capture_id": capture_id,
                    "capture_mode": capture_manifest.get("mode"),
                    "freeze_id": capture_manifest.get("freeze_id"),
                    "status": "complete",
                    "grade_manifest": str(
                        (
                            grade_dir
                            / "grade_manifest.json"
                        ).relative_to(ROOT)
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )

    print("=" * 118)
    print("EXTERNAL-TEST CAPTURE GRADED")
    print("=" * 118)
    print(f"Date:             {date_text}")
    print(f"Capture ID:       {capture_id}")
    print(f"Grade ID:         {grade_id}")
    print(f"Freeze ID:        {capture_manifest.get('freeze_id')}")
    print(f"Quote rows:       {len(graded):,}")
    print(f"Contract rows:    {len(contracts):,}")
    print(
        "Settled rows:     "
        f"{metadata['settled_rows']:,}"
    )
    print(
        "Unresolved rows:  "
        f"{metadata['unresolved_rows']:,}"
    )
    print(
        "Calibrated q src: "
        f"{calibrated_source}"
    )
    print(
        "Monitor side src: "
        f"{side_col}"
    )
    print(
        "Monitor edge src: "
        f"{edge_col}"
    )
    print()
    print(
        "No original capture file was modified. "
        "No automatic betting threshold was selected."
    )
    print(
        f"Grade archive: {grade_dir}"
    )


if __name__ == "__main__":
    main()
