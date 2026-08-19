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
PRICED_SOURCE = (
    ROOT / "data/processed/market_backtest/backtest_priced_2025.parquet"
)
STATS_SOURCE = (
    ROOT / "data/raw/seasons/season=2025/stats.parquet"
)
GAMES_SOURCE = (
    ROOT / "data/raw/seasons/season=2025/games.parquet"
)
OUT_ROOT = ROOT / "data/validation/grading_replay_2025"

SUPPORTED = {
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "threes",
    "points_rebounds",
    "points_assists",
    "rebounds_assists",
    "points_rebounds_assists",
    "stocks",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_checksums(directory: Path) -> None:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(
                f"{sha256_file(path)}  {path.relative_to(directory)}"
            )
    (directory / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def american_profit(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def expected_profit(row: pd.Series) -> float:
    actual = float(row["actual"])
    line = float(row["line_value"])
    side = str(row.get("bet_side", "")).strip().lower()

    if actual == line:
        return 0.0

    if side == "over":
        won = actual > line
        odds = float(row["over_odds"])
    elif side == "under":
        won = actual < line
        odds = float(row["under_odds"])
    else:
        return math.nan

    return american_profit(odds) if won else -1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a real 2025 development-market date through the new "
            "grader inside an isolated validation root."
        )
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--keep-shadow", action="store_true")
    args = parser.parse_args()

    grader = ROOT / "ops/grade_external_test_capture.py"
    verifier = ROOT / "ops/verify_external_test_capture.py"
    grade_verifier = ROOT / "ops/verify_external_test_grade.py"

    for path in (
        grader,
        verifier,
        grade_verifier,
        PRICED_SOURCE,
        STATS_SOURCE,
        GAMES_SOURCE,
    ):
        if not path.exists():
            raise SystemExit(f"ERROR: required file missing: {path}")

    priced = pd.read_parquet(PRICED_SOURCE)
    stats = pd.read_parquet(STATS_SOURCE)
    games = pd.read_parquet(GAMES_SOURCE)

    required = {
        "game_id",
        "player_id",
        "prop_type",
        "line_value",
        "actual",
        "over_odds",
        "under_odds",
        "bet_side",
    }
    missing = required - set(priced.columns)
    if missing:
        raise SystemExit(
            "ERROR: 2025 priced backtest is missing replay fields: "
            f"{sorted(missing)}"
        )

    game_dates = games[["id", "date"]].copy()
    game_dates["id"] = pd.to_numeric(game_dates["id"], errors="coerce")
    game_dates["date"] = pd.to_datetime(
        game_dates["date"],
        errors="coerce",
    ).dt.normalize()
    game_date_map = (
        game_dates.dropna().set_index("id")["date"].to_dict()
    )

    replay = priced.copy()
    replay["game_id"] = pd.to_numeric(
        replay["game_id"],
        errors="coerce",
    )

    if "date" in replay.columns:
        replay_date = pd.to_datetime(
            replay["date"],
            errors="coerce",
        ).dt.normalize()
    else:
        replay_date = pd.Series(
            pd.NaT,
            index=replay.index,
            dtype="datetime64[ns]",
        )

    missing_date_mask = replay_date.isna()
    replay_date.loc[missing_date_mask] = replay.loc[
        missing_date_mask,
        "game_id",
    ].map(game_date_map)
    replay["_replay_date"] = replay_date

    replay = replay[
        replay["prop_type"].astype(str).str.lower().isin(SUPPORTED)
        & replay["actual"].notna()
        & replay["line_value"].notna()
        & replay["_replay_date"].notna()
    ].copy()

    if replay.empty:
        raise SystemExit("ERROR: no eligible real 2025 replay rows.")

    if args.date is not None:
        chosen_date = pd.Timestamp(args.date).normalize()
        day = replay[
            replay["_replay_date"].eq(chosen_date)
        ].copy()
        if day.empty:
            raise SystemExit(
                f"ERROR: no eligible replay rows on {args.date}."
            )
    else:
        ranking = (
            replay.groupby("_replay_date")
            .agg(
                rows=("prop_type", "size"),
                prop_types=("prop_type", "nunique"),
                games=("game_id", "nunique"),
            )
            .sort_values(
                ["prop_types", "rows", "games"],
                ascending=False,
            )
        )
        chosen_date = pd.Timestamp(ranking.index[0]).normalize()
        day = replay[
            replay["_replay_date"].eq(chosen_date)
        ].copy()

    if len(day) > args.max_rows:
        groups = list(day.groupby("prop_type", sort=True))
        base = max(args.max_rows // max(len(groups), 1), 1)
        pieces = []
        remaining = args.max_rows

        for _, group in groups:
            take = min(len(group), base, remaining)
            pieces.append(
                group.sort_values(
                    ["game_id", "player_id", "line_value"]
                ).head(take)
            )
            remaining -= take

        if remaining > 0:
            used = pd.concat(pieces).index
            extra = (
                day.drop(index=used)
                .sort_values(
                    ["game_id", "player_id", "prop_type", "line_value"]
                )
                .head(remaining)
            )
            pieces.append(extra)

        day = pd.concat(pieces).head(args.max_rows).copy()

    day = day.reset_index(drop=True)
    day["replay_row_id"] = np.arange(len(day), dtype=int)
    day["external_test_record"] = False
    day["betting_threshold_policy"] = (
        "2025_development_replay_no_threshold"
    )

    date_text = chosen_date.strftime("%Y-%m-%d")
    selected_game_ids = set(
        day["game_id"].dropna().astype(int).unique()
    )

    stats_dates = pd.to_datetime(
        stats["date"],
        errors="coerce",
    ).dt.normalize()

    stats_subset = stats[
        stats_dates.eq(chosen_date)
        & pd.to_numeric(
            stats["game_id"],
            errors="coerce",
        ).isin(selected_game_ids)
    ].copy()

    games_subset = games[
        pd.to_numeric(
            games["id"],
            errors="coerce",
        ).isin(selected_game_ids)
    ].copy()

    if stats_subset.empty:
        raise SystemExit(
            f"ERROR: no raw 2025 final stats for replay date {date_text}."
        )
    if games_subset.empty:
        raise SystemExit(
            f"ERROR: no raw 2025 games for replay date {date_text}."
        )

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUT_ROOT / f"date={date_text}" / run_id
    shadow = run_dir / "shadow_root"
    shadow_ops = shadow / "ops"
    shadow_ops.mkdir(parents=True, exist_ok=False)

    for source in (grader, verifier, grade_verifier):
        shutil.copy2(source, shadow_ops / source.name)

    season_dir = shadow / "data/raw/seasons/season=2025"
    season_dir.mkdir(parents=True, exist_ok=True)
    stats_subset.to_parquet(
        season_dir / "stats.parquet",
        index=False,
    )
    games_subset.to_parquet(
        season_dir / "games.parquet",
        index=False,
    )

    capture_id = f"real_2025_replay_{date_text}"
    capture = shadow / "replay_capture"
    capture.mkdir(parents=True, exist_ok=False)

    day_for_capture = day.drop(
        columns=["_replay_date"],
        errors="ignore",
    )
    day_for_capture.to_parquet(
        capture / "priced_markets.parquet",
        index=False,
    )

    capture_manifest = {
        "schema_version": 1,
        "capture_id": capture_id,
        "captured_at_utc": now.isoformat(),
        "date": date_text,
        "mode": "engineering",
        "tag": "real_2025_development_replay",
        "status": "complete",
        "pricing_status": "historical_replay",
        "freeze_id": "2025_development_replay_not_external_test",
        "freeze_stage": "validation_replay",
        "operational_policy": {
            "auto_betting_threshold": None,
            "external_test_result_mutation": False,
        },
    }
    (capture / "capture_manifest.json").write_text(
        json.dumps(capture_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_checksums(capture)

    capture_hash_before = sha256_file(
        capture / "priced_markets.parquet"
    )

    subprocess.run(
        [
            sys.executable,
            "ops/grade_external_test_capture.py",
            "--capture",
            str(capture),
        ],
        cwd=shadow,
        check=True,
    )

    grade_base = (
        shadow
        / "data/external_test/season=2026/grades"
        / f"date={date_text}"
        / f"capture_id={capture_id}"
    )
    grade_dirs = sorted(p for p in grade_base.iterdir() if p.is_dir())
    if not grade_dirs:
        raise RuntimeError("Real replay produced no grade directory.")
    grade_dir = grade_dirs[-1]

    subprocess.run(
        [
            sys.executable,
            "ops/verify_external_test_grade.py",
            str(grade_dir),
        ],
        cwd=shadow,
        check=True,
    )

    graded = pd.read_parquet(
        grade_dir / "graded_quotes.parquet"
    )

    if len(graded) != len(day):
        raise AssertionError(
            f"Replay row count changed: source={len(day)}, graded={len(graded)}"
        )

    merged = day[
        [
            "replay_row_id",
            "actual",
            "line_value",
            "bet_side",
            "over_odds",
            "under_odds",
        ]
    ].merge(
        graded[
            [
                "replay_row_id",
                "actual",
                "settlement_status",
                "monitor_profit_units",
            ]
        ].rename(columns={"actual": "graded_actual"}),
        on="replay_row_id",
        how="inner",
        validate="one_to_one",
    )

    actual_diff = (
        pd.to_numeric(merged["actual"], errors="coerce")
        - pd.to_numeric(merged["graded_actual"], errors="coerce")
    ).abs()
    max_actual_diff = float(
        actual_diff.max() if len(actual_diff) else 0.0
    )

    if max_actual_diff > 1e-12:
        raise AssertionError(
            f"Real replay actual mismatch; max abs diff={max_actual_diff}"
        )

    expected_status = np.where(
        merged["actual"] > merged["line_value"],
        "over",
        np.where(
            merged["actual"] < merged["line_value"],
            "under",
            "push",
        ),
    )
    status_match = (
        pd.Series(expected_status, index=merged.index)
        == merged["settlement_status"]
    )
    if not bool(status_match.all()):
        bad = merged.loc[~status_match].head(10)
        raise AssertionError(
            "Real replay settlement mismatch:\n"
            + bad.to_string(index=False)
        )

    expected_pnl = merged.apply(expected_profit, axis=1)
    observed_pnl = pd.to_numeric(
        merged["monitor_profit_units"],
        errors="coerce",
    )
    pnl_mask = expected_pnl.notna() & observed_pnl.notna()

    max_pnl_diff = None
    if pnl_mask.any():
        max_pnl_diff = float(
            (
                expected_pnl[pnl_mask]
                - observed_pnl[pnl_mask]
            ).abs().max()
        )
        if max_pnl_diff > 1e-12:
            raise AssertionError(
                "Real replay monitored profit mismatch; "
                f"max diff={max_pnl_diff}"
            )

    capture_hash_after = sha256_file(
        capture / "priced_markets.parquet"
    )
    if capture_hash_before != capture_hash_after:
        raise AssertionError(
            "Replay capture priced file mutated during grading."
        )

    direct_brier_raw = None
    grader_brier_raw = None

    if "q_over_nonpush" in day.columns:
        nonpush = day["actual"].ne(day["line_value"])
        y = day.loc[nonpush, "actual"].gt(
            day.loc[nonpush, "line_value"]
        ).astype(float)
        p = pd.to_numeric(
            day.loc[nonpush, "q_over_nonpush"],
            errors="coerce",
        )
        mask = p.notna()
        if mask.any():
            direct_brier_raw = float(
                ((p[mask] - y[mask]) ** 2).mean()
            )
            grader_brier_raw = float(
                pd.to_numeric(
                    graded["brier_raw"],
                    errors="coerce",
                ).mean()
            )
            if abs(
                direct_brier_raw - grader_brier_raw
            ) > 1e-12:
                raise AssertionError(
                    "Real replay raw Brier mismatch: "
                    f"direct={direct_brier_raw}, grader={grader_brier_raw}"
                )

    report = {
        "run_id": run_id,
        "status": "PASS",
        "purpose": (
            "Real 2025 development-data realism/schema replay. "
            "Not a new external performance test."
        ),
        "date": date_text,
        "rows": int(len(day)),
        "games": int(day["game_id"].nunique()),
        "players": int(day["player_id"].nunique()),
        "prop_types": sorted(
            day["prop_type"].astype(str).unique().tolist()
        ),
        "vendors": int(
            day["vendor"].nunique()
            if "vendor" in day.columns
            else 0
        ),
        "settlement_counts": {
            str(k): int(v)
            for k, v in graded[
                "settlement_status"
            ].value_counts(
                dropna=False
            ).to_dict().items()
        },
        "max_abs_actual_difference": max_actual_diff,
        "max_abs_profit_difference": max_pnl_diff,
        "raw_brier_direct": direct_brier_raw,
        "raw_brier_grader": grader_brier_raw,
        "capture_immutability_check": True,
        "grade_checksum_verification": True,
        "real_external_test_ledger_touched": False,
        "source_priced_sha256": sha256_file(PRICED_SOURCE),
        "source_stats_sha256": sha256_file(STATS_SOURCE),
        "source_games_sha256": sha256_file(GAMES_SOURCE),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "real_2025_grading_replay_report.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    if not args.keep_shadow:
        shutil.rmtree(shadow)

    print("=" * 118)
    print("REAL 2025 GRADING REPLAY")
    print("=" * 118)
    print(f"Replay date:             {date_text}")
    print(f"Rows:                    {len(day):,}")
    print(f"Games:                   {day['game_id'].nunique():,}")
    print(f"Prop types:              {day['prop_type'].nunique():,}")
    print("Actual matching:          PASS")
    print("Settlement matching:      PASS")
    print("Profit matching:          PASS")
    if direct_brier_raw is not None:
        print("Raw Brier reconciliation: PASS")
    print("Capture immutability:     PASS")
    print("Grade checksums:          PASS")
    print("External-test ledger:     NOT TOUCHED")
    print()
    print(
        "PASS: real 2025 replay confirms grading realism/schema "
        "compatibility on development data only."
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
