from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUT_ROOT = ROOT / "data/validation/grading_synthetic"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled end-to-end integration test for the immutable grader. "
            "Runs in an isolated shadow root and never writes to the real "
            "external-test performance ledger."
        )
    )
    parser.add_argument("--keep-shadow", action="store_true")
    args = parser.parse_args()

    grader = ROOT / "ops/grade_external_test_capture.py"
    verifier = ROOT / "ops/verify_external_test_capture.py"
    grade_verifier = ROOT / "ops/verify_external_test_grade.py"

    for path in (grader, verifier, grade_verifier):
        if not path.exists():
            raise SystemExit(f"ERROR: missing required ops file: {path}")

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUT_ROOT / run_id
    shadow = run_dir / "shadow_root"
    shadow_ops = shadow / "ops"
    shadow_ops.mkdir(parents=True, exist_ok=False)

    for source in (grader, verifier, grade_verifier):
        shutil.copy2(source, shadow_ops / source.name)

    date_text = "2026-10-20"
    game_id = 990001
    player_id = 101
    dnp_player_id = 102
    missing_player_id = 999

    season_dir = shadow / "data/raw/seasons/season=2026"
    season_dir.mkdir(parents=True, exist_ok=True)

    games = pd.DataFrame(
        [
            {
                "id": game_id,
                "date": pd.Timestamp(date_text),
                "season": 2026,
                "status": "Final",
                "status_state": "post",
                "postponed": False,
                "home_team_id": 1,
                "visitor_team_id": 2,
            }
        ]
    )
    games.to_parquet(season_dir / "games.parquet", index=False)

    stats = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_id": player_id,
                "date": pd.Timestamp(date_text),
                "minutes": 34.0,
                "pts": 25,
                "reb": 8,
                "ast": 6,
                "stl": 2,
                "blk": 1,
                "fg3m": 4,
                "team_id": 1,
                "stat_id": 1,
            },
            {
                "game_id": game_id,
                "player_id": dnp_player_id,
                "date": pd.Timestamp(date_text),
                "minutes": 0.0,
                "pts": 0,
                "reb": 0,
                "ast": 0,
                "stl": 0,
                "blk": 0,
                "fg3m": 0,
                "team_id": 1,
                "stat_id": 2,
            },
        ]
    )
    stats.to_parquet(season_dir / "stats.parquet", index=False)

    cases = [
        ("pts_over", player_id, "points", 24.5, "over", "over"),
        ("reb_push", player_id, "rebounds", 8.0, "over", "push"),
        ("ast_under", player_id, "assists", 6.5, "under", "under"),
        ("stl_over", player_id, "steals", 1.5, "under", "over"),
        ("blk_under", player_id, "blocks", 1.5, "under", "under"),
        ("three_over", player_id, "threes", 3.5, "over", "over"),
        ("pr_over", player_id, "points_rebounds", 32.5, "over", "over"),
        ("pa_push", player_id, "points_assists", 31.0, "under", "push"),
        ("ra_over", player_id, "rebounds_assists", 13.5, "over", "over"),
        ("pra_under", player_id, "points_rebounds_assists", 39.5, "over", "under"),
        ("dnp", dnp_player_id, "points", 1.5, "over", "dnp_void_candidate"),
        (
            "missing_player",
            missing_player_id,
            "points",
            10.5,
            "under",
            "player_missing_final_game_void_candidate",
        ),
    ]

    priced_rows = []
    expected = {}

    for i, (
        case_id,
        pid,
        prop_type,
        line_value,
        bet_side,
        expected_status,
    ) in enumerate(cases):
        priced_rows.append(
            {
                "case_id": case_id,
                "game_id": game_id,
                "player_id": pid,
                "prop_type": prop_type,
                "line_value": float(line_value),
                "vendor": "synthetic_book_a" if i % 2 == 0 else "synthetic_book_b",
                "over_odds": -110 if i % 3 else 120,
                "under_odds": -110 if i % 4 else 105,
                "q_over_calibrated": 0.56 if bet_side == "over" else 0.44,
                "q_over_nonpush": 0.55 if bet_side == "over" else 0.45,
                "market_q_over": 0.50,
                "bet_side": bet_side,
                "bet_edge": 0.06,
                "bet_model_ev": 0.04,
                "external_test_record": False,
                "betting_threshold_policy": "synthetic_test_no_threshold",
            }
        )
        expected[case_id] = expected_status

    priced = pd.DataFrame(priced_rows)

    capture = shadow / "synthetic_capture"
    capture.mkdir(parents=True, exist_ok=False)
    priced.to_parquet(capture / "priced_markets.parquet", index=False)

    capture_manifest = {
        "schema_version": 1,
        "capture_id": "synthetic_grading_contract",
        "captured_at_utc": "2026-10-20T12:00:00+00:00",
        "date": date_text,
        "mode": "engineering",
        "tag": "synthetic_grading_integration",
        "status": "complete",
        "pricing_status": "priced_markets_archived",
        "freeze_id": "synthetic_contract_only_not_model_evidence",
        "freeze_stage": "synthetic_validation",
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

    source_hashes_before = {
        path.name: sha256_file(path)
        for path in capture.iterdir()
        if path.is_file()
    }

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
        / "capture_id=synthetic_grading_contract"
    )
    grade_dirs = sorted(p for p in grade_base.iterdir() if p.is_dir())
    if not grade_dirs:
        raise RuntimeError("Synthetic grader produced no grade directory.")
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

    graded = pd.read_parquet(grade_dir / "graded_quotes.parquet")
    if "case_id" not in graded.columns:
        raise AssertionError("case_id was not preserved through grading.")

    observed = (
        graded.set_index("case_id")["settlement_status"].to_dict()
    )
    for case_id, expected_status in expected.items():
        actual_status = observed.get(case_id)
        if actual_status != expected_status:
            raise AssertionError(
                f"{case_id}: expected {expected_status}, got {actual_status}"
            )

    expected_actual = {
        "pts_over": 25.0,
        "reb_push": 8.0,
        "ast_under": 6.0,
        "stl_over": 2.0,
        "blk_under": 1.0,
        "three_over": 4.0,
        "pr_over": 33.0,
        "pa_push": 31.0,
        "ra_over": 14.0,
        "pra_under": 39.0,
    }

    by_case = graded.set_index("case_id")

    for case_id, value in expected_actual.items():
        got = float(by_case.loc[case_id, "actual"])
        if got != value:
            raise AssertionError(
                f"{case_id}: expected actual={value}, got {got}"
            )

    pts_profit = float(by_case.loc["pts_over", "monitor_profit_units"])
    pts_odds = float(by_case.loc["pts_over", "over_odds"])
    if abs(pts_profit - american_profit(pts_odds)) > 1e-12:
        raise AssertionError("Winning monitored over profit is incorrect.")

    loss_profit = float(by_case.loc["stl_over", "monitor_profit_units"])
    if loss_profit != -1.0:
        raise AssertionError(
            f"Losing monitored side should be -1.0, got {loss_profit}"
        )

    push_profit = float(by_case.loc["reb_push", "monitor_profit_units"])
    if push_profit != 0.0:
        raise AssertionError(
            f"Push profit should be 0.0, got {push_profit}"
        )

    if not np.isnan(float(by_case.loc["reb_push", "brier_calibrated"])):
        raise AssertionError(
            "Push row must be excluded from binary Brier scoring."
        )

    source_hashes_after = {
        path.name: sha256_file(path)
        for path in capture.iterdir()
        if path.is_file()
    }
    if source_hashes_before != source_hashes_after:
        raise AssertionError(
            "Original synthetic capture mutated during grading."
        )

    status_counts = (
        graded["settlement_status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    report = {
        "run_id": run_id,
        "status": "PASS",
        "purpose": (
            "Controlled software validation only. "
            "No model-performance evidence."
        ),
        "synthetic_cases": int(len(graded)),
        "settlement_status_counts": {
            str(k): int(v)
            for k, v in status_counts.items()
        },
        "all_expected_settlements_matched": True,
        "win_loss_push_profit_checks": True,
        "push_excluded_from_binary_scoring": True,
        "capture_immutability_check": True,
        "grade_archive_checksum_verification": True,
        "shadow_external_test_ledger_only": True,
        "real_external_test_ledger_touched": False,
        "grader_sha256": sha256_file(grader),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "synthetic_grading_integration_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.keep_shadow:
        shutil.rmtree(shadow)

    print("=" * 118)
    print("SYNTHETIC GRADING INTEGRATION")
    print("=" * 118)
    print(f"Cases:                  {len(graded)}")
    print("Expected settlements:   PASS")
    print("Win/loss/push profit:   PASS")
    print("Push scoring exclusion: PASS")
    print("Capture immutability:   PASS")
    print("Grade checksums:         PASS")
    print("External-test ledger:    NOT TOUCHED")
    print()
    print(
        "PASS: synthetic fixture validated grading software logic only; "
        "it is not sports-model evidence."
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
