from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path.cwd()
OPS_DIR = ROOT / "ops"
MODEL_MANIFEST = ROOT / "models/frozen_manifests/LATEST.json"
ARCHIVE_ROOT = ROOT / "data/external_test/season=2026"

EASTERN = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and immutably archive one frozen NBA prop "
            "projection/market capture."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Target NBA slate date, YYYY-MM-DD.",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "external",
            "engineering",
        ],
        default="external",
        help=(
            "external: official 2026-27 external-test capture; "
            "engineering: preseason/no-score smoke capture."
        ),
    )

    parser.add_argument(
        "--combo-simulations",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Optional human tag such as morning, opening, pre_tip. "
            "The timestamp remains canonical."
        ),
    )

    parser.add_argument(
        "--skip-market-pricing",
        action="store_true",
        help=(
            "Archive a projection-only capture. Useful when "
            "sportsbook markets are not available yet."
        ),
    )

    return parser.parse_args()


def parse_iso_date(text: str) -> pd.Timestamp:
    try:
        value = pd.Timestamp(text)
    except Exception as exc:
        raise SystemExit(
            f"ERROR: invalid date {text!r}; use YYYY-MM-DD. {exc}"
        )

    if value.strftime("%Y-%m-%d") != text:
        raise SystemExit(
            f"ERROR: date must be exactly YYYY-MM-DD; got {text!r}"
        )

    return value.normalize()


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
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def run_command(
    args: list[str],
    log_path: Path,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = [
        "$ " + " ".join(args),
        "",
        "===== STDOUT =====",
        result.stdout.rstrip(),
        "",
        "===== STDERR =====",
        result.stderr.rstrip(),
        "",
        f"RETURN CODE: {result.returncode}",
        "",
    ]

    log_path.write_text(
        "\n".join(payload),
        encoding="utf-8",
    )

    print(
        result.stdout,
        end="",
    )

    if result.stderr:
        print(
            result.stderr,
            end="",
            file=sys.stderr,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): "
            + " ".join(args)
        )

    return result


def latest_raw_history_date() -> pd.Timestamp | None:
    latest: pd.Timestamp | None = None
    season_root = ROOT / "data/raw/seasons"
    game_date_map: dict[int, pd.Timestamp] | None = None

    def load_game_date_map() -> dict[int, pd.Timestamp]:
        mapping: dict[int, pd.Timestamp] = {}

        for games_path in sorted(
            season_root.rglob("games.parquet")
        ):
            games = pd.read_parquet(games_path)

            if games.empty:
                continue

            required = {"id", "date"}
            missing = required - set(games.columns)

            if missing:
                raise RuntimeError(
                    f"Raw games file missing {sorted(missing)}: {games_path}"
                )

            dates = pd.to_datetime(
                games["date"],
                errors="coerce",
            )

            for game_id, game_date in zip(
                games["id"],
                dates,
            ):
                if pd.isna(game_id) or pd.isna(game_date):
                    continue

                mapping[int(game_id)] = pd.Timestamp(
                    game_date
                ).normalize()

        return mapping

    for stats_path in sorted(
        season_root.rglob("stats.parquet")
    ):
        stats = pd.read_parquet(stats_path)

        # In-progress season files may legitimately be 0-row/0-column.
        if stats.empty:
            continue

        if "date" in stats.columns:
            dates = pd.to_datetime(
                stats["date"],
                errors="coerce",
            )

        elif "game_id" in stats.columns:
            if game_date_map is None:
                game_date_map = load_game_date_map()

            game_ids = pd.to_numeric(
                stats["game_id"],
                errors="coerce",
            )

            dates = pd.to_datetime(
                game_ids.map(
                    lambda value: (
                        game_date_map.get(int(value))
                        if pd.notna(value)
                        else pd.NaT
                    )
                ),
                errors="coerce",
            )

        else:
            raise RuntimeError(
                "Non-empty raw stats file has neither date nor game_id: "
                f"{stats_path}"
            )

        if not dates.notna().any():
            continue

        current = dates.max().normalize()

        if latest is None or current > latest:
            latest = current

    return latest


def latest_raw_advanced_date() -> pd.Timestamp | None:
    latest: pd.Timestamp | None = None
    root = ROOT / "data/raw/advanced"
    season_root = ROOT / "data/raw/seasons"
    game_date_map: dict[int, pd.Timestamp] | None = None

    def load_game_date_map() -> dict[int, pd.Timestamp]:
        mapping: dict[int, pd.Timestamp] = {}

        for games_path in sorted(
            season_root.rglob("games.parquet")
        ):
            games = pd.read_parquet(games_path)

            if games.empty:
                continue

            required = {"id", "date"}
            missing = required - set(games.columns)

            if missing:
                raise RuntimeError(
                    f"Raw games file missing {sorted(missing)}: {games_path}"
                )

            dates = pd.to_datetime(
                games["date"],
                errors="coerce",
            )

            for game_id, game_date in zip(
                games["id"],
                dates,
            ):
                if pd.isna(game_id) or pd.isna(game_date):
                    continue

                mapping[int(game_id)] = pd.Timestamp(
                    game_date
                ).normalize()

        return mapping

    for advanced_path in sorted(
        root.rglob("advanced.parquet")
    ):
        advanced = pd.read_parquet(advanced_path)

        # In-progress season files may legitimately be 0-row/0-column.
        if advanced.empty:
            continue

        if "date" in advanced.columns:
            dates = pd.to_datetime(
                advanced["date"],
                errors="coerce",
            )

        elif "game_id" in advanced.columns:
            if game_date_map is None:
                game_date_map = load_game_date_map()

            game_ids = pd.to_numeric(
                advanced["game_id"],
                errors="coerce",
            )

            dates = pd.to_datetime(
                game_ids.map(
                    lambda value: (
                        game_date_map.get(int(value))
                        if pd.notna(value)
                        else pd.NaT
                    )
                ),
                errors="coerce",
            )

        else:
            raise RuntimeError(
                "Non-empty raw advanced file has neither date nor game_id: "
                f"{advanced_path}"
            )

        if not dates.notna().any():
            continue

        current = dates.max().normalize()

        if latest is None or current > latest:
            latest = current

    return latest


def optional_file_record(
    path: Path,
) -> dict | None:
    if not path.exists():
        return None

    stat = path.stat()

    return {
        "path": str(
            path.relative_to(
                ROOT
            )
        ),
        "sha256": sha256_file(
            path
        ),
        "bytes": int(
            stat.st_size
        ),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def copy_artifact(
    source: Path,
    destination: Path,
) -> dict:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    return {
        "archive_path": str(
            destination.relative_to(
                ROOT
            )
        ),
        "source_path": str(
            source.relative_to(
                ROOT
            )
        ),
        "sha256": sha256_file(
            destination
        ),
        "bytes": int(
            destination.stat().st_size
        ),
    }


def main() -> None:
    args = parse_args()

    target_date = parse_iso_date(
        args.date
    )

    if not (
        ROOT
        / "src/nba_prop_quant"
    ).exists():
        raise SystemExit(
            "ERROR: run from nba_prop_quant_blueprint project root."
        )

    if not MODEL_MANIFEST.exists():
        raise SystemExit(
            f"ERROR: missing deployment manifest: {MODEL_MANIFEST}"
        )

    manifest = load_json(
        MODEL_MANIFEST
    )

    if manifest.get(
        "freeze_stage"
    ) != "external_test_deployment":
        raise SystemExit(
            "ERROR: LATEST manifest is not external_test_deployment."
        )

    now_utc = datetime.now(
        timezone.utc
    )

    now_et = now_utc.astimezone(
        EASTERN
    )

    if args.mode == "external":
        if target_date.date() != now_et.date():
            raise SystemExit(
                "ERROR: official external capture must be run on the "
                "actual target date in America/New_York. "
                f"Target={target_date.date()}, today={now_et.date()}. "
                "Use --mode engineering for preseason/system tests."
            )

    capture_timestamp = now_utc.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    safe_tag = ""

    if args.tag:
        safe_tag = "_" + "".join(
            char
            if char.isalnum()
            or char in {
                "-",
                "_",
            }
            else "_"
            for char in args.tag
        )

    capture_id = (
        capture_timestamp
        + safe_tag
    )

    capture_dir = (
        ARCHIVE_ROOT
        / f"date={args.date}"
        / "captures"
        / capture_id
    )

    if capture_dir.exists():
        raise SystemExit(
            f"ERROR: capture already exists: {capture_dir}"
        )

    capture_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_dir = (
        capture_dir
        / "logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 118
    )

    print(
        "NBA EXTERNAL-TEST CAPTURE"
    )

    print(
        "=" * 118
    )

    print(
        f"Date:       {args.date}"
    )

    print(
        f"Mode:       {args.mode}"
    )

    print(
        f"Capture ID: {capture_id}"
    )

    print(
        f"Freeze:     {manifest['freeze_id']}"
    )

    # ------------------------------------------------------------------
    # Frozen-state verification and production contract.
    # ------------------------------------------------------------------
    run_command(
        [
            sys.executable,
            "scripts/verify_frozen_manifest.py",
        ],
        log_dir
        / "01_verify_frozen_manifest.log",
    )

    run_command(
        [
            sys.executable,
            "scripts/10a_validate_production_contract.py",
        ],
        log_dir
        / "02_validate_production_contract.log",
    )

    history_latest = (
        latest_raw_history_date()
    )

    advanced_latest = (
        latest_raw_advanced_date()
    )

    if (
        history_latest
        is not None
        and history_latest
        >= target_date
    ):
        raise RuntimeError(
            "Raw history contains a date on/after the target slate date: "
            f"history_latest={history_latest.date()}, "
            f"target={target_date.date()}. "
            "This can create retrospective leakage."
        )

    if (
        advanced_latest
        is not None
        and advanced_latest
        >= target_date
    ):
        raise RuntimeError(
            "Raw advanced history contains a date on/after the target "
            f"slate date: advanced_latest={advanced_latest.date()}, "
            f"target={target_date.date()}."
        )

    # ------------------------------------------------------------------
    # Produce the contemporaneous projection.
    # ------------------------------------------------------------------
    projection_command = [
        sys.executable,
        "scripts/10_predict_slate.py",
        "--date",
        args.date,
    ]

    run_command(
        projection_command,
        log_dir
        / "03_predict_slate.log",
    )

    projection_path = (
        ROOT
        / "data/processed/projections"
        / f"{args.date}.parquet"
    )

    artifacts: list[dict] = []

    if not projection_path.exists():
        metadata = {
            "schema_version": 1,
            "capture_id": capture_id,
            "captured_at_utc": now_utc.isoformat(),
            "captured_at_et": now_et.isoformat(),
            "date": args.date,
            "mode": args.mode,
            "tag": args.tag,
            "status": "no_projection_file_no_games_or_no_slate",
            "freeze_id": manifest[
                "freeze_id"
            ],
            "freeze_stage": manifest[
                "freeze_stage"
            ],
            "deployment_manifest_sha256": sha256_file(
                MODEL_MANIFEST
            ),
            "history_latest_date": (
                str(
                    history_latest.date()
                )
                if history_latest
                is not None
                else None
            ),
            "advanced_latest_date": (
                str(
                    advanced_latest.date()
                )
                if advanced_latest
                is not None
                else None
            ),
            "artifacts": artifacts,
        }

        metadata_path = (
            capture_dir
            / "capture_manifest.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        artifacts.append(
            copy_artifact(
                MODEL_MANIFEST,
                capture_dir
                / "deployment_manifest.json",
            )
        )

        current_script = Path(
            __file__
        ).resolve()

        if current_script.exists():
            artifacts.append(
                copy_artifact(
                    current_script,
                    capture_dir
                    / "capture_external_test_day.py",
                )
            )

        metadata["artifacts"] = artifacts

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        checksum_lines = []

        for path in sorted(
            capture_dir.rglob("*")
        ):
            if (
                path.is_file()
                and path.name
                != "checksums.sha256"
            ):
                checksum_lines.append(
                    f"{sha256_file(path)}  "
                    f"{path.relative_to(capture_dir)}"
                )

        (
            capture_dir
            / "checksums.sha256"
        ).write_text(
            "\n".join(
                checksum_lines
            )
            + "\n",
            encoding="utf-8",
        )

        index_path = (
            ARCHIVE_ROOT
            / "capture_index.jsonl"
        )

        index_entry = {
            "capture_id": capture_id,
            "date": args.date,
            "mode": args.mode,
            "tag": args.tag,
            "freeze_id": manifest[
                "freeze_id"
            ],
            "capture_manifest": str(
                metadata_path.relative_to(
                    ROOT
                )
            ),
            "captured_at_utc": now_utc.isoformat(),
            "pricing_status": "no_slate",
        }

        with index_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    index_entry,
                    sort_keys=True,
                )
                + "\n"
            )

        print(
            "No projection file was created. "
            "Archived the complete no-slate capture."
        )

        print(
            f"Capture archive: {capture_dir}"
        )

        return

    artifacts.append(
        copy_artifact(
            projection_path,
            capture_dir
            / "projection.parquet",
        )
    )

    projection = pd.read_parquet(
        projection_path
    )

    if projection.empty:
        raise RuntimeError(
            "Projection file exists but contains zero rows."
        )

    projection_freezes = set(
        projection[
            "freeze_id"
        ].dropna().astype(
            str
        ).unique()
    )

    if projection_freezes != {
        manifest[
            "freeze_id"
        ]
    }:
        raise RuntimeError(
            "Projection freeze mismatch: "
            f"{projection_freezes}"
        )

    if "history_latest_date" in projection.columns:
        embedded_history_dates = pd.to_datetime(
            projection[
                "history_latest_date"
            ],
            errors="coerce",
        )

        if (
            embedded_history_dates.notna().any()
            and embedded_history_dates.max().normalize()
            >= target_date
        ):
            raise RuntimeError(
                "Projection embeds a history_latest_date "
                "on/after the target date."
            )

    # ------------------------------------------------------------------
    # Fetch/price current market quotes and archive the exact snapshot.
    # ------------------------------------------------------------------
    pricing_status = (
        "skipped"
        if args.skip_market_pricing
        else "attempted"
    )

    priced_path = (
        ROOT
        / "data/processed/priced_markets"
        / f"{args.date}.parquet"
    )

    rejected_path = (
        ROOT
        / "data/processed/priced_markets/rejected"
        / f"{args.date}.parquet"
    )

    if not args.skip_market_pricing:
        # Remove stale same-date outputs before fetching current markets.
        # The contemporaneous projection is already copied into the archive.
        for path in [
            priced_path,
            rejected_path,
        ]:
            if path.exists():
                path.unlink()

        run_command(
            [
                sys.executable,
                "scripts/15_price_markets.py",
                "--date",
                args.date,
                "--combo-simulations",
                str(
                    args.combo_simulations
                ),
            ],
            log_dir
            / "04_price_markets.log",
        )

        if priced_path.exists():
            artifacts.append(
                copy_artifact(
                    priced_path,
                    capture_dir
                    / "priced_markets.parquet",
                )
            )

            priced = pd.read_parquet(
                priced_path
            )

            if (
                "external_test_record"
                not in priced.columns
                or not bool(
                    priced[
                        "external_test_record"
                    ].all()
                )
            ):
                raise RuntimeError(
                    "Priced-market output is not marked "
                    "as external_test_record."
                )

            pricing_status = (
                "priced_markets_archived"
            )

        else:
            pricing_status = (
                "no_current_markets_returned"
            )

        if rejected_path.exists():
            artifacts.append(
                copy_artifact(
                    rejected_path,
                    capture_dir
                    / "rejected_markets.parquet",
                )
            )

    # ------------------------------------------------------------------
    # Archive deployment manifest, orchestration source, and run metadata.
    # ------------------------------------------------------------------
    artifacts.append(
        copy_artifact(
            MODEL_MANIFEST,
            capture_dir
            / "deployment_manifest.json",
        )
    )

    current_script = Path(
        __file__
    ).resolve()

    if current_script.exists():
        artifacts.append(
            copy_artifact(
                current_script,
                capture_dir
                / "capture_external_test_day.py",
            )
        )

    metadata = {
        "schema_version": 1,
        "capture_id": capture_id,
        "captured_at_utc": now_utc.isoformat(),
        "captured_at_et": now_et.isoformat(),
        "date": args.date,
        "mode": args.mode,
        "tag": args.tag,
        "status": "complete",
        "pricing_status": pricing_status,
        "combo_simulations": int(
            args.combo_simulations
        ),
        "freeze_id": manifest[
            "freeze_id"
        ],
        "freeze_stage": manifest[
            "freeze_stage"
        ],
        "deployment_manifest_sha256": sha256_file(
            MODEL_MANIFEST
        ),
        "history_latest_date": (
            str(
                history_latest.date()
            )
            if history_latest
            is not None
            else None
        ),
        "advanced_latest_date": (
            str(
                advanced_latest.date()
            )
            if advanced_latest
            is not None
            else None
        ),
        "projection_rows": int(
            len(
                projection
            )
        ),
        "projection_games": int(
            projection[
                "game_id"
            ].nunique()
        ),
        "artifacts": artifacts,
        "operational_policy": {
            "auto_betting_threshold": None,
            "external_test_result_mutation": False,
            "original_projection_mutation": False,
            "original_market_snapshot_mutation": False,
        },
    }

    metadata_path = (
        capture_dir
        / "capture_manifest.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    checksum_lines = []

    for path in sorted(
        capture_dir.rglob("*")
    ):
        if (
            path.is_file()
            and path.name
            != "checksums.sha256"
        ):
            checksum_lines.append(
                f"{sha256_file(path)}  "
                f"{path.relative_to(capture_dir)}"
            )

    (
        capture_dir
        / "checksums.sha256"
    ).write_text(
        "\n".join(
            checksum_lines
        )
        + "\n",
        encoding="utf-8",
    )

    index_path = (
        ARCHIVE_ROOT
        / "capture_index.jsonl"
    )

    index_entry = {
        "capture_id": capture_id,
        "date": args.date,
        "mode": args.mode,
        "tag": args.tag,
        "freeze_id": manifest[
            "freeze_id"
        ],
        "capture_manifest": str(
            metadata_path.relative_to(
                ROOT
            )
        ),
        "captured_at_utc": now_utc.isoformat(),
        "pricing_status": pricing_status,
    }

    with index_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                index_entry,
                sort_keys=True,
            )
            + "\n"
        )

    print()
    print(
        "=" * 118
    )

    print(
        "CAPTURE COMPLETE"
    )

    print(
        "=" * 118
    )

    print(
        f"Archive: {capture_dir}"
    )

    print(
        f"Projection rows: "
        f"{len(projection):,}"
    )

    print(
        f"Projection games: "
        f"{projection['game_id'].nunique():,}"
    )

    print(
        f"Pricing status: "
        f"{pricing_status}"
    )

    print(
        "Original archived artifacts should never be edited. "
        "Outcomes are attached later in separate grading files."
    )


if __name__ == "__main__":
    main()
