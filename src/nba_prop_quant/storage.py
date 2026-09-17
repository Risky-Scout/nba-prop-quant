from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        df.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def sort_by_keys(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """Order rows deterministically by the dataset's declared key columns.

    Rolling state is rewritten every refresh, so without an explicit order the
    stored row order depends on the arrival order of the incoming batch. Two
    runs over the same semantic records would then produce different files and
    neither diffing nor content fingerprinting would mean anything.

    The sort is stable and pins NaN placement so the result depends only on the
    key values, never on how the frame happened to be assembled.
    """
    missing = [column for column in key_columns if column not in df.columns]

    if missing:
        raise KeyError(
            f"cannot order rows by absent key column(s): {', '.join(missing)}"
        )

    ordered = df.sort_values(
        by=list(key_columns),
        kind="stable",
        na_position="last",
    )

    return ordered.reset_index(drop=True)


def upsert_parquet(
    df: pd.DataFrame,
    path: Path,
    key_columns: list[str],
) -> None:
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    # Deduplicate first: keep="last" resolves a replaced record in favour of
    # the incoming batch, and that depends on concat order. Sorting beforehand
    # could hand the row to the stale copy instead.
    df = df.drop_duplicates(subset=key_columns, keep="last")
    write_parquet_atomic(sort_by_keys(df, key_columns), path)


def read_parquet_tree(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)


def timestamped_jsonl_append(
    records: list[dict[str, Any]],
    path: Path,
    snapshot_type: str,
    *,
    captured_at: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            envelope = {
                "captured_at": captured_at,
                "snapshot_type": snapshot_type,
                "payload": record,
            }
            handle.write(json.dumps(envelope, separators=(",", ":")) + "\n")
