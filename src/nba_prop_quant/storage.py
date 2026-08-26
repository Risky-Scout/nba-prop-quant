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


def upsert_parquet(
    df: pd.DataFrame,
    path: Path,
    key_columns: list[str],
) -> None:
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=key_columns, keep="last")
    write_parquet_atomic(df, path)


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
