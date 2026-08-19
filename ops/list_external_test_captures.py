from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
INDEX_PATH = (
    ROOT
    / "data/external_test/season=2026/capture_index.jsonl"
)


def main() -> None:
    if not INDEX_PATH.exists():
        print(
            "No external-test captures recorded yet."
        )
        return

    rows = []

    for line in INDEX_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rows.append(
                json.loads(
                    line
                )
            )

    frame = pd.DataFrame(
        rows
    )

    if frame.empty:
        print(
            "No external-test captures recorded yet."
        )
        return

    print(
        frame.sort_values(
            "captured_at_utc"
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
