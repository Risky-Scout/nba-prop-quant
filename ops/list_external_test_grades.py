from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
INDEX = (
    ROOT
    / "data/external_test/season=2026/grades/grade_index.jsonl"
)


def main() -> None:
    if not INDEX.exists():
        print(
            "No external-test grades recorded yet."
        )
        return

    rows = []

    for line in INDEX.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rows.append(
                json.loads(line)
            )

    frame = pd.DataFrame(rows)

    if frame.empty:
        print(
            "No external-test grades recorded yet."
        )
        return

    print(
        frame.sort_values(
            "graded_at_utc"
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
