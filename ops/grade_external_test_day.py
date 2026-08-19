from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path.cwd()
ARCHIVE_ROOT = ROOT / "data/external_test/season=2026"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grade all completed captures for one slate date. "
            "Use only after final outcomes have been refreshed locally."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Slate date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "external",
            "engineering",
            "all",
        ],
        default="external",
    )

    args = parser.parse_args()

    capture_root = (
        ARCHIVE_ROOT
        / f"date={args.date}"
        / "captures"
    )

    if not capture_root.exists():
        raise SystemExit(
            f"ERROR: no capture directory for {args.date}: {capture_root}"
        )

    captures = []

    for manifest_path in sorted(
        capture_root.glob(
            "*/capture_manifest.json"
        )
    ):
        try:
            payload = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        mode = payload.get("mode")

        if args.mode != "all" and mode != args.mode:
            continue

        captures.append(
            manifest_path.parent
        )

    if not captures:
        raise SystemExit(
            f"ERROR: no completed {args.mode} captures found for {args.date}."
        )

    print(
        f"Found {len(captures)} capture(s) for {args.date}."
    )

    for capture in captures:
        print()
        print("=" * 118)
        print(
            f"GRADING {capture.name}"
        )
        print("=" * 118)

        subprocess.run(
            [
                sys.executable,
                "ops/grade_external_test_capture.py",
                "--capture",
                str(capture),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
