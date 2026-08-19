from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "capture",
        type=Path,
        help=(
            "Path to a capture directory containing "
            "capture_manifest.json and checksums.sha256."
        ),
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


def main() -> None:
    args = parse_args()

    capture = args.capture

    if not capture.is_absolute():
        capture = (
            ROOT
            / capture
        )

    capture = capture.resolve()

    manifest_path = (
        capture
        / "capture_manifest.json"
    )

    checksum_path = (
        capture
        / "checksums.sha256"
    )

    if not manifest_path.exists():
        raise SystemExit(
            f"ERROR: missing {manifest_path}"
        )

    if not checksum_path.exists():
        raise SystemExit(
            f"ERROR: missing {checksum_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    expected: dict[str, str] = {}

    for line in checksum_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        digest, relative = line.split(
            "  ",
            maxsplit=1,
        )

        expected[
            relative
        ] = digest

    missing = []
    mismatches = []

    for relative, digest in expected.items():
        path = (
            capture
            / relative
        )

        if not path.exists():
            missing.append(
                relative
            )
            continue

        actual = sha256_file(
            path
        )

        if actual != digest:
            mismatches.append(
                {
                    "path": relative,
                    "expected": digest,
                    "actual": actual,
                }
            )

    print(
        "=" * 118
    )

    print(
        "EXTERNAL-TEST CAPTURE VERIFICATION"
    )

    print(
        "=" * 118
    )

    print(
        f"Capture ID: "
        f"{manifest.get('capture_id')}"
    )

    print(
        f"Date:       "
        f"{manifest.get('date')}"
    )

    print(
        f"Mode:       "
        f"{manifest.get('mode')}"
    )

    print(
        f"Freeze:     "
        f"{manifest.get('freeze_id')}"
    )

    print(
        f"Files checked: "
        f"{len(expected)}"
    )

    print(
        f"Missing: "
        f"{len(missing)}"
    )

    print(
        f"Hash mismatches: "
        f"{len(mismatches)}"
    )

    if missing:
        print()
        print(
            "Missing files:"
        )

        for item in missing:
            print(
                " ",
                item,
            )

    if mismatches:
        print()
        print(
            "Hash mismatches:"
        )

        for item in mismatches:
            print(
                " ",
                item[
                    "path"
                ],
            )

    if missing or mismatches:
        raise SystemExit(
            "FAIL: archived capture has changed."
        )

    print()
    print(
        "PASS: archived capture matches its frozen checksums."
    )


if __name__ == "__main__":
    main()
