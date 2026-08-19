from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path.cwd()


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "grade",
        type=Path,
        help="Grade directory containing grade_manifest.json and checksums.sha256.",
    )

    args = parser.parse_args()

    grade = args.grade

    if not grade.is_absolute():
        grade = ROOT / grade

    grade = grade.resolve()

    manifest_path = grade / "grade_manifest.json"
    checksum_path = grade / "checksums.sha256"

    if not manifest_path.exists():
        raise SystemExit(
            f"ERROR: missing {manifest_path}"
        )

    if not checksum_path.exists():
        raise SystemExit(
            f"ERROR: missing {checksum_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    expected = {}

    for line in checksum_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        digest, relative = line.split(
            "  ",
            maxsplit=1,
        )

        expected[relative] = digest

    missing = []
    mismatches = []

    for relative, expected_digest in expected.items():
        path = grade / relative

        if not path.exists():
            missing.append(relative)
            continue

        actual = sha256_file(path)

        if actual != expected_digest:
            mismatches.append(
                {
                    "path": relative,
                    "expected": expected_digest,
                    "actual": actual,
                }
            )

    print("=" * 118)
    print("EXTERNAL-TEST GRADE VERIFICATION")
    print("=" * 118)
    print(
        f"Grade ID:    {manifest.get('grade_id')}"
    )
    print(
        f"Capture ID:  {manifest.get('capture_id')}"
    )
    print(
        f"Date:        {manifest.get('date')}"
    )
    print(
        f"Status:      {manifest.get('status')}"
    )
    print(
        f"Files checked: {len(expected)}"
    )
    print(
        f"Missing:       {len(missing)}"
    )
    print(
        f"Hash mismatch: {len(mismatches)}"
    )

    if missing:
        print("Missing files:")
        for item in missing:
            print(" ", item)

    if mismatches:
        print("Hash mismatches:")
        for item in mismatches:
            print(" ", item["path"])

    if missing or mismatches:
        raise SystemExit(
            "FAIL: grade archive has changed."
        )

    print()
    print(
        "PASS: grade archive matches its frozen checksums."
    )


if __name__ == "__main__":
    main()
