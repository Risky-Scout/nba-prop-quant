from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path.cwd()

DEFAULT_MANIFEST = Path(
    "models/frozen_manifests/LATEST.json"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    manifest_path = DEFAULT_MANIFEST

    if not manifest_path.exists():
        raise SystemExit(
            f"ERROR: missing {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    mismatches = []
    missing = []
    checked = 0

    for section, records in manifest[
        "files"
    ].items():
        for record in records:
            path = ROOT / record[
                "path"
            ]

            if not path.exists():
                missing.append(
                    record[
                        "path"
                    ]
                )
                continue

            checked += 1

            actual = sha256_file(
                path
            )

            if actual != record[
                "sha256"
            ]:
                mismatches.append(
                    {
                        "path": record[
                            "path"
                        ],
                        "expected": record[
                            "sha256"
                        ],
                        "actual": actual,
                    }
                )

    print(
        "=" * 118
    )

    print(
        "FROZEN MANIFEST VERIFICATION"
    )

    print(
        "=" * 118
    )

    print(
        f"Freeze ID: "
        f"{manifest['freeze_id']}"
    )

    print(
        f"Files checked: {checked}"
    )

    print(
        f"Missing: {len(missing)}"
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

        for path in missing:
            print(
                "  ",
                path,
            )

    if mismatches:
        print()
        print(
            "Changed files:"
        )

        for item in mismatches:
            print(
                "  ",
                item[
                    "path"
                ],
            )
            print(
                "    expected:",
                item[
                    "expected"
                ],
            )
            print(
                "    actual:  ",
                item[
                    "actual"
                ],
            )

    if missing or mismatches:
        raise SystemExit(
            "FAIL: frozen production state has changed."
        )

    print()
    print(
        "PASS: frozen production state matches the manifest."
    )


if __name__ == "__main__":
    main()
