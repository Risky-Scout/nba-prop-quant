#!/usr/bin/env python
"""Refuse a merge that would add a generated dataset or an oversized blob.

It answers the question a tree listing cannot: not "what does this branch
have" but "what would this branch add to production history".

This is the operator-facing view. The enforcement that actually runs on every
pull request lives in ``tests/test_game_latent_state_shadow_artifact_budget.py``
and shares this module's logic through
:func:`nba_prop_quant.research.game_latent_state.safety.audit_merge_path`, so
there is one definition rather than two that can drift. The guard is a test
rather than a workflow step because ``.github/workflows/`` and ``ops/`` are
protected production paths that the shadow branch may not modify, and CI runs
the test suite regardless.

The distinction is not academic. A blob that is committed and later deleted is
absent from every subsequent tree and still travels with an ordinary merge
commit, because the commits that contain it come along too. The latent-state
research branches are the worked example: a 65 MB residual parquet was
committed in 43e4d2a and removed in 4e15feb, so ``git ls-tree HEAD`` on any of
their heads shows nothing while the object stays reachable from all three. The
mitigation relied on until now was remembering to squash-merge. This script is
the mitigation that does not depend on remembering.

Exit codes
----------
0   no new object breaches the contract
1   a new object is over the size ceiling, or is a generated dataset or cache
2   the base could not be resolved, so nothing was actually checked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.research.game_latent_state.safety import (  # noqa: E402
    MAX_MERGE_PATH_BLOB_BYTES,
    NOTABLE_MERGE_PATH_BLOB_BYTES,
    audit_merge_path,
    resolve_audit_base,
)

MEGABYTE = 1024 * 1024


def megabytes(value: int) -> str:
    return f"{value / MEGABYTE:.3f} MB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=None,
        help=(
            "commit or ref to audit against; defaults to PRODUCTION_BASE_REF, "
            "then GITHUB_BASE_REF, then the production merge base"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="also write the full report here",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="append a short markdown summary here (for $GITHUB_STEP_SUMMARY)",
    )
    args = parser.parse_args(argv)

    base = args.base or resolve_audit_base(PROJECT_ROOT)
    if base is None:
        print(
            "merge-path audit could not resolve a base commit. Nothing was "
            "checked. Fetch the production branch (or set PRODUCTION_BASE_REF) "
            "and run again -- a guard that cannot see the base has not "
            "verified anything.",
            file=sys.stderr,
        )
        return 2

    report = audit_merge_path(PROJECT_ROOT, base=base)

    print(f"base                : {report['base']}")
    print(f"new objects in path : {report['new_blob_count']}")
    largest = report["largest"]
    if largest is None:
        print("largest new blob    : none")
    else:
        print(
            f"largest new blob    : {megabytes(largest['bytes'])}  "
            f"{largest['sha'][:12]}  {largest['path']}"
        )
    print(
        f"ceiling             : {megabytes(MAX_MERGE_PATH_BLOB_BYTES)} "
        f"(reported above {megabytes(NOTABLE_MERGE_PATH_BLOB_BYTES)})"
    )

    for entry in report["notable"]:
        print(
            f"  NOTABLE  {megabytes(entry['bytes'])}  {entry['sha'][:12]}  "
            f"{entry['path']}"
        )
    for entry in report["over_ceiling"]:
        print(
            f"  OVER CEILING  {megabytes(entry['bytes'])}  {entry['sha'][:12]}  "
            f"{entry['path']}"
        )
    for entry in report["generated_artifacts"]:
        print(
            f"  GENERATED ARTIFACT  {megabytes(entry['bytes'])}  "
            f"{entry['sha'][:12]}  {entry['path']}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_out}")

    if args.summary_path is not None:
        verdict = "PASS" if report["passed"] else "FAIL"
        lines = [
            "### Merge-path blob audit",
            "",
            f"- verdict: **{verdict}**",
            f"- base: `{report['base']}`",
            f"- new objects: {report['new_blob_count']}",
        ]
        if largest is not None:
            lines.append(
                f"- largest new blob: {megabytes(largest['bytes'])} "
                f"(`{largest['path']}`)"
            )
        for entry in report["over_ceiling"]:
            lines.append(
                f"- over ceiling: {megabytes(entry['bytes'])} `{entry['path']}`"
            )
        for entry in report["generated_artifacts"]:
            lines.append(f"- generated artifact: `{entry['path']}`")
        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    if not report["passed"]:
        print(
            "\nmerge-path audit FAILED: the objects listed above would enter "
            "production history. Rebuild the branch from the production base "
            "and copy only the files it should carry; deleting the file in a "
            "later commit does not remove the object.",
            file=sys.stderr,
        )
        return 1

    print("\nmerge-path audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
