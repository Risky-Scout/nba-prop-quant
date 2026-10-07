"""Verify the final clean integration tree against its declared carry manifest.

Run on the integration branch. It answers four questions and refuses to
report a pass on any of them being unanswerable:

1.  Does the tree hold exactly what ``carry_manifest.json`` says it should --
    nothing excluded present, nothing carried missing?
2.  Does the merge path satisfy the blob contract: no new object above the
    ceiling, no generated dataset, cache or model binary anywhere in it?
3.  Does every carried ``SHA256SUMS`` file verify?
4.  Does every provenance hash recorded in ``final_model_spec.json`` resolve
    to a file that is actually present, with that content?

Question 2 is the one a tree listing cannot answer. ``git ls-tree`` reports
what a branch *has*; a merge transfers what its history *reaches*. The whole
reason this branch is built by allowlist rather than by merge is that the
research line committed a 65 MB residual parquet and deleted it two days
later, leaving it absent from every current tree and reachable from three
heads. See :func:`safety.merge_path_blobs`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table

from nba_prop_quant.research.game_latent_state.artifacts import (
    git_sha,
    sha256_file,
    write_json,
)
from nba_prop_quant.research.game_latent_state.safety import (
    GENERATED_ARTIFACT_SUFFIXES,
    MAX_MERGE_PATH_BLOB_BYTES,
    audit_merge_path,
    merge_path_blobs,
)

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTEGRATION_DIR = Path("research/final_integration")

MANIFEST_NAME = "carry_manifest.json"
OUTPUT_NAME = "integration_verification.json"

FINAL_MODEL_SPEC = Path("research/final_model/final_model_spec.json")

#: Any of these appearing anywhere in the merge path fails the audit,
#: independent of size. ``safety.GENERATED_ARTIFACT_NAMES`` lists the three
#: datasets by name; this is the broader sweep the brief asks for, so a
#: parquet nobody thought to name cannot slip through.
GENERATED_SUFFIXES: tuple[str, ...] = (".parquet", *GENERATED_ARTIFACT_SUFFIXES)

#: Two parquet files predate this work and live on the production base. They
#: are not new objects in the merge path, so the sweep below reports them
#: only if the merge path introduces them, which it must not.
PREEXISTING_PARQUET: frozenset[str] = frozenset(
    {
        "research/v2_gate2_certification_outputs/certification_contracts.parquet",
        "research/v2_gate2_certification_outputs/role_minutes_oof.parquet",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=INTEGRATION_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (PROJECT_ROOT / INTEGRATION_DIR / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    base = args.base or manifest["production_base"]

    contents = check_manifest_contents(manifest)
    blobs = check_merge_path(base)
    checksums = check_checksum_files()
    provenance = check_provenance_hashes()

    passed = all(
        section["passed"] for section in (contents, blobs, checksums, provenance)
    )
    report = {
        "study": "final_clean_integration_verification_v1",
        "production_base": base,
        "head_sha": git_sha(PROJECT_ROOT),
        "manifest_sha256": sha256_file(PROJECT_ROOT / INTEGRATION_DIR / MANIFEST_NAME),
        "carry_manifest_contents": contents,
        "merge_path_blob_audit": blobs,
        "checksum_files": checksums,
        "provenance_hashes": provenance,
        "snapshot_is_a_record_not_the_contract": (
            "Committing this file adds an object to the merge path, so the counts "
            "below describe head_sha and read low for any later commit. The suite "
            "re-derives every check against the live head; this file is evidence "
            "that it was run, not the thing CI trusts."
        ),
        "passed": passed,
        "verdict": (
            "CLEAN INTEGRATION TREE VERIFIED"
            if passed
            else "CLEAN INTEGRATION TREE NOT VERIFIED"
        ),
    }

    _print(report)
    output = write_json(report, PROJECT_ROOT / args.output_root / OUTPUT_NAME)
    console.print(f"\nwrote {output}")
    if not passed:
        raise SystemExit(1)


# ----------------------------------------------------------------------
# 1. the tree matches the manifest
# ----------------------------------------------------------------------


def tracked_paths() -> frozenset[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(completed.stdout.splitlines())


def is_ignored(path: str) -> bool:
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def check_manifest_contents(manifest: dict) -> dict:
    """Everything carried is tracked; everything excluded is not.

    The question is about the branch, not about the filesystem. A machine that
    has run the pipeline has a 68 MB residual parquet sitting in the working
    tree, and that is correct: the file is a gitignored local build product.
    Reporting it as an exclusion breach would conflate "this branch carries
    it" with "this disk has it", and would make the check impossible to pass
    on exactly the machine that produced the evidence.

    So exclusions are checked against ``git ls-files``, and anything excluded
    that is nonetheless on disk must be ignored -- untracked *and* unignored
    is the state one ``git add -A`` away from the merge path, and that is a
    breach.
    """
    tracked = tracked_paths()
    missing: list[str] = []
    for section in manifest["carry"].values():
        for declared in section["paths"]:
            if declared.endswith("/"):
                if not any(entry.startswith(declared) for entry in tracked):
                    missing.append(declared)
            elif declared not in tracked:
                missing.append(declared)

    committed: list[str] = []
    unignored: list[str] = []
    ignored_build_products: list[str] = []
    for section in manifest["exclude"].values():
        for declared in section["paths"]:
            if "*" in declared or "@" in declared:
                continue
            if declared in tracked:
                committed.append(declared)
            elif (PROJECT_ROOT / declared).exists():
                if is_ignored(declared):
                    ignored_build_products.append(declared)
                else:
                    unignored.append(declared)

    return {
        "carried_paths_declared": sum(
            len(section["paths"]) for section in manifest["carry"].values()
        ),
        "carried_paths_not_tracked": sorted(missing),
        "excluded_paths_tracked": sorted(committed),
        "excluded_paths_on_disk_but_not_ignored": sorted(unignored),
        "excluded_paths_on_disk_and_correctly_ignored": sorted(
            ignored_build_products
        ),
        "passed": not missing and not committed and not unignored,
    }


# ----------------------------------------------------------------------
# 2. the blob contract
# ----------------------------------------------------------------------


def check_merge_path(base: str) -> dict:
    audit = audit_merge_path(PROJECT_ROOT, base=base)
    generated = [
        entry
        for entry in merge_path_blobs(PROJECT_ROOT, base=base)
        if Path(entry["path"]).suffix in GENERATED_SUFFIXES
        and entry["path"] not in PREEXISTING_PARQUET
    ]
    return {
        "base": audit["base"],
        "base_resolved": audit["base_resolved"],
        "new_blob_count": audit["new_blob_count"],
        "largest_new_blob": audit["largest"],
        "ceiling_bytes": MAX_MERGE_PATH_BLOB_BYTES,
        "over_ceiling": audit["over_ceiling"],
        "notable": audit["notable"],
        "generated_artifacts_by_name": audit["generated_artifacts"],
        "generated_artifacts_by_suffix": generated,
        "suffixes_swept": list(GENERATED_SUFFIXES),
        "passed": bool(audit["passed"]) and not generated,
    }


# ----------------------------------------------------------------------
# 3. the checksum files
# ----------------------------------------------------------------------


def check_checksum_files() -> dict:
    """Verify every ``SHA256SUMS*`` file the tree carries.

    Three outcomes per line, not two. A present file whose content differs is
    a failure. A missing file is a failure *unless* it is one of the generated
    artifacts the blob contract forbids carrying, in which case the line is
    doing the job it was written for: recording the hash of something
    deliberately regenerated rather than shipped.

    That distinction is the whole point. ``factor_loadings.parquet`` and
    ``joint_event_grades.parquet`` are gitignored exports whose hashes are
    recorded so a rebuild can be checked against them. Treating their absence
    as a verification failure would make the two requirements -- every
    checksum file verifies, and no generated parquet enters the merge path --
    impossible to satisfy together. Treating it as a silent skip would let a
    checksum file cover nothing and still report a pass. So it is counted,
    named and classified.
    """
    results: list[dict] = []
    for path in sorted((PROJECT_ROOT / "research").rglob("SHA256SUMS*")):
        mismatched: list[str] = []
        absent: list[str] = []
        recorded_only: list[str] = []
        lines = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            name = name.strip()
            if not name:
                continue
            lines += 1
            target = path.parent / name
            if not target.exists():
                if Path(name).suffix in GENERATED_SUFFIXES:
                    recorded_only.append(name)
                else:
                    absent.append(name)
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != digest.strip():
                mismatched.append(name)
        results.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "entries": lines,
                "checked": lines - len(recorded_only),
                "mismatched": mismatched,
                "absent": absent,
                "recorded_only_generated_artifacts": recorded_only,
                "verified": not mismatched and not absent and lines > 0,
            }
        )
    return {
        "files": results,
        "files_checked": len(results),
        "files_verified": sum(1 for entry in results if entry["verified"]),
        "lines_verified": sum(
            entry["checked"] - len(entry["mismatched"]) for entry in results
        ),
        "lines_recording_a_regenerated_artifact": sum(
            len(entry["recorded_only_generated_artifacts"]) for entry in results
        ),
        "generated_artifact_suffixes_treated_as_recorded_only": list(
            GENERATED_SUFFIXES
        ),
        "passed": bool(results) and all(entry["verified"] for entry in results),
    }


# ----------------------------------------------------------------------
# 4. the provenance hashes in the final model specification
# ----------------------------------------------------------------------


def check_provenance_hashes() -> dict:
    """Every ``{path, sha256}`` pair the final spec records must resolve."""
    spec = json.loads((PROJECT_ROOT / FINAL_MODEL_SPEC).read_text(encoding="utf-8"))
    unresolved: list[dict] = []
    resolved = 0
    for name, entry in sorted(spec.get("source_artifacts", {}).items()):
        declared = entry.get("path")
        digest = entry.get("sha256")
        if not declared or not digest:
            continue
        target = PROJECT_ROOT / declared
        if not target.exists():
            unresolved.append({"artifact": name, "path": declared, "why": "absent"})
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            unresolved.append(
                {
                    "artifact": name,
                    "path": declared,
                    "why": "content differs",
                    "declared": digest,
                    "actual": actual,
                }
            )
            continue
        resolved += 1

    factor_spec = spec["factor_spec_hash"]
    authoritative = spec["authoritative_spec_hash"]
    return {
        "artifacts_resolved": resolved,
        "artifacts_unresolved": unresolved,
        "factor_spec_hash": factor_spec,
        "factor_spec_hash_is_the_authoritative_one": factor_spec == authoritative,
        "passed": not unresolved and factor_spec == authoritative,
    }


def _print(report: dict) -> None:
    table = Table(show_header=True)
    table.add_column("check")
    table.add_column("result")
    table.add_column("detail")
    contents = report["carry_manifest_contents"]
    blobs = report["merge_path_blob_audit"]
    checksums = report["checksum_files"]
    provenance = report["provenance_hashes"]
    table.add_row(
        "tree matches the carry manifest",
        "PASS" if contents["passed"] else "FAIL",
        f"{contents['carried_paths_declared']} declared, "
        f"{len(contents['carried_paths_not_tracked'])} untracked, "
        f"{len(contents['excluded_paths_tracked'])} excluded but committed, "
        f"{len(contents['excluded_paths_on_disk_but_not_ignored'])} unignored",
    )
    largest = blobs["largest_new_blob"]
    table.add_row(
        "merge-path blob contract",
        "PASS" if blobs["passed"] else "FAIL",
        f"{blobs['new_blob_count']} new objects, largest "
        f"{(largest or {}).get('bytes', 0):,} B, "
        f"{len(blobs['over_ceiling'])} over ceiling, "
        f"{len(blobs['generated_artifacts_by_suffix'])} generated",
    )
    table.add_row(
        "SHA256SUMS files verify",
        "PASS" if checksums["passed"] else "FAIL",
        f"{checksums['files_verified']}/{checksums['files_checked']} files, "
        f"{checksums['lines_verified']} lines verified, "
        f"{checksums['lines_recording_a_regenerated_artifact']} recording a "
        "regenerated artifact",
    )
    table.add_row(
        "provenance hashes resolve",
        "PASS" if provenance["passed"] else "FAIL",
        f"{provenance['artifacts_resolved']} resolved, "
        f"{len(provenance['artifacts_unresolved'])} unresolved",
    )
    console.print(table)
    console.rule(report["verdict"])


if __name__ == "__main__":
    main()
