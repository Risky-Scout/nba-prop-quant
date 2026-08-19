from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
DOSSIER = ROOT / "review/SENIOR_QUANT_REVIEW_DOSSIER.md"
ARTIFACT_INDEX = ROOT / "review/SENIOR_QUANT_REVIEW_ARTIFACT_INDEX.json"
LATEST_MANIFEST = ROOT / "models/frozen_manifests/LATEST.json"
OUT_DIR = ROOT / "data/validation/senior_quant_handoff"


REQUIRED_HEADINGS = [
    "## 1. Review posture",
    "## 2. Data and information-set design",
    "## 3. Mean-model selection",
    "## 4. Marginal distributions",
    "## 5. Combination-prop dependence",
    "## 6. Probability calibration and market benchmark",
    "## 7. Development betting diagnostics",
    "## 8. Production architecture and governance",
    "## 9. Grading/settlement QA before external testing",
    "## 10. Known limitations a senior reviewer should attack",
    "## 11. Claims permitted vs. not permitted",
    "## 12. Suggested senior-quant review agenda",
    "## 13. Reproducibility commands",
    "## 14. Artifact index",
    "## 15. Reviewer sign-off",
]

REQUIRED_PHRASES = [
    "It is not presented as having a proven broad sportsbook edge.",
    "The 2025 sportsbook sample is development/model-selection evidence; 2026–27 is the first prospective external test.",
    "Automatic betting threshold: **none**",
    "Synthetic QA is software evidence only.",
    "The real 2025 replay is schema/settlement realism evidence only.",
    "A proven broad sportsbook edge.",
]

FORBIDDEN_UNRESOLVED_MARKERS = [
    "PENDING",
    "UNKNOWN",
    "TODO",
    "TBD",
    "_Not available from the current local artifact schema._",
]

EXPECTED_QA_STATUS = {
    "Synthetic controlled integration test": "PASS",
    "Real 2025 development-data grading replay": "PASS",
    "Pricing ↔ grading source-contract audit": "PASS_WITH_RUNTIME_CONFIRMATION",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_freeze_id(text: str) -> str | None:
    match = re.search(
        r"Frozen deployment ID:\s*`([^`]+)`",
        text,
    )
    return match.group(1) if match else None


def find_freeze_stage(text: str) -> str | None:
    match = re.search(
        r"Freeze stage:\s*`([^`]+)`",
        text,
    )
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only clarity and consistency audit of the generated "
            "senior-quant review dossier."
        )
    )
    parser.parse_args()

    for path in (DOSSIER, ARTIFACT_INDEX, LATEST_MANIFEST):
        if not path.exists():
            raise SystemExit(f"ERROR: required review artifact missing: {path}")

    text = DOSSIER.read_text(encoding="utf-8")
    index = load_json(ARTIFACT_INDEX)
    manifest = load_json(LATEST_MANIFEST)

    failures: list[str] = []
    warnings: list[str] = []
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "detail": detail,
            }
        )
        if not passed:
            failures.append(f"{name}: {detail}")

    # Structure.
    for heading in REQUIRED_HEADINGS:
        record(
            f"heading::{heading}",
            heading in text,
            "present" if heading in text else "missing",
        )

    # Core posture language.
    for phrase in REQUIRED_PHRASES:
        record(
            f"phrase::{phrase[:50]}",
            phrase in text,
            "present" if phrase in text else "missing",
        )

    # Unresolved placeholders.
    for marker in FORBIDDEN_UNRESOLVED_MARKERS:
        found = marker in text
        record(
            f"unresolved_marker::{marker}",
            not found,
            "absent" if not found else "FOUND",
        )

    # QA status language.
    for label, expected in EXPECTED_QA_STATUS.items():
        pattern = re.escape(label) + r":\s*\*\*([^*]+)\*\*"
        match = re.search(pattern, text)
        actual = match.group(1).strip() if match else None
        record(
            f"qa_status::{label}",
            actual == expected,
            f"expected={expected}, actual={actual}",
        )

    # Freeze consistency.
    dossier_freeze = find_freeze_id(text)
    dossier_stage = find_freeze_stage(text)
    manifest_freeze = manifest.get("freeze_id")
    manifest_stage = manifest.get(
        "freeze_stage",
        manifest.get("stage"),
    )
    index_freeze = index.get("freeze_id")
    index_stage = index.get("freeze_stage")

    record(
        "freeze_id_consistency",
        (
            dossier_freeze
            == manifest_freeze
            == index_freeze
        ),
        (
            f"dossier={dossier_freeze}, "
            f"manifest={manifest_freeze}, "
            f"index={index_freeze}"
        ),
    )

    record(
        "freeze_stage_consistency",
        (
            dossier_stage
            == manifest_stage
            == index_stage
        ),
        (
            f"dossier={dossier_stage}, "
            f"manifest={manifest_stage}, "
            f"index={index_stage}"
        ),
    )

    # Artifact-index hashes.
    artifact_records = index.get("artifacts", [])
    if not isinstance(artifact_records, list) or not artifact_records:
        record(
            "artifact_index_nonempty",
            False,
            "artifact index has no artifact records",
        )
    else:
        record(
            "artifact_index_nonempty",
            True,
            f"{len(artifact_records)} artifact records",
        )

        missing = []
        mismatches = []

        for item in artifact_records:
            relative = item.get("path")
            expected_hash = item.get("sha256")

            if not relative or not expected_hash:
                mismatches.append(
                    {
                        "path": relative,
                        "reason": "missing path/hash metadata",
                    }
                )
                continue

            path = ROOT / relative

            if not path.exists():
                missing.append(relative)
                continue

            actual_hash = sha256_file(path)

            if actual_hash != expected_hash:
                mismatches.append(
                    {
                        "path": relative,
                        "expected": expected_hash,
                        "actual": actual_hash,
                    }
                )

        record(
            "artifact_index_files_present",
            not missing,
            (
                "all present"
                if not missing
                else f"missing={missing[:10]}"
            ),
        )

        record(
            "artifact_index_hashes_match",
            not mismatches,
            (
                "all match"
                if not mismatches
                else f"mismatches={mismatches[:5]}"
            ),
        )

    # Clarity warnings that do not block packaging.
    if "PASS_WITH_RUNTIME_CONFIRMATION" in text:
        warnings.append(
            "Pricing↔grading contract remains explicitly pending first-real-file "
            "runtime field-name confirmation for side/edge/EV. This is correctly "
            "presented as an operational schema item, not a model-performance failure."
        )

    if "closing-line-value" not in text.lower() and "closing-line" not in text.lower():
        warnings.append(
            "Dossier does not visibly emphasize a future closing-line-value study."
        )

    if "quote shopping" not in text.lower() and "quote-shopping" not in text.lower():
        warnings.append(
            "Dossier does not visibly emphasize quote-shopping as a separate source of ROI."
        )

    now = datetime.now(timezone.utc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    status = "PASS" if not failures else "FAIL"

    report = {
        "generated_at_utc": now.isoformat(),
        "status": status,
        "dossier": str(DOSSIER.relative_to(ROOT)),
        "dossier_sha256": sha256_file(DOSSIER),
        "artifact_index": str(ARTIFACT_INDEX.relative_to(ROOT)),
        "artifact_index_sha256": sha256_file(ARTIFACT_INDEX),
        "freeze_id": manifest_freeze,
        "freeze_stage": manifest_stage,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }

    json_path = OUT_DIR / "dossier_clarity_audit.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md_lines = [
        "# Senior Quant Dossier Clarity Audit",
        "",
        f"- Status: **{status}**",
        f"- Generated UTC: `{now.isoformat()}`",
        f"- Freeze ID: `{manifest_freeze}`",
        f"- Freeze stage: `{manifest_stage}`",
        f"- Dossier SHA-256: `{report['dossier_sha256']}`",
        "",
        "## Result",
        "",
    ]

    if failures:
        md_lines.append(
            "The dossier is **not ready to package**. Resolve the blocking items below."
        )
        md_lines += ["", "### Blocking items", ""]
        for failure in failures:
            md_lines.append(f"- {failure}")
    else:
        md_lines.append(
            "The dossier passes the structural, posture-language, QA-status, "
            "freeze-consistency, and artifact-hash checks required for reviewer handoff."
        )

    if warnings:
        md_lines += ["", "## Non-blocking review notes", ""]
        for warning in warnings:
            md_lines.append(f"- {warning}")

    md_lines += [
        "",
        "## Interpretation",
        "",
        "This is a clarity/consistency audit, not a statistical endorsement. "
        "A senior reviewer should still challenge leakage controls, sample selection, "
        "calibration stability, distributional assumptions, dependence, quote shopping, "
        "and the prospective external-test design.",
        "",
    ]

    md_path = OUT_DIR / "dossier_clarity_audit.md"
    md_path.write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )

    print("=" * 118)
    print("SENIOR QUANT DOSSIER CLARITY AUDIT")
    print("=" * 118)
    print(f"Status:       {status}")
    print(f"Freeze ID:    {manifest_freeze}")
    print(f"Freeze stage: {manifest_stage}")
    print(f"Checks:       {len(checks)}")
    print(f"Failures:     {len(failures)}")
    print(f"Warnings:     {len(warnings)}")
    print()
    print(f"Report: {md_path}")

    if failures:
        print()
        for failure in failures[:20]:
            print("FAIL:", failure)
        raise SystemExit(
            "FAIL: dossier clarity audit did not pass. "
            "Reviewer package was not authorized."
        )

    print()
    print(
        "PASS: dossier is clear enough to package as the exact reviewed state."
    )


if __name__ == "__main__":
    main()
