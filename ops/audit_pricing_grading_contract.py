from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
PRICE_SCRIPT = ROOT / "scripts/15_price_markets.py"
NORMALIZE_SOURCE = ROOT / "src/nba_prop_quant/normalize.py"
GRADER = ROOT / "ops/grade_external_test_capture.py"
OUT_DIR = ROOT / "data/validation/pricing_grading_contract"

MANDATORY_SETTLEMENT_FIELDS = {
    "game_id",
    "player_id",
    "prop_type",
    "line_value",
}

ODDS_GROUP = {
    "over_odds",
    "market_over_odds",
    "under_odds",
    "market_under_odds",
}

SIDE_CANDIDATES = {
    "preferred_side",
    "preferred_side_calibrated",
    "selected_bet_side",
    "bet_side",
    "monitor_side",
}

EDGE_CANDIDATES = {
    "preferred_edge_calibrated",
    "preferred_edge",
    "selected_edge_calibrated",
    "selected_bet_edge",
    "bet_edge",
    "monitor_edge",
    "calibrated_edge_selected",
    "selected_edge",
}

EV_CANDIDATES = {
    "preferred_ev_calibrated",
    "preferred_ev",
    "selected_ev_calibrated",
    "selected_bet_ev",
    "bet_model_ev",
    "monitor_ev",
    "calibrated_ev_selected",
    "selected_ev",
}

PROBABILITY_CANDIDATES = {
    "q_over_calibrated",
    "q_selected",
    "selected_q_over",
    "calibrated_q_over",
    "q_over_selected",
    "model_q_over_calibrated",
    "q_over_raw",
    "q_over_nonpush",
    "raw_q_over",
    "model_q_over_raw",
    "p_over_calibrated",
    "p_under_calibrated",
    "p_over_raw",
    "p_under_raw",
    "p_over",
    "p_under",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.add(node.value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    for path in (PRICE_SCRIPT, GRADER):
        if not path.exists():
            raise SystemExit(f"ERROR: missing source file: {path}")

    sources = [PRICE_SCRIPT]
    if NORMALIZE_SOURCE.exists():
        sources.append(NORMALIZE_SOURCE)

    literals = set()
    source_records = []

    for source in sources:
        literals |= string_literals(source)
        source_records.append(
            {
                "path": str(source.relative_to(ROOT)),
                "sha256": sha256_file(source),
            }
        )

    mandatory_evidence = {
        field: field in literals
        for field in sorted(MANDATORY_SETTLEMENT_FIELDS)
    }

    group_evidence = {
        "odds": sorted(ODDS_GROUP & literals),
        "side": sorted(SIDE_CANDIDATES & literals),
        "edge": sorted(EDGE_CANDIDATES & literals),
        "ev": sorted(EV_CANDIDATES & literals),
        "probability": sorted(PROBABILITY_CANDIDATES & literals),
        "external_test_record": (
            ["external_test_record"]
            if "external_test_record" in literals
            else []
        ),
    }

    runtime_required = [
        field
        for field, evidenced in mandatory_evidence.items()
        if not evidenced
    ]

    warnings = []
    if runtime_required:
        warnings.append(
            "Some mandatory settlement keys are not directly evidenced as "
            "string literals in the pricing/normalization source. They may "
            "be inherited through DataFrame payload copies; confirm on the "
            "first real priced_markets.parquet runtime file."
        )

    for group in ("side", "edge", "ev", "probability"):
        if not group_evidence[group]:
            warnings.append(
                f"No recognized {group} candidate was found as a source "
                "literal. Runtime schema validation is required."
            )

    status = "PASS_WITH_RUNTIME_CONFIRMATION" if warnings else "PASS"

    now = datetime.now(timezone.utc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at_utc": now.isoformat(),
        "status": status,
        "purpose": (
            "Static contract audit between production market-pricing source "
            "and immutable grader. This does not replace first-live-file "
            "runtime schema validation."
        ),
        "source_records": source_records,
        "grader": {
            "path": str(GRADER.relative_to(ROOT)),
            "sha256": sha256_file(GRADER),
        },
        "mandatory_settlement_field_literal_evidence": mandatory_evidence,
        "recognized_group_literal_evidence": group_evidence,
        "runtime_confirmation_required_for": runtime_required,
        "warnings": warnings,
    }

    out_json = OUT_DIR / "pricing_grading_contract_audit.json"
    out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Pricing ↔ Grading Contract Audit",
        "",
        f"- Status: **{status}**",
        f"- Generated UTC: `{now.isoformat()}`",
        f"- Pricing source SHA-256: `{sha256_file(PRICE_SCRIPT)}`",
        f"- Grader SHA-256: `{sha256_file(GRADER)}`",
        "",
        "## Mandatory settlement keys",
        "",
    ]
    for field, evidenced in mandatory_evidence.items():
        lines.append(
            f"- `{field}`: "
            + (
                "source-evidenced"
                if evidenced
                else "runtime confirmation required"
            )
        )

    lines += ["", "## Recognized optional grading groups", ""]
    for group, values in group_evidence.items():
        rendered = (
            ", ".join(f"`{v}`" for v in values)
            if values
            else "none source-evidenced"
        )
        lines.append(f"- {group}: {rendered}")

    if warnings:
        lines += ["", "## Warnings", ""]
        for warning in warnings:
            lines.append(f"- {warning}")

    (
        OUT_DIR / "pricing_grading_contract_audit.md"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 118)
    print("PRICING ↔ GRADING CONTRACT AUDIT")
    print("=" * 118)
    print(f"Status: {status}")
    print(f"Pricing SHA-256: {sha256_file(PRICE_SCRIPT)}")
    print(f"Grader SHA-256:  {sha256_file(GRADER)}")
    print()
    for group, values in group_evidence.items():
        print(
            f"{group:22s}: "
            + (
                ", ".join(values)
                if values
                else "runtime confirmation required"
            )
        )
    print()
    print(
        "NOTE: source audit cannot prove the exact runtime parquet schema "
        "before the first real live priced file exists."
    )
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
