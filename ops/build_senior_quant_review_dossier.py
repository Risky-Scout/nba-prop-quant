from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.cwd()
REVIEW_DIR = ROOT / "review"
DOSSIER = REVIEW_DIR / "SENIOR_QUANT_REVIEW_DOSSIER.md"
INDEX = REVIEW_DIR / "SENIOR_QUANT_REVIEW_ARTIFACT_INDEX.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame is None or frame.empty:
        return "_Not available from the current local artifact schema._"

    frame = frame.head(max_rows).copy()
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for _, row in frame.iterrows():
        values = []
        for value in row:
            if value is None:
                values.append("")
            elif isinstance(value, float):
                if pd.isna(value):
                    values.append("")
                else:
                    values.append(f"{value:.6g}")
            else:
                try:
                    if pd.isna(value):
                        values.append("")
                        continue
                except Exception:
                    pass
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def latest_report(pattern: str) -> tuple[Path | None, dict]:
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        return None, {}
    path = paths[-1]
    try:
        return path, load_json(path)
    except Exception:
        return path, {}


def extract_selected_modes(payload: dict) -> pd.DataFrame:
    rows = []
    targets = {"pts", "reb", "ast", "stl", "blk", "fg3m"}

    def walk(obj: Any, path: str = "") -> None:
        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            key_lower = str(key).lower()

            if (
                key_lower in targets
                and isinstance(value, dict)
                and "selected_mode" in value
            ):
                weights = value.get("production_weights", {})
                rows.append(
                    {
                        "target": key_lower,
                        "selected_mode": value.get("selected_mode"),
                        "w_xgb": weights.get("xgb"),
                        "w_decay": weights.get("decay"),
                        "w_kalman": weights.get("kalman"),
                        "pooled_improvement_vs_xgb_pct": value.get(
                            "pooled_improvement_vs_xgb_pct"
                        ),
                        "positive_strict_folds": value.get(
                            "positive_strict_folds"
                        ),
                        "strict_fold_count": value.get(
                            "strict_fold_count"
                        ),
                    }
                )

            walk(
                value,
                f"{path}.{key}" if path else str(key),
            )

    walk(payload)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates("target")
        .sort_values("target")
    )


def marginal_summary(payload: dict) -> pd.DataFrame:
    rows = []
    targets = {"pts", "reb", "ast", "stl", "blk", "fg3m"}
    families = {"zinb", "nb", "poisson"}

    def walk(obj: Any) -> None:
        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            key_lower = str(key).lower()

            if key_lower in targets:
                selected = None

                if isinstance(value, str):
                    if value.lower() in families:
                        selected = value

                elif isinstance(value, dict):
                    for field in (
                        "selected",
                        "selected_distribution",
                        "selected_model",
                        "selected_family",
                        "family",
                    ):
                        candidate = value.get(field)
                        if isinstance(candidate, str):
                            selected = candidate
                            break

                if selected is not None:
                    rows.append(
                        {
                            "target": key_lower,
                            "selected_marginal": str(selected).lower(),
                        }
                    )

            walk(value)

    walk(payload)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates("target")
        .sort_values("target")
    )


def dependence_summary(payload: dict) -> pd.DataFrame:
    rows = []
    combos = {
        "points_rebounds",
        "points_assists",
        "rebounds_assists",
        "points_rebounds_assists",
        "stocks",
    }

    def walk(obj: Any, path: str = "") -> None:
        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            key_text = str(key)

            if key_text in combos:
                lam = None

                if isinstance(value, (int, float)):
                    lam = float(value)

                elif isinstance(value, dict):
                    for field in (
                        "production_lambda",
                        "selected_lambda",
                        "lambda",
                        "production_value",
                    ):
                        candidate = value.get(field)
                        if isinstance(candidate, (int, float)):
                            lam = float(candidate)
                            break

                if lam is not None:
                    rows.append(
                        {
                            "combo": key_text,
                            "lambda": lam,
                        }
                    )

            if isinstance(value, dict):
                if any(
                    field in value
                    for field in (
                        "production_lambda",
                        "selected_lambda",
                    )
                ):
                    lam = value.get(
                        "production_lambda",
                        value.get("selected_lambda"),
                    )
                    if isinstance(lam, (int, float)):
                        rows.append(
                            {
                                "combo": key_text,
                                "lambda": float(lam),
                            }
                        )

                walk(
                    value,
                    f"{path}.{key_text}" if path else key_text,
                )

    walk(payload)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates("combo")
        .sort_values("combo")
    )


def find_selected_methods(obj: Any, prefix: str = "") -> list[dict]:
    rows = []

    if isinstance(obj, dict):
        if "selected_method" in obj:
            rows.append(
                {
                    "prop_type": prefix.split(".")[-1],
                    "selected_method": obj.get("selected_method"),
                }
            )

        for key, value in obj.items():
            rows.extend(
                find_selected_methods(
                    value,
                    f"{prefix}.{key}" if prefix else str(key),
                )
            )

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            rows.extend(
                find_selected_methods(
                    value,
                    f"{prefix}[{index}]",
                )
            )

    return rows


def current_market_inventory() -> dict:
    result = {}

    raw_props = (
        ROOT
        / "data/raw/opening_props/season=2025/props.parquet"
    )
    priced = (
        ROOT
        / "data/processed/market_backtest/backtest_priced_2025.parquet"
    )

    if raw_props.exists():
        frame = pd.read_parquet(raw_props)
        result["opening_prop_rows"] = int(len(frame))

        if "market_type" in frame.columns:
            result["over_under_rows"] = int(
                frame["market_type"].eq("over_under").sum()
            )

        if "game_id" in frame.columns:
            result["opening_prop_games"] = int(
                frame["game_id"].nunique()
            )

        if "player_id" in frame.columns:
            result["opening_prop_players"] = int(
                frame["player_id"].nunique()
            )

        if "vendor" in frame.columns:
            result["opening_prop_vendors"] = int(
                frame["vendor"].nunique()
            )

        if "opened_at" in frame.columns:
            opened = pd.to_datetime(
                frame["opened_at"],
                utc=True,
                errors="coerce",
            )
            if opened.notna().any():
                result["opened_at_min_utc"] = opened.min().isoformat()
                result["opened_at_max_utc"] = opened.max().isoformat()

    if priced.exists():
        frame = pd.read_parquet(priced)
        result["priced_2025_rows"] = int(len(frame))

        if "game_id" in frame.columns:
            result["priced_2025_games"] = int(
                frame["game_id"].nunique()
            )

        if "player_id" in frame.columns:
            result["priced_2025_players"] = int(
                frame["player_id"].nunique()
            )

        if "vendor" in frame.columns:
            result["priced_2025_vendors"] = int(
                frame["vendor"].nunique()
            )

        event_cols = [
            c
            for c in ["game_id", "player_id", "prop_type"]
            if c in frame.columns
        ]

        if len(event_cols) == 3:
            result["priced_2025_events"] = int(
                frame[event_cols]
                .drop_duplicates()
                .shape[0]
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    manifest_path = ROOT / "models/frozen_manifests/LATEST.json"
    manifest = load_json(manifest_path)

    freeze_id = manifest.get("freeze_id", "UNKNOWN")
    freeze_stage = manifest.get(
        "freeze_stage",
        manifest.get("stage", "UNKNOWN"),
    )

    mean_policy_path = ROOT / "models/mean_model_selection.json"
    marginal_policy_path = ROOT / "models/marginal_selection.json"
    dependence_policy_path = ROOT / "models/combo_dependence_policy.json"
    calibration_policy_path = (
        ROOT / "models/market_probability_calibration_policy.json"
    )

    mean_table = extract_selected_modes(
        load_json(mean_policy_path)
    )
    marginal_table = marginal_summary(
        load_json(marginal_policy_path)
    )
    dependence_table = dependence_summary(
        load_json(dependence_policy_path)
    )
    calibration_table = pd.DataFrame(
        find_selected_methods(
            load_json(calibration_policy_path)
        )
    )
    if not calibration_table.empty:
        calibration_table = (
            calibration_table
            .drop_duplicates("prop_type")
            .sort_values("prop_type")
        )

    event_scoring_path = (
        ROOT
        / "data/processed/market_backtest/calibrated_oof/event_scoring_summary.csv"
    )
    event_scoring = (
        pd.read_csv(event_scoring_path)
        if event_scoring_path.exists()
        else pd.DataFrame()
    )

    bootstrap_path = (
        ROOT
        / "data/processed/market_backtest/calibrated_oof/event_game_cluster_bootstrap.csv"
    )
    bootstrap = (
        pd.read_csv(bootstrap_path)
        if bootstrap_path.exists()
        else pd.DataFrame()
    )

    synthetic_path, synthetic_report = latest_report(
        "data/validation/grading_synthetic/*/"
        "synthetic_grading_integration_report.json"
    )
    replay_path, replay_report = latest_report(
        "data/validation/grading_replay_2025/date=*/*/"
        "real_2025_grading_replay_report.json"
    )

    contract_path = (
        ROOT
        / "data/validation/pricing_grading_contract/"
        "pricing_grading_contract_audit.json"
    )
    contract_report = load_json(contract_path)

    market_inventory = current_market_inventory()

    strict_rows = None
    oof_selected_path = (
        ROOT / "data/processed/oof_selected_means.parquet"
    )
    if oof_selected_path.exists():
        try:
            strict_rows = len(
                pd.read_parquet(oof_selected_path)
            )
        except Exception:
            strict_rows = None

    artifact_paths = [
        manifest_path,
        mean_policy_path,
        marginal_policy_path,
        dependence_policy_path,
        calibration_policy_path,
        event_scoring_path,
        bootstrap_path,
        oof_selected_path,
        ROOT / "models/marginals.joblib",
        ROOT / "models/copula.joblib",
        ROOT / "scripts/10_predict_slate.py",
        ROOT / "scripts/15_price_markets.py",
        ROOT / "ops/capture_external_test_day.py",
        ROOT / "ops/grade_external_test_capture.py",
    ]

    if synthetic_path is not None:
        artifact_paths.append(synthetic_path)

    if replay_path is not None:
        artifact_paths.append(replay_path)

    if contract_path.exists():
        artifact_paths.append(contract_path)

    artifact_records = []

    for path in artifact_paths:
        if path.exists() and path.is_file():
            artifact_records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "bytes": int(path.stat().st_size),
                }
            )

    INDEX.write_text(
        json.dumps(
            {
                "generated_at_utc": now.isoformat(),
                "freeze_id": freeze_id,
                "freeze_stage": freeze_stage,
                "artifacts": artifact_records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    synthetic_status = synthetic_report.get(
        "status",
        "PENDING",
    )
    replay_status = replay_report.get(
        "status",
        "PENDING",
    )
    contract_status = contract_report.get(
        "status",
        "PENDING",
    )

    lines = [
        "# NBA Player Box-Score Prop Projection System",
        "## Senior Quant Review Dossier — 2026–27 External-Test Deployment",
        "",
        f"Generated UTC: `{now.isoformat()}`",
        "",
        "## 1. Review posture",
        "",
        "**This system is ready for expert methodology review. It is not presented as having a proven broad sportsbook edge.**",
        "",
        "The research record separates model development from the first untouched external market test. "
        "The 2025 sportsbook sample is development/model-selection evidence; 2026–27 is the first prospective external test. "
        "No automatic betting threshold was selected from the 2025 ROI grid.",
        "",
        f"- Frozen deployment ID: `{freeze_id}`",
        f"- Freeze stage: `{freeze_stage}`",
        f"- Strict selected-mean OOF rows: `{strict_rows if strict_rows is not None else 'not read'}`",
        "- Monitoring edge grid: `1%, 2%, 3%, 5%, 7.5%, 10%`",
        "- Automatic betting threshold: **none**",
        "",
        "## 2. Data and information-set design",
        "",
        "- Standard historical NBA player-game backbone: season labels 2001 through 2025.",
        "- Advanced statistics are used only where the BDL archive supports them.",
        "- BALLDONTLIE does not supply birth date in the player schema used here; production uses career-experience features rather than claiming exact biological age.",
        "- Current injury data are treated as a current snapshot. Historical injury reconstruction is not performed unless an injury snapshot was actually archived before tip-off.",
        "- Pregame feature construction is chronological; outcome fields are not allowed to enter the future row.",
        "",
        "### Current 2025 market inventory read from local artifacts",
        "",
        "```json",
        json.dumps(
            market_inventory,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "The market sample must be treated as non-random and coverage-limited. "
        "A senior reviewer should explicitly assess selection effects from the available opening-prop window, "
        "vendor mix, multiple lines per event, and quote-shopping scope.",
        "",
        "## 3. Mean-model selection",
        "",
        markdown_table(mean_table),
        "",
        "Selection is intentionally conservative: a blend is not promoted merely because pooled RMSE is marginally better.",
        "",
        "## 4. Marginal distributions",
        "",
        markdown_table(marginal_table),
        "",
        "The production distribution layer uses a mean-preserving zero-inflated negative-binomial parameterization. "
        "For a ZINB marginal, the count-component mean is adjusted so that the unconditional expected value equals the selected mean forecast.",
        "",
        "Reviewer focus: zero-inflation identification, dispersion stability, tail calibration, integer-line push mass, "
        "and whether minutes/role uncertainty should be integrated more explicitly.",
        "",
        "## 5. Combination-prop dependence",
        "",
        markdown_table(dependence_table),
        "",
        "Production dependence is intentionally sparse. Combination markets that did not clear the stability gate revert to independence rather than forcing a noisy copula effect.",
        "",
        "## 6. Probability calibration and market benchmark",
        "",
        markdown_table(calibration_table),
        "",
        "Calibration is chronological and frozen before the 2026–27 external test.",
        "",
        "### 2025 event-level proper scoring",
        "",
        markdown_table(event_scoring),
        "",
        "### Game-clustered uncertainty",
        "",
        markdown_table(bootstrap),
        "",
        "Interpretation: calibration materially improved the raw model overall, but the market remained approximately tied/slightly better in aggregate. "
        "That is not evidence of a broad proven edge.",
        "",
        "## 7. Development betting diagnostics",
        "",
        "The 2025 ROI grid is retained only as a development diagnostic. It must not be used to choose a production threshold after the fact. "
        "Quote-level, best-vendor-event, and best-event scopes answer different questions and must remain separately reported.",
        "",
        "## 8. Production architecture and governance",
        "",
        "The deployment separates historical ingestion, minutes/target means, mean-preserving marginals, combo dependence, "
        "probability calibration, live pricing, immutable prospective capture, and separate post-event grading.",
        "",
        "The frozen production manifest hashes the critical model artifacts and source files. "
        "Operational capture and grading are append-only evidence layers; original forecasts and quote snapshots are not rewritten after outcomes are known.",
        "",
        "## 9. Grading/settlement QA before external testing",
        "",
        f"- Synthetic controlled integration test: **{synthetic_status}**",
        f"- Real 2025 development-data grading replay: **{replay_status}**",
        f"- Pricing ↔ grading source-contract audit: **{contract_status}**",
        "",
        "Synthetic QA is software evidence only. The real 2025 replay is schema/settlement realism evidence only. Neither is counted as new model-performance evidence.",
        "",
        "## 10. Known limitations a senior reviewer should attack",
        "",
        "- Market-sample selection and limited opening-prop coverage.",
        "- Current injury snapshots do not reconstruct historical pre-tip information that was never archived.",
        "- Minutes uncertainty may be under-propagated into final prop tails.",
        "- DNP/missing-player outcomes are void candidates until book-specific rules are explicitly represented.",
        "- Prop-specific calibration may drift in 2026–27.",
        "- Dependence estimates can be weakly identified for small-history or role-changing players.",
        "- Best-event ROI mixes predictive signal with quote availability across vendors.",
        "- Multiple prop/vendor/threshold diagnostics create researcher degrees of freedom.",
        "- Opening-line comparison does not replace a timestamped closing-line-value study.",
        "- Early 2026–27 results require game-clustered uncertainty and should not be overinterpreted.",
        "",
        "## 11. Claims permitted vs. not permitted",
        "",
        "### Permitted",
        "",
        "- Reproducible, interpretable research/production candidate with frozen 2026–27 policies.",
        "- Walk-forward model selection and chronological probability calibration.",
        "- 2025 market results are development evidence.",
        "- Prospective capture/grading protocol preserves point-in-time evidence.",
        "",
        "### Not permitted yet",
        "",
        "- A proven broad sportsbook edge.",
        "- An optimal production betting threshold selected from 2025.",
        "- 2025 described as an untouched external market test.",
        "- Fully reconstructed historical injury context.",
        "- A claim that copula dependence improves every combo market.",
        "",
        "## 12. Suggested senior-quant review agenda",
        "",
        "1. Reproduce the frozen manifest and selected-policy tables.",
        "2. Audit point-in-time feature availability and season/date boundaries.",
        "3. Reproduce OOF minutes/target errors and conservative model-selection gates.",
        "4. Validate mean-preserving ZINB identities and tail calibration.",
        "5. Reproduce dependence-selection CV and independence fallbacks.",
        "6. Reproduce chronological calibration and market proper-scoring comparison.",
        "7. Challenge market-sample selection, vendor multiplicity, and quote-shopping assumptions.",
        "8. Audit DNP/push/settlement semantics.",
        "9. Review the prospective 2026–27 protocol before looking at external results.",
        "10. Predeclare what evidence would justify a model revision versus normal sampling noise.",
        "",
        "## 13. Reproducibility commands",
        "",
        "```bash",
        "python scripts/verify_frozen_manifest.py",
        "python scripts/10a_validate_production_contract.py",
        "python ops/qa_grading_synthetic_integration.py",
        "python ops/qa_replay_2025_grading.py",
        "python ops/audit_pricing_grading_contract.py",
        "python ops/build_senior_quant_review_dossier.py",
        "```",
        "",
        "## 14. Artifact index",
        "",
        f"Machine-readable hashes: `{INDEX.relative_to(ROOT)}`",
        "",
    ]

    for record in artifact_records:
        lines.append(
            f"- `{record['path']}` — SHA-256 `{record['sha256']}`"
        )

    lines += [
        "",
        "## 15. Reviewer sign-off",
        "",
        "- Reviewer:",
        "- Review date:",
        "- Reproduced freeze verification: YES / NO",
        "- Material leakage concern: YES / NO",
        "- Material calibration concern: YES / NO",
        "- Material distributional concern: YES / NO",
        "- Material dependence concern: YES / NO",
        "- Material market-sample concern: YES / NO",
        "- External-test protocol acceptable before results are viewed: YES / NO",
        "- Required changes before external testing:",
        "- Analyses to defer until a prespecified prospective sample exists:",
        "",
    ]

    DOSSIER.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 118)
    print("SENIOR QUANT REVIEW DOSSIER")
    print("=" * 118)
    print(f"Freeze ID:       {freeze_id}")
    print(f"Freeze stage:    {freeze_stage}")
    print(f"Synthetic QA:    {synthetic_status}")
    print(f"2025 replay QA:  {replay_status}")
    print(f"Contract audit:  {contract_status}")
    print()
    print(f"Wrote: {DOSSIER}")
    print(f"Wrote: {INDEX}")


if __name__ == "__main__":
    main()
