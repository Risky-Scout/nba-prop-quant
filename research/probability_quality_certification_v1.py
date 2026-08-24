from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import optimize, stats

MODEL_FREEZE_ID = "nba_prop_quant_20260818T205213Z"
PREREG_TAG = "nba_prop_quant_20260818T205213Z_probability_quality_prereg_v1"
PREREG_COMMIT = "5aacd0e81b578e2edbc0255f6f3afb39fcebbecb"
PROVENANCE_COMMIT = "5594625fb1fe00fda8c2eb13891b34b416de9203"

LOG_EPS = 1e-6
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260823
MIN_CONTRACTS = 250
MIN_GAMES = 30

CONTRACT_KEYS = ["game_id", "player_id", "prop_type", "line_value"]
TARGET_ORDER = ["ast", "blk", "fg3m", "pts", "reb", "stl"]

CONFIDENCE_BINS = [0.50, 0.525, 0.55, 0.60, 0.65, 0.70, 1.0000001]
CONFIDENCE_LABELS = [
    "0.5000-<0.5250",
    "0.5250-<0.5500",
    "0.5500-<0.6000",
    "0.6000-<0.6500",
    "0.6500-<0.7000",
    "0.7000-1.0000",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probability Quality Certification v1 for the frozen NBA prop model."
    )
    default_project = Path(os.environ["PROJECT_ROOT"]) if "PROJECT_ROOT" in os.environ else None
    p.add_argument("--project-root", type=Path, default=default_project, required=default_project is None)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/probability_quality_certification_v1_outputs"),
    )
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def inferential_status(rows: int, games: int) -> str:
    return "inferential" if rows >= MIN_CONTRACTS and games >= MIN_GAMES else "descriptive_only"


def validate_contracts(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id", "player_id", "prop_type", "line_value", "actual_over",
        "q_model", "q_market", "q_selected", "game_date", "selected_method",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Contract table missing required columns: {missing}")

    if df[CONTRACT_KEYS].duplicated().any():
        dup = int(df[CONTRACT_KEYS].duplicated(keep=False).sum())
        raise RuntimeError(f"Contract table contains {dup} duplicated logical contracts.")

    out = df.copy()
    out["y"] = pd.to_numeric(out["actual_over"], errors="raise").astype(float)
    for col in ["q_model", "q_market", "q_selected"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
        if not out[col].between(0.0, 1.0).all():
            raise RuntimeError(f"{col} contains values outside [0,1].")

    if not out["y"].isin([0.0, 1.0]).all():
        raise RuntimeError("actual_over contains non-binary outcomes.")

    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise")
    out["season"] = 2025
    return out


def add_losses(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    y = out["y"].to_numpy(float)
    for label, pcol in {
        "raw": "q_model",
        "selected": "q_selected",
        "market": "q_market",
    }.items():
        p = out[pcol].to_numpy(float)
        out[f"brier_{label}"] = (p - y) ** 2
        pc = np.clip(p, LOG_EPS, 1.0 - LOG_EPS)
        out[f"logloss_{label}"] = -(
            y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)
        )

    out["delta_brier_selected_market"] = out["brier_selected"] - out["brier_market"]
    out["delta_logloss_selected_market"] = out["logloss_selected"] - out["logloss_market"]
    return out


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float, bool]:
    pc = np.clip(np.asarray(p, float), LOG_EPS, 1.0 - LOG_EPS)
    yy = np.asarray(y, float)
    x = np.log(pc / (1.0 - pc))

    def objective(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * x
        return float(np.sum(np.logaddexp(0.0, eta) - yy * eta))

    res = optimize.minimize(
        objective,
        x0=np.array([0.0, 1.0]),
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 1000},
    )
    return float(res.x[0]), float(res.x[1]), bool(res.success)


def reliability_deciles(df: pd.DataFrame, pcol: str, group_fields: dict[str, Any]) -> list[dict[str, Any]]:
    work = df[["y", pcol]].copy()
    work["bin"] = pd.qcut(work[pcol], q=10, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for idx, (bin_value, g) in enumerate(work.groupby("bin", observed=True, sort=True), start=1):
        pmean = float(g[pcol].mean())
        ymean = float(g["y"].mean())
        rows.append({
            **group_fields,
            "bin_index": idx,
            "bin": str(bin_value),
            "rows": int(len(g)),
            "mean_probability": pmean,
            "observed_over_rate": ymean,
            "absolute_gap": abs(pmean - ymean),
        })
    return rows


def fixed_width_reliability(df: pd.DataFrame, pcol: str, group_fields: dict[str, Any]) -> list[dict[str, Any]]:
    work = df[["y", pcol]].copy()
    work["bin"] = pd.cut(work[pcol], bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
    rows: list[dict[str, Any]] = []
    for idx, (bin_value, g) in enumerate(work.groupby("bin", observed=False, sort=True), start=1):
        if g.empty:
            continue
        pmean = float(g[pcol].mean())
        ymean = float(g["y"].mean())
        rows.append({
            **group_fields,
            "bin_index": idx,
            "bin": str(bin_value),
            "rows": int(len(g)),
            "mean_probability": pmean,
            "observed_over_rate": ymean,
            "absolute_gap": abs(pmean - ymean),
        })
    return rows


def ece_from_rows(rows: list[dict[str, Any]]) -> float:
    n = sum(int(r["rows"]) for r in rows)
    if n == 0:
        return float("nan")
    return float(sum((int(r["rows"]) / n) * float(r["absolute_gap"]) for r in rows))


def brier_decomposition(df: pd.DataFrame, pcol: str) -> dict[str, float]:
    rel = reliability_deciles(df, pcol, {})
    ybar = float(df["y"].mean())
    n = len(df)
    reliability = sum((r["rows"] / n) * (r["mean_probability"] - r["observed_over_rate"]) ** 2 for r in rel)
    resolution = sum((r["rows"] / n) * (r["observed_over_rate"] - ybar) ** 2 for r in rel)
    uncertainty = ybar * (1.0 - ybar)
    return {
        "reliability_decile_approx": float(reliability),
        "resolution_decile_approx": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_from_decile_decomposition_approx": float(uncertainty - resolution + reliability),
    }


def probability_metrics(df: pd.DataFrame, scope: str, group_value: str) -> dict[str, Any]:
    y = df["y"].to_numpy(float)
    p = df["q_selected"].to_numpy(float)
    intercept, slope, converged = calibration_intercept_slope(y, p)
    rel_rows = reliability_deciles(df, "q_selected", {})
    decomp = brier_decomposition(df, "q_selected")
    games = int(df["game_id"].nunique())

    return {
        "scope": scope,
        "group_value": group_value,
        "rows": int(len(df)),
        "games": games,
        "players": int(df["player_id"].nunique()),
        "inferential_status": inferential_status(int(len(df)), games),
        "brier_raw": float(df["brier_raw"].mean()),
        "brier_selected": float(df["brier_selected"].mean()),
        "brier_market": float(df["brier_market"].mean()),
        "logloss_raw": float(df["logloss_raw"].mean()),
        "logloss_selected": float(df["logloss_selected"].mean()),
        "logloss_market": float(df["logloss_market"].mean()),
        "delta_brier_selected_market": float(df["delta_brier_selected_market"].mean()),
        "delta_logloss_selected_market": float(df["delta_logloss_selected_market"].mean()),
        "mean_probability_selected": float(df["q_selected"].mean()),
        "observed_over_rate": float(df["y"].mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_optimizer_converged": converged,
        "ece_deciles": ece_from_rows(rel_rows),
        "mean_abs_distance_from_0_5": float(np.mean(np.abs(p - 0.5))),
        "probability_std": float(np.std(p)),
        "probability_min": float(np.min(p)),
        "probability_p05": float(np.quantile(p, 0.05)),
        "probability_p25": float(np.quantile(p, 0.25)),
        "probability_median": float(np.median(p)),
        "probability_p75": float(np.quantile(p, 0.75)),
        "probability_p95": float(np.quantile(p, 0.95)),
        "probability_max": float(np.max(p)),
        **decomp,
    }


def confidence_bucket_rows(df: pd.DataFrame, scope: str, group_value: str) -> list[dict[str, Any]]:
    work = df.copy()
    p = work["q_selected"].to_numpy(float)
    work["confidence"] = np.maximum(p, 1.0 - p)
    work["predicted_over"] = (p >= 0.5).astype(float)
    work["correct"] = (work["predicted_over"] == work["y"]).astype(float)
    work["confidence_bucket"] = pd.cut(
        work["confidence"],
        bins=CONFIDENCE_BINS,
        labels=CONFIDENCE_LABELS,
        right=False,
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    for bucket, g in work.groupby("confidence_bucket", observed=False, sort=True):
        if g.empty:
            continue
        games = int(g["game_id"].nunique())
        rows.append({
            "scope": scope,
            "group_value": group_value,
            "confidence_bucket": str(bucket),
            "rows": int(len(g)),
            "games": games,
            "inferential_status": inferential_status(int(len(g)), games),
            "mean_confidence": float(g["confidence"].mean()),
            "realized_correctness_rate": float(g["correct"].mean()),
            "calibration_gap": float(g["correct"].mean() - g["confidence"].mean()),
            "brier_selected": float(g["brier_selected"].mean()),
            "logloss_selected": float(g["logloss_selected"].mean()),
        })
    return rows


def stable_seed(offset: int) -> int:
    return int((BOOTSTRAP_SEED + 1009 * offset) % (2**32 - 1))


def cluster_bootstrap_mean(df: pd.DataFrame, metric: str, reps: int, seed: int) -> dict[str, Any]:
    game = df.groupby("game_id")[metric].agg(["sum", "count"])
    point = float(df[metric].mean())
    if len(game) < 2:
        return {"point": point, "ci_low": np.nan, "ci_high": np.nan, "p_lt_zero": np.nan, "games": int(len(game))}

    sums = game["sum"].to_numpy(float)
    counts = game["count"].to_numpy(float)
    n = len(game)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)

    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        vals[i] = sums[idx].sum() / counts[idx].sum()

    return {
        "point": point,
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "p_lt_zero": float(np.mean(vals < 0.0)),
        "games": int(n),
    }


def bootstrap_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    groups = [("overall", "ALL", df)]
    groups += [("prop", str(prop), g) for prop, g in df.groupby("prop_type", sort=True)]

    seed_index = 0
    for scope, value, g in groups:
        for metric in ["delta_brier_selected_market", "delta_logloss_selected_market"]:
            r = cluster_bootstrap_mean(g, metric, BOOTSTRAP_REPS, stable_seed(seed_index))
            seed_index += 1
            games = int(g["game_id"].nunique())
            comparison_rows.append({
                "scope": scope,
                "group_value": value,
                "metric": metric,
                "rows": int(len(g)),
                "games": games,
                "inferential_status": inferential_status(int(len(g)), games),
                "point": r["point"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "p_model_better": r["p_lt_zero"],
            })

        for metric in ["brier_selected", "logloss_selected", "brier_market", "logloss_market"]:
            r = cluster_bootstrap_mean(g, metric, BOOTSTRAP_REPS, stable_seed(seed_index))
            seed_index += 1
            score_rows.append({
                "scope": scope,
                "group_value": value,
                "metric": metric,
                "rows": int(len(g)),
                "games": int(g["game_id"].nunique()),
                "point": r["point"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
            })

    return pd.DataFrame(comparison_rows), pd.DataFrame(score_rows)


def push_and_three_outcome(quote_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    cols = [
        "game_id", "player_id", "prop_type", "line_value",
        "actual_over", "actual_push",
        "p_over", "p_under", "p_push",
        "p_selected_over", "p_selected_under",
    ]
    q = pd.read_parquet(quote_path, columns=cols)
    c = q.groupby(CONTRACT_KEYS, as_index=False, dropna=False).agg({
        "actual_over": "first",
        "actual_push": "first",
        "p_over": "mean",
        "p_under": "mean",
        "p_push": "mean",
        "p_selected_over": "mean",
        "p_selected_under": "mean",
    })

    raw_sum = c["p_over"] + c["p_under"] + c["p_push"]
    selected_sum = c["p_selected_over"] + c["p_selected_under"] + c["p_push"]
    integrity = {
        "quote_rows": int(len(q)),
        "collapsed_contracts": int(len(c)),
        "raw_probability_sum_max_abs_error": float(np.max(np.abs(raw_sum.to_numpy(float) - 1.0))),
        "selected_probability_sum_max_abs_error": float(np.max(np.abs(selected_sum.to_numpy(float) - 1.0))),
        "raw_probability_sum_within_1e_10": bool(np.allclose(raw_sum.to_numpy(float), 1.0, atol=1e-10, rtol=0.0)),
        "selected_probability_sum_within_1e_10": bool(np.allclose(selected_sum.to_numpy(float), 1.0, atol=1e-10, rtol=0.0)),
        "actual_push_rate": float(c["actual_push"].mean()),
    }

    push_mask = c["actual_push"].astype(float).eq(1.0)
    c["actual_over_three"] = np.where(
        push_mask, 0.0, c["actual_over"].astype(float)
    )
    c["actual_under"] = np.where(
        push_mask, 0.0, 1.0 - c["actual_over"].astype(float)
    )
    probs = c[["p_selected_over", "p_selected_under", "p_push"]].to_numpy(float)
    outcomes = np.column_stack([
        c["actual_over_three"].astype(float).to_numpy(),
        c["actual_under"].astype(float).to_numpy(),
        c["actual_push"].astype(float).to_numpy(),
    ])
    if not np.allclose(outcomes.sum(axis=1), 1.0, atol=0.0, rtol=0.0):
        raise RuntimeError("Three-outcome observed categories do not sum to 1.")

    c["multiclass_brier_selected"] = np.sum((probs - outcomes) ** 2, axis=1)
    actual_prob = np.sum(probs * outcomes, axis=1)
    c["multiclass_logscore_selected"] = -np.log(np.clip(actual_prob, LOG_EPS, 1.0))

    rows = []
    groups = [("overall", "ALL", c)]
    groups += [("prop", str(prop), g) for prop, g in c.groupby("prop_type", sort=True)]
    for scope, value, g in groups:
        rows.append({
            "scope": scope,
            "group_value": value,
            "rows": int(len(g)),
            "games": int(g["game_id"].nunique()),
            "push_rate": float(g["actual_push"].mean()),
            "multiclass_brier_selected": float(g["multiclass_brier_selected"].mean()),
            "multiclass_logscore_selected": float(g["multiclass_logscore_selected"].mean()),
        })

    return integrity, pd.DataFrame(rows)


def randomized_pit(project_root: Path, repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    marginals = joblib.load(project_root / "models/marginals_pre2025.joblib")
    frame = pd.read_parquet(project_root / "data/processed/selected_means_distribution_split.parquet")
    holdout = frame.loc[frame["distribution_split"].astype(str).eq("holdout_2025")].copy()
    if holdout.empty:
        raise RuntimeError("No holdout_2025 rows found for PIT.")

    summaries: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    all_u: list[np.ndarray] = []

    for idx, target in enumerate(TARGET_ORDER):
        fitted = marginals[target]
        mu_col = f"mu_selected_{target}"
        g = holdout.loc[holdout[target].notna() & holdout[mu_col].notna()].copy()
        y = g[target].to_numpy(dtype=int)
        mu = g[mu_col].to_numpy(dtype=float)

        p_y = np.asarray(fitted.pmf(y, mu, g), dtype=float)
        f_prev = np.asarray(fitted.cdf(y - 1, mu, g), dtype=float)
        rng = np.random.default_rng(stable_seed(100 + idx))
        u = np.clip(f_prev + rng.random(len(g)) * p_y, 0.0, 1.0)
        all_u.append(u)

        ks = stats.kstest(u, "uniform")
        cvm = stats.cramervonmises(u, "uniform")
        summaries.append({
            "scope": "target",
            "target": target,
            "rows": int(len(g)),
            "games": int(g["game_id"].nunique()),
            "pit_mean": float(np.mean(u)),
            "pit_variance": float(np.var(u)),
            "pit_p05": float(np.quantile(u, 0.05)),
            "pit_p25": float(np.quantile(u, 0.25)),
            "pit_median": float(np.quantile(u, 0.50)),
            "pit_p75": float(np.quantile(u, 0.75)),
            "pit_p95": float(np.quantile(u, 0.95)),
            "ks_statistic_descriptive": float(ks.statistic),
            "ks_pvalue_descriptive_iid": float(ks.pvalue),
            "cvm_statistic_descriptive": float(cvm.statistic),
            "cvm_pvalue_descriptive_iid": float(cvm.pvalue),
            "iid_pvalue_warning": "Descriptive only; player-game observations are clustered.",
        })

        counts, edges = np.histogram(u, bins=np.linspace(0.0, 1.0, 11))
        for b, count in enumerate(counts):
            histogram_rows.append({
                "scope": "target",
                "target": target,
                "bin_index": b + 1,
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "rows": int(count),
                "expected_uniform_rows": float(len(u) / 10.0),
            })

    combined = np.concatenate(all_u)
    ks = stats.kstest(combined, "uniform")
    cvm = stats.cramervonmises(combined, "uniform")
    summaries.append({
        "scope": "combined_descriptive",
        "target": "ALL",
        "rows": int(len(combined)),
        "games": int(holdout["game_id"].nunique()),
        "pit_mean": float(np.mean(combined)),
        "pit_variance": float(np.var(combined)),
        "pit_p05": float(np.quantile(combined, 0.05)),
        "pit_p25": float(np.quantile(combined, 0.25)),
        "pit_median": float(np.quantile(combined, 0.50)),
        "pit_p75": float(np.quantile(combined, 0.75)),
        "pit_p95": float(np.quantile(combined, 0.95)),
        "ks_statistic_descriptive": float(ks.statistic),
        "ks_pvalue_descriptive_iid": float(ks.pvalue),
        "cvm_statistic_descriptive": float(cvm.statistic),
        "cvm_pvalue_descriptive_iid": float(cvm.pvalue),
        "iid_pvalue_warning": "Descriptive only; targets and player-game observations are clustered.",
    })
    counts, edges = np.histogram(combined, bins=np.linspace(0.0, 1.0, 11))
    for b, count in enumerate(counts):
        histogram_rows.append({
            "scope": "combined_descriptive",
            "target": "ALL",
            "bin_index": b + 1,
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "rows": int(count),
            "expected_uniform_rows": float(len(combined) / 10.0),
        })

    return pd.DataFrame(summaries), pd.DataFrame(histogram_rows)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir = output_dir.resolve()

    if output_dir.exists():
        raise SystemExit(f"ERROR: output directory already exists: {output_dir}")

    current_branch = git(repo_root, "branch", "--show-current")
    current_commit = git(repo_root, "rev-parse", "HEAD")
    if current_branch != "research/probability-quality-cert-v1":
        raise SystemExit("ERROR: runner must execute on research/probability-quality-cert-v1.")
    ancestor_check = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", PROVENANCE_COMMIT, current_commit],
        capture_output=True,
        text=True,
    )
    if ancestor_check.returncode != 0:
        raise SystemExit(
            f"ERROR: provenance commit {PROVENANCE_COMMIT} must be an ancestor of HEAD {current_commit}."
        )
    if git(repo_root, "status", "--short"):
        raise SystemExit("ERROR: worktree must be clean before certification.")

    contracts_path = project_root / "data/processed/market_backtest/calibrated_oof/selected_oof_contracts.parquet"
    quote_path = project_root / "data/processed/market_backtest/calibrated_oof/selected_oof_quote_rows.parquet"
    holdout_path = project_root / "data/processed/selected_means_distribution_split.parquet"
    marginal_path = project_root / "models/marginals_pre2025.joblib"
    manifest_path = project_root / "models/frozen_manifests/nba_prop_quant_20260818T205213Z.json"

    inputs = [
        contracts_path,
        quote_path,
        holdout_path,
        marginal_path,
        manifest_path,
        repo_root / "research/PROBABILITY_QUALITY_CERTIFICATION_PROTOCOL_V1.md",
        repo_root / "research/probability_quality_certification_v1.json",
        repo_root / "research/PROBABILITY_QUALITY_ARTIFACT_COVERAGE_V1.md",
    ]
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"ERROR: missing required frozen input: {path}")

    output_dir.mkdir(parents=True)

    contracts_df = add_losses(validate_contracts(pd.read_parquet(contracts_path)))

    overall_metrics = pd.DataFrame([probability_metrics(contracts_df, "overall", "ALL")])
    prop_metrics = pd.DataFrame([
        probability_metrics(g, "prop", str(prop))
        for prop, g in contracts_df.groupby("prop_type", sort=True)
    ])
    season_metrics = pd.DataFrame([probability_metrics(contracts_df, "season", "2025")])
    season_prop_metrics = pd.DataFrame([
        probability_metrics(g, "season_x_prop", f"2025::{prop}")
        for prop, g in contracts_df.groupby("prop_type", sort=True)
    ])

    reliability_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    groups = [("overall", "ALL", contracts_df)]
    groups += [("prop", str(prop), g) for prop, g in contracts_df.groupby("prop_type", sort=True)]

    for scope, value, g in groups:
        reliability_rows.extend(reliability_deciles(g, "q_selected", {"scope": scope, "group_value": value}))
        fixed_rows.extend(fixed_width_reliability(g, "q_selected", {"scope": scope, "group_value": value}))
        confidence_rows.extend(confidence_bucket_rows(g, scope, value))

    comparison_bootstrap, score_bootstrap = bootstrap_tables(contracts_df)
    push_integrity, three_outcome = push_and_three_outcome(quote_path)
    pit_summary, pit_hist = randomized_pit(project_root, repo_root)

    outputs = {
        "overall_metrics.csv": overall_metrics,
        "prop_metrics.csv": prop_metrics,
        "season_metrics.csv": season_metrics,
        "season_prop_metrics.csv": season_prop_metrics,
        "reliability_deciles.csv": pd.DataFrame(reliability_rows),
        "reliability_fixed_width.csv": pd.DataFrame(fixed_rows),
        "confidence_buckets.csv": pd.DataFrame(confidence_rows),
        "bootstrap_model_vs_market.csv": comparison_bootstrap,
        "bootstrap_score_intervals.csv": score_bootstrap,
        "three_outcome_metrics.csv": three_outcome,
        "randomized_pit_summary.csv": pit_summary,
        "randomized_pit_histogram.csv": pit_hist,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    write_json(output_dir / "push_integrity.json", push_integrity)
    write_json(output_dir / "INPUT_MANIFEST.json", {
        "schema_version": 1,
        "model_freeze_id": MODEL_FREEZE_ID,
        "preregistration_tag": PREREG_TAG,
        "preregistration_commit": PREREG_COMMIT,
        "pre_analysis_provenance_commit": PROVENANCE_COMMIT,
        "branch": current_branch,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "log_clip_epsilon": LOG_EPS,
        "brier_probability_clipping": False,
        "model_tuning_performed": False,
        "inputs": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in inputs
        ],
    })

    result_files = sorted(p for p in output_dir.iterdir() if p.is_file())
    write_json(output_dir / "RESULT_MANIFEST.json", {
        "schema_version": 1,
        "model_freeze_id": MODEL_FREEZE_ID,
        "certification": "probability_quality_certification_v1",
        "evaluation_only": True,
        "model_tuning_performed": False,
        "coverage": {
            "binary_line_level_probability_certification": "2025_only",
            "2018_2024_line_level_probability_certification": "NOT_AVAILABLE_FROM_FROZEN_ARTIFACTS",
            "randomized_pit": "SUPPORTED_FOR_2025_HOLDOUT",
        },
        "files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in result_files
        ],
    })

    checksum_files = sorted(p for p in output_dir.iterdir() if p.is_file())
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in checksum_files),
        encoding="utf-8",
    )

    print("=" * 100)
    print("PROBABILITY QUALITY CERTIFICATION V1")
    print("=" * 100)
    print(f"Model freeze:              {MODEL_FREEZE_ID}")
    print(f"Preregistration commit:    {PREREG_COMMIT}")
    print(f"Pre-analysis commit:       {PROVENANCE_COMMIT}")
    print(f"Contracts:                 {len(contracts_df):,}")
    print(f"Games:                     {contracts_df['game_id'].nunique():,}")
    print(f"Props:                     {contracts_df['prop_type'].nunique():,}")
    print(f"Bootstrap repetitions:     {BOOTSTRAP_REPS:,}")
    print("Randomized PIT:            SUPPORTED_FOR_2025_HOLDOUT")
    print(f"Output:                    {output_dir}")
    print("Model tuning performed:    NO")
    print("PASS: certification metrics generated.")


if __name__ == "__main__":
    main()
