from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
RAW_DIRECT = ("q_over_raw","q_over_nonpush","raw_q_over","model_q_over_raw")
SEL_DIRECT = ("q_over_selected","selected_q_over","q_selected","q_over_calibrated","calibrated_q_over","model_q_over_calibrated")
MKT_DIRECT = ("market_q_over","market_devig_p_over","market_q_over_nonpush","market_devig_over")
RAW_PAIRS = (("p_over_raw","p_under_raw"),("p_over","p_under"),("raw_p_over","raw_p_under"))
SEL_PAIRS = (("p_over_selected","p_under_selected"),("p_over_calibrated","p_under_calibrated"),("calibrated_p_over","calibrated_p_under"))
MKT_PAIRS = (("market_devig_p_over","market_devig_p_under"),("market_p_over","market_p_under"),("market_q_over","market_q_under"))
PROP_COLS = ("prop_type","prop","market_name")
LINE_COLS = ("line_value","line","market_line")
ACTUAL_COLS = ("actual","actual_value","observed","outcome_value")
ACTUAL_OVER_COLS = ("actual_over","y_over","over_hit")
DATE_COLS = ("date","game_date","slate_date","event_date")
VENDOR_COLS = ("vendor","book","sportsbook")
MIN_COLS = ("expected_minutes","minutes_pred","predicted_minutes","mu_minutes")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("research_outputs/proper_scoring_audit_2025"))
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--bins", type=int, default=10)
    return p.parse_args()


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def first(cols, candidates):
    s = set(cols)
    return next((c for c in candidates if c in s), None)


def q_from_pair(df, pairs):
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            x = pd.to_numeric(df[a], errors="coerce")
            y = pd.to_numeric(df[b], errors="coerce")
            d = x + y
            return (x / d).where(d > 0).clip(0,1), f"{a}/({a}+{b})"
    return None, None


def resolve_prob(df, direct, pairs):
    c = first(df.columns, direct)
    if c:
        return pd.to_numeric(df[c], errors="coerce").clip(0,1), c
    return q_from_pair(df, pairs)


def auto_project_root(repo):
    for c in [repo.parents[2], repo.parents[1], repo.parent]:
        if (c/"data/processed").exists() and (c/"models").exists():
            return c.resolve()
    raise SystemExit("ERROR: pass --project-root; full project root not auto-detected.")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_columns(path):
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(path).columns)


def score_cols(cols):
    s=set(cols); n=0
    if {"game_id","player_id"} <= s: n += 20
    if any(c in s for c in PROP_COLS): n += 5
    if any(c in s for c in LINE_COLS): n += 5
    if any(c in s for c in ACTUAL_COLS+ACTUAL_OVER_COLS): n += 15
    for d,pairs in [(RAW_DIRECT,RAW_PAIRS),(SEL_DIRECT,SEL_PAIRS),(MKT_DIRECT,MKT_PAIRS)]:
        if any(c in s for c in d): n += 20
        elif any(a in s and b in s for a,b in pairs): n += 18
    return n


def discover(project):
    candidates = sorted(set(
        project.glob("data/processed/market_backtest/calibrated_oof/**/*.parquet")
    ) | set(
        project.glob("data/processed/market_backtest/**/*.parquet")
    ))
    rows=[]
    for p in candidates:
        try:
            cols=parquet_columns(p)
            rows.append({"path":str(p),"score":score_cols(cols),"bytes":p.stat().st_size,"columns":cols})
        except Exception as e:
            rows.append({"path":str(p),"score":-1,"bytes":p.stat().st_size,"error":repr(e),"columns":[]})
    viable=[r for r in rows if r["score"] >= 70]
    if not viable:
        raise SystemExit("ERROR: no suitable calibrated row-level parquet discovered.")
    viable.sort(key=lambda r:(r["score"],r["bytes"]), reverse=True)
    return Path(viable[0]["path"]), rows


def normalize(df):
    out=df.copy()
    prop=first(out.columns,PROP_COLS); line=first(out.columns,LINE_COLS)
    if not prop or not line: raise SystemExit("ERROR: prop/line fields missing.")
    out["prop_type"]=out[prop].astype(str)
    out["line_value"]=pd.to_numeric(out[line],errors="coerce")

    ao=first(out.columns,ACTUAL_OVER_COLS); av=first(out.columns,ACTUAL_COLS)
    if ao:
        out["y"]=pd.to_numeric(out[ao],errors="coerce")
    elif av:
        a=pd.to_numeric(out[av],errors="coerce")
        out["y"]=np.where(a>out["line_value"],1.0,np.where(a<out["line_value"],0.0,np.nan))
    else:
        raise SystemExit("ERROR: actual outcome not found.")

    raw,raw_src=resolve_prob(out,RAW_DIRECT,RAW_PAIRS)
    sel,sel_src=resolve_prob(out,SEL_DIRECT,SEL_PAIRS)
    mkt,mkt_src=resolve_prob(out,MKT_DIRECT,MKT_PAIRS)
    if raw is None: raise SystemExit("ERROR: raw model probability missing.")
    if sel is None: raise SystemExit("ERROR: selected/calibrated probability missing.")
    if mkt is None: raise SystemExit("ERROR: market probability missing.")
    out["p_raw"]=raw; out["p_selected"]=sel; out["p_market"]=mkt

    dcol=first(out.columns,DATE_COLS)
    out["audit_date"]=pd.to_datetime(out[dcol],errors="coerce").dt.normalize() if dcol else pd.NaT
    vcol=first(out.columns,VENDOR_COLS)
    out["vendor"]=out[vcol].astype(str) if vcol else "NA"
    mcol=first(out.columns,MIN_COLS)
    out["expected_minutes_audit"]=pd.to_numeric(out[mcol],errors="coerce") if mcol else np.nan

    out["game_id"]=pd.to_numeric(out["game_id"],errors="coerce")
    out["player_id"]=pd.to_numeric(out["player_id"],errors="coerce")

    valid=(out["game_id"].notna() & out["player_id"].notna() & out["line_value"].notna()
           & out["y"].isin([0.0,1.0])
           & out["p_raw"].between(0,1) & out["p_selected"].between(0,1) & out["p_market"].between(0,1))
    out=out.loc[valid].copy()
    if out.empty: raise SystemExit("ERROR: no valid non-push rows.")

    diff=out["p_selected"]-out["p_market"]
    out["preferred_side"]=np.where(diff>0,"Over",np.where(diff<0,"Under","Tie"))
    out["abs_disagreement"]=diff.abs()
    out["disagreement_bucket"]=pd.cut(
        out["abs_disagreement"],
        [0,.01,.02,.03,.05,.075,.10,np.inf],
        labels=["<1%","1-2%","2-3%","3-5%","5-7.5%","7.5-10%","10%+"],
        right=False, include_lowest=True
    ).astype(str)
    out["market_prob_bucket"]=pd.cut(
        out["p_market"],
        [0,.35,.40,.45,.50,.55,.60,.65,1.000001],
        labels=["<35%","35-40%","40-45%","45-50%","50-55%","55-60%","60-65%","65%+"],
        right=False, include_lowest=True
    ).astype(str)
    out["month"]=out["audit_date"].dt.to_period("M").astype(str) if out["audit_date"].notna().any() else "NA"
    if out["expected_minutes_audit"].notna().any():
        out["minutes_bucket"]=pd.cut(
            out["expected_minutes_audit"],[-np.inf,12,20,28,34,40,np.inf],
            labels=["<12","12-20","20-28","28-34","34-40","40+"],right=False
        ).astype(str)
    else:
        out["minutes_bucket"]="NA"
    return out, {"raw":raw_src,"selected":sel_src,"market":mkt_src,"date":dcol,"vendor":vcol,"minutes":mcol}


def add_losses(df):
    out=df.copy()
    for label,pcol in [("raw","p_raw"),("selected","p_selected"),("market","p_market")]:
        p=out[pcol].clip(EPS,1-EPS); y=out["y"]
        out[f"brier_{label}"]=(p-y)**2
        out[f"logloss_{label}"]=-(y*np.log(p)+(1-y)*np.log(1-p))
    out["dbrier"]=out["brier_selected"]-out["brier_market"]
    out["dll"]=out["logloss_selected"]-out["logloss_market"]
    return out


def contracts(df):
    keys=["game_id","player_id","prop_type","line_value"]
    agg={c:"mean" for c in ["p_raw","p_selected","p_market","expected_minutes_audit"]}
    agg.update({"y":"first","audit_date":"first","preferred_side":"first",
                "disagreement_bucket":"first","market_prob_bucket":"first",
                "month":"first","minutes_bucket":"first"})
    return add_losses(df.groupby(keys,as_index=False,dropna=False).agg(agg))


def summarize(df,scope,group_type,group_col=None):
    groups=[("ALL",df)] if group_col is None else list(df.groupby(group_col,dropna=False,sort=False))
    rows=[]
    for value,g in groups:
        if g.empty: continue
        rows.append({
            "scope":scope,"group_type":group_type,"group_value":str(value),
            "rows":len(g),"games":g.game_id.nunique(),"players":g.player_id.nunique(),
            "brier_raw":g.brier_raw.mean(),"brier_selected":g.brier_selected.mean(),"brier_market":g.brier_market.mean(),
            "logloss_raw":g.logloss_raw.mean(),"logloss_selected":g.logloss_selected.mean(),"logloss_market":g.logloss_market.mean(),
            "delta_brier_selected_market":g.dbrier.mean(),"delta_logloss_selected_market":g.dll.mean(),
            "observed_over_rate":g.y.mean(),"mean_p_raw":g.p_raw.mean(),"mean_p_selected":g.p_selected.mean(),"mean_p_market":g.p_market.mean(),
        })
    return rows


def cluster_bootstrap(df, metric, reps, seed):
    game=df.groupby("game_id")[metric].agg(["sum","count"])
    point=float(df[metric].mean())
    if len(game)<2:
        return point,np.nan,np.nan,np.nan,len(game)
    sums=game["sum"].to_numpy(float); counts=game["count"].to_numpy(float)
    rng=np.random.default_rng(seed); n=len(game); vals=np.empty(reps)
    for i in range(reps):
        idx=rng.integers(0,n,size=n)
        vals[i]=sums[idx].sum()/counts[idx].sum()
    return point,float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5)),float(np.mean(vals<0)),n


def bootstrap_table(df,reps,seed):
    specs=[("overall",None),("prop_type","prop_type"),("preferred_side","preferred_side"),("disagreement","disagreement_bucket")]
    rows=[]; s=seed
    for gt,gc in specs:
        groups=[("ALL",df)] if gc is None else list(df.groupby(gc,dropna=False,sort=False))
        for value,g in groups:
            if len(g)<20 or g.game_id.nunique()<5: continue
            for metric in ("dbrier","dll"):
                point,lo,hi,pbetter,ng=cluster_bootstrap(g,metric,reps,s); s+=1
                rows.append({"group_type":gt,"group_value":str(value),"metric":metric,"rows":len(g),"games":ng,
                             "point":point,"ci_low":lo,"ci_high":hi,"p_model_better":pbetter})
    return pd.DataFrame(rows)


def calibration(df,pcol,bins):
    work=df[["y",pcol]].copy()
    work["bin"]=pd.cut(work[pcol],np.linspace(0,1,bins+1),include_lowest=True)
    rows=[]
    for b,g in work.groupby("bin",observed=False):
        if g.empty: continue
        rows.append({"bin":str(b),"rows":len(g),"mean_probability":g[pcol].mean(),"observed_over_rate":g.y.mean(),
                     "absolute_gap":abs(g[pcol].mean()-g.y.mean())})
    return rows


def ece(df,pcol,bins):
    rows=calibration(df,pcol,bins); n=len(df)
    return float(sum(r["rows"]/n*r["absolute_gap"] for r in rows))


def main():
    args=parse_args()
    branch=git("branch","--show-current")
    commit=git("rev-parse","HEAD")
    if not branch.startswith("research/"):
        raise SystemExit(f"ERROR: run only on research/* branch; current={branch!r}")

    repo=Path.cwd().resolve()
    project=args.project_root.resolve() if args.project_root else auto_project_root(repo)
    outdir=args.output_dir if args.output_dir.is_absolute() else repo/args.output_dir
    outdir.mkdir(parents=True,exist_ok=True)

    verifier=project/"scripts/verify_frozen_manifest.py"
    py=project/".venv/bin/python"
    if verifier.exists() and py.exists():
        vr=subprocess.run([str(py),str(verifier)],cwd=project,capture_output=True,text=True)
        (outdir/"frozen_manifest_verification.txt").write_text(vr.stdout+"\n"+vr.stderr,encoding="utf-8")
        if vr.returncode!=0: raise SystemExit("ERROR: frozen manifest verification failed.")

    if args.input:
        input_path=(args.input if args.input.is_absolute() else project/args.input).resolve()
        discovery=[]
    else:
        input_path,discovery=discover(project)
    (outdir/"input_discovery.json").write_text(json.dumps(discovery,indent=2,default=str),encoding="utf-8")

    rawdf=pd.read_parquet(input_path)
    norm,sources=normalize(rawdf)
    quote=add_losses(norm)
    contract=contracts(norm)

    specs=[("overall",None),("prop_type","prop_type"),("preferred_side","preferred_side"),
           ("disagreement","disagreement_bucket"),("market_probability","market_prob_bucket"),
           ("month","month"),("expected_minutes","minutes_bucket")]
    summary=[]
    for scope,df in [("quote",quote),("contract",contract)]:
        for gt,gc in specs:
            summary.extend(summarize(df,scope,gt,gc))
        if scope=="quote" and quote.vendor.nunique()>1:
            summary.extend(summarize(quote,"quote","vendor","vendor"))
    summary=pd.DataFrame(summary)
    summary.to_csv(outdir/"proper_scoring_summary.csv",index=False)

    boot=bootstrap_table(contract,args.bootstrap_reps,args.seed)
    boot.to_csv(outdir/"game_cluster_bootstrap.csv",index=False)

    curve_rows=[]
    for label,pcol in [("raw","p_raw"),("selected","p_selected"),("market","p_market")]:
        for row in calibration(contract,pcol,args.bins):
            curve_rows.append({"group_type":"overall","group_value":"ALL","source":label,**row})
        for prop,g in contract.groupby("prop_type"):
            for row in calibration(g,pcol,args.bins):
                curve_rows.append({"group_type":"prop_type","group_value":str(prop),"source":label,**row})
    curves=pd.DataFrame(curve_rows)
    curves.to_csv(outdir/"calibration_curves.csv",index=False)

    overall=summary[(summary.scope=="contract")&(summary.group_type=="overall")].iloc[0]
    eces={k:ece(contract,p,args.bins) for k,p in [("raw","p_raw"),("selected","p_selected"),("market","p_market")]}

    manifest={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "branch":branch,"commit":commit,"project_root":str(project),"input":str(input_path),
        "input_sha256":sha256_file(input_path),"input_rows":len(rawdf),"quote_rows":len(quote),
        "contract_rows":len(contract),"games":contract.game_id.nunique(),"players":contract.player_id.nunique(),
        "prop_types":sorted(contract.prop_type.unique().tolist()),"probability_sources":sources,
        "bootstrap_reps":args.bootstrap_reps,"seed":args.seed,"ece":eces,
        "interpretation":{
            "primary_scope":"contract",
            "delta_definition":"selected model minus market; negative means model better",
            "pushes":"excluded",
            "2025_status":"retrospective development audit, not external validation",
            "model_mutation":False,
            "threshold_selection":False,
        }
    }
    (outdir/"AUDIT_MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str),encoding="utf-8")

    report=f"""# Frozen 2025 Proper-Scoring Audit

Branch: `{branch}`
Commit: `{commit}`
Input: `{input_path}`

This is a **retrospective development audit**, not external validation.

## Overall contract-level results

| Source | Brier | Log loss | ECE |
|---|---:|---:|---:|
| Raw model | {overall.brier_raw:.6f} | {overall.logloss_raw:.6f} | {eces['raw']:.6f} |
| Selected calibrated model | {overall.brier_selected:.6f} | {overall.logloss_selected:.6f} | {eces['selected']:.6f} |
| De-vig market | {overall.brier_market:.6f} | {overall.logloss_market:.6f} | {eces['market']:.6f} |

Selected model minus market Brier: **{overall.delta_brier_selected_market:+.6f}**
Selected model minus market log loss: **{overall.delta_logloss_selected_market:+.6f}**

Negative means the model outscored the market.

## Outputs

- `proper_scoring_summary.csv`
- `game_cluster_bootstrap.csv`
- `calibration_curves.csv`
- `AUDIT_MANIFEST.json`
- `input_discovery.json`
- `frozen_manifest_verification.txt`

The summary contains overall, prop, side, disagreement, market-probability, month, minutes-bucket, and vendor diagnostics.

Do not select a new production threshold from this report. Any material model change requires a new freeze ID.
"""
    (outdir/"PROPER_SCORING_AUDIT.md").write_text(report,encoding="utf-8")

    print("="*100)
    print("FROZEN 2025 PROPER-SCORING AUDIT")
    print("="*100)
    print("Branch:",branch)
    print("Input:",input_path)
    print(f"Quote rows: {len(quote):,}")
    print(f"Contract rows: {len(contract):,}")
    print(f"Games: {contract.game_id.nunique():,}")
    print()
    print(f"Brier     raw={overall.brier_raw:.6f} selected={overall.brier_selected:.6f} market={overall.brier_market:.6f}")
    print(f"Log loss  raw={overall.logloss_raw:.6f} selected={overall.logloss_selected:.6f} market={overall.logloss_market:.6f}")
    print(f"ECE       raw={eces['raw']:.6f} selected={eces['selected']:.6f} market={eces['market']:.6f}")
    print()
    print(f"Selected-market Brier delta:   {overall.delta_brier_selected_market:+.6f}")
    print(f"Selected-market logloss delta: {overall.delta_logloss_selected_market:+.6f}")
    print()
    print("Saved:",outdir)
    print("PASS: research audit completed without fitting or mutating the frozen model.")


if __name__=="__main__":
    main()
