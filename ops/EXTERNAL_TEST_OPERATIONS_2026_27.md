# NBA Prop Quant — 2026–27 External-Test Operations

## Status

Deployment freeze: **external_test_deployment**

Primary rule: **2025 is development data. 2026–27 is the first external market test.**

No betting threshold was selected from 2025. The fixed monitoring grid remains:

`1%, 2%, 3%, 5%, 7.5%, 10%`

The archive is observational. It must never rewrite an original projection or market snapshot after capture.

---

## 1. Preseason engineering mode

Use preseason only to test plumbing, not performance.

```bash
DATE="2026-10-03"

python ops/capture_external_test_day.py \
  --date "$DATE" \
  --mode engineering \
  --tag preseason_smoke
```

If there are no markets yet, projection-only capture is allowed:

```bash
python ops/capture_external_test_day.py \
  --date "$DATE" \
  --mode engineering \
  --tag preseason_projection \
  --skip-market-pricing
```

Do not grade preseason results as part of the external test.

---

## 2. Official regular-season capture

On an actual 2026–27 slate date, use `external` mode. The script refuses to run an official capture for a date other than today's date in `America/New_York`.

```bash
DATE="2026-10-20"

python ops/capture_external_test_day.py \
  --date "$DATE" \
  --mode external \
  --tag first_capture
```

The workflow performs:

1. frozen deployment manifest verification;
2. production-contract validation;
3. leakage guard: local raw standard/advanced history must be strictly earlier than the slate date;
4. contemporaneous slate projection;
5. immediate immutable copy of that projection;
6. current sportsbook fetch and pricing;
7. immutable copy of priced and rejected quote files;
8. copy of the exact deployment manifest;
9. capture metadata and SHA-256 checksum manifest;
10. append-only capture index entry.

If no games exist, the no-slate result is archived and no fabricated projection is created.

---

## 3. Multiple snapshots during a slate day

Multiple snapshots are allowed and are useful for measuring information arrival.

Each capture gets a unique UTC timestamp. Optional tags are descriptive only:

```bash
python ops/capture_external_test_day.py \
  --date "$DATE" \
  --mode external \
  --tag morning

python ops/capture_external_test_day.py \
  --date "$DATE" \
  --mode external \
  --tag afternoon

python ops/capture_external_test_day.py \
  --date "$DATE" \
  --mode external \
  --tag pre_tip
```

Do **not** retrospectively choose whichever snapshot makes the model look best.

For later analysis, retain all snapshots. A future grading script should define the primary rule before reading outcomes—for example, first eligible captured quote per market—and keep secondary snapshot analyses separate.

---

## 4. Archive layout

Each capture is stored under:

```text
data/external_test/season=2026/
  capture_index.jsonl
  date=YYYY-MM-DD/
    captures/
      YYYYMMDDTHHMMSSZ_tag/
        projection.parquet
        priced_markets.parquet
        rejected_markets.parquet
        deployment_manifest.json
        capture_external_test_day.py
        capture_manifest.json
        checksums.sha256
        logs/
          01_verify_frozen_manifest.log
          02_validate_production_contract.log
          03_predict_slate.log
          04_price_markets.log
```

Files only appear when applicable. For example, a projection-only capture has no priced-market file.

---

## 5. Verify an archived capture

```bash
python ops/verify_external_test_capture.py \
  data/external_test/season=2026/date=2026-10-20/captures/<CAPTURE_ID>
```

Expected:

```text
PASS: archived capture matches its frozen checksums.
```

If verification fails, do not overwrite the archive. Investigate and preserve the mismatch.

---

## 6. List all captures

```bash
python ops/list_external_test_captures.py
```

The index is append-only and records date, capture ID, mode, freeze ID, capture manifest path, UTC timestamp, and pricing status.

---

## 7. Daily completed-history refresh rule

The deployment model is frozen, but **newly completed games are legitimate time-varying inputs**. During the 2026–27 external test, the routine historical refresh is limited to current-season standard box scores and advanced statistics.

### Canonical refresh command

Starting **2026-10-21** and on every later official slate date, after the previous day's games are complete and available from BDL, run:

```bash
python scripts/01_ingest_history_resume.py \
  --start-season 2026 \
  --end-season 2026 \
  --include-advanced \
  --skip-players \
  --force
```

`--force` is required because the resume-safe ingester otherwise skips an already-valid `season=2026` file. During an in-progress season, that valid file becomes stale as new games finish.

Do **not** add these flags to the routine pregame refresh:

```text
--include-opening-props
--include-lineups
--include-plays
```

Sportsbook quotes are captured contemporaneously by the market-pricing workflow. Lineups and play-by-play are not part of the frozen pregame feature contract.

Do **not** run:

```bash
python scripts/17_build_completed_history.py
```

during the external-test season. That is a rebuild/retraining workflow and would mutate the frozen architecture.

### Opening-day exception

For the first regular-season slate on **2026-10-20**, do **not** force-refresh season 2026 before the first official capture. There are no completed 2026–27 regular-season games yet.

Opening-day manual sequence:

```bash
DATE="2026-10-20"

python scripts/verify_frozen_manifest.py
python scripts/10a_validate_production_contract.py

python ops/capture_external_test_day.py \
  --date "$DATE" \
  --mode external \
  --tag morning
```

### October 21 and every later official slate date

Manual reference sequence:

```bash
DATE="YYYY-MM-DD"

# 1. Verify the frozen deployment.
python scripts/verify_frozen_manifest.py

# 2. Refresh only completed 2026-27 standard + advanced history.
python scripts/01_ingest_history_resume.py \
  --start-season 2026 \
  --end-season 2026 \
  --include-advanced \
  --skip-players \
  --force

# 3. Revalidate the frozen model/policy contract.
python scripts/10a_validate_production_contract.py

# 4. Create the contemporaneous immutable capture.
python ops/capture_external_test_day.py \
  --date "$DATE" \
  --mode external \
  --tag morning
```

Replace `YYYY-MM-DD` with the actual current NBA slate date.

### Same-day refresh prohibition

Once the **first official capture** for a slate date has been made:

> **Do not refresh standard or advanced historical data again until the next slate date.**

Later same-day market snapshots are allowed, but they must use the exact same historical feature state as the first capture.

Operational order:

```text
previous day's games finish
        ↓
next slate morning
        ↓
verify frozen deployment
        ↓
force-refresh season 2026 completed history
        ↓
validate production contract
        ↓
make first official capture
        ↓
optional later same-day market captures
        ↓
NO history refreshes for the rest of that slate date
        ↓
refresh again on the next slate morning
```

### Independent leakage guard

`ops/capture_external_test_day.py` independently requires:

```text
latest standard-history date < target slate date
latest advanced-history date < target slate date
```

If either source contains data on or after the slate date, the capture aborts. Never bypass this guard for an official external-test record.

### Allowed time-varying inputs

Allowed to update during the frozen external test:

- completed 2026 standard box-score history strictly before the slate date;
- completed 2026 advanced history strictly before the slate date;
- current injury/availability snapshots;
- current sportsbook quotes;
- immutable capture archives;
- later separate outcome/grading files.

Not allowed to change without declaring a new frozen deployment epoch:

- trained minutes or target models;
- mean-model selection policy;
- dynamic-prior parameters;
- experience curves;
- ZINB marginals;
- combo dependence lambdas;
- probability-calibration coefficients/policy;
- production pricing code;
- monitoring edge grid.

### One-command official runner

The manual policy above is authoritative. `ops/run_external_test_day.sh` implements it in the same order and includes both the opening-day exception and the same-day no-refresh rule.

Official usage:

```bash
ops/run_external_test_day.sh YYYY-MM-DD [tag]
```

Examples:

```bash
ops/run_external_test_day.sh 2026-10-20 morning
ops/run_external_test_day.sh 2026-10-21 morning
ops/run_external_test_day.sh 2026-10-21 pre_tip
```

Behavior:

- requires the target date to equal the actual current date in `America/New_York`;
- verifies the frozen deployment first;
- on 2026-10-20, skips the current-season refresh;
- from 2026-10-21 onward, refreshes season 2026 only if no official capture already exists that day;
- if a capture already exists, skips historical refresh so later snapshots preserve the morning feature state;
- validates the production contract;
- creates the immutable capture;
- never activates an automatic betting threshold.

Optional environment controls:

```bash
COMBO_SIMULATIONS=20000 ops/run_external_test_day.sh 2026-10-21 morning
SKIP_MARKET_PRICING=1 ops/run_external_test_day.sh 2026-10-21 projection_only
```

Preseason engineering work should continue to call `ops/capture_external_test_day.py --mode engineering` directly. The official runner is intentionally restricted to real external-test dates.


## 8. What is frozen

During the external test, do not change without declaring a new model epoch:

- target mean-model selection;
- trained minutes or target model artifacts;
- dynamic-prior parameters;
- experience curves;
- ZINB marginals;
- combo dependence lambdas;
- market-probability calibration policy or coefficients;
- production pricing code;
- deployment manifest;
- monitoring edge grid.

If any of those change, create a new `external_test_deployment` freeze and treat results before/after the change as separate model versions.

---

## 9. What is not a betting rule

The following are **monitoring diagnostics**, not automatic wagers:

- 1% edge;
- 2% edge;
- 3% edge;
- 5% edge;
- 7.5% edge;
- 10% edge.

`auto_bet` remains false.

Do not promote one threshold because it looks best during the 2026–27 test. That would turn the external test into another development set.

---

## 10. Later grading policy

Do not write final outcomes back into original capture files.

When grading is added:

- preserve original `projection.parquet` and `priced_markets.parquet`;
- create separate outcome/grading files;
- record the outcome-source timestamp;
- use the freeze ID and capture ID as keys;
- report proper scoring and betting metrics by frozen model epoch;
- report uncertainty clustered by game;
- keep opening/first-capture and later snapshots separate.

The original archive is the evidence of what the system knew and priced at the time.
