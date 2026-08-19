# 2026–27 External-Test Outcome Grading

## Purpose

Grading is deliberately separate from capture.

The immutable forecast/market archive is evidence of what the system knew at pricing time. Final box-score outcomes are attached later in a **new grade archive**. The original `projection.parquet`, `priced_markets.parquet`, capture manifest, logs, and checksums are never rewritten.

## Outcome source

The grading workflow uses the validated local BDL standard history:

```text
data/raw/seasons/season=2026/stats.parquet
data/raw/seasons/season=2026/games.parquet
```

For a 2026–27 regular-season capture, grade only after the game is final and the local completed-history refresh contains that game's final player rows.

The grade records SHA-256 hashes of the exact outcome-source files used.

## Settlement mapping

Supported model prop types are settled from final box scores as follows:

```text
points                         = pts
rebounds                       = reb
assists                        = ast
steals                         = stl
blocks                         = blk
threes                         = fg3m
points_rebounds                = pts + reb
points_assists                 = pts + ast
rebounds_assists               = reb + ast
points_rebounds_assists        = pts + reb + ast
stocks                         = stl + blk
```

`stocks` is retained for completeness even though the frozen BDL live market universe did not expose it as a normal supported over/under market.

## DNP and void policy

The grader does **not** assume sportsbook-specific DNP settlement rules.

A player with no final outcome row or no evidence of playing is marked:

```text
player_missing_final_game_void_candidate
dnp_void_candidate
```

Those rows are excluded from proper scoring and monitoring ROI.

This is intentionally conservative. If vendor-specific settlement rules are added later, they must be represented as a separate explicit rule layer rather than silently rewriting historical grades.

## Push policy

For a settled player:

```text
actual > line  -> over
actual < line  -> under
actual == line -> push
```

Push profit is zero.

Push rows are excluded from binary Brier score and log loss because the binary target is conditional on a non-push result.

## Probability scoring

The grader preserves the production pricing record and attempts to resolve:

- calibrated conditional non-push over probability;
- raw conditional non-push over probability;
- de-vigged market conditional over probability.

When the priced file stores unconditional over/under probabilities, the grader converts them to:

```text
q_over = p_over / (p_over + p_under)
```

No probability is refit after outcomes are known.

If a future production schema uses unrecognized probability column names, settlement still remains conceptually separate; the grading script will record unresolved probability-source metadata rather than invent a mapping.

## Monitoring thresholds

The predeclared grid remains:

```text
1%, 2%, 3%, 5%, 7.5%, 10%
```

These are evaluation buckets only.

No threshold is promoted to an automatic betting rule because it performs best during the 2026–27 external test.

The grader uses the side/edge/EV recorded by the contemporaneous production price file when those fields are available. It does not use the final outcome to choose a side. When a recorded EV field is available, monitoring eligibility also requires recorded EV > 0; otherwise the script records an edge-only monitoring calculation. When a recorded EV field is available, monitoring eligibility also requires recorded EV > 0, matching the development-era evaluation convention; otherwise the script records that the monitoring calculation is edge-only.

## Grade archive

Grades live outside the capture directory:

```text
data/external_test/season=2026/
  grades/
    grade_index.jsonl
    date=YYYY-MM-DD/
      capture_id=<CAPTURE_ID>/
        <GRADE_ID>/
          outcome_snapshot.parquet
          graded_quotes.parquet
          graded_contracts.parquet
          monitoring_summary.csv
          grade_manifest.json
          grade_external_test_capture.py
          checksums.sha256
```

A no-game/no-market capture receives a small immutable grade archive with status `no_priced_markets`.

## Grade one capture

```bash
python ops/grade_external_test_capture.py \
  --capture data/external_test/season=2026/date=YYYY-MM-DD/captures/<CAPTURE_ID>
```

Before grading, the script verifies the original capture checksums.

It refuses to grade a priced capture if any captured game is not confirmed final.

## Grade all official captures for a date

After the next completed-history refresh:

```bash
python ops/grade_external_test_day.py \
  --date YYYY-MM-DD \
  --mode external
```

Engineering captures can be graded separately:

```bash
python ops/grade_external_test_day.py \
  --date YYYY-MM-DD \
  --mode engineering
```

## Verify a grade

```bash
python ops/verify_external_test_grade.py \
  data/external_test/season=2026/grades/date=YYYY-MM-DD/capture_id=<CAPTURE_ID>/<GRADE_ID>
```

Expected:

```text
PASS: grade archive matches its frozen checksums.
```

## List grade ledger

```bash
python ops/list_external_test_grades.py
```

## Primary external-test interpretation rule

For later pooled reporting, the primary market evaluation should use the **first completed official external capture of each slate date**.

Later same-day captures are secondary information-arrival analyses and must be reported separately. Do not retrospectively pick the snapshot with the best realized result.

Within a capture:

- contract-level probability scoring uses one row per `game_id/player_id/prop_type/line_value`;
- vendor market probabilities may be averaged at contract level;
- quote-level and best-available-event ROI diagnostics remain separate scopes;
- uncertainty should be clustered by game.

## No model mutation

The grading scripts are an observational layer. They do not modify trained models, production pricing code, calibration coefficients, dependence policy, monitoring thresholds, or the external-test deployment manifest.
