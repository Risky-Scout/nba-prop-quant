# Probability Quality Artifact Coverage v1

## Frozen model

nba_prop_quant_20260818T205213Z

## Preregistration

nba_prop_quant_20260818T205213Z_probability_quality_prereg_v1

Preregistration commit:

5aacd0e81b578e2edbc0255f6f3afb39fcebbecb

## Artifact findings

### 2018-2025 strict OOF selected means

SUPPORTED

Artifact:

data/processed/selected_means_distribution_split.parquet

Rows:

217382

Seasons:

2018-2025

Unique games:

10231

This artifact supports historical OOF mean-level analysis but does not contain line-level selected probabilities.

### 2025 calibrated line-level probability certification

SUPPORTED

Primary artifact:

data/processed/market_backtest/calibrated_oof/selected_oof_quote_rows.parquet

Rows:

120156

Seasons:

2025

Unique games:

297

Contains line values, raw probabilities, calibrated selected probabilities, push probabilities, market probabilities, actual outcomes, prop types, folds, and game IDs.

### 2025 contract-level probability scoring

SUPPORTED

Artifact:

data/processed/market_backtest/calibrated_oof/selected_oof_contracts.parquet

Rows:

56051

Unique games:

297

Inferred seasons:

2025 only

Unmatched games:

0

### 2025 broader contract scoring

SUPPORTED

Artifact:

data/processed/market_backtest/contract_scores.parquet

Rows:

69105

Unique games:

381

Inferred seasons:

2025 only

Unmatched games:

0

## 2018-2024 line-level probability certification

NOT_AVAILABLE_FROM_FROZEN_ARTIFACTS

No frozen processed Parquet artifact was found containing historical 2018-2024 line values together with the selected model probabilities required by the preregistered binary probability tests.

These probabilities will not be reconstructed, refit, or regenerated after preregistration solely to expand certification coverage.

## Randomized PIT

NOT_YET_CONFIRMED

Randomized PIT will be attempted only if exact frozen row-level distribution parameters can be located without refitting.

## Interpretation

The certification may make strong retrospective probability-quality statements for the frozen 2025 matched-contract sample.

It may not claim that line-level calibration or market-relative probability performance was independently verified for every season from 2018 through 2024.

Historical 2018-2025 OOF selected means remain available as separate evidence of chronological model construction and temporal coverage.
