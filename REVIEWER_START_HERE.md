# NBA Prop Quant — Elite Quant Review

## Purpose

This repository contains a frozen NBA player-prop probability model prepared for independent expert review.

The statistical model is frozen as:

`nba_prop_quant_20260818T205213Z`

The review package is designed to support examination of:

- chronological / leakage controls;
- model architecture and feature construction;
- selected mean models;
- marginal distributions and dependence handling;
- probability calibration;
- proper scoring and market-relative probability quality;
- implementation reproducibility;
- runtime integrity;
- production fair-price generation.

This review package does **not** claim proven prospective 2026-27 profitability. The untouched 2026-27 season remains the external prospective test.

## Recommended review order

1. Read `docs/reviewer/MODEL_IDENTITY_AND_HASHES.md`.
2. Read `docs/reviewer/PROBABILITY_QUALITY_CERTIFICATION_REPORT_V1.md`.
3. Read `docs/reviewer/KNOWN_LIMITATIONS.md`.
4. Inspect the frozen model Release and certified runtime Release.
5. Inspect the fair-price integration Release.
6. Review `docs/reviewer/REPRODUCIBILITY_FOR_REVIEWER.md`.
7. Use `docs/reviewer/REVIEWER_SIGNOFF_CHECKLIST.md` for final sign-off notes.

## Important statistical conclusion

The selected/calibrated probabilities materially improve over the raw probabilities and are well calibrated overall.

On the frozen 2025 retrospective matched-contract sample:

- 56,051 contracts
- 297 games
- 10 prop markets
- 5,000 game-cluster bootstrap repetitions
- selected Brier: 0.243707
- market Brier: 0.242831
- selected log loss: 0.679960
- market log loss: 0.678144
- calibration intercept: -0.004498
- calibration slope: 0.915987
- decile ECE: 0.007764

The market is modestly but statistically better overall on proper scoring. Steals is a clear model strength. Assists and threes are the clearest market-relative weaknesses. No tuning was performed after certification results were observed.

## Reviewer posture

Unfavorable results are intentionally retained. Any statistical change prompted by this review requires a new model freeze.
