# Probability Quality Certification Protocol v1

## Purpose

Evaluate the quality, calibration, information content, temporal stability, and market-relative performance of the frozen NBA player-prop probability model.

This exercise is evaluation-only.

No production model parameter, feature, marginal distribution, calibration policy, dependence parameter, selection policy, or decision threshold may be changed as a consequence of this certification without creating a new statistical model freeze.

## Frozen model identity

Model freeze:
nba_prop_quant_20260818T205213Z

Certification research base:
eb1d49153c94e52714010e302df9c1f58221b5c6

Certification branch:
research/probability-quality-cert-v1

## Evaluation samples

### Primary intrinsic probability sample

Use strict chronological OOF predictions from seasons 2018 through 2025.

No in-sample fitted prediction may enter the certification sample.

### Market-comparison sample

Use the exact 2025 matched-contract construction already defined by:

research/proper_scoring_audit_2025.py

Contract key:

game_id / player_id / prop / line

The existing market-probability construction and contract-collapse rules are frozen for this certification and may not be changed after results are observed.

## Binary non-push evaluation

For over/under probability quality, evaluate the selected calibrated non-push probability:

q_over_nonpush

against the realized over/under outcome on non-push contracts.

Pushes are excluded from binary Brier score, binary log loss, calibration intercept/slope, and binary market comparison.

Probabilities must be clipped only for numerical log-loss and logit calculations at:

1e-6 <= p <= 1 - 1e-6

No probability clipping may be used to alter reported Brier scores.

## Push and three-outcome evaluation

Where selected unconditional probabilities are available, verify:

p_over + p_under + p_push = 1

within numerical tolerance.

Evaluate model-only three-outcome probability quality for:

over / under / push

using multiclass Brier score and multiclass logarithmic score when the required unconditional probabilities and outcomes are available.

Market comparison is not required for three-outcome scoring unless a directly comparable market push probability exists in the frozen source data.

## Primary scoring metrics

Report:

1. Brier score
2. Log loss
3. Calibration intercept
4. Calibration slope
5. Expected calibration error
6. Mean predicted probability
7. Observed event frequency

For model-versus-market comparisons define:

delta = model score - market score

Negative delta means the model is better.

Positive delta means the market is better.

## Calibration diagnostics

Calibration intercept and slope will be estimated using logistic calibration:

logit(P(Y=1)) = intercept + slope * logit(p_model)

Ideal values:

intercept = 0
slope = 1

Reliability tables will use predicted-probability deciles with:

sample count
mean predicted probability
observed event frequency
absolute calibration error

A fixed-width reliability table may also be reported as supplementary evidence.

## Sharpness and resolution

Report:

mean absolute deviation of probability from 0.50
standard deviation of predicted probability
minimum probability
maximum probability
5th percentile
25th percentile
median
75th percentile
95th percentile

Report Brier reliability/resolution/uncertainty decomposition where feasible.

## Confidence buckets

Define model confidence as:

max(p_over_nonpush, 1 - p_over_nonpush)

Use these fixed buckets:

0.5000 to <0.5250
0.5250 to <0.5500
0.5500 to <0.6000
0.6000 to <0.6500
0.6500 to <0.7000
0.7000 to 1.0000

For every bucket report:

contracts
games
mean confidence
realized correctness rate
calibration gap
Brier score
log loss

No bucket boundary may be changed after results are observed.

## Stratification

Report probability-quality metrics:

overall
by season
by prop

Where sample size permits, also report:

season x prop

All strata must be retained in the output.

A stratum with fewer than 250 contracts or fewer than 30 unique games must be labeled:

descriptive_only

Such strata may not support strong inferential conclusions.

## Cluster bootstrap

Use game-level clustered bootstrap resampling.

For multi-season overall estimates, preserve season composition through season-stratified game resampling.

Bootstrap repetitions:

5000

Deterministic seed:

20260823

Report 95% percentile confidence intervals.

For model-versus-market paired scoring, bootstrap the paired contract differences using the same resampled games.

## Market-relative certification

For the frozen 2025 matched-contract sample report:

model Brier
market Brier
model minus market Brier
model log loss
market log loss
model minus market log loss

Report game-cluster bootstrap confidence intervals and bootstrap probability that the model is better.

Perform this:

overall
by prop

Do not use these results to retune the frozen model.

## Temporal stability

For every available OOF season report:

contracts
games
Brier score
log loss
ECE
calibration intercept
calibration slope
mean confidence
observed event frequency

No season may be omitted because its results are unfavorable.

## Full-distribution diagnostics

If exact frozen row-level distribution parameters are available without refitting or post-certification tuning, perform discrete randomized-PIT diagnostics.

If the required frozen inputs are unavailable, mark this test:

NOT_AVAILABLE_FROM_FROZEN_ARTIFACTS

Do not reconstruct or refit a new distribution solely to improve this certification.

## Interpretation rules

The certification will distinguish:

calibration
discrimination/information content
sharpness
market-relative proper scoring
temporal stability

Excellent calibration alone is not evidence of sportsbook edge.

Market-relative superiority must be supported by paired proper scoring and game-cluster uncertainty.

A statistically indistinguishable result will be reported as such.

Unfavorable findings will remain in the certification package.

## Research integrity

No model tuning is permitted during this certification.

No metric may be deleted because it produces an unfavorable result.

No probability bucket may be redefined after results are observed.

No season or prop may be excluded except for a documented data-integrity reason.

All exclusions must be counted and reported.

All generated certification outputs must be reproducible from immutable source inputs and deterministic seeds.

## Reviewer scope

This certification may support reviewer conclusions regarding:

probability calibration
proper-scoring quality
temporal robustness
market-relative probability quality
implementation correctness
reproducibility

It does not establish prospective 2026-27 profitability.

True prospective performance remains an external-test question.
