# NBA Prop Quant v2 — Gate 2 Certification Preregistration

## Scope

Gate 1 model development is closed.

No further 2025 tuning is permitted before Gate 2 certification results are evaluated.

The locked Gate 2 change candidates are:

- assists: v2 role-shock calibrated
- points_assists: v2 role increment
- points_rebounds: v2 role increment

All other supported props retain the frozen v1 selected probability policy.

## Evidence status

The 2025 historical lineup backfill is development-only.

It lacks original pre-tip announcement timestamps and therefore cannot support a prospective market-superiority claim.

Prospective superiority claims require Gate 3 evidence collected by the timestamped 2026-27 pre-tip snapshot system.

## Gate 2 comparison set

Every supported prop must report:

- candidate probability
- frozen v1 selected probability
- no-vig market probability
- contract rows
- unique games
- Brier score
- log loss
- calibration intercept
- calibration slope
- decile ECE
- AUC
- sharpness

## Candidate survival requirements

For assists, points_assists, and points_rebounds:

1. Candidate Brier point estimate must be lower than frozen v1.
2. Candidate log-loss point estimate must be lower than frozen v1.
3. Game-cluster bootstrap must use 5,000 resamples.
4. For both proper scores, the bootstrap probability that the candidate beats frozen v1 must be at least 0.95.
5. At least one of the two proper-score 95% confidence intervals must exclude zero in the favorable direction.
6. Candidate ECE may not exceed frozen ECE by more than 0.010 absolute.
7. Candidate calibration slope must remain between 0.75 and 1.35.
8. Candidate AUC may not be more than 0.005 below frozen v1.
9. No predefined subgroup with at least 300 contracts may simultaneously worsen Brier by more than 0.010 and log loss by more than 0.020.

If any required candidate fails, that prop reverts to frozen v1 for the Gate 3 prospective candidate.

## Protected frozen props

Points, rebounds, steals, blocks, threes, rebounds_assists, and points_rebounds_assists remain frozen unless certification reveals an implementation defect.

Gate 2 is certification, not a new tuning phase.

## Market-relative reporting

Market comparison is mandatory for every prop.

Development market superiority may be described only when both Brier and log loss beat the market and the corresponding game-cluster bootstrap 95% confidence intervals exclude zero in the favorable direction.

Development market superiority is not equivalent to prospective market superiority.

## Gate 3 boundary

Gate 3 begins only after the final Gate 2 candidate policy is frozen.

Gate 3 uses only predictions generated before tip from the prospective 2026-27 collection system.
