# Reviewer Sign-off Checklist

This checklist is intended for independent expert review.

## Statistical methodology

- [ ] Chronological / point-in-time feature construction is satisfactory.
- [ ] OOF / holdout separation is satisfactory.
- [ ] Mean-model selection is defensible.
- [ ] Marginal-distribution selection is defensible.
- [ ] Mean-preserving ZINB implementation is satisfactory.
- [ ] Dependence / copula policy is defensible.
- [ ] Calibration policy is defensible.
- [ ] Push handling is correct.
- [ ] Fair-price conversion is correct.

## Probability quality

- [ ] Brier-score methodology is correct.
- [ ] Log-loss methodology is correct.
- [ ] Calibration intercept/slope methodology is correct.
- [ ] Reliability / ECE methodology is correct.
- [ ] Game-cluster bootstrap is appropriate.
- [ ] Market comparison is fair and correctly paired.
- [ ] Randomized PIT methodology is appropriate.
- [ ] Reported strengths and weaknesses are fairly characterized.

## Implementation

- [ ] Frozen model identity / hashes are satisfactory.
- [ ] Runtime reproduction is satisfactory.
- [ ] Production contract checks are satisfactory.
- [ ] Fair-price feed integration preserves statistical behavior.
- [ ] No evidence of leakage or post-result retuning is found.

## Reviewer conclusion

Choose one:

- [ ] Approved for prospective external testing as frozen.
- [ ] Approved with non-statistical implementation/documentation changes.
- [ ] Statistical changes required; create a new model freeze before deployment.
- [ ] Not approved.

## Notes

Reviewer:

Date:

Comments:
