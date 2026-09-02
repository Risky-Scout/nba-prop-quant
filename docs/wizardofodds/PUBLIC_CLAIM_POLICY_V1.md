# WizardOfOdds NBA Prop Quant v2 — Public Claim Policy v1

## Purpose

This policy controls what WizardOfOdds.com may say publicly about NBA Prop Quant v2.

It does not modify or refit the model.

Frozen model source commit:

`4def8ad33ccc56016fb19a97fceca6e027c9612a`

## Initial public label

**Prospective validation in progress**

The site may publish model probabilities, model fair odds, same-capture no-vig market probabilities, model-market probability differences, timestamps, model version, and the permanent prospective record.

## Historical statement permitted before prospective validation

Historical development testing shows the final v2 model is statistically better than frozen v1 and approximately market-level overall, with statistically supported retrospective market-relative advantages in steals and points+assists.

This statement must always be accompanied by:

**2025 results are retrospective development evidence, not untouched external validation.**

## Primary prospective sample

Only cryptographically verified T-20-minute records are eligible for the primary prospective classification.

There is no fallback primary window.

Model prediction and market comparison must use the identical capture.

## Minimum evidence before a formal prospective classification

Overall:

- at least 300 unique NBA games;
- at least 5,000 graded eligible contracts.

For a prop-specific superiority claim:

- at least 150 unique games;
- at least 1,000 graded eligible contracts for that prop.

No public claim may be upgraded before the applicable minimum sample is reached.

## Prospectively market-competitive

The exact allowed wording is:

**Prospectively market-competitive over the stated sample**

The overall model must satisfy all of the following:

- Brier model-minus-market point delta <= +0.001;
- Brier 95% game-cluster-bootstrap upper bound <= +0.001;
- log-loss model-minus-market point delta <= +0.002;
- log-loss 95% game-cluster-bootstrap upper bound <= +0.002;
- calibration slope between 0.75 and 1.25;
- model ECE no more than 0.01 absolute above market ECE.

## Prospectively market-superior

The exact allowed wording is:

**Prospectively outperformed the no-vig market over the stated sample**

The model must satisfy all of the following:

- Brier point delta < 0;
- Brier 95% game-cluster-bootstrap upper bound < 0;
- P(model better on Brier) >= 0.975;
- log-loss point delta < 0;
- log-loss 95% game-cluster-bootstrap upper bound < 0;
- P(model better on log loss) >= 0.975;
- calibration slope between 0.75 and 1.25;
- model ECE no more than 0.01 absolute above market ECE.

## Prop-specific market superiority

The prop must meet its minimum sample and both Brier and log-loss 95% confidence intervals must exclude zero in the favorable direction, with P(model better) >= 0.975 for both metrics.

## If the prospective test trails the market

The site should say:

**Prospective results currently trail the no-vig market over the stated sample.**

The result is reported rather than hidden or retuned away.

## Re-evaluation

After the minimum sample is reached, classification is recalculated monthly using every eligible frozen-model prospective observation to date.

A claim may be downgraded if later evidence deteriorates.

## Prohibited claims

Without separate evidence, do not use:

- guaranteed edge;
- guaranteed profit;
- profitable model;
- proven profitable;
- beats sportsbooks;
- market-beating across all props;
- best bets;
- lock;
- sure thing.

## Profitability

Probability-model performance is not the same as realized wagering profitability.

A profitability claim would require a separately frozen prospective wagering protocol using executable prices, stakes, limits, and realized returns.

## Betting status

`auto_bet=False`

The WizardOfOdds product is an analytics/probability product unless a separate betting policy is developed and prospectively validated.
