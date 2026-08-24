# Probability Quality Certification v1

## Scope

The binary line-level certification uses the frozen 2025 matched-contract sample:

- 56,051 contracts
- 297 games
- 10 prop markets
- 5,000 game-cluster bootstrap repetitions
- no post-result tuning

Historical 2018-2024 line-level probability certification is not claimed because the required frozen line/probability artifacts were not available.

Randomized PIT is evaluated on the leakage-safe 2025 holdout using the frozen `marginals_pre2025.joblib`, fit on 2018-2024 only.

## Overall proper scoring

| Metric | Raw | Selected / calibrated | Market |
|---|---:|---:|---:|
| Brier | 0.250177 | 0.243707 | 0.242831 |
| Log loss | 0.696443 | 0.679960 | 0.678144 |

The calibration layer materially improves the raw model.

## Calibration

- Calibration intercept: **-0.004498**
- Calibration slope: **0.915987**
- Decile ECE: **0.007764**
- Mean selected over probability: **0.466865**
- Observed over rate: **0.468216**
- Mean absolute distance from 0.50: **0.050207**
- Probability standard deviation: **0.078201**

Interpretation: overall centering is excellent. The slope below 1 indicates mild overconfidence.

## Market-relative inference

Selected minus market:

- Brier delta: **+0.000877**
  - 95% game-cluster CI: **[+0.000263, +0.001501]**
  - bootstrap probability model is better: **0.0028**
- Log-loss delta: **+0.001816**
  - 95% game-cluster CI: **[+0.000539, +0.003114]**
  - bootstrap probability model is better: **0.0044**

Lower is better, therefore the market is modestly but statistically better overall on this retrospective sample.

## Prop-level findings

### Clear model strength

**Steals**

- Brier delta vs market: **-0.003447**
- 95% CI: **[-0.005963, -0.000933]**
- Log-loss delta vs market: **-0.006807**
- 95% CI: **[-0.012511, -0.001094]**

### Approximately market-quality / statistically unresolved

- Blocks
- Points + Assists
- Points + Rebounds
- Points + Rebounds + Assists
- Rebounds + Assists
- Rebounds is borderline / uncertain under clustered inference

### Clear or detectable weaknesses

**Assists**
- Brier delta: **+0.004317**
- Log-loss delta: **+0.008880**

**Threes**
- Brier delta: **+0.003458**
- Log-loss delta: **+0.007541**

**Points**
- Brier delta: **+0.000843**
- Log-loss delta: **+0.001693**
- Calibration slope: **0.453385**

Combo-market calibration slopes:

- Points + Assists: **0.290469**
- Points + Rebounds: **0.356297**
- Points + Rebounds + Assists: **0.570423**

These low slopes indicate overconfidence in the probability logits and should receive explicit reviewer attention.

## Reliability / confidence

The largest overall reliability-decile absolute gaps are approximately 1.5 percentage points.

Upper confidence buckets show mild overconfidence:

- 60-65%: mean confidence 62.20%, realized 60.76%
- 65-70%: mean confidence 67.24%, realized 64.43%
- 70%+: mean confidence 78.32%, realized 76.64%

## Randomized PIT

Ideal randomized PIT has mean 0.5 and variance approximately 0.08333.

| Target | PIT mean | PIT variance |
|---|---:|---:|
| AST | 0.495916 | 0.082986 |
| BLK | 0.498509 | 0.083830 |
| FG3M | 0.496803 | 0.082858 |
| PTS | 0.504516 | 0.081571 |
| REB | 0.490832 | 0.082305 |
| STL | 0.498809 | 0.082921 |

AST, BLK, FG3M, and STL are close to uniform. Points and rebounds show modest departures.

IID KS/CvM p-values are descriptive only because player-game observations are clustered.

## Push integrity

- 56,051 collapsed contracts
- 56,049 half-point lines
- 2 integer lines
- 0 actual pushes
- 0 integer lines where actual equaled line
- raw over + under + push max absolute error: `2.22e-16`
- selected over + under + push max absolute error: `2.22e-16`

No push-accounting defect was found.

## Certification conclusion

The frozen model produces well-calibrated and informative probabilities. Calibration materially improves the raw model, and single-stat distribution diagnostics are generally strong.

The model is close to market quality overall but does not beat the market aggregate on 2025 proper scoring. Steals is a clear strength; assists and threes are the clearest weaknesses; points and several combo markets show evidence of overconfidence.

No model tuning was performed in response to these findings.

## Claims not established

This certification does not establish:

- prospective 2026-27 profitability;
- a universal sportsbook edge;
- line-level probability calibration for 2018-2024.

The untouched 2026-27 season remains the prospective external test.
