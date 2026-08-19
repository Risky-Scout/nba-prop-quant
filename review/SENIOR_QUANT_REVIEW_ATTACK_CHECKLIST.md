# Senior Quant Review — Adversarial Attack Checklist

A strong review should try to invalidate the system rather than merely confirm that it runs.

## Information-set integrity

- Can every production feature be reconstructed from information known before tip?
- Are all rolling/dynamic priors shifted correctly?
- Are season transitions, playoffs, Play-In and Cup classifications point-in-time safe?
- Are current injury fields ever used as historical proxies?
- Is any market quote timestamp later than the information set used to claim an edge?

## Mean layer

- Reproduce OOF minutes error by season.
- Reproduce target OOF error by season and target.
- Reproduce the conservative ensemble gate.
- Assess whether the REB/BLK blend gains are economically/statistically meaningful.
- Stress trades, rookies, returning injured players and abrupt role changes.

## Distribution layer

- Verify all PMFs integrate to one.
- Verify mean-preserving ZINB identities numerically.
- Audit zero mass, variance and tail calibration by role/minutes bucket.
- Verify integer-line push mass.
- Challenge whether minutes uncertainty should be integrated rather than conditioned on one expected-minutes estimate.

## Dependence layer

- Reproduce player-specific shrinkage and PSD correction.
- Reproduce grouped CV for combo lambdas.
- Verify independence fallback when the gate fails.
- Stress small-history and role-change players.
- Quantify price sensitivity to dependence uncertainty.

## Market calibration

- Reproduce strict whole-date chronological folds.
- Confirm future quotes/outcomes never enter calibration.
- Reproduce raw, calibrated and market proper scoring.
- Check reliability by prop, line region, role, vendor and time-to-tip.
- Monitor 2026–27 drift without silently refitting the frozen external-test model.

## Market evaluation

- Separate quote, vendor-event, event and contract units.
- Cluster uncertainty by game.
- Treat 2025 as development evidence.
- Quantify how much best-event ROI is quote shopping.
- Add timestamped closing-line evaluation before a market-superiority claim.
- Predeclare future release gates before looking at 2026–27 performance.

## Operations and grading

- Verify deployment manifest before every external capture.
- Require historical data dates strictly earlier than slate date.
- Do not refresh history again after the first same-day capture.
- Verify capture hashes before grading.
- Keep DNP/void policy explicit.
- Pushes are zero-profit and excluded from binary proper scoring.
- Synthetic/replay QA must never enter the external-test performance ledger.

## Review decision

1. Accept the external-test protocol unchanged.
2. Require a new frozen deployment before external testing begins.
3. Allow the external test but keep flagged analyses descriptive until more prospective evidence accumulates.
