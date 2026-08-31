# NBA Prop Quant v2 — Gate 3 Prospective Preregistration

## Frozen candidate

- points: frozen_selected_v1
- rebounds: frozen_selected_v1
- assists: v2_role_shock_calibrated
- steals: frozen_selected_v1
- blocks: frozen_selected_v1
- threes: frozen_selected_v1
- points_assists: v2_role_increment
- points_rebounds: v2_role_increment
- rebounds_assists: frozen_selected_v1
- points_rebounds_assists: frozen_selected_v1

## Prospective protocol

All Gate 3 predictions must be generated before scheduled tip.

Availability, lineup, starter, and market inputs must be timestamped.

The candidate policy may not be changed after observing Gate 3 outcomes.

Every supported prop will be reported against the same-as-of no-vig market using:

- Brier score
- log loss
- calibration intercept
- calibration slope
- decile ECE
- AUC
- sharpness
- game-cluster bootstrap confidence intervals

Development superiority is not prospective superiority.

A prospective market-superiority claim requires prospective evidence from the frozen 2026-27 candidate.
