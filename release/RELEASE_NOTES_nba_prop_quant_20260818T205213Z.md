# nba_prop_quant_20260818T205213Z

## NBA Prop Quant — 2026-27 External Test Deployment

This release freezes the pregame NBA player-prop fair-pricing system intended for prospective 2026-27 evaluation and downstream pricing integration for wizardofodds.com.

### Frozen policy

- PTS: XGB mean
- REB: XGB/decay/Kalman ensemble
- AST: XGB mean
- STL: XGB mean
- BLK: XGB/decay/Kalman ensemble
- FG3M: XGB mean
- all six marginals: mean-preserving ZINB
- combo dependence: R+A lambda 0.85; other supported combos lambda 0
- prop-specific probability calibration except blocks and steals raw
- no automatic betting threshold

### Evidence posture

2025 sportsbook results are development/model-selection evidence. They are not an untouched external market test.

The first prospective external market test is 2026-27.

### QA completed

- frozen-manifest verification
- production-contract validation
- synthetic grading integration
- 5,000-row real 2025 grading replay
- immutable capture/grade verification
- preseason current-season refresh preflight

### Open operational item

The first genuine live 2026-27 priced-market file must confirm the exact runtime side/edge/EV field names before sportsbook-comparison grading is considered fully runtime-validated.

### Integrity

Use the accompanying SHA-256 manifest to verify all release assets.
