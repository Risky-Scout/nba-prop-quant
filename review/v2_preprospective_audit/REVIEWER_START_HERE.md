# NBA Prop Quant v2 — Reviewer Start Here

This package is the pre-prospective independent-audit release of the NBA player-prop probability model.

Model source commit: `4def8ad33ccc56016fb19a97fceca6e027c9612a`

Primary prospective certification window: **T-20 minutes before scheduled tip**.

## Scientific status

Historical development and Gate 2 certification are complete.

The frozen Gate 3 candidate is implemented and deployed with cryptographic snapshot lineage.

Untouched 2026-27 outcomes do not yet exist. Therefore this release **does not claim prospective market superiority or profitability**.

The purpose of this release is to allow independent review of the methodology, code, model artifacts, retrospective certification, leakage controls, calibration, deployment logic, and preregistered prospective protocol before outcomes are observed.

## Frozen Gate 3 model

| Prop | Frozen candidate |
|---|---|
| assists | v2_role_shock_calibrated |
| blocks | frozen_selected_v1 |
| points | frozen_selected_v1 |
| points_assists | v2_role_increment |
| points_rebounds | v2_role_increment |
| points_rebounds_assists | frozen_selected_v1 |
| rebounds | frozen_selected_v1 |
| rebounds_assists | frozen_selected_v1 |
| steals | frozen_selected_v1 |
| threes | frozen_selected_v1 |

## Gate 2 changed-candidate bootstrap results

Candidate-minus-frozen deltas are favorable when negative.

| Prop | Metric | Point delta | 95% game-cluster bootstrap CI | P(candidate better) |
|---|---|---:|---:|---:|
| assists | brier | -0.000982 | [-0.001844, -0.000160] | 0.9894 |
| assists | logloss | -0.002022 | [-0.003776, -0.000335] | 0.9902 |
| points_assists | brier | -0.002551 | [-0.004870, -0.000158] | 0.9830 |
| points_assists | logloss | -0.005124 | [-0.009866, -0.000250] | 0.9812 |
| points_rebounds | brier | -0.002532 | [-0.005054, -0.000012] | 0.9756 |
| points_rebounds | logloss | -0.004903 | [-0.010093, 0.000311] | 0.9674 |

Gate 2 used 5,000 game-cluster bootstrap resamples.

Development-market-superior flags under the locked retrospective definition: **steals, points_assists**.

These are development findings only.

## Prospective integrity

The primary prospective sample requires the same cryptographically identified T-20m capture for projection and market pricing.

Each eligible capture contains hashes for games, active players, injuries, lineups, and player props.

Production fails closed when the required snapshot lineage is unavailable or mismatched.

Direct live-API prediction/pricing is not eligible for Gate 3 external-test certification.

## Recommended audit order

1. `research/v2_gate1_lock/`
2. `research/v2_gate2_certification_outputs/`
3. `research/v2_gate3_lock/`
4. `research/v2_gate3_deployment_artifacts/`
5. `research/v2_gate3_capture_lock/`
6. `src/nba_prop_quant/gate3_v2.py`
7. `src/nba_prop_quant/prospective_snapshot.py`
8. `src/nba_prop_quant/production.py`
9. `scripts/10_predict_slate.py`
10. `scripts/15_price_markets.py`
11. `tests/test_gate3_v2.py`
12. `tests/test_prospective_snapshot.py`

## Reproduction

Verify every supplied `SHA256SUMS.txt` before inspecting results.

The GitHub Release includes a versioned runtime bundle containing the exact model files and source used in the cold-room rehearsal.

The cold-room suite passed **43 tests** before this reviewer package was created.

See `KNOWN_LIMITATIONS.md` before interpreting any performance claim.
