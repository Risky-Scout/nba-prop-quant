# wizardofodds.com Integration Contract

## Purpose

`wizardofodds.com` should consume **model-implied fair probabilities/prices** from the frozen NBA prop model without requiring the website layer to know model internals.

The website integration should be versioned by:

```text
freeze_id = nba_prop_quant_20260818T205213Z
schema_version = 1
```

## Recommended output separation

Use two feeds.

### A. Model fair-price feed

This feed is independent of any sportsbook quote.

Recommended fields:

```text
schema_version
generated_at_utc
slate_date
freeze_id
freeze_stage
game_id
player_id
player_name
team_id
opponent_id
home_away
prop_type
line_value
expected_minutes
mu_selected
p_over_raw
p_under_raw
p_push_raw
q_over_raw_nonpush
p_over_selected
p_under_selected
p_push_selected
q_over_selected_nonpush
fair_over_american
fair_under_american
calibration_method
dependence_lambda
availability_status
history_latest_date
advanced_latest_date
projection_generated_at_utc
```

For combo props, include `dependence_lambda`; zero means the frozen production independence convolution.

### B. Optional sportsbook comparison feed

Only when current sportsbook prices are available:

```text
vendor
over_odds
under_odds
market_q_over
model_q_over_selected
model_edge_over
model_edge_under
model_ev_over
model_ev_under
preferred_side
preferred_edge
preferred_ev
market_opened_at_utc
priced_at_utc
```

Keep the fair-price feed usable even when no market quote exists.

## Probability semantics

For integer lines with push mass:

```text
q_over_nonpush = p_over / (p_over + p_under)
```

Do not silently collapse push probability into over or under.

The selected probability must follow the frozen RAW/PROP calibration policy. The website should not recalibrate probabilities.

## American fair odds

Recommended conversion from non-push probability `q`:

```text
if q >= 0.5:
    american = -100 * q / (1 - q)
else:
    american = 100 * (1 - q) / q
```

Round only for presentation. Preserve full-precision probabilities in the machine feed.

## Freshness

Every row must expose `generated_at_utc` and the frozen `freeze_id`.

The website should reject or visibly flag stale data rather than presenting stale fair prices as current.

## Fail-closed rules

Do not publish a row when:

- projection freeze ID is not the expected deployment;
- mandatory model features are missing;
- line value is invalid;
- selected probability is not finite or outside [0, 1];
- fair odds cannot be computed;
- player is OUT;
- historical data date is on/after the target slate;
- runtime schema differs from the approved website contract.

## Remaining live-runtime confirmation

Before wizardofodds.com relies on sportsbook-comparison fields, the first genuine 2026-27 `priced_markets.parquet` must confirm the exact runtime column names for selected side, selected edge, and selected EV.

That open item is operational schema confirmation, not evidence for or against model quality.
