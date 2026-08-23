# NBA Fair-Price Feeds v1

## Design

One frozen pricing engine feeds two destination adapters:

```text
nba_prop_quant_20260818T205213Z
        |
        v
nba_fair_price_v1
        |
        +-- wizardofodds_nba_feed_v1
        |
        +-- bet365_nba_feed_v1
```

Destination adapters must never change the statistical price.

## Canonical pricing

The core reuses the frozen production primitives:

- `price_single_prop_frame`
- `price_combo_lines`
- `calibrate_over_probability`
- `calibrated_unconditional_probabilities`
- `fair_american`

Combo dependence comes only from the frozen
`combo_dependence_policy.json`.

The deterministic combo seed is exactly the production formula from
`scripts/15_price_markets.py`.

The canonical feed is market-independent: no sportsbook odds are required or
used to produce fair probabilities.

## Probability semantics

`raw_*` fields are the pre-market-calibration model probabilities.

`selected_*` fields are the frozen selected/calibrated production
probabilities.

Push probability is preserved exactly:

```text
selected_p_push = raw_p_push
selected_p_over = (1 - p_push) * selected_q_over_nonpush
selected_p_under = (1 - p_push) * selected_q_under_nonpush
```

The public `fair_*` prices are based on the selected non-push probabilities.

## Quote mode

Create a request file with:

```text
game_id
player_id
prop_type
line_value
```

Optional destination mapping fields are passed through.

## Surface mode

Surface mode is independent of sportsbook quotes. The caller must explicitly
provide both `--surface-radius` and `--surface-step`; these are presentation
parameters, not frozen model parameters.

## No wagering instruction

The canonical feed always sets:

```text
market_independent = true
auto_bet = false
```
