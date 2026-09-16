# wizardofodds.com Integration Contract

## Purpose

`wizardofodds.com` consumes model-implied fair probabilities and prices from
the frozen NBA prop model without the website layer needing to know model
internals.

This document describes the schema the runtime **currently emits**, read from
the emitters themselves. Fields that do not exist yet are confined to
[section 10](#10-future-adaptive-lineage-additions-not-implemented) and are
marked as future work throughout.

---

## 1. Versioning

There is no single `schema_version`. The runtime emits **two separately
versioned feeds**, and both are currently at version 3.

| Feed | Emitter | Output | Version field | Value |
| --- | --- | --- | --- | --- |
| Projections | `scripts/10_predict_slate.py` | `<processed>/projections/<date>.parquet` | `projection_schema_version` | `3` |
| Priced markets | `scripts/15_price_markets.py` | `<processed>/priced_markets/<date>.parquet` | `market_pricing_schema_version` | `3` |

Both feeds also carry the deployment identity `freeze_id`, `freeze_stage` and
`manifest_sha256`, and the Gate 3 identity `gate3_candidate_policy_id`,
`gate3_policy_lock_commit` and `gate3_deployment_manifest_sha256`.

The current deployment is `freeze_id = nba_prop_quant_20260818T205213Z` at
`freeze_stage = external_test_deployment`.

The repository holds no changelog describing versions 1 and 2. Both emitters
were created at version 2 and moved to version 3 when the certified Gate 3
production candidate was integrated, which added the role-state, candidate
override and capture-lineage fields.

---

## 2. What each feed is

The two feeds are **not** "fair prices" and "optional market comparison". The
split is different, and integrating against the older description will fail.

**Projections** are per player and per *target*. This feed has no betting
line, no over/under probability and no fair odds, because pricing needs a
posted line. It carries the distribution summary the pricing step consumes.

**Priced markets** are per player, per *prop type* and per posted line. Every
probability, edge, EV and fair-odds field lives here, including the fields the
older document listed under the projections feed.

A website surfacing fair prices consumes **priced markets**.

---

## 3. Projections feed (`projection_schema_version = 3`)

Written by `scripts/10_predict_slate.py`.

### Contract-relevant fields

```text
projection_schema_version
projection_generated_at_utc
freeze_id
freeze_stage
manifest_sha256
gate3_candidate_policy_id
gate3_policy_lock_commit
gate3_deployment_manifest_sha256
gate3_external_test_candidate
game_id
player_id
player_name
first_name
last_name
date
season
team_id
opponent_id
is_home
expected_minutes
availability_status
availability_out
minutes_override_applied
gate3_role_ready
history_latest_date
advanced_latest_date
```

### Capture lineage (snapshot mode only)

When the slate is built from a captured Gate 3 prospective snapshot rather
than a live fetch, these are merged on `game_id` and identify exactly which
capture the projection was produced from:

```text
gate3_capture_id
gate3_capture_window_id
gate3_capture_offset_minutes
gate3_captured_at_utc
gate3_input_source
gate3_component_sha256_json
```

`gate3_capture_offset_minutes` is the T-N capture offset, 20 under the frozen
T-20m certification protocol. These fields are the basis for any freshness
check: prefer them to wall-clock arrival time.

### Per-target distribution summary

For each of the six primitive targets `pts`, `reb`, `ast`, `stl`, `blk`,
`fg3m`:

```text
mu_selected_<target>          selected mean under the frozen routing policy
mean_model_mode_<target>      which frozen mean model produced it
<target>_mean                 marginal distribution mean
<target>_p10                  10th percentile
<target>_p50                  median
<target>_p90                  90th percentile
```

Rows for players whose `availability_out` is 1 carry `NaN` in every
`<target>_{mean,p10,p50,p90}` field.

### Everything else in the file

The projections parquet also carries the full modelling feature set: prior
rate features, team and opponent context priors, advanced priors, decay and
Kalman dynamic priors, experience-curve values, per-target component means and
the Gate 3 role-state features. These are model internals. They are written so
the pricing step and any audit can reproduce a row, and they are **not part of
the website contract**. Do not build website behaviour on them.

---

## 4. Priced markets feed (`market_pricing_schema_version = 3`)

Written by `scripts/15_price_markets.py`, which reads the projections parquet
for the same date and joins it to captured sportsbook quotes on
`game_id` and `player_id`.

In the default `--input-source snapshot` mode the quotes come from a local
Gate 3 prospective snapshot on disk and **no network call is made**.

### Quote identity

```text
id                  sportsbook quote record id
vendor
game_id
player_id
prop_type
line_value
market_type         must be over_under to be priced
over_odds           posted American odds
under_odds          posted American odds
opened_at
updated_at
preprice_hold_pct   posted hold, used by the quote filter
quote_filter_reason empty string on priced rows
quote_eligible      true on priced rows
```

### Raw model probabilities

```text
p_over              P(result > line), including push mass
p_under             P(result < line), including push mass
p_push              P(result == line); non-zero only for integer lines
q_over_nonpush      p_over / (p_over + p_under)
q_under_nonpush     p_under / (p_over + p_under)
dependence_lambda   combo dependence; 0.0 for single-stat props
raw_p_over          copies of the five fields above, taken before calibration
raw_p_under
raw_p_push
raw_q_over_nonpush
raw_q_under_nonpush
```

### Calibrated (selected) probabilities

```text
calibrated_q_over_nonpush     selected non-push over probability
calibrated_q_under_nonpush    1 - calibrated_q_over_nonpush
calibrated_p_over             unconditional, = (1 - p_push) * q_over
calibrated_p_under            unconditional
calibrated_p_push             = raw_p_push
calibration_method            raw | prop | global | a Gate 3 method
calibration_intercept
calibration_slope
base_calibrated_q_over_nonpush   value before any Gate 3 override
base_calibration_method
base_calibration_intercept
base_calibration_slope
gate3_candidate_applied       whether a Gate 3 override replaced the value
```

### Market implied and devigged

```text
market_raw_implied_over     implied probability of over_odds
market_raw_implied_under    implied probability of under_odds
market_hold_pct             (over_implied + under_implied - 1) * 100
market_devig_q_over         over_implied / (over_implied + under_implied)
market_devig_q_under        under_implied / (over_implied + under_implied)
```

### Edge, EV and preferred side

```text
raw_edge_over                 raw_q_over_nonpush - market_devig_q_over
raw_edge_under
calibrated_edge_over          calibrated_q_over_nonpush - market_devig_q_over
calibrated_edge_under
calibrated_ev_over            EV of the over at the posted odds
calibrated_ev_under
model_preferred_side          "over" or "under", by higher EV
model_preferred_edge          edge on the preferred side
model_preferred_ev            EV on the preferred side
```

`model_preferred_side`, `model_preferred_edge` and `model_preferred_ev` are the
runtime names for what the earlier draft of this document called
`preferred_side`, `preferred_edge` and `preferred_ev`.

### Fair model odds

```text
calibrated_fair_over_american
calibrated_fair_under_american
```

### Monitoring flags

```text
monitor_edge_ge_0_01
monitor_edge_ge_0_02
monitor_edge_ge_0_03
monitor_edge_ge_0_05
monitor_edge_ge_0_075
monitor_edge_ge_0_10
auto_bet                      always false in the current runtime
betting_threshold_policy      "monitor_only_no_threshold_frozen"
```

No betting threshold is frozen. These flags are monitoring only and must not
be presented as recommendations.

### Gate 3 candidate override detail

```text
gate3_candidate_q_over_nonpush
gate3_candidate_method
gate3_candidate_gamma
gate3_candidate_standardization_mean
gate3_candidate_standardization_std
gate3_candidate_intercept
gate3_candidate_slope
```

Populated only for the three props whose Gate 3 routing changed: `assists`,
`points_assists` and `points_rebounds`.

### Capture lineage and run metadata

```text
priced_at_utc
freeze_id
freeze_stage
manifest_sha256
market_pricing_schema_version
gate3_candidate_policy_id
gate3_policy_lock_commit
gate3_deployment_manifest_sha256
gate3_capture_id
gate3_capture_window_id
gate3_captured_at_utc
gate3_capture_offset_minutes
gate3_input_source
gate3_component_sha256_json
gate3_external_test_record
external_test_record
```

Quote-side lineage columns that collide with projection-side columns are
suffixed `_market` by the join, for example `gate3_capture_id_market`.

The priced file also carries every projection column described in
[section 3](#3-projections-feed-projection_schema_version--3), because the join
is an inner merge of the whole projection row.

---

## 5. Probability semantics

For integer lines with push mass:

```text
q_over_nonpush = p_over / (p_over + p_under)
```

Push probability is never collapsed into over or under. `p_push` is non-zero
only when the line is an integer.

The selected probability follows the frozen RAW/PROP calibration policy. Under
that policy `blocks` and `steals` use `raw` and the other eight props use
`prop`. The website must not recalibrate probabilities.

`calibrated_*` fields are the selected values. `raw_*` fields are retained for
audit and should not be published as the model's price.

---

## 6. Fair American odds

The runtime conversion from a non-push probability `q`, in
`src/nba_prop_quant/pricing.py`:

```python
q = clip(q, 1e-6, 1 - 1e-6)

if q >= 0.5:
    american = round(-100 * q / (1 - q))
else:
    american = round(100 * (1 - q) / q)
```

The emitted `calibrated_fair_*_american` fields are already rounded integers.
Preserve the full-precision probability fields for any further computation.

---

## 7. Rejected quotes

Quotes that are not priced are written to
`<processed>/priced_markets/rejected/<date>.parquet` with the reason in
`quote_filter_reason`:

| Reason | Meaning |
| --- | --- |
| `unsupported_market_type` | Not an over/under market |
| `unsupported_or_unfrozen_prop` | Prop not in the frozen calibration policy |
| `missing_or_invalid_line_odds` | Line or odds missing or non-numeric |
| `hold_outside_frozen_range` | Posted hold outside the frozen acceptable range |
| `player_currently_out` | Player availability is OUT |
| `gate3_role_state_unavailable` | Gate 3 role state missing for a Gate 3 prop |

A rejected row is not a priced row. Never fall back to publishing one.

---

## 8. Prop coverage

Ten props are priced:

| Prop | Method |
| --- | --- |
| `points`, `rebounds`, `steals`, `blocks`, `threes` | ZINB marginal over the posted line |
| `assists` | Same, with the Gate 3 role-adjusted mean |
| `points_rebounds`, `points_assists`, `points_rebounds_assists` | Independent convolution, `dependence_lambda = 0.0` |
| `rebounds_assists` | Gaussian copula simulation, `dependence_lambda = 0.85` |

`stocks` (steals plus blocks) exists in the combo engine but is absent from the
frozen calibration policy, so it is filtered out and never priced.

---

## 9. Fail-closed rules

Do not publish a row when:

- `freeze_id` is not the expected deployment;
- `projection_schema_version` or `market_pricing_schema_version` differs from
  the approved contract version;
- `quote_eligible` is false or `quote_filter_reason` is non-empty;
- `availability_out` is 1;
- `calibrated_q_over_nonpush` is not finite or falls outside `[0, 1]`;
- `calibrated_fair_over_american` or `calibrated_fair_under_american` is
  missing;
- `line_value` is invalid;
- a Gate 3 prop has `gate3_role_ready` false;
- `history_latest_date` or `advanced_latest_date` is not strictly before the
  slate date.

The last rule is now also enforced upstream. `nba_prop_quant.slate` fails
closed before any feature is built when historical inputs are not strictly
older than the slate date, so a leaking projection file should not be produced
at all. The website check remains as defence in depth.

---

## 10. Future adaptive lineage additions (not implemented)

**None of the following exists in the current runtime.** They are recorded so
the website contract can plan for them, and must not be integrated against
yet.

| Future field | Status |
| --- | --- |
| `fit_id` | **Not emitted.** Neither emitter references it. |
| `fit_promoted_at` | Not emitted. |
| `fit_source_commit_sha` | Not emitted. |

`fit_id` identifies one immutable daily fit in the adaptive fit registry added
in Production Step 3A and documented in
`docs/wizardofodds/ADAPTIVE_FIT_REGISTRY.md`. It is a distinct namespace from
`freeze_id` and does not replace it.

Propagating `fit_id` into the projections and priced-markets feeds is Step 3D
work. Until that lands, deployment identity on both feeds is `freeze_id`,
`freeze_stage` and `manifest_sha256`.

---

## 11. Open items

**Runtime column names are resolved.** The previous open item asking for a
genuine `priced_markets.parquet` to confirm the selected side, edge and EV
column names is closed: they are `model_preferred_side`,
`model_preferred_edge` and `model_preferred_ev`, read from the emitter.

**Two serving-path defects were found and have since been corrected.** Both
blocked a feed from being produced at all. Neither changed any model
mathematics, and neither is evidence for or against model quality.

*1. The Gate 3 role-minutes model required two features the slate never built.*
`research/v2_gate3_deployment_artifacts/role_minutes_model.joblib` declares 36
`feature_names`. Two of them, `player_game_number` and `team_game_number`, were
produced only by `build_base_frame` on the historical training path, and
`apply_gate3_role_state` fails closed on any absent feature, so the projection
step raised before Gate 3 role state could be attached. Every `gate3_role_*`
and `gate3_delta_*` field in
[section 3](#3-projections-feed-projection_schema_version--3), and the
`gate3_role_state_unavailable` rejection path in
[section 7](#7-rejected-quotes), depend on that step completing.

`build_upcoming_slate_features` now emits both, reproducing the training
definitions exactly: `player_game_number` is the count of the player's games
strictly before the slate, which is what `build_base_frame` assigns as
`career_games_prior`; `team_game_number` is one past the team's games already
played in the season, which keeps `season_progress` equal to
`(team_game_number - 1) / 82`. Parity against `build_base_frame` is asserted in
the test suite, and the fail-closed feature check is unchanged.

*2. `external_test_record` was read before it was assigned.*
`scripts/15_price_markets.py` read the column while building
`gate3_external_test_record` immediately before creating it, and nothing
upstream produces it, so the lineage tail raised `KeyError`.

The two fields now have explicit, separately tested semantics:

| Field | Meaning |
| --- | --- |
| `external_test_record` | Run level: this pricing run's freeze stage is `external_test_deployment`. Established before anything reads it. |
| `gate3_external_test_record` | Row level: the projection marked the row `gate3_external_test_candidate` **and** the run is an external-test deployment. |

Because `scripts/15_price_markets.py` is byte-pinned to the architecture
reference by the Gate 3 runtime contract, that contract is left untouched and
the corrected serving source is locked separately by
`models/frozen_manifests/nba_prop_quant_v2_adaptive_serving_source_contract.json`,
described in `docs/wizardofodds/V2_REFERENCE_PROVENANCE_ADDENDUM.md`.

**One limitation remains open.**
`scripts/19_build_wizardofodds_runtime_bundle.py` still verifies serving
sources against the architecture reference alone, so building a runtime bundle
from the corrected source fails its frozen-source check. Teaching the bundle
builder about the adaptive serving contract is deployment work and is out of
scope for this change.
