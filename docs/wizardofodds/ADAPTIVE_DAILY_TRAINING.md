# Adaptive Daily Training

Production Step 3C. One NBA day's refit of the numerical parameters, under an
architecture that was already chosen and certified.

Implemented by `src/nba_prop_quant/adaptive_training.py`, operated through
`ops/run_adaptive_daily_fit.py`, and locked by
`models/frozen_manifests/nba_prop_quant_v2_adaptive_update_protocol.json`.

---

## 1. Fitting is not selecting

The whole design turns on one distinction.

**Fitting** re-estimates parameters under a decision that has already been
made. Yesterday's booster is replaced by today's booster; the model is the
same model.

**Selecting** makes that decision. Choosing XGB over an ensemble, ZINB over
negative binomial, a dependence lambda, a calibration method, a Gate 3 route.

A nightly job that quietly re-selects is no longer running the model that was
certified, and nobody finds out until a claim is challenged. So every known
selector entry point is denied for the duration of a fit, and every frozen
policy file is hashed before the run and re-verified after each stage. If one
moved, the candidate fails before it can be registered.

Daily fitting is therefore not daily model selection, and registering a
candidate is not promoting it.

---

## 2. What the protocol freezes

`nba_prop_quant_v2_adaptive_update_protocol.json` locks the *methodology*, not
any fitted value. It carries `contains_fitted_values: false`, and a test walks
it to prove no parameter leaked in.

It hashes and therefore binds itself to:

- the Step 3A adaptive architecture contract
- the Step 3B adaptive serving-source contract
- the model config projection (`config_hash`)
- the feature schema identity (`feature_schema_hash`)
- every frozen selection and policy digest
- the Gate 3 candidate policy identity

| Frozen | Value |
| --- | --- |
| Mean routing | PTS `xgb`, REB `ensemble`, AST `xgb`, STL `xgb`, BLK `ensemble`, 3PM `xgb` |
| Marginal family | `zinb`, all six primitive targets |
| Dependence lambda | `rebounds_assists` 0.85, all other combos 0.0 |
| Calibration routing | `raw` for blocks and steals, `prop` for the other eight |
| Core seed / Gate 3 seed / `n_jobs` | 73 / 20260830 / 2 |
| Training floors | raw history from 2001, mean models from 2015 |
| OOF methodology | season-keyed walk-forward only |

The training window's *end* is not frozen. It expands through each day's
verified cutoff, and the number of season-keyed folds grows naturally as
seasons become eligible. Nothing pins the end to 2025 and nothing hardcodes a
fold count.

---

## 3. The daily DAG

```
input_snapshot → feature_build → minutes_fit → target_fit
  → ensemble_weight_fit → marginal_fit → dependence_fit
  → calibration_fit → gate3_fit → candidate_assembly
  → validation → registration
```

Every stage calls the existing certified fitting function and supplies the
frozen decision rather than deriving one:

| Stage | Reuses |
| --- | --- |
| `feature_build` | `features.build_base_frame`, `features.add_dynamic_priors` |
| `minutes_fit` | `model.season_walk_forward_oof_minutes`, `model.fit_minutes_model` |
| `target_fit` | `model.season_walk_forward_oof_target`, `model.fit_target_model` |
| `ensemble_weight_fit` | `scripts/06b_fit_mean_ensemble.fit_simplex_weights` |
| `marginal_fit` | `scripts/07_fit_marginals.fit_candidate`, forced to `zinb` |
| `dependence_fit` | `copula.GaussianCopula.fit` |
| `calibration_fit` | `scripts/09c_fit_probability_calibration.fit_platt` |
| `gate3_fit` | `HistGradientBoostingRegressor` with the frozen Gate 3 seed and the 36-feature contract |

The functions that live in `scripts/` are reached by loading the module, the
same way `scripts/14_build_gate3_deployment_artifacts.py` already reaches the
Gate 2 module. That keeps exactly one definition of the mathematics: the daily
fit cannot drift from the certified code because it *is* the certified code.
No training script was modified to make this work.

The selection halves of those scripts are never called. `select_distribution`,
the mean-model gate in `06c`, the lambda CV in `08d`, the method choice in
`09d` and Gate 2 certification are all on the deny list.

---

## 4. Data, cutoff and the off day

The trainer never fetches data. It consumes a rolling state that the Step 3B
refresh already committed, and it verifies that state before spending time
fitting: every dataset is re-read and re-fingerprinted, and a mismatch stops
the run. A record the data no longer matches is worth nothing.

The training cutoff is **derived, never supplied**. It is the latest completed
history date the verified state actually contains, and it must be strictly
before the slate date. A caller may assert a narrower cutoff; a caller asking
for a later one is refused, because that would widen the information set
beyond what the data supports. `games` is excluded from the derivation since it
legitimately holds future scheduled rows.

### NO_NEW_TRAINING_DATA

Before fitting, the trainer builds the training-data manifest from the data
root and hashes it. The manifest is deterministic and carries no wall-clock
field, so an unchanged information set produces an unchanged hash. If that
hash and the cutoff both match the parent fit, the run returns
`NO_NEW_TRAINING_DATA`: no fit, no new `fit_id`, no promotion-state change.

That is the correct outcome for an NBA off day or an offseason day, and it
costs a few file hashes rather than a full retrain. Because the manifest also
covers the contracts, the protocol and the source commit, a code change with
unchanged data is still recognised as a genuinely new fit.

---

## 5. Isolated staging

Each run gets a unique workspace under the explicit work root:

```
<work-root>/
    staging/fit_<YYYYMMDD>_<unique>/
        inputs/        immutable snapshot of the rolling inputs
        models/        ordinary training outputs, written here and only here
        processed/     intermediate frames
        candidate/     the artifact tree that will be registered
        reports/
    locks/adaptive_daily_fit.lock
    reports/benchmark_<slate-date>.json
```

The workspace is refused if it sits inside the repository or overlaps the
registry's immutable fits. Existing training code may write its ordinary
`models/` paths, provided they resolve inside here. A failed candidate cannot
touch the serving tree, a finalized fit, `current_good_fit_id` or
`previous_good_fit_id`.

Inputs are snapshotted once and re-hashed; if they moved during the copy the
run stops rather than fit a moving information set. An exclusive `flock` under
the work root prevents two daily fits racing the same generation; a second
runner gets `ALREADY_RUNNING`.

---

## 6. Calibration rules

Recalibration runs whenever a candidate is produced, over an expanding
eligible sample through the training cutoff. No event dated on or after the
slate date may be used.

`raw` routes stay raw. Blocks and steals have no fitted calibration
parameters, and asking for them raises: manufacturing a Platt pair for a raw
route would silently change the calibration methodology.

A fitted `prop` unit needs at least **100 eligible rows** and **two outcome
classes**. Below that, it reuses the corresponding parameter from the
current or previous promoted good fit, but only when that parameter was fitted
under the same method, and the reuse is recorded in candidate metadata. With no
compatible prior parameter the candidate fails. An uncalibrated substitute is
never silently used.

---

## 7. Gate 3

Routing is frozen: which props receive a role adjustment does not change.
Only the numerical parameters under those frozen routes are refit — the
role/minutes model on eligible pre-cutoff data with seed 20260830 and the
36-feature contract, the AST role-shock coefficients, the P+A and P+R
role-increment gamma and standardization, and the role-state artifacts.

No Gate 2 or Gate 3 certification runs, and no new adjustment is invented.

---

## 8. Registration and validation

The candidate registers through the Step 3A immutable fit registry. The
registry is never bypassed, and **registration does not promote**.

The adaptive update protocol, the architecture contract, the serving-source
contract, the model config and every frozen policy travel inside the candidate
tree, so their hashes contribute to the immutable fit content.

Step 3C records the thirteen checks it can establish truthfully from an offline
fit:

`training_completed`, `data_refresh_valid`, `history_regression_check`,
`advanced_coverage_check`, `required_artifacts_present`,
`artifact_hashes_valid`, `finite_values_check`, `feature_schema_match`,
`architecture_contract_match`, `source_lineage_match`, `marginal_fit_valid`,
`calibration_valid`, `prediction_smoke_test`.

It deliberately leaves two unset:

| Deferred to Step 3D | Why |
| --- | --- |
| `gate3_role_readiness` | Needs a live captured lineup snapshot |
| `t20_protocol_compatible` | Needs a real T-20 capture |

Fabricating either would be asserting something about a capture that never
happened. Because they are unset, the registry refuses to promote the
candidate — which is the intended end state of Step 3C. Step 3D completes live
validation and owns promotion.

---

## 9. Running it

```bash
# verify environment, contracts and data state; fit nothing
ops/run_adaptive_daily_fit.py --dry-run \
    --slate-date 2026-11-15 \
    --data-root /srv/nba-prop-data \
    --work-root /srv/nba-prop-work

# real fit path, isolated staging, benchmark recorded, nothing registered
ops/run_adaptive_daily_fit.py --benchmark-only \
    --slate-date 2026-11-15 \
    --data-root /srv/nba-prop-data \
    --work-root /srv/nba-prop-work

# same real fit path, candidate registered immutably, never promoted
ops/run_adaptive_daily_fit.py --register-candidate \
    --slate-date 2026-11-15 \
    --data-root /srv/nba-prop-data \
    --work-root /srv/nba-prop-work \
    --registry-root /srv/nba-prop-fits
```

Every failure exits non-zero with a structured JSON object on stderr.

---

## 10. Benchmark

The benchmark instruments the production path; there is no parallel benchmark
mathematics. `--benchmark-only` and `--register-candidate` run the identical
DAG and differ only in whether the candidate is registered, and a test asserts
both modes execute the same stages in the same order.

Recorded per run, to `<work-root>/reports/benchmark_<slate-date>.json`, outside
the repository: start and end timestamps, total elapsed, elapsed per stage,
measured counts of XGBoost and other estimator fits, training and OOF row
counts, feature counts, CPU count, effective `n_jobs`, Python and package
versions, peak RSS via `resource`, and candidate artifact size.

Fit counts are **measured, not assumed**. The prior audit's 77 was an anchor
for one 2015–2025 window; the real count depends on how many seasons are
eligible on the day, and a test asserts that number appears nowhere in the
source.

### Required full-data benchmark

**FULL_DATA_BENCHMARK_REQUIRES_LOCAL_VALIDATION**

This environment contains no historical NBA data: `data/raw/seasons` and
`data/raw/advanced` are empty, and none of `features.parquet`,
`stack_training.parquet` or the market backtest inputs exist. A canonical
benchmark cannot be produced here, and synthetic data would not be a real
benchmark.

Run this once against the full local NBA data before merging:

```bash
/workspace/.venv/bin/python ops/run_adaptive_daily_fit.py \
    --benchmark-only \
    --slate-date <next NBA slate date, YYYY-MM-DD> \
    --data-root <full local NBA data root> \
    --work-root /tmp/nba_adaptive_benchmark \
    --project-root /workspace
```

It uses an isolated temporary work root, registers nothing and promotes
nothing. The report lands at
`/tmp/nba_adaptive_benchmark/reports/benchmark_<slate-date>.json`.

That run is also what validates `ProductionFitEngine` end to end. Its stages
call the certified fitting functions, but they have not been executed against
full historical data in this environment, and the module says so.

---

## 11. Relationship to the other steps

**Step 3A** owns the immutable registry, validation records, fail-closed
promotion and last-good rollback. Step 3C registers into it and stops.

**Step 3B** owns the rolling refresh, its crash-recoverable transaction and the
semantic state record Step 3C requires, plus the prediction-time cutoff and
slate-date semantics this trainer reuses.

**Step 3D** will schedule refresh → fit → validation → promotion → deployment,
complete the two live validation checks, and own promotion. No Step 3C code
promotes, and a test asserts the source contains no promotion call.
