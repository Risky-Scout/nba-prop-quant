# NBA Prop Model Constant Inventory

- Python files scanned: **73**
- Python literal occurrences: **10789**
- Named/default constants: **573**
- Shell constant/literal occurrences: **127**
- Configuration/policy leaf constants: **943**
- Parse errors: **0**

## Scope

The literal inventory includes every non-docstring `ast.Constant` occurrence in `src/nba_prop_quant`, `scripts`, and `ops`. This deliberately includes strings/column names/messages as well as numeric thresholds so the register cannot silently omit a behavior-affecting literal.

The named register separately identifies literal module/class assignments, function defaults, keyword-only defaults, and argparse defaults/choices/const values.

The shell register records hard-coded shell assignments plus numeric and quoted literal occurrences from `scripts/**/*.sh` and `ops/**/*.sh`.
The config register recursively records scalar leaves from model/config JSON, YAML, and TOML inputs.

## Literal types

- `NoneType`: 386
- `bool`: 474
- `bytes`: 14
- `ellipsis`: 9
- `float`: 690
- `int`: 899
- `str`: 8317

## Named constant kinds

- `argparse_choices`: 4
- `argparse_default`: 43
- `class_annotated_assignment`: 35
- `function_annotated_assignment`: 36
- `function_assignment`: 185
- `function_default`: 109
- `kwonly_default`: 2
- `module_annotated_assignment`: 1
- `module_assignment`: 92
- `nested_annotated_assignment`: 4
- `nested_assignment`: 62

## Python files by literal count

- `ops/grade_external_test_capture.py`: 699
- `src/nba_prop_quant/features.py`: 451
- `ops/inventory_model_constants.py`: 412
- `scripts/09_backtest.py`: 412
- `ops/build_senior_quant_review_dossier.py`: 385
- `scripts/09e_evaluate_oof_calibrated_betting.py`: 381
- `scripts/09b_diagnose_market_backtest.py`: 376
- `ops/qa_grading_synthetic_integration.py`: 331
- `ops/capture_external_test_day.py`: 325
- `ops/qa_replay_2025_grading.py`: 313
- `src/nba_prop_quant/slate.py`: 298
- `scripts/freeze_production_manifest.py`: 293
- `scripts/09c_fit_probability_calibration.py`: 283
- `scripts/06b_fit_mean_ensemble.py`: 271
- `scripts/08d_refine_copula_shrinkage_cv.py`: 267
- `src/nba_prop_quant/production.py`: 253
- `scripts/07_fit_marginals.py`: 250
- `scripts/09d_select_probability_calibration.py`: 212
- `scripts/08_fit_copula.py`: 211
- `scripts/08c_tune_copula_shrinkage.py`: 200
- `scripts/06_train_targets.py`: 194
- `ops/audit_senior_quant_dossier_clarity.py`: 189
- `scripts/15_price_markets.py`: 189
- `ops/build_full_model_package.py`: 181
- `scripts/01_ingest_history_resume.py`: 181
- `ops/build_senior_quant_reviewer_handoff.py`: 177
- `src/nba_prop_quant/game_context.py`: 175
- `src/nba_prop_quant/live.py`: 168
- `src/nba_prop_quant/normalize.py`: 165
- `src/nba_prop_quant/market.py`: 163
- `src/nba_prop_quant/pricing.py`: 161
- `src/nba_prop_quant/api.py`: 152
- `ops/audit_pricing_grading_contract.py`: 142
- `src/nba_prop_quant/distributions.py`: 141
- `scripts/06c_select_mean_models.py`: 137
- `ops/preflight_current_season_refresh.py`: 129
- `src/nba_prop_quant/model.py`: 119
- `scripts/10_predict_slate.py`: 114
- `scripts/10a_validate_production_contract.py`: 110
- `scripts/08b_bootstrap_copula_crps.py`: 105
- `scripts/05_train_minutes.py`: 82
- `scripts/14_live_price.py`: 82
- `src/nba_prop_quant/ingest.py`: 80
- `scripts/03_tune_dynamic_priors.py`: 57
- `src/nba_prop_quant/copula.py`: 57
- `src/nba_prop_quant/availability.py`: 56
- `src/nba_prop_quant/experience.py`: 47
- `ops/verify_external_test_capture.py`: 43
- `ops/verify_external_test_grade.py`: 42
- `scripts/verify_frozen_manifest.py`: 42
- `src/nba_prop_quant/kalman.py`: 38
- `ops/grade_external_test_day.py`: 34
- `scripts/16_explain_projection.py`: 32
- `ops/run_senior_quant_review_preflight.py`: 30
- `scripts/00_check_api.py`: 28
- `scripts/12_collect_live_box.py`: 28
- `src/nba_prop_quant/ctmc.py`: 28
- `scripts/13_train_live_ingarch.py`: 27
- `src/nba_prop_quant/pipeline.py`: 25
- `scripts/17_build_completed_history.py`: 23
- `src/nba_prop_quant/decay.py`: 23
- `src/nba_prop_quant/storage.py`: 23
- `src/nba_prop_quant/interpret.py`: 21
- `scripts/11_collect_snapshots.py`: 20
- `src/nba_prop_quant/settings.py`: 18
- `ops/run_dossier_review_and_package.py`: 17
- `scripts/01_ingest_history.py`: 17
- `scripts/02_build_base.py`: 14
- `src/nba_prop_quant/sequence.py`: 14
- `scripts/04_build_features.py`: 9
- `ops/list_external_test_captures.py`: 8
- `ops/list_external_test_grades.py`: 8
- `src/nba_prop_quant/__init__.py`: 1
