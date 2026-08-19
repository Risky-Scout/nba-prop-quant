# NBA Player Box-Score Prop Projection System
## Senior Quant Review Dossier — 2026–27 External-Test Deployment

Generated UTC: `2026-08-18T22:33:53.428259+00:00`

## 1. Review posture

**This system is ready for expert methodology review. It is not presented as having a proven broad sportsbook edge.**

The research record separates model development from the first untouched external market test. The 2025 sportsbook sample is development/model-selection evidence; 2026–27 is the first prospective external test. No automatic betting threshold was selected from the 2025 ROI grid.

- Frozen deployment ID: `nba_prop_quant_20260818T205213Z`
- Freeze stage: `external_test_deployment`
- Strict selected-mean OOF rows: `273082`
- Monitoring edge grid: `1%, 2%, 3%, 5%, 7.5%, 10%`
- Automatic betting threshold: **none**

## 2. Data and information-set design

- Standard historical NBA player-game backbone: season labels 2001 through 2025.
- Advanced statistics are used only where the BDL archive supports them.
- BALLDONTLIE does not supply birth date in the player schema used here; production uses career-experience features rather than claiming exact biological age.
- Current injury data are treated as a current snapshot. Historical injury reconstruction is not performed unless an injury snapshot was actually archived before tip-off.
- Pregame feature construction is chronological; outcome fields are not allowed to enter the future row.

### Current 2025 market inventory read from local artifacts

```json
{
  "opened_at_max_utc": "2026-06-14T00:28:14.705000+00:00",
  "opened_at_min_utc": "2026-01-08T19:59:46.237000+00:00",
  "opening_prop_games": 381,
  "opening_prop_players": 551,
  "opening_prop_rows": 356618,
  "opening_prop_vendors": 10,
  "over_under_rows": 154736,
  "priced_2025_events": 51723,
  "priced_2025_games": 381,
  "priced_2025_players": 410,
  "priced_2025_rows": 145083,
  "priced_2025_vendors": 9
}
```

The market sample must be treated as non-random and coverage-limited. A senior reviewer should explicitly assess selection effects from the available opening-prop window, vendor mix, multiple lines per event, and quote-shopping scope.

## 3. Mean-model selection

| target | selected_mode | w_xgb | w_decay | w_kalman | pooled_improvement_vs_xgb_pct | positive_strict_folds | strict_fold_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ast | xgb | 1 | 0 | 0 | 0.0337167 | 5 | 8 |
| blk | ensemble | 0.635748 | 0.302536 | 0.0617165 | 0.0957102 | 6 | 8 |
| fg3m | xgb | 1 | 0 | 0 | 0.0331431 | 5 | 8 |
| pts | xgb | 1 | 0 | 0 | -0.00141121 | 3 | 8 |
| reb | ensemble | 0.625572 | 0.328149 | 0.0462787 | 0.168049 | 8 | 8 |
| stl | xgb | 1 | 0 | 0 | 0.00418765 | 5 | 8 |

Selection is intentionally conservative: a blend is not promoted merely because pooled RMSE is marginally better.

## 4. Marginal distributions

| target | selected_marginal |
| --- | --- |
| ast | zinb |
| blk | zinb |
| fg3m | zinb |
| pts | zinb |
| reb | zinb |
| stl | zinb |

The production distribution layer uses a mean-preserving zero-inflated negative-binomial parameterization. For a ZINB marginal, the count-component mean is adjusted so that the unconditional expected value equals the selected mean forecast.

Reviewer focus: zero-inflation identification, dispersion stability, tail calibration, integer-line push mass, and whether minutes/role uncertainty should be integrated more explicitly.

## 5. Combination-prop dependence

| combo | lambda |
| --- | --- |
| points_assists | 0 |
| points_rebounds | 0 |
| points_rebounds_assists | 0 |
| rebounds_assists | 0.85 |
| stocks | 0 |

Production dependence is intentionally sparse. Combination markets that did not clear the stability gate revert to independence rather than forcing a noisy copula effect.

## 6. Probability calibration and market benchmark

| prop_type | selected_method |
| --- | --- |
| assists | prop |
| blocks | raw |
| points | prop |
| points_assists | prop |
| points_rebounds | prop |
| points_rebounds_assists | prop |
| rebounds | prop |
| rebounds_assists | prop |
| steals | raw |
| threes | prop |

Calibration is chronological and frozen before the 2026–27 external test.

### 2025 event-level proper scoring

| label | events | brier_raw | brier_selected | brier_market | logloss_raw | logloss_selected | logloss_market |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 40989 | 0.247862 | 0.242027 | 0.241515 | 0.691239 | 0.676379 | 0.675306 |
| assists | 3812 | 0.251298 | 0.246724 | 0.243107 | 0.69952 | 0.686669 | 0.679239 |
| blocks | 4344 | 0.202445 | 0.202445 | 0.201745 | 0.590845 | 0.590845 | 0.589733 |
| points | 4622 | 0.259392 | 0.250165 | 0.249347 | 0.715076 | 0.69349 | 0.691836 |
| points_assists | 3544 | 0.260466 | 0.250084 | 0.250103 | 0.719606 | 0.69333 | 0.693355 |
| points_rebounds | 4162 | 0.260046 | 0.249868 | 0.249628 | 0.718399 | 0.692877 | 0.692403 |
| points_rebounds_assists | 4431 | 0.262102 | 0.249611 | 0.249439 | 0.725523 | 0.692369 | 0.692025 |
| rebounds | 4421 | 0.251257 | 0.246793 | 0.246233 | 0.697282 | 0.686641 | 0.685501 |
| rebounds_assists | 3460 | 0.254892 | 0.248599 | 0.248577 | 0.707325 | 0.69034 | 0.690295 |
| steals | 4342 | 0.239125 | 0.239125 | 0.242251 | 0.671535 | 0.671535 | 0.677544 |
| threes | 3851 | 0.24034 | 0.239544 | 0.237022 | 0.6736 | 0.671451 | 0.666082 |

### Game-clustered uncertainty

| label | brier_vs_raw_delta | brier_vs_raw_ci_low | brier_vs_raw_ci_high | brier_vs_raw_prob_selected_better | brier_vs_market_delta | brier_vs_market_ci_low | brier_vs_market_ci_high | brier_vs_market_prob_selected_better | logloss_vs_raw_delta | logloss_vs_raw_ci_low | logloss_vs_raw_ci_high | logloss_vs_raw_prob_selected_better | logloss_vs_market_delta | logloss_vs_market_ci_low | logloss_vs_market_ci_high | logloss_vs_market_prob_selected_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | -0.00583476 | -0.00704324 | -0.00463492 | 1 | 0.00051194 | -0.000138472 | 0.00114158 | 0.0612 | -0.0148599 | -0.0178466 | -0.0118509 | 1 | 0.00107319 | -0.000237804 | 0.00241933 | 0.0564 |
| assists | -0.00457339 | -0.00670351 | -0.00243505 | 1 | 0.00361705 | 0.00185948 | 0.00538821 | 0 | -0.0128518 | -0.0180002 | -0.00788286 | 1 | 0.00742946 | 0.00383723 | 0.0110785 | 0 |
| blocks | 0 | 0 | 0 | 0 | 0.000699301 | -0.00112367 | 0.00252864 | 0.2272 | 0 | 0 | 0 | 0 | 0.00111199 | -0.00328812 | 0.00561092 | 0.3125 |
| points | -0.00922736 | -0.0116789 | -0.00677434 | 1 | 0.000818076 | 3.14018e-05 | 0.00160977 | 0.0212 | -0.0215861 | -0.0270176 | -0.0161379 | 1 | 0.00165349 | 7.43954e-05 | 0.00327175 | 0.0194 |
| points_assists | -0.0103813 | -0.013603 | -0.00723186 | 1 | -1.91725e-05 | -0.00119982 | 0.00112178 | 0.516 | -0.0262759 | -0.033934 | -0.0188495 | 1 | -2.49904e-05 | -0.00229049 | 0.00226593 | 0.5099 |
| points_rebounds | -0.0101789 | -0.0130266 | -0.00736107 | 1 | 0.000239415 | -0.000667782 | 0.00115652 | 0.2963 | -0.0255213 | -0.0324618 | -0.0187566 | 1 | 0.000474789 | -0.00137991 | 0.0023309 | 0.3107 |
| points_rebounds_assists | -0.012491 | -0.015827 | -0.00918561 | 1 | 0.000171525 | -0.000526472 | 0.000849997 | 0.3147 | -0.0331539 | -0.0414499 | -0.0252136 | 1 | 0.000344157 | -0.00104396 | 0.00172271 | 0.3095 |
| rebounds | -0.0044639 | -0.00626242 | -0.00270859 | 1 | 0.000559474 | -0.000926485 | 0.00204659 | 0.2284 | -0.0106402 | -0.0148221 | -0.00677097 | 1 | 0.00114089 | -0.00190509 | 0.00417332 | 0.239 |
| rebounds_assists | -0.00629296 | -0.00873225 | -0.00380093 | 1 | 2.22999e-05 | -0.00129502 | 0.00135112 | 0.4737 | -0.016985 | -0.0231851 | -0.0108621 | 1 | 4.47409e-05 | -0.00262509 | 0.00274915 | 0.4894 |
| steals | 0 | 0 | 0 | 0 | -0.00312622 | -0.00551377 | -0.000618549 | 0.9929 | 0 | 0 | 0 | 0 | -0.00600898 | -0.0115794 | -0.000703529 | 0.9864 |
| threes | -0.000796191 | -0.00175283 | 0.000166553 | 0.947 | 0.00252187 | 0.000628277 | 0.00440759 | 0.0042 | -0.00214939 | -0.00442245 | 7.14595e-05 | 0.97 | 0.00536866 | 0.00150998 | 0.00928203 | 0.003 |

Interpretation: calibration materially improved the raw model overall, but the market remained approximately tied/slightly better in aggregate. That is not evidence of a broad proven edge.

## 7. Development betting diagnostics

The 2025 ROI grid is retained only as a development diagnostic. It must not be used to choose a production threshold after the fact. Quote-level, best-vendor-event, and best-event scopes answer different questions and must remain separately reported.

## 8. Production architecture and governance

The deployment separates historical ingestion, minutes/target means, mean-preserving marginals, combo dependence, probability calibration, live pricing, immutable prospective capture, and separate post-event grading.

The frozen production manifest hashes the critical model artifacts and source files. Operational capture and grading are append-only evidence layers; original forecasts and quote snapshots are not rewritten after outcomes are known.

## 9. Grading/settlement QA before external testing

- Synthetic controlled integration test: **PASS**
- Real 2025 development-data grading replay: **PASS**
- Pricing ↔ grading source-contract audit: **PASS_WITH_RUNTIME_CONFIRMATION**

Synthetic QA is software evidence only. The real 2025 replay is schema/settlement realism evidence only. Neither is counted as new model-performance evidence.

## 10. Known limitations a senior reviewer should attack

- Market-sample selection and limited opening-prop coverage.
- Current injury snapshots do not reconstruct historical pre-tip information that was never archived.
- Minutes uncertainty may be under-propagated into final prop tails.
- DNP/missing-player outcomes are void candidates until book-specific rules are explicitly represented.
- Prop-specific calibration may drift in 2026–27.
- Dependence estimates can be weakly identified for small-history or role-changing players.
- Best-event ROI mixes predictive signal with quote availability across vendors.
- Multiple prop/vendor/threshold diagnostics create researcher degrees of freedom.
- Opening-line comparison does not replace a timestamped closing-line-value study.
- Early 2026–27 results require game-clustered uncertainty and should not be overinterpreted.

## 11. Claims permitted vs. not permitted

### Permitted

- Reproducible, interpretable research/production candidate with frozen 2026–27 policies.
- Walk-forward model selection and chronological probability calibration.
- 2025 market results are development evidence.
- Prospective capture/grading protocol preserves point-in-time evidence.

### Not permitted yet

- A proven broad sportsbook edge.
- An optimal production betting threshold selected from 2025.
- 2025 described as an untouched external market test.
- Fully reconstructed historical injury context.
- A claim that copula dependence improves every combo market.

## 12. Suggested senior-quant review agenda

1. Reproduce the frozen manifest and selected-policy tables.
2. Audit point-in-time feature availability and season/date boundaries.
3. Reproduce OOF minutes/target errors and conservative model-selection gates.
4. Validate mean-preserving ZINB identities and tail calibration.
5. Reproduce dependence-selection CV and independence fallbacks.
6. Reproduce chronological calibration and market proper-scoring comparison.
7. Challenge market-sample selection, vendor multiplicity, and quote-shopping assumptions.
8. Audit DNP/push/settlement semantics.
9. Review the prospective 2026–27 protocol before looking at external results.
10. Predeclare what evidence would justify a model revision versus normal sampling noise.

## 13. Reproducibility commands

```bash
python scripts/verify_frozen_manifest.py
python scripts/10a_validate_production_contract.py
python ops/qa_grading_synthetic_integration.py
python ops/qa_replay_2025_grading.py
python ops/audit_pricing_grading_contract.py
python ops/build_senior_quant_review_dossier.py
```

## 14. Artifact index

Machine-readable hashes: `review/SENIOR_QUANT_REVIEW_ARTIFACT_INDEX.json`

- `models/frozen_manifests/LATEST.json` — SHA-256 `7c70a1a82145ef0ea1ef69806a5edc363704a285df638796212b23810d8e6823`
- `models/mean_model_selection.json` — SHA-256 `d8961f7156514a7b5b92cde943c181566796254e46cea03f96bf6e7f7abbe438`
- `models/marginal_selection.json` — SHA-256 `5516f941bdb9d8d723650f46d90afb6f1a7d39de777f0dd95bb7060a6bfe84e3`
- `models/combo_dependence_policy.json` — SHA-256 `37dba3b7ff1c27d8cced09c6d8655c9c8ee7afc295427163a08846b0ccddf735`
- `models/market_probability_calibration_policy.json` — SHA-256 `484656bbc50ff111bb7990ccef75260a8f47b568c3f8a08816c82f65ab10f7a8`
- `data/processed/market_backtest/calibrated_oof/event_scoring_summary.csv` — SHA-256 `ed57cd1d10ae7b892dd936ff4de1ed724673a969c1880c4e54856396917a11eb`
- `data/processed/market_backtest/calibrated_oof/event_game_cluster_bootstrap.csv` — SHA-256 `79f90df6acd118ed7ce8aa6378d4d591cbc83225926972f52c1869d749a9e993`
- `data/processed/oof_selected_means.parquet` — SHA-256 `c64af789d3d8ec8232ed006afb5d00bd6e7731503cb85adb675a35d1d6dfff03`
- `models/marginals.joblib` — SHA-256 `e9d5ab0bcdb6635225a01526f3b682fecadb3d87d364bc93343f27f482cdc42f`
- `models/copula.joblib` — SHA-256 `4149fa702a7f31883786c966b245c711527f5b147f438e7112990a87a914bd39`
- `scripts/10_predict_slate.py` — SHA-256 `1d63c46b0cfc08ae117ec5bdddc6ef68e6c5c1b4eabeafc13889934f8875ed73`
- `scripts/15_price_markets.py` — SHA-256 `270e62c263a0d9ccc6e4f75762903d9fedecbb31c5f117ac99218c9af8d25778`
- `ops/capture_external_test_day.py` — SHA-256 `a7ca11b5d921dcfb231efa551e85951391f726ed203f13c9210cd61815dff279`
- `ops/grade_external_test_capture.py` — SHA-256 `2cd60d418e920291af3091d7d5135166f744d642194e13549be08de5e68d2bf5`
- `data/validation/grading_synthetic/20260818T223347Z/synthetic_grading_integration_report.json` — SHA-256 `9bd84a74a010c26003ba6a497dbede098e3e47263aa220cfece8cf860341a324`
- `data/validation/grading_replay_2025/date=2026-02-11/20260818T223348Z/real_2025_grading_replay_report.json` — SHA-256 `0f82d29a4f72de1703c8dbcb46a4e0e0be05e74a5b57bc43cf95663a10b0c4c3`
- `data/validation/pricing_grading_contract/pricing_grading_contract_audit.json` — SHA-256 `5343d00c6b034c3b885df5f70c5fa07221ecf3ea8a987b6334eed31651562bef`

## 15. Reviewer sign-off

- Reviewer:
- Review date:
- Reproduced freeze verification: YES / NO
- Material leakage concern: YES / NO
- Material calibration concern: YES / NO
- Material distributional concern: YES / NO
- Material dependence concern: YES / NO
- Material market-sample concern: YES / NO
- External-test protocol acceptable before results are viewed: YES / NO
- Required changes before external testing:
- Analyses to defer until a prespecified prospective sample exists:
