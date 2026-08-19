# Frozen NBA Prop Quant Production Manifest

- Freeze ID: `nba_prop_quant_20260818T205213Z`
- Created UTC: `2026-08-18T20:52:13.909596+00:00`
- Freeze stage: `external_test_deployment`
- Python: `3.14.5 (main, May 10 2026, 10:21:34) [Clang 21.0.0 (clang-2100.0.123.102)]`
- Platform: `macOS-26.3.1-arm64-arm-64bit-Mach-O`
- Pytest passed: `True`

## External-test rule

The 2025 sportsbook sample is development/model-selection data. The first external market test is 2026-27.

No betting threshold is frozen from 2025 ROI. The monitoring grid is recorded prospectively.

## Monitoring edge grid

1%, 2%, 3%, 5%, 7.5%, 10%

## Frozen files

### Model Artifacts

- `models/dynamic_params.json` — `b8951cf375fcd7b5b624823f5dfb7f0c0d5338453c35e2cbb363d963b57302ee`
- `models/experience_curves.joblib` — `605efef224a44b74702c797149f9b5ea8d4e8760700672143d42b650e3d46e39`
- `models/ensemble_weights.json` — `5735dd758ab4af7fb50d73cfccac629227f4d785c6264c3c5e52346ee42ce80c`
- `models/mean_model_selection.json` — `d8961f7156514a7b5b92cde943c181566796254e46cea03f96bf6e7f7abbe438`
- `models/marginal_selection.json` — `5516f941bdb9d8d723650f46d90afb6f1a7d39de777f0dd95bb7060a6bfe84e3`
- `models/marginals_pre2025.joblib` — `dcfe44c9ab0edd1b1fe86451934d8a1b65e9fcac2db877ae899f749ca1cb53ff`
- `models/marginals.joblib` — `e9d5ab0bcdb6635225a01526f3b682fecadb3d87d364bc93343f27f482cdc42f`
- `models/copula_pre2025.joblib` — `109d2cc9144b75957c562a471297a4526d473a07c776b46bf99f49e9153567a5`
- `models/copula.joblib` — `4149fa702a7f31883786c966b245c711527f5b147f438e7112990a87a914bd39`
- `models/combo_dependence_policy.json` — `37dba3b7ff1c27d8cced09c6d8655c9c8ee7afc295427163a08846b0ccddf735`
- `models/market_probability_calibration.json` — `330d8e30c6b641f6beb5b541a5df14b5a8be25d8ce48e5114a460bed8f793655`
- `models/market_probability_calibration_policy.json` — `484656bbc50ff111bb7990ccef75260a8f47b568c3f8a08816c82f65ab10f7a8`
- `models/pts.joblib` — `238bbf7651444ff319505db0152d30806596221c0400bdb6bcb6c27de3121c58`
- `models/reb.joblib` — `2506ea9369434eb85d344fc0c5f89304cbb9356ac47b1525d99e8812dcde9972`
- `models/ast.joblib` — `8a148381fe8779985e1b546d61d9d086f1a93b59a2c577276deef036a16d69ca`
- `models/stl.joblib` — `9b7e803c3de7bfac705d5d5ce72a791ebe892cd4e1ad10e3488850911d25c740`
- `models/blk.joblib` — `f9a9a3bbe7e3a7c9e9f3d93a772217d2b235953d3d498c2b42d48eaed84273d9`
- `models/fg3m.joblib` — `62f6880817e211a3f9909c77ee3d8b3b1d327f80b91b04b09eb3a83bdb267889`
- `models/minutes.joblib` — `d0b8741e05dab8e2b99c21cb054cf408eb5aa5cb9dc8ab4753a8bf55ac087b70`

### Source Files

- `src/nba_prop_quant/settings.py` — `642d03376ace773e44e4a531127a4d609b85eddbc0c408dd331e7b9c8cca5ded`
- `src/nba_prop_quant/game_context.py` — `8c2b015559b4ed3329ad7be5f02d4b8bca7a56c14b5e7eeca4e21f7840e6c22a`
- `src/nba_prop_quant/decay.py` — `bd31f506bc2dd545fb8296a3cbb32bd326ef46d09dccbbd3717071142063dec3`
- `src/nba_prop_quant/kalman.py` — `898213b89e7c96def9ec5e0fd151569619b675d22cdb3f478a091d9d5e124f78`
- `src/nba_prop_quant/experience.py` — `2605db999b087b804aea32bd60b33bdf08fe0d6301026f5acebba99a3e48e65a`
- `src/nba_prop_quant/features.py` — `956a95456dc5e344ef679d5e8bfacbabca9486caa745be2d1080d8517dcc4124`
- `src/nba_prop_quant/model.py` — `53f5c2257c96a6f079935e6ec881798a5337044b4977ee481d9c3ed3867b00d9`
- `src/nba_prop_quant/distributions.py` — `0fac50865b33dcbdb5ac2b4e63b1a66f179c2ab446f1575e62053e48b0402ba6`
- `src/nba_prop_quant/copula.py` — `7b51641a6404fe04047b5aa72fad82ff1df9a0c9073b12ab3cebeba9057ce76b`
- `src/nba_prop_quant/pricing.py` — `1a757e9ee4d955eea2d2e6150bafe647d5e3c4b6adafca0ea1be022c2589fb2a`
- `src/nba_prop_quant/market.py` — `6c6664cc2ede2566e07ff55a9bef10e5fd89a7f6a9c796b09fb210eaadd594c7`
- `src/nba_prop_quant/availability.py` — `2b6e342d0baeac5d1bd7a6bc097b5581ca4b90f5695d7effe613bb18080bd319`
- `src/nba_prop_quant/interpret.py` — `b7d443ae5001414c128e7c37d328965c5b44502d22cd2b294a201dff0d31056b`
- `src/nba_prop_quant/slate.py` — `c6c8880a37c933e0a15947c80e20750884204ae44aa33726c861a661aa608318`
- `src/nba_prop_quant/pipeline.py` — `c3cf7718daa892c5e094cd2e5983429b1954f8c143437d19cdafcb87cc30d552`
- `src/nba_prop_quant/live.py` — `db0830a7fbde2e7c8d879b4dc2e3c0dcac5cb68dca8e3822cff53797d258eb94`
- `src/nba_prop_quant/production.py` — `0f709dda1cae1d9d2ad7316372cdf6ab2600541fcb262de0440de8479e88ba93`
- `tests/test_production_live.py` — `666011074ec357dbae7b2cd873bd7ac7c802caf50fdfb2754848f631ad0f99b8`

### Scripts

- `scripts/05_train_minutes.py` — `c5261fa9e928edab665460e8813d93b82a9cb18232ba54199df5cfb53afe4847`
- `scripts/06_train_targets.py` — `e646b10936c2bf647e76ffa321a24b7966e11b778bac530c98fe12e5eb6ddbab`
- `scripts/06b_fit_mean_ensemble.py` — `80c99250fbaa07ada3e18a85cdb816ba88ab1236dc8636224cb5038b952d02b2`
- `scripts/06c_select_mean_models.py` — `d53c7a8e3f597fc73b4c945bfea5b469f8b42370e262c30eb81f916704dc5505`
- `scripts/07_fit_marginals.py` — `afe2616c3296c1f42a39e21203d8f6f6ecb5c2e8349daf8dc7bed0527a47bbed`
- `scripts/08_fit_copula.py` — `c98efe4f9b0a862dd1cd9ade846e379fa53728ea54a7d54c63c63598635ff7ba`
- `scripts/08b_bootstrap_copula_crps.py` — `b0e76ffe6c19740428419e4ebee17d21c7e276b76caf27ee036742bf47fc9db4`
- `scripts/08c_tune_copula_shrinkage.py` — `720586f2dcf29cb7e33baeb1d934985ba38f35cb7d2b7d1cadc3776c6deb9fdc`
- `scripts/08d_refine_copula_shrinkage_cv.py` — `d9d159f1ac436df3800027ae4ca867358daea972db8f36ded88192fac15dcf64`
- `scripts/09_backtest.py` — `c1b4dab650781cd98e3d7b0e44c1c01c63c3999f0918b9373e3bc7960c0b6a9b`
- `scripts/09c_fit_probability_calibration.py` — `cfa97ab8f1926cf0be0a9aad3f3a22484c837dc7ca1db4a85914991bfedeab6c`
- `scripts/09d_select_probability_calibration.py` — `efcf6cc9cbc815a1446c35ae219d00c7cc3172f1fef7f6af3f5c29b0a83f2e51`
- `scripts/09e_evaluate_oof_calibrated_betting.py` — `f98059195ea33b034039cbccf7d16c78d2416127dbac3e5dce09dac6061f7468`
- `scripts/10_predict_slate.py` — `1d63c46b0cfc08ae117ec5bdddc6ef68e6c5c1b4eabeafc13889934f8875ed73`
- `scripts/10a_validate_production_contract.py` — `7cb29035b590a7acfdca8bdd58bf5084a5aed0c37c070fc930c0ec7300353134`
- `scripts/15_price_markets.py` — `270e62c263a0d9ccc6e4f75762903d9fedecbb31c5f117ac99218c9af8d25778`
- `scripts/freeze_production_manifest.py` — `b74b2132bcd5efc288ee7b60d1d3e668f321a05c7bda50cb79c56b808493ad88`
- `scripts/verify_frozen_manifest.py` — `d684000e49db6afdb2f1236f990c8be545ec4b4ce7dbed46ad054da17f5f4cb1`

### Configs

- `configs/model.yaml` — `12afbb1df6a62f8121319e587f65ebb1403d7a59dd721eadefdc209d2efe9140`

### Data Audit Files

- `data/processed/oof_selected_means.parquet` — `c64af789d3d8ec8232ed006afb5d00bd6e7731503cb85adb675a35d1d6dfff03`
- `data/processed/selected_means_distribution_split.parquet` — `45b39ef4caaaa059680861bbe529b8efceed52349f07b3c51d153e6afb146916`
- `data/processed/market_backtest/calibration_walkforward/pooled_event_metrics.csv` — `7cc1d056c7f7d2c1bb59b7c3f4ca575ec01625f2f4abf811ce49b2cd1a198cd3`
- `data/processed/market_backtest/calibration_walkforward/calibration_selection_audit.csv` — `91f462ed813722fbf181e8a9e0eea2f433a365db5f3054daa4a6667b9b5be37d`
- `data/processed/market_backtest/calibrated_oof/event_scoring_summary.csv` — `ed57cd1d10ae7b892dd936ff4de1ed724673a969c1880c4e54856396917a11eb`
- `data/processed/market_backtest/calibrated_oof/event_game_cluster_bootstrap.csv` — `79f90df6acd118ed7ce8aa6378d4d591cbc83225926972f52c1869d749a9e993`
