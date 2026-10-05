# Shadow v1 containment contract

This branch is **shadow / research only**. It has no production promotion
path, no automatic registry promotion, no WizardOfOdds publishing changes, no
scheduled production mutation, and it does not replace the incumbent model.

Step 3C and Step 3D are complete and closed. This branch does not modify Step
3D automation; it found no incompatibility that would justify doing so.

## What this branch may not touch, and the test that proves it

Every contract below is enforced by a test in
`tests/test_game_latent_state_shadow_safety.py`, which fails the build rather
than relying on review to catch a violation.

| Contract | Test |
| --- | --- |
| No protected production path is modified | `test_no_protected_production_path_is_modified` |
| Production workflow files are byte-identical to the production ref | `test_production_automation_workflows_are_unchanged` |
| No `promotion_state.json` / `current_good_fit_id` change | `test_promotion_state_files_are_not_introduced_or_changed` |
| Changes confined to the research and test namespaces | `test_branch_changes_are_confined_to_research_namespaces` |
| No promotion / publish / deploy / register-fit entry point exists | `test_shadow_package_exposes_no_promotion_entry_point` |
| Promotion machinery is never imported | `test_shadow_package_does_not_import_promotion_machinery` |
| No WizardOfOdds publishing surface | `test_shadow_package_has_no_wizardofodds_publishing_surface` |
| No production write target referenced | `test_shadow_package_references_no_production_write_target` |
| Only `models/dynamic_params.json` is read from production artifacts | `test_only_read_only_production_artifacts_are_referenced` |
| Gate H evidence uses the same declaration these tests enforce | `test_gate_h_evidence_comes_from_the_shared_protected_path_declaration` |
| A rejected candidate cannot promote | `test_a_rejected_candidate_cannot_promote` |
| A **passing** candidate cannot promote either | `test_even_a_passing_candidate_cannot_promote` |
| Every artifact manifest declares itself non-promotable | `test_every_shadow_manifest_declares_itself_non_promotable` |

The protected-path lists are declared once, in
`src/nba_prop_quant/research/game_latent_state/safety.py`, and imported by both
the validation report and the safety tests. The report therefore cannot claim
a clean production surface using a narrower definition of "production" than
the tests enforce.

## Protected paths

Prefixes: `.github/workflows/`, `configs/`, `models/`, `ops/`, `scripts/`,
`docs/`, `release/`, `review/`.

Production modules read but never edited: `adaptive_fit_registry.py`,
`adaptive_training.py`, `copula.py`, `distributions.py`, `features.py`,
`gate3_v2.py`, `model.py`, `pipeline.py`, `pricing.py`, `production.py`,
`slate.py`.

`copula.py` is the load-bearing one: the entire no-double-count argument rests
on the incumbent same-player block being untouched and merely *read*.
`factors.incumbent_within_player_blocks` re-indexes the incumbent correlation
into the shadow layer's stat order and never mutates the copula object
(`test_reading_incumbent_blocks_does_not_mutate_the_copula`).

## What this branch writes

* `src/nba_prop_quant/research/**` — new package, imported by nothing in
  production.
* `research/game_latent_state/**` — scripts, docs and versioned artifacts.
* `tests/test_game_latent_state_shadow*.py` — new tests.
* `data/research/game_latent_state/**` — research-scoped data root, gitignored.
  The ingest writes here and never to `data/raw`.

## There is no promotion function

`gates.py` deliberately contains no promotion path. `assert_promotable`
**always raises** `ShadowPromotionRefused`:

* if any gate failed, because a rejected candidate cannot be promoted;
* if every gate passed, because a passing gate set authorises
  production-integration *design* only. Promotion remains owned by the Step 3D
  lifecycle.

Gate thresholds are declared as constants in `GateThresholds` before the
validation run produces the numbers they judge, and no gate is relaxed to
obtain a pass. The only data-dependent quantity in any threshold is gate A's
Bonferroni critical value, which is derived by arithmetic from the probe count
the report declares at a fixed family-wise alpha
(`test_gate_a_critical_value_scales_with_the_declared_probe_count`), and a
report that omits its probe count fails the gate rather than passing it
(`test_gate_a_rejects_a_report_that_hides_its_probe_count`).

## Hermeticity

Every test in both shadow test files is hermetic: no network, no API key, no
ingested data, no model binaries. The statistical tests build synthetic
residuals from a known factor structure and check recovery; the contract tests
read the repository and git metadata only. The tests that need the production
ref skip cleanly when it is unavailable rather than failing.
