# Shadow V2 structural refinement — retained evidence

**SHADOW / RESEARCH ONLY. NOT PROMOTABLE.**

Every structural addition the V2 round proposed was screened and **not
carried**. Nothing in this directory describes a component that ships. The
accepted production model does not use any of the rejected V2 modules, and no
dial they expose is set anywhere on this branch. See
`../game_latent_state_resolution/` for what does ship.

This directory is self-contained. The decision to reuse the accepted repair's
holdout grades instead of re-running the Monte Carlo rests on the equivalence
proof, and every input that proof names can now be checked here without
fetching any other branch.

## What is here

| file | what it settles |
|---|---|
| `equivalence_and_calibration.json` | the frozen candidate's point-probability model is bitwise the accepted repair's, in both refit windows — and the reused 2/3/4-leg log losses computed from the committed grades |
| `sd_calibration_blocker.json` | why the predictive-SD inflation factor was rejected, attributed to double-counted 2023 heterogeneity |
| `shadow_v2_candidate.json` | the frozen candidate the equivalence proof applies to |
| `temporal_diagnostic.json` | why `T0 pooled empirical Bayes` was selected over the random-effects, recency and random-walk treatments |
| `axis_indifference_resolution.json` | why `r_symmetric = 0` |
| `inner_screening.json` | why `role_deviation = false` and `bridge_weight = 0` |
| `factor_spec.json` | the V2 candidate's full loadings payload — the `v2_factor_spec.json` input fingerprint named by `manifest.equivalence.json` |
| `manifest.json` | the factor-model run record: seed, `code_sha`, seasons, the complete `v2_spec` dial values, parameter counts, parent SHAs and output hashes |
| `covariance_diagnostics.json` | the covariance structure the V2 gates were evaluated against, named in `manifest.json` outputs and in `SHA256SUMS.factors.txt` |
| `SHA256SUMS.factors.txt` | digests for the factor-model run's five outputs, so the compact record verifies locally; the one line naming `factor_loadings.parquet` identifies an export this branch refuses to carry rather than covering a file here |
| `07b_diagnose_sd_blocker.py` | regenerates `sd_calibration_blocker.json` from `equivalence_and_calibration.json` and `temporal_diagnostic.json`; imports only `nba_prop_quant.research.game_latent_state.artifacts` |

The last five entries are the **authoritative compact reproduction record** for
this round. They were carried here by an explicit path allowlist applied onto a
branch cut from the production head, never by a merge from the research branch.

## Retired research lineage

`research/nba-game-latent-state-shadow-v2-structural` (PR #18) and its base
branch `research/nba-game-latent-state-shadow-v1-bucket-repair` (PR #17) are
**retired**. Both pull requests were closed without merging and both branch
refs were deleted. Nothing in this repository is reachable from them any more,
and nothing here depends on them.

**Full executable reproduction of the rejected V2 search paths was
intentionally retired.** Seven of the ten V2 drivers import a rejected
search-space module, including `07_prove_equivalence_and_calibrate_sd.py`,
which produced `equivalence_and_calibration.json`. Carrying them would have
required making those modules importable, which this branch forbids. The
ability to re-run the V2 structural search is therefore gone on purpose: the
modules were rejected precisely so their dials cannot be set again. What
remains is the record of what was tried, what it produced and why each dial is
off — plus `07b_diagnose_sd_blocker.py`, the one driver that re-derives a
retained artifact without touching a rejected module.

Three `code_sha` values appear in records here and in `inner_screening.json`:

| value | record |
|---|---|
| `d51eae726cc99da04f4597e29f8ae83eee8beed6` | `manifest.json` |
| `470318f4f7305a0e5849ef9dba2eefb78a7a6f8e` | `manifest.equivalence.json` |
| `0468bdb1446164f8a5303d79399d6779bc40cdca` | `inner_screening.json` |

These are **historical identifiers for retired lineage and may no longer
resolve in the live repository.** They were only ever reachable from the
retired branch. They are retained verbatim because a provenance record should
say which commit produced it even once that commit is no longer hosted; the
artifact digests in the `SHA256SUMS` files, not the commit ids, are what makes
these artifacts checkable.

## What is deliberately absent

The four rejected search-space modules — `v2.py`, `temporal.py`, `bridge.py`,
`countspace.py` — and the seven drivers that import them. Also the two clean
drivers whose outputs this directory does not retain (`00_freeze_control.py`,
`06_evaluate_v2_gates.py`), the declined count-bridge artifacts
(`count_bridge.json`, `repair_control_baseline.json`), the duplicated screening
passes, the bridge-fitting cache, the regenerated parquet exports and
`tests/test_game_latent_state_shadow_v2.py`.

They are not missing by accident and the artifacts above do not need them. The
rejected components must not be importable from a production branch —
`tests/test_game_latent_state_shadow_clean_candidate.py` asserts that they are
not, including that `pooled_uncertainty_inflation` has no runtime surface here.

The earlier screening passes are superseded rather than lost:
`inner_screening.json` **is** `inner_screening_pass3.json` byte for byte, so
the pass-3 digest recorded in `axis_indifference_resolution.json` and
`shadow_v2_candidate.json` verifies directly against it, and passes 1 and 2 are
named in its `earlier_pass_artifacts` field.

## Hash-identified rather than carried

`.gitignore` here refuses the residual parquet, the regenerated exports, the
fitted-marginal caches and the declined count-bridge artifacts. Fingerprints
for the excluded files live in the `SHA256SUMS` files of the directories that
produced them — including the 68,627,706 B `oof_gaussian_residuals.parquet`,
whose digest is in `../game_latent_state/SHA256SUMS.residuals.txt` and which is
unreachable from every ref in this repository.
