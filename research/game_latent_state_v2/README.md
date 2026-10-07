# Shadow V2 structural refinement — retained evidence

**SHADOW / RESEARCH ONLY. NOT PROMOTABLE.**

Every structural addition the V2 round proposed was screened and **not
carried**. Nothing in this directory describes a component that ships. See
`../game_latent_state_resolution/` for what does.

This directory is kept on the clean branch for one reason: the decision to
reuse the accepted repair's holdout grades instead of re-running the Monte
Carlo rests on the equivalence proof, and a reader should be able to check that
proof without fetching another branch.

## What is here

| file | what it settles |
|---|---|
| `equivalence_and_calibration.json` | the frozen candidate's point-probability model is bitwise the accepted repair's, in both refit windows — and the reused 2/3/4-leg log losses computed from the committed grades |
| `sd_calibration_blocker.json` | why the predictive-SD inflation factor was rejected, attributed to double-counted 2023 heterogeneity |
| `shadow_v2_candidate.json` | the frozen candidate the equivalence proof applies to |
| `temporal_diagnostic.json` | why `T0 pooled empirical Bayes` was selected over the random-effects, recency and random-walk treatments |
| `axis_indifference_resolution.json` | why `r_symmetric = 0` |
| `inner_screening.json` | why `role_deviation = false` and `bridge_weight = 0` |

## What is deliberately absent

The drivers (`00_freeze_control.py` through `07b_diagnose_sd_blocker.py`), the
search-space modules they import (`v2.py`, `temporal.py`, `bridge.py`,
`countspace.py`), the duplicated screening passes, the bridge-fitting cache and
the regenerated parquet exports.

They are not missing by accident and the artifacts above do not need them. The
rejected components should not be importable from a branch headed for
production shadow integration — `tests/test_game_latent_state_shadow_clean_candidate.py`
asserts that they are not, including that `pooled_uncertainty_inflation` has no
runtime surface here.

The full reproduction path remains on
`research/nba-game-latent-state-shadow-v2-structural` (PR #18), which stays
open and **must not be merged directly**: it reaches a 65 MB residual parquet
object that an ordinary merge commit would carry into production history. See
`../game_latent_state_resolution/README.md`.

## Hash-identified rather than carried

`.gitignore` here refuses the residual parquet, the regenerated exports and the
fitted-marginal caches. Fingerprints for the excluded files live in the
`SHA256SUMS` files of the directories that produced them.
