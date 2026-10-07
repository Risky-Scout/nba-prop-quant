# Latent-state structural research: final resolution

**SHADOW / RESEARCH ONLY. NOT PROMOTABLE.**

Three rounds of research ran on the game-level latent dependence model. This
directory is the single place that says what survived.

| round | branch | PR | outcome |
|---|---|---|---|
| Shadow V1 | `research/nba-game-latent-state-shadow-v1` | #16 | accepted for production-integration design |
| Bucket repair | `research/nba-game-latent-state-shadow-v1-bucket-repair` | #17 | **accepted — this is the model that ships** |
| V2 structural refinement | `research/nba-game-latent-state-shadow-v2-structural` | #18 | every structural addition screened and not carried |

## What ships

The accepted bucket repair, unchanged.

```
k_game                 = 6
r_contrast             = 6
r_symmetric            = 0
role_deviation         = false
bridge_weight          = 0
temporal_treatment     = T0 pooled empirical Bayes
predictive_sd_inflation= none
pairwise_parameter_count = 0
player_indexed           = 0
```

## What was rejected, and on what evidence

| component | why not carried | evidence |
|---|---|---|
| `r_symmetric > 0` | At full rank the three-family construction is already exact, so every admissible symmetric rank reproduced the axis's null value on every judged metric — worst relative deviation 6.5e-15 against a 1e-9 tolerance. Those points are the same model written differently. | `../game_latent_state_v2/axis_indifference_resolution.json` |
| `role_deviation = true` | Cut support-weighted role-conditioned RMSE by 44.1% but left two role cells worse, by up to 23.1%. It bought an aggregate gain by moving error into specific cells. | `../game_latent_state_v2/inner_screening.json` |
| `bridge_weight > 0` | Cut primary count-space error by about 45% and pushed global latent RMSE to 0.004636, outside the pre-registered 0.003750 bound. Every positive-weight grid point sat outside the bound. | `../game_latent_state_v2/inner_screening.json` |
| dynamic temporal treatment | T0 — what the repair already does — had the lowest predictive MSE inside the parsimony tie band. The random-walk state could not even be supported: four training seasons is below the six a one-state walk needs. | `../game_latent_state_v2/temporal_diagnostic.json` |
| predictive-SD inflation `1.7659` | Failed its pre-registered out-of-sample criterion on untouched 2024–2025. See below. | `../game_latent_state_v2/sd_calibration_blocker.json` |

### The predictive-SD rejection

Raw mean z² was **0.6652**; inflating it gave **0.3908**. The target is
approximately 1, so inflation moved the reported uncertainty *away* from
calibration rather than toward it.

The cause is double counting. The factor is `sqrt` of a mean over two inner
folds that disagree by a factor of 8.9 (fold 2022 scored 0.633, fold 2023
scored 5.604), so it is effectively the 2023 fold's number alone — and the
quiet fold's 0.633 is what the holdout reproduced. Season 2023 dipped by 2.4 to
4.5 standard errors in three of four buckets and reverted over 2024–2025. Once
2023 is inside the training set, the random-effects τ² absorbs that dip and
widens the predictive SD by up to ×3.41 on its own, in exactly those three
buckets and not in the fourth. Inflating on top of that applies the same
correction twice.

No replacement factor was tuned against 2024–2025, and none should be: that
would be fitting a reported uncertainty to the data it is meant to be honest
about.

## Why no further holdout simulation is required

The frozen V2 candidate's point-probability model was proven **bitwise
identical** to the accepted repair's, in both the 2024 and 2025 refit windows.

`simulate_game` output is a pure function of
`(roster, marginals, loadings, within_player, simulations, seed)`. Five of
those six are model-independent — in particular `fit_season` takes
`(history, season, stats)` and no factor-model argument, so whatever a refit
window produces is the same object under both models. Only `loadings` carries
the dependence model, and it is bitwise equal.

Checked rather than argued: all three signatures inspected, every loading
array and derived block equal at `0.000e+00`, and the full game covariance and
its Cholesky equal at `0.000e+00` across all 600 graded rosters — 300 per refit
window, each window checked separately so a window-specific failure could not
hide inside a pooled maximum.

So the repair's committed grades *are* the candidate's grades. Re-running the
Monte Carlo would recompute identical numbers.

## Reused holdout metrics

Untouched 2024–2025, 600 graded games, read from the accepted repair's
committed validation run. Nothing was re-simulated.

| legs | base rate | repair Brier | production Brier | repair log loss | production log loss |
|---|---|---|---|---|---|
| 2 | 0.2314 | 0.165442 | 0.165411 | 0.504275 | 0.504199 |
| 3 | 0.1167 | 0.097409 | 0.097368 | 0.333822 | 0.333807 |
| 4 | 0.0608 | 0.055455 | 0.055423 | 0.213939 | 0.213865 |

Also retained in `final_resolution.json`: latent and count-space cross-player
RMSE, marginal preservation for all three models, the same-player copula
contract (`max_block_deviation` 2.2e-16 over 600 games), PSD and numerical
stability (minimum covariance eigenvalue 0.1025, zero numerical failures), and
all 12 repair acceptance gates passing.

## A note on the parameter accounting

Two role layers exist in the codebase and only one ships. They are reported
separately, because collapsing them would produce a number that is false.

- **`role_deviation`** — the *additive* layer the V2 search proposed. Screened,
  not carried, contributes **0** parameters.
- **`role_scale`** — the *multiplicative* layer the accepted repair fitted.
  **3 fitted values under 1 normalisation constraint, so 2 free, and it is
  active.** The normalisation makes the player-weighted mean scale 1, which
  leaves the pooled same-team block exactly where the base loadings put it —
  so no pooled-level check can see this layer. It is not inert: an individual
  pair's implied correlation is scaled by `s_a · s_b`, which ranges from 0.695
  to 1.540 across the fitted roles.

A manifest reporting "active role-indexed parameters: 0" would therefore be
wrong. `role_scale` was accepted in PR #17 as part of the winning model and is
carried unchanged; nothing here refits it.

## Why this branch was rebuilt rather than merged

All three research branches reach a 65.448 MB residual parquet object
(`4b64c56e6c778c279fde5467c5b72d762a6dfe1c`) that was committed in `43e4d2a`
and removed in `4e15feb`. Deleting a file does not remove the object: it is
absent from every later tree and still reachable from all three branch heads,
so an ordinary merge commit would carry it into production history
permanently. The mitigation relied on until now was remembering to
squash-merge.

This branch was built fresh from the production base and copies only the files
it should carry, so its merge path contains no such object.
`audit_merge_path_blobs.py` checks that rather than trusting it, and
`tests/test_game_latent_state_shadow_artifact_budget.py` runs the same check
on every pull request.

## Contents

| file | what it is |
|---|---|
| `00_write_final_resolution.py` | derives `final_resolution.json` from the committed artifacts; fits nothing, simulates nothing, reads no residual data |
| `final_resolution.json` | the machine-readable resolution: winning spec, rejected components with evidence, parameter accounting, reused metrics, equivalence proof, merge-path audit |
| `audit_merge_path_blobs.py` | operator-facing merge-path blob audit |
| `SHA256SUMS.resolution.txt` | fingerprints for the above |
