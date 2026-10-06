# Game-level latent-state dependence shadow model, v1

**SHADOW / RESEARCH ONLY.** Nothing in this directory or in
`src/nba_prop_quant/research/` is imported by the production pipeline, the
Step 3C adaptive machinery or the Step 3D automation. See
[`SHADOW_V1_SAFETY.md`](SHADOW_V1_SAFETY.md) for the containment contract.

## What this adds

The incumbent model produces a calibrated marginal distribution per
player-stat, and a Gaussian copula that couples the **six stats of one
player**. It has no representation of dependence *between* players, so a
query like

```
P(A PTS > l1, A AST > l2, B REB > l3, C PTS < l4)
```

is currently only answerable by multiplying per-player probabilities, which
assumes A, B and C are independent inside the same game. They are not: pace,
shot volume, the rebound environment and usage competition are all shared.

This branch adds the missing layer and nothing else. The pipeline is

```
existing game state
  -> existing conditional means              (unchanged)
  -> existing calibrated marginals           (unchanged)
  -> NEW hierarchical latent game-state layer
  -> correlated residual latent normals -> uniforms
  -> existing marginal inverse CDFs          (unchanged)
  -> coherent whole-game simulations
  -> joint prop-event probabilities
```

Because the last step before the output is the **production inverse CDF**,
every univariate margin the simulator produces is the production margin by
construction. Only the coupling is new.

## Architecture map of the existing dependence contract

Inspected before anything was written, so the new layer could be placed
without double-counting what already exists.

| Concern | Where it lives | What it does |
| --- | --- | --- |
| Same-player dependence | `src/nba_prop_quant/copula.py` | `GaussianCopula` over `pts, reb, ast, stl, blk, fg3m`. `fit` uses **mid-PIT** `u = F(y-1) + 0.5·pmf(y)` clipped to `[1e-5, 1-1e-5]`, Ledoit-Wolf shrinkage to a `global_corr`, plus per-player blocks when a player has at least 40 games, shrunk by `n/(n+100)`. |
| Cross-player dependence | *nowhere* | `GaussianCopula.simulate` draws **one player's** six stats from `rng.multivariate_normal`. Players are independent. This is the gap. |
| Marginals | `src/nba_prop_quant/distributions.py` | `PoissonCalibrator`, `NegativeBinomialCalibrator`, `ZeroInflatedNegativeBinomialCalibrator`; `FittedMarginal` exposes `cdf / pmf / ppf / over_under_push`. Production freezes the family at `zinb`. |
| Marginal fitting entry point | `scripts/07_fit_marginals.py` | `inflation_features_for(target, frame)` and `fit_candidate(kind, y, mu, frame, inflation_features)`. |
| Conditional means | `src/nba_prop_quant/model.py` | `season_walk_forward_oof_minutes`, `season_walk_forward_oof_target`, `fit_minutes_model`, `fit_target_model`. |
| Mean routing / ensemble | `src/nba_prop_quant/adaptive_training.py`, `scripts/06b_fit_mean_ensemble.py` | `FROZEN_MEAN_ROUTES` selects `xgb` or a convex `ensemble`; `fit_simplex_weights` fits the simplex weights. |
| Features | `src/nba_prop_quant/features.py` | `TARGETS`, `build_base_frame`, `add_dynamic_priors`, `feature_columns`. Hashed by `feature_schema_hash`, so untouched. |
| Adaptive / promotion | `src/nba_prop_quant/adaptive_training.py`, `adaptive_fit_registry.py` | `ProductionFitEngine`, `promotion_state.json`, `current_good_fit_id`. **Not referenced by this branch.** |

### Supported stat targets

`pts, reb, ast, stl, blk, fg3m` — exactly what the production marginals and
the incumbent copula cover. **TOV is deliberately absent**: production fits no
turnover marginal, and the brief forbids inventing support the current model
does not provide. `simulate_game` raises if asked for it.

The same constraint explains the one requested dependence bucket that cannot
be simulated. "Opponent missed-shot environment versus rebounds" needs an FGA
or FGM marginal, which production does not have. It is reported through the
simulatable proxies `opponent_pts_reb` and `opponent_fg3m_reb` instead, and
the FGA-based version is an observation-only diagnostic.

## How double-counting is avoided

Integration **strategy B** from the brief: the existing same-player block is
preserved exactly and the new factors are added *around* it. The game
covariance is

```
Sigma = (C L)(C L)^T  +  blockdiag_i( R_i - c_i^2 L_i L_i^T )
```

where `R_i` is the incumbent correlation for player `i` and `L_i` is the
shared-factor loading matrix. The block-diagonal term subtracts exactly the
shared-factor contribution to player `i`'s own block, so

* the within-player block of `Sigma` is `R_i` to float64 precision, and
* `Sigma` is a Gram plus a PSD block-diagonal, hence **PSD by construction** —
  not by a post-hoc eigenvalue repair.

`c_i` is the largest scalar in `(0, 1]` keeping the residual block PSD, found
by bisection on `c^2` (the minimum eigenvalue is concave and non-increasing in
it, so the bisection is exact up to tolerance).

Gate E is therefore provable rather than merely measured, and
`test_same_player_dependence_is_not_applied_twice` asserts it holds for shared
loadings scaled by 0, 0.5, 1 and 2.

Estimation respects the same split from the other side: the loadings are fitted
from **cross-player pairs only**, so the incumbent block is never re-estimated
(`test_pair_moments_never_touch_same_player_pairs`).

## Latent factor families

```
Z(i,s) = sum_k gamma[s,k] G_k            game-level (pace/volume, rebound environment)
       + side_i * d[s] * G_contrast      own-team minus opponent-team state
       + zero-sum competition term       within-team, see below
       + within-player residual          pinned to the incumbent copula block
```

Identification is the reason the structure looks like this rather than like
the brief's conceptual sketch of separate own-team and opponent-team factors.
Writing `S` for the same-team cross-player correlation and `X` for the
cross-team one, the model gives `S = A + B - Q` and `X = A - B`. A team-state
factor that *both* teams load on identically is indistinguishable from a
game-level factor, so only the **antisymmetric** part of the own/opponent
loadings is separately identified; the symmetric part is absorbed into the
game-level block. Estimating separate own-team and opponent-team loadings
would be fitting an unidentified parameterisation.

### The within-team competition family

A purely additive factor model can only produce **non-negative** same-team
same-stat correlation. That is the wrong sign when teammates compete for a
finite resource, and the data say they do: the fitted same-team `pts`-`fg3m`
entry is negative at z = -5.2. The competition family is

```
kron(I - 11'/n, n * Q)
```

a Kronecker product of two PSD matrices, giving `-Q` between distinct
teammates and `(n-1)·Q` on a player's own block. The `n` scaling makes the
pairwise effect roster-size free.

It is gated. `Q` is identified only through the PSD constraint, so an
indefinite `S` is the only evidence it is non-zero — and `min eig` is
**concave**, so by Jensen's inequality the point estimate of `min eig(S_hat)`
is biased *downward* and reads as indefinite even when the truth is PSD. A
bootstrap percentile inherits that bias. `factors.competition_gate` therefore
uses the basic (pivotal) bootstrap bound `2·theta_hat - q_{1-level}(theta*)`,
which cancels it, and admits the family only when that bound is still
negative. On purely additive synthetic residuals the percentile bound reads
-0.0027 and would have activated the family on noise; the pivotal bound reads
+0.0005 and correctly declines.

### Shrinkage and transfer to unseen players

Every parameter is indexed by **stat**, optionally modulated by a coarse
minutes-role bucket with partial pooling (`weight = n/(n + 400)` toward the
pooled value of 1.0). There are **zero** pair-specific or player-specific
dependence parameters, so the layer transfers to unseen players and new roster
combinations without refitting, and a role the fit never saw falls back to the
pooled estimate. Entries of `S` and `X` are soft-thresholded at 1.96
game-clustered standard errors before projection, so stat pairs the data
cannot resolve contribute exactly zero rather than noise.

## Pipeline

Run in order. All of it writes only under `data/research/` and
`research/game_latent_state/`.

```bash
# 1. research-scoped raw history (needs BDL_API_KEY); never touches data/raw
python research/game_latent_state/01_ingest_research_history.py

# 2. OOF randomized-PIT Gaussian residuals
python research/game_latent_state/02_build_oof_residuals.py

# 3. shared latent factor loadings, training seasons only
python research/game_latent_state/03_fit_latent_factors.py

# 4. held-out validation and the acceptance gates
python research/game_latent_state/04_validate_shadow_v1.py
```

### Why randomized PIT and not mid-PIT

For a discrete count, `u = F(y-1) + v·[F(y) - F(y-1)]` with `v ~ U(0,1)` is
exactly uniform; the incumbent's mid-PIT `u = F(y-1) + 0.5·pmf(y)` is
**under-dispersed** and attenuates estimated correlations toward zero. Since
the whole point here is to estimate dependence, the residual builder uses
randomized PIT, with `v` a keyed BLAKE2b digest of `(seed, game_id,
player_id, stat)` so it is deterministic and order-independent. Both are
stored (`z_*` and `zmid_*`) so the attenuation is measurable rather than
asserted.

### Leakage invariants

Every predictive quantity attached to a row in season `S` is fitted only on
seasons strictly before `S`: the expected minutes, the conditional means, the
ensemble weights, the ZINB marginals, the incumbent copula, the factor
loadings and the standardization constants. Production fits its marginals once
over all eligible history; refitting them walk-forward here is strictly more
conservative. Joint-event lines come from the predictive marginal only —
realized box scores are used solely for grading.

## Season window

The walk-forward requirements cascade, which is why the residual window is
narrower than the ingested history:

| Stage | First usable season | Why |
| --- | --- | --- |
| Ingested raw history | 2015 | `ADVANCED_START_SEASON` |
| Walk-forward expected minutes | 2016 | needs one earlier season |
| Walk-forward conditional means | 2017 | needs one earlier season with minutes |
| Walk-forward ensemble weights | 2018 | needs one earlier season with means |
| Walk-forward ZINB marginals | **2020** | needs two earlier OOF seasons |

So residuals cover **2020-2025**, split chronologically into training
2020-2023 and held-out validation 2024-2025. Two caveats worth stating:
2020 is the COVID-shortened season and is unusual in pace and schedule
density, and the raw window is narrower than production's
`history_start_season`, so these loadings are not a drop-in replacement for a
production fit over full history.

## Artifacts

Written to `research/game_latent_state/`, every one carrying the source
production SHA, the training cutoff, the seasons used, input data
fingerprints, the seed, the code SHA and SHA256 hashes.

| File | Contents |
| --- | --- |
| `oof_gaussian_residuals.parquet` | OOF randomized-PIT residual dataset |
| `oof_residual_build_report.json` | per-season marginal fits, ensemble weights, PIT diagnostics |
| `factor_spec.json` | factor families, loadings, identification note, `spec_hash` |
| `factor_loadings.parquet` | per-stat loadings, tabular |
| `covariance_diagnostics.json` | observed/shrunk `S` and `X`, clustered SEs, eigenvalue spectra, competition-gate evidence |
| `validation_report.json` | all held-out evidence plus the gate verdicts |
| `joint_event_grades.parquet` | every graded conjunction with each model's probability |
| `manifest.json`, `residual_manifest.json`, `manifest.validation.json` | provenance |
| `SHA256SUMS.*.txt` | hashes |

Every manifest declares
`promotion_eligibility = "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION"`.

## Acceptance gates

`gates.py` declares thresholds A-H **in code, before** the validation run
produces the numbers they judge. `evaluate_gates` reads the report and returns
verdicts; `assert_promotable` **always raises**, including for a candidate that
passes every gate, because a passing gate set authorises
production-integration *design* only.

One gate deserves explanation. Gate A cannot use a fixed z ceiling: the
validation run probes every player-stat of every probed game, so at ~10^4-10^5
simultaneous probes the null expectation of `max |z|` is already above 4, and a
flat 5-sigma bound would be close to a coin flip on noise alone. Gate A instead
derives its critical value from the probe count the report declares, via a
Bonferroni bound at a fixed family-wise alpha, and pairs it with two checks a
single-worst-case bound cannot make: the fraction of probes beyond 3 sigma
(which catches systematic bias that no individual probe reveals) and an
absolute over-probability tolerance (so a huge probe count can never license an
economically meaningful miss). The simulation budget is then chosen so Monte
Carlo noise sits well inside that absolute tolerance, rather than the tolerance
being widened to accommodate the noise.
