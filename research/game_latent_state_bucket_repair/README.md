# Latent-state bucket repair (shadow research)

SHADOW / RESEARCH ONLY. Nothing here is wired into production, and nothing here
can promote. The accepted shadow V1 at `1c5b8c93569ee25afd4eb4222158300702bd9471`
is the immutable control: every control number in this directory is copied from
its validation report and is never recomputed.

## What was wrong

Accepted V1 under-captured two same-team buckets on the untouched 2024-2025
holdout:

| bucket | observed | V1 implied | V1 z |
| --- | --- | --- | --- |
| `teammate_reb_reb` | +0.009417 | +0.001290 | -3.74 |
| `teammate_ast_ast` | +0.008971 | +0.000937 | -3.59 |

The V1 report attributed this to soft-thresholding. Measured on pre-2024 data
only, it is **two independent mechanisms**:

1. **Fixed-width thresholding.** A 1.96-SE soft threshold removes exactly
   `z_crit / |z_observed|` of the estimate. That is 52.9% for `teammate_reb_reb`
   at z=3.70 and 56.4% for `teammate_ast_ast` at z=3.47, against 6.3% for
   `passer_ast_teammate_pts` at z=31.15. The two targets are not mis-estimated;
   they are the weakest *significant* same-team buckets, which a fixed-width
   rule taxes hardest.
2. **Rank truncation.** A further 24.3% and 25.1%. The two targets lose through
   *different* channels: `teammate_reb_reb` through the rank-1 team-contrast
   factor, `teammate_ast_ast` through `k_game=2`.

The rank loss is parsimony, not statistics. At `k_game=6, r_contrast=6` the
construction reproduces the shrunk blocks with max deviation `6.3e-17`, and
nothing in the identification pins the contrast Gram `B` to rank 1.

## What was changed

`src/nba_prop_quant/research/game_latent_state/repair.py` is **additive**.
`V1_CONTROL_SPEC` reproduces `factors.fit_shared_factors` bit for bit and
matches the committed V1 `factor_spec.json`, so V1 stays the control rather
than a re-derivation. `covariance.py` gained a multi-rank team contrast; the
1-D wire format remains canonical at `r_contrast == 1`, which leaves the
committed V1 payload byte-identical.

The frozen candidate is `C_rank_k6_r6_eb_block_diagonal`:

- `k_game = 6`, `r_contrast = 6` (full rank, so no representation loss),
- empirical-Bayes shrinkage pooled within the block-diagonal family, replacing
  the fixed-width soft threshold. For `r ~ N(theta, se^2)` and
  `theta ~ N(0, tau_f^2)` within family `f`, the posterior mean is
  `r * tau_f^2 / (tau_f^2 + se^2)` with `tau_f^2` by method of moments on the
  upper triangle. The bias is proportional to the estimate rather than a fixed
  width, which is why it stops over-taxing the weak-but-real buckets,
- `pairwise_parameter_count = 0` and `player_indexed = 0`, unchanged.

## Why family D was not used

Role heterogeneity on the targets is real and very large: bench
`teammate_reb_reb` is +0.09824 against +0.00033 for rotation and +0.00581 for
starters, and bench `teammate_ast_ast` is +0.05527 against -0.01794 for
starters. But V1 renormalises role scales so `sum_r p_r s_r == 1` exactly, so
the pooled multiplier is `(sum_r p_r s_r)^2 == 1` and
`same_team_correlation()` is bit-identical with and without the role layer.
Family D is therefore provably inert on the metric the repair is judged on. It
is excluded from selection, and every candidate pins `role_column` to V1's
setting so the existing role layer is never silently dropped. Role-conditioned
*reporting* is a genuine finding and is out of scope here.

## No holdout tuning

Selection ran on a strictly chronological nested walk-forward inside pre-2024
history: fit[2020]->score2021, fit[2020,2021]->score2022,
fit[2020,2021,2022]->score2023. `01_diagnose_buckets.py` raises if a holdout row
reaches it at all. Eleven pre-registered candidates were scored; the winner was
chosen by a mechanical rule and frozen into `bucket_repair_candidate.json`
before the holdout was run once.

One correction was made to the inner rule *before* the holdout was touched: the
overshoot constraint originally aggregated by the per-fold maximum, which is the
wrong aggregation level because REPAIR GATE 2 is a single pooled number. The
maximum measures between-season dispersion in the truth (`teammate_ast_ast` is
+0.0092 / +0.0070 / -0.0019 across the three inner seasons, so no single pooled
parameter can sit inside every season's interval) rather than model bias, and it
ranked the more biased model higher. It was changed to the fold mean, with the
maximum retained as a dispersion diagnostic. See
`selection_rule.overshoot_aggregation_correction` in `inner_validation.json`.

## Scripts

| script | purpose |
| --- | --- |
| `01_diagnose_buckets.py` | pre-2024 diagnosis: SEs, signal/SE, attenuation decomposition, season and role stability, bootstrap CIs, rank sweep |
| `02_inner_validation.py` | nested chronological inner selection over the pre-registered grid |
| `03_fit_repair_candidate.py` | refits and freezes exactly one candidate into a spec the **unmodified** V1 driver consumes |
| `04_compare_joint_calibration.py` | paired game-clustered bootstrap of repair-vs-V1 Brier |
| `05_evaluate_repair_gates.py` | the twelve repair gates against the immutable control |

The expensive holdout validation is `research/game_latent_state/04_validate_shadow_v1.py`
pointed at this artifact root. It was run exactly once.

## Why the Brier comparison is paired

Differencing the two validation reports resolves a Brier difference only to
within two independent Monte Carlo errors, which at 4 legs is an order of
magnitude wider than the 0.0005 tolerance REPAIR GATE 10 uses. The two runs
share games, seed, simulation count, residual dataset and an independence
baseline that does not depend on the fitted loadings, so the graded
conjunctions line up row for row and a paired bootstrap with common random
numbers differences out the shared noise.
`04_compare_joint_calibration.py` asserts the pairing rather than assuming it:
realized outcomes and both baseline probability columns must agree exactly
before any difference is reported.
