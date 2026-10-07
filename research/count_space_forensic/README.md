# Count-space blocker forensic study (shadow research)

SHADOW / RESEARCH ONLY. Nothing here is wired into production, nothing here can
promote, and nothing here modifies the frozen remediation candidate. The study
reads `research/final_upstream_remediation/factor_spec.json` (spec hash
`3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4`) and the
accepted bucket repair that is its paired control (spec hash
`c9b46e3a7497cfee52832f29397397b843aafb8e5f770a39157ce508f8157bc0`), and writes
only `count_space_forensic.json` and this file.

## The question

The frozen candidate takes `passer_ast_teammate_pts`'s held-out count-space
error down 5.2% against the accepted repair. The research target was 20%. Is
that shortfall an **estimator** defect or a **structural** limit of the frozen
PSD factor architecture?

Every estimator, inversion and target in here reads seasons 2020-2023 only.
Seasons 2024 and 2025 are used where section 3 asks for them -- to *evaluate*
the envelope the constraints define -- and never to choose anything.

## Final conclusion

```
COUNT-SPACE BLOCKER IS AN ESTIMATOR ISSUE BUT FIX VIOLATES GLOBAL CONSTRAINTS
```

Section 4 classification: `IDENTIFIABLE_BUT_GLOBALLY_INCOMPATIBLE`.
Section 6 decision: `STOP`. No new factor family was fitted.

## The five answers

Reported twice: under the gates `05_evaluate_gates.py` actually implements, and
under the original remediation requirements as briefed. The second is strictly
tighter in both ways it differs -- 1.03 instead of 1.05 RMSE tolerances, and
"no worse than control" on the two repaired buckets enforced in count space as
well as latent space.

| | under the implemented pipeline gates | under the original remediation requirements |
| --- | --- | --- |
| **A** max feasible AST->teammate-PTS count correlation | **+0.04200935** (from latent parameter 0.05299496) | **+0.03952614** (from latent parameter 0.04800719) |
| **B** absolute-error reduction | **25.50%** | **12.15%** |
| **C** `teammate_reb_reb` degradation there | **+15.12%** (0.00130239 against the control's 0.00113134, absolute +0.00017105) | **0.00%** (0.00113134 against 0.00113134, absolute +1.0e-12) |
| **D** first binding constraint | **`latent_rmse_within_tolerance`** | **`teammate_reb_reb_no_worse`** |
| **E** original 20% gate achievable | **yes**, but only above the generic estimator's own value -- see below | **no** |

The entry that would deliver exactly 20% is 0.05081379, at which
`teammate_reb_reb` degrades 8.27%. Under the original requirements the
envelope stops at 0.04800719, so 20% is out of reach by 0.0028 of latent
parameter and by 7.85 percentage points of count-space reduction.

E needs one qualification even where it reads "yes". Reaching 20% requires
setting this entry to 0.0508138 or above, which is above the generic
estimator's own pre-2024 value of 0.05268495 only in the sense that the
estimator's value also clears it -- but the pre-2024 forward folds decline to
prefer that estimator at all (section 5), and the count moment wants a rho 5.28
sandwich errors above it (section 2). Choosing a value for this one bucket that
clears the gate, without forward-fold support for the estimator that produced
it, is the bucket-specific free parameter the brief forbids. That is why
section 6's decision is `STOP` under either gate set.

## The exact trade curve

Every row is an evaluated covariance assembly over all 400 held-out games
through the exact predictor, not a fit. `red%` is the focal bucket's
count-space absolute-error reduction against the accepted repair; `reb deg%` is
`teammate_reb_reb`'s count-space absolute-error degradation against the same
control. Negative degradation means the candidate is still *better* than the
control on that bucket. The curve in `count_space_forensic.json` has 47
evaluated points on a 0.00025 grid plus the three exact boundaries and the
exact 20% crossing; the rows below are an excerpt of it.

| entry | focal count corr | focal abs err | red% | reb_reb abs err | reb deg% | latent RMSE ratio | mean w^2 | pipeline | original |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.045773 | +0.03823120 | 0.0176341 | 5.19 | 0.00106470 | -5.89 | 0.8539 | 0.940013 | pass | pass |
| 0.046773 | +0.03882645 | 0.0170388 | 8.39 | 0.00109345 | -3.35 | 0.8445 | 0.933281 | pass | pass |
| 0.047773 | +0.03939565 | 0.0164696 | 11.45 | 0.00112402 | -0.65 | 0.8478 | 0.926139 | pass | pass |
| **0.048007** | **+0.03952614** | **0.0163391** | **12.15** | **0.00113134** | **0.00** | 0.8505 | 0.924431 | pass | **boundary** |
| 0.049023 | +0.04007702 | 0.0157883 | 15.11 | 0.00116388 | +2.88 | 0.8698 | 0.916836 | pass | fail |
| 0.049773 | +0.04047107 | 0.0153942 | 17.23 | 0.00118849 | +5.05 | 0.8919 | 0.911103 | pass | fail |
| **0.050814** | +0.04098649 | 0.0148788 | **20.00** | 0.00122486 | +8.27 | 0.9327 | 0.902529 | pass | fail |
| 0.051523 | +0.04132719 | 0.0145381 | 21.83 | 0.00124990 | +10.48 | 0.9665 | 0.896654 | pass | fail |
| 0.052667 | +0.04186018 | 0.0140051 | 24.70 | 0.00129065 | +14.08 | 1.0300 | 0.887142 | pass | fail |
| **0.052995** | **+0.04200935** | **0.0138559** | **25.50** | **0.00130239** | **+15.12** | **1.0500** | 0.884410 | **boundary** | fail |
| 0.053273 | +0.04213451 | 0.0137308 | 26.17 | 0.00131234 | +16.00 | 1.0675 | 0.882099 | fail | fail |
| 0.054773 | +0.04279313 | 0.0130722 | 29.71 | 0.00136605 | +20.75 | 1.1699 | 0.869729 | fail | fail |
| 0.056273 | +0.04342056 | 0.0124447 | 33.09 | 0.00141994 | +25.51 | 1.2833 | 0.857442 | fail | fail |

Both quantities are monotone in the entry over this range, so there is a single
trade and no region where both improve. Read as a budget:

| `teammate_reb_reb` give-back allowed | focal reduction available | first evaluated entry that exceeds the budget |
| --- | --- | --- |
| 0% (the original requirement) | 12.15% | 0.0480228 |
| 5% | 16.53% | 0.0497728 |
| 10% | 21.19% | 0.0515228 |
| 15% | 24.96% | 0.0529950 |
| 20% | 29.14% | 0.0547728 |

The 20% focal gate needs a give-back budget of roughly 8.3%. The latent-RMSE
tolerance is the second constraint to bind, at ratio 1.03 (entry 0.0526668) and
1.05 (entry 0.0529950); `teammate_ast_ast` stays better than the control
throughout -- it starts 19.7% better and is still 14.2% better at entry 0.056 --
so it never binds in this range, and `count_rmse` improves monotonically
because the focal bucket dominates it.

## 1. The five estimators, pooled 2020-2023

1,068,528 ordered same-team pairs over 5,133 games, all pre-2024.

| | estimator | value | uncertainty |
| --- | --- | --- | --- |
| A | randomized-PIT reading (what the pipeline's latent gate uses) | +0.04440020 | 0.00148757 game-clustered |
| B | A averaged over 48 deterministic jitter seeds | +0.04442616 | 0.00039946 across seeds |
| C | mid-PIT, standardized by its own spread (diagnostic only) | +0.04916418 | -- |
| D | interval-censored Gaussian-copula MLE | +0.05268495 | 0.00147010 clustered sandwich |
| E | direct observed count correlation | +0.05459493 | 0.00189765 game-clustered |

One scalar per bucket from one generic procedure. No bucket-specific free
covariance parameter is introduced anywhere in this study.

## 2. Transmission, and the attenuation verdict

```
RANDOMIZED-PIT ATTENUATION: YES
```

Three independent legs:

1. **Size.** D exceeds A by 0.00828, which is 3.96 pooled standard errors.
2. **Mechanism.** Averaging over 48 jitter seeds moves A by 2.6e-5. The jitter
   enters the statistic's *denominator*, so it contributes variance and not
   bias, and averaging can only remove variance. The across-seed spread
   (0.00040) is a quarter of A's own clustered standard error.
3. **Correctability.** Inverting A through its own analytic transmission series
   gives +0.05154, which is 0.78 sandwich errors below D. If the gap were noise
   or model error, the inversion would not reconcile the two.

The ordering A < C < E is the predicted one: mid-PIT carries no jitter variance,
so standardising by its own spread removes the part of the attenuation the
randomized PIT keeps in its denominator.

### What the pipeline does with it

`04_validate_shadow_v1.py::latent_dependence_summary` sets each model's
`implied_buckets` to `bucket_values(same_team_correlation(),
cross_team_correlation())` -- the architecture's **raw copula parameter
matrices**. The latent gate therefore compares a copula *parameter* against a
randomized-PIT *sample moment*, with no transmission in between. This study
reproduces every published latent implied bucket, for both runs, to 0.0 by
reading the parameter rather than by simulating. On the held-out universe the
focal bucket's parameter is +0.04577278 and the reading a simulation of it
actually produces is +0.03676583, a ratio of 0.803. That difference is the
attenuation the gate never charges, and it is why an estimator fitted to
minimise that error is driven to the attenuated value -- which the simulator
then uses as a real copula correlation.

### The part no estimator can fix

Every estimator of the *latent* correlation lands below the value the
count-space moment implies:

| estimator | latent rho | implied count | observed count | error |
| --- | --- | --- | --- | --- |
| A randomized PIT | +0.05154132 | +0.04652747 | +0.05459493 | -0.00806746 |
| B multi-seed | +0.05157145 | +0.04655476 | +0.05459493 | -0.00804018 |
| D censored MLE | +0.05268495 | +0.04756339 | +0.05459493 | -0.00703155 |
| C mid-PIT | +0.05310008 | +0.04793945 | +0.05459493 | -0.00665548 |
| E count moment | +0.06044314 | +0.05459493 | +0.05459493 | 0 |

The count moment wants rho = +0.06044, which is 5.28 sandwich errors above the
censored MLE. Over this range the count-space error is monotone in rho, so the
count-space ranking of the estimators is just their ordering in rho -- and the
*correct* estimate of the copula parameter is not the closest one. `C_mid_pit`
beats `D` purely by overshooting. Correcting the estimator removes 12.84% of
the pre-2024 count-space discrepancy and leaves the rest, and that remainder is
a statement about the Gaussian copula and the production margins, not about any
estimator.

## 3. The feasibility envelope

### The instrument

The envelope is read off an exact held-out predictor, not a simulation. Each
bucket is pooled over *all* its ordered pairs in all 400 held-out games using
the per-team factorisation `sum_{i != k} u_i v_k = (sum u)(sum v) - sum(u v)`
per Hermite order, which is the same identity `factors.pair_moments` uses, so
the pair weighting is identical to the observed statistic's. It reproduces both
published Monte Carlo runs to a maximum absolute difference of 7.7e-5 (control)
and 7.9e-5 (candidate), rms 5.0e-5 -- at the Monte Carlo noise floor -- and
takes seconds rather than eighty minutes. It is deterministic, so paired
comparisons through it carry no simulation noise at all.

### The lever

One same-team off-diagonal entry retargeted through the competition Gram. Every
cross-team block and every other same-team entry comes back identically
unmoved to machine precision, PSD never fails anywhere on the sweep, and the
same-player blocks stay pinned at 2.2e-16.

### What binds

The frozen contrast Gram sits *on* the PSD boundary (minimum eigenvalue
7.6e-19). Any step in either direction drives it negative and has to be bought
with competition inflation. Inflation adds `delta * n * I` to every
within-player shared block, `_largest_feasible_shrink` has to re-pin that, and
the pinning is paid for in per-player shrink. Shrink multiplies every realised
cross-player correlation, so the price of moving one entry is charged to all
twelve buckets.

| entry | inflation | mean w^2 | focal count | reduction | latent RMSE | count RMSE | fails |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.045773 (candidate) | 3.0e-18 | 0.940013 | +0.03823120 | 5.19% | 0.0023619 | 0.0060989 | -- |
| 0.048007 (commissioned max) | 1.9e-03 | 0.924431 | +0.03952614 | 12.15% | 0.0023523 | 0.0058293 | -- |
| 0.051500 | 5.1e-03 | -- | +0.04131637 | 21.77% | 0.0026702 | 0.0054976 | reb_reb |
| 0.052667 (latent max) | 6.2e-03 | 0.887142 | +0.04186018 | 24.70% | 0.0028489 | 0.0054080 | reb_reb |
| 0.070000 | 2.3e-02 | -- | +0.04732252 | 54.07% | 0.0070350 | 0.0050044 | latent, reb_reb |
| 0.200000 | 1.5e-01 | -- | +0.03448618 | -14.95% | 0.0442257 | 0.0102315 | latent, count, both teammates |

The reduction peaks at 54.26% at entry 0.080 and falls after, because past that
the shrink representability costs pulls the realised correlation back down
faster than the parameter pushes it up. The entry at which the 20% goal is
first reached is 0.0508138.

The *extremes* of the shrink barely move -- the binding player is the same one,
so minimum shrink sits at 0.825114 all the way to entry 0.0527 -- which is why
the range hides the cost. The quantity that carries it is the mean of `w^2`,
the factor every realised same-team correlation is scaled by: it falls 1.66% by
the commissioned boundary and 5.63% by the latent boundary, and a quarter to a
third of all player-slots have their shrink cut (by up to 0.066 and 0.205
respectively). That is how moving one entry degrades buckets whose parameters
never moved.

### The envelope depends on how one constraint is read

Two of the commissioned constraints -- "protected opponent buckets within
tolerance" and "`teammate_ast_ast` / `teammate_reb_reb` no worse than control"
-- are this repository's own language, so they are scored with this
repository's own definitions. `05_evaluate_gates.py` scores both as a *latent z
degradation* against the control: gate 5 allows one z on the protected set,
gate 1 allows a quarter of a z on the two repaired targets. Operationalising
the opponent constraint instead as a count-space error ratio rejects the frozen
candidate itself -- on the held-out universe its count-space opponent errors
already exceed the control's by up to 8.9%, and it passed its gates regardless,
because its gates never asked that.

"No worse than control" on the two repaired buckets is the one that moves the
answer, so three readings are reported -- the two in the answers table above
plus the intermediate one that isolates which of the two differences does the
work:

| reading | max feasible entry | reduction there | reaches 20%? | what binds |
| --- | --- | --- | --- | --- |
| `commissioned` -- 1.03 tolerances, give-back scored in count space too | 0.0480072 | 12.15% | **no** | `teammate_reb_reb` give-back |
| `give_back_in_latent_space` -- 1.03 tolerances, gate 1's latent z only | 0.0526668 | 24.70% | yes | latent RMSE |
| `pipeline_own_gates` -- the repository's own gate set, 1.05 tolerances | 0.0529950 | 25.50% | yes | latent RMSE |

The middle row isolates the cause: relaxing only the *space* the give-back is
measured in, with the 1.03 tolerances left intact, already moves the boundary
from 0.0480 to 0.0527. The tolerance difference between 1.03 and 1.05 is worth
only a further 0.0003. So the whole gap between the two answers is the
count-space give-back on `teammate_reb_reb`, not the RMSE tolerance.

`commissioned` is the primary reading, because the 20% goal is itself stated in
count space and it would be incoherent to demand a count-space gain on one
bucket while refusing to measure the count-space give-back on another. Under it
the entry needed for 20% is 0.0508138 and the largest feasible entry is
0.0480072, so the goal is out of reach. `teammate_reb_reb`'s count-space error
rises monotonically with the lever -- 0.00106470 at the candidate's own entry,
0.00113134 at 0.0480072, 0.00129065 at 0.0526668, 0.00133837 at 0.054 -- and
crosses the control's 0.00113134 at 0.0480072, which is where the boundary is.

The latent-RMSE boundary is available in closed form, because the lever moves
one of the twelve parameters the latent gate scores and nothing else:

```
12 * rmse(S)^2 = 6.540895651417e-05 + (S - 0.047011355026)^2
```

which gives `S_max = 0.052666838`. The bisection agrees to 6.9e-14, so the
boundary is a root of a quadratic rather than a bisection artefact.

## 4. Identifiability

`IDENTIFIABLE_BUT_GLOBALLY_INCOMPATIBLE`, from predicates rather than
assertion:

| predicate | |
| --- | --- |
| the bucket is identified from the current marginals | yes |
| the two estimators differ materially | yes |
| the envelope reaches the 20% goal | no |
| the generic estimator's own value is inside the envelope | no |
| the generic estimator's own value reaches the goal | yes |
| applying the generic estimator to all twelve buckets is feasible | no |
| the pre-2024 forward folds agree the estimator is better | no |
| the count moment wants a rho no estimator supports | yes |

The root cause stays in section 2's verdict -- randomized-PIT attenuation is
confirmed -- and is reported alongside the classification so the single label
does not bury it. The classification is the *feasibility* fact because that is
the one that entails the others: it already says the parameter is identified
and that the correction cannot be carried.

## 5. The inner forward test, and why no refit was run

Section 5 permits refitting the existing PSD factor model to an improved latent
target only if the generic estimator improves this bucket on the pre-2024
forward folds. **It does not.**

| fold | observed count | best estimator | D's error reduction vs A |
| --- | --- | --- | --- |
| 2022 | +0.05522488 | `C_mid_pit` | +6.84% |
| 2023 | +0.04572846 | `A_randomized_pit` | -15.16% |

The target moves 0.0095 between the two folds; the four estimators' implied
values differ by at most 0.0014 within either fold. Whichever estimator happens
to sit on the side the season moved towards wins that fold, so the folds pick
different winners. That is a statement about the target's year-to-year
stability, not about the estimators. The precondition is not met, no refit was
performed, and no new factor family was introduced.

### The generic estimator applied honestly

The estimator is generic, so it was also applied at face value in both the
forms that do not require a choice:

- **Focal bucket only, at its own pre-2024 value 0.05268495.** Reaches 24.74%,
  and fails `latent_rmse_within_tolerance` (ratio 1.0311 against a 1.03 bound)
  and `teammate_reb_reb_no_worse`. It *passes* under the pipeline's own 1.05
  tolerance. On the latent constraint alone the decision is 0.11% of latent
  RMSE from flipping and the value sits 0.0123 sandwich errors above the
  analytic latent boundary, so the two are statistically indistinguishable
  there. The count-space give-back on `teammate_reb_reb` is not
  indistinguishable: 0.00129130 against the control's 0.00113134, a 14.1%
  degradation, resolved exactly by a deterministic predictor.
- **All twelve buckets at once.** Reduces the focal count error 20.9% and
  breaks the latent gate outright: latent RMSE 0.0031559 against a bound of
  0.0028489. Every bucket's latent reading is attenuated, so a correct
  estimator raises every target in magnitude, and the gate charges for all of
  it.

## 6. Stop rule

No generic estimator reaches a 20% count-space error reduction at its own value
with every global constraint intact. Section 6 applies: **STOP**. No further
model was invented.

## What this means for the blocker

The estimator defect is real, confirmed three ways, and worth fixing on its own
terms -- the latent gate currently scores a copula parameter against an
attenuated sample moment, which biases every fitted dependence parameter low.
But fixing it does not deliver the 20% count-space target:

1. Under the commissioned constraints the architecture cannot carry the
   corrected value, because the frozen Grams sit on the PSD boundary and every
   step is paid for in per-player shrink, which gives back `teammate_reb_reb`.
2. Even with the constraints read the pipeline's own way, the corrected value
   gets to roughly 25% on the holdout but the pre-2024 forward folds do not
   support preferring the corrected estimator at all.
3. The count-space moment wants a latent rho 5.3 sandwich errors above the
   best latent estimator, so a residual gap survives any estimator choice.
   That residual belongs to the Gaussian copula and the production margins.

## A dormant defect found along the way

`src/nba_prop_quant/research/game_latent_state/transmission.py::conditional_hermite_moments`
carries the probabilists' Hermite recurrence as `z * He_k - (k - 1) * He_{k-1}`
where it should be `z * He_k - k * He_{k-1}`. The shifted coefficient is exact
for `He_0` and `He_1`, so the function is correct at the orders anything in the
pipeline asks of it -- `DEFAULT_BRIDGE_ORDER` is 2, and both the inner
selection and the frozen spec builder pass it -- and wrong from `M_3` on
(relative errors 0.35 to 3.2 against direct quadrature). **The frozen candidate
is numerically unaffected.** It is reported and not fixed, because the frozen
candidate is not this study's to touch; `forensic_lib.conditional_hermite_moments`
carries a correct implementation and `tests/test_game_latent_state_shadow_count_space_forensic.py` locks
both facts against quadrature.

## Two marginal-training conventions

`02_build_oof_residuals.py` drops rows missing any of the six stats or any of
the six selected means before splitting, so its marginals are fitted on the
intersection across stats, while `04_validate_shadow_v1.py::fit_season` drops
only the stat it is fitting. The two give CDF bounds that differ by up to 1e-2
on pts, ast, stl and fg3m. Pre-existing, reported, not introduced here. This
study declares the `validator` convention per season for everything it
computes, because that is the one the held-out simulation inverts.

`v2_bridge_count_residuals.parquet` is not read: its `analytic_mean_*` columns
reproduce under both conventions to 7.9e-13 but its `analytic_sd_*` columns
reproduce under neither, differing by 1.5e-2 to 1.1e-1, so its provenance
cannot be stated. The pre-2024 count residuals are rebuilt here instead.

## Layout on this branch

This study is CLOSED and its parameter change was **not** adopted. The final
clean integration branch therefore carries the report and this write-up but
not the drivers, because nothing at runtime and nothing in the retraining
path calls them. `research/final_integration/carry_manifest.json` records the
exclusion and the reason, and
`research/final_model/final_model_spec.json` records the report's SHA256 so
the conclusion stays attributable.

| file | carried here | |
| --- | --- | --- |
| `count_space_forensic.json` | yes | the report |
| `README.md` | yes | this write-up |
| `01_count_space_forensic.py` | no | the driver; writes `count_space_forensic.json` |
| `forensic_lib.py` | no | estimation windows, pooled pair statistics, the retarget lever, Hermite moments |
| `holdout_lib.py` | no | the exact held-out transmission predictor |
| `marginal.py` | no | the convention-parameterised marginal refit |
| `cache/` | no | gitignored; ZINB refits, the scored held-out universe, the estimator table |

To reproduce, check out one of the research branches listed in the carry
manifest and run
`python research/count_space_forensic/01_count_space_forensic.py`. Everything
in `cache/` is a derived intermediate of committed fits and is rebuilt from
them if absent; the first run takes about twenty minutes, later runs about
ten.

The Hermite finding this study reported has since been fixed in the pipeline
itself: `transmission.conditional_hermite_moments` now carries the correct
`He_{k+1} = z He_k - k He_{k-1}` coefficient, and
`tests/test_game_latent_state_shadow_hermite.py` checks it against
`numpy.polynomial.hermite_e` and against direct quadrature at orders one
through eight, so the cross-check no longer depends on this directory.

The measurement tools are locked in `tests/test_game_latent_state_shadow_count_space_forensic.py` (57
tests) against independent references: synthetic data with a known latent
correlation, direct quadrature, an independent bivariate-normal CDF, and both
published Monte Carlo runs.
