# Final upstream remediation (shadow research)

SHADOW / RESEARCH ONLY. Nothing here is wired into production and nothing here
can promote. The control is the accepted bucket repair
(`C_rank_k6_r6_eb_block_diagonal`, factor spec hash
`c9b46e3a7497cfee52832f29397397b843aafb8e5f770a39157ce508f8157bc0`).

The brief named six upstream weaknesses and asked whether each should be
repaired before the production shadow. Each was given real candidates, each was
decided on strictly pre-2024 chronological folds, and all six were frozen
before the 2024-2025 holdout was opened.

## The six decisions

| item | layer | candidates | selected |
| --- | --- | --- | --- |
| 1 | temporal AST-AST season effects | pooled EB, Gaussian RE, robust Student-t, recency-weighted robust | `A0_pooled_empirical_bayes` |
| 2 | multiplicative `role_scale` | ratio (control), hierarchically shrunk log-scale | `log_shrunk` |
| 3 | cross-team shrinkage prior | Gaussian, Student-t at nu in {40, 15, 8, 5, 3} | `gaussian` |
| 4 | latent-to-count transmission bridge | weight cap in {0.00, 0.15, 0.30, 0.50, 1.00} | cap `0.15` |
| 5 | cross-player dependence temperature | lambda in {0.00, 0.25, 0.50, 0.75, 1.00} | `1.00` |
| 6 | predictive uncertainty calibration | raw, Huber M-scale, median of folds, Student-t predictive | `U0_raw` |

Four of the six land on the control's own behaviour. That is the folds
declining to buy a layer, not the layer being unavailable: every alternative
was fitted and scored on the same folds, and `remediation.select_within_tie_band`
kept the simpler candidate whenever the margin sat inside one standard error of
the paired per-unit difference.

## Why the recorded latent column reads lower than the counts

The apparent contradiction that motivated item 4 -- count-space correlations
that look *amplified* against latent correlations that look attenuated -- is
discreteness, not a modelling error. The recorded `z` is a randomized PIT, so
inside each count atom it carries independent uniform noise. For a Gaussian
copula with cross-player correlation `rho`,

    Corr(z_a, z_b) = sum_k rho^k g_{a,k} g_{b,k} / k!

with `g_1 < 1` everywhere: 0.93 for points down to 0.37 for blocks. The
standardized count residual `e` obeys the same expansion with its own gains
`h`. So the two observables are two readings of one `rho` through two different
attenuations, and they are only comparable after both are inverted.

`transmission.py` does that inversion. The Hermite moments are taken
Rao-Blackwellized rather than from single draws: for an observation with PIT
bounds `[F_l, F_u]`,

    M_k(y) = E[He_k(Z) | Y = y]
           = (He_{k-1}(t_l) phi(t_l) - He_{k-1}(t_u) phi(t_u)) / (F_u - F_l)

with `t = Phi^{-1}(F)`, which removes the randomization noise entirely instead
of averaging over it. The count reading is inverted through `h` to `rho` and
mapped back through `g` into the recorded latent column's units, and the two
sources are then combined by inverse variance under a weight cap.

The combined blocks replace the *raw pooled moments*, before
empirical-Bayes shrinkage. Overriding the shrunk targets instead would have
given the bridged arm a different shrinkage from the control's and confounded
the comparison with the thing being tested.

The pre-2024 homogeneity test rejects at every fold and in both blocks
(max disagreement 8.5 sigma, same-team, 2023). The two sources are genuinely
inconsistent, which is the argument for the cap: 0.15 is the smallest grid
value, and it was also the strict argmax of the forward target improvement.

## Why the temperature is priced by integral and not by simulation

Item 5 is one scalar: lambda multiplies every shared cross-player loading by
`sqrt(lambda)`, so every cross-player block is multiplied by exactly lambda
while each player's own block stays pinned at the incumbent's `R_i` --
algebraically, not approximately, because `build_game_covariance` subtracts
whatever the shared factors contributed to player `i`'s own block.

The quantity separating two temperatures is the same order as the difference
the shadow run already records between the repaired model and the production
incumbent, about 3e-5 in Brier. Resolving that by counting draws needs more
draws than a grid search can afford, so a simulated search would be reading its
own noise. But every leg of a conjunction is a threshold on a count, and every
count is a monotone transform of one latent normal, so every conjunction is an
orthant of a multivariate normal and its price is an orthant integral.
`temperature.py` carries the translation and the integral; the search is exact
and deterministic.

## Files

| file | what it does |
| --- | --- |
| `upstream_spec.py` | one definition of the remediated fit, shared by all four drivers |
| `01_inner_selection.py` | items 1, 2, 3, 4 and 6 on pre-2024 forward folds |
| `02_dependence_temperature.py` | item 5, by exact orthant integration on 2022 and 2023 |
| `03_freeze_spec.py` | freezes all six and writes the spec the V1 validator reads |
| `04_paired_joint_calibration.py` | paired candidate-vs-control multi-leg Brier |
| `05_evaluate_gates.py` | the fourteen gates |

The confirmatory 2024-2025 run goes through the *unmodified*
`research/game_latent_state/04_validate_shadow_v1.py`, pointed at this
directory. The control is re-run through the same driver at the same seed,
simulation count and residual dataset, so the two runs are paired event for
event and gates 9 and 10 can read a paired difference rather than the
difference of two independently noisy reports.

## Holdout discipline

Every driver filters seasons 2024 and 2025 on first read and raises if either
reappears (`assert_holdout_absent`). `03_freeze_spec.py` refuses to freeze
unless both selection artifacts certify that the holdout was unused, and
gate 13 re-checks that certification against the training seasons actually
recorded in the written spec.
