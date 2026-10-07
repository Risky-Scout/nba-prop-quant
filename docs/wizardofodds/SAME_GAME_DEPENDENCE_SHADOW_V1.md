# NBA Prop Quant — Same-Game Dependence Shadow v1

## Status

**Shadow only. The incumbent remains the served and published authority.**

This document describes a frozen same-game dependence model that runs beside
production, logs what it would have said, and is graded against the incumbent
after results arrive. It publishes nothing. It cannot promote itself.

| | |
| --- | --- |
| final model specification | `research/final_model/final_model_spec.json` |
| final model specification SHA256 | `154109658127920ebeac381bd21e8bb29390505384124419899236cce37fdc64` |
| authoritative factor specification hash | `3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4` |
| status | `FROZEN` |
| served probability | the incumbent's, on every row, in every branch |
| publishing switch | `DISABLED` |
| promotion authority | `NONE__SHADOW_HAS_NO_AUTONOMOUS_PROMOTION_AUTHORITY` |

The incumbent marginal model and its public claim policy are unaffected. See
`docs/wizardofodds/PUBLIC_CLAIM_POLICY_V1.md`, which governs what the site may
say and which this document does not modify, widen or reinterpret.

## What it does

The shadow consumes the same live production marginals the incumbent consumes,
builds a full-game latent covariance across both teams, and answers arbitrary
same-game joint probability queries from it. For every query it records the
candidate probability, the incumbent probability, a cross-player independence
probability, the model and artifact hashes, PSD diagnostics, the same-player
block deviation, and a fallback reason when the candidate did not produce a
number.

The incumbent's within-player dependence block is pinned exactly. The shared
factors are added orthogonally around it, so the shadow adds cross-player
structure without re-deciding anything the incumbent already decides.

## Gate results

**14/14 IMPLEMENTED FINAL GATES PASS.**

The fourteen gates are the implemented final gate set, evaluated over 400 games
on the 2024–2025 holdout. All fourteen pass. The gate definitions and the
evaluated evidence are recorded in the final model specification named above.

## The original ≥20% count-space target was NOT achieved

This is the one original requirement that was retired rather than met, and it
must not be reported as a pass.

**The original `>=20%` `passer_ast_teammate_pts` count-space error-reduction
target was NOT achieved under the full original constraint set.**

It was retired because the forensic feasibility study proved it incompatible
with the other original constraints under the current architecture. The
original no-worse `teammate_reb_reb` constraint binds first:

- reaching a 20% focal count-space error reduction requires a latent entry of
  `0.0508138`;
- the largest entry that is feasible under the original constraint set is
  `0.0480072`, so the target is out of reach by `0.0028` of latent entry;
- at the entry that would deliver exactly 20%, `teammate_reb_reb`'s count-space
  absolute error degrades by `+8.27%` against the control, and the original
  requirement is that it be no worse;
- the first binding constraint under the original reading is therefore
  `teammate_reb_reb_no_worse`, not the latent-RMSE tolerance, which does not
  bind anywhere in this range.

The study classified the situation as `IDENTIFIABLE_BUT_GLOBALLY_INCOMPATIBLE`
and concluded `COUNT-SPACE BLOCKER IS AN ESTIMATOR ISSUE BUT FIX VIOLATES
GLOBAL CONSTRAINTS`. The parameter change the study identified was **not
adopted**: taking it would have met the focal target by breaking a different
original requirement, which is not a pass.

Evidence: `research/count_space_forensic/count_space_forensic.json` and
`research/count_space_forensic/README.md`.

## Recorded dispositions

These strings are recorded in `research/final_model/final_model_spec.json` and
are reproduced here because this is the production-facing record.

```
COUNT_SPACE_FORENSIC_CHANGE_ADOPTED = NO

COUNT_SPACE_20_PERCENT_REQUIREMENT_DISPOSITION =
INCOMPATIBLE_WITH_OTHER_ORIGINAL_CONSTRAINTS_UNDER_CURRENT_ARCHITECTURE

MARGINAL_CONVENTION_AUDIT = NON_MATERIAL

UNCERTAINTY_CALIBRATION =
RAW_MONITOR_ONLY_NOT_USED_FOR_MODEL_SELECTION_OR_PROMOTION
```

## What may not be said

Each prohibition below is written to stand on its own, so that quoting one
line out of context cannot invert it.

- It may not be said that the original 20% `passer_ast_teammate_pts`
  count-space target was met, achieved, satisfied or passed. It was not.
- It may not be said that the retired target was "effectively" met, or met
  "under a corrected estimator", or met in any other qualified form. The
  corrected estimator does reach the focal target, and taking it breaks
  `teammate_reb_reb_no_worse`, which is why it was not adopted.
- It may not be said that the shadow model is published, served, live or in
  production. It is none of those.
- It may not be said that the shadow model is better than the incumbent. Its
  graded sample is far too small to support any comparative claim, and the
  cross-player independence arm is not yet separable from it.

The shadow's own replay evidence is an operational check that the wiring works.
It is not evidence of superiority and may not be cited as such.

## Promotion

There is no autonomous promotion path and no autonomous publishing path.

`assert_no_promotion_authority` raises unconditionally. Grading raises if asked
for a promotion verdict. `publish_shadow_probabilities` refuses while the
switch is disabled and refuses when it is enabled, because no publisher has
been built. Enabling publication requires the committed switch state, an
environment variable and an approval token to agree, and then a separate,
reviewable change to build the publisher.

Promotion is a human decision taken on accumulated live evidence against the
incumbent, under the minimum-sample rules in the public claim policy.
