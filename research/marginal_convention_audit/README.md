# OOF-versus-validator marginal convention audit

SHADOW / RESEARCH ONLY. Nothing here is wired into production, nothing can
promote, and nothing here modifies the frozen remediation candidate. The audit
reads `research/final_upstream_remediation/factor_spec.json` (spec hash
`3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4`), its paired
control, the committed OOF residuals and the raw selected-mean history, and
writes only `marginal_convention_audit.json` and this file.

It changes no dependence architecture, fits no factor family, runs no Monte
Carlo and reads no held-out season.

## Verdict

```
NON_MATERIAL
```

Every one of the twelve cross-player dependence buckets moves less than
0.25 of its own standard error under every reading tried, and the global
latent target structure moves less than 3% under every reading tried. Per the
brief's rule the mismatch is **non-material and the audit closes with no code
change**.

| | measured | threshold | held |
| --- | --- | --- | --- |
| largest key bucket shift | **0.0270 z** | 0.25 z | yes |
| largest shift over all twelve buckets | **0.0270 z** | 0.25 z | yes |
| largest shift over every jitter seed and standardization | **0.0803 z** | 0.25 z | yes |
| strictest bound (largest shift over smallest bucket error) | **0.1186 z** | 0.25 z | yes |
| largest global latent movement | **0.95%** | 3% | yes |

## The mismatch

The pipeline fits the same production marginal family on two different row
sets.

`02_build_oof_residuals.py::build_residuals` takes a joint `dropna` across all
six stats and all six selected means *before* splitting into train and target,
so its marginals are trained on the intersection across stats. Its own
per-stat `dropna` runs afterwards, on a frame the joint one already cleaned,
and so cannot remove another row — which is what makes the deviation easy to
miss on a read. Those marginals produced the committed
`cdf_lower_`/`cdf_upper_`/`z_` columns, hence every latent dependence target
and the fitted factor structure.

`04_validate_shadow_v1.py::fit_season` drops only the stat it is fitting and
that stat's own mean. So does the production path,
`scripts/07_fit_marginals.py::candidate_frame`:

```python
return frame[
    frame[mean_col].notna() & frame[target].notna()
].copy()
```

**The validator convention is the live one. The residual builder is the single
deviation of the three.**

### The mechanism is exactly one season

A season survives the joint filter only if *every* stat has a selected mean in
it, so the stat available in the fewest seasons sets the training window for
all six. Season 2017 carries selected means for `pts`, `ast`, `stl` and `fg3m`
but not for `reb` or `blk`, so the joint filter removes 2017 entirely:

| residual season | stat | live training rows | OOF training rows | discarded | live window | OOF window |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | pts, ast, stl, fg3m | 79,683 | 51,855 | 27,828 (34.9%) | 2017-2019 | 2018-2019 |
| 2020 | reb, blk | 51,855 | 51,855 | 0 | 2018-2019 | 2018-2019 |
| 2021 | pts, ast, stl, fg3m | 104,635 | 76,807 | 27,828 (26.6%) | 2017-2020 | 2018-2020 |
| 2022 | pts, ast, stl, fg3m | 132,580 | 104,752 | 27,828 (21.0%) | 2017-2021 | 2018-2021 |
| 2023 | pts, ast, stl, fg3m | 160,218 | 132,390 | 27,828 (17.4%) | 2017-2022 | 2018-2022 |

The discarded count is the same 27,828 rows in every window because it is one
fixed season, so the damage shrinks as the window grows: a third of the
training data for the earliest fold, a sixth for the latest.

`reb` and `blk` are the binding stats, so for them the two conventions are the
same row set and the change is an exact identity. Those exact zeros propagate
all the way to the buckets and are the audit's internal consistency check.

### Only the training channel is live

`build_residuals` filters the training rows and the target rows with the same
joint `dropna`, so the convention has two channels in principle. The data
closes the second: inside the residual seasons 2020-2023 all 108,703 history
rows are already complete on all six stats and means, so the joint filter
removes nothing there. Even if it did, `pair_moments` applies its own joint
`dropna` across every `zs_` column before accumulating, so a row missing one
stat could never enter a bucket. The audit therefore holds the target rows at
the published set and swaps only the training rows, which keeps the comparison
paired on the same player-games.

## The reproduction is exact

Every difference reported below is a difference against a *recomputation* of
the published convention, so that recomputation has to land on the published
numbers. It does, three independent ways, all to 0.0:

| reference | max absolute difference |
| --- | --- |
| committed `v_`, `cdf_lower_`, `cdf_upper_`, `u_`, `z_` columns, all six stats, all four seasons | **0.0** |
| `oof_residual_build_report.json` per-stat `pit_mean`, `pit_variance`, `z_mean`, `z_sd`, `clipped_fraction` | **0.0** |
| `factor_spec.json` six recorded `standardization_moments` | **0.0** |

The third is the one that pins the *window*: those six numbers were computed
from the committed pre-2024 residuals when the candidate was frozen, so
reproducing them shows this audit's pre-2024 window and standardization are
the ones the frozen fit used. The focal bucket's OOF reading, +0.04440020 with
a clustered error of 0.00148757, also matches the count-space forensic study's
independently computed `A_randomized_pit` to every digit.

## 1. The effect on the pre-2024 PITs

Pooled over 108,703 player-games in 2020-2023, holding the target rows, the
jitter draw and the marginal family fixed:

| stat | max \|ΔF\| | mean \|ΔF\| | rms Δz | max \|Δz\| | mean Δz |
| --- | --- | --- | --- | --- | --- |
| pts | 3.669e-02 | 8.435e-04 | 6.071e-03 | 1.263e-01 | −7.879e-04 |
| reb | **0** | **0** | **0** | **0** | **0** |
| ast | 5.797e-02 | 1.061e-03 | 6.492e-03 | 1.412e-01 | +7.402e-04 |
| stl | 2.325e-02 | 5.735e-04 | 3.701e-03 | 1.020e-01 | −9.420e-05 |
| blk | **0** | **0** | **0** | **0** | **0** |
| fg3m | 3.250e-02 | 6.886e-04 | 4.373e-03 | 1.120e-01 | +2.552e-04 |

So the live convention moves a CDF bound by up to 0.058 on `ast` and by about
1e-3 on average, and moves the latent residual by about 0.005 of a standard
deviation in rms. This refines the count-space forensic study's "up to ~1e-2
on pts, ast, stl and fg3m", which was a fair statement of the typical size but
understated the tail: the maximum is 5.8e-2, not 1e-2.

The marginal-calibration diagnostics move in the direction more training data
predicts. Kolmogorov-Smirnov distance from the uniform improves on 12 of the
16 affected stat-seasons, and the four that worsen do so by at most 0.00061
against improvements up to 0.00097. For 2020, the fold where a third of the
training data was being discarded, all four improve (`ast` 0.01711 to 0.01617,
`fg3m` 0.00893 to 0.00796, `pts` 0.00823 to 0.00787, `stl` 0.00672 to
0.00657).

So the live convention's marginals are mildly *better* calibrated, which is
what one would expect from a fifth to a third more training rows. That is an
argument for the live convention on its own terms; it is not an argument that
the dependence layer is affected, which is what this audit measures.

## 2. The effect on the twelve dependence buckets

Pre-2024, 5,133 games, 1,068,528 ordered same-team pairs and 1,163,906
cross-team pairs. `z` is the shift divided by the bucket's own game-clustered
bootstrap standard error — the same `observed_bucket_se` the remediation gates
divide by. **T** marks a gate-1 target bucket, **P** a gate-5 protected one.

| bucket | OOF | live | shift | se | z | rel. | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `passer_ast_teammate_pts` | +0.04440020 | +0.04436006 | −4.014e-05 | 1.488e-03 | **−0.0270** | −0.09% | T |
| `opponent_ast_ast` | +0.00893793 | +0.00891288 | −2.505e-05 | 1.489e-03 | −0.0168 | −0.28% | P |
| `teammate_pts_pts` | −0.00265188 | −0.00263141 | +2.047e-05 | 1.690e-03 | +0.0121 | +0.77% | |
| `opponent_stl_pts` | −0.00064759 | −0.00063708 | +1.051e-05 | 9.930e-04 | +0.0106 | +1.62% | |
| `opponent_fg3m_reb` | −0.00434257 | −0.00435285 | −1.029e-05 | 1.025e-03 | −0.0100 | −0.24% | P |
| `teammate_fg3m_fg3m` | −0.00142219 | −0.00143536 | −1.318e-05 | 1.343e-03 | −0.0098 | −0.93% | |
| `opponent_pts_pts` | +0.01637376 | +0.01638064 | +6.881e-06 | 1.468e-03 | +0.0047 | +0.04% | P |
| `opponent_pts_reb` | −0.00480979 | −0.00481462 | −4.833e-06 | 1.157e-03 | −0.0042 | −0.10% | P |
| `teammate_pts_reb` | +0.01453622 | +0.01454189 | +5.667e-06 | 1.431e-03 | +0.0040 | +0.04% | P |
| `teammate_ast_ast` | +0.00549054 | +0.00549178 | +1.245e-06 | 1.634e-03 | +0.0008 | +0.02% | T |
| `teammate_reb_reb` | +0.00608248 | +0.00608248 | **0** | 1.704e-03 | **0** | 0% | T |
| `opponent_reb_reb` | +0.01195971 | +0.01195971 | **0** | 1.490e-03 | **0** | 0% | P |

The two `reb`×`reb` buckets are exactly unmoved, as they must be: `reb` is one
of the binding stats, so its PITs are identical under both conventions. The
largest move in the table is the focal bucket of the count-space study, and it
is 0.027 of a standard error — an order of magnitude inside the 0.25
threshold, and in the *wrong* direction to help that study's shortfall.

### The verdict does not rest on the pipeline's standardization

`standardize_residuals` divides each stat by its own empirical spread, and the
convention moves that spread by up to 0.3%, so the pipeline's own reading
absorbs the part of the marginal change that is pure level and scale. It does
absorb some: re-reading the buckets with the scaling held fixed, and with no
scaling at all, gives a larger answer.

| standardization | largest shift | at | matched z | conservative bound |
| --- | --- | --- | --- | --- |
| each convention's own moments (the pipeline's choice) | 4.014e-05 | `passer_ast_teammate_pts` | 0.0270 | 0.0404 |
| both at the frozen spec's moments | 1.095e-04 | `opponent_pts_pts` | 0.0746 | 0.1103 |
| none at all | 1.178e-04 | `opponent_pts_pts` | **0.0803** | **0.1186** |

Unstandardized, where nothing can be absorbed, the largest bucket move is
0.080 z — three times the pipeline's reading and still a third of the
threshold. "Matched z" divides each bucket's shift by its own standard error,
which is what a gate would read. "Conservative bound" divides the largest
shift by the *smallest* standard error over all twelve, a pairing no bucket
realises but none can exceed; the verdict holds on that too.

### The verdict does not rest on one jitter draw

The bucket reading is a randomized-PIT sample moment. Both conventions share
the committed draw, so the paired shift cancels most of the jitter variance,
but the size of the shift could still be draw-specific. Re-drawing under eight
seeds and re-reading both conventions:

| seed | largest shift | at | matched z |
| --- | --- | --- | --- |
| 73 (the committed draw) | 4.014e-05 | `passer_ast_teammate_pts` | 0.0270 |
| 1000 | 4.754e-05 | `passer_ast_teammate_pts` | 0.0320 |
| 1001 | 4.650e-05 | `passer_ast_teammate_pts` | 0.0313 |
| 1002 | 4.773e-05 | `passer_ast_teammate_pts` | 0.0321 |
| 1003 | 5.458e-05 | `passer_ast_teammate_pts` | **0.0367** |
| 1004 | 5.275e-05 | `passer_ast_teammate_pts` | 0.0355 |
| 1005 | 4.320e-05 | `passer_ast_teammate_pts` | 0.0290 |
| 1006 | 4.167e-05 | `passer_ast_teammate_pts` | 0.0280 |

The same bucket is the largest under every seed and the range is 0.027 to
0.037 z, so the measurement is not draw-specific.

## 3. The effect on the global latent structure

"Global latent structure" admits more than one defensible meaning, so four are
reported and the decision is taken on the largest. No model is refitted
anywhere: both sets of loadings are read off their frozen specs.

| reading | movement |
| --- | --- |
| twelve-bucket target vector, relative L2 | 0.11% |
| full 6×6 same-team pair-moment matrix, relative Frobenius | 0.16% |
| full 6×6 cross-team pair-moment matrix, relative Frobenius | 0.19% |
| frozen candidate's latent RMSE against the two target sets | **0.95%** |
| frozen control's latent RMSE against the two target sets | 0.94% |

The RMSE readings are the largest because they measure a *residual*: the
frozen models fit these targets to about 6.5e-4 against target magnitudes near
1e-2, so a 0.11% move in the targets is a 0.95% move in what is left over.
That is the gate-3 analogue and the strictest of the four, and it is a third
of the 3% threshold. The candidate's fit is 0.95% worse against the live
targets and the control's is 0.94% better, both negligible.

The standardization moments themselves move by at most 0.29% in spread
(`pts`, 0.99913 to 1.00203) and are exactly unmoved for `reb` and `blk`.

## 4. Where the mismatch still has a consequence

The audit's rule is satisfied, so the convention is not worth changing on the
evidence in scope. One channel is out of scope by the brief's own instruction
and should not be read as closed by this verdict.

The dependence parameters are calibrated in a latent space defined by the OOF
builder's marginals, while the held-out simulation inverts `fit_season`'s. The
latent space the parameters mean is therefore not quite the latent space the
simulator assumes. Aligning the residual builder would close that without
touching the simulator at all, since `fit_season` already uses the live
filter. The size of the misalignment in latent units is section 1's table —
about 0.005 of a standard deviation in rms, up to 0.14 at the tail, on four of
the six stats. Converting it into a count-space number requires re-inverting
the margins through the simulation, which the brief defers, so it is bounded
here and not estimated.

## Layout

| file | |
| --- | --- |
| `01_marginal_convention_audit.py` | the driver; writes `marginal_convention_audit.json` |
| `marginal_convention_audit.json` | the report |
| `cache/` | gitignored; shared with the count-space forensic study so a convention's marginals are one object in both |

The driver imports `research/count_space_forensic/forensic_lib.py` for the two
row filters rather than carrying a second copy: a copy of the thing under
audit could only measure itself. Reproduce with
`python research/marginal_convention_audit/01_marginal_convention_audit.py`.
A cold run refits eight walk-forward ZINB marginal sets and takes about twenty
minutes; with the cache warm it is about six.

The measurements are locked in
`tests/test_game_latent_state_shadow_marginal_convention_audit.py` against the
three pipeline sources, against synthetic data with a known latent
correlation, and against the published artifacts. One of those tests is there
specifically to keep this verdict honest: a "no material effect" conclusion is
worth nothing from a dead instrument, so the bucket measurement chain is shown
to recover a known correlation and to resolve a deliberate shift far more
sharply than the 0.25 z threshold.
