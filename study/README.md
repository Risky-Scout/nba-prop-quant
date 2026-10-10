# NBA Prop Quant: read the model in three parts

Start with `python -m study.check lesson`. You do not need to read this whole folder first.

This is a separate, offline study implementation of the **pregame calculation engine**. It consolidates the existing calculations into three files and adds two clear entry points. It is not a completed replacement for the production platform. The production source, fitted artifacts, jobs, and configuration are untouched.

| Read in this order | What it does | First functions to read |
| --- | --- | --- |
| `price.py` | Turn a distribution into probabilities and prices | `conditional_nonpush_probabilities`, `fair_american` |
| `model.py` | Predict minutes and means; fit distributions and dependence; adjust for lineups | `ModelBundle.predict`, `combine_mean_components` |
| `data.py` | Construct the information available before a game | `_lagged_ewm`, `build_upcoming_slate_features` |

`check.py` contains the executable lesson, comparisons with the original code, and an artifact exporter. It is supporting code, not a fourth modeling layer.

## What the repository actually contains

Inspected source: `production/wizardofodds-integration` at `a0aed68db0ff1ac6176a319f49dc646d75c8e914`. The default `main` branch at inspection was `69116137b1020c7cdf496fb63e7b247cfd2d3261`; its README describes an older snapshot. This distinction matters when identifying “your model.”

The production branch has 48 Python files and 24,179 physical lines under `src/nba_prop_quant` before this addition. Those lines include substantial whitespace and extensively split expressions. They are not all prediction mathematics.

| Part of the repository | Actual responsibility | Treatment here |
| --- | --- | --- |
| `normalize`, `game_context`, `features`, `slate` | Clean tables, identify games, construct lagged inputs and future rows | Consolidated in `data.py` |
| `decay`, `kalman`, `availability` | Update player priors and redistribute opportunity after absences | Consolidated in `data.py` |
| `experience`, `model` | Experience curves, minutes model, target models, chronological fitting helpers | Consolidated in `model.py` |
| `distributions`, `copula` | Fit count distributions and residual dependence | Consolidated in `model.py` |
| `production`, `gate3_v2`, `pricing` | Select mean components, apply role adjustments, calculate and calibrate prices | Relevant calculations in `model.py` and `price.py` |
| `api`, `ingest`, `pipeline`, `settings`, `storage`, `prospective_snapshot`, `snapshot_schedule` | Data collection, storage, configuration, and timestamped capture | Existing operational implementation retained; study inputs are supplied tables |
| `adaptive_training`, `adaptive_fit_registry`, `adaptive_validation` | Refit, validate, register, and promote fitted parameters | Existing implementation retained; full adaptive orchestration is not rewritten here |
| `ctmc`, `live`, `sequence`, `interpret` | Additional live/sequence/explanation capabilities | Outside this pregame extraction |
| `src/nba_prop_quant/research/*`, `research/*` | Experimental models, evaluation, and historical evidence | Retained separately; not silently made production behavior |
| `scripts`, `ops`, workflows, tests, manifests, docs | Entry points, automation, grading, delivery, and verification | Retained unchanged |

I traced the two serving entry points and their imports. This is an architecture and core-calculation dissection, **not a completed line-by-line audit of every research and operations file**.

## The calculation, in order

1. **Build information known before the game.** Completed box scores give recent per-minute rates, minutes history, team/opponent context, rest, experience, and dynamic priors. Training uses lagged values. Future-slate construction refuses history dated on or after the slate.
2. **Predict minutes.** XGBoost predicts minutes. Current injury information changes available minutes and usage. The role model later estimates a minutes adjustment from lineup changes.
3. **Predict six means.** Targets are points, rebounds, assists, steals, blocks, and made threes. XGBoost predicts the target count directly with expected minutes among its inputs. Rebounds and blocks blend that prediction with decay and Kalman rate-times-minutes estimates. The other four use the selected XGBoost route.
4. **Build six distributions.** The selected family is mean-preserving zero-inflated negative binomial. If structural-zero probability is `pi`, the count-component mean is `mu / (1 - pi)`. The mixture therefore still has mean `mu`.
5. **Price combinations.** P+R, P+A, and P+R+A use independent PMF convolution under the selected policy. R+A uses Gaussian-copula simulation with dependence strength 0.85. The production path does **not** price every combination from one common six-stat draw.
6. **Apply current role routing and calibration.** Assists uses its role-adjusted mean and selected role calibration. P+A and P+R receive role increments on the logit scale. Other markets retain their selected routes. Blocks and steals use raw probability calibration routing.
7. **Calculate prices and comparisons.** Keep over, under, and push probabilities distinct. Calibrate the over probability conditional on no push; retain push mass. Derive fair odds, compare with de-vig market probabilities, and calculate EV. No automatic bet threshold is enabled.

A useful distinction: “mean XGBoost uses a Poisson loss” does not mean “the final player-stat distribution is Poisson.” The final selected distribution is ZINB.

Another distinction: market-level calibration and separate combination policies do not establish one globally coherent joint distribution for all displayed markets. This extraction preserves those existing decisions; it does not claim to repair or redesign them.

## Where fitting lives

The original daily fitting sequence is input snapshot → features → minutes → targets → ensemble weights → marginals → dependence → calibration → role model → validation and registration.

This folder includes the existing core fit functions for minutes, targets, experience, dynamic priors, marginals, and copula. It does **not** include a replacement end-to-end trainer, ensemble/calibration selection scripts, or adaptive promotion system. The production daily refit locks model choices while updating permitted fitted values. A study rewrite must not accidentally turn that into daily model selection.

## First lesson: about five minutes

From the repository root, in the Python environment you already use for this project:

```bash
python -m study.check lesson
```

Expected output includes:

```text
SYNTHETIC LEARNING EXAMPLE; not a player forecast
Mean = 24.0, line = 24.5
Over = 0.448272, under = 0.551728, push = 0.000000
Fair over odds = +123
```

Open `check.py` and find `lesson()`. Concentrate on these four lines:

```python
over, under, push = marginal.over_under_push(line, mean, row)
q_over = over / (over + under)
fair_price = price.fair_american(q_over)
print(fair_price)
```

- The first line calls a function and assigns its three returned values to three variables.
- The second line removes pushes from the probability used to price a bet that refunds pushes.
- The third line calls another function and stores the fair American odds.
- The fourth line displays the answer.

**Your change:** in `lesson()`, change `line = 24.5` to `line = 24.0`. Run the command again. Push probability should become positive because an integer-valued score can equal 24. Then restore 24.5.

That is a real coding exercise: change an input, predict the effect, run the code, explain the result.

## Subsequent short lessons

| Lesson | Change or inspect | What you learn |
| --- | --- | --- |
| 2 | Read `fair_american`; calculate its answer for probability 0.60 | Functions, arguments, `if`, `return` |
| 3 | Read `combine_mean_components`; use weights 0.5, 0.3, 0.2 | Dictionaries, arrays, weighted sums |
| 4 | Read `_lagged_ewm`; explain why `shift(1)` comes first | DataFrame columns, grouping, avoiding future outcomes |
| 5 | Read `ModelBundle.predict` | How feature names connect your data to a fitted estimator |
| 6 | Read ZINB `_count_component_mean` and `implied_mean` | Translating a distribution identity into code |
| 7 | Read `independent_combo_pmf` and `simulate_combo_values` | Why independent convolution and dependent simulation differ |
| 8 | Read `project_slate`, then `price_markets` | Following the complete offline inference path |

Only move to the next lesson once you can explain the current function in your own words.

## Run the comparisons

```bash
python -m study.check
```

The checks compare copied function/class syntax against the original source, compare numerical outputs on synthetic inputs, and run a synthetic slate through all ten supported markets. They also exercise missing-role rejection, chronology checks, mean preservation, probability mass, calibration, and push handling.

The extraction keeps source references in comments. It has **no original-package imports in its inference path**. Its exporter and comparison helper intentionally load the original package to translate old serialized classes and check parity.

## Use your fitted artifacts

In the original runtime's matching Python/library environment, after verifying and extracting a trusted release:

```bash
PYTHONPATH=src python -m study.check export /path/to/extracted/runtime
```

This writes a separate, ignored `study/artifacts` folder. Original model files are read, not overwritten. Export finishes atomically; a load failure leaves no partial artifact directory.

Then call the two explicit entry points from Python:

```python
from pathlib import Path
from study.price import project_slate, price_markets

projections = project_slate(
    history, games, players, advanced, injuries, lineups,
    artifacts=Path("study/artifacts"), snapshots=Path("your/prior/snapshots"),
)
priced, rejected = price_markets(projections, quotes, Path("study/artifacts"))
```

The inputs are the same normalized DataFrame formats expected by the original code. `history` and `advanced` must precede the slate. `games`, `players`, `injuries`, `lineups`, and `quotes` must represent the same pregame information set. This study runner does not collect or certify that information set. It labels outputs `study_only` and does not write production feeds.

## Validation status and deliberate differences

- Core checks: **106 unchanged function/class syntax comparisons and 62 numerical/invariant checks passed** in the development environment.
- Both study entry points ran successfully using small synthetic fitted models. All ten markets were priced, and the three role-dependent markets were rejected when role information was marked unavailable.
- Runtime download SHA-256 matched the published digest: `0e81bf94e0620743d5ee93cd0bf50bf36f84e79c32601349140ddc8f4223c419`.
- Full published-runtime replay is **not certified**. The available environment differs from the release environment. In particular, the experience-curve pickle cannot load because its SciPy `_BSpline` class is unavailable. XGBoost and scikit-learn also report version differences. No compatibility shim was used to suppress that problem.
- One numerical correction is explicit: `_pmf_over_under_push` now returns under=1 when a line is above the represented PMF support. The original code could incorrectly return over=1 at high integer lines. A regression check covers integer and half-integer lines. This correction applies only to the study copy.
- The study runner omits production capture certification, manifest/source locks, manual minutes overrides, presentation summaries, feeds, grading, scheduler, and adaptive registration. Unmatched quotes are reported as rejected instead of disappearing in an inner join.
- No accuracy improvement, market superiority, or complete production equivalence is claimed. Full production replacement would still require the remaining operational/training rewrite and an end-to-end comparison in the matching runtime.

The goal is code you can explain and modify. Three understandable files are useful; an arbitrary promise to fit the entire production platform into three tiny scripts would not be honest.
