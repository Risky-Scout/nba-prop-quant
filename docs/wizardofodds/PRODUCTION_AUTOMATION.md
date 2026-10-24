# Production Automation

Production Step 3D. GitHub Actions orchestration of the NBA Prop Quant daily
production lifecycle.

Workflows: `.github/workflows/nba_production_lifecycle.yml` and
`.github/workflows/ci.yml`. Supporting tools:
`ops/production_lifecycle_preflight.py`, `ops/validate_workflows.py`,
`ops/classify_refresh_outcome.py`, `ops/summarise_production_run.py`.

---

## 1. GitHub orchestrates, Python decides

The workflow sequences calls and reports what happened. Every rule about
semantic state, training cutoff, information-set eligibility, adaptive action
selection, historical corpus selection, candidate construction, validation,
registration and promotion eligibility stays in
`ops/run_adaptive_daily_fit.py` and `nba_prop_quant.adaptive_training`.

No YAML conditional decides whether to retrain. If the workflow decided that,
the policy would exist in two places and they would drift. A test asserts the
workflow contains no modelling call and no retrain conditional.

---

## 2. The default branch is not the production branch

The repository default branch is **`main`**. Production code lives on
**`production/wizardofodds-integration`**.

This matters because GitHub only fires `schedule` events from the default
branch. A workflow that simply checks out `github.ref` on a scheduled run
would fit and promote using whatever `main` contains.

So the production workflow always names the authoritative ref explicitly:

```yaml
ref: ${{ github.event_name == 'schedule'
         && 'production/wizardofodds-integration'
         || (inputs.production_ref || 'production/wizardofodds-integration') }}
```

and then **proves** the checkout rather than trusting it. The preflight
re-resolves `HEAD` and refuses to continue unless it matches the expected
production ref, so a misconfigured checkout fails before it can touch
anything.

### Required configuration to activate the schedule

This PR places the workflow on the production branch. Because GitHub schedules
only from the default branch, the schedule does not fire until
`.github/workflows/nba_production_lifecycle.yml` **also exists on `main`**.

The file is written to be safe to run from either branch: it takes no
production behaviour from the branch it was invoked on. Copying it to `main`
verbatim is sufficient, and it will still check out and verify the production
ref.

The same constraint applies to `workflow_dispatch`: GitHub only offers the
manual trigger for workflows present on the default branch.

---

## 3. Triggers

| Trigger | Purpose |
| --- | --- |
| `workflow_dispatch` | Controlled manual execution. `mode` selects `validate-only` or `production`; `slate_date` and `production_ref` may be overridden. |
| `schedule` | `37 9 * * *` UTC, daily. |

The repository documented no prior schedule policy, so this is the minimum
reliable daily trigger. **09:37 UTC** is deliberate: off the top of the hour,
where GitHub's scheduler is most congested and runs are most often delayed or
dropped outright; and 05:37 US/Eastern in daylight time (04:37 in standard
time), which is after every North American box score has settled and well
before that evening's T-20 capture window.

`validate-only` proves the wiring end to end without production credentials or
a state backend. A scheduled run is always `production`.

---

## 4. Single writer

```yaml
concurrency:
  group: nba-production-lifecycle
  cancel-in-progress: false
```

A fixed group, so a scheduled run and a manual run queue against each other
rather than overlapping. Queued rather than cancelled on purpose: cancelling
mid-flight could interrupt a registry write or leave a refresh transaction
prepared but unresolved.

GitHub concurrency **supplements** the repository's own locks; it does not
replace them. The adaptive trainer takes an exclusive `flock` under the work
root and the refresh takes its own writer lock, and those remain the real
guarantee — GitHub concurrency does not protect against a run started outside
Actions.

---

## 5. Permissions, environment, secrets

`permissions: contents: read` on both workflows. Nothing in the lifecycle
writes to the repository.

The production job runs in the `wizardofodds-production` GitHub Environment,
which already exists and whose protection rules are preserved. Both the
credentials and the durable paths are scoped to that environment, so they do
not resolve for any job outside it.

| Name | Kind | Required | Source |
| --- | --- | --- | --- |
| `BDL_API_KEY` | secret | yes | `Settings.bdl_api_key`, used by the data refresh |
| `ODDS_API_KEY` | secret | optional | market pricing, when live quotes are used |
| `NBA_PROP_DATA_DIR` | variable | yes | rolling data root |
| `NBA_PROP_FIT_REGISTRY_DIR` | variable | yes | fit registry root |
| `NBA_PROP_WORK_DIR` | variable | yes | trainer work root and lock location |

Credentials come only from `${{ secrets.* }}` and paths only from
`${{ vars.* }}`. The split is deliberate: variable values are readable by
anyone who can read the repository settings, so a credential must never be one.
Tests assert both directions — that no path is hard-coded into the YAML and
that no credential is ever referenced as a variable.

Names are derived from production code, not invented. The preflight checks
presence **by name only**: it never reads, logs, returns or renders a value, a
test asserts a planted value cannot appear in its output, and the workflow
validator fails if a secret is ever interpolated into a command that writes to
the log.

If a required secret is missing, the preflight fails before any step that
could mutate state.

---

## 5a. Runner placement

```yaml
runs-on:
  - self-hosted
  - nba-production
```

Only the mutating production lifecycle job runs there. `ci.yml` — the reported
check for pull requests — stays on `ubuntu-latest`, and the production
workflow has no `pull_request` or `pull_request_target` trigger at all.

That separation is the point. The `nba-production` runner holds the full
2001→present data root, the fit registry and the promotion pointer. Code
arriving on a pull request is untrusted by definition, so it must never
execute on the machine that owns production state. The workflow validator
enforces both halves: the production job is rejected if it does not require
both labels, and any workflow reachable from a pull-request trigger is
rejected if any of its jobs is self-hosted.

---

## 6. Daily sequence

1. check out the authoritative production ref
2. install the supported Python environment
3. **preflight**: prove the checkout, verify the frozen contracts, verify the
   installed scikit-learn matches the frozen artifacts, check secrets by name,
   require a durable state backend
4. refresh the current-season rolling state (`ops/refresh_current_season_state.py`)
   and classify its exit code (`ops/classify_refresh_outcome.py`)
5. run the adaptive daily protocol (`ops/run_adaptive_daily_fit.py`)
6. install the verified frozen runtime bundle
   (`ops/install_frozen_model_artifacts.py`)
7. serve the slate with the incumbent
   (`ops/run_incumbent_production_serving.py`) — predictions, pricing and the
   serving receipt
8. **publish the served slate to the WizardOfOdds feed**
   (`ops/run_wizardofodds_publication.py --stage feed`)
9. **build and publish the WizardOfOdds player-props page from that exact
   release** (`ops/run_wizardofodds_publication.py --stage page`)
10. grade the incumbent's previous slate
    (`ops/grade_incumbent_production_slate.py`)
11. shadow the slate beside production (`ops/run_production_shadow.py`,
    non-blocking) and judge it (`ops/monitor_live_shadow_evidence.py`,
    `ops/evaluate_shadow_health.py`)
12. summarise the run and upload diagnostics

Steps 4 and 5 are gated behind the preflight and behind `production` mode, so
a failed preflight cannot leave a partial mutation. Step 5 is additionally
gated on `RUN_ADAPTIVE`, which step 4 decides in Python — see §7a.

Steps 8 and 9 are in that order and nowhere else in the file, which is §13's
whole subject: the page is built from the feed release published moments
earlier in the same run, so the two can never disagree about what the slate
said.

---

## 7. A no-op is a success

If the adaptive policy reports `NO_NEW_TRAINING_DATA` — an NBA off day, or a
day whose completed information has not changed — the run is a **success**:

- the incumbent is not overwritten
- no model version is manufactured
- the job does not fail
- the summary says so explicitly

The policy decides this, not the workflow. Turning every daily run into a full
retrain would be both wasteful and wrong.

---

## 7a. Preseason is a no-op, not a failure

`ops/refresh_current_season_state.py` returns the readiness preflight's own
exit code unchanged. Before NBA opening day the preflight reports
`Action: PRESEASON_BLOCK` and exits **20**: the slate date precedes the locked
opening day, so no 2026-27 regular-season box score can exist yet, nothing is
fetched and **no byte is written**.

Run [37060158442](https://github.com/Risky-Scout/nba-prop-quant/actions/runs/37060158442)
showed what that used to cost. On `2026-10-02`, 18 days before the locked
opening day `2026-10-20`, the refresh step exited 20, the step failed, the
adaptive stage never ran and the summary reported
`Failed at stage: adaptive_protocol` — a red run describing a day on which the
system had correctly done nothing.

`ops/classify_refresh_outcome.py` resolves that. It is the single place that
maps a refresh exit code onto the lifecycle decision, and it admits exactly
four answers:

| Refresh exit | Outcome | Lifecycle | Adaptive daily fit | Slate served |
| --- | --- | --- | --- | --- |
| `0` | `REFRESHED` | success | runs | yes |
| `11`, verified opening day | `OPENING_DAY_NO_REFRESH` | success | **skipped** | **yes** |
| `20`, verified preseason | `PRESEASON_BLOCK` | success | **skipped** | no |
| anything else | `REFRESH_FAILED` | **fails closed**, re-raising the refresh code | skipped | no |

"Verified" is load-bearing. The classifier does not read the preflight's
human-readable banner and does not take either code on trust; it re-derives
the condition from the `OPENING_DAY_BY_SEASON` table the preflight itself
owns — slate date strictly *before* opening day for `20`, slate date exactly
*on* it for `11`. The opening day is **imported, never restated**, and so is
the opening-day exit code, so the two cannot drift and this module has no
opinion of its own about the schedule. An exit 20 on or after opening day, an
exit 11 on any other date, an unparseable slate date, or a season with no
locked opening day are all unverifiable, and every unverifiable case fails
closed.

Codes `10`, `30` and `31` — `SKIP_REFRESH`, `HOLD_STANDARD_NOT_READY`,
`HOLD_ADVANCED_NOT_READY` — are deliberately **not** widened into successes.
Only `0`, a verified `11` and a verified `20` may pass.

`11` exists because opening day is the one date where "nothing was refreshed"
and "a real slate exists" are both true: no 2026-27 game has been played, so
there is nothing to ingest and nothing newly completed to fit, while that
evening's games are real and the incumbent must price them. It previously left
the preflight as a `SKIP_REFRESH` `10`, indistinguishable from a day whose
prerequisite games had simply not settled, so the lifecycle failed closed and
the first slate of the season would never have been served.

The decision reaches the workflow as two variables written to `$GITHUB_ENV`,
`REFRESH_OUTCOME` and `RUN_ADAPTIVE`, and the adaptive step is gated on the
latter:

```yaml
if: env.MODE == 'production' && env.RUN_ADAPTIVE == 'true'
```

No YAML conditional inspects an exit code, names a date or special-cases `20`.
`ops/validate_workflows.py` enforces that shape statically: a workflow that
runs the adaptive daily fit without calling the classifier, or without gating
on `RUN_ADAPTIVE`, fails validation.

On a preseason day the run is therefore a **success**:

- the refresh writes nothing
- the adaptive daily fit does not run, so no candidate is fitted or registered
- the incumbent `current_good_fit_id` is untouched
- the summary reports `PRESEASON_BLOCK` and says why it is a success

On opening day the run is a **success that continues**: the same four facts
hold, the summary reports `OPENING_DAY_NO_REFRESH`, and serving proceeds.
`ops/run_incumbent_production_serving.py` admits that outcome only against
the classifier's recorded derivation — `verified`, `writes_performed`,
`season`, `slate_date` and `opening_day` — and then re-derives the opening day
from the same locked table before serving. A status file that names the
outcome without the evidence, contradicts itself, disagrees with the slate
being served, or asserts an opening day the schedule never had is refused
rather than served.

Nothing about this fabricates a slate, moves opening day, or lets the model
train on information that does not exist.

---

## 8. Fail closed

Any failure in refresh, fitting, validation, registration or promotion leaves
the current good fit exactly where it was. The two no-ops in §7a — the
preseason block and the opening-day no-refresh — are the only refresh exit
codes other than `0` that are allowed to report success, and each is allowed
only after its own condition has been re-verified from the preflight's own
locked opening day. The registry's promotion state is
only replaced by a single atomic write after every check passes, candidates
are registered immutably and separately from promotion, and Step 3C records no
promotion call at all.

The workflow adds no force-repair behaviour. A failed run returns a failing
job, retains its diagnostics artifact, and names the failed stage in the
summary.

---

## 9. Observability

Every run writes a job summary containing the production code SHA, trigger,
mode, slate date, training cutoff, preflight result, refresh outcome, adaptive
action, candidate fit id, promotion outcome, current good fit id, validation
counts, runtime and failure stage, plus any blockers. The same content is written as machine-
readable JSON and uploaded as a run artifact with a 14-day retention.

No DataFrame is rendered into the log.

---

## 10. Durable production state — resolved

**`DURABLE_PRODUCTION_STATE_BACKEND_REQUIRED`** — resolved by the self-hosted
`nba-production` runner plus the three environment variables in §5. The
preflight still enforces it and still fails closed in `production` mode when
no backend is configured; what changed is that a backend now exists.

No new persistence implementation was written. The runner's filesystem is the
backend, the three configured paths address it, and the existing `flock`
locking keeps its original meaning because every writer is once again on one
machine.

### What needs to persist

| State | Written by | Why it must outlive the run |
| --- | --- | --- |
| Fit registry `fits/<fit_id>/` | Step 3A | Immutable historical evidence; a promoted fit must still exist tomorrow |
| `state/promotion_state.json` | Step 3A | Holds `current_good_fit_id` and `previous_good_fit_id` — the production pointer and the rollback target |
| `state/validations/` | Step 3A | Validation records gating promotion |
| Rolling data root `data/raw/seasons/`, `data/raw/advanced/` | Step 3B | 2001→present box scores and 2015→present advanced; the refresh is incremental |
| `.state/current_season_state.json` | Step 3B | Semantic authentication of the latest rolling generation |
| `.refresh_transactions/` | Step 3B | Crash-recovery journal |
| Snapshots `data/snapshots/` | Gate 3 | T-20 capture lineage |
| Generated projections and priced markets | Steps 10/15 | The artifact the publishing step consumes |

### What the repository currently expects

Both roots are **explicit and external by design**. The registry root has no
default and is refused if it resolves inside the repository:
"fits are production state and must live outside version control". The data
root comes from `NBA_PROP_DATA_DIR`. `.gitignore` excludes `data/raw/`,
`data/processed/`, every `*.parquet` and every model binary, so committing
state back to git is not the intended mechanism and is not possible without
reversing that decision.

`docs/GITHUB_DISTRIBUTION.md` covers the opposite direction: shipping the
frozen model package out via GitHub Releases. It is one-way deployment of an
immutable artifact, not a read-write daily state backend.

### Why a GitHub-hosted runner is insufficient

The runner and its filesystem are destroyed when the job ends. With no durable
backend:

- `current_good_fit_id` resets to null every night, so last-good and rollback
  become meaningless and every run looks like the first;
- Step 3C's no-new-data detection finds no parent fit, so every run performs a
  full historical retrain instead of the off-day no-op the policy intends;
- the refresh's non-regression guard compares staged counts against the
  **live** tree, and with nothing live it protects nothing;
- the full 2001→present corpus would have to be re-ingested from BALLDONTLIE
  daily, which is infeasible and a large avoidable API cost.

None of these fail loudly. That is precisely why the preflight refuses to
proceed rather than letting the lifecycle appear to work.

### What was configured

Option 1, the self-hosted runner on the machine that already holds the full
NBA data root. The paths are local, no data moves, and the three environment
variables — set as environment **variables** (`vars`), not secrets, because
they are paths and not credentials — address them:

```text
NBA_PROP_DATA_DIR
NBA_PROP_FIT_REGISTRY_DIR
NBA_PROP_WORK_DIR
```

No cloud SDK, no schema change, no new persistence backend. The preflight's
durability rules are unchanged and still reject a path that resolves inside
the repository or inside `RUNNER_TEMP`, so a misconfigured variable still
fails closed rather than quietly resetting production every night.

---

## 10a. Frozen-artifact dependency contract

The frozen Gate 3 role model is a pickled scikit-learn estimator carrying no
version of its own. Unpickling it under a different scikit-learn is not a
guaranteed round trip: at best it warns, at worst it reconstructs different
internals and produces different minutes. The first automated run installed
**1.9.1** against artifacts built under **1.9.0**.

So `pyproject.toml` pins `scikit-learn==1.9.0` — a pin, not a floor — and the
preflight re-checks it at run time in `production` mode, failing with
`INCOMPATIBLE_SCIKIT_LEARN_VERSION` and exit code 7 before any step that can
mutate state.

The required version is **read from the freeze manifest**, not written into
the preflight: `models/frozen_manifests/LATEST.json` records the environment
the artifacts were produced in, including the `pip_freeze` that names
`scikit-learn==1.9.0`. Deriving it rather than restating it means the
requirement cannot drift from the artifacts it describes, and a future refreeze
updates both at once. A test asserts the `pyproject.toml` pin and the manifest
still agree.

No other package was repinned.

---

## 11. Production prediction artifact

Step 3D ends each successful lifecycle with the existing production output,
generated by the existing pricing code against the current good fit. It
introduces no second public schema: the fields, provenance and version
identifiers remain those documented in `docs/WIZARDOFODDS_INTEGRATION.md`.

The artifact itself contacts nothing. Deployment belongs to the two
publication stages in §13, which read it and never recompute it.

---

## 12. CI check

`.github/workflows/ci.yml` gives production pull requests a reported pass/fail
status, which PR #9 lacked. It installs the supported environment, validates
the workflow definitions statically, runs the full test suite and runs the
preflight in `validate-only` mode. It grants `contents: read`, weakens no
protection rule, and trains and promotes nothing.

---

## 13. WizardOfOdds publication

Everything above produced a priced slate on one machine. These two steps are
what turn it into a public page without anybody logging in.

### Two stages, one entry point

`ops/run_wizardofodds_publication.py` runs twice per production day:

```yaml
- name: Publish the served slate to the WizardOfOdds feed
  run: python ops/run_wizardofodds_publication.py --stage feed --mode canary …

- name: Build and publish the WizardOfOdds player-props page
  run: python ops/run_wizardofodds_publication.py --stage page --mode canary …
```

The feed stage reads the serving receipt, publishes the release through
`ops/publish_incumbent_to_wizardofodds.py`, reads the promoted release back
over the transport and refuses unless it is a non-empty
`publication_contract_version = 2` release of the slate it was asked for. The
page stage reads that stage's receipt, pins the release id *and* the rows
digest from it, and hands both to
`ops/wizardofodds_player_props_page.py`, which refuses at the current pointer
if the id differs and at the manifest if the digest differs.

**The page cannot be built from a stale release**, and not because anything
compares the two afterwards. The pin is checked *before* the rows are read, so
a page is never rendered from the wrong bytes and then caught — it is never
rendered at all. The remote NBA feed canary spent a long time serving a
contract-1 release, which is exactly the shape of mistake this closes.

### Why the ordering is expressed in Python

GitHub's step order and fail-fast already stop the page stage from running
after a failed feed stage. That is necessary and not sufficient: a workflow
can be edited, and `continue-on-error` is one word. So the dependency is also
a file. `--feed-receipt` names the receipt the feed stage wrote, and the page
stage has no other way to learn what to build, so a page stage that ran out
of order would have nothing to read rather than something stale to read.

`ops/validate_workflows.py` holds the shape statically: a workflow that
publishes must invoke the serving entry point, must run the feed stage after
it and the page stage after that, must not mark either stage
`continue-on-error`, must pass `--feed-receipt` to the page stage, and must
name no publishing activation token.

### Fail closed

| if | then |
| --- | --- |
| serving reports no games on the slate | both stages record it and publish nothing — a no-op is a success, as in §7 |
| the feed publication fails or is refused | the job fails; the page stage does not run |
| the published feed is not contract version 2 | the job fails; **no** fallback to the previous contract-1 release |
| the published feed has no rows | the job fails |
| the page build fails | nothing is published |
| a page readback digest disagrees | the publication job fails |

No stage reuses an earlier artifact to make the run green.

### Idempotency

Rerunning a day republishes the same release rather than accumulating a new
one. A release id is derived from the digest of its own rows, so identical
rows are the same release; promoting the release that is already current is a
no-op, which preserves both `promoted_at` and the rollback target; and the
page renderer reads no clock, so the same release renders the same bytes.
Retention prunes only after the pointers have moved, and never below the two
releases the rollback target needs.

### Live is configured and disabled

Both stages accept `--mode live`, and both live destinations are configured by
name — see `docs/wizardofodds/INCUMBENT_PUBLISHER.md`. Neither can be reached.
`modes.live.state` is `DISABLED` in the committed switch, the switch is
consulted before any destination is resolved, and activation additionally
requires `NBA_PROP_WOO_PUBLISH` and `NBA_PROP_WOO_PUBLISH_APPROVAL` in the
environment. The lifecycle names none of the three.

That is the division this section exists to draw: **the destination is
engineering and the activation is a decision.** A reviewed activation sets the
switch and supplies the two variables. It writes no code, configures no
destination, copies no file by hand and rebuilds no page.

## 14. Early advisory windows

T-20 is twenty minutes before tip. That is the window production is *required*
to have, because it is the market the slate is graded against, but it is late
to read a prediction from. The same invocation therefore also serves the
T-180, T-90 and T-45 capture windows as **advisory** releases.

What a reader actually gets, in the wall clock they read it in:

| Tip (America/New_York) | T-180 | T-90 | T-45 | T-20 |
| --- | --- | --- | --- | --- |
| 7:00 PM | 4:00 PM | 5:30 PM | 6:15 PM | 6:40 PM |
| 9:30 PM | 6:30 PM | 8:00 PM | 8:45 PM | 9:10 PM |
| 10:00 PM | 7:00 PM | 8:30 PM | 9:15 PM | 9:40 PM |

### Nothing about the model changes

An advisory release is the same frozen incumbent, the same fitted artifacts,
the same marginals, copula, calibration, pricing and Gate 3 routing, priced
from a different point-in-time market snapshot. The only argument that differs
is `--snapshot-offset-minutes`, which both `10_predict_slate.py` and
`15_price_markets.py` already accepted; `ops/run_advisory_window_serving.py`
imports `ops/run_incumbent_production_serving.py` and calls that module's own
`resolve_incumbent`, `serve` and `build_receipt`, so the serving orchestrator
is unchanged byte for byte and there is no second definition of what the
incumbent is.

### Separate destinations, separate pointer

Six named destinations, three for the feed and three for the page, one pair
per offset:

| offset | feed destination | page destination | confinement |
| --- | --- | --- | --- |
| T-180 | `wizardofodds_nba_advisory_t180m_canary` | `wizardofodds_nba_player_props_page_advisory_t180m_canary` | `canary-advisory/t180m` |
| T-90 | `wizardofodds_nba_advisory_t90m_canary` | `wizardofodds_nba_player_props_page_advisory_t90m_canary` | `canary-advisory/t90m` |
| T-45 | `wizardofodds_nba_advisory_t45m_canary` | `wizardofodds_nba_player_props_page_advisory_t45m_canary` | `canary-advisory/t45m` |

Each is `mode: canary`. None is live, and live remains disabled for all of
them by the same switch described above.

The confinement is what makes contamination structural rather than
procedural. `RemoteDestination.published_root` is `public_target` joined to
the confinement, so the T-20 feed's served root is
`.../predictions/nba/canary` and T-180's is
`.../predictions/nba/canary-advisory/t180m`. Neither path is a prefix of the
other. Every pointer the publisher writes -- `current.json`, `previous.json`,
`canary_attestation.json`, `audit_log.jsonl` -- is written *inside* the served
root, so an advisory publication does not decline to repoint the T-20
`current.json`; it has no filesystem path by which it could reach it. The same
holds for the page: `.../NBA/player-props/canary` against
`.../NBA/player-props/canary-advisory/t180m`.

### Labelled with the offset it was captured at

Every advisory destination carries its own `release_prefix`, and a release id
is `{release_prefix}__{slate_date}__{rows_sha256[:16]}`. So the offset is in
the release id itself, which means it is in the feed manifest, in
`current.json`, in the page manifest's `released.release_id`, in the page's
rendered "Release" fact row, and in the public URL:

```
nba_prop_quant_advisory_t180m__2026-10-23__<digest>
https://.../predictions/nba/canary-advisory/t180m/
```

The page prefix carries it too (`page-preview-advisory-t180m`). The advisory
serving receipt adds three fields of its own -- `advisory: true`,
`advisory_capture_offset_minutes`, `advisory_capture_window_ids` -- additively,
so nothing a verifier already reads moves.

The one thing not labelled is the rows document's own bytes. `PUBLISHED_FIELDS`
in `ops/publish_incumbent_to_wizardofodds.py` is pinned by AST comparison
against the production ref, so a row cannot gain a field; the offset is
carried by the release id, the manifest and the URL instead.

### Three refusals, all additive

1. `assert_offset_is_advisory` in the advisory entry point refuses an offset
   that is not in `CAPTURE_OFFSETS_MINUTES`, refuses the production offset of
   20 specifically, and refuses any offset outside `ADVISORY_OFFSETS_MINUTES`.
2. `assert_window_ids_are_this_offset` refuses any window id ending `:T-20m`
   before it refuses an id belonging to some other offset, so "an advisory run
   wrote a T-20 window id" is its own named refusal rather than a side effect.
3. `morning_readiness` in the controller refuses a serving receipt carrying
   `advisory: true`. The durable receipt at `processed/incumbent_serving/` is
   what proves a verified incumbent for the slate; an advisory receipt is
   written to `processed/advisory_window_serving/` and the entry point refuses
   a `--receipt-path` equal to or inside the production record's directory,
   but if one ever arrived there anyway it would be refused on the way in
   rather than believed.

### The T-20 window keeps priority

The advisory path runs in the gaps of the loop that already exists, and the
T-20 loop itself is unchanged apart from the one call that fills a gap before
its sleep. Two rules keep it that way:

- **A reservation.** No advisory serve *starts* within
  `T20_RESERVATION_MINUTES` (10) of a T-20 capture the segment still owns, and
  the reservation is re-checked at every stage boundary rather than only
  before the first, so a stage that runs long costs one stage of exposure
  instead of the whole sequence. Ten minutes is roughly five times the 2m09s
  that run `40033448558`'s 23:40Z group took to go from capture to published
  page for a seven-game, 2,075-row slate, and deliberately under the
  twenty-five minutes between T-45 and T-20 so that T-45 is not reliably
  deferred past its own game's required window.
- **Capture and serve are separated.** The capture is the time-critical half
  and the cheap half; the serve is expensive and not time-critical, because
  the capture *is* the point-in-time market. A T-180 capture served an hour
  late still prices the market as it stood three hours before tip. So advisory
  captures are always taken on the clock and advisory serving is drained when
  there is room, coalesced per offset the way the T-20 path coalesces its own
  cumulative re-serving.

An advisory failure degrades the segment to
`T20_SEGMENT_COMPLETE_ADVISORY_DEGRADED`, which is a success outcome. The
asymmetry is deliberate: a T-20 time no segment covers fails the run, because
production is required to have that window, while an advisory time no segment
covers is recorded and the segment carries on.

### Why no fourth cron and no second workflow

The three segment crons are pinned contiguous by `ops/validate_workflows.py`
so that no capture window is owned twice, and every advisory time already
falls inside one of them. The decisive constraint is the runner: it executes
one job at a time, and the 19:25Z invocation holds it from 19:25Z until its
last publication, which covers nearly the whole 20:00Z-02:15Z span the
advisory windows live in. A separate workflow with its own concurrency group
would only move the queue from GitHub's gate to the runner's and start each
advisory run after the window it was for had passed.

Handling the advisory windows inside the invocation that already owns the
segment also *shortens* the longest contiguous sleep: the 19:25Z invocation's
three-and-a-quarter-hour wait for its first T-20 window becomes several short
waits. That is the failure mode recorded in the next section, which has twice
cost this schedule a whole evening to a job dropped while idling.

`nba_t20_production.yml` is unchanged by all of this, and deliberately. The
offsets are a module constant in the controller rather than a workflow input,
because a scheduled event fires from the default branch: a scheduled run
executes `main`'s copy of the workflow against the production branch's copy of
the controller, so an opt-in expressed in YAML would never be passed. The
probe already reports `t20_work=true` for a segment whose only work is
advisory, so the existing install gate covers it. The escape hatch is
`--no-advisory-windows` on the controller, for a manual invocation.

One consequence worth knowing: the 14:25Z segment used to be a cheap no-op on
almost every slate and now does work whenever a T-180 window falls inside it,
which on a slate with an early tip it does. That segment therefore installs the
production package where it previously exited before doing so.

## Segment cancellations, 2026-10-22 and 2026-10-23

Two evenings were lost to a cancellation rather than to any gate, and the
shape is identical both times: readiness passed, the controller entered its
sleep, and the job's capture step was cancelled before the first T-20 window.

| Slate | Run | Step started | Cancelled | Window it never reached |
| --- | --- | --- | --- | --- |
| 2026-10-22 | `39840850425` | 19:31:56Z | 22:29:23Z | 22:40Z |
| 2026-10-23 | `40016015624` | 20:03:14Z | 21:31:59Z | 22:40Z |

In both cases the runner stayed healthy -- the post-job steps completed
normally -- and in both cases the diagnostics upload then failed with
`403 Forbidden: job is completed`, so the run left no artifact either.

A cancelled segment has no automatic successor. The three scheduled segments
are half-open and non-overlapping by construction, so the windows the
cancelled segment owned belong to no other run, and the next scheduled
segment will not pick them up. Until that is addressed, recovering a
cancelled evening means getting a new job onto the production runner inside
the remaining window, and the only trigger available without write access to
the Actions API is a push to the production branch.
