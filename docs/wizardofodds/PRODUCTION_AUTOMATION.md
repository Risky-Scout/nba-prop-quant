# Production Automation

Production Step 3D. GitHub Actions orchestration of the NBA Prop Quant daily
production lifecycle.

Workflows: `.github/workflows/nba_production_lifecycle.yml` and
`.github/workflows/ci.yml`. Supporting tools:
`ops/production_lifecycle_preflight.py`, `ops/validate_workflows.py`,
`ops/summarise_production_run.py`.

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

## 5. Permissions and secrets

`permissions: contents: read` on both workflows. Nothing in the lifecycle
writes to the repository.

The production job runs in the `wizardofodds-production` GitHub Environment,
which already exists and whose protection rules are preserved.

| Secret | Required | Source |
| --- | --- | --- |
| `BDL_API_KEY` | yes | `Settings.bdl_api_key`, used by the data refresh |
| `ODDS_API_KEY` | optional | market pricing, when live quotes are used |

Names are derived from production code, not invented. The preflight checks
presence **by name only**: it never reads, logs, returns or renders a value, a
test asserts a planted value cannot appear in its output, and the workflow
validator fails if a secret is ever interpolated into a command that writes to
the log.

If a required secret is missing, the preflight fails before any step that
could mutate state.

---

## 6. Daily sequence

1. check out the authoritative production ref
2. install the supported Python environment
3. **preflight**: prove the checkout, verify the frozen contracts, check
   secrets by name, require a durable state backend
4. refresh the current-season rolling state (`ops/refresh_current_season_state.py`)
5. run the adaptive daily protocol (`ops/run_adaptive_daily_fit.py`)
6. summarise the run and upload diagnostics

Steps 4 and 5 are gated behind the preflight and behind `production` mode, so
a failed preflight cannot leave a partial mutation.

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

## 8. Fail closed

Any failure in refresh, fitting, validation, registration or promotion leaves
the current good fit exactly where it was. The registry's promotion state is
only replaced by a single atomic write after every check passes, candidates
are registered immutably and separately from promotion, and Step 3C records no
promotion call at all.

The workflow adds no force-repair behaviour. A failed run returns a failing
job, retains its diagnostics artifact, and names the failed stage in the
summary.

---

## 9. Observability

Every run writes a job summary containing the production code SHA, trigger,
mode, slate date, training cutoff, preflight result, adaptive action, candidate
fit id, promotion outcome, current good fit id, validation counts, runtime and
failure stage, plus any blockers. The same content is written as machine-
readable JSON and uploaded as a run artifact with a 14-day retention.

No DataFrame is rendered into the log.

---

## 10. Open blocker: durable production state

**`DURABLE_PRODUCTION_STATE_BACKEND_REQUIRED`**

The lifecycle cannot run unattended until production state survives between
runs. The preflight enforces this and fails closed in `production` mode when
no backend is configured.

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

### Smallest architecture-compatible implementation

Provide one durable POSIX-style location, reachable from the runner, holding
the data root, registry root and work root, and point the three environment
variables at it:

```text
NBA_PROP_DATA_DIR
NBA_PROP_FIT_REGISTRY_DIR
NBA_PROP_WORK_DIR
```

The code needs nothing more than paths that persist — no new cloud SDK, no
schema change and no redesign. Two options fit the existing architecture:

1. **Self-hosted runner** on the machine that already holds the full NBA data
   root. Smallest change: the paths are simply local, the existing `flock`
   locking keeps its meaning, and no data moves.
2. **Hosted runner plus a mounted durable volume or object store synced to
   those paths** before and after the job, preserving the same layout.

Option 1 is the smaller change and preserves every existing guarantee. The
choice is a deployment decision, not a code change, which is why this is
recorded as a Step 3D configuration blocker rather than a new project.

Set these as repository or environment **variables** (`vars`), not secrets;
they are paths, not credentials.

---

## 11. Production prediction artifact

Step 3D ends each successful lifecycle with the existing production output,
generated by the existing pricing code against the current good fit. It
introduces no second public schema: the fields, provenance and version
identifiers remain those documented in `docs/WIZARDOFODDS_INTEGRATION.md`.

Nothing here deploys to, publishes on, or contacts wizardofodds.com. That
belongs to the publishing step that follows Step 3D.

---

## 12. CI check

`.github/workflows/ci.yml` gives production pull requests a reported pass/fail
status, which PR #9 lacked. It installs the supported environment, validates
the workflow definitions statically, runs the full test suite and runs the
preflight in `validate-only` mode. It grants `contents: read`, weakens no
protection rule, and trains and promotes nothing.
