# Pre-opening-day operator actions

Three of the nine pre-opening-day production blockers cannot be closed from a
code change, because they are not in the repository. Two require
administrative access to infrastructure — the physical Mac that holds
production state, and the GitHub repository's own settings — and one requires
the full historical corpus, which lives on that Mac.

This document is the exact set of actions required, written so they can be
performed without further interpretation. Nothing here is a recommendation to
be weighed. Each item ends with the mechanical test that decides whether it is
done.

---

## 1. Self-hosted runner reliability — `BLOCKED_HUMAN_ACTION`

### What the evidence shows

Measured over the seven most recent scheduled lifecycle runs:

| run | created | started | wait |
| --- | --- | --- | --- |
| `37113947924` | 2026-10-03T09:42:19Z | 2026-10-03T10:02:09Z | 19m50s |
| `37197716819` | 2026-10-04T11:08:31Z | 2026-10-04T11:36:58Z | 28m27s |
| `37235586601` | 2026-10-04T21:18:50Z | 2026-10-04T21:18:53Z | 3s |
| `37293057379` | 2026-10-05T09:54:01Z | 2026-10-05T09:54:05Z | 4s |
| `37445390061` | 2026-10-06T09:49:20Z | 2026-10-06T10:17:02Z | 27m42s |
| `37603138394` | 2026-10-07T09:48:31Z | 2026-10-07T10:20:13Z | 31m42s |
| `37759297910` | 2026-10-08T09:49:20Z | 2026-10-08T15:36:22Z | **5h47m02s** |

Five of seven waited longer than five minutes. Two (`37445390061`,
`37603138394`) failed with *"The self-hosted runner lost communication with
the server"*. The runner is `Josephs-MacBook-Pro`, working directory
`/Users/josephshackelford/actions-runner-nba/_work`.

The five-hour run **concluded `success`**. That is why this was invisible: the
pass/fail column was green while the slate it was scheduled for was long
over. A runner that eventually answers is not a healthy runner.

The shape — long waits ending in a start, interleaved with lost-communication
drops — is a laptop asleep. A sleeping Mac's runner process is suspended, the
server loses its heartbeat, and the job queues until the lid opens.

### Why this cannot be fixed here

The runner is a physical machine. Nothing in this repository can install a
service on it, change its power management, or move its durable state. The
production state backend *is* that filesystem: `current_good_fit_id`, the
rollback target, the 2001-to-present rolling data root and the no-new-data
detection all live there and do not survive a hosted runner. Substituting a
GitHub-hosted runner is therefore not available as a fix — it would silently
reset all of them, which `PRODUCTION_AUTOMATION.md` §"Why a GitHub-hosted
runner is insufficient" already records.

### Actions, in order, on the runner Mac

Run each as the account that owns the runner
(`/Users/josephshackelford/actions-runner-nba`) unless it says `sudo`.

**1a. Confirm the runner is installed as a service rather than run by hand.**

```bash
cd ~/actions-runner-nba
ls ~/Library/LaunchAgents/actions.runner.*.plist
./svc.sh status
```

If no plist exists, the runner has been started interactively and dies with
the terminal session:

```bash
cd ~/actions-runner-nba
./svc.sh install
./svc.sh start
./svc.sh status
```

**1b. Make the service survive logout and reboot.**

`svc.sh install` creates a **LaunchAgent**, which only runs while the owning
user is logged in. The Mac must therefore log that user in automatically at
boot:

System Settings → Users & Groups → Automatic login → set to the runner's
account. (Requires FileVault to be off, or the Mac to be unlocked once after
each cold boot.)

Then confirm the agent is set to keep running:

```bash
plutil -p ~/Library/LaunchAgents/actions.runner.*.plist | grep -E 'KeepAlive|RunAtLoad'
```

Both must be present and true. If `KeepAlive` is absent, the service will not
restart after a crash.

**1c. Stop the machine sleeping. This is the actual cause.**

```bash
sudo pmset -a sleep 0
sudo pmset -a disksleep 0
sudo pmset -a powernap 0
sudo pmset -a standby 0
sudo pmset -a hibernatemode 0
sudo pmset -a autorestart 1
sudo pmset -a womp 1
sudo systemsetup -setcomputersleep Never
sudo systemsetup -setrestartpowerfailure on
```

On a laptop, also prevent sleep when the lid is closed while on mains power:

```bash
sudo pmset -c disablesleep 1
```

Leave `displaysleep` alone; the display sleeping is harmless.

Verify, and keep the output as evidence:

```bash
pmset -g custom
pmset -g assertions
```

`sleep` must read `0` for the AC profile and `PreventUserIdleSystemSleep`
must not be the only thing holding the machine awake.

**1d. Add a scheduled wake as a belt to that brace.**

The lifecycle cron is `37 9 * * *` UTC, which is 05:37 US/Eastern in daylight
time and 04:37 in standard time. Wake the machine before the earlier of the
two:

```bash
sudo pmset repeat wakeorpoweron MTWRFSU 04:20:00
pmset -g sched
```

**1e. Keep the network up.**

Prefer wired Ethernet. On Wi-Fi, disable the power-saving disconnect and
confirm the interface does not drop:

```bash
networksetup -listallhardwareports
sudo networksetup -setnetworkserviceenabled Wi-Fi on
```

If the Mac is on a network with a captive portal or a DHCP lease shorter than
the lifecycle's runtime, move it to a stable network. The lifecycle has a
360-minute timeout and a full fit is expected to use a significant part of it.

**1f. Fix the tool cache that broke `actions/setup-python`.**

Run `36845037743` failed at *Set up Python* with
`mkdir: /Users/runner: Permission denied`. `actions/setup-python` defaults its
tool cache to `/Users/runner/hostedtoolcache`, which does not exist on this
machine. Point it somewhere the runner owns, in the runner's own environment
file so every job inherits it:

```bash
cd ~/actions-runner-nba
printf 'AGENT_TOOLSDIRECTORY=%s/_work/_tool\n' "$PWD" >> .env
mkdir -p "$PWD/_work/_tool"
./svc.sh stop && ./svc.sh start
```

This is also the root of the interpreter-isolation defect closed in code:
when `setup-python` cannot use its tool cache it leaves the system
interpreter on `PATH`, and the install step then wrote into
`/Library/Frameworks/Python.framework/.../site-packages`. The workflow now
creates its own virtual environment under `RUNNER_TEMP` and
`ops/verify_production_interpreter.py` fails the job if the interpreter
resolves outside it, so a recurrence is a red job rather than a silent one.
Fixing `AGENT_TOOLSDIRECTORY` removes the cause as well as the symptom.

**1g. Confirm labels, version, disk and durable state.**

```bash
cd ~/actions-runner-nba
cat .runner                     # labels must include self-hosted and nba-production
cat .runner | grep -i agentName
ls -la _diag | tail -5          # most recent Runner_*.log
df -h "$PWD/_work"              # at least 50 GB free before a full fit
```

The durable production paths must exist and be writable by the runner
account. They are configured as GitHub repository variables; confirm each
resolves on this machine:

```bash
ls -la "$NBA_PROP_DATA_DIR" "$NBA_PROP_FIT_REGISTRY_DIR" "$NBA_PROP_WORK_DIR"
```

If the runner version is behind, let it auto-update or reconfigure it:

```bash
cd ~/actions-runner-nba
./run.sh --version
```

### How closure is decided

Not by inspection. Collect the next seven consecutive scheduled runs and let
the criterion decide:

```bash
REPO=risky-scout/nba-prop-quant

gh api "repos/$REPO/actions/workflows/nba_production_lifecycle.yml/runs?event=schedule&per_page=30" \
  --jq '[.workflow_runs[] | {id, created_at, run_started_at, conclusion, head_sha}]' > /tmp/runs.json

# then, per run id, collect the job-level start and the annotations:
#   gh api "repos/$REPO/actions/runs/<id>/jobs"
#   gh api "repos/$REPO/check-runs/<job id>/annotations"
# into the shape ops/audit_runner_reliability.py documents, and:

python ops/audit_runner_reliability.py \
  --observations /tmp/runner_observations.json \
  --report-path /tmp/runner_reliability.json
```

It exits `0` only on `PASS_PROVEN`, which requires all seven runs to have
started within five minutes of being created, with no lost-communication
annotation and no runner-offline failure. Anything less is
`IMPROVED_BUT_OBSERVATION_PENDING` or `FAIL`. The audited window above is
committed as a regression case in
`tests/test_runner_reliability_audit.py`, so that exact history can never be
reclassified as healthy.

---

## 2. Branch protection and required checks — `BLOCKED_HUMAN_ACTION`

### What the evidence shows

```
$ gh api repos/risky-scout/nba-prop-quant --jq '.permissions'
{"admin":false,"maintain":false,"pull":false,"push":false,"triage":false}

$ gh api repos/risky-scout/nba-prop-quant/branches/main/protection
HTTP 403: Resource not accessible by integration

$ gh api repos/risky-scout/nba-prop-quant/rulesets
[]

$ gh api repos/risky-scout/nba-prop-quant/branches --jq '.[] | "\(.name) protected=\(.protected)"'
main protected=false
production/wizardofodds-integration protected=false
```

Neither branch is protected. No ruleset exists. The available token has no
administrative scope, cannot read protection detail and cannot write it. This
portion stops here rather than being worked around.

### Order matters — do not invert it

Enabling a required check that does not run on a branch makes that branch
unmergeable. The remediation adds `Default branch guard`, which runs on pushes
to and pull requests against **both** branches, and `CI`, which runs on
production pushes and on pull requests into both branches. So:

1. Merge the remediation to `production/wizardofodds-integration`.
2. Sync the workflow files to `main`.
3. Confirm both checks have reported at least once on each branch.
4. Only then enable protection.

Step 3 is a real gate. Confirm with:

```bash
REPO=risky-scout/nba-prop-quant

for BRANCH in main production/wizardofodds-integration; do
  SHA="$(gh api "repos/$REPO/branches/$BRANCH" --jq .commit.sha)"
  echo "$BRANCH @ $SHA"
  gh api "repos/$REPO/commits/$SHA/check-runs" --jq '.check_runs[] | "  \(.name) \(.conclusion)"'
done
```

Both branches must list `Validate the default-branch scheduler role` with
conclusion `success` before the next step.

### The exact configuration

Via the UI: Settings → Branches → Add branch protection rule, once for
`main` and once for `production/wizardofodds-integration`:

- **Require a pull request before merging** — on
- **Require approvals** — 1
- **Dismiss stale pull request approvals when new commits are pushed** — on
- **Require status checks to pass before merging** — on
  - **Require branches to be up to date before merging** — on
  - Required checks: `Validate the default-branch scheduler role`
  - On `production/wizardofodds-integration` additionally:
    `Validate workflows and run tests`
- **Do not allow bypassing the above settings** — on (this is the setting that
  stops administrators silently bypassing)
- **Allow force pushes** — off
- **Allow deletions** — off

Via the API, as an account with admin on the repository:

```bash
REPO=risky-scout/nba-prop-quant

# main: the scheduler. Its check is the scheduler-role guard.
gh api -X PUT "repos/$REPO/branches/main/protection" \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["Validate the default-branch scheduler role"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false,
  "required_conversation_resolution": true
}
JSON

# production: the application lineage. Both checks are required here.
gh api -X PUT "repos/$REPO/branches/production%2Fwizardofodds-integration/protection" \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Validate the default-branch scheduler role",
      "Validate workflows and run tests"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false,
  "required_conversation_resolution": true
}
JSON
```

The check names above are the **job names**, which is what GitHub matches on.
They come from `name:` in `.github/workflows/default_branch_guard.yml` and
`.github/workflows/ci.yml`; if either job is renamed, the required context has
to be renamed with it or the branch becomes unmergeable.

### How closure is decided

```bash
REPO=risky-scout/nba-prop-quant

for BRANCH in main production%2Fwizardofodds-integration; do
  gh api "repos/$REPO/branches/$BRANCH/protection" \
    --jq '{
      checks: .required_status_checks.contexts,
      strict: .required_status_checks.strict,
      admins: .enforce_admins.enabled,
      reviews: .required_pull_request_reviews.required_approving_review_count,
      stale_dismissed: .required_pull_request_reviews.dismiss_stale_reviews,
      force_pushes: .allow_force_pushes.enabled,
      deletions: .allow_deletions.enabled
    }'
done
```

`force_pushes` and `deletions` must be `false`; `admins` and `strict` must be
`true`.

---

## 3. Full-data `ProductionFitEngine` benchmark — `BLOCKED_HUMAN_ACTION`

### Why it cannot run here

The harness is wired and verified, but the corpus is not present in this
environment. Running it produces exactly the right refusal:

```
$ python ops/benchmark_production_fit.py --slate-date 2026-11-15 \
    --data-root /workspace/data --work-root /tmp/bench-scratch
BenchmarkRefused: the fit exited 4: DataStateError: no rolling-state record at
/workspace/data/.state/current_season_state.json. Run the Step 3B refresh
before fitting; the trainer does not fetch data itself.
```

The 2001-to-present rolling data root exists only on the production runner.
Every timing currently on record comes from the test suite's stub engine,
which says nothing about whether a real fit finishes inside the lifecycle's
360-minute timeout on the machine that has to run it.

### The exact command

On the runner Mac, with the durable paths configured and after a successful
refresh, using a scratch root that is **not** the production work root:

```bash
cd ~/actions-runner-nba/_work/nba-prop-quant/nba-prop-quant   # the checkout
mkdir -p /tmp/nba-fit-benchmark

python ops/benchmark_production_fit.py \
  --slate-date "$(TZ=America/New_York date +%F)" \
  --data-root "$NBA_PROP_DATA_DIR" \
  --work-root /tmp/nba-fit-benchmark \
  --model-dir "$PWD/models" \
  --registry-root "$NBA_PROP_FIT_REGISTRY_DIR" \
  --production-sha "$(git rev-parse HEAD)" \
  --receipt-path ops/evidence/production_fit_benchmark.json
```

Expect hours, not minutes. Run it under `caffeinate -i` so item 1's power
settings are not the thing being tested:

```bash
caffeinate -i python ops/benchmark_production_fit.py ...
```

### What it is not allowed to do, and how that is enforced

No registry root is passed to the fit, so `--benchmark-only` constructs no
registry object and there is nothing for it to register to. `--registry-root`
above is used **only** to fingerprint the registry before and after. The
serving model tree, the registry tree and the three frozen identifiers are
digested on both sides of the run, and any difference fails the benchmark —
even when the fit itself succeeded, because a run that moved production state
has not measured production, it has changed it. The scratch root is refused if
it resolves inside the repository.

### How closure is decided

The harness exits `0` only when all of the following hold, and the receipt
records each:

- `outcome` is `COMPLETED`
- every one of the eleven required stages completed, and `registration` did
  not
- every computed validation check passed, including the real
  `prediction_smoke_test`
- no `FrozenPolicyViolation` was raised
- `production_state.mutation_occurred` is `false`

Commit only `ops/evidence/production_fit_benchmark.json`. It is a few
kilobytes. The staging workspace under the scratch root holds fitted model
binaries and must not be committed; the receipt names it by path and size
instead.
