# Adaptive Fit Registry

Immutable daily fit storage, fail-closed promotion and last-good rollback for
NBA Prop Quant adaptive production.

Implemented by `src/nba_prop_quant/adaptive_fit_registry.py` and operated
through `ops/adaptive_fit_registry.py`.

---

## 1. Purpose

Production needs the numbers to move every day and the model to stay still.

The registry is the boundary that makes those two statements compatible. It
stores each day's fitted parameters as an immutable, version-addressed fit, it
refuses to store a fit whose architecture drifted, and it never lets an
unvalidated candidate become the serving model.

It is infrastructure only. It contains no predictive mathematics and it never
trains, refits, recalibrates or selects a model.

Two statements govern everything below.

**Daily fitting is not daily model selection.** A daily fit re-estimates
numerical parameters under a frozen architecture. It does not choose a model
family, a feature set, a hyperparameter or a route.

**Promotion is not model selection.** Promotion asks only whether a candidate
is intact, reproducible and validated. It never asks whether the candidate
beats the incumbent. There is deliberately no nightly superiority test, and
the required validation check list contains no such criterion.

---

## 2. Frozen architecture versus adaptive fitted values

`models/frozen_manifests/nba_prop_quant_v2_adaptive_architecture_contract.json`
pins the architecture. It is a new, additive document: it does not replace or
amend any historical static runtime contract, and it is not a claim policy.

### Frozen choices

| Choice | Value |
| --- | --- |
| Architecture reference | `4def8ad33ccc56016fb19a97fceca6e027c9612a` |
| Model source generation | NBA Prop Quant v2 / Gate 3 architecture |
| Mean-model routing | PTS `xgb`, REB `ensemble`, AST `xgb`, STL `xgb`, BLK `ensemble`, 3PM (`fg3m`) `xgb` |
| Primitive marginal family | `zinb` for all six primitive targets |
| Gate 3 routing | all ten props, matching the assertion in `gate3_v2.py` |
| Calibration-family routing | `raw` for blocks and steals, `prop` for the other eight |
| Dependence production lambda | `rebounds_assists` 0.85, all other combos 0.0 |
| Core seed | 73 |
| Gate 2 / Gate 3 seed | 20260830 |
| Effective XGBoost `n_jobs` | 2 |
| Primary certification capture | T-20m, no fallback |
| Feature-schema hash | derived from `src/nba_prop_quant/features.py` |
| Config hash | derived from `configs/model.yaml` |
| Frozen policy digests | one per selection, calibration, dependence and Gate 3 policy |

### Daily fitted values

These are the registry payload and are expected to differ every day:

- XGBoost fitted booster parameters
- ensemble fitted weights
- ZINB fitted parameters
- adaptive copula correlation
- Platt fitted coefficients
- experience-curve fitted coefficients
- role and minutes fitted weights

None of these is frozen by the contract.

### Why the policy digests are projections

Several policy documents hold a frozen decision and a daily fitted value in
the same file:

- `models/mean_model_selection.json` carries `selected_mode` next to
  `production_weights`
- `models/market_probability_calibration_policy.json` carries
  `selected_method` next to `production_parameters`
- the Gate 3 deployment manifest carries `gate3_policy` next to
  `input_sha256`, row counts and a build timestamp that all move whenever the
  role model is rebuilt

Hashing those files whole would freeze exactly the numbers the contract
permits to change daily, and every legitimate adaptive fit would be refused.
So each frozen policy digest is taken over the document with its daily fitted
fields pruned first. The pruning rules are published in the contract under
`frozen_policy_digest_scope`, so the projection is auditable rather than
implicit.

The effect is a lock that is tight in the right dimension:

- refitting ensemble weights, Platt coefficients or the role model leaves the
  digest unchanged and the fit registers normally
- changing a mean-model route, a marginal family, a calibration family, a
  dependence lambda or a Gate 3 route changes the digest and registration is
  refused

Whole-file hashes are still recorded under `reference_file_sha256` as
provenance for the tree the contract was generated from. They are explicitly
not enforced, and the contract says so under `enforcement`.

---

## 3. The `fit_id` namespace

A fit identifier is a separate namespace. It does not overload either existing
`freeze_id`.

| Namespace | Example | Meaning |
| --- | --- | --- |
| model `freeze_id` | `nba_prop_quant_20260818T205213Z` | a frozen static model release |
| runtime bundle `freeze_id` | runtime bundle identifier | a frozen serving bundle |
| **`fit_id`** | `nba_prop_quant_fit_20261115_1f3a9c7d2b5e8a04` | **one immutable daily fit of numerical parameters** |

Format: `nba_prop_quant_fit_<YYYYMMDD>_<16 hex>`, where the hex is the leading
half of a SHA256 over the canonical identity inputs.

Identity is deterministic. The same inputs always derive the same `fit_id`,
and two artifact trees with different bytes never collide.

Identity inputs:

- fit date and training cutoff
- source commit SHA
- architecture reference SHA and architecture-contract SHA256
- feature-schema hash and config hash
- training-data-manifest hash
- artifact content hashes, calibration hashes, role-state hash
- core seed, Gate 3 seed, effective `n_jobs`
- Gate 3 candidate policy id
- dependency environment fingerprint (Python version and package versions)

Promotion state is deliberately **not** an identity input. Identity has to be
settled before anything decides whether the fit is good. Wall-clock fields
such as `created_at` are recorded in the manifest but kept out of the digest
so identity stays reproducible.

---

## 4. Registry layout

The registry root is always explicit: `--registry-root`, or the
`NBA_PROP_FIT_REGISTRY_DIR` environment variable. There is no default. If
neither is supplied the tool fails closed rather than inventing a location,
and a root inside the Git repository is refused outright — fits are production
state and do not belong in version control.

```
<registry-root>/
    fits/
        <fit_id>/
            manifest.json
            artifacts/
            SHA256SUMS
    state/
        validations/
            <fit_id>.json
        promotion_state.json
    locks/
        promotion.lock
    staging/                  # transient, used during registration only
```

---

## 5. Immutability

A finalized fit is never overwritten, never merged into and never modified.

Registration copies the candidate into a private staging directory under the
registry root, re-hashes every copied byte, writes the manifest and the
`SHA256SUMS` inventory, and only then renames the directory into
`fits/<fit_id>`. Because a rename onto a non-empty directory fails, the claim
is atomic even against a concurrent registration of the same `fit_id`: the
loser raises `FitAlreadyExists` and the existing fit is left untouched.

Tampering is detected without a separate signature:

- `fit_id` is a digest over the manifest's own identity inputs, so an edited
  manifest derives a different `fit_id` than the directory holding it
- every artifact is checked against the hash recorded in the manifest, and
  added or removed artifacts are reported
- the `SHA256SUMS` inventory covers `manifest.json` itself, so a manifest edit
  that avoids the identity inputs still fails

Nothing is ever deleted. Failed candidates, superseded fits and rolled-back
fits are all immutable production evidence.

---

## 6. Registration

```bash
ops/adaptive_fit_registry.py register \
    --registry-root /srv/nba-prop-fits \
    --staged-dir /srv/nba-training-workspace/run-2026-11-15/models \
    --metadata /srv/nba-training-workspace/run-2026-11-15/fit_metadata.json
```

Registration refuses symlinks, device nodes, sockets, FIFOs and empty trees.
It hashes every regular file and sorts paths deterministically.

Before anything is written it enforces the architecture lock, re-deriving the
frozen choices from the working tree and comparing them against the contract.
Registration fails on a wrong architecture reference, a wrong contract hash,
drifted routing, a drifted feature schema, a drifted config, a wrong Gate 3
policy id, or a wrong seed or `n_jobs`. That is what stops a nightly fitting
job from quietly becoming model selection, feature search, hyperparameter
search or architecture research.

Manifests must stay portable, so credential-shaped keys and values are refused
and so are absolute paths.

**Registration never promotes.** A new fit is `REGISTERED / NOT PROMOTED`.

---

## 7. Validation

Validation state lives in `state/validations/<fit_id>.json`, outside the
immutable fit directory, so recording a result never mutates the fit.

The registry records outcomes; it does not execute the model-side checks. A
later adaptive orchestrator runs them and reports them here. All fifteen are
required for promotion:

`training_completed`, `data_refresh_valid`, `history_regression_check`,
`advanced_coverage_check`, `required_artifacts_present`,
`artifact_hashes_valid`, `finite_values_check`, `feature_schema_match`,
`architecture_contract_match`, `source_lineage_match`, `marginal_fit_valid`,
`calibration_valid`, `prediction_smoke_test`, `gate3_role_readiness`,
`t20_protocol_compatible`.

```bash
ops/adaptive_fit_registry.py record-validation \
    --registry-root /srv/nba-prop-fits \
    --fit-id nba_prop_quant_fit_20261115_1f3a9c7d2b5e8a04 \
    --checks /srv/nba-training-workspace/run-2026-11-15/validation.json
```

Promotion rejects a record that is missing, malformed, of an unsupported
schema, carrying a missing or false required check, carrying a non-boolean
check, or belonging to a different `fit_id`. Unknown check names are refused
at recording time, which is what keeps a nightly model-superiority criterion
from being smuggled in as an extra check.

---

## 8. Promotion

```bash
ops/adaptive_fit_registry.py promote \
    --registry-root /srv/nba-prop-fits \
    --fit-id nba_prop_quant_fit_20261115_1f3a9c7d2b5e8a04 \
    --reason "nightly adaptive fit 2026-11-15" \
    --actor nightly-orchestrator
```

Promotion holds an exclusive `fcntl.flock` on `locks/promotion.lock` for its
whole duration. Under that lock it re-verifies that the fit exists, that the
manifest is intact, that every artifact still hashes correctly, that the
architecture contract still matches the tree, and that validation passed with
all required checks.

Only after every check passes is `state/promotion_state.json` replaced, once,
atomically: the new state is written to a temporary file in the same directory
and moved into place with `os.replace`.

Promotion state records `current_good_fit_id`, `previous_good_fit_id`,
`promoted_at`, `promotion_reason`, `promotion_actor` and an append-only
`history`.

---

## 9. Last-good behaviour and rollback

Because the single state replacement is the last step, any failure before it
leaves `current_good_fit_id` exactly as it was. A candidate that fails
validation, fails integrity or violates the contract does not displace the fit
that is serving. A failed nightly job degrades to "yesterday's model is still
live", never to "no model is live".

```bash
ops/adaptive_fit_registry.py rollback \
    --registry-root /srv/nba-prop-fits \
    --reason "regression observed in production" \
    --actor operator
```

Rollback restores `previous_good_fit_id` and is fail-closed in the same way:
it takes the same exclusive lock, and it re-verifies the target's existence,
manifest and artifact hashes and validation record before touching state. A
missing or corrupt rollback target is refused and the current fit keeps
serving. Rollback deletes nothing; it swaps the pointers and appends to
history.

---

## 10. Failure semantics

| Condition | Result | Effect on current-good |
| --- | --- | --- |
| No registry root supplied | `RegistryRootError` | unchanged |
| Registry root inside the repository | `RegistryRootError` | unchanged |
| Architecture drift at registration | `ArchitectureContractViolation` | unchanged |
| Symlink or special file staged | `StagedTreeError` | unchanged |
| `fits/<fit_id>` already exists | `FitAlreadyExists` | unchanged |
| Artifact or manifest tampering | `IntegrityError` | unchanged |
| Validation missing, failing or mismatched | `ValidationRefused` | unchanged |
| Lock held by another operation | `PromotionLocked` | unchanged |
| No previous good to roll back to | `PromotionRefused` | unchanged |

Every CLI failure exits non-zero and prints a structured JSON object on
stderr. Nothing is ever silently repaired.

---

## 11. Isolated training workspace

**The daily trainer runs in an isolated staging workspace.**

Training must never happen inside a promoted fit directory. Existing training
scripts may keep writing their normal `models/` paths, provided those paths
resolve inside the isolated workspace rather than over the serving artifacts.

```
1. isolated workspace   trainer writes models/ inside the workspace
2. register             candidate becomes an immutable fit
3. validate             orchestrator records the required checks
4. promote              validated fit becomes current-good
```

Only artifacts a training run actually produced may be registered, and only a
validated registered candidate may be promoted. This is the boundary that
stops today's in-place `models/` overwrite behaviour from destroying last-good
production state: a failed nightly run damages only its own workspace.

This PR does not modify any training script.

---

## 12. Relationship to existing identifiers

**Model `freeze_id`.** A frozen static model release is a complete, sealed
model. A `fit_id` is one day's numerical parameters under an architecture that
is already frozen. A fit never supersedes or reinterprets a frozen release.

**Runtime-bundle `freeze_id`.** A runtime bundle is a sealed serving
artifact set. The registry is upstream of it: a promoted fit is an input to a
future bundle, not a replacement for one.

**T-20 lineage.** The contract pins T-20m as the primary certification
capture with no fallback. Each fit records its `fit_date`, `training_cutoff`
and `source_commit_sha`, and `validation_lineage_reference` carries the
forward link to the T-20 evidence a later adaptive certification will attach.
Nothing in this PR captures odds or grades anything.

**`PUBLIC_CLAIM_POLICY_V1`.** Untouched. The registry makes no public claim.
The adaptive successor policy is a separate item in the production plan.

---

## 13. WizardOfOdds deployment seam

The format is host-independent. Manifests store relative paths only, absolute
paths are refused at registration, and no training-host or WizardOfOdds path
appears anywhere in the registry.

A finalized fit directory can therefore be copied to WizardOfOdds and verified
there without rewriting its manifest. Deployment transfers one exact fit
directory plus the promotion metadata identifying it as current-good.

Nothing is deployed in this PR, and no connection to WizardOfOdds is made.
