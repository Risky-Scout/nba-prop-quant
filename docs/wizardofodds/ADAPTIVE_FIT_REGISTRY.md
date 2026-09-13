# Adaptive fit registry

Immutable storage, fail-closed promotion and last-good rollback for daily
adaptive fits.

This document describes infrastructure only. Nothing here trains, refits,
recalibrates or selects a model, and nothing here changes predictive
mathematics or the T-20 capture protocol.

## Purpose

Training currently writes in place to `models/`. A failed or degraded nightly
run therefore overwrites the artifacts that are serving production, and there
is no way back. The registry removes that failure mode:

- a daily fit becomes an immutable, content-addressed directory;
- a registered fit is not production until it is separately validated and
  promoted;
- promotion is fail-closed, so a bad candidate never displaces the fit that is
  currently serving;
- the previously promoted fit stays on disk and can be restored.

## Frozen architecture versus adaptive fitted values

The registry exists to prove that a nightly job changed only the second
category below.

**Frozen choice.** Which mean mode each target uses, which marginal family,
which calibration family, which dependence lambda, which Gate 3 route, the
feature schema, and the hyperparameters in `configs/model.yaml`. These live in
`models/frozen_manifests/nba_prop_quant_v2_adaptive_architecture_contract.json`.

**Daily fitted value.** Boosters, ensemble simplex weights, experience-curve
coefficients, ZINB parameters, copula correlation, Platt intercept and slope,
and Gate 3 role-model weights. These are the registry payload and are expected
to differ every day.

A promotion is **not** model selection. A daily fit changes numerical fitted
parameters underneath a frozen architecture. Registration is refused outright
when the architecture moves.

### What the contract enforces

At registration the registry re-derives the frozen choices from the working
tree and refuses any drift in:

- `architecture_reference_sha` (`4def8ad33ccc56016fb19a97fceca6e027c9612a`)
- the five frozen routing digests (mean, marginal, calibration, dependence,
  Gate 3)
- `feature_schema_hash`
- `config_hash`
- `gate3_candidate_policy_id`
- `core_seed` (73), `gate3_seed` (20260830) and `effective_n_jobs` (2)

### Why routing digests rather than whole-file hashes

`models/mean_model_selection.json` carries `production_weights` and
`models/market_probability_calibration_policy.json` carries
`production_parameters`. Both files hold a frozen routing decision and a daily
fitted value in the same document. Hashing either file whole would freeze
numbers that are meant to move every day, which is exactly what the contract
must not do.

The registry therefore extracts only the routing maps and enforces digests over
those. Whole-file hashes are recorded in the contract under
`reference_policy_file_sha256` as provenance for the tree the contract was
generated from, and are explicitly not enforced.

The contract carries no wall-clock timestamp, so it is byte reproducible from
the working tree at `generated_from_commit`.

## The `fit_id` namespace

`fit_id` is a third, deliberately distinct identifier:

```
nba_prop_quant_fit_<YYYYMMDD>_<16 hex characters>
```

It does not overload either existing namespace:

- **model `freeze_id`** (for example `nba_prop_quant_20260818T205213Z`)
  identifies a frozen static release in `models/frozen_manifests/`;
- **runtime bundle `freeze_id`** is minted separately per bundle by the
  WizardOfOdds runtime bundle builder.

Those two already coexist. `fit_id` is added alongside them rather than
replacing or reusing either, so a fit, a freeze and a bundle can always be told
apart.

### Identity inputs

The digest is taken over a canonical JSON serialization of:

- `fit_date` and `training_cutoff`
- `source_commit_sha` and `architecture_reference_sha`
- `architecture_contract_sha256`
- `feature_schema_hash`, `config_hash`
- `training_data_manifest_hash`
- `artifact_hashes` for every registered file
- `calibration_hashes` and `role_state_hash` where applicable
- `core_seed`, `gate3_seed`, `effective_n_jobs`
- `gate3_candidate_policy_id`
- `dependency_environment_fingerprint` (Python version plus package versions)

Identity is deterministic: the same inputs always yield the same `fit_id`, and
two materially different artifact sets never collide. Promotion state is
deliberately excluded, because identity must be settled before anything decides
whether the fit is good. Wall-clock fields such as `training_started_at` are
recorded in the manifest but kept out of the digest so that determinism holds.

## Directory layout

The registry root is supplied explicitly with `--registry-root` or
`NBA_PROP_FIT_REGISTRY_DIR`. There is no default, and a root inside the
repository is refused: fits are production state and do not belong in version
control.

```
<registry-root>/
    fits/
        <fit_id>/
            manifest.json
            artifacts/
            SHA256SUMS
    state/
        promotion_state.json
        validation/
            <fit_id>.json
    locks/
        promotion.lock
        registration.lock
    staging/
```

`staging/` holds in-progress registrations only and is empty at rest.

## Registration

```
ops/adaptive_fit_registry.py register \
    --registry-root /srv/nba-fits \
    --staged-dir /srv/nba-training/run-2026-11-15/models \
    --metadata /srv/nba-training/run-2026-11-15/metadata.json
```

Registration hashes every regular file in the staged tree, refuses symlinks and
non-regular files, builds the manifest in `staging/`, re-verifies the copied
bytes, writes a `sha256  path` inventory covering both the artifacts and
`manifest.json`, and only then renames the directory into `fits/<fit_id>`.

The final directory is created exactly once. If `fits/<fit_id>` already exists,
registration fails: it is never replaced, never merged into, never altered.

**Registration never promotes.** A fit begins life as
`REGISTERED / NOT PROMOTED`.

### Immutability and tamper detection

Because `fit_id` is a digest over the manifest's own identity inputs, tampering
is self-evident and needs no separate signature:

- edit an artifact and its recorded hash no longer matches the bytes;
- edit the manifest to match and the identity inputs now derive a different
  `fit_id` than the directory name;
- edit `SHA256SUMS` and it disagrees with the manifest.

`verify` checks all three.

## Validation

A registered fit must be validated before it may be promoted. Validation
results are recorded in `state/validation/<fit_id>.json`, deliberately outside
the immutable fit directory, so that recording a result never mutates the fit.

The registry records these results; it does not execute the model-side checks.
The adaptive orchestrator built in later work is responsible for running them
and reporting outcomes:

```
training_completed          data_refresh_valid
history_regression_check    advanced_coverage_check
required_artifacts_present  artifact_hashes_valid
finite_values_check         feature_schema_match
architecture_contract_match source_lineage_match
marginal_fit_valid          calibration_valid
prediction_smoke_test       gate3_role_readiness
t20_protocol_compatible
```

Every one is required. Promotion fails if any check is `false` **or missing**.

There is deliberately no nightly model-superiority criterion. Adding one would
turn the daily job into model selection, which this contract forbids.

Note that `gate3_role_readiness` is load-bearing operationally: when role state
is unavailable at pricing time, assists, points+assists and points+rebounds are
rejected rather than degraded, so three of ten markets silently leave the feed.
Validation makes that an explicit gate instead of a surprise.

## Promotion

```
ops/adaptive_fit_registry.py promote \
    --registry-root /srv/nba-fits \
    --fit-id nba_prop_quant_fit_20261115_2abfb7037b663663 \
    --reason "nightly adaptive fit" \
    --actor orchestrator
```

Promotion takes an exclusive `fcntl.flock` on `locks/promotion.lock` and then,
before changing anything:

1. re-verifies the manifest and re-derives `fit_id` from its identity inputs;
2. re-hashes every stored artifact against the manifest and the inventory;
3. verifies the architecture contract still matches the working tree;
4. verifies the fit was registered against the current contract;
5. verifies every required validation check is present and passing.

Only then is `state/promotion_state.json` replaced, via a temporary file in the
same directory followed by `os.replace`. The write is a single atomic step, so
a failure at any earlier point leaves `current_good_fit_id` untouched.

Promoting the fit that is already current is refused, so a repeated call cannot
quietly destroy the recorded previous-good.

## Last-good and rollback

`promotion_state.json` records `current_good_fit_id`, `previous_good_fit_id`,
`promoted_at`, `promotion_reason`, `promotion_actor` and an append-only
`history`.

When a candidate fails training or validation, the existing current-good stays
active. Rollback restores the previous good fit:

```
ops/adaptive_fit_registry.py rollback \
    --registry-root /srv/nba-fits \
    --reason "regression detected in nightly fit"
```

Rollback is equally fail-closed and lock-protected. The target fit is fully
re-verified and must still pass validation before it is restored; a corrupt or
deleted rollback target is refused and the currently serving fit is left in
place.

Neither the failed fit nor the previously promoted fit is ever deleted. Fits
are historical evidence.

## Failure semantics

Every command returns a non-zero exit code and a structured JSON error on
stderr. Nothing is silently repaired.

| Exit | Meaning |
|---|---|
| 0 | success |
| 1 | other registry error |
| 2 | usage error |
| 3 | architecture contract violation |
| 4 | integrity or staged-tree error |
| 5 | validation refused |
| 6 | promotion refused or lock held |
| 7 | fit already exists or not found |
| 8 | registry root missing or unsafe |

## Training workspace isolation

**The daily trainer must run in an isolated staging workspace.** It must never
train directly over the currently promoted registry fit.

Existing scripts may continue to write their normal local `models/` paths, so
long as those paths are inside the isolated training workspace rather than a
promoted fit. Artifacts enter the registry only after training and validation
have completed.

This is the boundary that stops the current in-place `models/` overwrite from
destroying last-good production state. Nothing that serves traffic is ever
trained over.

## Relationship to T-20 lineage

Pricing already asserts several identity checks between projections and market
rows: projection `freeze_id` against the manifest, `gate3_candidate_policy_id`
against the loaded Gate 3 runtime, and `gate3_capture_id`, capture window and
capture timestamp between projections and market rows.

Binding a served quote to the `fit_id` that produced it extends that existing
pattern. This PR establishes the identifier and its guarantees; wiring it into
the projection and pricing lineage is later work under Production Step 3.

T-20 remains the primary certification capture and is recorded in the contract
as a frozen choice. It is one of seven capture offsets
(1440, 480, 180, 90, 45, 20, 5 minutes before tip), so any future binding must
name the offset explicitly rather than assume a single capture exists.

## WizardOfOdds deployment seam

The registry format does not depend on the training host or on any local path.
Manifests store artifact paths relative to the fit's `artifacts/` root, and
absolute paths are refused outright, as are credential-shaped fields and
values.

A promoted fit directory is therefore portable as-is. A later deployment system
can copy one exact fit directory, the promotion metadata, and the required
runtime artifacts to the serving host without rewriting the manifest.

No deployment is performed by this tooling.
