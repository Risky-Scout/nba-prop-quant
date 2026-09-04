# NBA Prop Quant v2 / Gate 3 production runtime contract

This document describes the deployment-integrity contract that lets the
already-frozen v2 / Gate 3 model be packaged, verified and executed
reproducibly on a clean hosted runner.

It is **not** a new statistical model version. No model artifact, policy,
calibration, dependence lambda, capture rule or pricing formula changes.

## 1. The defect this fixes

`nba_prop_quant.production.load_verified_manifest_metadata()` verifies every
file recorded in `models/frozen_manifests/LATEST.json`. That pointer is a copy
of the 2026-08-18 v1 freeze (`nba_prop_quant_20260818T205213Z`), which predates
the Gate 3 work. Three files it hashes have legitimately changed since:

- `scripts/10_predict_slate.py`
- `scripts/15_price_markets.py`
- `src/nba_prop_quant/production.py`

So the current Gate 3 production source fails its own integrity check.

The fix is a **new v2 runtime manifest** whose hashes describe the frozen v2 /
Gate 3 source and the exact unchanged model artifacts. Verification is not
disabled, relaxed, allowlisted or skipped, and the v1 freeze manifests are left
untouched.

## 2. Manifest schema finding

The existing verifier needs no changes. It requires only that
`<model_dir>/frozen_manifests/LATEST.json`:

- exists and parses as JSON;
- has `freeze_stage == "external_test_deployment"` (unless the caller passes
  `--allow-predeployment`);
- has a `freeze_id`;
- has a `files` object whose values are lists of `{"path", "sha256"}` records,
  resolved relative to `project_root`.

Group names are free-form, the required file set is not hard-coded, and unknown
top-level keys are ignored. A richer v2 manifest is therefore representable
without touching verification code.

## 3. Runtime bundle shape

The bundle is laid out as a project root, so the unchanged verifier accepts it
with `project_root=<bundle>` and `model_dir=<bundle>/models`:

```
<bundle>/
  RUNTIME_MANIFEST.json          canonical v2 runtime manifest
  RUNTIME_SHA256SUMS.txt         SHA-256 of every other bundled file
  pyproject.toml
  src/nba_prop_quant/*.py        runtime import closure
  scripts/*.py                   production entry points
  configs/model.yaml
  docs/wizardofodds/             frozen public claim policy
  research/v2_gate1_lock/        pre-registered policy locks
  research/v2_gate3_lock/
  research/v2_gate3_capture_lock/
  research/v2_gate3_deployment_artifacts/
  models/                        frozen model artifacts
  models/frozen_manifests/nba_prop_quant_v2_gate3_runtime_manifest.json
  models/frozen_manifests/LATEST.json      byte-identical pointer copy
  data/raw/seasons/**/stats.parquet        historical box-score bootstrap
  data/raw/advanced/**/*.parquet           historical advanced bootstrap
```

`RUNTIME_SHA256SUMS.txt` cannot contain its own hash; every other bundled file,
including all three manifest copies, is listed.

## 4. What is included, and why

`models/frozen_manifests/nba_prop_quant_v2_gate3_runtime_contract.json`
declares the runtime file set. Membership follows an actual read trace of
`scripts/10_predict_slate.py`, `scripts/15_price_markets.py` and their imports,
not package size.

| Group | Required | Why |
| --- | --- | --- |
| `source_files` | yes | Transitive `nba_prop_quant` import closure of the entry points. |
| `scripts` | yes | Snapshot capture, projection, pricing, and the frozen-contract preflight. |
| `model_artifacts` | yes | Read on every run by prediction, pricing or `10a_validate_production_contract.py`. |
| `model_provenance_artifacts` | no | Frozen release artifacts live inference does not read; bundled and hashed when present. |
| `gate3_deployment_artifacts` | yes | Loaded by `gate3_v2.load_gate3_runtime()`. |
| `gate3_policy_locks` | yes | Pre-registered Gate 1 / Gate 3 policy, including the T-20m capture lock. |
| `claim_policy` | yes | Frozen public claim policy travels with the runtime. |
| `configs` | no | Historical configuration retained for provenance. |
| `runtime_data` | yes | Historical state needed to rebuild upcoming-slate features. |
| `packaging` | yes | Declared dependencies for a clean runner. |

Every group above belongs to exactly one integrity domain; see section 4a.

Deliberately excluded, with reasons recorded in the contract's
`excluded_development_resources`: the 2025 out-of-fold and market-backtest
parquets, Gate 2 certification outputs, `data/raw/seasons/**/games.parquet`
(`load_history_box_stats()` globs `stats.parquet` only), development-only
`nba_prop_quant` modules outside the import closure, and all mutable runtime
state.

## 4a. Frozen integrity versus rolling integrity

The runtime resources split into two integrity domains that answer different
questions. The contract states the split explicitly, in `contract_version` 2,
through two top-level lists the builder reads directly:

```json
"frozen_integrity_groups": [
  "source_files", "scripts", "model_artifacts",
  "model_provenance_artifacts", "gate3_deployment_artifacts",
  "gate3_policy_locks", "claim_policy", "configs", "packaging"
],
"rolling_integrity_groups": ["runtime_data"]
```

Nothing is inferred from a naming convention, and neither verifier contains a
`runtime_data` special case. The builder fails the build if a listed group is
not a real contract group, if a group appears in both lists, or if a contract
group is assigned to no domain at all.

**Frozen integrity** covers the mathematical model, its source closure, the
production entry points, the Gate 3 deployment artifacts, the pre-registered
policy locks, the public claim policy and the packaging metadata. These are
the only groups emitted into the runtime manifest's `files` object, which is
the set `nba_prop_quant.production.load_verified_manifest_metadata()`
re-verifies on every production run. They must never change after the bundle
is built, and a single changed byte is still a fail-closed error.

**Rolling integrity** covers historical inference state that is *expected* to
advance as 2026-27 games finish: `data/raw/seasons/**/stats.parquet` and
`data/raw/advanced/**/*.parquet`. Under the previous single-domain layout the
first lawful postgame refresh changed those hashes and broke the frozen-model
verifier, which conflated "the model was tampered with" with "last night's
games were played". Those are different failures and now have different
homes.

Rolling status is not a weakening. `runtime_data` remains a **required** build
input: a build with no matching historical parquet still fails with
`Required historical runtime data missing`. It is still copied into the
bundle, still hashed at build time into `runtime_data_hashes`, still listed in
`RUNTIME_SHA256SUMS.txt`, and still shipped inside the archive.

This yields two separate, simultaneously true guarantees:

1. A freshly downloaded immutable runtime archive verifies byte-for-byte
   exactly as built, historical bootstrap included, via
   `RUNTIME_SHA256SUMS.txt` and `runtime_data_hashes`.
2. After deployment, replacing the current-season historical state with a
   legitimate refresh leaves frozen-model verification passing, because those
   bytes were never part of the permanent frozen hash set.

Post-deployment integrity of the rolling files is therefore **not yet
covered** by any contract. A separate rolling-state integrity contract is
future work and does not exist in this repository; until it lands, the
current-season historical state carries only its build-time bootstrap
attestation.

## 5. Immutable bundle versus mutable runtime state

The bundle carries source, models, policies, historical feature state, Gate 3
artifacts and integrity metadata. It never carries `data/snapshots/**`,
`capture_runs`, projections, priced markets or publication records. Those are
append-only runtime state produced after the bundle is deployed.

## 6. Model source versus production source

The manifest records both, and they are never conflated:

- `model_source_commit` — `4def8ad33ccc56016fb19a97fceca6e027c9612a`, the
  frozen mathematical model.
- `production_source_commit` — the commit on
  `production/wizardofodds-integration` that produced the bundle.

## 7. Building a bundle

```bash
python scripts/19_build_wizardofodds_runtime_bundle.py \
  --project-root . \
  --model-dir /path/to/frozen/models \
  --data-dir /path/to/runtime/data \
  --output-dir /path/to/output
```

Optional: `--contract`, `--gate3-artifact-dir`, `--pip-freeze`,
`--runtime-version`, `--allow-dirty`, `--keep-staging`.

No path is hard-coded. The builder makes no network calls and operates only on
local inputs.

It refuses to produce an archive unless all of the following pass:

1. `4def8ad` is an ancestor of `HEAD`, and the worktree has no uncommitted
   tracked changes (unless `--allow-dirty`).
2. All nine frozen mathematical source files are byte-identical to `4def8ad`.
3. Every Gate 3 `SHA256SUMS.txt` group verifies, including Gate 2 certification
   outputs, which are checked but not bundled.
4. The deployment manifest carries the exact 10-prop Gate 3 policy, the
   expected lock commit and candidate ID, and `prospective_claim_allowed`
   false.
5. The capture lock and `prospective_snapshot.PRIMARY_OFFSET_MINUTES` agree on
   T-20m with no fallback and no retuning from prospective results.
6. Every required model artifact and historical data file is present, and
   every contract group is declared in exactly one integrity domain.
7. No bundled path matches a credential filename pattern and no bundled text
   file contains a credential-shaped assignment or private key block.
8. The archive round-trips: it is extracted to a temporary directory, every
   SHA-256 is re-verified, the manifest is reloaded and checked for the model
   anchor, candidate identity, T-20m window, `auto_bet` false and
   `prospective_claim_allowed` false, and the unchanged production verifier is
   run against the extraction. The temporary directory is then deleted.

Any failure exits non-zero and writes no archive.

## 8. Running from a bundle

```bash
cd <bundle>
pip install -e .
export NBA_PROP_MODEL_DIR=<bundle>/models
export NBA_PROP_DATA_DIR=<bundle>/data
python scripts/10a_validate_production_contract.py
```

The runtime pointer `models/frozen_manifests/LATEST.json` inside the bundle is
the v2 runtime manifest, so the verifier passes against the current Gate 3
source. It keeps passing once the deployed `data/raw/**` bootstrap has been
refreshed with completed 2026-27 games, because those files are rolling, not
frozen.

The repository's own `models/frozen_manifests/LATEST.json` is intentionally
left as the v1 pointer. A committed v2 pointer could not include hashes for the
model artifacts, which are distributed as release assets rather than in git,
and publishing a pointer that silently omits them would weaken verification.
The canonical v2 pointer is therefore a build artifact, produced where the
artifacts actually exist. Running the entry points in place from a bare
checkout is not a supported production path.
