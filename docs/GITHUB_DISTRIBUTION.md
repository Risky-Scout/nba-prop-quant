# GitHub Distribution Plan

## Recommended architecture

Use two distribution layers.

### 1. Ordinary Git repository

Store:

```text
README.md
.gitignore
.gitattributes
.github/
configs/
src/
scripts/
ops/
tests/
models/
  *.json
  frozen_manifests/
review/
docs/
release/
pyproject.toml
```

Do **not** store the monolithic frozen model package in normal Git history.

GitHub blocks normal Git objects over 100 MiB and enforces a 2 GiB push limit. The source repository should therefore stay code/document focused.

### 2. GitHub Release asset

Tag the exact frozen deployment:

```text
nba_prop_quant_20260818T205213Z
```

Recommended release title:

```text
NBA Prop Quant — 2026-27 External Test Deployment
```

Recommended canonical asset name:

```text
nba_prop_quant_20260818T205213Z_production_model_package.zip
```

If the package exceeds the per-release-asset limit, split it into sub-2-GiB chunks using the included release-preparation script.

## Why Releases rather than committing the ZIP

The model ZIP is a deployable artifact, not source history. Keeping it in Releases:

- avoids bloating clones;
- keeps code history reviewable;
- allows the wizardofodds.com deployment process to download one versioned artifact;
- ties the artifact to the exact frozen tag;
- supports explicit SHA-256 verification.

## Git LFS

Git LFS is appropriate if individual large model binaries must be versioned inside the repository. It is **not required** when the complete binary bundle is distributed only as a Release asset.

This publication scaffold intentionally ignores `models/**/*.joblib` and similar local model binaries in ordinary Git.

If you later decide individual model binaries must be in-repo, configure Git LFS deliberately and review its storage/bandwidth implications before changing `.gitignore`.

## Public vs private

Default recommendation before deployment review: **private repository**.

If `wizardofodds.com` must fetch the model from a private repository, the production server must authenticate to GitHub using a narrowly scoped credential stored outside the repository.

If the repository/release is later made public, remove all proprietary or sensitive material intentionally; do not assume `.gitignore` can erase secrets that were previously committed.

## Release immutability

Treat `nba_prop_quant_20260818T205213Z` as immutable. Never replace an already published artifact under the same frozen ID. Any material model or production-pricing change must receive a new deployment freeze ID and a new release/tag.

## Repository creation

The safe command sequence is documented in `GITHUB_SAFE_UPLOAD_COMMANDS.md`.
