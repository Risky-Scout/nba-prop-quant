# Release Checklist — nba_prop_quant_20260818T205213Z

## Before creating the repository release

- [ ] `python scripts/verify_frozen_manifest.py` passes.
- [ ] `python scripts/10a_validate_production_contract.py` passes.
- [ ] `python -m pytest -q` passes.
- [ ] GitHub source snapshot preflight reports no secrets.
- [ ] No raw data or `.env` is staged.
- [ ] Model package ZIP SHA-256 is recorded.
- [ ] Release asset is under GitHub's per-asset limit, or safely split into chunks.
- [ ] Repository tag exactly matches `nba_prop_quant_20260818T205213Z`.
- [ ] Release notes state that 2025 is development evidence.
- [ ] Release notes state that no automatic betting threshold is frozen.
- [ ] Release notes state that first-live runtime schema confirmation is still pending.
- [ ] wizardofodds.com integration consumes the frozen release by explicit tag, never an unversioned "latest" URL in production.

## After upload

- [ ] Download the release asset to a clean directory.
- [ ] Recompute SHA-256 and compare with the local release manifest.
- [ ] Confirm the tag points to the intended source commit.
- [ ] Record the GitHub release URL in deployment documentation.
- [ ] Do not overwrite the asset under the same freeze ID.
