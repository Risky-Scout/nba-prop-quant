# Reviewer Reproduction Notes

Repository: `Risky-Scout/nba-prop-quant`

Model source commit: `4def8ad33ccc56016fb19a97fceca6e027c9612a`

Release tag: `nba_prop_quant_v2_preprospective_review_20260831_4def8ad`

## Integrity checks

Run SHA-256 verification in:

- `research/v2_gate1_lock`
- `research/v2_gate2_certification_outputs`
- `research/v2_gate3_lock`
- `research/v2_gate3_deployment_artifacts`
- `research/v2_gate3_capture_lock`

## Unit/integration suite

With dependencies installed:

`PYTHONPATH=src python -m pytest -q`

Expected pre-release result: **43 passed**.

## Runtime

The attached runtime ZIP contains the exact source/model deployment whose `RUNTIME_MANIFEST.json` records source commit `4def8ad33ccc56016fb19a97fceca6e027c9612a`.

The runtime bundle intentionally excludes API credentials and prospective snapshot outcomes.

## Interpretation boundary

This package is intended for architecture/methodology/model audit before prospective NBA outcomes are available.

The final prospective certification should be appended later without altering this frozen model candidate.
