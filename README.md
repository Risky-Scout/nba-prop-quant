# NBA Prop Quant

Frozen pregame NBA player-prop fair-pricing system.

**Frozen deployment:** `nba_prop_quant_20260818T205213Z`  
**Stage:** `external_test_deployment`  
**First prospective external market test:** 2026-27 NBA season  
**Intended downstream use:** predictive NBA player-prop fair pricing for `wizardofodds.com`

## What the system prices

Supported frozen production markets:

- points
- rebounds
- assists
- steals
- blocks
- made threes
- points + rebounds
- points + assists
- rebounds + assists
- points + rebounds + assists

The system produces model-implied probabilities and fair prices. It does **not** contain an automatic betting threshold.

## Frozen production policy

- Mean models: PTS XGB, REB ensemble, AST XGB, STL XGB, BLK ensemble, FG3M XGB.
- Marginals: mean-preserving ZINB for all six single-stat targets.
- Combination dependence: R+A uses production `lambda=0.85`; other frozen combos use `lambda=0.00`.
- Probability calibration: prop-specific calibration except blocks and steals, which remain raw.
- Monitoring edge grid: 1%, 2%, 3%, 5%, 7.5%, 10%.
- Automatic bet threshold: **none**.

## Evidence posture

The 2025 sportsbook sample is development/model-selection evidence, not the first untouched external test. After chronological calibration, the model was approximately competitive with the de-vig market overall on proper scoring, but broad sportsbook superiority is **not** claimed.

The 2026-27 season is the first prospective external market test.

## Repository vs. release asset

This repository should contain source code, tests, policies, manifests, review documentation, and small validation reports.

The complete frozen production model package is distributed as a **GitHub Release asset** under tag:

`nba_prop_quant_20260818T205213Z`

Do not commit the monolithic model ZIP, raw historical data, `.env`, API keys, or generated external-test archives to ordinary Git history.

## Start here

- `docs/GITHUB_DISTRIBUTION.md`
- `docs/WIZARDOFODDS_INTEGRATION.md`
- `docs/RELEASE_CHECKLIST.md`
- `review/NBA_PROP_QUANT_FULL_LLM_HANDOFF_2026_08_19.md` if present
- `models/frozen_manifests/LATEST.json`

## Reproduction checks

```bash
python scripts/verify_frozen_manifest.py
python scripts/10a_validate_production_contract.py
python -m pytest -q
```

## Current operational status

Completed:

- historical ingestion and audit
- dynamic priors
- walk-forward minutes and target mean models
- conservative mean-model selection
- mean-preserving ZINB marginals
- grouped-CV dependence policy
- strict-date probability calibration
- 2025 development market backtest
- frozen production integration
- immutable capture and grading infrastructure
- synthetic grading integration QA
- real 2025 5,000-row grading replay
- read-only preseason/current-season refresh preflight
- work paper, constant inventory tooling, and reviewer handoff

Still required before declaring the live pricing path fully operational:

1. first genuine nonempty live slate projection;
2. first genuine live `priced_markets.parquet`;
3. runtime confirmation of exact side/edge/EV field names;
4. checksum verification of the first real capture;
5. post-game grade verification;
6. prospective 2026-27 monitoring without retuning the frozen deployment.

## Security

Never commit:

- `.env`
- BALLDONTLIE API keys
- personal access tokens
- raw sportsbook credentials
- raw production secrets

Use repository secrets or deployment environment variables for server-side integrations.
