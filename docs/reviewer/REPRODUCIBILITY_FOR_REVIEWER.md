# Reproducibility for Reviewer

## Recommended immutable artifacts

Frozen model Release:
`nba_prop_quant_20260818T205213Z`

Certified runtime Release:
`nba_prop_quant_20260818T205213Z_runtime_bundle_v1`

Fair-price integration Release:
`nba_prop_quant_20260818T205213Z_fair_price_feeds_v1`

Probability preregistration tag:
`nba_prop_quant_20260818T205213Z_probability_quality_prereg_v1`

Probability certification results branch commit:
`f1b4fe4400a2313226b9e83bae99be17302f7491`

## Runtime verification

The certified runtime bundle previously passed:

- exact dependency installation;
- `pip check`;
- production manifest verification;
- production contract verification;
- 18 automated tests;
- offline bootstrap verification.

## Probability certification verification

Within:

`research/probability_quality_certification_v1_outputs/`

run:

`shasum -a 256 -c SHA256SUMS.txt`

The certification runner is:

`research/probability_quality_certification_v1.py`

The evaluation protocol is:

`research/PROBABILITY_QUALITY_CERTIFICATION_PROTOCOL_V1.md`

The artifact-coverage record is:

`research/PROBABILITY_QUALITY_ARTIFACT_COVERAGE_V1.md`

## Statistical-review principle

The reviewer should treat the certification results as fixed evidence. Do not modify probability buckets, exclusions, marginal choices, calibration policies, or dependence settings and then call the modified system the same freeze.
