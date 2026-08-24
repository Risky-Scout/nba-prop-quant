# Model Identity and Hashes

## Frozen statistical model

- Freeze ID: `nba_prop_quant_20260818T205213Z`
- Frozen source commit: `746f47ef029b8f07670e691365a66ed2694bf06b`
- Frozen model Release tag: `nba_prop_quant_20260818T205213Z`
- Frozen model package SHA-256:
  `9044a3161247868710a9de8b4a9cd393226234abf539dd1dcb82ffb9b68eb5a6`

## Certified runtime

- Runtime tag: `nba_prop_quant_20260818T205213Z_runtime_bundle_v1`
- Runtime commit: `3a5e035ebfc428e187933f84744fb1dec37dd7e3`
- Runtime ZIP SHA-256:
  `076b20aab940849f1c9b582c751f1fa434f3004b11aba4039c219f73599e52e8`
- Certified platform: macOS arm64 / Python 3.14.5

## Fair-price integration

- Integration tag: `nba_prop_quant_20260818T205213Z_fair_price_feeds_v1`
- Integration commit: `f92eaf8a8a94be840cdc82b107c5e9c3b7cf3b3b`
- Integration ZIP SHA-256:
  `3f8f5f591a2c462501713a8e0807f57c431efa3bda3e2fa40231ebc53a936dce`
- Statistical model changed by integration: **NO**

## Probability quality certification

- Preregistration tag:
  `nba_prop_quant_20260818T205213Z_probability_quality_prereg_v1`
- Preregistration commit:
  `5aacd0e81b578e2edbc0255f6f3afb39fcebbecb`
- Pre-analysis provenance commit:
  `5594625fb1fe00fda8c2eb13891b34b416de9203`
- Certification runner commit:
  `3c9bd2f444d807c9d2fc7d3ac19b326288ce6834`
- Certification results commit:
  `f1b4fe4400a2313226b9e83bae99be17302f7491`

## Statistical immutability

The review package does not silently alter the statistical model. Any reviewer-requested statistical change requires a new freeze ID, new manifests, and new evidence.
