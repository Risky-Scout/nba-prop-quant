# V2 Reference Provenance Addendum

Machine-readable companion:
`models/frozen_manifests/nba_prop_quant_v2_reference_provenance_addendum.json`
(SHA256 `3bab2a71bb453da79a7d60e0f9f18a87f6c72cf550158d3779504bf7b0ba790a`).

---

## 1. Why this document exists

The historical static freeze manifests record a `git.commit` value that does
not resolve in this repository. That value is therefore not usable as NBA
source provenance.

Nothing historical is corrected in place. The freeze manifests, the Gate 3
runtime contract, the frozen model artifacts and `PUBLIC_CLAIM_POLICY_V1` are
all left exactly as captured. Rewriting a historical artifact to make it look
right would destroy the evidence of how the freeze was actually taken, which
is worth more than a tidy field. This addendum is added alongside them
instead.

---

## 2. What is authoritative

| Role | Value |
| --- | --- |
| Immutable architecture and mathematical source reference | `4def8ad33ccc56016fb19a97fceca6e027c9612a` |
| Resolves in this repository | yes |

This is the commit the Gate 3 runtime contract pins `frozen_model_source_files`
to, and the commit the Step 3A adaptive architecture contract carries as
`architecture_reference_sha`. It is the reference for what the mathematical
model *is*.

---

## 3. What is not authoritative

| Field | Value |
| --- | --- |
| Recorded `git.commit` | `fda849a36ac8e61b122f07cb821e0071f2b26107` |
| Resolves in this repository | **no** |

It appears in three historical manifests, all unmodified:

| Manifest | freeze_id | freeze_stage |
| --- | --- | --- |
| `models/frozen_manifests/LATEST.json` | `nba_prop_quant_20260818T205213Z` | `external_test_deployment` |
| `models/frozen_manifests/nba_prop_quant_20260818T205213Z.json` | `nba_prop_quant_20260818T205213Z` | `external_test_deployment` |
| `models/frozen_manifests/nba_prop_quant_20260818T202318Z.json` | `nba_prop_quant_20260818T202318Z` | `pre_live_integration_architecture_baseline` |

### Why it does not resolve

The `status_porcelain` captured beside that commit explains it. It holds 7,656
entries and is rooted at a user home directory rather than at this repository:

- home-directory dotfiles appear at top level, including `.CFUserTextEncoding`,
  `.DS_Store`, `.Renviron`, `.bash_history` and `.cache`;
- the bulk of the entries sit under sibling project trees that do not exist in
  this repository at all, principally `woo_models/` and
  `SportsModels/wnba-player-props-pmf-model/`.

The freeze-time capture therefore ran in an outer working tree that contained
this project among others. Its `git.commit` identifies that outer tree, not
the NBA model source, which is exactly why the value is unresolvable here.

This addendum states only that the value is not a commit of this repository.
It makes no claim about where that commit does live.

### The dirty flag stays as recorded

The same manifests record `git.dirty = true`. That is preserved verbatim. It
is truthful evidence that the freeze was captured from a working tree with
uncommitted changes, and it is not rewritten, softened or removed.

---

## 4. Step 3B source lineage

| Field | Value |
| --- | --- |
| Base commit | `a9450a882a19e524c21599f5ea47493bab8fe3b3` |
| Base branch | `production/wizardofodds-integration` |
| Implementation branch | `cursor/nba-step3b-correctness-integrity` |
| Merge commit | not yet created |

Step 3B is under review and unmerged, so its merge commit does not exist. It is
recorded as `null` rather than invented. The base commit and implementation
branch identify the lineage while the change is open, and the merge commit can
be recorded once it exists.

---

## 5. Adaptive serving source diverges from the anchor, by declaration

The Gate 3 runtime contract byte-pins nine serving source files to the
architecture reference. Adaptive production needed a serving-correctness fix in
one of them, so a second contract now records what adaptive serving actually
runs:

`models/frozen_manifests/nba_prop_quant_v2_adaptive_serving_source_contract.json`
(SHA256 `97292af4dc9486de1bee1a446c9ef0f7f85a3db162c0a84eb17ce639727c4ad9`).

It is additive. It does not replace, amend or reinterpret the historical Gate 3
runtime contract, and it is not a new schema version of it. The historical
contract is unchanged and still pins those files to
`4def8ad33ccc56016fb19a97fceca6e027c9612a`.

For each of the nine files it records the current SHA256, the SHA256 of the
blob at the architecture reference, and whether the two match. Exactly one
diverges:

| File | Status |
| --- | --- |
| `scripts/15_price_markets.py` | diverged, reason recorded |
| the other eight | byte-identical to the architecture reference |

The divergence is a lineage-field ordering correction. The pricing emitter
assigned `gate3_external_test_record` from `priced["external_test_record"]`
before that column existed, and no upstream stage produces it, so the read
raised `KeyError` and no priced-markets output could be produced. The run-level
flag is now established before it is read.

The contract asserts, and the test suite enforces, that this changed no model
mathematics, no Gate 3 routing, no calibration or dependence methodology, no
marginal family or mean-model routing, no feature definition, no fitted
artifact and no T-20 protocol.

Integrity verification was widened rather than relaxed. The suite now checks
that the eight unchanged files are still byte-identical to the anchor, that a
declared divergence is a real one, that the anchor's own bytes are unchanged
for all nine, and that every file matches the new serving contract.

One limitation is recorded in the contract itself:
`scripts/19_build_wizardofodds_runtime_bundle.py` still verifies serving
sources against the architecture reference alone, so building a runtime bundle
from the corrected source would fail its frozen-source check. Teaching the
bundle builder about this contract is deployment work and is deliberately out
of scope here.

---

## 6. How provenance is established from here

The Step 3A immutable adaptive fit registry removes the need to infer source
provenance from a freeze-time capture at all.

Every registered daily fit records `source_commit_sha` in its immutable
manifest, and that field is one of the inputs the `fit_id` digest is taken
over. Source provenance is therefore:

- **per fit** rather than per freeze;
- **immutable**, because changing it changes the `fit_id` and no finalized fit
  is ever rewritten;
- **independently verifiable**, because the manifest can be re-derived and
  re-hashed at any time.

A future fit can be traced to the exact commit that produced it without
relying on any historical manifest field.

---

## 7. Scope

This addendum is a provenance statement and nothing else. It does not revise
any frozen model artifact, any historical runtime contract, or
`PUBLIC_CLAIM_POLICY_V1`, and it makes no statement about model quality or
predictive performance.
