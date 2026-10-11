# The incumbent WizardOfOdds publisher

## What this is

`ops/publish_incumbent_to_wizardofodds.py` is the publishing step
`docs/wizardofodds/PRODUCTION_AUTOMATION.md` §11 described as "the publishing
step that follows Step 3D". Until it existed, the production lifecycle served
a priced slate to local parquet and stopped, and getting those numbers onto
wizardofodds.com was an entirely manual act with no contract, no verification
and no way back.

It is **disabled for live publication** and will stay that way until a
separate reviewed change enables it. What it does today is prove the path
against a non-live destination.

## What it publishes, and what it does not compute

The publisher reads the artifact the incumbent already produced:

| Input | Path |
| --- | --- |
| Priced markets | `<data-root>/processed/priced_markets/<slate>.parquet` |
| Serving receipt | `<data-root>/processed/incumbent_serving/<slate>.json` |

Both are written by `ops/run_incumbent_production_serving.py`, and the second
is the same receipt `ops/grade_incumbent_production_slate.py` reads, so the
publisher and the grader provenance the same bytes.

**No number is computed here.** Every probability, price, edge and EV is
copied from the parquet cell to the published field, and
`docs/WIZARDOFODDS_INTEGRATION.md` §5 is explicit that the consumer "must not
recalibrate probabilities" — a publisher that recomputed them would be the
first thing to break that promise. There is no model, copula, calibration or
pricing import anywhere in the module, and two tests pin it: one asserts the
published value equals the source cell exactly, and one asserts the module
imports none of those paths.

### The published record

A projection of the runtime's own columns, under the runtime's own names. The
rest of the priced file is model internals, which §3 of the integration
contract says "are not part of the website contract", and they are left where
they are.

| Group | Fields |
| --- | --- |
| Who and which game | `player_id`, `player_name`, `team_id`, `opponent_id`, `is_home`, `game_id`, `date` |
| Which quote | `quote_id`, `vendor`, `prop_type`, `line_value`, `market_type` |
| Posted price | `over_odds`, `under_odds` |
| Model probability | `calibrated_q_over_nonpush`, `calibrated_q_under_nonpush`, `calibrated_p_push`, `calibration_method` |
| Market probability | `market_devig_q_over`, `market_devig_q_under`, `market_hold_pct` |
| Fair model odds | `calibrated_fair_over_american`, `calibrated_fair_under_american` |
| Edge and EV | `calibrated_edge_over`, `calibrated_edge_under`, `calibrated_ev_over`, `calibrated_ev_under`, `model_preferred_side`, `model_preferred_edge`, `model_preferred_ev` |
| Deployment identity | `freeze_id`, `freeze_stage`, `manifest_sha256`, `market_pricing_schema_version` |

No `raw_*` field is published. §5 of the integration contract says they "are
retained for audit and should not be published as the model's price", and the
field list is pinned so adding one is a test failure rather than a judgement
call later.

Run-level provenance lives once in the release manifest rather than on every
row: the production code SHA, the incumbent authority, version and fit id, the
input state fingerprint, the slate date, when it was served, and the sha256 of
the priced artifact the rows came from.

## Fail-closed rules

Section 9 of `docs/WIZARDOFODDS_INTEGRATION.md`, rule for rule, applied to
every row before anything is written.

The priced file holds only quotes that already passed the quote filter, so a
row breaking one of these rules is the runtime disagreeing with its own
contract. That is a **refusal**, not a filter: a publisher that dropped the row
would publish a thinner feed and say nothing about why, which is the failure
the fail-closed rules exist to make loud.

Beyond the contract's own list, the publisher also refuses:

- an input frame carrying any shadow-candidate column, by the exact names
  `ShadowDecision.payload` emits. The way a candidate probability reaches a
  feed is not somebody choosing to publish one; it is a file that looks close
  enough to the right file;
- a priced file missing a column the rules *read* but do not publish, because
  otherwise a file without `quote_eligible` would publish perfectly with the
  rule that reads it silently not applying;
- more than one row per logical quote (`vendor`, `game_id`, `player_id`,
  `prop_type`, `line_value`);
- an empty priced file, which is not a publication.

## Modes

Two, switched independently in `ops/wizardofodds_publishing_switch.json`.

### `live` — DISABLED, and not enableable from here

Enabling it needs three independent things, the same shape the shadow
publishing switch already uses: the committed state flipped in a reviewed
commit, `NBA_PROP_WOO_PUBLISH` set to that same state, and
`NBA_PROP_WOO_PUBLISH_APPROVAL` set to the switch's own approval token. No
single mistake publishes.

Even with all three, live publication refuses, and for a different reason:
`ops/wizardofodds_destinations.json` carries `live: null` because **every fact
needed to describe the destination is external to this repository**. The
refusal names them rather than saying "not configured", because the difference
between a missing line of config and a missing agreement with the destination
is the difference between finishing this and being blocked on it:

- the wizardofodds.com host the feed is published to;
- the remote directory or object prefix it is served from;
- the transport protocol that host accepts (FTPS, SFTP, an HTTPS API or an
  object store);
- the credential or secret names that transport authenticates with;
- a staging or canary destination on that host, separate from the live feed;
- the public URL or path the published rows appear at.

Inventing any of them would produce a publisher that looks configured and
writes nowhere real, which is worse than one that refuses.

### `canary` — ENABLED, and safe by construction

A canary nobody can run proves nothing, so this mode is enabled. It is safe
because of what exists rather than because of what is permitted:
`local_directory` is the only transport implemented, there is no live
destination for it to reach, and a canary whose destination resolves to the
live one is refused by name.

```
python ops/publish_incumbent_to_wizardofodds.py \
  --mode canary \
  --slate-date 2026-10-20 \
  --data-root "$NBA_PROP_DATA_DIR" \
  --destination-root /tmp/woo-canary \
  --receipt-path /tmp/publication.json
```

The canary root is supplied per run and is deliberately not committed: it is a
scratch location, and committing one would be a destination this repository
asserted without being told.

## Releases and rollback

A release is written whole, verified, and only then promoted.

```
<destination root>/
    releases/<release_id>/rows.json       the published rows
    releases/<release_id>/manifest.json   provenance, row count, digests
    current.json                          which release the destination serves
    previous.json                         the rollback target
```

1. **Write** both files. The transport stages beside the target and renames,
   so a reader never sees a half-written file.
2. **Verify** by reading both back *through the transport* and comparing
   digests, then re-counting the rows. Readback rather than recollection: the
   thing being caught is a destination that accepted the write and kept
   something else, and only a readback can see that.
3. **Promote** by replacing `current.json`, which is one atomic rename, so a
   reader sees either the release that was current before or the one after.
   Promotion is last because any earlier ordering has a window where the
   destination serves a release nobody checked.

The release id is derived from the payload digest rather than from a clock,
which is what makes a rerun idempotent: the same rows produce the same id, so
the second run recognises its own release instead of stacking a new one beside
it. A rerun of the release already current is a no-op on both pointers — it
has to be, because rotating `current` into `previous` when the two are the
same release would destroy the rollback target.

### Rolling back

```
python ops/publish_incumbent_to_wizardofodds.py --rollback \
  --slate-date 2026-10-20 \
  --data-root "$NBA_PROP_DATA_DIR" \
  --destination-root /tmp/woo-canary
```

Restoration, not reconstruction. The previous release's bytes are still at the
destination, so the pointer moves back and the destination serves what it
served before. Nothing is refitted, recalibrated or promoted, and no candidate
shadow state is read or written. It refuses rather than guessing when the
previous release is absent or its bytes are gone, because a rollback that
silently landed somewhere else would be worse than staying put.

## What this is not

- It is **not** wired into `.github/workflows/nba_production_lifecycle.yml`.
  The lifecycle serves, grades, shadows and monitors; publishing is a separate
  decision with a separate record, and the scheduled workflow naming this
  entry point would make running it the default rather than the decision. A
  test pins that the workflow does not mention it or its two variables.
- It is **not** a promotion path. The published authority is the incumbent and
  the candidate remains shadow-only, both recorded in the switch and asserted
  in the manifest of every release.
- It does **not** change the serving step. `run_incumbent_production_serving.py`
  is byte-identical to production and its docstring still says "It does not
  publish." A publisher that needed serving adjusted would be a publisher that
  was not reading the already-approved artifact.

## Before live publication

In addition to the three switch conditions:

1. the six external facts above, supplied by whoever owns the destination;
2. a transport implementing that protocol, written against the real contract
   rather than a guessed one;
3. a canary against the real staging destination, with the readback verified
   there;
4. the activation SHA and activation time recorded, as
   `research/final_model/live_shadow_promotion_policy.json` requires of any
   WizardOfOdds activation.
