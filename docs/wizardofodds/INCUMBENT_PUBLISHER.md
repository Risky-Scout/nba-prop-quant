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
| Who is playing, in words | `away_team`, `home_team`, `player_team`, `tip_off_utc` |
| Which quote | `quote_id`, `vendor`, `prop_type`, `line_value`, `market_type` |
| Model projection | `model_projection` |
| Posted price | `over_odds`, `under_odds` |
| Model probability | `calibrated_q_over_nonpush`, `calibrated_q_under_nonpush`, `calibrated_p_push`, `calibration_method` |
| Market probability | `market_devig_q_over`, `market_devig_q_under`, `market_hold_pct` |
| Fair model odds | `calibrated_fair_over_american`, `calibrated_fair_under_american` |
| Edge and EV | `calibrated_edge_over`, `calibrated_edge_under`, `calibrated_ev_over`, `calibrated_ev_under`, `model_preferred_side`, `model_preferred_edge`, `model_preferred_ev` |
| Deployment identity | `freeze_id`, `freeze_stage`, `manifest_sha256`, `market_pricing_schema_version` |

### The record's own version

`contract_version` names the shape of what is written, independently of the
two feed versions in `docs/WIZARDOFODDS_INTEGRATION.md` §1, which describe
what was read. It is **2**: version 1 published neither the model's projection
nor the teams.

A consumer must compare it for **equality**, not as a floor, and the page
renderer does. A renderer written against version 2 reading a version 1
release would find the projection and matchup fields simply absent and draw a
page with holes in it rather than refusing, and a page with holes is the
failure mode worth being loudest about.

Two groups are published under names the runtime does not use, which are the
only two renames in the list. The schedule's `visitor_team_abbreviation` and
`home_team_abbreviation` are published as `away_team` and `home_team`, because
a slate is read away side first and a consumer ordering a matchup from
"visitor" and "home" would have to know that. `player_team_abbreviation` is
published as `player_team` for the same reason and no other.

`model_projection` is the one number saying what the model projects for the
player's statistic, and like every other field it is copied, not computed:
`scripts/15_price_markets.py` writes it as a copy of the mean it priced that
row's line against. §4 of the integration contract says which column that is
per prop and why a combo line carries null. Publishing a null is deliberate —
the alternative is a projection the model never made.

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

### `player_name`

`player_name` is the authoritative active-player record's `first_name` and
`last_name`, joined once in `scripts/10_predict_slate.py` and merged forward
by the pricing step. The publisher reads it and never builds it.

It is not a sportsbook's spelling of the player, and it could not be:
`normalize_props` carries `player_id` and no name, so there is no vendor text
in the frame at all. It is also never derived from the identifier — a name
that is missing, null, blank, whitespace-only, or shaped like an identifier
(`12345`, `P12345`, `Player 12345`, `player_id 12345`, `Unknown Player`)
refuses the publication. None of those shapes can come from a roster that
answered, so one appearing means the identity was never resolved, and a row
number is not a person.

## Modes

Two, switched independently in `ops/wizardofodds_publishing_switch.json`.

### `live` — DISABLED, and not enableable from here

Enabling it needs three independent things, the same shape the shadow
publishing switch already uses: the committed state flipped in a reviewed
commit, `NBA_PROP_WOO_PUBLISH` set to that same state, and
`NBA_PROP_WOO_PUBLISH_APPROVAL` set to the switch's own approval token. No
single mistake publishes.

Even with all three, live publication refuses — and it now refuses for the
one reason that was always the real one. The address is no longer missing.
Both live destinations are configured by name below, with the same host, user,
credential variable and staging root the canary one directory below already
proves. What is missing is a decision, and the two things that carry it:

- the reviewed decision to publish the incumbent on wizardofodds.com at all,
  which no canary can stand in for;
- the approval token that decision would set `modes.live.approval_token` to;
- a successful live proof against that path, which cannot exist before the
  decision above is taken.

Three questions left that list when the live destinations were configured, and
they left it by being answered rather than deferred: the live feed is served
from the NBA target's own root, it is addressed through a current pointer
exactly as the canary is, and two releases are retained so the rollback target
survives. Those were facts about the address, and an address is something this
repository can hold.

The refusal names what remains rather than saying "not configured", because
the difference between a missing line of config and a missing agreement is the
difference between finishing and being blocked. An activation now supplies an
approval. It writes no code and invents no destination.

### `canary` — ENABLED, and safe by construction

A canary nobody can run proves nothing, so this mode is enabled. It is safe
because of what exists rather than because of what is permitted: a canary
against the real host can only address paths beneath its own non-live
subdirectory of that host's NBA target, and a canary whose destination
resolves to *any* live destination's served root is refused by name.

Configuring the live destinations changed none of that, and tightened one part
of it. The refusal used to compare against the single unnamed live slot; it
now compares against every live root the configuration declares, named ones
included, so the arrival of a live address made the canary's own containment
check stricter rather than looser.

## Destinations: one by mode, one by name

A **mode** decides what a run may do. A **name** decides which destination of
that mode it does it to. The two are checked against each other, so a
destination built as a canary is never reinterpreted as a live one.

### Unnamed — a local scratch directory

```
python ops/publish_incumbent_to_wizardofodds.py \
  --mode canary \
  --slate-date 2026-10-20 \
  --data-root "$NBA_PROP_DATA_DIR" \
  --destination-root /tmp/woo-canary \
  --receipt-path /tmp/publication.json
```

The `local_directory` transport, under a root supplied per run and
deliberately not committed: it is a scratch location, and committing one would
be a destination this repository asserted without being told. This is what
proves the publisher's own rules, and it reaches nothing.

### `wizardofodds_nba_canary` — the real destination, over `ssh_rsync`

```
NBA_WOO_SSH_PRIVATE_KEY="$(cat /path/to/key)" \
python ops/publish_incumbent_to_wizardofodds.py \
  --mode canary \
  --destination-name wizardofodds_nba_canary \
  --slate-date 2026-10-20 \
  --data-root "$NBA_PROP_DATA_DIR" \
  --receipt-path /tmp/publication.json
```

| fact | value |
| --- | --- |
| transport | `ssh_rsync` |
| user | `nba-prop-deploy` |
| port | 22 |
| credential | read from `NBA_WOO_SSH_PRIVATE_KEY` |
| private staging | `/var/www/.nba-prop-quant-deploy` (mode 700, unserved) |
| served target | `/var/www/sportsodds/tools/odds-scanner/predictions/nba` |
| canary subpath | `canary` |
| retained releases | 2 |

Reaching a real host is therefore something an operator asks for by name,
never something a default does — an unnamed destination asking for a remote
transport is refused.

**No credential is committed.** The destination records the *name* of the
environment variable its private key is read from; the key reaches the
publisher only through that variable, is written to a file mode 600 in a 700
temporary directory outside the checkout, and is deleted when the transport
closes.

**The subpath is the confinement.** Every path this destination addresses is
resolved beneath `<served target>/canary`, and a relative path that is
absolute, empty, or contains `..` is refused on the way in. The legacy NBA
pages and feeds — `nba-props.html`, `nba-live-props.html`,
`nba-pmf-research.html`, `nba_props_today.json`, `pmf_research.json` — sit in
the *parent* directory and are not expressible through this destination. A
canary that is non-live because of where it *can* write is a stronger
statement than one that is non-live because of what it was asked to write.

### `wizardofodds_nba_live` and `wizardofodds_nba_player_props_page_live`

Fully addressed, fully unreachable. These are the two destinations a reviewed
activation would publish to: the feed at the NBA target's own root, and the
NBA Player Props page at the directory a reader actually opens.

| fact | live feed | live page |
| --- | --- | --- |
| served target | `/var/www/sportsodds/tools/odds-scanner/predictions/nba` | `/var/www/sportsodds/tools/odds-scanner/predictions/NBA/player-props` |
| public URL | `…/predictions/nba/` | `…/predictions/NBA/player-props/` |
| subpath | none — this *is* the served root | none |
| transport | `ssh_rsync` | `ssh_rsync` |
| user | `nba-prop-deploy` | `nba-prop-deploy` |
| credential | `NBA_WOO_SSH_PRIVATE_KEY` | `NBA_WOO_SSH_PRIVATE_KEY` |
| private staging | `/var/www/.nba-prop-quant-deploy` | `/var/www/.nba-prop-quant-deploy` |
| retained releases | 2 | 2 |
| reachable | **no** — `modes.live.state` is `DISABLED` | **no** |

Every value in that table is the canary's value with the confinement removed,
which is the point: the canary proves the transport, the credential, the
staging directory and the public readback against the same host, so what an
activation changes is which directory the identical bytes land in.

**A live destination has no subpath, and that is enforced rather than
conventional.** A subpath is required of a canary and *refused* of a live
destination, because a live destination that published one directory below the
served root would be activating somewhere other than the page or feed a
reader fetches — a live publication nobody can see is a worse outcome than a
refusal. `assert_live_destination_is_the_served_root` checks the resolved
result, so the rule holds whichever way the record is written.

The staging directories do not collide. A confined destination stages under
its confinement and an unconfined one stages under its mode, so the live feed
stages beside the canary feed rather than into the directory above both.

### How a remote release is written

SSH and rsync, and nothing else. No FTP, no FTPS, no SFTP library, no HTTP
API and no object store: each would be a second contract to write, test and
review, and the destination speaks this one.

1. **Stage.** rsync puts the release in the private staging directory, which
   is mode 700 and outside every document root. No `--partial`, so an
   interrupted transfer leaves the staging path absent rather than short.
2. **Verify where nobody can read it.** The destination is asked for its own
   `sha256sum`. A transfer that ended early, arrived altered or went somewhere
   other than where it was addressed is caught here, while the bytes are still
   unreachable.
3. **Place.** A verified staging file is copied to a temporary name in the
   served directory and renamed, which is atomic within one filesystem.
4. **Verify what is served.** The digest is checked again at the served path,
   because the digest that matters is of what is served rather than of what
   was uploaded.
5. **Read it back publicly.** The release is fetched over its public HTTPS URL
   and the digests and row count compared. This is the only check that speaks
   for the destination *as a destination*: a file can be correct on disk and
   still be unreachable, misaddressed, or shadowed by a web server rule.

Promotion happens after all of that.

### Retention

The release served, the one before it, and the receipts. Every other release
payload is pruned once the pointers have rotated past it — pointer-driven, and
only after promotion, so nothing reachable is ever pruned. The audit log and
the manifests are kept regardless, so what was published stays readable after
the bytes it describes are gone.

A dated snapshot accumulating per day is how the legacy NBA output became
something nobody could reason about, which is why this is a fixed two rather
than a date window.

## Releases and rollback

A release is written whole, verified, and only then promoted.

```
<destination root>/
    releases/<release_id>/rows.json       the published rows
    releases/<release_id>/manifest.json   provenance, row count, digests
    current.json                          which release the destination serves
    previous.json                         the rollback target
    canary_attestation.json               canary only: what this canary proved
    audit_log.jsonl                       every action that changed this tree
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

Every completed action appends one line to `audit_log.jsonl` at the
destination, including a rerun that promoted nothing. It is written at the
destination rather than only to the operator's terminal, because a receipt
that exists on whichever machine happened to run the publisher is not a record
anybody looking at the destination can find. Refusals append nothing: a
refused run writes no bytes at all, and the operator's own receipt
(`--receipt-path`) carries the reason.

### Activating — what `--mode live` is gated on

Activation is `--mode live`, and seven independent facts have to hold, all
checked **before a byte is written**:

1. the mode is named explicitly — the default is `canary`;
2. the committed switch enables `live`;
3. the serving receipt's authority is the incumbent;
4. `NBA_PROP_WOO_PUBLISH` and `NBA_PROP_WOO_PUBLISH_APPROVAL` match the
   switch's state and its approval token;
5. a successful canary of **this same source artifact** is attested, via
   `--canary-attestation` naming the `canary_attestation.json` that canary
   wrote;
6. the release verifies at the destination after being written back;
7. a rollback target is already at the destination.

The canary gate matches on two digests: the source artifact's, which says the
canary ran on the same priced file, and the published rows', which says that
file still renders to the same rows. Either alone would let an activation
inherit a canary that proved a different thing. Only a canary writes an
attestation — an activation does not — so a live release can never authorise
the next one.

The rollback gate means no activation is the step that leaves a destination
with nowhere to go back to. That includes the first one: **before a destination
is activated against for the first time, its rollback target has to be seeded
from whatever that destination is already serving.** This is an operator
precondition, not something this repository can supply, and it is deliberate —
the alternative is special-casing the first activation, which is exactly the
activation most likely to need rolling back.

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
silently landed somewhere else would be worse than staying put. The restored
release, the one it superseded and the digest it was verified against are
appended to the destination's audit log.

## What this is not

- It is **not** an activation path, even though it is now wired into
  `.github/workflows/nba_production_lifecycle.yml`. The lifecycle publishes a
  canary feed and the page built from it on every production day, through
  `ops/run_wizardofodds_publication.py` — see
  `docs/wizardofodds/PRODUCTION_AUTOMATION.md` §13. It asks for the canary
  mode by name, and a test pins that the workflow names neither of the two
  activation variables nor a live mode anywhere, so no scheduled run can
  become an activation nobody reviewed.
- It is **not** a promotion path. The published authority is the incumbent and
  the candidate remains shadow-only, both recorded in the switch and asserted
  in the manifest of every release.
- It does **not** change the serving step. `run_incumbent_production_serving.py`
  is byte-identical to production and its docstring still says "It does not
  publish." A publisher that needed serving adjusted would be a publisher that
  was not reading the already-approved artifact.

## Before live publication

In addition to the three switch conditions:

1. the decision listed above, taken by whoever owns the destination — the
   transport, the credentials and the addresses are no longer among them;
2. ~~a live destination configured from those decisions~~ — **done**:
   `wizardofodds_nba_live` and
   `wizardofodds_nba_player_props_page_live` are configured by name;
3. a canary against the real destination, with both the readback and the
   public readback verified there, whose `canary_attestation.json` is then
   handed to the activation. One has now been run: see the
   `ssh_rsync` canary evidence on
   `/var/www/sportsodds/tools/odds-scanner/predictions/nba/canary`;
4. a rollback target seeded at the live destination from whatever it is
   already serving;
5. the activation SHA and activation time recorded, as
   `research/final_model/live_shadow_promotion_policy.json` requires of any
   WizardOfOdds activation.
