# NBA Prop Quant — Go Live Checklist for Wizard of Odds

Plain-language guide to what stands between this model and a live public page on
wizardofodds.com. No technical background assumed.

---

## Section 1 — Current state

| What | Status |
| --- | --- |
| Canary (preview) mode | **ENABLED** — but writes only to a non-live destination |
| Live mode | **DISABLED** |
| Live approval token | `no-approval-has-been-granted` |
| Model authority | `shadow_only` |
| Slate published to the live NBA Player Props path | **Never** |
| Slate published to the preview path | **Not yet either** — see Section 3 |

### What these mean

**Canary mode is ENABLED, and that is safe.** A canary is a preview. It is
described in full in Section 2.

**Live mode is DISABLED.** This is the real public page a reader opens. Nothing
has ever been published there.

**The approval token reads `no-approval-has-been-granted`.** This is not a
placeholder that someone forgot to fill in. It is the switch stating that the
decision to publish has not been made.

**Authority is `shadow_only`.** The newer candidate model runs, but its numbers
are never published. The approved incumbent model is the only thing that could
ever go to the page.

**The incumbent has not published a slate to the live NBA Player Props path.**
That sentence is the switch's own, and this is the exact wording it uses in
`ops/wizardofodds_publishing_switch.json`:

> "The incumbent has not yet published a slate to the live NBA Player Props
> path. Enabling this is a separate reviewed decision taken after a real
> natural-slate proof. The destination, its transport and its credential
> variable are no longer the obstacle: they are configured on the named live
> destinations and proven one directory below by the canary. What is missing is
> the decision itself and the two approval variables that carry it, so an
> activation supplies an approval rather than any new code or configuration."

---

## Section 2 — What the preview actually does

This is the part worth understanding before anything else, because it is what
makes the live step small.

**The preview writes the identical bytes, from the identical builder, one
directory lower.**

| | Preview | Live |
| --- | --- | --- |
| Files written | `index.html`, `page_manifest.json` | the same two files |
| Built by | the same page builder | the same page builder |
| Server, user, login key | the same | the same |
| Where it lands | `.../NBA/player-props/canary/` | `.../NBA/player-props/` |

In the configuration's own words, for the live page:

> "The page is built from a live feed release and never from fixture or proof
> data; the canary preview one directory below publishes the identical bytes
> from the identical builder, so what an activation changes is which directory
> they land in."

### Why the preview cannot leak onto the live page

The preview destination carries a setting called `canary_subpath`, whose value is
the single word `canary`. Every path the preview can address is resolved
**beneath** that folder. There is no way to address the folder above it.

In the configuration's own words, for the preview page:

> "… every path this destination can address resolves beneath the
> `canary_subpath`, so the final live index one directory up at
> `NBA/player-props/` is not addressable from here and cannot be written by
> mistake."

Three separate guards, all enforced in code rather than by convention:

- A preview can only write below its own `canary/` folder.
- The publisher refuses any preview whose destination resolves to a live one.
- A live destination is forbidden from having a subfolder of its own.

**So readers never see the preview, and the preview can never become the page.**

---

## Section 3 — What has actually happened so far

The preview machinery runs **every production day** and has so far published
**nothing**, because no slate has reached it. This matters, because the switch
quoted above expects a "real natural-slate proof" before live is considered, and
that proof has not happened yet.

What the runs report, in their own words:

| Date | Preview outcome | Reason given |
| --- | --- | --- |
| 16–19 Oct | `NOT_SERVED_NOTHING_TO_PUBLISH` | preseason — "no regular-season slate exists to serve" |
| 20 Oct (morning) | `NOT_SERVED_NOTHING_TO_PUBLISH` | "the incumbent recorded NO_GAMES_ON_SLATE … so there is no priced artifact to publish" |
| 20–21 Oct (evening) | step failed before publishing | "the morning production lifecycle has not proven a verified incumbent runtime … so the runtime it verified is not the code about to serve" |

**None of these is a transport failure and none is a permission failure.** Every
one is the pipeline declining to publish something it had not produced. That is
the designed behaviour. But it does mean the honest answer to *"has the preview
proved the path?"* is **not yet** — the address and the login are proven, a real
slate landing there is not.

---

## Section 4 — Three requirements before live

All three are required. **If any one is missing or mismatched, publication
refuses and writes nothing at all.** That is intended behaviour, not a fault.
Three separate places, deliberately, so that no single mistake can publish.

### Step 1 — Enable live mode in a reviewed commit

- **File:** `ops/wizardofodds_publishing_switch.json`
- **Change:** set `modes.live.state` from `"DISABLED"` to `"ENABLED"`
- **Also:** replace the `modes.live.approval_token` value
  `"no-approval-has-been-granted"` with the real token the decision chooses
- **How:** committed and reviewed through a normal pull request
- **Who:** a code owner
- **Why a commit:** so the decision is permanently on the record with a name and
  a date against it, rather than typed into a terminal and forgotten

### Step 2 — Set the publish environment variable

- **Variable:** `NBA_PROP_WOO_PUBLISH`
- **Set to:** the same state as the switch
- **Status now:** not set
- **Who:** the operator running the publish

### Step 3 — Set the approval environment variable

- **Variable:** `NBA_PROP_WOO_PUBLISH_APPROVAL`
- **Set to:** exactly the approval token chosen in Step 1
- **Status now:** not set
- **Who:** the operator running the publish

---

## Section 5 — Not a blocker

**The server connection is already wired. Transport is not what is missing.**

| Thing | Status |
| --- | --- |
| SSH key `NBA_WOO_SSH_PRIVATE_KEY` | **Already present** |
| Server hostname | Configured |
| Server username | Configured |
| Destination folders | Configured |
| Public URLs | Configured |

The key is confirmed present, not assumed: the production run on 21 October
shows it handed to the publishing step as a masked secret, exactly as the
known-present `BDL_API_KEY` is. The same key serves both the preview and the
live page — they are the same server and the same login.

All four destinations — the live feed, the live page, and a preview for each —
are fully described in `ops/wizardofodds_destinations.json`.

As the configuration itself puts it: *"an activation supplies an approval rather
than any new code or configuration."*

---

## Section 6 — Bottom line

**Enabling live is a human decision requiring three sign-offs. No code change is
missing for the switch itself.**

The judgement call — *do we want this model's numbers on the public site?* — is
what is absent, along with the three acts in Section 4 that record it.

No automated check can make that call, and nothing here will make it by
accident:

- The daily production automation runs the preview stage every production day.
  It has no access to the approval variables and names no live mode, so **no
  scheduled run can become a publish nobody reviewed.**
- If the switch file were deleted, every mode reads as disabled. The file says:
  *"The absence of a prohibition is not an approval."*
- A live destination is forbidden from having a subfolder. This is enforced, not
  a convention.

---

## Two things to settle before you take the decision

Section 4 is complete and accurate for turning live publishing on. Two separate
matters sit in front of it.

### 1. The packaging tool cannot build a bundle from production

Recorded in
`models/frozen_manifests/nba_prop_quant_v2_adaptive_serving_source_contract.json`:

> `scripts/19_build_wizardofodds_runtime_bundle.py` still verifies serving
> sources against the historical architecture reference only, so building a
> runtime bundle from this corrected source would fail its frozen-source check.

In plain terms: the tool that packages the model for the website checks every
file against an older approved list, and refuses anything that differs.

**It is worse than the note suggests, and it is not caused by any recent fix.**
Running the packaging tool against production fails on three files:

```
FAIL: runtime bundle not built.
Frozen mathematical model source verification failed:
  scripts/10_predict_slate.py: differs from frozen commit 4def8ad33ccc
  scripts/11_collect_snapshots.py: differs from frozen commit 4def8ad33ccc
  scripts/15_price_markets.py: differs from frozen commit 4def8ad33ccc
```

Those three files and the packaging tool are byte-identical on the production
branch. So **no runtime bundle can be built from production at all today.** Every
file the packaging tool checks has already been authorised to differ in a
separate register that the tool does not read.

This is an engineering task, not a decision, and it is the **first** blocker.

A fix for it was written and tested, then deliberately backed out. The
repository's own branch-safety rules forbid it on that lineage: `scripts/` is an
undeclarable path, and the research package is banned from even naming the
packaging tool. Those rules exist to keep unreviewed code off the publishing
path, so the fix needs a reviewed change on a lineage permitted to touch that
surface rather than a way around the rule.

### 2. The preview has not yet proved the path with a real slate

Section 3 above. The switch names a "real natural-slate proof" as what precedes
the live decision, and no slate has landed at the preview path yet. Enabling live
before that would mean the first bytes ever written to the public page are also
the first bytes the pipeline has ever successfully published.

---

## Where to read more

| Topic | File |
| --- | --- |
| The publisher in full detail | `docs/wizardofodds/INCUMBENT_PUBLISHER.md` |
| Daily automation | `docs/wizardofodds/PRODUCTION_AUTOMATION.md` |
| What may be claimed publicly | `docs/wizardofodds/PUBLIC_CLAIM_POLICY_V1.md` |
| The on/off switch | `ops/wizardofodds_publishing_switch.json` |
| Destinations and public URLs | `ops/wizardofodds_destinations.json` |
