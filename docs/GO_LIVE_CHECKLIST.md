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

### What these mean

**Canary mode is ENABLED, and that is safe.** A canary is a preview. It
publishes the exact same files the live page would get, but into a subfolder one
level below the real page. Readers never see it. It cannot reach the live page
even by accident: every path a canary can address resolves beneath its own
subfolder, and the publisher refuses any canary whose destination resolves to a
live one.

**Live mode is DISABLED.** This is the real public page a reader opens. Nothing
has ever been published there.

**The approval token reads `no-approval-has-been-granted`.** This is not a
placeholder that someone forgot to fill in. It is the switch stating that the
decision to publish has not been made.

**Authority is `shadow_only`.** The newer candidate model runs, but its numbers
are never published. The approved incumbent model is the only thing that could
ever go to the page.

**The incumbent has not published a slate to the live NBA Player Props path.**
In the switch's own words: *"The incumbent has not yet published a slate to the
live NBA Player Props path. Enabling this is a separate reviewed decision taken
after a real natural-slate proof."*

---

## Section 2 — Three requirements before live

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

## Section 3 — Not a blocker

**The server connection is already wired. Transport is not what is missing.**

| Thing | Status |
| --- | --- |
| SSH key `NBA_WOO_SSH_PRIVATE_KEY` | **Already present** |
| Server hostname | Configured |
| Server username | Configured |
| Destination folders | Configured |
| Public URLs | Configured |

All four destinations — the live feed, the live page, and a preview for each —
are fully described in `ops/wizardofodds_destinations.json`. The addresses are
settled, and the preview proves the same path one directory below.

As the configuration itself puts it: *"an activation supplies an approval rather
than any new code or configuration."*

---

## Section 4 — Bottom line

**Enabling live is a human decision. No code change is missing.**

Everything needed to publish is built, wired and tested. What is absent is the
judgement call — *do we want this model's numbers on the public site?* — and the
three acts in Section 2 that record it.

No automated check can make that call, and nothing here will make it by
accident:

- The daily production automation publishes a preview every production day. It
  has no access to the approval variables and names no live mode, so **no
  scheduled run can become a publish nobody reviewed.**
- If the switch file were deleted, every mode reads as disabled. The file says:
  *"The absence of a prohibition is not an approval."*
- A live destination is forbidden from having a subfolder. This is enforced, not
  a convention.

---

## One caveat worth knowing before you publish

The three steps above are complete and accurate for turning live publishing on.
But if the intention is to publish the **bias-fix corrected** model that landed
on 2026-10-21, one engineering task is outstanding.

Recorded in
`models/frozen_manifests/nba_prop_quant_v2_adaptive_serving_source_contract.json`:

> `scripts/19_build_wizardofodds_runtime_bundle.py` still verifies serving
> sources against the historical architecture reference only, so building a
> runtime bundle from this corrected source would fail its frozen-source check.

In plain terms: the tool that packages the model for the website checks against
an older approved file list, and the corrected file is not on it. Updating that
tool is a known, deliberately deferred task.

Publishing is still possible without it — but it would publish the pre-fix
behaviour, which is the behaviour that produced the heavy under bias.

---

## Where to read more

| Topic | File |
| --- | --- |
| The publisher in full detail | `docs/wizardofodds/INCUMBENT_PUBLISHER.md` |
| Daily automation | `docs/wizardofodds/PRODUCTION_AUTOMATION.md` |
| What may be claimed publicly | `docs/wizardofodds/PUBLIC_CLAIM_POLICY_V1.md` |
| The on/off switch | `ops/wizardofodds_publishing_switch.json` |
| Destinations and public URLs | `ops/wizardofodds_destinations.json` |
