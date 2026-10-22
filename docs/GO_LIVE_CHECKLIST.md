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
| Record of model accuracy against outcomes | Built, wired into the daily run, **no rows yet** |
| Record of model accuracy against the market | Built, wired into the daily run, **no rows yet** |

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
| 21 Oct (morning) | nothing published | the run **failed at candidate registration**, before serving — see below |

**Only the last of these was a defect, and it has been fixed.** The others are
the pipeline declining to publish something it had not produced, which is the
designed behaviour.

### The 21 October failure, and why it mattered

The 21 October production run died with:

```
"error": "CandidateIncomplete",
"message": "candidate failed computed validation checks: calibration_valid:
            raw-routed props that acquired a calibration digest: blocks, steals"
```

The frozen route table routes `blocks` and `steals` **raw**, meaning the daily
fit deliberately does not calibrate them. It wrote a short record saying so, and
then fingerprinted every record it had written — including those two. The
validation check reads a fingerprint's presence as proof that a prop *was*
calibrated, so it correctly refused a candidate for a change of method that had
never happened.

**The damage was not confined to the candidate.** The candidate publishes
nothing — it is shadow-only. But the step that registers it is not allowed to
fail quietly, so its refusal skipped everything after it: serving the slate,
publishing the preview, and grading the previous day. A healthy incumbent did
not serve, on a day it had nothing wrong with it.

This is fixed. The fingerprint map now excludes raw routes, and two tests drive
the production fitter itself and reproduce the exact error above without the fix.

---

## Section 4 — What it takes to go live

The switch names three requirements. **There is a fourth**, and it is the one
most likely to be missed, so all four are listed here.

Each of the first three is required on its own. **If any is missing or
mismatched, publication refuses and writes nothing at all.** That is intended
behaviour, not a fault. Three separate places, deliberately, so that no single
mistake can publish.

### Step 1 — Enable live mode in a reviewed commit

- **File:** `ops/wizardofodds_publishing_switch.json`
- **Change:** set `modes.live.state` from `"DISABLED"` to `"ENABLED"`
- **Also:** replace the `modes.live.approval_token` value
  `"no-approval-has-been-granted"` with a real token the decision chooses
- **How:** committed and reviewed through a normal pull request
- **Who:** a code owner
- **Why a commit:** so the decision is permanently on the record with a name and
  a date against it, rather than typed into a terminal and forgotten

**Replacing the token is not cosmetic.** The code does not treat
`no-approval-has-been-granted` as a special case — it is already a non-empty
string, and the gate only checks that the environment matches whatever the file
says. Leave it in place and anyone who knows that phrase, which is published in
this repository, satisfies Step 3. Choosing a new token is what makes Step 3 an
approval rather than a formality.

### Step 2 — Set the publish environment variable

- **Variable:** `NBA_PROP_WOO_PUBLISH`
- **Set to:** the literal word `ENABLED`, matching the switch exactly
- **Not accepted:** `enabled`, `true`, `1`, or anything else — it is a
  character-for-character string comparison against the committed file
- **Where:** a GitHub repository **variable**, the way `NBA_PROP_DATA_DIR` is set
- **Status now:** not set
- **Who:** whoever administers Actions variables

### Step 3 — Set the approval environment variable

- **Variable:** `NBA_PROP_WOO_PUBLISH_APPROVAL`
- **Set to:** exactly the approval token chosen in Step 1
- **Where:** a GitHub repository **secret**, the way `NBA_WOO_SSH_PRIVATE_KEY`
  is set — this string is the approval, so it should not be readable
- **Status now:** not set
- **Who:** whoever administers Actions secrets, ideally not the Step 1 reviewer

### Step 4 — Ask the publisher for live mode

The first three open the gate. **They do not walk through it.** The daily
automation asks for `--mode canary` in both publication steps and hands them
neither approval variable, so all three sign-offs could be complete and the
preview would still be the only thing published.

**This is on purpose, and it is enforced.** `ops/validate_workflows.py` refuses
any workflow line naming `NBA_PROP_WOO_PUBLISH`, `NBA_PROP_WOO_PUBLISH_APPROVAL`,
`--mode live` or `--stage live`, in its own words because

> Live publication is a separate reviewed decision taken through the committed
> switch and two environment variables. None of them belongs in a workflow file:
> a schedule that carried the approval would make every scheduled run an
> activation.

So the live steps are deliberately **not** pre-wired, and nothing but a reviewed
human commit can wire them. That commit has to do two things together, which is
the review gate working as designed:

1. Add live publication steps to `.github/workflows/nba_production_lifecycle.yml`,
   carrying both approval variables. They must be **added** beside the canary
   steps, not edited into them: that file is additive-only, and
   `test_a_declared_workflow_change_only_ever_adds_steps` requires every line
   production already has to survive.
2. Amend `PUBLICATION_ACTIVATION_TOKENS` in `ops/validate_workflows.py` so the
   guard permits what the decision has authorised.

A reviewer who sees step 2 in a diff knows exactly what is being asked for.
That is the entire point of the guard.

### Verifying the gate without opening it

`python ops/validate_workflows.py` must pass, and the gate itself can be
exercised against a scratch copy of the switch without touching the committed
one. Every partial combination refuses:

| What is supplied | Result |
| --- | --- |
| Switch flipped, no variables | Refused |
| Switch flipped, state variable only | Refused |
| Switch flipped, approval variable only | Refused |
| Both set, token wrong by one character | Refused |
| All three aligned | **Allowed** |

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
missing for the switch itself.** The fourth step above is a consequence of the
decision, not a precondition for it: it cannot be written until the decision
exists, because the guard refuses it.

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

## What sits in front of the decision

Section 4 is complete and accurate for turning live publishing on. Three
separate matters sit in front of it, in this order.

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

The 21 October defect that stopped a healthy slate from serving is fixed, so the
next production day with games on it is the first real chance at this proof.

### 3. The record against the market has only just started existing

**This is now built, and it was not before.** Every published row has always
carried the market's own de-vigged probability for the same wager, but nothing
ever scored it: grading measured the model against outcomes and stopped there.

That mattered because `docs/wizardofodds/PUBLIC_CLAIM_POLICY_V1.md` — the
document governing what the site may say — defines every one of its
classifications in market-relative terms, and requires the model's prediction
and the market comparison to come from the identical capture.

The daily grading loop now carries the market probability into each grade row
and scores it on the identical outcome, side and convention, reporting the
Brier and log-loss deltas as model minus market.

**What that does not do is produce a record retroactively.** The policy's
minimum prospective sample is:

| Claim | Minimum |
| --- | --- |
| Any upgrade past "Prospective validation in progress" | 300 games and 5,000 graded contracts |
| A claim about one specific prop | 150 games and 1,000 graded contracts for that prop |

Nothing has been graded yet, because nothing has served yet. The sample starts
accumulating on the first production day with games, and until it reaches those
minimums the only label the site may carry is **"Prospective validation in
progress."**

### What the one existing market comparison says

Measured retrospectively at Gate 2 certification on 2025 development data.
Lower Brier is better, so a negative delta is the model ahead:

| Prop | Served model | De-vig market | Model minus market |
| --- | --- | --- | --- |
| steals | 0.238054 | 0.241501 | **−0.003447** |
| points_assists | 0.249964 | 0.250294 | −0.000330 |
| points_rebounds | 0.249483 | 0.249508 | −0.000025 |
| rebounds_assists | 0.248511 | 0.248421 | +0.000089 |
| points_rebounds_assists | 0.249824 | 0.249433 | +0.000391 |
| blocks | 0.202499 | 0.202036 | +0.000464 |
| points | 0.249974 | 0.249130 | +0.000843 |
| rebounds | 0.246543 | 0.245269 | +0.001273 |
| assists | 0.246038 | 0.242139 | +0.003898 |
| threes | 0.240135 | 0.236676 | +0.003459 |

The model beats the market meaningfully on **steals** and nowhere else. The two
combination props it edges are smaller than the measurement's own noise. On
points, assists and threes the market wins, and the game-clustered bootstrap
puts those outside the 95% interval, so they are real rather than sampling
accidents.

Gate 2's own summary states the limit: *"No Gate 2 result constitutes
prospective market superiority."* That is why the prospective record above has
to exist before any claim is made, and why the claim policy forbids the words
"profitable model", "beats sportsbooks" and "guaranteed edge" without it.

---

## Where to read more

| Topic | File |
| --- | --- |
| The publisher in full detail | `docs/wizardofodds/INCUMBENT_PUBLISHER.md` |
| Daily automation | `docs/wizardofodds/PRODUCTION_AUTOMATION.md` |
| What may be claimed publicly | `docs/wizardofodds/PUBLIC_CLAIM_POLICY_V1.md` |
| The on/off switch | `ops/wizardofodds_publishing_switch.json` |
| Destinations and public URLs | `ops/wizardofodds_destinations.json` |
