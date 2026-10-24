"""Containment contract for the shadow latent-state branch.

SHADOW / RESEARCH ONLY.

This module is the single declaration of which repository paths decide
production behaviour. It only ever *reads* git: nothing here writes, and the
path lists exist so that both the validation run and the branch-safety tests
check containment against the same definition rather than two drifting copies.

Gate H is this file plus :func:`modified_production_paths` returning an empty
list.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

PRODUCTION_REF = "production/wizardofodds-integration"

#: Directory prefixes whose bytes decide production behaviour: the scheduled
#: automation, the frozen configs and model artifacts, the operational
#: runbooks, and the production scripts the pipeline executes.
PROTECTED_PRODUCTION_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "configs/",
    "models/",
    "ops/",
    "scripts/",
    "docs/",
    "release/",
    "review/",
)

#: The scripts that serve a slate. What they can reach is the served model.
SERVING_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/10_predict_slate.py",
    "scripts/15_price_markets.py",
)

#: Serving entry points a declaration may name, and only additively.
#:
#: ``scripts/`` was undeclarable as a flat prefix, which is the same mistake
#: the source partition above already corrected once: it read the same for two
#: different kinds of change, the numbers these scripts compute and the paths
#: they resolve. Only the first is what "may not edit the served model" is
#: about, and conflating them made the serving deployment plumbing permanently
#: unfixable. That was not hypothetical. Both scripts verified the frozen
#: manifest against the working directory, so the frozen runtime bundle the
#: lifecycle installs got them past nothing and the first non-empty slate
#: would have failed inside ``10_predict_slate.py`` -- and the only reviewed
#: way in was a declaration these paths could not receive.
#:
#: So they are declarable, under the rule the lifecycle workflow is already
#: held to: every line the production ref has must still be there, so no
#: existing computation, threshold, flag or call can be changed, reordered out
#: of existence or dropped. :data:`NUMERICAL_SURFACE_EXEMPT_FUNCTIONS` names
#: the only functions whose bodies may differ at all, and the branch-safety
#: tests compare every other function's syntax tree against production. The
#: rest of ``scripts/``, every model artifact, every frozen config, the
#: release surface and every production source module the serving path imports
#: all remain undeclarable.
ADDITIVE_ONLY_SERVING_ENTRY_POINTS: tuple[str, ...] = SERVING_ENTRY_POINTS

#: The only functions in a declared serving entry point whose bodies may
#: differ from production. ``parse_args`` is where an argument is declared and
#: ``main`` is where it is used; everything that computes a projection or a
#: price is outside both and is pinned syntax-tree-identical.
NUMERICAL_SURFACE_EXEMPT_FUNCTIONS: tuple[str, ...] = (
    "parse_args",
    "main",
)

#: Serving *sources* a declaration may name, and only additively.
#:
#: The carve-out above fixed where a serving script resolves the frozen
#: manifest. It could not fix what a serving script is given, because that
#: lives in the modules the scripts import and every one of those is
#: undeclarable by :data:`UNDECLARABLE_PRODUCTION_SOURCES`. The same shape of
#: deadlock, one layer in, and the same consequence: a serving-correctness fix
#: with nowhere to go.
#:
#: That was not hypothetical either. ``build_upcoming_slate_features``
#: assembled its rows from scratch and emitted none of the five
#: competition-regime flags the fitted models name in their feature lists,
#: ``assert_model_feature_contract`` is a column-presence check with no
#: allowance, and so every non-empty real slate was refused -- after the
#: frozen manifest had verified, which is to say after the fix above had done
#: its job. Production could install a verified bundle, resolve the incumbent
#: and predict nothing.
#:
#: So ``slate.py`` is declarable, on the entry points' terms and one more.
#: Every line production has must still be there, so no existing feature
#: definition, prior, window or guard can be changed or dropped.
#: :data:`SERVING_SOURCE_EXEMPT_FUNCTIONS` names the only function whose body
#: may differ, and the branch-safety tests compare every other function's
#: syntax tree against production. The additional term is that the module may
#: not grow its own copy of a rule production already states: what the live
#: slate says about a game's competition regime is ``classify_games``'
#: answer, and ``game_context.py`` stays undeclarable, so the only way to
#: change that answer is still a declared change to the classifier itself.
#:
#: The carve-out's second use was the same shape of defect one feature along.
#: ``season_progress`` is also in the fitted models' feature lists, it also
#: had a single canonical definition in ``classify_games`` that
#: ``build_base_frame`` read, and the live builder also stated its own --
#: distinct box-score games over a hard-coded 82, which diverges from the
#: canonical value for any team that has played a Cup Championship, a Play-In
#: game or a playoff game. That one does not fail a contract; it silently
#: serves the models a number they were not fit on, which is worse. The fix
#: reads the classifier's own two progress columns, so the additional term
#: above did the work it exists for: the rule stayed where production states
#: it and only the call into it was added.
#:
#: The carve-out's third use is the same defect read from the other side, and
#: it is the first one that is not in the slate module. The appearance-only
#: priors answer what a player played when he played. ``availability.py``
#: read them as what a player has to vacate, and pooled every OUT player's
#: projected minutes as freed opportunity for his team-mates. For an absence
#: the priors have already absorbed those minutes are not his to vacate: the
#: rotation that played took them, the priors record the rotation that played,
#: and the team-mates receiving them already hold them. On a slate whose
#: absentees are mostly settled -- an opening night, where the injury feed is
#: full of players who have not appeared for months and of players who have
#: never appeared at all -- the freed pool is almost entirely minutes the team
#: never had, and it is handed out on top of priors that already contain them.
#:
#: So ``availability.py`` is declarable, on the same terms, and it is the
#: narrowest file that can state the correction: the pool is summed there and
#: nowhere else, and the only other way to shrink it is to misstate an
#: absentee's projected minutes upstream, which would also move the usage
#: multiplier and would leave the recorded base minutes saying something
#: untrue. :data:`SERVING_SOURCE_EXEMPT_FUNCTIONS` names the one function in
#: it whose body may differ, every line production has must still be there --
#: including the unweighted pool, which is still what runs when the share is
#: absent -- and the elasticity weights, the iteration count, the minutes cap,
#: the OUT zeroing and the usage multiplier are all inside that function and
#: all unchanged.
#:
#: Every other module the serving path imports -- the model, the copula, the
#: marginals, the pricing, the production contract -- remains undeclarable.
ADDITIVE_ONLY_SERVING_SOURCES: tuple[str, ...] = (
    "src/nba_prop_quant/slate.py",
    "src/nba_prop_quant/availability.py",
)

#: The only functions in a declared serving source whose bodies may differ
#: from production. The live slate builder is where a feature the models
#: require is produced; every prior, window, leakage guard and slate-date rule
#: in the same module is outside it and is pinned syntax-tree-identical. The
#: availability adjustment is where an absence is turned into opportunity for
#: somebody else; ``_is_out`` and ``OUT_PATTERNS``, which decide who is absent
#: at all, are outside it and are pinned the same way.
SERVING_SOURCE_EXEMPT_FUNCTIONS: tuple[str, ...] = (
    "build_upcoming_slate_features",
    "apply_current_injury_adjustment",
)

#: What a statement added to an exempt function is allowed to be about.
#:
#: The permission is for named plumbing, not for an editable function. Each
#: added statement has to name one of these, so widening what an exempt
#: function may do means adding a name here and having that reviewed, rather
#: than it following silently from the function already being exempt.
#: ``model_projection`` and the three abbreviations are publication plumbing
#: rather than model plumbing: each one carries a quantity some earlier stage
#: already produced to a stage that was not being given it. The pricing script
#: copies the mean its own existing call priced the line against, and the
#: slate builder copies the two team names and the tip time the schedule
#: payload arrives with. Nothing here computes a projection or decides a
#: matchup, and neither name buys anything beyond carrying one forward.
#:
#: ``history_games``, ``context``, ``flags`` and ``season_progress`` are the
#: other kind, and the kind the carve-out above exists for: each names a
#: feature the fitted models were shown one definition of, which the live
#: builder stated a second definition of. The last two names are that again,
#: and they are about the *population* every one of those definitions is taken
#: over rather than about any one of them. ``build_base_frame`` opens with
#: ``df[df["minutes"].fillna(0) > 0]``, so every rolling prior, decay state
#: and Kalman state the models were fit on is defined over appearances; the
#: live builder was handed BALLDONTLIE's whole stats table, where a rostered
#: player who did not play arrives carrying zero minutes, and averaged each
#: feature over appearances and absences alike. ``played_advanced_rows`` is
#: the same population one table along, the restriction ``build_base_frame``
#: already applies to the advanced history and says why in place. Neither
#: name states a threshold, a window, a weight or a feature definition --
#: there is no new rule here, only training's own population reaching the
#: path that serves it -- and neither can reach a probability, a price, an
#: edge, an EV or a route.
#:
#: ``_prior_window_participation`` and ``prior_participation_rate`` are the
#: population names again, read from the other side. Those same appearance-only
#: priors answer "what did this player play when he played", and the
#: availability layer was reading them as "what is this player's to vacate" --
#: so an absentee the priors had already written out of the rotation freed his
#: minutes a second time, onto team-mates whose priors already held them. The
#: share names which of a team's recent games a player was actually in, over
#: the same window and the same appearance population those priors are built
#: from. It states no threshold and no weight: it is a count of games over a
#: count of games, taken at the span the EWM already uses, and the minutes it
#: governs are the ones the existing elasticity allocation was already
#: handing out.
#:
#: ``NO_APPEARANCE_PRIOR`` is the population names a third time, and the last
#: place that argument reaches: not which rows a prior averages over, nor what
#: an absence frees, but what one of those priors is worth when the player has
#: no appearances to average. ``build_base_frame`` takes ``prior_minutes10``
#: and every ``prior_*_rate10`` through a lagged EWM, so a first NBA
#: appearance is fit with all of them missing and the fitted models hold a
#: learned branch for that row; the live builder substituted the played
#: population's league mean instead. That is not a neutral stand-in. It is
#: what a rotation regular's prior looks like, so a player nobody has seen was
#: scored as one with twenty-three minutes a night behind him. The name states
#: no value of its own -- it is ``nan``, which is what training already puts
#: there -- and it cannot reach a probability, a price, an edge, an EV or a
#: route.
DECLARED_SERVING_PLUMBING_NAMES: tuple[str, ...] = (
    "frozen_bundle_root",
    "history_games",
    "context",
    "flags",
    "season_progress",
    "model_projection",
    "home_abbreviation",
    "visitor_abbreviation",
    "player_abbreviation",
    "tip_off_utc",
    "played_player_games",
    "played_advanced_rows",
    "_prior_window_participation",
    "prior_participation_rate",
    "NO_APPEARANCE_PRIOR",
)

#: The serving *orchestrator* a declaration may name, and only additively.
#:
#: ``ops/run_incumbent_production_serving.py`` is pinned byte-identical to
#: production by the publisher suite's numerical claim, and for a good reason:
#: the publisher consumes the artifact this script produces, so a publisher
#: that needed this script adjusted would be a publisher reading something
#: nobody had approved. That reason is about the publisher, and byte-equality
#: was the right way to say it for as long as no approved change could reach
#: the file.
#:
#: One can. This script is where "does a real slate exist tonight" is decided,
#: and that question had exactly one wrong answer in it: opening day. No
#: current-season game has been played, so the refresh writes nothing and
#: cannot report REFRESHED -- while tonight's games are real and the incumbent
#: must price them. Readiness had to be able to say so, and byte-equality reads
#: that as "a price moved", which is precisely what it is not: nothing here
#: computes a projection or a price. The two scripts that do are declarable
#: only under the carve-out above, and every model artifact and frozen config
#: they read stays undeclarable.
#:
#: So it is declarable, on the same terms as the entry points and the slate
#: builder. Every line production has must still be there, so no existing
#: refusal, authority check or provenance requirement can be changed or
#: dropped. :data:`SERVING_ORCHESTRATOR_EXEMPT_FUNCTIONS` names the only two
#: functions whose bodies may differ, the branch-safety tests compare every
#: other function's syntax tree against production, and what those two may
#: have gained is held to :data:`DECLARED_SERVING_READINESS_NAMES`.
ADDITIVE_ONLY_SERVING_ORCHESTRATORS: tuple[str, ...] = (
    "ops/run_incumbent_production_serving.py",
)

#: The only functions in a declared serving orchestrator whose bodies may
#: differ from production. Readiness is the decision this script makes on its
#: own account, and ``run`` is the one place the run's own arguments exist, so
#: a readiness check that needs to compare the status file against the date
#: this run was asked to serve can be reached from nowhere else. Resolving the
#: incumbent from the registry's promotion state, verifying the frozen bundle,
#: invoking the two serving scripts, building the receipt and refusing it for
#: missing provenance are all outside both and are pinned syntax-tree-identical.
#:
#: ``run`` is the narrowest way to reach the arguments, not a general
#: permission: what it may have gained is held to the same statement-level
#: check as the other exemptions, so it can only have acquired a statement
#: naming something in :data:`DECLARED_SERVING_READINESS_NAMES`.
SERVING_ORCHESTRATOR_EXEMPT_FUNCTIONS: tuple[str, ...] = (
    "slate_readiness",
    "run",
)

#: What a statement added to the readiness decision is allowed to be about.
#:
#: The rule :data:`DECLARED_SERVING_PLUMBING_NAMES` states, for the one
#: exemption whose subject is a decision rather than a quantity. Readiness
#: decides whether a real slate exists, so the way this exemption would widen
#: is by coming to accept a refresh outcome nobody reviewed -- and the defence
#: that keeps working is that each added statement has to name something here.
#:
#: The first two are the opening-day acceptance: the outcome constant, and the
#: function that refuses it unless the classifier's recorded derivation and an
#: independent re-derivation of opening day both hold. The third narrows
#: either acceptance to the slate actually being served, by comparing the date
#: the status file records against the date the run was given. The lifecycle
#: derives both from one value today, so that comparison cannot fail in the
#: scheduled path -- which is the point: it is there for the paths that are
#: not it, a stale or reused status file and a manual invocation, and for the
#: day a workflow change makes the invariant stop holding silently.
DECLARED_SERVING_READINESS_NAMES: tuple[str, ...] = (
    "REFRESH_OUTCOME_OPENING_DAY_NO_REFRESH",
    "_opening_day_readiness",
    "readiness_for_this_slate",
)

#: The *capture* entry point a declaration may name, and only additively.
#:
#: The third instance of the deadlock the two carve-outs above resolved, and
#: the one that had already cost production every slate of the season.
#: ``scripts/11_collect_snapshots.py`` is the sole producer of the T-20
#: point-in-time captures the pricing path reads, nothing in the repository
#: ever invoked it on a schedule, and with no capture present the pricing path
#: reports no games and exits 0. Automating it needs production to be able to
#: ask for the one window it consumes, because the collector captures whichever
#: of the schedule's seven offsets are due and six of them are research
#: windows production never reads.
#:
#: ``scripts/`` was undeclarable apart from the two serving entry points, so
#: that request had nowhere to go, and the alternatives were worse than the
#: carve-out: a second producer would duplicate the capture envelope and drift
#: from the one pricing verifies, and narrowing from outside would mean
#: rebinding the collector's own due-window function at run time, which is the
#: same edit made invisible.
#:
#: So it is declarable, on exactly the entry points' terms. Every line
#: production has must still be there, so no existing API call, component,
#: digest, grace window or envelope key can be changed or dropped.
#: :data:`CAPTURE_ENTRY_POINT_EXEMPT_FUNCTIONS` names the only functions whose
#: bodies may differ, the branch-safety tests compare every other function's
#: syntax tree against production, and what those two may have gained is held
#: to :data:`DECLARED_CAPTURE_SELECTION_NAMES`. The rest of ``scripts/`` --
#: the snapshot cycle driver included, so the research cadence stays where it
#: is -- remains undeclarable.
ADDITIVE_ONLY_CAPTURE_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/11_collect_snapshots.py",
)

#: The deployment-integrity tool under ``scripts/`` that packages the frozen
#: model rather than computing anything with it.
#:
#: ``scripts/`` was undeclarable apart from the serving entry points and the
#: collector, and the same conflation that made those unfixable made this one
#: unfixable too: it reads the same for the numbers a script computes and the
#: bytes a script verifies. This script computes nothing. Its own module
#: docstring states it "performs no training, fitting, selection or
#: recalibration, and it never contacts an external API", a test pins that it
#: cannot reach the network at all, and nothing on the serving path imports
#: it. What it does is verify and archive the already-frozen model, so the
#: lifecycle has a bundle to install.
#:
#: It could not do that. It verified the nine files
#: :data:`DECLARABLE_SERVING_SOURCE_REGISTRIES`' first entry locks against the
#: architecture reference alone, and three of them had been corrected under
#: that very registry, so every build refused. The refusal was unsatisfiable
#: rather than merely strict: two of those corrections add the
#: ``--frozen-bundle-root`` argument without which the entry points resolve
#: the frozen manifest against the working directory, so the only tree able to
#: serve was the one tree it would not package. The registry's own
#: ``known_limitation`` named teaching this script about it as the deferred
#: deployment obligation, and a declaration was the only reviewed way in.
#:
#: So it is declarable, and the carve-out is narrower in risk than the three
#: before it, because the file is not on the serving path and holds no model
#: state. The rest of ``scripts/`` remains undeclarable, and what this script
#: may do with the registry is itself bounded: a library module under ``src/``
#: that the historical runtime contract byte-pins can never be relocked, so
#: the mathematics stays pinned to the architecture reference and no
#: declaration here can move it.
DECLARABLE_DEPLOYMENT_PACKAGING_TOOLS: tuple[str, ...] = (
    "scripts/19_build_wizardofodds_runtime_bundle.py",
)

#: The only functions in the declared capture entry point whose bodies may
#: differ from production. ``parse_args`` is where the offset argument is
#: declared and ``main`` is where it narrows the due list; computing the
#: per-component digests, building the capture id and appending the envelope
#: are outside both and are pinned syntax-tree-identical, which is what keeps
#: the capture format the format every capture ever written was hashed under.
CAPTURE_ENTRY_POINT_EXEMPT_FUNCTIONS: tuple[str, ...] = (
    "parse_args",
    "main",
)

#: What a statement added to the capture entry point is allowed to be about.
#:
#: The rule :data:`DECLARED_SERVING_PLUMBING_NAMES` states, for the exemption
#: whose subject is which windows a scheduled capture records. The way this
#: exemption would widen is by coming to change *what* is captured rather than
#: *which window* it is recorded against, and the defence is that each added
#: statement has to name one of these. The first four are the offset
#: narrowing: the argument under both the spellings it appears in, the
#: validator that refuses an offset the schedule does not define, and the
#: local the due list is filtered by.
#:
#: ``capture_window_claim`` is the same subject read from the other side, and
#: the only name here that is about concurrency rather than selection. Which
#: windows a sweep records was decided by reading ``completed_window_ids``,
#: collecting, and then appending, with nothing holding the capture-run log
#: across the three. Two sweeps whose collections overlap therefore both read
#: a window as unclaimed and both record it, and ``select_capture_bundle``
#: refuses an ambiguous window permanently, because the log is append-only
#: and both captures are genuine. On 2026-10-23 that cost three games every
#: prediction they would have had. The name is an exclusive ``flock`` held
#: across the sweep and nothing else: it captures no component the collector
#: did not already capture, writes no key the envelope did not already have,
#: moves no digest, no grace window and no due-window rule, and decides only
#: which of two simultaneous sweeps is the one that records.
DECLARED_CAPTURE_SELECTION_NAMES: tuple[str, ...] = (
    "--offset-minutes",
    "offset_minutes",
    "selected_offsets",
    "offsets",
    "capture_window_claim",
)

#: The *publication transport* a declaration may name, and only additively.
#:
#: The fourth instance of the deadlock the three carve-outs above resolved,
#: and the one that had already cost opening day. The publisher is pinned
#: byte-identical by the lifecycle's numerical freeze, which is right about
#: what that pin is for -- this module decides which rows are publishable,
#: what a published record contains and which destination is allowed -- and
#: wrong about one thing inside it: the transport that moves the bytes runs
#: on the runner, and the runner's rsync is not the one this was written
#: against. ``--chmod=F600`` is rejected by the macOS runner's rsync while it
#: parses its own arguments, so the transfer failed before it opened a
#: connection and the canary feed could not be published at all. A pin that
#: cannot be moved makes a portability defect in the transport unfixable.
#:
#: So the module is declarable, and only on terms narrower than the byte pin
#: is in the one direction that matters. Nothing that states a number, a
#: schema, a threshold, a row rule or a destination may differ:
#: :data:`PUBLICATION_TRANSPORT_CLASS` is the only class whose methods may,
#: :data:`PUBLICATION_TRANSPORT_EXEMPT_METHODS` names the only method of it
#: that may, every other top-level definition and every module constant is
#: pinned syntax-tree-identical, and anything new is held to
#: :data:`DECLARED_PUBLICATION_TRANSPORT_NAMES`. The published-record
#: versions, the row rules, the pricing fields, the switch and the
#: destinations are all outside the class and are therefore frozen exactly as
#: the byte pin froze them.
ADDITIVE_ONLY_PUBLICATION_TRANSPORTS: tuple[str, ...] = (
    "ops/publish_incumbent_to_wizardofodds.py",
)

#: The only class in the declared publication transport whose methods may
#: differ from production. The remote transport and nothing else: a local
#: directory destination has no rsync to be incompatible with, and the
#: publisher's rules are not in a class at all.
PUBLICATION_TRANSPORT_CLASS = "SshRsyncTransport"

#: The only method of that class whose body may differ from production.
#: ``_upload`` is where the transfer is invoked and where the mode is now set
#: after it; ``_place`` is outside this list on purpose, so the copy into the
#: served tree, the mode it is served with and the rename that makes the
#: placement atomic are all pinned to production.
PUBLICATION_TRANSPORT_EXEMPT_METHODS: tuple[str, ...] = ("_upload",)

#: What a name added to the declared publication transport is allowed to be.
#:
#: The rule :data:`DECLARED_CAPTURE_SELECTION_NAMES` states, for the
#: exemption whose subject is how a file is transferred and what mode it then
#: carries. All three are that: the mode the staged upload is given, the
#: command that performs the transfer, and the command that sets the mode.
#: Nothing here can name a price, a probability, a row, a schema version or a
#: destination, which is what keeps this from becoming a declaration that the
#: publisher is editable.
DECLARED_PUBLICATION_TRANSPORT_NAMES: tuple[str, ...] = (
    "STAGED_FILE_MODE",
    "_rsync_argv",
    "_set_staged_mode",
)

#: The *page consistency check* a declaration may name, and only additively.
#:
#: The fifth instance of the deadlock the four carve-outs above resolved, and
#: the second to cost opening day. The renderer is pinned byte-identical by
#: the lifecycle's numerical freeze, which is right about what that pin is for
#: -- this module decides which published release may become a public claim --
#: and wrong about one thing inside it: the check that a wager's rows state
#: one model probability compared their *serialisations*. A wager several
#: books price is published once per book, every one of those rows carries the
#: same model output, and two arrivals at one double can differ in the bits
#: the format cannot hold. The feed published 603 rows and verified them; the
#: page then refused the whole release because two rows spelled one
#: probability ``0.5899928519036952`` and ``0.5899928519036954``. A pin that
#: cannot be moved makes a defect in the renderer's own validation unfixable,
#: and the defect refuses correct releases.
#:
#: So the module is declarable, and only on terms narrower than the byte pin
#: is in the one direction that matters. Nothing that states a number, a
#: format, a column, a destination or a routing rule may differ:
#: :data:`PAGE_CONSISTENCY_EXEMPT_FUNCTIONS` names the only top-level
#: definition whose body may, :data:`PAGE_CONSISTENCY_CLASS` and
#: :data:`PAGE_CONSISTENCY_EXEMPT_METHODS` name the only card accessor that
#: may, every other definition and every module constant is pinned
#: syntax-tree-identical, anything new is held to
#: :data:`DECLARED_PAGE_CONSISTENCY_NAMES`, and the function that aggregates
#: rows into cards is pinned to production's own tree once the declared
#: plumbing is taken back out of it. The odds conversions, the percentage
#: formatting, the column set, the provenance allow-list, the confinement and
#: the routing call are all outside this and are frozen exactly as the byte
#: pin froze them.
ADDITIVE_ONLY_PAGE_CONSISTENCY_CHECKS: tuple[str, ...] = (
    "ops/wizardofodds_player_props_page.py",
)

#: The only top-level definition in the declared renderer whose body may
#: differ from production. The agreement check and nothing else: the
#: conversions, the formatters, the cell builders and the confinement rules
#: are all separate definitions and all pinned.
PAGE_CONSISTENCY_EXEMPT_FUNCTIONS: tuple[str, ...] = ("_agree_on_the_model",)

#: The only class in the declared renderer whose members may differ. The card,
#: because the one accessor that states the settled probability is on it.
PAGE_CONSISTENCY_CLASS = "PageRow"

#: The only member of that class whose body may differ from production. The
#: market probability, the edge, the expected value, the fair odds, the
#: projection, the matchup and the card ordering are all outside this list, so
#: every number a card shows apart from the model's own probability is pinned
#: to the accessor production has.
PAGE_CONSISTENCY_EXEMPT_METHODS: tuple[str, ...] = ("model_probability",)

#: What a name added to the declared renderer is allowed to be.
#:
#: The rule :data:`DECLARED_PUBLICATION_TRANSPORT_NAMES` states, for the
#: exemption whose subject is whether a release's rows state one number. All
#: four are that: the two constants that say how close two copies of one
#: double must be, the predicate that answers it, and the field that carries
#: the answer onto the card. Nothing here can name a price, a column, a
#: format, a destination or a route, which is what keeps this from becoming a
#: declaration that the renderer is editable.
DECLARED_PAGE_CONSISTENCY_NAMES: tuple[str, ...] = (
    "MODEL_SCALAR_FLOOR",
    "MODEL_SCALAR_ULPS",
    "model_scalar",
    "model_scalars_settled",
)

#: The two registries under ``models/`` that record what serving runs, rather
#: than being what serving runs.
#:
#: ``models/`` is undeclarable because it holds the fitted artifacts and the
#: frozen freeze manifests, and those must not move. These two files are a
#: different kind of thing living in the same directory: the adaptive serving
#: source contract exists to be updated when an approved serving change lands
#: -- its own ``purpose`` requires "every difference from the historical
#: reference to be enumerated and justified here", and two reviewed commits
#: have done exactly that. A lock that cannot be updated when the thing it
#: locks is legitimately corrected is not a lock, it is a dead end, and the
#: branch that corrects serving has to be able to say so.
#:
#: What stays immutable is the evidence, not the record: every
#: ``historical_reference_sha256``, the architecture reference commit and the
#: historical Gate 3 runtime contract are pinned against Git by tests that
#: already exist, and a branch-safety test below pins that no locked file is
#: dropped and no historical hash is rewritten. Every other path under
#: ``models/`` remains undeclarable.
DECLARABLE_SERVING_SOURCE_REGISTRIES: tuple[str, ...] = (
    "models/frozen_manifests/"
    "nba_prop_quant_v2_adaptive_serving_source_contract.json",
    "models/frozen_manifests/nba_prop_quant_v2_adaptive_update_protocol.json",
)

#: Protected sources no declaration may ever name, because the live pricing
#: path reads them.
#:
#: Undeclarability used to be a property of being protected at all, which read
#: the same for two different kinds of module: the ones that decide what a
#: price is, and the ones that orchestrate a fit. Only the first kind is what
#: "may not edit the served model" is about, and conflating them made the fit
#: orchestration permanently unfixable -- a correctness fix to the daily fit's
#: own validation had nowhere to go, since a protected source could not be
#: declared and the declaration mechanism is the only reviewed way in.
#:
#: So the line is drawn where the repository already draws it: a protected
#: source is undeclarable exactly when a serving entry point can reach it by
#: import. That is computed by :func:`serving_reachable_sources` and the
#: branch-safety tests pin this set against it, so the partition cannot drift
#: and an import added to a serving script moves a module into this set rather
#: than leaving it declarable. ``copula.py`` is here because ``model.py``
#: imports it, not because somebody listed it.
UNDECLARABLE_PRODUCTION_SOURCES: frozenset[str] = frozenset(
    {
        "src/nba_prop_quant/api.py",
        "src/nba_prop_quant/availability.py",
        "src/nba_prop_quant/copula.py",
        "src/nba_prop_quant/decay.py",
        "src/nba_prop_quant/distributions.py",
        "src/nba_prop_quant/experience.py",
        "src/nba_prop_quant/features.py",
        "src/nba_prop_quant/game_context.py",
        "src/nba_prop_quant/gate3_v2.py",
        "src/nba_prop_quant/kalman.py",
        "src/nba_prop_quant/model.py",
        "src/nba_prop_quant/normalize.py",
        "src/nba_prop_quant/pipeline.py",
        "src/nba_prop_quant/pricing.py",
        "src/nba_prop_quant/production.py",
        "src/nba_prop_quant/prospective_snapshot.py",
        "src/nba_prop_quant/settings.py",
        "src/nba_prop_quant/slate.py",
        "src/nba_prop_quant/storage.py",
    }
)

#: Production modules that must not change without a declaration. The shadow
#: layer imports and reads them; the whole no-double-count argument rests on
#: ``copula.py`` being untouched.
#:
#: Everything the serving path can reach, plus the two fit-orchestration
#: modules. This used to be eleven hand-written names, which left ten modules
#: the serving scripts import -- ``api.py``, ``normalize.py``, ``settings.py``
#: and ``storage.py`` among them -- outside it. They were covered only
#: incidentally, by a containment check that also refused every unrelated
#: branch, so scoping that check to the shadow lineage had to come with
#: closing this gap properly. A branch-safety test pins that the serving
#: closure stays inside this set, so a new serving import cannot reopen it.
PROTECTED_PRODUCTION_SOURCES: frozenset[str] = UNDECLARABLE_PRODUCTION_SOURCES | {
    "src/nba_prop_quant/adaptive_fit_registry.py",
    "src/nba_prop_quant/adaptive_training.py",
}

#: Namespaces the shadow lineage's own work lives in.
SHADOW_NAMESPACES: tuple[str, ...] = (
    "research/",
    "src/nba_prop_quant/research/",
    "tests/test_game_latent_state_shadow",
)

#: Where tests live. Production cannot reach a test, and ``tests/`` is not a
#: protected prefix, so a test is not part of the surface containment is about.
#: Named here because :func:`shadow_lineage_offenders` reads it.
TEST_SURFACE_PREFIX = "tests/"

#: The containment guard's own source. It declares and checks containment
#: rather than being contained by it, so changing it is not shadow work --
#: every branch that touches a protected path has to edit the declaration, and
#: counting that edit would make the act of declaring turn an unrelated branch
#: into a shadow branch. The literal scan in the branch-safety tests already
#: skips the declaration module for the same reason.
CONTAINMENT_GUARD_PATHS: tuple[str, ...] = (
    "src/nba_prop_quant/research/game_latent_state/safety.py",
    "tests/test_game_latent_state_shadow_safety.py",
    "tests/test_game_latent_state_shadow_bucket_repair.py",
)

#: Kept as a separate name because the literal scan asks a narrower question:
#: which file is allowed to contain protected-path literals.
PATH_DECLARATION_MODULE = "safety.py"


#: Protected paths this integration is declared to change, each with its
#: reason. Gate H's absolute form — no protected path changes at all — held for
#: as long as the dependence work was pure research, and it is the right shape
#: for a research branch. Integrating the work into production is a production
#: change by construction, so the enforced invariant becomes "nothing
#: *undeclared* changes" and the declaration lives here, next to the path lists
#: it qualifies, rather than being implicit in whatever a diff happens to show.
#:
#: Nothing here may be a model artifact, a pricing path or a publishing path.
#: The branch-safety tests pin that, so widening this mapping cannot quietly
#: become permission to edit the served model.
#:
#: A declaration is permission for a *pending* change, so it is removed once
#: that change has landed in production; otherwise it sits here as standing
#: permission to edit a protected path unreviewed. What the lineage owns in
#: production once a change has landed is recorded in
#: :data:`SHADOW_OWNED_PRODUCTION_PATHS`, which grants no permission at all.
#: Empty is the resting state, and the only correct non-empty content is the
#: pending change on the branch being read. A stale declaration is exactly
#: what :func:`stale_integration_declarations` reports, and the branch that
#: leaves one behind is always already merged by the time anything notices, so
#: it breaks the rule for every later branch rather than for the one branch
#: doing something wrong. That happened once with
#: ``ops/evidence/production_shadow_closure.json``, and the
#: pre-opening-day remediation's eleven entries were retired the same way:
#: the eight ops, docs and workflow files it introduced are recorded as owned
#: below, and ``src/nba_prop_quant/adaptive_training.py`` is recorded nowhere,
#: because the remediation changed one block of a production module it does
#: not own and must declare again to touch again. The interpreter guard's
#: spelling fix was retired in turn when it landed, and so were the pre-live
#: closure's five entries: the two graders it introduced are recorded as owned
#: below, and the three files it extended were already owned. The frozen
#: runtime bundle install's four entries were retired the same way when they
#: landed; all four are recorded as owned below. The real-slate serving
#: correction's seven entries were retired in turn: the two ops entry points
#: and the lifecycle workflow among them were already owned, and the two
#: serving scripts and the two serving-source registries are recorded nowhere,
#: because they are pre-existing production files this lineage changed rather
#: than introduced. The live-slate game-flag fix's four entries went the same
#: way when they landed -- the slate module, the prediction entry point and
#: both registries, all pre-existing and so all recorded nowhere -- which is
#: why the season-progress parity fix declared the slate module and both
#: registries again, and declared nothing for the prediction entry point,
#: which it did not change. Those three entries were retired in turn when
#: they landed, and are recorded nowhere for the same reason: all three are
#: pre-existing production files this lineage changed rather than introduced,
#: so touching any of them again requires declaring it again.
#:
#: The incumbent publisher's four entries were retired in turn when they
#: landed. All four are files that lineage introduced rather than edited, so
#: all four are recorded as owned below, and touching any of them again
#: requires declaring it again. The activation-gate branch's two entries were
#: retired the same way when they landed, and both of the files they named are
#: owned, which is why the remote-transport branch declared them again.
#:
#: The remote-transport branch's three entries were retired in turn when they
#: landed. All three are recorded as owned below, so touching any of them again
#: requires declaring it again. The production-SHA provenance branch declared
#: nothing, because the only thing it changed was the production SHA a test
#: fixture provenances its own served slate with, which is test surface. This
#: branch is the owned case again: it changes the publisher, which is owned
#: below, so it declares it.
#:
#: The attestation-refresh branch's one entry was retired in turn when it
#: landed, for the reason that entry itself gave: the publisher is owned below,
#: so touching it again requires declaring it again. The player-props page
#: branch's four entries were retired the same way when they landed. Three of
#: the four are files that branch introduced and are recorded as owned below;
#: the fourth was the pre-existing destination configuration, which is owned
#: already and which this branch does not touch at all.
#:
#: The affiliate-correction branch's four entries were retired in turn when
#: they landed. Three of the four were already owned; the fourth, the
#: routing-proof module, is a file that branch introduced and is recorded as
#: owned below, so touching any of them again requires declaring it again.
#:
#: The fallback-destination branch's four entries were retired in turn when
#: they landed. All four are owned below -- two it introduced and two it
#: edited -- so touching any of them again requires declaring it again.
#:
#: The public-page-UI branch's two entries were retired in turn when they
#: landed. Both are owned below -- the renderer and the routing-proof script
#: are files this lineage introduced -- so touching either again requires
#: declaring it again.
#:
#: The iframe-handshake branch's one entry was retired in turn when it landed,
#: for the reason that entry itself gave: the renderer is owned below, so
#: touching it again requires declaring it again.
#:
#: The publication-contract branch's nine entries were retired in turn when
#: they landed. Four of them -- the slate module, the pricing script and both
#: serving-source registries -- are pre-existing production files that lineage
#: changed rather than introduced, so they are recorded nowhere and touching
#: any of them again requires declaring it again. The other five are owned
#: below, which is why this branch declares the two of them it touches.
#:
#: The player-identity branch's four entries were retired in turn when they
#: landed. All four are owned below, so touching any of them again requires
#: declaring it again, which is why this branch declares the three of them it
#: touches.
#:
#: The production-automation branch's eleven entries were retired in turn when
#: they landed. Nine of them are owned below -- the workflow, the publisher,
#: the renderer, both JSON records, the scheduler-role validator and the new
#: orchestrator among them -- so touching any of those again requires
#: declaring it again. The other two, the preflight and the lifecycle
#: document, are pre-existing production files that lineage changed rather
#: than introduced and are recorded nowhere.
#:
#: The no-op fail-closed branch's one entry was retired in turn when it
#: landed, for the reason that entry itself gave: the orchestrator is owned
#: below, so touching it again requires declaring it again.
#:
#: The distribution branch's six entries were retired in turn when they
#: landed. Five of the six are owned below, and the sixth, the distribution
#: stage that branch introduced, was recorded as owned when it landed, so
#: touching any of them again requires declaring it again.
#:
#: The public-model-name branch's one entry was retired in turn when it
#: landed, for the reason that entry itself gave: the renderer is owned below,
#: so touching it again requires declaring it again.
#:
#: The opening-day lifecycle branch's six entries were retired in turn when
#: they landed. Two of them, the serving orchestrator and the production
#: automation document, are pre-existing production files that branch changed
#: rather than introduced; the other four are too, and all six are recorded
#: nowhere, so touching any of them again requires declaring it again.
#:
#: The T-20 automation branch's six entries were retired in turn when they
#: landed. Four of them -- the controller, the workflow, and both frozen
#: manifests -- are recorded as owned or declarable below, and the other two,
#: the collector and the workflow validator, are pre-existing production
#: files that branch changed rather than introduced, so touching any of them
#: again requires declaring it again.
#:
#: The rsync portability branch's one entry was retired in turn when it
#: landed, for the reason that entry itself gave: the publisher is owned
#: below, so touching it again requires declaring it again.
#:
#: The page-float-consistency branch's one entry was retired in turn when it
#: landed, for the reason that entry itself gave: the renderer is owned below,
#: so touching it again requires declaring it again.
#:
#: The played-game population fix's three entries were retired in turn when
#: they landed. All three are pre-existing production files this lineage
#: changed rather than introduced and are recorded nowhere, which is why this
#: branch declares all three again. The combination-projection branch's one
#: entry, the raw-calibration-digest branch's one entry and the go-live
#: checklist's one entry were retired the same way: the pricing script and
#: the training module are pre-existing production files recorded nowhere,
#: and the checklist is a document that branch introduced and chose not to
#: own, so touching any of the three again requires declaring it again. The
#: grading entry point's one entry was retired in turn and is owned below,
#: which is the reason that entry itself gave.
#:
#: The persistent-absence branch's four entries were retired in turn when
#: they landed. All four -- the slate module, the availability module and both
#: serving-source registries -- are pre-existing production files that lineage
#: changed rather than introduced and are recorded nowhere, which is why this
#: branch declares again the three of them it touches. It does not touch the
#: availability module, so that entry is retired and not replaced.
#:
#: The no-appearance-prior branch's three entries were retired in turn when
#: they landed. All three -- the slate module and both serving-source
#: registries -- are pre-existing production files that lineage changed
#: rather than introduced and are recorded nowhere, so touching any of them
#: again requires declaring it again.
#:
#: The duplicate-capture branch's four entries -- the capture entry point,
#: the T-20 controller and both serving-source registries -- were retired in
#: turn when they landed. Its capture lock, its durable publication ledger
#: and its per-window recovery isolation are all in production now, so the
#: entry that described them has gone and its text with it. The controller
#: is owned below, which is the reason this branch declares it again rather
#: than inheriting anything from that entry.
#:
#: This branch declares the T-20 controller, and only the T-20 controller.
#: It adds the one refusal the integration never had: nothing is published
#: for a game that has already started.
#:
#: ``select_capture_bundle`` refuses a capture *taken* after its game tipped
#: and nothing refused a release *published* after its game tipped, which
#: are not the same question. The first is about when a snapshot may exist.
#: The second is about when a price may be shown, and between them sits
#: every way a sound pre-tip capture can be reached late: a runner picked up
#: hours after the window, a recovery of a capture an earlier run never
#: published, an advisory window held back for a T-20 reservation and
#: retried in a gap that has moved past the game.
#:
#: That is not hypothetical either. On 2026-10-23 the T-20 group holding
#: game 21717941 was priced and published two hours and forty minutes after
#: that game tipped, and every stage reported success: the capture was
#: pre-tip and eligible, the serving receipt said SERVED and the feed said
#: PUBLISHED. The release that landed was a pregame page for a game in its
#: third quarter.
#:
#: The guard is ``assert_no_window_has_tipped`` and it sits at
#: ``_serve_and_publish``, which every production release goes through --
#: the capture path and the recovery path both call it and there is no
#: third caller. The windows it is about are the ones the caller declares
#: the publication is for, as a required keyword-only argument, so a new
#: caller cannot reach a publication without answering the question. The
#: advisory path is guarded at its own publication entry the same way.
#:
#: The scoping is the load-bearing part. A release is cumulative -- one
#: serve at an offset prices every game with an eligible capture at that
#: offset -- so the last T-20 group of a staggered twelve-game slate is
#: published hours after the first game of it tipped, and that first game's
#: rows belong in the release because they were priced and published before
#: its own tip. A guard that read the whole release rather than the windows
#: the publication is for would refuse every late group of every staggered
#: evening.
#:
#: A refusal is taken before the serving entry point runs, so it writes
#: nothing: no priced artifact is rebuilt, no release is minted, no
#: ``current.json`` is repointed and no ledger row is appended. It declines
#: to add and it never retracts -- the release already serving stays exactly
#: as it is, and the pointer still names what it named.
#:
#: It is loud. ``POST_TIP_PUBLICATION_REFUSED`` is not in
#: ``OUTCOMES_THAT_SUCCEED``, the refusal names the game, its tip and how
#: long ago that was, and the run exits non-zero -- through the refused
#: window staying in the unserved set on the T-20 path, and through the
#: segment outcome on the advisory path. The advisory refusal is still never
#: raised from inside the advisory path: it is recorded there and escalated
#: by ``run`` once every capture is behind it, which is the arrangement
#: production already uses for a deferred recovery refusal, and which is
#: what keeps a reading aid from costing the slate a capture.
#:
#: It composes with the per-window recovery isolation production already
#: has rather than duplicating it. A post-tip window is recorded through
#: ``_record_unrecoverable`` beside the duplicate conflicts and the digest
#: mismatches, under its own outcome because the operator response differs,
#: and it is subtracted from the ledger row by the subtraction that is
#: already there. It is checked behind the selection, not in front of it,
#: so a capture that is itself post-tip is still reported with selection's
#: own more specific reason.
#:
#: Nothing else moves. No capture is taken, re-timed, relabelled, merged or
#: synthesised; the capture log is not read differently and not written at
#: all; no non-T-20 snapshot can stand in for a T-20 one; and no
#: probability, price, edge, EV, quantile, line, distribution, calibration,
#: dependence coefficient, Gate 3 route, routing decision or publication
#: destination is computed or changed anywhere in this change. No fitted
#: artifact is touched, no gate or receipt is weakened, and no workflow,
#: publishing switch, model, config, release or review path is modified.
DECLARED_INTEGRATION_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "ops/run_t20_production_controller.py": (
            "the refusal nothing anywhere performed: no game is priced, "
            "served or published after its own tip. select_capture_bundle "
            "refuses a capture taken after tip, which is a statement about "
            "when a snapshot may exist; nothing was a statement about when a "
            "price may be shown. On 2026-10-23 the T-20 group holding game "
            "21717941 was priced and published two hours and forty minutes "
            "after that game tipped and every stage reported success -- the "
            "capture was pre-tip and eligible, serving said SERVED and the "
            "feed said PUBLISHED -- so the release that landed was a pregame "
            "page for a game in its third quarter. "
            "What changes: assert_no_window_has_tipped, at "
            "_serve_and_publish, which every production release goes through "
            "and which now takes the windows the publication is for as a "
            "required keyword-only argument, so a caller cannot reach a "
            "release without declaring them; the same guard at the advisory "
            "publication entry; POST_TIP_PUBLICATION_REFUSED and its "
            "advisory twin, neither in OUTCOMES_THAT_SUCCEED nor in "
            "ADVISORY_OUTCOMES_WITHOUT_FAULT; an outcome argument on "
            "_record_unrecoverable so a post-tip window joins the permanent "
            "refusals already recorded there under its own name; and the "
            "receipt and summary fields that show both the refusals and the "
            "margin every cleared window had. The guard is scoped to the "
            "windows a publication is for and never to the whole release, "
            "because a release is cumulative and the last group of a "
            "staggered slate is correctly published after earlier games have "
            "tipped. The advisory refusal is recorded where it is taken and "
            "escalated by run() after every capture, never raised from "
            "inside the advisory path, so it cannot cost the slate a "
            "capture. "
            "What does not change: every existing refusal, outcome and exit "
            "code; the capture-side pre-tip check, which still reports its "
            "own more specific reason because the guard is checked behind "
            "selection; the per-window recovery isolation, which this joins "
            "rather than duplicates; the ledger subtraction; the cumulative "
            "re-serving; the advisory reservation and destinations; and the "
            "order recovery runs in. A refusal is taken before the serving "
            "entry point runs, so nothing is priced, no release is minted, "
            "no current.json is repointed and no ledger row is appended -- "
            "the guard declines to add and never retracts, and the release "
            "already serving is left naming exactly what it named. No "
            "capture is taken, re-timed, relabelled, merged or synthesised, "
            "no non-T-20 snapshot can stand in for a T-20 one, and no model, "
            "calibration, price, PMF, Gate 3 route, routing decision or "
            "publication destination is reached from here"
        ),
    }
)

#: Protected paths this lineage introduced into production and now owns.
#:
#: This is a registry, not permission. It exists because two different
#: questions were previously answered by one mapping: "may this branch change
#: this protected path" and "is this protected path part of the shadow's
#: surface". Those coincided only on the branch that first deployed the
#: shadow. Once that branch merged, every entry became permanently stale under
#: the first question while remaining permanently true under the second, which
#: made the production branch impossible to extend without either leaving loose
#: permission behind or dropping the surface from the record.
#:
#: Separating them keeps both guarantees and is strictly tighter than the
#: single mapping was: changing anything here still requires a fresh entry in
#: :data:`DECLARED_INTEGRATION_PATHS` and the review that comes with it, so
#: being owned buys nothing. The same restrictions apply — no model artifact,
#: no config, no production script, no release or review path, no protected
#: production source module — and the branch-safety tests pin that for this
#: mapping exactly as they do for the declaration.
SHADOW_OWNED_PRODUCTION_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "ops/run_production_shadow.py": (
            "the shadow's production entry point. Reads the marginals and the "
            "copula the live pricing path reads, records what the candidate "
            "would have said, and publishes nothing. It is additive: no "
            "existing production script calls it and it writes no production "
            "state"
        ),
        "ops/grade_incumbent_production_slate.py": (
            "grades the slate the incumbent actually served, against settled "
            "box scores. It is the second arm of the frozen policy's central "
            "comparison, which had only the candidate's side before. Reads "
            "priced markets, the serving receipt and the box-score tree; "
            "writes grade rows and its own receipt into a tree no serving "
            "script names. Cannot predict, price, promote or publish"
        ),
        "ops/monitor_live_shadow_evidence.py": (
            "accumulates the shadow's daily evidence durably and reads the "
            "frozen policy over the running total, which nothing could do "
            "while that evidence lived only in the run's temporary directory. "
            "States no threshold of its own: every number comes out of "
            "live_shadow_promotion_policy.json. Writes only into a monitoring "
            "tree no serving or pricing script names, and cannot predict, "
            "price, promote or publish"
        ),
        "ops/install_frozen_model_artifacts.py": (
            "installs the frozen model binaries the incumbent serves from. "
            "They are distributed as a release asset and ignored in Git, so "
            "no checkout has ever held them and the serving step's own "
            "frozen-bundle verification had nothing satisfiable to check. "
            "Reads the published immutable package and the frozen manifest; "
            "writes only into the durable production work root. Cannot fit, "
            "refit, recalibrate, predict, price, promote or publish"
        ),
        ".github/workflows/nba_production_lifecycle.yml": (
            "eight added steps, every pre-existing step unchanged. One invokes "
            "the shadow entry point after the daily fit and cannot fail the "
            "lifecycle. Two give the run its own Python environment under "
            "RUNNER_TEMP and refuse to continue outside it. One serves the "
            "slate with the incumbent. One reads the status the shadow "
            "persisted and may fail the job, because the shadow step is "
            "non-blocking by design and GitHub went green whether the shadow "
            "worked or not. One installs the verified frozen runtime bundle "
            "the incumbent serves from"
        ),
        ".github/workflows/default_branch_guard.yml": (
            "the default branch's only reported check. GitHub schedules "
            "production from main, and main carried no workflow triggered by "
            "a push to main or a pull request into main, so a change to the "
            "scheduler copy landed unchecked. Runs on GitHub-hosted runners, "
            "reads branch contents and asks for contents:read"
        ),
        "ops/validate_main_scheduler_role.py": (
            "the eleven checks that guard runs: that the lifecycle copy "
            "parses, passes the production branch's own static rules, is "
            "byte-identical to the production copy, and has not lost a step, "
            "a gate, the non-blocking shadow boundary or its production "
            "checkout ref. Reads two workflow copies and writes only its own "
            "report"
        ),
        "ops/run_incumbent_production_serving.py": (
            "runs the two serving scripts for the incumbent resolved from the "
            "registry's promotion state, and refuses rather than substituting "
            "when that authority cannot be established. Reads the promotion "
            "state, the frozen bundle and the refresh status; writes the two "
            "serving artifacts and its own receipt, and cannot publish or "
            "promote"
        ),
        "ops/evaluate_shadow_health.py": (
            "judges the persisted shadow status against the frozen live "
            "policy's own operational gates and may fail the job. Reads "
            "shadow.json and the policy; writes only its own report, states "
            "no threshold of its own, and cannot publish or promote"
        ),
        "ops/verify_production_interpreter.py": (
            "refuses the production run when the interpreter resolves outside "
            "the per-run virtual environment. Reads the running interpreter "
            "and the frozen scikit-learn pin from the preflight; writes only "
            "its own receipt and alters no dependency contract"
        ),
        "ops/benchmark_production_fit.py": (
            "wraps the existing --benchmark-only fit path and fingerprints "
            "the serving tree, the registry and the three frozen identifiers "
            "around it, failing the benchmark on any difference. Passes no "
            "registry root to the fit, so there is nothing to register to; "
            "writes only its own receipt"
        ),
        "ops/audit_runner_reliability.py": (
            "decides whether the self-hosted runner meets the "
            "seven-consecutive-clean-runs criterion from GitHub's own API "
            "answer. Reads a collected JSON window and writes only its own "
            "report; it cannot administer or contact anything"
        ),
        "ops/evidence/runner_reliability_audit.json": (
            "the measured seven-run window behind the runner reliability "
            "classification: the waits, the two lost-communication runs and "
            "the 5h47m02s outlier. Evidence only; nothing reads it and it "
            "decides no production behaviour"
        ),
        "docs/wizardofodds/PRE_OPENING_DAY_OPERATOR_ACTIONS.md": (
            "the exact runner, branch-protection and benchmark actions the "
            "three blockers outside this repository require, with the "
            "measured evidence for each and the mechanical test that decides "
            "closure. Documentation only"
        ),
        "ops/evidence/production_shadow_closure.json": (
            "the production-ops closure receipt: the reconciled test counts, "
            "the production SHA both hashes were verified at, and why one "
            "commit reported four different pass/skip splits. Evidence only: "
            "nothing reads it and it decides no production behaviour. Recorded "
            "here rather than declared, because it has already landed"
        ),
        "ops/publish_incumbent_to_wizardofodds.py": (
            "the publishing step PRODUCTION_AUTOMATION.md section 11 names as "
            "following Step 3D, which did not exist. Reads the priced markets "
            "and the serving receipt that step already wrote and copies the "
            "approved incumbent values to a destination; computes no "
            "probability, price, edge or EV, and imports neither the model, "
            "the copula, the calibration, the pricing path nor the shadow. "
            "Live publication is DISABLED and refuses twice over: the switch "
            "is not enabled and no live destination exists to reach"
        ),
        "ops/wizardofodds_publishing_switch.json": (
            "the committed disabled state, in the three-condition shape the "
            "shadow publishing switch already uses. Records the published "
            "authority as incumbent and the candidate authority as shadow "
            "only, so a release manifest asserting either is asserting this "
            "file rather than a literal in code"
        ),
        "ops/wizardofodds_destinations.json": (
            "the destination configuration, whose live entry is null because "
            "every fact needed to describe it -- host, directory, protocol, "
            "credentials, staging target, public URL -- is external to this "
            "repository. Enumerating the missing facts here is what lets the "
            "refusal name them instead of saying 'not configured'"
        ),
        "docs/wizardofodds/INCUMBENT_PUBLISHER.md": (
            "what the publisher publishes, which rules refuse, what an "
            "activation is gated on, how a release is verified before it is "
            "promoted, how a rollback restores the previous one, and the six "
            "external facts a live canary is blocked on. Documentation only: "
            "no code reads it"
        ),
        "ops/wizardofodds_affiliate_routing.py": (
            "which sportsbook a published wager points a reader at. A product "
            "layer downstream of everything: it reads a posted price and a "
            "registry and returns a selection, so it cannot move a "
            "probability, a fair price, an edge or an expected value. It "
            "publishes nothing and is reachable from no production entry "
            "point other than the page builder"
        ),
        "ops/wizardofodds_affiliates.json": (
            "the affiliate books the page routes, the sportsbook identifiers "
            "each is proven to appear under, and which redirect key a click "
            "uses. Evidence is quoted per book. No tracking link, affiliate "
            "id, campaign id or credential is in it, because the site's own "
            "convention is that a click goes to its redirect rather than to "
            "an affiliate URL"
        ),
        "ops/wizardofodds_player_props_page.py": (
            "renders the NBA player-props page from one published incumbent "
            "release, read back and cross-checked through the publisher's own "
            "transport, and writes it to the page canary. A renderer: every "
            "number is copied out of the publication, so there is no second "
            "pricing system here. It refuses a publication that is mid-write, "
            "that records a non-incumbent authority, or whose candidate is no "
            "longer shadow-only, and it renders provenance from an allow-list "
            "so the manifest's absolute artifact path cannot reach a public "
            "page"
        ),
        "ops/wizardofodds_routing_proof.py": (
            "the deterministic slate the routing rule is proven against in "
            "public, in the real published schema and declared test data "
            "rather than a model slate. Four wagers chosen so that all four "
            "routing outcomes are visible on one page. It builds rows and "
            "publishes them to a local destination only; it imports no "
            "model, pricing or calibration code and calls no serving entry "
            "point, so nothing it produces can reach the feed canary or the "
            "live page"
        ),
        "ops/run_wizardofodds_publication.py": (
            "the publication orchestrator the lifecycle runs twice a day: "
            "publish the served slate to the feed, then build the page from "
            "that exact release and publish it. It is the one place the "
            "ordering between the two is expressed, which is why it exists "
            "rather than the workflow calling the publisher and the renderer "
            "directly. It reads a serving receipt and writes only "
            "publication receipts; it cannot fit, serve, grade, price, "
            "promote or activate"
        ),
        "ops/nba_distribution_publication.py": (
            "the distribution stage the lifecycle runs between serving and "
            "publication. The distributions it writes are the ones the "
            "frozen engines produced for the prices already in the priced "
            "parquet: it calls those engines and reports what they return, "
            "so the page can show a bettor the forecast behind a price "
            "without any part of the stack recomputing it. It reads the "
            "priced frame and the frozen marginals and writes one JSON "
            "document; it cannot fit, serve, grade, price, promote, route or "
            "activate"
        ),
        "ops/run_t20_production_controller.py": (
            "the producer of the one market snapshot production prices from, "
            "which nothing in the repository had. It owns one evening segment "
            "at a time: sleep to each T-20 time that segment owns, capture "
            "that tip group once through the existing collector, prove the "
            "capture the pricing path requires exists by re-running that "
            "path's own selection, then serve, derive and publish the canary "
            "through the existing entry points. Fails closed by name rather "
            "than serving an empty slate. Installs no artifact, states no "
            "threshold, and cannot fit, train, calibrate, grade, shadow, "
            "promote or activate"
        ),
        ".github/workflows/nba_t20_production.yml": (
            "the schedule that starts the controller: three low-frequency "
            "triggers whose segments tile the production window exactly once "
            "each, so every T-20 time has one owner. Checks out the "
            "authoritative production ref by name and proves it landed, "
            "keeps an off-day segment to two small wheels and a probe, and "
            "invokes no refresh, fit, frozen installer, grader or shadow. "
            "Publication is canary only and no activation token appears in it"
        ),
    }
)

#: Workflows whose pre-existing steps must survive a change intact. Owning or
#: declaring a workflow buys the right to *add* a step, not to rewrite the
#: lifecycle: the modelling, refresh and registration steps are the production
#: path, and the branch-safety tests check them line by line against the
#: production ref rather than trusting either mapping.
ADDITIVE_ONLY_WORKFLOWS: tuple[str, ...] = (
    ".github/workflows/nba_production_lifecycle.yml",
)

#: The destination registry, declarable only for added named destinations.
#:
#: Same correction as the serving entry points above, in a different place.
#: The registry was pinned by its bytes because it decides where a published
#: record goes, and that is right about every destination in it -- but it read
#: the same for two different kinds of change: moving a destination production
#: already publishes to, and configuring somewhere new. Only the first is what
#: the pin is about, and conflating them made the file permanently unable to
#: gain a destination, which is the one thing a new publication stream needs
#: and the only thing it needs.
#:
#: So it is declarable, under a rule stronger than the byte pin was once a
#: file was allowed to differ at all: every top-level key and every
#: destination the production ref has must be deeply equal, and an added
#: destination must be a canary confined to a non-empty subpath of its served
#: root, with its own release prefix and a transport the file itself declares
#: supported. The two live destinations, the disabled live switch and the
#: transport allow-list are therefore unmovable, and a live destination cannot
#: be added at all.
ADDITIVE_ONLY_NAMED_DESTINATIONS: tuple[str, ...] = (
    "ops/wizardofodds_destinations.json",
)


def _git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def production_merge_base(project_root: Path) -> str | None:
    """The commit this branch diverged from production at, or ``None``."""
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return _git(project_root, "merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue
    return None


def changed_paths(project_root: Path) -> list[str]:
    base = production_merge_base(project_root)
    if base is None:
        return []
    return [
        line
        for line in _git(project_root, "diff", "--name-only", base, "HEAD").splitlines()
        if line
    ]


def modified_production_paths(project_root: Path) -> list[str]:
    """Protected paths this branch has modified, declared or not.

    Reported as-is. Gate H reads this, so a branch that changes a protected
    path is still visibly a branch that changed production, even when the
    change is a declared one.
    """
    return sorted(
        path
        for path in changed_paths(project_root)
        if path.startswith(PROTECTED_PRODUCTION_PREFIXES)
        or path in PROTECTED_PRODUCTION_SOURCES
    )


def undeclared_production_paths(project_root: Path) -> list[str]:
    """Protected paths this branch changed without declaring them.

    This is the invariant the branch-safety tests enforce. Empty means every
    production path this branch touches is named in
    :data:`DECLARED_INTEGRATION_PATHS` with a reason.
    """
    return sorted(
        set(modified_production_paths(project_root)) - set(DECLARED_INTEGRATION_PATHS)
    )


_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+nba_prop_quant\.([A-Za-z_][\w.]*)\s+import", re.M),
    re.compile(r"^\s*from\s+\.([A-Za-z_]\w*)\s+import", re.M),
    re.compile(r"^\s*import\s+nba_prop_quant\.([A-Za-z_]\w*)", re.M),
)


def _module_imports(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group(1).split(".")[0]
        for pattern in _IMPORT_PATTERNS
        for match in pattern.finditer(text)
    }


def serving_reachable_sources(project_root: Path) -> frozenset[str]:
    """Package modules a serving entry point can reach by import.

    Walks the import graph from :data:`SERVING_ENTRY_POINTS` through the
    package. This is what makes :data:`UNDECLARABLE_PRODUCTION_SOURCES` a
    measurement rather than a list somebody has to remember to update: if a
    serving script gains an import of a fit-orchestration module, that module
    becomes serving code and stops being declarable, and the branch-safety
    test that compares the two fails until the declaration is withdrawn.

    Deliberately *not* filtered through :data:`PROTECTED_PRODUCTION_SOURCES`.
    Filtering it there would make the comparison circular -- the protected set
    is derived from this one -- and a module the serving path newly imports
    would stay invisible precisely because it was not protected yet.
    """
    package = Path(project_root) / "src" / "nba_prop_quant"

    pending: list[str] = []

    for relative in SERVING_ENTRY_POINTS:
        entry = Path(project_root) / relative

        if entry.is_file():
            pending.extend(_module_imports(entry))

    seen: set[str] = set()

    while pending:
        name = pending.pop()

        if name in seen:
            continue

        seen.add(name)

        module = package / f"{name}.py"

        if module.is_file():
            pending.extend(_module_imports(module))

    return frozenset(
        f"src/nba_prop_quant/{name}.py"
        for name in seen
        if (package / f"{name}.py").is_file()
    )


def shadow_lineage_offenders(project_root: Path) -> list[str]:
    """Paths a shadow-lineage branch changes outside its own namespaces.

    Containment is a statement about the shadow lineage's work, so it is
    checked against branches that do shadow work. A branch that changes
    nothing under :data:`SHADOW_NAMESPACES` is not a shadow branch, and
    reporting its ordinary modules as "unexpected paths on the shadow branch"
    said nothing true about the shadow while making every unrelated production
    branch unable to add so much as a test.

    What it would otherwise have added is already covered: every path that
    decides production behaviour is a protected path, and
    :func:`undeclared_production_paths` holds every branch to it regardless of
    lineage.

    :data:`CONTAINMENT_GUARD_PATHS` does not count towards doing shadow work:
    it is the guard, not the surface the guard covers.

    Neither does :data:`TEST_SURFACE_PREFIX` count towards being an offender.
    The exemption for ``tests/test_game_latent_state_shadow`` was always about
    tests rather than about that prefix: it is simply how the lineage's own
    tests happen to be named. A branch that extends the shadow and also adds
    the test for a production entry point it wires the shadow into had the
    choice of leaving that entry point untested or declaring its test file as
    though a test file decided production behaviour, and neither of those is
    the containment this guard exists to enforce. No test is reachable from
    production: ``tests/`` is not a protected prefix, so nothing here relaxes
    what :func:`undeclared_production_paths` requires. What remains reported
    is the part that was never redundant -- source outside the lineage's
    namespaces.
    """
    changed = changed_paths(project_root)

    doing_shadow_work = [
        path
        for path in changed
        if path.startswith(SHADOW_NAMESPACES)
        and path not in CONTAINMENT_GUARD_PATHS
    ]

    if not doing_shadow_work:
        return []

    return sorted(
        path
        for path in changed
        if not path.startswith(SHADOW_NAMESPACES)
        and not path.startswith(TEST_SURFACE_PREFIX)
        and path not in DECLARED_INTEGRATION_PATHS
    )


def stale_integration_declarations(project_root: Path) -> list[str]:
    """Declared paths this branch does not actually change.

    A declaration that has outlived its change is permission left lying
    around, so it is reported rather than ignored.
    """
    touched = set(modified_production_paths(project_root))
    return sorted(set(DECLARED_INTEGRATION_PATHS) - touched)


# ---------------------------------------------------------------------
# What a merge would add to production history, as opposed to its tree
# ---------------------------------------------------------------------

#: No new object in the merge path may exceed this.
MAX_MERGE_PATH_BLOB_BYTES = 10 * 1024 * 1024

#: Reported, not failed. Worth a reviewer's attention before it becomes a
#: habit; the repository's largest legitimate blob is a 3.3 MB inventory.
NOTABLE_MERGE_PATH_BLOB_BYTES = 5 * 1024 * 1024

#: Generated datasets and caches. Never acceptable at any size: a 1 KB pickle
#: of fitted marginals is still a cache, and a small parquet is still a
#: regenerated export that nothing reads.
GENERATED_ARTIFACT_NAMES: frozenset[str] = frozenset(
    {
        "oof_gaussian_residuals.parquet",
        "factor_loadings.parquet",
        "joint_event_grades.parquet",
    }
)
GENERATED_ARTIFACT_SUFFIXES: tuple[str, ...] = (
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".joblib",
    ".h5",
    ".onnx",
    ".pt",
    ".pth",
)


def tree_blobs(project_root: Path, ref: str = "HEAD") -> dict[str, int]:
    """``{path: bytes}`` for every blob in one commit's tree."""
    out: dict[str, int] = {}
    for line in _git(project_root, "ls-tree", "-r", "-l", ref).splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) < 4 or fields[1] != "blob" or fields[3] == "-":
            continue
        out[path] = int(fields[3])
    return out


def merge_path_blobs(project_root: Path, base: str | None = None) -> list[dict]:
    """Every blob a merge into production would add, with size and path.

    This is the question a tree listing cannot answer. ``git ls-tree`` reports
    what a branch *has*; a merge transfers what the branch's history
    *reaches*. A blob that was committed and later deleted is absent from
    every subsequent tree and still travels with an ordinary merge commit,
    because the commits that contain it come along too. The latent-state
    research branches are a live example: a 65 MB residual parquet was
    committed once and removed two days later, so it is invisible to
    ``git ls-tree HEAD`` and reachable from all three of their heads.

    Entries are sorted largest first. A blob may appear under more than one
    path across history; each (object, path) pair is reported once.
    """
    base = base or production_merge_base(project_root)
    if base is None:
        return []
    listing = _git(project_root, "rev-list", f"{base}..HEAD", "--objects")
    candidates: dict[tuple[str, str], None] = {}
    for line in listing.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            candidates[(sha, path)] = None
    if not candidates:
        return []

    probe = "\n".join(sha for sha, _ in candidates) + "\n"
    completed = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        cwd=str(project_root),
        input=probe,
        capture_output=True,
        text=True,
        check=True,
    )
    sizes: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] == "blob":
            sizes[fields[0]] = int(fields[2])

    out = [
        {"sha": sha, "path": path, "bytes": sizes[sha]}
        for sha, path in candidates
        if sha in sizes
    ]
    out.sort(key=lambda entry: (-entry["bytes"], entry["path"]))
    return out


def is_generated_artifact(path: str) -> bool:
    name = Path(path).name
    return name in GENERATED_ARTIFACT_NAMES or Path(
        path
    ).suffix in GENERATED_ARTIFACT_SUFFIXES


#: CI checks out a pull request, not a branch, so the base it should audit
#: against is the PR's target. GitHub puts that in ``GITHUB_BASE_REF``.
BASE_REF_ENV_VARS: tuple[str, ...] = ("PRODUCTION_BASE_REF", "GITHUB_BASE_REF")


def resolve_audit_base(project_root: Path) -> str | None:
    """The commit to audit against, preferring an explicitly configured base.

    Falls back to :func:`production_merge_base`. Returns ``None`` only when
    nothing resolves, which callers must treat as a failure rather than as an
    empty merge path: a guard that cannot see the base has not checked
    anything, and reporting that as a pass is how a 65 MB blob gets in.
    """
    for variable in BASE_REF_ENV_VARS:
        ref = os.environ.get(variable, "").strip()
        if not ref:
            continue
        for candidate in (ref, f"origin/{ref}"):
            try:
                return _git(project_root, "merge-base", "HEAD", candidate)
            except subprocess.CalledProcessError:
                continue
    return production_merge_base(project_root)


def audit_merge_path(project_root: Path, base: str | None = None) -> dict:
    """Classify the merge path's new objects against the blob contract.

    ``passed`` is false when a new object breaches the size ceiling, is a
    generated dataset, cache or model binary, or when the base could not be
    resolved at all. ``notable`` is informational.
    """
    resolved_base = base or resolve_audit_base(project_root)
    blobs = merge_path_blobs(project_root, resolved_base)
    oversized = [entry for entry in blobs if entry["bytes"] > MAX_MERGE_PATH_BLOB_BYTES]
    notable = [
        entry
        for entry in blobs
        if entry["bytes"] > NOTABLE_MERGE_PATH_BLOB_BYTES
        and entry["bytes"] <= MAX_MERGE_PATH_BLOB_BYTES
    ]
    generated = [entry for entry in blobs if is_generated_artifact(entry["path"])]
    return {
        "base": resolved_base,
        "base_resolved": resolved_base is not None,
        "new_blob_count": len(blobs),
        "largest": blobs[0] if blobs else None,
        "over_ceiling": oversized,
        "notable": notable,
        "generated_artifacts": generated,
        "ceiling_bytes": MAX_MERGE_PATH_BLOB_BYTES,
        "notable_bytes": NOTABLE_MERGE_PATH_BLOB_BYTES,
        "passed": bool(resolved_base) and not oversized and not generated,
    }
