"""The WizardOfOdds NBA player-props page, built from a published release.

A renderer and nothing more. Every quantity on the page is copied out of the
incumbent's own publication; this module decides layout, ordering, which
sportsbook a reader is pointed at and what notation a price is written in,
and it computes no probability, no price, no edge, no expected value and no
distribution. There is no second pricing system here and no opinion of its
own to have.

WHAT IT READS
-------------

One publication, through the publisher's own transport, in the order a
consumer would: ``current.json`` names a release, the release's manifest
describes it, and the manifest names the rows. Nothing is taken on trust --
the rows are hashed against the manifest, the manifest against the pointer,
and the authority fields are checked before a single row is rendered. A
destination mid-publication, or one serving a candidate, produces a refusal
rather than a page, because a page is a claim about what the model says and a
half-written release cannot support one.

WHAT A BETTOR SEES
------------------

A dense light table in the site's own predictions styling, one row per wager:
the matchup, the player, the prop, the pick, the model's projection, the
model's own price, the price the featured sportsbook is posting, the expected
value at that price, a button, and a toggle that opens the model's whole
forecast for the stat. That is the whole hierarchy, and everything that is
not one of those things has been moved below it or into one collapsed
section -- the full affiliate disclosure, how a sportsbook comes to be the
featured one, and the release's complete provenance.

TWO PRICES, THREE NOTATIONS
---------------------------

Each row carries exactly two prices, and a reader chooses which notation both
are written in: American, decimal, or probability. One control at the top
switches all of them at once.

The switch is notation and nothing else, which is why it is not a conversion
performed in the browser. Every row is rendered with all three renderings of
both of its prices already in the HTML; the control sets one attribute on the
page and the stylesheet shows one rendering and hides two. No number is
computed, re-fetched or rewritten while a reader is reading, so no rendering
can drift from the published value it came from, and the page is as correct
with scripting off as with it on -- it simply shows American.

The model's price is the published calibrated probability that the pick wins,
excluding pushes, written three ways. The sportsbook's price is the exact
American price of the exact offer the row points at; its probability form is
what that single price implies, margin included, and is deliberately *not*
the release's devigged market probability, which is a different quantity
answering a different question. Expected value is published, is computed at
the price in its own row, and is identical in all three notations because it
is not a price.

Columns whose fields the release does not carry are absent rather than
invented. No projection field means no projection column; no team names means
no matchup column, because two integers are not a matchup; no published
distribution for a wager means no toggle on its row.

WHAT THE DISTRIBUTION TOGGLE SHOWS
----------------------------------

The probability of every outcome of the stat, as the model forecasts it,
copied out of the release exactly as the prices are. It is published by
:mod:`nba_distribution_publication`, which re-derives it from the frozen
engines that priced the slate; this page does not reconstruct a distribution
from a mean and a probability, and could not, because those two numbers do
not determine one. The remaining mass above the last listed outcome is one
labelled bucket, so the probabilities a reader is shown add to 100%.

The bars are table cells with a width, so there is no chart library, no
external request and no canvas; the number is printed beside every bar, and
the highlighted range is both tinted and bold, so nothing depends on seeing
colour.

TWO KINDS OF OUTBOUND LINK
--------------------------

A card's button is one of two different relationships and says which. An
affiliate selection goes through the site's own redirect and is labelled "BET
AT", carries ``rel="sponsored"`` and is disclosed as paid. A fallback
selection -- a book nobody is paid by, reached only when no affiliate offers
the wager -- goes straight to that sportsbook's own page, is labelled "VIEW
AT", carries no ``sponsored`` token, and says on the card that WizardOfOdds
earns nothing from it. The two destinations come from two separate files for
that reason; see :mod:`wizardofodds_fallback_destinations`.

Whether a book can be linked is asked *after* the selection is made and can
never change it. A sportsbook posting the best available price is shown at
that price whether or not a destination for it is known; not knowing one costs
the reader a button, not a better number.

WHAT IT WILL NOT PUT ON A PAGE
------------------------------

Provenance is rendered from an allow-list, not filtered by a deny-list. The
published manifest carries ``priced_market_artifact_path``, which is an
absolute path on whatever machine served the slate; it has no business in a
public document, and an allow-list means the next field like it is excluded by
default rather than by somebody remembering.

DETERMINISM
-----------

The output has no clock in it. "Generated" and "published" are the
publication's own ``served_at`` and ``promoted_at``, so building the same
release twice produces the same bytes and a rebuild that changed something is
a change in the release rather than in the hour.

NON-LIVE BY CONSTRUCTION
------------------------

The page canary destination is a canary the same way the feed canary is: every
path it can address resolves beneath its ``canary_subpath``, so it has no way
to express the live index one directory up, and publishing the live page is
not something this module can be asked to do by accident.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ops"))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
import wizardofodds_fallback_destinations as fallback_destinations  # noqa: E402

#: What the page writes, beside each other, in the canary directory.
PAGE_NAME = "index.html"

PAGE_MANIFEST_NAME = "page_manifest.json"

#: Where the consumer preview is published, relative to the page canary's own
#: root. A directory of its own so that previewing the page cannot write over
#: whatever is already published beside it, and so that the live index -- one
#: directory further up and not addressable from here at all -- stays a
#: separate, reviewed step.
PREVIEW_PREFIX = "page-preview"

#: The origins the page broadcasts its height to when it is embedded. The
#: WizardOfOdds site hosts a predictions page in an iframe and sizes that frame
#: from a ``postMessage`` the page sends; a target origin that does not match
#: the embedding page exactly is not an error, it is silently not delivered, so
#: a page carrying one hardcoded origin works on one host and silently sizes
#: wrong on the other. Both hosts the site itself uses are listed, production
#: first. Posting to an origin that is not the parent's does nothing, so the
#: message reaches the real parent once and goes nowhere else.
PARENT_ORIGINS: tuple[str, ...] = (
    "https://wizardofodds.com",
    "https://dev.wizardofodds.com",
)

#: The element the height is measured from, and the key the parent reads. Both
#: are the site's existing convention rather than this page's choice: every
#: embedded predictions page on the site wraps its content in ``app-root`` and
#: sends ``{frameHeight: <px>}``, and the listener on the other side is the
#: site's. Renaming either would leave the frame stuck at its default height.
EMBED_WRAPPER_ID = "app-root"

EMBED_MESSAGE_KEY = "frameHeight"

#: The published document's contract this renderer was written against.
#: Checked for equality rather than a floor: 3 is the release that carries the
#: outcome distribution behind each price, and reading a release from before
#: those existed would draw a page whose distribution panels were all empty
#: rather than refusing.
SUPPORTED_ROWS_CONTRACT = 3

#: The odds formats a reader may choose between, in the order they appear in
#: the control, with the first as the default. American first because that is
#: what a North American sportsbook posts and what the prices in the release
#: are published as.
#:
#: Every format is rendered into the document for every price. Choosing one
#: shows what is already there and hides the rest: nothing is fetched, no
#: probability is recomputed, no wager is repriced, no sportsbook selection
#: changes and no expected value moves. A reader on a page whose script never
#: ran sees the default.
ODDS_FORMAT_AMERICAN = "american"
ODDS_FORMAT_DECIMAL = "decimal"
ODDS_FORMAT_PROBABILITY = "probability"

ODDS_FORMATS: tuple[tuple[str, str], ...] = (
    (ODDS_FORMAT_AMERICAN, "American (-110)"),
    (ODDS_FORMAT_DECIMAL, "Decimal (1.91)"),
    (ODDS_FORMAT_PROBABILITY, "Probability (52.4%)"),
)

DEFAULT_ODDS_FORMAT = ODDS_FORMATS[0][0]

#: Where the reader's choice is remembered, and nowhere else. One key in this
#: browser's ``localStorage``, namespaced to this page, read on load and
#: written on change. It is never sent anywhere: it is not a query parameter,
#: not a cookie, not a field in any receipt, and the page makes no request of
#: any kind when it changes.
ODDS_FORMAT_STORAGE_KEY = "wf.nba.playerProps.oddsFormat"

#: The attribute the whole document's active format lives in, on the embed
#: wrapper. One attribute and a stylesheet rule per format, so switching is
#: one assignment and the browser repaints: no element is rewritten, so no
#: rendered number can differ from the number that was published.
ODDS_FORMAT_ATTRIBUTE = "data-odds-format"

#: The authority a publication must record to be renderable. A page is a
#: public claim about the incumbent, so a publication claiming anything else
#: is refused rather than relabelled.
REQUIRED_MODEL_AUTHORITY = publisher.MODEL_AUTHORITY

#: What the candidate's authority must still say. The shadow exists to be
#: measured, not published, and a destination that stopped saying so is not one
#: this page will read from.
REQUIRED_CANDIDATE_AUTHORITY = "shadow_only"

#: The published fields every row must carry for the page to render it. The
#: renderer's own requirement rather than the publisher's: a row missing one of
#: these would render as a blank where a number belongs.
REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "calibrated_edge_over",
    "calibrated_edge_under",
    "calibrated_ev_over",
    "calibrated_ev_under",
    "calibrated_q_over_nonpush",
    "calibrated_q_under_nonpush",
    "date",
    "freeze_id",
    "game_id",
    "line_value",
    "market_devig_q_over",
    "market_devig_q_under",
    "model_preferred_edge",
    "model_preferred_ev",
    "model_preferred_side",
    "opponent_id",
    "player_id",
    "player_name",
    "prop_type",
    "quote_id",
    "team_id",
    "vendor",
)

#: The provenance fields the page may show, and the label each is shown under.
#: An allow-list on purpose: see the module docstring.
RENDERED_PROVENANCE: tuple[tuple[str, str], ...] = (
    ("slate_date", "Slate"),
    ("incumbent_version", "Incumbent version"),
    ("incumbent_authority", "Incumbent authority"),
    ("model_authority", "Published authority"),
    ("production_code_sha", "Production code SHA"),
    ("served_at", "Slate served"),
    ("input_state_fingerprint", "Input state fingerprint"),
)

#: Provenance fields that must never reach the page, whatever else changes.
#: Redundant beside the allow-list above and kept anyway, because a test that
#: names the field is a test that still fails if somebody turns the allow-list
#: into a deny-list.
WITHHELD_PROVENANCE: tuple[str, ...] = ("priced_market_artifact_path",)

#: Side -> the published probability field for that side.
SIDE_MODEL_PROBABILITY: Mapping[str, str] = {
    routing.SIDE_OVER: "calibrated_q_over_nonpush",
    routing.SIDE_UNDER: "calibrated_q_under_nonpush",
}

SIDE_MARKET_PROBABILITY: Mapping[str, str] = {
    routing.SIDE_OVER: "market_devig_q_over",
    routing.SIDE_UNDER: "market_devig_q_under",
}

#: Side -> the published edge and expected value for that side. The side's own
#: columns rather than ``model_preferred_edge``, because a card is about one
#: wager and the row it reads may be a book whose own preferred side is the
#: other one. Both are published fields either way: nothing is recomputed.
SIDE_EDGE: Mapping[str, str] = {
    routing.SIDE_OVER: "calibrated_edge_over",
    routing.SIDE_UNDER: "calibrated_edge_under",
}

SIDE_EXPECTED_VALUE: Mapping[str, str] = {
    routing.SIDE_OVER: "calibrated_ev_over",
    routing.SIDE_UNDER: "calibrated_ev_under",
}

#: Side -> the published fair price for that side. The model's own view of
#: what the wager is worth, published, and independent of anybody's price: it
#: is the number a reader compares a sportsbook's odds against. Published, so
#: shown; nothing here derives it.
SIDE_FAIR_ODDS: Mapping[str, str] = {
    routing.SIDE_OVER: "calibrated_fair_over_american",
    routing.SIDE_UNDER: "calibrated_fair_under_american",
}

#: The names a published projection may arrive under, in the order they are
#: preferred. ``model_projection`` is the one the contract now carries: the
#: mean the line was priced against. The page reads it and renders it; it
#: computes no projection of its own and never turns a probability into one.
PROJECTION_FIELDS: tuple[str, ...] = (
    "model_projection",
    "projected_value",
    "model_mean",
    "projection",
)

#: Where the release keeps the outcome distributions, and how a wager's own
#: record is found in them. One record per game, player and stat: the three
#: fields a forecast belongs to, which is not the same as the quote it is
#: shown beside. Six books pricing the same player's points share one
#: forecast, and the panel under each of those rows draws that one.
DISTRIBUTIONS_KEY = publisher.DISTRIBUTIONS_KEY

DISTRIBUTION_KEY_FIELDS: tuple[str, str, str] = publisher.DISTRIBUTION_KEY_FIELDS

#: The fields of a distribution record this page draws. Required, because a
#: panel missing one of them would render a chart with a gap in it rather
#: than refusing, and a distribution with a gap in it is not a distribution.
DISTRIBUTION_FIELDS: tuple[str, ...] = publisher.REQUIRED_DISTRIBUTION_FIELDS

#: How close to one a published distribution's mass has to be before this page
#: will draw it. Not a tolerance for sloppiness: the stage that produces these
#: sums an exactly-normalised array and states the remainder as one bucket, so
#: the only error is the floating-point addition, and a release off by more
#: than this is one whose arithmetic went wrong somewhere the page cannot see.
DISTRIBUTION_MASS_TOLERANCE = 1e-9

#: What the panel says about the relationship between the chart and the
#: probability in the row above it. They are related and not identical: the
#: chart is the model's forecast of the outcomes, and the probability for the
#: wager is the betting probability, which carries the historical calibration
#: the release publishes. Saying so plainly, in the panel, rather than leaving
#: a reader to find a discrepancy and conclude one of the two is wrong.
DISTRIBUTION_NOTE = (
    "The distribution shows the model's underlying forecast of the possible "
    "outcomes for this statistic. The Model Probability shown for the wager "
    "above may include historical calibration, so the two are related but "
    "not identical."
)

#: How the panel describes the range it highlights. The rule, not an
#: adjective: the narrowest run of consecutive outcomes holding at least this
#: much of the distribution. Nothing on this page calls an outcome likely,
#: safe or probable, because those are opinions and this is an interval.
DISTRIBUTION_RANGE_NOTE = (
    "Highlighted: the narrowest run of consecutive outcomes holding at least "
    "{target} of the distribution."
)

#: Where a human-readable matchup comes from. ``away_team`` and ``home_team``
#: are the schedule's own abbreviations, published for exactly this. The older
#: names are still read because a matchup stated outright, or a pair of team
#: names, is also an answer -- but ``team_id`` and ``opponent_id`` are not:
#: two integers are not a matchup, "Team 71 vs Team 72" tells a reader
#: nothing, and nothing here looks up a schedule to manufacture one.
MATCHUP_FIELD = "matchup"

MATCHUP_SIDE_FIELDS: tuple[str, str] = ("away_team", "home_team")

MATCHUP_NAME_FIELDS: tuple[str, str] = ("team_name", "opponent_name")

#: The player's own side, shown beside the name so a reader knows whose team
#: the projection belongs to without reading the matchup twice.
PLAYER_TEAM_FIELD = "player_team"

#: When the game starts, if the schedule said. Rendered beside the matchup and
#: nowhere else: it is not a column, it is not sortable, and the page does no
#: timezone arithmetic on it -- the published string is UTC and says so.
TIP_OFF_FIELD = "tip_off_utc"

#: How a selected offer may be described, by tier. A fallback selection is the
#: best price in the whole universe by construction -- it is only reached when
#: no affiliate offers the wager -- so it is the one case where "best available"
#: is a statement of fact rather than a sales line.
TIER_LABEL: Mapping[str, str] = {
    routing.TIER_PRIMARY: "Best Affiliate Price",
    routing.TIER_FALLBACK: "Best Available Price",
    routing.TIER_NONE: "No sportsbook offers this wager",
}

#: The grid, in order: column key, header a reader sees, and whether the
#: column is a number and therefore right-aligned.
#:
#: What is deliberately *not* here is a market-probability column and an edge
#: column. Both are published and both stay published; neither is a sensible
#: headline number for a bettor. The published market probability is the
#: devigged pair implied by a book's two prices, which is not the implied
#: probability of the single price shown beside it, and a column labelled
#: "Market Probability" showing the first while a price shows the second
#: invites exactly the wrong arithmetic. Edge is that same quantity
#: subtracted from the model's, so it inherits the ambiguity.
#:
#: Nor are there three model columns. The model's probability and the model's
#: odds are one quantity written two ways, and a grid showing both beside a
#: format selector would offer a reader the same number three times and
#: invite them to look for a difference between them. There is one model
#: price column and one sportsbook price column, and the selector decides
#: which form of each is shown and what each is called.
COLUMN_MODEL = "model"
COLUMN_SPORTSBOOK = "sportsbook"
COLUMN_DISTRIBUTION = "distribution"

COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("matchup", "Matchup", False),
    ("player", "Player", False),
    ("prop", "Prop", False),
    ("pick", "Pick", False),
    ("projection", "Model Projection", True),
    (COLUMN_MODEL, "Model Price", True),
    (COLUMN_SPORTSBOOK, "Sportsbook Price", True),
    ("ev", "EV", True),
    ("book", "Sportsbook", False),
    (COLUMN_DISTRIBUTION, "Distribution", False),
)

#: What each price column is called in each format. The header has to change
#: with the selector: a column of decimal prices under a heading that says
#: American is a worse defect than either number alone, and "Sportsbook
#: Implied Probability" is a different claim from "Sportsbook Odds" and has
#: to be named as one.
#:
#: The sportsbook probability is the implied probability of the one price in
#: the cell beside it -- that price, converted -- and not the devigged pair
#: the release also publishes. Two books quoting the same wager have
#: different holds, and a column labelled as this one is labelled has to be
#: the price the reader would actually be taking.
COLUMN_FORMAT_LABEL: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        COLUMN_MODEL: MappingProxyType(
            {
                ODDS_FORMAT_AMERICAN: "Model Odds",
                ODDS_FORMAT_DECIMAL: "Model Odds",
                ODDS_FORMAT_PROBABILITY: "Model Probability",
            }
        ),
        COLUMN_SPORTSBOOK: MappingProxyType(
            {
                ODDS_FORMAT_AMERICAN: "Sportsbook Odds",
                ODDS_FORMAT_DECIMAL: "Sportsbook Odds",
                ODDS_FORMAT_PROBABILITY: "Sportsbook Implied Probability",
            }
        ),
    }
)

#: Columns rendered only when the publication carries the field behind them.
CONDITIONAL_COLUMNS: tuple[str, ...] = (
    "matchup",
    "projection",
    COLUMN_DISTRIBUTION,
)

#: Columns a wager nobody prices cannot honestly fill. A price nobody is
#: offering is not a price, and an expected value at a price that does not
#: exist is not an expected value, so the section for those wagers does not
#: have the columns rather than having them full of dashes.
UNOFFERED_COLUMNS: tuple[str, ...] = (COLUMN_SPORTSBOOK, "ev")

COLUMN_LABEL: Mapping[str, str] = MappingProxyType(
    {key: label for key, label, _ in COLUMNS}
)

COLUMN_NUMERIC: Mapping[str, bool] = MappingProxyType(
    {key: numeric for key, _, numeric in COLUMNS}
)

#: What a row says when the featured sportsbook has no established
#: destination. Two words on the row and one sentence under the table, rather
#: than the sentence on every row: on a slate where no book is linkable the
#: long version is the widest thing on the page and says the same thing ten
#: times.
UNLINKED_TAG = "Price only"

UNLINKED_NOTE = (
    "Rows marked “Price only” are sportsbooks WizardOfOdds has no established "
    "link for. The price is what the release publishes; there is nothing to "
    "click, and nothing there is a referral."
)

#: What the page says about commission where a reader will actually see it.
#: The full disclosure is still on the page, one disclosure away; a wall of
#: it above the predictions is a wall nobody reads.
SHORT_DISCLOSURE = (
    "WizardOfOdds may earn commission from certain sportsbook links."
)

AFFILIATE_DISCLOSURE = (
    "Affiliate disclosure: links to Bovada, BetOnline, BetUS and BetAnything "
    "are paid affiliate links and WizardOfOdds may earn a commission if you "
    "open an account through them. Which sportsbook is featured is decided by "
    "the published routing rule, which prefers those four books over any "
    "other book at the same wager, so a featured price is not necessarily the "
    "best price available anywhere. Where a card says Best Available Price, no "
    "affiliate book offered that wager and the price shown is the best across "
    "every sportsbook in the published data; a link on one of those cards goes "
    "straight to that sportsbook's own page, is not a paid placement, and "
    "earns WizardOfOdds nothing. Whether a sportsbook can be linked never "
    "affects which one is selected. Model numbers are never changed by this "
    "routing. Odds move; check the sportsbook before betting. 21+ where "
    "applicable, and nothing here is a guarantee of profit."
)

#: What a direct, unpaid link says on it. Deliberately not "BET AT", which is
#: what an affiliate call to action says: the two are different relationships
#: and a reader should be able to see which one they are looking at.
DIRECT_CTA_VERB = "VIEW AT"

AFFILIATE_CTA_VERB = "BET AT"


@dataclass(frozen=True)
class Presentation:
    """Who a build of this page is for.

    One renderer serves two readers who need different things said out loud.
    A bettor wants the player, the pick, the price and a button, and is owed a
    plain statement that the page is not live yet. Somebody checking the
    sportsbook-selection rule wants that rule's own reasoning on every row,
    and the slate they are looking at is constructed test data that must never
    be mistaken for a betting opinion.

    Keeping the difference in one object rather than in scattered flags is
    what stops the QA vocabulary -- the rule's wording, the word proof, the
    fixture's player names -- from reaching a page meant for bettors: a
    consumer build is simply never handed it.
    """

    #: The headline of the banner at the top of the page.
    banner: str

    #: The sentence under the banner, explaining what the reader is looking at.
    detail: str

    #: Which of the site's note styles the banner uses.
    tone: str

    #: Whether each row carries the routing rule's own explanation of why that
    #: sportsbook was selected. QA only: see the class docstring.
    show_routing_reason: bool


#: The consumer page, published to a canary path before activation. The bytes
#: here are the bytes the live page will be built from, which is the whole
#: point of previewing it rather than designing it twice.
CONSUMER_PREVIEW = Presentation(
    banner="CANARY PREVIEW — NOT YET LIVE",
    detail=(
        "This is the NBA player props page as it will be published. It is "
        "served from a preview path, it is not linked from the site, and the "
        "predictions below are the ones the current release carries."
    ),
    tone="warn",
    show_routing_reason=False,
)

#: The routing QA page. The slate behind it is constructed so that every
#: sportsbook-selection outcome is visible at once, which makes it useful
#: evidence and makes it something nobody should ever bet.
ROUTING_PROOF_QA = Presentation(
    banner="TEST DATA — ROUTING PROOF — NOT LIVE BETTING PICKS",
    detail=(
        "Every prediction on this page is constructed test data, built so "
        "that each sportsbook-selection outcome can be checked in public. "
        "These are not model picks for a real slate and nothing here should "
        "be bet."
    ),
    tone="err",
    show_routing_reason=True,
)


class PageRefused(RuntimeError):
    """The page builder declined to render anything."""


# ----------------------------------------------------------------------
# reading the publication
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedRelease:
    """The one release a build is allowed to read.

    Without this, a build reads whatever the destination's current pointer
    names, which is right for a human asking for the current page and wrong
    for automation: the page is supposed to show the release the same run just
    published, and "whatever is current" is only that release until something
    else publishes. A pointer naming an older release is the failure that
    looks most like success -- a page that renders, serves and verifies, about
    a slate from before.

    So automation states which release it means and the read refuses anything
    else, rather than rendering first and comparing afterwards. Both halves
    are checked because they answer different questions: the id says which
    release, and the digest says those exact rows, which is what catches a
    release republished under its own name with different bytes.
    """

    release_id: str
    rows_sha256: str

    @classmethod
    def from_publication_receipt(cls, receipt: Mapping[str, Any]) -> ExpectedRelease:
        """Read from the feed publisher's own receipt, with nothing inferred."""
        release = receipt.get("release") or {}

        release_id = str(release.get("release_id") or "")

        rows_sha256 = str(release.get("rows_sha256") or "")

        if not release_id or not rows_sha256:
            raise PageRefused(
                "the publication receipt names no release id and rows digest, "
                "so there is no release this page could be pinned to"
            )

        return cls(release_id=release_id, rows_sha256=rows_sha256)

    def payload(self) -> dict[str, str]:
        return {"release_id": self.release_id, "rows_sha256": self.rows_sha256}


@dataclass(frozen=True)
class Publication:
    """One release at a destination, read back and cross-checked."""

    release_id: str
    slate_date: str
    rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_relative: str
    rows_relative: str
    rows_sha256: str
    promoted_at: str
    mode: str

    #: The release's own distributions, keyed by game, player and stat. Read
    #: from the same document as the rows and not from anywhere else, so the
    #: forecast a panel draws and the price the row shows came out of one
    #: release and one digest.
    distributions: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict, repr=False
    )


def _pointer(transport: Any, relative: str, *, what: str) -> dict[str, Any]:
    if not transport.exists(relative):
        raise PageRefused(
            f"the destination has no {relative}, so there is no {what} to "
            "build a page from"
        )

    try:
        document = json.loads(transport.read_bytes(relative))
    except ValueError as error:
        raise PageRefused(
            f"{relative} is not readable JSON ({error}), so the {what} it "
            "describes cannot be rendered"
        ) from error

    if not isinstance(document, dict):
        raise PageRefused(
            f"{relative} is not a JSON object, so it does not describe a "
            f"{what}"
        )

    return document


def load_publication(
    transport: Any, *, expected: ExpectedRelease | None = None
) -> Publication:
    """The release the destination says it is serving, verified.

    Read in the order a consumer reads it and checked at every hop, because
    each hop is a way the destination could be mid-publication: a pointer at a
    release whose manifest is not there yet, a manifest at rows that are not,
    or rows that are there but are not the ones the manifest describes.

    ``expected`` pins the read to one release. Checked at the pointer, before
    the manifest is opened, so a destination serving a different release is
    refused rather than read; and again at the rows digest, which is the fact
    the page is actually about.
    """
    current = _pointer(
        transport, publisher.CURRENT_POINTER, what="current release"
    )

    release_id = str(current.get("release_id") or "")

    manifest_relative = str(current.get("manifest_relative") or "")

    if not release_id or not manifest_relative:
        raise PageRefused(
            f"{publisher.CURRENT_POINTER} names no release and manifest, so "
            "what the destination is serving is not settled"
        )

    if expected is not None and release_id != expected.release_id:
        raise PageRefused(
            f"{publisher.CURRENT_POINTER} names the release {release_id!r} "
            f"and this page was built for {expected.release_id!r}, so the "
            "destination is not serving the publication this run produced"
        )

    manifest = _pointer(transport, manifest_relative, what="release manifest")

    authority = str(manifest.get("model_authority") or "")

    if authority != REQUIRED_MODEL_AUTHORITY:
        raise PageRefused(
            f"{manifest_relative} records the model authority as "
            f"{authority or 'nothing'}, not {REQUIRED_MODEL_AUTHORITY!r}, so "
            "this is not an incumbent publication and no page is built from it"
        )

    candidate = str(manifest.get("candidate_authority") or "")

    if candidate != REQUIRED_CANDIDATE_AUTHORITY:
        raise PageRefused(
            f"{manifest_relative} records the candidate authority as "
            f"{candidate or 'nothing'}, not {REQUIRED_CANDIDATE_AUTHORITY!r}, "
            "so the shadow may no longer be shadow-only and no page is built "
            "from it"
        )

    for label, pointed, recorded in (
        ("release id", release_id, str(manifest.get("release_id") or "")),
        (
            "rows path",
            str(current.get("rows_relative") or ""),
            str(manifest.get("rows_relative") or ""),
        ),
        (
            "rows digest",
            str(current.get("rows_sha256") or ""),
            str(manifest.get("rows_sha256") or ""),
        ),
    ):
        if pointed != recorded:
            raise PageRefused(
                f"{publisher.CURRENT_POINTER} and {manifest_relative} "
                f"disagree on the {label} of the current release "
                f"({pointed or 'nothing'} against {recorded or 'nothing'}), "
                "so what the destination serves is not settled"
            )

    rows_relative = str(manifest.get("rows_relative") or "")

    rows_sha256 = str(manifest.get("rows_sha256") or "")

    if expected is not None and rows_sha256 != expected.rows_sha256:
        raise PageRefused(
            f"{manifest_relative} records the rows digest {rows_sha256!r} and "
            f"this page was built for {expected.rows_sha256!r}, so the "
            "release at the destination is not the bytes this run published"
        )

    if not rows_relative or not transport.exists(rows_relative):
        raise PageRefused(
            "the release manifest names rows at "
            f"{rows_relative or 'no path'}, which are not at the destination, "
            "so there is nothing to render"
        )

    rows_bytes = transport.read_bytes(rows_relative)

    served = publisher.digest(rows_bytes)

    if served != rows_sha256:
        raise PageRefused(
            f"{rows_relative} reads back as {served} but the manifest records "
            f"{rows_sha256}, so the destination is not serving the rows it "
            "says it is"
        )

    try:
        document = json.loads(rows_bytes)
    except ValueError as error:
        raise PageRefused(
            f"{rows_relative} is not readable JSON ({error})"
        ) from error

    if not isinstance(document, dict):
        raise PageRefused(f"{rows_relative} is not a JSON object")

    contract = document.get("contract_version")

    if contract != SUPPORTED_ROWS_CONTRACT:
        raise PageRefused(
            f"{rows_relative} declares rows contract {contract!r} and this "
            f"renderer was written against {SUPPORTED_ROWS_CONTRACT}, so the "
            "fields it would read are not the fields it would get"
        )

    rows = document.get("rows")

    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise PageRefused(
            f"{rows_relative} does not carry a list of row objects"
        )

    distributions = read_distributions(document, relative=rows_relative)

    slate_date = str(manifest.get("slate_date") or "")

    approved_freeze = str(manifest.get("approved_freeze_id") or "")

    for index, row in enumerate(rows):
        missing = [name for name in REQUIRED_ROW_FIELDS if name not in row]

        if missing:
            raise PageRefused(
                f"row {index} of {rows_relative} is missing fields this page "
                "renders: " + ", ".join(missing)
            )

        if approved_freeze and str(row.get("freeze_id") or "") != approved_freeze:
            raise PageRefused(
                f"row {index} of {rows_relative} was produced by "
                f"{row.get('freeze_id')!r} and the release approves "
                f"{approved_freeze!r}, so it is not the incumbent's own output "
                "and is not rendered"
            )

        # The publisher refuses this too, which is where it is meant to be
        # caught. Checked again here because this page reads a release it did
        # not write: a name is the one published field a reader identifies the
        # wager by, and the Player cell has no fallback, so a release that lost
        # one would render a row about nobody rather than fail.
        name_problem = publisher.player_name_problem(row.get("player_name"))

        if name_problem:
            raise PageRefused(
                f"row {index} of {rows_relative}: {name_problem}, so the "
                "prediction is not rendered under an identifier"
            )

    return Publication(
        release_id=release_id,
        slate_date=slate_date,
        rows=tuple(rows),
        provenance=dict(manifest.get("provenance") or {}),
        manifest=manifest,
        manifest_relative=manifest_relative,
        rows_relative=rows_relative,
        rows_sha256=rows_sha256,
        promoted_at=str(current.get("promoted_at") or ""),
        mode=str(manifest.get("mode") or current.get("mode") or ""),
        distributions=distributions,
    )


def read_distributions(
    document: Mapping[str, Any], *, relative: str
) -> dict[str, Mapping[str, Any]]:
    """The release's distributions, checked for the shape a panel draws.

    Every record is checked rather than the first, and checked here rather
    than when a panel is rendered, so a release with one malformed forecast
    publishes no page instead of publishing one good page with a silent hole
    in it. The publisher refuses the same defects where the bytes enter the
    contract; this page reads a release it did not write.

    What is not checked is the arithmetic that produced a record. That is the
    pricing engine's, and the stage that published it derives every number
    through that engine's own functions. What this looks at is whether the
    record it is about to draw is a distribution at all: the two arrays the
    same length, every probability finite and non-negative, the listed mass
    and the tail bucket summing to one, and the outcomes in ascending order
    with no repeats.
    """
    records = document.get(DISTRIBUTIONS_KEY)

    if not isinstance(records, dict):
        raise PageRefused(
            f"{relative} carries no {DISTRIBUTIONS_KEY} object, so the "
            "forecasts behind its prices are not in the release this page "
            "would be built from"
        )

    checked: dict[str, Mapping[str, Any]] = {}

    for key in sorted(records):
        record = records[key]

        if not isinstance(record, dict):
            raise PageRefused(
                f"the distribution for {key} in {relative} is not an object"
            )

        missing = [name for name in DISTRIBUTION_FIELDS if name not in record]

        if missing:
            raise PageRefused(
                f"the distribution for {key} in {relative} is missing fields "
                "this page draws: " + ", ".join(missing)
            )

        problem = distribution_problem(record)

        if problem:
            raise PageRefused(
                f"the distribution for {key} in {relative} is not one this "
                f"page can draw: {problem}"
            )

        checked[key] = record

    return checked


def distribution_problem(record: Mapping[str, Any]) -> str:
    """What is wrong with one published distribution, or ``""``.

    Separated from the loop above so a test can state each defect by name
    against one record rather than having to assemble a whole release to
    exercise it.
    """
    support = record.get("distribution_support")

    probability = record.get("distribution_probability")

    if not isinstance(support, list) or not isinstance(probability, list):
        return "its support and probabilities are not both lists"

    if not support:
        return "it lists no outcomes"

    if len(support) != len(probability):
        return (
            f"it lists {len(support)} outcomes and {len(probability)} "
            "probabilities"
        )

    outcomes: list[int] = []

    for value in support:
        if not isinstance(value, int) or isinstance(value, bool):
            return f"the outcome {value!r} is not a whole number"

        outcomes.append(value)

    if any(value < 0 for value in outcomes):
        return "an outcome is negative, and a count cannot be"

    if any(
        later <= earlier
        for earlier, later in zip(outcomes, outcomes[1:])
    ):
        return "its outcomes are not in ascending order without repeats"

    weights: list[float] = []

    for value in probability:
        number = _finite(value)

        if number is None:
            return f"the probability {value!r} is not a finite number"

        if number < 0.0:
            return f"the probability {number!r} is negative"

        weights.append(number)

    tail = _finite(record.get("distribution_tail_probability"))

    if tail is None or tail < 0.0:
        return "its tail probability is not a finite non-negative number"

    outcome = record.get("distribution_tail_outcome")

    if outcome is not None:
        if not isinstance(outcome, int) or isinstance(outcome, bool):
            return "its tail bucket is not labelled by a whole outcome"

        if outcome <= outcomes[-1]:
            return "its tail bucket starts at or below a listed outcome"

    elif tail > 0.0:
        return "it states tail mass without saying which outcome it is above"

    mass = sum(weights) + tail

    if abs(mass - 1.0) > DISTRIBUTION_MASS_TOLERANCE:
        return f"its probabilities and tail bucket sum to {mass!r} and not one"

    low = record.get("distribution_most_likely_low")

    high = record.get("distribution_most_likely_high")

    if not isinstance(low, int) or not isinstance(high, int):
        return "its highlighted range is not a pair of whole outcomes"

    if low > high:
        return "its highlighted range runs backwards"

    return ""


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


# ----------------------------------------------------------------------
# odds formats
# ----------------------------------------------------------------------
#
# Four conversions, all of them total functions of one published number, and
# all of them applied here rather than in the browser. The published
# probability is the exact one the release carries: it is converted at full
# precision and only the printed string is rounded, so a reader switching
# formats is shown two renderings of one number rather than two numbers.


def model_decimal(probability: Any) -> float | None:
    """What the model's probability is worth as a decimal price."""
    number = _finite(probability)

    if number is None or not 0.0 < number <= 1.0:
        return None

    return 1.0 / number


def model_american(probability: Any) -> int | None:
    """What the model's probability is worth as an American price.

    The sign convention a sportsbook uses: a favourite is negative and the
    number is the stake that wins 100, an underdog is positive and the number
    is what 100 wins. An even-money forecast is rendered ``+100``, which is
    the form a reader is shown everywhere else; ``-100`` is the same price and
    is not what anybody writes.
    """
    number = _finite(probability)

    if number is None or not 0.0 < number < 1.0:
        return None

    if number == 0.5:
        return 100

    if number > 0.5:
        return int(round(-100.0 * number / (1.0 - number)))

    return int(round(100.0 * (1.0 - number) / number))


def sportsbook_decimal(odds: Any) -> float | None:
    """One posted American price as a decimal price."""
    number = _finite(odds)

    if number is None or number == 0.0:
        return None

    if number > 0.0:
        return 1.0 + number / 100.0

    return 1.0 + 100.0 / abs(number)


def sportsbook_implied_probability(odds: Any) -> float | None:
    """What one posted price implies, including the sportsbook's hold.

    The implied probability of *this* price and nothing else. The release also
    publishes a devigged pair, which is a different quantity: it is the two
    prices with the hold divided out, it does not correspond to either price
    on its own, and it is not what a reader is being offered. This page never
    shows it under a sportsbook heading.
    """
    decimal = sportsbook_decimal(odds)

    return None if decimal is None else 1.0 / decimal


# ----------------------------------------------------------------------
# turning rows into what the page shows
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PageRow:
    """One card: the model's opinion on one wager, and where to take it.

    ``row`` is the published row of the sportsbook the reader is sent to, so
    the market probability, edge and expected value shown beside a price are
    that book's own published numbers at that price. ``quotes`` is every
    published row on the wager, which is the offer universe the routing
    considered.
    """

    wager: routing.Wager
    routed: routing.Routed
    row: Mapping[str, Any] = field(repr=False)
    quotes: tuple[Mapping[str, Any], ...] = field(default=(), repr=False)

    #: Where the selected sportsbook can be reached when it is not an
    #: affiliate and the site has an established destination for it. ``None``
    #: whenever the selection is an affiliate -- those go through the redirect
    #: -- and whenever no destination is proven for the book.
    direct_url: str | None = None

    #: The release's forecast for this player's stat in this game, or ``None``
    #: when the release published none. Read, never derived: the panel draws
    #: this record's own numbers and nothing on this page turns a probability,
    #: a mean or a line back into a distribution.
    distribution: Mapping[str, Any] | None = field(default=None, repr=False)

    @property
    def side(self) -> str:
        return self.wager.side

    @property
    def distribution_key(self) -> str:
        return distribution_key(self.row)

    @property
    def model_probability(self) -> Any:
        return self.row[SIDE_MODEL_PROBABILITY[self.side]]

    @property
    def market_probability(self) -> Any:
        return self.row[SIDE_MARKET_PROBABILITY[self.side]]

    @property
    def edge(self) -> Any:
        return self.row[SIDE_EDGE[self.side]]

    @property
    def expected_value(self) -> Any:
        return self.row[SIDE_EXPECTED_VALUE[self.side]]

    @property
    def fair_odds(self) -> Any:
        """The model's own price for this side, as published, or ``None``.

        ``None`` when the publication does not carry the field, which is the
        only reason the column would be missing: no fair price is computed
        here from anything.
        """
        return self.row.get(SIDE_FAIR_ODDS[self.side])

    @property
    def projection(self) -> Any:
        """The model's projected number, if the publication publishes one.

        Read, never derived. A combo line has none -- it is priced from a
        joint distribution with no published central forecast of the total --
        so this returns nothing for those rows and the cell stays empty rather
        than being filled with the sum of something.
        """
        for name in PROJECTION_FIELDS:
            value = self.row.get(name)

            if value not in (None, ""):
                return value

        return None

    @property
    def matchup(self) -> str | None:
        """Who is playing, in words, or ``None``.

        Always away before home, which is how a slate is read everywhere else,
        and taken from the two published sides rather than assembled from the
        player's own team and ``is_home`` -- those would give the same two
        teams in an order that changes row by row for the same game.

        ``None`` when the schedule published no names. Two integers are not a
        matchup, and nothing here looks one up.
        """
        stated = self.row.get(MATCHUP_FIELD)

        if stated not in (None, ""):
            return str(stated)

        away, home = (self.row.get(name) for name in MATCHUP_SIDE_FIELDS)

        if away not in (None, "") and home not in (None, ""):
            return f"{away} @ {home}"

        team, opponent = (self.row.get(name) for name in MATCHUP_NAME_FIELDS)

        if team in (None, "") or opponent in (None, ""):
            return None

        joiner = "vs" if _truthy(self.row.get("is_home")) else "@"

        return f"{team} {joiner} {opponent}"

    @property
    def player_team(self) -> str | None:
        """The player's own side, or ``None``."""
        value = self.row.get(PLAYER_TEAM_FIELD)

        return None if value in (None, "") else str(value)

    @property
    def tip_off(self) -> str | None:
        """When the game starts, as the schedule published it, or ``None``."""
        value = self.row.get(TIP_OFF_FIELD)

        return None if value in (None, "") else str(value)

    @property
    def stable_order(self) -> tuple[str, str, str, str, str]:
        """The last tie-break in the card order, so a build is reproducible.

        Slate date, game, player, prop, line: identifiers, in a fixed order,
        used only after edge and expected value have failed to separate two
        cards. It is not a time and it is not offered to a reader as one.
        The published feed does carry a tip-off timestamp, and the page shows
        it beside the matchup; what it does not offer is an ordering by it,
        which is a product decision rather than a missing field.
        """
        return (
            str(self.row.get("date") or ""),
            str(self.row.get("game_id") or ""),
            str(self.row.get("player_name") or ""),
            str(self.row.get("prop_type") or ""),
            str(self.row.get("line_value") or ""),
        )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}

    return bool(value)


def distribution_key(row: Mapping[str, Any]) -> str:
    """Which distribution a published row's wager is forecast by.

    The publisher's own three fields in the publisher's own order, read from
    it rather than restated, so a key written by one side and looked up by the
    other cannot be two different keys.
    """
    return "|".join(str(row.get(name)) for name in DISTRIBUTION_KEY_FIELDS)


def page_rows(
    publication: Publication,
    *,
    registry: routing.AffiliateRegistry,
    fallback: fallback_destinations.FallbackDestinations | None = None,
) -> tuple[PageRow, ...]:
    """One card per exact wager, routed, strongest model edge first.

    The publication is one row per quote, so a wager several books price
    appears several times. A card is about a wager rather than about a quote:
    the duplicate rows are that wager's offer universe, which is precisely
    what the routing rule is for, and rendering one card each would state the
    same opinion repeatedly at prices the reader was not sent to.

    The card then reads the row of the book it sends the reader to, so the
    market probability, edge and expected value shown are the published
    numbers at the published price of that sportsbook. The model's own
    probability does not depend on a price, so it is the same in every row on
    the wager -- asserted below rather than assumed, because showing one
    book's number as the model's opinion would be a claim this page cannot
    support if they disagreed.

    The default order is the highest expected value, because that is the
    column the page shows and the one number that answers whether a wager is
    worth taking: it is the model's probability measured against the price the
    reader is actually being pointed at. Published edge breaks ties and the
    wager's identifiers break those, so the order is total and a rebuild does
    not reshuffle. The sort is over published edge and published expected
    value and nothing derived from them: no ranking score was invented here.
    """
    grouped: dict[routing.Wager, list[Mapping[str, Any]]] = {}

    for row in publication.rows:
        wager = routing.wager_of(row)

        grouped.setdefault(wager, []).append(row)

    built: list[PageRow] = []

    for wager, quotes in grouped.items():
        ordered = tuple(
            sorted(
                quotes,
                key=lambda quote: (
                    routing.normalise_vendor(quote.get("vendor")),
                    str(quote.get("quote_id")),
                ),
            )
        )

        _agree_on_the_model(ordered, wager=wager)

        routed = routing.route(
            routing.offers_for(publication.rows, wager), registry=registry
        )

        quoting = _quoting_row(ordered, routed=routed)

        built.append(
            PageRow(
                wager=wager,
                routed=routed,
                row=quoting,
                quotes=ordered,
                direct_url=_direct_url(routed, fallback=fallback),
                distribution=publication.distributions.get(
                    distribution_key(quoting)
                ),
            )
        )

    return tuple(
        sorted(
            built,
            key=lambda page_row: (
                -_number(page_row.expected_value),
                -_number(page_row.edge),
                page_row.stable_order,
            ),
        )
    )


def _direct_url(
    routed: routing.Routed,
    *,
    fallback: fallback_destinations.FallbackDestinations | None,
) -> str | None:
    """A direct destination for a selection that is not a referral.

    Deliberately takes the routing result rather than the offer universe, and
    is called after :func:`wizardofodds_affiliate_routing.route` has returned.
    There is no argument here that could express a preference, so knowing a URL
    for a sportsbook cannot make that sportsbook win: the best available price
    is selected first and the question of whether it can be linked is asked
    second.

    An affiliate selection is excluded outright rather than merely expected to
    be absent from the fallback registry. An affiliate click goes through the
    site's redirect, which is the site's own convention, and a direct link for
    a book that pays commission would be an unattributed referral.
    """
    if fallback is None or routed.offer is None:
        return None

    if routed.tier != routing.TIER_FALLBACK or routed.is_affiliate:
        return None

    if routed.destination is not None:
        return None

    return fallback.url_for(routed.offer.vendor)


def _agree_on_the_model(
    quotes: tuple[Mapping[str, Any], ...], *, wager: routing.Wager
) -> None:
    """Refuse a wager whose rows disagree about the model's probability."""
    field_name = SIDE_MODEL_PROBABILITY[wager.side]

    values = {str(quote.get(field_name)) for quote in quotes}

    if len(values) > 1:
        raise PageRefused(
            f"the published rows for {wager.prop_type} {wager.side} "
            f"{wager.line_value} on player {wager.player_id} disagree about "
            f"{field_name} ({sorted(values)}), so what the model says about "
            "this wager is not settled and no card can state it"
        )


def _quoting_row(
    quotes: tuple[Mapping[str, Any], ...], *, routed: routing.Routed
) -> Mapping[str, Any]:
    """The row the card reads: the selected book's, or the first on the wager.

    Falling back to the first row matters only when nothing was selected, and
    then the card shows no price at all -- the model fields it still shows are
    the same in every row on the wager.
    """
    if routed.offer is not None:
        for quote in quotes:
            if (
                routing.normalise_vendor(quote.get("vendor")) == routed.offer.vendor
                and str(quote.get("quote_id")) == str(routed.offer.quote_id)
            ):
                return quote

    return quotes[0]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------


def american(value: Any) -> str:
    """American odds the way a sportsbook writes them."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return "—"

    return f"+{number}" if number > 0 else str(number)


def percent(value: Any, *, places: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{places}f}%"
    except (TypeError, ValueError):
        return "—"


def signed_percent(value: Any, *, places: int = 1) -> str:
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return "—"

    return f"{number:+.{places}f}%"


def line_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    return f"{number:g}"


#: Props a bettor already has a word for. Anything not here is title-cased
#: from the published name, which is right for "points" and "rebounds" and
#: would be wrong for "threes": nobody shopping a prop calls it Threes.
PROP_LABEL: Mapping[str, str] = MappingProxyType(
    {
        "points": "Points",
        "rebounds": "Rebounds",
        "assists": "Assists",
        "threes": "3PM",
        "three_pointers": "3PM",
        "blocks": "Blocks",
        "steals": "Steals",
        "turnovers": "Turnovers",
    }
)


def prop_label(prop_type: Any) -> str:
    key = str(prop_type or "").strip().lower()

    return PROP_LABEL.get(key, key.replace("_", " ").title())


def number_text(value: Any) -> str:
    """A published number, printed as published rather than rounded away."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    return f"{number:g}"


def _minute(value: Any) -> str:
    """An ISO timestamp from the publication, to the minute, in UTC.

    Reformatting, not clock-reading: the value is the publication's own
    ``served_at``, so two builds of one release still produce the same bytes.
    """
    text = str(value or "").strip()

    if not text:
        return ""

    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)

    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _clock(value: Any) -> str:
    """A published tip time, to the minute, in UTC and labelled as such.

    Without the date: every row on a page is one slate, and the slate's date
    is already in the line above the table. Without a local conversion too --
    the page would have to guess a reader's zone, and a guessed tip time is
    worse than an honest one.
    """
    text = _minute(value)

    return text.split(" ", 1)[1] if text.count(" ") == 2 else text


def _json_for_html(payload: Any) -> str:
    """JSON safe to embed in a script element.

    ``<`` escaped rather than the whole string HTML-escaped, because the
    content of a ``script`` element is not parsed as HTML: escaping ampersands
    there would corrupt the data, while leaving ``</script`` intact would end
    the element early.
    """
    return (
        json.dumps(payload, indent=2, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


#: The page's own stylesheet, in the site's light palette.
#:
#: The tokens, class names and rules are the ones the existing WizardOfOdds
#: predictions module already uses -- ``--wf-*`` for the palette, ``wf-body``
#: and ``wf`` for the page shell, ``wf-table`` for a data table, ``wf-bo-btn``
#: for a sportsbook button -- so this page reads as part of the same site
#: rather than as a second design that happens to live next door. It is
#: inlined rather than linked: the publisher writes one directory containing
#: one document, so a relative stylesheet reference would be a 404, and a page
#: that cannot fetch its own stylesheet is a page that renders unstyled prices.
STYLE = """
.wf-body{margin:0;background:#fff;-webkit-text-size-adjust:100%}
.wf{
  --wf-bg:#fff;--wf-bg2:#f6f7f8;--wf-bg3:#eef0f2;--wf-text:#23262b;
  --wf-muted:#626973;--wf-border:#dfe2e6;--wf-line:#eceef1;
  --wf-gold:#b8891a;--wf-gold-soft:#fbf4df;--wf-win:#1e7a3c;--wf-loss:#a8323e;
  --wf-warn-bg:#fff6dd;--wf-warn:#6b4d00;--wf-err-bg:#fdecee;--wf-err:#8a1f2b;
  color-scheme:light;color:var(--wf-text);background:var(--wf-bg);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  max-width:1180px;margin:0 auto;padding:12px 16px 24px;box-sizing:border-box;
}
.wf *,.wf *::before,.wf *::after{box-sizing:border-box}
/* An element asked to be hidden is hidden, whatever else sets a display on
   it. Without this the mobile rules below would repaint filtered-out rows as
   grids and the visible count would start describing a page nobody is
   looking at. */
.wf [hidden]{display:none!important}
.wf a{color:#7a5a0c;text-underline-offset:2px}
.wf a:hover{color:var(--wf-text)}
.wf :focus-visible{outline:2px solid var(--wf-gold);outline-offset:2px}
.wf-h1{font-size:22px;line-height:1.25;margin:6px 0 4px;letter-spacing:-.01em}
.wf h2{font-size:17px;margin:0 0 8px;letter-spacing:-.005em}
.wf-muted{color:var(--wf-muted)}
.wf-small{font-size:12.5px}
.wf-lead{margin:0 0 10px;color:var(--wf-muted);font-size:13px;max-width:62em}
.wf .num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.wf-tag{
  display:inline-block;font-size:10.5px;font-weight:600;padding:0 6px;
  border:1px solid var(--wf-border);border-radius:3px;color:var(--wf-muted);
  background:var(--wf-bg2);white-space:nowrap;
}
.wf-tag.paid{color:#7a5a0c;border-color:#e4cf96;background:var(--wf-gold-soft)}
/* banner */
.wf-note{
  padding:8px 12px;border-radius:6px;margin:0 0 10px;font-size:13px;
  background:var(--wf-bg2);border:1px solid var(--wf-border);
}
.wf-note.warn{background:var(--wf-warn-bg);color:var(--wf-warn);border-color:#efd999}
.wf-note.err{background:var(--wf-err-bg);color:var(--wf-err);border-color:#f0c2c8}
.wf-note b{display:block;font-size:12px;letter-spacing:.06em;text-transform:uppercase}
.wf-note span{display:block;margin-top:2px;font-size:12.5px}
/* compact header provenance */
.wf-prov{
  display:flex;flex-wrap:wrap;gap:2px 16px;margin:0 0 12px;padding:0;
  font-size:12px;color:var(--wf-muted);border-bottom:2px solid var(--wf-gold);
  padding-bottom:8px;
}
.wf-prov div{display:flex;gap:5px}
.wf-prov dt{font-weight:400}
.wf-prov dd{margin:0;color:var(--wf-text);font-weight:600;font-variant-numeric:tabular-nums}
/* controls: one row of four, widest where the text is longest. Explicit
   tracks rather than auto-fit, because an auto-fit track is sized by the
   narrowest control in the row and the sort option "Expected value (highest
   first)" then renders clipped. */
.wf-controls{
  display:grid;gap:8px 10px;margin:0 0 8px;
  grid-template-columns:
    minmax(0,1.5fr) minmax(0,1fr) minmax(0,1.1fr) minmax(0,1.5fr);
}
.wf-controls > div{display:flex;flex-direction:column;gap:3px;min-width:0}
.wf-controls label{
  font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
  color:var(--wf-muted);
}
/* 16px, not the 13px the rest of the page is set in: below 16px iOS Safari
   zooms the whole document when a field takes focus, and a bettor who taps
   the search box should not have to pinch their way back to the table. */
.wf-controls input,.wf-controls select{
  font:inherit;font-size:16px;padding:7px 9px;border:1px solid #c9ced5;
  border-radius:5px;background:#fff;color:var(--wf-text);min-height:40px;width:100%;
}
.wf-count{margin:0 0 6px;font-size:12.5px;font-weight:600;color:var(--wf-muted)}
.wf-disc{margin:0 0 8px;font-size:12px;color:var(--wf-muted)}
/* table */
.wf-tablewrap{
  overflow-x:auto;border:1px solid var(--wf-border);border-radius:6px;
  background:var(--wf-bg);
}
.wf-table{border-collapse:collapse;width:100%;font-size:13px}
.wf-table th,.wf-table td{padding:7px 9px;border-bottom:1px solid var(--wf-line);vertical-align:top}
.wf-table thead th{
  background:var(--wf-bg2);font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.04em;color:var(--wf-muted);text-align:left;white-space:nowrap;
  border-bottom:1px solid var(--wf-border);
}
.wf-table thead th.num{text-align:right}
.wf-table tbody tr:last-child > *{border-bottom:0}
.wf-table tbody tr:hover{background:#fafbfc}
.wf-table tbody tr:nth-child(even){background:#fafafb}
.wf-table tbody tr:nth-child(even):hover{background:#f6f8fa}
.c-player{font-weight:600;white-space:nowrap}
.c-matchup{white-space:nowrap}
/* The player's own side and the tip time: present, and quieter than the name
   and the matchup they hang off, because neither is what a reader scans for. */
.wf-side{font-weight:400;font-size:11.5px;color:var(--wf-muted)}
.wf-tip{
  display:block;font-size:11px;color:var(--wf-muted);
  font-variant-numeric:tabular-nums;
}
.c-prop{white-space:nowrap;color:var(--wf-muted)}
.c-pick{font-weight:700;white-space:nowrap}
.c-pick .over{color:var(--wf-win)}
.c-pick .under{color:#34506e}
.pos{color:var(--wf-win);font-weight:600}
.neg{color:var(--wf-muted)}
/* the sportsbook cell and its two kinds of button */
.wf-bookpick{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;min-width:0}
.wf-bo-book{font-weight:700;font-size:12px;letter-spacing:.02em;text-transform:uppercase}
.wf .wf-bo-btn{
  display:inline-block;padding:6px 12px;border-radius:5px;background:var(--wf-gold);
  color:#fff!important;font-size:11.5px;font-weight:700;letter-spacing:.04em;
  text-transform:uppercase;text-decoration:none;white-space:nowrap;line-height:1.3;
  border:1px solid #9c7413;min-height:30px;
}
.wf .wf-bo-btn:hover{background:#9c7413;color:#fff!important}
/* A direct link to a sportsbook nobody is paid by: same shape and the same
   tap target, deliberately not the gold of a paid placement, and "VIEW AT"
   rather than "BET AT". */
.wf .wf-bo-btn.direct{
  background:var(--wf-bg);color:var(--wf-text)!important;border:1px solid var(--wf-text);
}
.wf .wf-bo-btn.direct:hover{background:var(--wf-bg2);color:var(--wf-text)!important}
/* The two long explanations in a sportsbook cell take their own line rather
   than stretching the column: the numbers are what the table is for. */
.wf-bo-note{flex:1 0 100%;font-size:11px;color:var(--wf-muted);white-space:normal}
.wf-why{flex:1 0 100%;font-size:11px;color:var(--wf-muted);white-space:normal}
/* sections below the table */
.wf-pending{margin:16px 0 0}
.wf-pending h2{margin:0 0 2px}
.wf-empty{
  padding:22px 14px;text-align:center;color:var(--wf-muted);font-size:13px;
  border:1px solid var(--wf-border);border-top:0;border-radius:0 0 6px 6px;background:var(--wf-bg2);
}
.wf-gloss{margin:16px 0 0;font-size:13px}
.wf-gloss summary{cursor:pointer;font-weight:700}
.wf-gloss h3{font-size:13px;margin:12px 0 4px}
.wf-gloss p{margin:4px 0;color:var(--wf-muted);max-width:68em}
.wf-gloss dt{font-weight:700;margin-top:6px;font-size:12.5px}
.wf-gloss dd{margin:1px 0 0;color:var(--wf-muted);font-size:12.5px}
.wf-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:6px 16px;margin:6px 0 0}
.wf-facts dt{font-size:11px;color:var(--wf-muted);font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin:0}
.wf-facts dd{margin:0 0 4px;font-size:12px;color:var(--wf-text);font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
/* The odds format selector, and the three renderings it chooses between.
   Every price is in the document three times and the attribute on the wrapper
   decides which one paints: no element is rewritten, so what a reader sees in
   any format is the number that was published. The default rule comes first
   so a reader whose script never ran sees American rather than nothing. */
.of{display:none}
.of-american{display:inline}
[data-odds-format="american"] .of-american{display:inline}
[data-odds-format="american"] .of-decimal,
[data-odds-format="american"] .of-probability{display:none}
[data-odds-format="decimal"] .of-decimal{display:inline}
[data-odds-format="decimal"] .of-american,
[data-odds-format="decimal"] .of-probability{display:none}
[data-odds-format="probability"] .of-probability{display:inline}
[data-odds-format="probability"] .of-american,
[data-odds-format="probability"] .of-decimal{display:none}
/* the distribution disclosure and its panel */
.c-dist{white-space:nowrap}
.wf .wf-dist-toggle{
  padding:5px 10px;border-radius:5px;border:1px solid var(--wf-border);
  background:var(--wf-bg2);color:var(--wf-text);font-size:11.5px;font-weight:700;
  letter-spacing:.03em;text-transform:uppercase;cursor:pointer;min-height:30px;
  white-space:nowrap;
}
.wf .wf-dist-toggle:hover{background:var(--wf-bg3)}
.wf .wf-dist-toggle[aria-expanded="true"]{background:var(--wf-bg3)}
.wf-dist-row > td{background:var(--wf-bg2);padding:0}
.wf-dist-row[hidden]{display:none}
.wf-dist-panel{padding:12px 14px 14px}
.wf-dist-panel h3{margin:0 0 6px;font-size:13.5px}
.wf-dist-facts{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:4px 16px;margin:0 0 8px;
}
.wf-dist-facts dt{
  font-size:10px;color:var(--wf-muted);font-weight:700;text-transform:uppercase;
  letter-spacing:.04em;margin:0;
}
.wf-dist-facts dd{
  margin:0;font-size:13px;color:var(--wf-text);font-variant-numeric:tabular-nums;
}
.wf-dist-kind{
  margin:0 0 6px;font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.04em;color:var(--wf-muted);
}
.wf-dist-table{border-collapse:collapse;width:100%;max-width:46em;font-size:12.5px}
.wf-dist-table caption{
  caption-side:top;text-align:left;font-size:11.5px;color:var(--wf-muted);
  padding:0 0 6px;
}
.wf-dist-table th,.wf-dist-table td{
  padding:2px 8px 2px 0;border:0;text-align:left;font-weight:400;
  font-variant-numeric:tabular-nums;
}
.wf-dist-table thead th{
  font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
  color:var(--wf-muted);
}
.wf-dist-table th[scope="row"]{width:5em;font-weight:600}
.wf-dist-table td.num,.wf-dist-table th.num{width:6em;text-align:right}
.wf-dist-table td.bar,.wf-dist-table th.bar{width:auto;padding-left:10px}
/* The highlighted run is marked by weight as well as by tint, because a
   reader who cannot distinguish the tint still has to be able to see it. */
.wf-dist-table tbody tr.likely{background:#eef3f8}
.wf-dist-table tbody tr.likely th[scope="row"],
.wf-dist-table tbody tr.likely td.num{font-weight:700}
.wf-bar{
  height:9px;border-radius:2px;background:var(--wf-gold);min-width:1px;
}
.wf-dist-note{margin:8px 0 0;font-size:11.5px;color:var(--wf-muted);max-width:60em}
.wf-foot{
  margin-top:18px;padding-top:10px;border-top:1px solid var(--wf-border);
  font-size:12px;color:var(--wf-muted);
}
.wf-foot p{margin:3px 0}
@media (max-width:860px){
  .wf-controls{grid-template-columns:repeat(2,minmax(0,1fr))}
  .wf-controls .wide,.wf-controls .sort{grid-column:1/-1}
}
/* Below this width the table stops being a table: each prediction becomes a
   compact labelled record, so nothing has to be scrolled sideways to read a
   price or reach a button. */
@media (max-width:720px){
  .wf{padding:10px 12px 20px}
  .wf-tablewrap{overflow-x:visible;border-radius:6px}
  .wf-table{font-size:13.5px}
  .wf-table thead{display:none}
  .wf-table tbody tr{
    display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:2px 12px;
    padding:10px 12px;border-bottom:1px solid var(--wf-border);
  }
  .wf-table tbody tr[hidden]{display:none!important}
  .wf-table tbody td{display:block;border:0;padding:0;white-space:normal}
  .wf-table tbody td.num{text-align:left}
  .wf-table tbody td::before{
    content:attr(data-label);display:block;font-size:10px;font-weight:700;
    text-transform:uppercase;letter-spacing:.04em;color:var(--wf-muted);
  }
  .wf-table tbody td.c-player::before,.wf-table tbody td.c-pick::before,
  .wf-table tbody td.c-matchup::before{content:none}
  /* The matchup heads the card instead of being one more labelled field: it
     is the same for every prop in a game and it is what tells a reader which
     game they are looking at. The tip time sits on the same line here, where
     a card has less room to spare than a table cell. */
  .c-matchup{grid-column:1/-1;font-size:12px;color:var(--wf-muted)}
  .c-matchup .wf-tip{display:inline;margin-left:6px}
  .c-player{grid-column:1/-1;font-size:15px}
  .c-prop{grid-column:1/-1;font-size:12px}
  .c-pick{grid-column:1/-1;font-size:15px;margin-bottom:4px}
  .c-book{grid-column:1/-1;margin-top:6px}
  .c-dist{grid-column:1/-1;margin-top:6px}
  .wf .wf-bo-btn{padding:11px 16px;font-size:12.5px;min-height:44px}
  .wf .wf-dist-toggle{padding:10px 14px;font-size:12px;min-height:44px;width:100%}
  .wf-bo-note{max-width:none}
  /* The panel is a card of its own on a narrow screen rather than a cell
     inside the card above it, so the grid that reflows every other row has
     to be switched off for this one. */
  .wf-table tbody tr.wf-dist-row{display:block;padding:0}
  .wf-table tbody tr.wf-dist-row > td{display:block;padding:0}
  .wf-table tbody tr.wf-dist-row > td::before{content:none}
  .wf-dist-panel{padding:10px 12px 12px}
  .wf-dist-table{font-size:12px}
}
"""

SCRIPT = """
'use strict';
(function () {
  var rows = Array.prototype.slice.call(
    document.querySelectorAll('tr[data-row]')
  );
  var picks = document.getElementById('picks-body');
  var count = document.getElementById('count');
  var empty = document.getElementById('empty');
  var pending = document.getElementById('pending');
  var pendingCount = document.getElementById('pending-count');
  var inputs = {
    search: document.getElementById('f-search'),
    prop: document.getElementById('f-prop'),
    book: document.getElementById('f-book'),
    sort: document.getElementById('f-sort')
  };
  var offered = rows.filter(function (row) {
    return row.getAttribute('data-offer') === 'yes';
  });

  /* The odds format. Display only: it sets one attribute on the wrapper and
     the stylesheet shows the matching rendering of every price. Nothing is
     fetched, no probability is recomputed, no wager is repriced, no
     sportsbook selection changes and no expected value moves -- all three
     renderings of every price are already in the document.

     The preference is remembered in this browser and nowhere else. It is not
     a query parameter, not a cookie and not in any request: a storage that
     refuses to write, as a private window may, costs the reader the memory
     and nothing else. */
  var FORMAT_KEY = __FORMAT_KEY__;
  var FORMATS = __FORMATS__;
  var DEFAULT_FORMAT = __DEFAULT_FORMAT__;
  var wrapper = document.getElementById('__WRAPPER__');
  var formatInput = document.getElementById('f-odds');

  function known(name) {
    return FORMATS.indexOf(name) !== -1;
  }

  function remembered() {
    try {
      var stored = window.localStorage.getItem(FORMAT_KEY);
      return known(stored) ? stored : DEFAULT_FORMAT;
    } catch (error) {
      return DEFAULT_FORMAT;
    }
  }

  function remember(name) {
    try {
      window.localStorage.setItem(FORMAT_KEY, name);
    } catch (error) {
      /* A reader who has blocked storage still gets the format they chose
         for as long as this page is open. */
    }
  }

  function applyFormat(name, persist) {
    var chosen = known(name) ? name : DEFAULT_FORMAT;
    if (wrapper) {
      wrapper.setAttribute('data-odds-format', chosen);
    }
    if (formatInput && formatInput.value !== chosen) {
      formatInput.value = chosen;
    }
    if (persist) {
      remember(chosen);
    }
  }

  if (formatInput) {
    formatInput.addEventListener('change', function () {
      applyFormat(formatInput.value, true);
    });
  }

  applyFormat(remembered(), false);

  /* The distribution panels. Each one is a row that is already in the
     document, carrying the published forecast; the button beside it shows
     and hides that row. The panels start open for a reader with no script,
     because the button cannot work for them, and this closes them and
     reveals the buttons as the first thing it does. */
  var toggles = Array.prototype.slice.call(
    document.querySelectorAll('[data-dist-toggle]')
  );

  function panelOf(row) {
    var id = row.getAttribute('data-panel');
    return id ? document.getElementById(id) : null;
  }

  function openPanel(panel, open) {
    if (!panel) return;
    panel.hidden = !open;
    panel.style.display = open ? '' : 'none';
  }

  toggles.forEach(function (button) {
    var panel = document.getElementById(
      button.getAttribute('aria-controls')
    );
    button.hidden = false;
    openPanel(panel, false);
    button.setAttribute('aria-expanded', 'false');
    button.addEventListener('click', function () {
      var open = button.getAttribute('aria-expanded') !== 'true';
      button.setAttribute('aria-expanded', open ? 'true' : 'false');
      openPanel(panel, open);
    });
  });

  function value(row, name) {
    var parsed = parseFloat(row.getAttribute(name));
    return isNaN(parsed) ? -Infinity : parsed;
  }

  function choice(key) {
    return inputs[key] ? inputs[key].value : 'all';
  }

  /* Showing and hiding writes the inline style as well as the attribute.
     The attribute alone is a standing invitation to the bug this replaced:
     any later rule that sets a display on a row beats the hidden
     attribute, the row keeps painting, and the count below -- computed
     from the same decision -- then contradicts the page. An inline style
     cannot be overridden by a stylesheet, so what is counted is what is
     drawn. */
  function show(row, ok) {
    row.hidden = !ok;
    row.style.display = ok ? '' : 'none';
  }

  function matches(row, needle, prop, book) {
    if (needle && row.getAttribute('data-search').indexOf(needle) === -1) {
      return false;
    }
    if (prop !== 'all' && row.getAttribute('data-prop') !== prop) {
      return false;
    }
    return book === 'all' || row.getAttribute('data-book') === book;
  }

  function apply() {
    var needle = (inputs.search ? inputs.search.value || '' : '')
      .trim()
      .toLowerCase();
    var prop = choice('prop');
    var book = choice('book');
    var shown = 0;
    var waiting = 0;

    rows.forEach(function (row) {
      var ok = matches(row, needle, prop, book);
      show(row, ok);
      /* A hidden prediction's panel is hidden with it. Left visible it would
         be a forecast with no row above it saying whose it was. Shown again,
         it returns closed rather than remembering: the button beside it says
         closed, and the two must not disagree. */
      var panel = panelOf(row);
      if (panel) {
        openPanel(panel, false);
        var button = document.querySelector(
          '[aria-controls="' + panel.id + '"]'
        );
        if (button) {
          button.setAttribute('aria-expanded', 'false');
        }
      }
      if (!ok) {
        return;
      }
      if (row.getAttribute('data-offer') === 'yes') {
        shown += 1;
      } else {
        waiting += 1;
      }
    });

    var mode = choice('sort');
    var ordered = offered.slice().sort(function (a, b) {
      var key = mode === 'model' ? 'data-model-p' : 'data-ev';
      if (mode === 'player') {
        return a.getAttribute('data-sortname')
          .localeCompare(b.getAttribute('data-sortname'));
      }
      return value(b, key) - value(a, key) ||
        a.getAttribute('data-stable').localeCompare(
          b.getAttribute('data-stable')
        );
    });
    if (picks) {
      ordered.forEach(function (row) {
        picks.appendChild(row);
        /* Immediately after its own row, every time, so reordering cannot
           leave a panel under somebody else's prediction. */
        var panel = panelOf(row);
        if (panel) {
          picks.appendChild(panel);
        }
      });
    }

    if (count) {
      count.textContent = 'Showing ' + shown + ' of ' + offered.length +
        (offered.length === 1 ? ' prediction' : ' predictions');
    }
    if (empty) {
      empty.hidden = shown !== 0 || offered.length === 0;
    }
    if (pendingCount) {
      pendingCount.textContent = String(waiting);
    }
    if (pending) {
      pending.hidden = waiting === 0;
    }
  }

  Object.keys(inputs).forEach(function (key) {
    if (!inputs[key]) return;
    inputs[key].addEventListener('input', apply);
    inputs[key].addEventListener('change', apply);
  });

  apply();
})();
""".replace(
    "__FORMAT_KEY__", json.dumps(ODDS_FORMAT_STORAGE_KEY)
).replace(
    "__FORMATS__", json.dumps([key for key, _ in ODDS_FORMATS])
).replace(
    "__DEFAULT_FORMAT__", json.dumps(DEFAULT_ODDS_FORMAT)
).replace(
    "__WRAPPER__", EMBED_WRAPPER_ID
)

#: The site's own iframe height handshake, as it already runs on the football
#: index and the other embedded predictions pages. Kept recognisably identical
#: to those copies -- same measured element, same message, same triggers, same
#: 2px threshold -- because the receiving listener is the site's and is shared
#: with them. The one change is the loop: the copies on the site each name a
#: single parent origin, which is why the staging copies had to be hand-edited
#: to a second one, and why an embed on the other host silently sizes wrong.
EMBED_SCRIPT = """
(function () {
    if (window.self === window.top) {
        return;
    }

    const parentOrigins = __ORIGINS__;
    let lastSentHeight = 0;

    function sendHeightToParent() {
        try {
            const body = document.body;
            const html = document.documentElement;
            if (!body || !html) return;

            const contentWrapper = document.getElementById('__WRAPPER__') || body;
            const currentHeight = Math.ceil(contentWrapper.getBoundingClientRect().height || html.scrollHeight);

            if (currentHeight > 0 && Math.abs(currentHeight - lastSentHeight) > 2) {
                lastSentHeight = currentHeight;
                console.log("Iframe broadcast event! Height sent: " + currentHeight + "px");
                parentOrigins.forEach(function (origin) {
                    window.parent.postMessage({ __KEY__: currentHeight }, origin);
                });
            }
        } catch (error) {
            console.warn("Skipped frame height calculation:", error.message);
        }
    }

    window.addEventListener('load', sendHeightToParent);
    document.addEventListener('DOMContentLoaded', sendHeightToParent);

    document.addEventListener('click', function() {
        // Wait 150ms to give the layout time to physically render the change after the click
        setTimeout(sendHeightToParent, 150);
    });

    if (window.MutationObserver) {
        const observer = new MutationObserver(sendHeightToParent);
        observer.observe(document.body, { attributes: true, childList: true, subtree: true });
    }

    if (window.ResizeObserver) {
        const resizeObserver = new ResizeObserver(sendHeightToParent);
        resizeObserver.observe(document.body);
    }

    setInterval(sendHeightToParent, 1000);

})();
""".replace(
    "__ORIGINS__", json.dumps(list(PARENT_ORIGINS))
).replace(
    "__WRAPPER__", EMBED_WRAPPER_ID
).replace(
    "__KEY__", EMBED_MESSAGE_KEY
)


def _option(value: str, label: str) -> str:
    return f'<option value="{escape(value)}">{escape(label)}</option>'


def _select(
    element_id: str,
    label: str,
    options: Sequence[tuple[str, str]],
    *,
    klass: str = "",
) -> str:
    rendered = "".join(_option(value, text) for value, text in options)

    classes = f' class="{klass}"' if klass else ""

    return (
        f"<div{classes}><label for=\"{element_id}\">{escape(label)}</label>"
        f'<select id="{element_id}">{rendered}</select></div>'
    )


def _header_provenance(publication: Publication) -> str:
    """The three facts a reader has any use for, on one line.

    When the model last spoke, which model it was, and which build published
    it. Everything else the publication records is real and stays available in
    the methodology section and in ``page_manifest.json``; it is simply not
    what somebody looking for a prop bet came for, and a page that opens with
    a digest is a page that looks like it was built for its author.
    """
    cells: list[str] = []

    updated = _minute(publication.provenance.get("served_at"))

    if updated:
        cells.append(f"<div><dt>Model updated</dt><dd>{escape(updated)}</dd></div>")

    version = publication.provenance.get("incumbent_version")

    if version:
        cells.append(
            f"<div><dt>Model version</dt><dd>{escape(str(version))}</dd></div>"
        )

    build = str(publication.provenance.get("production_code_sha") or "")

    if build:
        cells.append(
            f"<div><dt>Production build</dt><dd>{escape(build[:12])}</dd></div>"
        )

    if not cells:
        return ""

    return '<dl class="wf-prov">' + "".join(cells) + "</dl>"


def _full_provenance(publication: Publication) -> str:
    """Everything the allow-list permits, inside the collapsed section.

    Nothing is dropped from the publication by making the header short: the
    same fields the previous page printed across the top are all still here,
    one disclosure away, and all of them are still in the page manifest.
    """
    cells: list[str] = []

    for name, label in RENDERED_PROVENANCE:
        value = publication.provenance.get(name)

        if value in (None, ""):
            continue

        cells.append(
            f"<dt>{escape(label)}</dt><dd>{escape(str(value))}</dd>"
        )

    cells.append(f"<dt>Release</dt><dd>{escape(publication.release_id)}</dd>")

    cells.append(
        f"<dt>Rows digest</dt><dd>{escape(publication.rows_sha256)}</dd>"
    )

    if publication.promoted_at:
        cells.append(
            f"<dt>Published</dt><dd>{escape(publication.promoted_at)}</dd>"
        )

    cells.append(f"<dt>Mode</dt><dd>{escape(publication.mode)}</dd>")

    return '<dl class="wf-facts">' + "".join(cells) + "</dl>"


def _cta(page_row: PageRow) -> str:
    """The sportsbook cell: who is offering the price, and how to get there.

    Three outcomes, and the markup says which without the reader having to
    read the small print. An affiliate selection is the site's gold button
    through the site's redirect and is tagged as paid. A selection nobody is
    paid by is the same button in the page's own colours, labelled "VIEW AT",
    with no sponsored token and a line saying the link earns nothing. A
    selection with no established destination keeps its name and its price and
    gets no link at all rather than an invented one.
    """
    routed = page_row.routed

    if routed.offer is None:
        return (
            '<span class="wf-bo-note">No sportsbook in the published data '
            "offers this wager.</span>"
        )

    price = american(routed.offer.american_odds)

    name = routed.display_name or routed.offer.vendor

    if routed.destination is not None:
        return (
            f'<a class="wf-bo-btn" href="{escape(routed.destination)}" '
            'target="_blank" rel="sponsored nofollow noopener">'
            f"{AFFILIATE_CTA_VERB} {escape(name.upper())} {escape(price)}</a>"
            '<span class="wf-tag paid">Paid link</span>'
        )

    if page_row.direct_url is not None:
        return (
            f'<a class="wf-bo-btn direct" href="{escape(page_row.direct_url)}" '
            f'target="_blank" rel="{fallback_destinations.REQUIRED_REL}">'
            f"{DIRECT_CTA_VERB} {escape(name.upper())} {escape(price)}</a>"
            '<span class="wf-bo-note">Goes straight to the sportsbook. Not a '
            "paid placement: WizardOfOdds earns nothing if you use it.</span>"
        )

    return (
        f'<span class="wf-bo-book">{escape(name)}</span>'
        f'<span class="wf-tag">{escape(UNLINKED_TAG)}</span>'
    )


def visible_columns(rows: Sequence[PageRow]) -> tuple[str, ...]:
    """Which columns this publication can actually fill.

    Three of them depend on fields a release may or may not carry, and the
    page asks rather than assumes. The contract now publishes a projection and
    both team names, so on an ordinary slate all three appear. A slate of
    nothing but combo lines has no projection to show, and a schedule that
    named no teams has no matchup, and in both cases the column is absent
    rather than full of dashes -- an absence that is a fact about the release
    and not an opinion held here.
    """
    return tuple(
        key
        for key, _, _ in COLUMNS
        if key not in CONDITIONAL_COLUMNS
        or any(_has_column(page_row, key) for page_row in rows)
    )


def _has_column(page_row: PageRow, key: str) -> bool:
    if key == "matchup":
        return bool(page_row.matchup)

    if key == "projection":
        return page_row.projection is not None

    if key == COLUMN_DISTRIBUTION:
        return page_row.distribution is not None

    return True


def _pick(page_row: PageRow) -> str:
    side = page_row.side

    return (
        f'<span class="{escape(side)}">{escape(side.upper())} '
        f'{escape(line_text(page_row.row.get("line_value")))}</span>'
    )


def _formats(rendered: Mapping[str, str]) -> str:
    """One price, written three ways, with the stylesheet choosing.

    All three are in the document. The selector sets one attribute on the
    wrapper and a rule hides the two that are not chosen, so switching format
    neither rewrites an element nor asks anything of the network, and a reader
    whose browser ran no script sees the default rather than an empty cell.

    The two that are hidden are hidden from assistive technology as well.
    Otherwise a screen reader would read every price three times, in three
    formats, with no indication that two of them are not on screen.
    """
    return "".join(
        f'<span class="of of-{key}" data-odds-format-value="{escape(key)}"'
        f'{"" if key == DEFAULT_ODDS_FORMAT else " aria-hidden=\"true\""}>'
        f"{escape(text)}</span>"
        for key, _ in ODDS_FORMATS
        for text in (rendered[key],)
    )


def decimal_odds(value: Any, *, places: int = 2) -> str:
    """A decimal price, to two places, which is how every book writes one."""
    number = _finite(value)

    return "—" if number is None else f"{number:.{places}f}"


def _distribution_toggle(page_row: PageRow) -> str:
    """The control that opens one wager's forecast.

    A real button with ``aria-expanded``     and ``aria-controls``, naming the
    panel row it governs, rather than a styled span: the panel is a disclosure
    and the platform already has one. Rendered ``hidden`` for a reader whose
    script never ran, because the button would then do nothing and offering a
    dead control is worse than not offering it -- the panel row itself stays
    in the document and visible in that case, which is the no-script reading
    of the same information.

    It is therefore rendered expanded, because in the document as served it
    is: the state the attribute reports and the state of the row it points at
    are the same state, and the script closes both together or neither.
    """
    if page_row.distribution is None:
        return ""

    panel = _panel_id(page_row)

    return (
        f'<button type="button" class="wf-dist-toggle" data-dist-toggle'
        f' aria-expanded="true" aria-controls="{escape(panel)}" hidden>'
        "View Distribution</button>"
    )


def _panel_id(page_row: PageRow) -> str:
    """A document-unique id for one wager's panel.

    Derived from the wager rather than from a counter, so two builds of one
    release produce the same ids and the published bytes are a function of the
    release.
    """
    return "dist-" + "-".join(
        _slug(part)
        for part in (
            page_row.row.get("game_id"),
            page_row.row.get("player_id"),
            page_row.row.get("prop_type"),
            page_row.row.get("line_value"),
            page_row.side,
        )
    )


def _slug(value: Any) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in str(value if value is not None else "")
    )


def _cell(page_row: PageRow, key: str) -> tuple[str, str]:
    """One cell's markup and its class, by column key."""
    row = page_row.row

    if key == "player":
        name = escape(str(row.get("player_name") or ""))

        team = page_row.player_team

        if team:
            name += f' <span class="wf-side">{escape(team)}</span>'

        return name, "c-player"

    if key == "matchup":
        text = escape(page_row.matchup or "")

        tip = page_row.tip_off

        if text and tip:
            text += f'<span class="wf-tip">{escape(_clock(tip))}</span>'

        return text, "c-matchup"

    if key == "prop":
        return escape(prop_label(row.get("prop_type"))), "c-prop"

    if key == "pick":
        return _pick(page_row), "c-pick"

    if key == "projection":
        return escape(number_text(page_row.projection)), "c-proj num"

    if key == COLUMN_MODEL:
        probability = page_row.model_probability

        return (
            _formats(
                {
                    ODDS_FORMAT_AMERICAN: american(model_american(probability)),
                    ODDS_FORMAT_DECIMAL: decimal_odds(
                        model_decimal(probability)
                    ),
                    ODDS_FORMAT_PROBABILITY: percent(probability),
                }
            ),
            "c-model num",
        )

    if key == COLUMN_SPORTSBOOK:
        offer = page_row.routed.offer

        if offer is None:
            return "—", "c-book-price num"

        posted = offer.american_odds

        return (
            _formats(
                {
                    ODDS_FORMAT_AMERICAN: american(posted),
                    ODDS_FORMAT_DECIMAL: decimal_odds(
                        sportsbook_decimal(posted)
                    ),
                    ODDS_FORMAT_PROBABILITY: percent(
                        sportsbook_implied_probability(posted)
                    ),
                }
            ),
            "c-book-price num",
        )

    if key == COLUMN_DISTRIBUTION:
        return _distribution_toggle(page_row), "c-dist"

    if key == "ev":
        value = page_row.expected_value

        if page_row.routed.offer is None:
            return "—", "c-ev num"

        tone = "pos" if _number(value) > 0 else "neg"

        return (
            f'<span class="{tone}">{escape(signed_percent(value))}</span>',
            "c-ev num",
        )

    return _cta(page_row), "c-book"


def _distribution_rows(record: Mapping[str, Any]) -> tuple[
    tuple[str, float, bool], ...
]:
    """The published distribution as the rows the panel draws.

    The listed outcomes under their own labels, then the tail bucket under
    the label the release gave it, which is the one row that stands for more
    than one outcome and says so by being called ``N+``. Nothing is dropped
    and nothing is merged: these rows are the record, and their probabilities
    sum to one because the record's do.
    """
    support = list(record["distribution_support"])

    probability = list(record["distribution_probability"])

    low = int(record["distribution_most_likely_low"])

    high = int(record["distribution_most_likely_high"])

    rows = [
        (str(outcome), float(weight), low <= index <= high)
        for index, (outcome, weight) in enumerate(zip(support, probability))
    ]

    tail = _finite(record.get("distribution_tail_probability")) or 0.0

    outcome = record.get("distribution_tail_outcome")

    if outcome is not None:
        label = str(record.get("distribution_tail_label") or f"{outcome}+")

        rows.append((label, tail, low <= len(support) <= high))

    return tuple(rows)


def _distribution_panel(page_row: PageRow, *, columns: Sequence[str]) -> str:
    """One wager's forecast, as a table that is also the chart.

    Every outcome's probability is printed as a number *and* drawn as a bar in
    the same cell, so the chart is not a second representation that could
    disagree with the table and the information does not depend on being able
    to see a colour or a length. The bars are a width in a ``div``: no chart
    library, no external script, no additional request, and nothing to fail to
    load.

    The widths are relative to the largest probability in this distribution
    rather than to one, because a bar scaled to one is invisible for every
    outcome of a blocks line. The printed numbers are the published
    probabilities and are not scaled by anything.
    """
    record = page_row.distribution

    if record is None:
        return ""

    rows = _distribution_rows(record)

    peak = max((weight for _, weight, _ in rows), default=0.0) or 1.0

    drawn = "".join(
        f'<tr{" class=\"likely\"" if likely else ""}>'
        f'<th scope="row">{escape(label)}</th>'
        f'<td class="num">{escape(percent(weight, places=2))}</td>'
        f'<td class="bar"><div class="wf-bar" style="width:'
        f'{max(weight / peak * 100.0, 0.4):.2f}%"></div></td>'
        "</tr>"
        for label, weight, likely in rows
    )

    line = page_row.row.get("line_value")

    facts = [
        ("Model Projection", number_text(page_row.projection))
        if page_row.projection is not None
        else None,
        ("Current Line", line_text(line)),
        (
            "Model Probability (over)",
            percent(page_row.row.get(SIDE_MODEL_PROBABILITY[routing.SIDE_OVER])),
        ),
        (
            "Model Probability (under)",
            percent(
                page_row.row.get(SIDE_MODEL_PROBABILITY[routing.SIDE_UNDER])
            ),
        ),
    ]

    push = _finite(page_row.row.get("calibrated_p_push"))

    if push:
        facts.append(("Push", percent(push)))

    summary = "".join(
        f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"
        for item in facts
        if item is not None
        for label, value in (item,)
    )

    target = percent(
        record.get("distribution_most_likely_target_mass"), places=0
    )

    return (
        f'<tr class="wf-dist-row" id="{escape(_panel_id(page_row))}">'
        f'<td colspan="{len(columns)}">'
        '<section class="wf-dist-panel">'
        f'<h3>{escape(str(page_row.row.get("player_name") or ""))} — '
        f'{escape(prop_label(page_row.row.get("prop_type")))}</h3>'
        f'<dl class="wf-dist-facts">{summary}</dl>'
        f'<p class="wf-dist-kind">'
        f'{escape(str(record.get("distribution_label") or ""))}</p>'
        '<table class="wf-dist-table">'
        "<caption>"
        f'{escape(prop_label(page_row.row.get("prop_type")))} '
        "— probability of each outcome. "
        f"{escape(DISTRIBUTION_RANGE_NOTE.format(target=target))}"
        "</caption>"
        '<thead><tr><th scope="col">Outcome</th>'
        '<th scope="col" class="num">Probability</th>'
        '<th scope="col" class="bar">Share</th></tr></thead>'
        f"<tbody>{drawn}</tbody>"
        "</table>"
        f'<p class="wf-dist-note">{escape(DISTRIBUTION_NOTE)}</p>'
        "</section></td></tr>"
    )


def _search_text(page_row: PageRow) -> str:
    routed = page_row.routed

    book = routed.display_name or (routed.offer.vendor if routed.offer else "")

    return " ".join(
        str(value).lower()
        for value in (
            page_row.row.get("player_name"),
            prop_label(page_row.row.get("prop_type")),
            page_row.matchup,
            book,
            page_row.side,
        )
        if value
    )


def _tier_label(page_row: PageRow) -> str:
    routed = page_row.routed

    label = TIER_LABEL.get(routed.tier, routed.tier)

    if routed.tier == routing.TIER_PRIMARY and routed.best_overall:
        label += " · also best overall"

    return label


def _row(
    page_row: PageRow,
    *,
    columns: Sequence[str],
    presentation: Presentation,
    offered: bool,
) -> str:
    """One prediction, as one table row.

    The routing rule's own wording is rendered only for the QA presentation.
    A bettor is owed the sportsbook, the price and a way to get there; how the
    sportsbook came to be the one shown is a methodology question, and it is
    answered once below the table instead of restated on every row.
    """
    routed = page_row.routed

    cells: list[str] = []

    for key in columns:
        markup, klass = _cell(page_row, key)

        header = COLUMN_LABEL[key]

        if key == "book":
            badge = (
                '<span class="wf-tag claim">'
                f"{escape(_tier_label(page_row))}</span>"
                if offered
                else ""
            )

            why = ""

            if presentation.show_routing_reason:
                considered = len(routed.considered)

                why = (
                    f'<span class="wf-why">{escape(routed.reason)}. '
                    f"{considered} sportsbook offer"
                    f"{'' if considered == 1 else 's'} matched this exact "
                    "wager in the published data.</span>"
                )

            markup = f'<div class="wf-bookpick">{badge}{markup}{why}</div>'

        cells.append(
            f'<td class="{klass}" data-label="{escape(header)}">{markup}</td>'
        )

    book_key = (
        routing.normalise_vendor(routed.offer.vendor)
        if routed.offer is not None
        else ""
    )

    panel = (
        _distribution_panel(page_row, columns=columns)
        if COLUMN_DISTRIBUTION in columns
        else ""
    )

    return (
        "<tr data-row"
        f' data-offer="{"yes" if offered else "no"}"'
        f' data-prop="{escape(str(page_row.row.get("prop_type")))}"'
        f' data-book="{escape(book_key)}"'
        f' data-ev="{escape(str(page_row.expected_value))}"'
        f' data-model-p="{escape(str(page_row.model_probability))}"'
        f' data-sortname="{escape(str(page_row.row.get("player_name") or ""))}"'
        f' data-stable="{escape("|".join(page_row.stable_order))}"'
        f' data-tier="{escape(routed.tier)}"'
        + (f' data-panel="{escape(_panel_id(page_row))}"' if panel else "")
        + f' data-search="{escape(_search_text(page_row))}">'
        + "".join(cells)
        + "</tr>"
        + panel
    )


def _header_cell(key: str) -> str:
    """One column heading, in every format it has a name in.

    A price column's heading changes with the selector, because "Sportsbook
    Implied Probability" and "Sportsbook Odds" are different claims and a
    column of one under the heading of the other is a worse defect than
    either number on its own. Every other column has one name.
    """
    classes = "num" if COLUMN_NUMERIC[key] else ""

    labels = COLUMN_FORMAT_LABEL.get(key)

    inner = (
        _formats({form: labels[form] for form, _ in ODDS_FORMATS})
        if labels is not None
        else escape(COLUMN_LABEL[key])
    )

    return f'<th scope="col" class="{classes}">{inner}</th>'


def _head(columns: Sequence[str]) -> str:
    cells = "".join(_header_cell(key) for key in columns)

    return f"<thead><tr>{cells}</tr></thead>"


def _methodology(
    publication: Publication,
    *,
    registry: routing.AffiliateRegistry,
    columns: Sequence[str],
) -> str:
    """The collapsed section: what the columns mean, how a book is chosen.

    Everything that used to compete with the predictions for the top of the
    page lives here. It is one click away and it is on the page, which is the
    difference between a disclosure and a wall.
    """
    key_terms = {
        "pick": (
            "The side and line the model prefers on this prop, as published."
        ),
        COLUMN_MODEL: (
            "The model's own price for the pick, in the odds format you have "
            "chosen. As Model Probability it is the published probability "
            "that the pick wins, excluding pushes; as Model Odds it is that "
            "same probability written as the price at which it would break "
            "even. Switching format changes how the number is written and "
            "nothing else."
        ),
        COLUMN_SPORTSBOOK: (
            "The price the featured sportsbook is posting on this exact "
            "wager, as published in the release below, in the odds format you "
            "have chosen. As Sportsbook Implied Probability it is what that "
            "one price implies, including the sportsbook's margin, so a "
            "sportsbook's two sides add up to more than 100%."
        ),
        "ev": (
            "Expected value at that posted price, as published. It is "
            "computed at the price in the same row, so it is the number that "
            "answers whether the bet is worth taking. It does not change with "
            "the odds format."
        ),
        "projection": (
            "The model's projected number for the stat, where the release "
            "publishes one."
        ),
        COLUMN_DISTRIBUTION: (
            "Opens the model's full forecast for the statistic: the "
            "probability of every outcome, with the remaining mass above the "
            "last listed outcome shown as one bucket, so the probabilities "
            "add up to 100%."
        ),
        "matchup": "Who is playing, where the release publishes team names.",
    }

    terms = "".join(
        f"<dt>{escape(COLUMN_LABEL[key])}</dt><dd>{escape(key_terms[key])}</dd>"
        for key in columns
        if key in key_terms
    )

    books = ", ".join(sorted(book.display_name for book in registry.books))

    missing = registry.missing_facts()

    missing_note = ""

    if missing:
        missing_note = (
            "<p>WizardOfOdds has no verified identifier or destination for "
            f"{escape('; '.join(sorted(missing)))}, so those sportsbooks are "
            "never shown and never linked rather than guessed at.</p>"
        )

    return f"""<details class="wf-gloss">
<summary>Methodology, sportsbook selection and full provenance</summary>
<h3>What the columns mean</h3>
<dl>{terms}</dl>
<h3>How the featured sportsbook is chosen</h3>
<p>Every prediction is one exact wager, and several sportsbooks may post a
price on it. The featured sportsbook is picked by price first. Where one of
WizardOfOdds' partner sportsbooks ({escape(books)}) posts the wager, the
featured price is the best of those, and the card says Best Affiliate Price:
another sportsbook may be posting better odds, which is why that label does
not claim to be the best price anywhere. Where none of them posts it, the
featured price is the best across every sportsbook in the release and the card
says Best Available Price. Nothing about this selection changes a single model
number, and whether a sportsbook can be linked to is decided after it has been
selected, never before.</p>
{missing_note}
<h3>Affiliate disclosure</h3>
<p>{escape(AFFILIATE_DISCLOSURE)}</p>
<h3>What this page does not have</h3>
<p>Each game's tip-off is published and is shown beside the matchup, but
there is no ordering by start time: the cards are ordered by the model's
expected value and by nothing else that pretends to be a schedule. Every
model number here is copied from the release: this page fits nothing,
forecasts nothing and prices nothing of its own. The odds format control
rewrites published numbers into another notation and changes no quantity, and
every player is named by the active-player record the prediction was made
from rather than by anything a sportsbook called them.</p>
<h3>Full provenance</h3>
{_full_provenance(publication)}
</details>"""


def render(
    publication: Publication,
    rows: tuple[PageRow, ...],
    *,
    registry: routing.AffiliateRegistry,
    presentation: Presentation = CONSUMER_PREVIEW,
) -> str:
    """The page, as one self-contained document.

    Every prediction is in the HTML. The script filters and reorders rows that
    are already there, so the table and its buttons are readable without it,
    and no filter can leave a row on screen while the count says it is gone --
    showing and hiding writes an inline style that no stylesheet can override.

    Wagers nobody prices are a separate section rather than rows with dashes
    where a price belongs. There is nothing to bet on them, so they are not
    offered as though there were, and they are not dropped either: the
    published record is still rendered, below, under its own heading.
    """
    columns = visible_columns(rows)

    offered = tuple(row for row in rows if row.routed.offer is not None)

    unoffered = tuple(row for row in rows if row.routed.offer is None)

    props = sorted({str(row.row.get("prop_type")) for row in rows})

    books = sorted(
        {
            (
                routing.normalise_vendor(row.routed.offer.vendor),
                row.routed.display_name or row.routed.offer.vendor,
            )
            for row in offered
        }
    )

    controls = "".join(
        (
            '<div class="wide"><label for="f-search">Player search</label>'
            '<input id="f-search" type="search" inputmode="search" '
            'placeholder="Search player, prop or sportsbook" '
            'autocomplete="off"></div>',
            _select(
                "f-prop",
                "Prop",
                [("all", "All props")]
                + [(prop, prop_label(prop)) for prop in props],
            ),
            _select(
                "f-book",
                "Featured book",
                [("all", "All sportsbooks")] + [(key, name) for key, name in books],
            ),
            _select(
                "f-odds",
                "Odds format",
                list(ODDS_FORMATS),
            ),
            _select(
                "f-sort",
                "Sort",
                [
                    ("ev", "Expected value (highest first)"),
                    ("model", "Model Probability (highest first)"),
                    ("player", "Player (A–Z)"),
                ],
                klass="sort",
            ),
        )
    )

    picks = "".join(
        _row(row, columns=columns, presentation=presentation, offered=True)
        for row in offered
    )

    unlinked = (
        f'<p class="wf-disc">{escape(UNLINKED_NOTE)}</p>'
        if any(
            row.routed.destination is None and row.direct_url is None
            for row in offered
        )
        else ""
    )

    table = (
        f"""<div class="wf-tablewrap">
<table class="wf-table" id="picks">
{_head(columns)}
<tbody id="picks-body">{picks}</tbody>
</table>
</div>
<p class="wf-empty" id="empty" hidden>No prediction matches these filters.</p>
{unlinked}"""
        if offered
        else (
            '<p class="wf-empty">No sportsbook in this release posts a price '
            "on any of the model's wagers, so there is nothing to bet on this "
            "slate. Nothing was withheld and nothing was invented.</p>"
        )
    )

    pending_columns = tuple(
        key for key in columns if key not in UNOFFERED_COLUMNS
    )

    pending_rows = "".join(
        _row(
            row,
            columns=pending_columns,
            presentation=presentation,
            offered=False,
        )
        for row in unoffered
    )

    pending = (
        f"""<section class="wf-pending" id="pending">
<h2>No current offer (<span id="pending-count">{len(unoffered)}</span>)</h2>
<p class="wf-muted wf-small">The model has an opinion on these wagers and no
sportsbook in the release posts a price on them, so there is no bet to place
and no expected value to quote. They are listed because the model published
them, not as betting opportunities.</p>
<div class="wf-tablewrap">
<table class="wf-table">
{_head(pending_columns)}
<tbody>{pending_rows}</tbody>
</table>
</div>
</section>"""
        if unoffered
        else ""
    )

    embedded = {
        "note": (
            "Every value here is copied from the published release named "
            "below. This page computes no probability, distribution, edge or "
            "expected value. It converts a published probability and a "
            "published price into the odds format a reader chose, which "
            "changes how a number is written and not what it is."
        ),
        "odds_formats": [key for key, _ in ODDS_FORMATS],
        "release_id": publication.release_id,
        "rows_sha256": publication.rows_sha256,
        "slate_date": publication.slate_date,
    }

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<meta name="referrer" content="no-referrer-when-downgrade">
<title>NBA Player Props Predictions | WizardOfOdds</title>
<meta name="description" content="NBA player prop predictions for {escape(publication.slate_date)}: model probability, model odds, the best sportsbook price on each exact wager, its expected value and the model's full outcome distribution.">
<style>{STYLE}</style>
</head>
<body class="wf-body">
<div id="{EMBED_WRAPPER_ID}" {ODDS_FORMAT_ATTRIBUTE}="{escape(DEFAULT_ODDS_FORMAT)}">
<div class="wf">
<main>
<p class="wf-note {escape(presentation.tone)}"><b>{escape(presentation.banner)}</b><span>{escape(presentation.detail)}</span></p>
<h1 class="wf-h1">NBA Player Props Predictions</h1>
<p class="wf-lead">Model projections against live sportsbook prices for the {escape(publication.slate_date)} slate: {len(offered)} wager{'' if len(offered) == 1 else 's'} a sportsbook is pricing, with the model's probability, the model's own odds, the expected value at the posted price, and the full distribution of outcomes each probability came out of. Prices are shown in the odds format you choose.</p>
{_header_provenance(publication)}
<section class="wf-controls" aria-label="Filter and sort">{controls}</section>
<p class="wf-count" id="count" aria-live="polite">Showing {len(offered)} of {len(offered)} prediction{'' if len(offered) == 1 else 's'}</p>
<p class="wf-disc">{escape(SHORT_DISCLOSURE)}</p>
{table}
{pending}
{_methodology(publication, registry=registry, columns=columns)}
</main>
<footer class="wf-foot">
<p>Published authority: incumbent. The research candidate is shadow-only and never appears here.</p>
<p>Odds change without notice. Nothing on this page is a guarantee of profit, and no offer here is risk-free unless the sportsbook says so in its own terms. 21+ where applicable.</p>
</footer>
</div>
</div>
<script type="application/json" id="source-note">{_json_for_html(embedded)}</script>
<script id="page-controls">{SCRIPT}</script>
<script id="frame-height">{EMBED_SCRIPT}</script>
</body>
</html>
"""


# ----------------------------------------------------------------------
# building and publishing
# ----------------------------------------------------------------------


def page_manifest(
    publication: Publication,
    rows: tuple[PageRow, ...],
    *,
    registry: routing.AffiliateRegistry,
    fallback: fallback_destinations.FallbackDestinations,
    presentation: Presentation,
    page_sha256: str,
) -> dict[str, Any]:
    """What this build was, in a form a check can read rather than parse HTML."""
    by_tier: dict[str, int] = {
        routing.TIER_PRIMARY: 0,
        routing.TIER_FALLBACK: 0,
        routing.TIER_NONE: 0,
    }

    for row in rows:
        by_tier[row.routed.tier] = by_tier.get(row.routed.tier, 0) + 1

    return {
        "affiliate_registry": {
            "missing_facts": {
                book: list(facts)
                for book, facts in sorted(registry.missing_facts().items())
            },
            "primary_books": [
                book.display_name
                for book in registry.books_in_tier(routing.TIER_PRIMARY)
            ],
            "redirect_path": registry.redirect_path,
            "routable_books": sorted(
                book.display_name for book in registry.books if book.routable
            ),
            "source": Path(registry.source).name if registry.source else "",
            "tie_priority": [
                book.display_name
                for book in sorted(
                    registry.books_in_tier(routing.TIER_PRIMARY),
                    key=lambda entry: entry.tie_priority,
                )
            ],
        },
        "contract_version": 5,
        "embed": {
            "message_key": EMBED_MESSAGE_KEY,
            "parent_origins": list(PARENT_ORIGINS),
            "wrapper_id": EMBED_WRAPPER_ID,
        },
        "fallback_destinations": {
            "linkable_books": dict(fallback.linkable_books()),
            "rel": fallback_destinations.REQUIRED_REL,
            "source": Path(fallback.source).name if fallback.source else "",
        },
        "mode": publication.mode,
        "page_name": PAGE_NAME,
        "page_sha256": page_sha256,
        "presentation": {
            "banner": presentation.banner,
            "routing_reason_shown": presentation.show_routing_reason,
            "tone": presentation.tone,
        },
        "provenance": {
            name: publication.provenance.get(name)
            for name, _ in RENDERED_PROVENANCE
        },
        "released": {
            "manifest_relative": publication.manifest_relative,
            "release_id": publication.release_id,
            "rows_relative": publication.rows_relative,
            "rows_sha256": publication.rows_sha256,
            "row_count": len(publication.rows),
            "slate_date": publication.slate_date,
        },
        "rendered": {
            "affiliate_cta_count": sum(
                1 for row in rows if row.routed.has_affiliate_destination
            ),
            "by_tier": by_tier,
            "columns": list(visible_columns(rows)),
            "direct_cta_count": sum(
                1 for row in rows if row.direct_url is not None
            ),
            "no_offer_count": sum(
                1 for row in rows if row.routed.offer is None
            ),
            "offered_count": sum(
                1 for row in rows if row.routed.offer is not None
            ),
            "prediction_count": len(rows),
            "published_row_count": len(publication.rows),
        },
        "routed": [
            {
                "best_overall": row.routed.best_overall,
                "destination": row.routed.destination,
                "direct_destination": row.direct_url,
                "display_name": row.routed.display_name,
                "offers_considered": len(row.routed.considered),
                "selected_odds": (
                    row.routed.offer.american_odds
                    if row.routed.offer is not None
                    else None
                ),
                "selected_vendor": (
                    row.routed.offer.vendor
                    if row.routed.offer is not None
                    else None
                ),
                "tier": row.routed.tier,
                "wager": {
                    "game_id": row.wager.game_id,
                    "line_value": row.wager.line_value,
                    "player_id": row.wager.player_id,
                    "prop_type": row.wager.prop_type,
                    "side": row.wager.side,
                },
            }
            for row in rows
        ],
    }


def build_page(
    transport: Any,
    *,
    registry: routing.AffiliateRegistry,
    fallback: fallback_destinations.FallbackDestinations | None = None,
    presentation: Presentation = CONSUMER_PREVIEW,
    expected: ExpectedRelease | None = None,
) -> dict[str, Any]:
    """Read one publication and render the page and its manifest.

    No clock is read, so the same release renders the same bytes every time.

    ``fallback`` defaults to the committed destination file, so an ordinary
    build links the books the site has established destinations for. A caller
    passing one explicitly gets exactly that one, which is how a test states
    which books are linkable without editing committed configuration.

    ``presentation`` defaults to the consumer page. A caller wanting the QA
    page has to ask for it, so no ordinary build can acquire the routing
    vocabulary or the test-data banner by accident.

    ``expected`` pins which release may be read. Defaults to none, which is
    the human invocation: build the page for whatever is current. Automation
    supplies it, because the release it means is the one it just published.
    """
    if fallback is None:
        fallback = fallback_destinations.FallbackDestinations.load()

    publication = load_publication(transport, expected=expected)

    rows = page_rows(publication, registry=registry, fallback=fallback)

    html = render(
        publication, rows, registry=registry, presentation=presentation
    )

    page_bytes = html.encode("utf-8")

    manifest = page_manifest(
        publication,
        rows,
        registry=registry,
        fallback=fallback,
        presentation=presentation,
        page_sha256=publisher.digest(page_bytes),
    )

    return {
        "files": {
            PAGE_NAME: page_bytes,
            PAGE_MANIFEST_NAME: publisher.serialise(manifest),
        },
        "manifest": manifest,
        "publication": publication,
        "rows": rows,
    }


def publish_page(
    *,
    source_transport: Any,
    page_transport: Any,
    registry: routing.AffiliateRegistry,
    fallback: fallback_destinations.FallbackDestinations | None = None,
    presentation: Presentation = CONSUMER_PREVIEW,
    prefix: str = "",
    expected: ExpectedRelease | None = None,
) -> dict[str, Any]:
    """Build the page from one publication and write it to the page canary.

    Two transports on purpose. The release is read through one and the page
    written through another, so the destination that holds the feed is only
    ever read from here and the destination that holds the page is the only one
    written to.

    ``prefix`` puts the two files in a child directory of the destination
    instead of at its root, which is how a proof page is published beside the
    page without the possibility of writing over it. It may only name a
    directory below the destination, and it is checked as given rather than
    cleaned up first: a prefix that cannot be published where it says it
    publishes is refused, rather than quietly reinterpreted as a relative path
    and written somewhere the caller did not ask for.
    """
    if prefix and not publisher._is_safe_remote_segment_path(prefix):
        raise PageRefused(
            f"{prefix!r} does not address a directory below the page "
            "destination, so this page is not published"
        )

    built = build_page(
        source_transport,
        registry=registry,
        fallback=fallback,
        presentation=presentation,
        expected=expected,
    )

    relative = {
        name: f"{prefix}/{name}" if prefix else name for name in built["files"]
    }

    for name, target in sorted(relative.items()):
        if not publisher._is_safe_remote_segment_path(target):
            raise PageRefused(
                f"{target!r} does not address a path below the page "
                "destination, so this page is not published"
            )

        page_transport.write_bytes(target, built["files"][name])

    readback: dict[str, Any] = {}

    for name, target in sorted(relative.items()):
        served = page_transport.read_bytes(target)

        if served != built["files"][name]:
            raise PageRefused(
                f"{target} at the page destination does not read back as what "
                "was written, so the page is not published"
            )

        if hasattr(page_transport, "public_readback"):
            public = page_transport.public_readback(target)

            if public != built["files"][name]:
                raise PageRefused(
                    f"{target} fetched from "
                    f"{page_transport.public_url(target)} is not what was "
                    "published, so what a reader would get is unproven"
                )

            readback[target] = {
                "sha256": publisher.digest(public),
                "url": page_transport.public_url(target),
            }

    publication = built["publication"]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": built["manifest"],
        "prefix": prefix,
        "public_readback": readback,
        # The release this page is about, named here rather than only inside
        # the page manifest, so the caller that pinned the build can check the
        # pin held without parsing the page's own record.
        "source_release": {
            "expected": expected.payload() if expected is not None else None,
            "release_id": publication.release_id,
            "rows_relative": publication.rows_relative,
            "rows_sha256": publication.rows_sha256,
            "slate_date": publication.slate_date,
        },
        "written": sorted(relative.values()),
    }


# ----------------------------------------------------------------------
# the entry point
# ----------------------------------------------------------------------


#: The feed a page of each mode is built from, and the destination it is
#: written to. Held here so that the difference between publishing a preview
#: and publishing the page a reader fetches is a mode rather than four
#: arguments somebody has to get right, which is what makes a later reviewed
#: activation an approval instead of a change to this file.
MODE_DESTINATIONS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        publisher.MODE_CANARY: MappingProxyType(
            {
                "source": "wizardofodds_nba_canary",
                "page": "wizardofodds_nba_player_props_page_canary",
                "prefix": PREVIEW_PREFIX,
            }
        ),
        publisher.MODE_LIVE: MappingProxyType(
            {
                "source": "wizardofodds_nba_live",
                "page": "wizardofodds_nba_player_props_page_live",
                # The served root: index.html where a reader asks for it.
                "prefix": "",
            }
        ),
    }
)


def assert_mode_is_permitted(mode: str, *, project_root: Path) -> dict[str, Any]:
    """Whether this process may publish a page in this mode at all.

    The feed publisher's switch, unchanged and read the same way. The page is
    the same claim about the same numbers at the same host, so it is gated by
    the same committed decision rather than by a second one that could drift
    away from it -- and live is DISABLED there, which is what keeps this entry
    point's live mode a configured path rather than an available one.

    The canary is read too, rather than waved through, so a repository that
    disabled canary publishing stops publishing previews as well as releases.
    """
    switch = publisher.read_publishing_switch(mode, project_root=project_root)

    publisher.assert_mode_enabled(switch)

    return switch.payload()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the WizardOfOdds NBA player-props page from a published "
            "incumbent release. The mode decides which feed it is built from "
            "and which destination it is written to, and live is refused by "
            "the publishing switch, not by this entry point lacking one."
        )
    )

    parser.add_argument(
        "--mode",
        choices=sorted(MODE_DESTINATIONS),
        default=publisher.MODE_CANARY,
        help="canary builds from the feed canary and writes the preview. "
        "live builds from the live feed and writes the page a reader fetches; "
        "it is refused unless the committed switch, the publish variable and "
        "the approval token all say so.",
    )

    parser.add_argument(
        "--source-destination-name",
        default=None,
        help="named destination the published release is read from. Defaults "
        "to the feed of this mode.",
    )

    parser.add_argument(
        "--source-destination-root",
        default=None,
        help="read the release from a local directory instead",
    )

    parser.add_argument(
        "--page-destination-name",
        default=None,
        help="named destination the page is written to. Defaults to the page "
        "destination of this mode.",
    )

    parser.add_argument(
        "--page-destination-root",
        default=None,
        help="write the page to a local directory instead",
    )

    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "directory below the page destination the page is written to. "
            "Defaults to the preview directory for a canary, so the ordinary "
            "invocation publishes a preview and does not write over anything "
            "already published beside it, and to the served root for live."
        ),
    )

    parser.add_argument(
        "--expect-release-id",
        default=None,
        help="refuse unless the destination is serving this release. Supplied "
        "by automation, which means the release it just published rather than "
        "whatever is current by the time the page is built.",
    )

    parser.add_argument(
        "--expect-rows-sha256",
        default=None,
        help="refuse unless that release's rows are these exact bytes",
    )

    parser.add_argument(
        "--project-root",
        default=None,
        help="repository root the configuration is read from",
    )

    return parser


def _transport_for(
    *, mode: str, name: str | None, root: str | None, project_root: Path
) -> Any:
    if root is not None:
        return publisher.LocalDirectoryTransport(Path(root))

    destination = publisher.resolve_destination(
        mode,
        publisher.read_destinations(project_root=project_root),
        name=name,
    )

    return publisher.build_transport(destination)


def expected_release_from(arguments: argparse.Namespace) -> ExpectedRelease | None:
    """The pin the arguments describe, or none, and never half of one.

    Half a pin is worse than none: a run that supplied only the id would
    accept the right release republished with different bytes, while looking
    from the outside like it had pinned the build.
    """
    release_id = arguments.expect_release_id

    rows_sha256 = arguments.expect_rows_sha256

    if release_id is None and rows_sha256 is None:
        return None

    if not release_id or not rows_sha256:
        raise PageRefused(
            "--expect-release-id and --expect-rows-sha256 are supplied "
            "together or not at all: one without the other pins which "
            "release without pinning which bytes of it"
        )

    return ExpectedRelease(release_id=release_id, rows_sha256=rows_sha256)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    root = Path(arguments.project_root or PROJECT_ROOT)

    defaults = MODE_DESTINATIONS[arguments.mode]

    try:
        switch = assert_mode_is_permitted(arguments.mode, project_root=root)

        expected = expected_release_from(arguments)
    except (PageRefused, publisher.PublishingRefused) as error:
        print(json.dumps({"outcome": "REFUSED", "reason": str(error)}, indent=2))

        return publisher.EXIT_REFUSED

    registry = routing.AffiliateRegistry.load(project_root=root)

    fallback = fallback_destinations.FallbackDestinations.load(project_root=root)

    source = _transport_for(
        mode=arguments.mode,
        name=arguments.source_destination_name or defaults["source"],
        root=arguments.source_destination_root,
        project_root=root,
    )

    page = _transport_for(
        mode=arguments.mode,
        name=arguments.page_destination_name or defaults["page"],
        root=arguments.page_destination_root,
        project_root=root,
    )

    try:
        receipt = publish_page(
            source_transport=source,
            page_transport=page,
            registry=registry,
            fallback=fallback,
            presentation=CONSUMER_PREVIEW,
            prefix=(
                arguments.prefix
                if arguments.prefix is not None
                else defaults["prefix"]
            ),
            expected=expected,
        )
    except (
        PageRefused,
        routing.RoutingRefused,
        fallback_destinations.DestinationRefused,
        publisher.PublishingRefused,
    ) as error:
        print(json.dumps({"outcome": "REFUSED", "reason": str(error)}, indent=2))

        return publisher.EXIT_REFUSED
    finally:
        source.close()

        page.close()

    print(json.dumps({**receipt, "mode": arguments.mode, "switch": switch}, indent=2, sort_keys=True))

    return publisher.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
