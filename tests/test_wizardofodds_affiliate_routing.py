"""The frozen tier rule, and the two things it must never do.

The rule is a commercial preference stated openly: among the books that offer
exactly the bet the model picked, a primary affiliate beats a secondary
affiliate, a secondary affiliate beats a non-affiliate, and only inside the
chosen tier does the best price win. These tests pin the rule, and they pin the
two boundaries that make it safe to state it out loud.

The first boundary is the model. Routing is a product decision taken after the
numbers exist, so no probability, edge or expected value may differ because a
book pays commission. That is asserted directly rather than reasoned about.

The second is the identifier. A book's name is not an identifier and a
resemblance is not evidence: the site's redirect answers an unrecognised book
by sending it to Bovada, so an invented sportsbook key would not fail loudly,
it would quietly bill someone else's traffic to Bovada. Five of the eight books
have no proven identifier, and these tests pin that they cannot be matched and
cannot be linked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import wizardofodds_affiliate_routing as routing  # noqa: E402

#: A committed registry, which is what production routes with.
COMMITTED = routing.AffiliateRegistry.load(project_root=PROJECT)

#: The four the product requirement calls primary, and the four it calls
#: secondary. Read from the committed registry rather than restated, so a
#: registry that disagreed with the requirement would fail here.
PRIMARY_NAMES = {
    book.display_name for book in COMMITTED.books_in_tier(routing.TIER_PRIMARY)
}

SECONDARY_NAMES = {
    book.display_name for book in COMMITTED.books_in_tier(routing.TIER_SECONDARY)
}


def registry(**keys: str) -> routing.AffiliateRegistry:
    """A registry whose books carry the identifiers a test needs.

    Built directly rather than loaded, because the identifier-dependent half of
    the rule cannot be exercised through the committed file: five of the eight
    books have no proven identifier there, and inventing one to make a test
    pass is the defect the registry exists to prevent. ``keys`` maps a registry
    id to the vendor string that book should answer to in this test.
    """
    assigned = set(keys.values())

    books = []

    for book in COMMITTED.books:
        vendor = keys.get(book.registry_id)

        # An assignment wins outright, and any committed key it collides with
        # is dropped rather than left to make two books claim one identifier.
        vendor_keys = (
            (vendor,)
            if vendor
            else tuple(key for key in book.vendor_keys if key not in assigned)
        )

        books.append(
            routing.Book(
                registry_id=book.registry_id,
                display_name=book.display_name,
                tier=book.tier,
                vendor_keys=vendor_keys,
                redirect_book=book.redirect_book,
                missing=book.missing,
            )
        )

    return routing.AffiliateRegistry(
        tuple(books),
        redirect_path=COMMITTED.redirect_path,
        redirect_parameter=COMMITTED.redirect_parameter,
        non_affiliate_names=COMMITTED.non_affiliate_names,
        source=COMMITTED.source,
    )


def row(
    *,
    vendor: str,
    over: int | None = -110,
    under: int | None = -110,
    side: str = "over",
    game_id: int = 9901,
    player_id: int = 1101,
    prop_type: str = "points",
    line_value: float = 18.5,
    quote_id: int | None = None,
) -> dict:
    """One published quote, carrying only the fields routing reads."""
    return {
        "game_id": game_id,
        "line_value": line_value,
        "model_preferred_side": side,
        "over_odds": over,
        "player_id": player_id,
        "prop_type": prop_type,
        "quote_id": quote_id if quote_id is not None else abs(hash(vendor)) % 10_000,
        "under_odds": under,
        "vendor": vendor,
    }


def route_rows(rows: list[dict], *, reg: routing.AffiliateRegistry) -> routing.Routed:
    """Route the first row's wager against the whole list."""
    _, routed = routing.route_row(rows[0], rows=rows, registry=reg)

    return routed


# ======================================================================
# 1-4: the tier rule
# ======================================================================


def test_a_primary_affiliate_wins_over_secondary_and_fallback():
    """Case 1. All three tiers offer it; primary takes the click.

    The non-affiliate has by far the best price and does not win, which is the
    rule being a tier rule rather than a price rule.
    """
    reg = registry(bovada="bovada", busr="busr")

    rows = [
        row(vendor="bovada", over=-130),
        row(vendor="busr", over=100),
        row(vendor="fanduel", over=250),
    ]

    routed = route_rows(rows, reg=reg)

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer is not None

    assert routed.offer.vendor == "bovada"

    assert routed.display_name == "Bovada"

    assert len(routed.considered) == 3

    assert routed.best_overall is False


def test_the_best_price_wins_among_several_primary_offers():
    """Case 2. Inside the tier, price decides -- as a payout, not an integer.

    ``+105`` is the best of these three and is numerically between the other
    two when read as a signed integer, so a router comparing the raw numbers
    would pick wrongly in one direction or the other.
    """
    reg = registry(bovada="bovada", betus="betus", betonline="betonlineag")

    rows = [
        row(vendor="bovada", over=-120),
        row(vendor="betus", over=105),
        row(vendor="betonlineag", over=-105),
    ]

    routed = route_rows(rows, reg=reg)

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == "betus"

    assert routed.offer.american_odds == 105

    assert routed.display_name == "BetUS"


def test_a_secondary_affiliate_wins_when_no_primary_offers_it():
    """Case 3. No primary book quotes the wager, so the secondary tier is next."""
    reg = registry(bovada="bovada", busr="busr", xbet="xbet")

    rows = [
        row(vendor="busr", over=-115),
        row(vendor="xbet", over=-108),
        row(vendor="fanduel", over=300),
    ]

    routed = route_rows(rows, reg=reg)

    assert routed.tier == routing.TIER_SECONDARY

    assert routed.offer.vendor == "xbet"

    assert routed.display_name == "Xbet"

    assert routed.best_overall is False


def test_the_fallback_tier_is_reached_only_when_no_affiliate_offers_it():
    """Case 4. Neither affiliate tier quotes it, so the best of the rest wins.

    A fallback selection is the best price in the whole universe by
    construction, which is why it is the one tier that may be labelled best
    available.
    """
    reg = registry(bovada="bovada", busr="busr")

    rows = [
        row(vendor="fanduel", over=-130),
        row(vendor="draftkings", over=-105),
        row(vendor="betmgm", over=-140),
    ]

    routed = route_rows(rows, reg=reg)

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.offer.vendor == "draftkings"

    assert routed.display_name == "DraftKings"

    assert routed.best_overall is True

    assert routed.destination is None


# ======================================================================
# 5: nothing to bet
# ======================================================================


def test_no_matching_offer_fabricates_no_call_to_action():
    """Case 5. Nobody quotes this side, so there is no selection at all."""
    reg = registry(bovada="bovada")

    rows = [row(vendor="bovada", over=None, side="over")]

    routed = route_rows(rows, reg=reg)

    assert routed.tier == routing.TIER_NONE

    assert routed.offer is None

    assert routed.book is None

    assert routed.destination is None

    assert routed.display_name == ""

    assert routed.has_affiliate_destination is False

    assert routed.considered == ()


def test_an_empty_offer_universe_is_answered_rather_than_guessed():
    reg = registry(bovada="bovada")

    routed = routing.route((), registry=reg)

    assert routed.tier == routing.TIER_NONE

    assert routed.offer is None

    assert routed.destination is None


# ======================================================================
# 6-9: an offer has to be the same bet
# ======================================================================


@pytest.mark.parametrize(
    ("label", "difference"),
    [
        ("player", {"player_id": 2201}),
        ("prop", {"prop_type": "rebounds"}),
        ("line", {"line_value": 19.5}),
        ("game", {"game_id": 9902}),
    ],
)
def test_a_different_bet_is_not_a_cheaper_version_of_this_one(label, difference):
    """Cases 6, 7, 9 and the game dimension they imply.

    Each of these books offers a far better price on something adjacent. None
    of them can be routed to, because a different player, prop, line or game is
    a different wager rather than the same wager priced better.
    """
    reg = registry(bovada="bovada", busr="busr")

    rows = [
        row(vendor="fanduel", over=-130),
        row(vendor="bovada", over=400, **difference),
        row(vendor="busr", over=400, **difference),
    ]

    routed = route_rows(rows, reg=reg)

    assert routed.tier == routing.TIER_FALLBACK, (
        f"a book quoting a different {label} was treated as offering this wager"
    )

    assert routed.offer.vendor == "fanduel"

    assert len(routed.considered) == 1


def test_the_wrong_side_cannot_match():
    """Case 8. The model picked under; an over price is not an offer on it.

    Both books quote this exact contract, so the row survives the contract
    match and is excluded on the side alone: ``bovada`` prices the over
    generously and the under not at all.
    """
    reg = registry(bovada="bovada")

    rows = [
        row(vendor="fanduel", over=500, under=-130, side="under"),
        row(vendor="bovada", over=500, under=None, side="under"),
    ]

    routed = route_rows(rows, reg=reg)

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.offer.vendor == "fanduel"

    assert routed.offer.american_odds == -130


def test_the_side_decides_which_published_price_is_the_offer():
    reg = registry(bovada="bovada")

    quote = row(vendor="bovada", over=120, under=-150, side="over")

    assert routing.offer_in(quote, side="over").american_odds == 120

    assert routing.offer_in(quote, side="under").american_odds == -150


def test_an_unknown_side_is_refused_rather_than_assumed():
    with pytest.raises(routing.RoutingRefused, match="not one of"):
        routing.wager_of(row(vendor="bovada", side="middle"))


def test_a_row_missing_its_contract_fields_is_refused():
    quote = row(vendor="bovada")

    del quote["line_value"]

    with pytest.raises(routing.RoutingRefused, match="line_value"):
        routing.wager_of(quote)


# ======================================================================
# 10: the tie-break
# ======================================================================


def test_an_exact_price_tie_breaks_on_stable_identifier_ordering():
    """Case 10. Equal best price, so the lower sportsbook identifier wins.

    ``betonlineag`` sorts before ``betus`` and before ``bovada``, and the
    answer does not depend on which order the rows arrived in.
    """
    reg = registry(bovada="bovada", betus="betus", betonline="betonlineag")

    rows = [
        row(vendor="betus", over=-110),
        row(vendor="bovada", over=-110),
        row(vendor="betonlineag", over=-110),
    ]

    forwards = route_rows(rows, reg=reg)

    backwards = route_rows(list(reversed(rows)), reg=reg)

    assert forwards.offer.vendor == "betonlineag"

    assert backwards.offer.vendor == "betonlineag"


def test_the_tie_break_is_the_identifier_and_then_the_quote_id():
    """Two quotes from one book still order deterministically.

    The publisher's logical key makes this impossible upstream, which is
    exactly why the router does not get to assume it.
    """
    reg = registry(bovada="bovada")

    rows = [
        row(vendor="bovada", over=-110, quote_id=900),
        row(vendor="bovada", over=-110, quote_id=100),
    ]

    assert route_rows(rows, reg=reg).offer.quote_id == 100

    assert route_rows(list(reversed(rows)), reg=reg).offer.quote_id == 100


def test_equal_payouts_from_differently_written_odds_still_tie_break():
    """``+100`` and ``-100`` are not the same payout, ``100`` and ``100.0`` are."""
    assert routing.payout_multiple(100) == 1.0

    assert routing.payout_multiple(-100) == 1.0

    assert routing.payout_multiple(100.0) == routing.payout_multiple(100)

    assert routing.payout_multiple(-110) < routing.payout_multiple(-105)

    assert routing.payout_multiple(-105) < routing.payout_multiple(100)

    assert routing.payout_multiple(100) < routing.payout_multiple(150)


@pytest.mark.parametrize("unusable", [None, "", "even", True, False, float("nan")])
def test_an_unusable_price_is_not_an_offer(unusable):
    assert routing.payout_multiple(unusable) is None


def test_zero_is_not_a_price():
    """American odds have no zero, so a zero is a bad value rather than evens."""
    assert routing.payout_multiple(0) is None


# ======================================================================
# 11-13: where a link may come from
# ======================================================================


def test_an_affiliate_destination_comes_only_from_the_registry():
    """Case 11. The registry is the only source, and it is addressed one way.

    Every destination the committed registry will produce is its redirect path
    with its own parameter and a book it records. Nothing is assembled from a
    book's name, and there is no default.
    """
    produced = {
        book.display_name: COMMITTED.destination_for(book.vendor_keys[0])
        for book in COMMITTED.books
        if book.routable
    }

    assert produced == {
        "BetOnline": "/tools/odds-scanner/predictions/api/go.php?book=betonline",
        "BetUS": "/tools/odds-scanner/predictions/api/go.php?book=betus",
        "Bovada": "/tools/odds-scanner/predictions/api/go.php?book=bovada",
    }

    for destination in produced.values():
        assert destination.startswith(
            f"{COMMITTED.redirect_path}?{COMMITTED.redirect_parameter}="
        )

    assert COMMITTED.destination_for("not-a-book-anyone-has-heard-of") is None


def test_a_book_with_no_verified_destination_never_produces_a_link():
    """Case 12. The five unproven books are recorded as unproven and stay so.

    Not a gap: the site's redirect answers an unrecognised book by sending it
    to Bovada, so a guessed key would silently bill another book's traffic to
    Bovada rather than failing.
    """
    unproven = [book for book in COMMITTED.books if not book.destination_known]

    assert {book.display_name for book in unproven} == {
        "BetAnything",
        "BUSR",
        "Betwhale",
        "Voltage Bet",
        "Xbet",
    }

    for book in unproven:
        assert book.vendor_keys == ()

        assert book.routable is False

        assert book.missing, f"{book.display_name} is unproven but says nothing"

    assert set(COMMITTED.missing_facts()) == {
        book.display_name for book in unproven
    }


def test_a_book_with_no_destination_is_selected_without_being_linked():
    """Routable and linkable are separate, and the router keeps them separate."""
    reg = registry(busr="busr")

    unlinkable = routing.Book(
        registry_id="busr",
        display_name="BUSR",
        tier=routing.TIER_SECONDARY,
        vendor_keys=("busr",),
        redirect_book=None,
        missing=("the affiliate destination",),
    )

    reg = routing.AffiliateRegistry(
        (unlinkable,),
        redirect_path=COMMITTED.redirect_path,
        redirect_parameter=COMMITTED.redirect_parameter,
    )

    routed = route_rows([row(vendor="busr", over=-105)], reg=reg)

    assert routed.tier == routing.TIER_SECONDARY

    assert routed.offer.vendor == "busr"

    assert routed.destination is None

    assert routed.has_affiliate_destination is False


def test_a_non_affiliate_sportsbook_never_receives_an_affiliate_url():
    """Case 13. Every book outside the registry, selected and never linked."""
    reg = registry(bovada="bovada")

    for vendor in sorted(COMMITTED.non_affiliate_names):
        routed = route_rows([row(vendor=vendor, over=-105)], reg=reg)

        assert routed.tier == routing.TIER_FALLBACK, vendor

        assert routed.destination is None, vendor

        assert routed.is_affiliate is False, vendor

    assert COMMITTED.destination_for("betanysports") is None

    assert COMMITTED.book_for("betanysports") is None


def test_betanysports_is_not_treated_as_betanything():
    """The resemblance the requirement names, pinned as a non-match.

    ``betanysports`` is a real affiliate of the site. It is not one of the
    eight books this page routes, it is labelled BetAnySports by the site's own
    book-name map, and nothing may quietly make it BetAnything.
    """
    assert COMMITTED.book_for("betanysports") is None

    assert COMMITTED.tier_of("betanysports") == routing.TIER_FALLBACK

    assert COMMITTED.destination_for("betanysports") is None

    betanything = next(
        book for book in COMMITTED.books if book.display_name == "BetAnything"
    )

    assert "betanysports" not in betanything.vendor_keys

    assert betanything.vendor_keys == ()

    assert "betanysports" not in dict(COMMITTED.non_affiliate_names)


def test_the_registry_refuses_two_books_claiming_one_identifier():
    with pytest.raises(routing.RoutingRefused, match="claimed by both"):
        routing.AffiliateRegistry(
            (
                routing.Book("a", "A", routing.TIER_PRIMARY, ("same",), "a", ()),
                routing.Book("b", "B", routing.TIER_SECONDARY, ("same",), "b", ()),
            ),
            redirect_path="/go",
            redirect_parameter="book",
        )


def test_the_registry_refuses_a_tier_it_does_not_recognise():
    with pytest.raises(routing.RoutingRefused, match="not one of"):
        routing.AffiliateRegistry(
            (routing.Book("a", "A", routing.TIER_FALLBACK, ("x",), "a", ()),),
            redirect_path="/go",
            redirect_parameter="book",
        )


def test_an_unreadable_registry_is_refused_rather_than_defaulted(tmp_path):
    missing = tmp_path / "nowhere.json"

    with pytest.raises(routing.RoutingRefused, match="none will be invented"):
        routing.AffiliateRegistry.load(registry_path=missing)


def test_the_committed_registry_matches_the_product_requirement():
    assert PRIMARY_NAMES == {"Bovada", "BetOnline", "BetUS", "BetAnything"}

    assert SECONDARY_NAMES == {"BUSR", "Voltage Bet", "Betwhale", "Xbet"}

    assert PRIMARY_NAMES.isdisjoint(SECONDARY_NAMES)

    assert len(COMMITTED.books) == 8


def test_the_registry_holds_no_tracking_link_or_credential():
    """The destinations live on the server; this file holds names, not secrets."""
    raw = (PROJECT / routing.REGISTRY_RELATIVE).read_text(encoding="utf-8")

    for forbidden in (
        "record.revenuenetwork.com",
        "record.commissionkings.ag",
        "record.revmasters.com",
        "btag=",
        "?ref=",
        "aff=",
        "PRIVATE KEY",
    ):
        assert forbidden not in raw, f"{forbidden} does not belong in the registry"


# ======================================================================
# 14-17: routing cannot move a model number
# ======================================================================


def _model_fields(quote: dict) -> dict:
    return {
        name: quote[name]
        for name in (
            "calibrated_edge_over",
            "calibrated_edge_under",
            "calibrated_ev_over",
            "calibrated_ev_under",
            "calibrated_q_over_nonpush",
            "calibrated_q_under_nonpush",
            "market_devig_q_over",
            "market_devig_q_under",
            "model_preferred_edge",
            "model_preferred_ev",
        )
    }


def priced_row(vendor: str, over: int) -> dict:
    quote = row(vendor=vendor, over=over)

    quote.update(
        {
            "calibrated_edge_over": 0.0294805232517650,
            "calibrated_edge_under": -0.0294805232517650,
            "calibrated_ev_over": 0.0108264534806423,
            "calibrated_ev_under": -0.1017355443897331,
            "calibrated_q_over_nonpush": 0.5294805232517650,
            "calibrated_q_under_nonpush": 0.4705194767482350,
            "market_devig_q_over": 0.5,
            "market_devig_q_under": 0.5,
            "model_preferred_edge": 0.0294805232517650,
            "model_preferred_ev": 0.0108264534806423,
        }
    )

    return quote


@pytest.mark.parametrize(
    ("label", "keys"),
    [
        ("a primary affiliate is selected", {"bovada": "bovada"}),
        ("a secondary affiliate is selected", {"busr": "bovada"}),
        ("no affiliate is selected", {}),
    ],
)
def test_routing_changes_no_model_or_market_number(label, keys):
    """Cases 14, 15, 16 and 17 at once, across all three tiers.

    One assertion over every published model and market field rather than four
    assertions over one each: the property is that routing touches none of
    them, and naming them individually would let the next field added escape
    the check.
    """
    reg = registry(**keys)

    rows = [priced_row("bovada", -130), priced_row("fanduel", 250)]

    before = [_model_fields(quote) for quote in rows]

    routed = route_rows(rows, reg=reg)

    after = [_model_fields(quote) for quote in rows]

    assert after == before, f"routing altered a model number when {label}"

    assert routed.offer is not None


def test_the_selected_price_is_the_books_price_and_not_the_models():
    """Case 14-17's other half: the price shown is the posted one.

    The fair American odds sit in the same row and are what the model thinks
    the bet is worth. They are nobody's offer, and selecting on them would
    route every wager to whichever book the model happened to disagree with
    most.
    """
    reg = registry(bovada="bovada")

    quote = priced_row("bovada", -130)

    quote["calibrated_fair_over_american"] = 113

    quote["calibrated_fair_under_american"] = -113

    routed = route_rows([quote], reg=reg)

    assert routed.offer.american_odds == -130

    assert routed.offer.american_odds == quote["over_odds"]

    assert routed.offer.american_odds != quote["calibrated_fair_over_american"]


def test_the_offer_universe_is_reported_rather_than_trimmed():
    """What was considered is returned, so a label can be honest about it."""
    reg = registry(bovada="bovada")

    rows = [
        row(vendor="bovada", over=-130),
        row(vendor="fanduel", over=250),
        row(vendor="draftkings", over=180),
    ]

    routed = route_rows(rows, reg=reg)

    assert [offer.vendor for offer in routed.considered] == [
        "bovada",
        "draftkings",
        "fanduel",
    ]

    assert routed.best_overall is False

    assert routing.best_offer(routed.considered).vendor == "fanduel"


def test_best_overall_is_true_only_when_it_actually_is():
    reg = registry(bovada="bovada")

    beaten = route_rows(
        [row(vendor="bovada", over=-130), row(vendor="fanduel", over=250)],
        reg=reg,
    )

    assert beaten.tier == routing.TIER_PRIMARY

    assert beaten.best_overall is False

    tied = route_rows(
        [row(vendor="bovada", over=250), row(vendor="fanduel", over=250)],
        reg=reg,
    )

    assert tied.tier == routing.TIER_PRIMARY

    assert tied.best_overall is True


def test_a_vendor_is_matched_on_case_and_space_only():
    """Enough normalisation to survive the data, not enough to blur a name."""
    assert routing.normalise_vendor("  BOVADA ") == "bovada"

    assert COMMITTED.book_for(" Bovada ").display_name == "Bovada"

    assert COMMITTED.book_for("bov-ada") is None

    assert COMMITTED.book_for("bovada.lv") is None

    assert COMMITTED.book_for("") is None

    assert COMMITTED.book_for(None) is None
