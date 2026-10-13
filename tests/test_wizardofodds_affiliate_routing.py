"""The frozen tier rule, and the two things it must never do.

The rule is a commercial preference stated openly: among the books that offer
exactly the bet the model picked, one of the four affiliate books beats every
other sportsbook, and only inside the affiliate tier does the best price win.
There is one affiliate tier. When no affiliate offers the wager the fallback is
the whole rest of the offer universe, not a shortlist. These tests pin the
rule, and they pin the two boundaries that make it safe to state it out loud.

The first boundary is the model. Routing is a product decision taken after the
numbers exist, so no probability, edge or expected value may differ because a
book pays commission. That is asserted directly rather than reasoned about.

The second is the price. The operator's priority order settles an exact tie
among affiliates and nothing else, so these tests pin that a priority never
takes a worse number: every selection is first the best payout available in
whichever tier was reached, and the order is consulted only when two payouts
are equal.
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

#: The committed registry, which is what production routes with. Every test
#: that can use it does: all four affiliate books carry a proven identifier
#: now, so there is no need to invent one to exercise the rule.
COMMITTED = routing.AffiliateRegistry.load(project_root=PROJECT)

#: Registry id -> the canonical sportsbook identifier, read from the committed
#: file rather than restated, so a registry disagreeing with the requirement
#: fails in the test that states the requirement rather than everywhere.
PRIMARY_KEYS = {
    book.registry_id: book.vendor_keys[0]
    for book in COMMITTED.books_in_tier(routing.TIER_PRIMARY)
}

PRIMARY_NAMES = {
    book.display_name for book in COMMITTED.books_in_tier(routing.TIER_PRIMARY)
}

#: The four books the operator has said are not part of automated routing.
UNSUPPORTED = ("busr", "voltage_bet", "voltagebet", "betwhale", "xbet")


def book(
    *,
    registry_id: str = "testbook",
    display_name: str = "Test Book",
    tier: str = routing.TIER_PRIMARY,
    vendor_keys: tuple[str, ...] = ("testbook",),
    redirect_book: str | None = "testbook",
    tie_priority: int = routing.UNRANKED_TIE_PRIORITY,
) -> routing.Book:
    """A book the committed registry does not hold.

    Used where a test needs a property no committed book has -- an affiliate
    with no destination, two books claiming one identifier -- rather than
    adding such a book to the registry to make a test pass.
    """
    return routing.Book(
        registry_id=registry_id,
        display_name=display_name,
        tier=tier,
        vendor_keys=vendor_keys,
        redirect_book=redirect_book,
        missing=() if redirect_book else ("the affiliate destination",),
        tie_priority=tie_priority,
    )


def registry_of(*books: routing.Book) -> routing.AffiliateRegistry:
    return routing.AffiliateRegistry(
        books,
        redirect_path=COMMITTED.redirect_path,
        redirect_parameter=COMMITTED.redirect_parameter,
        non_affiliate_names=COMMITTED.non_affiliate_names,
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


def route_rows(
    rows: list[dict], *, reg: routing.AffiliateRegistry | None = None
) -> routing.Routed:
    """Route the first row's wager against the whole list."""
    _, routed = routing.route_row(
        rows[0], rows=rows, registry=reg if reg is not None else COMMITTED
    )

    return routed


# ======================================================================
# the affiliate contract
# ======================================================================


def test_the_committed_registry_is_exactly_the_four_affiliate_books():
    """The contract, stated once, read from the file production routes with."""
    assert PRIMARY_NAMES == {"Bovada", "BetOnline", "BetUS", "BetAnything"}

    assert len(COMMITTED.books) == 4

    assert PRIMARY_KEYS == {
        "betanything": "betanysports",
        "betonline": "betonlineag",
        "betus": "betus",
        "bovada": "bovada",
    }

    for entry in COMMITTED.books:
        assert entry.tier == routing.TIER_PRIMARY

        assert entry.vendor_keys, entry.registry_id

        assert entry.destination_known, entry.registry_id

        assert entry.missing == (), entry.registry_id


def test_there_is_no_secondary_tier():
    """Not merely empty: the tier does not exist and cannot be declared."""
    assert not hasattr(routing, "TIER_SECONDARY")

    assert routing.TIER_ORDER == (routing.TIER_PRIMARY,)

    assert routing.DECLARABLE_TIERS == frozenset({routing.TIER_PRIMARY})

    with pytest.raises(routing.RoutingRefused, match="not one of"):
        registry_of(book(tier="secondary"))

    raw = (PROJECT / routing.REGISTRY_RELATIVE).read_text(encoding="utf-8")

    assert "secondary" not in raw.lower()


@pytest.mark.parametrize("unsupported", UNSUPPORTED)
def test_an_unsupported_book_takes_no_part_in_automated_routing(unsupported):
    """BUSR, Voltage Bet, Betwhale and Xbet are not routed to.

    Not blocked and not reported as missing facts: simply not in the routing
    configuration, so an offer from one is an offer from an ordinary
    sportsbook and can never carry an affiliate link.
    """
    assert COMMITTED.book_for(unsupported) is None

    assert COMMITTED.tier_of(unsupported) == routing.TIER_FALLBACK

    assert COMMITTED.destination_for(unsupported) is None

    assert unsupported not in {
        key for entry in COMMITTED.books for key in entry.vendor_keys
    }

    routed = route_rows([row(vendor=unsupported, over=-105)])

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.destination is None

    assert routed.is_affiliate is False


def test_the_registry_reports_nothing_missing():
    """Every book it routes is complete, so the page has nothing to disclaim."""
    assert COMMITTED.missing_facts() == {}


# ======================================================================
# every primary key, one at a time
# ======================================================================


@pytest.mark.parametrize(
    ("registry_id", "key", "name", "redirect"),
    [
        ("bovada", "bovada", "Bovada", "bovada"),
        ("betonline", "betonlineag", "BetOnline", "betonline"),
        ("betus", "betus", "BetUS", "betus"),
        ("betanything", "betanysports", "BetAnything", "betanysports"),
    ],
)
def test_each_primary_key_is_recognised_named_and_linked(
    registry_id, key, name, redirect
):
    """All four keys individually: recognised, named and sent somewhere real.

    One case per book rather than a loop over the registry, so the identifier,
    the display name and the redirect key are each written down here and a
    silent change to any of them fails.
    """
    entry = COMMITTED.book_for(key)

    assert entry is not None

    assert entry.registry_id == registry_id

    assert entry.display_name == name

    assert entry.tier == routing.TIER_PRIMARY

    assert COMMITTED.display_name_for(key) == name

    assert COMMITTED.destination_for(key) == (
        f"{COMMITTED.redirect_path}?{COMMITTED.redirect_parameter}={redirect}"
    )

    routed = route_rows(
        [row(vendor=key, over=-115), row(vendor="fanduel", over=-120)]
    )

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == key

    assert routed.display_name == name


def test_betanysports_is_betanything():
    """The operator fact this correction rests on.

    BetAnything is the rebranded BetAnySports, so the Odds API's
    ``betanysports`` is this book. The page shows the current name.
    """
    entry = COMMITTED.book_for("betanysports")

    assert entry is not None

    assert entry.display_name == "BetAnything"

    assert entry.tier == routing.TIER_PRIMARY

    assert COMMITTED.display_name_for("betanysports") == "BetAnything"

    assert "betanysports" not in dict(COMMITTED.non_affiliate_names)

    assert COMMITTED.destination_for("betanysports") == (
        "/tools/odds-scanner/predictions/api/go.php?book=betanysports"
    )

    routed = route_rows([row(vendor="BetAnySports", over=-110)])

    assert routed.display_name == "BetAnything"


# ======================================================================
# the tier rule
# ======================================================================


def test_an_affiliate_wins_over_a_better_non_affiliate_price():
    """The rule is a tier rule, so a better number outside it does not win."""
    rows = [
        row(vendor="bovada", over=-130),
        row(vendor="fanduel", over=250),
        row(vendor="draftkings", over=180),
    ]

    routed = route_rows(rows)

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == "bovada"

    assert routed.display_name == "Bovada"

    assert len(routed.considered) == 3

    assert routed.best_overall is False


def test_the_best_price_wins_among_several_primary_offers():
    """Inside the tier, price decides -- as a payout, not as an integer.

    ``+105`` is the best of these four and is numerically between the others
    when read as a signed integer, so a router comparing raw numbers would
    pick wrongly in one direction or the other.
    """
    rows = [
        row(vendor="bovada", over=-120),
        row(vendor="betus", over=105),
        row(vendor="betonlineag", over=-105),
        row(vendor="betanysports", over=-115),
    ]

    routed = route_rows(rows)

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == "betus"

    assert routed.offer.american_odds == 105

    assert routed.display_name == "BetUS"


def test_the_best_primary_price_wins_even_when_it_is_the_lowest_priority_book():
    """Price first. BetUS is last in the tie order and still wins on price."""
    rows = [
        row(vendor="betonlineag", over=-120),
        row(vendor="bovada", over=-118),
        row(vendor="betanysports", over=-115),
        row(vendor="betus", over=-110),
    ]

    routed = route_rows(rows)

    assert routed.offer.vendor == "betus"

    assert routed.offer.american_odds == -110


def test_the_fallback_is_reached_only_when_no_affiliate_offers_it():
    """No affiliate quotes it, so the best of everything else wins.

    A fallback selection is the best price in the whole universe by
    construction, which is why it is the one tier that may be labelled best
    available.
    """
    rows = [
        row(vendor="fanduel", over=-130),
        row(vendor="draftkings", over=-105),
        row(vendor="betmgm", over=-140),
    ]

    routed = route_rows(rows)

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.offer.vendor == "draftkings"

    assert routed.display_name == "DraftKings"

    assert routed.best_overall is True

    assert routed.destination is None


def test_the_fallback_is_the_whole_universe_and_not_a_shortlist():
    """Any sportsbook in the data is eligible, named in the registry or not.

    The winner here is a book this repository records nothing at all about,
    beating the three the registry happens to carry display names for. A
    fallback restricted to familiar books would quietly show a worse price.
    """
    rows = [
        row(vendor="fanduel", over=-130),
        row(vendor="draftkings", over=-125),
        row(vendor="fanatics", over=-120),
        row(vendor="some_new_odds_api_book", over=145),
    ]

    routed = route_rows(rows)

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.offer.vendor == "some_new_odds_api_book"

    assert routed.offer.american_odds == 145

    assert routed.best_overall is True

    assert routed.destination is None

    assert COMMITTED.book_for("some_new_odds_api_book") is None

    assert routed.display_name == "some_new_odds_api_book"


def test_the_best_fallback_price_wins_across_many_books():
    """Twelve books, one best payout, and it is the one selected."""
    prices = {
        "betmgm": -140,
        "betparx": -135,
        "betrivers": -130,
        "betway": -125,
        "caesars": -120,
        "draftkings": -118,
        "espnbet": -115,
        "fanatics": -112,
        "fanduel": -110,
        "fliff": -108,
        "pointsbet": -105,
        "superbook": 120,
    }

    rows = [row(vendor=vendor, over=price) for vendor, price in prices.items()]

    routed = route_rows(rows)

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.offer.vendor == "superbook"

    assert len(routed.considered) == len(prices)


# ======================================================================
# nothing to bet
# ======================================================================


def test_no_matching_offer_fabricates_no_call_to_action():
    """Nobody quotes this side, so there is no selection at all."""
    rows = [row(vendor="bovada", over=None, side="over")]

    routed = route_rows(rows)

    assert routed.tier == routing.TIER_NONE

    assert routed.offer is None

    assert routed.book is None

    assert routed.destination is None

    assert routed.display_name == ""

    assert routed.has_affiliate_destination is False

    assert routed.considered == ()


def test_an_empty_offer_universe_is_answered_rather_than_guessed():
    routed = routing.route((), registry=COMMITTED)

    assert routed.tier == routing.TIER_NONE

    assert routed.offer is None

    assert routed.destination is None


# ======================================================================
# an offer has to be the same bet
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
    """Each of the four contract fields, one at a time.

    Every affiliate book here offers a far better price on something adjacent.
    None can be routed to, because a different player, prop, line or game is a
    different wager rather than the same wager priced better.
    """
    rows = [row(vendor="fanduel", over=-130)]

    rows += [
        row(vendor=key, over=400, **difference) for key in sorted(PRIMARY_KEYS.values())
    ]

    routed = route_rows(rows)

    assert routed.tier == routing.TIER_FALLBACK, (
        f"a book quoting a different {label} was treated as offering this wager"
    )

    assert routed.offer.vendor == "fanduel"

    assert len(routed.considered) == 1


def test_the_wrong_side_cannot_match():
    """The model picked under; an over price is not an offer on it.

    Both books quote this exact contract, so the row survives the contract
    match and is excluded on the side alone: ``bovada`` prices the over
    generously and the under not at all.
    """
    rows = [
        row(vendor="fanduel", over=500, under=-130, side="under"),
        row(vendor="bovada", over=500, under=None, side="under"),
    ]

    routed = route_rows(rows)

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.offer.vendor == "fanduel"

    assert routed.offer.american_odds == -130


def test_the_side_decides_which_published_price_is_the_offer():
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
# ties
# ======================================================================


def test_an_exact_primary_tie_follows_the_operators_order():
    """The priority, in full, and only on an exact tie.

    All four books post the same price. The order is BetOnline, Bovada,
    BetAnything, BetUS, so dropping the winner each time walks down it.
    """
    order = ["betonlineag", "bovada", "betanysports", "betus"]

    assert [
        entry.vendor_keys[0]
        for entry in sorted(
            COMMITTED.books_in_tier(routing.TIER_PRIMARY),
            key=lambda entry: entry.tie_priority,
        )
    ] == order

    remaining = list(order)

    while remaining:
        rows = [row(vendor=vendor, over=-110) for vendor in reversed(remaining)]

        routed = route_rows(rows)

        assert routed.offer.vendor == remaining[0], remaining

        remaining.pop(0)


def test_the_tie_order_never_costs_a_reader_a_better_price():
    """Priority applies after the price, so a worse-priced favourite loses.

    ``betonlineag`` is first in the order and does not win here, because the
    comparison it would have won is only reached when the payouts are equal.
    """
    for better in ("bovada", "betanysports", "betus"):
        routed = route_rows(
            [
                row(vendor="betonlineag", over=-130),
                row(vendor=better, over=-105),
            ]
        )

        assert routed.offer.vendor == better, better

        assert routed.offer.american_odds == -105


def test_an_exact_fallback_tie_breaks_on_the_sportsbook_identifier():
    """No priority is declared outside the affiliate tier, so the key decides.

    ``betmgm`` sorts before ``draftkings`` and before ``fanduel``, and the
    answer does not depend on which order the rows arrived in.
    """
    rows = [
        row(vendor="fanduel", over=-110),
        row(vendor="draftkings", over=-110),
        row(vendor="betmgm", over=-110),
    ]

    assert route_rows(rows).offer.vendor == "betmgm"

    assert route_rows(list(reversed(rows))).offer.vendor == "betmgm"


def test_the_tie_break_is_the_identifier_and_then_the_quote_id():
    """Two quotes from one book still order deterministically.

    The publisher's logical key makes this impossible upstream, which is
    exactly why the router does not get to assume it.
    """
    rows = [
        row(vendor="bovada", over=-110, quote_id=900),
        row(vendor="bovada", over=-110, quote_id=100),
    ]

    assert route_rows(rows).offer.quote_id == 100

    assert route_rows(list(reversed(rows))).offer.quote_id == 100


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
# where a link may come from
# ======================================================================


def test_an_affiliate_destination_comes_only_from_the_registry():
    """The registry is the only source, and it is addressed one way.

    Every destination the committed registry will produce is its redirect path
    with its own parameter and a book it records. Nothing is assembled from a
    book's name, and there is no default.
    """
    produced = {
        entry.display_name: COMMITTED.destination_for(entry.vendor_keys[0])
        for entry in COMMITTED.books
    }

    assert produced == {
        "BetAnything": (
            "/tools/odds-scanner/predictions/api/go.php?book=betanysports"
        ),
        "BetOnline": "/tools/odds-scanner/predictions/api/go.php?book=betonline",
        "BetUS": "/tools/odds-scanner/predictions/api/go.php?book=betus",
        "Bovada": "/tools/odds-scanner/predictions/api/go.php?book=bovada",
    }

    assert COMMITTED.destination_for("not-a-book-anyone-has-heard-of") is None


def test_a_book_with_no_verified_destination_never_produces_a_link():
    """Routable and linkable stay separate, and the router keeps them separate.

    No committed book is in this state, which is the point: the behaviour is
    pinned here so that a book added without a proven destination is selected
    without being linked rather than linked to the redirect's Bovada fallback.
    """
    unlinkable = book(
        registry_id="newbook",
        display_name="New Book",
        vendor_keys=("newbook",),
        redirect_book=None,
    )

    routed = route_rows([row(vendor="newbook", over=-105)], reg=registry_of(unlinkable))

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == "newbook"

    assert routed.destination is None

    assert routed.has_affiliate_destination is False

    assert unlinkable.destination_known is False


def test_a_non_affiliate_sportsbook_never_receives_an_affiliate_url():
    """Every book outside the registry, selected and never linked."""
    for vendor in sorted(COMMITTED.non_affiliate_names):
        routed = route_rows([row(vendor=vendor, over=-105)])

        assert routed.tier == routing.TIER_FALLBACK, vendor

        assert routed.destination is None, vendor

        assert routed.is_affiliate is False, vendor


def test_the_registry_refuses_two_books_claiming_one_identifier():
    with pytest.raises(routing.RoutingRefused, match="claimed by both"):
        registry_of(
            book(registry_id="a", display_name="A", vendor_keys=("same",)),
            book(registry_id="b", display_name="B", vendor_keys=("same",)),
        )


def test_the_registry_refuses_a_tier_it_does_not_recognise():
    with pytest.raises(routing.RoutingRefused, match="not one of"):
        registry_of(book(tier=routing.TIER_FALLBACK))


def test_an_unreadable_registry_is_refused_rather_than_defaulted(tmp_path):
    missing = tmp_path / "nowhere.json"

    with pytest.raises(routing.RoutingRefused, match="none will be invented"):
        routing.AffiliateRegistry.load(registry_path=missing)


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
# routing cannot move a model number
# ======================================================================


MODEL_FIELDS = (
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
    "over_odds",
    "under_odds",
)


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
    ("label", "vendors"),
    [
        ("an affiliate is selected", ("bovada", "fanduel")),
        ("the tie order decides", ("betanysports", "betus")),
        ("no affiliate is selected", ("fanduel", "draftkings")),
        ("nothing is selected", ()),
    ],
)
def test_routing_changes_no_model_market_or_posted_number(label, vendors):
    """Model probability, market probability, edge, EV and the posted odds.

    One assertion over every published number rather than one per field: the
    property is that routing touches none of them, and naming them
    individually would let the next field added escape the check.
    """
    rows = [priced_row(vendor, -130 + index) for index, vendor in enumerate(vendors)]

    rows = rows or [priced_row("fanduel", -130)]

    before = [{name: quote[name] for name in MODEL_FIELDS} for quote in rows]

    routed = route_rows(rows)

    after = [{name: quote[name] for name in MODEL_FIELDS} for quote in rows]

    assert after == before, f"routing altered a published number when {label}"

    assert routed.tier in (
        routing.TIER_PRIMARY,
        routing.TIER_FALLBACK,
        routing.TIER_NONE,
    )


def test_the_selected_price_is_the_books_price_and_not_the_models():
    """The price shown is the posted one.

    The fair American odds sit in the same row and are what the model thinks
    the bet is worth. They are nobody's offer, and selecting on them would
    route every wager to whichever book the model happened to disagree with
    most.
    """
    quote = priced_row("bovada", -130)

    quote["calibrated_fair_over_american"] = 113

    quote["calibrated_fair_under_american"] = -113

    routed = route_rows([quote])

    assert routed.offer.american_odds == -130

    assert routed.offer.american_odds == quote["over_odds"]

    assert routed.offer.american_odds != quote["calibrated_fair_over_american"]


def test_the_offer_universe_is_reported_rather_than_trimmed():
    """What was considered is returned, so a label can be honest about it."""
    rows = [
        row(vendor="bovada", over=-130),
        row(vendor="fanduel", over=250),
        row(vendor="draftkings", over=180),
    ]

    routed = route_rows(rows)

    assert [offer.vendor for offer in routed.considered] == [
        "bovada",
        "draftkings",
        "fanduel",
    ]

    assert routed.best_overall is False

    assert routing.best_offer(routed.considered).vendor == "fanduel"


def test_best_overall_is_true_only_when_it_actually_is():
    beaten = route_rows(
        [row(vendor="bovada", over=-130), row(vendor="fanduel", over=250)]
    )

    assert beaten.tier == routing.TIER_PRIMARY

    assert beaten.best_overall is False

    tied = route_rows(
        [row(vendor="bovada", over=250), row(vendor="fanduel", over=250)]
    )

    assert tied.tier == routing.TIER_PRIMARY

    assert tied.best_overall is True


def test_a_vendor_is_matched_on_case_and_space_only():
    """Enough normalisation to survive the data, not enough to blur a name."""
    assert routing.normalise_vendor("  BOVADA ") == "bovada"

    assert COMMITTED.book_for(" Bovada ").display_name == "Bovada"

    assert COMMITTED.book_for("bov-ada") is None

    assert COMMITTED.book_for("bovada.lv") is None

    assert COMMITTED.book_for("betanysport") is None

    assert COMMITTED.book_for("") is None

    assert COMMITTED.book_for(None) is None
