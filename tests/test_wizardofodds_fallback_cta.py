"""The one defect this change fixes: a best price with nowhere to go.

Routing already selected the right sportsbook when no affiliate offered the
wager. The card showed its name and its price and then stopped, because the
only destination the page knew how to render was an affiliate redirect. These
tests are about the link that was missing, and about the two things that must
not come with it.

The first is attribution. A direct link is not a referral: it goes to the
sportsbook's own page, carries no tracking parameter, does not go through the
affiliate redirect and does not claim ``sponsored``. Several of these tests
exist only to make that unclaimable.

The second, and the one that would actually cost a reader money, is influence
on the selection. Knowing a URL for a sportsbook must never make that
sportsbook win. The decisive test builds a wager where the best-priced book has
no destination and a worse-priced book has one, and requires the best price to
be selected and shown anyway -- with no button.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
import wizardofodds_fallback_destinations as fallback  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402
import wizardofodds_publisher_fixture as fixture  # noqa: E402

REGISTRY = routing.AffiliateRegistry.load(project_root=PROJECT)

COMMITTED = fallback.FallbackDestinations.load(project_root=PROJECT)

#: The three books the requirement names by name.
REQUIRED = ("draftkings", "fanduel", "fanatics")

#: A registry holding nothing, for the comparisons that have to show the
#: destinations changed only the button.
NOTHING = fallback.FallbackDestinations(())


# ======================================================================
# building a slate whose books this test chooses
# ======================================================================


def frame(vendors: tuple[tuple[str, Any], ...]):
    """A priced slate where the named books quote one contract.

    Built from the publisher's own fixture frame, so the rows are the shape the
    publisher accepts rather than hand-written dictionaries. The price given is
    the price of the side the template's model prefers, which is the one the
    routing compares.
    """
    import pandas as pd

    base = fixture.priced_frame()

    template = base.iloc[0].to_dict()

    priced_side = routing.SIDE_PRICE_FIELD[template["model_preferred_side"]]

    rows = []

    for index, (vendor, price) in enumerate(vendors):
        quote = dict(template)

        quote.update({"id": 8100 + index, priced_side: price, "vendor": vendor})

        rows.append(quote)

    return pd.DataFrame(rows)


def publish(root: Path, vendors: tuple[tuple[str, Any], ...]) -> dict:
    built = fixture.build(root, frame=frame(vendors))

    destination = built["destination_root"]

    publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=built["slate_date"],
        data_root=built["data_root"],
        project_root=PROJECT,
        destination_root=destination,
    )

    return {
        "destination": destination,
        "transport": publisher.LocalDirectoryTransport(destination),
    }


def build(
    state: dict, *, destinations: fallback.FallbackDestinations | None = None
) -> dict:
    return page.build_page(
        state["transport"],
        registry=REGISTRY,
        fallback=COMMITTED if destinations is None else destinations,
    )


def one_card(built: dict) -> tuple[Any, str]:
    """The single prediction a one-wager slate renders, and its markup.

    Still called a card because that is what it is -- one wager, one opinion,
    one price, one button -- even though the page now draws it as a table row.
    """
    html = built["files"][page.PAGE_NAME].decode("utf-8")

    rows = re.findall(r"<tr data-row.*?</tr>", html, re.S)

    assert len(rows) == 1, len(rows)

    assert len(built["rows"]) == 1

    return built["rows"][0], rows[0]


def _link(markup: str, *, klass: str) -> tuple[str, str, str]:
    """(href, rel, label) of one call to action, or empty strings."""
    found = re.search(
        rf'<a class="{klass}" href="([^"]+)" target="_blank" '
        r'rel="([^"]+)">([^<]+)</a>',
        markup,
    )

    return (
        (found.group(1), found.group(2), found.group(3))
        if found
        else ("", "", "")
    )


def direct_link(markup: str) -> tuple[str, str, str]:
    return _link(markup, klass="wf-bo-btn direct")


def affiliate_link(markup: str) -> tuple[str, str, str]:
    return _link(markup, klass="wf-bo-btn")


# ======================================================================
# 1-3: the books the requirement names, each rendering a usable button
# ======================================================================


@pytest.mark.parametrize("vendor", REQUIRED)
def test_a_named_fallback_book_selected_alone_renders_a_usable_link(
    tmp_path, vendor
):
    """Cases 1, 2 and 3. The defect, fixed, one book at a time.

    Each is given the only quote on the wager, so it is the selection the
    routing already made before this change -- and the thing that was missing
    was the button, which is asserted present, pointing at the committed
    destination, and carrying the price the reader was quoted.
    """
    state = publish(tmp_path, ((vendor, 101),))

    built = build(state)

    row, card = one_card(built)

    expected = COMMITTED.url_for(vendor)

    assert expected, f"{vendor} has no committed destination"

    assert row.routed.tier == routing.TIER_FALLBACK

    assert row.routed.offer.vendor == vendor

    assert row.direct_url == expected

    href, rel, label = direct_link(card)

    assert href == expected

    assert rel == "nofollow noopener"

    assert label == (
        f"{page.DIRECT_CTA_VERB} {REGISTRY.display_name_for(vendor).upper()} +101"
    )

    assert built["manifest"]["rendered"]["direct_cta_count"] == 1

    assert built["manifest"]["rendered"]["affiliate_cta_count"] == 0

    assert built["manifest"]["routed"][0]["direct_destination"] == expected


def test_the_case_b_defect_is_fixed_for_draftkings_at_the_best_price(tmp_path):
    """Case 1 as the routing-proof canary actually exposed it.

    DraftKings at ``+101`` beating three other non-affiliate books is exactly
    what the published proof page showed with no button on it.
    """
    state = publish(
        tmp_path,
        (
            ("fanduel", -120),
            ("draftkings", 101),
            ("fanatics", -105),
            ("betmgm", -110),
        ),
    )

    row, card = one_card(build(state))

    assert row.routed.offer.vendor == "draftkings"

    assert row.routed.offer.american_odds == 101

    assert row.routed.best_overall is True

    assert "Best Available Price" in card

    assert direct_link(card)[0] == COMMITTED.url_for("draftkings")

    assert "VIEW AT DRAFTKINGS" in card

    assert "+101" in card


# ======================================================================
# 4: a book nobody has a destination for
# ======================================================================


@pytest.mark.parametrize(
    "vendor", ["fixturebook", "some_new_odds_api_book", "superbook", "fliff"]
)
def test_a_fallback_book_with_no_destination_gets_no_invented_link(
    tmp_path, vendor
):
    """Case 4. Fail closed: a price with no button, not a guessed URL.

    Two of these are books the registry knows the display name of and has no
    destination for, which is the case most likely to tempt a guess from the
    name.
    """
    state = publish(tmp_path, ((vendor, 101),))

    built = build(state)

    row, card = one_card(built)

    assert COMMITTED.url_for(vendor) is None

    assert row.routed.tier == routing.TIER_FALLBACK

    assert row.routed.offer.vendor == vendor

    assert row.direct_url is None

    assert direct_link(card) == ("", "", "")

    assert "<a " not in card

    assert page.UNLINKED_TAG in card

    assert built["manifest"]["rendered"]["direct_cta_count"] == 0

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert "no established link for" in html

    assert "nothing there is a referral" in html

    # No address anywhere in the document, rather than no occurrence of the
    # scheme: the page also names the origins it broadcasts its height to when
    # the site embeds it, and those are not somewhere a reader can be sent.
    assert "href" not in html, "no destination was invented for this book"

    assert "sportsbook.example" not in html


# ======================================================================
# 5-7: the paid link and the unpaid link are different things
# ======================================================================


def test_an_affiliate_selection_still_renders_exactly_what_it_did(tmp_path):
    """Case 5. The affiliate path is untouched by this change.

    Asserted against the committed registry's own destination and against the
    rel the affiliate link has always carried, and asserted to be unaffected by
    whether any fallback destination is known at all.
    """
    state = publish(tmp_path, (("bovada", -120), ("fanduel", 140)))

    built = build(state)

    row, card = one_card(built)

    assert row.routed.tier == routing.TIER_PRIMARY

    assert row.routed.offer.vendor == "bovada"

    href, rel, label = affiliate_link(card)

    assert href == REGISTRY.destination_for("bovada")

    assert href == "/tools/odds-scanner/predictions/api/go.php?book=bovada"

    assert rel == "sponsored nofollow noopener"

    assert label == f"{page.AFFILIATE_CTA_VERB} BOVADA -120"

    assert row.direct_url is None

    assert direct_link(card) == ("", "", "")

    assert built["manifest"]["rendered"]["affiliate_cta_count"] == 1

    assert built["manifest"]["rendered"]["direct_cta_count"] == 0

    without = build(state, destinations=NOTHING)

    assert without["files"][page.PAGE_NAME] == built["files"][page.PAGE_NAME]


def test_an_affiliate_book_is_never_given_a_direct_link(tmp_path):
    """Case 6, the half that matters: a paid click keeps its attribution.

    A direct link for a book that pays commission would be an unattributed
    referral -- the site's convention is that an affiliate click always goes
    through the redirect. Forced here by handing the page a destination
    registry that wrongly claims a primary affiliate.
    """
    wrong = fallback.FallbackDestinations(
        (
            fallback.Destination(
                vendor_key="bovada",
                display_name="Bovada",
                url="https://www.bovada.lv/sports/basketball/nba",
                verification_source="a test asserting this is ignored",
                verification_type="test",
            ),
        )
    )

    state = publish(tmp_path, (("bovada", -120),))

    built = build(state, destinations=wrong)

    row, card = one_card(built)

    assert row.routed.tier == routing.TIER_PRIMARY

    assert row.direct_url is None

    assert "bovada.lv" not in card

    assert affiliate_link(card)[0] == REGISTRY.destination_for("bovada")


def test_a_direct_link_never_uses_the_affiliate_redirect(tmp_path):
    """Case 6. go.php is the paid path and a direct link may not take it."""
    state = publish(tmp_path, (("draftkings", 101),))

    built = build(state)

    row, card = one_card(built)

    assert "go.php" not in row.direct_url

    assert row.direct_url.startswith("https://sportsbook.draftkings.com")

    links = re.findall(r'<a class="wf-bo-btn[^"]*" href="([^"]+)"', card)

    assert links == [row.direct_url]

    assert "go.php" not in card

    for destination in COMMITTED.linkable_books().values():
        assert "go.php" not in destination

        assert REGISTRY.book_for(
            next(
                key
                for key in COMMITTED.vendor_keys
                if COMMITTED.url_for(key) == destination
            )
        ) is None


def test_a_direct_link_makes_no_affiliate_claim(tmp_path):
    """Case 7. No sponsored rel, no affiliate badge, no affiliate styling.

    ``rel="sponsored"`` is a statement to a search engine that a link was paid
    for. It was not, so saying so would be false, and the card says the
    opposite in words a reader can see.
    """
    state = publish(tmp_path, (("fanduel", 101),))

    built = build(state)

    row, card = one_card(built)

    href, rel, _ = direct_link(card)

    assert href

    assert "sponsored" not in rel

    assert "noopener" in rel

    assert "nofollow" in rel

    assert "sponsored" not in card

    assert page.AFFILIATE_CTA_VERB not in card

    assert "Paid link" not in card

    assert row.routed.is_affiliate is False

    assert row.routed.has_affiliate_destination is False

    assert "Best Available Price" in card

    assert "Affiliate Price" not in card

    assert "Not a paid placement" in card

    assert "WizardOfOdds earns nothing if you use it" in card


# ======================================================================
# 8: the decisive one -- a URL may not buy a selection
# ======================================================================


def test_a_known_destination_never_wins_a_selection_from_a_better_price(
    tmp_path,
):
    """Case 8. The best price is shown even when it cannot be linked.

    ``superbook`` has no destination and the best number; ``betmgm`` has a
    destination and a worse one. Substituting BetMGM would hand a reader a
    worse price in exchange for a clickable button, which is the single worst
    thing this change could have done.
    """
    state = publish(tmp_path, (("betmgm", -110), ("superbook", 150)))

    built = build(state)

    row, card = one_card(built)

    assert COMMITTED.url_for("betmgm")

    assert COMMITTED.url_for("superbook") is None

    assert row.routed.offer.vendor == "superbook"

    assert row.routed.offer.american_odds == 150

    assert row.direct_url is None

    assert direct_link(card) == ("", "", "")

    assert "SuperBook" in card

    assert "+150" in card

    assert "betmgm" not in card.lower()


def test_the_selection_is_identical_with_and_without_any_destinations(
    tmp_path,
):
    """Case 8, over every shape of wager rather than one example.

    The routing result is compared field by field between a build that knows
    every committed destination and a build that knows none. If a destination
    could reach the selection, one of these would differ.
    """
    cases = (
        (("betmgm", -110), ("superbook", 150)),
        (("draftkings", 101), ("fanduel", -120)),
        (("bovada", -120), ("draftkings", 140)),
        (("fixturebook", -105), ("fanatics", -115)),
        (("caesars", -101), ("betparx", -102), ("betrivers", -103)),
    )

    for vendors in cases:
        state = publish(tmp_path / f"case{hash(vendors) & 0xFFFF}", vendors)

        with_all = build(state)

        with_none = build(state, destinations=NOTHING)

        selected = [
            (entry["tier"], entry["selected_vendor"], entry["selected_odds"])
            for entry in with_all["manifest"]["routed"]
        ]

        assert selected == [
            (entry["tier"], entry["selected_vendor"], entry["selected_odds"])
            for entry in with_none["manifest"]["routed"]
        ], vendors

        assert [entry["best_overall"] for entry in with_all["manifest"]["routed"]] == [
            entry["best_overall"] for entry in with_none["manifest"]["routed"]
        ], vendors

        assert [entry["destination"] for entry in with_all["manifest"]["routed"]] == [
            entry["destination"] for entry in with_none["manifest"]["routed"]
        ], vendors


def test_the_page_asks_about_a_destination_only_after_routing_has_decided():
    """Case 8, structurally: the resolver cannot see the offer universe.

    It is handed the routing result, which has already chosen, so there is no
    argument it could consult to prefer a linkable book. Read off the source
    rather than inferred, because this is the property the behaviour rests on.
    """
    import inspect

    signature = inspect.signature(page._direct_url)

    assert list(signature.parameters) == ["routed", "fallback"]

    source = inspect.getsource(page._direct_url)

    assert "routed.offer.vendor" in source

    assert "considered" not in source

    assert "payout" not in source

    routing_source = (
        PROJECT / "ops" / "wizardofodds_affiliate_routing.py"
    ).read_text(encoding="utf-8")

    assert "fallback_destinations" not in routing_source, (
        "the routing module must not know destinations exist"
    )

    assert "wizardofodds_fallback_destinations" not in routing_source


# ======================================================================
# 9-13: the numbers are the published numbers, still
# ======================================================================


@pytest.mark.parametrize(
    "vendors",
    [
        (("draftkings", 101), ("fanduel", -120)),
        (("bovada", -120), ("draftkings", 140)),
        (("superbook", 150), ("betmgm", -110)),
    ],
)
def test_no_published_number_moves_when_a_link_is_added(tmp_path, vendors):
    """Cases 9 to 13. Odds, model probability, market probability, edge, EV.

    Compared between a build with every destination and a build with none, on
    the ``PageRow`` accessors the renderer itself reads, and cross-checked
    against the published row so this is identity with the publication rather
    than merely self-consistency.
    """
    state = publish(tmp_path, vendors)

    with_all = build(state)

    with_none = build(state, destinations=NOTHING)

    by_quote = {
        row["quote_id"]: row for row in with_all["publication"].rows
    }

    for linked, unlinked in zip(with_all["rows"], with_none["rows"], strict=True):
        assert linked.wager == unlinked.wager

        assert linked.routed.offer.american_odds == unlinked.routed.offer.american_odds

        source = by_quote[linked.row["quote_id"]]

        side = linked.side

        for value, published in (
            (linked.model_probability, source[page.SIDE_MODEL_PROBABILITY[side]]),
            (linked.market_probability, source[page.SIDE_MARKET_PROBABILITY[side]]),
            (linked.edge, source[page.SIDE_EDGE[side]]),
            (linked.expected_value, source[page.SIDE_EXPECTED_VALUE[side]]),
        ):
            assert value == published

        assert linked.model_probability == unlinked.model_probability

        assert linked.market_probability == unlinked.market_probability

        assert linked.edge == unlinked.edge

        assert linked.expected_value == unlinked.expected_value

        assert linked.routed.offer.american_odds == source[
            routing.SIDE_PRICE_FIELD[side]
        ]


def test_the_rendered_page_differs_only_by_the_link(tmp_path):
    """The whole of the change, as a diff of two documents.

    All the fallback registry can do to a row is replace the book's name and
    the words saying there is nowhere to go with a button and the sentence
    under it, and take away the note that explains the former. Asserted by
    cutting all of those out of both documents and requiring the remainder to
    be identical, so anything else the registry had reached would show up here.
    """
    state = publish(tmp_path, (("draftkings", 101),))

    linked = build(state)["files"][page.PAGE_NAME].decode("utf-8")

    unlinked = build(state, destinations=NOTHING)["files"][page.PAGE_NAME].decode(
        "utf-8"
    )

    assert linked != unlinked

    assert '<a class="wf-bo-btn direct"' in linked

    assert page.UNLINKED_TAG in unlinked

    assert page.UNLINKED_NOTE in unlinked

    def without_the_call_to_action(html: str) -> str:
        html = re.sub(
            r'<a class="wf-bo-btn direct".*?</a>'
            r'<span class="wf-bo-note">.*?</span>',
            "",
            html,
            flags=re.S,
        )

        html = re.sub(
            r'<span class="wf-bo-book">.*?</span>'
            rf'<span class="wf-tag">{page.UNLINKED_TAG}</span>',
            "",
            html,
            flags=re.S,
        )

        return html.replace(
            f'<p class="wf-disc">{page.UNLINKED_NOTE}</p>', ""
        )

    assert without_the_call_to_action(linked) == without_the_call_to_action(
        unlinked
    )


# ======================================================================
# 14-16: nothing else moved
# ======================================================================


def test_the_betanything_mapping_is_unchanged(tmp_path):
    """Case 14."""
    state = publish(tmp_path, (("betanysports", -110), ("draftkings", 140)))

    built = build(state)

    row, card = one_card(built)

    assert row.routed.tier == routing.TIER_PRIMARY

    assert row.routed.offer.vendor == "betanysports"

    assert row.routed.display_name == "BetAnything"

    assert REGISTRY.destination_for("betanysports") == (
        "/tools/odds-scanner/predictions/api/go.php?book=betanysports"
    )

    assert affiliate_link(card)[0] == REGISTRY.destination_for("betanysports")

    assert f"{page.AFFILIATE_CTA_VERB} BETANYTHING" in card

    assert row.direct_url is None

    assert COMMITTED.url_for("betanysports") is None

    assert REGISTRY.tie_priorities() == {
        "betonlineag": 1,
        "bovada": 2,
        "betanysports": 3,
        "betus": 4,
    }


def test_no_secondary_affiliate_tier_is_reintroduced(tmp_path):
    """Case 15."""
    state = publish(tmp_path, (("draftkings", 101),))

    built = build(state)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert not hasattr(routing, "TIER_SECONDARY")

    assert routing.TIER_ORDER == (routing.TIER_PRIMARY,)

    assert routing.DECLARABLE_TIERS == frozenset({routing.TIER_PRIMARY})

    assert set(built["manifest"]["rendered"]["by_tier"]) == {
        routing.TIER_PRIMARY,
        routing.TIER_FALLBACK,
        routing.TIER_NONE,
    }

    assert "secondary" not in html.lower()

    for name in ("wizardofodds_player_props_page", "wizardofodds_fallback_destinations"):
        source = (PROJECT / "ops" / f"{name}.py").read_text(encoding="utf-8")

        assert "secondary" not in source.lower(), name


def test_the_game_time_sort_is_still_absent(tmp_path):
    """Case 16."""
    state = publish(tmp_path, (("draftkings", 101),))

    html = build(state)["files"][page.PAGE_NAME].decode("utf-8")

    assert "game time" not in html.lower()

    assert 'value="game"' not in html

    assert "data-order=" not in html

    sorts = re.search(r'<select id="f-sort">(.*?)</select>', html, re.S)

    assert re.findall(r'value="([^"]+)"', sorts.group(1)) == [
        "ev",
        "model",
        "player",
    ]


# ======================================================================
# the destination registry itself
# ======================================================================


def test_the_registry_holds_the_books_the_requirement_names():
    for vendor in REQUIRED:
        destination = COMMITTED.destination_for(vendor)

        assert destination is not None, vendor

        assert destination.url.startswith("https://"), vendor

        assert destination.verification_source, vendor

        assert destination.verification_type, vendor

        assert destination.display_name == REGISTRY.display_name_for(vendor)


def test_the_two_registries_name_no_book_in_common():
    """A book is paid or it is not, and it cannot be recorded as both."""
    affiliate_keys = set(REGISTRY.tie_priorities())

    assert affiliate_keys == {"bovada", "betonlineag", "betus", "betanysports"}

    assert affiliate_keys.isdisjoint(COMMITTED.vendor_keys)

    for key in COMMITTED.vendor_keys:
        assert REGISTRY.book_for(key) is None, key

        assert REGISTRY.destination_for(key) is None, key

    for key in affiliate_keys:
        assert COMMITTED.url_for(key) is None, key


def test_the_registry_holds_no_tracking_parameter_or_credential():
    """These are not referrals, so nothing in the file may attribute one."""
    raw = (PROJECT / fallback.REGISTRY_RELATIVE).read_text(encoding="utf-8")

    document = json.loads(raw)

    for entry in document["books"].values():
        url = entry["url"]

        for forbidden in fallback.FORBIDDEN_IN_URL:
            assert forbidden not in url, (url, forbidden)

        assert "?" not in url or url.endswith("?page=sportsbook"), url

    for forbidden in (
        "btag=",
        "?aff=",
        "ref=WOO",
        "BEGIN OPENSSH PRIVATE KEY",
        "NBA_WOO_SSH_PRIVATE_KEY",
        "nba-prop-deploy",
    ):
        assert forbidden not in raw, forbidden

    assert document["link_policy"]["may_claim_affiliate"] is False

    assert document["link_policy"]["may_use_affiliate_redirect"] is False


def test_fallback_direct_url_answers_one_identifier_without_a_registry():
    """The shape the requirement asked for, by name."""
    assert fallback.fallback_direct_url(
        "draftkings", project_root=PROJECT
    ) == COMMITTED.url_for("draftkings")

    assert fallback.fallback_direct_url("DraftKings", project_root=PROJECT) == (
        COMMITTED.url_for("draftkings")
    )

    assert (
        fallback.fallback_direct_url("not_a_sportsbook", project_root=PROJECT)
        is None
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://sportsbook.draftkings.com/", "not an https destination"),
        ("https://x/go.php?book=draftkings", "not a referral"),
        ("https://x/nba?btag=a_WOO", "not a referral"),
        ("https://x/nba?utm_source=woo", "not a referral"),
        ("https://x/nba?subid=123", "not a referral"),
    ],
)
def test_a_destination_that_is_a_referral_or_insecure_is_refused(url, expected):
    with pytest.raises(fallback.DestinationRefused, match=expected):
        fallback.FallbackDestinations(
            (
                fallback.Destination(
                    vendor_key="draftkings",
                    display_name="DraftKings",
                    url=url,
                    verification_source="site_configuration",
                    verification_type="site_configuration",
                ),
            )
        )


def test_a_destination_with_no_stated_verification_is_refused():
    with pytest.raises(fallback.DestinationRefused, match="no stated verification"):
        fallback.FallbackDestinations(
            (
                fallback.Destination(
                    vendor_key="draftkings",
                    display_name="DraftKings",
                    url="https://sportsbook.draftkings.com/",
                    verification_source="",
                    verification_type="",
                ),
            )
        )


def test_one_sportsbook_recorded_twice_is_refused():
    entry = fallback.Destination(
        vendor_key="draftkings",
        display_name="DraftKings",
        url="https://sportsbook.draftkings.com/",
        verification_source="site_configuration",
        verification_type="site_configuration",
    )

    with pytest.raises(fallback.DestinationRefused, match="recorded twice"):
        fallback.FallbackDestinations((entry, entry))


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"rel": "sponsored nofollow noopener"}, "declares rel"),
        ({"rel_must_not_contain": ""}, "no longer forbids"),
        ({"may_use_affiliate_redirect": True}, "may_use_affiliate_redirect"),
        ({"may_claim_affiliate": True}, "may_claim_affiliate"),
        ({"may_append_tracking_parameters": True}, "may_append_tracking"),
    ],
)
def test_a_link_policy_that_permits_a_referral_is_refused(
    tmp_path, change, expected
):
    """The file's own declaration that nothing in it is paid for, enforced.

    That declaration is what the page's claims rest on, so a file that had
    quietly relaxed it would change what the page says about who pays whom.
    """
    document = json.loads(
        (PROJECT / fallback.REGISTRY_RELATIVE).read_text(encoding="utf-8")
    )

    document["link_policy"].update(change)

    path = tmp_path / "destinations.json"

    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(fallback.DestinationRefused, match=expected):
        fallback.FallbackDestinations.load(registry_path=path)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"{ not json", "could not be read"),
        (b'{"contract_version": 99}', "declares contract"),
    ],
)
def test_an_unreadable_destination_file_refuses_rather_than_guesses(
    tmp_path, payload, expected
):
    path = tmp_path / "destinations.json"

    path.write_bytes(payload)

    with pytest.raises(fallback.DestinationRefused, match=expected):
        fallback.FallbackDestinations.load(registry_path=path)


def test_a_missing_destination_file_refuses_rather_than_guesses(tmp_path):
    with pytest.raises(fallback.DestinationRefused, match="could not be read"):
        fallback.FallbackDestinations.load(
            registry_path=tmp_path / "nothing.json"
        )


def test_the_frozen_identifiers_are_untouched():
    """This is a link-rendering fix, so the model must be where it was."""
    import hashlib

    model_spec = PROJECT / "research" / "final_model" / "final_model_spec.json"

    policy = (
        PROJECT / "research" / "final_model" / "live_shadow_promotion_policy.json"
    )

    assert (
        hashlib.sha256(model_spec.read_bytes()).hexdigest()
        == "154109658127920ebeac381bd21e8bb29390505384124419899236cce37fdc64"
    )

    assert (
        hashlib.sha256(policy.read_bytes()).hexdigest()
        == "a0784aff602d9ee14ba1ef91b30fe972fb7eeb02ec80e0b8afbb7d9ff764aee8"
    )

    assert (
        json.loads(model_spec.read_text(encoding="utf-8"))["factor_spec_hash"]
        == "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
    )
