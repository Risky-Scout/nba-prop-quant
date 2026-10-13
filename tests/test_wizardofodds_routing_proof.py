"""The routing-proof slate, and what it has to demonstrate.

The routing rule cannot be proven against the slate the feed canary holds:
every book in it is a non-affiliate, so every card is a fallback and nothing
about the preference, the tie order or the empty case is visible. This slate
exists to make all four outcomes visible at once, and these tests pin the
outcomes so that the page published to the public proof directory is the page
somebody reviewing it would be told to expect.

Each case is asserted twice over: once on the routing result the build
produced, and once on the rendered document, because the proof is a public
page and a reader only sees the second one.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402
import wizardofodds_routing_proof as proof  # noqa: E402

REGISTRY = routing.AffiliateRegistry.load(project_root=PROJECT)

PROOF_SHA = "764d5564ad4e1dcc250b724f458e2e452471c74e"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """The proof slate, published locally and rendered once."""
    root = tmp_path_factory.mktemp("routing-proof")

    transport = proof.publish_locally(root, production_sha=PROOF_SHA)

    return page.build_page(transport, registry=REGISTRY)


@pytest.fixture(scope="module")
def html(built) -> str:
    return built["files"][page.PAGE_NAME].decode("utf-8")


def card_for(html_text: str, player_name: str) -> str:
    """The one card about a player, so a case is read in isolation."""
    cards = re.findall(
        r"<article class=\"card[^\"]*\" data-card.*?</article>", html_text, re.S
    )

    matching = [card for card in cards if player_name in card]

    assert len(matching) == 1, (player_name, len(matching))

    return matching[0]


def routed_for(built_page: dict, case: str) -> routing.Routed:
    wager = proof.wagers()[case]

    matching = [row for row in built_page["rows"] if row.wager == wager]

    assert len(matching) == 1, (case, len(matching))

    return matching[0].routed


def quotes_for(case: str) -> dict[str, object]:
    """Vendor -> the price it posts on the side the model wants."""
    return {vendor: side for vendor, side, _ in proof.QUOTES[case]}


# ======================================================================
# the slate itself
# ======================================================================


def test_the_proof_slate_is_four_wagers_and_every_quote_is_rendered(built):
    """One card per wager, over a slate that is deliberately many quotes."""
    assert len(built["rows"]) == len(proof.CASES) == 4

    assert built["manifest"]["rendered"]["prediction_count"] == 4

    assert built["manifest"]["rendered"]["published_row_count"] == sum(
        len(quotes) for quotes in proof.QUOTES.values()
    )

    assert built["manifest"]["rendered"]["by_tier"] == {
        routing.TIER_PRIMARY: 2,
        routing.TIER_FALLBACK: 1,
        routing.TIER_NONE: 1,
    }


def test_republishing_the_proof_slate_proves_the_same_thing(built, tmp_path):
    """A reviewer rebuilding it gets the same proof, not merely a similar one.

    Not byte-identical, and it should not be claimed to be: a publication
    records when it was served and when it was promoted, and a second one is
    genuinely a second publication. Everything the proof consists of is fixed
    though -- the cards, the routing decisions and the tier tally -- so those
    are compared directly and the provenance strip is excluded by name rather
    than by a looser comparison that would hide a real difference.
    """
    transport = proof.publish_locally(tmp_path / "again", production_sha=PROOF_SHA)

    again = page.build_page(transport, registry=REGISTRY)

    assert again["manifest"]["routed"] == built["manifest"]["routed"]

    assert again["manifest"]["rendered"] == built["manifest"]["rendered"]

    assert again["publication"].rows_sha256 == built["publication"].rows_sha256

    first = built["files"][page.PAGE_NAME].decode("utf-8")

    second = again["files"][page.PAGE_NAME].decode("utf-8")

    def cards(text: str) -> list[str]:
        return re.findall(
            r"<article class=\"card[^\"]*\" data-card.*?</article>", text, re.S
        )

    assert cards(second) == cards(first)

    def without_provenance(text: str) -> str:
        return re.sub(r'<dl class="prov">.*?</dl>', "", text, flags=re.S)

    assert without_provenance(second) == without_provenance(first)

    assert (
        again["publication"].promoted_at != built["publication"].promoted_at
    ), "a second publication is expected to carry its own timestamps"


def test_the_proof_module_runs_no_model_or_pricing_code():
    """Fixture arithmetic only: it builds rows, it does not price them.

    The published column names are themselves full of ``calibrated_``, so the
    tokens here are the ways this module could actually reach the model: an
    import of the package, and the runtime entry points that produce a slate.
    """
    source = (PROJECT / "ops" / "wizardofodds_routing_proof.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "nba_prop_quant.pricing",
        "nba_prop_quant.model",
        "nba_prop_quant.predict",
        "from nba_prop_quant",
        "import nba_prop_quant",
        "price_markets(",
        "run_serving",
        "serving.main",
        "serve_slate",
    ):
        assert forbidden not in source, forbidden


def test_the_proof_page_is_published_beneath_its_own_directory(tmp_path):
    """It goes beside the page canary and cannot be published over it."""
    page_root = tmp_path / "page-destination"

    transport = publisher.LocalDirectoryTransport(page_root)

    transport.write_bytes(page.PAGE_NAME, b"the page canary's own index")

    assert proof.PROOF_PREFIX == "routing-proof"

    exit_code = proof.main(
        [
            "--work-root",
            str(tmp_path / "work"),
            "--page-destination-root",
            str(page_root),
            "--production-sha",
            PROOF_SHA,
        ]
    )

    assert exit_code == publisher.EXIT_OK

    assert (page_root / page.PAGE_NAME).read_bytes() == (
        b"the page canary's own index"
    )

    assert (page_root / "routing-proof" / page.PAGE_NAME).exists()

    assert (page_root / "routing-proof" / page.PAGE_MANIFEST_NAME).exists()


# ======================================================================
# case A: an affiliate is preferred over a better price elsewhere
# ======================================================================


def test_case_a_selects_an_affiliate_although_a_fallback_is_priced_better(
    built, html
):
    """The rule, at its least comfortable: the reader is not sent to +105.

    ``fanduel`` pays more here and is not an affiliate. The routing prefers
    the affiliate anyway, which is the arrangement the disclosure describes,
    and the card is therefore not allowed to call the price the best anywhere.
    """
    routed = routed_for(built, "A")

    prices = quotes_for("A")

    best_elsewhere = max(
        routing.payout_multiple(price)
        for vendor, price in prices.items()
        if REGISTRY.book_for(vendor) is None
    )

    assert routing.payout_multiple(prices["fanduel"]) == best_elsewhere

    assert best_elsewhere > routing.payout_multiple(routed.offer.american_odds)

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == "betonlineag"

    assert routed.offer.american_odds == -108

    assert routed.best_overall is False

    card = card_for(html, "Routing Proof A")

    assert ">BetOnline<" in card

    assert "BET AT BETONLINE" in card

    assert "+105" not in card

    assert "Best Affiliate Price" in card

    assert "also best overall" not in card

    assert "Best Available Price" not in card


def test_case_a_selects_the_best_priced_of_the_four_affiliate_books(built):
    """The preference is for the tier, not for a particular book in it.

    All four affiliates quote this wager at four different prices, so the one
    selected has to be the best of them rather than the first or the
    highest-priority.
    """
    routed = routed_for(built, "A")

    prices = quotes_for("A")

    affiliate = {
        vendor: routing.payout_multiple(price)
        for vendor, price in prices.items()
        if REGISTRY.book_for(vendor) is not None
    }

    assert sorted(affiliate) == sorted(REGISTRY.tie_priorities())

    assert len(set(affiliate.values())) == 4, (
        "this case is meant to separate the four affiliate books on price"
    )

    assert affiliate["betonlineag"] == max(affiliate.values())

    assert routed.offer.vendor == "betonlineag"

    assert REGISTRY.tie_priorities()["betonlineag"] == 1

    assert routed.display_name == "BetOnline"


def test_case_a_considered_every_sportsbook_on_the_wager(built, html):
    """The offer universe is the whole wager, not the affiliate books in it."""
    routed = routed_for(built, "A")

    assert len(routed.considered) == len(proof.QUOTES["A"]) == 6

    assert {offer.vendor for offer in routed.considered} == set(quotes_for("A"))

    card = card_for(html, "Routing Proof A")

    assert "6 sportsbook offers matched this exact wager" in card


def test_case_a_is_one_card_about_one_exact_wager(built, html):
    """Six quotes, one bet: the card states the wager the quotes are on."""
    wager = proof.wagers()["A"]

    matching = [row for row in built["rows"] if row.wager == wager]

    assert len(matching) == 1

    page_row = matching[0]

    assert len(page_row.quotes) == 6

    assert {
        (
            quote["game_id"],
            quote["player_id"],
            quote["prop_type"],
            quote["line_value"],
        )
        for quote in page_row.quotes
    } == {(proof.GAME_ID, 3301, "points", 25.5)}

    card = card_for(html, "Routing Proof A")

    assert "UNDER 25.5" in card

    assert "Points" in card

    assert f'data-game="{proof.GAME_ID}"' in card


# ======================================================================
# case B: nobody paid offers it, so the best price anywhere is shown
# ======================================================================


def test_case_b_selects_the_best_price_across_every_other_sportsbook(built, html):
    """No affiliate quotes this, and the fallback is not a shortlist.

    ``betmgm`` is in the case precisely because it is none of FanDuel,
    DraftKings or Fanatics: the fallback searches whatever the data holds.
    """
    routed = routed_for(built, "B")

    prices = quotes_for("B")

    assert all(REGISTRY.book_for(vendor) is None for vendor in prices)

    assert "betmgm" in prices

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.offer.vendor == "draftkings"

    assert routed.offer.american_odds == 101

    assert routing.payout_multiple(101) == max(
        routing.payout_multiple(price) for price in prices.values()
    )

    assert routed.best_overall is True

    assert routed.destination is None

    card = card_for(html, "Routing Proof B")

    assert "Best Available Price" in card

    assert "this is a price, not a referral" in card

    assert "+101" in card

    assert '<a class="cta"' not in card


# ======================================================================
# case C: nothing to bet
# ======================================================================


def test_case_c_offers_nothing_and_renders_no_button(built, html):
    """A prediction with no quote on the model's side is not a bet.

    The one book in this case prices the over and the model wants the under,
    so there is no offer on the wager at all -- which is different from a
    fallback at a poor price, and has to read differently.
    """
    routed = routed_for(built, "C")

    assert quotes_for("C") == {"fanduel": None}

    assert routed.tier == routing.TIER_NONE

    assert routed.offer is None

    assert routed.considered == ()

    assert routed.destination is None

    card = card_for(html, "Routing Proof C")

    assert "No sportsbook offers this wager" in card

    assert "there is nothing to bet here" in card

    assert '<a class="cta"' not in card

    assert "0 sportsbook offers matched this exact wager" in card


def test_case_c_publishes_its_missing_price_as_absent_rather_than_zero(built):
    """A price that does not exist is null, and shows as an em dash."""
    wager = proof.wagers()["C"]

    page_row = next(row for row in built["rows"] if row.wager == wager)

    assert page_row.expected_value is None

    assert page.american(None) == "—"

    assert page.signed_percent(None) == "—"


# ======================================================================
# case D: the tie order, and the name BetAnything
# ======================================================================


def test_case_d_breaks_an_exact_tie_on_the_operators_order(built, html):
    """Two affiliates at the same best price, so the order decides.

    ``betus`` and ``betanysports`` both post -110. Nothing about the price
    separates them, so the registry's tie priority does, and it must be the
    only thing that did: the loser is the same price, not a worse one.
    """
    routed = routed_for(built, "D")

    prices = quotes_for("D")

    assert prices["betanysports"] == prices["betus"] == -110

    priorities = REGISTRY.tie_priorities()

    assert priorities["betanysports"] < priorities["betus"]

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == "betanysports"

    assert routed.offer.american_odds == -110

    assert routed.best_overall is True

    card = card_for(html, "Routing Proof D")

    assert "-110" in card

    assert "also best overall" in card


def test_case_d_shows_betanysports_under_the_brand_betanything(built, html):
    """The Odds API key is betanysports; the operator's brand is BetAnything."""
    routed = routed_for(built, "D")

    assert routed.offer.vendor == "betanysports"

    assert routed.display_name == "BetAnything"

    assert routed.destination == (
        "/tools/odds-scanner/predictions/api/go.php?book=betanysports"
    )

    card = card_for(html, "Routing Proof D")

    assert ">BetAnything<" in card

    assert "BET AT BETANYTHING" in card

    assert routed.destination in card

    assert "BetAnySports" not in html


# ======================================================================
# what the proof page must not contain
# ======================================================================


def test_the_proof_page_has_no_secondary_tier_and_no_game_time_sort(html):
    """The two things this correction removed, absent from the public proof."""
    assert "secondary" not in html.lower()

    assert "game time" not in html.lower()

    assert 'value="game"' not in html

    assert "data-order=" not in html

    sorts = re.search(r'<select id="f-sort">(.*?)</select>', html, re.S)

    assert re.findall(r'value="([^"]+)"', sorts.group(1)) == ["edge", "ev"]


@pytest.mark.parametrize(
    ("key", "name"),
    [
        ("busr", "BUSR"),
        ("voltage_bet", "Voltage Bet"),
        ("betwhale", "Betwhale"),
        ("xbet", "Xbet"),
    ],
)
def test_the_proof_page_names_no_unsupported_sportsbook(html, key, name):
    assert name not in html

    assert f"go.php?book={key}" not in html

    assert REGISTRY.book_for(key) is None


def test_the_proof_page_links_only_the_four_affiliate_books(html):
    links = set(re.findall(r'<a class="cta" href="([^"]+)"', html))

    assert links == {
        REGISTRY.destination_for("betonlineag"),
        REGISTRY.destination_for("betanysports"),
    }

    for destination in links:
        assert destination.startswith(
            "/tools/odds-scanner/predictions/api/go.php?book="
        )

    assert "https://" not in html

    assert "http://" not in html


def test_the_proof_page_carries_the_provenance_of_its_own_publication(built, html):
    """It is proof data, and the page says which release it is proof of."""
    assert PROOF_SHA in html

    assert built["publication"].rows_sha256 in html

    assert built["manifest"]["mode"] == publisher.MODE_CANARY

    for withheld in page.WITHHELD_PROVENANCE:
        assert withheld not in html

    assert not re.search(r"/tmp/|/var/tmp/|/var/folders/|/var/www/", html)

    assert "nba-prop-deploy" not in html

    assert "NBA_WOO_SSH_PRIVATE_KEY" not in html


def test_the_proof_slate_identifies_itself_as_proof_data(built, html):
    """Nobody should mistake these four rows for a model slate.

    The player names are the only place a reader meets them, so they carry
    it, and the module says so in its own docstring for whoever finds the file.
    """
    for case in proof.CASES.values():
        assert "Routing Proof" in case["player_name"]

        assert str(case["player_name"]) in html

    assert "ROUTING CANARY / TEST DATA" in proof.__doc__

    assert json.loads(
        built["files"][page.PAGE_MANIFEST_NAME]
    )["released"]["slate_date"] == proof.SLATE_DATE
