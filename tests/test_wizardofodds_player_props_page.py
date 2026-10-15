"""The player-props page: what it renders, and what it refuses to.

A page is a public claim about what the model says, so these tests are mostly
about the gap between the publication and the page being zero. Every number a
card shows is asserted equal to the published row it came from, the selected
sportsbook and price are asserted equal to what the router independently
returns, and the provenance strip is asserted to carry the real production SHA.

The rest are refusals. A destination mid-publication, one recording a
non-incumbent authority, one whose candidate is no longer shadow-only, one
whose rows were produced by a freeze the release does not approve: each is a
page that would be a false claim, and each is refused rather than rendered
partially.

Two of the requirements are about what must not leak. The published manifest
carries an absolute path on whichever machine served the slate, so the page
renders provenance from an allow-list and these tests pin that the path and the
credential environment variable's value are both absent from the output.
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
import wizardofodds_publisher_fixture as fixture  # noqa: E402

REAL_SHA = "64f011cf8cfd0b14b39ee1132e363d17974aa409"

REGISTRY = routing.AffiliateRegistry.load(project_root=PROJECT)

#: The page destination the committed configuration describes. Resolved here so
#: the confinement tests speak about the destination production would use.
PAGE_DESTINATION = "wizardofodds_nba_player_props_page_canary"

#: The legacy NBA pages and feeds the website already publishes, which sit
#: three directories above the page canary.
LEGACY_NBA_FILES: tuple[str, ...] = (
    "nba-live-props.html",
    "nba-pmf-research.html",
    "nba-props.html",
    "nba_props_today.json",
    "pmf_research.json",
)


# ======================================================================
# building a destination to render from
# ======================================================================


def affiliate_frame(vendors: tuple[tuple[str, Any], ...]):
    """A priced slate where several books quote one contract.

    Built from the publisher's own fixture frame so the rows are the shape the
    publisher accepts and screens, rather than hand-written dictionaries that
    could drift from the published contract. One row per book, all on the same
    player, prop and line, which is what makes them offers on one wager.

    The price given is the price of the side the template's model prefers, so
    it is the one the routing compares. The other side is left as the fixture
    had it, because no card reads it.
    """
    import pandas as pd

    base = fixture.priced_frame()

    template = base.iloc[0].to_dict()

    priced_side = routing.SIDE_PRICE_FIELD[template["model_preferred_side"]]

    rows = []

    for index, (vendor, price) in enumerate(vendors):
        quote = dict(template)

        quote.update({"id": 8000 + index, priced_side: price, "vendor": vendor})

        rows.append(quote)

    return pd.DataFrame(rows)


def publish(root: Path, *, frame=None, sha: str = REAL_SHA) -> dict:
    """One canary publication into ``root``/destination."""
    built = fixture.build(
        root, frame=frame, receipt_changes={"production_code_sha": sha}
    )

    destination = built["destination_root"]

    receipt = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=built["slate_date"],
        data_root=built["data_root"],
        project_root=PROJECT,
        destination_root=destination,
    )

    return {
        "built": built,
        "destination": destination,
        "receipt": receipt,
        "transport": publisher.LocalDirectoryTransport(destination),
    }


@pytest.fixture
def published(tmp_path: Path) -> dict:
    """The fixture slate, published, with no affiliate book in it."""
    return publish(tmp_path)


@pytest.fixture
def with_affiliates(tmp_path: Path) -> dict:
    """A slate where one affiliate and two other books quote the same wager.

    ``fanduel`` has the best price of the three and is not an affiliate, so the
    page has to show the affiliate while a better number exists -- which is the
    labelling requirement's whole point. ``busr`` is here as a book the
    registry records as unsupported: it must be treated as any other
    non-affiliate rather than given a tier of its own.
    """
    return publish(
        tmp_path,
        frame=affiliate_frame(
            (("bovada", -120), ("busr", -115), ("fanduel", 140))
        ),
    )


def build(state: dict) -> dict:
    return page.build_page(state["transport"], registry=REGISTRY)


def html_of(state: dict) -> str:
    return build(state)["files"][page.PAGE_NAME].decode("utf-8")


def rewrite(state: dict, relative: str, payload: bytes) -> None:
    (state["destination"] / relative).write_bytes(payload)


def manifest_path(state: dict) -> str:
    current = json.loads(
        (state["destination"] / publisher.CURRENT_POINTER).read_text()
    )

    return current["manifest_relative"]


def load_manifest(state: dict) -> dict:
    return json.loads((state["destination"] / manifest_path(state)).read_text())


def cards_in(html: str) -> list[str]:
    """One card per wager, where a card is now drawn as a table row.

    A card is one wager, one opinion, one price and one button; the page draws
    it densely instead of as a tile, which changes how it is cut out of the
    document and nothing about how many there are.
    """
    return re.findall(r"<tr data-row.*?</tr>", html, re.S)


def badges_in(html: str) -> list[str]:
    """The tier label rendered on each card.

    Read out of the badge rather than searched for in the whole document,
    because the affiliate disclosure quotes the labels in order to explain
    them and a substring search would find those sentences too.
    """
    return re.findall(r'<span class="wf-tag claim">([^<]*)</span>', html)


#: The fixture slate is one quote per wager, which is what makes a card count
#: and a row count comparable in the tests that use it.
FIXTURE_WAGERS = len(fixture.QUOTES)


# ======================================================================
# 1-2: it builds, and it builds the same way twice
# ======================================================================


def test_the_page_builds_deterministically(published):
    """Case 1. No clock in the output, so the same release is the same bytes.

    Three builds rather than two: a single comparison would pass against a
    value that changes every other call, and the page manifest records the
    page's own digest, so a drifting page would also have to drift there
    consistently.
    """
    builds = [build(published) for _ in range(3)]

    first = builds[0]["files"]

    for other in builds[1:]:
        assert other["files"] == first

    digests = {
        build_result["manifest"]["page_sha256"] for build_result in builds
    }

    assert len(digests) == 1

    assert digests == {publisher.digest(first[page.PAGE_NAME])}


def test_a_valid_current_publication_renders_every_published_row(published):
    """Case 2. Every published wager gets a card, nothing added or dropped."""
    built = build(published)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    source = json.loads(
        (published["destination"] / built["publication"].rows_relative).read_text()
    )["rows"]

    assert len(source) == FIXTURE_WAGERS

    wagers = {
        (
            row["game_id"],
            row["player_id"],
            row["prop_type"],
            row["line_value"],
            row["model_preferred_side"],
        )
        for row in source
    }

    assert len(wagers) == FIXTURE_WAGERS, (
        "this fixture is meant to be one quote per wager, so a card count and "
        "a row count mean the same thing here"
    )

    assert len(built["rows"]) == FIXTURE_WAGERS

    assert len(cards_in(html)) == FIXTURE_WAGERS

    assert built["manifest"]["rendered"]["prediction_count"] == FIXTURE_WAGERS

    assert built["manifest"]["rendered"]["published_row_count"] == len(source)

    for row in source:
        assert str(row["player_name"]) in html

    assert sorted(built["files"]) == [page.PAGE_NAME, page.PAGE_MANIFEST_NAME]


def test_the_best_expected_value_is_rendered_first(published):
    """Default order, over the published expected value and nothing derived.

    Expected value rather than edge because expected value is the one number
    on the page that answers the reader's question -- what this wager is worth
    at the price actually shown next to it -- and because edge is no longer
    rendered at all.
    """
    built = build(published)

    values = [float(row.expected_value) for row in built["rows"]]

    assert values == sorted(values, reverse=True)


# ======================================================================
# 3: nothing to show
# ======================================================================


def zero_row_destination(tmp_path: Path) -> Path:
    """A destination serving a valid release that happens to carry no rows.

    Written directly rather than published, because the publisher refuses to
    publish an empty slate -- which is correct, and is why the renderer's
    empty state cannot be reached through it. The documents here are
    internally consistent: the pointer, the manifest and the rows all agree,
    and the rows digest is the digest of the rows actually written.
    """
    destination = tmp_path / "empty"

    transport = publisher.LocalDirectoryTransport(destination)

    release_id = "nba_prop_quant__2026-10-20__0000000000000000"

    rows_relative = f"releases/{release_id}/rows.json"

    manifest_relative = f"releases/{release_id}/manifest.json"

    # The publisher's own constant rather than a literal, so an empty release
    # stays readable by the renderer when the contract moves.
    rows_bytes = publisher.serialise(
        {
            "contract_version": publisher.PUBLICATION_CONTRACT_VERSION,
            "rows": [],
            "slate_date": "2026-10-20",
        }
    )

    transport.write_bytes(rows_relative, rows_bytes)

    manifest = {
        "approved_freeze_id": fixture.FREEZE_ID,
        "candidate_authority": page.REQUIRED_CANDIDATE_AUTHORITY,
        "contract_version": publisher.PUBLICATION_CONTRACT_VERSION,
        "mode": publisher.MODE_CANARY,
        "model_authority": publisher.MODEL_AUTHORITY,
        "provenance": {
            "incumbent_authority": "frozen_deployment_bundle",
            "incumbent_version": fixture.FREEZE_ID,
            "model_authority": publisher.MODEL_AUTHORITY,
            "production_code_sha": REAL_SHA,
            "served_at": "2026-10-20T00:00:00+00:00",
            "slate_date": "2026-10-20",
        },
        "release_id": release_id,
        "row_count": 0,
        "rows_relative": rows_relative,
        "rows_sha256": publisher.digest(rows_bytes),
        "slate_date": "2026-10-20",
    }

    transport.write_bytes(manifest_relative, publisher.serialise(manifest))

    transport.write_bytes(
        publisher.CURRENT_POINTER,
        publisher.serialise(
            {
                "manifest_relative": manifest_relative,
                "mode": publisher.MODE_CANARY,
                "promoted_at": "2026-10-20T00:00:01+00:00",
                "release_id": release_id,
                "rows_relative": rows_relative,
                "rows_sha256": manifest["rows_sha256"],
            }
        ),
    )

    return destination


def test_a_release_with_no_rows_renders_a_page_that_says_so(tmp_path):
    """Case 3. An empty slate is a page, not a crash and not a blank.

    It also must not look like a page whose rows were filtered away, so the
    empty state says that nothing was withheld.
    """
    destination = zero_row_destination(tmp_path)

    built = page.build_page(
        publisher.LocalDirectoryTransport(destination), registry=REGISTRY
    )

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert built["rows"] == ()

    assert cards_in(html) == []

    assert "there is nothing to bet on this slate" in html

    assert "Nothing was withheld and nothing was invented" in html

    assert built["manifest"]["rendered"]["prediction_count"] == 0

    assert built["manifest"]["rendered"]["affiliate_cta_count"] == 0

    assert REAL_SHA in html


# ======================================================================
# 4-6: refusals
# ======================================================================


def test_a_destination_with_no_current_pointer_refuses(tmp_path):
    with pytest.raises(page.PageRefused, match="current release"):
        page.build_page(
            publisher.LocalDirectoryTransport(tmp_path / "nothing"),
            registry=REGISTRY,
        )


@pytest.mark.parametrize(
    ("relative", "payload", "expected"),
    [
        (publisher.CURRENT_POINTER, b"{ not json", "not readable JSON"),
        (publisher.CURRENT_POINTER, b"[]", "not a JSON object"),
        (publisher.CURRENT_POINTER, b"{}", "names no release"),
    ],
)
def test_a_malformed_pointer_refuses_the_build(
    published, relative, payload, expected
):
    """Case 4. A pointer that is not a pointer is not interpreted."""
    rewrite(published, relative, payload)

    with pytest.raises(page.PageRefused, match=expected):
        build(published)


def test_a_malformed_manifest_refuses_the_build(published):
    """Case 4. The release manifest is read, not guessed at."""
    rewrite(published, manifest_path(published), b"{ not json")

    with pytest.raises(page.PageRefused, match="not readable JSON"):
        build(published)


def test_a_missing_manifest_refuses_the_build(published):
    (published["destination"] / manifest_path(published)).unlink()

    with pytest.raises(page.PageRefused, match="no releases/"):
        build(published)


def test_rows_that_do_not_match_the_manifest_digest_refuse_the_build(published):
    """A destination mid-publication is refused rather than rendered."""
    manifest = load_manifest(published)

    document = json.loads(
        (published["destination"] / manifest["rows_relative"]).read_text()
    )

    document["rows"] = document["rows"][:3]

    rewrite(
        published, manifest["rows_relative"], publisher.serialise(document)
    )

    with pytest.raises(page.PageRefused, match="not serving the rows it says"):
        build(published)


def test_a_rows_contract_this_renderer_does_not_know_refuses_the_build(published):
    manifest = load_manifest(published)

    document = json.loads(
        (published["destination"] / manifest["rows_relative"]).read_text()
    )

    document["contract_version"] = 99

    payload = publisher.serialise(document)

    rewrite(published, manifest["rows_relative"], payload)

    manifest["rows_sha256"] = publisher.digest(payload)

    rewrite(published, manifest_path(published), publisher.serialise(manifest))

    current = json.loads(
        (published["destination"] / publisher.CURRENT_POINTER).read_text()
    )

    current["rows_sha256"] = manifest["rows_sha256"]

    rewrite(
        published, publisher.CURRENT_POINTER, publisher.serialise(current)
    )

    with pytest.raises(page.PageRefused, match="rows contract"):
        build(published)


def test_a_pointer_and_manifest_that_disagree_refuse_the_build(published):
    manifest = load_manifest(published)

    manifest["release_id"] = "nba_prop_quant__2026-10-20__ffffffffffffffff"

    rewrite(published, manifest_path(published), publisher.serialise(manifest))

    with pytest.raises(page.PageRefused, match="disagree on the release id"):
        build(published)


def test_a_non_incumbent_authority_refuses_the_build(published):
    """Case 5. A page is a claim about the incumbent, so only it is rendered."""
    manifest = load_manifest(published)

    manifest["model_authority"] = "candidate"

    rewrite(published, manifest_path(published), publisher.serialise(manifest))

    with pytest.raises(page.PageRefused, match="not an incumbent publication"):
        build(published)


def test_a_candidate_that_is_no_longer_shadow_only_refuses_the_build(published):
    """Case 6. The shadow exists to be measured, and the page checks it still is."""
    manifest = load_manifest(published)

    manifest["candidate_authority"] = "published"

    rewrite(published, manifest_path(published), publisher.serialise(manifest))

    with pytest.raises(page.PageRefused, match="shadow may no longer be shadow-only"):
        build(published)


def test_a_row_from_a_freeze_the_release_does_not_approve_refuses_the_build(
    published,
):
    """Case 6 again, at the row level.

    The release approves one freeze. A row produced by another is not the
    incumbent's own output, whatever the manifest's authority field says, and
    is the shape candidate numbers would arrive in.
    """
    manifest = load_manifest(published)

    document = json.loads(
        (published["destination"] / manifest["rows_relative"]).read_text()
    )

    document["rows"][2]["freeze_id"] = "game_latent_state_candidate_v1"

    payload = publisher.serialise(document)

    rewrite(published, manifest["rows_relative"], payload)

    manifest["rows_sha256"] = publisher.digest(payload)

    rewrite(published, manifest_path(published), publisher.serialise(manifest))

    current = json.loads(
        (published["destination"] / publisher.CURRENT_POINTER).read_text()
    )

    current["rows_sha256"] = manifest["rows_sha256"]

    rewrite(published, publisher.CURRENT_POINTER, publisher.serialise(current))

    with pytest.raises(page.PageRefused, match="not the incumbent's own output"):
        build(published)


def test_a_row_missing_a_rendered_field_refuses_the_build(published):
    manifest = load_manifest(published)

    document = json.loads(
        (published["destination"] / manifest["rows_relative"]).read_text()
    )

    del document["rows"][0]["model_preferred_ev"]

    payload = publisher.serialise(document)

    rewrite(published, manifest["rows_relative"], payload)

    manifest["rows_sha256"] = publisher.digest(payload)

    rewrite(published, manifest_path(published), publisher.serialise(manifest))

    current = json.loads(
        (published["destination"] / publisher.CURRENT_POINTER).read_text()
    )

    current["rows_sha256"] = manifest["rows_sha256"]

    rewrite(published, publisher.CURRENT_POINTER, publisher.serialise(current))

    with pytest.raises(page.PageRefused, match="model_preferred_ev"):
        build(published)


# ======================================================================
# 7: provenance
# ======================================================================


def test_the_production_code_sha_is_rendered_and_is_the_published_one(published):
    """Case 7. The page's SHA is the manifest's, not the publisher's HEAD."""
    built = build(published)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    published_sha = load_manifest(published)["provenance"]["production_code_sha"]

    assert published_sha == REAL_SHA

    assert f"<dd>{REAL_SHA}</dd>" in html

    assert built["manifest"]["provenance"]["production_code_sha"] == REAL_SHA

    assert "unknown" not in html.split("<footer")[1]


def test_the_provenance_strip_carries_the_current_state_fields(published):
    html = html_of(published)

    for label in (
        "Slate",
        "Incumbent version",
        "Production code SHA",
        "Slate served",
        "Release",
        "Rows digest",
        "Published",
        "Mode",
    ):
        assert f"<dt>{label}</dt>" in html, label

    publication = build(published)["publication"]

    assert publication.release_id in html

    assert publication.rows_sha256 in html

    assert publication.promoted_at in html


# ======================================================================
# 8-10: the offer on the card is the routing result
# ======================================================================


def test_the_selected_sportsbook_and_price_match_the_routing_result(
    with_affiliates,
):
    """Case 8. The card and the router are checked against each other.

    The router is re-run here over the same rows rather than read off the
    page's own manifest, so the two have to agree rather than agreeing with
    themselves.
    """
    built = build(with_affiliates)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    publication = built["publication"]

    for row in built["rows"]:
        _, expected = routing.route_row(
            row.row, rows=publication.rows, registry=REGISTRY
        )

        assert row.routed.tier == expected.tier

        assert row.routed.offer == expected.offer

        assert row.routed.destination == expected.destination

        source = next(
            candidate
            for candidate in publication.rows
            if candidate["quote_id"] == expected.offer.quote_id
        )

        posted = source[routing.SIDE_PRICE_FIELD[row.side]]

        assert expected.offer.american_odds == posted

        assert page.american(posted) in html

    recorded = built["manifest"]["routed"]

    assert [entry["tier"] for entry in recorded] == [
        row.routed.tier for row in built["rows"]
    ]

    assert [entry["selected_odds"] for entry in recorded] == [
        row.routed.offer.american_odds for row in built["rows"]
    ]


def test_a_primary_affiliate_is_selected_over_a_better_non_affiliate_price(
    with_affiliates,
):
    built = build(with_affiliates)

    assert built["manifest"]["rendered"]["by_tier"] == {
        routing.TIER_PRIMARY: 1,
        routing.TIER_FALLBACK: 0,
        routing.TIER_NONE: 0,
    }

    routed = built["rows"][0].routed

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == "bovada"

    assert routed.best_overall is False


def test_the_affiliate_call_to_action_uses_the_verified_destination(
    with_affiliates,
):
    """Case 9. One link, and it is the registry's."""
    built = build(with_affiliates)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    expected = REGISTRY.destination_for("bovada")

    assert expected == "/tools/odds-scanner/predictions/api/go.php?book=bovada"

    links = re.findall(r'<a class="wf-bo-btn" href="([^"]+)"[^>]*>', html)

    assert links == [expected]

    assert 'rel="sponsored nofollow noopener"' in html

    assert 'target="_blank"' in html

    assert "BET AT BOVADA" in html

    assert page.american(-120) in html

    assert built["manifest"]["rendered"]["affiliate_cta_count"] == 1


def test_an_affiliate_selection_is_never_called_best_available(with_affiliates):
    """Case 10's other half: the label may not overstate the price.

    ``fanduel`` has the better number here, so the honest claim is that this
    is the best affiliate price and nothing more.
    """
    html = html_of(with_affiliates)

    assert badges_in(html) == ["Best Affiliate Price"]

    assert "also best overall" not in html

    assert "best odds overall" not in html.lower()


def test_an_affiliate_selection_that_ties_the_best_price_may_say_so(tmp_path):
    state = publish(
        tmp_path,
        frame=affiliate_frame((("bovada", 140), ("fanduel", 140))),
    )

    built = build(state)

    assert built["rows"][0].routed.tier == routing.TIER_PRIMARY

    assert built["rows"][0].routed.best_overall is True

    assert "also best overall" in built["files"][page.PAGE_NAME].decode("utf-8")


def test_a_fallback_offer_is_not_presented_as_an_affiliate_offer(published):
    """Case 10. No affiliate book quoted these, so no card claims one.

    The fixture's books are not affiliates, so every card is a fallback. There
    must be no link, no affiliate styling and no affiliate tier badge
    anywhere, and the card has to say plainly that a price is not a referral.
    """
    built = build(published)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert (
        built["manifest"]["rendered"]["by_tier"][routing.TIER_FALLBACK]
        == FIXTURE_WAGERS
    )

    assert built["manifest"]["rendered"]["affiliate_cta_count"] == 0

    assert re.findall(r'<a class="wf-bo-btn"', html) == []

    assert "sponsored nofollow noopener" not in html

    assert 'class="card aff"' not in html

    assert 'class="tier primary"' not in html

    assert 'class="tier secondary"' not in html

    assert "nothing there is a referral" in html

    assert set(badges_in(html)) == {"Best Available Price"}

    for row in built["rows"]:
        assert row.routed.tier == routing.TIER_FALLBACK

        assert row.routed.destination is None

        assert row.routed.best_overall is True


def test_a_wager_no_book_offers_renders_no_button_at_all(tmp_path):
    state = publish(
        tmp_path, frame=affiliate_frame((("bovada", -120),))
    )

    manifest = load_manifest(state)

    document = json.loads(
        (state["destination"] / manifest["rows_relative"]).read_text()
    )

    for row in document["rows"]:
        row["model_preferred_side"] = "under"

        row["under_odds"] = None

    payload = publisher.serialise(document)

    rewrite(state, manifest["rows_relative"], payload)

    manifest["rows_sha256"] = publisher.digest(payload)

    rewrite(state, manifest_path(state), publisher.serialise(manifest))

    current = json.loads(
        (state["destination"] / publisher.CURRENT_POINTER).read_text()
    )

    current["rows_sha256"] = manifest["rows_sha256"]

    rewrite(state, publisher.CURRENT_POINTER, publisher.serialise(current))

    built = build(state)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert built["manifest"]["rendered"]["by_tier"][routing.TIER_NONE] >= 1

    assert "No sportsbook in the published data offers this wager" in html

    assert "No current offer" in html

    assert re.findall(r'<a class="wf-bo-btn"', html) == []


def test_the_page_has_no_secondary_affiliate_tier(with_affiliates):
    """There is one affiliate tier, so there is no second one to render.

    Asserted on the rendered document, the build's tier tally and the page's
    own labels together, because a tier that still existed anywhere would have
    to show up in one of the three.
    """
    built = build(with_affiliates)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert set(built["manifest"]["rendered"]["by_tier"]) == {
        routing.TIER_PRIMARY,
        routing.TIER_FALLBACK,
        routing.TIER_NONE,
    }

    assert set(page.TIER_LABEL) == {
        routing.TIER_PRIMARY,
        routing.TIER_FALLBACK,
        routing.TIER_NONE,
    }

    assert "secondary" not in html.lower()

    source = (PROJECT / "ops" / "wizardofodds_player_props_page.py").read_text(
        encoding="utf-8"
    )

    assert "secondary" not in source.lower()


@pytest.mark.parametrize(
    ("vendor", "name"),
    [
        ("busr", "BUSR"),
        ("voltage_bet", "Voltage Bet"),
        ("betwhale", "Betwhale"),
        ("xbet", "Xbet"),
    ],
)
def test_a_book_the_integration_does_not_support_is_never_an_affiliate(
    tmp_path, vendor, name
):
    """The four unsupported books take no part in affiliate selection.

    Each is given the only quote on the wager, so if it were routable it would
    be selected and linked. It is selected -- a price is a price -- but as a
    fallback, with no referral and no affiliate styling.
    """
    state = publish(tmp_path, frame=affiliate_frame(((vendor, -115),)))

    built = build(state)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    routed = built["rows"][0].routed

    assert routed.tier == routing.TIER_FALLBACK

    assert routed.destination is None

    assert REGISTRY.book_for(vendor) is None

    assert f"go.php?book={vendor}" not in html

    assert f"BET AT {name.upper()}" not in html

    assert re.findall(r'<a class="wf-bo-btn"', html) == []


def test_betanysports_is_rendered_and_linked_as_betanything(tmp_path):
    """BetAnything is the rebranded BetAnySports, so its quotes are its own.

    The Odds API spells the book ``betanysports``; the operator's brand is
    BetAnything. A reader sees the brand, and the referral goes to the key the
    site's redirect declares for that sportsbook.
    """
    state = publish(
        tmp_path,
        frame=affiliate_frame((("betanysports", -120), ("fanduel", 140))),
    )

    built = build(state)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    routed = built["rows"][0].routed

    assert routed.tier == routing.TIER_PRIMARY

    assert routed.offer.vendor == "betanysports"

    assert routed.display_name == "BetAnything"

    assert "BET AT BETANYTHING" in html

    assert ">BetAnything<" in html

    assert "BetAnySports" not in html

    assert routed.destination == (
        "/tools/odds-scanner/predictions/api/go.php?book=betanysports"
    )

    assert routed.destination in html


# ======================================================================
# 11: the numbers are the published numbers
# ======================================================================


def test_every_rendered_model_value_equals_its_published_value(with_affiliates):
    """Case 11. Identity, field by field, row by row.

    Asserted on the ``PageRow`` accessors the renderer itself reads, so this
    is the same path the HTML is built from rather than a parallel reading of
    the source.
    """
    built = build(with_affiliates)

    by_quote = {row["quote_id"]: row for row in built["publication"].rows}

    for page_row in built["rows"]:
        source = by_quote[page_row.row["quote_id"]]

        side = page_row.side

        assert side == source["model_preferred_side"]

        assert page_row.model_probability == source[
            page.SIDE_MODEL_PROBABILITY[side]
        ]

        assert page_row.market_probability == source[
            page.SIDE_MARKET_PROBABILITY[side]
        ]

        assert page_row.edge == source["model_preferred_edge"]

        assert page_row.expected_value == source["model_preferred_ev"]

        assert page_row.wager.game_id == source["game_id"]

        assert page_row.wager.player_id == source["player_id"]

        assert page_row.wager.prop_type == source["prop_type"]

        assert page_row.wager.line_value == source["line_value"]


def test_the_rendered_percentages_round_the_published_values(published):
    """Formatting is a display choice; the value underneath is unchanged."""
    built = build(published)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    for page_row in built["rows"]:
        assert page.percent(page_row.model_probability) in html

        assert page.signed_percent(page_row.expected_value) in html

    assert page.percent(0.529480523251765) == "52.9%"

    assert page.signed_percent(0.0294805232517650) == "+2.9%"

    assert page.signed_percent(-0.1017355443897331) == "-10.2%"

    assert page.american(-110) == "-110"

    assert page.american(140) == "+140"


def test_the_page_computes_no_pricing_of_its_own():
    """The renderer imports no pricing or model module, and arithmetic shows.

    The only arithmetic on a published value in this module is multiplying a
    probability by a hundred to print it and rounding odds to an integer. A
    second pricing system would have to reach for something; nothing here
    does.
    """
    source = (PROJECT / "ops" / "wizardofodds_player_props_page.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "nba_prop_quant.pricing",
        "nba_prop_quant.model",
        "nba_prop_quant.predict",
        "from nba_prop_quant",
        "import numpy",
        "import pandas",
        "scipy",
        "sklearn",
    ):
        assert forbidden not in source, f"the page reaches for {forbidden}"

    routing_source = (
        PROJECT / "ops" / "wizardofodds_affiliate_routing.py"
    ).read_text(encoding="utf-8")

    for forbidden in ("from nba_prop_quant", "import numpy", "import pandas"):
        assert forbidden not in routing_source


# ======================================================================
# 12-14: what must not reach a reader
# ======================================================================


def test_untrusted_text_is_escaped(tmp_path):
    """Case 12. A player name is data from an API, not markup.

    The script tag is the one that matters: a name that could close the
    embedded JSON element would turn published data into executable markup.
    """
    import pandas as pd

    frame = fixture.priced_frame()

    hostile = '</script><img src=x onerror="alert(1)">&"\''

    frame = frame.copy()

    frame.loc[frame.index[0], "player_name"] = hostile

    frame.loc[frame.index[0], "vendor"] = "<b>book</b>"

    state = publish(tmp_path, frame=frame)

    html = html_of(state)

    assert hostile not in html

    assert '<img src=x onerror="alert(1)">' not in html

    assert "<img" not in html

    assert 'onerror="' not in html

    assert "&lt;/script&gt;" in html

    assert "&lt;b&gt;book&lt;/b&gt;" in html

    assert "<b>book</b>" not in html

    assert html.count("</script>") == 3


def test_the_embedded_json_cannot_close_its_own_element(published):
    html = html_of(published)

    embedded = re.search(
        r'<script type="application/json" id="source-note">(.*?)</script>',
        html,
        re.S,
    )

    assert embedded is not None

    payload = json.loads(
        embedded.group(1)
        .replace("\\u003c", "<")
        .replace("\\u003e", ">")
        .replace("\\u0026", "&")
    )

    assert payload["release_id"] == build(published)["publication"].release_id

    assert "<" not in embedded.group(1)


def test_no_credential_material_appears_in_the_output(published):
    """Case 13. Nothing the environment holds as a secret reaches the page."""
    files = build(published)["files"]

    blob = b"\n".join(files[name] for name in sorted(files)).decode("utf-8")

    for forbidden in (
        "BEGIN OPENSSH PRIVATE KEY",
        "BEGIN RSA PRIVATE KEY",
        "BEGIN PRIVATE KEY",
        "ssh-rsa ",
        "ssh-ed25519 ",
        "NBA_WOO_SSH_PRIVATE_KEY",
        "BDL_API_KEY",
        "ODDS_API_KEY",
        "nba-prop-deploy",
        "/var/www/",
    ):
        assert forbidden not in blob, f"{forbidden} reached the page"


def test_no_temporary_filesystem_path_appears_in_the_output(published):
    """Case 14. The manifest carries one and the page renders from an allow-list.

    The value is asserted present in the source manifest first. A test that
    only checked the page would also pass against a manifest that had stopped
    carrying the path, and would stop testing anything.
    """
    manifest = load_manifest(published)

    for withheld in page.WITHHELD_PROVENANCE:
        assert manifest["provenance"].get(withheld), (
            f"the source manifest no longer carries {withheld}, so this test "
            "is no longer checking that the page withholds it"
        )

    leaked = str(manifest["provenance"]["priced_market_artifact_path"])

    assert leaked.startswith("/")

    files = build(published)["files"]

    blob = b"\n".join(files[name] for name in sorted(files)).decode("utf-8")

    assert leaked not in blob

    for withheld in page.WITHHELD_PROVENANCE:
        assert withheld not in blob

    assert not re.search(r"/tmp/|/var/tmp/|/var/folders/", blob)

    assert withheld not in [name for name, _ in page.RENDERED_PROVENANCE]


def test_the_provenance_allow_list_holds_no_path_field():
    """The shape of the leak, excluded by construction rather than by review."""
    for name, _ in page.RENDERED_PROVENANCE:
        assert not name.endswith("_path"), name

    assert set(page.WITHHELD_PROVENANCE).isdisjoint(
        {name for name, _ in page.RENDERED_PROVENANCE}
    )


# ======================================================================
# 15: it works on a phone
# ======================================================================


def test_the_page_is_built_for_a_phone_first(published):
    """Case 15. The viewport, the single-column breakpoint, the tappable CTA.

    A smoke test of the things that actually break a betting page on a phone:
    no viewport meta and everything is 980px wide and zoomed out; inputs under
    16px and iOS zooms the page when one is focused; a short CTA and it is hard
    to hit.
    """
    built = build(published)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in html

    assert "@media (max-width:720px)" in html

    narrow = html.split("@media (max-width:720px)")[1]

    assert ".wf-table thead{display:none}" in narrow

    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in narrow

    assert "min-height:44px" in narrow

    assert "overflow-x:visible" in narrow

    assert "font-size:16px" in html

    assert "-webkit-text-size-adjust:100%" in html

    assert "<link" not in html

    assert "<img" not in html

    assert "http://" not in html


def test_the_page_needs_no_network_and_no_dependency(published):
    """Static and self-contained, which is what the host serves.

    Self-contained is about what the document *fetches*, not about where its
    links point: an outbound sportsbook link is a destination a reader chooses,
    while a stylesheet or an image is a request the page makes on their behalf
    before they have chosen anything. So no ``src``, no ``link`` and no
    ``@import``, and the only ``href`` permitted is a call to action.
    """
    html = html_of(published)

    assert "<style>" in html

    assert re.findall(r'src="([^"]+)"', html) == []

    assert "<link" not in html

    assert "@import" not in html

    hrefs = set(re.findall(r'href="([^"]+)"', html))

    cta_hrefs = set(re.findall(r'<a class="wf-bo-btn[^"]*" href="([^"]+)"', html))

    assert hrefs == cta_hrefs

    assert hrefs == set(), "this fixture's books are not linkable"


def test_the_controls_the_page_requires_are_all_present(published):
    """Player search, two filters, and the three sorts the feed supports.

    The game filter is gone rather than relabelled: its only possible labels
    were internal game identifiers, and a filter whose options are integers a
    reader has never seen is not a filter. ``Featured book`` is named for what
    it actually does -- it filters on the one sportsbook each prediction
    shows, not on every sportsbook quoting the wager.
    """
    html = html_of(published)

    for element in (
        'id="f-search"',
        'id="f-prop"',
        'id="f-book"',
        'id="f-sort"',
    ):
        assert element in html, element

    assert 'id="f-game"' not in html

    assert "Featured book" in html

    for option in (
        'value="ev"',
        'value="win"',
        'value="player"',
    ):
        assert option in html, option

    assert "Expected value (highest first)" in html

    assert "Win probability (highest first)" in html


def test_there_is_no_game_time_sort_and_the_page_no_longer_blames_the_feed(
    published,
):
    """The sort stays absent; the reason given for it is now the true one.

    It was absent because the publication carried no tip-off timestamp. The
    publication carries one now, and the page shows it, so a page still
    saying tip-off times are not published would be stating something its own
    table disproves two columns to the left. The control is still gone and the
    script still has no branch for it -- that part was a product decision, not
    a data limitation, and only the explanation had to change.
    """
    built = build(published)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert "Game time" not in html

    assert "game time" not in html.lower()

    assert 'value="game"' not in html

    sorts = re.search(r'<select id="f-sort">(.*?)</select>', html, re.S)

    assert sorts is not None

    assert re.findall(r'value="([^"]+)"', sorts.group(1)) == [
        "ev",
        "win",
        "player",
    ]

    assert "data-order=" not in html

    assert "mode === 'game'" not in html

    assert "Tip-off times are not in the published release" not in html

    assert "Team names are not published either" not in html

    assert "there is no ordering by start time" in html

    source = (PROJECT / "ops" / "wizardofodds_player_props_page.py").read_text(
        encoding="utf-8"
    )

    assert "Game time" not in source

    assert '("game",' not in source


def test_every_card_carries_what_the_filters_and_sorts_read(published):
    """Filtering is a DOM operation over attributes already in the HTML.

    Which is what makes "filters do not hide the CTA" structural: the script
    toggles whole rows, and a row's call to action is inside it.
    """
    built = build(published)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    cards = cards_in(html)

    assert len(cards) == FIXTURE_WAGERS

    for card in cards:
        for attribute in (
            "data-offer=",
            "data-prop=",
            "data-book=",
            "data-ev=",
            "data-win=",
            "data-sortname=",
            "data-stable=",
            "data-tier=",
            "data-search=",
        ):
            assert attribute in card, attribute

        assert 'class="c-book"' in card

    assert "row.style.display = ok ? '' : 'none'" in html


def test_the_affiliate_disclosure_is_rendered(published):
    html = html_of(published)

    assert "Affiliate disclosure:" in html

    assert "may earn a commission" in html

    assert "not necessarily the best price available anywhere" in html

    assert "nothing here is a guarantee of profit" in html

    for claim in (
        "risk-free",
        "guaranteed win",
        "can't lose",
        "sure thing",
        "act now",
        "hurry",
    ):
        assert claim not in html.lower().replace(
            "no offer here is risk-free unless", ""
        ), claim


def test_the_disclosure_names_the_four_affiliate_books_and_no_others(published):
    """The reader is told which four books are paid, by name.

    The registry's own primary books rather than a written-out list, so the
    disclosure and the routing cannot drift apart: a fifth paid book would
    have to be disclosed or this fails.
    """
    html = html_of(published)

    primary = [
        book.display_name
        for book in REGISTRY.books_in_tier(routing.TIER_PRIMARY)
    ]

    assert sorted(primary) == ["BetAnything", "BetOnline", "BetUS", "Bovada"]

    for name in primary:
        assert name in page.AFFILIATE_DISCLOSURE, name

        assert name in html, name

    assert "those four books" in page.AFFILIATE_DISCLOSURE

    assert "are paid affiliate links" in html


def test_no_unsupported_book_is_named_or_linked_anywhere_on_the_page(published):
    """Books the integration does not support are simply not in the product.

    They were previously listed as books with facts missing, which read as a
    promise to add them. They are not blockers and they are not pending: they
    take no part in routing, so the page neither names them nor links them.
    """
    html = html_of(published)

    for key, name in (
        ("busr", "BUSR"),
        ("voltage_bet", "Voltage Bet"),
        ("betwhale", "Betwhale"),
        ("xbet", "Xbet"),
    ):
        assert name not in html, name

        assert f"go.php?book={key}" not in html, key

        assert REGISTRY.book_for(key) is None, key

    assert REGISTRY.missing_facts() == {}

    assert "never shown and never linked" not in html


# ======================================================================
# 16-17: the page canary cannot reach anything else
# ======================================================================


def page_destination() -> publisher.Destination:
    return publisher.resolve_destination(
        publisher.MODE_CANARY,
        publisher.read_destinations(project_root=PROJECT),
        name=PAGE_DESTINATION,
    )


def test_the_page_canary_writes_only_beneath_its_own_subpath():
    """Case 16. The legacy files are unreachable because of where this can write.

    Three directories separate the page canary from the legacy NBA pages, and
    the published root is the canary directory, so no relative path this
    destination accepts can name one of them.
    """
    destination = page_destination()

    remote = destination.remote

    assert remote.canary_subpath == "canary"

    assert remote.published_root == (
        "/var/www/sportsodds/tools/odds-scanner/predictions/NBA/player-props"
        "/canary"
    )

    assert remote.published_base_url.endswith("/NBA/player-props/canary/")

    legacy_root = Path(
        "/var/www/sportsodds/tools/odds-scanner/predictions"
    )

    for name in LEGACY_NBA_FILES:
        legacy = legacy_root / name

        assert Path(remote.published_root) not in legacy.parents

        assert not str(legacy).startswith(remote.published_root + "/")


def test_the_page_canary_cannot_name_the_live_index():
    """The final live location is one directory up and is not expressible.

    Which is the whole of the non-live guarantee: it is not that the entry
    point declines to write the live index, it is that no path handed to this
    destination resolves there.
    """
    remote = page_destination().remote

    live = f"{remote.public_target}/{page.PAGE_NAME}"

    assert not live.startswith(remote.published_root)

    assert remote.published_root == f"{remote.public_target}/canary"

    assert page.PAGE_NAME not in remote.published_root


@pytest.mark.parametrize(
    "relative",
    [
        "../index.html",
        "../../nba-props.html",
        "../../../nba_props_today.json",
        "/var/www/sportsodds/tools/odds-scanner/predictions/nba-props.html",
        "canary/../../index.html",
    ],
)
def test_a_path_that_climbs_out_of_the_page_canary_is_refused(relative):
    """The remote transport's confinement, which is where the legacy files live.

    Asserted against the predicate the transport addresses every read and
    write through, rather than by constructing a transport, because the check
    is the whole of the protection and it needs no host to exercise.
    """
    assert publisher._is_safe_remote_segment_path(relative) is False

    assert publisher._is_safe_remote_segment_path(page.PAGE_NAME) is True

    assert publisher._is_safe_remote_segment_path(page.PAGE_MANIFEST_NAME) is True


def test_the_page_entry_points_live_mode_is_refused_by_the_feed_switch():
    """Case 17. There is a live path, and the switch is what closes it.

    This used to assert the entry point contained no live mode at all, which
    was true and was the safest thing available while activation was a
    question nobody had answered. It stopped being the right guarantee once
    unattended operation was the goal: an activation that required editing
    this file would be an activation reviewed as a code change, and the whole
    point of the gate is that it is an approval instead.

    So the live mode exists and is addressed, and what is asserted is that
    taking it is refused -- by the feed publisher's own committed switch, read
    through the same function the feed reads it through, so there is one
    decision rather than two that could drift apart.
    """
    with pytest.raises(publisher.PublishingRefused, match="live publishing is"):
        page.assert_mode_is_permitted(publisher.MODE_LIVE, project_root=PROJECT)

    # And the canary is not waved through: it is permitted because the
    # committed switch enables it, which is a fact that can change.
    assert page.assert_mode_is_permitted(
        publisher.MODE_CANARY, project_root=PROJECT
    )["state"] == publisher.STATE_ENABLED

    # The entry point itself takes the mode and nothing else decides it, so
    # there is no second place a live run could be authorised from.
    assert page._parser().parse_args([]).mode == publisher.MODE_CANARY

    assert sorted(page.MODE_DESTINATIONS) == [
        publisher.MODE_CANARY,
        publisher.MODE_LIVE,
    ]


def test_the_live_page_is_addressed_without_being_reachable():
    """Case 17. Fully configured, fully disabled: the two are different.

    What a later activation must not have to invent, stated as the facts it
    would otherwise have to supply: which feed the live page is built from,
    which destination it is written to, and where in that destination it
    lands. All three are committed; none of them is permission.
    """
    live = page.MODE_DESTINATIONS[publisher.MODE_LIVE]

    assert live["source"] == "wizardofodds_nba_live"

    assert live["page"] == "wizardofodds_nba_player_props_page_live"

    # The served root, so index.html is where a reader asks for it.
    assert live["prefix"] == ""

    document = publisher.read_destinations(project_root=PROJECT)

    destination = publisher.resolve_destination(
        publisher.MODE_LIVE, document, name=live["page"]
    )

    remote = destination.remote

    assert remote is not None

    assert remote.published_root == remote.public_target

    assert remote.published_root.endswith("/NBA/player-props")

    assert remote.user == "nba-prop-deploy"

    assert remote.credential_env == "NBA_WOO_SSH_PRIVATE_KEY"

    # Resolvable and still unpublishable: the switch refuses before a
    # destination is reached at all.
    switch = publisher.read_publishing_switch(
        publisher.MODE_LIVE, project_root=PROJECT
    )

    assert switch.state == publisher.STATE_DISABLED


def test_live_publishing_is_still_disabled():
    """Case 17. The committed switch, read rather than assumed."""
    switch = publisher.read_publishing_switch(
        publisher.MODE_LIVE, project_root=PROJECT
    )

    assert switch.state == publisher.STATE_DISABLED

    assert switch.enabled is False


def test_the_unnamed_live_destination_is_still_unconfigured():
    """No mode resolves to a real host by default, live least of all."""
    document = publisher.read_destinations(project_root=PROJECT)

    assert document["destinations"]["live"] is None

    with pytest.raises(publisher.PublishingRefused, match="no live destination"):
        publisher.resolve_destination(publisher.MODE_LIVE, document)


def test_the_page_destination_is_a_canary_and_declares_itself_one():
    document = publisher.read_destinations(project_root=PROJECT)

    entry = document[publisher.NAMED_DESTINATIONS_KEY][PAGE_DESTINATION]

    assert entry["mode"] == publisher.MODE_CANARY

    assert entry["user"] == "nba-prop-deploy"

    assert entry["credential_env"] == "NBA_WOO_SSH_PRIVATE_KEY"

    assert entry["transport"] == publisher.TRANSPORT_SSH_RSYNC

    assert "massey" not in json.dumps(entry).lower()

    assert "WIZARD_SSH" not in json.dumps(entry)

    with pytest.raises(publisher.PublishingRefused):
        publisher.resolve_destination(
            publisher.MODE_LIVE, document, name=PAGE_DESTINATION
        )


def test_the_page_destination_addresses_the_real_host_and_the_exact_path():
    """The host is the website's, and the public path is the one required.

    The path is case-sensitive at the server and ``NBA/player-props`` is what
    the product requires, so the lower-case feed path and the underscore
    spelling are both asserted absent. The host is asserted equal to the feed
    canary's rather than written out again, because the two are the same
    machine and a second copy of a hostname is a second thing to get wrong --
    which is exactly how this entry first went in holding a placeholder.
    """
    document = publisher.read_destinations(project_root=PROJECT)

    named = document[publisher.NAMED_DESTINATIONS_KEY]

    entry = named[PAGE_DESTINATION]

    host = named["wizardofodds_nba_canary"]["host"]

    assert entry["host"] == host

    assert entry["public_base_url"] == (
        f"https://{host}/tools/odds-scanner/predictions/NBA/player-props/"
    )

    assert entry["public_target"] == (
        "/var/www/sportsodds/tools/odds-scanner/predictions/NBA/player-props"
    )

    addressed = entry["public_base_url"] + " " + entry["public_target"]

    for wrong in (
        "/predictions/nba/player-props",
        "player_props",
        "basketball",
        "[REDACTED]",
        "example.com",
        "localhost",
    ):
        assert wrong not in addressed, wrong

    remote = page_destination().remote

    assert remote.published_base_url == (
        entry["public_base_url"] + "canary/"
    )


def test_the_existing_feed_canary_destination_is_untouched():
    """The page reads the feed canary and this branch does not re-address it."""
    document = publisher.read_destinations(project_root=PROJECT)

    feed = document[publisher.NAMED_DESTINATIONS_KEY]["wizardofodds_nba_canary"]

    assert feed["public_target"] == (
        "/var/www/sportsodds/tools/odds-scanner/predictions/nba"
    )

    assert feed["canary_subpath"] == "canary"

    assert feed["release_prefix"] == "nba_prop_quant"

    assert feed["retained_releases"] == 2


# ======================================================================
# publishing the page
# ======================================================================


def test_publishing_the_page_writes_both_files_and_reads_them_back(
    with_affiliates, tmp_path
):
    page_root = tmp_path / "page-destination"

    page_transport = publisher.LocalDirectoryTransport(page_root)

    receipt = page.publish_page(
        source_transport=with_affiliates["transport"],
        page_transport=page_transport,
        registry=REGISTRY,
    )

    assert receipt["written"] == [page.PAGE_NAME, page.PAGE_MANIFEST_NAME]

    assert sorted(page_transport.listing()) == sorted(
        (page.PAGE_NAME, page.PAGE_MANIFEST_NAME)
    )

    built = build(with_affiliates)

    for name, payload in built["files"].items():
        assert (page_root / name).read_bytes() == payload

    assert receipt["manifest"]["page_sha256"] == publisher.digest(
        built["files"][page.PAGE_NAME]
    )


def test_a_prefix_publishes_beside_the_page_without_replacing_it(
    with_affiliates, tmp_path
):
    """A proof page goes in a child directory, leaving the page where it is.

    Published twice into one destination: once at the root, once under a
    prefix. The second must not disturb the first, which is what makes a
    routing-proof canary publishable beside a page canary rather than over it.
    """
    page_root = tmp_path / "page-destination"

    page_transport = publisher.LocalDirectoryTransport(page_root)

    page.publish_page(
        source_transport=with_affiliates["transport"],
        page_transport=page_transport,
        registry=REGISTRY,
    )

    before = {
        name: (page_root / name).read_bytes()
        for name in (page.PAGE_NAME, page.PAGE_MANIFEST_NAME)
    }

    receipt = page.publish_page(
        source_transport=with_affiliates["transport"],
        page_transport=page_transport,
        registry=REGISTRY,
        prefix="routing-proof",
    )

    assert receipt["prefix"] == "routing-proof"

    assert receipt["written"] == [
        f"routing-proof/{page.PAGE_NAME}",
        f"routing-proof/{page.PAGE_MANIFEST_NAME}",
    ]

    for name, payload in before.items():
        assert (page_root / name).read_bytes() == payload

        assert (page_root / "routing-proof" / name).read_bytes() == payload


@pytest.mark.parametrize(
    "prefix",
    ["..", "../canary", "/var/www/sportsodds", "routing-proof/../..", "-rf"],
)
def test_a_prefix_that_climbs_out_of_the_destination_is_refused(
    with_affiliates, tmp_path, prefix
):
    """The prefix is confined by the same predicate every other path is.

    The absolute case is the one worth having: stripping its leading slash
    would make it a legal relative path, and the page would be published at a
    location nobody asked for while the receipt reported success.
    """
    page_root = tmp_path / "out"

    with pytest.raises(page.PageRefused, match="does not address a directory below"):
        page.publish_page(
            source_transport=with_affiliates["transport"],
            page_transport=publisher.LocalDirectoryTransport(page_root),
            registry=REGISTRY,
            prefix=prefix,
        )

    assert not page_root.exists() or list(page_root.rglob("*.html")) == []


def test_publishing_the_page_touches_nothing_at_the_source(with_affiliates, tmp_path):
    """The feed canary is read from and never written to."""
    before = {
        name: with_affiliates["transport"].read_bytes(name)
        for name in with_affiliates["transport"].listing()
    }

    page.publish_page(
        source_transport=with_affiliates["transport"],
        page_transport=publisher.LocalDirectoryTransport(tmp_path / "out"),
        registry=REGISTRY,
    )

    after = {
        name: with_affiliates["transport"].read_bytes(name)
        for name in with_affiliates["transport"].listing()
    }

    assert after == before


def test_a_refused_publication_writes_no_page(published, tmp_path):
    manifest = load_manifest(published)

    manifest["model_authority"] = "candidate"

    rewrite(published, manifest_path(published), publisher.serialise(manifest))

    page_root = tmp_path / "out"

    with pytest.raises(page.PageRefused):
        page.publish_page(
            source_transport=published["transport"],
            page_transport=publisher.LocalDirectoryTransport(page_root),
            registry=REGISTRY,
        )

    assert not page_root.exists() or list(page_root.rglob("*")) == []


def test_the_page_manifest_records_what_a_check_needs(with_affiliates):
    built = build(with_affiliates)

    manifest = built["manifest"]

    assert manifest["contract_version"] == 5

    assert manifest["mode"] == publisher.MODE_CANARY

    assert manifest["released"]["release_id"] == built["publication"].release_id

    assert manifest["released"]["rows_sha256"] == built["publication"].rows_sha256

    assert manifest["affiliate_registry"]["routable_books"] == [
        "BetAnything",
        "BetOnline",
        "BetUS",
        "Bovada",
    ]

    assert manifest["affiliate_registry"]["primary_books"] == [
        "BetAnything",
        "BetOnline",
        "BetUS",
        "Bovada",
    ]

    assert manifest["affiliate_registry"]["missing_facts"] == {}

    assert manifest["affiliate_registry"]["tie_priority"] == [
        "BetOnline",
        "Bovada",
        "BetAnything",
        "BetUS",
    ]

    assert len(manifest["routed"]) == len(built["rows"])

    assert manifest["routed"][0]["destination"] == REGISTRY.destination_for(
        "bovada"
    )

    assert "priced_market_artifact_path" not in json.dumps(manifest)


def test_the_frozen_identifiers_are_untouched():
    """This is a page task, so the model must be exactly where it was."""
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
