"""The public page as a bettor sees it, rather than as its author sees it.

The page this replaced was correct and unusable: a dark developer dashboard
whose widest elements were a rows digest, an input-state fingerprint and a
production SHA, whose predictions were tall cards, and whose filter count
could say "0 predictions shown of 4" while four cards stayed on screen. Every
test here pins one of the things that made it read as a QA artefact instead of
a sportsbook predictions page, so that fixing one of them later cannot quietly
undo another.

Three of them are worth saying out loud.

The filter count is tested by running the page's own script, in Node, against
the page's own markup -- see ``wizardofodds_page_dom.js``. Reimplementing the
filter predicate in Python would prove only that two copies of a rule agree
with each other, and the defect being guarded against was precisely a
disagreement between the count and the rows. The other half of that defect was
CSS: an author rule setting a display on a row beats the ``hidden`` attribute,
so the stylesheet is asserted separately.

Nothing here asserts a number the model produced. The numbers are asserted
equal to the published release in ``test_wizardofodds_player_props_page.py``;
what is asserted here is that the model's files and the routing's files were
not touched by a presentation change, by digest.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
import wizardofodds_fallback_destinations as fallback  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402
import wizardofodds_routing_proof as proof  # noqa: E402
from test_wizardofodds_player_props_page import (  # noqa: E402
    REGISTRY,
    affiliate_frame,
    publish,
)

HARNESS = Path(__file__).resolve().parent / "wizardofodds_page_dom.js"

#: The files this task was forbidden to touch, by digest. A presentation
#: change that reached the routing rule, the affiliate mapping, the fallback
#: destinations or the model specification would change one of these, and the
#: reviewer would be told which.
FROZEN_FILES: dict[str, str] = {
    "ops/wizardofodds_affiliate_routing.py": (
        "5dd7008c6007769bcffe53eb39fecd3a07526fb3b5adf4d9413b4600573a844c"
    ),
    "ops/wizardofodds_affiliates.json": (
        "157ad7626348a320b75285252af1d3d77ef5349aee50d78d85ea91f943a97f6d"
    ),
    "ops/wizardofodds_fallback_destinations.py": (
        "9c7b01111c9168d2010cd5a4a2ba3e549813410c27240e98059c23ecfa31647f"
    ),
    "ops/wizardofodds_fallback_destinations.json": (
        "b41da836dde91ea50603fcffd94cc535908e75cfa04997812a04b54e53b599cc"
    ),
}

#: The model identifiers the operator froze. ``FACTOR`` is the hash recorded
#: *inside* the model specification rather than the digest of the factor file,
#: because that is the value the specification itself commits to.
MODEL_SPEC_SHA256 = (
    "154109658127920ebeac381bd21e8bb29390505384124419899236cce37fdc64"
)

FACTOR_SPEC_HASH = (
    "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
)

PROMOTION_POLICY_SHA256 = (
    "a0784aff602d9ee14ba1ef91b30fe972fb7eeb02ec80e0b8afbb7d9ff764aee8"
)

#: Vocabulary that belongs to the routing-proof QA page and to the internal
#: routing rule, none of which a bettor should ever be shown.
QA_VOCABULARY: tuple[str, ...] = (
    "routing proof",
    "routing classification",
    "tier primary",
    "tier fallback",
    "source proof",
    "test fixture",
    "best price among the affiliate books offering this exact wager",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ======================================================================
# the three slates these tests need
# ======================================================================


@pytest.fixture(scope="module")
def consumer(tmp_path_factory) -> dict:
    """The ordinary publication, rendered for a bettor.

    Its books are not affiliates and none of them is linkable, which is the
    real canary slate's situation: every prediction is a fallback carrying a
    price and no button.
    """
    root = tmp_path_factory.mktemp("consumer")

    state = publish(root)

    return page.build_page(state["transport"], registry=REGISTRY)


@pytest.fixture(scope="module")
def consumer_html(consumer) -> str:
    return consumer["files"][page.PAGE_NAME].decode("utf-8")


@pytest.fixture(scope="module")
def both_links(tmp_path_factory) -> dict:
    """A slate with one affiliate wager and one directly linkable wager.

    Two wagers rather than one so the paid button and the unpaid button are
    both on the page at once and can be told apart by markup rather than by
    which build produced them.
    """
    root = tmp_path_factory.mktemp("links")

    import pandas as pd

    affiliate = affiliate_frame((("bovada", -120),))

    direct = affiliate_frame((("draftkings", 101),))

    second = direct.iloc[0].to_dict()

    second.update({"id": 8500, "player_id": 9500, "player_name": "Second Wager"})

    frame = pd.concat(
        [affiliate, pd.DataFrame([second])], ignore_index=True
    )

    state = publish(root, frame=frame)

    return page.build_page(state["transport"], registry=REGISTRY)


@pytest.fixture(scope="module")
def proof_html(tmp_path_factory) -> str:
    """The QA page, which keeps its own presentation and its own warning."""
    root = tmp_path_factory.mktemp("proof")

    transport = proof.publish_locally(
        root, production_sha="5b5e658492fa5acd84d84447d5a7c54c2d9f9dc9"
    )

    built = page.build_page(
        transport, registry=REGISTRY, presentation=page.ROUTING_PROOF_QA
    )

    return built["files"][page.PAGE_NAME].decode("utf-8")


def rows_in(html: str) -> list[str]:
    return re.findall(r"<tr data-row.*?</tr>", html, re.S)


def row_for(html: str, page_row) -> str:
    """The one row about one wager.

    Keyed on the stable sort key rather than the player's name, because one
    player has several props on a slate and a name matches all of them.
    """
    key = "|".join(page_row.stable_order)

    matching = [row for row in rows_in(html) if f'data-stable="{key}"' in row]

    assert len(matching) == 1, key

    return matching[0]


def visible_text(html: str) -> str:
    """The document without the machine-readable note embedded in it.

    The page carries its release id and a sentence about what it does not
    compute inside a ``application/json`` script element, for a checker rather
    than a reader. Tests about what a reader is shown have to exclude it or
    they are testing the wrong half of the file.
    """
    return re.sub(
        r'<script type="application/json".*?</script>', "", html, flags=re.S
    )


def run_script(html: str, tmp_path: Path, **controls: str) -> dict:
    """What the page's own script does to the page, with these controls set."""
    if shutil.which("node") is None:  # pragma: no cover - CI has node
        pytest.fail("node is required to execute the page's own filter script")

    document = tmp_path / "page.html"

    document.write_text(html, encoding="utf-8")

    finished = subprocess.run(
        [
            "node",
            str(HARNESS),
            str(document),
            json.dumps(controls),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(finished.stdout)


# ======================================================================
# 1-2: it looks like the site, and the table is the page
# ======================================================================


def test_the_page_is_light_and_is_not_a_dark_dashboard(consumer_html):
    """Case 1. White background, dark text, and no full-page dark theme.

    Asserted on the declarations that actually decide it rather than on a
    class name: the body's background, the page's text colour token, and the
    absence of the dark-theme scaffolding the previous page was built on.
    """
    assert '<body class="wf-body">' in consumer_html

    assert ".wf-body{margin:0;background:#fff" in consumer_html

    assert "--wf-bg:#fff" in consumer_html

    assert "--wf-text:#23262b" in consumer_html

    assert "color-scheme:light" in consumer_html

    for dark in (
        "color-scheme:dark",
        "prefers-color-scheme:dark",
        "background:#0",
        "background:#1",
        "background:#111",
        "#e7ecf3",
    ):
        assert dark not in consumer_html, dark


def test_the_primary_content_surface_is_a_compact_table(consumer_html):
    """Case 2. A dense table of predictions, not a column of large cards."""
    assert '<table class="wf-table" id="picks">' in consumer_html

    assert "<thead>" in consumer_html

    assert '<tbody id="picks-body">' in consumer_html

    assert len(rows_in(consumer_html)) == len(
        re.findall(r"<tr data-row", consumer_html)
    )

    assert "<article" not in consumer_html

    assert 'class="card' not in consumer_html

    assert ".wf-table{border-collapse:collapse" in consumer_html

    assert ".wf-table th,.wf-table td{padding:7px 9px" in consumer_html


# ======================================================================
# 3-4: the QA page and the consumer page are different pages
# ======================================================================


@pytest.mark.parametrize("phrase", QA_VOCABULARY)
def test_the_consumer_page_carries_no_qa_vocabulary(consumer_html, phrase):
    """Case 3. None of the routing rule's own words reach a bettor."""
    assert phrase not in consumer_html.lower()


def test_the_consumer_page_announces_itself_as_a_preview(consumer_html):
    """Case 3's other half: it says what it is, and it is not live yet."""
    assert page.CONSUMER_PREVIEW.banner == "CANARY PREVIEW — NOT YET LIVE"

    assert "CANARY PREVIEW — NOT YET LIVE" in consumer_html

    assert 'class="wf-note warn"' in consumer_html


def test_the_routing_proof_page_is_still_labelled_as_test_data(proof_html):
    """Case 4. The evidence survives, and it cannot be read as betting picks."""
    assert page.ROUTING_PROOF_QA.banner == (
        "TEST DATA — ROUTING PROOF — NOT LIVE BETTING PICKS"
    )

    assert "TEST DATA — ROUTING PROOF — NOT LIVE BETTING PICKS" in proof_html

    assert 'class="wf-note err"' in proof_html

    assert "Routing Proof A" in proof_html

    assert "sportsbook offers matched this exact wager" in proof_html


def test_the_two_presentations_cannot_be_confused_for_each_other(
    consumer_html, proof_html
):
    """A consumer build is never handed the QA wording, by construction."""
    assert page.CONSUMER_PREVIEW.show_routing_reason is False

    assert page.ROUTING_PROOF_QA.show_routing_reason is True

    assert "sportsbook offers matched this exact wager" not in consumer_html

    assert page.CONSUMER_PREVIEW.banner not in proof_html


# ======================================================================
# 5-6: columns the data cannot honestly fill
# ======================================================================


def test_no_internal_identifier_is_dressed_up_as_a_matchup(consumer, consumer_html):
    """Case 5. No "Team 71 vs Team 72", and no matchup column at all."""
    assert "matchup" not in page.visible_columns(consumer["rows"])

    assert "Matchup" not in consumer_html

    for page_row in consumer["rows"]:
        assert page_row.matchup is None

        for field in ("game_id", "team_id", "opponent_id"):
            identifier = page_row.row.get(field)

            if identifier in (None, ""):
                continue

            assert f"Team {identifier}" not in consumer_html

            assert f"Game {identifier}" not in consumer_html

    assert not re.search(r"Team \d+ (?:vs|@) Team \d+", consumer_html)


def test_a_projection_is_absent_rather_than_invented_and_fair_odds_are_shown(
    consumer, consumer_html
):
    """Case 5's other half: a column exists when a field does, not before.

    The published contract carries no projection, so the column is gone; it
    does carry fair odds on both sides, so those are rendered. Both are facts
    about the release rather than opinions in the renderer, which is why
    ``visible_columns`` is asked rather than told.
    """
    columns = page.visible_columns(consumer["rows"])

    assert "projection" not in columns

    assert "Model Proj" not in consumer_html

    assert "fair" in columns

    assert "Fair Odds" in consumer_html

    for page_row in consumer["rows"]:
        assert page_row.projection is None

        assert page_row.fair_odds is not None

        assert page.american(page_row.fair_odds) in consumer_html


def test_no_ambiguous_market_probability_column_is_rendered(consumer_html):
    """Case 6. The devigged pair is not the price's own probability.

    ``market_devig_q_over`` and ``market_devig_q_under`` are the pair implied
    by a book's two prices together, not the probability implied by the single
    price the row displays, and edge inherits that ambiguity. Both fields stay
    published and untouched; neither is given a column, because a column
    labelled Market Probability next to one price would be read as that
    price's probability. Win %, fair odds, market odds and EV say everything
    the reader needs and each is unambiguous.
    """
    lowered = visible_text(consumer_html).lower()

    for ambiguous in (
        "market probability",
        "market prob",
        "implied probability",
        "devig",
        "model edge",
        "edge",
    ):
        assert ambiguous not in lowered, ambiguous

    assert "data-edge=" not in consumer_html

    assert "Win %" in consumer_html

    assert "Market Odds" in consumer_html


# ======================================================================
# 7-8: what every row has to say
# ======================================================================


def test_every_row_leads_with_the_player_the_prop_and_the_pick(
    consumer, consumer_html
):
    """Case 7. The three things a bettor reads first, in that order."""
    columns = page.visible_columns(consumer["rows"])

    assert columns[:4] == ("player", "prop", "pick", "win")

    for page_row in consumer["rows"]:
        row = row_for(consumer_html, page_row)

        cells = re.findall(r'<td class="([^"]+)"', row)

        assert cells[:3] == ["c-player", "c-prop", "c-pick"]

        assert str(page_row.row["player_name"]) in row

        assert page.prop_label(page_row.row["prop_type"]) in row

        assert page_row.side.upper() in row

        assert str(page_row.wager.line_value) in row


def test_the_four_numbers_on_a_row_are_the_published_ones(consumer, consumer_html):
    """Case 8. Win %, fair odds, market odds and EV, copied not computed."""
    for page_row in consumer["rows"]:
        row = row_for(consumer_html, page_row)

        side = page_row.side

        source = page_row.row

        assert page.percent(source[page.SIDE_MODEL_PROBABILITY[side]]) in row

        assert page.american(source[page.SIDE_FAIR_ODDS[side]]) in row

        assert page.american(page_row.routed.offer.american_odds) in row

        assert page.signed_percent(source[page.SIDE_EXPECTED_VALUE[side]]) in row


def test_positive_odds_and_positive_expected_value_carry_their_sign():
    """Case 10's formatting half, on the two places a sign changes meaning."""
    assert page.american(101) == "+101"

    assert page.american(-130) == "-130"

    assert page.signed_percent(0.068) == "+6.8%"

    assert page.signed_percent(-0.068) == "-6.8%"

    assert page.prop_label("threes") == "3PM"

    assert page.prop_label("points") == "Points"


# ======================================================================
# 9-11: the three things a sportsbook cell can be
# ======================================================================


def test_the_affiliate_call_to_action_is_unchanged_and_obvious(both_links):
    """Case 9. The gold button, the site's redirect, and the paid rel."""
    html = both_links["files"][page.PAGE_NAME].decode("utf-8")

    paid = re.findall(
        r'<a class="wf-bo-btn" href="([^"]+)" target="_blank" '
        r'rel="([^"]+)">([^<]+)</a>',
        html,
    )

    assert len(paid) == 1

    href, rel, label = paid[0]

    assert href == REGISTRY.destination_for("bovada")

    assert href.startswith("/tools/odds-scanner/predictions/api/go.php?book=")

    assert rel == "sponsored nofollow noopener"

    assert label == "BET AT BOVADA -120"

    assert ".wf .wf-bo-btn{" in html

    assert "background:var(--wf-gold)" in html

    assert '<span class="wf-tag paid">Paid link</span>' in html


def test_the_fallback_call_to_action_is_distinct_and_still_obvious(both_links):
    """Case 10. Same shape, different colour, different verb, unpaid rel."""
    html = both_links["files"][page.PAGE_NAME].decode("utf-8")

    direct = re.findall(
        r'<a class="wf-bo-btn direct" href="([^"]+)" target="_blank" '
        r'rel="([^"]+)">([^<]+)</a>',
        html,
    )

    assert len(direct) == 1

    href, rel, label = direct[0]

    assert href == fallback.FallbackDestinations.load(
        project_root=PROJECT
    ).url_for("draftkings")

    assert href.startswith("https://")

    assert "go.php" not in href

    assert rel == fallback.REQUIRED_REL

    assert "sponsored" not in rel

    assert label == "VIEW AT DRAFTKINGS +101"

    assert ".wf .wf-bo-btn.direct{" in html

    assert "Not a paid placement" in html


def test_a_wager_nobody_prices_is_not_presented_as_actionable(
    tmp_path, monkeypatch
):
    """Case 11. Its own section, no price, no expected value, no button."""
    state = publish(tmp_path, frame=affiliate_frame((("bovada", -120),)))

    manifest_relative = json.loads(
        (state["destination"] / publisher.CURRENT_POINTER).read_text()
    )["manifest_relative"]

    manifest = json.loads((state["destination"] / manifest_relative).read_text())

    document = json.loads(
        (state["destination"] / manifest["rows_relative"]).read_text()
    )

    for row in document["rows"]:
        row["model_preferred_side"] = "under"

        row["under_odds"] = None

    payload = publisher.serialise(document)

    (state["destination"] / manifest["rows_relative"]).write_bytes(payload)

    manifest["rows_sha256"] = publisher.digest(payload)

    (state["destination"] / manifest_relative).write_bytes(
        publisher.serialise(manifest)
    )

    pointer = json.loads(
        (state["destination"] / publisher.CURRENT_POINTER).read_text()
    )

    pointer["rows_sha256"] = manifest["rows_sha256"]

    (state["destination"] / publisher.CURRENT_POINTER).write_bytes(
        publisher.serialise(pointer)
    )

    built = page.build_page(state["transport"], registry=REGISTRY)

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert built["manifest"]["rendered"]["no_offer_count"] >= 1

    assert built["manifest"]["rendered"]["offered_count"] == 0

    assert '<section class="wf-pending" id="pending">' in html

    assert "No current offer" in html

    assert "there is no bet to place" in html

    assert "<a " not in html

    for row in rows_in(html):
        assert 'data-offer="no"' in row

        assert 'class="c-market' not in row

        assert 'class="c-ev' not in row

    assert built["rows"][0].row["under_odds"] is None


# ======================================================================
# 12-13: the controls
# ======================================================================


def test_the_sportsbook_filter_says_what_it_actually_filters(consumer_html):
    """Case 12. One minimal fix: the honest label, not a redesigned filter.

    It filters on the one sportsbook each prediction features, which is not
    the same thing as every sportsbook quoting the wager, so it is named
    "Featured book" and left alone.
    """
    assert '<label for="f-book">Featured book</label>' in consumer_html

    assert "All sportsbooks" in consumer_html

    assert "Sportsbook filter" not in consumer_html

    assert 'id="f-game"' not in consumer_html

    assert "data-game=" not in consumer_html


def test_there_is_no_game_time_control_of_any_kind(consumer_html):
    """Case 13. No tip-off field is published, so no ordering claims one."""
    assert "game time" not in consumer_html.lower()

    assert "tip-off" not in consumer_html.lower().split("does not have")[0]

    assert 'value="game"' not in consumer_html

    sorts = re.search(r'<select id="f-sort">(.*?)</select>', consumer_html, re.S)

    assert re.findall(r'value="([^"]+)"', sorts.group(1)) == [
        "ev",
        "win",
        "player",
    ]


# ======================================================================
# 14-15: the count and the rows can never disagree
# ======================================================================


@pytest.mark.parametrize(
    "controls",
    [
        {},
        {"prop": "points"},
        {"prop": "rebounds"},
        {"search": "P1101"},
        {"search": "fixture"},
        {"search": "zzzznothing"},
        {"prop": "points", "search": "P1101"},
        {"sort": "win"},
        {"sort": "player"},
        {"sort": "player", "prop": "points"},
    ],
)
def test_the_visible_count_equals_the_rows_actually_visible(
    consumer_html, tmp_path, controls
):
    """Case 14. The defect this page was corrected for, pinned.

    The old page could say "0 predictions shown of 4" with four cards still on
    screen. Run here through the page's own script so the count being asserted
    is the one the page writes, over the search box, the prop filter, the
    featured-book filter and all three sorts.
    """
    result = run_script(consumer_html, tmp_path, **controls)

    shown = int(re.search(r"Showing (\d+) of", result["count_text"]).group(1))

    visible = [row for row in result["visible"] if row["offer"] == "yes"]

    assert shown == len(visible), result["count_text"]

    needle = controls.get("search", "").strip().lower()

    for row in visible:
        if needle:
            assert needle in row["search"]

        if controls.get("prop", "all") != "all":
            assert row["prop"] == controls["prop"]

    assert result["empty_hidden"] is (shown != 0)

    assert len(result["order"]) == 4


def test_the_featured_book_filter_also_keeps_the_count_honest(
    both_links, tmp_path
):
    """Case 14, over the one filter whose options come from the data."""
    html = both_links["files"][page.PAGE_NAME].decode("utf-8")

    for book, expected in (("all", 2), ("bovada", 1), ("draftkings", 1)):
        result = run_script(html, tmp_path, book=book)

        shown = int(
            re.search(r"Showing (\d+) of", result["count_text"]).group(1)
        )

        assert shown == expected, book

        assert len(result["visible"]) == expected, book

        if book != "all":
            assert {row["book"] for row in result["visible"]} == {book}


def test_a_hidden_row_is_hidden_by_two_mechanisms_not_one(consumer_html):
    """Case 15. The CSS half of the same defect.

    The original bug was not in the script: it set ``hidden`` correctly and
    counted correctly. It was that an author rule setting a display on a row
    beats the user agent's ``[hidden]{display:none}``, so the row kept
    painting. Both halves are fixed -- the stylesheet states it explicitly and
    the script writes an inline display that no stylesheet can override --
    and both are asserted, because either one alone would have shipped the
    defect.
    """
    assert ".wf [hidden]{display:none!important}" in consumer_html

    assert ".wf-table tbody tr[hidden]{display:none!important}" in consumer_html

    assert "row.style.display = ok ? '' : 'none'" in consumer_html

    assert "row.hidden = !ok" in consumer_html


# ======================================================================
# 16-17: provenance and disclosure, present but not the page
# ======================================================================


def test_the_engineering_provenance_is_no_longer_the_page(consumer, consumer_html):
    """Case 16. Three facts in the header; the rest one disclosure away.

    Nothing is deleted: the digest, the release id, the input fingerprint and
    the full SHA are all still published in the page manifest and all still
    rendered inside the collapsed methodology section. They are simply not
    what the top of a betting page is for.
    """
    header = consumer_html.split('<section class="wf-controls"')[0]

    assert "Model updated" in header

    assert "Model version" in header

    assert "Production build" in header

    full_sha = consumer["publication"].provenance["production_code_sha"]

    assert full_sha[:12] in header

    assert full_sha not in header

    for internal in (
        "<dt>Rows digest</dt>",
        "<dt>Release</dt>",
        "<dt>Mode</dt>",
        "input_state",
    ):
        assert internal not in header, internal

    assert "<dt>Rows digest</dt>" in consumer_html

    assert consumer["publication"].rows_sha256 in consumer_html

    assert full_sha in consumer_html

    released = consumer["manifest"]["released"]

    assert released["rows_sha256"] == consumer["publication"].rows_sha256

    assert consumer["manifest"]["provenance"]["production_code_sha"] == full_sha

    assert consumer["manifest"]["provenance"]["input_state_fingerprint"]


def test_the_disclosure_is_one_sentence_up_top_and_complete_below(
    consumer_html,
):
    """Case 17. Visible where it matters, in full where it is read."""
    above = consumer_html.split('<div class="wf-tablewrap">')[0]

    assert page.SHORT_DISCLOSURE in above

    assert "WizardOfOdds may earn commission from certain sportsbook links." == (
        page.SHORT_DISCLOSURE
    )

    full = escape(page.AFFILIATE_DISCLOSURE)

    assert full not in above

    assert full in consumer_html

    assert '<details class="wf-gloss">' in consumer_html

    assert "Methodology, sportsbook selection and full provenance" in (
        consumer_html
    )

    reading = re.sub(r"<[^>]+>", " ", above.split("<h1")[1])

    words = len(reading.split())

    assert words < 120, (
        f"{words} words of preamble before the first prediction; the "
        "disclosure block this replaced consumed the top of the page"
    )


# ======================================================================
# 18-19: it renders on a desktop and on a phone
# ======================================================================


def test_the_desktop_document_is_complete_and_self_contained(
    consumer, consumer_html
):
    """Case 18. Every prediction present, nothing fetched, nothing missing."""
    assert consumer_html.startswith("<!doctype html>")

    assert consumer_html.rstrip().endswith("</html>")

    assert len(rows_in(consumer_html)) == len(consumer["rows"])

    assert consumer_html.count("<table") == consumer_html.count("</table>")

    assert consumer_html.count("<tbody") == consumer_html.count("</tbody>")

    assert "<title>NBA Player Props Predictions | WizardOfOdds</title>" in (
        consumer_html
    )

    assert '<h1 class="wf-h1">NBA Player Props Predictions</h1>' in consumer_html

    assert re.findall(r'src="([^"]+)"', consumer_html) == []

    assert "<link" not in consumer_html

    assert "<img" not in consumer_html


def test_the_phone_layout_is_readable_without_sideways_scrolling(consumer_html):
    """Case 19. The table becomes a labelled record, and the button is big."""
    assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in (
        consumer_html
    )

    narrow = consumer_html.split("@media (max-width:720px)")[1]

    assert ".wf-table thead{display:none}" in narrow

    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in narrow

    assert "overflow-x:visible" in narrow

    assert "min-height:44px" in narrow

    assert "content:attr(data-label)" in narrow

    for row in rows_in(consumer_html):
        labels = re.findall(r'data-label="([^"]+)"', row)

        assert "Player" in labels

        assert "Pick" in labels

        assert "Sportsbook" in labels


# ======================================================================
# 20-21: the backend this task was not allowed to touch
# ======================================================================


@pytest.mark.parametrize("relative", sorted(FROZEN_FILES))
def test_the_routing_and_affiliate_files_are_untouched(relative):
    """Case 20. A presentation change did not reach the selection rule."""
    assert digest(PROJECT / relative) == FROZEN_FILES[relative], relative


def test_the_page_selects_nothing_and_prices_nothing(consumer):
    """Case 20's other half: selection is the router's answer, rendered.

    The page asks the router once per wager and renders what it is given. It
    recomputes no probability and no expected value, and it asks whether a
    sportsbook can be linked only after the router has already named one.
    """
    source = (PROJECT / "ops" / "wizardofodds_player_props_page.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "import nba_prop_quant",
        "from nba_prop_quant",
        "price_markets(",
        "payout_multiple(",
        "math.log",
        "math.exp",
    ):
        assert forbidden not in source, forbidden

    for page_row in consumer["rows"]:
        independently = routing.route(
            routing.offers_for(page_row.quotes, page_row.wager),
            registry=REGISTRY,
        )

        assert page_row.routed.tier == independently.tier

        assert page_row.routed.offer == independently.offer

        assert page_row.routed.best_overall == independently.best_overall


def test_the_frozen_model_identifiers_are_untouched():
    """Case 21. No model artefact, parameter or specification moved."""
    specification = PROJECT / "research" / "final_model" / "final_model_spec.json"

    assert digest(specification) == MODEL_SPEC_SHA256

    assert json.loads(specification.read_text())["factor_spec_hash"] == (
        FACTOR_SPEC_HASH
    )

    assert digest(
        PROJECT / "research" / "final_model" / "live_shadow_promotion_policy.json"
    ) == PROMOTION_POLICY_SHA256


# ======================================================================
# 22-25: where it is published, and what the manifest records
# ======================================================================


def test_the_preview_is_published_only_beneath_its_own_subpath(tmp_path):
    """Case 22. The live root is not written, and neither is the page canary."""
    destination = tmp_path / "page-destination"

    transport = publisher.LocalDirectoryTransport(destination)

    transport.write_bytes(page.PAGE_NAME, b"the page canary's own index")

    state = publish(tmp_path / "slate")

    assert page.PREVIEW_PREFIX == "page-preview"

    published = page.publish_page(
        source_transport=state["transport"],
        page_transport=transport,
        registry=REGISTRY,
        presentation=page.CONSUMER_PREVIEW,
        prefix=page.PREVIEW_PREFIX,
    )

    assert sorted(published["written"]) == [
        f"page-preview/{page.PAGE_NAME}",
        f"page-preview/{page.PAGE_MANIFEST_NAME}",
    ]

    assert (destination / page.PAGE_NAME).read_bytes() == (
        b"the page canary's own index"
    )

    assert (destination / "page-preview" / page.PAGE_NAME).exists()

    assert not (destination / "routing-proof").exists()


def test_the_manifest_records_the_presentation_and_the_columns(consumer):
    """Case 23. A check can read what was rendered without parsing HTML."""
    manifest = consumer["manifest"]

    assert manifest["contract_version"] == 4

    assert manifest["presentation"] == {
        "banner": page.CONSUMER_PREVIEW.banner,
        "routing_reason_shown": False,
        "tone": "warn",
    }

    rendered = manifest["rendered"]

    assert rendered["columns"] == list(page.visible_columns(consumer["rows"]))

    assert rendered["offered_count"] == len(consumer["rows"])

    assert rendered["no_offer_count"] == 0


def test_sorting_changes_the_order_and_never_the_visible_set(
    consumer_html, tmp_path
):
    """Case 24. A sort is a reordering, not a filter wearing a sort's label."""
    orders = {}

    for mode in ("ev", "win", "player"):
        result = run_script(consumer_html, tmp_path, sort=mode)

        orders[mode] = result["order"]

        assert len(result["visible"]) == 4

        assert "Showing 4 of 4 predictions" == result["count_text"]

    assert sorted(orders["ev"]) == sorted(orders["player"])

    assert orders["ev"] != orders["player"]


def test_the_page_still_fits_in_one_published_directory(consumer):
    """Case 25. Two files, as the publisher writes them, and nothing else."""
    assert sorted(consumer["files"]) == [
        page.PAGE_NAME,
        page.PAGE_MANIFEST_NAME,
    ]

    assert page.PAGE_NAME == "index.html"

    assert page.PAGE_MANIFEST_NAME == "page_manifest.json"
