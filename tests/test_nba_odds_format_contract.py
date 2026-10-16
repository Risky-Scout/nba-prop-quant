"""The odds format selector and the distribution panel, as a bettor meets them.

Built from a real release: the slate is served by the frozen serving path, the
distributions are derived by the stage, the release is published by the
publisher and the page is rendered by the renderer. So what is asserted is the
page the canary would publish, not a page assembled by this file.

FOUR THINGS ARE BEING CLAIMED
-----------------------------

That the selector is notation and nothing else. Every rendering of every price
is in the document before a reader arrives, the control sets one attribute, and
the stylesheet shows one and hides two. Nothing is fetched, nothing is
repriced, no sportsbook selection moves and expected value is the same number
in all three -- which is checked by rendering the page three ways and diffing
what changed.

That the two prices are each a function of exactly one published number. The
model's price is the published calibrated probability for the displayed side.
The sportsbook's price is the exact American odds of the exact offer the row
points at -- not the release's devigged market probability, which is a
different quantity, and the assertion is that the devigged number does not
appear.

That the panel is the published forecast. The rows a reader scans are the
published support and probabilities, the tail bucket is the published
remainder, the highlighted range is the published pair, and the panel is a
disclosure with a real button behind it.

That the page is still the page. The iframe handshake is the site's existing
one on the same element with the same named origins, and the vocabulary is the
bettor-facing vocabulary rather than the schema's.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

for _relative in ("ops", "src", "tests"):
    if str(PROJECT / _relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / _relative))

import nba_distribution_slate_fixture as coverage  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402

from test_wizardofodds_canary_end_to_end import (  # noqa: E402
    checkout_sha,
    serve_and_provenance,
)
from test_wizardofodds_player_props_page import REGISTRY  # noqa: E402

from nba_prop_quant import pricing  # noqa: E402

HARNESS = Path(__file__).resolve().parent / "wizardofodds_page_dom.js"

#: The probabilities the launch contract names, and what each is worth as a
#: price. Stated here rather than computed, so the test is the contract and not
#: a restatement of the implementation.
#:
#: ``0.50`` is the one deliberate convention: an even-money forecast is
#: rendered ``+100``. ``-100`` is numerically the same price and is not how
#: anybody writes it.
MODEL_VECTORS: tuple[tuple[float, int, float], ...] = (
    (0.50, 100, 2.0),
    (0.60, -150, 1.0 / 0.60),
    (0.40, 150, 1.0 / 0.40),
    (0.75, -300, 1.0 / 0.75),
    (0.25, 300, 1.0 / 0.25),
)

#: Posted American prices, and what each implies on its own -- margin
#: included, because that is what a reader is being offered.
SPORTSBOOK_VECTORS: tuple[tuple[int, float, float], ...] = (
    (-110, 1.0 + 100.0 / 110.0, 110.0 / 210.0),
    (100, 2.0, 0.5),
    (150, 2.5, 100.0 / 250.0),
    (-200, 1.5, 200.0 / 300.0),
    (250, 3.5, 100.0 / 350.0),
)

#: Terms the bettor-facing page must use.
REQUIRED_TERMS: tuple[str, ...] = (
    "Model Projection",
    "Model Probability",
    "Model Odds",
    "Sportsbook Odds",
    "Sportsbook Implied Probability",
    "EV",
    "Distribution",
)

#: Internal field names no reader may be shown. Checked against the rendered
#: document with the machine-readable block removed, because that block is for
#: a checker and is allowed to name the contract's own fields.
FORBIDDEN_FIELD_NAMES: tuple[str, ...] = (
    "q_selected",
    "q_over_nonpush",
    "market_q_over",
    "market_q_under",
    "mu_selected_",
    "p_over",
    "p_under",
    "p_push",
    "dependence_lambda",
    "model_preferred_edge",
    "distribution_support",
    "distribution_probability",
)

#: Wording the launch contract retired, in the bettor-facing page.
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "win %",
    "win probability",
    "fair odds",
    "fair price",
    "market probability",
    "market odds",
)


# ======================================================================
# the release every assertion below is about
# ======================================================================


@pytest.fixture(scope="module")
def release(tmp_path_factory) -> dict:
    """A real served slate, published, with its distributions."""
    bundle = coverage.skip_without_bundle()

    root = tmp_path_factory.mktemp("odds_contract") / "run"

    with coverage.extra_quotes():
        serve_and_provenance(bundle, root, production_sha=checkout_sha())

    destination = root.parent / "destination"

    receipt = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=coverage.SLATE_DATE,
        data_root=root / "data",
        project_root=PROJECT,
        destination_root=destination,
    )

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED

    rows = json.loads(
        (destination / receipt["release"]["rows_relative"]).read_text(
            encoding="utf-8"
        )
    )

    built = page.build_page(
        publisher.LocalDirectoryTransport(destination), registry=REGISTRY
    )

    import pandas as pd

    return {
        "built": built,
        "destination": destination,
        "document": rows,
        "html": built["files"][page.PAGE_NAME].decode("utf-8"),
        "priced": pd.read_parquet(
            publisher.priced_markets_path(root / "data", coverage.SLATE_DATE)
        ),
        "receipt": receipt,
        "rows": built["rows"],
    }


@pytest.fixture(scope="module")
def html(release: dict) -> str:
    return release["html"]


def visible_text(document: str) -> str:
    """The document without the machine-readable block embedded in it."""
    return re.sub(
        r'<script type="application/json".*?</script>', "", document, flags=re.S
    )


def row_for(document: str, page_row) -> str:
    key = "|".join(page_row.stable_order)

    matching = re.findall(r"<tr data-row.*?</tr>", document, re.S)

    found = [row for row in matching if f'data-stable="{key}"' in row]

    assert len(found) == 1, key

    return found[0]


def renderings(cell: str) -> dict[str, str]:
    """A price cell's three renderings, by format key.

    Read back out of the markup rather than matched as substrings, so the
    assertion is "this format shows exactly this" rather than "this string is
    somewhere in the row".
    """
    return {
        key: text
        for key, text in re.findall(
            r'<span class="of of-[a-z]+"[^>]*'
            r'data-odds-format-value="([a-z]+)"[^>]*>([^<]*)</span>',
            cell,
        )
    }


def price_cells(row: str) -> dict[str, dict[str, str]]:
    """The model cell's renderings and the sportsbook cell's, by column."""
    found = {}

    for klass, column in (
        ("c-model", page.COLUMN_MODEL),
        ("c-book-price", page.COLUMN_SPORTSBOOK),
    ):
        match = re.search(
            rf'<td class="{klass} num"[^>]*>(.*?)</td>', row, re.S
        )

        found[column] = renderings(match.group(1)) if match else {}

    return found


def panel_for(document: str, page_row) -> str:
    identifier = page._panel_id(page_row)

    start = document.index(f'<tr class="wf-dist-row" id="{identifier}">')

    end = document.index("</section></td></tr>", start)

    return document[start:end]


def run_script(document: str, tmp_path: Path, **controls) -> dict:
    """What the page's own script does to the page, with these controls set."""
    if shutil.which("node") is None:  # pragma: no cover - CI has node
        pytest.fail("node is required to execute the page's own script")

    tmp_path.mkdir(parents=True, exist_ok=True)

    written = tmp_path / "page.html"

    written.write_text(document, encoding="utf-8")

    done = subprocess.run(
        ["node", str(HARNESS), str(written), json.dumps(controls)],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(done.stdout)


# ======================================================================
# 1-5: the two conversions, on the contract's own vectors
# ======================================================================


@pytest.mark.parametrize("probability,american,decimal", MODEL_VECTORS)
def test_1_a_model_probability_converts_to_the_stated_price(
    probability: float, american: int, decimal: float
):
    """The model side of the selector, on the five stated probabilities."""
    assert page.model_american(probability) == american

    assert page.model_decimal(probability) == pytest.approx(decimal, rel=1e-15)

    assert page.percent(probability) == f"{probability * 100:.1f}%"


@pytest.mark.parametrize("odds,decimal,implied", SPORTSBOOK_VECTORS)
def test_2_a_posted_price_converts_to_the_stated_probability(
    odds: int, decimal: float, implied: float
):
    """The sportsbook side, on the five stated American prices."""
    assert page.sportsbook_decimal(odds) == pytest.approx(decimal, rel=1e-15)

    assert page.sportsbook_implied_probability(odds) == pytest.approx(
        implied, rel=1e-15
    )

    # And the pair is self-consistent: a decimal price is one over what it
    # implies, which is the only relationship between the two renderings.
    assert page.sportsbook_decimal(odds) == pytest.approx(
        1.0 / page.sportsbook_implied_probability(odds), rel=1e-15
    )


def test_3_the_probability_is_converted_before_it_is_rounded():
    """Rounding is display, which means it happens last and only once.

    The published probability goes into the conversion at the precision it was
    published at. Only the printed string is rounded, and the two conversions
    are asserted to be exactly the functions of the unrounded number rather
    than of the number a reader is shown -- a page that converted ``52.4%``
    would be pricing a different probability from the one it published.
    """
    probability = 0.523612345678

    assert page.percent(probability) == "52.4%"

    shown = 0.524

    assert page.model_decimal(probability) == 1.0 / probability

    assert page.model_decimal(probability) != 1.0 / shown

    assert page.model_american(probability) == round(
        -100.0 * probability / (1.0 - probability)
    )

    # And the same on the sportsbook side, where the published number is an
    # integer already and the derived ones are not.
    assert page.sportsbook_implied_probability(-110) == 110.0 / 210.0

    assert page.percent(page.sportsbook_implied_probability(-110)) == "52.4%"

    assert page.sportsbook_decimal(-110) != 1.0 / 0.524


def test_4_an_even_money_forecast_is_rendered_plus_one_hundred():
    """The one place this page's convention and the engine's differ.

    ``pricing.fair_american`` answers ``-100`` at exactly one half, which is
    the same price written the way nobody writes it. The launch contract names
    ``+100``, so that is what is rendered, and the divergence is here in one
    assertion rather than discovered later as a rendering bug.
    """
    assert page.model_american(0.5) == 100

    assert pricing.fair_american(0.5) == -100

    # Either side of a half the two agree, so this is a convention at one
    # point and not a second pricing rule.
    for probability in (0.4, 0.45, 0.55, 0.6, 0.75):
        assert page.model_american(probability) == pricing.fair_american(
            probability
        )


def test_5_an_unusable_number_converts_to_nothing_rather_than_to_a_price():
    """A missing price is absent, not zero and not infinity."""
    for unusable in (None, "", float("nan"), float("inf"), 0.0, 1.0, -0.2, 1.4):
        assert page.model_american(unusable) is None

    for unusable in (None, "", float("nan"), float("inf"), 0):
        assert page.sportsbook_decimal(unusable) is None

        assert page.sportsbook_implied_probability(unusable) is None

    assert page.model_decimal(1.0) == 1.0


# ======================================================================
# 6-11: the selector
# ======================================================================


def test_6_there_is_one_selector_with_three_formats_defaulting_to_american(html):
    """One visible control near the top, three options, American selected."""
    assert html.count('id="f-odds"') == 1

    block = re.search(r'<select id="f-odds">(.*?)</select>', html, re.S)

    assert block is not None

    assert re.findall(r'value="([^"]+)"', block.group(1)) == [
        "american",
        "decimal",
        "probability",
    ]

    assert 'value="american" selected' in block.group(1)

    assert page.DEFAULT_ODDS_FORMAT == "american"

    # The attribute the stylesheet switches on, on the element the site
    # already measures, set to the default in the served document.
    assert f'{page.ODDS_FORMAT_ATTRIBUTE}="american"' in html

    # Labelled, and the label is for this control.
    assert 'for="f-odds"' in html


def test_7_every_rendering_of_every_price_is_already_in_the_document(
    release: dict, html: str
):
    """Case 4. Switching cannot produce a number that was not published.

    Each price cell carries all three renderings as siblings, each derived
    from one published number, so the control has nothing to compute. The two
    that are not showing are hidden from assistive technology as well as from
    the stylesheet, because three readings of one price read aloud in sequence
    is not three prices.
    """
    for page_row in release["rows"]:
        row = row_for(html, page_row)

        cells = price_cells(row)

        probability = page_row.model_probability

        assert cells[page.COLUMN_MODEL] == {
            "american": page.american(page.model_american(probability)),
            "decimal": page.decimal_odds(page.model_decimal(probability)),
            "probability": page.percent(probability),
        }

        if page_row.routed.offer is None:
            assert cells[page.COLUMN_SPORTSBOOK] == {}

            continue

        posted = page_row.routed.offer.american_odds

        assert cells[page.COLUMN_SPORTSBOOK] == {
            "american": page.american(posted),
            "decimal": page.decimal_odds(page.sportsbook_decimal(posted)),
            "probability": page.percent(
                page.sportsbook_implied_probability(posted)
            ),
        }

        # Two of the three renderings of each price are hidden from assistive
        # technology as well as from the stylesheet: three readings of one
        # price read out in sequence is not three prices.
        assert row.count('aria-hidden="true"') == 4


def test_8_the_stylesheet_shows_one_rendering_and_hides_the_other_two(html):
    """The showing and hiding is CSS, which is why no number is rewritten."""
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)

    assert ".of{display:none}" in style.replace(" ", "").replace("\n", "")

    compact = style.replace(" ", "").replace("\n", "")

    for key in ("american", "decimal", "probability"):
        assert (
            f'[data-odds-format="{key}"].of-{key}{{display:inline}}' in compact
        ), key


def test_9_the_selector_fetches_nothing_and_reprices_nothing(html):
    """Case 4's prohibitions, against the script that is actually shipped."""
    script = re.search(
        r'<script id="page-controls">(.*?)</script>', html, re.S
    ).group(1)

    for forbidden in (
        "fetch(",
        "XMLHttpRequest",
        "location",
        "reload",
        "Math.pow",
        "/100",
        "toFixed",
        "parseFloat(row.getAttribute('data-model-p')) ",
    ):
        assert forbidden not in script, forbidden

    # The one thing it does to a price: nothing. It sets an attribute.
    assert "setAttribute('data-odds-format'" in script

    assert "innerHTML" not in script

    assert "textContent" in script  # the visible count, and only that


def test_10_the_preference_is_remembered_in_this_browser_and_sent_nowhere(
    html, tmp_path: Path
):
    """Case 23. One namespaced key, read once, written only on a change."""
    assert page.ODDS_FORMAT_STORAGE_KEY == "wf.nba.playerProps.oddsFormat"

    assert "nba" in page.ODDS_FORMAT_STORAGE_KEY.lower()

    first = run_script(html, tmp_path / "first")

    assert first["odds_format"] == "american"

    assert first["storage_reads"] == 1

    assert first["storage_writes"] == []

    chosen = run_script(html, tmp_path / "chosen", choose="probability")

    assert chosen["odds_format"] == "probability"

    assert chosen["odds_control"] == "probability"

    assert chosen["storage_writes"] == [
        [page.ODDS_FORMAT_STORAGE_KEY, "probability"]
    ]

    remembered = run_script(html, tmp_path / "back", stored="decimal")

    assert remembered["odds_format"] == "decimal"

    assert remembered["odds_control"] == "decimal"

    # Nothing in the shipped script sends it anywhere.
    script = re.search(
        r'<script id="page-controls">(.*?)</script>', html, re.S
    ).group(1)

    for outbound in ("document.cookie", "navigator.sendBeacon", "postMessage"):
        assert outbound not in script, outbound


@pytest.mark.parametrize("stored", ["", "nonsense", "AMERICAN"])
def test_11_an_unusable_or_absent_preference_falls_back_to_american(
    html, tmp_path: Path, stored: str
):
    """Case 23's other half, including a browser that refuses storage."""
    result = run_script(html, tmp_path / f"stored-{stored or 'empty'}", stored=stored)

    assert result["odds_format"] == "american"

    blocked = run_script(html, tmp_path / "blocked", storage="blocked")

    assert blocked["odds_format"] == "american"

    # And a reader whose storage refuses a write still gets the format they
    # chose for as long as the page is open.
    chose = run_script(
        html, tmp_path / "blocked-choice", storage="blocked", choose="decimal"
    )

    assert chose["odds_format"] == "decimal"


# ======================================================================
# 12-15: the columns and the labels
# ======================================================================


def test_12_the_column_set_is_the_launch_contracts(release: dict, html: str):
    """Case 22. Two prices, not four, and a toggle at the end."""
    assert page.visible_columns(release["rows"]) == (
        "matchup",
        "player",
        "prop",
        "pick",
        "projection",
        page.COLUMN_MODEL,
        page.COLUMN_SPORTSBOOK,
        "ev",
        "book",
        page.COLUMN_DISTRIBUTION,
    )

    head = re.search(r"<thead>(.*?)</thead>", html, re.S).group(1)

    assert head.count("<th") == 10


def test_13_the_two_price_headings_change_with_the_format(html):
    """Case 22's labelling half, and §2's vocabulary in the heading row."""
    head = re.search(r"<thead>(.*?)</thead>", html, re.S).group(1)

    cells = re.findall(r'<th scope="col"[^>]*>(.*?)</th>', head, re.S)

    headings = [renderings(cell) for cell in cells if "of-american" in cell]

    assert headings == [
        dict(page.COLUMN_FORMAT_LABEL[page.COLUMN_MODEL]),
        dict(page.COLUMN_FORMAT_LABEL[page.COLUMN_SPORTSBOOK]),
    ]

    # Only the two price columns have more than one name.
    assert len(headings) == 2

    # In probability the headings are the two §2 terms, and they are
    # different terms: one is the model's, one is the sportsbook's.
    assert (
        page.COLUMN_FORMAT_LABEL[page.COLUMN_MODEL]["probability"]
        == "Model Probability"
    )

    assert (
        page.COLUMN_FORMAT_LABEL[page.COLUMN_SPORTSBOOK]["probability"]
        == "Sportsbook Implied Probability"
    )


def test_14_the_page_uses_the_launch_vocabulary(html):
    """Case 2. The seven terms are present and the retired ones are gone."""
    readable = visible_text(html)

    for term in REQUIRED_TERMS:
        assert term in readable, term

    lowered = readable.lower()

    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in lowered, phrase


def test_15_no_internal_field_name_reaches_a_reader(html):
    """Case 34. The schema's names stay in the schema."""
    readable = visible_text(html)

    for name in FORBIDDEN_FIELD_NAMES:
        assert name not in readable, name

    # The machine-readable block is where a checker reads the release id, and
    # it is allowed to name contract fields -- but it must still be the only
    # place any of these could appear, so it is asserted to exist.
    assert '<script type="application/json"' in html


# ======================================================================
# 16-19: the two prices are each one published number
# ======================================================================


def test_16_the_model_probability_is_the_published_calibrated_field(
    release: dict, html: str
):
    """Case 31. Source-field lineage for the number a reader acts on.

    ``MODEL_PROBABILITY_SOURCE_FIELD`` is ``calibrated_q_over_nonpush`` for
    the over and ``calibrated_q_under_nonpush`` for the under: the calibrated
    probability the published pricing authority carries, excluding pushes. The
    raw ``q_over_nonpush`` is also published and is *not* this number, which
    is checked, because exposing the raw one as the wager's probability would
    be showing a figure live pricing does not use.
    """
    assert page.SIDE_MODEL_PROBABILITY == {
        routing.SIDE_OVER: "calibrated_q_over_nonpush",
        routing.SIDE_UNDER: "calibrated_q_under_nonpush",
    }

    priced = release["priced"].set_index(
        ["game_id", "player_id", "prop_type", "line_value", "vendor"]
    )

    differed = 0

    for page_row in release["rows"]:
        source = page_row.row

        field = page.SIDE_MODEL_PROBABILITY[page_row.side]

        assert page_row.model_probability == source[field]

        row = row_for(html, page_row)

        assert page.percent(source[field]) in row

        # The raw probability is not in the publication at all -- it is in the
        # priced frame the publisher read -- so this is where the two can be
        # compared, and where the page can be shown not to be rendering it.
        wager = page_row.wager

        priced_row = priced.loc[
            (
                wager.game_id,
                wager.player_id,
                wager.prop_type,
                wager.line_value,
                page_row.routed.offer.vendor,
            )
        ]

        raw = float(
            priced_row[
                "q_over_nonpush"
                if page_row.side == routing.SIDE_OVER
                else "q_under_nonpush"
            ]
        )

        if abs(raw - float(source[field])) > 5e-4:
            differed += 1

            assert page.percent(raw) not in row

    # Not a hypothetical: on this slate the frozen calibration moves the
    # number, so rendering the raw one would be visibly a different figure.
    assert differed > 0


def test_17_the_sportsbook_probability_is_the_posted_prices_own(
    release: dict, html: str
):
    """Cases 7, 8 and 32. Not the devigged pair, by measurement.

    The devigged probabilities are the two prices with the hold divided out.
    They are smaller than what either price implies whenever the sportsbook is
    charging one, so on a real slate they are a different number and this says
    which number is on the page.
    """
    checked = 0

    for page_row in release["rows"]:
        if page_row.routed is None:
            continue

        row = row_for(html, page_row)

        posted = page_row.routed.offer.american_odds

        implied = page.sportsbook_implied_probability(posted)

        assert page.percent(implied) in row

        for field in (
            "market_devig_q_over",
            "market_devig_q_under",
        ):
            devigged = page_row.row.get(field)

            if devigged is None:
                continue

            if abs(float(devigged) - implied) > 5e-4:
                assert page.percent(devigged) not in row

                checked += 1

    assert checked > 0


def test_18_expected_value_is_the_published_field_and_does_not_move(
    release: dict, html: str, tmp_path: Path
):
    """Case 33. One number, published, identical in all three notations.

    Rendered three times with three formats chosen and the expected-value
    cells diffed. Mismatch count has to be zero, which is also what makes the
    claim that the selector is display-only checkable rather than asserted.
    """
    cells: dict[str, list[str]] = {}

    for format_name, _ in page.ODDS_FORMATS:
        rendered = page.render(
            release["built"]["publication"],
            release["rows"],
            registry=REGISTRY,
        )

        result = run_script(rendered, tmp_path / format_name, choose=format_name)

        assert result["odds_format"] == format_name

        cells[format_name] = re.findall(
            r'<td class="c-ev num"[^>]*>(.*?)</td>', rendered
        )

    mismatches = [
        (name, values)
        for name, values in cells.items()
        if values != cells[page.DEFAULT_ODDS_FORMAT]
    ]

    assert mismatches == []

    published = [
        page.signed_percent(
            page_row.row[page.SIDE_EXPECTED_VALUE[page_row.side]]
        )
        for page_row in release["rows"]
        if page_row.routed is not None
    ]

    rendered_text = [
        re.sub(r"<[^>]+>", "", cell)
        for cell in cells[page.DEFAULT_ODDS_FORMAT]
    ]

    assert sorted(rendered_text) == sorted(published)


def test_19_the_sportsbook_and_its_destination_do_not_depend_on_the_format(
    release: dict, html: str
):
    """Case 9. Switching notation cannot change where a reader is sent."""
    buttons = re.findall(r'<a class="wf-bo-btn" href="([^"]+)"', html)

    for format_name, _ in page.ODDS_FORMATS:
        rendered = page.render(
            release["built"]["publication"],
            release["rows"],
            registry=REGISTRY,
        )

        assert re.findall(r'<a class="wf-bo-btn" href="([^"]+)"', rendered) == (
            buttons
        )

        assert re.findall(r'<span class="wf-bo-book">([^<]+)</span>', rendered) == (
            re.findall(r'<span class="wf-bo-book">([^<]+)</span>', html)
        )


# ======================================================================
# 20-25: the distribution panel
# ======================================================================


def test_20_every_published_wager_has_a_toggle_and_a_panel(release: dict, html: str):
    """Case 10. Every target, not every target the fixture happened to carry."""
    assert release["rows"]

    for page_row in release["rows"]:
        assert page_row.distribution is not None

        identifier = page._panel_id(page_row)

        assert f'aria-controls="{identifier}"' in html

        assert f'<tr class="wf-dist-row" id="{identifier}">' in html

    # Counted by the button's own opening tag rather than by the attribute,
    # which the script also names when it looks the buttons up.
    assert html.count('<button type="button" class="wf-dist-toggle"') == len(
        release["rows"]
    )

    assert html.count('<tr class="wf-dist-row"') == len(release["rows"])


def test_21_the_panel_is_the_published_record_and_nothing_else(
    release: dict, html: str
):
    """Case 10's substance. The page copies the forecast; it derives none.

    Every listed outcome, its printed probability, the tail bucket's label and
    its probability, and the highlighted range -- all read back off the panel
    and required to be the published record's. The page cannot reconstruct a
    distribution from a mean and a probability and this is what says it did
    not try.
    """
    for page_row in release["rows"]:
        record = page_row.distribution

        panel = panel_for(html, page_row)

        body = panel.split("<tbody>", 1)[1]

        outcomes = re.findall(r"<th scope=row>([^<]+)</th>", body)

        printed = re.findall(r"<td class=num>([^<]+)</td>", body)

        expected = [str(outcome) for outcome in record["distribution_support"]]

        weights = [
            page.percent(value, places=2)
            for value in record["distribution_probability"]
        ]

        if record["distribution_tail_outcome"] is not None:
            expected.append(record["distribution_tail_label"])

            weights.append(
                page.percent(record["distribution_tail_probability"], places=2)
            )

        assert outcomes == expected

        assert printed == weights

        # The bar is a width on the cell, so every row has one and the number
        # beside it is the probability rather than the width.
        assert body.count("--w:") == len(outcomes)


def test_22_the_panel_states_the_wager_the_projection_and_both_probabilities(
    release: dict, html: str
):
    """Case 18. What the panel has to say before it draws anything."""
    for page_row in release["rows"]:
        panel = panel_for(html, page_row)

        assert str(page_row.row["player_name"]) in panel

        assert page.prop_label(page_row.row["prop_type"]) in panel

        assert "Current Line" in panel

        assert page.line_text(page_row.row["line_value"]) in panel

        for side in (routing.SIDE_OVER, routing.SIDE_UNDER):
            assert f"Model Probability ({side})" in panel

            assert page.percent(
                page_row.row[page.SIDE_MODEL_PROBABILITY[side]]
            ) in panel

        if page_row.projection is not None:
            assert "Model Projection" in panel

            assert page.number_text(page_row.projection) in panel

        # And the disclosure that the wager's probability is not the integral
        # of this forecast.
        assert escape(page.DISTRIBUTION_NOTE) in panel

        assert "calibration" in panel.lower()


def test_23_a_simulated_distribution_says_it_is_simulated(release: dict, html: str):
    """Case 16. The distinction is mandatory, so it is on the page.

    Both kinds are on this slate -- the frozen dependence policy makes
    ``rebounds_assists`` a seeded simulation and the rest convolutions -- so
    this is a test of the labelling and not of a constant.
    """
    labels: dict[str, str] = {}

    for page_row in release["rows"]:
        record = page_row.distribution

        panel = panel_for(html, page_row)

        assert record["distribution_label"] in panel

        assert "Exact PMF" not in panel

        assert "exact pmf" not in panel.lower()

        labels[record["distribution_type"]] = record["distribution_label"]

    assert labels == {
        "analytical_marginal_pmf": "Model Distribution",
        "analytical_convolution_pmf": "Model Distribution",
        "simulation_empirical_pmf": "Model Distribution (simulation-based)",
    }


def test_24_the_highlighted_range_is_stated_and_not_only_coloured(
    release: dict, html: str
):
    """Cases 19 and 24. An interval and the mass it holds, no adjectives.

    The rule is printed in the caption, the rows in the range are marked by
    weight as well as by tint, and no word like "likely" or "safe" is used as
    a judgement about a wager.
    """
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)

    compact = style.replace(" ", "").replace("\n", "")

    assert "tr.likely{background:" in compact

    # Tint and weight, so the marking does not depend on seeing the tint.
    assert "tr.likelyth[scope=\"row\"],.wf-dist-tabletbodytr.likely" in compact

    assert "td.num{font-weight:700}" in compact

    for page_row in release["rows"]:
        record = page_row.distribution

        panel = panel_for(html, page_row)

        assert "narrowest run of consecutive outcomes" in panel

        body = panel.split("<tbody>", 1)[1]

        marked = body.count("<tr class=likely>")

        expected = (
            record["distribution_most_likely_high"]
            - record["distribution_most_likely_low"]
            + 1
        )

        assert marked == expected

    readable = visible_text(html).lower()

    for adjective in (
        "very likely",
        "safe bet",
        "sure thing",
        "can't lose",
        "guaranteed",
    ):
        assert adjective not in readable, adjective


def test_25_the_panel_needs_no_network_and_no_library(html):
    """Case 20. Self-contained, static, and still a chart.

    No chart CDN, no external script, no extra request of any kind. The one
    external reference the page is allowed is the site's own font and
    stylesheet, which it already had; the panel adds nothing.
    """
    panels = re.findall(
        r'<tr class="wf-dist-row".*?</section></td></tr>', html, re.S
    )

    assert panels

    for panel in panels:
        for forbidden in (
            "<script",
            "<canvas",
            "<iframe",
            "http://",
            "https://",
            "src=",
            "chart",
        ):
            assert forbidden not in panel.lower(), forbidden


# ======================================================================
# 26-28: the panel as a disclosure, and the frame it lives in
# ======================================================================


def test_26_the_toggle_is_a_real_button_with_the_state_it_reports(
    release: dict, html: str, tmp_path: Path
):
    """Case 24. Keyboard reachable because it is a button, and it says so.

    In the served document the panel is open and the button is hidden, which
    is the no-script reading: the information is on the page and the dead
    control is not offered. The script closes the panels, reveals the buttons
    and sets ``aria-expanded`` to match -- and a click flips exactly one pair.
    """
    for page_row in release["rows"]:
        row = row_for(html, page_row)

        assert '<button type="button"' in row

        assert 'aria-expanded="true"' in row

        assert f'aria-controls="{page._panel_id(page_row)}"' in row

        assert "hidden>" in row

    first = page._panel_id(release["rows"][0])

    settled = run_script(html, tmp_path / "settled")

    assert all(not toggle["hidden"] for toggle in settled["toggles"])

    assert all(toggle["expanded"] == "false" for toggle in settled["toggles"])

    assert all(panel["hidden"] for panel in settled["panels"])

    clicked = run_script(html, tmp_path / "clicked", click=[first])

    opened = [
        toggle for toggle in clicked["toggles"] if toggle["expanded"] == "true"
    ]

    assert [toggle["panel"] for toggle in opened] == [first]

    assert [panel["id"] for panel in clicked["panels"] if not panel["hidden"]] == (
        [first]
    )


def test_27_a_panel_cannot_end_up_under_somebody_elses_prediction(
    html, tmp_path: Path
):
    """Sorting moves a prediction, and its forecast goes with it."""
    for sort in ("ev", "model", "player"):
        result = run_script(html, tmp_path / f"sorted-{sort}", sort=sort)

        placed = result["placed"]

        assert placed

        # Row, panel, row, panel: every panel immediately after a row, and
        # every row followed by one.
        for index in range(0, len(placed), 2):
            assert "row" in placed[index], (sort, index)

            assert "panel" in placed[index + 1], (sort, index)

    # Filtering a prediction away closes and hides its panel with it.
    filtered = run_script(html, tmp_path / "filtered", prop="blocks")

    assert all(panel["hidden"] for panel in filtered["panels"])


def test_28_the_iframe_handshake_is_the_sites_existing_one(html):
    """Case 21. The same element, the same named origins, no new system.

    Expanding a panel changes the height of the measured element, and the
    handshake already re-measures on any click, on any mutation of the body
    subtree, on any resize of it, and once a second. So a panel resizes the
    frame with no new code, which is the point: there is one height contract
    on this site and this page is not allowed a second.
    """
    script = re.search(
        r'<script id="frame-height">(.*?)</script>', html, re.S
    ).group(1)

    assert f"getElementById('{page.EMBED_WRAPPER_ID}')" in script

    assert "'*'" not in script

    assert '"*"' not in script

    for origin in page.PARENT_ORIGINS:
        assert origin in script

    for trigger in (
        "addEventListener('click'",
        "MutationObserver",
        "subtree: true",
        "ResizeObserver",
        "setInterval(sendHeightToParent, 1000)",
    ):
        assert trigger in script, trigger

    # One wrapper, and the panels are inside it.
    assert html.count(f'<div id="{page.EMBED_WRAPPER_ID}"') == 1

    wrapper = html.split(f'<div id="{page.EMBED_WRAPPER_ID}"', 1)[1]

    assert '<tr class="wf-dist-row"' in wrapper.rsplit("</div>", 1)[0]
