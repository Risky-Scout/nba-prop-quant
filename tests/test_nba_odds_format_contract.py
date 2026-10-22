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


#: One bar of the histogram, read back out of the markup. Matched as a whole
#: element rather than searched for attribute by attribute, so a bar with a
#: missing geometry attribute is a bar this does not find.
BAR = re.compile(
    r'<rect class="(?P<classes>[^"]*)" x="(?P<x>[-\d.]+)" '
    r'y="(?P<y>[-\d.]+)" width="(?P<width>[-\d.]+)" '
    r'height="(?P<height>[-\d.]+)" data-outcome="(?P<outcome>[^"]*)">'
    r"<title>(?P<title>[^<]*)</title></rect>"
)


def chart_for(panel: str) -> str:
    """The histogram inside one panel."""
    found = re.search(r'<svg class="wf-hist".*?</svg>', panel, re.S)

    assert found is not None, "panel has no histogram"

    return found.group(0)


def bars(chart: str) -> list[dict[str, str]]:
    """Every bar of one histogram, in document order."""
    return [match.groupdict() for match in BAR.finditer(chart)]


def chart_scale(chart: str) -> tuple[float, float]:
    """The probability the chart's vertical axis tops out at, and its height."""
    top = re.search(r'data-scale-top="([-\d.e]+)"', chart)

    plot = re.search(r'data-plot-height="(\d+)"', chart)

    assert top is not None and plot is not None

    return float(top.group(1)), float(plot.group(1))


def chart_label(chart: str) -> str:
    """What the chart tells a screen reader it is."""
    found = re.search(r'aria-label="([^"]*)"', chart)

    assert found is not None

    return found.group(1)


def chart_size(chart: str) -> tuple[float, float]:
    """The chart's own pixel width and height."""
    found = re.search(r'width="([\d.]+)" height="([\d.]+)"', chart)

    assert found is not None

    return float(found.group(1)), float(found.group(2))


def published_weights(record) -> list[float]:
    """The record's probabilities, listed outcomes then the tail bucket."""
    weights = [float(value) for value in record["distribution_probability"]]

    if record["distribution_tail_outcome"] is not None:
        weights.append(float(record["distribution_tail_probability"]))

    return weights


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
    its probability, and the highlighted range -- all read back off the
    panel's exact-probabilities table and required to be the published
    record's. The page cannot reconstruct a distribution from a mean and a
    probability and this is what says it did not try.
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
            assert f"Model Probability ({side.capitalize()})" in panel

            assert page.percent(
                page_row.row[page.SIDE_MODEL_PROBABILITY[side]]
            ) in panel

        if page_row.projection is not None:
            assert "Model Projection" in panel

            assert page.projection_text(page_row.projection) in panel

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

    The rule is printed in the caption. In the table the rows in the range
    are marked by weight as well as by tint. In the chart the bars in the
    range are outlined as well as filled -- an outline is a shape and not a
    hue -- and the range is named in words under the axis, which is the part
    that does not depend on seeing the chart at all. No word like "likely" or
    "safe" is used as a judgement about a wager.
    """
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)

    compact = style.replace(" ", "").replace("\n", "")

    assert "tr.likely{background:" in compact

    # Tint and weight, so the marking does not depend on seeing the tint.
    assert "tr.likelyth[scope=\"row\"],.wf-dist-tabletbodytr.likely" in compact

    assert "td.num{font-weight:700}" in compact

    # Fill and outline, for the same reason.
    assert ".wf-hist-bar.likely{fill:var(--wf-gold);stroke:" in compact

    assert "stroke-width:" in compact.split(".wf-hist-bar.likely{", 1)[1]

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

        # And said in words, with the published endpoints in it.
        drawn = bars(chart_for(panel))

        low = record["distribution_most_likely_low"]

        high = record["distribution_most_likely_high"]

        named = re.search(
            r'<text class="wf-hist-span-label"[^>]*>Most likely '
            r"([^&]+)&#8211;([^<]+)</text>",
            panel,
        )

        assert named is not None

        assert named.group(1) == drawn[low]["outcome"]

        assert named.group(2) == drawn[high]["outcome"]

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
            "<img",
            "http://",
            "https://",
            "src=",
            "chart.js",
            "chartjs",
            "highcharts",
            "plotly",
            "d3.",
            "echarts",
            "apexcharts",
            "require(",
            "import(",
        ):
            assert forbidden not in panel.lower(), forbidden

        # The chart is markup. Inline SVG and nothing else draws it.
        assert '<svg class="wf-hist"' in panel


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

    inside = wrapper.rsplit("</div>", 1)[0]

    assert '<tr class="wf-dist-row"' in inside

    # Including the exact-probabilities disclosure, whose own expansion is
    # another height change the measured element has to carry.
    assert '<details class="wf-dist-exact">' in inside


# ======================================================================
# 29-39: the forecast drawn as a histogram
# ======================================================================
#
# What a bettor was shown before this section existed was one table row per
# outcome with a bar whose *width* was the probability. That reads as a league
# table rather than as a distribution, and for a points line it was a panel
# taller than the screen it was on. The eleven tests below are the
# presentation the correction requires, and each one is about the drawing
# rather than about the numbers: the numbers are tests 20 to 24's subject and
# they have not moved.


def test_29_there_is_one_vertical_bar_per_published_outcome(
    release: dict, html: str
):
    """One bar per listed outcome plus one for the tail, and they stand up.

    Standing up is checked rather than described: every bar in a chart has the
    same width and they differ in height, and every bar's foot is on the same
    baseline with its head above it. A bar chart laid on its side fails all
    three.
    """
    for page_row in release["rows"]:
        record = page_row.distribution

        chart = chart_for(panel_for(html, page_row))

        drawn = bars(chart)

        assert len(drawn) == len(published_weights(record))

        widths = {bar["width"] for bar in drawn}

        assert len(widths) == 1, "bars differ in width"

        feet = {round(float(bar["y"]) + float(bar["height"]), 2) for bar in drawn}

        assert len(feet) == 1, "bars do not share a baseline"

        for bar in drawn:
            assert float(bar["height"]) >= 0.0

        # And the shape has more than one height in it, so "vertical" is not
        # being satisfied by drawing every bar the same.
        assert len({bar["height"] for bar in drawn}) > 1


def test_30_a_bars_height_is_its_published_probability(release: dict, html: str):
    """The one piece of arithmetic in the renderer, checked against the record.

    Height is probability over the axis top times the plot height. Nothing
    else: no floor that would make a one-in-a-thousand outcome look like a
    one-in-fifty, no per-bar rescaling, and no clamping.
    """
    for page_row in release["rows"]:
        record = page_row.distribution

        chart = chart_for(panel_for(html, page_row))

        top, plot = chart_scale(chart)

        drawn = bars(chart)

        weights = published_weights(record)

        assert top in page.CHART_SCALE_TOPS

        assert max(weights) <= top

        for bar, weight in zip(drawn, weights, strict=True):
            assert float(bar["height"]) == pytest.approx(
                round(weight / top * plot, 2), abs=5e-3
            )

        # Ratios between bars are the ratios between probabilities, which is
        # the property a reader actually reads off a chart.
        tallest = max(range(len(weights)), key=weights.__getitem__)

        for index, weight in enumerate(weights):
            if weights[tallest] <= 0.0:
                continue

            assert float(drawn[index]["height"]) / float(
                drawn[tallest]["height"]
            ) == pytest.approx(weight / weights[tallest], abs=2e-3)


def test_31_the_tail_bucket_is_drawn_and_labelled_as_one(release: dict, html: str):
    """The remainder is a bar too, marked as the one that is not a single
    outcome.

    It is the only bar standing for more than one outcome, so it is drawn with
    its own marking and carries the published ``N+`` label -- and its label is
    always printed under the axis, however wide the support is.
    """
    seen = 0

    for page_row in release["rows"]:
        record = page_row.distribution

        if record["distribution_tail_outcome"] is None:
            continue

        seen += 1

        chart = chart_for(panel_for(html, page_row))

        drawn = bars(chart)

        tail = drawn[-1]

        assert "tail" in tail["classes"].split()

        assert tail["outcome"] == record["distribution_tail_label"]

        assert page.percent(
            record["distribution_tail_probability"], places=2
        ) in tail["title"]

        # No other bar is marked as the tail.
        assert [
            bar for bar in drawn if "tail" in bar["classes"].split()
        ] == [tail]

        # And the tail is one of the labels printed under the axis.
        ticks = re.findall(
            r'<text class="wf-hist-tick"[^>]*text-anchor="middle">([^<]*)</text>',
            chart,
        )

        assert record["distribution_tail_label"] in ticks

    assert seen == len(release["rows"])


def test_32_the_horizontal_axis_is_the_support_in_order(release: dict, html: str):
    """Outcomes along the bottom, in the order they were published.

    Left to right, ascending, one slot each, no gaps and no reordering -- and
    the tick labels that are printed are a subset of those outcomes in the
    same order.
    """
    for page_row in release["rows"]:
        record = page_row.distribution

        chart = chart_for(panel_for(html, page_row))

        drawn = bars(chart)

        expected = [
            str(outcome) for outcome in record["distribution_support"]
        ]

        if record["distribution_tail_outcome"] is not None:
            expected.append(record["distribution_tail_label"])

        assert [bar["outcome"] for bar in drawn] == expected

        positions = [float(bar["x"]) for bar in drawn]

        assert positions == sorted(positions)

        gaps = [
            positions[index + 1] - positions[index]
            for index in range(len(positions) - 1)
        ]

        # One slot each, to the precision the markup is written at.
        assert max(gaps) - min(gaps) < 0.02, "outcomes are not evenly spaced"

        assert min(gaps) == pytest.approx(
            page.chart_slot_width(len(drawn)), abs=0.02
        )

        ticks = re.findall(
            r'<text class="wf-hist-tick"[^>]*text-anchor="middle">([^<]*)</text>',
            chart,
        )

        assert ticks

        assert [tick for tick in expected if tick in ticks] == ticks

        # The vertical axis says what it is measuring, in words.
        assert '>Probability</text>' in chart

        assert page.prop_label(page_row.row["prop_type"]) in chart_label(chart)


def test_33_the_posted_line_is_marked_where_it_falls(release: dict, html: str):
    """Case 7. The published line, drawn, and nothing recomputed to draw it."""
    for page_row in release["rows"]:
        record = page_row.distribution

        panel = panel_for(html, page_row)

        chart = chart_for(panel)

        drawn = bars(chart)

        line = page_row.row["line_value"]

        marker = re.search(
            r'<line class="wf-hist-line" x1="([\d.]+)" y1="(\d+)" '
            r'x2="([\d.]+)" y2="(\d+)"/>',
            chart,
        )

        assert marker is not None

        assert marker.group(1) == marker.group(3), "the marker is not vertical"

        expected = page.chart_line_offset(
            page._chart_outcomes(record), line, count=len(drawn)
        )

        assert float(marker.group(1)) == pytest.approx(expected, abs=5e-3)

        assert f"Current Line {page.line_text(line)}" in chart

        # A half-point line separates two outcomes, so it is drawn on the
        # boundary between their two slots rather than through either bar.
        slot = page.chart_slot_width(len(drawn))

        edges = [
            page.CHART_PAD_LEFT + index * slot
            for index in range(len(drawn) + 1)
        ]

        centres = [edge + slot / 2.0 for edge in edges[:-1]]

        # To the precision the marker is written at, which is two decimals.
        on_edge = any(
            abs(float(marker.group(1)) - edge) < 0.01 for edge in edges
        )

        on_centre = any(
            abs(float(marker.group(1)) - centre) < 0.01 for centre in centres
        )

        if float(line).is_integer():
            assert on_centre, (line, marker.group(1))
        else:
            assert on_edge, (line, marker.group(1))


@pytest.mark.parametrize(
    "line,expected_index,on_boundary",
    (
        (14.5, 5, True),
        (14.0, 4, False),
        (10.5, 1, True),
        (10.0, 0, False),
        (20.0, 10, False),
        (19.5, 10, True),
    ),
)
def test_34_the_line_marker_handles_half_points_and_integers(
    line: float, expected_index: int, on_boundary: bool
):
    """The marker's placement rule, on a support stated here.

    ``14.5`` sits on the boundary that separates the ``14`` bar from the ``15``
    bar: everything to its left is an under and everything to its right is an
    over. ``14`` sits on the centre of the ``14`` bar, because that outcome is
    neither side of the line -- it is the push, and a marker on a boundary
    would be claiming it belonged to one of its neighbours.
    """
    outcomes = list(range(10, 21))

    count = len(outcomes)

    slot = page.chart_slot_width(count)

    offset = page.chart_line_offset(outcomes, line, count=count)

    edge = page.CHART_PAD_LEFT + expected_index * slot

    assert offset == pytest.approx(edge if on_boundary else edge + slot / 2.0)


def test_35_the_highlighted_run_is_the_published_pair(release: dict, html: str):
    """Case 8. The bars inside the published range, and no others."""
    for page_row in release["rows"]:
        record = page_row.distribution

        drawn = bars(chart_for(panel_for(html, page_row)))

        marked = [
            index
            for index, bar in enumerate(drawn)
            if "likely" in bar["classes"].split()
        ]

        assert marked == list(
            range(
                int(record["distribution_most_likely_low"]),
                int(record["distribution_most_likely_high"]) + 1,
            )
        )


def test_36_the_exact_probabilities_are_still_in_text(release: dict, html: str):
    """Case 9. The chart is a second reading of a table that is still there.

    Every outcome and its probability, in a real disclosure a keyboard can
    open and a screen reader can read, inside the panel and under the chart.
    Nothing is hidden permanently and nothing is only a rectangle.
    """
    for page_row in release["rows"]:
        record = page_row.distribution

        panel = panel_for(html, page_row)

        assert '<details class="wf-dist-exact">' in panel

        assert "<summary>Exact probabilities</summary>" in panel

        # The chart comes first and the numbers follow it.
        assert panel.index('<svg class="wf-hist"') < panel.index(
            '<details class="wf-dist-exact">'
        )

        exact = panel.split('<details class="wf-dist-exact">', 1)[1]

        printed = re.findall(r"<td class=num>([^<]+)</td>", exact)

        assert printed == [
            page.percent(weight, places=2)
            for weight in published_weights(record)
        ]

        listed = re.findall(r"<th scope=row>([^<]+)</th>", exact)

        assert listed == [
            bar["outcome"] for bar in bars(chart_for(panel))
        ]

        # Not hidden from anything: a disclosure, not a display rule.
        assert "wf-dist-exact{display:none" not in html.replace(" ", "")


def test_37_the_chart_computes_no_probability_of_its_own(release: dict, html: str):
    """Every probability the chart states is one the release published.

    The percentages in a chart are its bar titles and its axis labels, and
    this reads all of them back: a title has to be a published probability at
    two decimals, an axis label has to be a rung of the fixed ladder or its
    midpoint. A renderer that smoothed, renormalised or re-derived anything
    would state a number that is neither.
    """
    for page_row in release["rows"]:
        record = page_row.distribution

        chart = chart_for(panel_for(html, page_row))

        weights = published_weights(record)

        top, _ = chart_scale(chart)

        allowed = {page.percent(weight, places=2) for weight in weights} | {
            page.percent(top * fraction, places=1)
            for fraction in (1.0, 0.5, 0.0)
        }

        stated = set(re.findall(r"[\d.]+%", chart))

        assert stated <= allowed, stated - allowed

        # And every published probability is one of the stated ones, so the
        # chart drops nothing to make itself fit.
        titles = {bar["title"] for bar in bars(chart)}

        for outcome, weight in zip(
            [bar["outcome"] for bar in bars(chart)], weights, strict=True
        ):
            assert (
                f"{outcome} — {page.percent(weight, places=2)}" in titles
            )


def test_38_the_chart_is_the_same_bounded_height_whatever_the_support(
    release: dict, html: str
):
    """Case 5. A points distribution costs the same room as a blocks one.

    The height is a constant of the renderer rather than a function of how
    many outcomes there are, and the constant is inside the band the
    correction names. Width is what grows, and it grows inside a box that
    scrolls rather than pushing the page out.
    """
    heights = set()

    widths = set()

    for page_row in release["rows"]:
        chart = chart_for(panel_for(html, page_row))

        width, height = chart_size(chart)

        heights.add(height)

        widths.add(width)

        assert f'height:{height:.0f}px' in chart

    assert heights == {float(page.CHART_HEIGHT)}

    assert 260 <= page.CHART_HEIGHT <= 320

    # The slate carries supports of several sizes, so the constant height is
    # being observed over a real range rather than over one shape.
    assert len(widths) > 1

    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)

    compact = style.replace(" ", "").replace("\n", "")

    assert "max-height" not in compact.split(".wf-hist{", 1)[1].split("}", 1)[0]


def test_39_a_wide_support_scrolls_inside_the_chart_and_not_the_page(
    release: dict, html: str
):
    """Case 6. The chart gets its own sideways scroll; the page gets none.

    Checked as the three things that make that true: the chart's box is
    capped at the width of whatever contains it and scrolls horizontally, the
    chart itself is not capped and so can be wider than the box, and the
    panel is still a block on a narrow screen rather than a cell in the grid
    the other rows reflow into.
    """
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)

    compact = style.replace(" ", "").replace("\n", "")

    scroll = compact.split(".wf-hist-scroll{", 1)[1].split("}", 1)[0]

    assert "max-width:100%" in scroll

    assert "overflow-x:auto" in scroll

    assert "max-width:none" in compact.split(".wf-hist{", 1)[1].split("}", 1)[0]

    narrow = compact.split("@media(max-width:720px)", 1)[1]

    assert ".wf-tabletbodytr.wf-dist-row{display:block" in narrow

    # And the table it sits in is laid out from its own width rather than
    # from its contents. Without this the box is capped at 100% of a column
    # that grew to fit the chart, the cap buys nothing, and a 760px chart
    # drags the whole document sideways. Measured in a browser at 420px:
    # the document's scrollWidth is the viewport's with this rule and 797
    # without it.
    assert "table-layout:fixed" in narrow.split(".wf-table{", 1)[1].split(
        "}", 1
    )[0]

    # The widest chart on this slate is wider than a phone, which is what
    # makes the scroll box load-bearing rather than decorative.
    widest = max(
        chart_size(chart_for(panel_for(html, page_row)))[0]
        for page_row in release["rows"]
    )

    assert widest > 420

    # And no chart is wide because of empty space: the width is the support.
    for page_row in release["rows"]:
        chart = chart_for(panel_for(html, page_row))

        width, _ = chart_size(chart)

        assert width == pytest.approx(
            page.chart_width(len(bars(chart))), abs=0.5
        )


# ======================================================================
# 40-41: the projection, and whose name is on the row
# ======================================================================


@pytest.mark.parametrize(
    "value,shown",
    (
        (24.368217392, "24.37"),
        (8.2, "8.20"),
        # 1.005 is not a half in binary -- it is stored as very slightly less
        # than one and five thousandths -- so Python's own formatting, which
        # is what the page uses everywhere, rounds it down. Recorded as the
        # project's convention rather than corrected: the alternative is a
        # second rounding rule that disagrees with every other number shown.
        (1.005, "1.00"),
        (1.015, "1.01"),
        (0.0, "0.00"),
        (0.455388890, "0.46"),
        (1.049282, "1.05"),
    ),
)
def test_40_a_projection_is_shown_at_two_decimals(value: float, shown: str):
    """Case 1's vectors, and the untouched value behind them."""
    assert page.projection_text(value) == shown

    assert page.projection_text(value) == f"{value:.2f}"

    # Display only: formatting the value does not change the value, and the
    # full-precision number is still what everything else is a function of.
    assert float(value) == value


def test_41_the_projection_column_is_the_published_field_at_two_decimals(
    release: dict, html: str
):
    """Case 1. Both places a bettor meets a projection, from one source."""
    seen = 0

    for page_row in release["rows"]:
        published = page_row.row["model_projection"]

        if published is None:
            # A row the pricing path published no projection for, which the
            # panel says nothing about rather than inventing a figure for.
            assert "<dt>Model Projection</dt>" not in panel_for(html, page_row)

            continue

        seen += 1

        shown = page.projection_text(published)

        assert shown == f"{float(published):.2f}"

        assert (
            f'<td class="c-proj num" data-label="Model Projection">{shown}</td>'
            in row_for(html, page_row)
        )

        panel = panel_for(html, page_row)

        assert f"<dt>Model Projection</dt><dd>{shown}</dd>" in panel

        # The release still carries the number at full precision, so nothing
        # downstream of the page inherits the rounding.
        source = next(
            row
            for row in release["document"]["rows"]
            if (
                row["game_id"],
                row["player_id"],
                row["prop_type"],
                row["line_value"],
                row["vendor"],
            )
            == (
                page_row.row["game_id"],
                page_row.row["player_id"],
                page_row.row["prop_type"],
                page_row.row["line_value"],
                page_row.row["vendor"],
            )
        )

        assert source["model_projection"] == published

        assert float(source["model_projection"]) == float(published)

    assert seen > 0


def test_42_the_player_column_is_the_published_name_with_no_fallback(
    release: dict, html: str
):
    """Case 2. The name a reader sees is the published ``player_name``.

    Read off the cell rather than searched for in the row, and required to be
    the published field exactly. Then the renderer's own source is read: the
    player cell is built from that field and from nothing else, and in
    particular there is no ``or player_id`` behind it -- a fallback to an
    identifier is how ``Player 12345`` reaches a consumer page.
    """
    import inspect

    for page_row in release["rows"]:
        name = str(page_row.row["player_name"])

        assert name.strip()

        cell = re.search(
            r'<td class="c-player"[^>]*>(.*?)</td>', row_for(html, page_row), re.S
        )

        assert cell is not None

        # The team abbreviation rides along in its own span; the name is the
        # text before it.
        assert cell.group(1).startswith(escape(name))

        assert str(page_row.row["player_id"]) not in cell.group(1)

        assert f"<h3>{escape(name)} — " in panel_for(html, page_row)

    source = inspect.getsource(page._cell)

    player = source.split('if key == "player":', 1)[1].split(
        'if key == "matchup":', 1
    )[0]

    assert 'row.get("player_name")' in player

    assert "player_id" not in player

    for invented in ("first_name", "last_name", "Player ", "format(", "%s"):
        assert invented not in player, invented
