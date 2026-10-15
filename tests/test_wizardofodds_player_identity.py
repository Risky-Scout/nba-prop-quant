"""Who the published prediction is about, from the roster to the screen.

The publisher has carried a ``player_name`` field since it was written, and
the page has rendered it since it was written, and almost nothing proved the
two were connected to a real identity: the fixtures were named after their own
player ids, so a page that rendered the id instead of the name would have
produced the same bytes and passed. These tests fix that, in both senses.

Three groups.

Lineage, on the real path. One active-player record states a first and a last
name; the prediction step joins it, the pricing step merges it forward, the
publisher publishes it and the page renders it. Each hop is compared against
the hop before it, and the roster's own answer is stated here rather than read
out of the prediction script, so the two agreeing is a finding.

Fail-closed, on the publishing side. A name that is missing, null, blank or an
identifier wearing a person's slot is refused -- the whole publication, not
the row, because a publisher that dropped it would serve a thinner feed and
say nothing. The page refuses the same thing again, because it reads a release
it did not write and its Player cell has no fallback to catch it.

And non-regression, because this branch also corrected the page's own copy
about tip-off times and must not have moved anything else: the projection, the
matchup, the tip time, the sort controls, the routing or the embed contract.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import frozen_bundle_slate_fixture as slate_fixture  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_fallback_destinations as fallback_destinations  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
from test_real_slate_serving_root import installed_bundle  # noqa: E402
from test_wizardofodds_canary_end_to_end import (  # noqa: E402
    checkout_sha,
    serve_and_provenance,
)
from test_wizardofodds_player_props_page import REGISTRY  # noqa: E402

from nba_prop_quant.pricing import COMBO_COMPONENTS  # noqa: E402

#: The fallbacks a page must never invent, spelled the way a reader would see
#: them. Named here so the refusal tests and the rendering test argue about the
#: same list.
FORBIDDEN_RENDERINGS: tuple[str, ...] = (
    "Player 12345",
    "P12345",
    "player_id 12345",
    "player id 12345",
    "Unknown Player",
    "Unknown",
    "12345",
)


@pytest.fixture(scope="module")
def frozen_bundle() -> Path:
    bundle = installed_bundle()

    if bundle is None:
        pytest.skip(
            "no installed frozen runtime bundle; set "
            "NBA_PROP_TEST_FROZEN_BUNDLE_ROOT or NBA_PROP_WORK_DIR to a root "
            "ops/install_frozen_model_artifacts.py has installed into"
        )

    return bundle


@pytest.fixture(scope="module")
def served(frozen_bundle: Path, tmp_path_factory) -> dict:
    """The fixture slate through the real scripts, from the roster forward."""
    root = tmp_path_factory.mktemp("identity_served") / "run"

    serve_and_provenance(frozen_bundle, root, production_sha=checkout_sha())

    data_root = root / "data"

    projections = pd.read_parquet(
        data_root
        / "processed"
        / "projections"
        / f"{slate_fixture.SLATE_DATE}.parquet"
    )

    priced = pd.read_parquet(
        publisher.priced_markets_path(data_root, slate_fixture.SLATE_DATE)
    )

    # An empty frame would make every per-row assertion below vacuously true,
    # which is the one way a lineage test can pass while proving nothing.
    assert len(projections) == len(slate_fixture.ROSTER) > 0
    assert len(priced) == len(slate_fixture.QUOTED_PROPS) > 0

    return {
        "data_root": data_root,
        "priced": priced,
        "projections": projections,
        "root": root,
    }


@pytest.fixture(scope="module")
def release(served: dict, tmp_path_factory) -> dict:
    """That served slate, published to a canary and read back."""
    destination = tmp_path_factory.mktemp("identity_canary")

    receipt = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=slate_fixture.SLATE_DATE,
        data_root=served["data_root"],
        project_root=PROJECT,
        destination_root=destination,
    )

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED

    document = json.loads(
        (destination / receipt["release"]["rows_relative"]).read_text(
            encoding="utf-8"
        )
    )

    assert len(document["rows"]) == len(served["priced"]) > 0

    return {"destination": destination, "document": document, "receipt": receipt}


@pytest.fixture(scope="module")
def rendered(release: dict) -> dict:
    """The consumer page built from that release."""
    built = page.build_page(
        publisher.LocalDirectoryTransport(release["destination"]),
        registry=REGISTRY,
    )

    assert len(built["rows"]) > 0

    return built


@pytest.fixture(scope="module")
def rendered_html(rendered: dict) -> str:
    return rendered["files"][page.PAGE_NAME].decode("utf-8")


def published_switch() -> publisher.PublishingSwitch:
    return publisher.read_publishing_switch(
        publisher.MODE_CANARY, project_root=PROJECT
    )


# ======================================================================
# 1-5: the lineage, hop by hop
# ======================================================================


def test_1_the_roster_first_and_last_name_become_the_canonical_full_name(
    served: dict,
):
    """The active-player record's two fields, joined, and nothing else.

    The roster is the authority. The prediction step is the only place the
    public name is assembled, and it assembles it from ``first_name`` and
    ``last_name`` of the active-player record for that id -- so the name this
    test expects is stated from the fixture's roster rather than read back out
    of the artifact it is checking.
    """
    projections = served["projections"].set_index("player_id")

    assert len(projections) == len(slate_fixture.PLAYER_NAMES) > 0

    for player_id, (first, last) in slate_fixture.PLAYER_NAMES.items():
        row = projections.loc[player_id]

        assert row["first_name"] == first
        assert row["last_name"] == last

        assert row["player_name"] == f"{first} {last}"

        assert row["player_name"] == slate_fixture.full_name(player_id)


def test_2_the_projection_artifact_carries_the_exact_name(served: dict):
    """No truncation, no case change, no id anywhere in it."""
    for _, row in served["projections"].iterrows():
        expected = slate_fixture.full_name(row["player_id"])

        assert row["player_name"] == expected

        assert str(row["player_id"]) not in row["player_name"]


def test_3_pricing_preserves_the_name_the_projection_published(served: dict):
    """The priced row's name is the projection's, not the sportsbook's.

    Worth checking rather than assuming: pricing joins the model's slate to a
    vendor's market frame, and a vendor that shipped its own spelling of a
    player would be the obvious way for sportsbook text to become the public
    identity. It cannot be here -- ``normalize_props`` keeps no name at all --
    and this asserts the consequence rather than the cause.
    """
    by_player = served["projections"].set_index("player_id")["player_name"]

    for _, row in served["priced"].iterrows():
        assert row["player_name"] == by_player.loc[row["player_id"]]

    assert "player_name_market" not in served["priced"].columns


def test_4_publication_preserves_the_name_the_priced_row_carried(
    served: dict, release: dict
):
    priced = served["priced"].set_index(
        ["vendor", "game_id", "player_id", "prop_type", "line_value"]
    )

    for row in release["document"]["rows"]:
        key = (
            row["vendor"],
            row["game_id"],
            row["player_id"],
            row["prop_type"],
            row["line_value"],
        )

        assert row["player_name"] == priced.loc[key, "player_name"]

        assert row["player_name"] == slate_fixture.full_name(row["player_id"])


def test_5_the_page_renders_the_published_name_verbatim(
    rendered: dict, rendered_html: str
):
    """In the cell, in the search index and in the sort key.

    Three places, because the name is not only displayed: the page filters and
    orders by it, and a row whose visible name and sortable name disagreed
    would reorder under the reader's hands.
    """
    cells = re.findall(r'<td class="c-player"[^>]*>(.*?)</td>', rendered_html, re.S)

    assert len(cells) == len(rendered["rows"]) > 0

    for page_row, cell in zip(rendered["rows"], cells):
        published = page_row.row["player_name"]

        assert published == slate_fixture.full_name(page_row.row["player_id"])

        # The cell is the name followed by the player's own side, so the name
        # has to lead it rather than merely appear in it.
        assert cell.startswith(published)

        assert published.lower() in page._search_text(page_row)

        assert published in page_row.stable_order

    sortnames = re.findall(r'data-sortname="([^"]*)"', rendered_html)

    assert sorted(sortnames) == sorted(
        page_row.row["player_name"] for page_row in rendered["rows"]
    )


# ======================================================================
# 6-9: the fail-closed contract
# ======================================================================


def publish_one(frame: pd.DataFrame) -> list[dict]:
    return publisher.published_rows(
        frame, switch=published_switch(), slate_date=slate_fixture.SLATE_DATE
    )


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("null", None),
        ("not-a-number", float("nan")),
        ("blank", ""),
        ("whitespace-only", "   "),
        ("tab-only", "\t"),
    ],
)
def test_6_and_7_a_nameless_row_is_refused_rather_than_published(
    served: dict, label: str, value
):
    """One bad name refuses the publication, not just its own row.

    Dropping it would publish a feed one row short and report success, which
    is the failure the fail-closed rules exist to make loud. The contract's
    own wording, and the behaviour here, is that nothing is published.
    """
    frame = served["priced"].copy()

    frame.loc[frame.index[0], "player_name"] = value

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish_one(frame)

    assert "player_name" in str(refusal.value), label

    # And the good rows are not quietly published alongside the refusal.
    assert "none is published" in str(refusal.value)


def test_7_a_missing_column_is_refused_before_any_row_is_read(served: dict):
    frame = served["priced"].drop(columns=["player_name"])

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish_one(frame)

    assert "player_name" in str(refusal.value)


@pytest.mark.parametrize("name", FORBIDDEN_RENDERINGS)
def test_8_an_identifier_wearing_a_persons_slot_is_refused(
    served: dict, name: str
):
    """The four shapes the page must never show, refused at the source.

    None of these can come out of a roster that answered: the name is built
    from two fields of the active-player record, so an identifier in that slot
    means the join found nobody and something filled the gap. That is a
    source-data problem, and the contract's answer to a source-data problem is
    to refuse rather than to render it.
    """
    frame = served["priced"].copy()

    frame.loc[frame.index[0], "player_name"] = name

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish_one(frame)

    assert "identifier rather than a person" in str(refusal.value)


def test_8_a_real_name_that_merely_contains_digits_is_not_refused():
    """The rule is about shape, not about the presence of a digit.

    A refusal that fired on any digit would be a refusal nobody could ship
    behind, so it is pinned from the other side too.
    """
    for name in ("Alpha Example", "Walt Frazier III", "D'Angelo Example-Jones"):
        assert publisher.player_name_problem(name) is None


def test_9_the_page_refuses_a_release_that_names_a_row_by_its_identifier(
    release: dict, tmp_path: Path
):
    """The renderer's own boundary, not a restatement of the publisher's.

    The page reads a release it did not write, and its Player cell renders the
    published string with no fallback -- so a release that lost a name would
    produce a row about nobody rather than an error. Both halves are checked:
    the refusal, and the absence of the fallback that would have hidden it.
    """
    source = release["destination"]

    for forbidden in ("", "   ", "Player 12345", None):
        destination = tmp_path / f"rewritten_{abs(hash(str(forbidden)))}"

        copied = rewrite_published_name(source, destination, forbidden)

        assert copied > 0

        with pytest.raises(page.PageRefused) as refusal:
            page.build_page(
                publisher.LocalDirectoryTransport(destination), registry=REGISTRY
            )

        assert "player_name" in str(refusal.value)

    # And the cell that renders the name has no branch that could invent one:
    # it reads the published field and stops, so the refusal above is the only
    # thing standing between a nameless release and a nameless row.
    source = inspect.getsource(page._cell)

    player_branch = source.split('if key == "player":', 1)[1].split("if key ==", 1)[
        0
    ]

    assert "player_name" in player_branch

    assert "player_id" not in player_branch

    for forbidden in ("Unknown", "Player ", "P{"):
        assert forbidden not in player_branch, forbidden


def test_9_no_rendered_cell_falls_back_to_an_identifier(rendered_html: str):
    """Nothing on the page reads as a row number standing in for a person."""
    for forbidden in FORBIDDEN_RENDERINGS:
        assert forbidden not in rendered_html

    cells = re.findall(r'<td class="c-player"[^>]*>(.*?)</td>', rendered_html, re.S)

    for cell in cells:
        name = re.sub(r"<[^>]*>", "", cell).strip()

        assert publisher.player_name_problem(name) is None, name

        # The side badge is the only other thing in this cell, and it is three
        # letters, so nothing here should look like an id at all.
        assert not re.search(r"\b\d{3,}\b", name), name


def rewrite_published_name(source: Path, destination: Path, value) -> int:
    """Copy a release, replacing the first row's published name.

    The release is copied rather than edited in place so the module-scoped
    fixtures above stay the thing every other test in this file reads.
    """
    shutil.copytree(source, destination)

    current = json.loads(
        (destination / publisher.CURRENT_POINTER).read_text(encoding="utf-8")
    )

    rows_path = destination / current["rows_relative"]

    document = json.loads(rows_path.read_text(encoding="utf-8"))

    if value is None:
        document["rows"][0].pop("player_name", None)
    else:
        document["rows"][0]["player_name"] = value

    rows_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # The manifest records the rows digest, and the page checks it, so a
    # rewritten release has to re-record it or it would be refused for the
    # wrong reason.
    manifest_path = destination / current["manifest_relative"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()

    manifest["rows_sha256"] = digest

    current["rows_sha256"] = digest

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    (destination / publisher.CURRENT_POINTER).write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return len(document["rows"])


# ======================================================================
# 10-12: what this branch must not have moved
# ======================================================================


def test_10_the_model_projection_is_still_the_published_one(
    rendered: dict, release: dict
):
    by_key = {
        (
            row["vendor"],
            row["game_id"],
            row["player_id"],
            row["prop_type"],
            row["line_value"],
        ): row
        for row in release["document"]["rows"]
    }

    assert len(by_key) == len(release["document"]["rows"]) > 0

    for page_row in rendered["rows"]:
        source = by_key[
            (
                page_row.row["vendor"],
                page_row.row["game_id"],
                page_row.row["player_id"],
                page_row.row["prop_type"],
                page_row.row["line_value"],
            )
        ]

        assert page_row.projection == source["model_projection"]

        assert page_row.projection != source["line_value"]


def test_11_the_matchup_is_still_the_two_published_sides(
    rendered: dict, rendered_html: str
):
    for page_row in rendered["rows"]:
        away = page_row.row["away_team"]
        home = page_row.row["home_team"]

        assert page_row.matchup == f"{away} @ {home}"

    assert not re.search(r"Team \d+|team_id|opponent_id|game_id", rendered_html)


def test_12_the_tip_time_is_still_published_and_still_shown(
    rendered: dict, rendered_html: str
):
    for page_row in rendered["rows"]:
        assert page_row.tip_off == slate_fixture.TIP_OFF

    assert 'class="wf-tip"' in rendered_html

    assert "23:00 UTC" in rendered_html


# ======================================================================
# 13-14: the copy this branch corrected, and the control it kept absent
# ======================================================================


def test_13_the_page_no_longer_says_tip_off_times_are_unpublished(
    rendered_html: str,
):
    """The two claims the publication contract made false, gone.

    Both were true when they were written and neither is now: the release
    publishes the tip-off and both team abbreviations, and the page shows all
    three. A page that kept saying otherwise would be contradicted by its own
    table, which is worse than saying nothing.
    """
    stale = (
        "Tip-off times are not in the published release",
        "they are not shown",
        "Team names are not published either",
        "no matchup is shown rather than one assembled",
    )

    for claim in stale:
        assert claim not in rendered_html, claim

    renderer = (PROJECT / "ops" / "wizardofodds_player_props_page.py").read_text(
        encoding="utf-8"
    )

    for claim in stale:
        assert claim not in renderer, claim

    assert "Tip-off timestamps are not in the published feed" not in renderer


def test_14_there_is_still_no_game_time_sort(rendered_html: str):
    """The control stays absent. Only the reason given for it changed."""
    assert "game time" not in rendered_html.lower()

    assert 'value="game"' not in rendered_html

    assert "there is no ordering by start time" in rendered_html

    sorts = re.search(r'<select id="f-sort">(.*?)</select>', rendered_html, re.S)

    assert sorts is not None

    assert re.findall(r'value="([^"]+)"', sorts.group(1)) == ["ev", "win", "player"]


# ======================================================================
# 15-17: routing and the embed, untouched
# ======================================================================


def test_15_affiliate_routing_is_unchanged(rendered: dict, rendered_html: str):
    """Price first, one affiliate tier, and the same markup on every link."""
    for page_row in rendered["rows"]:
        routed = page_row.routed

        if routed.offer is None:
            continue

        if REGISTRY.book_for(routed.offer.vendor) is not None:
            assert routed.tier == routing.TIER_AFFILIATE

    for href, rel in re.findall(
        r'<a[^>]*href="([^"]*)"[^>]*rel="([^"]*)"', rendered_html
    ):
        if "go.php" in href:
            assert rel == "sponsored nofollow noopener"
        elif href.startswith("https://"):
            assert "nofollow noopener" in rel


def test_16_fallback_destinations_are_unchanged():
    """Read from the same registry file, with the same books in it."""
    destinations = fallback_destinations.FallbackDestinations.load(
        project_root=PROJECT
    )

    assert destinations.source.endswith("wizardofodds_fallback_destinations.json")

    assert destinations.linkable_books()

    for book, url in destinations.linkable_books().items():
        assert url.startswith("https://"), book


def test_17_the_iframe_contract_is_unchanged(rendered: dict, rendered_html: str):
    """One wrapper, one message key, two named parents, no wildcard."""
    assert rendered_html.count('id="app-root"') == 1

    assert '<script id="frame-height">' in rendered_html

    assert "frameHeight" in rendered_html

    assert "https://wizardofodds.com" in rendered_html

    assert "https://dev.wizardofodds.com" in rendered_html

    assert "postMessage" in rendered_html

    assert "'*'" not in rendered_html

    manifest = json.loads(
        rendered["files"][page.PAGE_MANIFEST_NAME].decode("utf-8")
    )

    assert manifest["embed"] == {
        "message_key": "frameHeight",
        "parent_origins": [
            "https://wizardofodds.com",
            "https://dev.wizardofodds.com",
        ],
        "wrapper_id": "app-root",
    }


# ======================================================================
# combo props: observed, not changed
# ======================================================================


def test_combo_props_still_publish_a_null_projection(served: dict):
    """Reported rather than corrected, which is this task's instruction.

    A combo line is priced from a joint distribution over its components and
    no central forecast of the combined total is produced anywhere on this
    path, so the publication carries null and the page leaves the cell empty.
    The fixture quotes no combo prop, so this records that fact instead of
    asserting over an empty selection.
    """
    combo_types = sorted(COMBO_COMPONENTS)

    assert combo_types

    priced = served["priced"]

    combo_rows = priced[priced["prop_type"].isin(combo_types)]

    if combo_rows.empty:
        assert not any(
            prop in combo_types for _, prop, _ in slate_fixture.QUOTED_PROPS
        )

        return

    assert combo_rows["model_projection"].isna().all()
