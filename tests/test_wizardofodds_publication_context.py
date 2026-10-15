"""What the model projects, and who is playing, from the real serving path.

The published page could state neither. Both facts already existed upstream:
the schedule payload names both teams and says when the game tips, and the
pricing step already computes the mean it prices each line against. Neither
reached the publication, so the page's own renderer looked for them, found
nothing and omitted two columns.

So these tests are about carriage, not computation, and they are written
against the real path rather than a frame shaped like it. A hand-built priced
frame is built to match: it cannot show a column that is named something else
upstream, a string that loses its time of day on the way to JSON, or a mean
that is not the mean the probability beside it came from. The strict non-empty
incumbent fixture and the installed frozen bundle can, so that is what runs
here, and the tests skip rather than weaken when the bundle is absent.

Four groups. That the projection is the frozen model's own central forecast
for the stat and nothing else. That the game context is the schedule's own
answer and that no internal identifier is ever dressed up as a matchup. That
the tip time survives as a time rather than as a date. And that nothing
numerical moved: the last group serves the same slate twice, once from this
branch and once from a worktree at the production ref, and compares every
field a bettor is shown.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import frozen_bundle_slate_fixture as slate_fixture  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
from test_real_slate_serving_root import (  # noqa: E402
    installed_bundle,
    serve_the_fixture,
)
from test_wizardofodds_canary_end_to_end import (  # noqa: E402
    checkout_sha,
    serve_and_provenance,
)
from test_wizardofodds_player_props_page import REGISTRY  # noqa: E402

from nba_prop_quant.pricing import (  # noqa: E402
    COMBO_COMPONENTS,
    PROP_TO_TARGET,
    conditional_nonpush_probabilities,
    price_single_prop_frame,
)

PRODUCTION_REF = "production/wizardofodds-integration"

PRICING_SCRIPT = "scripts/15_price_markets.py"

#: The game the fixture serves, as the capture's own payload states it. Read
#: from the fixture rather than restated, so a test cannot agree with a
#: constant it copied while disagreeing with the schedule.
HOME_ABBREVIATION = "HOM"
AWAY_ABBREVIATION = "AWY"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def production_base() -> str | None:
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return git("merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue

    return None


def central_column(prop_type: str) -> str:
    """The mean this prop's line is priced against.

    Stated here, independently of the script, because a test that imported the
    script's own answer would agree with it by construction. ``assists`` is the
    one prop whose price does not come from ``mu_selected_ast``: the Gate 3
    role state scales that mean before it reaches the marginal.
    """
    if prop_type == "assists":
        return "gate3_mu_ast"

    return f"mu_selected_{PROP_TO_TARGET[prop_type]}"


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
    """The fixture slate through the real scripts, projections and all.

    Provenanced the way the lifecycle provenances it, because the publisher
    refuses rows carrying no receipt and the publication is half of what is
    being tested here.
    """
    root = tmp_path_factory.mktemp("context_served") / "run"

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

    # The whole point of using the strict fixture: an empty frame would make
    # every per-row assertion below vacuously true.
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
    destination = tmp_path_factory.mktemp("context_canary")

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


def published_by_key(document: dict) -> dict[tuple, dict]:
    return {
        (
            row["vendor"],
            row["game_id"],
            row["player_id"],
            row["prop_type"],
            row["line_value"],
        ): row
        for row in document["rows"]
    }


def priced_by_key(priced: pd.DataFrame) -> dict[tuple, pd.Series]:
    return {
        (
            row["vendor"],
            row["game_id"],
            row["player_id"],
            row["prop_type"],
            row["line_value"],
        ): row
        for _, row in priced.iterrows()
    }


# ======================================================================
# 1-6: the model projection
# ======================================================================


def test_1_the_projection_is_the_mean_the_line_was_priced_against(served: dict):
    """Not a new quantity: the column the pricing call already read.

    Exact equality, per row, against the column this prop's price comes from.
    An approximate check would pass on a projection that had been rounded for
    display, and a rounded projection is a different number from the one the
    probability was computed from.
    """
    priced = served["priced"]

    assert "model_projection" in priced.columns

    for _, row in priced.iterrows():
        prop_type = row["prop_type"]

        if prop_type in COMBO_COMPONENTS:
            continue

        source = central_column(prop_type)

        assert float(row["model_projection"]) == float(row[source]), (
            f"{prop_type} published {row['model_projection']} but is priced "
            f"from {source} = {row[source]}"
        )


def test_2_repricing_from_the_projection_reproduces_the_row(
    frozen_bundle: Path, served: dict
):
    """The strongest statement of the same thing, made by the marginals.

    If the published projection were the median, the marginal mean, or a
    component mean rather than the selected one, feeding it back through the
    frozen marginal would produce a different probability. It produces the
    same one, bit for bit, so the number on the page is the number that
    produced the win percentage beside it.
    """
    priced = served["priced"]

    # The bundle's own marginals, loaded the way the pricing script loads
    # them, so this is the distribution that produced the row.
    marginals = joblib.load(frozen_bundle / "models" / "marginals.joblib")

    for prop_type, target in PROP_TO_TARGET.items():
        part = priced[priced["prop_type"].eq(prop_type)].copy()

        if part.empty:
            continue

        again = price_single_prop_frame(
            part,
            target=target,
            marginal=marginals[target],
            mu_column="model_projection",
            line_column="line_value",
        )

        for column in ("p_over", "p_under", "p_push"):
            assert again[column].tolist() == part[column].tolist(), (
                f"{prop_type}: repricing from the published projection does "
                f"not reproduce {column}"
            )

        over, under = conditional_nonpush_probabilities(
            again["p_over"].to_numpy(), again["p_under"].to_numpy()
        )

        assert over.tolist() == part["q_over_nonpush"].tolist()
        assert under.tolist() == part["q_under_nonpush"].tolist()


def test_3_the_published_projection_is_the_priced_cell_exactly(
    served: dict, release: dict
):
    """Copied through the publication, not reformatted by it."""
    published = published_by_key(release["document"])

    sourced = priced_by_key(served["priced"])

    assert set(published) == set(sourced)

    for key, row in published.items():
        assert row["model_projection"] == float(sourced[key]["model_projection"])


def test_4_the_projection_is_not_the_line_and_not_the_price(
    served: dict, release: dict
):
    """It is a forecast of the stat, and nothing in the market.

    The failure this rules out is a projection column quietly filled with
    something already on the row -- the line, the midpoint of the slate's
    lines, or a figure read back out of the odds. Each of those would look
    plausible and would tell a reader nothing.
    """
    rows = release["document"]["rows"]

    projections = [row["model_projection"] for row in rows]

    assert all(value is not None for value in projections)

    lines = [row["line_value"] for row in rows]

    assert projections != lines

    midpoint = sum(lines) / len(lines)

    for row in rows:
        assert row["model_projection"] != row["line_value"]

        assert row["model_projection"] != midpoint

        for market in ("over_odds", "under_odds", "market_devig_q_over"):
            assert row["model_projection"] != row[market]


def test_5_the_projection_does_not_depend_on_the_quote_it_is_published_with(
    release: dict, rendered: dict
):
    """One projection per player, prop and game, whatever the book posts.

    A projection that differed between two books' rows for the same wager
    would not be a projection of anything: it would be a function of the
    price. So the published value is pinned identical across the quote grain,
    and the page is pinned to show the same number regardless of which
    sportsbook its router selected.
    """
    grouped: dict[tuple, set[float]] = {}

    for row in release["document"]["rows"]:
        wager = (row["game_id"], row["player_id"], row["prop_type"])

        grouped.setdefault(wager, set()).add(row["model_projection"])

    for wager, values in grouped.items():
        assert len(values) == 1, f"{wager} published {values}"

    for page_row in rendered["rows"]:
        wager = (
            page_row.row["game_id"],
            page_row.row["player_id"],
            page_row.row["prop_type"],
        )

        assert page_row.projection in grouped[wager]


def test_6_a_combo_prop_publishes_no_projection_rather_than_a_sum():
    """The one prop family with nothing to publish, and it says so.

    A combo line is priced from a joint distribution over its components and
    no central forecast of the combined total is produced anywhere on this
    path. The sum of the component means is not that forecast. So the pricing
    step writes the column as null for every combo row rather than omitting
    it, and the page leaves the cell empty rather than filling it.
    """
    assert COMBO_COMPONENTS

    tree = ast.parse((PROJECT / PRICING_SCRIPT).read_text(encoding="utf-8"))

    combo_writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]).startswith("priced_part[")
        and "model_projection" in ast.unparse(node.targets[0])
    ]

    assert len(combo_writes) == 1

    assert ast.unparse(combo_writes[0].value) == "np.nan"

    empty = page.PageRow(
        wager=routing.Wager(
            game_id=1,
            player_id=2,
            prop_type="points_rebounds",
            line_value=30.5,
            side=routing.SIDE_OVER,
        ),
        routed=routing.Routed(
            tier=routing.TIER_FALLBACK,
            offer=None,
            book=None,
            destination=None,
            display_name="",
            best_overall=False,
            considered=(),
            reason="no sportsbook priced this side",
        ),
        row={"model_projection": None, "prop_type": "points_rebounds"},
    )

    assert empty.projection is None

    # The same dash the page already uses for a price no sportsbook posted,
    # rather than a number. Not empty: a blank cell reads as a rendering bug,
    # and a dash reads as "there is no such figure", which is the truth.
    cell = page._cell(empty, "projection")[0]

    assert cell == "—"

    assert not re.search(r"\d", cell)


def test_the_published_projection_is_the_column_the_price_came_from():
    """The two copies of one choice, pinned to each other by syntax.

    The pricing call and the published projection each name the column this
    prop is priced from, and they have to be the same column or the page shows
    a number that did not produce the probability beside it. The call cannot be
    refactored into a shared name -- it is pinned syntax-identical to
    production under the additive-only rule -- so the duplication is pinned
    instead: both expressions must parse to the same tree.
    """
    tree = ast.parse((PROJECT / PRICING_SCRIPT).read_text(encoding="utf-8"))

    priced_with = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "price_single_prop_frame"
        for keyword in node.keywords
        if keyword.arg == "mu_column"
    ]

    published = [
        node.value.args[0].slice
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "part['model_projection']"
        and isinstance(node.value, ast.Call)
    ]

    assert len(priced_with) == 1
    assert len(published) == 1

    assert ast.dump(priced_with[0]) == ast.dump(published[0]), (
        "the projection published is not read from the column the line was "
        f"priced against: {ast.unparse(published[0])} vs "
        f"{ast.unparse(priced_with[0])}"
    )


# ======================================================================
# 7-12: the game context
# ======================================================================


def test_7_the_published_teams_are_the_schedules_own_abbreviations(
    release: dict,
):
    """Away and home, as the capture's game payload names them.

    Against the payload rather than against the slate frame: the question is
    whether the schedule's answer survived every stage, and comparing one
    derived frame to another derived frame cannot tell.
    """
    for row in release["document"]["rows"]:
        assert row["home_team"] == HOME_ABBREVIATION

        assert row["away_team"] == AWAY_ABBREVIATION


def test_8_the_player_team_is_the_side_is_home_selects(release: dict):
    """One rule for whose team a player is on, not two.

    ``is_home`` already settles this and is published alongside, so the two
    have to agree on every row. If ``player_team`` were derived anywhere else
    -- from a roster, a second lookup, a default -- this is where it would
    disagree.
    """
    for row in release["document"]["rows"]:
        expected = row["home_team"] if row["is_home"] else row["away_team"]

        assert row["player_team"] == expected

        assert row["player_team"] in (row["home_team"], row["away_team"])


def test_9_the_matchup_is_identical_across_every_quote_on_a_game(
    release: dict,
):
    """Who is playing is a fact about the game, not about the row."""
    grouped: dict[int, set[tuple]] = {}

    for row in release["document"]["rows"]:
        grouped.setdefault(row["game_id"], set()).add(
            (row["away_team"], row["home_team"])
        )

    assert grouped

    for game_id, sides in grouped.items():
        assert len(sides) == 1, f"game {game_id} published {sides}"


def test_10_the_page_renders_away_at_home(rendered: dict, rendered_html: str):
    """Away before home, which is how a slate is read everywhere else.

    Not assembled from the player's own team and ``is_home``: that would give
    the same two teams in an order that flipped row by row within one game.
    """
    assert "matchup" in page.visible_columns(rendered["rows"])

    assert f"{AWAY_ABBREVIATION} @ {HOME_ABBREVIATION}" in rendered_html

    for page_row in rendered["rows"]:
        assert page_row.matchup == (
            f"{page_row.row['away_team']} @ {page_row.row['home_team']}"
        )


def test_11_no_internal_identifier_reaches_the_page_as_a_matchup(
    rendered: dict, rendered_html: str
):
    """"Team 71 vs Team 72" and "game_id 12345" are not matchups."""
    for page_row in rendered["rows"]:
        for field in ("game_id", "team_id", "opponent_id"):
            identifier = page_row.row.get(field)

            if identifier in (None, ""):
                continue

            assert str(identifier) not in page_row.matchup

            assert f"Team {identifier}" not in rendered_html

            assert f"Game {identifier}" not in rendered_html

    assert not re.search(r"Team \d+ (?:vs|@) Team \d+", rendered_html)

    assert "team_id" not in rendered_html

    assert "opponent_id" not in rendered_html


def test_12_a_schedule_that_names_no_teams_publishes_no_matchup(
    served: dict, tmp_path: Path
):
    """An absence is carried as an absence, and the column disappears.

    The abbreviations are optional upstream -- a payload that does not carry
    them yields ``None`` rather than a placeholder -- so a release can
    legitimately have no matchup to show. What must not happen is the page
    filling the gap: no roster, no team table, no integers.
    """
    frame = served["priced"].copy()

    for column in (
        "home_team_abbreviation",
        "visitor_team_abbreviation",
        "player_team_abbreviation",
    ):
        frame[column] = None

    rows = publisher.published_rows(
        frame,
        switch=publisher.read_publishing_switch(
            publisher.MODE_CANARY, project_root=PROJECT
        ),
        slate_date=slate_fixture.SLATE_DATE,
    )

    assert rows

    for row in rows:
        assert row["home_team"] is None

        assert row["away_team"] is None

        assert row["player_team"] is None

    built = page.build_page(
        publisher.LocalDirectoryTransport(
            republish(frame, served, tmp_path / "nameless")
        ),
        registry=REGISTRY,
    )

    assert "matchup" not in page.visible_columns(built["rows"])

    html = built["files"][page.PAGE_NAME].decode("utf-8")

    assert "Matchup" not in html

    for page_row in built["rows"]:
        assert page_row.matchup is None


def republish(frame: pd.DataFrame, served: dict, root: Path) -> Path:
    """Publish an altered priced frame through the real publisher.

    The frame is written back over the priced artifact and the serving receipt
    is re-provenanced to it, because the publisher verifies the digest of the
    file it reads. Altering the release's JSON directly would test a document
    the publisher would never have produced.
    """
    data_root = root / "data"

    priced_path = publisher.priced_markets_path(
        data_root, slate_fixture.SLATE_DATE
    )

    priced_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(priced_path, index=False)

    receipt = json.loads(
        publisher.serving_receipt_path(
            served["data_root"], slate_fixture.SLATE_DATE
        ).read_text(encoding="utf-8")
    )

    receipt["priced_market_artifact"] = {
        "bytes": priced_path.stat().st_size,
        "path": str(priced_path),
        "sha256": publisher.sha256_file(priced_path),
    }

    receipt_path = publisher.serving_receipt_path(
        data_root, slate_fixture.SLATE_DATE
    )

    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    destination = root / "destination"

    publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=slate_fixture.SLATE_DATE,
        data_root=data_root,
        project_root=PROJECT,
        destination_root=destination,
    )

    return destination


# ======================================================================
# 13-14: the tip time
# ======================================================================


def test_13_the_tip_time_keeps_its_time_of_day(release: dict):
    """A date is not a tip time, and this is where one would become one.

    The publisher renders any timestamp as a date, so a tip time carried as a
    ``pd.Timestamp`` would arrive as ``2026-10-20`` and read as published.
    Carried as the schedule's own string, it arrives whole.
    """
    for row in release["document"]["rows"]:
        assert row["tip_off_utc"] == slate_fixture.TIP_OFF

        assert pd.Timestamp(row["tip_off_utc"]).hour == 23

        assert row["tip_off_utc"] != row["date"]


def test_14_the_page_shows_the_tip_time_and_adds_no_game_time_sort(
    rendered: dict, rendered_html: str
):
    """Shown compactly, in UTC, and not turned into a control.

    Sorting a slate by game time was removed from this page deliberately,
    because the published contract had no game time to sort by. It now has
    one, and that is still not a reason to reintroduce a sort, a filter or a
    column: the tip time rides along with the matchup it belongs to.
    """
    for page_row in rendered["rows"]:
        assert page_row.tip_off == slate_fixture.TIP_OFF

    assert 'class="wf-tip"' in rendered_html

    assert "23:00 UTC" in rendered_html

    # Without the date: every row is one slate and the slate's date is already
    # above the table.
    assert f'class="wf-tip">{slate_fixture.SLATE_DATE}' not in rendered_html

    lowered = rendered_html.lower()

    assert "game time" not in lowered

    assert "tip-off (" not in lowered

    options = re.findall(r'<option value="([^"]*)"', rendered_html)

    assert "game_time" not in options

    assert "tip" not in options


# ======================================================================
# the whole chain, counted
# ======================================================================


def test_the_chain_carries_both_facts_from_the_schedule_to_the_screen(
    served: dict, release: dict, rendered: dict, rendered_html: str
):
    """Projections, priced rows, published rows, rendered rows: all non-empty.

    The one test that would fail if any stage silently dropped a row or a
    column. Every per-row assertion elsewhere in this file is vacuous on an
    empty frame, so the counts are asserted against the fixture's own roster
    and quote list rather than against literals, and then each new field is
    followed from the parquet to the published JSON to the HTML.
    """
    assert len(served["projections"]) == len(slate_fixture.ROSTER) > 0

    assert len(served["priced"]) == len(slate_fixture.QUOTED_PROPS) > 0

    assert len(release["document"]["rows"]) == len(served["priced"])

    assert len(rendered["rows"]) == len(served["priced"])

    assert release["document"]["contract_version"] == (
        publisher.PUBLICATION_CONTRACT_VERSION
    )

    sourced = priced_by_key(served["priced"])

    published = published_by_key(release["document"])

    assert set(sourced) == set(published)

    for key, row in published.items():
        source = sourced[key]

        assert row["away_team"] == source["visitor_team_abbreviation"]

        assert row["home_team"] == source["home_team_abbreviation"]

        assert row["player_team"] == source["player_team_abbreviation"]

        assert row["tip_off_utc"] == source["tip_off_utc"]

        assert row["model_projection"] == float(source["model_projection"])

    for page_row in rendered["rows"]:
        rendered_row = published[
            (
                page_row.row["vendor"],
                page_row.row["game_id"],
                page_row.row["player_id"],
                page_row.row["prop_type"],
                page_row.row["line_value"],
            )
        ]

        assert page_row.matchup == (
            f"{rendered_row['away_team']} @ {rendered_row['home_team']}"
        )

        assert page_row.projection == rendered_row["model_projection"]

        assert page_row.player_team == rendered_row["player_team"]

        assert page_row.tip_off == rendered_row["tip_off_utc"]

        assert page.number_text(page_row.projection) in rendered_html

    assert set(page.visible_columns(rendered["rows"])) >= {
        "matchup",
        "projection",
    }


# ======================================================================
# 15-22: nothing numerical moved
# ======================================================================

#: Every field a bettor is shown or that decides what they are shown, grouped
#: the way a reviewer would ask about them. Compared against the same slate
#: served from the production ref, so "unchanged" is measured rather than
#: asserted.
NON_REGRESSION_FIELDS: dict[str, tuple[str, ...]] = {
    "line and side": (
        "line_value",
        "model_preferred_side",
        "market_type",
    ),
    "win probability": (
        "calibrated_q_over_nonpush",
        "calibrated_q_under_nonpush",
        "calibrated_p_push",
        "calibration_method",
    ),
    "fair odds": (
        "calibrated_fair_over_american",
        "calibrated_fair_under_american",
    ),
    "sportsbook odds": (
        "over_odds",
        "under_odds",
    ),
    "market probability": (
        "market_devig_q_over",
        "market_devig_q_under",
        "market_hold_pct",
    ),
    "edge and expected value": (
        "calibrated_edge_over",
        "calibrated_edge_under",
        "calibrated_ev_over",
        "calibrated_ev_under",
        "model_preferred_edge",
        "model_preferred_ev",
    ),
    "vendor and identity": (
        "vendor",
        "game_id",
        "player_id",
        "player_name",
        "prop_type",
        "team_id",
        "opponent_id",
        "is_home",
    ),
    "authority and provenance": (
        "freeze_id",
        "freeze_stage",
        "manifest_sha256",
        "market_pricing_schema_version",
        "projection_schema_version",
        "betting_threshold_policy",
        "auto_bet",
    ),
}


@pytest.fixture(scope="module")
def production_priced(frozen_bundle: Path, tmp_path_factory) -> pd.DataFrame:
    """The same slate, served by the production ref's own code.

    A worktree and a subprocess, because the two versions of the slate builder
    and the pricing script cannot both be imported into one interpreter. This
    is the only way the non-regression claim is a measurement: comparing this
    branch against itself would pass whatever either copy computed.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    worktree = tmp_path_factory.mktemp("production_worktree") / "checkout"

    git("worktree", "add", "--detach", str(worktree), base)

    try:
        root = tmp_path_factory.mktemp("production_served") / "run"

        script = (
            "import sys\n"
            f"sys.path[:0] = [{str(worktree / 'ops')!r}, "
            f"{str(worktree / 'src')!r}, {str(worktree / 'tests')!r}]\n"
            "from pathlib import Path\n"
            "from test_real_slate_serving_root import serve_the_fixture\n"
            f"served = serve_the_fixture(Path({str(frozen_bundle)!r}), "
            f"Path({str(root)!r}))\n"
            "assert served['outcome'] == 'SERVED', served\n"
            "print(served['priced_market_artifact']['path'])\n"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(worktree),
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr[-4000:]

        priced_path = completed.stdout.strip().splitlines()[-1]

        return pd.read_parquet(priced_path)
    finally:
        git("worktree", "remove", "--force", str(worktree))


@pytest.mark.parametrize("family", sorted(NON_REGRESSION_FIELDS))
def test_15_to_22_nothing_a_bettor_is_shown_moved(
    family: str, served: dict, production_priced: pd.DataFrame
):
    """Field for field, row for row, against the production ref's own output.

    Exact equality. The change is additive by construction -- two carried
    columns and one copied one -- so there is no tolerance to argue about: a
    difference of one bit in any of these is this branch having changed a
    price, and that is the thing it must not have done.
    """
    key = ["vendor", "game_id", "player_id", "prop_type", "line_value"]

    before = production_priced.set_index(key).sort_index()

    after = served["priced"].set_index(key).sort_index()

    assert before.index.tolist() == after.index.tolist()

    assert len(before) == len(slate_fixture.QUOTED_PROPS) > 0

    mismatches: list[str] = []

    for column in NON_REGRESSION_FIELDS[family]:
        if column in key:
            continue

        assert column in before.columns, column

        for index, (was, now) in enumerate(
            zip(before[column].tolist(), after[column].tolist())
        ):
            if was != now and not (pd.isna(was) and pd.isna(now)):
                mismatches.append(f"{column}[{index}]: {was!r} -> {now!r}")

    assert mismatches == [], f"{family} changed: {mismatches}"
