"""Unattended publication: the ordering, the pin, and what stays switched off.

The feed publisher and the page renderer each have their own test file, and
both of them answer "does this entry point behave". This file answers the
question neither of them can: when nobody is at a terminal, does the page a
reader fetches come from the release this run published.

That is one property and it needs three kinds of test. The workflow is checked
statically, because the ordering has to be true of the file GitHub schedules
rather than of a function call made here. The orchestrator's refusals are
checked against built releases, because a refusal only means something if the
thing it refuses can be constructed. And the two stages are driven end to end
over local destinations, because the equality the whole arrangement exists to
guarantee -- the page's source digest and the feed's published digest -- is
not observable from either half alone.

The last group is about what did *not* happen. Both live destinations are
configured now, with the real served roots in them, so these tests pin that
configuring an address granted no permission: the switch still says DISABLED,
it is still read before any destination is resolved, and the scheduled
workflow still names none of the three things an activation needs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import production_lifecycle_preflight as preflight  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import run_incumbent_production_serving as serving  # noqa: E402
import run_wizardofodds_publication as automation  # noqa: E402
import validate_main_scheduler_role as scheduler  # noqa: E402
import validate_workflows as validator  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402
import wizardofodds_publisher_fixture as fixture  # noqa: E402

LIFECYCLE = PROJECT / ".github" / "workflows" / "nba_production_lifecycle.yml"

FEED_STEP = "Publish the served slate to the WizardOfOdds feed"

PAGE_STEP = "Build and publish the WizardOfOdds player-props page"

SERVING_STEP = "Serve the slate with the incumbent"

#: The real production SHA the fixture receipt provenances itself with. Shaped
#: like the one the lifecycle resolves, because the page renders the first
#: twelve characters of it into the provenance strip.
REAL_SHA = "64f011cf8cfd0b14b39ee1132e363d17974aa409"


# ======================================================================
# the workflow, read as the file GitHub schedules
# ======================================================================


def lifecycle() -> dict:
    return yaml.safe_load(LIFECYCLE.read_text(encoding="utf-8"))


def steps() -> list[dict]:
    return [
        step
        for step in lifecycle()["jobs"]["lifecycle"]["steps"]
        if isinstance(step, dict)
    ]


def index_of(name: str) -> int:
    for position, step in enumerate(steps()):
        if step.get("name") == name:
            return position

    raise AssertionError(f"no lifecycle step is named {name!r}")


def step_named(name: str) -> dict:
    return steps()[index_of(name)]


def test_1_the_lifecycle_publishes_the_feed_after_the_serving_receipt_exists():
    """Case 1. The receipt is the publication's only source of provenance.

    Serving writes the priced artifact and the receipt that names its digest,
    and the feed stage refuses without both, so a feed stage above serving
    would not publish stale rows -- it would publish nothing, every night.
    """
    assert index_of(SERVING_STEP) < index_of(FEED_STEP)

    command = str(step_named(FEED_STEP)["run"])

    assert "ops/run_wizardofodds_publication.py" in command
    assert "--stage feed" in command
    assert "--serving-receipt" in command


def test_2_the_lifecycle_page_stage_depends_on_the_feed_stage():
    """Case 2. Ordered in the file, and dependent through a file.

    Step order plus fail-fast already stops the page stage from running after
    a failed feed stage. The receipt argument is the half that survives the
    workflow being edited: the page stage learns what to build from the feed
    stage's receipt and has no other source, so a page stage that ran out of
    order would have nothing to read rather than something stale.
    """
    assert index_of(FEED_STEP) < index_of(PAGE_STEP)

    command = str(step_named(PAGE_STEP)["run"])

    assert "--stage page" in command
    assert "--feed-receipt" in command
    assert "--serving-receipt" not in command


def test_3_neither_publication_stage_can_fail_silently():
    """Case 3. A publication that failed quietly is a stale public page.

    The shadow step is deliberately ``continue-on-error`` because a broken
    shadow must not fail production. The opposite is true here: a feed that
    did not publish or a page that did not build is the deliverable not
    happening, and the run has to go red.
    """
    for name in (FEED_STEP, PAGE_STEP):
        assert "continue-on-error" not in step_named(name)


def test_4_both_stages_run_only_in_production_mode():
    """Case 4. A validate-only run reaches no destination."""
    for name in (FEED_STEP, PAGE_STEP):
        assert step_named(name)["if"] == "env.MODE == 'production'"


def test_5_the_static_validator_enforces_the_publication_ordering():
    """Case 5. The ordering is a rule, not an accident of this file.

    Checked by the validator CI already runs, so a later edit that reorders
    the stages, drops the pin or marks a stage non-blocking fails before it
    can be scheduled.
    """
    assert validator.problems_for(LIFECYCLE.name, lifecycle()) == []

    jobs = lifecycle()["jobs"]

    assert validator.publication_order_problems(LIFECYCLE.name, jobs) == []

    reordered = yaml.safe_load(LIFECYCLE.read_text(encoding="utf-8"))["jobs"]

    lifecycle_steps = reordered["lifecycle"]["steps"]

    feed = lifecycle_steps.pop(index_of(FEED_STEP))

    lifecycle_steps.append(feed)

    assert validator.publication_order_problems(LIFECYCLE.name, reordered) != []


def test_6_the_scheduler_guard_requires_both_stages():
    """Case 6. The default branch's copy is held to the same shape."""
    assert FEED_STEP in scheduler.REQUIRED_STEPS
    assert PAGE_STEP in scheduler.REQUIRED_STEPS
    assert "ops/run_wizardofodds_publication.py" in scheduler.REQUIRED_ENTRY_POINTS
    assert (
        "ops/run_wizardofodds_publication.py" in scheduler.BLOCKING_ENTRY_POINTS
    )


# ======================================================================
# the stages, driven over local destinations
# ======================================================================


def serve(root: Path, *, frame=None, slate_date: str | None = None) -> dict:
    """A priced artifact and the receipt production would have written."""
    return fixture.build(
        root,
        frame=frame,
        slate_date=slate_date or fixture.SLATE_DATE,
        receipt_changes={"production_code_sha": REAL_SHA},
    )


def feed_stage(built: dict, destination: Path, **overrides) -> dict:
    arguments = {
        "mode": publisher.MODE_CANARY,
        "slate_date": built["slate_date"],
        "data_root": built["data_root"],
        "project_root": PROJECT,
        "serving_receipt": built["receipt_path"],
        "destination_name": None,
        "destination_root": destination,
    }

    arguments.update(overrides)

    return automation.publish_feed(**arguments)


def page_stage(
    feed_receipt: Path, *, source: Path, destination: Path, **overrides
) -> dict:
    arguments = {
        "mode": publisher.MODE_CANARY,
        "slate_date": fixture.SLATE_DATE,
        "project_root": PROJECT,
        "feed_receipt": feed_receipt,
        "source_destination_name": None,
        "source_destination_root": source,
        "page_destination_name": None,
        "page_destination_root": destination,
        "prefix": page.PREVIEW_PREFIX,
    }

    arguments.update(overrides)

    return automation.publish_the_page(**arguments)


def write_receipt(path: Path, receipt: dict) -> Path:
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    return path


@pytest.fixture
def published(tmp_path: Path) -> dict:
    """One run of both stages, end to end, into two local destinations."""
    built = serve(tmp_path)

    feed_root = tmp_path / "feed"
    page_root = tmp_path / "page"

    feed = feed_stage(built, feed_root)

    receipt = write_receipt(tmp_path / "feed_receipt.json", feed)

    built_page = page_stage(receipt, source=feed_root, destination=page_root)

    return {
        "built": built,
        "feed": feed,
        "feed_receipt": receipt,
        "feed_root": feed_root,
        "page": built_page,
        "page_root": page_root,
    }


def test_7_the_feed_stage_publishes_this_runs_own_artifact(published):
    """Case 7. Nothing is published that this run did not produce.

    Three digests have to agree and they are agreed on from both ends: the
    serving receipt's priced artifact, the publication's provenance of it, and
    the release's own rows.
    """
    feed = published["feed"]

    assert feed["outcome"] == automation.OUTCOME_PUBLISHED

    served = feed["serving"]

    assert served["priced_market_artifact_sha256"] == publisher.sha256_file(
        published["built"]["priced_path"]
    )

    assert (
        feed["provenance"]["priced_market_artifact_sha256"]
        == served["priced_market_artifact_sha256"]
    )

    assert feed["provenance"]["production_code_sha"] == REAL_SHA


def test_8_the_feed_stage_reads_back_what_the_destination_serves(published):
    """Case 8. Contract 2 is established from the destination, not the call.

    The publisher verified the bytes it wrote. This asks the next question:
    after promotion, is the release the pointer names the one this run
    published, and is it the contract the renderer was written against.
    """
    feed = published["feed"]

    promoted = feed["promoted"]

    assert feed["contract_version"] == publisher.PUBLICATION_CONTRACT_VERSION
    assert promoted["contract_version"] == publisher.PUBLICATION_CONTRACT_VERSION
    assert promoted["release_id"] == feed["release"]["release_id"]
    assert promoted["rows_sha256"] == feed["release"]["rows_sha256"]
    assert promoted["row_count"] == len(fixture.QUOTES)

    pointer = json.loads(
        (published["feed_root"] / publisher.CURRENT_POINTER).read_text()
    )

    assert pointer["release_id"] == promoted["release_id"]


def test_9_a_promoted_release_on_the_old_contract_refuses(published):
    """Case 9. The remote canary served contract 1 for weeks.

    So this is the case that actually happened, in the direction it happened
    in. A page is not built from it, and -- the part that matters more -- the
    release that *was* there is not published in its place to make the run
    green.
    """
    root = published["feed_root"]

    rows_relative = published["feed"]["promoted"]["rows_relative"]

    document = json.loads((root / rows_relative).read_text())

    document["contract_version"] = 1

    (root / rows_relative).write_text(json.dumps(document), encoding="utf-8")

    transport = publisher.LocalDirectoryTransport(root)

    try:
        with pytest.raises(automation.PublicationStageFailed) as refusal:
            automation.verify_promoted_release(
                transport,
                release_id=published["feed"]["promoted"]["release_id"],
                rows_sha256=published["feed"]["promoted"]["rows_sha256"],
            )
    finally:
        transport.close()

    assert "publication contract" in str(refusal.value)
    assert "is not published in its place" in str(refusal.value)


def test_10_an_empty_promoted_release_refuses(published):
    """Case 10. A page with no rows is a public claim about nothing."""
    root = published["feed_root"]

    rows_relative = published["feed"]["promoted"]["rows_relative"]

    document = json.loads((root / rows_relative).read_text())

    document["rows"] = []

    (root / rows_relative).write_text(json.dumps(document), encoding="utf-8")

    transport = publisher.LocalDirectoryTransport(root)

    try:
        with pytest.raises(automation.PublicationStageFailed, match="no rows"):
            automation.verify_promoted_release(
                transport,
                release_id=published["feed"]["promoted"]["release_id"],
                rows_sha256=published["feed"]["promoted"]["rows_sha256"],
            )
    finally:
        transport.close()


def test_11_a_receipt_for_another_slate_refuses(tmp_path: Path):
    """Case 11. The one mistake no later digest could see.

    Publishing yesterday's rows under today's date leaves every digest in
    agreement and the page internally consistent and wrong, so the slate date
    is checked against the receipt that names it.
    """
    built = serve(tmp_path, slate_date="2026-10-21")

    with pytest.raises(
        automation.PublicationStageFailed, match="does not provenance"
    ):
        feed_stage(
            built, tmp_path / "feed", slate_date=fixture.SLATE_DATE
        )


@pytest.mark.parametrize("outcome", automation.NO_OP_OUTCOMES)
def test_12_a_slate_with_no_games_publishes_nothing_and_succeeds(
    tmp_path: Path, outcome: str
):
    """Case 12. A no-op is a success, as it is everywhere else in the run.

    An off day and a preseason day both reach the publication stages, and
    neither is a failure. What must not happen is yesterday's page staying up
    while the run reports success about today, so both stages say explicitly
    that they published nothing and why.
    """
    built = serve(tmp_path)

    receipt = json.loads(built["receipt_path"].read_text())

    receipt["outcome"] = outcome

    write_receipt(built["receipt_path"], receipt)

    feed = feed_stage(built, tmp_path / "feed")

    assert feed["outcome"] == automation.OUTCOME_NOT_SERVED
    assert feed["serving"]["served"] is False
    assert not (tmp_path / "feed" / publisher.CURRENT_POINTER).exists()

    built_page = page_stage(
        write_receipt(tmp_path / "feed_receipt.json", feed),
        source=tmp_path / "feed",
        destination=tmp_path / "page",
    )

    assert built_page["outcome"] == automation.OUTCOME_NOT_SERVED
    assert list((tmp_path / "page").rglob("*.html")) == []


def test_12a_a_failed_serving_run_is_not_a_no_op(tmp_path: Path):
    """Case 12a. A broken serving step produced no artifact either.

    Treating that like an off day would leave yesterday's page up while the
    run reported success, which is the one way a fail-closed chain can fail
    open. So only the two recorded no-ops are no-ops, and everything else --
    including a serving failure and an outcome nobody has heard of -- refuses.
    """
    for outcome in (serving.OUTCOME_FAILED, "SOMETHING_NEW"):
        built = serve(tmp_path / outcome)

        receipt = json.loads(built["receipt_path"].read_text())

        receipt["outcome"] = outcome

        write_receipt(built["receipt_path"], receipt)

        with pytest.raises(
            automation.PublicationStageFailed, match="is unknown and nothing"
        ):
            feed_stage(built, tmp_path / outcome / "feed")

    assert serving.OUTCOME_FAILED not in automation.NO_OP_OUTCOMES


def test_13_a_refused_feed_stage_stops_the_page_stage(tmp_path: Path):
    """Case 13. A failure is not a licence to rebuild the last page."""
    refused = {
        "mode": publisher.MODE_CANARY,
        "outcome": "PUBLICATION_REFUSED",
        "slate_date": fixture.SLATE_DATE,
        "stage": automation.STAGE_FEED,
    }

    with pytest.raises(
        automation.PublicationStageFailed, match="no verified release"
    ):
        page_stage(
            write_receipt(tmp_path / "feed_receipt.json", refused),
            source=tmp_path / "feed",
            destination=tmp_path / "page",
        )


def test_14_a_feed_receipt_on_the_old_contract_builds_no_page(tmp_path: Path):
    """Case 14. Checked twice, from the receipt as well as the destination."""
    built = serve(tmp_path)

    feed = feed_stage(built, tmp_path / "feed")

    feed["contract_version"] = 1

    with pytest.raises(
        automation.PublicationStageFailed, match="publication contract 1"
    ):
        page_stage(
            write_receipt(tmp_path / "feed_receipt.json", feed),
            source=tmp_path / "feed",
            destination=tmp_path / "page",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("slate_date", "2026-10-21"),
        ("mode", publisher.MODE_LIVE),
    ),
)
def test_15_a_feed_receipt_from_another_run_builds_no_page(
    tmp_path: Path, field: str, value: str
):
    """Case 15. The receipt has to be *this* run's feed publication."""
    built = serve(tmp_path)

    feed = feed_stage(built, tmp_path / "feed")

    feed[field] = value

    with pytest.raises(
        automation.PublicationStageFailed, match="is not this run's"
    ):
        page_stage(
            write_receipt(tmp_path / "feed_receipt.json", feed),
            source=tmp_path / "feed",
            destination=tmp_path / "page",
        )


def test_16_the_page_is_built_from_the_release_the_feed_published(published):
    """Case 16. The equality the whole arrangement exists to produce.

    Exact, on both halves of the release's identity: the id the pointer names
    and the digest of the rows themselves.
    """
    source_release = published["page"]["source_release"]

    promoted = published["feed"]["promoted"]

    assert source_release["release_id"] == promoted["release_id"]
    assert source_release["rows_sha256"] == promoted["rows_sha256"]
    assert source_release["expected"]["release_id"] == promoted["release_id"]
    assert source_release["expected"]["rows_sha256"] == promoted["rows_sha256"]
    assert source_release["slate_date"] == fixture.SLATE_DATE

    written = published["page"]["written"]

    assert sorted(written) == [
        f"{page.PREVIEW_PREFIX}/{page.PAGE_NAME}",
        f"{page.PREVIEW_PREFIX}/{page.PAGE_MANIFEST_NAME}",
    ]


def test_17_a_source_serving_a_different_release_refuses(published, tmp_path: Path):
    """Case 17. Refused at the pointer, before a row is read.

    The distinction this makes is the one the task is about. A page built from
    the wrong release and then caught is a page that was rendered; this one is
    never rendered, because the pin is checked at the current pointer on the
    way in.
    """
    second = serve(tmp_path / "second", frame=fixture.priced_frame().head(2))

    feed_stage(second, published["feed_root"])

    with pytest.raises(page.PageRefused) as refusal:
        page_stage(
            published["feed_receipt"],
            source=published["feed_root"],
            destination=tmp_path / "page-again",
        )

    assert published["feed"]["promoted"]["release_id"] in str(refusal.value)

    assert list((tmp_path / "page-again").rglob("*.html")) == []


def test_18_the_published_rows_carry_what_the_page_renders(published):
    """Case 18. Identity, matchup, projection and tip time, end to end.

    Not a re-test of the renderer: the question here is whether the automated
    path hands the renderer the same fields the hand-run path did, so the
    public page is unchanged by being automated.
    """
    rows_relative = published["feed"]["promoted"]["rows_relative"]

    rows = json.loads((published["feed_root"] / rows_relative).read_text())["rows"]

    for row in rows:
        assert row["player_name"].strip()
        assert row["player_team"].strip()
        assert row["away_team"].strip()
        assert row["home_team"].strip()
        assert row["tip_off_utc"]
        assert row["model_projection"] is not None

    html = (
        published["page_root"]
        / page.PREVIEW_PREFIX
        / page.PAGE_NAME
    ).read_text(encoding="utf-8")

    for row in rows:
        assert row["player_name"] in html

    assert "data-row" in html

    # The iframe handshake and the sportsbook routing are the two behaviours a
    # reader notices immediately if an automated page lost them. Read off the
    # page's own manifest rather than scraped out of the markup, because the
    # manifest is what the renderer states it did.
    manifest = json.loads(
        (
            published["page_root"] / page.PREVIEW_PREFIX / page.PAGE_MANIFEST_NAME
        ).read_text()
    )

    assert manifest["embed"]["message_key"] == page.EMBED_MESSAGE_KEY
    assert manifest["embed"]["parent_origins"] == list(page.PARENT_ORIGINS)
    assert page.EMBED_MESSAGE_KEY in html

    rendered = manifest["rendered"]

    assert sum(manifest["rendered"]["by_tier"].values()) == rendered[
        "prediction_count"
    ]

    assert set(manifest["rendered"]["by_tier"]) == {
        routing.TIER_PRIMARY,
        routing.TIER_FALLBACK,
        routing.TIER_NONE,
    }


def test_19_the_same_rows_republish_as_the_same_release(published, tmp_path: Path):
    """Case 19. A rerun republishes a day rather than accumulating one.

    A release id is the digest of its own rows, so identical rows are the same
    release; promoting the release that is already current is a no-op, which
    is what preserves both its promotion time and the rollback target behind
    it.
    """
    before = json.loads(
        (published["feed_root"] / publisher.CURRENT_POINTER).read_text()
    )

    again = feed_stage(published["built"], published["feed_root"])

    after = json.loads(
        (published["feed_root"] / publisher.CURRENT_POINTER).read_text()
    )

    assert again["promoted"]["release_id"] == published["feed"]["promoted"][
        "release_id"
    ]

    assert after == before

    releases = sorted(
        path.name
        for path in (published["feed_root"] / publisher.RELEASES_DIRECTORY).iterdir()
    )

    assert releases == [published["feed"]["promoted"]["release_id"]]


def test_20_the_same_release_renders_the_same_page_twice(published, tmp_path: Path):
    """Case 20. No clock in the page, so republication is not a diff."""
    first = (
        published["page_root"] / page.PREVIEW_PREFIX / page.PAGE_NAME
    ).read_bytes()

    second_root = tmp_path / "page-again"

    again = page_stage(
        published["feed_receipt"],
        source=published["feed_root"],
        destination=second_root,
    )

    assert (
        second_root / page.PREVIEW_PREFIX / page.PAGE_NAME
    ).read_bytes() == first

    assert again["page_sha256"] == published["page"]["page_sha256"]


def test_21_every_completed_stage_leaves_an_audit_trail(published):
    """Case 21. At the destination, where somebody looking at it can find it.

    Including the rerun that promoted nothing: a republication that left no
    line would make the audit log a record of changes rather than of actions.
    """
    log = published["feed_root"] / publisher.AUDIT_LOG_NAME

    assert log.exists()

    entries = [
        json.loads(line) for line in log.read_text().splitlines() if line.strip()
    ]

    assert entries

    assert all(
        entry["release_id"] == published["feed"]["promoted"]["release_id"]
        for entry in entries
    )

    manifest = json.loads(
        (
            published["page_root"] / page.PREVIEW_PREFIX / page.PAGE_MANIFEST_NAME
        ).read_text()
    )

    released = manifest["released"]

    assert released["release_id"] == published["feed"]["promoted"]["release_id"]
    assert released["rows_sha256"] == published["feed"]["promoted"]["rows_sha256"]
    assert manifest["page_sha256"] == published["page"]["page_sha256"]
    assert published["page"]["row_count"] == manifest["rendered"][
        "prediction_count"
    ]


def test_22_retention_never_drops_below_the_rollback_target(published, tmp_path: Path):
    """Case 22. Pruning after the pointers move, never below two."""
    for index in (2, 3, 4):
        feed_stage(
            serve(tmp_path / f"day{index}", frame=fixture.priced_frame().head(index)),
            published["feed_root"],
        )

    releases = sorted(
        path.name
        for path in (published["feed_root"] / publisher.RELEASES_DIRECTORY).iterdir()
    )

    previous = json.loads(
        (published["feed_root"] / publisher.PREVIOUS_POINTER).read_text()
    )

    assert len(releases) >= 2
    assert previous["release_id"] in releases


# ======================================================================
# live: configured, and switched off
# ======================================================================


def named(name: str) -> dict:
    document = publisher.read_destinations(project_root=PROJECT)

    return document["named_destinations"][name]


def test_23_both_live_destinations_address_the_served_roots():
    """Case 23. The deliverable's own address, and nothing else missing.

    Configuration completeness is the point of this case: a later reviewed
    activation must not have to invent a destination, write code or copy a
    file by hand. Every field the transport needs is here, and every one of
    them is the canary's value with the confinement removed.
    """
    feed = named(automation.FEED_DESTINATIONS[publisher.MODE_LIVE])

    page_live = named(page.MODE_DESTINATIONS[publisher.MODE_LIVE]["page"])

    for declared in (feed, page_live):
        assert declared["mode"] == publisher.MODE_LIVE
        assert declared["transport"] == "ssh_rsync"
        assert declared["user"] == "nba-prop-deploy"
        assert declared["credential_env"] == "NBA_WOO_SSH_PRIVATE_KEY"
        assert declared["retained_releases"] >= 2
        assert "canary_subpath" not in declared

        for field in publisher.REQUIRED_REMOTE_FIELDS:
            assert str(declared[field]).strip()

    assert feed["public_target"].endswith("/predictions/nba")

    assert page_live["public_target"].endswith("/predictions/NBA/player-props")

    # Resolved, because a served root is a property of the resolved
    # destination rather than of the way the record happens to be written.
    for name in (
        automation.FEED_DESTINATIONS[publisher.MODE_LIVE],
        page.MODE_DESTINATIONS[publisher.MODE_LIVE]["page"],
    ):
        remote = publisher.resolve_destination(
            publisher.MODE_LIVE,
            publisher.read_destinations(project_root=PROJECT),
            root=None,
            name=name,
        ).remote

        assert remote.published_root == remote.public_target
        assert remote.published_base_url == remote.public_base_url


def test_24_live_execution_is_gated_before_any_destination_resolves():
    """Case 24. An address is not a permission, and neither is a state.

    The switch is read first, so a fully configured live destination is
    unreachable while the gate is shut -- for the feed and for the page,
    through the one switch rather than two that could drift.

    The committed state is now ENABLED, taken in a reviewed commit, and the
    gate is still shut. ``enabled`` is the state *and* an empty blocker list,
    and the two approval variables live in the ``wizardofodds-production``
    GitHub environment where no commit can set them.
    """
    document = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    assert document["modes"]["live"]["state"] == publisher.STATE_ENABLED
    assert document["modes"]["canary"]["state"] == publisher.STATE_ENABLED
    assert document["published_authority"] == publisher.MODEL_AUTHORITY
    assert document["candidate_authority"] == "shadow_only"

    switch = publisher.read_publishing_switch(
        publisher.MODE_LIVE, project_root=PROJECT, environment={}
    )

    assert switch.enabled is False

    for variable in (
        publisher.PUBLISH_ENV_VAR,
        publisher.PUBLISH_APPROVAL_ENV_VAR,
    ):
        assert any(variable in blocker for blocker in switch.blockers)

    with pytest.raises(publisher.PublishingRefused, match="live publishing is"):
        page.assert_mode_is_permitted(
            publisher.MODE_LIVE, project_root=PROJECT
        )

    switch = publisher.read_publishing_switch(
        publisher.MODE_LIVE, project_root=PROJECT
    )

    with pytest.raises(publisher.PublishingRefused, match="live publishing is"):
        publisher.assert_mode_enabled(switch)


def test_25_a_scheduled_run_cannot_become_an_activation():
    """Case 25. Nothing in the file is an approval, and none can be inferred.

    The three things an activation needs are the committed switch, the publish
    variable and the approval token. The scheduled workflow names none of
    them, asks for the canary mode explicitly, and carries no override of any
    kind -- no skip flag, no force, no missing-feature fallback.
    """
    text = LIFECYCLE.read_text(encoding="utf-8")

    for token in (
        publisher.PUBLISH_ENV_VAR,
        publisher.PUBLISH_APPROVAL_ENV_VAR,
        f"--mode {publisher.MODE_LIVE}",
        "--stage live",
        "--force",
        "--allow-missing",
        "--skip-verification",
        "continue-on-error: true  # publication",
    ):
        assert token not in text, token

    for name in (FEED_STEP, PAGE_STEP):
        assert f"--mode {publisher.MODE_CANARY}" in str(step_named(name)["run"])

    assert validator.secret_leak_problems(LIFECYCLE.name) == []


def test_26_the_deployment_key_is_bound_by_name_only():
    """Case 26. The workflow carries a reference; the runner carries the key.

    Named in the preflight's optional secrets, so its absence is reported
    rather than discovered halfway through a publication, and optional rather
    than required because the transport refuses itself when the variable is
    empty.
    """
    assert "NBA_WOO_SSH_PRIVATE_KEY" in preflight.OPTIONAL_PRODUCTION_SECRETS
    assert "NBA_WOO_SSH_PRIVATE_KEY" not in preflight.REQUIRED_PRODUCTION_SECRETS

    for name in (FEED_STEP, PAGE_STEP):
        binding = step_named(name)["env"]["NBA_WOO_SSH_PRIVATE_KEY"]

        assert binding == "${{ secrets.NBA_WOO_SSH_PRIVATE_KEY }}"

    assert "vars.NBA_WOO_SSH_PRIVATE_KEY" not in LIFECYCLE.read_text(
        encoding="utf-8"
    )


def test_27_no_receipt_or_summary_can_carry_the_key(published, tmp_path: Path):
    """Case 27. Receipts are uploaded as artifacts and summaries are public."""
    rendered = json.dumps(published["feed"], default=str) + json.dumps(
        published["page"], default=str
    )

    rendered += automation.render(published["feed"])

    rendered += automation.render(published["page"])

    assert "NBA_WOO_SSH_PRIVATE_KEY" not in rendered

    assert "BEGIN OPENSSH PRIVATE KEY" not in rendered

    source = Path(automation.__file__).read_text(encoding="utf-8")

    assert "NBA_WOO_SSH_PRIVATE_KEY" not in source


def test_28_the_orchestrator_computes_nothing(published):
    """Case 28. No number in the publication originates in this module.

    Imports rather than output, because that is the version of this claim that
    cannot be satisfied accidentally: a module that imports no model, copula,
    calibration or pricing path has nothing to compute with.
    """
    source = Path(automation.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "nba_prop_quant.copula",
        "nba_prop_quant.calibration",
        "nba_prop_quant.pricing",
        "15_price_markets",
        "run_adaptive_daily_fit",
        "promotion_state",
        "current_good_fit_id",
    ):
        assert forbidden not in source, forbidden

    assert automation.SERVED_OUTCOME == serving.OUTCOME_SERVED

    assert automation.NO_OP_OUTCOMES == (
        serving.OUTCOME_NO_PRODUCTION_SLATE,
        serving.OUTCOME_NO_GAMES_ON_SLATE,
    )

    assert sorted(automation.FEED_DESTINATIONS) == sorted(page.MODE_DESTINATIONS)
