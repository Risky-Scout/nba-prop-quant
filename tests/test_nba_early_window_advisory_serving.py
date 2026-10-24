"""Tests for the early advisory serving path.

The claim this module exists to hold is narrow and absolute: an advisory run
cannot contaminate the T-20 record. Not "does not", "cannot" -- so the tests
are about structure rather than about behaviour under the arguments the
controller happens to pass. The destinations are proved disjoint from the
resolved configuration, the pointer paths are proved unreachable from each
other, and each refusal is called directly with the thing it refuses.

Nothing here trains, fits, prices, serves or publishes. The repository is only
read from, and the only files written are inside a pytest temporary directory.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from nba_prop_quant.snapshot_schedule import CAPTURE_OFFSETS_MINUTES

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from t20_capture_fixture import SLATE, at, game  # noqa: E402

OPS = PROJECT / "ops"

ADVISORY_ENTRY_POINT = OPS / "run_advisory_window_serving.py"

T20_ENTRY_POINT = OPS / "run_t20_production_controller.py"

DESTINATIONS = OPS / "wizardofodds_destinations.json"

EASTERN = ZoneInfo("America/New_York")

PRODUCTION_SHA = "a728b342a2f728c13a705abda8a5bbb79caf53ea"


def load_ops(name: str):
    spec = importlib.util.spec_from_file_location(
        f"advisory_ops_{name}", OPS / f"{name}.py"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


advisory = load_ops("run_advisory_window_serving")

controller = load_ops("run_t20_production_controller")

publisher = load_ops("publish_incumbent_to_wizardofodds")


@pytest.fixture(autouse=True)
def block_all_network(monkeypatch):
    """No test in this module may reach the network."""

    def deny(*args, **kwargs):
        raise RuntimeError(
            f"network access is forbidden in this test module: {args!r}"
        )

    import socket

    monkeypatch.setattr(socket.socket, "connect", deny)

    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture(scope="module")
def document() -> dict:
    return publisher.read_destinations(destinations_path=DESTINATIONS)


def resolve(document: dict, name: str):
    return publisher.resolve_destination("canary", document, name=name)


# ----------------------------------------------------------------------
# the pointer an advisory run cannot reach
# ----------------------------------------------------------------------


def test_no_advisory_destination_can_repoint_the_t20_pointer(document):
    """The central claim, proved from the resolved configuration.

    Every pointer the publisher writes -- current.json, previous.json, the
    canary attestation, the audit log -- is written inside the destination's
    own published root. So "an advisory run must not repoint the T-20
    current.json" is true if and only if no advisory published root contains
    the T-20 one, and that is a statement about two strings.
    """
    pairs = [
        (controller.T20_FEED_DESTINATION, controller.ADVISORY_FEED_DESTINATIONS),
        (controller.T20_PAGE_DESTINATION, controller.ADVISORY_PAGE_DESTINATIONS),
    ]

    for t20_name, advisory_names in pairs:
        t20_root = resolve(document, t20_name).remote.published_root.rstrip("/")

        t20_pointer = f"{t20_root}/{publisher.CURRENT_POINTER}"

        for offset, name in advisory_names.items():
            root = resolve(document, name).remote.published_root.rstrip("/")

            assert root != t20_root, (
                f"T-{offset} publishes to the same root as {t20_name}"
            )

            assert not root.startswith(t20_root + "/"), (
                f"T-{offset} publishes inside {t20_name}'s served root, so "
                "retention there would prune the T-20 releases"
            )

            assert not t20_root.startswith(root + "/"), (
                f"{t20_name} publishes inside T-{offset}'s served root"
            )

            assert f"{root}/{publisher.CURRENT_POINTER}" != t20_pointer


def test_every_advisory_destination_is_a_confined_canary(document):
    """Canary mode, and confined below the served root the site publishes."""
    names = [
        *controller.ADVISORY_FEED_DESTINATIONS.values(),
        *controller.ADVISORY_PAGE_DESTINATIONS.values(),
    ]

    assert len(names) == 6

    for name in names:
        destination = resolve(document, name)

        assert destination.mode == "canary"

        # Raises if the resolved root is the served root or outside it.
        publisher.assert_remote_canary_is_confined(destination)

        assert destination.remote.retained_releases >= 2


def test_an_advisory_destination_cannot_be_resolved_as_live(document):
    """The name declares its mode, so asking for live is a refusal."""
    for name in controller.ADVISORY_FEED_DESTINATIONS.values():
        with pytest.raises(publisher.PublishingRefused):
            publisher.resolve_destination("live", document, name=name)


# ----------------------------------------------------------------------
# the label
# ----------------------------------------------------------------------


def test_an_advisory_release_is_labelled_with_the_offset_it_captured(document):
    """The offset is in the release id, so it is in everything downstream.

    A release id is ``{release_prefix}__{slate_date}__{rows_sha256[:16]}``, and
    the release id is what the feed manifest records, what current.json names,
    what the page manifest's ``released.release_id`` carries and what the
    rendered page shows in its Release fact row. Putting the offset in the
    prefix therefore labels all of them at once.
    """
    digest = "0" * 64

    for offset, name in controller.ADVISORY_FEED_DESTINATIONS.items():
        destination = resolve(document, name)

        release_id = f"{destination.release_prefix}__{SLATE}__{digest[:16]}"

        assert f"advisory_t{offset}m" in destination.release_prefix

        assert f"advisory_t{offset}m" in release_id

        assert f"canary-advisory/t{offset}m" in destination.remote.published_root

        assert f"canary-advisory/t{offset}m/" in destination.remote.published_base_url

        t20_prefix = resolve(
            document, controller.T20_FEED_DESTINATION
        ).release_prefix

        assert not release_id.startswith(t20_prefix + "__"), (
            f"the T-{offset} release id would sort and read as a T-20 release"
        )

    for offset, name in controller.ADVISORY_PAGE_DESTINATIONS.items():
        destination = resolve(document, name)

        assert f"advisory_t{offset}m" in destination.release_prefix

        assert f"canary-advisory/t{offset}m" in destination.remote.published_root

    for offset, prefix in controller.ADVISORY_PAGE_PREFIXES.items():
        assert f"t{offset}m" in prefix

        assert "advisory" in prefix


def test_the_advisory_receipt_carries_the_offset_and_the_marker():
    """Additive labelling: the production outcome and fields are untouched."""
    produced = {
        "outcome": "SERVED",
        "incumbent_authority": "frozen_deployment_bundle",
        "model_authority": "incumbent",
        "served_game_ids": [101],
        "slate_date": SLATE,
    }

    receipt = advisory.build_advisory_receipt(
        receipt=dict(produced),
        receipt_path=Path("/tmp/advisory.json"),
        offset_minutes=180,
        window_ids=["101:T-180m"],
    )

    assert receipt[advisory.ADVISORY_MARKER_FIELD] is True

    assert receipt["advisory_capture_offset_minutes"] == 180

    assert receipt["advisory_capture_window_ids"] == ["101:T-180m"]

    # The outcome is carried through rather than renamed. The publication
    # orchestrator requires exactly SERVED and is a file this lineage may not
    # edit, so a renamed outcome would be an unpublishable receipt.
    for field, value in produced.items():
        assert receipt[field] == value


# ----------------------------------------------------------------------
# the refusals
# ----------------------------------------------------------------------


def test_an_offset_the_capture_schedule_does_not_define_is_refused():
    """Not in CAPTURE_OFFSETS_MINUTES, so no capture of it could exist."""
    for offset in (0, 19, 21, 600, -45):
        assert offset not in CAPTURE_OFFSETS_MINUTES

        with pytest.raises(advisory.AdvisoryServingRefused) as error:
            advisory.assert_offset_is_advisory(offset)

        assert "capture schedule defines" in str(error.value)


def test_the_production_offset_is_refused_by_name():
    """T-20 has one producer, and this entry point is not it."""
    with pytest.raises(advisory.AdvisoryServingRefused) as error:
        advisory.assert_offset_is_advisory(20)

    message = str(error.value)

    assert "production T-20 window" in message

    assert "run_incumbent_production_serving" in message


def test_a_scheduled_research_offset_is_refused_as_advisory():
    """Scheduled and not T-20 is still not advisory: nothing carries it."""
    for offset in (1440, 480, 5):
        assert offset in CAPTURE_OFFSETS_MINUTES

        assert offset not in advisory.ADVISORY_OFFSETS_MINUTES

        with pytest.raises(advisory.AdvisoryServingRefused) as error:
            advisory.assert_offset_is_advisory(offset)

        assert "research window" in str(error.value)


def test_every_advisory_offset_is_accepted():
    for offset in advisory.ADVISORY_OFFSETS_MINUTES:
        assert advisory.assert_offset_is_advisory(offset) == offset


def test_an_advisory_run_cannot_record_a_t20_window_id():
    """The one label an advisory receipt carries, checked rather than trusted."""
    with pytest.raises(advisory.AdvisoryServingRefused) as error:
        advisory.assert_window_ids_are_this_offset(
            ["101:T-180m", "102:T-20m"], offset_minutes=180
        )

    message = str(error.value)

    assert "102:T-20m" in message

    assert "the name of the production record" in message


def test_the_controller_cannot_capture_a_t20_window_as_advisory():
    """The same refusal on the capture side of the same window."""
    with pytest.raises(controller.ControllerRefused) as error:
        controller.assert_window_is_not_production(
            ["101:T-20m"], offset_minutes=180
        )

    assert "101:T-20m" in str(error.value)

    # And a window belonging to some other advisory offset is refused too, so
    # a T-90 capture cannot be served and published as a T-180 release.
    with pytest.raises(controller.ControllerRefused):
        controller.assert_window_is_not_production(
            ["101:T-90m"], offset_minutes=180
        )


def test_an_advisory_receipt_cannot_be_written_where_production_proves_itself(
    tmp_path,
):
    """processed/incumbent_serving is the record readiness is proven from."""
    data_root = tmp_path / "data"

    production = data_root / advisory.SERVING_RECEIPTS_RELATIVE

    production.mkdir(parents=True)

    for candidate in (
        production / f"{SLATE}.json",
        production / "nested" / f"{SLATE}.json",
    ):
        with pytest.raises(advisory.AdvisoryServingRefused) as error:
            advisory.assert_receipt_path_is_not_the_production_record(
                candidate, data_root=data_root
            )

        assert "the durable production serving receipt" in str(error.value)

    allowed = data_root / advisory.ADVISORY_RECEIPTS_RELATIVE / f"{SLATE}.json"

    assert advisory.assert_receipt_path_is_not_the_production_record(
        allowed, data_root=data_root
    ) == allowed.resolve()


def test_readiness_refuses_an_advisory_receipt_as_the_production_record(tmp_path):
    """The second, independent guard, on the reading side.

    An advisory receipt is the same shape as a production one by design --
    the same function builds it -- so the T-20 controller would accept it on
    its shape alone. It is refused on the marker instead.
    """
    data_root = tmp_path / "data"

    path = data_root / "processed" / "incumbent_serving" / f"{SLATE}.json"

    path.parent.mkdir(parents=True)

    receipt = {
        controller.ADVISORY_RECEIPT_MARKER_FIELD: True,
        "advisory_capture_offset_minutes": 180,
        "generated_at": "2026-10-20T20:05:00+00:00",
        "incumbent_authority": "frozen_deployment_bundle",
        "incumbent_version": "nba_prop_quant_20260818T205213Z",
        "model_authority": "incumbent",
        "outcome": "SERVED",
        "production_code_sha": PRODUCTION_SHA,
        "serving_bundle_root": str(tmp_path / "bundle"),
        "serving_model_dir": str(tmp_path / "bundle" / "models"),
        "slate_date": SLATE,
        "slate_readiness": {"ready": True, "refresh_outcome": "REFRESHED"},
    }

    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(controller.ControllerRefused) as error:
        controller.morning_readiness(
            data_root=data_root,
            slate_date=SLATE,
            production_sha=PRODUCTION_SHA,
            project_root=PROJECT,
        )

    assert "is an advisory receipt" in str(error.value)


def test_the_controller_refuses_an_advisory_offset_it_cannot_schedule():
    """Construction-time, so a bad offset never reaches a capture."""
    for offset in (600, 21):
        with pytest.raises(controller.ControllerRefused):
            controller.advisory_rows(
                [game(101, at(23, 0))], offsets=(offset,)
            )

    # The production offset is refused as an advisory offset specifically,
    # rather than being silently served twice.
    with pytest.raises(controller.ControllerRefused) as error:
        controller.advisory_rows([game(101, at(23, 0))], offsets=(20,))

    assert "T-20" in str(error.value)


def test_the_controller_refuses_a_t20_destination_for_an_advisory_release():
    """Addressing is checked, not assumed, at the point of publication."""
    with pytest.raises(controller.ControllerRefused):
        controller.assert_destinations_are_advisory(
            offset_minutes=180,
            feed=controller.T20_FEED_DESTINATION,
            page=controller.ADVISORY_PAGE_DESTINATIONS[180],
        )

    with pytest.raises(controller.ControllerRefused):
        controller.assert_destinations_are_advisory(
            offset_minutes=180,
            feed=controller.ADVISORY_FEED_DESTINATIONS[180],
            page=controller.T20_PAGE_DESTINATION,
        )

    # Another offset's destinations are refused too.
    with pytest.raises(controller.ControllerRefused):
        controller.assert_destinations_are_advisory(
            offset_minutes=180,
            feed=controller.ADVISORY_FEED_DESTINATIONS[90],
            page=controller.ADVISORY_PAGE_DESTINATIONS[180],
        )

    controller.assert_destinations_are_advisory(
        offset_minutes=180,
        feed=controller.ADVISORY_FEED_DESTINATIONS[180],
        page=controller.ADVISORY_PAGE_DESTINATIONS[180],
    )


# ----------------------------------------------------------------------
# live publishing stays refused
# ----------------------------------------------------------------------


def test_live_publishing_is_not_reachable_from_the_advisory_path():
    """Unchanged by this path, and asserted rather than assumed.

    The live state has since been set in a reviewed commit, and what this
    test is about survived it: the advisory path cannot publish live. The
    switch is one of three conditions, the other two are not set, and the
    advisory serving module names neither variable nor a live mode.
    """
    switch = publisher.read_publishing_switch(
        "live", project_root=PROJECT, environment={}
    )

    assert switch.enabled is False

    source = (PROJECT / "ops" / "run_advisory_window_serving.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        publisher.PUBLISH_ENV_VAR,
        publisher.PUBLISH_APPROVAL_ENV_VAR,
        "--mode live",
        "--stage live",
        f'"{publisher.MODE_LIVE}"',
    ):
        assert forbidden not in source, forbidden


def test_the_advisory_sources_name_no_publication_activation_material():
    """An advisory release is a canary release and has no other mode."""
    forbidden = (
        "NBA_PROP_WOO_PUBLISH",
        "NBA_PROP_WOO_PUBLISH_APPROVAL",
        "--mode live",
        "--stage live",
    )

    for path in (ADVISORY_ENTRY_POINT, T20_ENTRY_POINT):
        text = path.read_text(encoding="utf-8")

        for token in forbidden:
            for number, line in enumerate(text.splitlines(), 1):
                assert token not in line, (
                    f"{path.name}:{number} names {token!r}, which is live "
                    "publication activation material"
                )


def string_constants(tree: ast.AST) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_every_advisory_publication_the_controller_issues_is_canary():
    """Read off the source: the only mode in an advisory command is canary."""
    entry_point = ast.parse(ADVISORY_ENTRY_POINT.read_text(encoding="utf-8"))

    assert "--mode" not in string_constants(entry_point), (
        "the advisory serving entry point publishes nothing and has no mode"
    )

    assert controller.PUBLICATION_MODE == "canary"

    tree = ast.parse(T20_ENTRY_POINT.read_text(encoding="utf-8"))

    advisory_methods = {
        "_serve_and_publish_advisory",
        "_publish_advisory_record",
        "_advisory_serve",
    }

    modes: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        if node.name not in advisory_methods:
            continue

        for command in ast.walk(node):
            if not isinstance(command, ast.List):
                continue

            for position, element in enumerate(command.elts[:-1]):
                if (
                    isinstance(element, ast.Constant)
                    and element.value == "--mode"
                ):
                    modes.append(ast.unparse(command.elts[position + 1]))

    assert modes, "no advisory publication command was found to check"

    # The one module-level constant, which is "canary", rather than a literal
    # an edit could quietly change in one command and not the others.
    assert set(modes) == {"PUBLICATION_MODE"}


# ----------------------------------------------------------------------
# the model surface
# ----------------------------------------------------------------------


def test_this_branch_changes_no_model_surface_path():
    """Nothing fitted, configured or computed by the model path is touched.

    Derived from git rather than from a list kept here, so a file added to the
    branch later cannot escape it. The declaration in safety.py is the review
    record; this is the arithmetic behind it.
    """
    base = subprocess.run(
        [
            "git",
            "merge-base",
            "HEAD",
            "origin/production/wizardofodds-integration",
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )

    if base.returncode != 0:
        pytest.skip("the production ref is not available in this checkout")

    changed = subprocess.run(
        ["git", "diff", "--name-only", base.stdout.strip(), "HEAD"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    if not changed:
        pytest.skip("this head is production, so it changes nothing")

    model_surface = (
        "models/",
        "configs/",
        "scripts/",
        "release/",
    )

    offenders = [
        path for path in changed if path.startswith(model_surface)
    ]

    assert offenders == [], (
        f"{offenders} is model surface: a fitted artifact, a frozen config, "
        "a serving script or the release surface"
    )

    # The only source module this branch may touch is the containment
    # declaration itself, which computes nothing the model path reads.
    sources = [
        path
        for path in changed
        if path.startswith("src/")
        and path != "src/nba_prop_quant/research/game_latent_state/safety.py"
    ]

    assert sources == []


def test_the_advisory_entry_point_cannot_reach_the_model_surface():
    """It imports the registry and the schedule, and nothing else.

    Stronger than reading its arithmetic, because it has none: the modules
    that define marginals, the copula, calibration, minutes, rates and
    pricing are simply not importable from here, so there is no edit to this
    file that could change a number the model produces without first adding
    an import this test would fail on.
    """
    tree = ast.parse(ADVISORY_ENTRY_POINT.read_text(encoding="utf-8"))

    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}

        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    model_surface = {
        module
        for module in imported
        if module.startswith("nba_prop_quant.")
        and module
        not in {
            "nba_prop_quant.adaptive_fit_registry",
            "nba_prop_quant.snapshot_schedule",
        }
    }

    assert model_surface == set(), (
        f"{sorted(model_surface)} is reachable from the advisory entry point"
    )

    # And the one ops module it does import is the production orchestrator,
    # by name, so there is no second definition of what is serving.
    assert "run_incumbent_production_serving" in imported


# ----------------------------------------------------------------------
# the two modules agree, and the offsets are the ones asked for
# ----------------------------------------------------------------------


def test_the_controller_and_the_entry_point_agree_on_the_offsets():
    """One definition of which windows are advisory, pinned in both places."""
    assert (
        controller.ADVISORY_OFFSETS_MINUTES == advisory.ADVISORY_OFFSETS_MINUTES
    )

    assert (
        controller.ADVISORY_RECEIPT_MARKER_FIELD == advisory.ADVISORY_MARKER_FIELD
    )

    assert controller.ADVISORY_OFFSETS_MINUTES == (180, 90, 45)

    assert (
        advisory.PRODUCTION_OFFSET_MINUTES
        == controller.PRODUCTION_OFFSET_MINUTES
    )

    for offset in controller.ADVISORY_OFFSETS_MINUTES:
        assert offset in CAPTURE_OFFSETS_MINUTES

        assert offset != controller.PRODUCTION_OFFSET_MINUTES

        assert offset in controller.ADVISORY_FEED_DESTINATIONS

        assert offset in controller.ADVISORY_PAGE_DESTINATIONS

        assert offset in controller.ADVISORY_PAGE_PREFIXES


def test_the_offsets_deliver_predictions_hours_before_tip():
    """What a reader actually gets, in the wall clock they read it in."""
    expected = {
        "7:00 PM": ("4:00 PM", "5:30 PM", "6:15 PM"),
        "9:30 PM": ("6:30 PM", "8:00 PM", "8:45 PM"),
        "10:00 PM": ("7:00 PM", "8:30 PM", "9:15 PM"),
    }

    for tip_label, wanted in expected.items():
        tip = datetime.strptime(
            f"2026-10-23 {tip_label}", "%Y-%m-%d %I:%M %p"
        ).replace(tzinfo=EASTERN)

        rows = controller.advisory_rows([game(101, tip.astimezone(UTC))])

        served = tuple(
            row["capture_at_utc"]
            .astimezone(EASTERN)
            .strftime("%I:%M %p")
            .lstrip("0")
            for row in rows
        )

        assert served == wanted, f"{tip_label} tip"

        # And every one of them is hours or tens of minutes earlier than the
        # T-20 window, which is the whole point.
        t20 = tip - timedelta(minutes=controller.PRODUCTION_OFFSET_MINUTES)

        for row in rows:
            assert row["capture_at_utc"] < t20


def segments_for(slate_date: str) -> list[tuple[datetime, datetime]]:
    """The three scheduling segments, as UTC intervals, for one slate.

    The 00:25 segment is the one after midnight UTC, so it belongs to the day
    after the slate date -- the same derivation the workflow documents and the
    reason the slate date is taken in America/New_York rather than from the
    UTC calendar.
    """
    day = datetime.fromisoformat(slate_date).date()

    spans = []

    for hour, minute in controller.SEGMENT_STARTS_UTC:
        start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)

        if (hour, minute) < controller.SUPPORTED_WINDOW_START_UTC:
            start += timedelta(days=1)

        spans.append((start, start + timedelta(hours=controller.SEGMENT_HOURS)))

    return sorted(spans)


def test_a_normal_slate_puts_every_advisory_window_inside_a_segment():
    """Tips from 7:00 PM to 10:30 PM ET, which is every NBA evening."""
    day = datetime.fromisoformat(SLATE).date()

    tips = [
        datetime(
            day.year, day.month, day.day, hour, minute, tzinfo=EASTERN
        ).astimezone(UTC)
        for hour, minute in (
            (19, 0),
            (19, 30),
            (20, 0),
            (21, 30),
            (22, 0),
            (22, 30),
        )
    ]

    games = [game(101 + index, tip) for index, tip in enumerate(tips)]

    rows = controller.advisory_rows(games)

    assert len(rows) == len(tips) * len(controller.ADVISORY_OFFSETS_MINUTES)

    start, end = controller.supported_window(SLATE)

    for row in rows:
        assert start <= row["capture_at_utc"] < end, (
            f"{row['window_id']} falls outside the supported production "
            "window, so no segment would be awake for it"
        )

    # Every window is owned by exactly one of the three segments, which is
    # what makes a fourth cron unnecessary as well as forbidden.
    for row in rows:
        owners = [
            index
            for index, (segment_start, segment_end) in enumerate(
                segments_for(SLATE)
            )
            if segment_start <= row["capture_at_utc"] < segment_end
        ]

        assert len(owners) == 1, f"{row['window_id']} is owned by {owners}"
