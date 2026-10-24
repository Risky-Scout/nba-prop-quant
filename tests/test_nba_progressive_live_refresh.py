"""Tests for the progressive live refresh: T-180, T-90, T-45, T-20.

Four properties are under test and each one is a claim about the live
destination rather than about this module's intentions, so each is asserted on
what the destination holds afterwards.

*The bytes are identical.* The live release is the release the canary
validated, and nothing here re-derives it. Three independent statements of one
rows digest have to agree -- the canary feed publication receipt, the
attestation read back off the canary destination, and the release built for
the live destination -- and each is broken separately to see the refusal it is
for.

*The chain only moves closer to tip.* T-180 establishes, the three after it
replace, T-20 is final, and a later window is never replaced by an earlier
one. The same window twice is idempotent and writes nothing at all.

*The prior valid release survives every failure.* A failure is injected at
every distinct stage -- no priced artifact, an unusable serving receipt, a
canary that did not publish, a canary destination with no attestation, an
attestation of other bytes, a release whose digest does not match, a game that
has already tipped, a transport that will not talk, an approval that is absent
or disagrees -- and in every case the live pointer is byte-identical to what it
was before and the release it names is the last one that validated.

*Nothing publishes without both approval variables.* The committed switch says
ENABLED and that is one of three conditions. With either variable absent or
disagreeing the refusal names both of them, the GitHub environment they belong
to and the values they must carry, nothing is written to any destination, and
there is no fallback to a canary: a run that published a preview while
reporting a live success is the one failure no later check could see.

The publisher is real throughout. The destinations are local directories in a
pytest temporary directory, the canary half runs against the committed switch
which enables canary, and only the live half is given a forged switch -- so
every gate the frozen publisher applies to a live publication is applied here
by the publisher itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "ops"))

sys.path.insert(0, str(PROJECT / "tests"))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import run_progressive_live_refresh as live  # noqa: E402
import run_wizardofodds_publication as publication  # noqa: E402
import wizardofodds_publisher_fixture as fixture  # noqa: E402

#: The approval this repository cannot grant itself, forged here so the gates
#: after it are the thing a test is about. Deliberately not the committed
#: token: a test that passed only because it reused the real one would be
#: testing the file rather than the gate.
TOKEN = "the-approval-this-test-forged"

#: The two games the fixture prices, and when they tip. Read off the fixture
#: so a window's tip and the row's own ``tip_off_utc`` cannot disagree.
TIPS: dict[int, str] = dict(fixture.TIP_OFF)

#: Five hours before the earlier tip, which is before every window of the
#: chain. Every refusal about time is then a refusal the test asked for.
WELL_BEFORE_TIP = "2026-10-20T18:00:00+00:00"


class Chain:
    """One slate, four canary destinations and one live destination.

    Built per test. The canary half is published by the real publication
    orchestrator against the committed switch; the live half is invoked
    through :func:`run_progressive_live_refresh.main`, as the controller
    invokes it, so what is tested includes the receipt it writes and the exit
    code it returns.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

        self.live_root = self.root / "live"

        self.switch_path = self.root / "switch.json"

        self.destinations_path = self.root / "destinations.json"

        self.token = TOKEN

        self.built: dict | None = None

        self._write_switch(publisher.STATE_ENABLED, TOKEN)

        self._write_destinations()

    # -- the forged live approval -------------------------------------

    def _write_switch(self, state: str, token: str) -> None:
        document = json.loads(
            (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
        )

        document["modes"][publisher.MODE_LIVE] = {
            "approval_token": token,
            "reason": "forged for this test only",
            "state": state,
        }

        self.switch_path.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _write_destinations(self) -> None:
        document = json.loads(
            (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(
                encoding="utf-8"
            )
        )

        document["destinations"][publisher.MODE_LIVE] = {
            "release_prefix": "nba_prop_quant",
            "root": str(self.live_root),
            "transport": publisher.TRANSPORT_LOCAL_DIRECTORY,
        }

        self.destinations_path.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )

    def approve(self, monkeypatch) -> None:
        monkeypatch.setenv(publisher.PUBLISH_ENV_VAR, publisher.STATE_ENABLED)

        monkeypatch.setenv(publisher.PUBLISH_APPROVAL_ENV_VAR, self.token)

    # -- the canary half ----------------------------------------------

    def canary_root(self, offset: int) -> Path:
        return self.root / f"canary_t{offset}"

    def serve(self, *, odds: int = -110) -> dict:
        """Price the slate again, with prices that differ from last time.

        One serve per window, which is what the real chain does: each window
        prices the market as it stands at that window, so each release has its
        own rows digest and a replacement is a real replacement rather than a
        rewrite of identical bytes.
        """
        frame = fixture.priced_frame()

        frame["over_odds"] = odds

        self.built = fixture.build(self.root / "slate", frame=frame)

        return self.built

    def canary(self, offset: int, *, odds: int | None = None) -> Path:
        """Publish this window's canary and return the receipt it wrote."""
        if odds is not None or self.built is None:
            self.serve(odds=-110 - offset if odds is None else odds)

        built = self.built

        receipt = publication.publish_feed(
            mode=publisher.MODE_CANARY,
            slate_date=built["slate_date"],
            data_root=built["data_root"],
            project_root=PROJECT,
            serving_receipt=built["receipt_path"],
            destination_name=None,
            destination_root=self.canary_root(offset),
        )

        path = self.root / f"canary_feed_t{offset}.json"

        path.write_text(
            json.dumps(receipt, default=str, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return path

    # -- the live half ------------------------------------------------

    def argv(
        self,
        offset: int,
        *,
        canary_receipt: Path | None = None,
        serving_receipt: Path | None = None,
        canary_root: Path | None = None,
        now: str = WELL_BEFORE_TIP,
        tips: dict[str, str] | None = None,
        receipt_path: Path | None = None,
    ) -> list[str]:
        built = self.built

        argv = [
            "--slate-date",
            built["slate_date"],
            "--data-root",
            str(built["data_root"]),
            "--snapshot-offset-minutes",
            str(offset),
            "--canary-feed-receipt",
            str(
                canary_receipt
                if canary_receipt is not None
                else self.root / f"canary_feed_t{offset}.json"
            ),
            "--serving-receipt",
            str(
                serving_receipt
                if serving_receipt is not None
                else built["receipt_path"]
            ),
            "--canary-destination-root",
            str(
                canary_root
                if canary_root is not None
                else self.canary_root(offset)
            ),
            "--live-destination-root",
            str(self.live_root),
            "--switch-path",
            str(self.switch_path),
            "--destinations-path",
            str(self.destinations_path),
            "--project-root",
            str(PROJECT),
            "--now-utc",
            now,
            "--receipt-path",
            str(
                receipt_path
                if receipt_path is not None
                else self.root / f"live_t{offset}.json"
            ),
        ]

        for window_id, tip in (
            tips
            if tips is not None
            else {
                f"{game_id}:T-{offset}m": tip
                for game_id, tip in TIPS.items()
            }
        ).items():
            argv += ["--window-tip-utc", f"{window_id}={tip}"]

        return argv

    def refresh(self, offset: int, **overrides) -> tuple[int, dict]:
        path = Path(
            overrides.get("receipt_path")
            or self.root / f"live_t{offset}.json"
        )

        code = live.main(self.argv(offset, **overrides))

        return code, json.loads(path.read_text(encoding="utf-8"))

    def window(self, offset: int, **overrides) -> tuple[int, dict]:
        """Canary this window and then refresh live from it."""
        self.canary(offset)

        return self.refresh(offset, **overrides)

    # -- what the destination holds -----------------------------------

    def pointer(self, name: str = publisher.CURRENT_POINTER) -> dict | None:
        path = self.live_root / name

        if not path.exists():
            return None

        return json.loads(path.read_text(encoding="utf-8"))

    def ledger(self) -> list[dict]:
        return live.read_evidence(
            live.evidence_path(self.built["data_root"], self.built["slate_date"])
        )


@pytest.fixture
def chain(tmp_path: Path, monkeypatch) -> Chain:
    built = Chain(tmp_path)

    built.approve(monkeypatch)

    return built


# ======================================================================
# the chain
# ======================================================================


def test_the_four_windows_establish_and_then_replace_one_live_release(chain):
    """T-180 establishes, T-90, T-45 and T-20 replace, T-20 is final.

    One live destination through the whole evening: one ``current.json``, one
    release id at a time, four refreshes. What a reader opening the page is
    shown at any point is the most recent window that validated.
    """
    outcomes: list[str] = []

    releases: list[str] = []

    for offset in live.PROGRESSIVE_OFFSETS_MINUTES:
        code, receipt = chain.window(offset)

        assert code == live.EXIT_OK, receipt

        outcomes.append(receipt["live_outcome"])

        releases.append(receipt["release"]["release_id"])

        # At every point the destination names this window's release and
        # nothing else.
        assert chain.pointer()["release_id"] == releases[-1]

    assert outcomes == [
        live.OUTCOME_ESTABLISHED,
        live.OUTCOME_REPLACED,
        live.OUTCOME_REPLACED,
        live.OUTCOME_REPLACED,
    ]

    # Four distinct releases of one stream, each one replacing the last.
    assert len(set(releases)) == 4

    for release_id in releases:
        assert release_id.startswith("nba_prop_quant__2026-10-20__")

    # And the last one says in words that it is the final pregame version.
    assert receipt["final_pregame_window"] is True

    assert receipt["offset_minutes"] == live.FINAL_PREGAME_OFFSET_MINUTES

    # The one before it did not.
    assert (
        json.loads(
            (chain.root / "live_t45.json").read_text(encoding="utf-8")
        )["final_pregame_window"]
        is False
    )


def test_the_live_release_is_the_release_the_canary_validated(chain):
    """The rows digest, stated three times and equal in all three.

    The canary feed receipt names what the canary published. The attestation
    read back off the canary destination names what that destination is
    actually serving. The live manifest names what was promoted. The rows
    payload carries the contract version, the rows, the distributions and the
    slate date and nothing about the destination, so the three are equal by
    construction -- which is why they are asserted rather than assumed.
    """
    canary_receipt = chain.canary(180)

    code, receipt = chain.refresh(180)

    assert code == live.EXIT_OK, receipt

    published = json.loads(canary_receipt.read_text(encoding="utf-8"))

    attestation = json.loads(
        (chain.canary_root(180) / publisher.ATTESTATION_NAME).read_text(
            encoding="utf-8"
        )
    )

    manifest = json.loads(
        (chain.live_root / receipt["release"]["manifest_relative"]).read_text(
            encoding="utf-8"
        )
    )

    digest = published["release"]["rows_sha256"]

    assert digest

    for named in (
        attestation["rows_sha256"],
        manifest["rows_sha256"],
        receipt["release"]["rows_sha256"],
        chain.pointer()["rows_sha256"],
        receipt["canary"]["rows_sha256"],
        receipt["canary_attestation"]["rows_sha256"],
    ):
        assert named == digest

    # The same for the artifact behind them, which is the equality the
    # publisher's own canary gate re-checks against the parquet on disk.
    assert (
        attestation["source_artifact_sha256"]
        == published["provenance"]["priced_market_artifact_sha256"]
    )

    # The release *id* does differ, and that is the point: one set of rows,
    # published under the live stream's own name.
    assert receipt["release"]["release_id"] != published["release"]["release_id"]

    assert receipt["release"]["release_id"].endswith(digest[:16])

    # And the rows document at the live destination is byte-identical to the
    # one at the canary, which is the claim "the same already-validated
    # release" makes in the strongest available form.
    assert (
        chain.live_root / receipt["release"]["rows_relative"]
    ).read_bytes() == (
        chain.canary_root(180)
        / published["release"]["rows_relative"]
    ).read_bytes()


def test_the_live_page_stage_reads_the_refresh_receipt_unchanged(chain):
    """The page a reader opens is pinned to the release that went live.

    ``run_wizardofodds_publication.pin_from_feed_receipt`` is pinned
    syntax-tree-identical to production and may not learn a new word, so the
    refresh writes its receipt in the publication receipt's own shape and the
    live page stage consumes it with no change to that file at all.
    """
    code, receipt = chain.window(20)

    assert code == live.EXIT_OK, receipt

    pinned = publication.pin_from_feed_receipt(
        chain.root / "live_t20.json",
        slate_date=chain.built["slate_date"],
        mode=publisher.MODE_LIVE,
    )

    assert pinned.release_id == receipt["release"]["release_id"]

    assert pinned.rows_sha256 == receipt["release"]["rows_sha256"]

    # The progressive verdict is carried beside the publication word rather
    # than in place of it, so neither reader has to know about the other.
    assert receipt["outcome"] == publication.OUTCOME_PUBLISHED

    assert receipt["live_outcome"] == live.OUTCOME_ESTABLISHED

    assert receipt["mode"] == publisher.MODE_LIVE

    assert receipt["contract_version"] == publisher.PUBLICATION_CONTRACT_VERSION


# ======================================================================
# ordering and idempotency
# ======================================================================


def test_an_earlier_window_never_replaces_a_later_one(chain):
    """The refusal that keeps a late T-180 from walking the page back.

    Reachable on this repository's runner, where pickup has been delayed by
    hours: a T-180 invocation can finish after a T-45 one. What it would
    publish is a real validated release with every digest agreeing and every
    readback passing. It is simply the wrong one.
    """
    chain.window(180)

    chain.window(90)

    before = chain.pointer()

    before_previous = chain.pointer(publisher.PREVIOUS_POINTER)

    stale = chain.root / "canary_feed_t180.json"

    assert stale.exists()

    code, receipt = chain.refresh(
        180,
        canary_receipt=stale,
        receipt_path=chain.root / "backwards.json",
    )

    assert code == live.EXIT_REFUSED

    assert receipt["live_outcome"] == live.OUTCOME_REFUSED

    assert "T-90 release is already live" in receipt["message"]

    assert "never replaced by an earlier one" in receipt["message"]

    # Nothing moved, in either pointer.
    assert chain.pointer() == before

    assert chain.pointer(publisher.PREVIOUS_POINTER) == before_previous


def test_the_same_window_twice_writes_nothing_at_all(chain):
    """Idempotent, and deliberately not a republication.

    Rewriting byte-identical release files would change nothing a reader sees
    and would put a second audit entry against one release, so the refresh
    reports what is current and touches neither.
    """
    chain.window(20)

    before = chain.pointer()

    audit = (chain.live_root / publisher.AUDIT_LOG_NAME).read_text(
        encoding="utf-8"
    )

    releases = sorted(
        path.name
        for path in (chain.live_root / publisher.RELEASES_DIRECTORY).iterdir()
    )

    code, receipt = chain.refresh(20)

    assert code == live.EXIT_OK, receipt

    assert receipt["live_outcome"] == live.OUTCOME_ALREADY_CURRENT

    assert receipt["release"]["release_id"] == before["release_id"]

    assert chain.pointer() == before

    assert (chain.live_root / publisher.AUDIT_LOG_NAME).read_text(
        encoding="utf-8"
    ) == audit

    assert sorted(
        path.name
        for path in (chain.live_root / publisher.RELEASES_DIRECTORY).iterdir()
    ) == releases

    # The ordering read says why in words, so a same-window refresh is not
    # confused with a repeat of one that already happened.
    assert "the same window" in receipt["ordering"]


def test_a_later_group_in_the_same_window_replaces_the_earlier_one(chain):
    """A cumulative release grows within one window, and may replace itself.

    One segment that captures three tip groups serves and publishes three
    times at T-20, each release holding every game priced so far, so equality
    of the offset has to be allowed where a smaller offset is required
    elsewhere.
    """
    chain.window(20)

    first = chain.pointer()["release_id"]

    # The same window, more rows priced: a different digest and a real
    # replacement.
    chain.canary(20, odds=-125)

    code, receipt = chain.refresh(20)

    assert code == live.EXIT_OK, receipt

    assert receipt["live_outcome"] == live.OUTCOME_REPLACED

    assert receipt["previous_release_id"] == first

    assert chain.pointer()["release_id"] == receipt["release"]["release_id"]

    assert chain.pointer(publisher.PREVIOUS_POINTER)["release_id"] == first


def test_a_window_outside_the_chain_has_nothing_that_could_validate_it(chain):
    """T-1440, T-480 and T-5 have no canary destination and so no attestation."""
    chain.canary(180)

    for offset in (1440, 480, 5):
        code, receipt = chain.refresh(
            offset,
            canary_receipt=chain.root / "canary_feed_t180.json",
            canary_root=chain.canary_root(180),
            tips={f"9901:T-{offset}m": TIPS[9901]},
            receipt_path=chain.root / f"outside_t{offset}.json",
        )

        assert code == live.EXIT_REFUSED

        assert "is not one of the progressive live windows" in (
            receipt["message"]
        )

        assert chain.pointer() is None


# ======================================================================
# the approval this repository cannot grant itself
# ======================================================================


def test_the_committed_switch_is_enabled_and_opens_nothing(monkeypatch):
    """The decision is taken, and the gate is not open.

    ``modes.live.state`` is ENABLED in the committed file, which is the first
    of the three conditions that file names and the only one a commit can
    hold. The other two are variables in the ``wizardofodds-production``
    GitHub environment. Read with the environment a runner without them has,
    the switch is enabled and refused.
    """
    document = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    declared = document["modes"][publisher.MODE_LIVE]

    assert declared["state"] == publisher.STATE_ENABLED

    assert declared["approval_token"]

    assert declared["approval_token"] != "no-approval-has-been-granted"

    # The file still says what is required, and it still takes three things.
    assert len(document["what_enabling_live_requires"]) == 3

    for required in document["what_enabling_live_requires"]:
        assert isinstance(required, str) and required

    switch = publisher.read_publishing_switch(
        publisher.MODE_LIVE, project_root=PROJECT, environment={}
    )

    assert switch.state == publisher.STATE_ENABLED

    assert switch.enabled is False

    assert switch.candidate_authority == "shadow_only"

    assert switch.published_authority == publisher.MODEL_AUTHORITY


@pytest.mark.parametrize(
    "carried",
    [
        pytest.param({}, id="neither variable"),
        pytest.param(
            {publisher.PUBLISH_ENV_VAR: publisher.STATE_ENABLED},
            id="the state without the token",
        ),
        pytest.param(
            {publisher.PUBLISH_APPROVAL_ENV_VAR: TOKEN},
            id="the token without the state",
        ),
        pytest.param(
            {
                publisher.PUBLISH_ENV_VAR: publisher.STATE_ENABLED,
                publisher.PUBLISH_APPROVAL_ENV_VAR: "a-different-token",
            },
            id="a token that disagrees",
        ),
        pytest.param(
            {
                publisher.PUBLISH_ENV_VAR: "enabled",
                publisher.PUBLISH_APPROVAL_ENV_VAR: TOKEN,
            },
            id="the state in the wrong case",
        ),
    ],
)
def test_a_refresh_without_both_variables_refuses_and_names_them(
    tmp_path: Path, monkeypatch, carried: dict
):
    """Fail closed, loudly, and never at a lower privilege.

    The refusal has to say what to do rather than only what is wrong, so it
    names both variables, the exact value each must carry and the GitHub
    environment they belong to. And there is no fallback: no mode argument, no
    second destination and no retry as a canary, because a run that published
    a preview while reporting a live success would be the one failure no later
    check could see.
    """
    chain = Chain(tmp_path)

    for variable in (
        publisher.PUBLISH_ENV_VAR,
        publisher.PUBLISH_APPROVAL_ENV_VAR,
    ):
        monkeypatch.delenv(variable, raising=False)

    for variable, value in carried.items():
        monkeypatch.setenv(variable, value)

    chain.canary(180)

    code, receipt = chain.refresh(180)

    assert code == live.EXIT_REFUSED

    assert receipt["live_outcome"] == live.OUTCOME_REFUSED

    message = receipt["message"]

    for named in (
        publisher.PUBLISH_ENV_VAR,
        publisher.PUBLISH_APPROVAL_ENV_VAR,
        live.PRODUCTION_ENVIRONMENT,
        publisher.STATE_ENABLED,
        "modes.live.approval_token",
    ):
        assert named in message, named

    # Said in words, because the alternative is the failure nobody sees.
    assert "Nothing is published to a canary in their place" in message

    # And nothing was written anywhere live.
    assert chain.pointer() is None

    assert not chain.live_root.exists() or sorted(
        path.name for path in chain.live_root.iterdir()
    ) == []

    # The receipt is a refusal in the publication vocabulary too, so a page
    # stage handed it refuses rather than building from whatever was there.
    assert receipt["outcome"] != publication.OUTCOME_PUBLISHED

    with pytest.raises(publication.PublicationStageFailed):
        publication.pin_from_feed_receipt(
            chain.root / "live_t180.json",
            slate_date=chain.built["slate_date"],
            mode=publisher.MODE_LIVE,
        )


def test_a_disabled_switch_refuses_before_the_environment_is_consulted(
    tmp_path: Path, monkeypatch
):
    """Two different failures, distinguished because the action differs.

    DISABLED in the commit means the reviewed decision has not been taken.
    ENABLED with the environment silent means it has been taken and has not
    reached the runner, which is a misconfiguration of a decision rather than
    the absence of one.
    """
    chain = Chain(tmp_path)

    chain._write_switch(publisher.STATE_DISABLED, TOKEN)

    chain.approve(monkeypatch)

    chain.canary(180)

    code, receipt = chain.refresh(180)

    assert code == live.EXIT_REFUSED

    assert "live publishing is DISABLED" in receipt["message"]

    assert "the reviewed decision to publish has not been taken" in (
        receipt["message"]
    )

    assert chain.pointer() is None


# ======================================================================
# the prior valid release survives every failure
# ======================================================================


def seeded(chain: Chain) -> dict:
    """A live destination serving a validated T-90 release.

    The state every retention test needs: something valid is live, and the
    question is what a later failure does to it.
    """
    chain.window(180)

    chain.window(90)

    return chain.pointer()


@pytest.fixture
def served(chain: Chain) -> tuple[Chain, dict, dict, str]:
    current = seeded(chain)

    previous = chain.pointer(publisher.PREVIOUS_POINTER)

    audit = (chain.live_root / publisher.AUDIT_LOG_NAME).read_text(
        encoding="utf-8"
    )

    return chain, current, previous, audit


def assert_untouched(
    chain: Chain, current: dict, previous: dict, audit: str
) -> None:
    """The live pointer, its rollback target and the audit log are unchanged.

    All three, because "the prior valid release is retained" is a claim about
    what a reader gets *and* about still having somewhere to go back to. A
    refusal that left the pointer alone and consumed the rollback target
    would satisfy the first and lose the second.
    """
    assert chain.pointer() == current

    assert chain.pointer(publisher.PREVIOUS_POINTER) == previous

    assert (chain.live_root / publisher.AUDIT_LOG_NAME).read_text(
        encoding="utf-8"
    ) == audit

    # And the release the pointer names is still readable, which is what
    # makes it a retained release rather than a retained name.
    rows = chain.live_root / current["rows_relative"]

    assert rows.exists()

    assert publisher.digest(rows.read_bytes()) == current["rows_sha256"]


def test_a_missing_priced_artifact_leaves_the_last_valid_release_serving(
    served,
):
    """Stage one: nothing was served, so there is nothing to publish."""
    chain, current, previous, audit = served

    chain.canary(45)

    chain.built["priced_path"].unlink()

    code, receipt = chain.refresh(45)

    assert code == live.EXIT_REFUSED

    assert "does not exist" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_an_unusable_serving_receipt_leaves_the_last_valid_release_serving(
    served,
):
    """Stage two: the rows on disk carry no provenance."""
    chain, current, previous, audit = served

    chain.canary(45)

    broken = chain.root / "broken_serving.json"

    broken.write_text("{ not json", encoding="utf-8")

    code, receipt = chain.refresh(45, serving_receipt=broken)

    assert code == live.EXIT_REFUSED

    assert_untouched(chain, current, previous, audit)


def test_a_canary_that_did_not_publish_authorises_nothing(served):
    """Stage three: no canary, so no live release.

    A canary stage that did not publish is not a licence to publish live from
    whatever release happened to be current, which is the ordering mistake
    the page stage already prevents, read one stage further along.
    """
    chain, current, previous, audit = served

    receipt_path = chain.canary(45)

    document = json.loads(receipt_path.read_text(encoding="utf-8"))

    document["outcome"] = publication.OUTCOME_NOT_SERVED

    receipt_path.write_text(json.dumps(document), encoding="utf-8")

    code, receipt = chain.refresh(45)

    assert code == live.EXIT_REFUSED

    assert publication.OUTCOME_NOT_SERVED in receipt["message"]

    assert "has no validated canary release" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_a_canary_of_another_window_authorises_nothing(served):
    """Stage three again: the right receipt, the wrong window.

    The canary feed receipt names the destination it published to, and a
    canary of another window's prices does not validate this window's release
    however well its digests agree with themselves.
    """
    chain, current, previous, audit = served

    chain.canary(45)

    document = json.loads(
        (chain.root / "canary_feed_t45.json").read_text(encoding="utf-8")
    )

    document["destination"]["name"] = live.CANARY_FEED_DESTINATIONS[180]

    (chain.root / "canary_feed_t45.json").write_text(
        json.dumps(document), encoding="utf-8"
    )

    code, receipt = chain.refresh(45)

    assert code == live.EXIT_REFUSED

    assert live.CANARY_FEED_DESTINATIONS[180] in receipt["message"]

    assert "does not validate this window's release" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_a_canary_destination_with_no_attestation_authorises_nothing(served):
    """Stage four: the destination has no statement about what it serves.

    Read back off the canary host rather than taken from a caller, because
    "the canary validated these bytes" has to be a fact about that
    destination and not about this run's temporary directory.
    """
    chain, current, previous, audit = served

    chain.canary(45)

    (chain.canary_root(45) / publisher.ATTESTATION_NAME).unlink()

    code, receipt = chain.refresh(45)

    assert code == live.EXIT_REFUSED

    assert publisher.ATTESTATION_NAME in receipt["message"]

    assert "nothing goes live" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("rows_sha256", "0" * 64, id="a rows digest that differs"),
        pytest.param(
            "source_artifact_sha256", "0" * 64, id="an artifact that differs"
        ),
        pytest.param("release_id", "something-else", id="another release"),
        pytest.param("verified", False, id="no verified readback"),
        pytest.param("outcome", "PUBLISHING_REFUSED", id="a failed canary"),
        pytest.param("mode", publisher.MODE_LIVE, id="not a canary at all"),
    ],
)
def test_an_attestation_that_is_not_about_these_bytes_authorises_nothing(
    served, field: str, value
):
    """Stage four again, broken one field at a time.

    Each of these is a different mistake and each is refused separately. The
    rows digest says the canary has not been overtaken by another window
    between the two stages. The artifact digest says that canary ran on the
    priced file this run produced. The release id says the attestation is
    about this run's canary rather than a republication of the same rows. And
    the three state checks say a canary happened, succeeded and proved its own
    destination held what was sent.
    """
    chain, current, previous, audit = served

    chain.canary(45)

    path = chain.canary_root(45) / publisher.ATTESTATION_NAME

    attestation = json.loads(path.read_text(encoding="utf-8"))

    attestation[field] = value

    path.write_text(json.dumps(attestation), encoding="utf-8")

    code, receipt = chain.refresh(45)

    assert code == live.EXIT_REFUSED

    assert_untouched(chain, current, previous, audit)


def test_a_release_whose_digest_moved_is_never_promoted(served, monkeypatch):
    """Stage five: the bytes changed between validation and publication.

    It cannot normally happen -- the rows payload is a function of the priced
    parquet, the distributions and the slate date, and the publisher re-hashes
    the parquet against the serving receipt before it builds anything -- which
    is exactly why it is asserted rather than assumed. The only ways it could
    differ are the ways that matter, and all of them end with a reader being
    shown numbers no canary read back.
    """
    chain, current, previous, audit = served

    chain.canary(45)

    real = publisher.build_release

    def drifted(**arguments):
        release = real(**arguments)

        return release.__class__(
            **{
                **{
                    field: getattr(release, field)
                    for field in release.__dataclass_fields__
                },
                "rows_sha256": "9" * 64,
            }
        )

    monkeypatch.setattr(publisher, "build_release", drifted)

    code, receipt = chain.refresh(45)

    assert code == live.EXIT_REFUSED

    assert "9" * 64 in receipt["message"]

    assert "bytes no canary validated" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_a_transport_that_will_not_talk_leaves_the_pointer_alone(
    served, monkeypatch
):
    """Stage six: the destination is unreachable.

    The failure that happens for reasons outside this repository, and the one
    a retention claim is least able to be careful about: whatever the
    transport does, the promotion has not happened yet.
    """
    chain, current, previous, audit = served

    chain.canary(45)

    real = publisher.LocalDirectoryTransport.write_bytes

    def refuse(self, relative, payload):
        raise OSError(f"the destination refused {relative}")

    monkeypatch.setattr(
        publisher.LocalDirectoryTransport, "write_bytes", refuse
    )

    code, receipt = chain.refresh(45)

    assert code == live.EXIT_REFUSED

    monkeypatch.setattr(
        publisher.LocalDirectoryTransport, "write_bytes", real
    )

    assert_untouched(chain, current, previous, audit)


def test_a_window_whose_game_has_tipped_is_never_published_live(served):
    """Stage seven: the guard that did not exist anywhere.

    On 2026-10-23 the T-20 group holding game 21717941 -- CHI at MIL -- was
    captured, served and published two hours and forty minutes after that game
    tipped, because the capture grace is about when a snapshot may be taken
    rather than about when a price may be published. At a canary that is a
    stale preview nobody reads. At the served root it is a pregame price on a
    game in its third quarter.
    """
    chain, current, previous, audit = served

    chain.canary(45)

    code, receipt = chain.refresh(
        45, now="2026-10-21T02:00:00+00:00"
    )

    assert code == live.EXIT_REFUSED

    assert "have already started" in receipt["message"]

    assert "is not a pregame price and does not go live" in receipt["message"]

    assert "minutes ago" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_a_refresh_with_no_window_tip_at_all_is_refused(served):
    """The guard is a property of the caller's schedule read, not a default.

    A default would make the post-tip claim something this file asserts about
    itself, which is the shape of guarantee that is true until the day it
    matters.
    """
    chain, current, previous, audit = served

    chain.canary(45)

    code, receipt = chain.refresh(45, tips={})

    assert code == live.EXIT_REFUSED

    assert "no --window-tip-utc was supplied" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_a_tip_for_the_wrong_window_is_refused(served):
    """A guard checking a window this release is not being refreshed for."""
    chain, current, previous, audit = served

    chain.canary(45)

    code, receipt = chain.refresh(
        45, tips={"9901:T-20m": TIPS[9901], "9902:T-45m": TIPS[9902]}
    )

    assert code == live.EXIT_REFUSED

    assert "9901:T-20m" in receipt["message"]

    assert "which are not :T-45m windows" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_a_game_that_has_tipped_in_an_earlier_group_does_not_block_a_window(
    chain,
):
    """The guard's subject is the window, not the release.

    A release is cumulative -- one serve at an offset prices every game with
    an eligible capture at that offset -- so by the time the last T-20 group
    of a twelve-game slate is published, the first game tipped two hours
    earlier and its rows are correctly in the release because they were
    priced before their own tip.
    """
    chain.canary(20)

    # The second game has not tipped; the first tipped an hour ago, and its
    # rows are in this release because they were priced before it did.
    code, receipt = chain.refresh(
        20,
        now="2026-10-21T00:00:00+00:00",
        tips={"9902:T-20m": TIPS[9902]},
    )

    assert code == live.EXIT_OK, receipt

    assert receipt["live_outcome"] == live.OUTCOME_ESTABLISHED

    assert [window["window_id"] for window in receipt["windows"]] == [
        "9902:T-20m"
    ]

    assert receipt["windows"][0]["minutes_before_tip"] == 90.0


# ======================================================================
# the first live release
# ======================================================================


def test_the_first_release_seeds_its_own_rollback_target(chain):
    """Day one, and the gate that would otherwise make it unreachable.

    ``assert_rollback_target_exists`` refuses when the live root has no
    ``current.json``, and it is right to: an activation must never be the step
    that leaves live with nowhere to go back to. On a destination that has
    never served anything there is nothing to seed it with, so the first
    release of the chain seeds itself -- both pointers naming it, which is the
    truthful state of a destination holding one release.
    """
    assert chain.pointer() is None

    code, receipt = chain.window(180)

    assert code == live.EXIT_OK, receipt

    assert receipt["live_outcome"] == live.OUTCOME_ESTABLISHED

    release_id = receipt["release"]["release_id"]

    assert chain.pointer()["release_id"] == release_id

    assert chain.pointer(publisher.PREVIOUS_POINTER)["release_id"] == (
        release_id
    )

    # Every record a published release leaves is there, written by the
    # publisher's own functions.
    audit = [
        json.loads(line)
        for line in (chain.live_root / publisher.AUDIT_LOG_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [entry["action"] for entry in audit] == [publisher.ACTION_PUBLISHED]

    assert audit[0]["mode"] == publisher.MODE_LIVE

    assert audit[0]["model_authority"] == publisher.MODEL_AUTHORITY

    assert audit[0]["rows_sha256"] == receipt["release"]["rows_sha256"]

    assert receipt["verification"]["rows_sha256"] == (
        receipt["release"]["rows_sha256"]
    )

    # And from here on the publisher's own gate binds, unweakened: the next
    # refresh goes through publish() and finds a rollback target.
    code, replaced = chain.window(90)

    assert code == live.EXIT_OK, replaced

    assert replaced["live_outcome"] == live.OUTCOME_REPLACED

    assert replaced["promoted"]["release_id"] == replaced["release"][
        "release_id"
    ]


def test_the_establishment_path_is_reachable_exactly_once(chain):
    """It refuses the moment the destination is serving anything."""
    chain.window(180)

    transport = publisher.LocalDirectoryTransport(chain.live_root)

    try:
        with pytest.raises(live.LiveRefreshRefused) as error:
            live.establish_first_live_release(
                transport,
                release=None,
                slate_date=chain.built["slate_date"],
                provenance={},
            )
    finally:
        transport.close()

    assert publisher.CURRENT_POINTER in str(error.value)

    assert "the publisher's own rollback gate" in str(error.value)


# ======================================================================
# the durable record
# ======================================================================


def test_every_attempt_leaves_one_line_naming_what_it_did(chain):
    """Append-only, one line per attempt, refusals included.

    This is the only record of *which capture window* is live: the published
    rows cannot carry the offset because ``PUBLISHED_FIELDS`` is pinned, and
    ``current.json`` names a release id whose prefix is the live stream's
    rather than the window's. So a reader of one line knows the window, the
    offset, the release, the digests, the canary that validated them, what was
    live before and the outcome.
    """
    for offset in live.PROGRESSIVE_OFFSETS_MINUTES:
        chain.window(offset)

    # One refusal, which is recorded like everything else.
    chain.refresh(
        180,
        canary_receipt=chain.root / "canary_feed_t180.json",
        receipt_path=chain.root / "backwards.json",
    )

    entries = chain.ledger()

    assert [entry["offset_minutes"] for entry in entries] == [
        180,
        90,
        45,
        20,
        180,
    ]

    assert [entry["outcome"] for entry in entries] == [
        live.OUTCOME_ESTABLISHED,
        live.OUTCOME_REPLACED,
        live.OUTCOME_REPLACED,
        live.OUTCOME_REPLACED,
        live.OUTCOME_REFUSED,
    ]

    # The refusal names no release, which is what lets the ordering read pass
    # over refusals without having to interpret them.
    assert entries[-1]["live_release_id"] == ""

    assert live.last_live_entry(entries)["offset_minutes"] == 20

    published = entries[3]

    assert published["final_pregame_window"] is True

    assert published["mode"] == publisher.MODE_LIVE

    assert published["model_authority"] == publisher.MODEL_AUTHORITY

    assert published["feed_destination"] == live.LIVE_FEED_DESTINATION

    assert published["live_release_id"] == chain.pointer()["release_id"]

    assert published["rows_sha256"] == chain.pointer()["rows_sha256"]

    assert published["previous_release_id"] == entries[2]["live_release_id"]

    assert published["canary"]["canary_destination"] == (
        live.CANARY_FEED_DESTINATIONS[20]
    )

    assert published["canary"]["rows_sha256"] == published["rows_sha256"]

    assert published["canary_attestation"]["rows_sha256"] == (
        published["rows_sha256"]
    )

    assert [window["window_id"] for window in published["windows"]] == [
        "9901:T-20m",
        "9902:T-20m",
    ]

    # Its own tree, beside the witness ledgers and not inside one: this file
    # decides something and they decide nothing.
    assert live.LIVE_EVIDENCE_RELATIVE == Path("processed") / (
        "live_progressive_refresh"
    )

    assert live.evidence_path(
        chain.built["data_root"], chain.built["slate_date"]
    ) == (
        chain.built["data_root"]
        / "processed"
        / "live_progressive_refresh"
        / "2026-10-20.jsonl"
    )


def test_a_record_that_cannot_be_read_refuses_rather_than_reads_as_empty(
    chain,
):
    """A truncated line must not read as "nothing is live".

    Which is the state in which any window may go live, so carrying on past a
    hole in this file would turn a crash into a licence to walk the live page
    backwards.
    """
    chain.window(180)

    chain.window(90)

    current = chain.pointer()

    previous = chain.pointer(publisher.PREVIOUS_POINTER)

    audit = (chain.live_root / publisher.AUDIT_LOG_NAME).read_text(
        encoding="utf-8"
    )

    path = live.evidence_path(
        chain.built["data_root"], chain.built["slate_date"]
    )

    path.write_text(
        path.read_text(encoding="utf-8") + '{"offset_minutes": 4',
        encoding="utf-8",
    )

    chain.canary(45)

    code, receipt = chain.refresh(45)

    assert code == live.EXIT_REFUSED

    assert "is not readable JSON" in receipt["message"]

    assert "no refresh can be shown to be going forward" in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_a_record_that_disagrees_with_the_destination_refuses(chain):
    """Two authorities, and neither may be believed over the other.

    The pointer at the destination is the authority on *what* is live and the
    record is the authority on *which window* it came from, so they have to
    agree before anything is replaced. A disagreement means the answer to
    "which window is live" is unknown, and a refresh that cannot tell which
    window is live cannot tell whether it is going forwards.
    """
    chain.window(180)

    current = chain.pointer()

    previous = chain.pointer(publisher.PREVIOUS_POINTER)

    audit = (chain.live_root / publisher.AUDIT_LOG_NAME).read_text(
        encoding="utf-8"
    )

    path = live.evidence_path(
        chain.built["data_root"], chain.built["slate_date"]
    )

    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    entries[-1]["live_release_id"] = "something-nobody-published"

    path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )

    chain.canary(90)

    code, receipt = chain.refresh(90)

    assert code == live.EXIT_REFUSED

    assert "something other than this chain changed what is live" in (
        receipt["message"]
    )

    # The refusal names the destination's own log, so reconciling is a read.
    assert publisher.AUDIT_LOG_NAME in receipt["message"]

    assert_untouched(chain, current, previous, audit)


def test_a_destination_serving_this_slate_with_no_record_refuses(chain):
    """The other direction of the same disagreement."""
    chain.window(180)

    current = chain.pointer()

    live.evidence_path(
        chain.built["data_root"], chain.built["slate_date"]
    ).unlink()

    chain.canary(90)

    code, receipt = chain.refresh(90)

    assert code == live.EXIT_REFUSED

    assert "records no live publication for that slate" in receipt["message"]

    assert chain.pointer() == current


def test_a_release_from_an_earlier_slate_is_replaced_by_any_window(chain):
    """Yesterday's release is live, so today's first window establishes today.

    Which is the real state of the destination every evening: ``current.json``
    names the previous slate's final pregame release, the ordering rule has no
    window of today's to compare against, and the first window of the evening
    may therefore be any of the four.
    """
    chain.window(20)

    yesterday = chain.pointer()["release_id"]

    # A new slate on the same destination, which is what the next evening is.
    chain.built = fixture.build(
        chain.root / "tomorrow", slate_date="2026-10-21"
    )

    receipt_path = chain.canary(180)

    code, receipt = chain.refresh(
        180,
        canary_receipt=receipt_path,
        now="2026-10-21T18:00:00+00:00",
        tips={"9901:T-180m": "2026-10-21T23:00:00+00:00"},
    )

    assert code == live.EXIT_OK, receipt

    assert receipt["live_outcome"] == live.OUTCOME_REPLACED

    assert receipt["previous_release_id"] == yesterday

    assert "establishes the live release" in receipt["live_state_before"][
        "reason"
    ]

    assert chain.pointer()["release_id"] == receipt["release"]["release_id"]

    # Yesterday's release is the rollback target, which is the whole of what
    # "the prior valid release is retained" buys at a slate boundary.
    assert chain.pointer(publisher.PREVIOUS_POINTER)["release_id"] == yesterday


# ======================================================================
# what this module may not do
# ======================================================================


def test_the_refresh_names_the_single_live_pair_and_refuses_anything_else(
    chain,
):
    """One live feed and one live page, checked rather than trusted."""
    assert live.LIVE_FEED_DESTINATION == "wizardofodds_nba_live"

    assert live.LIVE_PAGE_DESTINATION == (
        "wizardofodds_nba_player_props_page_live"
    )

    for argument, wrong in (
        ("--live-feed-destination-name", live.CANARY_FEED_DESTINATIONS[20]),
        ("--live-page-destination-name", "wizardofodds_nba_live"),
        ("--canary-destination-name", live.CANARY_FEED_DESTINATIONS[90]),
    ):
        chain.canary(180)

        argv = chain.argv(180, receipt_path=chain.root / "wrong.json")

        argv[argv.index(argument) + 1] = wrong

        assert live.main(argv) == live.EXIT_REFUSED

        message = json.loads(
            (chain.root / "wrong.json").read_text(encoding="utf-8")
        )["message"]

        assert wrong in message

        assert chain.pointer() is None


def test_the_live_destinations_are_confined_to_the_served_root():
    """A live destination publishes the served root and nothing else.

    Asserted on the committed configuration, through the publisher's own
    confinement check, because the subpath rule is what separates the two
    modes: a live destination resolving below its served root would publish a
    page no reader fetches while reporting a successful activation.
    """
    document = publisher.read_destinations(project_root=PROJECT)

    for name in (live.LIVE_FEED_DESTINATION, live.LIVE_PAGE_DESTINATION):
        destination = publisher.resolve_destination(
            publisher.MODE_LIVE, document, name=name
        )

        assert destination.mode == publisher.MODE_LIVE

        publisher.assert_live_destination_is_the_served_root(destination)

        assert destination.remote is not None

        assert destination.remote.published_root.rstrip("/") == (
            destination.remote.public_target.rstrip("/")
        )

    # And every canary of the chain is confined to its own subpath, so no
    # canary of any window can address the live pointer.
    live_roots = {
        publisher.resolve_destination(
            publisher.MODE_LIVE, document, name=name
        ).remote.published_root.rstrip("/")
        for name in (live.LIVE_FEED_DESTINATION, live.LIVE_PAGE_DESTINATION)
    }

    for name in live.CANARY_FEED_DESTINATIONS.values():
        canary = publisher.resolve_destination(
            publisher.MODE_CANARY, document, name=name
        )

        root = canary.remote.published_root.rstrip("/")

        for served in live_roots:
            assert root != served

            assert not root.startswith(served + "/") or root != served


def test_this_module_computes_no_number_and_promotes_no_candidate():
    """It carries bytes somebody else produced and validated.

    Asserted on the source rather than described, because the claim the whole
    chain rests on is that a live publication is a transport decision and not
    a modelling one.
    """
    source = (PROJECT / "ops" / "run_progressive_live_refresh.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "predict",
        "calibrat",
        "fit(",
        "train",
        "promote_candidate",
        "adaptive",
        "sklearn",
        "numpy",
        "edge",
        "expected_value",
    ):
        assert forbidden not in source, forbidden

    # The candidate authority is read off the switch by the publisher and has
    # to be shadow_only, which this module cannot change.
    switch = publisher.read_publishing_switch(
        publisher.MODE_LIVE, project_root=PROJECT, environment={}
    )

    assert switch.candidate_authority == publisher.CANDIDATE_AUTHORITY

    assert switch.candidate_authority == "shadow_only"

    # pandas is imported inside the one function that reads the parquet and
    # nowhere else, so nothing at module scope can reach a frame.
    assert source.count("import pandas") == 1
