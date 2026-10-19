"""The incumbent WizardOfOdds publisher, its canary and its rollback.

Sixteen properties are required of this change, and they divide into three
kinds. Most are refusals: the publisher declining to write when the switch,
the approval, the provenance, the authority or the destination is not what it
has to be. A few are about the bytes: that the same rows produce the same
release, that a rerun is a no-op, that a verified release is the only kind
that gets promoted, and that a rollback restores the previous one. The last
is a statement about this change rather than about the publisher: that no
model, prediction or pricing code moved.

The refusal tests each break exactly one thing in a fixture that otherwise
publishes, which is the only way a refusal test says anything: a test whose
input would have been refused for three reasons proves nothing about the one
it names.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from nba_prop_quant.research.game_latent_state.safety import (
    DECLARABLE_SERVING_SOURCE_REGISTRIES,
)

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import run_incumbent_production_serving as serving  # noqa: E402
import wizardofodds_publisher_fixture as fixture  # noqa: E402
from serving_additive_pin import assert_numerically_unchanged  # noqa: E402

PRODUCTION_REF = "production/wizardofodds-integration"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def blob_text(ref: str, relative: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{relative}"],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def production_base() -> str | None:
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return git("merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue
    return None


def publish(built: dict, **overrides) -> dict:
    """The canary, over a built fixture, with the committed switch."""
    arguments = {
        "mode": publisher.MODE_CANARY,
        "slate_date": built["slate_date"],
        "data_root": built["data_root"],
        "project_root": PROJECT,
        "destination_root": built["destination_root"],
    }
    arguments.update(overrides)
    return publisher.publish(**arguments)


# ======================================================================
# 1, 2 and 3. the switch and the approval
# ======================================================================


def test_the_committed_switch_disables_live_publishing():
    """The state this change is required to leave behind, read off disk."""
    document = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    assert document["modes"]["live"]["state"] == publisher.STATE_DISABLED
    assert document["published_authority"] == publisher.MODEL_AUTHORITY
    assert document["candidate_authority"] == publisher.CANDIDATE_AUTHORITY
    assert len(document["what_enabling_live_requires"]) == 3


def test_a_disabled_mode_refuses(tmp_path: Path):
    """1. The switch is the first thing asked and a no is final."""
    built = fixture.build(tmp_path)

    switch = tmp_path / "switch.json"

    switch.write_text(
        json.dumps(
            {
                **json.loads(
                    (PROJECT / publisher.SWITCH_RELATIVE).read_text(
                        encoding="utf-8"
                    )
                ),
                "modes": {
                    publisher.MODE_CANARY: {
                        "state": publisher.STATE_DISABLED,
                        "approval_token": "irrelevant",
                        "reason": "disabled for this test",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(publisher.PublishingRefused, match="DISABLED"):
        publish(built, switch_path=switch)

    assert not built["destination_root"].exists(), "a refusal wrote bytes"


def test_live_publishing_refuses_with_no_approval_at_all(tmp_path: Path):
    """2. The committed live state, asked for directly, with a clean environment."""
    built = fixture.build(tmp_path)

    with pytest.raises(publisher.PublishingRefused, match="DISABLED"):
        publish(built, mode=publisher.MODE_LIVE, environment={})


def test_live_publishing_refuses_a_wrong_approval_token(tmp_path: Path):
    """3. Enabled, environment agreeing, token wrong: still refused.

    The three conditions are independent on purpose, and this is the one that
    makes that worth having. A switch file flipped in a commit plus an
    environment variable set by the same commit's workflow is one decision
    wearing two hats; the token is the part somebody has to supply separately.
    """
    built = fixture.build(tmp_path)

    switch = tmp_path / "switch.json"

    document = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    document["modes"][publisher.MODE_LIVE] = {
        "state": publisher.STATE_ENABLED,
        "approval_token": "the-real-token",
        "reason": "enabled for this test",
    }

    switch.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        publisher.PublishingRefused, match=publisher.PUBLISH_APPROVAL_ENV_VAR
    ):
        publish(
            built,
            mode=publisher.MODE_LIVE,
            switch_path=switch,
            environment={
                publisher.PUBLISH_ENV_VAR: publisher.STATE_ENABLED,
                publisher.PUBLISH_APPROVAL_ENV_VAR: "not-the-real-token",
            },
        )


def test_live_publishing_with_every_approval_still_has_nowhere_to_go(tmp_path: Path):
    """The destination, not the switch, is what blocks live publication.

    Worth separating because the two refusals mean opposite things. A disabled
    switch is a decision not yet taken; a null destination is something this
    repository does not hold. The refusal has to name what is missing so that
    distinction survives into the operator's terminal.

    What is missing has changed shape since the real canary destination was
    configured, and the test tracks it rather than the old wording. The host,
    the transport, the credential's name and the public URL are now known --
    they are on the named canary destination. What live still lacks is a
    reviewed decision and the facts that only follow one, so those are what
    the refusal is required to name. A refusal still quoting a missing host
    would now be misleading rather than reassuring.
    """
    built = fixture.build(tmp_path)

    switch = tmp_path / "switch.json"

    document = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    document["modes"][publisher.MODE_LIVE] = {
        "state": publisher.STATE_ENABLED,
        "approval_token": "the-real-token",
        "reason": "enabled for this test",
    }

    switch.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(publisher.PublishingRefused) as refusal:
        publish(
            built,
            mode=publisher.MODE_LIVE,
            switch_path=switch,
            environment={
                publisher.PUBLISH_ENV_VAR: publisher.STATE_ENABLED,
                publisher.PUBLISH_APPROVAL_ENV_VAR: "the-real-token",
            },
        )

    message = str(refusal.value)

    assert "configures no live destination" in message

    for fact in ("reviewed decision", "approval token", "live proof"):
        assert fact in message, f"the refusal does not name the missing {fact}"

    # The named live destinations are now complete, and that is not permission
    # either. This is the distinction the test is about, from the other side:
    # an address being known is not a decision having been taken, so a run
    # that names one is refused by the committed switch rather than reaching
    # it. The switch here is the committed file, not the enabled one above.
    named = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )[publisher.NAMED_DESTINATIONS_KEY]

    live = sorted(
        name
        for name, declared in named.items()
        if declared.get("mode") == publisher.MODE_LIVE
    )

    assert live == [
        "wizardofodds_nba_live",
        "wizardofodds_nba_player_props_page_live",
    ]

    for name in live:
        with pytest.raises(
            publisher.PublishingRefused, match="live publishing is DISABLED"
        ):
            publish(
                built,
                mode=publisher.MODE_LIVE,
                destination_name=name,
                destination_root=None,
            )


def test_an_unreadable_switch_reads_as_disabled(tmp_path: Path):
    """The absence of a prohibition is not an approval."""
    broken = tmp_path / "broken.json"

    broken.write_text("{not json", encoding="utf-8")

    switch = publisher.read_publishing_switch(
        publisher.MODE_CANARY, project_root=PROJECT, switch_path=broken
    )

    assert switch.state == publisher.STATE_DISABLED
    assert not switch.enabled

    missing = publisher.read_publishing_switch(
        publisher.MODE_CANARY,
        project_root=PROJECT,
        switch_path=tmp_path / "absent.json",
    )

    assert missing.state == publisher.STATE_DISABLED


# ======================================================================
# 4 and 5. authority and provenance
# ======================================================================


def test_a_non_incumbent_receipt_refuses(tmp_path: Path):
    """4. The published authority is read from the receipt, not assumed."""
    built = fixture.build(
        tmp_path, receipt_changes={"model_authority": "candidate"}
    )

    with pytest.raises(publisher.PublishingRefused, match="model authority"):
        publish(built)

    assert not built["destination_root"].exists()


@pytest.mark.parametrize(
    "field",
    [
        "production_code_sha",
        "input_state_fingerprint",
        "incumbent_version",
        "generated_at",
        "incumbent_authority",
    ],
)
def test_incomplete_provenance_refuses(tmp_path: Path, field: str):
    """5. Every provenance field the publication must carry, one at a time."""
    built = fixture.build(tmp_path, receipt_changes={field: ""})

    with pytest.raises(publisher.PublishingRefused, match=field):
        publish(built)


def test_a_dropped_provenance_key_refuses(tmp_path: Path):
    """A key that is absent, as opposed to empty.

    ``incumbent_fit_id`` is the one field whose value may legitimately be
    null -- before the first human promotion there is no fit to name, and
    requiring one would invite inventing it -- so absence and emptiness are
    genuinely different here and only absence is a defect.
    """
    built = fixture.build(
        tmp_path, receipt_changes={"incumbent_fit_id": fixture.DELETE}
    )

    with pytest.raises(publisher.PublishingRefused, match="incumbent_fit_id"):
        publish(built)

    allowed = fixture.build(
        tmp_path / "null-fit", receipt_changes={"incumbent_fit_id": None}
    )

    assert publish(allowed)["outcome"] == publisher.OUTCOME_PUBLISHED


def test_provenance_that_does_not_resolve_to_these_bytes_refuses(tmp_path: Path):
    """A receipt digest that no longer matches the file it provenances.

    The same question the grader asks. Publishing anyway would put this run's
    fit id, production SHA and input fingerprint on a manifest describing rows
    from some other run, which reads as provenanced and is not.
    """
    built = fixture.build(tmp_path)

    frame = fixture.priced_frame()

    frame.loc[0, "calibrated_q_over_nonpush"] = 0.4123

    frame.to_parquet(built["priced_path"], index=False)

    with pytest.raises(publisher.PublishingRefused, match="changed after they"):
        publish(built)


def test_a_receipt_for_another_slate_refuses(tmp_path: Path):
    built = fixture.build(tmp_path, receipt_changes={"slate_date": "2026-11-01"})

    with pytest.raises(publisher.PublishingRefused, match="does not provenance"):
        publish(built)


def test_a_missing_receipt_refuses(tmp_path: Path):
    built = fixture.build(tmp_path)

    built["receipt_path"].unlink()

    with pytest.raises(publisher.PublishingRefused, match="carry no"):
        publish(built)


# ======================================================================
# 6 and 7. determinism and idempotency
# ======================================================================


def test_the_same_rows_produce_the_same_release(tmp_path: Path):
    """6. Byte-for-byte, into two separate destinations.

    Separate destinations rather than two runs into one, because a second run
    into the same tree could agree with itself by reading what the first run
    left there. The release id is part of the comparison: it is derived from
    the payload digest, so two runs agreeing on the bytes and disagreeing on
    the name would still be a destination that cannot be reasoned about.
    """
    first = publish(fixture.build(tmp_path / "a"))

    second = publish(fixture.build(tmp_path / "b"))

    assert first["release"]["rows_sha256"] == second["release"]["rows_sha256"]
    assert first["release"]["release_id"] == second["release"]["release_id"]

    rows = [
        (tmp_path / side / "destination" / first["release"]["rows_relative"])
        .read_bytes()
        for side in ("a", "b")
    ]

    assert rows[0] == rows[1]


def test_a_rerun_changes_nothing(tmp_path: Path):
    """7. Including the rollback target, which a naive rerun would destroy.

    Promotion rotates ``current`` into ``previous``. Doing that for a rerun of
    the release already current would overwrite the real rollback target with
    a copy of the live one, so the destination would still look healthy and
    have nowhere to go back to.

    The audit log is excluded from the comparison and asserted separately: it
    is a record of what was attempted, not part of what is served, and a rerun
    that left no trace of itself would be the wrong kind of idempotent.
    """
    built = fixture.build(tmp_path)

    first = publish(built)

    transport = publisher.LocalDirectoryTransport(built["destination_root"])

    served = {
        name: transport.read_bytes(name)
        for name in transport.listing()
        if name != publisher.AUDIT_LOG_NAME
    }

    second = publish(built)

    assert second["release"]["release_id"] == first["release"]["release_id"]
    assert second["promotion"]["promoted"] is False
    assert {
        name: transport.read_bytes(name)
        for name in transport.listing()
        if name != publisher.AUDIT_LOG_NAME
    } == served, "a rerun changed what the destination serves"
    assert not transport.exists(publisher.PREVIOUS_POINTER)

    actions = [entry["action"] for entry in publisher.read_audit_log(transport)]

    assert actions == [publisher.ACTION_PUBLISHED, publisher.ACTION_REPUBLISHED]


def test_no_partial_file_is_left_at_the_destination(tmp_path: Path):
    """Every write lands whole or not at all.

    The transport stages beside the target and renames, so a reader never sees
    a half-written release and a crashed run leaves nothing to be mistaken for
    one.
    """
    built = fixture.build(tmp_path)

    publish(built)

    transport = publisher.LocalDirectoryTransport(built["destination_root"])

    assert [name for name in transport.listing() if name.endswith(".partial")] == []


# ======================================================================
# 8 and 9. the rows themselves
# ======================================================================


def test_duplicate_logical_rows_refuse(tmp_path: Path):
    """8. One quote published twice is a refusal, not a deduplication.

    Silently collapsing them would make the published row count disagree with
    the priced row count for a reason nobody could see, and the duplicate is
    evidence that the join upstream produced something unexpected.
    """
    frame = fixture.priced_frame()

    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    built = fixture.build(tmp_path, frame=doubled)

    with pytest.raises(publisher.PublishingRefused, match="more than one row"):
        publish(built)


@pytest.mark.parametrize(
    "column,value",
    [
        ("calibrated_q_over_nonpush", 1.4),
        ("calibrated_q_over_nonpush", -0.1),
        ("calibrated_q_over_nonpush", float("nan")),
        ("calibrated_q_over_nonpush", float("inf")),
        ("market_devig_q_over", 2.0),
        ("calibrated_p_push", float("nan")),
    ],
)
def test_a_malformed_probability_refuses(tmp_path: Path, column: str, value: float):
    """9. Not finite, or outside [0, 1], on any published probability."""
    frame = fixture.priced_frame()

    frame.loc[1, column] = value

    built = fixture.build(tmp_path, frame=frame)

    with pytest.raises(publisher.PublishingRefused, match="not a probability"):
        publish(built)

    assert not built["destination_root"].exists()


@pytest.mark.parametrize(
    "column,value,expected",
    [
        ("quote_eligible", False, "quote_eligible is false"),
        ("quote_filter_reason", "player_currently_out", "quote_filter_reason"),
        ("availability_out", 1, "availability_out is 1"),
        ("freeze_id", "nba_prop_quant_somethingelse", "not the approved"),
        ("market_pricing_schema_version", 2, "not the approved"),
        ("projection_schema_version", 2, "not the approved"),
        ("line_value", float("nan"), "line_value"),
        ("calibrated_fair_over_american", None, "is missing"),
    ],
)
def test_each_fail_closed_rule_refuses(
    tmp_path: Path, column: str, value: object, expected: str
):
    """Section 9 of the integration contract, rule by rule.

    Refused rather than filtered. The priced file holds only quotes that
    already passed the quote filter, so a row breaking one of these is the
    runtime disagreeing with its own contract, and a publisher that dropped it
    would publish a thinner feed and say nothing about why.
    """
    frame = fixture.priced_frame()

    frame[column] = frame[column].astype(object)

    frame.loc[2, column] = value

    built = fixture.build(tmp_path, frame=frame)

    with pytest.raises(publisher.PublishingRefused, match=expected):
        publish(built)


def test_a_gate3_prop_without_role_state_refuses(tmp_path: Path):
    """The three props whose Gate 3 routing changed, held to role readiness."""
    frame = fixture.priced_frame()

    frame.loc[0, "prop_type"] = "assists"
    frame.loc[0, "gate3_role_ready"] = 0

    built = fixture.build(tmp_path, frame=frame)

    with pytest.raises(publisher.PublishingRefused, match="gate3_role_ready"):
        publish(built)

    assert frame.loc[1, "prop_type"] not in publisher.GATE3_PROPS


def test_history_reaching_the_slate_date_refuses(tmp_path: Path):
    """The leakage rule, as defence in depth behind the slate builder's own."""
    frame = fixture.priced_frame()

    frame.loc[3, "history_latest_date"] = pd.Timestamp(fixture.SLATE_DATE)

    built = fixture.build(tmp_path, frame=frame)

    with pytest.raises(publisher.PublishingRefused, match="strictly before"):
        publish(built)


def test_an_empty_priced_file_refuses(tmp_path: Path):
    """An empty feed is not a publication."""
    built = fixture.build(tmp_path, frame=fixture.priced_frame().iloc[0:0])

    with pytest.raises(publisher.PublishingRefused, match="no rows"):
        publish(built)


def test_a_priced_file_missing_a_screened_column_refuses(tmp_path: Path):
    """A column the fail-closed rules read is as required as one published.

    Otherwise a file without ``quote_eligible`` would publish perfectly, with
    the rule that reads it silently not applying.
    """
    frame = fixture.priced_frame().drop(columns=["quote_eligible"])

    built = fixture.build(tmp_path, frame=frame)

    with pytest.raises(publisher.PublishingRefused, match="quote_eligible"):
        publish(built)


# ======================================================================
# 10 and 11. the canary, and the live destination
# ======================================================================


def test_the_canary_writes_only_its_own_destination(tmp_path: Path):
    """10. Everything it wrote, enumerated, and all of it under the root."""
    built = fixture.build(tmp_path)

    receipt = publish(built)

    assert receipt["mode"] == publisher.MODE_CANARY
    assert receipt["destination"]["transport"] == publisher.TRANSPORT_LOCAL_DIRECTORY

    release = receipt["release"]["release_id"]

    assert publisher.LocalDirectoryTransport(built["destination_root"]).listing() == [
        publisher.AUDIT_LOG_NAME,
        publisher.ATTESTATION_NAME,
        publisher.CURRENT_POINTER,
        f"{publisher.RELEASES_DIRECTORY}/{release}/{publisher.MANIFEST_NAME}",
        f"{publisher.RELEASES_DIRECTORY}/{release}/{publisher.ROWS_NAME}",
    ]


def test_the_live_destination_is_unchanged_by_a_canary(tmp_path: Path):
    """11. Asserted against a live destination that exists, not against null.

    Against the committed configuration this would pass for the wrong reason:
    there is no live destination, so of course the canary did not alter it. So
    the test configures one, points the canary somewhere else, and compares
    the live tree before and after.
    """
    built = fixture.build(tmp_path)

    live_root = tmp_path / "live"

    live_root.mkdir()

    (live_root / publisher.CURRENT_POINTER).write_text(
        json.dumps({"release_id": "yesterdays-live-release"}), encoding="utf-8"
    )

    document = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    document["destinations"][publisher.MODE_LIVE] = {
        "transport": publisher.TRANSPORT_LOCAL_DIRECTORY,
        "root": str(live_root),
        "release_prefix": "nba_prop_quant",
    }

    destinations = tmp_path / "destinations.json"

    destinations.write_text(json.dumps(document), encoding="utf-8")

    live = publisher.LocalDirectoryTransport(live_root)

    before = {name: live.read_bytes(name) for name in live.listing()}

    receipt = publish(built, destinations_path=destinations)

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED

    after = {name: live.read_bytes(name) for name in live.listing()}

    assert after == before, "the canary altered the live destination"


def test_a_canary_pointed_at_the_live_destination_refuses(tmp_path: Path):
    """The one way a canary could alter live output, closed off by name."""
    built = fixture.build(tmp_path)

    live_root = tmp_path / "live"

    live_root.mkdir()

    document = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    document["destinations"][publisher.MODE_LIVE] = {
        "transport": publisher.TRANSPORT_LOCAL_DIRECTORY,
        "root": str(live_root),
        "release_prefix": "nba_prop_quant",
    }

    destinations = tmp_path / "destinations.json"

    destinations.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(publisher.PublishingRefused, match="is a live destination"):
        publish(
            built, destinations_path=destinations, destination_root=live_root
        )


def test_the_committed_destinations_configure_no_unnamed_live_target():
    """The state this change is required to leave behind, read off disk.

    Two transports now, and exactly two: a remote one was added because the
    destination is real, and nothing else was. The unnamed live target is
    still null, which is what keeps a mode from resolving to a real host by
    default -- a remote destination is addressed by name, so reaching
    wizardofodds.com is always something asked for explicitly.

    The live *addresses* are no longer missing, and that is a separate fact
    from permission. Automation needs them settled so that a later reviewed
    activation supplies an approval rather than a destination, and the
    publisher refuses live mode on the committed switch before it resolves a
    destination at all, so a complete address buys nothing. What remains in
    ``missing_external_facts`` is the decision and what follows from it.
    """
    document = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    assert document["destinations"][publisher.MODE_LIVE] is None
    assert document["supported_transports"] == [
        publisher.TRANSPORT_LOCAL_DIRECTORY,
        publisher.TRANSPORT_SSH_RSYNC,
    ]

    # Three, and every one a decision: the reviewed decision itself, the
    # token it would set, and a live proof that cannot precede it.
    assert len(document["missing_external_facts"]) == 3

    for fact in ("reviewed decision", "approval token", "live proof"):
        assert any(
            fact in entry for entry in document["missing_external_facts"]
        ), fact

    # And nothing still claims an address is unknown.
    for settled in ("whether the live feed", "the retention the website"):
        assert not any(
            settled in entry for entry in document["missing_external_facts"]
        ), settled

    # The unnamed canary is still the local scratch directory, so a run that
    # named no destination cannot reach a host.
    canary = document["destinations"][publisher.MODE_CANARY]

    assert canary["transport"] == publisher.TRANSPORT_LOCAL_DIRECTORY
    assert canary["root"] is None


# ======================================================================
# 12 and 13. partial output, and verification before promotion
# ======================================================================


def test_a_partial_release_is_not_promoted(tmp_path: Path, monkeypatch):
    """12. A write the destination accepted and did not keep.

    The manifest is removed between the write and the readback, which is what
    a truncated upload or a path that silently went elsewhere looks like from
    here. The pointer must not move: a promoted release nobody verified is the
    failure this ordering exists to prevent.
    """
    built = fixture.build(tmp_path)

    first = publish(built)

    transport = publisher.LocalDirectoryTransport(built["destination_root"])

    current = transport.read_bytes(publisher.CURRENT_POINTER)

    frame = fixture.priced_frame()

    frame.loc[0, "calibrated_q_over_nonpush"] = 0.4321

    rebuilt = fixture.build(tmp_path / "second", frame=frame)

    original = publisher.write_release

    def lose_the_manifest(transport_, release):
        original(transport_, release)
        (Path(transport_.root) / release.manifest_relative).unlink()

    monkeypatch.setattr(publisher, "write_release", lose_the_manifest)

    with pytest.raises(publisher.PublishingRefused, match="is not at the destination"):
        publish(
            rebuilt,
            destination_root=built["destination_root"],
        )

    assert transport.read_bytes(publisher.CURRENT_POINTER) == current
    assert first["release"]["release_id"] == json.loads(current)["release_id"]


def test_a_release_that_reads_back_wrong_is_not_promoted(tmp_path: Path, monkeypatch):
    """13. Verification failing on content rather than on absence.

    A destination that accepted the bytes and kept different ones. Only a
    readback can see it, which is why the readback is between the write and
    the promotion rather than after it.
    """
    built = fixture.build(tmp_path)

    publish(built)

    transport = publisher.LocalDirectoryTransport(built["destination_root"])

    current = transport.read_bytes(publisher.CURRENT_POINTER)

    frame = fixture.priced_frame()

    frame.loc[0, "calibrated_q_over_nonpush"] = 0.4567

    rebuilt = fixture.build(tmp_path / "second", frame=frame)

    original = publisher.write_release

    def corrupt_the_rows(transport_, release):
        original(transport_, release)
        (Path(transport_.root) / release.rows_relative).write_bytes(b"{}\n")

    monkeypatch.setattr(publisher, "write_release", corrupt_the_rows)

    with pytest.raises(publisher.PublishingRefused, match="reads back as"):
        publish(rebuilt, destination_root=built["destination_root"])

    assert transport.read_bytes(publisher.CURRENT_POINTER) == current


def test_verification_reads_through_the_transport(tmp_path: Path):
    """Readback, not recollection.

    ``verify_release`` is given the release it just wrote, so it could satisfy
    itself from its own arguments. Deleting the file from under it proves it
    does not: a verifier that trusted what it was handed would pass here.
    """
    destination = publisher.Destination(
        mode=publisher.MODE_CANARY,
        transport=publisher.TRANSPORT_LOCAL_DIRECTORY,
        root=tmp_path / "destination",
        release_prefix="nba_prop_quant",
        source="test",
    )

    switch = publisher.read_publishing_switch(
        publisher.MODE_CANARY, project_root=PROJECT
    )

    built = fixture.build(tmp_path)

    rows = publisher.published_rows(
        pd.read_parquet(built["priced_path"]),
        switch=switch,
        slate_date=built["slate_date"],
    )

    document = publisher.read_distributions_document(built["distributions_path"])

    release = publisher.build_release(
        rows=rows,
        distributions=publisher.published_distributions(
            document,
            rows=rows,
            priced_sha256=publisher.sha256_file(built["priced_path"]),
        ),
        provenance={"slate_date": built["slate_date"]},
        switch=switch,
        destination=destination,
        slate_date=built["slate_date"],
    )

    transport = publisher.build_transport(destination)

    publisher.write_release(transport, release)

    assert publisher.verify_release(transport, release)["verified"] is True

    (destination.root / release.rows_relative).unlink()

    with pytest.raises(publisher.PublishingRefused):
        publisher.verify_release(transport, release)


# ======================================================================
# 14. rollback
# ======================================================================


def test_rollback_restores_the_previous_release(tmp_path: Path):
    """14. The pointer goes back and the old bytes are still there.

    Two releases, because a rollback with only one is a different question.
    The restored pointer has to name the first release and the destination has
    to still hold its rows: a rollback that repointed at bytes that were gone
    would be worse than staying put.
    """
    built = fixture.build(tmp_path)

    first = publish(built)

    frame = fixture.priced_frame()

    frame.loc[0, "calibrated_q_over_nonpush"] = 0.4711

    second_built = fixture.build(tmp_path / "second", frame=frame)

    second = publish(second_built, destination_root=built["destination_root"])

    assert second["release"]["release_id"] != first["release"]["release_id"]
    assert second["promotion"]["promoted"] is True
    assert second["promotion"]["rollback_target"] == first["release"]["release_id"]

    transport = publisher.LocalDirectoryTransport(built["destination_root"])

    restored = publisher.rollback(transport)

    assert restored["outcome"] == publisher.OUTCOME_ROLLED_BACK
    assert restored["restored_release_id"] == first["release"]["release_id"]
    assert restored["superseded_release_id"] == second["release"]["release_id"]

    pointer = json.loads(transport.read_bytes(publisher.CURRENT_POINTER))

    assert pointer["release_id"] == first["release"]["release_id"]
    assert pointer["rows_sha256"] == first["release"]["rows_sha256"]

    assert (
        publisher.digest(transport.read_bytes(pointer["rows_relative"]))
        == first["release"]["rows_sha256"]
    )


def test_rollback_refuses_when_there_is_nothing_to_restore(tmp_path: Path):
    built = fixture.build(tmp_path)

    publish(built)

    transport = publisher.LocalDirectoryTransport(built["destination_root"])

    with pytest.raises(publisher.PublishingRefused, match="nothing to roll back"):
        publisher.rollback(transport)


def test_rollback_refuses_when_the_previous_bytes_are_gone(tmp_path: Path):
    """Restoration, not reconstruction: with the bytes gone there is no answer."""
    built = fixture.build(tmp_path)

    first = publish(built)

    frame = fixture.priced_frame()

    frame.loc[0, "calibrated_q_over_nonpush"] = 0.4812

    second = fixture.build(tmp_path / "second", frame=frame)

    publish(second, destination_root=built["destination_root"])

    transport = publisher.LocalDirectoryTransport(built["destination_root"])

    (built["destination_root"] / first["release"]["rows_relative"]).unlink()

    with pytest.raises(publisher.PublishingRefused, match="point at nothing"):
        publisher.rollback(transport)


def test_rollback_touches_no_model_or_registry_path(tmp_path: Path):
    """A rollback is a pointer write and nothing else.

    Asserted structurally rather than by inspection: the function's syntax
    tree is checked for any call that could refit, recalibrate, promote a fit
    or reach the shadow.
    """
    source = (PROJECT / publisher.__file__.split("/")[-1]).name

    tree = ast.parse(
        (PROJECT / "ops" / source).read_text(encoding="utf-8")
    )

    rollback_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "rollback"
    )

    forbidden = ("fit", "promote", "calibrat", "train", "register", "shadow")

    offenders = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(rollback_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    ]

    assert [
        name for name in offenders if any(word in name.lower() for word in forbidden)
    ] == []


def test_rollback_writes_an_audit_receipt(tmp_path: Path):
    """What the destination is serving, and how it got there, both readable.

    At the destination rather than only in the operator's terminal: a receipt
    that exists on whichever machine happened to run the rollback is not a
    record anybody looking at the destination can find.
    """
    built = fixture.build(tmp_path)

    first = publish(built)

    frame = fixture.priced_frame()

    frame.loc[0, "calibrated_q_over_nonpush"] = 0.4615

    second = publish(
        fixture.build(tmp_path / "second", frame=frame),
        destination_root=built["destination_root"],
    )

    transport = publisher.LocalDirectoryTransport(built["destination_root"])

    restored = publisher.rollback(transport)

    assert restored["audit_relative"] == publisher.AUDIT_LOG_NAME

    entries = publisher.read_audit_log(transport)

    assert [entry["action"] for entry in entries] == [
        publisher.ACTION_PUBLISHED,
        publisher.ACTION_PUBLISHED,
        publisher.ACTION_ROLLED_BACK,
    ]

    assert entries[-1]["release_id"] == first["release"]["release_id"]
    assert entries[-1]["superseded_release_id"] == second["release"]["release_id"]
    assert entries[-1]["model_authority"] == publisher.MODEL_AUTHORITY


# ======================================================================
# the activation contract
#
# Seven conditions have to hold before anything goes live. Four of them --
# the mode, the switch, the authority and the token -- are refused in the
# first section above, against the committed switch, which is the state that
# matters. The two that only become observable once those are satisfied get
# their own section here, because a test that cannot tell which of five
# refusals it triggered is not testing the gate it names.
# ======================================================================


def approved_live(tmp_path: Path, *, live_root: Path) -> dict:
    """A live run with every earlier gate satisfied.

    The switch enabled, the environment agreeing, a matching token and a
    configured destination -- none of which exist in the committed state, and
    all of which have to be forged here for the two remaining gates to be the
    thing a refusal is about.
    """
    document = json.loads(
        (PROJECT / publisher.SWITCH_RELATIVE).read_text(encoding="utf-8")
    )

    document["modes"][publisher.MODE_LIVE] = {
        "state": publisher.STATE_ENABLED,
        "approval_token": "the-real-token",
        "reason": "enabled for this test only",
    }

    switch = tmp_path / "live_switch.json"

    switch.write_text(json.dumps(document), encoding="utf-8")

    destinations = json.loads(
        (PROJECT / publisher.DESTINATIONS_RELATIVE).read_text(encoding="utf-8")
    )

    destinations["destinations"][publisher.MODE_LIVE] = {
        "transport": publisher.TRANSPORT_LOCAL_DIRECTORY,
        "root": str(live_root),
        "release_prefix": "nba_prop_quant",
    }

    destinations_path = tmp_path / "live_destinations.json"

    destinations_path.write_text(json.dumps(destinations), encoding="utf-8")

    return {
        "mode": publisher.MODE_LIVE,
        "switch_path": switch,
        "destinations_path": destinations_path,
        "destination_root": live_root,
        "environment": {
            publisher.PUBLISH_ENV_VAR: publisher.STATE_ENABLED,
            publisher.PUBLISH_APPROVAL_ENV_VAR: "the-real-token",
        },
    }


def seed_rollback_target(live_root: Path) -> dict:
    """Give the live destination a release it is already serving.

    Written directly rather than published, because that is the real shape of
    it: the thing a first activation would be rolling back to was put there by
    something other than this publisher, and the gate has to be satisfiable by
    a destination state this publisher did not create.
    """
    transport = publisher.LocalDirectoryTransport(live_root)

    release_id = "nba_prop_quant__2026-10-19__alreadyserving"

    rows_relative = (
        f"{publisher.RELEASES_DIRECTORY}/{release_id}/{publisher.ROWS_NAME}"
    )

    rows = publisher.serialise(
        {
            "contract_version": publisher.PUBLICATION_CONTRACT_VERSION,
            "rows": [],
            "slate_date": "2026-10-19",
        }
    )

    transport.write_bytes(rows_relative, rows)

    transport.write_bytes(
        publisher.CURRENT_POINTER,
        publisher.serialise(
            {
                "mode": publisher.MODE_LIVE,
                "promoted_at": "2026-10-19T00:00:00Z",
                "release_id": release_id,
                "rows_relative": rows_relative,
                "rows_sha256": publisher.digest(rows),
            }
        ),
    )

    return {"release_id": release_id, "rows_relative": rows_relative}


def test_activation_refuses_without_a_canary_attestation(tmp_path: Path):
    """Nothing goes live that was not canaried first, and nothing is written."""
    built = fixture.build(tmp_path)

    live_root = tmp_path / "live"

    seed_rollback_target(live_root)

    live = publisher.LocalDirectoryTransport(live_root)

    before = {name: live.read_bytes(name) for name in live.listing()}

    with pytest.raises(
        publisher.PublishingRefused, match="requires --canary-attestation"
    ):
        publish(built, **approved_live(tmp_path, live_root=live_root))

    assert {
        name: live.read_bytes(name) for name in live.listing()
    } == before, "a refused activation wrote to the live destination"


def test_activation_refuses_a_canary_of_a_different_artifact(tmp_path: Path):
    """8. The canary has to have proved these bytes, not some bytes.

    The attestation is real and records a genuine successful canary. It is
    simply a canary of a different priced file, which is exactly the mistake
    a "was this canaried?" checkbox cannot catch.
    """
    built = fixture.build(tmp_path)

    live_root = tmp_path / "live"

    seed_rollback_target(live_root)

    other = fixture.priced_frame()

    other.loc[1, "calibrated_q_under_nonpush"] = 0.4123

    elsewhere = fixture.build(tmp_path / "elsewhere", frame=other)

    publish(elsewhere)

    attestation = elsewhere["destination_root"] / publisher.ATTESTATION_NAME

    with pytest.raises(publisher.PublishingRefused, match="different bytes"):
        publish(
            built,
            canary_attestation=attestation,
            **approved_live(tmp_path, live_root=live_root),
        )


def test_activation_refuses_an_unverified_canary(tmp_path: Path):
    """A canary that did not read its own release back authorises nothing."""
    built = fixture.build(tmp_path)

    live_root = tmp_path / "live"

    seed_rollback_target(live_root)

    publish(built)

    attestation = built["destination_root"] / publisher.ATTESTATION_NAME

    document = json.loads(attestation.read_text(encoding="utf-8"))

    document["verified"] = False

    forged = tmp_path / "forged_attestation.json"

    forged.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(publisher.PublishingRefused, match="verified readback"):
        publish(
            built,
            canary_attestation=forged,
            **approved_live(tmp_path, live_root=live_root),
        )


def test_activation_refuses_when_there_is_no_rollback_target(tmp_path: Path):
    """An activation is never the step that leaves live with nowhere to go.

    The canary here is genuine and matches, so the rollback gate is the only
    thing left to refuse on. The live destination is empty, which is also the
    position a real first activation starts from: its rollback target has to
    be seeded before it is allowed to proceed.
    """
    built = fixture.build(tmp_path)

    publish(built)

    attestation = built["destination_root"] / publisher.ATTESTATION_NAME

    live_root = tmp_path / "live"

    live_root.mkdir()

    with pytest.raises(publisher.PublishingRefused, match="no rollback target"):
        publish(
            built,
            canary_attestation=attestation,
            **approved_live(tmp_path, live_root=live_root),
        )

    assert publisher.LocalDirectoryTransport(live_root).listing() == []


def test_activation_refuses_a_rollback_target_whose_bytes_are_gone(tmp_path: Path):
    """A pointer naming absent bytes is not somewhere to go back to."""
    built = fixture.build(tmp_path)

    live_root = tmp_path / "live"

    seeded = seed_rollback_target(live_root)

    publish(built)

    attestation = built["destination_root"] / publisher.ATTESTATION_NAME

    (live_root / seeded["rows_relative"]).unlink()

    with pytest.raises(
        publisher.PublishingRefused, match="could not be rolled back to"
    ):
        publish(
            built,
            canary_attestation=attestation,
            **approved_live(tmp_path, live_root=live_root),
        )


def test_activation_with_all_seven_conditions_publishes_and_keeps_the_rollback(
    tmp_path: Path,
):
    """The positive case, so the gates are not passing by always refusing.

    Reached only with a forged switch, a forged destination and a forged
    environment, none of which exist in this repository: against the committed
    state this sequence is unreachable, which is the point. What it proves is
    that the seven conditions are jointly satisfiable and that satisfying them
    leaves the previous release in place as the rollback target.
    """
    built = fixture.build(tmp_path)

    live_root = tmp_path / "live"

    seeded = seed_rollback_target(live_root)

    publish(built)

    attestation = built["destination_root"] / publisher.ATTESTATION_NAME

    receipt = publish(
        built,
        canary_attestation=attestation,
        **approved_live(tmp_path, live_root=live_root),
    )

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED
    assert receipt["mode"] == publisher.MODE_LIVE
    assert receipt["model_authority"] == publisher.MODEL_AUTHORITY
    assert receipt["verification"]["verified"] is True
    assert receipt["promotion"]["promoted"] is True
    assert receipt["activation"]["rollback_target"]["release_id"] == (
        seeded["release_id"]
    )
    assert receipt["activation"]["canary"]["rows_sha256"] == (
        receipt["release"]["rows_sha256"]
    )

    live = publisher.LocalDirectoryTransport(live_root)

    assert (
        json.loads(live.read_bytes(publisher.PREVIOUS_POINTER))["release_id"]
        == seeded["release_id"]
    )

    restored = publisher.rollback(live)

    assert restored["restored_release_id"] == seeded["release_id"]


def test_an_activation_leaves_no_canary_attestation_of_its_own(tmp_path: Path):
    """Only a canary attests. Otherwise a live release would authorise itself.

    An activation that wrote an attestation beside its own release would make
    the next activation's canary gate satisfiable by the previous activation,
    which turns a required canary into a formality after the first one.
    """
    built = fixture.build(tmp_path)

    live_root = tmp_path / "live"

    seed_rollback_target(live_root)

    publish(built)

    publish(
        built,
        canary_attestation=built["destination_root"] / publisher.ATTESTATION_NAME,
        **approved_live(tmp_path, live_root=live_root),
    )

    assert publisher.ATTESTATION_NAME not in (
        publisher.LocalDirectoryTransport(live_root).listing()
    )


# ======================================================================
# 15. the candidate can never publish
# ======================================================================


def test_a_shadow_log_can_never_be_published(tmp_path: Path):
    """15. Refused on the columns, not on a label.

    The way a candidate probability reaches a feed is not somebody choosing to
    publish one. It is a file that looks close enough to the right file, with a
    receipt that says incumbent because the receipt was written by the
    incumbent run. So the frame itself is inspected for the columns
    ``ShadowDecision.payload`` emits.
    """
    frame = fixture.priced_frame()

    frame["candidate_probability"] = 0.51
    frame["served_probability"] = 0.49

    built = fixture.build(tmp_path, frame=frame)

    with pytest.raises(publisher.PublishingRefused, match="shadow-candidate"):
        publish(built)

    assert not built["destination_root"].exists()


def test_the_candidate_column_names_are_the_shadows_own():
    """The ban is on the real names, kept honest against the real payload.

    A list of plausible names would pass this file and miss the thing it is
    for, so every banned name is checked against the keys
    ``ShadowDecision.payload`` actually returns. Read off that method's syntax
    tree rather than by building a decision, so the check does not depend on
    the dataclass's constructor and keeps working if a field is added.
    """
    source = (
        PROJECT
        / "src"
        / "nba_prop_quant"
        / "research"
        / "game_latent_state"
        / "shadow_runtime.py"
    ).read_text(encoding="utf-8")

    decision = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "ShadowDecision"
    )

    payload = next(
        node
        for node in decision.body
        if isinstance(node, ast.FunctionDef) and node.name == "payload"
    )

    emitted = {
        key.value
        for node in ast.walk(payload)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    banned = set(publisher.CANDIDATE_SOURCE_COLUMNS)

    assert banned <= emitted, f"not shadow column names: {sorted(banned - emitted)}"

    for name in ("candidate_probability", "served_probability"):
        assert name in banned


def test_the_publisher_cannot_reach_the_shadow_or_the_model():
    """No import of the shadow, the model, the copula or the pricing path.

    Checked from the publisher's side, the way
    ``test_the_publishing_surface_cannot_reach_the_shadow_candidate`` checks it
    for the bundle builder. A publisher that imported the candidate would put
    a candidate probability one call away from a feed, and one that imported
    the pricing path would be able to reprice on the way out.
    """
    tree = ast.parse(
        (PROJECT / "ops" / "publish_incumbent_to_wizardofodds.py").read_text(
            encoding="utf-8"
        )
    )

    imported: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    forbidden = (
        "game_latent_state",
        "nba_prop_quant.copula",
        "nba_prop_quant.distributions",
        "nba_prop_quant.model",
        "nba_prop_quant.pricing",
        "nba_prop_quant.features",
    )

    assert [
        name for name in imported if name.startswith(forbidden)
    ] == [], f"the publisher imports: {imported}"


def test_the_published_probabilities_are_copied_not_computed(tmp_path: Path):
    """Every published number equals the source cell exactly.

    Exact equality rather than approximate. The integration contract says the
    consumer "must not recalibrate probabilities", and the only way to keep
    that promise through a publisher is for the publisher to be a projection:
    any arithmetic at all, even a round trip through a formatter, would make
    this test the place it showed up.
    """
    built = fixture.build(tmp_path)

    receipt = publish(built)

    source = pd.read_parquet(built["priced_path"])

    published = json.loads(
        (built["destination_root"] / receipt["release"]["rows_relative"]).read_text(
            encoding="utf-8"
        )
    )["rows"]

    assert len(published) == len(source)

    numeric = [
        *publisher.PUBLISHED_PROBABILITY_FIELDS,
        "calibrated_edge_over",
        "calibrated_edge_under",
        "calibrated_ev_over",
        "calibrated_ev_under",
        "model_preferred_edge",
        "model_preferred_ev",
        "calibrated_fair_over_american",
        "calibrated_fair_under_american",
        "over_odds",
        "under_odds",
        "line_value",
    ]

    by_quote = {int(row["quote_id"]): row for row in published}

    for _, row in source.iterrows():
        out = by_quote[int(row["id"])]

        for field in numeric:
            assert out[field] == pytest.approx(float(row[field]), abs=0.0), (
                f"{field} on quote {row['id']} was not copied verbatim"
            )

        assert out["model_preferred_side"] == row["model_preferred_side"]
        assert out["freeze_id"] == row["freeze_id"]


def test_no_raw_probability_is_published():
    """``raw_*`` is retained for audit and is not the model's price.

    The contract says so directly. Pinned as a property of the field list so
    adding one is a test failure rather than a judgement call later.
    """
    published = {name for _, name in publisher.PUBLISHED_FIELDS}

    assert [name for name in published if name.startswith("raw_")] == []

    assert set(publisher.PUBLISHED_PROBABILITY_FIELDS) <= published


def test_the_published_record_carries_everything_the_brief_requires():
    """player, game, prop, side, line, both probabilities, price, EV, book, identity."""
    published = {name for _, name in publisher.PUBLISHED_FIELDS}

    for required in (
        "player_id",
        "player_name",
        "game_id",
        "prop_type",
        "model_preferred_side",
        "line_value",
        "calibrated_q_over_nonpush",
        "market_devig_q_over",
        "over_odds",
        "under_odds",
        "calibrated_fair_over_american",
        "model_preferred_edge",
        "model_preferred_ev",
        "vendor",
        "freeze_id",
        "manifest_sha256",
    ):
        assert required in published, f"the published row omits {required}"


def test_the_manifest_carries_the_provenance_the_brief_requires(tmp_path: Path):
    """Production SHA, incumbent identity, fingerprint, slate, timestamps, digests."""
    built = fixture.build(tmp_path)

    receipt = publish(built)

    manifest = json.loads(
        (
            built["destination_root"] / receipt["release"]["manifest_relative"]
        ).read_text(encoding="utf-8")
    )

    assert manifest["model_authority"] == publisher.MODEL_AUTHORITY
    assert manifest["candidate_authority"] == publisher.CANDIDATE_AUTHORITY
    assert manifest["row_count"] == len(fixture.QUOTES)
    assert manifest["slate_date"] == built["slate_date"]

    provenance = manifest["provenance"]

    for field in (
        "production_code_sha",
        "incumbent_version",
        "input_state_fingerprint",
        "priced_market_artifact_sha256",
        "served_at",
        "slate_date",
    ):
        assert str(provenance.get(field) or "").strip(), f"manifest lacks {field}"

    assert "incumbent_fit_id" in provenance
    assert provenance["model_authority"] == publisher.MODEL_AUTHORITY

    assert manifest["approved_freeze_id"] == fixture.FREEZE_ID
    assert manifest["rows_sha256"] == receipt["release"]["rows_sha256"]


# ======================================================================
# 16. no model, prediction or pricing logic changed
# ======================================================================


#: Every module that decides a number production serves. Byte-identical to
#: production is the claim, and these are the files it is about.
NUMERICAL_MODULES: tuple[str, ...] = (
    "scripts/10_predict_slate.py",
    "scripts/15_price_markets.py",
    "src/nba_prop_quant/copula.py",
    "src/nba_prop_quant/distributions.py",
    "src/nba_prop_quant/features.py",
    "src/nba_prop_quant/game_context.py",
    "src/nba_prop_quant/model.py",
    "src/nba_prop_quant/pricing.py",
    "src/nba_prop_quant/production.py",
    "src/nba_prop_quant/slate.py",
    "src/nba_prop_quant/gate3_v2.py",
    "ops/run_incumbent_production_serving.py",
)


def test_no_model_prediction_or_pricing_module_changed():
    """16. The twelve numerical modules, against the production base.

    ``run_incumbent_production_serving.py`` is on the list deliberately: the
    publisher consumes its output and must not have had to change it to do so.
    A publisher that needed the serving step adjusted would be a publisher
    that was not reading the already-approved artifact.

    Three of the twelve are declared additive-only, and for those the claim is
    checked as the containment guard states it rather than as byte-equality --
    see ``tests/serving_additive_pin.py``. Nothing is conceded: a file that
    carries a column forward passes, and a file whose pricing moved does not.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    for relative in NUMERICAL_MODULES:
        assert_numerically_unchanged(base, relative)


def test_no_model_artifact_frozen_config_or_freeze_manifest_changed():
    """Nothing fitted, frozen or configured is in this diff.

    The two registries under ``models/`` that *record* what serving runs,
    rather than being what serving runs, are not part of that claim. One of
    them has to move whenever an approved additive serving change lands --
    enumerating the divergence is its own stated purpose -- and the
    ``test_wizardofodds_runtime_bundle`` suite is what holds it to that: every
    historical reference hash pinned, every current hash matching the file it
    locks, every divergence justified. Reading a relock as a changed model
    artifact would make that record unmaintainable and prove nothing.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    changed = [
        line
        for line in (
            git("diff", "--name-only", base).splitlines()
            + git("ls-files", "--others", "--exclude-standard").splitlines()
        )
        if line
    ]

    offenders = [
        path
        for path in changed
        if path not in DECLARABLE_SERVING_SOURCE_REGISTRIES
        and (
            path.endswith((".joblib", ".pkl", ".npy"))
            or path.startswith(("configs/", "models/", "data/", "research/"))
        )
    ]

    assert offenders == [], f"model surface changed: {offenders}"


def test_the_publisher_contains_no_arithmetic_on_a_published_number():
    """No published field is produced by an expression.

    The transform is a dictionary comprehension over ``PUBLISHED_FIELDS`` and
    nothing else, so the only thing between the parquet cell and the JSON
    value is ``_scalar``. Pinned structurally because "the publisher does not
    recompute" is the property the whole module exists to have, and a review
    reading for it is exactly the check that stops happening.
    """
    tree = ast.parse(
        (PROJECT / "ops" / "publish_incumbent_to_wizardofodds.py").read_text(
            encoding="utf-8"
        )
    )

    scalar = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_scalar"
    )

    assert [
        node
        for node in ast.walk(scalar)
        if isinstance(node, (ast.BinOp, ast.UnaryOp))
    ] == [], "_scalar does arithmetic on the value it is passed"


def test_the_frozen_identifiers_are_untouched():
    """Model spec, factor spec and live policy, recomputed."""
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


# ======================================================================
# the entry point
# ======================================================================


def test_the_entry_point_refuses_live_and_exits_nonzero(tmp_path: Path):
    """The CLI, as an operator would reach it, with the committed switch."""
    built = fixture.build(tmp_path)

    environment = {
        **os.environ,
        "PYTHONPATH": str(PROJECT / "src"),
    }

    environment.pop(publisher.PUBLISH_ENV_VAR, None)
    environment.pop(publisher.PUBLISH_APPROVAL_ENV_VAR, None)

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "ops" / "publish_incumbent_to_wizardofodds.py"),
            "--mode",
            publisher.MODE_LIVE,
            "--slate-date",
            built["slate_date"],
            "--data-root",
            str(built["data_root"]),
            "--destination-root",
            str(built["destination_root"]),
        ],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == publisher.EXIT_REFUSED
    assert "PUBLISHING_REFUSED" in completed.stderr
    assert not built["destination_root"].exists()


def test_the_entry_point_runs_the_canary_and_writes_a_receipt(tmp_path: Path):
    """Exit 0, rows published, and a receipt an operator can read."""
    built = fixture.build(tmp_path)

    receipt_path = tmp_path / "publication.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "ops" / "publish_incumbent_to_wizardofodds.py"),
            "--mode",
            publisher.MODE_CANARY,
            "--slate-date",
            built["slate_date"],
            "--data-root",
            str(built["data_root"]),
            "--destination-root",
            str(built["destination_root"]),
            "--receipt-path",
            str(receipt_path),
            "--summary-path",
            str(tmp_path / "summary.md"),
        ],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
    )

    assert completed.returncode == publisher.EXIT_OK, completed.stderr

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED
    assert receipt["mode"] == publisher.MODE_CANARY
    assert receipt["model_authority"] == publisher.MODEL_AUTHORITY
    assert receipt["live_publishing"] == publisher.STATE_DISABLED
    assert receipt["release"]["row_count"] == len(fixture.QUOTES)
    assert receipt["verification"]["verified"] is True

    assert "WizardOfOdds incumbent publication" in (
        tmp_path / "summary.md"
    ).read_text(encoding="utf-8")


def test_the_entry_point_rolls_back(tmp_path: Path):
    """The documented recovery, through the CLI rather than the library."""
    built = fixture.build(tmp_path)

    first = publish(built)

    frame = fixture.priced_frame()

    frame.loc[0, "calibrated_q_over_nonpush"] = 0.4913

    publish(
        fixture.build(tmp_path / "second", frame=frame),
        destination_root=built["destination_root"],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "ops" / "publish_incumbent_to_wizardofodds.py"),
            "--rollback",
            "--slate-date",
            built["slate_date"],
            "--data-root",
            str(built["data_root"]),
            "--destination-root",
            str(built["destination_root"]),
        ],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
    )

    assert completed.returncode == publisher.EXIT_OK, completed.stderr

    outcome = json.loads(completed.stdout)

    assert outcome["outcome"] == publisher.OUTCOME_ROLLED_BACK
    assert outcome["restored_release_id"] == first["release"]["release_id"]


def test_the_lifecycle_workflow_cannot_publish_live():
    """Live activation is not a consequence of the scheduled lifecycle.

    The lifecycle does publish now: a canary feed and the page built from it,
    every production day, because an integration nobody runs unattended is
    not an integration. What it cannot do is activate. It asks for the canary
    mode by name, and the three things that would make a run live -- the two
    approval variables and a live mode on the command line -- appear nowhere
    in the file.

    It reaches the publisher through ops/run_wizardofodds_publication.py
    rather than naming this entry point directly, so the ordering between the
    feed and the page is expressed once, in Python, rather than twice in YAML.
    """
    workflow = (
        PROJECT / ".github" / "workflows" / "nba_production_lifecycle.yml"
    ).read_text(encoding="utf-8")

    assert "publish_incumbent_to_wizardofodds" not in workflow
    assert publisher.PUBLISH_ENV_VAR not in workflow
    assert publisher.PUBLISH_APPROVAL_ENV_VAR not in workflow
    assert f"--mode {publisher.MODE_LIVE}" not in workflow

    assert "ops/run_wizardofodds_publication.py" in workflow
    assert f"--mode {publisher.MODE_CANARY}" in workflow


def test_only_the_publisher_reads_its_activation_variables():
    """One module consults the two variables. The rest only forbid them.

    The same property the shadow switch is held to. Activation cannot be a
    side effect of running some other entry point if only one module consults
    the variables.

    The two static validators name them as well, in the lists of tokens a
    workflow may not contain. That is the opposite of consulting one: a
    validator that finds the name fails the workflow rather than publishing
    from it, and neither validator reads the environment at all.
    """
    bearers = {
        path.relative_to(PROJECT).as_posix()
        for path in [
            *(PROJECT / "ops").glob("*.py"),
            *(PROJECT / "scripts").glob("*.py"),
            *(PROJECT / "src").rglob("*.py"),
        ]
        if publisher.PUBLISH_ENV_VAR in path.read_text(encoding="utf-8")
    }

    entry_point = "ops/publish_incumbent_to_wizardofodds.py"

    assert bearers == {
        entry_point,
        "ops/validate_main_scheduler_role.py",
        "ops/validate_workflows.py",
    }

    for name in sorted(bearers - {entry_point}):
        source = (PROJECT / name).read_text(encoding="utf-8")

        assert "os.environ" not in source, name
        assert "getenv" not in source, name


def test_the_serving_module_still_publishes_nothing():
    """The step before this one did not become the publishing step.

    Serving's own docstring says it does not publish. This change adds a
    separate module rather than teaching serving to publish, and that is the
    difference between a publication being a decision and it being a
    consequence of the nightly run.
    """
    source = (PROJECT / "ops" / "run_incumbent_production_serving.py").read_text(
        encoding="utf-8"
    )

    assert "It does not publish." in source
    assert "publish_incumbent_to_wizardofodds" not in source
    assert serving.MODEL_AUTHORITY == publisher.MODEL_AUTHORITY
