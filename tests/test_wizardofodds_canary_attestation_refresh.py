"""The canary attestation as a current-state document.

A destination keeps two records about itself and they are not the same kind of
record. ``audit_log.jsonl`` is history: one appended line per completed action,
never rewritten, because rewriting it would falsify what was published when.
``canary_attestation.json`` is current state: it describes the release the
destination is serving now.

The attestation used to be written only by a publication that promoted
something, on the reasoning that a rerun of the release already current changes
nothing a reader would see. That reasoning has a hole. The release id is the
rows digest, so a release whose *provenance* was corrected republishes under the
same id -- the rows are identical, nothing is promoted, and the manifest at the
destination is replaced with the corrected one. The attestation beside it was
left naming the value the correction had replaced, and the attestation, not the
manifest, is what ``assert_canary_attested`` reads before an activation.

So these tests are about one question asked two ways: after a successful canary
publish, does ``canary_attestation.json`` describe the release ``current.json``
names -- and does getting that right leave everything else at the destination
exactly as it was.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_publisher_fixture as fixture  # noqa: E402

#: The production SHA the end-to-end fixture used to emit, and the value a real
#: canary's attestation was found still carrying after the manifest beside it
#: had been corrected. Used here as the stale value, because it is the one this
#: defect was actually found with.
STALE_SHA = "e2e" + "0" * 37

#: Two real-shaped production SHAs, distinct from each other and from the stale
#: one. Which commit served a slate is not something this publisher has an
#: opinion about; what matters is that the attestation names the one the
#: current manifest names.
FIRST_SHA = "1186feff8a742f640a23b6bc59bd174279d2204d"

SECOND_SHA = "f38cf0c2b10387cdac08f639c977dee14dc46b5b"


def build_once(root: Path, sha: str | None) -> dict:
    return fixture.build(root, receipt_changes={"production_code_sha": sha})


def correct_receipt_sha(built: dict, sha: str) -> None:
    """Rewrite only the receipt's code SHA, leaving the priced bytes alone.

    The tightest form of the regression condition: the same served artifact,
    the same rows, the same release id, and one provenance field corrected. A
    rerun from a freshly built fixture would also change the artifact path and
    the served-at stamp, and then a manifest that differed for three reasons
    could not say which one the attestation failed to follow.
    """
    receipt = json.loads(built["receipt_path"].read_text(encoding="utf-8"))

    receipt["production_code_sha"] = sha

    built["receipt_path"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    built["receipt"] = receipt


def publish_into(destination: Path, built: dict) -> dict:
    """One canary into ``destination`` from an already-built fixture.

    The priced rows are the fixture's own, so every publication here lands on
    the same release id. That is the point: it is what makes a second call a
    no-op republish rather than a new release, which is the condition the
    defect lived in.
    """
    return publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=built["slate_date"],
        data_root=built["data_root"],
        project_root=PROJECT,
        destination_root=destination,
    )


def publish_with_sha(destination: Path, root: Path, sha: str | None) -> dict:
    """A fresh fixture naming ``sha``, published into ``destination``."""
    built = build_once(root, sha)

    return {"built": built, "receipt": publish_into(destination, built)}


def destination_state(destination: Path) -> dict:
    """Everything the destination holds, by name, as bytes."""
    transport = publisher.LocalDirectoryTransport(destination)

    return {name: transport.read_bytes(name) for name in transport.listing()}


def attestation_of(destination: Path) -> dict:
    return json.loads(
        (destination / publisher.ATTESTATION_NAME).read_text(encoding="utf-8")
    )


def current_of(destination: Path) -> dict:
    return json.loads(
        (destination / publisher.CURRENT_POINTER).read_text(encoding="utf-8")
    )


def manifest_of(destination: Path) -> dict:
    return json.loads(
        (destination / current_of(destination)["manifest_relative"]).read_text(
            encoding="utf-8"
        )
    )


def releases_in(destination: Path) -> list[str]:
    root = destination / publisher.RELEASES_DIRECTORY

    if not root.is_dir():
        return []

    return sorted(path.name for path in root.iterdir())


@pytest.fixture
def stale(tmp_path: Path) -> dict:
    """A destination serving a release whose attestation is about to go stale.

    Staged the way it arose rather than by writing a doctored file: publish the
    slate once with the SHA that was wrong, which leaves an attestation naming
    it, then correct that one field in the receipt. The publication under test
    is the rerun, which promotes nothing because the rows did not change.
    """
    destination = tmp_path / "destination"

    built = build_once(tmp_path / "run", STALE_SHA)

    first = publish_into(destination, built)

    assert attestation_of(destination)["production_code_sha"] == STALE_SHA
    assert manifest_of(destination)["provenance"]["production_code_sha"] == STALE_SHA

    correct_receipt_sha(built, FIRST_SHA)

    return {
        "before": destination_state(destination),
        "built": built,
        "destination": destination,
        "first": {"receipt": first},
        "republish": lambda: publish_into(destination, built),
        "tmp_path": tmp_path,
    }


# ======================================================================
# 1 to 4. the refresh, and what it has to agree with
# ======================================================================


def test_a_no_op_republish_refreshes_a_stale_attestation(stale: dict):
    """1. The defect, as it was found: same release, corrected manifest."""
    destination = stale["destination"]

    second = stale["republish"]()

    assert second["promotion"]["promoted"] is False
    assert (
        second["release"]["release_id"]
        == stale["first"]["receipt"]["release"]["release_id"]
    )
    assert second["audit_relative"] == publisher.AUDIT_LOG_NAME

    assert second["attestation"]["refreshed"] is True
    assert (
        second["attestation"]["reason"]
        == "the attestation did not describe the current release"
    )

    attestation = attestation_of(destination)

    assert attestation["production_code_sha"] == FIRST_SHA
    assert attestation["production_code_sha"] != STALE_SHA


def test_the_refreshed_attestation_names_the_current_manifests_sha(stale: dict):
    """2. Against the manifest at the destination, not against this run."""
    destination = stale["destination"]

    stale["republish"]()

    assert (
        attestation_of(destination)["production_code_sha"]
        == manifest_of(destination)["provenance"]["production_code_sha"]
    )


def test_the_refreshed_attestation_names_the_current_release(stale: dict):
    """3 and 4. Release id and rows digest, both read off current.json."""
    destination = stale["destination"]

    stale["republish"]()

    attestation = attestation_of(destination)

    current = current_of(destination)

    assert attestation["release_id"] == current["release_id"]
    assert attestation["rows_sha256"] == current["rows_sha256"]
    assert attestation["rows_sha256"] == manifest_of(destination)["rows_sha256"]
    assert attestation["slate_date"] == manifest_of(destination)["slate_date"]
    assert attestation["model_authority"] == publisher.MODEL_AUTHORITY
    assert attestation["mode"] == publisher.MODE_CANARY
    assert (
        attestation["source_artifact_sha256"]
        == manifest_of(destination)["provenance"]["priced_market_artifact_sha256"]
    )


# ======================================================================
# 5 to 9. and nothing else moved
# ======================================================================


def test_the_refresh_moves_only_the_attestation_the_manifest_and_the_log(
    stale: dict,
):
    """5, 6 and 7 at once, by comparing every byte the destination holds.

    Asserted as a whole-destination diff rather than as separate assertions
    about separate files, because the claim is that nothing else moved and a
    per-file check cannot say that.

    Three files are expected to differ and each for its own reason. The
    manifest differs because correcting the provenance is what this rerun is
    for, and rewriting the current release's manifest in place is the
    pre-existing behaviour that made the stale attestation possible. The audit
    log differs because it gained a line. The attestation differs because it
    now describes the corrected manifest, which is the fix.
    """
    destination = stale["destination"]

    stale["republish"]()

    after = destination_state(destination)

    moved = sorted(
        name
        for name in set(after) | set(stale["before"])
        if after.get(name) != stale["before"].get(name)
    )

    assert moved == sorted(
        (
            publisher.AUDIT_LOG_NAME,
            publisher.ATTESTATION_NAME,
            current_of(destination)["manifest_relative"],
        )
    ), "the refresh changed something it had no business changing"


def test_the_refresh_leaves_the_published_rows_untouched(stale: dict):
    """The payload a consumer reads is not rewritten to correct provenance."""
    destination = stale["destination"]

    rows_relative = current_of(destination)["rows_relative"]

    stale["republish"]()

    assert (
        destination_state(destination)[rows_relative]
        == stale["before"][rows_relative]
    )


def test_the_refresh_does_not_touch_the_current_pointer(stale: dict):
    """5. Stated on its own, because it is the pointer a consumer follows."""
    destination = stale["destination"]

    before = current_of(destination)

    stale["republish"]()

    assert current_of(destination) == before


def test_the_refresh_does_not_rotate_the_previous_pointer(stale: dict):
    """6. Rotating here would overwrite the real rollback target."""
    destination = stale["destination"]

    assert not (destination / publisher.PREVIOUS_POINTER).exists()

    stale["republish"]()

    assert not (destination / publisher.PREVIOUS_POINTER).exists()


def test_the_refresh_creates_no_release(stale: dict):
    """7. One release before, the same one release after."""
    destination = stale["destination"]

    before = releases_in(destination)

    assert len(before) == 1

    stale["republish"]()

    assert releases_in(destination) == before


def test_the_historical_audit_entries_are_untouched(stale: dict):
    """8 and 9. History is appended to, and what is appended is true.

    The audit log is the other half of this defect's answer: the attestation is
    corrected in place because it describes now, and the log is not because it
    describes then. The earlier line still records the SHA that run actually
    published, including when that SHA was the wrong one.
    """
    destination = stale["destination"]

    before = publisher.read_audit_log(
        publisher.LocalDirectoryTransport(destination)
    )

    before_bytes = (destination / publisher.AUDIT_LOG_NAME).read_bytes()

    stale["republish"]()

    after_bytes = (destination / publisher.AUDIT_LOG_NAME).read_bytes()

    assert after_bytes.startswith(before_bytes), (
        "the refresh rewrote history instead of appending to it"
    )

    after = publisher.read_audit_log(
        publisher.LocalDirectoryTransport(destination)
    )

    assert after[: len(before)] == before

    assert [entry["action"] for entry in after] == [
        publisher.ACTION_PUBLISHED,
        publisher.ACTION_REPUBLISHED,
    ]

    # 9. The appended line is truthful about the run that appended it, and the
    # earlier one is still truthful about the run that was wrong.
    assert before[0]["production_code_sha"] == STALE_SHA
    assert after[0]["production_code_sha"] == STALE_SHA
    assert after[-1]["production_code_sha"] == FIRST_SHA
    assert after[-1]["release_id"] == current_of(destination)["release_id"]


# ======================================================================
# 10. an attestation that is already right stays byte-identical
# ======================================================================


def test_an_already_correct_attestation_is_left_alone(tmp_path: Path):
    """10. Otherwise every rerun would restamp the destination.

    The attestation carries the moment it was written, so a refresh that did
    not first ask whether anything had changed would rewrite the file on every
    run and a no-op rerun would stop being a no-op.
    """
    destination = tmp_path / "destination"

    built = build_once(tmp_path / "run", FIRST_SHA)

    publish_into(destination, built)

    before = (destination / publisher.ATTESTATION_NAME).read_bytes()

    second = publish_into(destination, built)

    assert second["attestation"]["refreshed"] is False
    assert (
        second["attestation"]["reason"]
        == "the attestation already describes the current release"
    )
    assert (destination / publisher.ATTESTATION_NAME).read_bytes() == before

    third = publish_into(destination, built)

    assert third["attestation"]["refreshed"] is False
    assert (destination / publisher.ATTESTATION_NAME).read_bytes() == before


def test_a_second_correction_refreshes_again(tmp_path: Path):
    """Not a one-off repair: the attestation tracks whatever is current."""
    destination = tmp_path / "destination"

    built = build_once(tmp_path / "run", STALE_SHA)

    publish_into(destination, built)

    for sha in (FIRST_SHA, SECOND_SHA):
        correct_receipt_sha(built, sha)

        receipt = publish_into(destination, built)

        assert receipt["promotion"]["promoted"] is False
        assert receipt["attestation"]["refreshed"] is True
        assert attestation_of(destination)["production_code_sha"] == sha
        assert (
            attestation_of(destination)["production_code_sha"]
            == manifest_of(destination)["provenance"]["production_code_sha"]
        )


def test_a_destination_with_no_attestation_gets_one(tmp_path: Path):
    """The first canary still writes one, which is the old behaviour kept."""
    destination = tmp_path / "destination"

    published = publish_with_sha(destination, tmp_path / "first", FIRST_SHA)

    assert published["receipt"]["attestation"]["refreshed"] is True
    assert (
        published["receipt"]["attestation"]["reason"]
        == "the destination had no attestation"
    )
    assert attestation_of(destination)["production_code_sha"] == FIRST_SHA


def test_a_corrupt_attestation_is_replaced(tmp_path: Path):
    """Unreadable is not the same as absent, and both have to end up correct."""
    destination = tmp_path / "destination"

    built = build_once(tmp_path / "run", FIRST_SHA)

    publish_into(destination, built)

    (destination / publisher.ATTESTATION_NAME).write_bytes(b"{ not json")

    second = publish_into(destination, built)

    assert second["attestation"]["refreshed"] is True
    assert attestation_of(destination)["production_code_sha"] == FIRST_SHA


# ======================================================================
# 11, 12 and 13. the refresh fails closed
# ======================================================================
#
# Reached by calling the refresh against a doctored destination rather than
# through publish(), because publish() writes and verifies the release before
# it gets here -- which is the right ordering and also means these states are
# not reachable from the outside. What is under test is that the refresh does
# its own checking rather than trusting the caller to have done it.


def staged_release(tmp_path: Path) -> dict:
    """A published destination, plus the release object that published it."""
    destination = tmp_path / "destination"

    built = fixture.build(
        tmp_path / "run", receipt_changes={"production_code_sha": FIRST_SHA}
    )

    receipt = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=built["slate_date"],
        data_root=built["data_root"],
        project_root=PROJECT,
        destination_root=destination,
    )

    transport = publisher.LocalDirectoryTransport(destination)

    current = json.loads(transport.read_bytes(publisher.CURRENT_POINTER))

    release = publisher.Release(
        release_id=current["release_id"],
        rows_relative=current["rows_relative"],
        manifest_relative=current["manifest_relative"],
        rows_bytes=transport.read_bytes(current["rows_relative"]),
        manifest_bytes=transport.read_bytes(current["manifest_relative"]),
        rows_sha256=current["rows_sha256"],
        manifest_sha256=receipt["release"]["manifest_sha256"],
        row_count=receipt["release"]["row_count"],
    )

    return {
        "current": current,
        "destination": destination,
        "receipt": receipt,
        "release": release,
        "transport": transport,
    }


def refresh(staged: dict):
    return publisher.ensure_canary_attestation(
        staged["transport"],
        release=staged["release"],
        verification=staged["receipt"]["verification"],
    )


def test_a_missing_current_pointer_refuses_the_refresh(tmp_path: Path):
    """11. Nothing is current, so there is nothing to describe."""
    staged = staged_release(tmp_path)

    (staged["destination"] / publisher.CURRENT_POINTER).unlink()

    with pytest.raises(publisher.PublishingRefused, match="names no release"):
        refresh(staged)


def test_a_missing_current_manifest_refuses_the_refresh(tmp_path: Path):
    """11. The pointer names a manifest that is not there."""
    staged = staged_release(tmp_path)

    (staged["destination"] / staged["current"]["manifest_relative"]).unlink()

    with pytest.raises(publisher.PublishingRefused, match="not at the"):
        refresh(staged)


def test_an_unreadable_current_manifest_refuses_the_refresh(tmp_path: Path):
    """11. Present but not JSON is not a manifest either."""
    staged = staged_release(tmp_path)

    (staged["destination"] / staged["current"]["manifest_relative"]).write_bytes(
        b"{ not json"
    )

    with pytest.raises(publisher.PublishingRefused, match="not readable JSON"):
        refresh(staged)


def test_rows_that_read_back_wrong_refuse_the_refresh(tmp_path: Path):
    """12. The destination is not serving the release it says it is."""
    staged = staged_release(tmp_path)

    rows = staged["destination"] / staged["current"]["rows_relative"]

    rows.write_bytes(rows.read_bytes() + b"\n")

    with pytest.raises(publisher.PublishingRefused, match="reads back as"):
        refresh(staged)


def test_missing_rows_refuse_the_refresh(tmp_path: Path):
    """12. Nothing to attest when the payload is gone."""
    staged = staged_release(tmp_path)

    (staged["destination"] / staged["current"]["rows_relative"]).unlink()

    with pytest.raises(publisher.PublishingRefused, match="nothing to attest"):
        refresh(staged)


def test_a_non_incumbent_current_release_refuses_the_refresh(tmp_path: Path):
    """13. Only the incumbent is published, so only the incumbent is attested."""
    staged = staged_release(tmp_path)

    path = staged["destination"] / staged["current"]["manifest_relative"]

    manifest = json.loads(path.read_text(encoding="utf-8"))

    manifest["model_authority"] = "candidate"

    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(publisher.PublishingRefused, match="model authority"):
        refresh(staged)


@pytest.mark.parametrize("field", publisher.ATTESTABLE_PROVENANCE_FIELDS)
def test_incomplete_current_provenance_refuses_the_refresh(
    tmp_path: Path, field: str
):
    """A manifest that provenances nothing is not attested as if it did."""
    staged = staged_release(tmp_path)

    path = staged["destination"] / staged["current"]["manifest_relative"]

    manifest = json.loads(path.read_text(encoding="utf-8"))

    manifest["provenance"].pop(field)

    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        publisher.PublishingRefused, match=f"required provenance: {field}"
    ):
        refresh(staged)


def test_a_dropped_provenance_key_refuses_the_refresh(tmp_path: Path):
    """Absent is refused where null is accepted, as everywhere else here."""
    staged = staged_release(tmp_path)

    path = staged["destination"] / staged["current"]["manifest_relative"]

    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["provenance"]["incumbent_fit_id"] is None

    manifest["provenance"].pop("incumbent_fit_id")

    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        publisher.PublishingRefused, match="required provenance: incumbent_fit_id"
    ):
        refresh(staged)


@pytest.mark.parametrize(
    "field", ["release_id", "rows_relative", "rows_sha256"]
)
def test_a_pointer_that_disagrees_with_the_manifest_refuses(
    tmp_path: Path, field: str
):
    """11. Two documents that disagree settle nothing, so nothing is attested."""
    staged = staged_release(tmp_path)

    path = staged["destination"] / publisher.CURRENT_POINTER

    current = json.loads(path.read_text(encoding="utf-8"))

    current[field] = "something-else"

    path.write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(publisher.PublishingRefused):
        refresh(staged)


def test_a_manifest_that_is_not_the_one_written_refuses(tmp_path: Path):
    """The catch-all: anything the named checks did not already refuse."""
    staged = staged_release(tmp_path)

    path = staged["destination"] / staged["current"]["manifest_relative"]

    manifest = json.loads(path.read_text(encoding="utf-8"))

    manifest["row_count"] = manifest["row_count"] + 1

    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        publisher.PublishingRefused, match="does not hold the manifest"
    ):
        refresh(staged)


def test_a_refused_refresh_leaves_the_attestation_alone(tmp_path: Path):
    """Fail closed means the stale file stays stale rather than becoming wrong.

    A refusal that had already half-written the attestation would be worse than
    no refresh at all: the destination would hold a document assembled from a
    state the publisher had just declined to believe.
    """
    staged = staged_release(tmp_path)

    before = (staged["destination"] / publisher.ATTESTATION_NAME).read_bytes()

    (staged["destination"] / staged["current"]["rows_relative"]).unlink()

    with pytest.raises(publisher.PublishingRefused):
        refresh(staged)

    assert (
        staged["destination"] / publisher.ATTESTATION_NAME
    ).read_bytes() == before


# ======================================================================
# 14 and 15. the gates this change is not allowed to move
# ======================================================================


def test_live_mode_still_refuses_and_writes_no_attestation(tmp_path: Path):
    """14. The switch is DISABLED, and the refresh is a canary-only step."""
    built = fixture.build(
        tmp_path / "run", receipt_changes={"production_code_sha": FIRST_SHA}
    )

    with pytest.raises(publisher.PublishingRefused):
        publisher.publish(
            mode=publisher.MODE_LIVE,
            slate_date=built["slate_date"],
            data_root=built["data_root"],
            project_root=PROJECT,
            destination_root=tmp_path / "destination",
        )

    assert not (tmp_path / "destination").exists()

    # The live state is set in the committed switch, and the gate it is one
    # of three conditions for is still shut, which is why nothing above was
    # written.
    assert (
        publisher.read_publishing_switch(
            publisher.MODE_LIVE, project_root=PROJECT, environment={}
        ).enabled
        is False
    )


def test_a_shadow_candidate_slate_still_refuses(tmp_path: Path):
    """15. Refused before any of this runs, so no attestation is written."""
    import pandas as pd

    frame = fixture.priced_frame()

    frame[publisher.CANDIDATE_SOURCE_COLUMNS[0]] = 0.5

    built = fixture.build(
        tmp_path / "run",
        frame=frame,
        receipt_changes={"production_code_sha": FIRST_SHA},
    )

    assert isinstance(built["receipt"], dict)
    assert isinstance(frame, pd.DataFrame)

    with pytest.raises(publisher.PublishingRefused):
        publisher.publish(
            mode=publisher.MODE_CANARY,
            slate_date=built["slate_date"],
            data_root=built["data_root"],
            project_root=PROJECT,
            destination_root=tmp_path / "destination",
        )

    assert not (
        tmp_path / "destination" / publisher.ATTESTATION_NAME
    ).exists()


# ======================================================================
# the activation gate still reads what a canary left
# ======================================================================


def test_a_refreshed_attestation_is_still_one_an_activation_accepts(
    tmp_path: Path,
):
    """The refresh produces an attestation, not a differently shaped file.

    ``assert_canary_attested`` is the only consumer of this document, so a
    refresh that changed its shape would have fixed the SHA and broken the gate
    that reads it.
    """
    destination = tmp_path / "destination"

    built = build_once(tmp_path / "run", STALE_SHA)

    publish_into(destination, built)

    correct_receipt_sha(built, FIRST_SHA)

    second = publish_into(destination, built)

    attestation = attestation_of(destination)

    attested = publisher.assert_canary_attested(
        attestation,
        release=publisher.Release(
            release_id=second["release"]["release_id"],
            rows_relative=second["release"]["rows_relative"],
            manifest_relative=second["release"]["manifest_relative"],
            rows_bytes=b"",
            manifest_bytes=b"",
            rows_sha256=second["release"]["rows_sha256"],
            manifest_sha256=second["release"]["manifest_sha256"],
            row_count=second["release"]["row_count"],
        ),
        provenance=second["provenance"],
    )

    assert attested["release_id"] == current_of(destination)["release_id"]
    assert attested["rows_sha256"] == current_of(destination)["rows_sha256"]
