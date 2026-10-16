"""The whole path: frozen models score a slate, and the canary publishes it.

Everything in ``test_wizardofodds_incumbent_publisher.py`` is about the
publisher's rules, and it builds its priced rows directly because those rules
do not care where the numbers came from. This file is the other half, and it
cares about nothing else: the real frozen runtime bundle, the real prediction
and pricing scripts, the real serving receipt, and the publisher reading that
output rather than a fixture shaped like it.

It is the test that would catch the publisher being a contract for a feed the
runtime does not actually emit -- a column that is named something else, a
dtype that does not survive the round trip, a fail-closed rule that rejects
every genuine row. None of that is visible against a hand-built frame, because
a hand-built frame is built to match.

Skipped without an installed bundle. The twelve model binaries total 386MB and
are published as a release asset precisely so they are not in Git.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import frozen_bundle_slate_fixture as slate_fixture  # noqa: E402
import nba_distribution_publication as distribution_stage  # noqa: E402
import production_lifecycle_preflight as preflight  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import run_incumbent_production_serving as serving  # noqa: E402
from test_real_slate_serving_root import (  # noqa: E402
    installed_bundle,
    serve_the_fixture,
)
from install_frozen_model_artifacts import verify_frozen_bundle  # noqa: E402

#: The shape of a resolved commit. Necessary and nowhere near sufficient: the
#: placeholder this fixture used to emit was forty hex characters too, which is
#: exactly why it went unnoticed. Shape is checked first only so that an
#: obviously malformed value is reported as malformed.
FULL_COMMIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")


def names_a_commit(sha: str) -> bool:
    """Whether ``sha`` is a commit this repository actually holds.

    The property that separates a code SHA from something merely shaped like
    one. A stand-in can satisfy any amount of pattern matching; it cannot be
    an object in the object database.
    """
    if not FULL_COMMIT_SHA.fullmatch(sha or ""):
        return False

    return preflight.git("cat-file", "-t", sha) == "commit"


def checkout_sha() -> str:
    """The commit that generated this fixture's served slate.

    The lifecycle resolves the checkout and passes it to serving as
    ``--production-sha``, and ``preflight.git`` is the helper that resolves it
    there, so the fixture provenances itself exactly the way production does
    rather than inventing its own answer.

    Refused rather than defaulted when it will not resolve. A served slate
    whose code SHA is a placeholder provenances nothing, and publishing that
    placeholder to the destination is the defect this replaced.
    """
    sha = preflight.git("rev-parse", "HEAD")

    if not names_a_commit(sha):
        raise AssertionError(
            "the checkout's HEAD does not resolve to a commit "
            f"({sha or 'nothing'}), so this fixture cannot provenance the "
            "slate it serves with the SHA that generated it"
        )

    return sha


@pytest.fixture(scope="module")
def frozen_bundle() -> Path:
    bundle = installed_bundle()

    if bundle is None:
        pytest.skip(
            "no installed frozen runtime bundle; set "
            "NBA_PROP_TEST_FROZEN_BUNDLE_ROOT or NBA_PROP_WORK_DIR to a root "
            "ops/install_frozen_model_artifacts.py has installed into"
        )

    verify_frozen_bundle(bundle)

    return bundle


def serve_and_provenance(bundle: Path, root: Path, *, production_sha: str) -> dict:
    """A real served slate, with the receipt production would have written.

    ``build_receipt`` rather than a hand-written dict, and
    ``missing_provenance`` asserted, so the publisher is handed the same thing
    the lifecycle hands the grader -- including the parts that would make it
    refuse if they were absent.

    ``production_sha`` is required of the caller, because the commit that
    generated a served slate is a fact about that run and not something a
    fixture helper is entitled to decide. Callers serving the real slate pass
    :func:`checkout_sha`.
    """
    assert names_a_commit(production_sha), (
        f"{production_sha!r} is not a commit this repository holds, so a "
        "receipt carrying it would provenance this slate to nothing"
    )

    served = serve_the_fixture(bundle, root)

    assert served["outcome"] == serving.OUTCOME_SERVED

    receipt = serving.build_receipt(
        slate_date=slate_fixture.SLATE_DATE,
        production_sha=production_sha,
        incumbent={
            "authority": serving.AUTHORITY_FROZEN_BUNDLE,
            "bundle_root": str(bundle),
            "fit_id": None,
            "model_dir": str(bundle / "models"),
            "promoted_at": None,
            "reason": "the frozen bundle is the incumbent",
            "version": bundle.name.split("__")[0],
        },
        readiness={
            "ready": True,
            "reason": "the fixture slate is a production slate",
            "outcome": serving.REFRESH_OUTCOME_REFRESHED,
        },
        fingerprint="end-to-end-input-state-fingerprint",
        served=served,
    )

    assert serving.missing_provenance(receipt) == []

    receipt_path = publisher.serving_receipt_path(
        root / "data", slate_fixture.SLATE_DATE
    )

    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # The stage the lifecycle runs between serving and publication, here for
    # the same reason the receipt is built by serving's own builder: the
    # publisher is handed what production hands it. It reads the parquet the
    # serving run just wrote and the bundle that priced it, so the forecasts
    # these suites publish are the frozen engines' own and not a fixture's.
    assert (
        distribution_stage.main(
            [
                "--date",
                slate_fixture.SLATE_DATE,
                "--data-root",
                str(root / "data"),
                "--model-dir",
                str(bundle / "models"),
            ]
        )
        == 0
    )

    return {"receipt": receipt, "served": served}


def test_the_served_slate_publishes_to_a_canary_and_reads_back(
    frozen_bundle: Path, tmp_path: Path
):
    """Predictions, priced rows, published rows, and the readback, in one run.

    The row count is asserted against the fixture's quote list rather than
    against a literal, so a run that silently published a subset is a count
    mismatch rather than a number nobody checks.
    """
    root = tmp_path / "run"

    provenanced = serve_and_provenance(
        frozen_bundle, root, production_sha=checkout_sha()
    )

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

    assert len(projections) == len(slate_fixture.ROSTER) > 0
    assert len(priced) == len(slate_fixture.QUOTED_PROPS) > 0

    live_root = tmp_path / "live"

    live_root.mkdir()

    (live_root / publisher.CURRENT_POINTER).write_text(
        json.dumps({"release_id": "a-live-release-that-must-not-move"}),
        encoding="utf-8",
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

    live_before = {name: live.read_bytes(name) for name in live.listing()}

    receipt = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=slate_fixture.SLATE_DATE,
        data_root=data_root,
        project_root=PROJECT,
        destination_root=tmp_path / "canary",
        destinations_path=destinations,
    )

    assert receipt["outcome"] == publisher.OUTCOME_PUBLISHED
    assert receipt["mode"] == publisher.MODE_CANARY
    assert receipt["model_authority"] == publisher.MODEL_AUTHORITY
    assert receipt["live_publishing"] == publisher.STATE_DISABLED
    assert receipt["verification"]["verified"] is True
    assert receipt["release"]["row_count"] == len(slate_fixture.QUOTED_PROPS)
    assert receipt["verification"]["readback_rows"] == len(priced)

    published = json.loads(
        (tmp_path / "canary" / receipt["release"]["rows_relative"]).read_text(
            encoding="utf-8"
        )
    )

    assert len(published["rows"]) == len(priced) > 0

    # The live destination, byte for byte, before and after.
    assert {
        name: live.read_bytes(name) for name in live.listing()
    } == live_before, "the canary altered the live destination"

    served_receipt = provenanced["receipt"]

    provenance = receipt["provenance"]

    assert provenance["model_authority"] == publisher.MODEL_AUTHORITY
    assert provenance["production_code_sha"] == served_receipt["production_code_sha"]
    # Not merely equal to the receipt, but equal to the commit that produced
    # the slate. Equality alone was true of the placeholder this replaced.
    assert provenance["production_code_sha"] == checkout_sha()
    assert (
        provenance["input_state_fingerprint"]
        == served_receipt["input_state_fingerprint"]
    )
    assert provenance["incumbent_version"] == served_receipt["incumbent_version"]
    assert "incumbent_fit_id" in provenance
    assert (
        provenance["priced_market_artifact_sha256"]
        == served_receipt["priced_market_artifact"]["sha256"]
    )


def test_every_published_number_is_the_frozen_models_own(
    frozen_bundle: Path, tmp_path: Path
):
    """Exact equality against the priced parquet, field by field, row by row.

    This is the property the integration contract turns on -- the consumer
    "must not recalibrate probabilities" -- and a publisher is the first place
    it can quietly stop being true. Exact rather than approximate: a publisher
    that round-tripped a float through a formatter would pass an approximate
    check and would be changing the published price.
    """
    root = tmp_path / "run"

    serve_and_provenance(frozen_bundle, root, production_sha=checkout_sha())

    data_root = root / "data"

    priced = pd.read_parquet(
        publisher.priced_markets_path(data_root, slate_fixture.SLATE_DATE)
    )

    receipt = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=slate_fixture.SLATE_DATE,
        data_root=data_root,
        project_root=PROJECT,
        destination_root=tmp_path / "canary",
    )

    published = json.loads(
        (tmp_path / "canary" / receipt["release"]["rows_relative"]).read_text(
            encoding="utf-8"
        )
    )["rows"]

    by_quote = {int(row["quote_id"]): row for row in published}

    assert sorted(by_quote) == sorted(int(value) for value in priced["id"])

    numeric = (
        *publisher.PUBLISHED_PROBABILITY_FIELDS,
        "calibrated_edge_over",
        "calibrated_edge_under",
        "calibrated_ev_over",
        "calibrated_ev_under",
        "model_preferred_edge",
        "model_preferred_ev",
        "calibrated_fair_over_american",
        "calibrated_fair_under_american",
        "market_hold_pct",
        "over_odds",
        "under_odds",
        "line_value",
    )

    textual = (
        "model_preferred_side",
        "prop_type",
        "calibration_method",
        "freeze_id",
        "freeze_stage",
        "manifest_sha256",
        "vendor",
        "market_type",
    )

    for _, source in priced.iterrows():
        out = by_quote[int(source["id"])]

        for field in numeric:
            assert float(out[field]) == float(source[field]), (
                f"{field} on quote {source['id']} is not the served value"
            )

        for field in textual:
            assert out[field] == source[field], (
                f"{field} on quote {source['id']} is not the served value"
            )

        assert out["player_id"] == int(source["player_id"])
        assert out["game_id"] == int(source["game_id"])
        assert out["date"] == pd.Timestamp(source["date"]).date().isoformat()

    # Nothing beyond the declared contract reached the destination.
    assert set(published[0]) == {name for _, name in publisher.PUBLISHED_FIELDS}


def test_republishing_the_same_served_slate_is_a_no_op(
    frozen_bundle: Path, tmp_path: Path
):
    """Idempotency on the real artifact, not on a fixture.

    The release id is the payload digest, and the payload is a projection of
    the frozen models' output, so running the publisher twice over one served
    slate has to land on the same release. If it did not, every rerun would
    stack a release at the destination and rotate away the rollback target.
    """
    root = tmp_path / "run"

    serve_and_provenance(frozen_bundle, root, production_sha=checkout_sha())

    destination = tmp_path / "canary"

    arguments = {
        "mode": publisher.MODE_CANARY,
        "slate_date": slate_fixture.SLATE_DATE,
        "data_root": root / "data",
        "project_root": PROJECT,
        "destination_root": destination,
    }

    first = publisher.publish(**arguments)

    transport = publisher.LocalDirectoryTransport(destination)

    listing = transport.listing()

    second = publisher.publish(**arguments)

    assert second["release"]["release_id"] == first["release"]["release_id"]
    assert second["release"]["rows_sha256"] == first["release"]["rows_sha256"]
    assert second["promotion"]["promoted"] is False
    assert transport.listing() == listing
    assert not transport.exists(publisher.PREVIOUS_POINTER)
