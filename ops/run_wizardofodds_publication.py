#!/usr/bin/env python3
"""Publish the slate the incumbent just served, then the page that reads it.

WHAT THIS IS FOR
----------------

Both halves of the WizardOfOdds publication already worked and both were run
by hand. The feed publisher took the priced artifact and the serving receipt
and wrote a release; the page builder read a release and wrote the page. What
was missing between them was the only thing that cannot be supplied by an
operator remembering to run two commands in order: a statement, checkable
afterwards, that the page a reader fetches was built from the release this run
published and not from whichever release happened to be current by the time
the page was built.

That is not a hypothetical. The remote feed canary spent weeks serving a
contract-1 release while the renderer required contract 2, so an automation
that published the feed and then built the page from "whatever is current"
would have been one scheduling accident away from a page about a slate from
before -- rendering, verifying and reporting success the whole way.

So this sequences the two existing entry points and refuses every ordering
that is not the safe one. It computes nothing: no probability, no price, no
edge, no expected value, no row and no page byte originates here. Every number
it publishes was produced by the serving step and every byte of the page was
rendered by the page builder.

THE TWO STAGES
--------------

``--stage feed`` publishes the release and then reads back what the
destination is actually serving, which is where the contract version is
established. ``--stage page`` reads that stage's receipt, requires it to
record a successful contract-2 publication of this slate, and pins the page
build to the exact release id and rows digest it names.

Two stages rather than one because the dependency is the point, and GitHub
enforces step order and fail-fast for free: a failed feed stage means the page
stage does not run at all. The pin means that even if it somehow did, it could
not build a page from anything else.

The page build and the page publication are one invocation because the page
builder writes and verifies in one operation -- it renders, writes both files,
reads them back through the transport and fetches them over HTTPS, and
refusing partway through is the behaviour that makes the readback meaningful.
Splitting them would mean writing page bytes somewhere and publishing them
later from a place nothing verified.

FAIL CLOSED, AND NEVER FROM STALE BYTES
---------------------------------------

Every refusal below stops the lifecycle before the next publication stage can
claim success. A slate that was not served publishes nothing and says so with
exit 0, because a preseason day with no games is not a failure; everything
else is a failure and goes red.

Nothing here reuses an artifact from an earlier run to get a green result. The
priced artifact, the serving receipt and the production SHA all have to be
this run's, and the release the page reads has to be the one this run's feed
stage published.

LIVE
----

``--mode live`` is addressed and disabled. It resolves the live feed and the
live page destinations, which are fully described in the destination
configuration, and the feed publisher's committed switch refuses it: live
publication needs that switch set to ENABLED in a reviewed commit, the publish
variable set to the same state and the approval token supplied separately.
This entry point adds no fourth condition and relaxes none of the three, so a
later reviewed activation changes an approval rather than this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ops"))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
import wizardofodds_fallback_destinations as fallback_destinations  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402

STAGE_FEED = "feed"
STAGE_PAGE = "page"
STAGES: tuple[str, ...] = (STAGE_FEED, STAGE_PAGE)

#: The feed destination each mode publishes to. The page destinations live in
#: :data:`wizardofodds_player_props_page.MODE_DESTINATIONS`, next to the
#: prefix each mode writes under, rather than being restated here.
FEED_DESTINATIONS: dict[str, str] = {
    publisher.MODE_CANARY: "wizardofodds_nba_canary",
    publisher.MODE_LIVE: "wizardofodds_nba_live",
}

#: What the serving step records on a day it produced the artifact to publish.
#: Stated rather than imported, because importing the serving entry point to
#: read one string would put the whole prediction and pricing stack behind the
#: publication stages. A test pins it against
#: ``run_incumbent_production_serving.OUTCOME_SERVED``, so the two cannot
#: drift without something saying so.
SERVED_OUTCOME = "SERVED"

#: The outcomes that mean there was legitimately nothing to publish: a date
#: before opening day, and a date with no games on it. Both are successes
#: everywhere else in the lifecycle and both are successes here.
#:
#: Enumerated rather than inferred from "not SERVED", because the difference
#: matters: a serving step that *failed* has produced no artifact for the same
#: reason an off day has, and treating the two alike would turn a broken
#: serving run into a quiet no-op with yesterday's page still up. Pinned
#: against the serving module's own outcomes by a test.
NO_OP_OUTCOMES: tuple[str, ...] = ("NO_PRODUCTION_SLATE", "NO_GAMES_ON_SLATE")

OUTCOME_PUBLISHED = "PUBLISHED"
OUTCOME_NOT_SERVED = "NOT_SERVED_NOTHING_TO_PUBLISH"

EXIT_OK = 0
EXIT_REFUSED = 1


class PublicationStageFailed(RuntimeError):
    """A stage refused, and no later stage may claim success."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path, *, what: str) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise PublicationStageFailed(
            f"{path} could not be read ({error}), so there is no {what}"
        ) from error
    except ValueError as error:
        raise PublicationStageFailed(
            f"{path} is not readable JSON ({error}), so the {what} it should "
            "describe cannot be trusted"
        ) from error

    if not isinstance(document, dict):
        raise PublicationStageFailed(
            f"{path} is not a JSON object, so it does not describe a {what}"
        )

    return document


# ----------------------------------------------------------------------
# what the serving step left behind
# ----------------------------------------------------------------------


def read_serving_receipt(
    path: Path, *, slate_date: str, data_root: Path
) -> dict[str, Any]:
    """The receipt this run's serving step wrote, checked against this run.

    The slate date is checked because the receipt is the one thing that names
    which slate the priced artifact is of, and publishing yesterday's rows
    under today's date is the mistake that no later check could see: every
    digest would agree and the page would be internally consistent and wrong.

    The priced artifact's recorded digest is returned rather than verified
    here. The feed publisher verifies it against the file it is about to read,
    which is the only place that check means anything, and this stage reports
    the two agreeing rather than asserting it twice in different words.
    """
    receipt = _read_json(path, what="serving receipt")

    recorded = str(receipt.get("slate_date") or "")

    if recorded != slate_date:
        raise PublicationStageFailed(
            f"{path} is the receipt for {recorded or 'no slate'} and this run "
            f"is publishing {slate_date}, so it does not provenance the rows "
            "that would be published"
        )

    outcome = str(receipt.get("outcome") or "")

    if outcome in NO_OP_OUTCOMES:
        return {
            "outcome": outcome,
            "served": False,
            "receipt_path": str(path),
            "slate_date": recorded,
        }

    if outcome != SERVED_OUTCOME:
        raise PublicationStageFailed(
            f"{path} records the serving outcome {outcome or 'nothing'}, which "
            "is neither a served slate nor one of the recorded no-ops "
            f"({', '.join(NO_OP_OUTCOMES)}), so whether there is anything to "
            "publish is unknown and nothing is published"
        )

    artifact = receipt.get("priced_market_artifact") or {}

    digest = str(artifact.get("sha256") or "")

    if not digest:
        raise PublicationStageFailed(
            f"{path} records a served slate without the digest of the priced "
            "artifact it served, so the publication would have nothing to be "
            "provenanced against"
        )

    expected = publisher.priced_markets_path(Path(data_root), slate_date)

    return {
        "outcome": outcome,
        "priced_market_artifact_sha256": digest,
        "priced_markets_path": str(expected),
        "production_code_sha": receipt.get("production_code_sha"),
        "receipt_path": str(path),
        "served": True,
        "slate_date": recorded,
    }


def read_distributions(*, slate_date: str, data_root: Path) -> dict[str, Any]:
    """The forecasts the serving step derived behind this slate's prices.

    Read, not derived. The serving step runs the distribution stage with the
    incumbent's own model directory, immediately after pricing, so that the
    forecasts a reader is shown come out of the artifacts that produced the
    prices beside them. This module computes nothing and loads no model -- a
    test reads this file and fails if the pricing path appears in it at all.

    What is checked here is that the document is this run's. It records the
    digest of the priced parquet it was derived from, and the caller compares
    that against the serving receipt's, so a document left over from a run
    against different prices is refused rather than published.
    """
    path = publisher.distributions_path(Path(data_root), slate_date)

    if not path.exists():
        raise PublicationStageFailed(
            f"{path} does not exist, so the serving step derived no forecasts "
            f"behind the prices for {slate_date} and nothing is published: a "
            "release of prices with no distributions behind them would render "
            "a page with empty panels"
        )

    document = publisher.read_distributions_document(path)

    return {
        "distribution_count": len(document.get("distributions") or {}),
        "path": str(path),
        "priced_sha256": str(document.get("priced_sha256") or ""),
        "sha256": publisher.sha256_file(path),
    }


# ----------------------------------------------------------------------
# stage one: the feed
# ----------------------------------------------------------------------


def verify_promoted_release(
    transport: Any, *, release_id: str, rows_sha256: str
) -> dict[str, Any]:
    """Read back what the destination is serving, and say what contract it is.

    The publisher already verified the bytes it wrote and fetched them over
    HTTPS before promoting. This asks the next question, which is the one the
    page depends on: after promotion, is the release the destination points at
    the release this run published, and is it the contract the renderer was
    written against.

    Read in the consumer's order -- pointer, manifest, rows -- because that is
    the order a mid-publication destination falls apart in, and checked for
    equality rather than for a floor: a renderer written against the wider
    shape reading a narrower release would draw a page with holes in it rather
    than refuse.
    """
    pointer = publisher.read_pointer(transport, publisher.CURRENT_POINTER)

    if not pointer:
        raise PublicationStageFailed(
            f"the destination has no {publisher.CURRENT_POINTER} after "
            "publication, so nothing is serving the release that was written"
        )

    for label, served, published in (
        ("release id", str(pointer.get("release_id") or ""), release_id),
        ("rows digest", str(pointer.get("rows_sha256") or ""), rows_sha256),
    ):
        if served != published:
            raise PublicationStageFailed(
                f"{publisher.CURRENT_POINTER} names the {label} "
                f"{served or 'nothing'} and this run published {published}, "
                "so the destination is not serving what was just published"
            )

    manifest_relative = str(pointer.get("manifest_relative") or "")

    if not manifest_relative or not transport.exists(manifest_relative):
        raise PublicationStageFailed(
            f"{publisher.CURRENT_POINTER} names the manifest "
            f"{manifest_relative or 'nowhere'}, which is not at the "
            "destination"
        )

    manifest = json.loads(transport.read_bytes(manifest_relative))

    rows_relative = str(manifest.get("rows_relative") or "")

    if not rows_relative or not transport.exists(rows_relative):
        raise PublicationStageFailed(
            f"{manifest_relative} names rows at "
            f"{rows_relative or 'no path'}, which are not at the destination"
        )

    rows_document = json.loads(transport.read_bytes(rows_relative))

    contracts = {
        "manifest": manifest.get("contract_version"),
        "rows": rows_document.get("contract_version"),
    }

    wrong = {
        where: version
        for where, version in contracts.items()
        if version != publisher.PUBLICATION_CONTRACT_VERSION
    }

    if wrong:
        raise PublicationStageFailed(
            "the promoted release declares publication contract "
            + ", ".join(
                f"{version!r} in its {where}" for where, version in sorted(wrong.items())
            )
            + f" and the page is built against "
            f"{publisher.PUBLICATION_CONTRACT_VERSION}, so no page is built "
            "from it and the previous release is not published in its place"
        )

    rows = rows_document.get("rows")

    if not isinstance(rows, list) or not rows:
        raise PublicationStageFailed(
            f"{rows_relative} carries no rows, so there is nothing for a "
            "reader to be shown"
        )

    return {
        "contract_version": publisher.PUBLICATION_CONTRACT_VERSION,
        "manifest_relative": manifest_relative,
        "promoted_at": str(pointer.get("promoted_at") or ""),
        "release_id": release_id,
        "row_count": len(rows),
        "rows_relative": rows_relative,
        "rows_sha256": rows_sha256,
    }


def publish_feed(
    *,
    mode: str,
    slate_date: str,
    data_root: Path,
    project_root: Path,
    serving_receipt: Path,
    destination_name: str | None,
    destination_root: Path | None,
) -> dict[str, Any]:
    """Stage one: publish the release, then prove what is being served.

    The publication itself is the existing entry point's own function, called
    with this run's own artifact and receipt and nothing else.
    """
    served = read_serving_receipt(
        serving_receipt, slate_date=slate_date, data_root=Path(data_root)
    )

    if not served["served"]:
        return {
            "generated_at": _utc_now(),
            "mode": mode,
            "outcome": OUTCOME_NOT_SERVED,
            "reason": (
                f"the incumbent recorded {served['outcome'] or 'no outcome'} "
                f"for {slate_date}, so there is no priced artifact to publish "
                "and nothing is published"
            ),
            "serving": served,
            "slate_date": slate_date,
            "stage": STAGE_FEED,
        }

    derived = read_distributions(slate_date=slate_date, data_root=Path(data_root))

    if derived["priced_sha256"] != served["priced_market_artifact_sha256"]:
        raise PublicationStageFailed(
            f"the distributions at {derived['path']} were derived from priced "
            f"markets digesting {derived['priced_sha256'] or 'nothing'} and "
            f"the serving receipt records "
            f"{served['priced_market_artifact_sha256']}, so they are not the "
            "forecasts behind the prices this run would publish"
        )

    name = (
        destination_name
        if destination_name is not None
        else (None if destination_root is not None else FEED_DESTINATIONS[mode])
    )

    publication = publisher.publish(
        mode=mode,
        slate_date=slate_date,
        data_root=Path(data_root),
        project_root=Path(project_root),
        destination_root=destination_root,
        receipt_path=Path(serving_receipt),
        destination_name=name,
    )

    release = publication["release"]

    destination = publisher.resolve_destination(
        mode,
        publisher.read_destinations(project_root=Path(project_root)),
        root=destination_root,
        name=name,
    )

    transport = publisher.build_transport(destination)

    try:
        promoted = verify_promoted_release(
            transport,
            release_id=str(release["release_id"]),
            rows_sha256=str(release["rows_sha256"]),
        )
    finally:
        transport.close()

    provenance = publication.get("provenance") or {}

    # The equality the whole provenance chain rests on, stated rather than
    # implied: the rows that were published are the artifact the serving
    # receipt is about. The publisher refuses when they differ; this records
    # that they did not.
    source_digest = str(provenance.get("priced_market_artifact_sha256") or "")

    if source_digest != served["priced_market_artifact_sha256"]:
        raise PublicationStageFailed(
            f"the publication provenances {source_digest or 'nothing'} and the "
            f"serving receipt records {served['priced_market_artifact_sha256']}, "
            "so the rows published are not the artifact that was served"
        )

    return {
        "attestation": publication.get("attestation"),
        "contract_version": promoted["contract_version"],
        "destination": publication["destination"],
        "distributions": derived,
        "generated_at": _utc_now(),
        "live_publishing": publication.get("live_publishing"),
        "mode": mode,
        "outcome": OUTCOME_PUBLISHED,
        "promoted": promoted,
        "promotion": publication.get("promotion"),
        "provenance": provenance,
        "reason": (
            f"{promoted['row_count']} rows published as "
            f"{promoted['release_id']} under publication contract "
            f"{promoted['contract_version']}"
        ),
        "release": release,
        "serving": served,
        "slate_date": slate_date,
        "stage": STAGE_FEED,
        "verification": publication.get("verification"),
    }


# ----------------------------------------------------------------------
# stage two: the page
# ----------------------------------------------------------------------


def pin_from_feed_receipt(
    path: Path, *, slate_date: str, mode: str
) -> page.ExpectedRelease | None:
    """The release the page must be built from, read off stage one's receipt.

    ``None`` means stage one published nothing because nothing was served, and
    stage two is a no-op for the same reason. Every other shape of receipt is
    either the publication this page is for or a failure, and a failure is not
    a licence to build a page from the release that was there before.
    """
    receipt = _read_json(path, what="feed publication receipt")

    outcome = str(receipt.get("outcome") or "")

    if outcome == OUTCOME_NOT_SERVED:
        return None

    if outcome != OUTCOME_PUBLISHED:
        raise PublicationStageFailed(
            f"{path} records the feed publication outcome "
            f"{outcome or 'nothing'}, so there is no verified release to build "
            "a page from and none is built"
        )

    for label, recorded, expected in (
        ("slate date", str(receipt.get("slate_date") or ""), slate_date),
        ("mode", str(receipt.get("mode") or ""), mode),
    ):
        if recorded != expected:
            raise PublicationStageFailed(
                f"{path} records the {label} {recorded or 'nothing'} and this "
                f"page stage is {expected}, so the receipt is not this run's "
                "feed publication"
            )

    contract = receipt.get("contract_version")

    if contract != publisher.PUBLICATION_CONTRACT_VERSION:
        raise PublicationStageFailed(
            f"{path} records publication contract {contract!r} and the page is "
            f"built against {publisher.PUBLICATION_CONTRACT_VERSION}, so no "
            "page is built from it"
        )

    return page.ExpectedRelease.from_publication_receipt(receipt)


def publish_the_page(
    *,
    mode: str,
    slate_date: str,
    project_root: Path,
    feed_receipt: Path,
    source_destination_name: str | None,
    source_destination_root: Path | None,
    page_destination_name: str | None,
    page_destination_root: Path | None,
    prefix: str | None,
) -> dict[str, Any]:
    """Stages two and three: build the page from that release, and publish it.

    One invocation because the page builder renders, writes, reads back
    through the transport and fetches over HTTPS as one operation, and a
    failure anywhere in it publishes nothing.
    """
    expected = pin_from_feed_receipt(
        Path(feed_receipt), slate_date=slate_date, mode=mode
    )

    if expected is None:
        return {
            "generated_at": _utc_now(),
            "mode": mode,
            "outcome": OUTCOME_NOT_SERVED,
            "reason": (
                "the feed stage published nothing because the incumbent "
                f"served no slate on {slate_date}, so there is no release to "
                "build a page from"
            ),
            "slate_date": slate_date,
            "stage": STAGE_PAGE,
        }

    root = Path(project_root)

    defaults = page.MODE_DESTINATIONS[mode]

    switch = page.assert_mode_is_permitted(mode, project_root=root)

    source = page._transport_for(
        mode=mode,
        name=(
            source_destination_name
            if source_destination_name is not None
            else (None if source_destination_root is not None else defaults["source"])
        ),
        root=str(source_destination_root)
        if source_destination_root is not None
        else None,
        project_root=root,
    )

    destination = page._transport_for(
        mode=mode,
        name=(
            page_destination_name
            if page_destination_name is not None
            else (None if page_destination_root is not None else defaults["page"])
        ),
        root=str(page_destination_root)
        if page_destination_root is not None
        else None,
        project_root=root,
    )

    try:
        published = page.publish_page(
            source_transport=source,
            page_transport=destination,
            registry=routing.AffiliateRegistry.load(project_root=root),
            fallback=fallback_destinations.FallbackDestinations.load(
                project_root=root
            ),
            presentation=page.CONSUMER_PREVIEW,
            prefix=prefix if prefix is not None else defaults["prefix"],
            expected=expected,
        )
    finally:
        source.close()

        destination.close()

    source_release = published["source_release"]

    # The pin is checked inside the build, at the pointer and at the digest.
    # Checked again on the way out because this receipt is what the report is
    # written from, and a receipt that recorded an equality it had not
    # confirmed would be worse than no receipt.
    for label, built, pinned in (
        ("release id", source_release["release_id"], expected.release_id),
        ("rows digest", source_release["rows_sha256"], expected.rows_sha256),
    ):
        if built != pinned:
            raise PublicationStageFailed(
                f"the page was built from the {label} {built} and pinned to "
                f"{pinned}, so it is not the page for the release this run "
                "published"
            )

    manifest = published["manifest"]

    rendered = manifest.get("rendered") or {}

    return {
        "generated_at": _utc_now(),
        "manifest": manifest,
        "mode": mode,
        "outcome": OUTCOME_PUBLISHED,
        "page_sha256": manifest.get("page_sha256"),
        "public_readback": published["public_readback"],
        "reason": (
            f"the page for {source_release['release_id']} was published to "
            + ", ".join(published["written"])
        ),
        # The page's own count, not the release's. They differ legitimately --
        # several books quoting one wager are one row -- and the deliverable is
        # the number of predictions a reader is shown.
        "row_count": rendered.get("prediction_count"),
        "published_row_count": rendered.get("published_row_count"),
        "slate_date": slate_date,
        "source_release": source_release,
        "stage": STAGE_PAGE,
        "switch": switch,
        "written": published["written"],
    }


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def render(receipt: dict[str, Any]) -> str:
    stage = receipt.get("stage")

    heading = (
        "WizardOfOdds feed publication"
        if stage == STAGE_FEED
        else "WizardOfOdds page publication"
    )

    rows: list[tuple[str, Any]] = [
        ("outcome", receipt.get("outcome")),
        ("mode", receipt.get("mode")),
        ("slate date", receipt.get("slate_date")),
    ]

    if stage == STAGE_FEED and receipt.get("outcome") == OUTCOME_PUBLISHED:
        promoted = receipt.get("promoted") or {}

        rows += [
            ("release", promoted.get("release_id")),
            ("publication contract", promoted.get("contract_version")),
            ("rows published", promoted.get("row_count")),
            ("rows digest", str(promoted.get("rows_sha256") or "")[:16]),
        ]

    if stage == STAGE_PAGE and receipt.get("outcome") == OUTCOME_PUBLISHED:
        source_release = receipt.get("source_release") or {}

        rows += [
            ("built from release", source_release.get("release_id")),
            ("rows digest", str(source_release.get("rows_sha256") or "")[:16]),
            ("predictions on the page", receipt.get("row_count")),
            ("page digest", str(receipt.get("page_sha256") or "")[:16]),
            ("published", ", ".join(receipt.get("written") or [])),
        ]

        for target, read in sorted((receipt.get("public_readback") or {}).items()):
            rows.append((f"readback {target}", read.get("url")))

    lines = [f"## {heading}", "", "| field | value |", "| --- | --- |"]

    lines += [f"| {label} | {value} |" for label, value in rows]

    lines += ["", str(receipt.get("reason") or "")]

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_wizardofodds_publication",
        description=(
            "Publish the incumbent's served slate to WizardOfOdds, then the "
            "page built from that exact release. Computes nothing and "
            "publishes nothing a run did not itself produce."
        ),
    )

    parser.add_argument("--stage", choices=STAGES, required=True)

    parser.add_argument(
        "--mode",
        choices=sorted(FEED_DESTINATIONS),
        default=publisher.MODE_CANARY,
        help="canary publishes the non-live feed and the page preview. live "
        "publishes the served roots and is refused by the committed switch.",
    )

    parser.add_argument("--slate-date", required=True)

    parser.add_argument("--data-root", type=Path, default=None)

    parser.add_argument(
        "--serving-receipt",
        type=Path,
        default=None,
        help="the receipt ops/run_incumbent_production_serving.py wrote for "
        "this run. Required by the feed stage.",
    )

    parser.add_argument(
        "--feed-receipt",
        type=Path,
        default=None,
        help="the receipt the feed stage wrote. Required by the page stage, "
        "which is pinned to the release it names.",
    )

    parser.add_argument("--feed-destination-name", default=None)
    parser.add_argument("--feed-destination-root", type=Path, default=None)
    parser.add_argument("--page-source-destination-name", default=None)
    parser.add_argument("--page-source-destination-root", type=Path, default=None)
    parser.add_argument("--page-destination-name", default=None)
    parser.add_argument("--page-destination-root", type=Path, default=None)
    parser.add_argument("--page-prefix", default=None)

    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage == STAGE_FEED:
        if args.serving_receipt is None or args.data_root is None:
            raise PublicationStageFailed(
                "the feed stage publishes this run's own priced artifact, so "
                "--serving-receipt and --data-root are both required"
            )

        return publish_feed(
            mode=args.mode,
            slate_date=args.slate_date,
            data_root=args.data_root,
            project_root=args.project_root,
            serving_receipt=args.serving_receipt,
            destination_name=args.feed_destination_name,
            destination_root=args.feed_destination_root,
        )

    if args.feed_receipt is None:
        raise PublicationStageFailed(
            "the page stage is built from the release the feed stage "
            "published, so --feed-receipt is required: without it there is no "
            "release to pin to and the page would be built from whatever the "
            "destination happened to be serving"
        )

    return publish_the_page(
        mode=args.mode,
        slate_date=args.slate_date,
        project_root=args.project_root,
        feed_receipt=args.feed_receipt,
        source_destination_name=args.page_source_destination_name,
        source_destination_root=args.page_source_destination_root,
        page_destination_name=args.page_destination_name,
        page_destination_root=args.page_destination_root,
        prefix=args.page_prefix,
    )


def _write(path: Path | None, text: str, *, append: bool) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except (
        PublicationStageFailed,
        publisher.PublishingRefused,
        page.PageRefused,
        routing.RoutingRefused,
        fallback_destinations.DestinationRefused,
        OSError,
        ValueError,
    ) as error:
        failure = {
            "error": type(error).__name__,
            "message": str(error),
            "mode": args.mode,
            "outcome": "PUBLICATION_REFUSED",
            "slate_date": args.slate_date,
            "stage": args.stage,
        }

        _write(
            args.receipt_path,
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            append=False,
        )

        _write(
            args.summary_path,
            f"## WizardOfOdds {args.stage} publication\n\nREFUSED: {error}\n",
            append=True,
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_REFUSED

    _write(
        args.receipt_path,
        json.dumps(receipt, default=str, indent=2, sort_keys=True) + "\n",
        append=False,
    )

    _write(args.summary_path, render(receipt), append=True)

    print(json.dumps(receipt, default=str, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
