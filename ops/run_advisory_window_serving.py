#!/usr/bin/env python3
"""Serve one slate from an early capture window, for advisory reading only.

WHAT WAS MISSING
----------------

The production pricing path consumes exactly one capture window, T-20, and the
release built from it reaches a reader roughly twenty minutes before tip. The
capture schedule has always defined T-180, T-90 and T-45 as well, the two
serving scripts have always taken ``--snapshot-offset-minutes``, and
``ProspectiveSnapshotClient`` has always selected by window id -- so the
machinery to price an earlier snapshot existed end to end and nothing invoked
it. There was no entry point that would serve a window other than T-20, and
``ops/run_incumbent_production_serving.py`` hard-codes T-20 by never passing
the offset through to the scripts that accept it.

This is that entry point, and it is deliberately thin. It resolves the
incumbent, reads readiness and builds its receipt by calling
``run_incumbent_production_serving``'s own functions, by name. Not one of them
is reimplemented here, so the model that prices an advisory window is the
model that prices T-20, resolved by the same authority, verified against the
same frozen bundle, and recorded in a receipt of the same shape.

The one thing this module adds is the two arguments the orchestrator's own
``serve`` already accepts and never passes: ``--snapshot-offset-minutes N`` to
the prediction script and to the pricing script. That is the whole of the
difference. No probability, price, edge, EV, quantile, line, distribution,
calibration, dependence coefficient, simulation count, Gate 3 route, minutes
model or fitted artifact is touched, computed or substituted anywhere in it.

WHAT IT IS NOT
--------------

It is not a second T-20. ``--snapshot-offset-minutes 20`` is refused: the
production window has one producer, one receipt location and one release
stream, and an entry point that could serve it under a different name is an
entry point that could put advisory bytes into the T-20 record.

It is not a durable production serving run either. The orchestrator's receipt
is copied into ``processed/incumbent_serving/`` by the lifecycle, and that
file is what the grader provenances a slate's grades against and what the
T-20 controller reads to prove the morning established an incumbent. An
advisory receipt must never be mistaken for either, and two independent
things keep them apart: this module refuses a receipt path that resolves
inside that tree, naming ``processed/advisory_window_serving/`` instead, and
every receipt it writes carries an ``advisory`` marker that the T-20
controller's readiness check refuses. Either alone would be enough, which is
why there are two.

It cannot publish, promote, fit, refit, recalibrate, grade, shadow or
activate. It writes the two serving artifacts the frozen scripts write and
one receipt, and the destination an advisory release goes to is the caller's
decision, taken against the confined advisory canary destinations.

THE SHARED PRICED ARTIFACT
--------------------------

``15_price_markets.py`` writes ``processed/priced_markets/<slate>.parquet``,
one path per slate, and so an advisory run rewrites the same file a T-20 run
writes. That is not a new property -- the T-20 controller already rewrites it
once per due tip group -- and three facts keep it from contaminating anything:

* every priced row carries ``gate3_capture_window_id`` and
  ``gate3_capture_offset_minutes``, so the file states which window it was
  priced from and no reader has to infer it;
* the publisher re-hashes the parquet against the serving receipt it was
  handed and refuses when they differ, so an advisory rewrite cannot be
  published as a T-20 release and a T-20 rewrite cannot be published as an
  advisory one;
* ``ops/grade_incumbent_production_slate.py`` performs the same re-hash
  before it grades, so advisory bytes cannot be graded as the slate the
  incumbent served.

The caller is responsible for ordering: serving and publishing one window is
a single uninterrupted sequence, and the T-20 controller performs it as one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

if str(PROJECT_ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ops"))

# Imported as a module and used by name throughout. Every authority this
# entry point rests on -- which incumbent is serving, whether the slate is
# ready, what the receipt has to carry, which outcomes count as served -- is
# read out of the production orchestrator rather than restated here, so there
# is no second answer to any of those questions.
import run_incumbent_production_serving as serving  # noqa: E402

from nba_prop_quant.adaptive_fit_registry import (  # noqa: E402
    FitRegistry,
    RegistryError,
    resolve_registry_root,
)
from nba_prop_quant.snapshot_schedule import (  # noqa: E402
    CAPTURE_OFFSETS_MINUTES,
)

#: The window production consumes. Refused here, in both directions: this
#: entry point may not serve it, and the receipt it writes may not claim it.
PRODUCTION_OFFSET_MINUTES = 20

#: The early windows this entry point is for, which are a subset of the
#: capture schedule's own and add nothing to it. Stated here as well as in the
#: controller because this script is independently invocable, and an offset
#: filter that lived only in the caller would be an offset filter a direct
#: invocation could skip. A test pins the two tuples equal.
ADVISORY_OFFSETS_MINUTES: tuple[int, ...] = (180, 90, 45)

#: Where an advisory receipt belongs if it is kept in the data root at all: a
#: tree of its own, read by nothing, and specifically not
#: :data:`SERVING_RECEIPTS_RELATIVE` below. Named in the refusal that rejects
#: the production receipt tree, so the refusal says where to write instead of
#: only where not to.
ADVISORY_RECEIPTS_RELATIVE = Path("processed") / "advisory_window_serving"

#: Where the production serving receipt for a slate is durably kept: written
#: by the lifecycle, read by the grader and by the T-20 controller's readiness
#: check. Named here so this module can refuse to write into it rather than
#: merely not defaulting there.
SERVING_RECEIPTS_RELATIVE = Path("processed") / "incumbent_serving"

#: The marker field. Every receipt this module writes carries it, including
#: the failure receipt, and nothing else in the repository writes it -- so
#: "was this an advisory run" is answerable from the receipt alone rather than
#: from the directory it happens to sit in.
ADVISORY_MARKER_FIELD = "advisory"

#: What a failed advisory run records. Deliberately not one of the serving
#: module's outcomes and deliberately not one of the publication
#: orchestrator's no-ops: a failed advisory serve must be refused by the
#: publication stage rather than read as a day with nothing to publish.
OUTCOME_FAILED = "ADVISORY_SERVING_FAILED"

EXIT_OK = 0
EXIT_FAILED = 1


class AdvisoryServingRefused(RuntimeError):
    """A fail-closed refusal: nothing was served and nothing was written."""


def assert_offset_is_advisory(offset_minutes: int) -> int:
    """The offset this run may serve, or the refusal that names why not.

    Three separate refusals rather than one, because the three mistakes have
    nothing to do with each other. An offset the capture schedule does not
    define could never have a capture to select. The production offset is the
    T-20 window, which has its own producer and its own record and is not
    served from here under any circumstances. And an offset that is scheduled
    and is not T-20 but is also not one of the advisory windows -- T-1440,
    T-480, T-5 -- is a research window with no advisory destination to publish
    to, so serving it would produce a priced slate nothing is configured to
    carry.
    """
    offset = int(offset_minutes)

    if offset not in CAPTURE_OFFSETS_MINUTES:
        raise AdvisoryServingRefused(
            f"--snapshot-offset-minutes {offset} is not a window the capture "
            f"schedule defines {list(CAPTURE_OFFSETS_MINUTES)}, so no capture "
            "of it could ever be selected"
        )

    if offset == PRODUCTION_OFFSET_MINUTES:
        raise AdvisoryServingRefused(
            f"--snapshot-offset-minutes {offset} is the production T-"
            f"{PRODUCTION_OFFSET_MINUTES} window, which "
            f"{serving.__name__}.py serves and this entry point does not. "
            "The T-20 window has one producer, one durable receipt and one "
            "release stream, and serving it from here would put advisory "
            "bytes into that record under its name"
        )

    if offset not in ADVISORY_OFFSETS_MINUTES:
        raise AdvisoryServingRefused(
            f"--snapshot-offset-minutes {offset} is a research window rather "
            f"than one of the advisory windows {list(ADVISORY_OFFSETS_MINUTES)}, "
            "and no advisory destination is configured for it, so the slate it "
            "priced would have nowhere to go"
        )

    return offset


def assert_window_ids_are_this_offset(
    window_ids: list[str], *, offset_minutes: int
) -> list[str]:
    """Every window this run claims is a window at its own offset.

    The capture window id is the only label an advisory run carries into its
    receipt, so it is the one place a T-20 claim could be forged. Checked
    rather than trusted: a receipt naming ``:T-20m`` would describe this run
    as the production one to every reader of it, and the suffix is cheap to
    check and impossible to argue with.
    """
    suffix = f":T-{int(offset_minutes)}m"

    production_suffix = f":T-{PRODUCTION_OFFSET_MINUTES}m"

    claimed_t20 = sorted(
        str(window_id)
        for window_id in window_ids
        if str(window_id).endswith(production_suffix)
    )

    if claimed_t20:
        raise AdvisoryServingRefused(
            "an advisory run would record the production window(s) "
            + ", ".join(claimed_t20)
            + f" in its receipt, and a T-{PRODUCTION_OFFSET_MINUTES} window id "
            "is the name of the production record. Nothing here may write one"
        )

    wrong = sorted(
        str(window_id)
        for window_id in window_ids
        if not str(window_id).endswith(suffix)
    )

    if wrong:
        raise AdvisoryServingRefused(
            f"this run serves T-{offset_minutes} and was given the window(s) "
            + ", ".join(wrong)
            + f", which are not {suffix} windows, so the receipt would name "
            "captures it did not price from"
        )

    return sorted(str(window_id) for window_id in window_ids)


def assert_receipt_path_is_not_the_production_record(
    receipt_path: Path, *, data_root: Path
) -> Path:
    """An advisory receipt may not be written where the production one lives.

    ``processed/incumbent_serving/<slate>.json`` is read by the grader to
    provenance a slate's grades and by the T-20 controller to prove the
    morning lifecycle established a verified incumbent. Both would accept an
    advisory receipt on its shape alone -- it is the same shape, by design,
    because it is built by the same function -- so the separation has to be
    structural rather than conventional.
    """
    production = (Path(data_root) / SERVING_RECEIPTS_RELATIVE).expanduser()

    resolved = Path(receipt_path).expanduser().resolve()

    try:
        production_resolved = production.resolve()
    except OSError:  # pragma: no cover - resolve of a missing tree
        production_resolved = production

    if resolved == production_resolved or production_resolved in resolved.parents:
        raise AdvisoryServingRefused(
            f"--receipt-path {resolved} is inside {production_resolved}, "
            "which holds the durable production serving receipt the grader "
            "and the T-20 controller read. An advisory receipt written there "
            f"would be read as the production one; write it under "
            f"{ADVISORY_RECEIPTS_RELATIVE} instead"
        )

    return resolved


def build_advisory_receipt(
    *,
    receipt: dict[str, Any],
    receipt_path: Path,
    offset_minutes: int,
    window_ids: list[str],
) -> dict[str, Any]:
    """The orchestrator's own receipt, labelled with the window it priced.

    Strictly additive, and it has to be. The publication orchestrator reads
    this receipt, requires the serving outcome to be exactly ``SERVED`` and
    the provenance fields to be exactly the ones ``build_receipt`` emits, and
    it is one of the files this lineage may not edit -- so an advisory receipt
    that renamed its outcome or dropped a field would simply be unpublishable.
    Every value here is therefore the value the production path would have
    recorded for the same slate, which is the correct answer anyway: the
    incumbent, the frozen bundle, the production SHA and the input state are
    the same; the capture window is the only thing that differs.

    So the window is what is added, under names nothing else in the repository
    writes. Those three fields are what make an advisory receipt recognisable
    as one, and the T-20 controller refuses a readiness receipt carrying the
    marker for exactly that reason.
    """
    outcome = str(receipt.get("outcome") or "")

    known = {
        serving.OUTCOME_SERVED,
        serving.OUTCOME_NO_GAMES_ON_SLATE,
        serving.OUTCOME_NO_PRODUCTION_SLATE,
    }

    if outcome not in known:
        raise AdvisoryServingRefused(
            f"{serving.__name__} reported the outcome {outcome or 'nothing'}, "
            f"which is not one of {sorted(known)}, so no advisory receipt is "
            "written for it"
        )

    return {
        **receipt,
        ADVISORY_MARKER_FIELD: True,
        "advisory_capture_offset_minutes": int(offset_minutes),
        "advisory_capture_window_ids": list(window_ids),
        "advisory_receipt_path": str(receipt_path),
    }


def render(receipt: dict[str, Any]) -> str:
    offset = receipt.get("advisory_capture_offset_minutes")

    priced = receipt.get("priced_market_artifact") or {}

    rows: list[tuple[str, Any]] = [
        ("outcome", receipt.get("outcome")),
        ("capture window", f"T-{offset}" if offset is not None else "none"),
        ("slate date", receipt.get("slate_date")),
        (
            "windows priced",
            ", ".join(receipt.get("advisory_capture_window_ids") or [])
            or "none",
        ),
        ("incumbent authority", receipt.get("incumbent_authority")),
        ("incumbent version", receipt.get("incumbent_version")),
        ("priced artifact digest", str(priced.get("sha256") or "")[:16]),
    ]

    lines = [
        f"## NBA advisory window serving (T-{offset})",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    lines += [f"| {label} | {value} |" for label, value in rows]

    lines += ["", str(receipt.get("reason") or "")]

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_advisory_window_serving",
        description=(
            "Price one slate from an early capture window for advisory "
            "reading. Calls the production serving orchestrator's own "
            "functions with the snapshot offset those scripts already "
            "accept; refuses the T-20 window, publishes nothing, promotes "
            "nothing and fits nothing."
        ),
    )

    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", type=Path, required=True)

    parser.add_argument(
        "--snapshot-offset-minutes",
        type=int,
        required=True,
        help="the advisory capture window to price from. Required and not "
        "defaulted: a default would make the window a property of this "
        "file rather than of the caller's decision, and the one value it "
        "could plausibly default to is the T-20 window this entry point "
        "refuses.",
    )

    parser.add_argument(
        "--capture-window-id",
        action="append",
        default=None,
        dest="capture_window_ids",
        help="a window id this run is pricing from, repeatable. Recorded in "
        "the receipt and checked against the offset, so a receipt cannot "
        "name a window at another offset and specifically cannot name a "
        "T-20 one.",
    )

    parser.add_argument("--registry-root", type=Path, default=None)

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="the verified frozen runtime bundle's model directory, as "
        "ops/run_incumbent_production_serving.py requires it and for the "
        "same reason: the model binaries are ignored in Git, so the "
        "checkout's models/ directory could only resolve to an incomplete "
        "tree.",
    )

    parser.add_argument(
        "--frozen-bundle-root",
        type=Path,
        required=True,
        help="the verified frozen runtime bundle's root, which the frozen "
        "manifest's records are resolved against. Required for the reason "
        "the production orchestrator requires it.",
    )

    parser.add_argument(
        "--refresh-status",
        type=Path,
        default=None,
        help="the JSON ops/classify_refresh_outcome.py emitted. Readiness is "
        "read from it by the production orchestrator's own function.",
    )

    parser.add_argument("--production-sha", default=None)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    parser.add_argument(
        "--receipt-path",
        type=Path,
        required=True,
        help="where this run's receipt is written. Required and not "
        "defaulted, because the one refusal that keeps an advisory receipt "
        "out of the production record is about this path, and a default "
        "would make it a property of this file that a caller could not be "
        "held to.",
    )

    parser.add_argument("--summary-path", type=Path, default=None)

    return parser.parse_args(argv)


def _write(path: Path | None, text: str, *, append: bool) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    offset = assert_offset_is_advisory(args.snapshot_offset_minutes)

    window_ids = assert_window_ids_are_this_offset(
        list(args.capture_window_ids or []), offset_minutes=offset
    )

    project_root = Path(args.project_root).resolve()

    data_root = Path(args.data_root).resolve()

    receipt_path = assert_receipt_path_is_not_the_production_record(
        args.receipt_path, data_root=data_root
    )

    model_dir = Path(args.model_dir).resolve()

    bundle_root = Path(args.frozen_bundle_root).resolve()

    readiness = serving.readiness_for_this_slate(
        serving.slate_readiness(args.refresh_status),
        refresh_status=args.refresh_status,
        slate_date=args.slate_date,
    )

    registry = FitRegistry(
        root=resolve_registry_root(
            args.registry_root, project_root=project_root
        ),
        project_root=project_root,
    )

    incumbent = serving.resolve_incumbent(
        registry=registry,
        model_dir=model_dir,
        bundle_root=bundle_root,
    )

    if not readiness["ready"]:
        served = {
            "outcome": serving.OUTCOME_NO_PRODUCTION_SLATE,
            "prediction_artifact": None,
            "priced_market_artifact": None,
            "reason": readiness["reason"],
        }

    else:
        # The one and only difference from the production path: the offset
        # the two frozen scripts already take is passed to both of them.
        # Identical arguments for prediction and pricing, because the two
        # must read the same capture or the prices would be priced against
        # projections from another window.
        offset_arguments = ["--snapshot-offset-minutes", str(offset)]

        served = serving.serve(
            slate_date=args.slate_date,
            incumbent=incumbent,
            project_root=project_root,
            data_root=data_root,
            predict_arguments=offset_arguments,
            price_arguments=list(offset_arguments),
        )

    receipt = build_advisory_receipt(
        receipt_path=receipt_path,
        receipt=serving.build_receipt(
            slate_date=args.slate_date,
            production_sha=args.production_sha,
            incumbent=incumbent,
            readiness=readiness,
            fingerprint=serving.input_state_fingerprint(data_root),
            served=served,
        ),
        offset_minutes=offset,
        window_ids=window_ids,
    )

    # The publisher requires the same provenance of an advisory release as of
    # a production one, so the same completeness check is applied, by the
    # orchestrator's own function.
    incomplete = serving.missing_provenance(receipt)

    if incomplete:
        raise AdvisoryServingRefused(
            "the advisory serving receipt is missing required provenance: "
            + ", ".join(incomplete)
        )

    return receipt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except (
        AdvisoryServingRefused,
        serving.ServingRefused,
        serving.ServingFailed,
        RegistryError,
        OSError,
    ) as error:
        failure = {
            ADVISORY_MARKER_FIELD: True,
            "advisory_capture_offset_minutes": args.snapshot_offset_minutes,
            "error": type(error).__name__,
            "message": str(error),
            "model_authority": serving.MODEL_AUTHORITY,
            "outcome": OUTCOME_FAILED,
            "slate_date": args.slate_date,
        }

        _write(
            args.receipt_path,
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            append=False,
        )

        _write(
            args.summary_path,
            "## NBA advisory window serving\n\n"
            f"FAILED: {error}\n",
            append=True,
        )

        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)

        return EXIT_FAILED

    _write(
        args.receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        append=False,
    )

    _write(args.summary_path, render(receipt), append=True)

    print(json.dumps(receipt, indent=2, sort_keys=True))

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
