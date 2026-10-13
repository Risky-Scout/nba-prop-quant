"""A deterministic slate built to prove the affiliate routing rule in public.

ROUTING CANARY / TEST DATA. Not a model slate and not a betting opinion: the
probabilities here were chosen so that four cards exist and so that the four
routing outcomes are distinguishable, and the page built from them is published
to a ``routing-proof`` directory of its own, beside the page canary and never
over it.

It exists because the rule cannot be proven against the slate the feed canary
currently holds. That slate's only sportsbook is a single non-affiliate
vendor, so every card on it is a fallback and a reader of the published page
cannot see that an affiliate is preferred, that the preference survives a
better price elsewhere, that a tie is settled by the operator's order, or that
a wager nobody offers produces no button. This slate contains exactly those
four situations and nothing else.

WHAT IS FIXTURE ARITHMETIC AND WHAT IS NOT
------------------------------------------

The schema is the real published schema, and the columns are filled the way
the runtime fills them: the market's devigged probabilities come from the two
posted prices, the edge is the model probability minus the market's, and the
expected value is the usual one at the posted price. That arithmetic is here
so the numbers on the proof page are consistent with each other rather than
decorative. It is fixture construction; no prediction or pricing code is
touched, called or changed by this module, and nothing it produces is ever
published to the feed canary or to the live page.

CASE C AND ITS MISSING PRICE
----------------------------

Case C is a prediction no sportsbook prices on the side the model wants: the
one book in it quotes the over only. There is therefore no devigged pair to
compute, so the market probabilities are the one-sided implied pair, and there
is no expected value at a price that does not exist, so it is published as
null. The page renders those as absent rather than as zero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for _relative in ("ops", "src"):
    if str(PROJECT_ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / _relative))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_affiliate_routing as routing  # noqa: E402
import wizardofodds_player_props_page as page  # noqa: E402

#: Where the proof page is written, relative to the page canary's own root.
#: A child directory, so publishing it cannot touch the page canary's own
#: ``index.html`` and cannot reach the live index above it either.
PROOF_PREFIX = "routing-proof"

SLATE_DATE = "2026-10-20"

HISTORY_DATE = "2026-10-18"

GAME_ID = 7701

#: Case -> the player, prop, line and side the case is about. The side is the
#: model's, and every quote in a case is a price for that same bet.
CASES: Mapping[str, dict[str, Any]] = {
    "A": {
        "player_id": 3301,
        "player_name": "Routing Proof A",
        "prop_type": "points",
        "line_value": 25.5,
        "side": routing.SIDE_UNDER,
        "model_probability": 0.565,
        "proves": (
            "an affiliate book is selected even though a non-affiliate book "
            "posts a better price, and the affiliate selected is the best "
            "priced of the four"
        ),
    },
    "B": {
        "player_id": 3302,
        "player_name": "Routing Proof B",
        "prop_type": "rebounds",
        "line_value": 8.5,
        "side": routing.SIDE_UNDER,
        "model_probability": 0.545,
        "proves": (
            "no affiliate book offers the wager, so the best price across "
            "every other sportsbook in the data is selected"
        ),
    },
    "C": {
        "player_id": 3303,
        "player_name": "Routing Proof C",
        "prop_type": "threes",
        "line_value": 2.5,
        "side": routing.SIDE_UNDER,
        "model_probability": 0.58,
        "proves": "no sportsbook prices this side, so there is no button",
    },
    "D": {
        "player_id": 3304,
        "player_name": "Routing Proof D",
        "prop_type": "assists",
        "line_value": 6.5,
        "side": routing.SIDE_UNDER,
        "model_probability": 0.60,
        "proves": (
            "two affiliate books post exactly the same best price, so the "
            "operator's tie order decides, and BetAnything is the name "
            "betanysports is shown under"
        ),
    },
}

#: Case -> (vendor, price on the model's side, price on the other side).
#: ``None`` is a book that does not price that side at all.
QUOTES: Mapping[str, tuple[tuple[str, Any, Any], ...]] = {
    "A": (
        ("betonlineag", -108, -112),
        ("bovada", -112, -108),
        ("betanysports", -115, -105),
        ("betus", -120, 100),
        ("fanduel", 105, -125),
        ("draftkings", -102, -118),
    ),
    "B": (
        ("fanduel", -120, 100),
        ("draftkings", 101, -121),
        ("fanatics", -105, -115),
        ("betmgm", -110, -110),
    ),
    "C": (("fanduel", None, -110),),
    "D": (
        ("betanysports", -110, -110),
        ("betus", -110, -110),
        ("fanduel", -130, 110),
        ("draftkings", -125, 105),
    ),
}


# ----------------------------------------------------------------------
# fixture arithmetic
# ----------------------------------------------------------------------


def decimal_odds(american: Any) -> float | None:
    multiple = routing.payout_multiple(american)

    return None if multiple is None else 1.0 + multiple


def implied(american: Any) -> float | None:
    decimal = decimal_odds(american)

    return None if decimal is None else 1.0 / decimal


def fair_american(probability: float) -> int:
    """The price at which a bet of this probability breaks even."""
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError(probability)

    if probability > 0.5:
        return -int(round(100.0 * probability / (1.0 - probability)))

    return int(round(100.0 * (1.0 - probability) / probability))


def market_pair(side_price: Any, other_price: Any) -> tuple[float, float, float]:
    """Devigged probabilities for (side, other) and the hold they carried.

    Both prices present is the normal two-way market and this is the standard
    two-way devig. One price missing is a one-sided market: there is no vig to
    remove, so the pair is the implied probability and its complement, and the
    hold is reported as zero because none was observed.
    """
    side_implied = implied(side_price)

    other_implied = implied(other_price)

    if side_implied is None and other_implied is None:
        raise ValueError("a quote with no price on either side is not a quote")

    if side_implied is None:
        return 1.0 - other_implied, other_implied, 0.0

    if other_implied is None:
        return side_implied, 1.0 - side_implied, 0.0

    total = side_implied + other_implied

    return side_implied / total, other_implied / total, (total - 1.0) * 100.0


def expected_value(probability: float, price: Any) -> float | None:
    """One unit staked at ``price``, at ``probability``. Null without a price."""
    multiple = routing.payout_multiple(price)

    return None if multiple is None else probability * multiple - (1.0 - probability)


# ----------------------------------------------------------------------
# the slate
# ----------------------------------------------------------------------


def approved_switch() -> Any:
    return publisher.read_publishing_switch(
        publisher.MODE_CANARY, project_root=PROJECT_ROOT
    )


def priced_frame(*, switch: Any | None = None) -> pd.DataFrame:
    """One row per quote, in the published schema, for all four cases."""
    approved = switch if switch is not None else approved_switch()

    rows: list[dict[str, Any]] = []

    quote_id = 7_700_000

    for name, case in CASES.items():
        side = case["side"]

        other = (
            routing.SIDE_OVER if side == routing.SIDE_UNDER else routing.SIDE_UNDER
        )

        probability = float(case["model_probability"])

        for vendor, side_price, other_price in QUOTES[name]:
            quote_id += 1

            side_devig, other_devig, hold = market_pair(side_price, other_price)

            prices = {side: side_price, other: other_price}

            devig = {side: side_devig, other: other_devig}

            probabilities = {side: probability, other: 1.0 - probability}

            row: dict[str, Any] = {
                "id": quote_id,
                "vendor": vendor,
                "game_id": GAME_ID,
                "date": pd.Timestamp(SLATE_DATE),
                "player_id": case["player_id"],
                "player_name": case["player_name"],
                "team_id": 71,
                "opponent_id": 72,
                "is_home": 1,
                "prop_type": case["prop_type"],
                "line_value": case["line_value"],
                "market_type": "over_under",
                "over_odds": prices[routing.SIDE_OVER],
                "under_odds": prices[routing.SIDE_UNDER],
                "quote_eligible": True,
                "quote_filter_reason": "",
                "availability_out": 0,
                "gate3_role_ready": 1,
                "history_latest_date": pd.Timestamp(HISTORY_DATE),
                "advanced_latest_date": pd.Timestamp(HISTORY_DATE),
                "calibrated_q_over_nonpush": probabilities[routing.SIDE_OVER],
                "calibrated_q_under_nonpush": probabilities[routing.SIDE_UNDER],
                "calibrated_p_push": 0.0,
                "calibration_method": "routing_proof_fixture",
                "market_devig_q_over": devig[routing.SIDE_OVER],
                "market_devig_q_under": devig[routing.SIDE_UNDER],
                "market_hold_pct": hold,
                "calibrated_fair_over_american": fair_american(
                    probabilities[routing.SIDE_OVER]
                ),
                "calibrated_fair_under_american": fair_american(
                    probabilities[routing.SIDE_UNDER]
                ),
                "model_preferred_side": side,
                "freeze_id": approved.approved_freeze_id,
                "freeze_stage": approved.approved_freeze_stage,
                "manifest_sha256": "0" * 64,
                "market_pricing_schema_version": (
                    approved.approved_pricing_schema_version
                ),
                "projection_schema_version": (
                    approved.approved_projection_schema_version
                ),
                "betting_threshold_policy": "monitor_only_no_threshold_frozen",
                "auto_bet": False,
            }

            for direction in (routing.SIDE_OVER, routing.SIDE_UNDER):
                row[f"calibrated_edge_{direction}"] = (
                    probabilities[direction] - devig[direction]
                )

                row[f"calibrated_ev_{direction}"] = expected_value(
                    probabilities[direction], prices[direction]
                )

            row["model_preferred_edge"] = row[f"calibrated_edge_{side}"]

            row["model_preferred_ev"] = row[f"calibrated_ev_{side}"]

            rows.append(row)

    frame = pd.DataFrame(rows)

    # Object dtype so case C's absent price stays absent. The publisher turns
    # a NaN into a null either way, but a column that never held a NaN cannot
    # be confused with one that did.
    for column in ("over_odds", "under_odds"):
        frame[column] = pd.Series(
            [row[column] for row in rows], dtype=object, index=frame.index
        )

    return frame


def wagers() -> Mapping[str, routing.Wager]:
    """Case name -> the wager it is about, built the way the router builds it."""
    return {
        name: routing.Wager(
            game_id=GAME_ID,
            player_id=case["player_id"],
            prop_type=str(case["prop_type"]),
            line_value=case["line_value"],
            side=str(case["side"]),
        )
        for name, case in CASES.items()
    }


def build_receipt(priced_path: Path, *, production_sha: str) -> dict[str, Any]:
    """The receipt the serving run would have written for this slate.

    Built with that module's own builder rather than hand-written, so the
    proof publication carries the receipt shape production emits.
    """
    import run_incumbent_production_serving as serving

    approved = approved_switch()

    return serving.build_receipt(
        slate_date=SLATE_DATE,
        production_sha=production_sha,
        incumbent={
            "authority": serving.AUTHORITY_FROZEN_BUNDLE,
            "bundle_root": "routing-proof",
            "fit_id": None,
            "model_dir": "routing-proof",
            "promoted_at": None,
            "reason": "routing proof data, not a model slate",
            "version": approved.approved_freeze_id,
        },
        readiness={
            "ready": True,
            "reason": "routing proof data",
            "outcome": "REFRESHED",
        },
        fingerprint="routing-proof-slate",
        served={
            "outcome": serving.OUTCOME_SERVED,
            "prediction_artifact": {
                "bytes": 0,
                "path": "routing-proof",
                "sha256": "0" * 64,
            },
            "priced_market_artifact": {
                "bytes": priced_path.stat().st_size,
                "path": str(priced_path),
                "sha256": publisher.sha256_file(priced_path),
            },
            "reason": "routing proof data",
        },
    )


def publish_locally(root: Path, *, production_sha: str) -> Any:
    """Publish the proof slate to a local destination and return its transport.

    Local on purpose. The proof page is served from the real host; the release
    it is built from is proof data and has no business at the feed canary, so
    it is published where only this run can see it and read back from there.
    """
    root = Path(root)

    data_root = root / "data"

    priced_path = publisher.priced_markets_path(data_root, SLATE_DATE)

    priced_path.parent.mkdir(parents=True, exist_ok=True)

    priced_frame().to_parquet(priced_path, index=False)

    receipt_path = publisher.serving_receipt_path(data_root, SLATE_DATE)

    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    receipt_path.write_text(
        json.dumps(
            build_receipt(priced_path, production_sha=production_sha),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    destination = root / "destination"

    publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=SLATE_DATE,
        data_root=data_root,
        project_root=PROJECT_ROOT,
        destination_root=destination,
    )

    return publisher.LocalDirectoryTransport(destination)


# ----------------------------------------------------------------------
# the entry point
# ----------------------------------------------------------------------


def head_sha() -> str:
    import subprocess

    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "0" * 40

    return (done.stdout or "").strip() or "0" * 40


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the routing-proof slate, render the player-props page from "
            "it, and publish that page to the routing-proof directory of the "
            "page canary. Canary only; there is no live mode here."
        )
    )

    parser.add_argument(
        "--work-root",
        default=None,
        help="where the local proof publication is written",
    )

    parser.add_argument(
        "--page-destination-name",
        default="wizardofodds_nba_player_props_page_canary",
        help="named destination the proof page is written beneath",
    )

    parser.add_argument(
        "--page-destination-root",
        default=None,
        help="write the proof page to a local directory instead",
    )

    parser.add_argument(
        "--production-sha",
        default=None,
        help="the code revision recorded on the proof publication",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import tempfile

    arguments = _parser().parse_args(argv)

    work_root = Path(
        arguments.work_root
        or tempfile.mkdtemp(prefix="nba_woo_routing_proof_")
    )

    registry = routing.AffiliateRegistry.load(project_root=PROJECT_ROOT)

    source = publish_locally(
        work_root, production_sha=arguments.production_sha or head_sha()
    )

    if arguments.page_destination_root is not None:
        destination = publisher.LocalDirectoryTransport(
            Path(arguments.page_destination_root)
        )
    else:
        destination = publisher.build_transport(
            publisher.resolve_destination(
                publisher.MODE_CANARY,
                publisher.read_destinations(project_root=PROJECT_ROOT),
                name=arguments.page_destination_name,
            )
        )

    try:
        receipt = page.publish_page(
            source_transport=source,
            page_transport=destination,
            registry=registry,
            prefix=PROOF_PREFIX,
        )
    except (
        page.PageRefused,
        routing.RoutingRefused,
        publisher.PublishingRefused,
    ) as error:
        print(json.dumps({"outcome": "REFUSED", "reason": str(error)}, indent=2))

        return publisher.EXIT_REFUSED
    finally:
        source.close()

        destination.close()

    receipt["work_root"] = str(work_root)

    print(json.dumps(receipt, indent=2, sort_keys=True))

    return publisher.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
