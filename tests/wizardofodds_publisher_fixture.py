"""A priced slate and its serving receipt, without running the models.

The publisher's own behaviour -- which switch states refuse, which rows the
fail-closed rules reject, whether a release verifies, whether a rollback
restores -- does not depend on the numbers being the frozen models' output. It
depends on the columns being the ones the integration contract names and the
receipt provenancing the bytes on disk, and both of those can be built in a
millisecond.

So this builds the smallest thing the publisher accepts, and every refusal
test mutates one field of it. The real frozen models scoring a real slate and
being published end to end is a separate test, against the installed runtime
bundle, because that one is about the whole path rather than about the
publisher's rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

if str(PROJECT / "ops") not in sys.path:
    sys.path.insert(0, str(PROJECT / "ops"))

import publish_incumbent_to_wizardofodds as publisher  # noqa: E402

SLATE_DATE = "2026-10-20"
HISTORY_DATE = "2026-10-18"

#: Pass as a ``receipt_changes`` value to drop a key rather than reset it.
#: Dropping and emptying are different defects and the publisher distinguishes
#: them, so the fixture has to be able to produce both.
DELETE = object()

#: The deployment the committed switch approves. Read from the switch rather
#: than restated, so a fixture that disagreed with the approval would be a
#: fixture bug rather than a silent pass.
APPROVED = publisher.read_publishing_switch(
    publisher.MODE_CANARY, project_root=PROJECT
)

FREEZE_ID = APPROVED.approved_freeze_id
FREEZE_STAGE = APPROVED.approved_freeze_stage
MANIFEST_SHA256 = "7c70a1a82145ef0ea1ef69806a5edc363704a285df638796212b23810d8e6823"

#: Four quotes across two games, two books and two players, so the logical key
#: has something to distinguish and the sort order has something to sort.
QUOTES: tuple[tuple[int, int, int, str, float, str], ...] = (
    (7001, 9901, 1101, "points", 18.5, "fixturebook"),
    (7002, 9901, 1101, "rebounds", 4.5, "fixturebook"),
    (7003, 9901, 1102, "points", 14.5, "otherbook"),
    (7004, 9902, 2201, "threes", 1.5, "fixturebook"),
)

#: What the schedule calls each of the two teams. Three letters, because that
#: is what ``normalize_games`` carries out of the game payload.
ABBREVIATION: dict[int, str] = {11: "HOM", 22: "AWY"}

#: The public name of each player, shaped like a name. The publisher refuses a
#: name that is an identifier wearing a person's slot, and a fixture named
#: after its own player id would be exercising that refusal rather than the
#: happy path every other test here is about.
PLAYER_NAME: dict[int, str] = {
    1101: "Alpha Example",
    1102: "Beta Example",
    2201: "Zeta Example",
}

#: When each game tips, as a string. The publisher renders a timestamp as a
#: date, so a fixture carrying this as a ``pd.Timestamp`` would publish a tip
#: time with no time of day in it and prove nothing.
TIP_OFF: dict[int, str] = {
    9901: "2026-10-20T23:00:00+00:00",
    9902: "2026-10-21T01:30:00+00:00",
}


def priced_frame() -> pd.DataFrame:
    """One row per quote, carrying every column the publisher reads."""
    rows: list[dict[str, Any]] = []

    for index, (quote_id, game_id, player_id, prop, line, vendor) in enumerate(
        QUOTES
    ):
        over = 0.46 + (index / 100.0)

        team_id = 11 if player_id < 2000 else 22
        opponent_id = 22 if player_id < 2000 else 11
        is_home = 1 if player_id < 2000 else 0

        # Derived from the one fact the row already states rather than listed
        # again, so the fixture cannot name a home side that disagrees with
        # its own ``is_home``.
        home_team_id = team_id if is_home else opponent_id
        visitor_team_id = opponent_id if is_home else team_id

        rows.append(
            {
                "id": quote_id,
                "vendor": vendor,
                "game_id": game_id,
                "date": pd.Timestamp(SLATE_DATE),
                "player_id": player_id,
                "player_name": PLAYER_NAME[player_id],
                "team_id": team_id,
                "opponent_id": opponent_id,
                "is_home": is_home,
                "home_team_abbreviation": ABBREVIATION[home_team_id],
                "visitor_team_abbreviation": ABBREVIATION[visitor_team_id],
                "player_team_abbreviation": ABBREVIATION[team_id],
                "tip_off_utc": TIP_OFF[game_id],
                "prop_type": prop,
                "line_value": line,
                # Not the line, and not a function of the price: the same
                # distance above it for every row, so a test that confused
                # the projection with the line or the midpoint would read a
                # different number.
                "model_projection": line + 0.4,
                "market_type": "over_under",
                "over_odds": -110,
                "under_odds": -110,
                "quote_eligible": True,
                "quote_filter_reason": "",
                "availability_out": 0,
                "gate3_role_ready": 1,
                "history_latest_date": pd.Timestamp(HISTORY_DATE),
                "advanced_latest_date": pd.Timestamp(HISTORY_DATE),
                "calibrated_q_over_nonpush": over,
                "calibrated_q_under_nonpush": 1.0 - over,
                "calibrated_p_push": 0.0,
                "calibration_method": "prop",
                "market_devig_q_over": 0.5,
                "market_devig_q_under": 0.5,
                "market_hold_pct": 4.761904761904762,
                "calibrated_fair_over_american": 113 + index,
                "calibrated_fair_under_american": -113 - index,
                "calibrated_edge_over": over - 0.5,
                "calibrated_edge_under": 0.5 - over,
                "calibrated_ev_over": -0.1 + (index / 1000.0),
                "calibrated_ev_under": 0.01 + (index / 1000.0),
                "model_preferred_side": "under",
                "model_preferred_edge": 0.5 - over,
                "model_preferred_ev": 0.01 + (index / 1000.0),
                "freeze_id": FREEZE_ID,
                "freeze_stage": FREEZE_STAGE,
                "manifest_sha256": MANIFEST_SHA256,
                "market_pricing_schema_version": 3,
                "projection_schema_version": 3,
                "betting_threshold_policy": "monitor_only_no_threshold_frozen",
                "auto_bet": False,
            }
        )

    return pd.DataFrame(rows)


def receipt_for(priced_path: Path, *, slate_date: str = SLATE_DATE) -> dict[str, Any]:
    """The receipt ``run_incumbent_production_serving`` would have written.

    Built with that module's own ``build_receipt``, not a hand-written dict,
    so the publisher is tested against the receipt shape production actually
    emits and a change to that shape shows up here.
    """
    import run_incumbent_production_serving as serving

    return serving.build_receipt(
        slate_date=slate_date,
        production_sha="0" * 40,
        incumbent={
            "authority": serving.AUTHORITY_FROZEN_BUNDLE,
            "bundle_root": "/fixture/bundle",
            "fit_id": None,
            "model_dir": "/fixture/bundle/models",
            "promoted_at": None,
            "reason": "the frozen bundle is the incumbent in this fixture",
            "version": FREEZE_ID,
        },
        readiness={"ready": True, "reason": "fixture", "outcome": "REFRESHED"},
        fingerprint="fixture-input-state-fingerprint",
        served={
            "outcome": serving.OUTCOME_SERVED,
            "prediction_artifact": {
                "bytes": 1,
                "path": str(priced_path.with_name("projections.parquet")),
                "sha256": "0" * 64,
            },
            "priced_market_artifact": {
                "bytes": priced_path.stat().st_size,
                "path": str(priced_path),
                "sha256": publisher.sha256_file(priced_path),
            },
            "reason": "fixture",
        },
    )


def build(
    root: Path,
    *,
    frame: pd.DataFrame | None = None,
    slate_date: str = SLATE_DATE,
    receipt_changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the priced parquet and its receipt under ``root``.

    ``receipt_changes`` is applied after the receipt is built and after its
    digest was taken, which is what lets a test break exactly one provenance
    field and leave the rest correct.
    """
    data_root = Path(root) / "data"

    priced_path = publisher.priced_markets_path(data_root, slate_date)

    priced_path.parent.mkdir(parents=True, exist_ok=True)

    (frame if frame is not None else priced_frame()).to_parquet(
        priced_path, index=False
    )

    receipt = receipt_for(priced_path, slate_date=slate_date)

    if receipt_changes:
        for key, value in receipt_changes.items():
            if value is DELETE:
                receipt.pop(key, None)
            else:
                receipt[key] = value

    receipt_path = publisher.serving_receipt_path(data_root, slate_date)

    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "data_root": data_root,
        "destination_root": Path(root) / "destination",
        "priced_path": priced_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "slate_date": slate_date,
    }
