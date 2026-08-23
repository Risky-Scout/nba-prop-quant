from __future__ import annotations

from typing import Any

import pandas as pd

BET365_SCHEMA_VERSION = "bet365_nba_feed_v1"

BET365_ID_COLUMNS = (
    "bet365_event_id",
    "bet365_player_id",
    "bet365_market_id",
    "bet365_over_selection_id",
    "bet365_under_selection_id",
)


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _validate_external_ids(
    canonical: pd.DataFrame,
) -> None:
    missing_columns = [
        column
        for column in BET365_ID_COLUMNS
        if column not in canonical.columns
    ]
    if missing_columns:
        raise ValueError(
            "Bet365 authorized-API handoff requires mapping "
            f"columns: {missing_columns}"
        )

    missing_values = {
        column: int(canonical[column].isna().sum())
        for column in BET365_ID_COLUMNS
        if canonical[column].isna().any()
    }
    if missing_values:
        raise ValueError(
            "Bet365 mapping contains missing external IDs: "
            f"{missing_values}"
        )


def to_bet365_payload(
    canonical: pd.DataFrame,
    *,
    require_external_ids: bool = False,
) -> dict[str, Any]:
    if canonical.empty:
        return {
            "schema_version": BET365_SCHEMA_VERSION,
            "transport_contract": "authorized_api_unbound",
            "generated_at_utc": None,
            "model_freeze_id": None,
            "records": [],
        }

    required = {
        "generated_at_utc",
        "freeze_id",
        "game_id",
        "player_id",
        "prop_type",
        "line_value",
        "selected_p_over",
        "selected_p_under",
        "selected_p_push",
        "selected_q_over_nonpush",
        "selected_q_under_nonpush",
        "fair_over_american",
        "fair_under_american",
        "fair_over_decimal",
        "fair_under_decimal",
    }
    missing = required - set(canonical.columns)
    if missing:
        raise ValueError(
            "Bet365 adapter missing canonical fields: "
            f"{sorted(missing)}"
        )

    if require_external_ids:
        _validate_external_ids(canonical)

    records = []

    for _, row in canonical.iterrows():
        records.append(
            {
                "request_id": _clean(row.get("request_id")),
                "event": {
                    "internal_game_id": int(row["game_id"]),
                    "bet365_event_id": _clean(
                        row.get("bet365_event_id")
                    ),
                    "game_date": _clean(row.get("game_date")),
                    "team": _clean(row.get("team")),
                    "opponent": _clean(row.get("opponent")),
                },
                "participant": {
                    "internal_player_id": int(row["player_id"]),
                    "bet365_player_id": _clean(
                        row.get("bet365_player_id")
                    ),
                    "player_name": _clean(
                        row.get("player_name")
                    ),
                },
                "market": {
                    "prop_type": str(row["prop_type"]),
                    "line": float(row["line_value"]),
                    "bet365_market_id": _clean(
                        row.get("bet365_market_id")
                    ),
                },
                "selections": {
                    "over": {
                        "bet365_selection_id": _clean(
                            row.get("bet365_over_selection_id")
                        ),
                        "fair_probability": float(
                            row["selected_p_over"]
                        ),
                        "fair_probability_nonpush": float(
                            row["selected_q_over_nonpush"]
                        ),
                        "fair_american": int(
                            row["fair_over_american"]
                        ),
                        "fair_decimal": float(
                            row["fair_over_decimal"]
                        ),
                    },
                    "under": {
                        "bet365_selection_id": _clean(
                            row.get("bet365_under_selection_id")
                        ),
                        "fair_probability": float(
                            row["selected_p_under"]
                        ),
                        "fair_probability_nonpush": float(
                            row["selected_q_under_nonpush"]
                        ),
                        "fair_american": int(
                            row["fair_under_american"]
                        ),
                        "fair_decimal": float(
                            row["fair_under_decimal"]
                        ),
                    },
                },
                "push_probability": float(
                    row["selected_p_push"]
                ),
                "model": {
                    "freeze_id": str(row["freeze_id"]),
                    "calibration_method": _clean(
                        row.get("calibration_method")
                    ),
                    "dependence_lambda": _clean(
                        row.get("dependence_lambda")
                    ),
                    "pricing_method": _clean(
                        row.get("pricing_method")
                    ),
                    "market_independent": bool(
                        row.get("market_independent", True)
                    ),
                    "auto_bet": bool(
                        row.get("auto_bet", False)
                    ),
                },
            }
        )

    return {
        "schema_version": BET365_SCHEMA_VERSION,
        "transport_contract": "authorized_api_unbound",
        "generated_at_utc": str(
            canonical["generated_at_utc"].iloc[0]
        ),
        "model_freeze_id": str(
            canonical["freeze_id"].iloc[0]
        ),
        "records": records,
    }
