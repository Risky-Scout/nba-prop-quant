from __future__ import annotations

from typing import Any

import pandas as pd

WIZARDOFODDS_SCHEMA_VERSION = "wizardofodds_nba_feed_v1"


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def to_wizardofodds_payload(
    canonical: pd.DataFrame,
) -> dict[str, Any]:
    if canonical.empty:
        return {
            "schema_version": WIZARDOFODDS_SCHEMA_VERSION,
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
            "WizardOfOdds adapter missing canonical fields: "
            f"{sorted(missing)}"
        )

    records = []

    for _, row in canonical.iterrows():
        records.append(
            {
                "game_id": int(row["game_id"]),
                "game_date": _clean(row.get("game_date")),
                "player_id": int(row["player_id"]),
                "player_name": _clean(row.get("player_name")),
                "team": _clean(row.get("team")),
                "opponent": _clean(row.get("opponent")),
                "prop_type": str(row["prop_type"]),
                "line": float(row["line_value"]),
                "expected_minutes": _clean(
                    row.get("expected_minutes")
                ),
                "projected_mean": _clean(
                    row.get("mu_selected")
                ),
                "over": {
                    "probability": float(
                        row["selected_p_over"]
                    ),
                    "probability_nonpush": float(
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
                    "probability": float(
                        row["selected_p_under"]
                    ),
                    "probability_nonpush": float(
                        row["selected_q_under_nonpush"]
                    ),
                    "fair_american": int(
                        row["fair_under_american"]
                    ),
                    "fair_decimal": float(
                        row["fair_under_decimal"]
                    ),
                },
                "push_probability": float(
                    row["selected_p_push"]
                ),
                "audit": {
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
        "schema_version": WIZARDOFODDS_SCHEMA_VERSION,
        "generated_at_utc": str(
            canonical["generated_at_utc"].iloc[0]
        ),
        "model_freeze_id": str(
            canonical["freeze_id"].iloc[0]
        ),
        "records": records,
    }
