from __future__ import annotations

import numpy as np
import pandas as pd


OUT_PATTERNS = ("out", "inactive", "suspended")


def _is_out(status: object) -> bool:
    text = str(status or "").strip().lower()
    return any(pattern in text for pattern in OUT_PATTERNS)


def apply_current_injury_adjustment(
    slate: pd.DataFrame,
    injuries: pd.DataFrame,
    expected_minutes_col: str = "expected_minutes",
    max_minutes: float = 42.0,
) -> pd.DataFrame:
    """
    Production-only availability adjustment.

    BDL's injuries endpoint is a current snapshot, not a historical snapshot archive.
    Therefore this function should not be used in a historical backtest unless the
    injury record was captured before tip-off by your own collector.
    """
    df = slate.copy()
    injury_map = {}
    if injuries is not None and not injuries.empty:
        injury_map = (
            injuries.dropna(subset=["player_id"])
            .set_index("player_id")["status"]
            .to_dict()
        )

    df["availability_status"] = df["player_id"].map(injury_map).fillna("")
    df["availability_out"] = df["availability_status"].map(_is_out).astype(int)
    df["availability_base_minutes"] = df[expected_minutes_col].clip(lower=0.0)

    adjusted_groups = []
    for team_id, team in df.groupby("team_id", sort=False):
        team = team.copy()
        base = team["availability_base_minutes"].to_numpy(dtype=float)
        out = team["availability_out"].to_numpy(dtype=bool)

        adjusted = base.copy()
        freed = float(adjusted[out].sum())
        adjusted[out] = 0.0

        healthy = ~out
        if freed > 0 and healthy.any():
            # Elasticity favors players with a real rotation role but leaves more room
            # for players who are not already projected near a star-level minutes cap.
            weights = np.sqrt(np.clip(base, 0.0, None) + 0.5) * np.clip(
                1.0 - base / 48.0, 0.05, 1.0
            )
            weights[~healthy] = 0.0

            remaining = freed
            for _ in range(6):
                room = np.clip(max_minutes - adjusted, 0.0, None)
                eligible_weight = weights * (room > 1e-9)
                total_weight = float(eligible_weight.sum())
                if remaining <= 1e-6 or total_weight <= 0:
                    break
                allocation = remaining * eligible_weight / total_weight
                allocation = np.minimum(allocation, room)
                adjusted += allocation
                remaining -= float(allocation.sum())

        team["availability_expected_minutes"] = adjusted
        team["availability_minutes_delta"] = adjusted - base

        if "adv_prior_usage_percentage" in team.columns:
            usage = team["adv_prior_usage_percentage"].fillna(
                team["adv_prior_usage_percentage"].median()
            ).fillna(0.20)
            lost_usage_minutes = float((base[out] * usage.to_numpy()[out]).sum())
            healthy_minutes = np.clip(adjusted, 0.0, None)
            denom = float((healthy_minutes[healthy] * usage.to_numpy()[healthy]).sum())
            multiplier = np.ones(len(team), dtype=float)
            if lost_usage_minutes > 0 and denom > 0:
                share = (
                    healthy_minutes * usage.to_numpy()
                    / max(denom, 1e-12)
                )
                multiplier += 0.35 * lost_usage_minutes * share / np.maximum(
                    healthy_minutes, 1.0
                )
            multiplier[out] = 0.0
            team["availability_usage_multiplier"] = multiplier
        else:
            team["availability_usage_multiplier"] = np.where(out, 0.0, 1.0)

        adjusted_groups.append(team)

    return pd.concat(adjusted_groups, ignore_index=True)
