from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


@dataclass
class INGARCH11:
    omega: float = 0.05
    alpha: float = 0.10
    beta: float = 0.75

    def _intensity(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=float)
        if len(counts) == 0:
            return np.array([], dtype=float)
        lam = np.empty(len(counts), dtype=float)
        unconditional = self.omega / max(1.0 - self.alpha - self.beta, 1e-3)
        lam[0] = max(unconditional, 1e-6)
        for t in range(1, len(counts)):
            lam[t] = (
                self.omega
                + self.alpha * counts[t - 1]
                + self.beta * lam[t - 1]
            )
            lam[t] = max(lam[t], 1e-6)
        return lam

    def fit(self, counts: np.ndarray) -> "INGARCH11":
        counts = np.asarray(counts, dtype=float)
        if len(counts) < 30:
            raise ValueError("At least 30 bins are required to fit INGARCH(1,1)")

        def objective(theta: np.ndarray) -> float:
            omega, alpha, beta = theta
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.995:
                return 1e12
            old = (self.omega, self.alpha, self.beta)
            self.omega, self.alpha, self.beta = map(float, theta)
            lam = self._intensity(counts)
            self.omega, self.alpha, self.beta = old
            ll = counts * np.log(lam) - lam - gammaln(counts + 1.0)
            return float(-np.sum(ll[1:]))

        result = minimize(
            objective,
            x0=np.array([self.omega, self.alpha, self.beta]),
            method="Nelder-Mead",
            options={"maxiter": 5000},
        )
        if not result.success:
            raise RuntimeError(f"INGARCH fit failed: {result.message}")
        self.omega, self.alpha, self.beta = map(float, result.x)
        return self

    def fit_sequences(self, sequences: list[np.ndarray]) -> "INGARCH11":
        clean = [
            np.asarray(seq, dtype=float)
            for seq in sequences
            if len(seq) >= 5 and np.isfinite(seq).all()
        ]
        if len(clean) < 10:
            raise ValueError("At least 10 usable game/player sequences are required")

        def sequence_nll(
            seq: np.ndarray,
            omega: float,
            alpha: float,
            beta: float,
        ) -> float:
            unconditional = omega / max(1.0 - alpha - beta, 1e-3)
            lam_prev = max(unconditional, 1e-6)
            total = 0.0
            for t in range(1, len(seq)):
                lam = omega + alpha * seq[t - 1] + beta * lam_prev
                lam = max(lam, 1e-6)
                total -= seq[t] * np.log(lam) - lam - gammaln(seq[t] + 1.0)
                lam_prev = lam
            return total

        def objective(theta: np.ndarray) -> float:
            omega, alpha, beta = theta
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.995:
                return 1e12
            return float(
                sum(sequence_nll(seq, omega, alpha, beta) for seq in clean)
            )

        result = minimize(
            objective,
            x0=np.array([self.omega, self.alpha, self.beta]),
            method="Nelder-Mead",
            options={"maxiter": 5000},
        )
        if not result.success:
            raise RuntimeError(f"INGARCH sequence fit failed: {result.message}")
        self.omega, self.alpha, self.beta = map(float, result.x)
        return self

    def forecast_expected_counts(
        self,
        recent_counts: np.ndarray,
        future_bins: int,
    ) -> np.ndarray:
        recent_counts = np.asarray(recent_counts, dtype=float)
        if len(recent_counts) == 0:
            last_count = 0.0
            last_lambda = self.omega / max(
                1.0 - self.alpha - self.beta, 1e-3
            )
        else:
            historical_lambda = self._intensity(recent_counts)
            last_count = float(recent_counts[-1])
            last_lambda = float(historical_lambda[-1])

        forecast = np.empty(future_bins, dtype=float)
        for i in range(future_bins):
            next_lambda = (
                self.omega + self.alpha * last_count + self.beta * last_lambda
            )
            next_lambda = max(next_lambda, 1e-6)
            forecast[i] = next_lambda
            last_count = next_lambda
            last_lambda = next_lambda
        return forecast


def flatten_live_box_scores(box_scores: list[dict]) -> pd.DataFrame:
    rows = []
    for game in box_scores:
        home_block = game.get("home_team", {}) or {}
        visitor_block = game.get("visitor_team", {}) or {}
        home_id = (home_block.get("team", {}) or {}).get("id")
        visitor_id = (visitor_block.get("team", {}) or {}).get("id")

        for side in ("home_team", "visitor_team"):
            team_block = game.get(side, {}) or {}
            team = team_block.get("team", {}) or {}
            for player_row in team_block.get("players", []) or []:
                player = player_row.get("player", {}) or {}
                rows.append(
                    {
                        "game_id": game.get("id"),
                        "date": game.get("date"),
                        "season": game.get("season"),
                        "status": game.get("status"),
                        "status_state": game.get("status_state"),
                        "period": game.get("period"),
                        "clock": game.get("time"),
                        "home_team_id": home_id,
                        "visitor_team_id": visitor_id,
                        "home_team_score": game.get("home_team_score"),
                        "visitor_team_score": game.get("visitor_team_score"),
                        "team_id": team.get("id"),
                        "player_id": player.get("id"),
                        "min": player_row.get("min"),
                        "pts": player_row.get("pts"),
                        "reb": player_row.get("reb"),
                        "ast": player_row.get("ast"),
                        "stl": player_row.get("stl"),
                        "blk": player_row.get("blk"),
                        "fg3m": player_row.get("fg3m"),
                    }
                )
    return pd.DataFrame(rows)


def archived_snapshots_to_bins(
    snapshots: pd.DataFrame,
    target: str,
    bin_seconds: int = 60,
) -> pd.DataFrame:
    required = {"captured_at", "game_id", "player_id", target}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")

    df = snapshots.copy()
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=["game_id", "player_id", target, "captured_at"])
    df = df.sort_values(["game_id", "player_id", "captured_at"])
    df["jump"] = (
        df.groupby(["game_id", "player_id"])[target]
        .diff()
        .clip(lower=0)
        .fillna(0)
    )

    return (
        df.set_index("captured_at")
        .groupby(["game_id", "player_id"])["jump"]
        .resample(f"{bin_seconds}s")
        .sum()
        .rename("count")
        .reset_index()
    )


def regulation_minutes_remaining(period: int, clock: str | None) -> float | None:
    if period is None or period <= 0:
        return None
    if period > 4:
        return 0.0

    clock_minutes = 0.0
    if clock:
        text = str(clock).strip()
        if ":" in text:
            minute, second = text.split(":", maxsplit=1)
            try:
                clock_minutes = float(minute) + float(second) / 60.0
            except ValueError:
                clock_minutes = 0.0

    return max((4 - int(period)) * 12.0 + clock_minutes, 0.0)


def remaining_minutes_projection(
    pregame_expected_minutes: float,
    minutes_played: float,
    score_margin: float = 0.0,
    regulation_minutes_remaining_value: float | None = None,
) -> float:
    remaining = max(float(pregame_expected_minutes) - float(minutes_played), 0.0)

    if regulation_minutes_remaining_value is not None:
        remaining = min(remaining, float(regulation_minutes_remaining_value))

    abs_margin = abs(float(score_margin))
    if abs_margin >= 30:
        remaining *= 0.35
    elif abs_margin >= 24:
        remaining *= 0.55
    elif abs_margin >= 18:
        remaining *= 0.75

    return max(remaining, 0.0)


def live_mean_projection(
    current_count: int,
    remaining_minutes: float,
    pregame_rate_per_minute: float,
    ingarch: INGARCH11 | None = None,
    recent_counts_per_bin: np.ndarray | None = None,
    bin_minutes: float = 1.0,
) -> float:
    baseline_remaining = remaining_minutes * pregame_rate_per_minute

    if ingarch is None or recent_counts_per_bin is None:
        return float(current_count + baseline_remaining)

    bins = max(int(np.ceil(remaining_minutes / bin_minutes)), 0)
    if bins == 0:
        return float(current_count)

    dynamic = float(
        ingarch.forecast_expected_counts(recent_counts_per_bin, bins).sum()
    )
    remaining = 0.60 * baseline_remaining + 0.40 * dynamic
    return float(current_count + remaining)
