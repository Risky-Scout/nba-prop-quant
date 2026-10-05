"""Held-out validation metrics for the shadow latent-state layer.

SHADOW / RESEARCH ONLY.

Three families of evidence, matching the brief:

A. **Marginal preservation.** Every simulated univariate margin must still be
   the production margin. The reference is analytic, taken from the tabulated
   production CDF rather than from a second simulation, so the only noise in
   the comparison is the candidate's own Monte Carlo noise and the tolerance
   can be stated as a z-score against the binomial / sample-mean standard
   error.

B. **Residual dependence reproduction.** Observed held-out cross-player
   correlation against model-implied, reported per named relationship bucket
   with game-clustered bootstrap intervals, in both latent space (where the
   model is parameterised) and realized count space (where a bettor is
   exposed).

C. **Joint event calibration.** Historical 2-, 3- and 4-leg same-game
   conjunctions, graded against realized box scores, scored with Brier and
   log loss plus reliability buckets and game-clustered bootstrap intervals.

Lines for the joint events are set from the *predictive* marginal only, at
preregistered probability levels, so no realized value from the graded game
influences the event definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...distributions import FittedMarginal
from .query import PropLeg
from .simulator import GameRoster, GameSimulation, tabulate_inverse_cdf

# Relationship buckets reported for residual dependence. ``kind`` is one of
# ``same_player``, ``same_team`` or ``cross_team``; ``stats`` is the ordered
# stat pair.
DEPENDENCE_BUCKETS: tuple[tuple[str, str, tuple[str, str]], ...] = (
    ("same_player_pts_reb", "same_player", ("pts", "reb")),
    ("same_player_pts_ast", "same_player", ("pts", "ast")),
    ("same_player_reb_ast", "same_player", ("reb", "ast")),
    ("same_player_stl_blk", "same_player", ("stl", "blk")),
    ("same_player_pts_fg3m", "same_player", ("pts", "fg3m")),
    ("teammate_pts_pts", "same_team", ("pts", "pts")),
    ("teammate_reb_reb", "same_team", ("reb", "reb")),
    ("teammate_ast_ast", "same_team", ("ast", "ast")),
    ("passer_ast_teammate_pts", "same_team", ("ast", "pts")),
    ("teammate_pts_reb", "same_team", ("pts", "reb")),
    ("teammate_fg3m_fg3m", "same_team", ("fg3m", "fg3m")),
    ("opponent_pts_pts", "cross_team", ("pts", "pts")),
    ("opponent_reb_reb", "cross_team", ("reb", "reb")),
    ("opponent_pts_reb", "cross_team", ("pts", "reb")),
    ("opponent_fg3m_reb", "cross_team", ("fg3m", "reb")),
    ("opponent_ast_ast", "cross_team", ("ast", "ast")),
    ("opponent_stl_pts", "cross_team", ("stl", "pts")),
)

# Reliability buckets for joint-event calibration.
RELIABILITY_EDGES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.0)


# ----------------------------------------------------------------------
# A. marginal preservation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticMarginal:
    cdf: np.ndarray

    @property
    def survival(self) -> np.ndarray:
        return 1.0 - self.cdf

    def mean(self) -> float:
        return float(np.sum(self.survival))

    def variance(self) -> float:
        k = np.arange(len(self.cdf))
        second = float(np.sum((2.0 * k + 1.0) * self.survival))
        mean = self.mean()
        return float(max(second - mean**2, 0.0))

    def quantile(self, probability: float) -> float:
        return float(np.searchsorted(self.cdf, probability, side="left"))

    def probability_over(self, line: float) -> float:
        floor_line = int(np.floor(line))
        if floor_line < 0:
            return 1.0
        if floor_line >= len(self.cdf):
            return 0.0
        return float(1.0 - self.cdf[floor_line])


def analytic_marginals(
    roster: GameRoster,
    marginals: Mapping[str, FittedMarginal],
) -> dict[tuple[int, str], AnalyticMarginal]:
    out: dict[tuple[int, str], AnalyticMarginal] = {}
    for _, row in roster.frame.iterrows():
        for stat in roster.stats:
            cdf = tabulate_inverse_cdf(
                marginals[stat], float(row[f"mu_selected_{stat}"]), row
            )
            out[(int(row["player_id"]), stat)] = AnalyticMarginal(cdf=cdf)
    return out


def marginal_preservation(
    simulation: GameSimulation,
    reference: Mapping[tuple[int, str], AnalyticMarginal],
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    probe_quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    """Per-dimension simulated-vs-analytic marginal comparison.

    ``mean_z`` is the simulated/analytic mean gap in units of the Monte Carlo
    standard error of the sample mean; ``max_abs_over_z`` the same for the
    over-probability at the probe lines. Both are the quantities the Gate A
    tolerance is stated in.

    Each row also reports how many z-probes it contributed and how many of
    them exceeded 3 sigma. Gate A corrects its critical value for the total
    probe count, so that count has to be carried rather than reconstructed:
    only the per-row *maxima* survive aggregation otherwise, and a maximum
    cannot tell you how many comparisons produced it.
    """
    records = []
    for player_index, player_id in enumerate(simulation.player_ids):
        for stat_index, stat in enumerate(simulation.stats):
            draws = simulation.draws[:, player_index, stat_index]
            analytic = reference[(player_id, stat)]
            n = len(draws)

            simulated_mean = float(np.mean(draws))
            simulated_var = float(np.var(draws, ddof=1))
            analytic_mean = analytic.mean()
            analytic_var = analytic.variance()
            mean_se = float(np.sqrt(max(simulated_var, 1e-12) / n))

            # Monte Carlo standard error of the sample variance,
            # ``sqrt((mu4 - sigma^4) / n)``. Box-score counts are far from
            # normal -- a bench player's blocks are nearly Bernoulli -- so the
            # Gaussian ``sigma^2 sqrt(2/n)`` form would be badly wrong, and the
            # fourth central moment has to be taken from the draws.
            fourth = float(np.mean((draws - simulated_mean) ** 4))
            variance_se = float(
                np.sqrt(max(fourth - simulated_var**2, 1e-24) / n)
            )

            record: dict[str, object] = {
                "game_id": simulation.game_id,
                "player_id": player_id,
                "stat": stat,
                "simulations": n,
                "simulated_mean": simulated_mean,
                "analytic_mean": analytic_mean,
                "mean_abs_error": abs(simulated_mean - analytic_mean),
                "mean_z": (simulated_mean - analytic_mean) / mean_se if mean_se > 0 else 0.0,
                "simulated_variance": simulated_var,
                "analytic_variance": analytic_var,
                "variance_z": (
                    (simulated_var - analytic_var) / variance_se
                    if variance_se > 0
                    else 0.0
                ),
                "variance_relative_error": (
                    (simulated_var - analytic_var) / analytic_var
                    if analytic_var > 1e-9
                    else 0.0
                ),
            }

            quantile_errors = []
            for probability in quantiles:
                simulated_q = float(np.quantile(draws, probability))
                analytic_q = analytic.quantile(probability)
                quantile_errors.append(abs(simulated_q - analytic_q))
            record["max_abs_quantile_error"] = float(max(quantile_errors))

            over_z = []
            over_errors = []
            for probability in probe_quantiles:
                line = analytic.quantile(probability) + 0.5
                analytic_over = analytic.probability_over(line)
                simulated_over = float(np.mean(draws > line))
                se = float(
                    np.sqrt(max(analytic_over * (1.0 - analytic_over), 1e-12) / n)
                )
                over_errors.append(abs(simulated_over - analytic_over))
                over_z.append(abs(simulated_over - analytic_over) / se if se > 0 else 0.0)
            record["max_abs_over_error"] = float(max(over_errors))
            record["max_abs_over_z"] = float(max(over_z))

            all_z = [abs(float(record["mean_z"])), abs(float(record["variance_z"])), *over_z]
            record["z_probe_count"] = len(all_z)
            record["z_probes_beyond_3sigma"] = int(
                sum(1 for value in all_z if value > 3.0)
            )
            records.append(record)

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# B. residual dependence
# ----------------------------------------------------------------------


def standardized_count_residuals(
    frame: pd.DataFrame,
    stats: Sequence[str],
    reference: Mapping[tuple[int, str], AnalyticMarginal],
    prefix: str = "e_",
) -> pd.DataFrame:
    """``(y - E[Y]) / sd(Y)`` from the production marginal, per observation."""
    out = frame.copy()
    for stat in stats:
        values = np.empty(len(out), dtype=float)
        for position, (_, row) in enumerate(out.iterrows()):
            analytic = reference[(int(row["player_id"]), stat)]
            sd = float(np.sqrt(max(analytic.variance(), 1e-12)))
            values[position] = (float(row[f"y_{stat}"]) - analytic.mean()) / sd
        out[f"{prefix}{stat}"] = values
    return out


def simulated_pair_moments(
    simulation: GameSimulation,
    reference: Mapping[tuple[int, str], AnalyticMarginal],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Same-team and cross-team standardized pair moments of the simulation.

    Returns ``(same_team, cross_team, same_pairs, cross_pairs)`` where both
    matrices are averages over draws, so they are directly comparable with the
    observed pooled moments from :func:`factors.pair_moments`.
    """
    n_stats = len(simulation.stats)
    standardized = np.empty_like(simulation.draws, dtype=float)
    for player_index, player_id in enumerate(simulation.player_ids):
        for stat_index, stat in enumerate(simulation.stats):
            analytic = reference[(player_id, stat)]
            sd = float(np.sqrt(max(analytic.variance(), 1e-12)))
            standardized[:, player_index, stat_index] = (
                simulation.draws[:, player_index, stat_index] - analytic.mean()
            ) / sd

    teams = np.asarray(simulation.team_ids)
    unique_teams = sorted({int(team) for team in teams})

    same = np.zeros((n_stats, n_stats), dtype=float)
    same_pairs = 0.0
    totals: list[np.ndarray] = []
    counts: list[int] = []

    for team in unique_teams:
        mask = teams == team
        block = standardized[:, mask, :]
        count = int(mask.sum())
        total = block.sum(axis=1)
        totals.append(total)
        counts.append(count)
        if count >= 2:
            outer = np.einsum("ds,dt->st", total, total)
            within = np.einsum("dps,dpt->st", block, block)
            same += (outer - within) / simulation.simulations
            same_pairs += count * (count - 1)

    cross = np.zeros((n_stats, n_stats), dtype=float)
    cross_pairs = 0.0
    for first in range(len(totals)):
        for second in range(first + 1, len(totals)):
            outer = np.einsum("ds,dt->st", totals[first], totals[second])
            cross += (outer + outer.T) / simulation.simulations
            cross_pairs += 2.0 * counts[first] * counts[second]

    return same, cross, same_pairs, cross_pairs


def bucket_values(
    stats: Sequence[str],
    same_team: np.ndarray,
    cross_team: np.ndarray,
    same_player: np.ndarray | None = None,
) -> dict[str, float]:
    index = {stat: position for position, stat in enumerate(stats)}
    out: dict[str, float] = {}
    for name, kind, (first, second) in DEPENDENCE_BUCKETS:
        if first not in index or second not in index:
            continue
        i, j = index[first], index[second]
        if kind == "same_team":
            out[name] = float(same_team[i, j])
        elif kind == "cross_team":
            out[name] = float(cross_team[i, j])
        elif same_player is not None:
            out[name] = float(same_player[i, j])
    return out


# ----------------------------------------------------------------------
# C. joint event calibration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class JointEvent:
    game_id: int
    family: str
    legs: tuple[PropLeg, ...]
    realized: int

    @property
    def n_legs(self) -> int:
        return len(self.legs)


EVENT_FAMILIES: tuple[tuple[str, int, str], ...] = (
    ("2leg_same_player", 2, "same_player"),
    ("2leg_same_team", 2, "same_team"),
    ("2leg_cross_team", 2, "cross_team"),
    ("3leg_same_team", 3, "same_team"),
    ("3leg_mixed", 3, "mixed"),
    ("3leg_cross_team", 3, "cross_team"),
    ("4leg_mixed", 4, "mixed"),
    ("4leg_cross_team", 4, "cross_team"),
)

# Probability levels the generated lines sit at. A leg is placed so its
# predictive probability is near one of these, which keeps the generated
# conjunctions inside the range a real same-game parlay occupies.
LEG_PROBABILITY_LEVELS = (0.35, 0.5, 0.65)


def _leg_from_level(
    player_id: int,
    stat: str,
    analytic: AnalyticMarginal,
    level: float,
    side: str,
) -> PropLeg:
    """Place a half-integer line so the leg's predictive probability ~ level."""
    target = 1.0 - level if side == "over" else level
    line = analytic.quantile(target) + 0.5
    return PropLeg(player_id=player_id, stat=stat, side=side, line=float(line))


def generate_joint_events(
    observations: pd.DataFrame,
    reference: Mapping[tuple[int, str], AnalyticMarginal],
    stats: Sequence[str],
    game_id: int,
    seed: int,
    events_per_family: int = 2,
    min_expected_minutes: float = 12.0,
) -> list[JointEvent]:
    """Build graded same-game conjunctions for one held-out game.

    Leg players, stats, sides and probability levels are drawn from a locked
    per-game RNG; lines come from the predictive marginal alone. Realized
    grading uses the observed box score, which is the only place a realized
    value enters.
    """
    eligible = observations.loc[
        observations["expected_minutes"].fillna(0.0) >= min_expected_minutes
    ]
    if eligible["player_id"].nunique() < 4:
        return []

    rng = np.random.default_rng(seed)
    by_team: dict[int, list[int]] = {}
    for _, row in eligible.iterrows():
        by_team.setdefault(int(row["team_id"]), []).append(int(row["player_id"]))
    teams = [team for team, players in by_team.items() if len(players) >= 3]
    if len(teams) < 2:
        return []

    realized_lookup = {
        (int(row["player_id"]), stat): float(row[f"y_{stat}"])
        for _, row in eligible.iterrows()
        for stat in stats
    }

    events: list[JointEvent] = []

    def build(players: Sequence[int], count: int) -> tuple[PropLeg, ...] | None:
        legs: list[PropLeg] = []
        used: set[tuple[int, str]] = set()
        for index in range(count):
            player = int(players[index % len(players)])
            choices = [
                stat for stat in stats if (player, stat) not in used and stat != "blk"
            ]
            if not choices:
                return None
            stat = str(rng.choice(choices))
            used.add((player, stat))
            side = "over" if rng.random() < 0.6 else "under"
            level = float(rng.choice(LEG_PROBABILITY_LEVELS))
            analytic = reference.get((player, stat))
            if analytic is None:
                return None
            legs.append(_leg_from_level(player, stat, analytic, level, side))
        return tuple(legs)

    for family, n_legs, shape in EVENT_FAMILIES:
        for _ in range(events_per_family):
            team_a, team_b = (
                teams[0],
                teams[1],
            )
            if rng.random() < 0.5:
                team_a, team_b = team_b, team_a

            if shape == "same_player":
                player = int(rng.choice(by_team[team_a]))
                players: list[int] = [player] * n_legs
            elif shape == "same_team":
                players = list(
                    rng.choice(by_team[team_a], size=n_legs, replace=False)
                )
            elif shape == "cross_team":
                half = n_legs // 2
                players = list(
                    rng.choice(by_team[team_a], size=max(half, 1), replace=False)
                ) + list(
                    rng.choice(
                        by_team[team_b], size=n_legs - max(half, 1), replace=False
                    )
                )
            else:  # mixed: a repeated player plus teammates and an opponent
                anchor = int(rng.choice(by_team[team_a]))
                others = [
                    player for player in by_team[team_a] if player != anchor
                ]
                players = [anchor, anchor]
                if n_legs >= 3 and others:
                    players.append(int(rng.choice(others)))
                if n_legs >= 4:
                    players.append(int(rng.choice(by_team[team_b])))
                players = players[:n_legs]

            legs = build(players, n_legs)
            if legs is None:
                continue

            satisfied = True
            valid = True
            for leg in legs:
                key = (leg.player_id, leg.stat)
                if key not in realized_lookup:
                    valid = False
                    break
                value = realized_lookup[key]
                if leg.side == "over":
                    satisfied &= value > leg.line
                else:
                    satisfied &= value < leg.line
            if not valid:
                continue

            events.append(
                JointEvent(
                    game_id=int(game_id),
                    family=family,
                    legs=legs,
                    realized=int(satisfied),
                )
            )

    return events


def brier_score(probability: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((np.asarray(probability) - np.asarray(outcome)) ** 2))


def log_loss(
    probability: np.ndarray,
    outcome: np.ndarray,
    floor: float = 1e-6,
) -> float:
    """Log loss with a declared probability floor.

    A Monte Carlo probability can be exactly 0 or 1, at which point log loss
    is undefined. The floor is the stated numerical bound; the score is only
    reported when the floor binds on a small enough share of events for the
    number to be meaningful, which the caller checks.
    """
    p = np.clip(np.asarray(probability, dtype=float), floor, 1.0 - floor)
    y = np.asarray(outcome, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def floor_binding_fraction(probability: np.ndarray, floor: float = 1e-6) -> float:
    p = np.asarray(probability, dtype=float)
    return float(np.mean((p <= floor) | (p >= 1.0 - floor)))


def reliability_table(
    probability: np.ndarray,
    outcome: np.ndarray,
    edges: Sequence[float] = RELIABILITY_EDGES,
) -> pd.DataFrame:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcome, dtype=float)
    index = np.clip(np.digitize(p, np.asarray(edges)[1:-1], right=True), 0, len(edges) - 2)
    records = []
    for bucket in range(len(edges) - 1):
        mask = index == bucket
        if not np.any(mask):
            continue
        records.append(
            {
                "bucket_low": float(edges[bucket]),
                "bucket_high": float(edges[bucket + 1]),
                "count": int(mask.sum()),
                "mean_predicted": float(np.mean(p[mask])),
                "observed_rate": float(np.mean(y[mask])),
            }
        )
    return pd.DataFrame(records)


def clustered_bootstrap_ci(
    values: pd.DataFrame,
    cluster_column: str,
    statistic,
    draws: int = 500,
    seed: int = 73,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI resampling whole clusters (games)."""
    clusters = values[cluster_column].to_numpy()
    unique = np.unique(clusters)
    if len(unique) < 2:
        return (float("nan"), float("nan"))
    lookup = {key: values.loc[clusters == key] for key in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        sample = pd.concat([lookup[key] for key in picked], ignore_index=True)
        estimates[draw] = statistic(sample)
    return (
        float(np.nanquantile(estimates, alpha / 2.0)),
        float(np.nanquantile(estimates, 1.0 - alpha / 2.0)),
    )
