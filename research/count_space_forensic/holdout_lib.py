"""The exact held-out transmission predictor for the forensic study.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Why an exact predictor instead of the simulator
-----------------------------------------------
Section 3 asks for a *feasibility envelope*: the largest count-space
correlation the frozen architecture can reach on one bucket while every
global constraint still holds. Answering that means evaluating many candidate
loading sets, and the published evidence costs an 80-minute Monte Carlo run
per set. It does not have to.

For distinct players the assembled game covariance has a rank-one structure
in the per-player scale. ``build_game_covariance`` writes a player's shared
rows as ``shrink_i * design(side, role_i)``, and ``design`` carries the role
scale, so with ``w_i = shrink_i * role_scale_i`` the off-diagonal block of two
distinct same-team players is

    Corr[(i, s), (j, t)] = w_i w_j (A + B - Q)[s, t] = w_i w_j S[s, t]

and the opposing-team block is ``w_i w_j X[s, t]``. The per-player shrink is
the only thing that couples the lever to the pinning constraint, and it comes
straight out of the assembler, so nothing here re-derives it.

Given that, the count-space bucket the simulator would report is available in
closed form. Mehler's expansion makes one pair's implied count correlation

    sum_j d_j^a d_j^b (w_a w_b S)^(j + 1) / ((j + 1) sd_a sd_b)

so the pooled mean over ordered pairs factorises per team, order by order,
exactly the way ``factors.pair_moments`` factorises a second moment:
``sum_{i != k} u_i v_k = (sum u)(sum v) - sum u v``. So the pooled bucket is
computed over *every* pair in the held-out universe rather than a sample of
them, at the cost of one pass per Hermite order, and with no Monte Carlo
noise at all.

The predictor is only worth using if it reproduces the published runs, so
:func:`predict_buckets` is checked against both paired Monte Carlo runs in
the driver and in the test suite before any feasibility claim is read off it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from marginal import discrete_marginal, mehler_scores
from nba_prop_quant.research.game_latent_state.covariance import (
    SharedFactorLoadings,
    build_game_covariance,
    implied_within_player_correlation,
)
from nba_prop_quant.research.game_latent_state.factors import (
    incumbent_within_player_blocks,
)
from nba_prop_quant.research.game_latent_state.simulator import (
    GameRoster,
    tabulate_inverse_cdf,
)

from forensic_lib import (
    CROSS_PLAYER_BUCKETS,
    GAMES_PER_SEASON,
    MIN_EXPECTED_MINUTES,
    STATS,
)

#: Hermite orders kept in the transmission series. The order ``j`` term
#: carries ``rho ** (j + 1)``, and this layer's correlations are below 0.1, so
#: the first dropped term is under 1e-48 of the leading one.
DEFAULT_TERMS = 48


@dataclass
class HoldoutGame:
    """One held-out game's scored margins, roster and incumbent blocks."""

    game_id: int
    season: int
    player_ids: np.ndarray
    team_ids: np.ndarray
    roles: tuple[str | None, ...]
    #: ``(n_players, n_stats, terms)`` orthonormal-Hermite scores per space.
    count_scores: np.ndarray
    latent_scores: np.ndarray
    #: ``(n_players, n_stats)`` analytic standard deviations.
    analytic_sd: np.ndarray
    roster: GameRoster
    within_player: Mapping[int, np.ndarray]
    #: Filled by :func:`assemble`, which depends on the loadings.
    scale: np.ndarray = field(default_factory=lambda: np.empty(0))


def select_game_ids(
    residuals: pd.DataFrame,
    season: int,
    games_per_season: int = GAMES_PER_SEASON,
) -> list[int]:
    """The held-out game subsample, by the driver's own rule.

    ``04_validate_shadow_v1.py`` takes ``numpy.linspace(0, n - 1,
    games_per_season)`` rounded to an index over the sorted unique game ids of
    the season, so the subsample spans the calendar instead of clustering.
    Reproduced here so the forensic universe is the gate's universe.
    """
    season_rows = residuals.loc[residuals["season"] == int(season)]
    game_ids = sorted(int(value) for value in season_rows["game_id"].unique())
    if not games_per_season or len(game_ids) <= games_per_season:
        return game_ids
    picks = np.linspace(0, len(game_ids) - 1, games_per_season)
    return [game_ids[round(index)] for index in picks]


def build_holdout_games(
    residuals: pd.DataFrame,
    history: pd.DataFrame,
    season: int,
    marginals: Mapping[str, object],
    copula: object,
    stats: Sequence[str] = STATS,
    terms: int = DEFAULT_TERMS,
    games_per_season: int = GAMES_PER_SEASON,
    progress: object | None = None,
) -> list[HoldoutGame]:
    """Score every margin of one held-out season's simulated universe.

    Mirrors ``04_validate_shadow_v1.run_season``'s roster construction exactly:
    the expected-minutes floor, the two-team and six-player minima, the
    history join and the ``(team_id, player_id)`` sort order. A game the
    driver would have skipped is skipped here for the same reason, so the pair
    universe matches pair for pair.
    """
    season_rows = residuals.loc[residuals["season"] == int(season)]
    out: list[HoldoutGame] = []
    for position, game_id in enumerate(
        select_game_ids(residuals, season, games_per_season)
    ):
        observations = season_rows.loc[season_rows["game_id"] == game_id]
        observations = observations.loc[
            observations["expected_minutes"].fillna(0.0) >= MIN_EXPECTED_MINUTES
        ]
        if observations["team_id"].nunique() != 2 or len(observations) < 6:
            continue
        roster_frame = history.loc[
            (history["game_id"] == game_id)
            & history["player_id"].isin(observations["player_id"])
        ].copy()
        if len(roster_frame) != len(observations):
            continue
        home_team = int(
            observations.loc[observations["is_home"].astype(bool), "team_id"].iloc[0]
            if observations["is_home"].astype(bool).any()
            else observations["team_id"].iloc[0]
        )
        roster_frame = roster_frame.sort_values(
            ["team_id", "player_id"]
        ).reset_index(drop=True)
        roster = GameRoster(
            game_id=int(game_id),
            home_team_id=home_team,
            frame=roster_frame,
            stats=tuple(stats),
            role_column="role_bucket" if "role_bucket" in roster_frame else None,
        )

        size = len(roster_frame)
        count_scores = np.empty((size, len(stats), terms), dtype=float)
        latent_scores = np.empty((size, len(stats), terms), dtype=float)
        analytic_sd = np.empty((size, len(stats)), dtype=float)
        for player_index in range(size):
            row = roster_frame.iloc[player_index]
            for stat_index, stat in enumerate(stats):
                margin = discrete_marginal(
                    tabulate_inverse_cdf(
                        marginals[stat], float(row[f"mu_selected_{stat}"]), row
                    )
                )
                count_scores[player_index, stat_index] = mehler_scores(
                    margin, "count", terms
                )
                latent_scores[player_index, stat_index] = mehler_scores(
                    margin, "latent", terms
                )
                analytic_sd[player_index, stat_index] = margin.sd

        out.append(
            HoldoutGame(
                game_id=int(game_id),
                season=int(season),
                player_ids=roster_frame["player_id"].to_numpy(dtype=np.int64),
                team_ids=roster_frame["team_id"].to_numpy(dtype=np.int64),
                roles=tuple(
                    str(value) if pd.notna(value) else None
                    for value in roster_frame.get(
                        "role_bucket", pd.Series([None] * size)
                    )
                ),
                count_scores=count_scores,
                latent_scores=latent_scores,
                analytic_sd=analytic_sd,
                roster=roster,
                within_player=incumbent_within_player_blocks(
                    copula, stats, roster_frame["player_id"].astype(int)
                ),
            )
        )
        if progress is not None and position and position % 50 == 0:
            progress(f"  season {season}: {position} games scored")
    return out


@dataclass(frozen=True)
class AssemblyDiagnostics:
    """What the assembler reported across the whole held-out universe."""

    games: int
    min_covariance_eigenvalue: float
    min_residual_eigenvalue: float
    max_same_player_block_deviation: float
    min_shared_scale: float
    max_shared_scale: float
    numerical_failures: int


def assemble(
    games: Sequence[HoldoutGame],
    loadings: SharedFactorLoadings,
    check_pinning: bool = True,
) -> AssemblyDiagnostics:
    """Build every game covariance and record each player's realised scale.

    Writes ``w_i = shrink_i * role_scale_i`` onto each game, which is the only
    thing :func:`predict_buckets` needs from the assembler, and reports the
    PSD and pinning diagnostics the acceptance gates read. Pinning is verified
    against the assembled matrix rather than asserted, because the whole point
    of the lever is that it could in principle disturb it.
    """
    min_eigenvalue = float("inf")
    min_residual = float("inf")
    max_deviation = 0.0
    min_scale = float("inf")
    max_scale = 0.0
    failures = 0
    for game in games:
        try:
            covariance = build_game_covariance(
                game.roster.dimensions(), loadings, game.within_player
            )
        except (ValueError, KeyError, np.linalg.LinAlgError):
            failures += 1
            game.scale = np.zeros(game.player_ids.size, dtype=float)
            continue
        scale = np.empty(game.player_ids.size, dtype=float)
        for index, player_id in enumerate(game.player_ids):
            shrink = float(covariance.shared_shrink[int(player_id)])
            scale[index] = shrink * loadings.scale_for_role(game.roles[index])
            if check_pinning:
                induced = implied_within_player_correlation(
                    covariance, int(player_id)
                )
                max_deviation = max(
                    max_deviation,
                    float(
                        np.max(
                            np.abs(induced - game.within_player[int(player_id)])
                        )
                    ),
                )
        game.scale = scale
        min_eigenvalue = min(min_eigenvalue, float(covariance.min_eigenvalue))
        min_residual = min(min_residual, float(covariance.min_residual_eigenvalue))
        min_scale = min(min_scale, float(np.min(scale)))
        max_scale = max(max_scale, float(np.max(scale)))
    return AssemblyDiagnostics(
        games=len(games),
        min_covariance_eigenvalue=min_eigenvalue,
        min_residual_eigenvalue=min_residual,
        max_same_player_block_deviation=max_deviation,
        min_shared_scale=min_scale,
        max_shared_scale=max_scale,
        numerical_failures=failures,
    )


def _pooled_series(
    games: Sequence[HoldoutGame],
    scores: str,
    stat_a: int,
    stat_b: int,
    kind: str,
    normalise: bool,
    terms: int,
) -> np.ndarray:
    """``(1 / pairs) sum_pairs d_j^a d_j^b w_a^(j+1) w_b^(j+1) / denom``.

    One coefficient per Hermite order, pooled over every ordered pair of the
    bucket. The per-team factorisation is the same identity
    ``factors.pair_moments`` uses, so the pair weighting is identical to the
    observed statistic's.
    """
    order = np.arange(terms, dtype=float)
    total = np.zeros(terms, dtype=float)
    pairs = 0.0
    for game in games:
        if game.scale.size == 0 or not np.any(game.scale):
            continue
        matrix = getattr(game, scores)
        weight = game.scale[:, None] ** (order + 1.0)
        left = matrix[:, stat_a, :] * weight
        right = matrix[:, stat_b, :] * weight
        if normalise:
            left = left / game.analytic_sd[:, stat_a][:, None]
            right = right / game.analytic_sd[:, stat_b][:, None]
        if kind == "same_team":
            for team in np.unique(game.team_ids):
                members = game.team_ids == team
                if int(members.sum()) < 2:
                    continue
                first, second = left[members], right[members]
                total += first.sum(axis=0) * second.sum(axis=0) - np.sum(
                    first * second, axis=0
                )
                size = int(members.sum())
                pairs += float(size * (size - 1))
        elif kind == "cross_team":
            teams = np.unique(game.team_ids)
            if teams.size != 2:
                continue
            one, two = game.team_ids == teams[0], game.team_ids == teams[1]
            total += left[one].sum(axis=0) * right[two].sum(axis=0)
            total += left[two].sum(axis=0) * right[one].sum(axis=0)
            pairs += 2.0 * float(int(one.sum()) * int(two.sum()))
        else:
            raise ValueError(f"unsupported bucket kind {kind!r}")
    if pairs <= 0.0:
        raise ValueError("no pairs available in the held-out universe")
    return total / (pairs * (order + 1.0))


@dataclass(frozen=True)
class BucketSeries:
    """The transmission series of one bucket in one space.

    ``evaluate`` is the reading that space would show if the architecture's
    block entry for this bucket were ``value``; the per-player scales and the
    pair weighting are already baked into the coefficients.
    """

    bucket: str
    space: str
    coefficients: np.ndarray

    def evaluate(self, value: float) -> float:
        order = np.arange(self.coefficients.size, dtype=float)
        return float(self.coefficients @ (float(value) ** (order + 1.0)))

    def invert(self, target: float, bound: float = 0.60) -> float:
        low, high = -abs(bound), abs(bound)
        rising = self.evaluate(high) > self.evaluate(low)
        for _ in range(200):
            middle = 0.5 * (low + high)
            if (self.evaluate(middle) < target) == rising:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)


def bucket_series(
    games: Sequence[HoldoutGame],
    space: str,
    stats: Sequence[str] = STATS,
    terms: int = DEFAULT_TERMS,
) -> dict[str, BucketSeries]:
    """The transmission series of all twelve buckets in one space."""
    index = {stat: position for position, stat in enumerate(stats)}
    scores = "count_scores" if space == "count" else "latent_scores"
    out: dict[str, BucketSeries] = {}
    for name, kind, (first, second) in CROSS_PLAYER_BUCKETS:
        out[name] = BucketSeries(
            bucket=name,
            space=space,
            coefficients=_pooled_series(
                games,
                scores,
                index[first],
                index[second],
                kind,
                normalise=space == "count",
                terms=terms,
            ),
        )
    return out


def predict_buckets(
    series: Mapping[str, BucketSeries],
    loadings: SharedFactorLoadings,
) -> dict[str, float]:
    """The reading each bucket would show for these loadings."""
    index = {stat: position for position, stat in enumerate(loadings.stats)}
    same = loadings.same_team_correlation()
    cross = loadings.cross_team_correlation()
    return {
        name: series[name].evaluate(
            float(
                (same if kind == "same_team" else cross)[
                    index[first], index[second]
                ]
            )
        )
        for name, kind, (first, second) in CROSS_PLAYER_BUCKETS
    }
