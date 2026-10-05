"""Whole-game Monte Carlo simulator over the shadow latent-state layer.

SHADOW / RESEARCH ONLY.

The simulator owns no distributional knowledge. It samples a coherent latent
residual vector for a whole game, maps it to uniforms with ``Phi``, and pushes
those uniforms through the **existing production marginal inverse CDFs**
(``nba_prop_quant.distributions.FittedMarginal.ppf``). Every univariate margin
it produces is therefore the production margin by construction; only the joint
coupling is new.

Stat coverage is whatever the production marginals supply, which is
``pts, reb, ast, stl, blk, fg3m``. TOV is deliberately absent: production fits
no turnover marginal, and the brief forbids inventing support the current
model does not provide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from ...distributions import FittedMarginal
from .covariance import (
    GameCovariance,
    GameDimension,
    SharedFactorLoadings,
    build_game_covariance,
)

# The simulator's stat order. Fixed so a seed reproduces a draw exactly.
SUPPORTED_STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")

# Uniform guard before the inverse CDF. The production calibrators clip at
# 1e-10 internally; matching it here keeps the mapping identical to the
# incumbent path rather than introducing a second, looser bound.
UNIFORM_EPSILON = 1e-10

# Tail mass left outside the tabulated inverse-CDF grid. Smaller than the
# uniform guard above, so a clipped uniform can never fall past the grid.
GRID_TAIL_MASS = 1e-12

# Hard ceiling on the tabulated support. No NBA box-score count approaches
# it; it exists so a pathological mu cannot allocate without bound.
GRID_MAX_SUPPORT = 400


def tabulate_inverse_cdf(
    marginal: FittedMarginal,
    mu: float,
    row: pd.Series,
    tail_mass: float = GRID_TAIL_MASS,
) -> np.ndarray:
    """Tabulate ``F(0..K)`` for one production marginal at one ``mu``.

    The production ``ppf`` is the generalized inverse of a discrete CDF,
    ``ppf(u) = min{k : F(k) >= u}``, so ``np.searchsorted(F, u, side="left")``
    returns the identical integer. Tabulating once per player-stat and then
    searching is what makes whole-game Monte Carlo at validation scale
    affordable; ``tests/test_game_latent_state_shadow.py`` asserts exact
    agreement with the production ``ppf`` over a dense uniform grid, so this
    is an evaluation-order change and not a distributional one.
    """
    frame = row.to_frame().T
    support = max(int(np.ceil(mu * 3.0 + 40.0)), 40)
    while True:
        support = min(support, GRID_MAX_SUPPORT)
        grid = np.arange(support + 1)
        cdf = np.asarray(
            marginal.cdf(grid, np.full(grid.shape, float(mu)), frame), dtype=float
        )
        cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
        if cdf[-1] >= 1.0 - tail_mass or support >= GRID_MAX_SUPPORT:
            return cdf
        support *= 2


def grid_ppf(cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``min{k : F(k) >= u}`` for a tabulated CDF."""
    return np.searchsorted(cdf, np.asarray(u, dtype=float), side="left").astype(float)


@dataclass(frozen=True)
class GameRoster:
    """One game's production inputs for the shadow simulator.

    ``frame`` holds one row per active player with the production feature
    columns the marginals need (the ZINB inflation features), the selected
    conditional means in ``mu_selected_{stat}``, plus ``player_id`` and
    ``team_id``. ``home_team_id`` fixes the sign of the team-contrast factor
    deterministically so a reordering of the roster cannot change a draw.
    """

    game_id: int
    home_team_id: int
    frame: pd.DataFrame
    stats: tuple[str, ...] = SUPPORTED_STATS
    role_column: str | None = None

    def __post_init__(self) -> None:
        required = {"player_id", "team_id", *[f"mu_selected_{s}" for s in self.stats]}
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise KeyError(f"roster frame is missing {missing}")
        if self.frame["player_id"].duplicated().any():
            raise ValueError("roster frame has duplicate player_id rows")
        if self.frame["team_id"].nunique() > 2:
            raise ValueError("a game has at most two teams")

    def dimensions(self) -> tuple[GameDimension, ...]:
        dims: list[GameDimension] = []
        for _, row in self.frame.iterrows():
            side = 1 if int(row["team_id"]) == int(self.home_team_id) else -1
            role = (
                str(row[self.role_column])
                if self.role_column is not None and pd.notna(row.get(self.role_column))
                else None
            )
            for stat in self.stats:
                dims.append(
                    GameDimension(
                        player_id=int(row["player_id"]),
                        team_id=int(row["team_id"]),
                        stat=stat,
                        side=side,
                        role=role,
                    )
                )
        return tuple(dims)


@dataclass(frozen=True)
class GameSimulation:
    game_id: int
    stats: tuple[str, ...]
    player_ids: tuple[int, ...]
    team_ids: tuple[int, ...]
    #: shape (simulations, n_players, n_stats)
    draws: np.ndarray
    simulations: int
    seed: int
    #: ``None`` for baselines that do not build a joint game covariance, such
    #: as the incumbent per-player copula path. The query engine only counts
    #: draws, so it does not need one.
    covariance: GameCovariance | None = None

    def values(self, player_id: int, stat: str) -> np.ndarray:
        try:
            player_index = self.player_ids.index(int(player_id))
        except ValueError as error:
            raise KeyError(f"player {player_id} is not in game {self.game_id}") from error
        if stat not in self.stats:
            raise KeyError(f"stat {stat!r} is not simulated")
        return self.draws[:, player_index, self.stats.index(stat)]

    def combo(self, player_id: int, components: Sequence[str]) -> np.ndarray:
        total = np.zeros(self.simulations, dtype=float)
        for component in components:
            total = total + self.values(player_id, component)
        return total

    def to_long_frame(self) -> pd.DataFrame:
        records = []
        for player_index, player_id in enumerate(self.player_ids):
            for stat_index, stat in enumerate(self.stats):
                records.append(
                    pd.DataFrame(
                        {
                            "draw": np.arange(self.simulations),
                            "game_id": self.game_id,
                            "player_id": player_id,
                            "team_id": self.team_ids[player_index],
                            "stat": stat,
                            "value": self.draws[:, player_index, stat_index],
                        }
                    )
                )
        return pd.concat(records, ignore_index=True)


def simulate_game(
    roster: GameRoster,
    marginals: Mapping[str, FittedMarginal],
    loadings: SharedFactorLoadings,
    within_player: Mapping[int, np.ndarray],
    simulations: int = 20_000,
    seed: int = 73,
    tabulated_inverse_cdf: bool = True,
) -> GameSimulation:
    """Draw ``simulations`` coherent realizations of one whole game.

    Deterministic in ``seed``: the latent normals come from a single
    ``default_rng(seed)`` stream consumed in the fixed dimension order, and
    the dimension order is fixed by the roster frame's row order and
    ``roster.stats``.

    ``tabulated_inverse_cdf`` selects the tabulated inverse CDF rather than
    calling the production ``ppf`` once per draw. Both routes return the same
    integers; the tabulated one is what makes season-scale validation
    affordable.
    """
    if simulations < 1:
        raise ValueError("simulations must be positive")

    missing = sorted(set(roster.stats) - set(marginals))
    if missing:
        raise KeyError(f"no production marginal supplied for {missing}")

    dimensions = roster.dimensions()
    covariance = build_game_covariance(
        dimensions=dimensions,
        loadings=loadings,
        within_player=within_player,
    )

    rng = np.random.default_rng(seed)
    standard = rng.standard_normal(size=(simulations, covariance.size))
    latent = standard @ covariance.cholesky.T
    uniforms = np.clip(norm.cdf(latent), UNIFORM_EPSILON, 1.0 - UNIFORM_EPSILON)

    n_players = len(roster.frame)
    n_stats = len(roster.stats)
    draws = np.empty((simulations, n_players, n_stats), dtype=float)

    rows = [row for _, row in roster.frame.iterrows()]
    for player_index, row in enumerate(rows):
        for stat_index, stat in enumerate(roster.stats):
            column = player_index * n_stats + stat_index
            mu = float(row[f"mu_selected_{stat}"])
            if tabulated_inverse_cdf:
                cdf = tabulate_inverse_cdf(marginals[stat], mu, row)
                draws[:, player_index, stat_index] = grid_ppf(cdf, uniforms[:, column])
            else:
                draws[:, player_index, stat_index] = marginals[stat].ppf(
                    uniforms[:, column], mu, row
                )

    return GameSimulation(
        game_id=int(roster.game_id),
        stats=tuple(roster.stats),
        player_ids=tuple(int(row["player_id"]) for row in rows),
        team_ids=tuple(int(row["team_id"]) for row in rows),
        draws=draws,
        simulations=int(simulations),
        seed=int(seed),
        covariance=covariance,
    )
