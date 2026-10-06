"""Joint same-game prop query engine over shadow simulations.

SHADOW / RESEARCH ONLY. No sportsbook pricing, no vig, no WizardOfOdds
publishing: this returns probabilities and their Monte Carlo uncertainty and
nothing else.

A query is a conjunction of legs::

    PropLeg("A", "pts", "over",  24.5)
    PropLeg("A", "ast", "over",   6.5)
    PropLeg("B", "reb", "over",   8.5)
    PropLeg("C", "pts", "under", 19.5)

and the probability is the fraction of simulated whole games in which every
leg holds. Because the legs are evaluated on the *same* draw index, the
answer carries the full joint dependence structure instead of a product of
marginals.

Push handling follows the production convention in
``FittedMarginal.over_under_push``: on an integer line, ``value == line`` is a
push and satisfies neither over nor under. The engine reports the push-
affected fraction so an integer-line conjunction is never silently graded as
a loss.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import DEPENDENCE_MODEL_VERSION
from .simulator import GameSimulation

Side = Literal["over", "under"]

# Combination props the production marginals already support, expressed as
# sums of simulated component stats within one player.
COMBO_COMPONENTS: dict[str, tuple[str, ...]] = {
    "points_rebounds": ("pts", "reb"),
    "points_assists": ("pts", "ast"),
    "rebounds_assists": ("reb", "ast"),
    "points_rebounds_assists": ("pts", "reb", "ast"),
    "stocks": ("stl", "blk"),
}


@dataclass(frozen=True)
class PropLeg:
    player_id: int
    stat: str
    side: Side
    line: float

    def __post_init__(self) -> None:
        if self.side not in ("over", "under"):
            raise ValueError("side must be 'over' or 'under'")

    def describe(self) -> str:
        symbol = ">" if self.side == "over" else "<"
        return f"player {self.player_id} {self.stat} {symbol} {self.line}"


@dataclass(frozen=True)
class JointQueryResult:
    probability: float
    standard_error: float
    simulations: int
    satisfying_draws: int
    push_fraction: float
    seed: int
    game_id: int
    legs: tuple[str, ...]
    leg_marginal_probabilities: tuple[float, ...]
    independent_product: float
    dependence_model_version: str
    artifact_version: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "monte_carlo_standard_error": self.standard_error,
            "simulations": self.simulations,
            "satisfying_draws": self.satisfying_draws,
            "push_fraction": self.push_fraction,
            "seed": self.seed,
            "game_id": self.game_id,
            "legs": list(self.legs),
            "leg_marginal_probabilities": list(self.leg_marginal_probabilities),
            "independent_product": self.independent_product,
            "dependence_model_version": self.dependence_model_version,
            "artifact_version": self.artifact_version,
        }


def _leg_values(simulation: GameSimulation, leg: PropLeg) -> np.ndarray:
    if leg.stat in COMBO_COMPONENTS:
        return simulation.combo(leg.player_id, COMBO_COMPONENTS[leg.stat])
    return simulation.values(leg.player_id, leg.stat)


def leg_mask(simulation: GameSimulation, leg: PropLeg) -> np.ndarray:
    values = _leg_values(simulation, leg)
    if leg.side == "over":
        return values > leg.line
    return values < leg.line


def push_mask(simulation: GameSimulation, leg: PropLeg) -> np.ndarray:
    if not float(leg.line).is_integer():
        return np.zeros(simulation.simulations, dtype=bool)
    return _leg_values(simulation, leg) == float(leg.line)


def evaluate_joint(
    simulation: GameSimulation,
    legs: Sequence[PropLeg],
    artifact_version: str | None = None,
) -> JointQueryResult:
    """Probability that every leg holds in the same simulated game."""
    if not legs:
        raise ValueError("a joint query needs at least one leg")

    satisfied = np.ones(simulation.simulations, dtype=bool)
    pushed = np.zeros(simulation.simulations, dtype=bool)
    marginals: list[float] = []

    for leg in legs:
        mask = leg_mask(simulation, leg)
        marginals.append(float(np.mean(mask)))
        satisfied &= mask
        pushed |= push_mask(simulation, leg)

    count = int(np.sum(satisfied))
    probability = count / simulation.simulations
    standard_error = float(
        np.sqrt(max(probability * (1.0 - probability), 0.0) / simulation.simulations)
    )

    return JointQueryResult(
        probability=float(probability),
        standard_error=standard_error,
        simulations=int(simulation.simulations),
        satisfying_draws=count,
        push_fraction=float(np.mean(pushed)),
        seed=int(simulation.seed),
        game_id=int(simulation.game_id),
        legs=tuple(leg.describe() for leg in legs),
        leg_marginal_probabilities=tuple(marginals),
        independent_product=float(np.prod(marginals)),
        dependence_model_version=DEPENDENCE_MODEL_VERSION,
        artifact_version=artifact_version,
    )
