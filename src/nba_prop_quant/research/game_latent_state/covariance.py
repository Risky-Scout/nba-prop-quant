"""Hierarchical latent-state covariance for one game, PSD by construction.

SHADOW / RESEARCH ONLY.

Contract
--------
For a game with player-stat dimensions ``(i, s)``, the shadow layer builds a
latent Gaussian correlation matrix ``Sigma`` with two properties:

1.  **Within-player block is pinned.**  ``Sigma[(i, s), (i, t)]`` equals the
    incumbent ``GaussianCopula`` correlation ``R_i[s, t]`` exactly. The new
    layer therefore induces *identically* the incumbent same-player copula on
    every single-player margin, so same-player dependence cannot be applied
    twice. This is integration strategy **B** from the brief: the existing
    same-player block is preserved and the new factors are added around it.

2.  **Cross-player blocks come only from shared factors.**
    ``Sigma[(i, s), (j, t)] = c_i c_j * (L_i L_j^T)[s, t]`` for ``i != j``.

The algebra that makes both hold at once::

    Sigma = (C L)(C L)^T + blockdiag_i( R_i - c_i^2 L_i L_i^T )

``L_i L_i^T`` is the shared-factor contribution to player ``i``'s own block,
so subtracting it from ``R_i`` in the block-diagonal term leaves the within-
player block at exactly ``R_i``. ``Sigma`` is a sum of a Gram matrix and a
block-diagonal matrix, hence PSD whenever every residual block
``D_i = R_i - c_i^2 L_i L_i^T`` is PSD; ``c_i`` is the largest scalar in
``(0, 1]`` for which that holds, so PSD is guaranteed by construction rather
than by a post-hoc eigenvalue repair. Because ``diag(R_i) = 1`` and
``diag(D_i) = 1 - c_i^2 diag(L_i L_i^T)``, the unit diagonal is exact and no
renormalisation is needed.

Identification
--------------
Let ``S`` be the stat-by-stat correlation between two *distinct* players on
the same team and ``X`` the same between two players on opposite teams. With
the game-level Gram ``A``, the team-contrast Gram ``B`` and the zero-sum
competition Gram ``Q``, the model implies

    S = A + B - Q,    X = A - B

so the two observable blocks pin down two combinations of three Grams:

    A + B = S + Q,    A - B = X

A team-state factor that both teams load on *identically* is
indistinguishable from a game-level factor, so only the antisymmetric part of
the own-team/opponent-team loadings is separately identified; its symmetric
part is absorbed into ``A``. ``Q`` is separately identified only through the
constraint that ``A``, ``B`` and ``Q`` are each PSD: a purely additive model
forces ``S`` to be PSD, so an indefinite ``S`` is the *only* evidence that
``Q`` is non-zero, and ``factors.competition_gate`` requires that evidence to
survive a bias-corrected bootstrap before the family is admitted. The
canonical parameterisation is therefore

* ``K`` game-level factors with loadings ``gamma[s, k]`` from a rank-``K``
  PSD factorisation of ``A`` (factor 1 is the pace/volume factor, factor 2
  the rebound-environment contrast, fixed by eigenvalue ordering),
* one signed team-contrast factor with loading ``+d[s]`` for the player's own
  team and ``-d[s]`` for the opponent, from the rank-1 PSD factorisation of
  ``B``, and
* the within-team zero-sum family ``Q``, applied through the team projection
  rather than as a per-player loading.

The shared-factor contribution to a player's *own* block is then
``A + B + (n - 1) Q``, not ``S``: a player is on the same team as himself, so
he receives the additive factors plus his own share of the zero-sum term
rather than the negative share a distinct teammate receives. When ``Q = 0``
this reduces to ``L_i L_i^T = A + B = S``.

No parameter in ``L`` or ``Q`` is indexed by a player or a player pair, so the
layer extends to unseen players and new roster combinations without refitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# A residual block is accepted as PSD when its smallest eigenvalue is at or
# above this floor. Tighter than float64 noise on a 6x6 correlation matrix,
# loose enough that an exactly-singular block is not rejected.
PSD_EIGENVALUE_FLOOR = -1e-10

# Lower bound for the shared-loading shrink search. A game whose incumbent
# within-player block cannot absorb even this much shared structure is
# reported rather than silently simulated.
MIN_SHARED_SHRINK = 1e-6


def min_eigenvalue(matrix: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))))


def project_psd_rank(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Best rank-``rank`` PSD approximation and its loading factor.

    Returns ``(approximation, loadings)`` with
    ``approximation == loadings @ loadings.T``. Negative eigenvalues are
    dropped, which is the projection onto the PSD cone, not a clip of the
    matrix entries.
    """
    if rank < 0:
        raise ValueError("rank must be non-negative")

    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    keep = min(rank, symmetric.shape[0])
    selected = np.clip(eigenvalues[:keep], 0.0, None)
    loadings = eigenvectors[:, :keep] * np.sqrt(selected)
    return loadings @ loadings.T, loadings


@dataclass(frozen=True)
class SharedFactorLoadings:
    """Per-stat loadings on the identified shared factors.

    ``game`` has shape ``(n_stats, k_game)`` and ``competition`` shape
    ``(n_stats, r_comp)``. ``team_contrast`` is either a single signed factor
    of shape ``(n_stats,)`` or ``(n_stats, r_contrast)`` signed factors; the
    1-D form is canonical for ``r_contrast == 1`` so that existing rank-1
    artifacts round-trip byte-for-byte. Nothing in the identification argument
    pins ``B`` to rank 1 -- only the antisymmetric part ``A - B`` is
    identified, and ``B`` is a full PSD Gram in general -- so the rank is a
    parsimony parameter rather than a structural constraint.
    ``role_scale`` maps a role label to a multiplicative scalar applied to
    every shared loading of a player in that role; a role the fit never saw
    falls back to ``1.0``, which is the pooled estimate.

    The competition family is the within-team zero-sum factor. A purely
    additive factor model can only produce non-negative same-team same-stat
    correlation, which is the wrong sign if teammates compete for a finite
    resource (rebounds to collect, shots to take, assists to distribute).
    Writing the within-team effect as a zero-sum allocation over the ``n``
    usable players on that team,

        Cov_shared[(i, s), (j, t)] += n * Q[s, t] * (delta_ij - 1 / n)

    gives ``-Q[s, t]`` between distinct teammates and ``(n - 1) * Q[s, t]`` on
    a player's own block. That term is ``kron(I - 11'/n, n * Q)``, a Kronecker
    product of two PSD matrices, so the shared covariance stays PSD while
    negative teammate correlation becomes representable. The ``n`` scaling
    makes the pairwise effect independent of roster size, so a short rotation
    and a deep one carry the same teammate correlation.
    """

    stats: tuple[str, ...]
    game: np.ndarray
    team_contrast: np.ndarray
    competition: np.ndarray | None = None
    role_scale: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.game.ndim != 2 or self.game.shape[0] != len(self.stats):
            raise ValueError("game loadings must have shape (n_stats, k_game)")
        if self.team_contrast.shape not in {
            (len(self.stats),),
            *((len(self.stats), rank) for rank in range(1, len(self.stats) + 1)),
        }:
            raise ValueError(
                "team_contrast must have shape (n_stats,) or (n_stats, r_contrast)"
            )
        if self.competition is not None and (
            self.competition.ndim != 2 or self.competition.shape[0] != len(self.stats)
        ):
            raise ValueError("competition loadings must have shape (n_stats, r_comp)")

    @property
    def k_game(self) -> int:
        return int(self.game.shape[1])

    @property
    def r_competition(self) -> int:
        return 0 if self.competition is None else int(self.competition.shape[1])

    @property
    def contrast_matrix(self) -> np.ndarray:
        """``team_contrast`` as a ``(n_stats, r_contrast)`` matrix."""
        if self.team_contrast.ndim == 1:
            return self.team_contrast.reshape(-1, 1)
        return self.team_contrast

    @property
    def r_contrast(self) -> int:
        return int(self.contrast_matrix.shape[1])

    def competition_gram(self) -> np.ndarray:
        """``Q``: the within-team zero-sum Gram."""
        if self.competition is None:
            return np.zeros((len(self.stats), len(self.stats)), dtype=float)
        return self.competition @ self.competition.T

    def additive_gram(self) -> np.ndarray:
        """``A + B``: the team-blind plus team-contrast Gram."""
        contrast = self.contrast_matrix
        return self.game @ self.game.T + contrast @ contrast.T

    def same_team_correlation(self) -> np.ndarray:
        """``S = A + B - Q``: distinct players, same team."""
        return self.additive_gram() - self.competition_gram()

    def cross_team_correlation(self) -> np.ndarray:
        """``X = A - B``: distinct players, opposite teams."""
        contrast = self.contrast_matrix
        return self.game @ self.game.T - contrast @ contrast.T

    def within_player_shared_gram(self, team_size: int) -> np.ndarray:
        """``A + B + (n - 1) Q``: shared contribution to a player's own block."""
        return self.additive_gram() + max(int(team_size) - 1, 0) * self.competition_gram()

    def design(self, side: int, role: str | None = None) -> np.ndarray:
        """Additive shared-factor design rows for one player.

        Shape ``(n_stats, k_game + r_contrast)``. ``side`` is ``+1`` for one
        team and ``-1`` for the other; it flips the sign of every team-contrast
        loading and nothing else. The competition family is absent here because
        it is not an independent per-player factor: it enters through the team
        projection in :func:`build_game_covariance`.
        """
        if side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        scale = self.scale_for_role(role)
        contrast = (side * scale) * self.contrast_matrix
        return np.hstack([scale * self.game, contrast])

    def scale_for_role(self, role: str | None) -> float:
        if role is None:
            return 1.0
        return float(self.role_scale.get(role, 1.0))

    def to_payload(self) -> dict[str, object]:
        # No ``r_contrast`` key: the contrast rank is recoverable from the
        # shape of ``team_contrast_loadings``, and leaving it out keeps this
        # payload byte-identical to the accepted shadow V1 artifact.
        return {
            "stats": list(self.stats),
            "k_game": self.k_game,
            "r_competition": self.r_competition,
            "game_loadings": self.game.tolist(),
            "team_contrast_loadings": self.team_contrast.tolist(),
            "competition_loadings": (
                None if self.competition is None else self.competition.tolist()
            ),
            "role_scale": {str(key): float(value) for key, value in self.role_scale.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SharedFactorLoadings:
        competition = payload.get("competition_loadings")
        return cls(
            stats=tuple(payload["stats"]),  # type: ignore[arg-type]
            game=np.asarray(payload["game_loadings"], dtype=float),
            team_contrast=np.asarray(payload["team_contrast_loadings"], dtype=float),
            competition=(
                None if competition is None else np.asarray(competition, dtype=float)
            ),
            role_scale={
                str(key): float(value)
                for key, value in dict(payload.get("role_scale", {})).items()  # type: ignore[arg-type]
            },
        )

    @classmethod
    def independent(cls, stats: Sequence[str], k_game: int = 2) -> SharedFactorLoadings:
        """Zero shared structure: the conditional-independence baseline.

        Within-player blocks still come from the incumbent copula, so this is
        BASELINE 1 (cross-player independence, same-player dependence
        preserved) rather than full independence.
        """
        stats = tuple(stats)
        return cls(
            stats=stats,
            game=np.zeros((len(stats), k_game), dtype=float),
            team_contrast=np.zeros(len(stats), dtype=float),
        )


@dataclass(frozen=True)
class GameDimension:
    """One ``(player, stat)`` coordinate of a game's latent vector."""

    player_id: int
    team_id: int
    stat: str
    side: int
    role: str | None = None


@dataclass(frozen=True)
class GameCovariance:
    dimensions: tuple[GameDimension, ...]
    correlation: np.ndarray
    cholesky: np.ndarray
    shared_shrink: Mapping[int, float]
    min_residual_eigenvalue: float
    min_eigenvalue: float

    @property
    def size(self) -> int:
        return len(self.dimensions)

    @property
    def shrink_applied(self) -> bool:
        return any(value < 1.0 - 1e-12 for value in self.shared_shrink.values())


def _largest_feasible_shrink(
    within: np.ndarray,
    shared_gram: np.ndarray,
    floor: float = PSD_EIGENVALUE_FLOOR,
) -> float:
    """Largest ``c`` in (0, 1] with ``within - c^2 * shared_gram`` PSD.

    ``min_eig(within - t * shared_gram)`` is concave and non-increasing in
    ``t = c^2``, so a bisection on ``t`` is exact up to its tolerance.
    """
    if min_eigenvalue(within - shared_gram) >= floor:
        return 1.0

    low, high = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if min_eigenvalue(within - mid * shared_gram) >= floor:
            low = mid
        else:
            high = mid
    return float(np.sqrt(low))


def build_game_covariance(
    dimensions: Sequence[GameDimension],
    loadings: SharedFactorLoadings,
    within_player: Mapping[int, np.ndarray],
) -> GameCovariance:
    """Assemble the PSD latent correlation matrix for one game.

    ``within_player`` maps ``player_id`` to that player's incumbent
    stat-by-stat correlation matrix, ordered like ``loadings.stats``. A player
    the incumbent copula never fitted individually must still be present here,
    carrying the incumbent's global fallback correlation.
    """
    dimensions = tuple(dimensions)
    if not dimensions:
        raise ValueError("a game needs at least one player-stat dimension")

    stat_index = {stat: position for position, stat in enumerate(loadings.stats)}
    unknown = sorted({dim.stat for dim in dimensions} - set(stat_index))
    if unknown:
        raise ValueError(f"dimensions reference unmodelled stats: {unknown}")

    # Group coordinates by player so the within-player block can be pinned,
    # and by team so the zero-sum competition projection knows its roster size.
    by_player: dict[int, list[int]] = {}
    team_of_player: dict[int, int] = {}
    for position, dim in enumerate(dimensions):
        by_player.setdefault(dim.player_id, []).append(position)
        team_of_player[dim.player_id] = dim.team_id

    team_sizes: dict[int, int] = {}
    for player_id, team_id in team_of_player.items():
        team_sizes[team_id] = team_sizes.get(team_id, 0) + 1

    size = len(dimensions)
    shared = np.zeros((size, loadings.k_game + loadings.r_contrast), dtype=float)
    competition_gram = loadings.competition_gram()
    has_competition = bool(np.any(competition_gram))
    row_scale = np.ones(size, dtype=float)
    shrink: dict[int, float] = {}
    min_residual = np.inf

    residual_blocks: list[tuple[list[int], np.ndarray]] = []
    player_rows: dict[int, list[int]] = {}

    for player_id, positions in by_player.items():
        if player_id not in within_player:
            raise KeyError(
                f"no incumbent within-player correlation supplied for player {player_id}"
            )

        block = np.asarray(within_player[player_id], dtype=float)
        if block.shape != (len(loadings.stats), len(loadings.stats)):
            raise ValueError(
                f"within-player block for player {player_id} has shape "
                f"{block.shape}, expected "
                f"{(len(loadings.stats), len(loadings.stats))}"
            )

        sides = {dimensions[position].side for position in positions}
        if len(sides) != 1:
            raise ValueError(f"player {player_id} appears on both teams")
        side = sides.pop()

        roles = {dimensions[position].role for position in positions}
        role = roles.pop() if len(roles) == 1 else None

        rows = [stat_index[dimensions[position].stat] for position in positions]
        player_rows[player_id] = rows
        design = loadings.design(side=side, role=role)
        role_factor = loadings.scale_for_role(role)
        team_size = team_sizes[team_of_player[player_id]]

        # The shared contribution to this player's own block: the additive
        # factors plus the zero-sum term's diagonal share.
        full_gram = design @ design.T + max(team_size - 1, 0) * (
            role_factor**2
        ) * competition_gram

        scale = _largest_feasible_shrink(block, full_gram)
        if scale < MIN_SHARED_SHRINK:
            raise ValueError(
                f"player {player_id} cannot absorb any shared structure; the "
                "incumbent within-player block is numerically degenerate"
            )
        shrink[int(player_id)] = scale

        shared[positions, :] = (scale * design)[rows, :]
        for position in positions:
            row_scale[position] = scale * role_factor

        residual = block - (scale**2) * full_gram
        min_residual = min(min_residual, min_eigenvalue(residual))
        residual_blocks.append((positions, residual[np.ix_(rows, rows)]))

    correlation = shared @ shared.T

    if has_competition:
        # kron(I - 11'/n, n * Q) per team, then a congruence by the per-player
        # shrink and role scales. Both factors are PSD, so the sum stays PSD.
        for team_id, team_size in team_sizes.items():
            members = [
                player_id
                for player_id, member_team in team_of_player.items()
                if member_team == team_id
            ]
            for first in members:
                for second in members:
                    positions_a = by_player[first]
                    positions_b = by_player[second]
                    rows_a = player_rows[first]
                    rows_b = player_rows[second]
                    weight = team_size * (1.0 if first == second else 0.0) - 1.0
                    patch = weight * competition_gram[np.ix_(rows_a, rows_b)]
                    patch = patch * np.outer(
                        row_scale[positions_a], row_scale[positions_b]
                    )
                    correlation[np.ix_(positions_a, positions_b)] += patch

    for positions, residual in residual_blocks:
        correlation[np.ix_(positions, positions)] += residual

    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)

    observed_min = min_eigenvalue(correlation)
    # A Gram plus PSD-block-diagonal sum is PSD exactly; the jitter below only
    # absorbs float64 rounding so Cholesky cannot fail on a valid matrix.
    jitter = 0.0
    if observed_min <= 0.0:
        jitter = abs(observed_min) + 1e-12
    factor = np.linalg.cholesky(correlation + jitter * np.eye(size))

    return GameCovariance(
        dimensions=dimensions,
        correlation=correlation,
        cholesky=factor,
        shared_shrink=shrink,
        min_residual_eigenvalue=float(min_residual),
        min_eigenvalue=float(observed_min),
    )


def implied_within_player_correlation(
    covariance: GameCovariance,
    player_id: int,
) -> np.ndarray:
    """Extract the simulated within-player block for a no-double-count check."""
    positions = [
        position
        for position, dim in enumerate(covariance.dimensions)
        if dim.player_id == player_id
    ]
    if not positions:
        raise KeyError(f"player {player_id} is not in this game")
    return covariance.correlation[np.ix_(positions, positions)]
