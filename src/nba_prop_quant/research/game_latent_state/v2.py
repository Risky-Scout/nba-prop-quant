"""Shadow V2 structural refinement of the shared latent-factor fit.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The accepted bucket repair is the control. It repaired the two weak same-team
buckets but did so with a *global* change of shrinkage estimator, which moved
the cross-team block as a side effect, and it still misses
``passer_ast_teammate_pts`` in count space by roughly 0.0213. This module
isolates four structural changes so each can be screened and accepted on its
own evidence, and composes them without a single player- or pair-indexed
parameter.

1.  **Decoupled blocks.** ``shrink_blocks`` applies one estimator to both the
    same-team and cross-team pooled moments, so a same-team hyperparameter
    reaches the opponent buckets. :class:`V2Spec` gives the two blocks
    separate shrinkage families, and the *base* construction -- the one that
    produces ``A``, ``B`` and ``Q`` and therefore the fitted cross-team block
    -- always reads the V1 family. Everything the repair adds to the same-team
    block then arrives through the symmetric subspace, which is algebraically
    absent from ``X = A - B``. The cross-team block is consequently a function
    of the cross-team estimator alone.

2.  **The symmetric subspace carries the same-team repair.** The gap between
    the final same-team target and what the base construction represents is
    halved, projected onto the PSD cone at rank ``r_symmetric``, and appended
    to *both* loading blocks as ``U``. That lifts the same-team block by
    exactly ``2 U U'`` and leaves the cross-team block bit-for-bit unchanged,
    while ``A + U U'`` and ``B + U U'`` stay Grams, so PSD still holds by
    construction with no eigenvalue repair anywhere.

3.  **Role-conditioned deviations at the loading level.** ``U_i = U + h_r W``
    with ``h`` a weighted-centred per-role score and ``W`` a single loading
    block. The role layer is not a table of role-pair correlations: it is
    ``len(roles) - 1`` free scores plus one ``n_stats`` by ``r_symmetric``
    loading, shrunk toward the pooled fit by the role's own game count, and a
    role the fit never saw resolves to ``h = 0``, i.e. exactly the pooled
    loading.

4.  **A bridge-adjusted same-team target.** The same-team target can be
    blended toward the latent correlation that the *observed count-space*
    correlation implies, via the deterministic discrete-copula bridge in
    :mod:`bridge`. The blend weight is one pre-registered scalar for the whole
    fit, selected on pre-2024 folds.

Reproducing the controls
------------------------
:data:`V1_BASE_SPEC` reproduces ``factors.fit_shared_factors`` and
:data:`REPAIR_CONTROL_SPEC` reproduces ``repair.fit_repaired_factors`` with
the accepted repair specification, both exactly. That is what makes this an
additive refinement rather than a re-derivation: the controls are not
recomputed from a different code path, they are recomputed from *this* one and
asserted equal to the original.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .covariance import SharedFactorLoadings, project_psd_rank
from .factors import (
    DEFAULT_K_GAME,
    DEFAULT_SHRINK_Z,
    ROLE_POOLING_GAMES,
    FactorFit,
    PairMoments,
    competition_gate,
    dominating_additive_gram,
    fit_role_scales,
    pair_moments,
    soft_threshold,
)
from .repair import (
    DEFAULT_R_CONTRAST,
    EB_FAMILY_BLOCK_DIAGONAL,
    SHRINKAGE_EMPIRICAL_BAYES,
    SHRINKAGE_SOFT_THRESHOLD,
    empirical_bayes_shrink,
)
from .temporal import TREATMENT_POOLED

#: Ridge penalty on the role-deviation loading, in squared correlation units
#: per squared loading unit. A declared regulariser, not a searched one: it is
#: small enough to leave a well-supported role cell essentially unpenalised
#: and large enough that a role pair with almost no pairs cannot swing ``W``.
ROLE_DEVIATION_RIDGE = 1e-4

#: Minimum same-team ordered pairs a role cell needs before it is fitted or
#: reported as an estimate rather than as shrunk-to-pooled. One season of a
#: well-populated role pair is around 100k pairs, so this admits only cells
#: with a genuinely measurable moment.
MIN_ROLE_PAIRS = 20000.0

#: Label a role cell carries when its support is below :data:`MIN_ROLE_PAIRS`.
ROLE_CELL_SHRUNK = "SHRUNK_TO_POOLED"
ROLE_CELL_ESTIMATED = "ESTIMATED"

#: Absolute z whose first crossing makes a supported role cell "newly
#: exceeding". Read at cell level, on the cell's worst entry: an entry-level
#: reading is not informative here, because a cell carrying 218k ordered pairs
#: has standard errors near 0.002 and the role-blind model already misses its
#: worst entry by 20 standard errors, so almost every entry is past 3 before
#: the role layer touches anything.
ROLE_CELL_Z_LIMIT = 3.0

#: Relative regression a single supported role cell may not exceed, on either
#: its RMSE or its worst absolute z. The role layer is one shared low-rank
#: deviation, not a per-cell parameter, so it is expected to trade a little
#: accuracy in one cell for much more in another; what it may not do is make
#: a cell materially worse than the role-blind fit it replaces.
ROLE_CELL_REGRESSION_LIMIT = 0.05

#: Iteration budget for the role-layer least-squares fit.
ROLE_FIT_ITERATIONS = 40

#: Convergence tolerance for that fit, on the step, the cost and the gradient
#: alike. The quantities being solved for are order 0.1 to 1, so this is far
#: inside the precision any downstream number is reported to; asking for
#: 1e-14 only made Levenberg-Marquardt grind against the identification
#: ridge's nearly flat radial direction without moving the answer.
ROLE_FIT_TOLERANCE = 1e-10

#: Deterministic seed for the role-deviation loading, as a fraction of the
#: symmetric block. ``h = 0, W = 0`` is a stationary point of the objective --
#: every derivative of ``h_r W`` vanishes there -- so the fit has to start
#: away from it or it cannot move at all.
ROLE_FIT_SEED_FRACTION = 0.1

#: Passes of the fit-then-absorb loop for the role layer's pooled leak. The
#: leak depends on ``W``, which depends on the symmetric block the absorption
#: moves, so the two are solved alternately; the coupling is weak and two
#: passes close it to float64 noise.
ROLE_ABSORPTION_PASSES = 2

#: Floor on a role-cell standard error, as a fraction of the cell's median
#: standard error. A bootstrap standard error that comes out near zero is a
#: degenerate draw, not infinite precision, and without a floor it would take
#: over the whole objective.
WEIGHT_FLOOR_FRACTION = 0.1


@dataclass(frozen=True)
class V2Spec:
    """A pre-registered Shadow V2 candidate.

    Every field is a scalar, a structural label, or a rank. None is indexed by
    player, by player pair, or by roster, which is what keeps
    ``pairwise_parameter_count`` and ``player_indexed`` at zero.
    """

    name: str
    #: Shrinkage for the *final* same-team target. The repair's family.
    same_shrinkage: str = SHRINKAGE_EMPIRICAL_BAYES
    same_eb_family: str = EB_FAMILY_BLOCK_DIAGONAL
    #: Shrinkage for the cross-team target. Separate by design: no same-team
    #: hyperparameter may reach the opponent buckets.
    cross_shrinkage: str = SHRINKAGE_SOFT_THRESHOLD
    cross_eb_family: str = EB_FAMILY_BLOCK_DIAGONAL
    shrink_z: float = DEFAULT_SHRINK_Z
    #: Base representation ranks. The accepted V1 values; the same-team repair
    #: arrives through ``r_symmetric`` instead of by raising these.
    k_game: int = DEFAULT_K_GAME
    r_contrast: int = DEFAULT_R_CONTRAST
    #: Rank of the symmetric same-team subspace. ``0`` disables it.
    r_symmetric: int = 0
    #: Whether a centred role deviation is fitted inside that subspace. There
    #: is deliberately no weight on it: ``h`` and ``W`` are identified only up
    #: to reciprocal scaling, so a weight on the scores would not be a
    #: well-defined shrinkage. The layer's own hierarchical factor, derived
    #: from the support of the weakest cell it rests on, does that job.
    role_deviation: bool = False
    role_column: str | None = "role_bucket"
    #: Weight on the bridge-implied same-team target, in ``[0, 1]``.
    bridge_weight: float = 0.0
    temporal_treatment: str = TREATMENT_POOLED

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "same_shrinkage": self.same_shrinkage,
            "same_eb_family": self.same_eb_family,
            "cross_shrinkage": self.cross_shrinkage,
            "cross_eb_family": self.cross_eb_family,
            "shrink_z": float(self.shrink_z),
            "k_game": int(self.k_game),
            "r_contrast": int(self.r_contrast),
            "r_symmetric": int(self.r_symmetric),
            "role_deviation": bool(self.role_deviation),
            "role_column": self.role_column,
            "bridge_weight": float(self.bridge_weight),
            "temporal_treatment": self.temporal_treatment,
        }

    def parameter_count(self, n_stats: int, n_roles: int = 3) -> dict[str, int]:
        """Free parameters by kind. ``pairwise`` must always be zero."""
        triangle = n_stats * (n_stats + 1) // 2
        return {
            "stat_indexed_game": int(n_stats * self.k_game),
            "stat_indexed_contrast": int(n_stats * self.r_contrast),
            "stat_indexed_symmetric": int(n_stats * self.r_symmetric),
            "stat_indexed_competition": int(triangle),
            "stat_indexed_role_deviation": int(
                n_stats * self.r_symmetric if self.role_deviation else 0
            ),
            "role_indexed": int(
                (max(n_roles - 1, 0) if self.role_deviation else 0)
                + (0 if self.role_column is None else n_roles)
            ),
            "scalar_hyperparameters": 4,
            "player_indexed": 0,
            "pairwise": 0,
        }


#: The accepted V1 fit in this module's vocabulary: V1's family on both
#: blocks, V1's ranks, no symmetric subspace, no role deviation, no bridge.
V1_BASE_SPEC = V2Spec(
    name="v1_base",
    same_shrinkage=SHRINKAGE_SOFT_THRESHOLD,
    cross_shrinkage=SHRINKAGE_SOFT_THRESHOLD,
)

#: The accepted bucket repair in this module's vocabulary. ``base_shrinkage``
#: is overridden to the repair's own family so the base construction matches;
#: this is the one spec where the two blocks are deliberately coupled, because
#: that coupling is what the control did.
REPAIR_CONTROL_SPEC = V2Spec(
    name="repair_control",
    same_shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
    cross_shrinkage=SHRINKAGE_EMPIRICAL_BAYES,
    k_game=6,
    r_contrast=6,
)


def shrink_block(
    estimate: np.ndarray,
    standard_error: np.ndarray,
    method: str,
    shrink_z: float,
    eb_family: str,
    is_same_team: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one shrinkage estimator to one pooled block.

    The per-block entry point ``repair.shrink_blocks`` does not have: that
    function takes a single ``RepairSpec`` and applies it to both blocks at
    once, which is exactly the coupling V2 removes.
    """
    if method == SHRINKAGE_SOFT_THRESHOLD:
        return soft_threshold(estimate, standard_error, shrink_z), {
            "shrink_z": float(shrink_z)
        }
    if method == SHRINKAGE_EMPIRICAL_BAYES:
        return empirical_bayes_shrink(
            estimate,
            standard_error,
            family=eb_family,
            is_same_team=is_same_team,
        )
    raise ValueError(f"unknown shrinkage method {method!r}")


@dataclass(frozen=True)
class RolePairMoments:
    """Pooled same-team cross-player moments, split by ordered role pair."""

    stats: tuple[str, ...]
    roles: tuple[str, ...]
    blocks: Mapping[tuple[str, str], np.ndarray]
    pairs: Mapping[tuple[str, str], float]
    games: Mapping[str, int]
    player_shares: Mapping[str, float]
    pair_shares: Mapping[tuple[str, str], float]
    #: Game-clustered bootstrap standard errors of :meth:`symmetrised`, keyed
    #: by unordered role pair. Empty when no bootstrap was asked for.
    standard_errors: Mapping[tuple[str, str], np.ndarray] = field(
        default_factory=dict
    )
    bootstrap_draws: int = 0
    total_games: int = 0

    def symmetrised(self, role_a: str, role_b: str) -> np.ndarray:
        """The role cell as a symmetric matrix, pooling both pair orders.

        ``pair_moments`` symmetrises its pooled blocks, so a role cell has to
        be read the same way to be comparable with the pooled target.
        """
        forward = self.blocks[(role_a, role_b)]
        backward = self.blocks[(role_b, role_a)]
        weight = self.pairs[(role_a, role_b)] + self.pairs[(role_b, role_a)]
        if weight <= 0:
            return np.zeros_like(forward)
        total = (
            forward * self.pairs[(role_a, role_b)]
            + backward.T * self.pairs[(role_b, role_a)]
        ) / weight
        return 0.5 * (total + total.T)

    def support(self, role_a: str, role_b: str) -> float:
        return float(self.pairs[(role_a, role_b)] + self.pairs[(role_b, role_a)])

    def standard_error(self, role_a: str, role_b: str) -> np.ndarray | None:
        """Game-clustered standard error of the cell, or ``None``."""
        key = (role_a, role_b) if (role_a, role_b) in self.standard_errors else (
            role_b,
            role_a,
        )
        return self.standard_errors.get(key)


def role_pair_moments(
    frame: pd.DataFrame,
    stats: Sequence[str],
    role_column: str = "role_bucket",
    value_prefix: str = "zs_",
    bootstrap: int = 0,
    seed: int = 73,
) -> RolePairMoments:
    """Pool same-team cross-player moments by ordered role pair.

    Per game-team the ordered pair sum for roles ``(r, r')`` is

        sum_{a in r, b in r', a != b} z_a z_b'
            = (sum_{a in r} z_a)(sum_{b in r'} z_b)' - [r == r'] sum_{a in r} z_a z_a'

    so the whole table costs one pass over role groups rather than a pass over
    player pairs.

    ``bootstrap`` resamples whole games with replacement to give each cell a
    game-clustered standard error. The per-game numerators and pair counts are
    accumulated on the single pass, so a draw is a weighted sum over games
    rather than another pass over the frame; the acceptance gates are stated
    in standard-error units, so a cell without one cannot be judged at all.
    """
    stats = tuple(stats)
    columns = [f"{value_prefix}{stat}" for stat in stats]
    usable = frame.dropna(subset=["game_id", "team_id", role_column, *columns])
    if usable.empty:
        raise ValueError("no usable rows for role-pair moment estimation")

    roles = tuple(sorted(str(role) for role in usable[role_column].unique()))
    n_stats = len(stats)
    game_ids = sorted(int(value) for value in usable["game_id"].unique())
    game_index = {game_id: position for position, game_id in enumerate(game_ids)}
    n_games = len(game_ids)

    keys = [(a, b) for a in roles for b in roles]
    per_game_totals = {
        key: np.zeros((n_games, n_stats, n_stats), dtype=float) for key in keys
    }
    per_game_counts = {key: np.zeros(n_games, dtype=float) for key in keys}
    games: dict[str, set[int]] = {role: set() for role in roles}
    players = {role: 0.0 for role in roles}

    for (game_id, _), team in usable.groupby(["game_id", "team_id"], sort=True):
        position = game_index[int(game_id)]
        sums: dict[str, np.ndarray] = {}
        self_grams: dict[str, np.ndarray] = {}
        sizes: dict[str, int] = {}
        for role, group in team.groupby(role_column, sort=True):
            label = str(role)
            z = group[columns].to_numpy(dtype=float)
            sums[label] = z.sum(axis=0)
            self_grams[label] = z.T @ z
            sizes[label] = z.shape[0]
            games[label].add(int(game_id))
            players[label] += float(z.shape[0])

        for first, first_sum in sums.items():
            for second, second_sum in sums.items():
                block = np.outer(first_sum, second_sum)
                pair_count = float(sizes[first] * sizes[second])
                if first == second:
                    block = block - self_grams[first]
                    pair_count -= float(sizes[first])
                per_game_totals[(first, second)][position] += block
                per_game_counts[(first, second)][position] += pair_count

    totals = {key: per_game_totals[key].sum(axis=0) for key in keys}
    counts = {key: float(per_game_counts[key].sum()) for key in keys}
    blocks = {
        key: (
            totals[key] / counts[key]
            if counts[key] > 0
            else np.zeros((n_stats, n_stats))
        )
        for key in keys
    }

    def symmetrise(
        numerators: dict[tuple[str, str], np.ndarray],
        denominators: dict[tuple[str, str], float],
        role_a: str,
        role_b: str,
    ) -> np.ndarray:
        weight = denominators[(role_a, role_b)] + denominators[(role_b, role_a)]
        if weight <= 0:
            return np.zeros((n_stats, n_stats), dtype=float)
        total = (numerators[(role_a, role_b)] + numerators[(role_b, role_a)].T) / weight
        return 0.5 * (total + total.T)

    standard_errors: dict[tuple[str, str], np.ndarray] = {}
    if bootstrap > 0 and n_games > 1:
        rng = np.random.default_rng(seed)
        unordered = [
            (first, second)
            for index, first in enumerate(roles)
            for second in roles[index:]
        ]
        draws = {key: np.empty((bootstrap, n_stats, n_stats)) for key in unordered}
        for draw in range(bootstrap):
            picks = rng.integers(0, n_games, size=n_games)
            weights = np.bincount(picks, minlength=n_games).astype(float)
            numerators = {
                key: np.tensordot(weights, per_game_totals[key], axes=(0, 0))
                for key in keys
            }
            denominators = {
                key: float(weights @ per_game_counts[key]) for key in keys
            }
            for key in unordered:
                draws[key][draw] = symmetrise(numerators, denominators, *key)
        standard_errors = {
            key: draws[key].std(axis=0, ddof=1) for key in unordered
        }

    total_players = sum(players.values())
    total_pairs = sum(counts.values())
    return RolePairMoments(
        stats=stats,
        roles=roles,
        blocks=blocks,
        pairs=counts,
        games={role: len(value) for role, value in games.items()},
        player_shares={
            role: (players[role] / total_players if total_players > 0 else 0.0)
            for role in roles
        },
        pair_shares={
            key: (counts[key] / total_pairs if total_pairs > 0 else 0.0)
            for key in counts
        },
        standard_errors=standard_errors,
        bootstrap_draws=int(bootstrap),
        total_games=n_games,
    )


def role_pair_shares(moments: RolePairMoments) -> dict[str, float]:
    """Marginal share of same-team ordered pairs each role appears in.

    This is the measure the linear term ``(h_a + h_b)`` has to average to zero
    against, so it is the measure the role scores are centred on. Player
    shares would be the wrong one: the pooled target is a mean over ordered
    *pairs*, not over players.
    """
    marginal = {role: 0.0 for role in moments.roles}
    for (first, second), share in moments.pair_shares.items():
        marginal[first] += 0.5 * share
        marginal[second] += 0.5 * share
    total = sum(marginal.values())
    if total <= 0:
        return {role: 0.0 for role in moments.roles}
    return {role: marginal[role] / total for role in moments.roles}


def initial_role_scores(
    moments: RolePairMoments,
    pooled: np.ndarray,
    scales: Mapping[str, float] | None = None,
    pooling_games: float = ROLE_POOLING_GAMES,
) -> dict[str, float]:
    """A cheap starting point for the role scores.

    A role's raw score is the projection of its own within-role same-team
    block on the pooled fitted block, minus one: positive when the role is
    more dependent than the pooled level, negative when less. Partial-pooled
    by the role's own game count with ``fit_role_scales``' declared constant,
    then centred on the pair shares.

    This is only a starting point. It is a projection of the *within-role*
    cells alone, so it knows nothing about the cross-role cells it also has to
    explain, and using it as the final estimate is what made the first version
    of this layer miss ``bench``/``rotation`` badly while fitting
    ``bench``/``bench``. :func:`fit_role_layer` re-estimates it against all
    the supported cells at once.
    """
    scales = dict(scales or {})
    denominator = float(np.sum(pooled**2))
    raw: dict[str, float] = {}
    for role in moments.roles:
        scale = float(scales.get(role, 1.0)) ** 2
        observed = moments.symmetrised(role, role) / scale
        if denominator <= 0 or moments.support(role, role) < MIN_ROLE_PAIRS:
            raw[role] = 0.0
            continue
        ratio = float(np.sum(observed * pooled)) / denominator
        level = float(np.sqrt(max(ratio, 0.0)))
        seen = float(moments.games.get(role, 0))
        weight = seen / (seen + float(pooling_games))
        raw[role] = weight * (level - 1.0)

    shares = role_pair_shares(moments)
    centre = sum(shares[role] * raw[role] for role in moments.roles)
    return {role: raw[role] - centre for role in moments.roles}


def role_layer_diagnostics(
    moments: RolePairMoments,
    scores: Mapping[str, float],
) -> dict[str, float]:
    """How much of the pooled block the role layer moves, by term.

    The linear term is centred to zero by construction, so its leakage is a
    float64-noise check on an identity. The quadratic term is not centred --
    ``sum_r p_r h_r = 0`` does not make ``sum_{r, r'} p_{r r'} h_r h_r'``
    vanish -- so its leakage is a real, reported quantity.
    """
    shares = role_pair_shares(moments)
    return {
        "pooled_linear_leakage": float(
            sum(
                share * (scores[first] + scores[second])
                for (first, second), share in moments.pair_shares.items()
            )
        ),
        "pooled_quadratic_leakage": float(
            sum(
                share * scores[first] * scores[second]
                for (first, second), share in moments.pair_shares.items()
            )
        ),
        **{f"centred_score_{role}": float(scores[role]) for role in moments.roles},
        **{f"pair_share_{role}": float(shares[role]) for role in moments.roles},
    }


def canonical_role_split(
    scores: Mapping[str, float],
    deviation: np.ndarray,
    shares: Mapping[str, float],
    symmetric_norm: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Rescale ``(h, W)`` to the balanced split with a fixed sign.

    Only the product ``h_r W`` enters the model, so this changes nothing about
    the fit; it makes the two reported pieces comparable between folds and
    between runs, which they otherwise are not. The balanced split is
    ``||W|| == ||U|| ||h||_p`` with ``||h||_p`` the pair-share weighted norm,
    and the sign is fixed by requiring the largest-magnitude score positive.
    """
    roles = sorted(scores)
    values = np.array([float(scores[role]) for role in roles])
    score_norm = float(
        np.sqrt(sum(float(shares.get(role, 0.0)) * scores[role] ** 2 for role in roles))
    )
    loading_norm = float(np.linalg.norm(deviation))
    if score_norm <= 0.0 or loading_norm <= 0.0 or symmetric_norm <= 0.0:
        return {role: float(scores[role]) for role in roles}, deviation
    factor = float(np.sqrt(symmetric_norm * score_norm / loading_norm))
    values = values / factor
    out = factor * deviation
    dominant = min(roles, key=lambda role: (-abs(float(scores[role])), role))
    if values[roles.index(dominant)] < 0.0:
        values = -values
        out = -out
    return dict(zip(roles, (float(value) for value in values), strict=True)), out


def fit_role_layer(
    moments: RolePairMoments,
    symmetric: np.ndarray,
    pooled: np.ndarray,
    scales: Mapping[str, float] | None = None,
    start: Mapping[str, float] | None = None,
    ridge: float = ROLE_DEVIATION_RIDGE,
    prior_pairs: float = MIN_ROLE_PAIRS,
    iterations: int = ROLE_FIT_ITERATIONS,
    tolerance: float = ROLE_FIT_TOLERANCE,
) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    """Joint least-squares fit of the role scores ``h`` and the loading ``W``.

    With ``U_r = U + h_r W`` a same-team role cell is

        S(r, r') = s_r s_r' [S_pooled + (h_r + h_r') (U W' + W U') + 2 h_r h_r' W W']

    so the role layer is ``len(roles) - 1`` free scores plus one ``n_stats`` by
    ``r_symmetric`` loading. Not a role-pair table: six role cells are
    explained by three functions of ``(h_r, h_r')`` -- a constant, an additive
    term and a multiplicative one -- and a cell the data cannot measure is
    *predicted* from them rather than given its own parameter.

    ``scales`` is V1's accepted *multiplicative* role layer, ``s_r``, which the
    model applies to the whole same-team block. It is supplied here, and the
    cell targets are divided by ``s_r s_r'``, so this layer is fitted against
    the role structure the multiplicative one leaves behind. Fitting the two
    independently and then applying both -- the first version of this code --
    double-counts: the deviation re-explains a role effect the scales have
    already applied, which halved the measured residual while making
    well-fitted cells worse. The scales are held at their role-blind estimate
    while the deviation is fitted, so the deviation also cannot be confounded
    by a scale layer that moved to accommodate it.

    Both pieces are estimated here against all the supported cells at once.
    Estimating the scores separately, from a projection of the within-role
    cells only, fitted ``bench``/``bench`` and then missed
    ``bench``/``rotation`` by more than the pooled model did, because nothing
    in that projection knew the cross-role cells existed.

    Two declared regularisers, neither searched:

    ``ridge``
        a small conditioning ridge on both ``W`` and ``||U|| h``, with one
        coefficient, so neither factor can run away numerically. It is not
        what identifies the split -- see below -- and is reported as
        ``role_identification_penalty_share``, a fraction of a percent.
    ``prior_pairs``
        hierarchical shrinkage of the fitted deviation, ``kappa = S / (S +
        prior_pairs)`` with ``S`` the *least* supported cell the fit used. The
        layer is shared across cells, so it is only as well determined as its
        weakest one. On full pre-2024 history the weakest cell carries about
        218k ordered pairs against a pseudo-count of 20k, so ``kappa`` is
        0.92 and the shrinkage is nearly inactive; on a short fold it bites,
        and a role with no measurable cell resolves to ``h = 0``, which is the
        pooled loading exactly.

    Two hard constraints, both applied inside the parameterisation rather than
    as penalties, so the optimiser cannot violate them at any point it visits:

    *Centring.* The anchor role's score absorbs whatever the free scores
    imply, so ``sum_r p_r h_r == 0`` exactly and the linear term cannot move
    the pooled block however the optimiser moves.

    *Identification.* ``h`` and ``W`` appear only as the product ``h_r W``, so
    ``(h / c, c W)`` is the same model for every nonzero ``c``, sign included:
    the split is not identified by the data at all, and leaving it free gives
    the optimiser an exactly flat direction to wander along. Measured that
    way, five different starting points agreed on the fitted model to five
    decimal places and disagreed wildly on the scores, reporting ``bench`` as
    anything from -1.57 to +1.11. One score -- the role the starting point
    says carries the most signal -- is therefore pinned, which removes the
    flat direction, and the fitted pair is rescaled afterwards to the balanced
    split ``||W|| == ||U|| ||h||`` with a fixed sign convention. That is a
    change of coordinates on the answer, not a change of answer.
    """
    from scipy.optimize import least_squares

    n_stats, rank = symmetric.shape
    roles = moments.roles
    cells = [
        (first, second)
        for index, first in enumerate(roles)
        for second in roles[index:]
        if moments.support(first, second) >= MIN_ROLE_PAIRS
    ]
    zero_scores = {role: 0.0 for role in roles}
    if rank == 0 or not cells or len(roles) < 2:
        return (
            zero_scores,
            np.zeros((n_stats, rank), dtype=float),
            {"role_cells_used": 0.0, "role_layer_fitted": 0.0},
        )

    shares = role_pair_shares(moments)
    # The anchor carries the centring residual, so it is the role with the
    # most same-team pairs behind it: dividing by a small share would amplify
    # the other roles' scores into it.
    anchor = min(roles, key=lambda role: (-shares[role], role))
    free = [role for role in roles if role != anchor]
    if shares[anchor] <= 0:
        return (
            zero_scores,
            np.zeros((n_stats, rank), dtype=float),
            {"role_cells_used": 0.0, "role_layer_fitted": 0.0},
        )

    scales = dict(scales or {})
    scale_pairs = [
        float(scales.get(first, 1.0) * scales.get(second, 1.0))
        for first, second in cells
    ]
    # The model predicts ``s_r s_r'`` times the bracket, so the bracket is
    # fitted against the observed cell divided by ``s_r s_r'``.
    targets = [
        moments.symmetrised(first, second) / scale - pooled
        for (first, second), scale in zip(cells, scale_pairs, strict=True)
    ]
    # Weight each residual entry by one over its own game-clustered standard
    # error, not by the cell's pair count. The acceptance gate is stated in
    # standard-error units, so that is the metric the fit has to minimise:
    # pair count treats a precisely measured ``reb``/``reb`` entry and a noisy
    # ``blk``/``blk`` one as equally informative, and the fit then spends the
    # precise entry to buy the noisy one. Falls back to the pair count when no
    # bootstrap was run, which keeps the estimator defined either way. The
    # ``s_r s_r'`` factor carries the standard error into the divided units the
    # residual is expressed in.
    weights = []
    for (first, second), scale in zip(cells, scale_pairs, strict=True):
        error = moments.standard_error(first, second)
        if error is None:
            weights.append(
                np.full(
                    (n_stats, n_stats),
                    scale * float(np.sqrt(moments.support(first, second))),
                )
            )
        else:
            floor = float(np.median(error[error > 0])) if np.any(error > 0) else 1.0
            weights.append(
                scale / np.maximum(error, WEIGHT_FLOOR_FRACTION * floor)
            )
    # Support each role's score carries, as the pairs of every cell it is an
    # end of, and the weakest cell the layer rests on.
    role_support = {
        role: float(
            sum(
                moments.support(first, second)
                for first, second in cells
                if role in (first, second)
            )
        )
        for role in roles
    }
    weakest_cell = min(moments.support(first, second) for first, second in cells)
    shrinkage = float(weakest_cell / (weakest_cell + max(prior_pairs, 0.0)))

    # One coefficient for both halves of the identification ridge, in the
    # dimensionless units the weighted data residuals already live in.
    mean_weight = float(
        np.mean(np.concatenate([weight.ravel() for weight in weights]))
    )
    ridge_coefficient = float(np.sqrt(max(ridge, 0.0)) * mean_weight)
    symmetric_norm = float(np.linalg.norm(symmetric)) or 1.0
    # Pair-share weighted, so a role the data barely sees is not asked to
    # carry the same share of the norm as one it sees constantly.
    score_metric = np.array([np.sqrt(shares[role]) for role in roles])

    # The pinned score. Chosen as the role whose cells deviate most from the
    # pooled block, measured in the fit's own weighted units, because that is
    # the role whose score the data most clearly needs to be nonzero. Pinning
    # a near-null role instead -- ``rotation``, whose score comes out near
    # zero -- forces ``W`` to absorb a factor of fifty and the optimiser then
    # fails to converge at all.
    deviation_strength = {
        role: float(
            sum(
                np.linalg.norm(weight * target)
                for (first, second), target, weight in zip(
                    cells, targets, weights, strict=True
                )
                if role in (first, second)
            )
        )
        for role in roles
    }
    pivot = min(free, key=lambda role: (-deviation_strength[role], role))
    others = [role for role in free if role != pivot]

    def expand(parameters: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        values = {role: float(parameters[index]) for index, role in enumerate(others)}
        # Pinned at one. Any nonzero value is the same model -- only ``h_r W``
        # enters -- and the overall sign is an invariance too, so pinning the
        # value *and* the sign costs the fit nothing.
        values[pivot] = 1.0
        # Hard centring: the anchor absorbs whatever the free scores imply.
        values[anchor] = -sum(
            shares[role] * values[role] for role in free
        ) / shares[anchor]
        return values, parameters[len(others) :].reshape(n_stats, rank)

    def seed_deviation(scores: Mapping[str, float]) -> np.ndarray:
        """Least-squares ``W`` for fixed scores, ignoring the quadratic term.

        ``h = 0, W = 0`` is a stationary point of the full objective, so the
        fit needs a starting point away from it, and a starting point that is
        merely *not* zero leaves Levenberg-Marquardt to discover the scale of
        ``W`` by itself over tens of thousands of evaluations. Dropping the
        ``2 h_a h_b W W'`` term makes the remainder linear in ``W``, so the
        right scale comes from one ordinary least-squares solve.
        """
        eye = np.eye(n_stats)
        # d(U W' + W U')_ij / dW_pq, as a (i, j, p, q) tensor.
        basis = np.einsum("iq,jp->ijpq", symmetric, eye) + np.einsum(
            "ip,jq->ijpq", eye, symmetric
        )
        rows = []
        right = []
        for (first, second), target, weight in zip(
            cells, targets, weights, strict=True
        ):
            linear = scores[first] + scores[second]
            rows.append(
                (linear * weight[:, :, None, None] * basis).reshape(
                    n_stats * n_stats, n_stats * rank
                )
            )
            right.append((weight * target).ravel())
        solution, *_ = np.linalg.lstsq(
            np.vstack(rows), np.concatenate(right), rcond=None
        )
        seed = solution.reshape(n_stats, rank)
        if not np.any(seed):
            seed = ROLE_FIT_SEED_FRACTION * symmetric
        return seed

    def data_residual(scores: Mapping[str, float], w: np.ndarray) -> np.ndarray:
        cross = symmetric @ w.T
        cross = cross + cross.T
        gram = w @ w.T
        out = []
        for (first, second), target, weight in zip(
            cells, targets, weights, strict=True
        ):
            linear = scores[first] + scores[second]
            quadratic = 2.0 * scores[first] * scores[second]
            out.append(
                (weight * (linear * cross + quadratic * gram - target)).ravel()
            )
        return np.concatenate(out)

    def residual(parameters: np.ndarray) -> np.ndarray:
        scores, w = expand(parameters)
        score_vector = np.array([scores[role] for role in roles])
        return np.concatenate(
            (
                data_residual(scores, w),
                ridge_coefficient * w.ravel(),
                ridge_coefficient * symmetric_norm * score_metric * score_vector,
            )
        )

    initial = (
        start
        if start is not None
        else initial_role_scores(moments, pooled, scales=scales)
    )
    pivot_start = float(initial.get(pivot, 0.0))
    # Rescale the starting scores into the pinned role's units. When the cheap
    # starting estimate says nothing about the pinned role there is nothing to
    # rescale against, and the pinned score alone is already a nondegenerate
    # starting direction.
    initial_scores = (
        np.array([float(initial.get(role, 0.0)) / pivot_start for role in others])
        if abs(pivot_start) > 1e-6
        else np.zeros(len(others), dtype=float)
    )
    seed_scores, _ = expand(
        np.concatenate((initial_scores, np.zeros(n_stats * rank)))
    )
    parameters = np.concatenate(
        (initial_scores, seed_deviation(seed_scores).ravel())
    )
    solution = least_squares(
        residual,
        parameters,
        method="lm",
        max_nfev=iterations * (parameters.size + 1) * 20,
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
    )
    scores, w = expand(solution.x)
    # Hierarchical shrinkage of the fitted deviation. Applied to ``W`` rather
    # than to ``h`` so the reported scores stay the fitted, centred ones; the
    # two are the same model because only the product appears.
    w = shrinkage * w
    scores, w = canonical_role_split(scores, w, shares, symmetric_norm)

    zero_w = np.zeros((n_stats, rank), dtype=float)
    fitted_cost = float(np.sum(data_residual(scores, w) ** 2))
    pooled_cost = float(np.sum(data_residual(zero_scores, zero_w) ** 2))
    penalty = float(np.sum(residual(solution.x) ** 2)) - float(
        np.sum(data_residual(*expand(solution.x)) ** 2)
    )
    # A role layer that cannot beat the pooled fit on the *data* term, after
    # shrinkage, is not worth carrying, and reporting it as zero makes the
    # fallback explicit rather than silent.
    if fitted_cost >= pooled_cost:
        return (
            zero_scores,
            zero_w,
            {
                "role_cells_used": float(len(cells)),
                "role_layer_fitted": 0.0,
                "role_fit_cost": fitted_cost,
                "pooled_fit_cost": pooled_cost,
                "role_shrinkage_factor": shrinkage,
            },
        )

    diagnostics = {
        "role_cells_used": float(len(cells)),
        "role_layer_fitted": 1.0,
        "role_fit_cost": fitted_cost,
        "pooled_fit_cost": pooled_cost,
        "role_fit_data_cost_fraction": float(fitted_cost / pooled_cost)
        if pooled_cost > 0
        else 0.0,
        "role_identification_penalty_share": float(
            penalty / (penalty + fitted_cost)
        )
        if penalty + fitted_cost > 0
        else 0.0,
        "role_deviation_norm": float(np.linalg.norm(w)),
        "role_score_norm": float(
            np.linalg.norm(score_metric * np.array([scores[role] for role in roles]))
        ),
        # How far the role layer moves a cell, relative to the pooled block it
        # perturbs. Reported against the whole same-team block rather than
        # against ``U``: ``U`` only carries the part of the target the base
        # construction cannot represent, so it is small, and a ratio against
        # it reads as a huge perturbation of something that is itself a
        # correction.
        "role_relative_perturbation": float(
            max(
                np.linalg.norm(
                    (scores[first] + scores[second]) * (symmetric @ w.T + w @ symmetric.T)
                    + 2.0 * scores[first] * scores[second] * (w @ w.T)
                )
                for first, second in cells
            )
            / (float(np.linalg.norm(pooled)) or 1.0)
        ),
        "role_shrinkage_factor": shrinkage,
        "role_weakest_cell_pairs": float(weakest_cell),
        "role_fit_iterations": float(solution.nfev),
        "role_fit_status": float(solution.status),
        "role_fit_optimality": float(solution.optimality),
        **{
            f"role_support_{role}": float(role_support[role]) for role in roles
        },
        **role_layer_diagnostics(moments, scores),
    }
    return scores, w, diagnostics


@dataclass(frozen=True)
class V2Fit:
    """A fitted Shadow V2 candidate and everything needed to audit it."""

    spec: V2Spec
    loadings: SharedFactorLoadings
    moments: PairMoments
    base_same_target: np.ndarray
    same_target: np.ndarray
    cross_target: np.ndarray
    base_same_fitted: np.ndarray
    base_cross_fitted: np.ndarray
    symmetric_gap: np.ndarray
    role_scores: Mapping[str, float]
    role_diagnostics: Mapping[str, float]
    shrink_diagnostics: Mapping[str, float]
    competition_evidence: Mapping[str, float]
    bridge_targets: Mapping[str, float] = field(default_factory=dict)
    temporal_overrides: Mapping[str, float] = field(default_factory=dict)

    @property
    def factor_fit(self) -> FactorFit:
        """The V2 fit expressed in the control's own result type.

        The validation driver, the gate evaluator and the artifact writers all
        consume ``FactorFit``, so V2 hands back the same object rather than a
        parallel type the downstream code would have to learn.
        """
        game_gram = self.loadings.game_gram()
        contrast_gram = self.loadings.contrast_gram()
        competition = self.loadings.competition_gram()
        return FactorFit(
            loadings=self.loadings,
            moments=self.moments,
            same_team_shrunk=self.same_target,
            cross_team_shrunk=self.cross_target,
            game_gram_eigenvalues=np.linalg.eigvalsh(game_gram)[::-1],
            contrast_gram_eigenvalues=np.linalg.eigvalsh(contrast_gram)[::-1],
            competition_gram_eigenvalues=np.linalg.eigvalsh(competition)[::-1],
            k_game=self.loadings.k_game,
            r_competition=self.loadings.r_competition,
            shrink_z=float(self.spec.shrink_z),
            competition_evidence={
                **self.competition_evidence,
                **self.shrink_diagnostics,
            },
        )

    def cross_team_unchanged_deviation(self) -> float:
        """Largest entry by which the fitted cross-team block moved.

        The symmetric subspace and the role deviation cancel from ``A - B``
        algebraically, so this is a float64-noise check on an identity, not a
        tolerance on an estimate.
        """
        return float(
            np.max(
                np.abs(self.loadings.cross_team_correlation() - self.base_cross_fitted)
            )
        )

    def diagnostics(self) -> dict[str, object]:
        fitted_same = self.loadings.pooled_same_team_correlation()
        fitted_cross = self.loadings.cross_team_correlation()
        return {
            "spec": self.spec.payload(),
            "stats": list(self.moments.stats),
            "games": int(self.moments.games),
            "same_team_pairs": float(self.moments.same_team_pairs),
            "cross_team_pairs": float(self.moments.cross_team_pairs),
            "observed_same_team_correlation": self.moments.same_team.tolist(),
            "observed_cross_team_correlation": self.moments.cross_team.tolist(),
            "observed_same_team_se": self.moments.same_team_se.tolist(),
            "observed_cross_team_se": self.moments.cross_team_se.tolist(),
            "base_same_team_target": self.base_same_target.tolist(),
            "same_team_target": self.same_target.tolist(),
            "cross_team_target": self.cross_target.tolist(),
            "base_same_team_fitted": self.base_same_fitted.tolist(),
            "base_cross_team_fitted": self.base_cross_fitted.tolist(),
            "fitted_same_team_correlation": fitted_same.tolist(),
            "role_blind_same_team_correlation": (
                self.loadings.same_team_correlation().tolist()
            ),
            "role_quadratic_share": self.loadings.role_quadratic_share(),
            "fitted_cross_team_correlation": fitted_cross.tolist(),
            "symmetric_gap": self.symmetric_gap.tolist(),
            "r_symmetric": int(self.loadings.r_symmetric),
            "role_scores": {
                str(key): float(value) for key, value in self.role_scores.items()
            },
            "role_diagnostics": {
                str(key): float(value) for key, value in self.role_diagnostics.items()
            },
            "bridge_targets": {
                str(key): float(value) for key, value in self.bridge_targets.items()
            },
            "temporal_overrides": {
                str(key): float(value)
                for key, value in self.temporal_overrides.items()
            },
            "cross_team_unchanged_deviation": self.cross_team_unchanged_deviation(),
            "same_team_fit_rmse": float(
                np.sqrt(np.mean((fitted_same - self.same_target) ** 2))
            ),
            "cross_team_fit_rmse": float(
                np.sqrt(np.mean((fitted_cross - self.cross_target) ** 2))
            ),
        }


def _apply_entry_overrides(
    block: np.ndarray,
    stats: Sequence[str],
    overrides: Mapping[tuple[str, str], float] | None,
) -> np.ndarray:
    """Replace named stat-pair entries of a symmetric block, symmetrically."""
    if not overrides:
        return block
    index = {stat: position for position, stat in enumerate(stats)}
    out = block.copy()
    for (first, second), value in overrides.items():
        if first not in index or second not in index:
            continue
        i, j = index[first], index[second]
        out[i, j] = float(value)
        out[j, i] = float(value)
    return out


def fit_v2_factors(
    frame: pd.DataFrame,
    stats: Sequence[str],
    spec: V2Spec = V1_BASE_SPEC,
    bootstrap: int = 200,
    seed: int = 73,
    value_prefix: str = "zs_",
    temporal_overrides: Mapping[tuple[str, str], float] | None = None,
    bridge_targets: Mapping[tuple[str, str], float] | None = None,
    moments: PairMoments | None = None,
    role_moments: RolePairMoments | None = None,
    role_bootstrap: int = 200,
) -> V2Fit:
    """Fit a Shadow V2 candidate.

    ``temporal_overrides`` replaces named same-team entries with a temporal
    treatment's posterior mean; ``bridge_targets`` supplies the bridge-implied
    latent correlation per same-team entry, which is blended in at
    ``spec.bridge_weight``. Both are passed in rather than computed here
    because both are estimated on training folds by their own drivers, and
    keeping the estimation outside the fit is what makes the no-lookahead
    boundary auditable.
    """
    stats = tuple(stats)
    if moments is None:
        moments = pair_moments(
            frame,
            stats,
            value_prefix=value_prefix,
            bootstrap=bootstrap,
            seed=seed,
        )

    cross_target, cross_diagnostics = shrink_block(
        moments.cross_team,
        moments.cross_team_se,
        method=spec.cross_shrinkage,
        shrink_z=spec.shrink_z,
        eb_family=spec.cross_eb_family,
        is_same_team=False,
    )

    # The base construction reads the *cross-team* family on both blocks, so
    # the fitted cross-team block is a function of the cross-team estimator
    # alone. Everything the same-team estimator adds arrives later, through
    # the symmetric subspace, where it cannot reach ``A - B``.
    base_same_target, base_diagnostics = shrink_block(
        moments.same_team,
        moments.same_team_se,
        method=spec.cross_shrinkage,
        shrink_z=spec.shrink_z,
        eb_family=spec.cross_eb_family,
        is_same_team=True,
    )

    competition_allowed, competition_evidence = competition_gate(
        moments, base_same_target
    )
    additive = dominating_additive_gram(base_same_target, cross_target)
    competition = (
        (additive - base_same_target)
        if competition_allowed
        else np.zeros_like(base_same_target)
    )

    game_gram = 0.5 * (additive + cross_target)
    contrast_gram = 0.5 * (additive - cross_target)
    _, game_loadings = project_psd_rank(game_gram, rank=spec.k_game)
    _, contrast_loadings = project_psd_rank(contrast_gram, rank=spec.r_contrast)
    _, competition_loadings = project_psd_rank(competition, rank=len(stats))
    if not np.any(competition_loadings):
        competition_loadings = None

    team_contrast = (
        contrast_loadings[:, 0]
        if contrast_loadings.shape[1] == 1
        else contrast_loadings
    )

    base = SharedFactorLoadings(
        stats=stats,
        game=game_loadings,
        team_contrast=team_contrast,
        competition=competition_loadings,
    )
    base_same_fitted = base.same_team_correlation()
    base_cross_fitted = base.cross_team_correlation()

    same_target, same_diagnostics = shrink_block(
        moments.same_team,
        moments.same_team_se,
        method=spec.same_shrinkage,
        shrink_z=spec.shrink_z,
        eb_family=spec.same_eb_family,
        is_same_team=True,
    )
    same_target = _apply_entry_overrides(same_target, stats, temporal_overrides)
    if bridge_targets and spec.bridge_weight > 0:
        index = {stat: position for position, stat in enumerate(stats)}
        blended = same_target.copy()
        for (first, second), required in bridge_targets.items():
            if first not in index or second not in index:
                continue
            i, j = index[first], index[second]
            value = (1.0 - spec.bridge_weight) * same_target[
                i, j
            ] + spec.bridge_weight * float(required)
            blended[i, j] = value
            blended[j, i] = value
        same_target = blended

    # The symmetric subspace carries exactly the part of the same-team target
    # the base construction cannot represent. Halved because appending ``U``
    # to both loading blocks lifts the same-team block by ``2 U U'``.
    gap = same_target - base_same_fitted
    symmetric: np.ndarray | None = None
    if spec.r_symmetric > 0:
        _, symmetric = project_psd_rank(0.5 * gap, rank=spec.r_symmetric)
        if not np.any(symmetric):
            symmetric = None

    # V1's multiplicative role layer, estimated on the role-blind pooled block
    # exactly as ``factors.fit_shared_factors`` does, and then held fixed. The
    # additive role deviation is fitted against what it leaves, so the two
    # layers compose instead of competing for the same role effect.
    role_scale: dict[str, float] = {}
    if spec.role_column is not None:
        role_scale = fit_role_scales(
            frame,
            stats,
            base=SharedFactorLoadings(
                stats=stats,
                game=base.game,
                team_contrast=base.team_contrast,
                competition=base.competition,
                symmetric=symmetric,
            ),
            role_column=spec.role_column,
            value_prefix=value_prefix,
        )

    role_scores: dict[str, float] = {}
    role_diagnostics: dict[str, float] = {}
    role_deviation: np.ndarray | None = None
    role_shares: dict[tuple[str, str], float] = {}
    if spec.role_deviation and spec.role_column is not None and symmetric is not None:
        if role_moments is None:
            role_moments = role_pair_moments(
                frame,
                stats,
                role_column=spec.role_column,
                value_prefix=value_prefix,
                bootstrap=role_bootstrap,
                seed=seed,
            )
        role_shares = dict(role_moments.pair_shares)
        # Fit the role layer, then re-solve the symmetric block against a gap
        # reduced by the role layer's quadratic leak, so the pair-share average
        # lands on the same-team target rather than ``2 Q W W'`` away from it.
        # Two passes: the leak depends on ``W``, which depends on ``U``, and
        # the dependence is weak enough that it has converged by then. Reported
        # either way as ``pooled_leak_absorbed``.
        leak = 0.0
        for _ in range(ROLE_ABSORPTION_PASSES):
            pooled = SharedFactorLoadings(
                stats=stats,
                game=base.game,
                team_contrast=base.team_contrast,
                competition=base.competition,
                symmetric=symmetric,
            ).same_team_correlation()
            role_scores, role_deviation, role_diagnostics = fit_role_layer(
                role_moments, symmetric, pooled, scales=role_scale
            )
            leak = sum(
                share * role_scores[first] * role_scores[second]
                for (first, second), share in role_shares.items()
            )
            absorbed = 0.5 * gap - leak * (role_deviation @ role_deviation.T)
            _, symmetric = project_psd_rank(absorbed, rank=spec.r_symmetric)
            if not np.any(symmetric):
                symmetric = None
                break
        role_diagnostics = {
            **role_diagnostics,
            "pooled_leak_absorbed": float(leak),
            # What V1's multiplicative layer alone does to the pooled level.
            # Renormalisation sets the *player*-weighted mean scale to one, so
            # the pair-weighted mean of ``s_a s_b`` is near one but not equal
            # to it; the residual is inherited from V1, not introduced here,
            # and is reported so the role deviation is not blamed for it.
            "role_scale_pair_mean": float(
                sum(
                    share
                    * role_scale.get(first, 1.0)
                    * role_scale.get(second, 1.0)
                    for (first, second), share in role_shares.items()
                )
            ),
        }
        if role_deviation is None or not np.any(role_deviation) or symmetric is None:
            role_deviation = None
            role_scores = {}
            role_shares = {}

    loadings = SharedFactorLoadings(
        stats=stats,
        game=base.game,
        team_contrast=base.team_contrast,
        competition=base.competition,
        role_scale=role_scale,
        symmetric=symmetric,
        role_deviation=role_deviation,
        role_offset=dict(role_scores),
        role_pair_shares=role_shares,
    )

    return V2Fit(
        spec=spec,
        loadings=loadings,
        moments=moments,
        base_same_target=base_same_target,
        same_target=same_target,
        cross_target=cross_target,
        base_same_fitted=base_same_fitted,
        base_cross_fitted=base_cross_fitted,
        symmetric_gap=gap,
        role_scores=role_scores,
        role_diagnostics=role_diagnostics,
        shrink_diagnostics={
            **{f"base_{key}": value for key, value in base_diagnostics.items()},
            **{f"same_{key}": value for key, value in same_diagnostics.items()},
            **{f"cross_{key}": value for key, value in cross_diagnostics.items()},
        },
        competition_evidence=competition_evidence,
        bridge_targets={
            f"{first}_{second}": float(value)
            for (first, second), value in (bridge_targets or {}).items()
        },
        temporal_overrides={
            f"{first}_{second}": float(value)
            for (first, second), value in (temporal_overrides or {}).items()
        },
    )


def role_cell_report(
    moments: RolePairMoments,
    loadings: SharedFactorLoadings,
    buckets: Mapping[str, tuple[str, str]],
) -> dict[str, dict[str, object]]:
    """Observed-versus-fitted same-team dependence by role pair and bucket.

    A cell below :data:`MIN_ROLE_PAIRS` is reported as
    :data:`ROLE_CELL_SHRUNK` with the pooled prediction and no point estimate,
    which is the honest reading: the model does predict it, from pooled
    information, and the data cannot say whether that prediction is right.
    """
    index = {stat: position for position, stat in enumerate(moments.stats)}
    pooled = loadings.same_team_correlation()
    out: dict[str, dict[str, object]] = {}
    for position, first in enumerate(moments.roles):
        for second in moments.roles[position:]:
            observed = moments.symmetrised(first, second)
            predicted = loadings.same_team_correlation_for_roles(first, second)
            support = moments.support(first, second)
            estimated = support >= MIN_ROLE_PAIRS
            cell: dict[str, object] = {
                "role_pair": [first, second],
                "ordered_pairs": float(support),
                "status": ROLE_CELL_ESTIMATED if estimated else ROLE_CELL_SHRUNK,
                "buckets": {},
            }
            error = moments.standard_error(first, second)
            for name, (stat_a, stat_b) in buckets.items():
                if stat_a not in index or stat_b not in index:
                    continue
                i, j = index[stat_a], index[stat_b]
                se = None if error is None else float(error[i, j])
                entry: dict[str, float | None] = {
                    "pooled_prediction": float(pooled[i, j]),
                    "role_prediction": float(predicted[i, j]),
                    "observed": float(observed[i, j]) if estimated else None,
                    "observed_se": se if estimated else None,
                }
                if estimated and se is not None and se > 0:
                    entry["pooled_z"] = float((pooled[i, j] - observed[i, j]) / se)
                    entry["role_z"] = float((predicted[i, j] - observed[i, j]) / se)
                cell["buckets"][name] = entry  # type: ignore[index]
            out[f"{first}__{second}"] = cell
    return out


def role_conditioned_rmse(
    moments: RolePairMoments,
    loadings: SharedFactorLoadings,
    pooled_only: SharedFactorLoadings | None = None,
) -> dict[str, object]:
    """Support-weighted RMSE over the role cells the data can measure.

    ``pooled_only`` supplies the comparison fit whose role layer is absent, so
    the improvement attributable to the role deviation is reported rather than
    inferred.
    """
    numerator = 0.0
    pooled_numerator = 0.0
    weight_total = 0.0
    cells = 0
    worst_role_z = 0.0
    worst_pooled_z = 0.0
    newly_exceeding: list[str] = []
    rmse_regression = 0.0
    z_regression = 0.0
    regressed: list[str] = []
    for position, first in enumerate(moments.roles):
        for second in moments.roles[position:]:
            support = moments.support(first, second)
            if support < MIN_ROLE_PAIRS:
                continue
            observed = moments.symmetrised(first, second)
            predicted = loadings.same_team_correlation_for_roles(first, second)
            cell_rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
            numerator += support * cell_rmse**2
            reference = (
                None
                if pooled_only is None
                else pooled_only.same_team_correlation_for_roles(first, second)
            )
            if reference is not None:
                cell_pooled_rmse = float(np.sqrt(np.mean((observed - reference) ** 2)))
                pooled_numerator += support * cell_pooled_rmse**2
                if cell_pooled_rmse > 0:
                    ratio = cell_rmse / cell_pooled_rmse - 1.0
                    rmse_regression = max(rmse_regression, ratio)
                    if ratio > ROLE_CELL_REGRESSION_LIMIT:
                        regressed.append(f"{first}__{second}")
            error = moments.standard_error(first, second)
            if error is not None:
                safe = np.where(error > 0, error, np.inf)
                cell_role_z = float(np.max(np.abs(predicted - observed) / safe))
                worst_role_z = max(worst_role_z, cell_role_z)
                if reference is not None:
                    cell_pooled_z = float(np.max(np.abs(reference - observed) / safe))
                    worst_pooled_z = max(worst_pooled_z, cell_pooled_z)
                    # "Newly" exceeding: the gate forbids breaking a cell the
                    # role-blind model already fitted, not inheriting one it
                    # already missed.
                    if (
                        cell_role_z > ROLE_CELL_Z_LIMIT
                        and cell_pooled_z <= ROLE_CELL_Z_LIMIT
                    ):
                        newly_exceeding.append(f"{first}__{second}")
                    if cell_pooled_z > 0:
                        ratio = cell_role_z / cell_pooled_z - 1.0
                        z_regression = max(z_regression, ratio)
                        if ratio > ROLE_CELL_REGRESSION_LIMIT:
                            regressed.append(f"{first}__{second}")
            weight_total += support
            cells += 1

    # No supported cell means the data cannot measure the role layer at all,
    # which is a reportable outcome rather than a missing key: the caller still
    # needs the comparison fields to say "not measurable here".
    if weight_total <= 0:
        out: dict[str, object] = {
            "role_cells": 0.0,
            "role_conditioned_rmse": float("nan"),
            "measurable": 0.0,
            "worst_role_abs_z": float("nan"),
            "worst_pooled_abs_z": float("nan"),
            "newly_exceeding_cells": [],
            "regressed_cells": [],
            "worst_cell_rmse_regression": 0.0,
            "worst_cell_z_regression": 0.0,
        }
        if pooled_only is not None:
            out["pooled_only_role_conditioned_rmse"] = float("nan")
            out["role_rmse_improvement_fraction"] = 0.0
        return out

    out = {
        "role_cells": float(cells),
        "role_conditioned_rmse": float(np.sqrt(numerator / weight_total)),
        "measurable": 1.0,
        "worst_role_abs_z": worst_role_z,
        "worst_pooled_abs_z": worst_pooled_z,
        "newly_exceeding_cells": sorted(set(newly_exceeding)),
        "regressed_cells": sorted(set(regressed)),
        "worst_cell_rmse_regression": float(rmse_regression),
        "worst_cell_z_regression": float(z_regression),
    }
    if pooled_only is not None:
        pooled_rmse = float(np.sqrt(pooled_numerator / weight_total))
        out["pooled_only_role_conditioned_rmse"] = pooled_rmse
        out["role_rmse_improvement_fraction"] = (
            float((pooled_rmse - out["role_conditioned_rmse"]) / pooled_rmse)  # type: ignore[operator]
            if pooled_rmse > 0
            else 0.0
        )
    return out
