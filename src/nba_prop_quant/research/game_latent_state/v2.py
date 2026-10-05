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

#: Gauss-Newton iterations for the role-deviation loading. The residual is
#: quadratic in ``W`` and the linear least-squares solution is already close,
#: so this converges to float64 precision long before the cap.
ROLE_FIT_ITERATIONS = 40


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
    #: Whether a centred role deviation is fitted inside that subspace.
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


def role_pair_moments(
    frame: pd.DataFrame,
    stats: Sequence[str],
    role_column: str = "role_bucket",
    value_prefix: str = "zs_",
) -> RolePairMoments:
    """Pool same-team cross-player moments by ordered role pair.

    Per game-team the ordered pair sum for roles ``(r, r')`` is

        sum_{a in r, b in r', a != b} z_a z_b'
            = (sum_{a in r} z_a)(sum_{b in r'} z_b)' - [r == r'] sum_{a in r} z_a z_a'

    so the whole table costs one pass over role groups rather than a pass over
    player pairs.
    """
    stats = tuple(stats)
    columns = [f"{value_prefix}{stat}" for stat in stats]
    usable = frame.dropna(subset=["game_id", "team_id", role_column, *columns])
    if usable.empty:
        raise ValueError("no usable rows for role-pair moment estimation")

    roles = tuple(sorted(str(role) for role in usable[role_column].unique()))
    n_stats = len(stats)
    totals = {
        (a, b): np.zeros((n_stats, n_stats), dtype=float) for a in roles for b in roles
    }
    counts = {(a, b): 0.0 for a in roles for b in roles}
    games: dict[str, set[int]] = {role: set() for role in roles}
    players = {role: 0.0 for role in roles}

    for (game_id, _), team in usable.groupby(["game_id", "team_id"], sort=True):
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
                totals[(first, second)] += block
                counts[(first, second)] += pair_count

    blocks = {
        key: (totals[key] / counts[key] if counts[key] > 0 else np.zeros((n_stats, n_stats)))
        for key in totals
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
    )


def centred_role_scores(
    moments: RolePairMoments,
    pooled: np.ndarray,
    pooling_games: float = ROLE_POOLING_GAMES,
) -> tuple[dict[str, float], dict[str, float]]:
    """Weighted-centred per-role dependence scores and their diagnostics.

    A role's raw score is the projection of its own within-role same-team
    block on the pooled fitted block, minus one: positive when the role is
    more dependent than the pooled level, negative when less. It is shrunk
    toward zero by the role's own game count -- the same partial-pooling rule
    ``fit_role_scales`` uses, with the same declared constant -- and then
    centred on the *pair* shares so the pooled block is left where the base
    construction put it.

    Centring on pair shares rather than player shares is deliberate: the
    pooled target is a mean over same-team ordered pairs, so that is the
    measure the linear term has to average to zero against.
    """
    denominator = float(np.sum(pooled**2))
    raw: dict[str, float] = {}
    for role in moments.roles:
        observed = moments.symmetrised(role, role)
        if denominator <= 0 or moments.support(role, role) < MIN_ROLE_PAIRS:
            raw[role] = 0.0
            continue
        ratio = float(np.sum(observed * pooled)) / denominator
        level = float(np.sqrt(max(ratio, 0.0)))
        seen = float(moments.games.get(role, 0))
        weight = seen / (seen + float(pooling_games))
        raw[role] = weight * (level - 1.0)

    # Marginal pair share of each role, i.e. how often a role appears as one
    # end of a same-team ordered pair. This is the measure the linear term
    # ``(h_a + h_b)`` averages against.
    marginal = {role: 0.0 for role in moments.roles}
    for (first, second), share in moments.pair_shares.items():
        marginal[first] += 0.5 * share
        marginal[second] += 0.5 * share
    total = sum(marginal.values())
    if total <= 0:
        return {role: 0.0 for role in moments.roles}, {}

    centre = sum(marginal[role] / total * raw[role] for role in moments.roles)
    scores = {role: raw[role] - centre for role in moments.roles}

    linear_leak = sum(
        share * (scores[first] + scores[second])
        for (first, second), share in moments.pair_shares.items()
    )
    quadratic_leak = sum(
        share * scores[first] * scores[second]
        for (first, second), share in moments.pair_shares.items()
    )
    diagnostics = {
        "pooled_linear_leakage": float(linear_leak),
        "pooled_quadratic_leakage": float(quadratic_leak),
        **{f"raw_score_{role}": float(value) for role, value in raw.items()},
        **{f"centred_score_{role}": float(value) for role, value in scores.items()},
        **{
            f"pair_share_{role}": float(marginal[role] / total)
            for role in moments.roles
        },
    }
    return scores, diagnostics


def fit_role_deviation(
    moments: RolePairMoments,
    symmetric: np.ndarray,
    pooled: np.ndarray,
    scores: Mapping[str, float],
    ridge: float = ROLE_DEVIATION_RIDGE,
    iterations: int = ROLE_FIT_ITERATIONS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Least-squares fit of the role-deviation loading ``W``.

    With ``U_r = U + h_r W`` a same-team role cell is

        S(r, r') = S_pooled + (h_r + h_r') (U W' + W U') + 2 h_r h_r' W W'

    which is quadratic in ``W``. Starting from the linear least-squares
    solution (the quadratic term dropped) and taking Gauss-Newton steps on the
    full residual converges in a handful of iterations. Cells below
    :data:`MIN_ROLE_PAIRS` are excluded from the objective entirely, so a
    sparse role pair cannot pull ``W``; it is predicted by the fitted ``h``
    and ``W`` instead of estimating them.
    """
    n_stats, rank = symmetric.shape
    cells = [
        (first, second)
        for index, first in enumerate(moments.roles)
        for second in moments.roles[index:]
        if moments.support(first, second) >= MIN_ROLE_PAIRS
    ]
    if rank == 0 or not cells:
        return np.zeros((n_stats, rank), dtype=float), {"role_cells_used": 0.0}

    targets = []
    coefficients = []
    weights = []
    for first, second in cells:
        deviation = moments.symmetrised(first, second) - pooled
        linear = float(scores[first] + scores[second])
        quadratic = 2.0 * float(scores[first]) * float(scores[second])
        targets.append(deviation)
        coefficients.append((linear, quadratic))
        weights.append(np.sqrt(moments.support(first, second)))

    def residual(flat: np.ndarray) -> np.ndarray:
        w = flat.reshape(n_stats, rank)
        cross = symmetric @ w.T
        gram = w @ w.T
        out = []
        for (linear, quadratic), target, weight in zip(
            coefficients, targets, weights, strict=True
        ):
            predicted = linear * (cross + cross.T) + quadratic * gram
            out.append(weight * (predicted - target).ravel())
        out.append(np.sqrt(ridge) * flat * float(np.sum(weights)))
        return np.concatenate(out)

    def jacobian(flat: np.ndarray) -> np.ndarray:
        w = flat.reshape(n_stats, rank)
        size = flat.size
        rows = []
        basis = np.eye(size).reshape(size, n_stats, rank)
        for (linear, quadratic), weight in zip(coefficients, weights, strict=True):
            block = np.empty((n_stats * n_stats, size))
            for column in range(size):
                direction = basis[column]
                cross = symmetric @ direction.T
                gram = direction @ w.T + w @ direction.T
                block[:, column] = (
                    weight * (linear * (cross + cross.T) + quadratic * gram)
                ).ravel()
            rows.append(block)
        rows.append(np.sqrt(ridge) * float(np.sum(weights)) * np.eye(size))
        return np.vstack(rows)

    flat = np.zeros(n_stats * rank, dtype=float)
    # One Gauss-Newton step from zero is the linear least-squares solution,
    # because the quadratic term and its derivative both vanish there.
    for _ in range(iterations):
        r = residual(flat)
        j = jacobian(flat)
        step, *_ = np.linalg.lstsq(j, -r, rcond=None)
        flat = flat + step
        if float(np.max(np.abs(step))) < 1e-14:
            break

    w = flat.reshape(n_stats, rank)
    final = residual(flat)
    diagnostics = {
        "role_cells_used": float(len(cells)),
        "role_fit_residual_norm": float(np.linalg.norm(final)),
        "role_deviation_norm": float(np.linalg.norm(w)),
    }
    return w, diagnostics


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
        fitted_same = self.loadings.same_team_correlation()
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

    pooled_loadings = SharedFactorLoadings(
        stats=stats,
        game=base.game,
        team_contrast=base.team_contrast,
        competition=base.competition,
        symmetric=symmetric,
    )

    role_scores: dict[str, float] = {}
    role_diagnostics: dict[str, float] = {}
    role_deviation: np.ndarray | None = None
    if spec.role_deviation and spec.role_column is not None and symmetric is not None:
        role_moments = role_pair_moments(
            frame, stats, role_column=spec.role_column, value_prefix=value_prefix
        )
        pooled = pooled_loadings.same_team_correlation()
        role_scores, role_diagnostics = centred_role_scores(role_moments, pooled)
        role_deviation, fit_diagnostics = fit_role_deviation(
            role_moments, symmetric, pooled, role_scores
        )
        role_diagnostics = {**role_diagnostics, **fit_diagnostics}
        if not np.any(role_deviation):
            role_deviation = None
            role_scores = {}

    role_scale: dict[str, float] = {}
    if spec.role_column is not None:
        role_scale = fit_role_scales(
            frame,
            stats,
            base=pooled_loadings,
            role_column=spec.role_column,
            value_prefix=value_prefix,
        )

    loadings = SharedFactorLoadings(
        stats=stats,
        game=base.game,
        team_contrast=base.team_contrast,
        competition=base.competition,
        role_scale=role_scale,
        symmetric=symmetric,
        role_deviation=role_deviation,
        role_offset=dict(role_scores),
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
            for name, (stat_a, stat_b) in buckets.items():
                if stat_a not in index or stat_b not in index:
                    continue
                i, j = index[stat_a], index[stat_b]
                entry: dict[str, float | None] = {
                    "pooled_prediction": float(pooled[i, j]),
                    "role_prediction": float(predicted[i, j]),
                    "observed": float(observed[i, j]) if estimated else None,
                }
                cell["buckets"][name] = entry  # type: ignore[index]
            out[f"{first}__{second}"] = cell
    return out


def role_conditioned_rmse(
    moments: RolePairMoments,
    loadings: SharedFactorLoadings,
    pooled_only: SharedFactorLoadings | None = None,
) -> dict[str, float]:
    """Support-weighted RMSE over the role cells the data can measure.

    ``pooled_only`` supplies the comparison fit whose role layer is absent, so
    the improvement attributable to the role deviation is reported rather than
    inferred.
    """
    numerator = 0.0
    pooled_numerator = 0.0
    weight_total = 0.0
    cells = 0
    for position, first in enumerate(moments.roles):
        for second in moments.roles[position:]:
            support = moments.support(first, second)
            if support < MIN_ROLE_PAIRS:
                continue
            observed = moments.symmetrised(first, second)
            predicted = loadings.same_team_correlation_for_roles(first, second)
            numerator += support * float(np.mean((observed - predicted) ** 2))
            if pooled_only is not None:
                reference = pooled_only.same_team_correlation_for_roles(first, second)
                pooled_numerator += support * float(
                    np.mean((observed - reference) ** 2)
                )
            weight_total += support
            cells += 1

    if weight_total <= 0:
        return {"role_cells": 0.0, "role_conditioned_rmse": float("nan")}

    out = {
        "role_cells": float(cells),
        "role_conditioned_rmse": float(np.sqrt(numerator / weight_total)),
    }
    if pooled_only is not None:
        pooled_rmse = float(np.sqrt(pooled_numerator / weight_total))
        out["pooled_only_role_conditioned_rmse"] = pooled_rmse
        out["role_rmse_improvement_fraction"] = (
            float((pooled_rmse - out["role_conditioned_rmse"]) / pooled_rmse)
            if pooled_rmse > 0
            else 0.0
        )
    return out
