"""Bucket-repair candidate families for the latent-state shadow layer.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Accepted shadow V1 under-captures two same-team dependence buckets,
``teammate_reb_reb`` and ``teammate_ast_ast``. This module holds the
alternative *shrinkage* and *rank* choices that might repair them, and it is
deliberately additive: :mod:`factors` is not modified, so the accepted V1 fit
remains bit-for-bit reproducible as the control.
:func:`fit_repaired_factors` with :data:`V1_CONTROL_SPEC` is asserted by test
to reproduce :func:`factors.fit_shared_factors` exactly.

Why V1 attenuates those two buckets
-----------------------------------
Measured on the pre-2024 training history only, the loss decomposes into two
independent mechanisms:

1.  **Soft thresholding.** ``sign(r) * max(0, |r| - z * se)`` removes a *fixed*
    width ``z * se``, so it costs a fraction ``z / |z_observed|`` of the
    signal. That is 6.6% for ``passer_ast_teammate_pts`` (z = 29.9) but 54.9%
    for ``teammate_reb_reb`` (z = 3.6) and 58.3% for ``teammate_ast_ast``
    (z = 3.4). The two repair targets are simply the weakest *significant*
    same-team buckets, and a fixed-width rule taxes them hardest.

2.  **Rank truncation.** The fitted same-team block is
    ``rank_k(A + B + X) / 2 + rank_r((A + B - X) / 2) - Pi_+(Q)``. V1 uses
    ``k_game = 2`` and a hard-coded rank-1 contrast factor, which cannot
    represent these two diagonal directions. At full rank the construction
    reproduces the shrunk block *exactly*, so this component is a parsimony
    cost rather than a statistical one. The two targets lose through different
    channels: ``reb_reb`` through the rank-1 contrast factor, ``ast_ast``
    through ``k_game``.

Both mechanisms are addressed without introducing a single player- or
pair-indexed parameter: every quantity here is indexed by stat (and optionally
by a coarse role bucket), so :func:`gates` still reports
``pairwise_parameter_count == 0`` and the layer still transfers to unseen
players and rosters by construction.

Empirical-Bayes shrinkage
-------------------------
Treating each stat-pair entry as ``r ~ N(theta, se^2)`` with
``theta ~ N(0, tau_f^2)`` inside a *family* ``f`` gives the posterior mean

    theta_hat = r * tau_f^2 / (tau_f^2 + se^2)

which shrinks *multiplicatively*. Its bias is proportional to the estimate
rather than a fixed width, so a real-but-modest signal keeps most of its
magnitude while a pure-noise entry is still driven to near zero. ``tau_f^2``
comes from the method of moments inside the family,
``tau_f^2 = max(0, mean(r^2 - se^2))``, which is why the families must be
coarse enough to pool but fine enough not to let one dominant entry inflate
``tau`` for everything else. Family membership is a *structural* choice (block
and whether the stat pair is diagonal), never a per-pair choice, so no entry
gets its own tuned prior.
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
    FactorFit,
    PairMoments,
    competition_gate,
    dominating_additive_gram,
    fit_role_scales,
    pair_moments,
    soft_threshold,
)

#: V1 truncates the team-contrast Gram to a single signed factor. Nothing in
#: the identification argument requires that -- only the antisymmetric part
#: ``A - B`` is identified, and ``B`` is a full PSD Gram in general -- so the
#: rank is a free parsimony parameter the repair search is allowed to raise.
DEFAULT_R_CONTRAST = 1

#: Shrinkage methods the repair search may use.
SHRINKAGE_SOFT_THRESHOLD = "soft_threshold"
SHRINKAGE_EMPIRICAL_BAYES = "empirical_bayes"

#: Empirical-Bayes pooling families. ``block`` keeps same-team and cross-team
#: estimates apart because their dependence magnitudes differ; ``diagonal``
#: separates same-stat from different-stat pairs, which otherwise lets one
#: dominant off-diagonal entry inflate ``tau`` for the whole block.
EB_FAMILY_BLOCK = "block"
EB_FAMILY_BLOCK_DIAGONAL = "block_diagonal"
EB_FAMILY_GLOBAL = "global"

#: Floor on a family's prior variance, in squared correlation units. A family
#: whose method-of-moments estimate is non-positive carries no reliable signal,
#: and this keeps the resulting shrinkage factor at zero rather than undefined.
EB_TAU2_FLOOR = 0.0


def empirical_bayes_shrink(
    estimate: np.ndarray,
    standard_error: np.ndarray,
    family: str = EB_FAMILY_BLOCK_DIAGONAL,
    is_same_team: bool = True,
    tau2_override: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Posterior-mean shrinkage of a stat-by-stat correlation block.

    Returns ``(shrunk, diagnostics)``. ``diagnostics`` records each family's
    estimated prior variance and the resulting shrinkage factor range, so the
    strength of the shrinkage is auditable rather than implicit.

    The estimate is symmetric, so pooling uses the upper triangle only; using
    every cell would double-count every off-diagonal entry and bias ``tau``
    toward the off-diagonal magnitude.
    """
    estimate = np.asarray(estimate, dtype=float)
    error = np.asarray(standard_error, dtype=float)
    if estimate.shape != error.shape or estimate.ndim != 2:
        raise ValueError("estimate and standard_error must be square and aligned")
    if not np.all(np.isfinite(error)):
        # Without standard errors there is no evidence on which to shrink.
        return estimate, {"tau2_unavailable": 1.0}

    n = estimate.shape[0]
    rows, cols = np.triu_indices(n)
    diagonal_mask = rows == cols

    if family == EB_FAMILY_GLOBAL:
        groups = {"all": np.ones(len(rows), dtype=bool)}
    elif family == EB_FAMILY_BLOCK:
        label = "same_team" if is_same_team else "cross_team"
        groups = {label: np.ones(len(rows), dtype=bool)}
    elif family == EB_FAMILY_BLOCK_DIAGONAL:
        label = "same_team" if is_same_team else "cross_team"
        groups = {
            f"{label}_diagonal": diagonal_mask,
            f"{label}_offdiagonal": ~diagonal_mask,
        }
    else:
        raise ValueError(f"unknown empirical-Bayes family {family!r}")

    shrunk = np.zeros_like(estimate)
    diagnostics: dict[str, float] = {}

    for name, mask in groups.items():
        if not np.any(mask):
            continue
        r = estimate[rows[mask], cols[mask]]
        se = error[rows[mask], cols[mask]]
        if tau2_override is not None:
            tau2 = float(tau2_override)
        else:
            tau2 = float(max(np.mean(r**2 - se**2), EB_TAU2_FLOOR))
        factor = tau2 / (tau2 + se**2) if tau2 > 0 else np.zeros_like(se)
        shrunk[rows[mask], cols[mask]] = r * factor
        diagnostics[f"tau2_{name}"] = tau2
        diagnostics[f"min_shrink_factor_{name}"] = float(np.min(factor))
        diagnostics[f"max_shrink_factor_{name}"] = float(np.max(factor))

    shrunk = shrunk + shrunk.T - np.diag(np.diag(shrunk))
    return shrunk, diagnostics


@dataclass(frozen=True)
class RepairSpec:
    """A pre-registered bucket-repair candidate.

    ``shrinkage`` selects the estimator applied to the pooled pair moments,
    ``k_game`` and ``r_contrast`` the representation ranks. Every field is a
    scalar or a structural label; none is indexed by player, pair or roster.
    """

    name: str
    family: str
    shrinkage: str = SHRINKAGE_SOFT_THRESHOLD
    shrink_z: float = DEFAULT_SHRINK_Z
    eb_family: str = EB_FAMILY_BLOCK_DIAGONAL
    k_game: int = DEFAULT_K_GAME
    r_contrast: int = DEFAULT_R_CONTRAST
    role_column: str | None = "role_bucket"

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "shrinkage": self.shrinkage,
            "shrink_z": float(self.shrink_z),
            "eb_family": self.eb_family,
            "k_game": int(self.k_game),
            "r_contrast": int(self.r_contrast),
            "role_column": self.role_column,
        }

    def parameter_count(self, n_stats: int) -> dict[str, int]:
        """Free parameters by kind. ``pairwise`` must always be zero."""
        triangle = n_stats * (n_stats + 1) // 2
        return {
            "stat_indexed_game": int(n_stats * self.k_game),
            "stat_indexed_contrast": int(n_stats * self.r_contrast),
            "stat_indexed_competition": int(triangle),
            "scalar_hyperparameters": 2,
            "role_indexed": 0 if self.role_column is None else 3,
            "player_indexed": 0,
            "pairwise": 0,
        }


#: The accepted V1 fit, expressed in this module's vocabulary. Used as the
#: immutable control and asserted by test to reproduce `fit_shared_factors`.
V1_CONTROL_SPEC = RepairSpec(
    name="v1_control",
    family="control",
    shrinkage=SHRINKAGE_SOFT_THRESHOLD,
    shrink_z=DEFAULT_SHRINK_Z,
    k_game=DEFAULT_K_GAME,
    r_contrast=DEFAULT_R_CONTRAST,
)


def shrink_blocks(
    moments: PairMoments,
    spec: RepairSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply ``spec``'s shrinkage to the pooled same-team and cross-team blocks."""
    if spec.shrinkage == SHRINKAGE_SOFT_THRESHOLD:
        same = soft_threshold(moments.same_team, moments.same_team_se, spec.shrink_z)
        cross = soft_threshold(moments.cross_team, moments.cross_team_se, spec.shrink_z)
        return same, cross, {"shrink_z": float(spec.shrink_z)}

    if spec.shrinkage == SHRINKAGE_EMPIRICAL_BAYES:
        same, same_diag = empirical_bayes_shrink(
            moments.same_team,
            moments.same_team_se,
            family=spec.eb_family,
            is_same_team=True,
        )
        cross, cross_diag = empirical_bayes_shrink(
            moments.cross_team,
            moments.cross_team_se,
            family=spec.eb_family,
            is_same_team=False,
        )
        return same, cross, {**same_diag, **cross_diag}

    raise ValueError(f"unknown shrinkage method {spec.shrinkage!r}")


def fit_repaired_factors(
    frame: pd.DataFrame,
    stats: Sequence[str],
    spec: RepairSpec = V1_CONTROL_SPEC,
    bootstrap: int = 200,
    seed: int = 73,
    value_prefix: str = "zs_",
) -> FactorFit:
    """Estimate shared loadings under a bucket-repair specification.

    Mirrors :func:`factors.fit_shared_factors` step for step, differing only in
    the shrinkage estimator and the two representation ranks. With
    :data:`V1_CONTROL_SPEC` the two are identical, which is what makes the
    accepted V1 fit the control rather than a re-derivation.
    """
    stats = tuple(stats)
    moments = pair_moments(
        frame,
        stats,
        value_prefix=value_prefix,
        bootstrap=bootstrap,
        seed=seed,
    )

    same, cross, shrink_diagnostics = shrink_blocks(moments, spec)
    competition_allowed, competition_evidence = competition_gate(moments, same)

    additive = dominating_additive_gram(same, cross)
    competition = (additive - same) if competition_allowed else np.zeros_like(same)

    game_gram = 0.5 * (additive + cross)
    contrast_gram = 0.5 * (additive - cross)

    _, game_loadings = project_psd_rank(game_gram, rank=spec.k_game)
    _, contrast_loadings = project_psd_rank(contrast_gram, rank=spec.r_contrast)
    _, competition_loadings = project_psd_rank(competition, rank=len(stats))

    if not np.any(competition_loadings):
        competition_loadings = None

    # At rank 1 the loadings are stored as a flat vector, which is the format
    # the accepted V1 artifacts use; anything above rank 1 stays a matrix.
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

    role_scale: dict[str, float] = {}
    if spec.role_column is not None:
        role_scale = fit_role_scales(
            frame,
            stats,
            base=base,
            role_column=spec.role_column,
            value_prefix=value_prefix,
        )

    loadings = SharedFactorLoadings(
        stats=stats,
        game=base.game,
        team_contrast=base.team_contrast,
        competition=base.competition,
        role_scale=role_scale,
    )

    fit = FactorFit(
        loadings=loadings,
        moments=moments,
        same_team_shrunk=same,
        cross_team_shrunk=cross,
        game_gram_eigenvalues=np.linalg.eigvalsh(0.5 * (game_gram + game_gram.T))[::-1],
        contrast_gram_eigenvalues=np.linalg.eigvalsh(
            0.5 * (contrast_gram + contrast_gram.T)
        )[::-1],
        competition_gram_eigenvalues=np.linalg.eigvalsh(
            0.5 * (competition + competition.T)
        )[::-1],
        k_game=int(spec.k_game),
        r_competition=0 if loadings.competition is None else loadings.r_competition,
        shrink_z=float(spec.shrink_z),
        competition_evidence={**competition_evidence, **shrink_diagnostics},
    )
    return fit


def attenuation_report(
    moments: PairMoments,
    shrunk: np.ndarray,
    fitted: np.ndarray,
    buckets: Mapping[str, tuple[str, str]],
    standard_error: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-bucket decomposition of observed -> shrunk -> fitted.

    Separates the shrinkage loss from the rank-truncation loss, which is the
    diagnostic that distinguishes "the threshold ate the signal" from "the
    representation cannot express it".
    """
    index = {stat: position for position, stat in enumerate(moments.stats)}
    out: dict[str, dict[str, float]] = {}
    for name, (first, second) in buckets.items():
        if first not in index or second not in index:
            continue
        i, j = index[first], index[second]
        observed = float(moments.same_team[i, j])
        se = float(standard_error[i, j])
        after = float(shrunk[i, j])
        final = float(fitted[i, j])
        scale = abs(observed) if abs(observed) > 1e-12 else 1.0
        out[name] = {
            "observed": observed,
            "standard_error": se,
            "signal_to_se": observed / se if se > 0 else float("nan"),
            "after_shrinkage": after,
            "fitted": final,
            "shrinkage_loss_fraction": (observed - after) / scale,
            "rank_loss_fraction": (after - final) / scale,
            "total_loss_fraction": (observed - final) / scale,
        }
    return out
