"""Final upstream remediation estimators for the latent-state shadow layer.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The accepted bucket repair is the control. Every estimator here is an
*alternative* to one piece of it, selected on strictly pre-2024 chronological
folds, and :data:`CONTROL_SPEC` reproduces the accepted repair exactly so the
control is the same object rather than a re-derivation.

Six pieces, in the order the brief states them.

1.  **Robust season random effects.** A bucket's season estimates are
    ``r_s ~ N(theta_s, se_s^2)`` with ``theta_s ~ StudentT(nu, mu, tau)``. The
    accepted repair uses a fixed effect (all heterogeneity treated as noise);
    the Gaussian random-effects alternative lets ``tau > 0``; the Student-t
    alternative additionally stops one departing season from dragging the
    level. Which one is used is decided by forward predictive score, not by
    which looks better on the holdout.

2.  **Hierarchically shrunk role scale.** The accepted repair estimates the
    multiplicative ``role_scale`` as a ratio, partially pooled by game count.
    That pools toward 1.0 on a *linear* scale, which is the wrong geometry for
    a multiplier: 0.5 and 2.0 are equally far from neutral but the linear rule
    treats 0.5 as closer. Writing ``s_r = exp(delta_r)`` with
    ``delta_r ~ N(0, tau_role^2)`` shrinks symmetrically in the ratio and makes
    the pooling strength an estimated variance rather than a declared constant.
    Both forms impose the same weighted normalisation, so the pooled same-team
    block is untouched either way.

3.  **Robust cross-team shrinkage.** Same-team and cross-team blocks already
    get independently estimated Gaussian priors. A Gaussian prior shrinks two
    entries with equal standard errors by the *same* factor whatever their
    magnitudes, so it cannot tell a strong cross-team entry from a moderate
    one. A Student-t prior can: it shrinks moderate signals harder and leaves
    strong ones nearly alone. The posterior mean under the t prior has no
    closed form and is taken by quadrature.

4.  **Latent-to-count transmission.** See :mod:`transmission`.

5.  **Dependence temperature.** One scalar ``lambda`` in ``[0, 1]`` scaling
    every shared cross-player loading by ``sqrt(lambda)``. Because the
    covariance construction pins each player's own block to the incumbent
    correlation *after* the shared contribution is removed, scaling the
    loadings moves cross-player blocks to ``lambda`` times their value and
    leaves every same-player block exactly where it was. ``lambda = 1`` is the
    accepted repair and ``lambda = 0`` is cross-player conditional
    independence, so the incumbent is inside the family rather than outside it.

6.  **Forward-cross-fitted uncertainty scale.** A reported predictive standard
    deviation is judged by coverage, not by whether its mean squared z lands on
    1. The candidates are a raw model scale, a Huber M-scale, a median of
    per-fold scales and a Student-t predictive scale; each is estimated on
    folds strictly earlier than the one it is scored on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .covariance import SharedFactorLoadings, project_psd_rank
from .factors import FactorFit, PairMoments, competition_gate, dominating_additive_gram
from .repair import (
    EB_FAMILY_BLOCK_DIAGONAL,
    SHRINKAGE_EMPIRICAL_BAYES,
    empirical_bayes_shrink,
)

# ----------------------------------------------------------------------
# declared constants
# ----------------------------------------------------------------------

#: Degrees of freedom offered to the robust estimators. 2 is excluded because
#: a Student-t with two degrees of freedom has no finite variance, so its
#: predictive scale would be undefined; 40 is numerically Gaussian and is
#: carried so the robust family nests the Gaussian one.
DEFAULT_NU_GRID: tuple[float, ...] = (3.0, 5.0, 8.0, 15.0, 40.0)

#: Half-lives, in seasons, offered to the recency-weighted candidate.
DEFAULT_HALF_LIFE_GRID: tuple[float, ...] = (1.0, 2.0, 4.0)

#: A temporal or uncertainty candidate is only preferred over a simpler one
#: when it beats it by more than this many standard errors of the paired
#: difference. Inside the band the simpler model is kept.
PARSIMONY_TIE_BAND_SE = 1.0

#: Fewer than this many seasons in a fold's training window leaves ``tau``
#: unidentified, so the random-effects candidates are not scored there.
MIN_SEASONS_FOR_HETEROGENEITY = 2

#: Quadrature half-width, in prior scales, for the Student-t posterior mean.
T_POSTERIOR_QUADRATURE_WIDTH = 12.0
T_POSTERIOR_QUADRATURE_NODES = 801

#: Coverage levels the uncertainty candidates are judged on.
COVERAGE_LEVELS: tuple[float, ...] = (0.68, 0.80, 0.90, 0.95)

#: Ordered role-pair cells the role-scale layer is evaluated on.
ROLE_PAIR_CELLS: tuple[tuple[str, str], ...] = (
    ("starter", "starter"),
    ("starter", "rotation"),
    ("starter", "bench"),
    ("rotation", "rotation"),
    ("rotation", "bench"),
    ("bench", "bench"),
)

#: A role-pair cell needs at least this many games and ordered pairs before it
#: is read as evidence rather than shrunk to the pooled value.
MIN_CELL_GAMES = 200
MIN_CELL_PAIRS = 2000


# ======================================================================
# 1. season-level random effects
# ======================================================================


@dataclass(frozen=True)
class SeasonSeries:
    """One bucket's per-season estimates with their standard errors."""

    name: str
    seasons: tuple[int, ...]
    estimates: np.ndarray
    standard_errors: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.seasons) == len(self.estimates) == len(self.standard_errors)):
            raise ValueError("seasons, estimates and standard errors must align")

    def through(self, season: int) -> SeasonSeries:
        """The sub-series of seasons strictly before ``season``."""
        keep = [i for i, s in enumerate(self.seasons) if s < season]
        return SeasonSeries(
            name=self.name,
            seasons=tuple(self.seasons[i] for i in keep),
            estimates=self.estimates[keep],
            standard_errors=self.standard_errors[keep],
        )

    def at(self, season: int) -> tuple[float, float]:
        position = self.seasons.index(season)
        return float(self.estimates[position]), float(self.standard_errors[position])

    @property
    def n_seasons(self) -> int:
        return len(self.seasons)


@dataclass(frozen=True)
class TemporalFit:
    """A level estimate for one bucket, with its heterogeneity and spread."""

    model: str
    posterior_mean: float
    posterior_sd: float
    predictive_sd: float
    tau: float
    nu: float | None
    half_life: float | None
    probability_positive: float
    predictive_probability_positive: float
    stability: str
    q_statistic: float
    q_p_value: float
    i_squared: float
    weights: tuple[float, ...]

    def payload(self) -> dict[str, object]:
        return {
            "model": self.model,
            "posterior_mean": self.posterior_mean,
            "posterior_sd": self.posterior_sd,
            "predictive_sd": self.predictive_sd,
            "tau": self.tau,
            "nu": self.nu,
            "half_life": self.half_life,
            "probability_rho_positive": self.probability_positive,
            "predictive_probability_rho_positive": (
                self.predictive_probability_positive
            ),
            "stability_classification": self.stability,
            "q_statistic": self.q_statistic,
            "q_p_value": self.q_p_value,
            "i_squared": self.i_squared,
            "weights": list(self.weights),
        }


def _cochran_q(series: SeasonSeries) -> tuple[float, float, float]:
    """Cochran's Q, its p-value and I^2 for a season series."""
    if series.n_seasons < 2:
        return 0.0, 1.0, 0.0
    precision = 1.0 / np.maximum(series.standard_errors**2, 1e-24)
    pooled = float(np.sum(precision * series.estimates) / np.sum(precision))
    q = float(np.sum(precision * (series.estimates - pooled) ** 2))
    df = series.n_seasons - 1
    p_value = float(stats.chi2.sf(q, df))
    i_squared = float(max(0.0, (q - df) / q)) if q > 0 else 0.0
    return q, p_value, i_squared


def _classify(q_p_value: float, i_squared: float) -> str:
    """Stability label. Thresholds are declared, not tuned."""
    if q_p_value < 0.05 and i_squared >= 0.5:
        return "TIME_VARYING"
    if q_p_value < 0.20 or i_squared >= 0.25:
        return "UNCERTAIN"
    return "STABLE"


def _assemble(
    model: str,
    series: SeasonSeries,
    mean: float,
    variance: float,
    tau: float,
    weights: np.ndarray,
    nu: float | None = None,
    half_life: float | None = None,
    predictive_variance: float | None = None,
) -> TemporalFit:
    q, p_value, i_squared = _cochran_q(series)
    posterior_sd = float(np.sqrt(max(variance, 0.0)))
    if predictive_variance is None:
        predictive_variance = variance + tau**2
    predictive_sd = float(np.sqrt(max(predictive_variance, 0.0)))
    return TemporalFit(
        model=model,
        posterior_mean=float(mean),
        posterior_sd=posterior_sd,
        predictive_sd=predictive_sd,
        tau=float(tau),
        nu=None if nu is None else float(nu),
        half_life=None if half_life is None else float(half_life),
        probability_positive=float(
            stats.norm.cdf(mean / posterior_sd) if posterior_sd > 0 else float(mean > 0)
        ),
        predictive_probability_positive=float(
            stats.norm.cdf(mean / predictive_sd)
            if predictive_sd > 0
            else float(mean > 0)
        ),
        stability=_classify(p_value, i_squared),
        q_statistic=q,
        q_p_value=p_value,
        i_squared=i_squared,
        weights=tuple(float(value) for value in weights),
    )


def fit_pooled(series: SeasonSeries) -> TemporalFit:
    """A0: the accepted repair's fixed effect, all heterogeneity as noise.

    ``tau`` is not zero in the *report* -- the heterogeneity is still measured
    and carried into the predictive spread, which is what the accepted repair
    does -- but it does not enter the weights.
    """
    precision = 1.0 / np.maximum(series.standard_errors**2, 1e-24)
    total = float(np.sum(precision))
    mean = float(np.sum(precision * series.estimates) / total)
    q, _, _ = _cochran_q(series)
    df = max(series.n_seasons - 1, 1)
    # DerSimonian-Laird heterogeneity, reported but unused in the weights.
    scale = total - float(np.sum(precision**2)) / total
    tau2 = max(0.0, (q - df) / scale) if scale > 0 else 0.0
    return _assemble(
        "A0_pooled_empirical_bayes",
        series,
        mean,
        1.0 / total,
        math.sqrt(tau2),
        precision / total,
    )


def fit_gaussian_random_effects(series: SeasonSeries) -> TemporalFit:
    """A1: ``theta_s ~ N(mu, tau^2)`` with DerSimonian-Laird ``tau^2``."""
    precision = 1.0 / np.maximum(series.standard_errors**2, 1e-24)
    total = float(np.sum(precision))
    q, _, _ = _cochran_q(series)
    df = max(series.n_seasons - 1, 1)
    scale = total - float(np.sum(precision**2)) / total
    tau2 = max(0.0, (q - df) / scale) if scale > 0 else 0.0
    weights = 1.0 / (series.standard_errors**2 + tau2)
    total_w = float(np.sum(weights))
    mean = float(np.sum(weights * series.estimates) / total_w)
    return _assemble(
        "A1_gaussian_random_effects",
        series,
        mean,
        1.0 / total_w,
        math.sqrt(tau2),
        weights / total_w,
    )


def _t_profile_negative_log_likelihood(
    tau: float,
    series: SeasonSeries,
    nu: float,
    weights: np.ndarray,
) -> float:
    """Profiled negative log likelihood of ``tau`` under a Student-t level.

    The marginal for season ``s`` is taken as ``t_nu(mu, v_s)`` with
    ``v_s^2 = se_s^2 + tau^2``, the standard robust meta-analysis form. ``mu``
    is profiled out by the IRLS fixed point below.
    """
    scale = np.sqrt(series.standard_errors**2 + tau**2)
    mean = _irls_t_location(series.estimates, scale, nu, weights)
    standardized = (series.estimates - mean) / scale
    return float(
        -np.sum(weights * (stats.t.logpdf(standardized, df=nu) - np.log(scale)))
    )


def _irls_t_location(
    estimates: np.ndarray,
    scale: np.ndarray,
    nu: float,
    weights: np.ndarray,
    iterations: int = 200,
) -> float:
    """Weighted Student-t location by iteratively reweighted least squares.

    The t score equation is a weighted mean with weights
    ``(nu + 1) / (nu + d_s)`` over ``se_s^2 + tau^2``, where ``d_s`` is the
    squared standardized residual. Starting from the Gaussian solution, the
    fixed point converges monotonically for ``nu > 0``.
    """
    precision = weights / scale**2
    mean = float(np.sum(precision * estimates) / np.sum(precision))
    for _ in range(iterations):
        d = ((estimates - mean) / scale) ** 2
        robust = precision * (nu + 1.0) / (nu + d)
        updated = float(np.sum(robust * estimates) / np.sum(robust))
        if abs(updated - mean) < 1e-14:
            mean = updated
            break
        mean = updated
    return mean


def fit_student_t_random_effects(
    series: SeasonSeries,
    nu: float = 5.0,
    half_life: float | None = None,
) -> TemporalFit:
    """A2/A3: ``theta_s ~ t_nu(mu, tau)``, optionally recency-weighted.

    ``half_life`` is A3: each season's contribution is multiplied by
    ``0.5 ** (age / half_life)``, which down-weights old seasons without
    discarding them. ``half_life = None`` is A2.
    """
    if half_life is None:
        weights = np.ones(series.n_seasons, dtype=float)
    else:
        age = float(max(series.seasons)) - np.asarray(series.seasons, dtype=float)
        weights = 0.5 ** (age / float(half_life))

    upper = float(max(4.0 * np.max(series.standard_errors), 1e-6))
    result = optimize.minimize_scalar(
        _t_profile_negative_log_likelihood,
        bounds=(0.0, upper),
        args=(series, nu, weights),
        method="bounded",
        options={"xatol": 1e-12},
    )
    tau = float(max(result.x, 0.0))
    scale = np.sqrt(series.standard_errors**2 + tau**2)
    mean = _irls_t_location(series.estimates, scale, nu, weights)

    d = ((series.estimates - mean) / scale) ** 2
    robust = (weights / scale**2) * (nu + 1.0) / (nu + d)
    variance = 1.0 / float(np.sum(robust))

    # A Student-t with nu degrees of freedom and scale tau has variance
    # tau^2 * nu / (nu - 2); the predictive spread for an unseen season is
    # that plus the uncertainty in the level.
    inflation = nu / (nu - 2.0) if nu > 2.0 else np.inf
    predictive_variance = variance + (tau**2) * inflation

    return _assemble(
        "A3_recency_weighted_robust" if half_life else "A2_robust_student_t",
        series,
        mean,
        variance,
        tau,
        robust / float(np.sum(robust)),
        nu=nu,
        half_life=half_life,
        predictive_variance=predictive_variance,
    )


#: The temporal candidates, keyed by the label the report uses.
def temporal_candidates(
    nu: float = 5.0,
    half_life: float = 2.0,
) -> dict[str, object]:
    return {
        "A0_pooled_empirical_bayes": fit_pooled,
        "A1_gaussian_random_effects": fit_gaussian_random_effects,
        "A2_robust_student_t": lambda s: fit_student_t_random_effects(s, nu=nu),
        "A3_recency_weighted_robust": lambda s: fit_student_t_random_effects(
            s, nu=nu, half_life=half_life
        ),
    }


def forward_fold_seasons(
    seasons: Sequence[int],
    min_training_seasons: int = MIN_SEASONS_FOR_HETEROGENEITY,
) -> tuple[int, ...]:
    """Target seasons of the strictly forward chronological folds."""
    ordered = sorted(int(season) for season in seasons)
    return tuple(ordered[min_training_seasons:])


def forward_predictive_records(
    series: SeasonSeries,
    fitter,
    min_training_seasons: int = MIN_SEASONS_FOR_HETEROGENEITY,
) -> list[dict[str, float]]:
    """Walk-forward predictive records for one bucket under one candidate.

    Each record scores a season that the fit never saw. ``log_score`` is the
    Gaussian predictive negative log density, which is a proper score; the
    squared standardized error is reported alongside because that is what the
    coverage work in part 6 reads.
    """
    records: list[dict[str, float]] = []
    for target in forward_fold_seasons(series.seasons, min_training_seasons):
        history = series.through(target)
        if history.n_seasons < min_training_seasons:
            continue
        fit = fitter(history)
        observed, observed_se = series.at(target)
        variance = fit.predictive_sd**2 + observed_se**2
        error = observed - fit.posterior_mean
        records.append(
            {
                "bucket": series.name,
                "target_season": float(target),
                "training_seasons": float(history.n_seasons),
                "prediction": fit.posterior_mean,
                "predictive_sd": fit.predictive_sd,
                "observed": observed,
                "observed_se": observed_se,
                "error": error,
                "squared_z": error**2 / variance if variance > 0 else np.nan,
                "log_score": 0.5 * math.log(2.0 * math.pi * variance)
                + 0.5 * error**2 / variance,
                "tau": fit.tau,
            }
        )
    return records


def cluster_mean_and_se(
    values: Sequence[float],
    clusters: Sequence[str],
) -> tuple[float, float]:
    """Mean of ``values`` with a standard error clustered on ``clusters``.

    The fold-level records for one bucket are not independent of each other --
    they share training seasons -- so the bucket is the independent unit and
    the standard error has to be taken between buckets rather than between
    records.
    """
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "cluster": list(clusters)})
    frame = frame.dropna()
    if frame.empty:
        return float("nan"), float("nan")
    per_cluster = frame.groupby("cluster")["value"].mean()
    mean = float(per_cluster.mean())
    if len(per_cluster) < 2:
        return mean, float("nan")
    return mean, float(per_cluster.std(ddof=1) / math.sqrt(len(per_cluster)))


def paired_difference_se(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
) -> tuple[float, float]:
    """Mean paired difference and its between-bucket standard error.

    Pairing matters: the two candidates are scored on the *same* fold-bucket
    observations, so the difference has far less variance than either level.
    """
    keys = sorted(set(candidate) & set(reference))
    if not keys:
        return float("nan"), float("nan")
    differences = np.array([candidate[key] - reference[key] for key in keys])
    mean = float(np.mean(differences))
    if len(differences) < 2:
        return mean, float("nan")
    return mean, float(np.std(differences, ddof=1) / math.sqrt(len(differences)))


def select_within_tie_band(
    scores: Mapping[str, float],
    order: Sequence[str],
    per_bucket: Mapping[str, Mapping[str, float]],
    tie_band_se: float = PARSIMONY_TIE_BAND_SE,
) -> dict[str, object]:
    """Keep the simplest candidate within ``tie_band_se`` of the best score.

    ``order`` is the simplicity order, simplest first. Lower scores are
    better. The comparison is paired per bucket, so the tie band is measured
    on the difference rather than on two independent levels.
    """
    usable = {name: value for name, value in scores.items() if np.isfinite(value)}
    if not usable:
        return {"selected": None, "reason": "no candidate produced a finite score"}
    best = min(usable, key=lambda name: usable[name])
    for name in order:
        if name not in usable:
            continue
        difference, se = paired_difference_se(per_bucket[name], per_bucket[best])
        if name == best:
            return {
                "selected": name,
                "best": best,
                "reason": "this candidate has the best score",
                "difference_from_best": 0.0,
                "difference_se": se,
            }
        if not np.isfinite(se) or se <= 0:
            continue
        if difference <= tie_band_se * se:
            return {
                "selected": name,
                "best": best,
                "reason": (
                    f"simplest candidate within {tie_band_se} SE of {best}: "
                    f"worse by {difference:.6g} +/- {se:.6g}"
                ),
                "difference_from_best": float(difference),
                "difference_se": float(se),
            }
    return {
        "selected": best,
        "best": best,
        "reason": "no simpler candidate is within the tie band",
        "difference_from_best": 0.0,
    }


# ======================================================================
# 2. hierarchically shrunk role scale
# ======================================================================


def role_pair_cell_moments(
    frame: pd.DataFrame,
    stats: Sequence[str],
    role_column: str = "role_bucket",
    value_prefix: str = "zs_",
) -> dict[tuple[str, str], dict[str, object]]:
    """Observed same-team moments split by the ordered role pair.

    The pooled same-team bucket averages over every role pair at once, so it
    cannot see a layer whose weighted mean is 1 by construction. These cells
    can. The accumulation is the same team-sum identity the pooled version
    uses, applied within each role rather than across the whole team, so a cell
    costs O(n) in roster size and the game stays the independent unit.
    """
    stats = tuple(stats)
    columns = [f"{value_prefix}{stat}" for stat in stats]
    usable = frame.dropna(subset=["game_id", "team_id", role_column, *columns])
    roles = sorted(str(value) for value in usable[role_column].unique())
    n = len(stats)

    accumulated: dict[tuple[str, str], np.ndarray] = {
        (a, b): np.zeros((n, n)) for a in roles for b in roles
    }
    pairs: dict[tuple[str, str], float] = {key: 0.0 for key in accumulated}
    games: dict[tuple[str, str], set[int]] = {key: set() for key in accumulated}

    for game_id, game in usable.groupby("game_id", sort=True):
        for _, team in game.groupby("team_id", sort=True):
            totals: dict[str, np.ndarray] = {}
            own: dict[str, np.ndarray] = {}
            counts: dict[str, int] = {}
            for role, members in team.groupby(role_column, sort=True):
                values = members[columns].to_numpy(dtype=float)
                label = str(role)
                totals[label] = values.sum(axis=0)
                own[label] = values.T @ values
                counts[label] = values.shape[0]
            for first in totals:
                for second in totals:
                    block = np.outer(totals[first], totals[second])
                    count = counts[first] * counts[second]
                    if first == second:
                        block = block - own[first]
                        count -= counts[first]
                    if count <= 0:
                        continue
                    accumulated[(first, second)] += block
                    pairs[(first, second)] += count
                    games[(first, second)].add(int(game_id))

    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, total in accumulated.items():
        count = pairs[key]
        if count <= 0:
            continue
        matrix = total / count
        out[key] = {
            "correlation": 0.5 * (matrix + matrix.T),
            "pairs": float(count),
            "games": int(len(games[key])),
            "supported": bool(
                len(games[key]) >= MIN_CELL_GAMES and count >= MIN_CELL_PAIRS
            ),
        }
    return out


def per_game_role_cells(
    frame: pd.DataFrame,
    stats: Sequence[str],
    role_column: str = "role_bucket",
    value_prefix: str = "zs_",
) -> dict[tuple[str, str], dict[str, object]]:
    """:func:`role_pair_cell_moments` with the per-game terms kept.

    The accumulation is the same team-sum identity inside each ordered role
    pair, summed in the same game order, so ``correlation`` agrees with
    :func:`role_pair_cell_moments` entry for entry. What this adds is the
    per-game blocks and pair counts, which is what a game-clustered standard
    error and an effective sample size need; the pooled form discards them.
    """
    stats = tuple(stats)
    columns = [f"{value_prefix}{stat}" for stat in stats]
    usable = frame.dropna(subset=["game_id", "team_id", role_column, *columns])
    roles = sorted(str(value) for value in usable[role_column].unique())
    width = len(stats)

    blocks: dict[tuple[str, str], list[np.ndarray]] = {
        (a, b): [] for a in roles for b in roles
    }
    counts: dict[tuple[str, str], list[float]] = {key: [] for key in blocks}

    for _, game in usable.groupby("game_id", sort=True):
        game_block: dict[tuple[str, str], np.ndarray] = {}
        game_count: dict[tuple[str, str], float] = {}
        for _, team in game.groupby("team_id", sort=True):
            totals: dict[str, np.ndarray] = {}
            own: dict[str, np.ndarray] = {}
            size: dict[str, int] = {}
            for role, members in team.groupby(role_column, sort=True):
                values = members[columns].to_numpy(dtype=float)
                label = str(role)
                totals[label] = values.sum(axis=0)
                own[label] = values.T @ values
                size[label] = values.shape[0]
            for first in totals:
                for second in totals:
                    block = np.outer(totals[first], totals[second])
                    count = size[first] * size[second]
                    if first == second:
                        block = block - own[first]
                        count -= size[first]
                    if count <= 0:
                        continue
                    key = (first, second)
                    if key in game_block:
                        game_block[key] = game_block[key] + block
                        game_count[key] = game_count[key] + count
                    else:
                        game_block[key] = block
                        game_count[key] = float(count)
        for key, block in game_block.items():
            blocks[key].append(block)
            counts[key].append(game_count[key])

    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, terms in blocks.items():
        if not terms:
            continue
        stacked = np.stack(terms)
        count = np.asarray(counts[key], dtype=float)
        total = float(count.sum())
        if total <= 0:
            continue
        matrix = stacked.sum(axis=0) / total
        out[key] = {
            "correlation": 0.5 * (matrix + matrix.T),
            "per_game_blocks": stacked,
            "per_game_counts": count,
            "pairs": total,
            "games": int(len(count)),
            # Kish effective number of clusters for a pair-count-weighted mean
            # over games. The cell's estimate is one such mean, so this is the
            # number of equally sized games carrying the same weight
            # concentration -- not a claim about independence across stats.
            "effective_games": float(total**2 / float(np.sum(np.square(count)))),
            "supported": bool(
                len(count) >= MIN_CELL_GAMES and total >= MIN_CELL_PAIRS
            ),
        }
    return out


def cell_standard_errors(
    cell: Mapping[str, object],
    draws: int = 400,
    seed: int = 73,
) -> np.ndarray:
    """Game-clustered bootstrap standard error of every entry of one cell.

    Games are the resampling unit because a roster's residuals are shared
    within a game; resampling player-rows would treat the same game's pairs as
    independent and understate the error by roughly the roster size.
    """
    blocks = np.asarray(cell["per_game_blocks"], dtype=float)
    counts = np.asarray(cell["per_game_counts"], dtype=float)
    rng = np.random.default_rng(seed)
    games = len(counts)
    drawn = np.empty((draws, blocks.shape[1], blocks.shape[2]))
    for draw in range(draws):
        picks = rng.integers(0, games, size=games)
        matrix = blocks[picks].sum(axis=0) / counts[picks].sum()
        drawn[draw] = 0.5 * (matrix + matrix.T)
    return drawn.std(axis=0, ddof=1)


def role_scale_components(
    frame: pd.DataFrame,
    stats: Sequence[str],
    base: SharedFactorLoadings,
    role_column: str = "role_bucket",
    value_prefix: str = "zs_",
    bootstrap: int = 200,
    seed: int = 73,
    per_role_moments: Mapping[str, PairMoments] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Raw role scales, their log standard errors and the player shares.

    The raw scale is the accepted repair's: the projection of the within-role
    observed same-team block onto the model-implied one, square-rooted because
    a pair carries two of them. The standard error is new and is what the
    log-scale shrinkage needs -- the accepted repair pools by game count
    instead, which is a declared constant rather than a measured uncertainty.

    ``per_role_moments`` lets a grid search supply the within-role moments once.
    Only the *projection* depends on the candidate; the observed blocks and
    their bootstrap draws do not, and they are what costs anything here.
    """
    from .factors import pair_moments

    stats = tuple(stats)
    implied = base.same_team_correlation()
    denominator = float(np.sum(implied**2))
    if denominator <= 0:
        return {}, {}, {}

    labelled = frame.dropna(subset=[role_column])
    raw: dict[str, float] = {}
    log_se: dict[str, float] = {}
    shares: dict[str, float] = {}

    for role, group in labelled.groupby(role_column):
        if per_role_moments is not None:
            observed = per_role_moments.get(str(role))
            if observed is None:
                continue
        else:
            try:
                observed = pair_moments(
                    group,
                    stats,
                    value_prefix=value_prefix,
                    bootstrap=bootstrap,
                    seed=seed,
                    keep_draws=True,
                )
            except ValueError:
                continue
        numerator = float(np.sum(observed.same_team * implied))
        if numerator <= 0:
            continue
        raw[str(role)] = float(np.sqrt(numerator / denominator))
        shares[str(role)] = float(len(group))

        draws = observed.same_team_draws
        if draws is None:
            log_se[str(role)] = 0.0
            continue
        projected = np.einsum("dst,st->d", draws, implied) / denominator
        positive = projected[projected > 0]
        if positive.size < 2:
            log_se[str(role)] = 0.0
        else:
            # The scale is the square root of the projection, so its log
            # standard error is half the projection's on the log scale.
            log_se[str(role)] = float(0.5 * np.std(np.log(positive), ddof=1))

    return raw, log_se, shares


@dataclass(frozen=True)
class RoleScaleFit:
    mode: str
    scales: Mapping[str, float]
    raw_scales: Mapping[str, float]
    log_standard_errors: Mapping[str, float]
    tau_log: float
    player_shares: Mapping[str, float]
    normalisation_residual: float

    def payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "scales": {k: float(v) for k, v in self.scales.items()},
            "raw_scales": {k: float(v) for k, v in self.raw_scales.items()},
            "log_standard_errors": {
                k: float(v) for k, v in self.log_standard_errors.items()
            },
            "tau_log": float(self.tau_log),
            "player_shares": {k: float(v) for k, v in self.player_shares.items()},
            "weighted_mean_scale_minus_one": float(self.normalisation_residual),
        }


def fit_log_shrunk_role_scales(
    raw_scales: Mapping[str, float],
    log_standard_errors: Mapping[str, float],
    player_shares: Mapping[str, float],
    tau_log: float | None = None,
) -> RoleScaleFit:
    """Shrink multiplicative role scales symmetrically in the log ratio.

    ``delta_r = log s_r`` with ``delta_r ~ N(0, tau_log^2)`` gives the
    posterior mean ``delta_hat_r = delta_r * tau^2 / (tau^2 + selog_r^2)``,
    and ``tau^2`` comes from the method of moments across roles. The result is
    renormalised so the player-weighted mean scale is exactly 1, which is what
    keeps the pooled same-team block where the base loadings put it; the
    residual of that constraint is reported rather than assumed.
    """
    roles = sorted(raw_scales)
    if not roles:
        return RoleScaleFit("log_shrunk", {}, {}, {}, 0.0, {}, 0.0)

    delta = np.array([math.log(max(raw_scales[role], 1e-9)) for role in roles])
    se = np.array([max(log_standard_errors.get(role, 0.0), 0.0) for role in roles])

    if tau_log is None:
        tau2 = float(max(np.mean(delta**2 - se**2), 0.0))
    else:
        tau2 = float(tau_log) ** 2
    factor = tau2 / (tau2 + se**2) if tau2 > 0 else np.zeros_like(se)
    shrunk = np.exp(delta * factor)

    total = sum(player_shares.get(role, 0.0) for role in roles)
    if total <= 0:
        weights = np.full(len(roles), 1.0 / len(roles))
    else:
        weights = np.array([player_shares.get(role, 0.0) / total for role in roles])
    mean_scale = float(np.sum(weights * shrunk))
    if mean_scale <= 0:
        return RoleScaleFit("log_shrunk", {}, dict(raw_scales), dict(log_standard_errors), 0.0, {}, 0.0)
    scales = {role: float(value / mean_scale) for role, value in zip(roles, shrunk)}

    return RoleScaleFit(
        mode="log_shrunk",
        scales=scales,
        raw_scales={role: float(raw_scales[role]) for role in roles},
        log_standard_errors={role: float(value) for role, value in zip(roles, se)},
        tau_log=math.sqrt(tau2),
        player_shares={role: float(value) for role, value in zip(roles, weights)},
        normalisation_residual=float(
            np.sum(weights * np.array([scales[role] for role in roles])) - 1.0
        ),
    )


# ======================================================================
# 3. robust cross-team shrinkage
# ======================================================================


def student_t_posterior_mean(
    estimate: np.ndarray,
    standard_error: np.ndarray,
    tau: float,
    nu: float,
) -> np.ndarray:
    """Posterior mean of ``theta`` under ``theta ~ t_nu(0, tau)``.

    No closed form exists, so the one-dimensional integral is taken on a fixed
    Simpson grid wide enough to contain the posterior for every entry. Unlike
    the Gaussian case the shrinkage factor is *not* a function of the standard
    error alone: a large ``|r|`` is read as signal and kept, a moderate one as
    noise and shrunk, which is the whole point of using a heavy tail here.
    """
    estimate = np.asarray(estimate, dtype=float)
    error = np.asarray(standard_error, dtype=float)
    if tau <= 0:
        return np.zeros_like(estimate)

    width = T_POSTERIOR_QUADRATURE_WIDTH * max(
        float(tau), float(np.nanmax(error)) if np.any(np.isfinite(error)) else 0.0
    )
    nodes = np.linspace(-width, width, T_POSTERIOR_QUADRATURE_NODES)
    prior = stats.t.pdf(nodes / tau, df=nu) / tau

    flat_estimate = estimate.ravel()[:, None]
    flat_error = np.maximum(error.ravel()[:, None], 1e-12)
    likelihood = np.exp(
        -0.5 * ((flat_estimate - nodes[None, :]) / flat_error) ** 2
    ) / flat_error
    weight = likelihood * prior[None, :]

    mass = np.trapezoid(weight, nodes, axis=1)
    first = np.trapezoid(weight * nodes[None, :], nodes, axis=1)
    out = np.where(mass > 0, first / np.maximum(mass, 1e-300), 0.0)
    return out.reshape(estimate.shape)


def _t_marginal_negative_log_likelihood(
    tau: float,
    estimate: np.ndarray,
    error: np.ndarray,
    nu: float,
) -> float:
    """Marginal likelihood of ``tau`` under a t prior, by the same quadrature."""
    if tau <= 0:
        return float(
            -np.sum(stats.norm.logpdf(estimate, scale=np.maximum(error, 1e-12)))
        )
    width = T_POSTERIOR_QUADRATURE_WIDTH * max(float(tau), float(np.max(error)))
    nodes = np.linspace(-width, width, T_POSTERIOR_QUADRATURE_NODES)
    prior = stats.t.pdf(nodes / tau, df=nu) / tau
    likelihood = np.exp(
        -0.5 * ((estimate[:, None] - nodes[None, :]) / np.maximum(error[:, None], 1e-12)) ** 2
    ) / np.maximum(error[:, None], 1e-12)
    mass = np.trapezoid(likelihood * prior[None, :], nodes, axis=1)
    return float(-np.sum(np.log(np.maximum(mass, 1e-300))))


def robust_shrink_block(
    estimate: np.ndarray,
    standard_error: np.ndarray,
    nu: float,
    family: str = EB_FAMILY_BLOCK_DIAGONAL,
    is_same_team: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Student-t empirical-Bayes shrinkage of one correlation block.

    Pooling families are the accepted repair's, so the only change is the
    shape of the prior inside each family. ``tau`` is the maximum-marginal-
    likelihood estimate within the family, which is the t analogue of the
    method-of-moments estimate the Gaussian version uses.
    """
    estimate = np.asarray(estimate, dtype=float)
    error = np.asarray(standard_error, dtype=float)
    if not np.all(np.isfinite(error)):
        return estimate, {"tau_unavailable": 1.0}

    n = estimate.shape[0]
    rows, cols = np.triu_indices(n)
    diagonal = rows == cols
    label = "same_team" if is_same_team else "cross_team"
    if family == EB_FAMILY_BLOCK_DIAGONAL:
        groups = {
            f"{label}_diagonal": diagonal,
            f"{label}_offdiagonal": ~diagonal,
        }
    else:
        groups = {label: np.ones(len(rows), dtype=bool)}

    shrunk = np.zeros_like(estimate)
    diagnostics: dict[str, float] = {"nu": float(nu)}
    for name, mask in groups.items():
        if not np.any(mask):
            continue
        r = estimate[rows[mask], cols[mask]]
        se = error[rows[mask], cols[mask]]
        upper = float(max(4.0 * np.max(np.abs(r)), 4.0 * np.max(se), 1e-6))
        result = optimize.minimize_scalar(
            _t_marginal_negative_log_likelihood,
            bounds=(0.0, upper),
            args=(r, se, nu),
            method="bounded",
            options={"xatol": 1e-14},
        )
        tau = float(max(result.x, 0.0))
        posterior = student_t_posterior_mean(r, se, tau=tau, nu=nu)
        shrunk[rows[mask], cols[mask]] = posterior
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(r) > 1e-12, posterior / r, 1.0)
        diagnostics[f"tau_{name}"] = tau
        diagnostics[f"min_shrink_factor_{name}"] = float(np.min(ratio))
        diagnostics[f"max_shrink_factor_{name}"] = float(np.max(ratio))

    shrunk = shrunk + shrunk.T - np.diag(np.diag(shrunk))
    return shrunk, diagnostics


# ======================================================================
# 5. dependence temperature
# ======================================================================


def temper_loadings(
    loadings: SharedFactorLoadings,
    temperature: float,
) -> SharedFactorLoadings:
    """Scale every shared cross-player loading by ``sqrt(temperature)``.

    Each cross-player block is quadratic in the loadings, so this multiplies
    every cross-player correlation by ``temperature`` exactly -- same-team,
    cross-team and the zero-sum competition term alike. Same-player blocks are
    untouched, and not by approximation: :func:`covariance.build_game_covariance`
    pins player ``i``'s own block to the incumbent ``R_i`` by subtracting
    whatever the shared factors contributed to it, so scaling the shared part
    cancels exactly. ``temperature = 1`` returns an equal object; ``0`` returns
    cross-player conditional independence with same-player dependence intact.

    ``role_scale`` is deliberately *not* scaled. It is a ratio with a weighted
    mean of 1, so scaling it would change the level twice and break the
    normalisation the pooled block depends on.
    """
    temperature = float(temperature)
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("dependence temperature must lie in [0, 1]")
    if temperature == 1.0:
        return loadings
    root = math.sqrt(temperature)
    return replace(
        loadings,
        game=root * loadings.game,
        team_contrast=root * loadings.team_contrast,
        competition=None if loadings.competition is None else root * loadings.competition,
        symmetric=None if loadings.symmetric is None else root * loadings.symmetric,
        role_deviation=(
            None if loadings.role_deviation is None else root * loadings.role_deviation
        ),
    )


# ======================================================================
# 6. forward-cross-fitted uncertainty scales
# ======================================================================


def huber_scale(values: np.ndarray, c: float = 1.345, iterations: int = 100) -> float:
    """Huber M-estimate of scale for zero-centred standardized errors.

    Solves ``mean(psi_c(x / s)^2) = beta(c)`` where ``beta`` is the constant
    that makes the estimator consistent at the normal, so a clean sample
    returns the sample standard deviation rather than something smaller.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    # E[psi_c(Z)^2] for Z standard normal.
    beta = float(
        2.0 * (stats.norm.cdf(c) - 0.5 - c * stats.norm.pdf(c))
        + 2.0 * c**2 * stats.norm.sf(c)
    )
    scale = float(np.median(np.abs(x)) / 0.6745) or 1.0
    for _ in range(iterations):
        clipped = np.clip(x / scale, -c, c)
        updated = scale * math.sqrt(float(np.mean(clipped**2)) / beta)
        if not np.isfinite(updated) or updated <= 0:
            return scale
        if abs(updated - scale) < 1e-14:
            return updated
        scale = updated
    return scale


def median_of_fold_scales(frame: pd.DataFrame, fold_column: str = "target_season") -> float:
    """U2: median across folds of each fold's robust scale.

    A mean over folds lets one fold's departure set the scale for all of them,
    which is the failure the rejected global inflation factor made. The median
    cannot be moved by a single fold when there are three or more.
    """
    scales = [
        huber_scale(group["standardized"].to_numpy(float))
        for _, group in frame.groupby(fold_column)
        if len(group) >= 3
    ]
    scales = [value for value in scales if np.isfinite(value) and value > 0]
    if not scales:
        return float("nan")
    return float(np.median(scales))


def coverage_table(standardized: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    """Empirical two-sided coverage of a standardized error sample."""
    z = np.asarray(standardized, dtype=float)
    z = z[np.isfinite(z)] / float(scale)
    out: dict[str, float] = {}
    for level in COVERAGE_LEVELS:
        critical = float(stats.norm.ppf(0.5 + level / 2.0))
        out[f"coverage_{int(round(level * 100))}"] = float(np.mean(np.abs(z) <= critical))
    out["mean_squared_z"] = float(np.mean(z**2)) if z.size else float("nan")
    out["median_squared_z"] = float(np.median(z**2)) if z.size else float("nan")
    out["observations"] = float(z.size)
    return out


def coverage_loss(coverage: Mapping[str, float]) -> float:
    """Mean absolute coverage error across the declared levels.

    The objective is coverage, not a mean squared z of exactly 1: an interval
    that covers 90% of outcomes at its 90% level is calibrated whatever its
    second moment does, and a heavy-tailed error distribution can have both at
    once only by accident.
    """
    errors = [
        abs(coverage[f"coverage_{int(round(level * 100))}"] - level)
        for level in COVERAGE_LEVELS
        if f"coverage_{int(round(level * 100))}" in coverage
    ]
    return float(np.mean(errors)) if errors else float("nan")


# ======================================================================
# the remediation spec and its fit
# ======================================================================

ROLE_SCALE_RATIO = "ratio"
ROLE_SCALE_LOG_SHRUNK = "log_shrunk"

CROSS_PRIOR_GAUSSIAN = "gaussian"
CROSS_PRIOR_STUDENT_T = "student_t"

BRIDGE_OFF = "off"
BRIDGE_COMBINED = "inverse_variance"


@dataclass(frozen=True)
class RemediationSpec:
    """One candidate upstream model. Scalars and structural labels only.

    Nothing here is indexed by a player or a player pair, so the parameter
    accounting the gates check is unchanged by any setting of it.
    """

    name: str
    k_game: int = 6
    r_contrast: int = 6
    eb_family: str = EB_FAMILY_BLOCK_DIAGONAL
    role_column: str | None = "role_bucket"
    #: 2. role-scale geometry
    role_scale_mode: str = ROLE_SCALE_RATIO
    role_tau_log: float | None = None
    #: 3. cross-team prior
    cross_team_prior: str = CROSS_PRIOR_GAUSSIAN
    #: ``None`` under the Gaussian prior, where there are no degrees of freedom
    #: to report. Recording a number there would claim a parameter the model
    #: does not have.
    cross_team_nu: float | None = 5.0
    #: 4. latent-to-count transmission
    bridge_mode: str = BRIDGE_OFF
    bridge_weight_cap: float = 1.0
    #: 5. cross-player dependence temperature
    dependence_temperature: float = 1.0

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "k_game": int(self.k_game),
            "r_contrast": int(self.r_contrast),
            "eb_family": self.eb_family,
            "role_column": self.role_column,
            "role_scale_mode": self.role_scale_mode,
            "role_tau_log": None if self.role_tau_log is None else float(self.role_tau_log),
            "cross_team_prior": self.cross_team_prior,
            "cross_team_nu": (
                None if self.cross_team_nu is None else float(self.cross_team_nu)
            ),
            "bridge_mode": self.bridge_mode,
            "bridge_weight_cap": float(self.bridge_weight_cap),
            "dependence_temperature": float(self.dependence_temperature),
            "shrinkage": SHRINKAGE_EMPIRICAL_BAYES,
        }

    def parameter_count(self, n_stats: int) -> dict[str, int]:
        triangle = n_stats * (n_stats + 1) // 2
        scalars = 2  # the two block priors
        if self.cross_team_prior == CROSS_PRIOR_STUDENT_T:
            scalars += 1  # the cross-team degrees of freedom
        if self.role_scale_mode == ROLE_SCALE_LOG_SHRUNK:
            scalars += 1  # tau_role
        if self.bridge_mode != BRIDGE_OFF:
            scalars += 1  # the transmission weight cap
        if self.dependence_temperature != 1.0:
            scalars += 1  # lambda
        return {
            "stat_indexed_game": int(n_stats * self.k_game),
            "stat_indexed_contrast": int(n_stats * self.r_contrast),
            "stat_indexed_competition": int(triangle),
            "scalar_hyperparameters": int(scalars),
            "role_scale_values": 0 if self.role_column is None else 3,
            "role_scale_free_parameters": 0 if self.role_column is None else 2,
            "active_role_deviation_parameters": 0,
            "active_symmetric_subspace_parameters": 0,
            "player_indexed": 0,
            "pairwise": 0,
        }


#: The accepted bucket repair, written in this module's vocabulary. Asserted by
#: test to reproduce ``repair.fit_repaired_factors`` on the accepted spec
#: bit-for-bit, which is what makes it the control rather than a look-alike.
CONTROL_SPEC = RemediationSpec(name="accepted_bucket_repair_control")


def shrink_blocks_remediated(
    moments: PairMoments,
    spec: RemediationSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Shrink the two pooled blocks, each under its own prior."""
    same, same_diagnostics = empirical_bayes_shrink(
        moments.same_team,
        moments.same_team_se,
        family=spec.eb_family,
        is_same_team=True,
    )
    if spec.cross_team_prior == CROSS_PRIOR_GAUSSIAN:
        cross, cross_diagnostics = empirical_bayes_shrink(
            moments.cross_team,
            moments.cross_team_se,
            family=spec.eb_family,
            is_same_team=False,
        )
    elif spec.cross_team_prior == CROSS_PRIOR_STUDENT_T:
        if spec.cross_team_nu is None:
            raise ValueError("the Student-t cross-team prior needs its nu")
        cross, cross_diagnostics = robust_shrink_block(
            moments.cross_team,
            moments.cross_team_se,
            nu=spec.cross_team_nu,
            family=spec.eb_family,
            is_same_team=False,
        )
    else:
        raise ValueError(f"unknown cross-team prior {spec.cross_team_prior!r}")
    return same, cross, {**same_diagnostics, **cross_diagnostics}


def assemble_loadings(
    stats: Sequence[str],
    same: np.ndarray,
    cross: np.ndarray,
    moments: PairMoments,
    spec: RemediationSpec,
) -> tuple[SharedFactorLoadings, dict[str, object]]:
    """The accepted repair's PSD factor construction, on supplied targets.

    Separated out so the transmission work in part 4 can hand it combined
    targets without reimplementing the identification or the PSD argument.
    """
    stats = tuple(stats)
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
    diagnostics = {
        "competition_evidence": competition_evidence,
        "game_gram_eigenvalues": np.linalg.eigvalsh(
            0.5 * (game_gram + game_gram.T)
        )[::-1],
        "contrast_gram_eigenvalues": np.linalg.eigvalsh(
            0.5 * (contrast_gram + contrast_gram.T)
        )[::-1],
        "competition_gram_eigenvalues": np.linalg.eigvalsh(
            0.5 * (competition + competition.T)
        )[::-1],
    }
    return base, diagnostics


def fit_remediated_factors(
    frame: pd.DataFrame,
    stats: Sequence[str],
    spec: RemediationSpec = CONTROL_SPEC,
    bootstrap: int = 200,
    seed: int = 73,
    value_prefix: str = "zs_",
    target_override: tuple[np.ndarray, np.ndarray] | None = None,
    role_scale_override: Mapping[str, float] | None = None,
    moments: PairMoments | None = None,
) -> FactorFit:
    """Fit the shared loadings under a remediation spec.

    ``target_override`` replaces the shrunk ``(same, cross)`` blocks, which is
    how the transmission-combined targets enter without this function needing
    to know how they were produced. ``role_scale_override`` does the same for
    the role layer. ``moments`` lets a grid search pass the pooled moments in
    once: they are a function of the data alone, not of the spec, and the
    game-clustered bootstrap inside them costs more than everything else here
    put together.
    """
    from .factors import fit_role_scales, pair_moments

    stats = tuple(stats)
    if moments is None:
        moments = pair_moments(
            frame, stats, value_prefix=value_prefix, bootstrap=bootstrap, seed=seed
        )
    elif tuple(moments.stats) != stats:
        raise ValueError("supplied moments were pooled over different stats")

    same, cross, shrink_diagnostics = shrink_blocks_remediated(moments, spec)
    if target_override is not None:
        same, cross = target_override
        same = np.asarray(same, dtype=float)
        cross = np.asarray(cross, dtype=float)
        shrink_diagnostics = {**shrink_diagnostics, "targets_overridden": 1.0}

    base, diagnostics = assemble_loadings(stats, same, cross, moments, spec)

    if role_scale_override is not None:
        role_scale = dict(role_scale_override)
    elif spec.role_column is None:
        role_scale = {}
    elif spec.role_scale_mode == ROLE_SCALE_RATIO:
        role_scale = fit_role_scales(
            frame,
            stats,
            base=base,
            role_column=spec.role_column,
            value_prefix=value_prefix,
        )
    else:
        raise ValueError(
            "log-shrunk role scales need their bootstrap standard errors; "
            "fit them with role_scale_components and pass role_scale_override"
        )

    loadings = SharedFactorLoadings(
        stats=stats,
        game=base.game,
        team_contrast=base.team_contrast,
        competition=base.competition,
        role_scale=role_scale,
    )
    loadings = temper_loadings(loadings, spec.dependence_temperature)

    return FactorFit(
        loadings=loadings,
        moments=moments,
        same_team_shrunk=same,
        cross_team_shrunk=cross,
        game_gram_eigenvalues=diagnostics["game_gram_eigenvalues"],
        contrast_gram_eigenvalues=diagnostics["contrast_gram_eigenvalues"],
        competition_gram_eigenvalues=diagnostics["competition_gram_eigenvalues"],
        k_game=int(spec.k_game),
        r_competition=0 if loadings.competition is None else loadings.r_competition,
        shrink_z=float("nan"),
        competition_evidence={
            **diagnostics["competition_evidence"],
            **shrink_diagnostics,
        },
    )
