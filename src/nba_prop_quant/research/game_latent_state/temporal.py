"""Temporal structure of a cross-player dependence bucket.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

Why this module exists
----------------------
The accepted bucket repair fits ``teammate_ast_ast`` from the pooled pre-2024
history and lands at ``+0.004741`` against a held-out ``+0.008971``
(``z = -1.89``). Inspecting the training seasons one at a time shows a 2023
point estimate with the *opposite* sign. The tempting reaction -- give the
bucket a free time-varying parameter -- is exactly the move that turns a
sampling fluctuation into a modelling commitment, so this module answers the
prior question first: **is the season-to-season variation larger than the
season-to-season sampling noise?**

The model
---------
Each season's game-clustered estimate is treated as a noisy read of that
season's own correlation,

    observed_r_s ~ Normal(theta_s, se_s^2),    theta_s ~ Normal(mu, tau_time^2)

with ``mu`` and ``tau_time`` estimated from training seasons only. ``se_s``
comes from the game-clustered bootstrap, so the within-season uncertainty is
measured rather than assumed. Three quantities then separate the two
explanations:

* **Cochran's ``Q``** compares the between-season spread with the pooled
  within-season noise. Under exchangeability ``Q ~ chi2(k - 1)``.
* **``tau_time``** is the between-season standard deviation on the
  correlation scale, so it can be read against ``|mu|`` directly: a
  ``tau_time`` far below ``|mu|`` means the seasons differ by less than the
  level itself, and the sign is stable even if ``Q`` is significant.
* **``P(mu > 0)``** is the posterior sign probability of the *level*, which is
  what a pregame parameter actually needs to be confident about.

The treatments
--------------
``T0`` is the incumbent: one pooled empirical-Bayes estimate over all
training seasons at once, which is what the accepted repair does. ``T1``
replaces it with the random-effects posterior mean, which differs from
``T0`` by down-weighting seasons whose own estimate is noisy rather than
weighting every player-pair equally. ``T2`` adds exponential recency decay on
top of the random-effects weights. ``T3`` is a one-state random walk on
``theta_s``, run through a Kalman filter.

``T3`` is admitted to the inner comparison only when the season deviations
show real serial dependence: a random walk buys nothing over an exchangeable
prior unless last season's deviation predicts this season's, and with a
handful of seasons a dynamic model will otherwise win on flexibility alone
and lose out of sample. :data:`AR1_SUPPORT_THRESHOLD` is the pre-registered
bar, declared here rather than chosen after looking at the fit.

No-lookahead
------------
:func:`treatment_prediction` is the only way a treatment produces a number,
and it takes the target season explicitly and drops every estimate from that
season onward. A live season with too few observations therefore resolves to
the pooled posterior over completed seasons: no realised outcome from the
season being predicted can reach a pregame parameter, by construction rather
than by convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .factors import pair_moments

#: Stability classifications. ``STABLE_POSITIVE`` means the level is
#: confidently positive and the seasons differ by less than the level;
#: ``TIME_VARYING`` means the level is confidently signed but the seasons
#: differ by a comparable amount; ``UNCERTAIN`` means the level's own sign is
#: not established, which is the correct reading of a lone opposite-signed
#: season whose interval contains zero.
STABLE_POSITIVE = "STABLE_POSITIVE"
UNCERTAIN = "UNCERTAIN"
TIME_VARYING = "TIME_VARYING"

#: Posterior sign confidence a bucket must clear to be called stable or
#: time-varying rather than uncertain. The usual two-sided 5% level.
SIGN_CONFIDENCE = 0.95

#: Cochran ``Q`` significance level for declaring between-season heterogeneity.
HETEROGENEITY_LEVEL = 0.05

#: ``tau_time / |mu|`` above which the seasons differ by an amount comparable
#: with the level itself. Half the level is a deliberately coarse bar: this
#: classification is reported, not acted on, and a finer threshold would
#: invite tuning.
TIME_VARYING_TAU_RATIO = 0.5

#: Treatment labels. The selected one is recorded in the frozen candidate.
TREATMENT_POOLED = "T0_pooled_empirical_bayes"
TREATMENT_RANDOM_EFFECTS = "T1_random_effects"
TREATMENT_RECENCY = "T2_recency_weighted_random_effects"
TREATMENT_DYNAMIC = "T3_random_walk_state"

#: Exponential recency decay for ``T2``, as a half-life in seasons. Declared,
#: not searched: a half-life of two seasons is the shortest memory that still
#: pools more than two seasons' worth of games.
RECENCY_HALF_LIFE_SEASONS = 2.0

#: Lag-1 autocorrelation of the standardized season deviations that ``T3``
#: must clear before it is even compared. Below it there is no serial
#: structure for a random walk to exploit and parsimony decides.
AR1_SUPPORT_THRESHOLD = 0.30

#: Seasons of training history ``T3`` needs before its one free variance
#: parameter is estimable at all.
MIN_SEASONS_FOR_DYNAMIC = 6

#: Upper search bound for the between-season variance, in squared correlation
#: units. Far above any plausible value for a cross-player correlation.
TAU2_SEARCH_CEILING = 1.0

#: Relative predictive-MSE band inside which two treatments are called tied.
#: With a handful of folds the MSE ranking is itself noisy, so a difference
#: this small is not evidence; parsimony decides instead.
SELECTION_TIE_BAND = 0.05


@dataclass(frozen=True)
class SeasonEstimate:
    """One season's game-clustered read of a single dependence bucket."""

    season: int
    estimate: float
    standard_error: float
    bootstrap_low: float
    bootstrap_high: float
    games: int
    pairs: float
    posterior_sign_probability: float

    def payload(self) -> dict[str, object]:
        return {
            "season": int(self.season),
            "estimate": float(self.estimate),
            "standard_error": float(self.standard_error),
            "bootstrap_interval": [
                float(self.bootstrap_low),
                float(self.bootstrap_high),
            ],
            "effective_games": int(self.games),
            "effective_pairs": float(self.pairs),
            "posterior_sign_probability": float(self.posterior_sign_probability),
        }


def season_bucket_estimates(
    frame: pd.DataFrame,
    stats: Sequence[str],
    bucket: tuple[str, str],
    block: str = "same_team",
    value_prefix: str = "zs_",
    bootstrap: int = 400,
    seed: int = 73,
    interval: float = 0.95,
) -> list[SeasonEstimate]:
    """Estimate one bucket separately in every season present in ``frame``.

    Each season is bootstrapped over its own games, so the standard error is
    game-clustered within season and the seasons are independent reads. The
    percentile interval is reported alongside the standard error because a
    season with few games has a visibly asymmetric sampling distribution and
    the normal interval would understate how much of it sits across zero.
    """
    stats = tuple(stats)
    index = {stat: position for position, stat in enumerate(stats)}
    first, second = bucket
    i, j = index[first], index[second]
    tail = 0.5 * (1.0 - float(interval))

    out: list[SeasonEstimate] = []
    for season, group in frame.groupby("season", sort=True):
        moments = pair_moments(
            group,
            stats,
            value_prefix=value_prefix,
            bootstrap=bootstrap,
            seed=seed,
            keep_draws=True,
        )
        if block == "same_team":
            point = float(moments.same_team[i, j])
            error = float(moments.same_team_se[i, j])
            draws = moments.same_team_draws
            pairs = float(moments.same_team_pairs)
        elif block == "cross_team":
            point = float(moments.cross_team[i, j])
            error = float(moments.cross_team_se[i, j])
            draws = moments.cross_team_draws
            pairs = float(moments.cross_team_pairs)
        else:
            raise ValueError(f"unknown block {block!r}")

        if draws is None:
            low, high = float("nan"), float("nan")
        else:
            entry = 0.5 * (draws[:, i, j] + draws[:, j, i])
            low = float(np.quantile(entry, tail))
            high = float(np.quantile(entry, 1.0 - tail))

        out.append(
            SeasonEstimate(
                season=int(season),
                estimate=point,
                standard_error=error,
                bootstrap_low=low,
                bootstrap_high=high,
                games=int(moments.games),
                pairs=pairs,
                posterior_sign_probability=float(
                    norm.cdf(point / error) if error > 0 else float("nan")
                ),
            )
        )
    return out


def paule_mandel_tau2(
    estimates: np.ndarray,
    variances: np.ndarray,
    ceiling: float = TAU2_SEARCH_CEILING,
) -> float:
    """Between-season variance by the Paule-Mandel estimating equation.

    ``sum_s (r_s - mu(tau2))^2 / (se_s^2 + tau2) == k - 1`` is strictly
    decreasing in ``tau2``, so a bisection is exact up to its tolerance. The
    Paule-Mandel estimator is used in preference to DerSimonian-Laird because
    DL's moment correction is badly downward-biased when ``k`` is small, which
    is precisely the regime here; DL is still reported for comparison.
    """
    estimates = np.asarray(estimates, dtype=float)
    variances = np.asarray(variances, dtype=float)
    k = estimates.size
    if k < 2:
        return 0.0

    def discrepancy(tau2: float) -> float:
        weights = 1.0 / (variances + tau2)
        mean = float(np.sum(weights * estimates) / np.sum(weights))
        return float(np.sum(weights * (estimates - mean) ** 2) - (k - 1))

    if discrepancy(0.0) <= 0.0:
        return 0.0
    if discrepancy(ceiling) > 0.0:
        return float(ceiling)

    low, high = 0.0, float(ceiling)
    for _ in range(200):
        middle = 0.5 * (low + high)
        if discrepancy(middle) > 0.0:
            low = middle
        else:
            high = middle
        if high - low < 1e-18:
            break
    return 0.5 * (low + high)


def dersimonian_laird_tau2(
    estimates: np.ndarray,
    variances: np.ndarray,
) -> tuple[float, float, int, float]:
    """``(tau2, Q, df, p_value)`` from the DerSimonian-Laird moment estimator."""
    estimates = np.asarray(estimates, dtype=float)
    variances = np.asarray(variances, dtype=float)
    k = estimates.size
    if k < 2:
        return 0.0, 0.0, 0, float("nan")

    precision = 1.0 / variances
    fixed_mean = float(np.sum(precision * estimates) / np.sum(precision))
    q = float(np.sum(precision * (estimates - fixed_mean) ** 2))
    df = k - 1
    scaling = float(np.sum(precision) - np.sum(precision**2) / np.sum(precision))
    tau2 = max((q - df) / scaling, 0.0) if scaling > 0 else 0.0
    return float(tau2), q, df, float(chi2.sf(q, df))


@dataclass(frozen=True)
class RandomEffectsFit:
    """Posterior summary of a bucket's level and its temporal heterogeneity."""

    seasons: tuple[int, ...]
    estimates: tuple[float, ...]
    standard_errors: tuple[float, ...]
    weights: tuple[float, ...]
    posterior_mean: float
    posterior_sd: float
    tau2: float
    tau2_dersimonian_laird: float
    q_statistic: float
    degrees_of_freedom: int
    q_p_value: float
    i_squared: float
    lag1_autocorrelation: float
    posterior_sign_probability: float
    predictive_sd: float
    predictive_sign_probability: float
    classification: str
    decay_half_life: float | None = None

    @property
    def tau_time(self) -> float:
        return float(np.sqrt(max(self.tau2, 0.0)))

    def payload(self) -> dict[str, object]:
        return {
            "seasons": [int(season) for season in self.seasons],
            "estimates": [float(value) for value in self.estimates],
            "standard_errors": [float(value) for value in self.standard_errors],
            "weights": [float(value) for value in self.weights],
            "posterior_mean": float(self.posterior_mean),
            "posterior_sd": float(self.posterior_sd),
            "tau2": float(self.tau2),
            "tau_time": float(self.tau_time),
            "tau2_dersimonian_laird": float(self.tau2_dersimonian_laird),
            "q_statistic": float(self.q_statistic),
            "degrees_of_freedom": int(self.degrees_of_freedom),
            "q_p_value": float(self.q_p_value),
            "i_squared": float(self.i_squared),
            "lag1_autocorrelation": float(self.lag1_autocorrelation),
            "posterior_probability_positive": float(self.posterior_sign_probability),
            "predictive_sd": float(self.predictive_sd),
            "predictive_probability_positive": float(
                self.predictive_sign_probability
            ),
            "stability_classification": self.classification,
            "recency_half_life_seasons": (
                None if self.decay_half_life is None else float(self.decay_half_life)
            ),
        }


def classify_stability(
    posterior_mean: float,
    posterior_sign_probability: float,
    tau: float,
    q_p_value: float,
) -> str:
    """Pre-registered stability label for a dependence bucket.

    The order matters: a bucket whose *level* is not confidently signed is
    ``UNCERTAIN`` whatever the heterogeneity test says, because a significant
    ``Q`` on top of an interval that contains zero is evidence of noise, not
    of a regime. Only once the sign is established does the size of
    ``tau_time`` relative to the level decide between stable and time-varying.
    """
    confidence = max(
        posterior_sign_probability, 1.0 - posterior_sign_probability
    )
    if not np.isfinite(confidence) or confidence < SIGN_CONFIDENCE:
        return UNCERTAIN
    level = abs(float(posterior_mean))
    heterogeneous = np.isfinite(q_p_value) and q_p_value < HETEROGENEITY_LEVEL
    comparable = level > 0 and (tau / level) >= TIME_VARYING_TAU_RATIO
    if heterogeneous and comparable:
        return TIME_VARYING
    return STABLE_POSITIVE


def fit_random_effects(
    estimates: Sequence[SeasonEstimate],
    decay_half_life: float | None = None,
) -> RandomEffectsFit:
    """Hierarchical fit of ``theta_s ~ Normal(mu, tau_time^2)``.

    With ``decay_half_life`` set, the inverse-variance weights are multiplied
    by ``0.5 ** (age / half_life)``, which is treatment ``T2``. The posterior
    standard deviation then uses the general weighted-mean formula rather than
    ``1 / sqrt(sum w)``, since that shortcut only holds for exactly
    inverse-variance weights.
    """
    if not estimates:
        raise ValueError("a temporal fit needs at least one season estimate")

    ordered = sorted(estimates, key=lambda item: item.season)
    seasons = np.array([item.season for item in ordered], dtype=int)
    values = np.array([item.estimate for item in ordered], dtype=float)
    errors = np.array([item.standard_error for item in ordered], dtype=float)
    variances = np.maximum(errors**2, 1e-18)

    tau2_dl, q, df, q_p = dersimonian_laird_tau2(values, variances)
    tau2 = paule_mandel_tau2(values, variances)

    weights = 1.0 / (variances + tau2)
    if decay_half_life is not None:
        age = seasons.max() - seasons
        weights = weights * 0.5 ** (age / float(decay_half_life))

    total = float(np.sum(weights))
    mean = float(np.sum(weights * values) / total)
    posterior_variance = float(np.sum(weights**2 * (variances + tau2)) / total**2)
    posterior_sd = float(np.sqrt(max(posterior_variance, 0.0)))

    deviations = (values - mean) / np.sqrt(variances + tau2)
    if deviations.size >= 3:
        numerator = float(np.sum(deviations[:-1] * deviations[1:]))
        denominator = float(np.sum(deviations**2))
        lag1 = numerator / denominator if denominator > 0 else 0.0
    else:
        lag1 = float("nan")

    i_squared = float(max((q - df) / q, 0.0)) if q > 0 and df > 0 else 0.0
    sign_probability = (
        float(norm.cdf(mean / posterior_sd)) if posterior_sd > 0 else float("nan")
    )
    predictive_sd = float(np.sqrt(posterior_variance + max(tau2, 0.0)))
    predictive_sign = (
        float(norm.cdf(mean / predictive_sd)) if predictive_sd > 0 else float("nan")
    )

    return RandomEffectsFit(
        seasons=tuple(int(season) for season in seasons),
        estimates=tuple(float(value) for value in values),
        standard_errors=tuple(float(value) for value in errors),
        weights=tuple(float(value) for value in weights / total),
        posterior_mean=mean,
        posterior_sd=posterior_sd,
        tau2=float(tau2),
        tau2_dersimonian_laird=float(tau2_dl),
        q_statistic=float(q),
        degrees_of_freedom=int(df),
        q_p_value=float(q_p),
        i_squared=i_squared,
        lag1_autocorrelation=float(lag1),
        posterior_sign_probability=sign_probability,
        predictive_sd=predictive_sd,
        predictive_sign_probability=predictive_sign,
        classification=classify_stability(
            mean, sign_probability, float(np.sqrt(max(tau2, 0.0))), float(q_p)
        ),
        decay_half_life=decay_half_life,
    )


@dataclass(frozen=True)
class RandomWalkFit:
    """One-state random walk ``theta_s = theta_{s-1} + eta`` by Kalman filter.

    ``signal_variance`` is the single free parameter, fitted by maximising the
    filter's own one-step predictive likelihood on the training seasons. A
    value of zero collapses the walk onto the exchangeable pooled mean, so the
    estimator can decline the dynamics rather than being forced to use them.
    """

    seasons: tuple[int, ...]
    signal_variance: float
    filtered_mean: float
    filtered_sd: float
    predictive_log_likelihood: float
    supported: bool
    support_reason: str

    def payload(self) -> dict[str, object]:
        return {
            "seasons": [int(season) for season in self.seasons],
            "signal_variance": float(self.signal_variance),
            "filtered_mean": float(self.filtered_mean),
            "filtered_sd": float(self.filtered_sd),
            "predictive_log_likelihood": float(self.predictive_log_likelihood),
            "supported": bool(self.supported),
            "support_reason": self.support_reason,
        }


def _random_walk_likelihood(
    values: np.ndarray,
    variances: np.ndarray,
    signal_variance: float,
) -> tuple[float, float, float]:
    """``(log_likelihood, filtered_mean, filtered_variance)`` of the walk."""
    # Diffuse start: the first observation defines the state exactly up to its
    # own measurement error, which is the standard treatment and keeps the
    # likelihood comparable across signal variances.
    state = float(values[0])
    state_variance = float(variances[0])
    log_likelihood = 0.0
    for position in range(1, values.size):
        prior_variance = state_variance + signal_variance
        innovation = float(values[position]) - state
        innovation_variance = prior_variance + float(variances[position])
        log_likelihood += -0.5 * (
            np.log(2.0 * np.pi * innovation_variance)
            + innovation**2 / innovation_variance
        )
        gain = prior_variance / innovation_variance
        state = state + gain * innovation
        state_variance = (1.0 - gain) * prior_variance
    return float(log_likelihood), state, state_variance


def fit_random_walk(
    estimates: Sequence[SeasonEstimate],
    lag1_autocorrelation: float,
    grid: Sequence[float] | None = None,
) -> RandomWalkFit:
    """Fit ``T3`` and record whether the data support admitting it.

    ``supported`` is decided *before* the fit is compared with anything: too
    few seasons, or season deviations with no serial dependence, and the walk
    is declined on parsimony. That ordering is the point -- a dynamic model
    must earn its place with evidence of dynamics, not with goodness of fit.
    """
    ordered = sorted(estimates, key=lambda item: item.season)
    seasons = tuple(int(item.season) for item in ordered)
    values = np.array([item.estimate for item in ordered], dtype=float)
    variances = np.maximum(
        np.array([item.standard_error for item in ordered], dtype=float) ** 2, 1e-18
    )

    if grid is None:
        scale = float(np.mean(variances))
        grid = [0.0, *(scale * 2.0 ** exponent for exponent in range(-4, 7))]

    best = (-np.inf, 0.0, values[-1], variances[-1])
    for candidate in grid:
        likelihood, mean, variance = _random_walk_likelihood(
            values, variances, float(candidate)
        )
        if likelihood > best[0]:
            best = (likelihood, float(candidate), mean, variance)

    if len(seasons) < MIN_SEASONS_FOR_DYNAMIC:
        supported = False
        reason = (
            f"{len(seasons)} training seasons is below the "
            f"{MIN_SEASONS_FOR_DYNAMIC} a one-state walk needs"
        )
    elif not np.isfinite(lag1_autocorrelation):
        supported = False
        reason = "lag-1 autocorrelation of the season deviations is undefined"
    elif abs(lag1_autocorrelation) < AR1_SUPPORT_THRESHOLD:
        supported = False
        reason = (
            f"lag-1 autocorrelation {lag1_autocorrelation:+.4f} is inside the "
            f"pre-registered +/-{AR1_SUPPORT_THRESHOLD} band, so the season "
            "deviations carry no serial structure for a walk to exploit"
        )
    else:
        supported = True
        reason = (
            f"lag-1 autocorrelation {lag1_autocorrelation:+.4f} clears the "
            f"pre-registered {AR1_SUPPORT_THRESHOLD} bar"
        )

    return RandomWalkFit(
        seasons=seasons,
        signal_variance=best[1],
        filtered_mean=float(best[2]),
        filtered_sd=float(np.sqrt(max(best[3], 0.0))),
        predictive_log_likelihood=float(best[0]),
        supported=supported,
        support_reason=reason,
    )


@dataclass(frozen=True)
class TemporalPrediction:
    """A treatment's pregame read of a bucket for one target season."""

    treatment: str
    target_season: int
    training_seasons: tuple[int, ...]
    prediction: float
    prediction_sd: float
    classification: str
    fell_back_to_pooled: bool = False
    fallback_reason: str = ""

    def payload(self) -> dict[str, object]:
        return {
            "treatment": self.treatment,
            "target_season": int(self.target_season),
            "training_seasons": [int(season) for season in self.training_seasons],
            "prediction": float(self.prediction),
            "prediction_sd": float(self.prediction_sd),
            "stability_classification": self.classification,
            "fell_back_to_pooled": bool(self.fell_back_to_pooled),
            "fallback_reason": self.fallback_reason,
        }


#: Minimum completed seasons a treatment needs before it reports anything but
#: the pooled posterior. A live season is predicted from completed seasons
#: only, so this is also the live-season fallback trigger.
MIN_SEASONS_FOR_PREDICTION = 2


def treatment_prediction(
    treatment: str,
    estimates: Sequence[SeasonEstimate],
    target_season: int,
    pooled_estimate: float,
    pooled_sd: float = float("nan"),
) -> TemporalPrediction:
    """One treatment's prediction for ``target_season``, with no lookahead.

    Every estimate from ``target_season`` onward is discarded before anything
    is fitted, so a realised outcome from the season being predicted cannot
    enter the parameter. ``pooled_estimate`` is the ``T0`` control value --
    the accepted repair's own pooled fit -- and is also what every treatment
    falls back to when there are too few completed seasons to fit.
    """
    history = [item for item in estimates if item.season < int(target_season)]
    seasons = tuple(int(item.season) for item in history)

    if len(history) < MIN_SEASONS_FOR_PREDICTION:
        return TemporalPrediction(
            treatment=treatment,
            target_season=int(target_season),
            training_seasons=seasons,
            prediction=float(pooled_estimate),
            prediction_sd=float(pooled_sd),
            classification=UNCERTAIN,
            fell_back_to_pooled=True,
            fallback_reason=(
                f"{len(history)} completed seasons before {int(target_season)} is "
                f"below the {MIN_SEASONS_FOR_PREDICTION} a temporal fit needs"
            ),
        )

    if treatment == TREATMENT_POOLED:
        reference = fit_random_effects(history)
        return TemporalPrediction(
            treatment=treatment,
            target_season=int(target_season),
            training_seasons=seasons,
            prediction=float(pooled_estimate),
            prediction_sd=float(
                pooled_sd if np.isfinite(pooled_sd) else reference.predictive_sd
            ),
            classification=reference.classification,
        )

    if treatment == TREATMENT_RANDOM_EFFECTS:
        fit = fit_random_effects(history)
    elif treatment == TREATMENT_RECENCY:
        fit = fit_random_effects(history, decay_half_life=RECENCY_HALF_LIFE_SEASONS)
    elif treatment == TREATMENT_DYNAMIC:
        reference = fit_random_effects(history)
        walk = fit_random_walk(history, reference.lag1_autocorrelation)
        if not walk.supported:
            return TemporalPrediction(
                treatment=treatment,
                target_season=int(target_season),
                training_seasons=seasons,
                prediction=float(reference.posterior_mean),
                prediction_sd=float(reference.predictive_sd),
                classification=reference.classification,
                fell_back_to_pooled=True,
                fallback_reason=walk.support_reason,
            )
        return TemporalPrediction(
            treatment=treatment,
            target_season=int(target_season),
            training_seasons=seasons,
            prediction=float(walk.filtered_mean),
            prediction_sd=float(
                np.sqrt(walk.filtered_sd**2 + walk.signal_variance)
            ),
            classification=reference.classification,
        )
    else:
        raise ValueError(f"unknown temporal treatment {treatment!r}")

    return TemporalPrediction(
        treatment=treatment,
        target_season=int(target_season),
        training_seasons=seasons,
        prediction=float(fit.posterior_mean),
        prediction_sd=float(fit.predictive_sd),
        classification=fit.classification,
    )


@dataclass(frozen=True)
class TemporalScreening:
    """Walk-forward comparison of the temporal treatments on training folds."""

    bucket: str
    folds: tuple[int, ...]
    scores: Mapping[str, Mapping[str, float]]
    selected: str
    selection_reason: str
    detail: Sequence[Mapping[str, object]] = field(default_factory=tuple)

    def payload(self) -> dict[str, object]:
        return {
            "bucket": self.bucket,
            "folds": [int(fold) for fold in self.folds],
            "scores": {
                treatment: {key: float(value) for key, value in metrics.items()}
                for treatment, metrics in self.scores.items()
            },
            "selected_treatment": self.selected,
            "selection_reason": self.selection_reason,
            "fold_detail": [dict(entry) for entry in self.detail],
        }


def screen_temporal_treatments(
    estimates: Sequence[SeasonEstimate],
    pooled_by_fold: Mapping[int, float],
    bucket: str,
    treatments: Sequence[str] | None = None,
) -> TemporalScreening:
    """Nested chronological screening of the temporal treatments.

    For each fold season ``S`` every treatment is fitted on seasons ``< S``
    and scored against season ``S``'s own realised estimate. Two scores are
    kept: the predictive mean squared error, and a calibration statistic
    ``mean(z^2)`` where ``z`` is the error over the treatment's own claimed
    predictive standard deviation combined with the season's sampling error.
    A treatment that is accurate but over-confident is visible in the second
    score even when the first one looks good.

    Selection takes the lowest predictive MSE, and breaks a tie inside
    ``SELECTION_TIE_BAND`` in favour of the earlier (more parsimonious)
    treatment in ``treatments`` order, which is why ``T0`` is listed first.
    """
    if treatments is None:
        treatments = (
            TREATMENT_POOLED,
            TREATMENT_RANDOM_EFFECTS,
            TREATMENT_RECENCY,
            TREATMENT_DYNAMIC,
        )

    ordered = sorted(estimates, key=lambda item: item.season)
    folds = tuple(
        int(item.season)
        for item in ordered
        if len([other for other in ordered if other.season < item.season])
        >= MIN_SEASONS_FOR_PREDICTION
    )
    if not folds:
        raise ValueError("not enough seasons for a chronological temporal screen")

    by_season = {int(item.season): item for item in ordered}
    errors: dict[str, list[float]] = {name: [] for name in treatments}
    calibration: dict[str, list[float]] = {name: [] for name in treatments}
    detail: list[dict[str, object]] = []

    for fold in folds:
        realised = by_season[fold]
        entry: dict[str, object] = {
            "fold_season": int(fold),
            "realised_estimate": float(realised.estimate),
            "realised_standard_error": float(realised.standard_error),
        }
        for name in treatments:
            prediction = treatment_prediction(
                name,
                ordered,
                target_season=fold,
                pooled_estimate=float(pooled_by_fold[fold]),
            )
            error = float(realised.estimate - prediction.prediction)
            errors[name].append(error**2)
            total_variance = realised.standard_error**2 + (
                prediction.prediction_sd**2
                if np.isfinite(prediction.prediction_sd)
                else 0.0
            )
            calibration[name].append(
                error**2 / total_variance if total_variance > 0 else float("nan")
            )
            entry[name] = prediction.payload()
        detail.append(entry)

    scores = {
        name: {
            "predictive_mse": float(np.mean(errors[name])),
            "predictive_rmse": float(np.sqrt(np.mean(errors[name]))),
            "mean_squared_z": float(np.nanmean(calibration[name])),
            "calibration_gap": float(abs(np.nanmean(calibration[name]) - 1.0)),
            "folds": float(len(errors[name])),
        }
        for name in treatments
    }

    best = min(scores, key=lambda name: scores[name]["predictive_mse"])
    floor = scores[best]["predictive_mse"]
    for name in treatments:
        if scores[name]["predictive_mse"] <= floor * (1.0 + SELECTION_TIE_BAND):
            best = name
            break
    reason = (
        f"lowest predictive MSE within the {SELECTION_TIE_BAND:.0%} parsimony "
        f"tie band; {best} scores {scores[best]['predictive_mse']:.3e} against "
        f"{scores[TREATMENT_POOLED]['predictive_mse']:.3e} for the pooled control"
    )

    return TemporalScreening(
        bucket=bucket,
        folds=folds,
        scores=scores,
        selected=best,
        selection_reason=reason,
        detail=tuple(detail),
    )

