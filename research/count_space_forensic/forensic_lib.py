"""Shared plumbing for the count-space blocker forensic study.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE. This module reads the frozen
remediation candidate and never writes it.

Everything here is a *measurement* tool. The study asks whether the
count-space shortfall on ``passer_ast_teammate_pts`` is an estimator defect
or a structural limit, so the tools are: the five dependence estimators, the
exact transmission map from a latent correlation to what each space would
read, and the feasibility probe that perturbs the frozen architecture inside
its own PSD guarantees.

Two marginal-training conventions
---------------------------------
The pipeline fits the production marginals twice, on *different* row sets,
and the forensic study has to read each where it applies.

``02_build_oof_residuals.py`` drops rows that are missing *any* of the six
stats or *any* of the six selected means before it splits into train and
target, so its marginals are fitted on the intersection across stats. Those
are the marginals behind the committed ``cdf_lower_``/``cdf_upper_`` columns
and therefore behind every latent estimator.

``04_validate_shadow_v1.py::fit_season`` drops only the stat it is fitting
and its own mean, so its marginals are fitted on a per-stat superset. Those
are the marginals the held-out simulation inverts, so they are the ones that
set the count-space transmission.

The two give CDF bounds that differ by up to ~1e-2 on ``pts``, ``ast``,
``stl`` and ``fg3m``. That is a pre-existing pipeline inconsistency, not
something this study introduces; it is recorded in the report and each stage
states which convention it used.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nba_prop_quant.adaptive_training import load_script_module
from nba_prop_quant.research.game_latent_state.bridge import (
    DiscreteMarginal,
    discrete_marginal,
)
from nba_prop_quant.research.game_latent_state.covariance import (
    SharedFactorLoadings,
)
from nba_prop_quant.research.game_latent_state.simulator import (
    tabulate_inverse_cdf,
)
from nba_prop_quant.research.game_latent_state.validation import (
    DEPENDENCE_BUCKETS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3m")

#: The frozen production marginal family. Declared, never selected here.
FROZEN_MARGINAL_FAMILY = "zinb"

#: Seasons the study is forbidden to read for any model decision.
HOLDOUT_SEASONS: tuple[int, ...] = (2024, 2025)

#: Seasons every estimator and every selection in this study may read.
PRE_2024_SEASONS: tuple[int, ...] = (2020, 2021, 2022, 2023)

#: Pre-2024 seasons with at least two earlier seasons of marginal training,
#: so a forward fold has the same no-lookahead structure the holdout has.
PRE_2024_FORWARD_FOLDS: tuple[int, ...] = (2022, 2023)

#: The twelve cross-player buckets the dependence layer is judged on.
CROSS_PLAYER_BUCKETS = tuple(
    (name, kind, pair)
    for name, kind, pair in DEPENDENCE_BUCKETS
    if kind in {"same_team", "cross_team"}
)

#: The focal bucket of the study.
FOCAL_BUCKET = "passer_ast_teammate_pts"

#: Expected-minutes floor the held-out simulation applies to a roster. The
#: forensic pair universe has to use the same one or it is not measuring the
#: same bucket.
MIN_EXPECTED_MINUTES = 8.0

#: Held-out games per season the paired runs simulated.
GAMES_PER_SEASON = 200

#: Count-space error reduction the study is testing for feasibility.
TARGET_ERROR_REDUCTION = 0.20

#: Tolerance multiplier the global RMSE constraints carry.
RMSE_TOLERANCE = 1.03


# ----------------------------------------------------------------------
# marginals
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MarginalSet:
    """One season's walk-forward production marginals, with provenance."""

    season: int
    convention: str
    training_seasons: tuple[int, ...]
    training_rows: int
    fitted: Mapping[str, object]


def fit_walk_forward_marginals(
    history: pd.DataFrame,
    season: int,
    stats: Sequence[str] = STATS,
    convention: str = "validator",
) -> MarginalSet:
    """Refit the production marginals on seasons strictly before ``season``.

    ``convention`` picks which of the two pipeline row filters to reproduce:
    ``"validator"`` for ``fit_season``'s per-stat dropna, ``"residual_build"``
    for ``build_residuals``' joint dropna across all stats and means. The
    marginal mathematics is loaded from ``scripts/07_fit_marginals.py`` in
    both cases, so neither path can drift from the certified one.
    """
    module = load_script_module(PROJECT_ROOT, "scripts/07_fit_marginals.py")
    selected = [f"mu_selected_{stat}" for stat in stats]
    frame = history
    if convention == "residual_build":
        frame = history.dropna(subset=[*stats, *selected])
    elif convention != "validator":
        raise ValueError(f"unknown marginal convention {convention!r}")

    train = frame.loc[frame["season"].astype(int) < int(season)]
    fitted: dict[str, object] = {}
    for stat in stats:
        rows = train.dropna(subset=[stat, f"mu_selected_{stat}"])
        fitted[stat] = module.fit_candidate(
            FROZEN_MARGINAL_FAMILY,
            y=rows[stat].to_numpy(dtype=int),
            mu=rows[f"mu_selected_{stat}"].to_numpy(dtype=float),
            frame=rows,
            inflation_features=module.inflation_features_for(stat, rows),
        )
    return MarginalSet(
        season=int(season),
        convention=convention,
        training_seasons=tuple(
            sorted(int(value) for value in train["season"].unique())
        ),
        training_rows=int(len(train)),
        fitted=fitted,
    )


def cached_marginals(
    cache_dir: Path,
    history: pd.DataFrame,
    season: int,
    convention: str,
    stats: Sequence[str] = STATS,
) -> MarginalSet:
    """A disk-cached :func:`fit_walk_forward_marginals`.

    A ZINB refit over four seasons of rows costs about a minute per season and
    the study needs several, so the fits are cached. The cache key carries the
    convention, which is what makes the two row filters distinguishable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"marginals_{convention}_{int(season)}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    result = fit_walk_forward_marginals(history, season, stats, convention)
    with path.open("wb") as handle:
        pickle.dump(result, handle)
    return result


def tabulate_rows(
    frame: pd.DataFrame,
    marginals: Mapping[str, object],
    positions: Sequence[int],
    stats: Sequence[str] = STATS,
) -> dict[tuple[int, str], np.ndarray]:
    """``F(0..K)`` for the referenced rows only.

    The bridges pool over a sample of pairs, so only the margins that sample
    references need tabulating. Tabulating the whole frame would cost an order
    of magnitude more for margins no bridge reads.
    """
    out: dict[tuple[int, str], np.ndarray] = {}
    for position in sorted(set(int(value) for value in positions)):
        row = frame.iloc[position]
        for stat in stats:
            out[(position, stat)] = tabulate_inverse_cdf(
                marginals[stat], float(row[f"mu_selected_{stat}"]), row
            )
    return out


# ----------------------------------------------------------------------
# pair universes
# ----------------------------------------------------------------------


def ordered_pair_mean(
    frame: pd.DataFrame,
    kind: str,
    column_a: str,
    column_b: str,
) -> tuple[float, float]:
    """``(mean, pair count)`` of ``a_i b_j`` over the bucket's ordered pairs.

    Reproduces :func:`factors.pair_moments` for a single ordered stat pair:
    the same-team term is ``outer(total, total) - z' z`` over ``n (n - 1)``
    pairs and the cross-team term pools both orientations over ``2 n_1 n_2``.
    """
    total = 0.0
    pairs = 0.0
    if kind == "same_team":
        for _, team in frame.groupby(["game_id", "team_id"], sort=True):
            first = team[column_a].to_numpy(dtype=float)
            second = team[column_b].to_numpy(dtype=float)
            if first.size < 2:
                continue
            total += float(first.sum() * second.sum() - first @ second)
            pairs += float(first.size * (first.size - 1))
    elif kind == "cross_team":
        for _, game in frame.groupby("game_id", sort=True):
            sides = [team for _, team in game.groupby("team_id", sort=True)]
            if len(sides) != 2:
                continue
            left, right = sides
            total += float(
                left[column_a].to_numpy(dtype=float).sum()
                * right[column_b].to_numpy(dtype=float).sum()
                + right[column_a].to_numpy(dtype=float).sum()
                * left[column_b].to_numpy(dtype=float).sum()
            )
            pairs += 2.0 * float(len(left) * len(right))
    else:
        raise ValueError(f"unsupported bucket kind {kind!r}")
    if pairs <= 0.0:
        raise ValueError("no pairs available for this bucket")
    return total / pairs, pairs


def clustered_bootstrap_pair_mean(
    frame: pd.DataFrame,
    kind: str,
    column_a: str,
    column_b: str,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Game-clustered bootstrap of :func:`ordered_pair_mean`.

    Accumulates each game's numerator and pair count once, then resamples the
    per-game pairs, which is what :func:`factors.pair_moments` does and costs
    one pass over the frame instead of one per draw.
    """
    numerators: list[float] = []
    denominators: list[float] = []
    for _, game in frame.groupby("game_id", sort=True):
        if kind == "same_team":
            total = 0.0
            pairs = 0.0
            for _, team in game.groupby("team_id", sort=True):
                first = team[column_a].to_numpy(dtype=float)
                second = team[column_b].to_numpy(dtype=float)
                if first.size < 2:
                    continue
                total += float(first.sum() * second.sum() - first @ second)
                pairs += float(first.size * (first.size - 1))
        else:
            sides = [team for _, team in game.groupby("team_id", sort=True)]
            if len(sides) != 2:
                continue
            left, right = sides
            total = float(
                left[column_a].to_numpy(dtype=float).sum()
                * right[column_b].to_numpy(dtype=float).sum()
                + right[column_a].to_numpy(dtype=float).sum()
                * left[column_b].to_numpy(dtype=float).sum()
            )
            pairs = 2.0 * float(len(left) * len(right))
        if pairs <= 0.0:
            continue
        numerators.append(total)
        denominators.append(pairs)

    numerator = np.asarray(numerators, dtype=float)
    denominator = np.asarray(denominators, dtype=float)
    generator = np.random.default_rng(seed)
    out = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        picks = generator.integers(0, numerator.size, size=numerator.size)
        out[draw] = float(numerator[picks].sum() / denominator[picks].sum())
    return out


def sampled_pairs(
    frame: pd.DataFrame,
    kind: str,
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    """A deterministic sample of ordered cross-player row-position pairs.

    Drawn by game, and by team for the same-team block, so the sampled pairs
    carry the joint distribution of margins the pooled moment averages over.
    Sampling rows independently would pair starters with starters far too
    often and bias every transmission estimate toward high-volume margins.
    """
    positions = frame.reset_index(drop=True)
    generator = np.random.default_rng(seed)
    if kind == "same_team":
        groups = [
            group.index.to_numpy()
            for _, group in positions.groupby(["game_id", "team_id"], sort=True)
            if len(group) >= 2
        ]
        if not groups:
            raise ValueError("no same-team group available for sampling")
        order = generator.permutation(len(groups))
        out: list[tuple[int, int]] = []
        cursor = 0
        while len(out) < count:
            members = groups[order[cursor % len(order)]]
            first, second = generator.choice(len(members), size=2, replace=False)
            out.append((int(members[first]), int(members[second])))
            cursor += 1
        return out

    if kind == "cross_team":
        sides: list[tuple[np.ndarray, np.ndarray]] = []
        for _, game in positions.groupby("game_id", sort=True):
            teams = [
                team.index.to_numpy()
                for _, team in game.groupby("team_id", sort=True)
            ]
            if len(teams) == 2 and all(len(team) >= 1 for team in teams):
                sides.append((teams[0], teams[1]))
        if not sides:
            raise ValueError("no cross-team game available for sampling")
        order = generator.permutation(len(sides))
        out = []
        cursor = 0
        while len(out) < count:
            home, away = sides[order[cursor % len(order)]]
            out.append(
                (
                    int(home[generator.integers(0, len(home))]),
                    int(away[generator.integers(0, len(away))]),
                )
            )
            cursor += 1
        return out

    raise ValueError(f"unsupported bucket kind {kind!r}")


# ----------------------------------------------------------------------
# the exact transmission map
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TransmissionPairs:
    """Mehler scores and scales of a sampled pair set, ready to evaluate.

    ``scores_a``/``scores_b`` are ``(n_pairs, terms)`` orthonormal-Hermite
    score matrices of the two ends, ``scale`` the product of the two role
    scales the simulator applies to that pair, and ``denominator`` the
    product of the two analytic standard deviations in count space (unity in
    latent space, where the reading is already standardized).
    """

    space: str
    stat_a: str
    stat_b: str
    scores_a: np.ndarray
    scores_b: np.ndarray
    scale: np.ndarray
    denominator: np.ndarray

    def implied(self, latent: float) -> float:
        """The pooled reading ``space`` would show at latent ``rho``.

        Mehler's expansion makes the per-pair bridge a power series whose
        coefficients are the two ends' scores, so the pooled reading is

            mean_pairs sum_j d_j^a d_j^b (s_a s_b rho)^(j+1)
                       / ((j + 1) sd_a sd_b)

        exactly, with the role scales entering pair by pair rather than as a
        pooled average. ``rho = 0`` gives zero in either space for any
        margins, which the series satisfies term by term.
        """
        terms = self.scores_a.shape[1]
        order = np.arange(terms, dtype=float)
        argument = self.scale * float(latent)
        powers = argument[:, None] ** (order + 1.0)
        per_pair = np.sum(
            self.scores_a * self.scores_b * powers / (order + 1.0), axis=1
        )
        return float(np.mean(per_pair / self.denominator))

    def invert(self, target: float, bound: float = 0.60) -> float:
        """The latent ``rho`` whose pooled reading is ``target``.

        The map is strictly increasing through the origin at these
        magnitudes, so a bisection on a symmetric bracket is both sufficient
        and immune to the series' behaviour far outside it.
        """
        low, high = -abs(bound), abs(bound)
        if not self.implied(low) <= target <= self.implied(high):
            raise ValueError(
                f"target {target!r} lies outside the reachable range "
                f"[{self.implied(low)!r}, {self.implied(high)!r}]"
            )
        for _ in range(200):
            middle = 0.5 * (low + high)
            if self.implied(middle) < target:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)


def build_transmission_pairs(
    frame: pd.DataFrame,
    grids: Mapping[tuple[int, str], np.ndarray],
    pairs: Sequence[tuple[int, int]],
    stat_a: str,
    stat_b: str,
    space: str,
    role_scale: Mapping[str, float] | None = None,
    terms: int = 48,
) -> TransmissionPairs:
    """Score a sampled pair set for one stat pair in one space.

    Both orientations of the stat pair are pooled when the two stats differ,
    which is what makes the result comparable with the bucket statistic: the
    bucket's mean over ordered pairs is the same for ``(a of i, b of j)`` and
    ``(b of i, a of j)`` by relabelling, so a bridge that pools one and not
    the other is measuring a different thing.
    """
    from nba_prop_quant.research.game_latent_state.bridge import mehler_scores

    cache: dict[tuple[int, str], DiscreteMarginal] = {}

    def margin(position: int, stat: str) -> DiscreteMarginal:
        key = (position, stat)
        if key not in cache:
            cache[key] = discrete_marginal(grids[key])
        return cache[key]

    score_cache: dict[tuple[int, str], np.ndarray] = {}

    def score(position: int, stat: str) -> np.ndarray:
        key = (position, stat)
        if key not in score_cache:
            score_cache[key] = mehler_scores(margin(position, stat), space, terms)
        return score_cache[key]

    role_column = "role_bucket" if role_scale is not None else None
    positions = frame.reset_index(drop=True)

    def scale_of(position: int) -> float:
        if role_scale is None or role_column is None:
            return 1.0
        return float(role_scale.get(str(positions.iloc[position][role_column]), 1.0))

    rows_a: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []
    scales: list[float] = []
    denominators: list[float] = []
    orientations = (
        ((stat_a, stat_b),)
        if stat_a == stat_b
        else ((stat_a, stat_b), (stat_b, stat_a))
    )
    for first, second in pairs:
        for left_stat, right_stat in orientations:
            rows_a.append(score(first, left_stat))
            rows_b.append(score(second, right_stat))
            scales.append(scale_of(first) * scale_of(second))
            denominators.append(
                margin(first, left_stat).sd * margin(second, right_stat).sd
                if space == "count"
                else 1.0
            )
    return TransmissionPairs(
        space=space,
        stat_a=stat_a,
        stat_b=stat_b,
        scores_a=np.asarray(rows_a, dtype=float),
        scores_b=np.asarray(rows_b, dtype=float),
        scale=np.asarray(scales, dtype=float),
        denominator=np.asarray(denominators, dtype=float),
    )


# ----------------------------------------------------------------------
# architecture probes
# ----------------------------------------------------------------------


class ShiftNotRepresentable(ValueError):
    """A requested same-team shift leaves one of the two Grams indefinite."""


def _refactor_gram(gram: np.ndarray, label: str) -> np.ndarray:
    """A real loading matrix whose Gram is ``gram``, or a refusal.

    The architecture stores factors, not Grams, so a retargeted Gram has to be
    handed back as columns. Eigendecomposition gives them whenever the Gram is
    PSD; a negative eigenvalue beyond float64 noise means the requested block
    is not a covariance at all, which is a result rather than something to
    clip away.
    """
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    floor = -1e-12 * max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < floor:
        raise ShiftNotRepresentable(
            f"{label} Gram has minimum eigenvalue {float(np.min(values)):.6e}"
        )
    return vectors * np.sqrt(np.clip(values, 0.0, None))


def isolated_same_team_shift(
    loadings: SharedFactorLoadings,
    stat_a: str,
    stat_b: str,
    shift: float,
) -> SharedFactorLoadings:
    """Move one off-diagonal same-team entry by ``shift`` and nothing else.

    The architecture writes the same-team block as ``S = A + B - Q`` and the
    cross-team block as ``X = A - B``, with ``A = L_game L_game'`` and
    ``B = L_contrast L_contrast'`` free PSD Grams of rank up to the stat
    count. Adding the same symmetric bump ``(shift / 2) E`` to both Grams, for
    ``E = e_a e_b' + e_b e_a'``, moves ``S`` by ``shift E`` and leaves ``X``
    identically alone. So one same-team off-diagonal entry is retargetable
    without disturbing any cross-team bucket, any other same-team bucket, or
    the same-stat diagonals.

    ``E`` is indefinite, so the bump cannot be realised as an appended factor
    column; the retargeted Grams are refactorised instead. Both already carry
    full rank, so the refactorisation stays inside the frozen architecture
    rather than widening it, and it raises :class:`ShiftNotRepresentable`
    rather than clipping if a Gram stops being a covariance. That is the real
    structural limit on this lever: it is bounded by the smaller of the two
    Grams' minimum eigenvalues, not by the rank.
    """
    if stat_a == stat_b:
        raise ValueError("an isolated shift needs two distinct stats")
    index = {stat: position for position, stat in enumerate(loadings.stats)}
    size = len(loadings.stats)
    bump = np.zeros((size, size), dtype=float)
    bump[index[stat_a], index[stat_b]] = 0.5 * float(shift)
    bump[index[stat_b], index[stat_a]] = 0.5 * float(shift)

    contrast = loadings.contrast_matrix
    game = _refactor_gram(loadings.game @ loadings.game.T + bump, "game")
    team_contrast = _refactor_gram(contrast @ contrast.T + bump, "team_contrast")
    return SharedFactorLoadings(
        stats=loadings.stats,
        game=game,
        team_contrast=team_contrast,
        competition=loadings.competition,
        role_scale=dict(loadings.role_scale),
        symmetric=loadings.symmetric,
        role_deviation=loadings.role_deviation,
        role_offset=dict(loadings.role_offset),
        role_pair_shares=dict(loadings.role_pair_shares),
    )


@dataclass(frozen=True)
class Retarget:
    """A retargeted loading set and what it cost to represent."""

    loadings: SharedFactorLoadings
    #: Isotropic inflation of the competition Gram the representation needed.
    competition_inflation: float
    min_game_eigenvalue: float
    min_contrast_eigenvalue: float
    min_competition_eigenvalue: float


def retarget_same_team_entry(
    loadings: SharedFactorLoadings,
    stat_a: str,
    stat_b: str,
    target: float,
) -> Retarget:
    """Put one same-team off-diagonal entry on ``target``, moving nothing else.

    This is the architecture's *general* lever, and it is worth being precise
    about why it reaches further than appending a factor column does. The
    three Grams satisfy ``S = A + B - Q`` and ``X = A - B``, so fixing the two
    observable blocks leaves ``Q`` free and determines the other two::

        A = (S + X + Q) / 2,      B = (S - X + Q) / 2

    Any ``(S, X)`` is therefore representable for a large enough ``Q``,
    because adding ``c I`` to ``Q`` adds ``(c / 2) I`` to both ``A`` and ``B``.
    The frozen candidate's own Grams sit almost on the PSD boundary -- the
    contrast Gram's smallest eigenvalue is 7.6e-19 -- so a bump added to them
    directly is refused immediately, which is exactly why the representation
    has to go through ``Q`` instead. The minimal such ``c`` is taken, so the
    retarget never inflates ``Q`` more than representability demands.

    Inflating ``Q`` is not free. It appears in a player's *own* block as
    ``A + B + (n - 1) Q = S + n Q``, and ``build_game_covariance`` pins that
    block by shrinking the player's shared loadings until the residual is PSD.
    So the lever's cost is paid in the per-player shrink, which scales every
    realised cross-player correlation back down. That trade is the feasibility
    envelope, and it is measured rather than assumed.
    """
    if stat_a == stat_b:
        raise ValueError("this lever moves off-diagonal entries only")
    index = {stat: position for position, stat in enumerate(loadings.stats)}
    size = len(loadings.stats)
    same = loadings.same_team_correlation().copy()
    cross = loadings.cross_team_correlation()
    same[index[stat_a], index[stat_b]] = float(target)
    same[index[stat_b], index[stat_a]] = float(target)

    base = loadings.competition_gram()
    game_base = 0.5 * (same + cross + base)
    contrast_base = 0.5 * (same - cross + base)
    floor = min(
        float(np.min(np.linalg.eigvalsh(0.5 * (game_base + game_base.T)))),
        float(np.min(np.linalg.eigvalsh(0.5 * (contrast_base + contrast_base.T)))),
    )
    inflation = max(0.0, -2.0 * floor)
    competition = base + inflation * np.eye(size)
    game_gram = game_base + 0.5 * inflation * np.eye(size)
    contrast_gram = contrast_base + 0.5 * inflation * np.eye(size)

    retargeted = SharedFactorLoadings(
        stats=loadings.stats,
        game=_refactor_gram(game_gram, "game"),
        team_contrast=_refactor_gram(contrast_gram, "team_contrast"),
        competition=_refactor_gram(competition, "competition"),
        role_scale=dict(loadings.role_scale),
        symmetric=loadings.symmetric,
        role_deviation=loadings.role_deviation,
        role_offset=dict(loadings.role_offset),
        role_pair_shares=dict(loadings.role_pair_shares),
    )
    return Retarget(
        loadings=retargeted,
        competition_inflation=inflation,
        min_game_eigenvalue=float(np.min(np.linalg.eigvalsh(game_gram))),
        min_contrast_eigenvalue=float(np.min(np.linalg.eigvalsh(contrast_gram))),
        min_competition_eigenvalue=float(np.min(np.linalg.eigvalsh(competition))),
    )


def bucket_readout(
    loadings: SharedFactorLoadings,
) -> dict[str, float]:
    """The twelve cross-player buckets implied by a set of loadings."""
    index = {stat: position for position, stat in enumerate(loadings.stats)}
    same = loadings.same_team_correlation()
    cross = loadings.cross_team_correlation()
    return {
        name: float(
            (same if kind == "same_team" else cross)[index[first], index[second]]
        )
        for name, kind, (first, second) in CROSS_PLAYER_BUCKETS
    }
