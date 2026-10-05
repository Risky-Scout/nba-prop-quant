"""Count-space plumbing shared by the Shadow V2 drivers.

SHADOW / RESEARCH ONLY. NOT PROMOTABLE.

The latent-space research only ever needed the residual dataset's
Gaussianized ``z`` columns. Count space needs the *marginals themselves*: an
observed count-space correlation is ``(y - E[Y]) / sd(Y)`` pooled over pairs,
and a bridge needs the whole tabulated ``F(0..K)`` per player-game-stat. Both
come from the production ZINB fit, so this module holds the refit, the feature
join the residual dataset cannot supply on its own, the tabulation, and the
pooled bridge construction.

Why the bridge replaces a simulation in the inner screen
--------------------------------------------------------
A Gaussian-copula pair ``(stat_a of player a, stat_b of player b)`` has a
bivariate law that depends on nothing but its two margins and the single
latent correlation between them. :mod:`bridge` maps that correlation to the
implied count-space correlation exactly, by deterministic quadrature, and the
map is verified against Monte Carlo in the test suite. So the count-space
consequence of a fitted latent block can be *computed* rather than simulated,
which is what makes screening four components over chronological folds
affordable without spending an 80-minute simulation on each.

The margins the prediction uses are the evaluation season's own, which are
pregame quantities: ``mu_selected_{stat}`` comes from the walk-forward mean
model and the ZINB parameters come from seasons strictly before the fold. No
realised outcome of the evaluation season enters the predicted correlation.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .bridge import (
    DEFAULT_GAUSS_NODES,
    BridgeCurve,
    BridgeNotIdentified,
    DiscreteMarginal,
    build_bridge_curve,
    discrete_marginal,
)
from .simulator import tabulate_inverse_cdf

#: The frozen production marginal family. Declared, never selected here.
FROZEN_MARGINAL_FAMILY = "zinb"

#: Ordered pairs a pooled bridge is averaged over. The bridge is a smooth
#: functional of the two margins, so the pooled mean converges quickly.
DEFAULT_BRIDGE_PAIRS = 2000

#: Quadrature resolution for a pooled bridge. The kernel is analytic, so this
#: is exact to float64 at the correlation magnitudes this layer produces.
DEFAULT_BRIDGE_PANELS = 20
DEFAULT_BRIDGE_RHO_MAX = 0.40


def fit_training_marginals(
    project_root: Path,
    history: pd.DataFrame,
    season: int,
    stats: Sequence[str],
    family: str = FROZEN_MARGINAL_FAMILY,
) -> dict[str, object]:
    """Refit the production marginals on seasons strictly before ``season``.

    Identical in construction to ``04_validate_shadow_v1.fit_season``, minus
    the incumbent copula the count-space work does not need. The pipeline
    script is loaded rather than copied, so there is one definition of the
    marginal mathematics and the research path cannot drift from the certified
    one.
    """
    from ...adaptive_training import load_script_module

    module = load_script_module(project_root, "scripts/07_fit_marginals.py")
    train = history.loc[history["season"] < int(season)]
    out: dict[str, object] = {}
    for stat in stats:
        rows = train.dropna(subset=[stat, f"mu_selected_{stat}"])
        out[stat] = module.fit_candidate(
            family,
            y=rows[stat].to_numpy(dtype=int),
            mu=rows[f"mu_selected_{stat}"].to_numpy(dtype=float),
            frame=rows,
            inflation_features=module.inflation_features_for(stat, rows),
        )
    return out


def attach_production_features(
    residuals: pd.DataFrame,
    history: pd.DataFrame,
    stats: Sequence[str],
) -> pd.DataFrame:
    """Join the ZINB inflation features onto the residual rows.

    The residual dataset carries ``mu_{stat}`` but not the priors, pace and
    position indicators the fitted marginals condition on, so a CDF cannot be
    tabulated from it alone. ``oof_selected_means.parquet`` is both the frame
    the marginals were fitted on and the frame the validation driver builds
    its rosters from, so taking the features from there keeps the bridge
    reading exactly the inputs the simulator reads. The merge is asserted
    one-to-one: a duplicated key would silently double-weight a player-game in
    every pooled moment downstream.
    """
    keys = ["game_id", "player_id"]
    carried = [
        column
        for column in history.columns
        if column not in keys
        and (column not in residuals.columns or column.startswith("mu_selected_"))
    ]
    merged = residuals.merge(
        history[[*keys, *carried]], on=keys, how="left", validate="one_to_one"
    )
    required = [f"mu_selected_{stat}" for stat in stats]
    unmatched = int(merged[required].isna().any(axis=1).sum())
    if unmatched:
        raise AssertionError(
            f"{unmatched} residual rows have no production feature row; their "
            "marginals cannot be tabulated"
        )
    return merged


def tabulate_grids(
    frame: pd.DataFrame,
    marginals: Mapping[str, object],
    stats: Sequence[str],
    progress: object | None = None,
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    """Analytic mean/sd and the tabulated CDF per player-game-stat.

    Returns the frame with ``e_{stat}`` standardized count residuals attached
    and a mapping from ``(row position, stat)`` to ``F(0..K)``. The grids stay
    in memory: they are a derived intermediate of the committed marginal fit
    and reproducible from it, so writing them out would add a large generated
    artifact for nothing.
    """
    out = frame.reset_index(drop=True).copy()
    grids: dict[tuple[int, str], np.ndarray] = {}
    means = {stat: np.empty(len(out), dtype=float) for stat in stats}
    sds = {stat: np.empty(len(out), dtype=float) for stat in stats}
    started = time.time()
    # One row extraction per row rather than one per row-stat: the inflation
    # features are shared across the stats and the extraction dominates.
    for position in range(len(out)):
        row = out.iloc[position]
        for stat in stats:
            cdf = tabulate_inverse_cdf(
                marginals[stat], float(row[f"mu_selected_{stat}"]), row
            )
            grids[(position, stat)] = cdf
            survival = 1.0 - cdf
            mean = float(np.sum(survival))
            counts = np.arange(cdf.size)
            second = float(np.sum((2.0 * counts + 1.0) * survival))
            means[stat][position] = mean
            sds[stat][position] = float(np.sqrt(max(second - mean**2, 1e-12)))
        if progress is not None and position and position % 20000 == 0:
            progress.print(
                f"  {position:,}/{len(out):,} rows  "
                f"{time.time() - started:6.1f}s elapsed",
                highlight=False,
            )
    for stat in stats:
        out[f"analytic_mean_{stat}"] = means[stat]
        out[f"analytic_sd_{stat}"] = sds[stat]
        out[f"e_{stat}"] = (
            out[f"y_{stat}"].to_numpy(dtype=float) - means[stat]
        ) / sds[stat]
    if progress is not None:
        progress.print(
            f"  tabulated {len(out):,} rows in {time.time() - started:.1f}s"
        )
    return out, grids


def sample_pairs(
    frame: pd.DataFrame,
    block: str,
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    """A deterministic sample of ordered cross-player row pairs.

    Drawn by game (and by team, for the same-team block) so the sampled pairs
    carry the joint distribution of the two margins that the pooled moment
    actually averages over. Sampling rows independently would pair a starter
    with a starter far too often and bias every bridge toward the high-volume
    margins.
    """
    rng = np.random.default_rng(seed)
    if block == "same_team":
        groups = [
            group.index.to_numpy()
            for _, group in frame.groupby(["game_id", "team_id"], sort=True)
            if len(group) >= 2
        ]
        if not groups:
            raise ValueError("no same-team groups available for bridge sampling")
        order = rng.permutation(len(groups))
        pairs: list[tuple[int, int]] = []
        position = 0
        while len(pairs) < count:
            members = groups[order[position % len(order)]]
            first, second = rng.choice(len(members), size=2, replace=False)
            pairs.append((int(members[first]), int(members[second])))
            position += 1
        return pairs

    if block == "cross_team":
        sides: list[tuple[np.ndarray, np.ndarray]] = []
        for _, game in frame.groupby("game_id", sort=True):
            teams = [
                team.index.to_numpy() for _, team in game.groupby("team_id", sort=True)
            ]
            if len(teams) == 2 and all(len(team) >= 1 for team in teams):
                sides.append((teams[0], teams[1]))
        if not sides:
            raise ValueError("no cross-team game pairs available for bridge sampling")
        order = rng.permutation(len(sides))
        pairs = []
        position = 0
        while len(pairs) < count:
            home, away = sides[order[position % len(order)]]
            pairs.append(
                (
                    int(home[rng.integers(0, len(home))]),
                    int(away[rng.integers(0, len(away))]),
                )
            )
            position += 1
        return pairs

    raise ValueError(f"unknown block {block!r}")


def prepare_marginals(
    grids: Mapping[tuple[int, str], np.ndarray],
    pairs: Sequence[tuple[int, int]],
    stats: Sequence[str],
) -> dict[tuple[int, str], DiscreteMarginal]:
    """Truncate and score the CDF grids the sampled pairs reference."""
    wanted = {
        (position, stat) for pair in pairs for position in pair for stat in stats
    }
    return {key: discrete_marginal(grids[key]) for key in wanted}


def pooled_bridge_curves(
    prepared: Mapping[tuple[int, str], DiscreteMarginal],
    pairs: Sequence[tuple[int, int]],
    stats: Sequence[str],
    space: str = "count",
    rho_max: float = DEFAULT_BRIDGE_RHO_MAX,
    panels: int = DEFAULT_BRIDGE_PANELS,
    nodes: int = DEFAULT_GAUSS_NODES,
    progress: object | None = None,
) -> dict[tuple[str, str], BridgeCurve]:
    """One pooled bridge per unordered stat pair.

    The bucket statistic is a mean over *ordered* pairs, and that mean is the
    same for ``(stat_a of a, stat_b of b)`` and ``(stat_b of a, stat_a of b)``
    by relabelling, so one curve per unordered stat pair is the right object.
    Both orderings are pooled into it.
    """
    stats = tuple(stats)
    out: dict[tuple[str, str], BridgeCurve] = {}
    started = time.time()
    for position, first_stat in enumerate(stats):
        for second_stat in stats[position:]:
            marginal_pairs: list[tuple[DiscreteMarginal, DiscreteMarginal]] = []
            for a, b in pairs:
                marginal_pairs.append(
                    (prepared[(a, first_stat)], prepared[(b, second_stat)])
                )
                if first_stat != second_stat:
                    marginal_pairs.append(
                        (prepared[(a, second_stat)], prepared[(b, first_stat)])
                    )
            curve = build_bridge_curve(
                marginal_pairs,
                space=space,
                rho_max=rho_max,
                panels=panels,
                nodes=nodes,
            )
            out[(first_stat, second_stat)] = curve
            out[(second_stat, first_stat)] = curve
            if progress is not None:
                progress.print(
                    f"  {space:6s} {first_stat}_{second_stat:5s} "
                    f"slope {curve.slope_at_zero():.4f}  "
                    f"({time.time() - started:5.1f}s)",
                    highlight=False,
                )
    return out


def apply_bridge(
    curves: Mapping[tuple[str, str], BridgeCurve],
    latent: np.ndarray,
    stats: Sequence[str],
) -> np.ndarray:
    """Map a fitted latent correlation block to its implied count block."""
    stats = tuple(stats)
    out = np.zeros_like(np.asarray(latent, dtype=float))
    for i, first_stat in enumerate(stats):
        for j, second_stat in enumerate(stats):
            out[i, j] = curves[(first_stat, second_stat)].evaluate(
                float(latent[i, j])
            )
    return out


def invert_bridge_block(
    curves: Mapping[tuple[str, str], BridgeCurve],
    observed: np.ndarray,
    stats: Sequence[str],
) -> tuple[np.ndarray, dict[str, str]]:
    """Invert a whole observed count block to the latent block it implies.

    Returns ``(required, rejections)``. An entry whose target lies outside the
    range the latent grid can reach is left at the observed value and recorded
    in ``rejections`` rather than clipped silently: an unidentified inversion
    is a result, not something to paper over.
    """
    stats = tuple(stats)
    observed = np.asarray(observed, dtype=float)
    required = observed.copy()
    rejections: dict[str, str] = {}
    for i, first_stat in enumerate(stats):
        for j in range(i, len(stats)):
            second_stat = stats[j]
            try:
                value = curves[(first_stat, second_stat)].invert(
                    float(observed[i, j])
                )
            except BridgeNotIdentified as error:
                rejections[f"{first_stat}_{second_stat}"] = str(error)
                continue
            required[i, j] = value
            required[j, i] = value
    return required, rejections
