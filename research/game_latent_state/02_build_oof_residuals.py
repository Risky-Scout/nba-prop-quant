"""Build the shadow OOF Gaussian residual dataset.

SHADOW / RESEARCH ONLY. Reads production code, writes research artifacts.

Pipeline, in the order the brief requires:

1.  Assemble the production feature frame from the research raw snapshot using
    the certified ``build_base_frame`` / ``add_dynamic_priors`` and the
    production ``models/dynamic_params.json``. No feature definition changes.
2.  Season walk-forward OOF expected minutes
    (``season_walk_forward_oof_minutes``), the only OOF methodology the
    adaptive contract permits.
3.  Season walk-forward OOF conditional means per target
    (``season_walk_forward_oof_target``).
4.  Walk-forward convex ensemble weights for the two ensemble-routed targets,
    fitted with ``scripts/06b_fit_mean_ensemble.py::fit_simplex_weights`` on
    seasons strictly before each validation season, then
    ``mu_selected_{target}`` materialised under ``FROZEN_MEAN_ROUTES``.
5.  Walk-forward ZINB marginals: for validation season S, the marginal is
    fitted with ``scripts/07_fit_marginals.py::fit_candidate`` on OOF rows
    from seasons strictly before S. Production fits marginals once on all
    eligible history; refitting them walk-forward here is strictly more
    conservative and removes the last channel by which a realized value could
    inform its own predictive distribution.
6.  Randomized PIT under a locked seed, then ``z = Phi^-1(u)``.

Leakage invariants enforced by construction:

* every predictive quantity attached to a row in season S is fitted only on
  seasons < S;
* no realized value from season S is used to build any parameter applied to
  season S;
* the randomized PIT draw is a keyed digest of the observation identity, so
  it cannot carry information from any other row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.adaptive_training import (
    ADVANCED_START_SEASON,
    CORE_SEED,
    FROZEN_MARGINAL_FAMILY,
    FROZEN_MEAN_ROUTES,
    SnapshotSettings,
    load_script_module,
)
from nba_prop_quant.features import add_dynamic_priors, build_base_frame
from nba_prop_quant.model import (
    season_walk_forward_oof_minutes,
    season_walk_forward_oof_target,
)
from nba_prop_quant.pipeline import (
    load_advanced,
    load_history_box_stats,
    load_history_games,
    load_json,
    load_players,
)
from nba_prop_quant.research.game_latent_state.artifacts import (
    ArtifactManifest,
    directory_fingerprint,
    finalize_manifest,
    git_sha,
    write_json,
)
from nba_prop_quant.research.game_latent_state.paths import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RESEARCH_DATA_ROOT,
    RESIDUAL_DATASET_NAME,
    research_processed_dir,
    research_raw_dir,
)
from nba_prop_quant.research.game_latent_state.pit import (
    PIT_EPSILON,
    deterministic_pit_uniform,
    gaussianize,
    mid_pit,
    randomized_pit,
)
from nba_prop_quant.research.game_latent_state.simulator import SUPPORTED_STATS

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# First season that can carry a walk-forward OOF conditional mean given the
# research raw window: the walk-forward needs at least one earlier season.
# First season that can carry a walk-forward *marginal* needs two earlier
# OOF seasons, hence the residual window starts later than the OOF window.
MIN_MARGINAL_TRAIN_SEASONS = 2

ROLE_BUCKET_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RESEARCH_DATA_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--seed", type=int, default=CORE_SEED)
    parser.add_argument(
        "--first-residual-season",
        type=int,
        default=2018,
        help="First season that receives a randomized-PIT residual.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=4, help="XGBoost threads for the OOF refits."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached intermediate stages.",
    )
    return parser.parse_args()


def build_features(data_root: Path, cache: Path, force: bool) -> pd.DataFrame:
    if cache.exists() and not force:
        console.print(f"features cache hit: {cache}")
        return pd.read_parquet(cache)

    snapshot = SnapshotSettings(raw_dir=research_raw_dir(data_root))
    stats = load_history_box_stats(snapshot)
    games = load_history_games(snapshot)
    players = load_players(snapshot)
    advanced = load_advanced(snapshot)

    if stats.empty:
        raise SystemExit("the research raw snapshot holds no box-score history")

    console.print(f"raw box rows: {len(stats):,}")
    base = build_base_frame(stats, players=players, advanced=advanced, games=games)
    params = load_json(PROJECT_ROOT / "models" / "dynamic_params.json")
    features = add_dynamic_priors(base, params)

    cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache, index=False)
    console.print(f"features: {len(features):,} rows x {len(features.columns)} columns")
    return features


def build_oof_means(
    features: pd.DataFrame,
    cache: Path,
    force: bool,
    n_jobs: int,
) -> pd.DataFrame:
    if cache.exists() and not force:
        console.print(f"OOF means cache hit: {cache}")
        return pd.read_parquet(cache)

    frame = (
        features.loc[features["season"] >= ADVANCED_START_SEASON]
        .sort_values(["date", "game_id", "player_id"])
        .reset_index(drop=True)
    )

    params = {"n_jobs": int(n_jobs)}

    console.rule("Walk-forward OOF expected minutes")
    frame["expected_minutes"] = season_walk_forward_oof_minutes(frame, params=params)

    frame = frame.loc[frame["expected_minutes"].notna()].copy()
    console.print(f"rows with OOF expected minutes: {len(frame):,}")

    for target in sorted(FROZEN_MEAN_ROUTES):
        console.rule(f"Walk-forward OOF conditional mean: {target}")
        frame[f"mu_{target}"] = season_walk_forward_oof_target(
            frame, target=target, params=params
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def add_walk_forward_selected_means(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict[str, dict[str, float]]]]:
    """Materialise ``mu_selected_{target}`` with walk-forward ensemble weights."""
    ensemble = load_script_module(PROJECT_ROOT, "scripts/06b_fit_mean_ensemble.py")

    seasons = sorted(int(season) for season in frame["season"].dropna().unique())
    season_values = frame["season"].astype(int).to_numpy()
    weights_by_season: dict[int, dict[str, dict[str, float]]] = {}

    for target, route in sorted(FROZEN_MEAN_ROUTES.items()):
        frame[f"mu_selected_{target}"] = np.nan
        if route == "xgb":
            frame[f"mu_selected_{target}"] = frame[f"mu_{target}"]

    for season in seasons:
        train_mask = season_values < season
        apply_mask = season_values == season
        if train_mask.sum() == 0:
            continue

        season_weights: dict[str, dict[str, float]] = {}
        for target, route in sorted(FROZEN_MEAN_ROUTES.items()):
            if route != "ensemble":
                continue
            columns = [
                f"mu_{target}",
                f"decay_prior_{target}_rate",
                f"kalman_prior_{target}_rate",
            ]
            train = frame.loc[train_mask].dropna(subset=[target, *columns])
            if train.empty:
                continue
            fitted = ensemble.fit_simplex_weights(
                train[target].to_numpy(dtype=float),
                train[columns].to_numpy(dtype=float),
            )
            season_weights[target] = {
                "xgb": float(fitted[0]),
                "decay": float(fitted[1]),
                "kalman": float(fitted[2]),
            }
            frame.loc[apply_mask, f"mu_selected_{target}"] = (
                float(fitted[0]) * frame.loc[apply_mask, f"mu_{target}"]
                + float(fitted[1]) * frame.loc[apply_mask, columns[1]]
                + float(fitted[2]) * frame.loc[apply_mask, columns[2]]
            )
        weights_by_season[int(season)] = season_weights

    return frame, weights_by_season


def add_role_and_availability_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add team-level availability context derived only from OOF quantities."""
    team_keys = ["game_id", "team_id"]
    grouped = frame.groupby(team_keys, sort=False)["expected_minutes"]
    frame["team_active_players"] = grouped.transform("size").astype(int)
    team_total = grouped.transform("sum")
    frame["team_expected_minutes_total"] = team_total
    frame["expected_minutes_share"] = frame["expected_minutes"] / team_total.where(
        team_total > 0
    )
    return frame


def assign_role_buckets(
    frame: pd.DataFrame,
    training_seasons: list[int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Coarse minutes-role buckets with thresholds taken from training only."""
    training = frame.loc[frame["season"].astype(int).isin(training_seasons)]
    source = training if not training.empty else frame
    low, high = np.nanquantile(
        source["expected_minutes"].to_numpy(dtype=float), ROLE_BUCKET_QUANTILES
    )
    minutes = frame["expected_minutes"].to_numpy(dtype=float)
    frame["role_bucket"] = np.where(
        minutes <= low, "bench", np.where(minutes <= high, "rotation", "starter")
    )
    return frame, {"low": float(low), "high": float(high)}


def build_residuals(
    frame: pd.DataFrame,
    residual_seasons: list[int],
    stats: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Randomized-PIT Gaussian residuals with walk-forward marginals."""
    marginals_module = load_script_module(PROJECT_ROOT, "scripts/07_fit_marginals.py")

    selected_columns = [f"mu_selected_{stat}" for stat in stats]
    usable = frame.dropna(subset=[*stats, *selected_columns]).copy()
    usable["season"] = usable["season"].astype(int)

    context_columns = [
        "game_id",
        "game_date",
        "season",
        "team_id",
        "opponent_id",
        "is_home",
        "player_id",
        "expected_minutes",
        "expected_minutes_share",
        "team_active_players",
        "team_expected_minutes_total",
        "role_bucket",
        "days_rest",
        "b2b",
        "season_progress",
        "prior_minutes10",
        "position",
        "postseason",
    ]
    usable["game_date"] = pd.to_datetime(usable["date"]).dt.normalize()

    records: list[pd.DataFrame] = []
    marginal_report: dict[str, object] = {"by_season": {}}

    for season in residual_seasons:
        train = usable.loc[usable["season"] < season]
        target_rows = usable.loc[usable["season"] == season]
        if target_rows.empty:
            continue
        if train["season"].nunique() < MIN_MARGINAL_TRAIN_SEASONS:
            console.print(
                f"[yellow]season {season}: only "
                f"{train['season'].nunique()} marginal training season(s); skipped"
                "[/yellow]"
            )
            continue

        console.rule(f"Randomized PIT residuals: season {season}")
        season_report: dict[str, object] = {
            "marginal_training_seasons": sorted(
                int(value) for value in train["season"].unique()
            ),
            "marginal_training_rows": len(train),
            "residual_rows": len(target_rows),
            "stats": {},
        }

        block = target_rows[context_columns].copy()
        for stat in stats:
            selected = f"mu_selected_{stat}"
            fit_rows = train.dropna(subset=[stat, selected])
            inflation = marginals_module.inflation_features_for(stat, fit_rows)
            marginal = marginals_module.fit_candidate(
                FROZEN_MARGINAL_FAMILY,
                y=fit_rows[stat].to_numpy(dtype=int),
                mu=fit_rows[selected].to_numpy(dtype=float),
                frame=fit_rows,
                inflation_features=inflation,
            )

            y = target_rows[stat].to_numpy(dtype=int)
            mu = target_rows[selected].to_numpy(dtype=float)
            lower = marginal.cdf(y - 1, mu, target_rows)
            upper = marginal.cdf(y, mu, target_rows)
            mass = marginal.pmf(y, mu, target_rows)

            draw = deterministic_pit_uniform(
                seed=seed,
                game_id=target_rows["game_id"].to_numpy(),
                player_id=target_rows["player_id"].to_numpy(),
                stat=stat,
            )
            u = randomized_pit(lower, upper, draw)
            gaussian = gaussianize(u, epsilon=PIT_EPSILON)

            block[f"y_{stat}"] = y
            block[f"mu_{stat}"] = mu
            block[f"u_{stat}"] = gaussian.u
            block[f"z_{stat}"] = gaussian.z
            block[f"v_{stat}"] = draw
            block[f"zmid_{stat}"] = _safe_norm_ppf(mid_pit(lower, mass))
            block[f"cdf_lower_{stat}"] = lower
            block[f"cdf_upper_{stat}"] = upper

            season_report["stats"][stat] = {
                "marginal_kind": marginal.kind,
                "inflation_features": list(inflation),
                "clipped_fraction": gaussian.clipped_fraction,
                "pit_mean": float(np.mean(gaussian.u)),
                "pit_variance": float(np.var(gaussian.u)),
                "z_mean": float(np.mean(gaussian.z)),
                "z_sd": float(np.std(gaussian.z)),
            }

        records.append(block)
        marginal_report["by_season"][str(season)] = season_report

    if not records:
        raise SystemExit("no residual seasons were produced")

    residuals = pd.concat(records, ignore_index=True)
    residuals = residuals.sort_values(["game_date", "game_id", "player_id"]).reset_index(
        drop=True
    )
    return residuals, marginal_report


def _safe_norm_ppf(u: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return norm.ppf(np.clip(np.asarray(u, dtype=float), PIT_EPSILON, 1.0 - PIT_EPSILON))


def main() -> None:
    args = parse_args()

    processed = research_processed_dir(args.data_root)
    artifact_dir = Path(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(
        args.data_root, processed / "features.parquet", args.force
    )
    oof = build_oof_means(
        features, processed / "oof_predictions.parquet", args.force, args.n_jobs
    )

    oof, ensemble_weights = add_walk_forward_selected_means(oof)
    oof = add_role_and_availability_context(oof)

    seasons = sorted(int(season) for season in oof["season"].dropna().unique())
    residual_seasons = [
        season for season in seasons if season >= int(args.first_residual_season)
    ]
    training_seasons = [
        season for season in residual_seasons if season <= residual_seasons[-3]
    ] or residual_seasons[:1]

    oof, role_thresholds = assign_role_buckets(oof, training_seasons)

    selected_path = processed / "oof_selected_means.parquet"
    oof.to_parquet(selected_path, index=False)
    console.print(f"wrote {selected_path}")

    residuals, marginal_report = build_residuals(
        oof, residual_seasons, SUPPORTED_STATS, seed=args.seed
    )

    residual_path = artifact_dir / RESIDUAL_DATASET_NAME
    residuals.to_parquet(residual_path, index=False)
    console.print(
        f"residual dataset: {len(residuals):,} player-game rows "
        f"over {residuals['game_id'].nunique():,} games"
    )

    report_path = write_json(
        {
            "ensemble_weights_by_season": ensemble_weights,
            "role_bucket_thresholds": role_thresholds,
            "marginals": marginal_report,
        },
        artifact_dir / "oof_residual_build_report.json",
    )

    manifest = ArtifactManifest(
        artifact_name="oof_gaussian_residuals",
        source_production_sha=git_sha(PROJECT_ROOT, "origin/production/wizardofodds-integration"),
        source_production_ref="origin/production/wizardofodds-integration",
        code_sha=git_sha(PROJECT_ROOT),
        branch="research/nba-game-latent-state-shadow-v1",
        seed=int(args.seed),
        training_cutoff=str(pd.to_datetime(residuals["game_date"]).max().date()),
        seasons_used=sorted(int(value) for value in residuals["season"].unique()),
        input_fingerprints=directory_fingerprint(research_raw_dir(args.data_root)),
        parameters={
            "stats": list(SUPPORTED_STATS),
            "frozen_mean_routes": dict(FROZEN_MEAN_ROUTES),
            "frozen_marginal_family": FROZEN_MARGINAL_FAMILY,
            "pit_epsilon": PIT_EPSILON,
            "advanced_start_season": ADVANCED_START_SEASON,
            "first_residual_season": int(args.first_residual_season),
            "marginal_fit": "walk-forward, seasons strictly before each residual season",
            "mean_fit": "season_walk_forward_oof_target / _minutes",
            "role_bucket_thresholds": role_thresholds,
        },
        notes=[
            "Randomized PIT with a keyed deterministic v; no in-sample residuals.",
            "Every parameter applied to season S is fitted on seasons < S only.",
            "Research raw window is narrower than production history_start_season.",
        ],
    )
    finalize_manifest(
        manifest,
        artifact_dir,
        outputs={
            RESIDUAL_DATASET_NAME: residual_path,
            "oof_residual_build_report.json": report_path,
        },
        manifest_name="residual_manifest.json",
        checksum_name="SHA256SUMS.residuals.txt",
    )

    console.rule("OOF residual dataset complete")
    console.print(json.dumps({"rows": len(residuals)}, indent=2))


if __name__ == "__main__":
    main()
