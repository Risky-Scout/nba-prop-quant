from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.api import BDLClient
from nba_prop_quant.availability import (
    apply_current_injury_adjustment,
)
from nba_prop_quant.experience import (
    apply_production_experience_curves,
)
from nba_prop_quant.model import ModelBundle
from nba_prop_quant.normalize import (
    normalize_games,
    normalize_injuries,
    normalize_players,
)
from nba_prop_quant.pipeline import (
    dynamic_params_path,
    load_advanced,
    load_history_box_stats,
    load_json,
)
from nba_prop_quant.production import (
    TARGETS,
    apply_selected_mean_policy,
    assert_model_feature_contract,
    load_verified_manifest_metadata,
    single_stat_quantile_summary,
)
from nba_prop_quant.settings import get_settings
from nba_prop_quant.slate import (
    build_upcoming_slate_features,
)
from nba_prop_quant.storage import (
    write_parquet_atomic,
)


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="YYYY-MM-DD",
    )

    parser.add_argument(
        "--minutes-overrides",
        type=Path,
        default=None,
        help=(
            "Optional CSV with "
            "player_id,expected_minutes columns."
        ),
    )

    parser.add_argument(
        "--allow-predeployment",
        action="store_true",
        help=(
            "Integration smoke-test only. "
            "Allows the pre-live baseline manifest "
            "before the external_test_deployment freeze."
        ),
    )

    parser.add_argument(
        "--allow-missing-model-features",
        action="store_true",
        help=(
            "Diagnostic escape hatch only. "
            "External-test deployment should never use this."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = get_settings()

    project_root = Path.cwd()

    manifest = load_verified_manifest_metadata(
        model_dir=settings.nba_prop_model_dir,
        project_root=project_root,
        allow_predeployment=(
            args.allow_predeployment
        ),
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=(
            settings.bdl_requests_per_minute
        ),
    ) as client:
        games = normalize_games(
            list(
                client.games(
                    dates=[
                        args.date
                    ]
                )
            )
        )

        if games.empty:
            console.print(
                f"[yellow]No games found "
                f"for {args.date}[/yellow]"
            )
            return

        team_ids = sorted(
            set(
                games[
                    "home_team_id"
                ].astype(
                    int
                )
            )
            | set(
                games[
                    "visitor_team_id"
                ].astype(
                    int
                )
            )
        )

        active = normalize_players(
            list(
                client.active_players(
                    team_ids=team_ids
                )
            )
        )

        injuries = normalize_injuries(
            list(
                client.injuries(
                    team_ids=team_ids
                )
            )
        )

    history = load_history_box_stats(
        settings
    )

    advanced = load_advanced(
        settings
    )

    dynamic_params = load_json(
        dynamic_params_path(
            settings
        )
    )

    slate = build_upcoming_slate_features(
        history_stats=history,
        upcoming_games=games,
        active_players=active,
        advanced=advanced,
        dynamic_params=dynamic_params,
    )

    if slate.empty:
        raise SystemExit(
            "ERROR: upcoming slate feature builder "
            "returned zero player rows."
        )

    experience_curves = joblib.load(
        settings.nba_prop_model_dir
        / "experience_curves.joblib"
    )

    slate = (
        apply_production_experience_curves(
            slate,
            experience_curves,
        )
    )

    minutes_model = ModelBundle.load(
        settings.nba_prop_model_dir
        / "minutes.joblib"
    )

    missing_minutes_features = (
        assert_model_feature_contract(
            slate,
            minutes_model,
            label="minutes model",
            allow_missing=(
                args.allow_missing_model_features
            ),
        )
    )

    if missing_minutes_features:
        console.print(
            "[yellow]Minutes model missing "
            f"{len(missing_minutes_features)} "
            "trained features under diagnostic "
            "override.[/yellow]"
        )

    slate[
        "expected_minutes_model"
    ] = minutes_model.predict(
        slate
    )

    slate[
        "expected_minutes"
    ] = slate[
        "expected_minutes_model"
    ]

    slate = (
        apply_current_injury_adjustment(
            slate,
            injuries=injuries,
            expected_minutes_col=(
                "expected_minutes"
            ),
        )
    )

    slate[
        "expected_minutes"
    ] = slate[
        "availability_expected_minutes"
    ]

    if (
        args.minutes_overrides
        is not None
    ):
        overrides = pd.read_csv(
            args.minutes_overrides
        )

        required = {
            "player_id",
            "expected_minutes",
        }

        missing = required - set(
            overrides.columns
        )

        if missing:
            raise ValueError(
                "Minutes override file missing "
                f"columns: {sorted(missing)}"
            )

        override_map = (
            overrides
            .set_index(
                "player_id"
            )[
                "expected_minutes"
            ]
            .to_dict()
        )

        mask = slate[
            "player_id"
        ].isin(
            override_map
        )

        slate.loc[
            mask,
            "expected_minutes",
        ] = slate.loc[
            mask,
            "player_id",
        ].map(
            override_map
        )

        slate.loc[
            mask,
            "minutes_override_applied",
        ] = 1

    slate[
        "minutes_override_applied"
    ] = (
        slate.get(
            "minutes_override_applied",
            0,
        )
    )

    slate.loc[
        slate[
            "availability_out"
        ].eq(
            1
        ),
        "expected_minutes",
    ] = 0.0

    slate, missing_target_features = (
        apply_selected_mean_policy(
            slate,
            model_dir=(
                settings.nba_prop_model_dir
            ),
            allow_missing_features=(
                args.allow_missing_model_features
            ),
        )
    )

    for target, missing in (
        missing_target_features.items()
    ):
        if missing:
            console.print(
                "[yellow]"
                f"{target} model missing "
                f"{len(missing)} trained "
                "features under diagnostic "
                "override.[/yellow]"
            )

    marginals = joblib.load(
        settings.nba_prop_model_dir
        / "marginals.joblib"
    )

    mu_columns = {
        target: (
            f"mu_selected_{target}"
        )
        for target in TARGETS
    }

    summaries = []

    for _, row in slate.iterrows():
        if int(
            row[
                "availability_out"
            ]
        ) == 1:
            summaries.append(
                {
                    f"{target}_{suffix}": np.nan
                    for target in TARGETS
                    for suffix in [
                        "mean",
                        "p10",
                        "p50",
                        "p90",
                    ]
                }
            )

            continue

        summaries.append(
            single_stat_quantile_summary(
                row,
                marginals=marginals,
                mu_columns=mu_columns,
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    slate = pd.concat(
        [
            slate.reset_index(
                drop=True
            ),
            summary_df,
        ],
        axis=1,
    )

    names = active[
        [
            "id",
            "first_name",
            "last_name",
        ]
    ].rename(
        columns={
            "id": "player_id"
        }
    )

    slate = slate.merge(
        names,
        on="player_id",
        how="left",
    )

    slate[
        "player_name"
    ] = (
        slate[
            "first_name"
        ].fillna(
            ""
        )
        + " "
        + slate[
            "last_name"
        ].fillna(
            ""
        )
    ).str.strip()

    slate[
        "projection_generated_at_utc"
    ] = generated_at

    slate[
        "freeze_id"
    ] = manifest[
        "freeze_id"
    ]

    slate[
        "freeze_stage"
    ] = manifest[
        "freeze_stage"
    ]

    slate[
        "manifest_sha256"
    ] = manifest[
        "manifest_sha256"
    ]

    slate[
        "projection_schema_version"
    ] = 2

    history_dates = pd.to_datetime(
        history.get(
            "date",
            pd.Series(
                dtype="datetime64[ns]"
            ),
        ),
        errors="coerce",
    )

    advanced_dates = pd.to_datetime(
        advanced.get(
            "date",
            pd.Series(
                dtype="datetime64[ns]"
            ),
        ),
        errors="coerce",
    )

    slate[
        "history_latest_date"
    ] = (
        history_dates.max()
        if history_dates.notna().any()
        else pd.NaT
    )

    slate[
        "advanced_latest_date"
    ] = (
        advanced_dates.max()
        if advanced_dates.notna().any()
        else pd.NaT
    )

    out_dir = (
        settings.processed_dir
        / "projections"
    )

    out_path = (
        out_dir
        / f"{args.date}.parquet"
    )

    write_parquet_atomic(
        slate,
        out_path,
    )

    console.rule(
        "PRODUCTION SLATE PROJECTION"
    )

    console.print(
        f"Freeze: "
        f"{manifest['freeze_id']} "
        f"({manifest['freeze_stage']})"
    )

    console.print(
        "Mean-model policy: "
        "PTS=XGB, REB=ENSEMBLE, "
        "AST=XGB, STL=XGB, "
        "BLK=ENSEMBLE, FG3M=XGB"
    )

    console.print(
        f"Players projected: "
        f"{len(slate):,}"
    )

    console.print(
        f"Availability OUT: "
        f"{int(slate['availability_out'].sum()):,}"
    )

    console.print(
        f"History latest date: "
        f"{slate['history_latest_date'].iloc[0]}"
    )

    console.print(
        f"Advanced latest date: "
        f"{slate['advanced_latest_date'].iloc[0]}"
    )

    console.print(
        f"[green]Wrote player projections[/green] "
        f"-> {out_path}"
    )

    if args.allow_predeployment:
        console.print(
            "[yellow]PREDEPLOYMENT SMOKE MODE: "
            "these projections are not external-test "
            "production records.[/yellow]"
        )


if __name__ == "__main__":
    main()
