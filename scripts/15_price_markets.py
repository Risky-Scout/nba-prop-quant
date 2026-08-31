from __future__ import annotations

from nba_prop_quant.gate3_v2 import (
    GATE3_CHANGED_PROPS,
    load_gate3_runtime,
    prepare_gate3_candidate_probability_overrides,
)

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console

from nba_prop_quant.api import BDLClient
from nba_prop_quant.normalize import (
    normalize_props,
)
from nba_prop_quant.pricing import (
    COMBO_COMPONENTS,
    PROP_TO_TARGET,
    price_combo_lines,
    price_single_prop_frame,
)
from nba_prop_quant.production import (
    TARGETS,
    add_market_probability_layer,
    load_json,
    load_verified_manifest_metadata,
)
from nba_prop_quant.settings import (
    get_settings,
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
        "--combo-simulations",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--max-hold-pct",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--allow-predeployment",
        action="store_true",
        help=(
            "Integration smoke-test only. "
            "Allows the pre-live baseline manifest "
            "before the final deployment freeze."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=73,
    )

    return parser.parse_args()


def stable_event_seed(
    game_id: int,
    player_id: int,
    prop_type: str,
    base_seed: int,
) -> int:
    prop_code = sum(
        (
            index
            + 1
        )
        * ord(
            char
        )
        for index, char in enumerate(
            prop_type
        )
    )

    return int(
        (
            int(
                base_seed
            )
            + 1_000_003
            * int(
                game_id
            )
            + 9_176
            * int(
                player_id
            )
            + 37
            * prop_code
        )
        % (
            2**32
            - 1
        )
    )


def add_quote_filter_fields(
    markets: pd.DataFrame,
    max_hold_pct: float,
    supported_props: set[str],
) -> pd.DataFrame:
    out = markets.copy()

    out[
        "line_value"
    ] = pd.to_numeric(
        out[
            "line_value"
        ],
        errors="coerce",
    )

    out[
        "over_odds"
    ] = pd.to_numeric(
        out[
            "over_odds"
        ],
        errors="coerce",
    )

    out[
        "under_odds"
    ] = pd.to_numeric(
        out[
            "under_odds"
        ],
        errors="coerce",
    )

    out[
        "quote_filter_reason"
    ] = ""

    out.loc[
        ~out[
            "market_type"
        ].eq(
            "over_under"
        ),
        "quote_filter_reason",
    ] = "unsupported_market_type"

    out.loc[
        out[
            "market_type"
        ].eq(
            "over_under"
        )
        & ~out[
            "prop_type"
        ].isin(
            supported_props
        ),
        "quote_filter_reason",
    ] = "unsupported_or_unfrozen_prop"

    required_numeric = (
        out[
            "line_value"
        ].notna()
        & out[
            "over_odds"
        ].notna()
        & out[
            "under_odds"
        ].notna()
        & out[
            "over_odds"
        ].ne(
            0
        )
        & out[
            "under_odds"
        ].ne(
            0
        )
    )

    out.loc[
        out[
            "quote_filter_reason"
        ].eq(
            ""
        )
        & ~required_numeric,
        "quote_filter_reason",
    ] = "missing_or_invalid_line_odds"

    valid_numeric = out[
        "quote_filter_reason"
    ].eq(
        ""
    )

    # Hold is calculated again after pricing in the canonical
    # probability layer; this early calculation exists only to
    # enforce the frozen quote-quality filter.
    if valid_numeric.any():
        from nba_prop_quant.pricing import (
            american_implied_probability,
        )

        over_imp = np.array(
            [
                american_implied_probability(
                    value
                )
                for value in out.loc[
                    valid_numeric,
                    "over_odds",
                ].to_numpy()
            ]
        )

        under_imp = np.array(
            [
                american_implied_probability(
                    value
                )
                for value in out.loc[
                    valid_numeric,
                    "under_odds",
                ].to_numpy()
            ]
        )

        hold = (
            over_imp
            + under_imp
            - 1.0
        ) * 100.0

        out.loc[
            valid_numeric,
            "preprice_hold_pct",
        ] = hold

        bad_hold = (
            (
                hold
                < 0.0
            )
            | (
                hold
                > float(
                    max_hold_pct
                )
            )
        )

        bad_indices = out.loc[
            valid_numeric
        ].index[
            bad_hold
        ]

        out.loc[
            bad_indices,
            "quote_filter_reason",
        ] = "hold_outside_frozen_range"

    out[
        "quote_eligible"
    ] = out[
        "quote_filter_reason"
    ].eq(
        ""
    )

    return out


def main() -> None:
    args = parse_args()

    if args.combo_simulations < 200:
        raise SystemExit(
            "--combo-simulations must be at least 200"
        )

    settings = get_settings()

    project_root = Path.cwd()

    manifest = load_verified_manifest_metadata(
        model_dir=settings.nba_prop_model_dir,
        project_root=project_root,
        allow_predeployment=(
            args.allow_predeployment
        ),
    )

    projection_path = (
        settings.processed_dir
        / "projections"
        / f"{args.date}.parquet"
    )

    if not projection_path.exists():
        raise SystemExit(
            f"ERROR: missing projection file: "
            f"{projection_path}. "
            "Run scripts/10_predict_slate.py first."
        )

    projections = pd.read_parquet(
        projection_path
    )

    if projections.empty:
        raise SystemExit(
            "ERROR: projection file is empty"
        )

    projection_freeze_ids = set(
        projections[
            "freeze_id"
        ].dropna().astype(
            str
        ).unique()
    )

    if projection_freeze_ids != {
        manifest[
            "freeze_id"
        ]
    }:
        raise RuntimeError(
            "Projection freeze ID does not match "
            "the currently verified manifest. "
            f"projection={projection_freeze_ids}, "
            f"current={manifest['freeze_id']}"
        )

    marginals = joblib.load(
        settings.nba_prop_model_dir
        / "marginals.joblib"
    )

    copula = joblib.load(
        settings.nba_prop_model_dir
        / "copula.joblib"
    )

    dependence_policy = load_json(
        settings.nba_prop_model_dir
        / "combo_dependence_policy.json"
    )

    calibration_policy = load_json(
        settings.nba_prop_model_dir
        / "market_probability_calibration_policy.json"
    )

    gate3_runtime = load_gate3_runtime()

    required_candidate_id = gate3_runtime[
        "candidate_id"
    ]

    if (
        "gate3_candidate_policy_id"
        not in projections.columns
    ):
        raise RuntimeError(
            "Projection file missing Gate 3 candidate identity"
        )

    projection_candidate_ids = set(
        projections[
            "gate3_candidate_policy_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if projection_candidate_ids != {
        required_candidate_id
    }:
        raise RuntimeError(
            "Projection Gate 3 candidate ID mismatch. "
            f"projection={projection_candidate_ids}, "
            f"required={required_candidate_id}"
        )


    frozen_prop_types = set(
        calibration_policy[
            "props"
        ]
    )

    model_supported = (
        set(
            PROP_TO_TARGET
        )
        | set(
            COMBO_COMPONENTS
        )
    )

    supported_props = (
        model_supported
        & frozen_prop_types
    )

    mu_columns = {
        target: (
            f"mu_selected_{target}"
        )
        for target in TARGETS
    }

    for target, column in (
        mu_columns.items()
    ):
        if column not in projections.columns:
            raise RuntimeError(
                f"Projection file missing "
                f"canonical mean column {column}"
            )

    market_frames = []

    with BDLClient(
        api_key=settings.bdl_api_key,
        base_url=settings.bdl_base_url,
        requests_per_minute=(
            settings.bdl_requests_per_minute
        ),
    ) as client:
        for game_id in sorted(
            projections[
                "game_id"
            ].dropna().astype(
                int
            ).unique()
        ):
            rows = client.live_player_props(
                game_id
            )

            if rows:
                market_frames.append(
                    normalize_props(
                        rows
                    )
                )

    if not market_frames:
        console.print(
            "[yellow]No current player-prop "
            "markets returned.[/yellow]"
        )
        return

    raw_markets = pd.concat(
        market_frames,
        ignore_index=True,
    )

    audited_markets = (
        add_quote_filter_fields(
            raw_markets,
            max_hold_pct=(
                args.max_hold_pct
            ),
            supported_props=(
                supported_props
            ),
        )
    )

    rejected = audited_markets[
        ~audited_markets[
            "quote_eligible"
        ]
    ].copy()

    markets = audited_markets[
        audited_markets[
            "quote_eligible"
        ]
    ].copy()

    if markets.empty:
        console.print(
            "[yellow]No frozen-policy eligible "
            "two-way markets remain after "
            "quote-quality filters.[/yellow]"
        )
        return

    merged = markets.merge(
        projections,
        on=[
            "game_id",
            "player_id",
        ],
        how="inner",
        suffixes=(
            "_market",
            "",
        ),
        validate="many_to_one",
    )

    if "availability_out" in merged.columns:
        out_mask = pd.to_numeric(
            merged[
                "availability_out"
            ],
            errors="coerce",
        ).fillna(
            0
        ).astype(
            int
        ).eq(
            1
        )

        if out_mask.any():
            out_quotes = merged.loc[
                out_mask
            ].copy()

            out_quotes[
                "quote_filter_reason"
            ] = "player_currently_out"

            rejected = pd.concat(
                [
                    rejected,
                    out_quotes,
                ],
                ignore_index=True,
            )

            merged = merged.loc[
                ~out_mask
            ].copy()

    if (
        "gate3_role_ready"
        not in merged.columns
    ):
        raise RuntimeError(
            "Projection file missing gate3_role_ready"
        )

    gate3_role_ready = pd.to_numeric(
        merged[
            "gate3_role_ready"
        ],
        errors="coerce",
    ).fillna(
        0
    ).astype(
        int
    ).eq(1)

    changed_prop_mask = (
        merged[
            "prop_type"
        ].isin(
            GATE3_CHANGED_PROPS
        )
    )

    gate3_unavailable = (
        changed_prop_mask
        & ~gate3_role_ready
    )

    if gate3_unavailable.any():
        gate3_rejected = merged.loc[
            gate3_unavailable
        ].copy()

        gate3_rejected[
            "quote_filter_reason"
        ] = (
            "gate3_role_state_unavailable"
        )

        rejected = pd.concat(
            [
                rejected,
                gate3_rejected,
            ],
            ignore_index=True,
        )

        merged = merged.loc[
            ~gate3_unavailable
        ].copy()

    if merged.empty:
        console.print(
            "[yellow]No eligible current markets "
            "matched healthy player projections.[/yellow]"
        )
        return

    priced_parts = []

    for prop_type, target in (
        PROP_TO_TARGET.items()
    ):
        part = merged[
            merged[
                "prop_type"
            ].eq(
                prop_type
            )
        ].copy()

        if part.empty:
            continue

        probabilities = (
            price_single_prop_frame(
                part,
                target=target,
                marginal=marginals[
                    target
                ],
                mu_column=(
                    "gate3_mu_ast"
                    if prop_type
                    == "assists"
                    else mu_columns[
                        target
                    ]
                ),
                line_column="line_value",
            )
        )

        for column in probabilities.columns:
            part[
                column
            ] = probabilities[
                column
            ]

        part[
            "dependence_lambda"
        ] = 0.0

        priced_parts.append(
            part
        )

    combo_rows = merged[
        merged[
            "prop_type"
        ].isin(
            COMBO_COMPONENTS
        )
    ].copy()

    for prop_type in sorted(
        COMBO_COMPONENTS
    ):
        part = combo_rows[
            combo_rows[
                "prop_type"
            ].eq(
                prop_type
            )
        ].copy()

        if part.empty:
            continue

        if prop_type not in dependence_policy[
            "combos"
        ]:
            raise RuntimeError(
                "Missing frozen dependence policy "
                f"for {prop_type}"
            )

        lambda_value = float(
            dependence_policy[
                "combos"
            ][
                prop_type
            ][
                "production_lambda"
            ]
        )

        event_probabilities = []

        grouped = part.groupby(
            [
                "game_id",
                "player_id",
                "prop_type",
            ],
            sort=False,
        )

        for (
            game_id,
            player_id,
            _
        ), group in grouped:
            row = group.iloc[
                0
            ]

            lines = np.sort(
                group[
                    "line_value"
                ].unique()
            )

            probabilities = (
                price_combo_lines(
                    row=row,
                    prop_type=prop_type,
                    lines=lines,
                    marginals=marginals,
                    mu_columns=mu_columns,
                    copula=copula,
                    dependence_lambda=(
                        lambda_value
                    ),
                    simulations=(
                        args.combo_simulations
                    ),
                    seed=stable_event_seed(
                        int(
                            game_id
                        ),
                        int(
                            player_id
                        ),
                        prop_type,
                        args.seed,
                    ),
                )
            )

            probabilities[
                "game_id"
            ] = int(
                game_id
            )

            probabilities[
                "player_id"
            ] = int(
                player_id
            )

            probabilities[
                "prop_type"
            ] = prop_type

            event_probabilities.append(
                probabilities
            )

        lookup = pd.concat(
            event_probabilities,
            ignore_index=True,
        )

        priced_part = part.merge(
            lookup,
            on=[
                "game_id",
                "player_id",
                "prop_type",
                "line_value",
            ],
            how="left",
            validate="many_to_one",
        )

        priced_parts.append(
            priced_part
        )

    if not priced_parts:
        console.print(
            "[yellow]No supported market rows "
            "could be priced.[/yellow]"
        )
        return

    priced = pd.concat(
        priced_parts,
        ignore_index=True,
    )

    priced = (
        prepare_gate3_candidate_probability_overrides(
            priced,
            calibration_policy=(
                calibration_policy
            ),
            probability_parameters=(
                gate3_runtime[
                    "probability_parameters"
                ]
            ),
        )
    )

    probability_mass = (
        priced[
            "p_over"
        ]
        + priced[
            "p_under"
        ]
        + priced[
            "p_push"
        ]
    )

    max_mass_error = float(
        np.max(
            np.abs(
                probability_mass
                - 1.0
            )
        )
    )

    if max_mass_error > 5e-3:
        raise RuntimeError(
            "Pricing probability mass error "
            f"too large: {max_mass_error}"
        )

    priced = (
        add_market_probability_layer(
            priced,
            calibration_policy=(
                calibration_policy
            ),
        )
    )

    priced_at = datetime.now(
        timezone.utc
    ).isoformat()

    priced[
        "priced_at_utc"
    ] = priced_at

    priced[
        "freeze_id"
    ] = manifest[
        "freeze_id"
    ]

    priced[
        "freeze_stage"
    ] = manifest[
        "freeze_stage"
    ]

    priced[
        "manifest_sha256"
    ] = manifest[
        "manifest_sha256"
    ]

    priced[
        "market_pricing_schema_version"
    ] = 3

    priced[
        "gate3_candidate_policy_id"
    ] = gate3_runtime[
        "candidate_id"
    ]

    priced[
        "gate3_policy_lock_commit"
    ] = gate3_runtime[
        "gate3_lock_commit"
    ]

    priced[
        "gate3_deployment_manifest_sha256"
    ] = gate3_runtime[
        "deployment_manifest_sha256"
    ]

    priced[
        "gate3_external_test_record"
    ] = priced[
        "external_test_record"
    ]

    priced[
        "external_test_record"
    ] = (
        manifest[
            "freeze_stage"
        ]
        == "external_test_deployment"
    )

    out_dir = (
        settings.processed_dir
        / "priced_markets"
    )

    out_path = (
        out_dir
        / f"{args.date}.parquet"
    )

    write_parquet_atomic(
        priced,
        out_path,
    )

    rejected_path = (
        out_dir
        / "rejected"
        / f"{args.date}.parquet"
    )

    if not rejected.empty:
        write_parquet_atomic(
            rejected,
            rejected_path,
        )

    console.rule(
        "FROZEN LIVE MARKET PRICING"
    )

    console.print(
        f"Freeze: "
        f"{manifest['freeze_id']} "
        f"({manifest['freeze_stage']})"
    )

    console.print(
        f"Raw market quotes: "
        f"{len(raw_markets):,}"
    )

    console.print(
        f"Eligible priced quotes: "
        f"{len(priced):,}"
    )

    console.print(
        f"Rejected/unfrozen quotes: "
        f"{len(rejected):,}"
    )

    console.print(
        f"Maximum pricing mass error: "
        f"{max_mass_error:.3e}"
    )

    console.print(
        "No betting threshold is active. "
        "Edge thresholds are monitoring flags only."
    )

    display_cols = [
        "player_name",
        "vendor",
        "prop_type",
        "line_value",
        "calibration_method",
        "raw_q_over_nonpush",
        "calibrated_q_over_nonpush",
        "market_devig_q_over",
        "model_preferred_side",
        "model_preferred_edge",
        "model_preferred_ev",
        "dependence_lambda",
    ]

    display_cols = [
        column
        for column in display_cols
        if column in priced.columns
    ]

    console.print(
        priced[
            display_cols
        ]
        .sort_values(
            "model_preferred_ev",
            ascending=False,
        )
        .head(
            30
        )
        .to_string(
            index=False
        )
    )

    console.print(
        f"[green]Saved priced markets[/green] "
        f"-> {out_path}"
    )

    if not rejected.empty:
        console.print(
            f"Saved rejected quote audit "
            f"-> {rejected_path}"
        )

    if args.allow_predeployment:
        console.print(
            "[yellow]PREDEPLOYMENT SMOKE MODE: "
            "these prices are not external-test "
            "production records.[/yellow]"
        )


if __name__ == "__main__":
    main()
