from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


V2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V2_ROOT.parents[2]

GATE2_SCRIPT = (
    V2_ROOT
    / "scripts/13_gate2_certify_v2.py"
)

GATE2_OUTPUT = (
    V2_ROOT
    / "research/v2_gate2_certification_outputs"
)

GATE3_LOCK = (
    V2_ROOT
    / "research/v2_gate3_lock"
)

LINEUPS_PATH = (
    V2_ROOT
    / "research/v2_lineup_backfill_2025/"
      "lineups_normalized.parquet"
)

OUTPUT_DIR = (
    V2_ROOT
    / "research/v2_gate3_deployment_artifacts"
)

MEANS_PATH = (
    PROJECT_ROOT
    / "data/processed/"
      "selected_means_distribution_split.parquet"
)

QUOTES_PATH = (
    PROJECT_ROOT
    / "data/processed/market_backtest/calibrated_oof/"
      "selected_oof_quote_rows.parquet"
)

MARGINALS_PATH = (
    PROJECT_ROOT
    / "models/marginals_pre2025.joblib"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def verify_checksums(
    directory: Path,
) -> None:
    checksum_path = (
        directory
        / "SHA256SUMS.txt"
    )

    if not checksum_path.exists():
        raise RuntimeError(
            f"Missing checksums: {checksum_path}"
        )

    for raw in checksum_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not raw.strip():
            continue

        expected, filename = raw.split(
            None,
            1,
        )

        path = (
            directory
            / filename.strip()
        )

        actual = sha256(path)

        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch: {path}"
            )


def load_gate2_module():
    spec = (
        importlib.util
        .spec_from_file_location(
            "gate2_locked_runner",
            GATE2_SCRIPT,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to load Gate 2 runner"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def git_output(*args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(V2_ROOT),
            *args,
        ],
        text=True,
    ).strip()


def build_role_state_seed(
    means: pd.DataFrame,
    lineups: pd.DataFrame,
) -> dict:
    holdout = means.loc[
        means[
            "distribution_split"
        ]
        .astype(str)
        .eq("holdout_2025")
    ].copy()

    date_col = (
        "date"
        if "date" in holdout.columns
        else "game_date"
    )

    game_dates = (
        holdout[
            [
                "game_id",
                date_col,
            ]
        ]
        .drop_duplicates(
            "game_id"
        )
        .rename(
            columns={
                date_col: "_date",
            }
        )
    )

    frame = (
        lineups[
            [
                "game_id",
                "player_id",
                "team_id",
                "starter",
            ]
        ]
        .drop_duplicates(
            [
                "game_id",
                "player_id",
            ]
        )
        .merge(
            game_dates,
            on="game_id",
            how="inner",
            validate="many_to_one",
        )
    )

    frame["_date"] = pd.to_datetime(
        frame["_date"]
    )

    frame = frame.loc[
        frame["player_id"].notna()
        & frame["team_id"].notna()
        & frame["starter"].notna()
    ].copy()

    frame["player_id"] = (
        frame["player_id"]
        .astype(int)
    )

    frame["team_id"] = (
        frame["team_id"]
        .astype(int)
    )

    frame["starter"] = (
        frame["starter"]
        .astype(bool)
        .astype(int)
    )

    frame = frame.sort_values(
        [
            "_date",
            "game_id",
            "team_id",
            "player_id",
        ]
    )

    players = {}

    for player_id, g in frame.groupby(
        "player_id",
        sort=True,
    ):
        g = g.sort_values(
            [
                "_date",
                "game_id",
            ]
        )

        history = (
            g["starter"]
            .astype(int)
            .tolist()
        )

        last = g.iloc[-1]

        players[
            str(int(player_id))
        ] = {
            "starter_history": (
                history[-10:]
            ),
            "last_starter": int(
                history[-1]
            ),
            "last_team_id": int(
                last["team_id"]
            ),
            "last_game_id": int(
                last["game_id"]
            ),
            "last_date": (
                pd.Timestamp(
                    last["_date"]
                )
                .date()
                .isoformat()
            ),
        }

    teams = {}

    for team_id, g in frame.groupby(
        "team_id",
        sort=True,
    ):
        games = (
            g[
                [
                    "_date",
                    "game_id",
                ]
            ]
            .drop_duplicates(
                "game_id"
            )
            .sort_values(
                [
                    "_date",
                    "game_id",
                ]
            )
        )

        last_game = games.iloc[-1]

        last_game_id = int(
            last_game["game_id"]
        )

        last_rows = g.loc[
            g["game_id"].eq(
                last_game_id
            )
        ]

        starters = sorted(
            last_rows.loc[
                last_rows[
                    "starter"
                ].eq(1),
                "player_id",
            ]
            .astype(int)
            .tolist()
        )

        teams[
            str(int(team_id))
        ] = {
            "last_starters": starters,
            "last_game_id": (
                last_game_id
            ),
            "last_date": (
                pd.Timestamp(
                    last_game["_date"]
                )
                .date()
                .isoformat()
            ),
        }

    return {
        "season": 2025,
        "development_only_source": True,
        "player_count": len(
            players
        ),
        "team_count": len(
            teams
        ),
        "players": players,
        "teams": teams,
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Gate 3 deployment artifact "
            "directory already exists"
        )

    verify_checksums(
        GATE2_OUTPUT
    )

    verify_checksums(
        GATE3_LOCK
    )

    gate3_policy = json.loads(
        (
            GATE3_LOCK
            / "GATE3_CANDIDATE_POLICY.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    expected_policy = {
        "points": "frozen_selected_v1",
        "rebounds": "frozen_selected_v1",
        "assists": "v2_role_shock_calibrated",
        "steals": "frozen_selected_v1",
        "blocks": "frozen_selected_v1",
        "threes": "frozen_selected_v1",
        "points_assists": "v2_role_increment",
        "points_rebounds": "v2_role_increment",
        "rebounds_assists": "frozen_selected_v1",
        "points_rebounds_assists": "frozen_selected_v1",
    }

    if (
        gate3_policy[
            "gate3_policy"
        ]
        != expected_policy
    ):
        raise RuntimeError(
            "Gate 3 policy mismatch"
        )

    gate2 = load_gate2_module()

    means = pd.read_parquet(
        MEANS_PATH
    )

    quotes = pd.read_parquet(
        QUOTES_PATH
    )

    lineups = pd.read_parquet(
        LINEUPS_PATH
    )

    marginals = joblib.load(
        MARGINALS_PATH
    )

    role_frame, _, _ = (
        gate2.build_role_minutes(
            means,
            lineups,
        )
    )

    gate1_policy = json.loads(
        (
            V2_ROOT
            / "research/v2_gate1_lock/"
              "CANDIDATE_POLICY.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    base_features = [
        "expected_minutes",
        "prior_minutes10",
        "decay_prior_min",
        "kalman_prior_min",
        "team_change",
        "player_game_number",
        "career_games_prior",
        "team_game_number",
        "season_progress",
        "days_since_prev",
        "days_rest",
        "b2b",
        "adv_prior_usage_percentage",
        "adv_prior_estimated_usage_percentage",
        "adv_prior_assist_percentage",
        "adv_prior_touches",
        "adv_prior_passes",
        "team_pace_prior",
        "opp_pace_prior",
        "adv_prior_distance",
        "is_home",
        "pos_G",
        "pos_F",
        "pos_C",
        "experience_years",
    ]

    base_features = [
        feature
        for feature in base_features
        if feature in role_frame.columns
    ]

    role_features = (
        base_features
        + gate1_policy[
            "role_state_features"
        ]
    )

    missing_features = [
        feature
        for feature in role_features
        if feature not in role_frame.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Missing locked role features: "
            f"{missing_features}"
        )

    fit_rows = role_frame.loc[
        role_frame["minutes"].notna()
        & role_frame[
            "expected_minutes"
        ].notna()
    ].copy()

    X = fit_rows[
        role_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y = (
        fit_rows["minutes"]
        - fit_rows[
            "expected_minutes"
        ]
    )

    role_model = (
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=3.0,
            random_state=20260830,
        )
    )

    role_model.fit(
        X,
        y,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    role_model_path = (
        OUTPUT_DIR
        / "role_minutes_model.joblib"
    )

    joblib.dump(
        {
            "model": role_model,
            "feature_names": role_features,
            "target": (
                "minutes - expected_minutes"
            ),
            "training_rows": int(
                len(fit_rows)
            ),
            "training_games": int(
                fit_rows[
                    "game_id"
                ].nunique()
            ),
            "estimator_parameters": {
                "learning_rate": 0.05,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "l2_regularization": 3.0,
                "random_state": 20260830,
            },
        },
        role_model_path,
    )

    loaded_role = joblib.load(
        role_model_path
    )

    before = role_model.predict(
        X
    )

    after = loaded_role[
        "model"
    ].predict(
        X
    )

    if not np.allclose(
        before,
        after,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Role model round-trip mismatch"
        )

    candidate_meta = role_frame.loc[
        role_frame[
            "minutes_role_oof"
        ].notna(),
        [
            "game_id",
            "player_id",
            "_date",
            "expected_minutes",
            "minutes_role_oof",
            "mu_selected_pts",
            "mu_selected_reb",
            "mu_selected_ast",
        ],
    ].copy()

    ast = gate2.collapsed_quotes(
        quotes,
        "assists",
    )

    ast = ast.merge(
        candidate_meta,
        on=[
            "game_id",
            "player_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    ast = gate2.finalize_contract_frame(
        ast
    )

    ast = ast.loc[
        ast[
            "mu_selected_ast"
        ].notna()
        & ast[
            "expected_minutes"
        ].gt(0)
    ].copy()

    ast_mu = (
        ast[
            "mu_selected_ast"
        ].to_numpy(float)
        * ast[
            "minutes_role_oof"
        ].to_numpy(float)
        / ast[
            "expected_minutes"
        ].to_numpy(float)
    )

    ast[
        "q_role_raw"
    ] = gate2.probability_from_mu(
        ast,
        marginals[
            "ast"
        ],
        np.maximum(
            ast_mu,
            1e-8,
        ),
    )

    ast_beta = gate2.fit_platt(
        ast["y"],
        ast["q_role_raw"],
    )

    ast_final = gate2.apply_platt(
        ast[
            "q_role_raw"
        ],
        ast_beta,
    )

    ratio = (
        candidate_meta[
            "minutes_role_oof"
        ]
        / candidate_meta[
            "expected_minutes"
        ]
    )

    candidate_meta[
        "delta_pts"
    ] = (
        candidate_meta[
            "mu_selected_pts"
        ]
        * (
            ratio - 1.0
        )
    )

    candidate_meta[
        "delta_reb"
    ] = (
        candidate_meta[
            "mu_selected_reb"
        ]
        * (
            ratio - 1.0
        )
    )

    candidate_meta[
        "delta_ast"
    ] = (
        candidate_meta[
            "mu_selected_ast"
        ]
        * (
            ratio - 1.0
        )
    )

    candidate_meta[
        "delta_P+A"
    ] = (
        candidate_meta[
            "delta_pts"
        ]
        + candidate_meta[
            "delta_ast"
        ]
    )

    candidate_meta[
        "delta_P+R"
    ] = (
        candidate_meta[
            "delta_pts"
        ]
        + candidate_meta[
            "delta_reb"
        ]
    )

    probability_parameters = {
        "policy_version": 1,
        "development_fit_only": True,
        "assists": {
            "method": (
                "platt_on_role_adjusted_raw"
            ),
            "intercept": float(
                ast_beta[0]
            ),
            "slope": float(
                ast_beta[1]
            ),
            "fit_rows": int(
                len(ast)
            ),
            "fit_games": int(
                ast[
                    "game_id"
                ].nunique()
            ),
            "development_brier": float(
                gate2.brier(
                    ast["y"],
                    ast_final,
                )
            ),
            "development_logloss": float(
                gate2.logloss(
                    ast["y"],
                    ast_final,
                )
            ),
        },
    }

    for prop, delta_col in [
        (
            "points_assists",
            "delta_P+A",
        ),
        (
            "points_rebounds",
            "delta_P+R",
        ),
    ]:
        q = gate2.collapsed_quotes(
            quotes,
            prop,
        )

        q = q.merge(
            candidate_meta[
                [
                    "game_id",
                    "player_id",
                    "_date",
                    delta_col,
                ]
            ],
            on=[
                "game_id",
                "player_id",
            ],
            how="inner",
            validate="many_to_one",
        )

        q = gate2.finalize_contract_frame(
            q
        )

        q = q.loc[
            q[
                delta_col
            ].notna()
        ].copy()

        gamma, mean, std = (
            gate2.fit_gamma(
                q["y"],
                q["q_frozen"],
                q[delta_col],
            )
        )

        fitted_q = gate2.apply_gamma(
            q["q_frozen"],
            q[delta_col],
            gamma,
            mean,
            std,
        )

        probability_parameters[
            prop
        ] = {
            "method": (
                "logit_role_increment"
            ),
            "gamma": float(
                gamma
            ),
            "standardization_mean": float(
                mean
            ),
            "standardization_std": float(
                std
            ),
            "fit_rows": int(
                len(q)
            ),
            "fit_games": int(
                q[
                    "game_id"
                ].nunique()
            ),
            "development_brier": float(
                gate2.brier(
                    q["y"],
                    fitted_q,
                )
            ),
            "development_logloss": float(
                gate2.logloss(
                    q["y"],
                    fitted_q,
                )
            ),
        }

    parameter_path = (
        OUTPUT_DIR
        / "probability_parameters.json"
    )

    parameter_path.write_text(
        json.dumps(
            probability_parameters,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    seed = build_role_state_seed(
        means,
        lineups,
    )

    seed_path = (
        OUTPUT_DIR
        / "role_state_seed.json"
    )

    seed_path.write_text(
        json.dumps(
            seed,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    current_head = git_output(
        "rev-parse",
        "HEAD",
    )

    gate2_evidence_commit = (
        gate3_policy[
            "source_gate2_evidence_commit"
        ]
    )

    gate3_lock_commit = git_output(
        "log",
        "-1",
        "--format=%H",
        "--",
        "research/v2_gate3_lock/"
        "GATE3_CANDIDATE_POLICY.json",
    )

    manifest = {
        "artifact": (
            "nba_prop_quant_v2_gate3_deployment"
        ),
        "artifact_version": 1,
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "builder_commit": current_head,
        "gate2_evidence_commit": (
            gate2_evidence_commit
        ),
        "gate3_lock_commit": (
            gate3_lock_commit
        ),
        "gate3_policy": expected_policy,
        "role_minutes_training_rows": int(
            len(fit_rows)
        ),
        "role_minutes_training_games": int(
            fit_rows[
                "game_id"
            ].nunique()
        ),
        "role_minutes_features": (
            role_features
        ),
        "role_state_seed_players": (
            seed[
                "player_count"
            ]
        ),
        "role_state_seed_teams": (
            seed[
                "team_count"
            ]
        ),
        "deployment_parameter_fit": (
            "Locked methods refit once on all eligible "
            "2025 OOF development predictions. "
            "No candidate selection or retuning."
        ),
        "prospective_claim_allowed": False,
        "input_sha256": {
            str(
                MEANS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ): sha256(
                MEANS_PATH
            ),
            str(
                QUOTES_PATH.relative_to(
                    PROJECT_ROOT
                )
            ): sha256(
                QUOTES_PATH
            ),
            str(
                MARGINALS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ): sha256(
                MARGINALS_PATH
            ),
            str(
                LINEUPS_PATH.relative_to(
                    V2_ROOT
                )
            ): sha256(
                LINEUPS_PATH
            ),
            "research/v2_gate3_lock/"
            "GATE3_CANDIDATE_POLICY.json": sha256(
                GATE3_LOCK
                / "GATE3_CANDIDATE_POLICY.json"
            ),
        },
    }

    manifest_path = (
        OUTPUT_DIR
        / "deployment_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    targets = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
        and path.name
        != "SHA256SUMS.txt"
    )

    checksum_path = (
        OUTPUT_DIR
        / "SHA256SUMS.txt"
    )

    checksum_path.write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in targets
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 110)
    print("NBA V2 GATE 3 DEPLOYMENT ARTIFACTS")
    print("=" * 110)

    print(
        "role model rows:",
        len(fit_rows),
    )

    print(
        "role model games:",
        fit_rows[
            "game_id"
        ].nunique(),
    )

    print(
        "role seed players:",
        seed[
            "player_count"
        ],
    )

    print(
        "role seed teams:",
        seed[
            "team_count"
        ],
    )

    print()
    print(
        "FINAL DEPLOYMENT PARAMETERS"
    )

    print(
        json.dumps(
            probability_parameters,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "PASS: Gate 3 deployment artifacts built "
        "from locked Gate 2 formulas."
    )


if __name__ == "__main__":
    main()
