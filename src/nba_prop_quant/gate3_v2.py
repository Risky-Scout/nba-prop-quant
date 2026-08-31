from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd


GATE3_CHANGED_PROPS = frozenset(
    {
        "assists",
        "points_assists",
        "points_rebounds",
    }
)

EPS = 1e-6


def _sha256(path: Path) -> str:
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


def _default_artifact_dir() -> Path:
    override = os.getenv(
        "NBA_PROP_GATE3_ARTIFACT_DIR"
    )

    if override:
        return Path(
            override
        ).expanduser().resolve()

    project = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    return (
        project
        / "research/v2_gate3_deployment_artifacts"
    )


def resolve_gate3_snapshot_dir(
    default_snapshot_dir: Path,
) -> Path:
    override = os.getenv(
        "NBA_PROP_GATE3_SNAPSHOT_DIR"
    )

    if override:
        return Path(
            override
        ).expanduser().resolve()

    return Path(
        default_snapshot_dir
    ).expanduser().resolve()


def _verify_artifacts(
    artifact_dir: Path,
) -> None:
    checksum_path = (
        artifact_dir
        / "SHA256SUMS.txt"
    )

    if not checksum_path.exists():
        raise RuntimeError(
            "Gate 3 deployment checksums missing: "
            f"{checksum_path}"
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
            artifact_dir
            / filename.strip()
        )

        if not path.exists():
            raise RuntimeError(
                "Gate 3 deployment artifact missing: "
                f"{path}"
            )

        actual = _sha256(
            path
        )

        if actual != expected:
            raise RuntimeError(
                "Gate 3 deployment artifact hash mismatch: "
                f"{path.name}"
            )


def load_gate3_runtime(
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    directory = (
        Path(artifact_dir)
        if artifact_dir is not None
        else _default_artifact_dir()
    ).resolve()

    _verify_artifacts(
        directory
    )

    manifest_path = (
        directory
        / "deployment_manifest.json"
    )

    parameter_path = (
        directory
        / "probability_parameters.json"
    )

    seed_path = (
        directory
        / "role_state_seed.json"
    )

    model_path = (
        directory
        / "role_minutes_model.joblib"
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    params = json.loads(
        parameter_path.read_text(
            encoding="utf-8"
        )
    )

    seed = json.loads(
        seed_path.read_text(
            encoding="utf-8"
        )
    )

    model_payload = joblib.load(
        model_path
    )

    required_policy = {
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
        manifest.get(
            "gate3_policy"
        )
        != required_policy
    ):
        raise RuntimeError(
            "Gate 3 deployment policy mismatch"
        )

    lock_commit = str(
        manifest[
            "gate3_lock_commit"
        ]
    )

    candidate_id = (
        "nba_prop_quant_v2_gate3_"
        + lock_commit[:12]
    )

    return {
        "artifact_dir": directory,
        "manifest": manifest,
        "probability_parameters": params,
        "role_state_seed": seed,
        "role_model_payload": model_payload,
        "candidate_id": candidate_id,
        "gate3_lock_commit": lock_commit,
        "deployment_manifest_sha256": _sha256(
            manifest_path
        ),
        "probability_parameters_sha256": _sha256(
            parameter_path
        ),
        "role_minutes_model_sha256": _sha256(
            model_path
        ),
        "role_state_seed_sha256": _sha256(
            seed_path
        ),
    }


def _parse_snapshot_payload(
    payload: dict[str, Any],
    snapshot_date: str,
    captured_at: str,
) -> dict[str, Any] | None:
    player = (
        payload.get(
            "player"
        )
        or {}
    )

    team = (
        payload.get(
            "team"
        )
        or {}
    )

    player_id = player.get(
        "id"
    )

    team_id = team.get(
        "id",
        player.get(
            "team_id"
        ),
    )

    game_id = payload.get(
        "game_id"
    )

    starter = payload.get(
        "starter"
    )

    if (
        player_id is None
        or team_id is None
        or game_id is None
        or starter is None
    ):
        return None

    return {
        "game_id": int(
            game_id
        ),
        "player_id": int(
            player_id
        ),
        "team_id": int(
            team_id
        ),
        "starter": int(
            bool(starter)
        ),
        "_date": snapshot_date,
        "_captured_at": str(
            captured_at
        ),
    }


def _load_prior_lineup_snapshots(
    snapshot_dir: Path,
    target_date: str,
) -> pd.DataFrame:
    lineup_dir = (
        Path(snapshot_dir)
        / "lineups"
    )

    columns = [
        "game_id",
        "player_id",
        "team_id",
        "starter",
        "_date",
        "_captured_at",
    ]

    if not lineup_dir.exists():
        return pd.DataFrame(
            columns=columns
        )

    target = date.fromisoformat(
        target_date
    )

    rows: list[
        dict[str, Any]
    ] = []

    for path in sorted(
        lineup_dir.glob(
            "*.jsonl"
        )
    ):
        try:
            snapshot_date = (
                date.fromisoformat(
                    path.stem
                )
            )
        except ValueError:
            continue

        if snapshot_date >= target:
            continue

        for raw in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            if not raw.strip():
                continue

            try:
                envelope = json.loads(
                    raw
                )
            except json.JSONDecodeError:
                continue

            payload = (
                envelope.get(
                    "payload"
                )
                or {}
            )

            row = _parse_snapshot_payload(
                payload,
                snapshot_date.isoformat(),
                str(
                    envelope.get(
                        "captured_at",
                        "",
                    )
                ),
            )

            if row is not None:
                rows.append(
                    row
                )

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    frame = pd.DataFrame(
        rows
    )

    frame = frame.sort_values(
        [
            "_date",
            "_captured_at",
            "game_id",
            "player_id",
        ]
    )

    frame = (
        frame.drop_duplicates(
            [
                "game_id",
                "player_id",
            ],
            keep="last",
        )
    )

    return frame


def _current_lineup_frame(
    current_lineups: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "game_id",
        "player_id",
        "team_id",
        "starter",
    ]

    if (
        current_lineups is None
        or current_lineups.empty
    ):
        return pd.DataFrame(
            columns=columns
        )

    missing = (
        set(
            columns
        )
        - set(
            current_lineups.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Current lineup frame missing columns: "
            f"{sorted(missing)}"
        )

    frame = (
        current_lineups[
            columns
        ]
        .dropna(
            subset=[
                "game_id",
                "player_id",
                "team_id",
            ]
        )
        .copy()
    )

    frame[
        "game_id"
    ] = frame[
        "game_id"
    ].astype(int)

    frame[
        "player_id"
    ] = frame[
        "player_id"
    ].astype(int)

    frame[
        "team_id"
    ] = frame[
        "team_id"
    ].astype(int)

    frame[
        "starter"
    ] = pd.to_numeric(
        frame[
            "starter"
        ],
        errors="coerce",
    )

    return (
        frame.drop_duplicates(
            [
                "game_id",
                "player_id",
            ],
            keep="last",
        )
    )


def _player_history_map(
    seed: dict[str, Any],
    prior: pd.DataFrame,
) -> dict[int, list[int]]:
    history: dict[
        int,
        list[int],
    ] = {}

    for key, value in (
        seed.get(
            "players",
            {}
        )
        .items()
    ):
        values = [
            int(x)
            for x in value.get(
                "starter_history",
                []
            )
        ]

        history[
            int(key)
        ] = values[
            -10:
        ]

    if not prior.empty:
        ordered = prior.sort_values(
            [
                "_date",
                "game_id",
            ]
        )

        for player_id, group in (
            ordered.groupby(
                "player_id",
                sort=False,
            )
        ):
            values = history.get(
                int(
                    player_id
                ),
                [],
            )

            values = (
                values
                + group[
                    "starter"
                ]
                .astype(int)
                .tolist()
            )

            history[
                int(
                    player_id
                )
            ] = values[
                -10:
            ]

    return history


def _team_prior_starters(
    seed: dict[str, Any],
    prior: pd.DataFrame,
) -> dict[
    int,
    set[int],
]:
    result: dict[
        int,
        set[int],
    ] = {}

    for key, value in (
        seed.get(
            "teams",
            {}
        )
        .items()
    ):
        result[
            int(key)
        ] = {
            int(x)
            for x in value.get(
                "last_starters",
                []
            )
        }

    if prior.empty:
        return result

    team_games = (
        prior[
            [
                "team_id",
                "_date",
                "game_id",
            ]
        ]
        .drop_duplicates(
            [
                "team_id",
                "game_id",
            ]
        )
        .sort_values(
            [
                "team_id",
                "_date",
                "game_id",
            ]
        )
    )

    for team_id, games in (
        team_games.groupby(
            "team_id",
            sort=False,
        )
    ):
        latest = games.iloc[
            -1
        ]

        game_id = int(
            latest[
                "game_id"
            ]
        )

        starters = set(
            prior.loc[
                prior[
                    "game_id"
                ].eq(
                    game_id
                )
                & prior[
                    "team_id"
                ].eq(
                    int(
                        team_id
                    )
                )
                & prior[
                    "starter"
                ].eq(1),
                "player_id",
            ]
            .astype(int)
            .tolist()
        )

        result[
            int(
                team_id
            )
        ] = starters

    return result


def build_current_role_features(
    current_lineups: pd.DataFrame,
    prior_snapshots: pd.DataFrame,
    seed: dict[str, Any],
) -> pd.DataFrame:
    current = _current_lineup_frame(
        current_lineups
    )

    output_columns = [
        "game_id",
        "player_id",
        "starter",
        "prev_starter",
        "starter_rate5",
        "starter_rate10",
        "starter_surprise",
        "promoted_to_starter",
        "demoted_to_bench",
        "team_starter_overlap",
        "team_new_starters",
        "team_lost_starters",
        "team_lineup_rows",
        "gate3_role_ready",
    ]

    if current.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    player_history = (
        _player_history_map(
            seed,
            prior_snapshots,
        )
    )

    team_prior = (
        _team_prior_starters(
            seed,
            prior_snapshots,
        )
    )

    team_summary = (
        current.groupby(
            [
                "game_id",
                "team_id",
            ],
            as_index=False,
        )
        .agg(
            team_lineup_rows=(
                "player_id",
                "size",
            ),
            team_starter_count=(
                "starter",
                lambda x:
                    int(
                        pd.to_numeric(
                            x,
                            errors="coerce",
                        )
                        .fillna(0)
                        .sum()
                    ),
            ),
        )
    )

    current = current.merge(
        team_summary,
        on=[
            "game_id",
            "team_id",
        ],
        how="left",
        validate="many_to_one",
    )

    rows = []

    for row in current.itertuples(
        index=False
    ):
        player_id = int(
            row.player_id
        )

        team_id = int(
            row.team_id
        )

        starter_value = (
            np.nan
            if pd.isna(
                row.starter
            )
            else int(
                row.starter
            )
        )

        history = (
            player_history.get(
                player_id,
                [],
            )
        )

        prev_starter = (
            float(
                history[
                    -1
                ]
            )
            if history
            else np.nan
        )

        starter_rate5 = (
            float(
                np.mean(
                    history[
                        -5:
                    ]
                )
            )
            if history
            else np.nan
        )

        starter_rate10 = (
            float(
                np.mean(
                    history[
                        -10:
                    ]
                )
            )
            if history
            else np.nan
        )

        current_team = current.loc[
            current[
                "game_id"
            ].eq(
                int(
                    row.game_id
                )
            )
            & current[
                "team_id"
            ].eq(
                team_id
            )
            & current[
                "starter"
            ].eq(1),
            "player_id",
        ]

        current_starters = {
            int(x)
            for x in current_team
            .dropna()
            .tolist()
        }

        prior_starters = (
            team_prior.get(
                team_id
            )
        )

        if prior_starters is None:
            overlap = np.nan
            new_starters = np.nan
            lost_starters = np.nan
        else:
            overlap = float(
                len(
                    current_starters
                    & prior_starters
                )
            )

            new_starters = float(
                len(
                    current_starters
                    - prior_starters
                )
            )

            lost_starters = float(
                len(
                    prior_starters
                    - current_starters
                )
            )

        team_complete = (
            int(
                row.team_starter_count
            )
            == 5
        )

        ready = bool(
            team_complete
            and not pd.isna(
                starter_value
            )
        )

        rows.append(
            {
                "game_id": int(
                    row.game_id
                ),
                "player_id": player_id,
                "starter": starter_value,
                "prev_starter": prev_starter,
                "starter_rate5": starter_rate5,
                "starter_rate10": starter_rate10,
                "starter_surprise": (
                    starter_value
                    - starter_rate10
                    if (
                        not pd.isna(
                            starter_value
                        )
                        and not pd.isna(
                            starter_rate10
                        )
                    )
                    else np.nan
                ),
                "promoted_to_starter": int(
                    ready
                    and starter_value == 1
                    and prev_starter == 0
                ),
                "demoted_to_bench": int(
                    ready
                    and starter_value == 0
                    and prev_starter == 1
                ),
                "team_starter_overlap": overlap,
                "team_new_starters": new_starters,
                "team_lost_starters": lost_starters,
                "team_lineup_rows": int(
                    row.team_lineup_rows
                ),
                "gate3_role_ready": int(
                    ready
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def apply_gate3_role_state(
    slate: pd.DataFrame,
    current_lineups: pd.DataFrame,
    *,
    target_date: str,
    snapshot_dir: Path,
    runtime: dict[str, Any],
) -> pd.DataFrame:
    out = slate.copy()

    prior = (
        _load_prior_lineup_snapshots(
            snapshot_dir,
            target_date,
        )
    )

    role = (
        build_current_role_features(
            current_lineups,
            prior,
            runtime[
                "role_state_seed"
            ],
        )
    )

    role_columns = [
        "starter",
        "prev_starter",
        "starter_rate5",
        "starter_rate10",
        "starter_surprise",
        "promoted_to_starter",
        "demoted_to_bench",
        "team_starter_overlap",
        "team_new_starters",
        "team_lost_starters",
        "team_lineup_rows",
        "gate3_role_ready",
    ]

    for column in role_columns:
        if column in out.columns:
            out = out.drop(
                columns=[
                    column
                ]
            )

    if role.empty:
        for column in role_columns:
            out[
                column
            ] = (
                0
                if column
                == "gate3_role_ready"
                else np.nan
            )
    else:
        out = out.merge(
            role,
            on=[
                "game_id",
                "player_id",
            ],
            how="left",
            validate="one_to_one",
        )

        out[
            "gate3_role_ready"
        ] = pd.to_numeric(
            out[
                "gate3_role_ready"
            ],
            errors="coerce",
        ).fillna(
            0
        ).astype(
            int
        )

    payload = runtime[
        "role_model_payload"
    ]

    feature_names = list(
        payload[
            "feature_names"
        ]
    )

    missing = [
        feature
        for feature in feature_names
        if feature not in out.columns
    ]

    if missing:
        raise RuntimeError(
            "Gate 3 role model missing live features: "
            f"{missing}"
        )

    expected_minutes = pd.to_numeric(
        out[
            "expected_minutes"
        ],
        errors="coerce",
    )

    availability_out = pd.to_numeric(
        out.get(
            "availability_out",
            pd.Series(
                0,
                index=out.index,
            ),
        ),
        errors="coerce",
    ).fillna(
        0
    ).astype(
        int
    )

    ready = (
        out[
            "gate3_role_ready"
        ].eq(1)
        & expected_minutes.notna()
        & expected_minutes.gt(0)
        & availability_out.eq(0)
    )

    out[
        "gate3_role_minutes"
    ] = np.nan

    if ready.any():
        X = out.loc[
            ready,
            feature_names,
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        residual = (
            payload[
                "model"
            ].predict(
                X
            )
        )

        out.loc[
            ready,
            "gate3_role_minutes",
        ] = np.maximum(
            expected_minutes.loc[
                ready
            ].to_numpy(
                dtype=float
            )
            + residual,
            0.0,
        )

    out[
        "gate3_role_ratio"
    ] = (
        pd.to_numeric(
            out[
                "gate3_role_minutes"
            ],
            errors="coerce",
        )
        / expected_minutes
    )

    out[
        "gate3_mu_ast"
    ] = (
        pd.to_numeric(
            out[
                "mu_selected_ast"
            ],
            errors="coerce",
        )
        * out[
            "gate3_role_ratio"
        ]
    )

    out[
        "gate3_delta_pts"
    ] = (
        pd.to_numeric(
            out[
                "mu_selected_pts"
            ],
            errors="coerce",
        )
        * (
            out[
                "gate3_role_ratio"
            ]
            - 1.0
        )
    )

    out[
        "gate3_delta_reb"
    ] = (
        pd.to_numeric(
            out[
                "mu_selected_reb"
            ],
            errors="coerce",
        )
        * (
            out[
                "gate3_role_ratio"
            ]
            - 1.0
        )
    )

    out[
        "gate3_delta_ast"
    ] = (
        pd.to_numeric(
            out[
                "mu_selected_ast"
            ],
            errors="coerce",
        )
        * (
            out[
                "gate3_role_ratio"
            ]
            - 1.0
        )
    )

    out[
        "gate3_delta_points_assists"
    ] = (
        out[
            "gate3_delta_pts"
        ]
        + out[
            "gate3_delta_ast"
        ]
    )

    out[
        "gate3_delta_points_rebounds"
    ] = (
        out[
            "gate3_delta_pts"
        ]
        + out[
            "gate3_delta_reb"
        ]
    )

    out[
        "gate3_candidate_policy_id"
    ] = runtime[
        "candidate_id"
    ]

    out[
        "gate3_policy_lock_commit"
    ] = runtime[
        "gate3_lock_commit"
    ]

    out[
        "gate3_deployment_manifest_sha256"
    ] = runtime[
        "deployment_manifest_sha256"
    ]

    return out


def _logit(
    p: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        clipped
        / (
            1.0
            - clipped
        )
    )


def _sigmoid(
    value: np.ndarray,
) -> np.ndarray:
    value = np.clip(
        np.asarray(
            value,
            dtype=float,
        ),
        -40.0,
        40.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -value
            )
        )
    )


def _frozen_calibrate(
    raw_q: np.ndarray,
    prop_type: str,
    calibration_policy: dict[str, Any],
) -> np.ndarray:
    entry = calibration_policy[
        "props"
    ][
        prop_type
    ]

    method = str(
        entry[
            "selected_method"
        ]
    )

    raw = np.clip(
        np.asarray(
            raw_q,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    if method == "raw":
        return raw

    if method not in {
        "prop",
        "global",
    }:
        raise RuntimeError(
            "Unsupported frozen calibration method "
            f"for {prop_type}: {method}"
        )

    parameters = entry.get(
        "production_parameters"
    )

    if not parameters:
        raise RuntimeError(
            "Missing frozen calibration parameters "
            f"for {prop_type}"
        )

    intercept = float(
        parameters[
            "intercept"
        ]
    )

    slope = float(
        parameters[
            "slope"
        ]
    )

    return _sigmoid(
        intercept
        + slope
        * _logit(
            raw
        )
    )


def prepare_gate3_candidate_probability_overrides(
    frame: pd.DataFrame,
    *,
    calibration_policy: dict[str, Any],
    probability_parameters: dict[str, Any],
) -> pd.DataFrame:
    out = frame.copy()

    out[
        "gate3_candidate_q_over_nonpush"
    ] = np.nan

    out[
        "gate3_candidate_method"
    ] = "frozen_selected_v1"

    out[
        "gate3_candidate_gamma"
    ] = np.nan

    out[
        "gate3_candidate_standardization_mean"
    ] = np.nan

    out[
        "gate3_candidate_standardization_std"
    ] = np.nan

    out[
        "gate3_candidate_intercept"
    ] = np.nan

    out[
        "gate3_candidate_slope"
    ] = np.nan

    changed = out[
        "prop_type"
    ].isin(
        GATE3_CHANGED_PROPS
    )

    if changed.any():
        if (
            "gate3_role_ready"
            not in out.columns
        ):
            raise RuntimeError(
                "Gate 3 changed prop rows missing "
                "gate3_role_ready"
            )

        ready = pd.to_numeric(
            out[
                "gate3_role_ready"
            ],
            errors="coerce",
        ).fillna(
            0
        ).astype(
            int
        ).eq(1)

        if (
            changed
            & ~ready
        ).any():
            raise RuntimeError(
                "Gate 3 changed prop reached pricing "
                "without ready role state"
            )

    assists_mask = out[
        "prop_type"
    ].eq(
        "assists"
    )

    if assists_mask.any():
        params = probability_parameters[
            "assists"
        ]

        raw = pd.to_numeric(
            out.loc[
                assists_mask,
                "q_over_nonpush",
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        candidate = _sigmoid(
            float(
                params[
                    "intercept"
                ]
            )
            + float(
                params[
                    "slope"
                ]
            )
            * _logit(
                raw
            )
        )

        out.loc[
            assists_mask,
            "gate3_candidate_q_over_nonpush",
        ] = candidate

        out.loc[
            assists_mask,
            "gate3_candidate_method",
        ] = (
            "v2_role_shock_calibrated"
        )

        out.loc[
            assists_mask,
            "gate3_candidate_intercept",
        ] = float(
            params[
                "intercept"
            ]
        )

        out.loc[
            assists_mask,
            "gate3_candidate_slope",
        ] = float(
            params[
                "slope"
            ]
        )

    combo_config = {
        "points_assists": (
            "gate3_delta_points_assists"
        ),
        "points_rebounds": (
            "gate3_delta_points_rebounds"
        ),
    }

    for prop_type, delta_column in (
        combo_config.items()
    ):
        mask = out[
            "prop_type"
        ].eq(
            prop_type
        )

        if not mask.any():
            continue

        if delta_column not in out.columns:
            raise RuntimeError(
                f"{prop_type} missing {delta_column}"
            )

        params = probability_parameters[
            prop_type
        ]

        raw = pd.to_numeric(
            out.loc[
                mask,
                "q_over_nonpush",
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        base = _frozen_calibrate(
            raw,
            prop_type,
            calibration_policy,
        )

        delta = pd.to_numeric(
            out.loc[
                mask,
                delta_column,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        mean = float(
            params[
                "standardization_mean"
            ]
        )

        std = float(
            params[
                "standardization_std"
            ]
        )

        gamma = float(
            params[
                "gamma"
            ]
        )

        if (
            not np.isfinite(
                std
            )
            or std <= 0
        ):
            raise RuntimeError(
                f"{prop_type}: invalid Gate 3 standardization std"
            )

        z = (
            delta - mean
        ) / std

        candidate = _sigmoid(
            _logit(
                base
            )
            + gamma
            * z
        )

        out.loc[
            mask,
            "gate3_candidate_q_over_nonpush",
        ] = candidate

        out.loc[
            mask,
            "gate3_candidate_method",
        ] = "v2_role_increment"

        out.loc[
            mask,
            "gate3_candidate_gamma",
        ] = gamma

        out.loc[
            mask,
            "gate3_candidate_standardization_mean",
        ] = mean

        out.loc[
            mask,
            "gate3_candidate_standardization_std",
        ] = std

    return out
