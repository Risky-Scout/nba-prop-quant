from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

import joblib
import numpy as np
import pandas as pd

from scipy import optimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


V2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V2_ROOT.parents[2]

LOCK_DIR = V2_ROOT / "research/v2_gate1_lock"
OUTPUT_DIR = V2_ROOT / "research/v2_gate2_certification_outputs"

MEANS_PATH = (
    PROJECT_ROOT
    / "data/processed/selected_means_distribution_split.parquet"
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

LINEUPS_PATH = (
    V2_ROOT
    / "research/v2_lineup_backfill_2025/lineups_normalized.parquet"
)

POLICY_PATH = (
    LOCK_DIR
    / "CANDIDATE_POLICY.json"
)

PREREG_PATH = (
    LOCK_DIR
    / "GATE2_CERTIFICATION_PREREGISTRATION.md"
)

LOCK_CHECKSUM_PATH = (
    LOCK_DIR
    / "SHA256SUMS.txt"
)

PROP_TYPES = [
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "threes",
    "points_assists",
    "points_rebounds",
    "rebounds_assists",
    "points_rebounds_assists",
]

CHANGED_PROPS = {
    "assists",
    "points_assists",
    "points_rebounds",
}

N_BOOTSTRAP = 5000
BOOTSTRAP_BASE_SEED = 20260830
EPS = 1e-6


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


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(V2_ROOT), *args],
        text=True,
    ).strip()


def verify_lock() -> dict:
    if not POLICY_PATH.exists():
        raise RuntimeError(
            "Gate 1 candidate policy missing"
        )

    expected = {}

    for raw in LOCK_CHECKSUM_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        if not raw.strip():
            continue

        checksum, filename = raw.split(
            None,
            1,
        )

        expected[
            filename.strip()
        ] = checksum.strip()

    for filename, checksum in expected.items():
        path = LOCK_DIR / filename

        actual = sha256(path)

        if actual != checksum:
            raise RuntimeError(
                f"Gate 1 lock checksum mismatch: {filename}"
            )

    policy = json.loads(
        POLICY_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not policy.get(
        "no_more_2025_tuning_before_gate2"
    ):
        raise RuntimeError(
            "Gate 1 policy does not prohibit further 2025 tuning"
        )

    expected_candidates = {
        "assists": "v2_role_shock_calibrated",
        "points_assists": "v2_role_increment",
        "points_rebounds": "v2_role_increment",
    }

    for prop, candidate in expected_candidates.items():
        actual = (
            policy[
                "locked_prop_policy"
            ][prop]["candidate"]
        )

        if actual != candidate:
            raise RuntimeError(
                f"Unexpected locked candidate for {prop}: {actual}"
            )

    return policy


def logit(p):
    p = np.clip(
        np.asarray(p, float),
        EPS,
        1 - EPS,
    )

    return np.log(
        p / (1 - p)
    )


def sigmoid(x):
    x = np.clip(
        np.asarray(x, float),
        -40.0,
        40.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    )


def fit_platt(y, p):
    y = np.asarray(
        y,
        float,
    )

    x = logit(p)

    def loss(beta):
        eta = (
            beta[0]
            + beta[1] * x
        )

        return float(
            np.sum(
                np.logaddexp(
                    0,
                    eta,
                )
                - y * eta
            )
        )

    result = optimize.minimize(
        loss,
        np.array(
            [0.0, 1.0]
        ),
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(
            f"Platt optimization failed: {result.message}"
        )

    return result.x


def apply_platt(p, beta):
    return sigmoid(
        beta[0]
        + beta[1] * logit(p)
    )


def fit_gamma(
    y,
    p,
    delta,
):
    y = np.asarray(
        y,
        float,
    )

    delta = np.asarray(
        delta,
        float,
    )

    mean = float(
        np.mean(delta)
    )

    std = float(
        np.std(delta)
    )

    if std <= 1e-12:
        return (
            0.0,
            mean,
            1.0,
        )

    z = (
        delta - mean
    ) / std

    base = logit(p)

    def loss(x):
        gamma = float(
            np.atleast_1d(x)[0]
        )

        eta = (
            base
            + gamma * z
        )

        nll = np.sum(
            np.logaddexp(
                0,
                eta,
            )
            - y * eta
        )

        penalty = (
            5.0
            * gamma ** 2
        )

        return float(
            nll + penalty
        )

    result = optimize.minimize(
        loss,
        np.array([0.0]),
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(
            f"Gamma optimization failed: {result.message}"
        )

    return (
        float(result.x[0]),
        mean,
        std,
    )


def apply_gamma(
    p,
    delta,
    gamma,
    mean,
    std,
):
    z = (
        np.asarray(delta, float)
        - mean
    ) / std

    return sigmoid(
        logit(p)
        + gamma * z
    )


def brier(y, p):
    return float(
        np.mean(
            (
                np.asarray(p, float)
                - np.asarray(y, float)
            )
            ** 2
        )
    )


def per_row_brier(y, p):
    return (
        np.asarray(p, float)
        - np.asarray(y, float)
    ) ** 2


def logloss(y, p):
    y = np.asarray(
        y,
        float,
    )

    p = np.clip(
        np.asarray(p, float),
        EPS,
        1 - EPS,
    )

    return float(
        np.mean(
            -(
                y * np.log(p)
                + (1-y)
                * np.log(1-p)
            )
        )
    )


def per_row_logloss(y, p):
    y = np.asarray(
        y,
        float,
    )

    p = np.clip(
        np.asarray(p, float),
        EPS,
        1 - EPS,
    )

    return -(
        y * np.log(p)
        + (1-y)
        * np.log(1-p)
    )


def calibration_fit(y, p):
    beta = fit_platt(
        y,
        p,
    )

    return (
        float(beta[0]),
        float(beta[1]),
    )


def ece(y, p):
    z = pd.DataFrame(
        {
            "y": np.asarray(y, float),
            "p": np.asarray(p, float),
        }
    )

    z["bin"] = pd.qcut(
        z["p"],
        10,
        duplicates="drop",
    )

    total = len(z)
    value = 0.0

    for _, g in z.groupby(
        "bin",
        observed=True,
    ):
        value += (
            len(g)
            / total
            * abs(
                g["p"].mean()
                - g["y"].mean()
            )
        )

    return float(value)


def safe_auc(y, p):
    y = np.asarray(
        y,
        float,
    )

    if len(
        np.unique(y)
    ) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def build_role_minutes(
    means: pd.DataFrame,
    lineups: pd.DataFrame,
):
    h = means.loc[
        means[
            "distribution_split"
        ]
        .astype(str)
        .eq("holdout_2025")
    ].copy()

    date_col = (
        "date"
        if "date" in h.columns
        else "game_date"
    )

    game_dates = (
        h[
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

    l = (
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

    l["_date"] = pd.to_datetime(
        l["_date"]
    )

    l = l.sort_values(
        [
            "_date",
            "game_id",
            "team_id",
            "player_id",
        ]
    ).copy()

    l["starter"] = (
        l["starter"]
        .astype(bool)
        .astype(int)
    )

    l["prev_starter"] = (
        l.groupby(
            "player_id"
        )["starter"]
        .shift(1)
    )

    l["starter_rate5"] = (
        l.groupby(
            "player_id",
            group_keys=False,
        )["starter"]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    5,
                    min_periods=1,
                )
                .mean()
        )
    )

    l["starter_rate10"] = (
        l.groupby(
            "player_id",
            group_keys=False,
        )["starter"]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(
                    10,
                    min_periods=1,
                )
                .mean()
        )
    )

    l["starter_surprise"] = (
        l["starter"]
        - l["starter_rate10"]
    )

    l["promoted_to_starter"] = (
        l["starter"].eq(1)
        & l["prev_starter"].eq(0)
    ).astype(int)

    l["demoted_to_bench"] = (
        l["starter"].eq(0)
        & l["prev_starter"].eq(1)
    ).astype(int)

    team_rows = []

    for (
        game_id,
        team_id,
        date,
    ), g in l.groupby(
        [
            "game_id",
            "team_id",
            "_date",
        ],
        sort=False,
    ):
        starters = set(
            g.loc[
                g["starter"].eq(1),
                "player_id",
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

        team_rows.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "_date": date,
                "starter_set": starters,
                "team_lineup_rows": len(g),
            }
        )

    team_games = (
        pd.DataFrame(
            team_rows
        )
        .sort_values(
            [
                "team_id",
                "_date",
                "game_id",
            ]
        )
    )

    previous = {}
    team_features = []

    for row in team_games.itertuples(
        index=False
    ):
        current = set(
            row.starter_set
        )

        prior = previous.get(
            row.team_id
        )

        if prior is None:
            overlap = np.nan
            new_starters = np.nan
            lost_starters = np.nan
        else:
            overlap = len(
                current & prior
            )

            new_starters = len(
                current - prior
            )

            lost_starters = len(
                prior - current
            )

        team_features.append(
            {
                "game_id": row.game_id,
                "team_id": row.team_id,
                "team_starter_overlap": overlap,
                "team_new_starters": new_starters,
                "team_lost_starters": lost_starters,
                "team_lineup_rows": row.team_lineup_rows,
            }
        )

        previous[
            row.team_id
        ] = current

    l = l.merge(
        pd.DataFrame(
            team_features
        ),
        on=[
            "game_id",
            "team_id",
        ],
        how="left",
        validate="many_to_one",
    )

    role_cols = [
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
    ]

    h = h.merge(
        l[
            role_cols
        ],
        on=[
            "game_id",
            "player_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    h["_date"] = pd.to_datetime(
        h[
            date_col
        ]
    )

    h = h.loc[
        h["minutes"].notna()
        & h["expected_minutes"].notna()
        & h["expected_minutes"].gt(0)
    ].copy()

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
        c
        for c in base_features
        if c in h.columns
    ]

    role_features = (
        base_features
        + [
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
        ]
    )

    game_ids = (
        h[
            [
                "game_id",
                "_date",
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
        )[
            "game_id"
        ]
        .tolist()
    )

    split = TimeSeriesSplit(
        n_splits=5
    )

    h[
        "minutes_role_oof"
    ] = np.nan

    fold_rows = []

    for fold, (
        tr,
        te,
    ) in enumerate(
        split.split(
            game_ids
        ),
        1,
    ):
        train_games = {
            game_ids[i]
            for i in tr
        }

        test_games = {
            game_ids[i]
            for i in te
        }

        train = h.loc[
            h["game_id"]
            .isin(
                train_games
            )
        ]

        test = h.loc[
            h["game_id"]
            .isin(
                test_games
            )
        ]

        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=3.0,
            random_state=20260830,
        )

        model.fit(
            train[
                role_features
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ),
            (
                train["minutes"]
                - train["expected_minutes"]
            ),
        )

        pred = np.maximum(
            test[
                "expected_minutes"
            ].to_numpy(float)
            + model.predict(
                test[
                    role_features
                ].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
            ),
            0.0,
        )

        h.loc[
            test.index,
            "minutes_role_oof",
        ] = pred

        fold_rows.append(
            {
                "stage": "role_minutes",
                "fold": fold,
                "train_games": len(
                    train_games
                ),
                "test_games": len(
                    test_games
                ),
                "train_rows": len(
                    train
                ),
                "test_rows": len(
                    test
                ),
            }
        )

    return (
        h,
        pd.DataFrame(
            fold_rows
        ),
        l[
            [
                "game_id",
                "player_id",
                "starter",
            ]
        ].drop_duplicates(
            [
                "game_id",
                "player_id",
            ]
        ),
    )


def collapsed_quotes(
    quotes,
    prop,
):
    q = quotes.loc[
        quotes[
            "prop_type"
        ]
        .astype(str)
        .eq(prop)
    ].copy()

    if q.empty:
        raise RuntimeError(
            f"No quote rows for {prop}"
        )

    q = (
        q.groupby(
            [
                "game_id",
                "player_id",
                "prop_type",
                "line_value",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            actual=(
                "actual",
                "first",
            ),
            q_frozen=(
                "q_selected",
                "mean",
            ),
            q_market=(
                "market_q_over",
                "mean",
            ),
        )
    )

    return q


def finalize_contract_frame(q):
    q = q.loc[
        q["actual"].notna()
        & q["line_value"].notna()
        & q["q_frozen"].notna()
        & q["q_market"].notna()
    ].copy()

    push = np.isclose(
        q["actual"],
        q["line_value"],
    )

    q = q.loc[
        ~push
    ].copy()

    q["y"] = (
        q["actual"]
        > q["line_value"]
    ).astype(float)

    return q


def probability_from_mu(
    frame,
    fitted,
    mu,
):
    line = (
        frame[
            "line_value"
        ]
        .to_numpy(float)
    )

    lower = (
        np.ceil(
            line
        ).astype(int)
        - 1
    )

    upper = (
        np.floor(
            line
        ).astype(int)
    )

    under = np.asarray(
        fitted.cdf(
            lower,
            np.asarray(
                mu,
                float,
            ),
            frame,
        ),
        dtype=float,
    )

    over = (
        1.0
        - np.asarray(
            fitted.cdf(
                upper,
                np.asarray(
                    mu,
                    float,
                ),
                frame,
            ),
            dtype=float,
        )
    )

    return (
        over
        / np.maximum(
            over + under,
            1e-12,
        )
    )


def build_assists_candidate(
    quotes,
    role_oof,
    marginals,
):
    meta = role_oof.loc[
        role_oof[
            "minutes_role_oof"
        ].notna(),
        [
            "game_id",
            "player_id",
            "_date",
            "starter",
            "expected_minutes",
            "minutes_role_oof",
            "mu_selected_ast",
        ],
    ].copy()

    q = collapsed_quotes(
        quotes,
        "assists",
    )

    q = q.merge(
        meta,
        on=[
            "game_id",
            "player_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    q = finalize_contract_frame(
        q
    )

    q = q.loc[
        q[
            "mu_selected_ast"
        ].notna()
        & q[
            "expected_minutes"
        ].gt(0)
    ].copy()

    candidate_mu = (
        q[
            "mu_selected_ast"
        ].to_numpy(float)
        * q[
            "minutes_role_oof"
        ].to_numpy(float)
        / q[
            "expected_minutes"
        ].to_numpy(float)
    )

    q[
        "q_role_raw"
    ] = probability_from_mu(
        q,
        marginals[
            "ast"
        ],
        np.maximum(
            candidate_mu,
            1e-8,
        ),
    )

    ordered_games = (
        q[
            [
                "game_id",
                "_date",
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
        )[
            "game_id"
        ]
        .tolist()
    )

    split = TimeSeriesSplit(
        n_splits=4
    )

    q[
        "q_candidate"
    ] = np.nan

    params = []

    for fold, (
        tr,
        te,
    ) in enumerate(
        split.split(
            ordered_games
        ),
        1,
    ):
        train_games = {
            ordered_games[i]
            for i in tr
        }

        test_games = {
            ordered_games[i]
            for i in te
        }

        train = q.loc[
            q["game_id"]
            .isin(
                train_games
            )
        ]

        mask = q[
            "game_id"
        ].isin(
            test_games
        )

        beta = fit_platt(
            train["y"],
            train[
                "q_role_raw"
            ],
        )

        q.loc[
            mask,
            "q_candidate",
        ] = apply_platt(
            q.loc[
                mask,
                "q_role_raw",
            ],
            beta,
        )

        params.append(
            {
                "prop_type": "assists",
                "candidate": "v2_role_shock_calibrated",
                "fold": fold,
                "parameter": "platt",
                "value_1": float(
                    beta[0]
                ),
                "value_2": float(
                    beta[1]
                ),
                "train_games": len(
                    train_games
                ),
                "test_games": len(
                    test_games
                ),
            }
        )

    q = q.loc[
        q[
            "q_candidate"
        ].notna()
    ].copy()

    q[
        "candidate_method"
    ] = (
        "v2_role_shock_calibrated"
    )

    return (
        q,
        params,
    )


def build_combo_candidate(
    quotes,
    meta,
    prop,
    delta_col,
):
    q = collapsed_quotes(
        quotes,
        prop,
    )

    q = q.merge(
        meta[
            [
                "game_id",
                "player_id",
                "_date",
                "starter",
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

    q = finalize_contract_frame(
        q
    )

    q = q.loc[
        q[
            delta_col
        ].notna()
    ].copy()

    ordered_games = (
        q[
            [
                "game_id",
                "_date",
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
        )[
            "game_id"
        ]
        .tolist()
    )

    split = TimeSeriesSplit(
        n_splits=4
    )

    q[
        "q_candidate"
    ] = np.nan

    params = []

    for fold, (
        tr,
        te,
    ) in enumerate(
        split.split(
            ordered_games
        ),
        1,
    ):
        train_games = {
            ordered_games[i]
            for i in tr
        }

        test_games = {
            ordered_games[i]
            for i in te
        }

        train = q.loc[
            q[
                "game_id"
            ].isin(
                train_games
            )
        ]

        mask = q[
            "game_id"
        ].isin(
            test_games
        )

        gamma, mean, std = fit_gamma(
            train[
                "y"
            ],
            train[
                "q_frozen"
            ],
            train[
                delta_col
            ],
        )

        q.loc[
            mask,
            "q_candidate",
        ] = apply_gamma(
            q.loc[
                mask,
                "q_frozen",
            ],
            q.loc[
                mask,
                delta_col,
            ],
            gamma,
            mean,
            std,
        )

        params.append(
            {
                "prop_type": prop,
                "candidate": "v2_role_increment",
                "fold": fold,
                "parameter": "gamma",
                "value_1": gamma,
                "value_2": np.nan,
                "standardization_mean": mean,
                "standardization_std": std,
                "train_games": len(
                    train_games
                ),
                "test_games": len(
                    test_games
                ),
            }
        )

    q = q.loc[
        q[
            "q_candidate"
        ].notna()
    ].copy()

    q[
        "candidate_method"
    ] = "v2_role_increment"

    return (
        q,
        params,
    )


def build_frozen_candidate(
    quotes,
    prop,
    base_meta,
):
    q = collapsed_quotes(
        quotes,
        prop,
    )

    q = q.merge(
        base_meta,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
        validate="many_to_one",
    )

    q = finalize_contract_frame(
        q
    )

    q[
        "q_candidate"
    ] = q[
        "q_frozen"
    ]

    q[
        "candidate_method"
    ] = "frozen_selected_v1"

    return q


def metric_rows(
    prop,
    frame,
):
    rows = []

    for label, col in [
        (
            "candidate",
            "q_candidate",
        ),
        (
            "frozen",
            "q_frozen",
        ),
        (
            "market",
            "q_market",
        ),
    ]:
        g = frame[
            [
                "y",
                col,
            ]
        ].dropna()

        intercept, slope = (
            calibration_fit(
                g["y"],
                g[col],
            )
        )

        rows.append(
            {
                "prop_type": prop,
                "model": label,
                "rows": len(g),
                "games": int(
                    frame.loc[
                        g.index,
                        "game_id",
                    ].nunique()
                ),
                "brier": brier(
                    g["y"],
                    g[col],
                ),
                "logloss": logloss(
                    g["y"],
                    g[col],
                ),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece_deciles": ece(
                    g["y"],
                    g[col],
                ),
                "auc": safe_auc(
                    g["y"],
                    g[col],
                ),
                "sharpness": float(
                    np.mean(
                        np.abs(
                            g[col]
                            - 0.5
                        )
                    )
                ),
                "mean_probability": float(
                    g[col].mean()
                ),
                "observed_over_rate": float(
                    g["y"].mean()
                ),
            }
        )

    return rows


def bootstrap_comparison(
    prop,
    frame,
    comparator_col,
    comparison,
):
    y = frame[
        "y"
    ].to_numpy(float)

    candidate = frame[
        "q_candidate"
    ].to_numpy(float)

    comparator = frame[
        comparator_col
    ].to_numpy(float)

    diff_brier = (
        per_row_brier(
            y,
            candidate,
        )
        - per_row_brier(
            y,
            comparator,
        )
    )

    diff_logloss = (
        per_row_logloss(
            y,
            candidate,
        )
        - per_row_logloss(
            y,
            comparator,
        )
    )

    grouped = pd.DataFrame(
        {
            "game_id": frame[
                "game_id"
            ].to_numpy(),
            "brier_diff": diff_brier,
            "logloss_diff": diff_logloss,
        }
    ).groupby(
        "game_id",
        sort=True,
    ).agg(
        brier_sum=(
            "brier_diff",
            "sum",
        ),
        logloss_sum=(
            "logloss_diff",
            "sum",
        ),
        rows=(
            "brier_diff",
            "size",
        ),
    )

    brier_sum = grouped[
        "brier_sum"
    ].to_numpy(float)

    logloss_sum = grouped[
        "logloss_sum"
    ].to_numpy(float)

    counts = grouped[
        "rows"
    ].to_numpy(float)

    games = len(
        grouped
    )

    seed_text = (
        f"{BOOTSTRAP_BASE_SEED}:"
        f"{prop}:"
        f"{comparison}"
    )

    seed = int(
        hashlib.sha256(
            seed_text.encode(
                "utf-8"
            )
        ).hexdigest()[:8],
        16,
    )

    rng = np.random.default_rng(
        seed
    )

    sample_index = rng.integers(
        0,
        games,
        size=(
            N_BOOTSTRAP,
            games,
        ),
    )

    denominator = counts[
        sample_index
    ].sum(
        axis=1
    )

    brier_boot = (
        brier_sum[
            sample_index
        ].sum(
            axis=1
        )
        / denominator
    )

    logloss_boot = (
        logloss_sum[
            sample_index
        ].sum(
            axis=1
        )
        / denominator
    )

    rows = []

    for metric, point, boot in [
        (
            "brier",
            float(
                brier_sum.sum()
                / counts.sum()
            ),
            brier_boot,
        ),
        (
            "logloss",
            float(
                logloss_sum.sum()
                / counts.sum()
            ),
            logloss_boot,
        ),
    ]:
        ci_low, ci_high = np.quantile(
            boot,
            [
                0.025,
                0.975,
            ],
        )

        rows.append(
            {
                "prop_type": prop,
                "comparison": comparison,
                "metric": metric,
                "rows": len(frame),
                "games": games,
                "bootstrap_resamples": N_BOOTSTRAP,
                "seed": seed,
                "point_delta_candidate_minus_comparator": point,
                "ci95_low": float(
                    ci_low
                ),
                "ci95_high": float(
                    ci_high
                ),
                "p_candidate_better": float(
                    np.mean(
                        boot < 0
                    )
                ),
            }
        )

    return rows


def subgroup_rows(
    prop,
    frame,
):
    f = frame.copy()

    f[
        "month"
    ] = pd.to_datetime(
        f["_date"]
    ).dt.strftime(
        "%Y-%m"
    )

    try:
        f[
            "line_quintile"
        ] = pd.qcut(
            f[
                "line_value"
            ],
            5,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        f[
            "line_quintile"
        ] = "not_available"

    confidence = np.abs(
        f[
            "q_frozen"
        ]
        .to_numpy(float)
        - 0.5
    )

    f[
        "frozen_confidence"
    ] = pd.cut(
        confidence,
        bins=[
            0.0,
            0.025,
            0.05,
            0.10,
            0.15,
            1.0,
        ],
        labels=[
            "0-.025",
            ".025-.05",
            ".05-.10",
            ".10-.15",
            ".15+",
        ],
        right=False,
        include_lowest=True,
    ).astype(str)

    if "starter" in f.columns:
        f[
            "starter_status"
        ] = np.where(
            f[
                "starter"
            ].eq(1),
            "starter",
            np.where(
                f[
                    "starter"
                ].eq(0),
                "bench",
                "unknown",
            ),
        )
    else:
        f[
            "starter_status"
        ] = "unknown"

    rows = []

    for family in [
        "month",
        "line_quintile",
        "frozen_confidence",
        "starter_status",
    ]:
        for group_value, g in f.groupby(
            family,
            observed=True,
            dropna=False,
        ):
            if len(g) == 0:
                continue

            frozen_brier = brier(
                g[
                    "y"
                ],
                g[
                    "q_frozen"
                ],
            )

            candidate_brier = brier(
                g[
                    "y"
                ],
                g[
                    "q_candidate"
                ],
            )

            frozen_ll = logloss(
                g[
                    "y"
                ],
                g[
                    "q_frozen"
                ],
            )

            candidate_ll = logloss(
                g[
                    "y"
                ],
                g[
                    "q_candidate"
                ],
            )

            brier_delta = (
                candidate_brier
                - frozen_brier
            )

            ll_delta = (
                candidate_ll
                - frozen_ll
            )

            eligible = (
                len(g) >= 300
            )

            subgroup_fail = (
                eligible
                and brier_delta > 0.010
                and ll_delta > 0.020
            )

            rows.append(
                {
                    "prop_type": prop,
                    "subgroup_family": family,
                    "subgroup_value": str(
                        group_value
                    ),
                    "rows": len(g),
                    "games": int(
                        g[
                            "game_id"
                        ].nunique()
                    ),
                    "candidate_brier": candidate_brier,
                    "frozen_brier": frozen_brier,
                    "brier_delta_candidate_minus_frozen": brier_delta,
                    "candidate_logloss": candidate_ll,
                    "frozen_logloss": frozen_ll,
                    "logloss_delta_candidate_minus_frozen": ll_delta,
                    "eligible_n_ge_300": eligible,
                    "subgroup_gate_fail": subgroup_fail,
                }
            )

    return rows


def lookup_metric(
    metrics,
    prop,
    model,
):
    return metrics.loc[
        metrics[
            "prop_type"
        ].eq(prop)
        & metrics[
            "model"
        ].eq(model)
    ].iloc[0]


def lookup_boot(
    bootstrap,
    prop,
    comparison,
    metric,
):
    return bootstrap.loc[
        bootstrap[
            "prop_type"
        ].eq(prop)
        & bootstrap[
            "comparison"
        ].eq(comparison)
        & bootstrap[
            "metric"
        ].eq(metric)
    ].iloc[0]


def build_verdicts(
    policy,
    metrics,
    bootstrap,
    subgroups,
):
    rows = []

    for prop in PROP_TYPES:
        locked = (
            policy[
                "locked_prop_policy"
            ][prop]
        )

        candidate_metric = lookup_metric(
            metrics,
            prop,
            "candidate",
        )

        frozen_metric = lookup_metric(
            metrics,
            prop,
            "frozen",
        )

        market_brier_boot = lookup_boot(
            bootstrap,
            prop,
            "candidate_vs_market",
            "brier",
        )

        market_ll_boot = lookup_boot(
            bootstrap,
            prop,
            "candidate_vs_market",
            "logloss",
        )

        development_market_superior = bool(
            market_brier_boot[
                "point_delta_candidate_minus_comparator"
            ] < 0
            and market_brier_boot[
                "ci95_high"
            ] < 0
            and market_ll_boot[
                "point_delta_candidate_minus_comparator"
            ] < 0
            and market_ll_boot[
                "ci95_high"
            ] < 0
        )

        if prop not in CHANGED_PROPS:
            rows.append(
                {
                    "prop_type": prop,
                    "locked_candidate": locked[
                        "candidate"
                    ],
                    "gate2_candidate_type": "retained_frozen",
                    "gate2_survives": True,
                    "gate3_candidate": "frozen_selected_v1",
                    "brier_point_better_than_frozen": True,
                    "logloss_point_better_than_frozen": True,
                    "brier_p_better_ge_095": True,
                    "logloss_p_better_ge_095": True,
                    "at_least_one_ci_excludes_zero_favorable": True,
                    "ece_gate": True,
                    "calibration_slope_gate": True,
                    "auc_gate": True,
                    "subgroup_gate": True,
                    "development_market_superior": development_market_superior,
                }
            )

            continue

        brier_boot = lookup_boot(
            bootstrap,
            prop,
            "candidate_vs_frozen",
            "brier",
        )

        ll_boot = lookup_boot(
            bootstrap,
            prop,
            "candidate_vs_frozen",
            "logloss",
        )

        brier_point = bool(
            candidate_metric[
                "brier"
            ]
            < frozen_metric[
                "brier"
            ]
        )

        ll_point = bool(
            candidate_metric[
                "logloss"
            ]
            < frozen_metric[
                "logloss"
            ]
        )

        brier_prob = bool(
            brier_boot[
                "p_candidate_better"
            ] >= 0.95
        )

        ll_prob = bool(
            ll_boot[
                "p_candidate_better"
            ] >= 0.95
        )

        ci_gate = bool(
            brier_boot[
                "ci95_high"
            ] < 0
            or ll_boot[
                "ci95_high"
            ] < 0
        )

        ece_gate = bool(
            candidate_metric[
                "ece_deciles"
            ]
            <= frozen_metric[
                "ece_deciles"
            ]
            + 0.010
        )

        slope_gate = bool(
            0.75
            <= candidate_metric[
                "calibration_slope"
            ]
            <= 1.35
        )

        auc_gate = bool(
            candidate_metric[
                "auc"
            ]
            >= frozen_metric[
                "auc"
            ]
            - 0.005
        )

        subgroup_failures = (
            subgroups[
                subgroups[
                    "prop_type"
                ].eq(prop)
                & subgroups[
                    "subgroup_gate_fail"
                ].eq(True)
            ]
        )

        subgroup_gate = bool(
            len(
                subgroup_failures
            ) == 0
        )

        survives = bool(
            brier_point
            and ll_point
            and brier_prob
            and ll_prob
            and ci_gate
            and ece_gate
            and slope_gate
            and auc_gate
            and subgroup_gate
        )

        rows.append(
            {
                "prop_type": prop,
                "locked_candidate": locked[
                    "candidate"
                ],
                "gate2_candidate_type": "changed_candidate",
                "gate2_survives": survives,
                "gate3_candidate": (
                    locked[
                        "candidate"
                    ]
                    if survives
                    else "frozen_selected_v1"
                ),
                "brier_point_better_than_frozen": brier_point,
                "logloss_point_better_than_frozen": ll_point,
                "brier_p_better_ge_095": brier_prob,
                "logloss_p_better_ge_095": ll_prob,
                "at_least_one_ci_excludes_zero_favorable": ci_gate,
                "ece_gate": ece_gate,
                "calibration_slope_gate": slope_gate,
                "auc_gate": auc_gate,
                "subgroup_gate": subgroup_gate,
                "development_market_superior": development_market_superior,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    policy = verify_lock()

    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Gate 2 output directory already exists; "
            "refusing to overwrite certification evidence"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    branch = git_output(
        "branch",
        "--show-current",
    )

    runner_commit = git_output(
        "rev-parse",
        "HEAD",
    )

    lock_commit = git_output(
        "log",
        "-1",
        "--format=%H",
        "--",
        str(
            POLICY_PATH.relative_to(
                V2_ROOT
            )
        ),
    )

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

    role_oof, role_fold_rows, starter_lookup = (
        build_role_minutes(
            means,
            lineups,
        )
    )

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

    base_meta = (
        holdout[
            [
                "game_id",
                "player_id",
                date_col,
            ]
        ]
        .drop_duplicates(
            [
                "game_id",
                "player_id",
            ]
        )
        .rename(
            columns={
                date_col: "_date",
            }
        )
        .merge(
            starter_lookup,
            on=[
                "game_id",
                "player_id",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    base_meta[
        "_date"
    ] = pd.to_datetime(
        base_meta[
            "_date"
        ]
    )

    candidate_frames = {}
    parameter_rows = []

    assists_frame, assists_params = (
        build_assists_candidate(
            quotes,
            role_oof,
            marginals,
        )
    )

    candidate_frames[
        "assists"
    ] = assists_frame

    parameter_rows.extend(
        assists_params
    )

    combo_meta = role_oof.loc[
        role_oof[
            "minutes_role_oof"
        ].notna()
        & role_oof[
            "expected_minutes"
        ].gt(0)
    ].copy()

    ratio = (
        combo_meta[
            "minutes_role_oof"
        ]
        / combo_meta[
            "expected_minutes"
        ]
    )

    combo_meta[
        "delta_pts"
    ] = (
        combo_meta[
            "mu_selected_pts"
        ]
        * (
            ratio - 1.0
        )
    )

    combo_meta[
        "delta_reb"
    ] = (
        combo_meta[
            "mu_selected_reb"
        ]
        * (
            ratio - 1.0
        )
    )

    combo_meta[
        "delta_ast"
    ] = (
        combo_meta[
            "mu_selected_ast"
        ]
        * (
            ratio - 1.0
        )
    )

    combo_meta[
        "delta_P+A"
    ] = (
        combo_meta[
            "delta_pts"
        ]
        + combo_meta[
            "delta_ast"
        ]
    )

    combo_meta[
        "delta_P+R"
    ] = (
        combo_meta[
            "delta_pts"
        ]
        + combo_meta[
            "delta_reb"
        ]
    )

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
        frame, params = (
            build_combo_candidate(
                quotes,
                combo_meta,
                prop,
                delta_col,
            )
        )

        candidate_frames[
            prop
        ] = frame

        parameter_rows.extend(
            params
        )

    for prop in PROP_TYPES:
        if prop in candidate_frames:
            continue

        candidate_frames[
            prop
        ] = build_frozen_candidate(
            quotes,
            prop,
            base_meta,
        )

    contract_frames = []

    metric_output = []
    bootstrap_output = []
    subgroup_output = []

    for prop in PROP_TYPES:
        frame = candidate_frames[
            prop
        ].copy()

        required = [
            "q_candidate",
            "q_frozen",
            "q_market",
            "y",
            "game_id",
            "player_id",
            "line_value",
            "_date",
        ]

        missing = [
            c
            for c in required
            if c not in frame.columns
        ]

        if missing:
            raise RuntimeError(
                f"{prop} missing certification columns: {missing}"
            )

        frame[
            "prop_type"
        ] = prop

        frame[
            "candidate_probability"
        ] = frame[
            "q_candidate"
        ]

        contract_frames.append(
            frame[
                [
                    "prop_type",
                    "game_id",
                    "player_id",
                    "line_value",
                    "actual",
                    "y",
                    "_date",
                    "starter",
                    "candidate_method",
                    "q_candidate",
                    "q_frozen",
                    "q_market",
                ]
            ].copy()
        )

        metric_output.extend(
            metric_rows(
                prop,
                frame,
            )
        )

        bootstrap_output.extend(
            bootstrap_comparison(
                prop,
                frame,
                "q_frozen",
                "candidate_vs_frozen",
            )
        )

        bootstrap_output.extend(
            bootstrap_comparison(
                prop,
                frame,
                "q_market",
                "candidate_vs_market",
            )
        )

        subgroup_output.extend(
            subgroup_rows(
                prop,
                frame,
            )
        )

    metrics = pd.DataFrame(
        metric_output
    )

    bootstrap = pd.DataFrame(
        bootstrap_output
    )

    subgroups = pd.DataFrame(
        subgroup_output
    )

    parameters = pd.DataFrame(
        parameter_rows
    )

    role_fold_rows = role_fold_rows.copy()

    contracts = pd.concat(
        contract_frames,
        ignore_index=True,
    )

    verdicts = build_verdicts(
        policy,
        metrics,
        bootstrap,
        subgroups,
    )

    metrics.to_csv(
        OUTPUT_DIR
        / "certification_metrics.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUTPUT_DIR
        / "cluster_bootstrap_results.csv",
        index=False,
    )

    subgroups.to_csv(
        OUTPUT_DIR
        / "subgroup_robustness.csv",
        index=False,
    )

    parameters.to_csv(
        OUTPUT_DIR
        / "candidate_parameters.csv",
        index=False,
    )

    role_fold_rows.to_csv(
        OUTPUT_DIR
        / "role_minutes_folds.csv",
        index=False,
    )

    verdicts.to_csv(
        OUTPUT_DIR
        / "candidate_verdicts.csv",
        index=False,
    )

    contracts.to_parquet(
        OUTPUT_DIR
        / "certification_contracts.parquet",
        index=False,
    )

    role_oof[
        [
            "game_id",
            "player_id",
            "_date",
            "starter",
            "expected_minutes",
            "minutes",
            "minutes_role_oof",
        ]
    ].to_parquet(
        OUTPUT_DIR
        / "role_minutes_oof.parquet",
        index=False,
    )

    gate3_policy = {
        row[
            "prop_type"
        ]: row[
            "gate3_candidate"
        ]
        for _, row in verdicts.iterrows()
    }

    (
        OUTPUT_DIR
        / "PROVISIONAL_GATE3_POLICY.json"
    ).write_text(
        json.dumps(
            {
                "generated_by_gate2": True,
                "development_only": True,
                "gate3_policy": gate3_policy,
                "warning": (
                    "This policy is based on retrospective "
                    "2025 development evidence. Prospective "
                    "market-superiority claims require Gate 3."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    changed = verdicts.loc[
        verdicts[
            "gate2_candidate_type"
        ].eq(
            "changed_candidate"
        )
    ]

    summary_lines = [
        "# NBA Prop Quant v2 — Gate 2 Development Certification",
        "",
        f"Runner commit: `{runner_commit}`",
        f"Gate 1 lock commit: `{lock_commit}`",
        f"Bootstrap resamples: `{N_BOOTSTRAP}`",
        "",
        "## Locked candidate verdicts",
        "",
        "| Prop | Gate 2 survives | Gate 3 candidate | Development market superior |",
        "|---|---:|---|---:|",
    ]

    for _, row in verdicts.iterrows():
        summary_lines.append(
            f"| {row['prop_type']} "
            f"| {bool(row['gate2_survives'])} "
            f"| {row['gate3_candidate']} "
            f"| {bool(row['development_market_superior'])} |"
        )

    summary_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Gate 2 uses retrospective 2025 development evidence.",
            "Historical lineup records are not timestamped at original announcement time.",
            "No Gate 2 result constitutes prospective market superiority.",
            "Gate 3 prospective testing is required before such a claim.",
            "",
        ]
    )

    (
        OUTPUT_DIR
        / "GATE2_CERTIFICATION_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    input_hashes = {
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
        str(
            POLICY_PATH.relative_to(
                V2_ROOT
            )
        ): sha256(
            POLICY_PATH
        ),
        str(
            PREREG_PATH.relative_to(
                V2_ROOT
            )
        ): sha256(
            PREREG_PATH
        ),
    }

    manifest = {
        "certification": "nba_prop_quant_v2_gate2",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "development_only": True,
        "branch": branch,
        "runner_commit": runner_commit,
        "gate1_lock_commit": lock_commit,
        "gate1_source_commit": policy[
            "gate1_source_commit"
        ],
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "supported_props": PROP_TYPES,
        "changed_candidates": sorted(
            CHANGED_PROPS
        ),
        "predefined_subgroups": [
            "calendar month",
            "line quintile",
            "frozen probability confidence band",
            "starter/bench status",
        ],
        "input_sha256": input_hashes,
        "contract_rows_by_prop": {
            prop: int(
                len(
                    candidate_frames[
                        prop
                    ]
                )
            )
            for prop in PROP_TYPES
        },
        "games_by_prop": {
            prop: int(
                candidate_frames[
                    prop
                ][
                    "game_id"
                ].nunique()
            )
            for prop in PROP_TYPES
        },
        "historical_lineup_warning": (
            "2025 historical lineup records are development-only "
            "because original announcement timestamps are unavailable."
        ),
        "prospective_claim_allowed": False,
    }

    (
        OUTPUT_DIR
        / "certification_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    checksum_targets = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
    )

    checksum_lines = [
        f"{sha256(path)}  {path.name}"
        for path in checksum_targets
    ]

    (
        OUTPUT_DIR
        / "SHA256SUMS.txt"
    ).write_text(
        "\n".join(
            checksum_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("GATE 2 CERTIFICATION METRICS")
    print("=" * 120)

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("CHANGED CANDIDATE VERDICTS")
    print("=" * 120)

    print(
        changed.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("CANDIDATE VS FROZEN BOOTSTRAP")
    print("=" * 120)

    print(
        bootstrap.loc[
            bootstrap[
                "prop_type"
            ].isin(
                sorted(
                    CHANGED_PROPS
                )
            )
            & bootstrap[
                "comparison"
            ].eq(
                "candidate_vs_frozen"
            )
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("DEVELOPMENT MARKET-SUPERIOR FLAGS")
    print("=" * 120)

    print(
        verdicts[
            [
                "prop_type",
                "gate2_survives",
                "gate3_candidate",
                "development_market_superior",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "PASS: Gate 2 certification run completed under locked preregistration."
    )


if __name__ == "__main__":
    main()
