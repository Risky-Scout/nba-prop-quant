from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from nba_prop_quant.pricing import (
    PROP_TO_TARGET,
    fair_american,
    price_combo_lines,
    price_single_prop_frame,
)
from nba_prop_quant.production import (
    calibrate_over_probability,
    calibrated_unconditional_probabilities,
    load_verified_manifest_metadata,
)

CANONICAL_SCHEMA_VERSION = "nba_fair_price_v1"
EXPECTED_FREEZE_ID = "nba_prop_quant_20260818T205213Z"
EXPECTED_FREEZE_STAGE = "external_test_deployment"
DEFAULT_COMBO_SIMULATIONS = 20_000
DEFAULT_BASE_SEED = 73

COMBO_TARGETS: dict[str, tuple[str, ...]] = {
    "points_rebounds": ("pts", "reb"),
    "points_assists": ("pts", "ast"),
    "rebounds_assists": ("reb", "ast"),
    "points_rebounds_assists": ("pts", "reb", "ast"),
    "stocks": ("stl", "blk"),
}

REQUEST_KEYS = ("game_id", "player_id", "prop_type", "line_value")


def stable_event_seed(
    game_id: int,
    player_id: int,
    prop_type: str,
    base_seed: int,
) -> int:
    prop_code = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(prop_type)
    )
    return int(
        (
            int(base_seed)
            + 1_000_003 * int(game_id)
            + 9_176 * int(player_id)
            + 37 * prop_code
        )
        % (2**32 - 1)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(
    columns: Iterable[str],
    candidates: tuple[str, ...],
) -> str | None:
    present = set(columns)
    return next((name for name in candidates if name in present), None)


def _coalesce_metadata(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    default: Any = None,
) -> pd.Series:
    source = _first_present(frame.columns, candidates)
    if source is None:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[source]


def _load_and_validate_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    freeze_id = str(manifest.get("freeze_id", ""))
    freeze_stage = str(manifest.get("freeze_stage", ""))

    if freeze_id != EXPECTED_FREEZE_ID:
        raise RuntimeError(
            f"Integration feed expects freeze {EXPECTED_FREEZE_ID}, "
            f"got {freeze_id!r}"
        )
    if freeze_stage != EXPECTED_FREEZE_STAGE:
        raise RuntimeError(
            f"Integration feed requires freeze stage "
            f"{EXPECTED_FREEZE_STAGE}, got {freeze_stage!r}"
        )
    return manifest


@dataclass(frozen=True)
class FairPriceArtifacts:
    marginals: dict[str, Any]
    copula: Any
    dependence_policy: dict[str, Any]
    calibration_policy: dict[str, Any]
    manifest: dict[str, Any]

    @classmethod
    def load(
        cls,
        model_dir: Path,
        manifest_path: Path | None = None,
    ) -> "FairPriceArtifacts":
        model_dir = Path(model_dir).resolve()
        project_root = model_dir.parent

        verified = load_verified_manifest_metadata(
            model_dir=model_dir,
            project_root=project_root,
            allow_predeployment=False,
        )

        verified_manifest_path = Path(
            verified["manifest_path"]
        ).resolve()

        selected_manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else verified_manifest_path
        )

        if not selected_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing selected frozen manifest: "
                f"{selected_manifest_path}"
            )

        selected_manifest_sha256 = sha256_file(
            selected_manifest_path
        )

        if (
            selected_manifest_sha256
            != verified["manifest_sha256"]
        ):
            raise RuntimeError(
                "Selected manifest does not match the "
                "verified production LATEST manifest."
            )

        manifest = _load_and_validate_manifest(
            selected_manifest_path
        )

        if (
            str(manifest["freeze_id"])
            != verified["freeze_id"]
            or str(manifest["freeze_stage"])
            != verified["freeze_stage"]
        ):
            raise RuntimeError(
                "Selected manifest identity does not match "
                "the verified production manifest."
            )

        manifest = dict(manifest)
        manifest["verified_manifest_sha256"] = (
            verified["manifest_sha256"]
        )

        marginals = joblib.load(model_dir / "marginals.joblib")
        copula = joblib.load(model_dir / "copula.joblib")
        dependence_policy = load_json(
            model_dir / "combo_dependence_policy.json"
        )
        calibration_policy = load_json(
            model_dir / "market_probability_calibration_policy.json"
        )

        return cls(
            marginals=marginals,
            copula=copula,
            dependence_policy=dependence_policy,
            calibration_policy=calibration_policy,
            manifest=manifest,
        )

    @property
    def supported_props(self) -> set[str]:
        calibrated = set(self.calibration_policy.get("props", {}))
        singles = set(PROP_TO_TARGET)
        combos = set(
            self.dependence_policy.get("combos", {})
        )
        return calibrated & (singles | combos)


def _validate_requests(
    requests: pd.DataFrame,
    supported_props: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = set(REQUEST_KEYS) - set(requests.columns)
    if missing:
        raise ValueError(
            f"Fair-price requests missing columns: {sorted(missing)}"
        )

    work = requests.copy()
    work["game_id"] = pd.to_numeric(
        work["game_id"], errors="coerce"
    )
    work["player_id"] = pd.to_numeric(
        work["player_id"], errors="coerce"
    )
    work["line_value"] = pd.to_numeric(
        work["line_value"], errors="coerce"
    )
    work["prop_type"] = work["prop_type"].astype(str)

    reason = pd.Series("", index=work.index, dtype=object)
    reason.loc[work["game_id"].isna()] = "invalid_game_id"
    reason.loc[work["player_id"].isna()] = "invalid_player_id"
    reason.loc[work["line_value"].isna()] = "invalid_line_value"
    reason.loc[
        ~work["prop_type"].isin(supported_props)
    ] = "unsupported_prop_type"

    rejected = work.loc[reason.ne("")].copy()
    if not rejected.empty:
        rejected["reject_reason"] = reason.loc[
            rejected.index
        ].to_numpy()

    valid = work.loc[reason.eq("")].copy()
    valid["game_id"] = valid["game_id"].astype(int)
    valid["player_id"] = valid["player_id"].astype(int)

    return valid, rejected


def _expected_value_for_prop(
    frame: pd.DataFrame,
) -> pd.Series:
    values = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    for prop_type, target in PROP_TO_TARGET.items():
        mask = frame["prop_type"].eq(prop_type)
        column = f"mu_selected_{target}"
        if mask.any() and column in frame.columns:
            values.loc[mask] = pd.to_numeric(
                frame.loc[mask, column],
                errors="coerce",
            )

    for prop_type, targets in COMBO_TARGETS.items():
        mask = frame["prop_type"].eq(prop_type)
        if not mask.any():
            continue
        columns = [
            f"mu_selected_{target}"
            for target in targets
        ]
        if all(column in frame.columns for column in columns):
            values.loc[mask] = (
                frame.loc[mask, columns]
                .apply(pd.to_numeric, errors="coerce")
                .sum(axis=1)
            )

    return values


class FairPriceEngine:
    def __init__(
        self,
        artifacts: FairPriceArtifacts,
        *,
        combo_simulations: int = DEFAULT_COMBO_SIMULATIONS,
        base_seed: int = DEFAULT_BASE_SEED,
    ) -> None:
        self.artifacts = artifacts
        self.combo_simulations = int(combo_simulations)
        self.base_seed = int(base_seed)
        self.mu_columns = {
            target: f"mu_selected_{target}"
            for target in ("pts", "reb", "ast", "stl", "blk", "fg3m")
        }

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        *,
        manifest_path: Path | None = None,
        combo_simulations: int = DEFAULT_COMBO_SIMULATIONS,
        base_seed: int = DEFAULT_BASE_SEED,
    ) -> "FairPriceEngine":
        return cls(
            FairPriceArtifacts.load(
                model_dir=Path(model_dir),
                manifest_path=manifest_path,
            ),
            combo_simulations=combo_simulations,
            base_seed=base_seed,
        )

    def price_requests(
        self,
        projections: pd.DataFrame,
        requests: pd.DataFrame,
        *,
        reject_out_players: bool = True,
        generated_at_utc: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        valid, rejected = _validate_requests(
            requests,
            self.artifacts.supported_props,
        )

        if valid.empty:
            return pd.DataFrame(), rejected

        projection_keys = ["game_id", "player_id"]
        missing_projection_keys = (
            set(projection_keys) - set(projections.columns)
        )
        if missing_projection_keys:
            raise ValueError(
                "Projection frame missing columns: "
                f"{sorted(missing_projection_keys)}"
            )

        projection_copy = projections.copy()
        projection_copy["game_id"] = pd.to_numeric(
            projection_copy["game_id"],
            errors="coerce",
        )
        projection_copy["player_id"] = pd.to_numeric(
            projection_copy["player_id"],
            errors="coerce",
        )

        merged = valid.merge(
            projection_copy,
            on=projection_keys,
            how="left",
            suffixes=("_request", ""),
            indicator=True,
            validate="many_to_one",
        )

        unmatched = merged["_merge"].ne("both")
        if unmatched.any():
            part = merged.loc[unmatched, valid.columns].copy()
            part["reject_reason"] = "projection_not_found"
            rejected = pd.concat(
                [rejected, part],
                ignore_index=True,
            )
            merged = merged.loc[~unmatched].copy()

        merged = merged.drop(columns=["_merge"])

        if reject_out_players and "availability_out" in merged.columns:
            out_mask = pd.to_numeric(
                merged["availability_out"],
                errors="coerce",
            ).fillna(0).astype(int).eq(1)
            if out_mask.any():
                part = merged.loc[out_mask, valid.columns].copy()
                part["reject_reason"] = "player_currently_out"
                rejected = pd.concat(
                    [rejected, part],
                    ignore_index=True,
                )
                merged = merged.loc[~out_mask].copy()

        if merged.empty:
            return pd.DataFrame(), rejected

        for target, column in self.mu_columns.items():
            if column not in merged.columns:
                raise RuntimeError(
                    f"Projection frame missing canonical mean column {column}"
                )

        priced_parts: list[pd.DataFrame] = []

        for prop_type, target in PROP_TO_TARGET.items():
            part = merged.loc[
                merged["prop_type"].eq(prop_type)
            ].copy()
            if part.empty:
                continue

            probabilities = price_single_prop_frame(
                part,
                target=target,
                marginal=self.artifacts.marginals[target],
                mu_column=self.mu_columns[target],
                line_column="line_value",
            )
            for column in probabilities.columns:
                part[column] = probabilities[column].to_numpy()

            part["dependence_lambda"] = 0.0
            part["pricing_method"] = "single_exact"
            part["pricing_seed"] = np.nan
            part["combo_simulations"] = 0
            priced_parts.append(part)

        combo_props = (
            set(self.artifacts.dependence_policy.get("combos", {}))
            & set(self.artifacts.calibration_policy.get("props", {}))
        )

        for prop_type in sorted(combo_props):
            part = merged.loc[
                merged["prop_type"].eq(prop_type)
            ].copy()
            if part.empty:
                continue

            lambda_value = float(
                self.artifacts.dependence_policy[
                    "combos"
                ][prop_type]["production_lambda"]
            )

            event_probabilities: list[pd.DataFrame] = []

            for (
                game_id,
                player_id,
                _,
            ), group in part.groupby(
                ["game_id", "player_id", "prop_type"],
                sort=False,
            ):
                row = group.iloc[0]
                lines = np.sort(
                    group["line_value"].unique()
                )
                event_seed = stable_event_seed(
                    int(game_id),
                    int(player_id),
                    prop_type,
                    self.base_seed,
                )

                probabilities = price_combo_lines(
                    row=row,
                    prop_type=prop_type,
                    lines=lines,
                    marginals=self.artifacts.marginals,
                    mu_columns=self.mu_columns,
                    copula=self.artifacts.copula,
                    dependence_lambda=lambda_value,
                    simulations=self.combo_simulations,
                    seed=event_seed,
                )
                probabilities["game_id"] = int(game_id)
                probabilities["player_id"] = int(player_id)
                probabilities["prop_type"] = prop_type
                probabilities["pricing_seed"] = event_seed
                event_probabilities.append(probabilities)

            lookup = pd.concat(
                event_probabilities,
                ignore_index=True,
            )

            part = part.merge(
                lookup,
                on=[
                    "game_id",
                    "player_id",
                    "prop_type",
                    "line_value",
                ],
                how="left",
                validate="many_to_one",
                suffixes=("", "_priced"),
            )
            part["pricing_method"] = np.where(
                lambda_value <= 1e-12,
                "combo_independent_exact",
                "combo_copula_mc",
            )
            part["combo_simulations"] = np.where(
                lambda_value <= 1e-12,
                0,
                self.combo_simulations,
            )
            priced_parts.append(part)

        if not priced_parts:
            return pd.DataFrame(), rejected

        priced = pd.concat(
            priced_parts,
            ignore_index=True,
        )

        raw_mass = (
            priced["p_over"]
            + priced["p_under"]
            + priced["p_push"]
        )
        max_raw_error = float(
            np.max(np.abs(raw_mass - 1.0))
        )
        if max_raw_error > 5e-3:
            raise RuntimeError(
                f"Raw pricing probability mass error too large: "
                f"{max_raw_error}"
            )

        priced["raw_p_over"] = pd.to_numeric(
            priced["p_over"],
            errors="coerce",
        )
        priced["raw_p_under"] = pd.to_numeric(
            priced["p_under"],
            errors="coerce",
        )
        priced["raw_p_push"] = pd.to_numeric(
            priced["p_push"],
            errors="coerce",
        )
        priced["raw_q_over_nonpush"] = pd.to_numeric(
            priced["q_over_nonpush"],
            errors="coerce",
        )
        priced["raw_q_under_nonpush"] = pd.to_numeric(
            priced["q_under_nonpush"],
            errors="coerce",
        )

        priced["selected_q_over_nonpush"] = np.nan
        priced["calibration_method"] = ""
        priced["calibration_intercept"] = np.nan
        priced["calibration_slope"] = np.nan

        for prop_type in sorted(
            priced["prop_type"].dropna().unique()
        ):
            mask = priced["prop_type"].eq(prop_type)
            calibrated, method, intercept, slope = (
                calibrate_over_probability(
                    priced.loc[
                        mask,
                        "raw_q_over_nonpush",
                    ].to_numpy(dtype=float),
                    str(prop_type),
                    self.artifacts.calibration_policy,
                )
            )
            priced.loc[
                mask,
                "selected_q_over_nonpush",
            ] = calibrated
            priced.loc[
                mask,
                "calibration_method",
            ] = method
            if intercept is not None:
                priced.loc[
                    mask,
                    "calibration_intercept",
                ] = intercept
            if slope is not None:
                priced.loc[
                    mask,
                    "calibration_slope",
                ] = slope

        priced["selected_q_under_nonpush"] = (
            1.0 - priced["selected_q_over_nonpush"]
        )

        selected_p_over, selected_p_under = (
            calibrated_unconditional_probabilities(
                priced[
                    "selected_q_over_nonpush"
                ].to_numpy(dtype=float),
                priced[
                    "raw_p_push"
                ].to_numpy(dtype=float),
            )
        )
        priced["selected_p_over"] = selected_p_over
        priced["selected_p_under"] = selected_p_under
        priced["selected_p_push"] = priced["raw_p_push"]

        selected_mass = (
            priced["selected_p_over"]
            + priced["selected_p_under"]
            + priced["selected_p_push"]
        )
        max_selected_error = float(
            np.max(np.abs(selected_mass - 1.0))
        )
        if max_selected_error > 1e-9:
            raise RuntimeError(
                f"Selected pricing probability mass error too large: "
                f"{max_selected_error}"
            )

        priced["fair_over_american"] = [
            fair_american(value)
            for value in priced[
                "selected_q_over_nonpush"
            ].to_numpy(dtype=float)
        ]
        priced["fair_under_american"] = [
            fair_american(value)
            for value in priced[
                "selected_q_under_nonpush"
            ].to_numpy(dtype=float)
        ]
        priced["fair_over_decimal"] = (
            1.0
            / priced["selected_q_over_nonpush"]
        )
        priced["fair_under_decimal"] = (
            1.0
            / priced["selected_q_under_nonpush"]
        )

        priced["mu_selected"] = _expected_value_for_prop(
            priced
        )

        priced["expected_minutes"] = pd.to_numeric(
            _coalesce_metadata(
                priced,
                (
                    "expected_minutes",
                    "minutes_pred",
                    "predicted_minutes",
                    "mu_minutes",
                ),
                np.nan,
            ),
            errors="coerce",
        )
        priced["player_name"] = _coalesce_metadata(
            priced,
            (
                "player_name",
                "full_name",
                "name",
            ),
            None,
        )
        priced["team"] = _coalesce_metadata(
            priced,
            (
                "team",
                "team_abbreviation",
                "team_abbr",
            ),
            None,
        )
        priced["opponent"] = _coalesce_metadata(
            priced,
            (
                "opponent",
                "opponent_abbreviation",
                "opponent_abbr",
            ),
            None,
        )
        raw_game_date = _coalesce_metadata(
            priced,
            (
                "game_date",
                "slate_date",
                "date",
            ),
            None,
        )
        parsed_game_date = pd.to_datetime(
            raw_game_date,
            errors="coerce",
        )
        priced["game_date"] = (
            parsed_game_date.dt.strftime("%Y-%m-%d")
        )
        unparsed_game_date = (
            parsed_game_date.isna()
            & raw_game_date.notna()
        )
        priced.loc[
            unparsed_game_date,
            "game_date",
        ] = raw_game_date.loc[
            unparsed_game_date
        ].astype(str)
        priced["availability_status"] = _coalesce_metadata(
            priced,
            ("availability_status",),
            "unknown",
        )

        generated_at_utc = generated_at_utc or datetime.now(
            timezone.utc
        ).isoformat()
        priced["schema_version"] = CANONICAL_SCHEMA_VERSION
        priced["generated_at_utc"] = generated_at_utc
        priced["freeze_id"] = self.artifacts.manifest["freeze_id"]
        priced["freeze_stage"] = self.artifacts.manifest["freeze_stage"]
        priced["market_odds_used"] = False
        priced["market_independent"] = True
        priced["auto_bet"] = False

        canonical_columns = [
            "schema_version",
            "generated_at_utc",
            "freeze_id",
            "freeze_stage",
            "game_id",
            "game_date",
            "player_id",
            "player_name",
            "team",
            "opponent",
            "prop_type",
            "line_value",
            "expected_minutes",
            "mu_selected",
            "raw_p_over",
            "raw_p_under",
            "raw_p_push",
            "raw_q_over_nonpush",
            "raw_q_under_nonpush",
            "selected_p_over",
            "selected_p_under",
            "selected_p_push",
            "selected_q_over_nonpush",
            "selected_q_under_nonpush",
            "fair_over_american",
            "fair_under_american",
            "fair_over_decimal",
            "fair_under_decimal",
            "calibration_method",
            "calibration_intercept",
            "calibration_slope",
            "dependence_lambda",
            "pricing_method",
            "pricing_seed",
            "combo_simulations",
            "availability_status",
            "market_odds_used",
            "market_independent",
            "auto_bet",
        ]

        passthrough = [
            column
            for column in valid.columns
            if column not in canonical_columns
            and column not in REQUEST_KEYS
        ]

        output_columns = canonical_columns + [
            column
            for column in passthrough
            if column in priced.columns
        ]

        return (
            priced[output_columns].copy(),
            rejected.reset_index(drop=True),
        )


def build_surface_requests(
    projections: pd.DataFrame,
    prop_types: Iterable[str],
    *,
    radius: float,
    step: float,
) -> pd.DataFrame:
    if radius <= 0:
        raise ValueError("surface radius must be positive")
    if step <= 0:
        raise ValueError("surface step must be positive")

    required = {"game_id", "player_id"}
    missing = required - set(projections.columns)
    if missing:
        raise ValueError(
            f"Projection frame missing columns: {sorted(missing)}"
        )

    records: list[dict[str, Any]] = []

    for _, row in projections.iterrows():
        for prop_type in prop_types:
            if prop_type in PROP_TO_TARGET:
                targets = (PROP_TO_TARGET[prop_type],)
            elif prop_type in COMBO_TARGETS:
                targets = COMBO_TARGETS[prop_type]
            else:
                continue

            columns = [
                f"mu_selected_{target}"
                for target in targets
            ]
            if not all(column in row.index for column in columns):
                continue

            mu = float(
                sum(float(row[column]) for column in columns)
            )
            low = max(
                0.0,
                np.floor((mu - radius) / step) * step,
            )
            high = np.ceil((mu + radius) / step) * step
            line_values = np.arange(
                low,
                high + step * 0.5,
                step,
            )

            for line_value in line_values:
                records.append(
                    {
                        "game_id": int(row["game_id"]),
                        "player_id": int(row["player_id"]),
                        "prop_type": prop_type,
                        "line_value": float(
                            np.round(line_value, 6)
                        ),
                    }
                )

    return pd.DataFrame.from_records(records)
