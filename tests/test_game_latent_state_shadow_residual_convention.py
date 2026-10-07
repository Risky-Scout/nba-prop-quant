"""The versioned OOF marginal-training convention.

SHADOW / RESEARCH ONLY.

The OOF residual builder used to filter to rows carrying every stat and every
selected mean before splitting by stat, so the per-stat filter under it could
never remove anything and the effective convention was a six-way intersection.
The validator and production both filter per target. The marginal-convention
audit measured the gap and classified it NON_MATERIAL, so the frozen candidate
keeps the old dataset; what changes is which convention a *future* retraining
gets by default.

These tests pin three things: that the aligned convention is the default, that
the historical one is still reachable so the frozen dataset stays reproducible,
and that the two differ in exactly the way the audit described -- a season
missing one stat's mean used to cost all six marginals their rows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT_ROOT / "research/game_latent_state/02_build_oof_residuals.py"
AUDIT = (
    PROJECT_ROOT
    / "research/marginal_convention_audit/marginal_convention_audit.json"
)

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")


@pytest.fixture(scope="module")
def builder():
    """The driver, imported without running it."""
    if not DRIVER.exists():
        pytest.skip(f"missing driver: {DRIVER}")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("oof_residual_builder", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(missing_mean_in_season: int | None = 2017) -> pd.DataFrame:
    """Two seasons; optionally one of them has no ``blk`` selected mean.

    This is the shape the audit found in the real record: 2017 carried no
    ``reb`` or ``blk`` selected mean, so every row of it failed the joint
    filter.
    """
    rows = []
    for season in (2017, 2018):
        for index in range(50):
            row = {
                "season": season,
                "game_id": 1000 * season + index // 10,
                "player_id": index,
            }
            for stat in STATS:
                row[stat] = float(index % 7)
                row[f"mu_selected_{stat}"] = 1.0 + (index % 5)
            if missing_mean_in_season == season:
                row["mu_selected_blk"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# the default is the live convention
# ----------------------------------------------------------------------


def test_the_default_convention_is_the_live_per_stat_one(builder) -> None:
    assert builder.DEFAULT_RESIDUAL_CONVENTION == builder.RESIDUAL_CONVENTION_V2
    assert "per_stat" in builder.RESIDUAL_CONVENTION_V2


def test_the_command_line_default_is_the_live_convention(
    builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default that only lives in a keyword argument is not a default."""
    monkeypatch.setattr(sys, "argv", ["02_build_oof_residuals.py"])
    args = builder.parse_args()
    assert args.residual_convention == builder.RESIDUAL_CONVENTION_V2


def test_both_conventions_are_offered_and_nothing_else_is(builder) -> None:
    assert builder.RESIDUAL_CONVENTIONS == (
        builder.RESIDUAL_CONVENTION_V1,
        builder.RESIDUAL_CONVENTION_V2,
    )
    with pytest.raises(ValueError, match="unknown residual convention"):
        builder.marginal_training_pool(_frame(), _frame(), "v3_whatever")


# ----------------------------------------------------------------------
# the two conventions differ in the audited way
# ----------------------------------------------------------------------


def test_the_historical_convention_loses_a_season_from_every_marginal(
    builder,
) -> None:
    frame = _frame(missing_mean_in_season=2017)
    selected = [f"mu_selected_{stat}" for stat in STATS]
    target_pool = frame.dropna(subset=[*STATS, *selected])

    historical = builder.marginal_training_pool(
        frame, target_pool, builder.RESIDUAL_CONVENTION_V1
    )
    aligned = builder.marginal_training_pool(
        frame, target_pool, builder.RESIDUAL_CONVENTION_V2
    )

    # Under v1 the 2017 rows are gone before any stat is considered, so even
    # the five stats whose own mean is present lose them.
    assert set(historical["season"].unique()) == {2018}
    for stat in STATS:
        rows = historical.dropna(subset=[stat, f"mu_selected_{stat}"])
        assert set(rows["season"].unique()) == {2018}

    # Under v2 only the stat whose mean is missing loses them.
    assert set(aligned["season"].unique()) == {2017, 2018}
    for stat in STATS:
        rows = aligned.dropna(subset=[stat, f"mu_selected_{stat}"])
        expected = {2018} if stat == "blk" else {2017, 2018}
        assert set(rows["season"].unique()) == expected, stat


def test_the_conventions_agree_when_nothing_is_missing(builder) -> None:
    """The alignment only bites where the record is ragged."""
    frame = _frame(missing_mean_in_season=None)
    selected = [f"mu_selected_{stat}" for stat in STATS]
    target_pool = frame.dropna(subset=[*STATS, *selected])

    historical = builder.marginal_training_pool(
        frame, target_pool, builder.RESIDUAL_CONVENTION_V1
    )
    aligned = builder.marginal_training_pool(
        frame, target_pool, builder.RESIDUAL_CONVENTION_V2
    )
    assert len(historical) == len(aligned) == len(frame)


def test_which_rows_receive_a_residual_does_not_depend_on_the_convention(
    builder,
) -> None:
    """Only the training side moves; the target pool is the joint one in both.

    A row has to carry all six residuals before it can contribute to a
    cross-stat second moment, so widening the target pool would add rows the
    factor fit then drops. Pinned so a later change has to be deliberate.
    """
    source = DRIVER.read_text(encoding="utf-8")
    assert "usable = frame.dropna(subset=[*stats, *selected_columns]).copy()" in source
    assert "target_rows = usable.loc[usable[\"season\"] == season]" in source
    assert (
        "train = training_pool.loc[training_pool[\"season\"] < season]" in source
    )


# ----------------------------------------------------------------------
# the certification is the audit's, not a retyped version of it
# ----------------------------------------------------------------------


def test_the_certification_constants_match_the_committed_audit(builder) -> None:
    if not AUDIT.exists():
        pytest.skip(f"missing audit artifact: {AUDIT}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    decision = audit["decision"]
    certification = builder.RESIDUAL_CONVENTION_CERTIFICATION

    assert certification["classification"] == decision["classification"] == "NON_MATERIAL"
    assert certification["max_robust_bucket_shift_z"] == pytest.approx(
        decision["measured"]["largest_key_bucket_shift_z"]
    )
    assert certification["bucket_threshold_z"] == pytest.approx(
        decision["thresholds"]["max_key_bucket_shift_z"]
    )
    assert certification["global_latent_movement"] == pytest.approx(
        decision["measured"]["largest_global_latent_movement"]
    )
    assert certification["global_threshold"] == pytest.approx(
        decision["thresholds"]["max_global_latent_movement"]
    )
    assert certification["max_robust_bucket_shift_z"] < certification[
        "bucket_threshold_z"
    ]
    assert certification["global_latent_movement"] < certification["global_threshold"]


def test_the_certification_points_at_a_file_that_exists(builder) -> None:
    referenced = PROJECT_ROOT / builder.RESIDUAL_CONVENTION_CERTIFICATION["audit"]
    assert referenced.exists()
