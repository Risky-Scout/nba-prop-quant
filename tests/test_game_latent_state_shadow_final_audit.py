"""Audit regression tests for the final upstream remediation.

SHADOW / RESEARCH ONLY. These tests add no modelling. Every one of them is a
regression lock on a claim the audit makes about the *committed artifacts*, so
that a later edit cannot quietly change the answer to a question the audit
already settled:

* the two held-out game universes are what they are, and the rule that picks
  them reads nothing about any model,
* the count-space RMSE is one definition that reproduces every stored value,
* the item-6 coverage table is twelve cross-fitted observations and its
  coverage and mean squared z come from the same twelve,
* the leave-one-out sensitivity is arithmetic, not assertion,
* every provenance hash matches the file on disk,
* the half-life dial does nothing under the selected temporal treatment,
* every acceptance requirement in the brief is mapped onto an implemented gate
  or recorded as never implemented, including the three that fail,
* the stored joint-event probabilities recompute to the reported log loss.

The residual parquet and the grades parquet are generated artifacts excluded
by ``.gitignore``, so the tests that read them skip rather than fail when they
are absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nba_prop_quant.research.game_latent_state.remediation import (
    COVERAGE_LEVELS,
    MIN_CELL_GAMES,
    MIN_CELL_PAIRS,
    SeasonSeries,
    coverage_loss,
    coverage_table,
    fit_student_t_random_effects,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "research" / "final_upstream_remediation"
REPAIR_DIR = PROJECT_ROOT / "research" / "game_latent_state_bucket_repair"

STATS = ("pts", "reb", "ast", "stl", "blk", "fg3m")
HOLDOUT_SEASONS = (2024, 2025)

CANDIDATE_SPEC_HASH = (
    "3229bbc83a4388c270a21008a75afacc3479cfd6f6390c8737f041412d2f2eb4"
)
CONTROL_SPEC_HASH = (
    "c9b46e3a7497cfee52832f29397397b843aafb8e5f770a39157ce508f8157bc0"
)

#: The driver's default held-out games per validation season. The published
#: repair run passed 300 explicitly; the paired runs took this default.
PAIRED_GAMES_PER_SEASON = 200
REPAIR_GAMES_PER_SEASON = 300


# ----------------------------------------------------------------------
# artifacts
# ----------------------------------------------------------------------


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inner() -> dict:
    return _load(ARTIFACT_DIR / "inner_selection.json")


@pytest.fixture(scope="module")
def frozen() -> dict:
    return _load(ARTIFACT_DIR / "frozen_spec.json")


@pytest.fixture(scope="module")
def candidate_spec() -> dict:
    return _load(ARTIFACT_DIR / "factor_spec.json")


@pytest.fixture(scope="module")
def gates() -> dict:
    return _load(ARTIFACT_DIR / "gate_report.json")


@pytest.fixture(scope="module")
def audit() -> dict:
    path = ARTIFACT_DIR / "final_audit.json"
    if not path.exists():
        pytest.skip("final_audit.json has not been generated in this tree")
    return _load(path)


@pytest.fixture(scope="module")
def reports() -> dict[str, dict]:
    return {
        "published_repair_600": _load(REPAIR_DIR / "validation_report.json"),
        "paired_control_400": _load(ARTIFACT_DIR / "control_validation_report.json"),
        "paired_candidate_400": _load(ARTIFACT_DIR / "validation_report.json"),
    }


@pytest.fixture(scope="module")
def residuals() -> pd.DataFrame:
    path = ARTIFACT_DIR / "oof_gaussian_residuals.parquet"
    if not path.exists():
        pytest.skip("the residual parquet is a generated artifact and is absent")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def grades() -> dict[str, pd.DataFrame]:
    candidate = ARTIFACT_DIR / "joint_event_grades.parquet"
    # The control run writes into a scratch root, recorded in
    # paired_joint_calibration.json; it is never committed.
    control = Path(
        _load(ARTIFACT_DIR / "paired_joint_calibration.json")["control_artifact"]
    )
    if not candidate.exists():
        pytest.skip("the grades parquet is a generated artifact and is absent")
    frames = {"candidate": pd.read_parquet(candidate)}
    if control.exists():
        frames["control"] = pd.read_parquet(control)
    return frames


def _upstream_spec():
    path = ARTIFACT_DIR / "upstream_spec.py"
    spec = importlib.util.spec_from_file_location(
        "final_remediation_upstream_spec_audit", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# 1. evaluation universe identity
# ----------------------------------------------------------------------


def _evenly_spaced(game_ids: list[int], count: int) -> list[int]:
    """The validator's own subsample rule, copied term for term."""
    if count <= 0 or count >= len(game_ids):
        return list(game_ids)
    picks = np.linspace(0, len(game_ids) - 1, count)
    return [game_ids[round(index)] for index in picks]


def _universe(residuals: pd.DataFrame, per_season: int) -> set[int]:
    chosen: set[int] = set()
    for season in HOLDOUT_SEASONS:
        available = sorted(
            int(value)
            for value in residuals.loc[
                residuals["season"] == season, "game_id"
            ].unique()
        )
        chosen |= set(_evenly_spaced(available, per_season))
    return chosen


def test_the_two_held_out_game_universes_have_the_sizes_the_reports_record(
    residuals: pd.DataFrame, reports: dict[str, dict]
) -> None:
    small = _universe(residuals, PAIRED_GAMES_PER_SEASON)
    large = _universe(residuals, REPAIR_GAMES_PER_SEASON)

    assert len(small) == reports["paired_control_400"]["games_simulated"] == 400
    assert len(small) == reports["paired_candidate_400"]["games_simulated"]
    assert len(large) == reports["published_repair_600"]["games_simulated"] == 600


def test_the_four_hundred_game_universe_is_not_nested_in_the_six_hundred(
    residuals: pd.DataFrame,
) -> None:
    """Evenly spaced indices at two densities are not a sample and a subsample.

    This is the whole reason a count-space statistic measured on the simulated
    games moves between the two runs, so it is worth a lock: if the two sets
    ever became nested the reconciliation in the audit would need rewriting.
    """
    small = _universe(residuals, PAIRED_GAMES_PER_SEASON)
    large = _universe(residuals, REPAIR_GAMES_PER_SEASON)

    assert not small <= large
    assert len(small & large) == 92
    assert len(small - large) == 308


def test_every_run_evaluates_the_same_two_holdout_seasons(
    reports: dict[str, dict]
) -> None:
    for report in reports.values():
        assert report["validation_seasons"] == list(HOLDOUT_SEASONS)
        assert report["training_seasons"] == [2020, 2021, 2022, 2023]


def test_the_candidate_and_the_control_are_paired_on_one_universe(
    reports: dict[str, dict]
) -> None:
    control = reports["paired_control_400"]
    candidate = reports["paired_candidate_400"]

    assert control["games_simulated"] == candidate["games_simulated"]
    assert control["simulations_per_game"] == candidate["simulations_per_game"]
    assert control["seed"] == candidate["seed"]
    assert control["games_skipped"] == candidate["games_skipped"] == 0
    assert control["factor_spec_hash"] == CONTROL_SPEC_HASH
    assert candidate["factor_spec_hash"] == CANDIDATE_SPEC_HASH


def test_the_candidate_and_the_control_grade_the_same_events(
    grades: dict[str, pd.DataFrame]
) -> None:
    if "control" not in grades:
        pytest.skip("the control grades parquet is absent from this tree")
    candidate = grades["candidate"]
    control = grades["control"]
    keys = ["game_id", "family", "n_legs", "realized"]

    assert candidate[keys].equals(control[keys])
    assert np.allclose(
        candidate["p_baseline_production"], control["p_baseline_production"]
    )


# ----------------------------------------------------------------------
# 2. no model-specific game filtering
# ----------------------------------------------------------------------


def test_the_subsample_rule_reads_only_the_ordering_and_the_count() -> None:
    """The rule is a function of two arguments, so nothing else can enter it.

    Permuting the input changes the answer only through the sort the driver
    applies first, and re-labelling the games without changing their order
    leaves the *positions* chosen identical. A performance-aware filter could
    not have either property.
    """
    ordered = list(range(1000, 1600))
    first = _evenly_spaced(ordered, 200)
    second = _evenly_spaced(list(ordered), 200)
    assert first == second

    relabelled = [value + 10_000 for value in ordered]
    positions_of_first = [ordered.index(value) for value in first]
    positions_of_relabelled = [
        relabelled.index(value) for value in _evenly_spaced(relabelled, 200)
    ]
    assert positions_of_first == positions_of_relabelled


def test_no_run_dropped_a_game_for_any_reason_other_than_the_rule(
    reports: dict[str, dict]
) -> None:
    """The only other exclusion path is the eligibility filter, and it fired
    zero times, so the simulated set is exactly the rule's output."""
    for name, report in reports.items():
        assert report["games_skipped"] == 0, name


def test_the_validation_driver_is_byte_identical_across_the_two_runs() -> None:
    """The count-space numbers come from two runs of one driver.

    The provenance line the newer revision added reads the branch instead of
    hard-coding it; everything the metric touches is the same blob. If the
    driver ever diverges in a way that touches ``validation.py`` the audit's
    "one definition" claim stops holding, which is what this locks.
    """
    repair_sha = _load(REPAIR_DIR / "manifest.validation.json")["code_sha"]
    paired_sha = _load(ARTIFACT_DIR / "manifest.validation.json")["code_sha"]
    path = "src/nba_prop_quant/research/game_latent_state/validation.py"

    def blob(revision: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", f"{revision}:{path}"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    assert blob(repair_sha) == blob(paired_sha)


# ----------------------------------------------------------------------
# 3. count-RMSE definition consistency
# ----------------------------------------------------------------------


def test_the_count_rmse_is_an_unweighted_mean_over_twelve_buckets(
    reports: dict[str, dict]
) -> None:
    """Every stored value reproduces from its own stored bucket errors.

    Nine cases: three runs by three models. An unweighted root mean square
    over the twelve cross-player buckets reproduces all nine exactly, which
    rules out a changed weighting, a changed bucket list and a changed
    estimator as explanations for the 600-versus-400 difference.
    """
    checked = 0
    for name, report in reports.items():
        residual = report["residual_dependence"]
        assert len(residual["observed_buckets"]) == 12, name
        for model, block in residual["by_model"].items():
            errors = np.array(
                [block["bucket_errors"][key] for key in sorted(block["bucket_errors"])]
            )
            assert len(errors) == 12, (name, model)
            recomputed = float(np.sqrt(np.mean(np.square(errors))))
            stored = float(residual["count_space_cross_player_rmse"][model])
            assert recomputed == pytest.approx(stored, abs=1e-15), (name, model)
            checked += 1
    assert checked == 9


def test_the_count_space_difference_is_the_universe_and_not_the_metric(
    reports: dict[str, dict]
) -> None:
    """Latent buckets agree bitwise; count buckets do not.

    The latent dependence summary reads every held-out game rather than the
    simulated subset, so it is the one reading the subsample cannot move.
    Identical latent observations plus differing count observations localises
    the 0.007861 -> 0.006405 move to the simulated game set.
    """
    repair = reports["published_repair_600"]
    control = reports["paired_control_400"]

    assert (
        repair["latent_dependence"]["observed_buckets"]
        == control["latent_dependence"]["observed_buckets"]
    )
    assert (
        repair["latent_dependence"]["by_model"]["candidate"]["implied_buckets"]
        == control["latent_dependence"]["by_model"]["candidate"]["implied_buckets"]
    )
    assert (
        repair["residual_dependence"]["observed_buckets"]
        != control["residual_dependence"]["observed_buckets"]
    )
    assert repair["residual_dependence"]["cross_player_rmse"]["candidate"] == (
        control["residual_dependence"]["cross_player_rmse"]["candidate"]
    )


def test_the_candidate_and_control_count_observations_are_identical(
    reports: dict[str, dict]
) -> None:
    """The paired comparison is on one set of observed counts."""
    assert (
        reports["paired_control_400"]["residual_dependence"]["observed_buckets"]
        == reports["paired_candidate_400"]["residual_dependence"]["observed_buckets"]
    )


# ----------------------------------------------------------------------
# 4. uncertainty sample-size accounting
# ----------------------------------------------------------------------


def _scored_z(inner: dict) -> pd.DataFrame:
    """The twelve cross-fitted standardized errors item 6 actually scores.

    ``decide_uncertainty`` standardizes every forward record, then walks the
    folds in order and marks a fold cross-fitted only when a strictly earlier
    fold existed to set the scale from. The first fold therefore has no scale
    and is dropped, so the scored set is the second fold alone.
    """
    records = pd.DataFrame(inner["item_1_temporal"]["forward_records"]["A2_robust_student_t"])
    records["standardized"] = records["error"] / np.sqrt(
        records["predictive_sd"] ** 2 + records["observed_se"] ** 2
    )
    folds = sorted(records["target_season"].unique())
    return records.loc[records["target_season"] == folds[-1]].reset_index(drop=True)


def test_the_uncertainty_table_rests_on_twelve_bucket_seasonal_observations(
    inner: dict,
) -> None:
    records = pd.DataFrame(
        inner["item_1_temporal"]["forward_records"]["A2_robust_student_t"]
    )
    folds = sorted(int(value) for value in records["target_season"].unique())
    scored = _scored_z(inner)

    assert len(records) == 24
    assert folds == [2022, 2023]
    assert len(scored) == 12
    assert scored["bucket"].nunique() == 12
    # One target season, so twelve correlated bucket estimates rather than
    # twelve independent draws.
    assert scored["target_season"].nunique() == 1


def test_coverage_and_mean_squared_z_come_from_the_same_twelve_observations(
    inner: dict,
) -> None:
    """The brief's apparent tension needs the two to be on one sample.

    They are: ``coverage_table`` and the mean squared z both read the same
    ``usable`` block. Reproducing the stored coverage from the scored rows and
    computing the mean squared z from those same rows shows the direction
    agrees -- mean squared z above one is narrow intervals, which is what
    under-coverage reports.
    """
    scored = _scored_z(inner)
    values = scored["standardized"].to_numpy(float)
    stored = inner["item_6_uncertainty"]["candidates"]["U0_raw"]["coverage"]
    reproduced = coverage_table(values, scale=1.0)

    for key, value in stored.items():
        assert reproduced[key] == pytest.approx(value, abs=1e-12), key

    mean_squared_z = float(np.mean(np.square(values)))
    assert mean_squared_z > 1.0
    for level in COVERAGE_LEVELS:
        key = f"coverage_{int(round(level * 100))}"
        assert reproduced[key] < level


def test_the_raw_uncertainty_model_was_kept_because_nothing_beat_it(
    inner: dict,
) -> None:
    item = inner["item_6_uncertainty"]
    losses = {
        label: coverage_loss(block["coverage"])
        for label, block in item["candidates"].items()
    }
    assert item["selected"] == "U0_raw"
    assert losses["U0_raw"] == min(losses.values())
    assert len([v for v in losses.values() if v < losses["U0_raw"]]) == 0


def test_the_quoted_v2_mean_squared_z_is_a_different_universe() -> None:
    """0.665227 is not computable from the final artifact's observations.

    It is the V2 structural round's holdout statistic over four buckets. The
    audit's resolution of the tension depends on that, so the source is
    locked here rather than asserted in prose.
    """
    path = (
        PROJECT_ROOT
        / "research"
        / "game_latent_state_v2"
        / "sd_calibration_blocker.json"
    )
    blocker = _load(path)

    assert blocker["holdout_raw_mean_squared_z"] == pytest.approx(
        0.6652266156010309, abs=1e-15
    )
    assert blocker["holdout_calibrated_mean_squared_z"] == pytest.approx(
        0.390769082372168, abs=1e-15
    )
    assert blocker["holdout_seasons"] == [2024, 2025]
    # Four buckets, against item 6's twelve, and the 2024-2025 holdout rather
    # than a pre-2024 fold.
    assert len(blocker["season_estimate_path"]) == 4
    assert blocker["code_sha"] != _load(ARTIFACT_DIR / "inner_selection.json")[
        "code_sha"
    ]
    # The same artifact records a pooled inner value above three, so the sign
    # of the miscalibration is not even stable inside that round.
    assert blocker["inner_pooled_mean_squared_z"] > 3.0


# ----------------------------------------------------------------------
# 5. leave-one-out uncertainty calculations
# ----------------------------------------------------------------------


def test_leave_one_out_mean_squared_z_never_falls_below_one(inner: dict) -> None:
    """The direction of the miscalibration survives removing any observation.

    This is what separates "two heavy observations invented the result" from
    "the sample is small and tail-sensitive but the sign is stable", and it
    is the evidence behind the CAUTION verdict rather than a FAIL.
    """
    values = _scored_z(inner)["standardized"].to_numpy(float)
    for index in range(len(values)):
        held = np.delete(values, index)
        assert float(np.mean(np.square(held))) > 1.0


def test_leave_one_out_coverage_moves_by_exactly_one_observation(
    inner: dict,
) -> None:
    """Each removal can only move a coverage level by 0 or 1/11.

    An arithmetic lock on the leave-one-out table: with eleven remaining
    observations, dropping one either removes a covered observation or an
    uncovered one, so the recount must land on a multiple of 1/11 and must sit
    within one step of the eleven-observation value of the full count.
    """
    values = _scored_z(inner)["standardized"].to_numpy(float)
    for level in COVERAGE_LEVELS:
        key = f"coverage_{int(round(level * 100))}"
        full = coverage_table(values, scale=1.0)[key]
        covered_total = round(full * len(values))
        for index in range(len(values)):
            held = np.delete(values, index)
            observed = coverage_table(held, scale=1.0)[key]
            successes = round(observed * len(held))
            assert successes in (covered_total, covered_total - 1)
            assert observed == pytest.approx(successes / len(held), abs=1e-12)


def test_the_tail_is_concentrated_in_two_of_the_twelve_observations(
    inner: dict,
) -> None:
    squared = np.sort(np.square(_scored_z(inner)["standardized"].to_numpy(float)))[::-1]
    share_of_two = float(squared[:2].sum() / squared.sum())

    assert share_of_two > 0.70
    assert float(np.mean(squared) / np.median(squared)) > 3.0


# ----------------------------------------------------------------------
# 6. provenance hashes
# ----------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Each lineage row that carries an artifact digest, and the file it describes.
LINEAGE_ARTIFACTS = (
    ("04_inner_selection_artifact_sha256", "inner_selection.json"),
    ("06_dependence_temperature_artifact_sha256", "dependence_temperature.json"),
    ("08_frozen_spec_file_sha256", "frozen_spec.json"),
    ("09b_factor_spec_file_sha256", "factor_spec.json"),
    ("11_validation_artifact_sha256", "validation_report.json"),
    ("13_gate_report_sha256", "gate_report.json"),
)

#: Each lineage row that carries a commit.
LINEAGE_COMMITS = (
    "01_clean_control_base_sha",
    "02_remediation_branch_fork_sha",
    "03_inner_selection_code_sha",
    "05_dependence_temperature_code_sha",
    "07_frozen_spec_code_sha",
    "10_confirmatory_validation_code_sha",
    "12_gate_evaluator_code_sha",
    "14_report_generator_sha",
    "15_final_branch_head_sha",
)


@pytest.mark.parametrize(("row", "artifact"), LINEAGE_ARTIFACTS)
def test_every_recorded_artifact_hash_matches_the_file_on_disk(
    audit: dict, row: str, artifact: str
) -> None:
    assert audit["provenance"][row] == _sha256(ARTIFACT_DIR / artifact)


def test_the_lineage_table_has_a_row_for_every_declared_stage(
    audit: dict,
) -> None:
    """Fifteen numbered rows, so a missing stage is a failure not a silence."""
    numbered = sorted(
        key for key in audit["provenance"] if key[:2].isdigit()
    )
    leading = sorted({key[:2] for key in numbered})
    assert leading == [f"{index:02d}" for index in range(1, 16)]


def test_the_manifest_agrees_with_the_lineage_on_every_shared_hash(
    audit: dict,
) -> None:
    provenance = audit["provenance"]
    assert (
        provenance["09b_factor_spec_file_sha256"]
        == provenance["09c_factor_spec_sha256_in_manifest"]
    )
    assert (
        provenance["11_validation_artifact_sha256"]
        == provenance["11b_validation_sha256_in_manifest"]
    )


def test_the_manifest_hashes_match_the_files_they_describe() -> None:
    manifest = _load(ARTIFACT_DIR / "manifest.validation.json")
    for entry in manifest.get("artifacts", []):
        name = Path(str(entry["path"])).name
        path = ARTIFACT_DIR / name
        if not path.exists():
            continue
        assert _sha256(path) == entry["sha256"], name


@pytest.mark.parametrize("row", LINEAGE_COMMITS)
def test_every_recorded_code_sha_is_a_commit_in_this_history(
    audit: dict, row: str
) -> None:
    revision = str(audit["provenance"][row])
    assert len(revision) == 40, row
    probe = subprocess.run(
        ["git", "cat-file", "-t", revision],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, row
    assert probe.stdout.strip() == "commit", row


def test_the_selection_code_predates_the_confirmatory_run(audit: dict) -> None:
    """Selection must have been frozen before the holdout was opened.

    An ancestry check rather than a timestamp comparison, because commit
    dates can be rewritten and the parent graph cannot.
    """
    provenance = audit["provenance"]
    probe = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            str(provenance["03_inner_selection_code_sha"]),
            str(provenance["10_confirmatory_validation_code_sha"]),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0


# ----------------------------------------------------------------------
# 7. frozen-spec identity
# ----------------------------------------------------------------------


def test_the_candidate_spec_hash_is_the_same_object_everywhere(
    frozen: dict, candidate_spec: dict, gates: dict
) -> None:
    assert frozen["factor_spec_hash"] == CANDIDATE_SPEC_HASH
    assert candidate_spec["spec_hash"] == CANDIDATE_SPEC_HASH
    assert gates["candidate_factor_spec_hash"] == CANDIDATE_SPEC_HASH
    assert (
        _load(ARTIFACT_DIR / "validation_report.json")["factor_spec_hash"]
        == CANDIDATE_SPEC_HASH
    )
    assert candidate_spec["control_factor_spec_hash"] == CONTROL_SPEC_HASH


def test_the_control_really_is_the_accepted_repair(gates: dict) -> None:
    assert gates["control_factor_spec_hash"] == CONTROL_SPEC_HASH
    assert (
        _load(REPAIR_DIR / "factor_spec.json")["spec_hash"] == CONTROL_SPEC_HASH
    )


def test_the_frozen_spec_declares_no_undeclared_dial(gates: dict) -> None:
    evidence = next(
        entry["evidence"] for entry in gates["gates"] if entry["gate"] == 13
    )
    assert evidence["undeclared_spec_keys"] == []
    assert evidence["parameters_equal_to_it"] == []
    assert evidence["inner_selection_holdout_used"] is False
    assert evidence["temperature_holdout_used"] is False


def test_the_shared_season_series_reproduces_the_recorded_one(
    residuals: pd.DataFrame, inner: dict
) -> None:
    """The post-selection refactor moved code, not behaviour.

    ``season_series`` was lifted out of ``01_inner_selection.py`` into
    ``upstream_spec.py`` after ``inner_selection.json`` was written, so the
    artifact's recorded code SHA no longer names the file the estimator lives
    in. This is the equivalence evidence the provenance section cites: the
    shared helper reproduces the recorded primary-bucket series exactly.
    """
    module = _upstream_spec()
    frame = residuals.loc[~residuals["season"].isin(HOLDOUT_SEASONS)].copy()
    series = module.season_series(
        frame, STATS, bootstrap=inner["bootstrap"], seed=inner["seed"]
    )

    recorded = inner["item_1_temporal"]["primary_bucket_series"]
    reproduced = series[inner["item_1_temporal"]["primary_bucket"]]

    assert list(int(value) for value in reproduced.seasons) == recorded["seasons"]
    assert np.max(
        np.abs(np.asarray(reproduced.estimates) - np.array(recorded["estimates"]))
    ) == 0.0
    assert np.max(
        np.abs(
            np.asarray(reproduced.standard_errors)
            - np.array(recorded["standard_errors"])
        )
    ) == 0.0


# ----------------------------------------------------------------------
# 8. half-life inactivity under A2
# ----------------------------------------------------------------------


def _recorded_series(inner: dict) -> SeasonSeries:
    """The primary bucket's per-season series, as the artifact recorded it."""
    recorded = inner["item_1_temporal"]["primary_bucket_series"]
    return SeasonSeries(
        name=inner["item_1_temporal"]["primary_bucket"],
        seasons=tuple(int(value) for value in recorded["seasons"]),
        estimates=np.array(recorded["estimates"], dtype=float),
        standard_errors=np.array(recorded["standard_errors"], dtype=float),
    )


def test_the_half_life_dial_is_never_recorded_in_a_bucket_fit(inner: dict) -> None:
    fits = inner["item_1_temporal"]["fits_on_all_training_seasons"]
    assert len(fits) == 12
    for bucket, fit in fits.items():
        assert fit["model"] == "A2_robust_student_t", bucket
        assert fit["half_life"] is None, bucket


def test_the_half_life_dial_leaves_the_a2_fit_bitwise_identical(inner: dict) -> None:
    """A2 never receives the dial, so changing it cannot move an output.

    ``temporal_fitter`` closes over ``half_life`` only on the A3 branch. The
    probe therefore asks the A2 fitter for the same series at six very
    different half-lives and compares the exact bit patterns of the posterior
    mean, the posterior SD, tau and every per-season weight. A3 at the carried
    value is fitted alongside, so the test also shows the dial is not simply
    inert everywhere.
    """
    module = _upstream_spec()
    series = _recorded_series(inner)

    def fingerprint(half_life: float, treatment: str) -> tuple[str, ...]:
        fitter = module.temporal_fitter(treatment, nu=3.0, half_life=half_life)
        fit = fitter(series)
        return (
            float(fit.posterior_mean).hex(),
            float(fit.posterior_sd).hex(),
            float(fit.predictive_sd).hex(),
            float(fit.tau).hex(),
            *(float(weight).hex() for weight in fit.weights),
        )

    a2 = {
        half_life: fingerprint(half_life, module.TEMPORAL_STUDENT_T)
        for half_life in (0.5, 1.0, 2.0, 4.0, 8.0, 1000.0)
    }
    assert len(set(a2.values())) == 1

    # The dial is live on the branch that declares it, which is why it was
    # carried at all; it is simply not on the selected branch.
    recency = module.temporal_fitter(
        module.TEMPORAL_RECENCY, nu=3.0, half_life=4.0
    )(series)
    plain = module.temporal_fitter(module.TEMPORAL_STUDENT_T, nu=3.0, half_life=4.0)(
        series
    )
    assert recency.half_life == 4.0
    assert plain.half_life is None
    assert recency.posterior_mean != plain.posterior_mean


def test_the_uniform_weighted_student_t_fit_is_what_the_artifact_recorded(
    inner: dict,
) -> None:
    """The selected treatment's fit reproduces from the library directly."""
    fit = fit_student_t_random_effects(_recorded_series(inner), nu=3.0)
    stored = inner["item_1_temporal"]["fits_on_all_training_seasons"][
        inner["item_1_temporal"]["primary_bucket"]
    ]
    assert fit.posterior_mean == pytest.approx(stored["posterior_mean"], abs=0.0)
    assert fit.half_life is None
    assert fit.model == stored["model"]


# ----------------------------------------------------------------------
# 9. original-versus-final gate mapping
# ----------------------------------------------------------------------


def test_every_implemented_gate_appears_in_the_mapping(
    audit: dict, gates: dict
) -> None:
    mapping = audit["original_versus_implemented_gates"]
    mapped = {
        entry["final_implemented_gate"]
        for entry in mapping
        if entry["final_implemented_gate"] != "NOT IMPLEMENTED"
    }
    for entry in gates["gates"]:
        assert f"gate {entry['gate']}: {entry['name']}" in mapped


def test_the_original_count_space_gate_is_reported_as_a_failure(
    audit: dict,
) -> None:
    """The brief asked for 20% in count space on one named bucket.

    No gate encoded it, and on its own terms it is not met. The audit must
    say so rather than substituting the no-regression gate that did get
    written, which is what this locks.
    """
    row = next(
        entry
        for entry in audit["original_versus_implemented_gates"]
        if entry["original_gate"].startswith("passer_ast_teammate_pts improves")
    )
    assert row["final_implemented_gate"] == "NOT IMPLEMENTED"
    assert row["would_the_original_gate_pass"] is False
    assert row["evidence"]["required_percentage_error_reduction"] == 20.0
    assert row["evidence"]["percentage_error_reduction"] < 20.0


def test_the_original_uncertainty_and_role_cell_gates_are_reported_as_failures(
    audit: dict,
) -> None:
    failing = set(audit["original_gates_that_would_fail"])
    assert len(failing) == 3
    assert any("uncertainty" in entry for entry in failing)
    assert any("role-pair cell" in entry for entry in failing)


def test_no_implemented_gate_is_claimed_to_satisfy_a_different_requirement(
    audit: dict,
) -> None:
    """A row that names an implemented gate must be the same requirement.

    The failure mode this guards is the one the brief warned about:
    re-pointing an original requirement at a gate that happens to pass. Every
    row either reports "none" as its difference or carries "NOT IMPLEMENTED".
    """
    for entry in audit["original_versus_implemented_gates"]:
        if entry["exact_difference"] == "none":
            assert entry["final_implemented_gate"] != "NOT IMPLEMENTED"
            assert entry["identical"] is True
        else:
            assert entry["final_implemented_gate"] == "NOT IMPLEMENTED"
            assert entry["identical"] is False


# ----------------------------------------------------------------------
# 10. stored-probability log-loss recomputation
# ----------------------------------------------------------------------


def test_log_loss_recomputes_from_the_stored_probabilities(
    audit: dict, grades: dict[str, pd.DataFrame]
) -> None:
    candidate = grades["candidate"]
    realized = candidate["realized"].to_numpy(float)
    probability = np.clip(candidate["p_candidate"].to_numpy(float), 1e-6, 1 - 1e-6)
    per_event = -(
        realized * np.log(probability) + (1.0 - realized) * np.log1p(-probability)
    )

    for legs, block in audit["proper_scores"]["by_legs"].items():
        mask = (candidate["n_legs"] == int(legs)).to_numpy()
        assert float(per_event[mask].mean()) == pytest.approx(
            block["log_loss_candidate"], abs=1e-12
        ), legs


def test_the_reported_log_loss_is_finite_and_the_clip_never_binds(
    audit: dict,
) -> None:
    """A clip that binds would make the reported log loss an artefact of it."""
    for block in audit["proper_scores"]["by_legs"].values():
        assert math.isfinite(block["log_loss_candidate"])
        assert math.isfinite(block["log_loss_control"])
        low, high = block["probability_range"]
        assert low > audit["proper_scores"]["clip"]
        assert high < 1.0 - audit["proper_scores"]["clip"]


def test_pushes_are_impossible_because_every_line_is_a_half_integer() -> None:
    """Counts are integers, so a half-integer line cannot be met exactly.

    The leg constructor places the line at a predictive quantile plus 0.5, and
    the audit's proper-score section rests on there being no push mass. A
    change to the line rule would silently introduce ties.
    """
    from nba_prop_quant.research.game_latent_state import validation

    source = Path(validation.__file__).read_text(encoding="utf-8")
    assert "analytic.quantile(target) + 0.5" in source


def test_the_dependence_temperature_was_priced_without_the_holdout() -> None:
    temperature = _load(ARTIFACT_DIR / "dependence_temperature.json")

    assert temperature["holdout_used_for_selection"] is False
    assert sorted(temperature["holdout_seasons_excluded"]) == list(HOLDOUT_SEASONS)
    assert all(
        season not in HOLDOUT_SEASONS
        for season in temperature["fold_target_seasons"]
    )
    assert float(temperature["selected_temperature"]) == 1.0


# ----------------------------------------------------------------------
# role-pair cells and the identification constraint
# ----------------------------------------------------------------------


def test_all_six_role_pair_cells_clear_both_support_thresholds(
    audit: dict,
) -> None:
    roles = audit["role_scale"]
    assert roles["cells_total"] == 6
    assert roles["cells_supported"] == 6
    assert roles["cells_shrunk_to_pooled"] == 0
    for label, entry in roles["by_cell"].items():
        assert entry["present"] is True, label
        assert entry["games"] >= MIN_CELL_GAMES, label
        assert entry["pairs"] >= MIN_CELL_PAIRS, label


def test_the_role_scale_weighted_mean_is_exactly_one(
    candidate_spec: dict,
) -> None:
    """The identification constraint, recomputed from the published numbers.

    If the player-share-weighted mean drifted off one the pooled same-team
    block would no longer be the object the unroled fit produced, and the
    role layer would be adding a global scale rather than redistributing one.
    """
    diagnostics = _load(ARTIFACT_DIR / "covariance_diagnostics.json")["role_scale"]
    scales = candidate_spec["loadings"]["role_scale"]
    shares = diagnostics["player_shares"]
    total = sum(shares.values())
    weighted = sum(shares[role] / total * scales[role] for role in scales)

    assert scales == diagnostics["scales"]
    assert weighted == pytest.approx(1.0, abs=1e-12)
    assert diagnostics["weighted_mean_scale_minus_one"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_no_role_is_fully_shrunk_towards_the_pooled_scale() -> None:
    """The log-scale shrinkage factor is reported, not assumed.

    ``tau^2 / (tau^2 + se^2)`` is what multiplies each role's log deviation.
    All three roles are measured well enough that the factor sits above 0.98,
    so the published scales are close to their raw values and the layer is not
    a no-op dressed as a fit.
    """
    diagnostics = _load(ARTIFACT_DIR / "covariance_diagnostics.json")["role_scale"]
    tau_squared = diagnostics["tau_log"] ** 2
    for role, log_se in diagnostics["log_standard_errors"].items():
        factor = tau_squared / (tau_squared + log_se**2)
        assert 0.98 < factor < 1.0, role


def test_the_worst_supported_cell_deterioration_is_reported_not_hidden(
    audit: dict,
) -> None:
    worst = audit["role_scale"]["worst_supported_cell_deterioration_rmse"]
    assert worst["cell"] == "starter+bench"
    assert worst["delta_rmse"] > 0.0

    cell = audit["role_scale"]["by_cell"]["starter+bench"]
    assert cell["candidate_rmse"] > cell["control_rmse"]
    assert cell["candidate_rms_z"] > cell["control_rms_z"]


# ----------------------------------------------------------------------
# the audit changed no model
# ----------------------------------------------------------------------


def test_the_audit_declares_itself_audit_only(audit: dict) -> None:
    assert audit["audit_only"] is True
    assert audit["model_changes_made"] == "NONE"
    assert audit["candidate_factor_spec_hash"] == CANDIDATE_SPEC_HASH
    assert (
        audit["promotion_eligibility"]
        == "SHADOW_RESEARCH_ONLY__NOT_ELIGIBLE_FOR_PROMOTION"
    )


def test_the_audit_did_not_move_the_frozen_candidate(frozen: dict) -> None:
    """The spec hash is a hash of the loadings, so this is a real lock."""
    spec = _load(ARTIFACT_DIR / "factor_spec.json")
    assert spec["spec_hash"] == frozen["factor_spec_hash"] == CANDIDATE_SPEC_HASH
    assert spec["remediation_spec"]["dependence_temperature"] == 1.0
    assert spec["remediation_spec"]["role_scale_mode"] == "log_shrunk"
    assert spec["remediation_spec"]["bridge_weight_cap"] == 0.15
    assert spec["remediation_spec"]["cross_team_prior"] == "gaussian"


def test_the_binomial_reading_of_the_coverage_shortfall_is_not_significant(
    inner: dict,
) -> None:
    """None of the four shortfalls clears two sigma even assuming independence.

    The twelve observations come from one season, so the independent binomial
    standard error is the optimistic case. Even there, every nominal level
    sits inside a 95% interval around the observed proportion, which is why
    the verdict is CAUTION rather than FAIL.
    """
    values = _scored_z(inner)["standardized"].to_numpy(float)
    observed = coverage_table(values, scale=1.0)
    trials = len(values)
    for level in COVERAGE_LEVELS:
        key = f"coverage_{int(round(level * 100))}"
        proportion = observed[key]
        standard_error = math.sqrt(level * (1.0 - level) / trials)
        z = (proportion - level) / standard_error
        assert abs(z) < stats.norm.ppf(0.975), key
