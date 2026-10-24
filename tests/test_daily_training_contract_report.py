"""Tests for the daily training contract report.

The contract is one relation spanning four lifecycle steps: is the authority
that priced tonight's slate trained through the most recent completed NBA
game? Before this report that relation was answerable only by reading four
step logs and a registry by hand, which is how it went unexamined.

The properties pinned here are the ones that make the report worth reading:

* the three data dates come from the rolling tree, and a completed game is
  the grader's ``is_final_game`` rather than a date comparison -- so tonight's
  scheduled games and a postponed game never count as completed history;
* the verdict is the stated inequality and nothing else, and an authority with
  no dated cutoff is UNMET rather than unknown;
* ``FALLBACK_ACTIVE`` is YES whenever the fit registered this morning is not
  the serving authority, and the reason distinguishes a failed validation from
  the frozen protocol's prohibition on automatic promotion;
* the record is durable and the consecutive-day count is derived from it, so
  "keep retrying" has evidence and a lost ledger under-reports rather than
  inventing a streak;
* it cannot gate. The default exit code is 0 on an unmet contract, because a
  red lifecycle would stop the slate being served.

Nothing here fits, prices, promotes, publishes or serves, and nothing reaches
the network.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]

SLATE = "2026-10-24"

#: A well-formed fit_id, in the shape the registry's own pattern requires.
TODAYS_FIT = "nba_prop_quant_fit_20261024_a80527f3de137567"
YESTERDAYS_FIT = "nba_prop_quant_fit_20261023_b1105f2ce4480921"


def load_ops(name: str):
    spec = importlib.util.spec_from_file_location(
        f"ops_{name}", PROJECT / "ops" / f"{name}.py"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


reporter = load_ops("report_daily_training_contract")

rules = reporter.load_settlement_rules()


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


def write_rolling_tree(
    root: Path,
    *,
    season: int = 2026,
    games: list[dict] | None = None,
    stats_dates: list[str] | None = None,
    advanced_dates: list[str] | None = None,
) -> None:
    """A season's rolling tree, in the layout the refresh writes it in."""
    season_dir = root / "raw" / "seasons" / f"season={season}"
    season_dir.mkdir(parents=True, exist_ok=True)

    if games is None:
        games = [
            {"id": 1, "date": "2026-10-22", "status": "Final"},
            {"id": 2, "date": "2026-10-23", "status": "Final"},
        ]

    pd.DataFrame(games).to_parquet(season_dir / "games.parquet", index=False)

    pd.DataFrame(
        {
            "date": stats_dates if stats_dates is not None else ["2026-10-23"],
            "game_id": [1] * len(stats_dates if stats_dates is not None else [1]),
        }
    ).to_parquet(season_dir / "stats.parquet", index=False)

    advanced_dir = root / "raw" / "advanced" / f"season={season}"
    advanced_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "date": (
                advanced_dates if advanced_dates is not None else ["2026-10-23"]
            ),
            "game_id": [
                1
            ]
            * len(advanced_dates if advanced_dates is not None else [1]),
        }
    ).to_parquet(advanced_dir / "advanced.parquet", index=False)


class FakeRegistry:
    """Exactly the four reads the report makes of the registry.

    A real ``FitRegistry`` would need a staged artifact tree, an architecture
    contract and a registration per fit, none of which this report looks at.
    What it looks at is the promotion state, one manifest and one validation
    record, so that is what is stood up.
    """

    def __init__(
        self,
        *,
        manifests: dict[str, dict] | None = None,
        validations: dict[str, dict] | None = None,
        current_good: str | None = None,
        promoted_at: str | None = None,
    ) -> None:
        self._manifests = dict(manifests or {})
        self._validations = dict(validations or {})
        self._current_good = current_good
        self._promoted_at = promoted_at

    def list_fits(self) -> list[str]:
        return sorted(self._manifests)

    def load_manifest(self, fit_id: str) -> dict:
        if fit_id not in self._manifests:
            raise RuntimeError(f"fit {fit_id} has no manifest")

        return self._manifests[fit_id]

    def load_validation(self, fit_id: str) -> dict | None:
        return self._validations.get(fit_id)

    def current(self) -> dict:
        return {
            "current_good_fit_id": self._current_good,
            "previous_good_fit_id": None,
            "promoted_at": self._promoted_at,
            "promotion_actor": None,
            "promotion_reason": None,
        }


def passing_validation(fit_id: str) -> dict:
    return {
        "fit_id": fit_id,
        "passed": True,
        "failed_checks": [],
        "missing_checks": [],
        "recorded_at": "2026-10-24T09:50:00+00:00",
    }


def failing_validation(fit_id: str, *, failed: list[str]) -> dict:
    return {
        "fit_id": fit_id,
        "passed": False,
        "failed_checks": failed,
        "missing_checks": [],
        "recorded_at": "2026-10-24T09:50:00+00:00",
    }


def adaptive_status(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def report_for(
    *,
    data_root: Path,
    registry: FakeRegistry,
    adaptive: Path | None = None,
    serving_receipt: Path | None = None,
    slate_date: str = SLATE,
) -> dict:
    return reporter.build_report(
        slate_date=slate_date,
        data_root=data_root,
        registry=registry,
        adaptive_status=adaptive,
        serving_receipt=serving_receipt,
        production_sha="a88a0fe45ed1ed4c838f709e39fa4a49b4961c11",
        rules=rules,
        now=datetime(2026, 10, 24, 9, 55, tzinfo=UTC),
    )


# ----------------------------------------------------------------------
# the three data dates
# ----------------------------------------------------------------------


def test_the_three_data_dates_come_from_the_rolling_tree(tmp_path: Path) -> None:
    write_rolling_tree(
        tmp_path,
        stats_dates=["2026-10-21", "2026-10-23"],
        advanced_dates=["2026-10-22", "2026-10-23"],
    )

    report = report_for(data_root=tmp_path, registry=FakeRegistry())

    assert report["latest_completed_game_date"] == "2026-10-23"
    assert report["latest_stats_date"] == "2026-10-23"
    assert report["latest_advanced_date"] == "2026-10-23"


def test_a_scheduled_game_is_not_a_completed_game(tmp_path: Path) -> None:
    """Tonight's games sit in the table before they are played.

    Dating the latest completed game by the newest row in the schedule would
    report tonight as already complete, which would make the contract look met
    on exactly the mornings it is not.
    """
    write_rolling_tree(
        tmp_path,
        games=[
            {"id": 1, "date": "2026-10-23", "status": "Final"},
            {"id": 2, "date": "2026-10-24", "status": "Scheduled"},
        ],
    )

    report = report_for(data_root=tmp_path, registry=FakeRegistry())

    assert report["latest_completed_game_date"] == "2026-10-23"


def test_a_postponed_game_never_becomes_completed_history(tmp_path: Path) -> None:
    write_rolling_tree(
        tmp_path,
        games=[
            {"id": 1, "date": "2026-10-22", "status": "Final", "postponed": False},
            {"id": 2, "date": "2026-10-23", "status": "Final", "postponed": True},
        ],
    )

    report = report_for(data_root=tmp_path, registry=FakeRegistry())

    assert report["latest_completed_game_date"] == "2026-10-22"


def test_completion_is_the_graders_rule_and_not_a_second_copy_of_it() -> None:
    """The report may not develop its own opinion about a final game."""
    source = (PROJECT / "ops" / "report_daily_training_contract.py").read_text(
        encoding="utf-8"
    )

    assert "def is_final_game" not in source
    assert "def target_season" not in source
    assert "rules.is_final_game" in source
    assert "rules.target_season" in source


def test_an_empty_season_reports_no_dates_rather_than_guessing(
    tmp_path: Path,
) -> None:
    report = report_for(data_root=tmp_path, registry=FakeRegistry())

    assert report["latest_completed_game_date"] is None
    assert report["latest_stats_date"] is None
    assert report["latest_advanced_date"] is None


def test_an_unreadable_game_source_refuses_the_whole_report(
    tmp_path: Path,
) -> None:
    """A field that cannot be reported truthfully stops all of them.

    A report that silently dropped one field would read as a met contract with
    a blank line in it, which is the shape of answer this exists to replace.
    """
    season_dir = tmp_path / "raw" / "seasons" / "season=2026"
    season_dir.mkdir(parents=True)
    (season_dir / "games.parquet").write_bytes(b"not a parquet file")

    with pytest.raises(reporter.ContractUnreportable) as refusal:
        report_for(data_root=tmp_path, registry=FakeRegistry())

    assert "latest completed game date" in str(refusal.value)


# ----------------------------------------------------------------------
# the daily fit
# ----------------------------------------------------------------------


def test_the_daily_fit_is_read_from_the_protocols_own_status(
    tmp_path: Path,
) -> None:
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"}
        },
        validations={TODAYS_FIT: passing_validation(TODAYS_FIT)},
    )

    report = report_for(
        data_root=tmp_path,
        registry=registry,
        adaptive=adaptive_status(
            tmp_path,
            {
                "outcome": "COMPLETED",
                "mode": "register-candidate",
                "fit_id": TODAYS_FIT,
                "promoted": False,
            },
        ),
    )

    assert report["daily_fit_id"] == TODAYS_FIT
    assert report["daily_fit_training_cutoff"] == "2026-10-23"
    assert report["daily_fit_promoted"] is False
    assert report["daily_fit_source"] == "adaptive_status"
    assert "COMPLETED" in report["daily_fit_status"]
    assert "validation PASS" in report["daily_fit_status"]


def test_a_failed_validation_is_named_in_the_fit_status(tmp_path: Path) -> None:
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"}
        },
        validations={
            TODAYS_FIT: failing_validation(
                TODAYS_FIT, failed=["calibration_valid"]
            )
        },
    )

    report = report_for(
        data_root=tmp_path,
        registry=registry,
        adaptive=adaptive_status(
            tmp_path,
            {"outcome": "COMPLETED", "fit_id": TODAYS_FIT, "promoted": False},
        ),
    )

    assert "validation FAIL" in report["daily_fit_status"]
    assert "calibration_valid" in report["daily_fit_status"]


def test_the_registry_answers_when_the_lifecycle_status_is_absent(
    tmp_path: Path,
) -> None:
    """A report run outside the lifecycle still has to name today's fit."""
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            YESTERDAYS_FIT: {
                "fit_date": "2026-10-23",
                "training_cutoff": "2026-10-22",
            },
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"},
        },
        validations={TODAYS_FIT: passing_validation(TODAYS_FIT)},
    )

    report = report_for(data_root=tmp_path, registry=registry)

    assert report["daily_fit_source"] == "registry"
    assert report["daily_fit_id"] == TODAYS_FIT
    assert report["daily_fit_training_cutoff"] == "2026-10-23"


def test_a_day_with_no_fit_reports_no_fit(tmp_path: Path) -> None:
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            YESTERDAYS_FIT: {
                "fit_date": "2026-10-23",
                "training_cutoff": "2026-10-22",
            }
        }
    )

    report = report_for(data_root=tmp_path, registry=registry)

    assert report["daily_fit_id"] is None
    assert report["daily_fit_status"] == "NO_FIT"


# ----------------------------------------------------------------------
# the serving authority
# ----------------------------------------------------------------------


def test_the_serving_authority_is_the_promotion_state(tmp_path: Path) -> None:
    """Not the newest fit, not the candidate registered this morning."""
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            YESTERDAYS_FIT: {
                "fit_date": "2026-10-23",
                "training_cutoff": "2026-10-22",
            },
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"},
        },
        validations={TODAYS_FIT: passing_validation(TODAYS_FIT)},
        current_good=YESTERDAYS_FIT,
        promoted_at="2026-10-23T10:00:00+00:00",
    )

    report = report_for(data_root=tmp_path, registry=registry)

    assert report["current_serving_fit_id"] == YESTERDAYS_FIT
    assert report["current_serving_training_cutoff"] == "2026-10-22"
    assert report["current_serving_promoted_at"] == "2026-10-23T10:00:00+00:00"


def test_an_unpromoted_registry_reports_the_frozen_bundle(tmp_path: Path) -> None:
    """The state production is in before the first reviewed promotion.

    The frozen deployment bundle is pinned to a training season rather than to
    a date, so the cutoff is reported as undated rather than invented.
    """
    write_rolling_tree(tmp_path)

    receipt = tmp_path / "incumbent_serving.json"
    receipt.write_text(
        json.dumps(
            {
                "incumbent_authority": "frozen_deployment_bundle",
                "incumbent_fit_id": None,
                "incumbent_version": "nba_prop_quant_20260818T205213Z",
                "outcome": "NO_GAMES_ON_SLATE",
            }
        ),
        encoding="utf-8",
    )

    report = report_for(
        data_root=tmp_path, registry=FakeRegistry(), serving_receipt=receipt
    )

    assert report["current_serving_fit_id"] is None
    assert report["current_serving_training_cutoff"] == reporter.CUTOFF_UNDATED
    assert (
        report["current_serving_frozen_bundle_version"]
        == "nba_prop_quant_20260818T205213Z"
    )
    assert "no adaptive fit has been promoted" in report["current_serving_detail"]


# ----------------------------------------------------------------------
# the verdict
# ----------------------------------------------------------------------


def test_the_contract_is_met_when_the_serving_cutoff_reaches_the_last_game(
    tmp_path: Path,
) -> None:
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"}
        },
        validations={TODAYS_FIT: passing_validation(TODAYS_FIT)},
        current_good=TODAYS_FIT,
    )

    report = report_for(data_root=tmp_path, registry=registry)

    assert report["verdict"] == reporter.VERDICT_MET
    assert report["fallback_active"] is False
    assert report["fallback_reason"] == reporter.FALLBACK_NONE


def test_a_serving_cutoff_behind_the_last_completed_game_is_unmet(
    tmp_path: Path,
) -> None:
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            YESTERDAYS_FIT: {
                "fit_date": "2026-10-23",
                "training_cutoff": "2026-10-21",
            },
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"},
        },
        validations={TODAYS_FIT: passing_validation(TODAYS_FIT)},
        current_good=YESTERDAYS_FIT,
    )

    report = report_for(data_root=tmp_path, registry=registry)

    assert report["verdict"] == reporter.VERDICT_UNMET
    assert "2026-10-21" in report["verdict_detail"]
    assert "2026-10-23" in report["verdict_detail"]


def test_an_undated_cutoff_is_unmet_rather_than_unknown(tmp_path: Path) -> None:
    """An authority that cannot name a cutoff date demonstrably is not current."""
    write_rolling_tree(tmp_path)

    report = report_for(data_root=tmp_path, registry=FakeRegistry())

    assert report["verdict"] == reporter.VERDICT_UNMET
    assert "no dated training cutoff" in report["verdict_detail"]


def test_a_season_with_no_completed_game_cannot_be_behind_it(
    tmp_path: Path,
) -> None:
    """Opening day, before any game has been played."""
    write_rolling_tree(
        tmp_path,
        games=[{"id": 1, "date": "2026-10-24", "status": "Scheduled"}],
    )

    report = report_for(data_root=tmp_path, registry=FakeRegistry())

    assert report["verdict"] == reporter.VERDICT_MET
    assert "no game in this season is confirmed complete" in report[
        "verdict_detail"
    ]


# ----------------------------------------------------------------------
# the fallback
# ----------------------------------------------------------------------


def test_a_registered_but_unpromoted_fit_is_the_protocols_fallback(
    tmp_path: Path,
) -> None:
    """Validation passed and the fit still is not serving.

    That is not a fault. Automatic promotion is prohibited by the frozen
    update protocol, and the reason has to say so rather than implying the fit
    broke.
    """
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"}
        },
        validations={TODAYS_FIT: passing_validation(TODAYS_FIT)},
        current_good=YESTERDAYS_FIT,
    )
    registry._manifests[YESTERDAYS_FIT] = {
        "fit_date": "2026-10-23",
        "training_cutoff": "2026-10-22",
    }

    report = report_for(data_root=tmp_path, registry=registry)

    assert report["fallback_active"] is True
    assert report["fallback_reason"] == reporter.FALLBACK_DAILY_FIT_NOT_PROMOTED
    assert "Automatic promotion is prohibited" in report["fallback_detail"]


def test_a_fit_that_failed_validation_is_the_controlled_fallback(
    tmp_path: Path,
) -> None:
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"},
            YESTERDAYS_FIT: {
                "fit_date": "2026-10-23",
                "training_cutoff": "2026-10-22",
            },
        },
        validations={
            TODAYS_FIT: failing_validation(TODAYS_FIT, failed=["marginal_fit_valid"])
        },
        current_good=YESTERDAYS_FIT,
    )

    report = report_for(data_root=tmp_path, registry=registry)

    assert report["fallback_active"] is True
    assert (
        report["fallback_reason"] == reporter.FALLBACK_DAILY_FIT_FAILED_VALIDATION
    )
    assert "most recently validated authority" in report["fallback_detail"]
    assert "retried by the next scheduled lifecycle" in report["fallback_detail"]


def test_no_fit_at_all_is_distinguished_from_a_fit_that_failed(
    tmp_path: Path,
) -> None:
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            YESTERDAYS_FIT: {
                "fit_date": "2026-10-23",
                "training_cutoff": "2026-10-22",
            }
        },
        current_good=YESTERDAYS_FIT,
    )

    report = report_for(data_root=tmp_path, registry=registry)

    assert report["fallback_reason"] == reporter.FALLBACK_NO_DAILY_FIT


# ----------------------------------------------------------------------
# the durable record
# ----------------------------------------------------------------------


def test_the_record_and_the_ledger_both_survive_the_run(tmp_path: Path) -> None:
    write_rolling_tree(tmp_path)

    code = reporter.main(
        [
            "--slate-date",
            SLATE,
            "--data-root",
            str(tmp_path),
            "--registry-root",
            str(tmp_path / "registry"),
        ]
    )

    assert code == reporter.EXIT_OK

    root = tmp_path / reporter.CONTRACT_RELATIVE

    record = json.loads((root / f"{SLATE}.json").read_text(encoding="utf-8"))

    assert record["slate_date"] == SLATE
    assert record["verdict"] == reporter.VERDICT_UNMET
    assert record["fallback_active"] is True

    ledger = [
        json.loads(line)
        for line in (root / reporter.CONTRACT_LEDGER_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert [row["slate_date"] for row in ledger] == [SLATE]
    assert ledger[0]["fallback_active"] is True


def test_consecutive_fallback_days_counts_the_run_of_unmet_days(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / reporter.CONTRACT_LEDGER_NAME
    ledger.parent.mkdir(parents=True, exist_ok=True)

    ledger.write_text(
        "\n".join(
            json.dumps({"slate_date": date, "fallback_active": True})
            for date in ("2026-10-21", "2026-10-22", "2026-10-23")
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        reporter.consecutive_fallback_days(
            ledger, slate_date="2026-10-24", active=True
        )
        == 4
    )


def test_a_met_day_breaks_the_streak(tmp_path: Path) -> None:
    ledger = tmp_path / reporter.CONTRACT_LEDGER_NAME
    ledger.parent.mkdir(parents=True, exist_ok=True)

    ledger.write_text(
        "\n".join(
            json.dumps({"slate_date": date, "fallback_active": active})
            for date, active in (
                ("2026-10-21", True),
                ("2026-10-22", False),
                ("2026-10-23", True),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        reporter.consecutive_fallback_days(
            ledger, slate_date="2026-10-24", active=True
        )
        == 2
    )


def test_a_met_contract_reports_no_streak(tmp_path: Path) -> None:
    ledger = tmp_path / reporter.CONTRACT_LEDGER_NAME

    assert (
        reporter.consecutive_fallback_days(
            ledger, slate_date=SLATE, active=False
        )
        == 0
    )


def test_a_truncated_ledger_under_reports_rather_than_inventing_a_streak(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / reporter.CONTRACT_LEDGER_NAME
    ledger.parent.mkdir(parents=True, exist_ok=True)

    ledger.write_text(
        json.dumps({"slate_date": "2026-10-21", "fallback_active": True})
        + "\n"
        + '{"slate_date": "2026-10-2',
        encoding="utf-8",
    )

    assert (
        reporter.consecutive_fallback_days(
            ledger, slate_date="2026-10-24", active=True
        )
        == 1
    )


def test_a_record_that_cannot_be_written_refuses_rather_than_reporting(
    tmp_path: Path,
) -> None:
    """An unmet contract that leaves no trace is an unmet contract nobody sees."""
    write_rolling_tree(tmp_path)

    blocked = tmp_path / reporter.CONTRACT_RELATIVE
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("a regular file where the record tree belongs\n")

    code = reporter.main(
        [
            "--slate-date",
            SLATE,
            "--data-root",
            str(tmp_path),
            "--registry-root",
            str(tmp_path / "registry"),
        ]
    )

    assert code == reporter.EXIT_USAGE


# ----------------------------------------------------------------------
# reporting, and the refusal to gate
# ----------------------------------------------------------------------


def test_an_unmet_contract_does_not_fail_the_lifecycle_by_default(
    tmp_path: Path,
) -> None:
    """The slate must still be served by a slightly older validated authority."""
    write_rolling_tree(tmp_path)

    assert (
        reporter.main(
            [
                "--slate-date",
                SLATE,
                "--data-root",
                str(tmp_path),
                "--registry-root",
                str(tmp_path / "registry"),
            ]
        )
        == reporter.EXIT_OK
    )


def test_strict_turns_the_verdict_into_an_exit_code(tmp_path: Path) -> None:
    """For an acceptance run, which wants the contract as a pass or a fail."""
    write_rolling_tree(tmp_path)

    assert (
        reporter.main(
            [
                "--slate-date",
                SLATE,
                "--data-root",
                str(tmp_path),
                "--registry-root",
                str(tmp_path / "registry"),
                "--strict",
            ]
        )
        == reporter.EXIT_CONTRACT_UNMET
    )


def test_an_unmet_contract_raises_a_workflow_annotation(tmp_path: Path) -> None:
    write_rolling_tree(tmp_path)

    report = report_for(data_root=tmp_path, registry=FakeRegistry())
    report["fallback_consecutive_days"] = 3

    note = reporter.annotation(report)

    assert note is not None
    assert note.startswith("::error title=Daily training contract UNMET::")
    assert "FALLBACK_ACTIVE=YES" in note
    assert "consecutive days 3" in note


def test_a_met_contract_raises_no_annotation(tmp_path: Path) -> None:
    write_rolling_tree(tmp_path)

    registry = FakeRegistry(
        manifests={
            TODAYS_FIT: {"fit_date": SLATE, "training_cutoff": "2026-10-23"}
        },
        validations={TODAYS_FIT: passing_validation(TODAYS_FIT)},
        current_good=TODAYS_FIT,
    )

    assert reporter.annotation(report_for(data_root=tmp_path, registry=registry)) is None


def test_every_field_the_contract_is_stated_in_is_rendered(tmp_path: Path) -> None:
    write_rolling_tree(tmp_path)

    report = report_for(data_root=tmp_path, registry=FakeRegistry())
    report["fallback_consecutive_days"] = 1

    rendered = reporter.render(report)

    for name in reporter.REPORT_FIELDS:
        assert f"`{name}`" in rendered

    assert "FALLBACK_ACTIVE" in rendered
    assert "CURRENT_SERVING_TRAINING_CUTOFF >= LATEST_COMPLETED_GAME_DATE" in rendered


def test_the_report_names_all_nine_contract_fields() -> None:
    assert reporter.REPORT_FIELDS == (
        "LATEST_COMPLETED_GAME_DATE",
        "LATEST_STATS_DATE",
        "LATEST_ADVANCED_DATE",
        "DAILY_FIT_ID",
        "DAILY_FIT_TRAINING_CUTOFF",
        "DAILY_FIT_STATUS",
        "DAILY_FIT_PROMOTED",
        "CURRENT_SERVING_FIT_ID",
        "CURRENT_SERVING_TRAINING_CUTOFF",
    )


def test_the_report_cannot_fit_promote_price_publish_or_serve() -> None:
    """Structural, not a convention.

    Read from the syntax tree rather than from the text, so the prose is free
    to name the serving entry point it mirrors while the code remains unable
    to reach it. A mutating registry call or an import of a fitting, pricing,
    serving or publishing module would be visible here.
    """
    tree = ast.parse(
        (PROJECT / "ops" / "report_daily_training_contract.py").read_text(
            encoding="utf-8"
        )
    )

    mutating = {"promote", "register", "record_validation", "rollback", "initialise"}

    forbidden_modules = {
        "run_incumbent_production_serving",
        "run_advisory_window_serving",
        "run_wizardofodds_publication",
        "run_adaptive_daily_fit",
        "run_t20_production_controller",
        "nba_prop_quant.adaptive_training",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in mutating:
            raise AssertionError(f"the report calls registry.{node.attr}")

        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules

        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module not in forbidden_modules
