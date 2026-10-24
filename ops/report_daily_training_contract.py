#!/usr/bin/env python3
"""Report the daily training contract, and say out loud when it is not met.

WHAT THIS IS FOR
----------------

The lifecycle refreshes the rolling state, fits a candidate, registers it,
serves the slate with the incumbent and grades yesterday. Every one of those
steps reports what *it* did. Nothing reported the one relation that spans
them, which is the only question an operator actually needs answered each
morning:

    is the authority that priced tonight's slate trained through the most
    recent completed NBA game?

That relation was answerable only by reading four step logs and a registry by
hand. This reports it as nine named fields and one verdict, from durable
state, every production day.

REPORTED, NEVER GATED
---------------------

Nothing here predicts, prices, fits, validates, promotes, publishes or
touches serving authority. It reads and it writes its own record. By default
it exits 0 whatever the verdict, because the alternative -- a red lifecycle
on a day the contract is unmet -- would stop the slate being served, and a
slate served by a slightly older validated authority is strictly better than
no slate at all. ``--strict`` is for an acceptance run that wants the verdict
as an exit code.

WHY IT IS STILL NOT SILENT
--------------------------

"Do not gate" and "do not mention" are different things. An unmet contract
here writes ``FALLBACK_ACTIVE = YES`` into a durable per-slate record, counts
how many consecutive days it has been unmet in a durable ledger, prints the
reason, and -- when handed ``--github-annotation`` -- raises a workflow
annotation that appears on the run without failing it. The daily cron is the
retry; the consecutive-day count is the evidence that retrying is happening
and the evidence of how long it has not worked.

WHERE THE FIELDS COME FROM
--------------------------

The three data dates are derived from the durable rolling tree, which is the
same tree the fit trained from and the grader settles against. The fit fields
are read from the adaptive protocol's own status document when the lifecycle
ran it, and from the registry otherwise. The serving fields are read from the
registry's promotion state, which is what ``run_incumbent_production_serving``
resolves the incumbent from.

Whether a game is complete and which season a date belongs to are not decided
here. Both are imported from the grader that already owns them, through the
grader's own loader, so this cannot develop an opinion the settlement rules
do not share.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

#: The incumbent grader. Imported for its settlement-rule loader rather than
#: for its grading: it already resolves which module owns ``is_final_game``
#: and ``target_season`` and checks that the module still provides them, and a
#: second resolution of either would be a second settlement rule.
GRADER_MODULE_PATH = PROJECT_ROOT / "ops" / "grade_incumbent_production_slate.py"

#: The durable rolling tree, relative to the data root.
SEASONS_RELATIVE = Path("raw") / "seasons"
ADVANCED_RELATIVE = Path("raw") / "advanced"

#: The column every rolling dataset dates its rows by.
ROLLING_DATE_COLUMN = "date"

#: Where this record lives. Under ``processed`` because it is derived, and in
#: its own tree because no serving script may read it.
CONTRACT_RELATIVE = Path("processed") / "daily_training_contract"
CONTRACT_LEDGER_NAME = "contract_ledger.jsonl"

#: The verdict, as the two words the acceptance report asks for.
VERDICT_MET = "MET"
VERDICT_UNMET = "UNMET"

#: Why today's fit is not the serving authority, when it is not. Distinguished
#: because they call for different operator responses: a failed validation is
#: a fault to investigate, a non-promotion is the frozen protocol working as
#: written, and an absent fit is normally an off day.
FALLBACK_NONE = "NONE"
FALLBACK_DAILY_FIT_FAILED_VALIDATION = "DAILY_FIT_FAILED_VALIDATION"
FALLBACK_DAILY_FIT_NOT_PROMOTED = "DAILY_FIT_NOT_PROMOTED"
FALLBACK_NO_DAILY_FIT = "NO_DAILY_FIT"

#: Reported when the serving authority carries no dated training cutoff at
#: all, which is the case while the frozen deployment bundle is the incumbent:
#: it is pinned to a training *season*, not to a date.
CUTOFF_UNDATED = "UNDATED"

EXIT_OK = 0
EXIT_CONTRACT_UNMET = 1
EXIT_USAGE = 2


class ContractUnreportable(RuntimeError):
    """A field cannot be reported truthfully, so none of them are reported."""


# ----------------------------------------------------------------------
# shared rules
# ----------------------------------------------------------------------


def load_settlement_rules() -> Any:
    """The settlement vocabulary, through the grader's own loader."""
    spec = importlib.util.spec_from_file_location(
        "daily_training_contract_grader", GRADER_MODULE_PATH
    )

    if spec is None or spec.loader is None:
        raise ContractUnreportable(
            f"cannot load the grader from {GRADER_MODULE_PATH}, so the "
            "settlement rules it owns cannot be shared with this report"
        )

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001
        raise ContractUnreportable(
            f"cannot load the grader from {GRADER_MODULE_PATH}: "
            f"{type(error).__name__}: {error}"
        ) from error

    loader = getattr(module, "load_settlement_rules", None)

    if loader is None:
        raise ContractUnreportable(
            f"{GRADER_MODULE_PATH} no longer exposes load_settlement_rules, "
            "so this report has no way to reach the settlement rules without "
            "restating them"
        )

    return loader()


# ----------------------------------------------------------------------
# the three data dates
# ----------------------------------------------------------------------


def _latest_date(path: Path, label: str) -> str | None:
    """The newest date in one rolling dataset, or None when it has none."""
    if not path.is_file():
        return None

    try:
        frame = pd.read_parquet(path, columns=[ROLLING_DATE_COLUMN])
    except Exception as error:  # noqa: BLE001
        raise ContractUnreportable(
            f"the {label} source at {path} could not be read "
            f"({type(error).__name__}: {error}), so the latest {label} date "
            "cannot be reported"
        ) from error

    dates = pd.to_datetime(frame[ROLLING_DATE_COLUMN], errors="coerce").dropna()

    if dates.empty:
        return None

    return pd.Timestamp(dates.max()).normalize().strftime("%Y-%m-%d")


def latest_completed_game_date(path: Path, rules: Any) -> str | None:
    """The newest date on which a game is confirmed complete.

    Completion is the grader's ``is_final_game``, not a date comparison: a
    scheduled game sitting in the table with tonight's date is not a completed
    game, and a postponed one never becomes one.
    """
    if not path.is_file():
        return None

    try:
        games = pd.read_parquet(path)
    except Exception as error:  # noqa: BLE001
        raise ContractUnreportable(
            f"the game-status source at {path} could not be read "
            f"({type(error).__name__}: {error}), so no game can be confirmed "
            "complete and the latest completed game date cannot be reported"
        ) from error

    if ROLLING_DATE_COLUMN not in games.columns:
        raise ContractUnreportable(
            f"the game-status source at {path} carries no "
            f"{ROLLING_DATE_COLUMN} column, so a completed game cannot be "
            "dated"
        )

    dates = pd.to_datetime(games[ROLLING_DATE_COLUMN], errors="coerce")

    latest: pd.Timestamp | None = None

    for (_, row), date in zip(games.iterrows(), dates, strict=True):
        if pd.isna(date) or not bool(rules.is_final_game(row)):
            continue

        normalized = pd.Timestamp(date).normalize()

        if latest is None or normalized > latest:
            latest = normalized

    return None if latest is None else latest.strftime("%Y-%m-%d")


def data_dates(*, data_root: Path, slate_date: str, rules: Any) -> dict[str, Any]:
    season = int(rules.target_season(pd.Timestamp(slate_date)))
    season_dir = Path(data_root) / SEASONS_RELATIVE / f"season={season}"
    advanced = (
        Path(data_root) / ADVANCED_RELATIVE / f"season={season}" / "advanced.parquet"
    )

    return {
        "season": season,
        "latest_completed_game_date": latest_completed_game_date(
            season_dir / "games.parquet", rules
        ),
        "latest_stats_date": _latest_date(season_dir / "stats.parquet", "stats"),
        "latest_advanced_date": _latest_date(advanced, "advanced"),
    }


# ----------------------------------------------------------------------
# the daily fit
# ----------------------------------------------------------------------


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        raise ContractUnreportable(
            f"the {label} at {path} could not be read "
            f"({type(error).__name__}: {error})"
        ) from error


def daily_fit(
    *,
    adaptive_status: Path | None,
    registry: Any,
    slate_date: str,
) -> dict[str, Any]:
    """What this morning's adaptive protocol produced, if anything.

    The adaptive protocol's own status document is preferred, because it
    records the outcome of the run rather than an inference about it. The
    registry is the fallback for a report run outside the lifecycle.
    """
    if adaptive_status is not None and adaptive_status.is_file():
        status = _load_json(adaptive_status, "adaptive protocol status")

        if not isinstance(status, dict):
            raise ContractUnreportable(
                f"the adaptive protocol status at {adaptive_status} is not a "
                "JSON object, so the daily fit cannot be reported"
            )

        fit_id = status.get("fit_id")

        return {
            "source": "adaptive_status",
            "fit_id": fit_id,
            "outcome": status.get("outcome"),
            "mode": status.get("mode"),
            "training_cutoff": _fit_training_cutoff(registry, fit_id)
            or status.get("training_cutoff"),
            "validation": _fit_validation(registry, fit_id),
            "promoted": bool(status.get("promoted")),
        }

    fit_id = _newest_fit_registered_on(registry, slate_date)

    return {
        "source": "registry",
        "fit_id": fit_id,
        "outcome": None if fit_id is None else "REGISTERED",
        "mode": None,
        "training_cutoff": _fit_training_cutoff(registry, fit_id),
        "validation": _fit_validation(registry, fit_id),
        "promoted": False,
    }


def _manifest(registry: Any, fit_id: str | None) -> dict[str, Any] | None:
    if registry is None or not fit_id:
        return None

    try:
        manifest = registry.load_manifest(fit_id)
    except Exception:  # noqa: BLE001
        # A fit the registry cannot describe is reported as undescribed. It is
        # not this report's place to decide that means the registry is broken.
        return None

    return manifest if isinstance(manifest, dict) else None


def _fit_training_cutoff(registry: Any, fit_id: str | None) -> str | None:
    manifest = _manifest(registry, fit_id)

    if manifest is None:
        return None

    cutoff = manifest.get("training_cutoff")

    return None if cutoff is None else str(cutoff)


def _fit_validation(registry: Any, fit_id: str | None) -> dict[str, Any] | None:
    if registry is None or not fit_id:
        return None

    try:
        record = registry.load_validation(fit_id)
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(record, dict):
        return None

    return {
        "passed": bool(record.get("passed")),
        "failed_checks": sorted(str(name) for name in record.get("failed_checks") or []),
        "missing_checks": sorted(
            str(name) for name in record.get("missing_checks") or []
        ),
        "recorded_at": record.get("recorded_at"),
    }


def _newest_fit_registered_on(registry: Any, slate_date: str) -> str | None:
    """The newest fit whose own manifest dates it to this slate date."""
    if registry is None:
        return None

    try:
        fit_ids = list(registry.list_fits())
    except Exception:  # noqa: BLE001
        return None

    for fit_id in reversed(sorted(fit_ids)):
        manifest = _manifest(registry, fit_id)

        if manifest is not None and str(manifest.get("fit_date")) == slate_date:
            return fit_id

    return None


# ----------------------------------------------------------------------
# the serving authority
# ----------------------------------------------------------------------


def serving_authority(
    *,
    registry: Any,
    serving_receipt: Path | None,
) -> dict[str, Any]:
    """The fit the incumbent actually prices with, and its training cutoff.

    Read from the registry's promotion state, which is what the serving entry
    point resolves the incumbent from. When no adaptive fit has ever been
    promoted the authority is the frozen deployment bundle, which is pinned to
    a training season rather than to a date, and that is reported as an
    undated cutoff rather than guessed at.
    """
    current: dict[str, Any] = {}

    if registry is not None:
        try:
            current = registry.current() or {}
        except Exception:  # noqa: BLE001
            current = {}

    fit_id = current.get("current_good_fit_id")

    authority = {
        "fit_id": fit_id,
        "promoted_at": current.get("promoted_at"),
        "previous_good_fit_id": current.get("previous_good_fit_id"),
        "training_cutoff": _fit_training_cutoff(registry, fit_id),
        "frozen_bundle_version": None,
        "detail": None,
    }

    if fit_id:
        if authority["training_cutoff"] is None:
            authority["training_cutoff"] = CUTOFF_UNDATED
            authority["detail"] = (
                f"the promoted fit {fit_id} has no readable manifest, so its "
                "training cutoff cannot be reported"
            )

        return authority

    authority["training_cutoff"] = CUTOFF_UNDATED
    authority["detail"] = (
        "no adaptive fit has been promoted, so the frozen deployment bundle "
        "is the serving authority; it is pinned to a training season rather "
        "than to a date and therefore carries no dated training cutoff"
    )

    if serving_receipt is not None and serving_receipt.is_file():
        receipt = _load_json(serving_receipt, "serving receipt")

        # The serving receipt is flat and carries the authority identity the
        # brief asks for under one key: the promoted fit_id when one is in
        # force and the bundle's freeze_id when none is. Reaching for it here
        # names the bundle rather than leaving the field null.
        if isinstance(receipt, dict):
            authority["frozen_bundle_version"] = receipt.get("incumbent_version")

    return authority


# ----------------------------------------------------------------------
# the verdict
# ----------------------------------------------------------------------


def evaluate(
    *,
    dates: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Compare the serving cutoff against the latest completed game date.

    An undated cutoff is UNMET, not unknown. The question is whether the
    authority pricing tonight is trained through the last completed game, and
    an authority that cannot name a cutoff date demonstrably is not.
    """
    completed = dates["latest_completed_game_date"]
    cutoff = authority["training_cutoff"]

    if completed is None:
        return {
            "verdict": VERDICT_MET,
            "detail": (
                "no game in this season is confirmed complete yet, so there "
                "is no completed game the serving authority could be behind"
            ),
        }

    if cutoff == CUTOFF_UNDATED or cutoff is None:
        return {
            "verdict": VERDICT_UNMET,
            "detail": (
                "the serving authority carries no dated training cutoff, so "
                f"it cannot be shown to train through {completed}"
            ),
        }

    if str(cutoff) >= str(completed):
        return {
            "verdict": VERDICT_MET,
            "detail": (
                f"the serving authority trains through {cutoff}, at or past "
                f"the latest completed game date {completed}"
            ),
        }

    return {
        "verdict": VERDICT_UNMET,
        "detail": (
            f"the serving authority trains through {cutoff}, behind the "
            f"latest completed game date {completed}"
        ),
    }


def fallback_reason(*, fit: dict[str, Any], authority: dict[str, Any]) -> str:
    """Why the slate is not being priced by the fit trained this morning."""
    fit_id = fit.get("fit_id")

    if fit_id and authority.get("fit_id") == fit_id:
        return FALLBACK_NONE

    if not fit_id:
        return FALLBACK_NO_DAILY_FIT

    validation = fit.get("validation")

    if isinstance(validation, dict) and not validation.get("passed"):
        return FALLBACK_DAILY_FIT_FAILED_VALIDATION

    return FALLBACK_DAILY_FIT_NOT_PROMOTED


FALLBACK_DETAIL = {
    FALLBACK_NONE: (
        "the fit registered this morning is the serving authority, so no "
        "fallback is in use"
    ),
    FALLBACK_NO_DAILY_FIT: (
        "no fit was registered for this slate date, so the most recently "
        "validated authority continues to serve with current-day features"
    ),
    FALLBACK_DAILY_FIT_FAILED_VALIDATION: (
        "this morning's fit did not pass validation, so the most recently "
        "validated authority serves with current-day features instead. This "
        "is the controlled fallback, and it is retried by the next scheduled "
        "lifecycle"
    ),
    FALLBACK_DAILY_FIT_NOT_PROMOTED: (
        "this morning's fit was registered but is not the serving authority. "
        "Automatic promotion is prohibited by the frozen update protocol, so "
        "the most recently validated authority serves with current-day "
        "features until a reviewed promotion"
    ),
}


# ----------------------------------------------------------------------
# the durable record
# ----------------------------------------------------------------------


def consecutive_fallback_days(ledger: Path, *, slate_date: str, active: bool) -> int:
    """How many days running the fallback has been active, including today.

    Counted from the ledger rather than from a stored counter, so a lost or
    truncated ledger under-reports rather than inventing a streak, and a day
    recorded twice does not advance it twice.
    """
    if not active:
        return 0

    seen: dict[str, bool] = {}

    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            date = row.get("slate_date")

            if isinstance(date, str):
                seen[date] = bool(row.get("fallback_active"))

    seen[slate_date] = True

    streak = 0
    date = pd.Timestamp(slate_date).normalize()

    while True:
        key = date.strftime("%Y-%m-%d")

        if not seen.get(key, False):
            return streak

        streak += 1
        date = date - pd.Timedelta(days=1)


def write_record(
    report: dict[str, Any],
    *,
    contract_root: Path,
    slate_date: str,
) -> dict[str, str]:
    """Persist the record durably before anything is reported about it."""
    record_path = contract_root / f"{slate_date}.json"
    ledger_path = contract_root / CONTRACT_LEDGER_NAME

    try:
        contract_root.mkdir(parents=True, exist_ok=True)

        temporary = record_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, record_path)

        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "slate_date": slate_date,
                        "recorded_at_utc": report["recorded_at_utc"],
                        "verdict": report["verdict"],
                        "fallback_active": report["fallback_active"],
                        "fallback_reason": report["fallback_reason"],
                        "daily_fit_id": report["daily_fit_id"],
                        "current_serving_fit_id": report["current_serving_fit_id"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    except OSError as error:
        raise ContractUnreportable(
            f"the daily training contract for {slate_date} could not be "
            f"recorded under {contract_root} ({error}), so an unmet contract "
            "would leave no durable trace and the consecutive-day count would "
            "silently restart"
        ) from error

    return {"record_path": str(record_path), "ledger_path": str(ledger_path)}


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

#: The nine fields the contract is stated in, in the order it states them.
REPORT_FIELDS = (
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


def named_fields(report: dict[str, Any]) -> list[tuple[str, str]]:
    def shown(value: Any) -> str:
        if value is None:
            return "null"

        if isinstance(value, bool):
            return "YES" if value else "NO"

        return str(value)

    return [
        (name, shown(report[name.lower()]))
        for name in REPORT_FIELDS
    ]


def build_report(
    *,
    slate_date: str,
    data_root: Path,
    registry: Any,
    adaptive_status: Path | None,
    serving_receipt: Path | None,
    production_sha: str | None,
    rules: Any,
    now: datetime | None = None,
) -> dict[str, Any]:
    dates = data_dates(data_root=data_root, slate_date=slate_date, rules=rules)
    fit = daily_fit(
        adaptive_status=adaptive_status, registry=registry, slate_date=slate_date
    )
    authority = serving_authority(registry=registry, serving_receipt=serving_receipt)
    outcome = evaluate(dates=dates, authority=authority)
    reason = fallback_reason(fit=fit, authority=authority)

    validation = fit.get("validation")

    if validation is None:
        fit_status = str(fit.get("outcome") or "NO_FIT")
    elif validation.get("passed"):
        fit_status = f"{fit.get('outcome') or 'REGISTERED'} (validation PASS)"
    else:
        failed = validation.get("failed_checks") or validation.get("missing_checks")
        fit_status = (
            f"{fit.get('outcome') or 'REGISTERED'} (validation FAIL: "
            f"{', '.join(failed) if failed else 'unspecified'})"
        )

    return {
        "schema_version": 1,
        "slate_date": slate_date,
        "season": dates["season"],
        "recorded_at_utc": (now or datetime.now(UTC)).isoformat(),
        "production_sha": production_sha,
        "data_root": str(data_root),
        "latest_completed_game_date": dates["latest_completed_game_date"],
        "latest_stats_date": dates["latest_stats_date"],
        "latest_advanced_date": dates["latest_advanced_date"],
        "daily_fit_id": fit.get("fit_id"),
        "daily_fit_training_cutoff": fit.get("training_cutoff"),
        "daily_fit_status": fit_status,
        "daily_fit_promoted": bool(fit.get("promoted")),
        "daily_fit_source": fit.get("source"),
        "daily_fit_validation": validation,
        "current_serving_fit_id": authority.get("fit_id"),
        "current_serving_training_cutoff": authority.get("training_cutoff"),
        "current_serving_promoted_at": authority.get("promoted_at"),
        "current_serving_frozen_bundle_version": authority.get(
            "frozen_bundle_version"
        ),
        "current_serving_detail": authority.get("detail"),
        "verdict": outcome["verdict"],
        "verdict_detail": outcome["detail"],
        "fallback_active": reason != FALLBACK_NONE,
        "fallback_reason": reason,
        "fallback_detail": FALLBACK_DETAIL[reason],
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "## Daily training contract",
        "",
        f"Slate {report['slate_date']}, season {report['season']}.",
        "",
        "| field | value |",
        "| --- | --- |",
    ]

    for name, value in named_fields(report):
        lines.append(f"| `{name}` | {value} |")

    lines += [
        f"| `CONTRACT` | **{report['verdict']}** |",
        f"| `FALLBACK_ACTIVE` | "
        f"{'YES' if report['fallback_active'] else 'NO'} |",
        f"| `FALLBACK_REASON` | {report['fallback_reason']} |",
        f"| `FALLBACK_CONSECUTIVE_DAYS` | "
        f"{report.get('fallback_consecutive_days', 0)} |",
        "",
        f"`CURRENT_SERVING_TRAINING_CUTOFF >= LATEST_COMPLETED_GAME_DATE`: "
        f"**{report['verdict']}** — {report['verdict_detail']}.",
        "",
        report["fallback_detail"] + ".",
        "",
    ]

    return "\n".join(lines) + "\n"


def annotation(report: dict[str, Any]) -> str | None:
    """A workflow annotation for an unmet contract, or None when it is met."""
    if report["verdict"] == VERDICT_MET:
        return None

    days = report.get("fallback_consecutive_days", 0)

    return (
        "::error title=Daily training contract UNMET::"
        f"slate {report['slate_date']}: {report['verdict_detail']}. "
        f"FALLBACK_ACTIVE=YES, reason {report['fallback_reason']}, "
        f"consecutive days {days}. "
        f"{report['fallback_detail']}."
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="report_daily_training_contract",
        description=(
            "Report the daily training contract from durable state. This "
            "tool does not fit, promote, price, publish or serve."
        ),
    )

    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--adaptive-status", type=Path, default=None)
    parser.add_argument("--serving-receipt", type=Path, default=None)
    parser.add_argument("--production-sha", default=None)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument(
        "--github-annotation",
        action="store_true",
        help=(
            "Raise a workflow annotation when the contract is unmet. The "
            "annotation does not fail the job; it makes an unmet contract "
            "visible on the run."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit nonzero when the contract is unmet. Off by default: a red "
            "lifecycle would stop the slate being served, which is worse "
            "than serving it with a slightly older validated authority."
        ),
    )

    return parser.parse_args(argv)


def open_registry(
    registry_root: Path | None,
    *,
    project_root: Path | None = None,
) -> Any:
    """The fit registry, resolved the way the serving entry point resolves it."""
    from nba_prop_quant.adaptive_fit_registry import (  # noqa: PLC0415
        FitRegistry,
        resolve_registry_root,
    )

    root = PROJECT_ROOT if project_root is None else Path(project_root)

    try:
        return FitRegistry(
            root=resolve_registry_root(registry_root, project_root=root),
            project_root=root,
        )

    except Exception as error:  # noqa: BLE001
        raise ContractUnreportable(
            "the fit registry root could not be resolved "
            f"({type(error).__name__}: {error}), so neither the daily fit nor "
            "the serving authority can be reported"
        ) from error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        rules = load_settlement_rules()
        registry = open_registry(args.registry_root)

        report = build_report(
            slate_date=args.slate_date,
            data_root=args.data_root,
            registry=registry,
            adaptive_status=args.adaptive_status,
            serving_receipt=args.serving_receipt,
            production_sha=args.production_sha,
            rules=rules,
        )

        contract_root = Path(args.data_root) / CONTRACT_RELATIVE

        report["fallback_consecutive_days"] = consecutive_fallback_days(
            contract_root / CONTRACT_LEDGER_NAME,
            slate_date=args.slate_date,
            active=report["fallback_active"],
        )

        report.update(
            write_record(
                report, contract_root=contract_root, slate_date=args.slate_date
            )
        )

    except ContractUnreportable as error:
        print(f"DAILY TRAINING CONTRACT UNREPORTABLE: {error}", file=sys.stderr)
        return EXIT_USAGE

    rendered = render(report)
    print(rendered, end="")

    if args.status_path:
        args.status_path.parent.mkdir(parents=True, exist_ok=True)
        args.status_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(rendered)

    if args.github_annotation:
        note = annotation(report)

        if note is not None:
            print(note)

    if args.strict and report["verdict"] != VERDICT_MET:
        print(report["verdict_detail"], file=sys.stderr)
        return EXIT_CONTRACT_UNMET

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
