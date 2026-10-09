"""Run the same-game dependence shadow beside the production pricing path.

SHADOW ONLY. The incumbent remains the served and published authority. This
entry point records what the candidate would have said; it prices nothing,
publishes nothing and promotes nothing.

What it reads
-------------

The artifacts the daily fit just produced: ``models/marginals.joblib`` and
``models/copula.joblib`` from the candidate tree, which are the same objects
``scripts/15_price_markets.py`` prices with and
``scripts/10_predict_slate.py`` predicts with, plus that fit's own
``processed/oof_selected_means.parquet`` as the slate frame. Not a copy and
not a refit: the same files, hashed into every row's provenance. Pass
``--adaptive-status`` and all three are resolved from the workspace the
adaptive daily fit recorded, so the shadow cannot drift onto a different fit
than the one production registered.

What it writes
--------------

An append-only JSONL log of candidate / incumbent / independence probabilities
with provenance, PSD diagnostics, same-player block deviation and a fallback
reason where one applies; a grading report over the rows whose outcome is
known; and a status JSON for the run summary. It writes nothing else and
mutates no production state.

Why it cannot break production
------------------------------

The lifecycle calls this after the model work is done, and a shadow that can
fail the production job is worse than no shadow. So every failure is caught,
recorded in the status file and reported as exit 0 -- unless ``--strict`` is
passed, which is for humans debugging the shadow, never for the lifecycle.

That is the opposite of how the shadow treats *serving*: the served
probability is the incumbent's on every row, and an incumbent failure
propagates rather than being papered over. Fail-open for the lifecycle,
fail-closed for the number.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Exit codes. The lifecycle only ever sees 0; ``--strict`` is for humans.
EXIT_OK = 0
EXIT_SHADOW_FAILED = 1

#: The production objects the live pricing path uses. Named here so a reader
#: can see that the shadow consumes production's own marginals rather than a
#: parallel fit of its own.
PRODUCTION_MARGINALS_NAME = "marginals.joblib"
PRODUCTION_COPULA_NAME = "copula.joblib"

#: The frame the daily fit's marginal and dependence stages are themselves fit
#: from. It carries ``mu_selected_{stat}``, the ZINB inflation features, the
#: roster identifiers and ``expected_minutes``, which is exactly the slate
#: shape the shadow needs.
PRODUCTION_SLATE_NAME = "oof_selected_means.parquet"

#: Hard-coded because it is the thing the deployment is pinned to. A mismatch
#: is a refusal, not a warning.
FINAL_MODEL_SPEC_RELATIVE = "research/final_model/final_model_spec.json"
FROZEN_FACTOR_SPEC_RELATIVE = "research/final_upstream_remediation/factor_spec.json"

#: The carried driver whose game preparation this reuses. Loaded rather than
#: copied: a second definition here would be free to drift from the one that
#: was graded, and nobody would notice.
SHADOW_DRIVER_RELATIVE = "research/game_latent_state/shadow/01_run_production_shadow.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_production_shadow",
        description=(
            "Shadow the production pricing path. The incumbent remains the "
            "served and published authority; this publishes nothing."
        ),
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument(
        "--adaptive-status",
        type=Path,
        default=None,
        help="the JSON ops/run_adaptive_daily_fit.py emitted. Its recorded "
        "workspace is where the marginals, the copula and the slate frame "
        "are read from, so the shadow runs on the fit production registered.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="override the directory holding marginals.joblib and "
        "copula.joblib. Resolved from --adaptive-status when omitted.",
    )
    parser.add_argument(
        "--slate",
        type=Path,
        default=None,
        help="slate frame (parquet or csv). Resolved from --adaptive-status "
        "when omitted. With neither, the run is declare-only: it proves the "
        "pin and the refusals without pricing anything.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="frame the roster rows are taken from. Defaults to the slate, "
        "which is the live case: the slate is the roster.",
    )
    parser.add_argument(
        "--factor-spec", type=Path, default=PROJECT_ROOT / FROZEN_FACTOR_SPEC_RELATIVE
    )
    parser.add_argument(
        "--final-model-spec",
        type=Path,
        default=PROJECT_ROOT / FINAL_MODEL_SPEC_RELATIVE,
    )
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--grading-path", type=Path, default=None)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--events-per-family", type=int, default=2)
    parser.add_argument("--min-expected-minutes", type=float, default=12.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero when the shadow fails. Never set by the lifecycle.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_frame(path: Path):
    import pandas as pd

    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def resolve_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Find the fit the shadow is supposed to read, and say where from.

    Explicit flags win, because a human debugging the shadow needs to be able
    to point it at one specific tree. Otherwise the adaptive status file
    decides, which is the production case: the shadow reads the workspace of
    the fit that just ran rather than whatever happens to be on disk.
    """
    resolution: dict[str, Any] = {
        "model_dir": args.model_dir,
        "slate": args.slate,
        "history": args.history,
        "model_dir_source": "--model-dir" if args.model_dir else None,
        "slate_source": "--slate" if args.slate else None,
    }

    if args.adaptive_status is not None and args.adaptive_status.exists():
        status = json.loads(args.adaptive_status.read_text(encoding="utf-8"))
        resolution["adaptive_outcome"] = status.get("outcome")
        resolution["adaptive_fit_id"] = status.get("fit_id")
        recorded = status.get("workspace")
        if recorded:
            workspace = Path(recorded)
            resolution["adaptive_workspace"] = str(workspace)
            candidate_models = workspace / "candidate" / "models"
            if resolution["model_dir"] is None and candidate_models.is_dir():
                resolution["model_dir"] = candidate_models
                resolution["model_dir_source"] = "--adaptive-status workspace"
            slate = workspace / "processed" / PRODUCTION_SLATE_NAME
            if resolution["slate"] is None and slate.exists():
                resolution["slate"] = slate
                resolution["slate_source"] = "--adaptive-status workspace"

    if resolution["model_dir"] is None:
        resolution["model_dir"] = PROJECT_ROOT / "models"
        resolution["model_dir_source"] = "repository models/ directory"

    if resolution["history"] is None:
        resolution["history"] = resolution["slate"]

    return resolution


def check_the_model_on_disk_is_the_frozen_one(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Refuse unless the specification on disk is the frozen one.

    A shadow running a model nobody froze produces a log nobody can interpret,
    so the hashes are compared before any probability is computed.
    """
    from nba_prop_quant.research.game_latent_state.artifacts import sha256_file

    spec = json.loads(args.factor_spec.read_text(encoding="utf-8"))
    final = json.loads(args.final_model_spec.read_text(encoding="utf-8"))
    pinned = {
        "factor_spec_path": str(args.factor_spec),
        "factor_spec_hash": spec.get("spec_hash"),
        "factor_spec_hash_expected": final.get("factor_spec_hash"),
        "final_model_spec_sha256": sha256_file(args.final_model_spec),
        "final_model_status": final.get("status"),
    }
    pinned["factor_spec_matches_the_final_model_specification"] = (
        pinned["factor_spec_hash"] == pinned["factor_spec_hash_expected"]
    )
    pinned["final_model_is_frozen"] = final.get("status") == "FROZEN"
    pinned["passed"] = bool(
        pinned["factor_spec_matches_the_final_model_specification"]
        and pinned["final_model_is_frozen"]
    )
    return pinned


def check_the_shadow_has_no_authority() -> dict[str, Any]:
    """Prove, in the deployed process, that neither path is reachable.

    Recorded rather than assumed: the status file carries the refusal, so an
    operator reading a run can see that the deployed code refused rather than
    trusting that it would have.
    """
    from nba_prop_quant.research.game_latent_state import shadow_runtime as sr

    record: dict[str, Any] = {
        "promotion_authority": sr.PROMOTION_AUTHORITY,
        "published_authority": "incumbent",
    }

    try:
        sr.assert_no_promotion_authority("ops/run_production_shadow.py")
        record["promotion_attempt"] = "DID NOT RAISE"
    except sr.ShadowPromotionRefused as exc:
        record["promotion_attempt"] = type(exc).__name__
        record["promotion_refusal"] = str(exc)

    switch = sr.read_publishing_switch(PROJECT_ROOT / sr.PUBLISHING_SWITCH_PATH)
    record["switch"] = switch.payload()
    try:
        sr.publish_shadow_probabilities([], switch)
        record["publish_attempt"] = "DID NOT RAISE"
    except sr.ShadowPublishingDisabled as exc:
        record["publish_attempt"] = type(exc).__name__
        record["publish_refusal"] = str(exc)

    record["rows_published"] = 0
    record["passed"] = (
        record["promotion_attempt"] != "DID NOT RAISE"
        and record["publish_attempt"] != "DID NOT RAISE"
    )
    return record


def shadow_driver():
    """The carried shadow driver, imported for its game preparation.

    Loaded the way the adaptive daily fit loads the certified pipeline
    scripts: one definition of how a game's roster is assembled from a slate.
    """
    from nba_prop_quant.adaptive_training import load_script_module

    return load_script_module(PROJECT_ROOT, SHADOW_DRIVER_RELATIVE)


def load_production_dependence(model_dir: Path):
    """The marginals and copula the live pricing path prices with."""
    import joblib

    marginal_path = model_dir / PRODUCTION_MARGINALS_NAME
    copula_path = model_dir / PRODUCTION_COPULA_NAME
    missing = [str(p) for p in (marginal_path, copula_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} missing, so the shadow cannot read the "
            "objects production prices with. Nothing was shadowed, and the "
            "incumbent is unaffected."
        )
    return (
        marginal_path,
        joblib.load(marginal_path),
        copula_path,
        joblib.load(copula_path),
    )


def attach_realized_outcomes(frame, stats) -> bool:
    """Alias the box-score columns to the ``y_{stat}`` names grading expects.

    An event is only gradeable once the game it refers to has settled, and
    the event generator reads the realized value under ``y_{stat}``. The
    lifecycle runs after every North American box score has landed, so the
    most recent completed slate has them; tonight's unplayed slate does not,
    and this reports that rather than inventing a number to stand in.
    """
    for stat in stats:
        realized = f"y_{stat}"
        if realized not in frame.columns:
            if stat not in frame.columns:
                return False
            frame[realized] = frame[stat]
        if frame[realized].isna().all():
            return False
    return True


def select_slate(frame, slate_date: str, stats):
    """Narrow a fit's frame to the slate the shadow should price.

    The frame the daily fit writes is its whole training corpus, not one
    night. Pricing all of it would be a backtest rather than a shadow, so the
    slate date is selected explicitly.

    The lifecycle runs in the morning, so the named date's games have usually
    not been played. The most recent date whose outcomes have settled is used
    instead, and which date that was is reported, because a shadow row is only
    worth logging if it can eventually be graded.
    """
    import pandas as pd

    if "date" not in frame.columns:
        return frame, {"selected_date": None, "reason": "the frame carries no date"}

    dates = pd.to_datetime(frame["date"])
    wanted = pd.Timestamp(slate_date)
    exact = frame.loc[dates == wanted]
    if len(exact) and attach_realized_outcomes(exact.copy(), stats):
        return exact.copy(), {
            "selected_date": slate_date,
            "reason": "the named slate has settled outcomes",
        }

    settled = frame.loc[dates < wanted]
    if not len(settled):
        return exact.copy(), {
            "selected_date": slate_date if len(exact) else None,
            "reason": "no earlier date is present in this frame",
        }

    for candidate in sorted(pd.to_datetime(settled["date"]).unique(), reverse=True):
        block = settled.loc[pd.to_datetime(settled["date"]) == candidate].copy()
        if attach_realized_outcomes(block, stats):
            return block, {
                "selected_date": str(pd.Timestamp(candidate).date()),
                "reason": (
                    f"the {slate_date} slate has no settled outcomes yet, so the "
                    "most recent settled date was shadowed instead"
                ),
            }

    return exact.copy(), {
        "selected_date": None,
        "reason": "no date in this frame has settled outcomes",
    }


def is_a_psd_refusal(reason: str | None) -> bool:
    """Whether a fallback came from the covariance refusing to assemble.

    Named rather than inlined because the distinction matters to an operator:
    a PSD refusal is the shared-structure contract biting, which is a
    different thing from a bug in the query engine.
    """
    if not reason:
        return False
    return "cannot absorb any shared structure" in reason or "positive" in reason


def run_shadow(
    args: argparse.Namespace,
    resolution: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Price one slate under both models. Returns the per-run summary."""

    from nba_prop_quant.research.game_latent_state.artifacts import git_sha, sha256_file
    from nba_prop_quant.research.game_latent_state.covariance import (
        PSD_EIGENVALUE_FLOOR,
        SharedFactorLoadings,
    )
    from nba_prop_quant.research.game_latent_state.factors import (
        incumbent_within_player_blocks,
    )
    from nba_prop_quant.research.game_latent_state.shadow_runtime import (
        ShadowConfig,
        build_provenance,
        evaluate_shadow_game,
        grade_shadow_log,
        live_dependence_spaces,
        observed_pair_moments,
        shadow_log_frame,
    )
    from nba_prop_quant.research.game_latent_state.simulator import SUPPORTED_STATS
    from nba_prop_quant.research.game_latent_state.validation import (
        analytic_marginals,
        generate_joint_events,
    )

    model_dir = Path(resolution["model_dir"])
    marginal_path, marginals, copula_path, copula = load_production_dependence(
        model_dir
    )

    spec = json.loads(args.factor_spec.read_text(encoding="utf-8"))
    loadings = SharedFactorLoadings.from_payload(spec["loadings"])
    provenance = build_provenance(
        loadings=loadings,
        factor_spec=spec,
        factor_spec_path=args.factor_spec,
        marginal_source=(
            f"{marginal_path} (the object scripts/15_price_markets.py prices with)"
        ),
        copula_source=f"{copula_path} (the incumbent)",
        simulations=int(args.simulations),
        seed=int(args.seed),
        final_model_spec_path=args.final_model_spec,
        marginal_source_path=marginal_path,
        copula_source_path=copula_path,
        code_sha=git_sha(PROJECT_ROOT),
        built_at=utc_now(),
    )

    corpus = read_frame(Path(resolution["slate"]))
    slate, selection = select_slate(corpus, args.slate_date, SUPPORTED_STATS)
    history = (
        slate
        if Path(resolution["history"]) == Path(resolution["slate"])
        else read_frame(Path(resolution["history"]))
    )
    settled = attach_realized_outcomes(slate, SUPPORTED_STATS)
    driver = shadow_driver()
    config = ShadowConfig(simulations=int(args.simulations), seed=int(args.seed))

    game_ids = sorted(int(value) for value in slate["game_id"].unique())
    if args.max_games is not None:
        game_ids = game_ids[: args.max_games]
    if not settled:
        game_ids = []

    results = []
    realized: dict[str, int] = {}
    fallbacks: list[dict[str, Any]] = []
    skipped = 0

    for game_id in game_ids:
        prepared = driver._prepare_game(
            game_id, slate, history, float(args.min_expected_minutes)
        )
        if prepared is None:
            skipped += 1
            continue
        roster, observations = prepared

        reference = analytic_marginals(roster, marginals)
        joint_events = generate_joint_events(
            observations,
            reference,
            SUPPORTED_STATS,
            game_id=game_id,
            seed=int(args.seed) + game_id,
            events_per_family=int(args.events_per_family),
            min_expected_minutes=float(args.min_expected_minutes),
        )
        if not joint_events:
            skipped += 1
            continue

        events = {}
        for index, event in enumerate(joint_events):
            event_id = f"{args.slate_date}-{game_id}-{event.family}-{index}"
            events[event_id] = event.legs
            if getattr(event, "realized", None) is not None:
                realized[event_id] = int(event.realized)

        within = incumbent_within_player_blocks(
            copula, SUPPORTED_STATS, roster.frame["player_id"].astype(int)
        )
        # No try/except around the incumbent arm: evaluate_shadow_game lets an
        # incumbent failure propagate on purpose, and catching it here would
        # undo exactly the thing that makes the served number trustworthy. A
        # candidate failure never reaches this level; it is already a recorded
        # fallback on every row of the game.
        result = evaluate_shadow_game(
            roster,
            events=events,
            marginals=marginals,
            copula=copula,
            loadings=loadings,
            provenance=provenance,
            config=config,
            within_player=within,
        )
        # The observed side of the dependence comparison. The simulated arms
        # say what each model claims; without the realized reading there is no
        # target for the frozen policy's two dependence RMSEs to be an error
        # against. Recorded per game so the monitoring state can pool it by
        # pair count, the way the held-out accumulator did.
        if result.dependence and settled:
            try:
                observed = observed_pair_moments(
                    observations,
                    SUPPORTED_STATS,
                    reference,
                    seed=int(args.seed),
                )
                result.dependence.update(
                    live_dependence_spaces(result.dependence, observed)
                )
            except Exception as error:  # noqa: BLE001
                # A dependence reading is a diagnostic. Failing to take one
                # must not cost the run its shadow rows, so it is recorded and
                # the game continues.
                result.dependence["observed_reading_error"] = (
                    f"{type(error).__name__}: {error}"
                )

        results.append(result)
        if result.fell_back:
            fallbacks.append(
                {
                    "game_id": game_id,
                    "reason": result.failure_reason,
                    "was_a_psd_refusal": is_a_psd_refusal(result.failure_reason),
                }
            )

    log = shadow_log_frame(results, provenance)
    grading = grade_shadow_log(log, realized) if len(log) else {}

    rows_written = 0
    if args.log_path is not None and len(log):
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("a", encoding="utf-8") as handle:
            for record in log.to_dict(orient="records"):
                record["slate_date"] = args.slate_date
                record["realized"] = realized.get(str(record.get("event_id")))
                handle.write(json.dumps(record, default=str) + "\n")
                rows_written += 1

    numerical = [result.numerical for result in results if result.numerical]
    eigenvalues = [
        float(d["min_eigenvalue"]) for d in numerical if "min_eigenvalue" in d
    ]
    deviations = [
        float(d["same_player_max_block_deviation"])
        for d in numerical
        if "same_player_max_block_deviation" in d
    ]
    summary: dict[str, Any] = {
        "slate_date": args.slate_date,
        "slate_source": str(resolution["slate"]),
        "slate_resolved_from": resolution.get("slate_source"),
        "model_dir": str(model_dir),
        "model_dir_resolved_from": resolution.get("model_dir_source"),
        "marginal_source": str(marginal_path),
        "marginal_source_sha256": sha256_file(marginal_path),
        "copula_source": str(copula_path),
        "copula_source_sha256": sha256_file(copula_path),
        "provenance_fingerprint": provenance.fingerprint,
        "provenance": provenance.payload(),
        "slate_selection": selection,
        "slate_has_settled_outcomes": settled,
        "games_offered": len(game_ids),
        "games_shadowed": len(results),
        "games_skipped_for_too_few_players_or_events": skipped,
        "events_shadowed": len(log),
        "games_that_fell_back": len(fallbacks),
        "fallbacks": fallbacks,
        "psd_failures": sum(1 for f in fallbacks if f["was_a_psd_refusal"])
        + sum(1 for value in eigenvalues if value < PSD_EIGENVALUE_FLOOR),
        "psd_eigenvalue_floor": PSD_EIGENVALUE_FLOOR,
        "min_covariance_eigenvalue": min(eigenvalues) if eigenvalues else None,
        "same_player_max_block_deviation": max(deviations) if deviations else None,
        "dependence_diagnostics_recorded": sum(
            1 for result in results if result.dependence
        ),
        "dependence_buckets": [
            result.dependence for result in results if result.dependence
        ],
        "log_path": str(args.log_path) if args.log_path else None,
        "rows_written": rows_written,
        "rows_published": authority["rows_published"],
        "grading": grading,
    }

    if args.grading_path is not None:
        args.grading_path.parent.mkdir(parents=True, exist_ok=True)
        args.grading_path.write_text(
            json.dumps(
                {
                    "slate_date": args.slate_date,
                    "served_model": "incumbent",
                    "published": False,
                    "provenance_fingerprint": provenance.fingerprint,
                    "grading": grading,
                    "numerical_diagnostics": numerical,
                    "dependence_diagnostics": summary["dependence_buckets"],
                    "fallbacks": fallbacks,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

    # Carried in the status file but not in the grading report: the per-game
    # bucket payloads are large and the report is read by people.
    summary.pop("dependence_buckets")
    return summary


def write_summary(path: Path, status: dict[str, Any]) -> None:
    shadow = status.get("shadow") or {}
    authority = status.get("authority") or {}
    lines = [
        "## Same-game dependence shadow",
        "",
        f"- outcome: `{status['outcome']}`",
        f"- served authority: `{authority.get('published_authority', 'incumbent')}`",
        f"- promotion authority: `{authority.get('promotion_authority', 'unknown')}`",
        f"- publish attempt: `{authority.get('publish_attempt', 'not attempted')}`",
        f"- rows published: `{authority.get('rows_published', 0)}`",
    ]
    if shadow:
        lines += [
            f"- games shadowed: `{shadow.get('games_shadowed')}`",
            f"- events shadowed: `{shadow.get('events_shadowed')}`",
            f"- fallbacks: `{shadow.get('games_that_fell_back')}`",
            f"- PSD failures: `{shadow.get('psd_failures')}`",
        ]
    if status.get("error"):
        lines.append(f"- shadow error (production unaffected): `{status['error']}`")
    lines += ["", "The incumbent remains the served and published authority."]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    status: dict[str, Any] = {
        "entry_point": "ops/run_production_shadow.py",
        "mode": "SHADOW_ONLY",
        "started_at": utc_now(),
        "slate_date": args.slate_date,
    }

    try:
        status["pinned"] = check_the_model_on_disk_is_the_frozen_one(args)
        status["authority"] = check_the_shadow_has_no_authority()
        if not status["pinned"]["passed"]:
            raise RuntimeError(
                "the specification on disk is not the frozen final model: "
                f"{status['pinned']}"
            )
        if not status["authority"]["passed"]:
            raise RuntimeError(
                "the deployed shadow did not refuse a promotion or a publish "
                "attempt, so it is not safe to run"
            )
        resolution = resolve_inputs(args)
        status["inputs"] = {k: str(v) for k, v in resolution.items() if v is not None}
        if resolution["slate"] is None:
            status["outcome"] = "DECLARE_ONLY_NO_SLATE_RESOLVED"
        else:
            status["shadow"] = run_shadow(args, resolution, status["authority"])
            if not status["shadow"]["slate_has_settled_outcomes"]:
                # The slate's games have not been played, so there is nothing
                # to grade against. Recorded as its own outcome rather than
                # as a failure: an unsettled slate is a normal state, and the
                # lifecycle runs after the previous night's scores land.
                status["outcome"] = "SLATE_NOT_SETTLED_NOTHING_TO_GRADE"
            else:
                status["outcome"] = (
                    "SHADOWED"
                    if status["shadow"]["games_that_fell_back"] == 0
                    else "SHADOWED_WITH_FALLBACKS"
                )
    except Exception as exc:  # noqa: BLE001 - fail open for the lifecycle
        status["outcome"] = "SHADOW_FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()

    status["finished_at"] = utc_now()
    status["served_authority"] = "incumbent"
    status["rows_published"] = 0
    status["production_serving_was_affected"] = False

    if args.status_path is not None:
        args.status_path.parent.mkdir(parents=True, exist_ok=True)
        args.status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if args.summary_path is not None:
        write_summary(args.summary_path, status)

    print(json.dumps(status, indent=2, sort_keys=True, default=str))

    if status["outcome"] == "SHADOW_FAILED" and args.strict:
        return EXIT_SHADOW_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
