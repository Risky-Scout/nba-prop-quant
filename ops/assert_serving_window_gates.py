#!/usr/bin/env python3
"""Hold the serving window to the two checks a morning fit cannot answer.

WHY THIS EXISTS
---------------

``REQUIRED_VALIDATION_CHECKS`` has fifteen names. Thirteen of them judge the
fitted model and are answerable from the candidate tree the moment it is
built. Two do not judge the model at all:

    gate3_role_readiness
        needs a live captured lineup snapshot for the slate.

    t20_protocol_compatible
        needs a real capture at the frozen serving offset for the slate.

Both were promotion prerequisites, and both ask about a capture that is taken
in the evening about a fit produced that morning. So promotion required
information that does not exist when promotion has to happen, and the daily
fit was never promotable -- which is why production served an August freeze
while a validated fit sat in the registry every day.

Moving them off the promotion prerequisite would weaken the system if that
were all that happened. It is not. They are enforced here instead, and here
is strictly more checking than there:

    there
        answered once, from whatever capture happened to exist when the
        validation record was written, and then never asked again. A fit
        promoted on Monday carried Monday's answer into every slate after it.

    here
        re-evaluated at the window that is about to publish, against that
        window's own capture, every time. Yesterday's answer cannot satisfy
        today's window, and a recorded ``true`` is not read as evidence --
        the check is recomputed from the capture on disk.

TWO GATES, TWO FAILURE DOMAINS
------------------------------

They are not refused the same way, because they are not about the same thing.

``t20_protocol_compatible`` is about the capture every route reads. The
pricing path resolves one verified bundle per game through
``select_capture_bundle`` and prices nothing without it, so a capture that is
absent or will not verify is an anomaly in the single input the whole slate
depends on. The caller reaches this script only after it has captured and
verified this window's group, which makes a failure here a contradiction of
the thing that got us here. It refuses the window, and because a late lineup
or odds append can still arrive, it is retried inside the pre-tip margin
before that refusal stands.

``gate3_role_readiness`` is about role state, and role state is consumed by
exactly three prop routes -- ``assists``, ``points_assists`` and
``points_rebounds``, which is ``gate3_v2.GATE3_CHANGED_PROPS``. Nothing else
in the slate reads it. A game whose capture carries no lineup records is a
*serving-window input failure* scoped to those three routes for that game,
and the pricing path already handles it exactly that way: ``15_price_markets``
moves a Gate 3 row with no ready role state into the rejected set under
``quote_filter_reason = gate3_role_state_unavailable`` and prices the rest of
the slate normally, which the frozen integration contract documents as a
rejection path.

So this script does not refuse a window, a game or a slate for it. Refusing
would convert one game's missing lineups into no forecast for anybody, and
promoting it to a model-level event would be the wrong failure domain
entirely: changing the serving fit repairs no lineup, and a mid-evening
authority change is its own inconsistency. The measurement is attributed to
the games it came from, recorded in this window's receipt, and the affected
routes are left to the per-row rejection that already exists. The games that
advanced keep what they published and a game that did not advance keeps the
release it already has, which is the per-game live composition's guarantee
and not a new one.

WHAT IT WILL NEVER DO
---------------------

Promote, roll back, re-point serving authority, write a model artifact, or
write a false ``gate3_role_readiness`` into the fit's validation record. That
last one matters: ``assert_validation_pass`` requires any recorded window
answer to be true and ``resolve_incumbent`` calls it, so recording this
window's missing lineups against the fit would make the fit permanently
unservable -- an input failure silently retired as a model failure. Measured
passes are recorded; a measured failure belongs to the window and stays in
the window's receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_prop_quant.adaptive_fit_registry import (  # noqa: E402
    SERVING_WINDOW_VALIDATION_CHECKS,
    FitRegistry,
    RegistryError,
    resolve_registry_root,
)
from nba_prop_quant.adaptive_training import (  # noqa: E402
    snapshot_directory,
)
from nba_prop_quant.adaptive_validation import (  # noqa: E402
    check_gate3_role_readiness,
    check_t20_protocol_compatible,
)
from nba_prop_quant.gate3_v2 import GATE3_CHANGED_PROPS  # noqa: E402

AUTHORITY_PROMOTED_FIT = "promoted_fit"
AUTHORITY_FROZEN_BUNDLE = "frozen_deployment_bundle"

#: The gate that refuses a window. One name, and the reason it is the only
#: one is in the module docstring: it is the gate about the capture every
#: route reads rather than about one route's inputs.
WINDOW_REFUSING_GATES: tuple[str, ...] = ("t20_protocol_compatible",)

#: The gate that scopes to routes instead, and the routes it scopes to.
#: Imported from the module that owns the routing rather than restated, so a
#: fourth Gate 3 prop cannot become degradable here without becoming
#: degradable there.
ROUTE_SCOPED_GATE = "gate3_role_readiness"

GATE3_DEPENDENT_ROUTES: tuple[str, ...] = tuple(sorted(GATE3_CHANGED_PROPS))

#: What the pricing path already calls a row it rejected for this reason.
#: Named here so this receipt and the rejected-quote feed agree.
GATE3_ROUTE_FILTER_REASON = "gate3_role_state_unavailable"

#: How many entries ``check_gate3_role_readiness`` keeps in each evidence
#: list. A list at this length may have been cut, so the attribution built
#: from it may be missing games. Stated here because the check does not
#: export it; a test pins the two equal.
_EVIDENCE_LIMIT = 10

OUTCOME_PASS = "SERVING_WINDOW_GATES_PASS"
OUTCOME_DEGRADED = "SERVING_WINDOW_ROUTES_DEGRADED"
OUTCOME_REFUSED = "SERVING_WINDOW_GATES_REFUSED"

#: Outcomes under which the window goes on to serve.
OUTCOMES_THAT_SERVE = (OUTCOME_PASS, OUTCOME_DEGRADED)

EXIT_OK = 0
EXIT_FAILED = 1


class WindowGateRefused(RuntimeError):
    """A serving-window gate did not pass, so this window may not publish."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _game_of(window_id: Any) -> str:
    """The game a ``<game_id>:T-<n>m`` window belongs to."""
    return str(window_id).split(":", 1)[0]


def degraded_games(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Which games a failed ``gate3_role_readiness`` is actually about.

    Read out of the check's own recorded evidence rather than recomputed, so
    the attribution cannot disagree with the verdict it is attributing.

    Conservative in the one direction that matters. The check truncates its
    evidence lists, so a slate with more affected captures than it records
    would attribute the failure to fewer games than it is about -- and a game
    left out of this mapping is a game whose Gate 3 routes would price as
    normal on role state nothing checked. When the evidence cannot name every
    affected game, every advertised game is named instead, which costs three
    routes on games that may have been fine and cannot cost the reverse.
    """
    values = payload.get("values") or {}

    advertised = [
        str(window) for window in values.get("advertised_windows") or []
    ]

    without = [
        str(window) for window in values.get("captures_without_lineups") or []
    ]

    refused = [str(entry) for entry in values.get("refused_captures") or []]

    attributed: dict[str, list[str]] = {}

    for window in without:
        attributed.setdefault(_game_of(window), []).append(
            "the capture carries no lineup records"
        )

    for entry in refused:
        attributed.setdefault(_game_of(entry), []).append(
            "the capture would not verify"
        )

    # Three ways the attribution cannot be trusted to be complete, and the
    # same conservative answer to all of them: the evidence named no game at
    # all; it named games up against the limit it truncates at, so it may
    # have named only some; or the failure is the fit's own role-state hash,
    # which is not about any one game's capture.
    truncated = len(without) >= _EVIDENCE_LIMIT or len(refused) >= _EVIDENCE_LIMIT

    hashed = str(values.get("role_state_hash") or "")

    malformed = len(hashed) != 64 or any(
        character not in "0123456789abcdef" for character in hashed.lower()
    )

    if not attributed or truncated or malformed:
        return {
            _game_of(window): [
                "the failure could not be attributed to this game alone"
            ]
            for window in advertised
        }

    return {game: sorted(set(reasons)) for game, reasons in attributed.items()}


def evaluate_window_gates(
    *,
    project_root: Path,
    slate_date: str,
    data_root: Path,
    registry: FitRegistry | None,
) -> dict[str, Any]:
    """Recompute both window checks against this slate's capture on disk.

    Returns the measurement. Deciding what to do about it is
    :func:`assert_window_gates`, so the measurement can be reported by a
    caller that is not entitled to refuse.
    """
    snapshot_dir = snapshot_directory(Path(data_root))

    fit_id: str | None = None
    role_state_hash: Any = None
    candidate: Path | None = None

    if registry is not None:
        try:
            fit_id = registry.current().get("current_good_fit_id")
        except RegistryError as error:
            raise WindowGateRefused(
                "the registry's promotion state could not be read, so the "
                "serving authority this window would publish under is "
                f"unknown: {error}"
            ) from error

    if fit_id is not None:
        assert registry is not None
        manifest = registry.load_manifest(fit_id)
        role_state_hash = manifest.get("role_state_hash")
        candidate = registry.fit_dir(fit_id) / "artifacts"

    authority = (
        AUTHORITY_PROMOTED_FIT
        if fit_id is not None
        else AUTHORITY_FROZEN_BUNDLE
    )

    protocol = check_t20_protocol_compatible(
        slate_date=slate_date,
        snapshot_dir=snapshot_dir,
    )

    readiness = (
        check_gate3_role_readiness(
            project_root=Path(project_root),
            candidate=candidate,
            role_state_hash=role_state_hash,
            slate_date=slate_date,
            snapshot_dir=snapshot_dir,
        )
        if candidate is not None
        else None
    )

    results = {"t20_protocol_compatible": protocol}

    if readiness is not None:
        results["gate3_role_readiness"] = readiness

    checks = {name: result.payload() for name, result in results.items()}

    role = checks.get(ROUTE_SCOPED_GATE)

    degraded = (
        degraded_games(role)
        if role is not None and role["passed"] is not True
        else {}
    )

    return {
        "authority": authority,
        "checks": checks,
        "degraded_routes": {
            game: list(GATE3_DEPENDENT_ROUTES) for game in sorted(degraded)
        },
        "degraded_reasons": {game: degraded[game] for game in sorted(degraded)},
        "enforced": sorted(
            name for name in results if name in WINDOW_REFUSING_GATES
        ),
        "evaluated_at": _utc_now(),
        "fit_id": fit_id,
        "route_filter_reason": GATE3_ROUTE_FILTER_REASON,
        "route_scoped": sorted(
            name for name in results if name == ROUTE_SCOPED_GATE
        ),
        "slate_date": str(slate_date),
        "snapshot_dir": str(snapshot_dir),
    }


def assert_window_gates(measurement: dict[str, Any]) -> None:
    """Refuse the window when a *window-refusing* check is false or unmeasurable.

    Only :data:`WINDOW_REFUSING_GATES` reaches this. The route-scoped gate is
    deliberately not consulted: a game whose capture carries no lineup
    records costs that game three prop routes through a rejection path the
    pricing step already owns, and costs nothing else. Refusing here would
    turn it into no forecast for any game, which is the failure the per-row
    rejection exists to avoid.

    ``NOT_EVALUABLE`` is a refusal and not a pass. The caller has already
    captured and verified this window, so "there is no capture to measure
    against" is a contradiction of the thing that got us here, and treating
    it as a pass would be the exact substitution this module exists to
    prevent.
    """
    refusals: list[str] = []

    for name in measurement["enforced"]:
        payload = measurement["checks"][name]

        if payload["passed"] is None:
            refusals.append(
                f"{name} could not be measured for "
                f"{measurement['slate_date']} ({payload['evidence']}), and a "
                "window that has just captured cannot also have no capture"
            )
            continue

        if not payload["passed"]:
            refusals.append(
                f"{name} failed for {measurement['slate_date']}: "
                f"{payload['error'] or payload['evidence']}"
            )

    if refusals:
        raise WindowGateRefused(
            "this serving window may not publish: " + "; ".join(refusals)
        )


def record_window_gates(
    registry: FitRegistry,
    fit_id: str,
    measurement: dict[str, Any],
) -> dict[str, Any] | None:
    """Fold the measured window *passes* into the fit's validation record.

    The record is the registry's own accounting of a fit and it was
    permanently incomplete while nothing could answer these two names. The
    live evaluation above is what gates; this only stops the record claiming
    a fit was never asked. Existing answers are carried through unchanged, so
    recording can add a name and can never remove or flip a fit-time one.

    Passes only, and that asymmetry is the whole point of the failure-domain
    split. ``assert_validation_pass`` requires any recorded window answer to
    be true and ``resolve_incumbent`` calls it, so writing this evening's
    missing lineups in here would make the fit unservable for good -- a
    serving-window input failure retired as a permanent model failure, which
    is precisely the attribution this file refuses to make. A measured
    failure is a fact about this window and is recorded in this window's
    receipt, where the operator and the degraded-route accounting read it.
    """
    existing = registry.load_validation(fit_id)

    if existing is None:
        return None

    checks = dict(existing.get("checks") or {})

    for name in SERVING_WINDOW_VALIDATION_CHECKS:
        payload = measurement["checks"].get(name)

        if payload is None or payload["passed"] is not True:
            continue

        checks[name] = True

    if checks == dict(existing.get("checks") or {}):
        return existing

    return registry.record_validation(
        fit_id,
        checks,
        actor=existing.get("actor"),
        notes=(
            f"serving-window checks measured at the {measurement['slate_date']}"
            " window against that window's own capture"
        ),
    )


def run(
    args: argparse.Namespace,
    *,
    sleep: Any = None,
) -> dict[str, Any]:
    registry: FitRegistry | None = None

    if args.registry_root is not None:
        registry = FitRegistry(
            root=resolve_registry_root(
                args.registry_root, project_root=PROJECT_ROOT
            ),
            project_root=PROJECT_ROOT,
        )

    wait = sleep if sleep is not None else time.sleep

    def measure() -> dict[str, Any]:
        return evaluate_window_gates(
            project_root=PROJECT_ROOT,
            slate_date=args.slate_date,
            data_root=Path(args.data_root),
            registry=registry,
        )

    attempts: list[dict[str, Any]] = []

    refusal: WindowGateRefused | None = None

    measurement: dict[str, Any] = {}

    # Retried, not just refused, and strictly inside the room this window
    # already has before its own tip. The collector appends a capture's
    # components as it gets them, so a component that was not on disk when
    # this was first asked can be there a few seconds later -- and the thing
    # that must not happen is a window refused for an input that arrived
    # while nobody looked again. The budget is small and bounded on purpose:
    # the caller has already proved the tip is ahead, the post-tip guard is
    # re-taken by the caller after this returns, and a retry loop that could
    # outlast the margin would be a way of reaching a tip rather than a way
    # of avoiding one.
    for attempt in range(1, max(int(args.max_attempts), 1) + 1):
        if attempt > 1:
            wait(float(args.retry_seconds))

        measurement = measure()

        try:
            assert_window_gates(measurement)
        except WindowGateRefused as error:
            refusal = error
        else:
            refusal = None

        attempts.append(
            {
                "attempt": attempt,
                "evaluated_at": measurement["evaluated_at"],
                "refused": refusal is not None,
            }
        )

        if refusal is None:
            break

    measurement["attempts"] = attempts

    if refusal is not None:
        measurement["outcome"] = OUTCOME_REFUSED
        measurement["reason"] = str(refusal)
        return measurement

    if registry is not None and measurement["fit_id"] is not None:
        record_window_gates(
            registry, measurement["fit_id"], measurement
        )

        if not measurement["degraded_routes"]:
            # Every window check passed and is now recorded, so the
            # fail-closed whole-set assertion is available and is taken. It
            # cannot pass on a record the live evaluation did not just
            # satisfy. Skipped when routes are degraded, because the record
            # deliberately carries no answer for a gate this window measured
            # as an input failure.
            registry.assert_serving_window_validation_pass(
                measurement["fit_id"]
            )

    if measurement["degraded_routes"]:
        games = ", ".join(sorted(measurement["degraded_routes"]))

        measurement["outcome"] = OUTCOME_DEGRADED
        measurement["reason"] = (
            f"{ROUTE_SCOPED_GATE} did not pass for game(s) {games}, so the "
            f"{len(GATE3_DEPENDENT_ROUTES)} route(s) that read role state -- "
            f"{', '.join(GATE3_DEPENDENT_ROUTES)} -- are rejected for those "
            f"games under {GATE3_ROUTE_FILTER_REASON}. Every other route on "
            "every game serves, the serving authority is unchanged, and the "
            "affected routes are retried at the next scheduled pre-tip "
            "window"
        )

        return measurement

    measurement["outcome"] = OUTCOME_PASS
    measurement["reason"] = (
        "every enforced serving-window check was recomputed against the "
        f"{measurement['slate_date']} capture and passed"
    )

    return measurement


def render(receipt: dict[str, Any]) -> str:
    lines = [
        "## Serving-window gates",
        "",
        f"- Outcome: `{receipt['outcome']}`",
        f"- Slate date: `{receipt['slate_date']}`",
        f"- Serving authority: `{receipt['authority']}`",
        f"- Fit: `{receipt['fit_id'] or 'none'}`",
        f"- Refuses the window: `{', '.join(receipt['enforced']) or 'none'}`",
        f"- Scopes to routes: `{', '.join(receipt.get('route_scoped') or []) or 'none'}`",
        f"- Attempts: `{len(receipt.get('attempts') or []) or 1}`",
        "",
        "| Check | On failure | Result | Evidence |",
        "| --- | --- | --- | --- |",
    ]

    for name in sorted(receipt["checks"]):
        payload = receipt["checks"][name]

        result = (
            "NOT_EVALUABLE"
            if payload["passed"] is None
            else ("PASS" if payload["passed"] else "FAIL")
        )

        consequence = (
            "refuses the window"
            if name in receipt["enforced"]
            else "rejects its own routes"
        )

        lines.append(
            f"| `{name}` "
            f"| {consequence} "
            f"| {result} "
            f"| {payload['error'] or payload['evidence']} |"
        )

    degraded = receipt.get("degraded_routes") or {}

    if degraded:
        reasons = receipt.get("degraded_reasons") or {}

        lines += [
            "",
            "### Routes rejected for this window",
            "",
            "The serving fit is unchanged. Every route not listed here "
            "serves for every game.",
            "",
            "| Game | Rejected routes | Why |",
            "| --- | --- | --- |",
        ]

        for game in sorted(degraded):
            lines.append(
                f"| `{game}` "
                f"| `{', '.join(degraded[game])}` "
                f"| {'; '.join(reasons.get(game) or [])} |"
            )

    lines += ["", receipt["reason"], ""]

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the two serving-window validation checks against "
            "this slate's capture. Refuse the window on the gate about the "
            "capture every route reads; reject only the routes that read "
            "role state on the gate about role state."
        )
    )
    parser.add_argument("--slate-date", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--registry-root", default=None)
    parser.add_argument("--status-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help=(
            "how many times to re-measure before a window-refusing gate's "
            "failure stands"
        ),
    )
    parser.add_argument(
        "--retry-seconds",
        type=float,
        default=20.0,
        help="how long to wait between those measurements",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        receipt = run(args)

    except (RegistryError, WindowGateRefused) as error:
        receipt = {
            "attempts": [],
            "authority": None,
            "checks": {},
            "degraded_reasons": {},
            "degraded_routes": {},
            "enforced": [],
            "evaluated_at": _utc_now(),
            "fit_id": None,
            "outcome": OUTCOME_REFUSED,
            "reason": str(error),
            "route_filter_reason": GATE3_ROUTE_FILTER_REASON,
            "route_scoped": [],
            "slate_date": args.slate_date,
            "snapshot_dir": None,
        }

    if args.status_path is not None:
        path = Path(args.status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    text = render(receipt)

    print(text)

    if args.summary_path is not None:
        with Path(args.summary_path).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    # Degraded exits 0. A window that priced and published every route but
    # three on one game is a window that served, and reporting it as a
    # failure would red a segment that did its job -- and, worse, would make
    # one game's missing lineups look like the slate's problem.
    return EXIT_OK if receipt["outcome"] in OUTCOMES_THAT_SERVE else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
