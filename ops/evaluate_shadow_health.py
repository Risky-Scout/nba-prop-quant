#!/usr/bin/env python3
"""Decide whether the candidate shadow was healthy, and say so visibly.

The shadow step is ``continue-on-error: true`` and must stay that way: a
candidate that can fail the production lifecycle is worse than no candidate,
and the entry point catches its own failures and exits 0 on purpose. The cost
of that design was that GitHub went green whether the shadow worked or not.
A run whose shadow raised on every game, refused to assemble a covariance, or
produced nothing at all was indistinguishable from a clean one.

So the health decision is made here instead, from the status file the shadow
persisted rather than from the step's conclusion -- which is exactly the signal
that cannot be trusted. This step may fail the job. Failing it leaves incumbent
serving untouched, because the incumbent was served before this ran and this
writes nothing: the visible red is the point.

Nothing here invents a threshold. The gates are
``research/final_model/live_shadow_promotion_policy.json``'s own
``operational_gates``, frozen before any live result was seen, read at runtime
so that a policy change moves this check rather than silently disagreeing with
it.

Exit codes
----------
0   healthy, or legitimately nothing to evaluate
1   unhealthy: at least one gate or structural check failed
2   the shadow was expected to run and left no status behind
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The frozen live policy. Read, never written.
POLICY_RELATIVE_PATH = Path("research/final_model/live_shadow_promotion_policy.json")

#: Outcomes in which the shadow legitimately produced no graded rows. Not
#: healthy-by-exception: the structural checks still apply, and only the gates
#: that need shadowed games become inapplicable.
NOTHING_TO_GRADE_OUTCOMES = frozenset(
    {
        "DECLARE_ONLY_NO_SLATE_RESOLVED",
        "SLATE_NOT_SETTLED_NOTHING_TO_GRADE",
    }
)

#: O5 names the identities a graded row must resolve to. The shadow's
#: provenance payload carries them under its own field names, so the mapping
#: between the two is written down rather than guessed at each call site.
PROVENANCE_FIELD_SOURCES = {
    "final_model_spec_sha256": ("final_model_spec_sha256",),
    "factor_spec_hash": ("factor_spec_hash",),
    "artifact_hash_or_version": (
        "marginal_source_sha256",
        "copula_source_sha256",
    ),
    "production_code_sha": ("code_sha",),
}

FAIL = "FAIL"
PASS = "PASS"
WARN = "WARN"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Finding:
    name: str
    status: str
    detail: str
    values: dict[str, Any] = field(default_factory=dict)
    gate: str | None = None


@dataclass
class Health:
    findings: list[Finding] = field(default_factory=list)

    def record(
        self,
        name: str,
        status: str,
        detail: str,
        gate: str | None = None,
        **values: Any,
    ) -> None:
        self.findings.append(
            Finding(
                name=name,
                status=status,
                detail=detail,
                values=values,
                gate=gate,
            )
        )

    def judge(
        self,
        name: str,
        ok: bool,
        detail: str,
        gate: str | None = None,
        **values: Any,
    ) -> None:
        self.record(
            name,
            PASS if ok else FAIL,
            detail,
            gate=gate,
            **values,
        )

    @property
    def failures(self) -> list[str]:
        return [f.name for f in self.findings if f.status == FAIL]

    @property
    def warnings(self) -> list[str]:
        return [f.name for f in self.findings if f.status == WARN]

    @property
    def healthy(self) -> bool:
        return not self.failures

    def payload(self) -> dict[str, Any]:
        return {
            "failures": self.failures,
            "findings": [
                {
                    "detail": f.detail,
                    "gate": f.gate,
                    "name": f.name,
                    "status": f.status,
                    "values": f.values,
                }
                for f in self.findings
            ],
            "healthy": self.healthy,
            "warnings": self.warnings,
        }


def load_policy(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number


def evaluate(status: dict[str, Any], policy: dict[str, Any]) -> Health:
    """Judge one persisted shadow status against the frozen gates."""
    health = Health()

    gates = policy.get("operational_gates") or {}

    shadow = status.get("shadow") or {}
    authority = status.get("authority") or {}

    outcome = str(status.get("outcome", "NO_OUTCOME_RECORDED"))

    # ---------------------------------------------------------------
    # Did the shadow run at all? The status file is written by the entry
    # point, so its contents are the only evidence that the step did more
    # than get scheduled.
    # ---------------------------------------------------------------
    started = bool(status.get("started_at")) and bool(status.get("entry_point"))

    health.judge(
        "shadow_actually_started",
        started,
        (
            f"entry point {status.get('entry_point', 'unrecorded')} started at "
            f"{status.get('started_at', 'unrecorded')}"
        ),
        entry_point=status.get("entry_point"),
        started_at=status.get("started_at"),
    )

    health.judge(
        "shadow_reached_its_own_end",
        bool(status.get("finished_at")),
        f"finished at {status.get('finished_at', 'unrecorded')}",
        finished_at=status.get("finished_at"),
    )

    error = status.get("error")

    health.judge(
        "shadow_recorded_no_error",
        not error,
        str(error) if error else "no error field",
        error=error,
    )

    health.judge(
        "shadow_outcome_is_recognised",
        outcome
        in (
            NOTHING_TO_GRADE_OUTCOMES
            | {"SHADOWED", "SHADOWED_WITH_FALLBACKS"}
        ),
        f"outcome {outcome}",
        outcome=outcome,
    )

    # ---------------------------------------------------------------
    # The production safety invariants. These hold in every outcome,
    # including the ones with nothing to grade, because they are statements
    # about what the shadow did to production rather than about its results.
    # ---------------------------------------------------------------
    served = str(status.get("served_authority", "unrecorded"))

    health.judge(
        "served_model_remained_the_incumbent",
        served == "incumbent",
        f"served authority {served}",
        gate="O6_incumbent_serving_unaffected",
        served_authority=served,
    )

    health.judge(
        "production_serving_was_not_affected",
        status.get("production_serving_was_affected") is False,
        (
            "production_serving_was_affected = "
            f"{status.get('production_serving_was_affected')!r}"
        ),
        gate="O6_incumbent_serving_unaffected",
        production_serving_was_affected=status.get(
            "production_serving_was_affected"
        ),
    )

    published = _number(status.get("rows_published"))

    published_max = _number((gates.get("O1_shadow_publishing_violations") or {}).get("max"))

    health.judge(
        "nothing_was_published",
        published is not None
        and published_max is not None
        and published <= published_max,
        f"rows_published {status.get('rows_published')!r} against max {published_max!r}",
        gate="O1_shadow_publishing_violations",
        maximum=published_max,
        rows_published=status.get("rows_published"),
    )

    health.judge(
        "the_shadow_refused_to_publish",
        str(authority.get("publish_attempt", "")) == "ShadowPublishingDisabled",
        f"publish attempt {authority.get('publish_attempt')!r}",
        publish_attempt=authority.get("publish_attempt"),
    )

    health.judge(
        "the_shadow_refused_to_promote",
        str(authority.get("promotion_attempt", "")) == "ShadowPromotionRefused",
        f"promotion attempt {authority.get('promotion_attempt')!r}",
        promotion_attempt=authority.get("promotion_attempt"),
    )

    # ---------------------------------------------------------------
    # The numerical gates. Inapplicable when no game was shadowed, which is
    # a normal state before a settled slate exists.
    # ---------------------------------------------------------------
    if not shadow:
        health.record(
            "numerical_gates",
            NOT_APPLICABLE if outcome in NOTHING_TO_GRADE_OUTCOMES else FAIL,
            (
                f"outcome {outcome} carries no shadow results"
                if outcome in NOTHING_TO_GRADE_OUTCOMES
                else f"outcome {outcome} should have carried shadow results"
            ),
            outcome=outcome,
        )

        return health

    offered = _number(shadow.get("games_offered")) or 0.0
    shadowed = _number(shadow.get("games_shadowed")) or 0.0
    fell_back = _number(shadow.get("games_that_fell_back")) or 0.0

    health.record(
        "games_were_shadowed",
        PASS if shadowed > 0 else WARN,
        (
            f"{shadow.get('games_shadowed')} of {shadow.get('games_offered')} "
            "offered game(s) shadowed, "
            f"{shadow.get('games_skipped_for_too_few_players_or_events')} "
            "skipped for too few players or events"
        ),
        games_offered=shadow.get("games_offered"),
        games_shadowed=shadow.get("games_shadowed"),
        games_skipped=shadow.get(
            "games_skipped_for_too_few_players_or_events"
        ),
    )

    health.record(
        "rows_and_events_were_written",
        PASS
        if (_number(shadow.get("rows_written")) or 0) > 0
        else (WARN if shadowed == 0 else FAIL),
        (
            f"{shadow.get('rows_written')} row(s) from "
            f"{shadow.get('events_shadowed')} shadowed event(s)"
        ),
        events_shadowed=shadow.get("events_shadowed"),
        rows_written=shadow.get("rows_written"),
    )

    psd_max = _number((gates.get("O2_psd_failures") or {}).get("max"))

    psd_failures = _number(shadow.get("psd_failures"))

    health.judge(
        "no_psd_failures",
        psd_failures is not None and psd_max is not None and psd_failures <= psd_max,
        f"psd_failures {shadow.get('psd_failures')!r} against max {psd_max!r}",
        gate="O2_psd_failures",
        maximum=psd_max,
        psd_failures=shadow.get("psd_failures"),
    )

    floor = _number(shadow.get("psd_eigenvalue_floor"))

    eigenvalue = _number(shadow.get("min_covariance_eigenvalue"))

    health.judge(
        "minimum_covariance_eigenvalue_is_above_the_floor",
        eigenvalue is None or floor is None or eigenvalue >= floor,
        (
            f"minimum covariance eigenvalue "
            f"{shadow.get('min_covariance_eigenvalue')!r} against floor "
            f"{shadow.get('psd_eigenvalue_floor')!r}"
        ),
        gate="O2_psd_failures",
        floor=shadow.get("psd_eigenvalue_floor"),
        min_covariance_eigenvalue=shadow.get("min_covariance_eigenvalue"),
    )

    deviation_max = _number(
        (gates.get("O3_same_player_block_deviation") or {}).get("max")
    )

    deviation = _number(shadow.get("same_player_max_block_deviation"))

    health.judge(
        "same_player_block_deviation_is_within_the_gate",
        deviation is None
        or (deviation_max is not None and deviation <= deviation_max),
        (
            "same-player maximum block deviation "
            f"{shadow.get('same_player_max_block_deviation')!r} against max "
            f"{deviation_max!r}"
        ),
        gate="O3_same_player_block_deviation",
        maximum=deviation_max,
        same_player_max_block_deviation=shadow.get(
            "same_player_max_block_deviation"
        ),
    )

    rate_max = _number((gates.get("O4_candidate_fallback_rate") or {}).get("max"))

    rate = (fell_back / shadowed) if shadowed else None

    health.judge(
        "candidate_fallback_rate_is_within_the_gate",
        rate is None or (rate_max is not None and rate <= rate_max),
        (
            f"{shadow.get('games_that_fell_back')} of "
            f"{shadow.get('games_shadowed')} shadowed game(s) fell back "
            f"(rate {rate if rate is None else round(rate, 6)!r} against max "
            f"{rate_max!r})"
        ),
        gate="O4_candidate_fallback_rate",
        games_that_fell_back=shadow.get("games_that_fell_back"),
        maximum=rate_max,
        rate=rate,
    )

    # O4 again, in the shape the policy itself singles out: a repeated
    # deterministic failure sharing one root cause requires review even when
    # the total rate is inside the threshold. Reported as a warning, because
    # the policy asks for review rather than for a refusal.
    reasons: dict[str, int] = {}

    for fallback in shadow.get("fallbacks") or []:
        reason = str(fallback.get("failure_reason") or fallback.get("reason") or "")

        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1

    repeated = sorted(
        f"{reason} x{count}" for reason, count in reasons.items() if count > 1
    )

    health.record(
        "no_repeated_fallback_root_cause",
        PASS if not repeated else WARN,
        (
            "repeated deterministic fallbacks requiring review: "
            + ", ".join(repeated)
            if repeated
            else f"{len(reasons)} distinct fallback reason(s), none repeated"
        ),
        gate="O4_candidate_fallback_rate",
        repeated=repeated,
    )

    # ---------------------------------------------------------------
    # O5: every identity a graded row must resolve to is recorded.
    # ---------------------------------------------------------------
    provenance = shadow.get("provenance") or {}

    unresolved = sorted(
        required
        for required, sources in PROVENANCE_FIELD_SOURCES.items()
        if not any(provenance.get(source) for source in sources)
    )

    health.judge(
        "provenance_is_complete",
        not unresolved and bool(shadow.get("provenance_fingerprint")),
        (
            "unresolved provenance identities: " + ", ".join(unresolved)
            if unresolved
            else (
                "every required identity resolves under fingerprint "
                f"{str(shadow.get('provenance_fingerprint'))[:12]}"
            )
        ),
        gate="O5_provenance_completeness",
        provenance_fingerprint=shadow.get("provenance_fingerprint"),
        unresolved=unresolved,
    )

    return health


def render(
    health: Health,
    *,
    status_path: Path,
    present: bool,
    expected: bool,
) -> str:
    if not present:
        headline = (
            "**FAIL** — the shadow was expected to run and left no status file"
            if expected
            else "**SKIPPED** — the shadow was not expected to run"
        )
    else:
        headline = f"**{'HEALTHY' if health.healthy else 'UNHEALTHY'}**"

    lines = [
        "## Candidate shadow health",
        "",
        headline,
        "",
        f"Status file: `{status_path}`",
        "",
    ]

    if health.findings:
        lines += ["| check | result | gate | detail |", "| --- | --- | --- | --- |"]

        for finding in health.findings:
            lines.append(
                f"| {finding.name} | {finding.status} | "
                f"{finding.gate or '—'} | {finding.detail} |"
            )

    if present and not health.healthy:
        lines += [
            "",
            "Incumbent serving is unaffected: the incumbent was served before "
            "this check ran and this check writes nothing. The red signal is "
            "the point — the shadow step is non-blocking by design, so without "
            "this the job would have gone green on an unhealthy shadow.",
        ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-path",
        required=True,
        help="the shadow.json the shadow entry point persisted",
    )
    parser.add_argument(
        "--shadow-was-expected",
        action="store_true",
        help="the lifecycle expected the shadow to run on this slate",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="the frozen live policy to read the operational gates from",
    )
    parser.add_argument("--health-path", default=None)
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args(argv)

    status_path = Path(args.status_path)

    policy_path = (
        Path(args.policy)
        if args.policy
        else PROJECT_ROOT / POLICY_RELATIVE_PATH
    )

    present = status_path.is_file()

    health = Health()

    if present:
        health = evaluate(
            json.loads(status_path.read_text(encoding="utf-8")),
            load_policy(policy_path),
        )

    rendered = render(
        health,
        status_path=status_path,
        present=present,
        expected=bool(args.shadow_was_expected),
    )

    payload = {
        "expected": bool(args.shadow_was_expected),
        "policy": str(policy_path),
        "status_path": str(status_path),
        "status_present": present,
        **health.payload(),
    }

    if not present:
        payload["healthy"] = not args.shadow_was_expected
        payload["failures"] = (
            ["shadow_status_is_missing"] if args.shadow_was_expected else []
        )

    if args.health_path:
        path = Path(args.health_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    if args.summary_path:
        with Path(args.summary_path).open("a", encoding="utf-8") as handle:
            handle.write(rendered)

    print(rendered)

    if not present:
        # A shadow that was expected to run and persisted nothing is the exact
        # case the step conclusion could not distinguish from success.
        return 2 if args.shadow_was_expected else 0

    return 0 if health.healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
