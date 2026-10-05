"""Acceptance gates A-H for the shadow latent-state candidate.

SHADOW / RESEARCH ONLY.

The gate thresholds are declared here, in code, before the validation run
produces the numbers they are applied to. A gate is never relaxed to obtain a
pass: :func:`evaluate_gates` reads a validation report and returns verdicts,
and :func:`assert_promotable` refuses for *any* failing gate.

There is deliberately no promotion function in this package. Even a candidate
that passes every gate is only eligible for *production-integration design*;
promotion remains the exclusive responsibility of the Step 3D lifecycle,
which this branch does not touch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

VERDICT_ACCEPTED = "SHADOW V1 ACCEPTED FOR PRODUCTION-INTEGRATION DESIGN."

VERDICT_REJECTED_PREFIX = "SHADOW V1 REJECTED:"


class ShadowPromotionRefused(RuntimeError):
    """Raised whenever anything asks this package to promote a candidate."""


@dataclass(frozen=True)
class GateThresholds:
    """Pre-registered gate thresholds. Declared before the numbers exist."""

    # GATE A: marginal preservation. The simulated margin is the production
    # margin by construction, so the only admissible discrepancy is Monte
    # Carlo noise. A z-score against the Monte Carlo standard error is the
    # right currency; 5 sigma on the worst dimension of the worst game.
    max_marginal_over_z: float = 5.0
    max_marginal_mean_z: float = 5.0
    max_variance_relative_error: float = 0.05

    # GATE B: cross-player dependence reproduction must beat conditional
    # independence materially. Independence predicts exactly zero for every
    # cross-player bucket, so the candidate must cut the observed-vs-implied
    # RMSE by at least this fraction.
    min_cross_player_rmse_reduction: float = 0.50

    # GATE C: 2-leg joint calibration no worse than baseline.
    max_two_leg_brier_regression: float = 0.0005

    # GATE D: 3/4-leg calibration must not degrade materially.
    max_multi_leg_brier_regression: float = 0.0020

    # GATE E: same-player double counting. The induced within-player block
    # must equal the incumbent block to float64 precision.
    max_same_player_block_deviation: float = 1e-9

    # GATE F: PSD / numerical stability on every held-out game tested.
    min_covariance_eigenvalue: float = -1e-10

    # GATE G: sparse / unseen player fallback must work with no pair-specific
    # parameters.
    require_zero_pairwise_parameters: bool = True

    # GATE H: production model untouched.
    require_clean_production_surface: bool = True


@dataclass(frozen=True)
class GateResult:
    gate: str
    name: str
    passed: bool
    evidence: dict[str, Any]

    def describe(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"GATE {self.gate} ({self.name}): {status}"


def _get(report: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    node: Any = report
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def evaluate_gates(
    report: Mapping[str, Any],
    thresholds: GateThresholds | None = None,
) -> list[GateResult]:
    """Apply gates A-H to a validation report."""
    limits = thresholds or GateThresholds()
    results: list[GateResult] = []

    # ---- GATE A: marginal preservation ------------------------------
    over_z = _get(report, "marginal_preservation", "candidate", "max_abs_over_z")
    mean_z = _get(report, "marginal_preservation", "candidate", "max_abs_mean_z")
    variance_error = _get(
        report, "marginal_preservation", "candidate", "max_abs_variance_relative_error"
    )
    results.append(
        GateResult(
            gate="A",
            name="marginal calibration preserved within Monte Carlo tolerance",
            passed=(
                over_z is not None
                and mean_z is not None
                and variance_error is not None
                and over_z <= limits.max_marginal_over_z
                and mean_z <= limits.max_marginal_mean_z
                and variance_error <= limits.max_variance_relative_error
            ),
            evidence={
                "max_abs_over_z": over_z,
                "max_abs_mean_z": mean_z,
                "max_abs_variance_relative_error": variance_error,
                "limit_over_z": limits.max_marginal_over_z,
                "limit_mean_z": limits.max_marginal_mean_z,
                "limit_variance_relative_error": limits.max_variance_relative_error,
            },
        )
    )

    # ---- GATE B: cross-player dependence ----------------------------
    candidate_rmse = _get(report, "residual_dependence", "cross_player_rmse", "candidate")
    baseline_rmse = _get(
        report, "residual_dependence", "cross_player_rmse", "baseline_independence"
    )
    reduction = (
        None
        if candidate_rmse is None or not baseline_rmse
        else 1.0 - candidate_rmse / baseline_rmse
    )
    results.append(
        GateResult(
            gate="B",
            name="held-out cross-player dependence beats conditional independence",
            passed=(
                reduction is not None
                and reduction >= limits.min_cross_player_rmse_reduction
            ),
            evidence={
                "candidate_rmse": candidate_rmse,
                "baseline_independence_rmse": baseline_rmse,
                "rmse_reduction": reduction,
                "limit_rmse_reduction": limits.min_cross_player_rmse_reduction,
            },
        )
    )

    # ---- GATE C: 2-leg calibration ----------------------------------
    two_candidate = _get(report, "joint_events", "by_legs", "2", "candidate", "brier")
    two_baselines = [
        _get(report, "joint_events", "by_legs", "2", name, "brier")
        for name in ("baseline_independence", "baseline_production")
    ]
    two_best_baseline = min(
        [value for value in two_baselines if value is not None], default=None
    )
    results.append(
        GateResult(
            gate="C",
            name="2-leg joint calibration no worse than baseline",
            passed=(
                two_candidate is not None
                and two_best_baseline is not None
                and two_candidate
                <= two_best_baseline + limits.max_two_leg_brier_regression
            ),
            evidence={
                "candidate_brier": two_candidate,
                "best_baseline_brier": two_best_baseline,
                "limit_regression": limits.max_two_leg_brier_regression,
            },
        )
    )

    # ---- GATE D: 3/4-leg calibration --------------------------------
    multi: dict[str, Any] = {}
    multi_passed = True
    for legs in ("3", "4"):
        candidate = _get(report, "joint_events", "by_legs", legs, "candidate", "brier")
        baselines = [
            _get(report, "joint_events", "by_legs", legs, name, "brier")
            for name in ("baseline_independence", "baseline_production")
        ]
        best = min([value for value in baselines if value is not None], default=None)
        multi[f"{legs}_leg_candidate_brier"] = candidate
        multi[f"{legs}_leg_best_baseline_brier"] = best
        if candidate is None or best is None or candidate > best + limits.max_multi_leg_brier_regression:
            multi_passed = False
    multi["limit_regression"] = limits.max_multi_leg_brier_regression
    results.append(
        GateResult(
            gate="D",
            name="3-leg / 4-leg calibration shows no material degradation",
            passed=multi_passed,
            evidence=multi,
        )
    )

    # ---- GATE E: no same-player double counting ---------------------
    deviation = _get(report, "same_player_contract", "max_block_deviation")
    checked = _get(report, "same_player_contract", "games_checked", default=0)
    results.append(
        GateResult(
            gate="E",
            name="no same-player dependence double counting",
            passed=(
                deviation is not None
                and checked > 0
                and deviation <= limits.max_same_player_block_deviation
            ),
            evidence={
                "max_block_deviation": deviation,
                "games_checked": checked,
                "limit_block_deviation": limits.max_same_player_block_deviation,
            },
        )
    )

    # ---- GATE F: PSD / numerical stability --------------------------
    min_eig = _get(report, "stability", "min_covariance_eigenvalue")
    failures = _get(report, "stability", "numerical_failures", default=None)
    games = _get(report, "stability", "games_tested", default=0)
    results.append(
        GateResult(
            gate="F",
            name="PSD and numerical stability on every held-out game tested",
            passed=(
                min_eig is not None
                and failures == 0
                and games > 0
                and min_eig >= limits.min_covariance_eigenvalue
            ),
            evidence={
                "min_covariance_eigenvalue": min_eig,
                "numerical_failures": failures,
                "games_tested": games,
                "limit_min_eigenvalue": limits.min_covariance_eigenvalue,
            },
        )
    )

    # ---- GATE G: sparse / unseen fallback ---------------------------
    pairwise = _get(report, "stability", "pairwise_parameter_count")
    fallback_ok = _get(report, "stability", "unseen_player_simulation_ok")
    results.append(
        GateResult(
            gate="G",
            name="unseen / sparse player fallback without pair-specific overfitting",
            passed=(
                fallback_ok is True
                and pairwise == 0
                if limits.require_zero_pairwise_parameters
                else fallback_ok is True
            ),
            evidence={
                "pairwise_parameter_count": pairwise,
                "unseen_player_simulation_ok": fallback_ok,
            },
        )
    )

    # ---- GATE H: production untouched -------------------------------
    touched = _get(report, "production_surface", "modified_paths", default=None)
    results.append(
        GateResult(
            gate="H",
            name="production model remains untouched",
            passed=(touched == [] if limits.require_clean_production_surface else True),
            evidence={"modified_production_paths": touched},
        )
    )

    return results


def shadow_verdict(results: list[GateResult]) -> str:
    """The single permitted verdict string."""
    failed = [result for result in results if not result.passed]
    if not failed:
        return VERDICT_ACCEPTED
    detail = "; ".join(
        f"GATE {result.gate} ({result.name}) evidence={result.evidence}"
        for result in failed
    )
    return f"{VERDICT_REJECTED_PREFIX} {detail}"


def assert_promotable(results: list[GateResult]) -> None:
    """Always refuses.

    Passing every gate makes the candidate eligible for production-integration
    *design*, not for promotion. Promotion is owned by the Step 3D lifecycle,
    and this research branch has no promotion path by construction, so the
    only correct behaviour for a promotion request reaching this package is to
    refuse -- whether the gates passed or not.
    """
    failed = [result.gate for result in results if not result.passed]
    if failed:
        raise ShadowPromotionRefused(
            "shadow candidate failed gates "
            f"{failed}; rejected candidates cannot be promoted"
        )
    raise ShadowPromotionRefused(
        "the game-latent-state shadow v1 has no promotion path; a passing "
        "gate set authorises production-integration design only, and "
        "promotion remains owned by the Step 3D lifecycle"
    )
