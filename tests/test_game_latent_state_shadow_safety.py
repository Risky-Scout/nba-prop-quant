"""Branch-safety contracts for the game-level latent-state shadow branch.

These tests prove containment, not statistics. The shadow layer may not touch
the production model, the Step 3C adaptive machinery, the Step 3D automation,
the promotion state or the WizardOfOdds publishing surface, and it may not
offer a promotion path even to a candidate that passes every acceptance gate.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from nba_prop_quant.research.game_latent_state.artifacts import (
    PROMOTION_ELIGIBILITY,
    ArtifactManifest,
)
from nba_prop_quant.research.game_latent_state.gates import (
    VERDICT_ACCEPTED,
    VERDICT_REJECTED_PREFIX,
    GateThresholds,
    ShadowPromotionRefused,
    assert_promotable,
    bonferroni_z,
    evaluate_gates,
    shadow_verdict,
)

PROJECT = Path(__file__).resolve().parents[1]

PRODUCTION_REF = "production/wizardofodds-integration"

SHADOW_PACKAGE = PROJECT / "src" / "nba_prop_quant" / "research" / "game_latent_state"

SHADOW_SCRIPTS = PROJECT / "research" / "game_latent_state"

#: Every path whose bytes decide production behaviour. The shadow branch must
#: leave all of them exactly as the production ref has them.
PROTECTED_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "configs/",
    "models/",
    "ops/",
    "scripts/",
    "docs/",
    "release/",
    "review/",
)

#: Production modules the shadow layer reads but must never edit.
PROTECTED_SOURCE_FILES: tuple[str, ...] = (
    "src/nba_prop_quant/adaptive_fit_registry.py",
    "src/nba_prop_quant/adaptive_training.py",
    "src/nba_prop_quant/copula.py",
    "src/nba_prop_quant/distributions.py",
    "src/nba_prop_quant/features.py",
    "src/nba_prop_quant/gate3_v2.py",
    "src/nba_prop_quant/model.py",
    "src/nba_prop_quant/pipeline.py",
    "src/nba_prop_quant/pricing.py",
    "src/nba_prop_quant/production.py",
    "src/nba_prop_quant/slate.py",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def production_base() -> str | None:
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return git("merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue
    return None


def changed_paths() -> list[str]:
    base = production_base()
    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")
    return [line for line in git("diff", "--name-only", base, "HEAD").splitlines() if line]


def python_sources() -> list[Path]:
    return sorted(
        [*SHADOW_PACKAGE.rglob("*.py"), *SHADOW_SCRIPTS.rglob("*.py")]
    )


# ----------------------------------------------------------------------
# GATE H: the production surface is byte-identical
# ----------------------------------------------------------------------


def test_no_protected_production_path_is_modified():
    offenders = [
        path
        for path in changed_paths()
        if path.startswith(PROTECTED_PREFIXES) or path in PROTECTED_SOURCE_FILES
    ]
    assert offenders == [], f"shadow branch modified production paths: {offenders}"


def test_production_automation_workflows_are_unchanged():
    base = production_base()
    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")
    for workflow in ("nba_production_lifecycle.yml", "ci.yml"):
        relative = f".github/workflows/{workflow}"
        assert git("rev-parse", f"{base}:{relative}") == git(
            "rev-parse", f"HEAD:{relative}"
        ), f"{relative} differs from the production ref"


def test_promotion_state_files_are_not_introduced_or_changed():
    offenders = [
        path
        for path in changed_paths()
        if "promotion_state" in path or "current_good_fit_id" in path
    ]
    assert offenders == []


def test_shadow_changes_live_only_in_research_and_test_namespaces():
    allowed = (
        "research/",
        "src/nba_prop_quant/research/",
        "tests/test_game_latent_state_shadow",
    )
    offenders = [path for path in changed_paths() if not path.startswith(allowed)]
    assert offenders == [], f"unexpected paths on the shadow branch: {offenders}"


# ----------------------------------------------------------------------
# no promotion path, no publishing path
# ----------------------------------------------------------------------


def test_shadow_package_defines_no_promotion_entry_point():
    """No callable in the shadow namespace may promote, register or publish."""
    forbidden = ("promote", "register_fit", "publish", "deploy")
    offenders: list[str] = []
    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            lowered = node.name.lower()
            if any(token in lowered for token in forbidden):
                # The refusal guards are the one permitted exception: their
                # whole purpose is to deny promotion.
                if node.name in {"assert_promotable", "ShadowPromotionRefused"}:
                    continue
                offenders.append(f"{path.relative_to(PROJECT)}::{node.name}")
    assert offenders == []


def test_shadow_package_never_imports_the_promotion_machinery():
    forbidden_modules = {
        "nba_prop_quant.adaptive_fit_registry",
        "nba_prop_quant.gate3_v2",
        "nba_prop_quant.production",
        "nba_prop_quant.pricing",
        "nba_prop_quant.market",
        "nba_prop_quant.live",
    }
    offenders: list[str] = []
    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name in forbidden_modules:
                    offenders.append(f"{path.relative_to(PROJECT)} imports {name}")
    assert offenders == []


def test_shadow_package_contains_no_wizardofodds_publishing_surface():
    tokens = ("wizardofodds_bundle", "runtime_bundle", "SHADOW_PUBLISH")
    offenders = [
        f"{path.relative_to(PROJECT)}: {token}"
        for path in python_sources()
        for token in tokens
        if token in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def code_string_literals(path: Path) -> list[str]:
    """Every string literal in a module except its docstrings.

    Docstrings are excluded so prose about a protected path is not mistaken
    for code that touches it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    ]


#: The one production path the shadow layer may reference, and only to read
#: the frozen dynamic-prior parameters that ``add_dynamic_priors`` needs.
READ_ONLY_PRODUCTION_LITERALS = frozenset({"models", "dynamic_params.json"})


def test_shadow_package_references_no_production_write_target():
    offenders: list[str] = []
    for path in python_sources():
        for literal in code_string_literals(path):
            lowered = literal.lower()
            if "promotion_state" in lowered or "current_good_fit_id" in lowered:
                offenders.append(f"{path.relative_to(PROJECT)}: {literal}")
            if literal.startswith("data/processed"):
                offenders.append(f"{path.relative_to(PROJECT)}: {literal}")
            if literal.startswith(("models/", "models\\")):
                offenders.append(f"{path.relative_to(PROJECT)}: {literal}")
            if literal == "models" and literal not in READ_ONLY_PRODUCTION_LITERALS:
                offenders.append(f"{path.relative_to(PROJECT)}: {literal}")
    assert offenders == []


def test_the_only_production_artifact_the_shadow_reads_is_dynamic_params():
    """Every other frozen production artifact must stay out of reach."""
    production_artifacts = {
        path.name
        for path in (PROJECT / "models").rglob("*")
        if path.is_file() and path.suffix in {".json", ".joblib", ".md", ".txt"}
    }
    referenced = {
        literal
        for path in python_sources()
        for literal in code_string_literals(path)
    }
    assert referenced & production_artifacts == {"dynamic_params.json"}


# ----------------------------------------------------------------------
# acceptance gates and the promotion refusal
# ----------------------------------------------------------------------


def passing_report() -> dict:
    return {
        "marginal_preservation": {
            "candidate": {
                "max_abs_over_z": 4.1,
                "max_abs_mean_z": 3.8,
                "z_probe_count": 800_000,
                "three_sigma_exceedance_fraction": 0.0030,
                "max_abs_over_probability_error": 0.004,
                "max_abs_variance_relative_error": 0.01,
            }
        },
        "residual_dependence": {
            "cross_player_rmse": {"candidate": 0.004, "baseline_independence": 0.030}
        },
        "joint_events": {
            "by_legs": {
                "2": {
                    "candidate": {"brier": 0.1800},
                    "baseline_independence": {"brier": 0.1850},
                    "baseline_production": {"brier": 0.1852},
                },
                "3": {
                    "candidate": {"brier": 0.0900},
                    "baseline_independence": {"brier": 0.0930},
                    "baseline_production": {"brier": 0.0931},
                },
                "4": {
                    "candidate": {"brier": 0.0400},
                    "baseline_independence": {"brier": 0.0420},
                    "baseline_production": {"brier": 0.0421},
                },
            }
        },
        "same_player_contract": {"max_block_deviation": 0.0, "games_checked": 500},
        "stability": {
            "min_covariance_eigenvalue": 0.21,
            "numerical_failures": 0,
            "games_tested": 500,
            "pairwise_parameter_count": 0,
            "unseen_player_simulation_ok": True,
        },
        "production_surface": {"modified_paths": []},
    }


def test_all_gates_pass_on_a_passing_report():
    results = evaluate_gates(passing_report())
    assert [result.gate for result in results] == list("ABCDEFGH")
    assert all(result.passed for result in results)
    assert shadow_verdict(results) == VERDICT_ACCEPTED


@pytest.mark.parametrize(
    ("gate", "mutate"),
    [
        (
            "A",
            lambda r: r["marginal_preservation"]["candidate"].update(
                max_abs_over_z=9.0
            ),
        ),
        (
            "A",
            lambda r: r["marginal_preservation"]["candidate"].update(
                three_sigma_exceedance_fraction=0.05
            ),
        ),
        (
            "A",
            lambda r: r["marginal_preservation"]["candidate"].update(
                max_abs_over_probability_error=0.2
            ),
        ),
        (
            "B",
            lambda r: r["residual_dependence"]["cross_player_rmse"].update(
                candidate=0.029
            ),
        ),
        ("C", lambda r: r["joint_events"]["by_legs"]["2"]["candidate"].update(brier=0.30)),
        ("D", lambda r: r["joint_events"]["by_legs"]["4"]["candidate"].update(brier=0.30)),
        ("E", lambda r: r["same_player_contract"].update(max_block_deviation=1e-3)),
        ("F", lambda r: r["stability"].update(min_covariance_eigenvalue=-0.2)),
        ("G", lambda r: r["stability"].update(pairwise_parameter_count=12)),
        ("H", lambda r: r["production_surface"].update(modified_paths=["models/x.json"])),
    ],
)
def test_each_gate_fails_independently(gate, mutate):
    report = passing_report()
    mutate(report)
    results = evaluate_gates(report)
    failed = {result.gate for result in results if not result.passed}
    assert failed == {gate}
    verdict = shadow_verdict(results)
    assert verdict.startswith(VERDICT_REJECTED_PREFIX)
    assert f"GATE {gate}" in verdict


def test_missing_evidence_fails_a_gate_rather_than_passing_it():
    results = evaluate_gates({})
    assert all(not result.passed for result in results)
    assert shadow_verdict(results).startswith(VERDICT_REJECTED_PREFIX)


def test_a_rejected_candidate_cannot_promote():
    report = passing_report()
    report["stability"]["min_covariance_eigenvalue"] = -1.0
    results = evaluate_gates(report)
    with pytest.raises(ShadowPromotionRefused, match="failed gates"):
        assert_promotable(results)


def test_even_a_passing_candidate_cannot_promote():
    results = evaluate_gates(passing_report())
    assert all(result.passed for result in results)
    with pytest.raises(ShadowPromotionRefused, match="no promotion path"):
        assert_promotable(results)


def test_gate_thresholds_are_declared_not_derived():
    """Thresholds are constants of the module, not functions of the data."""
    defaults = GateThresholds()
    assert defaults.marginal_family_wise_alpha == 0.01
    assert defaults.max_three_sigma_exceedance_ratio == 3.0
    assert defaults.max_over_probability_error == 0.01
    assert defaults.min_cross_player_rmse_reduction == 0.50
    assert defaults.max_same_player_block_deviation == 1e-9
    assert defaults.require_clean_production_surface is True


def test_gate_a_critical_value_scales_with_the_declared_probe_count():
    """The only data-dependent part of gate A is the probe count it corrects for.

    The family-wise alpha is fixed in code; the critical z is derived from the
    number of probes the report declares, so a larger validation run earns a
    wider bound by arithmetic rather than by a post-hoc threshold change.
    """
    alpha = GateThresholds().marginal_family_wise_alpha
    small = bonferroni_z(1_000, alpha)
    large = bonferroni_z(800_000, alpha)
    assert 3.0 < small < large < 7.0

    # Absent or nonsensical probe counts yield no bound, which fails the gate.
    assert bonferroni_z(None, alpha) is None
    assert bonferroni_z(0, alpha) is None


def test_gate_a_rejects_a_report_that_hides_its_probe_count():
    """A report cannot earn a pass by omitting the multiplicity evidence."""
    report = passing_report()
    del report["marginal_preservation"]["candidate"]["z_probe_count"]
    results = evaluate_gates(report)
    assert {result.gate for result in results if not result.passed} == {"A"}


def test_every_shadow_manifest_declares_itself_non_promotable():
    manifest = ArtifactManifest(artifact_name="probe")
    assert manifest.promotion_eligibility == PROMOTION_ELIGIBILITY
    assert "NOT_ELIGIBLE_FOR_PROMOTION" in PROMOTION_ELIGIBILITY

    for path in sorted((PROJECT / "research" / "game_latent_state").glob("*manifest*.json")):
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["promotion_eligibility"] == PROMOTION_ELIGIBILITY, path
