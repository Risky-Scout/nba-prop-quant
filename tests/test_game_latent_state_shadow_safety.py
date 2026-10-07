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
from nba_prop_quant.research.game_latent_state.safety import (
    DECLARED_INTEGRATION_PATHS,
    PRODUCTION_REF,
    PROTECTED_PRODUCTION_PREFIXES,
    PROTECTED_PRODUCTION_SOURCES,
    modified_production_paths,
    production_merge_base,
    stale_integration_declarations,
    undeclared_production_paths,
)

PROJECT = Path(__file__).resolve().parents[1]

SHADOW_PACKAGE = PROJECT / "src" / "nba_prop_quant" / "research" / "game_latent_state"

SHADOW_SCRIPTS = PROJECT / "research" / "game_latent_state"

#: The protected-path definition is imported rather than restated, so the
#: containment the validation report claims and the containment these tests
#: check cannot drift apart.
PROTECTED_PREFIXES = PROTECTED_PRODUCTION_PREFIXES
PROTECTED_SOURCE_FILES = PROTECTED_PRODUCTION_SOURCES

#: ``safety.py`` is the module that *declares* the protected production paths,
#: so it necessarily contains those path literals. They are a read-only guard
#: list, not write targets, which is why the literal scan below skips it.
PATH_DECLARATION_MODULE = "safety.py"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def production_base() -> str | None:
    return production_merge_base(PROJECT)


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


def test_no_undeclared_protected_production_path_is_modified():
    """Every production path this branch touches is declared, with a reason.

    The absolute form of this check — no protected path changes at all — was
    right while the dependence work was pure research, and it is wrong for the
    branch that integrates it, because integrating is a production change. So
    the enforced invariant is that nothing *undeclared* changes, and the
    declaration is reviewed like any other code.
    """
    offenders = [
        path
        for path in changed_paths()
        if (path.startswith(PROTECTED_PREFIXES) or path in PROTECTED_SOURCE_FILES)
        and path not in DECLARED_INTEGRATION_PATHS
    ]
    assert offenders == [], f"undeclared production paths modified: {offenders}"


def test_the_declared_integration_surface_cannot_reach_the_served_model():
    """Declaring a path is not permission to edit the model it serves.

    This is what stops the declaration mechanism from becoming a general
    exemption: the model artifacts, the frozen configs, the production scripts,
    the release surface and every protected production source module remain
    undeclarable, so the only thing a declaration can buy is documentation and
    the shadow's own operational surface.
    """
    undeclarable_prefixes = ("models/", "configs/", "scripts/", "release/", "review/")
    for path in DECLARED_INTEGRATION_PATHS:
        assert not path.startswith(undeclarable_prefixes), (
            f"{path} is a model, config, production script or release path and "
            "may not be declared"
        )
        assert path not in PROTECTED_PRODUCTION_SOURCES, (
            f"{path} is a protected production source module and may not be declared"
        )
        assert DECLARED_INTEGRATION_PATHS[path].strip(), f"{path} is declared without a reason"


def test_no_declaration_outlives_the_change_it_was_made_for():
    """A declaration left behind after its change landed is loose permission."""
    if production_base() is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")
    if not changed_paths():
        pytest.skip("this head is production, so it changes nothing relative to itself")
    assert stale_integration_declarations(PROJECT) == []


def test_gate_h_evidence_comes_from_the_shared_protected_path_declaration():
    """The validation report's gate H evidence is this same computation.

    The report cannot claim a clean production surface by using a narrower
    definition of "production" than these tests enforce, because both sides
    call the same function over the same declared path lists. Gate H still
    reports every protected path touched, declared or not; what these tests
    enforce is the narrower ``undeclared`` form.
    """
    if production_base() is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")
    assert undeclared_production_paths(PROJECT) == []
    assert set(modified_production_paths(PROJECT)) <= set(DECLARED_INTEGRATION_PATHS)
    assert "src/nba_prop_quant/copula.py" in PROTECTED_PRODUCTION_SOURCES
    assert ".github/workflows/" in PROTECTED_PRODUCTION_PREFIXES


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


#: Names in the shadow namespace that read as a promotion or publishing entry
#: point and are in fact refusals. Every one of them must raise on every
#: reachable input, which ``test_every_exempt_name_is_a_refusal_not_a_path``
#: checks by calling them rather than by reading their names.
#:
#: The publication guards joined this list when the controlled shadow was
#: built: the brief requires a *declared, disabled* publishing switch and a
#: prepared fallback, and a declared switch has to be nameable. A name scan
#: cannot tell a switch that refuses from a switch that publishes, so the
#: exemption is paired with the behavioural test below.
REFUSAL_GUARD_NAMES: frozenset[str] = frozenset(
    {
        "assert_promotable",
        "ShadowPromotionRefused",
        "assert_no_promotion_authority",
        "publish_shadow_probabilities",
        "read_publishing_switch",
        "PublishingSwitch",
        "ShadowPublishingDisabled",
    }
)


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
                if node.name in REFUSAL_GUARD_NAMES:
                    continue
                offenders.append(f"{path.relative_to(PROJECT)}::{node.name}")
    assert offenders == []


def test_every_exempt_name_is_a_refusal_not_a_path():
    """The exemption list is checked by behaviour, not taken on trust.

    Each exempt name is called. A promotion guard must raise for every input.
    A publication guard must refuse for every switch state, including the
    state where every declared activation condition is satisfied -- otherwise
    the exemption would be a hole rather than a guard.
    """
    from nba_prop_quant.research.game_latent_state import shadow_runtime as runtime

    passing = evaluate_gates(passing_report())
    assert shadow_verdict(passing) == VERDICT_ACCEPTED
    with pytest.raises(ShadowPromotionRefused):
        assert_promotable(passing)

    for context in ("", "an operator with a reason", "ci"):
        with pytest.raises(runtime.ShadowPromotionRefused):
            runtime.assert_no_promotion_authority(context)

    # A missing switch, a disabled switch and a fully enabled switch all end
    # in a refusal. There is no fourth state.
    for switch in (
        runtime.read_publishing_switch(None, environment={}),
        runtime.PublishingSwitch(state=runtime.PUBLISHING_DISABLED, reason="test"),
        runtime.PublishingSwitch(
            state=runtime.PUBLISHING_ENABLED,
            reason="test",
            approval_token="token",
            environment_agrees=True,
            approval_supplied=True,
        ),
    ):
        with pytest.raises(runtime.ShadowPublishingDisabled):
            runtime.publish_shadow_probabilities([], switch)

    # And the exemption list does not name anything that no longer exists.
    declared = {
        node.name
        for path in python_sources()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert REFUSAL_GUARD_NAMES <= declared


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


#: The real WizardOfOdds publishing surface: the bundle builder script and its
#: entry point. Banning these by name is precise, because these are the things
#: that actually publish.
#:
#: ``docs/wizardofodds`` was on this list and has been removed, because it was
#: wrong: that directory is documentation — the claim policy and the
#: automation runbooks — and the bundle builder never reads or writes it. What
#: the list is for is reaching the publisher. What stops the shadow *writing*
#: production documentation is the declared-path check above, and
#: :func:`test_no_shadow_source_writes_to_a_protected_production_path` below.
WIZARDOFODDS_SURFACE_TOKENS: tuple[str, ...] = (
    "wizardofodds_bundle",
    "runtime_bundle",
    "build_runtime_bundle",
    "19_build_wizardofodds_runtime_bundle",
)

#: Calls that put bytes on disk.
WRITE_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "write_text",
        "write_bytes",
        "mkdir",
        "to_parquet",
        "to_csv",
        "to_json",
        "touch",
        "unlink",
        "rmdir",
        "replace",
    }
)


def test_shadow_package_contains_no_wizardofodds_publishing_surface():
    offenders = [
        f"{path.relative_to(PROJECT)}: {token}"
        for path in python_sources()
        for token in WIZARDOFODDS_SURFACE_TOKENS
        if token in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_no_shadow_source_writes_to_a_protected_production_path():
    """No shadow source names a protected production path in a write.

    The token ban above is about reaching the publisher. This is about reaching
    production bytes by any route: a string literal under a protected prefix
    appearing anywhere near a write call is reported, whether or not the
    publisher is involved. The declared documentation path is the one thing
    the integration is allowed to add, and it is added by a human edit in a
    reviewed commit, never by shadow code at runtime.
    """
    offenders: list[str] = []
    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            is_write = (
                isinstance(target, ast.Attribute) and target.attr in WRITE_ATTRIBUTES
            ) or (isinstance(target, ast.Name) and target.id == "open")
            if not is_write:
                continue
            literals = [
                sub.value
                for sub in ast.walk(node)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            ]
            for literal in literals:
                if literal.startswith(PROTECTED_PREFIXES):
                    offenders.append(f"{path.relative_to(PROJECT)}: writes {literal}")
    assert offenders == [], f"shadow code writes production paths: {offenders}"


def test_the_publishing_surface_cannot_reach_the_shadow_candidate():
    """The bundle builder must not import the shadow package.

    Checked from the other direction: even if the shadow never names the
    publisher, the publisher importing the shadow would put candidate
    probabilities one call away from a published bundle.
    """
    builder = PROJECT / "scripts" / "19_build_wizardofodds_runtime_bundle.py"
    tree = ast.parse(builder.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    offenders = [name for name in imported if "game_latent_state" in name]
    assert offenders == [], f"the publishing surface imports the shadow: {offenders}"


def test_the_shadow_publication_switch_is_a_declared_disabled_file():
    """The one publication-shaped thing the shadow may own, pinned.

    This replaces a blanket ban on the token ``SHADOW_PUBLISH``. That ban was
    a proxy for "no environment variable can turn publishing on", and it
    stopped being the right check once the brief required a declared switch.
    The property is checked directly instead, which is strictly stronger than
    the token scan: a rename could evade a token, but nothing can evade the
    requirements that the committed state is disabled, that enabling needs
    three independent conditions, and that the entry point refuses anyway.
    """
    import json

    from nba_prop_quant.research.game_latent_state import shadow_runtime as runtime

    switch_path = PROJECT / runtime.PUBLISHING_SWITCH_PATH
    assert switch_path.exists()
    payload = json.loads(switch_path.read_text(encoding="utf-8"))
    assert payload["state"] == runtime.PUBLISHING_DISABLED
    assert payload["published_authority"] == "incumbent"
    assert len(payload["what_enabling_requires"]) == 3

    # Only the shadow runtime may carry the switch's environment variables,
    # and no shadow *script* may read them: activation cannot be a side effect
    # of running a research driver.
    bearers = {
        path.relative_to(PROJECT).as_posix()
        for path in python_sources()
        if runtime.PUBLISH_ENV_VAR in path.read_text(encoding="utf-8")
    }
    assert bearers == {
        "src/nba_prop_quant/research/game_latent_state/shadow_runtime.py"
    }

    # The environment is read in exactly one place, and that place defaults to
    # refusing. No other module may consult os.environ for the switch.
    source = (
        PROJECT
        / "src/nba_prop_quant/research/game_latent_state/shadow_runtime.py"
    ).read_text(encoding="utf-8")
    assert source.count("os.environ") == 1


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
        if path.name == PATH_DECLARATION_MODULE:
            continue
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
                "max_abs_variance_z": 3.9,
                "z_probe_count": 800_000,
                "three_sigma_exceedance_fraction": 0.0030,
                "max_abs_over_probability_error": 0.004,
                "max_abs_variance_relative_error_well_conditioned": 0.01,
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
            "A",
            lambda r: r["marginal_preservation"]["candidate"].update(
                max_abs_variance_z=9.0
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
