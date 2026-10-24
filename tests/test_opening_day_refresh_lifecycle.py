"""Opening day reaches serving, and nothing else widens with it.

The defect this suite pins: the readiness preflight reported opening day as a
``SKIP_REFRESH`` exit 10 — the same code as a day whose prerequisite games had
not settled yet — while the classifier admitted only exit 0 and a verified
exit 20. So the one date on which "nothing was refreshed" and "a real slate
exists" are both true classified as ``REFRESH_FAILED``, the refresh step
failed closed, and every step behind it was skipped. The first slate of the
2026-27 season would never have been served or published.

The fix gives opening day its own exit code and its own verified outcome, and
that is all it gives it. The tests below hold the three questions apart:

* the day **before** opening day is still a preseason no-op with no serving
* opening day is a no-op the lifecycle **continues** through, and serves
* every other nonzero refresh code still fails closed

Nothing here fits, prices, promotes, or contacts an external service. The last
two tests are the containment claim: no numerical module and no line of the
lifecycle workflow moved.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nba_prop_quant.research.game_latent_state.safety import (
    ADDITIVE_ONLY_CAPTURE_SELECTION_SOURCES,
    ADDITIVE_ONLY_NAMED_DESTINATIONS,
    ADDITIVE_ONLY_PAGE_CONSISTENCY_CHECKS,
    ADDITIVE_ONLY_PUBLICATION_TRANSPORTS,
    ADDITIVE_ONLY_PUBLISHING_SWITCH,
    ADDITIVE_ONLY_SERVING_ENTRY_POINTS,
    ADDITIVE_ONLY_SERVING_SOURCES,
    ADDITIVE_ONLY_WORKFLOWS,
    CAPTURE_SELECTION_CLASS,
    CAPTURE_SELECTION_EXEMPT_FUNCTIONS,
    CAPTURE_SELECTION_EXEMPT_METHODS,
    DECLARABLE_SERVING_SOURCE_REGISTRIES,
    NUMERICAL_SURFACE_EXEMPT_FUNCTIONS,
    DECLARED_CAPTURE_ISOLATION_NAMES,
    DECLARED_CAPTURE_RESOLUTION_NAMES,
    DECLARED_CAPTURE_SELECTION_DEFINITIONS,
    DECLARED_INTEGRATION_PATHS,
    DECLARED_PAGE_CONSISTENCY_NAMES,
    DECLARED_PUBLICATION_TRANSPORT_NAMES,
    DECLARED_SERVING_EXCLUSION_NAMES,
    DECLARED_SERVING_PLUMBING_NAMES,
    PAGE_CONSISTENCY_CLASS,
    PAGE_CONSISTENCY_EXEMPT_FUNCTIONS,
    PAGE_CONSISTENCY_EXEMPT_METHODS,
    PUBLICATION_TRANSPORT_CLASS,
    PUBLICATION_TRANSPORT_EXEMPT_METHODS,
    SERVING_SOURCE_EXEMPT_FUNCTIONS,
)

PROJECT = Path(__file__).resolve().parents[1]

PRODUCTION_REF = "production/wizardofodds-integration"

LIFECYCLE_WORKFLOW = (
    PROJECT / ".github" / "workflows" / "nba_production_lifecycle.yml"
)

#: The locked season-2026 opening day, and the dates either side of it. Stated
#: here only so the assertions read; every module under test takes it from the
#: preflight's own table, and the first test proves these agree.
OPENING_DAY = "2026-10-20"
DAY_BEFORE_OPENING = "2026-10-19"
ORDINARY_IN_SEASON_DAY = "2026-11-15"

SEASON = 2026


def load_ops(name: str):
    spec = importlib.util.spec_from_file_location(
        f"opening_day_ops_{name}", PROJECT / "ops" / f"{name}.py"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


classifier = load_ops("classify_refresh_outcome")

refresh_wrapper = load_ops("refresh_current_season_state")

serving = load_ops("run_incumbent_production_serving")

summary = load_ops("summarise_production_run")


def refresh_status(tmp_path: Path, decision: dict) -> Path:
    """One classifier decision on disk, exactly as the workflow writes it."""
    path = tmp_path / f"refresh_{decision.get('outcome')}.json"

    path.write_text(json.dumps(decision, sort_keys=True), encoding="utf-8")

    return path


def classify(code: int, slate_date: str) -> dict:
    return classifier.classify(refresh_exit_code=code, slate_date=slate_date)


# ----------------------------------------------------------------------
# the opening-day contract is owned upstream, not restated
# ----------------------------------------------------------------------


def test_the_opening_day_code_and_date_are_read_from_the_preflight():
    """Neither the code nor the date may be a second opinion."""
    preflight_source = (
        PROJECT / "ops" / "preflight_current_season_refresh.py"
    ).read_text(encoding="utf-8")

    classifier_source = (
        PROJECT / "ops" / "classify_refresh_outcome.py"
    ).read_text(encoding="utf-8")

    assert classifier.load_opening_day_table()[SEASON] == OPENING_DAY

    code = classifier.load_opening_day_exit_code()

    assert f"OPENING_DAY_NO_REFRESH_CODE = {code}" in preflight_source

    # Neither the date nor the code is written down in the classifier.
    assert OPENING_DAY not in classifier_source
    assert f"= {code}" not in classifier_source

    # And the wrapper propagates it rather than reinterpreting it.
    assert code in refresh_wrapper.PREFLIGHT_STOP_CODES


def test_the_opening_day_code_is_distinct_from_every_other_stop_code():
    """A shared code is what made the two no-ops indistinguishable."""
    code = classifier.load_opening_day_exit_code()

    assert code != classifier.EXIT_OK
    assert code != classifier.REFRESH_EXIT_PRESEASON_BLOCK


# ----------------------------------------------------------------------
# A. the day before opening day: preseason no-op, and no serving
# ----------------------------------------------------------------------


def test_day_before_opening_is_a_preseason_no_op(tmp_path: Path):
    decision = classify(
        classifier.REFRESH_EXIT_PRESEASON_BLOCK, DAY_BEFORE_OPENING
    )

    assert decision["outcome"] == classifier.OUTCOME_PRESEASON_BLOCK
    assert decision["verified"] is True
    assert decision["lifecycle_status"] == "success"
    assert decision["writes_performed"] is False
    assert decision["run_adaptive_fit"] is False
    assert classifier.exit_code_for(decision) == 0

    # The lifecycle continues, but there is nothing to serve.
    readiness = serving.slate_readiness(refresh_status(tmp_path, decision))

    assert readiness["ready"] is False
    assert readiness["refresh_outcome"] == (
        classifier.OUTCOME_PRESEASON_BLOCK
    )


# ----------------------------------------------------------------------
# B. opening day: a no-op the lifecycle continues through, and serves
# ----------------------------------------------------------------------


def test_opening_day_is_its_own_verified_outcome():
    decision = classify(
        classifier.load_opening_day_exit_code(), OPENING_DAY
    )

    assert decision["outcome"] == (
        classifier.OUTCOME_OPENING_DAY_NO_REFRESH
    )
    assert decision["verified"] is True
    assert decision["opening_day"] == OPENING_DAY
    assert decision["slate_date"] == OPENING_DAY
    assert decision["season"] == SEASON

    # Nothing was written, and nothing is fitted: there is no completed
    # current-season game to fit yet.
    assert decision["writes_performed"] is False
    assert decision["run_adaptive_fit"] is False


def test_opening_day_lets_the_workflow_continue():
    """The exact production failure: the refresh step must not go red."""
    decision = classify(
        classifier.load_opening_day_exit_code(), OPENING_DAY
    )

    assert decision["lifecycle_status"] == "success"
    assert classifier.exit_code_for(decision) == 0

    environment = classifier.github_env_lines(decision)

    assert "REFRESH_OUTCOME=OPENING_DAY_NO_REFRESH" in environment
    assert "RUN_ADAPTIVE=false" in environment


def test_opening_day_serving_readiness_is_true(tmp_path: Path):
    """The point of the fix: the incumbent prices the season's first slate."""
    decision = classify(
        classifier.load_opening_day_exit_code(), OPENING_DAY
    )

    readiness = serving.slate_readiness(refresh_status(tmp_path, decision))

    assert readiness["ready"] is True
    assert readiness["refresh_outcome"] == (
        classifier.OUTCOME_OPENING_DAY_NO_REFRESH
    )


def test_opening_day_is_reported_as_a_success_with_no_adaptive_stage():
    """A skipped adaptive fit on opening day is intended, not a failure."""
    decision = classify(
        classifier.load_opening_day_exit_code(), OPENING_DAY
    )

    status = summary.build_status(
        {"status": "ok", "blockers": []},
        None,
        "a" * 40,
        "schedule",
        "production",
        refresh=decision,
    )

    assert status["failure_stage"] is None
    assert status["refresh_outcome"] == (
        classifier.OUTCOME_OPENING_DAY_NO_REFRESH
    )
    assert "opening day" in status["refresh_outcome_detail"]
    assert "opening day" in status["adaptive_action_detail"]

    assert "successful outcome" in summary.render(status)


# ----------------------------------------------------------------------
# C. an ordinary in-season day is untouched
# ----------------------------------------------------------------------


def test_ordinary_in_season_day_still_refreshes_and_serves(tmp_path: Path):
    decision = classify(classifier.EXIT_OK, ORDINARY_IN_SEASON_DAY)

    assert decision["outcome"] == classifier.OUTCOME_REFRESHED
    assert decision["lifecycle_status"] == "success"
    assert decision["run_adaptive_fit"] is True
    assert decision["writes_performed"] is True
    assert classifier.exit_code_for(decision) == 0

    readiness = serving.slate_readiness(refresh_status(tmp_path, decision))

    assert readiness["ready"] is True
    assert readiness["refresh_outcome"] == classifier.OUTCOME_REFRESHED


# ----------------------------------------------------------------------
# D. everything else still fails closed
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "code", [1, 10, 12, 30, 31, 70, 71, 72, 73, 74, 75, 78, 79]
)
def test_every_other_nonzero_refresh_code_still_fails_closed(code: int):
    """Opening day is one code, not a licence for the rest.

    10 is in this list on purpose. It is the code opening day used to leave
    under, and it still means a withheld readiness — a day whose prerequisite
    games have not settled — which must not be served.
    """
    decision = classify(code, OPENING_DAY)

    assert decision["outcome"] == classifier.OUTCOME_REFRESH_FAILED
    assert decision["lifecycle_status"] == "failed"
    assert decision["verified"] is False
    assert decision["run_adaptive_fit"] is False
    assert classifier.exit_code_for(decision) == code


def test_a_failed_refresh_is_not_served(tmp_path: Path):
    decision = classify(10, OPENING_DAY)

    readiness = serving.slate_readiness(refresh_status(tmp_path, decision))

    assert readiness["ready"] is False


# ----------------------------------------------------------------------
# E. the opening-day state cannot be forged for another date
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "slate_date",
    [DAY_BEFORE_OPENING, "2026-10-21", ORDINARY_IN_SEASON_DAY, "2027-10-20"],
)
def test_the_classifier_refuses_the_opening_day_code_on_another_date(
    slate_date: str,
):
    code = classifier.load_opening_day_exit_code()

    decision = classify(code, slate_date)

    assert decision["outcome"] == classifier.OUTCOME_REFRESH_FAILED
    assert decision["lifecycle_status"] == "failed"
    assert decision["verified"] is False
    assert classifier.exit_code_for(decision) == code


def test_the_classifier_refuses_an_unconfigured_season():
    decision = classifier.classify(
        refresh_exit_code=classifier.load_opening_day_exit_code(),
        season=2031,
        slate_date=OPENING_DAY,
    )

    assert decision["lifecycle_status"] == "failed"
    assert "no locked opening day" in decision["detail"]


@pytest.mark.parametrize("slate_date", ["", "not-a-date", "20261020"])
def test_the_classifier_refuses_an_unparseable_slate_date(slate_date: str):
    decision = classify(
        classifier.load_opening_day_exit_code(), slate_date
    )

    assert decision["lifecycle_status"] == "failed"
    assert "must be exact YYYY-MM-DD" in decision["detail"]


def test_serving_refuses_an_opening_day_claim_with_no_evidence(
    tmp_path: Path,
):
    """The outcome name is not the evidence."""
    with pytest.raises(serving.ServingRefused) as caught:
        serving.slate_readiness(
            refresh_status(
                tmp_path,
                {"outcome": classifier.OUTCOME_OPENING_DAY_NO_REFRESH},
            )
        )

    assert "without the evidence" in str(caught.value)


@pytest.mark.parametrize(
    "missing", ["verified", "writes_performed", "opening_day", "season"]
)
def test_serving_refuses_an_opening_day_claim_missing_one_field(
    tmp_path: Path, missing: str
):
    decision = classify(
        classifier.load_opening_day_exit_code(), OPENING_DAY
    )

    del decision[missing]

    with pytest.raises(serving.ServingRefused):
        serving.slate_readiness(refresh_status(tmp_path, decision))


def test_serving_refuses_a_self_contradictory_opening_day_claim(
    tmp_path: Path,
):
    decision = classify(
        classifier.load_opening_day_exit_code(), OPENING_DAY
    )

    decision["slate_date"] = ORDINARY_IN_SEASON_DAY

    with pytest.raises(serving.ServingRefused) as caught:
        serving.slate_readiness(refresh_status(tmp_path, decision))

    assert "contradicts itself" in str(caught.value)


def test_serving_re_derives_the_opening_day_rather_than_trusting_it(
    tmp_path: Path,
):
    """A forged claim that is internally consistent is still refused."""
    forged = {
        "opening_day": ORDINARY_IN_SEASON_DAY,
        "outcome": classifier.OUTCOME_OPENING_DAY_NO_REFRESH,
        "season": SEASON,
        "slate_date": ORDINARY_IN_SEASON_DAY,
        "verified": True,
        "writes_performed": False,
    }

    with pytest.raises(serving.ServingRefused) as caught:
        serving.slate_readiness(refresh_status(tmp_path, forged))

    assert "could not be re-derived" in str(caught.value)


# ----------------------------------------------------------------------
# E2. readiness is about one date, and it has to be this one
# ----------------------------------------------------------------------
#
# The serving run knows which slate it was told to price; the status file
# records which slate its readiness is about. In the scheduled path those
# agree by construction, because the refresh step computes the date once and
# hands the same value to the classifier. That is an invariant of one
# workflow, not a property of this contract, so serving compares the two
# itself. What it defends against is everything that is not the scheduled
# path: a stale status file, one reused across days, a manual invocation
# given the wrong one, and a future workflow change that breaks the invariant
# without breaking anything visibly.


def narrow(decision: dict, tmp_path: Path, requested: str) -> dict:
    """Readiness for ``requested``, decided from ``decision`` on disk."""
    status = refresh_status(tmp_path, decision)

    return serving.readiness_for_this_slate(
        serving.slate_readiness(status),
        refresh_status=status,
        slate_date=requested,
    )


@pytest.mark.parametrize(
    ("status_slate", "requested", "code"),
    [
        # A. the run serves opening day; the status file is about the day after
        (ORDINARY_IN_SEASON_DAY, OPENING_DAY, classifier.EXIT_OK),
        ("2026-10-21", OPENING_DAY, classifier.EXIT_OK),
        # B. the run serves the day after; the status file is about opening day
        (OPENING_DAY, "2026-10-21", None),
        # E. internally valid, and from another day
        (ORDINARY_IN_SEASON_DAY, "2026-11-16", classifier.EXIT_OK),
        ("2026-11-16", ORDINARY_IN_SEASON_DAY, classifier.EXIT_OK),
    ],
)
def test_a_status_file_about_another_date_is_refused(
    tmp_path: Path, status_slate: str, requested: str, code: int | None
):
    """A and B and E: readiness for one date does not license another."""
    decision = classify(
        classifier.load_opening_day_exit_code() if code is None else code,
        status_slate,
    )

    # The decision itself is sound. Only the date it is about is wrong.
    assert decision["lifecycle_status"] == "success"
    assert decision["slate_date"] == status_slate

    with pytest.raises(serving.ServingRefused) as caught:
        narrow(decision, tmp_path, requested)

    assert "does not describe this slate" in str(caught.value)


def test_an_opening_day_status_copied_to_a_later_date_is_refused(
    tmp_path: Path,
):
    """C. The opening-day outcome is the one most worth copying forward.

    It is the only verified outcome under which nothing was refreshed and a
    slate is served, so a status file carrying it is exactly the artifact a
    cross-day reuse would reach for. The claim verifies -- it is a real
    opening-day decision -- and the date it is about is not the date being
    served.
    """
    decision = classify(classifier.load_opening_day_exit_code(), OPENING_DAY)

    assert decision["outcome"] == classifier.OUTCOME_OPENING_DAY_NO_REFRESH
    assert decision["verified"] is True

    for requested in ("2026-10-21", ORDINARY_IN_SEASON_DAY, "2027-10-20"):
        with pytest.raises(serving.ServingRefused):
            narrow(decision, tmp_path, requested)


@pytest.mark.parametrize("outcome_code", [classifier.EXIT_OK, None])
def test_a_status_file_with_no_slate_date_is_refused(
    tmp_path: Path, outcome_code: int | None
):
    """D. A readiness that does not say which date it is about settles nothing.

    Both servable outcomes, because the refreshed case has the same exposure.
    """
    code = (
        classifier.load_opening_day_exit_code()
        if outcome_code is None
        else outcome_code
    )

    decision = classify(code, OPENING_DAY)

    del decision["slate_date"]

    with pytest.raises(serving.ServingRefused):
        narrow(decision, tmp_path, OPENING_DAY)


def test_the_matching_date_is_the_only_one_that_serves(tmp_path: Path):
    """The check is an equality, so the correct date still passes."""
    for code, slate in (
        (classifier.EXIT_OK, ORDINARY_IN_SEASON_DAY),
        (classifier.load_opening_day_exit_code(), OPENING_DAY),
    ):
        decision = classify(code, slate)

        assert narrow(decision, tmp_path, slate)["ready"] is True


def test_nothing_in_the_date_check_can_turn_a_refusal_into_a_slate(
    tmp_path: Path,
):
    """A refused outcome is returned as it arrived, whatever the dates say."""
    for code, slate in (
        (classifier.REFRESH_EXIT_PRESEASON_BLOCK, DAY_BEFORE_OPENING),
        (10, OPENING_DAY),
    ):
        decision = classify(code, slate)

        for requested in (slate, OPENING_DAY, ORDINARY_IN_SEASON_DAY):
            assert narrow(decision, tmp_path, requested)["ready"] is False


# ----------------------------------------------------------------------
# F and G. containment
# ----------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def blob_text(ref: str, relative: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{relative}"],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def production_base() -> str | None:
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return git("merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue

    return None


#: Everything that decides a number a reader is shown. This fix is a state
#: transition between lifecycle stages; none of these may move for it.
NUMERICAL_MODULES: tuple[str, ...] = (
    "src/nba_prop_quant/copula.py",
    "src/nba_prop_quant/decay.py",
    "src/nba_prop_quant/distributions.py",
    "src/nba_prop_quant/experience.py",
    "src/nba_prop_quant/features.py",
    "src/nba_prop_quant/game_context.py",
    "src/nba_prop_quant/gate3_v2.py",
    "src/nba_prop_quant/kalman.py",
    "src/nba_prop_quant/model.py",
    "src/nba_prop_quant/pricing.py",
    "src/nba_prop_quant/production.py",
    "src/nba_prop_quant/prospective_snapshot.py",
    "scripts/10_predict_slate.py",
    "scripts/15_price_markets.py",
    "src/nba_prop_quant/slate.py",
    "ops/publish_incumbent_to_wizardofodds.py",
    "ops/nba_distribution_publication.py",
    "ops/wizardofodds_player_props_page.py",
    "ops/wizardofodds_publishing_switch.json",
    "ops/wizardofodds_destinations.json",
)


def top_level(source: str) -> dict[str, ast.AST]:
    """Top-level functions and classes by name, and nothing nested."""
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def module_constants(source: str) -> dict[str, ast.AST]:
    """Top-level assignments by name, which is where this module's rules are.

    The published-record versions, the row limits, the timeouts, the relative
    paths and the mode names are all module constants, so pinning these is
    pinning every number and every path the module states.
    """
    constants: dict[str, ast.AST] = {}

    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            constants[node.target.id] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node

    return constants


def class_methods(source: str, name: str) -> dict[str, ast.AST]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {
                child.name: child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

    return {}


def assert_only_the_publication_transport_moved(
    relative: str, before: str, after: str
) -> None:
    """Nothing that states a number, a schema or a rule may differ.

    Narrower than the byte pin in the direction the pin exists for, and wider
    only in the one the transport needs. The module's rules -- which rows are
    publishable, what a published record contains, which destination a mode
    may reach, every version and every threshold -- are module constants and
    top-level functions, and all of them are held syntax-tree-identical to
    production here. Only the remote transport's class may differ, only in
    the method that invokes rsync, and only by gaining names that were
    declared for it.
    """
    definitions_before = top_level(before)
    definitions_after = top_level(after)

    assert set(definitions_before) <= set(definitions_after), (
        f"{relative} dropped "
        f"{sorted(set(definitions_before) - set(definitions_after))}"
    )

    for name, node in definitions_before.items():
        if name == PUBLICATION_TRANSPORT_CLASS:
            continue

        assert ast.dump(node) == ast.dump(definitions_after[name]), (
            f"{relative}:{name} is not what production has, and only "
            f"{PUBLICATION_TRANSPORT_CLASS} may differ"
        )

    constants_before = module_constants(before)
    constants_after = module_constants(after)

    assert set(constants_before) <= set(constants_after), (
        f"{relative} dropped "
        f"{sorted(set(constants_before) - set(constants_after))}"
    )

    for name, node in constants_before.items():
        assert ast.dump(node) == ast.dump(constants_after[name]), (
            f"{relative}:{name} is a module constant and may not move"
        )

    methods_before = class_methods(before, PUBLICATION_TRANSPORT_CLASS)
    methods_after = class_methods(after, PUBLICATION_TRANSPORT_CLASS)

    assert methods_before, f"{relative} has no {PUBLICATION_TRANSPORT_CLASS}"

    assert set(methods_before) <= set(methods_after), (
        f"{relative} dropped "
        f"{sorted(set(methods_before) - set(methods_after))}"
    )

    for name, node in methods_before.items():
        if name in PUBLICATION_TRANSPORT_EXEMPT_METHODS:
            continue

        assert ast.dump(node) == ast.dump(methods_after[name]), (
            f"{relative}:{PUBLICATION_TRANSPORT_CLASS}.{name} is not the "
            "method production has, and only "
            f"{list(PUBLICATION_TRANSPORT_EXEMPT_METHODS)} may differ"
        )

    added = (
        (set(definitions_after) - set(definitions_before))
        | (set(constants_after) - set(constants_before))
        | (set(methods_after) - set(methods_before))
    )

    assert added <= set(DECLARED_PUBLICATION_TRANSPORT_NAMES), (
        f"{relative} added {sorted(added - set(DECLARED_PUBLICATION_TRANSPORT_NAMES))}, "
        "which is not declared publication transport plumbing"
    )


class StripDeclaredPlumbing(ast.NodeTransformer):
    """A tree with the declared names' binding and hand-off taken back out.

    Used to state what a function gained rather than to exempt it from being
    compared at all. A binding of a declared name is unwrapped back to the
    bare expression it wraps, and a keyword argument named for a declared name
    is dropped; if what is left is production's own tree, then the declared
    plumbing is the whole of what the function gained and no second change
    rode along inside it.
    """

    def __init__(self, declared: tuple[str, ...]) -> None:
        self.declared = frozenset(declared)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)

        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.declared
        ):
            return ast.Expr(value=node.value)

        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        node.keywords = [
            keyword
            for keyword in node.keywords
            if keyword.arg not in self.declared
        ]

        return node


def class_fields(source: str, name: str) -> dict[str, ast.AST]:
    """A class's own annotated attributes by name, which for a dataclass is its fields."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {
                child.target.id: child
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            }

    return {}


def assert_only_the_page_consistency_check_moved(
    relative: str, before: str, after: str
) -> None:
    """Nothing the renderer states about a number, a format or a route may differ.

    Narrower than the byte pin in the direction the pin exists for, and wider
    only in the one the agreement check needs. The odds conversions, the
    percentage formatting, the column set, the cell builders, the provenance
    allow-list, the confinement rules and the routing call are top-level
    definitions and module constants, and all of them are held
    syntax-tree-identical to production here.

    Only the agreement check's own body may differ, and only one accessor of
    the card may. The function that aggregates rows into cards is not exempt:
    it is compared against production's tree with the declared plumbing
    stripped back out, so it may have gained the binding of the settled value
    and the passing of it to the card, and nothing else.
    """
    strip = StripDeclaredPlumbing(DECLARED_PAGE_CONSISTENCY_NAMES)

    definitions_before = top_level(before)
    definitions_after = top_level(after)

    assert set(definitions_before) <= set(definitions_after), (
        f"{relative} dropped "
        f"{sorted(set(definitions_before) - set(definitions_after))}"
    )

    for name, node in definitions_before.items():
        if name in PAGE_CONSISTENCY_EXEMPT_FUNCTIONS:
            continue

        if name == PAGE_CONSISTENCY_CLASS:
            continue

        stripped = strip.visit(definitions_after[name])

        assert ast.dump(node) == ast.dump(stripped), (
            f"{relative}:{name} is not what production has once the declared "
            f"names are removed, and only "
            f"{list(PAGE_CONSISTENCY_EXEMPT_FUNCTIONS)} may differ"
        )

    constants_before = module_constants(before)
    constants_after = module_constants(after)

    assert set(constants_before) <= set(constants_after), (
        f"{relative} dropped "
        f"{sorted(set(constants_before) - set(constants_after))}"
    )

    for name, node in constants_before.items():
        assert ast.dump(node) == ast.dump(constants_after[name]), (
            f"{relative}:{name} is a module constant and may not move"
        )

    methods_before = class_methods(before, PAGE_CONSISTENCY_CLASS)
    methods_after = class_methods(after, PAGE_CONSISTENCY_CLASS)

    assert methods_before, f"{relative} has no {PAGE_CONSISTENCY_CLASS}"

    assert set(methods_before) <= set(methods_after), (
        f"{relative} dropped "
        f"{sorted(set(methods_before) - set(methods_after))}"
    )

    for name, node in methods_before.items():
        if name in PAGE_CONSISTENCY_EXEMPT_METHODS:
            continue

        assert ast.dump(node) == ast.dump(methods_after[name]), (
            f"{relative}:{PAGE_CONSISTENCY_CLASS}.{name} is not the member "
            "production has, and only "
            f"{list(PAGE_CONSISTENCY_EXEMPT_METHODS)} may differ"
        )

    fields_before = class_fields(before, PAGE_CONSISTENCY_CLASS)
    fields_after = class_fields(after, PAGE_CONSISTENCY_CLASS)

    assert set(fields_before) <= set(fields_after), (
        f"{relative} dropped "
        f"{sorted(set(fields_before) - set(fields_after))}"
    )

    for name, node in fields_before.items():
        assert ast.dump(node) == ast.dump(fields_after[name]), (
            f"{relative}:{PAGE_CONSISTENCY_CLASS}.{name} is a published field "
            "the card reads and may not move"
        )

    added = (
        (set(definitions_after) - set(definitions_before))
        | (set(constants_after) - set(constants_before))
        | (set(methods_after) - set(methods_before))
        | (set(fields_after) - set(fields_before))
    )

    assert added <= set(DECLARED_PAGE_CONSISTENCY_NAMES), (
        f"{relative} added {sorted(added - set(DECLARED_PAGE_CONSISTENCY_NAMES))}, "
        "which is not declared page consistency plumbing"
    )


def assert_only_the_slate_builder_moved(
    relative: str, before: str, after: str
) -> None:
    """Nothing the live builder computes may differ except its population.

    The slate module's byte pin protects a different thing from the other two
    on this list: not what a published record says but what the fitted models
    are shown. It is restated rather than lifted in the only direction a
    serving-parity fix needs, which is the direction the carve-out in
    ``safety`` already defines -- ``build_upcoming_slate_features`` is the one
    function whose body may differ, every leakage guard, slate-date rule,
    prior, rolling window and classifier call in the module is a separate
    top-level definition and is held syntax-tree-identical to production
    here, every module constant is too, and what the exempt function may have
    gained is held to the declared plumbing names.

    That is strictly more than this file's byte comparison asserted, which
    said nothing about content at all once a file was allowed to differ.
    """
    removed = [
        line for line in before.splitlines() if line not in after.splitlines()
    ]

    assert removed == [], (
        f"{relative} removed or edited lines production has: {removed}"
    )

    definitions_before = top_level(before)
    definitions_after = top_level(after)

    assert set(definitions_before) <= set(definitions_after), (
        f"{relative} dropped "
        f"{sorted(set(definitions_before) - set(definitions_after))}"
    )

    for name, node in definitions_before.items():
        if name in SERVING_SOURCE_EXEMPT_FUNCTIONS:
            continue

        assert ast.dump(node) == ast.dump(definitions_after[name]), (
            f"{relative}:{name} is not what production has, and only "
            f"{list(SERVING_SOURCE_EXEMPT_FUNCTIONS)} may differ"
        )

    constants_before = module_constants(before)
    constants_after = module_constants(after)

    assert set(constants_before) <= set(constants_after), (
        f"{relative} dropped "
        f"{sorted(set(constants_before) - set(constants_after))}"
    )

    for name, node in constants_before.items():
        assert ast.dump(node) == ast.dump(constants_after[name]), (
            f"{relative}:{name} is a module constant and may not move"
        )

    for name in SERVING_SOURCE_EXEMPT_FUNCTIONS:
        if name not in definitions_before:
            continue

        statements = {
            ast.dump(node) for node in definitions_before[name].body
        }

        for node in definitions_after[name].body:
            rendered = ast.dump(node)

            if rendered in statements:
                continue

            assert any(
                declared in rendered
                for declared in DECLARED_SERVING_PLUMBING_NAMES
            ), (
                f"{relative}:{name} added a statement that is not declared "
                f"serving plumbing: {ast.unparse(node)[:160]}"
            )


def assert_only_the_capture_selection_moved(
    relative: str, before: str, after: str
) -> None:
    """Nothing about what a capture *is* may differ, only how it is read.

    The snapshot module's byte pin protects a third thing again: not what a
    published record says and not what the fitted models are shown, but which
    point-in-time capture pricing is allowed to read at all. It is restated
    rather than lifted in the one direction the carve-out in ``safety``
    defines -- ``select_capture_bundle`` is the one definition whose body may
    differ, and in particular ``canonical_records_sha256`` and
    ``build_capture_id`` are separate top-level definitions and are held
    syntax-tree-identical here, which is what keeps the addressing of every
    capture ever written out of a change to the reading of the log. Every
    module constant, the component specification among them, is pinned too,
    and what the exempt function may have gained is held to the declared
    resolution names.

    That is strictly more than this file's byte comparison asserted, which
    said nothing about content at all once a file was allowed to differ.

    The second defect in this module is in a method rather than a function:
    the client assembled the whole slate in one comprehension, so one game's
    refusal aborted every game's. A class compared whole cannot say "this
    one method may differ" -- it reads the isolation and a rewritten
    accessor the same way -- so the client is compared member by member in
    the shape the page-consistency pin above already uses, ``__init__`` is
    the only member exempt, and a definition the module gained has to name
    something declared.
    """
    removed = [
        line for line in before.splitlines() if line not in after.splitlines()
    ]

    assert removed == [], (
        f"{relative} removed or edited lines production has: {removed}"
    )

    definitions_before = top_level(before)
    definitions_after = top_level(after)

    assert set(definitions_before) <= set(definitions_after), (
        f"{relative} dropped "
        f"{sorted(set(definitions_before) - set(definitions_after))}"
    )

    for name, node in definitions_before.items():
        if name in CAPTURE_SELECTION_EXEMPT_FUNCTIONS:
            continue

        if name == CAPTURE_SELECTION_CLASS:
            continue

        assert ast.dump(node) == ast.dump(definitions_after[name]), (
            f"{relative}:{name} is not what production has, and only "
            f"{list(CAPTURE_SELECTION_EXEMPT_FUNCTIONS)} may differ"
        )

    methods_before = class_methods(before, CAPTURE_SELECTION_CLASS)
    methods_after = class_methods(after, CAPTURE_SELECTION_CLASS)

    assert methods_before, f"{relative} has no {CAPTURE_SELECTION_CLASS}"

    assert set(methods_before) <= set(methods_after), (
        f"{relative} dropped "
        f"{sorted(set(methods_before) - set(methods_after))}"
    )

    for name, node in methods_before.items():
        if name in CAPTURE_SELECTION_EXEMPT_METHODS:
            continue

        assert ast.dump(node) == ast.dump(methods_after[name]), (
            f"{relative}:{CAPTURE_SELECTION_CLASS}.{name} is not the member "
            "production has, and only "
            f"{list(CAPTURE_SELECTION_EXEMPT_METHODS)} may differ"
        )

    # A definition the module gained is as much a change as one it edited,
    # and the line-set rule above cannot see it: adding is all that rule
    # permits. So the additions are enumerated by exact name.
    gained = sorted(
        (set(definitions_after) - set(definitions_before))
        | (set(methods_after) - set(methods_before))
    )

    assert set(gained) <= set(DECLARED_CAPTURE_SELECTION_DEFINITIONS), (
        f"{relative} gained "
        f"{sorted(set(gained) - set(DECLARED_CAPTURE_SELECTION_DEFINITIONS))}"
        ", which nothing declares"
    )

    constants_before = module_constants(before)
    constants_after = module_constants(after)

    assert set(constants_before) <= set(constants_after), (
        f"{relative} dropped "
        f"{sorted(set(constants_before) - set(constants_after))}"
    )

    for name, node in constants_before.items():
        assert ast.dump(node) == ast.dump(constants_after[name]), (
            f"{relative}:{name} is a module constant and may not move"
        )

    # The two exemptions have different subjects and so different declared
    # names. The function may only gain statements about recognising that
    # two records are one capture; the method may only gain statements about
    # which games of the slate a refusal costs.
    for members, before_nodes, after_nodes, declared_names, subject in (
        (
            CAPTURE_SELECTION_EXEMPT_FUNCTIONS,
            definitions_before,
            definitions_after,
            DECLARED_CAPTURE_RESOLUTION_NAMES,
            "capture resolution",
        ),
        (
            CAPTURE_SELECTION_EXEMPT_METHODS,
            methods_before,
            methods_after,
            DECLARED_CAPTURE_ISOLATION_NAMES,
            "capture isolation",
        ),
    ):
        for name in members:
            if name not in before_nodes:
                continue

            statements = {
                ast.dump(node) for node in before_nodes[name].body
            }

            for node in after_nodes[name].body:
                rendered = ast.dump(node)

                if rendered in statements:
                    continue

                assert any(
                    declared in rendered for declared in declared_names
                ), (
                    f"{relative}:{name} added a statement that is not "
                    f"declared {subject}: {ast.unparse(node)[:160]}"
                )


def assert_only_the_serving_entry_point_moved(
    relative: str, before: str, after: str
) -> None:
    """Nothing a serving script computes may differ except what it carries.

    A serving entry point's byte pin protects the numbers the script itself
    computes. It is restated rather than lifted in the one direction the
    carve-out in ``safety`` defines: ``parse_args``, where an argument is
    declared, and ``main``, where it is used, are the only definitions whose
    bodies may differ, every other function -- which is everything that
    prices, projects, verifies a manifest or routes a prop -- is held
    syntax-tree-identical to production here, every module constant is too,
    and what the two exempt functions may have gained is held to the declared
    plumbing names.

    That is strictly more than this file's byte comparison asserted, which
    said nothing about content at all once a file was allowed to differ.
    """
    removed = [
        line for line in before.splitlines() if line not in after.splitlines()
    ]

    assert removed == [], (
        f"{relative} removed or edited lines production has: {removed}"
    )

    definitions_before = top_level(before)
    definitions_after = top_level(after)

    assert set(definitions_before) <= set(definitions_after), (
        f"{relative} dropped "
        f"{sorted(set(definitions_before) - set(definitions_after))}"
    )

    for name, node in definitions_before.items():
        if name in NUMERICAL_SURFACE_EXEMPT_FUNCTIONS:
            continue

        assert ast.dump(node) == ast.dump(definitions_after[name]), (
            f"{relative}:{name} is not what production has, and only "
            f"{list(NUMERICAL_SURFACE_EXEMPT_FUNCTIONS)} may differ"
        )

    constants_before = module_constants(before)
    constants_after = module_constants(after)

    assert set(constants_before) <= set(constants_after), (
        f"{relative} dropped "
        f"{sorted(set(constants_before) - set(constants_after))}"
    )

    for name, node in constants_before.items():
        assert ast.dump(node) == ast.dump(constants_after[name]), (
            f"{relative}:{name} is a module constant and may not move"
        )

    for name in NUMERICAL_SURFACE_EXEMPT_FUNCTIONS:
        if name not in definitions_before:
            continue

        statements = {
            ast.dump(node) for node in definitions_before[name].body
        }

        for node in definitions_after[name].body:
            rendered = ast.dump(node)

            if rendered in statements:
                continue

            # Both lists, because an entry point may gain a quantity it
            # carries forward and -- since the capture-selection carve-out
            # made a short slate possible -- a refusal on a game it could
            # not serve. The serving-source pin above still accepts only
            # the first list.
            assert any(
                declared in rendered
                for declared in (
                    *DECLARED_SERVING_PLUMBING_NAMES,
                    *DECLARED_SERVING_EXCLUSION_NAMES,
                )
            ), (
                f"{relative}:{name} added a statement that is not declared "
                f"serving plumbing: {ast.unparse(node)[:160]}"
            )


def assert_only_named_destinations_were_added(
    relative: str, before: str, after: str
) -> None:
    """Nowhere production publishes to may move; somewhere new may be added.

    The byte pin on this registry is right about every destination in it and
    was reading the same for two different kinds of change: moving a
    destination production already publishes to, and configuring somewhere
    new. Only the first is what "the published record is frozen" is about, and
    conflating them made the file unable to gain a destination at all, which
    is the one thing a new publication stream needs.

    So the pin is restated here in the form that distinguishes them, and it is
    stronger than the byte comparison was once a file was allowed to differ:
    every top-level key and every destination the production ref has must be
    deeply equal, and an added destination must be a canary confined to a
    non-empty subpath, with its own release prefix and a transport this file
    itself declares supported. The two live destinations, the transport
    allow-list and the contract version are therefore unmovable, and a live
    destination cannot be added.
    """
    document_before = json.loads(before)

    document_after = json.loads(after)

    for key, value in document_before.items():
        if key == "named_destinations":
            continue

        assert document_after.get(key) == value, (
            f"{relative}:{key} is not what production has, and only added "
            "named destinations may differ"
        )

    named_before = document_before.get("named_destinations") or {}

    named_after = document_after.get("named_destinations") or {}

    assert set(named_before) <= set(named_after), (
        f"{relative} dropped {sorted(set(named_before) - set(named_after))}"
    )

    for name, declared in named_before.items():
        assert named_after[name] == declared, (
            f"{relative}:{name} is a destination production already "
            "publishes to and may not move"
        )

    supported = set(document_after.get("supported_transports") or [])

    reserved = {
        str(declared.get("release_prefix") or "")
        for declared in named_before.values()
    }

    for name in sorted(set(named_after) - set(named_before)):
        declared = named_after[name]

        assert declared.get("mode") == "canary", (
            f"{relative}:{name} is added as {declared.get('mode')!r}; an "
            "added destination may only be a canary, because a live "
            "destination is an activation decision and not configuration"
        )

        subpath = str(declared.get("canary_subpath") or "")

        assert subpath and not subpath.startswith("/"), (
            f"{relative}:{name} declares canary_subpath {subpath!r}, so it "
            "would resolve at or above the served root rather than confined "
            "below it"
        )

        prefix = str(declared.get("release_prefix") or "")

        assert prefix and prefix not in reserved, (
            f"{relative}:{name} reuses the release prefix {prefix!r}, so its "
            "releases would be indistinguishable from an existing stream's"
        )

        assert declared.get("transport") in supported, (
            f"{relative}:{name} declares the transport "
            f"{declared.get('transport')!r}, which this file does not list "
            "as supported"
        )

        assert int(declared.get("retained_releases") or 0) >= 2, (
            f"{relative}:{name} retains fewer than the two releases a "
            "rollback target needs"
        )


def assert_only_the_live_decision_moved(
    relative: str, before: str, after: str
) -> None:
    """No gate may be weakened; the one decision the file carries may be taken.

    The byte pin on the switch is right about every field in it except the
    three it exists to carry. ``modes.live.state`` is a field whose entire
    purpose is to be set once, by a human, in a reviewed commit, and a pin
    that forbade the file to differ at all made it permanently unable to be
    set -- which is not a freeze on publication but a freeze on the review
    that authorises it.

    So the pin is restated in the form that tells the two apart, and once the
    file is allowed to differ at all this is stronger than the byte
    comparison was: only ``state``, ``approval_token`` and ``reason`` inside
    ``modes.live`` may move, the state may only move towards ENABLED, the
    token may not stay the placeholder, and every other key in the document
    -- the canary mode, the three conditions, the contract version, the
    approved freeze, both schema versions, the published authority and the
    candidate authority -- must be deeply equal to the production ref.

    None of which opens anything. ``PublishingSwitch.enabled`` is the state
    *and* an empty blocker list, and the two approval variables are blockers
    until they are set in the ``wizardofodds-production`` GitHub environment
    and agree with this file, which no commit can do.
    """
    document_before = json.loads(before)

    document_after = json.loads(after)

    for key, value in document_before.items():
        if key == "modes":
            continue

        assert document_after.get(key) == value, (
            f"{relative}:{key} is not what production has, and only the live "
            "decision may differ"
        )

    modes_before = document_before.get("modes") or {}

    modes_after = document_after.get("modes") or {}

    assert set(modes_after) == set(modes_before), (
        f"{relative} changed which modes exist, which is not a decision this "
        "file carries"
    )

    for mode, declared in modes_before.items():
        if mode == "live":
            continue

        assert modes_after[mode] == declared, (
            f"{relative}:modes.{mode} may not move; only the live decision may"
        )

    live_before = modes_before.get("live") or {}

    live_after = modes_after.get("live") or {}

    assert set(live_after) == set(live_before), (
        f"{relative}:modes.live changed which fields it has, and the decision "
        "is taken by setting the three it already had"
    )

    for field, value in live_before.items():
        if field in ("approval_token", "reason", "state"):
            continue

        assert live_after[field] == value, (
            f"{relative}:modes.live.{field} is not a field the decision moves"
        )

    assert live_after["state"] in (live_before["state"], "ENABLED"), (
        f"{relative}:modes.live.state moved to {live_after['state']!r}, and "
        "the only state a reviewed decision sets is ENABLED"
    )

    assert "no-approval-has-been-granted" not in str(
        live_after["approval_token"]
    ), (
        f"{relative}:modes.live.state is {live_after['state']!r} while "
        "approval_token is still the placeholder that says no approval was "
        "granted, so the file would be claiming a decision nobody took"
    )

    assert live_after["approval_token"], (
        f"{relative}:modes.live.approval_token is empty, which an absent "
        "NBA_PROP_WOO_PUBLISH_APPROVAL would match"
    )

    assert len(str(live_after["reason"])) > len(str(live_before["reason"])), (
        f"{relative}:modes.live.reason got shorter while the state moved; the "
        "reason is where the authorisation is recorded"
    )


#: Each declared file on the numerical list, and the stronger statement it is
#: held to instead of its bytes. A module reaches this mapping by having a
#: defect a byte pin makes unfixable while protecting nothing numerical, and
#: it stays on the numerical list: the pin is restated, not lifted.
NARROWED_NUMERICAL_PINS: tuple[tuple[tuple[str, ...], object], ...] = (
    (ADDITIVE_ONLY_PUBLICATION_TRANSPORTS, assert_only_the_publication_transport_moved),
    (ADDITIVE_ONLY_PAGE_CONSISTENCY_CHECKS, assert_only_the_page_consistency_check_moved),
    (ADDITIVE_ONLY_SERVING_SOURCES, assert_only_the_slate_builder_moved),
    (ADDITIVE_ONLY_SERVING_ENTRY_POINTS, assert_only_the_serving_entry_point_moved),
    (ADDITIVE_ONLY_NAMED_DESTINATIONS, assert_only_named_destinations_were_added),
    (ADDITIVE_ONLY_PUBLISHING_SWITCH, assert_only_the_live_decision_moved),
    (
        ADDITIVE_ONLY_CAPTURE_SELECTION_SOURCES,
        assert_only_the_capture_selection_moved,
    ),
)


def test_no_model_pricing_or_calibration_bytes_changed():
    """F. The model, its features, Gate 3, pricing and the PMFs are frozen.

    Byte-identical for every module here except the declared ones, where the
    freeze is restated rather than relaxed. Each is on this list because it
    decides what is publishable, what the fitted models are shown or what a
    published record says, and each also happens to contain a defect in
    machinery that states no number of its own -- the rsync the runner
    refuses, a string comparison of two spellings of one double, which
    player-games a prior averages over, and a column that said a figure did
    not exist while the stage beside it published that figure -- which a byte
    pin makes unfixable while protecting nothing numerical. So each declared
    file is held to its own stronger statement above and every other file is
    held to its bytes.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    for relative in NUMERICAL_MODULES:
        before = blob_text(base, relative)

        after = (PROJECT / relative).read_text(encoding="utf-8")

        if after == before:
            continue

        narrowed = [
            check
            for declared, check in NARROWED_NUMERICAL_PINS
            if relative in declared
        ]

        assert len(narrowed) == 1, relative

        narrowed[0](relative, before, after)


def test_no_model_artifact_or_freeze_manifest_changed():
    """No fitted artifact and no freeze manifest moved.

    Everything under ``models/`` except the two registries that record what
    serving runs. Those two are not what serving runs: the serving-source
    contract's own purpose is to have a divergence enumerated and justified in
    it when an approved change lands, so a branch that may never touch it
    could never land one. What stays immutable there is the evidence rather
    than the record, and that is asserted directly below: every
    ``historical_reference_sha256`` is byte-identical to production and no
    locked file is dropped.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    changed = git("diff", "--name-only", base, "HEAD").splitlines()

    for relative in changed:
        if relative not in DECLARABLE_SERVING_SOURCE_REGISTRIES:
            assert not relative.startswith("models/"), relative
        assert not relative.endswith(".joblib"), relative

    for relative in DECLARABLE_SERVING_SOURCE_REGISTRIES:
        if relative not in changed:
            continue

        before = json.loads(blob_text(base, relative))
        after = json.loads((PROJECT / relative).read_text(encoding="utf-8"))

        locked_before = before.get("locked_serving_source_files") or {}
        locked_after = after.get("locked_serving_source_files") or {}

        assert set(locked_before) <= set(locked_after), relative

        for name, entry in locked_before.items():
            assert entry.get("historical_reference_sha256") == locked_after[
                name
            ].get("historical_reference_sha256"), f"{relative}:{name}"


#: The lifecycle's stages, in the order the production ref runs them. Named
#: here so the ordering claim is readable rather than implied by a digest.
EXPECTED_STEP_ORDER: tuple[str, ...] = (
    "Check out the authoritative production code",
    "Record the production checkout",
    "Set up Python",
    "Create the per-run Python environment",
    "Install the production package",
    "Verify the production interpreter is isolated",
    "Preflight",
    "Refresh the current-season rolling state",
    "Run the adaptive daily protocol",
    "Install the verified frozen runtime bundle",
    "Serve the slate with the incumbent",
    "Derive the distributions behind the served prices",
    "Publish the served slate to the WizardOfOdds feed",
    "Build and publish the WizardOfOdds player-props page",
    "Grade the incumbent's previous slate",
    "Shadow the slate beside production",
    "Monitor the accumulated live shadow evidence",
    "Assert the candidate shadow was healthy",
    "Report the run",
    "Upload run diagnostics",
)


def lines_only_appended(*, before: str, after: str) -> list[str]:
    """Why ``after`` is not ``before`` plus new lines, or an empty list.

    The additive-only rule is "every line production has is still there,
    unedited and in order", and this is the one place it is computed, so the
    two guards below and the regression tests beside them cannot read it
    differently.

    Stated as a subsequence rather than as a filtered-list equality, which is
    the same statement only when no line repeats. A YAML workflow repeats
    plenty of them -- ``run: |``, ``set -euo pipefail``, ``env:``, a bare
    comment marker -- so an appended step that used any of those would read as
    a reordering under the filtered form while having reordered nothing. The
    production lines are consumed in order and with multiplicity, which is
    what the rule actually means.
    """
    kept = before.splitlines()

    lines = after.splitlines()

    removed = [line for line in kept if line not in lines]

    if removed:
        return [f"removed or edited existing lines: {removed}"]

    position = 0

    for line in kept:
        while position < len(lines) and lines[position] != line:
            position += 1

        if position >= len(lines):
            return [
                f"reordered the lines production has; {line!r} no longer "
                "appears in its original order"
            ]

        position += 1

    return []


def test_the_additive_only_rule_still_refuses_an_edited_line():
    assert lines_only_appended(before="a\nb\n", after="a\nB\n") != []


def test_the_additive_only_rule_still_refuses_a_removed_line():
    assert lines_only_appended(before="a\nb\nc\n", after="a\nc\n") != []


def test_the_additive_only_rule_still_refuses_a_reordering():
    assert lines_only_appended(before="a\nb\nc\n", after="a\nc\nb\n") != []


def test_the_additive_only_rule_accepts_an_appended_block():
    assert lines_only_appended(before="a\nb\n", after="a\nb\nc\nd\n") == []


def test_the_additive_only_rule_accepts_an_insertion_before_the_last_step():
    """Which is where a reporting step goes: after the work, before the report."""
    assert lines_only_appended(before="a\nb\nz\n", after="a\nb\nc\nz\n") == []


def test_the_additive_only_rule_tolerates_a_repeated_appended_line():
    """The case the filtered-list form got wrong.

    ``run: |`` appears in every step that runs anything, so an appended step
    repeats lines production already has. Repeating one is not reordering it.
    """
    assert (
        lines_only_appended(
            before="step: one\nrun: |\nstep: two\nrun: |\n",
            after="step: one\nrun: |\nstep: two\nrun: |\nstep: three\nrun: |\n",
        )
        == []
    )


def test_no_existing_workflow_step_moves_and_an_added_one_is_declared():
    """G. The fix is Python state semantics; the YAML does not move.

    Stated as "every step this workflow had is still there, in order" rather
    than "the step list is exactly this tuple", because those are the same
    sentence only on a branch that adds no step. The stronger form was true of
    the branch that made this fix and is a statement about that branch rather
    than about this fix, so it would fail for the next branch to append a
    reporting step rather than for anything doing something wrong. That is the
    same correction ``test_no_workflow_production_already_has_was_changed``
    already took, for the same reason, and the mechanism is reused rather than
    reinvented: ``nba_production_lifecycle.yml`` is in
    ``ADDITIVE_ONLY_WORKFLOWS``, so adding a step is governed by the
    containment declaration and rewriting one is not possible at all.

    What this pins is still everything it pinned. No existing step may be
    renamed, removed, reordered or interleaved with a new one out of position,
    and an added step must name a declaration that says both what changes and
    what does not.
    """
    import yaml

    document = yaml.safe_load(
        LIFECYCLE_WORKFLOW.read_text(encoding="utf-8")
    )

    steps = tuple(step["name"] for step in document["jobs"]["lifecycle"]["steps"])

    missing = [name for name in EXPECTED_STEP_ORDER if name not in steps]

    assert missing == [], (
        f"these steps the lifecycle had are gone or renamed: {missing}"
    )

    assert [name for name in steps if name in set(EXPECTED_STEP_ORDER)] == list(
        EXPECTED_STEP_ORDER
    ), "the lifecycle reordered the steps it already had"

    added = [name for name in steps if name not in set(EXPECTED_STEP_ORDER)]

    if not added:
        return

    relative = LIFECYCLE_WORKFLOW.relative_to(PROJECT).as_posix()

    assert relative in ADDITIVE_ONLY_WORKFLOWS, (
        f"{relative} gained {added} and is not additive-only"
    )

    assert relative in DECLARED_INTEGRATION_PATHS, (
        f"{relative} gained {added} without a declaration"
    )

    reason = DECLARED_INTEGRATION_PATHS[relative]

    assert "What changes" in reason and "What does not change" in reason, (
        f"the declaration for {relative} does not say both what changes and "
        "what does not"
    )


def test_the_lifecycle_workflow_is_byte_identical_to_production_or_declared():
    """Byte-identical, or an additive declared change whose lines all survive.

    Narrowed for the same reason and under the same rule as the step-order
    pin above. A branch that must add a reporting step to the scheduled
    lifecycle has to be able to, and the thing worth banning is doing it
    without a reviewed statement of what changed and what did not. So the
    allowance is exactly the line-set rule: every line production has is
    still there, unedited and in order. No cron, trigger, permission,
    concurrency group, runner label, checkout ref or existing command can
    survive being changed under it.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    relative = LIFECYCLE_WORKFLOW.relative_to(PROJECT).as_posix()

    before = blob_text(base, relative)

    after = LIFECYCLE_WORKFLOW.read_text(encoding="utf-8")

    if after == before:
        return

    assert relative in ADDITIVE_ONLY_WORKFLOWS, (
        f"{relative} differs from production and is not additive-only"
    )

    assert relative in DECLARED_INTEGRATION_PATHS, (
        f"{relative} differs from production without a declaration"
    )

    assert lines_only_appended(before=before, after=after) == [], (
        f"{relative} is additive-only and "
        + "; ".join(lines_only_appended(before=before, after=after))
    )


def test_the_classifier_and_the_serving_step_are_given_the_same_slate_date():
    """The invariant serving's date cross-check deliberately does not rely on.

    The refresh step computes the slate date once, exports it, and hands the
    same value to the classifier, so in the scheduled path the status file's
    date and the served date agree by construction and
    ``readiness_for_this_slate`` can never fire. That is the point of pinning
    it here: if a workflow change breaks this, the change is visible in a
    failing test rather than silent, and serving refuses either way because it
    compares the two itself rather than trusting this.
    """
    import yaml

    document = yaml.safe_load(LIFECYCLE_WORKFLOW.read_text(encoding="utf-8"))

    steps = {
        step["name"]: step.get("run") or ""
        for step in document["jobs"]["lifecycle"]["steps"]
    }

    refresh = steps["Refresh the current-season rolling state"]

    assert 'echo "slate=$SLATE" >> "$GITHUB_ENV"' in refresh
    assert '--slate-date "$SLATE"' in refresh
    assert "classify_refresh_outcome.py" in refresh

    serve = steps["Serve the slate with the incumbent"]

    assert '--slate-date "$slate"' in serve
    assert '--refresh-status "$RUNNER_TEMP/run/refresh.json"' in serve


def test_no_workflow_production_already_has_was_changed_silently():
    """A workflow production already runs may not change without saying so.

    Stated as "every workflow the production ref has is byte-identical"
    rather than "no path under .github/workflows/ appears in the diff",
    because those are the same sentence only on a branch that adds no
    workflow. The stronger form was true of the branch that made this fix and
    is a statement about that branch rather than about this fix, so it failed
    for the next branch to add a schedule rather than for anything doing
    something wrong. Adding a workflow is governed by the containment
    declaration; silently editing one production already runs is what this
    pins.

    Byte-identical-or-declared, for the same reason: a branch that must fix
    a scheduled workflow -- the readiness refresh's catch-up job ran its probe
    on segment bounds its own resolver had refused to emit, and turned four
    successful readiness refreshes into red runs -- has to be able to, and
    the thing worth banning is doing it without a reviewed statement of what
    changed and what did not. ``.github/workflows/`` is declarable: the
    undeclarable prefixes are the model artifacts, the configs, the scripts
    and the release and review surfaces. So the ban is on the silence, and
    ``DECLARED_INTEGRATION_PATHS`` is where it is broken.

    A declared additive-only workflow is held to the line-set rule instead,
    for the same reason and not as an exception to it. GitHub does not put a
    repository variable or a secret into a step's process unless the step
    binds it, so a value that must reach a subprocess cannot be supplied
    anywhere but the workflow -- and "every line production has is still
    there, unedited and in order" is the form of this pin that says a step was
    appended and not rewritten. No cron, trigger, permission, concurrency
    group, runner label, checkout ref or existing command can survive it
    being changed.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    existing = git("ls-tree", "--name-only", f"{base}:.github/workflows").split()

    assert existing, "the production ref has no workflows, which cannot be right"

    edited: list[str] = []

    for name in existing:
        relative = f".github/workflows/{name}"

        before = blob_text(base, relative)

        after = (PROJECT / relative).read_text(encoding="utf-8")

        if after == before:
            continue

        if relative not in ADDITIVE_ONLY_WORKFLOWS:
            edited.append(name)

            continue

        lines = after.splitlines()

        removed = [line for line in before.splitlines() if line not in lines]

        assert removed == [], (
            f"{relative} is additive-only and removed or edited existing "
            f"lines: {removed}"
        )

        assert [line for line in before.splitlines() if line in lines] == [
            line for line in lines if line in set(before.splitlines())
        ], f"{relative} reordered the lines production has"

    undeclared = [
        name
        for name in edited
        if f".github/workflows/{name}" not in DECLARED_INTEGRATION_PATHS
    ]

    assert undeclared == [], (
        "these workflows production already runs were edited without a "
        f"declaration: {undeclared}"
    )

    for name in edited:
        reason = DECLARED_INTEGRATION_PATHS[f".github/workflows/{name}"]

        assert "What changes" in reason and "What does not change" in reason, (
            f"the declaration for {name} does not say both what changes and "
            "what does not"
        )
