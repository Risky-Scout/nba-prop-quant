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
    ADDITIVE_ONLY_PAGE_CONSISTENCY_CHECKS,
    ADDITIVE_ONLY_PUBLICATION_TRANSPORTS,
    DECLARABLE_SERVING_SOURCE_REGISTRIES,
    DECLARED_PAGE_CONSISTENCY_NAMES,
    DECLARED_PUBLICATION_TRANSPORT_NAMES,
    PAGE_CONSISTENCY_CLASS,
    PAGE_CONSISTENCY_EXEMPT_FUNCTIONS,
    PAGE_CONSISTENCY_EXEMPT_METHODS,
    PUBLICATION_TRANSPORT_CLASS,
    PUBLICATION_TRANSPORT_EXEMPT_METHODS,
)

PROJECT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT / "tests"))

from workflow_additive_pin import (  # noqa: E402
    assert_only_the_declared_workflow_lines_moved,
)

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


#: Each declared file on the numerical list, and the stronger statement it is
#: held to instead of its bytes. A module reaches this mapping by having a
#: defect a byte pin makes unfixable while protecting nothing numerical, and
#: it stays on the numerical list: the pin is restated, not lifted.
NARROWED_NUMERICAL_PINS: tuple[tuple[tuple[str, ...], object], ...] = (
    (ADDITIVE_ONLY_PUBLICATION_TRANSPORTS, assert_only_the_publication_transport_moved),
    (ADDITIVE_ONLY_PAGE_CONSISTENCY_CHECKS, assert_only_the_page_consistency_check_moved),
)




def test_no_model_pricing_or_calibration_bytes_changed():
    """F. The model, its features, Gate 3, pricing and the PMFs are frozen.

    Byte-identical for every module here except the two declared ones, where
    the freeze is restated rather than relaxed. Both are on this list because
    they decide what is publishable and what a published record says, and both
    also happen to contain a defect in machinery that states no number -- the
    rsync the runner refuses, and a string comparison of two spellings of one
    double -- which a byte pin makes unfixable while protecting nothing
    numerical. So each declared file is held to its own stronger statement
    above and every other file is held to its bytes.
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


def test_workflow_step_order_is_unchanged():
    """G. The fix is Python state semantics; the YAML does not move."""
    import yaml

    document = yaml.safe_load(
        LIFECYCLE_WORKFLOW.read_text(encoding="utf-8")
    )

    steps = document["jobs"]["lifecycle"]["steps"]

    assert tuple(step["name"] for step in steps) == EXPECTED_STEP_ORDER


def test_the_lifecycle_workflow_still_runs_what_production_runs():
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    relative = LIFECYCLE_WORKFLOW.relative_to(PROJECT).as_posix()

    assert_only_the_declared_workflow_lines_moved(
        relative,
        blob_text(base, relative),
        LIFECYCLE_WORKFLOW.read_text(encoding="utf-8"),
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


def test_no_workflow_production_already_has_lost_a_step():
    """Every job production runs still runs the same thing.

    Stated over every workflow the production ref has rather than over the
    paths in the diff, because those are the same sentence only on a branch
    that adds no workflow. The stronger form, byte identity, was a statement
    about the branch that wrote it -- that fix needed no workflow change --
    rather than about what the pin is for, which is that nothing silently
    rewrites a job production already runs. Adding a workflow is governed by
    the containment declaration; what this pins is that an edited one keeps
    every command, condition, ref, label and entry point it had, with only
    the declared action versions, installs and cache inputs moving.
    """
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    existing = git("ls-tree", "--name-only", f"{base}:.github/workflows").split()

    assert existing, "the production ref has no workflows, which cannot be right"

    for name in existing:
        relative = f".github/workflows/{name}"

        before = blob_text(base, relative)

        after = (PROJECT / relative).read_text(encoding="utf-8")

        if after == before:
            continue

        assert_only_the_declared_workflow_lines_moved(relative, before, after)
