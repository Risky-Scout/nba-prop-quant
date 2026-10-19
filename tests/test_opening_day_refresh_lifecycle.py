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

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_no_model_pricing_or_calibration_bytes_changed():
    """F. The model, its features, Gate 3, pricing and the PMFs are frozen."""
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    for relative in NUMERICAL_MODULES:
        assert blob_text(base, relative) == (
            PROJECT / relative
        ).read_text(encoding="utf-8"), relative


def test_no_model_artifact_or_freeze_manifest_changed():
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    changed = git("diff", "--name-only", base, "HEAD").splitlines()

    for relative in changed:
        assert not relative.startswith("models/"), relative
        assert not relative.endswith(".joblib"), relative


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


def test_the_lifecycle_workflow_is_byte_identical_to_production():
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    relative = LIFECYCLE_WORKFLOW.relative_to(PROJECT).as_posix()

    assert blob_text(base, relative) == LIFECYCLE_WORKFLOW.read_text(
        encoding="utf-8"
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


def test_the_fix_changed_no_workflow_at_all():
    base = production_base()

    if base is None:
        pytest.skip(f"{PRODUCTION_REF} is not available in this checkout")

    changed = git("diff", "--name-only", base, "HEAD").splitlines()

    assert [
        relative
        for relative in changed
        if relative.startswith(".github/workflows/")
    ] == []
