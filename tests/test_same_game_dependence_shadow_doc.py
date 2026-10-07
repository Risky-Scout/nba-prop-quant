"""The production-facing record must state the retired requirement as retired.

The failure mode this file exists to prevent is a later edit that rounds "we
did not meet the original 20% count-space target, and retired it because it was
incompatible with another original constraint" up to "all requirements met".
That reading is available to anyone who sees `14/14 PASS` next to a list of
original requirements and does not read further, so the document has to say the
uncomfortable part explicitly and the test has to check that it still does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

DOC_PATH = PROJECT / "docs" / "wizardofodds" / "SAME_GAME_DEPENDENCE_SHADOW_V1.md"

FINAL_SPEC_PATH = PROJECT / "research" / "final_model" / "final_model_spec.json"

FORENSIC_REPORT_PATH = (
    PROJECT / "research" / "count_space_forensic" / "count_space_forensic.json"
)

#: Recorded verbatim in the production-facing document, not paraphrased.
REQUIRED_DISPOSITIONS = (
    "COUNT_SPACE_FORENSIC_CHANGE_ADOPTED = NO",
    "COUNT_SPACE_20_PERCENT_REQUIREMENT_DISPOSITION",
    "INCOMPATIBLE_WITH_OTHER_ORIGINAL_CONSTRAINTS_UNDER_CURRENT_ARCHITECTURE",
    "MARGINAL_CONVENTION_AUDIT = NON_MATERIAL",
    "RAW_MONITOR_ONLY_NOT_USED_FOR_MODEL_SELECTION_OR_PROMOTION",
)


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_the_production_facing_document_exists(doc):
    assert doc.strip()


def test_it_states_that_the_implemented_final_gates_pass(doc):
    assert "14/14 IMPLEMENTED FINAL GATES PASS" in doc


def test_it_states_that_the_original_twenty_percent_target_was_not_achieved(doc):
    assert "NOT achieved under the full original constraint set" in doc
    assert "passer_ast_teammate_pts" in doc


def test_it_names_the_constraint_that_binds_first(doc):
    assert "teammate_reb_reb" in doc
    assert "teammate_reb_reb_no_worse" in doc
    assert "binds first" in doc


def test_it_records_every_required_disposition(doc):
    missing = [token for token in REQUIRED_DISPOSITIONS if token not in doc]
    assert missing == [], f"the production record omits: {missing}"


def test_it_never_claims_the_twenty_percent_target_passed(doc):
    """No sentence may put the retired target next to a word meaning success.

    Checked per sentence rather than over the whole document, because the
    document legitimately contains both the target and the word "pass" — in
    "14/14 ... PASS" and in the prohibition list. What must not exist is a
    sentence that attaches success to the target itself.
    """
    success = re.compile(
        r"\b(pass(es|ed)?|met|achiev(e|ed|es)|satisfi(es|ed)|clear(s|ed)?|"
        r"reach(es|ed)?)\b",
        re.IGNORECASE,
    )
    target = re.compile(r"20\s*%|20 percent|passer_ast_teammate_pts", re.IGNORECASE)
    negation = re.compile(
        r"\b(not|never|no|without|fail(s|ed)?|unmet|retired|out of reach|"
        r"may not|do not|cannot)\b",
        re.IGNORECASE,
    )
    offenders = []
    for sentence in re.split(r"(?<=[.;:])\s+|\n", doc):
        if not target.search(sentence):
            continue
        if success.search(sentence) and not negation.search(sentence):
            offenders.append(sentence.strip())
    assert offenders == [], (
        "these sentences read as the retired 20% target having succeeded: "
        f"{offenders}"
    )


def test_it_forbids_the_qualified_restatements_too(doc):
    """"Effectively met" and "met under a corrected estimator" are the risk.

    The forensic study really did find a corrected estimator that reaches 20%.
    Saying so without saying that taking it breaks `teammate_reb_reb_no_worse`
    is the most plausible way this gets misreported.
    """
    assert "effectively" in doc.lower()
    assert "under a corrected estimator" in doc.lower()


def test_it_does_not_claim_the_shadow_is_published_or_better(doc):
    assert "Shadow only" in doc
    assert "incumbent remains the served and published authority" in doc
    assert "not evidence of superiority" in doc


def test_the_numbers_it_quotes_come_from_the_forensic_record(doc):
    """The entry values in the document are the study's, not invented."""
    report = json.loads(FORENSIC_REPORT_PATH.read_text(encoding="utf-8"))
    curve = report["section_3_feasibility_envelope"]["trade_curve"]["curve"]
    entries = [row["entry"] for row in curve]

    # the entry that delivers exactly 20%, quoted as 0.0508138
    twenty = [
        row
        for row in curve
        if abs(row["focal_count_error_reduction"] - 0.20) < 5e-4
    ]
    assert twenty, "the study's curve no longer contains a 20% crossing"
    assert f"{twenty[0]['entry']:.7f}" in doc

    # the largest entry feasible under the original ("commissioned") reading
    feasible = [
        row["entry"] for row in curve if row["feasible_under"]["commissioned"]
    ]
    assert f"{max(feasible):.7f}" in doc
    assert max(feasible) < twenty[0]["entry"], (
        "the 20% crossing is no longer out of reach, so this document is stale"
    )
    assert entries == sorted(entries)


def test_the_dispositions_match_the_final_model_specification(doc):
    """The document is a copy of the spec's record, not a second opinion."""
    spec = json.loads(FINAL_SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["COUNT_SPACE_FORENSIC_CHANGE_ADOPTED"] == "NO"
    assert (
        spec["COUNT_SPACE_20_PERCENT_REQUIREMENT_DISPOSITION"]
        == "INCOMPATIBLE_WITH_OTHER_ORIGINAL_CONSTRAINTS_UNDER_CURRENT_ARCHITECTURE"
    )
    assert spec["MARGINAL_CONVENTION_AUDIT"] == "NON_MATERIAL"
    assert (
        spec["UNCERTAINTY_CALIBRATION"]
        == "RAW_MONITOR_ONLY_NOT_USED_FOR_MODEL_SELECTION_OR_PROMOTION"
    )
    for key in (
        "COUNT_SPACE_FORENSIC_CHANGE_ADOPTED",
        "COUNT_SPACE_20_PERCENT_REQUIREMENT_DISPOSITION",
        "MARGINAL_CONVENTION_AUDIT",
        "UNCERTAINTY_CALIBRATION",
    ):
        assert spec[key] in doc, f"{key}'s recorded value is not in the document"


def test_it_does_not_reinterpret_the_incumbent_claim_policy(doc):
    """The marginal model's public claim policy is referenced, not restated."""
    assert "PUBLIC_CLAIM_POLICY_V1.md" in doc
    assert "does not modify, widen or reinterpret" in doc
    policy = PROJECT / "docs" / "wizardofodds" / "PUBLIC_CLAIM_POLICY_V1.md"
    sums = (
        PROJECT / "docs" / "wizardofodds" / "PUBLIC_CLAIM_POLICY_V1_SHA256SUMS.txt"
    ).read_text(encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert digest in sums, "the incumbent claim policy no longer matches its checksum"
