"""The code SHA a publication provenances its numbers to.

A WizardOfOdds release manifest names the commit that produced the rows it
carries. The point of that field is that somebody holding the published feed
can go back to the code which priced it, so the only value that does any work
there is a commit that actually exists.

The strict end-to-end fixture used to hand serving a stand-in -- ``"e2e"``
followed by thirty-seven zeroes -- and the publisher did exactly what it is
supposed to do with a receipt's provenance: it copied it. So a real canary,
published to the real destination and read back over its real public URL,
carried a manifest naming a commit nobody can check out. Every other number
in that manifest was genuine, which is what made it worth fixing: the one
unverifiable field was the one a consumer would use to verify the rest.

These tests pin the chain in both directions. The publisher copies the
receipt's SHA and never substitutes its own, whatever that SHA is; and the
fixture that writes the receipt supplies the commit it is actually running
from, so the chain ends at something real rather than merely consistent.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]

for relative in ("ops", "src", "tests"):
    if str(PROJECT / relative) not in sys.path:
        sys.path.insert(0, str(PROJECT / relative))

import production_lifecycle_preflight as preflight  # noqa: E402
import publish_incumbent_to_wizardofodds as publisher  # noqa: E402
import wizardofodds_publisher_fixture as fixture  # noqa: E402
from test_wizardofodds_canary_end_to_end import (  # noqa: E402
    FULL_COMMIT_SHA,
    checkout_sha,
    names_a_commit,
    serve_and_provenance,
)
from test_real_slate_serving_root import installed_bundle  # noqa: E402

#: The value the end-to-end fixture used to hand serving, spelled the way the
#: fixture spelled it so that restoring it in either form fails here.
RETIRED_PLACEHOLDER = "e2e" + "0" * 37

END_TO_END_MODULE = PROJECT / "tests" / "test_wizardofodds_canary_end_to_end.py"

PUBLISHER_MODULE = PROJECT / "ops" / "publish_incumbent_to_wizardofodds.py"


def publish_with_receipt_sha(root: Path, sha: str) -> dict:
    """One canary over a fixture whose receipt names ``sha`` and nothing else.

    ``receipt_changes`` edits the provenance after the priced parquet's digest
    was taken, so the code SHA is the only field that differs from a receipt
    which would otherwise publish.
    """
    built = fixture.build(root, receipt_changes={"production_code_sha": sha})

    assert built["receipt"]["production_code_sha"] == sha

    receipt = publisher.publish(
        mode=publisher.MODE_CANARY,
        slate_date=built["slate_date"],
        data_root=built["data_root"],
        project_root=PROJECT,
        destination_root=built["destination_root"],
    )

    manifest = json.loads(
        (
            built["destination_root"] / receipt["release"]["manifest_relative"]
        ).read_text(encoding="utf-8")
    )

    return {"built": built, "manifest": manifest, "receipt": receipt}


# ======================================================================
# 1. the placeholder is gone, and cannot come back unnoticed
# ======================================================================


def test_the_end_to_end_fixture_no_longer_names_the_placeholder():
    """The retired value appears nowhere in the fixture that used to emit it.

    Checked against the source rather than against a run, because the
    placeholder was assembled from two literals and a run only shows what one
    particular path produced.
    """
    source = END_TO_END_MODULE.read_text(encoding="utf-8")

    assert RETIRED_PLACEHOLDER not in source
    assert '"e2e" + "0" * 37' not in source
    assert "e2e0000000" not in source


def test_the_fixture_takes_its_code_sha_from_its_caller():
    """``build_receipt`` is handed a name, never a literal.

    The structural form of the defect: a constant in that argument position is
    a SHA decided by the fixture, which is the one thing it is not entitled to
    decide. Asserted over the syntax tree so that any spelling of a constant
    -- one literal, two concatenated, an f-string -- fails the same way.
    """
    tree = ast.parse(END_TO_END_MODULE.read_text(encoding="utf-8"))

    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "serve_and_provenance"
    )

    assert "production_sha" in {
        argument.arg for argument in helper.args.kwonlyargs
    }, "the caller has to supply the SHA for the caller to be the one deciding it"

    passed = [
        keyword.value
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_receipt"
        for keyword in node.keywords
        if keyword.arg == "production_sha"
    ]

    assert len(passed) == 1, "exactly one receipt is built, with one code SHA"

    assert isinstance(passed[0], ast.Name), (
        "the code SHA reaching build_receipt is a constant, so the fixture is "
        "deciding it rather than provenancing the run it is part of"
    )


# ======================================================================
# 2. the fixture's receipt names the commit it is running from
# ======================================================================


def test_the_checkout_sha_helper_resolves_this_commit():
    """The helper's answer is a commit, and it is this checkout's commit."""
    sha = checkout_sha()

    assert FULL_COMMIT_SHA.fullmatch(sha)
    assert sha == preflight.git("rev-parse", "HEAD")
    assert preflight.git("cat-file", "-t", sha) == "commit", (
        "the SHA the fixture provenances with does not name an object in this "
        "repository"
    )
    assert sha == preflight.git("rev-parse", "HEAD^{commit}")


def test_a_code_sha_that_will_not_resolve_is_refused(monkeypatch):
    """Refusing beats defaulting.

    If HEAD cannot be resolved the honest outcome is a failure that says so.
    Returning something shaped like a SHA is how the retired placeholder
    survived long enough to be published.
    """
    monkeypatch.setattr(preflight, "git", lambda *_: "")

    with pytest.raises(AssertionError, match="does not resolve to a commit"):
        checkout_sha()


def test_the_retired_placeholder_is_shaped_exactly_like_a_commit():
    """Why pattern matching could never have caught this.

    Forty characters, every one of them hex. Any check that asks what a code
    SHA looks like accepts it, which is why the fixture has to ask whether the
    repository holds it instead.
    """
    assert FULL_COMMIT_SHA.fullmatch(RETIRED_PLACEHOLDER)
    assert not names_a_commit(RETIRED_PLACEHOLDER)
    assert names_a_commit(checkout_sha())


@pytest.mark.parametrize(
    "sha", ["", "e2e" + "0" * 37, "abc123", "0" * 40, "0" * 41, "Z" * 40]
)
def test_the_fixture_refuses_a_code_sha_that_is_not_a_commit(sha: str):
    """Including the retired placeholder, by the value it actually had.

    ``serve_and_provenance`` serves a real slate through the frozen bundle, so
    this refusal is asserted before any of that happens -- which is also why
    it needs no bundle to run.
    """
    with pytest.raises(
        AssertionError, match="is not a commit this repository holds"
    ):
        serve_and_provenance(Path("/unused"), Path("/unused"), production_sha=sha)


def test_the_served_fixture_receipt_carries_the_actual_checkout_sha(tmp_path: Path):
    """The whole fixture, through the frozen bundle, down to the receipt field.

    The end-to-end file asserts this as part of a larger publication. Asserted
    here on its own as well, because it is the defect's own property and it
    should fail by itself when it breaks.
    """
    bundle = installed_bundle()

    if bundle is None:
        pytest.skip(
            "no installed frozen runtime bundle; set "
            "NBA_PROP_TEST_FROZEN_BUNDLE_ROOT or NBA_PROP_WORK_DIR to a root "
            "ops/install_frozen_model_artifacts.py has installed into"
        )

    provenanced = serve_and_provenance(
        bundle, tmp_path / "run", production_sha=checkout_sha()
    )

    recorded = provenanced["receipt"]["production_code_sha"]

    assert recorded == checkout_sha()
    assert FULL_COMMIT_SHA.fullmatch(recorded)
    assert recorded != RETIRED_PLACEHOLDER


# ======================================================================
# 3 and 4. the publisher copies that SHA, and only that SHA
# ======================================================================


def test_the_manifest_carries_the_receipts_code_sha_exactly(tmp_path: Path):
    """Receipt in, manifest out, byte for byte, with nothing else moved.

    Also the authority: a publication that got the SHA right while publishing
    something other than the incumbent would not be the fix.
    """
    sha = checkout_sha()

    published = publish_with_receipt_sha(tmp_path, sha)

    provenance = published["manifest"]["provenance"]

    assert provenance["production_code_sha"] == sha
    assert (
        provenance["production_code_sha"]
        == published["built"]["receipt"]["production_code_sha"]
    )
    assert published["receipt"]["provenance"]["production_code_sha"] == sha

    assert published["manifest"]["model_authority"] == publisher.MODEL_AUTHORITY
    assert published["manifest"]["candidate_authority"] == publisher.CANDIDATE_AUTHORITY
    # The state the switch records, not a claim about this run, which is a
    # canary.
    assert published["receipt"]["live_publishing"] == publisher.STATE_ENABLED
    assert published["receipt"]["mode"] == publisher.MODE_CANARY


def test_a_different_valid_source_sha_propagates_unchanged(tmp_path: Path):
    """Two real commits, two publications, each manifest naming its own.

    The property is propagation rather than correctness-of-one-value: the
    publisher has no opinion about which commit served a slate, so a second
    receipt naming a different real commit has to come out the other side as
    that commit and not as the first one, nor as whatever the publisher is
    running from.
    """
    first = checkout_sha()

    second = preflight.git("rev-parse", "HEAD~1")

    if not FULL_COMMIT_SHA.fullmatch(second):
        pytest.skip("this checkout has no parent commit to use as a second SHA")

    assert second != first

    for name, sha in (("first", first), ("second", second)):
        published = publish_with_receipt_sha(tmp_path / name, sha)

        assert published["manifest"]["provenance"]["production_code_sha"] == sha

    # And the publisher's own HEAD is not what either of them carries by
    # accident: the second release names a commit the publisher is not on.
    assert second != preflight.git("rev-parse", "HEAD")


def test_the_publisher_never_asks_git_anything():
    """No publisher-HEAD substitution, as a property of the module.

    The publisher cannot substitute a SHA it has no way to obtain. It shells
    out only to move bytes to a remote destination, so the absence of git here
    is what makes "the receipt's SHA, unchanged" the only reachable behaviour.
    """
    source = PUBLISHER_MODULE.read_text(encoding="utf-8")

    assert not re.search(r"rev-parse|\bgit\b", source), (
        "the publisher reaches for git, so it could name a commit other than "
        "the one that served the rows"
    )

    assert "production_code_sha" in publisher.REQUIRED_PROVENANCE_FIELDS, (
        "a receipt with no code SHA has to be refused rather than filled in"
    )


def test_a_receipt_with_no_code_sha_refuses_rather_than_inventing_one(tmp_path: Path):
    """The alternative to copying is refusing; it is never synthesising.

    This is the behaviour that makes the one-line fix sufficient. If the
    publisher filled the field in when it was absent, correcting the fixture
    would have hidden the defect rather than removed it.
    """
    built = fixture.build(
        tmp_path, receipt_changes={"production_code_sha": None}
    )

    with pytest.raises(publisher.PublishingRefused, match="production_code_sha"):
        publisher.publish(
            mode=publisher.MODE_CANARY,
            slate_date=built["slate_date"],
            data_root=built["data_root"],
            project_root=PROJECT,
            destination_root=built["destination_root"],
        )
