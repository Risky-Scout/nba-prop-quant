"""What "no workflow changed" has to mean once a workflow may change.

Three suites claim that their branch moved nothing in the production path,
and each stated it over the workflow files: byte identity in two of them and
"every production line survives" in the third. That is the right claim, and
byte identity was the right way to say it for as long as no approved change
could reach those files.

One can. A deprecated action version and an install command cannot be
corrected by appending a step -- a job that checks out twice checks out twice,
and a second ``pip install`` leaves the first one running -- so an append-only
pin forbids the one class of edit that changes nothing about what runs while
permitting, in principle, any amount of appended behaviour. Byte identity
reads an action version bump as "the lifecycle was rewritten", which is
exactly what it is not.

So the claim is stated here once, the way the containment guard states its
own carve-outs, and it is strictly stronger than either form was. Every line
the production ref has must still be present, except the substitutions and
the one removal enumerated in the declaration module, and a substitution
counts only when its replacement is present, so none of them can be used to
delete. And nothing may be added except a comment, a declared replacement or
a declared cache input: a new command, flag, condition, ref or entry point is
not addable through this, which byte identity's successor would otherwise
have allowed.
"""

from __future__ import annotations

from nba_prop_quant.research.game_latent_state.safety import (
    DECLARED_WORKFLOW_ADDED_LINES,
    DECLARED_WORKFLOW_LINE_REMOVALS,
    DECLARED_WORKFLOW_LINE_SUBSTITUTIONS,
)


def shell_lines(text: str) -> list[str]:
    """The file's lines, indentation intact, with continuations joined.

    A ``pip install`` written over two lines with a trailing backslash is one
    command, and reading its two halves separately would compare an install
    against an argument list. Indentation is kept because in YAML it is the
    structure: a line that moved to another job or another key is a different
    line, and comparing stripped text would not notice.
    """
    joined: list[str] = []

    for line in text.splitlines():
        if joined and joined[-1].rstrip().endswith("\\"):
            joined[-1] = (
                joined[-1].rstrip().removesuffix("\\").rstrip()
                + " "
                + line.strip()
            )
        else:
            joined.append(line)

    return joined


def assert_only_the_declared_workflow_lines_moved(
    relative: str, before: str, after: str
) -> None:
    """Assert this workflow still runs what the production ref runs."""
    produced = shell_lines(after)

    retained = shell_lines(before)

    stated = {line.strip() for line in produced}

    substitutable = {
        old
        for old, replacements in DECLARED_WORKFLOW_LINE_SUBSTITUTIONS.items()
        if stated & set(replacements)
    }

    allowed_absent = substitutable | set(DECLARED_WORKFLOW_LINE_REMOVALS)

    missing = [
        line
        for line in retained
        if line not in produced and line.strip() not in allowed_absent
    ]

    assert missing == [], (
        f"{relative} no longer runs {missing}, and only the declared action "
        "versions, installs and pip self-upgrade may go"
    )

    replacements = {
        new
        for old, candidates in DECLARED_WORKFLOW_LINE_SUBSTITUTIONS.items()
        if old in substitutable
        for new in candidates
    }

    allowed_new = replacements | set(DECLARED_WORKFLOW_ADDED_LINES)

    added = [
        line
        for line in produced
        if line not in retained
        and line.strip() not in allowed_new
        and not line.strip().startswith("#")
    ]

    assert added == [], (
        f"{relative} gained {added}, which is neither a comment, a declared "
        "replacement nor a declared cache input"
    )
