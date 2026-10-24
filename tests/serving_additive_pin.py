"""What "no pricing logic changed" has to mean once a serving file may change.

Several suites claim that their own branch moved nothing a model computes, and
each stated it as byte-equality against the production ref over a list of
numerical modules. That is the right claim and byte-equality was the right way
to say it for as long as no approved change could reach those files.

Three of them it can. ``scripts/15_price_markets.py``,
``src/nba_prop_quant/slate.py`` and
``ops/run_incumbent_production_serving.py`` are declared additive-only in
``nba_prop_quant.research.game_latent_state.safety``, because a
serving-correctness or serving-plumbing fix has to have somewhere to go, and
the repository already carries five such changes. Byte-equality reads those as
"a price moved", which is exactly what they are not: it cannot tell a copied
column apart from an edited coefficient, and it cannot tell "opening day has a
slate" apart from an edited price.

So the claim is stated here the way the containment guard states it, and it is
strictly stronger than byte-equality over a list that had to shrink to stay
true. Every line the production ref has must still be present, so no existing
call, threshold, flag or computation can have been changed, reordered out of
existence or dropped -- and every function except the one named plumbing
surface must be syntax-tree-identical to production, so the permission cannot
widen into "this file is editable".
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

#: The serving files an approved additive change may reach, each with the only
#: functions whose bodies may differ. Mirrors
#: ``ADDITIVE_ONLY_SERVING_ENTRY_POINTS``, ``ADDITIVE_ONLY_SERVING_SOURCES``,
#: ``ADDITIVE_ONLY_SERVING_ORCHESTRATORS`` and
#: ``ADDITIVE_ONLY_CAPTURE_SELECTION_SOURCES`` and their exempt-function
#: lists; the branch-safety suite is what enforces the mapping, and this is
#: how the numerical claims read it.
#:
#: ``prospective_snapshot.py`` is the fifth, and it is the one entry where
#: byte-equality was reading a *correctness* claim as a numerical one. The
#: module computes no probability and no price; it decides which
#: point-in-time capture pricing may read, and requiring exactly one
#: scheduled record per window made a window with two records permanently
#: unpriceable. ``select_capture_bundle`` is the one function that may
#: differ, so the digest functions every capture ever written is addressed by
#: are held syntax-tree-identical here exactly as they are there.
#:
#: It is also the one entry whose exemption names a method. Selecting one
#: game's capture refuses per game, and the client assembled the whole slate
#: in one comprehension, so a refusal that belonged to one game aborted the
#: slate; ``ProspectiveSnapshotClient.__init__`` is where that is isolated.
#: The unit compared is therefore the method rather than the class -- see
#: :func:`module_definitions` -- so every other method of the client, each
#: accessor serving reads its records through among them, is pinned
#: individually instead of disappearing into one class-shaped comparison.
ADDITIVE_ONLY_SERVING_FILES: dict[str, tuple[str, ...]] = {
    "scripts/10_predict_slate.py": ("parse_args", "main"),
    "scripts/15_price_markets.py": ("parse_args", "main"),
    "src/nba_prop_quant/slate.py": ("build_upcoming_slate_features",),
    "src/nba_prop_quant/prospective_snapshot.py": (
        "select_capture_bundle",
        "ProspectiveSnapshotClient.__init__",
    ),
    "ops/run_incumbent_production_serving.py": ("slate_readiness", "run"),
}


def _blob(ref: str, relative: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{relative}"],
        cwd=str(PROJECT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _functions(source: str) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }


def module_definitions(source: str) -> dict[str, str]:
    """Every definition in a module, with class methods named one at a time.

    The pins above compare top-level functions, and a class as one node or
    not at all. Neither can say "this one method may differ": comparing the
    class whole reads a one-method change and a rewritten accessor the same
    way, and comparing only top-level functions does not read a method
    change at all. ``prospective_snapshot.py`` is the first declared file
    where the thing that has to change is a method, so the unit the pins
    compare has to become the method.

    A class contributes one entry per method, keyed ``Class.method``, plus a
    shell entry under its own name holding its bases, decorators and every
    statement in its body that is not a method -- so a class attribute, a
    changed base or a dropped ``__slots__`` is still pinned, and the only
    thing an exemption can buy is the one method it names.

    Returns dumps rather than nodes because a shell is assembled from parts
    of a node rather than being one.
    """
    found: dict[str, str] = {}

    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = ast.dump(node)

        elif isinstance(node, ast.ClassDef):
            methods = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            ]

            for method in methods:
                found[f"{node.name}.{method.name}"] = ast.dump(method)

            found[node.name] = repr(
                (
                    node.name,
                    [ast.dump(base) for base in node.bases],
                    [ast.dump(item) for item in node.keywords],
                    [ast.dump(item) for item in node.decorator_list],
                    [
                        ast.dump(item)
                        for item in node.body
                        if not isinstance(item, ast.FunctionDef)
                    ],
                )
            )

    return found


def assert_numerically_unchanged(base: str, relative: str) -> None:
    """Assert this file computes what production computes.

    Byte-equality for an ordinary numerical module. For one of the declared
    additive-only serving files, the additive-only pin instead: production's
    lines all survive and every non-plumbing function is syntax-tree-identical.
    """
    after = (PROJECT / relative).read_text(encoding="utf-8")
    before = _blob(base, relative)

    exempt = ADDITIVE_ONLY_SERVING_FILES.get(relative)

    if exempt is None:
        assert before == after, relative
        return

    removed = [
        line for line in before.splitlines() if line not in after.splitlines()
    ]

    assert removed == [], (
        f"{relative} removed or edited lines production has: {removed}"
    )

    production_functions = module_definitions(before)
    branch_functions = module_definitions(after)

    missing = sorted(set(production_functions) - set(branch_functions))

    assert missing == [], f"{relative} dropped functions: {missing}"

    for name, dumped in production_functions.items():
        if name in exempt:
            continue

        assert dumped == branch_functions[name], (
            f"{relative}:{name} is not the function production has, and only "
            f"{list(exempt)} may differ"
        )
