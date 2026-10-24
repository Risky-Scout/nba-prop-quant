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
ADDITIVE_ONLY_SERVING_FILES: dict[str, tuple[str, ...]] = {
    "scripts/10_predict_slate.py": ("parse_args", "main"),
    "scripts/15_price_markets.py": ("parse_args", "main"),
    "src/nba_prop_quant/slate.py": ("build_upcoming_slate_features",),
    "src/nba_prop_quant/prospective_snapshot.py": ("select_capture_bundle",),
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

    production_functions = _functions(before)
    branch_functions = _functions(after)

    missing = sorted(set(production_functions) - set(branch_functions))

    assert missing == [], f"{relative} dropped functions: {missing}"

    for name, node in production_functions.items():
        if name in exempt:
            continue

        assert ast.dump(node) == ast.dump(branch_functions[name]), (
            f"{relative}:{name} is not the function production has, and only "
            f"{list(exempt)} may differ"
        )
