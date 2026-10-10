"""Containment contract for the shadow latent-state branch.

SHADOW / RESEARCH ONLY.

This module is the single declaration of which repository paths decide
production behaviour. It only ever *reads* git: nothing here writes, and the
path lists exist so that both the validation run and the branch-safety tests
check containment against the same definition rather than two drifting copies.

Gate H is this file plus :func:`modified_production_paths` returning an empty
list.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

PRODUCTION_REF = "production/wizardofodds-integration"

#: Directory prefixes whose bytes decide production behaviour: the scheduled
#: automation, the frozen configs and model artifacts, the operational
#: runbooks, and the production scripts the pipeline executes.
PROTECTED_PRODUCTION_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "configs/",
    "models/",
    "ops/",
    "scripts/",
    "docs/",
    "release/",
    "review/",
)

#: The scripts that serve a slate. What they can reach is the served model.
SERVING_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/10_predict_slate.py",
    "scripts/15_price_markets.py",
)

#: Serving entry points a declaration may name, and only additively.
#:
#: ``scripts/`` was undeclarable as a flat prefix, which is the same mistake
#: the source partition above already corrected once: it read the same for two
#: different kinds of change, the numbers these scripts compute and the paths
#: they resolve. Only the first is what "may not edit the served model" is
#: about, and conflating them made the serving deployment plumbing permanently
#: unfixable. That was not hypothetical. Both scripts verified the frozen
#: manifest against the working directory, so the frozen runtime bundle the
#: lifecycle installs got them past nothing and the first non-empty slate
#: would have failed inside ``10_predict_slate.py`` -- and the only reviewed
#: way in was a declaration these paths could not receive.
#:
#: So they are declarable, under the rule the lifecycle workflow is already
#: held to: every line the production ref has must still be there, so no
#: existing computation, threshold, flag or call can be changed, reordered out
#: of existence or dropped. :data:`NUMERICAL_SURFACE_EXEMPT_FUNCTIONS` names
#: the only functions whose bodies may differ at all, and the branch-safety
#: tests compare every other function's syntax tree against production. The
#: rest of ``scripts/``, every model artifact, every frozen config, the
#: release surface and every production source module the serving path imports
#: all remain undeclarable.
ADDITIVE_ONLY_SERVING_ENTRY_POINTS: tuple[str, ...] = SERVING_ENTRY_POINTS

#: The only functions in a declared serving entry point whose bodies may
#: differ from production. ``parse_args`` is where an argument is declared and
#: ``main`` is where it is used; everything that computes a projection or a
#: price is outside both and is pinned syntax-tree-identical.
NUMERICAL_SURFACE_EXEMPT_FUNCTIONS: tuple[str, ...] = (
    "parse_args",
    "main",
)

#: Protected sources no declaration may ever name, because the live pricing
#: path reads them.
#:
#: Undeclarability used to be a property of being protected at all, which read
#: the same for two different kinds of module: the ones that decide what a
#: price is, and the ones that orchestrate a fit. Only the first kind is what
#: "may not edit the served model" is about, and conflating them made the fit
#: orchestration permanently unfixable -- a correctness fix to the daily fit's
#: own validation had nowhere to go, since a protected source could not be
#: declared and the declaration mechanism is the only reviewed way in.
#:
#: So the line is drawn where the repository already draws it: a protected
#: source is undeclarable exactly when a serving entry point can reach it by
#: import. That is computed by :func:`serving_reachable_sources` and the
#: branch-safety tests pin this set against it, so the partition cannot drift
#: and an import added to a serving script moves a module into this set rather
#: than leaving it declarable. ``copula.py`` is here because ``model.py``
#: imports it, not because somebody listed it.
UNDECLARABLE_PRODUCTION_SOURCES: frozenset[str] = frozenset(
    {
        "src/nba_prop_quant/api.py",
        "src/nba_prop_quant/availability.py",
        "src/nba_prop_quant/copula.py",
        "src/nba_prop_quant/decay.py",
        "src/nba_prop_quant/distributions.py",
        "src/nba_prop_quant/experience.py",
        "src/nba_prop_quant/features.py",
        "src/nba_prop_quant/game_context.py",
        "src/nba_prop_quant/gate3_v2.py",
        "src/nba_prop_quant/kalman.py",
        "src/nba_prop_quant/model.py",
        "src/nba_prop_quant/normalize.py",
        "src/nba_prop_quant/pipeline.py",
        "src/nba_prop_quant/pricing.py",
        "src/nba_prop_quant/production.py",
        "src/nba_prop_quant/prospective_snapshot.py",
        "src/nba_prop_quant/settings.py",
        "src/nba_prop_quant/slate.py",
        "src/nba_prop_quant/storage.py",
    }
)

#: Production modules that must not change without a declaration. The shadow
#: layer imports and reads them; the whole no-double-count argument rests on
#: ``copula.py`` being untouched.
#:
#: Everything the serving path can reach, plus the two fit-orchestration
#: modules. This used to be eleven hand-written names, which left ten modules
#: the serving scripts import -- ``api.py``, ``normalize.py``, ``settings.py``
#: and ``storage.py`` among them -- outside it. They were covered only
#: incidentally, by a containment check that also refused every unrelated
#: branch, so scoping that check to the shadow lineage had to come with
#: closing this gap properly. A branch-safety test pins that the serving
#: closure stays inside this set, so a new serving import cannot reopen it.
PROTECTED_PRODUCTION_SOURCES: frozenset[str] = UNDECLARABLE_PRODUCTION_SOURCES | {
    "src/nba_prop_quant/adaptive_fit_registry.py",
    "src/nba_prop_quant/adaptive_training.py",
}

#: Namespaces the shadow lineage's own work lives in.
SHADOW_NAMESPACES: tuple[str, ...] = (
    "research/",
    "src/nba_prop_quant/research/",
    "tests/test_game_latent_state_shadow",
)

#: Where tests live. Production cannot reach a test, and ``tests/`` is not a
#: protected prefix, so a test is not part of the surface containment is about.
#: Named here because :func:`shadow_lineage_offenders` reads it.
TEST_SURFACE_PREFIX = "tests/"

#: The containment guard's own source. It declares and checks containment
#: rather than being contained by it, so changing it is not shadow work --
#: every branch that touches a protected path has to edit the declaration, and
#: counting that edit would make the act of declaring turn an unrelated branch
#: into a shadow branch. The literal scan in the branch-safety tests already
#: skips the declaration module for the same reason.
CONTAINMENT_GUARD_PATHS: tuple[str, ...] = (
    "src/nba_prop_quant/research/game_latent_state/safety.py",
    "tests/test_game_latent_state_shadow_safety.py",
    "tests/test_game_latent_state_shadow_bucket_repair.py",
)

#: Kept as a separate name because the literal scan asks a narrower question:
#: which file is allowed to contain protected-path literals.
PATH_DECLARATION_MODULE = "safety.py"


#: Protected paths this integration is declared to change, each with its
#: reason. Gate H's absolute form — no protected path changes at all — held for
#: as long as the dependence work was pure research, and it is the right shape
#: for a research branch. Integrating the work into production is a production
#: change by construction, so the enforced invariant becomes "nothing
#: *undeclared* changes" and the declaration lives here, next to the path lists
#: it qualifies, rather than being implicit in whatever a diff happens to show.
#:
#: Nothing here may be a model artifact, a pricing path or a publishing path.
#: The branch-safety tests pin that, so widening this mapping cannot quietly
#: become permission to edit the served model.
#:
#: A declaration is permission for a *pending* change, so it is removed once
#: that change has landed in production; otherwise it sits here as standing
#: permission to edit a protected path unreviewed. What the lineage owns in
#: production once a change has landed is recorded in
#: :data:`SHADOW_OWNED_PRODUCTION_PATHS`, which grants no permission at all.
#: Empty is the resting state, and the only correct non-empty content is the
#: pending change on the branch being read. A stale declaration is exactly
#: what :func:`stale_integration_declarations` reports, and the branch that
#: leaves one behind is always already merged by the time anything notices, so
#: it breaks the rule for every later branch rather than for the one branch
#: doing something wrong. That happened once with
#: ``ops/evidence/production_shadow_closure.json``, and the
#: pre-opening-day remediation's eleven entries were retired the same way:
#: the eight ops, docs and workflow files it introduced are recorded as owned
#: below, and ``src/nba_prop_quant/adaptive_training.py`` is recorded nowhere,
#: because the remediation changed one block of a production module it does
#: not own and must declare again to touch again. The interpreter guard's
#: spelling fix was retired in turn when it landed, and so were the pre-live
#: closure's five entries: the two graders it introduced are recorded as owned
#: below, and the three files it extended were already owned. The frozen
#: runtime bundle install's four entries were retired the same way when they
#: landed; all four are recorded as owned below, and the one of them this
#: branch changes again is declared again here.
DECLARED_INTEGRATION_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "ops/install_frozen_model_artifacts.py": (
            "installs the whole frozen project tree rather than only its "
            "model directory, so the bundle root is a tree the frozen "
            "manifest's 62 records all resolve against, and verifies them "
            "with the repository's own whole-manifest verifier rather than a "
            "second implementation of it. Publishes that root beside the "
            "model directory. Reads only the published package and the "
            "repository's frozen manifest, writes only into the durable "
            "production work root, and cannot fit, refit, recalibrate, "
            "predict, price, promote or publish"
        ),
        "ops/run_incumbent_production_serving.py": (
            "requires the verified frozen bundle root as well as the verified "
            "model directory, verifies the whole frozen manifest against that "
            "root before invoking anything, refuses when the two name "
            "different freezes, and passes the root to both serving scripts "
            "explicitly. No prediction, pricing, threshold or authority "
            "change: the incumbent is still resolved from the promotion state "
            "alone and still refuses rather than substituting"
        ),
        "scripts/10_predict_slate.py": (
            "one added argument, --frozen-bundle-root, and three added lines "
            "that resolve the frozen manifest against it when it is supplied. "
            "Declared under the additive-only rule: every line production has "
            "is still present, and every function except parse_args and main "
            "is syntax-tree-identical to production, so no projection, "
            "quantile, experience curve or feature computation changes. "
            "Without it the frozen manifest is resolved against the working "
            "directory, which cannot hold the ignored model binaries or the "
            "2025 audit outputs, and the first non-empty slate fails here"
        ),
        "scripts/15_price_markets.py": (
            "one added argument, --frozen-bundle-root, and three added lines "
            "that resolve the frozen manifest against it when it is supplied. "
            "Declared under the additive-only rule on the same terms as the "
            "prediction script: no pricing, probability, hold, edge, combo or "
            "seed computation changes, and the syntax-tree comparison against "
            "production pins that rather than asserting it"
        ),
        ".github/workflows/nba_production_lifecycle.yml": (
            "one added line, which hands the serving step the verified frozen "
            "bundle root the install step published. Every pre-existing line "
            "is unchanged"
        ),
    }
)

#: Protected paths this lineage introduced into production and now owns.
#:
#: This is a registry, not permission. It exists because two different
#: questions were previously answered by one mapping: "may this branch change
#: this protected path" and "is this protected path part of the shadow's
#: surface". Those coincided only on the branch that first deployed the
#: shadow. Once that branch merged, every entry became permanently stale under
#: the first question while remaining permanently true under the second, which
#: made the production branch impossible to extend without either leaving loose
#: permission behind or dropping the surface from the record.
#:
#: Separating them keeps both guarantees and is strictly tighter than the
#: single mapping was: changing anything here still requires a fresh entry in
#: :data:`DECLARED_INTEGRATION_PATHS` and the review that comes with it, so
#: being owned buys nothing. The same restrictions apply — no model artifact,
#: no config, no production script, no release or review path, no protected
#: production source module — and the branch-safety tests pin that for this
#: mapping exactly as they do for the declaration.
SHADOW_OWNED_PRODUCTION_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "ops/run_production_shadow.py": (
            "the shadow's production entry point. Reads the marginals and the "
            "copula the live pricing path reads, records what the candidate "
            "would have said, and publishes nothing. It is additive: no "
            "existing production script calls it and it writes no production "
            "state"
        ),
        "ops/grade_incumbent_production_slate.py": (
            "grades the slate the incumbent actually served, against settled "
            "box scores. It is the second arm of the frozen policy's central "
            "comparison, which had only the candidate's side before. Reads "
            "priced markets, the serving receipt and the box-score tree; "
            "writes grade rows and its own receipt into a tree no serving "
            "script names. Cannot predict, price, promote or publish"
        ),
        "ops/monitor_live_shadow_evidence.py": (
            "accumulates the shadow's daily evidence durably and reads the "
            "frozen policy over the running total, which nothing could do "
            "while that evidence lived only in the run's temporary directory. "
            "States no threshold of its own: every number comes out of "
            "live_shadow_promotion_policy.json. Writes only into a monitoring "
            "tree no serving or pricing script names, and cannot predict, "
            "price, promote or publish"
        ),
        "ops/install_frozen_model_artifacts.py": (
            "installs the frozen model binaries the incumbent serves from. "
            "They are distributed as a release asset and ignored in Git, so "
            "no checkout has ever held them and the serving step's own "
            "frozen-bundle verification had nothing satisfiable to check. "
            "Reads the published immutable package and the frozen manifest; "
            "writes only into the durable production work root. Cannot fit, "
            "refit, recalibrate, predict, price, promote or publish"
        ),
        ".github/workflows/nba_production_lifecycle.yml": (
            "eight added steps, every pre-existing step unchanged. One invokes "
            "the shadow entry point after the daily fit and cannot fail the "
            "lifecycle. Two give the run its own Python environment under "
            "RUNNER_TEMP and refuse to continue outside it. One serves the "
            "slate with the incumbent. One reads the status the shadow "
            "persisted and may fail the job, because the shadow step is "
            "non-blocking by design and GitHub went green whether the shadow "
            "worked or not. One installs the verified frozen runtime bundle "
            "the incumbent serves from"
        ),
        ".github/workflows/default_branch_guard.yml": (
            "the default branch's only reported check. GitHub schedules "
            "production from main, and main carried no workflow triggered by "
            "a push to main or a pull request into main, so a change to the "
            "scheduler copy landed unchecked. Runs on GitHub-hosted runners, "
            "reads branch contents and asks for contents:read"
        ),
        "ops/validate_main_scheduler_role.py": (
            "the eleven checks that guard runs: that the lifecycle copy "
            "parses, passes the production branch's own static rules, is "
            "byte-identical to the production copy, and has not lost a step, "
            "a gate, the non-blocking shadow boundary or its production "
            "checkout ref. Reads two workflow copies and writes only its own "
            "report"
        ),
        "ops/run_incumbent_production_serving.py": (
            "runs the two serving scripts for the incumbent resolved from the "
            "registry's promotion state, and refuses rather than substituting "
            "when that authority cannot be established. Reads the promotion "
            "state, the frozen bundle and the refresh status; writes the two "
            "serving artifacts and its own receipt, and cannot publish or "
            "promote"
        ),
        "ops/evaluate_shadow_health.py": (
            "judges the persisted shadow status against the frozen live "
            "policy's own operational gates and may fail the job. Reads "
            "shadow.json and the policy; writes only its own report, states "
            "no threshold of its own, and cannot publish or promote"
        ),
        "ops/verify_production_interpreter.py": (
            "refuses the production run when the interpreter resolves outside "
            "the per-run virtual environment. Reads the running interpreter "
            "and the frozen scikit-learn pin from the preflight; writes only "
            "its own receipt and alters no dependency contract"
        ),
        "ops/benchmark_production_fit.py": (
            "wraps the existing --benchmark-only fit path and fingerprints "
            "the serving tree, the registry and the three frozen identifiers "
            "around it, failing the benchmark on any difference. Passes no "
            "registry root to the fit, so there is nothing to register to; "
            "writes only its own receipt"
        ),
        "ops/audit_runner_reliability.py": (
            "decides whether the self-hosted runner meets the "
            "seven-consecutive-clean-runs criterion from GitHub's own API "
            "answer. Reads a collected JSON window and writes only its own "
            "report; it cannot administer or contact anything"
        ),
        "ops/evidence/runner_reliability_audit.json": (
            "the measured seven-run window behind the runner reliability "
            "classification: the waits, the two lost-communication runs and "
            "the 5h47m02s outlier. Evidence only; nothing reads it and it "
            "decides no production behaviour"
        ),
        "docs/wizardofodds/PRE_OPENING_DAY_OPERATOR_ACTIONS.md": (
            "the exact runner, branch-protection and benchmark actions the "
            "three blockers outside this repository require, with the "
            "measured evidence for each and the mechanical test that decides "
            "closure. Documentation only"
        ),
        "ops/evidence/production_shadow_closure.json": (
            "the production-ops closure receipt: the reconciled test counts, "
            "the production SHA both hashes were verified at, and why one "
            "commit reported four different pass/skip splits. Evidence only: "
            "nothing reads it and it decides no production behaviour. Recorded "
            "here rather than declared, because it has already landed"
        ),
    }
)

#: Workflows whose pre-existing steps must survive a change intact. Owning or
#: declaring a workflow buys the right to *add* a step, not to rewrite the
#: lifecycle: the modelling, refresh and registration steps are the production
#: path, and the branch-safety tests check them line by line against the
#: production ref rather than trusting either mapping.
ADDITIVE_ONLY_WORKFLOWS: tuple[str, ...] = (
    ".github/workflows/nba_production_lifecycle.yml",
)


def _git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def production_merge_base(project_root: Path) -> str | None:
    """The commit this branch diverged from production at, or ``None``."""
    for ref in (f"origin/{PRODUCTION_REF}", PRODUCTION_REF):
        try:
            return _git(project_root, "merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue
    return None


def changed_paths(project_root: Path) -> list[str]:
    base = production_merge_base(project_root)
    if base is None:
        return []
    return [
        line
        for line in _git(project_root, "diff", "--name-only", base, "HEAD").splitlines()
        if line
    ]


def modified_production_paths(project_root: Path) -> list[str]:
    """Protected paths this branch has modified, declared or not.

    Reported as-is. Gate H reads this, so a branch that changes a protected
    path is still visibly a branch that changed production, even when the
    change is a declared one.
    """
    return sorted(
        path
        for path in changed_paths(project_root)
        if path.startswith(PROTECTED_PRODUCTION_PREFIXES)
        or path in PROTECTED_PRODUCTION_SOURCES
    )


def undeclared_production_paths(project_root: Path) -> list[str]:
    """Protected paths this branch changed without declaring them.

    This is the invariant the branch-safety tests enforce. Empty means every
    production path this branch touches is named in
    :data:`DECLARED_INTEGRATION_PATHS` with a reason.
    """
    return sorted(
        set(modified_production_paths(project_root)) - set(DECLARED_INTEGRATION_PATHS)
    )


_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+nba_prop_quant\.([A-Za-z_][\w.]*)\s+import", re.M),
    re.compile(r"^\s*from\s+\.([A-Za-z_]\w*)\s+import", re.M),
    re.compile(r"^\s*import\s+nba_prop_quant\.([A-Za-z_]\w*)", re.M),
)


def _module_imports(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group(1).split(".")[0]
        for pattern in _IMPORT_PATTERNS
        for match in pattern.finditer(text)
    }


def serving_reachable_sources(project_root: Path) -> frozenset[str]:
    """Package modules a serving entry point can reach by import.

    Walks the import graph from :data:`SERVING_ENTRY_POINTS` through the
    package. This is what makes :data:`UNDECLARABLE_PRODUCTION_SOURCES` a
    measurement rather than a list somebody has to remember to update: if a
    serving script gains an import of a fit-orchestration module, that module
    becomes serving code and stops being declarable, and the branch-safety
    test that compares the two fails until the declaration is withdrawn.

    Deliberately *not* filtered through :data:`PROTECTED_PRODUCTION_SOURCES`.
    Filtering it there would make the comparison circular -- the protected set
    is derived from this one -- and a module the serving path newly imports
    would stay invisible precisely because it was not protected yet.
    """
    package = Path(project_root) / "src" / "nba_prop_quant"

    pending: list[str] = []

    for relative in SERVING_ENTRY_POINTS:
        entry = Path(project_root) / relative

        if entry.is_file():
            pending.extend(_module_imports(entry))

    seen: set[str] = set()

    while pending:
        name = pending.pop()

        if name in seen:
            continue

        seen.add(name)

        module = package / f"{name}.py"

        if module.is_file():
            pending.extend(_module_imports(module))

    return frozenset(
        f"src/nba_prop_quant/{name}.py"
        for name in seen
        if (package / f"{name}.py").is_file()
    )


def shadow_lineage_offenders(project_root: Path) -> list[str]:
    """Paths a shadow-lineage branch changes outside its own namespaces.

    Containment is a statement about the shadow lineage's work, so it is
    checked against branches that do shadow work. A branch that changes
    nothing under :data:`SHADOW_NAMESPACES` is not a shadow branch, and
    reporting its ordinary modules as "unexpected paths on the shadow branch"
    said nothing true about the shadow while making every unrelated production
    branch unable to add so much as a test.

    What it would otherwise have added is already covered: every path that
    decides production behaviour is a protected path, and
    :func:`undeclared_production_paths` holds every branch to it regardless of
    lineage.

    :data:`CONTAINMENT_GUARD_PATHS` does not count towards doing shadow work:
    it is the guard, not the surface the guard covers.

    Neither does :data:`TEST_SURFACE_PREFIX` count towards being an offender.
    The exemption for ``tests/test_game_latent_state_shadow`` was always about
    tests rather than about that prefix: it is simply how the lineage's own
    tests happen to be named. A branch that extends the shadow and also adds
    the test for a production entry point it wires the shadow into had the
    choice of leaving that entry point untested or declaring its test file as
    though a test file decided production behaviour, and neither of those is
    the containment this guard exists to enforce. No test is reachable from
    production: ``tests/`` is not a protected prefix, so nothing here relaxes
    what :func:`undeclared_production_paths` requires. What remains reported
    is the part that was never redundant -- source outside the lineage's
    namespaces.
    """
    changed = changed_paths(project_root)

    doing_shadow_work = [
        path
        for path in changed
        if path.startswith(SHADOW_NAMESPACES)
        and path not in CONTAINMENT_GUARD_PATHS
    ]

    if not doing_shadow_work:
        return []

    return sorted(
        path
        for path in changed
        if not path.startswith(SHADOW_NAMESPACES)
        and not path.startswith(TEST_SURFACE_PREFIX)
        and path not in DECLARED_INTEGRATION_PATHS
    )


def stale_integration_declarations(project_root: Path) -> list[str]:
    """Declared paths this branch does not actually change.

    A declaration that has outlived its change is permission left lying
    around, so it is reported rather than ignored.
    """
    touched = set(modified_production_paths(project_root))
    return sorted(set(DECLARED_INTEGRATION_PATHS) - touched)


# ---------------------------------------------------------------------
# What a merge would add to production history, as opposed to its tree
# ---------------------------------------------------------------------

#: No new object in the merge path may exceed this.
MAX_MERGE_PATH_BLOB_BYTES = 10 * 1024 * 1024

#: Reported, not failed. Worth a reviewer's attention before it becomes a
#: habit; the repository's largest legitimate blob is a 3.3 MB inventory.
NOTABLE_MERGE_PATH_BLOB_BYTES = 5 * 1024 * 1024

#: Generated datasets and caches. Never acceptable at any size: a 1 KB pickle
#: of fitted marginals is still a cache, and a small parquet is still a
#: regenerated export that nothing reads.
GENERATED_ARTIFACT_NAMES: frozenset[str] = frozenset(
    {
        "oof_gaussian_residuals.parquet",
        "factor_loadings.parquet",
        "joint_event_grades.parquet",
    }
)
GENERATED_ARTIFACT_SUFFIXES: tuple[str, ...] = (
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".joblib",
    ".h5",
    ".onnx",
    ".pt",
    ".pth",
)


def tree_blobs(project_root: Path, ref: str = "HEAD") -> dict[str, int]:
    """``{path: bytes}`` for every blob in one commit's tree."""
    out: dict[str, int] = {}
    for line in _git(project_root, "ls-tree", "-r", "-l", ref).splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) < 4 or fields[1] != "blob" or fields[3] == "-":
            continue
        out[path] = int(fields[3])
    return out


def merge_path_blobs(project_root: Path, base: str | None = None) -> list[dict]:
    """Every blob a merge into production would add, with size and path.

    This is the question a tree listing cannot answer. ``git ls-tree`` reports
    what a branch *has*; a merge transfers what the branch's history
    *reaches*. A blob that was committed and later deleted is absent from
    every subsequent tree and still travels with an ordinary merge commit,
    because the commits that contain it come along too. The latent-state
    research branches are a live example: a 65 MB residual parquet was
    committed once and removed two days later, so it is invisible to
    ``git ls-tree HEAD`` and reachable from all three of their heads.

    Entries are sorted largest first. A blob may appear under more than one
    path across history; each (object, path) pair is reported once.
    """
    base = base or production_merge_base(project_root)
    if base is None:
        return []
    listing = _git(project_root, "rev-list", f"{base}..HEAD", "--objects")
    candidates: dict[tuple[str, str], None] = {}
    for line in listing.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            candidates[(sha, path)] = None
    if not candidates:
        return []

    probe = "\n".join(sha for sha, _ in candidates) + "\n"
    completed = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        cwd=str(project_root),
        input=probe,
        capture_output=True,
        text=True,
        check=True,
    )
    sizes: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] == "blob":
            sizes[fields[0]] = int(fields[2])

    out = [
        {"sha": sha, "path": path, "bytes": sizes[sha]}
        for sha, path in candidates
        if sha in sizes
    ]
    out.sort(key=lambda entry: (-entry["bytes"], entry["path"]))
    return out


def is_generated_artifact(path: str) -> bool:
    name = Path(path).name
    return name in GENERATED_ARTIFACT_NAMES or Path(
        path
    ).suffix in GENERATED_ARTIFACT_SUFFIXES


#: CI checks out a pull request, not a branch, so the base it should audit
#: against is the PR's target. GitHub puts that in ``GITHUB_BASE_REF``.
BASE_REF_ENV_VARS: tuple[str, ...] = ("PRODUCTION_BASE_REF", "GITHUB_BASE_REF")


def resolve_audit_base(project_root: Path) -> str | None:
    """The commit to audit against, preferring an explicitly configured base.

    Falls back to :func:`production_merge_base`. Returns ``None`` only when
    nothing resolves, which callers must treat as a failure rather than as an
    empty merge path: a guard that cannot see the base has not checked
    anything, and reporting that as a pass is how a 65 MB blob gets in.
    """
    for variable in BASE_REF_ENV_VARS:
        ref = os.environ.get(variable, "").strip()
        if not ref:
            continue
        for candidate in (ref, f"origin/{ref}"):
            try:
                return _git(project_root, "merge-base", "HEAD", candidate)
            except subprocess.CalledProcessError:
                continue
    return production_merge_base(project_root)


def audit_merge_path(project_root: Path, base: str | None = None) -> dict:
    """Classify the merge path's new objects against the blob contract.

    ``passed`` is false when a new object breaches the size ceiling, is a
    generated dataset, cache or model binary, or when the base could not be
    resolved at all. ``notable`` is informational.
    """
    resolved_base = base or resolve_audit_base(project_root)
    blobs = merge_path_blobs(project_root, resolved_base)
    oversized = [entry for entry in blobs if entry["bytes"] > MAX_MERGE_PATH_BLOB_BYTES]
    notable = [
        entry
        for entry in blobs
        if entry["bytes"] > NOTABLE_MERGE_PATH_BLOB_BYTES
        and entry["bytes"] <= MAX_MERGE_PATH_BLOB_BYTES
    ]
    generated = [entry for entry in blobs if is_generated_artifact(entry["path"])]
    return {
        "base": resolved_base,
        "base_resolved": resolved_base is not None,
        "new_blob_count": len(blobs),
        "largest": blobs[0] if blobs else None,
        "over_ceiling": oversized,
        "notable": notable,
        "generated_artifacts": generated,
        "ceiling_bytes": MAX_MERGE_PATH_BLOB_BYTES,
        "notable_bytes": NOTABLE_MERGE_PATH_BLOB_BYTES,
        "passed": bool(resolved_base) and not oversized and not generated,
    }
