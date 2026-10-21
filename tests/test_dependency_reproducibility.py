"""Tests for the pinned dependency set production and CI install under.

Three properties, none of which this repository could see before the
constraint file existed.

That the file is *complete*: every package the resolver reaches from this
project's own metadata is named in it. A package that is not named is not
constrained, because ``pip install -c`` constrains only what it names, so an
omission is the same morning-by-morning drift one level down.

That the file *agrees with the environment the suite just ran under*. A
pinned version nobody installed is a claim, not a proof.

And that the workflows actually install through it. A complete, agreed file
that no ``pip install`` references changes nothing about what production
resolves.

Nothing here installs, downloads, trains or serves. Every check reads
installed metadata, the constraint file, or workflow YAML.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[1]

CONSTRAINTS = REPO / "constraints" / "py312-production.txt"

PYPROJECT = REPO / "pyproject.toml"

WORKFLOW_DIR = REPO / ".github" / "workflows"

DISTRIBUTION = "nba-prop-quant"

#: Extras whose dependencies production is never asked to resolve. ``dev`` is
#: deliberately absent: CI installs it, so it is pinned and checked like the
#: rest. ``sequence`` pulls torch, which is research-only and not part of any
#: environment this file governs.
UNGOVERNED_EXTRAS = ("sequence",)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_workflows", REPO / "ops" / "validate_workflows.py"
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


validator = _load_validator()


def canonical(name: str) -> str:
    """The one spelling of a distribution name, per PEP 503."""
    return name.lower().replace("_", "-").replace(".", "-")


def pinned_versions() -> dict[str, set[str]]:
    """Every name the constraint file pins, and the versions it allows it.

    A name maps to more than one version only where the pins carry disjoint
    environment markers, which is how XGBoost is selected by platform.
    """
    found: dict[str, set[str]] = {}

    for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        bare = line.split("#")[0].strip()

        if not bare:
            continue

        specifier, _, _marker = bare.partition(";")

        name, separator, version = specifier.partition("==")

        assert separator, f"{bare!r} is not an exact pin"

        found.setdefault(canonical(name.strip()), set()).add(version.strip())

    return found


def parse(requirement: str) -> Requirement:
    return Requirement(requirement)


def applicable(requirement: Requirement, extras: frozenset[str]) -> bool:
    """Whether a package asked for with ``extras`` resolves this requirement.

    ``extra`` in a marker names an extra of the package the requirement came
    *from*, so it must be evaluated against that package's requested extras
    and no others. Evaluating it against this project's extra names instead
    would pull in every upstream package's own ``test`` and ``dev`` trees,
    which pip never installs and the constraint file has no business pinning.
    """
    if requirement.marker is None:
        return True

    # A marker that names no extra still has to be given one to evaluate.
    return any(
        requirement.marker.evaluate({"extra": extra})
        for extra in extras or {""}
    )


def declared_extras() -> tuple[str, ...]:
    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    return tuple(payload["project"].get("optional-dependencies", {}))


def requested_extras() -> frozenset[str]:
    """The extras an install this file governs asks this project for."""
    return frozenset(
        extra
        for extra in declared_extras()
        if extra not in UNGOVERNED_EXTRAS
    )


def resolved_closure() -> dict[str, str]:
    """Every distribution reachable from this project, at its installed version.

    Walked from installed metadata rather than from the network, so the answer
    describes the environment the suite is running in rather than one a
    resolver would build today. A requirement that is reachable but not
    installed is reported as a failure: in an environment built from the
    constraint file there should be none.
    """
    found: dict[str, str] = {}

    missing: list[str] = []

    seen: set[tuple[str, frozenset[str]]] = set()

    pending = [(canonical(DISTRIBUTION), requested_extras())]

    while pending:
        node = pending.pop()

        if node in seen:
            continue

        seen.add(node)

        name, extras = node

        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue

        if name != canonical(DISTRIBUTION):
            found[name] = distribution.version

        for text in distribution.requires or []:
            requirement = parse(text)

            if applicable(requirement, extras):
                pending.append(
                    (
                        canonical(requirement.name),
                        frozenset(requirement.extras),
                    )
                )

    assert missing == [], f"reachable but not installed: {sorted(missing)}"

    return found


# ----------------------------------------------------------------------
# the file is complete, and describes this environment
# ----------------------------------------------------------------------


def test_every_reachable_dependency_is_pinned():
    pinned = pinned_versions()

    unpinned = sorted(
        name for name in resolved_closure() if name not in pinned
    )

    assert unpinned == [], (
        "these packages are installed by this project but not pinned in "
        f"{CONSTRAINTS.name}, so a fresh environment resolves them against "
        f"whatever was published that morning: {unpinned}"
    )


def test_the_pinned_versions_are_the_installed_ones():
    pinned = pinned_versions()

    disagreements = {
        name: (version, sorted(pinned[name]))
        for name, version in resolved_closure().items()
        if version not in pinned[name]
    }

    assert disagreements == {}, (
        f"{CONSTRAINTS.name} pins versions this environment did not install, "
        "so the suite proved a different set than production would build: "
        f"{disagreements}"
    )


def test_the_file_pins_exact_versions_only():
    text = CONSTRAINTS.read_text(encoding="utf-8")

    for number, line in enumerate(text.splitlines(), 1):
        bare = line.split("#")[0].strip()

        if not bare:
            continue

        specifier = bare.split(";")[0]

        assert "==" in specifier, f"{CONSTRAINTS.name}:{number}: {line!r}"

        for loose in (">=", "<=", "~=", ">", "<"):
            assert loose not in specifier, (
                f"{CONSTRAINTS.name}:{number}: {loose!r} is a range, and a "
                "range is what this file exists to replace"
            )


def test_every_dependency_pyproject_declares_is_pinned():
    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    declared = list(payload["project"]["dependencies"])

    for extra, requirements in (
        payload["project"].get("optional-dependencies", {}).items()
    ):
        if extra not in UNGOVERNED_EXTRAS:
            declared += list(requirements)

    pinned = pinned_versions()

    absent = sorted(
        {
            canonical(parse(requirement).name)
            for requirement in declared
            if canonical(parse(requirement).name) not in pinned
        }
    )

    assert absent == [], (
        f"pyproject.toml declares these and {CONSTRAINTS.name} does not pin "
        f"them: {absent}"
    )


# ----------------------------------------------------------------------
# the workflows install through it
# ----------------------------------------------------------------------


def workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def installs_in(name: str) -> list[str]:
    jobs = workflow(name).get("jobs") or {}

    return [
        command
        for _, step in validator.steps_of(jobs)
        for command in validator.shell_commands(str(step.get("run", "")))
        if "pip install" in command
    ]


@pytest.mark.parametrize("name", validator.CONSTRAINED_INSTALL_WORKFLOWS)
def test_every_install_names_the_constraint_file(name):
    commands = installs_in(name)

    assert commands, f"{name} installs nothing"

    for command in commands:
        assert validator.PRODUCTION_CONSTRAINT_FILE in command, command


@pytest.mark.parametrize(
    "name", (validator.PRODUCTION_WORKFLOW, validator.T20_WORKFLOW)
)
def test_production_installs_neither_the_dev_extra_nor_a_new_pip(name):
    text = (WORKFLOW_DIR / name).read_text(encoding="utf-8")

    executable = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )

    for token in validator.PRODUCTION_FORBIDDEN_INSTALL_TOKENS:
        assert token not in executable, f"{name}: {token}"


def test_ci_installs_the_dev_extra_through_the_same_file():
    commands = installs_in(validator.CI_WORKFLOW)

    assert any(
        ".[dev]" in command
        and validator.PRODUCTION_CONSTRAINT_FILE in command
        for command in commands
    ), commands


def test_no_workflow_upgrades_pip():
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.strip().startswith("#"):
                continue

            assert "install --upgrade pip" not in line, f"{path.name}:{number}"


def test_the_constraint_file_is_the_cache_key_where_there_is_a_cache():
    """A cache keyed on anything else is a cache that serves a stale set.

    Only the GitHub-hosted job caches at all: the self-hosted runner already
    reuses its own pip cache directory across runs, and adding a second
    caching system on top of it would be two caches for one directory.
    """
    cached = [
        (name, step)
        for name in sorted(path.name for path in WORKFLOW_DIR.glob("*.yml"))
        for _, step in validator.steps_of(workflow(name).get("jobs") or {})
        if (step.get("with") or {}).get("cache")
    ]

    assert [name for name, _ in cached] == [validator.CI_WORKFLOW]

    for name, step in cached:
        with_ = step["with"]

        assert with_["cache"] == "pip", name

        assert (
            with_["cache-dependency-path"]
            == validator.PRODUCTION_CONSTRAINT_FILE
        ), name
