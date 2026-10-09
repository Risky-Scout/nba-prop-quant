#!/usr/bin/env python3
"""Prove the production run is using its own per-run Python environment.

The production runner is a long-lived Mac. ``actions/setup-python`` reported a
hosted tool-cache interpreter while the install step's ``pip`` resolved to
``/Library/Frameworks/Python.framework/.../site-packages``, so every production
run installed its dependencies into machine-global site-packages and then ran
under whatever that machine had accumulated. The frozen scikit-learn pin still
held -- pip installs into whatever ``python`` resolves to, and the preflight
re-checks the version -- but nothing about the environment was reproducible,
and a package installed by hand between runs was indistinguishable from one the
lifecycle installed.

This verifies the fix rather than describing it: the workflow creates a fresh
virtual environment under ``RUNNER_TEMP`` and puts it on ``PATH``, and this
refuses to let the run continue unless the interpreter actually resolves inside
it. Everything it reports is read from the running interpreter, so a report
that says the environment is isolated is a report written by an isolated
interpreter.

It changes no dependency contract. The scikit-learn requirement is read from
``production_lifecycle_preflight``, which reads it from the freeze manifest, so
there is exactly one place that decides which version production serves under.

Exit codes
----------
0   every check passed
1   at least one check failed
2   the expected environment was not named
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import sysconfig
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))

#: The Python the production model was frozen under.
REQUIRED_PYTHON = (3, 12)

#: Prefixes that are somebody's machine rather than this run. A site-packages
#: directory under one of these is the defect this exists to catch: it is
#: shared between runs, it outlives the job, and nothing records what is in it.
#:
#: The macOS framework prefix is the one the audit actually found. The others
#: are the same mistake in the other shapes a self-hosted runner can take.
GLOBAL_SITE_PACKAGE_PREFIXES = (
    "/Library/Frameworks/",
    "/System/Library/",
    "/usr/local/lib/",
    "/opt/homebrew/",
    "/Users/Shared/",
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        name: str,
        passed: bool,
        detail: str,
        **values: Any,
    ) -> None:
        self.checks.append(
            Check(name=name, passed=passed, detail=detail, values=values)
        )

    @property
    def failed(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]

    @property
    def passed(self) -> bool:
        return not self.failed

    def payload(self) -> dict[str, Any]:
        return {
            "checks": [
                {
                    "detail": check.detail,
                    "name": check.name,
                    "passed": check.passed,
                    "values": check.values,
                }
                for check in self.checks
            ],
            "environment": self.facts,
            "failed": self.failed,
            "passed": self.passed,
        }


def site_package_paths() -> list[str]:
    """Every site-packages directory the running interpreter imports from."""
    return [
        entry
        for entry in sys.path
        if entry and "site-packages" in entry.replace("\\", "/")
    ]


def pip_module_location() -> str | None:
    try:
        import pip
    except Exception:
        return None

    return str(Path(pip.__file__).resolve().parent)


def environment_facts() -> dict[str, Any]:
    """What this run is actually running on.

    Recorded whether or not the checks pass, because a failed run is the one
    whose environment somebody needs to read.
    """
    return {
        "architecture": platform.machine(),
        "base_prefix": sys.base_prefix,
        "pip_executable": shutil.which("pip"),
        "pip_module_path": pip_module_location(),
        "platform": platform.platform(),
        "purelib": sysconfig.get_paths().get("purelib"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "scikit_learn_version": _installed_sklearn(),
        "site_packages": site_package_paths(),
        "venv_prefix": sys.prefix,
        "which_python": shutil.which("python"),
    }


def _installed_sklearn() -> str | None:
    from production_lifecycle_preflight import installed_sklearn_version

    return installed_sklearn_version()


def _frozen_sklearn() -> str | None:
    from production_lifecycle_preflight import frozen_sklearn_version

    return frozen_sklearn_version(ROOT)


def _inside(path: str | None, root: Path) -> bool:
    """Whether ``path`` lies under ``root``, under either normalisation.

    Both are tried because the two ways this can be asked disagree about
    symlinks and each needs a different answer. A virtual environment's
    ``bin/python`` is a symlink to the base interpreter, so resolving it leaves
    the environment it is unmistakably part of. A temporary directory is often
    itself reached through a symlink -- ``/tmp`` on macOS -- so refusing to
    resolve puts the environment outside the path it was created at. Either
    answer alone rejects a correct environment, so a path is inside when
    either normalisation says it is.
    """
    if not path:
        return False

    for candidate, base in (
        (Path(os.path.abspath(path)), Path(os.path.abspath(root))),
        (Path(path).resolve(), Path(root).resolve()),
    ):
        try:
            candidate.relative_to(base)
        except (ValueError, OSError):
            continue

        return True

    return False


def evaluate(
    expected_venv: Path,
    expected_architecture: str | None = None,
) -> Report:
    report = Report(facts=environment_facts())

    venv = Path(expected_venv).resolve()

    # 1. A virtual environment at all. sys.prefix diverging from
    #    sys.base_prefix is what makes an environment this run's own rather
    #    than the machine's.
    isolated = Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve()

    report.add(
        "interpreter_is_isolated",
        isolated,
        (
            f"sys.prefix {sys.prefix} "
            + ("differs from" if isolated else "is")
            + f" sys.base_prefix {sys.base_prefix}"
        ),
        base_prefix=sys.base_prefix,
        prefix=sys.prefix,
    )

    # 2. The specific environment this run created, not merely any venv.
    expected = _inside(sys.prefix, venv) and _inside(sys.executable, venv)

    report.add(
        "interpreter_resolves_inside_the_per_run_environment",
        expected,
        (
            f"{sys.executable} "
            + ("resolves inside" if expected else "resolves outside")
            + f" {venv}"
        ),
        executable=sys.executable,
        expected_venv=str(venv),
    )

    # 3. The frozen interpreter version.
    version_ok = sys.version_info[:2] == REQUIRED_PYTHON

    report.add(
        "python_version_matches_the_frozen_requirement",
        version_ok,
        (
            f"Python {platform.python_version()} against required "
            f"{'.'.join(str(part) for part in REQUIRED_PYTHON)}"
        ),
        observed=platform.python_version(),
        required=".".join(str(part) for part in REQUIRED_PYTHON),
    )

    # 4. The interpreter's architecture is the machine's. An x86_64 Python
    #    translated onto an arm64 host runs, imports and prices, and does it
    #    through a different numerical stack than the one the model was frozen
    #    under. The expected value comes from the shell's uname, which is not
    #    translated.
    if expected_architecture:
        architecture_ok = platform.machine() == expected_architecture

        report.add(
            "interpreter_architecture_matches_the_runner",
            architecture_ok,
            (
                f"interpreter reports {platform.machine()}, runner reports "
                f"{expected_architecture}"
            ),
            expected=expected_architecture,
            observed=platform.machine(),
        )

    # 5. Installs land in this environment. purelib is where pip puts a pure
    #    Python package, so it is the question "where would an install go"
    #    answered by the interpreter that would do it.
    purelib = sysconfig.get_paths().get("purelib")

    install_ok = _inside(purelib, venv)

    report.add(
        "installs_target_the_per_run_environment",
        install_ok,
        (
            f"purelib {purelib} "
            + ("is inside" if install_ok else "is outside")
            + f" {venv}"
        ),
        expected_venv=str(venv),
        purelib=purelib,
    )

    # 6. Nothing global is importable. This is the defect in its own terms:
    #    a site-packages directory that outlives the run.
    offenders = sorted(
        entry
        for entry in site_package_paths()
        if entry.replace("\\", "/").startswith(GLOBAL_SITE_PACKAGE_PREFIXES)
        or not _inside(entry, venv)
    )

    report.add(
        "no_machine_global_site_packages_are_importable",
        not offenders,
        (
            f"{len(offenders)} site-packages director(y/ies) outside {venv}"
            if offenders
            else f"every importable site-packages directory is inside {venv}"
        ),
        offenders=offenders,
    )

    # 7. The dependency contract, read from the freeze manifest rather than
    #    restated here.
    required = _frozen_sklearn()
    installed = _installed_sklearn()

    sklearn_ok = bool(required) and installed == required

    report.add(
        "scikit_learn_matches_the_frozen_pin",
        sklearn_ok,
        (
            f"installed {installed or 'none'} against frozen "
            f"{required or 'unrecorded'}"
        ),
        installed=installed,
        required=required,
    )

    return report


def render(report: Report, production_sha: str | None = None) -> str:
    facts = report.facts

    lines = [
        "## Python environment isolation",
        "",
        f"**{'PASS' if report.passed else 'FAIL'}**",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| python executable | `{facts['python_executable']}` |",
        f"| python version | {facts['python_version']} |",
        f"| architecture | {facts['architecture']} |",
        f"| virtual environment | `{facts['venv_prefix']}` |",
        f"| pip executable | `{facts['pip_executable']}` |",
        f"| pip module | `{facts['pip_module_path']}` |",
        f"| install target | `{facts['purelib']}` |",
        f"| scikit-learn | {facts['scikit_learn_version']} |",
    ]

    if production_sha:
        lines.append(f"| production code SHA | `{production_sha}` |")

    lines += ["", "| check | result | detail |", "| --- | --- | --- |"]

    for check in report.checks:
        lines.append(
            f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | "
            f"{check.detail} |"
        )

    if not report.passed:
        lines += [
            "",
            "The production lifecycle refuses to run outside its own "
            "environment. Failing here is the guard working: a run that "
            "installs into machine-global site-packages is not reproducible, "
            "so it must not produce production state.",
        ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-venv",
        required=True,
        help="the per-run virtual environment the interpreter must resolve in",
    )
    parser.add_argument(
        "--expected-architecture",
        default=None,
        help="the runner's architecture, as reported by uname -m",
    )
    parser.add_argument("--production-sha", default=None)
    parser.add_argument("--status-path", default=None)
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args(argv)

    if not str(args.expected_venv).strip():
        print(
            "no expected environment was named, so nothing was verified",
            file=sys.stderr,
        )
        return 2

    report = evaluate(
        Path(args.expected_venv),
        expected_architecture=(
            args.expected_architecture.strip()
            if args.expected_architecture
            else None
        ),
    )

    payload = report.payload()
    payload["production_sha"] = args.production_sha

    if args.status_path:
        path = Path(args.status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    rendered = render(report, production_sha=args.production_sha)

    if args.summary_path:
        with Path(args.summary_path).open("a", encoding="utf-8") as handle:
            handle.write(rendered)

    print(rendered)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
