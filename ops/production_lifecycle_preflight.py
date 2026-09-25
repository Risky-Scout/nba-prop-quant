#!/usr/bin/env python3
"""Preflight for the automated NBA production lifecycle.

GitHub Actions orchestrates; this decides. Everything the workflow needs to
know before it is allowed to touch production state is resolved here in Python
rather than in YAML conditionals, so the rules stay testable and stay in one
place.

It answers four questions, in order, and fails closed on the first that is not
satisfied:

    1. Is the checkout the authoritative production code?
    2. Are the frozen architecture contracts intact?
    3. Is every required secret present? (names only; values are never read)
    4. Is a durable production state backend configured?

Nothing here refreshes data, fits a model, promotes a fit or contacts an
external service. It reads the repository and the environment and reports.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# The branch that holds production code. GitHub only fires scheduled events
# from the repository default branch, which is not this branch, so a scheduled
# run must check this out explicitly rather than run whatever the default
# branch happens to contain.
AUTHORITATIVE_PRODUCTION_BRANCH = "production/wizardofodds-integration"

# Secrets the production lifecycle needs. Names only: this module never reads
# a secret's value and never prints one.
REQUIRED_PRODUCTION_SECRETS = ("BDL_API_KEY",)

OPTIONAL_PRODUCTION_SECRETS = ("ODDS_API_KEY",)

# Environment variables that must point at durable, run-to-run storage.
STATE_BACKEND_VARIABLES = (
    "NBA_PROP_DATA_DIR",
    "NBA_PROP_FIT_REGISTRY_DIR",
    "NBA_PROP_WORK_DIR",
)

BLOCKER_DURABLE_STATE = "DURABLE_PRODUCTION_STATE_BACKEND_REQUIRED"

MODE_VALIDATE_ONLY = "validate-only"
MODE_PRODUCTION = "production"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_AUTHORITATIVE = 3
EXIT_CONTRACT = 4
EXIT_MISSING_SECRET = 5
EXIT_NO_STATE_BACKEND = 6


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocker: str | None = None


@dataclass
class PreflightResult:
    mode: str
    checks: list[Check] = field(default_factory=list)

    def add(
        self,
        name: str,
        passed: bool,
        detail: str,
        blocker: str | None = None,
    ) -> Check:
        check = Check(name=name, passed=passed, detail=detail, blocker=blocker)

        self.checks.append(check)

        return check

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def blockers(self) -> list[str]:
        return sorted(
            {
                check.blocker
                for check in self.checks
                if check.blocker and not check.passed
            }
        )

    def as_dict(self) -> dict:
        return {
            "blockers": self.blockers,
            "checks": [
                {
                    "blocker": check.blocker,
                    "detail": check.detail,
                    "name": check.name,
                    "passed": check.passed,
                }
                for check in self.checks
            ],
            "mode": self.mode,
            "status": "ok" if self.ok else "failed",
        }


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.stdout.strip() if result.returncode == 0 else ""


def check_authoritative_checkout(
    result: PreflightResult,
    expected_ref: str | None,
) -> None:
    """Prove the workspace holds production code, not the default branch.

    A scheduled run starts from the default branch. If the checkout step were
    ever misconfigured the job would silently fit and promote using whatever
    code that branch carries, so the resolved commit is checked rather than
    assumed.
    """
    head = git("rev-parse", "HEAD")

    if not head:
        result.add(
            "authoritative_checkout",
            False,
            "not a git checkout, so the production ref cannot be proven",
        )

        return

    if expected_ref:
        expected = git("rev-parse", expected_ref) or expected_ref

        if head != expected:
            result.add(
                "authoritative_checkout",
                False,
                f"HEAD {head} is not the expected production ref {expected}",
            )

            return

        result.add(
            "authoritative_checkout",
            True,
            f"HEAD {head} matches the expected production ref",
        )

        return

    # Without an explicit expectation, require that HEAD is contained by the
    # authoritative production branch.
    remote = f"refs/remotes/origin/{AUTHORITATIVE_PRODUCTION_BRANCH}"

    contained = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, remote],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode == 0

    result.add(
        "authoritative_checkout",
        contained,
        (
            f"HEAD {head} is contained by {AUTHORITATIVE_PRODUCTION_BRANCH}"
            if contained
            else f"HEAD {head} is not contained by "
            f"{AUTHORITATIVE_PRODUCTION_BRANCH}"
        ),
    )


def check_frozen_contracts(result: PreflightResult) -> None:
    """Verify the frozen architecture before anything is allowed to fit."""
    try:
        from nba_prop_quant.adaptive_fit_registry import (
            load_architecture_contract,
        )
        from nba_prop_quant.adaptive_training import (
            UPDATE_PROTOCOL_RELATIVE_PATH,
            assert_protocol_matches_tree,
        )

        contract = load_architecture_contract(ROOT)

        protocol = json.loads(
            (ROOT / UPDATE_PROTOCOL_RELATIVE_PATH).read_text(encoding="utf-8")
        )

        assert_protocol_matches_tree(protocol, ROOT, contract)

    except Exception as error:
        result.add(
            "frozen_contracts",
            False,
            f"{type(error).__name__}: {error}",
        )

        return

    result.add(
        "frozen_contracts",
        True,
        (
            "architecture contract and adaptive update protocol match the "
            f"working tree (contract {contract.sha256[:12]})"
        ),
    )


def check_required_secrets(
    result: PreflightResult,
    environ: dict[str, str] | None = None,
) -> None:
    """Confirm presence by name. Values are never read, logged or returned."""
    env = os.environ if environ is None else environ

    missing = [
        name
        for name in REQUIRED_PRODUCTION_SECRETS
        if not str(env.get(name, "")).strip()
    ]

    absent_optional = [
        name
        for name in OPTIONAL_PRODUCTION_SECRETS
        if not str(env.get(name, "")).strip()
    ]

    detail = (
        f"missing required secret(s): {', '.join(missing)}"
        if missing
        else "every required secret is present"
    )

    if absent_optional and not missing:
        detail += f"; optional not set: {', '.join(absent_optional)}"

    result.add("required_secrets", not missing, detail)


def check_state_backend(
    result: PreflightResult,
    environ: dict[str, str] | None = None,
) -> None:
    """Require durable, run-to-run storage for production state.

    A GitHub-hosted runner is destroyed when the job ends. The fit registry's
    current-good pointer, the rolling data tree and the fitted artifacts all
    have to outlive the run, and none of them can live in git: the registry is
    documented as production state that does not belong in version control,
    and .gitignore excludes the data tree and every model binary.

    Pointing these at runner-local paths would not fail loudly; it would
    quietly reset the promotion pointer every night and re-ingest history from
    scratch. So the backend must be configured explicitly and must not resolve
    inside the repository or the runner's temporary space.
    """
    env = os.environ if environ is None else environ

    unset = [
        name
        for name in STATE_BACKEND_VARIABLES
        if not str(env.get(name, "")).strip()
    ]

    if unset:
        result.add(
            "durable_state_backend",
            False,
            (
                "no durable production state backend is configured: "
                f"{', '.join(unset)} unset. A GitHub-hosted runner does not "
                "survive the job, so production state cannot live on it."
            ),
            blocker=BLOCKER_DURABLE_STATE,
        )

        return

    ephemeral = []

    runner_temp = str(env.get("RUNNER_TEMP", "")).strip()

    for name in STATE_BACKEND_VARIABLES:
        candidate = Path(str(env[name])).expanduser()

        resolved = (
            candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        ).resolve()

        if resolved == ROOT or ROOT in resolved.parents:
            ephemeral.append(f"{name} resolves inside the repository")

        elif runner_temp and str(resolved).startswith(runner_temp):
            ephemeral.append(f"{name} resolves inside RUNNER_TEMP")

    if ephemeral:
        result.add(
            "durable_state_backend",
            False,
            (
                "configured state paths are not durable: "
                + "; ".join(sorted(ephemeral))
            ),
            blocker=BLOCKER_DURABLE_STATE,
        )

        return

    result.add(
        "durable_state_backend",
        True,
        "a durable production state backend is configured",
    )


def run_preflight(
    mode: str,
    expected_ref: str | None = None,
    environ: dict[str, str] | None = None,
) -> PreflightResult:
    result = PreflightResult(mode=mode)

    # Production always proves its checkout. validate-only does so only when
    # the caller names a ref, because CI runs against a pull request's
    # synthetic merge commit, which is correctly not yet on production; making
    # that a failure would mean no production PR could ever report a green
    # check.
    if mode == MODE_PRODUCTION or expected_ref:
        check_authoritative_checkout(result, expected_ref)

    check_frozen_contracts(result)

    # validate-only proves the workflow and the code are sound without
    # requiring production credentials or a state backend.
    if mode == MODE_PRODUCTION:
        check_required_secrets(result, environ)
        check_state_backend(result, environ)

    return result


def exit_code_for(result: PreflightResult) -> int:
    if result.ok:
        return EXIT_OK

    failed = {check.name for check in result.checks if not check.passed}

    if "authoritative_checkout" in failed:
        return EXIT_NOT_AUTHORITATIVE

    if "frozen_contracts" in failed:
        return EXIT_CONTRACT

    if "required_secrets" in failed:
        return EXIT_MISSING_SECRET

    if "durable_state_backend" in failed:
        return EXIT_NO_STATE_BACKEND

    return EXIT_ERROR


def render_summary(result: PreflightResult, head: str) -> str:
    lines = [
        "## NBA production lifecycle preflight",
        "",
        f"- mode: `{result.mode}`",
        f"- production code SHA: `{head}`",
        f"- status: **{'ok' if result.ok else 'failed'}**",
        "",
        "| check | result | detail |",
        "| --- | --- | --- |",
    ]

    for check in result.checks:
        detail = check.detail.replace("|", "\\|")

        lines.append(
            f"| {check.name} | {'pass' if check.passed else 'FAIL'} | "
            f"{detail} |"
        )

    if result.blockers:
        lines += ["", "### Blockers", ""]
        lines += [f"- `{blocker}`" for blocker in result.blockers]

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="production_lifecycle_preflight",
        description=(
            "Decide whether the automated NBA production lifecycle may "
            "proceed. Never prints a secret value."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=(MODE_VALIDATE_ONLY, MODE_PRODUCTION),
        default=MODE_VALIDATE_ONLY,
    )

    parser.add_argument(
        "--expected-ref",
        default=None,
        help="Ref or SHA the checkout must equal, for scheduled runs.",
    )

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Write a Markdown job summary here.",
    )

    parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help="Write the machine-readable result here.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    result = run_preflight(args.mode, expected_ref=args.expected_ref)

    payload = result.as_dict()

    head = git("rev-parse", "HEAD")

    payload["production_code_sha"] = head

    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.status_path:
        args.status_path.parent.mkdir(parents=True, exist_ok=True)

        args.status_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)

        with args.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(render_summary(result, head))

    return exit_code_for(result)


if __name__ == "__main__":
    raise SystemExit(main())
