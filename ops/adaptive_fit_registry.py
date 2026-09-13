"""Operate the immutable adaptive fit registry.

Subcommands:

    register            register a staged candidate tree as an immutable fit
    verify              re-derive identity and re-hash every stored byte
    record-validation   record validation results for a registered fit
    promote             promote a validated fit under an exclusive lock
    current             print the current and previous good fit
    rollback            restore the previous good fit

This tool never trains, refits or recalibrates anything, never promotes
implicitly during registration, and never replaces an existing fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "src") not in sys.path:
    # Keeps the CLI usable on a serving host with no editable install.
    sys.path.insert(0, str(ROOT / "src"))

from nba_prop_quant.adaptive_fit_registry import (  # noqa: E402
    ArchitectureContractViolation,
    FitAlreadyExists,
    FitMetadata,
    FitNotFound,
    FitRegistry,
    IntegrityError,
    PromotionLocked,
    PromotionRefused,
    REGISTRY_ENV_VAR,
    REQUIRED_VALIDATION_CHECKS,
    RegistryError,
    RegistryRootError,
    StagedTreeError,
    ValidationRefused,
    resolve_registry_root,
)


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_CONTRACT = 3
EXIT_INTEGRITY = 4
EXIT_VALIDATION = 5
EXIT_PROMOTION = 6
EXIT_FIT_STATE = 7
EXIT_REGISTRY_ROOT = 8


_EXIT_BY_ERROR = (
    (ArchitectureContractViolation, EXIT_CONTRACT),
    (IntegrityError, EXIT_INTEGRITY),
    (StagedTreeError, EXIT_INTEGRITY),
    (ValidationRefused, EXIT_VALIDATION),
    (PromotionLocked, EXIT_PROMOTION),
    (PromotionRefused, EXIT_PROMOTION),
    (FitAlreadyExists, EXIT_FIT_STATE),
    (FitNotFound, EXIT_FIT_STATE),
    (RegistryRootError, EXIT_REGISTRY_ROOT),
    (RegistryError, EXIT_ERROR),
)


def exit_code_for(error: Exception) -> int:
    for kind, code in _EXIT_BY_ERROR:
        if isinstance(error, kind):
            return code

    return EXIT_ERROR


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help=(
            "Registry root directory. Falls back to "
            f"{REGISTRY_ENV_VAR}. Required: there is no default, and "
            "the registry may not live inside the repository."
        ),
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Repository root holding the architecture contract.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="adaptive_fit_registry",
        description=(
            "Immutable adaptive fit registry with fail-closed "
            "promotion and last-good rollback. This tool does not "
            "train models."
        ),
    )

    sub = parser.add_subparsers(
        dest="command", required=True
    )

    register = sub.add_parser(
        "register",
        help=(
            "Register a staged candidate tree. Never promotes."
        ),
    )
    add_common(register)
    register.add_argument(
        "--staged-dir",
        type=Path,
        required=True,
        help="Directory holding the candidate artifact tree.",
    )
    register.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="JSON file of fit provenance metadata.",
    )

    verify = sub.add_parser(
        "verify",
        help="Re-derive identity and re-hash a registered fit.",
    )
    add_common(verify)
    verify.add_argument(
        "--fit-id",
        default=None,
        help="Fit to verify. Defaults to the current good fit.",
    )

    validation = sub.add_parser(
        "record-validation",
        help="Record validation results outside the fit directory.",
    )
    add_common(validation)
    validation.add_argument(
        "--fit-id", required=True
    )
    validation.add_argument(
        "--checks",
        type=Path,
        required=True,
        help=(
            "JSON file mapping validation check name to boolean. "
            "Required checks: "
            + ", ".join(REQUIRED_VALIDATION_CHECKS)
        ),
    )
    validation.add_argument("--actor", default=None)
    validation.add_argument("--notes", default=None)

    promote = sub.add_parser(
        "promote",
        help="Promote a validated fit to current good.",
    )
    add_common(promote)
    promote.add_argument("--fit-id", required=True)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--actor", default=None)
    promote.add_argument(
        "--no-wait",
        action="store_true",
        help="Fail instead of waiting for the promotion lock.",
    )

    current = sub.add_parser(
        "current",
        help="Print the current and previous good fit.",
    )
    add_common(current)

    rollback = sub.add_parser(
        "rollback",
        help="Restore the previous good fit.",
    )
    add_common(rollback)
    rollback.add_argument("--reason", required=True)
    rollback.add_argument("--actor", default=None)
    rollback.add_argument(
        "--no-wait",
        action="store_true",
        help="Fail instead of waiting for the promotion lock.",
    )

    return parser.parse_args(argv)


def build_registry(
    args: argparse.Namespace,
) -> FitRegistry:
    project_root = Path(args.project_root).resolve()

    root = resolve_registry_root(
        args.registry_root, project_root=project_root
    )

    return FitRegistry(
        root=root, project_root=project_root
    )


def load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def command_register(
    args: argparse.Namespace,
) -> dict:
    registry = build_registry(args)

    metadata = FitMetadata.from_dict(
        load_json_file(args.metadata)
    )

    fit_id = registry.register(
        staged_dir=Path(args.staged_dir).resolve(),
        metadata=metadata,
    )

    return {
        "command": "register",
        "fit_id": fit_id,
        "fit_dir": str(
            registry.fit_dir(fit_id).relative_to(
                registry.root
            )
        ),
        "promoted": False,
        "registry_root": str(registry.root),
        "status": "registered_not_promoted",
    }


def command_verify(args: argparse.Namespace) -> dict:
    registry = build_registry(args)

    fit_id = args.fit_id

    if fit_id is None:
        fit_id = registry.current()[
            "current_good_fit_id"
        ]

        if fit_id is None:
            raise FitNotFound(
                "no fit specified and no current good fit is set"
            )

    manifest = registry.verify(fit_id)

    validation = registry.load_validation(fit_id)

    return {
        "artifact_count": len(
            manifest["artifact_hashes"]
        ),
        "command": "verify",
        "fit_id": fit_id,
        "integrity": "ok",
        "is_current_good": (
            registry.current()["current_good_fit_id"]
            == fit_id
        ),
        "validation_passed": (
            None
            if validation is None
            else bool(validation.get("passed"))
        ),
    }


def command_record_validation(
    args: argparse.Namespace,
) -> dict:
    registry = build_registry(args)

    checks = load_json_file(args.checks)

    record = registry.record_validation(
        fit_id=args.fit_id,
        checks=checks,
        actor=args.actor,
        notes=args.notes,
    )

    return {
        "command": "record-validation",
        "failed_checks": record["failed_checks"],
        "fit_id": args.fit_id,
        "missing_checks": record["missing_checks"],
        "passed": record["passed"],
    }


def command_promote(args: argparse.Namespace) -> dict:
    registry = build_registry(args)

    state = registry.promote(
        fit_id=args.fit_id,
        reason=args.reason,
        actor=args.actor,
        blocking=not args.no_wait,
    )

    return {
        "command": "promote",
        "current_good_fit_id": state[
            "current_good_fit_id"
        ],
        "previous_good_fit_id": state[
            "previous_good_fit_id"
        ],
        "promoted_at": state["promoted_at"],
    }


def command_current(args: argparse.Namespace) -> dict:
    registry = build_registry(args)

    state = registry.current()

    state["command"] = "current"
    state["registered_fits"] = len(registry.list_fits())

    return state


def command_rollback(args: argparse.Namespace) -> dict:
    registry = build_registry(args)

    state = registry.rollback(
        reason=args.reason,
        actor=args.actor,
        blocking=not args.no_wait,
    )

    return {
        "command": "rollback",
        "current_good_fit_id": state[
            "current_good_fit_id"
        ],
        "previous_good_fit_id": state[
            "previous_good_fit_id"
        ],
        "rolled_back_at": state["promoted_at"],
    }


COMMANDS = {
    "register": command_register,
    "verify": command_verify,
    "record-validation": command_record_validation,
    "promote": command_promote,
    "current": command_current,
    "rollback": command_rollback,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    handler = COMMANDS[args.command]

    try:
        payload = handler(args)
    except RegistryError as error:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "error": type(error).__name__,
                    "message": str(error),
                    "status": "failed",
                },
                sort_keys=True,
                indent=2,
            ),
            file=sys.stderr,
        )

        return exit_code_for(error)
    except (OSError, ValueError, KeyError) as error:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "error": type(error).__name__,
                    "message": str(error),
                    "status": "failed",
                },
                sort_keys=True,
                indent=2,
            ),
            file=sys.stderr,
        )

        return EXIT_ERROR

    payload["status"] = payload.get("status", "ok")

    emit(payload)

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
