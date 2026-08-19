from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

DEFAULT_SCAN_ROOTS = (
    Path("src/nba_prop_quant"),
    Path("scripts"),
    Path("ops"),
)

DEFAULT_SHELL_ROOTS = (
    Path("scripts"),
    Path("ops"),
)

DEFAULT_CONFIG_PATHS = (
    Path("configs"),
    Path("models/dynamic_params.json"),
    Path("models/mean_model_selection.json"),
    Path("models/ensemble_weights.json"),
    Path("models/marginal_selection.json"),
    Path("models/combo_dependence_policy.json"),
    Path("models/market_probability_calibration_policy.json"),
    Path("pyproject.toml"),
)


def json_safe(value: Any) -> Any:
    if value is Ellipsis:
        return "Ellipsis"

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return repr(value)


def literal_text(value: Any) -> str:
    safe = json_safe(value)

    if isinstance(safe, str):
        return safe

    return json.dumps(
        safe,
        sort_keys=True,
        ensure_ascii=False,
    )


def is_docstring_constant(
    node: ast.Constant,
    parents: dict[int, ast.AST],
) -> bool:
    parent = parents.get(
        id(node)
    )

    if not isinstance(
        parent,
        ast.Expr,
    ):
        return False

    grand = parents.get(
        id(parent)
    )

    if not isinstance(
        grand,
        (
            ast.Module,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    ):
        return False

    if not grand.body:
        return False

    return grand.body[0] is parent


def node_scope(
    node: ast.AST,
    parents: dict[int, ast.AST],
) -> str:
    names = []
    current = node

    while True:
        parent = parents.get(
            id(current)
        )

        if parent is None:
            break

        if isinstance(
            parent,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            names.append(
                parent.name
            )

        current = parent

    return ".".join(
        reversed(
            names
        )
    )


def parent_kind(
    node: ast.AST,
    parents: dict[int, ast.AST],
) -> str:
    parent = parents.get(
        id(node)
    )

    return (
        type(parent).__name__
        if parent is not None
        else ""
    )


def source_line(
    lines: list[str],
    lineno: int,
) -> str:
    if (
        lineno <= 0
        or lineno > len(
            lines
        )
    ):
        return ""

    return lines[
        lineno
        - 1
    ].strip()


def target_names(
    target: ast.AST,
) -> list[str]:
    if isinstance(
        target,
        ast.Name,
    ):
        return [
            target.id
        ]

    if isinstance(
        target,
        (
            ast.Tuple,
            ast.List,
        ),
    ):
        result = []

        for item in target.elts:
            result.extend(
                target_names(
                    item
                )
            )

        return result

    if isinstance(
        target,
        ast.Attribute,
    ):
        return [
            target.attr
        ]

    return []


def literal_eval_safe(
    node: ast.AST | None,
) -> tuple[bool, Any]:
    if node is None:
        return (
            True,
            None,
        )

    try:
        return (
            True,
            ast.literal_eval(
                node
            ),
        )
    except Exception:
        return (
            False,
            None,
        )


def build_parents(
    tree: ast.AST,
) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}

    for parent in ast.walk(
        tree
    ):
        for child in ast.iter_child_nodes(
            parent
        ):
            parents[
                id(child)
            ] = parent

    return parents


def assignment_scope_kind(
    node: ast.AST,
    parents: dict[int, ast.AST],
) -> str:
    parent = parents.get(
        id(node)
    )

    if isinstance(
        parent,
        ast.Module,
    ):
        return "module"

    if isinstance(
        parent,
        ast.ClassDef,
    ):
        return "class"

    if isinstance(
        parent,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        return "function"

    return "nested"


def scan_python(
    path: Path,
) -> tuple[
    list[dict],
    list[dict],
    dict,
]:
    relative = str(
        path.relative_to(
            ROOT
        )
    )

    text = path.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()

    try:
        tree = ast.parse(
            text,
            filename=relative,
        )
    except Exception as exc:
        return (
            [],
            [],
            {
                "path": relative,
                "status": "parse_error",
                "error": repr(
                    exc
                ),
            },
        )

    parents = build_parents(
        tree
    )

    literal_rows = []

    signed_constant_children: set[int] = set()

    for candidate in ast.walk(
        tree
    ):
        if not isinstance(
            candidate,
            ast.UnaryOp,
        ):
            continue

        if not isinstance(
            candidate.op,
            (
                ast.USub,
                ast.UAdd,
            ),
        ):
            continue

        if not isinstance(
            candidate.operand,
            ast.Constant,
        ):
            continue

        if not isinstance(
            candidate.operand.value,
            (
                int,
                float,
                complex,
            ),
        ):
            continue

        signed_constant_children.add(
            id(
                candidate.operand
            )
        )

        signed_value = (
            -candidate.operand.value
            if isinstance(
                candidate.op,
                ast.USub,
            )
            else candidate.operand.value
        )

        literal_rows.append(
            {
                "file": relative,
                "line": int(
                    getattr(
                        candidate,
                        "lineno",
                        0,
                    )
                ),
                "column": int(
                    getattr(
                        candidate,
                        "col_offset",
                        0,
                    )
                ),
                "scope": node_scope(
                    candidate,
                    parents,
                ),
                "parent_node": parent_kind(
                    candidate,
                    parents,
                ),
                "literal_type": type(
                    signed_value
                ).__name__,
                "value": literal_text(
                    signed_value
                ),
                "source_line": source_line(
                    lines,
                    int(
                        getattr(
                            candidate,
                            "lineno",
                            0,
                        )
                    ),
                ),
            }
        )

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Constant,
        ):
            continue

        if id(node) in signed_constant_children:
            continue

        if is_docstring_constant(
            node,
            parents,
        ):
            continue

        literal_rows.append(
            {
                "file": relative,
                "line": int(
                    getattr(
                        node,
                        "lineno",
                        0,
                    )
                ),
                "column": int(
                    getattr(
                        node,
                        "col_offset",
                        0,
                    )
                ),
                "scope": node_scope(
                    node,
                    parents,
                ),
                "parent_node": parent_kind(
                    node,
                    parents,
                ),
                "literal_type": type(
                    node.value
                ).__name__,
                "value": literal_text(
                    node.value
                ),
                "source_line": source_line(
                    lines,
                    int(
                        getattr(
                            node,
                            "lineno",
                            0,
                        )
                    ),
                ),
            }
        )

    named_rows = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Assign,
        ):
            ok, value = literal_eval_safe(
                node.value
            )

            if not ok:
                continue

            names = []

            for target in node.targets:
                names.extend(
                    target_names(
                        target
                    )
                )

            for name in names:
                named_rows.append(
                    {
                        "file": relative,
                        "line": int(
                            node.lineno
                        ),
                        "scope": node_scope(
                            node,
                            parents,
                        ),
                        "kind": (
                            assignment_scope_kind(
                                node,
                                parents,
                            )
                            + "_assignment"
                        ),
                        "name": name,
                        "value_type": type(
                            value
                        ).__name__,
                        "value": literal_text(
                            value
                        ),
                        "source_line": source_line(
                            lines,
                            int(
                                node.lineno
                            ),
                        ),
                    }
                )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            ok, value = literal_eval_safe(
                node.value
            )

            if not ok:
                continue

            names = target_names(
                node.target
            )

            for name in names:
                named_rows.append(
                    {
                        "file": relative,
                        "line": int(
                            node.lineno
                        ),
                        "scope": node_scope(
                            node,
                            parents,
                        ),
                        "kind": (
                            assignment_scope_kind(
                                node,
                                parents,
                            )
                            + "_annotated_assignment"
                        ),
                        "name": name,
                        "value_type": type(
                            value
                        ).__name__,
                        "value": literal_text(
                            value
                        ),
                        "source_line": source_line(
                            lines,
                            int(
                                node.lineno
                            ),
                        ),
                    }
                )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            positional_args = (
                list(
                    node.args.posonlyargs
                )
                + list(
                    node.args.args
                )
            )

            positional_defaults = list(
                node.args.defaults
            )

            if positional_defaults:
                start = (
                    len(
                        positional_args
                    )
                    - len(
                        positional_defaults
                    )
                )

                for arg, default in zip(
                    positional_args[
                        start:
                    ],
                    positional_defaults,
                ):
                    ok, value = literal_eval_safe(
                        default
                    )

                    if ok:
                        named_rows.append(
                            {
                                "file": relative,
                                "line": int(
                                    getattr(
                                        default,
                                        "lineno",
                                        node.lineno,
                                    )
                                ),
                                "scope": node_scope(
                                    node,
                                    parents,
                                ),
                                "kind": "function_default",
                                "name": (
                                    f"{node.name}.{arg.arg}"
                                ),
                                "value_type": type(
                                    value
                                ).__name__,
                                "value": literal_text(
                                    value
                                ),
                                "source_line": source_line(
                                    lines,
                                    int(
                                        getattr(
                                            default,
                                            "lineno",
                                            node.lineno,
                                        )
                                    ),
                                ),
                            }
                        )

            for arg, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
            ):
                if default is None:
                    continue

                ok, value = literal_eval_safe(
                    default
                )

                if ok:
                    named_rows.append(
                        {
                            "file": relative,
                            "line": int(
                                getattr(
                                    default,
                                    "lineno",
                                    node.lineno,
                                )
                            ),
                            "scope": node_scope(
                                node,
                                parents,
                            ),
                            "kind": "kwonly_default",
                            "name": (
                                f"{node.name}.{arg.arg}"
                            ),
                            "value_type": type(
                                value
                            ).__name__,
                            "value": literal_text(
                                value
                            ),
                            "source_line": source_line(
                                lines,
                                int(
                                    getattr(
                                        default,
                                        "lineno",
                                        node.lineno,
                                    )
                                ),
                            ),
                        }
                    )

        elif isinstance(
            node,
            ast.Call,
        ):
            func = node.func

            is_add_argument = (
                isinstance(
                    func,
                    ast.Attribute,
                )
                and func.attr
                == "add_argument"
            )

            if not is_add_argument:
                continue

            option_names = []

            for arg in node.args:
                ok, value = literal_eval_safe(
                    arg
                )

                if (
                    ok
                    and isinstance(
                        value,
                        str,
                    )
                ):
                    option_names.append(
                        value
                    )

            option_label = (
                "|".join(
                    option_names
                )
                or "add_argument"
            )

            for keyword in node.keywords:
                if keyword.arg not in {
                    "default",
                    "choices",
                    "const",
                }:
                    continue

                ok, value = literal_eval_safe(
                    keyword.value
                )

                if not ok:
                    continue

                named_rows.append(
                    {
                        "file": relative,
                        "line": int(
                            getattr(
                                keyword.value,
                                "lineno",
                                node.lineno,
                            )
                        ),
                        "scope": node_scope(
                            node,
                            parents,
                        ),
                        "kind": (
                            "argparse_"
                            + keyword.arg
                        ),
                        "name": (
                            option_label
                            + "."
                            + keyword.arg
                        ),
                        "value_type": type(
                            value
                        ).__name__,
                        "value": literal_text(
                            value
                        ),
                        "source_line": source_line(
                            lines,
                            int(
                                getattr(
                                    keyword.value,
                                    "lineno",
                                    node.lineno,
                                )
                            ),
                        ),
                    }
                )

    return (
        literal_rows,
        named_rows,
        {
            "path": relative,
            "status": "ok",
            "literal_occurrences": len(
                literal_rows
            ),
            "named_constants": len(
                named_rows
            ),
        },
    )


def flatten_config(
    value: Any,
    *,
    file: str,
    path: str = "",
) -> list[dict]:
    rows = []

    if isinstance(
        value,
        dict,
    ):
        for key in sorted(
            value,
            key=lambda item:
            str(item),
        ):
            next_path = (
                f"{path}.{key}"
                if path
                else str(
                    key
                )
            )

            rows.extend(
                flatten_config(
                    value[
                        key
                    ],
                    file=file,
                    path=next_path,
                )
            )

        return rows

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        for index, item in enumerate(
            value
        ):
            next_path = (
                f"{path}[{index}]"
            )

            rows.extend(
                flatten_config(
                    item,
                    file=file,
                    path=next_path,
                )
            )

        return rows

    rows.append(
        {
            "file": file,
            "config_path": path,
            "value_type": type(
                value
            ).__name__,
            "value": literal_text(
                value
            ),
        }
    )

    return rows


def parse_scalar(
    text: str,
) -> Any:
    value = text.strip()

    if not value:
        return ""

    lower = value.lower()

    if lower in {
        "null",
        "none",
        "~",
    }:
        return None

    if lower in {
        "true",
        "yes",
        "on",
    }:
        return True

    if lower in {
        "false",
        "no",
        "off",
    }:
        return False

    try:
        return ast.literal_eval(
            value
        )
    except Exception:
        pass

    try:
        if re.fullmatch(
            r"[-+]?\d+",
            value,
        ):
            return int(
                value
            )

        if re.fullmatch(
            r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?",
            value,
        ):
            return float(
                value
            )
    except Exception:
        pass

    return value.strip(
        "\"'"
    )


def parse_simple_yaml(
    path: Path,
) -> Any:
    try:
        import yaml  # type: ignore

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return yaml.safe_load(
                handle
            )
    except Exception:
        pass

    # Conservative fallback for ordinary mapping-style project YAML.
    root: dict[str, Any] = {}
    stack: list[
        tuple[int, dict[str, Any]]
    ] = [
        (
            -1,
            root,
        )
    ]

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not raw_line.strip():
            continue

        stripped = raw_line.lstrip()

        if stripped.startswith(
            "#"
        ):
            continue

        indent = (
            len(raw_line)
            - len(
                stripped
            )
        )

        # Remove simple trailing comments.
        content = stripped.split(
            " #",
            1,
        )[
            0
        ].rstrip()

        if ":" not in content:
            continue

        key, raw_value = content.split(
            ":",
            1,
        )

        key = key.strip()
        raw_value = raw_value.strip()

        while (
            len(
                stack
            )
            > 1
            and indent
            <= stack[
                -1
            ][
                0
            ]
        ):
            stack.pop()

        parent = stack[
            -1
        ][
            1
        ]

        if raw_value == "":
            child: dict[str, Any] = {}
            parent[
                key
            ] = child
            stack.append(
                (
                    indent,
                    child,
                )
            )
        else:
            parent[
                key
            ] = parse_scalar(
                raw_value
            )

    return root


def strip_shell_comment(line: str) -> str:
    """Remove unquoted shell comments conservatively."""
    out = []
    quote = None
    escaped = False

    for i, char in enumerate(line):
        if escaped:
            out.append(char)
            escaped = False
            continue

        if char == "\\":
            out.append(char)
            escaped = True
            continue

        if quote is not None:
            out.append(char)
            if char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
            out.append(char)
            continue

        if char == "#":
            if i == 0 or line[i - 1].isspace():
                break

        out.append(char)

    return "".join(out)


def scan_shell(path: Path) -> tuple[list[dict], dict]:
    relative = str(path.relative_to(ROOT))
    rows: list[dict] = []

    try:
        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()
    except Exception as exc:
        return (
            [],
            {
                "path": relative,
                "status": "parse_error",
                "error": repr(exc),
            },
        )

    assignment_pattern = re.compile(
        r"^\s*(?:export\s+|readonly\s+|local\s+)?"
        r"([A-Za-z_][A-Za-z0-9_]*)=(.*)$"
    )
    numeric_pattern = re.compile(
        r"(?<![A-Za-z0-9_.])[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?"
    )
    quoted_pattern = re.compile(
        r"""(?P<q>["'])(?P<value>(?:\\.|(?!\1).)*)\1"""
    )

    seen = set()

    for line_number, raw_line in enumerate(lines, start=1):
        line = strip_shell_comment(raw_line).rstrip()

        if not line.strip():
            continue

        assignment_match = assignment_pattern.match(line)

        if assignment_match:
            name = assignment_match.group(1)
            raw_value = assignment_match.group(2).strip()

            key = (
                line_number,
                "assignment",
                name,
                raw_value,
            )

            if key not in seen:
                seen.add(key)
                rows.append(
                    {
                        "file": relative,
                        "line": line_number,
                        "kind": "shell_assignment",
                        "name": name,
                        "value": raw_value,
                        "source_line": raw_line.strip(),
                    }
                )

        for match in numeric_pattern.finditer(line):
            value = match.group(0)

            key = (
                line_number,
                "numeric_literal",
                "",
                value,
                match.start(),
            )

            if key in seen:
                continue

            seen.add(key)
            rows.append(
                {
                    "file": relative,
                    "line": line_number,
                    "kind": "shell_numeric_literal",
                    "name": "",
                    "value": value,
                    "source_line": raw_line.strip(),
                }
            )

        for match in quoted_pattern.finditer(line):
            value = match.group("value")

            key = (
                line_number,
                "quoted_literal",
                "",
                value,
                match.start(),
            )

            if key in seen:
                continue

            seen.add(key)
            rows.append(
                {
                    "file": relative,
                    "line": line_number,
                    "kind": "shell_quoted_literal",
                    "name": "",
                    "value": value,
                    "source_line": raw_line.strip(),
                }
            )

    return (
        rows,
        {
            "path": relative,
            "status": "ok",
            "shell_constant_occurrences": len(rows),
        },
    )


def config_files() -> list[Path]:
    paths = []

    for configured in DEFAULT_CONFIG_PATHS:
        path = ROOT / configured

        if not path.exists():
            continue

        if path.is_dir():
            for pattern in (
                "*.json",
                "*.yaml",
                "*.yml",
                "*.toml",
            ):
                paths.extend(
                    sorted(
                        path.rglob(
                            pattern
                        )
                    )
                )
        else:
            paths.append(
                path
            )

    # De-duplicate.
    seen = set()
    result = []

    for path in paths:
        resolved = path.resolve()

        if resolved in seen:
            continue

        seen.add(
            resolved
        )
        result.append(
            path
        )

    return result


def scan_config(
    path: Path,
) -> tuple[
    list[dict],
    dict,
]:
    relative = str(
        path.relative_to(
            ROOT
        )
    )

    try:
        suffix = path.suffix.lower()

        if suffix == ".json":
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        elif suffix in {
            ".yaml",
            ".yml",
        }:
            payload = parse_simple_yaml(
                path
            )

        elif suffix == ".toml":
            payload = tomllib.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        else:
            return (
                [],
                {
                    "path": relative,
                    "status": "unsupported",
                },
            )

        rows = flatten_config(
            payload,
            file=relative,
        )

        return (
            rows,
            {
                "path": relative,
                "status": "ok",
                "leaf_constants": len(
                    rows
                ),
            },
        )

    except Exception as exc:
        return (
            [],
            {
                "path": relative,
                "status": "parse_error",
                "error": repr(
                    exc
                ),
            },
        )


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory every Python literal occurrence plus named/default "
            "constants and configuration leaves for the NBA prop model."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/validation/model_constant_inventory"
        ),
    )

    args = parser.parse_args()

    if not (
        ROOT
        / "src/nba_prop_quant"
    ).exists():
        raise SystemExit(
            "ERROR: run from nba_prop_quant_blueprint project root."
        )

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT
        / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    py_files = []

    for relative_root in DEFAULT_SCAN_ROOTS:
        root = ROOT / relative_root

        if not root.exists():
            continue

        py_files.extend(
            sorted(
                path
                for path in root.rglob(
                    "*.py"
                )
                if "__pycache__"
                not in path.parts
                and ".venv"
                not in path.parts
            )
        )

    literal_rows = []
    named_rows = []
    python_status = []

    for path in py_files:
        literals, named, status = scan_python(
            path
        )
        literal_rows.extend(
            literals
        )
        named_rows.extend(
            named
        )
        python_status.append(
            status
        )

    shell_files = []

    for relative_root in DEFAULT_SHELL_ROOTS:
        shell_root = ROOT / relative_root

        if not shell_root.exists():
            continue

        shell_files.extend(
            sorted(
                path
                for path in shell_root.rglob("*.sh")
                if "__pycache__" not in path.parts
                and ".venv" not in path.parts
            )
        )

    # De-duplicate shell files.
    shell_files = list(
        dict.fromkeys(shell_files)
    )

    shell_rows = []
    shell_status = []

    for path in shell_files:
        rows, status = scan_shell(path)
        shell_rows.extend(rows)
        shell_status.append(status)

    config_rows = []
    config_status = []

    for path in config_files():
        rows, status = scan_config(
            path
        )
        config_rows.extend(
            rows
        )
        config_status.append(
            status
        )

    parse_errors = [
        row
        for row in (
            python_status
            + shell_status
            + config_status
        )
        if row.get(
            "status"
        )
        == "parse_error"
    ]

    write_csv(
        output_dir
        / "FULL_CONSTANT_INVENTORY.csv",
        literal_rows,
        [
            "file",
            "line",
            "column",
            "scope",
            "parent_node",
            "literal_type",
            "value",
            "source_line",
        ],
    )

    write_csv(
        output_dir
        / "NAMED_CONSTANTS.csv",
        named_rows,
        [
            "file",
            "line",
            "scope",
            "kind",
            "name",
            "value_type",
            "value",
            "source_line",
        ],
    )

    write_csv(
        output_dir
        / "SHELL_CONSTANTS.csv",
        shell_rows,
        [
            "file",
            "line",
            "kind",
            "name",
            "value",
            "source_line",
        ],
    )

    write_csv(
        output_dir
        / "CONFIG_CONSTANTS.csv",
        config_rows,
        [
            "file",
            "config_path",
            "value_type",
            "value",
        ],
    )

    payload = {
        "schema_version": 1,
        "scope": {
            "python_scan_roots": [
                str(
                    item
                )
                for item in DEFAULT_SCAN_ROOTS
            ],
            "shell_scan_roots": [
                str(
                    item
                )
                for item in DEFAULT_SHELL_ROOTS
            ],
            "config_paths": [
                str(
                    item
                )
                for item in DEFAULT_CONFIG_PATHS
            ],
            "docstrings_excluded": True,
            "note": (
                "FULL_CONSTANT_INVENTORY records every non-docstring "
                "ast.Constant occurrence in scanned Python source. "
                "NAMED_CONSTANTS records literal assignments/defaults. "
                "CONFIG_CONSTANTS records scalar configuration/policy leaves."
            ),
        },
        "python_files": python_status,
        "shell_files": shell_status,
        "config_files": config_status,
        "literal_inventory": literal_rows,
        "named_constants": named_rows,
        "shell_constants": shell_rows,
        "config_constants": config_rows,
        "parse_errors": parse_errors,
    }

    (
        output_dir
        / "FULL_CONSTANT_INVENTORY.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    literal_type_counts = Counter(
        row[
            "literal_type"
        ]
        for row in literal_rows
    )

    named_kind_counts = Counter(
        row[
            "kind"
        ]
        for row in named_rows
    )

    file_counts = Counter(
        row[
            "file"
        ]
        for row in literal_rows
    )

    lines = [
        "# NBA Prop Model Constant Inventory",
        "",
        f"- Python files scanned: **{len(py_files)}**",
        f"- Python literal occurrences: **{len(literal_rows)}**",
        f"- Named/default constants: **{len(named_rows)}**",
        f"- Shell constant/literal occurrences: **{len(shell_rows)}**",
        f"- Configuration/policy leaf constants: **{len(config_rows)}**",
        f"- Parse errors: **{len(parse_errors)}**",
        "",
        "## Scope",
        "",
        "The literal inventory includes every non-docstring `ast.Constant` "
        "occurrence in `src/nba_prop_quant`, `scripts`, and `ops`. "
        "This deliberately includes strings/column names/messages as well as "
        "numeric thresholds so the register cannot silently omit a behavior-affecting literal.",
        "",
        "The named register separately identifies literal module/class assignments, "
        "function defaults, keyword-only defaults, and argparse defaults/choices/const values.",
        "",
        "The shell register records hard-coded shell assignments plus numeric and quoted literal occurrences from `scripts/**/*.sh` and `ops/**/*.sh`.",

        "The config register recursively records scalar leaves from model/config JSON, YAML, and TOML inputs.",
        "",
        "## Literal types",
        "",
    ]

    for key, value in sorted(
        literal_type_counts.items()
    ):
        lines.append(
            f"- `{key}`: {value}"
        )

    lines += [
        "",
        "## Named constant kinds",
        "",
    ]

    for key, value in sorted(
        named_kind_counts.items()
    ):
        lines.append(
            f"- `{key}`: {value}"
        )

    lines += [
        "",
        "## Python files by literal count",
        "",
    ]

    for file, count in sorted(
        file_counts.items(),
        key=lambda item:
        (
            -item[
                1
            ],
            item[
                0
            ],
        ),
    ):
        lines.append(
            f"- `{file}`: {count}"
        )

    if parse_errors:
        lines += [
            "",
            "## BLOCKING parse errors",
            "",
        ]

        for row in parse_errors:
            lines.append(
                f"- `{row.get('path')}`: {row.get('error')}"
            )

    (
        output_dir
        / "CONSTANT_INVENTORY_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 118)
    print("NBA PROP MODEL CONSTANT INVENTORY")
    print("=" * 118)
    print(
        f"Python files scanned:        {len(py_files):,}"
    )
    print(
        f"Literal occurrences:         {len(literal_rows):,}"
    )
    print(
        f"Named/default constants:     {len(named_rows):,}"
    )
    print(
        f"Shell constants/literals:    {len(shell_rows):,}"
    )
    print(
        f"Config/policy leaf constants:{len(config_rows):,}"
    )
    print(
        f"Parse errors:                {len(parse_errors):,}"
    )
    print()
    print(
        f"Saved: {output_dir}"
    )

    if parse_errors:
        raise SystemExit(
            "FAIL: constant inventory has parse errors; "
            "package construction must not continue."
        )

    print()
    print(
        "PASS: exhaustive constant register completed for the configured source scope."
    )


if __name__ == "__main__":
    main()
