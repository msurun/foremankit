"""Mechanical check of the measurable clean-code rules from config/conventions.md (python files only).

What a machine can judge: function length, number of parameters, nesting depth, missing docstrings, bare `except`,
one-letter names. Whether a name is GOOD or a comment explains the WHY stays a matter for the prompt and for review.
Used by `foremankit check <files>`; add it to `verify` to make a delegated file pass these rules before it is accepted.

A rule can be waived for ONE function on purpose, but only the rules in WAIVABLE_RULES: put
`# codecheck: allow <rule> (reason)` on the `def` line or on the line right above it. A waiver without a non-blank
reason in parentheses, or for a rule not in WAIVABLE_RULES, is ignored and the issue is still reported.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LIMITS = {
    "max_function_lines": 20,
    "max_parameters": 4,
    "max_nesting": 3,
    "allowed_short_names": ["i", "j", "k", "_"],
}
NESTING_STATEMENTS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
SELF_PARAMETERS = {"self", "cls"}
# Only a signature fixed by a protocol (e.g. an MCP tool schema) has a legitimate reason to break the parameter limit.
# Waiving any other rule would let delegated code dodge `foremankit check`, so the other rules stay strict.
WAIVABLE_RULES = ("too-many-parameters",)
# `# codecheck: allow <rule> (reason)`: the reason must contain at least one non-blank character.
WAIVER_PATTERN = re.compile(r"codecheck:\s*allow\s+([\w-]+)\s*\(([^()]*\S[^()]*)\)")


@dataclass(frozen=True)
class Issue:
    """One rule broken at one place."""
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.message}"


def limits_from(routing: dict[str, Any] | None) -> dict[str, Any]:
    """The limits of routing.yaml `conventions.limits` on top of the defaults."""
    given = ((routing or {}).get("conventions") or {}).get("limits") or {}
    return {**DEFAULT_LIMITS, **given}


def check_source(source: str, path: str, limits: dict[str, Any] | None = None) -> list[Issue]:
    """All issues in one python source text, in line order. A syntax error is reported as an issue, not raised."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [Issue(path, error.lineno or 1, "syntax-error", str(error.msg))]
    issues = _missing_docstring(tree, path, is_module=True)
    lines = source.splitlines()
    for node in ast.walk(tree):
        issues += _node_issues(node, path, limits, lines)
    issues += _short_names(tree, path, set(limits["allowed_short_names"]))
    unwaived = [issue for issue in issues if not _is_waived(issue, lines)]
    return sorted(unwaived, key=lambda issue: (issue.line, issue.rule))


def _node_issues(node: ast.AST, path: str, limits: dict[str, Any], lines: list[str]) -> list[Issue]:
    """The issues that belong to one syntax node (a function, a class or a bare `except:`)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _function_issues(node, path, limits, lines)
    if isinstance(node, ast.ClassDef):
        return _missing_docstring(node, path)
    if isinstance(node, ast.ExceptHandler) and node.type is None:
        return [Issue(path, node.lineno, "bare-except", "name the exception you expect instead of `except:`")]
    return []


def _is_waived(issue: Issue, lines: list[str]) -> bool:
    """True when the issue's line, or the line above it, carries a valid waiver for that rule (see WAIVER_PATTERN)."""
    if issue.rule not in WAIVABLE_RULES:
        return False
    nearby = lines[max(0, issue.line - 2):issue.line]
    return any(match.group(1) == issue.rule
               for line in nearby for match in WAIVER_PATTERN.finditer(line))


def check_files(paths: list[str], limits: dict[str, Any] | None = None) -> list[Issue]:
    """Check several files; a file that cannot be read is an issue too."""
    issues: list[Issue] = []
    for name in paths:
        try:
            text = Path(name).read_text(encoding="utf-8")
        except OSError as error:
            issues.append(Issue(name, 1, "unreadable", str(error)))
            continue
        issues += check_source(text, name, limits)
    return issues


def _is_public(name: str) -> bool:
    return not name.startswith("_") and not name.startswith("test_")


def _missing_docstring(node: ast.AST, path: str, is_module: bool = False) -> list[Issue]:
    """Modules, classes and public functions need a docstring (test functions explain themselves by their name)."""
    name = "module" if is_module else getattr(node, "name", "")
    if not is_module and not _is_public(name):
        return []
    if ast.get_docstring(node):
        return []
    return [Issue(path, getattr(node, "lineno", 1), "missing-docstring", f"{name} has no docstring")]


def _code_line_count(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> int:
    """Lines of code in a function: blank lines, comment-only lines and the docstring are not counted.

    Comments and docstrings are wanted (they teach the reader the model), so they must not push a
    function over the length limit; the limit is about how much logic one function holds.
    """
    first_statement = node.body[0]
    docstring_lines: set[int] = set()
    if ast.get_docstring(node) is not None:
        docstring_lines = set(range(first_statement.lineno, (first_statement.end_lineno or first_statement.lineno) + 1))
    last = node.end_lineno or node.lineno
    code = [number for number in range(node.lineno, last + 1)
            if number not in docstring_lines and lines[number - 1].strip() != ""
            and not lines[number - 1].lstrip().startswith("#")]
    return len(code)


def _function_issues(node: ast.FunctionDef | ast.AsyncFunctionDef, path: str, limits: dict[str, Any],
                     lines: list[str]) -> list[Issue]:
    issues = _missing_docstring(node, path)
    length = _code_line_count(node, lines)
    if length > limits["max_function_lines"]:
        issues.append(Issue(path, node.lineno, "function-too-long",
                            f"{node.name} has {length} code lines (limit {limits['max_function_lines']}); split it"))
    parameters = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    count = len([name for name in parameters if name not in SELF_PARAMETERS])
    if count > limits["max_parameters"]:
        issues.append(Issue(path, node.lineno, "too-many-parameters",
                            f"{node.name} takes {count} parameters (limit {limits['max_parameters']}); group them"))
    depth = _deepest_nesting(node.body)
    if depth > limits["max_nesting"]:
        issues.append(Issue(path, node.lineno, "too-deep-nesting",
                            f"{node.name} nests {depth} levels (limit {limits['max_nesting']}); extract a function"))
    return issues


def _deepest_nesting(statements: list[ast.stmt], depth: int = 0) -> int:
    """How many control-flow statements (if, for, while, try, with) are nested at the deepest point."""
    deepest = depth
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue                               # a nested function is judged on its own
        inner = depth + 1 if isinstance(statement, NESTING_STATEMENTS) else depth
        for field in ("body", "orelse", "finalbody", "handlers"):
            children = getattr(statement, field, None)
            if children:
                deepest = max(deepest, _deepest_nesting(_as_statements(children), inner))
        deepest = max(deepest, inner)
    return deepest


def _as_statements(children: list[Any]) -> list[ast.stmt]:
    """`handlers` holds ExceptHandler nodes: look inside them as if they were statement lists."""
    flat: list[ast.stmt] = []
    for child in children:
        flat += child.body if isinstance(child, ast.ExceptHandler) else [child]
    return flat


def _short_names(tree: ast.AST, path: str, allowed: set[str]) -> list[Issue]:
    """One-letter variables and parameters hide intent; only the names in `allowed` may stay."""
    issues = []
    for node in ast.walk(tree):
        name, line = None, getattr(node, "lineno", 1)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
        elif isinstance(node, ast.arg):
            name = node.arg
        if name and len(name) == 1 and name not in allowed and name not in SELF_PARAMETERS:
            issues.append(Issue(path, line, "short-name", f"`{name}` says nothing; use a descriptive name"))
    return issues
