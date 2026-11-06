"""Path globs for `agent-run`: which changed files belong to the task scope and which paths need a mandatory review.
`*` stays inside one folder, `**` crosses folders, `dir/` means everything below `dir`.
"""
from __future__ import annotations

import re

Change = tuple[str, str]   # (git status letter, posix path relative to the repository)


def glob_regex(pattern: str) -> re.Pattern[str]:
    """The compiled regex of one glob pattern (matched against a whole posix path)."""
    pattern = pattern.replace("\\", "/").removeprefix("./")
    if pattern.endswith("/"):
        pattern += "**"
    parts, index = [], 0
    while index < len(pattern):
        text, index = _translate(pattern, index)
        parts.append(text)
    return re.compile("".join(parts))


def _translate(pattern: str, index: int) -> tuple[str, int]:
    """The regex text for the glob token at `index` and the index after that token."""
    if pattern.startswith("**/", index):
        return "(?:.*/)?", index + 3
    if pattern.startswith("**", index):
        return ".*", index + 2
    char = pattern[index]
    if char == "*":
        return "[^/]*", index + 1
    if char == "?":
        return "[^/]", index + 1
    return re.escape(char), index + 1


def matches_any(path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    """Whether `path` matches at least one glob pattern."""
    return any(glob_regex(pattern).fullmatch(path) for pattern in patterns)


def split_by_scope(changes: list[Change], scope: tuple[str, ...]) -> tuple[list[Change], list[Change]]:
    """(changes inside the scope, changes outside it)."""
    inside = [change for change in changes if matches_any(change[1], scope)]
    outside = [change for change in changes if change not in inside]
    return inside, outside


def review_reasons(paths: list[str], review_globs: list[str]) -> list[str]:
    """The changed paths that match a security glob (empty list = no mandatory review)."""
    return [path for path in paths if matches_any(path, review_globs)]
