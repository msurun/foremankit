"""Overlap detection for `agent-run --scope` pattern lists.
Two scopes overlap when they can touch the same file; the reasons explain why.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from foremankit.agent_scope import matches_any


def scopes_overlap(scope_a: Sequence[str], scope_b: Sequence[str], tracked: Sequence[str] = ()) -> list[str]:
    """Reasons why two scope lists can touch the same file (empty when they are disjoint)."""
    left = [_normalize(pattern) for pattern in scope_a]
    right = [_normalize(pattern) for pattern in scope_b]
    reasons: list[str] = []
    for pattern_a in left:
        for pattern_b in right:
            reasons.extend(_pair_reasons(pattern_a, pattern_b))
    for path in tracked:
        normal = _normalize(path)
        if matches_any(normal, left) and matches_any(normal, right):
            reasons.append(f"{normal} matches both scopes")
    return _unique(reasons)


def overlapping_pairs(
    scopes: Mapping[str, Sequence[str]], tracked: Sequence[str] = ()
) -> list[tuple[str, str, list[str]]]:
    """Every pair of task ids whose scopes overlap, with the reasons for that overlap."""
    ids = list(scopes)
    pairs: list[tuple[str, str, list[str]]] = []
    for index, id_a in enumerate(ids):
        for id_b in ids[index + 1:]:
            reasons = scopes_overlap(scopes[id_a], scopes[id_b], tracked)
            if reasons:
                pairs.append((id_a, id_b, reasons))
    return pairs


def _pair_reasons(pattern_a: str, pattern_b: str) -> list[str]:
    """The overlap reasons for one pair of normalized patterns."""
    wild_a, wild_b = _has_wildcard(pattern_a), _has_wildcard(pattern_b)
    if not wild_a and not wild_b:
        return [f"{pattern_a} is in both scopes"] if pattern_a == pattern_b else []
    if wild_a != wild_b:
        literal, pattern = (pattern_b, pattern_a) if wild_a else (pattern_a, pattern_b)
        return [f"{literal} matches {pattern}"] if matches_any(literal, [pattern]) else []
    prefix_a, prefix_b = _prefix(pattern_a), _prefix(pattern_b)
    if prefix_a.startswith(prefix_b) or prefix_b.startswith(prefix_a):
        return [f"{pattern_a} and {pattern_b} may overlap"]
    return []


def _normalize(pattern: str) -> str:
    """A pattern with backslashes and a leading ./ removed, and a trailing / turned into **."""
    pattern = pattern.replace("\\", "/").removeprefix("./")
    return pattern + "**" if pattern.endswith("/") else pattern


def _has_wildcard(pattern: str) -> bool:
    """Whether the pattern contains a `*` or `?` wildcard."""
    return "*" in pattern or "?" in pattern


def _prefix(pattern: str) -> str:
    """The literal text of a pattern before its first wildcard."""
    stops = [index for index in (pattern.find("*"), pattern.find("?")) if index >= 0]
    return pattern[: min(stops)] if stops else pattern


def _unique(items: Sequence[str]) -> list[str]:
    """`items` without duplicates, in their first-seen order."""
    unique: list[str] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique
