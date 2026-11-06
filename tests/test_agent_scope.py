# tests/test_agent_scope.py
"""Tests for ForemanKit's agent_scope module.
Ensures glob pattern handling, scope splitting, and review reason extraction work as intended.
"""

import pytest
from foremankit.agent_scope import (
    glob_regex,
    matches_any,
    split_by_scope,
    review_reasons,
)


@pytest.mark.parametrize(
    ("pattern", "expected_regex"),
    [
        ("*.txt", r"[^/]*\.txt"),
        ("**/*.py", r"(?:.*/)?[^/]*\.py"),
        ("src/", r"src/.*"),
        ("?.md", r"[^/]\.md"),
        ("./*.md", r"[^/]*\.md"),
        ("src\\*.py", r"src/[^/]*\.py"),
        ("**/test", r"(?:.*/)?test"),
    ],
)
def test_glob_regex_compiles_correct_pattern(pattern: str, expected_regex: str) -> None:
    """Verify that glob_regex produces the expected regular expression for a glob."""
    compiled = glob_regex(pattern)
    assert compiled.pattern == expected_regex


@pytest.mark.parametrize(
    ("pattern", "path", "should_match"),
    [
        # * does not cross folder separators
        ("*.txt", "file.txt", True),
        ("*.txt", "sub/file.txt", False),
        # ** crosses folder separators
        ("**/*.py", "script.py", True),
        ("**/*.py", "module/util.py", True),
        ("**/*.py", "module/sub/util.py", True),
        ("**/*.py", "module/sub/util.txt", False),
        # **/x also matches root‑level x
        ("**/test", "test", True),
        ("**/test", "sub/test", True),
        ("**/test", "sub/sub2/test", True),
        ("**/test", "test.txt", False),
        # trailing dir/ means everything below it
        ("src/", "src/main.py", True),
        ("src/", "src/sub/file", True),
        ("src/", "src", False),
        ("src/", "src2/file", False),
        # ? matches a single non‑separator character
        ("?.py", "a.py", True),
        ("?.py", "ab.py", False),
        ("?.py", "a/b.py", False),
        # leading ./ is stripped
        ("./*.md", "readme.md", True),
        ("./*.md", "docs/readme.md", False),
        # backslashes are normalized to forward slashes
        ("src\\*.py", "src/util.py", True),
        ("src\\*.py", "src/sub/util.py", False),
        # multiple patterns – any match wins
        (["*.txt", "src/*"], "src/file.txt", True),
        (["*.txt", "src/*"], "docs/file.md", False),
    ],
)
def test_matches_any_various_patterns(pattern, path: str, should_match: bool) -> None:
    """Check that matches_any respects the glob semantics for a range of features."""
    # pattern may be a single string or a list of strings
    patterns = pattern if isinstance(pattern, (list, tuple)) else [pattern]
    assert matches_any(path, patterns) is should_match


def test_split_by_scope_categorizes_correctly() -> None:
    """split_by_scope should separate changes that match the scope from those that do not."""
    changes = [
        ("A", "src/main.py"),
        ("M", "test/unit/test.py"),
        ("D", "README.md"),
        ("R", "docs/guide.txt"),
    ]
    scope = ("src/**", "**/test.py")
    inside, outside = split_by_scope(changes, scope)

    expected_inside = [
        ("A", "src/main.py"),
        ("M", "test/unit/test.py"),
    ]
    expected_outside = [
        ("D", "README.md"),
        ("R", "docs/guide.txt"),
    ]
    assert inside == expected_inside
    assert outside == expected_outside


def test_split_by_scope_preserves_order() -> None:
    """The relative order of changes must be kept in both result lists."""
    changes = [
        ("A", "a.txt"),
        ("B", "src/b.txt"),
        ("C", "a/b/c.txt"),
        ("D", "src/d/e.txt"),
    ]
    scope = ("src/**",)
    inside, outside = split_by_scope(changes, scope)

    assert inside == [
        ("B", "src/b.txt"),
        ("D", "src/d/e.txt"),
    ]
    assert outside == [
        ("A", "a.txt"),
        ("C", "a/b/c.txt"),
    ]


def test_split_by_scope_empty_scope() -> None:
    """An empty scope should result in no inside changes."""
    changes = [("M", "any/file.txt")]
    scope = ()
    inside, outside = split_by_scope(changes, scope)
    assert inside == []
    assert outside == changes


def test_review_reasons_filters_paths() -> None:
    """review_reasons should return only the paths matching any review glob."""
    paths = [
        "gateway/config.yaml",
        "gateway/sub/secret.txt",
        "auth/login.py",
        "policy/terms.txt",
        "src/main.py",
        "docs/readme.md",
    ]
    review_globs = ["gateway/**", "auth/**", "policy/**"]
    result = review_reasons(paths, review_globs)

    expected = [
        "gateway/config.yaml",
        "gateway/sub/secret.txt",
        "auth/login.py",
        "policy/terms.txt",
    ]
    assert result == expected


def test_review_reasons_empty_globs_returns_empty() -> None:
    """An empty list of review globs yields no mandatory review paths."""
    paths = ["a.txt", "b.py"]
    assert review_reasons(paths, []) == []
