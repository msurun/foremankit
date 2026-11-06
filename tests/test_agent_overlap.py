# tests/test_agent_overlap.py
"""Tests for ForemanKit's agent_overlap module.
Ensures scope overlap detection covers literals, wildcards, directories and tracked files.
"""

import pytest

from foremankit.agent_overlap import overlapping_pairs, scopes_overlap


@pytest.mark.parametrize(
    ("scope_a", "scope_b", "expected"),
    [
        # identical patterns overlap
        (["src/**"], ["src/**"], ["src/** and src/** may overlap"]),
        # literal file inside a directory glob
        (["src/main.py"], ["src/**"], ["src/main.py matches src/**"]),
        # trailing-slash directory pattern counts as <dir>/**
        (["src/"], ["src/main.py"], ["src/main.py matches src/**"]),
        (["src/"], ["src/*.py"], ["src/** and src/*.py may overlap"]),
        # different literal files do not overlap, even with a shared stem
        (["frontend/src/a.ts"], ["frontend/src/a.tsx"], []),
        # wildcard below a folder overlaps a broad folder glob
        (
            ["gateway/tests/test_adv_*.py"],
            ["gateway/**"],
            ["gateway/tests/test_adv_*.py and gateway/** may overlap"],
        ),
        # unrelated folders do not overlap
        (["a/*.py"], ["b/*.py"], []),
        # **/auth* can reach a literal gateway path
        (["**/auth*"], ["gateway/auth.py"], ["gateway/auth.py matches **/auth*"]),
    ],
)
def test_scopes_overlap_cases(scope_a, scope_b, expected) -> None:
    """Check the overlap rules for literals, wildcards and directory patterns."""
    assert scopes_overlap(scope_a, scope_b) == expected


def test_scopes_overlap_equal_after_normalization() -> None:
    """Backslashes and a leading ./ are ignored when comparing literal patterns."""
    reasons = scopes_overlap([".\\src\\main.py"], ["src/main.py"])
    assert reasons == ["src/main.py is in both scopes"]


def test_scopes_overlap_empty_scope_lists() -> None:
    """An empty scope list cannot touch any file."""
    assert scopes_overlap([], ["src/**"]) == []
    assert scopes_overlap(["src/**"], []) == []
    assert scopes_overlap([], []) == []


def test_scopes_overlap_tracked_file_matched_by_both() -> None:
    """A tracked path matching a pattern of each scope is reported once."""
    reasons = scopes_overlap(["**/*.py"], ["**/shared.py"], ["shared.py"])
    assert reasons == [
        "**/*.py and **/shared.py may overlap",
        "shared.py matches both scopes",
    ]


def test_scopes_overlap_tracked_file_must_match_both_scopes() -> None:
    """A tracked path matched by only one scope creates no overlap reason."""
    assert scopes_overlap(["docs/**"], ["src/*.py"], ["src/util.py"]) == []


def test_scopes_overlap_reasons_have_no_duplicates() -> None:
    """Repeated patterns and repeated tracked paths yield each reason once."""
    reasons = scopes_overlap(
        ["src/**", "src/**"],
        ["src/**", "src/**"],
        ["src/main.py", "src/main.py"],
    )
    assert reasons == [
        "src/** and src/** may overlap",
        "src/main.py matches both scopes",
    ]


def test_overlapping_pairs_reports_each_pair_once() -> None:
    """Only overlapping task pairs are returned, in dict order and once per pair."""
    scopes = {
        "frontend": ["frontend/**"],
        "frontend_tests": ["frontend/src/a.tsx"],
        "backend": ["backend/**"],
    }
    pairs = overlapping_pairs(scopes)
    assert pairs == [
        (
            "frontend",
            "frontend_tests",
            ["frontend/src/a.tsx matches frontend/**"],
        )
    ]


def test_overlapping_pairs_no_overlap_returns_empty() -> None:
    """Disjoint task scopes produce no pairs."""
    scopes = {"api": ["api/**"], "web": ["web/**"]}
    assert overlapping_pairs(scopes) == []


def test_overlapping_pairs_passes_tracked_paths() -> None:
    """Tracked paths are forwarded when comparing pairs of task scopes."""
    scopes = {"api": ["**/*.py"], "shared": ["**/shared.py"]}
    pairs = overlapping_pairs(scopes, ["shared.py"])
    assert pairs == [
        (
            "api",
            "shared",
            ["**/*.py and **/shared.py may overlap", "shared.py matches both scopes"],
        )
    ]
