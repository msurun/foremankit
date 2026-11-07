"""Tests that README.md documents agent-run-batch and agent-merge honestly.

The README section must name every CLI flag, the depends_on key and the
REVIEW REQUIRED rule, so the docs cannot drift away from foremankit/cli.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
HEADING = "### agent-run-batch and agent-merge (several tasks, then review and merge)"
FLAGS = (
    "--plan",
    "--project",
    "--report-out",
    "--report",
    "--id",
    "--reviewed",
    "--reject",
    "--verdict",
    "--blockers",
    "--check",
)


def read_subsection() -> str:
    """Return the README text from the new heading up to the next `### ` heading."""
    text = README.read_text(encoding="utf-8")
    start = text.index(HEADING)
    rest = text[start + len(HEADING):]
    end = rest.find("\n### ")
    return rest if end == -1 else rest[:end]


def test_section_heading_exists() -> None:
    """The new subsection heading is present in README.md."""
    assert HEADING in README.read_text(encoding="utf-8")


@pytest.mark.parametrize("flag", FLAGS)
def test_flag_documented(flag: str) -> None:
    """Every flag of agent-run-batch and agent-merge appears in the subsection."""
    assert flag in read_subsection(), f"{flag} missing from the agent batch section"


def test_depends_on_key_documented() -> None:
    """The plan key `depends_on` is documented, not just used implicitly."""
    assert "depends_on" in read_subsection()


def test_review_required_rule_documented() -> None:
    """The REVIEW REQUIRED precondition of agent-merge is documented."""
    assert "REVIEW REQUIRED" in read_subsection()
