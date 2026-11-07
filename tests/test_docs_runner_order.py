"""Keep the documented three-runner order and prompt size consistent."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_batch_runner_order() -> None:
    """The batch section names privacy rules and the automatic runner order."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("### agent-run-batch and agent-merge", 1)[1]
    section = section.split("\n### ", 1)[0].lower()
    assert all(word in section for word in ("codex", "internal", "agy"))
    assert any(re.search(r"opencode.{0,140}codex.{0,140}agy", line)
               for line in section.splitlines())


def test_orchestrator_names_codex_and_stays_small() -> None:
    """The UTF-8 orchestrator prompt includes Codex within its byte budget."""
    path = ROOT / "config" / "orchestrator.md"
    assert "codex" in path.read_text(encoding="utf-8").lower()
    assert path.stat().st_size <= 4400


def test_codex_order_heading() -> None:
    """The Codex runner notes explain its position in the runner order."""
    text = (ROOT / "docs" / "codex-runner.md").read_text(encoding="utf-8")
    assert "## Where Codex sits in the order" in text
