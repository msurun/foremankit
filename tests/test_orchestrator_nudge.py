#!/usr/bin/env python3
"""Pytest tests for the batch nudge in config/orchestrator.md."""
from pathlib import Path

ORCHESTRATOR = Path(__file__).parent.parent / "config" / "orchestrator.md"
MAX_BYTES = 4400


def read_text() -> str:
    """The text of config/orchestrator.md with normalised newlines."""
    return ORCHESTRATOR.read_text(encoding="utf-8").replace("\r\n", "\n")


def bullet_lines(text: str) -> list[str]:
    """Every bullet line of the `# Execution modes` section, in file order."""
    lines = text.split("\n")
    start = lines.index("# Execution modes")
    out: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("# "):
            break
        if line.startswith("- "):
            out.append(line)
    return out


def test_the_file_names_both_batch_commands() -> None:
    """The orchestrator prompt mentions agent-run-batch and agent-merge."""
    text = read_text()
    assert "agent-run-batch" in text
    assert "agent-merge" in text


def test_the_batch_bullet_sits_after_agent_run_and_before_delegate() -> None:
    """The batch nudge is the bullet right after `agent-run`, before `delegate`."""
    bullets = bullet_lines(read_text())
    index = next(i for i, line in enumerate(bullets) if "agent-run-batch" in line)
    assert "`agent-run`" in bullets[index - 1]
    assert "`delegate`" in bullets[index + 1]


def test_the_batch_bullet_keeps_the_expected_flags() -> None:
    """The bullet carries the plan, project, report-out and merge flags."""
    bullet = next(line for line in bullet_lines(read_text()) if "agent-run-batch" in line)
    for flag in ("--plan FILE", "--project DIR", "--report-out R.json"):
        assert flag in bullet
    for flag in ("--report R.json", "--id ID", "[--reviewed]"):
        assert flag in bullet
    assert len(bullet) <= 300


def test_the_file_stays_small() -> None:
    """The prompt file is still at most 4400 bytes."""
    assert ORCHESTRATOR.stat().st_size <= MAX_BYTES
