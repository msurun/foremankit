"""Tests for the `agent-ask` and `agent-ask-batch` CLI subcommands: parsing, refusal paths and printed output only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foremankit.cli import build_parser, main


def _ok_result(**changes: Any) -> dict[str, Any]:
    """A fixed ask result (ok unless changed) in the shape agent_ask.ask returns."""
    result: dict[str, Any] = {"ok": True, "name": "q1", "runner": "agy", "model": "", "answer": "42",
                              "tokens_in": 100, "tokens_out": 7, "tool_calls": 2, "thinking_tokens": None,
                              "cache_read_tokens": None, "turns": 1, "seconds": 3.2, "modified_files": [],
                              "raw_path": "", "error": ""}
    result.update(changes)
    return result


def _ask_argv(tmp_path: Path, task: Path) -> list[str]:
    """The agent-ask command line for one task file in tmp_path."""
    return ["agent-ask", "--task-file", str(task), "--project", str(tmp_path), "--runner", "agy"]


def test_agent_ask_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["agent-ask", "--help"])
    assert stop.value.code == 0


def test_agent_ask_batch_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["agent-ask-batch", "--help"])
    assert stop.value.code == 0


def test_agent_ask_refuses_unreadable_task_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_ask_argv(tmp_path, tmp_path / "missing.md"))
    assert code == 2
    assert "FAIL agent-ask refused" in capsys.readouterr().out


def test_agent_ask_batch_refuses_empty_tasks_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    code = main(["agent-ask-batch", "--tasks-dir", str(tasks), "--out-dir", str(tmp_path / "out"),
                 "--runner", "agy", "--project", str(tmp_path)])
    assert code == 2
    assert "FAIL agent-ask-batch refused: no *.md or *.txt files" in capsys.readouterr().out


def test_agent_ask_prints_line_and_exits_zero_when_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    task = tmp_path / "q1.md"
    task.write_text("Where is the pump?", encoding="utf-8")
    monkeypatch.setattr("foremankit.agent_ask.ask", lambda job, raw_dir=None: _ok_result())
    code = main(_ask_argv(tmp_path, task))
    line = capsys.readouterr().out
    assert code == 0
    assert line.startswith("OK q1 agy")
    assert "tokens_in=100" in line and "tokens_out=7" in line


def test_agent_ask_json_output_parses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    task = tmp_path / "q1.txt"
    task.write_text("Where is the pump?", encoding="utf-8")
    monkeypatch.setattr("foremankit.agent_ask.ask", lambda job, raw_dir=None: _ok_result())
    main([*_ask_argv(tmp_path, task), "--json"])
    assert json.loads(capsys.readouterr().out)["answer"] == "42"


def test_agent_ask_exits_one_when_not_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    task = tmp_path / "q1.md"
    task.write_text("Where is the pump?", encoding="utf-8")
    failed = _ok_result(ok=False, error="timed out")
    monkeypatch.setattr("foremankit.agent_ask.ask", lambda job, raw_dir=None: failed)
    code = main(_ask_argv(tmp_path, task))
    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("FAIL q1") and "error: timed out" in out
