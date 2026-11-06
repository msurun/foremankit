"""Tests for the `agent-run-batch` and `agent-merge` CLI subcommands: parsing and refusal paths only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from foremankit.cli import build_parser, main


def test_agent_run_batch_refuses_missing_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["agent-run-batch", "--plan", str(tmp_path / "missing.yaml"), "--project", str(tmp_path)])
    assert code == 2
    assert "FAIL agent-run-batch refused" in capsys.readouterr().out


def test_agent_merge_refuses_unknown_task_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"tasks": [{"id": "a", "status": "passed"}]}), encoding="utf-8")
    code = main(["agent-merge", "--project", str(tmp_path), "--report", str(report), "--id", "zzz"])
    assert code == 2
    assert "no task 'zzz'" in capsys.readouterr().out


def test_agent_merge_parser_defaults_to_merged_verdict() -> None:
    args = build_parser().parse_args(["agent-merge", "--project", "p", "--report", "r.json", "--id", "t1"])
    assert args.verdict == "merged"
    assert args.reject is False
    assert args.blockers == 0


def test_agent_run_batch_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["agent-run-batch", "--help"])
    assert stop.value.code == 0
