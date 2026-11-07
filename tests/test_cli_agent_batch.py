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



def test_missing_resume_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing resume reports produce a clear refusal."""
    code = main(["agent-run-batch", "--plan", "plan.yaml", "--project", str(tmp_path),
                 "--resume", str(tmp_path / "missing.json")])
    assert code == 2
    assert "FAIL agent-run-batch refused: cannot resume" in capsys.readouterr().out


def test_waiting_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Waiting output includes local retry time and one resume hint."""
    import time

    from foremankit.cli import _print_batch
    _print_batch({"tasks": [{"id": "a", "runner": "codex", "status": "waiting", "retry_after": 2000}],
                  "summary": {"waiting": 1}}, "report.json")
    output = capsys.readouterr().out
    assert "waiting a codex until " + time.strftime("%H:%M", time.localtime(2000)) in output
    assert output.count("waiting tasks: rerun with --resume report.json") == 1



def test_resume_runs_only_unfinished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid CLI resume passes saved entries into the real batch engine."""
    from foremankit import agent_plan, cli
    from foremankit.agent_batch import BatchContext
    calls = []
    report = tmp_path / "report.json"
    old = {"id": "a", "status": "passed", "branch": "saved"}
    report.write_text(json.dumps({"tasks": [old]}))
    (tmp_path / "b.md").write_text("task")
    tasks = [agent_plan.PlanTask(id="a", task_file="missing", scope=("a.py",), check="pytest"),
             agent_plan.PlanTask(id="b", task_file="b.md", scope=("b.py",), check="pytest", depends_on=("a",))]
    monkeypatch.setattr(agent_plan, "load_plan", lambda path: tasks)
    monkeypatch.setattr(cli, "_batch_context", lambda plan, project, done: BatchContext(
        project=tmp_path, plan_dir=tmp_path, run_one=lambda job: calls.append(job) or {"passed": True},
        settings={}, state_path=tmp_path / "quota.json", done=done))
    assert main(["agent-run-batch", "--plan", "p", "--project", str(tmp_path), "--resume", str(report),
                 "--report-out", str(report)]) == 0
    assert len(calls) == 1 and calls[0].base == "saved"
    assert json.loads(report.read_text())["tasks"][0] == old
