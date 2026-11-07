"""Verify agent-plan CLI output and process status without live models.
Keep structured output stable for supervisors consuming draft plans.
"""
import json

import pytest

from foremankit import agent_planner, cli


@pytest.mark.parametrize("status,expected", [("ok", 0), ("invalid", 2), ("skipped", 2),
                                          ("unparsable", 2), ("failed", 2)])
def test_json_output_preserves_status(monkeypatch, capsys, status, expected):
    """Only a successfully published plan exits zero."""
    result = {"status": status, "tasks": []}
    monkeypatch.setattr(agent_planner, "plan", lambda job: result)
    args = cli.build_parser().parse_args(["agent-plan", "Add export", "--out", "draft", "--json"])
    assert cli.cmd_agent_plan(None, args) == expected
    assert json.loads(capsys.readouterr().out) == result


def test_text_output_includes_the_next_command(monkeypatch, capsys):
    """The lead can copy the command after reviewing the generated files."""
    result = {"status": "ok", "planner": "codex", "tasks": [
        {"id": "export", "title": "Add export", "depends_on": []}],
        "lead_items": ["Review policy"], "reasons": [], "out_dir": "draft"}
    monkeypatch.setattr(agent_planner, "plan", lambda job: result)
    args = cli.build_parser().parse_args(["agent-plan", "Add export", "--out", "draft"])
    assert cli.cmd_agent_plan(None, args) == 0
    output = capsys.readouterr().out
    assert "export <- -: Add export" in output
    assert "foremankit agent-run-batch --plan" in output
    assert "--project ." in output


def test_at_file_goal_is_forwarded(monkeypatch, tmp_path):
    """Goal resolution belongs to the planner so its failures remain structured."""
    goal_file = tmp_path / "goal.md"
    goal_file.write_text("Export data", encoding="utf-8")
    observed = []
    monkeypatch.setattr(agent_planner, "plan", lambda job: observed.append(job) or
                        {"status": "skipped", "tasks": []})
    args = cli.build_parser().parse_args(["agent-plan", f"@{goal_file}", "--out", "draft", "--json"])
    assert cli.cmd_agent_plan(None, args) == 2
    assert observed[0].goal == f"@{goal_file}"
