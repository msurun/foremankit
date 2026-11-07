"""Test CLI registration and machine-readable autopilot outcomes.
Fake orchestration keeps these checks independent of runners and git.
"""
import json
from unittest.mock import Mock

import pytest

from foremankit import agent_autopilot, cli


@pytest.mark.parametrize("exit_code", [0, 2])
def test_cli_returns_the_orchestration_exit_code(exit_code, monkeypatch, tmp_path):
    """CLI preserves both successful and attention-needed outcomes."""
    fake = Mock(return_value={"exit_code": exit_code})
    monkeypatch.setattr(agent_autopilot, "autopilot", fake)
    monkeypatch.setattr(agent_autopilot, "render_text", Mock(return_value="report"))
    arguments = cli.build_parser().parse_args([
        "agent-autopilot", "Goal", "--work", str(tmp_path), "--plan-only"])
    assert cli.COMMANDS[arguments.cmd](None, arguments) == exit_code
    assert fake.call_args.args[0].plan_only is True


def test_json_output_is_a_structured_result(monkeypatch, tmp_path, capsys):
    """JSON mode prints a single parseable document."""
    expected = {"exit_code": 0, "status": "planned", "tasks": []}
    monkeypatch.setattr(agent_autopilot, "autopilot", Mock(return_value=expected))
    arguments = cli.build_parser().parse_args([
        "agent-autopilot", "@goal.md", "--work", str(tmp_path), "--json",
        "--max-fix-rounds", "0"])
    assert cli.cmd_agent_autopilot(None, arguments) == 0
    assert json.loads(capsys.readouterr().out) == expected
