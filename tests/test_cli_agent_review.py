"""Check agent-review command output and process status without live models.
The CLI must keep machine-readable JSON intact for every review outcome.
"""
import json

import pytest

from foremankit import agent_review, cli


@pytest.mark.parametrize("status,expected", [("ok", 0), ("skipped", 2),
                                          ("unparsable", 2), ("failed", 2)])
def test_json_output_preserves_status_and_exit_code(monkeypatch, capsys, status, expected):
    """Only successful parsing and execution have exit code zero."""
    result = {"status": status, "findings": []}
    monkeypatch.setattr(agent_review, "review", lambda job: result)
    args = cli.build_parser().parse_args(["agent-review", "report.json#test", "--json"])
    assert cli.cmd_agent_review(None, args) == expected
    assert json.loads(capsys.readouterr().out) == result


def test_text_output_shows_counts_before_findings(monkeypatch, capsys):
    """The short summary puts the severity counts first."""
    result = {"status": "ok", "findings": [],
              "counts_by_severity": {"high": 0, "medium": 0, "low": 0}}
    monkeypatch.setattr(agent_review, "review", lambda job: result)
    args = cli.build_parser().parse_args(["agent-review", "agent-run/test"])
    assert cli.cmd_agent_review(None, args) == 0
    assert capsys.readouterr().out.startswith("0 high, 0 medium, 0 low")
