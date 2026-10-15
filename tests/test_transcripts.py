"""Tests for the transcript summaries behind `foremankit sessions`: token counting, delegation detection, file discovery and result formatting."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from foremankit.formatting import format_result
from foremankit.transcripts import find_transcripts, summarize_many, summarize_transcript

THIRTY_DAYS = 30 * 86400


def usage(input_tokens: int = 0, cache_creation: int = 0, cache_read: int = 0, output_tokens: int = 0) -> dict[str, int]:
    """Return a usage record shaped like the one Claude Code writes into each assistant message."""
    return {"input_tokens": input_tokens, "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read, "output_tokens": output_tokens}


def tool_use(name: str, tool_input: Any) -> dict[str, Any]:
    """Return a tool_use content block with the given tool name and input."""
    return {"type": "tool_use", "name": name, "input": tool_input}


def assistant_line(request_id: str, token_usage: dict[str, int] | None = None,
                   content: list[dict[str, Any]] | None = None) -> str:
    """Return one JSONL assistant line; the usage is attached to the message only when given."""
    message: dict[str, Any] = {"model": "claude-test-model", "content": content or []}
    if token_usage is not None:
        message["usage"] = token_usage
    entry = {"type": "assistant", "requestId": request_id, "timestamp": "2026-10-14T10:00:00Z", "message": message}
    return json.dumps(entry)


def write_transcript(path: Path, lines: list[str]) -> Path:
    """Write the given JSONL lines to a file, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize_lines(tmp_path: Path, lines: list[str]):
    """Write the lines as one transcript file and return its summary."""
    return summarize_transcript(write_transcript(tmp_path / "session.jsonl", lines))


def make_result(verify: dict[str, Any] | None) -> dict[str, Any]:
    """Return a minimal delegation result dict as the router produces it."""
    return {"provider": "groq", "model": "llama-test", "latency_s": 1.2, "tokens_in": 10, "cache_hit": 0,
            "tokens_out": 5, "cost_usd": 0.0, "path": ".agent-out/task.md", "preview": "print('hi')",
            "verify": verify}


def failed_verify() -> dict[str, Any]:
    """Return a verify record for a check that still failed after two fix rounds."""
    return {"passed": False, "fix_rounds": 2, "command": "python -m pytest -q", "target": "app.py",
            "output_tail": "E501 line too long"}


def passed_verify() -> dict[str, Any]:
    """Return a verify record for a check that passed on the first try."""
    return {"passed": True, "fix_rounds": 0, "command": "python -m pytest -q", "target": "app.py",
            "output_tail": ""}


# ---------------------------------------------------------------- token counting

def test_a_request_spread_over_two_lines_is_counted_once(tmp_path):
    lines = [assistant_line("req-1", usage(input_tokens=100, output_tokens=10), [tool_use("Read", {})]),
             assistant_line("req-1", usage(input_tokens=100, output_tokens=10), [tool_use("Grep", {})])]
    summary = summarize_lines(tmp_path, lines)
    assert summary.requests == 1
    assert summary.input_tokens == 100 and summary.output_tokens == 10


def test_a_line_with_broken_json_is_skipped(tmp_path):
    lines = [assistant_line("req-1", usage(input_tokens=100)),
             '{"type": "assistant", "requestId": "req-2", "message": {"usage": {"input_tok',
             assistant_line("req-3", usage(input_tokens=50))]
    summary = summarize_lines(tmp_path, lines)
    assert summary.requests == 2
    assert summary.input_tokens == 150


def test_total_tokens_leaves_out_cache_reads(tmp_path):
    lines = [assistant_line("req-1", usage(input_tokens=10, cache_creation=20, cache_read=1000, output_tokens=5))]
    summary = summarize_lines(tmp_path, lines)
    assert summary.total_tokens() == 35
    assert summary.cache_read_tokens == 1000


# ---------------------------------------------------------------- delegations, agent calls and reads

def test_a_delegate_mcp_tool_counts_as_a_delegation(tmp_path):
    lines = [assistant_line("req-1", usage(input_tokens=1), [tool_use("mcp__foremankit__delegate", {"task": "x"})])]
    assert summarize_lines(tmp_path, lines).delegations == 1


def test_a_bash_command_running_foremankit_delegate_counts_as_a_delegation(tmp_path):
    delegate_command = tool_use("Bash", {"command": 'foremankit delegate "write tests" --kind bulk_code'})
    git_command = tool_use("Bash", {"command": "git status"})
    lines = [assistant_line("req-1", usage(input_tokens=1), [delegate_command]),
             assistant_line("req-2", usage(input_tokens=1), [git_command])]
    assert summarize_lines(tmp_path, lines).delegations == 1


def test_an_agent_tool_use_counts_as_an_agent_call(tmp_path):
    lines = [assistant_line("req-1", usage(input_tokens=1), [tool_use("Agent", {"prompt": "find files"})])]
    summary = summarize_lines(tmp_path, lines)
    assert summary.agent_calls == 1
    assert summary.delegations == 0


def test_a_read_of_an_agent_out_file_counts_as_an_agent_out_read(tmp_path):
    read_block = tool_use("Read", {"file_path": "C:/work/.agent-out/abc123.md"})
    lines = [assistant_line("req-1", usage(input_tokens=1), [read_block])]
    assert summarize_lines(tmp_path, lines).agent_out_reads == 1


def test_a_read_of_another_file_does_not_count_as_an_agent_out_read(tmp_path):
    read_block = tool_use("Read", {"file_path": "C:/work/foremankit/cli.py"})
    lines = [assistant_line("req-1", usage(input_tokens=1), [read_block])]
    assert summarize_lines(tmp_path, lines).agent_out_reads == 0


# ---------------------------------------------------------------- finding transcript files

def test_a_missing_root_gives_no_transcripts(tmp_path):
    assert find_transcripts(7, root=tmp_path / "absent") == []


def test_subagent_transcripts_are_ignored(tmp_path):
    main = write_transcript(tmp_path / "projects" / "app" / "main.jsonl", [assistant_line("req-1")])
    write_transcript(tmp_path / "projects" / "app" / "subagents" / "agent.jsonl", [assistant_line("req-2")])
    assert find_transcripts(None, root=tmp_path / "projects") == [main]


def test_transcripts_older_than_the_window_are_ignored(tmp_path):
    root = tmp_path / "projects"
    fresh = write_transcript(root / "fresh.jsonl", [assistant_line("req-1")])
    old = write_transcript(root / "old.jsonl", [assistant_line("req-2")])
    long_ago = time.time() - THIRTY_DAYS
    os.utime(old, (long_ago, long_ago))
    assert find_transcripts(7, root=root) == [fresh]
    assert set(find_transcripts(None, root=root)) == {fresh, old}


def test_an_unreadable_path_is_skipped_when_summarising_many(tmp_path):
    good = write_transcript(tmp_path / "good.jsonl", [assistant_line("req-1", usage(input_tokens=5))])
    missing = tmp_path / "gone.jsonl"
    summaries = summarize_many([missing, good])
    assert [summary.session_id for summary in summaries] == ["good"]


# ---------------------------------------------------------------- result formatting

@pytest.mark.parametrize("verify", [None, failed_verify()], ids=["no-verify", "verify-failed"])
def test_a_result_without_a_passed_verify_still_shows_the_preview(verify):
    text = format_result(make_result(verify))
    assert "preview:" in text
    assert "print('hi')" in text


def test_a_passed_verify_line_reports_the_pass():
    text = format_result(make_result(passed_verify()))
    assert "verify: PASSED on the first try" in text


def test_a_passed_verify_tells_the_reader_the_agent_out_file_need_not_be_read():
    text = format_result(make_result(passed_verify()))
    assert "no need to read the .agent-out file" in text
    assert "preview:" not in text
