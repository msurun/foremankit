"""Tests for the gateway's no-code policy: a Claude orchestrator (Sonnet/Opus) loses its Write/Edit tools and gets an
extra system line that sends file changes to `delegate` or to a Haiku agent; Haiku subagents keep their tools.
Pure tests: no network, no llama-server, the window is pinned with a fake `server_window`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from foremankit import gateway

WRITE = ("Write", "Edit", "MultiEdit", "NotebookEdit")
DEFAULT_YAML = "trim:\n  enabled: false\n"
WINDOW_TOKENS = 32768            # the pinned context window of the fake llama-server
SPILL_FILL = 100_000             # user text in characters: ~76 % of a 32k window, above the spill line, below the stop


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a minimal orchestrator.yaml/.md to a temp config dir and point the gateway at it."""
    (tmp_path / "orchestrator.md").write_text("PROMPT", encoding="utf-8")
    (tmp_path / "orchestrator.yaml").write_text(DEFAULT_YAML, encoding="utf-8")
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


@pytest.fixture
def fixed_window(monkeypatch: pytest.MonkeyPatch) -> int:
    """Pin the llama-server window so the overflow threshold is deterministic."""
    monkeypatch.setattr(gateway, "server_window", lambda cfg: WINDOW_TOKENS)
    return WINDOW_TOKENS


def make_request(model: str, tools: list[str], system: Any = "s", fill: int = 10) -> dict[str, Any]:
    """A /v1/messages request with the named tools and a user message of `fill` characters."""
    return {
        "model": model,
        "system": system,
        "tools": [{"name": name, "description": name} for name in tools],
        "messages": [{"role": "user", "content": "x" * fill}],
    }


def run_prepare(local: bool, data: dict[str, Any]) -> tuple[gateway.Call, bytes, Any]:
    """Run prepare_body on one request; returns the call, the body sent in and the prepared result."""
    raw = json.dumps(data).encode("utf-8")
    call = gateway.Call("/v1/messages", data["model"], local, time.time())
    return call, raw, gateway.prepare_body(call, data, raw, gateway.settings())


def tool_names(body: dict[str, Any]) -> list[str]:
    """The tool names of a prepared request body."""
    return [tool["name"] for tool in body.get("tools") or []]


def test_sonnet_cloud_request_loses_write_tools(config_dir: Path) -> None:
    """A cloud Sonnet request keeps Read, Bash and Agent, loses every writing tool and gets the policy line."""
    data = make_request("claude-sonnet-5", ["Read", "Write", "Edit", "Bash", "Agent"])
    _, _, out = run_prepare(False, data)
    body = json.loads(out)
    assert not set(tool_names(body)) & set(WRITE)
    assert {"Read", "Bash", "Agent"} <= set(tool_names(body))
    assert "no-code policy" in body["system"]


def test_opus_cloud_request_loses_write_tools(config_dir: Path) -> None:
    """The same policy applies to an Opus orchestrator: no writing tools, Read and Agent kept."""
    _, _, out = run_prepare(False, make_request("claude-opus-5-5", ["Read", "Write", "Edit", "Bash", "Agent"]))
    body = json.loads(out)
    assert not set(tool_names(body)) & set(WRITE)
    assert "Read" in tool_names(body) and "Agent" in tool_names(body)
    assert "no-code policy" in body["system"]


def test_haiku_subagent_keeps_all_tools_and_body(config_dir: Path) -> None:
    """A Haiku subagent keeps its writing tools and its body is sent byte-identical."""
    _, raw, out = run_prepare(False, make_request("claude-haiku-5-5", ["Read", "Write", "Edit", "Bash", "Agent"]))
    assert "Write" in tool_names(json.loads(out))
    assert out == raw


def test_local_call_is_not_touched_by_policy(config_dir: Path) -> None:
    """A local request stays local with the Write tool and the local model name."""
    call, _, out = run_prepare(True, make_request("claude-sonnet-5", ["Agent", "Write", "Read"]))
    body = json.loads(out)
    assert call.local is True
    assert "Write" in tool_names(body)
    assert body["model"] == gateway.settings()["local_target"]


def test_no_code_false_keeps_write_tool(config_dir: Path) -> None:
    """With `no_code: false` in orchestrator.yaml a cloud Sonnet request is sent byte-identical."""
    (config_dir / "orchestrator.yaml").write_text("no_code: false\n" + DEFAULT_YAML, encoding="utf-8")
    _, raw, out = run_prepare(False, make_request("claude-sonnet-5", ["Read", "Write", "Bash", "Agent"]))
    assert out == raw


def test_spilled_request_loses_write_tools(config_dir: Path, fixed_window: int) -> None:
    """A local request spilled to the cloud keeps the cloud model id and also loses its writing tools."""
    call, _, out = run_prepare(True, make_request("claude-sonnet-5", ["Agent", "Write", "Edit", "Read"], fill=SPILL_FILL))
    body = json.loads(out)
    assert call.local is False
    assert body["model"] == "claude-sonnet-5"
    assert not set(tool_names(body)) & set(WRITE)


def test_list_system_gains_a_no_code_block(config_dir: Path) -> None:
    """A system given as a list of blocks gets one extra text block that carries the policy."""
    _, _, out = run_prepare(False, make_request("claude-sonnet-5", ["Write", "Read"], system=[{"type": "text", "text": "s"}]))
    system = json.loads(out)["system"]
    assert len(system) == 2
    assert "no-code policy" in system[-1]["text"]


def test_cloud_request_without_writing_tool_is_unchanged(config_dir: Path) -> None:
    """A cloud Sonnet request with no writing tool comes back byte-identical."""
    _, raw, out = run_prepare(False, make_request("claude-sonnet-5", ["Read", "Bash"]))
    assert out == raw
