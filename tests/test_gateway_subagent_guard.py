"""Tests for the subagent guard (foremankit.gateway): a Claude Code subagent loses its spawn tools and ends after a
successful SubagentHandback. Requests are built in memory; the llama window is pinned, nothing leaves the machine.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.responses import Response

from foremankit import gateway

TRIM_YAML = "trim:\n  enabled: true\n  tools_keep: [Agent]\n"


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal orchestrator config in a temp dir, so settings() reads it."""
    (tmp_path / "orchestrator.md").write_text("PROMPT", encoding="utf-8")
    (tmp_path / "orchestrator.yaml").write_text(TRIM_YAML, encoding="utf-8")
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


@pytest.fixture
def fixed_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the llama-server window so no server is queried."""
    monkeypatch.setattr(gateway, "server_window", lambda cfg: 10_000)


def subagent_request(results: list[dict[str, Any]]) -> dict[str, Any]:
    """A subagent request: it carries SubagentHandback and Agent, and the messages end with `results`."""
    messages = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "h1", "name": gateway.HANDBACK_TOOL, "input": {"report": "done"}}]},
        {"role": "user", "content": results},
    ]
    return {"model": "claude-sonnet-5", "system": "s", "messages": messages,
            "tools": [{"name": "Agent"}, {"name": "Read"}, {"name": gateway.HANDBACK_TOOL}]}


def ok_result(tool_use_id: str = "h1") -> dict[str, Any]:
    """A successful tool_result block."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": "ok"}


def run(data: dict[str, Any], config_dir: Path) -> tuple[gateway.Call, Any]:
    """prepare_body on a fresh local call; returns the call and the raw result (bytes or Response)."""
    cfg = gateway.settings()
    call = gateway.Call("/v1/messages", "claude-sonnet-5", True, time.time())
    return call, gateway.prepare_body(call, data, json.dumps(data).encode("utf-8"), cfg)


def test_a_subagent_request_loses_the_spawn_tools_but_keeps_the_others() -> None:
    """Agent is removed, Read and SubagentHandback stay."""
    data = {"tools": [{"name": "Agent"}, {"name": "Task"}, {"name": "Read"}, {"name": gateway.HANDBACK_TOOL}]}
    assert gateway.is_subagent(data)
    names = [tool["name"] for tool in gateway.without_spawn_tools(data, ["Agent", "Task"])["tools"]]
    assert names == ["Read", gateway.HANDBACK_TOOL]


def test_a_subagent_request_is_forwarded_without_agent(config_dir: Path, fixed_window: None) -> None:
    """The forwarded body of a subagent request has no Agent tool."""
    data = {"model": "claude-sonnet-5", "system": "s", "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "Agent"}, {"name": "Read"}, {"name": gateway.HANDBACK_TOOL}]}
    _, result = run(data, config_dir)
    assert isinstance(result, bytes)
    assert "Agent" not in [tool["name"] for tool in json.loads(result)["tools"]]


def test_a_main_loop_request_keeps_agent(config_dir: Path, fixed_window: None) -> None:
    """Without SubagentHandback the request is not a subagent: Agent is kept."""
    data = {"model": "claude-sonnet-5", "system": "s", "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "Agent"}]}
    assert not gateway.is_subagent(data)
    _, result = run(data, config_dir)
    assert isinstance(result, bytes)
    assert "Agent" in [tool["name"] for tool in json.loads(result)["tools"]]


def test_a_successful_handback_ends_the_run_non_streaming(config_dir: Path, fixed_window: None) -> None:
    """After a successful handback the answer is a JSON end_turn and the call notes the end."""
    call, result = run(subagent_request([ok_result()]), config_dir)
    assert isinstance(result, Response)
    body = json.loads(result.body)
    assert body["stop_reason"] == "end_turn"
    assert "run finished" in body["content"][0]["text"]
    assert "subagent handed back" in call.note


def test_a_successful_handback_ends_the_run_streaming(config_dir: Path, fixed_window: None) -> None:
    """With stream true the end_turn answer is an SSE stream."""
    data = {**subagent_request([ok_result()]), "stream": True}
    _, result = run(data, config_dir)
    assert isinstance(result, Response)
    assert result.media_type == "text/event-stream"
    body = result.body.decode("utf-8")
    assert "event: message_stop" in body and "end_turn" in body and "run finished" in body


def test_an_error_handback_does_not_end_the_run(config_dir: Path, fixed_window: None) -> None:
    """A tool_result marked is_error is not a successful handback: the subagent keeps going."""
    data = subagent_request([{**ok_result(), "is_error": True}])
    assert not gateway.handed_back(data)
    call, result = run(data, config_dir)
    assert isinstance(result, bytes)
    assert "subagent handed back" not in call.note


def haiku_subagent(model: str = "claude-haiku-5-5") -> dict[str, Any]:
    """A subagent request for a cloud model id, with the handback tool and a small prompt."""
    return {"model": model, "system": "s", "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "Read"}, {"name": gateway.HANDBACK_TOOL}]}


def cloud_when_local_off(config_dir: Path) -> None:
    """Write cloud_when_local: false into the temp orchestrator config."""
    (config_dir / "orchestrator.yaml").write_text(TRIM_YAML + "cloud_when_local: false\n", encoding="utf-8")


def context_stop_off(config_dir: Path) -> None:
    """Disable the context-stop policy so only the overflow branch decides."""
    with (config_dir / "orchestrator.yaml").open("a", encoding="utf-8") as fh:
        fh.write("context_stop:\n  enabled: false\n")


def set_driver(tmp_path: Path, local: bool) -> None:
    """Record who drives the main loop in the temp state dir."""
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "driver.json").write_text(
        json.dumps({"local": local, "model": "x", "time": time.time()}), encoding="utf-8")


def test_a_main_loop_request_records_the_driver(config_dir: Path, tmp_path: Path) -> None:
    """A main-loop request writes driver.json with the local flag of its call."""
    data = {"model": "claude-sonnet-5", "tools": [{"name": "Agent"}]}
    cfg = gateway.settings()
    gateway.record_main_loop(gateway.Call("/v1/messages", "claude-sonnet-5", True, time.time()), data, cfg)
    assert gateway.driver.read_driver()["local"] is True
    gateway.record_main_loop(gateway.Call("/v1/messages", "claude-opus-5", False, time.time()),
                             {**data, "model": "claude-opus-5"}, cfg)
    assert gateway.driver.local_drives() is False


def test_a_subagent_does_not_record_the_driver(config_dir: Path, tmp_path: Path) -> None:
    """A subagent request is not the main loop: driver.json stays absent."""
    gateway.record_main_loop(gateway.Call("/v1/messages", "claude-haiku-5-5", False, time.time()),
                             subagent_request([]), gateway.settings())
    assert gateway.driver.read_driver() is None


def test_a_haiku_subagent_is_served_locally_when_local_drives(config_dir: Path, tmp_path: Path) -> None:
    """With cloud_when_local false and the local model driving, a claude-haiku subagent is localised."""
    cloud_when_local_off(config_dir)
    set_driver(tmp_path, local=True)
    cfg = gateway.settings()
    call = gateway.Call("/v1/messages", "claude-haiku-5-5", False, time.time())
    gateway.localize_cloud_subagent(call, haiku_subagent(), cfg)
    assert call.local is True and "served locally" in call.note


def test_a_haiku_subagent_goes_to_cloud_when_opus_drives(config_dir: Path, tmp_path: Path) -> None:
    """A cloud driver keeps the subagent in the cloud."""
    cloud_when_local_off(config_dir)
    set_driver(tmp_path, local=False)
    call = gateway.Call("/v1/messages", "claude-haiku-5-5", False, time.time())
    gateway.localize_cloud_subagent(call, haiku_subagent(), gateway.settings())
    assert call.local is False


def test_a_haiku_subagent_goes_to_cloud_when_cloud_when_local_is_on(config_dir: Path, tmp_path: Path) -> None:
    """cloud_when_local true (the default) keeps the subagent in the cloud even while local drives."""
    set_driver(tmp_path, local=True)
    call = gateway.Call("/v1/messages", "claude-haiku-5-5", False, time.time())
    gateway.localize_cloud_subagent(call, haiku_subagent(), gateway.settings())
    assert call.local is False


def test_an_overflowing_subagent_is_stopped_when_local_drives(config_dir: Path, tmp_path: Path,
                                                              fixed_window: None) -> None:
    """Over the local window the subagent gets the stop answer, not a spill, while local drives."""
    cloud_when_local_off(config_dir)
    context_stop_off(config_dir)
    set_driver(tmp_path, local=True)
    data = subagent_request([])
    data["messages"][0]["content"] = "x" * 60_000
    call, result = run(data, config_dir)
    assert isinstance(result, Response)
    assert gateway.OVERFLOW_STOP_TEXT in json.loads(result.body)["content"][0]["text"]
    assert call.local is True


def test_an_overflowing_subagent_still_spills_when_not_local(config_dir: Path, tmp_path: Path,
                                                             fixed_window: None) -> None:
    """With a cloud driver the overflowing subagent is spilled to the cloud as before."""
    cloud_when_local_off(config_dir)
    context_stop_off(config_dir)
    set_driver(tmp_path, local=False)
    data = subagent_request([])
    data["messages"][0]["content"] = "x" * 60_000
    call, result = run(data, config_dir)
    assert isinstance(result, bytes)
    assert call.local is False
