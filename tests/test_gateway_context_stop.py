"""Integration tests for the gateway's hard context stop (foremankit.gateway + foremankit.context_stop).

A request whose prompt already fills `ratio` of the model window is answered on the spot with a handoff
message instead of being forwarded; nothing leaves the machine, the window is pinned with a fake
`server_window`, and the handoff files land in a temporary state dir.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.responses import Response

from foremankit import gateway

WINDOW_TOKENS = 10_000      # the pinned context window of the fake llama-server
STOP_FILL = 45_000          # user text characters: above the 0.90 ratio (~36_000 chars)
SMALL_FILL = 1_000
NEAR_FILL = 37_000          # between the 0.90 and 0.95 thresholds (~36_000 and ~38_000 chars)
# `tools_keep` must keep the Agent tool: the shaping drops every tool it does not name, and the stop only gates
# requests that still carry tools (a tool-less request would be a side request).
TRIM_YAML = "trim:\n  enabled: true\n  tools_keep: [Agent]\n"


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a minimal orchestrator.yaml/.md to a temp config dir and point the gateway at it."""
    (tmp_path / "orchestrator.md").write_text("PROMPT", encoding="utf-8")
    (tmp_path / "orchestrator.yaml").write_text(TRIM_YAML, encoding="utf-8")
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


@pytest.fixture
def fixed_window(monkeypatch: pytest.MonkeyPatch) -> int:
    """Pin the llama-server window so the stop threshold is deterministic."""
    monkeypatch.setattr(gateway, "server_window", lambda cfg: WINDOW_TOKENS)
    return WINDOW_TOKENS


def make_request(fill_chars: int) -> dict[str, Any]:
    """A main-loop /v1/messages request for the local model, with a user message of `fill_chars` characters."""
    return {
        "model": "claude-sonnet-5",
        "system": "s",
        "tools": [{"name": "Agent", "description": "spawn"}],
        "messages": [{"role": "user", "content": "x" * fill_chars}],
    }


def handoff_files(config_dir: Path) -> list[Path]:
    """Handoff markdown files written under the temporary state dir."""
    return sorted((config_dir / "state" / "handoffs").glob("handoff-*.md"))


def prepare(fill_chars: int, config_dir: Path) -> tuple[gateway.Call, Any]:
    """Run prepare_body on a fresh local call; returns the call and its raw result (bytes or Response)."""
    cfg = gateway.settings()
    call = gateway.Call("/v1/messages", "claude-sonnet-5", True, time.time())
    data = make_request(fill_chars)
    return call, gateway.prepare_body(call, data, json.dumps(data).encode("utf-8"), cfg)


def reply_text_of(response: Response) -> str:
    """The text block of a JSON Anthropic answer, as the gateway's stop answer builds it."""
    return json.loads(response.body)["content"][0]["text"]


def test_an_over_ratio_local_request_is_stopped_and_not_spilled(config_dir: Path, fixed_window: int) -> None:
    """The stop answers locally: a Response, the stop note, and the call still marked local (no cloud spill)."""
    call, result = prepare(STOP_FILL, config_dir)
    assert isinstance(result, Response)
    text = reply_text_of(result)
    assert "context stop" in text and "handoff: " in text and "/clear" in text
    assert json.loads(result.body)["stop_reason"] == "end_turn"
    assert "tool_use" not in result.body.decode("utf-8")
    assert "context stop" in call.note
    assert call.local is True


def test_a_stopped_request_writes_exactly_one_handoff_file(config_dir: Path, fixed_window: int) -> None:
    """The written handoff has the expected sections and is the file the reply names."""
    _, result = prepare(STOP_FILL, config_dir)
    files = handoff_files(config_dir)
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "# Handoff - context stop" in text
    assert "## What was asked" in text and "## Next step" in text
    assert str(files[0].resolve()) in reply_text_of(result)


def test_a_streaming_over_ratio_request_is_stopped_with_an_sse_answer(config_dir: Path, fixed_window: int) -> None:
    """`stream: true` gets an Anthropic SSE stream that carries the stop reply, not a tool call."""
    cfg = gateway.settings()
    call = gateway.Call("/v1/messages", "claude-sonnet-5", True, time.time())
    data = {**make_request(STOP_FILL), "stream": True}
    result = gateway.prepare_body(call, data, json.dumps(data).encode("utf-8"), cfg)
    assert isinstance(result, Response)
    assert result.media_type == "text/event-stream"
    body = result.body.decode("utf-8")
    for event in ("event: message_start", "event: message_delta", "event: message_stop"):
        assert event in body
    assert "tool_use" not in body
    assert "context stop" in body and "handoff: " in body


def test_a_request_below_the_ratio_is_forwarded_not_stopped(config_dir: Path, fixed_window: int) -> None:
    """A small main-loop request keeps its bytes body and writes no handoff."""
    call, result = prepare(SMALL_FILL, config_dir)
    assert isinstance(result, bytes) and not isinstance(result, Response)
    assert call.local is True
    assert handoff_files(config_dir) == []


def test_a_toolless_side_request_is_never_stopped(config_dir: Path, fixed_window: int) -> None:
    """A tool-less request goes to the cloud as a side request, however big, and writes no handoff."""
    cfg = gateway.settings()
    call = gateway.Call("/v1/messages", "claude-sonnet-5", True, time.time())
    data = make_request(STOP_FILL)
    data.pop("tools")
    result = gateway.prepare_body(call, data, json.dumps(data).encode("utf-8"), cfg)
    assert isinstance(result, bytes) and not isinstance(result, Response)
    assert "side request" in call.note
    assert "context stop" not in call.note
    assert handoff_files(config_dir) == []


def test_the_stop_is_off_when_context_stop_is_disabled(config_dir: Path, fixed_window: int) -> None:
    """`context_stop: enabled: false` restores the old behaviour: the big request goes to the local model, no handoff."""
    (config_dir / "orchestrator.yaml").write_text(
        TRIM_YAML + "overflow_to_cloud: false\ncontext_stop:\n  enabled: false\n", encoding="utf-8")
    call, result = prepare(STOP_FILL, config_dir)
    assert isinstance(result, bytes) and not isinstance(result, Response)
    assert call.local is True
    assert handoff_files(config_dir) == []


def test_a_raised_ratio_lets_a_near_fill_request_through(config_dir: Path, fixed_window: int) -> None:
    """At ratio 0.95 a 0.90-0.95 fill is below the stop, so it is forwarded as bytes (settings re-reads the file)."""
    (config_dir / "orchestrator.yaml").write_text(
        TRIM_YAML + "overflow_to_cloud: false\ncontext_stop:\n  ratio: 0.95\n", encoding="utf-8")
    call, result = prepare(NEAR_FILL, config_dir)
    assert isinstance(result, bytes) and not isinstance(result, Response)
    assert call.local is True
    assert handoff_files(config_dir) == []


def test_the_same_near_fill_is_stopped_at_the_default_ratio(config_dir: Path, fixed_window: int) -> None:
    """The default 0.90 ratio does stop that same fill, so the ratio really decides."""
    (config_dir / "orchestrator.yaml").write_text(
        TRIM_YAML + "overflow_to_cloud: false\n", encoding="utf-8")
    call, result = prepare(NEAR_FILL, config_dir)
    assert isinstance(result, Response)
    assert "context stop" in call.note
    assert len(handoff_files(config_dir)) == 1


def test_an_external_orchestrator_request_is_stopped_with_the_handoff(config_dir: Path, fixed_window: int) -> None:
    """The external path answers the stop too, and the handoff names the mode taken from call.route."""
    cfg = gateway.settings()
    up = {"name": "deepseek", "url": "https://api.deepseek.com/anthropic", "context": WINDOW_TOKENS,
          "tools_keep": ["Agent", "Read"], "schema": {}, "models": {"sonnet": "flash"}}
    call = gateway.Call("/v1/messages", "claude-sonnet-5", False, time.time(), route="external:deepseek")
    result = gateway.prepare_external(call, make_request(STOP_FILL), up, cfg)
    assert isinstance(result, Response)
    assert "context stop" in reply_text_of(result)
    files = handoff_files(config_dir)
    assert len(files) == 1 and " via deepseek" in files[0].read_text(encoding="utf-8")


def test_context_stop_response_reports_the_external_mode_in_the_handoff(config_dir: Path) -> None:
    """The stop decision alone (no request shaping) uses `where` for the handoff model line."""
    cfg = gateway.settings()
    call = gateway.Call("/v1/messages", "claude-sonnet-5", False, time.time(), route="external:deepseek")
    where = {"mode": "deepseek", "model": "flash", "window": WINDOW_TOKENS}
    result = gateway.context_stop_response(call, make_request(STOP_FILL), where, cfg)
    assert result is not None
    files = handoff_files(config_dir)
    assert len(files) == 1 and " via deepseek" in files[0].read_text(encoding="utf-8")
