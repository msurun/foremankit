"""Behaviour tests for foremankit/context_stop.py, the hard context stop of the gateway.

Every test is pure: request bodies are built in memory and handoffs are written
to pytest's tmp_path, so no server or network is involved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from foremankit.context_stop import (
    DEFAULTS,
    estimate_tokens,
    folder,
    handoff_markdown,
    message_json,
    policy,
    reply_text,
    sse_bytes,
    stop_at,
    write_handoff,
)
from foremankit.gateway_trim import request_chars

# now=1_700_000_000 is 2023-11-14 in the UTC timezone of this test environment.
FIXED_NOW = 1_700_000_000

# At 4 chars per token, 36_000 'x' plus the JSON wrapper around them exceeds
# 0.90 * 10_000 tokens, so a request of this size is always stopped.
STOP_WINDOW = 10_000
LOCKED_WINDOW = 10_000_000


def tool_use(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """An assistant content block asking to call a tool."""
    return {"type": "tool_use", "name": name, "input": tool_input}


def user_message(text: str) -> dict[str, Any]:
    """A user message made of one text block."""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant_message(text: str) -> dict[str, Any]:
    """An assistant message made of one text block."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def request_with_messages(*messages: dict[str, Any]) -> dict[str, Any]:
    """A minimal request body carrying tools and the given messages."""
    return {"tools": [{"name": "spawn"}], "messages": list(messages)}


def stop_info(tokens: int = 9_100, window: int = 10_000, ratio: float = 0.90) -> dict[str, Any]:
    """The `stop` dict stop_at() returns for a full window."""
    return {"tokens": tokens, "window": window, "ratio": ratio}


def where_info() -> dict[str, Any]:
    """The `where` dict handed to handoff_markdown()."""
    return {"mode": "local", "model": "qwen-35b"}


def session_body() -> dict[str, Any]:
    """A realistic small session: user prompt, tool call and the model's answer."""
    tool = tool_use("Read", {"file_path": "C:/repo/a.py"})
    assistant = {"role": "assistant", "content": [{"type": "text", "text": "Reading a.py"}, tool]}
    return request_with_messages(user_message("please fix the bug"), assistant)


def full_handoff(data: dict[str, Any]) -> str:
    """The handoff markdown of `data` rendered at the fixed timestamp."""
    return handoff_markdown(data, stop_info(), where_info(), now=FIXED_NOW)


def test_policy_defaults_when_block_is_missing() -> None:
    result = policy({})

    assert result["enabled"] is True
    assert result["ratio"] == 0.85
    assert result["chars_per_token"] == 4
    assert result["dir"] == ""


def test_policy_keeps_a_valid_ratio_and_fills_the_rest() -> None:
    result = policy({"context_stop": {"ratio": 0.95}})

    assert result["ratio"] == 0.95
    assert result["enabled"] is True
    assert result["chars_per_token"] == 4
    assert result["dir"] == ""


@pytest.mark.parametrize("bad_ratio", [1.5, "x"])
def test_policy_falls_back_for_an_invalid_ratio(bad_ratio: object) -> None:
    result = policy({"context_stop": {"ratio": bad_ratio}})

    assert result["ratio"] == DEFAULTS["ratio"]


def test_policy_falls_back_for_a_zero_chars_per_token() -> None:
    result = policy({"context_stop": {"chars_per_token": 0}})

    assert result["chars_per_token"] == DEFAULTS["chars_per_token"]


def test_policy_falls_back_for_a_non_bool_enabled() -> None:
    result = policy({"context_stop": {"enabled": "yes"}})

    assert result["enabled"] is True


def test_policy_falls_back_for_a_non_str_dir() -> None:
    result = policy({"context_stop": {"dir": 5}})

    assert result["dir"] == ""


def test_estimate_tokens_matches_request_chars_divided_by_four() -> None:
    data = request_with_messages(user_message("x" * 1_000))

    assert estimate_tokens(data) == request_chars(data) // 4


def test_estimate_tokens_is_zero_for_a_zero_chars_per_token() -> None:
    data = request_with_messages(user_message("x" * 1_000))

    assert estimate_tokens(data, chars_per_token=0) == 0


def test_stop_at_returns_none_when_disabled() -> None:
    pol = {**DEFAULTS, "enabled": False}
    data = request_with_messages(user_message("x" * 36_000))

    assert stop_at(data, STOP_WINDOW, pol) is None


def test_stop_at_returns_none_for_a_zero_window() -> None:
    data = request_with_messages(user_message("x" * 36_000))

    assert stop_at(data, 0, DEFAULTS) is None


def test_stop_at_returns_none_without_tools() -> None:
    data = {"messages": [user_message("x" * 36_000)]}

    assert stop_at(data, STOP_WINDOW, DEFAULTS) is None


def test_stop_at_returns_none_below_the_ratio() -> None:
    data = request_with_messages(user_message("short prompt"))

    assert stop_at(data, STOP_WINDOW, DEFAULTS) is None


def test_stop_at_stops_once_the_prompt_reaches_the_ratio() -> None:
    data = request_with_messages(user_message("x" * 36_000))

    stop = stop_at(data, STOP_WINDOW, DEFAULTS)

    assert stop is not None
    assert set(stop) == {"tokens", "window", "ratio"}
    assert stop["tokens"] >= 0.90 * STOP_WINDOW


def test_handoff_contains_every_section_heading() -> None:
    text = full_handoff(session_body())

    for heading in (
        "# Handoff - context stop",
        "- when: ",
        "## What was asked (last user messages)",
        "## Last answer of the model",
        "## Tools called",
        "## Files touched",
        "## Next step",
    ):
        assert heading in text


def test_handoff_shows_model_and_mode() -> None:
    text = full_handoff(session_body())

    assert "qwen-35b" in text
    assert "local" in text


def test_handoff_keeps_only_the_last_three_user_texts() -> None:
    requests = [user_message(f"request-{index}") for index in range(5)]

    text = full_handoff(request_with_messages(*requests))

    assert "request-0" not in text
    assert "request-1" not in text
    assert "request-2" in text
    assert "request-3" in text
    assert "request-4" in text


def test_handoff_keeps_the_last_assistant_text() -> None:
    data = request_with_messages(
        assistant_message("first answer"),
        user_message("again"),
        assistant_message("final answer"),
    )

    text = full_handoff(data)

    assert "final answer" in text


def test_handoff_lists_a_touched_file_and_its_tool_hint() -> None:
    text = full_handoff(session_body())

    files_part = text.split("## Files touched", 1)[1]
    tools_part = text.split("## Tools called", 1)[1].split("## Files touched", 1)[0]
    assert "C:/repo/a.py" in files_part
    assert "Read" in tools_part
    assert "C:/repo/a.py" in tools_part


def test_handoff_shortens_a_long_user_text() -> None:
    text = full_handoff(request_with_messages(user_message("y" * 900)))

    assert "…" in text
    assert "y" * 900 not in text


def test_handoff_renders_the_fixed_timestamp() -> None:
    text = full_handoff(session_body())

    assert "2023-11-14" in text


@pytest.mark.parametrize(
    "data",
    [
        {"messages": "not a list"},
        {"messages": [None, 5, {"role": "user", "content": "plain string"}]},
        {},
    ],
)
def test_handoff_survives_a_malformed_body(data: dict[str, Any]) -> None:
    text = handoff_markdown(data, stop_info(), where_info(), now=FIXED_NOW)

    assert isinstance(text, str)
    assert "# Handoff - context stop" in text


def test_folder_defaults_to_handoffs_under_base(tmp_path: Path) -> None:
    assert folder({"dir": ""}, tmp_path) == tmp_path / "handoffs"


def test_folder_uses_the_configured_dir(tmp_path: Path) -> None:
    custom = tmp_path / "custom"

    assert folder({"dir": str(custom)}, tmp_path) == custom


def test_write_handoff_creates_folder_and_writes_content(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "handoffs"

    path = write_handoff(target, "hello handoff", now=FIXED_NOW)

    assert target.is_dir()
    assert re.search(r"handoff-\d{8}-\d{6}\.md$", path.name)
    assert path.read_text(encoding="utf-8") == "hello handoff"


def test_write_handoff_does_not_overwrite_in_the_same_second(tmp_path: Path) -> None:
    first = write_handoff(tmp_path, "first", now=FIXED_NOW)

    second = write_handoff(tmp_path, "second", now=FIXED_NOW)

    assert second.name.endswith("-2.md")
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"


def test_reply_text_english_reports_the_actual_fill(tmp_path: Path) -> None:
    path = tmp_path / "handoff.md"

    text = reply_text(stop_info(tokens=9_100, window=10_000), path)

    assert "context stop" in text
    assert "91%" in text
    assert f"handoff: {path.resolve()}" in text
    assert "/clear" in text


def test_reply_text_polish_reports_the_actual_fill(tmp_path: Path) -> None:
    path = tmp_path / "handoff.md"

    text = reply_text(stop_info(tokens=9_100, window=10_000), path, language="Polish")

    assert "kontekst zapełniony w" in text
    assert "91%" in text
    assert "handoff: " in text
    assert "/clear" in text
    assert "NOWĄ sesję" in text


def test_message_json_carries_a_text_only_answer() -> None:
    payload = message_json("stopped", "qwen-35b", stop_info())

    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["model"] == "qwen-35b"
    assert payload["content"] == [{"type": "text", "text": "stopped"}]
    assert payload["stop_reason"] == "end_turn"
    assert payload["usage"]["input_tokens"] == 9_100
    assert "tool_use" not in json.dumps(payload)


def test_sse_bytes_is_a_text_only_anthropic_stream() -> None:
    body = sse_bytes("stopped", "qwen-35b", stop_info())

    text = body.decode("utf-8")
    assert "event: message_start" in text
    assert "event: content_block_delta" in text
    assert "event: message_stop" in text
    assert '"type": "text_delta"' in text
    assert '"stop_reason": "end_turn"' in text
    assert "stopped" in text
    assert "tool_use" not in text
