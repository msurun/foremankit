"""
Translation layer between the Anthropic Messages API and the OpenAI chat-completions API.

Converts request and response bodies so the foremankit gateway can speak to Ollama,
vLLM, or llama-server endpoints while the rest of the application continues to use
Anthropic-shaped data structures.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Iterable
from typing import Any
from uuid import uuid4


def openai_path(path: str) -> str | None:
    """Map an Anthropic API path to its OpenAI equivalent, or None if unmappable."""
    if path == "/v1/messages":
        return "/v1/chat/completions"
    return None


def request_to_openai(data: dict, model: str) -> dict:
    """Convert an Anthropic Messages request body to an OpenAI chat-completions body."""
    result: dict[str, Any] = {"model": model, "messages": _build_messages(data)}
    _copy_common_params(data, result)

    tools = _openai_tools(data.get("tools", []))
    if tools:
        result["tools"] = tools

    tool_choice = _openai_tool_choice(data.get("tool_choice"))
    if tool_choice is not None:
        result["tool_choice"] = tool_choice

    if data.get("stream"):
        result["stream"] = True
        result["stream_options"] = {"include_usage": True}

    return result


def _build_messages(data: dict) -> list[dict]:
    """Build the OpenAI message list: the system message first, then every turn IN ITS ORIGINAL ORDER."""
    messages: list[dict] = []
    system_block = _system_message(data.get("system"))
    if system_block is not None:
        messages.append(system_block)
    for message in data.get("messages", []):
        messages.extend(_turn_messages(message))
    return messages


def _turn_messages(message: dict) -> list[dict]:
    """Convert one Anthropic message into its OpenAI counterpart ([] for an unknown role)."""
    if message.get("role") == "user":
        return _user_messages([message])
    if message.get("role") == "assistant":
        return _assistant_messages([message])
    return []


def response_to_anthropic(data: dict, model: str) -> dict:
    """Convert one OpenAI completion JSON to an Anthropic message dict."""
    content = _extract_content(data)
    stop_reason = _stop_reason_of(data, content)

    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    message_id = data.get("id") or uuid4().hex
    return {
        "id": f"msg_{message_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


def count_tokens(data: dict) -> dict:
    """Estimate input tokens from the system prompt and the messages, as the count_tokens API expects."""
    total_chars = _sum_text_length(data.get("system")) + _sum_text_length(data.get("messages"))
    return {"input_tokens": max(1, total_chars // 4)}


def _finish_reason_of(data: dict) -> str | None:
    """Return the finish_reason of the first choice (a few servers repeat it at the top level)."""
    choices = data.get("choices") or []
    if choices:
        return choices[0].get("finish_reason") or data.get("finish_reason")
    return data.get("finish_reason")


def _stop_reason_of(data: dict, content: list[dict]) -> str:
    """Map the completion's finish_reason, falling back to tool_use when the answer carries tool calls."""
    finish_reason = _finish_reason_of(data)
    if finish_reason is None and any(block.get("type") == "tool_use" for block in content):
        return "tool_use"
    return _map_stop_reason(finish_reason)


def _sse_payload(line: str) -> str | None:
    """Return the text after an SSE 'data:' prefix, or None for any other line."""
    if line.startswith("data: "):
        return line[6:]
    if line.startswith("data:"):
        return line[5:]
    return None


class AnthropicStream:
    """Convert OpenAI streaming chunks into Anthropic server-sent events."""

    def __init__(self, model: str, message_id: str | None = None) -> None:
        """Initialise the stream converter with model and optional message identifier."""
        self._model = model
        self._message_id = message_id or f"msg_{uuid4().hex}"
        self._buffer = ""
        self._started = False
        self._finished = False
        self._block_index = 0
        self._open_block: int | None = None
        self._text_block: int | None = None
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._finish_reason: str | None = None

    def feed(self, chunk: str) -> list[bytes]:
        """Accept raw text from OpenAI and return Anthropic SSE byte events."""
        events: list[bytes] = []
        self._buffer += chunk

        while True:
            line = self._pop_line()
            if line is None:
                break
            if line == "[DONE]":
                self._buffer = ""
                break
            events.extend(self._handle_line(line))

        return events

    def _pop_line(self) -> str | None:
        """Remove and return the next complete stripped line from the buffer, or None."""
        newline_index = self._buffer.find("\n")
        if newline_index == -1:
            return None
        line = self._buffer[:newline_index].strip()
        self._buffer = self._buffer[newline_index + 1 :]
        return line

    def _handle_line(self, line: str) -> list[bytes]:
        """Parse one SSE line and return the Anthropic events it produces, if any."""
        payload = _sse_payload(line)
        if payload is None:
            return []
        try:
            event_data = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(event_data, dict):
            return []
        return self._process_openai_chunk(event_data)

    def finish(self) -> list[bytes]:
        """Close any open block and emit the final message_delta and message_stop events."""
        if self._finished:
            return []

        events: list[bytes] = self._close_open_block()
        stop_reason = self._stop_reason()
        usage = {
            "input_tokens": self._prompt_tokens,
            "output_tokens": self._completion_tokens,
        }
        delta = {"stop_reason": stop_reason, "stop_sequence": None, "usage": usage}
        events.append(self._sse_event("message_delta", {"delta": delta, "usage": usage}))
        events.append(self._sse_event("message_stop", {}))
        self._finished = True
        return events

    def _process_openai_chunk(self, chunk: dict) -> list[bytes]:
        """Convert one parsed OpenAI streaming chunk into Anthropic SSE events."""
        events: list[bytes] = []

        choices = chunk.get("choices") or []
        if choices:
            events.extend(self._handle_choice(choices[0]))

        usage = chunk.get("usage")
        if usage:
            self._prompt_tokens = usage.get("prompt_tokens", self._prompt_tokens)
            self._completion_tokens = usage.get("completion_tokens", self._completion_tokens)

        return events

    def _handle_choice(self, choice: dict) -> list[bytes]:
        """Process one OpenAI choice delta and produce the corresponding events."""
        events: list[bytes] = []
        delta = choice.get("delta") or {}

        if "content" in delta and delta["content"]:
            events.extend(self._handle_content_delta(delta["content"] or ""))
        if "tool_calls" in delta:
            events.extend(self._handle_tool_calls(delta["tool_calls"] or []))

        if choice.get("finish_reason"):
            self._finish_reason = choice["finish_reason"]

        return events

    def _handle_content_delta(self, text: str) -> list[bytes]:
        """Generate events for a text content delta."""
        events: list[bytes] = []
        if not self._started:
            events.extend(self._emit_message_start())
        events.extend(self._ensure_text_block())
        events.append(
            self._sse_event(
                "content_block_delta",
                {"index": self._text_block, "delta": {"type": "text_delta", "text": text}},
            )
        )
        return events

    def _ensure_text_block(self) -> list[bytes]:
        """Close any other open block and open a text block unless one is already open."""
        if self._text_block is not None:
            return []
        events = self._close_open_block()
        events.append(
            self._sse_event(
                "content_block_start",
                {"index": self._block_index, "content_block": {"type": "text", "text": ""}},
            )
        )
        self._open_block = self._text_block = self._block_index
        self._block_index += 1
        return events

    def _handle_tool_calls(self, tool_calls: list[dict]) -> list[bytes]:
        """Generate events for tool call deltas, opening new content blocks as needed."""
        events: list[bytes] = []
        for tool_call in tool_calls:
            index = tool_call.get("index", 0)
            if index not in self._tool_calls:
                events.extend(self._open_tool_block(index, tool_call))
            events.extend(self._tool_arguments_events(index, tool_call))
        return events

    def _open_tool_block(self, index: int, tool_call: dict) -> list[bytes]:
        """Register a new tool call and emit its content_block_start event."""
        events = self._close_open_block()
        if not self._started:
            events.extend(self._emit_message_start())
        tool_id = tool_call.get("id") or f"toolu_{uuid4().hex[:8]}"
        function_info = tool_call.get("function") or {}
        tool_name = function_info.get("name", "")
        self._tool_calls[index] = {"id": tool_id, "name": tool_name, "block": self._block_index}
        block = {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}}
        events.append(
            self._sse_event(
                "content_block_start",
                {"index": self._block_index, "content_block": block},
            )
        )
        self._open_block = self._block_index
        self._block_index += 1
        return events

    def _tool_arguments_events(self, index: int, tool_call: dict) -> list[bytes]:
        """Emit an input_json_delta event for the arguments fragment of a tool call, if any."""
        function_info = tool_call.get("function") or {}
        arguments = function_info.get("arguments", "")
        if not arguments:
            return []
        delta = {"type": "input_json_delta", "partial_json": arguments}
        block_index = self._tool_calls[index]["block"]
        return [self._sse_event("content_block_delta", {"index": block_index, "delta": delta})]

    def _close_open_block(self) -> list[bytes]:
        """Emit a content_block_stop event for the open block, whatever its type, and forget it."""
        if self._open_block is None:
            return []
        index = self._open_block
        self._open_block = None
        self._text_block = None
        return [self._sse_event("content_block_stop", {"index": index})]

    def _stop_reason(self) -> str:
        """Return the stop reason, choosing tool_use when tool calls arrived without a finish_reason."""
        if not self._finish_reason and self._tool_calls:
            return "tool_use"
        return _map_stop_reason(self._finish_reason)

    def _emit_message_start(self) -> list[bytes]:
        """Emit the message_start event which must precede any content."""
        self._started = True
        message = {
            "id": self._message_id,
            "type": "message",
            "role": "assistant",
            "model": self._model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": self._prompt_tokens, "output_tokens": 0},
        }
        return [self._sse_event("message_start", {"message": message})]

    def _sse_event(self, event_type: str, data: dict) -> bytes:
        """Serialise an SSE event to its byte representation."""
        payload = json.dumps(data)
        return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")


async def stream_to_anthropic(chunks: AsyncIterable[bytes | str], model: str) -> AsyncIterable[bytes]:
    """Convert an async stream of OpenAI chunks into an async stream of Anthropic SSE bytes."""
    converter = AnthropicStream(model)
    async for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        for event in converter.feed(chunk):
            yield event
    for event in converter.finish():
        yield event


def _system_message(system: Any) -> dict | None:
    """Extract a single system message from the Anthropic 'system' field."""
    if system is None:
        return None
    if isinstance(system, str):
        return {"role": "system", "content": system}
    if isinstance(system, list):
        text = "\n\n".join(block.get("text", "") for block in system if block.get("type") == "text")
        if text:
            return {"role": "system", "content": text}
    return None


def _user_messages(messages: list[dict]) -> list[dict]:
    """Convert Anthropic user messages into OpenAI chat messages."""
    result: list[dict] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            result.append({"role": "user", "content": content})
        elif isinstance(content, list):
            result.extend(_user_block_messages(content))
    return result


def _user_block_messages(blocks: list[dict]) -> list[dict]:
    """Convert a single user content block list into zero or more OpenAI messages."""
    tool_messages: list[dict] = []
    text_parts: list[str] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "tool_result":
            tool_messages.append(_tool_result_message(block))
        else:
            part = _user_text_part(block)
            if part is not None:
                text_parts.append(part)

    if text_parts:
        tool_messages.append({"role": "user", "content": "".join(text_parts)})
    return tool_messages


def _user_text_part(block: dict) -> str | None:
    """Return the text a user text or image block contributes to a user message, or None."""
    block_type = block.get("type")
    if block_type == "text":
        return block.get("text", "")
    if block_type == "image":
        return "[image omitted]"
    return None


def _tool_result_message(block: dict) -> dict:
    """Convert an Anthropic tool_result block into an OpenAI tool message."""
    content = block.get("content", "")
    if isinstance(content, list):
        content = _text_of(content)
    return {"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": content}


def _assistant_messages(messages: list[dict]) -> list[dict]:
    """Convert Anthropic assistant messages into OpenAI chat messages."""
    result: list[dict] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            result.append({"role": "assistant", "content": content})
        elif isinstance(content, list):
            result.append(_assistant_block_message(content))
    return result


def _assistant_block_message(blocks: list[dict]) -> dict:
    """Convert one assistant content block list into a single OpenAI message."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append(_tool_use_call(block))

    result: dict[str, Any] = {"role": "assistant", "content": _assistant_text(text_parts, tool_calls)}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _tool_use_call(block: dict) -> dict:
    """Convert an Anthropic tool_use block into an OpenAI tool call entry."""
    return {
        "id": block.get("id", ""),
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input") or {}),
        },
    }


def _assistant_text(text_parts: list[str], tool_calls: list[dict]) -> str | None:
    """Pick the assistant message content: joined text, None when only tool calls, else empty."""
    if text_parts:
        return "".join(text_parts)
    if tool_calls:
        return None
    return ""


def _openai_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function definitions."""
    result = []
    for tool in tools:
        schema = tool.get("input_schema") or {"type": "object", "properties": {}}
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": schema,
                },
            }
        )
    return result


def _openai_tool_choice(tool_choice: Any) -> Any:
    """Map an Anthropic tool_choice to its OpenAI equivalent, or None if absent."""
    if tool_choice is None:
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return None


def _copy_common_params(data: dict, result: dict) -> None:
    """Copy whitelisted optional parameters from the Anthropic request to the OpenAI request."""
    for key in ("max_tokens", "temperature", "top_p"):
        if key in data:
            result[key] = data[key]
    if "stop_sequences" in data:
        result["stop"] = data["stop_sequences"]


def _text_of(blocks: list[dict]) -> str:
    """Join all text blocks in a list into a single string."""
    return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")


def _extract_content(data: dict) -> list[dict]:
    """Extract Anthropic-style content blocks from an OpenAI completion."""
    content_blocks: list[dict] = []

    choices = data.get("choices") or []
    if not choices:
        return content_blocks

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content:
        content_blocks.append({"type": "text", "text": content})

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        content_blocks.append(_tool_use_block(tool_call))

    return content_blocks


def _tool_use_block(tool_call: dict) -> dict:
    """Convert one OpenAI tool call into an Anthropic tool_use content block."""
    function_info = tool_call.get("function") or {}
    return {
        "type": "tool_use",
        "id": tool_call.get("id", ""),
        "name": function_info.get("name", ""),
        "input": _parse_tool_input(function_info.get("arguments") or "{}"),
    }


def _parse_tool_input(arguments: str) -> dict:
    """Parse a tool call's JSON arguments, returning an empty dict if they are not a JSON object."""
    try:
        parsed_input = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed_input if isinstance(parsed_input, dict) else {}


def _map_stop_reason(finish_reason: str | None) -> str:
    """Map an OpenAI finish_reason to an Anthropic stop_reason."""
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_use"
    return "end_turn"


def _sum_text_length(data: Any) -> int:
    """Recursively sum the character length of all strings and JSON-serialised content."""
    if isinstance(data, str):
        return len(data)
    if isinstance(data, dict):
        return sum(_sum_text_length(value) for value in data.values())
    if isinstance(data, list):
        return sum(_sum_text_length(item) for item in data)
    if data is None or isinstance(data, (int, float, bool)):
        return 0
    return _sum_text_length(json.dumps(data))
