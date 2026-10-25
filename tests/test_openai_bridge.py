"""Behaviour tests for the Anthropic <-> OpenAI translation bridge.

Each test exercises one piece of the contract of `foremankit.openai_bridge`:
path mapping, request/response conversion, token counting and streaming.
"""

import asyncio
import json

import pytest

from foremankit import openai_bridge as ob

SYSTEM_MESSAGE_TEXT = "sys"
FILLER_CHARS = 8000  # enough filler to dwarf the token estimate of real text
LONG_USER_TEXT_CHARS = 400  # 400 chars // 4 == 100 tokens


def parse_event(event: bytes) -> tuple[str, object]:
    """Split a streamed byte event into its SSE event type and decoded JSON payload."""
    text = event.decode()
    event_line, data_line = text.split("\n", 1)
    event_type = event_line.removeprefix("event: ")
    payload = json.loads(data_line.removeprefix("data: ").strip())
    return event_type, payload


def delta_text_chunks(events: list[bytes]) -> str:
    """Concatenate every text_delta payload found in a list of streamed events."""
    pieces = []
    for event in events:
        event_type, payload = parse_event(event)
        if event_type == "content_block_delta" and payload["delta"]["type"] == "text_delta":
            pieces.append(payload["delta"]["text"])
    return "".join(pieces)


def user_turn(text: str) -> dict:
    """Build a minimal Anthropic user message containing one text block."""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant_text_turn(text: str) -> dict:
    """Build a minimal Anthropic assistant message containing one text block."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def text_response(finish_reason: str | None) -> dict:
    """Build an OpenAI chat-completion response carrying one text answer."""
    message = {"role": "assistant", "content": "hi"}
    choice = {"message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"id": "abc", "model": "gpt-x", "choices": [choice]}


def tool_response(arguments: str, finish_reason: str | None) -> dict:
    """Build an OpenAI chat-completion response carrying one tool call."""
    call = {"id": "call_1", "function": {"name": "calc", "arguments": arguments}}
    choice = {"message": {"role": "assistant", "content": None, "tool_calls": [call]}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"id": "abc", "model": "gpt-x", "choices": [choice]}


def stream_chunk(delta: dict, finish_reason: str | None = None) -> str:
    """Serialise one OpenAI stream chunk line with a single-choice delta."""
    chunk = {
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n"


def test_openai_path_maps_messages_endpoint():
    """The Anthropic messages path becomes the OpenAI chat completions path."""
    assert ob.openai_path("/v1/messages") == "/v1/chat/completions"


def test_openai_path_returns_none_for_other_endpoints():
    """Any other path is not translated."""
    assert ob.openai_path("/v1/complete") is None


def rich_request_body() -> dict:
    """Build a request using every feature the translation must carry over."""
    return {
        "system": SYSTEM_MESSAGE_TEXT,
        "max_tokens": 100,
        "temperature": 0.2,
        "stop_sequences": ["END"],
        "tools": [
            {
                "name": "calc",
                "description": "Add numbers",
                "input_schema": {"type": "object"},
            }
        ],
        "tool_choice": {"type": "any"},
        "stream": True,
        "messages": [user_turn("hello")],
    }


def test_request_conversion_sets_messages_and_parameters():
    """System, sampling, tool and stream settings are translated exactly."""
    result = ob.request_to_openai(rich_request_body(), "gpt-x")

    assert result["messages"][0] == {"role": "system", "content": SYSTEM_MESSAGE_TEXT}
    assert result["max_tokens"] == 100
    assert result["temperature"] == 0.2
    assert result["stop"] == ["END"]
    assert result["tools"][0]["function"]["name"] == "calc"
    assert result["tool_choice"] == "required"
    assert result["stream"] is True
    assert result["stream_options"] == {"include_usage": True}


def test_request_without_stream_has_no_stream_options():
    """A non-streaming request must not ask for usage in a stream."""
    body = rich_request_body()
    body["stream"] = False
    result = ob.request_to_openai(body, "gpt-x")

    assert "stream_options" not in result


@pytest.mark.parametrize(
    ("anthropic_choice", "openai_choice"),
    [
        ({"type": "auto"}, "auto"),
        ({"type": "none"}, "none"),
        ({"type": "tool", "name": "calc"}, {"type": "function", "function": {"name": "calc"}}),
    ],
)
def test_tool_choice_mapping(anthropic_choice: dict, openai_choice: object):
    """Each Anthropic tool_choice shorthand becomes its OpenAI counterpart."""
    body = {"messages": [user_turn("hi")], "tool_choice": anthropic_choice}
    assert ob.request_to_openai(body, "gpt-x")["tool_choice"] == openai_choice


def test_absent_tool_choice_is_not_invented():
    """Without a tool_choice in the body, none appears in the request."""
    result = ob.request_to_openai({"messages": [user_turn("hi")]}, "gpt-x")

    assert "tool_choice" not in result


def mixed_conversation() -> list[dict]:
    """Build a four-turn conversation mixing text, a tool call and a tool result."""
    tool_use = {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}
    tool_result = {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
    return [
        user_turn("first"),
        {"role": "assistant", "content": [{"type": "text", "text": "mid"}, tool_use]},
        {"role": "user", "content": [tool_result, {"type": "text", "text": "after"}]},
        user_turn("last"),
    ]


def test_conversation_preserves_order_and_splits_tool_result():
    """Turns keep their order; a tool result becomes its own message before the text."""
    result = ob.request_to_openai({"messages": mixed_conversation()}, "gpt-x")
    messages = result["messages"]

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "user",
        "user",
    ]
    assert messages[2]["tool_call_id"] == "t1"
    arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"a": 1}


def test_text_response_converts_content_finish_and_usage():
    """A plain text answer maps to end_turn with echoed model and usage."""
    result = ob.response_to_anthropic(text_response("stop"), "echo-model")

    assert result["content"] == [{"type": "text", "text": "hi"}]
    assert result["stop_reason"] == "end_turn"
    assert result["model"] == "echo-model"
    assert result["id"].startswith("msg_")


def test_text_response_copies_usage_tokens():
    """Prompt and completion tokens become input and output tokens."""
    raw = text_response("stop")
    raw["usage"] = {"prompt_tokens": 7, "completion_tokens": 3}
    usage = ob.response_to_anthropic(raw, "gpt-x")["usage"]

    assert usage == {"input_tokens": 7, "output_tokens": 3}


def test_length_finish_reason_maps_to_max_tokens():
    """A truncated answer signals max_tokens."""
    assert ob.response_to_anthropic(text_response("length"), "m")["stop_reason"] == "max_tokens"


def test_tool_call_finish_reason_maps_to_tool_use_with_parsed_input():
    """A tool call answer maps to tool_use and its JSON arguments are parsed."""
    result = ob.response_to_anthropic(tool_response('{"a": 1}', "tool_calls"), "m")
    block = result["content"][0]

    assert result["stop_reason"] == "tool_use"
    assert block["type"] == "tool_use"
    assert block["input"] == {"a": 1}


def test_tool_call_without_finish_reason_is_tool_use():
    """A tool call with no finish_reason at all still signals tool_use."""
    result = ob.response_to_anthropic(tool_response('{"a": 1}', None), "m")

    assert result["stop_reason"] == "tool_use"


def test_invalid_tool_arguments_yield_empty_input():
    """Arguments that are not valid JSON fall back to an empty input dict."""
    result = ob.response_to_anthropic(tool_response("{oops", "tool_calls"), "m")

    assert result["content"][0]["input"] == {}


def test_count_tokens_ignores_tools_filler():
    """Tools are not counted, so a large tool schema does not inflate tokens."""
    body = {
        "messages": [user_turn("hi")],
        "tools": [{"name": "t", "description": "x" * FILLER_CHARS}],
    }
    assert ob.count_tokens(body)["input_tokens"] < 100


def test_count_tokens_counts_message_characters():
    """A 400-character user message counts as roughly 100 tokens (structural keys add a few)."""
    body = {"messages": [user_turn("a" * LONG_USER_TEXT_CHARS)]}
    tokens = ob.count_tokens(body)["input_tokens"]

    assert 100 <= tokens <= 104


def test_text_stream_buffers_partial_line_across_feeds():
    """A chunk split mid-line produces events only once the newline arrives."""
    stream = ob.AnthropicStream("gpt-x")
    line = stream_chunk({"content": "Hello"})
    assert stream.feed(line[:10]) == []

    events = stream.feed(line[10:])
    types = [parse_event(event)[0] for event in events]

    assert types == ["message_start", "content_block_start", "content_block_delta"]
    assert "Hello" in delta_text_chunks(events)


def test_finish_closes_text_block_and_reports_usage():
    """Finish closes the block, reports usage and stops; a second call is empty."""
    stream = ob.AnthropicStream("gpt-x")
    stream.feed(stream_chunk({"content": "Hello"}))
    stream.feed(stream_chunk({}))
    usage_line = 'data: {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4}}\n'
    stream.feed(usage_line)

    events = stream.finish()
    types = [parse_event(event)[0] for event in events]
    message_delta = next(
        parse_event(event)[1] for event in events if parse_event(event)[0] == "message_delta"
    )

    assert types[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert message_delta["delta"]["stop_reason"] == "end_turn"
    assert message_delta["usage"] == {"input_tokens": 11, "output_tokens": 4}
    assert stream.finish() == []


def test_finish_on_untouched_stream_still_stops():
    """Finishing a stream that produced nothing still ends with message_stop."""
    events = ob.AnthropicStream("gpt-x").finish()

    assert parse_event(events[-1])[0] == "message_stop"


def tool_call_chunks() -> list[str]:
    """Build two argument fragments plus the tool-call finish chunk."""
    call_start = {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "calc"}}]}
    first = stream_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]})
    second = stream_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]})
    return [stream_chunk(call_start), first, second, stream_chunk({}, "tool_calls")]


def test_tool_stream_emits_partial_json_on_one_index():
    """Arguments arrive as two partial_json deltas on the tool_use block index."""
    stream = ob.AnthropicStream("gpt-x")
    events = [event for line in tool_call_chunks() for event in stream.feed(line)]
    start = next(
        parse_event(event)[1] for event in events if parse_event(event)[0] == "content_block_start"
    )
    partials = [
        parse_event(event)[1] for event in events if parse_event(event)[0] == "content_block_delta"
    ]

    assert start["content_block"]["type"] == "tool_use"
    assert start["index"] == 0
    assert [partial["index"] for partial in partials] == [0, 0]
    assert "".join(partial["delta"]["partial_json"] for partial in partials) == '{"a":1}'


def test_tool_stream_finish_closes_block_before_message_delta():
    """Finish emits content_block_stop on the tool block then a tool_use stop reason."""
    stream = ob.AnthropicStream("gpt-x")
    for line in tool_call_chunks():
        stream.feed(line)

    events = stream.finish()
    types = [parse_event(event)[0] for event in events]
    message_delta = next(
        parse_event(event)[1] for event in events if parse_event(event)[0] == "message_delta"
    )

    assert types[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_text_then_tool_stream_switches_block_index():
    """Opening a tool block closes the text block first and shifts the index."""
    stream = ob.AnthropicStream("gpt-x")
    events = stream.feed(stream_chunk({"content": "Hi"}))

    tool_start = {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "calc"}}]}
    events += stream.feed(stream_chunk(tool_start))
    events += stream.feed(stream_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}))
    types = [parse_event(event)[0] for event in events]
    payloads = [parse_event(event)[1] for event in events]
    indexes = [payload["index"] for payload in payloads if "index" in payload]

    assert types.index("content_block_stop") < len(types) - 2
    assert indexes[-1] == 1


@pytest.mark.parametrize(
    "noise_line",
    ["[DONE]\n", "data: not-json\n", "data: [1, 2]\n", "comment\n"],
)
def test_stream_ignores_unparseable_lines(noise_line: str):
    """Done markers, bad JSON, lists and comments produce no events and no errors."""
    assert ob.AnthropicStream("gpt-x").feed(noise_line) == []


def test_stream_to_anthropic_handles_str_and_bytes_chunks():
    """The async bridge yields a full event sequence across chunk types."""

    async def chunks():
        """Yield one text chunk as text and one finish chunk as bytes."""
        yield stream_chunk({"content": "Hi"})
        yield stream_chunk({}, "stop").encode()

    async def collect() -> list[bytes]:
        """Collect every event the async bridge yields into a list."""
        return [event async for event in ob.stream_to_anthropic(chunks(), "gpt-x")]

    output = b"".join(asyncio.run(collect()))
    types = [
        parse_event(fragment)[0] for fragment in output.split(b"\n\n") if fragment.startswith(b"event:")
    ]

    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
