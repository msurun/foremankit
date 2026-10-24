"""Hard context stop for Claude Code sessions proxied through the gateway.

A session hook can only ask the model to hand over; at 90% the model ignores
it. This module decides the stop from the request body alone and drafts a
handoff message without calling any model, so the gateway can answer
"stopped, handoff written" instead of running another degraded turn.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from foremankit.gateway_trim import request_chars

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "ratio": 0.90,
    "chars_per_token": 4,
    "dir": "",
}

# A tool-less side request (session title, safety check) must never be stopped.
_LAST_MESSAGES_SCANNED = 40
_LAST_USER_MESSAGES = 3
_LAST_TOOLS_SHOWN = 25
_LAST_FILES_SHOWN = 25
_USER_TEXT_CUT = 500
_ASSISTANT_TEXT_CUT = 1000
_HINT_CUT = 160
_APPROX_CHARS_PER_TOKEN = 4
_FILE_INPUT_KEYS = ("file_path", "path", "notebook_path")
_HINT_INPUT_KEYS = (
    "file_path",
    "path",
    "notebook_path",
    "command",
    "description",
    "pattern",
    "prompt",
)
_POLISH_PREFIX = "pol"
_MESSAGE_ID = "msg_contextstop"
_NEXT_STEP_TEXT = (
    "This turn was cut before the model could finish. In the NEW session (or after /clear) "
    "read this file first and continue the last request above, without re-deriving the "
    "whole history."
)


def policy(cfg: dict[str, Any]) -> dict[str, Any]:
    """The `context_stop:` block of orchestrator.yaml over DEFAULTS; invalid or missing values fall back to DEFAULTS."""
    block = cfg.get("context_stop")
    if not isinstance(block, dict):
        return dict(DEFAULTS)
    merged = {**DEFAULTS, **block}
    ratio = merged["ratio"]
    if not isinstance(ratio, (int, float)) or not 0 < ratio <= 1:
        merged["ratio"] = DEFAULTS["ratio"]
    chars_per_token = merged["chars_per_token"]
    if not isinstance(chars_per_token, (int, float)) or chars_per_token <= 0:
        merged["chars_per_token"] = DEFAULTS["chars_per_token"]
    if not isinstance(merged["enabled"], bool):
        merged["enabled"] = DEFAULTS["enabled"]
    if not isinstance(merged["dir"], str):
        merged["dir"] = DEFAULTS["dir"]
    return merged


def estimate_tokens(data: dict[str, Any], chars_per_token: float = 4) -> int:
    """Rough prompt size in tokens: request_chars(data) // chars_per_token."""
    if chars_per_token <= 0:
        return 0
    return int(request_chars(data) // chars_per_token)


def stop_at(
    data: dict[str, Any], window: int, pol: dict[str, Any]
) -> dict[str, Any] | None:
    """None when a turn may run; else {"tokens", "window", "ratio"} when the prompt already fills `ratio` of `window`.

    Requires pol["enabled"], window > 0 and a request that carries tools: a
    tool-less side request is never stopped.
    """
    if not pol["enabled"] or window <= 0 or not data.get("tools"):
        return None
    tokens = estimate_tokens(data, pol["chars_per_token"])
    if tokens < float(pol["ratio"]) * window:
        return None
    return {"tokens": tokens, "window": window, "ratio": float(pol["ratio"])}


def handoff_markdown(
    data: dict[str, Any],
    stop: dict[str, Any],
    where: dict[str, Any],
    now: float | None = None,
) -> str:
    """A markdown handoff of the cut session, built from the request body alone.

    `where` carries {"mode": str, "model": str}. A malformed `messages` field
    is treated as empty: this runs inside a live proxy and must never crash.
    """
    messages = _messages_of(data)
    recent = messages[-_LAST_MESSAGES_SCANNED:]
    sections = [
        _header_section(stop, where, now),
        _user_section(recent),
        _assistant_section(recent),
        _tools_section(recent),
        _files_section(recent),
        _next_step_section(),
    ]
    return "\n\n".join(section for section in sections if section)


def folder(pol: dict[str, Any], base: Path) -> Path:
    """Directory for handoff files: `context_stop.dir` when set (expanded), else `<base>/handoffs`."""
    if pol.get("dir"):
        return Path(pol["dir"]).expanduser()
    return base / "handoffs"


def write_handoff(folder: Path, text: str, now: float | None = None) -> Path:
    """Write `text` to `handoff-YYYYmmdd-HHMMSS.md` in `folder` (created if missing); returns the path.

    Two calls in the same second must not overwrite each other: append -2,
    -3... when the name is taken.
    """
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))
    candidate = folder / f"handoff-{stamp}.md"
    suffix = 2
    while candidate.exists():
        candidate = folder / f"handoff-{stamp}-{suffix}.md"
        suffix += 1
    candidate.write_text(text, encoding="utf-8")
    return candidate


def reply_text(stop: dict[str, Any], path: Path, language: str = "English") -> str:
    """The 3-line message shown in Claude Code (Polish when `language` starts with "pol").

    Both variants report the ACTUAL fill (tokens / window), never the
    configured threshold, so the reader sees how full the window already is.
    """
    if language.lower().startswith(_POLISH_PREFIX):
        return _polish_reply(stop, path)
    fill = _fill_ratio(stop)
    line_one = (
        f"context stop: the prompt already fills {fill:.0%} of the window "
        f"({stop['tokens']:,} of {stop['window']:,} tokens)"
    )
    line_two = f"handoff: {path.resolve()}"
    line_three = (
        "Start a NEW session or run /clear, then tell the model to read that "
        "handoff file first and continue."
    )
    return f"{line_one}\n{line_two}\n{line_three}"


def message_json(text: str, model: str, stop: dict[str, Any]) -> dict[str, Any]:
    """A non-streaming Anthropic Messages answer carrying `text`."""
    return {
        "id": _MESSAGE_ID,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": stop["tokens"],
            "output_tokens": len(text) // _APPROX_CHARS_PER_TOKEN,
        },
    }


def sse_bytes(text: str, model: str, stop: dict[str, Any]) -> bytes:
    """The same answer as an Anthropic SSE stream (UTF-8 bytes) for Claude Code."""
    chunks = [
        f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        for name, payload in _sse_events(text, model, stop)
    ]
    return "".join(chunks).encode("utf-8")


def _sse_events(text: str, model: str, stop: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The Anthropic stream events of a text-only answer, in the order Claude Code expects."""
    start = {
        "id": _MESSAGE_ID, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": stop["tokens"], "output_tokens": 0},
    }
    delta = {"stop_reason": "end_turn", "stop_sequence": None}
    return [
        ("message_start", {"type": "message_start", "message": start}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": delta,
                           "usage": {"output_tokens": len(text) // _APPROX_CHARS_PER_TOKEN}}),
        ("message_stop", {"type": "message_stop"}),
    ]


def _messages_of(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The `messages` list of the request body, or [] when missing or malformed."""
    messages = data.get("messages")
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """The content blocks of a message as a list, or [] when the content is a string or malformed."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _text_blocks(message: dict[str, Any], role: str) -> list[str]:
    """Text of the content blocks of a message, but only when its role matches."""
    if message.get("role") != role:
        return []
    return [block["text"] for block in _content_blocks(message) if _is_text_block(block)]


def _is_text_block(block: Any) -> bool:
    """True for an Anthropic text block carrying a string."""
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _tool_inputs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The `input` dict of every tool_use block, oldest first."""
    return [block["input"] for block in _tool_use_blocks(messages)]


def _is_tool_use(block: Any) -> bool:
    """True for a tool_use block whose `input` is a dict."""
    return (
        isinstance(block, dict)
        and block.get("type") == "tool_use"
        and isinstance(block.get("input"), dict)
    )


def _path_of(tool_input: dict[str, Any]) -> str:
    """The first file path key present in a tool input, or ""."""
    for key in _FILE_INPUT_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _hint_of(tool_input: dict[str, Any]) -> str:
    """A one-line hint from the first recognised input key, cut and collapsed."""
    for key in _HINT_INPUT_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return _cut(re.sub(r"\s+", " ", value).strip(), _HINT_CUT)
    return ""


def _cut(text: str, limit: int) -> str:
    """`text` shortened to at most `limit` characters."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _bullet_list(items: list[str], empty: str = "_(none)_") -> str:
    """Bullet lines for `items`, or `empty` when there are none."""
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def _fill_ratio(stop: dict[str, Any]) -> float:
    """The actual fill of the window: tokens / window (0.0 when window is not positive)."""
    window = stop["window"]
    if window <= 0:
        return 0.0
    return stop["tokens"] / window


def _header_section(stop: dict[str, Any], where: dict[str, Any], now: float | None) -> str:
    """Title and the when/model bullet list of the handoff."""
    moment = time.strftime("%Y-%m-%d %H:%M", time.localtime(now if now is not None else time.time()))
    model_line = (
        f"- model: {where.get('model', '')} via {where.get('mode', '')} "
        f"(window {stop['window']:,}, prompt ~{stop['tokens']:,} tokens, "
        f"{_fill_ratio(stop):.0%} of the window, stop at {stop['ratio']:.0%})"
    )
    return "\n".join(["# Handoff - context stop", "", f"- when: {moment}", model_line])


def _user_section(messages: list[dict[str, Any]]) -> str:
    """The up-to-three last user text blocks, oldest first."""
    texts: list[str] = []
    for message in messages:
        texts.extend(_text_blocks(message, "user"))
    kept = [_cut(text, _USER_TEXT_CUT) for text in texts[-_LAST_USER_MESSAGES:]]
    return "\n".join(["## What was asked (last user messages)", "", _bullet_list(kept)])


def _assistant_section(messages: list[dict[str, Any]]) -> str:
    """The last assistant text block, cut to 1000 characters."""
    texts = [text for message in messages for text in _text_blocks(message, "assistant")]
    body = _cut(texts[-1], _ASSISTANT_TEXT_CUT) if texts else "_(none)_"
    return f"## Last answer of the model\n\n{body}"


def _tools_section(messages: list[dict[str, Any]]) -> str:
    """Up to 25 tool calls, each as "- <name>: <hint>"."""
    lines = []
    for block in _tool_use_blocks(messages):
        line = f"- {block.get('name', '')}: {_hint_of(block['input'])}".rstrip(": ")
        lines.append(line)
    return "\n".join(["## Tools called", "", _bullet_list(lines[-_LAST_TOOLS_SHOWN:])])


def _files_section(messages: list[dict[str, Any]]) -> str:
    """Unique file paths touched by tools, most recent last; "" when none."""
    paths: list[str] = []
    for tool_input in _tool_inputs(messages):
        path = _path_of(tool_input)
        if path and path not in paths:
            paths.append(path)
    if not paths:
        return ""
    lines = _bullet_list(paths[-_LAST_FILES_SHOWN:])
    return f"## Files touched\n\n{lines}"


def _tool_use_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every tool_use block in order, oldest first."""
    return [
        block
        for message in messages
        for block in _content_blocks(message)
        if _is_tool_use(block)
    ]


def _next_step_section() -> str:
    """The closing "what to do next" section in plain words."""
    return f"## Next step\n\n{_NEXT_STEP_TEXT}"


def _polish_reply(stop: dict[str, Any], path: Path) -> str:
    """The three-line reply in Polish, with the actual fill of the window."""
    fill = _fill_ratio(stop)
    line_one = (
        f"kontekst zapełniony w {fill:.0%} okna "
        f"({stop['tokens']:,} z {stop['window']:,} tokenów), tura zatrzymana"
    )
    line_two = f"handoff: {path.resolve()}"
    line_three = (
        "Zacznij NOWĄ sesję lub wpisz /clear, potem poleć modelowi przeczytać "
        "ten plik handoffu jako pierwszy i kontynuować."
    )
    return f"{line_one}\n{line_two}\n{line_three}"
