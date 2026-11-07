"""Read Codex exec JSONL without altering the original stdout.
Completed agent messages contain answers; turn usage contains token totals.
"""
from __future__ import annotations

import json

TOOL_ITEMS = {"command_execution", "mcp_tool_call", "web_search", "file_change"}


def events(raw: str) -> list[dict]:
    """JSON object lines only; malformed output is ignored."""
    result = []
    for line in (raw or "").splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            result.append(event)
    return result


def item(event: dict) -> dict:
    """An event's item, or an empty object for missing or malformed items."""
    value = event.get("item")
    return value if isinstance(value, dict) else {}


def answer(rows: list[dict]) -> str:
    """The last completed agent message, preserving its newlines and spacing."""
    for event in reversed(rows):
        message = item(event)
        text = message.get("text")
        if (event.get("type") == "item.completed" and message.get("type") == "agent_message"
                and isinstance(text, str) and text):
            return text
    return ""


def failure(rows: list[dict]) -> str:
    """The latest error or failed turn message, when present."""
    for event in reversed(rows):
        if event.get("type") not in ("error", "turn.failed"):
            continue
        error = event.get("error")
        message = error.get("message") if isinstance(error, dict) else event.get("message")
        if isinstance(message, str) and message:
            return f"FAIL {message}"
    return ""


def tool_calls(rows: list[dict]) -> int:
    """Count each tool item once across its started, updated and completed events."""
    calls = set()
    for event in rows:
        value = item(event)
        kind, identity = value.get("type"), value.get("id")
        if isinstance(kind, str) and kind in TOOL_ITEMS and isinstance(identity, str):
            calls.add((kind, identity))
    return len(calls)
