"""Request shaping for the local orchestrator (pure functions, no I/O).

Claude Code sends 66k-100k+ prompt tokens per request: its own system prompt, ~40 tool definitions with long
descriptions, skill and MCP lists. A local 35B model on a laptop chews through that at CPU speed, so for the plugin's
main loop the gateway swaps the system prompt for `config/orchestrator.md`, keeps only the tools the orchestrator needs
and drops the bulky reminder blocks. Subagent and cloud requests are never touched (see `is_main_loop`).
"""
from __future__ import annotations

import copy
import fnmatch
import json
import re
from typing import Any

REMINDER_TAG = "<system-reminder>"
ELLIPSIS = "…"
TRUNCATION_NOTE = "[... {removed} characters removed from an earlier tool result; run the tool again if you need them]"


def is_main_loop(data: dict[str, Any], spawn_tools: list[str]) -> bool:
    """True when the request carries a tool that spawns subagents: only the plugin's main loop has one."""
    return any(isinstance(tool, dict) and tool.get("name") in spawn_tools for tool in data.get("tools") or [])


def system_text(system: Any) -> str:
    """Flatten a `system` field (string or list of text blocks) into one string."""
    if isinstance(system, str):
        return system
    blocks = system if isinstance(system, list) else []
    return "\n".join(block.get("text", "") for block in blocks if isinstance(block, dict))


def _heading(part: str) -> str:
    """Lower-cased title of a '# Title' section, or '' when the part does not start with one."""
    match = re.match(r"# (.+)", part)
    return match.group(1).strip().lower() if match else ""


def keep_sections(text: str, headings: list[str]) -> str:
    """Return only the '# Heading' sections of `text` whose title is in `headings`."""
    wanted = {heading.strip().lower() for heading in headings}
    parts = re.split(r"(?m)^(?=# )", text)
    return "\n\n".join(part.strip() for part in parts if _heading(part) in wanted)


def shape_system(original: Any, prompt: str, trim: dict[str, Any]) -> Any:
    """New `system` value for the chosen mode: replace (prompt + kept sections), append, or keep (unchanged)."""
    mode = trim.get("system", "replace") if trim.get("enabled", True) else "append"
    if mode == "keep" or not prompt:
        return original
    if mode == "append":
        blocks = [{"type": "text", "text": system_text(original)}] if system_text(original) else []
        return blocks + [{"type": "text", "text": prompt}]
    kept = keep_sections(system_text(original), trim.get("keep_sections", []))
    return [{"type": "text", "text": "\n\n".join(part for part in (prompt, kept) if part)}]


def _cut(text: str, limit: int) -> str:
    """Shorten `text` to about `limit` characters (0 = no limit)."""
    return text if not limit or len(text) <= limit else text[:limit].rstrip() + ELLIPSIS


def trim_tools(tools: list[dict[str, Any]], keep: list[str], max_chars: int) -> list[dict[str, Any]]:
    """Keep tools whose name matches `keep` (exact or fnmatch pattern) and shorten their descriptions."""
    shaped = []
    for tool in tools:
        if not any(fnmatch.fnmatchcase(tool.get("name", ""), pattern) for pattern in keep):
            continue
        shaped.append({**tool, "description": _cut(tool.get("description", ""), max_chars)})
    return shaped


def _is_bulky_reminder(block: Any, patterns: list[str]) -> bool:
    """True for a system-reminder text block that matches one of the drop patterns."""
    if not isinstance(block, dict) or block.get("type") != "text":
        return False
    text = block.get("text", "").lstrip()
    return text.startswith(REMINDER_TAG) and any(re.search(pattern, text) for pattern in patterns)


def trim_messages(messages: list[dict[str, Any]], patterns: list[str]) -> list[dict[str, Any]]:
    """Drop bulky reminder blocks from user messages; a message left empty is kept as it was."""
    shaped = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            shaped.append(message)
            continue
        kept = [block for block in content if not _is_bulky_reminder(block, patterns)]
        shaped.append({**message, "content": kept} if kept else message)
    return shaped


def _shorten_result(block: dict[str, Any], limit: int) -> dict[str, Any]:
    """Cut the text of one tool_result block to `limit` characters and say so, so the model can re-read if needed."""
    content = block.get("content")
    if isinstance(content, str):
        cut = _cut_with_note(content, limit)
        return block if cut == content else {**block, "content": cut}
    if isinstance(content, list):
        parts = [{**part, "text": _cut_with_note(part["text"], limit)} if part.get("type") == "text" else part
                 for part in content]
        return {**block, "content": parts}
    return block


def _cut_with_note(text: str, limit: int) -> str:
    """`text` cut to `limit` characters plus a note with the number of characters removed."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n{TRUNCATION_NOTE.format(removed=len(text) - limit)}"


def prune_old_tool_results(messages: list[dict[str, Any]], keep_recent: int, limit: int) -> list[dict[str, Any]]:
    """Shorten tool results in all but the last `keep_recent` messages (0 limit = off). Old file dumps are the bulk of a long session."""
    if not limit:
        return messages
    cutoff = max(len(messages) - keep_recent, 0)
    shaped = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if index >= cutoff or not isinstance(content, list):
            shaped.append(message)
            continue
        blocks = [_shorten_result(block, limit) if block.get("type") == "tool_result" else block for block in content]
        shaped.append({**message, "content": blocks})
    return shaped


def trim_system_messages(messages: list[dict[str, Any]], headings: list[str]) -> list[dict[str, Any]]:
    """Mid-conversation messages with role 'system' (Claude Code puts environment and MCP texts there): keep only `headings`."""
    shaped = []
    for message in messages:
        text = system_text(message.get("content")) if message.get("role") == "system" else ""
        kept = keep_sections(text.replace("<system-reminder>", "").replace("</system-reminder>", ""), headings)
        shaped.append({**message, "content": kept} if message.get("role") == "system" and kept else message)
    return shaped


def request_chars(data: dict[str, Any]) -> int:
    """Rough size of the prompt parts that matter (about 4 characters per token)."""
    parts = (data.get("system"), data.get("tools"), data.get("messages"))
    return sum(len(json.dumps(part, ensure_ascii=False)) for part in parts if part)


def shape_request(data: dict[str, Any], prompt: str, trim: dict[str, Any]) -> dict[str, Any]:
    """Return a shaped copy of a main-loop request; `trim` is the `trim:` block of orchestrator.yaml."""
    shaped = copy.copy(data)
    shaped["system"] = shape_system(data.get("system"), prompt, trim)
    if not trim.get("enabled", True):
        return shaped
    if data.get("tools"):
        shaped["tools"] = trim_tools(data["tools"], trim.get("tools_keep", []),
                                     trim.get("tool_description_max_chars", 0))
    if data.get("messages"):
        messages = trim_messages(data["messages"], trim.get("drop_message_blocks", []))
        messages = trim_system_messages(messages, trim.get("keep_sections", []))
        shaped["messages"] = prune_old_tool_results(
            messages, trim.get("keep_recent_messages", 8), trim.get("old_tool_result_max_chars", 0))
    return shaped
