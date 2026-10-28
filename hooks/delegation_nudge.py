#!/usr/bin/env python3
"""Claude Code hook: nudge the model to delegate bulk work, and keep it quiet about delegation when that is switched off.

Registered by install.py for two events:

* PostToolUse (Write | Edit | Bash): counts the lines this session has written itself since its last delegation (Write
  content, Edit replacements, and shell scripts written through a heredoc, which is how most code gets written in
  practice) by reading the transcript incrementally. When the count passes FOREMANKIT_NUDGE_LINES (default 80) it adds a
  reminder to the model's context and a one-line message for the user, then waits for another such amount before it
  repeats. A call to the `delegate` tool or to an agent resets the count.
* UserPromptSubmit: while ALL delegation categories are off (panel switch or `foremankit delegation all off`) it tells
  the model to do the work itself and not to mention delegation at all.

When delegation is off the PostToolUse nudge stays silent. The hook never blocks anything and never fails the session.
"""
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_THRESHOLD = 80                      # lines written without delegating before the first reminder
DELEGATION_TOOLS = {"mcp__foremankit__delegate", "Agent", "Task"}
CATEGORIES = ("paid", "free", "local")      # the delegation categories of foremankit/delegation.py
SHELL_WRITE_MARKERS = ("<<", "cat >", "tee ")   # a shell command that writes a file or a script
MAX_SESSIONS = 50
OFF_NOTICE = (
    "foremankit delegation is switched OFF by the user. Do all the work yourself. Do not call the foremankit delegate tool "
    "or hand work to cheaper helper models, and do not mention, suggest or ask about delegation in your replies (no "
    "'delegating' lines, no helper summaries).")
# Kept to one short line: it lands in the orchestrator's context every ~80 written lines.
NUDGE_CONTEXT = "foremankit: ~{lines} lines written yourself; send tests, docs and new files to bulk-supervisor."
NUDGE_MESSAGE = {
    "en": "foremankit: ~{lines} lines written without delegating (tests, docs and standalone files can go to cheaper models).",
    "pl": "foremankit: ok. {lines} linii napisanych bez delegowania (testy, dokumentacja i osobne pliki mogą iść do tańszych modeli).",
}


def state_dir() -> Path:
    """The foremankit state folder (FOREMANKIT_STATE_DIR in tests)."""
    return Path(os.getenv("FOREMANKIT_STATE_DIR", "~/.config/foremankit")).expanduser()


def load_json(path: Path) -> dict:
    """JSON object from `path`, or {} when it is missing or broken."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json_atomic(path: Path, data: dict) -> None:
    """Write `data` as JSON through a temp file in the same folder, so a concurrent reader never sees half a file."""
    temp_file = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_file.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temp_file, path)


def delegation_off(folder: Path) -> bool:
    """True when every delegation category is switched off in the router state (`toggles` of state.json)."""
    toggles = load_json(folder / "state.json").get("toggles", {})
    return all(toggles.get(f"delegation:{category}") == "off" for category in CATEGORIES)


def lines_of(text: object) -> int:
    """Number of lines in a string (0 for anything that is not text)."""
    return str(text).count("\n") + 1 if isinstance(text, str) and text else 0


def written_lines(tool: str, tool_input: dict) -> int:
    """Lines one tool call wrote: Write content, Edit replacement, or a shell command that writes a file."""
    if tool == "Write":
        return lines_of(tool_input.get("content"))
    if tool == "Edit":
        return lines_of(tool_input.get("new_string"))
    if tool == "Bash":
        command = tool_input.get("command")
        if isinstance(command, str) and any(marker in command for marker in SHELL_WRITE_MARKERS):
            return lines_of(command)
    return 0


def tool_calls(line: str):
    """(tool name, input dict) for every tool_use block of one transcript line."""
    try:
        message = json.loads(line).get("message")
    except ValueError:
        return
    content = message.get("content") if isinstance(message, dict) else None
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
            yield str(block.get("name", "")), block["input"]


def advance(counters: dict, transcript_path: str) -> dict:
    """Read the transcript from the saved offset and update the line counters (a delegation resets `since_delegate`)."""
    try:
        size = os.path.getsize(transcript_path)
        offset = int(counters.get("offset", 0))
        if size < offset:
            offset = 0
        with open(transcript_path, "rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
    except OSError:
        return counters
    complete = chunk[:chunk.rfind(b"\n") + 1]            # a half-written last line is read next time
    since_delegate, since_nudge = int(counters.get("since_delegate", 0)), int(counters.get("since_nudge", 0))
    for line in complete.decode("utf-8", errors="ignore").splitlines():
        for tool, tool_input in tool_calls(line):
            if tool in DELEGATION_TOOLS:
                since_delegate = 0
            written = written_lines(tool, tool_input)
            since_delegate, since_nudge = since_delegate + written, since_nudge + written
    return {"offset": offset + len(complete), "since_delegate": since_delegate, "since_nudge": since_nudge}


def number_env(name: str, default: int) -> int:
    """Positive integer from an environment variable; anything else falls back to `default`."""
    try:
        value = int(os.getenv(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def remember(state: dict, state_file: Path) -> None:
    """Write the per-session counters, keeping only the most recently used sessions."""
    for key, _ in sorted(state.items(), key=lambda item: item[1].get("time", 0))[:-MAX_SESSIONS]:
        del state[key]
    write_json_atomic(state_file, state)


def emit(additional_context: str, event: str, system_message: str = "") -> None:
    """Print the single JSON object a hook may answer with."""
    answer = {"hookSpecificOutput": {"hookEventName": event, "additionalContext": additional_context}}
    if system_message:
        answer["systemMessage"] = system_message
    print(json.dumps(answer))


def nudge(event: dict, folder: Path) -> None:
    """PostToolUse: count, and remind the model (and the user) when too much was written without delegating."""
    session_id, transcript = event.get("session_id"), event.get("transcript_path")
    if not session_id or not transcript:
        return
    state_file = folder / "delegation_nudge.json"
    state = load_json(state_file)
    counters = advance(state.get(session_id, {}), transcript)
    threshold = number_env("FOREMANKIT_NUDGE_LINES", DEFAULT_THRESHOLD)
    due = counters["since_delegate"] >= threshold and counters["since_nudge"] >= threshold
    if due:
        lines = counters["since_delegate"]
        counters["since_nudge"] = 0
    state[session_id] = {**counters, "time": time.time()}
    remember(state, state_file)
    if due:
        language = os.getenv("FOREMANKIT_LANG", "en").strip().lower()
        emit(NUDGE_CONTEXT.format(lines=lines), "PostToolUse", NUDGE_MESSAGE.get(language, NUDGE_MESSAGE["en"]).format(lines=lines))


def run() -> int:
    """Dispatch on the hook event; stay silent whenever there is nothing to say."""
    try:
        event = json.loads(sys.stdin.read().lstrip("﻿"))
    except ValueError:
        return 0
    folder = state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    off = delegation_off(folder)
    if event.get("hook_event_name") == "UserPromptSubmit":
        if off:
            emit(OFF_NOTICE, "UserPromptSubmit")
    elif not off:
        nudge(event, folder)
    return 0


def main() -> int:
    """A hook must never break the user's session: any unexpected error ends silently with exit code 0."""
    try:
        return run()
    except Exception:  # noqa: BLE001 - hook must never break the session
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
