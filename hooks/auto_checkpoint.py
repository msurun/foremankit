#!/usr/bin/env python3
"""Claude Code hook: auto-checkpoint after an foremankit `Agent` tool call finishes.

Registered by install.py for PostToolUse on mcp__foremankit__delegate.
When the delegated task uses kind="bulk_code"/"data_script"/"hard_code" etc.
(i.e. a real agent task, not the quick status probe), this hook writes a compact
summary to `.agent-out/checkpoint-<timestamp>.md` and updates the session state.

The hook never blocks the agent: every code path returns exit code 0.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    """A hook must never break the user's prompt: any unexpected error ends silently with exit code 0."""
    try:
        return run()
    except Exception:  # noqa: BLE001 - hook must never break the session
        return 0


def run() -> int:
    """Hook entry: record a checkpoint after a delegate call and tell the user; returns the exit code."""
    try:
        event = json.loads(sys.stdin.read().lstrip("﻿"))
    except ValueError:
        return 0

    # Only trigger on PostToolUse for the delegate tool
    if not _is_delegate_post_tool_event(event):
        return 0

    kind, task_desc = _task_fields(event)
    # Skip quick/status probes — only checkpoint real agent tasks
    if kind in ("quick",) or not task_desc:
        return 0

    # Determine project root: look for .agent-out in current working dir or parent
    project_root = _find_project_root()
    state_dir = _prepare_state_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    checkpoints = _record_checkpoint(state_dir, kind, task_desc, timestamp)
    if project_root:
        compact = _build_compact(kind, task_desc, _response_text(event), checkpoints)
        checkpoint_file = _write_checkpoint(project_root, timestamp, compact)
        _notify_user(checkpoint_file, kind, task_desc)

    return 0


def _is_delegate_post_tool_event(event: dict) -> bool:
    """True only for a PostToolUse event of the foremankit delegate tool."""
    return (
        event.get("hook_event_name") == "PostToolUse"
        and event.get("tool_name", "") == "mcp__foremankit__delegate"
    )


def _task_fields(event: dict) -> tuple[str, str]:
    """Return (kind, task description) from the delegate call's tool input."""
    tool_input = event.get("tool_input") or {}
    return tool_input.get("kind", ""), tool_input.get("task", "")


def _response_text(event: dict) -> str:
    """Flatten the tool response (dict, list or plain value) into plain text."""
    tool_response = event.get("tool_response") or ""
    if isinstance(tool_response, dict):
        return json.dumps(tool_response, ensure_ascii=False)
    if isinstance(tool_response, list):
        return "\n".join(
            json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
            for item in tool_response
        )
    return str(tool_response)


def _prepare_state_dir() -> Path:
    """Return the state directory (FOREMANKIT_STATE_DIR or ~/.config/foremankit), creating it."""
    state_dir = Path(os.getenv("FOREMANKIT_STATE_DIR", "~/.config/foremankit")).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _record_checkpoint(state_dir: Path, kind: str, task_desc: str, timestamp: str) -> list:
    """Append this task to the session state, save it, and return the full checkpoint list."""
    state = _load_state(state_dir)
    checkpoint_entry = {
        "timestamp": timestamp,
        "kind": kind,
        "task": task_desc[:300],  # cap length
        "status": "completed",
    }
    checkpoints = state.get("checkpoints", [])
    checkpoints.append(checkpoint_entry)
    state["checkpoints"] = checkpoints
    state["last_checkpoint"] = timestamp
    state["last_checkpoint_task"] = task_desc[:200]
    _save_state(state_dir, state)
    return checkpoints


def _write_checkpoint(project_root: Path, timestamp: str, compact: str) -> Path:
    """Write the compact summary into <project root>/.agent-out and return its path."""
    checkpoint_dir = project_root / ".agent-out"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"checkpoint-{timestamp}.md"
    checkpoint_file.write_text(compact, encoding="utf-8")
    return checkpoint_file


def _notify_user(checkpoint_file: Path, kind: str, task_desc: str) -> None:
    """Print the hook JSON that shows the saved checkpoint to the user and the model."""
    print(json.dumps({
        "systemMessage": f"foremankit: auto-checkpoint zapisany → {checkpoint_file.name}",
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"Checkpoint saved to {checkpoint_file.name}. "
                                 f"Task: {kind} — {task_desc[:150]}"
        }
    }))


def _find_project_root() -> Path | None:
    """Find the project root by looking for .agent-out directory going up from cwd."""
    cwd = Path.cwd()
    # Check current dir and up to 5 parents
    for candidate in [cwd] + list(cwd.parents)[:4]:
        if (candidate / ".agent-out").exists():
            return candidate
    return None


def _load_state(state_dir: Path) -> dict:
    """Load session state from disk."""
    state_file = state_dir / "session_checkpoint.json"
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(state_dir: Path, data: dict) -> None:
    """Persist session state to disk through a temp file, so a concurrent reader never sees half a file."""
    state_file = state_dir / "session_checkpoint.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    temp_file = state_file.with_name(f"{state_file.name}.{os.getpid()}.tmp")
    temp_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_file, state_file)


def _build_compact(kind: str, task: str, response: str, checkpoints: list) -> str:
    """Build a compact markdown checkpoint summary (max ~2-3k tokens)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Truncate response to keep compact
    response_preview = response[:500] + ("..." if len(response) > 500 else "")

    # Last 5 checkpoints summary
    recent = checkpoints[-5:] if len(checkpoints) > 5 else checkpoints
    recent_lines = "\n".join(
        f"- [{entry.get('timestamp', '?')}] {entry.get('kind', '?')}: {entry.get('task', '?')[:120]}"
        for entry in recent
    )

    compact = f"""## Kontekst sesji
- Cel: {task[:200]}
- Decyzje: Zadanie delegowane jako `{kind}` — {response_preview[:300]}
- Stan: Ostatni checkpoint {now}
- Ostatnie zadania:
{recent_lines}
"""
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
