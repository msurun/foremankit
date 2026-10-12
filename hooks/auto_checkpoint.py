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
    try:
        event = json.loads(sys.stdin.read().lstrip("﻿"))
    except ValueError:
        return 0

    # Only trigger on PostToolUse for the delegate tool
    if event.get("hook_event_name") != "PostToolUse":
        return 0

    tool_name = event.get("tool_name", "")
    if tool_name != "mcp__foremankit__delegate":
        return 0

    # Parse tool input to get task context
    tool_input = event.get("tool_input") or {}
    kind = tool_input.get("kind", "")
    task_desc = tool_input.get("task", "")

    # Skip quick/status probes — only checkpoint real agent tasks
    if kind in ("quick",) or not task_desc:
        return 0

    # Parse tool response
    tool_response = event.get("tool_response") or ""
    if isinstance(tool_response, dict):
        response_text = json.dumps(tool_response, ensure_ascii=False)
    elif isinstance(tool_response, list):
        response_text = "\n".join(
            json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
            for item in tool_response
        )
    else:
        response_text = str(tool_response)

    # Determine project root: look for .agent-out in current working dir or parent
    project_root = _find_project_root()

    # State dir
    state_dir = Path(os.getenv("FOREMANKIT_STATE_DIR", "~/.config/foremankit")).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_dir)

    # Build checkpoint content
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    checkpoint_entry = {
        "timestamp": timestamp,
        "kind": kind,
        "task": task_desc[:300],  # cap length
        "status": "completed",
    }

    # Update state
    checkpoints = state.get("checkpoints", [])
    checkpoints.append(checkpoint_entry)
    state["checkpoints"] = checkpoints
    state["last_checkpoint"] = timestamp
    state["last_checkpoint_task"] = task_desc[:200]
    _save_state(state_dir, state)

    # Write the checkpoint file into <project root>/.agent-out
    if project_root:
        checkpoint_dir = project_root / ".agent-out"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"checkpoint-{timestamp}.md"

        compact = _build_compact(kind, task_desc, response_text, checkpoints)
        checkpoint_file.write_text(compact, encoding="utf-8")

        # Notify user
        print(json.dumps({
            "systemMessage": f"foremankit: auto-checkpoint zapisany → {checkpoint_file.name}",
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"Checkpoint saved to {checkpoint_file.name}. "
                                     f"Task: {kind} — {task_desc[:150]}"
            }
        }))

    return 0


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
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (ValueError, FileNotFoundError):
        return {}


def _save_state(state_dir: Path, data: dict) -> None:
    """Persist session state to disk."""
    state_file = state_dir / "session_checkpoint.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_compact(kind: str, task: str, response: str, checkpoints: list) -> str:
    """Build a compact markdown checkpoint summary (max ~2-3k tokens)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Truncate response to keep compact
    response_preview = response[:500] + ("..." if len(response) > 500 else "")

    # Last 5 checkpoints summary
    recent = checkpoints[-5:] if len(checkpoints) > 5 else checkpoints
    recent_lines = "\n".join(
        f"- [{c.get('timestamp', '?')}] {c.get('kind', '?')}: {c.get('task', '?')[:120]}"
        for c in recent
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
