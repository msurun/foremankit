#!/usr/bin/env python3
"""Claude Code hook: show the user a one-line notice before/after every foremankit `delegate` call.

Registered by install.py for PreToolUse and PostToolUse on mcp__foremankit__delegate. The notice goes out as
`systemMessage`, so it reaches the user even if the model forgets the CLAUDE.md rule. It never blocks the call.
"""
import json
import sys


def flatten(value) -> str:
    """Turn a tool response (string, list, dict or other value) into plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(flatten(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "result"):
            if key in value:
                return flatten(value[key])
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def main() -> int:
    """Read the hook event from stdin and print the one-line systemMessage notice."""
    try:
        event = json.loads(sys.stdin.read().lstrip("﻿"))
    except ValueError:
        return 0
    if event.get("hook_event_name") == "PreToolUse":
        args = event.get("tool_input") or {}
        msg = (f"→ calling foremankit: {args.get('kind', 'auto')}, {args.get('sensitivity', 'normal')}"
               f"{', escalate' if args.get('escalate') else ''}")
    else:
        text = flatten(event.get("tool_response")).strip()
        line = next((ln for ln in text.splitlines() if ln.startswith("←")), "")
        msg = line or "← foremankit: " + (text.splitlines()[0][:200] if text else "no output")
    print(json.dumps({"systemMessage": msg}))   # ASCII-escaped: Windows pipes use a legacy codepage
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
