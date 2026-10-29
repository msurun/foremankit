#!/usr/bin/env python3
"""Claude Code hook: continue a task from the night run's plan and handoff.

Registered by install.py for Stop. The hook is called when the model is about to stop (the user's prompt is empty and
the model is not in the middle of a tool call or a delegation). The hook decides whether to continue the task from the
night run's plan and handoff. The hook never blocks the call unless the night run is active and the task should continue.
"""

import json
import sys
from pathlib import Path


def read_event() -> dict:
    """The hook event from stdin (a UTF-8 BOM is tolerated); an empty dict when stdin holds no JSON."""
    try:
        return json.loads(sys.stdin.read().lstrip("﻿"))
    except ValueError:
        return {}


def main() -> int:
    """A hook must never break the user's prompt: any unexpected error ends silently with exit code 0."""
    try:
        return run()
    except Exception:  # noqa: BLE001 - hook must never break the session
        return 0


def run() -> int:
    """Read the hook event and decide whether to continue the task from the night run's plan and handoff."""
    event = read_event()
    if not event:
        return 0
    from foremankit.night_run import should_continue, record_continue, read_night
    should, _ = should_continue()
    if should:
        record_continue()
        night = read_night()
        done_file = night.get("done_file")
        if done_file:
            print(json.dumps({"decision": "block",
                              "reason": f"Continue the task from the plan and the latest handoff; when everything is "
                                        f"finished create the DONE file {done_file}."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
