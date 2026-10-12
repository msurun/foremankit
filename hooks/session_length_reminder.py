#!/usr/bin/env python3
"""Claude Code hook: remind the user once when the conversation grows too large.

Registered by install.py for UserPromptSubmit. The reminder goes out as `systemMessage`, so it reaches the user
even if the model forgets the CLAUDE.md rule. It never blocks the call. Two independent triggers, one message per call:

1. Transcript size (all models): `transcript_path` bigger than FOREMANKIT_SESSION_REMINDER_MB (8 MB), then at most every
   90 minutes and after 4 MB of growth.
2. Context fill (local orchestrator only): the gateway writes the size of the last prompt it sent to the local model
   and that model's window to `gateway_context.json` (state dir). While that file is fresh the hook warns once per
   level at 60 %, 75 % and 85 % of the window. The size of a transcript file is no usable token count (JSONL
   overhead, tool results the gateway cuts), so cloud sessions get no token levels.
"""
import json
import os
import sys
import time
from pathlib import Path

LEVELS = (0.60, 0.75, 0.85)       # fractions of the context window
GATEWAY_FRESH_S = 10 * 60         # older gateway data belongs to a finished local session
# User-facing texts per language: English by default, Polish with FOREMANKIT_LANG=pl.
LEVEL_TEXT = {
    "en": ("ℹ foremankit: local model context is {pct:.0%} full (~{tokens:,} / {window:,} tokens). "
           "Save the state when the task ends.",
           "⚠ foremankit: local model context is {pct:.0%} full (~{tokens:,} / {window:,} tokens). "
           "At the next breakpoint save the state (handoff) and start a new session.",
           "⚠ foremankit: local model context is almost full, {pct:.0%} (~{tokens:,} / {window:,} tokens). "
           "Save the state now and start a new session, otherwise the server rejects the request (HTTP 400)."),
    "pl": ("ℹ foremankit: kontekst lokalnego modelu zapełniony w {pct:.0%} (~{tokens:,} / {window:,} tokenów). "
           "Przy końcu zadania zapisz stan.",
           "⚠ foremankit: kontekst lokalnego modelu zapełniony w {pct:.0%} (~{tokens:,} / {window:,} tokenów). "
           "Przy najbliższym punkcie przerwania zapisz stan (handoff) i zacznij nową sesję.",
           "⚠ foremankit: kontekst lokalnego modelu prawie pełny, {pct:.0%} (~{tokens:,} / {window:,} tokenów). "
           "Zapisz stan teraz i zacznij nową sesję, inaczej serwer odrzuci żądanie (HTTP 400)."),
}
LONG_TEXT = {
    "en": "foremankit: this conversation is already long (transcript about {mb:.0f} MB) and every turn re-reads it. "
          "At the next natural breakpoint consider a new session (save the state to memory or the repo).",
    "pl": "foremankit: ta rozmowa jest już długa (transkrypcja ok. {mb:.0f} MB), a każda tura czyta ją od nowa. "
          "Przy najbliższym naturalnym punkcie przerwania rozważ nową sesję (stan zapisz w pamięci lub repozytorium).",
}


def ui_language() -> str:
    """Language of the messages shown to the user: FOREMANKIT_LANG (en or pl), English otherwise."""
    language = os.environ.get("FOREMANKIT_LANG", "en").strip().lower()
    return language if language in LEVEL_TEXT else "en"


def number_env(name: str, default: float, positive: bool = False) -> float:
    """Float from an environment variable; unset, empty or invalid values fall back to `default`."""
    try:
        value = float(os.getenv(name, ""))
    except ValueError:
        return default
    return default if positive and value <= 0 else value


def main() -> int:
    """A hook must never break the user's prompt: any unexpected error ends silently with exit code 0."""
    try:
        return run()
    except Exception:
        return 0


def load_json(path: Path) -> dict:
    """JSON object from `path`, or {} when the file is missing or broken."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def context_level(state_dir: Path, now: float) -> tuple[int, int, int, float]:
    """(level 0-3, tokens, window, fraction) from fresh gateway data; level 0 when there is none."""
    usage = load_json(state_dir / "gateway_context.json")
    try:
        tokens, window = int(usage["tokens"]), int(usage["window"])
        fresh = now - float(usage["time"]) <= GATEWAY_FRESH_S
    except (KeyError, TypeError, ValueError):
        return 0, 0, 0, 0.0
    if not fresh or window <= 0:
        return 0, tokens, window, 0.0
    fraction = tokens / window
    return sum(fraction >= level for level in LEVELS), tokens, window, fraction


def emit(system_message: str, additional_context: str) -> None:
    """Print the single JSON object a UserPromptSubmit hook may answer with."""
    print(json.dumps({"systemMessage": system_message,
                      "hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                             "additionalContext": additional_context}}))


def size_due(session_state: dict, size_mb: float, now: float) -> bool:
    """True when the transcript-size reminder should go out for this session now."""
    if "mb" not in session_state:
        return True
    return now - float(session_state.get("time", 0)) >= 90 * 60 and size_mb - float(session_state.get("mb", 0)) >= 4


def remember(state: dict, state_file: Path) -> None:
    """Write the state, keeping only the 50 most recently reminded sessions."""
    if len(state) > 50:
        for key, _ in sorted(state.items(), key=lambda item: item[1].get("time", 0))[:-50]:
            del state[key]
    state_file.write_text(json.dumps(state), encoding="utf-8")


def run() -> int:
    """Read the hook event and print at most one reminder."""
    try:
        event = json.loads(sys.stdin.read().lstrip("﻿"))
    except ValueError:
        return 0
    session_id, transcript_path = event.get("session_id"), event.get("transcript_path")
    if not session_id or not transcript_path:
        return 0
    state_dir = Path(os.getenv("FOREMANKIT_STATE_DIR", "~/.config/foremankit")).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "session_reminder.json"
    state = load_json(state_file)
    now = number_env("FOREMANKIT_FAKE_NOW", time.time())
    session_state = state.get(session_id)
    session_state = session_state if isinstance(session_state, dict) else {}

    level, tokens, window, fraction = context_level(state_dir, now)
    if level > int(session_state.get("level", 0)):
        state[session_id] = {**session_state, "level": level}
        remember(state, state_file)
        text = LEVEL_TEXT[ui_language()][level - 1].format(pct=fraction, tokens=tokens, window=window)
        emit(text, f"The local model's context is {fraction:.0%} full (~{tokens:,} of {window:,} tokens). At the next "
                   f"natural breakpoint save the state to a handoff file and suggest a NEW session in ONE line.")
        return 0

    try:
        size_mb = os.path.getsize(transcript_path) / (1024 * 1024)
    except OSError:
        return 0
    threshold = number_env("FOREMANKIT_SESSION_REMINDER_MB", 8.0, positive=True)
    if size_mb >= threshold and size_due(session_state, size_mb, now):
        state[session_id] = {**session_state, "time": now, "mb": size_mb}
        remember(state, state_file)
        emit(LONG_TEXT[ui_language()].format(mb=size_mb),
             f"The conversation transcript is about {size_mb:.0f} MB and every turn re-reads it. At the next natural "
             f"breakpoint (a task finished and committed, results delivered) suggest in ONE line continuing in a NEW "
             f"session after saving the state to memory or the repo. Never interrupt a task in progress; do not "
             f"repeat this suggestion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
