#!/usr/bin/env python3
"""Claude Code hook: remind the user once when the conversation grows too large.

Registered by install.py for UserPromptSubmit. The reminder goes out as `systemMessage`, so it reaches the user
even if the model forgets the CLAUDE.md rule. It never blocks the call. Two independent triggers, one message per call:

1. Transcript size (all models): `transcript_path` bigger than FOREMANKIT_SESSION_REMINDER_MB (8 MB), then at most every
   90 minutes and after 4 MB of growth.
2. Context fill (sessions that run on the local model only): the gateway writes the size of the last prompt it sent to
   the local model and that model's window to `gateway_context.json` (state dir). The file is shared by every session, so
   the hook first checks that THIS session talks to the local model (its last answer was not produced by a `claude-*`
   model, see `uses_local_model`); a cloud session never gets the local model's fill level. While the file is fresh the hook
   warns once per level at 60 % and 75 %; from 85 % it orders the model to stop its work and write a handoff, on every
   prompt until the session is replaced. The size of a transcript file is no usable token count (JSONL overhead, tool
   results the gateway cuts).
3. Context fill (cloud sessions): the last `claude-*` answer in the transcript carries a `usage` record; its input +
   cache-read + cache-creation tokens ARE the context of that turn. One reminder per level (70 %, 85 % of the window),
   never a stop order (Claude Code compacts by itself). The level resets after a /compact, so the reminder can fire
   again. While such a record exists it replaces trigger 1, which would keep warning after a compaction.
"""
import json
import os
import sys
import time
from pathlib import Path

LEVELS = (0.60, 0.75, 0.85)       # fractions of the context window; the last one means: stop and hand over
TAIL_BYTES = 256 * 1024           # how much of the transcript's end is searched for the model of the last answer
GATEWAY_FRESH_S = 10 * 60         # older gateway data belongs to a finished local session
CLOUD_LEVELS = (0.70, 0.85)       # fractions of the cloud window: heads-up, then "hand over at the next breakpoint"
CLOUD_WINDOW = 200_000            # tokens; a "[1m]" model id or an observed larger context means the 1M window
CLOUD_WINDOW_BIG = 1_000_000
USAGE_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
CLOUD_TEXT = {
    "en": ("ℹ foremankit: context is {pct:.0%} full (~{tokens:,} / {window:,} tokens). Plan a /compact or a new "
           "session at the next breakpoint.",
           "⚠ foremankit: context is {pct:.0%} full (~{tokens:,} / {window:,} tokens). At the next breakpoint save "
           "the state (memory / repo) and start a new session or /compact."),
    "pl": ("ℹ foremankit: kontekst zapełniony w {pct:.0%} (~{tokens:,} / {window:,} tokenów). Zaplanuj /compact lub "
           "nową sesję przy najbliższym punkcie przerwania.",
           "⚠ foremankit: kontekst zapełniony w {pct:.0%} (~{tokens:,} / {window:,} tokenów). Przy najbliższym punkcie "
           "przerwania zapisz stan (pamięć / repo) i zacznij nową sesję lub zrób /compact."),
}
# User-facing texts per language: English by default, Polish with FOREMANKIT_LANG=pl.
LEVEL_TEXT = {
    "en": (("ℹ foremankit: local model context is {pct:.0%} full (~{tokens:,} / {window:,} tokens). "
            "Save the state when the task ends."),
           ("⚠ foremankit: local model context is {pct:.0%} full (~{tokens:,} / {window:,} tokens). "
            "At the next breakpoint save the state (handoff) and start a new session."),
           ("⛔ foremankit: local model context is {pct:.0%} full (~{tokens:,} / {window:,} tokens). "
            "The model was told to stop and write the handoff; start a new session after it.")),
    "pl": (("ℹ foremankit: kontekst lokalnego modelu zapełniony w {pct:.0%} (~{tokens:,} / {window:,} tokenów). "
            "Przy końcu zadania zapisz stan."),
           ("⚠ foremankit: kontekst lokalnego modelu zapełniony w {pct:.0%} (~{tokens:,} / {window:,} tokenów). "
            "Przy najbliższym punkcie przerwania zapisz stan (handoff) i zacznij nową sesję."),
           ("⛔ foremankit: kontekst lokalnego modelu zapełniony w {pct:.0%} (~{tokens:,} / {window:,} tokenów). "
            "Model dostał polecenie zatrzymania pracy i zapisania handoffu; potem zacznij nową sesję.")),
}
LONG_TEXT = {
    "en": "foremankit: this conversation is already long (transcript about {mb:.0f} MB) and every turn re-reads it. "
          "At the next natural breakpoint consider a new session (save the state to memory or the repo).",
    "pl": "foremankit: ta rozmowa jest już długa (transkrypcja ok. {mb:.0f} MB), a każda tura czyta ją od nowa. "
          "Przy najbliższym naturalnym punkcie przerwania rozważ nową sesję (stan zapisz w pamięci lub repozytorium).",
}


STOP_INSTRUCTION = (
    "STOP. The local model's context is {pct:.0%} full (~{tokens:,} of {window:,} tokens); the next requests will be "
    "rejected. Do not start any new step, tool call or delegation, and do not finish the current task. Do exactly this now: "
    "(1) write a handoff file HANDOFF.md in the project root (or save it to memory) with what is done, which files changed, "
    "what is still pending, decisions taken and the exact next step; (2) answer in at most three lines: the path of the "
    "handoff and the sentence that the user should continue in a NEW session. Nothing else.")


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
    except Exception:  # noqa: BLE001 - hook must never break the session
        return 0


def load_json(path: Path) -> dict:
    """JSON object from `path`, or {} when the file is missing or broken."""
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


def uses_local_model(transcript_path: str) -> bool:
    """True when the last answer in this session's transcript came from a local model.

    A transcript line carries `message.model`: `claude-...` for the cloud, the llama-server alias (`qwen3.6-...`,
    `spark-...`) for the local model. Only the tail of the file is read; `<synthetic>` entries are ignored. A session
    without an answer yet is not counted as local, so a brand-new session is never told about another session's fill."""
    try:
        with open(transcript_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return False
    for line in reversed(tail.splitlines()):
        try:
            message = json.loads(line).get("message")
        except ValueError:
            continue
        model = message.get("model") if isinstance(message, dict) else None
        if isinstance(model, str) and model != "<synthetic>":
            return not model.startswith("claude-")
    return False


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
    write_json_atomic(state_file, state)


def read_event() -> dict:
    """The hook event from stdin (a UTF-8 BOM is tolerated); an empty dict when stdin holds no JSON."""
    try:
        return json.loads(sys.stdin.read().lstrip("﻿"))
    except ValueError:
        return {}


def session_entry(state: dict, session_id: str) -> dict:
    """This session's entry in the reminder state; an empty dict for a new or broken entry."""
    entry = state.get(session_id)
    return entry if isinstance(entry, dict) else {}


def remind_context(state: dict, state_file: Path, session_id: str, fill: tuple) -> bool:
    """Warn about the local model's context fill (level, tokens, window, fraction); True when this call printed a reminder."""
    level, tokens, window, fraction = fill
    session_state = session_entry(state, session_id)
    if level == len(LEVELS):
        emit(LEVEL_TEXT[ui_language()][level - 1].format(pct=fraction, tokens=tokens, window=window),
             STOP_INSTRUCTION.format(pct=fraction, tokens=tokens, window=window))
        return True
    if level <= int(session_state.get("level", 0)):
        return False
    state[session_id] = {**session_state, "level": level}
    remember(state, state_file)
    text = LEVEL_TEXT[ui_language()][level - 1].format(pct=fraction, tokens=tokens, window=window)
    emit(text, f"The local model's context is {fraction:.0%} full (~{tokens:,} of {window:,} tokens). At the next "
               f"natural breakpoint save the state to a handoff file and suggest a NEW session in ONE line.")
    return True


def tail_lines(transcript_path: str) -> list[str]:
    """Lines from the last TAIL_BYTES of the transcript, newest last; empty when the file cannot be read."""
    try:
        with open(transcript_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - TAIL_BYTES))
            return handle.read().decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return []


def line_message(line: str) -> dict:
    """The `message` object of one transcript line; {} when the line is cut off, not JSON or has no message."""
    try:
        message = json.loads(line).get("message")
    except (ValueError, AttributeError):
        return {}
    return message if isinstance(message, dict) else {}


def cloud_window(model: str, tokens: int) -> int:
    """Context window of a cloud model: FOREMANKIT_CLOUD_WINDOW, else 1M for a "[1m]" id or a larger observed context."""
    configured = int(number_env("FOREMANKIT_CLOUD_WINDOW", 0.0))
    if configured > 0:
        return configured
    return CLOUD_WINDOW_BIG if "[1m]" in model or tokens > CLOUD_WINDOW else CLOUD_WINDOW


def cloud_context(transcript_path: str) -> tuple[int, int] | None:
    """(context tokens, window) from the newest `claude-*` answer with a usage record; None when there is none."""
    for line in reversed(tail_lines(transcript_path)):
        message = line_message(line)
        usage, model = message.get("usage"), message.get("model")
        if isinstance(usage, dict) and isinstance(model, str) and model.startswith("claude-"):
            tokens = sum(int(usage.get(key) or 0) for key in USAGE_KEYS)
            return tokens, cloud_window(model, tokens)
    return None


def remind_cloud(state: dict, state_file: Path, session_id: str, usage: tuple[int, int]) -> None:
    """Remind once per cloud context level; a lower level (after /compact) is stored silently so it can fire again."""
    tokens, window = usage
    fraction = tokens / window
    level = sum(fraction >= value for value in CLOUD_LEVELS)
    session_state = session_entry(state, session_id)
    previous = int(session_state.get("cloud_level", 0))
    if level == previous:
        return
    state[session_id] = {**session_state, "cloud_level": level}
    remember(state, state_file)
    if level > previous:
        text = CLOUD_TEXT[ui_language()][level - 1].format(pct=fraction, tokens=tokens, window=window)
        emit(text, f"The context is {fraction:.0%} full (~{tokens:,} of {window:,} tokens). At the next natural "
                   f"breakpoint suggest in ONE line a /compact or a NEW session after saving the state to memory "
                   f"or the repo. Never interrupt a task in progress; do not repeat this suggestion.")


def size_reminder(session_state: dict, transcript_path: str, now: float) -> dict:
    """The session entry to store when the transcript-size reminder is due now; an empty dict when it is not."""
    try:
        size_mb = os.path.getsize(transcript_path) / (1024 * 1024)
    except OSError:
        return {}
    threshold = number_env("FOREMANKIT_SESSION_REMINDER_MB", 8.0, positive=True)
    if size_mb < threshold or not size_due(session_state, size_mb, now):
        return {}
    return {**session_state, "time": now, "mb": size_mb}


def emit_size_reminder(size_mb: float) -> None:
    """Print the note that the conversation transcript has grown large (the model suggests a new session at a breakpoint)."""
    emit(LONG_TEXT[ui_language()].format(mb=size_mb),
         f"The conversation transcript is about {size_mb:.0f} MB and every turn re-reads it. At the next natural "
         f"breakpoint (a task finished and committed, results delivered) suggest in ONE line continuing in a NEW "
         f"session after saving the state to memory or the repo. Never interrupt a task in progress; do not "
         f"repeat this suggestion.")


def remind_by_context(state: dict, state_dir: Path, event: dict, now: float) -> bool:
    """Local-model fill or cloud context level; True when the context logic handled this call (the MB proxy is skipped)."""
    state_file, session_id, transcript_path = state_dir / "session_reminder.json", event["session_id"], event["transcript_path"]
    if uses_local_model(transcript_path):
        return remind_context(state, state_file, session_id, context_level(state_dir, now))
    usage = cloud_context(transcript_path)
    if usage:
        remind_cloud(state, state_file, session_id, usage)
    return bool(usage)


def run() -> int:
    """Read the hook event and print at most one reminder."""
    event = read_event()
    session_id, transcript_path = event.get("session_id"), event.get("transcript_path")
    if not session_id or not transcript_path:
        return 0
    state_dir = Path(os.getenv("FOREMANKIT_STATE_DIR", "~/.config/foremankit")).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "session_reminder.json"
    state = load_json(state_file)
    now = number_env("FOREMANKIT_FAKE_NOW", time.time())
    if remind_by_context(state, state_dir, event, now):
        return 0
    updated = size_reminder(session_entry(state, session_id), transcript_path, now)
    if updated:
        state[session_id] = updated
        remember(state, state_file)
        emit_size_reminder(updated["mb"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
