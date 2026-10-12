"""Builds the prompt of a task: reads the context files (with data guards) and arranges the chat messages."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import conventions
from .outputs import relative_path
from .records import RouterError, TaskRequest

BINARY_SNIFF_BYTES = 8192
DEFAULT_MAX_DATA_LINES = 60          # lines of a data file an external (free) model may receive
DEFAULT_MAX_TOTAL_CHARS = 400000
DEFAULT_CHARS_PER_TOKEN = 3.5


def read_context_files(routing: dict[str, Any], names: list[str], trusted: bool) -> list[tuple[str, str]]:
    """(relative path, text) of every context file. trusted = sensitivity internal/private: the data-file line cap
    (meant for external free models) is lifted."""
    guard = routing.get("context_guard") or {}
    max_total = int(guard.get("max_total_chars", DEFAULT_MAX_TOTAL_CHARS))
    files, total = [], 0
    for name in names:
        path, text = _read_text_file(name)
        if not trusted:
            _refuse_big_data_file(guard, name, path, text)
        total += len(text)
        if total > max_total:
            raise RouterError(f"context too large (> {max_total} chars); split the task")
        files.append((relative_path(path), text))
    return files


def _read_text_file(name: str) -> tuple[Path, str]:
    path = Path(name)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise RouterError(f"context file not found: {name}")
    raw = path.read_bytes()
    if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
        raise RouterError(f"binary file refused: {name} (send a text summary/schema instead)")
    return path, raw.decode("utf-8", errors="replace")


def _refuse_big_data_file(guard: dict[str, Any], name: str, path: Path, text: str) -> None:
    """Whole datasets never go to an external model: only a schema and a few anonymised rows."""
    data_extensions = {extension.lower() for extension in guard.get("data_extensions", [])}
    max_lines = int(guard.get("max_data_lines_non_private", DEFAULT_MAX_DATA_LINES))
    lines = text.count("\n")
    if path.suffix.lower() in data_extensions and lines > max_lines:
        raise RouterError(f"{name}: {lines} lines of data refused for an external model; "
                          f"send schema + <= {max_lines} anonymised rows, or use sensitivity='internal'/'private'.")


def build_messages(routing: dict[str, Any], request: TaskRequest, files: list[tuple[str, str]]) -> list[dict[str, str]]:
    """System prompt (with the coding conventions for code kinds) and one user message: files, notes, task."""
    prompts = routing.get("system_prompts") or {}
    system = (prompts.get(request.kind) or prompts.get("_default") or "").strip()
    if conventions.conventions_apply(routing, request.kind):    # the same coding standard in every session and tool
        system = f"{system}\n\n{conventions.load_text(routing, Path.cwd())}".strip()
    parts = [f"### File: {name}\n```\n{text}\n```" for name, text in files]
    if request.system_hint:
        parts.append(f"### Notes\n{request.system_hint.strip()}")
    parts.append(f"### Task\n{request.task.strip()}")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def estimate_input_tokens(routing: dict[str, Any], messages: list[dict[str, str]]) -> int:
    """Rough token count of the prompt from its length (routing.yaml budget.chars_per_token)."""
    chars = sum(len(message["content"]) for message in messages)
    chars_per_token = float((routing.get("budget") or {}).get("chars_per_token", DEFAULT_CHARS_PER_TOKEN))
    return int(chars / chars_per_token) + 1
