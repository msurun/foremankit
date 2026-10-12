"""JSON state file (toggles, quota counters, cooldowns, spend) with atomic writes."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import state_dir

_LOCK = threading.Lock()


def utc_day(ts: float) -> str:
    """Convert a Unix timestamp to a UTC date string (YYYY-MM-DD)."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def utc_month(ts: float) -> str:
    """Convert a Unix timestamp to a UTC month string (YYYY-MM)."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


class StateStore:
    """Read-modify-write on every change so CLI toggles are seen by a running server."""

    def __init__(self, path: Path | None = None):
        self.path = path or state_dir() / "state.json"

    def load(self) -> dict[str, Any]:
        """Load the state file, returning an empty dict if missing or invalid."""
        try:
            with self.path.open(encoding="utf-8") as file_handle:
                data = json.load(file_handle)
                return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        """Atomically save data to the state file using a temporary file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f".{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file_handle:
            json.dump(data, file_handle, indent=1)
        os.replace(tmp_path, self.path)

    def update(self, fn: Callable[[dict[str, Any]], Any]) -> Any:
        """Atomically read, modify, and write the state using a provided function."""
        with _LOCK:
            data = self.load()
            result = fn(data)
            self._save(data)
            return result


class CallLog:
    """Append-only JSONL with call metadata (never prompt bodies unless enabled)."""

    def __init__(self, path: Path | None = None):
        self.path = path or state_dir() / "calls.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        """Append a single JSON record to the log file with a timestamp."""
        record = {"ts": round(time.time(), 3), **record}
        with _LOCK:
            self._ensure_parent_directory_exists()
            self._append_json_line(record)

    def read(self) -> list[dict[str, Any]]:
        """Read all valid JSON lines from the log file."""
        if not self.path.is_file():
            return []
        return self._parse_valid_json_lines()

    def _ensure_parent_directory_exists(self) -> None:
        """Create the parent directory for the log file if it does not exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append_json_line(self, record: dict[str, Any]) -> None:
        """Append a JSON-serialized record to the log file."""
        with self.path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _parse_valid_json_lines(self) -> list[dict[str, Any]]:
        """Parse all non-empty lines in the log file into a list of dicts."""
        with self.path.open(encoding="utf-8") as file_handle:
            return self._collect_valid_json_records(file_handle)

    def _collect_valid_json_records(self, file_handle: Any) -> list[dict[str, Any]]:
        """Collect valid JSON records from a file handle, skipping invalid lines."""
        rows = []
        for line in file_handle:
            stripped_line = line.strip()
            if stripped_line:
                record = self._parse_single_line(stripped_line)
                if record is not None:
                    rows.append(record)
        return rows

    def _parse_single_line(self, line: str) -> dict[str, Any] | None:
        """Parse a single line into a dict, returning None if invalid."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
