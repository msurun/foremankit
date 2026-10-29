"""Records who drives Claude Code's main loop: the local model or a cloud model (last writer wins, no TTL).

The file driver.json in the state dir holds {"local", "model", "time"}. Subagents read it to tell whether the local
orchestrator is in charge, which the cloud-spend rules depend on.
"""
from __future__ import annotations

import json
import time
from typing import Any

from foremankit.config import state_dir

DRIVER_FILE = "driver.json"


def write_driver(local: bool, model: str) -> None:
    """Record whether the local model drives the main loop; an OSError is swallowed so a request never fails."""
    record = {"local": bool(local), "model": model, "time": time.time()}
    try:
        (state_dir() / DRIVER_FILE).write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        pass


def read_driver() -> dict[str, Any] | None:
    """The last recorded driver, or None when the file is missing or invalid."""
    try:
        record = json.loads((state_dir() / DRIVER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def local_drives() -> bool:
    """True only when the last main-loop request was driven by the local model."""
    record = read_driver()
    return bool(record and record.get("local") is True)
