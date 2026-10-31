"""
Night-run state for unattended local work.

Stores enabled flag, deadline, continue limit and done-file path in night_run.json.
"""
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import load_config
from foremankit.config import state_dir
from . import driver
from . import utils

local_drives = driver.local_drives

NIGHT_FILE = "night_run.json"


def _state_path() -> Path:
    """Return the absolute path of the night-run state file."""
    return state_dir() / NIGHT_FILE


def read_night() -> dict[str, Any] | None:
    """Return the parsed state dict, or None if the file is missing or invalid."""
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_night(state: dict[str, Any]) -> None:
    """Persist the state dict, ignoring OS errors."""
    try:
        _state_path().write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def start_night(until: float, max_continues: int, done_file: str) -> None:
    """Enable a night run with a fresh continue counter."""
    _write_night({
        "enabled": True,
        "until": float(until),
        "max_continues": int(max_continues),
        "count": 0,
        "done_file": str(done_file),
    })


def stop_night() -> None:
    """Disable the night run, keeping the other keys when present."""
    state = read_night() or {"until": 0.0, "max_continues": 0, "count": 0, "done_file": ""}
    state["enabled"] = False
    _write_night(state)


def record_continue() -> int:
    """Increment the continue counter and return the new value."""
    state = read_night() or {}
    try:
        count = int(state.get("count", 0)) + 1
    except (TypeError, ValueError):
        count = 1
    state["count"] = count
    _write_night(state)
    return count

def should_continue(now: float | None = None) -> tuple[bool, str]:
    """Return (True, "ok") if the night run may continue, else (False, reason)."""
    # In plain words: decide whether the nightly automation should keep running.
    cfg = load_config()
    state = read_night()
    if not state or not state.get("enabled"):
        return False, "off"
    kind = driver.drives_kind() or "local"
    allowed = cfg.get("night_drivers", ["local", "external:kaggle"])
    if kind not in allowed:
        return False, "off"
    if kind.startswith("external:kaggle"):
        url = os.getenv("FOREMANKIT_NODE_KAGGLE_LAN")
        token = os.getenv("KAGGLE_LLM_KEY")
        if not url or not utils.ping_endpoint(url, token, retries=3, backoff=0.3):
            return False, "tunnel"
    if not local_drives():
        return False, "drives"
    done = str(state.get("done_file", ""))
    if done and Path(done).exists():
        return False, "done"
    if int(state.get("count", 0)) >= int(state.get("max_continues", 0)):
        return False, "limit"
    if state.get("context_stop"):
        return False, "context_stop"
    current = time.time() if now is None else now
    if current >= float(state.get("until", 0.0)):
        return False, "expired"
    return True, "ok"



