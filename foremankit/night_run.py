"""
Night-run state for unattended local work.

Stores enabled flag, deadline, continue limit and done-file path in night_run.json.
"""
import json
import os
import time
from pathlib import Path
from typing import Any

from foremankit.config import state_dir

from . import driver, utils
from .config import load_config, load_env, refresh_live_env

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

def _check_kaggle_tunnel() -> bool:
    """Verify that the Kaggle upstream Cloudflare tunnel is reachable."""
    url = os.getenv("FOREMANKIT_NODE_KAGGLE_LAN")
    token = os.getenv("KAGGLE_LLM_KEY")
    if not url:
        return False
    endpoint = url if "/v1" in url else f"{url.rstrip('/')}/v1/models"
    return utils.ping_endpoint(endpoint, token, retries=3, backoff=0.3)


def _check_driver(kind: str, allowed: list[str]) -> str | None:
    """Return an error reason ('drives' or 'tunnel') if the driver is not ready, else None."""
    if kind not in allowed:
        return "drives"
    if kind.startswith("external:kaggle"):
        if not _check_kaggle_tunnel():
            return "tunnel"
    elif kind == "local" and not local_drives():
        return "drives"
    return None


def should_continue(now: float | None = None) -> tuple[bool, str]:
    """Return (True, "ok") if the night run may continue, else (False, reason)."""
    # In plain words: decide whether the nightly automation should continue running.
    # The decision checks the active driver, tunnel reachability, done marker, and continue bounds.
    load_env()
    refresh_live_env()
    cfg = load_config()
    state = read_night()
    if not state or not state.get("enabled"):
        return False, "off"
    err = _check_driver(driver.drives_kind() or "local", cfg.get("night_drivers", ["local", "external:kaggle"]))
    if err:
        return False, err
    if (done := str(state.get("done_file", ""))) and Path(done).exists():
        return False, "done"
    if int(state.get("count", 0)) >= int(state.get("max_continues", 0)):
        return False, "limit"
    if state.get("context_stop"):
        return False, "context_stop"
    current = time.time() if now is None else now
    if current >= float(state.get("until", 0.0)):
        return False, "expired"
    return True, "ok"



