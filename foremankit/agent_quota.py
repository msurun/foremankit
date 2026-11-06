"""Whether an `agent-run` runner (agy / opencode) may start now.
Decides the parallel slots, the daily run limit and the cooldown after a quota error; the only I/O is load/save of one
JSON file whose path the caller passes in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agy_provider import RATE_LIMIT_TEXT
from .state import utc_day

DEFAULT_MAX_PARALLEL = {"opencode": 3, "agy": 1}
DEFAULT_SLOTS = 1
DEFAULT_COOLDOWN_S = 1800.0
NO_DAILY_LIMIT = 0


def slots(settings: dict, runner: str) -> int:
    """How many runs of `runner` may run in parallel (per-runner defaults when not configured)."""
    configured = settings.get("max_parallel") or {}
    value = configured.get(runner, DEFAULT_MAX_PARALLEL.get(runner, DEFAULT_SLOTS))
    return int(value)


def is_quota_error(text: str | None) -> bool:
    """Whether `text` looks like a rate limit or an exhausted quota (empty/None is not an error)."""
    return bool(text) and RATE_LIMIT_TEXT.search(text) is not None


def load_state(path: Path) -> dict:
    """Load the quota state, returning {} when the file is missing or corrupt."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(path: Path, state: dict) -> None:
    """Write the quota state as JSON, creating the parent folder when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")


def record_run(state: dict, runner: str, now: float) -> None:
    """Count a started run for the UTC day of `now`."""
    entry = _bucket(state, runner, now)
    entry["runs"] = int(entry.get("runs", 0)) + 1


def record_quota_error(state: dict, runner: str, now: float, settings: dict) -> float:
    """Pause `runner` after a quota error; returns the timestamp it may start again at."""
    entry = _bucket(state, runner, now)
    quota = _quota_settings(settings, runner)
    cooldown = float(quota.get("cooldown_on_error_s", DEFAULT_COOLDOWN_S))
    until = max(float(entry.get("paused_until", 0.0)), now + cooldown)
    entry["paused_until"] = until
    return until


def can_start(state: dict, runner: str, now: float, settings: dict) -> tuple[bool, str]:
    """(True, "") when `runner` may start now, else (False, reason); the pause is checked before the daily limit."""
    entry = _bucket(state, runner, now)
    paused_until = float(entry.get("paused_until", 0.0))
    if now < paused_until:
        return False, f"{runner} paused for {paused_until - now:.0f}s after a quota error"
    limit = int(_quota_settings(settings, runner).get("max_runs_per_day", NO_DAILY_LIMIT) or 0)
    if limit > 0 and int(entry.get("runs", 0)) >= limit:
        return False, f"{runner} reached its daily limit of {limit} runs"
    return True, ""


def _bucket(state: dict, runner: str, now: float) -> dict[str, Any]:
    """The runner's counters, reset on a new UTC day (the pause survives the reset)."""
    entry = state.setdefault(runner, {})
    if entry.get("day") != utc_day(now):
        entry.update(day=utc_day(now), runs=0)
    entry.setdefault("runs", 0)
    entry.setdefault("paused_until", 0.0)
    return entry


def _quota_settings(settings: dict, runner: str) -> dict[str, Any]:
    """The `quota.<runner>` settings block (empty when missing); never mutates `settings`."""
    return (settings.get("quota") or {}).get(runner) or {}
