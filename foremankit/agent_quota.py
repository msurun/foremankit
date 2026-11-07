"""Whether an `agent-run` runner (agy / opencode / codex) may start now.
Decides the parallel slots, the daily run limit and the cooldown after a quota error; the only I/O is load/save of one
JSON file whose path the caller passes in.
"""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .agy_provider import RATE_LIMIT_TEXT
from .state import utc_day

DEFAULT_MAX_PARALLEL = {"opencode": 3, "agy": 1, "codex": 1}
DEFAULT_SLOTS = 1
DEFAULT_COOLDOWN_S = 1800.0
DEFAULT_BACKOFF_S = [1800, 3600, 7200, 18000]
PROBE_WINDOW_S = 1800
QUOTA_TEXT = re.compile(
    RATE_LIMIT_TEXT.pattern + r"|usage limit|hit your limit|limit reached|reached your limit|"
    r"out of credits|insufficient credits|try again later|try again in", re.IGNORECASE,
)
NO_DAILY_LIMIT = 0


@dataclass(frozen=True)
class LimitHit:
    """One runner quota refusal, with its Unix timestamp in seconds and optional reset text."""

    runner: str
    now: float
    message: str = ""


def state_path() -> Path:
    """The quota state file shared by `agent-run-batch` and the panel (temp folder, next to the run worktrees)."""
    return Path(tempfile.gettempdir()) / "foremankit-agent-run" / "quota-state.json"


def slots(settings: dict, runner: str) -> int:
    """Return parallel capacity, clamping negative values to zero (runner unavailable)."""
    configured = settings.get("max_parallel") or {}
    value = configured.get(runner, DEFAULT_MAX_PARALLEL.get(runner, DEFAULT_SLOTS))
    return max(0, int(value))


def is_quota_error(text: str | None) -> bool:
    """Whether `text` looks like a rate limit or an exhausted quota (empty/None is not an error)."""
    return bool(text) and QUOTA_TEXT.search(text) is not None


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
    if entry["strikes"] > 0:
        entry["probe_until"] = now + PROBE_WINDOW_S


def record_quota_error(state: dict, hit: LimitHit, settings: dict) -> float:
    """Pause `runner` after a quota error; returns the timestamp it may start again at."""
    runner, now, message = hit.runner, hit.now, hit.message
    entry = _bucket(state, runner, now)
    quota = _quota_settings(settings, runner)
    steps = quota.get("backoff_s") or [quota.get("cooldown_on_error_s", DEFAULT_COOLDOWN_S), *DEFAULT_BACKOFF_S[1:]]
    cooldown = float(steps[min(entry["strikes"], len(steps) - 1)])
    reset = parse_reset(message, now)
    proposed = min(max(reset, now + 60), now + 86400) if reset is not None else now + cooldown
    until = max(float(entry.get("paused_until", 0.0)), proposed)
    entry["strikes"] += 1
    entry["paused_until"] = until
    entry["probe_until"] = 0.0      # a failed probe is over: after the pause the next run is a fresh probe
    return until


def can_start(state: dict, runner: str, now: float, settings: dict) -> tuple[bool, str]:
    """(True, "") when `runner` may start now, else (False, reason); the pause is checked before the daily limit."""
    entry = _bucket(state, runner, now)
    paused_until = float(entry.get("paused_until", 0.0))
    if now < paused_until:
        return False, f"{runner} paused for {paused_until - now:.0f}s after a quota error"
    if entry["strikes"] > 0 and entry["probe_until"] > now:
        return False, f"{runner} is being probed after a limit pause"
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
    entry.setdefault("strikes", 0)
    entry.setdefault("probe_until", 0.0)
    return entry


def _quota_settings(settings: dict, runner: str) -> dict[str, Any]:
    """The `quota.<runner>` settings block (empty when missing); never mutates `settings`."""
    return (settings.get("quota") or {}).get(runner) or {}


def record_success(state: dict, runner: str) -> None:
    """Mark a runner healthy after a successful run, retaining its expired pause."""
    state.setdefault(runner, {}).update(strikes=0, probe_until=0.0)


def pause_info(state: dict, runner: str, now: float) -> dict:
    """Describe the temporary pause and the recovery probe phase."""
    entry = _bucket(state, runner, now)
    until = float(entry["paused_until"])
    return {"paused": now < until, "until": until, "remaining_s": int(max(0, until - now)),
            "strikes": entry["strikes"], "probing": now >= until and entry["strikes"] > 0}


def clear_pause(state: dict, runner: str) -> None:
    """Allow an immediate recovery probe without forgetting quota strikes."""
    state.setdefault(runner, {})["paused_until"] = 0.0


def parse_reset(message: str, now: float) -> float | None:
    """Parse relative durations or the next local clock time; malformed messages return None."""
    try:
        relative = re.search(r"(?:\bin\s+|retry\s+after\s+)((?:\d+\s*(?:hours?|h|minutes?|m|seconds?|s)\b\s*)+)",
                             message, re.IGNORECASE)
        if relative:
            parts = re.findall(r"(\d+)\s*([hms])", relative[1], re.IGNORECASE)
            return now + sum(int(value) * {"h": 3600, "m": 60, "s": 1}[unit.lower()] for value, unit in parts)
        header = re.search(r"retry-after\s*:\s*(\d+)\b", message, re.IGNORECASE)
        if header:
            return now + int(header[1])
        return _clock_reset(message, now)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _clock_reset(message: str, now: float) -> float | None:
    """Resolve an 'at' clock time using the machine's local timezone and next calendar day."""
    clock = re.search(r"\bat\s+(\d{1,2}):(\d{2})\s*(AM|PM)?\b", message, re.IGNORECASE)
    if not clock:
        return None
    hour, minute = int(clock[1]), int(clock[2])
    if clock[3]:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if clock[3].lower() == "pm" else 0)
    local = datetime.fromtimestamp(now)
    reset = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset.timestamp() <= now:
        reset += timedelta(days=1)
    return reset.timestamp()
