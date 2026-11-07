"""Data behind the panel's "Agent runs" section: switches, limits, recent runs, reviews and usage.

`build_status` is pure: it turns the given state, settings, quota counters, log rows and usage
reading into the dict the VS Code panel shows. The wrappers load those inputs from the real
state folder and files; agent runs cost nothing, so no cost field appears here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import agent_quota, agent_review_log, agent_switch, claude_usage
from .config import _read_yaml, config_dir
from .state import CallLog, StateStore, state_dir, utc_day

RECENT_RUNS = 8
LABELS = {"agy": "Antigravity", "opencode": "opencode", "codex": "Codex"}
AGY_DEFAULT_MODEL = "flash_high"


@dataclass(frozen=True)
class StatusInputs:
    """Everything build_status needs, already loaded by the caller."""

    toggles_state: dict
    settings: dict
    quota_state: dict
    rows: list[dict]
    usage: dict
    now: float


def build_status(inputs: StatusInputs) -> dict:
    """Build the panel's agent-runs status from the given inputs without touching any file."""
    master = agent_switch.is_enabled(inputs.toggles_state)
    runners = {runner: _runner_entry(inputs, runner, master) for runner in agent_switch.RUNNERS}
    reviews = agent_review_log.summarize_reviews(inputs.rows)
    return {"enabled": master,
            "runners": runners,
            "recent": _recent(inputs.rows),
            "reviews": reviews,
            "review_text": agent_review_log.format_reviews(reviews),
            "usage": inputs.usage}


def _runner_entry(inputs: StatusInputs, runner: str, master: bool) -> dict:
    """One runner's switches, parallel slots, counters, pause and latest model."""
    own = _own_switch(inputs.toggles_state, runner)
    quota = inputs.quota_state.get(runner) or {}
    pause = agent_quota.pause_info({runner: dict(quota)}, runner, inputs.now)
    limit, hard = _daily_limit(inputs.settings, runner)
    return {"label": LABELS[runner],
            "own": own,
            "active": master and own,
            "slots": agent_quota.slots(inputs.settings, runner),
            "runs_today": _runs_today(quota, inputs.now),
            "daily_limit": limit, "daily_limit_hard": hard,
            "paused_for_s": pause["remaining_s"],
            "strikes": int(pause["strikes"]), "probing": bool(pause["probing"]),
            "retry_at": pause["until"] if pause["paused"] else 0,
            "internal_data": _internal_data(inputs.settings, runner),
            "model": _latest_model(inputs.rows, runner, inputs.settings)}


def _internal_data(settings: dict, runner: str) -> bool:
    """Whether the runner's configured sensitivity ceiling permits internal data."""
    return runner == "agy" or (runner == "codex" and
                               (settings.get("codex") or {}).get("max_sensitivity", "normal") == "internal")


def _own_switch(toggles_state: dict, runner: str) -> bool:
    """The runner's own switch, independent of the master switch."""
    state = {"toggles": {**(toggles_state.get("toggles") or {}), "agent_run:all": "on"}}
    return agent_switch.is_enabled(state, runner)


def _runs_today(quota: dict, now: float) -> int:
    """The run counter, but only while it belongs to today's UTC day."""
    return int(quota.get("runs") or 0) if quota.get("day") == utc_day(now) else 0


def _daily_limit(settings: dict, runner: str) -> tuple[int, bool]:
    """The positive hard limit, else the positive soft limit, else no limit."""
    quota = (settings.get("quota") or {}).get(runner) or {}
    hard = int(quota.get("max_runs_per_day") or 0)
    return (hard, True) if hard > 0 else (max(0, int(quota.get("soft_runs_per_day") or 0)), False)


def _latest_model(rows: list[dict], runner: str, settings: dict) -> str:
    """The newest model logged for a runner, else the runner's configured default."""
    seen = next((row.get("model") for row in reversed(rows)
                 if row.get("provider") == runner and row.get("model")), None)
    if seen:
        return str(seen)
    if runner in ("opencode", "codex"):
        return str((settings.get(runner) or {}).get("default_model") or "")
    return AGY_DEFAULT_MODEL


def _recent(rows: list[dict]) -> list[dict]:
    """The last agent-run outcomes, newest first, at most RECENT_RUNS of them."""
    outcomes = [row for row in rows if row.get("event") == "outcome" and row.get("kind") == "agent_run"]
    indexed = list(enumerate(outcomes[-RECENT_RUNS:]))
    indexed.sort(key=lambda item: (float(item[1].get("ts") or 0.0), item[0]), reverse=True)
    return [_recent_entry(row) for _, row in indexed]


def _recent_entry(row: dict) -> dict:
    """One recent run as the panel shows it."""
    seconds = row.get("seconds")
    return {"ts": float(row.get("ts") or 0.0),
            "runner": str(row.get("provider") or ""),
            "model": str(row.get("model") or ""),
            "passed": bool(row.get("accepted")),
            "seconds": None if seconds is None else float(seconds),
            "id": str(row.get("id") or "")}


def agent_runs_status(now: float | None = None) -> dict:
    """Load the real state, settings, quota counters, log rows and usage, then build the status."""
    now = time.time() if now is None else now
    inputs = StatusInputs(toggles_state=StateStore().load(),
                          settings=_agent_run_settings(),
                          quota_state=agent_quota.load_state(agent_quota.state_path()),
                          rows=CallLog().read(),
                          usage=claude_usage.report(usage_path(), now),
                          now=now)
    return build_status(inputs)


def _agent_run_settings() -> dict:
    """The `agent_run` block of routing.yaml (an empty dict when it is missing)."""
    return _read_yaml(config_dir() / "routing.yaml").get("agent_run") or {}


def set_agent_run(target: str, enabled: bool) -> str:
    """Turn the master switch or one runner on or off and report the result."""
    try:
        agent_switch.set_switch(StateStore(), target, "on" if enabled else "off")
    except ValueError as exc:
        return f"not done: {exc}"
    return f"agent runs {'on' if enabled else 'off'}: {target}"


def resume_runner(runner: str, now: float | None = None) -> str:
    """Clear a known runner's quota pause early, retaining strikes for the recovery probe."""
    if runner not in agent_switch.RUNNERS:
        return f"not done: unknown runner: {runner}"
    path = agent_quota.state_path()
    state = agent_quota.load_state(path)
    agent_quota.clear_pause(state, runner)
    agent_quota.save_state(path, state)
    return f"pause cleared: {runner} (the next run is a probe)"


def refresh_usage(now: float | None = None) -> str:
    """Fetch a fresh Claude usage reading and report the session and week percentages."""
    now = time.time() if now is None else now
    try:
        result = claude_usage.refresh(usage_path(), now)
    except claude_usage.UsageError as exc:
        return f"usage not read: {exc}"
    if not result.get("ok"):
        return f"usage not read: {result.get('error') or 'unknown error'}"
    reading = result.get("last") or {}
    return f"usage: session {_pct(reading, 'session_pct')}%, week {_pct(reading, 'week_pct')}%"


def _pct(reading: dict, key: str) -> Any:
    """A percentage from a reading, with `?` for a missing window."""
    value = reading.get(key)
    return "?" if value is None else value


def mark_usage(now: float | None = None) -> str:
    """Copy the latest usage reading into the baseline and report the result."""
    now = time.time() if now is None else now
    try:
        claude_usage.mark_baseline(usage_path(), now)
    except ValueError as exc:
        return f"not done: {exc}"
    return "baseline marked"


def usage_path() -> Path:
    """The cached Claude usage file in the foremankit state folder."""
    return state_dir() / "usage.json"
