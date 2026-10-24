"""Escalation policy for the gateway's main-loop orchestrator.

The gateway runs on a cheap model and, after repeated failing turns, escalates to an
expensive model. It only returns to the cheap model after a clean streak AND a minimum
time on the expensive model, so a single flaky turn cannot flap the tier back and forth.
All functions here are pure: no clock, no I/O, no mutation of the inputs.
"""

from __future__ import annotations

from typing import Any

CHEAP_TIER = "cheap"
EXPENSIVE_TIER = "expensive"

DEFAULT_ERRORS_TO_ESCALATE = 2
DEFAULT_CLEAN_TO_FALL_BACK = 6
DEFAULT_MIN_EXPENSIVE_S = 120.0


def last_tool_error(messages: list[dict]) -> bool:
    """Return True when the last tool_result block in the history has is_error truthy.

    A string content, a message without content, or a history with no tool_result at
    all is treated as no error (False).
    """
    last_tool_result: dict | None = None
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        last_tool_result = _last_tool_result_in(content, last_tool_result)
    if last_tool_result is None:
        return False
    return bool(last_tool_result.get("is_error"))


def _last_tool_result_in(
    content: list[Any], current: dict | None
) -> dict | None:
    """Return the last tool_result block of `content`, or `current` if there is none."""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            current = block
    return current


def default_state() -> dict:
    """Return a fresh escalation state: cheap tier, no streaks, never escalated."""
    return {"tier": CHEAP_TIER, "errors": 0, "clean": 0, "escalated_at": 0.0}


def next_state(state: dict, had_error: bool, policy: dict, now: float) -> dict:
    """Return a new state after one turn, applying streaks and hysteresis.

    `now` is a monotonic timestamp in seconds. The input `state` is never mutated.
    Policy keys `errors`, `clean` and `min_expensive_s` fall back to their defaults.
    """
    errors_to_escalate = policy.get("errors", DEFAULT_ERRORS_TO_ESCALATE)
    clean_to_fall_back = policy.get("clean", DEFAULT_CLEAN_TO_FALL_BACK)
    min_expensive_s = policy.get("min_expensive_s", DEFAULT_MIN_EXPENSIVE_S)

    updated = _with_updated_streaks(state, had_error)
    if _should_escalate(updated, had_error, errors_to_escalate):
        return _escalate(now)
    if _should_fall_back(updated, clean_to_fall_back, min_expensive_s, now):
        return _fall_back(updated)
    return updated


def _with_updated_streaks(state: dict, had_error: bool) -> dict:
    """Return a copy of `state` with the error/clean streak counters updated."""
    updated = dict(state)
    if had_error:
        updated["errors"] = state["errors"] + 1
        updated["clean"] = 0
    else:
        updated["clean"] = state["clean"] + 1
        updated["errors"] = 0
    return updated


def _should_escalate(state: dict, had_error: bool, threshold: int) -> bool:
    """Return True when the cheap tier has failed enough turns in a row."""
    return state["tier"] == CHEAP_TIER and had_error and state["errors"] >= threshold


def _should_fall_back(
    state: dict, clean_threshold: int, min_expensive_s: float, now: float
) -> bool:
    """Return True when the expensive tier has been clean long enough to step down."""
    if state["tier"] != EXPENSIVE_TIER:
        return False
    return state["clean"] >= clean_threshold and now - state["escalated_at"] >= min_expensive_s


def _escalate(now: float) -> dict:
    """Return a state on the expensive tier, streaks reset, escalation time recorded."""
    return {"tier": EXPENSIVE_TIER, "errors": 0, "clean": 0, "escalated_at": now}


def _fall_back(state: dict) -> dict:
    """Return a state back on the cheap tier, clean streak reset, escalation time kept."""
    return {
        "tier": CHEAP_TIER,
        "errors": 0,
        "clean": 0,
        "escalated_at": state["escalated_at"],
    }
