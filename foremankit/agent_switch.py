"""On/off switches for agent runs: one master switch plus one switch per runner.

Toggles live in the state file under state["toggles"] with the keys
"agent_run:all" (master), "agent_run:agy", "agent_run:opencode" and "agent_run:codex".
A missing key means "on": agent runs were always available, so switching
must not change the current behaviour until the owner turns something off.
"""
from __future__ import annotations

from .state import StateStore

RUNNERS = ("agy", "opencode", "codex")
TARGETS = ("all", *RUNNERS)

_MASTER = "all"
_VALUE_ON = "on"
_VALUE_OFF = "off"

MASTER_OFF_MESSAGE = ("AGENT RUNS ARE SWITCHED OFF by the user (ForemanKit panel). "
                      "Do this task another way; do not look for a way around it.")


def _key(name: str) -> str:
    """State toggle key for a runner name or the master switch ("all")."""
    return f"agent_run:{name}"


def _switch_on(state: dict, name: str) -> bool:
    """True unless the toggle for this runner (or the master) is explicitly off."""
    return state.get("toggles", {}).get(_key(name)) != _VALUE_OFF


def is_enabled(state: dict, runner: str | None = None) -> bool:
    """True when the master switch is on and, if given, the runner's own switch is on.

    Raises:
        ValueError: If runner is not one of RUNNERS.
    """
    if runner is None:
        return _switch_on(state, _MASTER)
    if runner not in RUNNERS:
        raise ValueError(f"Invalid runner: {runner!r}. Must be one of {RUNNERS}.")
    return _switch_on(state, _MASTER) and _switch_on(state, runner)


def disabled_runners(state: dict) -> frozenset[str]:
    """Runners that may not run: all of them when the master is off, else those switched off."""
    if not _switch_on(state, _MASTER):
        return frozenset(RUNNERS)
    return frozenset(runner for runner in RUNNERS if not _switch_on(state, runner))


def set_switch(store: StateStore, target: str, value: str) -> list[str]:
    """Turn the master switch or a single runner on or off and persist it.

    "all" changes only the master key, so the runner switches keep their own value.

    Returns:
        The list of changed target names (one element).

    Raises:
        ValueError: If target is not in TARGETS or value is not "on"/"off".
    """
    if target not in TARGETS:
        raise ValueError(f"Invalid target: {target!r}. Must be one of {TARGETS}.")
    if value not in (_VALUE_ON, _VALUE_OFF):
        raise ValueError(f"Invalid value: {value!r}. Must be 'on' or 'off'.")

    def _apply_change(state: dict) -> list[str]:
        state.setdefault("toggles", {})[_key(target)] = value
        return [target]

    return store.update(_apply_change)


def refusal(state: dict, runner: str) -> str:
    """Why this runner may not run, or "" when it may.

    Raises:
        ValueError: If runner is not one of RUNNERS.
    """
    if is_enabled(state, runner):
        return ""
    if not _switch_on(state, _MASTER):
        return MASTER_OFF_MESSAGE
    return f"{runner} is switched off by the user (ForemanKit panel); use the other runner or do the task another way."


def describe(state: dict) -> str:
    """Describe the master switch and each runner's own switch, one line per switch."""
    status = _VALUE_ON if _switch_on(state, _MASTER) else _VALUE_OFF
    lines = [f"agent runs: {status}"]
    for runner in RUNNERS:
        enabled = _VALUE_ON if _switch_on(state, runner) else _VALUE_OFF
        lines.append(f"{runner}: {enabled}")
    return "\n".join(lines)
