"""Pure functions that translate one preset choice into the settings to change.

The VS Code panel asks a single question, "who orchestrates Claude Code", and
this module decides the mode, the local server role and whether VS Code goes
direct to Anthropic. No file, network or process access happens here.
"""

from __future__ import annotations

MODE_CLAUDE = "claude"
MODE_LOCAL = "local"

ROLE_ORCHESTRATOR = "orchestrator"
ROLE_WORKER = "worker"
ROLE_BOTH = "both"

# Order matters: "direct" goes last because it makes the VS Code window reload.
STEP_ORDER = ("mode", "role", "direct")


def preset_names(upstreams: dict) -> list[str]:
    """Return the panel choices: the two built-ins first, then each upstream name."""
    return [MODE_CLAUDE, MODE_LOCAL, *upstreams]


def preset_plan(name: str, current: dict) -> dict:
    """Return the steps that turn `current` state into the state of preset `name`.

    Returns a dict with keys "steps" (settings that differ, in STEP_ORDER),
    "reload" (True when a "direct" step exists), "needs_server" and "summary".
    An unknown `name` is treated as an upstream provider, so it is not an error.
    """
    wanted = _wanted_settings(name, current)
    steps = _steps_to_reach(wanted, current)
    return {
        "steps": steps,
        "reload": any(step["action"] == "direct" for step in steps),
        "needs_server": name == MODE_LOCAL and not current["server_up"],
        "summary": _summarise(steps),
    }


def current_preset(current: dict, upstreams: dict) -> str:
    """Return the preset name that matches `current` state, or "custom" if none does."""
    mode = current["mode"]
    role = current["role"]
    if current["direct"]:
        return MODE_CLAUDE if mode == MODE_CLAUDE else "custom"
    if mode == MODE_LOCAL:
        return MODE_LOCAL if role in (ROLE_ORCHESTRATOR, ROLE_BOTH) else "custom"
    return mode if mode in upstreams else "custom"


def issues(state: dict) -> list[dict]:
    """Return the configuration problems in `state`, in a fixed priority order.

    `state` is `current` plus "llama_installed" and "key_missing". Each issue is
    {"id", "text", "fix"} where "fix" is a panel action string, or "" if the
    user must act outside the panel (for example, editing .env).
    """
    found = [
        _router_stopped_issue(state),
        _router_needed_issue(state),
        _llama_missing_issue(state),
        _server_down_issue(state),
        _key_missing_issue(state),
    ]
    return [issue for issue in found if issue is not None]


def _wanted_settings(name: str, current: dict) -> dict:
    """Return {"mode", "role", "direct"} that preset `name` asks for."""
    if name == MODE_CLAUDE:
        return {"mode": MODE_CLAUDE, "role": ROLE_WORKER, "direct": True}
    if name == MODE_LOCAL:
        return {"mode": MODE_LOCAL, "role": _local_role(current["role"]), "direct": False}
    # Any other name is an upstream provider: it only changes the mode.
    return {"mode": name, "role": current["role"], "direct": False}


def _local_role(current_role: str) -> str:
    """Keep the orchestrator role when it is already set, otherwise serve both."""
    return ROLE_ORCHESTRATOR if current_role == ROLE_ORCHESTRATOR else ROLE_BOTH


def _steps_to_reach(wanted: dict, current: dict) -> list[dict]:
    """Return one step per setting that differs, ordered by STEP_ORDER."""
    steps = []
    for action in STEP_ORDER:
        if wanted[action] != current[action]:
            steps.append({"action": action, "value": wanted[action]})
    return steps


def _summarise(steps: list[dict]) -> list[str]:
    """Return one short English sentence per step, or "Already active"."""
    if not steps:
        return ["Already active"]
    return [_describe(step) for step in steps]


def _describe(step: dict) -> str:
    """Return the English sentence for a single step."""
    if step["action"] == "mode":
        return f"Switch the orchestrator to {step['value']}"
    if step["action"] == "role":
        return f"Set the local server role to {step['value']}"
    if step["value"]:
        return "Go straight to Anthropic, router off (VS Code window reloads)"
    return "Turn the router on (VS Code window reloads)"


def _router_stopped_issue(state: dict) -> dict | None:
    """Return the issue where Claude mode is on but the router is stopped."""
    is_stuck = not state["direct"] and not state["router_up"] and state["mode"] == MODE_CLAUDE
    if not is_stuck:
        return None
    return {
        "id": "router_stopped",
        "text": "Router mode is on but the router is stopped: "
        "Claude Code cannot reach Anthropic.",
        "fix": f"apply-preset {MODE_CLAUDE}",
    }


def _router_needed_issue(state: dict) -> dict | None:
    """Return the issue where a non-Claude mode cannot reach the router."""
    needs_router = state["mode"] != MODE_CLAUDE and (state["direct"] or not state["router_up"])
    if not needs_router:
        return None
    return {
        "id": "router_needed",
        "text": "This orchestrator reaches its model only through the router, "
        "which is not in use.",
        "fix": f"apply-preset {state['mode']}",
    }


def _llama_missing_issue(state: dict) -> dict | None:
    """Return the issue where local mode is selected but llama-server is absent."""
    if state["mode"] != MODE_LOCAL or state["llama_installed"]:
        return None
    return {
        "id": "llama_missing",
        "text": "The local model server is not installed.",
        "fix": "download-llama",
    }


def _server_down_issue(state: dict) -> dict | None:
    """Return the issue where local mode has llama-server installed but not running."""
    is_down = state["mode"] == MODE_LOCAL and state["llama_installed"] and not state["server_up"]
    if not is_down:
        return None
    return {
        "id": "server_down",
        "text": "The local model server is not running.",
        "fix": "start-server",
    }


def _key_missing_issue(state: dict) -> dict | None:
    """Return the issue where an upstream provider has no API key."""
    is_builtin = state["mode"] in (MODE_CLAUDE, MODE_LOCAL)
    if is_builtin or not state["key_missing"]:
        return None
    return {
        "id": "key_missing",
        "text": "The API key of this provider is not set (add it to .env).",
        "fix": "",
    }
