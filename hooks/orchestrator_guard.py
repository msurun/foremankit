#!/usr/bin/env python3
"""Claude Code PreToolUse hook: keep files above the upstream ceiling away from external orchestrators.

When the active orchestrator mode is an external API (Gemini, DeepSeek, ...) that API provider sees every file the model
reads. This hook denies a tool call whose paths belong to a repository whose sensitivity level is above the upstream's
ceiling, and denies paths matching `deny_globs`. Builtin modes (claude, local) and errors in evaluation fail closed:
nothing reaches the provider unnoticed. It never blocks a session and never writes anything but a deny decision.
"""
import fnmatch
import json
import sys
from pathlib import Path

from foremankit import upstream
from foremankit.config import _read_yaml, config_dir

CONFIG_NAME = "orchestrator.yaml"

# Tool input keys that carry a filesystem path, in the order we check them.
PATH_KEYS = ("file_path", "path", "notebook_path")

# Tools whose input is a shell command rather than a path.
SHELL_TOOLS = ("Bash", "PowerShell")

# Globs shorter than this after removing '*' are too generic to search for in shell commands.
MIN_STEM_LENGTH = 4

DENY_REASON = (
    "orchestrator_guard blocked this tool call: sensitivity level '{level}' of '{path}' "
    "exceeds the '{ceiling}' ceiling of upstream mode '{mode}'."
)

UNKNOWN_MODE_REASON = (
    "orchestrator_guard blocked this tool call: upstream mode '{mode}' is not declared "
    "in {config_name}."
)

ERROR_REASON = "orchestrator_guard error: {error_type}"


def load_config() -> dict:
    """Return the orchestrator configuration, or {} when the file is missing or broken."""
    try:
        config = _read_yaml(config_dir() / CONFIG_NAME)
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def is_declared_mode(upstreams: dict, mode: str) -> bool:
    """True when `mode` names an upstream declared in the configuration."""
    return isinstance(upstreams.get(mode), dict)


def ceiling_of(upstreams: dict, mode: str) -> str:
    """Sensitivity ceiling declared for a known `mode`; defaults to the strictest level."""
    return upstreams[mode].get("ceiling", "normal")


def candidate_paths(event: dict, cwd: Path) -> list[Path]:
    """The cwd plus every path-valued tool input, resolved against `cwd`."""
    tool_input = event.get("tool_input") or {}
    paths = [cwd]
    for key in PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            paths.append((cwd / value).resolve())
    return paths


def sensitivity_of(path: Path) -> str:
    """Sensitivity level of `path`: its own directory when it is one, else its parent directory."""
    start = path if path.is_dir() else path.parent
    return upstream.repo_sensitivity(start)


def matches_glob(path: Path, globs: list) -> bool:
    """True when `path` matches one of `globs` (forward slashes, case-insensitive)."""
    normalized = str(path).replace("\\", "/").lower()
    return any(fnmatch.fnmatch(normalized, str(glob).lower()) for glob in globs)


def glob_stems(globs: list) -> list[str]:
    """Literals to search for in shell commands: globs without '*' and long enough to be meaningful."""
    stems = (str(glob).replace("*", "") for glob in globs)
    return [stem.lower() for stem in stems if len(stem) >= MIN_STEM_LENGTH]


def shell_touches_denied(command: str, globs: list) -> bool:
    """True when a shell command contains the literal stem of a deny glob."""
    lowered = command.lower()
    return any(stem in lowered for stem in glob_stems(globs))


def deny_reason(mode: str, ceiling: str, path: Path, level: str) -> str:
    """The single-sentence English reason shown to the model for a deny decision."""
    return DENY_REASON.format(mode=mode, ceiling=ceiling, level=level, path=path)


def check_paths(mode: str, ceiling: str, paths: list[Path], globs: list) -> str | None:
    """Deny reason for the first path above the ceiling or matching a deny glob, else None."""
    for path in paths:
        if matches_glob(path, globs):
            return deny_reason(mode, ceiling, path, "denied")
        level = sensitivity_of(path)
        if not upstream.allowed(ceiling, level):
            return deny_reason(mode, ceiling, path, level)
    return None


def check_event(event: dict, cfg: dict) -> str | None:
    """Deny reason for the tool call in `event`, or None when it may proceed."""
    mode = upstream.active_mode()
    upstreams = cfg.get("upstreams", {})
    if not is_declared_mode(upstreams, mode):
        return UNKNOWN_MODE_REASON.format(mode=mode, config_name=CONFIG_NAME)
    ceiling = ceiling_of(upstreams, mode)
    globs = cfg.get("deny_globs", [])
    cwd = Path(event.get("cwd") or Path.cwd())
    tool_name = str(event.get("tool_name", ""))
    if tool_name in SHELL_TOOLS:
        command = str((event.get("tool_input") or {}).get("command", ""))
        if shell_touches_denied(command, globs):
            return deny_reason(mode, ceiling, cwd, "denied")
        return check_paths(mode, ceiling, [cwd], globs)
    return check_paths(mode, ceiling, candidate_paths(event, cwd), globs)


def emit_deny(reason: str) -> None:
    """Print the PreToolUse deny decision as JSON."""
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(output))


def read_event() -> dict:
    """Parse the hook event from stdin; raises json.JSONDecodeError on invalid input."""
    return json.loads(sys.stdin.read().lstrip("\ufeff"))


def main() -> int:
    """Read the hook event; emit a deny decision in a non-builtin mode when policy or an error demands it."""
    try:
        mode = upstream.active_mode()
    except Exception as error:  # noqa: BLE001 - fail closed when the mode cannot be read
        emit_deny(ERROR_REASON.format(error_type=type(error).__name__))
        return 0
    if mode in upstream.BUILTIN_MODES:
        return 0
    try:
        event = read_event()
        reason = check_event(event, load_config())
    except Exception as error:  # noqa: BLE001 - fail closed in a non-builtin mode
        emit_deny(ERROR_REASON.format(error_type=type(error).__name__))
        return 0
    if reason is not None:
        emit_deny(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
