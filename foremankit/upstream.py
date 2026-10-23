"""Persistence and policy checks for the orchestrator upstream mode.

The active mode lives in a small file inside the state directory so that every
command (wrapper, CLI, hooks) reads the same value. Upstream names, API keys and
sensitivity ceilings come from `orchestrator.yaml`; the per-repository
sensitivity level is read from a file inside the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from foremankit.config import state_dir

# File inside the state directory that stores the currently selected mode.
MODE_FILE = "orchestrator_mode"

# Modes handled directly by the orchestrator, no upstream config required.
BUILTIN_MODES = ("claude", "local")

# Sensitivity levels in ascending order: a ceiling allows every level up to it.
LEVELS = ("normal", "internal", "private")

# File, relative to a repository root, that declares its sensitivity level.
SENSITIVITY_FILE = Path(".foremankit") / "sensitivity"

# Mode assumed when the mode file is missing, empty or unreadable.
DEFAULT_MODE = "claude"

# Model alias assumed when the requested model matches no known family.
DEFAULT_ALIAS = "sonnet"

# Sensitivity assumed when no repository declares a level (fail closed).
DEFAULT_SENSITIVITY = "private"


def active_mode() -> str:
    """Return the selected mode, falling back to the default when unavailable."""
    path = state_dir() / MODE_FILE
    if not path.exists():
        return DEFAULT_MODE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_MODE
    mode = text.strip()
    return mode or DEFAULT_MODE


def set_mode(name: str) -> None:
    """Persist the selected mode atomically so readers never see a partial file."""
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MODE_FILE
    # Write next to the target, then rename: os.replace is atomic on one filesystem.
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(name, encoding="utf-8")
    os.replace(temporary, target)


def alias_of(model: str) -> str:
    """Map a model identifier to one of the aliases "opus", "haiku" or "sonnet"."""
    lowered = model.lower()
    for alias in ("opus", "haiku", "sonnet"):
        if alias in lowered:
            return alias
    return DEFAULT_ALIAS


def upstream_model(upstream: dict, model: str) -> str:
    """Return the upstream model name configured for the alias of `model`."""
    models = upstream["models"]
    alias = alias_of(model)
    if alias in models:
        return models[alias]
    return models["sonnet"]


def repo_sensitivity(start: Path) -> str:
    """Return the sensitivity level declared by the closest parent, default "private"."""
    for directory in (start, *start.parents):
        marker = directory / SENSITIVITY_FILE
        if not marker.is_file():
            continue
        level = marker.read_text(encoding="utf-8").strip().lower()
        # The first marker found wins, even when it is invalid: fail closed.
        return level if level in LEVELS else DEFAULT_SENSITIVITY
    return DEFAULT_SENSITIVITY


def allowed(ceiling: str, level: str) -> bool:
    """Return True when a repository `level` is at or below the upstream `ceiling`."""
    if ceiling not in LEVELS or level not in LEVELS:
        return False
    return LEVELS.index(level) <= LEVELS.index(ceiling)


def check_switch(cfg: dict, mode: str, cwd: Path) -> str | None:
    """Return an error message when switching to `mode` is not permitted, else None."""
    if mode in BUILTIN_MODES:
        return None
    upstreams = cfg.get("upstreams", {})
    if mode not in upstreams:
        valid = ", ".join((*BUILTIN_MODES, *upstreams))
        return f"Unknown mode '{mode}'. Valid modes: {valid}."
    upstream = upstreams[mode]
    key_error = _missing_key_error(mode, upstream)
    if key_error is not None:
        return key_error
    ceiling = upstream.get("ceiling", "normal")
    level = repo_sensitivity(cwd)
    if allowed(ceiling, level):
        return None
    return (
        f"Upstream '{mode}' allows sensitivity up to '{ceiling}' "
        f"but this repository is '{level}'."
    )


def _missing_key_error(mode: str, upstream: dict) -> str | None:
    """Return an error message when the upstream's API key is not set, else None."""
    key_env = upstream["key_env"]
    if os.environ.get(key_env):
        return None
    return f"Upstream '{mode}' requires the environment variable {key_env} to be set."
