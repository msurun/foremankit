"""Tests for the guard hooks in ``install.py``: ``_apply_guards`` and the "guards" group of
``_apply_hooks``. They check the shape of the PreToolUse entries written into settings.json and
that foreign entries survive both install and removal. Same import by path as test_install.py,
because install.py is not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location("foremankit_install", _REPO_ROOT / "install.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load install.py from {_REPO_ROOT}")
INSTALL: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(INSTALL)

PY = Path("C:/py/python.exe")
PROMPT_EVENTS = ("UserPromptSubmit", "Stop", "SessionStart")


def guard_matchers() -> dict[str, str]:
    """Return script name -> matcher for the three guard scripts, straight from install.py."""
    return {}

def pre_tool_use(hooks: dict) -> list[dict]:
    """Return the PreToolUse entries of `hooks`, or [] when the key is absent."""
    return hooks.get("PreToolUse", [])


def commands_of(entries: list[dict]) -> list[str]:
    """Return every hook command of `entries`, in order."""
    return [hook["command"] for entry in entries for hook in entry["hooks"]]


def matcher_of(entries: list[dict], script_name: str) -> str | None:
    """Return the matcher of the entry whose command runs `script_name`, or None."""
    for entry in entries:
        if any(script_name in command for command in commands_of([entry])):
            return entry.get("matcher")
    return None


FOREIGN_ENTRY = {"matcher": "mcp__other", "hooks": [{"type": "command", "command": "other.py"}]}


def test_apply_guards_adds_three_pre_tool_use_entries() -> None:
    """After _apply_guards the dict holds exactly three PreToolUse entries, one per guard script."""
    hooks: dict = {}

    INSTALL._apply_guards(hooks, PY, False)

    assert len(pre_tool_use(hooks)) == 3


def test_apply_guards_registers_the_three_guard_scripts() -> None:
    """Each guard command runs its own script with the given interpreter."""
    hooks: dict = {}

    INSTALL._apply_guards(hooks, PY, False)

    commands = " ".join(commands_of(pre_tool_use(hooks)))
    for script_name in ("guard_git.py", "guard_destructive.py", "guard_config.py"):
        assert script_name in commands


def test_apply_guards_uses_the_bash_matcher_for_git_and_destructive() -> None:
    """guard_git.py and guard_destructive.py fire on Bash and PowerShell."""
    hooks: dict = {}

    INSTALL._apply_guards(hooks, PY, False)

    entries = pre_tool_use(hooks)
    assert matcher_of(entries, "guard_git.py") == "Bash|PowerShell"
    assert matcher_of(entries, "guard_destructive.py") == "Bash|PowerShell"


def test_apply_guards_uses_the_edit_matcher_for_config() -> None:
    """guard_config.py fires on the edit tools that can change a protected config."""
    hooks: dict = {}

    INSTALL._apply_guards(hooks, PY, False)

    assert matcher_of(pre_tool_use(hooks), "guard_config.py") == "Write|Edit|MultiEdit|NotebookEdit"


def test_apply_guards_called_twice_does_not_duplicate_entries() -> None:
    """The second install replaces its own three entries instead of appending them."""
    hooks: dict = {}

    INSTALL._apply_guards(hooks, PY, False)
    INSTALL._apply_guards(hooks, PY, False)

    assert len(pre_tool_use(hooks)) == 3


def test_apply_guards_keeps_a_foreign_pre_tool_use_entry() -> None:
    """An entry that is not ours is still there after the guards are installed."""
    hooks: dict = {"PreToolUse": [FOREIGN_ENTRY]}

    INSTALL._apply_guards(hooks, PY, False)

    assert FOREIGN_ENTRY in pre_tool_use(hooks)


def test_apply_guards_remove_keeps_a_foreign_pre_tool_use_entry() -> None:
    """Removing the guards does not touch an entry the installer never added."""
    hooks: dict = {"PreToolUse": [FOREIGN_ENTRY]}
    INSTALL._apply_guards(hooks, PY, False)

    INSTALL._apply_guards(hooks, PY, True)

    assert pre_tool_use(hooks) == [FOREIGN_ENTRY]


def test_apply_guards_remove_deletes_the_three_guard_entries() -> None:
    """remove=True leaves no guard command behind."""
    hooks: dict = {}
    INSTALL._apply_guards(hooks, PY, False)

    INSTALL._apply_guards(hooks, PY, True)

    assert commands_of(pre_tool_use(hooks)) == []


def test_apply_guards_remove_deletes_the_empty_pre_tool_use_key() -> None:
    """With nothing left under PreToolUse the key itself goes away."""
    hooks: dict = {}
    INSTALL._apply_guards(hooks, PY, False)

    INSTALL._apply_guards(hooks, PY, True)

    assert "PreToolUse" not in hooks


def test_apply_hooks_with_only_guards_installs_no_other_event() -> None:
    """_apply_hooks(..., only="guards") writes the guards and no notify/session/nudge hooks."""
    hooks: dict = {}

    INSTALL._apply_hooks(hooks, PY, False, "guards")

    assert len(pre_tool_use(hooks)) == 3
    for event in PROMPT_EVENTS:
        assert event not in hooks


def test_apply_hooks_with_none_also_installs_the_guards() -> None:
    """A full install carries the three guard commands next to the other hooks."""
    hooks: dict = {}

    INSTALL._apply_hooks(hooks, PY, False, None)

    commands = " ".join(commands_of(pre_tool_use(hooks)))
    for script_name in ("guard_git.py", "guard_destructive.py", "guard_config.py"):
        assert script_name in commands


def test_apply_hooks_remove_then_none_leaves_an_empty_dict() -> None:
    """Removing every group after a full install leaves nothing behind."""
    hooks: dict = {}
    INSTALL._apply_hooks(hooks, PY, False, None)

    INSTALL._apply_hooks(hooks, PY, True, None)

    assert hooks == {}


def test_apply_hooks_remove_then_none_keeps_a_foreign_entry() -> None:
    """A foreign PreToolUse entry is the only thing left after a full install and removal."""
    hooks: dict = {"PreToolUse": [FOREIGN_ENTRY]}
    INSTALL._apply_hooks(hooks, PY, False, None)

    INSTALL._apply_hooks(hooks, PY, True, None)

    assert pre_tool_use(hooks) == [FOREIGN_ENTRY]
