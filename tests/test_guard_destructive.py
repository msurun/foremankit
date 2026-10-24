"""Tests for hooks/guard_destructive.py: which shell commands the guard blocks.

The guard is loaded as a plain module (the hooks folder is added to sys.path)
so the pure rule function can be tested directly. The process tests run the
script exactly as Claude Code does: a JSON hook event on stdin, and the exit
code plus stderr decide whether the tool call may proceed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"
GUARD_SCRIPT = HOOKS_DIR / "guard_destructive.py"
OVERRIDE_ENV = "FOREMANKIT_ALLOW_DESTRUCTIVE"

sys.path.insert(0, str(HOOKS_DIR))

import guard_destructive  # noqa: E402

BLOCKED_COMMANDS = [
    "git push --force",
    "git push -f",
    "git push --force-with-lease",
    "git push origin +main",
    "git reset --hard",
    "git clean -fd",
    "git clean -fdx",
    "git clean -xdf",
    "git checkout .",
    "git checkout -- .",
    "git restore .",
    "git branch -D x",
    "git stash drop",
    "git stash clear",
    "rm -rf src",
    "rm -r /",
    "rm -rf ~",
    "rm -rf *",
    "rmdir /s",
    "Remove-Item -Recurse -Force C:\\proj",
    "Remove-Item -recurse",
    "del /s",
    "format",
    'psql -c "DROP TABLE t"',
]

ALLOWED_COMMANDS = [
    "git push origin main",
    "git reset --soft HEAD~1",
    "git status",
    'git commit -m "git push --force"',
    "rm -rf /tmp/x",
    "rm -rf .pytest_cache",
    "rm -rf a/__pycache__",
    "rm -rf node_modules",
    "rm -rf C:/Users/u/AppData/Local/Temp/claude/x/scratchpad/y",
    "Remove-Item -Recurse C:\\work\\scratchpad\\x",
    "rm file.txt",
    "git branch -d x",
    "git stash list",
    "ls",
    "git clean -n",
    "git reset",
    "git push",
]


def run_hook(stdin_text: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the guard script as a subprocess with the given stdin text."""
    env = {**os.environ}
    env.pop(OVERRIDE_ENV, None)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(GUARD_SCRIPT)],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=30,
        check=False,
    )


def bash_event(command: str) -> str:
    """Return the JSON hook event a Bash tool call would produce."""
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})


@pytest.mark.parametrize("command", BLOCKED_COMMANDS)
def test_a_destructive_command_is_blocked_by_a_rule(command):
    tokens = [command.split()[0], *command.split()[1:]]
    command_tokens = guard_destructive.simple_commands(command)[0]

    reason = guard_destructive.rule_for(command_tokens)

    assert reason != "", f"expected {command!r} to be blocked, got tokens {tokens}"


@pytest.mark.parametrize("command", ALLOWED_COMMANDS)
def test_a_safe_command_is_allowed_by_a_rule(command):
    command_tokens = guard_destructive.simple_commands(command)[0]

    reason = guard_destructive.rule_for(command_tokens)

    assert reason == "", f"expected {command!r} to be allowed, got {reason!r}"


def test_an_empty_token_list_is_allowed():
    assert guard_destructive.rule_for([]) == ""


@pytest.mark.parametrize("command", BLOCKED_COMMANDS)
def test_a_destructive_command_exits_two_with_a_message(command):
    result = run_hook(bash_event(command))

    assert result.returncode == 2
    assert "guard_destructive" in result.stderr


@pytest.mark.parametrize("command", ALLOWED_COMMANDS)
def test_a_safe_command_exits_zero_with_no_output(command):
    result = run_hook(bash_event(command))

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == ""


def test_a_force_push_after_another_command_is_blocked():
    result = run_hook(bash_event("echo hi && git push --force"))

    assert result.returncode == 2
    assert "force push" in result.stderr


def test_a_wrapped_powershell_reset_is_blocked():
    result = run_hook(bash_event('powershell -Command "git reset --hard"'))

    assert result.returncode == 2


def test_the_override_environment_variable_allows_a_blocked_command():
    result = run_hook(bash_event("git push --force"), {OVERRIDE_ENV: "1"})

    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.parametrize(
    "stdin_text",
    ["", "   ", "not json at all", "{}"],
)
def test_malformed_stdin_exits_zero_silently(stdin_text):
    result = run_hook(stdin_text)

    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.parametrize("tool_name", ["Write", "Edit", "Read", "Grep"])
def test_an_event_for_another_tool_exits_zero_silently(tool_name):
    event = json.dumps(
        {"tool_name": tool_name, "tool_input": {"command": "git push --force"}}
    )

    result = run_hook(event)

    assert result.returncode == 0
    assert result.stderr == ""


def test_an_empty_command_exits_zero_silently():
    result = run_hook(bash_event(""))

    assert result.returncode == 0
    assert result.stderr == ""
