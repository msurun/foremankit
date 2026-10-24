"""Tests for hooks/guard_config.py, the PreToolUse config guard hook.

Covers the pure decisions (path matching, new text extraction, decide) and the
process behaviour: exit codes, stderr, stdout and the environment overrides.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"
HOOK_SCRIPT = HOOKS_DIR / "guard_config.py"
sys.path.insert(0, str(HOOKS_DIR))

import guard_config  # noqa: E402  (import after sys.path setup on purpose)

PROTECTED_FILES = [
    "project/ruff.toml",
    "project/.flake8",
    "project/mypy.ini",
    "project/.claude/settings.json",
    "project/hooks/guard_git.py",
]


def run_hook(stdin_text, extra_env=None):
    """Run the hook as a subprocess and return (exit_code, stdout, stderr)."""
    env = dict(os.environ)
    env.pop(guard_config.PROTECTED_PATHS_ENV, None)
    env.pop(guard_config.ALLOW_CONFIG_EDIT_ENV, None)
    env.update(extra_env or {})
    completed = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=stdin_text.encode("utf-8"),
        capture_output=True,
        env=env,
        timeout=30,
    )
    return (
        completed.returncode,
        completed.stdout.decode("utf-8"),
        completed.stderr.decode("utf-8"),
    )


def write_event(file_path, content=""):
    """Return a JSON Write event for the given path."""
    event = {"tool_name": "Write", "tool_input": {"file_path": file_path, "content": content}}
    return json.dumps(event)


@pytest.mark.parametrize("file_path", PROTECTED_FILES)
def test_protected_paths_are_blocked(file_path):
    decision, message = guard_config.decide(
        {"tool_input": {"file_path": file_path, "content": ""}}
    )

    assert decision == "block"
    assert file_path in message


@pytest.mark.parametrize(
    "file_path",
    ["project\\ruff.toml", "project\\.flake8", "project\\mypy.ini"],
)
def test_backslash_windows_paths_are_blocked(file_path):
    decision, _ = guard_config.decide(
        {"tool_input": {"file_path": file_path, "content": ""}}
    )

    assert decision == "block"


@pytest.mark.parametrize(
    "file_path",
    ["Project/RUFF.TOML", "Project/.FLAKE8", "Project/MYPY.INI"],
)
def test_upper_case_paths_are_blocked(file_path):
    decision, _ = guard_config.decide(
        {"tool_input": {"file_path": file_path, "content": ""}}
    )

    assert decision == "block"


def test_a_normal_python_write_is_allowed():
    decision, message = guard_config.decide(
        {"tool_input": {"file_path": "project/app/main.py", "content": "print(1)"}}
    )

    assert (decision, message) == ("allow", "")


@pytest.mark.parametrize("marker", ["[tool.ruff", "ignore =", "fail_under"])
def test_pyproject_edit_mentioning_a_setting_warns(marker):
    event = {"tool_input": {"file_path": "project/pyproject.toml", "content": marker}}

    decision, message = guard_config.decide(event)

    assert decision == "warn"
    assert message == guard_config.WARN_MESSAGE


def test_pyproject_edit_without_settings_is_allowed():
    event = {"tool_input": {"file_path": "project/pyproject.toml", "content": "name = 'x'"}}

    decision, _ = guard_config.decide(event)

    assert decision == "allow"


def test_multiedit_text_comes_from_each_edits_new_string():
    event = {
        "tool_input": {
            "file_path": "project/pyproject.toml",
            "edits": [{"old_string": "a", "new_string": "nothing"}, {"new_string": "fail_under"}],
        }
    }

    decision, _ = guard_config.decide(event)

    assert decision == "warn"


def test_multiedit_touching_protected_file_is_blocked():
    event = {
        "tool_input": {
            "file_path": "project/ruff.toml",
            "edits": [{"old_string": "a", "new_string": "b"}],
        }
    }

    decision, _ = guard_config.decide(event)

    assert decision == "block"


def test_the_allow_override_var_allows_a_protected_file(monkeypatch):
    monkeypatch.setenv(guard_config.ALLOW_CONFIG_EDIT_ENV, "1")
    event = {"tool_input": {"file_path": "project/ruff.toml", "content": ""}}

    decision, _ = guard_config.decide(event)

    assert decision == "allow"


def test_extra_protected_globs_are_added(monkeypatch):
    monkeypatch.setenv(guard_config.PROTECTED_PATHS_ENV, "*/secret.cfg")
    event = {"tool_input": {"file_path": "project/secret.cfg", "content": ""}}

    decision, _ = guard_config.decide(event)

    assert decision == "block"


def test_process_blocks_protected_file_with_exit_2():
    code, _, err = run_hook(write_event("project/ruff.toml"))

    assert code == 2
    assert "project/ruff.toml" in err
    assert guard_config.ALLOW_CONFIG_EDIT_ENV in err


def test_process_warns_and_prints_json_context():
    code, out, _ = run_hook(write_event("project/pyproject.toml", "[tool.ruff"))

    assert code == 0
    payload = json.loads(out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert payload["hookSpecificOutput"]["additionalContext"] == guard_config.WARN_MESSAGE


def test_process_allows_a_normal_file_silently():
    code, out, err = run_hook(write_event("project/app/main.py"))

    assert (code, out, err) == (0, "", "")


def test_process_allow_override_var_permits_protected_file():
    code, _, _ = run_hook(
        write_event("project/ruff.toml"),
        {guard_config.ALLOW_CONFIG_EDIT_ENV: "1"},
    )

    assert code == 0


def test_process_extra_protected_glob_blocks_matching_file():
    code, _, err = run_hook(
        write_event("project/secret.cfg"),
        {guard_config.PROTECTED_PATHS_ENV: "*/secret.cfg"},
    )

    assert code == 2
    assert "secret.cfg" in err


@pytest.mark.parametrize("stdin_text", ["not json", "", json.dumps({"tool_name": "Write"})])
def test_broken_events_exit_zero_without_blocking(stdin_text):
    code, out, _ = run_hook(stdin_text)

    assert code == 0
    assert out == ""


def test_a_bom_prefixed_event_is_read_and_still_blocks():
    stdin_text = "\ufeff" + write_event("project/ruff.toml")

    code, _, err = run_hook(stdin_text)

    assert code == 2
    assert "project/ruff.toml" in err
