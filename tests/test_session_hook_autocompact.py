"""Tests for the auto-compact window lookup in ForemanKit/hooks/session_length_reminder.py.

The hook is not an installable package, so it is loaded in-process from its file path. `Path.home` is redirected to a
temp folder so the tests never touch the real ~/.claude/settings.json.
"""
import importlib.util
import json
from pathlib import Path

import pytest

HOOK_PATH = Path(__file__).resolve().parent.parent / "hooks" / "session_length_reminder.py"


@pytest.fixture(scope="session")
def session_hook():
    """The session_length_reminder module, loaded once from its file for all tests in this session."""
    spec = importlib.util.spec_from_file_location("session_length_reminder", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook(session_hook, tmp_path, monkeypatch):
    """The hook module with Path.home pointing at tmp_path and no FOREMANKIT_CLOUD_WINDOW override."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FOREMANKIT_CLOUD_WINDOW", raising=False)
    return session_hook


def write_settings(home_dir: Path, settings: dict) -> None:
    """Write `settings` to <home_dir>/.claude/settings.json, creating the folder when needed."""
    claude_dir = home_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")


def test_autocompact_window_returns_configured_value(hook, tmp_path):
    """A modelSettings entry for the exact model id supplies its autoCompactWindow to cloud_window."""
    write_settings(tmp_path, {"modelSettings": {"claude-sonnet-5": {"autoCompactWindow": 100000}}})
    assert hook.autocompact_window("claude-sonnet-5") == 100000
    assert hook.cloud_window("claude-sonnet-5", 50_000) == 100000


def test_prefix_key_matches_dated_model_id(hook, tmp_path):
    """A settings key that is a prefix of the model id (versioned id) still supplies the window."""
    write_settings(tmp_path, {"modelSettings": {"claude-sonnet-5": {"autoCompactWindow": 100000}}})
    assert hook.autocompact_window("claude-sonnet-5-20260101") == 100000


def test_missing_settings_falls_back_to_default_window(hook):
    """Without settings.json there is no override: small contexts use 200k, a "[1m]" id uses 1M."""
    assert hook.autocompact_window("claude-sonnet-5") == 0
    assert hook.cloud_window("claude-sonnet-5", 50_000) == 200_000
    assert hook.cloud_window("claude-sonnet-5[1m]", 50_000) == 1_000_000


def test_environment_variable_wins_over_settings(hook, tmp_path, monkeypatch):
    """FOREMANKIT_CLOUD_WINDOW takes precedence over the autoCompactWindow from settings.json."""
    write_settings(tmp_path, {"modelSettings": {"claude-sonnet-5": {"autoCompactWindow": 100000}}})
    monkeypatch.setenv("FOREMANKIT_CLOUD_WINDOW", "300000")
    assert hook.cloud_window("claude-sonnet-5", 50_000) == 300000


def test_broken_autocompact_value_returns_zero(hook, tmp_path):
    """A non-numeric autoCompactWindow is ignored: autocompact_window returns 0 instead of raising."""
    write_settings(tmp_path, {"modelSettings": {"claude-sonnet-5": {"autoCompactWindow": "abc"}}})
    assert hook.autocompact_window("claude-sonnet-5") == 0
