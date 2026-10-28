"""Tests for foremankit/picker_guard.py: model_picker_block, merged_claude_settings, router_mode_active,
ensure_picker, claude_settings_path, vscode_settings_path, start.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from foremankit import picker_guard


def test_model_picker_block() -> None:
    """model_picker_block maps behaves_as->behavesAs and replace_built_in->replaceBuiltInOptions, {} without options."""
    assert picker_guard.model_picker_block({}) == {}
    assert picker_guard.model_picker_block({"model_picker": {}}) == {}
    assert picker_guard.model_picker_block({"model_picker": {"options": []}}) == {}
    assert picker_guard.model_picker_block({"model_picker": {"options": [{"behaves_as": "foo"}]}}) == {
        "options": [{"behavesAs": "foo"}],
        "replaceBuiltInOptions": False,
    }
    assert picker_guard.model_picker_block({"model_picker": {"options": [{"behaves_as": "foo"}], "replace_built_in": True}}) == {
        "options": [{"behavesAs": "foo"}],
        "replaceBuiltInOptions": True,
    }


def test_merged_claude_settings() -> None:
    """merged_claude_settings drops/sets the key."""
    assert picker_guard.merged_claude_settings({}, {}) == {}
    assert picker_guard.merged_claude_settings({"foo": "bar"}, {}) == {"foo": "bar"}
    assert picker_guard.merged_claude_settings({"foo": "bar"}, {"options": []}) == {"foo": "bar", "modelPicker": {"options": []}}


def test_router_mode_active(tmp_path: Path) -> None:
    """router_mode_active true with an {"name": "ANTHROPIC_BASE_URL", "value": "..."} entry in
    "claudeCode.environmentVariables", false for missing file, bad JSON, no entry, non-list value.
    """
    assert not picker_guard.router_mode_active(tmp_path / "missing.json")
    assert not picker_guard.router_mode_active(tmp_path / "bad.json")
    write_settings(tmp_path / "bad.json", "not a JSON object")
    assert not picker_guard.router_mode_active(tmp_path / "bad.json")
    write_settings(tmp_path / "empty.json", {})
    assert not picker_guard.router_mode_active(tmp_path / "empty.json")
    write_settings(tmp_path / "no-entries.json", {"claudeCode.environmentVariables": "not a list"})
    assert not picker_guard.router_mode_active(tmp_path / "no-entries.json")
    write_settings(tmp_path / "no-entry.json", {"claudeCode.environmentVariables": []})
    assert not picker_guard.router_mode_active(tmp_path / "no-entry.json")
    write_settings(tmp_path / "entry.json", {"claudeCode.environmentVariables": [{"name": "ANTHROPIC_BASE_URL"}]})
    assert picker_guard.router_mode_active(tmp_path / "entry.json")


def test_ensure_picker(tmp_path: Path) -> None:
    """ensure_picker restores a deleted modelPicker keeping every other key, returns False with file bytes unchanged
    when already equal, when router mode is off, when cfg has no model_picker, when the Claude settings file is missing
    (not created) or holds invalid JSON; written file is indent-4 JSON without BOM.
    """
    claude = tmp_path / "claude.json"
    vscode = tmp_path / "vscode.json"
    cfg = {"model_picker": {"options": [{"behaves_as": "foo"}]}}
    write_settings(vscode, {"claudeCode.environmentVariables": [{"name": "ANTHROPIC_BASE_URL"}]})
    assert not picker_guard.ensure_picker({}, claude, vscode)
    assert not picker_guard.ensure_picker(cfg, claude, tmp_path / "missing.json")
    assert not picker_guard.ensure_picker(cfg, claude, tmp_path / "bad.json")
    write_settings(tmp_path / "bad.json", "not a JSON object")
    assert not picker_guard.ensure_picker(cfg, claude, tmp_path / "bad.json")
    write_settings(claude, {"foo": "bar"})
    assert picker_guard.ensure_picker(cfg, claude, vscode)
    assert claude.read_text(encoding="utf-8") == '{\n    "foo": "bar",\n    "modelPicker": {\n        "options": [\n            {\n                "behavesAs": "foo"\n            }\n        ],\n        "replaceBuiltInOptions": false\n    }\n}\n'
    assert not picker_guard.ensure_picker(cfg, claude, vscode)
    write_settings(claude, {"foo": "bar", "modelPicker": {"options": [{"behavesAs": "foo"}], "replaceBuiltInOptions": False}})
    assert not picker_guard.ensure_picker(cfg, claude, vscode)
    write_settings(vscode, {"claudeCode.environmentVariables": []})
    assert not picker_guard.ensure_picker(cfg, claude, vscode)


def test_claude_settings_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """claude_settings_path honours FOREMANKIT_CLAUDE_SETTINGS."""
    monkeypatch.setenv(picker_guard.CLAUDE_SETTINGS_ENV, str(tmp_path / "claude.json"))
    assert picker_guard.claude_settings_path() == tmp_path / "claude.json"


def test_vscode_settings_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """vscode_settings_path honours FOREMANKIT_VSCODE_SETTINGS."""
    monkeypatch.setenv(picker_guard.VSCODE_SETTINGS_ENV, str(tmp_path / "vscode.json"))
    assert picker_guard.vscode_settings_path() == tmp_path / "vscode.json"


def test_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """start() contract: it resolves both paths once at call time and returns a threading.Event; setting it stops the
    daemon thread named "picker-guard". The start test must setenv both vars to tmp files, call
    `stop = picker_guard.start(lambda: cfg, interval=0.05)`, and in try/finally always `stop.set()` then join the thread
    found via threading.enumerate() by name (join(timeout=2), assert not alive). Inside the try: delete the key, poll up
    to 2 s until it is restored.
    """
    claude = tmp_path / "claude.json"
    vscode = tmp_path / "vscode.json"
    monkeypatch.setenv(picker_guard.CLAUDE_SETTINGS_ENV, str(claude))
    monkeypatch.setenv(picker_guard.VSCODE_SETTINGS_ENV, str(vscode))
    cfg = {"model_picker": {"options": [{"behaves_as": "foo"}]}}
    write_settings(vscode, {"claudeCode.environmentVariables": [{"name": "ANTHROPIC_BASE_URL"}]})
    write_settings(claude, {"foo": "bar"})
    stop = picker_guard.start(lambda: cfg, interval=0.05)
    try:
        _wait_for_restored_key(claude)
    finally:
        _stop_thread(stop)


def _wait_for_restored_key(claude: Path) -> None:
    """Wait up to 2 s for the modelPicker key to be restored in claude.json."""
    time.sleep(0.1)
    write_settings(claude, {"foo": "bar"})
    start = time.time()
    while time.time() - start < 2.0:
        if claude.read_text(encoding="utf-8") == '{\n    "foo": "bar",\n    "modelPicker": {\n        "options": [\n            {\n                "behavesAs": "foo"\n            }\n        ],\n        "replaceBuiltInOptions": false\n    }\n}\n':
            break
        time.sleep(0.1)
    else:
        pytest.fail("modelPicker not restored")


def _stop_thread(stop: threading.Event) -> None:
    """Stop the thread and wait for it to finish."""
    stop.set()
    thread = next(thread for thread in threading.enumerate() if thread.name == "picker-guard")
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def write_settings(path: Path, data: dict | str) -> None:
    """Write `data` to `path` as JSON or a string."""
    if isinstance(data, dict):
        path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(data, encoding="utf-8")
