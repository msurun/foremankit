"""Tests for the night_continue hook in ``hooks/night_continue.py``.

The hook is loaded by path through importlib. Every test runs inside a tmp_path sandbox: the real ``~/.claude``
is never read or written.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_HOOKS_DIR = Path(__file__).resolve().parents[1] / "hooks"
sys.path.append(str(_HOOKS_DIR))

_SPEC = importlib.util.spec_from_file_location("night_continue", _HOOKS_DIR / "night_continue.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load night_continue.py from {_HOOKS_DIR}")
NIGHT_CONTINUE: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(NIGHT_CONTINUE)

@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Create a fake foremankit.night_run module with stub functions."""
    fake_module = ModuleType("foremankit.night_run")
    monkeypatch.setitem(sys.modules, "foremankit.night_run", fake_module)
    return fake_module

def test_should_continue_true_prints_json_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_module: ModuleType,
) -> None:
    """should_continue True prints JSON with decision 'block' and exits 0."""
    fake_module.should_continue = lambda: (True, None)
    fake_module.record_continue = lambda: None
    fake_module.read_night = lambda: {"done_file": "/path/to/done"}
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"stop_hook_active": True})))

    exit_code = NIGHT_CONTINUE.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {
        "decision": "block",
        "reason": "Continue the task from the plan and the latest handoff; when everything is "
                  "finished create the DONE file /path/to/done.",
    }

def test_should_continue_false_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_module: ModuleType,
) -> None:
    """should_continue False prints nothing and exits 0."""
    fake_module.should_continue = lambda: (False, None)
    fake_module.record_continue = lambda: None
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"stop_hook_active": True})))

    exit_code = NIGHT_CONTINUE.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""

def test_should_continue_raises_runtime_error_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_module: ModuleType,
) -> None:
    """should_continue raises RuntimeError prints nothing and exits 0."""
    fake_module.should_continue = lambda: (_ for _ in ()).throw(RuntimeError)
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"stop_hook_active": True})))

    exit_code = NIGHT_CONTINUE.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""

def test_stop_hook_active_true_still_continues_when_should_continue_true(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_module: ModuleType,
) -> None:
    """stop_hook_active True still continues when should_continue is True."""
    fake_module.should_continue = lambda: (True, None)
    fake_module.record_continue = lambda: None
    fake_module.read_night = lambda: {"done_file": "/path/to/done"}
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"stop_hook_active": True})))

    exit_code = NIGHT_CONTINUE.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {
        "decision": "block",
        "reason": "Continue the task from the plan and the latest handoff; when everything is "
                  "finished create the DONE file /path/to/done.",
    }

def test_invalid_json_on_stdin_still_works_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_module: ModuleType,
) -> None:
    """Invalid JSON on stdin is ignored and exits 0."""
    fake_module.should_continue = lambda: (True, None)
    fake_module.record_continue = lambda: None
    fake_module.read_night = lambda: {"done_file": "/path/to/done"}
    monkeypatch.setattr(sys, "stdin", StringIO("invalid json"))

    exit_code = NIGHT_CONTINUE.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""

def test_read_night_raises_os_error_after_should_continue_true_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_module: ModuleType,
) -> None:
    """read_night raises OSError after should_continue True prints nothing and exits 0."""
    fake_module.should_continue = lambda: (True, None)
    fake_module.record_continue = lambda: None
    fake_module.read_night = lambda: (_ for _ in ()).throw(OSError)
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps({"stop_hook_active": True})))

    exit_code = NIGHT_CONTINUE.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
