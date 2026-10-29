"""
Tests for ForemanKit's night_run module.

The tests verify that the night‑run state file is respected and that the
decision logic in `should_continue` behaves correctly for all edge cases.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import foremankit.night_run as night


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the night‑run state directory to a temporary location."""
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    return tmp_path


def test_off_by_default_when_no_state_file(state_dir):
    enabled, reason = night.should_continue()
    assert not enabled
    assert reason == "off"


def test_should_continue_true_when_enabled_and_local_drives(state_dir, monkeypatch):
    night.start_night(until=time.time() + 1000, max_continues=10, done_file="does_not_exist")
    monkeypatch.setattr(night, "local_drives", lambda: True)
    enabled, reason = night.should_continue()
    assert enabled
    assert reason == "ok"


def test_should_continue_fails_when_no_local_drive(state_dir, monkeypatch):
    night.start_night(until=time.time() + 1000, max_continues=10, done_file="")
    monkeypatch.setattr(night, "local_drives", lambda: False)
    enabled, reason = night.should_continue()
    assert not enabled
    assert reason == "drives"


def test_should_continue_fails_when_done_file_exists(state_dir, monkeypatch):
    done_path = state_dir / "done.marker"
    done_path.touch()
    night.start_night(until=time.time() + 1000, max_continues=10, done_file=str(done_path))
    monkeypatch.setattr(night, "local_drives", lambda: True)
    enabled, reason = night.should_continue()
    assert not enabled
    assert reason == "done"


def test_should_continue_fails_when_limit_reached(state_dir, monkeypatch):
    night.start_night(until=time.time() + 1000, max_continues=2, done_file="")
    monkeypatch.setattr(night, "local_drives", lambda: True)
    assert night.record_continue() == 1
    assert night.record_continue() == 2
    enabled, reason = night.should_continue()
    assert not enabled
    assert reason == "limit"


def test_should_continue_fails_when_expired(state_dir, monkeypatch):
    past = time.time() - 10
    night.start_night(until=past, max_continues=10, done_file="")
    monkeypatch.setattr(night, "local_drives", lambda: True)
    enabled, reason = night.should_continue()
    assert not enabled
    assert reason == "expired"


def test_stop_night_disables(state_dir, monkeypatch):
    night.start_night(until=time.time() + 1000, max_continues=5, done_file="dummy")
    monkeypatch.setattr(night, "local_drives", lambda: True)
    pre_enabled, _ = night.should_continue()
    assert pre_enabled
    night.stop_night()
    post_enabled, reason = night.should_continue()
    assert not post_enabled
    assert reason == "off"


def test_record_continue_updates_state(state_dir):
    night.start_night(until=time.time() + 1000, max_continues=10, done_file="")
    first = night.record_continue()
    second = night.record_continue()
    assert first == 1
    assert second == 2
    state = night.read_night()
    assert isinstance(state, dict)
    assert state.get("count") == 2
