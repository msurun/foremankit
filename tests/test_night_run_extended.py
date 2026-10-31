"""Extended tests for night_run with new driver kinds and tunnel checks."""

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


def test_external_kaggle_allowed_and_tunnel_ok(state_dir, monkeypatch):
    # Enable night run
    night.start_night(until=time.time() + 1000, max_continues=5, done_file="does_not_exist")
    # Stub driver kind
    monkeypatch.setattr(night.driver, "drives_kind", lambda: "external:kaggle")
    # Stub ping to succeed
    monkeypatch.setattr(night.utils, "ping_endpoint", lambda url, token, retries=3, backoff=0.3: True)
    # Stub local_drives to True
    monkeypatch.setattr(night, "local_drives", lambda: True)
    enabled, reason = night.should_continue()
    assert enabled
    assert reason == "ok"


def test_external_kaggle_tunnel_fails(state_dir, monkeypatch):
    night.start_night(until=time.time() + 1000, max_continues=5, done_file="does_not_exist")
    monkeypatch.setattr(night.driver, "drives_kind", lambda: "external:kaggle")
    monkeypatch.setattr(night.utils, "ping_endpoint", lambda url, token, retries=3, backoff=0.3: False)
    monkeypatch.setattr(night, "local_drives", lambda: True)
    enabled, reason = night.should_continue()
    assert not enabled
    assert reason == "tunnel"


def test_context_stop_prevents_continuation(state_dir, monkeypatch):
    night.start_night(until=time.time() + 1000, max_continues=5, done_file="does_not_exist")
    # Add context_stop flag to state file
    state = night.read_night() or {}
    state["context_stop"] = True
    # Write back using internal helper
    night._write_night(state)  # type: ignore[arg-type]
    monkeypatch.setattr(night, "local_drives", lambda: True)
    enabled, reason = night.should_continue()
    assert not enabled
    assert reason == "context_stop"
