"""Tests for the gateway's overflow spill: a local request that does not fit the llama-server window is sent
to the cloud untouched, so the Claude Code session survives and auto-compacts as usual.
Pure tests: no network, no llama-server, the window is pinned with a fake `server_window`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from foremankit import gateway

WINDOW_TOKENS = 32768            # the pinned context window of the fake llama-server
BIG_FILL = 100_000               # user text in the 70-90 % band: above the ~91_750 char spill, below the 90 % stop
SMALL_FILL = 1_000


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a minimal orchestrator.yaml/.md to a temp config dir and point the gateway at it."""
    (tmp_path / "orchestrator.md").write_text("PROMPT", encoding="utf-8")
    (tmp_path / "orchestrator.yaml").write_text("trim:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


@pytest.fixture
def fixed_window(monkeypatch: pytest.MonkeyPatch) -> int:
    """Pin the llama-server window so the overflow threshold is deterministic."""
    monkeypatch.setattr(gateway, "server_window", lambda cfg: WINDOW_TOKENS)
    return WINDOW_TOKENS


def make_request(fill_chars: int) -> dict[str, Any]:
    """A main-loop /v1/messages request for the local model, with a user message of `fill_chars` characters."""
    return {
        "model": "claude-sonnet-5",
        "system": "s",
        "tools": [{"name": "Agent", "description": "spawn"}],
        "messages": [{"role": "user", "content": "x" * fill_chars}],
    }


def prepare(fill_chars: int) -> tuple[gateway.Call, dict[str, Any], dict[str, Any]]:
    """Run prepare_body on a fresh local call; returns the call, its body JSON and the config."""
    cfg = gateway.settings()
    call = gateway.Call("/v1/messages", "claude-sonnet-5", True, time.time())
    data = make_request(fill_chars)
    body = gateway.prepare_body(call, data, json.dumps(data).encode("utf-8"), cfg)
    return call, json.loads(body), cfg


def test_overflows_is_false_for_a_small_request(config_dir: Path, fixed_window: int) -> None:
    """A request well inside the window is not an overflow."""
    assert gateway.overflows(make_request(SMALL_FILL), gateway.settings()) is False


def test_overflows_is_true_for_a_huge_request(config_dir: Path, fixed_window: int) -> None:
    """A request above 70 % of the window is an overflow."""
    assert gateway.overflows(make_request(BIG_FILL), gateway.settings()) is True


def test_overflowing_main_loop_request_is_spilled_to_cloud(config_dir: Path, fixed_window: int) -> None:
    """The spill keeps the client's model id so the cloud answers as the model the plugin asked for."""
    call, body, _ = prepare(BIG_FILL)
    assert call.local is False
    assert "does not fit" in call.note
    assert body["model"] == "claude-sonnet-5"


def test_small_main_loop_request_stays_local(config_dir: Path, fixed_window: int) -> None:
    """A request that fits is shaped for llama-server and stays local."""
    call, body, cfg = prepare(SMALL_FILL)
    assert call.local is True
    assert body["model"] == cfg["local_target"]


def test_overflow_spill_is_off_when_disabled(config_dir: Path, fixed_window: int) -> None:
    """`overflow_to_cloud: false` restores the old behaviour: the request goes to the local model."""
    (config_dir / "orchestrator.yaml").write_text(
        "trim:\n  enabled: true\noverflow_to_cloud: false\n", encoding="utf-8")
    call, body, cfg = prepare(BIG_FILL)
    assert call.local is True
    assert body["model"] == cfg["local_target"]


def test_overflowing_request_records_no_window_usage(config_dir: Path, fixed_window: int) -> None:
    """A request that leaves for the cloud must not feed the local context hint."""
    prepare(BIG_FILL)
    assert not (config_dir / "state" / "gateway_context.json").exists()


def test_small_request_records_window_usage(config_dir: Path, fixed_window: int) -> None:
    """A local request still records how full the local window is."""
    prepare(SMALL_FILL)
    assert (config_dir / "state" / "gateway_context.json").exists()
