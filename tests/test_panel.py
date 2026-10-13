"""The VS Code panel backend: status JSON, model list with profiles, action commands and the local on/off switch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from foremankit import panel


def test_model_entry_reads_the_profile_of_this_machine(tmp_path: Path):
    model = tmp_path / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    model.write_bytes(b"x" * 1024)
    profiles = {"cpu | gpu | 8gb || qwen3.6-35b-a3b-ud-q4_k_xl": {"ctx": 65536, "tg_tok_s": 27.8, "ncmoe": 34, "reasoning": "off"}}
    entry = panel.model_entry(model, profiles, "cpu | gpu | 8gb")
    assert entry["tuned"] and entry["moe"] and entry["ctx"] == 65536 and entry["reasoning"] == "off"


def test_untuned_dense_model_is_listed_without_profile_fields(tmp_path: Path):
    model = tmp_path / "Spark-X2.5-4B-Q8_0.gguf"
    model.write_bytes(b"x")
    entry = panel.model_entry(model, {}, "cpu | gpu | 8gb")
    assert not entry["tuned"] and not entry["moe"] and "ctx" not in entry


def test_long_actions_become_terminal_commands_for_the_chosen_model():
    start = panel.command_for("start-server", "a.gguf")
    assert "start-llama.ps1" in start and "-Model 'a.gguf'" in start and "-NoMenu" in start
    tune = panel.command_for("tune", "a.gguf", dense=True)
    assert "stop-server" in tune and "tune_llama.py" in tune and tune.endswith("--dense")
    assert "sweep_ctx.py" in panel.command_for("sweep", "a.gguf")


def test_unknown_action_is_rejected():
    with pytest.raises(KeyError):
        panel.command_for("format-disk")


def test_local_switch_writes_and_removes_the_flag_the_gateway_checks(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    assert panel.set_local(False) == "local models off"
    assert (panel.state_dir() / panel.LOCAL_OFF_FLAG).exists()
    assert panel.set_local(True) == "local models on"
    assert not (panel.state_dir() / panel.LOCAL_OFF_FLAG).exists()


def test_status_reports_router_server_and_local_switch(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(panel, "port_open", lambda port: True)
    monkeypatch.setattr(panel, "probe", lambda url: {"model_alias": "m", "default_generation_settings": {"n_ctx": 4096}})
    result = panel.status()
    assert result["router"]["up"] and result["server"] == {"up": True, "model": "m", "n_ctx": 4096}
    assert result["local_enabled"] is True and json.dumps(result)
