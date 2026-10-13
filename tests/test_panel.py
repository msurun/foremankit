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


def test_start_command_picks_the_chosen_model_without_the_menu():
    start = panel.command_for("start-server", "a.gguf")
    assert "start-llama.ps1" in start and "-Model 'a.gguf'" in start and "-NoMenu" in start


def test_tune_runs_the_tuner_and_then_the_context_sweep_as_one_job():
    tune = panel.command_for("tune", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
    assert tune.index("stop-server") < tune.index("tune_llama.py") < tune.index("sweep_ctx.py")
    assert "--dense" not in tune


def test_a_model_without_active_parameter_suffix_is_tuned_as_dense():
    assert "--dense" in panel.command_for("tune", "Spark-X2.5-4B-Q8_0.gguf")


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


def write_orchestrator_yaml(folder: Path) -> Path:
    path = folder / "orchestrator.yaml"
    path.write_text("port: 8081\nthinking_by_effort: {low: false, medium: false, high: true, xhigh: true, max: true}\n"
                    "language: English\n", encoding="utf-8")
    return path


def test_thinking_levels_are_rewritten_in_the_yaml_and_read_back(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    path = write_orchestrator_yaml(tmp_path)
    message = panel.set_thinking(panel.parse_levels("medium=true,max=false"))
    assert panel.get_thinking() == {"low": False, "medium": True, "high": True, "xhigh": True, "max": False}
    assert "language: English" in path.read_text(encoding="utf-8") and "medium" in message


def test_levels_not_named_keep_their_value(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    write_orchestrator_yaml(tmp_path)
    assert panel.parse_levels("low=true")["high"] is True


def test_direct_mode_removes_the_router_from_vscode_and_normal_mode_puts_it_back(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    write_orchestrator_yaml(tmp_path)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"editor.fontSize": 14}), encoding="utf-8")
    monkeypatch.setattr(panel, "vscode_settings_path", lambda: settings)
    stopped: list[str] = []
    monkeypatch.setattr(panel, "run_quick", lambda action: stopped.append(action) or "ok")
    assert panel.set_direct_mode(False).endswith("RELOAD THE VS CODE WINDOW to apply.")
    assert not panel.direct_mode()
    assert panel.set_direct_mode(True).startswith("direct mode")
    assert panel.direct_mode() and stopped == ["start-router", "stop-router"]
    assert json.loads(settings.read_text(encoding="utf-8"))["editor.fontSize"] == 14
