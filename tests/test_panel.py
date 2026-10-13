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


def test_server_start_is_detached_and_needs_a_model(monkeypatch):
    calls = []
    monkeypatch.setattr(panel.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    message = panel.start_server("a.gguf")
    command = " ".join(calls[0])
    assert "Start-Process" in command and "'-Model','a.gguf'" in command and "'-NoMenu'" in command and "a.gguf" in message
    with pytest.raises(ValueError):
        panel.start_server("")


def test_tune_runs_the_tuner_and_then_the_context_sweep_as_one_job():
    tune = panel.command_for("tune", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
    assert tune.index("stop-server") < tune.index("tune_llama.py") < tune.index("sweep_ctx.py")
    assert "--dense" not in tune


def test_a_model_without_active_parameter_suffix_is_tuned_as_dense():
    assert "--dense" in panel.command_for("tune", "Spark-X2.5-4B-Q8_0.gguf")


def test_unknown_action_is_rejected():
    with pytest.raises(KeyError):
        panel.command_for("format-disk")


def test_status_reports_router_server_and_role(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(panel, "port_open", lambda port: True)
    monkeypatch.setattr(panel, "probe", lambda url: {"model_alias": "m", "default_generation_settings": {"n_ctx": 4096}})
    result = panel.status()
    assert result["router"]["up"] and result["server"] == {"up": True, "model": "m", "n_ctx": 4096}
    assert result["role"] == "both" and json.dumps(result)


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


def test_orchestrator_role_keeps_delegated_tasks_off_this_machine(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    assert panel.set_role("orchestrator") == "local server role: orchestrator"
    assert panel.local_role() == "orchestrator"
    assert panel.StateStore().load()["toggles"]["llm:self"] == "off"
    panel.set_role("worker")
    assert panel.StateStore().load()["toggles"]["llm:self"] == "on"
    panel.set_role("both")
    assert panel.StateStore().load()["toggles"]["llm:self"] == "auto"


def test_unknown_role_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        panel.set_role("boss")


def test_added_worker_becomes_a_node_of_the_pool(tmp_path: Path, monkeypatch):
    from foremankit import nodes
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("FOREMANKIT_NODES", raising=False)   # the test fixture turns the node pool off
    panel.add_worker("garage", "http://192.168.1.50:8080/", "private", 3)
    entry = nodes.load_nodes()["nodes"]["garage"]
    assert entry["lan"] == ["http://192.168.1.50:8080"] and entry["accept"] == "private" and entry["priority"] == 3
    assert panel.remove_worker("garage") == "worker garage removed"
    assert "garage" not in (nodes.load_nodes().get("nodes") or {})


@pytest.mark.parametrize("name, url, accept", [
    ("Bad Name", "http://h:1", "internal"), ("ok", "ftp://h", "internal"), ("ok", "http://h:1", "secret"),
])
def test_invalid_worker_is_rejected(tmp_path: Path, monkeypatch, name, url, accept):
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        panel.add_worker(name, url, accept)


def test_workers_of_nodes_yaml_cannot_be_removed_here(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        panel.remove_worker("firefly")


def sandbox_profiles(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(panel, "LLAMA", tmp_path)
    monkeypatch.setattr(panel, "load_profiles", lambda: (json.loads((tmp_path / "profiles.json").read_text("utf-8")), "m"))
    (tmp_path / "profiles.json").write_text("{}", encoding="utf-8")
    return tmp_path / "profiles.json"


def test_manual_parameters_are_stored_in_the_profile_and_marked_manual(tmp_path: Path, monkeypatch):
    path = sandbox_profiles(tmp_path, monkeypatch)
    panel.set_profile("Spark-X2.5-4B-Q8_0.gguf", {"ctx": "65536", "temp": "0.3", "kv_cache": "q8_0", "reasoning": "off"})
    stored = json.loads(path.read_text(encoding="utf-8"))["m || spark-x2.5-4b-q8_0"]
    assert stored["ctx"] == 65536 and stored["temp"] == 0.3 and stored["kv_cache"] == "q8_0" and stored["manual"] is True


def test_empty_manual_value_removes_the_parameter(tmp_path: Path, monkeypatch):
    path = sandbox_profiles(tmp_path, monkeypatch)
    panel.set_profile("a.gguf", {"ncmoe": 30})
    panel.set_profile("a.gguf", {"ncmoe": ""})
    assert "ncmoe" not in json.loads(path.read_text(encoding="utf-8"))["m || a"]


@pytest.mark.parametrize("fields", [{"ctx": 100}, {"temp": 5}, {"threads": "many"}, {"kv_cache": "q2"}, {"color": "red"}])
def test_out_of_range_or_unknown_manual_parameter_is_rejected(tmp_path: Path, monkeypatch, fields):
    sandbox_profiles(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        panel.set_profile("a.gguf", fields)


def test_direct_mode_leaves_a_flag_that_keeps_the_scheduled_task_from_reviving_the_router(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    write_orchestrator_yaml(tmp_path)
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(panel, "vscode_settings_path", lambda: settings)
    monkeypatch.setattr(panel, "run_quick", lambda action: "ok")
    panel.set_direct_mode(True)
    assert (tmp_path / panel.ROUTER_OFF_FLAG).exists()
    panel.set_direct_mode(False)
    assert not (tmp_path / panel.ROUTER_OFF_FLAG).exists()
