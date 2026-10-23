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


def test_server_can_be_started_shared_with_the_lab(monkeypatch):
    calls = []
    monkeypatch.setattr(panel.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert "-Share" in panel.start_server("a.gguf", share=True)
    assert "'-Share'" in " ".join(calls[0])
    panel.start_server("a.gguf")
    assert "'-Share'" not in " ".join(calls[1])


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
    assert result["role"] == "worker" and json.dumps(result)


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


def test_pair_find_hides_this_computer(monkeypatch):
    monkeypatch.setattr(panel.platform, "node", lambda: "Neuron")
    monkeypatch.setattr(panel.pairing, "find_hosts", lambda: [
        {"name": "neuron", "ip": "192.168.137.1", "port": 8767}, {"name": "neuron", "ip": "127.0.0.1", "port": 8767},
        {"name": "firefly", "ip": "192.168.1.20", "port": 8767}])
    assert [host["name"] for host in panel.other_hosts()] == ["firefly"]


def test_a_running_process_without_an_answer_is_reported_as_starting(monkeypatch):
    monkeypatch.setattr(panel, "probe", lambda url: None)
    monkeypatch.setattr(panel, "server_process_running", lambda: True)
    assert panel.server_status() == {"up": False, "starting": True}
    monkeypatch.setattr(panel, "server_process_running", lambda: False)
    assert panel.server_status() == {"up": False, "starting": False}


def lab_with_worker(tmp_path: Path, monkeypatch, up: bool = True) -> None:
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("FOREMANKIT_NODES", raising=False)
    monkeypatch.setattr(panel, "worker_status", lambda url: {"up": up, "model": "m" if up else None})
    panel.add_worker("garage", "http://192.168.1.50:8080", "private", 3)


def test_orchestrator_can_move_to_another_machine_and_back(tmp_path: Path, monkeypatch):
    from foremankit import gateway
    lab_with_worker(tmp_path, monkeypatch)
    assert "garage" in panel.set_orchestrator("garage")
    assert gateway.settings()["local_url"] == "http://192.168.1.50:8080"
    assert panel.list_workers()[0]["orchestrator"] is True
    panel.set_orchestrator("this")
    assert gateway.settings()["local_url"] != "http://192.168.1.50:8080" and gateway.orchestrator_target() is None


def test_a_worker_role_does_not_refuse_claude_code_while_the_orchestrator_is_remote(tmp_path: Path, monkeypatch):
    from foremankit import gateway
    lab_with_worker(tmp_path, monkeypatch)
    panel.set_role("worker")
    assert gateway.refusal() is not None
    panel.set_orchestrator("garage")
    assert gateway.refusal() is None


def test_unknown_machine_cannot_become_the_orchestrator(tmp_path: Path, monkeypatch):
    lab_with_worker(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        panel.set_orchestrator("nowhere")


def test_remote_orchestrator_that_is_down_is_chosen_with_a_hint(tmp_path: Path, monkeypatch):
    lab_with_worker(tmp_path, monkeypatch, up=False)
    assert "-Share" in panel.set_orchestrator("garage")


def test_worker_can_be_disabled_for_the_router_without_deleting_it(tmp_path: Path, monkeypatch):
    lab_with_worker(tmp_path, monkeypatch)
    assert panel.list_workers()[0]["enabled"] is True
    panel.set_worker_enabled("garage", False)
    assert panel.list_workers()[0]["enabled"] is False and panel.StateStore().load()["toggles"]["llm:garage"] == "off"
    panel.set_worker_enabled("garage", True)
    assert panel.list_workers()[0]["enabled"] is True


def serve_fake_llama(status_after_health: int = 200):
    """A tiny llama-server stand-in on a free local port: /health and /v1/chat/completions."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(status_after_health)
            self.end_headers()
            self.wfile.write(json.dumps({"usage": {"completion_tokens": 3}}).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_worker_test_reports_a_working_server(tmp_path: Path, monkeypatch):
    lab_with_worker(tmp_path, monkeypatch)
    server = serve_fake_llama()
    try:
        panel.add_worker("fake", f"http://127.0.0.1:{server.server_port}")
        assert panel.test_worker("fake").startswith("fake: OK") and "3 tokens" in panel.test_worker("fake")
    finally:
        server.shutdown()


def test_worker_test_points_at_the_lab_key_when_generation_is_refused(tmp_path: Path, monkeypatch):
    lab_with_worker(tmp_path, monkeypatch)
    server = serve_fake_llama(401)
    try:
        panel.add_worker("fake", f"http://127.0.0.1:{server.server_port}")
        assert "FOREMANKIT_PEER_KEY" in panel.test_worker("fake")
    finally:
        server.shutdown()


def test_worker_test_says_so_when_nothing_answers(tmp_path: Path, monkeypatch):
    lab_with_worker(tmp_path, monkeypatch)
    panel.add_worker("dead", "http://127.0.0.1:9")
    assert "no connection" in panel.test_worker("dead")


def test_delegations_sum_today_and_name_the_targets(tmp_path: Path, monkeypatch):
    import datetime as dt
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    now = dt.datetime(2026, 10, 13, 15, 0).timestamp()
    rows = [(now - 86400 * 2, "groq", None, "llama", 0.5, True), (now - 3600, "deepseek", None, "deepseek-flash", 0.002, True),
            (now - 600, "deepseek", None, "deepseek-flash", 0.001, True), (now - 300, "llm", "legion", "qwen", 0.0, False)]
    lines = [json.dumps({"ts": ts, "event": "call", "kind": "quick", "provider": provider, "node": node, "model": model,
                         "success": ok, "cost_usd": cost, "tokens_in": 10, "tokens_out": 5, "dry_run": False})
             for ts, provider, node, model, cost, ok in rows]
    (tmp_path / "calls.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = panel.delegations(now)
    assert result["today"]["calls"] == 3 and abs(result["today"]["cost"] - 0.003) < 1e-9
    assert set(result["today"]["by_target"]) == {"deepseek / deepseek-flash", "llm@legion / qwen"}
    assert abs(result["week_cost"] - 0.503) < 1e-9 and len(result["days"]) == 7
    assert result["recent"][0]["target"] == "llm@legion / qwen" and result["recent"][0]["ok"] is False


def test_delegations_with_no_log_are_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    result = panel.delegations()
    assert result["today"]["calls"] == 0 and result["recent"] == [] and result["week_cost"] == 0


def sandboxed_installer(tmp_path: Path, monkeypatch) -> Path:
    """Point the installer module used by the panel at a temporary VS Code user folder."""
    install = panel.load_installer()
    vscode = tmp_path / "vscode"
    vscode.mkdir()
    monkeypatch.setattr(install.Installer, "vscode_user_dir", staticmethod(lambda: vscode))
    monkeypatch.setattr(install, "STATE_DIR", tmp_path / "install-state")
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    write_orchestrator_yaml(tmp_path)
    monkeypatch.setattr(panel, "run_quick", lambda action, *a: "ok")
    return vscode


def test_direct_mode_is_the_default_when_nothing_is_configured(tmp_path: Path, monkeypatch):
    sandboxed_installer(tmp_path, monkeypatch)
    assert panel.direct_mode() is True


def test_router_mode_adds_the_entries_and_direct_mode_removes_every_trace(tmp_path: Path, monkeypatch):
    vscode = sandboxed_installer(tmp_path, monkeypatch)
    (vscode / "settings.json").write_text(json.dumps({"editor.fontSize": 14}), encoding="utf-8")
    install = panel.load_installer()
    assert panel.set_direct_mode(False).endswith("RELOAD THE VS CODE WINDOW to apply.")
    assert not panel.direct_mode() and install._fake_scheduler["registered"] is True
    assert panel.set_direct_mode(True).startswith("direct mode")
    settings = json.loads((vscode / "settings.json").read_text(encoding="utf-8"))
    assert panel.direct_mode() and settings == {"editor.fontSize": 14} and install._fake_scheduler["registered"] is False


def test_direct_mode_creates_no_vscode_files_that_were_not_there(tmp_path: Path, monkeypatch):
    vscode = sandboxed_installer(tmp_path, monkeypatch)
    panel.set_direct_mode(True)
    assert list(vscode.iterdir()) == []


def test_delegated_tokens_are_priced_as_if_they_ran_on_claude(tmp_path: Path, monkeypatch):
    import datetime as dt
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    now = dt.datetime(2026, 10, 13, 15, 0).timestamp()
    call = {"ts": now - 60, "event": "call", "kind": "quick", "provider": "groq", "model": "m", "success": True,
            "cost_usd": 0.0, "tokens_in": 1_000_000, "tokens_out": 100_000, "dry_run": False}
    (tmp_path / "calls.jsonl").write_text(json.dumps(call) + "\n", encoding="utf-8")
    worth = panel.delegations(now)["claude_equivalent"]
    assert abs(worth["today"]["haiku-5.5"] - 0.15) < 1e-9 and abs(worth["today"]["sonnet-5.5"] - 3.0) < 1e-9
    assert worth["week"] == worth["today"]


def test_delegation_can_be_switched_off_as_a_whole_and_per_category(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    assert panel.delegation_switches() == {"paid": True, "free": True, "local": True}
    assert "everything runs on Claude" in panel.set_delegation("all", False)
    assert panel.delegation_switches() == {"paid": False, "free": False, "local": False}
    panel.set_delegation("local", True)
    assert panel.delegation_switches() == {"paid": False, "free": False, "local": True}
    panel.set_delegation("all", True)
    assert all(panel.delegation_switches().values())


def test_firewall_rules_name_the_venv_launcher_and_the_base_interpreter(monkeypatch):
    monkeypatch.setattr(panel.sys, "executable", r"C:\proj\.venv\Scripts\python.exe")
    monkeypatch.setattr(panel.sys, "_base_executable", r"C:\Python310\python.exe", raising=False)
    assert panel.listening_programs() == [r"C:\proj\.venv\Scripts\python.exe", r"C:\Python310\python.exe"]


def test_firewall_counts_as_open_only_when_private_and_public_are_both_allowed(monkeypatch):
    monkeypatch.setattr(panel.os, "name", "nt")
    for output, expected in (("Private,Public", True), ("Private", False), ("Public", False), ("", False)):
        monkeypatch.setattr(panel.subprocess, "run", lambda *a, _o=output, **k: type("R", (), {"stdout": _o})())
        assert panel.pair_firewall_open() is expected


def test_no_firewall_step_is_offered_off_windows(monkeypatch):
    monkeypatch.setattr(panel.os, "name", "posix")
    assert panel.pair_firewall_open() is None and "no firewall rule needed" in panel.open_pair_firewall()


def test_the_firewall_script_names_both_profiles_and_limits_the_remote_scope():
    script = (panel.LLAMA / "pair-firewall.ps1").read_text(encoding="utf-8")
    assert "'Private', 'Public'" in script and "-RemoteAddress LocalSubnet" in script and "8766" in script and "8767" in script


def test_orchestrator_mode_lists_upstreams_and_refuses_a_too_sensitive_folder(monkeypatch, tmp_path: Path):
    """The panel lists claude, local and the upstreams; switching to DeepSeek is refused for a private folder."""
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TEST_PANEL_KEY", "k")
    (tmp_path / "orchestrator.yaml").write_text(
        "upstreams:\n  deepseek: {url: http://x, key_env: TEST_PANEL_KEY, ceiling: normal, models: {sonnet: m}}\n",
        encoding="utf-8")
    assert panel.orchestrator_mode_status() == {"mode": "claude", "modes": ["claude", "local", "deepseek"]}
    with pytest.raises(ValueError, match="private"):
        panel.set_orchestrator_mode("deepseek", str(tmp_path))
    (tmp_path / ".foremankit").mkdir()
    (tmp_path / ".foremankit" / "sensitivity").write_text("normal", encoding="utf-8")
    assert "deepseek" in panel.set_orchestrator_mode("deepseek", str(tmp_path))
    assert panel.orchestrator_mode_status()["mode"] == "deepseek"


def test_apply_preset_changes_nothing_when_it_is_refused_and_puts_direct_last(monkeypatch, tmp_path: Path):
    """A refused preset (private folder) touches no setting; an accepted one applies mode, role and direct in that order."""
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TEST_PANEL_KEY", "k")
    (tmp_path / "orchestrator.yaml").write_text(
        "upstreams:\n  deepseek: {url: http://x, key_env: TEST_PANEL_KEY, ceiling: normal, models: {sonnet: m}}\n", encoding="utf-8")
    calls: list[tuple] = []
    monkeypatch.setattr(panel, "direct_mode", lambda: True)
    monkeypatch.setattr(panel, "server_status", lambda: {"up": False})
    monkeypatch.setattr(panel, "port_open", lambda port: False)
    monkeypatch.setattr(panel, "set_role", lambda role: calls.append(("role", role)) or role)
    monkeypatch.setattr(panel, "set_direct_mode", lambda direct: calls.append(("direct", direct)) or "direct")
    with pytest.raises(ValueError, match="private"):
        panel.apply_preset("deepseek", str(tmp_path))
    assert calls == [] and panel.upstream.active_mode() == "claude"
    (tmp_path / ".foremankit").mkdir()
    (tmp_path / ".foremankit" / "sensitivity").write_text("normal", encoding="utf-8")
    result = panel.apply_preset("deepseek", str(tmp_path))
    assert calls == [("direct", False)] and result["reload"] is True
    assert panel.upstream.active_mode() == "deepseek"


def test_sensitivity_refusal_names_the_level_the_user_could_declare_and_marking_unblocks_it(monkeypatch, tmp_path: Path):
    """The panel answer for a too-sensitive folder carries `mark`; after mark_folder the same preset plans fine."""
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TEST_PANEL_KEY", "k")
    (tmp_path / "orchestrator.yaml").write_text(
        "upstreams:\n  deepseek: {url: http://x, key_env: TEST_PANEL_KEY, ceiling: normal, models: {sonnet: m}}\n", encoding="utf-8")
    monkeypatch.setattr(panel, "direct_mode", lambda: True)
    monkeypatch.setattr(panel, "server_status", lambda: {"up": False})
    monkeypatch.setattr(panel, "port_open", lambda port: False)
    refused = panel.guarded(panel.plan_preset, "deepseek", str(tmp_path))
    assert refused["mark"] == "normal" and "private" in refused["error"]
    panel.mark_folder("normal", str(tmp_path))
    assert "error" not in panel.guarded(panel.plan_preset, "deepseek", str(tmp_path))
    with pytest.raises(ValueError):
        panel.mark_folder("secret", str(tmp_path))


def test_default_model_follows_the_preset_and_the_users_own_choice_comes_back(monkeypatch, tmp_path: Path):
    """A non-claude preset sets `model` to the ForemanKit picker entry, keeps other keys, and claude restores the old value."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "sonnet", "theme": "dark"}), encoding="utf-8")
    monkeypatch.setenv("FOREMANKIT_CLAUDE_SETTINGS", str(settings))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "orchestrator.yaml").write_text("local_models: [claude-sonnet-5]\n", encoding="utf-8")
    assert "claude-sonnet-5" in panel.sync_default_model("deepseek")
    assert json.loads(settings.read_text(encoding="utf-8")) == {"model": "claude-sonnet-5", "theme": "dark"}
    assert panel.sync_default_model("gemini") == ""            # already ours: the backup of the user's value must survive
    panel.sync_default_model("claude")
    assert json.loads(settings.read_text(encoding="utf-8")) == {"model": "sonnet", "theme": "dark"}
    assert panel.sync_default_model("claude") == ""
