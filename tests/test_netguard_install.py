from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from foremankit.netguard import PrivacyError, assert_private_ok, check_host


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:8080", "http://10.1.2.3:11434",
                                 "http://172.20.0.5:11434", "http://192.168.0.10:11434",
                                 "http://100.101.102.103:11434", "http://[::1]:11434"])
def test_private_addresses_allowed(url):
    assert check_host(url, resolver=lambda h: ["8.8.8.8"])[0]


@pytest.mark.parametrize("url", ["http://8.8.8.8:11434", "http://172.32.0.1:11434", "http://100.128.0.1:1"])
def test_public_addresses_refused(url):
    assert not check_host(url)[0]


def test_hostname_resolving_to_public_is_refused():
    ok, reason = check_host("http://box.example.com:11434", resolver=lambda h: ["100.64.1.1", "52.1.1.1"])
    assert not ok and "52.1.1.1" in reason


def test_tailnet_hostname_allowed():
    assert check_host("http://gpu-box.tailnet.ts.net:11434", resolver=lambda h: ["100.90.1.2"])[0]


def test_unresolvable_fails_closed():
    def boom(host):
        raise OSError("nxdomain")
    assert not check_host("http://nope.invalid", resolver=boom)[0]


def test_assert_private_ok_override_warns():
    warnings = []
    assert_private_ok("http://8.8.8.8", allow_public_endpoint=True, warn=warnings.append)
    assert warnings
    with pytest.raises(PrivacyError):
        assert_private_ok("http://8.8.8.8")


def _installer():
    path = Path(__file__).resolve().parent.parent / "install.py"
    spec = importlib.util.spec_from_file_location("foremankit_install", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_claude_md_block_is_idempotent():
    inst = _installer()
    merge = inst.Installer.merged_claude_md
    base = "# My rules\n\nkeep me\n"
    once = merge(base, "rule A")
    assert merge(once, "rule A") == once
    assert once.count(inst.BEGIN) == 1 and "keep me" in once
    updated = merge(once, "rule B")
    assert "rule B" in updated and "rule A" not in updated
    removed = merge(updated, None)
    assert inst.BEGIN not in removed and "keep me" in removed


def test_notify_hook_merge_is_idempotent_and_keeps_other_settings(tmp_path):
    import json
    inst = _installer()
    inst.CLAUDE_HOME = tmp_path
    settings = tmp_path / "settings.json"
    settings.write_bytes(b"\xef\xbb\xbf" + json.dumps(
        {"theme": "dark", "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "x"}]}]}}
    ).encode())                                     # BOM on purpose: the installer must read it and drop it
    py = tmp_path / ".venv" / "Scripts" / "python.exe"
    installer = inst.Installer(dry=False)
    installer.install_hooks(py)
    installer.install_hooks(py)
    raw = settings.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    data = json.loads(raw)
    assert data["theme"] == "dark"
    pre = data["hooks"]["PreToolUse"]
    assert len(pre) == 6 and pre[0]["matcher"] == "Bash"      # the user's own + notify + orchestrator_guard + 3 guard hooks
    assert sum(inst.GUARD_HOOK_MARK in h["command"] for e in pre for h in e["hooks"]) == 1
    assert sum(inst.HOOK_MARK in h["command"] for e in data["hooks"]["PostToolUse"] for h in e["hooks"]) == 1
    installer.install_hooks(py, remove=True)
    data = json.loads(settings.read_bytes())
    assert "PostToolUse" not in data["hooks"] and len(data["hooks"]["PreToolUse"]) == 1


def test_session_length_hook_registers_alone_keeps_others_and_removes_cleanly(tmp_path):
    import json
    inst = _installer()
    inst.CLAUDE_HOME = tmp_path
    settings = tmp_path / "settings.json"
    other = {"hooks": [{"type": "command", "command": "keep-me"}]}
    settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [other]}}), encoding="utf-8")
    py = tmp_path / ".venv" / "Scripts" / "python.exe"
    installer = inst.Installer(dry=False)
    installer.install_hooks(py, only="session")
    installer.install_hooks(py, only="session")                  # idempotent
    data = json.loads(settings.read_bytes())
    assert set(data["hooks"]) == {"UserPromptSubmit"}            # the notify hooks were NOT added
    entries = data["hooks"]["UserPromptSubmit"]
    assert entries[0] == other and len(entries) == 2
    assert sum(inst.SESSION_HOOK_MARK in h["command"] for e in entries for h in e["hooks"]) == 1
    installer.install_hooks(py, remove=True, only="session")
    assert json.loads(settings.read_bytes())["hooks"]["UserPromptSubmit"] == [other]


def test_delegation_nudge_hook_registers_on_two_events_and_removes_cleanly(tmp_path):
    import json
    inst = _installer()
    inst.CLAUDE_HOME = tmp_path
    settings = tmp_path / "settings.json"
    other = {"hooks": [{"type": "command", "command": "keep-me"}]}
    settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [other]}}), encoding="utf-8")
    py = tmp_path / ".venv" / "Scripts" / "python.exe"
    installer = inst.Installer(dry=False)
    installer.install_hooks(py, only="nudge")
    installer.install_hooks(py, only="nudge")                    # idempotent
    hooks = json.loads(settings.read_bytes())["hooks"]
    prompt = [e for e in hooks["UserPromptSubmit"] if any(inst.NUDGE_HOOK_MARK in h["command"] for h in e["hooks"])]
    post = [e for e in hooks["PostToolUse"] if any(inst.NUDGE_HOOK_MARK in h["command"] for h in e["hooks"])]
    assert hooks["UserPromptSubmit"][0] == other and len(prompt) == 1
    assert len(post) == 1 and post[0]["matcher"] == "Write|Edit|Bash"
    installer.install_hooks(py, remove=True, only="nudge")
    after = json.loads(settings.read_bytes())["hooks"]
    assert after["UserPromptSubmit"] == [other] and "PostToolUse" not in after
