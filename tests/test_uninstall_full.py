"""Tests for the full uninstall path in ``install.py``: VS Code tracking, leftovers, purge.

install.py lives in the repository root and is not a package, so it is loaded by path through
importlib. Every test runs inside a tmp_path sandbox: the real ``~/.claude``, the real
``~/.config/foremankit`` and the real VS Code user folder are never read or written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location("foremankit_install", _REPO_ROOT / "install.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load install.py from {_REPO_ROOT}")
INSTALL: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(INSTALL)

CLAUDE_MD = "CLAUDE.md"
SETTINGS_JSON = "settings.json"
VSCODE_SETTINGS_JSON = "vscode/settings.json"
VSCODE_TASKS_JSON = "vscode/tasks.json"

# settings.json before the installation: a plain user file whose bytes must survive a restore.
SETTINGS_JSON_BYTES = b'{"permissions": {"allow": ["Bash"]}}\n'
VSCODE_SETTINGS_JSON_BYTES = b'{"editor.fontSize": 14}\n'
VSCODE_TASKS_JSON_BYTES = b'{"version": "2.0.0", "tasks": []}\n'
ADDED_MARKER = "vscode-added.json"

# The install_local_orchestrator tests call the method with this placeholder: no real interpreter
# is executed, the method only stores the value in the settings it writes.
FAKE_PYTHON = Path("python")

# A command line of a gateway started from this repository, used by select_foremankit_processes.
GATEWAY_MARKER = "foremankit-gateway"
LLAMA_MARKER = "llama-server"


@dataclass(frozen=True)
class Sandbox:
    """The throwaway folders a test may look at; the returned state is immutable."""

    claude: Path
    vscode: Path
    state: Path


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    """Point install.py at tmp_path folders and return them.

    CLAUDE_HOME, STATE_DIR and the VS Code user folder all live under tmp_path, so the developer's
    real configuration stays untouched.
    """
    claude_home = tmp_path / "claude"
    state_dir = tmp_path / "state"
    vscode_dir = tmp_path / "vscode"
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(INSTALL, "CLAUDE_HOME", claude_home)
    monkeypatch.setattr(INSTALL, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(INSTALL, "STATE_DIR", state_dir)
    monkeypatch.setattr(INSTALL.Installer, "vscode_user_dir", staticmethod(lambda: vscode_dir))
    claude_home.mkdir(parents=True)
    vscode_dir.mkdir(parents=True)
    return Sandbox(claude=claude_home, vscode=vscode_dir, state=state_dir)


def write_file(path: Path, content: bytes) -> Path:
    """Write ``content`` at ``path``, creating parent directories, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def full_install(sandbox: Sandbox) -> None:
    """Run the real install steps that touch files, so leftovers() has something to report."""
    installer = INSTALL.Installer(dry=False)
    installer.take_snapshot(INSTALL.VSCODE_TRACKED)
    installer.install_agents()
    installer.install_rules()
    installer.install_hooks(FAKE_PYTHON)
    installer.install_local_orchestrator(FAKE_PYTHON)


def full_uninstall() -> None:
    """Run the real uninstall steps that remove the files full_install() created."""
    installer = INSTALL.Installer(dry=False)
    installer.remove_agents()
    installer.install_rules(remove=True)
    installer.install_hooks(FAKE_PYTHON, remove=True)
    installer.install_local_orchestrator(FAKE_PYTHON, remove=True)
    installer.remove_empty_created_files()


def settings_hook_commands(path: Path) -> str:
    """Return every hook command stored in a Claude settings.json as one string."""
    data = json.loads(path.read_text(encoding="utf-8"))
    commands = [hook["command"] for entries in data.get("hooks", {}).values()
                for entry in entries for hook in entry.get("hooks", [])]
    return " ".join(commands)


def test_tracked_path_maps_vscode_keys_and_claude_keys_to_their_own_folder(sandbox: Sandbox) -> None:
    """tracked_path resolves vscode/ keys under the VS Code folder and plain keys under CLAUDE_HOME."""
    installer = INSTALL.Installer(dry=False)

    vscode_path = installer.tracked_path(VSCODE_SETTINGS_JSON)
    claude_path = installer.tracked_path(CLAUDE_MD)

    assert vscode_path == sandbox.vscode / "settings.json"
    assert claude_path == sandbox.claude / CLAUDE_MD


def test_take_snapshot_with_extra_keys_records_the_vscode_files(sandbox: Sandbox) -> None:
    """take_snapshot(extra=VSCODE_TRACKED) records the VS Code keys and copies settings.json."""
    write_file(sandbox.vscode / SETTINGS_JSON, VSCODE_SETTINGS_JSON_BYTES)

    INSTALL.Installer(dry=False).take_snapshot(INSTALL.VSCODE_TRACKED)

    snapshot = json.loads((sandbox.state / "snapshot.json").read_text(encoding="utf-8"))
    copied = (sandbox.state / "files" / VSCODE_SETTINGS_JSON).read_bytes()
    assert snapshot["existed"][VSCODE_SETTINGS_JSON] is True
    assert copied == VSCODE_SETTINGS_JSON_BYTES


def test_take_snapshot_later_extra_keys_are_added_without_touching_recorded_keys(
    sandbox: Sandbox,
) -> None:
    """A second take_snapshot adds the VS Code keys and keeps the Claude record as it was."""
    write_file(sandbox.claude / CLAUDE_MD, b"# mine\n")
    INSTALL.Installer(dry=False).take_snapshot()
    write_file(sandbox.vscode / SETTINGS_JSON, VSCODE_SETTINGS_JSON_BYTES)

    INSTALL.Installer(dry=False).take_snapshot(INSTALL.VSCODE_TRACKED)

    snapshot = json.loads((sandbox.state / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["existed"][CLAUDE_MD] is True
    assert snapshot["existed"][VSCODE_SETTINGS_JSON] is True
    assert snapshot["existed"][VSCODE_TASKS_JSON] is False


def test_restore_original_brings_back_the_exact_bytes_of_a_vscode_file(sandbox: Sandbox) -> None:
    """restore_original rewrites a VS Code settings.json that existed before, byte for byte."""
    write_file(sandbox.vscode / SETTINGS_JSON, VSCODE_SETTINGS_JSON_BYTES)
    installer = INSTALL.Installer(dry=False)
    installer.take_snapshot(INSTALL.VSCODE_TRACKED)
    write_file(sandbox.vscode / SETTINGS_JSON, b'{"editor.fontSize": 40}\n')

    installer.restore_original()

    assert (sandbox.vscode / SETTINGS_JSON).read_bytes() == VSCODE_SETTINGS_JSON_BYTES


def test_is_blank_json_accepts_an_empty_object_and_an_empty_task_list_but_not_a_real_task() -> None:
    """is_blank_json is True for {} and for the bare tasks file, False once a real task exists."""
    installer = INSTALL.Installer(dry=False)

    assert installer.is_blank_json({}) is True
    assert installer.is_blank_json({"version": "2.0.0", "tasks": []}) is True
    assert installer.is_blank_json({"version": "2.0.0", "tasks": [{"label": "my task"}]}) is False


def test_remove_empty_created_files_deletes_a_created_blank_tasks_and_keeps_a_user_task(
    sandbox: Sandbox,
) -> None:
    """A created blank tasks.json goes away; one holding a user task stays."""
    installer = INSTALL.Installer(dry=False)
    installer.take_snapshot(INSTALL.VSCODE_TRACKED)
    blank_tasks = write_file(sandbox.vscode / "tasks.json", VSCODE_TASKS_JSON_BYTES)

    installer.remove_empty_created_files()
    blank_was_deleted = not blank_tasks.exists()

    tasks_with_user_entry = b'{"version": "2.0.0", "tasks": [{"label": "my task"}]}\n'
    user_tasks = write_file(sandbox.vscode / "tasks.json", tasks_with_user_entry)
    installer.remove_empty_created_files()

    assert blank_was_deleted
    assert user_tasks.read_bytes() == tasks_with_user_entry


def test_install_local_orchestrator_records_allow_automatic_tasks_and_uninstall_removes_it(
    sandbox: Sandbox,
) -> None:
    """The install writes the marker and the value; remove=True takes the value out again."""
    installer = INSTALL.Installer(dry=False)

    installer.install_local_orchestrator(FAKE_PYTHON)
    marker = json.loads((sandbox.state / ADDED_MARKER).read_text(encoding="utf-8"))
    value_after_install = json.loads(
        (sandbox.vscode / SETTINGS_JSON).read_text(encoding="utf-8"))["task.allowAutomaticTasks"]

    installer.install_local_orchestrator(FAKE_PYTHON, remove=True)
    settings = json.loads((sandbox.vscode / SETTINGS_JSON).read_text(encoding="utf-8"))

    assert marker == {"allowAutomaticTasks": True}
    assert value_after_install == "auto"
    assert "task.allowAutomaticTasks" not in settings


def test_uninstall_keeps_a_user_value_of_allow_automatic_tasks(sandbox: Sandbox) -> None:
    """A task.allowAutomaticTasks the user set before the installation is never removed."""
    write_file(sandbox.vscode / SETTINGS_JSON, b'{"task.allowAutomaticTasks": "on"}\n')
    installer = INSTALL.Installer(dry=False)
    installer.install_local_orchestrator(FAKE_PYTHON)

    installer.install_local_orchestrator(FAKE_PYTHON, remove=True)

    settings = json.loads((sandbox.vscode / SETTINGS_JSON).read_text(encoding="utf-8"))
    assert settings["task.allowAutomaticTasks"] == "on"


def test_leftovers_is_empty_after_a_full_install_followed_by_a_full_uninstall(sandbox: Sandbox) -> None:
    """A full install and the matching uninstall leave no trace for leftovers()."""
    full_install(sandbox)

    full_uninstall()

    assert INSTALL.Installer(dry=False).leftovers() == []


def test_leftovers_lists_every_trace_a_full_install_leaves_behind(sandbox: Sandbox) -> None:
    """After a full install leftovers() mentions the agent, rules block, hook, gateway env and task."""
    full_install(sandbox)

    report = " ".join(INSTALL.Installer(dry=False).leftovers())

    assert INSTALL.AGENTS[0] in report
    assert INSTALL.BEGIN in report
    assert INSTALL.HOOK_MARK in report
    assert INSTALL.GATEWAY_ENV[0] in report
    assert INSTALL.LOGON_TASK in report


def test_select_foremankit_processes_keeps_this_folder_gateway_and_drops_the_rest() -> None:
    """Only the gateway of this folder survives: another folder, an unrelated pid and own_pid do not."""
    root = Path("/home/dev/foremankit")
    own_pid = 4242
    rows = [
        (11, f"{root}/.venv/bin/foremankit-gateway"),
        (12, "/usr/bin/python3 -c print('hello')"),
        (13, "/home/other/foremankit/.venv/bin/foremankit-gateway"),
        (own_pid, f"{root}/.venv/bin/foremankit-gateway"),
    ]

    result = INSTALL.select_foremankit_processes(rows, root, own_pid)

    assert result == [(11, f"{root}/.venv/bin/foremankit-gateway")]


def test_purge_state_removes_the_state_folder_and_leaves_a_sibling_folder(sandbox: Sandbox) -> None:
    """purge_state deletes STATE_DIR with its contents and touches no other folder."""
    write_file(sandbox.state / "snapshot.json", b"{}\n")
    sibling = write_file(sandbox.state.parent / "sibling" / "keep.txt", b"keep\n")

    INSTALL.Installer(dry=False).purge_state()

    assert not sandbox.state.exists()
    assert sibling.exists()


def test_purge_state_in_dry_run_mode_removes_nothing(sandbox: Sandbox) -> None:
    """purge_state with dry=True reports but leaves the state folder in place."""
    write_file(sandbox.state / "snapshot.json", b"{}\n")

    INSTALL.Installer(dry=True).purge_state()

    assert (sandbox.state / "snapshot.json").exists()


def _fake_run(stdout: str, returncode: int = 0) -> Any:
    """A stand-in for subprocess.run that returns the given process listing."""

    def fake(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    return fake


def test_list_processes_parses_pid_and_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each listing line becomes (pid, command line); garbage lines are skipped."""
    listing = "  101 /usr/bin/python3 -m foremankit.gateway\nnot-a-pid something\n202 bash\n7\n"
    monkeypatch.setattr(INSTALL.subprocess, "run", _fake_run(listing))

    assert INSTALL.list_processes() == [(101, "/usr/bin/python3 -m foremankit.gateway"), (202, "bash")]


def test_list_processes_returns_nothing_when_the_listing_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing process listing means no processes, never an exception."""
    monkeypatch.setattr(INSTALL.subprocess, "run", _fake_run("101 python", returncode=1))

    assert INSTALL.list_processes() == []


def test_purge_state_also_removes_the_gateway_files(sandbox: Sandbox) -> None:
    """--purge deletes the gateway context counter and its router logs, and only those."""
    temp = INSTALL.TEMP_DIR
    temp.mkdir()
    counter = sandbox.state.parent / "gateway_context.json"
    counter.write_text("{}")
    (temp / "router.log").write_text("log")
    (temp / "router.err").write_text("err")
    other = temp / "other.log"
    other.write_text("keep")

    INSTALL.Installer(dry=False).purge_state()

    assert not counter.exists()
    assert not (temp / "router.log").exists()
    assert not (temp / "router.err").exists()
    assert other.exists()


def _stop_args(dry_run: bool = False) -> Any:
    """The parsed command line of `--uninstall --stop-services`."""
    return argparse.Namespace(stop_services=True, dry_run=dry_run)


def _count_stop_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace Installer.stop_services by a counter so no real process is touched."""
    calls: list[int] = []

    def fake_stop(_self: Any) -> int:
        calls.append(1)
        return 1

    monkeypatch.setattr(INSTALL.Installer, "stop_services", fake_stop)
    return calls


def test_gateway_references_finds_the_vscode_gateway_settings_after_install(
    sandbox: Sandbox,
) -> None:
    """After --local-orchestrator the VS Code settings point Claude Code at the gateway."""
    full_install(sandbox)

    assert INSTALL.Installer(dry=False).gateway_references()


def test_gateway_references_is_empty_after_the_uninstall(sandbox: Sandbox) -> None:
    """Once the VS Code entries are removed nothing points at the gateway any more."""
    full_install(sandbox)
    installer = INSTALL.Installer(dry=False)
    installer.install_local_orchestrator(Path("python"), remove=True)

    assert installer.gateway_references() == []


def test_gateway_references_finds_a_base_url_in_the_claude_settings_env(sandbox: Sandbox) -> None:
    """A gateway URL placed by hand in ~/.claude/settings.json (env) counts as a reference."""
    settings = {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8081"}}
    (sandbox.claude / "settings.json").write_text(json.dumps(settings))

    assert INSTALL.Installer(dry=False).gateway_references()


def test_the_gateway_is_not_stopped_while_something_still_points_at_it(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping the gateway first would cut Claude Code off from the cloud."""
    full_install(sandbox)
    stop_calls = _count_stop_calls(monkeypatch)

    INSTALL.stop_services_safely(INSTALL.Installer(dry=False), _stop_args())

    assert stop_calls == []


def test_the_gateway_is_stopped_once_nothing_points_at_it(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the settings cleaned up the gateway may be stopped."""
    stop_calls = _count_stop_calls(monkeypatch)

    INSTALL.stop_services_safely(INSTALL.Installer(dry=False), _stop_args())

    assert stop_calls == [1]
