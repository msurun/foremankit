"""Tests for the agentkit -> foremankit migration path.

The project was renamed. A machine with an old "agentkit" installation must be migrated by the
new installer. Every test runs inside a tmp_path sandbox: the real ``~/.claude``, the real
``~/.config/foremankit`` and the real VS Code user folder are never read or written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location("foremankit_install", _REPO_ROOT / "install.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load install.py from {_REPO_ROOT}")
INSTALL: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(INSTALL)

# The install_local_orchestrator tests call the method with this placeholder: no real interpreter
# is executed, the method only stores the value in the settings it writes.
FAKE_PYTHON = Path("python")


@dataclass(frozen=True)
class Sandbox:
    """The throwaway folders a test may look at; the returned state is immutable."""

    claude: Path
    vscode: Path
    state: Path
    legacy_config: Path
    config: Path


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    """Point install.py at tmp_path folders and return them.

    CLAUDE_HOME, STATE_DIR and the VS Code user folder all live under tmp_path, so the developer's
    real configuration stays untouched.
    """
    claude_home = tmp_path / "claude"
    state_dir = tmp_path / "config" / "install-state"
    vscode_dir = tmp_path / "vscode"
    legacy_config = tmp_path / "legacy-config"
    monkeypatch.setattr(INSTALL, "CLAUDE_HOME", claude_home)
    monkeypatch.setattr(INSTALL, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(INSTALL, "STATE_DIR", state_dir)
    monkeypatch.setattr(INSTALL.Installer, "vscode_user_dir", staticmethod(lambda: vscode_dir))
    monkeypatch.setattr(INSTALL, "LEGACY_CONFIG_DIR", legacy_config)
    claude_home.mkdir(parents=True)
    vscode_dir.mkdir(parents=True)
    legacy_config.mkdir(parents=True)
    return Sandbox(claude=claude_home, vscode=vscode_dir, state=state_dir,
                   legacy_config=legacy_config, config=state_dir.parent)


def write_file(path: Path, content: bytes) -> Path:
    """Write ``content`` at ``path``, creating parent directories, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_migrate_config_dir_moves_the_legacy_folder_and_keeps_the_bytes_of_its_env(
    sandbox: Sandbox,
) -> None:
    """migrate_config_dir moves the legacy folder and keeps the bytes of its .env."""
    write_file(sandbox.legacy_config / ".env", b"AGENTKIT_API_KEY=old\n")

    installer = INSTALL.Installer(dry=False)
    moved = installer.migrate_config_dir()

    assert moved is True
    assert (sandbox.config / ".env").read_bytes() == b"AGENTKIT_API_KEY=old\n"
    assert not sandbox.legacy_config.exists()


def test_migrate_config_dir_returns_false_when_there_is_no_legacy_folder(
    sandbox: Sandbox,
) -> None:
    """migrate_config_dir returns False when there is no legacy folder."""
    shutil.rmtree(sandbox.legacy_config)
    installer = INSTALL.Installer(dry=False)
    moved = installer.migrate_config_dir()

    assert moved is False


def test_migrate_config_dir_copies_only_missing_files_when_both_folders_exist_and_never_overwrites(
    sandbox: Sandbox,
) -> None:
    """migrate_config_dir copies only missing files when both folders exist and never overwrites."""
    write_file(sandbox.legacy_config / ".env", b"AGENTKIT_API_KEY=old\n")
    write_file(sandbox.config / ".env", b"FOREMANKIT_API_KEY=new\n")
    write_file(sandbox.legacy_config / "state.json", b"{}")

    installer = INSTALL.Installer(dry=False)
    moved = installer.migrate_config_dir()

    assert moved is True
    assert (sandbox.config / ".env").read_bytes() == b"FOREMANKIT_API_KEY=new\n"
    assert (sandbox.config / "state.json").read_bytes() == b"{}"
    assert sandbox.legacy_config.exists()


def test_migrate_config_dir_in_dry_run_mode_moves_nothing(
    sandbox: Sandbox,
) -> None:
    """migrate_config_dir in dry-run mode moves nothing."""
    write_file(sandbox.legacy_config / ".env", b"AGENTKIT_API_KEY=old\n")

    installer = INSTALL.Installer(dry=True)
    moved = installer.migrate_config_dir()

    assert moved is True
    assert sandbox.legacy_config.exists()
    assert not (sandbox.config / ".env").exists()


def test_remove_legacy_rules_block_removes_the_old_block_keeps_the_user_text_and_leaves_a_backup(
    sandbox: Sandbox,
) -> None:
    """remove_legacy_rules_block removes the old block, keeps the user text and leaves a backup."""
    content = "Hello\n<!-- BEGIN agentkit -->\nold stuff\n<!-- END agentkit -->\nWorld\n"
    write_file(sandbox.claude / "CLAUDE.md", content.encode("utf-8"))

    installer = INSTALL.Installer(dry=False)
    installer.remove_legacy_rules_block()

    restored = (sandbox.claude / "CLAUDE.md").read_text(encoding="utf-8")
    assert "old stuff" not in restored
    assert "Hello" in restored
    assert "World" in restored
    assert "<!-- BEGIN agentkit -->" not in restored
    assert "<!-- END agentkit -->" not in restored
    backups = list(sandbox.claude.glob("CLAUDE.md.bak-foremankit-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == content


def test_remove_legacy_rules_block_keeps_a_block_of_the_current_name(
    sandbox: Sandbox,
) -> None:
    """remove_legacy_rules_block keeps a block of the current name."""
    content = "Hello\n<!-- BEGIN foremankit -->\nnew stuff\n<!-- END foremankit -->\nWorld\n"
    write_file(sandbox.claude / "CLAUDE.md", content.encode("utf-8"))

    installer = INSTALL.Installer(dry=False)
    installer.remove_legacy_rules_block()

    restored = (sandbox.claude / "CLAUDE.md").read_text(encoding="utf-8")
    assert restored == content


def test_remove_legacy_tasks_removes_the_old_labels_and_keeps_a_user_task(
    sandbox: Sandbox,
) -> None:
    """remove_legacy_tasks removes the old labels and keeps a user task."""
    tasks = [
        {"label": "AgentKit: router 8081", "type": "shell", "command": "old"},
        {"label": "User task", "type": "shell", "command": "user"},
    ]
    write_file(
        sandbox.vscode / "tasks.json",
        json.dumps({"version": "2.0.0", "tasks": tasks}).encode("utf-8"),
    )

    installer = INSTALL.Installer(dry=False)
    installer.remove_legacy_tasks()

    data = json.loads((sandbox.vscode / "tasks.json").read_text(encoding="utf-8"))
    assert data["tasks"] == [{"label": "User task", "type": "shell", "command": "user"}]


def test_remove_legacy_mcp_without_the_claude_cli_does_nothing(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """remove_legacy_mcp without the claude CLI does nothing."""
    monkeypatch.setattr(INSTALL.shutil, "which", lambda _: None)

    installer = INSTALL.Installer(dry=False)
    installer.remove_legacy_mcp()

    # No exception, no side effects on files
    assert not (sandbox.claude / "settings.json").exists()


def test_migrate_legacy_moves_the_config_folder_and_removes_the_old_rules_block_in_one_call(
    sandbox: Sandbox,
) -> None:
    """migrate_legacy moves the config folder and removes the old rules block in one call."""
    write_file(sandbox.legacy_config / ".env", b"AGENTKIT_API_KEY=old\n")
    content = "Hello\n<!-- BEGIN agentkit -->\nold stuff\n<!-- END agentkit -->\nWorld\n"
    write_file(sandbox.claude / "CLAUDE.md", content.encode("utf-8"))

    installer = INSTALL.Installer(dry=False)
    installer.migrate_legacy()

    assert (sandbox.config / ".env").read_bytes() == b"AGENTKIT_API_KEY=old\n"
    assert not sandbox.legacy_config.exists()
    restored = (sandbox.claude / "CLAUDE.md").read_text(encoding="utf-8")
    assert "old stuff" not in restored
    assert "Hello" in restored
    assert "World" in restored


def test_earliest_backup_finds_a_backup_with_the_legacy_mark(
    sandbox: Sandbox,
) -> None:
    """_earliest_backup finds a backup with the legacy mark."""
    write_file(sandbox.claude / "CLAUDE.md", b"current\n")
    write_file(sandbox.claude / "CLAUDE.md.bak-agentkit-20240101-000000", b"legacy\n")

    installer = INSTALL.Installer(dry=False)
    backup = installer._earliest_backup("CLAUDE.md")

    assert backup is not None
    assert backup.read_bytes() == b"legacy\n"


def test_leftovers_reports_a_legacy_rules_block(
    sandbox: Sandbox,
) -> None:
    """leftovers reports a legacy rules block."""
    content = "Hello\n<!-- BEGIN agentkit -->\nold stuff\n<!-- END agentkit -->\nWorld\n"
    write_file(sandbox.claude / "CLAUDE.md", content.encode("utf-8"))

    installer = INSTALL.Installer(dry=False)
    left = installer.leftovers()

    assert any(INSTALL.LEGACY_BEGIN in trace for trace in left)


def test_adopt_legacy_env_copies_only_the_variables_whose_new_name_is_not_set_and_returns_their_new_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """adopt_legacy_env copies only the variables whose new name is not set and returns their new names."""
    import foremankit

    environ: MutableMapping[str, str] = {}
    environ["AGENTKIT_API_KEY"] = "old"
    environ["AGENTKIT_OTHER"] = "other"
    environ["FOREMANKIT_API_KEY"] = "new"

    created = foremankit.adopt_legacy_env(environ)

    assert created == ["FOREMANKIT_OTHER"]
    assert environ["FOREMANKIT_API_KEY"] == "new"
    assert environ["FOREMANKIT_OTHER"] == "other"
    assert environ["AGENTKIT_API_KEY"] == "old"
    assert environ["AGENTKIT_OTHER"] == "other"
