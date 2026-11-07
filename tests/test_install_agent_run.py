"""Verify opt-in agent wrappers and Claude permissions in an isolated home.
Existing installer behavior must remain unchanged without the flag.
"""
import json
from pathlib import Path

import pytest
from test_install import INSTALL


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every installer path inside a temporary home."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(INSTALL, "CLAUDE_HOME", tmp_path / ".claude")
    monkeypatch.setattr(INSTALL, "STATE_DIR", tmp_path / "state")
    return tmp_path


def settings(home: Path) -> dict:
    """Read the settings created in the temporary home."""
    return json.loads((home / ".claude/settings.json").read_text(encoding="utf-8"))


def seed_settings(home: Path) -> bytes:
    """Preserve foreign settings and an existing worker permission."""
    target = home / ".claude/settings.json"
    target.parent.mkdir()
    content = b'{"permissions":{"allow":["Bash(existing:*)","Bash(fk agent-run:*)"]}}'
    target.write_bytes(content)
    return content


def test_dry_agent_setup_writes_nothing(home: Path) -> None:
    """Preview creates neither wrappers, settings nor snapshot."""
    INSTALL.Installer(dry=True).install_agent_run(True)
    assert list(home.iterdir()) == []


@pytest.mark.parametrize("name", ["fk", "fk.cmd"])
def test_opt_in_creates_wrappers(home: Path, name: str) -> None:
    """Both shells receive a wrapper forwarding arguments to the venv."""
    INSTALL.Installer(dry=False).install_agent_run(True)
    assert "foremankit" in (home / ".local/bin" / name).read_text(encoding="utf-8")


def test_permissions_are_complete_and_idempotent(home: Path) -> None:
    """A repeated installation preserves foreign rules without duplicates."""
    seed_settings(home)
    installer = INSTALL.Installer(dry=False)
    installer.install_agent_run(True)
    installer.install_agent_run(True)
    expected = {f"{shell}(fk {command}:*)" for shell in ("Bash", "PowerShell")
                for command in INSTALL.AGENT_RUN_COMMANDS} | {"Bash(existing:*)"}
    allowed = settings(home)["permissions"]["allow"]
    assert set(allowed) == expected
    assert len(allowed) == len(expected)
    assert "agent-merge" not in json.dumps(allowed)


def test_restore_removes_new_wrappers(home: Path) -> None:
    """Exact restore removes both opt-in files that did not exist."""
    installer = INSTALL.Installer(dry=False)
    installer.install_agent_run(True)
    installer.restore_original()
    assert not (home / ".local/bin/fk").exists()
    assert not (home / ".local/bin/fk.cmd").exists()


def test_restore_keeps_original_settings_bytes(home: Path) -> None:
    """Existing permissions survive byte-for-byte restoration."""
    original = seed_settings(home)
    installer = INSTALL.Installer(dry=False)
    installer.install_agent_run(True)
    installer.restore_original()
    assert (home / ".claude/settings.json").read_bytes() == original


def test_without_opt_in_settings_are_untouched(home: Path) -> None:
    """The default agent setup prints a hint without writing settings."""
    original = seed_settings(home)
    INSTALL.Installer(dry=False).install_agent_run(False)
    assert (home / ".claude/settings.json").read_bytes() == original
    assert not (home / ".local").exists()


def test_parser_accepts_agent_run_and_dry_alias() -> None:
    """Both requested opt-in and preview flags reach the installer."""
    arguments = INSTALL.build_parser().parse_args(["--agent-run", "--dry"])
    assert arguments.agent_run and arguments.dry_run


@pytest.mark.parametrize("name, forwarding", [("fk", '"$@"'), ("fk.cmd", "%*")])
def test_wrappers_forward_all_arguments(home: Path, name: str, forwarding: str) -> None:
    """Arguments remain separate when forwarded to the installed executable."""
    INSTALL.Installer(dry=False).install_agent_run(True)
    wrapper = (home / ".local/bin" / name).read_text(encoding="utf-8")
    assert forwarding in wrapper
    assert str(INSTALL.ROOT / ".venv").replace("\\", "/") in wrapper.replace("\\", "/")


def test_restore_preserves_an_existing_wrapper(home: Path) -> None:
    """The optional wrapper participates in the original-byte snapshot."""
    wrapper = home / ".local/bin/fk"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_bytes(b"original wrapper")
    installer = INSTALL.Installer(dry=False)
    installer.install_agent_run(True)
    installer.restore_original()
    assert wrapper.read_bytes() == b"original wrapper"


@pytest.mark.parametrize("enabled", [False, True])
def test_install_dispatches_the_opt_in_flag(
    home: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool,
) -> None:
    """A minimal installation activates wrappers only when requested."""
    arguments = INSTALL.build_parser().parse_args(["--skip-claude", "--no-panel"])
    arguments.agent_run = enabled
    monkeypatch.setattr(INSTALL.Installer, "make_venv", lambda self: home / "python")
    monkeypatch.setattr(INSTALL.Installer, "migrate_legacy", lambda self: None)
    INSTALL.run_install(INSTALL.Installer(dry=False), arguments)
    assert (home / ".local/bin/fk").exists() == enabled


def test_posix_wrapper_uses_lf_on_every_platform(home: Path) -> None:
    """A Windows installation must still produce a valid POSIX shebang."""
    INSTALL.Installer(dry=False).install_agent_run(True)
    assert b"\r" not in (home / ".local/bin/fk").read_bytes()
