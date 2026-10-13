"""Tests for the install/uninstall bookkeeping in ``install.py``.

install.py lives in the repository root and is not a package, so it is loaded by path
through importlib. Every test runs inside a tmp_path sandbox: the real ``~/.claude``
is never read or written.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
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
OPUS_ADVISOR = "agents/opus-advisor.md"
DELEGATOR = "agents/delegator.md"

# CLAUDE.md with a non-ASCII character and CRLF line endings: byte equality must survive a
# snapshot and restore round trip, so the fixture content must be awkward on purpose.
CLAUDE_MD_BYTES = b"# ForemanKit \xe2\x80\x94 caf\xc3\xa9\r\n\r\nrules\r\n"
SETTINGS_JSON_BYTES = b'{"permissions": {"allow": ["Bash"]}}\n'
OPUS_ADVISOR_BYTES = b"# opus advisor\n"
DELEGATOR_BYTES = b"# delegator\n"

# Two backup stamps in chronological order: the lexicographically smaller one is the earlier
# installation, and restore_original must pick it when there is no snapshot.
EARLIEST_STAMP = "20240101-000000"
LATEST_STAMP = "20240601-000000"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point install.py at a throwaway CLAUDE_HOME and STATE_DIR and return CLAUDE_HOME.

    Nothing below tmp_path is touched, so the developer's real ``~/.claude`` stays intact.
    """
    claude_home = tmp_path / "claude"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(INSTALL, "CLAUDE_HOME", claude_home)
    monkeypatch.setattr(INSTALL, "STATE_DIR", state_dir)
    claude_home.mkdir(parents=True)
    return claude_home


def write_tracked_file(claude_home: Path, relative_path: str, content: bytes) -> Path:
    """Write ``content`` at ``relative_path`` under CLAUDE_HOME, creating parent dirs.

    Returns:
        The absolute path of the file that was written.
    """
    target = claude_home / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def read_snapshot(state_dir: Path) -> dict[str, Any]:
    """Read and parse ``state_dir/snapshot.json``.

    Returns:
        The parsed snapshot dictionary.

    Raises:
        FileNotFoundError: if the snapshot was not written.
    """
    return json.loads((state_dir / "snapshot.json").read_text(encoding="utf-8"))


def write_legacy_snapshotless_backups(claude_home: Path, stamps: tuple[str, str]) -> Path:
    """Create two backups of CLAUDE.md with the given stamps and return their base path.

    Also removes the live CLAUDE.md so only the backups remain, mimicking a legacy install.
    """
    target = write_tracked_file(claude_home, CLAUDE_MD, b"# live\n")
    for stamp in stamps:
        shutil.copyfile(target, target.with_name(f"{CLAUDE_MD}.bak-foremankit-{stamp}"))
    target.unlink()
    return target


def test_snapshot_copies_existing_tracked_files_and_records_which_existed(sandbox: Path) -> None:
    """take_snapshot copies existing tracked files and records the existence flags."""
    write_tracked_file(sandbox, CLAUDE_MD, CLAUDE_MD_BYTES)
    write_tracked_file(sandbox, SETTINGS_JSON, SETTINGS_JSON_BYTES)

    INSTALL.Installer(dry=False).take_snapshot()

    state_dir = INSTALL.STATE_DIR
    snapshot = read_snapshot(state_dir)
    copied_claude_md = (state_dir / "files" / CLAUDE_MD).read_bytes()
    assert snapshot["existed"] == {CLAUDE_MD: True, SETTINGS_JSON: True, OPUS_ADVISOR: False,
                                   DELEGATOR: False}
    assert copied_claude_md == CLAUDE_MD_BYTES


def test_snapshot_is_never_overwritten_by_a_second_call(sandbox: Path) -> None:
    """A second take_snapshot call leaves the first snapshot untouched."""
    write_tracked_file(sandbox, CLAUDE_MD, CLAUDE_MD_BYTES)
    INSTALL.Installer(dry=False).take_snapshot()

    INSTALL.Installer(dry=False).take_snapshot()

    snapshot = read_snapshot(INSTALL.STATE_DIR)
    assert snapshot["existed"][CLAUDE_MD] is True
    assert snapshot["existed"][SETTINGS_JSON] is False


def test_snapshot_in_dry_run_mode_writes_nothing(sandbox: Path) -> None:
    """take_snapshot with dry=True writes neither files nor snapshot.json."""
    write_tracked_file(sandbox, CLAUDE_MD, CLAUDE_MD_BYTES)

    INSTALL.Installer(dry=True).take_snapshot()

    assert not (INSTALL.STATE_DIR / "snapshot.json").exists()
    assert not (INSTALL.STATE_DIR / "files").exists()


def test_restore_original_brings_back_the_exact_bytes_of_tracked_files(sandbox: Path) -> None:
    """restore_original rewrites CLAUDE.md and settings.json byte for byte.

    The CLAUDE.md fixture carries a non-ASCII character and CRLF endings, so a restore that
    normalises text or rewrites the file would fail here.
    """
    write_tracked_file(sandbox, CLAUDE_MD, CLAUDE_MD_BYTES)
    write_tracked_file(sandbox, SETTINGS_JSON, SETTINGS_JSON_BYTES)
    installer = INSTALL.Installer(dry=False)
    installer.take_snapshot()
    write_tracked_file(sandbox, CLAUDE_MD, b"# overwritten by foremankit\n")
    write_tracked_file(sandbox, SETTINGS_JSON, b"{}\n")

    installer.restore_original()

    assert (sandbox / CLAUDE_MD).read_bytes() == CLAUDE_MD_BYTES
    assert (sandbox / SETTINGS_JSON).read_bytes() == SETTINGS_JSON_BYTES


def test_restore_original_deletes_a_tracked_file_that_did_not_exist_before(sandbox: Path) -> None:
    """restore_original removes a tracked file the installer created out of nothing."""
    INSTALL.Installer(dry=False).take_snapshot()
    created_settings = write_tracked_file(sandbox, SETTINGS_JSON, SETTINGS_JSON_BYTES)

    INSTALL.Installer(dry=False).restore_original()

    assert not created_settings.exists()


def test_restore_original_keeps_a_backup_of_the_file_it_overwrites(sandbox: Path) -> None:
    """restore_original copies the current file aside before writing the snapshot back."""
    write_tracked_file(sandbox, CLAUDE_MD, CLAUDE_MD_BYTES)
    installer = INSTALL.Installer(dry=False)
    installer.take_snapshot()
    edited_bytes = b"# edited by the user\n"
    write_tracked_file(sandbox, CLAUDE_MD, edited_bytes)

    installer.restore_original()

    backups = list(sandbox.glob(f"{CLAUDE_MD}.bak-foremankit-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == edited_bytes


def test_restore_original_without_a_snapshot_uses_the_earliest_backup(sandbox: Path) -> None:
    """With no snapshot.json, restore_original falls back to the oldest backup file."""
    target = write_legacy_snapshotless_backups(sandbox, (LATEST_STAMP, EARLIEST_STAMP))
    earliest_bytes = (target.with_name(f"{CLAUDE_MD}.bak-foremankit-{EARLIEST_STAMP}")).read_bytes()

    INSTALL.Installer(dry=False).restore_original()

    assert target.read_bytes() == earliest_bytes


def test_restore_original_without_a_snapshot_leaves_a_file_without_backup_untouched(
    sandbox: Path,
) -> None:
    """A tracked file with no snapshot and no backup is left exactly as it is."""
    untouched_bytes = b"# written by hand, no snapshot, no backup\n"
    target = write_tracked_file(sandbox, DELEGATOR, untouched_bytes)

    INSTALL.Installer(dry=False).restore_original()

    assert target.read_bytes() == untouched_bytes


def test_remove_agents_restores_an_agent_file_that_existed_before(sandbox: Path) -> None:
    """remove_agents puts the snapshot copy of a pre-existing agent file back."""
    write_tracked_file(sandbox, OPUS_ADVISOR, OPUS_ADVISOR_BYTES)
    installer = INSTALL.Installer(dry=False)
    installer.take_snapshot()
    write_tracked_file(sandbox, OPUS_ADVISOR, b"# replaced by foremankit\n")

    installer.remove_agents()

    assert (sandbox / OPUS_ADVISOR).read_bytes() == OPUS_ADVISOR_BYTES


def test_remove_agents_deletes_an_agent_file_that_did_not_exist_before(sandbox: Path) -> None:
    """remove_agents removes an agent file the installer created from scratch."""
    INSTALL.Installer(dry=False).take_snapshot()
    created_agent = write_tracked_file(sandbox, DELEGATOR, DELEGATOR_BYTES)

    INSTALL.Installer(dry=False).remove_agents()

    assert not created_agent.exists()


def test_remove_empty_created_files_deletes_an_empty_claude_md_but_keeps_one_with_text(
    sandbox: Path,
) -> None:
    """remove_empty_created_files drops a blank installer-made CLAUDE.md and keeps a filled one."""
    INSTALL.Installer(dry=False).take_snapshot()
    blank_claude_md = write_tracked_file(sandbox, CLAUDE_MD, b"   \n\n")

    INSTALL.Installer(dry=False).remove_empty_created_files()
    blank_was_deleted = not blank_claude_md.exists()

    filled_claude_md = write_tracked_file(sandbox, CLAUDE_MD, b"# my own rules\n")
    INSTALL.Installer(dry=False).remove_empty_created_files()

    assert blank_was_deleted
    assert filled_claude_md.read_bytes() == b"# my own rules\n"


def test_restore_original_in_dry_run_mode_changes_nothing(sandbox: Path) -> None:
    """restore_original with dry=True reports but does not touch the file system."""
    write_tracked_file(sandbox, CLAUDE_MD, CLAUDE_MD_BYTES)
    INSTALL.Installer(dry=False).take_snapshot()
    edited_bytes = b"# edited by the user\n"
    write_tracked_file(sandbox, CLAUDE_MD, edited_bytes)

    INSTALL.Installer(dry=True).restore_original()

    assert (sandbox / CLAUDE_MD).read_bytes() == edited_bytes
    assert list(sandbox.glob(f"{CLAUDE_MD}.bak-foremankit-*")) == []


def test_router_mode_is_detected_from_the_vscode_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(INSTALL.Installer, "vscode_user_dir", staticmethod(lambda: tmp_path))
    installer = INSTALL.Installer.__new__(INSTALL.Installer)
    assert installer.router_mode_active() is False

    (tmp_path / SETTINGS_JSON).write_text('{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8081"}}', encoding="utf-8")
    assert installer.router_mode_active() is True

    (tmp_path / SETTINGS_JSON).write_text('{"editor.fontSize": 14}', encoding="utf-8")
    assert installer.router_mode_active() is False
