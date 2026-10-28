"""Keep the `modelPicker` key of Claude Code's user settings in place while the ForemanKit gateway runs.

In router/external mode the gateway makes local models available through the model picker; any
tool or test that deletes or changes the picker block is undone within a few seconds. In direct
mode (no gateway entry in the VS Code settings) the guard does nothing.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable

CLAUDE_SETTINGS_ENV = "FOREMANKIT_CLAUDE_SETTINGS"
VSCODE_SETTINGS_ENV = "FOREMANKIT_VSCODE_SETTINGS"

logger = logging.getLogger(__name__)


def model_picker_block(cfg: dict) -> dict:
    """The Claude Code `modelPicker` settings value built from orchestrator.yaml `model_picker:` ({} if absent)."""
    spec = cfg.get("model_picker") or {}
    options = [{("behavesAs" if key == "behaves_as" else key): value for key, value in row.items()}
               for row in spec.get("options") or []]
    return {"options": options, "replaceBuiltInOptions": bool(spec.get("replace_built_in"))} if options else {}


def merged_claude_settings(data: dict, block: dict) -> dict:
    """Return `data` with `modelPicker` set to `block`, or without the key when `block` is empty."""
    merged = {key: value for key, value in data.items() if key != "modelPicker"}
    if block:
        merged["modelPicker"] = block
    return merged


def claude_settings_path() -> Path:
    """Path of Claude Code's user settings.json; FOREMANKIT_CLAUDE_SETTINGS overrides the default."""
    configured = os.environ.get(CLAUDE_SETTINGS_ENV)
    return Path(configured) if configured else Path.home() / ".claude" / "settings.json"


def vscode_settings_path() -> Path:
    """Path of the VS Code user settings.json; FOREMANKIT_VSCODE_SETTINGS overrides the per-OS default."""
    configured = os.environ.get(VSCODE_SETTINGS_ENV)
    if configured:
        return Path(configured)
    return _default_vscode_user_dir() / "settings.json"


def _default_vscode_user_dir() -> Path:
    """Per-OS folder of the VS Code user settings (same layout as Installer.vscode_user_dir)."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Code" / "User"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User"
    return Path.home() / ".config" / "Code" / "User"


def router_mode_active(vscode_settings: Path) -> bool:
    """True when the VS Code settings point Claude Code at the ForemanKit gateway (ANTHROPIC_BASE_URL entry)."""
    try:
        data = json.loads(vscode_settings.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    entries = data.get("claudeCode.environmentVariables", [])
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(entry, dict) and entry.get("name") == "ANTHROPIC_BASE_URL"
        for entry in entries
    )


def ensure_picker(cfg: dict, claude_settings: Path, vscode_settings: Path) -> bool:
    """Make the Claude settings' modelPicker match cfg; True only when the file was rewritten.

    Does nothing when router mode is off, the picker block is empty, the Claude settings file is
    missing or not a JSON object, or modelPicker already equals the block. The write is atomic.
    """
    block = model_picker_block(cfg)
    if not block or not router_mode_active(vscode_settings):
        return False
    try:
        data = json.loads(claude_settings.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("modelPicker") == block:
        return False
    merged = merged_claude_settings(data, block)
    text = json.dumps(merged, indent=4, ensure_ascii=False) + "\n"
    _atomic_write(claude_settings, text)
    return True


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: a temp file in the same folder, then os.replace."""
    folder = path.parent
    folder.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=folder, prefix=".settings-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def start(get_cfg: Callable[[], dict], interval: float = 3.0) -> threading.Event:
    """Start a daemon thread that keeps modelPicker in place while router mode is active.

    The settings paths are resolved once, before the thread starts, and fixed for its lifetime.
    Every `interval` seconds the thread calls ensure_picker; OSError and ValueError are logged and
    the loop continues. Returns the Event that stops the thread when set.
    """
    claude = claude_settings_path()
    vscode = vscode_settings_path()
    stop = threading.Event()

    def loop() -> None:
        """Run ensure_picker until stop is set, logging recoverable errors between checks."""
        while not stop.is_set():
            try:
                ensure_picker(get_cfg(), claude, vscode)
            except (OSError, ValueError) as error:
                logger.warning("picker guard failed: %s", error)
            stop.wait(interval)

    threading.Thread(target=loop, name="picker-guard", daemon=True).start()
    return stop
