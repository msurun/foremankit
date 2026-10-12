"""The shared coding standard (config/conventions.md) that is added to the prompt of every code-writing task."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import REPO_DIR, config_dir

DEFAULT_FILE = "conventions.md"
PROJECT_OVERRIDE_FOLDER = ".foremankit"


def conventions_apply(routing: dict[str, Any], kind: str) -> bool:
    """True when tasks of this kind (bulk_code, data_script, hard_code...) must follow the standard."""
    return kind in ((routing.get("conventions") or {}).get("kinds") or [])


def load_text(routing: dict[str, Any], project: Path) -> str:
    """The standard as text. The project's own `.foremankit/conventions.md` wins over the config folder's, which wins
    over the one shipped with foremankit; an empty string when none exists (the standard is then simply not added)."""
    name = (routing.get("conventions") or {}).get("file", DEFAULT_FILE)
    for folder in (project / PROJECT_OVERRIDE_FOLDER, config_dir(), REPO_DIR / "config"):
        candidate = folder / name
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return ""
