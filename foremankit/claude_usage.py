"""Read Claude subscription limits from `claude -p "/usage"`.

Cache the last reading plus a baseline, and report the points a work stretch used.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

EXTENSION_PREFIX = "anthropic.claude-code-"
EXTENSION_GLOB = EXTENSION_PREFIX + "*"
NATIVE_BINARY = Path("resources") / "native-binary"
BINARY_NAMES = ("claude.exe", "claude")

SESSION_PATTERN = re.compile(r"Current\s+session\s*:.*?(\d{1,3})\s*%")
WEEK_PATTERN = re.compile(r"Current\s+week\b[^:]*:.*?(\d{1,3})\s*%")
RESETS_PATTERN = re.compile(r"resets\s+(.+?)\s*$")
VERSION_PATTERN = re.compile(r"(\d+(?:\.\d+)*)")


class UsageError(RuntimeError):
    """A usage reading could not be obtained or understood."""


def parse_usage(text: str) -> dict | None:
    """Parse the session and week percent lines; None when neither line is present."""
    if not isinstance(text, str):
        return None
    session = _parse_line(text, SESSION_PATTERN)
    week = _parse_line(text, WEEK_PATTERN)
    if session is None and week is None:
        return None
    return {
        "session_pct": session[0] if session else None,
        "session_resets": session[1] if session else None,
        "week_pct": week[0] if week else None,
        "week_resets": week[1] if week else None,
    }


def _parse_line(text: str, pattern: re.Pattern[str]) -> tuple[int, str | None] | None:
    """Return (percent, resets text) from the first line matching a window pattern."""
    for line in text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        pct = int(match.group(1))
        if not 0 <= pct <= 100:
            return None
        resets = RESETS_PATTERN.search(line)
        return pct, resets.group(1).strip() if resets else None
    return None


def find_claude_exe(home: Path | None = None) -> Path | None:
    """Find the newest Claude Code native binary in the VS Code extensions folder."""
    root = (home or Path.home()) / ".vscode" / "extensions"
    found = []
    for folder in root.glob(EXTENSION_GLOB):
        version = _version_tuple(folder.name)
        if version is None:
            continue
        for name in BINARY_NAMES:
            exe = folder / NATIVE_BINARY / name
            if exe.is_file():
                found.append((version, exe))
                break
    return max(found, key=lambda item: item[0])[1] if found else None


def _version_tuple(folder_name: str) -> tuple[int, ...] | None:
    """Return the numeric version encoded in an extension folder name."""
    if not folder_name.startswith(EXTENSION_PREFIX):
        return None
    match = VERSION_PATTERN.match(folder_name[len(EXTENSION_PREFIX):])
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def fetch_usage(exe: Path, timeout_s: float = 60.0) -> str:
    """Run `claude -p "/usage"` with an argument list and no shell; return stdout."""
    try:
        result = subprocess.run(
            [str(exe), "-p", "/usage", "--output-format", "text"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UsageError(f"claude timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise UsageError(f"could not run claude: {exc.strerror or exc}") from exc
    return result.stdout


def refresh(path: Path, now: float, fetch: Callable[[], str] | None = None) -> dict:
    """Fetch a reading and store it as last; a failed read never erases a good one."""
    state = _load(path)
    previous = state.get("last")
    try:
        parsed = parse_usage(_fetch_text(fetch))
        if parsed is None:
            raise UsageError("could not parse usage output")
    except UsageError as exc:
        return {"ok": False, "error": str(exc), "last": previous}
    reading = {**parsed, "read_at": now}
    state["last"] = reading
    state.setdefault("baseline", None)
    _save(path, state)
    return {"ok": True, "last": reading}


def _fetch_text(fetch: Callable[[], str] | None) -> str:
    """Return text from the injected fetch, or run the installed claude binary."""
    if fetch is not None:
        return fetch()
    exe = find_claude_exe()
    if exe is None:
        raise UsageError("claude binary not found")
    return fetch_usage(exe)


def mark_baseline(path: Path, now: float) -> dict:
    """Copy the latest reading into the baseline slot and return the baseline."""
    state = _load(path)
    last = state.get("last")
    if not last:
        raise ValueError("no reading yet: refresh first")
    state["baseline"] = dict(last)
    state.setdefault("last", None)
    _save(path, state)
    return state["baseline"]


def report(path: Path, now: float) -> dict:
    """Summarize the cached last reading, the baseline, and the change between them."""
    state = _load(path)
    last = state.get("last")
    baseline = state.get("baseline")
    return {
        "last": last,
        "age_s": _age(last, now),
        "baseline": baseline,
        "baseline_age_s": _age(baseline, now),
        "delta": _delta(last, baseline),
    }


def _age(reading: Any, now: float) -> float | None:
    """Seconds since a reading was taken, or None without a usable timestamp."""
    if not isinstance(reading, dict):
        return None
    read_at = reading.get("read_at")
    if not isinstance(read_at, (int, float)):
        return None
    return float(now) - float(read_at)


def _delta(last: Any, baseline: Any) -> dict | None:
    """Percentage-point change since the baseline, per window that is still the same."""
    if not isinstance(last, dict) or not isinstance(baseline, dict):
        return None
    return {
        "session_pct": _points(last, baseline, "session"),
        "week_pct": _points(last, baseline, "week"),
    }


def _points(last: dict, baseline: dict, prefix: str) -> int | None:
    """Change for one window, or None when that window reset or a percent is missing."""
    if last.get(f"{prefix}_resets") != baseline.get(f"{prefix}_resets"):
        return None
    new_pct = last.get(f"{prefix}_pct")
    old_pct = baseline.get(f"{prefix}_pct")
    if not isinstance(new_pct, int) or not isinstance(old_pct, int):
        return None
    return new_pct - old_pct


def _load(path: Path) -> dict[str, Any]:
    """Read the JSON state file, returning empty slots when it is missing or corrupt."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"last": None, "baseline": None}
    return data if isinstance(data, dict) else {"last": None, "baseline": None}


def _save(path: Path, state: dict[str, Any]) -> None:
    """Write state atomically through a temporary file in the target folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp_path, path)
