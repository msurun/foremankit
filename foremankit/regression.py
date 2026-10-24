"""Parse and run the REGRESSION.md ledger: a table that maps a module to the downstream
checks that must still pass when that module changes. The ledger answers "if module A
changes, what downstream must still work, and which command proves it?".

The public API parses the markdown table, finds changed files with git, selects the
affected ledger rows, runs their commands without a shell and reports the outcome.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

# git status is a fast local command; 20 s is a generous ceiling before we give up.
GIT_TIMEOUT_S = 20

# Conventional exit codes chosen so a harness can tell a timeout from a missing program.
RETURNCODE_TIMEOUT = 124
RETURNCODE_PROGRAM_MISSING = 127

# A ledger line is a markdown table row: columns separated by pipes.
COLUMN_SEPARATOR = "|"
EXPECTED_COLUMNS = 3

# Characters that make a table line a header or separator rather than data.
SEPARATOR_CELL_CHARS = frozenset("-: ")
HEADER_CELLS = ("module", "downstream", "command")

COMMAND_TAG_OK = "OK"
COMMAND_TAG_FAIL = "FAIL"


def parse_ledger(text: str) -> list[dict]:
    """Parse the markdown table in ``text`` into ledger entries.

    Returns a list of ``{"module", "downstream", "command"}`` dicts in file order.
    Skips header rows, ``|---|`` separator rows, blank cells and rows whose column
    count is not three. Backticks around a cell are stripped.
    """
    entries: list[dict] = []
    for line in text.splitlines():
        cells = _split_table_row(line)
        if _is_data_row(cells):
            module, downstream, command = (_clean_cell(cell) for cell in cells)
            entries.append({"module": module, "downstream": downstream, "command": command})
    return entries


def _split_table_row(line: str) -> list[str] | None:
    """Split a markdown row into stripped cells, or return None if it is not a table row.

    A table row starts and ends with a pipe; the outer pipes are not columns, so they
    are dropped and the interior is split on pipes.
    """
    stripped = line.strip()
    if not (stripped.startswith(COLUMN_SEPARATOR) and stripped.endswith(COLUMN_SEPARATOR)):
        return None
    interior = stripped[1:-1]
    return [cell.strip() for cell in interior.split(COLUMN_SEPARATOR)]


def _is_data_row(cells: list[str] | None) -> bool:
    """Return True when the cells form a complete three-column row with no blank cell."""
    if cells is None or len(cells) != EXPECTED_COLUMNS:
        return False
    if _is_separator_row(cells) or [cell.lower() for cell in cells] == list(HEADER_CELLS):
        return False
    return all(cell for cell in cells)


def _is_separator_row(cells: list[str]) -> bool:
    """Return True for a ``|---|:--:|---|`` alignment row: every cell only dashes/colons/spaces."""
    return all(set(cell) <= SEPARATOR_CELL_CHARS for cell in cells)


def _clean_cell(cell: str) -> str:
    """Strip surrounding backticks and whitespace from a table cell."""
    return cell.strip().strip("`").strip()


def changed_files(cwd: str) -> list[str]:
    """Return files changed in the git working tree at ``cwd``, as forward-slash paths.

    Includes staged, unstaged and untracked files; for a rename, only the new target is
    kept. Returns an empty list when git is missing or the command fails.
    """
    status_lines = _git_status_lines(cwd)
    files: list[str] = []
    for line in status_lines:
        path = _changed_path(line)
        if path is not None:
            files.append(path)
    return files


def _git_status_lines(cwd: str) -> list[str]:
    """Run ``git status --porcelain`` and return its non-empty lines, or [] on any failure."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _changed_path(porcelain_line: str) -> str | None:
    """Extract the forward-slash path from one ``git status --porcelain`` line.

    The porcelain format is ``XY <path>``: two status columns, a space, then the path.
    For a rename (``R``) the path is ``old -> new``; only the new target matters.
    """
    if len(porcelain_line) < 4:
        return None
    path = porcelain_line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.replace("\\", "/")


def affected(entries: list[dict], files: list[str]) -> list[dict]:
    """Return the entries whose module is a path prefix of at least one changed file.

    Keeps ledger order and drops duplicates. Paths are normalised to forward slashes
    with a single leading ``./`` (if any) removed before comparison.
    """
    normalised_files = [_normalise_path(file) for file in files]
    selected: list[dict] = []
    for entry in entries:
        module = _normalise_path(entry["module"])
        if _module_touches_any(module, normalised_files):
            selected.append(entry)
    return selected


def _normalise_path(path: str) -> str:
    """Convert to forward slashes and strip one leading ``./``."""
    slashed = path.replace("\\", "/")
    if slashed.startswith("./"):
        return slashed[2:]
    return slashed


def _module_touches_any(module: str, normalised_files: list[str]) -> bool:
    """Return True when ``module`` is a path prefix of any of the given files."""
    return any(file.startswith(module) for file in normalised_files)


def run_entries(entries: list[dict], cwd: str, timeout: int = 900) -> list[dict]:
    """Run every entry's command in ``cwd`` without a shell; never raises.

    Returns one result per entry with ``{"module", "downstream", "command",
    "returncode", "seconds"}``. ``returncode`` is 124 on timeout and 127 when the
    program itself is missing.
    """
    return [_run_one(entry, cwd, timeout) for entry in entries]


def _run_one(entry: dict, cwd: str, timeout: int) -> dict:
    """Run a single entry and return its result dict, mapping failures to exit codes."""
    started_at = time.monotonic()
    try:
        returncode = _execute_command(entry["command"], cwd, timeout)
    except FileNotFoundError:
        returncode = RETURNCODE_PROGRAM_MISSING
    except subprocess.TimeoutExpired:
        returncode = RETURNCODE_TIMEOUT
    except OSError:
        returncode = RETURNCODE_PROGRAM_MISSING
    return {
        "module": entry["module"],
        "downstream": entry["downstream"],
        "command": entry["command"],
        "returncode": returncode,
        "seconds": round(time.monotonic() - started_at, 2),
    }


def _unquote(token: str) -> str:
    """Remove one pair of matching surrounding quotes from ``token``."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def _execute_command(command: str, cwd: str, timeout: int) -> int:
    """Run ``command`` split into argv (no shell) and return its exit code.

    posix=False on Windows keeps backslashes in paths intact but also keeps the quote
    characters, so each token is unquoted by hand afterwards.
    """
    argv = [_unquote(token) for token in shlex.split(command, posix=(os.name != "nt"))]
    completed = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    return completed.returncode


def format_report(results: list[dict]) -> str:
    """Format results as one status line per entry plus a final ``n failed of m`` line.

    Returns the literal ``nothing affected`` when ``results`` is empty.
    """
    if not results:
        return "nothing affected"
    lines = [_format_line(result) for result in results]
    failed_count = sum(1 for result in results if result["returncode"] != 0)
    lines.append(f"{failed_count} failed of {len(results)}")
    return "\n".join(lines)


def _format_line(result: dict) -> str:
    """Render one result as ``OK|FAIL downstream (module): command``."""
    tag = COMMAND_TAG_OK if result["returncode"] == 0 else COMMAND_TAG_FAIL
    return f"{tag} {result['downstream']} ({result['module']}): {result['command']}"


def regress(ledger_path: str, cwd: str, run: bool = True) -> tuple[int, str]:
    """Read the ledger, select affected entries and optionally run them.

    Returns ``(exit_code, report_text)``. Exit code is 0 when everything passed or
    nothing was affected, 1 when any command failed, and ``(2, "no ledger at <path>")``
    when the ledger file does not exist.
    """
    ledger_file = Path(ledger_path)
    if not ledger_file.is_file():
        return 2, f"no ledger at {ledger_file}"
    entries = parse_ledger(ledger_file.read_text(encoding="utf-8"))
    selected = affected(entries, changed_files(cwd))
    if not run:
        return _report_without_running(selected)
    results = run_entries(selected, cwd)
    exit_code = 1 if any(result["returncode"] != 0 for result in results) else 0
    return exit_code, format_report(results)


def _report_without_running(selected: list[dict]) -> tuple[int, str]:
    """List selected entries without running them; exit code is always 0."""
    if not selected:
        return 0, format_report([])
    lines = [
        f"PENDING {entry['downstream']} ({entry['module']}): {entry['command']}"
        for entry in selected
    ]
    return 0, "\n".join(lines)
