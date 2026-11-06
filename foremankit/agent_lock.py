"""A cross-process lock for the git plumbing of parallel `agent-run`s (worktree add/remove/prune, branch delete).
Parallel runs share one `.git` folder (here inside OneDrive); two `worktree add`/`prune` calls at once race on its locks.
The lock file lives in the temp folder, not in the repository, and holds the PID of its owner so a crashed run's lock is
taken over after its process is gone (or after MAX_AGE_S).
"""
from __future__ import annotations

import ctypes
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

WAIT_S = 120.0
POLL_S = 0.1
MAX_AGE_S = 600.0               # a lock older than this is taken over even if its PID looks alive (PID reuse)
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED = 0x1000


class LockTimeout(RuntimeError):
    """The lock stayed taken for the whole waiting time."""


def lock_path(project: Path) -> Path:
    """One lock file per repository, in the temp folder."""
    name = "".join(char if char.isalnum() else "_" for char in str(project.resolve()))[-80:]
    return Path(tempfile.gettempdir()) / "foremankit-agent-run" / f"{name}.lock"


def pid_alive(pid: int) -> bool:
    """Whether a process with this PID exists. On Windows `os.kill(pid, 0)` would terminate it, so ask the OS instead."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return bool(ok) and code.value == STILL_ACTIVE


def _is_stale(path: Path) -> bool:
    """The lock file's owner is gone, the file is unreadable, or it is older than MAX_AGE_S."""
    try:
        age = time.time() - path.stat().st_mtime
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return True
    return age > MAX_AGE_S or not pid_alive(pid)


def _try_create(path: Path) -> bool:
    """Create the lock file exclusively and write our PID; False when somebody else holds it."""
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    return True


@contextmanager
def git_lock(project: Path, wait_s: float = WAIT_S) -> Iterator[None]:
    """Hold the repository's agent-run lock for the `with` block; LockTimeout after `wait_s` seconds."""
    path = lock_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_s
    while not _try_create(path):
        if _is_stale(path):
            path.unlink(missing_ok=True)
        elif time.monotonic() > deadline:
            raise LockTimeout(f"git lock {path} is held by another agent-run")
        else:
            time.sleep(POLL_S)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
