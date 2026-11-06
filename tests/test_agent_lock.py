"""Tests for foremankit.agent_lock: per-project paths, acquire/release, stale takeover and mutual exclusion.

Every test redirects the lock file into a tmp_path folder so nothing touches the real temp folder.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from foremankit import agent_lock

SHORT_WAIT_S = 0.3
THREAD_COUNT = 4
HOLD_S = 0.2                    # time each thread spends inside the lock (long enough to overlap)
EXITED_PROCESS_SLEEP_S = 0.5


@pytest.fixture
def temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `tempfile.gettempdir` at a private folder so lock files stay inside the test."""
    fake_temp = tmp_path / "temp"
    fake_temp.mkdir()
    monkeypatch.setattr(agent_lock.tempfile, "gettempdir", lambda: str(fake_temp))
    return fake_temp


def test_two_projects_have_different_lock_paths(tmp_path: Path, temp_dir: Path) -> None:
    """Each repository gets its own lock file."""
    first = agent_lock.lock_path(tmp_path / "alpha")
    second = agent_lock.lock_path(tmp_path / "beta")
    assert first != second


def test_same_project_has_same_lock_path(tmp_path: Path, temp_dir: Path) -> None:
    """Calling lock_path twice for one project returns the same path."""
    project = tmp_path / "alpha"
    assert agent_lock.lock_path(project) == agent_lock.lock_path(project)


def test_lock_path_lives_in_the_temp_folder(tmp_path: Path, temp_dir: Path) -> None:
    """The lock file is inside the temp folder, never inside the repository."""
    path = agent_lock.lock_path(tmp_path / "alpha")
    assert path.parent.parent == temp_dir


def test_git_lock_writes_owner_pid_and_removes_file(tmp_path: Path, temp_dir: Path) -> None:
    """Inside the block the file holds our PID; after the block it is gone."""
    project = tmp_path / "alpha"
    path = agent_lock.lock_path(project)

    with agent_lock.git_lock(project):
        owner_pid = int(path.read_text(encoding="utf-8"))

    assert owner_pid == os.getpid()
    assert not path.exists()


def test_git_lock_removes_file_after_an_exception(tmp_path: Path, temp_dir: Path) -> None:
    """An exception inside the block still releases the lock."""
    project = tmp_path / "alpha"
    path = agent_lock.lock_path(project)

    with pytest.raises(ValueError), agent_lock.git_lock(project):
        raise ValueError("boom")

    assert not path.exists()


def test_a_held_lock_times_out(tmp_path: Path, temp_dir: Path) -> None:
    """A second acquisition waits and then raises LockTimeout while the first is held."""
    project = tmp_path / "alpha"

    with agent_lock.git_lock(project), pytest.raises(agent_lock.LockTimeout), \
            agent_lock.git_lock(project, wait_s=SHORT_WAIT_S):
        pass


def test_git_lock_lets_only_one_holder_run(tmp_path: Path, temp_dir: Path) -> None:
    """Four threads doing read-modify-write on a counter never overlap; the counter ends at 4."""
    project = tmp_path / "alpha"
    counter_file = tmp_path / "counter.txt"
    counter_file.write_text("0", encoding="utf-8")
    tracker = _OverlapTracker()

    def bump_counter(_: int) -> None:
        """Hold the lock, pause so overlaps would be visible, then increment the counter."""
        with agent_lock.git_lock(project):
            tracker.enter()
            try:
                time.sleep(HOLD_S)
                value = int(counter_file.read_text(encoding="utf-8"))
                counter_file.write_text(str(value + 1), encoding="utf-8")
            finally:
                tracker.leave()

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        list(executor.map(bump_counter, range(THREAD_COUNT)))

    assert tracker.max_concurrent == 1
    assert counter_file.read_text(encoding="utf-8") == str(THREAD_COUNT)


class _OverlapTracker:
    """Thread-safe high-water mark of how many threads are inside the lock at once."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._active = 0
        self.max_concurrent = 0

    def enter(self) -> None:
        """Record that one more thread entered."""
        with self._guard:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)

    def leave(self) -> None:
        """Record that one thread left."""
        with self._guard:
            self._active -= 1


def _spawn_and_reap_child_pid() -> int:
    """Start a short-lived Python process, wait for it to exit, and return its now-dead PID."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    time.sleep(EXITED_PROCESS_SLEEP_S)
    return child.pid


def test_git_lock_takes_over_a_lock_of_an_exited_process(
    tmp_path: Path, temp_dir: Path
) -> None:
    """A lock file whose owner PID is gone is taken over."""
    project = tmp_path / "alpha"
    path = agent_lock.lock_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(_spawn_and_reap_child_pid()), encoding="utf-8")

    with agent_lock.git_lock(project, wait_s=SHORT_WAIT_S):
        pass

    assert not path.exists()


def test_git_lock_takes_over_a_lock_with_garbage_content(
    tmp_path: Path, temp_dir: Path
) -> None:
    """An unreadable lock file is treated as stale and taken over."""
    project = tmp_path / "alpha"
    path = agent_lock.lock_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-a-pid", encoding="utf-8")

    with agent_lock.git_lock(project, wait_s=SHORT_WAIT_S):
        pass

    assert not path.exists()


def test_git_lock_takes_over_a_lock_older_than_max_age(
    tmp_path: Path, temp_dir: Path
) -> None:
    """A lock file with a live PID but an ancient mtime is taken over."""
    project = tmp_path / "alpha"
    path = agent_lock.lock_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")
    ancient = time.time() - (agent_lock.MAX_AGE_S + 60.0)
    os.utime(path, (ancient, ancient))

    with agent_lock.git_lock(project, wait_s=SHORT_WAIT_S):
        pass

    assert not path.exists()


def test_git_lock_does_not_take_over_a_fresh_lock_of_a_live_process(
    tmp_path: Path, temp_dir: Path
) -> None:
    """A fresh lock file owned by this live process blocks acquisition."""
    project = tmp_path / "alpha"
    path = agent_lock.lock_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(agent_lock.LockTimeout), agent_lock.git_lock(project, wait_s=SHORT_WAIT_S):
        pass


def test_pid_alive_is_true_for_our_own_process() -> None:
    """The current process is obviously alive."""
    assert agent_lock.pid_alive(os.getpid()) is True


def test_pid_alive_is_false_for_an_exited_process() -> None:
    """A PID whose process has finished is not alive."""
    assert agent_lock.pid_alive(_spawn_and_reap_child_pid()) is False


def test_fresh_empty_lock_file_is_not_taken_over(tmp_path: Path, temp_dir: Path) -> None:
    """An empty file is a PID being written by its owner: waiting for it must time out, not delete the lock."""
    project = tmp_path / "alpha"
    path = agent_lock.lock_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    with pytest.raises(agent_lock.LockTimeout), agent_lock.git_lock(project, wait_s=0.3):
        pass
    assert path.exists()
