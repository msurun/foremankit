"""Tests for agent_merge: verifying safe merging, conflict abortion, and branch rejection.

Validates all merge preconditions, conflict handling, post-merge recheck hooks,
and branch deletion behaviors using isolated local git repositories without mocks.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from foremankit.agent_merge import (
    MergeOptions,
    MergeRefused,
    merge_branch,
    reject_branch,
)

IDENTITY = ["-c", "user.name=t", "-c", "user.email=t@t"]


def git(repo: Path, *args: str) -> str:
    """Execute a git command within repo using test identity and return its stdout."""
    return subprocess.run(
        ["git", *IDENTITY, *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def write_files(root: Path, files: dict[str, str]) -> None:
    """Write or overwrite text files under root directory."""
    for relative_path, text in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def commit_all(repo: Path, files: dict[str, str], message: str = "commit") -> None:
    """Write files into repo, stage everything, and commit with message."""
    write_files(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an initialized git repository on main with test author identity configured."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    git(project_path, "init", "-q", "-b", "main")
    git(project_path, "config", "user.name", "t")
    git(project_path, "config", "user.email", "t@t")
    commit_all(project_path, {"README.md": "# test\n", "base.py": "x = 1\n"}, "initial")
    scratch = tmp_path / "lock_dir"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(scratch))
    return project_path


def test_clean_merge_creates_merge_commit_with_two_parents_and_deletes_branch(
    repo: Path,
) -> None:
    """A clean merge creates a 2-parent merge commit on main and removes the branch."""
    git(repo, "checkout", "-q", "-b", "agent-run/task-1")
    commit_all(repo, {"new_module.py": "content = 1\n"}, "agent commit")
    git(repo, "checkout", "-q", "main")
    report = {
        "passed": True,
        "branch": "agent-run/task-1",
        "files": ["new_module.py"],
        "summary": "add new module",
    }
    result = merge_branch(repo, report, MergeOptions())
    parents = git(repo, "rev-list", "--parents", "-n", "1", "HEAD").strip().split()
    branch_listing = git(repo, "branch", "--list", "agent-run/task-1").strip()
    assert result["status"] == "merged"
    assert result["branch"] == "agent-run/task-1"
    assert (repo / "new_module.py").exists()
    assert len(parents) == 3
    assert branch_listing == ""


def test_conflict_aborts_merge_preserves_working_tree_and_names_conflicts(
    repo: Path,
) -> None:
    """A merge conflict aborts, leaves working tree untouched, and names conflicting file."""
    git(repo, "checkout", "-q", "-b", "agent-run/conflict-branch")
    commit_all(repo, {"base.py": "x = 2\n"}, "agent change")
    git(repo, "checkout", "-q", "main")
    commit_all(repo, {"base.py": "x = 3\n"}, "main change")
    pre_status = git(repo, "status", "--porcelain")
    report = {
        "passed": True,
        "branch": "agent-run/conflict-branch",
        "files": ["base.py"],
    }
    result = merge_branch(repo, report, MergeOptions())
    assert result["status"] == "conflict"
    assert result["merge_commit"] == ""
    assert "base.py" in result["detail"]
    assert git(repo, "status", "--porcelain") == pre_status
    assert git(repo, "rev-parse", "--verify", "agent-run/conflict-branch").strip() != ""


def test_merge_refused_when_run_did_not_pass(repo: Path) -> None:
    """MergeRefused is raised when report passed flag is false."""
    report = {"passed": False, "branch": "agent-run/failed", "files": ["base.py"]}
    with pytest.raises(MergeRefused, match="did not pass"):
        merge_branch(repo, report, MergeOptions())


def test_merge_refused_when_review_required_without_reviewed_option(
    repo: Path,
) -> None:
    """MergeRefused is raised when review is required but options.reviewed is false."""
    report = {
        "passed": True,
        "review_required": True,
        "branch": "agent-run/needs-review",
        "files": [],
    }
    with pytest.raises(MergeRefused, match="review is required"):
        merge_branch(repo, report, MergeOptions(reviewed=False))


def test_merge_succeeds_when_review_required_and_reviewed_is_true(repo: Path) -> None:
    """Merge succeeds when review_required is true and options.reviewed is true."""
    git(repo, "checkout", "-q", "-b", "agent-run/approved")
    commit_all(repo, {"approved.py": "ok = True\n"}, "approved work")
    git(repo, "checkout", "-q", "main")
    report = {
        "passed": True,
        "review_required": True,
        "branch": "agent-run/approved",
        "files": ["approved.py"],
    }
    result = merge_branch(repo, report, MergeOptions(reviewed=True))
    assert result["status"] == "merged"
    assert (repo / "approved.py").exists()


def test_merge_refused_when_target_branch_does_not_exist(repo: Path) -> None:
    """MergeRefused is raised when the branch in the report does not exist in repo."""
    report = {"passed": True, "branch": "agent-run/missing-branch", "files": []}
    with pytest.raises(MergeRefused, match="does not exist"):
        merge_branch(repo, report, MergeOptions())


def test_merge_refused_when_current_branch_is_an_agent_run_branch(repo: Path) -> None:
    """MergeRefused is raised when project is checked out on an agent-run branch."""
    git(repo, "checkout", "-q", "-b", "agent-run/active-run")
    git(repo, "branch", "agent-run/other-run")
    report = {"passed": True, "branch": "agent-run/other-run", "files": []}
    with pytest.raises(MergeRefused, match="agent-run"):
        merge_branch(repo, report, MergeOptions())


def test_merge_refused_when_head_is_detached(repo: Path) -> None:
    """MergeRefused is raised when project HEAD is in a detached state."""
    git(repo, "branch", "agent-run/detached-source")
    git(repo, "checkout", "-q", "--detach", "main")
    report = {"passed": True, "branch": "agent-run/detached-source", "files": []}
    with pytest.raises(MergeRefused, match="detached"):
        merge_branch(repo, report, MergeOptions())


def test_merge_refused_when_target_file_has_uncommitted_changes(repo: Path) -> None:
    """MergeRefused is raised when a file modified by the branch is dirty locally."""
    git(repo, "branch", "agent-run/dirty-check")
    (repo / "base.py").write_text("x = 999\n", encoding="utf-8")
    report = {
        "passed": True,
        "branch": "agent-run/dirty-check",
        "files": ["base.py"],
    }
    with pytest.raises(MergeRefused, match="uncommitted local changes"):
        merge_branch(repo, report, MergeOptions())


def test_merge_succeeds_when_unrelated_file_has_uncommitted_changes(
    repo: Path,
) -> None:
    """Unrelated dirty files are preserved and do not prevent clean merge."""
    git(repo, "checkout", "-q", "-b", "agent-run/isolated-change")
    commit_all(repo, {"new_feature.py": "def f(): pass\n"}, "feature commit")
    git(repo, "checkout", "-q", "main")
    (repo / "unrelated_notes.txt").write_text("my notes in progress\n", encoding="utf-8")
    report = {
        "passed": True,
        "branch": "agent-run/isolated-change",
        "files": ["new_feature.py"],
    }
    result = merge_branch(repo, report, MergeOptions())
    assert result["status"] == "merged"
    assert (repo / "unrelated_notes.txt").read_text(encoding="utf-8") == "my notes in progress\n"
    assert (repo / "new_feature.py").exists()


def test_merge_refused_when_a_merge_is_already_in_progress(repo: Path) -> None:
    """MergeRefused is raised when .git/MERGE_HEAD already exists from another merge."""
    git(repo, "checkout", "-q", "-b", "conflict-side")
    commit_all(repo, {"base.py": "x = 100\n"}, "conflict commit")
    git(repo, "checkout", "-q", "main")
    git(repo, "branch", "agent-run/another")
    commit_all(repo, {"base.py": "x = 200\n"}, "main conflict commit")
    subprocess.run(
        ["git", *IDENTITY, "merge", "conflict-side"],
        check=False,
        cwd=repo,
        capture_output=True,
    )
    report = {"passed": True, "branch": "agent-run/another", "files": []}
    try:
        with pytest.raises(MergeRefused, match="merge is already in progress"):
            merge_branch(repo, report, MergeOptions())
    finally:
        git(repo, "merge", "--abort")


def test_recheck_fails_preserves_merge_and_branch_with_status_and_detail(
    repo: Path,
) -> None:
    """Failed post-merge recheck retains both the merge commit and the branch."""
    git(repo, "checkout", "-q", "-b", "agent-run/recheck-fail")
    commit_all(repo, {"feature_broken.py": "syntax error!\n"}, "broken feature")
    git(repo, "checkout", "-q", "main")
    report = {
        "passed": True,
        "branch": "agent-run/recheck-fail",
        "files": ["feature_broken.py"],
    }
    options = MergeOptions(recheck=lambda: (False, "lint error on feature_broken.py"))
    result = merge_branch(repo, report, options)
    assert result["status"] == "check_failed_after_merge"
    assert result["merge_commit"] != ""
    assert "lint error on feature_broken.py" in result["detail"]
    assert "git reset --hard ORIG_HEAD" in result["detail"]
    assert (repo / "feature_broken.py").exists()
    assert git(repo, "rev-parse", "--verify", "agent-run/recheck-fail").strip() != ""


def test_recheck_passes_completes_merge_and_deletes_branch(repo: Path) -> None:
    """Passing post-merge recheck completes merge and removes the branch."""
    git(repo, "checkout", "-q", "-b", "agent-run/recheck-pass")
    commit_all(repo, {"valid.py": "valid = True\n"}, "valid feature")
    git(repo, "checkout", "-q", "main")
    report = {
        "passed": True,
        "branch": "agent-run/recheck-pass",
        "files": ["valid.py"],
    }
    options = MergeOptions(recheck=lambda: (True, "all tests passed"))
    result = merge_branch(repo, report, options)
    branch_listing = git(repo, "branch", "--list", "agent-run/recheck-pass").strip()
    assert result["status"] == "merged"
    assert result["merge_commit"] != ""
    assert (repo / "valid.py").exists()
    assert branch_listing == ""


def test_reject_branch_deletes_specified_branch(repo: Path) -> None:
    """reject_branch deletes the agent-run branch without modifying main."""
    git(repo, "checkout", "-q", "-b", "agent-run/to-reject")
    commit_all(repo, {"junk.py": "bad code\n"}, "unwanted work")
    git(repo, "checkout", "-q", "main")
    report = {"branch": "agent-run/to-reject"}
    result = reject_branch(repo, report)
    branch_listing = git(repo, "branch", "--list", "agent-run/to-reject").strip()
    assert result["status"] == "rejected"
    assert result["branch"] == "agent-run/to-reject"
    assert branch_listing == ""


def test_reject_refused_when_branch_does_not_exist(repo: Path) -> None:
    """reject_branch raises MergeRefused when branch does not exist."""
    report = {"branch": "agent-run/ghost-branch"}
    with pytest.raises(MergeRefused, match="does not exist"):
        reject_branch(repo, report)


def test_reject_refused_when_branch_is_currently_active(repo: Path) -> None:
    """reject_branch raises MergeRefused when attempting to reject active branch."""
    report = {"branch": "main"}
    with pytest.raises(MergeRefused, match="cannot reject active branch"):
        reject_branch(repo, report)


def test_merge_commit_message_is_truncated_to_maximum_limit(repo: Path) -> None:
    """Merge commit message is truncated to maximum allowed characters."""
    git(repo, "checkout", "-q", "-b", "agent-run/long-summary")
    commit_all(repo, {"doc.txt": "text\n"}, "doc commit")
    git(repo, "checkout", "-q", "main")
    long_summary = "X" * 300
    report = {
        "passed": True,
        "branch": "agent-run/long-summary",
        "files": ["doc.txt"],
        "summary": long_summary,
    }
    result = merge_branch(repo, report, MergeOptions())
    subject = git(repo, "log", "-1", "--format=%s").strip()
    assert result["status"] == "merged"
    assert len(subject) <= 100


def test_merge_operation_never_requires_or_invokes_push(repo: Path) -> None:
    """Merge operation works in a standalone local repo without any remote."""
    assert git(repo, "remote").strip() == ""
    git(repo, "checkout", "-q", "-b", "agent-run/no-remote")
    commit_all(repo, {"local_only.py": "value = 42\n"}, "local commit")
    git(repo, "checkout", "-q", "main")
    report = {
        "passed": True,
        "branch": "agent-run/no-remote",
        "files": ["local_only.py"],
    }
    result = merge_branch(repo, report, MergeOptions())
    assert result["status"] == "merged"
    assert git(repo, "remote").strip() == ""
