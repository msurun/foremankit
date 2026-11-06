"""Git plumbing for `agent-run`: a throw-away worktree on its own branch, the changes against the base commit,
reverting a path, and committing the patch. Everything runs in the worktree, never in the owner's working directory.
The changes are always read from the index against the BASE commit, so an agent that commits or amends on its own
cannot hide a change.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .agent_lock import git_lock
from .agent_scope import Change

IDENTITY = ["-c", "user.name=foremankit-agent-run", "-c", "user.email=agent-run@foremankit.invalid"]


class AgentGitError(RuntimeError):
    """A git command failed (not a repository, branch exists, commit hook refused...)."""


def git(args: list[str], cwd: Path) -> str:
    """stdout of `git <args>` in `cwd`; AgentGitError with git's message when it fails."""
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=False)
    except OSError as error:
        raise AgentGitError(f"git could not be started: {error}") from error
    if done.returncode != 0:
        raise AgentGitError(f"git {' '.join(args[:2])} failed: {(done.stderr or done.stdout).strip()[:300]}")
    return done.stdout


def repo_head(project: Path) -> str:
    """The commit the worktree starts from: HEAD of `project`, which must be the root of a git repository."""
    top = Path(git(["rev-parse", "--show-toplevel"], project).strip())
    if top.resolve() != project.resolve():
        raise AgentGitError(f"{project} is not the root of a git repository (root: {top})")
    return git(["rev-parse", "HEAD"], project).strip()


def resolve_commit(project: Path, ref: str) -> str:
    """The commit a branch name or hash points to (for `--base`: a dependent run starts from its predecessor's branch)."""
    try:
        return git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], project).strip()
    except AgentGitError as error:
        raise AgentGitError(f"base {ref!r} is not a commit of {project}") from error


def add_worktree(project: Path, path: Path, branch: str, base: str) -> None:
    """Create `path` as a worktree on the new branch `branch` at commit `base`."""
    with git_lock(project):
        git(["worktree", "add", "-q", "-b", branch, str(path), base], project)


def remove_worktree(project: Path, path: Path) -> None:
    """Delete the worktree folder (the branch and its commit stay) and forget it."""
    with git_lock(project):
        git(["worktree", "remove", "--force", str(path)], project)
        git(["worktree", "prune"], project)


def delete_branch(project: Path, branch: str) -> None:
    """Delete a branch that holds nothing worth keeping."""
    with git_lock(project):
        git(["branch", "-D", branch], project)


def stage_all(worktree: Path) -> None:
    """Stage every change (new, modified, deleted) so the index shows the whole state against the base."""
    git(["add", "-A"], worktree)


def changed_files(worktree: Path, base: str) -> list[Change]:
    """[(status letter, path)] of the staged state against `base`; renames are shown as delete + add."""
    raw = git(["diff", "--cached", "--name-status", "-z", "--no-renames", base], worktree)
    fields = raw.split("\0")
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields) - 1, 2)]


def diff_stat(worktree: Path, base: str) -> str:
    """`git diff --stat` of the staged state against `base`."""
    return git(["diff", "--cached", "--stat", base], worktree).strip()


def revert_path(worktree: Path, base: str, change: Change) -> None:
    """Undo one change: a file the agent added is removed, a modified or deleted file is restored from `base`."""
    status, path = change
    if status == "A":
        git(["rm", "-f", "-q", "--", path], worktree)
    else:
        git(["checkout", base, "--", path], worktree)


def commit_all(worktree: Path, base: str, message: str) -> bool:
    """Put the staged state on the branch as ONE commit on top of `base`. False when there is nothing to commit."""
    git(["reset", "--soft", base], worktree)
    if not git(["diff", "--cached", "--name-only"], worktree).strip():
        return False
    git([*IDENTITY, "commit", "-q", "-m", message], worktree)
    return True
