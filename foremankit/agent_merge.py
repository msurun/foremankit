"""Merge or reject finished agent-run branches into the owner's active git branch.

This module provides safe git plumbing on the repository checkout, enforcing strict
preconditions before any merge, aborting cleanly on conflicts, and never pushing.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .agent_git import AgentGitError, delete_branch, git
from .agent_lock import git_lock

MAX_COMMIT_MESSAGE_CHARS = 100
AGENT_BRANCH_PREFIX = "agent-run/"
MERGE_PARENT_COUNT = 2


class MergeRefused(RuntimeError):
    """Precondition for merging or rejecting an agent-run branch was not met."""


@dataclass(frozen=True)
class MergeOptions:
    """Options governing agent-run branch merge verification and post-merge checks."""

    reviewed: bool = False
    recheck: Callable[[], tuple[bool, str]] | None = None


@dataclass(frozen=True)
class _MergeSource:
    """Snapshot the source commit and owner cleanliness before merging."""

    branch: str
    commit_hash: str
    was_clean: bool


class MergeExecution(NamedTuple):
    """Result of attempting a git merge: commit hash if clean, or conflict paths.

    In the Git Directed Acyclic Graph (DAG) model, a non-fast-forward merge creates
    a commit node with two parents: parent 1 is HEAD (current branch) and parent 2
    is the merged branch tip. When two parallel histories edit conflicting hunks,
    git leaves unmerged index entries across stages 1, 2, and 3.
    """

    is_conflict: bool
    commit_hash: str
    conflicts: list[str]


def _check_passed(report: dict) -> None:
    """Verify that the agent run succeeded and passed all validation checks."""
    if not report.get("passed", False):
        raise MergeRefused("agent run did not pass checks; refusing to merge")


def _check_review(report: dict, reviewed: bool) -> None:
    """Verify that modifications to sensitive files received required review sign-off."""
    if report.get("review_required", False) and not reviewed:
        raise MergeRefused("review is required for this run but options.reviewed is false")


def _check_branch_exists(project: Path, branch: str) -> str:
    """Refuse owner branches and return the agent branch's resolved commit."""
    if not branch:
        raise MergeRefused("branch name is empty or missing from report")
    if not branch.startswith(AGENT_BRANCH_PREFIX):
        raise MergeRefused("branch name must start with agent-run/; refusing owner branch")
    try:
        return git(["rev-parse", "--verify", "--quiet", f"{branch}^{{commit}}"], project).strip()
    except AgentGitError as error:
        raise MergeRefused(f"branch {branch!r} does not exist in {project}") from error


def _active_branch(project: Path) -> str | None:
    """Return the active branch name, or None if HEAD is detached."""
    try:
        return git(["symbolic-ref", "--quiet", "--short", "HEAD"], project).strip()
    except AgentGitError:
        return None


def _check_target_branch(project: Path) -> str:
    """Ensure HEAD is attached and current branch is not a temporary agent-run branch."""
    current = _active_branch(project)
    if current is None:
        raise MergeRefused("HEAD is detached; cannot merge into an unnamed commit")
    if current.startswith(AGENT_BRANCH_PREFIX):
        raise MergeRefused(f"cannot merge into temporary agent-run branch {current!r}")
    return current


def _check_target_files_clean(project: Path, files: list[str]) -> None:
    """Ensure files changed by the branch have no uncommitted local working tree changes."""
    if not files:
        return
    # Ignoring a local file does not give a merge permission to overwrite it.
    status = git(["status", "--porcelain", "--ignored", "--", *files], project).strip()
    if status:
        # In plain words:
        # The user has uncommitted changes in one or more files modified by the agent.
        # Merging now would risk clobbering uncommitted work, so we refuse early.
        raise MergeRefused(f"uncommitted local changes in target files: {status}")


def _check_no_merge_in_progress(project: Path) -> None:
    """Ensure no previous merge operation is currently in progress (.git/MERGE_HEAD)."""
    try:
        git(["rev-parse", "-q", "--verify", "MERGE_HEAD"], project)
    except AgentGitError:
        # In plain words:
        # rev-parse fails when MERGE_HEAD is absent. That confirms no unresolved
        # merge conflict is pending, which is the required pristine state.
        return
    raise MergeRefused("a merge is already in progress (.git/MERGE_HEAD exists)")


def _verify_preconditions(project: Path, report: dict, options: MergeOptions) -> _MergeSource:
    """Run all pre-merge precondition checks in strict required sequence."""
    _check_passed(report)
    _check_review(report, options.reviewed)
    branch = str(report.get("branch") or "").strip()
    commit_hash = _check_branch_exists(project, branch)
    _check_target_branch(project)
    files = [str(file_path) for file_path in report.get("files") or [] if str(file_path).strip()]
    _check_target_files_clean(project, files)
    _check_no_merge_in_progress(project)
    was_clean = not git(["status", "--porcelain"], project).strip()
    return _MergeSource(branch, commit_hash, was_clean)


def _format_merge_message(branch: str, summary: str) -> str:
    """Format merge commit message truncated to MAX_COMMIT_MESSAGE_CHARS."""
    first_line = summary.splitlines()[0].strip() if summary.strip() else ""
    return f"merge {branch}: {first_line}"[:MAX_COMMIT_MESSAGE_CHARS]


def _attempt_merge(project: Path, source_commit_hash: str, message: str) -> MergeExecution:
    """Attempt a non-fast-forward merge; abort and extract conflicts if unmerged."""
    try:
        with git_lock(project):
            git(["merge", "--no-ff", "--commit", "-m", message, source_commit_hash], project)
            _verify_merge_finished(project)
        commit_hash = git(["rev-parse", "HEAD"], project).strip()
        return MergeExecution(is_conflict=False, commit_hash=commit_hash, conflicts=[])
    except AgentGitError as error:
        return _abort_merge(project, str(error))


def _verify_merge_finished(project: Path) -> None:
    """Reject successful exits that did not finish a two-parent merge commit."""
    if _merge_in_progress(project):
        raise AgentGitError("MERGE_HEAD remains; finish the merge commit before retrying")
    parents = git(["rev-list", "--parents", "-n", "1", "HEAD"], project).split()[1:]
    if len(parents) != MERGE_PARENT_COUNT:
        raise AgentGitError("HEAD must have two parents; no completed merge commit was created")


def _merge_in_progress(project: Path) -> bool:
    """Whether MERGE_HEAD exists (git refused before it started a merge: nothing to abort)."""
    try:
        git(["rev-parse", "-q", "--verify", "MERGE_HEAD"], project)
    except AgentGitError:
        return False
    return True


def _abort_merge(project: Path, merge_error: str) -> MergeExecution:
    """Collect the unmerged files, abort the merge and report; no conflicts means git refused for another reason."""
    raw_conflicts = git(["diff", "--name-only", "--diff-filter=U"], project).strip()
    conflicts = [line.strip() for line in raw_conflicts.splitlines() if line.strip()]
    try:
        if _merge_in_progress(project):
            git(["merge", "--abort"], project)
    except AgentGitError as abort_error:
        raise MergeRefused(f"merge failed ({merge_error}) and --abort failed: {abort_error}; "
                           "the repository may be mid-merge, fix it by hand") from abort_error
    if not conflicts:
        raise MergeRefused(f"git refused the merge without conflicts: {merge_error}")
    return MergeExecution(is_conflict=True, commit_hash="", conflicts=conflicts)


def _conflict_dict(branch: str, conflicts: list[str]) -> dict:
    """Construct the return dictionary for an aborted conflicting merge."""
    return {
        "status": "conflict",
        "branch": branch,
        "merge_commit": "",
        "detail": ", ".join(conflicts),
    }


def _run_post_merge_recheck(
    source: _MergeSource, commit_hash: str, recheck: Callable[[], tuple[bool, str]]
) -> dict | None:
    """Execute post-merge check hook; return failure dictionary if check fails."""
    passed, output_tail = recheck()
    if passed:
        return None
    # In plain words:
    # Post-merge verification failed on the merged tree. We do NOT roll back or reset;
    # we preserve both the merge commit and the branch for the human lead to decide.
    # We mention the undo command in the detail text only.
    undo = ("git reset --hard ORIG_HEAD" if source.was_clean
            else f"git revert -m 1 {commit_hash} (preserves other uncommitted changes)")
    detail = (
        f"{output_tail.strip()}\n"
        f"Recheck failed after merge; branch {source.branch} kept. "
        f"Undo with: {undo}"
    ).strip()
    return {
        "status": "check_failed_after_merge",
        "branch": source.branch,
        "merge_commit": commit_hash,
        "detail": detail,
    }


def _safe_delete_branch(project: Path, source: _MergeSource) -> str:
    """Delete the merged branch; return error message on failure without raising."""
    try:
        current_tip = git(["rev-parse", f"refs/heads/{source.branch}^{{commit}}"], project).strip()
        if current_tip != source.commit_hash:
            return f"Source branch {source.branch} advanced after the merge; branch kept."
        delete_branch(project, source.branch)
        return ""
    except AgentGitError as error:
        # In plain words:
        # The merge succeeded, but deleting the temporary branch failed.
        # We record the warning in the detail string rather than failing the merge.
        return f"merged, but failed to delete branch {source.branch}: {error}"


def merge_branch(project: Path, report: dict, options: MergeOptions) -> dict:
    """Merge an agent-run branch into the current branch safely with strict preconditions."""
    source = _verify_preconditions(project, report, options)
    message = _format_merge_message(source.branch, str(report.get("summary") or ""))
    # Merge the reviewed snapshot, even if another process advances the branch name.
    merge_result = _attempt_merge(project, source.commit_hash, message)
    if merge_result.is_conflict:
        return _conflict_dict(source.branch, merge_result.conflicts)
    if options.recheck is not None:
        recheck_dict = _run_post_merge_recheck(source, merge_result.commit_hash, options.recheck)
        if recheck_dict is not None:
            return recheck_dict
    delete_detail = _safe_delete_branch(project, source)
    return {
        "status": "merged",
        "branch": source.branch,
        "merge_commit": merge_result.commit_hash,
        "detail": delete_detail,
    }


def reject_branch(project: Path, report: dict) -> dict:
    """Delete an agent-run branch without merging its changes."""
    branch = str(report.get("branch") or "").strip()
    _check_branch_exists(project, branch)
    active = _active_branch(project)
    if active == branch:
        raise MergeRefused(f"cannot reject active branch {branch!r}")
    delete_branch(project, branch)
    return {
        "status": "rejected",
        "branch": branch,
        "detail": "",
    }
