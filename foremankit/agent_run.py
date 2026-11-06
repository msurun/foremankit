"""`foremankit agent-run`: a multi-file task for a headless coding agent (agy) in its OWN git worktree.
The worktree keeps `--dangerously-skip-permissions` away from the owner's working directory. After the run ForemanKit
(1) reverts every change outside --scope and reports it, (2) re-runs the check itself through verify.py with PYTHONPATH
pointing at the worktree (editable installs would otherwise test the main checkout), (3) commits the in-scope patch on
the branch `agent-run/<id>` and (4) returns a short report. The agent's own PASS is never trusted; the review stays.
"""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import agent_git as vcs
from .agent_runner import Runner, RunnerError, RunnerResult, make_runner
from .agent_scope import Change, review_reasons, split_by_scope
from .agy_provider import MAX_PROMPT_CHARS
from .config import load_config
from .state import CallLog
from .verify import VerifyError, check_folder, run_command, split_command

TASK_FILE = ".agent-task.md"
AGENT_TIMEOUT_S = 1800
CHECK_TIMEOUT_S = 600
REPLY_CHARS = 200
TAIL_CHARS = 400
PROMPT_RESERVE = 1500          # room for the working rules next to the task text in one command line
TRACKED_CHANGES = ("M", "D", "T")


class AgentRunError(RuntimeError):
    """The run is refused or cannot complete (private data, no scope, command not allowed, git or runner failure)."""


@dataclass(frozen=True)
class AgentRunJob:
    """One multi-file task: what to do, which files it may touch, how to check it."""
    task: str
    project: Path
    scope: tuple[str, ...]
    check: str
    runner: str = "agy"
    model: str = ""
    sensitivity: str = "normal"
    timeout: float = AGENT_TIMEOUT_S
    check_cwd: str = ""
    keep_worktree: bool = False


def agent_run(job: AgentRunJob, runner: Runner | None = None, log: CallLog | None = None) -> dict[str, Any]:
    """Run the job and return the report (see `render_text`). `runner` is injectable for tests."""
    config = load_config()
    try:
        runner = runner or make_runner(job.runner, config, job.model)
    except RunnerError as error:
        raise AgentRunError(str(error)) from error
    _refuse(job, runner)
    tokens = _check_tokens(job, config)
    report = _Run(job, runner, config, tokens).execute()
    (log or CallLog()).write(_outcome(report))
    return report


def _refuse(job: AgentRunJob, runner: Runner) -> None:
    """Refuse before anything is created: private data, empty scope or task, a runner that may not see this data."""
    if job.sensitivity == "private" or not runner.eligible(job.sensitivity):
        raise AgentRunError(f"{runner.name} may not receive {job.sensitivity} data")
    if not job.scope:
        raise AgentRunError("at least one --scope pattern is required")
    if not job.task.strip():
        raise AgentRunError("the task is empty")


def _check_tokens(job: AgentRunJob, config: dict[str, Any]) -> list[str]:
    allowed = config["routing"]["verify"]["allowed_commands"]
    try:
        return split_command(job.check, allowed, job.project)
    except VerifyError as error:
        raise AgentRunError(str(error)) from error


class _Run:
    """The state of one run: ids, the worktree path and the steps in order."""

    def __init__(self, job: AgentRunJob, runner: Runner, config: dict[str, Any], tokens: list[str]):
        self.job, self.runner, self.tokens = job, runner, tokens
        self.settings = config["routing"].get("agent_run", {})
        self.id = uuid.uuid4().hex[:8]
        self.branch = f"agent-run/{self.id}"
        self.worktree = Path(tempfile.gettempdir()) / "foremankit-agent-run" / f"{job.project.name}-{self.id}"
        self.started = time.perf_counter()

    def execute(self) -> dict[str, Any]:
        """Worktree, agent, scope guard, check, commit, cleanup. The branch survives; the worktree goes unless kept."""
        base = self._start_worktree()
        try:
            report = self._finish(base, self.runner.run(build_prompt(self.job, self.worktree, self.tokens),
                                                        self.worktree, self.job.timeout))
        except RunnerError as error:
            self._discard()
            raise AgentRunError(str(error)) from error
        except (VerifyError, vcs.AgentGitError, OSError) as error:
            raise AgentRunError(f"{error} (worktree kept: {self.worktree}, branch {self.branch})") from error
        if report["committed"] and not self.job.keep_worktree:
            vcs.remove_worktree(self.job.project, self.worktree)
            report["worktree"] = None
        return report

    def _start_worktree(self) -> str:
        try:
            base = vcs.repo_head(self.job.project)
            self.worktree.parent.mkdir(parents=True, exist_ok=True)
            vcs.add_worktree(self.job.project, self.worktree, self.branch, base)
        except vcs.AgentGitError as error:
            raise AgentRunError(str(error)) from error
        return base

    def _discard(self) -> None:
        """The runner never started: nothing to keep."""
        vcs.remove_worktree(self.job.project, self.worktree)
        vcs.delete_branch(self.job.project, self.branch)

    def _finish(self, base: str, result: RunnerResult) -> dict[str, Any]:
        (self.worktree / TASK_FILE).unlink(missing_ok=True)
        vcs.stage_all(self.worktree)
        inside, outside = split_by_scope(vcs.changed_files(self.worktree, base), self.job.scope)
        for change in outside:
            vcs.revert_path(self.worktree, base, change)
        check_passed, tail = self._run_check()
        stat = vcs.diff_stat(self.worktree, base)
        committed = vcs.commit_all(self.worktree, base, f"agent-run {self.id}: {self.job.task.strip()[:60]}")
        return self._report(base, result, (inside, outside), (check_passed, tail, stat, committed))

    def _run_check(self) -> tuple[bool, str]:
        """Our own run of the check in the worktree, never the agent's word for it."""
        folder = check_folder(self.job.check_cwd, self.worktree)
        tokens = [locate_program(self.tokens[0], self.worktree, self.job.project), *self.tokens[1:]]
        timeout = float(self.settings.get("check_timeout_s", CHECK_TIMEOUT_S))
        return run_command(tokens, folder, timeout, env=worktree_env(self.worktree))

    def _report(self, base: str, result: RunnerResult, changes: tuple[list[Change], list[Change]],
                check: tuple[bool, str, str, bool]) -> dict[str, Any]:
        inside, outside = changes
        check_passed, tail, stat, committed = check
        tracked = [path for status, path in outside if status in TRACKED_CHANGES]
        paths = [path for _, path in inside]
        reasons = review_reasons(paths, list(self.settings.get("review_paths", [])))
        return {"passed": check_passed and bool(inside) and not tracked, "run_id": self.id, "branch": self.branch,
                "base": base[:10], "runner": self.runner.name, "model": self.runner.model, "files": paths,
                "diff_stat": stat, "check_passed": check_passed, "check_output_tail": tail[-TAIL_CHARS:].strip(),
                "reverted": [f"{status} {path}" for status, path in outside], "tracked_violations": tracked,
                "agent_reply": result.reply[:REPLY_CHARS], "timed_out": result.timed_out, "committed": committed,
                "review_required": bool(reasons), "review_paths": reasons, "worktree": str(self.worktree),
                "seconds": round(time.perf_counter() - self.started, 1)}


def build_prompt(job: AgentRunJob, worktree: Path, tokens: list[str]) -> str:
    """The task plus the fixed working rules. A task too long for the command line goes into a file in the worktree."""
    task = job.task.strip()
    rules = _rules(job, worktree, tokens)
    if len(task) + len(rules) > MAX_PROMPT_CHARS - PROMPT_RESERVE:
        (worktree / TASK_FILE).write_text(task, encoding="utf-8")
        task = f"The task is in the file {TASK_FILE} in the current folder: read it first and do exactly that."
    return f"{task}\n\n{rules}"


def _rules(job: AgentRunJob, worktree: Path, tokens: list[str]) -> str:
    program = locate_program(tokens[0], worktree, job.project)
    return (f"Work only inside {worktree}. You may create or change only files matching: {', '.join(job.scope)}. "
            "Do not touch any other file, do not commit, do not switch branches. Read the real source files the task "
            "points at before you write code that depends on them. "
            f"ForemanKit re-runs `{job.check}` itself after you finish. To try it yourself use `{program}` as the program, "
            f"run it in {worktree} and set the environment variable PYTHONPATH={worktree_env(worktree)['PYTHONPATH']}, "
            "otherwise the tests import the OTHER checkout. Fix and run again, at most 3 rounds. "
            "Your final reply is one line: PASS or FAIL and the reason.")


def worktree_env(worktree: Path) -> dict[str, str]:
    """The environment for a check in the worktree: PYTHONPATH starts with the worktree and its src folders.
    Measured 2026-11-05: an editable install in the shared venv resolves the MAIN checkout, so a plain
    `pytest` in the worktree passes or fails on the wrong code."""
    roots = [worktree, worktree / "src", *sorted(worktree.glob("*/src"))]
    paths = [str(root) for root in roots if root.is_dir()]
    inherited = os.environ.get("PYTHONPATH")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([*paths, inherited] if inherited else paths)
    return env


def locate_program(token: str, worktree: Path, project: Path) -> str:
    """A relative program path (`ForemanKit/.venv/Scripts/python`) as an absolute path: the venv is not in the worktree,
    so look in the worktree, the main checkout and its parent folder. Bare names and absolute paths stay as they are."""
    if "/" not in token or Path(token).is_absolute():
        return token
    for root in (worktree, project, project.parent):
        found = _existing_file(root / token)
        if found:
            return found
    return token


def _existing_file(path: Path) -> str | None:
    candidates = [path, path.parent / (path.name + ".exe")] if os.name == "nt" else [path]
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


def _outcome(report: dict[str, Any]) -> dict[str, Any]:
    """The `outcome` log event, so `foremankit stats` and pass rates include agent-run results (kind agent_run)."""
    return {"event": "outcome", "id": report["run_id"], "provider": report["runner"], "node": None,
            "model": report["model"], "kind": "agent_run", "cost_usd": 0.0, "verified": True,
            "verify_passed": report["check_passed"], "fix_rounds": 0, "reviewed": False, "blockers": 0,
            "review_fixed": False, "accepted": report["passed"], "seconds": report["seconds"]}


def render_text(report: dict[str, Any]) -> str:
    """The short text report: verdict line, branch, diff stat, check, scope, reply, and REVIEW REQUIRED when it applies."""
    verdict = "PASS" if report["passed"] else "FAIL"
    tail = " ".join(report["check_output_tail"].split())[-150:]
    lines = [f"{verdict} {report['branch']} ({report['runner']} {report['model']}) {report['seconds']}s",
             f"base {report['base']}, {len(report['files'])} file(s): {', '.join(report['files']) or 'none'}",
             report["diff_stat"].splitlines()[-1] if report["diff_stat"] else "no changes",
             f"check {'PASS' if report['check_passed'] else 'FAIL'}: {tail}"]
    if report["reverted"]:
        lines.append("reverted outside scope: " + ", ".join(report["reverted"]))
    lines.append(f"reply: {report['agent_reply']}")
    if report["review_required"]:
        lines.append("REVIEW REQUIRED (security paths): " + ", ".join(report["review_paths"]))
    return "\n".join(lines)
