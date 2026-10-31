"""Supervisor on the Antigravity CLI: `agy` writes or fixes ONE file and runs the check itself, with no Claude model in the loop.
The owner accepted full autonomy (--dangerously-skip-permissions: agy may run any command on this PC, 2026-10-31).
ForemanKit never trusts agy's own PASS: after agy ends it re-runs the check through the verify module.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agy_provider import MAX_PROMPT_CHARS
from .config import load_config
from .providers import Provider
from .verify import VerifyError, check_target, run_command, split_command

DEFAULT_TIMEOUT_S = 900
CHECK_TIMEOUT_S = 300
TASK_FILE = ".agent-task.md"
MODEL_ALIAS = "flash_high"
REPLY_CHARS = 200
TAIL_CHARS = 300
# --sandbox was tried on 2026-10-31 together with --dangerously-skip-permissions: agy still wrote the file and ran pytest,
# so it stays on as an extra restriction on terminal commands.
AGY_FLAGS = ["--output-format", "json", "--sandbox", "--dangerously-skip-permissions"]


class SuperviseError(RuntimeError):
    """The supervise request is refused before agy starts (private data, bad target, command not allowed)."""


@dataclass(frozen=True)
class SuperviseJob:
    """One supervised file: what to do, where, how to check it, and the folder the check runs in."""
    task: str
    target: str
    check: str
    project: Path
    sensitivity: str = "normal"
    timeout: float = DEFAULT_TIMEOUT_S


def supervise(job: SuperviseJob, runner=subprocess.run) -> dict[str, Any]:
    """Let agy do the task and run the check; return {passed, target, check_output_tail, agy_reply, seconds}."""
    config = load_config()
    agy = Provider("agy", config["providers"]["agy"])
    _refuse_if_not_allowed(job, agy, config)
    tokens = _check_tokens(job, config)
    started = time.perf_counter()
    reply = _run_agy(job, agy, runner)
    passed, tail = run_command(tokens, job.project, CHECK_TIMEOUT_S)
    return {"passed": passed, "target": job.target, "check_output_tail": tail[-TAIL_CHARS:].strip(),
            "agy_reply": reply[:REPLY_CHARS], "seconds": round(time.perf_counter() - started, 1)}


def _refuse_if_not_allowed(job: SuperviseJob, agy: Provider, config: dict[str, Any]) -> None:
    if job.sensitivity == "private" or not agy.eligible(job.sensitivity):
        raise SuperviseError(f"agy may not receive {job.sensitivity} data")
    try:
        relative = os.path.relpath(job.target, job.project)
        check_target(relative, job.project)
    except (VerifyError, ValueError) as error:
        raise SuperviseError(str(error)) from error


def _check_tokens(job: SuperviseJob, config: dict[str, Any]) -> list[str]:
    allowed = config["routing"]["verify"]["allowed_commands"]
    try:
        return split_command(job.check, allowed, job.project)
    except VerifyError as error:
        raise SuperviseError(str(error)) from error


def _build_prompt(job: SuperviseJob) -> str:
    """The task plus the fixed working rules; a task too long for the command line goes into a file next to the target."""
    task = job.task
    if len(task) > MAX_PROMPT_CHARS - 1000:
        task = f"The task is in the file {Path(job.target).parent / TASK_FILE}: read it first."
    return (f"{task}\n\nWrite or fix this file: {job.target}\nThen run exactly `{job.check}` in {job.project}. If it fails, "
            f"fix the file and run it again, at most 3 rounds. Do not touch any other file. "
            f"Your final reply is one line: PASS or FAIL and the reason.")


def _run_agy(job: SuperviseJob, agy: Provider, runner) -> str:
    """Start agy in the project folder and return its one-line reply ('' when it printed nothing usable)."""
    model = agy.model(MODEL_ALIAS)[1]
    task_file = Path(job.target).parent / TASK_FILE
    long_task = len(job.task) > MAX_PROMPT_CHARS - 1000
    if long_task:
        task_file.write_text(job.task, encoding="utf-8")
    command = [agy.agy_binary, "-p", _build_prompt(job), "--model", model, *AGY_FLAGS]
    try:
        return _reply_of(_start(runner, command, job))
    finally:
        if long_task:
            task_file.unlink(missing_ok=True)


def _start(runner, command: list[str], job: SuperviseJob):
    try:
        return runner(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                      timeout=job.timeout, cwd=job.project)
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError as error:
        raise SuperviseError(f"agy binary not found: {command[0]}") from error


def _reply_of(process) -> str:
    """agy's reply text from its JSON output; a short note when it timed out or printed something else."""
    if process is None:
        return "FAIL agy timed out"
    try:
        data = json.loads(process.stdout)
    except (TypeError, ValueError):
        return f"FAIL agy printed no JSON (exit {process.returncode})"
    return str(data.get("response") or data.get("error") or "").strip().replace("\n", " ")
