"""Batch execution engine for topological levels of agent-run tasks.
Runs independent tasks concurrently within per-runner slot and quota limits,
propagating base git branches across dependency chains.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import agent_quota
from .agent_choice import RunnerFacts, decide_runner
from .agent_plan import PlanTask, plan_levels
from .agent_run import AgentRunError, AgentRunJob


@dataclass(frozen=True)
class BatchContext:
    """Execution context and injected dependencies for running an agent batch plan."""

    project: Path
    plan_dir: Path
    run_one: Callable[[AgentRunJob], dict[str, Any]]
    settings: dict[str, Any]
    state_path: Path
    clock: Callable[[], float] = time.time
    review_globs: Sequence[str] = ()
    tracked: Sequence[str] = ()
    disabled_runners: frozenset[str] = frozenset()   # switched off in the panel (agent_switch)


@dataclass(frozen=True)
class _TaskPlan:
    """Pre-processed task configuration ready for slot reservation and execution."""

    task: PlanTask
    task_text: str
    runner: str
    base: str
    multi_dep_reason: str


class _BatchState:
    """Encapsulates runtime synchronization and state tracking for a batch run."""

    def __init__(self, context: BatchContext, tasks: Sequence[PlanTask]) -> None:
        self.context = context
        self.quota_lock = threading.Lock()
        self.results: dict[str, dict[str, Any]] = {}
        self.semaphores: dict[str, threading.Semaphore] = {}
        self._init_semaphores(tasks)

    def _init_semaphores(self, tasks: Sequence[PlanTask]) -> None:
        """Create one semaphore per runner configured for this batch."""
        # In plain words:
        # Each runner type (e.g. agy, opencode) has its own maximum concurrency limit.
        # We pre-allocate a semaphore for every runner mentioned in the plan or settings.
        configured_runners = {"agy", "opencode"} | {t.runner for t in tasks if t.runner != "auto"}
        for runner in configured_runners:
            slot_count = agent_quota.slots(self.context.settings, runner)
            self.semaphores[runner] = threading.Semaphore(slot_count)

    def get_semaphore(self, runner: str) -> threading.Semaphore:
        """Retrieve or create the concurrency semaphore for a runner."""
        with self.quota_lock:
            if runner not in self.semaphores:
                slot_count = agent_quota.slots(self.context.settings, runner)
                self.semaphores[runner] = threading.Semaphore(slot_count)
            return self.semaphores[runner]


def run_batch(tasks: Sequence[PlanTask], context: BatchContext) -> dict[str, Any]:
    """Execute a validated batch plan level by level using concurrent worker threads.

    Respects per-runner slot limits and quota boundaries, recording all task outcomes
    in plan order into a JSON-serializable report.
    """
    levels = plan_levels(tasks, context.tracked)
    state = _BatchState(context, tasks)
    for level in levels:
        _execute_level(level, state)
    task_reports = [state.results[task.id] for task in tasks]
    return {
        "tasks": task_reports,
        "summary": _build_summary(task_reports),
    }


def _execute_level(level: list[PlanTask], state: _BatchState) -> None:
    """Execute all tasks in a single topological level concurrently."""
    # In plain words:
    # All tasks in this level have independent scopes and satisfied dependencies.
    # We submit them to a thread pool so each runner's tasks can progress concurrently
    # within its allotted semaphore slots. When exiting the with block, the executor
    # shuts down cleanly before any unexpected exception from future.result() propagates.
    with ThreadPoolExecutor(max_workers=max(len(level), 1)) as executor:
        futures = [executor.submit(_execute_task, task, state) for task in level]
        for future in futures:
            task_id, outcome = future.result()
            state.results[task_id] = outcome


def _execute_task(task: PlanTask, state: _BatchState) -> tuple[str, dict[str, Any]]:
    """Prepare and execute a single plan task."""
    early_exit, plan = _prepare_task(task, state)
    if early_exit is not None:
        return task.id, early_exit
    assert plan is not None
    return task.id, _run_with_slots(plan, state)


def _prepare_task(
    task: PlanTask, state: _BatchState
) -> tuple[dict[str, Any] | None, _TaskPlan | None]:
    """Validate dependencies, load task content, and resolve runner."""
    dep_error, base, multi_reason = _check_dependencies(task, state)
    if dep_error is not None:
        return dep_error, None
    text, read_error = _read_task_text(task, state.context.plan_dir)
    if read_error is not None:
        return read_error, None
    assert text is not None
    runner, runner_error = _resolve_runner(task, text, state.context)
    if runner_error is not None:
        return runner_error, None
    return None, _TaskPlan(task, text, runner, base, multi_reason)


def _check_dependencies(
    task: PlanTask, state: _BatchState
) -> tuple[dict[str, Any] | None, str, str]:
    """Verify that all dependencies passed, returning early skip or base branch details."""
    for dep_id in task.depends_on:
        dep_entry = state.results.get(dep_id)
        if dep_entry is None or dep_entry.get("status") != "passed":
            reason = f"dependency '{dep_id}' did not pass"
            return _task_entry(task.id, task.runner, "skipped", reason), "", ""
    if not task.depends_on:
        return None, "", ""
    first_dep = task.depends_on[0]
    base = state.results[first_dep].get("branch", "")
    others = ", ".join(task.depends_on[1:])
    reason = f"base is the branch of {first_dep}; {others} not included" if others else ""
    return None, base, reason


def _read_task_text(
    task: PlanTask, plan_dir: Path
) -> tuple[str | None, dict[str, Any] | None]:
    """Read task markdown file from plan directory."""
    path = plan_dir / task.task_file
    try:
        return path.read_text(encoding="utf-8"), None
    except (OSError, UnicodeDecodeError) as error:
        entry = _task_entry(task.id, task.runner, "refused", f"cannot read task file: {error}")
        return None, entry


def _resolve_runner(
    task: PlanTask, text: str, context: BatchContext
) -> tuple[str, dict[str, Any] | None]:
    """Resolve runner via heuristics if 'auto', returning runner or not_for_runner entry."""
    if task.runner != "auto":
        return task.runner, None
    facts = RunnerFacts(
        task_text=text, sensitivity=task.sensitivity, scope=task.scope, check=task.check,
        agy_enabled="agy" not in context.disabled_runners,
        opencode_enabled="opencode" not in context.disabled_runners,
    )
    choice, reasons = decide_runner(facts, context.review_globs)
    if choice in ("haiku", "lead"):
        first_reason = reasons[0] if reasons else ""
        return choice, _task_entry(task.id, choice, "not_for_runner", first_reason)
    return choice, None


def _run_with_slots(plan: _TaskPlan, state: _BatchState) -> dict[str, Any]:
    """Execute the task within its runner concurrency slot and quota boundaries."""
    # In plain words:
    # A runner (e.g. agy or opencode) can only run a limited number of tasks in parallel.
    # We acquire the runner's semaphore slot, check the quota under lock, and call run_one.
    if plan.runner in state.context.disabled_runners:
        return _task_entry(plan.task.id, plan.runner, "skipped", f"{plan.runner} is switched off in the panel")
    semaphore = state.get_semaphore(plan.runner)
    with semaphore:
        quota_rejection = _check_and_record_quota(plan.runner, state)
        if quota_rejection is not None:
            return _task_entry(plan.task.id, plan.runner, "skipped", quota_rejection)
        return _dispatch_run(plan, state)


def _check_and_record_quota(runner: str, state: _BatchState) -> str | None:
    """Check whether runner can start under quota, recording start on success."""
    with state.quota_lock:
        quota_state = agent_quota.load_state(state.context.state_path)
        now = state.context.clock()
        ok, reason = agent_quota.can_start(quota_state, runner, now, state.context.settings)
        if not ok:
            return reason
        agent_quota.record_run(quota_state, runner, now)
        agent_quota.save_state(state.context.state_path, quota_state)
        return None


def _dispatch_run(plan: _TaskPlan, state: _BatchState) -> dict[str, Any]:
    """Invoke injected run_one and translate result or AgentRunError into a task entry."""
    job = AgentRunJob(
        task=plan.task_text,
        project=state.context.project,
        scope=plan.task.scope,
        check=plan.task.check,
        runner=plan.runner,
        model=plan.task.model,
        sensitivity=plan.task.sensitivity,
        base=plan.base,
    )
    try:
        report = state.context.run_one(job)
    except AgentRunError as error:
        return _handle_run_error(plan, str(error), state)
    return _handle_run_report(plan, report, state)


def _handle_run_error(plan: _TaskPlan, message: str, state: _BatchState) -> dict[str, Any]:
    """Process an AgentRunError, checking if it indicates quota exhaustion."""
    if agent_quota.is_quota_error(message):
        _record_exhaustion(plan.runner, state)
        return _task_entry(plan.task.id, plan.runner, "quota_exhausted", message)
    return _task_entry(plan.task.id, plan.runner, "refused", message)


def _handle_run_report(
    plan: _TaskPlan, report: dict[str, Any], state: _BatchState
) -> dict[str, Any]:
    """Process normal completion report from run_one."""
    reply = str(report.get("agent_reply") or "")
    if not report.get("passed") and agent_quota.is_quota_error(reply):
        _record_exhaustion(plan.runner, state)
        return _run_entry(plan.task.id, plan.runner, "quota_exhausted", report)
    passed = bool(report.get("passed"))
    status = "passed" if passed else "failed"
    entry = _run_entry(plan.task.id, plan.runner, status, report)
    if plan.multi_dep_reason:
        entry["reason"] = plan.multi_dep_reason
    return entry


def _record_exhaustion(runner: str, state: _BatchState) -> None:
    """Record runner quota error and save updated state under lock."""
    with state.quota_lock:
        quota_state = agent_quota.load_state(state.context.state_path)
        now = state.context.clock()
        agent_quota.record_quota_error(quota_state, runner, now, state.context.settings)
        agent_quota.save_state(state.context.state_path, quota_state)


def _task_entry(task_id: str, runner: str, status: str, reason: str = "") -> dict[str, Any]:
    """Create a task entry with empty branch and files."""
    return {
        "id": task_id,
        "runner": runner,
        "status": status,
        "branch": "",
        "files": [],
        "reason": reason,
    }


def _run_entry(
    task_id: str, runner: str, status: str, report: dict[str, Any]
) -> dict[str, Any]:
    """Create a task entry from a run_one execution report."""
    reason = str(report.get("agent_reply") or "") if status == "quota_exhausted" else ""
    return {
        "id": task_id,
        "runner": runner,
        "status": status,
        "branch": str(report.get("branch") or ""),
        "files": list(report.get("files") or []),
        "reason": reason,
        "passed": status == "passed",
        "run_id": str(report.get("run_id") or ""),
        "model": str(report.get("model") or ""),
        "review_required": bool(report.get("review_required")),
        "review_paths": list(report.get("review_paths") or []),
    }


def _build_summary(task_reports: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Calculate aggregate task status counts."""
    statuses = [entry["status"] for entry in task_reports]
    return {
        "passed": statuses.count("passed"),
        "failed": statuses.count("failed"),
        "refused": statuses.count("refused"),
        "quota_exhausted": statuses.count("quota_exhausted"),
        "skipped": statuses.count("skipped"),
        "not_for_runner": statuses.count("not_for_runner"),
    }
