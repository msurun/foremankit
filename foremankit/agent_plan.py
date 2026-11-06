"""Batch plan loading, validation, and topological execution leveling for agent-run.
Validates task dependencies and scopes to ensure safe concurrent task execution.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from foremankit.agent_overlap import overlapping_pairs

VALID_RUNNERS = ("auto", "agy", "opencode")
VALID_SENSITIVITIES = ("normal", "internal")
DEFAULT_RUNNER = "auto"
DEFAULT_SENSITIVITY = "normal"


class PlanError(ValueError):
    """The plan is invalid; the message names the task id and the problem."""


@dataclass(frozen=True)
class PlanTask:
    """A validated task entry for batch agent-run execution."""

    id: str
    task_file: str
    scope: tuple[str, ...]
    check: str
    runner: str = DEFAULT_RUNNER
    model: str = ""
    sensitivity: str = DEFAULT_SENSITIVITY
    depends_on: tuple[str, ...] = ()


def parse_plan(data: Any) -> list[PlanTask]:
    """Parse and validate a batch plan mapping into a list of PlanTasks."""
    raw_tasks = _extract_raw_tasks(data)
    tasks: list[PlanTask] = []
    seen_ids: set[str] = set()
    for raw in raw_tasks:
        task = _parse_task_item(raw, seen_ids)
        seen_ids.add(task.id)
        tasks.append(task)
    _validate_task_dependencies(tasks, seen_ids)
    return tasks


def _extract_raw_tasks(data: Any) -> list[Any]:
    """Validate top-level plan structure and return the tasks list."""
    if not isinstance(data, dict):
        raise PlanError("plan must be a mapping with a 'tasks' list")
    if "tasks" not in data:
        raise PlanError("plan is missing required 'tasks' list")
    raw_tasks = data["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise PlanError("plan 'tasks' must be a non-empty list")
    return raw_tasks


def _parse_task_item(raw: Any, seen_ids: set[str]) -> PlanTask:
    """Validate and convert a raw task mapping into a PlanTask."""
    if not isinstance(raw, dict):
        raise PlanError("each task in 'tasks' must be a mapping")
    task_id, task_file, check = _validate_core_fields(raw, seen_ids)
    scope = _validate_scope(raw, task_id)
    runner, sensitivity, model = _validate_runner_fields(raw, task_id)
    depends_on = _validate_depends_on_list(raw, task_id)
    return PlanTask(
        id=task_id,
        task_file=task_file,
        scope=scope,
        check=check,
        runner=runner,
        model=model,
        sensitivity=sensitivity,
        depends_on=depends_on,
    )


def _validate_core_fields(raw: dict[str, Any], seen_ids: set[str]) -> tuple[str, str, str]:
    """Validate that id, task_file, and check are present and non-empty."""
    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise PlanError("task is missing a non-empty 'id'")
    clean_id = task_id.strip()
    if clean_id in seen_ids:
        raise PlanError(f"duplicate task id '{clean_id}'")
    task_file = raw.get("task_file")
    if not isinstance(task_file, str) or not task_file.strip():
        raise PlanError(f"task '{clean_id}' has missing or empty 'task_file'")
    check = raw.get("check")
    if not isinstance(check, str) or not check.strip():
        raise PlanError(f"task '{clean_id}' has missing or empty 'check'")
    return clean_id, task_file.strip(), check.strip()


def _validate_scope(raw: dict[str, Any], task_id: str) -> tuple[str, ...]:
    """Validate that scope is a non-empty list of non-empty string patterns."""
    if "scope" not in raw:
        raise PlanError(f"task '{task_id}' is missing required 'scope'")
    scope_list = raw["scope"]
    if not isinstance(scope_list, list) or not scope_list:
        raise PlanError(f"task '{task_id}' scope must be a non-empty list")
    for pattern in scope_list:
        if not isinstance(pattern, str) or not pattern.strip():
            raise PlanError(f"task '{task_id}' scope patterns must be non-empty strings")
    return tuple(pattern.strip() for pattern in scope_list)


def _validate_runner_fields(raw: dict[str, Any], task_id: str) -> tuple[str, str, str]:
    """Validate runner, sensitivity, and optional model settings."""
    runner = raw.get("runner", DEFAULT_RUNNER)
    if runner not in VALID_RUNNERS:
        raise PlanError(
            f"task '{task_id}' invalid runner '{runner}', must be one of auto, agy, opencode"
        )
    sensitivity = raw.get("sensitivity", DEFAULT_SENSITIVITY)
    if sensitivity == "private":
        raise PlanError(f"task '{task_id}': private data never goes to a runner")
    if sensitivity not in VALID_SENSITIVITIES:
        raise PlanError(
            f"task '{task_id}' invalid sensitivity '{sensitivity}', must be normal or internal"
        )
    if sensitivity == "internal" and runner == "opencode":
        raise PlanError(f"task '{task_id}': internal data may never go to opencode")
    model = raw.get("model", "")
    model_str = "" if model is None else model
    if not isinstance(model_str, str):
        raise PlanError(f"task '{task_id}' model must be a string")
    return runner, sensitivity, model_str


def _validate_depends_on_list(raw: dict[str, Any], task_id: str) -> tuple[str, ...]:
    """Validate that depends_on is a list of valid task id strings."""
    if "depends_on" not in raw:
        return ()
    depends = raw["depends_on"]
    if not isinstance(depends, list):
        raise PlanError(f"task '{task_id}' 'depends_on' must be a list of task ids")
    for dep in depends:
        if not isinstance(dep, str) or not dep.strip():
            raise PlanError(f"task '{task_id}' depends_on contains invalid id {dep!r}")
    return tuple(dep.strip() for dep in depends)


def _validate_task_dependencies(tasks: Sequence[PlanTask], valid_ids: set[str]) -> None:
    """Validate that dependencies do not target self or unknown tasks."""
    for task in tasks:
        for dep in task.depends_on:
            if dep == task.id:
                raise PlanError(f"task '{task.id}' cannot depend on itself")
            if dep not in valid_ids:
                raise PlanError(f"task '{task.id}' depends on unknown task '{dep}'")


def load_plan(path: Path) -> list[PlanTask]:
    """Read a YAML file (utf-8) and return the validated list of PlanTasks."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PlanError(f"cannot read plan file '{path}': {error}") from error
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise PlanError(f"invalid YAML syntax in plan file '{path}': {error}") from error
    return parse_plan(data)


def plan_levels(
    tasks: Sequence[PlanTask], tracked: Sequence[str] = ()
) -> list[list[PlanTask]]:
    """Partition tasks into parallel execution levels using Kahn's topological sort algorithm."""
    # In plain words:
    # We partition tasks into sequential waves (levels). Level 0 has tasks with no dependencies.
    # Level n contains tasks whose dependencies have all completed in earlier levels.
    # Tasks in the same level can run concurrently, so their scopes must not overlap.
    remaining = list(tasks)
    resolved_ids: set[str] = set()
    levels: list[list[PlanTask]] = []
    while remaining:
        current_level = _extract_ready_level(remaining, resolved_ids)
        if not current_level:
            cycle_ids = _detect_cycle_ids(remaining)
            raise PlanError(f"dependency cycle detected among tasks: {', '.join(cycle_ids)}")
        _verify_level_disjoint(current_level, tracked)
        levels.append(current_level)
        resolved_ids.update(task.id for task in current_level)
        remaining = [task for task in remaining if task.id not in resolved_ids]
    return levels


def _extract_ready_level(
    candidates: Sequence[PlanTask], resolved_ids: set[str]
) -> list[PlanTask]:
    """Find all tasks whose dependencies are already resolved, maintaining file order."""
    return [task for task in candidates if all(dep in resolved_ids for dep in task.depends_on)]


def _detect_cycle_ids(remaining: Sequence[PlanTask]) -> list[str]:
    """Find the task IDs participating in a dependency cycle among unresolved tasks."""
    # Intuition:
    # Every task remaining has at least one unresolved dependency. Following dependency links
    # within the finite set of unresolved tasks must eventually loop back on itself.
    task_map = {task.id: task for task in remaining}
    remaining_ids = set(task_map)
    visited: list[str] = []
    current_id = remaining[0].id
    while current_id not in visited:
        visited.append(current_id)
        current_task = task_map[current_id]
        current_id = next(dep for dep in current_task.depends_on if dep in remaining_ids)
    cycle_start = visited.index(current_id)
    return visited[cycle_start:]


def _verify_level_disjoint(level: Sequence[PlanTask], tracked: Sequence[str]) -> None:
    """Ensure no two tasks in the same parallel level share an overlapping scope."""
    scopes = {task.id: task.scope for task in level}
    pairs = overlapping_pairs(scopes, tracked)
    if pairs:
        id_a, id_b, reasons = pairs[0]
        reasons_text = "; ".join(reasons)
        raise PlanError(
            f"tasks '{id_a}' and '{id_b}' in the same level have overlapping scopes: {reasons_text}"
        )
