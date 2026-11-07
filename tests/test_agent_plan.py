"""Tests for ForemanKit's agent_plan module.
Ensures plan loading, validation, topological leveling, and scope overlap detection work.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from foremankit.agent_plan import (
    PlanError,
    PlanTask,
    load_plan,
    parse_plan,
    plan_levels,
)


def test_valid_minimal_task_applies_defaults() -> None:
    """Verify that omitting optional task fields populates default values."""
    raw = {"id": "task_a", "task_file": "docs/a.md", "scope": ["a.py"], "check": "pytest"}
    tasks = parse_plan({"tasks": [raw]})
    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "task_a"
    assert task.task_file == "docs/a.md"
    assert task.scope == ("a.py",)
    assert task.check == "pytest"
    assert (task.runner, task.model, task.sensitivity, task.depends_on) == (
        "auto",
        "",
        "normal",
        (),
    )


def test_diamond_dependency_produces_three_levels_in_file_order() -> None:
    """A diamond dependency (a; b,c depend on a; d depends on b,c) yields three levels."""
    tasks = [
        PlanTask(id="a", task_file="f", scope=("a.py",), check="c"),
        PlanTask(id="b", task_file="f", scope=("b.py",), check="c", depends_on=("a",)),
        PlanTask(id="c", task_file="f", scope=("c.py",), check="c", depends_on=("a",)),
        PlanTask(id="d", task_file="f", scope=("d.py",), check="c", depends_on=("b", "c")),
    ]
    levels = plan_levels(tasks)
    assert len(levels) == 3
    assert [task.id for task in levels[0]] == ["a"]
    assert [task.id for task in levels[1]] == ["b", "c"]
    assert [task.id for task in levels[2]] == ["d"]


def test_dependency_three_cycle_raises_plan_error() -> None:
    """A 3-cycle (a -> b -> c -> a) raises PlanError naming the cycle task IDs."""
    tasks = [
        PlanTask(id="a", task_file="f", scope=("a.py",), check="c", depends_on=("c",)),
        PlanTask(id="b", task_file="f", scope=("b.py",), check="c", depends_on=("a",)),
        PlanTask(id="c", task_file="f", scope=("c.py",), check="c", depends_on=("b",)),
    ]
    with pytest.raises(PlanError) as exc_info:
        plan_levels(tasks)
    message = str(exc_info.value)
    assert "dependency cycle detected" in message
    assert "a" in message and "b" in message and "c" in message


def test_self_dependency_raises_plan_error() -> None:
    """A task depending directly on itself raises PlanError naming the task ID."""
    raw = {
        "id": "task_self",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "depends_on": ["task_self"],
    }
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "task 'task_self' cannot depend on itself" in str(exc_info.value)


def test_unknown_dependency_raises_plan_error() -> None:
    """A task depending on a non-existent task ID raises PlanError naming both IDs."""
    raw = {
        "id": "task_a",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "depends_on": ["ghost_task"],
    }
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "task 'task_a' depends on unknown task 'ghost_task'" in str(exc_info.value)


def test_duplicate_task_ids_raise_plan_error() -> None:
    """Duplicate task IDs raise PlanError identifying the duplicated ID."""
    plan_data = {
        "tasks": [
            {"id": "dup", "task_file": "f1.md", "scope": ["a.py"], "check": "pytest"},
            {"id": "dup", "task_file": "f2.md", "scope": ["b.py"], "check": "pytest"},
        ]
    }
    with pytest.raises(PlanError) as exc_info:
        parse_plan(plan_data)
    assert "duplicate task id 'dup'" in str(exc_info.value)


def test_independent_tasks_with_overlapping_scopes_raise_plan_error() -> None:
    """Two independent tasks with overlapping scopes raise PlanError with IDs and reasons."""
    tasks = [
        PlanTask(id="task_x", task_file="f", scope=("src/shared.py",), check="pytest"),
        PlanTask(id="task_y", task_file="f", scope=("src/**",), check="pytest"),
    ]
    with pytest.raises(PlanError) as exc_info:
        plan_levels(tasks)
    message = str(exc_info.value)
    assert "task_x" in message and "task_y" in message
    assert "overlapping scopes" in message
    assert "src/shared.py matches src/**" in message


def test_dependent_tasks_with_overlapping_scopes_are_allowed() -> None:
    """Overlapping scopes are allowed when tasks run sequentially across different levels."""
    tasks = [
        PlanTask(id="base", task_file="f", scope=("src/shared.py",), check="pytest"),
        PlanTask(
            id="follow",
            task_file="f",
            scope=("src/**",),
            check="pytest",
            depends_on=("base",),
        ),
    ]
    levels = plan_levels(tasks)
    assert len(levels) == 2
    assert [task.id for task in levels[0]] == ["base"]
    assert [task.id for task in levels[1]] == ["follow"]


@pytest.mark.parametrize("runner", ["opencode", "codex"])
def test_internal_sensitivity_with_external_runner_is_rejected(runner) -> None:
    """Internal data can never go to opencode or Codex."""
    raw = {
        "id": "task_secret",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "sensitivity": "internal",
        "runner": runner,
    }
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert f"internal data may never go to {runner}" in str(exc_info.value)


def test_codex_is_valid_for_normal_data() -> None:
    """A batch may explicitly pin Codex for public or synthetic data."""
    task = {"id": "c", "task_file": "c.md", "scope": ["c.py"], "check": "pytest", "runner": "codex"}
    parsed = parse_plan({"tasks": [task]})[0]
    assert parsed.runner == "codex" and parsed.sensitivity == "normal"


def test_internal_sensitivity_with_agy_runner_is_accepted() -> None:
    """Sensitivity 'internal' paired with runner 'agy' is valid."""
    raw = {
        "id": "task_agy",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "sensitivity": "internal",
        "runner": "agy",
    }
    tasks = parse_plan({"tasks": [raw]})
    assert tasks[0].sensitivity == "internal"
    assert tasks[0].runner == "agy"


def test_internal_sensitivity_with_auto_runner_is_accepted() -> None:
    """Sensitivity 'internal' paired with runner 'auto' is valid."""
    raw = {
        "id": "task_auto",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "sensitivity": "internal",
        "runner": "auto",
    }
    tasks = parse_plan({"tasks": [raw]})
    assert tasks[0].sensitivity == "internal"
    assert tasks[0].runner == "auto"


def test_private_sensitivity_is_rejected() -> None:
    """Sensitivity 'private' is rejected with explicit explanation in PlanError."""
    raw = {
        "id": "task_priv",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "sensitivity": "private",
    }
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "private data never goes to a runner" in str(exc_info.value)


def test_empty_scope_list_raises_plan_error() -> None:
    """An empty scope list raises PlanError naming the task ID."""
    raw = {"id": "task_1", "task_file": "f.md", "scope": [], "check": "pytest"}
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "scope must be a non-empty list" in str(exc_info.value)


def test_non_list_scope_raises_plan_error() -> None:
    """A scope that is not a list raises PlanError naming the task ID."""
    raw = {"id": "task_1", "task_file": "f.md", "scope": "a.py", "check": "pytest"}
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "scope must be a non-empty list" in str(exc_info.value)


def test_missing_tasks_key_raises_plan_error() -> None:
    """A plan mapping missing the 'tasks' key raises PlanError."""
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"version": 1})
    assert "missing required 'tasks' list" in str(exc_info.value)


def test_invalid_yaml_syntax_raises_plan_error(tmp_path: Path) -> None:
    """Loading a plan file with invalid YAML syntax raises PlanError."""
    plan_file = tmp_path / "broken.yaml"
    plan_file.write_text("tasks: [unterminated_list", encoding="utf-8")
    with pytest.raises(PlanError) as exc_info:
        load_plan(plan_file)
    assert "invalid YAML syntax" in str(exc_info.value)


def test_load_plan_from_tmp_path_file(tmp_path: Path) -> None:
    """A valid YAML file is successfully loaded and parsed from disk."""
    plan_file = tmp_path / "valid.yaml"
    content = (
        "tasks:\n"
        "  - id: step_one\n"
        "    task_file: docs/one.md\n"
        "    scope: ['src/one.py']\n"
        "    check: 'pytest test_one.py'\n"
    )
    plan_file.write_text(content, encoding="utf-8")
    tasks = load_plan(plan_file)
    assert len(tasks) == 1
    assert tasks[0].id == "step_one"


def test_load_plan_missing_file_raises_plan_error(tmp_path: Path) -> None:
    """Loading a non-existent plan file raises PlanError."""
    missing_file = tmp_path / "nonexistent.yaml"
    with pytest.raises(PlanError) as exc_info:
        load_plan(missing_file)
    assert "cannot read plan file" in str(exc_info.value)


def test_top_level_not_a_mapping_raises_plan_error() -> None:
    """A non-dict root object raises PlanError."""
    with pytest.raises(PlanError) as exc_info:
        parse_plan(["tasks", "not", "dict"])
    assert "plan must be a mapping" in str(exc_info.value)


def test_task_entry_not_a_mapping_raises_plan_error() -> None:
    """A list item in 'tasks' that is not a mapping raises PlanError."""
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": ["just_a_string"]})
    assert "each task in 'tasks' must be a mapping" in str(exc_info.value)


def test_invalid_runner_raises_plan_error() -> None:
    """An unsupported runner value raises PlanError naming the task ID."""
    raw = {
        "id": "task_1",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "runner": "unsupported",
    }
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "invalid runner 'unsupported'" in str(exc_info.value)


def test_invalid_sensitivity_raises_plan_error() -> None:
    """An unsupported sensitivity label raises PlanError naming the task ID."""
    raw = {
        "id": "task_1",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "sensitivity": "confidential",
    }
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "invalid sensitivity 'confidential'" in str(exc_info.value)


def test_depends_on_not_a_list_raises_plan_error() -> None:
    """A non-list depends_on raises PlanError naming the task ID."""
    raw = {
        "id": "task_1",
        "task_file": "f.md",
        "scope": ["a.py"],
        "check": "pytest",
        "depends_on": "task_a",
    }
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "'depends_on' must be a list of task ids" in str(exc_info.value)


def test_missing_or_empty_id_raises_plan_error() -> None:
    """A task with an empty ID raises PlanError."""
    raw = {"id": "  ", "task_file": "f.md", "scope": ["a.py"], "check": "pytest"}
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "missing a non-empty 'id'" in str(exc_info.value)


def test_missing_or_empty_task_file_raises_plan_error() -> None:
    """A task with an empty task_file raises PlanError naming the task ID."""
    raw = {"id": "task_1", "task_file": "", "scope": ["a.py"], "check": "pytest"}
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "task 'task_1' has missing or empty 'task_file'" in str(exc_info.value)


def test_missing_or_empty_check_raises_plan_error() -> None:
    """A task with an empty check command raises PlanError naming the task ID."""
    raw = {"id": "task_1", "task_file": "f.md", "scope": ["a.py"], "check": " "}
    with pytest.raises(PlanError) as exc_info:
        parse_plan({"tasks": [raw]})
    assert "task 'task_1' has missing or empty 'check'" in str(exc_info.value)


def test_overlapping_scopes_on_tracked_files_raises_plan_error() -> None:
    """Two independent tasks whose scopes intersect on a tracked file raise PlanError."""
    tasks = [
        PlanTask(id="api", task_file="f", scope=("**/*.py",), check="pytest"),
        PlanTask(id="shared", task_file="f", scope=("**/shared.py",), check="pytest"),
    ]
    with pytest.raises(PlanError) as exc_info:
        plan_levels(tasks, tracked=["src/shared.py"])
    message = str(exc_info.value)
    assert "api" in message and "shared" in message
    assert "shared.py matches both scopes" in message
