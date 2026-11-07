"""Exercise planner drafts with fake runners and real temporary repositories.
Ensure rejected drafts never publish files and accepted plans remain loadable.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from foremankit import agent_planner as planner
from foremankit.agent_git import git
from foremankit.agent_plan import load_plan
from foremankit.agent_runner import RunnerResult

TASK = {"id": "first", "title": "Add a value", "task_md":
        "# Value\n## Read first\nmodule.py\n## Build\nChange VALUE\n## Tests\ntest_value\n"
        "## Check\npython -m pytest", "scope": ["module.py"],
        "check": "python -m pytest", "depends_on": [], "runner": "auto"}


class FakeRunner:
    """Return a canned draft without contacting any vendor."""

    name = "codex"
    model = "test"

    def __init__(self, answer: str, edits: bool = False):
        """Remember the answer and whether to exercise the revert guard."""
        self.answer = answer
        self.edits = edits
        self.calls = []

    def eligible(self, sensitivity: str) -> bool:
        """Accept normal data only to exercise sensitivity filtering."""
        return sensitivity == "normal"

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Record the clean snapshot and optionally violate read-only rules."""
        self.calls.append((prompt, (cwd / "module.py").read_text(encoding="utf-8")))
        if self.edits:
            (cwd / "module.py").write_text("broken", encoding="utf-8")
        return RunnerResult(self.answer)


@pytest.fixture
def job(tmp_path, monkeypatch):
    """Create a committed project and isolate quota pauses from the machine."""
    project = tmp_path / "repo"
    project.mkdir()
    git(["init", "-q"], project)
    git(["config", "user.email", "test@example.invalid"], project)
    git(["config", "user.name", "Test"], project)
    (project / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(["add", "."], project)
    git(["commit", "-qm", "base"], project)
    monkeypatch.setattr(planner.agent_review.agent_quota, "state_path",
                        lambda: tmp_path / "quota.json")
    return planner.PlannerJob("Add a value", project, tmp_path / "draft")


def run_draft(job, tasks, lead_items=()):
    """Execute a draft through the real read-only ask guard."""
    runner = FakeRunner(json.dumps({"tasks": tasks, "lead_items": list(lead_items)}))
    return planner.plan(job, {"codex": runner}, {})


def test_valid_plan_is_written_and_loadable(job):
    """Batch execution reads the generated task file relative to the plan."""
    result = run_draft(job, [TASK])
    assert result["status"] == "ok"
    assert load_plan(job.out_dir / "plan.yaml")[0].task_file == "first.md"
    assert (job.out_dir / "first.md").read_text(encoding="utf-8") == TASK["task_md"]


@pytest.mark.parametrize("tasks,reason", [
    ([TASK, TASK], "duplicate"),
    ([{**TASK, "depends_on": ["missing"]}], "unknown"),
    ([{**TASK, "depends_on": ["second"]},
      {**TASK, "id": "second", "depends_on": ["first"]}], "cycle"),
    ([{**TASK, "scope": ["/absolute"]}], "absolute"),
    ([{**TASK, "scope": ["C:\\absolute"]}], "absolute"),
    ([{**TASK, "scope": ["../escape"]}], ".."),
    ([{**TASK, "scope": ["folder\\..\\escape"]}], ".."),
    ([{**TASK, "check": ""}], "check"),
    ([TASK, {**TASK, "id": "second"}], "overlap"),
    ([{**TASK, "scope": ["folder"]},
      {**TASK, "id": "second", "scope": ["folder/file.py"]}], "overlap"),
    ([{**TASK, "runner": "invented"}], "runner"),
    ([{**TASK, "id": "../bad"}], "id"),
])
def test_invalid_drafts_write_nothing(job, tasks, reason):
    """Every rejected draft leaves the output directory absent."""
    result = run_draft(job, tasks)
    assert result["status"] == "invalid"
    assert reason in " ".join(result["reasons"])
    assert not job.out_dir.exists()


def test_dependencies_make_shared_scopes_legal(job):
    """Shared files are safe when tasks are ordered."""
    result = run_draft(job, [TASK, {**TASK, "id": "second", "depends_on": ["first"]}])
    assert result["status"] == "ok"


def test_transitive_dependencies_make_shared_scopes_legal(job):
    """Indirect ordering also prevents simultaneous edits."""
    tasks = [TASK, {**TASK, "id": "second", "depends_on": ["first"]},
             {**TASK, "id": "third", "depends_on": ["second"]}]
    assert run_draft(job, tasks)["status"] == "ok"


def test_too_many_tasks_write_nothing(job):
    """The explicit task limit is enforced before publication."""
    result = run_draft(replace(job, max_tasks=1), [TASK, {**TASK, "id": "second"}])
    assert result["status"] == "invalid"
    assert not job.out_dir.exists()


def test_lead_items_are_written_for_claude(job):
    """Non-agent work is preserved beside the batch plan."""
    result = run_draft(job, [TASK], ["Review credential policy"])
    assert result["lead_items"] == ["Review credential policy"]
    assert "Review credential policy" in (job.out_dir / "lead_items.md").read_text()


def test_existing_files_are_preserved(job):
    """An occupied output directory is refused before asking a model."""
    job.out_dir.mkdir()
    existing = job.out_dir / "first.md"
    existing.write_text("owner draft", encoding="utf-8")
    result = run_draft(job, [TASK])
    assert result["status"] == "invalid"
    assert existing.read_text() == "owner draft"
    assert not (job.out_dir / "plan.yaml").exists()


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Draft: {} end"])
def test_wrapped_json_is_accepted(job, wrapper):
    """Model formatting does not discard a valid object."""
    runner = FakeRunner(wrapper.format(json.dumps({"tasks": [TASK]})))
    assert planner.plan(job, {"codex": runner}, {})["status"] == "ok"


def test_garbage_is_unparsable(job):
    """Non-JSON output never creates files."""
    assert planner.plan(job, {"codex": FakeRunner("garbage")}, {})["status"] == "unparsable"
    assert not job.out_dir.exists()


def test_no_planner_is_skipped(job):
    """Unavailable capacity returns a reason without invoking a runner."""
    result = planner.plan(job, {}, {})
    assert result["status"] == "skipped"
    assert result["reasons"]


def test_internal_data_excludes_ineligible_planners(job):
    """A preferred runner cannot override its sensitivity ceiling."""
    assert planner.plan(replace(job, sensitivity="internal"),
                        {"codex": FakeRunner("")}, {})["status"] == "skipped"


def test_snapshot_keeps_owner_edits_out_of_the_prompt(job):
    """The planner reads HEAD while the owner's dirty file stays untouched."""
    (job.project / "module.py").write_text("owner edit", encoding="utf-8")
    runner = FakeRunner(json.dumps({"tasks": [TASK]}))
    result = planner.plan(job, {"codex": runner}, {})
    assert result["status"] == "ok"
    assert runner.calls[0][1] == "VALUE = 1\n"
    assert (job.project / "module.py").read_text() == "owner edit"


def test_runner_edits_fail_the_read_only_guard(job):
    """A model that changes the snapshot cannot publish a plan."""
    runner = FakeRunner(json.dumps({"tasks": [TASK]}), edits=True)
    assert planner.plan(job, {"codex": runner}, {})["status"] == "failed"
    assert not job.out_dir.exists()


def test_goal_file_is_read_before_planning(job):
    """The at-file syntax passes file contents to the planner."""
    goal_file = job.project / "goal.txt"
    goal_file.write_text("Implement export", encoding="utf-8")
    runner = FakeRunner(json.dumps({"tasks": [TASK]}))
    result = planner.plan(replace(job, goal=f"@{goal_file}"), {"codex": runner}, {})
    assert result["status"] == "ok"
    assert "Implement export" in runner.calls[0][0]


def test_plan_event_records_task_count(job, tmp_path):
    """Planning uses a separate event from execution or review."""
    run_draft(job, [TASK])
    event = json.loads((tmp_path / "state" / "calls.jsonl").read_text().splitlines()[-1])
    assert (event["event"], event["task_count"], event["status"]) == ("agent_plan", 1, "ok")


def test_independent_tasks_in_different_levels_still_conflict(job):
    """A long-running task may overlap a later wave unless explicitly ordered."""
    tasks = [TASK, {**TASK, "id": "second", "scope": ["other.py"]},
             {**TASK, "id": "third", "depends_on": ["second"]}]
    result = run_draft(job, tasks)
    assert result["status"] == "invalid"
    assert "first and third" in " ".join(result["reasons"])


@pytest.mark.parametrize("scope", ["./module.py", " module.py ", "MODULE.PY", "."])
def test_equivalent_scopes_cannot_hide_overlap(job, scope):
    """Equivalent spellings and repository root scopes remain conflicting."""
    result = run_draft(job, [TASK, {**TASK, "id": "second", "scope": [scope]}])
    assert result["status"] == "invalid"
    assert not job.out_dir.exists()


def test_multiple_errors_are_reported_together(job):
    """Field failures do not hide independent dependency failures."""
    result = run_draft(job, [{**TASK, "check": "", "depends_on": ["missing"]}])
    assert "check" in " ".join(result["reasons"])
    assert "unknown dependency" in " ".join(result["reasons"])


def test_configuration_supplies_the_task_limit(job):
    """An omitted CLI limit uses agent_run.planner.max_tasks."""
    config = {"routing": {"agent_run": {"planner": {"max_tasks": 1}}}}
    answer = json.dumps({"tasks": [TASK, {**TASK, "id": "second"}]})
    result = planner.plan(job, {"codex": FakeRunner(answer)}, config)
    assert result["status"] == "invalid"
    assert "maximum is 1" in " ".join(result["reasons"])


def test_a_disabled_planner_is_skipped(job, monkeypatch):
    """Runner switches apply to planning as well as execution."""
    monkeypatch.setattr(planner.agent_review.agent_switch, "refusal",
                        lambda state, name: "runner is off")
    assert planner.plan(job, {"codex": FakeRunner("")}, {})["status"] == "skipped"


def test_a_paused_planner_is_skipped(job, monkeypatch):
    """Quota pauses apply before a planner is launched."""
    monkeypatch.setattr(planner.agent_review.agent_quota, "load_state",
                        lambda path: {"codex": {"paused_until": 1e20}})
    assert planner.plan(job, {"codex": FakeRunner("")}, {})["status"] == "skipped"


def test_configured_planner_order_is_respected(job):
    """Planning has no vendor restriction when selecting from its own order."""
    runner = FakeRunner(json.dumps({"tasks": [TASK]}))
    runner.name = "agy"
    config = {"routing": {"agent_run": {"planner": {"order": ["agy", "codex"]}}}}
    result = planner.plan(job, {"agy": runner, "codex": FakeRunner("garbage")}, config)
    assert result["planner"] == "agy"
    assert result["status"] == "ok"
