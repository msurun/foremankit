"""Tests for ForemanKit's agent_batch module.
Verifies multi-level parallel batch execution, slot gating, quota tracking,
and dependency propagation under simulated runner conditions.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from foremankit.agent_batch import BatchContext, run_batch
from foremankit.agent_plan import PlanError, PlanTask
from foremankit.agent_run import AgentRunError, AgentRunJob


def _create_task_file(plan_dir: Path, filename: str, content: str = "task content") -> str:
    """Create a task file and return its relative name."""
    (plan_dir / filename).write_text(content, encoding="utf-8")
    return filename


def _make_context(
    tmp_path: Path,
    run_one: Callable[[AgentRunJob], dict[str, Any]],
    settings: dict[str, Any] | None = None,
    review_globs: Sequence[str] = (),
) -> BatchContext:
    """Build a test BatchContext with temporary state and plan paths."""
    return BatchContext(
        project=tmp_path,
        plan_dir=tmp_path,
        run_one=run_one,
        settings=settings or {},
        state_path=tmp_path / "quota_state.json",
        review_globs=review_globs,
    )


def _make_four_opencode_tasks(tmp_path: Path) -> list[PlanTask]:
    """Create four independent opencode plan tasks for slot concurrency testing."""
    f1 = _create_task_file(tmp_path, "s1.md")
    f2 = _create_task_file(tmp_path, "s2.md")
    f3 = _create_task_file(tmp_path, "s3.md")
    f4 = _create_task_file(tmp_path, "s4.md")
    return [
        PlanTask(id="o1", task_file=f1, scope=("s1.py",), check="pytest", runner="opencode"),
        PlanTask(id="o2", task_file=f2, scope=("s2.py",), check="pytest", runner="opencode"),
        PlanTask(id="o3", task_file=f3, scope=("s3.py",), check="pytest", runner="opencode"),
        PlanTask(id="o4", task_file=f4, scope=("s4.py",), check="pytest", runner="opencode"),
    ]


def test_dependent_task_receives_branch_of_its_dependency_as_base(tmp_path: Path) -> None:
    """Verify that a dependent task receives its dependency's branch as base."""
    f1 = _create_task_file(tmp_path, "t1.md")
    f2 = _create_task_file(tmp_path, "t2.md")
    jobs: list[AgentRunJob] = []

    def fake_run_one(job: AgentRunJob) -> dict[str, Any]:
        jobs.append(job)
        return {"passed": True, "branch": f"branch-{job.scope[0]}", "files": []}

    context = _make_context(tmp_path, fake_run_one)
    t1 = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest")
    t2 = PlanTask(id="t2", task_file=f2, scope=("b.py",), check="pytest", depends_on=("t1",))
    run_batch([t1, t2], context)
    assert len(jobs) == 2
    assert jobs[1].base == "branch-a.py"


def test_report_preserves_original_plan_order_across_levels(tmp_path: Path) -> None:
    """Verify that the final report preserves the original plan ordering."""
    f1 = _create_task_file(tmp_path, "t1.md")
    f2 = _create_task_file(tmp_path, "t2.md")
    f3 = _create_task_file(tmp_path, "t3.md")
    context = _make_context(tmp_path, lambda job: {"passed": True, "branch": "b", "files": []})
    t1 = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest")
    t2 = PlanTask(id="t2", task_file=f2, scope=("b.py",), check="pytest")
    t3 = PlanTask(id="t3", task_file=f3, scope=("c.py",), check="pytest", depends_on=("t1",))
    report = run_batch([t1, t2, t3], context)
    task_ids = [t["id"] for t in report["tasks"]]
    assert task_ids == ["t1", "t2", "t3"]


def test_failed_dependency_causes_dependent_task_to_be_skipped(tmp_path: Path) -> None:
    """Verify that a task whose dependency failed is marked skipped."""
    f1 = _create_task_file(tmp_path, "t1.md")
    f2 = _create_task_file(tmp_path, "t2.md")
    context = _make_context(tmp_path, lambda job: {"passed": False, "branch": "b", "files": []})
    t1 = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest")
    t2 = PlanTask(id="t2", task_file=f2, scope=("b.py",), check="pytest", depends_on=("t1",))
    report = run_batch([t1, t2], context)
    assert report["tasks"][0]["status"] == "failed"
    assert report["tasks"][1]["status"] == "skipped"


def test_refused_dependency_causes_dependent_task_to_be_skipped(tmp_path: Path) -> None:
    """Verify that a task whose dependency was refused is marked skipped."""
    f1 = _create_task_file(tmp_path, "t1.md")
    f2 = _create_task_file(tmp_path, "t2.md")

    def fake_run_one(job: AgentRunJob) -> dict[str, Any]:
        raise AgentRunError("private sensitivity refused")

    context = _make_context(tmp_path, fake_run_one)
    t1 = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest")
    t2 = PlanTask(id="t2", task_file=f2, scope=("b.py",), check="pytest", depends_on=("t1",))
    report = run_batch([t1, t2], context)
    assert report["tasks"][0]["status"] == "refused"
    assert report["tasks"][1]["status"] == "skipped"


def test_parallel_slots_limit_maximum_concurrent_runs_to_configured_value(tmp_path: Path) -> None:
    """Verify that concurrent runs for a runner never exceed its configured slots."""
    current, max_observed = 0, 0
    lock = threading.Lock()

    def fake_run_one(job: AgentRunJob) -> dict[str, Any]:
        nonlocal current, max_observed
        with lock:
            current += 1
            max_observed = max(max_observed, current)
        time.sleep(0.02)
        with lock:
            current -= 1
        return {"passed": True, "branch": "b", "files": []}

    settings = {"max_parallel": {"opencode": 2}}
    context = _make_context(tmp_path, fake_run_one, settings=settings)
    tasks = _make_four_opencode_tasks(tmp_path)
    report = run_batch(tasks, context)
    assert max_observed == 2
    assert report["summary"]["passed"] == 4


def test_quota_error_exhausts_runner_and_skips_second_run_while_other_runner_passes(
    tmp_path: Path,
) -> None:
    """Verify quota error pauses runner and skips later runs while other runner passes."""
    f1 = _create_task_file(tmp_path, "a1.md")
    f2 = _create_task_file(tmp_path, "a2.md")
    f3 = _create_task_file(tmp_path, "o1.md")

    def fake_run_one(job: AgentRunJob) -> dict[str, Any]:
        if job.runner == "agy":
            raise AgentRunError("quota limit exceeded: rate limit (429)")
        return {"passed": True, "branch": "b-open", "files": []}

    context = _make_context(tmp_path, fake_run_one)
    t1 = PlanTask(id="agy1", task_file=f1, scope=("a.py",), check="pytest", runner="agy")
    t2 = PlanTask(id="agy2", task_file=f2, scope=("b.py",), check="pytest", runner="agy")
    t3 = PlanTask(id="open1", task_file=f3, scope=("c.py",), check="pytest", runner="opencode")
    report = run_batch([t1, t2, t3], context)
    state = json.loads((tmp_path / "quota_state.json").read_text(encoding="utf-8"))
    assert report["tasks"][0]["status"] == "quota_exhausted"
    assert report["tasks"][1]["status"] == "skipped"
    assert report["tasks"][2]["status"] == "passed"
    assert state["agy"]["paused_until"] > time.time()


def test_daily_limit_skips_second_task_when_exceeded(tmp_path: Path) -> None:
    """Verify reaching daily max runs skips subsequent tasks for that runner."""
    f1 = _create_task_file(tmp_path, "a1.md")
    f2 = _create_task_file(tmp_path, "a2.md")
    settings = {"quota": {"agy": {"max_runs_per_day": 1}}}
    context = _make_context(
        tmp_path, lambda job: {"passed": True, "branch": "b", "files": []}, settings=settings
    )
    t1 = PlanTask(id="agy1", task_file=f1, scope=("a.py",), check="pytest", runner="agy")
    t2 = PlanTask(id="agy2", task_file=f2, scope=("b.py",), check="pytest", runner="agy")
    report = run_batch([t1, t2], context)
    by_status = {entry["status"]: entry for entry in report["tasks"]}   # which thread gets the only slot first is not fixed
    assert sorted(entry["status"] for entry in report["tasks"]) == ["passed", "skipped"]
    assert "daily limit" in by_status["skipped"]["reason"]


def test_auto_runner_resolves_to_not_for_runner_for_review_glob_and_runner_otherwise(
    tmp_path: Path,
) -> None:
    """Verify auto runner resolves to lead for review globs and opencode otherwise."""
    f1 = _create_task_file(tmp_path, "t1.md")
    f2 = _create_task_file(tmp_path, "t2.md")
    context = _make_context(
        tmp_path,
        lambda job: {"passed": True, "branch": "b", "files": []},
        review_globs=["security/*"],
    )
    t1 = PlanTask(id="t1", task_file=f1, scope=("security/auth.py",), check="pytest")
    t2 = PlanTask(id="t2", task_file=f2, scope=("core/math.py",), check="pytest")
    report = run_batch([t1, t2], context)
    assert report["tasks"][0]["status"] == "not_for_runner"
    assert report["tasks"][0]["runner"] == "lead"
    assert "review" in report["tasks"][0]["reason"]
    assert report["tasks"][1]["status"] == "passed"
    assert report["tasks"][1]["runner"] == "opencode"


def test_unreadable_task_file_results_in_refused_status(tmp_path: Path) -> None:
    """Verify missing or unreadable task file produces refused status."""
    context = _make_context(tmp_path, lambda job: {"passed": True, "branch": "b", "files": []})
    task = PlanTask(id="t1", task_file="nonexistent.md", scope=("a.py",), check="pytest")
    report = run_batch([task], context)
    assert report["tasks"][0]["status"] == "refused"
    assert "cannot read task file" in report["tasks"][0]["reason"]


def test_unexpected_runtime_error_from_run_one_propagates_out_of_batch(tmp_path: Path) -> None:
    """Verify unhandled exception from run_one propagates after executor shutdown."""
    f1 = _create_task_file(tmp_path, "t1.md")

    def broken_run_one(job: AgentRunJob) -> dict[str, Any]:
        raise RuntimeError("catastrophic failure")

    context = _make_context(tmp_path, broken_run_one)
    task = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest")
    with pytest.raises(RuntimeError, match="catastrophic failure"):
        run_batch([task], context)


def test_summary_counts_match_all_task_statuses(tmp_path: Path) -> None:
    """Verify aggregate summary counts accurately reflect individual task statuses."""
    f1 = _create_task_file(tmp_path, "t1.md")
    f2 = _create_task_file(tmp_path, "t2.md")
    context = _make_context(
        tmp_path,
        lambda job: {"passed": True, "branch": "b", "files": []},
        review_globs=["guarded/*"],
    )
    t1 = PlanTask(id="p1", task_file=f1, scope=("a.py",), check="pytest")
    t2 = PlanTask(id="n1", task_file=f2, scope=("guarded/b.py",), check="pytest")
    t3 = PlanTask(id="r1", task_file="missing.md", scope=("c.py",), check="pytest")
    t4 = PlanTask(id="s1", task_file=f1, scope=("d.py",), check="pytest", depends_on=("r1",))
    report = run_batch([t1, t2, t3, t4], context)
    summary = report["summary"]
    assert summary["passed"] == 1
    assert summary["not_for_runner"] == 1
    assert summary["refused"] == 1
    assert summary["skipped"] == 1
    assert summary["failed"] == 0
    assert summary["quota_exhausted"] == 0


def test_multiple_dependencies_sets_lead_decision_reason(tmp_path: Path) -> None:
    """Verify tasks with multiple dependencies record the lead base choice in reason."""
    f1 = _create_task_file(tmp_path, "t1.md")
    f2 = _create_task_file(tmp_path, "t2.md")
    f3 = _create_task_file(tmp_path, "t3.md")
    context = _make_context(
        tmp_path, lambda job: {"passed": True, "branch": f"b-{job.scope[0]}", "files": []}
    )
    t1 = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest")
    t2 = PlanTask(id="t2", task_file=f2, scope=("b.py",), check="pytest")
    t3 = PlanTask(id="t3", task_file=f3, scope=("c.py",), check="pytest", depends_on=("t1", "t2"))
    report = run_batch([t1, t2, t3], context)
    assert report["tasks"][2]["reason"] == "base is the branch of t1; t2 not included"


def test_quota_error_in_agent_reply_marks_task_quota_exhausted(tmp_path: Path) -> None:
    """Verify quota error in agent_reply text marks task as quota exhausted."""
    f1 = _create_task_file(tmp_path, "t1.md")

    def reply_quota_run_one(job: AgentRunJob) -> dict[str, Any]:
        return {
            "passed": False,
            "branch": "b",
            "files": [],
            "agent_reply": "429 rate limit exceeded",
        }

    context = _make_context(tmp_path, reply_quota_run_one)
    task = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest", runner="agy")
    report = run_batch([task], context)
    assert report["tasks"][0]["status"] == "quota_exhausted"
    assert "rate limit" in report["tasks"][0]["reason"]


def test_plan_error_propagates_when_plan_is_invalid(tmp_path: Path) -> None:
    """Verify PlanError propagates immediately when topological leveling fails."""
    f1 = _create_task_file(tmp_path, "t1.md")
    f2 = _create_task_file(tmp_path, "t2.md")
    context = _make_context(tmp_path, lambda job: {"passed": True, "branch": "b", "files": []})
    t1 = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest", depends_on=("t2",))
    t2 = PlanTask(id="t2", task_file=f2, scope=("b.py",), check="pytest", depends_on=("t1",))
    with pytest.raises(PlanError):
        run_batch([t1, t2], context)


def test_passed_run_mentioning_quota_is_not_a_quota_error(tmp_path: Path) -> None:
    """A passed run about quotas must neither be marked exhausted nor pause its runner."""
    f1 = _create_task_file(tmp_path, "t1.md")

    def passed_run_one(job: AgentRunJob) -> dict[str, Any]:
        return {"passed": True, "branch": "b", "files": ["a.py"], "agent_reply": "added the quota module, 429 handled"}

    context = _make_context(tmp_path, passed_run_one)
    task = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest", runner="agy")
    report = run_batch([task], context)
    assert report["tasks"][0]["status"] == "passed"
    assert "paused_until" not in json.loads(context.state_path.read_text()).get("agy", {}) or \
        json.loads(context.state_path.read_text())["agy"]["paused_until"] == 0.0


def test_run_entry_carries_what_agent_merge_needs(tmp_path: Path) -> None:
    """The batch report entry has passed, run_id, model and the review flags of the run."""
    f1 = _create_task_file(tmp_path, "t1.md")

    def run_one(job: AgentRunJob) -> dict[str, Any]:
        return {"passed": True, "branch": "agent-run/x", "files": ["a.py"], "run_id": "x", "model": "m",
                "review_required": True, "review_paths": ["gateway/**"], "agent_reply": "ok"}

    task = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest", runner="agy")
    entry = run_batch([task], _make_context(tmp_path, run_one))["tasks"][0]
    assert entry["passed"] is True and entry["run_id"] == "x" and entry["model"] == "m"
    assert entry["review_required"] is True and entry["review_paths"] == ["gateway/**"]


@pytest.mark.parametrize("runner", ["agy", "codex"])
def test_disabled_runner_is_skipped_without_calling_run_one(tmp_path: Path, runner: str) -> None:
    """A runner switched off in the panel gets `skipped`, no quota run is counted, the other runner still runs."""
    files = [_create_task_file(tmp_path, f"t{n}.md") for n in (1, 2)]
    calls: list[str] = []

    def run_one(job: AgentRunJob) -> dict[str, Any]:
        calls.append(job.runner)
        return {"passed": True, "branch": "b", "files": []}

    context = _make_context(tmp_path, run_one)
    context = BatchContext(**{**context.__dict__, "disabled_runners": frozenset({runner})})
    tasks = [PlanTask(id="a", task_file=files[0], scope=("a.py",), check="pytest", runner=runner),
             PlanTask(id="o", task_file=files[1], scope=("o.py",), check="pytest", runner="opencode")]
    entries = {entry["id"]: entry for entry in run_batch(tasks, context)["tasks"]}
    assert entries["a"]["status"] == "skipped" and "switched off" in entries["a"]["reason"]
    assert entries["o"]["status"] == "passed" and calls == ["opencode"]
    assert runner not in json.loads(context.state_path.read_text())


def test_batch_dispatches_pinned_codex(tmp_path: Path) -> None:
    """An explicit Codex pin reaches the coding job and its own quota bucket."""
    filename = _create_task_file(tmp_path, "codex.md")
    jobs = []

    def run_one(job: AgentRunJob) -> dict:
        jobs.append(job)
        return {"passed": True, "branch": "codex-result", "files": []}

    context = _make_context(tmp_path, run_one)
    task = PlanTask(id="c", task_file=filename, scope=("c.py",), check="pytest", runner="codex")
    entry = run_batch([task], context)["tasks"][0]
    assert len(jobs) == 1 and jobs[0].runner == "codex" and entry["status"] == "passed"
    assert json.loads(context.state_path.read_text())["codex"]["runs"] == 1
