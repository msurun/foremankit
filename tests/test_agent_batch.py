"""Tests for ForemanKit's agent_batch module.
Verifies multi-level parallel batch execution, slot gating, quota tracking,
and dependency propagation under simulated runner conditions.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from foremankit.agent_batch import BatchContext, run_batch
from foremankit.agent_plan import PlanError, PlanTask
from foremankit.agent_run import AgentRunError, AgentRunJob

BATCH_TIMEOUT_S = 5


@pytest.mark.parametrize("slot_count", [0, -2])
def test_nonpositive_slots_skip_without_waiting(tmp_path: Path, slot_count: int) -> None:
    """An unavailable runner must return promptly without dispatching or counting quota."""
    script = """
import json
import sys
from pathlib import Path
from foremankit.agent_batch import BatchContext, run_batch
from foremankit.agent_plan import PlanTask
root = Path(sys.argv[1])
(root / 'task.md').write_text('task')
context = BatchContext(root, root, lambda job: sys.exit('runner was dispatched'),
                       {'max_parallel': {'codex': int(sys.argv[2])}}, root / 'quota.json')
task = PlanTask(id='task', task_file='task.md', scope=('a.py',), check='pytest', runner='codex')
print(json.dumps(run_batch([task], context)))
"""
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(slot_count)],
                               capture_output=True, text=True, check=True, timeout=BATCH_TIMEOUT_S)
    entry = json.loads(completed.stdout)["tasks"][0]
    assert entry["status"] == "skipped"
    assert entry["reason"] == "codex has max_parallel 0"
    assert not (tmp_path / "quota.json").exists()


@pytest.mark.parametrize("slot_count", [0, -2])
def test_auto_tasks_avoid_runners_without_slots(tmp_path: Path, slot_count: int) -> None:
    """Automatic routing excludes runners with no capacity before requesting a slot."""
    context = _make_context(tmp_path, lambda job: {"passed": True},
                            settings={"max_parallel": {"codex": slot_count}})
    task = PlanTask(id="task", task_file=_create_task_file(tmp_path, "task.md"),
                    scope=("a.py",), check="pytest")
    # Fail immediately on a zero-capacity semaphore instead of hanging this test process.
    assert_auto_runner_has_capacity(context)
    entry = run_batch([task], context)["tasks"][0]
    assert entry["status"] == "passed"
    assert entry["runner"] == "opencode"


def assert_auto_runner_has_capacity(context: BatchContext) -> None:
    """Check routing independently so a regression cannot block the test worker."""
    from foremankit.agent_batch import _BatchState, _resolve_runner

    task = PlanTask(id="task", task_file="task.md", scope=("a.py",), check="pytest")
    runner, refusal = _resolve_runner(task, "task", _BatchState(context, [task]))
    assert refusal is None
    assert runner == "opencode"


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
    t1 = PlanTask(id="t1", task_file=f1, scope=("a.py",), check="pytest", runner="codex")
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
    assert report["tasks"][0]["status"] == "waiting"
    assert report["tasks"][1]["status"] == "waiting"
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
    """Verify auto runner resolves to lead for review globs and Codex otherwise."""
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
    assert report["tasks"][1]["runner"] == "codex"


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
    assert report["tasks"][0]["status"] == "waiting"
    assert "limit: retry after" in report["tasks"][0]["reason"]


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


@pytest.mark.parametrize("runner", ["auto", "codex"])
def test_limit_fallback_or_wait(tmp_path: Path, runner: str) -> None:
    """Auto reroutes limits while a pinned runner waits until the parsed reset."""
    calls = []

    def run_one(job: AgentRunJob) -> dict:
        calls.append(job.runner)
        if len(calls) == 1:
            raise AgentRunError("usage limit, try again in 2 hours")
        return {"passed": True}

    context = _make_context(tmp_path, run_one)
    context = BatchContext(**{**context.__dict__, "clock": lambda: 1000.0})
    task = PlanTask(id="a", task_file=_create_task_file(tmp_path, "a.md"), scope=("a.py",),
                    check="pytest", runner=runner)
    entry = run_batch([task], context)["tasks"][0]
    assert json.loads(context.state_path.read_text())[calls[0]]["paused_until"] == 8200.0
    assert entry["status"] == ("passed" if runner == "auto" else "waiting")
    assert "rerouted from codex (limit)" in entry["reason"] if runner == "auto" else entry["retry_after"] == 8200


@pytest.mark.parametrize("runner,sensitivity", [("codex", "normal"), ("auto", "internal")])
def test_paused_tasks_and_dependents_wait(tmp_path: Path, runner: str, sensitivity: str) -> None:
    """Paused pins and internal tasks propagate waiting to dependents."""
    context = _make_context(tmp_path, lambda job: pytest.fail("paused runner started"))
    context = BatchContext(**{**context.__dict__, "clock": lambda: 1000.0})
    context.state_path.write_text(json.dumps({r: {"paused_until": 2000} for r in ("codex", "agy")}))
    file = _create_task_file(tmp_path, "a.md")
    tasks = [PlanTask(id="a", task_file=file, scope=("a.py",), check="pytest", runner=runner,
                      sensitivity=sensitivity),
             PlanTask(id="b", task_file=file, scope=("b.py",), check="pytest", depends_on=("a",))]
    report = run_batch(tasks, context)
    assert report["summary"]["waiting"] == 2
    assert all(e["retry_after"] == 2000 for e in report["tasks"])
    assert "dependency 'a'" in report["tasks"][1]["reason"]


def test_resume_uses_old_branch(tmp_path: Path) -> None:
    """Completed tasks return unchanged and supply their saved branch to children."""
    jobs = []
    context = _make_context(tmp_path, lambda job: jobs.append(job) or {"passed": True})
    old = {"id": "a", "status": "passed", "branch": "old-branch"}
    context = BatchContext(**{**context.__dict__, "done": {"a": old}})
    file = _create_task_file(tmp_path, "a.md")
    tasks = [PlanTask(id="a", task_file="missing.md", scope=("a.py",), check="pytest"),
             PlanTask(id="b", task_file=file, scope=("b.py",), check="pytest", depends_on=("a",))]
    report = run_batch(tasks, context)
    assert report["tasks"][0] is old
    assert len(jobs) == 1 and jobs[0].base == "old-branch"


def test_auto_skips_pause_and_normal_failure_resets_strikes(tmp_path: Path) -> None:
    """Auto avoids paused opencode and a failed check still clears codex strikes."""
    context = _make_context(tmp_path, lambda job: {"passed": job.runner == "agy"})
    context = BatchContext(**{**context.__dict__, "clock": lambda: 1000.0})
    context.state_path.write_text(json.dumps({"opencode": {"paused_until": 2000}, "codex": {"strikes": 2}}))
    task = PlanTask(id="a", task_file=_create_task_file(tmp_path, "a.md"), scope=("a.py",), check="pytest")
    entry = run_batch([task], context)["tasks"][0]
    assert entry["runner"] == "agy" and entry["status"] == "passed"
    assert json.loads(context.state_path.read_text())["codex"]["strikes"] == 0


def test_failed_run_that_changed_nothing_and_mentions_a_quota_is_a_limit_refusal(tmp_path: Path) -> None:
    """Only a failed run without changes counts as a usage-limit refusal; a failed check on a quota task does not."""
    files = [_create_task_file(tmp_path, f"t{n}.md") for n in (1, 2)]

    def run_one(job: AgentRunJob) -> dict[str, Any]:
        if job.scope == ("limit.py",):
            return {"passed": False, "branch": "", "files": [], "agent_reply": "You've hit your usage limit"}
        return {"passed": False, "branch": "b", "files": ["quota.py"], "agent_reply": "quota tests still fail"}

    tasks = [PlanTask(id="lim", task_file=files[0], scope=("limit.py",), check="pytest", runner="codex"),
             PlanTask(id="chk", task_file=files[1], scope=("quota.py",), check="pytest", runner="opencode")]
    entries = {entry["id"]: entry for entry in run_batch(tasks, _make_context(tmp_path, run_one))["tasks"]}
    assert entries["lim"]["status"] == "waiting" and entries["chk"]["status"] == "failed"
    state = json.loads(_make_context(tmp_path, run_one).state_path.read_text())
    assert state["codex"]["paused_until"] > 0 and state["opencode"]["paused_until"] == 0.0



@pytest.mark.parametrize("issues", [[], ["file.py:1: function-too-long"]])
def test_batch_entry_carries_convention_issues(issues):
    from foremankit.agent_batch import _run_entry

    entry = _run_entry("task", "codex", "passed", {"conventions_issues": issues})
    assert entry["conventions_issues"] == issues
    assert entry["conventions_issues"] is not issues


def test_batch_entry_defaults_to_no_convention_issues():
    from foremankit.agent_batch import _run_entry

    assert _run_entry("task", "codex", "passed", {})["conventions_issues"] == []


def test_batch_honours_the_configured_auto_order(tmp_path: Path) -> None:
    """Settings can restore opencode priority without editing routing code."""
    context = _make_context(tmp_path, lambda job: {"passed": True},
                            settings={"auto_order": {"normal": ["opencode", "codex"]}})
    task = PlanTask(id="a", task_file=_create_task_file(tmp_path, "a.md"),
                    scope=("a.py",), check="pytest")
    assert run_batch([task], context)["tasks"][0]["runner"] == "opencode"


def test_batch_retries_codex_twice_before_opencode(tmp_path: Path) -> None:
    """Only failures of this task in this batch consume its runner attempts."""
    calls = []

    def run_one(job: AgentRunJob) -> dict:
        calls.append(job.runner)
        return {"passed": job.runner == "opencode"}

    context = _make_context(tmp_path, run_one)
    task = PlanTask(id="a", task_file=_create_task_file(tmp_path, "a.md"),
                    scope=("a.py",), check="pytest")
    entry = run_batch([task], context)["tasks"][0]
    assert calls == ["codex", "codex", "opencode"]
    assert entry["status"] == "passed"


def test_batch_skips_paused_codex(tmp_path: Path) -> None:
    """A subscription pause makes Codex unavailable before dispatch."""
    context = _make_context(tmp_path, lambda job: {"passed": True})
    context.state_path.write_text(json.dumps({"codex": {"paused_until": time.time() + 2000}}))
    task = PlanTask(id="a", task_file=_create_task_file(tmp_path, "a.md"),
                    scope=("a.py",), check="pytest")
    assert run_batch([task], context)["tasks"][0]["runner"] == "opencode"


@pytest.mark.parametrize("sensitivity, expected", [
    ("normal", ["codex", "codex", "opencode", "opencode", "agy", "agy"]),
    ("internal", ["codex", "codex", "agy", "agy"]),
])
def test_batch_exhausts_each_allowed_runner_twice(tmp_path, sensitivity, expected):
    """Repeated check failures eventually return to the lead with no retry loop."""
    calls = []

    def run_one(job: AgentRunJob) -> dict:
        calls.append(job.runner)
        return {"passed": False}

    context = _make_context(tmp_path, run_one)
    task = PlanTask(id="a", task_file=_create_task_file(tmp_path, "a.md"),
                    scope=("a.py",), check="pytest", sensitivity=sensitivity)
    entry = run_batch([task], context)["tasks"][0]
    assert calls == expected
    assert entry["runner"] == "lead" and entry["reason"] == "escalation is exhausted"


def test_batch_failure_counts_are_specific_to_each_task(tmp_path: Path) -> None:
    """A previous task's failed Codex attempts do not change another task's priority."""
    jobs = []

    def run_one(job: AgentRunJob) -> dict:
        jobs.append((job.scope[0], job.runner))
        return {"passed": job.scope == ("b.py",) or job.runner == "opencode"}

    context = _make_context(tmp_path, run_one)
    filename = _create_task_file(tmp_path, "a.md")
    first = PlanTask(id="a", task_file=filename, scope=("a.py",), check="pytest")
    second = PlanTask(id="b", task_file=filename, scope=("b.py",),
                      check="pytest", depends_on=("a",))
    run_batch([first, second], context)
    assert jobs == [("a.py", "codex"), ("a.py", "codex"),
                    ("a.py", "opencode"), ("b.py", "codex")]
