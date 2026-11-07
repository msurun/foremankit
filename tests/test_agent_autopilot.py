"""Exercise orchestration with fake stages instead of external model runners.
Keep safety and supervision outcomes visible at the public boundary.
"""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import yaml

from foremankit import agent_autopilot as module


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Create a fake plan and isolate read-only git safety queries."""
    job = module.AutopilotJob("Build feature", tmp_path, tmp_path / "work")
    plan_dir = job.work / "plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "task.md").write_text("Original task", encoding="utf-8")
    task = {"id": "task", "task_file": "task.md", "scope": ["feature.py"],
            "check": "python -m pytest", "sensitivity": "internal"}
    (plan_dir / "plan.yaml").write_text(yaml.safe_dump({"tasks": [task]}), encoding="utf-8")
    monkeypatch.setattr(module, "resolve_commit", Mock(return_value="original-base"))
    monkeypatch.setattr(module, "_check_target_files_clean", Mock())
    entry = {"id": "task", "runner": "codex", "status": "passed",
             "branch": "agent-run/first", "files": ["feature.py"], "passed": True}
    stages = module.Stages(Mock(return_value={"status": "ok", "lead_items": ["Design"]}),
                           Mock(return_value={"tasks": [entry]}),
                           Mock(return_value={"status": "ok", "reviewer": "agy", "findings": []}),
                           Mock(return_value={**entry, "branch": "agent-run/fixed"}))
    return job, stages


def execute(pipeline, **options):
    """Run a fixture with explicit controls and no configuration side effects."""
    job, stages = pipeline
    return module.autopilot(replace(job, **options), stages, {})


def test_clean_tasks_produce_a_report_and_success(pipeline):
    """A clean reviewed branch has a concrete merge suggestion."""
    result = execute(pipeline)
    report = pipeline[0].work / "autopilot-report.md"
    assert result["exit_code"] == 0
    assert "agent-merge" in report.read_text(encoding="utf-8")
    assert "--reviewed" in report.read_text(encoding="utf-8")


def test_invalid_plans_stop_before_execution(pipeline):
    """Planning failures retain their reasons and never start a batch."""
    pipeline[1].plan.return_value = {"status": "invalid", "reasons": ["overlap"]}
    result = execute(pipeline)
    assert result["reasons"] == ["overlap"]
    pipeline[1].batch.assert_not_called()
    assert result["exit_code"] == 2


def test_failed_tasks_are_listed_without_review(pipeline):
    """Execution failure is visible and is not repaired."""
    pipeline[1].batch.return_value["tasks"][0]["status"] = "failed"
    result = execute(pipeline)
    assert result["tasks"][0]["status"] == "failed"
    pipeline[1].review.assert_not_called()
    pipeline[1].followup.assert_not_called()
    assert result["exit_code"] == 2


def test_waiting_tasks_retain_their_retry_time(pipeline):
    """Quota pauses remain available to the existing resume command."""
    pipeline[1].batch.return_value["tasks"][0].update(status="waiting", retry_after=12345)
    result = execute(pipeline)
    assert result["tasks"][0]["retry_after"] == 12345
    assert "retry_after=12345" in (pipeline[0].work / "autopilot-report.md").read_text()
    pipeline[1].followup.assert_not_called()


def test_high_findings_trigger_exactly_one_fix_and_rereview(pipeline):
    """Persistent high findings cannot create an unbounded repair loop."""
    pipeline[1].review.return_value["findings"] = [{"severity": "high", "problem": "bug"}]
    result = execute(pipeline)
    pipeline[1].followup.assert_called_once()
    assert pipeline[1].review.call_count == 2
    assert result["exit_code"] == 2
    followup = pipeline[1].followup.call_args.args[0]
    assert followup.base == "agent-run/first"
    assert followup.scope == ("feature.py",)
    assert followup.sensitivity == "internal"
    assert followup.check == "python -m pytest"
    assert "Original task" in followup.task and "bug" in followup.task
    assert pipeline[1].review.call_args.args[0].source["base"] == "original-base"


def test_zero_fix_rounds_disable_repairs(pipeline):
    """An explicit zero overrides the default repair budget."""
    pipeline[1].review.return_value["findings"] = [{"severity": "high"}]
    execute(pipeline, max_fix_rounds=0)
    pipeline[1].followup.assert_not_called()


def test_a_successful_fix_replaces_old_findings(pipeline):
    """Only the last review determines the final high finding count."""
    pipeline[1].review.side_effect = [
        {"status": "ok", "findings": [{"severity": "medium"}]},
        {"status": "ok", "findings": []}]
    result = execute(pipeline)
    assert result["exit_code"] == 0
    assert result["tasks"][0]["counts_by_severity"]["medium"] == 0
    assert result["tasks"][0]["branch"] == "agent-run/fixed"


def test_skipped_reviews_are_flagged_unreviewed(pipeline):
    """Missing independent reviewers must remain visible to the lead."""
    pipeline[1].review.return_value = {"status": "skipped", "reason": "unavailable"}
    result = execute(pipeline)
    assert "unreviewed" in result["tasks"][0]["flags"]
    assert "agent-merge" not in (pipeline[0].work / "autopilot-report.md").read_text()


def test_review_paths_require_lead_inspection(pipeline):
    """Sensitive paths carry their lead flag even with no model findings."""
    pipeline[1].batch.return_value["tasks"][0]["review_paths"] = ["auth.py"]
    assert "lead must read" in execute(pipeline)["tasks"][0]["flags"]


def test_riskiest_tasks_sort_first():
    """Risk priorities follow the lead's inspection order."""
    clean = {"id": "clean", "status": "passed", "review": {"status": "ok"}}
    failed = {**clean, "id": "failed", "status": "failed"}
    lead = {**clean, "id": "lead", "review_required": True}
    unreviewed = {**clean, "id": "unreviewed", "review": {"status": "skipped"}}
    high = {**clean, "id": "high", "review": {"status": "ok",
            "findings": [{"severity": "high"}]}}
    assert sorted([clean, failed, lead, unreviewed, high], key=module._risk) == [
        high, unreviewed, lead, failed, clean]


def test_reports_have_a_bounded_line_count(pipeline):
    """Long planner explanations cannot overwhelm the supervision report."""
    pipeline[1].plan.return_value = {"status": "invalid", "reasons": ["reason\n" * 100] * 100}
    execute(pipeline)
    assert len((pipeline[0].work / "autopilot-report.md").read_text().splitlines()) <= 60


def test_plan_only_does_not_start_workers(pipeline):
    """Planning alone succeeds without batch or review calls."""
    assert execute(pipeline, plan_only=True)["exit_code"] == 0
    pipeline[1].batch.assert_not_called()
    pipeline[1].review.assert_not_called()


def test_dirty_scopes_refuse_before_any_worker_starts(pipeline, monkeypatch):
    """The existing cleanliness guard protects owner edits."""
    guard = Mock(side_effect=module.MergeRefused("uncommitted changes"))
    monkeypatch.setattr(module, "_check_target_files_clean", guard)
    assert execute(pipeline)["exit_code"] == 2
    guard.assert_called_once_with(pipeline[0].project, ["feature.py"])
    pipeline[1].batch.assert_not_called()


def test_orchestration_never_merges_or_pushes(pipeline, monkeypatch):
    """No orchestration stage may mutate the owner's branch."""
    from foremankit import agent_git, agent_merge
    git_call = Mock(side_effect=AssertionError("unexpected git mutation"))
    monkeypatch.setattr(agent_git, "git", git_call)
    monkeypatch.setattr(agent_merge, "git", git_call)
    execute(pipeline)
    git_call.assert_not_called()


def test_configuration_controls_the_maximum_fix_rounds(pipeline):
    """Configured repair limits apply when the CLI leaves them unspecified."""
    pipeline[1].review.return_value["findings"] = [{"severity": "high"}]
    result = module.autopilot(pipeline[0], pipeline[1], {
        "routing": {"agent_run": {"autopilot": {"max_fix_rounds": 2}}}})
    assert pipeline[1].followup.call_count == 2
    assert pipeline[1].review.call_count == 3
    assert result["tasks"][0]["fix_rounds"] == 2


def test_waiting_fixes_retain_the_branch_and_last_findings(pipeline):
    """A paused repair cannot turn an unresolved branch into a clean result."""
    pipeline[1].review.return_value["findings"] = [{"severity": "high"}]
    pipeline[1].followup.return_value = {"status": "waiting", "branch": "", "retry_after": 99}
    entry = execute(pipeline)["tasks"][0]
    assert entry["branch"] == "agent-run/first"
    assert entry["retry_after"] == 99
    assert entry["counts_by_severity"]["high"] == 1
    assert entry["passed"] is False
    assert "unreviewed" in entry["flags"]


def test_review_receives_the_original_sensitivity(pipeline):
    """Batch reports must not silently downgrade internal tasks for review."""
    execute(pipeline)
    assert pipeline[1].review.call_args.args[0].source["sensitivity"] == "internal"
