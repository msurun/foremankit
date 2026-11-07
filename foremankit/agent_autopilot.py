"""Coordinate planning, execution and independent review without merging.
Publish a compact lead report while preserving resumable batch outcomes.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import agent_planner, agent_review
from .agent_git import AgentGitError, resolve_commit
from .agent_merge import MergeRefused, _check_target_files_clean
from .agent_plan import PlanTask, load_plan
from .agent_run import AgentRunError, AgentRunJob
from .config import load_config

DEFAULT_FIX_ROUNDS = 1
REPORT_LINE_LIMIT = 60  # Lines; the full structured report retains all details.


@dataclass(frozen=True)
class AutopilotJob:
    """Hold the goal and bounded execution controls."""

    goal: str
    project: Path
    work: Path
    plan_only: bool = False
    max_fix_rounds: int | None = None


@dataclass(frozen=True)
class Stages:
    """Inject model stages so orchestration can be tested without runners."""

    plan: Callable = agent_planner.plan
    batch: Callable | None = None
    review: Callable = agent_review.review
    followup: Callable | None = None


def _batch(plan_path: Path, project: Path) -> dict:
    """Use the same programmatic entry as agent-run-batch."""
    from .cli import _run_plan
    return _run_plan(plan_path, project)


def _followup(job: AgentRunJob) -> dict:
    """Run one fix through the public agent_run entry.

    agent_run itself refuses when the runner is switched off or paused, so no
    batch internals are needed; a refusal comes back as a non-passed outcome
    and the report shows it instead of the loop retrying.
    """
    from .agent_run import agent_run
    return agent_run(job)


def _counts(entry: dict) -> dict:
    """Count the last review's findings rather than accumulated fixed findings."""
    findings = entry.get("review", {}).get("findings", [])
    return {level: sum(item["severity"] == level for item in findings)
            for level in agent_review.SEVERITIES}


def _flags(entry: dict) -> list[str]:
    """Expose missing review and mandatory lead inspection explicitly."""
    flags = []
    if entry.get("review", {}).get("status") != "ok":
        flags.append("unreviewed")
    if entry.get("review_required") or entry.get("review_paths"):
        flags.append("lead must read")
    if entry["status"] != "passed":
        flags.append(entry["status"])
    return flags


def _risk(entry: dict) -> tuple:
    """Put high findings, missing reviews, lead paths and failures first."""
    flags = _flags(entry)
    return (-_counts(entry)["high"], "unreviewed" not in flags,
            "lead must read" not in flags, entry["status"] == "passed", entry["id"])


def _review(entry: dict, job: AutopilotJob, stages: Stages) -> None:
    """Review the complete branch against its original base after every fix."""
    entry["review"] = stages.review(agent_review.ReviewJob(dict(entry), job.project))


def _fix_job(entry: dict, task: PlanTask, job: AutopilotJob) -> AgentRunJob:
    """Retain the original task contract and append the review's repair list."""
    findings = json.dumps(entry["review"]["findings"], indent=2)
    text = entry["task_text"] + "\n\nFix these review findings:\n" + findings
    return AgentRunJob(text, job.project, task.scope, task.check, runner=entry["runner"],
                       model=entry.get("model", ""), sensitivity=task.sensitivity,
                       base=entry["branch"])


def _apply_fix(entry: dict, outcome: dict) -> None:
    """Preserve cumulative scope and lead flags across a follow-up branch."""
    protected = {key: entry[key] for key in
                 ("id", "base", "task_text", "scope", "sensitivity", "fix_rounds")}
    files = sorted(set(entry.get("files", [])) | set(outcome.get("files", [])))
    required = entry.get("review_required") or outcome.get("review_required")
    paths = sorted(set(entry.get("review_paths", [])) | set(outcome.get("review_paths", [])))
    branch = outcome.get("branch") or entry["branch"]
    entry.update(outcome, branch=branch)
    entry.update(protected, files=files, review_required=bool(required), review_paths=paths)
    entry["status"] = outcome.get("status", "passed" if outcome.get("passed") else "failed")
    entry["passed"] = entry["status"] == "passed"


def _process(entry: dict, task: PlanTask, context: tuple) -> None:
    """Review passed work and execute at most the configured repair rounds."""
    job, stages, maximum = context
    entry.update(scope=list(task.scope), sensitivity=task.sensitivity, fix_rounds=0)
    entry["task_text"] = (job.work / "plan" / task.task_file).read_text(encoding="utf-8")
    if entry["status"] != "passed":
        return
    # Batch entries omit base; resolve it before fixes create more commits.
    entry.setdefault("base", resolve_commit(job.project, f"{entry['branch']}^"))
    _review(entry, job, stages)
    for round_number in range(maximum):
        if not (_counts(entry)["high"] or _counts(entry)["medium"]):
            break
        outcome = (stages.followup or _followup)(_fix_job(entry, task, job))
        entry["fix_rounds"] = round_number + 1
        _apply_fix(entry, outcome)
        if entry["status"] != "passed":
            entry["review"].update(status="stale", reason="fix did not pass")
            break
        _review(entry, job, stages)


def _execute(job: AutopilotJob, stages: Stages, result: dict, maximum: int) -> None:
    """Validate all planned scopes before running any task."""
    planned = stages.plan(agent_planner.PlannerJob(job.goal, job.project, job.work / "plan"))
    result.update(plan=planned, reasons=planned.get("reasons", []))
    if planned["status"] != "ok":
        return
    if job.plan_only:
        result.update(status="planned", exit_code=0)
        return
    tasks = load_plan(job.work / "plan" / "plan.yaml")
    for task in tasks:
        _check_target_files_clean(job.project, list(task.scope))
    batch = (stages.batch or _batch)(job.work / "plan" / "plan.yaml", job.project)
    result["tasks"] = batch["tasks"]
    _write_batch_report(job, batch["tasks"])
    task_map = {task.id: task for task in tasks}
    for entry in result["tasks"]:
        _process(entry, task_map[entry["id"]], (job, stages, maximum))
    result["exit_code"] = 0 if all(entry["status"] == "passed" and not _counts(entry)["high"]
                                   for entry in result["tasks"]) else 2
    result["status"] = "ok" if result["exit_code"] == 0 else "attention"


def _write_json(path: Path, data: dict) -> None:
    """Persist machine-readable outcomes for resume and merge selection."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_batch_report(job: AutopilotJob, entries: list[dict]) -> None:
    """Preserve the batch report contract for resume and merge commands."""
    from .agent_batch import _build_summary
    _write_json(job.work / "report.json", {"tasks": entries, "summary": _build_summary(entries)})


def _merge_command(entry: dict, job: AutopilotJob) -> str:
    """Suggest a merge only for passed, reviewed branches without findings."""
    if entry["status"] != "passed" or "unreviewed" in _flags(entry) or any(_counts(entry).values()):
        return ""
    if not entry.get("branch", "").startswith("agent-run/"):
        return ""
    return (f'foremankit agent-merge --report "{job.work / "report.json"}" '
            f'--id "{entry["id"]}" --project "{job.project}" --reviewed')


def _task_lines(entry: dict, job: AutopilotJob) -> list[str]:
    """Render a compact task block with status, review counts and actionable flags."""
    review = entry.get("review", {})
    lines = [(f'{entry["id"]} | {entry.get("runner", "-")} | {entry.get("branch", "-")} | '
             f'check {entry["status"]} | reviewer {review.get("reviewer", "-")}'),
             (f'  {_counts(entry)} | {", ".join(_flags(entry)) or "clean"} | '
             f'{entry.get("reason", "")} {review.get("reason", "")} '
             f'{"retry_after=" + str(entry["retry_after"]) if "retry_after" in entry else ""}')]
    command = _merge_command(entry, job)
    return lines + ([command] if command else [])


def _single_line(value: object) -> str:
    """Keep model-supplied text from expanding the lead report unexpectedly."""
    return " ".join(str(value).splitlines())


def _publish(job: AutopilotJob, result: dict) -> None:
    """Write a bounded report and retain complete details in structured JSON."""
    result["tasks"].sort(key=_risk)
    for entry in result["tasks"]:
        entry.update(flags=_flags(entry), counts_by_severity=_counts(entry),
                     passed=entry["status"] == "passed")
    lines = [f'Goal: {_single_line(result["goal"])}', f'Status: {result["status"]}']
    lines.extend(_single_line(reason) for reason in result["reasons"])
    lines.append("Lead items: " + _single_line(result.get("plan", {}).get("lead_items", [])))
    lines.extend(_single_line(line) for entry in result["tasks"] for line in _task_lines(entry, job))
    if len(lines) > REPORT_LINE_LIMIT:
        lines = lines[:REPORT_LINE_LIMIT - 1] + ["More details: autopilot-result.json"]
    job.work.mkdir(parents=True, exist_ok=True)
    Path(result["report_path"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_batch_report(job, result["tasks"])
    _write_json(job.work / "autopilot-result.json", result)


def autopilot(job: AutopilotJob, stages: Stages | None = None, config: dict | None = None) -> dict:
    """Plan, run, review and repair; return a report and process exit code."""
    settings = (load_config() if config is None else config).get("routing", {}).get("agent_run", {})
    maximum = job.max_fix_rounds
    if maximum is None:
        maximum = settings.get("autopilot", {}).get("max_fix_rounds", DEFAULT_FIX_ROUNDS)
    result = {"goal": job.goal, "status": "failed", "tasks": [], "reasons": [], "exit_code": 2,
              "report_path": str(job.work / "autopilot-report.md")}
    try:
        if not isinstance(maximum, int) or maximum < 0:
            raise ValueError("max-fix-rounds must be a nonnegative integer")
        if job.goal.startswith("@"):
            result["goal"] = Path(job.goal[1:]).read_text(encoding="utf-8")
        _execute(job, stages or Stages(), result, maximum)
    except (ValueError, OSError, AgentGitError, AgentRunError, MergeRefused) as error:
        result.update(status="failed", exit_code=2, reasons=[str(error)])
    _publish(job, result)
    return result


def render_text(result: dict) -> str:
    """Print the report path and a three-line supervision summary."""
    entries = result["tasks"]
    return (f'{result["report_path"]}\nStatus: {result["status"]}\n'
            f'Tasks: {len(entries)}, passed: {sum(item["status"] == "passed" for item in entries)}\n'
            f'High findings: {sum(_counts(item)["high"] for item in entries)}')
