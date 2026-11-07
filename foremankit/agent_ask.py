"""`foremankit agent-ask`: ask a headless coding agent ONE read-only question about a folder.
The answer, token counts and tool calls are read from the runner's raw JSON, and two git status reads prove the run
changed no file: anything new or different after the run is reverted. Nothing is ever committed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import agent_git as vcs
from . import agent_quota, agent_switch
from .agent_runner import Runner, RunnerError, RunnerResult, make_runner
from .config import load_config
from .state import CallLog, StateStore

PROMPT_HEADER = ("You are answering a question about the code in the current folder. This is READ-ONLY: do not create, "
                 "edit, rename or delete any file and do not commit. You may read files and run read-only commands. "
                 "Answer in plain text.")
CHANGED = "the agent changed files: "
USAGE_KEYS = ("tokens_in", "tokens_out", "tool_calls", "thinking_tokens", "cache_read_tokens", "turns")
QUESTION_SUFFIXES = {".md", ".txt"}
QUOTA_TEXT_MAX_CHARS = 300      # a real quota refusal is a short message


@dataclass(frozen=True)
class AskJob:
    """One read-only question about `project`."""
    task: str
    project: Path
    runner: str
    model: str = ""
    timeout: float = 900.0
    sensitivity: str = "normal"
    name: str = "task"


@dataclass(frozen=True)
class AskBatch:
    """A folder of question files answered one after another, with one result JSON each."""
    tasks_dir: Path
    out_dir: Path
    runner: str
    project: Path
    model: str = ""
    timeout: float = 900.0
    sensitivity: str = "normal"
    state_path: Path | None = None


def parse_usage(runner_name: str, raw: str) -> dict:
    """Token counts, tool calls and turns from a runner's raw stdout; unreadable fields stay None, never raises."""
    try:
        if runner_name == "opencode":
            return _opencode_usage(raw or "")
        if runner_name == "agy":
            return _agy_usage(raw or "")
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError):
        pass
    return _no_usage()


def _no_usage() -> dict:
    """Every usage field unknown."""
    return dict.fromkeys(USAGE_KEYS)


def _events(raw: str) -> list[dict]:
    """The JSON objects of opencode's one-object-per-line output (other lines are ignored)."""
    events = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _part(event: dict) -> dict:
    """The `part` object of an opencode event ({} when it is missing or not an object)."""
    part = event.get("part")
    return part if isinstance(part, dict) else {}


def _as_int(value: Any) -> int | None:
    """A JSON number or numeric string as int; None when it is missing or not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return None


def _tokens_of(step: dict) -> dict:
    """The `tokens` object of one opencode step_finish event."""
    tokens = _part(step).get("tokens")
    return tokens if isinstance(tokens, dict) else {}


def _sum_tokens(tokens: list[dict]) -> dict:
    """Sum opencode's per-step input, output, reasoning and cache.read counters."""
    return {"tokens_in": sum(_as_int(item.get("input")) or 0 for item in tokens),
            "tokens_out": sum(_as_int(item.get("output")) or 0 for item in tokens),
            "thinking_tokens": sum(_as_int(item.get("reasoning")) or 0 for item in tokens),
            "cache_read_tokens": sum(_cache_read(item) or 0 for item in tokens)}


def _cache_read(tokens: dict) -> int | None:
    """The cache.read counter of one step, when it is there."""
    cache = tokens.get("cache")
    return _as_int(cache.get("read")) if isinstance(cache, dict) else None


def _opencode_usage(raw: str) -> dict:
    """Sum every step_finish line and count tool_use lines; without a step_finish the token fields stay None."""
    events = _events(raw)
    steps = [event for event in events if event.get("type") == "step_finish"]
    usage = _no_usage()
    usage["tool_calls"] = sum(event.get("type") == "tool_use" for event in events)
    usage["turns"] = len(steps)
    if steps:
        usage.update(_sum_tokens([_tokens_of(step) for step in steps]))
    return usage


def _agy_usage(raw: str) -> dict:
    """agy's one JSON object: usage counters (numbers may be strings) and num_turns; it reports no tool calls."""
    usage = _no_usage()
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return usage
    if not isinstance(data, dict):
        return usage
    counters = data.get("usage")
    counters = counters if isinstance(counters, dict) else {}
    usage.update(tokens_in=_as_int(counters.get("input_tokens")),
                 tokens_out=_as_int(counters.get("output_tokens")),
                 thinking_tokens=_as_int(counters.get("thinking_tokens")),
                 cache_read_tokens=_as_int(counters.get("cache_read_tokens")),
                 turns=_as_int(data.get("num_turns")))
    return usage


def build_prompt(job: AskJob) -> str:
    """The fixed read-only rules plus the question."""
    return f"{PROMPT_HEADER}\n\nQUESTION:\n{job.task}"


def _seconds(started: float) -> float:
    """Wall time since `started`, rounded to a tenth of a second."""
    return round(time.perf_counter() - started, 1)


def ask(job: AskJob, runner: Runner | None = None, raw_dir: Path | None = None) -> dict:
    """Ask ONE question read-only; returns the answer, usage, timing and the files the agent changed (and reverted)."""
    result = _ask(job, runner, raw_dir)
    CallLog().write(_event(result))
    return result


def _ask(job: AskJob, runner: Runner | None, raw_dir: Path | None) -> dict:
    """Build or take the runner, refuse early when it may not run, then do the guarded run."""
    started = time.perf_counter()
    try:
        runner = runner or make_runner(job.runner, load_config(), job.model)
    except RunnerError as error:
        return _result(job, runner, {"error": str(error), "seconds": _seconds(started)})
    refusal = _refusal(job, runner)
    if refusal:
        return _result(job, runner, {"error": refusal, "seconds": _seconds(started)})
    return _run(job, runner, raw_dir, started)


def _refusal(job: AskJob, runner: Runner) -> str:
    """Why this question may not be asked at all, or "" when it may."""
    if job.sensitivity == "private" or not runner.eligible(job.sensitivity):
        return f"{runner.name} may not receive {job.sensitivity} data"
    if runner.name in agent_switch.RUNNERS:
        switched_off = agent_switch.refusal(StateStore().load(), runner.name)
        if switched_off:
            return switched_off
    if not job.project.is_dir():
        return f"project is not an existing folder: {job.project}"
    if not job.task.strip():
        return "the question is empty"
    return ""


def _run(job: AskJob, runner: Runner, raw_dir: Path | None, started: float) -> dict:
    """Run the agent between two git reads, revert any change it made and read usage from its raw output."""
    before = _status(job.project)
    try:
        run = runner.run(build_prompt(job), job.project, job.timeout)
    except RunnerError as error:
        return _result(job, runner, {"error": str(error), "seconds": _seconds(started)})
    changed = _revert(job.project, before, _status(job.project)) if before is not None else []
    extra = {"answer": _answer(runner.name, run), "seconds": _seconds(started), "modified_files": changed,
             "raw_path": _write_raw(raw_dir, job.name, run.raw), "error": _run_error(run, changed)}
    return _result(job, runner, extra, parse_usage(runner.name, run.raw))


def _run_error(result: RunnerResult, changed: list[str]) -> str:
    """Why the run failed after it finished: a timeout first, then the files it changed."""
    if result.timed_out:
        return "timed out"
    return f"{CHANGED}{', '.join(changed)}" if changed else ""


def _answer(runner_name: str, result: RunnerResult) -> str:
    """The full answer from the raw JSON; the flattened reply when the raw output carries no text."""
    return _raw_answer(runner_name, result.raw) or result.reply


def _raw_answer(runner_name: str, raw: str) -> str:
    """The full text of the runner's raw output ("" when it has none)."""
    if runner_name == "agy":
        return _agy_response(raw)
    if runner_name == "opencode":
        return _opencode_text(raw)
    return _opencode_text(raw) or _agy_response(raw)


def _opencode_text(raw: str) -> str:
    """The last `text` event's full text from opencode's JSON lines."""
    texts = [str(_part(event).get("text") or "") for event in _events(raw) if event.get("type") == "text"]
    return next((text for text in reversed(texts) if text), "")


def _agy_response(raw: str) -> str:
    """agy's `response` field from its one JSON object ("" when it printed no JSON)."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    return str(data.get("response") or "") if isinstance(data, dict) else ""


def _write_raw(raw_dir: Path | None, name: str, raw: str) -> str:
    """Write the runner's untouched stdout to `<name>.raw.txt` and return the path ("" without a folder)."""
    if raw_dir is None:
        return ""
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{name}.raw.txt"
        path.write_text(raw or "", encoding="utf-8")
    except OSError:
        return ""
    return str(path)


def _status(project: Path) -> dict[str, str] | None:
    """{path: porcelain status line} of the project, or None when it is not a git repository."""
    try:
        raw = vcs.git(["status", "--porcelain", "--untracked-files=all"], project)
    except vcs.AgentGitError:
        return None
    return {_status_path(line): line for line in raw.splitlines() if line.strip()}


def _status_path(line: str) -> str:
    """The path a porcelain status line is about (the new name for a rename)."""
    path = line[3:] if len(line) > 3 else line.strip()
    return path.split(" -> ")[-1].strip('"')


def _revert(project: Path, before: dict[str, str], after: dict[str, str] | None) -> list[str]:
    """Undo every path whose status line is new or different; paths that were dirty before are left alone."""
    current = after or {}
    changed = sorted(path for path, line in current.items() if before.get(path) != line)
    for path in changed:
        _revert_one(project, path, current[path])
    return changed


def _revert_one(project: Path, path: str, line: str) -> None:
    """Delete a new untracked file or restore a tracked one from the index; the report still names it."""
    try:
        if line.startswith("??"):
            (project / path).unlink()
        else:
            vcs.git(["checkout", "--", path], project)
    except (vcs.AgentGitError, OSError):
        pass


def _result(job: AskJob, runner: Runner | None, extra: dict, usage: dict | None = None) -> dict:
    """The exact result shape of one ask (every documented key, neutral values for what did not happen)."""
    usage = usage or {}
    error = str(extra.get("error") or "")
    return {"ok": not error, "name": job.name, "runner": runner.name if runner else job.runner,
            "model": runner.model if runner else job.model, "answer": str(extra.get("answer") or ""),
            "tokens_in": usage.get("tokens_in"), "tokens_out": usage.get("tokens_out"),
            "tool_calls": usage.get("tool_calls"), "thinking_tokens": usage.get("thinking_tokens"),
            "cache_read_tokens": usage.get("cache_read_tokens"), "turns": usage.get("turns"),
            "seconds": float(extra.get("seconds") or 0.0), "modified_files": list(extra.get("modified_files") or []),
            "raw_path": str(extra.get("raw_path") or ""), "error": error}


def _event(result: dict) -> dict:
    """The `outcome` log event of one ask (kind agent_ask), so stats and the panel see it."""
    return {"event": "outcome", "id": result["name"], "provider": result["runner"], "node": None,
            "model": result["model"], "kind": "agent_ask", "cost_usd": 0.0, "verified": False,
            "verify_passed": False, "fix_rounds": 0, "reviewed": False, "blockers": 0,
            "review_fixed": False, "accepted": result["ok"], "seconds": result["seconds"]}


def ask_batch(batch: AskBatch, runner: Runner | None = None) -> list[dict]:
    """Answer every question file in name order; finished tasks are skipped so a stopped batch can be resumed."""
    runner, error = _batch_runner(batch, runner)
    results, stopped = [], error
    for path in _question_files(batch.tasks_dir):
        stored = _stored(batch.out_dir / f"{path.stem}.json")
        if stored is not None:
            results.append(stored)
            continue
        if stopped:
            results.append(_failed(batch, path.stem, stopped))
            continue
        result, stopped = _batch_task(batch, path, runner)
        results.append(result)
    return results


def _batch_runner(batch: AskBatch, runner: Runner | None) -> tuple[Runner | None, str]:
    """The injected runner, or the one named by the batch; the error when it cannot be built."""
    if runner is not None:
        return runner, ""
    try:
        return make_runner(batch.runner, load_config(), batch.model), ""
    except RunnerError as error:
        return None, str(error)


def _question_files(tasks_dir: Path) -> list[Path]:
    """Every `*.md` and `*.txt` file in `tasks_dir`, sorted by file name."""
    try:
        files = [path for path in tasks_dir.iterdir()
                 if path.is_file() and path.suffix.lower() in QUESTION_SUFFIXES]
    except OSError:
        return []
    return sorted(files, key=lambda path: path.name)


def _stored(path: Path) -> dict | None:
    """The result JSON already in `out_dir`, or None when the task still has to run."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _batch_task(batch: AskBatch, path: Path, runner: Runner | None) -> tuple[dict, str]:
    """Quota-check, ask and store one question; also returns the reason the batch must stop ("" = go on)."""
    settings = load_config()["routing"].get("agent_run", {})
    state_file = batch.state_path or agent_quota.state_path()
    state = agent_quota.load_state(state_file)
    now = time.time()
    allowed, reason = agent_quota.can_start(state, batch.runner, now, settings)
    if not allowed:
        return _failed(batch, path.stem, reason), reason
    agent_quota.record_run(state, batch.runner, now)
    agent_quota.save_state(state_file, state)
    result = _ask_question(batch, path, runner)
    quota = {"state": state, "path": state_file, "settings": settings, "runner": batch.runner}
    stopped = _quota_stop(result, quota)
    _store_result(batch.out_dir, path.stem, result)
    return result, stopped


def _ask_question(batch: AskBatch, path: Path, runner: Runner | None) -> dict:
    """Read one question file and ask it with the batch settings, writing the raw output next to the result."""
    try:
        task = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return _result(_job(batch, path, ""), runner, {"error": f"cannot read the question: {error}"})
    return ask(_job(batch, path, task), runner=runner, raw_dir=batch.out_dir)


def _job(batch: AskBatch, path: Path, task: str) -> AskJob:
    """The AskJob for one question file of the batch."""
    return AskJob(task=task, project=batch.project, runner=batch.runner, model=batch.model,
                  timeout=batch.timeout, sensitivity=batch.sensitivity, name=path.stem)


def _quota_stop(result: dict, quota: dict) -> str:
    """Record a quota error and return the stop reason; "" when the reply is not a quota error."""
    text = result["answer"] or result["error"]
    if len(text) > QUOTA_TEXT_MAX_CHARS or not agent_quota.is_quota_error(text):
        return ""     # a long answer about code that mentions "rate limit" or 429 is an answer, not a quota error
    state, state_file = quota["state"], quota["path"]
    agent_quota.record_quota_error(state, quota["runner"], time.time(), quota["settings"])
    agent_quota.save_state(state_file, state)
    result["ok"], result["error"] = False, result["error"] or "quota error"
    return result["error"]


def _store_result(out_dir: Path, name: str, result: dict) -> None:
    """Write the result as `<name>.json` with indent 2, creating the folder when needed."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except OSError:
        pass


def _failed(batch: AskBatch, name: str, error: str) -> dict:
    """A failed result for a question that never started."""
    return _result(_job(batch, Path(name), ""), None, {"error": error})
