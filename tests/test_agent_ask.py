"""Tests for foremankit.agent_ask: read-only questions, usage parsing, the git guard and batches."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from foremankit import agent_ask as aa
from foremankit import agent_switch
from foremankit.agent_ask import AskBatch, AskJob, ask, ask_batch, parse_usage
from foremankit.agent_runner import RunnerError, RunnerResult
from foremankit.state import StateStore, utc_day

IDENTITY = ["-c", "user.name=t", "-c", "user.email=t@t"]
PROMPT_TAIL = "QUESTION:\nWhat does module.py do?"


def git(repo: Path, *args: str) -> str:
    """stdout of a git command run by the test itself (identity given inline)."""
    return subprocess.run(["git", *IDENTITY, *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def write_files(root: Path, files: dict[str, str]) -> None:
    """Create or overwrite text files under `root`."""
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def commit_all(repo: Path, files: dict[str, str], message: str = "c") -> None:
    """Write `files` into `repo`, stage everything and commit."""
    write_files(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


class FakeRunner:
    """Stands in for a headless agent: writes `writes` into the folder and returns the canned raw output."""

    def __init__(self, name="opencode", model="fake-model", raw="", reply="flattened reply", writes=None,
                 allow=True, timed_out=False, raises=None):
        self.name, self.model, self.raw, self.reply = name, model, raw, reply
        self.writes, self.allow, self.timed_out, self.raises = writes or {}, allow, timed_out, raises
        self.calls: list[str] = []

    def eligible(self, sensitivity: str) -> bool:
        """The fake may receive what `allow` says."""
        return self.allow

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Record the call, write the files into `cwd` and return the canned reply."""
        self.calls.append(prompt)
        write_files(cwd, self.writes)
        if self.raises:
            raise self.raises
        return RunnerResult(self.reply, timed_out=self.timed_out, raw=self.raw)


@pytest.fixture
def repo(tmp_path) -> Path:
    """A git repo with README.md and module.py committed on main."""
    path = tmp_path / "proj"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    commit_all(path, {"README.md": "# proj\n", "module.py": "VALUE = 1\n"}, "init")
    return path


def oc(*events: dict) -> str:
    """opencode's JSON-lines output from the given events."""
    return "\n".join(json.dumps(event) for event in events)


def oc_text(text: str) -> dict:
    """An opencode text event."""
    return {"type": "text", "part": {"type": "text", "text": text}}


def oc_tool() -> dict:
    """An opencode completed tool-call event."""
    return {"type": "tool_use", "part": {"type": "tool", "tool": "shell", "state": {"status": "completed"}}}


def oc_step(input_tokens=0, output_tokens=0, reasoning=0, cache_read=0) -> dict:
    """An opencode step_finish event with token counters."""
    tokens = {"input": input_tokens, "output": output_tokens, "reasoning": reasoning,
              "cache": {"read": cache_read, "write": 0}}
    return {"type": "step_finish", "part": {"type": "step-finish", "tokens": tokens}}


def agy(response="done", num_turns=1, **counters) -> str:
    """agy's one JSON object; every usage counter is a string, as the real CLI prints them."""
    usage = {"input_tokens": "39390", "output_tokens": "1944", "thinking_tokens": "1438",
             "cache_read_tokens": "150355", "total_tokens": "41334", **counters}
    return json.dumps({"status": "SUCCESS", "response": response, "duration_seconds": "70.6",
                       "num_turns": num_turns, "usage": usage})


def make_job(project: Path, **changes) -> AskJob:
    """An AskJob with a default question; `changes` replace fields."""
    fields = {"task": "What does module.py do?", "project": project, "runner": "opencode"}
    return AskJob(**{**fields, **changes})


def make_batch(tmp_path: Path, repo: Path, **changes) -> AskBatch:
    """An AskBatch over three question files (a, b, c) with an isolated quota state file."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for name in ("a.txt", "b.md", "c.md"):
        (tasks_dir / name).write_text(f"Question {name}", encoding="utf-8")
    fields = {"tasks_dir": tasks_dir, "out_dir": tmp_path / "out", "runner": "opencode", "project": repo,
              "state_path": tmp_path / "quota.json"}
    return AskBatch(**{**fields, **changes})


def test_parse_usage_opencode_sums_every_step_and_counts_tools():
    raw = oc(oc_text("x"), oc_tool(), oc_step(7855, 75), oc_tool(), oc_step(100, 5, reasoning=2, cache_read=7))
    usage = parse_usage("opencode", raw)
    assert usage == {"tokens_in": 7955, "tokens_out": 80, "tool_calls": 2, "thinking_tokens": 2,
                     "cache_read_tokens": 7, "turns": 2}


def test_parse_usage_opencode_without_step_finish_keeps_the_tool_count():
    usage = parse_usage("opencode", oc(oc_tool(), oc_tool()))
    assert usage["tokens_in"] is None and usage["tokens_out"] is None and usage["thinking_tokens"] is None
    assert usage["cache_read_tokens"] is None and usage["tool_calls"] == 2 and usage["turns"] == 0


def test_parse_usage_opencode_garbage_and_empty_count_nothing():
    for raw in ("", "not json", "[]", "{broken"):
        usage = parse_usage("opencode", raw)
        assert usage["tokens_in"] is None and usage["tool_calls"] == 0 and usage["turns"] == 0


def test_parse_usage_agy_reads_string_numbers():
    usage = parse_usage("agy", agy())
    assert usage == {"tokens_in": 39390, "tokens_out": 1944, "tool_calls": None, "thinking_tokens": 1438,
                     "cache_read_tokens": 150355, "turns": 1}


def test_parse_usage_agy_garbage_and_empty_is_all_none():
    for raw in ("", "not json", "[]", '{"usage": "x"}'):
        assert all(value is None for value in parse_usage("agy", raw).values())


def test_parse_usage_unknown_runner_is_all_none():
    assert all(value is None for value in parse_usage("nope", agy()).values())


def codex_raw() -> str:
    """Untouched JSONL captured from Codex CLI reading a synthetic answer.txt on 2026-11-06."""
    return (Path(__file__).parent / "fixtures" / "codex_exec.jsonl").read_text(encoding="utf-8")


def test_codex_usage_matches_live_sample():
    """One command appears twice in the stream; count its item ID once and keep all token counters."""
    assert parse_usage("codex", codex_raw()) == {
        "tokens_in": 29444, "tokens_out": 63, "tool_calls": 1, "thinking_tokens": 0,
        "cache_read_tokens": 27008, "turns": 1}


@pytest.mark.parametrize("raw", ["", "garbage", "[]", "{}", '{"type": []}', "{broken", None])
def test_codex_unreadable_usage_stays_unknown(raw):
    """Garbage is not evidence of zero usage."""
    assert all(value is None for value in parse_usage("codex", raw).values())
    assert aa._raw_answer("codex", raw) == ""


@pytest.mark.parametrize("counters", [{}, {"input_tokens": "bad", "output_tokens": []}, []])
def test_codex_missing_or_invalid_counters_stay_unknown(counters):
    """Malformed nested JSON must neither raise nor invent token totals."""
    raw = oc({"type": "turn.completed", "usage": counters}, {"type": "item.completed", "item": []})
    usage = parse_usage("codex", raw)
    assert usage["tokens_in"] is None and usage["tokens_out"] is None
    assert usage["thinking_tokens"] is None and usage["cache_read_tokens"] is None
    assert usage["turns"] == 1 and usage["tool_calls"] == 0
    assert aa._raw_answer("codex", raw) == ""


def test_codex_usage_sums_turns_and_deduplicates_tools():
    """Only tool items count, once per ID; final answers and reasoning are not tool calls."""
    raw = codex_raw() + oc({"type": "item.completed", "item": {"id": "mcp", "type": "mcp_tool_call"}},
                           {"type": "item.completed", "item": {"id": "reason", "type": "reasoning"}},
                           {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}})
    usage = parse_usage("codex", raw)
    assert (usage["tokens_in"], usage["tokens_out"], usage["tool_calls"], usage["turns"]) == (29454, 65, 2, 2)
    assert usage["thinking_tokens"] is None and usage["cache_read_tokens"] is None


def test_codex_malformed_tool_keeps_known_usage():
    """An invalid item type or ID cannot erase valid usage elsewhere in the stream."""
    raw = codex_raw() + oc({"type": "item.completed", "item": {"type": [], "id": {}}})
    assert parse_usage("codex", raw)["tokens_in"] == 29444
    assert parse_usage("codex", raw)["tool_calls"] == 1


def test_codex_ask_preserves_full_answer_and_raw(tmp_path, repo):
    """Ask uses the final agent message with newlines and the measured usage, without a live call."""
    raw = codex_raw()
    result = ask(make_job(repo, runner="codex"), runner=FakeRunner(name="codex", raw=raw), raw_dir=tmp_path)
    assert result["ok"] and result["answer"] == "42\nNo files changed."
    assert result["tokens_in"] == 29444 and result["tokens_out"] == 63 and result["tool_calls"] == 1
    assert result["modified_files"] == [] and Path(result["raw_path"]).read_text(encoding="utf-8") == raw


@pytest.mark.parametrize("sensitivity", ["internal", "private"])
def test_codex_ask_refuses_sensitive_data(repo, sensitivity):
    """Codex questions use the same eligibility gate as coding runs."""
    from foremankit.agent_runner import CodexRunner
    result = ask(make_job(repo, runner="codex", sensitivity=sensitivity), runner=CodexRunner("missing.exe", "m"))
    assert not result["ok"] and f"codex may not receive {sensitivity}" in result["error"]


def test_ask_happy_path_takes_the_full_answer_and_usage(tmp_path, repo):
    raw = oc(oc_text("line one\nline two"), oc_tool(), oc_step(7855, 75), oc_tool(),
             oc_step(100, 5, reasoning=2, cache_read=7))
    fake = FakeRunner(raw=raw, reply="flattened one line")
    raw_dir = tmp_path / "raw"
    result = ask(make_job(repo), runner=fake, raw_dir=raw_dir)
    assert result["ok"] is True and result["error"] == ""
    assert result["answer"] == "line one\nline two"
    assert (result["tokens_in"], result["tokens_out"], result["tool_calls"]) == (7955, 80, 2)
    assert (result["thinking_tokens"], result["cache_read_tokens"], result["turns"]) == (2, 7, 2)
    assert result["name"] == "task" and result["runner"] == "opencode" and result["model"] == "fake-model"
    assert isinstance(result["seconds"], float) and result["seconds"] >= 0
    assert result["modified_files"] == []
    assert result["raw_path"] == str(raw_dir / "task.raw.txt")
    assert (raw_dir / "task.raw.txt").read_text(encoding="utf-8") == raw
    assert fake.calls[0].startswith("You are answering a question about the code")
    assert fake.calls[0].endswith(PROMPT_TAIL)


def test_ask_logs_one_agent_ask_outcome(tmp_path, repo):
    ask(make_job(repo), runner=FakeRunner(raw=oc(oc_text("answer"))))
    events = [json.loads(line) for line in (tmp_path / "state" / "calls.jsonl").read_text().splitlines()]
    event = [item for item in events if item.get("kind") == "agent_ask"][-1]
    assert event["id"] == "task" and event["accepted"] is True and event["verified"] is False
    assert event["cost_usd"] == 0.0 and event["provider"] == "opencode"


def test_ask_reverts_a_tracked_file_the_agent_edited(repo):
    fake = FakeRunner(raw=oc(oc_text("answer")), writes={"module.py": "VALUE = 2\n"})
    result = ask(make_job(repo), runner=fake)
    assert result["ok"] is False and result["modified_files"] == ["module.py"]
    assert result["error"].startswith("the agent changed files: ")
    assert (repo / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git(repo, "status", "--porcelain") == ""


def test_ask_deletes_an_untracked_file_the_agent_created(repo):
    fake = FakeRunner(raw=oc(oc_text("answer")), writes={"notes/new.txt": "new\n"})
    result = ask(make_job(repo), runner=fake)
    assert result["ok"] is False and result["modified_files"] == ["notes/new.txt"]
    assert not (repo / "notes" / "new.txt").exists()
    assert git(repo, "status", "--porcelain") == ""


def test_ask_leaves_paths_that_were_dirty_before_the_run(repo):
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("before\n", encoding="utf-8")
    fake = FakeRunner(raw=oc(oc_text("answer")),
                      writes={"README.md": "agent\n", "scratch.txt": "after\n"})
    result = ask(make_job(repo), runner=fake)
    assert result["ok"] is True and result["modified_files"] == []
    assert (repo / "README.md").read_text(encoding="utf-8") == "agent\n"
    assert (repo / "scratch.txt").read_text(encoding="utf-8") == "after\n"


def test_ask_reads_the_answer_from_agy_json(tmp_path, repo):
    result = ask(make_job(repo, runner="agy"), runner=FakeRunner(name="agy", raw=agy("the full answer")))
    assert result["ok"] is True and result["answer"] == "the full answer"
    assert result["turns"] == 1 and result["tool_calls"] is None


def test_ask_falls_back_to_the_flattened_reply_without_raw_text(repo):
    result = ask(make_job(repo), runner=FakeRunner(raw="", reply="flattened"))
    assert result["ok"] is True and result["answer"] == "flattened"


def test_ask_private_data_is_refused_without_running(repo):
    fake = FakeRunner()
    result = ask(make_job(repo, sensitivity="private"), runner=fake)
    assert result["ok"] is False and "private" in result["error"] and fake.calls == []


def test_ask_ineligible_runner_is_refused_without_running(repo):
    fake = FakeRunner(allow=False)
    result = ask(make_job(repo, sensitivity="internal"), runner=fake)
    assert result["ok"] is False and "internal" in result["error"] and fake.calls == []


@pytest.mark.parametrize("runner", ["agy", "opencode", "codex"])
def test_ask_runner_switched_off_is_refused_without_running(repo, runner):
    """Each runner's switch refuses questions before any subprocess starts."""
    agent_switch.set_switch(StateStore(), runner, "off")
    fake = FakeRunner(name=runner)
    result = ask(make_job(repo, runner=runner), runner=fake)
    assert result["ok"] is False and "switched off" in result["error"] and fake.calls == []


def test_ask_empty_task_is_refused_without_running(tmp_path, repo):
    fake = FakeRunner()
    result = ask(make_job(repo, task="   "), runner=fake)
    assert result["ok"] is False and "empty" in result["error"] and fake.calls == []


def test_ask_missing_folder_is_refused_without_running(tmp_path):
    fake = FakeRunner()
    result = ask(make_job(tmp_path / "nope"), runner=fake)
    assert result["ok"] is False and "folder" in result["error"] and fake.calls == []


def test_ask_timeout_is_a_failed_result(repo):
    result = ask(make_job(repo), runner=FakeRunner(raw=oc(oc_text("x")), timed_out=True))
    assert result["ok"] is False and result["error"] == "timed out"


def test_ask_runner_error_is_a_failed_result(repo):
    result = ask(make_job(repo), runner=FakeRunner(raises=RunnerError("no binary")))
    assert result["ok"] is False and result["error"] == "no binary"


def test_ask_unknown_runner_name_is_a_failed_result(repo):
    result = ask(make_job(repo, runner="nope"))
    assert result["ok"] is False and "nope" in result["error"]


def test_ask_works_in_a_non_git_folder(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = ask(make_job(plain), runner=FakeRunner(raw=oc(oc_text("answer"))))
    assert result["ok"] is True and result["answer"] == "answer" and result["modified_files"] == []


def test_ask_batch_answers_in_order_and_writes_result_and_raw_files(tmp_path, repo):
    batch = make_batch(tmp_path, repo)
    raw = oc(oc_text("full answer"))
    results = ask_batch(batch, runner=FakeRunner(raw=raw))
    assert [result["name"] for result in results] == ["a", "b", "c"]
    assert all(result["ok"] for result in results)
    for name in ("a", "b", "c"):
        stored = json.loads((batch.out_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert stored["name"] == name and stored["answer"] == "full answer"
        assert (batch.out_dir / f"{name}.raw.txt").read_text(encoding="utf-8") == raw


def test_ask_batch_rerun_skips_finished_tasks(tmp_path, repo):
    batch = make_batch(tmp_path, repo)
    fake = FakeRunner(raw=oc(oc_text("answer")))
    first = ask_batch(batch, runner=fake)
    second = ask_batch(batch, runner=fake)
    assert len(fake.calls) == 3
    assert [result["name"] for result in second] == [result["name"] for result in first]
    assert all(result["ok"] for result in second)


def test_ask_batch_quota_error_stops_the_batch(tmp_path, repo):
    batch = make_batch(tmp_path, repo, runner="agy")
    fake = FakeRunner(name="agy", raw=agy("429 too many requests"))
    results = ask_batch(batch, runner=fake)
    assert len(fake.calls) == 1
    assert [result["ok"] for result in results] == [False, False, False]
    assert "quota" in results[0]["error"] and "429" in results[0]["answer"]
    assert (batch.out_dir / "a.json").is_file()
    assert not (batch.out_dir / "b.json").exists() and not (batch.out_dir / "c.json").exists()


def test_ask_batch_daily_limit_stops_before_the_first_task(tmp_path, repo, monkeypatch):
    settings = {"quota": {"agy": {"max_runs_per_day": 1, "cooldown_on_error_s": 60}}}
    monkeypatch.setattr(aa, "load_config", lambda: {"routing": {"agent_run": settings}})
    state_file = tmp_path / "quota.json"
    state_file.write_text(json.dumps({"agy": {"day": utc_day(time.time()), "runs": 1, "paused_until": 0.0}}),
                          encoding="utf-8")
    batch = make_batch(tmp_path, repo, runner="agy", state_path=state_file)
    fake = FakeRunner(name="agy", raw=agy("answer"))
    results = ask_batch(batch, runner=fake)
    assert all(result["ok"] is False for result in results)
    assert "daily limit" in results[0]["error"]
    assert fake.calls == [] and not batch.out_dir.exists()


def test_long_answer_mentioning_a_rate_limit_does_not_stop_the_batch(tmp_path):
    """Questions about code often say 'rate limit' or 429: only a short refusal counts as a quota error."""
    from foremankit import agent_ask
    long_answer = "The gateway retries after a 429 rate limit response. " * 20
    result = {"answer": long_answer, "error": ""}
    quota = {"state": {}, "path": tmp_path / "q.json", "settings": {}, "runner": "opencode"}
    assert agent_ask._quota_stop(result, quota) == ""
    short = {"answer": "Rate limit exceeded: quota exhausted", "error": ""}
    assert agent_ask._quota_stop(short, quota) != ""


@pytest.mark.parametrize("ceiling", ["normal", "internal"])
def test_codex_ask_internal_eligibility(repo, ceiling):
    """Question runs apply the configured Codex eligibility before sending internal data."""
    from foremankit.agent_runner import CodexRunner
    fake = FakeRunner(name="codex", raw=codex_raw())
    fake.eligible = CodexRunner("unused.exe", "m", max_sensitivity=ceiling).eligible
    result = ask(make_job(repo, runner="codex", sensitivity="internal"), runner=fake)
    assert result["ok"] is (ceiling == "internal")
    assert bool(fake.calls) is (ceiling == "internal")


LIMIT_RAW = json.dumps({"type": "error", "error": {"name": "APIError", "data": {
    "message": "Free usage limit reached", "statusCode": 429}}})


def test_provider_limit_event_is_a_failure_not_an_answer(repo):
    """Crosscut report: a 429 error event used to come back as ok=true with the error dict as the answer."""
    runner = FakeRunner(raw=LIMIT_RAW, reply="{'type': 'error'}")
    result = ask(AskJob("q", repo, "opencode"), runner=runner)
    assert result["ok"] is False
    assert result["answer"] == ""
    assert result["error"].startswith("provider_limit:")
    assert "Free usage limit reached" in result["error"]


def test_other_provider_error_is_tagged_but_a_real_answer_wins(repo):
    """A non-limit error is still a failure; an answer text next to a stray error event is kept."""
    boom = json.dumps({"type": "error", "error": {"message": "boom"}})
    failed = ask(AskJob("q", repo, "opencode"), runner=FakeRunner(raw=boom, reply=""))
    assert failed["error"].startswith("provider_error:")
    text = json.dumps({"type": "text", "part": {"text": "real answer"}})
    kept = ask(AskJob("q", repo, "opencode"), runner=FakeRunner(raw=boom + "\n" + text))
    assert kept["ok"] is True and kept["answer"] == "real answer"


def test_batch_stops_at_the_provider_limit_and_reports_the_rest(repo, tmp_path):
    """The quota stop covers provider limits: later questions fail with the reason instead of burning calls."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for number in range(4):
        (tasks / f"q{number}.md").write_text("question", encoding="utf-8")
    batch = AskBatch(tasks_dir=tasks, out_dir=tmp_path / "out", project=repo, runner="opencode",
                     state_path=tmp_path / "quota.json")
    runner = FakeRunner(raw=LIMIT_RAW)
    results = ask_batch(batch, runner=runner)
    assert [r["ok"] for r in results] == [False] * 4
    assert len(runner.calls) == 1   # the other three never started
    assert all(r["error"].startswith("provider_limit") for r in results)
