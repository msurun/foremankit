"""Tests for foremankit.agent_run, agent_git and agent_runner."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from foremankit import agent_run as ar
from foremankit.agent_run import (
    AgentRunError,
    AgentRunJob,
    agent_run,
    build_prompt,
    locate_program,
    render_text,
    worktree_env,
)
from foremankit.agent_runner import (
    AgyRunner,
    OpenCodeRunner,
    RunnerError,
    RunnerResult,
    make_runner,
)
from foremankit.providers import Provider
from foremankit.state import CallLog

CHECK = "python -m pytest -q"
IDENTITY = ["-c", "user.name=t", "-c", "user.email=t@t"]


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
    """Write `files` into `repo`, stage everything and commit on the current branch."""
    write_files(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


class FakeRunner:
    """Stands in for a headless agent: writes `writes` into the worktree it is given and records what it was sent."""

    def __init__(self, allow=True, writes=None, reply="PASS ok", timed_out=False, on_run=None):
        self.name, self.model, self.allow = "fake", "fake-1", allow
        self.writes = writes or {}
        self.reply, self.timed_out, self.on_run = reply, timed_out, on_run
        self.prompts: list[str] = []

    def eligible(self, sensitivity: str) -> bool:
        """The fake may receive what `allow` says."""
        return self.allow

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Record the call, write the files into `cwd` and return the canned reply."""
        self.prompts.append(prompt)
        write_files(cwd, self.writes)
        if self.on_run:
            self.on_run(cwd)
        return RunnerResult(self.reply, timed_out=self.timed_out)


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A git repo with tests/test_ok.py and README.md on main; worktrees land in tmp_path/tmpdir."""
    path = tmp_path / "proj"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    commit_all(path, {"tests/test_ok.py": "def test_ok(): assert True\n", "README.md": "# proj\n"}, "init")
    scratch = tmp_path / "tmpdir"
    scratch.mkdir()
    monkeypatch.setattr(ar.tempfile, "gettempdir", lambda: str(scratch))
    return path


def run_job(tmp_path: Path, runner: FakeRunner, **changes) -> dict:
    """agent_run with a default job; `changes` replace fields of AgentRunJob."""
    fields = {"task": "Add a test", "project": tmp_path / "proj", "scope": ("tests/test_new.py",), "check": CHECK}
    job = AgentRunJob(**{**fields, **changes})
    return agent_run(job, runner=runner, log=CallLog(tmp_path / "calls.jsonl"))


def branch_names(repo: Path) -> str:
    """`git branch --list agent-run/*` output."""
    return git(repo, "branch", "--list", "agent-run/*")


def test_happy_path_commits_the_in_scope_file(tmp_path, repo):
    report = run_job(tmp_path, FakeRunner(writes={"tests/test_new.py": "def test_new(): assert True\n"}))
    assert report["passed"] is True and report["check_passed"] is True
    assert report["files"] == ["tests/test_new.py"] and report["committed"] is True
    assert "agent-run/" in branch_names(repo)
    assert report["worktree"] is None
    assert list((tmp_path / "tmpdir" / "foremankit-agent-run").iterdir()) == []
    assert git(repo, "worktree", "list").count("\n") == 1
    assert git(repo, "status", "--porcelain") == ""
    events = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert any(event.get("kind") == "agent_run" for event in events)


def test_out_of_scope_changes_are_reverted_and_reported(tmp_path, repo):
    writes = {"tests/test_new.py": "def test_new(): assert True\n", "README.md": "changed\n", "notes.txt": "x\n"}
    report = run_job(tmp_path, FakeRunner(writes=writes))
    assert any(item.endswith("README.md") for item in report["reverted"])
    assert any(item.endswith("notes.txt") for item in report["reverted"])
    assert report["tracked_violations"] == ["README.md"]
    assert report["passed"] is False
    assert git(repo, "show", f"{report['branch']}:README.md") == "# proj\n"
    assert "notes.txt" not in git(repo, "ls-tree", "-r", "--name-only", report["branch"])


def test_failing_check_gives_passed_false_and_keeps_the_branch(tmp_path, repo):
    report = run_job(tmp_path, FakeRunner(writes={"tests/test_new.py": "def test_bad(): assert False\n"}))
    assert report["passed"] is False and report["check_passed"] is False
    assert report["check_output_tail"]
    assert report["branch"] in branch_names(repo)


def test_private_data_is_refused_before_any_worktree(tmp_path, repo):
    with pytest.raises(AgentRunError):
        run_job(tmp_path, FakeRunner(), sensitivity="private")
    assert not (tmp_path / "tmpdir" / "foremankit-agent-run").exists()


def test_internal_data_is_refused_when_the_runner_is_not_eligible(tmp_path, repo):
    with pytest.raises(AgentRunError):
        run_job(tmp_path, FakeRunner(allow=False), sensitivity="internal")
    assert not (tmp_path / "tmpdir" / "foremankit-agent-run").exists()


@pytest.mark.parametrize("changes", [{"scope": ()}, {"task": "   "}])
def test_empty_scope_or_blank_task_is_refused(tmp_path, repo, changes):
    with pytest.raises(AgentRunError):
        run_job(tmp_path, FakeRunner(), **changes)


def test_timed_out_run_is_still_checked_and_committed(tmp_path, repo):
    writes = {"tests/test_new.py": "def test_new(): assert True\n"}
    report = run_job(tmp_path, FakeRunner(writes=writes, timed_out=True))
    assert report["timed_out"] is True and report["files"] == ["tests/test_new.py"]
    assert report["check_passed"] is True and "passed" in report["check_output_tail"]


def test_no_changes_gives_passed_false(tmp_path, repo):
    report = run_job(tmp_path, FakeRunner(writes={}))
    assert report["passed"] is False and report["files"] == [] and report["committed"] is False


def test_long_task_goes_to_a_file_and_is_not_committed(tmp_path, repo):
    task = "Add the new test. " * 1800
    assert len(task) >= 30000

    def check_task_file(cwd: Path) -> None:
        assert (cwd / ".agent-task.md").read_text(encoding="utf-8") == task.strip()

    fake = FakeRunner(writes={"tests/test_new.py": "def test_new(): assert True\n"}, on_run=check_task_file)
    report = run_job(tmp_path, fake, task=task)
    assert len(fake.prompts[0]) < 24000 and ".agent-task.md" in fake.prompts[0]
    assert ".agent-task.md" not in git(tmp_path / "proj", "ls-tree", "-r", "--name-only", report["branch"])


def test_security_path_needs_review(tmp_path, repo):
    report = run_job(tmp_path, FakeRunner(writes={"gateway/auth.py": "x = 1\n"}), scope=("gateway/**",))
    assert report["review_required"] is True
    assert "REVIEW REQUIRED" in render_text(report)


def test_non_security_path_needs_no_review(tmp_path, repo):
    report = run_job(tmp_path, FakeRunner(writes={"tests/test_new.py": "def test_new(): pass\n"}))
    assert report["review_required"] is False


def test_check_runs_the_worktree_code_not_the_main_checkout(tmp_path, repo, monkeypatch):
    probe = {"src/mypkg/__init__.py": "WHERE = 'main'\n",
             "tests/test_where.py": "import mypkg\n\n\ndef test_where():\n    assert mypkg.WHERE == 'worktree'\n"}
    commit_all(repo, probe, "probe")
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))
    report = run_job(tmp_path, FakeRunner(writes={"src/mypkg/__init__.py": "WHERE = 'worktree'\n"}),
                     scope=("src/**",))
    assert report["passed"] is True


def test_worktree_env_puts_the_worktree_first_and_keeps_inherited_path(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    monkeypatch.setenv("PYTHONPATH", "/existing")
    paths = worktree_env(tmp_path)["PYTHONPATH"].split(os.pathsep)
    assert paths[0] == str(tmp_path) and str(tmp_path / "src") in paths
    assert paths[-1] == "/existing"


def test_locate_program_keeps_bare_names(tmp_path):
    assert locate_program("python", tmp_path, tmp_path) == "python"


def test_locate_program_finds_a_relative_venv_program_in_the_project_parent(tmp_path):
    name = "python.exe" if os.name == "nt" else "python"
    venv = tmp_path / "ForemanKit" / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / name).write_text("", encoding="utf-8")
    worktree, project = tmp_path / "wt", tmp_path / "main"
    worktree.mkdir()
    project.mkdir()
    assert locate_program("ForemanKit/.venv/Scripts/python", worktree, project) == str(venv / name)


def test_build_prompt_inlines_a_short_task(tmp_path):
    job = AgentRunJob(task="Fix it", project=tmp_path, scope=("tests/**",), check=CHECK)
    prompt = build_prompt(job, tmp_path, ["python", "-m", "pytest", "-q"])
    assert prompt.startswith("Fix it") and not (tmp_path / ".agent-task.md").exists()


def test_build_prompt_moves_a_long_task_to_a_file(tmp_path):
    job = AgentRunJob(task="x" * 30000, project=tmp_path, scope=("tests/**",), check=CHECK)
    prompt = build_prompt(job, tmp_path, ["python", "-m", "pytest", "-q"])
    assert ".agent-task.md" in prompt and (tmp_path / ".agent-task.md").read_text(encoding="utf-8") == "x" * 30000


def test_render_text_has_verdict_branch_and_check_lines():
    report = {"passed": True, "branch": "agent-run/abc", "runner": "fake", "model": "fake-1", "seconds": 1.0,
              "base": "0123456789", "files": ["tests/test_new.py"], "diff_stat": " 1 file changed",
              "check_passed": True, "check_output_tail": "1 passed", "reverted": [], "agent_reply": "PASS ok",
              "review_required": False, "review_paths": []}
    lines = render_text(report).splitlines()
    assert lines[0].startswith("PASS agent-run/abc (fake fake-1)")
    assert lines[3] == "check PASS: 1 passed" and not any("reverted" in line for line in lines)


AGY_CONFIG = {"type": "agy", "binary": "agy.exe", "enabled": True, "free": True, "trains_on_data": True,
              "models": {"flash_high": {"id": "gemini-3.8-flash-high"}}}


def agy_popen(stdout: str = "", raises: Exception | None = None, seen: list | None = None):
    """A subprocess.run stand-in for agy: records the command, raises `raises` or returns `stdout`."""
    def runner(command, **kwargs):
        if seen is not None:
            seen.append(command)
        if raises:
            raise raises
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)
    return runner


def test_agy_reply_is_parsed_from_json(tmp_path):
    seen: list = []
    runner = AgyRunner(Provider("agy", AGY_CONFIG), runner=agy_popen(json.dumps({"response": "PASS done"}), seen=seen))
    result = runner.run("do it", tmp_path, 10)
    assert result.reply == "PASS done" and result.timed_out is False
    assert "--sandbox" in seen[0] and "--dangerously-skip-permissions" in seen[0]
    assert "gemini-3.8-flash-high" in seen[0]


def test_agy_timeout_is_a_timed_out_result(tmp_path):
    runner = AgyRunner(Provider("agy", AGY_CONFIG),
                       runner=agy_popen(raises=subprocess.TimeoutExpired(cmd="agy", timeout=1)))
    result = runner.run("do it", tmp_path, 1)
    assert result.timed_out is True and result.exit_code is None


def test_agy_missing_binary_is_a_runner_error(tmp_path):
    runner = AgyRunner(Provider("agy", AGY_CONFIG), runner=agy_popen(raises=FileNotFoundError("agy")))
    with pytest.raises(RunnerError):
        runner.run("do it", tmp_path, 1)


OC_CONFIG = {"routing": {"agent_run": {"opencode": {"binary": "oc.exe", "default_model": "opencode/x-free"}}}}


def test_make_runner_builds_agy_and_refuses_unknown_names():
    runner = make_runner("agy", {"providers": {"agy": AGY_CONFIG}})
    assert runner.name == "agy" and runner.model == "gemini-3.8-flash-high"
    with pytest.raises(RunnerError):
        make_runner("nope", {"providers": {"agy": AGY_CONFIG}})


def test_make_runner_builds_opencode_from_the_routing_settings():
    assert make_runner("opencode", OC_CONFIG).model == "opencode/x-free"
    assert make_runner("opencode", OC_CONFIG, "opencode/other").model == "opencode/other"
    with pytest.raises(RunnerError):
        make_runner("opencode", {"routing": {}})


def test_opencode_takes_only_normal_data():
    runner = OpenCodeRunner("oc.exe", "opencode/x-free")
    assert [runner.eligible(level) for level in ("normal", "internal", "private")] == [True, False, False]


def oc_lines(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


def test_opencode_reply_is_the_last_text_event(tmp_path):
    seen: list = []
    out = oc_lines({"type": "step_start"}, {"type": "text", "part": {"text": "OK"}})
    reply = OpenCodeRunner("oc.exe", "m", agy_popen(out, seen=seen)).run("do it", tmp_path, 5).reply
    assert reply == "OK"
    assert "--auto" in seen[0] and seen[0][seen[0].index("--format") + 1] == "json" and "m" in seen[0]


@pytest.mark.parametrize("stdout, expected", [
    (oc_lines({"type": "error", "message": "quota"}), "quota"),
    ("not json at all", "FAIL opencode printed no usable output"),
])
def test_opencode_error_and_garbage_output(tmp_path, stdout, expected):
    reply = OpenCodeRunner("oc.exe", "m", agy_popen(stdout)).run("do it", tmp_path, 5).reply
    assert reply.startswith(expected)


def test_opencode_timeout_and_missing_binary(tmp_path):
    timed_out = OpenCodeRunner("oc.exe", "m", agy_popen(raises=subprocess.TimeoutExpired(cmd="oc", timeout=1)))
    assert timed_out.run("x", tmp_path, 1).timed_out is True
    missing = OpenCodeRunner("oc.exe", "m", agy_popen(raises=FileNotFoundError("oc")))
    with pytest.raises(RunnerError):
        missing.run("x", tmp_path, 1)


def test_agent_run_refuses_internal_data_for_opencode(tmp_path, repo):
    job = AgentRunJob("task", repo, ("a.py",), CHECK, sensitivity="internal")
    with pytest.raises(AgentRunError):
        agent_run(job, runner=OpenCodeRunner("oc.exe", "m"), log=CallLog(tmp_path / "calls.jsonl"))


def test_dependent_run_starts_from_the_predecessor_branch(tmp_path, repo):
    first = run_job(tmp_path, FakeRunner(writes={"tests/test_new.py": "def test_new(): assert True\n"}))
    seen = {}

    def see_predecessor(worktree: Path) -> None:
        seen["file"] = (worktree / "tests" / "test_new.py").exists()

    second = run_job(tmp_path, FakeRunner(writes={"tests/test_two.py": "def test_two(): assert True\n"},
                                          on_run=see_predecessor),
                     scope=("tests/test_two.py",), base=first["branch"])
    assert seen["file"] is True and second["files"] == ["tests/test_two.py"] and second["passed"] is True


def test_unknown_base_is_refused_without_leaving_a_worktree(tmp_path, repo):
    with pytest.raises(AgentRunError):
        run_job(tmp_path, FakeRunner(), base="agent-run/nope")
    assert git(repo, "worktree", "list").count("\n") == 1


@pytest.mark.parametrize("make", [
    lambda run: AgyRunner(Provider("agy", AGY_CONFIG), runner=run),
    lambda run: OpenCodeRunner("oc.exe", "m", run),
])
def test_runners_never_inherit_stdin(tmp_path, make):
    """opencode waits for the end of an inherited stdin (a pipe of a background process): hung for 10-30 minutes."""
    seen: dict = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout="{}", stderr="", returncode=0)

    make(run).run("do it", tmp_path, 5)
    assert seen["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("target", ["all", "agy"])
def test_agent_run_refuses_a_runner_switched_off_in_the_panel(tmp_path, repo, target):
    from foremankit import agent_switch
    from foremankit.state import StateStore
    agent_switch.set_switch(StateStore(), target, "off")
    runner = FakeRunner()
    runner.name = "agy"
    with pytest.raises(AgentRunError, match="(?i)switched off"):
        run_job(tmp_path, runner)
    assert git(repo, "worktree", "list").count("\n") == 1


def test_agent_run_ignores_the_switch_of_the_other_runner(tmp_path, repo):
    from foremankit import agent_switch
    from foremankit.state import StateStore
    agent_switch.set_switch(StateStore(), "opencode", "off")
    report = run_job(tmp_path, FakeRunner(writes={"tests/test_new.py": "def test_new(): assert True\n"}))
    assert report["passed"] is True


@pytest.mark.parametrize("make", [
    lambda run: AgyRunner(Provider("agy", AGY_CONFIG), runner=run),
    lambda run: OpenCodeRunner("oc.exe", "m", run),
])
def test_runners_set_pwd_to_the_run_folder(tmp_path, make):
    """opencode reads its working folder from PWD: an inherited PWD made it work in the wrong repository."""
    seen: dict = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout="{}", stderr="", returncode=0)

    make(run).run("do it", tmp_path, 5)
    assert seen["env"]["PWD"] == str(tmp_path) and seen["cwd"] == tmp_path
