"""Tests for foremankit.agent_run, agent_git and agent_runner."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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
    TIMEOUT_REPLY,
    AgyRunner,
    CodexRunner,
    OpenCodeRunner,
    RunnerError,
    RunnerResult,
    make_runner,
    resolve_binary,
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


def test_resolve_binary_exact_path(tmp_path):
    """Plain paths are preserved even when the binary does not exist."""
    binary = str(tmp_path / "codex.exe")
    assert resolve_binary(binary) == binary


@pytest.mark.parametrize("versions", [("2.9.0", "2.10.0"), ("26.1021.9", "26.1021.22027")])
def test_resolve_binary_numeric_versions(tmp_path, versions):
    """Version components compare as numbers rather than text."""
    for version in versions:
        write_files(tmp_path, {f"openai.chatgpt-{version}-win32-x64/codex.exe": ""})
    pattern = str(tmp_path / "openai.chatgpt-*" / "codex.exe")
    assert resolve_binary(pattern) == str(tmp_path / f"openai.chatgpt-{versions[-1]}-win32-x64" / "codex.exe")


def test_resolve_binary_single_and_missing_matches(tmp_path):
    """One match resolves directly; no match preserves the pattern for error reporting."""
    pattern = str(tmp_path / "extension-*" / "codex.exe")
    assert resolve_binary(pattern) == pattern
    write_files(tmp_path, {"extension-1.0/codex.exe": ""})
    assert resolve_binary(pattern) == str(tmp_path / "extension-1.0" / "codex.exe")


def test_resolve_binary_expands_home(tmp_path, monkeypatch):
    """Home expansion applies to both plain paths and patterns."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    write_files(tmp_path, {"extension-1.0/codex.exe": ""})
    assert resolve_binary("~/codex.exe") == os.path.expanduser("~/codex.exe")
    assert resolve_binary("~/extension-*/codex.exe") == str(tmp_path / "extension-1.0" / "codex.exe")


def test_resolve_binary_version_ties_use_path_text(tmp_path):
    """Equal numeric versions have a deterministic text tie breaker."""
    for suffix in ("a", "z"):
        write_files(tmp_path, {f"extension-2.10.0-{suffix}/codex.exe": ""})
    assert resolve_binary(str(tmp_path / "extension-*" / "codex.exe")) == str(
        tmp_path / "extension-2.10.0-z" / "codex.exe")


@pytest.mark.parametrize("name", ["codex", "opencode"])
def test_make_runner_resolves_binary_pattern(tmp_path, name):
    """Both configured runners select the newest fake executable without running it."""
    for version in ("2.9.0", "2.10.0"):
        write_files(tmp_path, {f"extension-{version}/binary.exe": ""})
    config = {"routing": {"agent_run": {name: {
        "binary": str(tmp_path / "extension-*" / "binary.exe"), "default_model": "m"}}}}
    assert make_runner(name, config).binary == str(tmp_path / "extension-2.10.0" / "binary.exe")


def test_real_routing_codex_binary_is_a_pattern():
    """The repository routing YAML loads and retains the version-independent binary setting."""
    routing = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "routing.yaml").read_text(encoding="utf-8"))
    assert "*" in routing["agent_run"]["codex"]["binary"]
    assert routing["agent_run"]["quota"]["codex"] == {"cooldown_on_error_s": 1800}


def test_make_runner_builds_agy_and_refuses_unknown_names():
    runner = make_runner("agy", {"providers": {"agy": AGY_CONFIG}})
    assert runner.name == "agy" and runner.model == "gemini-3.8-flash-high"
    with pytest.raises(RunnerError):
        make_runner("nope", {"providers": {"agy": AGY_CONFIG}})


def test_make_runner_builds_opencode_from_the_routing_settings():
    assert make_runner("opencode", OC_CONFIG).binary == "oc.exe"
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
    lambda run: CodexRunner("codex.exe", "m", run),
])
def test_runners_never_inherit_stdin(tmp_path, make):
    """opencode waits for the end of an inherited stdin (a pipe of a background process): hung for 10-30 minutes."""
    seen: dict = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout="{}", stderr="", returncode=0)

    make(run).run("do it", tmp_path, 5)
    assert seen["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("target", ["all", "agy", "codex"])
def test_agent_run_refuses_a_runner_switched_off_in_the_panel(tmp_path, repo, target):
    from foremankit import agent_switch
    from foremankit.state import StateStore
    agent_switch.set_switch(StateStore(), target, "off")
    runner = FakeRunner()
    runner.name = "codex" if target == "codex" else "agy"
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
    lambda run: CodexRunner("codex.exe", "m", run),
])
def test_runners_set_pwd_to_the_run_folder(tmp_path, make):
    """opencode reads its working folder from PWD: an inherited PWD made it work in the wrong repository."""
    seen: dict = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout="{}", stderr="", returncode=0)

    make(run).run("do it", tmp_path, 5)
    assert seen["env"]["PWD"] == str(tmp_path) and seen["cwd"] == tmp_path


def test_codex_command_and_raw_output(tmp_path):
    """Use the measured headless flags and preserve the raw response with a short one-line reply."""
    seen = []
    raw = (Path(__file__).parent / "fixtures" / "codex_exec.jsonl").read_text(encoding="utf-8")
    result = CodexRunner("codex.exe", "gpt-6.1-sol", agy_popen(raw, seen=seen)).run("-prompt", tmp_path, 12)
    command = seen[0]
    for flag, value in (("--ask-for-approval", "never"), ("--sandbox", "workspace-write"),
                        ("--cd", str(tmp_path)), ("--model", "gpt-6.1-sol")):
        assert command[command.index(flag) + 1] == value
    assert "exec" in command and "--json" in command and "--ignore-user-config" in command
    assert "sandbox_workspace_write.exclude_tmpdir_env_var=true" in command
    assert "sandbox_workspace_write.exclude_slash_tmp=true" in command
    assert command[-2:] == ["--", "-prompt"]
    assert result.raw == raw and result.reply == "42 No files changed." and result.exit_code == 0


def test_codex_drops_parent_permission_profile(tmp_path, monkeypatch):
    """A Codex app with unrestricted permissions cannot grant them to the child runner."""
    seen = {}
    monkeypatch.setenv("CODEX_PERMISSION_PROFILE", "unrestricted-parent")

    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout="", returncode=1)

    result = CodexRunner("codex.exe", "m", run).run("x", tmp_path, 12)
    assert "CODEX_PERMISSION_PROFILE" not in seen["env"] and seen["timeout"] == 12
    assert result.exit_code == 1 and "exit 1" in result.reply


def test_codex_recovered_error_does_not_hide_final_answer(tmp_path):
    """A transient stream error followed by a successful answer is not a failed run."""
    raw = oc_lines({"type": "error", "message": "reconnecting"},
                   {"type": "item.completed", "item": {"type": "agent_message", "text": "PASS done"}})
    result = CodexRunner("codex.exe", "m", agy_popen(raw)).run("x", tmp_path, 1)
    assert result.reply == "PASS done"


def test_make_runner_builds_codex_from_routing():
    """Configured defaults and an explicit model override both reach the Codex runner."""
    config = {"routing": {"agent_run": {"codex": {"binary": "codex.exe", "default_model": "configured"}}}}
    runner = make_runner("codex", config)
    assert isinstance(runner, CodexRunner) and runner.name == "codex" and runner.model == "configured"
    assert make_runner("codex", config, "override").model == "override"
    with pytest.raises(RunnerError, match="agent_run.codex.binary"):
        make_runner("codex", {"routing": {}})
    with pytest.raises(RunnerError, match="agent_run.codex.default_model"):
        make_runner("codex", {"routing": {"agent_run": {"codex": {"binary": "codex.exe"}}}})


def test_codex_eligibility_and_start_failures(tmp_path):
    """Internal/private data never reaches Codex; timeouts and missing binaries follow the runner contract."""
    runner = CodexRunner("codex.exe", "m")
    assert [runner.eligible(level) for level in ("normal", "internal", "private", "unknown")] == [True, False, False, False]
    runner = CodexRunner("codex.exe", "m", agy_popen(raises=subprocess.TimeoutExpired(cmd="codex", timeout=1)))
    result = runner.run("x", tmp_path, 1)
    assert result == RunnerResult(TIMEOUT_REPLY, timed_out=True, exit_code=None)
    missing = CodexRunner("codex.exe", "m", agy_popen(raises=FileNotFoundError("codex")))
    with pytest.raises(RunnerError, match="codex binary not found"):
        missing.run("x", tmp_path, 1)


@pytest.mark.parametrize("raw, expected", [
    ('{"type":"turn.failed","error":{"message":"quota exhausted"}}', "FAIL quota exhausted"),
    ('{"type":"error","message":"connection lost"}', "FAIL connection lost"),
    ('{"type":"item.completed","item":[]}', "FAIL codex printed no usable output"),
    ("not JSON\n[]\n{}", "FAIL codex printed no usable output"),
])
def test_codex_errors_and_malformed_output(tmp_path, raw, expected):
    """Failed turns and malformed JSON produce usable short replies without a parser exception."""
    result = CodexRunner("codex.exe", "m", agy_popen(raw)).run("x", tmp_path, 1)
    assert result.reply.startswith(expected) and result.raw == raw


@pytest.mark.parametrize("sensitivity", ["internal", "private"])
def test_agent_run_refuses_codex_sensitive_data(repo, sensitivity):
    """Refuse sensitive data before starting a worktree or a subprocess."""
    job = AgentRunJob("task", repo, ("a.py",), CHECK, runner="codex", sensitivity=sensitivity)
    with pytest.raises(AgentRunError, match=f"codex may not receive {sensitivity}"):
        agent_run(job, runner=CodexRunner("codex.exe", "m"))


def test_nested_repository_left_by_the_agent_is_reverted_not_fatal(tmp_path, repo):
    """Tests run inside a sandboxed worktree leave `pytest-of-*/.../proj` folders with their own .git: they are staged as
    gitlinks, and `git rm` used to fail on them (the whole run was refused)."""
    def leave_nested_repo(worktree: Path) -> None:
        nested = worktree / "pytest-of-x" / "pytest-0" / "proj"
        nested.mkdir(parents=True)
        (nested / "a.txt").write_text("x\n", encoding="utf-8")
        git(nested, "init", "-q", "-b", "main")
        git(nested, "add", "-A")
        git(nested, *IDENTITY, "commit", "-q", "-m", "n")

    runner = FakeRunner(writes={"tests/test_new.py": "def test_new(): assert True\n"}, on_run=leave_nested_repo)
    report = run_job(tmp_path, runner)
    assert report["passed"] is True and report["files"] == ["tests/test_new.py"]
    assert any("pytest-of-x" in item for item in report["reverted"])
    assert "pytest-of-x" not in git(repo, "ls-tree", "-r", "--name-only", report["branch"])
    assert git(repo, "worktree", "list").count("\n") == 1
