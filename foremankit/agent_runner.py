"""Runners for `agent-run`: a headless coding agent that works in a folder and answers with a short reply.
`Runner` is the shared interface for Antigravity, opencode and Codex CLI runs.
Each runner keeps untouched stdout for usage and full-answer parsing.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import agent_codex
from .providers import Provider
from .supervise import AGY_FLAGS, MODEL_ALIAS

TIMEOUT_REPLY = "FAIL the runner timed out"


class RunnerError(RuntimeError):
    """The runner cannot start at all (binary missing, runner not supported yet)."""


@dataclass(frozen=True)
class RunnerResult:
    """What the agent said and whether it ended normally. The files it changed are read from git, not from here."""
    reply: str
    timed_out: bool = False
    exit_code: int | None = 0
    raw: str = ""            # the runner's untouched stdout (JSON): `agent-ask` reads tokens and tool calls from it


class Runner(Protocol):
    """A headless coding agent."""
    name: str
    model: str

    def eligible(self, sensitivity: str) -> bool:
        """Whether this runner may receive data of that sensitivity."""

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Run the agent with `prompt` inside `cwd` and return its reply."""


def _env_for(cwd: Path) -> dict[str, str]:
    """The environment of a runner: the parent's, but with PWD = the run folder (opencode takes its working folder from
    PWD, not from the process cwd, and answered about the wrong repository when PWD was inherited)."""
    return {**os.environ, "PWD": str(cwd)}


class AgyRunner:
    """`agy -p <prompt> --model <id> --output-format json --sandbox --dangerously-skip-permissions` in the worktree."""
    name = "agy"

    def __init__(self, provider: Provider, model_alias: str = MODEL_ALIAS, runner=subprocess.run):
        self.provider = provider
        self.model = provider.model(model_alias)[1]
        self._runner = runner

    def eligible(self, sensitivity: str) -> bool:
        """Same rule as `supervise`: the provider's own sensitivity rule (private never)."""
        return sensitivity != "private" and self.provider.eligible(sensitivity)

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Start agy; a timeout is a result (the work done so far is still checked), a missing binary is an error."""
        command = [self.provider.agy_binary, "-p", prompt, "--model", self.model, *AGY_FLAGS]
        try:
            done = self._runner(command, stdin=subprocess.DEVNULL, env=_env_for(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired:
            return RunnerResult(TIMEOUT_REPLY, timed_out=True, exit_code=None)
        except FileNotFoundError as error:
            raise RunnerError(f"agy binary not found: {command[0]}") from error
        return RunnerResult(_reply_of(done.stdout, done.returncode), exit_code=done.returncode, raw=done.stdout or "")


class OpenCodeRunner:
    """`opencode run --auto --model <provider/model> --format json <prompt>` in the worktree (opencode 2.x).
    Its free models train on prompts, so only `normal` (synthetic or public) data is eligible."""
    name = "opencode"

    def __init__(self, binary: str, model: str, runner=subprocess.run):
        self.binary = os.path.expanduser(binary)
        self.model = model
        self._runner = runner

    def eligible(self, sensitivity: str) -> bool:
        """Only `normal` data: free opencode models may train on what they receive."""
        return sensitivity == "normal"

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Start opencode; same contract as AgyRunner.run."""
        command = [self.binary, "run", "--auto", "--model", self.model, "--format", "json", prompt]
        try:
            done = self._runner(command, stdin=subprocess.DEVNULL, env=_env_for(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired:
            return RunnerResult(TIMEOUT_REPLY, timed_out=True, exit_code=None)
        except FileNotFoundError as error:
            raise RunnerError(f"opencode binary not found: {self.binary}") from error
        return RunnerResult(_opencode_reply(done.stdout, done.returncode), exit_code=done.returncode, raw=done.stdout or "")


class CodexRunner:
    """Non-interactive Codex CLI on the owner's subscription; only normal data is eligible."""
    name = "codex"

    def __init__(self, binary: str, model: str, runner=subprocess.run):
        """Use the configured executable and model, with an injectable subprocess runner."""
        self.binary, self.model = os.path.expanduser(binary), model
        self._runner = runner

    def eligible(self, sensitivity: str) -> bool:
        """Only public or synthetic data until the owner permits another sensitivity."""
        return sensitivity == "normal"

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Run with closed stdin and bounded workspace writes; preserve the raw JSONL."""
        command = self._command(prompt, cwd)
        env = _env_for(cwd)
        env.pop("CODEX_PERMISSION_PROFILE", None)  # the parent Codex app may grant unrestricted access
        try:
            done = self._runner(command, stdin=subprocess.DEVNULL, env=env, cwd=cwd, timeout=timeout,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return RunnerResult(TIMEOUT_REPLY, timed_out=True, exit_code=None)
        except FileNotFoundError as error:
            raise RunnerError(f"codex binary not found: {self.binary}") from error
        return RunnerResult(_codex_reply(done.stdout, done.returncode), exit_code=done.returncode, raw=done.stdout or "")

    def _command(self, prompt: str, cwd: Path) -> list[str]:
        """Ignore personal config so its writable roots and tools cannot widen the run's sandbox."""
        settings = ["sandbox_workspace_write.exclude_tmpdir_env_var=true",
                    "sandbox_workspace_write.exclude_slash_tmp=true"]
        if os.name == "nt":
            settings.append('windows.sandbox="elevated"')
        overrides = [argument for setting in settings for argument in ("-c", setting)]
        return [self.binary, "--ask-for-approval", "never", "exec", "--ignore-user-config", *overrides,
                "--sandbox", "workspace-write", "--json", "--cd", str(cwd), "--model", self.model, "--", prompt]


def _codex_reply(stdout: str, exit_code: int) -> str:
    """The last full message, or a failure; the public reply stays on one line."""
    events = agent_codex.events(stdout)
    error = agent_codex.failure(events)
    answer = agent_codex.answer(events)
    reply = (answer if exit_code == 0 else error) or error or answer
    reply = reply or f"FAIL codex printed no usable output (exit {exit_code})"
    return " ".join(reply.split())


def _opencode_reply(stdout: str, exit_code: int) -> str:
    """The last `text` event of opencode's JSON-lines output; the error event's message or a short note otherwise."""
    texts, errors = [], []
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "text":
            texts.append(str((event.get("part") or {}).get("text") or ""))
        elif isinstance(event, dict) and event.get("type") == "error":
            errors.append(str(event.get("error") or event.get("message") or event))
    reply = (texts or errors or [f"FAIL opencode printed no usable output (exit {exit_code})"])[-1]
    return reply.strip().replace("\n", " ")


def _reply_of(stdout: str, exit_code: int) -> str:
    """agy's reply text from its JSON output; a short note when it printed something else."""
    try:
        data = json.loads(stdout)
    except (TypeError, ValueError):
        return f"FAIL agy printed no JSON (exit {exit_code})"
    return str(data.get("response") or data.get("error") or "").strip().replace("\n", " ")


def make_runner(name: str, config: dict[str, Any], model_alias: str = "") -> Runner:
    """The runner called `name`; RunnerError for a runner that is not available yet."""
    if name == "agy":
        return AgyRunner(Provider("agy", config["providers"]["agy"]), model_alias or MODEL_ALIAS)
    if name in ("opencode", "codex"):
        settings = config["routing"].get("agent_run", {}).get(name) or {}
        if not settings.get("binary"):
            raise RunnerError(f"routing.yaml agent_run.{name}.binary is not set")
        if not (model_alias or settings.get("default_model")):
            raise RunnerError(f"routing.yaml agent_run.{name}.default_model is not set")
        runner_type = CodexRunner if name == "codex" else OpenCodeRunner
        return runner_type(settings["binary"], model_alias or settings["default_model"])
    raise RunnerError(f"runner {name!r} is not available (supported: agy, opencode, codex)")
