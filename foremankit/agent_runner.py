"""Runners for `agent-run`: a headless coding agent that works in a folder and answers with a short reply.
`Runner` is the interface; `AgyRunner` is the Antigravity CLI. An opencode runner can be added later with the same
three members (name, model, eligible) and `run`.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


class Runner(Protocol):
    """A headless coding agent."""
    name: str
    model: str

    def eligible(self, sensitivity: str) -> bool:
        """Whether this runner may receive data of that sensitivity."""

    def run(self, prompt: str, cwd: Path, timeout: float) -> RunnerResult:
        """Run the agent with `prompt` inside `cwd` and return its reply."""


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
            done = self._runner(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired:
            return RunnerResult(TIMEOUT_REPLY, timed_out=True, exit_code=None)
        except FileNotFoundError as error:
            raise RunnerError(f"agy binary not found: {command[0]}") from error
        return RunnerResult(_reply_of(done.stdout, done.returncode), exit_code=done.returncode)


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
    raise RunnerError(f"runner {name!r} is not available (supported: agy; opencode is planned)")
