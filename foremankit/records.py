"""Small data classes passed between the steps of a delegation (request, call context, candidate, answer)."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import netguard
from .providers import Provider
from .state import CallLog, StateStore

SENSITIVITIES = ("normal", "internal", "private")
MAX_OUTPUT_CAP = 16000   # tokens: hard upper limit for one answer
MIN_OUTPUT_TOKENS = 64
DEFAULT_MAX_OUTPUT_TOKENS = 4000    # keeps one answer short enough to review; bigger jobs are split into several calls


class RouterError(RuntimeError):
    """The router cannot do what was asked (bad request, nothing available, missing file)."""


class TaskTooLarge(RouterError):
    """A paid model ran out of output tokens (cut off in the middle of code, or only reasoned) after more than
    calls.abort_waste_usd was spent. Retrying on another paid model would pay twice for the same mistake, so the call
    stops here: split the task or lower the output size. (Cheaper failures just move on to the next source.)"""


@dataclass
class RouterEnv:
    """The outside world of a Router: where state and logs live, the network and the clock (tests replace them)."""
    store: StateStore | None = None
    log: CallLog | None = None
    transport: httpx.BaseTransport | None = None
    resolver: netguard.Resolver | None = None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    out_dir: Path | None = None


@dataclass
class TaskRequest:
    """What the caller asked for (the arguments of `Router.delegate`)."""
    task: str
    kind: str = "quick"
    sensitivity: str = "normal"
    context_files: list[str] | None = None
    system_hint: str = ""
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    escalate: bool = False
    allow_cost: bool = False
    verify: str = ""
    verify_target: str = ""
    review: bool = False
    continue_from: str = ""       # id of a cut-off answer in .agent-out to finish (same task, same model)
    max_continuations: int = 0    # extra continuation rounds for this call (0 = routing.yaml calls.max_continuations)


@dataclass(frozen=True)
class VerifySpec:
    """A check command and the file it checks (already validated against routing.yaml verify.allowed_commands)."""
    command: str
    target: str
    tokens: list[str]
    path: Path


@dataclass(frozen=True)
class ReviewJob:
    """Code to be reviewed by another model."""
    code: str
    target: str
    task: str
    check_summary: str = ""
    sensitivity: str = "normal"
    allow_cost: bool = False


@dataclass(frozen=True)
class CallContext:
    """Everything one call to a model needs to know about the task it belongs to."""
    call_id: str
    kind: str
    sensitivity: str
    escalate: bool
    est_in: int                    # estimated input tokens
    max_out: int                   # requested answer size, tokens
    messages: list[dict[str, str]] = field(default_factory=list)
    reasoning: str | None = None
    allow_cost: bool = False
    prompt_sha: str = ""           # fingerprint of the whole prompt, so a resume can prove it is the same task
    max_continuations: int = 0     # per-call override of calls.max_continuations (0 = use the configuration)


@dataclass
class Candidate:
    """A source (cloud model or lab node) that can serve the call, with everything needed to send it."""
    provider: Provider
    alias: str
    model_id: str
    price: dict[str, Any]
    base_url: str
    api_key: str | None
    endpoint: str = ""
    node: str = ""       # lab node serving this call (provider "llm" only)
    send_max: int = 0  # max_tokens actually sent (requested + thinking headroom)
    extra: dict[str, Any] | None = None  # extra request fields (e.g. reasoning_effort)
    speed: dict[str, Any] | None = None  # measured speed of the serving node (tok/s)
    loan_max_s: float | None = None      # hard cap (also on the last "force" pass) for a loan from a higher-priority node
    kinds: dict[str, Any] | None = None  # per-node input limits per kind, else routing.yaml llm.kinds
    limit: Any = None                    # max concurrent requests to this source (None = unlimited)
    learn: bool = True                   # whether this call's measured speed feeds the node's moving average

    @property
    def key(self) -> str:
        """Name under which quota, cooldown and statistics are kept: a node of the pool counts separately."""
        return f"{self.provider.name}@{self.node}" if self.node else self.provider.name


@dataclass
class Answer:
    """What a model returned, with the cost and time of getting it."""
    text: str
    finish_reason: str | None
    usage: dict[str, int]
    cost_usd: float = 0.0
    latency_s: float = 0.0
    timings: dict[str, Any] = field(default_factory=dict)   # llama-server's exact speed report, when present
    continuations: int = 0   # how many extra requests finished an answer that stopped at its token limit
