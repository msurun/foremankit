"""Which lab node should serve a task: switches, data levels, expected finishing time, suitability, concurrency."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from . import speed as speed_module
from .nodes import Node
from .providers import NodeProbe, normalize_toggle
from .records import CallContext

if TYPE_CHECKING:
    from .router import Router

DEFAULT_PROMPT_TPS = 100.0         # tokens read per second, when a node has no measured speed
DEFAULT_GEN_TPS = 10.0             # tokens written per second, when a node has no measured speed
DEFAULT_QUEUE_PENALTY_S = 30.0     # added to the estimate of a node that is generating for someone else
UNLIMITED_INPUT_TOKENS = 10 ** 9
UNLIMITED_SECONDS = 1e9


def estimate_seconds(speed: dict[str, Any] | None, est_in: int, max_out: int) -> float:
    """Wall-clock estimate: reading the prompt plus writing the answer, at the node's speed (tokens/s)."""
    speed = speed or {}
    written = min(max_out, int(speed.get("expected_output_cap", max_out)))
    return est_in / float(speed.get("prompt_tps", DEFAULT_PROMPT_TPS)) + written / float(speed.get("gen_tps", DEFAULT_GEN_TPS))


def unsuited_reason(policy: dict[str, Any] | None, ctx: CallContext, speed: dict[str, Any]) -> str | None:
    """Why this task should NOT go to this (slow) node (None = fine): the allowed kinds and input sizes first
    (routing.yaml `llm.kinds` or the node's own `kinds`), then a time estimate against the node's `max_seconds`."""
    problem = _kind_policy_problem(policy, ctx)
    if problem:
        return problem
    est_s = estimate_seconds(speed, ctx.est_in, ctx.max_out)
    if est_s > float(speed.get("max_seconds", UNLIMITED_SECONDS)):
        return f"too long for this node (~{est_s:.0f} s > {speed['max_seconds']} s)"
    return None


def _kind_policy_problem(policy: dict[str, Any] | None, ctx: CallContext) -> str | None:
    if policy is None:
        return None
    rule = policy.get(ctx.kind)
    if rule is None:
        return f"kind '{ctx.kind}' is not set up for local models (routing.yaml llm.kinds)"
    limit = int(rule.get("max_input_tokens", UNLIMITED_INPUT_TOKENS))
    if ctx.est_in > limit:
        return f"input too big for this node (~{ctx.est_in} > {limit} tokens)"
    return None


class NodeSelector:
    """Decides which nodes of the pool may serve a task and in which order."""

    def __init__(self, router: Router):
        self.router = router

    def is_enabled(self, node: Node, state: dict[str, Any]) -> tuple[str, str]:
        """Effective toggle and its source. Priority: CLI state > env > config; within a source the most specific wins:
        the node itself, then `self`/`others`, then `all`."""
        group = "self" if node.is_self else "others"
        toggles = state.get("toggles", {})
        for selector in (node.name, group, "all"):
            if f"llm:{selector}" in toggles:
                return normalize_toggle(toggles[f"llm:{selector}"]), "cli"
        for variable in (f"FOREMANKIT_NODE_{node.name.upper()}_ENABLED", f"LLM_{group.upper()}_ENABLED", "LLM_ENABLED"):
            if os.environ.get(variable):
                return normalize_toggle(os.environ[variable]), "env"
        return normalize_toggle(self.router.providers["llm"].cfg.get("enabled", "auto")), "config"

    def node_speed(self, node: Node, state: dict[str, Any], now: float) -> dict[str, Any] | None:
        """The speed to plan with for `node`: its written speed moved towards what it delivered lately (speed.py)."""
        learning = speed_module.settings(self.router.cfg.get("nodes"))
        return speed_module.effective(speed_module.DeclaredSpeed(node.speed, node.name, node.learn),
                                      state, learning, now)

    def llm_entries(self, ctx: CallContext, state: dict[str, Any], skipped: list[str]) -> list[str]:
        """The chain entry `llm` becomes `llm@<node>` for every node that is on, accepts this data level and answers,
        the one that should finish first coming first: estimated time from the node's measured speed (a busy
        server counts `llm.queue_penalty_s` more); equal estimates go to the higher-priority machine."""
        nodes = self.router.nodes()
        if not nodes:
            skipped.append("llm(no node configured)")
        usable = self._usable_nodes(nodes, ctx.sensitivity, state, skipped)
        probes = self.router.prober.nodes(usable, private=ctx.sensitivity == "private")
        reachable = self._reachable(usable, probes, skipped)
        ranked = self._rank(reachable, ctx, state)
        return [f"llm@{name}" for _seconds, _priority, name in ranked]

    def _usable_nodes(self, nodes: list[Node], sensitivity: str, state: dict[str, Any], skipped: list[str]) -> list[Node]:
        usable = []
        for node in nodes:
            toggle, source = self.is_enabled(node, state)
            if toggle == "off":
                skipped.append(f"{node.entry}(disabled ({source}))")
            elif not node.accepts(sensitivity):
                skipped.append(f"{node.entry}(accepts up to {node.accept})")
            else:
                usable.append(node)
        return usable

    @staticmethod
    def _reachable(usable: list[Node], probes: list[NodeProbe], skipped: list[str]) -> list[tuple[Node, NodeProbe]]:
        reachable = []
        for node, probe in zip(usable, probes):
            if probe.endpoint is None:
                skipped.append(f"{node.entry}(unreachable: {'; '.join(probe.errors) or 'no endpoint'})")
            else:
                reachable.append((node, probe))
        return reachable

    def _busy_flags(self, reachable: list[tuple[Node, NodeProbe]]) -> list[bool | None]:
        probes = [probe for _node, probe in reachable]
        if len(probes) <= 1:
            return [self.router.prober.is_busy(probe) for probe in probes]
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            return list(pool.map(self.router.prober.is_busy, probes))

    def _rank(self, reachable: list[tuple[Node, NodeProbe]], ctx: CallContext,
              state: dict[str, Any]) -> list[tuple[float, int, str]]:
        penalty = float((self.router.routing.get("llm") or {}).get("queue_penalty_s", DEFAULT_QUEUE_PENALTY_S))
        now = self.router.clock()
        ranked = []
        for (node, _probe), busy in zip(reachable, self._busy_flags(reachable)):
            seconds = estimate_seconds(self.node_speed(node, state, now), ctx.est_in, ctx.max_out)
            ranked.append((seconds + (penalty if busy else 0.0), node.priority, node.name))
        return sorted(ranked)


class ConcurrencySlots:
    """Per-source limit on simultaneous requests (a node's or provider's `max_concurrent`)."""

    def __init__(self):
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.Lock()

    def get(self, key: str, limit: Any) -> threading.BoundedSemaphore | None:
        """The semaphore guarding `key`, or None when the source is unlimited."""
        if not limit:
            return None
        with self._lock:
            if key not in self._slots:
                self._slots[key] = threading.BoundedSemaphore(int(limit))
            return self._slots[key]
