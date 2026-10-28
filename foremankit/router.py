"""Routing core (facade): wires the collaborators together and offers the public API used by the CLI and the MCP server.

The work itself lives in small modules: chain (walk the sources), candidates (describe one source), caller (send one
request), postcheck (verify + review), probing, selection, reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .caller import Caller
from .candidates import CandidateBuilder
from .chain import ChainRun
from .config import check_config, load_config, load_env
from .nodes import Node, build_nodes
from .outputs import resolve_out_dir
from .outcomes import build_outcome
from .postcheck import ReviewStage, make_verify_spec, verify_with_handoffs
from .probing import Prober
from .providers import NodeProbe, Provider, normalize_toggle
from .records import ReviewJob, RouterEnv, RouterError, TaskRequest, TaskTooLarge
from .reports import Reporter
from .selection import ConcurrencySlots, NodeSelector
from .state import CallLog, StateStore

__all__ = ["Router", "RouterEnv", "RouterError", "TaskTooLarge"]


class Router:
    """Picks the first usable source for a task, calls it, stores the answer, and can check and review the code."""

    def __init__(self, cfg: dict[str, Any] | None = None, env: RouterEnv | None = None):
        env = env or RouterEnv()
        if cfg is None:
            load_env()
            cfg = load_config()
        self.cfg = check_config(cfg)              # also for configs built by hand: removed provider names are refused
        self.providers = {name: Provider(name, section or {}) for name, section in (cfg.get("providers") or {}).items()}
        self.routing = cfg.get("routing") or {}
        self.profiles = cfg.get("profiles") or {}
        self.store = env.store or StateStore()
        self.log = env.log or CallLog()
        self.transport, self.resolver, self.clock, self.sleep = env.transport, env.resolver, env.clock, env.sleep
        self._out_dir = env.out_dir
        self.warnings: list[str] = []
        self._create_collaborators()

    def _create_collaborators(self) -> None:
        """The helpers that do the work; each keeps a reference to this router for the shared settings and state."""
        self.prober = Prober(self)
        self.selector = NodeSelector(self)
        self.slots = ConcurrencySlots()
        self.candidate_builder = CandidateBuilder(self)
        self.caller = Caller(self)
        self.reviewer = ReviewStage(self)
        self.reporter = Reporter(self)

    # ------------------------------------------------------------ small helpers used by the collaborators
    def out_dir(self) -> Path:
        """Where answers are written."""
        return resolve_out_dir(self._out_dir)

    def http_client(self) -> httpx.Client:
        """A new HTTP client (on the test transport when one was given)."""
        return httpx.Client(transport=self.transport) if self.transport else httpx.Client()

    def warn(self, message: str) -> None:
        """Keep a warning for the caller and write it to the call log."""
        self.warnings.append(message)
        self.log.write({"event": "warning", "message": message})

    def kinds(self) -> list[str]:
        """The kinds of task that have a chain in routing.yaml."""
        return list((self.routing.get("chains") or {}).keys())

    def nodes(self) -> list[Node]:
        """Every node of the pool: this machine first (when it serves), then the others. See config/nodes.yaml."""
        provider = self.providers.get("llm")
        return build_nodes(self.cfg.get("nodes"), provider.cfg) if provider else []

    def node_probe(self, node: Node, *, private: bool = False, force: bool = False) -> NodeProbe:
        """Reachability of one node (cached for a short time unless `force`)."""
        return self.prober.node(node, private=private, force=force)

    def cloud_enabled(self, provider: Provider) -> str:
        """'' when usable, else the skip reason. enabled: true | false | auto (probe)."""
        enabled = provider.cfg.get("enabled", True)
        if enabled == "auto":
            reachable, reason = self.prober.cloud(provider)
            return "" if reachable else f"unreachable ({reason})"
        return "" if enabled else "disabled in config"

    # ------------------------------------------------------------ delegation
    def delegate(self, task: str, kind: str = "quick", sensitivity: str = "normal", **options: Any) -> dict[str, Any]:
        """Run `task` on the first usable source of the kind's chain.
        options (see TaskRequest): context_files, system_hint, max_output_tokens, escalate, allow_cost, verify,
        verify_target, review.
        allow_cost: accept a paid call whose worst-case cost is above routing.yaml budget.per_call_usd.
        verify + verify_target: write the answer into that file, run the check command (routing.yaml
        verify.allowed_commands) and give failures back to the same model; the file stays changed only when it passes.
        review: after the checks, a DIFFERENT model (own local LLM or a paid one, routing.yaml chains.review) reviews
        the code; grounded blocker findings go back to the author once."""
        request = TaskRequest(task, kind, sensitivity, **options)
        spec = make_verify_spec(self.routing, request)   # refuse a bad request before any model is paid
        result = self.run_chain(request)
        if spec:
            result = verify_with_handoffs(self, result, request, spec)
        if request.review:
            self.reviewer.run(result, request, spec)
        if not result.get("dry_run"):
            self.log.write(build_outcome(result, request.kind))
        return {key: value for key, value in result.items() if not key.startswith("_")}

    def run_chain(self, request: TaskRequest, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
        """The result of the first source of the chain that answers. `exclude` = sources that must not serve it."""
        return ChainRun(self, request, exclude).execute()

    def review_code(self, job: ReviewJob, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
        """Findings of a reviewer model on `job.code` (see ReviewStage.review_code)."""
        return self.reviewer.review_code(job, exclude)

    # ------------------------------------------------------------ toggles / profiles
    def set_toggle(self, selector: str, value: str) -> None:
        """Switch lab nodes: `selector` is a node name, `self` (this machine), `others` or `all`."""
        valid = {"all", "self", "others"} | {node.name for node in self.nodes()}
        if selector not in valid:
            raise RouterError(f"unknown node {selector!r}; use one of {sorted(valid)}")
        value = normalize_toggle(value)
        self.store.update(lambda state: state.setdefault("toggles", {}).__setitem__(f"llm:{selector}", value))
        self.prober.forget_node()

    def apply_profile(self, name: str) -> dict[str, str]:
        """Apply a named set of node switches from profiles.yaml; returns the switches that were set."""
        if name not in self.profiles:
            raise RouterError(f"unknown profile {name!r}; available: {sorted(self.profiles)}")
        flags = {key: normalize_toggle(value) for key, value in (self.profiles[name] or {}).items()}

        def apply(state: dict[str, Any]) -> None:
            """Write the profile's switches and its name into the state file."""
            state.setdefault("toggles", {}).update(flags)
            state["profile"] = name
        self.store.update(apply)
        self.prober.forget_node()
        return flags

    # ------------------------------------------------------------ views (see reports.py)
    def node_status(self, node: Node, *, force: bool = True, detail: bool = False) -> dict[str, Any]:
        """One node of the pool: switch, address, reachability, model."""
        return self.reporter.node_status(node, force=force, detail=detail)

    def status(self) -> dict[str, Any]:
        """Budget, cloud providers and lab nodes at a glance."""
        return self.reporter.status()

    def models(self, only: str | None = None) -> dict[str, Any]:
        """Models each source offers now, next to the configured ones."""
        return self.reporter.models(only)

    def stats(self, days: int | None = None) -> dict[str, Any]:
        """Calls, tokens and cost per provider and kind."""
        return self.reporter.stats(days)

    def node_test(self, selector: str | None = None, timeout: float = 5.0) -> list[dict[str, Any]]:
        """Probe every endpoint of the chosen node(s) and run a tiny generation."""
        return self.reporter.node_test(selector, timeout)

    def nodes_status(self) -> dict[str, Any]:
        """This machine and every node of config/nodes.yaml (used by `foremankit nodes`)."""
        return self.reporter.nodes_status()
