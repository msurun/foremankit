"""Read-only views of the router: status, models, statistics and tests of the lab nodes."""
from __future__ import annotations

import platform
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx

from . import budget, netguard, quota
from .config import is_dry_run
from .outcomes import summarize_outcomes
from .nodes import Node, lab_key, load_nodes, self_name
from .providers import (
    ChatRequest,
    Endpoint,
    NodeProbe,
    Provider,
    ProviderError,
    chat_completion,
    list_models,
    pick_model,
    probe_endpoint,
)
from .records import RouterError

if TYPE_CHECKING:
    from .router import Router

SECONDS_PER_DAY = 86400
DEFAULT_REFERENCE_PRICE = {"input": 3.0, "output": 15.0}   # USD per million tokens (Sonnet), for "what it would cost"
TEST_PROMPT = "Reply with the single word: OK"
TEST_MAX_TOKENS = 8
TEST_MIN_TIMEOUT_S = 30.0
TEST_REPLY_CHARS = 40
TEST_ERROR_CHARS = 120
STATS_COUNTERS = ("tokens_in", "tokens_out", "cost_usd", "truncated", "continuations")


class Reporter:
    """Builds the dictionaries behind `foremankit status / models / stats / llm / nodes`."""

    def __init__(self, router: Router):
        self.router = router

    # ------------------------------------------------------------ nodes
    def node_status(self, node: Node, *, force: bool = True, detail: bool = False) -> dict[str, Any]:
        """One node of the pool: switch, address, reachability, model. detail=True also asks whether it is busy right
        now and whether private data may travel to it (netguard)."""
        state = self.router.store.load()
        toggle, source = self.router.selector.is_enabled(node, state)
        info = self._node_basics(node, state, toggle, source)
        if toggle == "off":
            info.update(reachable=False, endpoint=None, models=[], note="off")
            return info
        probe = self.router.prober.node(node, force=force)
        info.update(self._probe_fields(node, probe))
        if detail and probe.endpoint is not None:
            allowed, reason = netguard.check_host(probe.endpoint.url, self.router.resolver)
            info.update(busy=self.router.prober.is_busy(probe), private_ok=allowed, netguard=reason)
        return info

    def _node_basics(self, node: Node, state: dict[str, Any], toggle: str, source: str) -> dict[str, Any]:
        rule = (f"priority loan: only when idle, tasks <= {node.loan_max_s:.0f} s" if node.idle_only
                else "unrestricted")
        return {"name": node.name, "self": node.is_self, "enabled": toggle, "source": source, "accept": node.accept,
                "priority": node.priority, "speed": self.router.selector.node_speed(node, state, self.router.clock()),
                "speed_written": node.speed, "speed_pinned": not node.learn,
                "speed_learned": (state.get("node_speed") or {}).get(node.name),
                "max_concurrent": node.max_concurrent, "endpoints": [endpoint.name for endpoint in node.endpoints],
                "rule": rule}

    def _probe_fields(self, node: Node, probe: NodeProbe) -> dict[str, Any]:
        llm_cfg = self.router.providers["llm"].cfg
        endpoint = probe.endpoint
        latency_ms = round(probe.latency_s * 1000) if probe.latency_s is not None else None
        return {"reachable": endpoint is not None,
                "endpoint": f"{endpoint.name} {endpoint.url}" if endpoint else None,
                "latency_ms": latency_ms, "models": probe.models, "errors": probe.errors,
                "chosen_model": pick_model(node.preferred_models or llm_cfg.get("preferred_models"),
                                           llm_cfg.get("exclude_models"), probe.models)}

    def nodes_status(self) -> dict[str, Any]:
        """This machine and every node of config/nodes.yaml: reachable? busy? which data levels? Used by `foremankit nodes`."""
        nodes_cfg = self.router.cfg.get("nodes") if self.router.cfg.get("nodes") is not None else load_nodes()
        nodes_cfg = nodes_cfg or {}
        nodes = self.router.nodes()
        with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
            rows = list(pool.map(lambda node: self.node_status(node, detail=True), nodes))
        return {"hostname": platform.node(), "self": self_name(nodes_cfg), "key_set": bool(lab_key(nodes_cfg)),
                "tunnel_enabled": bool((nodes_cfg.get("settings") or {}).get("tunnel_enabled")), "nodes": rows}

    def node_test(self, selector: str | None = None, timeout: float = 5.0) -> list[dict[str, Any]]:
        """Probe every endpoint of the chosen node(s) and run a tiny generation (local hardware only)."""
        chosen = self._chosen_nodes(selector)
        if not chosen:
            raise RouterError(f"no such node: {selector!r}")
        with self.router.http_client() as client:
            return [self._test_endpoint(client, node, endpoint, timeout)
                    for node in chosen for endpoint in node.endpoints]

    def _chosen_nodes(self, selector: str | None) -> list[Node]:
        group = {"self": True, "others": False}.get(selector or "")
        return [node for node in self.router.nodes()
                if selector in (None, "all") or node.name == selector or (group is not None and node.is_self == group)]

    def _test_endpoint(self, client: httpx.Client, node: Node, endpoint: Endpoint, timeout: float) -> dict[str, Any]:
        allowed, reason = netguard.check_host(endpoint.url, self.router.resolver)
        row: dict[str, Any] = {"node": node.name, "endpoint": endpoint.name, "url": endpoint.url,
                               "private_ok": allowed, "netguard": reason}
        models, latency, error = probe_endpoint(client, endpoint, timeout)
        row.update(reachable=models is not None, latency_ms=round(latency * 1000), models=models or [],
                   error=error or None)
        llm_cfg = self.router.providers["llm"].cfg
        model = pick_model(node.preferred_models or llm_cfg.get("preferred_models"),
                           llm_cfg.get("exclude_models"), models or [])
        if models is not None and model and not is_dry_run():
            row["generation"] = self._tiny_generation(client, endpoint, model, timeout)
        return row

    @staticmethod
    def _tiny_generation(client: httpx.Client, endpoint: Endpoint, model: str, timeout: float) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            reply = chat_completion(client, ChatRequest(
                endpoint.url + "/v1", endpoint.api_key, model, [{"role": "user", "content": TEST_PROMPT}],
                TEST_MAX_TOKENS, max(TEST_MIN_TIMEOUT_S, timeout)))
        except ProviderError as error:
            return {"model": model, "ok": False, "error": str(error)[:TEST_ERROR_CHARS]}
        return {"model": model, "ok": True, "reply": reply["text"][:TEST_REPLY_CHARS],
                "seconds": round(time.perf_counter() - start, 2)}

    # ------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        """Budget, cloud providers and lab nodes at a glance."""
        router = self.router
        now, state = router.clock(), router.store.load()
        budget_cfg = router.routing.get("budget") or {}
        day, month = budget.spent(state, now)
        providers = [self._provider_row(provider, state, now)
                     for provider in router.providers.values() if not provider.is_llm]
        return {"dry_run": is_dry_run(), "config_dir": router.cfg.get("dir"), "profile": state.get("profile"),
                "budget": {"day_usd": day, "month_usd": month, "daily_cap": budget_cfg.get("daily_usd"),
                           "monthly_cap": budget_cfg.get("monthly_usd"),
                           "categories": self._budget_categories(state, budget_cfg, now)},
                "providers": providers, "llm": [self.node_status(node) for node in router.nodes()]}

    @staticmethod
    def _budget_categories(state: dict[str, Any], budget_cfg: dict[str, Any], now: float) -> dict[str, dict[str, Any]]:
        """Spend and caps per category (delegation, orchestrator)."""
        rows = {}
        for category in (budget.DELEGATION, budget.ORCHESTRATOR):
            day, month = budget.spent(state, now, category)
            daily_cap, monthly_cap = budget.caps(budget_cfg, category)
            rows[category] = {"day_usd": day, "month_usd": month, "daily_cap": daily_cap, "monthly_cap": monthly_cap}
        return rows

    def _provider_row(self, provider: Provider, state: dict[str, Any], now: float) -> dict[str, Any]:
        if provider.cfg.get("enabled", True) is False:
            reason = "disabled"
        else:
            reason = "no key" if not provider.api_key() else self.router.cloud_enabled(provider)
        return {"name": provider.name, "active": not reason, "reason": reason, "free": provider.free,
                "trains_on_data": provider.trains_on_data, "internal_ok": provider.eligible("internal"),
                "remaining": quota.remaining(state, provider.name, provider.limits, now),
                "cooldown_s": round(quota.cooldown_left(state, provider.name, now))}

    # ------------------------------------------------------------ models
    def models(self, only: str | None = None) -> dict[str, Any]:
        """Models each source offers now, next to the ones configured (which ones are missing)."""
        report: dict[str, Any] = {}
        with self.router.http_client() as client:
            for provider in self.router.providers.values():
                if only and only.partition("@")[0] != provider.name:
                    continue
                if provider.is_llm:
                    report.update(self._lab_node_models(provider))
                else:
                    report[provider.name] = self._cloud_models(client, provider)
        return report

    def _lab_node_models(self, provider: Provider) -> dict[str, Any]:
        rows = {}
        for node in self.router.nodes():
            info = self.node_status(node)
            rows[f"llm@{node.name}"] = {
                "available": info.get("models", []),
                "configured": node.preferred_models or provider.cfg.get("preferred_models", []),
                "error": None if info.get("reachable") else "unreachable/off"}
        return rows

    @staticmethod
    def _cloud_models(client: httpx.Client, provider: Provider) -> dict[str, Any]:
        key = provider.api_key()
        configured = [model["id"] for model in (provider.cfg.get("models") or {}).values()]
        if not key or not provider.cfg.get("enabled", True):
            return {"available": [], "configured": configured, "error": "inactive (no key or disabled)"}
        try:
            available = list_models(client, provider.cfg["base_url"], key)
        except (httpx.HTTPError, ValueError) as error:
            return {"available": [], "configured": configured, "error": error.__class__.__name__}
        missing = [model for model in configured if model not in available
                   and model.removeprefix("models/") not in available and f"models/{model}" not in available]
        return {"available": available, "configured": configured, "error": None, "missing": missing}

    # ------------------------------------------------------------ statistics
    def stats(self, days: int | None = None) -> dict[str, Any]:
        """Calls, tokens and cost per provider and per kind, and what the same tokens would cost on the reference model."""
        reference = (self.router.routing.get("reference") or {})
        price = reference.get("price") or DEFAULT_REFERENCE_PRICE
        cutoff = time.time() - days * SECONDS_PER_DAY if days else 0
        by_provider: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        by_kind: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_cost = reference_cost = 0.0
        for row in self.router.log.read():
            if row.get("event") != "call" or row.get("dry_run") or row.get("ts", 0) < cutoff:
                continue
            self._count_row(row, by_provider[row.get("provider", "?")], by_kind[row.get("kind", "?")])
            total_cost += row.get("cost_usd", 0.0)   # rejected (cut-off, empty) calls were paid for too
            if row.get("success"):
                reference_cost += (row.get("tokens_in", 0) * price["input"]
                                   + row.get("tokens_out", 0) * price["output"]) / 1e6
        return {"by_provider": {name: dict(counts) for name, counts in by_provider.items()},
                "by_kind": {name: dict(counts) for name, counts in by_kind.items()},
                "outcomes": summarize_outcomes(self.router.log.read(), cutoff),
                "cost_usd": total_cost, "reference_usd": reference_cost, "saved_usd": reference_cost - total_cost,
                "reference": reference.get("name", "sonnet")}

    @staticmethod
    def _count_row(row: dict[str, Any], *buckets: dict[str, float]) -> None:
        for bucket in buckets:
            bucket["calls"] += 1
            bucket["ok"] += 1 if row.get("success") else 0
            for counter in STATS_COUNTERS:
                bucket[counter] += row.get(counter, 0)
