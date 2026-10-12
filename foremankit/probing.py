"""Reachability probes: lab nodes (local LLM servers) and cloud providers, both cached for a short time."""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import TYPE_CHECKING

import httpx

from . import netguard
from .config import is_dry_run
from .nodes import Node
from .providers import Endpoint, NodeProbe, Provider, endpoint_busy, probe_endpoint

if TYPE_CHECKING:
    from .router import Router

NODE_PROBE_CACHE_S = 30.0
BUSY_CHECK_TIMEOUT_S = 1.0
DEFAULT_NODE_PROBE_TIMEOUT_S = 1.0
DEFAULT_CLOUD_TTL_S = 600.0         # a success is remembered this long
DEFAULT_CLOUD_NEG_TTL_S = 60.0      # a failure much shorter: one dropped connection must not hide the provider
DEFAULT_CLOUD_TIMEOUT_S = 4.0
CLOUD_RETRY_PAUSE_S = 0.5
HTTP_OK = 200                       # the only status that means "the endpoint answers"; anything else counts as down


class Prober:
    """Asks lab nodes and cloud providers whether they answer, and remembers the answer for a while."""

    def __init__(self, router: Router):
        self.router = router
        self._node_cache: dict[tuple[str, bool], tuple[float, NodeProbe]] = {}
        self._cloud_cache: dict[str, tuple[float, bool, str]] = {}

    # ------------------------------------------------------------ lab nodes
    def node(self, node: Node, *, private: bool = False, force: bool = False,
             timeout: float | None = None) -> NodeProbe:
        """Probe one node (all its addresses at once); the answer is cached unless `force`."""
        key = (node.name, private)
        now = time.monotonic()
        cached = self._node_cache.get(key)
        if cached and not force and now - cached[0] < NODE_PROBE_CACHE_S:
            return cached[1]
        result = self._probe_node_now(node, private, timeout)
        self._node_cache[key] = (now, result)
        return result

    def nodes(self, nodes: list[Node], *, private: bool) -> list[NodeProbe]:
        """All nodes at the same time (a pool of 5 laptops must not be probed one after the other)."""
        if len(nodes) <= 1:
            return [self.node(node, private=private) for node in nodes]
        with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            return list(pool.map(lambda node: self.node(node, private=private), nodes))

    def forget_node(self, node_name: str | None = None) -> None:
        """Drop the cached probes of one node (after a failed call) or of every node."""
        if node_name is None:
            self._node_cache.clear()
        else:
            self._node_cache = {key: value for key, value in self._node_cache.items() if key[0] != node_name}

    def is_busy(self, probe: NodeProbe) -> bool | None:
        """True/False when the server tells whether it is generating right now, None when unknown."""
        if probe.endpoint is None or is_dry_run():
            return None
        with self.router.http_client() as client:
            return endpoint_busy(client, probe.endpoint, BUSY_CHECK_TIMEOUT_S)

    def _llm_cfg(self) -> dict:
        return self.router.providers["llm"].cfg

    def _probe_node_now(self, node: Node, private: bool, timeout: float | None) -> NodeProbe:
        if not node.endpoints:
            return NodeProbe(endpoint=None, errors=["no endpoint address configured"])
        if is_dry_run():
            preferred = node.preferred_models or self._llm_cfg().get("preferred_models") or ["dry-run-model"]
            return NodeProbe(node.endpoints[0], [preferred[0]], 0.0, [])
        timeout = timeout or float(self._llm_cfg().get("probe_timeout", DEFAULT_NODE_PROBE_TIMEOUT_S))
        return self._probe_endpoints(node.endpoints, private, timeout)

    def _probe_endpoints(self, endpoints: list[Endpoint], private: bool, timeout: float) -> NodeProbe:
        """All endpoints are probed at the same time under one wall-clock limit: a switched-off laptop (a host name
        with several addresses, each timing out in turn) must not stall every task for 10+ seconds."""
        deadline = float(self._llm_cfg().get("probe_deadline", 2 * timeout + 0.5))
        pool = ThreadPoolExecutor(max_workers=len(endpoints))
        futures = [(endpoint, pool.submit(self._probe_one, endpoint, private, timeout)) for endpoint in endpoints]
        try:
            return self._first_answer(futures, deadline)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _first_answer(futures: list[tuple[Endpoint, Future]], deadline_s: float) -> NodeProbe:
        """Preference order: the first endpoint that answered wins; the others only add to the error list."""
        errors: list[str] = []
        end = time.monotonic() + deadline_s
        for endpoint, future in futures:
            try:
                models, latency, error = future.result(timeout=max(0.0, end - time.monotonic()))
            except FutureTimeout:
                errors.append(f"{endpoint.name}: no answer within {deadline_s:.1f} s")
                continue
            if models is not None:
                return NodeProbe(endpoint, models, latency, errors)
            errors.append(f"{endpoint.name}: {error}")
        return NodeProbe(endpoint=None, errors=errors)

    def _probe_one(self, endpoint: Endpoint, private: bool, timeout: float) -> tuple[list[str] | None, float, str]:
        """(models or None, latency, error text) of one address; private data never goes to a public address."""
        if private:
            allowed, reason = netguard.check_host(endpoint.url, self.router.resolver)
            if not allowed and not self._llm_cfg().get("allow_public_endpoint", False):
                return None, 0.0, f"refused: {reason}"
        with self.router.http_client() as client:
            models, latency, error = probe_endpoint(client, endpoint, timeout)
        return models, latency, ("" if models is not None else f"unreachable ({error})")

    # ------------------------------------------------------------ cloud providers
    def cloud(self, provider: Provider, *, force: bool = False) -> tuple[bool, str]:
        """(reachable, reason) via a free GET (default /models). Cached; the key is only sent over verified TLS."""
        if is_dry_run():
            return True, "dry-run"
        settings = provider.cfg.get("probe") or {}
        now = self.router.clock()
        cached = self._cloud_cache.get(provider.name)
        if cached and not force and now - cached[0] < self._cloud_ttl(cached[1], settings):
            return cached[1], cached[2]
        reachable, reason = self._ask_cloud(provider, settings)
        self._cloud_cache[provider.name] = (now, reachable, reason)
        return reachable, reason

    def forget_clouds(self) -> None:
        """Drop every cached cloud probe."""
        self._cloud_cache.clear()

    @staticmethod
    def _cloud_ttl(was_reachable: bool, settings: dict) -> float:
        if was_reachable:
            return float(settings.get("ttl_s", DEFAULT_CLOUD_TTL_S))
        return float(settings.get("neg_ttl_s", DEFAULT_CLOUD_NEG_TTL_S))

    def _ask_cloud(self, provider: Provider, settings: dict) -> tuple[bool, str]:
        """One GET with one quick retry after a connection error."""
        timeout = float(settings.get("timeout_s", DEFAULT_CLOUD_TIMEOUT_S))
        reason = "ConnectError"
        for attempt in range(2):
            try:
                return self._get_models_status(provider, settings, timeout)
            except httpx.HTTPError as error:
                reason = error.__class__.__name__
                if attempt == 0:
                    self.router.sleep(CLOUD_RETRY_PAUSE_S)
        return False, reason

    def _get_models_status(self, provider: Provider, settings: dict, timeout: float) -> tuple[bool, str]:
        key = provider.api_key()
        url = provider.cfg["base_url"].rstrip("/") + settings.get("path", "/models")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        with self.router.http_client() as client:
            response = client.get(url, headers=headers, timeout=httpx.Timeout(timeout, connect=timeout))
        ok = response.status_code == HTTP_OK
        return ok, ("" if ok else f"HTTP {response.status_code}")
