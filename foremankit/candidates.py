"""Turns a chain entry (a cloud model or a lab node) into a Candidate that is ready to send, or into a skip reason."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import budget, quota
from .config import is_dry_run
from .delegation import category_of, is_switched_off
from .nodes import Node
from .providers import Provider, endpoint_busy, pick_model
from .records import CallContext, Candidate

if TYPE_CHECKING:
    from .router import Router


class CandidateBuilder:
    """Checks one source against the task (data level, switches, keys, quotas, budget) and describes how to call it."""

    def __init__(self, router: Router):
        self.router = router

    def build(self, provider: Provider, alias: str | None, ctx: CallContext, node: Node | None = None) -> Candidate | str:
        """A Candidate, or the reason to skip this source. For the pool `llm`, `node` is the machine that will serve."""
        if not provider.eligible(ctx.sensitivity):
            return f"not allowed for {ctx.sensitivity} data"
        state, now = self.router.store.load(), self.router.clock()
        category = category_of(provider)
        if is_switched_off(state, category):
            return f"delegation to {category} sources is switched off (foremankit delegation {category} on)"
        extra, send_max = self._thinking_settings(provider, ctx)
        if provider.is_llm:
            candidate = self._lab_node_candidate(provider, node, ctx, state)
        else:
            candidate = self._cloud_candidate(provider, alias)
        if isinstance(candidate, str):
            return candidate
        candidate.send_max, candidate.extra = send_max, extra
        problem = self._limit_problem(candidate, ctx, state, now)
        return problem or candidate

    @staticmethod
    def _thinking_settings(provider: Provider, ctx: CallContext) -> tuple[dict[str, Any] | None, int]:
        """Thinking models spend max_tokens on hidden reasoning: send the effort level and add headroom so the answer
        is not cut or empty (the headroom is also counted in the worst-case cost estimate)."""
        thinking = provider.cfg.get("thinking") or {}
        reasoning = ctx.reasoning or thinking.get("default_level")   # fail closed: an unset kind must not think
        if not (thinking and reasoning):
            return None, ctx.max_out
        level = (thinking.get("level_map") or {}).get(reasoning, reasoning)   # e.g. Gemini has no "minimal"
        extra = {thinking["param"]: level} if thinking.get("param") else None
        by_level = (thinking.get("extra_by_level") or {}).get(level)
        if by_level:   # per-level request fields: local Qwen enable_thinking=false (142 s -> 23 s), DeepSeek thinking on/off
            extra = {**(extra or {}), **by_level}
        headroom = int((thinking.get("headroom") or {}).get(level, 0))
        return extra, ctx.max_out + headroom

    def _lab_node_candidate(self, provider: Provider, node: Node | None, ctx: CallContext,
                            state: dict[str, Any]) -> Candidate | str:
        if node is None:
            return "no node selected"
        problem = self._node_problem(node, ctx, state)
        if problem:
            return problem
        probe = self.router.prober.node(node, private=ctx.sensitivity == "private")
        if probe.endpoint is None:
            return "unreachable: " + "; ".join(probe.errors or ["no endpoint"])
        model = pick_model(node.preferred_models or provider.cfg.get("preferred_models"),
                           provider.cfg.get("exclude_models"), probe.models)
        if not model:
            return f"{probe.endpoint.name}: no models pulled"
        if node.idle_only and self._owner_is_busy(probe.endpoint):
            return f"{probe.endpoint.name}: busy with its owner's request (priority loan only when idle)"
        return self._node_candidate(provider, node, (probe.endpoint, model), state)

    def _node_candidate(self, provider: Provider, node: Node, target: tuple[Any, str],
                        state: dict[str, Any]) -> Candidate:
        endpoint, model = target
        speed = self.router.selector.node_speed(node, state, self.router.clock())
        return Candidate(provider, "local", model, {}, endpoint.url + "/v1", endpoint.api_key, endpoint.name,
                         node=node.name, speed=speed, loan_max_s=node.loan_max_s if node.idle_only else None,
                         kinds=node.kinds, limit=node.max_concurrent, learn=node.learn)

    def _node_problem(self, node: Node, ctx: CallContext, state: dict[str, Any]) -> str | None:
        if not node.accepts(ctx.sensitivity):    # defence in depth: the node's own data level, whatever the chain says
            return f"{node.name}: accepts up to {node.accept} data"
        toggle, source = self.router.selector.is_enabled(node, state)
        return f"disabled ({source})" if toggle == "off" else None

    def _owner_is_busy(self, endpoint: Any) -> bool:
        """A loan from a higher-priority machine is only allowed while its owner is not generating."""
        if is_dry_run():
            return False
        with self.router.http_client() as client:
            return endpoint_busy(client, endpoint) is True

    def _cloud_candidate(self, provider: Provider, alias: str | None) -> Candidate | str:
        if provider.cfg.get("enabled", True) is False:
            return "disabled in config"
        key = provider.api_key()
        if not key:
            return f"no key ({provider.cfg.get('key_env')})"
        reason = self.router.cloud_enabled(provider)  # probes only providers with a key
        if reason:
            return reason
        try:
            alias, model_id, price = provider.model(alias)
        except KeyError as error:
            return str(error)
        return Candidate(provider, alias, model_id, price, provider.cfg["base_url"], key,
                         limit=provider.cfg.get("max_concurrent"))

    def _limit_problem(self, candidate: Candidate, ctx: CallContext, state: dict[str, Any], now: float) -> str | None:
        """Cooldown, quota and budget checks (None = the source may be used)."""
        cooldown = quota.cooldown_left(state, candidate.key, now)
        if cooldown > 0:
            return f"cooldown {cooldown:.0f}s"
        provider = candidate.provider
        reason = quota.check(state, quota.QuotaRequest(candidate.key, provider.limits, now,
                                                       ctx.est_in + candidate.send_max))
        if reason or provider.free:
            return reason or None
        return self._cost_problem(candidate, ctx, state, now)

    def _cost_problem(self, candidate: Candidate, ctx: CallContext, state: dict[str, Any], now: float) -> str | None:
        budget_cfg = self.router.routing.get("budget") or {}
        worst_case = budget.worst_case_usd(candidate.price, ctx.est_in, candidate.send_max)
        per_call = float(budget_cfg.get("per_call_usd") or 0)
        if per_call and worst_case > per_call and not ctx.allow_cost:
            return (f"worst-case cost ~${worst_case:.3f} is above the per-call cap ${per_call:.2f} "
                    f"(split the task or pass allow_cost=true)")
        return budget.check(state, budget_cfg, now, worst_case) or None
