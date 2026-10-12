"""Walks the chain of sources of a task until one of them answers (cheap first, deferred ones on a last pass)."""
from __future__ import annotations

import secrets
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterator

from . import netguard, prompting, reasoning
from .records import (MAX_OUTPUT_CAP, MIN_OUTPUT_TOKENS, SENSITIVITIES, CallContext, Candidate, RouterError,
                      TaskRequest)
from .selection import estimate_seconds, unsuited_reason

if TYPE_CHECKING:
    from .router import Router

OK, SKIP, DEFER = "ok", "skip", "defer"   # outcomes of trying one source
Outcome = tuple[str, Any]


def build_chain(routing: dict[str, Any], kind: str, sensitivity: str, escalate: bool) -> list[str]:
    """The sources to try, in order. Private tasks use only the private chain; internal ones add the local LLM:
    in front when the cheap pass is meant (free and private), behind when the strongest model is wanted."""
    chains = routing.get("chains") or {}
    if sensitivity == "private":
        return list(chains.get("private") or [])
    chain = list(routing.get("escalation") or chains[kind]) if escalate else list(chains[kind])
    if sensitivity != "internal":
        return chain
    local = list(chains.get("private", []))
    if escalate or kind == "hard_code":
        return chain + [entry for entry in local if entry not in chain]
    return local + [entry for entry in chain if entry not in local]


def parse_entry(entry: Any) -> tuple[str, str | None, str]:
    """'provider:alias' or 'llm@node' -> (provider name, alias or None, node name or '')."""
    name, _, alias = str(entry).partition(":")
    name, _, node_name = name.partition("@")
    return name, alias or None, node_name


def prepare_context(router: Router, request: TaskRequest) -> tuple[CallContext, list[str]]:
    """Validate the request and gather what every call of this task needs; also the notes shown with the result."""
    if request.sensitivity not in SENSITIVITIES:
        raise RouterError(f"sensitivity must be one of {SENSITIVITIES}")
    chains = router.routing.get("chains") or {}
    if request.kind not in chains:
        raise RouterError(f"unknown kind {request.kind!r}; use one of {sorted(chains)}")
    sensitivity = "private" if request.kind == "private" else request.sensitivity
    max_out = max(MIN_OUTPUT_TOKENS, min(int(request.max_output_tokens), MAX_OUTPUT_CAP))
    level = reasoning.reasoning_level(router.routing, request.kind, sensitivity, request.escalate)
    files = prompting.read_context_files(router.routing, request.context_files or [], sensitivity != "normal")
    messages = prompting.build_messages(router.routing, request, files)
    ctx = CallContext(
        call_id=f"{datetime.now():%Y%m%d-%H%M%S}-{request.kind}-{secrets.token_hex(2)}", kind=request.kind,
        sensitivity=sensitivity, escalate=request.escalate,
        est_in=prompting.estimate_input_tokens(router.routing, messages), max_out=max_out, messages=messages,
        reasoning=reasoning.cap_for_large_output(router.routing, level, max_out), allow_cost=request.allow_cost)
    return ctx, reasoning.size_notes(router.routing, max_out)


class ChainRun:
    """One walk along the chain of one task. `exclude` = sources that must not serve it (the author, for a review)."""

    def __init__(self, router: Router, request: TaskRequest, exclude: tuple[str, ...] = ()):
        self.router = router
        self.exclude = exclude
        self.ctx, self.notes = prepare_context(router, request)
        self.chain = build_chain(router.routing, self.ctx.kind, self.ctx.sensitivity, self.ctx.escalate)
        self.nodes = {node.name: node for node in router.nodes()}
        self.skipped: list[str] = []
        self.deferred: list[str] = []

    def execute(self) -> dict[str, Any]:
        """The result of the first source that answers; raises when none does."""
        for entry in self._entries():
            status, payload = self._attempt(entry, wait=False)
            if status == OK:
                return self._finished(payload)
            if status == DEFER:
                self.deferred.append(entry)
            self.skipped.append(f"{entry}({payload})")
        return self._run_deferred()

    def _run_deferred(self) -> dict[str, Any]:
        """Nothing else worked: run the deferred sources (waiting for a free slot) instead of failing."""
        for entry in self.deferred:
            status, payload = self._attempt(entry, wait=True, force=True)
            self.skipped = [text for text in self.skipped if not text.startswith(f"{entry}(")]
            if status == OK:
                return self._finished(payload)
            reason = payload if status == SKIP else "timed out waiting for a free slot"
            self.skipped.append(f"{entry}({reason})")
        raise self._failure()

    def _finished(self, result: dict[str, Any]) -> dict[str, Any]:
        result["skipped"], result["notes"] = self.skipped, self.notes
        return result

    def _failure(self) -> Exception:
        tried = "; ".join(self.skipped)
        if self.ctx.sensitivity != "normal":
            return netguard.PrivacyError(
                f"{self.ctx.sensitivity} task refused (fail closed): no eligible provider available. "
                f"Nothing was sent to any provider not allowed for {self.ctx.sensitivity} data. Tried: {tried}")
        return RouterError(f"no provider available for kind={self.ctx.kind}. Tried: {tried}")

    def _entries(self) -> Iterator[str]:
        """The chain entry `llm` is replaced, when it is reached, by the pool's nodes (fastest expected first)."""
        for entry in self.chain:
            source = str(entry).partition(":")[0]   # keeps the node of an explicit `llm@node` entry (that one is not expanded)
            if source == "llm":
                state = self.router.store.load()
                yield from (name for name in self.router.selector.llm_entries(self.ctx, state, self.skipped)
                            if name not in self.exclude)
            elif source not in self.exclude:
                yield entry

    def _attempt(self, entry: str, wait: bool, force: bool = False) -> Outcome:
        """('ok', result) | ('skip', reason) | ('defer', reason). 'defer' = usable but not now: all its max_concurrent
        slots are busy, or the task is too long for a slow local model (providers.yaml `speed`). The first pass moves
        on to the next source; the last pass (force=True) runs the deferred ones."""
        name, alias, node_name = parse_entry(entry)
        provider = self.router.providers.get(name)
        if provider is None:
            return SKIP, "unknown"
        candidate = self.router.candidate_builder.build(provider, alias, self.ctx, self.nodes.get(node_name))
        if isinstance(candidate, str):
            return SKIP, candidate
        problem = (self._private_transport_problem(candidate) or self._suitability_problem(candidate, force)
                   or self._busy_problem(candidate, force))
        if problem:
            return problem
        return self._run_in_slot(candidate, wait)

    def _private_transport_problem(self, candidate: Candidate) -> Outcome | None:
        """Defence in depth: re-check the transport right before private data is sent."""
        if self.ctx.sensitivity != "private":
            return None
        provider = candidate.provider
        try:
            netguard.assert_private_ok(
                candidate.base_url, allow_public_endpoint=bool(provider.cfg.get("allow_public_endpoint")),
                resolver=self.router.resolver, warn=self.router.warn)
        except netguard.PrivacyError as error:
            return SKIP, str(error)
        return None

    def _suitability_problem(self, candidate: Candidate, force: bool) -> Outcome | None:
        """A slow local node only takes suitable kinds and sizes; a loan from a higher-priority laptop never takes a
        long task, even when forced, so its owner is not blocked."""
        speed = candidate.speed     # a local-LLM node brings its own measured speed (nodes.yaml)
        if speed and not force:
            policy = candidate.kinds if candidate.kinds is not None else (self.router.routing.get("llm") or {}).get("kinds")
            why = unsuited_reason(policy, self.ctx, speed)
            if why:
                return DEFER, why
        if candidate.loan_max_s:
            seconds = estimate_seconds(speed, self.ctx.est_in, self.ctx.max_out)
            if seconds > candidate.loan_max_s:
                return SKIP, (f"{candidate.endpoint}: task too long for a priority loan "
                              f"(~{seconds:.0f} s > {candidate.loan_max_s:.0f} s)")
        return None

    def _busy_problem(self, candidate: Candidate, force: bool) -> Outcome | None:
        """Another session or machine may have started a request on this node since the ranking looked at it. The node
        runs one request at a time, so a second one would just wait in its queue (and lose its prompt cache): ask once
        more right before sending and, on the first pass, move on to the next source. The last pass (force) waits."""
        node = self.nodes.get(candidate.node)
        if force or node is None:
            return None
        probe = self.router.prober.node(node, private=self.ctx.sensitivity == "private")
        return (DEFER, "busy: another request is using it") if self.router.prober.is_busy(probe) else None

    def _run_in_slot(self, candidate: Candidate, wait: bool) -> Outcome:
        slot = self.router.slots.get(candidate.key, candidate.limit)
        if slot is not None:
            acquired = slot.acquire(timeout=candidate.provider.timeout) if wait else slot.acquire(blocking=False)
            if not acquired:
                return DEFER, "busy: another request is using it"
        try:
            result = self.router.caller.call(candidate, self.ctx)
        finally:
            if slot is not None:
                slot.release()
        return (SKIP, result) if isinstance(result, str) else (OK, result)
