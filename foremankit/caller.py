"""Sends one request to one candidate: retries, cooldowns, usage bookkeeping, logging and rejecting unusable answers."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx

from . import budget, quota
from . import speed as speed_module
from .config import is_dry_run
from .continuation import complete_if_cut
from .outputs import write_answer
from .providers import ChatRequest, ProviderError, chat_completion
from .records import Answer, CallContext, Candidate, TaskTooLarge
from .stitch import open_fence

if TYPE_CHECKING:
    from .router import Router

DEFAULT_MAX_RETRIES = 1
RETRY_PAUSE_S = 1.0
DEFAULT_ABORT_WASTE_USD = 0.02
DEFAULT_PREVIEW_CHARS = 600
DEFAULT_PROMPT_TPS = 100.0
ERROR_TEXT_CHARS = 200
SKIP_REASON_CHARS = 120
HTTP_RATE_LIMITED = 429          # "slow down": the provider counts this as a rate limit, so cooldown and retry apply
HTTP_SERVER_ERROR = 500          # first 5xx code: the provider failed on its own side, the call may succeed elsewhere


def reject_served_model(candidate: Candidate, reply: dict[str, Any]) -> None:
    """Raise a transient ProviderError when a router answered with a model the provider lists in
    `reject_served_models` (substrings): openrouter/free sometimes picks a safety classifier ("User Safety: safe")."""
    served = str(reply.get("served_model") or "")
    patterns = candidate.provider.cfg.get("reject_served_models") or []
    if served and any(pattern in served for pattern in patterns):
        raise ProviderError(f"router picked a rejected model: {served}", transient=True)


def cut_off_in_code(answer: Answer) -> bool:
    """True when the model stopped at its token limit with a code block still open (``` or ~~~, any fence length)."""
    return answer.finish_reason == "length" and open_fence(answer.text) is not None


class Caller:
    """Runs one call. `call` returns the result dict, or a short failure reason so the chain can move on."""

    def __init__(self, router: Router):
        self.router = router

    @property
    def settings(self) -> dict[str, Any]:
        """routing.yaml `calls`."""
        return self.router.routing.get("calls") or {}

    def call(self, candidate: Candidate, ctx: CallContext, resume: Answer | None = None) -> dict[str, Any] | str:
        """Run one provider (max `calls.max_retries` retries on transient errors). `resume` = a saved cut-off answer
        to finish instead of asking the question again (continue_from)."""
        log_fields = self._log_fields(candidate, ctx)
        answer = resume or self._first_answer(candidate, ctx, log_fields)
        if isinstance(answer, str):
            return answer
        if not is_dry_run():
            answer = complete_if_cut(self, candidate, ctx, answer)
            problem = self._unusable_reason(candidate, ctx, answer, log_fields)
            if problem:
                return problem
        return self._success(candidate, ctx, answer, log_fields)

    def _first_answer(self, candidate: Candidate, ctx: CallContext, log_fields: dict[str, Any]) -> Answer | str:
        """The model's reply to the task (its usage recorded), or the failure reason."""
        start = time.perf_counter()
        try:
            reply = self._send(candidate, ctx)
        except ProviderError as error:
            return self._handle_failure(error, candidate, log_fields, time.perf_counter() - start)
        answer = self._to_answer(reply, candidate, ctx, time.perf_counter() - start)
        if not is_dry_run():
            self._record_usage(candidate, answer)
        return answer

    def request_once(self, candidate: Candidate, ctx: CallContext) -> Answer | None:
        """One more request to the same candidate (a continuation round): usage is recorded; None when it failed."""
        start = time.perf_counter()
        try:
            reply = self._send(candidate, ctx)
        except ProviderError:
            return None
        answer = self._to_answer(reply, candidate, ctx, time.perf_counter() - start)
        self._record_usage(candidate, answer)
        return answer

    @staticmethod
    def _log_fields(candidate: Candidate, ctx: CallContext) -> dict[str, Any]:
        return {"event": "call", "id": ctx.call_id, "kind": ctx.kind, "sensitivity": ctx.sensitivity,
                "provider": candidate.provider.name, "node": candidate.node or None, "model": candidate.model_id,
                "endpoint": candidate.endpoint, "escalate": ctx.escalate, "dry_run": is_dry_run(),
                "reasoning": (candidate.extra or {}).get("reasoning_effort"), "extra": candidate.extra}

    def _send(self, candidate: Candidate, ctx: CallContext) -> dict[str, Any]:
        if is_dry_run():
            return {"text": f"[DRY RUN] would call {candidate.provider.name}/{candidate.model_id} "
                            f"(~{ctx.est_in} input tokens, max_output={ctx.max_out}). No network call made.",
                    "usage": {"prompt_tokens": ctx.est_in, "completion_tokens": 0, "cache_hit": 0,
                              "reasoning_tokens": 0},
                    "finish_reason": "dry_run"}
        with self.router.http_client() as client:
            return self._chat_with_retries(client, candidate, ctx)

    def _chat_with_retries(self, client: httpx.Client, candidate: Candidate, ctx: CallContext) -> dict[str, Any]:
        """Transient errors (not timeouts) are retried up to `calls.max_retries` times; the last error is raised."""
        max_retries = int(self.settings.get("max_retries", DEFAULT_MAX_RETRIES))
        attempt = 0
        while True:
            try:
                reply = chat_completion(client, self._chat_request(candidate, ctx))
                reject_served_model(candidate, reply)
                return reply
            except ProviderError as error:
                if not self._worth_retrying(error) or attempt >= max_retries:
                    raise
                attempt += 1
                self.router.sleep(RETRY_PAUSE_S)

    @staticmethod
    def _chat_request(candidate: Candidate, ctx: CallContext) -> ChatRequest:
        return ChatRequest(candidate.base_url, candidate.api_key, candidate.model_id, ctx.messages,
                           candidate.send_max or ctx.max_out, candidate.provider.timeout, candidate.extra)

    @staticmethod
    def _worth_retrying(error: ProviderError) -> bool:
        return bool(error.transient) and not error.timed_out

    def _handle_failure(self, error: ProviderError, candidate: Candidate, log_fields: dict[str, Any],
                        latency_s: float) -> str:
        """Start a cooldown for rate limits and server errors, forget the node's probe, log; return the skip reason."""
        if error.status == HTTP_RATE_LIMITED or (error.status or 0) >= HTTP_SERVER_ERROR or error.transient:
            self._start_cooldown(error, candidate)
        if candidate.provider.is_llm:
            self.router.prober.forget_node(candidate.node)
        self.router.log.write({**log_fields, "success": False, "error": str(error)[:ERROR_TEXT_CHARS],
                               "latency_s": round(latency_s, 2)})
        return str(error)[:SKIP_REASON_CHARS]

    def _start_cooldown(self, error: ProviderError, candidate: Candidate) -> None:
        failure = quota.FailedCall(candidate.key, self.router.clock(), error.status, error.retry_after)
        self.router.store.update(lambda state: quota.set_cooldown(state, failure, self.settings))

    @staticmethod
    def _to_answer(reply: dict[str, Any], candidate: Candidate, ctx: CallContext, latency_s: float) -> Answer:
        usage = reply["usage"]
        if not usage["prompt_tokens"] and not is_dry_run():
            usage["prompt_tokens"] = ctx.est_in  # provider omitted usage: estimate for quotas
        cost = 0.0 if is_dry_run() else budget.cost_usd(candidate.price, usage)
        return Answer(reply["text"], reply.get("finish_reason"), usage, cost, latency_s, reply.get("timings") or {},
                      served_model=reply.get("served_model"))

    def _record_usage(self, candidate: Candidate, answer: Answer) -> None:
        """Count the tokens against the quota, end a cooldown, feed the node's measured speed and the spend."""
        now = self.router.clock()
        tokens = answer.usage["prompt_tokens"] + answer.usage["completion_tokens"]
        learning = speed_module.settings(self.router.cfg.get("nodes"))
        measured = self._measured_speed(candidate, answer)

        def record(state: dict[str, Any]) -> None:
            """Apply the call to the state file: quota, cooldown, node speed, spend."""
            quota.record(state, candidate.key, tokens, now)
            quota.clear_cooldown(state, candidate.key)
            if measured and candidate.learn:
                speed_module.observe(state, speed_module.Observation(candidate.node, measured, now),
                                     float(learning["alpha"]))
            if answer.cost_usd > 0:
                budget.record(state, now, answer.cost_usd)
        self.router.store.update(record)

    @staticmethod
    def _measured_speed(candidate: Candidate, answer: Answer) -> dict[str, Any]:
        if not candidate.node:
            return {}
        prompt_tps = float((candidate.speed or {}).get("prompt_tps", DEFAULT_PROMPT_TPS))
        return speed_module.measure({"usage": answer.usage, "timings": answer.timings}, answer.latency_s, prompt_tps)

    def _unusable_reason(self, candidate: Candidate, ctx: CallContext, answer: Answer,
                         log_fields: dict[str, Any]) -> str | None:
        """A failure reason for an empty or cut-off answer (None = usable). Raises TaskTooLarge when a paid model
        wasted more than `calls.abort_waste_usd`: retrying on another paid model would pay twice for the same mistake."""
        name = candidate.provider.name
        if not answer.text.strip() and answer.finish_reason == "length":
            self._log_rejected(log_fields, answer, "empty answer: reasoning used the whole budget")
            self._abort_if_wasteful(answer, f"{name} spent ${answer.cost_usd:.3f} and returned nothing: reasoning used the "
                                    "whole token budget. Split the task or ask for a smaller answer; not retried on "
                                    "another paid model.")
            return "empty answer (reasoning used the whole token budget)"
        if cut_off_in_code(answer):
            partial = write_answer(self.router.out_dir(), candidate, ctx, answer)
            self._log_rejected(log_fields, answer, "answer cut off inside a code block", path=partial)
            self._abort_if_wasteful(answer, f"{name} spent ${answer.cost_usd:.3f} and the answer was cut off inside a "
                                    f"code block (partial text: {partial}). Finish it with continue_from=\"{ctx.call_id}\" "
                                    "(same task) or split the task; not retried on another paid model.")
            return f"answer cut off inside a code block (partial text: {partial}; continue_from=\"{ctx.call_id}\")"
        return None

    def _log_rejected(self, log_fields: dict[str, Any], answer: Answer, error: str, path: str | None = None) -> None:
        record = {**log_fields, "success": False, "error": error, "tokens_out": answer.usage["completion_tokens"],
                  "reasoning_tokens": answer.usage.get("reasoning_tokens", 0),
                  "cost_usd": round(answer.cost_usd, 6), "latency_s": round(answer.latency_s, 2)}
        if path:
            record["path"] = path
        self.router.log.write(record)

    def _abort_if_wasteful(self, answer: Answer, message: str) -> None:
        if answer.cost_usd > float(self.settings.get("abort_waste_usd", DEFAULT_ABORT_WASTE_USD)):
            raise TaskTooLarge(message)

    def _success(self, candidate: Candidate, ctx: CallContext, answer: Answer, log_fields: dict[str, Any]) -> dict[str, Any]:
        path = write_answer(self.router.out_dir(), candidate, ctx, answer)
        record = {**log_fields, "success": True, "tokens_in": answer.usage["prompt_tokens"],
                  "tokens_out": answer.usage["completion_tokens"], "cache_hit": answer.usage["cache_hit"],
                  "reasoning_tokens": answer.usage.get("reasoning_tokens", 0),
                  "cost_usd": round(answer.cost_usd, 6), "latency_s": round(answer.latency_s, 2), "path": path,
                  "truncated": answer.finish_reason == "length", "continuations": answer.continuations,
                  "served_model": answer.served_model}
        if self.settings.get("log_prompts"):
            record["prompt"] = ctx.messages
        self.router.log.write(record)
        return self._result(candidate, ctx, answer, path)

    def _result(self, candidate: Candidate, ctx: CallContext, answer: Answer, path: str) -> dict[str, Any]:
        limit = int(self.settings.get("preview_chars", DEFAULT_PREVIEW_CHARS))
        text = answer.text
        return {"id": ctx.call_id, "provider": candidate.provider.name, "node": candidate.node or None,
                "model": candidate.model_id, "served_model": answer.served_model, "endpoint": candidate.endpoint,
                "tokens_in": answer.usage["prompt_tokens"], "tokens_out": answer.usage["completion_tokens"],
                "cache_hit": answer.usage["cache_hit"], "cost_usd": answer.cost_usd, "latency_s": answer.latency_s,
                "path": path, "truncated": answer.finish_reason == "length", "dry_run": is_dry_run(),
                "preview": text[:limit] + ("…" if len(text) > limit else ""), "_text": text, "_cand": candidate,
                "_ctx": ctx}
