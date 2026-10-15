"""Finishes an answer that stopped at its token limit by asking the same model to continue it.

A cut-off paid answer used to be thrown away (and regenerated, paying twice) or returned as a "success" with a warning.
Continuing costs only the missing tail: the prompt is re-sent, but a provider's input cache makes that cheap."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from . import budget
from .config import is_dry_run
from .records import Answer, CallContext, Candidate
from .stitch import continuation_request, seam_is_broken, stitch

if TYPE_CHECKING:
    from .caller import Caller

DEFAULT_MAX_CONTINUATIONS = 2
DEFAULT_CONTINUATIONS_CAP = 6      # most extra rounds one call may ask for (ctx.max_continuations)
DEFAULT_TOTAL_CONTINUATIONS = 8    # most rounds one answer gets over all its resumes: a runaway task stops here
CHARS_PER_TOKEN = 3        # rough size of the partial answer when it is sent back as input


def complete_if_cut(caller: Caller, candidate: Candidate, ctx: CallContext, answer: Answer) -> Answer:
    """The answer, continued up to `calls.max_continuations` times while the model keeps stopping at its limit.

    Every round is a separate call (its usage is recorded by `request_once`); the returned Answer is the joined text
    with the summed tokens, cost and time. It still says `finish_reason == "length"` when the limit was not enough."""
    rounds = _rounds_allowed(caller, ctx)
    total_limit = int(caller.settings.get("max_total_continuations", DEFAULT_TOTAL_CONTINUATIONS))
    for _ in range(rounds):
        if answer.continuations >= total_limit or not _worth_continuing(caller, candidate, ctx, answer):
            break
        more = caller.request_once(candidate, _continuation_context(ctx, answer))
        if more is None:
            break
        previous, answer = answer, _joined(answer, more)
        if more.finish_reason != "length" and seam_is_broken(previous.text, answer.text):
            answer = replace(answer, finish_reason="length")   # the model says it is done but the join is not: go on
    return answer


def _rounds_allowed(caller: Caller, ctx: CallContext) -> int:
    """Rounds for this call: the configured default, or what the caller asked for, never above the configured cap."""
    settings = caller.settings
    if not ctx.max_continuations:
        return int(settings.get("max_continuations", DEFAULT_MAX_CONTINUATIONS))
    return min(ctx.max_continuations, int(settings.get("max_continuations_cap", DEFAULT_CONTINUATIONS_CAP)))


def _worth_continuing(caller: Caller, candidate: Candidate, ctx: CallContext, answer: Answer) -> bool:
    """Cut at the limit, something to continue from, and (for a paid source) the next round fits the budget."""
    if is_dry_run() or answer.finish_reason != "length" or not answer.text.strip():
        return False
    return candidate.provider.free or _round_is_affordable(caller, candidate, ctx, answer)


def _round_is_affordable(caller: Caller, candidate: Candidate, ctx: CallContext, answer: Answer) -> bool:
    """The worst case of one more round (whole prompt plus the partial answer in, a full ceiling out) is within budget."""
    cfg = caller.router.routing.get("budget") or {}
    input_tokens = ctx.est_in + len(answer.text) // CHARS_PER_TOKEN
    worst_case = budget.worst_case_usd(candidate.price, input_tokens, candidate.send_max or ctx.max_out)
    per_call = float(cfg.get("per_call_usd") or 0)
    if per_call and worst_case > per_call and not ctx.allow_cost:
        return False
    state = caller.router.store.load()
    return budget.check(state, cfg, caller.router.clock(), worst_case) is None


def _continuation_context(ctx: CallContext, answer: Answer) -> CallContext:
    """The conversation so far plus the partial answer and the request to go on."""
    messages = ctx.messages + [{"role": "assistant", "content": answer.text},
                               {"role": "user", "content": continuation_request(answer.text)}]
    return replace(ctx, messages=messages)


def _joined(first: Answer, more: Answer) -> Answer:
    """One Answer for both rounds: stitched text, summed usage, cost and time, the last round's stop reason."""
    usage = {key: first.usage.get(key, 0) + more.usage.get(key, 0) for key in first.usage}
    return Answer(stitch(first.text, more.text), more.finish_reason, usage,
                  first.cost_usd + more.cost_usd, first.latency_s + more.latency_s, more.timings, first.continuations + 1)
