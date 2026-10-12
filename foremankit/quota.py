"""Per-provider rate counters (rpm/rpd/tpd) and cooldowns after 429/5xx."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import utc_day

SECONDS_PER_MINUTE = 60
DEFAULT_COOLDOWN_429_S = [30, 120, 600, 1800]   # waits after the 1st, 2nd, ... rate-limit answer in a row
DEFAULT_COOLDOWN_5XX_S = 30.0
FALLBACK_COOLDOWN_S = 60.0                      # when no 429 steps are configured at all
HTTP_RATE_LIMITED = 429


@dataclass(frozen=True)
class QuotaRequest:
    """One planned call, to be checked against a provider's limits."""
    provider: str
    limits: dict[str, Any]
    now: float
    est_tokens: int


@dataclass(frozen=True)
class FailedCall:
    """A call that failed: who, when, and what the server said."""
    provider: str
    now: float
    status: int | None = None
    retry_after: float | None = None


def _bucket(state: dict[str, Any], provider: str, now: float) -> dict[str, Any]:
    quota = state.setdefault("quota", {}).setdefault(provider, {})
    day = utc_day(now)
    if quota.get("day") != day:
        quota.update(day=day, rpd=0, tpd=0)
    quota["minute"] = [stamp for stamp in quota.get("minute", []) if now - stamp < SECONDS_PER_MINUTE]
    return quota


def remaining(state: dict[str, Any], provider: str, limits: dict[str, Any], now: float) -> dict[str, Any]:
    """What is left of each limit (rpm, rpd, tpd); None = that limit is not set."""
    bucket = _bucket(state, provider, now)
    left = {}
    for key, used in (("rpm", len(bucket["minute"])), ("rpd", bucket["rpd"]), ("tpd", bucket["tpd"])):
        limit = (limits or {}).get(key)
        left[key] = None if limit is None else max(0, int(limit) - used)
    return left


def check(state: dict[str, Any], request: QuotaRequest) -> str | None:
    """Return a skip reason, or None if the call fits within limits."""
    left = remaining(state, request.provider, request.limits, request.now)
    if left["rpm"] is not None and left["rpm"] <= 0:
        return "rpm limit reached"
    if left["rpd"] is not None and left["rpd"] <= 0:
        return "rpd limit reached"
    if left["tpd"] is not None and left["tpd"] < request.est_tokens:
        return f"tpd limit ({left['tpd']} tokens left, need ~{request.est_tokens})"
    return None


def record(state: dict[str, Any], provider: str, tokens: int, now: float) -> None:
    """Count one finished call and its tokens."""
    bucket = _bucket(state, provider, now)
    bucket["minute"].append(now)
    bucket["rpd"] += 1
    bucket["tpd"] += int(tokens)


def cooldown_left(state: dict[str, Any], provider: str, now: float) -> float:
    """Seconds the provider is still paused after an error (0 = free to use)."""
    entry = state.get("cooldown", {}).get(provider)
    if not entry:
        return 0.0
    return max(0.0, float(entry.get("until", 0)) - now)


def set_cooldown(state: dict[str, Any], failure: FailedCall, calls_cfg: dict[str, Any]) -> float:
    """Pause a provider after a rate limit (growing steps) or a server error (fixed time); returns the pause in s.
    The server's own Retry-After wins when it asks for longer."""
    entry = state.setdefault("cooldown", {}).setdefault(failure.provider, {"strikes": 0, "until": 0})
    if failure.status == HTTP_RATE_LIMITED:
        delay = _next_rate_limit_delay(entry, calls_cfg.get("cooldown_429_s") or DEFAULT_COOLDOWN_429_S)
    else:
        delay = float(calls_cfg.get("cooldown_5xx_s", DEFAULT_COOLDOWN_5XX_S))
    if failure.retry_after:
        delay = max(delay, failure.retry_after)
    entry["until"] = failure.now + delay
    return delay


def _next_rate_limit_delay(entry: dict[str, Any], steps: list[float]) -> float:
    """Each rate limit in a row waits longer (the last step repeats); the strike count lives in the state file."""
    strikes = int(entry.get("strikes", 0))
    entry["strikes"] = strikes + 1
    return steps[min(strikes, len(steps) - 1)] if steps else FALLBACK_COOLDOWN_S


def clear_cooldown(state: dict[str, Any], provider: str) -> None:
    """End the pause (and forget the strikes) after a successful call."""
    state.get("cooldown", {}).pop(provider, None)
