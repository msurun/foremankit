"""USD cost accounting and hard daily/monthly stops for paid providers, kept apart for delegated work and for an
external orchestrator (Claude Code running on DeepSeek/Gemini): each has its own caps and its own counters."""
from __future__ import annotations

from typing import Any

from .state import utc_day, utc_month

# Spend categories. The caps of a category are `daily_usd` / `monthly_usd` in routing.yaml `budget:`, with this prefix.
DELEGATION = "delegation"
ORCHESTRATOR = "orchestrator"
CAP_PREFIX = {DELEGATION: "", ORCHESTRATOR: "orchestrator_"}

# Constants for token pricing keys
INPUT_MISS_KEY = "input_miss"
INPUT_HIT_KEY = "input_hit"
OUTPUT_KEY = "output"


def cost_usd(price: dict[str, Any] | None, usage: dict[str, int]) -> float:
    """Calculate the cost in USD for a given price and usage.

    Handles cache hits by distinguishing between input_miss and input_hit prices.
    If input_hit is not specified, it falls back to input_miss.
    """
    if not price:
        return 0.0

    input_miss_count, cache_hit_count, completion_tokens = _extract_usage_counts(usage)
    input_miss_price, input_hit_price, output_price = _extract_prices(price)

    total = (input_miss_count * input_miss_price
             + cache_hit_count * input_hit_price
             + completion_tokens * output_price)

    return total / 1_000_000


def _extract_usage_counts(usage: dict[str, int]) -> tuple[int, int, int]:
    """Extract prompt miss count, cache hit count, and completion count from usage."""
    cache_hit_count = usage.get("cache_hit", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    input_miss_count = max(0, prompt_tokens - cache_hit_count)
    return input_miss_count, cache_hit_count, completion_tokens


def _extract_prices(price: dict[str, Any]) -> tuple[float, float, float]:
    """Extract input miss price, input hit price, and output price from price dict."""
    input_miss_price = float(price.get(INPUT_MISS_KEY, 0))
    input_hit_price = float(price.get(INPUT_HIT_KEY, input_miss_price))
    output_price = float(price.get(OUTPUT_KEY, 0))
    return input_miss_price, input_hit_price, output_price


def worst_case_usd(price: dict[str, Any] | None, est_input_tokens: int, max_output_tokens: int) -> float:
    """Calculate the worst-case cost assuming no cache hits."""
    return cost_usd(price, {"prompt_tokens": est_input_tokens, "cache_hit": 0,
                            "completion_tokens": max_output_tokens})


def _spend(state: dict[str, Any], now: float) -> dict[str, Any]:
    """Ensure spend tracking for the current day and month exists in the state.

    Resets daily and monthly counters if the time period has changed.
    """
    spend = state.setdefault("spend", {})
    current_day = utc_day(now)
    current_month = utc_month(now)

    if spend.get("day") != current_day:
        spend.update(day=current_day, day_usd=0.0)
    if spend.get("month") != current_month:
        spend.update(month=current_month, month_usd=0.0)
    return spend


def _category_spend(state: dict[str, Any], now: float, category: str) -> dict[str, Any]:
    """The counters of one spend category (day and month, reset when the UTC day or month changes)."""
    entry = _spend(state, now).setdefault("categories", {}).setdefault(category, {})
    if entry.get("day") != utc_day(now):
        entry.update(day=utc_day(now), day_usd=0.0)
    if entry.get("month") != utc_month(now):
        entry.update(month=utc_month(now), month_usd=0.0)
    return entry


def spent(state: dict[str, Any], now: float, category: str | None = None) -> tuple[float, float]:
    """Return the amount spent so far today and this month: of one category, or of everything when `category` is None.

    Returns a tuple of (daily_usd, monthly_usd).
    """
    spend_data = _spend(state, now) if category is None else _category_spend(state, now, category)
    return float(spend_data["day_usd"]), float(spend_data["month_usd"])


def caps(budget_cfg: dict[str, Any], category: str = DELEGATION) -> tuple[float | None, float | None]:
    """The (daily, monthly) caps of a category; None = no cap. Delegation defaults to 1 / 10 USD as before, the
    orchestrator has no cap until `orchestrator_daily_usd` / `orchestrator_monthly_usd` are set."""
    prefix = CAP_PREFIX[category]
    default_daily, default_monthly = (1.0, 10.0) if category == DELEGATION else (None, None)
    daily = budget_cfg.get(f"{prefix}daily_usd", default_daily)
    monthly = budget_cfg.get(f"{prefix}monthly_usd", default_monthly)
    return (None if daily is None else float(daily)), (None if monthly is None else float(monthly))


# codecheck: allow too-many-parameters (callers pass state, budget, now and the estimate positionally; category is optional)
def check(state: dict[str, Any], budget_cfg: dict[str, Any], now: float, est_usd: float,
          category: str = DELEGATION) -> str | None:
    """Return a skip reason when the call could exceed the daily or monthly cap of its category.

    Returns None if the budget is sufficient.
    """
    day_spent, month_spent = spent(state, now, category)
    daily_limit, monthly_limit = caps(budget_cfg, category)
    label = "" if category == DELEGATION else f"{category} "
    if daily_limit is not None and day_spent + est_usd > daily_limit:
        return f"{label}daily budget (${day_spent:.4f} spent + ~${est_usd:.4f} > ${daily_limit:.2f})"
    if monthly_limit is not None and month_spent + est_usd > monthly_limit:
        return f"{label}monthly budget (${month_spent:.4f} spent + ~${est_usd:.4f} > ${monthly_limit:.2f})"
    return None


def record(state: dict[str, Any], now: float, usd: float, category: str = DELEGATION) -> None:
    """Record the specified USD amount for the current day and month: in the total and in its category."""
    for counters in (_spend(state, now), _category_spend(state, now, category)):
        counters["day_usd"] = round(float(counters["day_usd"]) + usd, 8)
        counters["month_usd"] = round(float(counters["month_usd"]) + usd, 8)
