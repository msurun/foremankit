"""USD cost accounting and hard daily/monthly stop for paid providers."""
from __future__ import annotations

from typing import Any

from .state import utc_day, utc_month

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


def spent(state: dict[str, Any], now: float) -> tuple[float, float]:
    """Return the amount spent so far today and this month.

    Returns a tuple of (daily_usd, monthly_usd).
    """
    spend_data = _spend(state, now)
    daily_amount = float(spend_data["day_usd"])
    monthly_amount = float(spend_data["month_usd"])
    return daily_amount, monthly_amount


def check(state: dict[str, Any], budget_cfg: dict[str, Any], now: float, est_usd: float) -> str | None:
    """Return a skip reason when the call could exceed the daily or monthly cap.

    Returns None if the budget is sufficient.
    """
    current_day_spent, current_month_spent = spent(state, now)
    daily_limit = float(budget_cfg.get("daily_usd", 1.0))
    monthly_limit = float(budget_cfg.get("monthly_usd", 10.0))

    if current_day_spent + est_usd > daily_limit:
        return f"daily budget (${current_day_spent:.4f} spent + ~${est_usd:.4f} > ${daily_limit:.2f})"
    if current_month_spent + est_usd > monthly_limit:
        return f"monthly budget (${current_month_spent:.4f} spent + ~${est_usd:.4f} > ${monthly_limit:.2f})"
    return None


def record(state: dict[str, Any], now: float, usd: float) -> None:
    """Record the specified USD amount in the state for the current day and month."""
    spend_data = _spend(state, now)
    current_day_usd = float(spend_data["day_usd"])
    current_month_usd = float(spend_data["month_usd"])

    spend_data["day_usd"] = round(current_day_usd + usd, 8)
    spend_data["month_usd"] = round(current_month_usd + usd, 8)
