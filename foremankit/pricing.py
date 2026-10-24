"""Pricing logic for the ForemanKit.

This module calculates the effective price of a model based on the provider's pricing schedule and price changes.
It does not perform I/O or use the system clock; the timestamp is a parameter.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

def is_peak(schedule: dict | None, ts: float) -> bool | None:
    """Return whether the timestamp is in a peak period.

    Args:
        schedule: The pricing schedule, or None.
        ts: The timestamp to check.

    Returns:
        None if there is no schedule, otherwise True if the timestamp is in a peak period.
    """
    if not schedule:
        return None

    dt = datetime.fromtimestamp(ts, timezone.utc)
    weekday = dt.weekday()
    hour = dt.hour

    if weekday not in schedule.get("peak_weekdays", []):
        return False

    for start, end in schedule.get("peak_utc_hours", []):
        if start <= hour < end:
            return True

    return False

def base_price(entry: dict, ts: float) -> dict:
    """Return the base price of a model at a given timestamp.

    Args:
        entry: The model entry from providers.yaml.
        ts: The timestamp to check.

    Returns:
        The base price of the model at the given timestamp.
    """
    price = entry.get("price", {})
    price_from = entry.get("price_from", {})
    dt = datetime.fromtimestamp(ts, timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")

    for date in sorted(price_from.keys()):
        if date <= date_str:
            price = price_from[date]

    return price.copy()

def effective_price(entry: dict, provider_cfg: dict, ts: float) -> dict:
    """Return the effective price of a model at a given timestamp.

    Args:
        entry: The model entry from providers.yaml.
        provider_cfg: The provider configuration.
        ts: The timestamp to check.

    Returns:
        The effective price of the model at the given timestamp.
    """
    schedule = provider_cfg.get("price_schedule")
    peak = is_peak(schedule, ts)
    price = base_price(entry, ts)

    if peak is False:
        factor = schedule.get("off_peak_factor", 0.5)
        for key, value in price.items():
            if isinstance(value, (int, float)):
                price[key] = value * factor

    return price

def next_phase_change(schedule: dict | None, ts: float) -> float | None:
    """Return the next timestamp when the peak status changes.

    Args:
        schedule: The pricing schedule, or None.
        ts: The timestamp to check.

    Returns:
        The next timestamp when the peak status changes, or None if there is no schedule or no change.
    """
    if not schedule:
        return None

    current_peak = is_peak(schedule, ts)
    top_of_hour = datetime.fromtimestamp(ts, timezone.utc).replace(minute=0, second=0, microsecond=0)
    for hour in range(1, 8 * 24 + 1):          # at most 8 days ahead, always at the top of a UTC hour
        next_ts = (top_of_hour + timedelta(hours=hour)).timestamp()
        if is_peak(schedule, next_ts) != current_peak:
            return next_ts
    return None

def upcoming_price(entry: dict, ts: float) -> dict | None:
    """The nearest `price_from` change strictly after the UTC date of `ts`: {"from": date, "price": {...}} or None."""
    date_str = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
    later = sorted(date for date in entry.get("price_from", {}) if date > date_str)
    return {"from": later[0], "price": dict(entry["price_from"][later[0]])} if later else None


def price_status(entry: dict, provider_cfg: dict, ts: float) -> dict:
    """Return what the GUI shows for a model at `ts`: phase (peak / off-peak / None), the price now, the peak price,
    when the phase changes next and the next announced price change."""
    schedule = provider_cfg.get("price_schedule")
    peak = is_peak(schedule, ts)
    return {
        "phase": None if peak is None else "peak" if peak else "off-peak",
        "price": effective_price(entry, provider_cfg, ts),
        "peak_price": base_price(entry, ts),
        "next_change": next_phase_change(schedule, ts),
        "upcoming": upcoming_price(entry, ts),
    }
