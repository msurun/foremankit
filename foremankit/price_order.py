"""Re-order the paid entries of a routing chain by their current estimated cost.

The chain keeps every entry: fixed entries (unknown, LLM, free or unpriced) stay at their
exact index, while paid entries are sorted by cost and placed in the remaining slots. Prices
within `tolerance` of each other fall in the same bucket, so the configured order survives
for near-equal prices.
"""

# Cost is in USD for one call: est_in tokens at the miss price plus max_out tokens at the
# output price, both priced per 1M tokens. This constant converts tokens to millions.
TOKENS_PER_MILLION = 1_000_000


def estimated_cost(
    provider: object, alias: str | None, est_in: int, max_out: int
) -> float | None:
    """Return the estimated USD cost of one call, or None when the model is not priced.

    `provider` is the object stored in the providers mapping; `alias` may be None.
    Returns None when the provider is fixed (unknown attributes), the alias is unknown
    (KeyError from model()) or the price is missing the numeric keys.
    """
    if _is_fixed_provider(provider):
        return None
    price = _lookup_price(provider, alias)
    if price is None or not _has_numeric_prices(price):
        return None
    return _cost_from_price(price, est_in, max_out)


def order_by_price(
    chain: list[str],
    providers: dict,
    tokens: tuple[int, int],
    tolerance: float,
) -> list[str]:
    """Return a new list where paid entries are sorted by cost within the paid slots.

    Fixed entries keep their index; paid entries fill the rest cheapest group first,
    and keep the configured order inside a group. The result is a permutation of
    `chain`. `tokens` is (est_in, max_out); `tolerance` <= 0 sorts by the exact cost.
    """
    est_in, max_out = tokens
    if len(chain) <= 1:
        return list(chain)

    prices = [_entry_cost(entry, providers, est_in, max_out) for entry in chain]
    paid_indexes = _paid_indexes(prices)
    paid_entries = [chain[index] for index in paid_indexes]
    paid_costs = [prices[index] for index in paid_indexes]
    ordered_entries = _sort_by_bucket(paid_entries, paid_costs, tolerance)

    result = list(chain)
    for slot, entry in zip(paid_indexes, ordered_entries):
        result[slot] = entry
    return result


def _is_fixed_provider(provider: object) -> bool:
    """Return True when the provider is unknown, an LLM, free or missing its flags."""
    if not hasattr(provider, "is_llm") or not hasattr(provider, "free"):
        return True
    return bool(provider.is_llm) or bool(provider.free)


def _lookup_price(provider: object, alias: str | None) -> dict | None:
    """Call provider.model(alias) and return its price dict, or None when it raises KeyError.

    model() returns (alias, model_id, price); an unknown alias raises KeyError.
    """
    if not hasattr(provider, "model"):
        return None
    try:
        _alias, _model_id, price = provider.model(alias)
    except KeyError:
        return None
    return price


def _has_numeric_prices(price: dict) -> bool:
    """Return True when the price dict has numeric `input_miss` and `output` entries."""
    input_miss = price.get("input_miss")
    output = price.get("output")
    return _is_number(input_miss) and _is_number(output)


def _is_number(value: object) -> bool:
    """Return True when `value` is an int or float (booleans are not prices)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cost_from_price(price: dict, est_in: int, max_out: int) -> float:
    """Return est_in*input_miss/1e6 + max_out*output/1e6 for the given token counts."""
    input_cost = est_in * price["input_miss"] / TOKENS_PER_MILLION
    output_cost = max_out * price["output"] / TOKENS_PER_MILLION
    return input_cost + output_cost


def _entry_cost(entry: str, providers: dict, est_in: int, max_out: int) -> float | None:
    """Return the estimated cost of one chain entry, or None when it is fixed."""
    provider_name, _, alias = str(entry).partition(":")
    provider_name = provider_name.partition("@")[0]
    provider = providers.get(provider_name)
    if provider is None:
        return None
    return estimated_cost(provider, alias or None, est_in, max_out)


def _paid_indexes(prices: list[float | None]) -> list[int]:
    """Return the indexes whose entry has a numeric cost (the paid slots)."""
    return [index for index, cost in enumerate(prices) if cost is not None]


def _sort_by_bucket(entries: list[str], costs: list[float], tolerance: float) -> list[str]:
    """Return `entries` cheapest group first; costs within `tolerance` of the group's cheapest form one group.

    Inside a group the configured order is kept. Groups start at the cheapest cost, so a 10 % gap is never split
    by an arbitrary boundary (a fixed log grid would).
    """
    by_cost = sorted(range(len(entries)), key=lambda position: costs[position])
    group_of: dict[int, int] = {}
    group, floor = 0, None
    for position in by_cost:
        if floor is None or costs[position] > floor * (1 + max(tolerance, 0.0)):
            group, floor = group + 1, costs[position]
        group_of[position] = group
    ordered = sorted(range(len(entries)), key=lambda position: (group_of[position], position))
    return [entries[position] for position in ordered]
