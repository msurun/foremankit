"""Tests of the price ordering of a routing chain (foremankit/price_order.py).

They pin the tolerance rule, the fixed places of free and local entries and the cost arithmetic, and check that
ChainRun applies the ordering only to the kinds listed under `price_order`.
"""
from dataclasses import dataclass, field

import pytest

from foremankit.price_order import estimated_cost, order_by_price

TOKENS = (1000, 2000)          # est_in, max_out


@dataclass
class FakeProvider:
    """The three things the orderer reads from a provider; prices are USD per 1M tokens."""
    free: bool = False
    is_llm: bool = False
    prices: dict = field(default_factory=dict)

    def model(self, alias):
        """(alias, id, price) like the real provider; an unknown alias raises KeyError."""
        alias = alias or "m"
        return alias, f"id-{alias}", self.prices[alias]


def paid(output: float, alias: str = "m") -> FakeProvider:
    """A paid provider whose only model has the given output price (input is a tenth of it)."""
    return FakeProvider(prices={alias: {"input_miss": output / 10, "output": output}})


def test_the_cheaper_paid_entry_goes_first():
    """Two paid entries are sorted by the estimated cost of the call."""
    providers = {"dear": paid(10.0), "cheap": paid(1.0)}
    assert order_by_price(["dear", "cheap"], providers, TOKENS, 0.2) == ["cheap", "dear"]


def test_prices_within_the_tolerance_keep_the_configured_order():
    """A 10 % price gap with a 20 % tolerance does not move anything."""
    providers = {"a": paid(1.1), "b": paid(1.0)}
    assert order_by_price(["a", "b"], providers, TOKENS, 0.2) == ["a", "b"]


def test_tolerance_zero_sorts_exactly():
    """Without tolerance even a small gap decides."""
    providers = {"a": paid(1.1), "b": paid(1.0)}
    assert order_by_price(["a", "b"], providers, TOKENS, 0.0) == ["b", "a"]


def test_free_and_local_entries_keep_their_place():
    """Only the paid slots are re-filled; free and local entries stay at their index."""
    providers = {"dear": paid(10.0), "cheap": paid(1.0), "groq": FakeProvider(free=True), "llm": FakeProvider(is_llm=True)}
    chain = ["dear", "groq", "llm", "cheap"]
    assert order_by_price(chain, providers, TOKENS, 0.2) == ["cheap", "groq", "llm", "dear"]


def test_unknown_provider_and_unknown_alias_stay_where_they_are():
    """An entry that cannot be priced is never moved."""
    providers = {"dear": paid(10.0), "cheap": paid(1.0)}
    chain = ["dear", "ghost", "cheap:nope", "cheap"]
    assert order_by_price(chain, providers, TOKENS, 0.2) == ["cheap", "ghost", "cheap:nope", "dear"]


def test_the_result_is_a_permutation_and_the_input_is_untouched():
    """Nothing is dropped or duplicated, and the caller's list is not modified."""
    providers = {"dear": paid(10.0), "cheap": paid(1.0), "mid": paid(3.0)}
    chain = ["dear", "mid", "cheap"]
    result = order_by_price(chain, providers, TOKENS, 0.2)
    assert sorted(result) == sorted(chain) and chain == ["dear", "mid", "cheap"]


@pytest.mark.parametrize("chain", [[], ["cheap"]])
def test_short_chains_come_back_unchanged(chain):
    """An empty or one-entry chain has nothing to order."""
    assert order_by_price(chain, {"cheap": paid(1.0)}, TOKENS, 0.2) == chain


def test_the_estimate_adds_input_and_output_cost():
    """1000 tokens in at 1.0 and 2000 tokens out at 2.0 per 1M tokens cost 0.005 USD."""
    provider = FakeProvider(prices={"m": {"input_miss": 1.0, "output": 2.0}})
    assert estimated_cost(provider, "m", 1000, 2000) == pytest.approx(0.005)


def test_an_off_peak_half_price_lets_a_later_entry_overtake():
    """DeepSeek-like peak 4.0 loses to a 3.0 entry, the same model at the off-peak 2.0 wins."""
    chain = ["deepseek", "gemini"]
    gemini = paid(3.0)
    assert order_by_price(chain, {"deepseek": paid(4.0), "gemini": gemini}, TOKENS, 0.2) == ["gemini", "deepseek"]
    assert order_by_price(chain, {"deepseek": paid(2.0), "gemini": gemini}, TOKENS, 0.2) == ["deepseek", "gemini"]


# ---------------------------------------------------------------- applied by ChainRun

def chain_for(make_router, cfg, kind, **setting):
    """The chain a ChainRun walks for `kind` when chains.bulk_code is flash-then-lite and price_order is `setting`."""
    from foremankit.chain import ChainRun
    from foremankit.records import TaskRequest
    cfg["routing"]["chains"]["bulk_code"] = ["gemini:flash", "gemini:lite31"]
    cfg["routing"]["chains"]["hard_code"] = ["gemini:flash", "gemini:lite31"]
    cfg["routing"]["price_order"] = setting
    return ChainRun(make_router(cfg), TaskRequest(task="x", kind=kind)).chain


def test_a_listed_kind_is_tried_cheapest_first(make_router, cfg):
    """With price_order on, the lite model overtakes the dearer flash although it is configured second."""
    assert chain_for(make_router, cfg, "bulk_code", kinds=["bulk_code"], tolerance=0.2) == ["gemini:lite31", "gemini:flash"]


def test_a_kind_that_is_not_listed_keeps_the_configured_order(make_router, cfg):
    """hard_code is not under price_order, so quality order stays."""
    chain = chain_for(make_router, cfg, "hard_code", kinds=["bulk_code"], tolerance=0.2)
    assert chain == ["gemini:flash", "gemini:lite31"]


def test_without_a_price_order_section_nothing_changes(make_router, cfg):
    """An old routing.yaml has no price_order: the chain is exactly the configured one."""
    assert chain_for(make_router, cfg, "bulk_code") == ["gemini:flash", "gemini:lite31"]
