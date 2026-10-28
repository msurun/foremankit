"""Tests for the served-model check: the model id a router reports is parsed, can be rejected, and is shown in the header.
"""

import types

import httpx
import pytest

from foremankit import caller, formatting, providers


def _candidate(cfg):
    """A minimal stand-in for a Candidate: only the provider config is read."""
    return types.SimpleNamespace(provider=types.SimpleNamespace(cfg=cfg))


def test_parse_chat_reports_the_model_the_router_served():
    """The served model id from the response body is kept as served_model."""
    response = httpx.Response(200, json={
        "model": "nvidia/x:free",
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
    assert providers._parse_chat(response)["served_model"] == "nvidia/x:free"


def test_rejected_served_model_raises_transient_provider_error():
    """A served model matching a reject pattern raises a transient ProviderError."""
    candidate = _candidate({"reject_served_models": ["content-safety", "guard"]})
    reply = {"served_model": "nvidia/nemotron-3.5-content-safety:free"}
    with pytest.raises(providers.ProviderError) as info:
        caller.reject_served_model(candidate, reply)
    assert info.value.transient is True


def test_served_model_not_matching_any_pattern_is_accepted():
    """A served model matching no reject pattern passes without error."""
    candidate = _candidate({"reject_served_models": ["content-safety", "guard"]})
    assert caller.reject_served_model(candidate, {"served_model": "nvidia/nemotron-3-ultra-550b-a55b:free"}) is None


def test_missing_served_model_is_accepted():
    """A reply without a served model is never rejected."""
    candidate = _candidate({"reject_served_models": ["content-safety"]})
    assert caller.reject_served_model(candidate, {"served_model": None}) is None


def test_provider_without_reject_list_accepts_any_served_model():
    """A provider with no reject_served_models config never rejects a reply."""
    candidate = _candidate({})
    assert caller.reject_served_model(candidate, {"served_model": "nvidia/nemotron-3.5-content-safety:free"}) is None


def _result(served_model):
    """A result dict with every key the header reads, for the given served model."""
    return {"provider": "openrouter", "model": "openrouter/free", "endpoint": None, "latency_s": 1.5,
            "dry_run": False, "tokens_in": 10, "cache_hit": 0, "tokens_out": 5, "cost_usd": 0.0,
            "served_model": served_model}


def test_header_names_the_served_model_when_it_differs():
    """The header appends the served model when it differs from the requested one."""
    header = formatting._result_header(_result("google/gemma-4-31b-it:free"))
    assert "(served by google/gemma-4-31b-it:free)" in header


def test_header_omits_served_by_when_served_model_is_none():
    """The header has no 'served by' part when no served model is known."""
    assert "served by" not in formatting._result_header(_result(None))
