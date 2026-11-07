"""Vendor identity and reviewer independence are shared by provider and agent routing.

These pure tests keep unknown identities from silently allowing self-review.
"""
from pathlib import Path

import pytest

from foremankit.config import load_config
from foremankit.vendors import (
    DEFAULT_MODEL_VENDORS,
    different_vendor,
    pick_reviewer,
    runner_vendor,
    vendor_of_model,
    vendor_of_provider,
)


def test_provider_uses_its_configured_vendor():
    """Aliases of one company share a vendor."""
    assert vendor_of_provider("agy", {"agy": {"vendor": "google"}}) == "google"


def test_provider_defaults_to_its_own_name():
    """Older provider configurations remain usable."""
    assert vendor_of_provider("custom", {"custom": {}}) == "custom"


@pytest.mark.parametrize("model,vendor", list(DEFAULT_MODEL_VENDORS.items()) + [("other", "unknown")])
def test_model_substrings_resolve_case_insensitively(model, vendor):
    """Every default family can be identified inside a model ID."""
    assert vendor_of_model(f"host/{model.upper()}-latest") == vendor


def test_model_table_keeps_first_matching_substring():
    """Configuration order resolves overlapping model families."""
    assert vendor_of_model("gpt-custom", {"custom": "first", "gpt": "second"}) == "first"


@pytest.mark.parametrize("runner,model,vendor", [
    ("agy", "claude", "google"), ("codex", "gemini", "openai"),
    ("opencode", "opencode/deepseek-v4.1-flash-free", "deepseek"),
    ("opencode", "gpt-5", "openai"), ("opencode", "gemini-flash", "google"),
    ("opencode", "claude-sonnet", "anthropic"), ("opencode", "other", "unknown"),
    ("other", "gpt-5", "unknown"),
])
def test_runner_vendor_uses_the_runner_or_selected_model(runner, model, vendor):
    """Only OpenCode derives identity from its model."""
    assert runner_vendor(runner, model) == vendor


@pytest.mark.parametrize("author,candidate,expected", [
    ("google", "openai", True), ("google", "google", False),
    ("unknown", "openai", False), ("google", "unknown", False),
    ("unknown", "unknown", False), ("", "google", False),
])
def test_independence_requires_known_distinct_vendors(author, candidate, expected):
    """Unknown identity must fail closed in either position."""
    assert different_vendor(author, candidate) is expected


def test_reviewer_selection_keeps_candidate_order():
    """The first independent candidate wins."""
    vendors = {"agy": "google", "gemini": "google", "codex": "openai", "claude": "anthropic"}
    assert pick_reviewer("agy", ["gemini", "codex", "claude"], vendors) == "codex"


def test_reviewer_selection_returns_none_for_same_vendor():
    """Same-company aliases cannot verify each other."""
    assert pick_reviewer("agy", ["gemini"], {"agy": "google", "gemini": "google"}) is None


def test_reviewer_selection_refuses_an_unknown_author():
    """A missing author identity cannot establish independence."""
    assert pick_reviewer("other", ["codex"], {"codex": "openai"}) is None


@pytest.mark.parametrize("name,expected", [
    ("deepseek", "deepseek"), ("gemini", "google"), ("gemini_free", "google"),
    ("agy", "google"), ("groq", "groq"), ("openrouter", "openrouter"),
    ("mistral", "mistral"), ("cerebras", "cerebras"), ("kaggle", "local"), ("llm", "local"),
])
def test_real_configuration_resolves_each_provider(name, expected):
    """Every production provider declares its intended vendor."""
    configuration = load_config(Path(__file__).resolve().parents[1] / "config")
    assert vendor_of_provider(name, configuration["providers"]) == expected


def test_real_routing_table_matches_the_default_vendor_table():
    """Agent callers can use either the configured table or pure defaults."""
    configuration = load_config(Path(__file__).resolve().parents[1] / "config")
    assert configuration["routing"]["vendors"] == DEFAULT_MODEL_VENDORS
