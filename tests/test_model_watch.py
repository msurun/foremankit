"""Tests for the pure model-name helpers in foremankit.model_watch.

These tests describe the behaviour of candidate suggestions, stale-model detection and the
comment-preserving YAML id rewrite. Everything here runs on strings and dicts: no file, no network.
"""

import pytest

from foremankit import model_watch as mw


# --------------------------------------------------------------------------- find_candidates


def test_a_higher_version_of_the_same_family_is_suggested():
    """deepseek-v4-pro is offered deepseek-v5-pro, so the later version is a candidate."""
    candidates = mw.find_candidates(
        "deepseek-v4-pro",
        ["deepseek-flash", "deepseek-v5-pro", "gemini-3-pro"],
    )
    assert candidates == ["deepseek-v5-pro"]


def test_a_candidate_with_a_tier_the_old_model_lacks_is_rejected():
    """deepseek-v5-pro-mini adds the 'mini' tier, which the old model does not have."""
    candidates = mw.find_candidates(
        "deepseek-v4-pro",
        ["deepseek-flash", "deepseek-v5-pro", "deepseek-v5-pro-mini", "gemini-3-pro"],
    )
    assert "deepseek-v5-pro-mini" not in candidates


def test_a_preview_variant_is_suggested_alongside_the_plain_model():
    """'preview' is optional, so both the preview and the plain successor qualify."""
    candidates = mw.find_candidates(
        "gemini-3.1-pro-preview",
        ["gemini-3.2-pro-preview", "gemini-3.1-pro"],
    )
    assert sorted(candidates) == ["gemini-3.1-pro", "gemini-3.2-pro-preview"]


def test_no_candidate_yields_an_empty_list():
    """An unrelated family leaves nothing to suggest."""
    assert mw.find_candidates("deepseek-v4-pro", ["gemini-3-pro", "llama-4"]) == []


def test_the_old_model_id_is_never_its_own_candidate():
    """A model cannot be suggested as a replacement for itself."""
    assert "deepseek-v4-pro" not in mw.find_candidates("deepseek-v4-pro", ["deepseek-v4-pro"])


def test_no_more_than_three_candidates_are_returned():
    """The suggestion list is capped at three, best first."""
    offered = [f"deepseek-v5-pro-{suffix}" for suffix in ("a", "b", "c", "d", "e")]
    assert len(mw.find_candidates("deepseek-v4-pro", offered)) == 3


def test_candidates_that_share_more_tokens_come_first():
    """deepseek-v5-pro shares family, version and tier; deepseek-v6-pro only family and tier."""
    candidates = mw.find_candidates(
        "deepseek-v4-pro",
        ["deepseek-pro", "deepseek-v6-pro", "deepseek-v5-pro"],
    )
    assert candidates[0] == "deepseek-v5-pro"


# ----------------------------------------------------------------------------- stale_models


def test_a_model_missing_from_offered_is_reported_as_stale():
    """The alias is offered under a different id, so the configured id is stale."""
    providers = {"deepseek": {"models": {"chat": {"id": "deepseek-v4-pro", "price": 1.0}}}}
    offered = {"deepseek": ["deepseek-v5-pro"]}
    stale = mw.stale_models(providers, offered)
    assert [entry["alias"] for entry in stale] == ["chat"]
    assert stale[0]["candidates"] == ["deepseek-v5-pro"]


def test_nothing_is_reported_when_every_id_is_offered():
    """Every configured id appears in the live list, so there is nothing to warn about."""
    providers = {"deepseek": {"models": {"chat": {"id": "deepseek-v4-pro", "price": 1.0}}}}
    offered = {"deepseek": ["deepseek-v4-pro"]}
    assert mw.stale_models(providers, offered) == []


@pytest.mark.parametrize(
    "offered",
    [
        {},
        {"deepseek": []},
        {"deepseek": ["deepseek-v4-pro"]},
    ],
    ids=["provider_absent", "empty_offered_list", "provider_is_llm"],
)
def test_providers_without_usable_offered_data_are_skipped(offered):
    """Absent, empty and 'llm'-typed providers are not checked for stale models."""
    providers = {
        "deepseek": {"type": "llm", "models": {"chat": {"id": "deepseek-v3", "price": 1.0}}},
    }
    assert mw.stale_models(providers, offered) == []


# ------------------------------------------------------------------------- rewrite_model_id


FRAGMENT = (
    "providers:\n"
    "  deepseek:\n"
    "    models:\n"
    "      chat:\n"
    "        id: deepseek-v4-pro            # = DeepSeek-V4-Pro-0813\n"
    "        price: 1.0\n"
    "      # the reasoner is internal only\n"
    "      reasoner:\n"
    "        id: deepseek-v4-pro\n"
    "        price: 2.0\n"
    "  other:\n"
    "    models:\n"
    "      chat:\n"
    "        id: deepseek-v4-pro\n"
)


def test_rewrite_changes_only_the_targeted_id():
    """The 'other' provider has the same alias and the same id, and must stay untouched."""
    rewritten = mw.rewrite_model_id(FRAGMENT, "deepseek", "chat", "deepseek-v5-pro")
    lines = rewritten.splitlines()
    assert "        id: deepseek-v5-pro            # = DeepSeek-V4-Pro-0813" in lines


def test_rewrite_keeps_the_price_line_and_sibling_entries():
    """Nothing but the targeted value changes: the price is still 1.0 and the reasoner unchanged."""
    rewritten = mw.rewrite_model_id(FRAGMENT, "deepseek", "chat", "deepseek-v5-pro")
    assert "        price: 1.0" in rewritten.splitlines()
    assert rewritten.count("id: deepseek-v4-pro") == 2


def test_rewrite_keeps_windows_line_endings():
    """A CRLF document stays CRLF after the rewrite."""
    crlf_fragment = FRAGMENT.replace("\n", "\r\n")
    rewritten = mw.rewrite_model_id(crlf_fragment, "deepseek", "chat", "deepseek-v5-pro")
    assert "\r\n" in rewritten
    assert "\n" not in rewritten.replace("\r\n", "")      # no bare LF left: every ending is still CRLF
    assert rewritten == crlf_fragment.replace("deepseek-v4-pro", "deepseek-v5-pro", 1)


def test_rewrite_keeps_the_quote_style_of_the_id():
    """A double-quoted id is still double-quoted after the rewrite."""
    fragment = 'providers:\n  deepseek:\n    models:\n      chat:\n        id: "deepseek-v4-pro"\n'
    rewritten = mw.rewrite_model_id(fragment, "deepseek", "chat", "deepseek-v5-pro")
    assert '        id: "deepseek-v5-pro"' in rewritten.splitlines()


def test_rewrite_keeps_the_final_newline_state():
    """A document without a trailing newline does not gain one."""
    fragment = (
        "providers:\n  deepseek:\n    models:\n      chat:\n        id: deepseek-v4-pro"
    )
    rewritten = mw.rewrite_model_id(fragment, "deepseek", "chat", "deepseek-v5-pro")
    assert not rewritten.endswith("\n")


def test_rewrite_raises_key_error_for_a_missing_alias():
    """An unknown alias is a loud failure naming provider and alias."""
    with pytest.raises(KeyError, match=r"deepseek\.missing"):
        mw.rewrite_model_id(FRAGMENT, "deepseek", "missing", "deepseek-v5-pro")
