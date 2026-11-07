"""Vendor exclusions apply before chain expansion and preserve other vendors.

Exercise the real chain walker without calling external models.
"""
from unittest.mock import Mock

import pytest

from foremankit.chain import ChainRun
from foremankit.records import TaskRequest


@pytest.mark.parametrize("author,vendor", [("agy", "google"), ("gemini_free", "google")])
def test_google_authors_exclude_both_gemini_providers(author, vendor, make_router, cfg):
    """Provider aliases cannot bypass company exclusion."""
    cfg["routing"]["chains"]["review"] = ["gemini:flash", "gemini_free:flash", "deepseek:flash"]
    chain = ChainRun(make_router(), TaskRequest("review", "review"), (author, f"vendor:{vendor}"))
    assert list(chain._entries()) == ["deepseek:flash"]


def test_local_vendor_excludes_explicit_and_expanded_nodes(make_router, cfg):
    """The pool must be excluded before probing or expanding its nodes."""
    cfg["routing"]["chains"]["review"] = ["llm", "llm@other", "deepseek:flash"]
    router = make_router()
    router.selector.llm_entries = Mock(return_value=["llm@local"])
    chain = ChainRun(router, TaskRequest("review", "review"), ("llm@local", "vendor:local"))
    assert list(chain._entries()) == ["deepseek:flash"]
    router.selector.llm_entries.assert_not_called()


def test_unknown_vendor_excludes_every_source(make_router, cfg):
    """Unknown authors cannot receive a purportedly independent review."""
    chain = ChainRun(make_router(), TaskRequest("review", "review"), ("vendor:unknown",))
    assert list(chain._entries()) == []
