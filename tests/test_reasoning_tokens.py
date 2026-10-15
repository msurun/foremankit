"""Tests that hidden reasoning tokens are measured from provider usage and written to the call log.

Thinking models bill reasoning as output tokens. Some report it in completion_tokens_details, others (Gemini)
only inside total_tokens. These tests check both ways of reading it and the log row the router writes.
"""
from __future__ import annotations

from conftest import chat_json, read_log

from foremankit.providers import normalize_usage

GEMINI = "https://generativelanguage.googleapis.com"


def test_normalize_usage_reads_reasoning_from_completion_tokens_details():
    usage = normalize_usage({"prompt_tokens": 10, "completion_tokens": 20,
                             "completion_tokens_details": {"reasoning_tokens": 15}})
    assert usage["reasoning_tokens"] == 15
    assert usage["completion_tokens"] == 20


def test_normalize_usage_derives_hidden_reasoning_from_total_tokens():
    usage = normalize_usage({"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 68})
    assert usage["reasoning_tokens"] == 59
    assert usage["completion_tokens"] == 60


def test_normalize_usage_reports_no_reasoning_when_provider_gives_no_hint():
    usage = normalize_usage({"prompt_tokens": 10, "completion_tokens": 20})
    assert usage == {"prompt_tokens": 10, "completion_tokens": 20, "cache_hit": 0, "reasoning_tokens": 0}


def test_call_log_row_records_reasoning_tokens_and_request_extra(monkeypatch, make_router, mock, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json("OK", prompt=8, completion=1, total_tokens=68))
    make_router().delegate("t", kind="long_context", max_output_tokens=64)
    [row] = [r for r in read_log(tmp_path) if r.get("event") == "call" and r.get("success") is True]
    assert row["reasoning_tokens"] == 59
    assert row["extra"]["reasoning_effort"] == "low"
