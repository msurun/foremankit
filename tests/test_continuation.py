"""Tests for continuation.py through Router.delegate: a cut-off answer is continued by the same model and the pieces are
joined into one result, with summed usage, one success row in the call log, and an honest INCOMPLETE note when the
continuations do not finish the answer."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import chat_json, read_log

from foremankit import budget
from foremankit.formatting import format_result

GROQ = "https://api.groq.com"
DEEPSEEK = "https://api.deepseek.com"
FIRST_PART = "The first step is to open the valve and check the gauge before"
SECOND_PART = " the pump starts, then log the pressure."


def cut(text: str, prompt: int = 100, completion: int = 40) -> dict:
    """A reply that stopped at its token limit (finish_reason length) with plain prose."""
    return {"choices": [{"message": {"content": text}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def saved_text(result: dict) -> str:
    """The text of the answer file, without its front matter."""
    return Path(result["path"]).read_text(encoding="utf-8").split("---\n\n", 1)[1]


def test_a_cut_off_answer_is_continued_into_one_stitched_text(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART), chat_json(SECOND_PART))
    result = make_router().delegate("explain", kind="quick")
    assert saved_text(result) == FIRST_PART + SECOND_PART
    assert result["truncated"] is False and "INCOMPLETE" not in format_result(result)


def test_the_continuation_request_carries_the_partial_answer_as_an_assistant_turn(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART), chat_json(SECOND_PART))
    make_router().delegate("explain", kind="quick")
    first, second = mock.bodies[0]["messages"], mock.bodies[1]["messages"]
    assert second[:len(first)] == first
    assert second[-2] == {"role": "assistant", "content": FIRST_PART}
    assert second[-1]["role"] == "user" and "Continue exactly where you stopped" in second[-1]["content"]


def test_the_tokens_and_cost_of_both_rounds_are_summed(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    mock.on(DEEPSEEK, "/chat/completions", cut(FIRST_PART, prompt=100, completion=40),
            chat_json(SECOND_PART, prompt=150, completion=30))
    router = make_router()
    result = router.delegate("explain", kind="data_script")
    price = router.providers["deepseek"].cfg["models"]["flash"]["price"]
    first = {"prompt_tokens": 100, "completion_tokens": 40, "cache_hit": 0}
    second = {"prompt_tokens": 150, "completion_tokens": 30, "cache_hit": 0}
    assert result["tokens_in"] == 250 and result["tokens_out"] == 70
    assert result["cost_usd"] == pytest.approx(budget.cost_usd(price, first) + budget.cost_usd(price, second))


def test_the_call_log_has_one_success_row_that_counts_one_continuation(monkeypatch, make_router, mock, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART), chat_json(SECOND_PART))
    make_router().delegate("explain", kind="quick")
    successes = [row for row in read_log(tmp_path) if row.get("success") is True]
    assert len(successes) == 1 and successes[0]["continuations"] == 1 and successes[0]["truncated"] is False


def test_with_no_continuations_allowed_the_cut_off_answer_stays_truncated(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART), chat_json(SECOND_PART))
    result = make_router(cfg).delegate("explain", kind="quick")
    assert len(mock.bodies) == 1 and result["truncated"] is True
    assert saved_text(result) == FIRST_PART


def test_when_every_reply_is_cut_off_the_result_stays_truncated_and_says_incomplete(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART), cut(SECOND_PART), cut(" and the valve stays shut."))
    result = make_router().delegate("explain", kind="quick")
    assert len(mock.bodies) == 3 and result["truncated"] is True
    assert "INCOMPLETE" in format_result(result)
