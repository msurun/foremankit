"""Tests for the seam guard: stitch.seam_is_broken on plain strings, and complete_if_cut through Router.delegate.

A continuation that says it finished can still leave a code fence open or repeat the seam text. The guard marks
such a round as cut (finish_reason length) so another round runs, and an answer that stays broken is reported
as INCOMPLETE. These tests use the same Groq mock and cut-off replies as test_continuation.py."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import chat_json, read_log

from foremankit.formatting import format_result
from foremankit.records import RouterError
from foremankit.stitch import seam_is_broken

GROQ = "https://api.groq.com"
PARTIAL = "Open the valve slowly and check the gauge reading before the pump starts."
FENCE_PARTIAL = "Here is the code:\n```python\ndef add(a, b):\n    return a +"
FENCE_MIDDLE = " b\n\ndef sub(a, b):\n    return a - b\n"
FENCE_CLOSE = "\n```\n"


def cut(text: str) -> dict:
    """A reply that stopped at its token limit (finish_reason length)."""
    return {"choices": [{"message": {"content": text}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40}}


def test_a_clean_join_is_not_reported_as_broken():
    joined = PARTIAL + " Then the pump starts and the pressure is logged."
    assert seam_is_broken(PARTIAL, joined) is False


def test_a_join_that_ends_inside_an_open_fence_is_broken():
    joined = FENCE_PARTIAL + FENCE_MIDDLE
    assert seam_is_broken(FENCE_PARTIAL, joined) is True


def test_a_join_that_repeats_the_forty_character_seam_twice_is_broken():
    seam = PARTIAL[-40:]
    assert seam_is_broken(PARTIAL, PARTIAL + seam + seam) is True


def test_a_partial_shorter_than_the_seam_window_is_never_a_repeated_seam():
    short = "short"
    assert seam_is_broken(short, short + short) is False


def test_a_cut_inside_a_fence_that_the_next_stop_leaves_open_gets_a_third_round(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", cut(FENCE_PARTIAL), chat_json(FENCE_MIDDLE), chat_json(FENCE_CLOSE))
    result = make_router().delegate("explain", kind="quick")
    text = Path(result["path"]).read_text(encoding="utf-8").split("---\n\n", 1)[1]
    assert len(mock.bodies) == 3
    assert text.count("```") == 2 and text.rstrip().endswith("```")


def test_the_fence_case_logs_one_success_row_with_two_continuations(monkeypatch, make_router, mock, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", cut(FENCE_PARTIAL), chat_json(FENCE_MIDDLE), chat_json(FENCE_CLOSE))
    make_router().delegate("explain", kind="quick")
    successes = [row for row in read_log(tmp_path) if row.get("success") is True]
    assert len(successes) == 1 and successes[0]["continuations"] == 2


def test_with_one_round_allowed_a_fence_left_open_is_refused_not_marked_incomplete(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 1
    mock.on(GROQ, "/chat/completions", cut(FENCE_PARTIAL), chat_json(FENCE_MIDDLE), chat_json(FENCE_CLOSE))
    with pytest.raises(RouterError, match="cut off inside a code block"):
        make_router(cfg).delegate("explain", kind="quick")
    assert len(mock.bodies) == 2
