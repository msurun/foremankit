"""Tests for stitch.py: finding an open code fence, joining a cut answer with its continuation, and reporting seam problems.

A model that stops at its token limit leaves a partial answer behind. These tests pin down how the partial and the
continuation are joined: no duplicated fence line, no duplicated overlap, no invented whitespace, and a clear report
when the seam itself looks wrong."""
from __future__ import annotations

from foremankit.stitch import open_fence, seam_problems, stitch

LONG_PROSE = "The first paragraph of the long answer explains the method in some detail. " * 4


def test_open_fence_detects_a_python_fence_with_an_info_string():
    assert open_fence("Here is the code:\n```python\ndef add(a, b):") == "```python"


def test_open_fence_detects_a_tilde_fence():
    assert open_fence("Example:\n~~~\nprint('half")  == "~~~"


def test_open_fence_keeps_a_four_backtick_block_open_past_an_inner_three_backtick_line():
    text = "````markdown\n```python\nx = 1\n```\n"
    assert open_fence(text) == "````markdown"


def test_open_fence_returns_none_for_a_closed_block():
    assert open_fence("```python\nx = 1\n```\nand then prose") is None


def test_stitch_drops_a_reopened_fence_line():
    partial = "Here is the code:\n```python\ndef add(a, b):\n"
    continuation = "```python\n    return a + b\n```"
    assert stitch(partial, continuation) == partial + "    return a + b\n```"


def test_stitch_removes_an_overlap_when_the_last_line_is_repeated():
    partial = "Step one: open the valve slowly.\nStep two: check the gauge"
    continuation = "Step two: check the gauge and log the reading."
    assert stitch(partial, continuation) == partial + " and log the reading."


def test_stitch_does_not_invent_whitespace_between_a_cut_word_and_its_rest():
    partial = "The configuration is saved to the disk and the conf"
    continuation = "iguration file is then reloaded."
    assert stitch(partial, continuation) == partial + continuation


def test_seam_problems_is_empty_for_a_clean_join():
    joined = LONG_PROSE + "It ends here."
    assert seam_problems(LONG_PROSE, joined) == []


def test_seam_problems_reports_an_open_fence_at_the_seam():
    partial = "Some text before the block.\n```python\nx = 1\n"
    problems = seam_problems(partial, partial)
    assert any("open code block" in problem for problem in problems)


def test_seam_problems_reports_a_repeated_seam():
    seam = LONG_PROSE[-40:]
    joined = LONG_PROSE + seam
    problems = seam_problems(LONG_PROSE, joined)
    assert any("repeated twice" in problem for problem in problems)


def test_seam_problems_reports_a_join_shorter_than_the_partial():
    joined = LONG_PROSE[: len(LONG_PROSE) // 2]
    problems = seam_problems(LONG_PROSE, joined)
    assert any("shorter than the partial" in problem for problem in problems)
