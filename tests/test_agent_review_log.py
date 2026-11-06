"""Behaviour tests for foremankit.agent_review_log.

Pins the review-event validation, the last-review-wins aggregation and the formatted summary, so a lead
verdict recorded after a merge or reject is counted once per agent-run branch.
"""

import pytest

from foremankit.agent_review_log import format_reviews, log_review, summarize_reviews
from foremankit.state import CallLog

KEY = "agy/gemini-3.8-flash-high"


def review(**overrides):
    """Build one agent-run review row, overriding any field by name."""
    row = {
        "event": "agent_run_review",
        "id": "run-1",
        "provider": "agy",
        "model": "gemini-3.8-flash-high",
        "verdict": "merged",
        "blockers": 0,
    }
    row.update(overrides)
    return row


# --- log_review ------------------------------------------------------------


def test_an_unknown_verdict_is_rejected(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")

    with pytest.raises(ValueError):
        log_review(log, "run-1", "agy", "gemini-3.8-flash-high", "approved")


def test_negative_blockers_are_rejected(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")

    with pytest.raises(ValueError):
        log_review(log, "run-1", "agy", "gemini-3.8-flash-high", "rejected", blockers=-1)


def test_the_event_is_written_and_parses_back(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")

    event = log_review(log, "run-7", "agy", "gemini-3.8-flash-high", "merged_after_fix", blockers=2)

    record = log.read()[0]
    assert record.pop("ts") > 0
    assert record == event
    assert event["event"] == "agent_run_review"


# --- summarize_reviews -----------------------------------------------------


def test_an_empty_history_gives_an_empty_table():
    assert summarize_reviews([]) == {}


def test_non_review_events_are_ignored():
    assert summarize_reviews([review(event="outcome"), review(event="started")]) == {}


def test_counts_and_clean_rate_are_summed_per_group():
    rows = [review(), review(id="run-2"), review(id="run-3"),
            review(id="run-4", verdict="merged_after_fix"),
            review(id="run-5", verdict="rejected", blockers=4)]

    assert summarize_reviews(rows)[KEY] == {
        "reviews": 5, "merged": 3, "merged_after_fix": 1, "rejected": 1,
        "blockers": 4, "clean_rate": 0.6,
    }


def test_a_later_review_of_the_same_run_replaces_the_earlier_one():
    rows = [review(id="run-1", verdict="merged"), review(id="run-1", verdict="rejected", blockers=1)]

    stats = summarize_reviews(rows)[KEY]

    assert stats == {"reviews": 1, "merged": 0, "merged_after_fix": 0, "rejected": 1,
                     "blockers": 1, "clean_rate": 0.0}


def test_a_group_with_no_clean_merge_has_a_zero_clean_rate():
    rows = [review(verdict="rejected"), review(id="run-2", verdict="merged_after_fix")]

    assert summarize_reviews(rows)[KEY]["clean_rate"] == 0.0


def test_a_missing_model_falls_back_to_a_question_mark():
    row = review()
    del row["model"]

    assert set(summarize_reviews([row])) == {"agy/?"}


def test_two_runners_get_separate_groups():
    rows = [review(), review(id="run-2", provider="claude", model="sonnet")]

    assert set(summarize_reviews(rows)) == {KEY, "claude/sonnet"}


# --- format_reviews --------------------------------------------------------


def test_an_empty_table_shows_a_placeholder_line():
    assert format_reviews({}).splitlines() == ["no agent-run reviews yet"]


def test_one_group_is_rendered_with_its_counts():
    rows = [review(), review(id="run-2"), review(id="run-3"),
            review(id="run-4", verdict="merged_after_fix"),
            review(id="run-5", verdict="rejected", blockers=4)]

    assert format_reviews(summarize_reviews(rows)).splitlines() == [
        f"{KEY}: 5 reviews, clean 60%, fixed 1, rejected 1, blockers 4",
    ]


def test_groups_are_sorted_by_key():
    table = summarize_reviews([review(provider="zeta", model="m"), review(id="run-2", provider="alpha", model="m")])

    lines = format_reviews(table).splitlines()

    assert lines[0].startswith("alpha/m:") and lines[1].startswith("zeta/m:")
