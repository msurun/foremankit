"""Behaviour tests for foremankit.agent_review_log.

Pins the review-event validation, the last-review-wins aggregation and the formatted summary, so a lead
verdict recorded after a merge or reject is counted once per agent-run branch.
"""

import pytest

from foremankit import agent_review_log
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



def model_review(**overrides) -> dict:
    """Build one model review attempt with optional event fields."""
    row = {"event": "agent_run_model_review", "branch": "agent-run/example",
           "reviewer": "codex", "status": "ok", "ts": 10}
    row.update(overrides)
    return row


def test_model_reviews_return_only_summary_fields():
    """Findings and raw model output stay outside the summary."""
    row = model_review(counts_by_severity={"high": 2, "medium": 3, "low": 4},
                       findings=[{"problem": "details"}], raw_text="private output")
    assert agent_review_log.summarize_model_reviews([row]) == [
        {"branch": "agent-run/example", "reviewer": "codex", "status": "ok",
         "counts_by_severity": {"high": 2, "medium": 3, "low": 4}}]


def test_empty_model_review_history_returns_a_list():
    """No attempts produce an empty list."""
    assert agent_review_log.summarize_model_reviews([]) == []


def test_model_reviews_exclude_unrelated_events():
    """Owner verdicts and run outcomes are not model review attempts."""
    rows = [review(), model_review(event="outcome")]
    assert agent_review_log.summarize_model_reviews(rows) == []


@pytest.mark.parametrize("fields, high_count", [({}, 0),
    ({"counts_by_severity": {}}, 0), ({"counts_by_severity": {"high": 2}}, 2)])
def test_model_reviews_fill_missing_severity_counts(fields, high_count):
    """Absent counters default to zero without losing supplied counters."""
    row = model_review()
    row.update(fields)
    expected = {"high": high_count, "medium": 0, "low": 0}
    assert agent_review_log.summarize_model_reviews([row])[0]["counts_by_severity"] == expected


def test_model_reviews_sort_by_timestamp_before_log_position():
    """A later timestamp wins even when its row appears earlier in the log."""
    rows = [model_review(ts=20, reviewer="newer"), model_review(ts=10, reviewer="older")]
    summaries = agent_review_log.summarize_model_reviews(rows)
    assert (summaries[0]["reviewer"], summaries[1]["reviewer"]) == ("newer", "older")


def test_model_review_timestamp_ties_use_latest_log_position():
    """Later log entries win timestamp ties."""
    rows = [model_review(reviewer="first"), model_review(reviewer="last")]
    summaries = agent_review_log.summarize_model_reviews(rows)
    assert (summaries[0]["reviewer"], summaries[1]["reviewer"]) == ("last", "first")


def test_repeated_model_review_attempts_remain_distinct():
    """Identical branch and reviewer pairs are not collapsed."""
    row = model_review()
    summaries = agent_review_log.summarize_model_reviews([row, row])
    assert len(summaries) == 2


@pytest.mark.parametrize("status", ["failed", "skipped", "unparsable"])
def test_unsuccessful_model_reviews_keep_their_status(status):
    """Zero findings must not disguise an unsuccessful attempt as a clean review."""
    row = model_review(status=status)
    assert agent_review_log.summarize_model_reviews([row]) == [
        {"branch": "agent-run/example", "reviewer": "codex", "status": status,
         "counts_by_severity": {"high": 0, "medium": 0, "low": 0}}]
