"""Behaviour tests for foremankit.passrates.

These tests pin the pass@k summary math and the human-readable formatting so
aggregation bugs (filtering, rounding, ordering) fail loudly here.
"""

import pytest

from foremankit.passrates import format_pass_rates, summarize_pass_rates

BULK_CODE_KEY = "bulk_code/deepseek/flash"


def outcome(**overrides):
    """Build one delegation outcome row, overriding any field by name."""
    row = {
        "event": "outcome",
        "ts": 100,
        "kind": "bulk_code",
        "provider": "deepseek",
        "model": "flash",
        "verified": True,
        "accepted": True,
        "fix_rounds": 0,
    }
    row.update(overrides)
    return row


# --- summarize_pass_rates -------------------------------------------------


def test_an_empty_history_gives_an_empty_table():
    table = summarize_pass_rates([])

    assert table == {}


def test_unverified_rows_are_ignored():
    table = summarize_pass_rates([outcome(verified=False)])

    assert table == {}


def test_non_outcome_events_are_ignored():
    table = summarize_pass_rates([outcome(event="started")])

    assert table == {}


def test_rows_before_the_cutoff_are_dropped():
    table = summarize_pass_rates([outcome(ts=50)], cutoff=100)

    assert table == {}


def test_rows_at_the_cutoff_are_kept():
    table = summarize_pass_rates([outcome(ts=100)], cutoff=100)

    assert table[BULK_CODE_KEY]["delegations"] == 1


def test_mixed_outcomes_give_expected_shares():
    rows = [
        outcome(fix_rounds=0),
        outcome(fix_rounds=0),
        outcome(fix_rounds=2),
        outcome(accepted=False, fix_rounds=0),
    ]

    stats = summarize_pass_rates(rows)[BULK_CODE_KEY]

    assert stats == {"delegations": 4, "pass_at_1": 0.5, "pass_at_3": 0.75}


def test_a_slow_fix_counts_for_neither_pass_rate():
    rows = [outcome(accepted=True, fix_rounds=3)]

    stats = summarize_pass_rates(rows)[BULK_CODE_KEY]

    assert stats == {"delegations": 1, "pass_at_1": 0.0, "pass_at_3": 0.0}


def test_different_kinds_and_models_get_separate_keys():
    rows = [
        outcome(kind="debug", provider="openai", model="pro"),
        outcome(provider="deepseek", model="flash"),
    ]

    table = summarize_pass_rates(rows)

    assert set(table) == {"debug/openai/pro", BULK_CODE_KEY}


def test_shares_are_rounded_to_three_decimals():
    rows = [outcome(fix_rounds=0)] + [outcome(accepted=False) for _ in range(2)]

    stats = summarize_pass_rates(rows)[BULK_CODE_KEY]

    assert stats["pass_at_1"] == 0.333


# --- format_pass_rates ----------------------------------------------------


def test_an_empty_table_shows_a_placeholder_line():
    lines = format_pass_rates({}).splitlines()

    assert lines == ["no verified delegations yet"]


def test_lines_show_exact_text_and_counts():
    table = {BULK_CODE_KEY: {"delegations": 4, "pass_at_1": 0.5, "pass_at_3": 0.75}}

    lines = format_pass_rates(table).splitlines()

    assert lines == [f"{BULK_CODE_KEY}: pass@1 50%, pass@3 75% (4 checked)"]


def test_lines_are_sorted_by_pass_at_1_descending():
    table = {
        "a/x/y": {"delegations": 1, "pass_at_1": 0.2, "pass_at_3": 0.2},
        "b/x/y": {"delegations": 1, "pass_at_1": 0.9, "pass_at_3": 0.9},
    }

    lines = format_pass_rates(table).splitlines()

    assert lines[0].startswith("b/x/y")
    assert lines[1].startswith("a/x/y")


def test_ties_are_broken_by_key():
    table = {
        "zeta/x/y": {"delegations": 1, "pass_at_1": 0.5, "pass_at_3": 0.5},
        "alpha/x/y": {"delegations": 1, "pass_at_1": 0.5, "pass_at_3": 0.5},
    }

    lines = format_pass_rates(table).splitlines()

    assert lines == [
        "alpha/x/y: pass@1 50%, pass@3 50% (1 checked)",
        "zeta/x/y: pass@1 50%, pass@3 50% (1 checked)",
    ]


def test_min_delegations_filters_thin_entries():
    table = {
        "a/x/y": {"delegations": 1, "pass_at_1": 0.9, "pass_at_3": 0.9},
        "b/x/y": {"delegations": 5, "pass_at_1": 0.4, "pass_at_3": 0.4},
    }

    lines = format_pass_rates(table, min_delegations=3).splitlines()

    assert lines == ["b/x/y: pass@1 40%, pass@3 40% (5 checked)"]
