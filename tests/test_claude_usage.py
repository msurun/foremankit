"""Tests for foremankit.claude_usage: parsing, discovery, caching, and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foremankit import claude_usage
from foremankit.claude_usage import UsageError

SESSION_RESETS = "Nov 6, 11:49pm (America/Santiago)"
WEEK_RESETS = "Nov 10, 2:59am (America/Santiago)"

SAMPLE = (
    "You are currently using your subscription to power your Claude Code usage\n"
    "\n"
    f"Current session: 60% used · resets {SESSION_RESETS}\n"
    f"Current week (all models): 56% used · resets {WEEK_RESETS}\n"
    "\n"
    "What's contributing to your usage?\n"
    "Approximate · sessions on this machine only · shares overlap\n"
)


def usage_text(
    session: int = 60,
    week: int = 56,
    session_resets: str = SESSION_RESETS,
    week_resets: str = WEEK_RESETS,
) -> str:
    """Build a two-line usage output with the given percents and reset labels."""
    return (
        f"Current session: {session}% used · resets {session_resets}\n"
        f"Current week (all models): {week}% used · resets {week_resets}\n"
    )


def make_extension(home: Path, version: str, name: str = "claude.exe") -> Path:
    """Create a fake VS Code extension binary and return its path."""
    exe = home / ".vscode" / "extensions" / f"anthropic.claude-code-{version}" / "resources" / "native-binary" / name
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    return exe


def sample_fetch() -> str:
    """A fake fetch that returns the measured usage text."""
    return SAMPLE


def failing_fetch() -> str:
    """A fake fetch that always fails like the real binary would."""
    raise UsageError("boom")


def test_sample_text_parses_both_windows():
    """The measured output gives 60% session and 56% week with both reset labels."""
    assert claude_usage.parse_usage(SAMPLE) == {
        "session_pct": 60,
        "session_resets": SESSION_RESETS,
        "week_pct": 56,
        "week_resets": WEEK_RESETS,
    }


def test_only_session_line_leaves_week_keys_none():
    """A missing week line gives None for both of its keys."""
    text = f"Current session: 42% used · resets {SESSION_RESETS}\n"
    assert claude_usage.parse_usage(text) == {
        "session_pct": 42,
        "session_resets": SESSION_RESETS,
        "week_pct": None,
        "week_resets": None,
    }


def test_only_week_line_leaves_session_keys_none():
    """A missing session line gives None for both of its keys."""
    text = f"Current week (all models): 17% used · resets {WEEK_RESETS}\n"
    assert claude_usage.parse_usage(text) == {
        "session_pct": None,
        "session_resets": None,
        "week_pct": 17,
        "week_resets": WEEK_RESETS,
    }


def test_no_usage_lines_returns_none():
    """Garbage and empty output contain no readings."""
    assert claude_usage.parse_usage("") is None
    assert claude_usage.parse_usage("nothing useful here") is None


def test_percent_bounds_zero_and_one_hundred():
    """Both 0% and 100% are valid percentages."""
    assert claude_usage.parse_usage(usage_text(session=0, week=100))["session_pct"] == 0
    assert claude_usage.parse_usage(usage_text(session=0, week=100))["week_pct"] == 100


def test_separator_and_extra_spaces_are_tolerated():
    """Any separator between the percent and 'resets' works, as do extra spaces."""
    text = "Current session:   7 %  used - resets   Nov 6  \nCurrent week: 8% used ... resets Nov 7\n"
    assert claude_usage.parse_usage(text) == {
        "session_pct": 7,
        "session_resets": "Nov 6",
        "week_pct": 8,
        "week_resets": "Nov 7",
    }


def test_find_claude_exe_prefers_numeric_newest(tmp_path):
    """Version 2.10.0 beats 2.9.0 even though it sorts lower as text."""
    make_extension(tmp_path, "2.9.0")
    newest = make_extension(tmp_path, "2.10.0")
    assert claude_usage.find_claude_exe(tmp_path) == newest


def test_find_claude_exe_without_extensions_folder(tmp_path):
    """No VS Code extensions folder means no binary."""
    assert claude_usage.find_claude_exe(tmp_path) is None


def test_find_claude_exe_accepts_plain_claude_name(tmp_path):
    """The non-Windows binary name 'claude' is found."""
    exe = make_extension(tmp_path, "2.1.0", name="claude")
    assert claude_usage.find_claude_exe(tmp_path) == exe


def test_fetch_usage_with_missing_binary_raises(tmp_path):
    """A nonexistent executable becomes a UsageError, not an OSError."""
    with pytest.raises(UsageError):
        claude_usage.fetch_usage(tmp_path / "missing" / "claude.exe", timeout_s=5)


def test_refresh_success_writes_the_file(tmp_path):
    """A good fetch stores the reading with its timestamp and returns ok."""
    path = tmp_path / "usage.json"
    result = claude_usage.refresh(path, 1000.0, fetch=sample_fetch)
    assert result["ok"] is True
    assert result["last"]["session_pct"] == 60
    assert result["last"]["week_pct"] == 56
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["last"]["read_at"] == 1000.0
    assert saved["baseline"] is None


def test_refresh_failure_keeps_the_previous_reading(tmp_path):
    """A failing fetch reports the error and leaves the last good reading alone."""
    path = tmp_path / "usage.json"
    previous = claude_usage.refresh(path, 1000.0, fetch=sample_fetch)["last"]
    result = claude_usage.refresh(path, 2000.0, fetch=failing_fetch)
    assert result == {"ok": False, "error": "boom", "last": previous}
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["last"] == previous


def test_refresh_with_unparsable_text_keeps_the_previous_reading(tmp_path):
    """Text without usage lines is an error that does not erase the old reading."""
    path = tmp_path / "usage.json"
    previous = claude_usage.refresh(path, 1000.0, fetch=sample_fetch)["last"]
    result = claude_usage.refresh(path, 2000.0, fetch=lambda: "not a usage report")
    assert result["ok"] is False
    assert result["error"]
    assert result["last"] == previous


def test_refresh_failure_without_a_previous_reading(tmp_path):
    """The first ever refresh can fail cleanly with last set to None."""
    path = tmp_path / "usage.json"
    result = claude_usage.refresh(path, 1000.0, fetch=failing_fetch)
    assert result == {"ok": False, "error": "boom", "last": None}


def test_mark_baseline_without_a_reading_raises(tmp_path):
    """There is nothing to mark before the first successful refresh."""
    with pytest.raises(ValueError) as error:
        claude_usage.mark_baseline(tmp_path / "usage.json", 1000.0)
    assert str(error.value) == "no reading yet: refresh first"


def test_mark_baseline_copies_the_last_reading(tmp_path):
    """The baseline is a copy of the current reading, including its timestamp."""
    path = tmp_path / "usage.json"
    last = claude_usage.refresh(path, 1000.0, fetch=sample_fetch)["last"]
    baseline = claude_usage.mark_baseline(path, 2000.0)
    assert baseline == last
    assert claude_usage.report(path, 2000.0)["baseline"] == last


def test_report_shows_week_delta_of_five_points(tmp_path):
    """56% at the baseline and 61% later is +5 week percentage points."""
    path = tmp_path / "usage.json"
    claude_usage.refresh(path, 100.0, fetch=lambda: usage_text(week=56))
    claude_usage.mark_baseline(path, 110.0)
    claude_usage.refresh(path, 200.0, fetch=lambda: usage_text(week=61))
    data = claude_usage.report(path, 250.0)
    assert data["delta"] == {"session_pct": 0, "week_pct": 5}
    assert data["age_s"] == 50.0


def test_report_week_delta_is_none_after_a_week_reset(tmp_path):
    """A different week reset label means the weekly window restarted."""
    path = tmp_path / "usage.json"
    claude_usage.refresh(path, 100.0, fetch=lambda: usage_text(session=10, week=56))
    claude_usage.mark_baseline(path, 110.0)
    claude_usage.refresh(path, 200.0, fetch=lambda: usage_text(session=15, week=61, week_resets="Nov 17, 2:59am"))
    delta = claude_usage.report(path, 200.0)["delta"]
    assert delta == {"session_pct": 5, "week_pct": None}


def test_report_session_delta_is_none_after_a_session_reset(tmp_path):
    """A different session reset label means the 5-hour window restarted."""
    path = tmp_path / "usage.json"
    claude_usage.refresh(path, 100.0, fetch=lambda: usage_text(session=10, week=56))
    claude_usage.mark_baseline(path, 110.0)
    claude_usage.refresh(path, 200.0, fetch=lambda: usage_text(session=15, week=61, session_resets="Nov 7, 4:49am"))
    delta = claude_usage.report(path, 200.0)["delta"]
    assert delta == {"session_pct": None, "week_pct": 5}


def test_report_delta_is_none_without_both_readings(tmp_path):
    """A baseline is required before a delta can be computed."""
    path = tmp_path / "usage.json"
    claude_usage.refresh(path, 100.0, fetch=sample_fetch)
    assert claude_usage.report(path, 200.0)["delta"] is None


def test_report_ages_come_from_now(tmp_path):
    """Both ages are measured from the supplied clock, not the wall clock."""
    path = tmp_path / "usage.json"
    claude_usage.refresh(path, 100.0, fetch=sample_fetch)
    claude_usage.mark_baseline(path, 150.0)
    data = claude_usage.report(path, 400.0)
    assert data["age_s"] == 300.0
    assert data["baseline_age_s"] == 300.0


def test_report_on_corrupt_json_returns_all_none(tmp_path):
    """A corrupt state file is treated as missing data, not an exception."""
    path = tmp_path / "usage.json"
    path.write_text("{not json", encoding="utf-8")
    assert claude_usage.report(path, 100.0) == {
        "last": None,
        "age_s": None,
        "baseline": None,
        "baseline_age_s": None,
        "delta": None,
    }
