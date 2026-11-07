# tests/test_agent_quota.py
"""Tests for ForemanKit's agent_quota module.
Covers slots, quota-error detection, state persistence, daily limits and cooldowns, with no I/O beyond tmp_path.
"""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from foremankit.agent_quota import (
    PROBE_WINDOW_S,
    QUOTA_TEXT,
    can_start,
    clear_pause,
    is_quota_error,
    load_state,
    parse_reset,
    pause_info,
    record_quota_error,
    record_run,
    record_success,
    save_state,
    slots,
)


def at(day: int, hour: int = 12) -> float:
    """A Unix timestamp on 2026-11-<day> at `hour` UTC."""
    return datetime(2026, 11, day, hour, tzinfo=timezone.utc).timestamp()


def test_slots_defaults() -> None:
    """opencode defaults to 3 slots, agy and Codex to 1, and any other runner to 1."""
    assert slots({}, "opencode") == 3
    assert slots({}, "codex") == 1
    assert slots({}, "agy") == 1
    assert slots({}, "something-else") == 1
    assert slots({"max_parallel": None}, "agy") == 1


def test_slots_configured() -> None:
    """Configured max_parallel wins over the per-runner default."""
    settings = {"max_parallel": {"opencode": 5, "agy": 2, "other": 4}}
    assert slots(settings, "opencode") == 5
    assert slots(settings, "agy") == 2
    assert slots(settings, "other") == 4
    assert slots(settings, "unlisted") == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HTTP 429", True),
        ("Quota exceeded", True),
        ("rate limit", True),
        ("You've hit your usage limit. Try again in 3h 20m", True),
        ("Rate limit exceeded", True),
        ("429 Too Many Requests", True),
        ("insufficient credits", True),
        ("hit your limit", True),
        ("limit reached", True),
        ("reached your limit", True),
        ("out of credits", True),
        ("try again later", True),
        ("Here is a detailed explanation of the implementation and its tests. " * 50, False),
        ("all good", False),
        ("", False),
        (None, False),
    ],
)
def test_is_quota_error(text, expected) -> None:
    """Only rate-limit/quota texts match RATE_LIMIT_TEXT; empty and None are never errors."""
    assert is_quota_error(text) is expected
    assert QUOTA_TEXT.search(text or "") is not None if expected else not QUOTA_TEXT.search(text or "")


def test_load_state_missing_file(tmp_path: Path) -> None:
    """A missing state file loads as an empty dict."""
    assert load_state(tmp_path / "missing.json") == {}


def test_load_state_corrupt_json(tmp_path: Path) -> None:
    """A corrupt state file loads as an empty dict instead of raising."""
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_state(path) == {}


def test_load_state_non_dict_json(tmp_path: Path) -> None:
    """A JSON file that is not an object loads as an empty dict."""
    path = tmp_path / "state.json"
    path.write_text("[1, 2]", encoding="utf-8")
    assert load_state(path) == {}


def test_save_load_round_trip(tmp_path: Path) -> None:
    """State survives a save/load round trip, and save creates the parent folder."""
    path = tmp_path / "nested" / "dir" / "state.json"
    state = {"agy": {"day": "2026-11-06", "runs": 3, "paused_until": 1234.5}}
    save_state(path, state)
    assert load_state(path) == state
    assert json.loads(path.read_text(encoding="utf-8")) == state


def test_unknown_runner_is_free() -> None:
    """A runner missing from the state has zero runs and is not paused."""
    assert can_start({}, "agy", at(6), {}) == (True, "")


def test_record_run_counts_for_the_day() -> None:
    """Each recorded run adds one to the runner's count for that UTC day."""
    state: dict = {}
    record_run(state, "agy", at(6, 8))
    record_run(state, "agy", at(6, 9))
    assert state["agy"]["runs"] == 2
    assert state["agy"]["day"] == "2026-11-06"
    assert can_start(state, "agy", at(6, 10), {"quota": {"agy": {"max_runs_per_day": 2}}})[0] is False


def test_limit_exactly_reached() -> None:
    """With a limit of 15, 14 runs are fine and the 15th starts, the 16th does not."""
    settings = {"quota": {"agy": {"max_runs_per_day": 15}}}
    state: dict = {}
    for _ in range(14):
        record_run(state, "agy", at(6))
    assert can_start(state, "agy", at(6), settings) == (True, "")
    record_run(state, "agy", at(6))
    ok, reason = can_start(state, "agy", at(6), settings)
    assert ok is False
    assert "15" in reason
    assert "agy" in reason


def test_limit_zero_is_unlimited() -> None:
    """max_runs_per_day 0 (and a missing quota block) means no daily limit."""
    state: dict = {}
    for _ in range(100):
        record_run(state, "opencode", at(6))
    assert can_start(state, "opencode", at(6), {"quota": {"opencode": {"max_runs_per_day": 0}}}) == (True, "")
    assert can_start(state, "opencode", at(6), {}) == (True, "")


def test_missing_limit_is_unlimited() -> None:
    """A quota block without max_runs_per_day has no daily limit."""
    state: dict = {}
    for _ in range(50):
        record_run(state, "agy", at(6))
    assert can_start(state, "agy", at(6), {"quota": {"agy": {"cooldown_on_error_s": 60}}}) == (True, "")


def test_day_rollover_resets_runs_but_not_the_pause() -> None:
    """A new UTC day resets the run count while a pause that crosses midnight stays."""
    settings = {"quota": {"agy": {"max_runs_per_day": 2, "cooldown_on_error_s": 7200}}}
    state: dict = {}
    record_run(state, "agy", at(6, 22))
    record_run(state, "agy", at(6, 22))
    until = record_quota_error(state, "agy", at(6, 23), settings)   # Nov 7 01:00 UTC
    assert can_start(state, "agy", at(6, 23), settings)[0] is False
    ok, reason = can_start(state, "agy", at(7, 0), settings)
    assert ok is False and "agy" in reason   # pause survives the rollover
    assert state["agy"]["runs"] == 0         # runs were reset
    assert state["agy"]["paused_until"] == until
    assert can_start(state, "agy", until, settings) == (True, "")   # then the new day allows a run


def test_pause_blocks_until_and_expires_exactly_at_now() -> None:
    """While paused the reason names the runner and seconds left; at paused_until it may start."""
    settings = {"quota": {"agy": {"backoff_s": [1800]}}}
    state: dict = {}
    until = record_quota_error(state, "agy", at(6), settings)
    assert until == at(6) + 1800
    ok, reason = can_start(state, "agy", at(6) + 60, settings)
    assert ok is False
    assert "agy" in reason
    assert "1740" in reason
    assert can_start(state, "agy", until, settings) == (True, "")


def test_record_quota_error_does_not_count_a_run() -> None:
    """A quota error only pauses the runner; the run count is untouched."""
    state: dict = {}
    record_run(state, "agy", at(6))
    record_quota_error(state, "agy", at(6), {})
    assert state["agy"]["runs"] == 1


def test_second_quota_error_never_stacks_the_cooldown() -> None:
    """A second error keeps paused_until = max(old, now + cooldown) instead of adding a whole cooldown again."""
    settings = {"quota": {"agy": {"backoff_s": [1800]}}}
    state: dict = {}
    first = record_quota_error(state, "agy", at(6), settings)
    same_time = record_quota_error(state, "agy", at(6), {"quota": {"agy": {"backoff_s": [60]}}})
    assert same_time == first                       # a shorter cooldown cannot shorten the pause
    later = record_quota_error(state, "agy", at(6) + 600, settings)
    assert later == at(6) + 600 + 1800
    assert later < first + 1800                     # max(), not old + cooldown
    assert state["agy"]["paused_until"] == later


def test_two_runners_are_independent() -> None:
    """Pausing or exhausting agy does not stop opencode, and their counters differ."""
    settings = {"quota": {"agy": {"max_runs_per_day": 1, "cooldown_on_error_s": 1800}}}
    state: dict = {}
    record_run(state, "agy", at(6))
    record_quota_error(state, "agy", at(6), settings)
    assert can_start(state, "agy", at(6), settings)[0] is False
    assert can_start(state, "opencode", at(6), settings) == (True, "")
    assert state.get("opencode", {}).get("runs", 0) == 0


def test_settings_are_never_mutated() -> None:
    """Every function leaves the settings dict exactly as it was."""
    settings = {"max_parallel": {"opencode": 3, "agy": 1}, "quota": {"agy": {"max_runs_per_day": 15, "cooldown_on_error_s": 1800}}}
    before = copy.deepcopy(settings)
    state: dict = {}
    slots(settings, "agy")
    record_run(state, "agy", at(6))
    record_quota_error(state, "agy", at(6), settings)
    can_start(state, "agy", at(6), settings)
    can_start(state, "nya", at(6), settings)
    assert settings == before


def test_backoff_and_success() -> None:
    """Repeated errors escalate, and success restores the first backoff step."""
    state: dict = {}
    for strike, delay in enumerate([1800, 3600, 7200, 18000, 18000], start=1):
        assert record_quota_error(state, "agy", at(6), {}) == at(6) + delay
        assert state["agy"]["strikes"] == strike
    record_run(state, "agy", at(7))
    record_success(state, "agy")
    assert state["agy"]["strikes"] == state["agy"]["probe_until"] == 0
    assert state["agy"]["paused_until"] == at(6) + 18000
    assert record_quota_error(state, "agy", at(7), {}) == at(7) + 1800


def test_legacy_cooldown_first_step() -> None:
    """The old cooldown supplies only the first backoff step."""
    state: dict = {}
    settings = {"quota": {"agy": {"cooldown_on_error_s": 90}}}
    assert record_quota_error(state, "agy", at(6), settings) == at(6) + 90
    assert record_quota_error(state, "agy", at(6), settings) == at(6) + 3600


@pytest.mark.parametrize(("message", "seconds"), [
    ("try again in 3h 20m", 12000), ("in 45 minutes", 2700), ("in 2 hours", 7200),
    ("retry after 120 seconds", 120), ("retry-after: 120", 120), ("RESETS IN 1h", 3600),
    ("garbage", None), ("", None), ("at 99:99", None),
])
def test_parse_reset(message, seconds) -> None:
    """Relative reset messages resolve to absolute timestamps; garbage is ignored."""
    assert parse_reset(message, at(6)) == (None if seconds is None else at(6) + seconds)


@pytest.mark.parametrize("message", ["try again at 4:40 PM", "resets at 16:40", "at 12:00 AM"])
@pytest.mark.parametrize("hour", [0, 23])
def test_clock_reset(message, hour) -> None:
    """Clock reset messages choose the next occurrence in local time."""
    now = datetime(2026, 11, 6, hour, 1).timestamp()
    reset = parse_reset(message, now)
    assert now < reset < now + 86400
    expected_hour = 0 if "AM" in message else 16
    assert datetime.fromtimestamp(reset).hour == expected_hour


@pytest.mark.parametrize(("message", "delay"), [
    ("retry-after: 1", 60), ("in 2 hours", 7200), ("in 48 hours", 86400),
])
def test_reset_overrides_backoff_and_clamps(message, delay) -> None:
    """Parsed resets win over backoff within the one-minute to one-day bounds."""
    assert record_quota_error({}, "agy", at(6), {}, message) == at(6) + delay


def test_probe_recovery_and_crash() -> None:
    """Only one probe runs until success or the stale probe window expires."""
    state: dict = {}
    until = record_quota_error(state, "agy", at(6), {})
    assert can_start(state, "agy", until, {}) == (True, "")
    record_run(state, "agy", until)
    assert can_start(state, "agy", until, {}) == (False, "agy is being probed after a limit pause")
    assert not can_start(state, "agy", until + PROBE_WINDOW_S - 1, {})[0]
    assert can_start(state, "agy", until + PROBE_WINDOW_S, {})[0]
    record_run(state, "agy", until + PROBE_WINDOW_S)
    record_success(state, "agy")
    assert can_start(state, "agy", until + PROBE_WINDOW_S, {})[0]


def test_pause_info_and_clear() -> None:
    """Clearing a pause retains strikes so an immediate run becomes a probe."""
    state: dict = {}
    until = record_quota_error(state, "agy", at(6), {})
    assert pause_info(state, "agy", at(6)) == {
        "paused": True, "until": until, "remaining_s": 1800, "strikes": 1, "probing": False}
    assert pause_info(state, "agy", until)["probing"]
    clear_pause(state, "agy")
    assert pause_info(state, "agy", at(6)) == {
        "paused": False, "until": 0.0, "remaining_s": 0, "strikes": 1, "probing": True}
    record_run(state, "agy", at(6))
    assert not can_start(state, "agy", at(6), {})[0]


def test_old_state_and_soft_limit(tmp_path: Path) -> None:
    """Old state gains recovery fields lazily and soft daily limits never block."""
    path = tmp_path / "old.json"
    save_state(path, {"agy": {"day": "2026-11-06", "runs": 20, "paused_until": at(6) - 1}})
    state = load_state(path)
    assert can_start(state, "agy", at(6), {"quota": {"agy": {"soft_runs_per_day": 15}}})[0]
    assert state["agy"]["strikes"] == state["agy"]["probe_until"] == 0
    assert record_quota_error(state, "agy", at(6), {}) == at(6) + 1800
