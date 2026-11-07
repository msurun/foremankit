"""Tests for foremankit.panel_agent_runs: the pure status builder and its thin wrappers."""
from __future__ import annotations

from pathlib import Path

from foremankit import panel_agent_runs
from foremankit.agent_switch import is_enabled
from foremankit.panel_agent_runs import (
    StatusInputs,
    agent_runs_status,
    build_status,
    mark_usage,
    refresh_usage,
    set_agent_run,
    usage_path,
)
from foremankit.state import CallLog, StateStore, utc_day

NOW = 1_700_000_000.0


def make_inputs(**overrides) -> StatusInputs:
    """StatusInputs with empty data, overridden per test."""
    values = {"toggles_state": {}, "settings": {}, "quota_state": {}, "rows": [], "usage": {}, "now": NOW}
    values.update(overrides)
    return StatusInputs(**values)


def outcome(index: int, **fields) -> dict:
    """One agent-run outcome row with an increasing timestamp."""
    row = {"event": "outcome", "kind": "agent_run", "id": f"r{index}", "provider": "opencode",
           "model": "opencode/free", "accepted": index % 2 == 0, "verify_passed": True,
           "seconds": 1.5, "ts": NOW + index}
    row.update(fields)
    return row


def test_codex_panel_entry_and_model():
    """Codex exposes its own switch, one slot, normal-only data and configured/latest model."""
    inputs = make_inputs(settings={"codex": {"default_model": "configured"}})
    entry = build_status(inputs)["runners"]["codex"]
    assert entry["label"] == "Codex" and entry["model"] == "configured"
    assert entry["slots"] == 1 and entry["internal_data"] is False and entry["active"]
    inputs = make_inputs(toggles_state={"toggles": {"agent_run:codex": "off"}},
                         rows=[outcome(0, provider="codex", model="latest")])
    entry = build_status(inputs)["runners"]["codex"]
    assert entry["model"] == "latest" and not entry["own"] and not entry["active"]
    assert build_status(inputs)["runners"]["agy"]["active"]


class TestBuildStatus:
    """The pure builder with synthetic inputs."""

    def test_empty_inputs_give_all_runners_and_zeros(self):
        status = build_status(make_inputs())
        assert status["enabled"] is True
        assert set(status["runners"]) == {"agy", "opencode", "codex"}
        assert status["recent"] == []
        assert status["reviews"] == {}
        assert status["review_text"] == "no agent-run reviews yet"
        assert status["usage"] == {}
        for entry in status["runners"].values():
            assert entry["own"] is True and entry["active"] is True
            assert entry["runs_today"] == 0 and entry["daily_limit"] == 0
            assert entry["paused_for_s"] == 0
        assert status["runners"]["agy"]["slots"] == 1
        assert status["runners"]["opencode"]["slots"] == 3
        assert status["runners"]["agy"]["internal_data"] is True
        assert status["runners"]["opencode"]["internal_data"] is False

    def test_master_off_keeps_own_switches_but_deactivates_all(self):
        status = build_status(make_inputs(toggles_state={"toggles": {"agent_run:all": "off"}}))
        assert status["enabled"] is False
        assert [entry["own"] for entry in status["runners"].values()] == [True, True, True]
        assert [entry["active"] for entry in status["runners"].values()] == [False, False, False]

    def test_agy_off_deactivates_only_agy(self):
        status = build_status(make_inputs(toggles_state={"toggles": {"agent_run:agy": "off"}}))
        assert status["runners"]["agy"]["own"] is False
        assert status["runners"]["agy"]["active"] is False
        assert status["runners"]["opencode"]["active"] is True

    def test_runs_today_ignores_yesterdays_counter(self):
        quota = {"agy": {"day": utc_day(NOW), "runs": 2},
                 "opencode": {"day": utc_day(NOW - 86400), "runs": 9}}
        status = build_status(make_inputs(quota_state=quota))
        assert status["runners"]["agy"]["runs_today"] == 2
        assert status["runners"]["opencode"]["runs_today"] == 0

    def test_paused_for_s_counts_down_and_never_goes_negative(self):
        quota = {"agy": {"paused_until": NOW + 120.4}, "opencode": {"paused_until": NOW - 30}}
        status = build_status(make_inputs(quota_state=quota))
        assert status["runners"]["agy"]["paused_for_s"] == 120
        assert status["runners"]["opencode"]["paused_for_s"] == 0

    def test_settings_set_the_daily_limits_and_slots(self):
        settings = {"quota": {"agy": {"max_runs_per_day": 15}, "opencode": {"max_runs_per_day": 4}},
                    "max_parallel": {"agy": 2}}
        status = build_status(make_inputs(settings=settings))
        assert status["runners"]["agy"]["daily_limit"] == 15
        assert status["runners"]["opencode"]["daily_limit"] == 4
        assert status["runners"]["agy"]["slots"] == 2
        assert status["runners"]["opencode"]["slots"] == 3

    def test_recent_keeps_only_agent_run_outcomes_newest_first_and_max_eight(self):
        rows = [outcome(index) for index in range(10)]
        rows += [{"event": "call", "kind": "agent_run", "id": "call"},
                 {"event": "outcome", "kind": "delegate", "id": "other"},
                 {"event": "agent_run_review", "id": "r1", "provider": "agy", "model": "flash_high",
                  "verdict": "merged"}]
        status = build_status(make_inputs(rows=rows))
        recent = status["recent"]
        assert [row["id"] for row in recent] == [f"r{index}" for index in range(9, 1, -1)]
        assert recent[0]["passed"] is False
        assert recent[1]["passed"] is True
        assert recent[0]["runner"] == "opencode" and recent[0]["model"] == "opencode/free"
        assert recent[0]["ts"] == NOW + 9

    def test_recent_keeps_a_missing_seconds_value_as_none(self):
        rows = [outcome(0, seconds=None)]
        assert build_status(make_inputs(rows=rows))["recent"][0]["seconds"] is None

    def test_one_review_row_fills_reviews_and_review_text(self):
        rows = [{"event": "agent_run_review", "id": "r1", "provider": "agy", "model": "flash_high",
                 "verdict": "merged", "blockers": 0}]
        status = build_status(make_inputs(rows=rows))
        assert status["reviews"]["agy/flash_high"]["reviews"] == 1
        assert status["reviews"]["agy/flash_high"]["clean_rate"] == 1.0
        assert "agy/flash_high: 1 reviews" in status["review_text"]

    def test_models_fall_back_then_use_the_latest_logged_one(self):
        settings = {"opencode": {"default_model": "opencode/default"}}
        status = build_status(make_inputs(settings=settings))
        assert status["runners"]["opencode"]["model"] == "opencode/default"
        assert status["runners"]["agy"]["model"] == "flash_high"
        rows = [outcome(0, model="opencode/old"), outcome(1, model="opencode/new"),
                outcome(2, provider="agy", model="flash_low")]
        status = build_status(make_inputs(settings=settings, rows=rows))
        assert status["runners"]["opencode"]["model"] == "opencode/new"
        assert status["runners"]["agy"]["model"] == "flash_low"

    def test_usage_is_returned_unchanged_and_there_are_no_cost_fields(self):
        usage = {"last": {"session_pct": 60}, "delta": None}
        status = build_status(make_inputs(usage=usage))
        assert status["usage"] == usage
        assert "cost" not in repr(status).lower()


class TestWrappers:
    """The wrappers that read and write the real state folder."""

    def test_set_agent_run_persists_the_switch(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
        assert set_agent_run("opencode", False) == "agent runs off: opencode"
        state = StateStore().load()
        assert is_enabled(state, "opencode") is False
        assert is_enabled(state, "agy") is True

    def test_set_agent_run_reports_a_bad_target(self):
        assert set_agent_run("claude", True).startswith("not done:")

    def test_refresh_usage_reports_the_percentages(self, monkeypatch):
        calls = []
        monkeypatch.setattr(panel_agent_runs.claude_usage, "refresh",
                            lambda path, now: calls.append((path, now)) or
                            {"ok": True, "last": {"session_pct": 60, "week_pct": 56}})
        assert refresh_usage(NOW) == "usage: session 60%, week 56%"
        assert calls == [(usage_path(), NOW)]

    def test_refresh_usage_reports_a_failed_read(self, monkeypatch):
        monkeypatch.setattr(panel_agent_runs.claude_usage, "refresh",
                            lambda path, now: {"ok": False, "error": "claude binary not found"})
        assert refresh_usage(NOW) == "usage not read: claude binary not found"

    def test_mark_usage_reports_success_and_missing_reading(self, monkeypatch):
        monkeypatch.setattr(panel_agent_runs.claude_usage, "mark_baseline", lambda path, now: {})
        assert mark_usage(NOW) == "baseline marked"

        def fail(path: Path, now: float) -> dict:
            raise ValueError("no reading yet: refresh first")

        monkeypatch.setattr(panel_agent_runs.claude_usage, "mark_baseline", fail)
        assert mark_usage(NOW) == "not done: no reading yet: refresh first"

    def test_usage_path_lives_under_the_state_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
        assert usage_path() == tmp_path / "state" / "usage.json"

    def test_agent_runs_status_loads_state_and_log(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.setattr(panel_agent_runs, "_agent_run_settings", dict)
        monkeypatch.setattr(panel_agent_runs.agent_quota, "state_path", lambda: tmp_path / "quota.json")
        monkeypatch.setattr(panel_agent_runs.claude_usage, "report", lambda path, now: {"last": None})
        StateStore().update(lambda state: state.setdefault("toggles", {}).update({"agent_run:opencode": "off"}))
        CallLog().write({"event": "outcome", "kind": "agent_run", "id": "x", "provider": "opencode",
                         "model": "m", "accepted": True, "verify_passed": True, "seconds": 1.5})
        status = agent_runs_status(NOW)
        assert status["runners"]["opencode"]["own"] is False
        assert status["recent"][0]["id"] == "x"
        assert status["usage"] == {"last": None}


def test_internal_data_follows_configured_ceiling():
    """All three panel entries reflect eligibility for both Codex ceilings and its default."""
    for ceiling in (None, "normal", "internal"):
        settings = {} if ceiling is None else {"codex": {"max_sensitivity": ceiling}}
        runners = build_status(make_inputs(settings=settings))["runners"]
        assert runners["codex"]["internal_data"] is (ceiling == "internal")
        assert runners["agy"]["internal_data"] is True
        assert runners["opencode"]["internal_data"] is False
