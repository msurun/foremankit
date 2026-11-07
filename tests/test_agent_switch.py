"""Tests for foremankit.agent_switch module."""
from __future__ import annotations

from pathlib import Path

import pytest

from foremankit.agent_switch import (
    MASTER_OFF_MESSAGE,
    RUNNERS,
    TARGETS,
    describe,
    disabled_runners,
    is_enabled,
    refusal,
    set_switch,
)
from foremankit.state import StateStore


class TestIsEnabled:
    """Tests for is_enabled function."""

    def test_empty_state_enables_the_master_and_every_runner(self):
        state = {"toggles": {}}
        assert is_enabled(state)
        for runner in RUNNERS:
            assert is_enabled(state, runner)

    def test_missing_toggles_key_enables_everything(self):
        assert is_enabled({})
        assert is_enabled({}, "agy")

    def test_master_off_disables_a_runner_whose_own_switch_is_on(self):
        state = {"toggles": {"agent_run:all": "off", "agent_run:agy": "on"}}
        assert not is_enabled(state)
        assert not is_enabled(state, "agy")
        assert not is_enabled(state, "opencode")

    def test_a_runner_off_disables_only_itself(self):
        state = {"toggles": {"agent_run:agy": "off"}}
        assert is_enabled(state)
        assert not is_enabled(state, "agy")
        assert is_enabled(state, "opencode")

    def test_unknown_runner_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid runner"):
            is_enabled({"toggles": {}}, "claude")


class TestDisabledRunners:
    """Tests for disabled_runners function."""

    def test_empty_state_disables_nobody(self):
        assert disabled_runners({"toggles": {}}) == frozenset()

    def test_master_off_disables_all_runners(self):
        state = {"toggles": {"agent_run:all": "off", "agent_run:agy": "on", "agent_run:opencode": "on"}}
        assert disabled_runners(state) == frozenset(RUNNERS)

    def test_a_runner_off_disables_only_itself(self):
        state = {"toggles": {"agent_run:opencode": "off"}}
        assert disabled_runners(state) == frozenset({"opencode"})


class TestSetSwitch:
    """Tests for set_switch function."""

    def test_setting_a_runner_returns_the_changed_target(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        assert set_switch(store, "agy", "off") == ["agy"]

    def test_setting_all_returns_all(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        assert set_switch(store, "all", "off") == ["all"]

    def test_all_off_changes_only_the_master_key(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        set_switch(store, "all", "off")
        assert store.load()["toggles"] == {"agent_run:all": "off"}

    def test_all_on_restores_the_runners_own_settings(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        set_switch(store, "agy", "off")
        set_switch(store, "all", "off")
        set_switch(store, "all", "on")
        state = store.load()
        assert is_enabled(state)
        assert not is_enabled(state, "agy")
        assert is_enabled(state, "opencode")

    def test_setting_on_clears_an_off_runner(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        set_switch(store, "opencode", "off")
        set_switch(store, "opencode", "on")
        assert is_enabled(store.load(), "opencode")

    def test_invalid_target_raises_value_error(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        with pytest.raises(ValueError, match="Invalid target"):
            set_switch(store, "invalid", "on")

    def test_invalid_value_raises_value_error(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        with pytest.raises(ValueError, match="Invalid value"):
            set_switch(store, "agy", "maybe")

    def test_targets_are_the_master_and_the_runners(self):
        assert TARGETS == ("all", *RUNNERS)

    def test_change_persists_across_a_new_store_on_the_same_file(self, tmp_path: Path):
        path = tmp_path / "state.json"
        set_switch(StateStore(path), "opencode", "off")
        fresh = StateStore(path)
        assert not is_enabled(fresh.load(), "opencode")
        assert disabled_runners(fresh.load()) == frozenset({"opencode"})

    def test_other_toggles_are_untouched(self, tmp_path: Path):
        path = tmp_path / "state.json"
        store = StateStore(path)
        store.update(lambda state: state.setdefault("toggles", {}).update({"delegation:agy": "off", "supervisor": "auto"}))
        set_switch(store, "agy", "off")
        set_switch(store, "all", "off")
        toggles = store.load()["toggles"]
        assert toggles["delegation:agy"] == "off"
        assert toggles["supervisor"] == "auto"


class TestRefusal:
    """Tests for refusal function."""

    def test_an_enabled_runner_is_not_refused(self):
        assert refusal({"toggles": {}}, "agy") == ""
        assert refusal({"toggles": {"agent_run:agy": "on"}}, "agy") == ""

    def test_master_off_says_agent_runs_are_switched_off(self):
        message = refusal({"toggles": {"agent_run:all": "off"}}, "opencode")
        assert message == MASTER_OFF_MESSAGE
        assert message.startswith("AGENT RUNS ARE SWITCHED OFF")
        assert "Do this task another way" in message
        assert "do not look for a way around it" in message

    def test_a_runner_off_names_the_runner(self):
        message = refusal({"toggles": {"agent_run:opencode": "off"}}, "opencode")
        assert message == ("opencode is switched off by the user (ForemanKit panel); "
                           "use the other runner or do the task another way.")

    def test_master_off_text_wins_when_both_are_off(self):
        state = {"toggles": {"agent_run:all": "off", "agent_run:agy": "off"}}
        assert refusal(state, "agy") == MASTER_OFF_MESSAGE


class TestDescribe:
    """Tests for describe function."""

    def test_default_state_describes_everything_on(self):
        lines = describe({"toggles": {}}).split("\n")
        assert lines == ["agent runs: on", "agy: on", "opencode: on"]

    def test_a_mixed_state_shows_each_own_switch(self):
        state = {"toggles": {"agent_run:all": "on", "agent_run:agy": "off"}}
        lines = describe(state).split("\n")
        assert lines == ["agent runs: on", "agy: off", "opencode: on"]

    def test_a_runner_line_shows_its_own_switch_not_the_master_combined_value(self):
        state = {"toggles": {"agent_run:all": "off", "agent_run:agy": "on"}}
        lines = describe(state).split("\n")
        assert lines == ["agent runs: off", "agy: on", "opencode: on"]
