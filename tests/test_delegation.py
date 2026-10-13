"""Tests for foremankit.delegation module."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from foremankit.delegation import (
    CATEGORIES,
    category_of,
    describe,
    SWITCHED_OFF_MESSAGE,
    all_switched_off,
    is_switched_off,
    set_switch,
)
from foremankit.state import StateStore


class TestCategoryOf:
    """Tests for category_of function."""

    def test_a_llm_provider_is_classified_as_local(self):
        provider = SimpleNamespace(is_llm=True, free=False)
        assert category_of(provider) == "local"

    def test_a_free_non_llm_provider_is_classified_as_free(self):
        provider = SimpleNamespace(is_llm=False, free=True)
        assert category_of(provider) == "free"

    def test_a_paid_non_llm_provider_is_classified_as_paid(self):
        provider = SimpleNamespace(is_llm=False, free=False)
        assert category_of(provider) == "paid"


class TestIsSwitchedOff:
    """Tests for is_switched_off function."""

    def test_a_missing_key_means_on(self):
        state = {"toggles": {}}
        assert not is_switched_off(state, "paid")

    def test_an_explicit_on_means_not_off(self):
        state = {"toggles": {"delegation:paid": "on"}}
        assert not is_switched_off(state, "paid")

    def test_an_explicit_off_means_off(self):
        state = {"toggles": {"delegation:paid": "off"}}
        assert is_switched_off(state, "paid")

    def test_only_the_specific_category_is_checked(self):
        state = {"toggles": {"delegation:paid": "off", "delegation:free": "on"}}
        assert is_switched_off(state, "paid")
        assert not is_switched_off(state, "free")


class TestSetSwitch:
    """Tests for set_switch function."""

    def test_setting_a_single_category_returns_it(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        changed = set_switch(store, "paid", "off")
        assert changed == ["paid"]

    def test_setting_all_categories_returns_all(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        changed = set_switch(store, "all", "off")
        assert set(changed) == set(CATEGORIES)

    def test_setting_invalid_target_raises_value_error(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        with pytest.raises(ValueError, match="Invalid target"):
            set_switch(store, "invalid", "on")

    def test_setting_invalid_value_raises_value_error(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        with pytest.raises(ValueError, match="Invalid value"):
            set_switch(store, "paid", "maybe")

    def test_persisted_state_reflects_the_change(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        set_switch(store, "paid", "off")
        state = store.load()
        assert is_switched_off(state, "paid")
        assert not is_switched_off(state, "free")
        assert not is_switched_off(state, "local")

    def test_setting_on_cleans_off_state(self, tmp_path: Path):
        store = StateStore(tmp_path / "state.json")
        set_switch(store, "paid", "off")
        set_switch(store, "paid", "on")
        state = store.load()
        assert not is_switched_off(state, "paid")


class TestDescribe:
    """Tests for describe function."""

    def test_default_state_describes_all_on(self):
        state = {"toggles": {}}
        description = describe(state)
        lines = description.split("\n")
        assert len(lines) == 3
        assert "paid: on" in lines
        assert "free: on" in lines
        assert "local: on" in lines

    def test_describe_reflects_off_state(self):
        state = {"toggles": {"delegation:paid": "off"}}
        description = describe(state)
        lines = description.split("\n")
        assert "paid: off" in lines
        assert "free: on" in lines
        assert "local: on" in lines


class TestSwitchedOffMessage:
    """When the user turns every category off, the caller must be told clearly to do the work itself."""

    def test_all_switched_off_needs_every_category_off(self):
        assert not all_switched_off({"toggles": {"delegation:paid": "off", "delegation:free": "off"}})
        assert all_switched_off({"toggles": {f"delegation:{c}": "off" for c in CATEGORIES}})

    def test_the_message_says_to_do_the_task_yourself_and_not_to_retry(self):
        assert "Do this task yourself" in SWITCHED_OFF_MESSAGE and "do not retry" in SWITCHED_OFF_MESSAGE
