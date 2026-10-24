"""Tests for the escalation state machine in foremankit/escalation.py.

The gateway runs the main-loop orchestrator on a cheap model and moves it to an expensive one after repeated tool
errors, then back down after a clean streak plus a minimum time (hysteresis). Tool results arrive as `tool_result`
content blocks in user messages, which is what the gateway actually reads from the Claude Code request.
"""

import pytest

from foremankit import escalation as esc


def _tool_result(is_error):
    """A message whose content holds one tool_result block (Claude Code's Anthropic shape)."""
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "is_error": is_error}]}


# --- last_tool_error ---------------------------------------------------------


def test_no_messages_is_not_an_error():
    assert esc.last_tool_error([]) is False


def test_a_message_with_string_content_is_not_an_error():
    assert esc.last_tool_error([{"role": "assistant", "content": "all good"}]) is False


def test_a_tool_result_without_is_error_is_not_an_error():
    assert esc.last_tool_error([{"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}]) is False


def test_only_the_last_tool_result_matters():
    messages = [_tool_result(True), _tool_result(False)]
    assert esc.last_tool_error(messages) is False


def test_last_tool_result_that_is_an_error_is_an_error():
    messages = [_tool_result(False), _tool_result(True)]
    assert esc.last_tool_error(messages) is True


# --- default_state / next_state ---------------------------------------------


def test_default_state_starts_cheap():
    assert esc.default_state() == {"tier": "cheap", "errors": 0, "clean": 0, "escalated_at": 0.0}


def test_next_state_does_not_mutate_input():
    state = esc.default_state()
    esc.next_state(state, False, {}, 1.0)
    assert state == esc.default_state()


def test_one_error_does_not_escalate():
    state = esc.next_state(esc.default_state(), True, {}, 1.0)
    assert state["tier"] == "cheap" and state["errors"] == 1


def test_two_errors_in_a_row_escalate():
    after_first = esc.next_state(esc.default_state(), True, {}, 1.0)
    state = esc.next_state(after_first, True, {}, 2.0)
    assert state["tier"] == "expensive" and state["escalated_at"] == 2.0
    assert state["errors"] == 0 and state["clean"] == 0


def test_clean_below_threshold_does_not_downgrade():
    expensive = {"tier": "expensive", "errors": 0, "clean": 4, "escalated_at": 0.0}
    state = esc.next_state(expensive, False, {}, 1000.0)
    assert state["tier"] == "expensive" and state["clean"] == 5


def test_downgrade_waits_for_the_minimum_expensive_time():
    expensive = {"tier": "expensive", "errors": 0, "clean": 5, "escalated_at": 0.0}
    assert esc.next_state(expensive, False, {}, 60.0)["tier"] == "expensive"
    assert esc.next_state(expensive, False, {}, 200.0)["tier"] == "cheap"


def test_error_while_expensive_resets_clean_and_stays():
    expensive = {"tier": "expensive", "errors": 0, "clean": 5, "escalated_at": 0.0}
    state = esc.next_state(expensive, True, {}, 200.0)
    assert state["tier"] == "expensive" and state["clean"] == 0 and state["errors"] == 1


def test_clean_turn_while_cheap_does_not_escalate():
    state = esc.next_state(esc.default_state(), False, {}, 1.0)
    assert state["tier"] == "cheap" and state["clean"] == 1


def test_policy_values_override_the_defaults():
    policy = {"errors": 1, "clean": 2, "min_expensive_s": 0.0}
    first = esc.next_state(esc.default_state(), True, policy, 1.0)
    assert first["tier"] == "expensive"
    second = esc.next_state(first, False, policy, 1.0)
    assert second["tier"] == "expensive" and second["clean"] == 1
    assert esc.next_state(second, False, policy, 2.0)["tier"] == "cheap"


# --- mode_of / tier_for_mode -------------------------------------------------


@pytest.mark.parametrize("value", [esc.AUTO, esc.FLASH, esc.PRO])
def test_mode_of_keeps_a_known_mode(value):
    """A known panel value is returned unchanged."""
    assert esc.mode_of(value) == value


@pytest.mark.parametrize("value", [None, "", "AUTO", "expensive", 3])
def test_mode_of_maps_unknown_values_to_auto(value):
    """Anything the panel can send that is not a known mode falls back to auto."""
    assert esc.mode_of(value) == esc.AUTO


def test_tier_for_mode_flash_pins_cheap():
    """The flash switch forces the cheap tier."""
    assert esc.tier_for_mode(esc.FLASH) == esc.CHEAP_TIER


def test_tier_for_mode_pro_pins_expensive():
    """The pro switch forces the expensive tier."""
    assert esc.tier_for_mode(esc.PRO) == esc.EXPENSIVE_TIER


def test_tier_for_mode_auto_pins_nothing():
    """Auto leaves the hysteresis in charge, so no tier is pinned."""
    assert esc.tier_for_mode(esc.AUTO) is None


@pytest.mark.parametrize("value", [None, "", "AUTO", "expensive", 3])
def test_tier_for_mode_unknown_value_pins_nothing(value):
    """An unknown mode is treated as auto and pins no tier."""
    assert esc.tier_for_mode(value) is None
