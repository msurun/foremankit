"""Tests for foremankit.presets: preset names, transition plans, detection and issues.

Presets are pure data transformations, so every test builds a plain `current`
dict and checks the returned plan, preset name or issue list.
"""

import pytest

from foremankit import presets


CLAUDE_DIRECT_STATE = {
    "direct": True,
    "mode": "claude",
    "role": "worker",
    "server_up": False,
    "router_up": False,
}

ROUTER_LOCAL_STATE = {
    "direct": False,
    "mode": "local",
    "role": "orchestrator",
    "server_up": False,
    "router_up": True,
}

HEALTHY_CLAUDE_STATE = {
    "direct": True,
    "mode": "claude",
    "role": "worker",
    "server_up": False,
    "router_up": False,
    "llama_installed": True,
    "key_missing": False,
}

UPSTREAMS = ["openai", "ollama"]


def test_preset_names_lists_builtin_presets_before_upstreams():
    names = presets.preset_names(UPSTREAMS)

    assert names == ["claude", "local", "openai", "ollama"]


@pytest.mark.parametrize("name", ["claude", "local", "openai"])
def test_every_preset_from_claude_direct_returns_a_plan(name):
    plan = presets.preset_plan(name, CLAUDE_DIRECT_STATE)

    assert "steps" in plan
    assert isinstance(plan["summary"], list)


@pytest.mark.parametrize("name", ["claude", "local", "openai"])
def test_every_preset_from_router_state_returns_a_plan(name):
    plan = presets.preset_plan(name, ROUTER_LOCAL_STATE)

    assert "steps" in plan
    assert isinstance(plan["summary"], list)


def test_claude_preset_switches_mode_role_and_direct():
    plan = presets.preset_plan("claude", ROUTER_LOCAL_STATE)

    assert plan["steps"] == [
        {"action": "mode", "value": "claude"},
        {"action": "role", "value": "worker"},
        {"action": "direct", "value": True},
    ]


def test_local_preset_from_claude_direct_targets_local_both_via_router():
    plan = presets.preset_plan("local", CLAUDE_DIRECT_STATE)

    assert plan["steps"] == [
        {"action": "mode", "value": "local"},
        {"action": "role", "value": "both"},
        {"action": "direct", "value": False},
    ]


def test_local_preset_keeps_orchestrator_role_when_already_orchestrator():
    plan = presets.preset_plan("local", {**ROUTER_LOCAL_STATE, "mode": "claude"})

    assert plan["steps"] == [{"action": "mode", "value": "local"}]


def test_upstream_preset_sets_mode_and_keeps_role():
    plan = presets.preset_plan("openai", ROUTER_LOCAL_STATE)

    assert plan["steps"] == [{"action": "mode", "value": "openai"}]


def test_direct_step_is_always_last_when_present():
    plan = presets.preset_plan("claude", ROUTER_LOCAL_STATE)

    actions = [step["action"] for step in plan["steps"]]
    assert actions[-1] == "direct"


def test_reload_is_true_when_a_direct_step_exists():
    plan = presets.preset_plan("claude", ROUTER_LOCAL_STATE)

    assert plan["reload"] is True


def test_reload_is_false_when_no_direct_step_exists():
    plan = presets.preset_plan("openai", ROUTER_LOCAL_STATE)

    assert plan["reload"] is False


def test_preset_already_active_has_no_steps_and_summary_says_so():
    plan = presets.preset_plan("claude", CLAUDE_DIRECT_STATE)

    assert plan["steps"] == []
    assert plan["summary"] == ["Already active"]


def test_local_needs_server_when_server_is_down():
    plan = presets.preset_plan("local", CLAUDE_DIRECT_STATE)

    assert plan["needs_server"] is True


def test_local_does_not_need_server_when_server_is_up():
    running = dict(ROUTER_LOCAL_STATE, server_up=True)

    plan = presets.preset_plan("local", running)

    assert plan["needs_server"] is False


def test_non_local_preset_never_needs_server():
    plan = presets.preset_plan("claude", CLAUDE_DIRECT_STATE)

    assert plan["needs_server"] is False


@pytest.mark.parametrize(
    ("current", "expected_name"),
    [
        (CLAUDE_DIRECT_STATE, "claude"),
        (ROUTER_LOCAL_STATE, "local"),
        ({"direct": False, "mode": "local", "role": "both", "server_up": True,
          "router_up": True}, "local"),
        ({"direct": False, "mode": "openai", "role": "worker", "server_up": False,
          "router_up": True}, "openai"),
        ({"direct": False, "mode": "custom", "role": "worker", "server_up": False,
          "router_up": True}, "custom"),
    ],
)
def test_current_preset_reports_the_matching_preset(current, expected_name):
    detected = presets.current_preset(current, UPSTREAMS)

    assert detected == expected_name


def test_router_stopped_issue_fires_when_claude_is_routed_but_router_down():
    state = dict(
        mode="claude", direct=False, router_up=False,
        llama_installed=True, server_up=False, key_missing=False,
    )

    issues = _ids_and_fixes(presets.issues(state))

    assert issues == [
        {"id": "router_stopped", "fix": "apply-preset claude"},
    ]


def test_router_needed_issue_fires_when_upstream_is_routed_but_router_down():
    state = dict(
        mode="openai", direct=False, router_up=False,
        llama_installed=True, server_up=False, key_missing=False,
    )

    issues = _ids_and_fixes(presets.issues(state))

    assert issues == [
        {"id": "router_needed", "fix": "apply-preset openai"},
    ]


def test_llama_missing_issue_fires_when_local_mode_and_llama_absent():
    state = dict(
        mode="local", direct=False, router_up=True,
        llama_installed=False, server_up=False, key_missing=False,
    )

    issues = _ids_and_fixes(presets.issues(state))

    assert {"id": "llama_missing", "fix": "download-llama"} in issues


def test_server_down_issue_fires_when_local_llama_installed_but_server_down():
    state = dict(
        mode="local", direct=False, router_up=True,
        llama_installed=True, server_up=False, key_missing=False,
    )

    issues = _ids_and_fixes(presets.issues(state))

    assert {"id": "server_down", "fix": "start-server"} in issues


def test_key_missing_issue_fires_when_upstream_needs_a_key():
    state = dict(
        mode="openai", direct=False, router_up=True,
        llama_installed=True, server_up=False, key_missing=True,
    )

    issues = _ids_and_fixes(presets.issues(state))

    assert {"id": "key_missing", "fix": ""} in issues


def test_healthy_claude_state_has_no_issues():
    issues = presets.issues(HEALTHY_CLAUDE_STATE)

    assert issues == []


def _ids_and_fixes(found: list[dict]) -> list[dict]:
    """Issues reduced to id and fix: the texts are for people and may change."""
    return [{"id": issue["id"], "fix": issue["fix"]} for issue in found]
