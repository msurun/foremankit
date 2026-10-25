"""Tests for foremankit.upstream: mode persistence, alias mapping, model lookup,
repository sensitivity detection and upstream switch policy checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from foremankit import upstream

API_KEY_ENV = "FOREMANKIT_TEST_API_KEY"


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the state directory at a per-test temporary folder."""
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    return tmp_path


def test_active_mode_defaults_to_claude_when_no_file_exists(tmp_path: Path) -> None:
    """Without a mode file the default mode is returned."""
    assert upstream.active_mode() == "claude"


def test_active_mode_reads_the_persisted_value(tmp_path: Path) -> None:
    """A mode file written by set_mode is read back."""
    upstream.set_mode("local")

    assert upstream.active_mode() == "local"


def test_active_mode_falls_back_when_the_file_is_empty(tmp_path: Path) -> None:
    """An empty mode file yields the default instead of an empty string."""
    (tmp_path / upstream.MODE_FILE).write_text("   \n", encoding="utf-8")

    assert upstream.active_mode() == "claude"


def test_set_mode_round_trip_preserves_the_name(tmp_path: Path) -> None:
    """Writing a mode and reading it back returns the same value."""
    upstream.set_mode("mistral")

    assert upstream.active_mode() == "mistral"


@pytest.mark.parametrize(
    ("model", "expected_alias"),
    [
        ("claude-opus-5-5", "opus"),
        ("claude-haiku-5-5", "haiku"),
        ("claude-sonnet-5-5", "sonnet"),
        ("something-else", "sonnet"),
    ],
)
def test_alias_of_maps_model_identifiers(model: str, expected_alias: str) -> None:
    """Known families map to their alias; unknown models fall back to sonnet."""
    assert upstream.alias_of(model) == expected_alias


def test_upstream_model_returns_the_configured_name() -> None:
    """The configured model name for the detected alias is returned."""
    config = {"models": {"opus": "big-upstream-model", "sonnet": "small-upstream-model"}}

    assert upstream.upstream_model(config, "claude-opus-5-5") == "big-upstream-model"


def test_upstream_model_falls_back_to_sonnet_when_alias_missing() -> None:
    """When the alias is absent the sonnet entry is used instead."""
    config = {"models": {"sonnet": "small-upstream-model"}}

    assert upstream.upstream_model(config, "claude-haiku-5-5") == "small-upstream-model"


def test_repo_sensitivity_finds_marker_in_parent_from_nested_child(tmp_path: Path) -> None:
    """A marker in a parent directory is found from a deeply nested child."""
    marker = tmp_path / upstream.SENSITIVITY_FILE
    marker.parent.mkdir(parents=True)
    marker.write_text("internal", encoding="utf-8")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    assert upstream.repo_sensitivity(nested) == "internal"


def test_repo_sensitivity_defaults_to_private_without_marker(tmp_path: Path) -> None:
    """No marker anywhere means the most restrictive level, private."""
    assert upstream.repo_sensitivity(tmp_path) == "private"


def test_repo_sensitivity_defaults_to_private_for_invalid_text(tmp_path: Path) -> None:
    """An unrecognised level fails closed to private."""
    marker = tmp_path / upstream.SENSITIVITY_FILE
    marker.parent.mkdir(parents=True)
    marker.write_text("top-secret", encoding="utf-8")

    assert upstream.repo_sensitivity(tmp_path) == "private"


@pytest.mark.parametrize(
    ("ceiling", "level", "expected_allowed"),
    [
        ("internal", "internal", True),
        ("internal", "normal", True),
        ("internal", "private", False),
        ("everything", "normal", False),
    ],
)
def test_allowed_compares_level_against_ceiling(
    ceiling: str, level: str, expected_allowed: bool
) -> None:
    """A level is allowed at or below the ceiling; unknown levels are refused."""
    assert upstream.allowed(ceiling, level) is expected_allowed


@pytest.mark.parametrize("mode", ["claude", "local"])
def test_check_switch_allows_builtin_modes(mode: str, tmp_path: Path) -> None:
    """Builtin modes need no upstream configuration."""
    assert upstream.check_switch({}, mode, tmp_path) is None


def test_check_switch_lists_valid_modes_for_unknown_mode(tmp_path: Path) -> None:
    """An unknown mode produces an error that names the valid choices."""
    config = {"upstreams": {"mistral": {"key_env": API_KEY_ENV, "ceiling": "normal"}}}

    message = upstream.check_switch(config, "unknown-mode", tmp_path)

    assert message is not None
    assert "claude" in message
    assert "mistral" in message


def test_check_switch_reports_missing_key_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing API key yields an error naming the environment variable."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    config = {"upstreams": {"mistral": {"key_env": API_KEY_ENV, "ceiling": "normal"}}}

    message = upstream.check_switch(config, "mistral", tmp_path)

    assert message is not None
    assert API_KEY_ENV in message


def test_check_switch_reports_missing_url_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared but unset url_env yields an error naming that variable."""
    monkeypatch.setenv(API_KEY_ENV, "secret")
    monkeypatch.delenv("FOREMANKIT_TEST_URL", raising=False)
    config = {
        "upstreams": {
            "mistral": {
                "url_env": "FOREMANKIT_TEST_URL",
                "key_env": API_KEY_ENV,
                "ceiling": "private",
            }
        }
    }

    message = upstream.check_switch(config, "mistral", tmp_path)

    assert message is not None
    assert "FOREMANKIT_TEST_URL" in message


def test_check_switch_refuses_private_repo_with_internal_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private repository is refused by an upstream capped at internal."""
    monkeypatch.setenv(API_KEY_ENV, "secret")
    marker = tmp_path / upstream.SENSITIVITY_FILE
    marker.parent.mkdir(parents=True)
    marker.write_text("private", encoding="utf-8")
    config = {"upstreams": {"mistral": {"key_env": API_KEY_ENV, "ceiling": "internal"}}}

    message = upstream.check_switch(config, "mistral", tmp_path)

    assert message is not None
    assert "internal" in message


def test_check_switch_allows_internal_repo_with_internal_ceiling_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An internal repository is allowed by an internal ceiling when the key is set."""
    monkeypatch.setenv(API_KEY_ENV, "secret")
    marker = tmp_path / upstream.SENSITIVITY_FILE
    marker.parent.mkdir(parents=True)
    marker.write_text("internal", encoding="utf-8")
    config = {"upstreams": {"mistral": {"key_env": API_KEY_ENV, "ceiling": "internal"}}}

    assert upstream.check_switch(config, "mistral", tmp_path) is None
