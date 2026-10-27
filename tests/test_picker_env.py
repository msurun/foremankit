"""Pytest tests for the picker env that install.py writes while the router is on (pure functions, no disk)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("install_script", Path(__file__).parent.parent / "install.py")
install = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install)
Installer = install.Installer
URL, MODEL = "http://127.0.0.1:8081", "claude-sonnet-5"
PICKER = {
    "SONNET": {"model": "claude-sonnet-5", "name": "DeepSeek Pro"},
    "HAIKU": {"model": "claude-haiku-5", "name": "DeepSeek Flash"},
    "OPUS": {"model": "claude-opus-5", "name": "Local LLM"},
    "FABLE": {"model": "claude-opus-4-5", "name": "Kaggle"},
    "custom": {"model": "claude-sonnet-4-5", "name": "Gemini Pro"},
}


def env_map(settings: dict) -> dict:
    """Flatten the claudeCode.environmentVariables list into a name -> value dict."""
    return {entry["name"]: entry["value"] for entry in settings.get("claudeCode.environmentVariables", [])}


def test_picker_env_maps_each_slot_to_model_and_name():
    env = install.picker_env(PICKER)
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-5"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME"] == "DeepSeek Flash"
    assert env["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "claude-opus-4-5"
    assert len([k for k in env if k.startswith("ANTHROPIC_DEFAULT_")]) == 8


def test_picker_env_maps_custom_to_its_name_only():
    env = install.picker_env({"custom": {"model": "claude-sonnet-4-5", "name": "Gemini Pro"}})
    assert env == {"ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Gemini Pro"}


def test_picker_settings_uses_custom_model_and_picker_env():
    model_id, env = install.picker_settings({"picker": PICKER, "local_models": ["claude-opus-5"]})
    assert model_id == "claude-sonnet-4-5"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-opus-5"


def test_picker_settings_falls_back_to_local_model_without_custom():
    model_id, env = install.picker_settings({"picker": {"SONNET": PICKER["SONNET"]}, "local_models": ["claude-opus-5"]})
    assert model_id == "claude-opus-5"
    assert "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME" not in env


def test_picker_settings_without_picker_section_gives_empty_env():
    model_id, env = install.picker_settings({"local_models": ["claude-opus-5"]})
    assert model_id == "claude-opus-5"
    assert env == {}


def test_extra_picker_entries_are_written_and_custom_name_overrides_default():
    _, env = install.picker_settings({"picker": PICKER})
    merged = Installer.merged_vscode_settings({}, URL, MODEL, extra=env)
    values = env_map(merged)
    assert values["ANTHROPIC_DEFAULT_SONNET_MODEL_NAME"] == "DeepSeek Pro"
    assert values["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] == "Gemini Pro"
    assert "ForemanKit (local / DeepSeek / Gemini)" not in values.values()


def test_default_custom_name_is_the_gateway_picker_name_without_extra():
    values = env_map(Installer.merged_vscode_settings({}, URL, MODEL))
    assert values["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] == install.PICKER_NAME


def test_remove_drops_gateway_and_slot_entries_and_keeps_foreign_ones():
    data = {"claudeCode.environmentVariables": [
        {"name": "MY_VAR", "value": "1"},
        {"name": "ANTHROPIC_BASE_URL", "value": URL},
        {"name": "ANTHROPIC_DEFAULT_SONNET_MODEL", "value": "claude-sonnet-5"},
    ]}
    merged = Installer.merged_vscode_settings(data, URL, MODEL, remove=True)
    assert env_map(merged) == {"MY_VAR": "1"}


def test_writing_again_replaces_slot_entries_without_duplicates():
    _, env = install.picker_settings({"picker": PICKER})
    once = Installer.merged_vscode_settings({}, URL, MODEL, extra=env)
    _, env2 = install.picker_settings({"picker": {"SONNET": {"model": "claude-sonnet-5", "name": "Renamed"}}})
    twice = Installer.merged_vscode_settings(once, URL, MODEL, extra=env2)
    names = [entry["name"] for entry in twice["claudeCode.environmentVariables"]]
    assert len(names) == len(set(names))
    assert env_map(twice)["ANTHROPIC_DEFAULT_SONNET_MODEL_NAME"] == "Renamed"
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in names
