"""Pytest tests for the --local-orchestrator step of install.py (pure merge functions)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("install_script", Path(__file__).parent.parent / "install.py")
install = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install)
Installer = install.Installer
URL, MODEL = "http://127.0.0.1:8081", "claude-sonnet-5"


def names(settings: dict) -> list[str]:
    return [entry["name"] for entry in settings.get("claudeCode.environmentVariables", [])]


def test_settings_add_keeps_foreign_entries_and_is_idempotent():
    base = {"claudeCode.environmentVariables": [{"name": "MY_VAR", "value": "1"}], "editor.fontSize": 14}
    once = Installer.merged_vscode_settings(base, URL, MODEL)
    twice = Installer.merged_vscode_settings(once, URL, MODEL)
    assert once == twice
    assert "MY_VAR" in names(once) and "ANTHROPIC_BASE_URL" in names(once) and once["editor.fontSize"] == 14


def test_the_task_policy_is_only_added_together_with_a_folder_open_task():
    assert "task.allowAutomaticTasks" not in Installer.merged_vscode_settings({}, URL, MODEL)
    assert Installer.merged_vscode_settings({}, URL, MODEL, tasks=True)["task.allowAutomaticTasks"] == "auto"


def test_settings_do_not_override_an_existing_task_policy():
    merged = Installer.merged_vscode_settings({"task.allowAutomaticTasks": "off"}, URL, MODEL, tasks=True)
    assert merged["task.allowAutomaticTasks"] == "off"


def test_settings_remove_leaves_only_foreign_entries():
    added = Installer.merged_vscode_settings({"claudeCode.environmentVariables": [{"name": "MY_VAR", "value": "1"}]}, URL, MODEL)
    assert names(Installer.merged_vscode_settings(added, URL, MODEL, remove=True)) == ["MY_VAR"]
    only_ours = Installer.merged_vscode_settings({}, URL, MODEL)
    assert "claudeCode.environmentVariables" not in Installer.merged_vscode_settings(only_ours, URL, MODEL, remove=True)


def test_tasks_replace_same_label_and_keep_user_tasks():
    mine = [{"label": "ForemanKit: router 8081", "command": "new"}]
    base = {"tasks": [{"label": "ForemanKit: router 8081", "command": "old"}, {"label": "build", "command": "make"}]}
    merged = Installer.merged_vscode_tasks(base, mine)
    assert [t["command"] for t in merged["tasks"]] == ["make", "new"]
    assert Installer.merged_vscode_tasks(merged, mine) == merged
    assert [t["label"] for t in Installer.merged_vscode_tasks(merged, mine, remove=True)["tasks"]] == ["build"]


def test_stale_llama_server_task_is_dropped_but_user_tasks_stay():
    base = {"tasks": [{"label": "ForemanKit: llama-server 8080", "command": "old"}, {"label": "build", "command": "make"}]}
    merged = Installer.merged_vscode_tasks(base, [], drop_labels=("ForemanKit: llama-server 8080",))
    assert [t["label"] for t in merged["tasks"]] == ["build"]
