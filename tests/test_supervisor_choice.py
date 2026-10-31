"""Tests for the auto supervisor: the choice heuristic, its routing.yaml settings, the CLI `supervisor choose`, the mode."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from foremankit import cli
from foremankit.delegation import set_supervisor, supervisor_mode
from foremankit.state import StateStore
from foremankit.supervisor_choice import AutoSettings, TaskFacts, auto_settings, choose_supervisor, decide_supervisor

ONE_FILE_CHECK = ".venv/Scripts/python -m pytest -q tests/test_x.py"
THREE_FILES = ("a.py", "b.py", "c.py")


def choose(task="Write tests for x", targets=("tests/test_x.py",), check=ONE_FILE_CHECK, settings=None, **options):
    """The decision for one task with the default thresholds, or with `settings`."""
    return decide_supervisor(TaskFacts(task, list(targets), check, **options), settings)


def test_the_plain_function_has_the_documented_signature_and_defaults():
    assert choose_supervisor("t", ["a.py"], ONE_FILE_CHECK) == ("haiku", "small single-file task")
    assert choose_supervisor("t", list(THREE_FILES), ONE_FILE_CHECK, 0, "normal", True)[0] == "agy"
    assert choose_supervisor("t", list(THREE_FILES), ONE_FILE_CHECK, sensitivity="private")[0] == "haiku"


def test_a_small_single_file_task_goes_to_haiku():
    assert choose() == ("haiku", "small single-file task")


def test_private_data_always_goes_to_haiku_even_when_the_task_is_heavy():
    who, reason = choose(targets=THREE_FILES, sensitivity="private")
    assert who == "haiku" and "private" in reason


def test_a_switched_off_agy_means_haiku_even_when_the_task_is_heavy():
    assert choose(targets=THREE_FILES, agy_enabled=False) == ("haiku", "agy is switched off")


def test_three_or_more_target_files_go_to_agy():
    who, reason = choose(targets=THREE_FILES)
    assert who == "agy" and "multi-file" in reason


def test_two_targets_alone_stay_with_haiku():
    assert choose(targets=("a.py", "b.py")) == ("haiku", "small single-file task")


def test_three_targets_alone_go_to_agy():
    who, reason = choose(targets=THREE_FILES)
    assert who == "agy" and "multi-file" in reason


@pytest.mark.parametrize("check", [
    ".venv/Scripts/python -m pytest -q",                              # no path: the whole suite
    ".venv/Scripts/python -m pytest -q tests",                        # a directory
    ".venv/Scripts/python -m pytest -q tests/test_a.py tests/test_b.py",
    "pytest -q -k smoke tests/",                                      # -k takes a value, the directory is the path
])
def test_a_pytest_check_over_a_directory_or_several_files_goes_to_agy(check):
    who, reason = choose(targets=("a.py", "b.py"), check=check)
    assert who == "agy" and "test-heavy" in reason


def test_a_test_heavy_check_with_one_target_stays_with_haiku():
    check = ".venv/Scripts/python -m pytest -q tests"
    assert choose(targets=("a.py",), check=check) == ("haiku", "small single-file task")


def test_a_test_heavy_check_with_two_targets_goes_to_agy():
    who, reason = choose(targets=("a.py", "b.py"), check=".venv/Scripts/python -m pytest -q tests")
    assert who == "agy" and "test-heavy" in reason


@pytest.mark.parametrize("check", [
    ONE_FILE_CHECK,
    ".venv/Scripts/python -m pytest -q tests/test_x.py::test_one",
    ".venv/Scripts/python -m pytest -q -k smoke tests/test_x.py",
    "foremankit check foremankit/x.py",                               # not pytest at all
    "ruff check foremankit/x.py",
])
def test_a_single_file_check_stays_with_haiku(check):
    assert choose(check=check)[0] == "haiku"


def test_a_large_context_or_a_long_task_goes_to_agy():
    assert choose(context_lines=800) == ("agy", "large task")
    assert choose(task="x" * 8000) == ("agy", "large task")
    assert choose(context_lines=799, task="x" * 7999)[0] == "haiku"


@pytest.mark.parametrize("task", ["Refactor the loader", "migrate to the new API", "Rename across the package", "Update ALL MODULES",
                                  "Popraw wszystkie pliki", "Zrób refaktoryzację", "migracja konfiguracji", "zmień we wszystkich testach"])
def test_broad_change_keywords_in_english_and_polish_go_to_agy(task):
    who, reason = choose(task=task, targets=("a.py", "b.py"))
    assert who == "agy" and "broad change" in reason


def test_a_broad_change_keyword_alone_on_single_small_task_stays_with_haiku():
    assert choose(task="Refactor the loader") == ("haiku", "small single-file task")


def test_a_broad_change_keyword_with_two_targets_goes_to_agy():
    who, reason = choose(task="Refactor the loader", targets=("a.py", "b.py"))
    assert who == "agy" and "broad change" in reason


def test_the_first_matching_rule_names_the_reason():
    reason = choose(targets=THREE_FILES, check=".venv/Scripts/python -m pytest -q", task="refactor")[1]
    assert reason.startswith("multi-file")


def test_routing_settings_override_the_defaults():
    block = {"min_targets": 3, "min_context_lines": 50, "min_task_chars": 10, "keywords": ["overhaul"]}
    settings = auto_settings({"supervisor": {"auto": block}})
    assert settings == AutoSettings(3, 50, 10, ("overhaul",))
    assert choose(targets=("a.py", "b.py"), settings=AutoSettings(min_targets=3), task="ok")[0] == "haiku"
    assert choose(task="Write tests for x", settings=settings)[0] == "agy"                   # longer than 10 characters
    assert choose(settings=AutoSettings(keywords=("overhaul",)), task="Overhaul it", targets=("a.py", "b.py"))[0] == "agy"
    assert choose(settings=AutoSettings(keywords=("overhaul",)), task="refactor it", targets=("a.py", "b.py"))[0] == "haiku"


def test_a_missing_or_empty_settings_block_gives_the_defaults():
    assert auto_settings(None) == AutoSettings() == auto_settings({"supervisor": None}) == auto_settings({"supervisor": {"auto": {}}})


# --- CLI ---------------------------------------------------------------------------------------------------------------

def router_for(tmp_path, agy_cfg=None, toggles=None):
    """A stand-in router: a real state store, the agy provider config and an empty routing."""
    store = StateStore(tmp_path / "state.json")
    if toggles:
        store.update(lambda state: state.setdefault("toggles", {}).update(toggles))
    providers = {"agy": SimpleNamespace(cfg=agy_cfg if agy_cfg is not None else {"enabled": True})}
    return SimpleNamespace(store=store, providers=providers, routing={})


def args_for(tmp_path, task="Write tests", **changes):
    """The parsed arguments of `supervisor choose`, with a task file in tmp_path."""
    task_file = tmp_path / "task.md"
    task_file.write_text(task, encoding="utf-8")
    base = dict(choice="choose", task_file=str(task_file), target=["tests/test_x.py"], check=ONE_FILE_CHECK,
                context_lines=0, sensitivity="normal")
    return argparse.Namespace(**{**base, **changes})


def test_choose_prints_one_line_and_exits_zero(tmp_path, capsys):
    router = router_for(tmp_path, toggles={"delegation:agy": "on"})
    assert cli.cmd_supervisor(router, args_for(tmp_path, target=list(THREE_FILES))) == 0
    assert capsys.readouterr().out.strip() == "agy multi-file (3 files)"
    assert cli.cmd_supervisor(router, args_for(tmp_path)) == 0
    assert capsys.readouterr().out.strip() == "haiku small single-file task"


def test_choose_says_haiku_while_the_agy_delegation_category_is_off(tmp_path, capsys):
    cli.cmd_supervisor(router_for(tmp_path), args_for(tmp_path, target=list(THREE_FILES)))
    assert capsys.readouterr().out.strip() == "haiku agy is switched off"


def test_choose_says_haiku_when_the_agy_provider_is_disabled_in_the_config(tmp_path, capsys):
    router = router_for(tmp_path, agy_cfg={"enabled": False}, toggles={"delegation:agy": "on"})
    cli.cmd_supervisor(router, args_for(tmp_path, target=list(THREE_FILES)))
    assert capsys.readouterr().out.strip() == "haiku agy is switched off"


def test_choose_never_picks_agy_for_private_data(tmp_path, capsys):
    router = router_for(tmp_path, toggles={"delegation:agy": "on"})
    cli.cmd_supervisor(router, args_for(tmp_path, target=list(THREE_FILES), sensitivity="private"))
    assert capsys.readouterr().out.startswith("haiku")


def test_choose_without_its_arguments_is_an_error(tmp_path, capsys):
    assert cli.cmd_supervisor(router_for(tmp_path), args_for(tmp_path, target=[])) == 2
    assert "needs --task-file" in capsys.readouterr().err


def test_choose_reads_the_thresholds_from_routing(tmp_path, capsys):
    router = router_for(tmp_path, toggles={"delegation:agy": "on"})
    router.routing = {"supervisor": {"auto": {"min_task_chars": 5}}}
    cli.cmd_supervisor(router, args_for(tmp_path, task="a longer task"))
    assert capsys.readouterr().out.strip() == "agy large task"


# --- the mode ------------------------------------------------------------------------------------------------------------

def test_the_mode_accepts_auto_and_the_cli_shows_it(tmp_path, capsys):
    store = StateStore(tmp_path / "state.json")
    assert set_supervisor(store, "auto") == "auto" and supervisor_mode(store.load()) == "auto"
    router = SimpleNamespace(store=store)
    cli.cmd_supervisor(router, argparse.Namespace(choice="status"))
    assert capsys.readouterr().out.strip() == "supervisor: auto"
    cli.cmd_supervisor(router, argparse.Namespace(choice="haiku"))
    assert supervisor_mode(store.load()) == "haiku"


def test_the_panel_accepts_auto(tmp_path, monkeypatch):
    from foremankit import panel
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    assert panel.set_supervisor_choice("auto") == "supervisor: auto" and panel.supervisor_choice() == "auto"


def test_the_shipped_routing_yaml_has_the_auto_block_with_the_default_values():
    from foremankit.config import load_config
    assert auto_settings(load_config()["routing"]) == AutoSettings()
