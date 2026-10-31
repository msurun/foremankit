"""Tests for foremankit.supervise (the Antigravity supervisor) and the supervisor switch."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from foremankit import supervise as sup
from foremankit.delegation import SUPERVISORS, set_supervisor, supervisor_mode
from foremankit.state import StateStore

CHECK = ".venv/Scripts/python -m pytest -q"


@pytest.fixture(autouse=True)
def internal_ok(monkeypatch):
    monkeypatch.setenv("FOREMANKIT_ALLOW_INTERNAL_TRAINING", "agy")


def fake_agy(reply: str = "PASS", seen: list | None = None, side_effect=None):
    """A runner standing in for agy.exe: records the command, optionally acts in the cwd, prints agy's JSON."""
    def runner(command, **kwargs):
        if seen is not None:
            seen.append((command, kwargs))
        if side_effect:
            side_effect(kwargs["cwd"])
        out = json.dumps({"status": "SUCCESS", "response": reply})
        return SimpleNamespace(stdout=out, stderr="", returncode=0)
    return runner


def job(tmp_path: Path, **changes) -> sup.SuperviseJob:
    base = dict(task="Write x.py", target=str(tmp_path / "x.py"), check=CHECK, project=tmp_path)
    return sup.SuperviseJob(**{**base, **changes})


def stub_check(monkeypatch, passed: bool, tail: str = "1 passed") -> list:
    calls: list = []
    monkeypatch.setattr(sup, "run_command", lambda tokens, project, timeout: calls.append(tokens) or (passed, tail))
    return calls


def test_a_passing_check_gives_passed_true_and_the_reply(tmp_path, monkeypatch):
    stub_check(monkeypatch, True)
    result = sup.supervise(job(tmp_path), runner=fake_agy("PASS all green"))
    assert result["passed"] is True and result["agy_reply"] == "PASS all green"
    assert result["check_output_tail"] == "1 passed" and result["target"].endswith("x.py")


def test_agy_saying_pass_is_not_trusted_when_our_check_fails(tmp_path, monkeypatch):
    stub_check(monkeypatch, False, "1 failed")
    result = sup.supervise(job(tmp_path), runner=fake_agy("PASS"))
    assert result["passed"] is False and "1 failed" in result["check_output_tail"]


def test_the_command_has_model_flags_and_runs_in_the_project(tmp_path, monkeypatch):
    stub_check(monkeypatch, True)
    seen: list = []
    sup.supervise(job(tmp_path), runner=fake_agy(seen=seen))
    command, kwargs = seen[0]
    assert "--dangerously-skip-permissions" in command and "--sandbox" in command
    assert command[command.index("--model") + 1] == "gemini-3.8-flash-high"
    assert kwargs["cwd"] == tmp_path and CHECK in command[2] and str(tmp_path / "x.py") in command[2]


def test_private_data_is_refused_before_agy_starts(tmp_path):
    seen: list = []
    with pytest.raises(sup.SuperviseError, match="private"):
        sup.supervise(job(tmp_path, sensitivity="private"), runner=fake_agy(seen=seen))
    assert seen == []


def test_internal_data_needs_the_per_machine_acceptance(tmp_path, monkeypatch):
    monkeypatch.delenv("FOREMANKIT_ALLOW_INTERNAL_TRAINING")
    with pytest.raises(sup.SuperviseError, match="internal"):
        sup.supervise(job(tmp_path, sensitivity="internal"), runner=fake_agy())


def test_a_check_command_outside_the_allow_list_is_refused(tmp_path):
    with pytest.raises(sup.SuperviseError, match="not allowed"):
        sup.supervise(job(tmp_path, check="rm -rf ."), runner=fake_agy())


def test_a_target_outside_the_project_is_refused(tmp_path):
    outside = str(tmp_path.parent / "elsewhere.py")
    with pytest.raises(sup.SuperviseError, match="outside"):
        sup.supervise(job(tmp_path, target=outside), runner=fake_agy())


def test_a_long_task_goes_into_a_file_that_is_removed_afterwards(tmp_path, monkeypatch):
    stub_check(monkeypatch, True)
    existed: list = []
    seen: list = []
    runner = fake_agy(seen=seen, side_effect=lambda cwd: existed.append((tmp_path / sup.TASK_FILE).exists()))
    sup.supervise(job(tmp_path, task="x" * 30000), runner=runner)
    assert existed == [True] and not (tmp_path / sup.TASK_FILE).exists()
    assert len(seen[0][0][2]) < 3000


def test_an_agy_timeout_still_runs_our_check(tmp_path, monkeypatch):
    calls = stub_check(monkeypatch, False, "nothing written")

    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1)
    result = sup.supervise(job(tmp_path), runner=runner)
    assert result["passed"] is False and "timed out" in result["agy_reply"] and len(calls) == 1


def test_a_missing_agy_binary_is_a_clear_error(tmp_path, monkeypatch):
    stub_check(monkeypatch, True)

    def runner(command, **kwargs):
        raise FileNotFoundError(command[0])
    with pytest.raises(sup.SuperviseError, match="not found"):
        sup.supervise(job(tmp_path), runner=runner)


def test_non_json_output_is_reported_not_raised(tmp_path, monkeypatch):
    stub_check(monkeypatch, True)

    def runner(command, **kwargs):
        return SimpleNamespace(stdout="boom", stderr="", returncode=3)
    assert "no JSON" in sup.supervise(job(tmp_path), runner=runner)["agy_reply"]


def test_the_supervisor_switch_defaults_to_haiku_and_round_trips(tmp_path):
    store = StateStore(tmp_path / "state.json")
    assert supervisor_mode(store.load()) == "haiku"
    assert set_supervisor(store, "agy") == "agy" and supervisor_mode(store.load()) == "agy"
    set_supervisor(store, "haiku")
    assert supervisor_mode(store.load()) == "haiku" and SUPERVISORS == ("haiku", "agy")


def test_a_bad_supervisor_value_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        set_supervisor(StateStore(tmp_path / "state.json"), "opus")


def test_the_panel_exposes_and_changes_the_supervisor(tmp_path, monkeypatch):
    from foremankit import panel
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    assert panel.supervisor_choice() == "haiku"
    assert panel.set_supervisor_choice("agy") == "supervisor: agy" and panel.supervisor_choice() == "agy"
    assert panel.set_supervisor_choice("x").startswith("not done")
