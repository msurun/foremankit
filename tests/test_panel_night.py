"""Tests for the night-run switch of the panel: status, on/off, command line and language."""
from __future__ import annotations

import argparse
import time

import pytest

from foremankit import night_run, panel


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Point both modules at a temporary state folder and make this computer the driver."""
    monkeypatch.setattr(panel, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(night_run, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(panel, "local_drives", lambda: True)
    monkeypatch.setenv("FOREMANKIT_LANG", "en")
    return tmp_path


def test_status_is_off_without_state(state):
    status = panel.night_status()
    assert status["enabled"] is False
    assert status["line"] == "Night run: OFF"
    assert status["driver_local"] is True


def test_on_sets_deadline_and_limit(state):
    message = panel.set_night(True, hours=2, max_continues=5)
    status = panel.night_status()
    assert status["enabled"] is True
    assert status["max_continues"] == 5
    assert abs(status["until"] - (time.time() + 7200)) < 60
    assert status["line"].startswith("Night run: ON, 0/5, until ")
    assert "up to 5 continues" in message


def test_on_removes_stale_done_file(state):
    done = state / panel.NIGHT_DONE_FILE
    done.write_text("old", encoding="utf-8")
    panel.set_night(True)
    assert not done.exists()


def test_off_disables_night_run(state):
    panel.set_night(True)
    assert panel.set_night(False) == "night run off"
    assert panel.night_status()["enabled"] is False


def test_rejects_non_positive_values(state):
    with pytest.raises(ValueError):
        panel.set_night(True, hours=0)
    with pytest.raises(ValueError):
        panel.set_night(True, max_continues=0)


def test_polish_texts_when_language_is_pl(state, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_LANG", "pl")
    assert panel.night_status()["line"] == "Nocny przebieg: WYŁ."
    assert panel.night_status()["texts"]["title"] == "Nocny przebieg"


def test_unknown_language_falls_back_to_english(state, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_LANG", "de")
    assert panel.night_lang() == "en"


def test_command_line_on_off_status(state):
    parser = panel.build_parser()
    on = parser.parse_args(["night-run", "on", "--hours", "3", "--max", "7"])
    assert panel.night_command(on).startswith("night run on until")
    assert panel.night_command(parser.parse_args(["night-run", "status"])).startswith("Night run: ON, 0/7")
    assert panel.night_command(parser.parse_args(["night-run", "off"])) == "night run off"


def test_command_line_defaults(state):
    args = panel.build_parser().parse_args(["night-run", "on"])
    assert (args.hours, args.max_continues) == (panel.NIGHT_HOURS, panel.NIGHT_MAX_CONTINUES)


def test_command_line_rejects_unknown_action(state):
    args = argparse.Namespace(action="maybe", hours=8, max_continues=20)
    with pytest.raises(ValueError):
        panel.night_command(args)


def test_config_action_routes_night_run(state):
    args = panel.build_parser().parse_args(["night-run", "on", "--hours", "1"])
    assert panel.run_config_action(args).startswith("night run on until")
