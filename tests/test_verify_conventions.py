"""verify also enforces the clean-code rules: a file whose tests pass but which breaks them goes back to the author."""
from __future__ import annotations

import sys

import pytest

from conftest import chat_json

GROQ = "https://api.groq.com"
PYTHON = sys.executable.replace("\\", "/")
NO_DOCSTRING = "```python\nVALUE = 1\n```"
CLEAN = '```python\n"""A module."""\nVALUE = 1\n```'


@pytest.fixture
def project(tmp_path, cfg):
    """Work inside tmp_path and allow the current Python as the check (the check itself always passes)."""
    cfg["routing"]["verify"]["allowed_commands"] = [PYTHON]
    (tmp_path / "check.py").write_text("print('original')\n", encoding="utf-8")
    return tmp_path


def delegate(make_router, cfg):
    return make_router(cfg).delegate("write check.py", kind="quick", verify=f"{PYTHON} check.py",
                                     verify_target="check.py")


def test_passing_tests_are_not_enough_when_the_file_breaks_the_rules(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json(NO_DOCSTRING), chat_json(CLEAN))
    result = delegate(make_router, cfg)
    assert result["verify"]["passed"] and result["verify"]["fix_rounds"] == 1
    assert "clean-code rules" in mock.bodies[-1]["messages"][-1]["content"]
    assert "A module." in (project / "check.py").read_text(encoding="utf-8")


def test_a_file_that_never_gets_clean_is_restored(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json(NO_DOCSTRING))
    result = delegate(make_router, cfg)
    assert not result["verify"]["passed"]
    assert (project / "check.py").read_text(encoding="utf-8") == "print('original')\n"


def test_the_rules_can_be_switched_off(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["verify"]["conventions"] = False
    mock.on(GROQ, "/chat/completions", chat_json(NO_DOCSTRING))
    assert delegate(make_router, cfg)["verify"]["passed"]
