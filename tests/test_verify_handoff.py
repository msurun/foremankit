"""Tests for the handoff logic in ``foremankit.postcheck.verify_with_handoffs``.

The suite checks that a failing check can be handed off to a second model,
that handoffs can be disabled, that missing API keys prevent a handoff,
and that token usage from all attempts is aggregated.
"""

from __future__ import annotations

import sys

import pytest
from conftest import chat_json

# --------------------------------------------------------------------------- #
# Constants used by the tests
# --------------------------------------------------------------------------- #
GROQ = "https://api.groq.com"
CEREBRAS = "https://api.cerebras.ai"
PYTHON = sys.executable.replace("\\", "/")
FAILS = "```python\nraise SystemExit(1)\n```"
CLEAN = '```python\n"""A module."""\nVALUE = 1\n```'

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def project(tmp_path, cfg):
    """Work inside ``tmp_path`` and allow the current Python as the check command."""
    cfg["routing"]["verify"]["allowed_commands"] = [PYTHON]
    (tmp_path / "check.py").write_text("print('original')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def run_delegate(monkeypatch, make_router, cfg, mock):
    """Helper that runs ``Router.delegate`` with a quick chain containing Groq and Cerebras.

    Parameters ``handoffs`` and ``max_rounds`` control the verify behaviour.
    """
    def _run(*, handoffs: int = 1, max_rounds: int = 0, cerebras_key: bool = True):
        # set the required API keys
        monkeypatch.setenv("GROQ_API_KEY", "q")
        if cerebras_key:
            monkeypatch.setenv("CEREBRAS_API_KEY", "c")
        else:
            monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

        # routing configuration for the test
        cfg["routing"]["verify"]["conventions"] = False
        cfg["routing"]["verify"]["handoffs"] = handoffs
        cfg["routing"]["verify"]["max_rounds"] = max_rounds
        cfg["routing"]["chains"]["quick"] = ["groq", "cerebras"]

        # mock responses – one for each provider
        mock.on(GROQ, "/chat/completions", chat_json(FAILS))
        mock.on(CEREBRAS, "/chat/completions", chat_json(CLEAN))

        # perform the delegated call
        return make_router(cfg).delegate(
            "write check.py",
            kind="quick",
            verify=f"{PYTHON} check.py",
            verify_target="check.py",
        )
    return _run


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_handoff_success(project, cfg, mock, run_delegate):
    """A failing Groq check is handed off to Cerebras, which succeeds."""
    result = run_delegate(handoffs=1, max_rounds=0)

    assert result["verify"]["passed"]
    # Groq is the first source, Cerebras the second – handoff list should contain Groq only
    assert result["verify"]["handed_off_from"] == ["groq"]
    # the last mocked call must be to Cerebras
    assert mock.calls[-1].startswith(CEREBRAS)

    # the file should now contain the cleaned code (docstring + VALUE)
    file_text = (project / "check.py").read_text(encoding="utf-8")
    assert '"""A module."""' in file_text
    assert "VALUE = 1" in file_text
    # ensure no stray characters from the previous content remain after the code block
    assert file_text.endswith("VALUE = 1\n")


def test_no_handoff_when_handoffs_zero(project, cfg, mock, run_delegate):
    """When ``handoffs`` is 0 the failing Groq attempt is not handed off."""
    result = run_delegate(handoffs=0, max_rounds=0)

    assert not result["verify"]["passed"]
    assert result["verify"]["handed_off_from"] == []
    # the original file must be restored unchanged
    assert (project / "check.py").read_text(encoding="utf-8") == "print('original')\n"


def test_handoff_fails_without_second_api_key(run_delegate, project, cfg, mock):
    """If the second provider's API key is missing the handoff is not performed."""
    result = run_delegate(handoffs=1, cerebras_key=False)

    assert not result["verify"]["passed"]
    assert result["verify"]["handed_off_from"] == []
    # file must be restored to the original content
    assert (project / "check.py").read_text(encoding="utf-8") == "print('original')\n"


def test_costs_are_folded(project, cfg, mock, run_delegate):
    """The token usage from both attempts is summed into the final result."""
    result = run_delegate(handoffs=1, max_rounds=0)

    # each mock response uses the default 100 prompt tokens and 50 completion tokens
    # two calls → 200 prompt tokens total (the test checks the summed ``tokens_in`` field)
    assert result["tokens_in"] == 200
    # the output tail should reflect the successful Cerebras run, not the Groq failure
    assert "The tests pass" not in result["verify"]["output_tail"]
