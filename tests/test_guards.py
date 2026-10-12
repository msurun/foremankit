"""Cost guards (per-call cap, big-answer reasoning cap, cut-off detection) and the verify-and-fix loop."""
from __future__ import annotations

import sys

import pytest

from foremankit import verify
from foremankit import reasoning
from foremankit.formatting import format_result
from foremankit.router import RouterError, TaskTooLarge
from conftest import chat_json

GEMINI = "https://generativelanguage.googleapis.com"
DEEPSEEK = "https://api.deepseek.com"
GROQ = "https://api.groq.com"


def cut_off_in_code(text="```python\nprint('half"):
    """A reply that hit the token limit with a code block still open."""
    return {"choices": [{"message": {"content": text}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20000}}


# ---------------------------------------------------------------- per-call cost cap

def test_a_paid_call_above_the_per_call_cap_is_refused_unless_allowed(monkeypatch, make_router, mock):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(GEMINI, "/chat/completions", chat_json("done"))
    router = make_router()
    with pytest.raises(RouterError, match="per-call cap"):
        router.delegate("big", kind="hard_code", max_output_tokens=16000)      # ~$0.24 worst case
    assert not any("chat/completions" in call for call in mock.calls)
    assert router.delegate("big", kind="hard_code", max_output_tokens=16000, allow_cost=True)["provider"] == "gemini"


def test_free_providers_are_not_subject_to_the_per_call_cap(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json("free answer"))
    assert make_router().delegate("big", kind="quick", max_output_tokens=16000)["provider"] == "groq"


# ---------------------------------------------------------------- big answers

def test_a_big_requested_answer_lowers_the_reasoning_level(make_router):
    router = make_router()
    assert reasoning.cap_for_large_output(router.routing, "high", 4000) == "high"
    assert reasoning.cap_for_large_output(router.routing, "high", 16000) == "medium"
    assert reasoning.cap_for_large_output(router.routing, "low", 16000) == "low"          # never raised
    assert reasoning.cap_for_large_output(router.routing, None, 16000) is None


def test_a_big_requested_answer_comes_with_a_split_the_task_note(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json())
    result = make_router().delegate("big", kind="quick", max_output_tokens=12000)
    assert any("prefer tasks with an answer under" in note for note in result["notes"])
    assert "NOTE:" in format_result(result)


# ---------------------------------------------------------------- cut-off and empty answers from paid models

def test_an_expensive_cut_off_answer_stops_the_chain_instead_of_paying_twice(monkeypatch, make_router, mock):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(DEEPSEEK, "/chat/completions", cut_off_in_code())
    mock.on(GEMINI, "/chat/completions", chat_json("never asked"))
    with pytest.raises(TaskTooLarge, match="cut off inside a code block"):
        make_router().delegate("big", kind="data_script", max_output_tokens=4000)
    assert not any(call.startswith(GEMINI) for call in mock.calls)


def test_a_cheap_cut_off_answer_moves_on_to_the_next_source(monkeypatch, make_router, mock, cfg):
    cfg["routing"]["calls"]["abort_waste_usd"] = 100                             # waste is never "expensive" here
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(DEEPSEEK, "/chat/completions", cut_off_in_code())
    mock.on(GEMINI, "/chat/completions", chat_json("complete answer"))
    result = make_router(cfg).delegate("big", kind="data_script", max_output_tokens=4000)
    assert result["provider"] == "gemini" and any("cut off inside a code block" in s for s in result["skipped"])


def test_a_cut_off_answer_with_closed_code_is_still_a_result(monkeypatch, make_router, mock):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", cut_off_in_code("```python\nx = 1\n```\nand then it stopped mid-sentence"))
    result = make_router().delegate("t", kind="quick")
    assert result["provider"] == "groq" and result["truncated"] is True


# ---------------------------------------------------------------- verify and fix

PYTHON = sys.executable.replace("\\", "/")
GOOD = "```python\nassert 1 + 1 == 2\n```"
BAD = "```python\nassert 1 + 1 == 3, 'arithmetic is broken'\n```"


@pytest.fixture
def project(tmp_path, cfg):
    """Work inside tmp_path (conftest chdirs there) and allow running the current Python as the check."""
    cfg["routing"]["verify"]["allowed_commands"] = [f"{PYTHON}"]
    cfg["routing"]["verify"]["conventions"] = False   # these tests check the loop itself; see test_verify_conventions.py
    (tmp_path / "check.py").write_text("print('original')\n", encoding="utf-8")
    return tmp_path


def run_verified(make_router, cfg, **kwargs):
    return make_router(cfg).delegate("write check.py", kind="quick", verify=f"{PYTHON} check.py",
                                     verify_target="check.py", **kwargs)


def test_a_passing_answer_replaces_the_file(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json(GOOD))
    result = run_verified(make_router, cfg)
    assert result["verify"]["passed"] and result["verify"]["fix_rounds"] == 0
    assert (project / "check.py").read_text(encoding="utf-8") == "assert 1 + 1 == 2\n"
    assert "verify: PASSED on the first try" in format_result(result)


def test_a_failing_answer_is_sent_back_and_the_fixed_one_is_kept(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json(BAD), chat_json(GOOD))
    result = run_verified(make_router, cfg)
    assert result["verify"]["passed"] and result["verify"]["fix_rounds"] == 1
    feedback = mock.bodies[-1]["messages"][-1]["content"]
    assert "arithmetic is broken" in feedback and "check.py" in feedback          # the error output went back to the model
    assert (project / "check.py").read_text(encoding="utf-8") == "assert 1 + 1 == 2\n"
    assert result["tokens_out"] == 100                                           # both calls are counted (50 + 50)


def test_a_check_that_keeps_failing_restores_the_original_file(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json(BAD))
    result = run_verified(make_router, cfg)
    assert not result["verify"]["passed"] and result["verify"]["fix_rounds"] == cfg["routing"]["verify"]["max_rounds"]
    assert (project / "check.py").read_text(encoding="utf-8") == "print('original')\n"
    assert "verify: FAILED" in format_result(result) and "was restored" in format_result(result)


def test_a_new_file_that_fails_the_check_is_removed(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json(BAD))
    make_router(cfg).delegate("t", kind="quick", verify=f"{PYTHON} fresh.py", verify_target="fresh.py")
    assert not (project / "fresh.py").exists()


def test_an_answer_without_code_leaves_the_file_alone(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json("I cannot do that."))
    result = run_verified(make_router, cfg)
    assert not result["verify"]["passed"] and "no code block" in result["verify"]["output_tail"]
    assert (project / "check.py").read_text(encoding="utf-8") == "print('original')\n"


def test_a_command_that_is_not_allowed_is_refused_before_any_model_is_paid(project, make_router, mock, cfg):
    with pytest.raises(verify.VerifyError, match="command not allowed"):
        make_router(cfg).delegate("t", kind="quick", verify="rm -rf .", verify_target="check.py")
    assert mock.calls == []


@pytest.mark.parametrize("target", ["../outside.py", "/etc/passwd", ".env", "sub/.env.local", ".git/config"])
def test_a_target_outside_the_project_or_protected_is_refused(project, make_router, cfg, target):
    with pytest.raises(verify.VerifyError):
        make_router(cfg).delegate("t", kind="quick", verify=f"{PYTHON} x.py", verify_target=target)


def test_verify_and_verify_target_must_come_together(make_router):
    with pytest.raises(RouterError, match="go together"):
        make_router().delegate("t", kind="quick", verify="pytest")


def test_the_answer_block_named_after_the_target_is_chosen(tmp_path):
    answer = "### other.py\n```python\nprint('other, much longer than the wanted file ........')\n```\n### check.py\n```python\nprint('wanted')\n```"
    assert verify.extract_code(answer, "check.py") == "print('wanted')\n"
    assert verify.extract_code("no code here", "check.py") is None


def test_a_check_that_cannot_start_ends_the_loop_without_asking_the_model_again(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["verify"]["allowed_commands"] = ["no-such-program"]
    mock.on(GROQ, "/chat/completions", chat_json(GOOD))
    result = make_router(cfg).delegate("t", kind="quick", verify="no-such-program check.py", verify_target="check.py")
    assert not result["verify"]["passed"] and result["verify"]["fix_rounds"] == 0
    assert "could not be started" in result["verify"]["output_tail"]
    assert sum("chat/completions" in call for call in mock.calls) == 1
    assert (project / "check.py").read_text(encoding="utf-8") == "print('original')\n"


def test_a_program_path_with_a_slash_is_taken_relative_to_the_project(tmp_path):
    suffix = ".exe" if verify.os.name == "nt" else ""
    assert verify.resolve_executable(".venv/Scripts/python", tmp_path) == str(tmp_path / ".venv/Scripts/python") + suffix
    assert verify.resolve_executable("definitely-not-installed", tmp_path) == "definitely-not-installed"


def test_a_file_that_contains_code_fences_inside_a_string_is_extracted_whole():
    inner = 'answer = """text\n```python\nx = 1\n```\nmore"""\nprint(answer)\n'
    reply = "Here it is:\n```python\n" + inner + "```\nDone."
    assert verify.extract_code(reply, "t.py") == inner


def test_an_answer_cut_off_inside_a_block_still_yields_the_text_so_far():
    assert verify.extract_code("```python\nprint(1)\n", "t.py") == "print(1)\n"
