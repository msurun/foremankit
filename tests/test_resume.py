"""Resuming cut-off answers (continue_from): the saved partial text and its request sidecar are found, the same task
and the same source are required, and the finished answer is written to a new id. A resume must reuse the partial
text instead of paying for the question again, yet must refuse to continue someone else's task or another source."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import chat_json

from foremankit.outputs import read_answer_text
from foremankit.records import RouterError
from foremankit.resume import prompt_fingerprint, resumed_id

GROQ = "https://api.groq.com"
DEEPSEEK = "https://api.deepseek.com"
GEMINI = "https://generativelanguage.googleapis.com"
FIRST_PART = "The first step is to open the valve and check the gauge before"
SECOND_PART = " the pump starts, then log the pressure."


def cut(text: str, prompt: int = 100, completion: int = 40) -> dict:
    """A reply that stopped at its token limit (finish_reason length) with plain prose."""
    return {"choices": [{"message": {"content": text}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def cut_off_in_code(text: str = "```python\nprint('half") -> dict:
    """A reply that hit the token limit with a code block still open (rejected by the caller)."""
    return {"choices": [{"message": {"content": text}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20000}}


def saved_text(path: str | Path) -> str:
    """The text of an answer file, without its front matter."""
    return Path(path).read_text(encoding="utf-8").split("---\n\n", 1)[1]


def sidecar(tmp_path: Path, answer_id: str) -> dict:
    """The parsed .request.json written next to a saved answer."""
    return json.loads((tmp_path / ".agent-out" / f"{answer_id}.request.json").read_text(encoding="utf-8"))


def lost_groq_key(monkeypatch) -> None:
    """Remove the key of the source that produced the partial answer (it is then unreachable)."""
    monkeypatch.delenv("GROQ_API_KEY")


# ------------------------------------------------------------------ (1) a cut-off answer is saved with its sidecar

def test_a_cut_off_answer_is_saved_with_a_request_sidecar(monkeypatch, make_router, mock, tmp_path, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART))
    result = make_router(cfg).delegate("explain", kind="quick")
    assert result["truncated"] is True
    assert (tmp_path / ".agent-out" / f"{result['id']}.md").is_file()
    assert sidecar(tmp_path, result["id"])["finish_reason"] == "length"
    assert sidecar(tmp_path, result["id"])["prompt_sha256"]


def test_the_sidecar_names_kind_sensitivity_and_the_source(monkeypatch, make_router, mock, tmp_path, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART))
    result = make_router(cfg).delegate("explain", kind="quick")
    record = sidecar(tmp_path, result["id"])
    assert record["kind"] == "quick" and record["sensitivity"] == "normal"
    assert record["chain_entry"] == "groq:main"


def test_the_truncated_result_points_at_continue_from_for_the_same_id(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART))
    result = make_router(cfg).delegate("explain", kind="quick")
    assert f'continue_from="{result["id"]}"' in format_result_of(result)


def format_result_of(result: dict) -> str:
    """The plain-text rendering of one delegation result (kept here to show the INCOMPLETE line)."""
    from foremankit.formatting import format_result
    return format_result(result)


# ------------------------------------------------------------------ (2) the resume finishes the answer

def test_a_resume_joins_the_partial_and_the_new_text_under_a_new_id(monkeypatch, make_router, mock, tmp_path, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART))
    first = make_router(cfg).delegate("explain", kind="quick")
    mock.on(GROQ, "/chat/completions", chat_json(SECOND_PART))
    resumed = make_router(cfg).delegate("explain", kind="quick", continue_from=first["id"], max_continuations=1)
    assert resumed["path"].endswith(f"{first['id']}-r1.md")
    assert saved_text(resumed["path"]) == FIRST_PART + SECOND_PART and resumed["truncated"] is False


def test_the_resume_sends_the_partial_as_an_assistant_turn(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART))
    first = make_router(cfg).delegate("explain", kind="quick")
    mock.on(GROQ, "/chat/completions", chat_json(SECOND_PART))
    make_router(cfg).delegate("explain", kind="quick", continue_from=first["id"], max_continuations=1)
    assert {"role": "assistant", "content": FIRST_PART} in mock.bodies[-1]["messages"]


def test_the_original_partial_answer_file_is_left_untouched(monkeypatch, make_router, mock, tmp_path, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART))
    first = make_router(cfg).delegate("explain", kind="quick")
    before = (tmp_path / ".agent-out" / f"{first['id']}.md").read_text(encoding="utf-8")
    mock.on(GROQ, "/chat/completions", chat_json(SECOND_PART))
    make_router(cfg).delegate("explain", kind="quick", continue_from=first["id"], max_continuations=1)
    after = (tmp_path / ".agent-out" / f"{first['id']}.md").read_text(encoding="utf-8")
    assert after == before


# ------------------------------------------------------------------ (3)(4) a resume must fit the task and the id

def test_a_different_task_cannot_continue_the_saved_answer(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART))
    first = make_router(cfg).delegate("explain", kind="quick")
    with pytest.raises(RouterError, match="differs"):
        make_router(cfg).delegate("a different task", kind="quick", continue_from=first["id"], max_continuations=1)


def test_continue_from_rejects_a_path_instead_of_an_id(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    with pytest.raises(RouterError):
        make_router(cfg).delegate("explain", kind="quick", continue_from="../x")


def test_continue_from_rejects_an_id_that_was_never_saved(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    with pytest.raises(RouterError):
        make_router(cfg).delegate("explain", kind="quick", continue_from="no-such-id")


# ------------------------------------------------------------------ (5) only a cut-off answer can be continued

def test_an_answer_that_ended_normally_is_not_cut_off_and_cannot_be_resumed(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json("a complete answer"))
    first = make_router(cfg).delegate("explain", kind="quick")
    with pytest.raises(RouterError, match="not cut off"):
        make_router(cfg).delegate("explain", kind="quick", continue_from=first["id"], max_continuations=1)


# ------------------------------------------------------------------ (6) the saving source must still be reachable

def test_a_resume_uses_only_the_source_that_wrote_the_partial(monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    cfg["routing"]["calls"]["max_continuations"] = 0
    mock.on(GROQ, "/chat/completions", cut(FIRST_PART))
    mock.on(DEEPSEEK, "/chat/completions", chat_json("deepseek answer"))
    first = make_router(cfg).delegate("explain", kind="quick")
    lost_groq_key(monkeypatch)
    with pytest.raises(RouterError):
        make_router(cfg).delegate("explain", kind="quick", continue_from=first["id"], max_continuations=1)
    assert mock.hosts() == {"api.groq.com"}


# ------------------------------------------------------------------ (7) each attempt keeps its own partial file

def test_a_rejected_attempt_keeps_its_partial_under_the_a2_suffix(monkeypatch, make_router, mock, tmp_path, cfg):
    cfg["routing"]["calls"]["abort_waste_usd"] = 100
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(DEEPSEEK, "/chat/completions", cut_off_in_code())
    mock.on(GEMINI, "/chat/completions", chat_json("complete answer"))
    result = make_router(cfg).delegate("t", kind="data_script")
    assert result["id"].endswith("-a2") and result["path"].endswith("-a2.md")


def test_the_first_attempt_partial_file_leaves_the_base_id_for_the_next_attempt(monkeypatch, make_router, mock, tmp_path, cfg):
    cfg["routing"]["calls"]["abort_waste_usd"] = 100
    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    mock.on(DEEPSEEK, "/chat/completions", cut_off_in_code())
    mock.on(GEMINI, "/chat/completions", chat_json("complete answer"))
    result = make_router(cfg).delegate("t", kind="data_script")
    base = result["id"].removesuffix("-a2")
    assert (tmp_path / ".agent-out" / f"{base}.md").is_file()


# ------------------------------------------------------------------ (8) the pure helpers

def test_read_answer_text_strips_the_front_matter(tmp_path):
    path = tmp_path / "answer.md"
    path.write_text("---\nid: x\n---\n\nbody", encoding="utf-8")
    assert read_answer_text(path) == "body"


def test_prompt_fingerprint_is_stable_and_changes_with_the_content():
    messages = [{"role": "user", "content": "same"}]
    assert prompt_fingerprint(messages) == prompt_fingerprint([{"role": "user", "content": "same"}])
    assert prompt_fingerprint(messages) != prompt_fingerprint([{"role": "user", "content": "other"}])


def test_resumed_id_skips_numbers_that_are_already_taken():
    assert resumed_id("x", set()) == "x-r1"
    assert resumed_id("x", {"x-r1"}) == "x-r2"
    assert resumed_id("x", {"x-r1", "x-r2"}) == "x-r3"
