"""Unit tests for foremankit.virtual_clear: valid cuts, the saved marker and its reapply."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from foremankit import virtual_clear


def text_turn(role: str, text: str) -> dict[str, Any]:
    """A plain text message."""
    return {"role": role, "content": text}


def tool_turn(tool_id: str) -> list[dict[str, Any]]:
    """An assistant tool_use followed by the user tool_result that answers it."""
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}]},
    ]


def long_conversation() -> dict[str, Any]:
    """Ten plain turns, then tool turns, then the final user question."""
    messages: list[dict[str, Any]] = []
    for i in range(10):
        messages += [text_turn("user", f"question {i}"), text_turn("assistant", f"answer {i}")]
    for i in range(3):
        messages += [text_turn("user", f"work {i}")] + tool_turn(f"t{i}")
    messages.append(text_turn("user", "final"))
    return {"model": "m", "messages": messages, "metadata": {"user_id": "session-a"}}


def test_a_valid_cut_never_starts_with_a_tool_result() -> None:
    """Every index the cut can pick is a user message holding no tool_result."""
    messages = long_conversation()["messages"]
    for index in virtual_clear.valid_cut_indexes(messages):
        content = messages[index]["content"]
        assert messages[index]["role"] == "user"
        assert not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in (
            content if isinstance(content, list) else [content]))
    assert virtual_clear.choose_cut(messages) is not None


def test_a_cut_keeps_every_kept_tool_use_with_its_result() -> None:
    """Whatever cut is chosen, each kept tool_use still has its tool_result right after it."""
    messages = long_conversation()["messages"]
    for index in virtual_clear.valid_cut_indexes(messages):
        kept = messages[index:]
        uses = {b["id"] for m in kept if isinstance(m["content"], list)
                for b in m["content"] if b.get("type") == "tool_use"}
        results = {b["tool_use_id"] for m in kept if isinstance(m["content"], list)
                   for b in m["content"] if b.get("type") == "tool_result"}
        assert uses <= results


def test_choose_cut_returns_none_without_a_usable_cut() -> None:
    """No user message at all, or only the last message is a candidate, gives None."""
    assert virtual_clear.choose_cut([text_turn("assistant", "hi")]) is None
    assert virtual_clear.choose_cut([text_turn("user", "only")]) is None
    tool_only = [text_turn("assistant", "start")] + tool_turn("t0")
    assert virtual_clear.choose_cut(tool_only) is None


def test_apply_cut_prepends_the_handoff_and_keeps_the_tail() -> None:
    """The cut body starts with the handoff user message and continues from the cut index."""
    data = long_conversation()
    cut = virtual_clear.choose_cut(data["messages"])
    cut_data = virtual_clear.apply_cut(data, "state", cut)
    assert cut_data["messages"][0] == {"role": "user", "content": "handoff: state"}
    assert cut_data["messages"][2:] == data["messages"][cut:]


def test_a_saved_marker_reapplies_the_same_cut(tmp_path: Path) -> None:
    """After save_marker, reapply on the raw body gives the same cut as apply_cut."""
    data = long_conversation()
    cut = virtual_clear.choose_cut(data["messages"])
    virtual_clear.save_marker(tmp_path, virtual_clear.session_key(data), data["messages"][cut], "state")
    assert virtual_clear.reapply(data, tmp_path) == virtual_clear.apply_cut(data, "state", cut)


def test_a_marker_for_another_session_is_ignored(tmp_path: Path) -> None:
    """A marker saved for session A does nothing to a request from session B."""
    data = long_conversation()
    cut = virtual_clear.choose_cut(data["messages"])
    virtual_clear.save_marker(tmp_path, "someone-else", data["messages"][cut], "state")
    assert virtual_clear.reapply(data, tmp_path) is None


def test_a_missing_or_unmatched_marker_gives_none(tmp_path: Path) -> None:
    """No marker file, or a marker whose message is gone from the body, reapplies nothing."""
    data = long_conversation()
    assert virtual_clear.reapply(data, tmp_path) is None
    virtual_clear.save_marker(tmp_path, virtual_clear.session_key(data), text_turn("user", "gone"), "state")
    assert virtual_clear.reapply(data, tmp_path) is None


def test_session_key_reads_block_list_user_content() -> None:
    """A user message whose content is a list of text blocks gives a stable session key."""
    data = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    plain = {"messages": [text_turn("user", "hello")]}
    assert virtual_clear.session_key(data) == virtual_clear.session_key(plain)
