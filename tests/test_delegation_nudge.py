#!/usr/bin/env python3
"""Pytest tests for hooks/delegation_nudge.py."""
import json
import os
import subprocess
import sys
from pathlib import Path

THRESHOLD = 10  # FOREMANKIT_NUDGE_LINES used by every test, keeps the fixtures small


def hook_env(tmp_path: Path, language: str | None, extra_env: dict | None) -> dict:
    """The environment the hook subprocess runs with, including any overrides."""
    env = {"FOREMANKIT_STATE_DIR": str(tmp_path), "FOREMANKIT_NUDGE_LINES": str(THRESHOLD)}
    if language is not None:
        env["FOREMANKIT_LANG"] = language
    if extra_env is not None:
        env.update(extra_env)
    return env


def run_hook(
    event: object,
    tmp_path: Path,
    language: str | None = None,
    extra_env: dict | None = None,
) -> tuple[str, int]:
    """Run the hook as a subprocess with the given event on stdin; return (stdout, exit code)."""
    hook_path = Path(__file__).parent.parent / "hooks" / "delegation_nudge.py"
    payload = event if isinstance(event, str) else json.dumps(event)
    result = subprocess.run(
        [sys.executable, str(hook_path)],
        input=payload,
        capture_output=True,
        text=True,
        env={**os.environ, **hook_env(tmp_path, language, extra_env)},
        check=False,
    )
    return result.stdout, result.returncode


def write_transcript(path: Path, calls: list) -> Path:
    """Write a JSONL transcript where every item of `calls` is one tool_use block (name, input)."""
    lines = []
    for name, tool_input in calls:
        block = {"type": "tool_use", "name": name, "input": tool_input}
        message = {"role": "assistant", "content": [block]}
        lines.append(json.dumps({"type": "assistant", "message": message}))
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    return path


def append_transcript(path: Path, calls: list) -> None:
    """Append more tool_use lines to an existing transcript."""
    text = "\n".join(
        json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "tool_use", "name": name, "input": tool_input}]},
            }
        )
        for name, tool_input in calls
    )
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def post_event(tmp_path: Path, transcript: Path) -> dict:
    """The PostToolUse event dict for the given transcript."""
    return {"hook_event_name": "PostToolUse", "session_id": "s1", "transcript_path": str(transcript)}


def write_text_lines(count: int) -> str:
    """Content of a Write call with exactly `count` lines."""
    return "\n".join(f"line {i}" for i in range(count))


def write_toggles(tmp_path: Path, states: dict) -> None:
    """Write state.json with a `toggles` map (keys like 'delegation:paid', values 'on' or 'off')."""
    (tmp_path / "state.json").write_text(json.dumps({"toggles": states}), encoding="utf-8")


def all_delegation_off() -> dict:
    """Toggles that switch every delegation category off."""
    return {
        "delegation:paid": "off",
        "delegation:free": "off",
        "delegation:local": "off",
    }


def test_fewer_lines_than_the_threshold_stay_silent(tmp_path: Path) -> None:
    transcript = write_transcript(tmp_path / "t.jsonl", [("Write", {"content": write_text_lines(5)})])
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_a_write_of_twelve_lines_reminds_the_model_and_the_user(tmp_path: Path) -> None:
    transcript = write_transcript(tmp_path / "t.jsonl", [("Write", {"content": write_text_lines(12)})])
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path)
    answer = json.loads(stdout)
    assert answer["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "lines" in answer["hookSpecificOutput"]["additionalContext"]
    assert "systemMessage" in answer
    assert returncode == 0


def test_a_second_call_with_no_new_data_is_silent(tmp_path: Path) -> None:
    transcript = write_transcript(tmp_path / "t.jsonl", [("Write", {"content": write_text_lines(12)})])
    event = post_event(tmp_path, transcript)
    run_hook(event, tmp_path)
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_a_delegate_call_resets_the_count(tmp_path: Path) -> None:
    transcript = write_transcript(
        tmp_path / "t.jsonl",
        [
            ("Write", {"content": write_text_lines(12)}),
            ("mcp__foremankit__delegate", {"task": "tests"}),
            ("Write", {"content": write_text_lines(5)}),
        ],
    )
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_an_agent_call_resets_the_count(tmp_path: Path) -> None:
    transcript = write_transcript(
        tmp_path / "t.jsonl",
        [
            ("Write", {"content": write_text_lines(12)}),
            ("Agent", {"prompt": "write the tests"}),
            ("Write", {"content": write_text_lines(5)}),
        ],
    )
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_an_edit_counts_the_lines_of_its_new_string(tmp_path: Path) -> None:
    transcript = write_transcript(
        tmp_path / "t.jsonl", [("Edit", {"file_path": "a.py", "new_string": write_text_lines(12)})]
    )
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path)
    assert "systemMessage" in stdout
    assert returncode == 0


def test_a_bash_heredoc_is_counted_and_a_plain_command_is_not(tmp_path: Path) -> None:
    transcript = write_transcript(
        tmp_path / "t.jsonl",
        [("Bash", {"command": "ls -la"}), ("Bash", {"command": "cat > out.sh <<'EOF'\n" + write_text_lines(12) + "\nEOF"})],
    )
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path)
    assert "systemMessage" in stdout
    assert returncode == 0


def test_a_long_plain_bash_command_without_a_file_write_is_silent(tmp_path: Path) -> None:
    long_command = "\n".join(f"echo step {number}" for number in range(25))   # 25 lines, but it writes no file
    transcript = write_transcript(tmp_path / "t.jsonl", [("Bash", {"command": long_command})])
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_the_next_reminder_waits_for_another_threshold_of_new_lines(tmp_path: Path) -> None:
    transcript = write_transcript(tmp_path / "t.jsonl", [("Write", {"content": write_text_lines(12)})])
    event = post_event(tmp_path, transcript)
    run_hook(event, tmp_path)
    append_transcript(transcript, [("Write", {"content": write_text_lines(5)})])
    assert run_hook(event, tmp_path)[0] == ""
    append_transcript(transcript, [("Write", {"content": write_text_lines(6)})])
    stdout, returncode = run_hook(event, tmp_path)
    assert "systemMessage" in stdout
    assert returncode == 0


def test_post_tool_use_is_silent_when_all_delegation_is_off(tmp_path: Path) -> None:
    write_toggles(tmp_path, all_delegation_off())
    transcript = write_transcript(tmp_path / "t.jsonl", [("Write", {"content": write_text_lines(12)})])
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_user_prompt_submit_says_delegation_is_switched_off(tmp_path: Path) -> None:
    write_toggles(tmp_path, all_delegation_off())
    event = {"hook_event_name": "UserPromptSubmit", "session_id": "s1"}
    stdout, returncode = run_hook(event, tmp_path)
    answer = json.loads(stdout)
    assert "switched OFF" in answer["hookSpecificOutput"]["additionalContext"]
    assert returncode == 0


def test_user_prompt_submit_is_silent_with_only_two_categories_off(tmp_path: Path) -> None:
    toggles = {"delegation:paid": "off", "delegation:free": "off", "delegation:local": "on"}
    write_toggles(tmp_path, toggles)
    event = {"hook_event_name": "UserPromptSubmit", "session_id": "s1"}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_user_prompt_submit_is_silent_when_delegation_is_on(tmp_path: Path) -> None:
    event = {"hook_event_name": "UserPromptSubmit", "session_id": "s1"}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_a_polish_system_message_is_used_when_foremankit_lang_is_pl(tmp_path: Path) -> None:
    transcript = write_transcript(tmp_path / "t.jsonl", [("Write", {"content": write_text_lines(12)})])
    stdout, returncode = run_hook(post_event(tmp_path, transcript), tmp_path, language="pl")
    assert "linii" in json.loads(stdout)["systemMessage"]
    assert returncode == 0


def test_invalid_json_on_stdin_is_silent(tmp_path: Path) -> None:
    stdout, returncode = run_hook("invalid json", tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_a_missing_transcript_file_is_silent(tmp_path: Path) -> None:
    event = {"hook_event_name": "PostToolUse", "session_id": "s1", "transcript_path": str(tmp_path / "missing.jsonl")}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_a_cut_off_last_line_is_counted_only_when_it_is_complete(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    complete = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Write", "input": {"content": write_text_lines(12)}}]},
        }
    )
    transcript.write_text(complete, encoding="utf-8")  # no trailing newline: the line is still half written
    event = post_event(tmp_path, transcript)
    assert run_hook(event, tmp_path)[0] == ""
    transcript.write_text(complete + "\n", encoding="utf-8")
    stdout, returncode = run_hook(event, tmp_path)
    assert "systemMessage" in stdout
    assert returncode == 0
