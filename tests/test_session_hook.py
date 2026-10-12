#!/usr/bin/env python3
"""Pytest tests for hooks/session_length_reminder.py."""
import json
import os
import subprocess
import sys
from pathlib import Path


def run_hook(
    event: dict, tmp_path: Path, now: float | None = None, language: str | None = None
) -> tuple[str, int]:
    hook_path = Path(__file__).parent.parent / "hooks" / "session_length_reminder.py"
    env = {"FOREMANKIT_STATE_DIR": str(tmp_path), "FOREMANKIT_SESSION_REMINDER_MB": "1"}
    if now is not None:
        env["FOREMANKIT_FAKE_NOW"] = str(now)
    if language is not None:
        env["FOREMANKIT_LANG"] = language
    result = subprocess.run(
        [sys.executable, str(hook_path)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )
    return result.stdout, result.returncode


def create_transcript(tmp_path: Path, size_mb: float) -> Path:
    transcript = tmp_path / "transcript.txt"
    transcript.write_bytes(b"x" * int(size_mb * 1024 * 1024))
    return transcript


def test_below_threshold(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 0.5)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_above_threshold(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 2)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert "systemMessage" in stdout
    assert "hookSpecificOutput" in stdout
    assert returncode == 0


def test_immediate_second_call(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 2)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    run_hook(event, tmp_path)
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_after_100_minutes(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 2)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    run_hook(event, tmp_path, now=0)
    transcript.write_bytes(b"x" * 7 * 1024 * 1024)  # Grow by 5 MB
    stdout, returncode = run_hook(event, tmp_path, now=100 * 60)
    assert "systemMessage" in stdout
    assert "hookSpecificOutput" in stdout
    assert returncode == 0


def test_after_100_minutes_small_growth(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 2)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    run_hook(event, tmp_path, now=0)
    transcript.write_bytes(b"x" * 3 * 1024 * 1024)  # Grow by 1 MB
    stdout, returncode = run_hook(event, tmp_path, now=100 * 60)
    assert stdout == ""
    assert returncode == 0


def test_another_session_id(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 2)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    run_hook(event, tmp_path)
    event["session_id"] = "2"
    stdout, returncode = run_hook(event, tmp_path)
    assert "systemMessage" in stdout
    assert "hookSpecificOutput" in stdout
    assert returncode == 0


def test_invalid_json(tmp_path: Path) -> None:
    stdout, returncode = run_hook("invalid json", tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_missing_transcript(tmp_path: Path) -> None:
    event = {"session_id": "1", "transcript_path": str(tmp_path / "missing.txt")}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_missing_transcript_path(tmp_path: Path) -> None:
    event = {"session_id": "1"}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_invalid_threshold(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 2)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    env = {
        "FOREMANKIT_STATE_DIR": str(tmp_path),
        "FOREMANKIT_SESSION_REMINDER_MB": "abc",
        "FOREMANKIT_FAKE_NOW": ""
    }
    hook_path = Path(__file__).parent.parent / "hooks" / "session_length_reminder.py"
    result = subprocess.run(
        [sys.executable, str(hook_path)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )
    assert result.stdout == ""
    assert result.returncode == 0


def test_more_than_50_sessions(tmp_path: Path) -> None:
    for i in range(51):
        transcript = create_transcript(tmp_path, 2)
        event = {"session_id": str(i), "transcript_path": str(transcript)}
        run_hook(event, tmp_path, now=i)

    state_file = tmp_path / "session_reminder.json"
    with open(state_file, "r") as f:
        state = json.load(f)

    assert len(state) == 50


def write_gateway_usage(tmp_path: Path, tokens: int, window: int = 65536, age_s: float = 0) -> None:
    usage = {"tokens": tokens, "window": window, "time": 1_000_000 - age_s}
    (tmp_path / "gateway_context.json").write_text(json.dumps(usage), encoding="utf-8")


def test_local_context_levels_escalate_once_each(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 0.01)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    write_gateway_usage(tmp_path, 20_000)                       # 31 %: below the first level
    assert run_hook(event, tmp_path, now=1_000_000)[0] == ""
    write_gateway_usage(tmp_path, 42_000)                       # 64 %: level 1
    stdout, _ = run_hook(event, tmp_path, now=1_000_000)
    assert "systemMessage" in stdout and "64%" in stdout
    assert run_hook(event, tmp_path, now=1_000_000)[0] == ""    # same level: silent
    write_gateway_usage(tmp_path, 58_000)                       # 88 %: level 3
    assert "almost full" in json.loads(run_hook(event, tmp_path, now=1_000_000)[0])["systemMessage"]


def test_messages_are_polish_when_foremankit_lang_is_pl(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 0.01)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    write_gateway_usage(tmp_path, 58_000)                       # 88 %: level 3
    stdout, _ = run_hook(event, tmp_path, now=1_000_000, language="pl")
    assert "prawie pełny" in json.loads(stdout)["systemMessage"]


def test_unknown_language_falls_back_to_english(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 0.01)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    write_gateway_usage(tmp_path, 58_000)
    stdout, _ = run_hook(event, tmp_path, now=1_000_000, language="xx")
    assert "almost full" in json.loads(stdout)["systemMessage"]


def test_stale_gateway_data_is_ignored(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 0.01)
    event = {"session_id": "1", "transcript_path": str(transcript)}
    write_gateway_usage(tmp_path, 60_000, age_s=3600)
    assert run_hook(event, tmp_path, now=1_000_000)[0] == ""


def test_context_level_is_per_session(tmp_path: Path) -> None:
    transcript = create_transcript(tmp_path, 0.01)
    write_gateway_usage(tmp_path, 60_000)
    assert "systemMessage" in run_hook({"session_id": "1", "transcript_path": str(transcript)}, tmp_path, now=1_000_000)[0]
    assert "systemMessage" in run_hook({"session_id": "2", "transcript_path": str(transcript)}, tmp_path, now=1_000_000)[0]
