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
    # English unless the test asks otherwise: the developer's machine may have FOREMANKIT_LANG=pl set
    env = {"FOREMANKIT_STATE_DIR": str(tmp_path), "FOREMANKIT_SESSION_REMINDER_MB": "1", "FOREMANKIT_LANG": "en"}
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


def create_model_transcript(tmp_path: Path, model: str, name: str = "transcript.txt") -> Path:
    """A transcript whose last answer was produced by `model` (claude-... = cloud, anything else = local llama alias)."""
    transcript = tmp_path / name
    lines = [{"type": "user", "message": {"role": "user", "content": "hi"}},
             {"type": "assistant", "message": {"role": "assistant", "model": model, "content": "ok"}},
             {"type": "assistant", "message": {"role": "assistant", "model": "<synthetic>", "content": "x"}}]
    transcript.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
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
    transcript = create_model_transcript(tmp_path, "qwen3.6-35b-a3b-ud-q4_k_xl")
    event = {"session_id": "1", "transcript_path": str(transcript)}
    write_gateway_usage(tmp_path, 20_000)                       # 31 %: below the first level
    assert run_hook(event, tmp_path, now=1_000_000)[0] == ""
    write_gateway_usage(tmp_path, 42_000)                       # 64 %: level 1
    stdout, _ = run_hook(event, tmp_path, now=1_000_000)
    assert "systemMessage" in stdout and "64%" in stdout
    assert run_hook(event, tmp_path, now=1_000_000)[0] == ""    # same level: silent
    write_gateway_usage(tmp_path, 60_000)                       # 91 %: level 3 (stop and hand over)
    assert "told to stop" in json.loads(run_hook(event, tmp_path, now=1_000_000)[0])["systemMessage"]


def test_messages_are_polish_when_foremankit_lang_is_pl(tmp_path: Path) -> None:
    transcript = create_model_transcript(tmp_path, "qwen3.6-35b-a3b-ud-q4_k_xl")
    event = {"session_id": "1", "transcript_path": str(transcript)}
    write_gateway_usage(tmp_path, 60_000)                       # 91 %: level 3
    stdout, _ = run_hook(event, tmp_path, now=1_000_000, language="pl")
    assert "zatrzymania pracy" in json.loads(stdout)["systemMessage"]


def test_unknown_language_falls_back_to_english(tmp_path: Path) -> None:
    transcript = create_model_transcript(tmp_path, "qwen3.6-35b-a3b-ud-q4_k_xl")
    event = {"session_id": "1", "transcript_path": str(transcript)}
    write_gateway_usage(tmp_path, 60_000)
    stdout, _ = run_hook(event, tmp_path, now=1_000_000, language="xx")
    assert "told to stop" in json.loads(stdout)["systemMessage"]


def test_stale_gateway_data_is_ignored(tmp_path: Path) -> None:
    transcript = create_model_transcript(tmp_path, "qwen3.6-35b-a3b-ud-q4_k_xl")
    event = {"session_id": "1", "transcript_path": str(transcript)}
    write_gateway_usage(tmp_path, 60_000, age_s=3600)
    assert run_hook(event, tmp_path, now=1_000_000)[0] == ""


def test_context_level_is_per_session(tmp_path: Path) -> None:
    transcript = create_model_transcript(tmp_path, "qwen3.6-35b-a3b-ud-q4_k_xl")
    write_gateway_usage(tmp_path, 60_000)
    assert "systemMessage" in run_hook({"session_id": "1", "transcript_path": str(transcript)}, tmp_path, now=1_000_000)[0]
    assert "systemMessage" in run_hook({"session_id": "2", "transcript_path": str(transcript)}, tmp_path, now=1_000_000)[0]


def write_usage(tmp_path: Path, tokens: int, window: int = 65536) -> None:
    import time
    (tmp_path / "gateway_context.json").write_text(
        json.dumps({"tokens": tokens, "window": window, "time": time.time()}), encoding="utf-8")


def test_a_cloud_session_never_gets_the_local_models_fill_level(tmp_path: Path) -> None:
    write_usage(tmp_path, 60000)
    transcript = create_model_transcript(tmp_path, "claude-sonnet-5-5")
    stdout, _ = run_hook({"session_id": "cloud", "transcript_path": str(transcript)}, tmp_path)
    assert stdout == ""


def test_a_session_without_an_answer_yet_is_not_treated_as_local(tmp_path: Path) -> None:
    write_usage(tmp_path, 60000)
    transcript = tmp_path / "new.jsonl"
    transcript.write_text(json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}) + "\n", encoding="utf-8")
    stdout, _ = run_hook({"session_id": "new", "transcript_path": str(transcript)}, tmp_path)
    assert stdout == ""


def test_at_eighty_five_percent_the_local_model_is_told_to_stop_and_hand_over(tmp_path: Path) -> None:
    write_usage(tmp_path, 56000)
    transcript = create_model_transcript(tmp_path, "qwen3.6-35b-a3b-ud-q4_k_xl")
    stdout, _ = run_hook({"session_id": "local", "transcript_path": str(transcript)}, tmp_path)
    context = json.loads(stdout)["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("STOP") and "HANDOFF.md" in context and "NEW session" in context
    assert "Do not start any new step" in context


def test_the_stop_order_is_repeated_on_every_prompt_while_the_context_stays_full(tmp_path: Path) -> None:
    write_usage(tmp_path, 62000)
    transcript = create_model_transcript(tmp_path, "spark-x2.5-4b-q8_0")
    event = {"session_id": "local", "transcript_path": str(transcript)}
    assert "STOP" in run_hook(event, tmp_path)[0] and "STOP" in run_hook(event, tmp_path)[0]


def test_below_eighty_five_percent_there_is_no_stop_order(tmp_path: Path) -> None:
    write_usage(tmp_path, 54000)
    transcript = create_model_transcript(tmp_path, "qwen3.6-35b-a3b-ud-q4_k_xl")
    stdout, _ = run_hook({"session_id": "local", "transcript_path": str(transcript)}, tmp_path)
    assert "STOP" not in stdout
