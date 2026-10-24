#!/usr/bin/env python3
"""Pytest tests for cloud context reminders in hooks/session_length_reminder.py."""
import json
import os
import subprocess
import sys
from pathlib import Path


def run_hook(
    event: dict, tmp_path: Path, env_overrides: dict | None = None, language: str | None = None
) -> tuple[str, int]:
    """Run the hook as a subprocess with specified environment and language."""
    hook_path = Path(__file__).parent.parent / "hooks" / "session_length_reminder.py"
    env = {"FOREMANKIT_STATE_DIR": str(tmp_path), "FOREMANKIT_SESSION_REMINDER_MB": "1", "FOREMANKIT_LANG": "en"}
    if env_overrides:
        env.update(env_overrides)
    if language:
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


def create_cloud_transcript(tmp_path: Path, model: str, usage_dict: dict | None = None) -> Path:
    """Create a transcript with a final assistant message containing usage data."""
    transcript = tmp_path / "transcript.jsonl"
    lines = []
    # Add some user context
    lines.append(json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}))
    
    if usage_dict:
        msg = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": model,
                "usage": usage_dict,
                "content": "ok",
            },
        }
        lines.append(json.dumps(msg))
    
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript


def create_large_padding_transcript(tmp_path: Path, padding_mb: float) -> Path:
    """Create a transcript with a large padding line to trigger size reminders."""
    transcript = tmp_path / "large.jsonl"
    padding = "x" * int(padding_mb * 1024 * 1024)
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": padding}}),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript


def test_cloud_100k_tokens_is_below_threshold(tmp_path: Path) -> None:
    """100k tokens (50% of 200k) -> no output."""
    transcript = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 100_000})
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_cloud_150k_tokens_triggers_level_1(tmp_path: Path) -> None:
    """150k tokens split over keys -> level 1 'ℹ', text contains '75%'."""
    usage = {"input_tokens": 10_000, "cache_read_input_tokens": 130_000, "cache_creation_input_tokens": 10_000}
    transcript = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", usage)
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert returncode == 0
    data = json.loads(stdout)
    assert data["systemMessage"].startswith("ℹ")
    assert "75%" in data["systemMessage"]


def test_cloud_180k_tokens_triggers_level_2(tmp_path: Path) -> None:
    """180k tokens -> level 2 '⚠'."""
    transcript = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 180_000})
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert returncode == 0
    data = json.loads(stdout)
    assert data["systemMessage"].startswith("⚠")


def test_cloud_same_level_twice_is_silent(tmp_path: Path) -> None:
    """The same level twice in the same session -> second call prints nothing."""
    transcript = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 150_000})
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    run_hook(event, tmp_path)
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_cloud_escalation_from_level_1_to_level_2(tmp_path: Path) -> None:
    """150k then 180k -> second call fires level 2 (escalation)."""
    event = {"session_id": "s1"}
    
    t1 = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 150_000})
    event["transcript_path"] = str(t1)
    run_hook(event, tmp_path)
    
    t2 = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 180_000})
    event["transcript_path"] = str(t2)
    stdout, returncode = run_hook(event, tmp_path)
    assert "⚠" in json.loads(stdout)["systemMessage"]
    assert returncode == 0


def test_cloud_level_resets_after_compact(tmp_path: Path) -> None:
    """150k, then 20k (after /compact), then 150k again -> fires again."""
    event = {"session_id": "s1"}
    
    t1 = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 150_000})
    event["transcript_path"] = str(t1)
    run_hook(event, tmp_path)
    
    t2 = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 20_000})
    event["transcript_path"] = str(t2)
    run_hook(event, tmp_path) # Should be silent (level 0)
    
    t3 = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 150_000})
    event["transcript_path"] = str(t3)
    stdout, returncode = run_hook(event, tmp_path)
    assert "ℹ" in json.loads(stdout)["systemMessage"]
    assert returncode == 0


def test_cloud_1m_window_model_id(tmp_path: Path) -> None:
    """Model 'claude-opus-5-5[1m]' with 150k -> no output (window 1M)."""
    transcript = create_cloud_transcript(tmp_path, "claude-opus-5-5[1m]", {"input_tokens": 150_000})
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_cloud_inferred_1m_window(tmp_path: Path) -> None:
    """Model 'claude-opus-5-5' with 250k (above 200k) -> window inferred as 1M, so 25% -> no output."""
    transcript = create_cloud_transcript(tmp_path, "claude-opus-5-5", {"input_tokens": 250_000})
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_cloud_env_override_window(tmp_path: Path) -> None:
    """FOREMANKIT_CLOUD_WINDOW=100000 with 80k tokens -> level 1."""
    transcript = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 80_000})
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path, env_overrides={"FOREMANKIT_CLOUD_WINDOW": "100000"})
    assert "ℹ" in json.loads(stdout)["systemMessage"]
    assert returncode == 0


def test_local_model_ignores_cloud_usage(tmp_path: Path) -> None:
    """A local model alias with usage record -> no cloud reminder."""
    transcript = create_cloud_transcript(tmp_path, "qwen3.6-27b", {"input_tokens": 190_000})
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0


def test_usage_replaces_size_reminder(tmp_path: Path) -> None:
    """A transcript > 1MB whose last claude answer has only 20k tokens -> NO size reminder."""
    # Create a file with 1.2MB of padding, then a small cloud answer
    transcript = tmp_path / "mixed.jsonl"
    padding = "x" * int(1.2 * 1024 * 1024)
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": padding}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-3-5", "usage": {"input_tokens": 20_000}, "content": "ok"}})
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    # Usage (20k/200k = 10%) is lower than cloud levels, so no cloud reminder.
    # Usage presence prevents the size reminder.
    assert stdout == ""
    assert returncode == 0


def test_fallback_to_size_reminder_if_no_usage(tmp_path: Path) -> None:
    """A transcript > 1MB made only of 'x' bytes (no usage) -> MB reminder fires."""
    transcript = create_large_padding_transcript(tmp_path, 2.0)
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert "conversation is already long" in stdout
    assert returncode == 0


def test_polish_language_cloud_reminder(tmp_path: Path) -> None:
    """Polish: FOREMANKIT_LANG=pl, 150k -> message contains 'zapełniony'."""
    transcript = create_cloud_transcript(tmp_path, "claude-3-5-sonnet", {"input_tokens": 150_000})
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path, language="pl")
    assert "zapełniony" in json.loads(stdout)["systemMessage"]
    assert returncode == 0


def test_cut_off_json_uses_previous_valid_line(tmp_path: Path) -> None:
    """A last line cut off mid-JSON plus an older valid usage line -> uses the valid line."""
    transcript = tmp_path / "cutoff.jsonl"
    lines = [
        json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-3-5", "usage": {"input_tokens": 150_000}, "content": "ok"}}),
        '{"type": "assistant", "message": {"role": "assistant", "model": "claude-3-5", "usage": {"input_tokens": 150,' # Cut off
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert "ℹ" in json.loads(stdout)["systemMessage"]
    assert returncode == 0


def test_empty_transcript_is_silent(tmp_path: Path) -> None:
    """An empty transcript file -> no output, exit code 0."""
    transcript = tmp_path / "empty.jsonl"
    transcript.write_text("", encoding="utf-8")
    event = {"session_id": "s1", "transcript_path": str(transcript)}
    stdout, returncode = run_hook(event, tmp_path)
    assert stdout == ""
    assert returncode == 0
