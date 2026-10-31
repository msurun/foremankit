"""Subprocess provider for the Google Antigravity CLI `agy` (print mode, JSON output).
agy has no stdin mode and no API key: the prompt is a command-line argument and the sign-in is the Antigravity app's.
The run happens in an empty temp directory so the agent cannot touch the project.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from typing import Any

from .providers import ChatRequest, ProviderError, ProviderTimeout, normalize_usage

MAX_PROMPT_CHARS = 24000        # the Windows command line is limited to ~32k characters
ERROR_CHARS = 200
RATE_LIMIT_TEXT = re.compile(r"quota|rate.?limit|too many requests|\b429\b|exhausted", re.IGNORECASE)


def agy_completion(request: ChatRequest, binary: str, runner=subprocess.run) -> dict[str, Any]:
    """Run `agy -p <prompt> --model <id> --output-format json`; same result shape as providers.chat_completion.

    Raises ProviderError (prompt too long, binary missing, agy error, bad output) or ProviderTimeout.
    """
    prompt = _build_prompt(request.messages)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ProviderError(f"prompt too long for agy ({len(prompt)} chars)")
    command = [binary, "-p", prompt, "--model", request.model_id, "--output-format", "json"]
    process = _run(runner, command, request.timeout, binary)
    data = _parse_json(process.stdout)
    if process.returncode != 0 or data.get("status") != "SUCCESS":
        _raise_agy_error(data, process.stderr)
    return {"text": data.get("response") or "", "usage": normalize_usage(_openai_usage(data.get("usage") or {})),
            "finish_reason": "stop", "timings": None, "served_model": request.model_id}


def _run(runner, command: list[str], timeout: float, binary: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as workdir:
        try:
            return runner(command, capture_output=True, text=True, encoding="utf-8", timeout=timeout, cwd=workdir)
        except subprocess.TimeoutExpired as error:
            raise ProviderTimeout(f"timeout after {timeout:.0f}s", transient=True) from error
        except FileNotFoundError as error:
            raise ProviderError(f"agy binary not found: {binary}") from error


def _build_prompt(messages: list[dict[str, str]]) -> str:
    """System messages first, then the rest, separated by blank lines (agy takes a single turn)."""
    system = [m["content"] for m in messages if m.get("role") == "system"]
    others = [m["content"] for m in messages if m.get("role") != "system"]
    return "\n\n".join(system + others)


def _parse_json(stdout: str) -> dict[str, Any]:
    """The JSON object agy prints; {} when stdout is empty, ProviderError when it is not JSON."""
    if not (stdout or "").strip():
        return {}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ProviderError(f"agy returned invalid JSON: {error}") from error
    return data if isinstance(data, dict) else {}


def _openai_usage(raw: dict[str, Any]) -> dict[str, Any]:
    """agy usage keys in the OpenAI-style shape that normalize_usage reads."""
    return {"prompt_tokens": raw.get("input_tokens"), "completion_tokens": raw.get("output_tokens"),
            "prompt_cache_hit_tokens": raw.get("cache_read_tokens"),
            "completion_tokens_details": {"reasoning_tokens": raw.get("thinking_tokens")},
            "total_tokens": raw.get("total_tokens")}


def _raise_agy_error(data: dict[str, Any], stderr: str) -> None:
    """ProviderError with a short message; status 429 (cooldown) when the text looks like a quota/rate limit."""
    message = str(data.get("error") or stderr or "agy failed").strip()[:ERROR_CHARS]
    raise ProviderError(message, status=429 if RATE_LIMIT_TEXT.search(message) else None)
