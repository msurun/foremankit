"""Behaviour tests for the agy subprocess provider (foremankit.agy_provider).

No real `agy` is ever started: a fake runner stands in for subprocess.run and records the
command and keyword arguments the provider builds. Every test is pure and deterministic.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from foremankit import agy_provider
from foremankit.providers import ChatRequest, ProviderError, ProviderTimeout

# --- helpers and fixtures ---------------------------------------------------

SUCCESS_STDOUT = json.dumps({
    "status": "SUCCESS",
    "response": "the answer",
    "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 3, "cache_read_tokens": 2},
})


class FakeCompletedProcess:
    """The part of subprocess.CompletedProcess the provider reads: returncode, stdout and stderr."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeRunner:
    """Stands in for subprocess.run: records each call and answers with a fixed process or raises."""

    def __init__(self, process: FakeCompletedProcess | None = None, error: Exception | None = None):
        self.process = process if process is not None else FakeCompletedProcess(SUCCESS_STDOUT)
        self.error = error
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, command: list[str], **kwargs) -> FakeCompletedProcess:
        self.calls.append((command, kwargs))
        if self.error is not None:
            raise self.error
        return self.process


@pytest.fixture
def chat_request() -> ChatRequest:
    """A minimal two-message request: one system message, one user message."""
    return ChatRequest(
        base_url="",
        api_key=None,
        model_id="model-1",
        messages=[
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
        ],
        max_tokens=100,
        timeout=10.0,
    )


# --- success path -----------------------------------------------------------

def test_a_successful_run_returns_the_response_text(chat_request: ChatRequest):
    result = agy_provider.agy_completion(chat_request, "agy", runner=FakeRunner())
    assert result["text"] == "the answer"


def test_a_successful_run_maps_the_usage_to_normalized_token_counts(chat_request: ChatRequest):
    result = agy_provider.agy_completion(chat_request, "agy", runner=FakeRunner())
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cache_hit": 2,
        "reasoning_tokens": 3,
    }


def test_a_successful_run_reports_a_stopped_finish_and_the_requested_model(chat_request: ChatRequest):
    result = agy_provider.agy_completion(chat_request, "agy", runner=FakeRunner())
    assert result["finish_reason"] == "stop"
    assert result["timings"] is None
    assert result["served_model"] == "model-1"


# --- the command agy receives ----------------------------------------------

def test_the_command_carries_the_model_id_in_json_output_mode(chat_request: ChatRequest):
    runner = FakeRunner()
    agy_provider.agy_completion(chat_request, "agy", runner=runner)
    command, _ = runner.calls[-1]
    assert "--model" in command and command[command.index("--model") + 1] == "model-1"
    assert "--output-format" in command and "json" in command


def test_the_system_message_comes_first_in_the_prompt(chat_request: ChatRequest):
    runner = FakeRunner()
    agy_provider.agy_completion(chat_request, "agy", runner=runner)
    command, _ = runner.calls[-1]
    assert command[command.index("-p") + 1] == "system message\n\nuser message"


def test_the_run_happens_in_a_temporary_directory(chat_request: ChatRequest):
    """The command gets a cwd that is a fresh temp dir, so the agent cannot see the project."""
    runner = FakeRunner()
    agy_provider.agy_completion(chat_request, "agy", runner=runner)
    _, kwargs = runner.calls[-1]
    temp_root = Path(tempfile.gettempdir()).resolve()
    assert Path(kwargs["cwd"]).resolve().is_relative_to(temp_root)


# --- error paths ------------------------------------------------------------

def test_a_timeout_raises_provider_timeout(chat_request: ChatRequest):
    runner = FakeRunner(error=subprocess.TimeoutExpired("agy", 10.0))
    with pytest.raises(ProviderTimeout):
        agy_provider.agy_completion(chat_request, "agy", runner=runner)


def test_a_missing_binary_raises_provider_error(chat_request: ChatRequest):
    runner = FakeRunner(error=FileNotFoundError())
    with pytest.raises(ProviderError, match="agy binary not found"):
        agy_provider.agy_completion(chat_request, "agy", runner=runner)


def test_an_invalid_json_answer_raises_provider_error(chat_request: ChatRequest):
    runner = FakeRunner(FakeCompletedProcess(stdout="{oops"))
    with pytest.raises(ProviderError, match="invalid JSON"):
        agy_provider.agy_completion(chat_request, "agy", runner=runner)


def test_an_error_status_raises_provider_error_with_the_error_text(chat_request: ChatRequest):
    process = FakeCompletedProcess(stdout=json.dumps({"status": "ERROR", "error": "model exploded"}))
    with pytest.raises(ProviderError, match="model exploded"):
        agy_provider.agy_completion(chat_request, "agy", runner=FakeRunner(process))


def test_a_non_zero_exit_raises_provider_error(chat_request: ChatRequest):
    process = FakeCompletedProcess(stdout=json.dumps({"status": "SUCCESS"}), stderr="crash", returncode=1)
    with pytest.raises(ProviderError):
        agy_provider.agy_completion(chat_request, "agy", runner=FakeRunner(process))


def test_a_quota_error_carries_the_rate_limit_status(chat_request: ChatRequest):
    process = FakeCompletedProcess(stdout=json.dumps({"status": "ERROR", "error": "quota exhausted"}))
    with pytest.raises(ProviderError) as raised:
        agy_provider.agy_completion(chat_request, "agy", runner=FakeRunner(process))
    assert raised.value.status == 429


def test_an_over_long_prompt_raises_provider_error_without_running(chat_request: ChatRequest):
    long_request = ChatRequest(
        base_url="", api_key=None, model_id="model-1",
        messages=[{"role": "user", "content": "x" * (agy_provider.MAX_PROMPT_CHARS + 1)}],
        max_tokens=100, timeout=10.0,
    )
    runner = FakeRunner()
    with pytest.raises(ProviderError, match="prompt too long"):
        agy_provider.agy_completion(long_request, "agy", runner=runner)
    assert runner.calls == []
