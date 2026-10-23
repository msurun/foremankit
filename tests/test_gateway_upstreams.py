"""Tests for the external-orchestrator routing of the gateway (foremankit.gateway).

The gateway forwards Claude Code to a third-party Anthropic-compatible endpoint when the orchestrator mode
is not a built-in one. Every test runs against the app with a fake httpx upstream, so nothing leaves the
machine; configuration lives in a temporary config dir and the mode in a temporary state dir.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from starlette.requests import Request

from foremankit import gateway
from foremankit import upstream
from foremankit.state import CallLog, StateStore

DEEPSEEK_URL = "http://deepseek.test/anthropic"
PROVIDER_KEY = "provider-secret"
KEY_ENV = "TEST_DS_KEY"

# Written to orchestrator.yaml: one declared external orchestrator with aliases for the three model tiers.
UPSTREAMS_YAML = (
    "upstreams:\n"
    "  deepseek:\n"
    f"    url: {DEEPSEEK_URL}\n"
    f"    key_env: {KEY_ENV}\n"
    "    ceiling: normal\n"
    "    models:\n"
    "      opus: ds-pro\n"
    "      sonnet: ds-pro\n"
    "      haiku: ds-flash\n"
)

EXTERNAL_PROMPT = "EXTERNAL RULES"


# --- fixtures ---------------------------------------------------------------

class FakeUpstreamResponse:
    """Stands in for an httpx streaming response: a status, headers and a small JSON body."""
    status_code = 200
    headers = {"content-type": "application/json"}

    async def aiter_raw(self):
        yield b"{}"

    async def aclose(self):
        return None


@pytest.fixture
def fake_upstream(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replaces the shared httpx client so each forwarded request is recorded instead of sent."""
    sent: list = []

    async def fake_send(request, stream=True):
        sent.append(request)
        return FakeUpstreamResponse()

    monkeypatch.setattr(gateway.client, "send", fake_send)
    return sent


@pytest.fixture
def gateway_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Config and state dirs in tmp_path, wired via the env vars; the local role is orchestrator."""
    config = tmp_path / "config"
    state = tmp_path / "state"
    config.mkdir()
    state.mkdir()
    (config / "orchestrator.yaml").write_text(UPSTREAMS_YAML, encoding="utf-8")
    (config / "orchestrator.external.md").write_text(EXTERNAL_PROMPT, encoding="utf-8")
    (state / gateway.LOCAL_ROLE_FILE).write_text("orchestrator", encoding="utf-8")
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(config))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(state))
    monkeypatch.setenv(KEY_ENV, PROVIDER_KEY)
    upstream.set_mode("deepseek")      # the default for these tests; a test can switch with use_mode()
    return config


def use_mode(mode: str) -> None:
    """Switch the orchestrator mode as the panel/CLI would (state dir already holds the mode file)."""
    upstream.set_mode(mode)


def post(path: str, data: dict):
    """Send one POST with a JSON body through gateway.handle, as Claude Code would; returns the response."""
    body = json.dumps(data).encode("utf-8")
    scope = {"type": "http", "method": "POST", "path": path, "query_string": b"",
             "headers": [(b"content-type", b"application/json"), (b"x-api-key", b"user-key"),
                         (b"authorization", b"Bearer user-token"), (b"anthropic-beta", b"oauth-2025-04-20")]}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return asyncio.run(gateway.handle(Request(scope, receive)))


def sent_model(request) -> str:
    """The "model" field of a recorded upstream request body."""
    return json.loads(request.content)["model"]


def sent_headers(request) -> dict:
    """Lower-cased header names of a recorded upstream request."""
    return {name.lower(): value for name, value in request.headers.items()}


# --- external_mode ----------------------------------------------------------

@pytest.mark.parametrize("builtin", sorted(upstream.BUILTIN_MODES))
def test_a_builtin_mode_is_not_external(builtin: str, gateway_dirs: Path):
    use_mode(builtin)
    assert gateway.external_mode(gateway.settings()) is None


def test_a_declared_external_mode_returns_its_name_and_config(gateway_dirs: Path):
    use_mode("deepseek")
    external = gateway.external_mode(gateway.settings())
    assert external is not None
    name, up = external
    assert name == "deepseek" and up["url"] == DEEPSEEK_URL


def test_an_undeclared_mode_is_external_with_an_empty_config(gateway_dirs: Path):
    use_mode("not-there")
    external = gateway.external_mode(gateway.settings())
    assert external == ("not-there", {})


# --- external_refusal -------------------------------------------------------

def test_an_undeclared_mode_is_refused_with_the_reason(gateway_dirs: Path):
    reason = gateway.external_refusal("not-there", {})
    assert reason is not None and "not declared" in reason


def test_a_declared_mode_without_its_key_env_is_refused(
        gateway_dirs: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(KEY_ENV)
    _, up = gateway.external_mode(gateway.settings())
    reason = gateway.external_refusal("deepseek", up)
    assert reason is not None and KEY_ENV in reason


def test_a_declared_mode_with_its_key_env_is_allowed(gateway_dirs: Path):
    _, up = gateway.external_mode(gateway.settings())
    assert gateway.external_refusal("deepseek", up) is None


# --- shape_external ---------------------------------------------------------

def tooled_request() -> dict:
    return {"model": "claude-sonnet-5", "system": "# Environment\ncwd=/x",
            "tools": [{"name": "Read", "description": "read"}],
            "messages": [{"role": "user", "content": "hi"}]}


def toolless_request() -> dict:
    return {"model": "claude-sonnet-5", "system": "short system",
            "messages": [{"role": "user", "content": "hi"}]}


def main_loop_request() -> dict:
    return {"model": "claude-sonnet-5", "system": "# Environment\ncwd=/x",
            "tools": [{"name": "Agent", "description": "spawn"}],
            "messages": [{"role": "user", "content": "hi"}]}


def test_a_tooled_sonnet_request_is_aliased_to_the_provider_model(gateway_dirs: Path):
    _, up = gateway.external_mode(gateway.settings())
    shaped, note = gateway.shape_external(tooled_request(), up, gateway.settings())
    assert shaped["model"] == "ds-pro" and "ds-pro" in note


def test_a_toolless_request_uses_the_cheap_alias(gateway_dirs: Path):
    _, up = gateway.external_mode(gateway.settings())
    shaped, note = gateway.shape_external(toolless_request(), up, gateway.settings())
    assert shaped["model"] == "ds-flash" and "side request" in note


def test_the_main_loop_keeps_its_tools_and_gets_the_external_prompt_appended(gateway_dirs: Path):
    _, up = gateway.external_mode(gateway.settings())
    shaped, note = gateway.shape_external(main_loop_request(), up, gateway.settings())
    assert [tool["name"] for tool in shaped["tools"]] == ["Agent"]
    assert shaped["system"][-1] == {"type": "text", "text": EXTERNAL_PROMPT}
    assert "main loop" in note


# --- external_headers -------------------------------------------------------

def asgi_request(headers: list[tuple[bytes, bytes]]) -> Request:
    """Build a Starlette Request from a minimal ASGI scope carrying the given headers."""
    scope = {"type": "http", "headers": headers, "method": "POST", "path": "/v1/messages", "query_string": b""}
    return Request(scope)


def test_external_headers_replace_the_client_credentials_with_the_provider_key(gateway_dirs: Path):
    _, up = gateway.external_mode(gateway.settings())
    request = asgi_request([(b"x-api-key", b"user-key"), (b"authorization", b"Bearer user-token")])
    headers = gateway.external_headers(request, up)
    assert headers["x-api-key"] == PROVIDER_KEY
    assert headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_external_headers_drop_anthropic_beta(gateway_dirs: Path):
    _, up = gateway.external_mode(gateway.settings())
    request = asgi_request([(b"anthropic-beta", b"oauth-2025-04-20")])
    headers = gateway.external_headers(request, up)
    assert "anthropic-beta" not in {name.lower() for name in headers}


# --- handle_external and handle --------------------------------------------

def test_a_tooled_messages_request_goes_to_the_deepseek_url_with_the_provider_key(
        gateway_dirs: Path, fake_upstream: list):
    use_mode("deepseek")
    post("/v1/messages", tooled_request())
    sent = fake_upstream[-1]
    assert str(sent.url).startswith(f"{DEEPSEEK_URL}/v1/messages")
    assert sent_headers(sent)["x-api-key"] == PROVIDER_KEY
    assert sent_headers(sent)["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_a_tooled_messages_request_drops_anthropic_beta(gateway_dirs: Path, fake_upstream: list):
    use_mode("deepseek")
    post("/v1/messages", tooled_request())
    assert "anthropic-beta" not in sent_headers(fake_upstream[-1])


def test_a_tooled_messages_request_is_sent_with_the_sonnet_alias(
        gateway_dirs: Path, fake_upstream: list):
    use_mode("deepseek")
    post("/v1/messages", tooled_request())
    assert sent_model(fake_upstream[-1]) == "ds-pro"


def test_a_toolless_messages_request_is_sent_with_the_cheap_alias(
        gateway_dirs: Path, fake_upstream: list):
    use_mode("deepseek")
    post("/v1/messages", toolless_request())
    assert sent_model(fake_upstream[-1]) == "ds-flash"


def test_a_main_loop_request_carries_tools_and_the_external_prompt(
        gateway_dirs: Path, fake_upstream: list):
    use_mode("deepseek")
    post("/v1/messages", main_loop_request())
    body = json.loads(fake_upstream[-1].content)
    assert [tool["name"] for tool in body["tools"]] == ["Agent"]
    assert body["system"][-1]["text"] == EXTERNAL_PROMPT


def test_a_non_messages_path_is_forwarded_unchanged_with_the_provider_key(
        gateway_dirs: Path, fake_upstream: list):
    use_mode("deepseek")
    data = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}]}
    post("/v1/messages/count_tokens", data)
    sent = fake_upstream[-1]
    assert str(sent.url).startswith(f"{DEEPSEEK_URL}/v1/messages/count_tokens")
    assert json.loads(sent.content) == data
    assert sent_headers(sent)["x-api-key"] == PROVIDER_KEY


def test_an_undeclared_mode_answers_503_and_sends_nothing(
        gateway_dirs: Path, fake_upstream: list):
    use_mode("not-there")
    response = post("/v1/messages", tooled_request())
    assert response.status_code == 503
    error = json.loads(response.body)["error"]
    assert error["type"] == "api_error" and "not declared" in error["message"]
    assert fake_upstream == []


def test_a_missing_key_env_answers_503_and_sends_nothing(
        gateway_dirs: Path, fake_upstream: list, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(KEY_ENV)
    use_mode("deepseek")
    response = post("/v1/messages", tooled_request())
    assert response.status_code == 503
    assert KEY_ENV in json.loads(response.body)["error"]["message"]
    assert fake_upstream == []


def test_claude_only_blocks_become_text_and_tool_search_flags_are_dropped(gateway_dirs: Path):
    """A tool_definition block (tool search) must not reach a third-party API; standard blocks and tool_reference stay."""
    request = {"model": "claude-sonnet-5", "tools": [{"name": "X", "defer_loading": True, "description": "d"}],
               "messages": [{"role": "user", "content": [
                   {"type": "tool_definition", "name": "Y"}, {"type": "text", "text": "hi"},
                   {"type": "tool_result", "tool_use_id": "1", "content": [{"type": "tool_reference", "tool_name": "Y"},
                                                                          {"type": "tool_definition", "name": "Z"}]}]},
                   {"role": "assistant", "content": "plain string"}]}
    shaped = gateway.plain_request(request)
    first = shaped["messages"][0]["content"]
    assert first[0]["type"] == "text" and "tool_definition" in first[0]["text"] and first[1] == {"type": "text", "text": "hi"}
    inner = first[2]["content"]
    assert inner[0]["type"] == "tool_reference" and inner[1]["type"] == "text"
    assert shaped["messages"][1]["content"] == "plain string" and "defer_loading" not in shaped["tools"][0]


def test_a_normal_claude_model_still_goes_to_anthropic_in_an_external_mode(gateway_dirs: Path, fake_upstream: list):
    """Only the ForemanKit picker entry follows the mode: Sonnet chosen in the picker is sent to the cloud, key untouched."""
    use_mode("deepseek")
    post("/v1/messages", {**tooled_request(), "model": "claude-sonnet-5-5"})
    sent = fake_upstream[-1]
    assert str(sent.url).startswith("https://api.anthropic.com/v1/messages")
    assert sent_model(sent) == "claude-sonnet-5-5" and sent_headers(sent)["x-api-key"] == "user-key"


def test_tools_outside_tools_keep_are_removed_and_schemas_made_acceptable():
    """Artifact is no use to an external model and its schema pattern is refused by DeepSeek: it is cut, the rest is cleaned."""
    request = {"model": "claude-sonnet-5", "context_management": {"x": 1}, "messages": [], "tools": [
        {"name": "Artifact", "input_schema": {"type": "object", "properties": {"p": {"type": "string", "pattern": "^[^\0]*$"}}}},
        {"name": "Read", "defer_loading": True, "input_schema": {"type": "object", "properties": {
            "file_path": {"type": "string", "format": "path"}}}}]}
    shaped = gateway.adapt_for_provider(request, {"name": "deepseek", "tools_keep": ["Read", "mcp__foremankit__*"]})
    assert [tool["name"] for tool in shaped["tools"]] == ["Read"] and "context_management" not in shaped
    text = json.dumps(shaped)
    assert '"format"' not in text and "defer_loading" not in text and "format: " in text


def test_a_provider_error_is_returned_unchanged_and_recorded(gateway_dirs: Path, monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path):
    """A 4xx from the external provider reaches the plugin as it is and request + answer are saved for diagnosis."""
    class Refused(FakeUpstreamResponse):
        status_code = 400
        async def aread(self):
            return b'{"error": "Invalid schema"}'

    async def fake_send(request, stream=True):
        return Refused()

    monkeypatch.setattr(gateway.client, "send", fake_send)
    response = post("/v1/messages", tooled_request())
    assert response.status_code == 400 and response.body == b'{"error": "Invalid schema"}'
    saved = list((tmp_path / "state" / gateway.ERRORS_DIR).glob("*-400.json"))
    assert len(saved) == 1
    record = json.loads(saved[0].read_text(encoding="utf-8"))
    assert record["answer"] == '{"error": "Invalid schema"}' and record["request"]["model"] == "ds-pro"


def test_an_external_answer_is_counted_in_the_budget_and_call_log(gateway_dirs: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   tmp_path: Path):
    """Token usage of a streamed DeepSeek answer is priced and recorded like any delegated call (spend counter)."""
    (tmp_path / "config" / "providers.yaml").write_text(
        "providers:\n  deepseek:\n    models:\n      pro:\n        id: ds-pro\n"
        "        price: {input_miss: 1.32, input_hit: 0.044, output: 3.96}\n", encoding="utf-8")

    class SseResponse(FakeUpstreamResponse):
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            yield (b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
                   b'{"input_tokens":1000,"cache_read_input_tokens":500,"output_tokens":1}}}\n\n')
            yield b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":200}}\n\n'

    async def fake_send(request, stream=True):
        return SseResponse()

    monkeypatch.setattr(gateway.client, "send", fake_send)

    async def drain(response):
        async for _ in response.body_iterator:
            pass

    use_mode("deepseek")
    asyncio.run(drain(post("/v1/messages", tooled_request())))
    state = StateStore(tmp_path / "state" / "state.json").load()
    assert state["spend"]["day_usd"] == pytest.approx(0.001474, abs=1e-9)
    rows = CallLog(tmp_path / "state" / "calls.jsonl").read()
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "orchestrator" and row["provider"] == "deepseek" and row["model"] == "ds-pro"
    assert row["tokens_in"] == 1000 and row["tokens_out"] == 200 and row["cache_hit"] == 500
    assert row["cost_usd"] == pytest.approx(0.001474, abs=1e-9)

