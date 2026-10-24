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

from foremankit import escalation
from foremankit import gateway
from foremankit import upstream
from foremankit.state import CallLog, StateStore

DEEPSEEK_URL = "http://deepseek.test/anthropic"
PROVIDER_KEY = "provider-secret"
KEY_ENV = "TEST_DS_KEY"

# Written to orchestrator.yaml: one declared external orchestrator; models are provider aliases resolved via providers.yaml.
UPSTREAMS_YAML = (
    "upstreams:\n"
    "  deepseek:\n"
    f"    url: {DEEPSEEK_URL}\n"
    f"    key_env: {KEY_ENV}\n"
    "    ceiling: normal\n"
    "    models: {opus: pro, sonnet: pro, haiku: flash}\n"
)

# providers.yaml is the single source of truth for ids and prices: the upstream aliases resolve through it.
PROVIDERS_YAML = (
    "providers:\n"
    "  deepseek:\n"
    "    models:\n"
    "      pro: {id: ds-pro, price: {input_miss: 1.32, input_hit: 0.044, output: 3.96}}\n"
    "      flash: {id: ds-flash, price: {input_miss: 0.3, input_hit: 0.006, output: 1.2}}\n"
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
    (config / "providers.yaml").write_text(PROVIDERS_YAML, encoding="utf-8")
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
    assert external[0] == "not-there" and external[1].get("name") == "not-there" and not external[1].get("url")


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
    assert state["spend"]["day_usd"] == pytest.approx(0.002134, abs=1e-9)
    rows = CallLog(tmp_path / "state" / "calls.jsonl").read()
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "orchestrator" and row["provider"] == "deepseek" and row["model"] == "ds-pro"
    assert row["tokens_in"] == 1500 and row["tokens_out"] == 200 and row["cache_hit"] == 500   # input 1000 miss + 500 hit
    assert row["cost_usd"] == pytest.approx(0.002134, abs=1e-9)


def test_the_main_loop_starts_cheap_and_escalates_after_repeated_tool_errors(monkeypatch):
    """Cheap first; two failing turns in a row move the main loop to the expensive alias, two clean turns move it back."""
    gateway._escalation.clear()
    monkeypatch.setattr(gateway.time, "time", lambda: 1000.0)
    up = {"name": "deepseek", "models": {"opus": "pro", "sonnet": "flash", "haiku": "flash"},
          "escalation": {"expensive": "pro", "errors": 2, "clean": 2, "min_expensive_s": 0}}
    error = {"messages": [{"role": "user", "content": [{"type": "tool_result", "is_error": True}]}]}
    clean = {"messages": [{"role": "user", "content": [{"type": "tool_result", "is_error": False}]}]}
    assert gateway.main_loop_alias("deepseek", up, clean) == "flash"
    assert gateway.main_loop_alias("deepseek", up, error) == "flash"
    assert gateway.main_loop_alias("deepseek", up, error) == "pro"
    assert gateway.main_loop_alias("deepseek", up, clean) == "pro"
    assert gateway.main_loop_alias("deepseek", up, clean) == "flash"
    gateway._escalation.clear()


def test_without_an_escalation_block_the_main_loop_keeps_the_configured_alias():
    gateway._escalation.clear()
    up = {"name": "deepseek", "models": {"sonnet": "flash"}}
    assert gateway.main_loop_alias("deepseek", up, {"messages": []}) == "flash"



def test_the_provider_model_id_comes_from_providers_yaml_so_a_rename_is_fixed_in_one_place(gateway_dirs: Path):
    """The upstream configures an alias (pro / flash); the id sent and the price both follow providers.yaml."""
    (gateway_dirs / "providers.yaml").write_text(
        "providers:\n  deepseek:\n    models:\n      pro: {id: deepseek-v5-pro, price: {input_miss: 1, output: 2}}\n",
        encoding="utf-8")
    assert gateway.model_id_for("deepseek", "pro") == "deepseek-v5-pro"
    assert gateway.model_id_for("deepseek", "unknown") == "unknown"      # an unknown alias is sent as written
    assert gateway.price_for_model("deepseek-v5-pro") == {"input_miss": 1, "output": 2}


def test_the_usage_reader_counts_a_message_start_buried_in_a_long_chunk():
    """A chunk longer than the kept tail must still be scanned from its start: the cache numbers sit in message_start."""
    collector = gateway.UsageCollector()
    start = b'event: message_start\ndata: {"message":{"usage":{"input_tokens":236,"cache_read_input_tokens":640}}}\n\n'
    collector.feed(start + b"event: content_block_delta\ndata: " + b"x" * 3000 + b"\n\n")
    collector.feed(b'event: message_delta\ndata: {"usage":{"output_tokens":8}}\n\n')
    assert collector.normalized() == {"prompt_tokens": 876, "cache_hit": 640, "completion_tokens": 8}


def test_cache_writes_are_prompt_tokens_that_are_not_hits():
    """Anthropic format: input_tokens excludes cache reads; writes are billed as misses."""
    collector = gateway.UsageCollector()
    collector.feed(b'"usage":{"input_tokens":100,"cache_creation_input_tokens":50,"cache_read_input_tokens":400,"output_tokens":7}')
    assert collector.normalized() == {"prompt_tokens": 550, "cache_hit": 400, "completion_tokens": 7}


def test_the_price_of_a_model_follows_the_peak_schedule(gateway_dirs: Path, monkeypatch: pytest.MonkeyPatch):
    """Listed prices are peak prices; outside the peak hours the answer is priced at the off-peak factor."""
    (gateway_dirs / "providers.yaml").write_text(
        "providers:\n  deepseek:\n    price_schedule: {peak_utc_hours: [[1, 4]], peak_weekdays: [0, 1, 2, 3, 4], off_peak_factor: 0.5}\n"
        "    models:\n      pro: {id: ds-pro, price: {input_miss: 2.0, input_hit: 0.1, output: 4.0}}\n", encoding="utf-8")
    from datetime import datetime, timezone
    friday_peak = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc).timestamp()
    saturday = datetime(2026, 10, 24, 2, 0, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(gateway.time, "time", lambda: friday_peak)
    assert gateway.price_for_model("ds-pro") == {"input_miss": 2.0, "input_hit": 0.1, "output": 4.0}
    monkeypatch.setattr(gateway.time, "time", lambda: saturday)
    assert gateway.price_for_model("ds-pro") == {"input_miss": 1.0, "input_hit": 0.05, "output": 2.0}


def test_the_main_loop_tier_is_written_for_the_panel(gateway_dirs: Path, tmp_path: Path):
    """The panel shows cheap/expensive from a small file the gateway keeps in the state dir."""
    gateway._escalation.clear()
    up = {"name": "deepseek", "models": {"sonnet": "flash"}, "escalation": {"expensive": "pro"}}
    gateway.main_loop_alias("deepseek", up, {"messages": []})
    written = json.loads((tmp_path / "state" / gateway.TIER_FILE).read_text(encoding="utf-8"))
    assert written["mode"] == "deepseek" and written["tier"] == "cheap" and written["alias"] == "flash"
    gateway._escalation.clear()


def test_the_orchestrator_budget_is_separate_from_the_delegation_budget(gateway_dirs: Path, tmp_path: Path, fake_upstream: list):
    """Spending the orchestrator cap refuses orchestrator requests (503) but leaves delegation untouched, and vice versa."""
    from foremankit import budget
    (gateway_dirs / "routing.yaml").write_text("budget: {daily_usd: 1.5, orchestrator_daily_usd: 1.5}\n", encoding="utf-8")
    import time as time_module
    now = time_module.time()
    store = StateStore(tmp_path / "state" / "state.json")
    store.update(lambda state: budget.record(state, now, 1.4, budget.DELEGATION))      # delegation nearly used up
    store.update(lambda state: budget.record(state, now, 0.1, budget.ORCHESTRATOR))
    use_mode("deepseek")
    assert post("/v1/messages", tooled_request()).status_code == 200                    # orchestrator still within 1.5
    store.update(lambda state: budget.record(state, now, 1.5, budget.ORCHESTRATOR))     # now 1.6 > 1.5
    refused = post("/v1/messages", tooled_request())
    assert refused.status_code == 503 and b"orchestrator daily budget" in refused.body
    state = store.load()
    assert budget.check(state, {"daily_usd": 1.5}, now, 0.0) is None                    # delegation (1.4) is not blocked
    assert budget.spent(state, now, budget.DELEGATION)[0] == pytest.approx(1.4)


# --- the panel switch over the escalation tier --------------------------------


def write_escalation_mode(state: Path, raw: str) -> None:
    """Write the panel switch file (`escalation_mode.json`) of the state dir with the given raw text."""
    (state / escalation.MODE_FILE).write_text(raw, encoding="utf-8")


def reset_escalation_state() -> None:
    """Forget the in-memory escalation state so each test starts from a known, order-independent point."""
    gateway._escalation.clear()
    gateway._escalation_control.clear()


def escalation_policy() -> dict:
    """The upstream config of the escalation tests: sonnet on flash, escalation to pro after two errors."""
    return {"name": "deepseek", "models": {"opus": "pro", "sonnet": "flash", "haiku": "flash"},
            "escalation": {"expensive": "pro", "errors": 2, "clean": 2, "min_expensive_s": 0}}


def clean_history() -> dict:
    """A one-turn history whose last tool_result succeeded."""
    return {"messages": [{"role": "user", "content": [{"type": "tool_result", "is_error": False}]}]}


def failing_history() -> dict:
    """A one-turn history whose last tool_result failed."""
    return {"messages": [{"role": "user", "content": [{"type": "tool_result", "is_error": True}]}]}


def test_a_pro_switch_pins_the_main_loop_to_the_expensive_alias(gateway_dirs: Path, tmp_path: Path):
    """{"mode": "pro"} in the panel file sends the very next clean turn to the expensive alias."""
    reset_escalation_state()
    write_escalation_mode(tmp_path / "state", '{"mode": "pro"}')
    alias = gateway.main_loop_alias("deepseek", escalation_policy(), clean_history())
    assert alias == "pro"
    written = json.loads((tmp_path / "state" / gateway.TIER_FILE).read_text(encoding="utf-8"))
    assert written["tier"] == "expensive" and written["control"] == "pro"


def test_a_flash_switch_overrides_an_escalated_main_loop_immediately(gateway_dirs: Path, tmp_path: Path):
    """After auto escalated to pro on tool errors, {"mode": "flash"} brings the next call straight back to flash."""
    reset_escalation_state()
    up = escalation_policy()
    assert gateway.main_loop_alias("deepseek", up, failing_history()) == "flash"
    assert gateway.main_loop_alias("deepseek", up, failing_history()) == "pro"
    write_escalation_mode(tmp_path / "state", '{"mode": "flash"}')
    assert gateway.main_loop_alias("deepseek", up, failing_history()) == "flash"
    written = json.loads((tmp_path / "state" / gateway.TIER_FILE).read_text(encoding="utf-8"))
    assert written["tier"] == "cheap" and written["control"] == "flash"


def test_going_back_to_auto_restarts_the_main_loop_on_the_cheap_alias(gateway_dirs: Path, tmp_path: Path):
    """Leaving the pro pin for auto starts from a fresh state, so a clean turn runs on flash, not on pro."""
    reset_escalation_state()
    write_escalation_mode(tmp_path / "state", '{"mode": "pro"}')
    assert gateway.main_loop_alias("deepseek", escalation_policy(), clean_history()) == "pro"
    write_escalation_mode(tmp_path / "state", '{"mode": "auto"}')
    assert gateway.main_loop_alias("deepseek", escalation_policy(), clean_history()) == "flash"
    written = json.loads((tmp_path / "state" / gateway.TIER_FILE).read_text(encoding="utf-8"))
    assert written["control"] == "auto"


@pytest.mark.parametrize("raw", [None, "{not json", '{"mode": "sideways"}'])
def test_a_missing_or_broken_panel_file_means_auto(gateway_dirs: Path, tmp_path: Path, raw):
    """No file, invalid JSON or an unknown mode all fall back to auto, so the gateway behaves as before."""
    path = tmp_path / "state" / escalation.MODE_FILE
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    assert gateway.escalation_mode() == escalation.AUTO
