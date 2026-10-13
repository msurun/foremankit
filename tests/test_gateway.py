"""Behaviour tests for the gateway request shaping (foremankit.gateway_trim) and the gateway helpers
(foremankit.gateway).

All tests are pure and deterministic: no network, no llama-server. The gateway helpers read their
configuration from FOREMANKIT_CONFIG_DIR, which every test points at a fresh tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.requests import Request

from foremankit import gateway
from foremankit import gateway_trim as trim

# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def main_loop_request() -> dict:
    """A minimal plugin main-loop request: it carries the Agent tool that spawns subagents."""
    return {
        "model": "claude-sonnet-5",
        "system": "# Environment\ncwd=/home/user\n\n# Other\nignore me",
        "tools": [
            {"name": "Agent", "description": "spawn a subagent"},
            {"name": "Bash", "description": "x" * 50},
            {"name": "Noise", "description": "should be dropped"},
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>The following skills are available: foo</system-reminder>"},
                {"type": "text", "text": "do the thing"},
            ]},
        ],
    }


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary config dir holding orchestrator.yaml and orchestrator.md, wired via the env var."""
    (tmp_path / "orchestrator.md").write_text("  ORCHESTRATOR PROMPT  \n", encoding="utf-8")
    (tmp_path / "orchestrator.yaml").write_text(
        "trim:\n"
        "  enabled: true\n"
        "  system: replace\n"
        "  keep_sections: [Environment]\n"
        "  tools_keep: [Agent, Bash]\n"
        "  tool_description_max_chars: 5\n"
        "  drop_message_blocks: ['The following skills are available']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    return tmp_path


def asgi_request(headers: list[tuple[bytes, bytes]]) -> Request:
    """Build a Starlette Request from a minimal ASGI scope carrying the given headers."""
    scope = {"type": "http", "headers": headers, "method": "POST", "path": "/v1/messages", "query_string": b""}
    return Request(scope)


# --- gateway_trim: is_main_loop --------------------------------------------

def test_a_request_with_the_agent_tool_is_the_main_loop():
    data = {"tools": [{"name": "Agent"}]}
    assert trim.is_main_loop(data, ["Agent", "Task"]) is True


def test_a_request_without_a_spawn_tool_is_not_the_main_loop():
    data = {"tools": [{"name": "Bash"}]}
    assert trim.is_main_loop(data, ["Agent", "Task"]) is False


# --- gateway_trim: system_text ---------------------------------------------

def test_system_text_returns_a_string_unchanged():
    assert trim.system_text("hello") == "hello"


def test_system_text_joins_the_text_of_a_list_of_blocks():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert trim.system_text(blocks) == "a\nb"


def test_system_text_of_none_is_empty():
    assert trim.system_text(None) == ""


# --- gateway_trim: keep_sections -------------------------------------------

def test_keep_sections_picks_only_the_named_headings_case_insensitively():
    text = "# environment\ncwd=/x\n\n# other\nnoise"
    assert trim.keep_sections(text, ["Environment"]) == "# environment\ncwd=/x"


# --- gateway_trim: shape_system --------------------------------------------

def test_shape_system_replace_uses_the_prompt_and_kept_sections():
    original = "# Environment\ncwd=/x\n\n# Other\nnoise"
    shaped = trim.shape_system(original, "PROMPT", {"system": "replace", "keep_sections": ["Environment"]})
    assert shaped == [{"type": "text", "text": "PROMPT\n\n# Environment\ncwd=/x"}]


def test_shape_system_append_puts_the_prompt_after_the_original():
    shaped = trim.shape_system("# Environment\ncwd=/x", "PROMPT", {"system": "append"})
    assert shaped == [
        {"type": "text", "text": "# Environment\ncwd=/x"},
        {"type": "text", "text": "PROMPT"},
    ]


def test_shape_system_keep_returns_the_original():
    original = "# Environment\ncwd=/x"
    assert trim.shape_system(original, "PROMPT", {"system": "keep"}) == original


def test_shape_system_with_an_empty_prompt_returns_the_original():
    original = "# Environment\ncwd=/x"
    assert trim.shape_system(original, "", {"system": "replace"}) == original


def test_shape_system_when_trim_is_disabled_behaves_as_append():
    shaped = trim.shape_system("orig", "PROMPT", {"enabled": False, "system": "replace"})
    assert shaped == [{"type": "text", "text": "orig"}, {"type": "text", "text": "PROMPT"}]


# --- gateway_trim: trim_tools ----------------------------------------------

def test_trim_tools_keeps_an_exact_name():
    tools = [{"name": "Bash", "description": "run a shell"}]
    assert trim.trim_tools(tools, ["Bash"], 0) == [{"name": "Bash", "description": "run a shell"}]


def test_trim_tools_keeps_tools_matching_an_fnmatch_pattern():
    tools = [{"name": "mcp__foremankit__foo", "description": "x"}]
    kept = trim.trim_tools(tools, ["mcp__foremankit__*"], 0)
    assert [tool["name"] for tool in kept] == ["mcp__foremankit__foo"]


def test_trim_tools_cuts_a_long_description_with_an_ellipsis():
    tools = [{"name": "Bash", "description": "abcdefghij"}]
    allowed = 4  # keep "abcd", drop the rest and append the ellipsis
    kept = trim.trim_tools(tools, ["Bash"], allowed)
    assert kept[0]["description"] == "abcd" + trim.ELLIPSIS


def test_trim_tools_with_zero_char_limit_does_not_cut_descriptions():
    tools = [{"name": "Bash", "description": "a very long description indeed"}]
    kept = trim.trim_tools(tools, ["Bash"], 0)
    assert kept[0]["description"] == "a very long description indeed"


# --- gateway_trim: trim_messages -------------------------------------------

def test_trim_messages_drops_a_matching_reminder_text_block():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>The following skills are available</system-reminder>"},
        {"type": "text", "text": "real work"},
    ]}]
    shaped = trim.trim_messages(messages, ["The following skills are available"])
    assert shaped[0]["content"] == [{"type": "text", "text": "real work"}]


def test_trim_messages_keeps_a_normal_text_that_only_contains_the_pattern():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "The following skills are available in the docs"},
    ]}]
    shaped = trim.trim_messages(messages, ["The following skills are available"])
    assert shaped[0]["content"] == [{"type": "text", "text": "The following skills are available in the docs"}]


def test_trim_messages_never_drops_tool_result_blocks():
    messages = [{"role": "user", "content": [
        {"type": "tool_result", "content": "The following skills are available"},
    ]}]
    shaped = trim.trim_messages(messages, ["The following skills are available"])
    assert shaped[0]["content"] == [{"type": "tool_result", "content": "The following skills are available"}]


def test_trim_messages_never_touches_assistant_messages():
    messages = [{"role": "assistant", "content": [
        {"type": "text", "text": "<system-reminder>The following skills are available</system-reminder>"},
    ]}]
    shaped = trim.trim_messages(messages, ["The following skills are available"])
    assert shaped == messages


def test_trim_messages_keeps_a_message_with_all_blocks_dropped_unchanged():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>The following skills are available</system-reminder>"},
    ]}]
    shaped = trim.trim_messages(messages, ["The following skills are available"])
    assert shaped == messages


# --- gateway_trim: shape_request and request_chars -------------------------

def test_shape_request_replaces_system_keeps_named_tools_and_drops_reminder(
        main_loop_request: dict):
    shaped = trim.shape_request(main_loop_request, "PROMPT", {
        "enabled": True, "system": "replace", "keep_sections": ["Environment"],
        "tools_keep": ["Agent", "Bash"], "tool_description_max_chars": 0,
        "drop_message_blocks": ["The following skills are available"],
    })
    assert shaped["system"] == [{"type": "text", "text": "PROMPT\n\n# Environment\ncwd=/home/user"}]
    assert [tool["name"] for tool in shaped["tools"]] == ["Agent", "Bash"]
    assert shaped["messages"][0]["content"] == [{"type": "text", "text": "do the thing"}]


def test_shape_request_does_not_mutate_the_input(main_loop_request: dict):
    before = json.dumps(main_loop_request, sort_keys=True)
    trim.shape_request(main_loop_request, "PROMPT", {
        "enabled": True, "system": "replace", "keep_sections": ["Environment"],
        "tools_keep": ["Agent"], "tool_description_max_chars": 0,
        "drop_message_blocks": ["The following skills are available"],
    })
    assert json.dumps(main_loop_request, sort_keys=True) == before


def test_request_chars_grows_when_content_is_added():
    small = trim.request_chars({"system": "hi"})
    large = trim.request_chars({"system": "hi", "messages": [{"role": "user", "content": "hello"}]})
    assert large > small


# --- gateway: settings and read_prompt -------------------------------------

def test_settings_merges_the_yaml_over_the_defaults(config_dir: Path):
    cfg = gateway.settings()
    assert cfg["port"] == gateway.DEFAULTS["port"]
    assert cfg["trim"]["tools_keep"] == ["Agent", "Bash"]


def test_read_prompt_returns_the_stripped_file_text(config_dir: Path):
    assert gateway.read_prompt(gateway.settings()) == "ORCHESTRATOR PROMPT"


def test_read_prompt_returns_empty_string_when_the_file_is_missing(config_dir: Path):
    (config_dir / "orchestrator.md").unlink()
    assert gateway.read_prompt(gateway.settings()) == ""


# --- gateway: shape_local --------------------------------------------------

def test_shape_local_on_the_main_loop_rewrites_the_request(main_loop_request: dict, config_dir: Path):
    data, note = gateway.shape_local(dict(main_loop_request), gateway.settings())
    assert data["model"] == "local-llm"
    assert data["system"] == [{"type": "text", "text": "ORCHESTRATOR PROMPT\n\n# Environment\ncwd=/home/user"}]
    assert [tool["name"] for tool in data["tools"]] == ["Agent", "Bash"]
    assert "main loop" in note


def test_shape_local_without_a_spawn_tool_only_rewrites_the_model(main_loop_request: dict, config_dir: Path):
    request = dict(main_loop_request)
    request["tools"] = [tool for tool in request["tools"] if tool["name"] != "Agent"]
    data, note = gateway.shape_local(request, gateway.settings())
    assert data["model"] == "local-llm"
    assert data["system"] == main_loop_request["system"]
    assert "subagent" in note


def test_shape_local_writes_the_shaped_json_when_the_dump_env_is_set(
        main_loop_request: dict, config_dir: Path, monkeypatch: pytest.MonkeyPatch):
    dump_path = config_dir / "dump.json"
    monkeypatch.setenv(gateway.DUMP_ENV, str(dump_path))
    gateway.shape_local(dict(main_loop_request), gateway.settings())
    dumped = json.loads(dump_path.read_text(encoding="utf-8"))
    assert dumped["model"] == "local-llm"


# --- gateway: upstream_headers ---------------------------------------------

def test_upstream_headers_for_local_strips_credentials_and_adds_the_peer_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FOREMANKIT_PEER_KEY", "secret")
    request = asgi_request([(b"x-api-key", b"k"), (b"authorization", b"Bearer cloud")])
    headers = gateway.upstream_headers(request, local=True)
    assert "x-api-key" not in {name.lower() for name in headers}
    assert headers["authorization"] == "Bearer secret"


def test_upstream_headers_for_cloud_keeps_credentials_and_drops_hop_by_hop_headers():
    request = asgi_request([(b"x-api-key", b"k"), (b"connection", b"keep-alive")])
    headers = gateway.upstream_headers(request, local=False)
    assert headers["x-api-key"] == "k"
    assert "connection" not in {name.lower() for name in headers}


# --- gateway: parse_body ---------------------------------------------------

def test_parse_body_of_empty_input_is_none():
    assert gateway.parse_body(b"") is None


def test_parse_body_of_invalid_json_is_none():
    assert gateway.parse_body(b"{oops") is None


def test_parse_body_of_a_json_array_is_none():
    assert gateway.parse_body(b"[1, 2]") is None


def test_parse_body_of_a_json_object_is_the_dict():
    assert gateway.parse_body(b'{"model": "x"}') == {"model": "x"}


# --- gateway: target_url ---------------------------------------------------

def test_target_url_appends_path_and_query():
    request = asgi_request([])
    cfg = {"local_url": "http://127.0.0.1:8080", "cloud_url": "https://api.anthropic.com"}
    url = gateway.target_url(request, cfg, local=True)
    assert url == "http://127.0.0.1:8080/v1/messages"


def test_target_url_uses_the_cloud_base_when_not_local():
    request = asgi_request([])
    cfg = {"local_url": "http://127.0.0.1:8080", "cloud_url": "https://api.anthropic.com"}
    url = gateway.target_url(request, cfg, local=False)
    assert url == "https://api.anthropic.com/v1/messages"


def _result(text, content_type="str"):
    content = text if content_type == "str" else [{"type": "text", "text": text}]
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": content}]}


def test_prune_old_tool_results_cuts_old_keeps_recent():
    messages = [_result("x" * 5000), _result("y" * 5000, "list"), _result("z" * 5000)]
    out = trim.prune_old_tool_results(messages, keep_recent=1, limit=100)
    assert len(out[0]["content"][0]["content"]) < 300 and "characters removed" in out[0]["content"][0]["content"]
    assert len(out[1]["content"][0]["content"][0]["text"]) < 300
    assert out[2] == messages[2]
    assert messages[0]["content"][0]["content"] == "x" * 5000   # input not mutated


def test_prune_old_tool_results_off_when_limit_zero_or_text_short():
    messages = [_result("x" * 5000), _result("short")]
    assert trim.prune_old_tool_results(messages, 0, 0) == messages
    assert trim.prune_old_tool_results(messages, 0, 100)[1] == messages[1]


def test_trim_system_messages_keeps_only_wanted_sections():
    text = "<system-reminder>\n# Environment\ncwd here\n\n# MCP Server Instructions\nlots of text\n</system-reminder>"
    out = trim.trim_system_messages([{"role": "system", "content": text}, {"role": "user", "content": "hi"}], ["Environment"])
    assert "cwd here" in out[0]["content"] and "lots of text" not in out[0]["content"]
    assert out[1]["content"] == "hi"
    untouched = [{"role": "system", "content": "no headings"}]
    assert trim.trim_system_messages(untouched, ["Environment"]) == untouched


def test_shape_local_records_context_usage_for_the_session_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "orchestrator.yaml").write_text("context_tokens: 65536\n", encoding="utf-8")
    monkeypatch.setattr(gateway, "server_window", lambda cfg: None)
    request = {"model": "claude-sonnet-5", "system": "s", "tools": [{"name": "Agent"}],
               "messages": [{"role": "user", "content": "x" * 4000}]}
    gateway.shape_local(request, gateway.settings())
    usage = json.loads((gateway.state_dir() / "gateway_context.json").read_text(encoding="utf-8"))
    assert usage["window"] == 65536 and usage["tokens"] > 900


def test_session_hook_gets_the_window_the_server_really_has(tmp_path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    (tmp_path / "orchestrator.yaml").write_text("context_tokens: 98304\n", encoding="utf-8")
    monkeypatch.setattr(gateway, "server_window", lambda cfg: 32768)
    gateway.record_usage(1000, gateway.settings())
    usage = json.loads((gateway.state_dir() / "gateway_context.json").read_text(encoding="utf-8"))
    assert usage["window"] == 32768


def test_read_prompt_fills_in_the_conversation_language(config_dir: Path):
    (config_dir / "orchestrator.md").write_text("Talk in {language}.", encoding="utf-8")
    assert gateway.read_prompt({**gateway.settings(), "language": "Spanish"}) == "Talk in Spanish."


def test_the_default_conversation_language_is_english(config_dir: Path):
    (config_dir / "orchestrator.md").write_text("Talk in {language}.", encoding="utf-8")
    assert gateway.read_prompt(gateway.settings()) == "Talk in English."


def test_effort_slider_decides_whether_the_local_model_thinks():
    by_effort = gateway.settings()["thinking_by_effort"]
    for effort, expected in (("low", False), ("medium", False), ("high", True), ("xhigh", True), ("max", True)):
        assert trim.thinking_wanted({"output_config": {"effort": effort}}, by_effort) is expected
    assert trim.thinking_wanted({"messages": []}, by_effort) is None


def test_anthropic_thinking_field_is_used_when_there_is_no_effort():
    assert trim.thinking_wanted({"thinking": {"type": "disabled"}}, {}) is False
    assert trim.thinking_wanted({"thinking": {"type": "enabled", "budget_tokens": 4000}}, {}) is True


def test_shape_local_sets_enable_thinking_per_request(tmp_path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(tmp_path))
    request = {"model": "claude-sonnet-5", "output_config": {"effort": "low"}, "messages": [{"role": "user", "content": "x"}]}
    shaped, note = gateway.shape_local(request, gateway.settings())
    assert shaped["chat_template_kwargs"] == {"enable_thinking": False} and "thinking=off" in note


def test_worker_role_refuses_claude_code_requests_but_orchestrator_and_both_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    assert gateway.refusal() is None
    for role, refused in (("worker", True), ("orchestrator", False), ("both", False)):
        (tmp_path / gateway.LOCAL_ROLE_FILE).write_text(role, encoding="utf-8")
        assert (gateway.refusal() is not None) is refused


def test_refused_local_request_gets_a_503_in_the_anthropic_error_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path))
    (tmp_path / gateway.LOCAL_ROLE_FILE).write_text("worker", encoding="utf-8")
    response = gateway.switched_off(gateway.refusal())
    assert response.status_code == 503 and b"worker" in response.body
