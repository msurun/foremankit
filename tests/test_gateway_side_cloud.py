"""Tests for gateway side requests: toolless /v1/messages calls go to the cloud first, then the fallback upstream.

No network: the shared httpx client is replaced by a recorder, and the external orchestrator is stubbed.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from starlette.requests import Request

from foremankit import gateway

KAGGLE = {"url": "https://kaggle.example", "key_env": "KAGGLE_KEY", "models": {"sonnet": "main", "haiku": "main"}}
DEEPSEEK = {"url": "https://deepseek.example", "key_env": "DEEPSEEK_KEY",
            "models": {"sonnet": "ds-pro", "haiku": "ds-flash"}}
MESSAGE = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}]}


class FakeResponse:
    """Stands in for an httpx response: status, headers, body and the streaming hooks the relay uses."""

    def __init__(self, status: int = 200, content: bytes = b'{"type": "message"}'):
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self.content = content

    async def aiter_raw(self):
        yield self.content

    async def aclose(self):
        return None


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list:
    """Records each request sent upstream; a side_cloud answer is set by the test through `plan`."""
    calls: list = []
    monkeypatch.setattr(gateway.client, "send", _recorder(calls))
    return calls


def _recorder(calls: list):
    async def fake_send(request, stream=False):
        calls.append(request)
        plan = _plan.get(str(request.url).split("/v1/")[0])
        if isinstance(plan, Exception):
            raise plan
        return plan or FakeResponse()
    return fake_send


_plan: dict = {}


@pytest.fixture
def gateway_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Config and state dirs in tmp_path; the orchestrator.yaml declares Kaggle and the DeepSeek fallback."""
    config, state = tmp_path / "config", tmp_path / "state"
    config.mkdir()
    state.mkdir()
    (state / gateway.LOCAL_ROLE_FILE).write_text("orchestrator", encoding="utf-8")
    monkeypatch.setenv("FOREMANKIT_CONFIG_DIR", str(config))
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(state))
    monkeypatch.setenv("KAGGLE_KEY", "k1")
    monkeypatch.setenv("DEEPSEEK_KEY", "k2")
    monkeypatch.setattr(gateway, "external_target", lambda body, cfg: ("kaggle", {**KAGGLE, "name": "kaggle"}))
    _plan.clear()
    _write_yaml(config, {"kaggle": KAGGLE, "deepseek": DEEPSEEK})
    return config


def _write_yaml(config: Path, upstreams: dict) -> None:
    """orchestrator.yaml with the given upstreams declared (JSON is valid YAML)."""
    body = {"side_fallback_upstream": "deepseek", "upstreams": upstreams}
    (config / "orchestrator.yaml").write_text(json.dumps(body), encoding="utf-8")


def send_through_gateway(path: str, data: dict):
    """One POST with a JSON body through gateway.handle, as Claude Code would; returns the gateway's response."""
    import asyncio
    body = json.dumps(data).encode("utf-8")
    scope = {"type": "http", "method": "POST", "path": path, "query_string": b"",
             "headers": [(b"content-type", b"application/json"), (b"x-api-key", b"user-key")]}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return asyncio.run(gateway.handle(Request(scope, receive)))


def urls(calls: list) -> list[str]:
    """The URL of each recorded upstream request, in order."""
    return [str(call.url) for call in calls]


def test_a_toolless_request_is_answered_by_the_cloud_and_kaggle_is_not_called(gateway_dirs: Path, sent: list):
    _plan["https://api.anthropic.com"] = FakeResponse(200, b'{"answer": "from cloud"}')
    response = send_through_gateway("/v1/messages", MESSAGE)
    assert response.status_code == 200 and response.body == b'{"answer": "from cloud"}'
    assert len(sent) == 1 and urls(sent)[0].startswith("https://api.anthropic.com/v1/messages")
    assert json.loads(sent[0].content)["model"] == "claude-sonnet-5-5"


def test_a_cloud_connection_error_falls_back_to_the_deepseek_upstream(gateway_dirs: Path, sent: list):
    _plan["https://api.anthropic.com"] = httpx.ConnectError("down")
    send_through_gateway("/v1/messages", MESSAGE)
    assert urls(sent)[-1].startswith("https://deepseek.example/v1/messages")


def test_a_cloud_rate_limit_falls_back_to_the_deepseek_upstream(gateway_dirs: Path, sent: list):
    _plan["https://api.anthropic.com"] = FakeResponse(429, b"{}")
    send_through_gateway("/v1/messages", MESSAGE)
    assert urls(sent)[0].startswith("https://api.anthropic.com/v1/messages")
    assert urls(sent)[-1].startswith("https://deepseek.example/v1/messages")


def test_an_undeclared_fallback_keeps_the_current_upstream(gateway_dirs: Path, sent: list):
    _write_yaml(gateway_dirs, {"kaggle": KAGGLE})
    _plan["https://api.anthropic.com"] = httpx.ConnectError("down")
    send_through_gateway("/v1/messages", MESSAGE)
    assert urls(sent)[-1].startswith("https://kaggle.example/v1/messages")


def test_a_fallback_without_its_key_keeps_the_current_upstream(gateway_dirs: Path, sent: list, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_KEY")
    _plan["https://api.anthropic.com"] = httpx.ConnectError("down")
    send_through_gateway("/v1/messages", MESSAGE)
    assert urls(sent)[-1].startswith("https://kaggle.example/v1/messages")


def test_a_worker_role_toolless_request_goes_to_the_cloud_not_503(gateway_dirs: Path, sent: list, monkeypatch):
    monkeypatch.setattr(gateway, "external_target", lambda body, cfg: None)
    (gateway_dirs.parent / "state" / gateway.LOCAL_ROLE_FILE).write_text("worker", encoding="utf-8")
    response = send_through_gateway("/v1/messages", MESSAGE)
    assert response.status_code == 200
    assert urls(sent)[0].startswith("https://api.anthropic.com/v1/messages")


def test_a_request_with_tools_in_external_mode_goes_to_kaggle_unchanged(gateway_dirs: Path, sent: list):
    _plan["https://api.anthropic.com"] = FakeResponse(200)
    send_through_gateway("/v1/messages", {**MESSAGE, "tools": [{"name": "Read"}]})
    assert len(sent) == 1 and urls(sent)[0].startswith("https://kaggle.example/v1/messages")
    assert json.loads(sent[0].content)["model"] == "main"
