from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from foremankit.config import load_config
from foremankit.router import Router, RouterEnv
from foremankit.state import CallLog, StateStore

REPO = Path(__file__).resolve().parent.parent
ENV_VARS = ("DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY",
            "CEREBRAS_API_KEY", "OLLAMA_LOCAL_URL", "LLAMACPP_URL", "LLM_ENABLED", "LLM_SELF_ENABLED", "LLM_OTHERS_ENABLED",
            "FOREMANKIT_ALLOW_INTERNAL", "FOREMANKIT_ALLOW_INTERNAL_TRAINING", "FOREMANKIT_DRY_RUN", "FOREMANKIT_OUT_DIR", "FOREMANKIT_CONFIG_DIR", "FOREMANKIT_SELF",
            "FOREMANKIT_PEER_KEY", "LOCAL_LLM_GEN_TPS", "LOCAL_LLM_PROMPT_TPS")

LAN = "http://192.168.1.50:11434"
TUNNEL = "http://100.101.102.103:11434"
LOCAL = "http://localhost:11434"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FOREMANKIT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("FOREMANKIT_PEER_RESOLVE", "off")   # no real DNS in tests (peers tests that need it patch socket)
    monkeypatch.setenv("FOREMANKIT_NODES", "off")   # tests must not depend on the deployed config/nodes.yaml (node tests set cfg["nodes"])
    monkeypatch.setenv("FOREMANKIT_SELF", "test-machine")   # never match the hostname of a real lab machine
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def cfg() -> dict[str, Any]:
    config = copy.deepcopy(load_config(REPO / "config"))
    # Tests must not depend on local deployment choices (deepseek is disabled on the work laptop).
    config["providers"]["deepseek"]["enabled"] = True
    return config


def chat_json(text: str = "result text", prompt: int = 100, completion: int = 50, **usage: int) -> dict:
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, **usage}}


class Mock:
    """Route requests by (scheme://host:port, path) to handlers; record every call."""

    def __init__(self):
        self.routes: dict[tuple[str, str], Callable[[httpx.Request], httpx.Response] | list] = {}
        self.calls: list[str] = []
        self.bodies: list[dict] = []

    def on(self, base: str, path: str, *responses: Any) -> None:
        self.routes[(base.rstrip("/"), path)] = list(responses)

    def hosts(self) -> set[str]:
        return {c.split("/")[2] for c in self.calls}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}" + (f":{request.url.port}" if request.url.port else "")
        self.calls.append(f"{base}{request.url.path}")
        if request.content:
            self.bodies.append(json.loads(request.content))
        for (b, path), queue in self.routes.items():
            if b == base and request.url.path.endswith(path):
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if callable(item):  # lets a test block / count / vary the answer
                    item = item(request)
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, int):
                    return httpx.Response(item, json={"error": "x"})
                return httpx.Response(200, json=item)
        raise httpx.ConnectError("no route", request=request)


def tags(*names: str) -> dict:
    return {"models": [{"name": n} for n in names]}


@pytest.fixture
def mock() -> Mock:
    return Mock()


@pytest.fixture
def make_router(cfg, mock, tmp_path):
    def _make(config: dict | None = None, resolver=None, clock=None) -> Router:
        env = RouterEnv(store=StateStore(tmp_path / "state" / "state.json"),
                        log=CallLog(tmp_path / "state" / "calls.jsonl"),
                        transport=httpx.MockTransport(mock), resolver=resolver or (lambda h: ["8.8.8.8"]),
                        sleep=lambda s: None, out_dir=tmp_path / ".agent-out")
        if clock:
            env.clock = clock
        return Router(config or cfg, env)
    return _make


def read_log(tmp_path: Path) -> list[dict]:
    path = tmp_path / "state" / "calls.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
