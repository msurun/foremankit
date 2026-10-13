"""Gateway between the Claude Code plugin and the models (`foremankit-gateway`, default port 8081).

The plugin is pointed at it with ANTHROPIC_BASE_URL. Requests whose "model" is in `local_models` (config/orchestrator.yaml)
go to the local llama-server with the model name rewritten and the credentials stripped; for the plugin's main loop the
prompt is first slimmed down (see gateway_trim). Everything else is forwarded unchanged to the Anthropic cloud, so the
plugin keeps working with the user's normal login. While the gateway is down the plugin cannot reach the cloud either.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from foremankit.config import _read_yaml, config_dir, load_env, state_dir
from foremankit.gateway_trim import is_main_loop, request_chars, shape_request, thinking_wanted, with_thinking

CONFIG_NAME = "orchestrator.yaml"
DROP_REQUEST = {"host", "content-length", "connection", "keep-alive", "te", "trailer",
                "transfer-encoding", "upgrade", "proxy-authorization", "proxy-authenticate"}
DROP_RESPONSE = {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
CREDENTIALS = {"x-api-key", "authorization"}
DEFAULTS = {"local_models": ["claude-sonnet-5"], "local_url": "http://127.0.0.1:8080", "local_target": "local-llm",
            "cloud_url": "https://api.anthropic.com", "port": 8081, "prompt_file": "orchestrator.md", "language": "English",
            "main_loop_tools": ["Agent", "Task"], "trim": {},
            "thinking_by_effort": {"low": False, "medium": False, "high": True, "xhigh": True, "max": True}}
LOCAL_ROLE_FILE = "local_role"          # file in the state dir: orchestrator | worker | both (default both)
DUMP_ENV = "FOREMANKIT_GATEWAY_DUMP"   # path: the last shaped main-loop request is written there as JSON (for tuning)

client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=1800, write=60, pool=None))


def settings() -> dict[str, Any]:
    """orchestrator.yaml over the defaults; read on every request so edits need no restart."""
    return {**DEFAULTS, **_read_yaml(config_dir() / CONFIG_NAME)}


def read_prompt(cfg: dict[str, Any]) -> str:
    """Text of the orchestrator prompt file with {language} filled in, or '' when it is missing."""
    try:
        text = (config_dir() / cfg["prompt_file"]).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text.replace("{language}", str(cfg["language"]))


@dataclass
class Call:
    """What is known about one proxied request, for its log line."""
    path: str
    model: str | None
    local: bool
    started: float
    note: str = ""

    def log(self, status: int) -> None:
        """Print one line: path, model, route, status, time and the optional size note."""
        route = "local" if self.local else "cloud"
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self.path} model={self.model} route={route} "
              f"status={status} {time.time() - self.started:.2f}s{self.note}", flush=True)


WINDOW_CACHE_SECONDS = 30
_window_cache: dict[str, Any] = {"at": 0.0, "value": None}


def server_window(cfg: dict[str, Any]) -> int | None:
    """Context size the running llama-server really has (/props n_ctx), cached; None when it cannot be read."""
    if time.time() - _window_cache["at"] < WINDOW_CACHE_SECONDS:
        return _window_cache["value"]
    try:
        props = httpx.get(cfg["local_url"].rstrip("/") + "/props", timeout=1.0).json()
        value = int(props["default_generation_settings"]["n_ctx"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        value = None
    _window_cache.update(at=time.time(), value=value)
    return value


def record_usage(tokens: int, cfg: dict[str, Any]) -> None:
    """Tell the session-length hook how full the local model's window is (hooks/session_length_reminder.py).

    The window comes from the running server, because the model can be switched (llama/start-llama.ps1 menu) while
    `context_tokens` in orchestrator.yaml stays the same; the yaml value is only the fallback."""
    window = server_window(cfg) or cfg.get("context_tokens")
    if not window:
        return
    usage = {"tokens": tokens, "window": int(window), "time": time.time()}
    try:
        (state_dir() / "gateway_context.json").write_text(json.dumps(usage), encoding="utf-8")
    except OSError:
        pass   # a missing hint must never fail the request


def shape_local(data: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Rewrite a local request; returns it with a log note on how much the prompt shrank (main loop only)."""
    data = {**data, "model": cfg["local_target"]}
    wanted = thinking_wanted(data, cfg["thinking_by_effort"])
    if wanted is not None:
        data = with_thinking(data, wanted)
    thinking_note = "" if wanted is None else f" thinking={'on' if wanted else 'off'}"
    if not is_main_loop(data, cfg["main_loop_tools"]):
        return data, " (subagent: untouched)" + thinking_note
    before = request_chars(data)
    data = shape_request(data, read_prompt(cfg), cfg["trim"])
    after = request_chars(data)
    record_usage(after // 4, cfg)
    if os.getenv(DUMP_ENV):
        Path(os.environ[DUMP_ENV]).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data, f" (main loop: ~{before // 4000}k -> ~{after // 4000}k tokens)" + thinking_note


def peer_key() -> str | None:
    """Key for a llama-server started with -Share; it is sent to the local URL only."""
    return os.getenv("FOREMANKIT_PEER_KEY") or None


def upstream_headers(request: Request, local: bool) -> dict[str, str]:
    """Request headers for the upstream call; local calls carry the peer key instead of the user's credentials."""
    headers = {name: value for name, value in request.headers.items() if name.lower() not in DROP_REQUEST}
    if local:
        headers = {name: value for name, value in headers.items() if name.lower() not in CREDENTIALS}
        if peer_key():
            headers["authorization"] = f"Bearer {peer_key()}"
    headers["accept-encoding"] = "identity"
    return headers


def parse_body(body: bytes) -> dict[str, Any] | None:
    """The JSON object in the request body, or None when it is empty or not a JSON object."""
    try:
        data = json.loads(body) if body else None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def local_role() -> str:
    """What the local llama-server is for: orchestrator (Claude Code), worker (delegated tasks) or both."""
    try:
        role = (state_dir() / LOCAL_ROLE_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return "both"
    return role if role in ("orchestrator", "worker", "both") else "both"


def refusal() -> str | None:
    """Why a request to the local model must be refused right now, or None when it may go through."""
    if local_role() == "worker":
        return "the local server works as a worker (delegated tasks only); switch it to orchestrator in the ForemanKit panel"
    return None


def switched_off(message: str) -> Response:
    """503 in the Anthropic error shape for a request to the local model that its role refuses."""
    return JSONResponse({"type": "error", "error": {"type": "api_error", "message": message}}, status_code=503)


def failure(local: bool, exc: httpx.HTTPError) -> Response:
    """502 in the Anthropic error shape, so the plugin shows a readable message."""
    what = "local llama-server unavailable" if local else f"cloud unreachable: {type(exc).__name__}"
    return JSONResponse({"type": "error", "error": {"type": "api_error", "message": what}}, status_code=502)


def target_url(request: Request, cfg: dict[str, Any], local: bool) -> str:
    """Upstream URL: the local llama-server or the cloud, with the request's path and query."""
    base = (cfg["local_url"] if local else cfg["cloud_url"]).rstrip("/")
    return base + request.url.path + (f"?{request.url.query}" if request.url.query else "")


async def handle(request: Request) -> Response:
    """Forward one request to the local llama-server or the cloud and stream the answer back."""
    body, cfg = await request.body(), settings()
    data = parse_body(body)
    model = data.get("model") if data else None
    call = Call(request.url.path, model, model in cfg["local_models"], time.time())
    reason = refusal() if call.local else None
    if reason:
        call.log(503)
        return switched_off(reason)
    if call.local and data is not None:
        data, call.note = shape_local(data, cfg)
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    try:
        upstream = await client.send(client.build_request(
            request.method, target_url(request, cfg, call.local), headers=upstream_headers(request, call.local),
            content=body or None), stream=True)
    except httpx.HTTPError as exc:
        call.log(502)
        return failure(call.local, exc)
    call.log(upstream.status_code)
    headers = {name: value for name, value in upstream.headers.items() if name.lower() not in DROP_RESPONSE}
    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=headers,
                             background=BackgroundTask(upstream.aclose))


app = Starlette(routes=[Route("/{path:path}", handle,
                              methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])])


def main() -> None:
    """Console entry point: serve on 127.0.0.1 at the configured port."""
    load_env()
    uvicorn.run(app, host="127.0.0.1", port=int(settings()["port"]), log_level="warning")


if __name__ == "__main__":
    main()
