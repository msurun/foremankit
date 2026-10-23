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
from foremankit import upstream
from foremankit.live import LiveMeter
from foremankit.gateway_trim import (is_main_loop, request_chars, shape_request, system_text, thinking_wanted,
                                     with_thinking)

CONFIG_NAME = "orchestrator.yaml"
DROP_REQUEST = {"host", "content-length", "connection", "keep-alive", "te", "trailer",
                "transfer-encoding", "upgrade", "proxy-authorization", "proxy-authenticate"}
DROP_RESPONSE = {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
CREDENTIALS = {"x-api-key", "authorization"}
DEFAULTS = {"local_models": ["claude-sonnet-5"], "local_url": "http://127.0.0.1:8080", "local_target": "local-llm",
            "cloud_url": "https://api.anthropic.com", "port": 8081, "prompt_file": "orchestrator.md", "language": "English",
            "main_loop_tools": ["Agent", "Task"], "trim": {},
            "side_requests_to_cloud": True, "side_request_model": "claude-sonnet-5-5",
            "thinking_by_effort": {"low": False, "medium": False, "high": True, "xhigh": True, "max": True}}
DEFAULT_ROLE = "worker"      # a fresh machine only takes delegated tasks; Claude Code goes to the local model after an explicit choice
ORCHESTRATOR_FILE = "orchestrator_target.json"   # state dir: {"name", "url"} of the machine that serves Claude Code; absent = this PC
LIVE_FILE = "gateway_live.json"                 # state dir: what the local model is doing right now (foremankit/live.py)
LOCAL_ROLE_FILE = "local_role"          # file in the state dir: orchestrator | worker | both (default worker)
DUMP_ENV = "FOREMANKIT_GATEWAY_DUMP"   # path: the last shaped main-loop request is written there as JSON (for tuning)

client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=1800, write=60, pool=None))


def orchestrator_target() -> dict[str, str] | None:
    """The remote machine chosen in the panel to serve Claude Code ({"name", "url"}), or None for this PC."""
    try:
        target = json.loads((state_dir() / ORCHESTRATOR_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return target if isinstance(target, dict) and str(target.get("url", "")).startswith(("http://", "https://")) else None


def settings() -> dict[str, Any]:
    """orchestrator.yaml over the defaults; read on every request so edits need no restart.

    A remote orchestrator chosen in the panel replaces `local_url`: requests for the local models then go to that
    machine's llama-server (with the lab key, see `peer_key`)."""
    merged = {**DEFAULTS, **_read_yaml(config_dir() / CONFIG_NAME)}
    target = orchestrator_target()
    return {**merged, "local_url": target["url"]} if target else merged


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
    route: str = ""      # set for an external orchestrator (Gemini, DeepSeek...); empty = local or cloud by `local`

    def log(self, status: int) -> None:
        """Print one line: path, model, route, status, time and the optional size note."""
        route = self.route or ("local" if self.local else "cloud")
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self.path} model={self.model} route={route} "
              f"status={status} {time.time() - self.started:.2f}s{self.note}", flush=True)


WINDOW_CACHE_SECONDS = 30
_window_cache: dict[str, Any] = {"at": 0.0, "value": None, "url": ""}


def server_window(cfg: dict[str, Any]) -> int | None:
    """Context size the running llama-server really has (/props n_ctx), cached; None when it cannot be read."""
    if _window_cache["url"] == cfg["local_url"] and time.time() - _window_cache["at"] < WINDOW_CACHE_SECONDS:
        return _window_cache["value"]
    try:
        headers = {"authorization": f"Bearer {peer_key()}"} if peer_key() else {}
        props = httpx.get(cfg["local_url"].rstrip("/") + "/props", headers=headers, timeout=1.0).json()
        value = int(props["default_generation_settings"]["n_ctx"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        value = None
    _window_cache.update(at=time.time(), value=value, url=cfg["local_url"])
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


def describe_side_request(data: dict[str, Any]) -> str:
    """One short line about a request that is not the main loop (tools, streaming, size, first words of the system
    prompt), so the log shows what kind of side request (subagent, auto-mode check, title) reaches the local model."""
    system = " ".join(system_text(data.get("system")).split())[:50]
    return (f"tools={len(data.get('tools') or [])} stream={bool(data.get('stream'))} "
            f"~{request_chars(data) // 4000}k tokens system=\"{system}\"")


def shape_local(data: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Rewrite a local request; returns it with a log note on how much the prompt shrank (main loop only)."""
    data = {**data, "model": cfg["local_target"]}
    wanted = thinking_wanted(data, cfg["thinking_by_effort"])
    if wanted is not None:
        data = with_thinking(data, wanted)
    thinking_note = "" if wanted is None else f" thinking={'on' if wanted else 'off'}"
    if not is_main_loop(data, cfg["main_loop_tools"]):
        return data, f" (subagent: untouched; {describe_side_request(data)})" + thinking_note
    before = request_chars(data)
    data = shape_request(data, read_prompt(cfg), cfg["trim"])
    after = request_chars(data)
    record_usage(after // 4, cfg)
    if os.getenv(DUMP_ENV):
        Path(os.environ[DUMP_ENV]).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data, f" (main loop: ~{before // 4000}k -> ~{after // 4000}k tokens)" + thinking_note


def write_live(state: dict[str, Any]) -> None:
    """Publish the live state for the panel; a missing hint must never fail the request."""
    try:
        (state_dir() / LIVE_FILE).write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


async def metered(chunks: Any) -> Any:
    """Pass the answer stream through unchanged while a LiveMeter watches it."""
    meter = LiveMeter(write_live)
    meter.begin()
    try:
        async for chunk in chunks:
            meter.feed(chunk)
            yield chunk
    finally:
        meter.end()


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


def is_toolless_message(path: str, data: dict[str, Any] | None) -> bool:
    """True for a /v1/messages request without tools: the auto-mode safety check, titles and similar side calls.

    Claude Code expects a real Claude model to answer them in a strict format; the local model does not, and the
    session fails with a bare 'Error executing tool'. Subagents with tools stay local."""
    return path == "/v1/messages" and data is not None and not data.get("tools")


def send_to_cloud(call: Call, data: dict[str, Any], cfg: dict[str, Any]) -> bytes:
    """Re-aim a local call at the cloud with a real Claude model; returns the new request body."""
    call.local = False
    call.note = f" (side request without tools -> cloud as {cfg['side_request_model']})"
    return json.dumps({**data, "model": cfg["side_request_model"]}, ensure_ascii=False).encode("utf-8")


def local_role() -> str:
    """What the local llama-server is for: orchestrator (Claude Code), worker (delegated tasks) or both."""
    try:
        role = (state_dir() / LOCAL_ROLE_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_ROLE
    return role if role in ("orchestrator", "worker", "both") else DEFAULT_ROLE


def refusal() -> str | None:
    """Why a request to the local model must be refused right now, or None when it may go through."""
    if orchestrator_target() is None and local_role() == "worker":
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


def prepare_body(call: Call, data: dict[str, Any] | None, body: bytes, cfg: dict[str, Any]) -> bytes:
    """Body for the upstream call: side requests go to the cloud, local main-loop requests are shaped, rest unchanged."""
    if call.local and cfg["side_requests_to_cloud"] and is_toolless_message(call.path, data):
        return send_to_cloud(call, data, cfg)
    if call.local and data is not None:
        shaped, call.note = shape_local(data, cfg)
        return json.dumps(shaped, ensure_ascii=False).encode("utf-8")
    return body


async def relay(request: Request, call: Call, cfg: dict[str, Any], body: bytes,
                route: tuple[str, dict[str, str]] | None = None) -> Response:
    """Send the prepared request upstream and stream the answer back to the plugin.

    `route` = (url, headers) overrides the local/cloud choice (used for an external orchestrator)."""
    url, headers = route or (target_url(request, cfg, call.local), upstream_headers(request, call.local))
    try:
        response = await client.send(client.build_request(request.method, url, headers=headers, content=body or None),
                                     stream=True)
    except httpx.HTTPError as exc:
        call.log(502)
        return failure(call.local, exc)
    call.log(response.status_code)
    out_headers = {name: value for name, value in response.headers.items() if name.lower() not in DROP_RESPONSE}
    streamed = call.local and "text/event-stream" in response.headers.get("content-type", "")
    body_stream = metered(response.aiter_raw()) if streamed else response.aiter_raw()
    return StreamingResponse(body_stream, status_code=response.status_code, headers=out_headers,
                             background=BackgroundTask(response.aclose))


def external_mode(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """(name, upstream config) when Claude Code is switched to an external orchestrator, else None.

    A mode that is neither built in nor declared in `upstreams` counts as external with an empty config, so that
    the request is refused instead of silently going to the Anthropic cloud."""
    mode = upstream.active_mode()
    if mode in upstream.BUILTIN_MODES:
        return None
    declared = cfg.get("upstreams") or {}
    return mode, declared.get(mode) or {}


def external_refusal(mode: str, up: dict[str, Any]) -> str | None:
    """Why a request for the external orchestrator `mode` must be refused (undeclared mode, missing key), or None."""
    if not up.get("url") or not up.get("models"):
        return f"orchestrator mode '{mode}' is not declared in orchestrator.yaml (upstreams); run `foremankit orchestrator use claude`"
    if not os.getenv(str(up.get("key_env", ""))):
        return f"orchestrator mode '{mode}': environment variable {up.get('key_env')} is not set"
    return None


def shape_external(data: dict[str, Any], up: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Rewrite a request for an external orchestrator: model by alias (tool-less side requests use the cheap alias)
    and, for the main loop, the extra rules prompt; returns it with a log note."""
    side = is_toolless_message("/v1/messages", data)
    data = {**data, "model": upstream.upstream_model(up, "haiku" if side else str(data.get("model", "")))}
    if side or not is_main_loop(data, cfg["main_loop_tools"]):
        return data, f" (-> {data['model']}{' side request' if side else ''})"
    prompt_cfg = {**cfg, "prompt_file": up.get("prompt_file", "orchestrator.external.md")}
    trim = {"enabled": False, "system": "append", **(up.get("trim") or {})}
    return shape_request(data, read_prompt(prompt_cfg), trim), f" (-> {data['model']}, main loop)"


def external_headers(request: Request, up: dict[str, Any]) -> dict[str, str]:
    """Headers for an external orchestrator: the client's credentials are replaced by the provider key, and the
    headers listed in `drop_headers` (default anthropic-beta, which third-party endpoints reject) are removed."""
    drop = {name.lower() for name in up.get("drop_headers", ["anthropic-beta"])} | CREDENTIALS
    headers = {name: value for name, value in upstream_headers(request, False).items() if name.lower() not in drop}
    key = os.environ[up["key_env"]]
    return {**headers, "x-api-key": key, "authorization": f"Bearer {key}"}


async def handle_external(request: Request, body: bytes, cfg: dict[str, Any], mode: str, up: dict[str, Any]) -> Response:
    """Forward one request to the external orchestrator `mode` (an Anthropic-compatible endpoint)."""
    data = parse_body(body)
    call = Call(request.url.path, data.get("model") if data else None, False, time.time(), route=f"external:{mode}")
    reason = external_refusal(mode, up)
    if reason:
        call.log(503)
        return switched_off(reason)
    if data is not None and request.url.path == "/v1/messages":
        shaped, call.note = shape_external(data, up, cfg)
        body = json.dumps(shaped, ensure_ascii=False).encode("utf-8")
    url = up["url"].rstrip("/") + request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return await relay(request, call, cfg, body, (url, external_headers(request, up)))


async def handle(request: Request) -> Response:
    """Forward one request to the local llama-server or the cloud and stream the answer back."""
    body, cfg = await request.body(), settings()
    external = external_mode(cfg)
    if external:
        return await handle_external(request, body, cfg, *external)
    data = parse_body(body)
    model = data.get("model") if data else None
    call = Call(request.url.path, model, model in cfg["local_models"], time.time())
    reason = refusal() if call.local else None
    if reason:
        call.log(503)
        return switched_off(reason)
    body = prepare_body(call, data, body, cfg)
    return await relay(request, call, cfg, body)


app = Starlette(routes=[Route("/{path:path}", handle,
                              methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])])


def main() -> None:
    """Console entry point: serve on 127.0.0.1 at the configured port."""
    load_env()
    uvicorn.run(app, host="127.0.0.1", port=int(settings()["port"]), log_level="warning")


if __name__ == "__main__":
    main()
