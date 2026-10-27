"""Gateway between the Claude Code plugin and the models (`foremankit-gateway`, default port 8081).

The plugin is pointed at it with ANTHROPIC_BASE_URL. Requests whose "model" is in `local_models` (config/orchestrator.yaml)
go to the local llama-server with the model name rewritten and the credentials stripped; for the plugin's main loop the
prompt is first slimmed down (see gateway_trim). Everything else is forwarded unchanged to the Anthropic cloud, so the
plugin keeps working with the user's normal login. While the gateway is down the plugin cannot reach the cloud either.
"""
from __future__ import annotations

import json
import os
import re
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

from foremankit import (
    budget,
    context_stop,
    escalation,
    openai_bridge,
    pricing,
    provider_schema,
    providers,
    upstream,
)
from foremankit.config import _read_yaml, config_dir, load_env, state_dir
from foremankit.gateway_trim import (
    is_main_loop,
    request_chars,
    shape_request,
    system_text,
    thinking_wanted,
    trim_tools,
    with_thinking,
)
from foremankit.live import LiveMeter
from foremankit.state import CallLog, StateStore

CONFIG_NAME = "orchestrator.yaml"
DROP_REQUEST = {"host", "content-length", "connection", "keep-alive", "te", "trailer",
                "transfer-encoding", "upgrade", "proxy-authorization", "proxy-authenticate"}
DROP_RESPONSE = {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
CREDENTIALS = {"x-api-key", "authorization"}
DEFAULTS = {"local_models": ["claude-sonnet-5"], "local_url": "http://127.0.0.1:8080", "local_target": "local-llm",
            "cloud_url": "https://api.anthropic.com", "port": 8081, "prompt_file": "orchestrator.md", "language": "English",
            "main_loop_tools": ["Agent", "Task"], "trim": {},
            "side_requests_to_cloud": True, "side_request_model": "claude-sonnet-5-5",
            "overflow_to_cloud": True, "overflow_ratio": 0.70,
            "no_code": True,
            "context_stop": {},
            "thinking_by_effort": {"low": False, "medium": False, "high": True, "xhigh": True, "max": True}}
DEFAULT_ROLE = "worker"      # a fresh machine only takes delegated tasks; Claude Code goes to the local model after an explicit choice
ORCHESTRATOR_FILE = "orchestrator_target.json"   # state dir: {"name", "url"} of the machine that serves Claude Code; absent = this PC
TIER_FILE = "orchestrator_tier.json"            # state dir: {"mode", "tier", "alias", "time"} of the main loop (panel)
LIVE_FILE = "gateway_live.json"                 # state dir: what the local model is doing right now (foremankit/live.py)
LOCAL_ROLE_FILE = "local_role"          # file in the state dir: orchestrator | worker | both (default worker)
DUMP_ENV = "FOREMANKIT_GATEWAY_DUMP"   # path: the last shaped main-loop request is written there as JSON (for tuning)
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"   # answered locally for an OpenAI-speaking upstream (it has no such route)

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
    sent: bytes = b""    # the body sent to an external orchestrator, kept for the error record
    route: str = ""      # set for an external orchestrator (Gemini, DeepSeek...); empty = local or cloud by `local`
    provider_model: str = ""   # the provider's own model id (e.g. deepseek-v4-pro) the external answer was generated by
    openai: bool = False       # the upstream speaks /chat/completions (`format: openai`), so both ways need translating

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


def overflows(data: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """True when the request no longer fits the running llama-server, even after trimming.

    A local-mode session keeps its own model id, so Claude Code's auto-compact window (100k) is far above the window
    of the tuned local model (32k): without a spill the llama-server refuses the request long before auto-compact
    fires. The request is then sent to the cloud untouched, so the session survives and compacts as usual."""
    if not cfg.get("overflow_to_cloud", True):
        return False
    window = int(server_window(cfg) or cfg.get("context_tokens") or 0)
    return bool(window) and request_chars(data) // 4 > window * float(cfg.get("overflow_ratio", 0.70))


def context_window(up: dict[str, Any], cfg: dict[str, Any]) -> int:
    """Context window in tokens for an external upstream: its `context`, else the local fallback."""
    return int(up.get("context") or cfg.get("context_tokens") or 0)


def context_stop_response(call: Call, data: dict[str, Any],
                          where: dict[str, Any], cfg: dict[str, Any]) -> Response | None:
    """Answer "stopped, handoff written" when the prompt fills `ratio` of the model window, else None to run the turn.

    `where` = {"mode", "model", "window"}: who would have answered and its context window. The handoff is drafted from
    the request body alone (context_stop.py), so a degraded local model never gets another turn."""
    window = int(where.get("window") or 0)
    pol = context_stop.policy(cfg)
    stop = context_stop.stop_at(data, window, pol)
    if stop is None:
        return None
    call.note = f" (context stop: {stop['tokens']:,} / {stop['window']:,} tokens)"
    path = context_stop.write_handoff(context_stop.folder(pol, state_dir()),
                                      context_stop.handoff_markdown(data, stop, where))
    text = context_stop.reply_text(stop, path, str(cfg.get("language", "English")))
    model = str(where.get("model", ""))
    if data.get("stream"):
        return Response(context_stop.sse_bytes(text, model, stop), media_type="text/event-stream")
    return JSONResponse(context_stop.message_json(text, model, stop))


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
    if not overflows(data, cfg):      # an overflowing request leaves for the cloud, its fill is no local fill
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


WRITE_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")   # what a cloud model needs to write code itself
NO_CODE_TEXT = (
    "foremankit no-code policy: you are the orchestrator and never write or edit files yourself — your Write, Edit and "
    "NotebookEdit tools are not available. To change a file call mcp__foremankit__delegate, or Agent with model "
    "'haiku' when the work must stay in this session; do not write files through Bash either. Reading, searching, "
    "running tests and deciding stay yours.")


def add_system_text(data: dict[str, Any], text: str) -> Any:
    """The `system` field of `data` with `text` appended: after a string, as a new block after a list, or the text alone."""
    system = data.get("system")
    if isinstance(system, str):
        return system + "\n\n" + text
    if isinstance(system, list):
        return [*system, {"type": "text", "text": text}]
    return text


def orchestrator_no_code(data: dict[str, Any], cfg: dict[str, Any]) -> bytes | None:
    """The body for a request the Claude orchestrator (Sonnet/Opus) would run, with the file-writing tools removed so it
    cannot write code at all; None when the model keeps its tools (a Haiku subagent, the side-request model), when the
    orchestrator is an external provider or when the policy is off. Haiku subagents send their own `claude-haiku-*` id
    through the same gateway, so the model id is enough to tell them apart; the extra system line sends the work to
    `delegate` or to a Haiku agent."""
    if not cfg.get("no_code", True) or external_mode(cfg) is not None:
        return None
    model = str(data.get("model") or "")
    if not model.startswith("claude-") or "haiku" in model:
        return None
    tools = data.get("tools") or []
    kept = [tool for tool in tools if not isinstance(tool, dict) or tool.get("name") not in WRITE_TOOLS]
    if len(kept) == len(tools):
        return None                      # no writing tool to remove: the instruction alone would only cost tokens
    shaped = {**data, "tools": kept, "system": add_system_text(data, NO_CODE_TEXT)}
    return json.dumps(shaped, ensure_ascii=False).encode("utf-8")


def send_to_cloud(call: Call, data: dict[str, Any], cfg: dict[str, Any]) -> bytes:
    """Re-aim a local call at the cloud with a real Claude model; returns the new request body."""
    call.local = False
    call.note = f" (side request without tools -> cloud as {cfg['side_request_model']})"
    return json.dumps({**data, "model": cfg["side_request_model"]}, ensure_ascii=False).encode("utf-8")


def spill_to_cloud(call: Call, data: dict[str, Any], cfg: dict[str, Any]) -> bytes:
    """Send a local request that does not fit the llama window to the cloud; returns the body, write tools removed when
    the no-code policy applies to the model it will reach."""
    call.local = False
    call.note = f" (does not fit the local window -> cloud as {data.get('model')})"
    return orchestrator_no_code(data, cfg) or json.dumps(data, ensure_ascii=False).encode("utf-8")


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


def prepare_body(call: Call, data: dict[str, Any] | None, body: bytes, cfg: dict[str, Any]) -> bytes | Response:
    """Body for the upstream call, or a synthetic stop answer: cloud requests lose the write tools when the no-code
    policy applies, side requests go to the cloud, local requests shaped, a full window answered with a handoff instead
    of another turn, an oversized one spilled there."""
    if data is None:
        return body
    if not call.local:
        return orchestrator_no_code(data, cfg) or body
    if cfg["side_requests_to_cloud"] and is_toolless_message(call.path, data):
        return send_to_cloud(call, data, cfg)
    shaped, call.note = shape_local(data, cfg)
    window = int(server_window(cfg) or cfg.get("context_tokens") or 0)
    stopped = context_stop_response(call, shaped,
                                    {"mode": "local", "model": shaped.get("model", ""), "window": window}, cfg)
    if stopped is not None:
        return stopped
    if overflows(shaped, cfg):
        return spill_to_cloud(call, data, cfg)
    return json.dumps(shaped, ensure_ascii=False).encode("utf-8")


async def relay(request: Request, call: Call, cfg: dict[str, Any],
                route: tuple[str, dict[str, str]] | None = None) -> Response:
    """Send the prepared request (call.sent) upstream and stream the answer back to the plugin.

    `route` = (url, headers) overrides the local/cloud choice (used for an external orchestrator)."""
    url, headers = route or (target_url(request, cfg, call.local), upstream_headers(request, call.local))
    try:
        response = await client.send(client.build_request(request.method, url, headers=headers,
                                                          content=call.sent or None), stream=True)
    except httpx.HTTPError as exc:
        call.log(502)
        return failure(call.local, exc)
    call.log(response.status_code)
    out_headers = _response_headers(response)
    if call.route and response.status_code >= 400:
        return await recorded_error(call, response, out_headers)
    return await _streamed_response(call, cfg, response, out_headers)


def _response_headers(response: httpx.Response) -> dict[str, str]:
    """Response headers with the hop-by-hop ones dropped."""
    return {name: value for name, value in response.headers.items() if name.lower() not in DROP_RESPONSE}


async def _streamed_response(call: Call, cfg: dict[str, Any], response: httpx.Response,
                             headers: dict[str, str]) -> Response:
    """The answer: translated from OpenAI when the upstream speaks it, then metered for the local model, counted for an
    external orchestrator, else passed through unchanged."""
    if call.openai:
        return await openai_response(call, cfg, response, headers)
    streamed = call.local and "text/event-stream" in response.headers.get("content-type", "")
    if call.route:
        body_stream = counted_stream(response.aiter_raw(), call, cfg)
    elif streamed:
        body_stream = metered(response.aiter_raw())
    else:
        body_stream = response.aiter_raw()
    return StreamingResponse(body_stream, status_code=response.status_code, headers=headers,
                             background=BackgroundTask(response.aclose))


async def openai_response(call: Call, cfg: dict[str, Any], response: httpx.Response,
                          headers: dict[str, str]) -> Response:
    """An OpenAI-format answer rewritten into the Anthropic Messages API: a translated SSE stream, or a JSON message."""
    if "text/event-stream" in response.headers.get("content-type", ""):
        events = openai_bridge.stream_to_anthropic(response.aiter_raw(), external_model(call))
        return StreamingResponse(counted_stream(events, call, cfg), status_code=response.status_code,
                                 headers={**headers, "content-type": "text/event-stream; charset=utf-8"},
                                 background=BackgroundTask(response.aclose))
    return await translated_message(call, cfg, response, headers)


async def translated_message(call: Call, cfg: dict[str, Any], response: httpx.Response,
                             headers: dict[str, str]) -> Response:
    """One non-streamed OpenAI answer as an Anthropic message, its usage counted like a streamed one."""
    answer = await response.aread()
    await response.aclose()
    try:
        data = json.loads(answer)
    except ValueError:
        return Response(answer, status_code=response.status_code, headers=headers)
    record_spend(call, cfg, data.get("usage") if isinstance(data, dict) else None)
    message = openai_bridge.response_to_anthropic(data, external_model(call))
    return JSONResponse(message, status_code=response.status_code, headers={**headers, "content-type": "application/json"})


def external_model(call: Call) -> str:
    """The model id the external answer came from (the provider's own id, falling back to the requested one)."""
    return call.provider_model or str(call.model or "")


ERRORS_DIR = "gateway-errors"      # state dir: what an external provider refused (request + its answer), newest 20 kept
KEEP_ERRORS = 20


async def recorded_error(call: Call, response: httpx.Response, headers: dict[str, str]) -> Response:
    """Return the provider's error unchanged, after saving request and answer so the cause can be read, not guessed."""
    answer = await response.aread()
    await response.aclose()
    try:
        folder = state_dir() / ERRORS_DIR
        folder.mkdir(exist_ok=True)
        text = answer.decode("utf-8", "replace")
        record = {"route": call.route, "path": call.path, "status": response.status_code, "answer": text,
                  "request": parse_body(call.sent)}
        (folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{response.status_code}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        for old in sorted(folder.glob("*.json"))[:-KEEP_ERRORS]:
            old.unlink(missing_ok=True)
        print(f"   provider answer: {text[:300]}", flush=True)
    except OSError:
        pass   # a missing record must never hide the error from the plugin
    return Response(answer, status_code=response.status_code, headers=headers)


def external_mode(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """(name, upstream config) when Claude Code is switched to an external orchestrator, else None.

    A mode that is neither built in nor declared in `upstreams` counts as external with an empty config, so that
    the request is refused instead of silently going to the Anthropic cloud."""
    mode = upstream.active_mode()
    if mode in upstream.BUILTIN_MODES:
        return None
    declared = cfg.get("upstreams") or {}
    return mode, {**(declared.get(mode) or {}), "name": mode}


def serves_picker_entry(body: bytes, cfg: dict[str, Any]) -> bool:
    """True when the request is for the plugin's ForemanKit picker entry (a model of `local_models`). Only that entry
    follows the orchestrator mode; a normal Claude model chosen in the picker still goes to Anthropic."""
    data = parse_body(body)
    return bool(data) and data.get("model") in cfg["local_models"]


def model_id_for(provider: str, alias: str) -> str:
    """The provider's own model id for a configured alias. providers.yaml is the single source of truth, so when a
    provider ships new names you fix them there once (run `foremankit models`) and both this request and its price
    follow; unknown provider/alias falls back to the alias itself."""
    models = (_read_yaml(config_dir() / "providers.yaml").get("providers") or {}).get(provider, {}).get("models") or {}
    return (models.get(alias) or {}).get("id") or alias


def price_for_model(model_id: str) -> dict[str, Any] | None:
    """The USD-per-1M-token price of a provider model right now (peak / off-peak, announced changes), found by id in
    providers.yaml; None when the model is unknown."""
    providers = _read_yaml(config_dir() / "providers.yaml").get("providers") or {}
    for section in providers.values():
        for entry in ((section or {}).get("models") or {}).values():
            if (entry or {}).get("id") == model_id:
                return pricing.effective_price(entry, section, time.time())
    return None


USAGE_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


class UsageCollector:
    """Token usage of one streamed Anthropic answer, read from the SSE events (message_start / message_delta)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self._tail = ""

    def feed(self, chunk: bytes) -> None:
        """Scan the whole new chunk (plus the unfinished end of the last one) for usage fields; cumulative values win.
        Only the tail kept for the next round is cut, never the part being scanned."""
        text = self._tail + chunk.decode("utf-8", "ignore")
        for field in USAGE_FIELDS:
            for value in re.findall(rf'"{field}"\s*:\s*(\d+)', text):
                self.counts[field] = max(self.counts.get(field, 0), int(value))
        self._tail = text[-256:]

    def normalized(self) -> dict[str, int]:
        """Token counts in the shape budget.cost_usd expects. In the Anthropic format `input_tokens` is only what was
        NOT served from the cache, so the prompt is input + cache writes + cache reads; the reads are the cache hits."""
        read = self.counts.get("cache_read_input_tokens", 0)
        uncached = self.counts.get("input_tokens", 0) + self.counts.get("cache_creation_input_tokens", 0)
        return {"prompt_tokens": uncached + read, "cache_hit": read,
                "completion_tokens": self.counts.get("output_tokens", 0)}


def record_orchestrator_spend(model: str, usage: dict[str, int], mode: str, ceiling: str) -> None:
    """Count an external orchestrator answer against the same budget and call log as delegated work, so the spend
    counter and the hard daily/monthly limit include Gemini/DeepSeek running Claude Code, not only the router's calls."""
    if not model:
        return
    price = price_for_model(model)
    if price is None:
        print(f"   no price for model={model} in providers.yaml: the answer is not counted in the budget", flush=True)
    usd = budget.cost_usd(price, usage)
    if usd > 0:
        StateStore().update(lambda state: budget.record(state, time.time(), usd, budget.ORCHESTRATOR))
    if usage["prompt_tokens"] or usage["completion_tokens"]:
        CallLog().write({"event": "call", "kind": "orchestrator", "sensitivity": ceiling, "provider": mode,
                         "node": None, "model": model, "escalate": False, "dry_run": False, "success": True,
                         "tokens_in": usage["prompt_tokens"], "tokens_out": usage["completion_tokens"],
                         "cache_hit": usage["cache_hit"], "reasoning_tokens": 0, "cost_usd": round(usd, 6)})


def upstream_name(call: Call) -> str:
    """Name of the external orchestrator this call went to ('' when the route is not external)."""
    return call.route.split(":", 1)[-1] if call.route.startswith("external:") else ""


def ceiling_of(cfg: dict[str, Any], mode: str) -> str:
    """The sensitivity ceiling configured for an external orchestrator in orchestrator.yaml."""
    return ((cfg.get("upstreams") or {}).get(mode) or {}).get("ceiling", "normal")


def record_spend(call: Call, cfg: dict[str, Any], usage: dict[str, Any] | None) -> None:
    """Count a non-streamed external answer against the orchestrator budget (streamed ones use counted_stream)."""
    mode = upstream_name(call)
    record_orchestrator_spend(external_model(call), providers.normalize_usage(usage), mode, ceiling_of(cfg, mode))


async def counted_stream(chunks: Any, call: Call, cfg: dict[str, Any]) -> Any:
    """Pass the answer through unchanged while the usage is read and the spend recorded (external orchestrator)."""
    collector = UsageCollector()
    try:
        async for chunk in chunks:
            collector.feed(chunk)
            yield chunk
    finally:
        mode = upstream_name(call)
        record_orchestrator_spend(call.provider_model, collector.normalized(), mode, ceiling_of(cfg, mode))


def orchestrator_budget_refusal() -> str | None:
    """Why an external orchestrator request must be refused: its own daily/monthly budget is used up (routing.yaml
    `budget: orchestrator_daily_usd`). The delegation budget is a different counter and is not touched."""
    caps = _read_yaml(config_dir() / "routing.yaml").get("budget") or {}
    reason = budget.check(StateStore().load(), caps, time.time(), 0.0, budget.ORCHESTRATOR)
    if reason:
        return f"{reason}. Switch the orchestrator to claude in the panel, or raise orchestrator_daily_usd in routing.yaml"
    return None


def external_refusal(mode: str, up: dict[str, Any]) -> str | None:
    """Why a request for the external orchestrator `mode` must be refused (undeclared mode, missing key), or None."""
    if not upstream.upstream_url(up) or not up.get("models"):
        return f"orchestrator mode '{mode}' is not declared in orchestrator.yaml (upstreams) or has no address; run `foremankit orchestrator use claude`"
    if not os.getenv(str(up.get("key_env", ""))):
        return f"orchestrator mode '{mode}': environment variable {up.get('key_env')} is not set"
    return None


_escalation: dict[str, dict[str, Any]] = {}   # mode -> escalation state (cheap/expensive), in-memory per gateway run
_escalation_control: dict[str, str] = {}      # mode -> the panel switch that in-memory state belongs to


def escalation_mode(provider: str) -> str:
    """The per-provider panel switch for the main loop: auto (hysteresis) or a pinned provider alias (flash / pro /
    lite31). A missing or broken file, or a provider without an entry, means auto, so the gateway behaves as before."""
    try:
        raw = json.loads((state_dir() / escalation.MODE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return escalation.AUTO
    return escalation.mode_of(raw.get(provider)) if isinstance(raw, dict) else escalation.AUTO


def main_loop_alias(mode: str, up: dict[str, Any], data: dict[str, Any]) -> str:
    """The provider alias for the main loop: the panel's per-provider pin when set, else cheap by default switched to
    the expensive one after repeated tool errors (hysteresis, foremankit/escalation.py). No `escalation` block = as
    configured."""
    if up.get("pin"):
        return up["pin"]
    base = upstream.upstream_model(up, "sonnet")
    control = escalation_mode(mode)
    if control != escalation.AUTO:
        return control                    # the panel pinned a provider alias (flash / pro / lite31)
    policy = up.get("escalation")
    if not policy:
        return base
    state = next_escalation_state(mode, control, policy, data)
    alias = policy.get("expensive", base) if state["tier"] == escalation.EXPENSIVE_TIER else base
    write_tier(mode, state["tier"], alias, control)
    return alias


def next_escalation_state(mode: str, control: str, policy: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """The tier state after this turn: pinned by the panel switch, else the automatic hysteresis. Moving the switch
    starts from a fresh state, so going back to auto begins cheap again."""
    previous = _escalation.get(mode, escalation.default_state())
    if _escalation_control.get(mode) != control:
        previous = escalation.default_state()
        _escalation_control[mode] = control
    forced = escalation.tier_for_mode(control)
    if forced:
        state = {**previous, "tier": forced}
    else:
        state = escalation.next_state(previous, escalation.last_tool_error(data.get("messages") or []),
                                      policy, time.time())
    _escalation[mode] = state
    if state["tier"] != previous["tier"]:
        reason = f"panel switch: {control}" if forced else f"errors={state['errors']}, clean={state['clean']}"
        print(f"   orchestrator {mode}: switched to {state['tier']} ({reason})", flush=True)
    return state


def write_tier(mode: str, tier: str, alias: str, control: str) -> None:
    """Tell the panel which tier the main loop runs on (cheap / expensive), with which model alias, and who chose it."""
    payload = {"mode": mode, "tier": tier, "alias": alias, "control": control, "time": time.time()}
    try:
        (state_dir() / TIER_FILE).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass   # a missing hint must never fail the request


def shape_external(data: dict[str, Any], up: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Rewrite a request for an external orchestrator: cheap main-loop model (escalated to the expensive one on tool
    errors), haiku for tool-less side requests; for the main loop, the extra rules prompt; returns it with a log note."""
    side = is_toolless_message("/v1/messages", data)
    main = is_main_loop(data, cfg["main_loop_tools"])
    provider = up.get("provider", up.get("name", "?"))
    if side:
        alias = upstream.upstream_model(up, "haiku")
    elif main or up.get("pin"):
        alias = main_loop_alias(up.get("name", "?"), up, data)
    else:
        alias = upstream.upstream_model(up, str(data.get("model", "")))
    data = {**data, "model": model_id_for(provider, alias)}
    data = adapt_for_provider(data, up)
    if side or not main:
        return data, f" (-> {data['model']}{' side request' if side else ''})"
    prompt_cfg = {**cfg, "prompt_file": up.get("prompt_file", "orchestrator.external.md")}
    trim = {"enabled": False, "system": "append", **(up.get("trim") or {})}
    return shape_request(data, read_prompt(prompt_cfg), trim), f" (-> {data['model']}, main loop)"


def adapt_for_provider(data: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
    """Make a request acceptable to the provider: Claude-only blocks as text, only the tools in `tools_keep` (the
    others, e.g. Artifact, are no use to an external model), then the schema profile of the upstream (`schema:`)."""
    data = plain_request(data)
    if up.get("tools_keep") and data.get("tools"):
        data = {**data, "tools": trim_tools(data["tools"], up["tools_keep"], 0)}
    return provider_schema.sanitize_for_provider(data, provider_schema.profile_for(str(up.get("name", "")), up.get("schema")))


STANDARD_BLOCKS = {"text", "image", "tool_use", "tool_result", "thinking", "redacted_thinking", "document", "tool_reference"}
TOOL_FIELDS_DROPPED = ("defer_loading",)    # Claude Code's tool-search flags: third-party APIs reject unknown fields


def plain_block(block: dict[str, Any]) -> dict[str, Any]:
    """A content block of a type only Claude knows (tool_definition from tool search...) becomes plain text, so a
    third-party Anthropic-compatible API does not answer 422 'unknown variant'; standard blocks pass through."""
    if not isinstance(block, dict) or block.get("type") in STANDARD_BLOCKS:
        return block
    return {"type": "text", "text": json.dumps(block, ensure_ascii=False)}


def plain_content(content: Any) -> Any:
    """`plain_block` over a message content list, also inside tool_result blocks; strings pass through."""
    if not isinstance(content, list):
        return content
    blocks = [plain_block(block) for block in content]
    return [{**block, "content": plain_content(block["content"])} if block.get("type") == "tool_result" and "content" in block
            else block for block in blocks]


def plain_request(data: dict[str, Any]) -> dict[str, Any]:
    """The request without the Claude-only parts: unknown content block types as text, tool-search flags removed."""
    messages = [{**message, "content": plain_content(message.get("content"))} for message in data.get("messages") or []]
    tools = [{key: value for key, value in tool.items() if key not in TOOL_FIELDS_DROPPED} for tool in data.get("tools") or []]
    shaped = {**data, "messages": messages}
    return {**shaped, "tools": tools} if tools else shaped


def external_headers(request: Request, up: dict[str, Any]) -> dict[str, str]:
    """Headers for an external orchestrator: the client's credentials are replaced by the provider key, and the
    headers listed in `drop_headers` (default anthropic-beta, which third-party endpoints reject) are removed."""
    drop = {name.lower() for name in up.get("drop_headers", ["anthropic-beta"])} | CREDENTIALS
    headers = {name: value for name, value in upstream_headers(request, False).items() if name.lower() not in drop}
    key = os.environ[up["key_env"]]
    return {**headers, "x-api-key": key, "authorization": f"Bearer {key}"}


def prepare_external(call: Call, data: dict[str, Any], up: dict[str, Any],
                     cfg: dict[str, Any]) -> bytes | Response:
    """Body for an external orchestrator, or the synthetic stop answer: shaped for its model, a full window stopped
    with a handoff (context_stop.py) instead of another turn."""
    mode = call.route[len("external:"):] if call.route.startswith("external:") else ""
    shaped, call.note = shape_external(data, up, cfg)
    call.provider_model = shaped.get("model", "")
    where = {"mode": mode, "model": shaped.get("model", ""), "window": context_window(up, cfg)}
    stopped = context_stop_response(call, shaped, where, cfg)
    if stopped is not None:
        call.log(200)
        return stopped
    if os.getenv(DUMP_ENV):
        Path(os.environ[DUMP_ENV]).write_text(json.dumps(shaped, ensure_ascii=False, indent=1), encoding="utf-8")
    return json.dumps(shaped, ensure_ascii=False).encode("utf-8")


async def handle_external(request: Request, body: bytes, cfg: dict[str, Any],
                          external: tuple[str, dict[str, Any]]) -> Response:
    """Forward one request to the external orchestrator in `external` = (mode, upstream config)."""
    mode, up = external
    openai = up.get("format") == "openai"
    data = parse_body(body)
    call = Call(request.url.path, data.get("model") if data else None, False, time.time(),
                route=f"external:{mode}", openai=openai)
    reason = external_refusal(mode, up) or orchestrator_budget_refusal()
    if reason:
        return refused_external(call, reason)
    answer = local_answer(call, request, data)
    if answer is not None:
        return answer
    if data is not None and request.url.path == "/v1/messages":
        prepared = external_body(call, data, up, cfg)
        if isinstance(prepared, Response):
            return prepared
        body = prepared
    call.sent = body
    return await relay(request, call, cfg, (external_url(request, up, openai), external_headers(request, up)))


def refused_external(call: Call, reason: str) -> Response:
    """Log a refused external request and return the 503 that tells Claude Code to switch back."""
    call.log(503)
    return switched_off(reason)


def local_answer(call: Call, request: Request, data: dict[str, Any] | None) -> Response | None:
    """The answer the gateway gives itself for a route an OpenAI endpoint does not have (token counting), else None."""
    if not (call.openai and request.url.path == COUNT_TOKENS_PATH and data is not None):
        return None
    call.log(200)
    return JSONResponse(openai_bridge.count_tokens(data))


def external_body(call: Call, data: dict[str, Any], up: dict[str, Any], cfg: dict[str, Any]) -> bytes | Response:
    """The body to send: shaped for the external orchestrator, translated into chat-completions form for `format: openai`."""
    prepared = prepare_external(call, data, up, cfg)
    if isinstance(prepared, Response) or not call.openai:
        return prepared
    try:
        translated = openai_bridge.request_to_openai(json.loads(prepared), call.provider_model)
    except ValueError:
        return prepared            # not JSON after all: send it unchanged instead of losing the turn
    return json.dumps(translated, ensure_ascii=False).encode("utf-8")


def external_url(request: Request, up: dict[str, Any], openai: bool) -> str:
    """The upstream address for this request: mapped to the chat-completions path for an OpenAI endpoint, query kept."""
    base = (upstream.upstream_url(up) or "").rstrip("/")
    path = (openai_bridge.openai_path(request.url.path) or request.url.path) if openai else request.url.path
    query = f"?{request.url.query}" if request.url.query else ""
    return f"{base}{path}{query}"


def model_route(body: bytes, cfg: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """(name, upstream config) for a request whose model is a route in `model_routes`, else None. The route pins the
    tier, so it does not depend on the panel's orchestrator mode or the escalation state."""
    data = parse_body(body)
    route = (cfg.get("model_routes") or {}).get(data.get("model")) if data else None
    if not route:
        return None
    name = str(route.get("upstream", ""))
    declared = (cfg.get("upstreams") or {}).get(name) or {}
    return name, {**declared, "name": name, "pin": str(route.get("tier", ""))}


def external_target(body: bytes, cfg: dict):
    """Return the upstream a request must go to (model route or external mode), else None."""
    routed = model_route(body, cfg)
    if routed:
        return routed
    external = external_mode(cfg)
    return external if external and serves_picker_entry(body, cfg) else None


async def handle(request: Request) -> Response:
    """Forward one request to the local llama-server or the cloud and stream the answer back."""
    body, cfg = await request.body(), settings()
    target = external_target(body, cfg)
    if target:
        return await handle_external(request, body, cfg, target)
    data = parse_body(body)
    model = data.get("model") if data else None
    call = Call(request.url.path, model, model in cfg["local_models"], time.time())
    reason = refusal() if call.local else None
    if reason:
        call.log(503)
        return switched_off(reason)
    prepared = prepare_body(call, data, body, cfg)
    if isinstance(prepared, Response):
        call.log(200)
        return prepared
    call.sent = prepared
    return await relay(request, call, cfg)


app = Starlette(routes=[Route("/{path:path}", handle,
                              methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])])


def main() -> None:
    """Console entry point: serve on 127.0.0.1 at the configured port."""
    load_env()
    uvicorn.run(app, host="127.0.0.1", port=int(settings()["port"]), log_level="warning")


if __name__ == "__main__":
    main()
