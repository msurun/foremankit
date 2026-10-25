"""Generic OpenAI-compatible client plus local-LLM server probing (llama.cpp, Ollama, ...)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from foremankit import pricing

TOGGLE_VALUES = ("auto", "on", "off")
DEFAULT_TIMEOUT_S = 120              # seconds; a slow local LLM may need a minute or more for one answer


class ProviderError(RuntimeError):
    """A call to a provider failed. Carries what the retry and cooldown logic needs."""
    timed_out = False   # True for ProviderTimeout: a hung server, retrying would only double the wait

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None,
                 transient: bool = False):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.transient = transient


class ProviderTimeout(ProviderError):
    """The provider did not answer within the timeout."""
    timed_out = True


def internal_allowlist() -> set[str]:
    """Per-machine opt-in (env FOREMANKIT_ALLOW_INTERNAL=deepseek,groq): providers allowed to receive
    sensitivity=internal data on THIS machine, in addition to those with allow_internal in the config.
    A provider with trains_on_data=true is still refused."""
    raw = os.environ.get("FOREMANKIT_ALLOW_INTERNAL", "")
    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def internal_training_ok() -> set[str]:
    """Per-machine, deliberate override (env FOREMANKIT_ALLOW_INTERNAL_TRAINING=deepseek): these providers may receive
    sensitivity=internal data EVEN THOUGH they may train on prompts (trains_on_data stays true in the config).
    Never applies to sensitivity=private."""
    raw = os.environ.get("FOREMANKIT_ALLOW_INTERNAL_TRAINING", "")
    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def normalize_toggle(value: Any) -> str:
    """YAML may turn on/off into booleans; accept both."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    value = str(value or "auto").strip().lower()
    return value if value in TOGGLE_VALUES else "auto"


@dataclass
class Provider:
    """One provider section of providers.yaml (a cloud API, or the pool `llm` of lab nodes)."""
    name: str
    cfg: dict[str, Any]

    @property
    def is_llm(self) -> bool:
        """The pool of local-LLM servers (provider type "llm"; the nodes come from config/nodes.yaml)."""
        return self.cfg.get("type") == "llm"

    @property
    def free(self) -> bool:
        """Whether the provider costs nothing (free tier)."""
        return bool(self.cfg.get("free", False))

    @property
    def trains_on_data(self) -> bool:
        """Whether the provider may train on what it receives."""
        return bool(self.cfg.get("trains_on_data", False))

    @property
    def limits(self) -> dict[str, Any]:
        """Rate limits of the provider (rpm, rpd, tpd)."""
        return self.cfg.get("limits") or {}

    @property
    def timeout(self) -> float:
        """Seconds to wait for an answer."""
        return float(self.cfg.get("timeout", DEFAULT_TIMEOUT_S))

    @property
    def private_eligible(self) -> bool:
        """Only local-LLM nodes may ever receive private data."""
        return self.is_llm and bool(self.cfg.get("allow_private", True)) and not self.trains_on_data

    def eligible(self, sensitivity: str) -> bool:
        """normal: any. internal: a local-LLM node, or a cloud provider that does not train on data AND is
        explicitly marked allow_internal. private: local-LLM nodes only (fail closed)."""
        if sensitivity == "private":
            return self.private_eligible
        if sensitivity == "internal":
            if self.private_eligible:
                return True
            if self.is_llm:
                return False
            if self.name in internal_training_ok():  # deliberate per-machine acceptance that it may train on prompts
                return True
            allowed = bool(self.cfg.get("allow_internal", False)) or self.name in internal_allowlist()
            return allowed and not self.trains_on_data
        return True

    def api_key(self) -> str | None:
        """The API key from the environment variable named in the config (None when unset)."""
        env = self.cfg.get("key_env")
        return os.environ.get(env) if env else None

    def model(self, alias: str | None) -> tuple[str, str, dict[str, Any]]:
        """(alias, model_id, price) for a cloud provider."""
        models = self.cfg.get("models") or {}
        alias = alias or self.cfg.get("default_model") or next(iter(models), None)
        if alias not in models:
            raise KeyError(f"{self.name}: unknown model alias {alias!r}")
        entry = models[alias]
        return alias, entry["id"], pricing.effective_price(entry, self.cfg, time.time())


# ---------------------------------------------------------------- HTTP helpers

def _retry_after(response: httpx.Response) -> float | None:
    """Seconds from the Retry-After header, if the server sent a number."""
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Token counts of any OpenAI-compatible server as {prompt_tokens, completion_tokens, cache_hit, reasoning_tokens}."""
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    hit = usage.get("prompt_cache_hit_tokens")  # DeepSeek
    if hit is None:
        hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")  # OpenAI-style
    hit = int(hit or 0)
    reasoning = _reasoning_tokens(usage, prompt, completion)
    # Thinking models (Gemini 3.x) bill hidden reasoning tokens as output but leave them out of
    # completion_tokens; total_tokens includes them, so derive the billable output from it.
    total = int(usage.get("total_tokens") or 0)
    if total:
        completion = max(completion, total - prompt)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cache_hit": min(hit, prompt),
            "reasoning_tokens": reasoning}


def _reasoning_tokens(usage: dict[str, Any], prompt: int, completion: int) -> int:
    """Hidden reasoning tokens: the server's own count when it sends one (OpenAI-style, DeepSeek), else the
    gap between total_tokens and the reported output (Gemini hides them there), else 0."""
    details = usage.get("completion_tokens_details") or {}
    if details.get("reasoning_tokens") is not None:
        return int(details["reasoning_tokens"])
    total = int(usage.get("total_tokens") or 0)
    return max(0, total - prompt - completion) if total else 0


@dataclass(frozen=True)
class ChatRequest:
    """One /chat/completions request."""
    base_url: str
    api_key: str | None
    model_id: str
    messages: list[dict[str, str]]
    max_tokens: int
    timeout: float
    extra: dict[str, Any] | None = None   # extra request fields, e.g. reasoning_effort


CONNECT_TIMEOUT_CAP_S = 10.0       # a blocked or unreachable host must not stall the chain for minutes
ERROR_BODY_CHARS = 200             # the body may echo the request: keep only a short, key-free snippet


def chat_completion(client: httpx.Client, request: ChatRequest) -> dict[str, Any]:
    """POST /chat/completions. Returns {text, usage, finish_reason, timings}. Raises ProviderError."""
    response = _post_chat(client, request)
    _raise_for_status(response)
    return _parse_chat(response)


def _post_chat(client: httpx.Client, request: ChatRequest) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    if request.api_key:
        headers["Authorization"] = f"Bearer {request.api_key}"
    payload = {"model": request.model_id, "messages": request.messages, "max_tokens": request.max_tokens,
               "stream": False, **(request.extra or {})}
    timeout = httpx.Timeout(request.timeout, connect=min(CONNECT_TIMEOUT_CAP_S, request.timeout))
    try:
        return client.post(request.base_url.rstrip("/") + "/chat/completions", json=payload, headers=headers,
                           timeout=timeout)
    except httpx.TimeoutException as error:
        raise ProviderTimeout(f"timeout after {request.timeout:.0f}s", transient=True) from error
    except httpx.HTTPError as error:
        raise ProviderError(f"connection error: {error.__class__.__name__}", transient=True) from error
    except (TypeError, ValueError) as error:
        raise ProviderError(f"bad request payload for {request.base_url}: {error}") from error


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status == 429:
        raise ProviderError("429 rate limited", status=429, retry_after=_retry_after(response))
    if status >= 500:
        raise ProviderError(f"{status} server error", status=status, retry_after=_retry_after(response),
                            transient=True)
    if status >= 400:
        raise ProviderError(f"{status} {response.text[:ERROR_BODY_CHARS]}", status=status)


def _parse_chat(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ProviderError("malformed response") from error
    return {"text": text, "usage": normalize_usage(data.get("usage")), "finish_reason": choice.get("finish_reason"),
            "timings": data.get("timings")}      # llama-server reports its own speed here; other servers do not


def list_models(client: httpx.Client, base_url: str, api_key: str | None, timeout: float = 15) -> list[str]:
    """Sorted model ids a provider offers (GET /models)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = client.get(base_url.rstrip("/") + "/models", headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
    ids = []
    for item in items:
        if isinstance(item, dict):
            ids.append(str(item.get("id") or item.get("name")))
        else:
            ids.append(str(item))
    return sorted(ids)


# ---------------------------------------------------------------- local LLM servers

@dataclass
class Endpoint:
    """One address of a local-LLM server."""
    name: str
    url: str
    api_key: str | None = None            # the lab key for a server on another node (config/nodes.yaml)

    def headers(self) -> dict[str, str]:
        """Authorization header with the lab key, when this server needs one."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


@dataclass
class NodeProbe:
    """What probing a node found: the endpoint that answered, its models, the latency, the errors of the others."""
    endpoint: Endpoint | None
    models: list[str] = field(default_factory=list)
    latency_s: float | None = None
    errors: list[str] = field(default_factory=list)


# (path, key of the model list in the answer, fields that hold a model's name): Ollama first, then OpenAI-style
PROBE_PATHS = (("/api/tags", "models", ("name", "model")), ("/v1/models", "data", ("id",)))
WRONG_KEY_STATUSES = (401, 403)


def probe_endpoint(client: httpx.Client, endpoint: Endpoint, timeout: float) -> tuple[list[str] | None, float, str]:
    """Generic local-LLM probe: Ollama (/api/tags), else any OpenAI-compatible server (/v1/models:
    llama.cpp, LM Studio, vLLM). A host that is down is not tried twice."""
    start = time.perf_counter()
    error = ""
    for probe_path in PROBE_PATHS:
        names, error, stop = _try_probe_path(client, endpoint, probe_path, timeout)
        if stop:
            return names, time.perf_counter() - start, error
    return None, time.perf_counter() - start, error


def _try_probe_path(client: httpx.Client, endpoint: Endpoint, probe_path: tuple[str, str, tuple[str, ...]],
                    timeout: float) -> tuple[list[str] | None, str, bool]:
    """(model names or None, error text, stop). stop = answered, wrong key or host down: no other path is tried."""
    path, list_key, name_fields = probe_path
    try:
        response = client.get(endpoint.url + path, timeout=timeout, headers=endpoint.headers())
        if response.status_code in WRONG_KEY_STATUSES:
            return None, f"HTTP {response.status_code}: wrong or missing API key, or a proxy refused the request", True
        if response.status_code == 200:
            return _model_names(response.json().get(list_key, []), name_fields), "", True
        return None, f"HTTP {response.status_code}", False   # server is up but lacks this API: try the next one
    except (httpx.HTTPError, ValueError) as error:
        return None, error.__class__.__name__, isinstance(error, httpx.TransportError)   # refused/timeout = host down


def _model_names(items: list[Any], name_fields: tuple[str, ...]) -> list[str]:
    names = [next((item.get(field_name) for field_name in name_fields if item.get(field_name)), None)
             for item in items if isinstance(item, dict)]
    return [name for name in names if name]


def endpoint_busy(client: httpx.Client, endpoint: Endpoint, timeout: float = 2.0) -> bool | None:
    """True if the llama-server behind `endpoint` is processing a request right now (GET /slots, enabled by default),
    False if idle, None if it cannot tell (Ollama, /slots disabled, network error)."""
    try:
        resp = client.get(endpoint.url + "/slots", timeout=timeout, headers=endpoint.headers())
        if resp.status_code != 200:
            return None
        slots = resp.json()
        return any(bool(slot.get("is_processing")) for slot in slots if isinstance(slot, dict))
    except (httpx.HTTPError, ValueError):
        return None


def pick_model(preferred: list[str] | None, exclude: list[str] | None, available: list[str]) -> str | None:
    """First preferred model present on the server, else any chat-capable model it has."""
    for wanted in preferred or []:
        if wanted in available:
            return wanted
        # allow "qwen3" to match "qwen3:latest"
        if ":" not in wanted and f"{wanted}:latest" in available:
            return f"{wanted}:latest"
    skip = [word.lower() for word in (exclude if exclude is not None else ["embed", "-base"])]
    usable = [model for model in available if not any(word in model.lower() for word in skip)]
    return usable[0] if usable else None
