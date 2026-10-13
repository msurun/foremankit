"""Backend of the ForemanKit VS Code panel: status of router and llama-server, the model list and the action commands.

The VS Code extension (vscode-extension/) is only a thin shell: it runs `python -m foremankit.panel <what>` and shows the
JSON this module prints. Keeping the logic here makes it testable and keeps the extension free of build steps.

    status                  router up?, llama-server up (model, context size), context fill reported by the gateway
    models                  every .gguf next to llama/start-llama.ps1 with its tuned profile (speed, context, offload)
    command <action> ...    the PowerShell line the extension runs in a visible terminal (long jobs: start, tune, sweep)
    start-router | stop-router | start-server --model F | stop-server | router-direct | router-normal
    role <orchestrator|worker|both>    what the local server is for
    profile-set --model F --fields '{"ctx": 65536, "temp": 0.6}'    hand-picked launch parameters
    orchestrator --name <worker|this>    serve Claude Code from another machine of the lab
    worker-test --name N    reachability, lab key and a short generation
    worker-enable true|false --name N    let the router use a worker or not
    delegations    delegated tasks, who did them and what they cost (today, last days)
    workers | worker-add --name N --url U [--accept A --priority P] | worker-remove --name N
    thinking --levels low=false,high=true    quick actions executed here (router-direct/-normal edit the VS Code
                            settings: the window must be reloaded afterwards)
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import functools
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from foremankit.config import _read_yaml, config_dir, state_dir
from foremankit.nodes import LEVELS, WORKERS_FILE, _addresses, _url, load_nodes, load_workers, self_name
from foremankit.providers import normalize_toggle
from foremankit.state import StateStore
from foremankit.gateway import DEFAULTS as GATEWAY_DEFAULTS
from foremankit.gateway import LOCAL_ROLE_FILE, ORCHESTRATOR_FILE, local_role, orchestrator_target

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama"
SERVER_URL = os.getenv("LLAMACPP_URL", "http://127.0.0.1:8080")
PROBE_TIMEOUT_S = 1.0
CONTEXT_FRESH_S = 600          # the gateway's context counter older than this is not shown
PROFILE_FIELDS = ("ctx", "tg_tok_s", "ncmoe", "threads", "kv_cache", "reasoning", "think_budget", "temp", "manual")
# manual launch parameters: name -> (type, lowest, highest); None allowed for ncmoe (dense model or --fit decides)
MANUAL_RANGES = {"ctx": (int, 2048, 262144), "ncmoe": (int, 0, 99), "threads": (int, 1, 64),
                 "think_budget": (int, 0, 32768), "temp": (float, 0.0, 2.0)}
MANUAL_CHOICES = {"kv_cache": ("f16", "q8_0", "q4_0"), "reasoning": ("off", "on")}
SERVER_LOG = "llama-server.log"   # in the temp folder: output of the server started from the panel
RECENT_CALLS = 8              # delegated calls shown in the panel
HISTORY_DAYS = 7              # daily cost bars
ROLES = {"orchestrator": "off", "worker": "on", "both": "auto"}   # role -> router toggle `llm:self` (delegated tasks)
WORKER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")   # Claude Code's effort slider, lowest first
THINKING_LINE = re.compile(r"^thinking_by_effort:.*$", re.MULTILINE)
MOE_NAME = re.compile(r"-A\d+B", re.IGNORECASE)   # Qwen3.6-35B-A3B, gemma-4-26B-A4B: active-parameter suffix of MoE models


def probe(url: str) -> dict[str, Any] | None:
    """JSON answer of a GET, or None when nothing answers (or the answer is not JSON)."""
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT_S) as answer:
            return json.loads(answer.read())
    except (OSError, ValueError):
        return None


def port_open(port: int) -> bool:
    """True when something accepts a TCP connection on localhost:port (any HTTP answer counts, even a 404)."""
    import socket
    with socket.socket() as sock:
        sock.settimeout(PROBE_TIMEOUT_S)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def server_process_running() -> bool:
    """True when a llama-server process exists. Without an answer on /props that means it is still loading the model."""
    if os.name != "nt":
        return subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, check=False).returncode == 0
    listing = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/NH"], capture_output=True, text=True,
                             check=False).stdout
    return "llama-server.exe" in listing


def server_status() -> dict[str, Any]:
    """llama-server: up?, the model it serves and the context size it was started with."""
    props = probe(SERVER_URL.rstrip("/") + "/props")
    if props is None:
        return {"up": False, "starting": server_process_running()}
    settings = props.get("default_generation_settings", {})
    return {"up": True, "model": props.get("model_alias") or props.get("model_path", "?"), "n_ctx": settings.get("n_ctx")}


def context_usage() -> dict[str, Any] | None:
    """How full the local model's window is, as last reported by the gateway; None when stale or absent."""
    try:
        usage = json.loads((state_dir() / "gateway_context.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - usage.get("time", 0) > CONTEXT_FRESH_S:
        return None
    return {"tokens": usage["tokens"], "window": usage["window"]}


def status() -> dict[str, Any]:
    """Everything the panel header shows."""
    port = int(_read_yaml(config_dir() / "orchestrator.yaml").get("port", 8081))
    return {"router": {"up": port_open(port), "port": port}, "server": server_status(), "context": context_usage(),
            "direct_mode": direct_mode(), "role": local_role(), "orchestrator": orchestrator_status(), "llama_installed": llama_installed(),
            "thinking": get_thinking()}


def load_profiles() -> tuple[dict[str, Any], str]:
    """(profiles.json, this machine's fingerprint); the fingerprint comes from the tuner so both always agree."""
    spec = importlib.util.spec_from_file_location("tune_llama", LLAMA / "tune_llama.py")
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)
    profiles = json.loads((LLAMA / "profiles.json").read_text(encoding="utf-8"))
    return profiles, tuner.fingerprint()[0]


def model_entry(path: Path, profiles: dict[str, Any], machine: str) -> dict[str, Any]:
    """One row of the model list: file, size and, when tuned for this machine, the stored profile."""
    profile = profiles.get(f"{machine} || {path.stem.lower()}")
    entry = {"file": path.name, "size_gb": round(path.stat().st_size / 2**30, 1), "tuned": profile is not None,
             "moe": bool(MOE_NAME.search(path.name))}
    if profile:
        entry.update({key: profile.get(key) for key in PROFILE_FIELDS})
    return entry


def list_models() -> list[dict[str, Any]]:
    """Every .gguf in llama/ (and $FOREMANKIT_MODELS_DIR), vision projectors skipped, sorted by name."""
    folders = [LLAMA] + ([Path(os.environ["FOREMANKIT_MODELS_DIR"])] if os.getenv("FOREMANKIT_MODELS_DIR") else [])
    files = sorted((p for folder in folders if folder.is_dir() for p in folder.glob("*.gguf") if "mmproj" not in p.name),
                   key=lambda p: p.name.lower())
    try:
        profiles, machine = load_profiles()
    except (OSError, ValueError, RuntimeError):
        profiles, machine = {}, ""
    return [model_entry(path, profiles, machine) for path in files]


def llama_installed() -> bool:
    """True when llama/LLAMA.CPP holds the llama-server binary (it is not in git: llama/get_llama.py downloads it)."""
    return (LLAMA / "LLAMA.CPP" / ("llama-server.exe" if os.name == "nt" else "llama-server")).exists()


def python_exe() -> str:
    """The interpreter of this repository's virtual environment (the one running this module)."""
    return sys.executable


def command_for(action: str, model: str = "") -> str:
    """PowerShell line for a long action, to be run in a visible terminal. Raises KeyError for an unknown action.

    `tune` stops the server, tunes the offload and threads, then finds the largest fast context (one job). Models whose
    name carries no active-parameter suffix (-A3B, -A4B) are treated as dense: they have no experts to offload."""
    py = f"& '{python_exe()}'"
    dense = "" if MOE_NAME.search(model) else " --dense"
    stop = f"{py} -m foremankit.panel stop-server"
    tune = f"{py} '{LLAMA / 'tune_llama.py'}' --model '{model}'{dense}"
    sweep = f"{py} '{LLAMA / 'sweep_ctx.py'}' '{model}'"
    commands = {
        "tune": f"{stop}; {tune}; {sweep}",
        "download-llama": f"{py} '{LLAMA / 'get_llama.py'}'",
    }
    return commands[action]


def call_target(call: dict[str, Any]) -> str:
    """Who did the work: provider (and node for a local machine) plus model, as far as the log knows."""
    place = f"{call.get('provider', '?')}@{call['node']}" if call.get("node") else call.get("provider", "?")
    return f"{place} / {call['model']}" if call.get("model") else place


def delegations(now: float | None = None) -> dict[str, Any]:
    """Delegated tasks from the router's call log: the latest ones, today's and the last days' cost, split by target.

    Costs are what the providers charged according to the log (`cost_usd`; free and local calls are 0). Days are local."""
    now = time.time() if now is None else now
    today = datetime.date.fromtimestamp(now)
    calls = [call for call in call_log().read() if call.get("event") == "call" and not call.get("dry_run")]
    per_day: dict[str, float] = collections.defaultdict(float)
    today_by_target: dict[str, dict[str, float]] = collections.defaultdict(lambda: {"calls": 0, "cost": 0.0})
    for call in calls:
        day = datetime.date.fromtimestamp(call["ts"])
        per_day[day.isoformat()] += call.get("cost_usd") or 0.0
        if day == today:
            row = today_by_target[call_target(call)]
            row["calls"] += 1
            row["cost"] += call.get("cost_usd") or 0.0
    days = [(today - datetime.timedelta(days=back)).isoformat() for back in range(HISTORY_DAYS - 1, -1, -1)]
    recent = [{"ts": call["ts"], "kind": call.get("kind"), "target": call_target(call), "ok": bool(call.get("success")),
               "tokens": (call.get("tokens_in") or 0) + (call.get("tokens_out") or 0), "cost": call.get("cost_usd") or 0.0,
               "seconds": call.get("latency_s"), "escalated": bool(call.get("escalate"))} for call in calls[-RECENT_CALLS:]]
    return {"today": {"calls": sum(row["calls"] for row in today_by_target.values()), "cost": per_day.get(today.isoformat(), 0.0),
                      "by_target": dict(today_by_target)},
            "days": [{"day": day, "cost": round(per_day.get(day, 0.0), 6)} for day in days],
            "week_cost": round(sum(per_day.get(day, 0.0) for day in days), 6), "recent": list(reversed(recent))}


def call_log() -> Any:
    """The router's call log (a function so tests can point it at a temporary state dir)."""
    from foremankit.state import CallLog
    return CallLog()


def set_role(role: str) -> str:
    """What the local server is for. The gateway reads the role file (orchestrator answers Claude Code, worker refuses);
    the router's own toggle `llm:self` decides whether delegated tasks may use this machine (off for the orchestrator)."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    (state_dir() / LOCAL_ROLE_FILE).write_text(role, encoding="utf-8")
    StateStore().update(lambda state: state.setdefault("toggles", {}).__setitem__("llm:self", ROLES[role]))
    return f"local server role: {role}"


def worker_url(name: str, entry: dict[str, Any]) -> str:
    """The URL of a node entry of nodes.yaml (first LAN address and the node's port)."""
    addresses = _addresses(name, entry, "lan")
    return _url(addresses[0], int(entry.get("port", 8080))) if addresses else ""


def worker_status(url: str) -> dict[str, Any]:
    """Is the server at `url` answering, and with which model (llama.cpp /props, else the OpenAI /v1/models list)."""
    props = probe(url.rstrip("/") + "/props")
    if props is not None:
        return {"up": True, "model": props.get("model_alias") or props.get("model_path", "?")}
    models = probe(url.rstrip("/") + "/v1/models")
    listed = (models or {}).get("data") or [{}]
    return {"up": models is not None, "model": listed[0].get("id")}


def worker_enabled(name: str) -> bool:
    """False while the router toggle `llm:<name>` is off (the worker stays configured but gets no tasks)."""
    return normalize_toggle(StateStore().load().get("toggles", {}).get(f"llm:{name}")) != "off"


def list_workers() -> list[dict[str, Any]]:
    """Every node except this machine: the lab's machines from nodes.yaml (read-only here) and the extra workers added in
    the panel (`editable`), each with its live status, whether the router may use it and whether it is the orchestrator."""
    nodes = load_nodes()
    me = self_name(nodes)
    extra = load_workers()
    chosen = (orchestrator_target() or {}).get("name")
    rows = []
    for name, entry in (nodes.get("nodes") or {}).items():
        if name != me:
            url = worker_url(name, entry or {})
            rows.append({"name": name, "url": url, "accept": (entry or {}).get("accept", "private"),
                         "priority": (entry or {}).get("priority"), "editable": name in extra,
                         "enabled": worker_enabled(name), "orchestrator": name == chosen, **worker_status(url)})
    return rows


def find_worker_url(name: str) -> str:
    """Address of a worker of the pool (nodes.yaml or added in the panel) without probing it."""
    entry = (load_nodes().get("nodes") or {}).get(name)
    if entry is None:
        raise ValueError(f"{name} is not a known worker")
    return worker_url(name, entry)


def timed_get(url: str, key: str | None) -> tuple[float, int]:
    """(seconds, HTTP status) of a GET with the lab key; an HTTP error answer is a result, a network failure is raised."""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=5) as answer:
            status = answer.status
    except urllib.error.HTTPError as problem:
        status = problem.code
    return time.monotonic() - started, status


def test_worker(name: str) -> str:
    """Check a worker end to end: it answers, the lab key is accepted and a short generation works; one line of result."""
    url = find_worker_url(name).rstrip("/")
    key = os.getenv("FOREMANKIT_PEER_KEY") or None
    try:
        seconds, status = timed_get(url + "/health", key)
    except OSError as problem:
        return f"{name}: no connection to {url} ({problem}). Is its llama-server running with -Share, and the firewall open?"
    if status != 200:
        return f"{name}: reachable in {seconds * 1000:.0f} ms but answered HTTP {status}"
    body = json.dumps({"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 16}).encode()
    request = urllib.request.Request(url + "/v1/chat/completions", body, {
        "Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            used = json.load(answer).get("usage", {}).get("completion_tokens", 0)
    except urllib.error.HTTPError as problem:
        hint = " (the lab key FOREMANKIT_PEER_KEY differs from the server's)" if problem.code in (401, 403) else ""
        return f"{name}: reachable in {seconds * 1000:.0f} ms, but generation failed with HTTP {problem.code}{hint}"
    except OSError as problem:
        return f"{name}: reachable, but generation failed ({problem})"
    return f"{name}: OK. Reachable in {seconds * 1000:.0f} ms, generated {used} tokens in {time.monotonic() - started:.1f} s"


def set_worker_enabled(name: str, enabled: bool) -> str:
    """Let the router use a worker (auto) or not (off) without deleting it."""
    if name not in {row["name"] for row in list_workers()}:
        raise ValueError(f"{name} is not a known worker")
    StateStore().update(lambda state: state.setdefault("toggles", {}).__setitem__(f"llm:{name}", "auto" if enabled else "off"))
    return f"worker {name} {'enabled' if enabled else 'disabled'}"


def orchestrator_status() -> dict[str, Any]:
    """Where Claude Code's local model runs: {"name": "this PC" or a worker name, "url", "up", "model"}."""
    target = orchestrator_target()
    if target is None:
        return {"name": "this PC", "url": SERVER_URL}
    return {"name": target.get("name", "?"), "url": target["url"], **worker_status(target["url"])}


def set_orchestrator(name: str) -> str:
    """Serve Claude Code from `name` (a worker of the pool, which must run its llama-server with -Share) or from this PC.

    The gateway reads the file on every request: no restart. A remote machine needs the same FOREMANKIT_PEER_KEY."""
    path = state_dir() / ORCHESTRATOR_FILE
    if name in ("", "this", "this PC"):
        path.unlink(missing_ok=True)
        return "orchestrator: this PC"
    row = next((worker for worker in list_workers() if worker["name"] == name), None)
    if row is None or not row["url"]:
        raise ValueError(f"{name} is not a known worker with an address")
    path.write_text(json.dumps({"name": name, "url": row["url"]}), encoding="utf-8")
    note = "" if row["up"] else " (it does not answer right now: start its llama-server with -Share)"
    return f"orchestrator: {name} at {row['url']}{note}"


def add_worker(name: str, url: str, accept: str = "internal", priority: int = 10) -> str:
    """Register an extra llama-server worker (a node of the pool `llm`); replaces a worker of the same name."""
    if not WORKER_NAME.match(name):
        raise ValueError("name: 1-32 characters, lower-case letters, digits, - and _")
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https:// (for example http://192.168.1.50:8080)")
    if accept not in LEVELS:
        raise ValueError(f"accept must be one of {', '.join(LEVELS)}")
    workers = load_workers()
    workers[name] = {"url": url.rstrip("/"), "accept": accept, "priority": int(priority)}
    (state_dir() / WORKERS_FILE).write_text(json.dumps(workers, indent=1), encoding="utf-8")
    return f"worker {name} saved (accepts data up to {accept})"


def remove_worker(name: str) -> str:
    """Remove an extra worker added in the panel; workers of nodes.yaml cannot be removed here."""
    workers = load_workers()
    if name not in workers:
        raise ValueError(f"{name} is not an extra worker (workers of nodes.yaml are edited in the file)")
    del workers[name]
    (state_dir() / WORKERS_FILE).write_text(json.dumps(workers, indent=1), encoding="utf-8")
    return f"worker {name} removed"


def checked_manual_value(name: str, value: Any) -> Any:
    """One manual launch parameter, converted and range-checked; ValueError names the offending field."""
    if name in MANUAL_RANGES:
        kind, low, high = MANUAL_RANGES[name]
        try:
            number = kind(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number") from None
        if not low <= number <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return number
    if value not in MANUAL_CHOICES[name]:
        raise ValueError(f"{name} must be one of {', '.join(MANUAL_CHOICES[name])}")
    return value


def set_profile(model: str, fields: dict[str, Any]) -> str:
    """Store hand-picked launch parameters in the model's profile (created when the model was never tuned).

    Empty values are dropped from the profile, so the launcher's defaults apply again. The launcher reads the profile
    on every start: nothing else needs to change. `manual: true` marks the entry as set by hand."""
    profiles, machine = load_profiles()
    key = f"{machine} || {Path(model).stem.lower()}"
    profile = profiles.setdefault(key, {"model": model})
    for name, value in fields.items():
        if name not in MANUAL_RANGES and name not in MANUAL_CHOICES:
            raise ValueError(f"unknown parameter {name}")
        if value in (None, ""):
            profile.pop(name, None)
        else:
            profile[name] = checked_manual_value(name, value)
    profile["manual"] = True
    (LLAMA / "profiles.json").write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"{model}: profile saved (applies at the next server start)"


@functools.cache
def load_installer() -> Any:
    """install.py is a script, not a package: load it by path (once) to reuse its VS Code and scheduler logic."""
    spec = importlib.util.spec_from_file_location("foremankit_install", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vscode_settings_path() -> Path:
    """The VS Code user settings.json (the installer knows where it lives on every platform)."""
    return load_installer().Installer.vscode_user_dir() / "settings.json"


def direct_mode() -> bool:
    """True when VS Code no longer points Claude Code at the router, i.e. it talks straight to Anthropic (the default)."""
    try:
        settings = json.loads(vscode_settings_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return True
    names = {entry.get("name") for entry in settings.get("claudeCode.environmentVariables", [])}
    return "ANTHROPIC_BASE_URL" not in names


def set_direct_mode(direct: bool) -> str:
    """Direct mode (default): nothing of the router stays in VS Code or the Task Scheduler and the router is stopped.
    Router mode: the settings entries and the autostart are put in place and the router is started.

    Claude Code reads these variables only when a window starts, so the caller must tell the user to reload the window."""
    install = load_installer()
    installer = install.Installer(dry=False)
    with contextlib.redirect_stdout(io.StringIO()):   # the installer narrates; stdout must stay pure JSON
        if direct:
            installer.disable_router_mode()
        else:
            installer.enable_router_mode(Path(sys.executable))
    run_quick("stop-router" if direct else "start-router")
    mode = ("direct mode: Claude Code will talk straight to Anthropic" if direct
            else "router mode: Claude Code goes through the router (local models available)")
    return mode + ". RELOAD THE VS CODE WINDOW to apply."


def get_thinking() -> dict[str, bool]:
    """Which effort levels make the local model think (config/orchestrator.yaml, defaults from the gateway)."""
    configured = _read_yaml(config_dir() / "orchestrator.yaml").get("thinking_by_effort") or {}
    merged = {**GATEWAY_DEFAULTS["thinking_by_effort"], **configured}
    return {level: bool(merged[level]) for level in EFFORT_LEVELS}


def set_thinking(levels: dict[str, bool]) -> str:
    """Rewrite the thinking_by_effort line of orchestrator.yaml; the gateway re-reads the file on every request."""
    path = config_dir() / "orchestrator.yaml"
    flow = ", ".join(f"{level}: {'true' if levels[level] else 'false'}" for level in EFFORT_LEVELS)
    text = THINKING_LINE.sub(lambda _: "thinking_by_effort: {" + flow + "}", path.read_text(encoding="utf-8"))
    path.write_text(text, encoding="utf-8")
    return "thinking on for: " + (", ".join(level for level in EFFORT_LEVELS if levels[level]) or "no level")


def parse_levels(text: str) -> dict[str, bool]:
    """'low=false,high=true' -> {'low': False, ...}; levels not named keep their current value."""
    levels = get_thinking()
    for pair in filter(None, text.split(",")):
        name, _, value = pair.partition("=")
        if name in levels:
            levels[name] = value.strip().lower() == "true"
    return levels


def start_server(model: str, share: bool = False) -> str:
    """Start llama-server detached from VS Code: a terminal child dies when the window reloads (the router switch does that).

    The output goes to llama-server.log in the temp folder; -Replace swaps a server that is already running."""
    if not model:
        raise ValueError("choose a model first")
    log = Path(tempfile.gettempdir()) / SERVER_LOG
    script = LLAMA / "start-llama.ps1"
    # redirect at process level: PowerShell 5.1's `*>` turns the server's stderr lines into errors and stops the script
    share_flag = ",'-Share'" if share else ""
    arguments = f"'-NoProfile','-ExecutionPolicy','Bypass','-File','\"{script}\"','-Model','{model}','-NoMenu','-Replace'{share_flag}"
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Start-Process -WindowStyle Hidden -FilePath powershell.exe -ArgumentList {arguments} "
                    f"-RedirectStandardOutput '{log}' -RedirectStandardError '{log}.err'"], check=False)
    shared = " shared with the lab (-Share)" if share else ""
    return f"starting {model}{shared} (log: {log}); loading takes about a minute"


def run_quick(action: str, model: str = "", share: bool = False) -> str:
    """Execute a quick action and return one line saying what happened."""
    if action == "start-server":
        return start_server(model, share)
    if action in ("router-direct", "router-normal"):
        return set_direct_mode(action == "router-direct")
    if action == "stop-server":
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"], check=False)
        return "llama-server stopped"
    flag = {"start-router": [], "stop-router": ["-Stop"]}[action]
    result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(LLAMA / "Start-Router.ps1"), *flag], capture_output=True, text=True, check=False)
    return (result.stdout.strip().splitlines() or ["done"])[-1]


def run_config_action(args: argparse.Namespace) -> str:
    """role / worker-add / worker-remove from the command line; a rejected value comes back as a message, not a crash."""
    try:
        if args.what == "role":
            return set_role(args.action or "")
        if args.what == "worker-test":
            return test_worker(args.name)
        if args.what == "orchestrator":
            return set_orchestrator(args.name)
        if args.what == "worker-enable":
            return set_worker_enabled(args.name, args.action == "true")
        if args.what == "profile-set":
            return set_profile(args.model, json.loads(args.fields))
        if args.what == "worker-add":
            return add_worker(args.name, args.url, args.accept, args.priority)
        return remove_worker(args.name)
    except ValueError as problem:
        return f"not done: {problem}"


def main(argv: list[str] | None = None) -> None:
    """Console entry: print JSON for status/models/command, run the quick actions."""
    parser = argparse.ArgumentParser(prog="foremankit.panel")
    parser.add_argument("what", choices=["status", "models", "command", "start-router", "stop-router", "stop-server", "start-server", "router-direct", "router-normal", "thinking", "role", "workers", "worker-add", "worker-remove", "worker-enable", "worker-test", "orchestrator", "profile-set", "delegations"])
    parser.add_argument("action", nargs="?")
    parser.add_argument("--model", default="")
    parser.add_argument("--fields", default="{}", help="profile-set: JSON object of launch parameters")
    parser.add_argument("--share", action="store_true", help="start-server: listen on the LAN with the lab key (-Share)")
    parser.add_argument("--name", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--accept", default="internal")
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--levels", default="", help="thinking: comma list like low=false,high=true")
    args = parser.parse_args(argv)
    if args.what == "status":
        print(json.dumps(status()))
    elif args.what == "models":
        print(json.dumps(list_models()))
    elif args.what == "command":
        print(json.dumps({"command": command_for(args.action, args.model)}))
    elif args.what == "delegations":
        print(json.dumps(delegations()))
    elif args.what == "workers":
        print(json.dumps(list_workers()))
    elif args.what in ("role", "worker-add", "worker-remove", "worker-enable", "worker-test", "orchestrator", "profile-set"):
        print(json.dumps({"message": run_config_action(args)}))
    elif args.what == "thinking":
        print(json.dumps({"message": set_thinking(parse_levels(args.levels))}))
    else:
        print(json.dumps({"message": run_quick(args.what, args.model, args.share)}))


if __name__ == "__main__":
    main()
