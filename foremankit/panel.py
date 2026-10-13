"""Backend of the ForemanKit VS Code panel: status of router and llama-server, the model list and the action commands.

The VS Code extension (vscode-extension/) is only a thin shell: it runs `python -m foremankit.panel <what>` and shows the
JSON this module prints. Keeping the logic here makes it testable and keeps the extension free of build steps.

    status                  router up?, llama-server up (model, context size), context fill reported by the gateway
    models                  every .gguf next to llama/start-llama.ps1 with its tuned profile (speed, context, offload)
    command <action> ...    the PowerShell line the extension runs in a visible terminal (long jobs: start, tune, sweep)
    start-router | stop-router | stop-server | router-direct | router-normal
    role <orchestrator|worker|both>    what the local server is for
    profile-set --model F --fields '{"ctx": 65536, "temp": 0.6}'    hand-picked launch parameters
    workers | worker-add --name N --url U [--accept A --priority P] | worker-remove --name N
    thinking --levels low=false,high=true    quick actions executed here (router-direct/-normal edit the VS Code
                            settings: the window must be reloaded afterwards)
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from foremankit.config import _read_yaml, config_dir, state_dir
from foremankit.nodes import LEVELS, WORKERS_FILE, _addresses, _url, load_nodes, load_workers, self_name
from foremankit.state import StateStore
from foremankit.gateway import DEFAULTS as GATEWAY_DEFAULTS
from foremankit.gateway import LOCAL_ROLE_FILE, local_role

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


def server_status() -> dict[str, Any]:
    """llama-server: up?, the model it serves and the context size it was started with."""
    props = probe(SERVER_URL.rstrip("/") + "/props")
    if props is None:
        return {"up": False}
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
            "direct_mode": direct_mode(), "role": local_role(), "llama_installed": llama_installed(),
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
        "start-server": f"& '{LLAMA / 'start-llama.ps1'}' -Model '{model}' -NoMenu -Replace",
        "tune": f"{stop}; {tune}; {sweep}",
        "download-llama": f"{py} '{LLAMA / 'get_llama.py'}'",
    }
    return commands[action]


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


def list_workers() -> list[dict[str, Any]]:
    """Every node except this machine: the lab's machines from nodes.yaml (read-only here) and the extra workers added in
    the panel (`editable`), each with its live status."""
    nodes = load_nodes()
    me = self_name(nodes)
    extra = load_workers()
    rows = []
    for name, entry in (nodes.get("nodes") or {}).items():
        if name != me:
            url = worker_url(name, entry or {})
            rows.append({"name": name, "url": url, "accept": (entry or {}).get("accept", "private"),
                         "priority": (entry or {}).get("priority"), "editable": name in extra, **worker_status(url)})
    return rows


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


def load_installer() -> Any:
    """install.py is a script, not a package: load it by path to reuse its VS Code settings logic."""
    spec = importlib.util.spec_from_file_location("foremankit_install", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vscode_settings_path() -> Path:
    """The VS Code user settings.json (the installer knows where it lives on every platform)."""
    return load_installer().Installer.vscode_user_dir() / "settings.json"


def direct_mode() -> bool:
    """True when VS Code no longer points Claude Code at the router, i.e. it talks straight to Anthropic."""
    try:
        settings = json.loads(vscode_settings_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    names = {entry.get("name") for entry in settings.get("claudeCode.environmentVariables", [])}
    return "ANTHROPIC_BASE_URL" not in names


def set_direct_mode(direct: bool) -> str:
    """Direct mode: stop the router and remove its entries from VS Code settings; otherwise put them back and start it.

    Claude Code reads these variables only when a window starts, so the caller must tell the user to reload the window."""
    install = load_installer()
    config = _read_yaml(config_dir() / "orchestrator.yaml")
    base_url = f"http://127.0.0.1:{config.get('port', 8081)}"
    model_id = (config.get("local_models") or ["claude-sonnet-5"])[0]
    with contextlib.redirect_stdout(io.StringIO()):   # the installer narrates; stdout must stay pure JSON
        install.Installer(dry=False).write_json(
            vscode_settings_path(),
            lambda data: install.Installer.merged_vscode_settings(data, base_url, model_id, remove=direct))
    run_quick("stop-router" if direct else "start-router")
    mode = ("direct mode: Claude Code will talk straight to Anthropic" if direct
            else "router mode: Claude Code goes through the router again")
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


def run_quick(action: str) -> str:
    """Execute a quick action and return one line saying what happened."""
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
    parser.add_argument("what", choices=["status", "models", "command", "start-router", "stop-router", "stop-server", "router-direct", "router-normal", "thinking", "role", "workers", "worker-add", "worker-remove", "profile-set"])
    parser.add_argument("action", nargs="?")
    parser.add_argument("--model", default="")
    parser.add_argument("--fields", default="{}", help="profile-set: JSON object of launch parameters")
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
    elif args.what == "workers":
        print(json.dumps(list_workers()))
    elif args.what in ("role", "worker-add", "worker-remove", "profile-set"):
        print(json.dumps({"message": run_config_action(args)}))
    elif args.what == "thinking":
        print(json.dumps({"message": set_thinking(parse_levels(args.levels))}))
    else:
        print(json.dumps({"message": run_quick(args.what)}))


if __name__ == "__main__":
    main()
