"""Backend of the ForemanKit VS Code panel: status of router and llama-server, the model list and the action commands.

The VS Code extension (vscode-extension/) is only a thin shell: it runs `python -m foremankit.panel <what>` and shows the
JSON this module prints. Keeping the logic here makes it testable and keeps the extension free of build steps.

    status                  router up?, llama-server up (model, context size), context fill reported by the gateway
    models                  every .gguf next to llama/start-llama.ps1 with its tuned profile (speed, context, offload)
    command <action> ...    the PowerShell line the extension runs in a visible terminal (long jobs: start, tune, sweep)
    start-router | stop-router | stop-server | local-on | local-off    quick actions executed here
"""
from __future__ import annotations

import argparse
import importlib.util
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
from foremankit.gateway import LOCAL_OFF_FLAG

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama"
SERVER_URL = os.getenv("LLAMACPP_URL", "http://127.0.0.1:8080")
PROBE_TIMEOUT_S = 1.0
CONTEXT_FRESH_S = 600          # the gateway's context counter older than this is not shown
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
            "local_enabled": not (state_dir() / LOCAL_OFF_FLAG).exists()}


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
        entry.update({key: profile.get(key) for key in ("ctx", "tg_tok_s", "ncmoe", "reasoning", "think_budget")})
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


def python_exe() -> str:
    """The interpreter of this repository's virtual environment (the one running this module)."""
    return sys.executable


def command_for(action: str, model: str = "", dense: bool = False) -> str:
    """PowerShell line for a long action, to be run in a visible terminal. Raises KeyError for an unknown action."""
    py = f"& '{python_exe()}'"
    stop = f"{py} -m foremankit.panel stop-server"
    start = f"& '{LLAMA / 'start-llama.ps1'}' -Model '{model}' -NoMenu -Replace"
    commands = {
        "start-server": start,
        "tune": f"{stop}; {py} '{LLAMA / 'tune_llama.py'}' --model '{model}'" + (" --dense" if dense else ""),
        "sweep": f"{stop}; {py} '{LLAMA / 'sweep_ctx.py'}' '{model}'",
        "install": f"{py} '{ROOT / 'install.py'}' --local-orchestrator",
        "uninstall": f"{py} '{ROOT / 'install.py'}' --uninstall",
    }
    return commands[action]


def set_local(enabled: bool) -> str:
    """Switch the local models on or off in the running gateway (a flag file it checks on every request)."""
    flag = state_dir() / LOCAL_OFF_FLAG
    if enabled:
        flag.unlink(missing_ok=True)
    else:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("off", encoding="utf-8")
    return "local models " + ("on" if enabled else "off")


def run_quick(action: str) -> str:
    """Execute a quick action and return one line saying what happened."""
    if action in ("local-on", "local-off"):
        return set_local(action == "local-on")
    if action == "stop-server":
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"], check=False)
        return "llama-server stopped"
    flag = {"start-router": [], "stop-router": ["-Stop"]}[action]
    result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(LLAMA / "Start-Router.ps1"), *flag], capture_output=True, text=True, check=False)
    return (result.stdout.strip().splitlines() or ["done"])[-1]


def main(argv: list[str] | None = None) -> None:
    """Console entry: print JSON for status/models/command, run the quick actions."""
    parser = argparse.ArgumentParser(prog="foremankit.panel")
    parser.add_argument("what", choices=["status", "models", "command", "start-router", "stop-router", "stop-server", "local-on", "local-off"])
    parser.add_argument("action", nargs="?")
    parser.add_argument("--model", default="")
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args(argv)
    if args.what == "status":
        print(json.dumps(status()))
    elif args.what == "models":
        print(json.dumps(list_models()))
    elif args.what == "command":
        print(json.dumps({"command": command_for(args.action, args.model, args.dense)}))
    else:
        print(json.dumps({"message": run_quick(args.what)}))


if __name__ == "__main__":
    main()
