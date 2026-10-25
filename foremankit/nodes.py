"""Lab nodes: every machine that can run a local LLM server (llama.cpp llama-server, Ollama, ...), this one included.

The SAME config/nodes.yaml works on every machine: each one finds itself by hostname (or env FOREMANKIT_SELF) and treats
all the others as extra sources of the pool `llm`. Nothing here is specific to one lab: add a block per machine.
"""
from __future__ import annotations

import ipaddress
import json
import os
import platform
import secrets
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

from .config import REPO_DIR, _read_yaml, config_dir, state_dir
from .providers import Endpoint

LEVELS = ("normal", "internal", "private")        # data sensitivity, least to most confidential
DEFAULT_LOAN = {"when_idle": True, "max_seconds": 45}
DEFAULT_ACCEPT = "private"                         # a node of the lab may receive any data unless its entry says otherwise
LOCAL_NAME = "local"                               # name of this machine when it is not listed in nodes.yaml
_RESOLVED: dict[str, tuple[float, str | None]] = {}
RESOLVE_TTL_S = 60.0
DEFAULT_LLM_PORT = 8080              # llama-server's default port; used when a node entry in nodes.yaml names none


@dataclass
class Node:
    """One machine of the lab that serves a local LLM (config/nodes.yaml) and how to reach and use it."""
    name: str
    is_self: bool = False
    priority: int = 10**6
    accept: str = DEFAULT_ACCEPT              # highest sensitivity this node may receive (normal < internal < private)
    speed: dict[str, Any] | None = None       # expected tok/s: prompt_tps, gen_tps, expected_output_cap, max_seconds
    learn: bool = True                        # False = the written `speed` is final (it was measured); see foremankit/speed.py
    kinds: dict[str, Any] | None = None       # per-kind input limits; None = routing.yaml llm.kinds
    max_concurrent: int = 1
    preferred_models: list[str] | None = None
    endpoints: list[Endpoint] = field(default_factory=list)
    idle_only: bool = False                   # a loan from a higher-priority machine: only while its server is idle
    loan_max_s: float | None = None           # ... and only for tasks estimated shorter than this
    hostname: str = ""

    def accepts(self, sensitivity: str) -> bool:
        """Whether data of this sensitivity may be sent to the node."""
        return LEVELS.index(sensitivity) <= LEVELS.index(self.accept)

    @property
    def entry(self) -> str:
        """Chain/log name of this node."""
        return f"llm@{self.name}"


def _ipv4(host: str) -> str | None:
    """Current IPv4 address of a LAN host name (cached for a minute), or None. A name often resolves first to an IPv6
    link-local address (fe80::, unusable without an interface id): the client then waits ~10 s on it before trying IPv4.
    Resolving to IPv4 ourselves avoids that and still follows DHCP changes. Off with FOREMANKIT_PEER_RESOLVE=off."""
    if os.environ.get("FOREMANKIT_PEER_RESOLVE", "").strip().lower() in {"off", "0", "false", "no"}:
        return None
    try:
        ipaddress.ip_address(host)
        return None                                   # already an IP literal
    except ValueError:
        pass
    cached = _RESOLVED.get(host.lower())
    if cached and time.monotonic() - cached[0] < RESOLVE_TTL_S:
        return cached[1]
    try:
        ip = socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)[0][4][0]
    except OSError:
        ip = None
    _RESOLVED[host.lower()] = (time.monotonic(), ip)
    return ip


WORKERS_FILE = "workers.json"        # extra llama-server workers added in the VS Code panel (user-level, never committed)


def load_workers() -> dict[str, dict[str, Any]]:
    """Extra workers {name: {url, accept, priority, max_concurrent}} from the state dir; {} when none or unreadable."""
    try:
        data = json.loads((state_dir() / WORKERS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def worker_entry(worker: dict[str, Any]) -> dict[str, Any]:
    """The nodes.yaml-style entry of an extra worker; its URL is used exactly as written (see `_url`)."""
    return {"lan": [worker["url"]], "serves": True, "accept": worker.get("accept", "internal"),
            "priority": worker.get("priority", 10), "max_concurrent": worker.get("max_concurrent", 1)}


def with_workers(data: dict[str, Any]) -> dict[str, Any]:
    """`data` (the parsed nodes.yaml) plus the extra workers; a node already named in the file wins."""
    workers = load_workers()
    if not workers:
        return data
    nodes = {name: worker_entry(worker) for name, worker in workers.items()}
    nodes.update(data.get("nodes") or {})
    return {**data, "nodes": nodes}


def load_nodes(directory: Path | None = None) -> dict[str, Any]:
    """nodes.yaml from the active config dir, else from the repo's config/ (a user-level config dir may lack it),
    plus the extra workers added in the panel (only for the active config, never for an explicit directory)."""
    if os.environ.get("FOREMANKIT_NODES", "").strip().lower() in {"off", "0", "false", "no"}:
        return {}
    for folder in ([directory] if directory else [config_dir(), REPO_DIR / "config"]):
        data = _read_yaml(folder / "nodes.yaml")
        if data:
            return data if directory else with_workers(data)
    return {} if directory else with_workers({})


def self_name(cfg: dict[str, Any]) -> str | None:
    """Which entry of nodes.yaml is THIS machine: env FOREMANKIT_SELF (name or hostname), else the hostname."""
    wanted = (os.environ.get("FOREMANKIT_SELF") or platform.node()).strip().lower()
    for name, node in (cfg.get("nodes") or {}).items():
        if wanted in {name.lower(), str((node or {}).get("hostname", "")).lower()}:
            return name
    return None


def lab_key(cfg: dict[str, Any]) -> str | None:
    """The shared lab key (from the environment variable named in nodes.yaml), or None."""
    return os.environ.get((cfg.get("settings") or {}).get("key_env", "FOREMANKIT_PEER_KEY")) or None


def _addresses(name: str, node: dict[str, Any], kind: str) -> list[str]:
    """Addresses of a node for one kind of link (lan/tunnel); an environment variable overrides the file."""
    override = os.environ.get(f"FOREMANKIT_NODE_{name.upper()}_{kind.upper()}")
    return [part.strip() for part in override.split(",") if part.strip()] if override else list(node.get(kind) or [])


def _url(address: str, port: int) -> str:
    if "://" in address:                      # a full URL (e.g. https://neuron.tailnet.ts.net) is used exactly as written
        return address.rstrip("/")
    host = _ipv4(address) or address
    return f"http://{host}" if ":" in host else f"http://{host}:{port}"   # name or IPv4 + the node's llama-server port


def _machine_speed(speed: dict[str, Any] | None) -> dict[str, Any] | None:
    """`speed` of THIS machine with overrides from LOCAL_LLM_GEN_TPS / LOCAL_LLM_PROMPT_TPS (invalid values are ignored)."""
    if not speed:
        return speed
    out = dict(speed)
    for env, key in (("LOCAL_LLM_GEN_TPS", "gen_tps"), ("LOCAL_LLM_PROMPT_TPS", "prompt_tps")):
        try:
            value = float(os.environ.get(env, ""))
        except ValueError:
            continue
        if value > 0:
            out[key] = value
    return out


LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def local_endpoints(provider_cfg: dict[str, Any], key: str | None = None) -> list[Endpoint]:
    """The servers on THIS machine (providers.yaml llm.endpoints: llama.cpp first, then Ollama). `key` = the lab key:
    a server started with -Share demands it from every client, this machine included; it is only sent to a loopback
    address, so a url_env pointing elsewhere never receives it."""
    out = []
    for ep in provider_cfg.get("endpoints") or []:
        url = (os.environ.get(ep["url_env"]) if ep.get("url_env") else None) or ep.get("url") or ep.get("default_url")
        if url:
            loopback = urlsplit(url).hostname in {host.strip("[]") for host in LOOPBACK_HOSTS}
            out.append(Endpoint(ep.get("name", "default"), url.rstrip("/"), key if loopback else None))
    return out


UNLISTED_PRIORITY = 10 ** 6    # a machine that is not in nodes.yaml is the least important one


class _EntryReader:
    """Reads one setting of a node entry, falling back to the `defaults` block of nodes.yaml."""

    def __init__(self, defaults: dict[str, Any]):
        self.defaults = defaults

    def get(self, entry: dict[str, Any], key: str, fallback: Any = None) -> Any:
        """The entry's own value, else the shared default, else `fallback`."""
        return entry.get(key, self.defaults.get(key, fallback))


@dataclass(frozen=True)
class _Lab:
    """Settings shared by all the nodes of one nodes.yaml."""
    reader: _EntryReader
    default_speed: dict[str, Any] | None
    my_priority: int
    loan: dict[str, Any]
    key: str | None
    link_kinds: list[str]


def build_nodes(nodes_cfg: dict[str, Any] | None, provider_cfg: dict[str, Any]) -> list[Node]:
    """This machine first (when it serves), then the others, higher priority first. Without a nodes.yaml (or when this
    machine is not listed) this machine is the single node `local`, with the provider's default speed."""
    nodes_cfg = nodes_cfg or {}
    entries = nodes_cfg.get("nodes") or {}
    me = self_name(nodes_cfg)
    reader = _EntryReader(nodes_cfg.get("defaults") or {})
    my_entry = (entries.get(me) or {}) if me else {}
    lab = _Lab(reader, provider_cfg.get("speed"),
               int(reader.get(my_entry, "priority", 1)) if me else UNLISTED_PRIORITY,
               {**DEFAULT_LOAN, **((nodes_cfg.get("settings") or {}).get("loan_to_higher_priority") or {})},
               lab_key(nodes_cfg), _link_kinds(nodes_cfg))
    nodes = []
    if not me or reader.get(my_entry, "serves", True):
        nodes.append(_this_machine(me, my_entry, lab, provider_cfg))
    others = sorted(((name, entry or {}) for name, entry in entries.items() if name != me),
                    key=lambda item: int(reader.get(item[1], "priority", 1)))
    nodes += [_other_node(name, entry, lab) for name, entry in others if reader.get(entry, "serves", True)]
    return nodes


def _link_kinds(nodes_cfg: dict[str, Any]) -> list[str]:
    """Kinds of link tried for the other nodes: the LAN always, a tunnel only when enabled."""
    tunnel = bool((nodes_cfg.get("settings") or {}).get("tunnel_enabled"))
    return ["lan"] + (["tunnel"] if tunnel else [])


def _this_machine(me: str | None, entry: dict[str, Any], lab: _Lab, provider_cfg: dict[str, Any]) -> Node:
    reader = lab.reader
    return Node(
        name=me or LOCAL_NAME, is_self=True, priority=lab.my_priority,
        accept=reader.get(entry, "accept", DEFAULT_ACCEPT) if me else DEFAULT_ACCEPT,
        speed=_machine_speed(entry.get("speed") or lab.default_speed), kinds=entry.get("kinds"),
        learn=bool(reader.get(entry, "learn", True)),
        max_concurrent=int(reader.get(entry, "max_concurrent", provider_cfg.get("max_concurrent", 1))),
        preferred_models=entry.get("preferred_models"), endpoints=local_endpoints(provider_cfg, lab.key),
        hostname=str(entry.get("hostname", "")))


def _other_node(name: str, entry: dict[str, Any], lab: _Lab) -> Node:
    reader = lab.reader
    priority = int(reader.get(entry, "priority", 1))
    restricted = lab.my_priority > priority                           # borrowing from a more important machine
    return Node(
        name=name, priority=priority, accept=reader.get(entry, "accept", DEFAULT_ACCEPT),
        speed=entry.get("speed") or lab.default_speed, kinds=entry.get("kinds"),
        learn=bool(reader.get(entry, "learn", True)), max_concurrent=int(reader.get(entry, "max_concurrent", 1)),
        preferred_models=entry.get("preferred_models"), endpoints=_remote_endpoints(name, entry, lab),
        idle_only=bool(restricted and lab.loan.get("when_idle")),
        loan_max_s=float(lab.loan["max_seconds"]) if restricted else None,
        hostname=str(entry.get("hostname", "")))


def _remote_endpoints(name: str, entry: dict[str, Any], lab: _Lab) -> list[Endpoint]:
    """One endpoint per address of every kind of link; the 2nd, 3rd ... address of a kind get a #number.
    A node with its own `key_env` sends THAT key (no header when the variable is unset), so the shared lab key
    never travels to a server outside the lab (e.g. a tunnelled notebook)."""
    port = int(lab.reader.get(entry, "port", DEFAULT_LLM_PORT))
    key = os.environ.get(entry["key_env"]) if entry.get("key_env") else lab.key
    endpoints = []
    for kind in lab.link_kinds:
        for number, address in enumerate(_addresses(name, entry, kind), 1):
            label = f"{name}/{kind}" + (f"#{number}" if number > 1 else "")
            endpoints.append(Endpoint(label, _url(address, port), key))
    return endpoints


def ensure_key(env_file: Path | None = None, var: str = "FOREMANKIT_PEER_KEY") -> tuple[str, bool]:
    """Make sure the lab key exists in the repo .env; returns (path, created). The key is never printed."""
    env_file = env_file or REPO_DIR / ".env"
    text = env_file.read_text(encoding="utf-8") if env_file.is_file() else ""
    if any(line.startswith(f"{var}=") and line.split("=", 1)[1].strip() for line in text.splitlines()):
        return str(env_file), False
    lines = [ln for ln in text.splitlines() if not ln.startswith(f"{var}=")]
    lines.append(f"{var}=" + secrets.token_urlsafe(32))
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(env_file), True
