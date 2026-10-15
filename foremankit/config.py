"""Config and path resolution. No absolute paths are hard-coded anywhere."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent


def state_dir() -> Path:
    """Return the user-level state directory for counters, toggles, spend, and call logs."""
    override = os.environ.get("FOREMANKIT_STATE_DIR")
    path = Path(override) if override else Path.home() / ".config" / "foremankit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    """Return the configuration directory based on environment, user config, or repo defaults."""
    override = os.environ.get("FOREMANKIT_CONFIG_DIR")
    if override:
        return Path(override)
    user_cfg = Path.home() / ".config" / "foremankit" / "config"
    if (user_cfg / "providers.yaml").exists():
        return user_cfg
    return REPO_DIR / "config"


def load_env() -> None:
    """Load API keys from .env files without overriding the real environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    for candidate in (REPO_DIR / ".env", state_dir() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary, or empty dict if missing."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle) or {}


class ConfigError(RuntimeError):
    """Raised when the configuration is invalid or uses deprecated settings."""


# The providers llm_local / llm_remote (ollama_local / ollama_remote before) were replaced by ONE pool `llm` whose nodes
# are listed in nodes.yaml. Old names are not translated: a config that still uses them is refused with the fix.
OLD_NAMES = ("llm_local", "llm_remote", "ollama_local", "ollama_remote")
MIGRATION = ("use the single provider `llm` (providers.yaml), list the machines in config/nodes.yaml "
             "(replaces peers.yaml) and switch them with `foremankit llm [self|others|<node>] on|off|auto`")


def _extract_provider_names_from_chain(chain_entry: list[str]) -> set[str]:
    """Extract provider names from a chain entry string (format: 'provider:step')."""
    return {str(entry).partition(":")[0] for entry in chain_entry if entry}


def _collect_providers_from_list(items: list[Any]) -> set[str]:
    """Collect provider names from a list of chain entries."""
    used_providers: set[str] = set()
    for entry in items:
        if isinstance(entry, list):
            used_providers |= _extract_provider_names_from_chain(entry)
        else:
            used_providers.add(str(entry).partition(":")[0])
    return used_providers


def _collect_routing_providers(routing: dict[str, Any]) -> set[str]:
    """Collect provider names referenced in routing chains and escalation paths."""
    used_providers: set[str] = set()
    chains = _get_routing_chains(routing)
    escalation = _get_escalation_path(routing)
    
    for chain in list(chains) + [escalation]:
        if chain:
            used_providers |= _collect_providers_from_list(chain)
            
    return used_providers


def _get_routing_chains(routing: dict[str, Any]) -> list[Any]:
    """Get the list of routing chains from the routing configuration."""
    return list((routing.get("chains") or {}).values())


def _get_escalation_path(routing: dict[str, Any]) -> list[str]:
    """Get the escalation path from the routing configuration."""
    return routing.get("escalation") or []


def _collect_profile_providers(profiles: dict[str, Any]) -> set[str]:
    """Collect provider names referenced in profile flags."""
    used_providers: set[str] = set()
    for flags in (profiles or {}).values():
        if flags:
            used_providers.update(flags)
    return used_providers


def _collect_used_providers(cfg: dict[str, Any]) -> set[str]:
    """Collect all provider names referenced in routing, profiles, and providers."""
    routing = cfg.get("routing") or {}
    profiles = cfg.get("profiles") or {}
    
    used_providers: set[str] = set(cfg.get("providers") or {})
    used_providers |= _collect_routing_providers(routing)
    used_providers |= _collect_profile_providers(profiles)
    
    return used_providers


def check_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Refuse a config that still uses the removed provider names (returns cfg unchanged otherwise)."""
    used_providers = _collect_used_providers(cfg)
    bad_names = sorted(used_providers & set(OLD_NAMES))
    if bad_names:
        raise ConfigError(f"config uses removed provider name(s) {', '.join(bad_names)}: {MIGRATION}")
    return cfg


def load_config(directory: Path | None = None) -> dict[str, Any]:
    """Load and merge all configuration YAML files from the specified directory."""
    directory = directory or config_dir()
    providers = _read_yaml(directory / "providers.yaml").get("providers", {})
    routing = _read_yaml(directory / "routing.yaml")
    profiles = _read_yaml(directory / "profiles.yaml").get("profiles", {})
    from .nodes import load_nodes
    if (directory / "peers.yaml").is_file() and not (directory / "nodes.yaml").is_file():
        raise ConfigError(f"{directory / 'peers.yaml'} is no longer read: rename it to nodes.yaml (key `peers:` -> `nodes:`)")
    return check_config({"providers": providers, "routing": routing, "profiles": profiles,
                         "nodes": load_nodes(directory), "dir": str(directory)})


def is_dry_run() -> bool:
    """Check if the dry-run mode is enabled via environment variable."""
    return os.environ.get("FOREMANKIT_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
