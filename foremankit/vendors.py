"""Identify model vendors independently of provider and runner names.

Keep reviewer selection pure so agent runners can share the same policy.
"""
from collections.abc import Sequence

UNKNOWN_VENDOR = "unknown"
VENDOR_PREFIX = "vendor:"
DEFAULT_MODEL_VENDORS = {
    "deepseek": "deepseek", "gemini": "google", "gpt": "openai", "o3": "openai",
    "codex": "openai", "claude": "anthropic", "qwen": "alibaba", "llama": "meta",
    "mistral": "mistral", "kimi": "moonshot", "glm": "zhipu",
}
RUNNER_VENDORS = {"agy": "google", "codex": "openai"}


def vendor_of_provider(name: str, providers: dict) -> str:
    """Return a configured vendor, defaulting to the provider's own name."""
    provider = providers.get(name, {})
    configuration = getattr(provider, "cfg", provider)
    return configuration.get("vendor", name)


def vendor_of_model(model_id: str, table: dict[str, str] | None = None) -> str:
    """Return the first matching model vendor, or unknown when no family matches."""
    model_id = model_id.lower()
    mappings = DEFAULT_MODEL_VENDORS if table is None else table
    return next((vendor for substring, vendor in mappings.items() if substring in model_id),
                UNKNOWN_VENDOR)


def runner_vendor(runner: str, model: str = "") -> str:
    """Identify fixed-vendor runners and the selected OpenCode model's vendor."""
    if runner == "opencode":
        return vendor_of_model(model)
    return RUNNER_VENDORS.get(runner, UNKNOWN_VENDOR)


def different_vendor(author_vendor: str, candidate_vendor: str) -> bool:
    """Require two known, distinct vendors before trusting an independent review."""
    # Missing identity cannot establish independence, so unknown fails closed.
    return bool(author_vendor and candidate_vendor
                and UNKNOWN_VENDOR not in (author_vendor, candidate_vendor)
                and author_vendor != candidate_vendor)


def pick_reviewer(author: str, candidates: Sequence[str], vendors: dict[str, str]) -> str | None:
    """Return the first independently vendored candidate, preserving preference order."""
    author_vendor = vendors.get(author, UNKNOWN_VENDOR)
    return next((candidate for candidate in candidates
                 if different_vendor(author_vendor, vendors.get(candidate, UNKNOWN_VENDOR))), None)
