"""Category switch for delegation sources: paid, free, local.

Toggles are stored in the state file under state["toggles"] with keys
"delegation:paid", "delegation:free", "delegation:local". A missing key
defaults to "on".
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .state import StateStore

if TYPE_CHECKING:
    from .providers import Provider

CATEGORIES = ("paid", "free", "local")

_VALUE_ON = "on"
_VALUE_OFF = "off"


def category_of(provider: Provider) -> str:
    """Classify a provider into one of the three delegation categories.

    Returns "local" if the provider is an LLM, "free" if it is free,
    otherwise "paid".
    """
    if provider.is_llm:
        return "local"
    if provider.free:
        return "free"
    return "paid"


def is_switched_off(state: dict, category: str) -> bool:
    """Return True if the given category is explicitly switched off.

    A missing key or a value of "on" means the category is enabled.
    Only "off" means disabled.
    """
    key = f"delegation:{category}"
    value = state.get("toggles", {}).get(key)
    return value == _VALUE_OFF


def set_switch(store: StateStore, target: str, value: str) -> list[str]:
    """Turn on or off the specified category or all categories.

    Args:
        store: The state store to update.
        target: One of CATEGORIES or "all".
        value: "on" or "off".

    Returns:
        A list of category names that were changed.

    Raises:
        ValueError: If target is not valid or value is not "on"/"off".
    """
    if target not in CATEGORIES and target != "all":
        raise ValueError(f"Invalid target: {target!r}. Must be one of {CATEGORIES} or 'all'.")
    if value not in (_VALUE_ON, _VALUE_OFF):
        raise ValueError(f"Invalid value: {value!r}. Must be 'on' or 'off'.")

    if target == "all":
        targets_to_update = list(CATEGORIES)
    else:
        targets_to_update = [target]

    def _apply_change(state: dict) -> list[str]:
        toggles = state.setdefault("toggles", {})
        changed = []
        for cat in targets_to_update:
            key = f"delegation:{cat}"
            toggles[key] = value
            changed.append(cat)
        return changed

    return store.update(_apply_change)


def describe(state: dict) -> str:
    """Describe the current state of all delegation categories.

    Returns a string with one line per category, e.g. "paid: on".
    """
    lines = []
    for cat in CATEGORIES:
        status = _VALUE_OFF if is_switched_off(state, cat) else _VALUE_ON
        lines.append(f"{cat}: {status}")
    return "\n".join(lines)
