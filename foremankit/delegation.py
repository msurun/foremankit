"""Category switch for delegation sources: paid, free, local, agy (Antigravity subscription).

Toggles are stored in the state file under state["toggles"] with keys
"delegation:paid", "delegation:free", "delegation:local", "delegation:agy". A missing key
defaults to "on", except agy (opt-in: "off" until the owner switches it on).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .state import StateStore

if TYPE_CHECKING:
    from .providers import Provider

CATEGORIES = ("paid", "free", "local", "agy")
DEFAULT_OFF = frozenset({"agy"})   # opt-in sources: "all on" leaves them alone

_VALUE_ON = "on"
_VALUE_OFF = "off"


def category_of(provider: Provider) -> str:
    """Classify a provider into one of the four delegation categories.

    Returns "agy" for the Antigravity CLI, "local" if the provider is an LLM,
    "free" if it is free, otherwise "paid".
    """
    if provider.is_agy:
        return "agy"
    if provider.is_llm:
        return "local"
    if provider.free:
        return "free"
    return "paid"


def is_switched_off(state: dict, category: str) -> bool:
    """Return True if the given category is explicitly switched off.

    A missing key means enabled, except for the opt-in categories (DEFAULT_OFF).
    """
    value = state.get("toggles", {}).get(f"delegation:{category}")
    if value is None:
        return category in DEFAULT_OFF
    return value == _VALUE_OFF


def all_switched_off(state: dict) -> bool:
    """True when every category is off: the user turned delegation off as a whole."""
    return all(is_switched_off(state, category) for category in CATEGORIES)


SWITCHED_OFF_MESSAGE = ("DELEGATION IS SWITCHED OFF by the user (ForemanKit panel or `foremankit delegation all off`). "
                        "Do this task yourself; do not retry delegation and do not look for another way around it.")


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
        targets_to_update = [name for name in CATEGORIES if value == _VALUE_OFF or name not in DEFAULT_OFF]
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


SUPERVISORS = ("haiku", "agy", "auto")   # who supervises bulk_code: the Claude Haiku agent, the Antigravity CLI
                                         # (`foremankit supervise`), or "auto": chosen per task (supervisor_choice.py)
_SUPERVISOR_KEY = "supervisor"


def supervisor_mode(state: dict) -> str:
    """The chosen supervisor: "haiku" (default), "agy" or "auto" (per task)."""
    value = state.get("toggles", {}).get(_SUPERVISOR_KEY)
    return value if value in SUPERVISORS else "haiku"


def set_supervisor(store: StateStore, value: str) -> str:
    """Choose the supervisor; raises ValueError for anything but "haiku", "agy" or "auto"."""
    if value not in SUPERVISORS:
        raise ValueError(f"Invalid supervisor: {value!r}. Must be one of {SUPERVISORS}.")

    def _apply_change(state: dict) -> str:
        state.setdefault("toggles", {})[_SUPERVISOR_KEY] = value
        return value

    return store.update(_apply_change)
