"""How hard a model should think about a task (effort level) and advice about big answers. Pure functions."""
from __future__ import annotations

from typing import Any

LEVELS = ("minimal", "low", "medium", "high")


def reasoning_level(routing: dict[str, Any], kind: str, sensitivity: str, escalate: bool) -> str | None:
    """Effort level for this task: per-kind default, raised by the internal floor / escalation."""
    settings = routing.get("reasoning") or {}
    candidates = [(settings.get("kinds") or {}).get(kind)]
    if sensitivity == "internal":
        candidates.append(settings.get("internal_floor"))
    if escalate:
        candidates.append(settings.get("escalate_level"))
    valid = [level for level in candidates if level in LEVELS]
    return max(valid, key=LEVELS.index) if valid else None


def cap_for_large_output(routing: dict[str, Any], level: str | None, max_out: int) -> str | None:
    """A big answer plus deep reasoning is how a call ends up empty or cut off (measured 2026-10-10: ~24k reasoning
    tokens, no answer). Above `reasoning.large_output_tokens` the level is lowered to `large_output_level`."""
    settings = routing.get("reasoning") or {}
    limit, cap = settings.get("large_output_tokens"), settings.get("large_output_level")
    if level in LEVELS and cap in LEVELS and limit and max_out > int(limit):
        return min(level, cap, key=LEVELS.index)
    return level


def size_notes(routing: dict[str, Any], max_out: int) -> list[str]:
    """Advice shown with the result when the requested answer is big."""
    limit = int(((routing.get("calls") or {}).get("warn_output_tokens")) or 0)
    if limit and max_out > limit:
        return [(f"requested output of {max_out} tokens is big: cheap models cut off or lose track; "
                 f"prefer tasks with an answer under {limit} tokens (one file or one part at a time)")]
    return []
