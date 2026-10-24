"""Pass@k figures per (kind, provider, model), built from the `outcome` log events.

A delegation without a check (`verified == False`) says nothing about correctness, so only verified outcomes
are counted. `fix_rounds` is the number of correction attempts the author made before the check passed, so
pass@1 means the first answer passed and pass@3 means it passed within three attempts."""
from __future__ import annotations

from typing import Any

DECIMALS = 3


def summarize_pass_rates(rows: list[dict[str, Any]], cutoff: float = 0.0) -> dict[str, dict]:
    """Per `kind/provider/model`: how many verified delegations, pass@1 and pass@3 as shares rounded to 3 decimals.

    pass@1 = accepted with `fix_rounds == 0`; pass@3 = accepted with `fix_rounds <= 2`. Rows older than `cutoff`
    (unix seconds), of another event type, or unverified are ignored, because they carry no pass/fail signal.
    """
    table: dict[str, dict] = {}
    for row in rows:
        if not _is_counted(row, cutoff):
            continue
        entry = table.setdefault(_key(row), {"delegations": 0, "pass_at_1": 0, "pass_at_3": 0})
        entry["delegations"] += 1
        entry["pass_at_1"] += 1 if _passed_within(row, 0) else 0
        entry["pass_at_3"] += 1 if _passed_within(row, 2) else 0
    for entry in table.values():
        entry["pass_at_1"] = round(entry["pass_at_1"] / entry["delegations"], DECIMALS)
        entry["pass_at_3"] = round(entry["pass_at_3"] / entry["delegations"], DECIMALS)
    return table


def format_pass_rates(table: dict[str, dict], min_delegations: int = 1) -> str:
    """One line per key, sorted by pass@1 descending then key, as `key: pass@1 80%, pass@3 90% (5 checked)`.

    Keys with fewer than `min_delegations` checked delegations are left out. Falls back to a notice when no key
    qualifies, so callers can print the result unconditionally."""
    shown = [(key, entry) for key, entry in table.items() if entry["delegations"] >= min_delegations]
    if not shown:
        return "no verified delegations yet"
    shown.sort(key=lambda pair: (-pair[1]["pass_at_1"], pair[0]))
    return "\n".join(_format_line(key, entry) for key, entry in shown)


def _is_counted(row: dict[str, Any], cutoff: float) -> bool:
    """An outcome event, not older than `cutoff`, whose check actually ran."""
    return row.get("event") == "outcome" and row.get("ts", 0) >= cutoff and row.get("verified") is True


def _passed_within(row: dict[str, Any], max_fix_rounds: int) -> bool:
    """The delegation was accepted after at most `max_fix_rounds` correction attempts."""
    return bool(row.get("accepted")) and row.get("fix_rounds", 0) <= max_fix_rounds


def _key(row: dict[str, Any]) -> str:
    """Grouping key `kind/provider/model` for one outcome row."""
    return f"{row.get('kind')}/{row.get('provider')}/{row.get('model')}"


def _format_line(key: str, entry: dict) -> str:
    """The human-readable line for one pass-rate entry."""
    return (f"{key}: pass@1 {entry['pass_at_1']:.0%}, pass@3 {entry['pass_at_3']:.0%} "
            f"({entry['delegations']} checked)")
