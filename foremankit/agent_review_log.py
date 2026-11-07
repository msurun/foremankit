"""Review verdicts for agent-run branches, grouped per (provider, model).

`accepted` on an `outcome` event only means the check passed; the lead's merge or reject decision is a
separate `agent_run_review` event, so `clean_rate` counts the branches merged without any fix."""

from __future__ import annotations

from typing import Any

from .state import CallLog

VERDICTS = ("merged", "merged_after_fix", "rejected")
MODEL_REVIEW_EVENT = "agent_run_model_review"
MODEL_REVIEW_SEVERITIES = ("high", "medium", "low")


# codecheck: allow too-many-parameters (the log protocol fixes them)
def log_review(log: CallLog, run_id: str, runner: str, model: str, verdict: str, blockers: int = 0) -> dict:
    """Validate one lead verdict and append it to the call log as an `agent_run_review` event."""
    if verdict not in VERDICTS:
        raise ValueError(f"unknown review verdict: {verdict!r}")
    if blockers < 0:
        raise ValueError(f"blockers must be >= 0, got {blockers}")
    event = {"event": "agent_run_review", "id": run_id, "provider": runner,
             "model": model, "verdict": verdict, "blockers": blockers}
    log.write(event)
    return event


def summarize_reviews(rows: list[dict[str, Any]]) -> dict[str, dict]:
    """Per provider/model: review counts, blockers and the share merged with no fix (last review of an id wins)."""
    latest: dict[Any, dict[str, Any]] = {}
    for row in rows:
        if row.get("event") == "agent_run_review" and row.get("verdict") in VERDICTS:
            latest[row.get("id")] = row
    table: dict[str, dict] = {}
    for row in latest.values():
        entry = table.setdefault(_key(row), {"reviews": 0, "merged": 0, "merged_after_fix": 0,
                                             "rejected": 0, "blockers": 0, "clean_rate": 0.0})
        entry["reviews"] += 1
        entry[row["verdict"]] += 1
        entry["blockers"] += row.get("blockers", 0)
    for entry in table.values():
        entry["clean_rate"] = entry["merged"] / entry["reviews"]
    return table


def format_reviews(table: dict[str, dict]) -> str:
    """One sorted line per provider/model, or a notice when there are no reviews yet."""
    if not table:
        return "no agent-run reviews yet"
    return "\n".join(_format_line(key, table[key]) for key in sorted(table))


def _key(row: dict[str, Any]) -> str:
    """Grouping key `provider/model`, with `?` standing in for an unknown model."""
    return f"{row.get('provider')}/{row.get('model') or '?'}"


def _format_line(key: str, entry: dict) -> str:
    """The human-readable line for one review entry."""
    return (f"{key}: {entry['reviews']} reviews, clean {entry['clean_rate']:.0%}, "
            f"fixed {entry['merged_after_fix']}, rejected {entry['rejected']}, blockers {entry['blockers']}")



def summarize_model_reviews(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every model review attempt, newest first, without findings or raw output."""
    attempts = [(position, row) for position, row in enumerate(rows)
                if row.get("event") == MODEL_REVIEW_EVENT]
    # Timestamps are seconds since the epoch; later log positions break timestamp ties.
    attempts.sort(key=lambda item: (float(item[1].get("ts") or 0.0), item[0]), reverse=True)
    return [_model_review_summary(row) for _, row in attempts]


def _model_review_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Keep an attempt's identity and status alongside complete severity counters."""
    counts = row.get("counts_by_severity") or {}
    return {"branch": row.get("branch", ""),
            "reviewer": row.get("reviewer", ""),
            "status": row.get("status", ""),
            "counts_by_severity": {severity: counts.get(severity, 0)
                                   for severity in MODEL_REVIEW_SEVERITIES}}
