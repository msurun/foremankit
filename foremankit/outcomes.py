"""What became of a delegated task: did the check pass, did the review find blockers, what did it cost in total.

`calls.jsonl` already logs every model call; this adds ONE `outcome` event per `delegate`, so `foremankit stats` can
report the cost per ACCEPTED result instead of counting every delegated token as saved Claude tokens."""
from __future__ import annotations

from typing import Any


def build_outcome(result: dict[str, Any], kind: str) -> dict[str, Any]:
    """The `outcome` log record for a finished delegation (`result` as built by Router.delegate, private keys allowed).

    `accepted` is an approximation: the check passed, or there was no check and no unfixed blocker. It says nothing
    about whether the caller later rewrote the file."""
    verify = result.get("verify")
    review = result.get("review")
    blockers = _blockers(review)
    return {"event": "outcome", "id": result.get("id"), "provider": result.get("provider"),
            "node": result.get("node"), "model": result.get("model"), "kind": kind,
            "cost_usd": round(result.get("cost_usd", 0.0), 6),
            "verified": verify is not None, "verify_passed": bool(verify and verify["passed"]),
            "fix_rounds": verify["fix_rounds"] if verify else 0,
            "reviewed": review is not None, "blockers": blockers,
            "review_fixed": bool(review and review.get("fixed")),
            "accepted": _accepted(verify, review, blockers)}


def _blockers(review: dict[str, Any] | None) -> int:
    """How many blocker findings the review reported (0 when there was no review)."""
    if not review:
        return 0
    return sum(1 for finding in review.get("findings", []) if finding.get("severity") == "blocker")


def _accepted(verify: dict[str, Any] | None, review: dict[str, Any] | None, blockers: int) -> bool:
    """Passed the check (when there was one) and left no blocker that the author did not fix."""
    if verify is not None and not verify["passed"]:
        return False
    return not (review and blockers and not review.get("fixed"))


def summarize_outcomes(rows: list[dict[str, Any]], cutoff: float = 0.0) -> dict[str, dict[str, float]]:
    """Per `provider/model`: delegations, accepted ones, total cost and cost per accepted result (None when none)."""
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("event") != "outcome" or row.get("ts", 0) < cutoff:
            continue
        entry = table.setdefault(f"{row.get('provider')}/{row.get('model')}",
                                 {"delegations": 0, "accepted": 0, "cost_usd": 0.0})
        entry["delegations"] += 1
        entry["accepted"] += 1 if row.get("accepted") else 0
        entry["cost_usd"] += row.get("cost_usd", 0.0)
    for entry in table.values():
        entry["cost_per_accepted"] = entry["cost_usd"] / entry["accepted"] if entry["accepted"] else None
    return table
