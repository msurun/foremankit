"""Outcome events: whether a delegated result was accepted (check passed, no unfixed blocker) and what accepted results cost."""
from __future__ import annotations

import sys

import pytest
from conftest import chat_json

from foremankit.outcomes import build_outcome, summarize_outcomes

GROQ = "https://api.groq.com"
PYTHON = sys.executable.replace("\\", "/")
ANSWER = '```python\n"""A check file."""\nVALUE = 1\n```'
BLOCKER = {"severity": "blocker", "where": "add", "fix": "guard against None"}
MINOR = {"severity": "minor", "where": "add", "fix": "add hints"}


def result(**overrides) -> dict:
    """A finished delegation as Router.delegate returns it, without any check or review."""
    base = {"id": "abc", "provider": "groq", "node": "cloud", "model": "llama", "cost_usd": 0.0123456789}
    return {**base, **overrides}


def test_a_delegation_without_check_or_review_is_accepted():
    outcome = build_outcome(result(), "quick")
    assert outcome["accepted"] is True and outcome["verified"] is False and outcome["reviewed"] is False


def test_a_passed_check_after_two_fix_rounds_is_accepted():
    outcome = build_outcome(result(verify={"passed": True, "fix_rounds": 2}), "bulk_code")
    assert (outcome["verify_passed"], outcome["fix_rounds"], outcome["accepted"]) == (True, 2, True)


def test_a_failed_check_is_not_accepted():
    outcome = build_outcome(result(verify={"passed": False, "fix_rounds": 2}), "bulk_code")
    assert outcome["verify_passed"] is False and outcome["accepted"] is False


def test_an_unfixed_blocker_makes_the_result_not_accepted():
    review = {"findings": [BLOCKER], "fixed": False}
    outcome = build_outcome(result(review=review), "hard_code")
    assert outcome["blockers"] == 1 and outcome["accepted"] is False


def test_a_blocker_the_author_fixed_is_accepted():
    review = {"findings": [BLOCKER, MINOR], "fixed": True}
    outcome = build_outcome(result(review=review), "hard_code")
    assert outcome["blockers"] == 1 and outcome["review_fixed"] is True and outcome["accepted"] is True


def test_the_cost_is_rounded_to_six_decimals():
    assert build_outcome(result(), "quick")["cost_usd"] == 0.012346


def outcome_row(provider: str, accepted: bool, cost: float, ts: float = 100) -> dict:
    """A logged outcome event for one delegation."""
    return {"event": "outcome", "provider": provider, "model": "m1", "accepted": accepted, "cost_usd": cost, "ts": ts}


def test_the_summary_groups_by_provider_and_model_and_sums_cost():
    rows = [outcome_row("gemini", True, 0.2), outcome_row("gemini", False, 0.1), outcome_row("groq", True, 0.0)]
    summary = summarize_outcomes(rows)
    assert set(summary) == {"gemini/m1", "groq/m1"}
    assert summary["gemini/m1"]["delegations"] == 2 and summary["gemini/m1"]["accepted"] == 1
    assert summary["gemini/m1"]["cost_usd"] == pytest.approx(0.3)
    assert summary["gemini/m1"]["cost_per_accepted"] == pytest.approx(0.3)


def test_the_cost_per_accepted_result_is_none_when_nothing_was_accepted():
    summary = summarize_outcomes([outcome_row("gemini", False, 0.5)])
    assert summary["gemini/m1"]["accepted"] == 0 and summary["gemini/m1"]["cost_per_accepted"] is None


def test_the_summary_ignores_other_events_and_rows_older_than_the_cutoff():
    rows = [outcome_row("gemini", True, 0.2, ts=100),
            outcome_row("gemini", True, 5.0, ts=1),
            {"event": "call", "provider": "gemini", "model": "m1", "cost_usd": 9.0, "ts": 100}]
    summary = summarize_outcomes(rows, cutoff=50)
    assert summary["gemini/m1"]["delegations"] == 1 and summary["gemini/m1"]["cost_usd"] == pytest.approx(0.2)


@pytest.fixture
def project(tmp_path, cfg, monkeypatch):
    """Work inside tmp_path with the current Python allowed as the check, which just runs the file."""
    monkeypatch.chdir(tmp_path)
    cfg["routing"]["verify"]["allowed_commands"] = [PYTHON]
    cfg["routing"]["verify"]["conventions"] = False
    (tmp_path / "check.py").write_text("print('original')\n", encoding="utf-8")
    return tmp_path


def test_a_delegation_with_a_check_writes_one_outcome_event_to_the_log(project, monkeypatch, make_router, mock, cfg):
    monkeypatch.setenv("GROQ_API_KEY", "q")
    mock.on(GROQ, "/chat/completions", chat_json(ANSWER))
    router = make_router(cfg)
    router.delegate("write check.py", kind="quick", verify=f"{PYTHON} check.py", verify_target="check.py")
    [outcome] = [row for row in router.log.read() if row.get("event") == "outcome"]
    assert (outcome["verified"], outcome["verify_passed"], outcome["accepted"]) == (True, True, True)
