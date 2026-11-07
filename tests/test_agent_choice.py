"""Test configurable runner priority and the gates that precede it.
The cases preserve privacy and readiness boundaries while checking fallback.
"""

from __future__ import annotations

import pytest

from foremankit.agent_choice import RunnerFacts, decide_runner

CHECK = "pytest tests/test_agent_choice.py -q"


def facts(**overrides: object) -> RunnerFacts:
    """Facts with a valid check and a short task, changed only where the row says so."""
    base = {"check": CHECK, "task_text": "a small task"}
    return RunnerFacts(**{**base, **overrides})        # type: ignore[arg-type]


CASES = [
    pytest.param(facts(), (), "codex", "first", id="small-normal-codex-first"),
    pytest.param(facts(codex_enabled=False), (), "opencode", "switched off", id="codex-off"),
    pytest.param(facts(codex_enabled=False, opencode_enabled=False), (), "agy", "switched off",
                 id="only-agy"),
    pytest.param(facts(codex_failures=1), (), "codex", "first", id="one-failure-retries"),
    pytest.param(facts(codex_failures=2), (), "opencode", "codex failed", id="codex-failed"),
    pytest.param(facts(codex_failures=2, opencode_failures=2), (), "agy", "opencode failed",
                 id="both-failed"),
    pytest.param(facts(codex_failures=2, opencode_failures=2, agy_failures=2),
                 (), "lead", "exhausted", id="all-failed"),
    pytest.param(facts(codex_enabled=False, opencode_enabled=False, agy_enabled=False),
                 (), "lead", "no runner is enabled", id="all-off"),
    pytest.param(facts(files=7), (), "codex", "first", id="large-also-codex"),
    pytest.param(facts(context_lines=1500), (), "codex", "first", id="context-also-codex"),
    pytest.param(facts(sensitivity="private", agy_enabled=True, opencode_enabled=True,
                       edits_existing_only=True, files=1, changed_lines=10),
                 (), "lead", "private", id="private-never-reaches-a-runner"),
    pytest.param(facts(sensitivity="private", files=20, changed_lines=900, agy_enabled=True),
                 (), "lead", "private", id="private-gate-before-the-size-rule"),
    pytest.param(facts(check="   "), (), "lead", "check", id="blank-check"),
    pytest.param(facts(needs_session_context=True), (), "lead", "context", id="session-context"),
    pytest.param(facts(touches_shared_interface=True), (), "lead", "review", id="shared-interface"),
    pytest.param(facts(scope=("gateway/**",)), ["gateway/**"], "lead", "review", id="review-glob-exact"),
    pytest.param(facts(scope=("src/auth_*.py",)), ["**/auth*"], "lead", "review", id="review-glob-star"),
    pytest.param(facts(scope=("src/other.py",)), ["gateway/**"], "codex", "codex", id="scope-without-a-review-main"),
    pytest.param(facts(contract_cited=False, edge_cases_listed=False), (), "revise", "not ready",
                 id="task-file-not-ready"),
    pytest.param(facts(contract_cited=False, edge_cases_listed=True), (), "codex", "codex",
                 id="contract-alone-is-not-enough-to-revise"),
    pytest.param(facts(files=9), (), "split", "files", id="nine-files-split"),
    pytest.param(facts(files=8), (), "codex", "first", id="eight-files-not-split"),
    pytest.param(facts(changed_lines=401), (), "split", "lines", id="401-lines-split"),
    pytest.param(facts(changed_lines=400), (), "codex", "codex", id="400-lines-not-split"),
    pytest.param(facts(task_text="x" * 22001), (), "split", "text", id="22001-chars-split"),
    pytest.param(facts(task_text="x" * 22000, agy_enabled=False), (), "codex", "first",
                 id="22000-chars-not-split"),
    pytest.param(facts(edits_existing_only=True, files=2, changed_lines=40), (), "haiku", "small edit",
                 id="small-edit-to-existing-code"),
    pytest.param(facts(edits_existing_only=True, files=2, changed_lines=41), (), "codex", "codex",
                 id="41-lines-are-no-haiku"),
    pytest.param(facts(sensitivity="secret", agy_enabled=True, opencode_enabled=True), (), "lead", "sensitivity",
                 id="unknown-sensitivity-fails-closed"),
]

@pytest.mark.parametrize(("case_facts", "review_globs", "choice", "keyword"), CASES)
def test_the_first_matching_rule_decides(case_facts: RunnerFacts, review_globs: list[str],
                                         choice: str, keyword: str) -> None:
    """The choice is the expected one and at least one reason mentions the keyword."""
    picked, reasons = decide_runner(case_facts, review_globs)
    assert picked == choice
    assert keyword in " ".join(reasons).lower()


def test_the_result_is_a_choice_and_a_list_of_short_reasons() -> None:
    """The shape callers rely on: a string choice and a non-empty list of strings."""
    picked, reasons = decide_runner(facts())
    assert picked == "codex"
    assert isinstance(reasons, list) and reasons
    assert all(isinstance(reason, str) and reason for reason in reasons)


def test_the_scope_default_is_empty_and_a_review_glob_is_optional() -> None:
    """A plain task with no scope never trips the review gate."""
    picked, _ = decide_runner(facts())
    assert picked == "codex"


@pytest.mark.parametrize("overrides, expected, keyword", [
    ({}, "codex", "first"),
    ({"codex_enabled": False}, "agy", "switched off"),
    ({"codex_enabled": False, "agy_enabled": False}, "queue", "waits"),
    ({"codex_failures": 2}, "agy", "failed"),
    ({"codex_failures": 2, "agy_failures": 2}, "lead", "exhausted"),
])
def test_internal_routing_order_and_escalation(overrides, expected, keyword):
    """Internal routing uses Codex then agy without allowing opencode."""
    picked, reasons = decide_runner(facts(sensitivity="internal", **overrides))
    assert picked == expected
    assert keyword in reasons[0]


@pytest.mark.parametrize("label, runners, expected", [
    ("normal", ["opencode", "codex"], "opencode"),
    ("internal", ["opencode", "agy", "codex"], "agy"),
    ("internal", ["opencode"], "queue"),
])
def test_custom_order_respects_the_data_boundary(label, runners, expected):
    """The setting changes priority but cannot authorize internal opencode use."""
    picked, _ = decide_runner(facts(sensitivity=label), order={label: runners})
    assert picked == expected
