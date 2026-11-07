# tests/test_agent_choice.py
"""Table tests for agent_choice.decide_runner: the gates, the boundaries and the (choice, reasons) contract."""

from __future__ import annotations

import pytest

from foremankit.agent_choice import RunnerFacts, decide_runner

CHECK = "pytest tests/test_agent_choice.py -q"


def facts(**overrides: object) -> RunnerFacts:
    """Facts with a valid check and a short task, changed only where the row says so."""
    base = {"check": CHECK, "task_text": "a small task"}
    return RunnerFacts(**{**base, **overrides})        # type: ignore[arg-type]


CASES = [
    pytest.param(facts(files=6), (), "codex", "6 files", id="six-files-big-boundary"),
    pytest.param(facts(files=5), (), "opencode", "default", id="five-files-small-boundary"),
    pytest.param(facts(files=7, codex_enabled=False), (), "agy", "7 files", id="big-codex-off"),
    pytest.param(facts(files=7, codex_enabled=False, agy_enabled=False), (), "opencode", "7 files",
                 id="big-codex-and-agy-off"),
    pytest.param(facts(opencode_failures=2, codex_enabled=False), (), "agy", "opencode failed",
                 id="opencode-failed-codex-off"),
    pytest.param(facts(opencode_failures=2, codex_enabled=False, agy_enabled=False), (), "lead", "exhausted",
                 id="opencode-failed-no-forward-runner"),
    pytest.param(facts(codex_failures=1), (), "opencode", "default", id="one-codex-failure"),
    pytest.param(facts(codex_failures=2), (), "agy", "codex failed", id="codex-failed-escalates"),
    pytest.param(facts(codex_failures=2, agy_enabled=False), (), "lead", "escalation is exhausted",
                 id="codex-failed-agy-off-no-loop"),
    pytest.param(facts(codex_failures=2, opencode_failures=2, files=7), (), "agy", "codex failed",
                 id="codex-failure-before-opencode-failure-and-size"),
    pytest.param(facts(codex_failures=3, agy_enabled=False, codex_enabled=False), (), "lead", "exhausted",
                 id="codex-failure-above-threshold"),
    pytest.param(facts(opencode_failures=3, files=7), (), "codex", "opencode failed",
                 id="opencode-failure-before-size"),
    pytest.param(facts(opencode_enabled=False, codex_enabled=False), (), "agy", "switched off",
                 id="only-agy-enabled"),
    pytest.param(facts(opencode_enabled=False, agy_enabled=False), (), "codex", "switched off",
                 id="only-codex-enabled"),
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
    pytest.param(facts(scope=("src/other.py",)), ["gateway/**"], "opencode", "opencode", id="scope-without-a-review-main"),
    pytest.param(facts(contract_cited=False, edge_cases_listed=False), (), "revise", "not ready",
                 id="task-file-not-ready"),
    pytest.param(facts(contract_cited=False, edge_cases_listed=True), (), "opencode", "opencode",
                 id="contract-alone-is-not-enough-to-revise"),
    pytest.param(facts(files=9), (), "split", "files", id="nine-files-split"),
    pytest.param(facts(files=8), (), "codex", "files", id="eight-files-not-split"),
    pytest.param(facts(changed_lines=401), (), "split", "lines", id="401-lines-split"),
    pytest.param(facts(changed_lines=400), (), "opencode", "opencode", id="400-lines-not-split"),
    pytest.param(facts(task_text="x" * 22001), (), "split", "text", id="22001-chars-split"),
    pytest.param(facts(task_text="x" * 22000, agy_enabled=False), (), "codex", "text",
                 id="22000-chars-not-split"),
    pytest.param(facts(edits_existing_only=True, files=2, changed_lines=40), (), "haiku", "small edit",
                 id="small-edit-to-existing-code"),
    pytest.param(facts(edits_existing_only=True, files=2, changed_lines=41), (), "opencode", "opencode",
                 id="41-lines-are-no-haiku"),
    pytest.param(facts(sensitivity="internal", scope=("foremankit/**",)), (), "agy", "internal",
                 id="internal-needs-agy"),
    pytest.param(facts(sensitivity="internal", opencode_enabled=False, agy_enabled=False), (), "queue", "wait",
                 id="internal-without-agy-queues"),
    pytest.param(facts(sensitivity="internal", files=7), (), "agy", "internal", id="internal-seven-files-agy"),
    pytest.param(facts(sensitivity="internal", opencode_enabled=True), (), "agy", "agy",
                 id="internal-never-opencode"),
    pytest.param(facts(opencode_failures=2), (), "codex", "twice", id="two-opencode-failures-escalate"),
    pytest.param(facts(opencode_failures=1), (), "opencode", "opencode", id="one-failure-stays-opencode"),
    pytest.param(facts(opencode_failures=2, agy_enabled=False), (), "codex", "twice",
                 id="escalation-does-not-need-agy"),
    pytest.param(facts(context_lines=1500), (), "codex", "context", id="1500-context-lines-codex"),
    pytest.param(facts(context_lines=1499), (), "opencode", "opencode", id="1499-context-lines-opencode"),
    pytest.param(facts(task_text="y" * 8001), (), "codex", "long", id="8001-chars-codex"),
    pytest.param(facts(task_text="y" * 8000), (), "opencode", "opencode", id="8000-chars-opencode"),
    pytest.param(facts(opencode_enabled=False), (), "codex", "switched off", id="opencode-off-falls-to-codex"),
    pytest.param(facts(opencode_enabled=False, codex_enabled=False, agy_enabled=False), (), "lead", "no runner is enabled",
                 id="all-runners-off"),
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
    assert picked == "opencode"
    assert isinstance(reasons, list) and reasons
    assert all(isinstance(reason, str) and reason for reason in reasons)


def test_the_scope_default_is_empty_and_a_review_glob_is_optional() -> None:
    """A plain task with no scope never trips the review gate."""
    picked, _ = decide_runner(facts())
    assert picked == "opencode"


@pytest.mark.parametrize("codex_enabled", [False, True])
def test_internal_routing_ignores_codex(codex_enabled: bool) -> None:
    """Internal data uses agy or waits, regardless of Codex availability."""
    assert decide_runner(facts(codex_enabled=codex_enabled, sensitivity="internal"))[0] == "agy"
    picked, reasons = decide_runner(facts(
        codex_enabled=codex_enabled, sensitivity="internal", agy_enabled=False,
    ))
    assert picked == "queue"
    assert "wait" in " ".join(reasons)
